#include "flagos-spacemit-ops.h"

#include "flagos-spacemit-attn.h"
#include "flagos-spacemit-ime.h"
#include "flagos-spacemit-kernels.h"
#include "flagos-spacemit-weights.h"

#include "../../../ggml-impl.h"

#include <cstring>

// an operand is usable when it is not allocated yet (placement time), or lives in our buffer or in host memory
static bool spacemit_buffer_usable(const ggml_tensor * t, ggml_backend_buffer_type_t own_buft) {
    const ggml_backend_buffer_t buffer = t->view_src != nullptr ? t->view_src->buffer : t->buffer;
    if (buffer == nullptr) {
        return true;
    }
    const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buffer);
    return buft == own_buft || ggml_backend_buft_is_host(buft);
}

// F32 with contiguous elements in each row; rows may have any stride
static bool spacemit_f32_rows(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float);
}

// F32 or F16 with contiguous elements in each row; rows may have any stride
static bool spacemit_f32_f16_rows(const ggml_tensor * t) {
    return (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) && t->nb[0] == ggml_type_size(t->type);
}

// ADD, MUL: src1 repeats over src0 in every dimension, as in ggml-cpu's binary-ops.cpp
static bool spacemit_binary_f32_supported(const ggml_tensor * op) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    return spacemit_f32_rows(op) && spacemit_f32_rows(a) && b->type == GGML_TYPE_F32 && ggml_can_repeat(b, a) &&
           ggml_are_same_shape(a, op);
}

// RMS_NORM: any eps >= 0 (ggml-cpu asserts the same)
static bool spacemit_rms_norm_f32_supported(const ggml_tensor * op) {
    float eps;
    std::memcpy(&eps, op->op_params, sizeof(float));
    return spacemit_f32_rows(op) && spacemit_f32_rows(op->src[0]) && ggml_are_same_shape(op->src[0], op) && eps >= 0.0f;
}

// ROPE (forward): F32 or F16 rows into rows of the same type (F16: the context shift's ROPE in place on the KV cache,
// src/llama-kv-cache.cpp:1878-1883); modes NORMAL and NEOX; I32 positions, one per token; optional F32 frequency
// factors; the channels past n_dims are copied in pairs, so ne0 is even
static bool spacemit_rope_supported(const ggml_tensor * op) {
    const ggml_tensor * src0   = op->src[0];
    const ggml_tensor * pos    = op->src[1];
    const ggml_tensor * ff     = op->src[2];
    const int           n_dims = ggml_get_op_params_i32(op, 1);
    const int           mode   = ggml_get_op_params_i32(op, 2);
    return spacemit_f32_f16_rows(op) && spacemit_f32_f16_rows(src0) && op->type == src0->type &&
           ggml_are_same_shape(src0, op) &&
           (mode == GGML_ROPE_TYPE_NORMAL || mode == GGML_ROPE_TYPE_NEOX) && n_dims > 0 && n_dims % 2 == 0 &&
           n_dims <= src0->ne[0] && src0->ne[0] % 2 == 0 && pos->type == GGML_TYPE_I32 && ggml_is_contiguous(pos) &&
           pos->ne[0] >= src0->ne[2] &&
           (ff == nullptr || (ff->type == GGML_TYPE_F32 && ggml_is_contiguous(ff) && ff->ne[0] >= n_dims / 2));
}

// the KV cache's types besides F32 and F16 (-ctk, -ctv) that the provider writes from F32 rows, with ggml's reference
// quantizers (from_float_ref): whole blocks, ggml's layout
static bool spacemit_cache_block_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

// SET_ROWS (the KV cache writes, src/llama-kv-cache.cpp:1333, :1368): F32 or F16 rows into F32 or F16 rows, or F32 rows
// into rows of a block type above in whole blocks (an F16 source there stays unclaimed); I64 or I32 indices; never
// into a tensor in an IME layout (op is a view of the destination; the layout rule resolves its view_src)
static bool spacemit_set_rows_supported(const ggml_tensor * op) {
    const ggml_tensor * src = op->src[0];
    const ggml_tensor * idx = op->src[1];
    if (op->ne[0] != src->ne[0] || (idx->type != GGML_TYPE_I64 && idx->type != GGML_TYPE_I32) ||
        spacemit_tensor_is_repacked(op)) {
        return false;
    }
    if (spacemit_f32_f16_rows(op)) {
        return spacemit_f32_f16_rows(src);
    }
    return spacemit_cache_block_type(op->type) && spacemit_f32_rows(src) && op->nb[0] == ggml_type_size(op->type) &&
           op->ne[0] % ggml_blck_size(op->type) == 0;
}

// CPY: F32 rows into a contiguous tensor of a block type above, of the same shape, in whole blocks (the context shift's
// write-back into a quantized KV cache, src/llama-kv-cache.cpp:1863-1877); op is ggml_cpy's view of the destination,
// with its ne and nb (ggml.c:3526-3535), never in an IME layout. No other CPY: no same-type copies, no other source
// types, no non-contiguous destinations
static bool spacemit_cpy_quant_supported(const ggml_tensor * op) {
    const ggml_tensor * src = op->src[0];
    return spacemit_cache_block_type(op->type) && spacemit_f32_rows(src) && ggml_are_same_shape(src, op) &&
           ggml_is_contiguous(op) && op->ne[0] % ggml_blck_size(op->type) == 0 && !spacemit_tensor_is_repacked(op);
}

// GET_ROWS: F32 rows by I32 indices
static bool spacemit_get_rows_f32_supported(const ggml_tensor * op) {
    return spacemit_f32_rows(op) && spacemit_f32_rows(op->src[0]) && op->src[1]->type == GGML_TYPE_I32 &&
           op->ne[0] == op->src[0]->ne[0];
}

// GLU: SwiGLU only, split (two tensors) and single-tensor forms; rows addressed as in ggml-cpu (ggml_is_contiguous_1),
// dst shaped as ggml-cpu asserts (the kernels write nrows x nc elements of it)
static bool spacemit_swiglu_f32_supported(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const bool          dst_shape =
        src1 != nullptr ? ggml_are_same_shape(src0, op) :
                          op->ne[0] == src0->ne[0] / 2 && ggml_nrows(op) == ggml_nrows(src0);
    return ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU && op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
           ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(op) && dst_shape &&
           (src1 == nullptr ||
            (src1->type == GGML_TYPE_F32 && ggml_is_contiguous_1(src1) && ggml_are_same_shape(src0, src1)));
}

// a weight in an IME layout (Q4_0 32x256, Q4_1 32x32, Q8_0 and Q6_K in q8_0 32x32), read in that layout: so it must
// sit in our buffer (or be unallocated at placement time), never in a host buffer. The layout comes from the weight's
// type, shape and name (spacemit_weight_layout): llama.cpp's cache tensors never have one, so a MUL_MAT never reads a
// quantized KV cache as a weight. The answer must not depend on the row count: llama.cpp asks once per weight, at load,
// with 512 rows (plan.md M2b design).
static bool spacemit_mul_mat_ime_supported(const ggml_tensor * op, ggml_backend_buffer_type_t own_buft) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (spacemit_weight_layout(w) == spacemit_layout::plain) {
        return false;
    }
    if (w->buffer != nullptr && ggml_backend_buffer_get_type(w->buffer) != own_buft) {
        return false;
    }
    return x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_is_contiguous(op);
}

#define SPACEMIT_ONE_STEP_OP(name, kernel, workspace)                  \
    static const spacemit_kernel_fn name##_steps[] = { kernel };       \
    static const spacemit_op        name           = { name##_steps, 1, workspace }

SPACEMIT_ONE_STEP_OP(k_add_f32, spacemit_kernel_add_f32, nullptr);
SPACEMIT_ONE_STEP_OP(k_mul_f32, spacemit_kernel_mul_f32, nullptr);
SPACEMIT_ONE_STEP_OP(k_rms_norm_f32, spacemit_kernel_rms_norm_f32, nullptr);
SPACEMIT_ONE_STEP_OP(k_rope, spacemit_kernel_rope, spacemit_rope_workspace);
SPACEMIT_ONE_STEP_OP(k_set_rows, spacemit_kernel_set_rows, nullptr);
SPACEMIT_ONE_STEP_OP(k_cpy_quant, spacemit_kernel_cpy_quant, nullptr);
SPACEMIT_ONE_STEP_OP(k_get_rows_f32, spacemit_kernel_get_rows_f32, nullptr);
SPACEMIT_ONE_STEP_OP(k_swiglu_f32, spacemit_kernel_swiglu_f32, nullptr);
SPACEMIT_ONE_STEP_OP(k_flash_attn, spacemit_kernel_flash_attn, nullptr);

static const spacemit_kernel_fn k_mul_mat_ime_steps[] = { spacemit_mul_mat_quantize, spacemit_mul_mat_gemm };
static const spacemit_op        k_mul_mat_ime         = { k_mul_mat_ime_steps, 2, spacemit_mul_mat_workspace };

// M2e.3 round 2 (adopted 2026-10-14; the paired op is the default): one-token attention split over the tiles, partial
// results then their merge
static const spacemit_kernel_fn k_flash_attn_split_steps[]        = { spacemit_kernel_flash_attn_split_partial,
                                                                      spacemit_kernel_flash_attn_split_merge };
static const spacemit_op        k_flash_attn_split                = { k_flash_attn_split_steps, 2,
                                                                      spacemit_flash_attn_split_workspace };
static const spacemit_kernel_fn k_flash_attn_split_paired_steps[] = { spacemit_kernel_flash_attn_split_partial_paired,
                                                                      spacemit_kernel_flash_attn_split_merge };
static const spacemit_op        k_flash_attn_split_paired         = { k_flash_attn_split_paired_steps, 2,
                                                                      spacemit_flash_attn_split_workspace };

const spacemit_op * spacemit_find_op(const ggml_tensor * op, ggml_backend_buffer_type_t own_buft) {
    const spacemit_op * impl = nullptr;
    switch (op->op) {
        case GGML_OP_ADD:
            impl = spacemit_binary_f32_supported(op) ? &k_add_f32 : nullptr;
            break;
        case GGML_OP_MUL:
            impl = spacemit_binary_f32_supported(op) ? &k_mul_f32 : nullptr;
            break;
        case GGML_OP_RMS_NORM:
            impl = spacemit_rms_norm_f32_supported(op) ? &k_rms_norm_f32 : nullptr;
            break;
        case GGML_OP_ROPE:
            impl = spacemit_rope_supported(op) ? &k_rope : nullptr;
            break;
        case GGML_OP_SET_ROWS:
            impl = spacemit_set_rows_supported(op) ? &k_set_rows : nullptr;
            break;
        case GGML_OP_CPY:
            impl = spacemit_cpy_quant_supported(op) ? &k_cpy_quant : nullptr;
            break;
        case GGML_OP_GET_ROWS:
            impl = spacemit_get_rows_f32_supported(op) ? &k_get_rows_f32 : nullptr;
            break;
        case GGML_OP_GLU:
            impl = spacemit_swiglu_f32_supported(op) ? &k_swiglu_f32 : nullptr;
            break;
        case GGML_OP_MUL_MAT:
            impl = spacemit_mul_mat_ime_supported(op, own_buft) ? &k_mul_mat_ime : nullptr;
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            if (spacemit_flash_attn_supported(op)) {
                const int split = spacemit_flash_attn_split(op);
                impl            = split == 2 ? &k_flash_attn_split_paired :
                                      split == 1 ? &k_flash_attn_split :
                                                   &k_flash_attn;
            }
            break;
        default:
            break;
    }
    if (impl == nullptr || !spacemit_buffer_usable(op, own_buft)) {
        return nullptr;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] != nullptr && !spacemit_buffer_usable(op->src[i], own_buft)) {
            return nullptr;
        }
    }
    return impl;
}
