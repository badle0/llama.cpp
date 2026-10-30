#include "flagos-spacemit-ops.h"

#include "../../../ggml-impl.h"

#include <cstring>

// The core's contract table (plan.md "S1 step 2 spec"): per contract the ggml op, the structural check (operand types,
// shapes, modes and placement, the layout rule through the plugin's tensor_layout) and the step names in order. Every
// name is bound once, at load, to the plugin's kernel entry (spacemit_ops_bind), so this file calls no kernel code and
// includes no kernel header; the kernels' own shape facts are their entries' `supports`.

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
static bool spacemit_binary_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                          ggml_backend_buffer_type_t) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    return spacemit_f32_rows(op) && spacemit_f32_rows(a) && b->type == GGML_TYPE_F32 && ggml_can_repeat(b, a) &&
           ggml_are_same_shape(a, op);
}

// RMS_NORM: any eps >= 0 (ggml-cpu asserts the same)
static bool spacemit_rms_norm_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                            ggml_backend_buffer_type_t) {
    float eps;
    std::memcpy(&eps, op->op_params, sizeof(float));
    return spacemit_f32_rows(op) && spacemit_f32_rows(op->src[0]) && ggml_are_same_shape(op->src[0], op) && eps >= 0.0f;
}

// ROPE (forward): F32 or F16 rows into rows of the same type (F16: the context shift's ROPE in place on the KV cache,
// src/llama-kv-cache.cpp:1878-1883); modes NORMAL and NEOX; I32 positions, one per token; optional F32 frequency
// factors; the channels past n_dims are copied in pairs, so ne0 is even
static bool spacemit_rope_supported(const ggml_tensor * op, const spacemit_plugin_v1 *, ggml_backend_buffer_type_t) {
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
static bool spacemit_set_rows_supported(const ggml_tensor * op, const spacemit_plugin_v1 * table,
                                        ggml_backend_buffer_type_t) {
    const ggml_tensor * src = op->src[0];
    const ggml_tensor * idx = op->src[1];
    if (op->ne[0] != src->ne[0] || (idx->type != GGML_TYPE_I64 && idx->type != GGML_TYPE_I32) ||
        table->tensor_layout(op) != 0) {
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
static bool spacemit_cpy_quant_supported(const ggml_tensor * op, const spacemit_plugin_v1 * table,
                                         ggml_backend_buffer_type_t) {
    const ggml_tensor * src = op->src[0];
    return spacemit_cache_block_type(op->type) && spacemit_f32_rows(src) && ggml_are_same_shape(src, op) &&
           ggml_is_contiguous(op) && op->ne[0] % ggml_blck_size(op->type) == 0 && table->tensor_layout(op) == 0;
}

// GET_ROWS: F32 rows by I32 indices
static bool spacemit_get_rows_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                            ggml_backend_buffer_type_t) {
    return spacemit_f32_rows(op) && spacemit_f32_rows(op->src[0]) && op->src[1]->type == GGML_TYPE_I32 &&
           op->ne[0] == op->src[0]->ne[0];
}

// GLU: SwiGLU only, split (two tensors) and single-tensor forms; rows addressed as in ggml-cpu (ggml_is_contiguous_1),
// dst shaped as ggml-cpu asserts (the kernels write nrows x nc elements of it)
static bool spacemit_swiglu_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                          ggml_backend_buffer_type_t) {
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
// type, shape and name, the plugin's rule through the table (tensor_layout, which answers for a view's storage, so a
// view is excluded here): llama.cpp's cache tensors never have one, so a MUL_MAT never reads a quantized KV cache as a
// weight. The answer must not depend on the row count: llama.cpp asks once per weight, at load, with 512 rows
// (plan.md M2b design).
static bool spacemit_mul_mat_ime_supported(const ggml_tensor * op, const spacemit_plugin_v1 * table,
                                           ggml_backend_buffer_type_t own_buft) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (w->view_src != nullptr || table->tensor_layout(w) == 0) {
        return false;
    }
    if (w->buffer != nullptr && ggml_backend_buffer_get_type(w->buffer) != own_buft) {
        return false;
    }
    return x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_is_contiguous(op);
}

// the attention contracts' type and shape facts all sit in the kernels' supports today; moving their type part into
// the core is the Q4-vocabulary step (plan.md "S1 design" §1)
static bool spacemit_always(const ggml_tensor *, const spacemit_plugin_v1 *, ggml_backend_buffer_type_t) {
    return true;
}

// one contract: the ggml op it claims, its structural check, and its steps' names in order. Several contracts may
// claim one op (attention's three variants); spacemit_find_op takes the first whose checks pass
struct spacemit_contract {
    ggml_op op;
    bool (*check)(const ggml_tensor * op, const spacemit_plugin_v1 * table, ggml_backend_buffer_type_t own_buft);
    const char * steps[SPACEMIT_MAX_OP_STEPS];
    uint32_t     n_steps;
};

// M2e.3 round 2 (adopted 2026-10-14; the paired split is the default): one-token attention split over the tiles,
// partial results then their merge. The three FLASH_ATTN_EXT contracts are tried in this order and the kernels'
// supports pick one (spacemit_flash_attn_split, which reads the shape and FLAGOS_SPACEMIT_ATTN)
static const spacemit_contract k_contracts[] = {
    { GGML_OP_ADD,             spacemit_binary_f32_supported,   { "add.f32" },                                      1 },
    { GGML_OP_MUL,             spacemit_binary_f32_supported,   { "mul.f32" },                                      1 },
    { GGML_OP_RMS_NORM,        spacemit_rms_norm_f32_supported, { "rms_norm.f32" },                                  1 },
    { GGML_OP_ROPE,            spacemit_rope_supported,         { "rope" },                                          1 },
    { GGML_OP_SET_ROWS,        spacemit_set_rows_supported,     { "set_rows" },                                      1 },
    { GGML_OP_CPY,             spacemit_cpy_quant_supported,    { "cpy.quant" },                                     1 },
    { GGML_OP_GET_ROWS,        spacemit_get_rows_f32_supported, { "get_rows.f32" },                                  1 },
    { GGML_OP_GLU,             spacemit_swiglu_f32_supported,   { "swiglu.f32" },                                    1 },
    { GGML_OP_MUL_MAT,         spacemit_mul_mat_ime_supported,  { "mul_mat.quantize", "mul_mat.gemm" },              2 },
    { GGML_OP_FLASH_ATTN_EXT,  spacemit_always,
      { "flash_attn.split.partial_paired", "flash_attn.split.merge" },                                               2 },
    { GGML_OP_FLASH_ATTN_EXT,  spacemit_always, { "flash_attn.split.partial", "flash_attn.split.merge" },            2 },
    { GGML_OP_FLASH_ATTN_EXT,  spacemit_always, { "flash_attn" },                                                    1 },
};

static constexpr size_t k_n_contracts = sizeof(k_contracts) / sizeof(k_contracts[0]);

static_assert(k_n_contracts <= sizeof(spacemit_ops::ops) / sizeof(spacemit_ops::ops[0]),
              "spacemit_ops::ops must hold every contract");

// the contract names of the 14 kernels the plugin offers, in the order of its kernel table (plan.md "S1 step 1 spec" §3)
const char * const spacemit_required_kernels[] = {
    "add.f32",          "mul.f32",          "rms_norm.f32",             "rope",
    "set_rows",         "cpy.quant",        "get_rows.f32",             "swiglu.f32",
    "mul_mat.quantize", "mul_mat.gemm",     "flash_attn",               "flash_attn.split.partial",
    "flash_attn.split.partial_paired",      "flash_attn.split.merge",
};
const size_t spacemit_n_required_kernels = sizeof(spacemit_required_kernels) / sizeof(spacemit_required_kernels[0]);

static bool spacemit_is_required_kernel(const char * name) {
    for (size_t i = 0; i < spacemit_n_required_kernels; i++) {
        if (std::strcmp(name, spacemit_required_kernels[i]) == 0) {
            return true;
        }
    }
    return false;
}

bool spacemit_ops_bind(const spacemit_plugin_v1 * table, spacemit_ops * out, std::string * detail) {
    const auto fail = [detail](const std::string & text) {
        if (detail != nullptr) {
            *detail = text;
        }
        return false;
    };
    if (table == nullptr || out == nullptr) {
        return fail("no plugin table");
    }
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    if (kernels == nullptr) {
        return fail("the plugin offers no kernels");
    }
    out->table = nullptr;
    out->n_ops = 0;
    for (size_t i = 0; i < k_n_contracts; i++) {
        const spacemit_contract & contract = k_contracts[i];
        spacemit_op &             op       = out->ops[i];
        op.n_steps                         = contract.n_steps;
        for (uint32_t s = 0; s < contract.n_steps; s++) {
            const char * name = contract.steps[s];
            if (!spacemit_is_required_kernel(name)) {
                return fail(std::string(name) + " is not a required kernel");
            }
            const spacemit_plugin_kernel * entry = nullptr;
            for (uint32_t k = 0; k < n_kernels && entry == nullptr; k++) {
                if (kernels[k].name != nullptr && std::strcmp(kernels[k].name, name) == 0) {
                    entry = &kernels[k];
                }
            }
            if (entry == nullptr) {
                return fail(std::string("no kernel ") + name);
            }
            op.steps[s] = entry;
        }
    }
    out->table = table;
    out->n_ops = (uint32_t) k_n_contracts;
    return true;
}

const spacemit_op * spacemit_find_op(const spacemit_ops & ops, const ggml_tensor * op,
                                     ggml_backend_buffer_type_t own_buft) {
    if (ops.table == nullptr || !spacemit_buffer_usable(op, own_buft)) {
        return nullptr;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] != nullptr && !spacemit_buffer_usable(op->src[i], own_buft)) {
            return nullptr;
        }
    }
    for (uint32_t i = 0; i < ops.n_ops; i++) {
        if (k_contracts[i].op != op->op || !k_contracts[i].check(op, ops.table, own_buft)) {
            continue;
        }
        const spacemit_op & impl    = ops.ops[i];
        bool                claimed = true;
        for (uint32_t s = 0; s < impl.n_steps && claimed; s++) {
            claimed = impl.steps[s]->supports == nullptr || impl.steps[s]->supports(op);
        }
        if (claimed) {
            return &impl;
        }
    }
    return nullptr;
}

size_t spacemit_op_workspace(const spacemit_op * op, const ggml_tensor * node) {
    if (op == nullptr) {
        return 0;
    }
    size_t bytes = 0;
    for (uint32_t s = 0; s < op->n_steps; s++) {
        size_t (*const rule)(const ggml_tensor *) = op->steps[s]->workspace_size;
        if (rule == nullptr) {
            continue;
        }
        bool asked = false;  // two steps may share one rule (both MUL_MAT steps, both split steps): ask it once
        for (uint32_t t = 0; t < s; t++) {
            asked = asked || op->steps[t]->workspace_size == rule;
        }
        if (!asked) {
            const size_t size = rule(node);
            bytes             = size > bytes ? size : bytes;
        }
    }
    return bytes;
}
