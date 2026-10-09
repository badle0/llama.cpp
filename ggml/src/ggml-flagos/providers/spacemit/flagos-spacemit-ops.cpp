#include "flagos-spacemit-ops.h"

#include "flagos-spacemit-ime.h"
#include "flagos-spacemit-kernels.h"
#include "flagos-spacemit-weights.h"

// an operand is usable when it is not allocated yet (placement time), or lives in our buffer or in host memory
static bool spacemit_buffer_usable(const ggml_tensor * t, ggml_backend_buffer_type_t own_buft) {
    const ggml_backend_buffer_t buffer = t->view_src != nullptr ? t->view_src->buffer : t->buffer;
    if (buffer == nullptr) {
        return true;
    }
    const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buffer);
    return buft == own_buft || ggml_backend_buft_is_host(buft);
}

static bool spacemit_add_f32_supported(const ggml_tensor * op) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    return op->type == GGML_TYPE_F32 && a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(a, b) && ggml_are_same_shape(a, op) &&
           ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(op);
}

// Q4_0 weight in the IME layout, read in that layout: so it must sit in our buffer (or be unallocated at placement
// time), never in a host buffer. The answer must not depend on the row count: llama.cpp asks once per weight, at
// load, with 512 rows (plan.md M2b design).
static bool spacemit_mul_mat_q4_0_supported(const ggml_tensor * op, ggml_backend_buffer_type_t own_buft) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (spacemit_weight_layout(w) != spacemit_layout::q4_0_32x256) {
        return false;
    }
    if (w->buffer != nullptr && ggml_backend_buffer_get_type(w->buffer) != own_buft) {
        return false;
    }
    return x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_is_contiguous(op);
}

static const spacemit_kernel_fn k_add_f32_steps[] = { spacemit_kernel_add_f32 };
static const spacemit_op        k_add_f32         = { k_add_f32_steps, 1, nullptr };

static const spacemit_kernel_fn k_mul_mat_q4_0_steps[] = { spacemit_mul_mat_q4_0_quantize, spacemit_mul_mat_q4_0_gemm };
static const spacemit_op        k_mul_mat_q4_0         = { k_mul_mat_q4_0_steps, 2, spacemit_mul_mat_q4_0_workspace };

const spacemit_op * spacemit_find_op(const ggml_tensor * op, ggml_backend_buffer_type_t own_buft) {
    const spacemit_op * impl = nullptr;
    switch (op->op) {
        case GGML_OP_ADD:
            impl = spacemit_add_f32_supported(op) ? &k_add_f32 : nullptr;
            break;
        case GGML_OP_MUL_MAT:
            impl = spacemit_mul_mat_q4_0_supported(op, own_buft) ? &k_mul_mat_q4_0 : nullptr;
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
