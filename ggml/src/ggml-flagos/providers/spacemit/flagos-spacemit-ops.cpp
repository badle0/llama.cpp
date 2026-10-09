#include "flagos-spacemit-ops.h"

#include "flagos-spacemit-kernels.h"

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

spacemit_kernel_fn spacemit_find_kernel(const ggml_tensor * op, ggml_backend_buffer_type_t own_buft) {
    spacemit_kernel_fn kernel = nullptr;
    switch (op->op) {
        case GGML_OP_ADD:
            kernel = spacemit_add_f32_supported(op) ? spacemit_kernel_add_f32 : nullptr;
            break;
        default:
            break;
    }
    if (kernel == nullptr || !spacemit_buffer_usable(op, own_buft)) {
        return nullptr;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] != nullptr && !spacemit_buffer_usable(op->src[i], own_buft)) {
            return nullptr;
        }
    }
    return kernel;
}
