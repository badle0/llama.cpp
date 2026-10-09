#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

// how a tensor is stored in the provider's buffer (plan.md §2.3)
enum class spacemit_layout {
    plain,        // ggml's own layout
    q4_0_32x256,  // IME layout for Q4_0 matmul weights (lossless, same size)
};

// decided from the tensor's type and shape alone, never from how the buffer is used: test-backend-ops places
// weights in ordinary buffers. supports_op, the buffer functions and the kernels all ask this one function.
// A view is stored the way its view_src is; it never has a layout of its own.
spacemit_layout spacemit_weight_layout(const ggml_tensor * t);

// read and write [offset, offset + size) of a tensor in the provider's buffer in ggml's layout, converting repacked
// tensors. Whole-tensor writes repack directly; partial access goes through a temporary copy of the whole tensor.
void spacemit_tensor_write(ggml_tensor * t, const void * data, size_t offset, size_t size);
void spacemit_tensor_read(const ggml_tensor * t, void * data, size_t offset, size_t size);
void spacemit_tensor_fill(ggml_tensor * t, uint8_t value, size_t offset, size_t size);

// true when t, or the tensor it views, is stored repacked
bool spacemit_tensor_is_repacked(const ggml_tensor * t);
