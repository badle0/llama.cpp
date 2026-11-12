#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

// how a tensor is stored in the provider's buffer (plan.md §2.3)
enum class spacemit_layout {
    plain,            // ggml's own layout
    q4_0_32x256,      // IME layout for Q4_0 matmul weights (lossless, same size)
    q4_1_q8_0_32x32,  // Q4_1 matmul weights requantized to Q8_0 in the q8_0 32x32 layout, as Q6_K (34 bytes per 32
                      // weights, not 20; reads are approximate; fusion-and-splitting.md §7.3, 2026-10-23: the in-tree
                      // route's integer zero point cost Llama-3.2-1B +5.5% perplexity)
    q8_0_32x32,       // IME layout for Q8_0 matmul weights (lossless, same size)
    q6_k_q8_0_32x32,  // Q6_K matmul weights requantized to Q8_0 in the q8_0 32x32 layout (needs more than ggml_nbytes:
                      // 34 bytes per 32 weights, not 26.25; reads are approximate)
};

// decided from the tensor's type, shape and name alone, never from how the buffer is used: test-backend-ops places
// weights in ordinary buffers. supports_op, the buffer functions and the kernels all ask this one function.
// A view is stored the way its view_src is; it never has a layout of its own. llama.cpp's cache tensors (named
// cache_k_, cache_v_, cache_r_ and cache_s_ before allocation) always keep ggml's layout, whatever their type and shape,
// as they are written row by row (SET_ROWS, CPY, the context shift's ROPE).
spacemit_layout spacemit_weight_layout(const ggml_tensor * t);

// the layout of the storage behind t: a view answers for its source
spacemit_layout spacemit_storage_layout(const ggml_tensor * t);

// bytes the tensor takes in the provider's buffer (the buffer type's get_alloc_size): ggml_nbytes, except for the
// layouts requantized to Q8_0 (Q4_1, Q6_K)
size_t spacemit_weight_alloc_size(const ggml_tensor * t);

// read and write [offset, offset + size) of a tensor in the provider's buffer in ggml's layout, converting repacked
// tensors. Whole-tensor writes repack directly; partial access goes through a temporary copy of the whole tensor.
// The layouts requantized to Q8_0 read back as the nearest
// Q4_1 or Q6_K (approximate) and accept only whole-tensor writes and fills (plan.md, M2c.2 design).
void spacemit_tensor_write(ggml_tensor * t, const void * data, size_t offset, size_t size);
void spacemit_tensor_read(const ggml_tensor * t, void * data, size_t offset, size_t size);
void spacemit_tensor_fill(ggml_tensor * t, uint8_t value, size_t offset, size_t size);

// true when t, or the tensor it views, is stored repacked
bool spacemit_tensor_is_repacked(const ggml_tensor * t);

// FLAGOS_SPACEMIT_REPACK_THREADS as parsed (1 to 16): the threads a whole-weight repack may use. Read at registration
// too, as spacemit_embd_read_switch, so a bad value's warning shows
uint32_t spacemit_repack_threads();
// test-only: the repacks since the last call that ran on several threads, and the most ranges one of them used.
// flagos-check-spacemit checks FLAGOS_SPACEMIT_REPACK_THREADS by it
void spacemit_repack_take(uint64_t * threaded, uint64_t * max_ranges);
