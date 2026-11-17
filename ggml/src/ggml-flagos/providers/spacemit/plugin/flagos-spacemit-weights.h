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
    q5_0_q8_0_32x32,  // Q5_0 matmul weights as Q8_0 in the q8_0 32x32 layout, exactly (d (q - 16) is Q8_0's d qs with
                      // qs = q - 16); 34 bytes per 32 weights, not 22; reads restore the GGUF bytes
    q4_1_zp4_32x256,  // 3-D Q4_1 MUL_MAT_ID weights in SpacemiT's 4-bit zero-point layout, under
                      // FLAGOS_SPACEMIT_EXPERT_Q4_1=zp4: d and the nibbles of q4_0 32x256 plus a 4-bit zero point per
                      // row and 32 weights, the stored weight d (q - zp) with zp = round(-m/d) clamped to [0, 15]
                      // (19 bytes per 32 weights, not 20; reads give the stored values as Q4_1, m = fp16(-zp d))
    q4_1_min_32x256,  // 3-D Q4_1 MUL_MAT_ID weights under FLAGOS_SPACEMIT_EXPERT_Q4_1=exact: each 32-row tile in the
                      // q4_0 32x256 form (Q4_1's d and nibbles), then the tile's fp16 minimums; d q + m = d (q - 8) +
                      // (m + 8 d), the q4_0 kernels and a correction (lossless, same size; reads restore the GGUF bytes)
};

// decided from the tensor's type, shape and name alone, never from how the buffer is used: test-backend-ops places
// weights in ordinary buffers. supports_op, the buffer functions and the kernels all ask this one function.
// A 3-D tensor (MUL_MAT_ID's experts) takes the layout of its type under the same rule, its ggml_nrows rows stored as
// one 2-D weight's are, so each expert's rows start at a 32-row group; a 3-D Q4_1 tensor has a layout only under
// FLAGOS_SPACEMIT_EXPERT_Q4_1 (zp4: the 4-bit zero-point layout; q8: the Q8_0 route of 2-D Q4_1; exact: the q4_1 min
// layout), and is plain by default (2-D Q4_1 keeps the Q8_0 route either way). A 4-D tensor never has a layout.
// A view is stored the way its view_src is; it never has a layout of its own. llama.cpp's cache tensors (named
// cache_k_, cache_v_, cache_r_ and cache_s_ before allocation) always keep ggml's layout, whatever their type and shape,
// as they are written row by row (SET_ROWS, CPY, the context shift's ROPE).
spacemit_layout spacemit_weight_layout(const ggml_tensor * t);

// the layout of the storage behind t: a view answers for its source
spacemit_layout spacemit_storage_layout(const ggml_tensor * t);

// bytes the tensor takes in the provider's buffer (the buffer type's get_alloc_size): ggml_nbytes, except for the
// layouts stored as Q8_0 (Q4_1, Q6_K requantized; Q5_0 exactly), which need more. The q4_1 zp4 layout needs fewer
// (19 bytes per 32 weights against Q4_1's 20), so it keeps ggml_nbytes; the q4_1 min layout takes exactly Q4_1's 20
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

// FLAGOS_SPACEMIT_EXPERT_Q4_1 as parsed: 0 cpu (unset or empty, the default: a 3-D Q4_1 weight has no layout, so
// MUL_MAT_ID refuses it and the node stays on the CPU), 1 zp4 (the 4-bit zero-point layout), 2 q8 (requantized to Q8_0
// as 2-D Q4_1 is), 3 exact (the q4_1 min layout). Read at registration too, so a bad value's warning shows
uint32_t spacemit_expert_q4_1();

// FLAGOS_SPACEMIT_REPACK_THREADS as parsed (1 to 16): the threads a whole-weight repack may use. Read at registration
// too, as spacemit_embd_read_switch, so a bad value's warning shows
uint32_t spacemit_repack_threads();
// test-only: the repacks since the last call that ran on several threads (on either route), and the most ranges one of
// them used. flagos-check-spacemit checks FLAGOS_SPACEMIT_REPACK_THREADS by it
void spacemit_repack_take(uint64_t * threaded, uint64_t * max_ranges);
// FLAGOS_SPACEMIT_REPACK_VIA as parsed: 0 threads, 1 x100 (only in a build with spine-runtime, where it is the
// default). Read at registration too, so a bad value's warning shows
uint32_t spacemit_repack_via();
// test-only: under x100 since the last call, the weights repacked through an admitted launch, those that fell back to
// the calling thread (no cores, no stream, launch refused), and the ranges run again on the calling thread
void spacemit_repack_x100_take(uint64_t * through_stream, uint64_t * not_through, uint64_t * reran);
