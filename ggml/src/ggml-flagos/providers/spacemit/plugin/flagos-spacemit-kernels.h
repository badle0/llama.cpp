#pragma once

#include "../core/flagos-spacemit-ops.h"

#include <algorithm>
#include <atomic>
#include <cstdint>

// One-step kernels for the element-wise and row ops of a layer (M2a, M2d). On the K3 each uses the RVV version ported
// from ggml-spacemit (flagos-spacemit-rvv-kernels.cpp) when that version handles the tensor's layout correctly, and
// otherwise a portable reference that follows ggml-cpu's implementation; other hosts always use the reference.

// test-only: GGML_FLAGOS_KERNEL_VARIANTS=0 runs the portable references (and the scalar IME references) on the AI
// cores, to tell a ported-kernel error from an orchestration error; hosts without the K3 kernels always use them
bool spacemit_use_reference();

// test-only: the plugin's environment switches (FLAGOS_SPACEMIT_MM, ATTN, ATTN_KVH, EXPERT_Q4_0) are parsed once and
// cached, not read at every call on every tile (S1 step 2b, 2026-10-30). The check program, which flips them inside
// one process (env_override), calls spacemit_switches_reset, by name through get_proc_address, so the next read parses
// again
uint32_t spacemit_switches_generation();
void     spacemit_switches_reset();

// a switch's cached value: parse() runs on the first read and after every spacemit_switches_reset; T must be an atomic's
// value type (an int, a bool). Readers that see the same stale generation parse the same value twice,
// which is harmless (the environment changes only between graphs, by the check program)
template <typename T, typename Parse>
T spacemit_cached_switch(std::atomic<T> & value, std::atomic<uint32_t> & seen, Parse parse) {
    const uint32_t generation = spacemit_switches_generation();
    if (seen.load(std::memory_order_acquire) != generation) {
        value.store(parse(), std::memory_order_relaxed);
        seen.store(generation, std::memory_order_release);
    }
    return value.load(std::memory_order_relaxed);
}

// [begin, end) of n items for one tile, the way ggml-cpu splits rows over its threads
inline void tile_range(int64_t n, const spacemit_tile & tile, int64_t & begin, int64_t & end) {
    const int64_t per = (n + tile.nth - 1) / tile.nth;
    begin             = std::min(n, per * tile.ith);
    end               = std::min(n, begin + per);
}

bool spacemit_kernel_add_f32(const spacemit_tile * tile, ggml_tensor * dst);
bool spacemit_kernel_mul_f32(const spacemit_tile * tile, ggml_tensor * dst);
bool spacemit_kernel_rms_norm_f32(const spacemit_tile * tile, ggml_tensor * dst);
bool spacemit_kernel_rope(const spacemit_tile * tile, ggml_tensor * dst);       // F32 or F16
bool spacemit_kernel_set_rows(const spacemit_tile * tile, ggml_tensor * dst);   // into F32, F16, quantized or BF16 rows
bool spacemit_kernel_cpy_quant(const spacemit_tile * tile, ggml_tensor * dst);  // F32 into quantized or BF16 rows
bool spacemit_kernel_cpy_f32(const spacemit_tile * tile, ggml_tensor * dst);    // F32 into F32
bool spacemit_kernel_scale_f32(const spacemit_tile * tile, ggml_tensor * dst);  // F32 rows, bias 0
bool spacemit_kernel_get_rows_f32(const spacemit_tile * tile, ggml_tensor * dst);
bool spacemit_kernel_get_rows_quant(const spacemit_tile * tile, ggml_tensor * dst);  // quantized rows into F32
// test-only: the calls of spacemit_kernel_get_rows_quant since the last call (counted by tile 0)
void spacemit_embd_paths_take(uint64_t * calls);
bool spacemit_kernel_swiglu_f32(const spacemit_tile * tile, ggml_tensor * dst);

// ROPE keeps one row of cos/sin values per tile in the workspace
size_t spacemit_rope_workspace(const ggml_tensor * node);
