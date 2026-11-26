#pragma once

#include "../core/flagos-spacemit-ops.h"

#include <cstdint>

// MUL_MAT on 2-D Q4_0 weights (q4_0 32x256) in FP16 arithmetic (plan.md "The dense Q4_0 matmuls in FP16: the port"),
// behind FLAGOS_SPACEMIT_DENSE_Q4_0 (off by default). K is cut in 8 slices; the products step writes each slice's fp16
// partial sums of a 16 x 32 output tile to the workspace, the sum step adds the 8 slices into the f32 output. On the
// K3 the IME2 kernels (flagos-spacemit-f16-kernels.h), elsewhere the scalar references. The arithmetic, the passes and
// the workspace layout are in the .cpp.

// the claim of both steps: the switch on, a 2-D weight in q4_0 32x256, K % 256 == 0 and K <= 32768, M at or above the
// threshold, GGML_PAD(M, 16) * GGML_PAD(N, 32) <= 2^24. Otherwise the int8 contract takes the node
bool   spacemit_mul_mat_f16_supported(const ggml_tensor * op);
size_t spacemit_mul_mat_f16_workspace(const ggml_tensor * node);
bool   spacemit_mul_mat_f16_products(const spacemit_tile * tile, ggml_tensor * node);
bool   spacemit_mul_mat_f16_sum(const spacemit_tile * tile, ggml_tensor * node);

// FLAGOS_SPACEMIT_DENSE_Q4_0 as parsed: 0 off (unset, empty or int8, the default), else the row threshold (f16: 64,
// f16:<rows>: 1 to 4096). Test-only: FLAGOS_SPACEMIT_F16_STATS (1 when set). Both read at registration too, so a bad
// value's warning shows
uint32_t spacemit_dense_q4_0_threshold();
uint32_t spacemit_f16_stats_on();

// test-only (FLAGOS_SPACEMIT_F16_STATS=<path>): appends what the products steps recorded since the last flush to the
// file and clears it; nothing when no matmul was taken. Called when a stream is destroyed
void spacemit_mul_mat_f16_stats_flush();

// test-only: the fp16 bits of weight w's slice `slice` and 32-column tile `col_tile` as the reference dequantization
// stages them, [32 rows][K / 8] into out
void spacemit_mul_mat_f16_stage_weights(const ggml_tensor * w, int64_t slice, int64_t col_tile, uint16_t * out);

// test-only: the two steps with the scalar references whatever the build, so safe on any thread (the kernels' IME code
// runs only on the AI cores)
bool spacemit_mul_mat_f16_products_reference(const spacemit_tile * tile, ggml_tensor * node);
bool spacemit_mul_mat_f16_sum_reference(const spacemit_tile * tile, ggml_tensor * node);
