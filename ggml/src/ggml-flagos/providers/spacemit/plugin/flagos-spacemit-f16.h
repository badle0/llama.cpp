#pragma once

#include "../core/flagos-spacemit-ops.h"

#include <cstdint>

// MUL_MAT on 2-D Q4_0 weights (q4_0 32x256) in FP16 arithmetic (plan.md "The dense Q4_0 matmuls in FP16: the port"),
// behind FLAGOS_SPACEMIT_DENSE_Q4_0 (on by default). K is cut in 8 slices (the plain route) or 4 (the pairs route,
// FLAGOS_SPACEMIT_F16_ROUTE=pairs); the products step writes each slice's fp16 partial sums of a 16 x 32 output tile to
// the workspace, the sum step adds the slices into the f32 output. On the K3 the IME2 kernels
// (flagos-spacemit-f16-kernels.h), elsewhere the scalar references. The routes, the arithmetic, the passes and the
// workspace layout are in the .cpp.

// the claim of both steps: the switch on, a 2-D weight in q4_0 32x256, K % 256 == 0 and K <= 32768, M at or above the
// threshold, GGML_PAD(M, 16) * GGML_PAD(N, 32) <= 2^24. Otherwise the int8 contract takes the node
bool   spacemit_mul_mat_f16_supported(const ggml_tensor * op);
size_t spacemit_mul_mat_f16_workspace(const ggml_tensor * node);
bool   spacemit_mul_mat_f16_products(const spacemit_tile * tile, ggml_tensor * node);
bool   spacemit_mul_mat_f16_sum(const spacemit_tile * tile, ggml_tensor * node);

// FLAGOS_SPACEMIT_DENSE_Q4_0 as parsed: 0 off (int8), else the row threshold (unset, empty, f16 or a bad value: 64,
// the default; f16:<rows>: 1 to 4096). Test-only: FLAGOS_SPACEMIT_F16_STATS (1 when set). Both read at registration
// too, so a bad value's warning shows
uint32_t spacemit_dense_q4_0_threshold();
uint32_t spacemit_f16_stats_on();

// FLAGOS_SPACEMIT_F16_ROUTE (test-only) as parsed: 0 the plain route (unset, empty, plain or a bad value, the
// default), 1 the pairs route where the shape, 8 tiles and the stage admit it (the .cpp's route_of). Read at
// registration too
uint32_t spacemit_f16_route_pairs();

// test-only (FLAGOS_SPACEMIT_F16_STATS=<path>): appends what the products steps recorded since the last flush to the
// file and clears it; nothing when no matmul was taken. Called when a stream is destroyed
void spacemit_mul_mat_f16_stats_flush();

// test-only: the fp16 bits of weight w's slice `slice` and 32-column tile `col_tile` as the reference dequantization
// stages them, [32 rows][K / 8] into out
void spacemit_mul_mat_f16_stage_weights(const ggml_tensor * w, int64_t slice, int64_t col_tile, uint16_t * out);

// test-only: the route both steps take for node on nth tiles with TCM of tcm_size bytes (0: none), under the switch
// as parsed now: out[0] the slices (8 or 4), out[1] the weight slots, out[2] the row passes, out[3] 1 when the stage
// is the TCM (out[1] to out[3] 0 on the plain route)
void spacemit_mul_mat_f16_route(const ggml_tensor * node, uint32_t nth, size_t tcm_size, int64_t * out);

// test-only: the two steps with the scalar references whatever the build, so safe on any thread (the kernels' IME code
// runs only on the AI cores)
bool spacemit_mul_mat_f16_products_reference(const spacemit_tile * tile, ggml_tensor * node);
bool spacemit_mul_mat_f16_sum_reference(const spacemit_tile * tile, ggml_tensor * node);
