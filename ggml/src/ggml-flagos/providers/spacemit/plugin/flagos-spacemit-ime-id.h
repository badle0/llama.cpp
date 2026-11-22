#pragma once

#include "../core/flagos-spacemit-ops.h"

// quantized MUL_MAT_ID on the IME (the experts' matmuls of a mixture-of-experts layer), the experts' weight in an IME
// layout (flagos-spacemit-weights.h: a 3-D weight's rows in 32-row groups, expert e's from row e * N). Both steps
// group the (token, slot) pairs by expert as ggml-cpu does, every tile alone from the same ids.
// One token (and FLAGOS_SPACEMIT_MMID=rows for any count): step 1 quantizes every activation row once, step 2 deals
// (pair, 32 columns) units over the tiles in the grouping's order and multiplies each pair's activation row by its
// expert's rows with the 1-row kernels, the weights read from DRAM.
// Several tokens: step 1 quantizes one row per pair, at the pair's place in that order, the 4 rows of a group
// interleaved for the 4-row kernel; step 2's tiles claim chunks of (expert, 32-column slab) units, so an expert's
// weights are read once per node, and multiply a slab by every group of its expert, staged through TCM by default. The
// output rows are ggml-cpu's. A weight in the q4_0s layout (3-D Q4_0 unless FLAGOS_SPACEMIT_EXPERT_Q4_0=plain) packs
// an expert's rows in groups of 8, then one of 4 and single rows, for SpacemiT's fork's signed kernels.

// the shared workspace: the quantized activations, then one id grouping, one gathered group of 4 rows and one 4 x 32
// result per tile for SPACEMIT_MAX_TILES tiles, each padded to a cache line, then the cache line of step 2's claim
// counter (several tokens)
size_t spacemit_mul_mat_id_workspace(const ggml_tensor * node);
bool   spacemit_mul_mat_id_quantize(const spacemit_tile * tile, ggml_tensor * node);
// false (the graph fails) when an id is outside [0, n_expert), which ggml-cpu asserts in debug builds only
bool   spacemit_mul_mat_id_gemm(const spacemit_tile * tile, ggml_tensor * node);
