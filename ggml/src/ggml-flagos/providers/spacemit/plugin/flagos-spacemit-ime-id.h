#pragma once

#include "../core/flagos-spacemit-ops.h"

// quantized MUL_MAT_ID on the IME (the experts' matmuls of a mixture-of-experts layer), the experts' weight in an IME
// layout (flagos-spacemit-weights.h: a 3-D weight's rows in 32-row groups, expert e's from row e * N). Step 1
// quantizes every activation row once into the workspace; step 2 groups the (token, slot) pairs by expert as ggml-cpu
// does, every tile alone from the same ids, deals (pair, 32 columns) units over the tiles in that order and multiplies
// each pair's activation row by its expert's rows with the 1-row kernels, the weights read from DRAM. The output rows
// are ggml-cpu's. Rows of one expert are consecutive in a tile's share, where a 4-row kernel and TCM staging (the
// fork's ime.cpp forward_mul_mat_id) can be added later.

// the shared workspace: the quantized activations, then one id grouping per tile for SPACEMIT_MAX_TILES tiles, each
// padded to a cache line
size_t spacemit_mul_mat_id_workspace(const ggml_tensor * node);
bool   spacemit_mul_mat_id_quantize(const spacemit_tile * tile, ggml_tensor * node);
// false (the graph fails) when an id is outside [0, n_expert), which ggml-cpu asserts in debug builds only
bool   spacemit_mul_mat_id_gemm(const spacemit_tile * tile, ggml_tensor * node);
