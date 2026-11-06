#pragma once

#include "../core/flagos-spacemit-ops.h"

// quantized MUL_MAT on the IME, weights in an IME layout (flagos-spacemit-weights.h: Q4_0 32x256; Q8_0, and Q4_1 and
// Q6_K requantized, in q8_0 32x32): step 1 quantizes the activations into the workspace, step 2 multiplies. Ported
// from ggml-spacemit's forward_mul_mat (ime.cpp @ 4e782bc) without its in-kernel barriers, except that 1-row matmuls
// keep the in-tree path B's waits between the two tiles of a pair (spacemit_pair_wait), and that path A deals the
// batch's last m % 4 rows to the tiles idle in its last round instead of leaving them all on one.

size_t spacemit_mul_mat_workspace(const ggml_tensor * node);
bool   spacemit_mul_mat_quantize(const spacemit_tile * tile, ggml_tensor * node);
bool   spacemit_mul_mat_gemm(const spacemit_tile * tile, ggml_tensor * node);

// whether MUL_MAT b quantizes the same activations as MUL_MAT a into the same layout, so that b can multiply what a's
// quantize step wrote (FLAGOS_SPACEMIT_QUANTIZE=shared, spacemit_backend_graph_compute)
bool spacemit_mul_mat_same_activations(const ggml_tensor * a, const ggml_tensor * b);

// test-only: the multiply steps that took path A, path B (several rows) and path C since the last call, one count per
// step; the 1-row path is not counted (flagos-check-spacemit checks the paths the default and FLAGOS_SPACEMIT_MM=split
// take)
void spacemit_mm_paths_take(uint64_t * path_a, uint64_t * path_b, uint64_t * path_c);

// test-only: of path A's last m % 4 rows since the last call, how many each tile multiplied and in how many calls
// (rows[i] and calls[i] for tile i < n_tiles; all tiles' counts reset). flagos-check-spacemit checks that they are
// dealt to the tiles idle in the last round, a tile's in one call
void spacemit_mm_tail_take(uint64_t * rows, uint64_t * calls, uint32_t n_tiles);
