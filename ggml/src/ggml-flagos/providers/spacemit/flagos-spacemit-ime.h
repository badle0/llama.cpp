#pragma once

#include "flagos-spacemit-ops.h"

// Q4_0 MUL_MAT on the IME (weights in layout q4_0 32x256): step 1 quantizes the activations into the workspace,
// step 2 multiplies. Ported from ggml-spacemit's forward_mul_mat (ime.cpp @ 4e782bc) without its in-kernel barriers.

size_t spacemit_mul_mat_q4_0_workspace(const ggml_tensor * node);
bool   spacemit_mul_mat_q4_0_quantize(const spacemit_tile & tile, ggml_tensor * node);
bool   spacemit_mul_mat_q4_0_gemm(const spacemit_tile & tile, ggml_tensor * node);
