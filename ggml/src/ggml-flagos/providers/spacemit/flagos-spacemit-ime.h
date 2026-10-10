#pragma once

#include "flagos-spacemit-ops.h"

// quantized MUL_MAT on the IME, weights in an IME layout (flagos-spacemit-weights.h: Q4_0 32x256, Q4_1 32x32): step 1
// quantizes the activations into the workspace, step 2 multiplies. Ported from ggml-spacemit's forward_mul_mat
// (ime.cpp @ 4e782bc) without its in-kernel barriers.

size_t spacemit_mul_mat_workspace(const ggml_tensor * node);
bool   spacemit_mul_mat_quantize(const spacemit_tile & tile, ggml_tensor * node);
bool   spacemit_mul_mat_gemm(const spacemit_tile & tile, ggml_tensor * node);
