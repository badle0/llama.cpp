#pragma once

#include "../core/flagos-spacemit-ops.h"

// The kernels of Qwen3.5's Gated DeltaNet layers beside the state update (plan.md "Step 2 spec", pieces D to G):
// SILU, SIGMOID and SOFTPLUS on F32 rows, L2_NORM, SSM_CONV, dimension-0 CONCAT, and MUL_MAT on an F32 weight with no
// IME layout (ssm_alpha, ssm_beta). Each is a portable reference that follows ggml-cpu's own code, with the rows split
// over the tiles. CONT needs no kernel of its own: its contract names cpy.f32 (flagos-spacemit-kernels.cpp).

bool spacemit_kernel_silu_f32(const spacemit_tile * tile, ggml_tensor * dst);
bool spacemit_kernel_sigmoid_f32(const spacemit_tile * tile, ggml_tensor * dst);
bool spacemit_kernel_softplus_f32(const spacemit_tile * tile, ggml_tensor * dst);
bool spacemit_kernel_l2_norm_f32(const spacemit_tile * tile, ggml_tensor * dst);
bool spacemit_kernel_ssm_conv_f32(const spacemit_tile * tile, ggml_tensor * dst);
bool spacemit_kernel_concat_f32(const spacemit_tile * tile, ggml_tensor * dst);
bool spacemit_kernel_mul_mat_f32(const spacemit_tile * tile, ggml_tensor * dst);
