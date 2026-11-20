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
// reads FLAGOS_SPACEMIT_MM_F32 (blocked or rows) at registration, as spacemit_embd_read_switch does
void spacemit_mm_f32_read_switch();

// the convolution pair (core/flagos-spacemit-ops.h spacemit_conv_pair; layout in the .cpp): concat.f32.tail writes the
// CONCAT's true last 3 columns and, in the rest of its buffer, the convolution's input token-major; ssm_conv.f32.stashed
// reads it, its weights de-interleaved into 4 * C floats of the workspace (spacemit_ssm_conv_stashed_workspace). Both
// split the channels over the tiles alike; their bits are concat.f32's last columns and ssm_conv.f32's outputs
bool   spacemit_kernel_concat_f32_tail(const spacemit_tile * tile, ggml_tensor * dst);
bool   spacemit_kernel_ssm_conv_f32_stashed(const spacemit_tile * tile, ggml_tensor * dst);
size_t spacemit_ssm_conv_stashed_workspace(const ggml_tensor * node);
