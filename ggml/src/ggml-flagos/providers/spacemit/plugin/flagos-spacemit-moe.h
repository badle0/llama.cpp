#pragma once

#include "../core/flagos-spacemit-ops.h"

// The kernels of the MoE router (Qwen3.6-35B-A3B, build_moe_ffn in src/llama-graph.cpp): SOFT_MAX without a mask,
// ARGSORT, SUM_ROWS, CLAMP and DIV, all on F32 (ARGSORT into I32). Each has a portable reference that follows
// ggml-cpu's own code with the rows split over the tiles; SOFT_MAX, SUM_ROWS and DIV also have the RVV body of
// SpacemiT's ggml-spacemit, used on the K3 unless GGML_FLAGOS_KERNEL_VARIANTS=0.

bool spacemit_kernel_soft_max_f32(const spacemit_tile * tile, ggml_tensor * dst);
bool spacemit_kernel_argsort_f32(const spacemit_tile * tile, ggml_tensor * dst);
bool spacemit_kernel_sum_rows_f32(const spacemit_tile * tile, ggml_tensor * dst);
bool spacemit_kernel_clamp_f32(const spacemit_tile * tile, ggml_tensor * dst);
bool spacemit_kernel_div_f32(const spacemit_tile * tile, ggml_tensor * dst);
