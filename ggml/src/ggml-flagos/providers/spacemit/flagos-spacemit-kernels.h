#pragma once

#include "flagos-spacemit-ops.h"

// One-step kernels for the element-wise and row ops of a layer (M2a, M2d). On the K3 each uses the RVV version ported
// from ggml-spacemit (flagos-spacemit-rvv-kernels.cpp) when that version handles the tensor's layout correctly, and
// otherwise a portable reference that follows ggml-cpu's implementation; other hosts always use the reference.

// test-only: FLAGOS_SPACEMIT_TEST_REFERENCE=1 runs the portable references (and the scalar IME references) on the AI
// cores, to tell a ported-kernel error from an orchestration error; hosts without the K3 kernels always use them
bool spacemit_use_reference();

bool spacemit_kernel_add_f32(const spacemit_tile & tile, ggml_tensor * dst);
bool spacemit_kernel_mul_f32(const spacemit_tile & tile, ggml_tensor * dst);
bool spacemit_kernel_rms_norm_f32(const spacemit_tile & tile, ggml_tensor * dst);
bool spacemit_kernel_rope_f32(const spacemit_tile & tile, ggml_tensor * dst);
bool spacemit_kernel_set_rows(const spacemit_tile & tile, ggml_tensor * dst);
bool spacemit_kernel_get_rows_f32(const spacemit_tile & tile, ggml_tensor * dst);
bool spacemit_kernel_swiglu_f32(const spacemit_tile & tile, ggml_tensor * dst);

// ROPE keeps one row of cos/sin values per tile in the workspace
size_t spacemit_rope_workspace(const ggml_tensor * node);
