#pragma once

// RVV kernels for the element-wise and row ops, ported from ggml-spacemit (origin and changes: the .cpp); built only
// for the K3 (GGML_FLAGOS_SPACEMIT_RVV). flagos-spacemit-kernels.cpp calls them where they apply.

#include "ggml.h"

#include <cstddef>

#if defined(GGML_FLAGOS_SPACEMIT_RVV)

namespace spacemit_rvv {

// the fields of ggml-spacemit's per-tile context (spacemit-context.h) these kernels read. It has no sync(): a kernel
// cannot wait inside a step (plan.md §2.4)
struct shared_view {
    void * data;
    size_t size;
};

struct context {
    int         ith;
    int         nth;
    void *      workspace;
    size_t      workspace_size;
    shared_view shared;
};

// forward_binary reads src1 rows as contiguous elements; the other layouts use the reference
bool binary_applies(const ggml_tensor * op);

template <ggml_op op_type, typename T> void forward_binary(context & ctx, ggml_tensor * op);
void forward_rms_norm_f32(context & ctx, ggml_tensor * op);
void forward_rope_f32(context & ctx, ggml_tensor * op);  // needs the workspace of spacemit_rope_workspace()
bool forward_set_rows(context & ctx, ggml_tensor * op);  // false: an index out of range
bool forward_get_rows_f32(context & ctx, ggml_tensor * op);  // false: an index out of range
void forward_glu_swiglu_f32(context & ctx, ggml_tensor * op);

}  // namespace spacemit_rvv

#endif  // GGML_FLAGOS_SPACEMIT_RVV
