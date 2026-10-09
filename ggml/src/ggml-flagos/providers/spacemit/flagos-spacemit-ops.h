#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>

// one AI-core tile of a launch: tile ith of nth, with that core's TCM
struct spacemit_tile {
    uint32_t ith;
    uint32_t nth;
    void *   tcm;
    size_t   tcm_size;
};

// computes this tile's share of one node; false means the node failed and the graph fails
using spacemit_kernel_fn = bool (*)(const spacemit_tile & tile, ggml_tensor * node);

// kernel for a node, or nullptr when the provider does not handle exactly this op, type, shape and placement.
// supports_op and graph_compute both call it, so placement and execution cannot disagree.
spacemit_kernel_fn spacemit_find_kernel(const ggml_tensor * op, ggml_backend_buffer_type_t own_buft);
