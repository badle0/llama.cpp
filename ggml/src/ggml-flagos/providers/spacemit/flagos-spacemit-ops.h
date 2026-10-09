#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>

// most tiles one launch can have (the K3 has 8 AI cores); per-tile workspace slices are sized for this many
constexpr uint32_t SPACEMIT_MAX_TILES = 64;

// one AI-core tile of a launch: tile ith of nth, with that core's TCM and the workspace all tiles share.
// All tiles of a launch see the same tcm_size (0 and tcm == nullptr when any tile lacks TCM), so they take the
// same kernel path.
struct spacemit_tile {
    uint32_t ith;
    uint32_t nth;
    void *   tcm;
    size_t   tcm_size;
    void *   workspace;
    size_t   workspace_size;
};

// computes this tile's share of one step of a node; false means the node failed and the graph fails
using spacemit_kernel_fn = bool (*)(const spacemit_tile & tile, ggml_tensor * node);

// how the provider runs one op: its steps run in order on every tile with a barrier after each, so a step may read
// what all tiles wrote in the step before; kernels never wait inside a step (plan.md §2.4)
struct spacemit_op {
    const spacemit_kernel_fn * steps;
    uint32_t                   n_steps;
    size_t (*workspace_size)(const ggml_tensor * node);  // bytes of shared workspace, or nullptr for none
};

// the op for a node, or nullptr when the provider does not handle exactly this op, type, shape and placement.
// supports_op and graph_compute both call it, so placement and execution cannot disagree.
const spacemit_op * spacemit_find_op(const ggml_tensor * op, ggml_backend_buffer_type_t own_buft);
