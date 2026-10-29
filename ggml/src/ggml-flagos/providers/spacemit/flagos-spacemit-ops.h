#pragma once

#include "core/flagos-spacemit-plugin.h"  // spacemit_tile, spacemit_step_fn, spacemit_plugin_step

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>

// most tiles one launch can have (the K3 has 8 AI cores); per-tile workspace slices are sized for this many
constexpr uint32_t SPACEMIT_MAX_TILES = 64;

// for the 1-row matmuls (flagos-spacemit-ime.cpp; plan.md, "1-row matmul bandwidth"): tile ith waits until its partner,
// tile ith ^ 1 (on the other core of its pair), has called this as often as it has in this launch. Both tiles of a pair
// must call it equally often in every step. A wait gives up, returning false, once the launch has failed (so a partner
// that skipped the step cannot hang it) or after 10 s; the kernel should then return false. The host executor, by
// default running the tiles one after another, only counts the calls there, and fails the launch when a pair's counts
// differ after a step.
bool spacemit_pair_wait(const spacemit_tile & tile);

// computes this tile's share of one step of a node; false means the node failed and the graph fails
using spacemit_kernel_fn = spacemit_step_fn;

// how the provider runs one op: its steps run in order on every tile with a barrier after each, so a step may read
// what all tiles wrote in the step before; kernels never wait inside a step (plan.md §2.4), except the two tiles of a
// pair for each other in 1-row matmuls (spacemit_pair_wait)
struct spacemit_op {
    const spacemit_kernel_fn * steps;
    uint32_t                   n_steps;
    size_t (*workspace_size)(const ggml_tensor * node);  // bytes of shared workspace, or nullptr for none
};

// the op for a node, or nullptr when the provider does not handle exactly this op, type, shape and placement.
// supports_op and graph_compute both call it, so placement and execution cannot disagree.
const spacemit_op * spacemit_find_op(const ggml_tensor * op, ggml_backend_buffer_type_t own_buft);

// the kernels the core's contracts name; the loader refuses a plugin whose kernels() misses one (plan.md "S1 design"
// §6). Step 2 binds the step functions by these names; in step 1 the contracts still hold the functions themselves
extern const char * const spacemit_required_kernels[];
extern const size_t       spacemit_n_required_kernels;
