#pragma once

#include "flagos-spacemit-plugin.h"  // spacemit_tile, spacemit_step_fn, spacemit_plugin_step

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>

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
constexpr uint32_t SPACEMIT_MAX_OP_STEPS = 2;

// a contract bound to the plugin's kernels: its steps in order, each the plugin's entry under the contract's name
struct spacemit_op {
    const spacemit_plugin_kernel * steps[SPACEMIT_MAX_OP_STEPS];
    uint32_t                       n_steps;
};

// the core's contracts bound to one plugin's kernel table (spacemit_ops_bind); the table is kept for the layout rule
struct spacemit_ops {
    const spacemit_plugin_v1 * table = nullptr;
    spacemit_op                ops[32];  // one per contract, in the contract table's order
    uint32_t                   n_ops = 0;
    // not contracts: the steps of a convolution pair (spacemit_conv_pair), concat.f32.tail for its CONCAT and
    // ssm_conv.f32.stashed for its SSM_CONV, one step each
    spacemit_op                conv_pair_concat;
    spacemit_op                conv_pair_ssm_conv;
};

// binds every contract's step names to table->kernels(); false, with detail naming the step, when a name is missing
// (the loader refuses such a plugin first; this is the core's own guard) or when a contract names a kernel that
// spacemit_required_kernels does not list (the two lists must agree)
bool spacemit_ops_bind(const spacemit_plugin_v1 * table, spacemit_ops * out, std::string * detail);

// the op for a node, or nullptr when the provider does not handle exactly this op, type, shape and placement: the
// first contract of the node's op whose structural check and whose every step's supports (where not NULL) pass, and
// whose op GGML_FLAGOS_DISABLE_OPS does not name (the diagnostic that sends an op back to the CPU).
// supports_op and graph_compute both call it, so placement and execution cannot disagree.
const spacemit_op * spacemit_find_op(const spacemit_ops & ops, const ggml_tensor * op,
                                     ggml_backend_buffer_type_t own_buft);

// the shared workspace the op needs for node: the largest of its steps' workspace_size (each distinct function once)
size_t spacemit_op_workspace(const spacemit_op * op, const ggml_tensor * node);

// A convolution pair, by the graph's structure alone (no data address): node ci of g is a CONCAT along dimension 0 of
// F32 [3, C, 1] state columns and F32 [n, C, 1] tokens (any strides) into a contiguous F32 [n + 3, C, 1, 1], not an
// output, n >= 32, whose consumers in g are exactly one SSM_CONV after it (*si: its index; the CONCAT its src[0], a
// kernel of width 4, a contiguous F32 [C, n, 1] output) and views of it; and every tensor of g viewing it (a node, or
// a node's source) is a window of whole columns among its last 3 (same type and strides, row 0 on, ne[2] and ne[3]
// 1). Both nodes must pass their contracts (spacemit_find_op), so GGML_FLAGOS_DISABLE_OPS still sends them to the CPU.
// Then the CONCAT's buffer but those columns is the pair's to lay out (flagos-spacemit-seq.cpp): its CONCAT runs
// concat.f32.tail and its SSM_CONV ssm_conv.f32.stashed. ggml's allocator keeps that buffer until its last consumer
// ran and never puts a CONCAT in place of a source. The model: SpacemiT's fork, ssm_conv_concat_fusable (ours admits
// views of the last 3 columns only). Limit, as the fork's: a consumer placed in another backend's split is not in g,
// so not seen; llama.cpp's graphs (delta-net-base.cpp build_conv_state, n_rs_seq 0) have only the two above
bool spacemit_conv_pair(const spacemit_ops & ops, ggml_backend_buffer_type_t own_buft, const ggml_cgraph * g, int ci,
                        int * si);

// test-only: the convolution pairs graph_compute expanded since the last call (a trusted plan-cache hit reuses its
// stored steps and counts none), over all backends
uint64_t spacemit_conv_pairs_take();
void     spacemit_conv_pairs_add(uint64_t n);

// the kernels the core's contracts name; the loader refuses a plugin whose kernels() misses one (plan.md "S1 design"
// §6), and spacemit_ops_bind refuses a contract that names a kernel this list does not carry
extern const char * const spacemit_required_kernels[];
extern const size_t       spacemit_n_required_kernels;
