#pragma once

#include "flagos-spacemit-plugin.h"  // spacemit_tile, spacemit_step_fn, spacemit_plugin_step

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>

// most tiles one launch can have (the K3 has 8 AI cores); per-tile workspace slices are sized for this many
constexpr uint32_t SPACEMIT_MAX_TILES = 64;

// for the 1-row matmuls (flagos-spacemit-ime.cpp; plan.md, "1-row matmul bandwidth") and the FP16 path's pairs route
// (flagos-spacemit-f16.cpp): tile ith waits until its partner, tile ith ^ 1 (on the other core of its pair), has called
// this as often as it has in this launch. Both tiles of a pair must call it equally often in every step. A wait gives
// up, returning false, once the launch has failed (so a partner that skipped the step cannot hang it) or after 10 s;
// the kernel should then return false. The host executor, by default running the tiles one after another, only counts
// the calls there, and fails the launch when a pair's counts differ after a step.
bool spacemit_pair_wait(const spacemit_tile & tile);

// true when tile ith has a partner ((ith ^ 1) < nth) and the launch's tiles run at the same time, so that the two may
// read each other's memory between two pair waits; false on the host's serial executor, where a tile must not read
// what its partner will only write after it ran
bool spacemit_pair_shared(const spacemit_tile & tile);

// how the provider runs one op: its steps run in order on every tile with a barrier after each, so a step may read
// what all tiles wrote in the step before; kernels never wait inside a step (plan.md §2.4), except the two tiles of a
// pair for each other in 1-row matmuls and in the FP16 pairs route (spacemit_pair_wait)
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
    // ssm_conv.f32.stashed for its SSM_CONV, one step each, and a state pass's one step (spacemit_state_pass), in its
    // two forms. The folding one is optional: n_steps 0 when the plugin does not offer it, and the pass keeps
    // state_direct with the clearing SCALE as its own step
    spacemit_op                conv_pair_concat;
    spacemit_op                conv_pair_ssm_conv;
    spacemit_op                state_direct;
    spacemit_op                state_direct_cleared;
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

// A state pass, by the graph's structure alone (no data address): node gi of g is a GATED_DELTA_NET of one sequence
// with K 1 whose state comes, through views only, from a GET_ROWS of one row of a persistent F32 tensor (*ri: its
// index), and whose new state one CPY after it writes back into a row of the same tensor (*ci: its index). All of:
// the op passes gdn.f32; the GET_ROWS reads one I32 index and its source's rows are the state's S_v * S_v * H floats,
// contiguous; the row read in has no other consumer than the op; of the op's output only the attention scores are read
// elsewhere, its state region by that one CPY alone, whose destination is a whole contiguous row of the same
// persistent tensor; nothing after the GET_ROWS but that CPY computes the tensor, and nothing between the two copies
// reads or writes it; none of the row read in, the op's result, or a view of either is a requested output
// (GGML_TENSOR_FLAG_OUTPUT; ggml_set_output marks a view's source too); and all three nodes pass their contracts
// (spacemit_find_op), so GGML_FLAGOS_DISABLE_OPS still sends them to the CPU.
// Then the one step gated_delta_net.f32.direct, running on the CPY node, reads the persistent row and writes the new
// state into the CPY's destination, and the GET_ROWS and the CPY emit no steps (about 10 ms of a Qwen3.5 token on the
// K3). The model: src/llama-graph.cpp build_rs with src/models/delta-net-base.cpp build_recurrent_attn (the !keep
// branch), whose other build_rs nodes (the extra GET_ROWS and its CPY) are empty and ignored.
// A node that writes the tensor before the GET_ROWS is admitted: it wrote before the row was read in the original
// order, and still has when the direct step reads it at the CPY's position. Such a writer keeps its own step, except
// for one clearing node (*zi, -1 without it) folded into the step: at a sequence's start build_rs clears the state
// with a SCALE by 0 in place, and where that SCALE is the only admitted writer, covers exactly the whole contiguous
// row the CPY writes (the same address and byte size), has factor 0 and bias 0, passes its own contract and nothing
// but the pass's two copies reads or writes the tensor between it and the CPY, the step
// gated_delta_net.f32.direct.cleared multiplies that row by 0 itself, each tile its own heads' part, and the SCALE
// emits no step.
// Limit, as the convolution pair's: a consumer placed in another backend's split is not in g, so not seen
bool spacemit_state_pass(const spacemit_ops & ops, ggml_backend_buffer_type_t own_buft, const ggml_cgraph * g, int gi,
                         int * ri, int * ci, int * zi);

// test-only: the state passes graph_compute expanded since the last call (a trusted plan-cache hit reuses its stored
// steps and counts none), over all backends
uint64_t spacemit_state_passes_take();
void     spacemit_state_passes_add(uint64_t n);

// the kernels the core's contracts name; the loader refuses a plugin whose kernels() misses one (plan.md "S1 design"
// §6), and spacemit_ops_bind refuses a contract that names a kernel this list does not carry
extern const char * const spacemit_required_kernels[];
extern const size_t       spacemit_n_required_kernels;
