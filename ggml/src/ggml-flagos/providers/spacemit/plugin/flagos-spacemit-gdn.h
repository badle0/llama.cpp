#pragma once

#include "../core/flagos-spacemit-ops.h"

// The Gated DeltaNet step (GGML_OP_GATED_DELTA_NET, the contract gdn.f32): Qwen3.5's recurrent layers in one node
// (plan.md "Step 2 spec", piece A). One kernel with a portable reference that follows ggml-cpu's and, on the K3, the
// RVV version ported from SpacemiT's; the origin, the changes and what is bit for bit are in the .cpp.

// the rows, (head, sequence) pairs, are split over the tiles; the scratch comes from the shared workspace, this
// kernel's slice per tile (spacemit_gdn_workspace)
bool spacemit_kernel_gated_delta_net(const spacemit_tile * tile, ggml_tensor * dst);

// the state pass's one step (core/flagos-spacemit-ops.h spacemit_state_pass), running on the CPY node: the same
// computation, reading the state from the persistent row the pass's GET_ROWS would have copied in and writing the new
// state into the CPY's destination row, usually the same row. Its workspace is the op's (spacemit_gdn_workspace)
bool spacemit_kernel_gated_delta_net_direct(const spacemit_tile * tile, ggml_tensor * cpy);

// the scratch per tile: 3 * S_v floats (the RVV kernel's delta, decay and attention sums; ggml-cpu's kernel and the
// reference use the first S_v, delta), plus S_v * S_v for the working state when K > 1; with K 1 and 16 tokens and up
// at least the chunkwise form's S_v * S_v + 112 * S_v + 832; padded to a cache line, for SPACEMIT_MAX_TILES tiles
size_t spacemit_gdn_workspace(const ggml_tensor * node);

// reads FLAGOS_SPACEMIT_GDN_PREFILL (auto, recurrence, chunk or tcm) at registration, as spacemit_embd_read_switch
void spacemit_gdn_read_switch();

// test-only (spacemit_gdn_paths_take_fn): the tile calls with rows since the last call by the form they ran
void spacemit_gdn_paths_take(uint64_t * reference, uint64_t * fused, uint64_t * two_pass, uint64_t * chunk,
                             uint64_t * tcm);
