#pragma once

#include "../core/flagos-spacemit-ops.h"

// The Gated DeltaNet step (GGML_OP_GATED_DELTA_NET, the contract gdn.f32): Qwen3.5's recurrent layers in one node
// (plan.md "Step 2 spec", piece A). One kernel with a portable reference that follows ggml-cpu's and, on the K3, the
// RVV version ported from SpacemiT's; the origin, the changes and what is bit for bit are in the .cpp.

// the rows, (head, sequence) pairs, are split over the tiles; the scratch comes from the shared workspace, this
// kernel's slice per tile (spacemit_gdn_workspace)
bool spacemit_kernel_gated_delta_net(const spacemit_tile * tile, ggml_tensor * dst);

// the scratch of ggml-cpu's kernel, per tile: S_v floats (delta), plus S_v * S_v for the working state when K > 1,
// padded to a cache line, for SPACEMIT_MAX_TILES tiles
size_t spacemit_gdn_workspace(const ggml_tensor * node);
