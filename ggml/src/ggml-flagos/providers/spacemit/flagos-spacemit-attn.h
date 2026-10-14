#pragma once

#include "flagos-spacemit-ops.h"

// FLASH_ATTN_EXT on the AI cores (plan.md, M2e design): one step; the rows (query head x token x stream) are split
// statically over the tiles. On the K3 the VLEN-1024 kernels of flagos-spacemit-rvv-attn.cpp, elsewhere (and with
// FLAGOS_SPACEMIT_TEST_REFERENCE) a portable reference following ggml-cpu.

// exactly the FLASH_ATTN_EXT nodes the kernels handle (buffers are checked by spacemit_find_op)
bool spacemit_flash_attn_supported(const ggml_tensor * op);

bool spacemit_kernel_flash_attn(const spacemit_tile & tile, ggml_tensor * dst);

// M2e.3 round 2, candidate (a) (plan.md "M2e.3 design"; the default since 2026-10-14): one-token calls from n_kv 512
// (one stream, llama.cpp's cache layout, no ALiBi or softcap) with the KV positions split over the tiles. Step 1
// computes each tile's partial results into the workspace, the blocks of positions dealt round-robin (with the turns,
// the two tiles of a pair take turns between staging and arithmetic); step 2 merges them per query head and writes dst.
// spacemit_flash_attn_split tells spacemit_find_op which: 2 the split with the turns (the default), 1 the split without
// them, 0 the one-step op above (other calls, or FLAGOS_SPACEMIT_ATTN=one-chunk)
int    spacemit_flash_attn_split(const ggml_tensor * op);
size_t spacemit_flash_attn_split_workspace(const ggml_tensor * node);
bool   spacemit_kernel_flash_attn_split_partial(const spacemit_tile & tile, ggml_tensor * dst);
bool   spacemit_kernel_flash_attn_split_partial_paired(const spacemit_tile & tile, ggml_tensor * dst);
bool   spacemit_kernel_flash_attn_split_merge(const spacemit_tile & tile, ggml_tensor * dst);
