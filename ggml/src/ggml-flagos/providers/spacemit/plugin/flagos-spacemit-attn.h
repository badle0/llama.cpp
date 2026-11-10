#pragma once

#include "../core/flagos-spacemit-ops.h"

// FLASH_ATTN_EXT on the AI cores (plan.md, M2e design): one step; the rows (query head x token x stream) are split
// statically over the tiles. On the K3 the VLEN-1024 kernels of flagos-spacemit-rvv-attn.cpp, elsewhere (and with
// GGML_FLAGOS_KERNEL_VARIANTS=0) a portable reference following ggml-cpu.

// exactly the FLASH_ATTN_EXT nodes the kernels handle (buffers are checked by spacemit_find_op)
bool spacemit_flash_attn_supported(const ggml_tensor * op);

bool spacemit_kernel_flash_attn(const spacemit_tile * tile, ggml_tensor * dst);

// M2e.3 round 2, candidate (a) (plan.md "M2e.3 design"; the default since 2026-10-14): one-token calls from n_kv 512
// (one stream, llama.cpp's cache layout, no ALiBi or softcap) with the KV positions split over the tiles. Step 1
// computes each tile's partial results into the workspace, the blocks of positions dealt round-robin (with the turns,
// the two tiles of a pair take turns between staging and arithmetic); step 2 merges them per query head and writes dst.
// spacemit_flash_attn_split tells the contracts apart: 2 the split with the turns (the default), 1 the split without
// them, 0 the one-step op above (other calls, or FLAGOS_SPACEMIT_ATTN=one-chunk)
int    spacemit_flash_attn_split(const ggml_tensor * op);
// reads FLAGOS_SPACEMIT_ATTN and FLAGOS_SPACEMIT_ATTN_KVH at registration, so that their warnings are not lost in
// llama.cpp's trial load (common/fit.cpp)
void   spacemit_flash_attn_read_switch();
size_t spacemit_flash_attn_split_workspace(const ggml_tensor * node);
bool   spacemit_kernel_flash_attn_split_partial(const spacemit_tile * tile, ggml_tensor * dst);
bool   spacemit_kernel_flash_attn_split_partial_paired(const spacemit_tile * tile, ggml_tensor * dst);
bool   spacemit_kernel_flash_attn_split_merge(const spacemit_tile * tile, ggml_tensor * dst);

// the shape facts of each attention kernel, the plugin's `supports` in its kernel table (plan.md "S1 step 2 spec" §4):
// the core's FLASH_ATTN_EXT contracts are tried in order and the first whose every step's supports passes is taken, so
// these four and the KV-head split's below carry the variant choice (and with it the switches)
bool spacemit_flash_attn_one_step_supported(const ggml_tensor * op);
bool spacemit_flash_attn_split_supported(const ggml_tensor * op);
bool spacemit_flash_attn_split_paired_supported(const ggml_tensor * op);
bool spacemit_flash_attn_split_any_supported(const ggml_tensor * op);  // the merge step, shared by both split contracts

// the KV-head split for a few tokens (plan.md "The KV-head split for a few tokens, from SpacemiT's fork"): the calls of
// 2 to 8 tokens, turned off by FLAGOS_SPACEMIT_ATTN_KVH=off (one stream, 2, 4 or 8 query heads per KV head, no ALiBi,
// from 128 positions per chunk). Step 1 computes per unit (KV head x chunk of positions, dealt
// round-robin over the tiles) the partial results of the KV head's query heads and tokens into the workspace; step 2
// merges each row's chunks and writes dst. The first attention contract: its supports is false for other calls and
// under FLAGOS_SPACEMIT_ATTN_KVH=off, so the others decide as before
bool   spacemit_flash_attn_kvh_supported(const ggml_tensor * op);
size_t spacemit_flash_attn_kvh_workspace(const ggml_tensor * node);
bool   spacemit_kernel_flash_attn_kvh_partial(const spacemit_tile * tile, ggml_tensor * dst);
bool   spacemit_kernel_flash_attn_kvh_merge(const spacemit_tile * tile, ggml_tensor * dst);
