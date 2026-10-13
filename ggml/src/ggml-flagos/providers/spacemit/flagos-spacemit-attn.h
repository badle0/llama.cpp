#pragma once

#include "flagos-spacemit-ops.h"

// FLASH_ATTN_EXT on the AI cores (plan.md, M2e design): one step; the rows (query head x token x stream) are split
// statically over the tiles. On the K3 the VLEN-1024 kernels of flagos-spacemit-rvv-attn.cpp, elsewhere (and with
// FLAGOS_SPACEMIT_TEST_REFERENCE) a portable reference following ggml-cpu.

// exactly the FLASH_ATTN_EXT nodes the kernels handle (buffers are checked by spacemit_find_op)
bool spacemit_flash_attn_supported(const ggml_tensor * op);

bool spacemit_kernel_flash_attn(const spacemit_tile & tile, ggml_tensor * dst);
