#pragma once

// Flash attention (FLASH_ATTN_EXT) for VLEN 1024, copied from the in-tree IME path (origin and changes: the .cpp);
// built only for the K3 (GGML_FLAGOS_SPACEMIT_RVV). flagos-spacemit-attn.cpp decides when they apply.
// Both entry points need, mostly unchecked: VLEN 1024 on the calling core (__riscv_vlenb() == 128); q F32 and K, V
// F16, each with contiguous rows, DK equal to q's row length; dst F32 and contiguous as ggml_flash_attn_ext makes it;
// the mask, if any, F16 with contiguous rows covering n_kv and q's rows; the sinks, if any, F32, one per query head.

#include "ggml.h"

#include <cstddef>
#include <cstdint>

#if defined(GGML_FLAGOS_SPACEMIT_RVV)

namespace spacemit_rvv_attn {

// the head sizes the kernels handle: DK and DV in 1..128 (one register group at VLEN 1024)
bool supported_shape(int64_t DK, int64_t DV);

// rows [ir0, ir1) of dst = FLASH_ATTN_EXT(q, k, v, mask, sinks), a row being one query (token i1, head i2, stream i3)
// in ggml-cpu's order, ir = (i3 * ne_q2 + i2) * ne_q1 + i1. With tcm (not null), and without ALiBi, rows that share
// K, V and the mask row (at decode the query heads of one KV head) go 4 or 2 at a time through it, else one at a time
void forward_one_chunk(const ggml_tensor * dst, int64_t ir0, int64_t ir1, void * tcm, size_t tcm_size);

// the same rows in tiles of up to 64 rows of one head (prefill), in scratch: float-aligned, at least
// tiled_scratch_bytes(DK, DV) bytes (160 KiB at DK = DV = 128), e.g. the tile's TCM
size_t tiled_scratch_bytes(int64_t DK, int64_t DV);
void   forward_tiled(const ggml_tensor * dst, int64_t ir0, int64_t ir1, void * scratch, size_t scratch_size);

// test-only: ggml-spacemit's variant of forward_tiled, which skips each query row's masked tail of every KV tile;
// same requirements and results (plan.md M2e.3, L2: FLAGOS_SPACEMIT_ATTN=trim)
void forward_tiled_trim(const ggml_tensor * dst, int64_t ir0, int64_t ir1, void * scratch, size_t scratch_size);

}  // namespace spacemit_rvv_attn

#endif  // GGML_FLAGOS_SPACEMIT_RVV
