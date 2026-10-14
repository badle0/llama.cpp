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
// K, V and the mask row (with one token per call, the query heads of one KV head) go 4 or 2 at a time through it,
// else one at a time
void forward_one_chunk(const ggml_tensor * dst, int64_t ir0, int64_t ir1, void * tcm, size_t tcm_size);

// the same rows in tiles of up to 64 rows of one head (calls of 64 tokens or more), in scratch: float-aligned, at least
// tiled_scratch_bytes(DK, DV) bytes (160 KiB at DK = DV = 128), e.g. the tile's TCM. Each row stops at its last
// unmasked column of every 64-position KV tile
size_t tiled_scratch_bytes(int64_t DK, int64_t DV);
void   forward_tiled(const ggml_tensor * dst, int64_t ir0, int64_t ir1, void * scratch, size_t scratch_size);

// M2e.3 round 2, candidate (a) (plan.md "M2e.3 design"): one-token attention with the KV positions split over the tiles
// in blocks dealt round-robin by flagos-spacemit-attn.cpp, which also merges the tiles' partial results. Each tile
// stages its blocks' whole cells (every KV head) in its TCM and keeps per query head a running maximum, sum and F32 V
// sum. Needs one token and one stream, F16 K and V whose cells hold every KV head (llama.cpp's cache), no ALiBi and no
// softcap, DK and DV as supported_shape. A tile calls split_begin, then for each of its blocks split_stage and, when
// that returns true, split_compute, then split_end, all with the same tcm.
// positions per block that fit tcm_size: a multiple of 8 up to 64, 0 when fewer than 8 fit
int64_t split_block_positions(const ggml_tensor * dst, size_t tcm_size);
void    split_begin(const ggml_tensor * dst, void * tcm, size_t tcm_size);
// stages block b; false, with nothing copied, when the mask hides every position of it
bool    split_stage(const ggml_tensor * dst, int64_t b, void * tcm, size_t tcm_size);
void    split_compute(const ggml_tensor * dst, int64_t b, void * tcm, size_t tcm_size);
// the tile's partial result per query head h at partial + h * (DV + 2): the maximum score, the sum of exp(score -
// maximum), then DV floats, V summed with those weights; -inf, 0 and zeros when it saw no visible position
void    split_end(const ggml_tensor * dst, void * tcm, size_t tcm_size, float * partial);

}  // namespace spacemit_rvv_attn

#endif  // GGML_FLAGOS_SPACEMIT_RVV
