#pragma once

// IME2 Q4_0 matrix multiplication pieces, copied from ggml-cpu/spacemit (origin and changes: the .cpp).
// Weights use the IME layout "q4_0 32x256": 32 rows interleaved, K in blocks of 256; activations are quantized to
// int8 with fp16 scales ("hp"). Without GGML_FLAGOS_SPACEMIT_IME2 only the scalar references exist and are used.

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

namespace spacemit_ime {

constexpr size_t q4_0_k_block  = 256;  // K values per IME block
constexpr size_t q4_0_row_tile = 32;   // weight rows interleaved together

constexpr size_t div_round_up(size_t up, size_t down) {
    return (up + down - 1) / down;
}

// copied from rvv_kernels.h: bytes of quantized activations for blk_len values of one row:
// K is split into 32-value subblocks, each [fp16 scale][int8 x 32], then optional fp16 sums and one fp16 scale
constexpr size_t q8_hp_blk_size(size_t blk_len, bool with_blk_sum = false, bool with_blk_scale = false) {
    const size_t subblk_count = div_round_up(blk_len, size_t(32));
    const size_t blk_size     = blk_len * sizeof(int8_t) + subblk_count * sizeof(_Float16) +
                            (with_blk_sum ? subblk_count * sizeof(_Float16) : 0) +
                            (with_blk_scale ? sizeof(_Float16) : 0);
    return blk_size;
}

// bytes of one quantized activation row of length k (a multiple of q4_0_k_block)
constexpr size_t act_row_bytes(size_t k) {
    return k / q4_0_k_block * q8_hp_blk_size(q4_0_k_block, true, true);
}

bool have_ime();

// quantize 1 row, or 4 rows (row stride count_k) interleaved for the 4-row kernel; reference: scalar code
void quantize_row(const float * a, size_t count_k, uint8_t * qa, bool reference);
void quantize_4rows(const float * a, size_t count_k, uint8_t * qa, bool reference);

// c[count_m x count_n] (row stride ldc) for 4 rows when count_m >= 4, else 1 row; returns the rows done.
// count_n is a multiple of q4_0_row_tile; qb points at the first of those weight rows.
size_t gemm(const uint8_t * qa, const uint8_t * qb, float * c, size_t count_m, size_t count_n, size_t k_blocks,
            size_t ldc, bool reference);

// copy into or out of TCM
void copy(void * dst, const void * src, size_t size);

// Q4_0 GGUF bytes <-> layout q4_0 32x256 in t->data; repack returns 0 on success
int  repack_q4_0(ggml_tensor * t, const void * data, size_t size);
void unpack_q4_0(const ggml_tensor * t, void * data);

}  // namespace spacemit_ime
