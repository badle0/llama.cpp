#pragma once

// IME2 matrix multiplication pieces, copied from ggml-cpu/spacemit (origin and changes: the .cpp). Three weight
// layouts, all with 32 rows interleaved:
// - "q4_0 32x256" (Q4_0): K in blocks of 256; activations quantized to int8 with fp16 scales per 32 values and a
//   block scale ("hp"). Lossless: a byte permutation of the GGUF blocks.
// - "q4_1 32x32" (Q4_1): K in blocks of 32, each row's block an fp16 scale d, an integer zero point zp and 32 4-bit
//   values q, so a weight is d * (q - zp); activations quantized to int8 with an fp32 scale and the negated sum per 32
//   values ("i8"). Lossy: Q4_1's fp16 minimum m (a weight is d * q + m) becomes zp = clamp(round(-m / d), 0, 15).
// - "q8_0 32x32" (Q8_0, Q6_K): K in blocks of 32, each [fp16 scale x 32][int8 x 32 x 32] (34 bytes per row and block);
//   activations "i8" as for q4_1 32x32. From Q8_0 lossless (a byte permutation); from Q6_K requantized at load, per 32
//   values scale = max|w| / 127 (stored in fp16), so each weight moves by about half a Q8_0 step at most (26.25 bytes
//   per 32 weights become 34).
// Without GGML_FLAGOS_SPACEMIT_IME2 only the scalar references exist and are used.

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

namespace spacemit_ime {

constexpr size_t row_tile     = 32;   // weight rows interleaved together, in every layout
constexpr size_t q4_0_k_block = 256;  // K values per IME block, layout q4_0 32x256
constexpr size_t q4_1_k_block = 32;   // K values per IME block, layout q4_1 32x32
constexpr size_t q8_0_k_block = 32;   // K values per IME block, layout q8_0 32x32
constexpr size_t i8_k_block   = 32;   // K values per block of the i8 activations, which both 32x32 layouts use
static_assert(q4_1_k_block == i8_k_block && q8_0_k_block == i8_k_block, "the 32x32 layouts share the i8 activations");

constexpr size_t div_round_up(size_t up, size_t down) {
    return (up + down - 1) / down;
}

// copied from rvv_kernels.h: bytes of quantized activations for blk_len values of one row: [f32 scale], the optional
// [s16 negated sum], [int8 x blk_len]
constexpr size_t q8_blk_size(size_t blk_len, bool with_blk_sum = false) {
    const size_t blk_size = sizeof(float) + blk_len * sizeof(int8_t) + (with_blk_sum ? sizeof(int16_t) : 0);
    return blk_size;
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

// bytes of one quantized activation row and of one repacked weight row of length k (a multiple of the layout's K
// block). Q4_0 keeps its size (18 bytes per 32 values); Q4_1 needs 19 of its 20 (an 8-bit zero point, not an fp16
// minimum); Q8_0 keeps its 34, which a Q6_K weight (26.25) grows to
constexpr size_t q4_0_act_row_bytes(size_t k) {
    return k / q4_0_k_block * q8_hp_blk_size(q4_0_k_block, true, true);
}
constexpr size_t q4_0_weight_row_bytes(size_t k) {
    return k / 32 * (sizeof(uint16_t) + 16);
}
constexpr size_t i8_act_row_bytes(size_t k) {
    return k / i8_k_block * q8_blk_size(i8_k_block, true);
}
constexpr size_t q4_1_weight_row_bytes(size_t k) {
    return k / q4_1_k_block * (sizeof(uint16_t) + sizeof(uint8_t) + q4_1_k_block / 2);
}
constexpr size_t q8_0_weight_row_bytes(size_t k) {
    return k / q8_0_k_block * (sizeof(uint16_t) + q8_0_k_block);
}

bool have_ime();

// quantize 1 row, or 4 rows (row stride count_k) interleaved for the 4-row kernel, for the q4_0 layout or for both
// 32x32 layouts (i8); reference: scalar code.
// gemm, per layout: c[count_m x count_n] (row stride ldc) for 4 rows when count_m >= 4, else 1 row; returns the rows
// done. count_n is a multiple of row_tile (q8_0_gemm: at most row_tile for 4 rows, plan.md R13); qb points at the
// first of those weight rows.
void   q4_0_quantize_row(const float * a, size_t count_k, uint8_t * qa, bool reference);
void   q4_0_quantize_4rows(const float * a, size_t count_k, uint8_t * qa, bool reference);
size_t q4_0_gemm(const uint8_t * qa, const uint8_t * qb, float * c, size_t count_m, size_t count_n, size_t k_blocks,
                 size_t ldc, bool reference);
void   i8_quantize_row(const float * a, size_t count_k, uint8_t * qa, bool reference);
void   i8_quantize_4rows(const float * a, size_t count_k, uint8_t * qa, bool reference);
size_t q4_1_gemm(const uint8_t * qa, const uint8_t * qb, float * c, size_t count_m, size_t count_n, size_t k_blocks,
                 size_t ldc, bool reference);
size_t q8_0_gemm(const uint8_t * qa, const uint8_t * qb, float * c, size_t count_m, size_t count_n, size_t k_blocks,
                 size_t ldc, bool reference);

// copy into or out of TCM
void copy(void * dst, const void * src, size_t size);

// GGUF bytes <-> IME layout in t->data; repack returns 0 on success. unpack_q4_0 and unpack_q8_0 restore the GGUF
// bytes; unpack_q4_1 returns the converted tensor (m = -zp * d), the weights the kernels compute with; unpack_q6_k
// returns the nearest Q6_K to the stored Q8_0 values (approximate). repack_q6_k writes
// nrows * q8_0_weight_row_bytes(ne[0]) bytes, more than ggml_nbytes(t)
int  repack_q4_0(ggml_tensor * t, const void * data, size_t size);
void unpack_q4_0(const ggml_tensor * t, void * data);
int  repack_q4_1(ggml_tensor * t, const void * data, size_t size);
void unpack_q4_1(const ggml_tensor * t, void * data);
int  repack_q8_0(ggml_tensor * t, const void * data, size_t size);
void unpack_q8_0(const ggml_tensor * t, void * data);
int  repack_q6_k(ggml_tensor * t, const void * data, size_t size);
void unpack_q6_k(const ggml_tensor * t, void * data);

}  // namespace spacemit_ime
