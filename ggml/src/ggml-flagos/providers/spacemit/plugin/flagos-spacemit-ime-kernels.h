#pragma once

// IME2 matrix multiplication pieces, copied from ggml-cpu/spacemit (origin and changes: the .cpp). Three weight
// layouts, all with 32 rows interleaved:
// - "q4_0 32x256" (Q4_0): K in blocks of 256; activations quantized to int8 with fp16 scales per 32 values and a
//   block scale ("hp"). Lossless: a byte permutation of the GGUF blocks.
// - "q8_0 32x32" (Q8_0, Q4_1, Q6_K, Q5_0, the F32 router): K in blocks of 32, each [fp16 scale x 32][int8 x 32 x 32]
//   (34 bytes per row and block); activations quantized to int8 with an fp32 scale and the negated sum per 32 values
//   ("i8"). From Q8_0 lossless (a byte permutation); from Q4_1, Q6_K and F32 quantized at load, per 32 values scale =
//   max|w| / 127 (stored in fp16), so each weight moves by about half a Q8_0 step at most (20 and 26.25 bytes per 32
//   weights become 34; F32's 128 become 34); from Q5_0 exactly, its scale kept and qs = q - 16 (22 bytes per 32 weights
//   become 34).
// - "q4_1 min 32x256" (3-D Q4_1 experts): each 32-row tile is the q4_0 32x256
//   form of the Q4_1 d and nibbles, then the tile's fp16 minimums, [K/32 block][32 rows]. A weight d q + m is
//   d (q - 8) + (m + 8 d): the q4_0 kernels give the first part, q4_1_min_gemm adds the second. Lossless, 20 bytes per
//   32 weights as Q4_1; the activations are the q4_0 layout's.
// - "q4_0s 32x256" (3-D Q4_0 experts unless FLAGOS_SPACEMIT_EXPERT_Q4_0=plain): the q4_0 32x256 bytes with every
//   nibble byte ^ 0x88, so a nibble holds q - 8 as a signed 4-bit value (SpacemiT's fork). Lossless, same size. Its
//   kernels take the activations in groups of 8, 4 or 1 rows (the fork's packs) and need no sum(a) correction.
// A 3-D weight (MUL_MAT_ID's experts) is stored as the 2-D weight of its ggml_nrows rows in any layout.
// Without GGML_FLAGOS_SPACEMIT_IME2 only the scalar references exist and are used.

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

namespace spacemit_ime {

constexpr size_t row_tile     = 32;   // weight rows interleaved together, in every layout
constexpr size_t q4_0_k_block = 256;  // K values per IME block, layout q4_0 32x256
constexpr size_t q8_0_k_block = 32;   // K values per IME block, layout q8_0 32x32
constexpr size_t i8_k_block   = 32;   // K values per block of the i8 activations, which the 32x32 layout uses
static_assert(q8_0_k_block == i8_k_block, "the 32x32 layout takes the i8 activations");

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

// copied from rvv_kernels.h (SpacemiT's fork, mtmd-backend b847bf396, 43-52): bytes per 256 values of the q4_0s
// layout's 4-row and 8-row activation groups (payload, shared scales, sums and average scale, padded), and the
// alignment of a group's start
constexpr size_t q8_hp_m4al_blk_size =
    (8 * 4 * 32 + 8 * 4 * sizeof(_Float16) + 4 * 8 * sizeof(_Float16) + 4 * sizeof(_Float16) + 63) / 64 * 64;
constexpr size_t q8_hp_m8_blk_size = (8 * 8 * 32 + 8 * sizeof(_Float16) + sizeof(_Float16) + 127) / 128 * 128;
static_assert(q8_hp_m4al_blk_size == 1216, "m4al group layout");
static_assert(q8_hp_m8_blk_size == 2176, "m8 group layout");
constexpr size_t q8_hp_al_align = 128;

// bytes of one quantized activation row and of one repacked weight row of length k (a multiple of the layout's K
// block). Q4_0 keeps its size (18 bytes per 32 values); Q8_0 keeps its 34, which a Q4_1 (20) or Q6_K weight (26.25)
// grows to
constexpr size_t q4_0_act_row_bytes(size_t k) {
    return k / q4_0_k_block * q8_hp_blk_size(q4_0_k_block, true, true);
}
constexpr size_t q4_0_weight_row_bytes(size_t k) {
    return k / 32 * (sizeof(uint16_t) + 16);
}
// the q4_1 min layout: the q4_0 part (18 bytes per 32 weights) and the minimum (2)
constexpr size_t q4_1_min_weight_row_bytes(size_t k) {
    return q4_0_weight_row_bytes(k) + k / 32 * sizeof(uint16_t);
}
constexpr size_t i8_act_row_bytes(size_t k) {
    return k / i8_k_block * q8_blk_size(i8_k_block, true);
}
constexpr size_t q8_0_weight_row_bytes(size_t k) {
    return k / q8_0_k_block * (sizeof(uint16_t) + q8_0_k_block);
}

bool have_ime();

// quantize 1 row, or 4 rows (row stride count_k) interleaved for the 4-row kernel, for the q4_0 layout or for the
// 32x32 layout (i8); reference: scalar code.
// gemm, per layout: c[count_m x count_n] (row stride ldc) for 4 rows when count_m >= 4, else 1 row; returns the rows
// done. count_n is a multiple of row_tile (q8_0_gemm: at most row_tile for 4 rows, plan.md R13); qb points at the
// first of those weight rows.
void   q4_0_quantize_row(const float * a, size_t count_k, uint8_t * qa, bool reference);
void   q4_0_quantize_4rows(const float * a, size_t count_k, uint8_t * qa, bool reference);
size_t q4_0_gemm(const uint8_t * qa, const uint8_t * qb, float * c, size_t count_m, size_t count_n, size_t k_blocks,
                 size_t ldc, bool reference);
void   i8_quantize_row(const float * a, size_t count_k, uint8_t * qa, bool reference);
void   i8_quantize_4rows(const float * a, size_t count_k, uint8_t * qa, bool reference);
size_t q8_0_gemm(const uint8_t * qa, const uint8_t * qb, float * c, size_t count_m, size_t count_n, size_t k_blocks,
                 size_t ldc, bool reference);
// the q4_1 min layout: the q4_0 activations (q4_0_quantize_row, q4_0_quantize_4rows) and q4_0_gemm's contract, count_n
// a multiple of row_tile
size_t q4_1_min_gemm(const uint8_t * qa, const uint8_t * qb, float * c, size_t count_m, size_t count_n,
                     size_t k_blocks, size_t ldc, bool reference);
// the q4_0s layout: 8 or 4 rows (row stride count_k) packed as one group, q8_hp_m8_blk_size or q8_hp_m4al_blk_size
// bytes per 256 values; one row is q4_0_quantize_row's pack. q4_0s_gemm multiplies one group of rows (8, 4 or 1, the
// pack it was given) by count_n columns (a multiple of row_tile) into c (row stride ldc) and returns rows
void   q4_0s_quantize_8rows(const float * a, size_t count_k, uint8_t * qa, bool reference);
void   q4_0s_quantize_4rows(const float * a, size_t count_k, uint8_t * qa, bool reference);
size_t q4_0s_gemm(const uint8_t * qa, const uint8_t * qb, float * c, size_t rows, size_t count_n, size_t k_blocks,
                  size_t ldc, bool reference);
// bytes per 256 values of a group of rows (8, 4 or 1) in the q4_0s layout's packs
constexpr size_t q4_0s_group_blk_bytes(size_t rows) {
    return rows == 8 ? q8_hp_m8_blk_size : rows == 4 ? q8_hp_m4al_blk_size : q8_hp_blk_size(q4_0_k_block, true, true);
}

// copy into or out of TCM
void copy(void * dst, const void * src, size_t size);

// GGUF bytes <-> IME layout in t->data; repack returns 0 on success. unpack_q4_0 and unpack_q8_0 restore the GGUF
// bytes; unpack_q4_1_q8_0 and unpack_q6_k return the nearest Q4_1 and Q6_K to the stored Q8_0 values (approximate). repack_q4_1_q8_0 and
// repack_q6_k write nrows * q8_0_weight_row_bytes(ne[0]) bytes, more than ggml_nbytes(t)
int  repack_q4_0(ggml_tensor * t, const void * data, size_t size);
void unpack_q4_0(const ggml_tensor * t, void * data);
int  repack_q4_1_q8_0(ggml_tensor * t, const void * data, size_t size);
void unpack_q4_1_q8_0(const ggml_tensor * t, void * data);
int  repack_q8_0(ggml_tensor * t, const void * data, size_t size);
void unpack_q8_0(const ggml_tensor * t, void * data);
int  repack_q6_k(ggml_tensor * t, const void * data, size_t size);
void unpack_q6_k(const ggml_tensor * t, void * data);
// Q5_0 as Q8_0, exactly both ways (d (q - 16) is Q8_0's d qs with qs = q - 16): repack writes
// nrows * q8_0_weight_row_bytes(ne[0]) bytes, more than ggml_nbytes(t); unpack restores the GGUF bytes
int  repack_q5_0_q8_0(ggml_tensor * t, const void * data, size_t size);
void unpack_q5_0_q8_0(const ggml_tensor * t, void * data);
// Q4_1 in the q4_1 min layout, exactly both ways: repack writes ggml_nbytes(t) bytes; unpack restores the GGUF bytes
int  repack_q4_1_min(ggml_tensor * t, const void * data, size_t size);
void unpack_q4_1_min(const ggml_tensor * t, void * data);
// Q4_0 in the q4_0s layout, exactly both ways: repack_q4_0 and the nibble bytes ^ 0x88; unpack restores the GGUF bytes
int  repack_q4_0s(ggml_tensor * t, const void * data, size_t size);
void unpack_q4_0s(const ggml_tensor * t, void * data);
// F32 quantized to Q8_0 by ggml's quantizer in the q8_0 32x32 layout: repack writes
// nrows * q8_0_weight_row_bytes(ne[0]) bytes, less than ggml_nbytes(t); unpack returns the stored values d * qs as
// floats (approximate)
int  repack_f32_q8_0(ggml_tensor * t, const void * data, size_t size);
void unpack_f32_q8_0(const ggml_tensor * t, void * data);

}  // namespace spacemit_ime
