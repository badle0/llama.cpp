#pragma once

// IME2 matrix multiplication pieces, copied from ggml-cpu/spacemit and from SpacemiT's fork (origin and changes: the
// .cpp). Three weight layouts, all with 32 rows interleaved:
// - "q4_0 32x256" (Q4_0): K in blocks of 256; activations quantized to int8 with fp16 scales per 32 values and a
//   block scale ("hp"). Lossless: a byte permutation of the GGUF blocks.
// - "q8_0 32x32" (Q8_0, Q4_1, Q6_K, Q5_0): K in blocks of 32, each [fp16 scale x 32][int8 x 32 x 32] (34 bytes per
//   row and block); activations quantized to int8 with an fp32 scale and the negated sum per 32 values ("i8"). From
//   Q8_0 lossless (a byte permutation); from Q4_1 and Q6_K requantized at load, per 32 values scale = max|w| / 127
//   (stored in fp16), so each weight moves by about half a Q8_0 step at most (20 and 26.25 bytes per 32 weights become
//   34); from Q5_0 exactly, its scale kept and qs = q - 16 (22 bytes per 32 weights become 34).
// - "q4_1 zp4 32x256" (3-D Q4_1 experts under FLAGOS_SPACEMIT_EXPERT_Q4_1=zp4): the q4_0 32x256 nibbles and scales
//   plus one unsigned 4-bit zero point per row and 32 values, which replaces the 2-byte minimum (19 bytes per 32
//   weights, not 20). A stored weight is d * (q - zp), q the GGUF nibble, so the loss is the rounding of -m/d to a
//   whole step. The activations are the "hp" ones; the 4-row kernel wants its own "aligned" pack (q8_hp_al_*).
// A 3-D weight (MUL_MAT_ID's experts) is stored as the 2-D weight of its ggml_nrows rows in any layout.
// Without GGML_FLAGOS_SPACEMIT_IME2 only the scalar references exist and are used.

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

namespace spacemit_ime {

constexpr size_t row_tile     = 32;   // weight rows interleaved together, in every layout
constexpr size_t q4_0_k_block = 256;  // K values per IME block, layout q4_0 32x256
constexpr size_t q4_1_k_block = 256;  // K values per IME block, layout q4_1 zp4 32x256
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

// copied from rvv_kernels.h (43-52): the "aligned" 4-row pack of the hp activations, one K256 group of 4 rows padded
// to 64 bytes (the 1-row form's 4 x 290 bytes become 1216), and the alignment its base wants
constexpr size_t q8_hp_m4al_blk_size =
    (8 * 4 * 32 + 8 * 4 * sizeof(_Float16) + 4 * 8 * sizeof(_Float16) + 4 * sizeof(_Float16) + 63) / 64 * 64;
constexpr size_t q8_hp_al_row_blk_size = q8_hp_m4al_blk_size / 4;
constexpr size_t q8_hp_al_align        = 128;
static_assert(q8_hp_m4al_blk_size == 1216, "m4al group layout");

// bytes of one quantized activation row and of one repacked weight row of length k (a multiple of the layout's K
// block). Q4_0 keeps its size (18 bytes per 32 values); Q8_0 keeps its 34, which a Q4_1 (20) or Q6_K weight (26.25)
// grows to; the q4_1 zp4 layout takes 19 per 32 weights (the 2-byte minimum becomes a 1-byte zero point)
constexpr size_t q4_0_act_row_bytes(size_t k) {
    return k / q4_0_k_block * q8_hp_blk_size(q4_0_k_block, true, true);
}
constexpr size_t q4_0_weight_row_bytes(size_t k) {
    return k / 32 * (sizeof(uint16_t) + 16);
}
// the q4_1 zp4 layout: one row of hp activations (the q4_0 form), one row inside a 4-row aligned group, one weight row
constexpr size_t q4_1_act_row_bytes(size_t k) {
    return k / q4_1_k_block * q8_hp_blk_size(q4_1_k_block, true, true);
}
constexpr size_t q4_1_act_4row_bytes(size_t k) {
    return k / q4_1_k_block * q8_hp_al_row_blk_size;
}
constexpr size_t q4_1_weight_row_bytes(size_t k) {
    return k / 32 * (sizeof(uint16_t) + sizeof(uint8_t) + 16);
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
// the q4_1 zp4 layout. One row takes the q4_0 hp pack (q4_1_quantize_row == q4_0_quantize_row's pack), 4 rows the
// aligned pack of q8_hp_m4al_blk_size bytes a group, whose base wants q8_hp_al_align. count_n is row_tile for 4 rows
void   q4_1_quantize_row(const float * a, size_t count_k, uint8_t * qa, bool reference);
void   q4_1_quantize_4rows(const float * a, size_t count_k, uint8_t * qa, bool reference);
size_t q4_1_gemm(const uint8_t * qa, const uint8_t * qb, float * c, size_t count_m, size_t count_n, size_t k_blocks,
                 size_t ldc, bool reference);

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
// Q4_1 in the q4_1 zp4 layout: repack writes nrows * q4_1_weight_row_bytes(ne[0]) bytes, fewer than ggml_nbytes(t).
// unpack returns the Q4_1 bytes of the stored values: the same d and nibbles, m = fp16(-zp * d) (approximate, since
// the repack rounds -m/d to a whole step)
int  repack_q4_1(ggml_tensor * t, const void * data, size_t size);
void unpack_q4_1(const ggml_tensor * t, void * data);

}  // namespace spacemit_ime
