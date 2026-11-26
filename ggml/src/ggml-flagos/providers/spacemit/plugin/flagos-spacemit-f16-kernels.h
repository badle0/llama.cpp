#pragma once

// The IME2 and RVV leaves of the FP16 path for dense Q4_0 matmuls (flagos-spacemit-f16.cpp; origin and changes: the
// .cpp). Layouts: A [ks / 8][16 rows][8 values] fp16, B [ks / 8][32 columns][8 values] fp16, a partial tile
// [16 rows][8 or 4 slices][32 columns] fp16. The scalar references live in flagos-spacemit-f16.cpp.
// Only with GGML_FLAGOS_SPACEMIT_IME2, and only on an AI-core tile: the X100 cores have no IME and VLEN 256.

#include <cstdint>

namespace spacemit_f16 {

bool have();  // true when built with the kernels

#if defined(GGML_FLAGOS_SPACEMIT_IME2)
// 16 rows of ks floats (row stride x_stride_floats) as fp16 into a; rows from `rows` (1 to 16) on are zero
void cast_tile(const float * x, int64_t x_stride_floats, int64_t rows, int64_t ks, uint16_t * a);
// one slice of one 32-row q4_0 32x256 tile (ks / 32 blocks of [fp16 d x 32][16 nibble bytes x 32] from w) into b:
// b[g][c][j] = fp16(d_c (q - 8)) for k = 8 g + j
void dequant_tile(const uint8_t * w, int64_t ks, uint16_t * b);
// the 16 x 32 product of a and b over ks with fp32 sums, stored as fp16: rows (1 to 16) rows of cols (1 to 32)
// values, row stride c_stride_elems
void products(int64_t ks, const uint16_t * b, const uint16_t * a, uint16_t * c, int64_t rows, int64_t cols,
              int64_t c_stride_elems);
// a partial tile [16 rows][slices][32 columns]: its slices added in order from +0.0f into rows x cols floats (row
// stride out_stride_bytes)
void reduce_tile(const uint16_t * partials, float * out, int64_t out_stride_bytes, int64_t rows, int64_t cols,
                 int64_t slices);
// the accrued floating-point flags: cleared, and whether overflow (OF) was raised since
void clear_flags();
bool overflow_raised();
#endif

}  // namespace spacemit_f16
