// IME2 matrix multiplication pieces for the SpacemiT provider (plan.md §2.5, D6: copy per milestone): Q4_0 in the
// layout q4_0 32x256 (M2b) and Q4_1 in the layout q4_1 32x32 (M2c).
//
// Copied from ggml/src/ggml-cpu/spacemit @ ba360ef; identical in ggml-spacemit (spacemit-com/llama.cpp 4e782bc and
// mtmd-backend 64316cd) and in upstream llama.cpp master (Q4_0: 2026-10-08; Q4_1: 2026-10-10):
//   ime2_kernels.cpp  Q4_0: gemm_kernel_i8i4_hp_mrow_ref (324-436), gemm_kernel_i8i4_hp_m1 (2883-3005),
//                     gemm_kernel_i8i4_hp_m4 (3360-3741)
//                     Q4_1: gemm_kernel_i8i4_mrow_ref (217-322), gemm_kernel_i8i4_m1 (2430-2881),
//                     gemm_kernel_i8i4_m4 (3007-3358)
//   rvv_kernels.cpp   memcpy1d (1013-1113)
//                     Q4_0: quantize_a_nrow_i8_hp_ref (1739-1795), quantize_a_row_i8_hp (1989-2098),
//                     quantize_a_4row_i8_hp (2100-2303)
//                     Q4_1: quantize_a_nrow_i8_ref (1708-1737), quantize_a_row_i8 (1848-1914),
//                     quantize_a_4row_i8 (1916-1987)
//   repack.cpp        QK_0, block (40-53), block_q4_0x32, block_q4_0x32x256 (76-82), make_block_q4_0x32 (292-320),
//                     repack_q4_0_to_q4_0_256_32_bl_ref (592-628)
//                     block_with_zp (55-59), block_q4_1x32 (77), make_block_q4_1x32 (322-355),
//                     repack_q4_1_to_q4_1_32_bl_ref (779-812)
// Bodies are unchanged except, marked "flagos": two zero guards in quantize_a_nrow_i8_hp_ref and one in
// quantize_a_nrow_i8_ref; the last K blocks in gemm_kernel_i8i4_mrow_ref (dropped unless their count was a multiple of
// 16, which it asserted); unaligned fp32 scales read and written with memcpy in those two references; and in
// gemm_kernel_i8i4_m1 the exact zero-point branch upstream disables with #if 0, instead of its 16-bit one. Everything
// below the copies (dispatch, unpack) is new. When upstream changes these functions, take the fix by hand (plan.md R11).
//
// The IME and RVV code is built only with GGML_FLAGOS_SPACEMIT_IME2 (riscv64, GCC >= 15); the scalar references
// (*_ref, plain C++) are built everywhere, so hosts without the AI cores can run the whole path.

#define GGML_COMMON_DECL_CPP
#include "../../../ggml-common.h"
#include "../../../ggml-impl.h"
#include "flagos-spacemit-ime-kernels.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

#if defined(GGML_FLAGOS_SPACEMIT_IME2)
#    include <riscv_vector.h>
#endif

#if defined(__GNUC__)
#    pragma GCC diagnostic ignored "-Woverlength-strings"
#    pragma GCC diagnostic ignored "-Wcast-qual"
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

namespace spacemit_ime {

//
// copied: weight layout and repack (repack.cpp)
//

template <int K> constexpr int QK_0() {
    if constexpr (K == 4) {
        return QK4_0;
    }
    if constexpr (K == 8) {
        return QK8_0;
    }
    return -1;
}

template <int K, int N> struct block {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    uint8_t   qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

using block_q4_0x32 = block<4, 32>;

struct block_q4_0x32x256 {
    block_q4_0x32 blocks[8];  // [f16 * 32 | i4 * 32 * 32] * 8
};

static block_q4_0x32 make_block_q4_0x32(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x32 out;
    assert(QK4_0 / blck_size_interleave == 1);
    GGML_UNUSED(blck_size_interleave);

    for (int i = 0; i < 32; i++) {
        out.d[i] = in[i].d;
    }

    for (int i = 0; i < 32; i++) {
        // [0, 15], in.d & 0x0F
        for (int j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b1] .........  [b14 b15]
            out.qs[i * QK4_0 / 2 + j] = (in[i].qs[j * 2] & 0x0F) | ((in[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
    }

    for (int i = 0; i < 32; i++) {
        // [16, 31], in.d & 0xF0
        for (int j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b17] ......... [b30 b31]
            out.qs[i * QK4_0 / 2 + QK4_0 / 4 + j] = ((in[i].qs[j * 2] & 0xF0) >> 4) | (in[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    return out;
}

static int repack_q4_0_to_q4_0_256_32_bl_ref(ggml_tensor *              t,
                                             int                        interleave_block,
                                             const void * GGML_RESTRICT data,
                                             size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 32);  // unused

    constexpr int nrows_interleaved = 32;

    block_q4_0x32x256 * dst = (block_q4_0x32x256 *) t->data;
    const block_q4_0 *  src = (const block_q4_0 *) data;
    block_q4_0          dst_tmp[32];
    int                 nrow    = ggml_nrows(t);
    int                 nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));
    GGML_ASSERT(nblocks % 8 == 0);  // for 256-block interleaving
    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x += 8) {
            for (int j = 0; j < 8; j++) {
                for (int i = 0; i < nrows_interleaved; i++) {
                    dst_tmp[i] = src[x + j + i * nblocks];
                }
                dst->blocks[j] = make_block_q4_0x32(dst_tmp, interleave_block);
            }
            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

template <int K, int N> struct block_with_zp {
    ggml_half d[N];                         // deltas for N qK_1 blocks
    uint8_t   zp[N];                        // zero points for N qK_1 blocks
    uint8_t   qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_1 blocks
};

using block_q4_1x32 = block_with_zp<4, 32>;

static block_q4_1x32 make_block_q4_1x32(block_q4_1 * in, unsigned int blck_size_interleave) {
    block_q4_1x32 out;
    GGML_ASSERT(QK4_1 / blck_size_interleave == 1);
    GGML_UNUSED(blck_size_interleave);

    for (int i = 0; i < 32; i++) {
        float d   = GGML_FP16_TO_FP32(in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        float m   = GGML_FP16_TO_FP32(in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
        float mid = -std::nearbyintf(m / d);
        mid       = std::min(15.0f, std::max(0.0f, mid));
        out.d[i]  = GGML_FP32_TO_FP16(d);
        out.zp[i] = static_cast<uint8_t>(mid);
    }

    for (int i = 0; i < 32; i++) {
        // [0, 15], in.d & 0x0F
        for (int j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b1] ......... [b14 b15]
            out.qs[i * QK4_1 / 2 + j] = (in[i].qs[j * 2] & 0x0F) | ((in[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
    }

    for (int i = 0; i < 32; i++) {
        // [16, 31], in.d & 0xF0
        for (int j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            out.qs[i * QK4_1 / 2 + QK4_1 / 4 + j] = ((in[i].qs[j * 2] & 0xF0) >> 4) | (in[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    return out;
}

static int repack_q4_1_to_q4_1_32_bl_ref(ggml_tensor *              t,
                                         int                        interleave_block,
                                         const void * GGML_RESTRICT data,
                                         size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_block == 32);  // unused

    constexpr int nrows_interleaved = 32;

    block_q4_1x32 *    dst = (block_q4_1x32 *) t->data;
    const block_q4_1 * src = (const block_q4_1 *) data;
    block_q4_1         dst_tmp[32];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_1));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_1x32(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

//
// copied: scalar references (ime2_kernels.cpp, rvv_kernels.cpp)
//

template <size_t MB_ROWS, size_t NB_COLS>
void gemm_kernel_i8i4_hp_mrow_ref(size_t          blk_len,
                                  const uint8_t * quant_a_ptr,
                                  const uint8_t * quant_b_data,
                                  const uint8_t * quant_b_zp,
                                  float *         c_ptr,
                                  size_t          count_m,
                                  size_t          count_n,
                                  size_t          k_blks,
                                  size_t          ldc) {
    constexpr size_t k_subblks_per_superblk = 8;

    struct block_q4_0x32_layout {
        _Float16 d[NB_COLS];
        uint8_t  qs[16 * NB_COLS];
    };

    GGML_ASSERT(blk_len == 256);

    const size_t b_superblk_stride = sizeof(block_q4_0x32_layout) * k_subblks_per_superblk +
                                     (quant_b_zp ? NB_COLS * k_subblks_per_superblk * sizeof(uint8_t) : 0);
    const size_t b_tile_stride = k_blks * b_superblk_stride;

    const size_t a_nrow_block_stride = q8_hp_blk_size(blk_len, true, true) * MB_ROWS;
    const size_t a_subblk_stride     = q8_hp_blk_size(32, false, false) * MB_ROWS;

    float output[MB_ROWS * NB_COLS] = { 0 };
    for (size_t ni = 0; ni < count_n; ni += NB_COLS, c_ptr += NB_COLS) {
        size_t          nb_real     = std::min<size_t>(NB_COLS, count_n - ni);
        const uint8_t * b_tile_base = quant_b_data + (ni / NB_COLS) * b_tile_stride;
        int8_t *        a_data      = (int8_t *) quant_a_ptr;

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < NB_COLS; ci++) {
                output[ci + mi * NB_COLS] = 0.0f;
            }
        }

        for (size_t ki = 0; ki < k_blks; ki++, a_data += a_nrow_block_stride) {
            _Float16 output_f16[MB_ROWS * NB_COLS] = { 0 };

            const uint8_t *              b_superblk_ptr = b_tile_base + ki * b_superblk_stride;
            const block_q4_0x32_layout * b_blocks = reinterpret_cast<const block_q4_0x32_layout *>(b_superblk_ptr);
            const uint8_t *              b_zps =
                quant_b_zp ? b_superblk_ptr + sizeof(block_q4_0x32_layout) * k_subblks_per_superblk : nullptr;

            _Float16 * a_sum_row       = (_Float16 *) (a_data + a_subblk_stride * k_subblks_per_superblk);
            _Float16 * a_scale_avg_row = (_Float16 *) (a_data + a_nrow_block_stride - sizeof(_Float16) * MB_ROWS);
            _Float16   scale_factor    = a_scale_avg_row[0];

            for (size_t ksi = 0; ksi < k_subblks_per_superblk; ++ksi) {
                const _Float16 * a_scale_row = reinterpret_cast<const _Float16 *>(a_data + a_subblk_stride * ksi);
                int8_t *         a_subblk    = a_data + a_subblk_stride * ksi + MB_ROWS * sizeof(_Float16);
                const _Float16   a_scale     = a_scale_row[0];
                const block_q4_0x32_layout & b_block = b_blocks[ksi];

                for (size_t mi = 0; mi < MB_ROWS; mi++) {
                    for (size_t ci = 0; ci < NB_COLS; ci++) {
                        const uint8_t * b_qs    = b_block.qs + ci * 16;
                        _Float16        b_scale = b_block.d[ci] * a_scale;

                        int16_t acc = 0;
                        for (size_t bi = 0; bi < 16; bi++) {
                            uint8_t b  = b_qs[bi];
                            int8_t  b0 = static_cast<int8_t>(b & 0x0F);
                            int8_t  b1 = static_cast<int8_t>((b & 0xF0) >> 4);

                            acc += static_cast<int16_t>(a_subblk[mi * 32 + 2 * bi]) * static_cast<int16_t>(b0) +
                                   static_cast<int16_t>(a_subblk[mi * 32 + 2 * bi + 1]) * static_cast<int16_t>(b1);
                        }

                        const _Float16 scaled_acc = static_cast<_Float16>(acc) * b_scale;
                        output_f16[ci + mi * NB_COLS] += scaled_acc;
                    }
                }
            }

            for (size_t ksi = 0; ksi < k_subblks_per_superblk; ++ksi) {
                const _Float16 * a_scale_row = reinterpret_cast<const _Float16 *>(a_data + a_subblk_stride * ksi);
                const block_q4_0x32_layout & b_block  = b_blocks[ksi];
                const uint8_t *              b_zp_row = b_zps ? b_zps + ksi * NB_COLS : nullptr;
                const _Float16               a_scale  = a_scale_row[0];

                for (size_t mi = 0; mi < MB_ROWS; mi++) {
                    const _Float16 a_sum = a_sum_row[mi * k_subblks_per_superblk + ksi];
                    for (size_t ci = 0; ci < NB_COLS; ci++) {
                        _Float16 b_scale   = b_block.d[ci] * a_scale;
                        _Float16 a_sum_bzp = a_sum;
                        if (b_zp_row) {
                            a_sum_bzp = a_sum * static_cast<_Float16>(0.125f) * static_cast<_Float16>(b_zp_row[ci]);
                        }

                        const _Float16 scaled_acc = a_sum_bzp * b_scale;
                        output[ci + mi * NB_COLS] += scaled_acc * scale_factor;
                    }
                }
            }

            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                for (size_t ci = 0; ci < NB_COLS; ci++) {
                    auto val = static_cast<float>(output_f16[ci + mi * NB_COLS]) * static_cast<float>(scale_factor);
                    output[ci + mi * NB_COLS] += val;
                }
            }
        }

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < nb_real; ci++) {
                c_ptr[mi * ldc + ci] = output[mi * NB_COLS + ci];
            }
        }
    }
}

template <size_t MB_ROWS>
void quantize_a_nrow_i8_hp_ref(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    const size_t     subblk_count = blk_len / k_subblk_len;

    GGML_ASSERT(blk_len == 256);

    float   scale_temp[8]       = { 0.0f };
    int64_t a_blk_stride        = q8_hp_blk_size(blk_len, true, true);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    int64_t a_subblk_stride     = q8_hp_blk_size(k_subblk_len, false, false) * MB_ROWS;

    for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
        _Float16 * a_sum_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_count);

        float scale_avg = 0.0f;
        for (size_t kk = 0; kk < subblk_count; kk++) {
            float max_abs_a = 0.0f;
            for (size_t row = 0; row < MB_ROWS; row++) {
                for (size_t bk = 0; bk < k_subblk_len; bk++) {
                    max_abs_a = std::max(max_abs_a, std::abs(a_ptr[row * count_k + k + bk + kk * k_subblk_len]));
                }
            }
            scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
            scale_avg += scale_temp[kk];
        }

        scale_avg /= subblk_count;
        float scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;  // flagos: zero guard from the RVV version

        _Float16 * scale_avg_ptr =
            reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_block_stride - sizeof(_Float16) * MB_ROWS);
        scale_avg_ptr[0] = scale_avg;

        for (size_t kk = 0; kk < subblk_count; kk++) {
            uint8_t *  a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
            _Float16 * scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
            int8_t *   quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + sizeof(_Float16) * MB_ROWS);

            scale_a_ptr[0] = static_cast<_Float16>(scale_temp[kk] * scale_factor);

            const float rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;  // flagos: zero guard

            for (size_t row = 0; row < MB_ROWS; row++) {
                int16_t a_sum = 0;
                for (size_t bk = 0; bk < k_subblk_len; bk++) {
                    const int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[row * count_k + k + bk + kk * k_subblk_len] * rep_scale_a),
                                   -128.0f, 127.0f));
                    quant_a_blk[row * k_subblk_len + bk] = quantized;
                    a_sum += quantized;
                }
                a_sum_ptr[row * subblk_count + kk] = static_cast<_Float16>(-a_sum) * static_cast<_Float16>(8.0f);
            }
        }
    }
}

template <size_t MB_ROWS, size_t NB_COLS>
void gemm_kernel_i8i4_mrow_ref(size_t          blk_len,
                               const uint8_t * quant_a_ptr,
                               const uint8_t * quant_b_data,
                               const uint8_t * quant_b_zp,
                               float *         c_ptr,
                               size_t          count_m,
                               size_t          count_n,
                               size_t          k_blks,
                               size_t          ldc) {
    constexpr size_t kblks_per_blk = 16;
    // flagos: no GGML_ASSERT(k_blks % kblks_per_blk == 0); the tail below flushes a partial group instead

    int64_t b_blk_stride        = (sizeof(_Float16) + (blk_len / 2) + (quant_b_zp ? sizeof(uint8_t) : 0));
    int64_t b_stride            = k_blks * b_blk_stride;
    int64_t a_blk_stride        = q8_blk_size(blk_len, true);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    int64_t b_ncol_block_stride = b_blk_stride * NB_COLS;

    float    output[MB_ROWS * NB_COLS]     = { 0 };
    _Float16 output_f16[MB_ROWS * NB_COLS] = { 0 };

    for (size_t ni = 0; ni < count_n; ni += NB_COLS, c_ptr += NB_COLS) {
        size_t    nb_real = std::min<size_t>(NB_COLS, count_n - ni);
        uint8_t * b_data  = (uint8_t *) quant_b_data + ni * b_stride + NB_COLS * sizeof(_Float16);
        if (quant_b_zp) {
            b_data += NB_COLS * sizeof(uint8_t);
        }

        int8_t * a_data = (int8_t *) quant_a_ptr + sizeof(float) * MB_ROWS + sizeof(int16_t) * MB_ROWS;

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < NB_COLS; ci++) {
                output[ci + mi * NB_COLS]     = 0.0f;
                output_f16[ci + mi * NB_COLS] = static_cast<_Float16>(0.0f);
            }
        }

        size_t kii = 0;
        for (size_t ki = 0; ki < k_blks; ki++, a_data += a_nrow_block_stride, b_data += b_ncol_block_stride) {
            _Float16 * b_scale_fp16 = (_Float16 *) (b_data - NB_COLS * sizeof(_Float16));
            uint8_t *  b_zp         = nullptr;
            if (quant_b_zp) {
                b_scale_fp16 = (_Float16 *) (b_data - NB_COLS * sizeof(_Float16) - NB_COLS * sizeof(uint8_t));
                b_zp         = (uint8_t *) (b_data - NB_COLS * sizeof(uint8_t));
            }

            float *   a_scale_row = (float *) (a_data - sizeof(float) * MB_ROWS - sizeof(int16_t) * MB_ROWS);
            int16_t * a_sum_row   = (int16_t *) (a_data - sizeof(int16_t) * MB_ROWS);

            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                float a_scale_f32;  // flagos: memcpy, as 1-row blocks (38 bytes) are unaligned
                std::memcpy(&a_scale_f32, a_scale_row + mi, sizeof(float));
                _Float16 a_scale = a_scale_f32;
                int16_t  a_sum   = a_sum_row[mi];

                for (size_t ci = 0; ci < NB_COLS; ci++) {
                    _Float16 b_scale = b_scale_fp16[ci];
                    int32_t  acc     = 0;
                    if (b_zp) {
                        acc += a_sum * b_zp[ci];
                    } else {
                        acc += a_sum * 8;
                    }
                    for (size_t bi = 0; bi < blk_len / 2; bi++) {
                        int8_t  a0 = a_data[mi * blk_len + 2 * bi];
                        int8_t  a1 = a_data[mi * blk_len + 2 * bi + 1];
                        uint8_t b  = b_data[ci * blk_len / 2 + bi];
                        int8_t  b0 = static_cast<int8_t>(b & 0x0F);
                        int8_t  b1 = static_cast<int8_t>((b & 0xF0) >> 4);
                        acc += static_cast<int32_t>(a0) * static_cast<int32_t>(b0) +
                               static_cast<int32_t>(a1) * static_cast<int32_t>(b1);
                    }
                    output_f16[ci + mi * NB_COLS] +=
                        static_cast<float>(acc) * static_cast<float>(a_scale) * static_cast<float>(b_scale);
                }
            }

            if (kii == kblks_per_blk - 1) {
                for (size_t mi = 0; mi < MB_ROWS; mi++) {
                    for (size_t ci = 0; ci < NB_COLS; ci++) {
                        output[ci + mi * NB_COLS] += static_cast<float>(output_f16[ci + mi * NB_COLS]);
                        output_f16[ci + mi * NB_COLS] = 0.0f;
                    }
                }
                kii = 0;
            } else {
                kii++;
            }
        }

        if (kii != 0) {  // flagos: was kii == kblks_per_blk - 1, which dropped a last group of 1 to 14 K blocks
            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                for (size_t ci = 0; ci < NB_COLS; ci++) {
                    output[ci + mi * NB_COLS] += static_cast<float>(output_f16[ci + mi * NB_COLS]);
                    output_f16[ci + mi * NB_COLS] = 0.0f;
                }
            }
            kii = 0;
        }

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < nb_real; ci++) {
                c_ptr[mi * ldc + ci] = output[mi * NB_COLS + ci];
            }
        }
    }
}

template <size_t MB_ROWS>
void quantize_a_nrow_i8_ref(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8_blk_size(blk_len, true);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
        float *   scale_a_ptr = reinterpret_cast<float *>(quant_a_ptr);
        int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + sizeof(float) * MB_ROWS);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + sizeof(float) * MB_ROWS + sizeof(int16_t) * MB_ROWS);

        for (size_t row = 0; row < MB_ROWS; row++) {
            float max_abs_a = 0.0f;
            for (size_t bk = 0; bk < blk_len; bk++) {
                max_abs_a = std::max(max_abs_a, std::abs(a_ptr[row * count_k + k + bk]));
            }

            float rep_scale_a = max_abs_a ? ((1 << 7) - 1) / max_abs_a : 0.0f;  // flagos: zero guard from the RVV version
            const float scale_a = rep_scale_a ? 1 / rep_scale_a : 0.0f;        // flagos: zero guard; memcpy, as 1-row
            std::memcpy(scale_a_ptr + row, &scale_a, sizeof(float));           // blocks (38 bytes) are unaligned

            int16_t a_sum = 0;
            for (size_t bk = 0; bk < blk_len; bk++) {
                const int8_t quantized = static_cast<int8_t>(
                    std::clamp(std::nearbyintf(a_ptr[row * count_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                quant_a_blk[row * blk_len + bk] = quantized;
                a_sum += quantized;
            }
            a_sum_ptr[row] = -a_sum;
        }
    }
}

#if defined(GGML_FLAGOS_SPACEMIT_IME2)

//
// copied: IME2 kernels (ime2_kernels.cpp) and RVV helpers (rvv_kernels.cpp); they need VLEN 1024 (the A100 cores)
//

namespace {  // upstream declares these in ime_kernels.h and rvv_kernels.h; here they are local to this file

void gemm_kernel_i8i4_hp_m1(size_t          blk_len,
                            const uint8_t * quant_a_ptr,
                            const uint8_t * quant_b_data,
                            const uint8_t * quant_b_zp,
                            float *         c_ptr,
                            size_t          count_m,
                            size_t          count_n,
                            size_t          k_blks,
                            size_t          ldc) {
    constexpr size_t NB_COLS                = 32;
    constexpr size_t k_subblks_per_superblk = 8;

    struct block_q4_0x32_layout {
        _Float16 d[NB_COLS];
        uint8_t  qs[16 * NB_COLS];
    };

    GGML_ASSERT(blk_len == 256);

    const size_t b_superblk_stride = sizeof(block_q4_0x32_layout) * k_subblks_per_superblk +
                                     (quant_b_zp ? NB_COLS * k_subblks_per_superblk * sizeof(uint8_t) : 0);
    const size_t b_tile_stride = k_blks * b_superblk_stride;

    if (quant_b_zp == NULL) {
        for (size_t ni = 0; ni < count_n; ni += 32) {
            uint8_t * b_data = (uint8_t *) quant_b_data + (ni / NB_COLS) * b_tile_stride;
            int8_t *  a_data = (int8_t *) quant_a_ptr;
            float *   dst_c  = c_ptr + ni;

            asm volatile(
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vxor.vv        v31, v31, v31           \n\t"  // init acc to zero
                "mv             t4, %[BK]               \n\t"
                "li             t0, 0x4c00              \n\t"  // 16 in fp16
                "fmv.h.x        fa0, t0                 \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                "li             t5, 8                   \n\t"
                "addi           t6, %[A], 288           \n\t"  // point to blk scale
                "flh            ft1, (t6)               \n\t"
                "addi           t6, %[A], 272           \n\t"  // point to asum

                // init the acc fp16
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vxor.vv        v16, v18, v18           \n\t"
                "vxor.vv        v17, v18, v18           \n\t"
                "vxor.vv        v18, v18, v18           \n\t"
                "vxor.vv        v19, v18, v18           \n\t"

                "INNER_BLK_LOOP%=:                      \n\t"
                // load a sum and scale
                "flh            fa1, (t6)               \n\t"
                "addi           t6, t6, 2               \n\t"
                "flh            ft0, (%[A])             \n\t"
                "addi           %[A], %[A], 2           \n\t"
                // load A
                "vsetvli        t0, x0, e8, mf4         \n\t"
                "vle8.v         v3, (%[A])              \n\t"  // 1x32@i8
                "addi           %[A], %[A], 32          \n\t"

                // load scale B and B
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vle16.v        v8, (%[B])              \n\t"  // b_scale fp16
                "addi           %[B], %[B], 64          \n\t"
                "vl4r.v         v4, (%[B])              \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512         \n\t"
                "vfmul.vf       v8, v8, ft0             \n\t"  // scale b * scale a
                "vfmul.vf       v9, v8, fa0             \n\t"
                "vfmul.vf       v10, v8, fa1            \n\t"  // scale b * scale a * asm
                "vfwmacc.vf     v31, ft1, v10           \n\t"  // asum * scale a * scale b * blk scale

                "vsetvli        t0, x0, e8, m1          \n\t"
                "vpack.vv       v0, v8, v9, 3           \n\t"
                "vsrl.vi        v28, v3, 4              \n\t"

                "vsetvli        t0, x0, e16, m1         \n\t"
                "vnpack4.vv     v2, v3, v3, 3           \n\t"  // lo4 of A
                "vnpack4.vv     v3, v28, v28, 3         \n\t"  // hi4 of A

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vmadotsu.hp    v16, v3, v4, v0, 4, i4  \n\t"  // high 4
                "vmadotsu.hp    v17, v3, v5, v0, 5, i4  \n\t"
                "vmadotsu.hp    v18, v3, v6, v0, 6, i4  \n\t"
                "vmadotsu.hp    v19, v3, v7, v0, 7, i4  \n\t"
                "vmadotu.hp     v16, v2, v4, v0, 0, i4  \n\t"  // low 4
                "vmadotu.hp     v17, v2, v5, v0, 1, i4  \n\t"
                "vmadotu.hp     v18, v2, v6, v0, 2, i4  \n\t"
                "vmadotu.hp     v19, v2, v7, v0, 3, i4  \n\t"

                "addi           t5, t5, -1              \n\t"
                "bgtz           t5, INNER_BLK_LOOP%=    \n\t"

                "vpack.vv       v8, v16, v17, 1         \n\t"
                "vpack.vv       v12, v18, v19, 1        \n\t"
                "vpack.vv       v20, v8, v12, 2         \n\t"

                "vsetvli        t0, x0, e16, mf2        \n\t"
                "addi           t4, t4, -1              \n\t"
                "vfwmacc.vf     v31, ft1, v20           \n\t"
                //"vsetvli        t0, x0, e32, m1         \n\t"
                //"vfmul.vf       v31, v31, ft1           \n\t"  // blk scale

                // update A ptr
                "addi           %[A], t6, 2             \n\t"

                "bgtz           t4, BLK_LOOP%=          \n\t"

                // save
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vse32.v        v31, (%[DST])           \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                  "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                  "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "ft0", "ft1");
        }
    } else {
        // TODO: support quant_b_zp for i8i4 hp kernel
        GGML_ABORT("gemm_kernel_i8i4_hp_m1 with quant_b_zp is not supported yet");
    }
}

void gemm_kernel_i8i4_hp_m4(size_t          blk_len,
                            const uint8_t * quant_a_ptr,
                            const uint8_t * quant_b_data,
                            const uint8_t * quant_b_zp,
                            float *         c_ptr,
                            size_t          count_m,
                            size_t          count_n,
                            size_t          k_blks,
                            size_t          ldc) {
    constexpr size_t NB_COLS                = 32;
    constexpr size_t K_SUBBLKS_PER_SUPERBLK = 8;
    constexpr size_t K_SUBBLK_LEN           = 32;

    struct block_q4_0x32_layout {
        _Float16 d[NB_COLS];
        uint8_t  qs[16 * NB_COLS];
    };

    GGML_ASSERT(blk_len == 256);
    GGML_ASSERT(count_m >= 4);

    // Contract:
    // - computes a 4-row x 32-col tile per inner invocation
    // - A is q8 HP packed in m4 layout, one logical K256 block at a time
    // - B is q4 HP packed in N32 tiles, optionally with a separate zp area
    // - tail-N is currently not handled here; the caller must provide full N32 tiles

    const size_t b_superblk_stride = sizeof(block_q4_0x32_layout) * K_SUBBLKS_PER_SUPERBLK +
                                     (quant_b_zp ? NB_COLS * K_SUBBLKS_PER_SUPERBLK * sizeof(uint8_t) : 0);
    const size_t b_tile_stride       = k_blks * b_superblk_stride;
    const size_t a_nrow_block_stride = q8_hp_blk_size(blk_len, true, true) * 4;
    const size_t a_subblk_stride     = q8_hp_blk_size(K_SUBBLK_LEN, false, false) * 4;

    if (quant_b_zp != nullptr) {
        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            const size_t nb_real = std::min<size_t>(NB_COLS, count_n - ni);
            if (nb_real != NB_COLS) {
                break;
            }

            uint8_t * b_tile_base = (uint8_t *) quant_b_data + (ni / NB_COLS) * b_tile_stride;
            uint8_t * a_block     = (uint8_t *) quant_a_ptr;
            float *   dst_c       = c_ptr + ni;

            // Data layout summary for the with-zp path.
            //
            // A: M4 x K256 q8 HP block
            //   - split into 8 x K32 subblocks
            //   - each K32 subblock is 136B:
            //       8B   = 4 x fp16 row scales
            //       128B = 4 x int8[32] row payloads
            //   - trailer after 8 subblocks is 72B:
            //       4 rows x fp16[8] a_sum values, indexed as [row][ksi]
            //       4 rows x fp16 scale_avg tail
            //
            // B: N32 x K256 q4 HP block with explicit zp area
            //   - each K32 subblock is 576B:
            //       64B  = fp16 scale[32]
            //       512B = packed q4 payload for 32 columns x 32 k-elements
            //   - zp is stored separately, not interleaved with the 576B payload block
            //   - one K256 superblock is laid out as:
            //       8 x (scale + qs) blocks = 4608B
            //       8 x zp[32]              =  256B
            //
            // C: 4 rows x 32 fp32 outputs
            //
            // ASM pointer convention:
            //   - t6: current A K32 subblock base
            //   - t2: current A a_sum base for this ksi
            //         row1/row2/row3 are at +16/+32/+48 bytes
            //   - s5: current B (scale + qs) K32 subblock base
            //   - s6: current B zp[32] base for this ksi
            //
            // Loop progression:
            //   - per ksi: A += 136, a_sum += 2, B_data += 576, B_zp += 32
            //   - per ki : skip the 72B A trailer and advance B to the next 4864B superblock

            const _Float16 hp_scale_16   = (_Float16) 16.0f;
            const _Float16 hp_scale_1    = (_Float16) 1.0f;
            const _Float16 hp_scale_0125 = (_Float16) 0.125f;

            // VPR grouping used below:
            // - v4-v7   : B q4 payload for N32 split as 4 x N8 groups
            // - v8/v10  : zp u8 / widened fp16
            // - v12     : B fp16 scale[32]
            // - v14-v15 : packed (Bscale * Ascale) for rows [0,1] / [2,3]
            // - v16-v19 : temporary per-row scaled B scales
            // - v28-v31 : final fp32 accumulators for rows 0..3

            asm volatile(
                "mv             t5, %[BK]                 \n\t"
                "mv             t6, %[A]                  \n\t"
                "mv             s5, %[B]                  \n\t"
                "vsetvli        t0, x0, e32, m1           \n\t"
                "vxor.vv        v28, v28, v28             \n\t"
                "vxor.vv        v29, v29, v29             \n\t"
                "vxor.vv        v30, v30, v30             \n\t"
                "vxor.vv        v31, v31, v31             \n\t"
                "li             t4, 8                     \n\t"
                "li             t1, 4608                  \n\t"
                "addi           t2, t6, 1088              \n\t"  // 8 * 136B A K32 subblocks, a_sum trailer starts here
                "add            s6, s5, t1                \n\t"  // 8 * 576B B(scale+qs), zp area starts here

                ".align 4                                 \n\t"
                "_BLK_LPST%=:                             \n\t"
                "flh            fa1, 64(t2)               \n\t"  // a_scale_avg_row[0]
                "vsetvli        t0, x0, e32, m1           \n\t"
                "vxor.vv        v18, v30, v30             \n\t"
                "vxor.vv        v19, v31, v31             \n\t"
                "vxor.vv        v20, v30, v30             \n\t"
                "vxor.vv        v21, v31, v31             \n\t"
                "_KsubBLK_LPST%=:                         \n\t"
                // load first subblock scales for 4 rows
                "flh            fa0,   0(t6)              \n\t"  // ascale_fp16

                // load B fp16 scales[32]
                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vle16.v        v12, (s5)                 \n\t"

                // load Bzp[32] for the current ksi from the dedicated zp area
                "vsetvli        t0, x0, e8, mf4           \n\t"
                "vle8.v         v8, (s6)                  \n\t"

                "fmul.h         fa2, fa0, %[HP16]         \n\t"
                "vfwcvt.f.xu.v  v10, v8                   \n\t"  // uint8 -> fp16

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vfmul.vf       v16, v12, fa0             \n\t"  // row0: Bscale * Ascale
                "vfmul.vf       v17, v12, fa2             \n\t"

                // load a_sum[row][ksi] from the trailer; t2 points to row0[ksi]
                "flh            ft1, 0(t2)                \n\t"
                "flh            ft2, 16(t2)               \n\t"
                "flh            ft3, 32(t2)               \n\t"
                "flh            ft4, 48(t2)               \n\t"

                "fmul.h         ft1, ft1, %[HP0125]       \n\t"
                "fmul.h         ft2, ft2, %[HP0125]       \n\t"
                "fmul.h         ft3, ft3, %[HP0125]       \n\t"
                "fmul.h         ft4, ft4, %[HP0125]       \n\t"

                // load A payload from current K32 subblock and B q4 payload from current 576B block
                "addi           t3, t6, 8                 \n\t"
                "vsetvli        t0, x0, e8, m1            \n\t"
                "vl1r.v         v0, (t3)                  \n\t"  //A
                "addi           t3, s5, 64                \n\t"
                "vl4r.v         v4, (t3)                  \n\t"  //B

                "vsetvli        t0, x0, e8, m1            \n\t"
                "vsrl.vi        v1, v0, 4                 \n\t"
                "vnpack4.vv     v12, v0, v1, 3            \n\t"
                "vpack.vv       v0, v17, v16, 3           \n\t"
                "vupack.vv      v2, v12, v12, 2           \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"  // mf2 -> mf2
                "vfmul.vv       v10, v10, v16             \n\t"  // zp * ascale * bscale; fp16*fp16

                "vsetvli        t0, x0, e16, mf2          \n\t"  // mf2 -> m1
                "vfmul.vf       v12, v10, ft1             \n\t"  // zp(1:n)* abscale * asum_m0; fp16*fp16
                "vfmul.vf       v13, v10, ft2             \n\t"  // zp(1:n)* abscale * asum_m1; fp16*fp16
                "vfmul.vf       v24, v10, ft3             \n\t"  // zp(1:n)* abscale * asum_m2; fp16*fp16
                "vfmul.vf       v25, v10, ft4             \n\t"  // zp(1:n)* abscale * asum_m3; fp16*fp16

                "vsetvli        t0, x0, e16, mf2           \n\t"
                "vfwmacc.vf     v28, fa1, v12             \n\t"  // row0/1 accum += dot * packed scale
                "vfwmacc.vf     v29, fa1, v13             \n\t"
                "vfwmacc.vf     v30, fa1, v24             \n\t"
                "vfwmacc.vf     v31, fa1, v25             \n\t"

                "vsetvli        t0, x0, e32, m1           \n\t"
                "vmadotsu.hp    v18, v3, v4, v0, 0, i4    \n\t"  //lo4;n0n7
                "vmadotsu.hp    v19, v3, v5, v0, 1, i4    \n\t"  //lo4;n8n15
                "vmadotsu.hp    v20, v3, v6, v0, 2, i4    \n\t"  //lo4;n16n23
                "vmadotsu.hp    v21, v3, v7, v0, 3, i4    \n\t"  //lo4;n24n31
                "vmadotu.hp     v18, v2, v4, v0, 4, i4    \n\t"  //hi4;n0n7
                "vmadotu.hp     v19, v2, v5, v0, 5, i4    \n\t"  //hi4;n8n15
                "vmadotu.hp     v20, v2, v6, v0, 6, i4    \n\t"  //hi4;n16n23
                "vmadotu.hp     v21, v2, v7, v0, 7, i4    \n\t"  //hi4;n24n31

                "addi           t4, t4, -1                \n\t"
                "addi           t6, t6, 8+128             \n\t"  // next A K32 subblock
                "addi           t2, t2, 2                 \n\t"  // next ksi entry in each a_sum row
                "addi           s5, s5, 64+512            \n\t"  // next B (scale + qs) K32 block
                "addi           s6, s6, 32                \n\t"  // next zp[32]
                "bgtz           t4, _KsubBLK_LPST%=       \n\t"

                "vsetvli        t0, x0, e16, m1           \n\t"
                "vpack.vv       v8, v18, v19, 1           \n\t"  // 128(16*8)->256(16*16)
                "vpack.vv       v12, v20, v21, 1          \n\t"
                "vpack.vv       v26, v8, v12, 2           \n\t"  // 256(16*16)->512(16*32)

                "vsetvli        t0, x0, e16, m1           \n\t"
                "vfwmacc.vf     v28, fa1, v26             \n\t"  // row0/1 accum += dot * packed scale
                "vfwmacc.vf     v30, fa1, v27             \n\t"

                "li             t4, 8                     \n\t"
                "addi           t5, t5, -1                \n\t"
                "addi           t6, t6, 72                \n\t"  // skip A trailer after 8 subblocks and scale_avg tail
                "mv             s5, s6                    \n\t"  // s6 already points to next B superblock base
                "addi           t2, t6, 1088              \n\t"  // 8 * 136B A K32 subblocks, a_sum trailer starts here
                "add            s6, s5, t1                \n\t"  // 8 * 576B B(scale+qs), zp area starts here
                "bgtz           t5, _BLK_LPST%=           \n\t"

                "_BLK_LPND%=:                             \n\t"
                "vsetvli        t0, x0, e32, m1           \n\t"
                "add            t2, %[LDC], %[DST]        \n\t"
                "vse32.v        v28, (%[DST])             \n\t"
                "add            t3, %[LDC], t2            \n\t"
                "vse32.v        v29, (t2)                 \n\t"
                "add            t2, %[LDC], t3            \n\t"
                "vse32.v        v30, (t3)                 \n\t"
                "vse32.v        v31, (t2)                 \n\t"
                : [A] "+r"(a_block), [B] "+r"(b_tile_base)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks), [HP16] "f"(hp_scale_16),
                  [HP1] "f"(hp_scale_1), [HP0125] "f"(hp_scale_0125)
                : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s5", "s6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                  "v8", "v10", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "ft1", "ft2", "ft3", "ft4",
                  "memory");
        }
        return;
    } else {
        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            const size_t nb_real = std::min<size_t>(NB_COLS, count_n - ni);
            if (nb_real != NB_COLS) {
                break;
            }

            uint8_t * b_tile_base = (uint8_t *) quant_b_data + (ni / NB_COLS) * b_tile_stride;
            uint8_t * a_block     = (uint8_t *) quant_a_ptr;
            float *   dst_c       = c_ptr + ni;

            // Data layout summary for the no-zp path.
            //
            // A layout is identical to the with-zp branch.
            //
            // B: N32 x K256 q4 HP block without explicit zp storage
            //   - each K32 subblock is still 576B:
            //       64B  = fp16 scale[32]
            //       512B = packed q4 payload
            //   - zp is implicit and treated as a constant value 8 in the kernel
            //   - one K256 superblock therefore contains only:
            //       8 x (scale + qs) blocks = 4608B
            //
            // C: 4 rows x 32 fp32 outputs
            //
            // ASM pointer convention:
            //   - t6: current A K32 subblock base
            //   - t2: current A a_sum base for this ksi
            //   - s5: current B (scale + qs) K32 subblock base
            //
            // Loop progression:
            //   - per ksi: A += 136, a_sum += 2, B_data += 576
            //   - per ki : skip the 72B A trailer and advance B to the next 4608B superblock

            const _Float16 hp_scale_16 = (_Float16) 16.0f;
            const _Float16 hp_scale_1  = (_Float16) 1.0f;

            // VPR grouping used below matches the with-zp path:
            // - v4-v7   : B q4 payload for N32 split as 4 x N8 groups
            // - v8/v10  : implicit zp lane / widened fp16
            // - v12     : B fp16 scale[32]
            // - v14-v15 : packed (Bscale * Ascale) for rows [0,1] / [2,3]
            // - v16-v19 : temporary per-row scaled B scales
            // - v28-v31 : final fp32 accumulators for rows 0..3

            asm volatile(
                "mv             t5, %[BK]                 \n\t"
                "mv             t6, %[A]                  \n\t"
                "mv             s5, %[B]                  \n\t"
                "vsetvli        t0, x0, e32, m1           \n\t"
                "vxor.vv        v28, v28, v28             \n\t"
                "vxor.vv        v29, v29, v29             \n\t"
                "vxor.vv        v30, v30, v30             \n\t"
                "vxor.vv        v31, v31, v31             \n\t"
                "li             t4, 8                     \n\t"
                "addi           t2, t6, 1088              \n\t"  // 8 * 136B A K32 subblocks, a_sum trailer starts here

                ".align 4                                 \n\t"
                "_BLK_LPST%=:                             \n\t"
                "flh            fa1, 64(t2)               \n\t"  // a_scale_avg_row[0]
                "vsetvli        t0, x0, e32, m1           \n\t"
                "vxor.vv        v18, v30, v30             \n\t"
                "vxor.vv        v19, v31, v31             \n\t"
                "vxor.vv        v20, v30, v30             \n\t"
                "vxor.vv        v21, v31, v31             \n\t"
                "_KsubBLK_LPST%=:                         \n\t"
                // load first subblock scales for 4 rows
                "flh            fa0,   0(t6)              \n\t"  // ascale_fp16

                // load B fp16 scales[32]
                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vle16.v        v12, (s5)                 \n\t"

                "fmul.h         fa2, fa0, %[HP16]         \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vfmul.vf       v16, v12, fa0             \n\t"  // row0: Bscale * Ascale
                "vfmul.vf       v17, v12, fa2             \n\t"

                // load a_sum[row][ksi] from the trailer; t2 points to row0[ksi]
                "flh            ft1, 0(t2)                \n\t"
                "flh            ft2, 16(t2)               \n\t"
                "flh            ft3, 32(t2)               \n\t"
                "flh            ft4, 48(t2)               \n\t"

                // load A payload from current K32 subblock and B q4 payload from current 576B block
                "addi           t3, t6, 8                 \n\t"
                "vsetvli        t0, x0, e8, m1            \n\t"
                "vl1r.v         v0, (t3)                  \n\t"  //A
                "addi           t3, s5, 64                \n\t"
                "vl4r.v         v4, (t3)                  \n\t"  //B

                "vsetvli        t0, x0, e8, m1            \n\t"
                "vsrl.vi        v1, v0, 4                 \n\t"
                "vnpack4.vv     v12, v0, v1, 3            \n\t"
                "vpack.vv       v0, v17, v16, 3           \n\t"
                "vupack.vv      v2, v12, v12, 2           \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"  // mf2 -> m1
                "vfmul.vf       v12, v16, ft1             \n\t"  // zp(1:n)* abscale * asum_m0; fp16*fp16
                "vfmul.vf       v13, v16, ft2             \n\t"  // zp(1:n)* abscale * asum_m1; fp16*fp16
                "vfmul.vf       v24, v16, ft3             \n\t"  // zp(1:n)* abscale * asum_m2; fp16*fp16
                "vfmul.vf       v25, v16, ft4             \n\t"  // zp(1:n)* abscale * asum_m3; fp16*fp16

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vfwmacc.vf     v28, fa1, v12             \n\t"
                "vfwmacc.vf     v29, fa1, v13             \n\t"
                "vfwmacc.vf     v30, fa1, v24             \n\t"
                "vfwmacc.vf     v31, fa1, v25             \n\t"

                "vsetvli        t0, x0, e32, m1           \n\t"
                "vmadotsu.hp    v18, v3, v4, v0, 0, i4    \n\t"  //lo4;n0n7
                "vmadotsu.hp    v19, v3, v5, v0, 1, i4    \n\t"  //lo4;n8n15
                "vmadotsu.hp    v20, v3, v6, v0, 2, i4    \n\t"  //lo4;n16n23
                "vmadotsu.hp    v21, v3, v7, v0, 3, i4    \n\t"  //lo4;n24n31
                "vmadotu.hp     v18, v2, v4, v0, 4, i4    \n\t"  //hi4;n0n7
                "vmadotu.hp     v19, v2, v5, v0, 5, i4    \n\t"  //hi4;n8n15
                "vmadotu.hp     v20, v2, v6, v0, 6, i4    \n\t"  //hi4;n16n23
                "vmadotu.hp     v21, v2, v7, v0, 7, i4    \n\t"  //hi4;n24n31

                "addi           t4, t4, -1                \n\t"

                "addi           t6, t6, 8+128             \n\t"  // next A K32 subblock
                "addi           t2, t2, 2                 \n\t"  // next ksi entry in each a_sum row
                "addi           s5, s5, 64+512            \n\t"  // next B (scale + qs) K32 block
                "bgtz           t4, _KsubBLK_LPST%=       \n\t"

                "vsetvli        t0, x0, e16, m1           \n\t"  //N32in1register
                "vpack.vv       v8, v18, v19, 1           \n\t"  // 128(16*8)->256(16*16)
                "vpack.vv       v12, v20, v21, 1          \n\t"
                "vpack.vv       v26, v8, v12, 2           \n\t"  // 256(16*16)->512(16*32)

                "vsetvli        t0, x0, e16, m1           \n\t"
                "vfwmacc.vf     v28, fa1, v26             \n\t"  // row0/1 accum += dot * packed scale
                "vfwmacc.vf     v30, fa1, v27             \n\t"

                "li             t4, 8                     \n\t"
                "addi           t5, t5, -1                \n\t"
                "addi           t6, t6, 72                \n\t"  // skip A trailer after 8 subblocks and scale_avg tail
                // s5 already points to next B superblock base
                "addi           t2, t6, 1088              \n\t"  // 8 * 136B A K32 subblocks, a_sum trailer starts here
                "bgtz           t5, _BLK_LPST%=           \n\t"

                "_BLK_LPND%=:                             \n\t"
                "vsetvli        t0, x0, e32, m1           \n\t"
                "add            t2, %[LDC], %[DST]        \n\t"
                "vse32.v        v28, (%[DST])             \n\t"
                "add            t3, %[LDC], t2            \n\t"
                "vse32.v        v29, (t2)                 \n\t"
                "add            t2, %[LDC], t3            \n\t"
                "vse32.v        v30, (t3)                 \n\t"
                "vse32.v        v31, (t2)                 \n\t"
                : [A] "+r"(a_block), [B] "+r"(b_tile_base)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks), [HP16] "f"(hp_scale_16), [HP1] "f"(hp_scale_1)
                : "t0", "t2", "t3", "t4", "t5", "t6", "s5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10",
                  "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v24", "v25", "v26",
                  "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "ft1", "ft2", "ft3", "ft4", "memory");
        }
        return;
    }
}

void memcpy1d(void * dst, const void * src, int64_t size) {
    size_t byte_size_all = size;
    size_t vlen          = __riscv_vlenb() * 8;
    if (vlen == 256) {
        // 1024 bytes
        __asm__ volatile(
            //
            "srli           t0, %[size], 10             \n\t"
            "blez           t0, memcpy_tail%=           \n\t"
            "vsetvli        t1, x0, e8, m8, tu, mu      \n\t"
            "memcpy_main_loop%=:                        \n\t"
            "addi           t0, t0, -1                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "addi           %[s], %[s], 256             \n\t"
            "vle8.v         v8, (%[s])                  \n\t"
            "addi           %[s], %[s], 256             \n\t"
            "vle8.v         v16, (%[s])                 \n\t"
            "addi           %[s], %[s], 256             \n\t"
            "vle8.v         v24, (%[s])                 \n\t"
            "addi           %[s], %[s], 256             \n\t"
            //
            "vse8.v         v0, (%[d])                  \n\t"
            "addi           %[d], %[d], 256             \n\t"
            "vse8.v         v8, (%[d])                  \n\t"
            "addi           %[d], %[d], 256             \n\t"
            "vse8.v         v16, (%[d])                 \n\t"
            "addi           %[d], %[d], 256             \n\t"
            "vse8.v         v24, (%[d])                 \n\t"
            "addi           %[d], %[d], 256             \n\t"
            //
            "bnez           t0, memcpy_main_loop%=      \n\t"
            "memcpy_tail%=:                             \n\t"
            "andi           t1, %[size], 1023           \n\t"
            "blez           t1, out%=                   \n\t"
            "memcpy_tail_loop%=:                        \n\t"
            "vsetvli        t0, t1, e8, m8, tu, mu      \n\t"
            "sub            t1, t1, t0                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "add            %[s], %[s], t0              \n\t"
            "vse8.v         v0, (%[d])                  \n\t"
            "add            %[d], %[d], t0              \n\t"
            "bnez           t1, memcpy_tail_loop%=      \n\t"
            "out%=:                                     \n\t"
            : [s] "+r"(src), [d] "+r"(dst)
            : [size] "r"(byte_size_all)
            : "cc", "t0", "t1");
    } else if (vlen == 1024) {
        // 2048 bytes
        __asm__ volatile(
            //
            "srli           t0, %[size], 11             \n\t"
            "blez           t0, memcpy_tail%=           \n\t"
            "vsetvli        t1, x0, e8, m8, tu, mu      \n\t"
            "addi           t2, %[s], 1024              \n\t"
            "addi           t3, %[d], 1024              \n\t"
            "li             t5, 2048                    \n\t"
            "memcpy_main_loop%=:                        \n\t"
            "addi           t0, t0, -1                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "add            %[s], %[s], t5              \n\t"
            "vle8.v         v8, (t2)                    \n\t"
            "add            t2, t2, t5                  \n\t"
            //
            "vse8.v         v0, (%[d])                  \n\t"
            "add            %[d], %[d], t5              \n\t"
            "vse8.v         v8, (t3)                    \n\t"
            "add            t3, t3, t5                  \n\t"
            //
            "bnez           t0, memcpy_main_loop%=      \n\t"
            "memcpy_tail%=:                             \n\t"
            "andi           t1, %[size], 2047           \n\t"
            "blez           t1, out%=                   \n\t"
            "memcpy_tail_loop%=:                        \n\t"
            "vsetvli        t0, t1, e8, m2, tu, mu      \n\t"
            "sub            t1, t1, t0                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "add            %[s], %[s], t0              \n\t"
            "vse8.v         v0, (%[d])                  \n\t"
            "add            %[d], %[d], t0              \n\t"
            "bnez           t1, memcpy_tail_loop%=      \n\t"
            "out%=:                                     \n\t"
            : [s] "+r"(src), [d] "+r"(dst)
            : [size] "r"(byte_size_all)
            : "cc", "t0", "t1", "t2", "t3", "t5");
    } else {
        __asm__ volatile(
            //
            "add            t1, %[size], zero           \n\t"
            "memcpy_tail_loop%=:                        \n\t"
            "vsetvli        t0, t1, e8, m8, tu, mu      \n\t"
            "sub            t1, t1, t0                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "add            %[s], %[s], t0              \n\t"
            "vse8.v         v0, (%[d])                  \n\t"
            "add            %[d], %[d], t0              \n\t"
            "bnez           t1, memcpy_tail_loop%=      \n\t"
            : [s] "+r"(src), [d] "+r"(dst)
            : [size] "r"(byte_size_all)
            : "cc", "t0", "t1", "t2", "t4", "t3");
    }
}

void quantize_a_row_i8_hp(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_count             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_size(blk_len, true, true);
    int64_t          a_subblk_stride          = q8_hp_blk_size(k_subblk_len, false, false);
    size_t           vlenb                    = __riscv_vlenb();
    float            scale_temp[subblk_count] = { 0.0f };

    if (vlenb == 128) {
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_sum_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_count);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - sizeof(_Float16));
            float      scale_avg     = 0.0f;

            for (size_t kk = 0; kk < subblk_count; ++kk) {
                const float * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                float        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_temp[kk];
            }

            scale_avg /= subblk_count;
            const float scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            for (size_t kk = 0; kk < subblk_count; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + sizeof(_Float16));
                const float * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a         = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                float        rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;
                scale_a_ptr[0]           = static_cast<_Float16>(scale_temp[kk] * scale_factor);

                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_sum, vl);
                int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                a_sum_ptr[kk]      = static_cast<_Float16>(-a_sum) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk, v_a_quant_i8, vl);
            }
        }
    } else if (vlenb == 32) {
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_sum_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_count);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - sizeof(_Float16));
            float      scale_avg     = 0.0f;

            for (size_t kk = 0; kk < subblk_count; ++kk) {
                const float * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                float        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_temp[kk];
            }

            scale_avg /= subblk_count;
            const float scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            for (size_t kk = 0; kk < subblk_count; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + sizeof(_Float16));
                const float * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a         = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                float        rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;
                scale_a_ptr[0]           = static_cast<_Float16>(scale_temp[kk] * scale_factor);

                vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
                vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
                vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

                vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_sum, vl);
                int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                a_sum_ptr[kk]      = static_cast<_Float16>(-a_sum) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk, v_a_quant_i8, vl);
            }
        }
    } else {
        quantize_a_nrow_i8_hp_ref<1>(blk_len, a_ptr, count_k, quant_a_ptr);
    }
}

void quantize_a_4row_i8_hp(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_count             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_size(blk_len, true, true);
    int64_t          a_nrow_block_stride      = a_blk_stride * 4;
    int64_t          a_subblk_stride          = q8_hp_blk_size(k_subblk_len, false, false) * 4;
    size_t           vlenb                    = __riscv_vlenb();
    float            scale_temp[subblk_count] = { 0.0f };

    if (vlenb == 128) {
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            _Float16 * a_sum_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_count);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_block_stride - sizeof(_Float16) * 4);
            float scale_avg = 0.0f;

            for (size_t kk = 0; kk < subblk_count; ++kk) {
                const float * a_src_ptr0 = a_ptr + 0 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr1 = a_ptr + 1 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr2 = a_ptr + 2 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr3 = a_ptr + 3 * count_k + k + kk * k_subblk_len;

                size_t       vl       = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a0     = __riscv_vle32_v_f32m1(a_src_ptr0, vl);
                vfloat32m1_t v_a1     = __riscv_vle32_v_f32m1(a_src_ptr1, vl);
                vfloat32m1_t v_a2     = __riscv_vle32_v_f32m1(a_src_ptr2, vl);
                vfloat32m1_t v_a3     = __riscv_vle32_v_f32m1(a_src_ptr3, vl);
                vfloat32m1_t v_a0_abs = __riscv_vfabs_v_f32m1(v_a0, vl);
                vfloat32m1_t v_a1_abs = __riscv_vfabs_v_f32m1(v_a1, vl);
                vfloat32m1_t v_a2_abs = __riscv_vfabs_v_f32m1(v_a2, vl);
                vfloat32m1_t v_a3_abs = __riscv_vfabs_v_f32m1(v_a3, vl);

                vfloat32m1_t v_max_abs = __riscv_vfmax_vv_f32m1(v_a0_abs, v_a1_abs, vl);
                v_max_abs              = __riscv_vfmax_vv_f32m1(v_max_abs, v_a2_abs, vl);
                v_max_abs              = __riscv_vfmax_vv_f32m1(v_max_abs, v_a3_abs, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
                float        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_temp[kk];
            }

            scale_avg /= subblk_count;
            const float scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            for (size_t kk = 0; kk < subblk_count; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + sizeof(_Float16) * 4);
                const float * a_src_ptr0    = a_ptr + 0 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr1    = a_ptr + 1 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr2    = a_ptr + 2 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr3    = a_ptr + 3 * count_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a0 = __riscv_vle32_v_f32m1(a_src_ptr0, vl);
                vfloat32m1_t v_a1 = __riscv_vle32_v_f32m1(a_src_ptr1, vl);
                vfloat32m1_t v_a2 = __riscv_vle32_v_f32m1(a_src_ptr2, vl);
                vfloat32m1_t v_a3 = __riscv_vle32_v_f32m1(a_src_ptr3, vl);

                float rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;
                scale_a_ptr[0]    = static_cast<_Float16>(scale_temp[kk] * scale_factor);

                vfloat32m1_t v_a0_scale    = __riscv_vfmul_vf_f32m1(v_a0, rep_scale_a, vl);
                vfloat32m1_t v_a1_scale    = __riscv_vfmul_vf_f32m1(v_a1, rep_scale_a, vl);
                vfloat32m1_t v_a2_scale    = __riscv_vfmul_vf_f32m1(v_a2, rep_scale_a, vl);
                vfloat32m1_t v_a3_scale    = __riscv_vfmul_vf_f32m1(v_a3, rep_scale_a, vl);
                vint16mf2_t  v_a0_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a0_scale, vl);
                vint16mf2_t  v_a1_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a1_scale, vl);
                vint16mf2_t  v_a2_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a2_scale, vl);
                vint16mf2_t  v_a3_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a3_scale, vl);
                vint8mf4_t   v_a0_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a0_quant, vl);
                vint8mf4_t   v_a1_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a1_quant, vl);
                vint8mf4_t   v_a2_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a2_quant, vl);
                vint8mf4_t   v_a3_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a3_quant, vl);

                vint16m1_t tmp_sum0 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum1 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum2 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum3 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a0_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a0_quant_i8, tmp_sum0, vl);
                vint16m1_t v_a1_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a1_quant_i8, tmp_sum1, vl);
                vint16m1_t v_a2_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a2_quant_i8, tmp_sum2, vl);
                vint16m1_t v_a3_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a3_quant_i8, tmp_sum3, vl);

                a_sum_ptr[0 * subblk_count + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[1 * subblk_count + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[2 * subblk_count + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[3 * subblk_count + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_sum)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } else if (vlenb == 32) {
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            _Float16 * a_sum_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_count);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_block_stride - sizeof(_Float16) * 4);
            float scale_avg = 0.0f;

            for (size_t kk = 0; kk < subblk_count; ++kk) {
                const float * a_src_ptr0 = a_ptr + 0 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr1 = a_ptr + 1 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr2 = a_ptr + 2 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr3 = a_ptr + 3 * count_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a0 = __riscv_vle32_v_f32m4(a_src_ptr0, vl);
                vfloat32m4_t v_a1 = __riscv_vle32_v_f32m4(a_src_ptr1, vl);
                vfloat32m4_t v_a2 = __riscv_vle32_v_f32m4(a_src_ptr2, vl);
                vfloat32m4_t v_a3 = __riscv_vle32_v_f32m4(a_src_ptr3, vl);

                vfloat32m4_t v_a0_abs = __riscv_vfabs_v_f32m4(v_a0, vl);
                vfloat32m4_t v_a1_abs = __riscv_vfabs_v_f32m4(v_a1, vl);
                vfloat32m4_t v_a2_abs = __riscv_vfabs_v_f32m4(v_a2, vl);
                vfloat32m4_t v_a3_abs = __riscv_vfabs_v_f32m4(v_a3, vl);

                vfloat32m4_t v_max_abs = __riscv_vfmax_vv_f32m4(v_a0_abs, v_a1_abs, vl);
                v_max_abs              = __riscv_vfmax_vv_f32m4(v_max_abs, v_a2_abs, vl);
                v_max_abs              = __riscv_vfmax_vv_f32m4(v_max_abs, v_a3_abs, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_max_abs, tmp, vl);
                float        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_temp[kk];
            }

            scale_avg /= subblk_count;
            const float scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            for (size_t kk = 0; kk < subblk_count; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + sizeof(_Float16) * 4);
                const float * a_src_ptr0    = a_ptr + 0 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr1    = a_ptr + 1 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr2    = a_ptr + 2 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr3    = a_ptr + 3 * count_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a0 = __riscv_vle32_v_f32m4(a_src_ptr0, vl);
                vfloat32m4_t v_a1 = __riscv_vle32_v_f32m4(a_src_ptr1, vl);
                vfloat32m4_t v_a2 = __riscv_vle32_v_f32m4(a_src_ptr2, vl);
                vfloat32m4_t v_a3 = __riscv_vle32_v_f32m4(a_src_ptr3, vl);

                float rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;
                scale_a_ptr[0]    = static_cast<_Float16>(scale_temp[kk] * scale_factor);

                vfloat32m4_t v_a0_scale    = __riscv_vfmul_vf_f32m4(v_a0, rep_scale_a, vl);
                vfloat32m4_t v_a1_scale    = __riscv_vfmul_vf_f32m4(v_a1, rep_scale_a, vl);
                vfloat32m4_t v_a2_scale    = __riscv_vfmul_vf_f32m4(v_a2, rep_scale_a, vl);
                vfloat32m4_t v_a3_scale    = __riscv_vfmul_vf_f32m4(v_a3, rep_scale_a, vl);
                vint16m2_t   v_a0_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a0_scale, vl);
                vint16m2_t   v_a1_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a1_scale, vl);
                vint16m2_t   v_a2_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a2_scale, vl);
                vint16m2_t   v_a3_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a3_scale, vl);
                vint8m1_t    v_a0_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a0_quant, vl);
                vint8m1_t    v_a1_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a1_quant, vl);
                vint8m1_t    v_a2_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a2_quant, vl);
                vint8m1_t    v_a3_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a3_quant, vl);

                vint16m1_t tmp_sum0 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum1 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum2 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum3 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a0_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a0_quant_i8, tmp_sum0, vl);
                vint16m1_t v_a1_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a1_quant_i8, tmp_sum1, vl);
                vint16m1_t v_a2_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a2_quant_i8, tmp_sum2, vl);
                vint16m1_t v_a3_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a3_quant_i8, tmp_sum3, vl);

                a_sum_ptr[0 * subblk_count + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[1 * subblk_count + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[2 * subblk_count + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[3 * subblk_count + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_sum)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } else {
        quantize_a_nrow_i8_hp_ref<4>(blk_len, a_ptr, count_k, quant_a_ptr);
    }
}

void gemm_kernel_i8i4_m1(size_t          blk_len,
                         const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_m,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc) {
    if (quant_b_zp == NULL) {
        for (size_t n = 0; n < count_n; n += 32) {
            size_t    nblks         = (count_n - n) > 32 ? 32 : count_n - n;
            uint8_t * QuantBDataPtr = (uint8_t *) quant_b_data +      //
                                      n * k_blks * blk_len / 2 +      // b data
                                      n * k_blks * sizeof(_Float16);  // scale
            float * CPtr = c_ptr + n;
            size_t  cnt  = k_blks;

            // A format Version_1 (FP32 SCALE FOR Normal VMADOTins of IME2)
            // A M1K32 int8    256bit
            // Ascale fp32 * 1  32bit
            // || scl*1(fp32) | Asum(int16) | blk0 || scl*1(fp32) | Asum(int16) | blk0 || ...
            // || Element                          || Element                          || ...
            // B format
            // B N8K32 int4    1024bit
            //   4VRF, N32K32, 4096bit
            // Bscale fp16 * N32 512bit;
            // || scl*32..(fp16) | blk0 blk1 ... blk31 || scl*32..(fp16) | blk0 blk1 ... blk31 || ...
            // || Element                              || Element                              || ...
#if 0
            //bias always be nullptr
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "mv           s2, %[pA]               \n\t"  // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*2         \n\t"  // s5 = pBdata;
                "mv           s6, %[pC]               \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v2, v0, v0              \n\t"  // clear acc

                // ordinary vmadot: vle*6 flw*1 vecIns*21 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+64        \n\t"  // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v0, (s4)                \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64+128*4        \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t"  // A Scale fp32
                "lh           t2, 4(s2)               \n\t"  // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, zero, e8, m1        \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vnpack4.vv   v8, v3, v3, 3           \n\t"  // lo4 of A
                "vnpack4.vv   v10, v24, v24, 3        \n\t"  // hi4 of A

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v16, v16           \n\t"
                "vxor.vv      v20, v16, v16           \n\t"
                "vxor.vv      v22, v16, v16           \n\t"

                "vmadotsu     v16, v10, v4, i4        \n\t"  // M0 N0 - N7 INT32(256bit)
                "vmadotsu     v18, v10, v5, i4        \n\t"  // M0 N8 - N15
                "vmadotsu     v20, v10, v6, i4        \n\t"  // M0 N16 - N23
                "vmadotsu     v22, v10, v7, i4        \n\t"  // M0 N24 - N31

                "vsll.vi      v16, v16, 4             \n\t"
                "vsll.vi      v18, v18, 4             \n\t"
                "vsll.vi      v20, v20, 4             \n\t"
                "vsll.vi      v22, v22, 4             \n\t"

                "vmadotu      v16, v8, v4, i4         \n\t"
                "vmadotu      v18, v8, v5, i4         \n\t"
                "vmadotu      v20, v8, v6, i4         \n\t"
                "vmadotu      v22, v8, v7, i4         \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v28, 8                  \n\t"
                "vpack.vv     v24, v16, v18, 2        \n\t"
                "vpack.vv     v26, v20, v22, 2        \n\t"
                "vpack.vv     v16, v24, v26, 3        \n\t"

                "vwmul.vx     v24, v28, t2            \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vadd.vv      v16, v16, v24           \n\t"

                // b_scale fp16 -> fp32
                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v v24, v0                 \n\t"
                // mac result i32 -> fp32
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vfcvt.f.x.v  v26, v16                \n\t"
                // a_scale * b_scale;
                "vfmul.vf     v1, v24, f0             \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv    v2, v1, v26             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
#else
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v0, 1                   \n\t"  // init the scale
                "mv           s2, %[pA]               \n\t"  // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*2         \n\t"  // s5 = pBdata;
                "mv           s6, %[pC]               \n\t"

                "vsll.vi      v1, v0, 4               \n\t"
                "vxor.vv      v2, v0, v0              \n\t"  // clear acc
                "vfcvt.f.x.v  v0, v0                  \n\t"
                "vfcvt.f.x.v  v1, v1                  \n\t"

                // vmadot hp: vle*7 flw*1 vecIns*14 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+64        \n\t"  // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v30, (s4)               \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64+128*4        \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t"  // A Scale fp32
                "lh           t2, 4(s2)               \n\t"  // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v28, 8                  \n\t"  // Bzp u8 -> u16
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmul.vx      v26, v28, t2            \n\t"  // asum*zp i16*i16
                "vnpack4.vv   v8, v3, v3, 3           \n\t"  // lo4 of A
                "vnpack4.vv   v10, v24, v24, 3        \n\t"  // hi4 of A

                "vfcvt.f.x.v  v16, v26                \n\t"  // zp i16 -> fp16
                "vadd.vi      v18, v16, 0             \n\t"
                "vadd.vi      v20, v16, 0             \n\t"
                "vadd.vi      v22, v16, 0             \n\t"

                "vmadotsu.hp  v16, v10, v4, v1, 0, i4 \n\t"  // high 4
                "vmadotsu.hp  v18, v10, v5, v1, 0, i4 \n\t"
                "vmadotsu.hp  v20, v10, v6, v1, 0, i4 \n\t"
                "vmadotsu.hp  v22, v10, v7, v1, 0, i4 \n\t"
                "vmadotu.hp   v16, v8, v4, v0, 0, i4  \n\t"  // low 4
                "vmadotu.hp   v18, v8, v5, v0, 0, i4  \n\t"
                "vmadotu.hp   v20, v8, v6, v0, 0, i4  \n\t"
                "vmadotu.hp   v22, v8, v7, v0, 0, i4  \n\t"

                "vpack.vv     v24, v16, v18, 1        \n\t"
                "vpack.vv     v26, v20, v22, 1        \n\t"
                "vpack.vv     v16, v24, v26, 2        \n\t"

                "vsetvli      t0, x0, e16, mf2        \n\t"
                // mac result * b_scale; f16*f16->f32
                "vfwmul.vv     v31, v30, v16          \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                // static_cast<float>(qsum * b_scale) * a_scale;
                "vfmacc.vf    v2, f0, v31             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");

#endif
        }
    } else {
        for (size_t n = 0; n < count_n; n += 32) {
            size_t    nblks         = (count_n - n) > 32 ? 32 : count_n - n;
            uint8_t * QuantBDataPtr = (uint8_t *) quant_b_data +      //
                                      n * k_blks * blk_len / 2 +      // b data
                                      n * k_blks * sizeof(uint8_t) +  // b zp
                                      n * k_blks * sizeof(_Float16);  // scale
            float * CPtr = c_ptr + n;
            size_t  cnt  = k_blks;

            // A format Version_1 (FP32 SCALE FOR Normal VMADOTins of IME2)
            // A M1K32 int8    256bit
            // Ascale fp32 * 1  32bit
            // || scl*1(fp32) | Asum(int16) | blk0 || scl*1(fp32) | Asum(int16) | blk0 || ...
            // || Element                          || Element                          || ...
            // B format
            // B N8K32 int4    1024bit
            //   4VRF, N32K32, 4096bit
            // Bscale fp16 * N32 512bit;
            // Bzp uint8_t * N32 256bit;
            // || scl*32..(fp16) | zp*32(uint8) | blk0 blk1 ... blk31 || scl*32..(fp16)  ...
            // || Element                                             || Element         ...

            //bias always be nullptr
#if 1  // flagos: upstream #if 0. This exact branch sums each block in 32 bits (vwmul.vx); the #else branch it
       // replaces adds the products in fp16 and multiplies zero point and activation sum in 16 bits (vmul.vx at
       // e16), which wraps above 32767 (plan.md, M2c design)
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "mv           s2, %[pA]               \n\t"  // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*3         \n\t"  // s5 = pBdata, (pB+BScl+Bzp)
                "mv           s6, %[pC]               \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v2, v0, v0              \n\t"  // clear acc

                // ordinary vmadot: vle*6 flw*1 vecIns*21 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+96        \n\t"  // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v0, (s4)                \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64              \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t"  // A Scale fp32
                "lh           t2, 4(s2)               \n\t"  // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, zero, e8, m1        \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vnpack4.vv   v8, v3, v3, 3           \n\t"  // lo4 of A
                "vnpack4.vv   v10, v24, v24, 3        \n\t"  // hi4 of A

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v16, v16           \n\t"
                "vxor.vv      v20, v16, v16           \n\t"
                "vxor.vv      v22, v16, v16           \n\t"

                "vmadotsu     v16, v10, v4, i4        \n\t"  // M0 N0 - N7 INT32(256bit)
                "vmadotsu     v18, v10, v5, i4        \n\t"  // M0 N8 - N15
                "vmadotsu     v20, v10, v6, i4        \n\t"  // M0 N16 - N23
                "vmadotsu     v22, v10, v7, i4        \n\t"  // M0 N24 - N31

                "vsll.vi      v16, v16, 4             \n\t"
                "vsll.vi      v18, v18, 4             \n\t"
                "vsll.vi      v20, v20, 4             \n\t"
                "vsll.vi      v22, v22, 4             \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vle8.v       v1, (s4)                \n\t"  // Bzp
                "addi         s4, s4, 32+128*4        \n\t"

                "vmadotu      v16, v8, v4, i4         \n\t"
                "vmadotu      v18, v8, v5, i4         \n\t"
                "vmadotu      v20, v8, v6, i4         \n\t"
                "vmadotu      v22, v8, v7, i4         \n\t"

                "vwaddu.vx    v28, v1, x0             \n\t"  // uint8 -> uint16
                "vpack.vv     v24, v16, v18, 2        \n\t"
                "vpack.vv     v26, v20, v22, 2        \n\t"
                "vpack.vv     v16, v24, v26, 3        \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vwmul.vx     v24, v28, t2            \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vadd.vv      v16, v16, v24           \n\t"

                // b_scale fp16 -> fp32
                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v v24, v0                 \n\t"
                // mac result i32 -> fp32
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vfcvt.f.x.v  v26, v16                \n\t"
                // a_scale * b_scale;
                "vfmul.vf     v1, v24, f0             \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv    v2, v1, v26             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
#else
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v0, 1                   \n\t"  // init the scale
                "mv           s2, %[pA]               \n\t"  // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*3         \n\t"  // s5 = pBdata, (pB+BScl+Bzp)
                "mv           s6, %[pC]               \n\t"

                "vsll.vi      v1, v0, 4               \n\t"
                "vxor.vv      v2, v0, v0              \n\t"  // clear acc
                "vfcvt.f.x.v  v0, v0                  \n\t"
                "vfcvt.f.x.v  v1, v1                  \n\t"

                // vmadot hp: vle*6 flw*1 vecIns*14 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+96        \n\t"  // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v30, (s4)               \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64              \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v31, (s4)               \n\t"  // B zp 32Row*uint8 = 256bit
                "addi         s4, s4, 32+128*4        \n\t"

                "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t"  // A Scale fp32
                "lh           t2, 4(s2)               \n\t"  // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vnpack4.vv   v8, v3, v3, 3           \n\t"  // lo4 of A
                "vnpack4.vv   v10, v24, v24, 3        \n\t"  // hi4 of A

                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v16, v16           \n\t"
                "vxor.vv      v20, v16, v16           \n\t"
                "vxor.vv      v22, v16, v16           \n\t"

                "vmadotsu.hp  v16, v10, v4, v1, 0, i4 \n\t"  // high 4
                "vmadotsu.hp  v18, v10, v5, v1, 0, i4 \n\t"
                "vmadotsu.hp  v20, v10, v6, v1, 0, i4 \n\t"
                "vmadotsu.hp  v22, v10, v7, v1, 0, i4 \n\t"
                "vmadotu.hp   v16, v8, v4, v0, 0, i4  \n\t"  // low 4
                "vmadotu.hp   v18, v8, v5, v0, 0, i4  \n\t"
                "vmadotu.hp   v20, v8, v6, v0, 0, i4  \n\t"
                "vmadotu.hp   v22, v8, v7, v0, 0, i4  \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vwaddu.vx    v28, v31, x0            \n\t"  // Bzp u8 -> u16

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vpack.vv     v24, v16, v18, 1        \n\t"
                "vpack.vv     v26, v20, v22, 1        \n\t"
                "vpack.vv     v16, v24, v26, 2        \n\t"

                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vmul.vx      v26, v28, t2            \n\t"  // asum*zp i16*i16
                "vfwcvt.f.f.v v22, v30                \n\t"  // b_scale fp16 -> fp32
                "vfcvt.f.x.v  v18, v26                \n\t"  // zp i16 -> fp16
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vfwadd.vv    v20, v18, v16           \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                // mac result * b_scale; f32*f32->f32
                "vfmul.vv     v31, v22, v20           \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                // static_cast<float>(qsum * b_scale) * a_scale;
                "vfmacc.vf    v2, f0, v31             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
#endif
        }
    }
}

void gemm_kernel_i8i4_m4(size_t          blk_len,
                         const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_m,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc) {
    int64_t b_data_stride =
        k_blks * (sizeof(ggml_fp16_t) + 16 * sizeof(int8_t) + (quant_b_zp != NULL ? sizeof(int8_t) : 0));
    if (quant_b_zp == NULL) {
        for (size_t ni = 0; ni < count_n; ni += 32) {
            uint8_t * b_data = (uint8_t *) quant_b_data + ni * b_data_stride;
            int8_t *  a_data = (int8_t *) quant_a_ptr;
            float *   dst_c  = c_ptr + ni;
#if 0
            asm volatile(
                "li             t1,  8              \n\t"
                "vsetvli        t0, x0, e32, m1     \n\t"
                "vxor.vv        v28, v28, v28       \n\t"
                "vxor.vv        v29, v29, v29       \n\t"
                "vxor.vv        v30, v30, v30       \n\t"
                "vxor.vv        v31, v31, v31       \n\t"
                "mv             t4, %[BK]           \n\t"

                ".align 4                           \n\t"
                "BLK_LOOP%=:                        \n\t"
                // load scale A
                "flw            fa0, (%[A])         \n\t"
                "flw            fa1, 4(%[A])        \n\t"
                "flw            fa2, 8(%[A])        \n\t"
                "flw            fa3, 12(%[A])       \n\t"
                "addi           %[A], %[A], 16      \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2    \n\t"
                "vle16.v        v12, (%[B])         \n\t"
                "addi           %[B], %[B], 64      \n\t"
                "vfwcvt.f.f.v   v14, v12            \n\t"

                "vsetivli       t0, 4, e16, mf2     \n\t"
                "vle16.v        v8, (%[A])          \n\t"  // asum
                "addi           %[A], %[A], 8       \n\t"
                "vwmul.vx       v10, v8, t1         \n\t"  // 8*asum

                "vsetvli        t0, x0, e8, m1      \n\t"
                "vl1r.v         v0, (%[A])          \n\t"
                "addi           %[A], %[A], 128     \n\t"  // 4*32@i8
                "vl4r.v         v4, (%[B])          \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512     \n\t"
                "vsrl.vi        v1, v0, 4           \n\t"
                "vnpack4.vv     v12, v0, v1, 3      \n\t"  // A low  u4
                "vupack.vv      v2, v12, v12, 2     \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e32, m1     \n\t"
                "vxor.vv        v16, v16, v16       \n\t"
                "vxor.vv        v18, v16, v16       \n\t"
                "vxor.vv        v20, v16, v16       \n\t"
                "vxor.vv        v22, v16, v16       \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e32, m1     \n\t"
                "vmadotsu       v16, v3, v4, i4     \n\t"   // high 4
                "vmadotsu       v18, v3, v5, i4     \n\t"
                "vmadotsu       v20, v3, v6, i4     \n\t"
                "vmadotsu       v22, v3, v7, i4     \n\t"
                "vsll.vi        v16, v16, 4         \n\t"
                "vsll.vi        v18, v18, 4         \n\t"
                "vsll.vi        v20, v20, 4         \n\t"
                "vsll.vi        v22, v22, 4         \n\t"
                "vmadotu        v16, v2, v4, i4     \n\t"   // low 4
                "vmadotu        v18, v2, v5, i4     \n\t"
                "vmadotu        v20, v2, v6, i4     \n\t"
                "vmadotu        v22, v2, v7, i4     \n\t"

                "vpack.vv       v0, v16, v18, 2     \n\t"
                "vpack.vv       v2, v20, v22, 2     \n\t"
                "vpack.vv       v16, v0, v2, 3      \n\t"
                "vpack.vv       v18, v1, v3, 3      \n\t"

                "vrgather.vi    v0, v10, 0          \n\t"
                "vrgather.vi    v1, v10, 1          \n\t"
                "vrgather.vi    v2, v10, 2          \n\t"
                "vrgather.vi    v3, v10, 3          \n\t"

                "vadd.vv        v16, v16, v0        \n\t"
                "vadd.vv        v17, v17, v1        \n\t"
                "vadd.vv        v18, v18, v2        \n\t"
                "vadd.vv        v19, v19, v3        \n\t"

                "vfcvt.f.x.v    v16, v16            \n\t"
                "vfcvt.f.x.v    v17, v17            \n\t"
                "vfcvt.f.x.v    v18, v18            \n\t"
                "vfcvt.f.x.v    v19, v19            \n\t"

                // mul scale
                "vfmul.vv       v16, v16, v14       \n\t"
                "vfmul.vv       v17, v17, v14       \n\t"
                "vfmul.vv       v18, v18, v14       \n\t"
                "vfmul.vv       v19, v19, v14       \n\t"

                "addi           t4, t4, -1          \n\t"
                "vfmacc.vf      v28, fa0, v16       \n\t"
                "vfmacc.vf      v29, fa1, v17       \n\t"
                "vfmacc.vf      v30, fa2, v18       \n\t"
                "vfmacc.vf      v31, fa3, v19       \n\t"

                "bgtz           t4, BLK_LOOP%=      \n\t"

                // save
                "vsetvli        t0, x0, e32, m1     \n\t"
                "add            t2, %[LDC], %[DST]  \n\t"
                "vse32.v        v28, (%[DST])       \n\t"
                "add            t3, %[LDC], t2      \n\t"
                "vse32.v        v29, (t2)           \n\t"
                "add            t2, %[LDC], t3      \n\t"
                "vse32.v        v30, (t3)           \n\t"
                "vse32.v        v31, (t2)           \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [LDC] "r"(ldc*4), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                  "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                  "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3");
#else
            asm volatile(
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vxor.vv        v28, v28, v28           \n\t"
                "vxor.vv        v29, v29, v29           \n\t"
                "vxor.vv        v30, v30, v30           \n\t"
                "vxor.vv        v31, v31, v31           \n\t"
                "vmv.v.i        v0, 1                   \n\t"  // init the scale
                "vsll.vi        v1, v0, 4               \n\t"
                "vfcvt.f.x.v    v0, v0                  \n\t"
                "vfcvt.f.x.v    v1, v1                  \n\t"
                "mv             t4, %[BK]               \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // load scale A
                "flw            fa0, (%[A])             \n\t"
                "flw            fa1, 4(%[A])            \n\t"
                "flw            fa2, 8(%[A])            \n\t"
                "flw            fa3, 12(%[A])           \n\t"
                "addi           %[A], %[A], 16          \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vle16.v        v12, (%[B])             \n\t"
                "addi           %[B], %[B], 64          \n\t"
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vpack.vv       v14, v12, v12, 3        \n\t"

                "vsetivli       t0, 4, e16, mf2         \n\t"
                "vle16.v        v8, (%[A])              \n\t"  // asum
                "addi           %[A], %[A], 8           \n\t"
                "vsll.vi        v8, v8, 3               \n\t"  // asum * 8
                "vfcvt.f.x.v    v9, v8                  \n\t"
                "vsetvli        t0, x0, e64, m1         \n\t"
                "vrgather.vi    v10, v9, 0              \n\t"

                "vsetvli        t0, x0, e8, m1          \n\t"
                "vl1r.v         v16, (%[A])             \n\t"
                "addi           %[A], %[A], 128         \n\t"  // 4*32@i8
                "vl4r.v         v4, (%[B])              \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512         \n\t"
                "vsrl.vi        v17, v16, 4             \n\t"
                "vnpack4.vv     v12, v16, v17, 3        \n\t"  // A low  u4
                "vupack.vv      v2, v12, v12, 2         \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vpack.vv       v16, v10, v10,0         \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vpack.vv       v20, v16, v16,0         \n\t"
                "vsetvli        t0, x0, e64, m1         \n\t"
                "vpack.vv       v18, v20, v20, 0        \n\t"
                "vor.vv         v20, v18, v18           \n\t"
                "vor.vv         v21, v18, v18           \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vmadotsu.hp    v18, v3, v4, v1, 0, i4  \n\t"  // high 4
                "vmadotsu.hp    v19, v3, v5, v1, 0, i4  \n\t"
                "vmadotsu.hp    v20, v3, v6, v1, 0, i4  \n\t"
                "vmadotsu.hp    v21, v3, v7, v1, 0, i4  \n\t"
                "vmadotu.hp     v18, v2, v4, v0, 0, i4  \n\t"  // low 4
                "vmadotu.hp     v19, v2, v5, v0, 0, i4  \n\t"
                "vmadotu.hp     v20, v2, v6, v0, 0, i4  \n\t"
                "vmadotu.hp     v21, v2, v7, v0, 0, i4  \n\t"

                "vpack.vv       v8, v18, v19, 1         \n\t"
                "vpack.vv       v12, v20, v21, 1        \n\t"
                "vpack.vv       v20, v8, v12, 2         \n\t"

                "vfwmul.vv      v16, v20, v14           \n\t"
                "vfwmul.vv      v18, v21, v14           \n\t"

                "vsetvli        t0, x0, e32, m1         \n\t"

                "addi           t4, t4, -1              \n\t"
                "vfmacc.vf      v28, fa0, v16           \n\t"
                "vfmacc.vf      v29, fa1, v17           \n\t"
                "vfmacc.vf      v30, fa2, v18           \n\t"
                "vfmacc.vf      v31, fa3, v19           \n\t"

                "bgtz           t4, BLK_LOOP%=          \n\t"

                // save
                "vsetvli        t0, x0, e32, m1         \n\t"
                "add            t2, %[LDC], %[DST]      \n\t"
                "vse32.v        v28, (%[DST])           \n\t"
                "add            t3, %[LDC], t2          \n\t"
                "vse32.v        v29, (t2)               \n\t"
                "add            t2, %[LDC], t3          \n\t"
                "vse32.v        v30, (t3)               \n\t"
                "vse32.v        v31, (t2)               \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                  "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3");
#endif
        }
    } else {
        for (size_t ni = 0; ni < count_n; ni += 32) {
            uint8_t * b_data = (uint8_t *) quant_b_data + ni * b_data_stride;
            int8_t *  a_data = (int8_t *) quant_a_ptr;
            float *   dst_c  = c_ptr + ni;

            asm volatile(
                "li             t1,  8          \n\t"
                "vsetvli        t0, x0, e32, m1 \n\t"
                "vxor.vv        v28, v28, v28   \n\t"
                "vxor.vv        v29, v29, v29   \n\t"
                "vxor.vv        v30, v30, v30   \n\t"
                "vxor.vv        v31, v31, v31   \n\t"
                "mv             t4, %[BK]       \n\t"

                ".align 4                        \n\t"
                "BLK_LOOP%=:                     \n\t"
                // load scale A
                "flw            fa0, (%[A])     \n\t"
                "flw            fa1, 4(%[A])    \n\t"
                "flw            fa2, 8(%[A])    \n\t"
                "flw            fa3, 12(%[A])   \n\t"
                "addi           %[A], %[A], 16  \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2\n\t"
                "vle16.v        v12, (%[B])     \n\t"
                "addi           %[B], %[B], 64  \n\t"
                "vfwcvt.f.f.v   v14, v12        \n\t"

                // load zp
                "vsetvli        t0, x0, e8, mf4 \n\t"
                "vle8.v         v8, (%[B])      \n\t"
                "addi           %[B], %[B], 32  \n\t"
                "vwaddu.vx      v10, v8, x0     \n\t"

                // load a sum
                "lh             s1, (%[A])      \n\t"
                "lh             s2, 2(%[A])     \n\t"
                "lh             s3, 4(%[A])     \n\t"
                "lh             s4, 6(%[A])     \n\t"
                "addi           %[A], %[A], 8   \n\t"

                "vsetvli        t0, x0, e8, m1  \n\t"
                "vl1r.v         v0, (%[A])      \n\t"
                "addi           %[A], %[A], 128 \n\t"  // 4*32@i8
                "vl4r.v         v4, (%[B])      \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512 \n\t"
                "vsrl.vi        v1, v0, 4       \n\t"
                "vnpack4.vv     v12, v0, v1, 3  \n\t"  // A low  u4
                "vupack.vv      v2, v12, v12, 2 \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e32, m1 \n\t"
                "vxor.vv        v16, v16, v16   \n\t"
                "vxor.vv        v18, v16, v16   \n\t"
                "vxor.vv        v20, v16, v16   \n\t"
                "vxor.vv        v22, v16, v16   \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e32, m1 \n\t"
                "vmadotsu       v16, v3, v4, i4 \n\t"  // high 4
                "vmadotsu       v18, v3, v5, i4 \n\t"
                "vmadotsu       v20, v3, v6, i4 \n\t"
                "vmadotsu       v22, v3, v7, i4 \n\t"
                "vsll.vi        v16, v16, 4     \n\t"
                "vsll.vi        v18, v18, 4     \n\t"
                "vsll.vi        v20, v20, 4     \n\t"
                "vsll.vi        v22, v22, 4     \n\t"
                "vmadotu        v16, v2, v4, i4 \n\t"  // low 4
                "vmadotu        v18, v2, v5, i4 \n\t"
                "vmadotu        v20, v2, v6, i4 \n\t"
                "vmadotu        v22, v2, v7, i4 \n\t"

                "vpack.vv       v0, v16, v18, 2 \n\t"
                "vpack.vv       v2, v20, v22, 2 \n\t"
                "vpack.vv       v16, v0, v2, 3  \n\t"
                "vpack.vv       v18, v1, v3, 3  \n\t"

                "vsetvli        t0, x0, e16, m1 \n\t"
                "vwmul.vx       v0, v10, s1     \n\t"
                "vwmul.vx       v2, v10, s2     \n\t"
                "vwmul.vx       v4, v10, s3     \n\t"
                "vwmul.vx       v6, v10, s4     \n\t"

                "vsetvli        t0, x0, e32, m1 \n\t"
                "vadd.vv        v16, v16, v0    \n\t"
                "vadd.vv        v17, v17, v2    \n\t"
                "vadd.vv        v18, v18, v4    \n\t"
                "vadd.vv        v19, v19, v6    \n\t"

                "vfcvt.f.x.v    v16, v16        \n\t"
                "vfcvt.f.x.v    v17, v17        \n\t"
                "vfcvt.f.x.v    v18, v18        \n\t"
                "vfcvt.f.x.v    v19, v19        \n\t"

                // mul scale
                "vfmul.vv       v16, v16, v14   \n\t"
                "vfmul.vv       v17, v17, v14   \n\t"
                "vfmul.vv       v18, v18, v14   \n\t"
                "vfmul.vv       v19, v19, v14   \n\t"

                "addi           t4, t4, -1      \n\t"
                "vfmacc.vf      v28, fa0, v16   \n\t"
                "vfmacc.vf      v29, fa1, v17   \n\t"
                "vfmacc.vf      v30, fa2, v18   \n\t"
                "vfmacc.vf      v31, fa3, v19   \n\t"

                "bgtz           t4, BLK_LOOP%=  \n\t"

                // save
                "vsetvli        t0, x0, e32, m1 \n\t"
                "add            t2, %[LDC], %[DST]\n\t"
                "vse32.v        v28, (%[DST])   \n\t"
                "add            t3, %[LDC], t2  \n\t"
                "vse32.v        v29, (t2)       \n\t"
                "add            t2, %[LDC], t3  \n\t"
                "vse32.v        v30, (t3)       \n\t"
                "vse32.v        v31, (t2)       \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                  "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3", "s1", "s2", "s3", "s4");
        }
    }
}

void quantize_a_row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride = q8_blk_size(blk_len, true);
    size_t  vlenb        = __riscv_vlenb();

    if (vlenb == 128) {
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            float *   scale_a_ptr = reinterpret_cast<float *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + sizeof(float));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + sizeof(float) + sizeof(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
            vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k, vl);
            vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
            float        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            float scale_a     = max_abs_a / ((1 << 7) - 1);
            float rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
            vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
            vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

            vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_sum, vl);
            int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
            a_sum_ptr[0]       = -a_sum;

            __riscv_vse8_v_i8mf4(quant_a_blk, v_a_quant_i8, vl);
        }
    } else if (vlenb == 32) {
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            float *   scale_a_ptr = reinterpret_cast<float *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + sizeof(float));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + sizeof(float) + sizeof(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
            vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + k, vl);
            vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
            float        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            float scale_a     = max_abs_a / ((1 << 7) - 1);
            float rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
            vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
            vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

            vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_sum, vl);
            int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
            a_sum_ptr[0]       = -a_sum;

            __riscv_vse8_v_i8m1(quant_a_blk, v_a_quant_i8, vl);
        }
    } else {
        quantize_a_nrow_i8_ref<1>(blk_len, a_ptr, count_k, quant_a_ptr);
    }
}

void quantize_a_4row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride        = q8_blk_size(blk_len, true);
    int64_t a_nrow_block_stride = a_blk_stride * 4;
    size_t  vlenb               = __riscv_vlenb();

    if (vlenb == 128) {
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            float *   scale_a_ptr = reinterpret_cast<float *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + sizeof(float) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + sizeof(float) * 4 + sizeof(int16_t) * 4);

            for (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * count_k + k, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                float        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                float scale_a     = max_abs_a / ((1 << 7) - 1);
                float rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_sum, vl);
                int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                a_sum_ptr[mi]      = -a_sum;

                __riscv_vse8_v_i8mf4(quant_a_blk + mi * blk_len, v_a_quant_i8, vl);
            }
        }
    } else if (vlenb == 32) {
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            float *   scale_a_ptr = reinterpret_cast<float *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + sizeof(float) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + sizeof(float) * 4 + sizeof(int16_t) * 4);

            for (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + mi * count_k + k, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                float        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                float scale_a     = max_abs_a / ((1 << 7) - 1);
                float rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
                vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
                vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

                vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_sum, vl);
                int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                a_sum_ptr[mi]      = -a_sum;

                __riscv_vse8_v_i8m1(quant_a_blk + mi * blk_len, v_a_quant_i8, vl);
            }
        }
    } else {
        quantize_a_nrow_i8_ref<4>(blk_len, a_ptr, count_k, quant_a_ptr);
    }
}

}  // namespace

#endif  // GGML_FLAGOS_SPACEMIT_IME2

//
// new: dispatch and the inverse repack
//

bool have_ime() {
#if defined(GGML_FLAGOS_SPACEMIT_IME2)
    return true;
#else
    return false;
#endif
}

void q4_0_quantize_row(const float * a, size_t count_k, uint8_t * qa, bool reference) {
#if defined(GGML_FLAGOS_SPACEMIT_IME2)
    if (!reference) {
        quantize_a_row_i8_hp(q4_0_k_block, a, count_k, qa);
        return;
    }
#endif
    quantize_a_nrow_i8_hp_ref<1>(q4_0_k_block, a, count_k, qa);
}

void q4_0_quantize_4rows(const float * a, size_t count_k, uint8_t * qa, bool reference) {
#if defined(GGML_FLAGOS_SPACEMIT_IME2)
    if (!reference) {
        quantize_a_4row_i8_hp(q4_0_k_block, a, count_k, qa);
        return;
    }
#endif
    quantize_a_nrow_i8_hp_ref<4>(q4_0_k_block, a, count_k, qa);
}

// same contract as gemm_kernel_i8i4_hp (ime2_kernels.cpp:5611): 4 rows when at least 4 remain, else 1
size_t q4_0_gemm(const uint8_t * qa, const uint8_t * qb, float * c, size_t count_m, size_t count_n, size_t k_blocks,
                 size_t ldc, bool reference) {
#if defined(GGML_FLAGOS_SPACEMIT_IME2)
    if (!reference) {
        if (count_m >= 4) {
            gemm_kernel_i8i4_hp_m4(q4_0_k_block, qa, qb, nullptr, c, count_m, count_n, k_blocks, ldc);
            return 4;
        }
        gemm_kernel_i8i4_hp_m1(q4_0_k_block, qa, qb, nullptr, c, count_m, count_n, k_blocks, ldc);
        return 1;
    }
#endif
    if (count_m >= 4) {
        gemm_kernel_i8i4_hp_mrow_ref<4, row_tile>(q4_0_k_block, qa, qb, nullptr, c, count_m, count_n, k_blocks, ldc);
        return 4;
    }
    gemm_kernel_i8i4_hp_mrow_ref<1, row_tile>(q4_0_k_block, qa, qb, nullptr, c, count_m, count_n, k_blocks, ldc);
    return 1;
}

void q4_1_quantize_row(const float * a, size_t count_k, uint8_t * qa, bool reference) {
#if defined(GGML_FLAGOS_SPACEMIT_IME2)
    if (!reference) {
        quantize_a_row_i8(q4_1_k_block, a, count_k, qa);
        return;
    }
#endif
    quantize_a_nrow_i8_ref<1>(q4_1_k_block, a, count_k, qa);
}

void q4_1_quantize_4rows(const float * a, size_t count_k, uint8_t * qa, bool reference) {
#if defined(GGML_FLAGOS_SPACEMIT_IME2)
    if (!reference) {
        quantize_a_4row_i8(q4_1_k_block, a, count_k, qa);
        return;
    }
#endif
    quantize_a_nrow_i8_ref<4>(q4_1_k_block, a, count_k, qa);
}

// same contract as gemm_kernel_i8i4 (ime2_kernels.cpp:5583). The kernels read the zero points from the weight blocks
// and use the zero-point pointer only as a flag (non-null: blocks with zero points), as ime.cpp passes it (b_col_zp).
// gemm_kernel_i8i4_m1's asm does not declare the vector registers it uses: call it only from functions like this one,
// which hold no vector values (as copy() does for memcpy1d)
size_t q4_1_gemm(const uint8_t * qa, const uint8_t * qb, float * c, size_t count_m, size_t count_n, size_t k_blocks,
                 size_t ldc, bool reference) {
#if defined(GGML_FLAGOS_SPACEMIT_IME2)
    if (!reference) {
        if (count_m >= 4) {
            gemm_kernel_i8i4_m4(q4_1_k_block, qa, qb, qb, c, count_m, count_n, k_blocks, ldc);
            return 4;
        }
        gemm_kernel_i8i4_m1(q4_1_k_block, qa, qb, qb, c, count_m, count_n, k_blocks, ldc);
        return 1;
    }
#endif
    if (count_m >= 4) {
        gemm_kernel_i8i4_mrow_ref<4, row_tile>(q4_1_k_block, qa, qb, qb, c, count_m, count_n, k_blocks, ldc);
        return 4;
    }
    gemm_kernel_i8i4_mrow_ref<1, row_tile>(q4_1_k_block, qa, qb, qb, c, count_m, count_n, k_blocks, ldc);
    return 1;
}

// never inlined: memcpy1d's inline asm does not declare the vector registers it overwrites, so it must stay behind a
// call, where the vector ABI makes every vector register caller-saved
__attribute__((noinline)) void copy(void * dst, const void * src, size_t size) {
#if defined(GGML_FLAGOS_SPACEMIT_IME2)
    memcpy1d(dst, src, (int64_t) size);
#else
    std::memcpy(dst, src, size);
#endif
}

int repack_q4_0(ggml_tensor * t, const void * data, size_t size) {
    return repack_q4_0_to_q4_0_256_32_bl_ref(t, 32, data, size);
}

// inverse of repack_q4_0_to_q4_0_256_32_bl_ref and make_block_q4_0x32: restores the GGUF Q4_0 bytes
void unpack_q4_0(const ggml_tensor * t, void * data) {
    const block_q4_0x32x256 * src     = static_cast<const block_q4_0x32x256 *>(t->data);
    block_q4_0 *              dst     = static_cast<block_q4_0 *>(data);
    const int64_t             nrow    = ggml_nrows(t);
    const int64_t             nblocks = t->ne[0] / QK4_0;

    for (int64_t b = 0; b < nrow; b += 32, dst += 32 * nblocks) {
        for (int64_t x = 0; x < nblocks; x += 8, src++) {
            for (int64_t j = 0; j < 8; j++) {
                const block_q4_0x32 & in = src->blocks[j];
                for (int64_t i = 0; i < 32; i++) {
                    block_q4_0 & out = dst[x + j + i * nblocks];
                    out.d            = in.d[i];
                    for (int64_t k = 0; k < QK4_0 / 4; k++) {
                        const uint8_t lo     = in.qs[i * QK4_0 / 2 + k];               // values 2k, 2k + 1
                        const uint8_t hi     = in.qs[i * QK4_0 / 2 + QK4_0 / 4 + k];  // values 2k + 16, 2k + 17
                        out.qs[2 * k]        = (uint8_t) ((lo & 0x0F) | ((hi & 0x0F) << 4));
                        out.qs[2 * k + 1]    = (uint8_t) ((lo >> 4) | (hi & 0xF0));
                    }
                }
            }
        }
    }
}

int repack_q4_1(ggml_tensor * t, const void * data, size_t size) {
    return repack_q4_1_to_q4_1_32_bl_ref(t, 32, data, size);
}

// inverse of repack_q4_1_to_q4_1_32_bl_ref and make_block_q4_1x32, up to the conversion they make: the integer zero point
// zp becomes the minimum m = -zp * d. The result is the Q4_1 tensor the kernels compute with, and repacking it gives the
// same zero points again (plan.md, M2c design)
void unpack_q4_1(const ggml_tensor * t, void * data) {
    const block_q4_1x32 * src     = static_cast<const block_q4_1x32 *>(t->data);
    block_q4_1 *          dst     = static_cast<block_q4_1 *>(data);
    const int64_t         nrow    = ggml_nrows(t);
    const int64_t         nblocks = t->ne[0] / QK4_1;

    for (int64_t b = 0; b < nrow; b += 32, dst += 32 * nblocks) {
        for (int64_t x = 0; x < nblocks; x++, src++) {
            for (int64_t i = 0; i < 32; i++) {
                block_q4_1 & out = dst[x + i * nblocks];
                const float  m   = -(float) src->zp[i] * GGML_FP16_TO_FP32(src->d[i]);
                out.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = src->d[i];
                out.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m = GGML_FP32_TO_FP16(m == 0.0f ? 0.0f : m);  // never -0
                for (int64_t k = 0; k < QK4_1 / 4; k++) {
                    const uint8_t lo  = src->qs[i * QK4_1 / 2 + k];               // values 2k, 2k + 1
                    const uint8_t hi  = src->qs[i * QK4_1 / 2 + QK4_1 / 4 + k];  // values 2k + 16, 2k + 17
                    out.qs[2 * k]     = (uint8_t) ((lo & 0x0F) | ((hi & 0x0F) << 4));
                    out.qs[2 * k + 1] = (uint8_t) ((lo >> 4) | (hi & 0xF0));
                }
            }
        }
    }
}

}  // namespace spacemit_ime
