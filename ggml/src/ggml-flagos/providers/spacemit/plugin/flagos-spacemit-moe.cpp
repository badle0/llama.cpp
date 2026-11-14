// The MoE router's kernels (Qwen3.6-35B-A3B, build_moe_ffn in src/llama-graph.cpp): the contracts soft_max.f32,
// argsort.f32, sum_rows.f32, clamp.f32 and div.f32 (core/flagos-spacemit-ops.cpp).
//
// The references follow ggml-cpu's own code (ggml/src/ggml-cpu/ops.cpp, binary-ops.cpp, vec.h), with the rows split
// over the tiles as ggml-cpu splits them over its threads; each output row comes from one source row on one tile, so
// the result does not depend on the number of tiles. ARGSORT, CLAMP and DIV are ggml-cpu's bits; SUM_ROWS is where
// ggml-cpu sums in double (not with Accelerate, vDSP_sve); SOFT_MAX is ggml-cpu's scalar loop, whose exp and sum
// ggml-cpu vectorizes on most hosts (ggml_vec_soft_max_f32), so it is close but not its bits.
// One row is one tile's work: a single token's softmax or sort (one row of 256 experts) runs on one tile.
//
// The RVV variants are SpacemiT's ggml-spacemit (branch mtmd-backend @ ddfb41ddc, ggml/src/ggml-spacemit/rvv_kernels.cpp):
// forward_soft_max_plain_f32 (4467-4545) with rvv_softmax_exp_inplace_f32 (232-245), reduce_sum_f32m2_vlen1024
// (136-140) and rvv_expf_approx_f32m2 (143-180); forward_sum_rows<float> (3128-3208); the F32 DIV paths of
// forward_binary (2907-3126). Changes, each marked "flagos": the row range comes from the tile; SOFT_MAX takes the
// maximum of the scaled row (the fork scaled the source's maximum, wrong for a negative scale); SUM_ROWS accumulates
// tail-undisturbed (_tu), since its last pass of fewer than VLMAX elements left the other lanes undefined and the
// reduction reads every lane. ARGSORT and CLAMP have no RVV body in the fork (scalar_kernels.cpp: std::sort, and a
// clamp by std::min and std::max, which keeps a NaN where ggml-cpu's macros give the upper bound), so the references
// run there too.

#include "flagos-spacemit-moe.h"

#include "flagos-spacemit-kernels.h"  // spacemit_use_reference

#if defined(GGML_FLAGOS_SPACEMIT_RVV)
#    include <riscv_vector.h>
#endif

#include "../../../../ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

// [begin, end) of n items for one tile, the way ggml-cpu splits rows over its threads (a copy of
// flagos-spacemit-kernels.cpp's file-local helper, kept here so this file stands alone)
void tile_range(int64_t n, const spacemit_tile & tile, int64_t & begin, int64_t & end) {
    const int64_t per = (n + tile.nth - 1) / tile.nth;
    begin             = std::min(n, per * tile.ith);
    end               = std::min(n, begin + per);
}

// row ir of a tensor with ne[1..3] rows, as ggml-cpu numbers them
void row_index(int64_t ir, const ggml_tensor * t, int64_t & i1, int64_t & i2, int64_t & i3) {
    i3 = ir / (t->ne[2] * t->ne[1]);
    i2 = (ir - i3 * t->ne[2] * t->ne[1]) / t->ne[1];
    i1 = ir - i3 * t->ne[2] * t->ne[1] - i2 * t->ne[1];
}

// ops.cpp ggml_compute_forward_soft_max_f32 without a mask or sinks: the row copied and scaled (ggml_vec_cpy_f32,
// ggml_vec_scale_f32), its maximum (ggml_vec_max_f32), exp(x - max) summed in double (ggml_float) (the scalar tail of
// ggml_vec_soft_max_f32), then one multiply by 1/sum as a float (ggml_vec_scale_f32)
void soft_max_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_TENSOR_UNARY_OP_LOCALS

    float scale;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    int64_t ir0, ir1;
    tile_range(ne01 * ne02 * ne03, tile, ir0, ir1);
    for (int64_t ir = ir0; ir < ir1; ++ir) {
        int64_t i1, i2, i3;
        row_index(ir, src0, i1, i2, i3);
        const float * sp = (const float *) ((const char *) src0->data + i1 * nb01 + i2 * nb02 + i3 * nb03);
        float *       dp = (float *) ((char *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3);

        for (int64_t i = 0; i < ne00; ++i) {
            dp[i] = sp[i] * scale;
        }
        float max = -INFINITY;
        for (int64_t i = 0; i < ne00; ++i) {
            max = MAX(max, dp[i]);
        }
        double     sum = 0.0;
        for (int64_t i = 0; i < ne00; ++i) {
            const float val = expf(dp[i] - max);
            sum += (double) val;
            dp[i] = val;
        }
        sum             = 1.0 / sum;
        const float inv = (float) sum;
        for (int64_t i = 0; i < ne00; ++i) {
            dp[i] *= inv;
        }
    }
}

// ops.cpp ggml_compute_forward_argsort_f32: the indices sorted by std::sort with ggml-cpu's comparator, so ties and
// NaNs come out in ggml-cpu's order (the same library sort on the same comparisons)
struct argsort_cmp {
    const float * data;
    bool          desc;

    bool operator()(int32_t a, int32_t b) const { return desc ? data[a] > data[b] : data[a] < data[b]; }
};

void argsort_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_TENSOR_UNARY_OP_LOCALS

    const bool desc = ggml_get_op_params_i32(dst, 0) == GGML_SORT_ORDER_DESC;

    int64_t ir0, ir1;
    tile_range(ne01 * ne02 * ne03, tile, ir0, ir1);
    for (int64_t ir = ir0; ir < ir1; ++ir) {
        int64_t i1, i2, i3;
        row_index(ir, src0, i1, i2, i3);
        const float * x = (const float *) ((const char *) src0->data + i1 * nb01 + i2 * nb02 + i3 * nb03);
        int32_t *     y = (int32_t *) ((char *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3);
        for (int64_t j = 0; j < ne0; j++) {
            y[j] = (int32_t) j;
        }
        if (desc) {
            std::sort(y, y + ne0, argsort_cmp{ x, true });
        } else {
            std::sort(y, y + ne0, argsort_cmp{ x, false });
        }
    }
}

// ops.cpp ggml_compute_forward_sum_rows_f32 with vec.h's ggml_vec_sum_f32: the row summed in double (ggml_float) in order
void sum_rows_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_TENSOR_UNARY_OP_LOCALS

    int64_t ir0, ir1;
    tile_range(ne01 * ne02 * ne03, tile, ir0, ir1);
    for (int64_t ir = ir0; ir < ir1; ++ir) {
        int64_t i1, i2, i3;
        row_index(ir, src0, i1, i2, i3);
        const float * x   = (const float *) ((const char *) src0->data + i1 * nb01 + i2 * nb02 + i3 * nb03);
        double        sum = 0.0;
        for (int64_t i = 0; i < ne00; ++i) {
            sum += (double) x[i];
        }
        *(float *) ((char *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3) = (float) sum;
    }
}

// ops.cpp ggml_compute_forward_clamp_f32: MAX(MIN(x, max), min) with ggml's macros, so a NaN gives max
void clamp_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_TENSOR_UNARY_OP_LOCALS

    float mn;
    float mx;
    std::memcpy(&mn, (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&mx, (const float *) dst->op_params + 1, sizeof(float));

    int64_t ir0, ir1;
    tile_range(ne01 * ne02 * ne03, tile, ir0, ir1);
    for (int64_t ir = ir0; ir < ir1; ++ir) {
        int64_t i1, i2, i3;
        row_index(ir, src0, i1, i2, i3);
        const float * x = (const float *) ((const char *) src0->data + i1 * nb01 + i2 * nb02 + i3 * nb03);
        float *       y = (float *) ((char *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3);
        for (int64_t i = 0; i < ne00; ++i) {
            y[i] = MAX(MIN(x[i], mx), mn);
        }
    }
}

// binary-ops.cpp apply_binary_op with op_div, F32: src1 repeats over src0 in every dimension (the loop of
// flagos-spacemit-kernels.cpp's binary_f32_ref). One IEEE division per element, so ggml-cpu's bits on every path
void div_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_contiguous_rows = ggml_is_contiguous_rows(src1);
    int64_t    ir0, ir1;
    tile_range(ne01 * ne02 * ne03, tile, ir0, ir1);
    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir / (ne02 * ne01);
        const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;
        const int64_t i13 = i03 % ne13;
        const int64_t i12 = i02 % ne12;
        const int64_t i11 = i01 % ne11;

        float *       z = (float *) ((char *) dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
        const float * x = (const float *) ((const char *) src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
        const char *  y = (const char *) src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
        if (src1_contiguous_rows) {
            for (int64_t r = 0; r < ne00 / ne10; ++r) {
                for (int64_t i = 0; i < ne10; ++i) {
                    z[r * ne10 + i] = x[r * ne10 + i] / ((const float *) y)[i];
                }
            }
        } else {
            for (int64_t i = 0; i < ne0; ++i) {
                z[i] = x[i] / *(const float *) (y + (i % ne10) * nb10);
            }
        }
    }
}

#if defined(GGML_FLAGOS_SPACEMIT_RVV)

inline float reduce_sum_f32m2_vlen1024(vfloat32m2_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m2_f32m1(v, s_v, vl);
    return __riscv_vfmv_f_s_f32m1_f32(s_v);
}

// Adapted from ggml_v_expf_m2 in vec.h. This is accurate enough for softmax.
inline vfloat32m2_t rvv_expf_approx_f32m2(vfloat32m2_t x, size_t vl) {
    const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
    const vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
    const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    const vfloat32m2_t b =
        __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
    const vuint32m2_t  e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
    const vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl));
    const vbool16_t    c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
    const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    const vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
        __riscv_vfmacc_vv_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl), 0x1.555e66p-3f, b, vl),
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl), 0x1.0e4020p-7f, b, vl), u, vl),
        u, vl);

    if (!__riscv_vcpop_m_b16(c, vl)) {
        return __riscv_vfmacc_vv_f32m2(k, j, k, vl);
    }

    const vbool16_t    dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
    const vuint32m2_t  d  = __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
    const vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
    const vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
    const vfloat32m2_t r1 =
        __riscv_vmerge_vvm_f32m2(__riscv_vfmacc_vv_f32m2(k, k, j, vl),
                                 __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl), c, vl);
    return __riscv_vmerge_vvm_f32m2(r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
                                    __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl), vl);
}

inline float rvv_softmax_exp_inplace_f32(float * dst, int64_t n, float max_value) {
    float row_sum = 0.0f;
    while (n > 0) {
        const size_t vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst, vl);
        v               = __riscv_vfsub_vf_f32m2(v, max_value, vl);
        v               = rvv_expf_approx_f32m2(v, vl);
        __riscv_vse32_v_f32m2(dst, v, vl);
        row_sum += reduce_sum_f32m2_vlen1024(v, vl);
        dst += vl;
        n -= vl;
    }
    return row_sum;
}

// forward_soft_max_plain_f32
void soft_max_f32_rvv(const spacemit_tile & tile, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    GGML_TENSOR_UNARY_OP_LOCALS

    float scale;
    std::memcpy(&scale, (const float *) op->op_params + 0, sizeof(float));

    const int64_t ncols = ne00;
    int64_t       row0, row1;
    tile_range(ne01 * ne02 * ne03, tile, row0, row1);  // flagos: the tile's rows

    for (int64_t row = row0; row < row1; row++) {
        const int64_t i03 = row / (ne02 * ne01);
        const int64_t i02 = (row - i03 * ne02 * ne01) / ne01;
        const int64_t i01 = row - i03 * ne02 * ne01 - i02 * ne01;

        const float * src_row = (const float *) ((const char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
        float *       dst_row = (float *) ((char *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        // copy src*scale into dst
        {
            int64_t       n  = ncols;
            const float * sp = src_row;
            float *       dp = dst_row;
            while (n > 0) {
                const size_t vl = __riscv_vsetvl_e32m4(n);
                vfloat32m4_t v  = __riscv_vle32_v_f32m4(sp, vl);
                v               = __riscv_vfmul_vf_f32m4(v, scale, vl);
                __riscv_vse32_v_f32m4(dp, v, vl);
                sp += vl;
                dp += vl;
                n -= vl;
            }
        }
        // flagos: the maximum of the scaled row, after the copy (the fork took the source's and scaled it)
        float max_val = -INFINITY;
        {
            int64_t       n = ncols;
            const float * p = dst_row;
            while (n > 0) {
                const size_t vl      = __riscv_vsetvl_e32m4(n);
                vfloat32m4_t v       = __riscv_vle32_v_f32m4(p, vl);
                vfloat32m1_t red     = __riscv_vfredmax_vs_f32m4_f32m1(v, __riscv_vfmv_v_f_f32m1(-INFINITY, 1), vl);
                float        cur_max = __riscv_vfmv_f_s_f32m1_f32(red);
                if (cur_max > max_val) {
                    max_val = cur_max;
                }
                p += vl;
                n -= vl;
            }
        }
        const float sum = rvv_softmax_exp_inplace_f32(dst_row, ncols, max_val);

        // normalize
        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        {
            int64_t n  = ncols;
            float * dp = dst_row;
            while (n > 0) {
                const size_t vl = __riscv_vsetvl_e32m4(n);
                vfloat32m4_t v  = __riscv_vle32_v_f32m4(dp, vl);
                v               = __riscv_vfmul_vf_f32m4(v, inv_sum, vl);
                __riscv_vse32_v_f32m4(dp, v, vl);
                dp += vl;
                n -= vl;
            }
        }
    }
}

// forward_sum_rows<float>
void sum_rows_f32_rvv(const spacemit_tile & tile, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    GGML_TENSOR_UNARY_OP_LOCALS

    int64_t ir_start, ir_end;
    tile_range(ne01 * ne02 * ne03, tile, ir_start, ir_end);  // flagos: the tile's rows

    for (int64_t ir = ir_start; ir < ir_end; ir++) {
        const int64_t i3 = ir / (ne02 * ne01);
        const int64_t i2 = (ir - i3 * ne02 * ne01) / ne01;
        const int64_t i1 = (ir - i3 * ne02 * ne01 - i2 * ne01);

        const float * src_row = (const float *) ((const char *) src0->data + i1 * nb01 + i2 * nb02 + i3 * nb03);
        float *       dst_row = (float *) ((char *) op->data + i1 * nb1 + i2 * nb2 + i3 * nb3);

        size_t        gvl     = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t  acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
        int64_t       length  = ne00;
        const float * p_data  = src_row;

        while (length > 0) {
            size_t       vl  = __riscv_vsetvl_e32m4(length);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(p_data, vl);
            acc_vec          = __riscv_vfadd_vv_f32m4_tu(acc_vec, acc_vec, vec, vl);  // flagos: tail-undisturbed
            p_data += vl;
            length -= vl;
        }

        gvl                 = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.0f, gvl);
        vfloat32m1_t sum_v  = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(acc_vec, 0),
                                                     __riscv_vget_v_f32m4_f32m1(acc_vec, 1), gvl);
        sum_v               = __riscv_vfadd_vv_f32m1(sum_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 2), gvl);
        sum_v               = __riscv_vfadd_vv_f32m1(sum_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 3), gvl);
        sum_v               = __riscv_vfredusum_vs_f32m1_f32m1(sum_v, zero_v, gvl);
        dst_row[0]          = __riscv_vfmv_f_s_f32m1_f32(sum_v);
    }
}

// forward_binary<GGML_OP_DIV, float>: its three paths (one row of src1's length, src1 rows of several elements, a
// scalar src1 per row); src1's rows must be contiguous (the caller checks, as binary_applies)
void div_f32_rvv(const spacemit_tile & tile, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    const int64_t src0_rows = ggml_nrows(src0);
    const int64_t src1_rows = ggml_nrows(src1);

    GGML_TENSOR_BINARY_OP_LOCALS

    int64_t ir0, ir1;
    tile_range(src0_rows, tile, ir0, ir1);  // flagos: the tile's rows

    auto compute_func_vv = [&](int64_t blk_len, int64_t r, const float * src0_ptr, const float * src1_ptr,
                               float * dst_ptr) {
        int64_t idx = 0;
        for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
            vl               = __riscv_vsetvl_e32m4(blk_len);
            vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
            vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
            vfloat32m4_t res = __riscv_vfdiv_vv_f32m4(lhs, rhs, vl);
            __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
        }
    };

    if (src0_rows == src1_rows && src0_rows == 1 && ne00 == ne10) {
        int64_t task_begin, task_end;
        tile_range(ne00, tile, task_begin, task_end);  // flagos: the tile's columns

        float *       dst_ptr  = ((float *) dst->data) + task_begin;
        const float * src0_ptr = ((const float *) src0->data) + task_begin;
        const float * src1_ptr = ((const float *) src1->data) + task_begin;

        compute_func_vv(task_end - task_begin, 0, src0_ptr, src1_ptr, dst_ptr);
    } else if (ne10 > 1) {
        for (int64_t ir = ir0; ir < ir1; ++ir) {
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float *       dst_ptr  = (float *) ((char *) dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
            const float * src0_ptr = (const float *) ((const char *) src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
            const float * src1_ptr = (const float *) ((const char *) src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

            // src1 is broadcastable across src0 and dst in i1, i2, i3
            for (int64_t r = 0; r < ne00; r += ne10) {
                compute_func_vv(ne10, r, src0_ptr, src1_ptr, dst_ptr);
            }
        }
    } else {
        for (int64_t ir = ir0; ir < ir1; ++ir) {
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float *       dst_ptr  = (float *) ((char *) dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
            const float * src0_ptr = (const float *) ((const char *) src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
            const float * src1_ptr = (const float *) ((const char *) src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

            const float rhs_scalar = src1_ptr[0];
            int64_t     blk_len    = ne00;
            int64_t     r          = 0;

            for (size_t vl; blk_len > 0; blk_len -= vl, r += vl) {
                vl               = __riscv_vsetvl_e32m4(blk_len);
                vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                vfloat32m4_t res = __riscv_vfdiv_vf_f32m4(lhs, rhs_scalar, vl);
                __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
            }
        }
    }
}

#endif  // GGML_FLAGOS_SPACEMIT_RVV

}  // namespace

bool spacemit_kernel_soft_max_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference()) {
        soft_max_f32_rvv(*tile_p, dst);
        return true;
    }
#endif
    soft_max_f32_ref(*tile_p, dst);
    return true;
}

bool spacemit_kernel_argsort_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
    argsort_f32_ref(*tile_p, dst);
    return true;
}

bool spacemit_kernel_sum_rows_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference()) {
        sum_rows_f32_rvv(*tile_p, dst);
        return true;
    }
#endif
    sum_rows_f32_ref(*tile_p, dst);
    return true;
}

bool spacemit_kernel_clamp_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
    clamp_f32_ref(*tile_p, dst);
    return true;
}

bool spacemit_kernel_div_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference() && ggml_is_contiguous_rows(dst->src[1])) {
        div_f32_rvv(*tile_p, dst);
        return true;
    }
#endif
    div_f32_ref(*tile_p, dst);
    return true;
}
