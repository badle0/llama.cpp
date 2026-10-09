// RVV kernels for the element-wise and row ops of a layer (plan.md, M2d design), K3 builds only.
//
// Copied from ggml-spacemit (spacemit-com/llama.cpp @ 4e782bc, ggml/src/ggml-spacemit/rvv_kernels.cpp; identical in its
// branch mtmd-backend @ 64316cd): rvv_expf_approx_f32m2 (57-88), forward_rms_norm_f32 (1643-1719),
// forward_binary (2635-2854), forward_get_rows (3057-3109), forward_rope_impl (3943-4146),
// forward_set_rows (4156-4209), forward_glu_swiglu_f32 (4211-4260).
// Changes, each marked "flagos": the functions take this file's context, which has no sync(), so a kernel cannot wait
// inside a step; RMS_NORM accepts eps 0 (its assert aborted ggml-spacemit's X0 run); GET_ROWS and SET_ROWS report an
// out-of-range index as a failed step instead of aborting, and GET_ROWS copies with M2b's copy of the same memcpy1d
// and skips an empty column range (it computed a negative size there); RMS_NORM accumulates tail-undisturbed (_tu).
// The flagos-spacemit-kernels.cpp entry points decide when these apply; otherwise they use the portable references.

#include "flagos-spacemit-rvv-kernels.h"

#if defined(GGML_FLAGOS_SPACEMIT_RVV)

#    include "flagos-spacemit-ime-kernels.h"

#    include "../../../ggml-impl.h"

#    include <riscv_vector.h>

#    include <algorithm>
#    include <cassert>
#    include <cmath>
#    include <cstring>
#    include <type_traits>

#    if defined(__GNUC__)
#        pragma GCC diagnostic ignored "-Wcast-qual"
#        pragma GCC diagnostic ignored "-Wunused-parameter"
#    endif

namespace spacemit_rvv {

constexpr size_t cache_line_size_f32 = 64 / sizeof(float);  // ggml::spacemit::cache_line_size_f32 (spacemit-context.h)

namespace {

// Adapted from ggml_v_expf_m2 in vec.h. This is accurate enough for softmax.
static inline vfloat32m2_t rvv_expf_approx_f32m2(vfloat32m2_t x, size_t vl) {
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

}  // namespace

void forward_rms_norm_f32(context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    int ith = ctx.ith;
    int nth = ctx.nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    float epsilon = *((float *) dst->op_params);

    // flagos: eps 0 is allowed, as in ggml-cpu (this assert aborted ggml-spacemit's X0 run)

    auto * input  = (char *) src0->data;
    auto * output = (char *) dst->data;

    const auto hidden_size     = ne00;
    const auto task_count      = ne01 * ne02 * ne03;
    const auto task_per_thread = (task_count + nth - 1) / nth;

    const auto task_begin = ith * task_per_thread;
    const auto task_end   = std::min((ith + 1) * task_per_thread, task_count);

    for (auto task_idx = task_begin; task_idx < task_end; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        auto * p_input       = (float *) (input + i01 * nb01 + i02 * nb02 + i03 * nb03);
        auto * p_output      = (float *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        auto * p_temp_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t sum_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      length = hidden_size;
        while (length > 0) {
            gvl                   = __riscv_vsetvl_e32m4(length);
            vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_input, gvl);
            // flagos: _tu keeps the lanes past a short final chunk, which the full-width reduction below adds
            sum_sq                = __riscv_vfmacc_vv_f32m4_tu(sum_sq, src_data, src_data, gvl);
            __riscv_vse32_v_f32m4(p_temp_output, src_data, gvl);

            p_input += gvl;
            p_temp_output += gvl;
            length -= gvl;
        }

        gvl                 = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        vfloat32m1_t mean_square_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(sum_sq, 0), __riscv_vget_v_f32m4_f32m1(sum_sq, 1), gvl);

        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 2), gvl);
        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 3), gvl);
        mean_square_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_square_v, zero_v, gvl);

        // flagos note (code unchanged): the sum is in float, ggml-cpu's in double, so a row whose RMS exceeds about
        // sqrt(FLT_MAX / ne00) (3.6e17 for 2560) gives 0 here and finite values on the CPU; activations stay far below
        float mean_square = __riscv_vfmv_f_s_f32m1_f32(mean_square_v);
        mean_square /= hidden_size;

        mean_square = sqrt(mean_square + epsilon);

        mean_square   = 1.0f / mean_square;
        length        = hidden_size;
        p_temp_output = p_output;

        while (length > 0) {
            gvl                   = __riscv_vsetvl_e32m4(length);
            vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_temp_output, gvl);
            src_data              = __riscv_vfmul_vf_f32m4(src_data, mean_square, gvl);
            __riscv_vse32_v_f32m4(p_output, src_data, gvl);
            p_temp_output += gvl;
            p_output += gvl;
            length -= gvl;
        }
    }
}

template <ggml_op op_type, typename T> void forward_binary(context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    auto src0_rows = ggml_nrows(src0);
    auto src1_rows = ggml_nrows(src1);

    int ith = ctx.ith;
    int nth = ctx.nth;

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(nb0 == sizeof(T));
    GGML_ASSERT(nb00 == sizeof(T));

    const int64_t nr  = ggml_nrows(src0);
    const int64_t dr  = (nr + nth - 1) / nth;
    const int64_t ir0 = dr * ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    auto compute_func_vv = [&](int64_t blk_len, int64_t r, T * src0_ptr, T * src1_ptr, T * dst_ptr) {
        int64_t idx = 0;
        if constexpr (op_type == GGML_OP_ADD) {
            if constexpr (std::is_same_v<T, float>) {
                for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfadd_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } else if constexpr (std::is_same_v<T, _Float16>) {
                for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfadd_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } else {
                GGML_ABORT("fatal error");
            }
        } else if constexpr (op_type == GGML_OP_SUB) {
            if constexpr (std::is_same_v<T, float>) {
                for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfsub_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } else if constexpr (std::is_same_v<T, _Float16>) {
                for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfsub_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } else {
                GGML_ABORT("fatal error");
            }
        } else if constexpr (op_type == GGML_OP_MUL) {
            if constexpr (std::is_same_v<T, float>) {
                for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfmul_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } else if constexpr (std::is_same_v<T, _Float16>) {
                for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfmul_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } else {
                GGML_ABORT("fatal error");
            }
        } else if constexpr (op_type == GGML_OP_DIV) {
            if constexpr (std::is_same_v<T, float>) {
                for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfdiv_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } else if constexpr (std::is_same_v<T, _Float16>) {
                for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfdiv_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } else {
                GGML_ABORT("fatal error");
            }
        } else {
            GGML_ABORT("fatal error");
        }
    };

    if (src0_rows == src1_rows && src0_rows == 1 && ne00 == ne10) {
        int64_t task_per_thread = (ne00 + nth - 1) / nth;
        int64_t task_begin      = ith * task_per_thread;
        int64_t task_end        = std::min((ith + 1) * task_per_thread, ne00);

        T * dst_ptr  = ((T *) dst->data) + task_begin;
        T * src0_ptr = ((T *) src0->data) + task_begin;
        T * src1_ptr = ((T *) src1->data) + task_begin;

        compute_func_vv(task_end - task_begin, 0, src0_ptr, src1_ptr, dst_ptr);
    } else if (ne10 > 1) {
        for (int64_t ir = ir0; ir < ir1; ++ir) {
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((char *) dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((char *) src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((char *) src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

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

            T * dst_ptr  = (T *) ((char *) dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((char *) src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((char *) src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

            T       rhs_scalar = src1_ptr[0];
            int64_t blk_len    = ne00;
            int64_t r          = 0;

            for (size_t vl; blk_len > 0; blk_len -= vl, r += vl) {
                if constexpr (op_type == GGML_OP_ADD) {
                    if constexpr (std::is_same_v<T, float>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfadd_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } else if constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfadd_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } else {
                        GGML_ABORT("fatal error");
                    }
                } else if constexpr (op_type == GGML_OP_SUB) {
                    if constexpr (std::is_same_v<T, float>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfsub_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } else if constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfsub_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } else {
                        GGML_ABORT("fatal error");
                    }
                } else if constexpr (op_type == GGML_OP_MUL) {
                    if constexpr (std::is_same_v<T, float>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfmul_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } else if constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfmul_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } else {
                        GGML_ABORT("fatal error");
                    }
                } else if constexpr (op_type == GGML_OP_DIV) {
                    if constexpr (std::is_same_v<T, float>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfdiv_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } else if constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfdiv_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } else {
                        GGML_ABORT("fatal error");
                    }
                } else {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

template <typename T> bool forward_get_rows(context & ctx, ggml_tensor * op) {  // flagos: bool, see below
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = ggml_nelements(src1);

    assert(ne0 == nc);
    assert(ne02 == ne11);
    assert(nb00 == sizeof(T));
    assert(ggml_nrows(op) == nr);

    const int ith = ctx.ith;
    const int nth = ctx.nth;

    int rows_nth = nth;
    int cols_nth = 1;

    if (nr == 1) {
        rows_nth = 1;
        cols_nth = nth;
    }

    // rows per thread
    const int dr = (nr + rows_nth - 1) / rows_nth;
    const int dc = (nc + cols_nth - 1) / cols_nth;

    int rows_ith = ith % rows_nth;
    int cols_ith = ith % cols_nth;

    // row range for this thread
    const int ir0 = dr * rows_ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const int cr0 = dc * cols_ith;
    const int cr1 = MIN(cr0 + dc, nc);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i / (ne11 * ne10);
        const int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
        const int64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->data + i10 * nb10 + i11 * nb11 + i12 * nb12);

        if (i01 < 0 || i01 >= ne01) {  // flagos: a failed step instead of an abort
            return false;
        }

        if (cr1 > cr0) {  // flagos: with one row and few columns a tile's range can start past the end
            spacemit_ime::copy(((char *) dst->data + i10 * nb1 + i11 * nb2 + i12 * nb3) + cr0 * sizeof(T),
                     ((char *) src0->data + i01 * nb01 + i11 * nb02 + i12 * nb03) + cr0 * sizeof(T),
                     (cr1 - cr0) * sizeof(T));
        }
    }
    return true;
}

template <typename T>
static void forward_rope_impl(context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0         = op->src[0];
    const ggml_tensor * src1         = op->src[1];
    const ggml_tensor * src2         = op->src[2];
    const float *       freq_factors = src2 ? (const float *) src2->data : nullptr;

    const int n_dims     = ggml_get_op_params_i32(op, 1);
    const int mode       = ggml_get_op_params_i32(op, 2);
    const int n_ctx_orig = ggml_get_op_params_i32(op, 4);

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   op->op_params + 5,  sizeof(float));
    memcpy(&freq_scale,  op->op_params + 6,  sizeof(float));
    memcpy(&ext_factor,  op->op_params + 7,  sizeof(float));
    memcpy(&attn_factor, op->op_params + 8,  sizeof(float));
    memcpy(&beta_fast,   op->op_params + 9,  sizeof(float));
    memcpy(&beta_slow,   op->op_params + 10, sizeof(float));

    int sections[4] = {0, 0, 0, 0};
    memcpy(sections, op->op_params + 11, sizeof(int) * 4);

    const bool mrope_used = (mode & GGML_ROPE_TYPE_MROPE) != 0;
    const bool is_imrope  = (mode == GGML_ROPE_TYPE_IMROPE);

    // YaRN correction dims
    float corr_dims[2] = { 0.0f, 0.0f };
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    const int64_t ne2 = op->ne[2];  // seq-len dimension
    const int64_t nr  = ggml_nrows(op);
    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);
    const int32_t * pos = (const int32_t *) src1->data;
    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    const int sec_w     = sections[0] + sections[1];
    const int sec_e     = sec_w + sections[2];

    // The small F32 NEOX shape used by Qwen3-0.6B is latency-sensitive and
    // was faster on the original scalar path.  Keep that path shape-based
    // (never model-name based), while larger rotary dimensions use the RVV
    // implementation below.
    const bool legacy_small_f32 = std::is_same_v<T, float> &&
        mode == GGML_ROPE_TYPE_NEOX && op->ne[0] <= 128 &&
        src0->nb[0] == sizeof(float);
    float legacy_cache[512];
    float * cache = legacy_small_f32 ? legacy_cache : nullptr;
    if (!legacy_small_f32) {
        GGML_ASSERT(ctx.workspace_size >= (size_t) ctx.nth * (size_t) (op->ne[0] + cache_line_size_f32) * sizeof(float));
        cache = (float *) ctx.workspace +
                ctx.ith * (op->ne[0] + cache_line_size_f32);
    }
    int64_t last_i2 = -1;

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i3 = ir / (op->ne[2] * op->ne[1]);
        const int64_t i2 = (ir / op->ne[1]) % op->ne[2];
        const int64_t i1 = ir % op->ne[1];

        if (i2 != last_i2) {
            if (!mrope_used) {
                // Standard / YaRN rope
                float theta = (float) pos[i2];
                for (int i0 = 0; i0 < n_dims; i0 += 2) {
                    const float ff           = freq_factors ? freq_factors[i0 / 2] : 1.0f;
                    const float theta_extrap = theta / ff;
                    const float theta_interp = freq_scale * theta_extrap;
                    float t;
                    if (ext_factor != 0.0f) {
                        const float y        = (i0 / 2 - corr_dims[0]) / fmaxf(0.001f, corr_dims[1] - corr_dims[0]);
                        const float ramp_mix = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
                        t = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                    } else {
                        t = theta_interp;
                    }
                    float s, c;
                    s = sinf(t);
                    c = cosf(t);
                    cache[i0 + 0] = c * mscale;
                    cache[i0 + 1] = s * mscale;
                    theta *= theta_scale;
                }
            } else {
                // M-RoPE / IMROPE: multiple position sequences in src1
                const float p_t = (float) pos[i2];
                const float p_h = (float) pos[i2 + ne2];
                const float p_w = (float) pos[i2 + ne2 * 2];
                const float p_e = (float) pos[i2 + ne2 * 3];

                float theta_t = p_t;
                float theta_h = p_h;
                float theta_w = p_w;
                float theta_e = p_e;

                for (int i0 = 0; i0 < n_dims; i0 += 2) {
                    const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
                    int sector = (sect_dims > 0) ? (i0 / 2) % sect_dims : 0;

                    float theta;
                    if (is_imrope) {
                        if      (sector % 3 == 0 && sector < 3 * sections[0]) theta = theta_t;
                        else if (sector % 3 == 1 && sector < 3 * sections[1]) theta = theta_h;
                        else if (sector % 3 == 2 && sector < 3 * sections[2]) theta = theta_w;
                        else                                                    theta = theta_e;
                    } else {
                        if      (sector < sections[0])  theta = theta_t;
                        else if (sector < sec_w)         theta = theta_h;
                        else if (sector < sec_e)         theta = theta_w;
                        else                             theta = theta_e;
                    }

                    const float theta_extrap = theta / ff;
                    const float theta_interp = freq_scale * theta_extrap;
                    float t;
                    if (ext_factor != 0.0f) {
                        const float y        = (i0 / 2 - corr_dims[0]) / fmaxf(0.001f, corr_dims[1] - corr_dims[0]);
                        const float ramp_mix = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
                        t = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                    } else {
                        t = theta_interp;
                    }
                    float s, c;
                    s = sinf(t);
                    c = cosf(t);
                    cache[i0 + 0] = c * mscale;
                    cache[i0 + 1] = s * mscale;

                    theta_t *= theta_scale;
                    theta_h *= theta_scale;
                    theta_w *= theta_scale;
                    theta_e *= theta_scale;
                }
            }
            last_i2 = i2;
        }

        const T * src = (const T *) ((const char *) src0->data + i3 * src0->nb[3] + i2 * src0->nb[2] + i1 * src0->nb[1]);
        T * dst_row   = (T *) ((char *) op->data + i3 * op->nb[3] + i2 * op->nb[2] + i1 * op->nb[1]);

        if (legacy_small_f32) {
            // Preserve the original low-dimensional F32 NEOX implementation.
            const int offset = n_dims / 2;
            for (int i0 = 0; i0 < n_dims; i0 += 2) {
                const int ic = i0 / 2;
                const float x0 = src[ic];
                const float x1 = src[ic + offset];
                dst_row[ic]          = x0 * cache[i0] - x1 * cache[i0 + 1];
                dst_row[ic + offset] = x0 * cache[i0 + 1] + x1 * cache[i0];
            }
        } else {
        // NEOX / MROPE / IMROPE use half-offset rotation.  Qwen3.5 uses
        // contiguous F32 rows here; process all pairs with RVV loads instead
        // of the old scalar element loop.  Keep NORMAL and non-F32 layouts on
        // the reference path because their pair mapping differs.
        if constexpr (std::is_same_v<T, float>) {
            if (mode == GGML_ROPE_TYPE_NEOX || mode == GGML_ROPE_TYPE_MROPE || mode == GGML_ROPE_TYPE_IMROPE) {
                const int64_t pairs = n_dims / 2;
                int64_t j = 0;
                while (j < pairs) {
                    const size_t vl = __riscv_vsetvl_e32m4((size_t) (pairs - j));
                    const vfloat32m4_t x0 = __riscv_vle32_v_f32m4(src + j, vl);
                    const vfloat32m4_t x1 = __riscv_vle32_v_f32m4(src + j + pairs, vl);
                    const vfloat32m4_t c = __riscv_vlse32_v_f32m4(cache + 2*j, (ptrdiff_t) (2*sizeof(float)), vl);
                    const vfloat32m4_t s = __riscv_vlse32_v_f32m4(cache + 2*j + 1, (ptrdiff_t) (2*sizeof(float)), vl);
                    const vfloat32m4_t y0 = __riscv_vfsub_vv_f32m4(__riscv_vfmul_vv_f32m4(x0, c, vl),
                                                                    __riscv_vfmul_vv_f32m4(x1, s, vl), vl);
                    const vfloat32m4_t y1 = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(x0, s, vl),
                                                                    __riscv_vfmul_vv_f32m4(x1, c, vl), vl);
                    __riscv_vse32_v_f32m4(dst_row + j, y0, vl);
                    __riscv_vse32_v_f32m4(dst_row + j + pairs, y1, vl);
                    j += vl;
                }
            } else {
                for (int i0 = 0; i0 < n_dims; i0 += 2) {
                    const float x0 = src[i0];
                    const float x1 = src[i0 + 1];
                    dst_row[i0] = x0 * cache[i0] - x1 * cache[i0 + 1];
                    dst_row[i0 + 1] = x0 * cache[i0 + 1] + x1 * cache[i0];
                }
            }
        } else {
            const int offset = n_dims / 2;
            for (int i0 = 0; i0 < n_dims; i0 += 2) {
                const int ic = (mode == GGML_ROPE_TYPE_NORMAL) ? i0 : i0 / 2;
                const int no = (mode == GGML_ROPE_TYPE_NORMAL) ? 1 : offset;
                const float x0 = (float) src[ic];
                const float x1 = (float) src[ic + no];
                dst_row[ic]          = (T) (x0 * cache[i0] - x1 * cache[i0 + 1]);
                dst_row[ic + no] = (T) (x0 * cache[i0 + 1] + x1 * cache[i0]);
            }
        }
        }
        for (int64_t i0 = n_dims; i0 < op->ne[0]; ++i0) {
            dst_row[i0] = src[i0];
        }
    }
}

bool forward_set_rows(context & ctx, ggml_tensor * op) {  // flagos: bool, see below
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    const int64_t nc  = src0->ne[0];
    const int64_t nr  = src0->ne[1];
    const int64_t dc  = (nc + ctx.nth - 1) / ctx.nth;
    const int64_t c0  = dc * ctx.ith;
    const int64_t c1  = MIN(c0 + dc, nc);

    // Every core owns a disjoint column range but visits source rows in the
    // same order. If indices repeat, this preserves SET_ROWS' last-row-wins
    // semantics without cross-core write races.
    for (int64_t i03 = 0; i03 < src0->ne[3]; ++i03) {
        for (int64_t i02 = 0; i02 < src0->ne[2]; ++i02) {
            const int64_t i11 = i02 % src1->ne[1];
            const int64_t i12 = i03 % src1->ne[2];
            for (int64_t i = 0; i < nr; ++i) {
                const int64_t idx = src1->type == GGML_TYPE_I64
                        ? *(const int64_t *) ((const char *) src1->data + i * src1->nb[0] + i11 * src1->nb[1] + i12 * src1->nb[2])
                        : *(const int32_t *) ((const char *) src1->data + i * src1->nb[0] + i11 * src1->nb[1] + i12 * src1->nb[2]);
                if (idx < 0 || idx >= dst->ne[1]) {  // flagos: a failed step instead of an abort
                    return false;
                }

                const char * src = (const char *) src0->data + i * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3];
                char * out = (char *) dst->data + idx * dst->nb[1] + i02 * dst->nb[2] + i03 * dst->nb[3];

                int64_t c = c0;
                while (c < c1) {
                    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                        const size_t vl = __riscv_vsetvl_e32m4(c1 - c);
                        const vfloat32m4_t v32 = __riscv_vle32_v_f32m4((const float *) src + c, vl);
                        __riscv_vse16_v_f16m2((_Float16 *) out + c, __riscv_vfncvt_f_f_w_f16m2(v32, vl), vl);
                        c += vl;
                    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
                        const size_t vl = __riscv_vsetvl_e16m2(c1 - c);
                        const vfloat16m2_t v16 = __riscv_vle16_v_f16m2((const _Float16 *) src + c, vl);
                        __riscv_vse32_v_f32m4((float *) out + c, __riscv_vfwcvt_f_f_v_f32m4(v16, vl), vl);
                        c += vl;
                    } else if (src0->type == GGML_TYPE_F32) {
                        const size_t vl = __riscv_vsetvl_e32m4(c1 - c);
                        __riscv_vse32_v_f32m4((float *) out + c, __riscv_vle32_v_f32m4((const float *) src + c, vl), vl);
                        c += vl;
                    } else {
                        const size_t vl = __riscv_vsetvl_e16m2(c1 - c);
                        __riscv_vse16_v_f16m2((_Float16 *) out + c,
                                              __riscv_vle16_v_f16m2((const _Float16 *) src + c, vl), vl);
                        c += vl;
                    }
                }
            }
        }
    }
    return true;
}

void forward_glu_swiglu_f32(context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);

    const int64_t nc      = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int64_t nr      = ggml_nrows(src0);
    const int64_t total   = nr * nc;
    const int64_t dr      = (total + ctx.nth - 1) / ctx.nth;
    const int64_t e0      = dr * ctx.ith;
    const int64_t e1      = MIN(e0 + dr, total);
    const int32_t swapped = ggml_get_op_params_i32(op, 1);

    int64_t e = e0;
    while (e < e1) {
        const int64_t r   = e / nc;
        const int64_t c   = e % nc;
        const int64_t run = MIN(nc - c, e1 - e);

        const float * x_row = (const float *) ((const char *) src0->data + r * src0->nb[1]);
        const float * g_row;
        if (src1) {
            g_row = (const float *) ((const char *) src1->data + r * src1->nb[1]);
        } else {
            x_row += swapped ? nc : 0;
            g_row = (const float *) ((const char *) src0->data + r * src0->nb[1]) + (swapped ? 0 : nc);
        }
        float * y_row = (float *) ((char *) op->data + r * op->nb[1]);

        const float * xp = x_row + c;
        const float * gp = g_row + c;
        float *       yp = y_row + c;
        int64_t remaining = run;
        while (remaining > 0) {
            const size_t vl = __riscv_vsetvl_e32m2(remaining);
            const vfloat32m2_t x = __riscv_vle32_v_f32m2(xp, vl);
            const vfloat32m2_t g = __riscv_vle32_v_f32m2(gp, vl);
            const vfloat32m2_t exp_neg_x = rvv_expf_approx_f32m2(__riscv_vfneg_v_f32m2(x, vl), vl);
            const vfloat32m2_t silu = __riscv_vfdiv_vv_f32m2(
                    x, __riscv_vfadd_vf_f32m2(exp_neg_x, 1.0f, vl), vl);
            __riscv_vse32_v_f32m2(yp, __riscv_vfmul_vv_f32m2(silu, g, vl), vl);
            xp += vl;
            gp += vl;
            yp += vl;
            remaining -= vl;
        }
        e += run;
    }
}

//
// flagos: entry points and the instances the provider uses
//

bool binary_applies(const ggml_tensor * op) {
    return ggml_is_contiguous_rows(op->src[1]);  // forward_binary reads src1 rows as contiguous elements
}

template void forward_binary<GGML_OP_ADD, float>(context & ctx, ggml_tensor * op);
template void forward_binary<GGML_OP_MUL, float>(context & ctx, ggml_tensor * op);

void forward_rope_f32(context & ctx, ggml_tensor * op) {
    forward_rope_impl<float>(ctx, op);
}

bool forward_get_rows_f32(context & ctx, ggml_tensor * op) {
    return forward_get_rows<int32_t>(ctx, op);  // copies 4-byte elements
}

}  // namespace spacemit_rvv

#endif  // GGML_FLAGOS_SPACEMIT_RVV
