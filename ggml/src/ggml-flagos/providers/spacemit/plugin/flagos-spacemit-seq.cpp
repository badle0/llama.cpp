#include "flagos-spacemit-seq.h"

#include "flagos-spacemit-kernels.h"  // spacemit_use_reference

#if defined(GGML_FLAGOS_SPACEMIT_RVV)
#    include <riscv_vector.h>
#endif

#include "../../../../ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// Reference kernels for the Gated DeltaNet layers' remaining ops (plan.md "Step 2 spec", pieces D to G). Each follows
// ggml-cpu's own code (ggml/src/ggml-cpu @ ba360ef) with the rows split over the tiles, so the arithmetic order is
// ggml-cpu's where ggml-cpu is scalar there.
//
// RVV variants (on the K3, unless GGML_FLAGOS_KERNEL_VARIANTS=0 keeps the references) for SILU, SIGMOID, SOFTPLUS,
// SSM_CONV and CONCAT are ported from SpacemiT's fork (ggml-spacemit's rvv_kernels.cpp, branch mtmd-backend @
// ddfb41ddc); the one for L2_NORM is ours. Each takes the same rows of the same tile as its reference. Changes to the
// ported code are marked "flagos". MUL_MAT has no RVV variant.

namespace {

// [begin, end) of n items for one tile, as flagos-spacemit-kernels.cpp splits them (ggml-cpu's row split)
void tile_range(int64_t n, const spacemit_tile & tile, int64_t & begin, int64_t & end) {
    const int64_t per = (n + tile.nth - 1) / tile.nth;
    begin             = std::min(n, per * tile.ith);
    end               = std::min(n, begin + per);
}

// one element, as ggml-cpu computes it: SILU is vec.h's ggml_silu_f32 (the scalar tail of ggml_vec_silu_f32, whose
// vector part uses an exp approximation on some platforms), SIGMOID and SOFTPLUS are unary-ops.cpp's op_sigmoid and
// op_softplus, which ggml-cpu runs scalar everywhere
float op_silu(float x) {
    return x / (1.0f + expf(-x));
}

float op_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float op_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

// unary-ops.cpp apply_unary_op<op, float, float>: rows contiguous, any row strides; the rows split over the tiles
template <float (*op)(float)> void unary_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_TENSOR_UNARY_OP_LOCALS

    int64_t ir0, ir1;
    tile_range(ne01 * ne02 * ne03, tile, ir0, ir1);
    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir / (ne02 * ne01);
        const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        const float * x = (const float *) ((const char *) src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
        float *       y = (float *) ((char *) dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
        for (int64_t i = 0; i < ne0; ++i) {
            y[i] = op(x[i]);
        }
    }
}

// ops.cpp ggml_compute_forward_l2_norm_f32: the sum of squares in ggml_float (double), then one multiply per element
// (ggml_vec_scale_f32), so the result is ggml-cpu's bit for bit
void l2_norm_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_TENSOR_UNARY_OP_LOCALS

    float eps;
    std::memcpy(&eps, dst->op_params, sizeof(float));

    int64_t ir0, ir1;
    tile_range(ne01 * ne02 * ne03, tile, ir0, ir1);
    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir / (ne02 * ne01);
        const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        const float * x = (const float *) ((const char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
        float *       y = (float *) ((char *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        double sum = 0.0;
        for (int64_t i00 = 0; i00 < ne00; i00++) {
            sum += (double) (x[i00] * x[i00]);
        }
        const float scale = 1.0f / fmaxf(sqrtf((float) sum), eps);
        for (int64_t i00 = 0; i00 < ne00; i00++) {
            y[i00] = x[i00] * scale;
        }
    }
}

// ops.cpp ggml_compute_forward_ssm_conv_f32, copied with its row split: one dot product of the sliding window with
// the kernel's row, summed in float in ggml-cpu's order, so the result is ggml-cpu's bit for bit
void ssm_conv_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // conv_x {d_conv - 1 + n_t, d_inner, n_s}
    const ggml_tensor * src1 = dst->src[1];  // conv1d.weight {d_conv, d_inner}

    const int64_t nc  = src1->ne[0];  // d_conv
    const int64_t ncs = src0->ne[0];  // d_conv - 1 + n_t
    const int64_t nr  = src0->ne[1];  // d_inner
    const int64_t n_t = dst->ne[1];   // tokens per sequence
    const int64_t n_s = dst->ne[2];   // sequences in the batch

    int64_t ir0, ir1;
    tile_range(nr, tile, ir0, ir1);
    for (int64_t i3 = 0; i3 < n_s; ++i3) {
        for (int64_t i2 = 0; i2 < n_t; ++i2) {
            const float * s = (const float *) ((const char *) src0->data + ir0 * src0->nb[1] + i2 * src0->nb[0] +
                                               i3 * src0->nb[2]);
            const float * c = (const float *) ((const char *) src1->data + ir0 * src1->nb[1]);
            float *       x = (float *) ((char *) dst->data + ir0 * dst->nb[0] + i2 * dst->nb[1] + i3 * dst->nb[2]);

            for (int64_t i1 = 0; i1 < ir1 - ir0; ++i1) {
                float sumf = 0.0f;
                for (int64_t i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1 * ncs] * c[i0 + i1 * nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

// ops.cpp ggml_compute_forward_concat_f32 for dimension 0 (spacemit_concat_f32_supported): each output element is one
// source element, read with ggml-cpu's strides (the model's second source is a transposed view), the destination's
// rows split over the tiles. A copy, so the destination's bytes are ggml-cpu's
void concat_dim0_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    GGML_TENSOR_BINARY_OP_LOCALS

    int64_t ir0, ir1;
    tile_range(ne1 * ne2 * ne3, tile, ir0, ir1);
    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i3 = ir / (ne2 * ne1);
        const int64_t i2 = (ir / ne1) % ne2;
        const int64_t i1 = ir % ne1;

        float *      y = (float *) ((char *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3);
        const char * a = (const char *) src0->data + i1 * nb01 + i2 * nb02 + i3 * nb03;
        const char * b = (const char *) src1->data + i1 * nb11 + i2 * nb12 + i3 * nb13;
        for (int64_t i0 = 0; i0 < ne00; ++i0) {
            y[i0] = *(const float *) (a + i0 * nb00);
        }
        for (int64_t i0 = 0; i0 < ne10; ++i0) {
            y[ne00 + i0] = *(const float *) (b + i0 * nb10);
        }
    }
}

// MUL_MAT of an F32 2-D weight by F32 activations (spacemit_mul_mat_f32_supported): one dot product per output
// element, a weight row against an activation row, summed in float; the weight broadcasts over the activations'
// higher dimensions. The output's rows are split over the tiles. ggml-cpu adds in vector lanes (ggml_vec_dot_f32) or
// through llamafile's sgemm, so these sums are not its bits, only within test-backend-ops' bound
void mul_mat_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // the weight [K, N]
    const ggml_tensor * src1 = dst->src[1];  // the activations [K, rows, ...]
    GGML_TENSOR_BINARY_OP_LOCALS

    int64_t ir0, ir1;
    tile_range(ne1 * ne2 * ne3, tile, ir0, ir1);
    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i3 = ir / (ne2 * ne1);
        const int64_t i2 = (ir / ne1) % ne2;
        const int64_t i1 = ir % ne1;

        const float * x = (const float *) ((const char *) src1->data + i1 * nb11 + i2 * nb12 + i3 * nb13);
        float *       y = (float *) ((char *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3);
        for (int64_t r = 0; r < ne0; ++r) {
            const float * w   = (const float *) ((const char *) src0->data + r * nb01);
            float         sum = 0.0f;
            for (int64_t k = 0; k < ne00; ++k) {
                sum += w[k] * x[k];
            }
            y[r] = sum;
        }
    }
}

#if defined(GGML_FLAGOS_SPACEMIT_RVV)

// SpacemiT's rvv_expf_approx_f32m2, copied; the same operations as ggml-cpu's ggml_v_expf_m2 (vec.h), ARM's optimized
// routine: n = round(x / ln2) by the 0x1.8p23 shifter, b = x - n ln2 in two fused steps, exp(b) - 1 by a degree-5
// polynomial, 2^n built in the exponent bits; |n| > 126 scales in two steps, |n| > 192 gives inf or 0
vfloat32m2_t rvv_expf_f32m2(vfloat32m2_t x, size_t vl) {
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

// SpacemiT's rvv_log1pf_01_f32m2, copied: log1p(t) for t in [0, 1] by the Cephes logf polynomial, the rounding of 1 + t
// added back as a first-order correction
vfloat32m2_t rvv_log1pf_01_f32m2(vfloat32m2_t t, size_t vl) {
    const vfloat32m2_t u   = __riscv_vfadd_vf_f32m2(t, 1.0f, vl);
    const vfloat32m2_t err = __riscv_vfdiv_vv_f32m2(
        __riscv_vfsub_vv_f32m2(t, __riscv_vfsub_vf_f32m2(u, 1.0f, vl), vl), u, vl);
    // u in [1, 2]: reduce to m in [sqrt(2)/2, sqrt(2)], u = m * 2^e with e in {0, 1}
    const vbool16_t    big = __riscv_vmfgt_vf_f32m2_b16(u, 0x1.6a09e6p+0f, vl);
    const vfloat32m2_t m   = __riscv_vmerge_vvm_f32m2(u, __riscv_vfmul_vf_f32m2(u, 0.5f, vl), big, vl);
    const vfloat32m2_t e   = __riscv_vfmerge_vfm_f32m2(__riscv_vfmv_v_f_f32m2(0.0f, vl), 1.0f, big, vl);
    const vfloat32m2_t f   = __riscv_vfsub_vf_f32m2(m, 1.0f, vl);
    const vfloat32m2_t z   = __riscv_vfmul_vv_f32m2(f, f, vl);
    vfloat32m2_t       p   = __riscv_vfmv_v_f_f32m2(7.0376836292E-2f, vl);
    p = __riscv_vfmadd_vv_f32m2(p, f, __riscv_vfmv_v_f_f32m2(-1.1514610310E-1f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, f, __riscv_vfmv_v_f_f32m2(1.1676998740E-1f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, f, __riscv_vfmv_v_f_f32m2(-1.2420140846E-1f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, f, __riscv_vfmv_v_f_f32m2(1.4249322787E-1f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, f, __riscv_vfmv_v_f_f32m2(-1.6668057665E-1f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, f, __riscv_vfmv_v_f_f32m2(2.0000714765E-1f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, f, __riscv_vfmv_v_f_f32m2(-2.4999993993E-1f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, f, __riscv_vfmv_v_f_f32m2(3.3333331174E-1f, vl), vl);
    vfloat32m2_t y = __riscv_vfmul_vv_f32m2(__riscv_vfmul_vv_f32m2(p, f, vl), z, vl);  // f^3 P(f)
    y              = __riscv_vfmacc_vf_f32m2(y, -2.12194440e-4f, e, vl);
    y              = __riscv_vfmacc_vf_f32m2(y, -0.5f, z, vl);
    y              = __riscv_vfadd_vv_f32m2(y, err, vl);
    return __riscv_vfmacc_vf_f32m2(__riscv_vfadd_vv_f32m2(f, y, vl), 0.693359375f, e, vl);
}

// SpacemiT's forward_unary_silu_f32 and its DEFINE_UNARY_F32_RVV bodies for SIGMOID and SOFTPLUS, one vector each
struct rvv_silu {
    static vfloat32m2_t apply(vfloat32m2_t x, size_t vl) {
        const vfloat32m2_t exp_neg = rvv_expf_f32m2(__riscv_vfneg_v_f32m2(x, vl), vl);
        return __riscv_vfdiv_vv_f32m2(x, __riscv_vfadd_vf_f32m2(exp_neg, 1.0f, vl), vl);
    }
};

struct rvv_sigmoid {
    static vfloat32m2_t apply(vfloat32m2_t v, size_t vl) {
        return __riscv_vfdiv_vv_f32m2(
            __riscv_vfmv_v_f_f32m2(1.0f, vl),
            __riscv_vfadd_vf_f32m2(rvv_expf_f32m2(__riscv_vfneg_v_f32m2(v, vl), vl), 1.0f, vl), vl);
    }
};

// softplus(x) = max(x, 0) + log1p(exp(-|x|)): no cut-off at 20 as in ggml-cpu's op_softplus, which returns x there
struct rvv_softplus {
    static vfloat32m2_t apply(vfloat32m2_t v, size_t vl) {
        return __riscv_vfadd_vv_f32m2(
            __riscv_vfmax_vf_f32m2(v, 0.0f, vl),
            rvv_log1pf_01_f32m2(rvv_expf_f32m2(__riscv_vfneg_v_f32m2(__riscv_vfabs_v_f32m2(v, vl), vl), vl), vl), vl);
    }
};

// flagos: the reference's rows (any row strides, the rows split over the tiles) instead of the fork's flat elements
// of a contiguous tensor; each lane's result depends on its own input only, so neither the split nor vl changes it
template <typename op> void unary_f32_rvv(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_TENSOR_UNARY_OP_LOCALS

    int64_t ir0, ir1;
    tile_range(ne01 * ne02 * ne03, tile, ir0, ir1);
    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir / (ne02 * ne01);
        const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        const float * x = (const float *) ((const char *) src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
        float *       y = (float *) ((char *) dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
        for (int64_t i = 0; i < ne0;) {
            const size_t vl = __riscv_vsetvl_e32m2(ne0 - i);
            __riscv_vse32_v_f32m2(y + i, op::apply(__riscv_vle32_v_f32m2(x + i, vl), vl), vl);
            i += (int64_t) vl;
        }
    }
}

// ours, not a port (SpacemiT's L2_NORM is a scalar loop): the reference with the squares in vector lanes. Each square
// is rounded to float and added in double, as ggml-cpu does, but lane by lane and then across the lanes, so the sum
// can differ from ggml-cpu's in its last double bits and, rarely, the float scale by an ulp
void l2_norm_f32_rvv(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_TENSOR_UNARY_OP_LOCALS

    float eps;
    std::memcpy(&eps, dst->op_params, sizeof(float));

    const size_t vlmax = __riscv_vsetvlmax_e64m4();
    int64_t      ir0, ir1;
    tile_range(ne01 * ne02 * ne03, tile, ir0, ir1);
    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir / (ne02 * ne01);
        const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        const float * x = (const float *) ((const char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
        float *       y = (float *) ((char *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        vfloat64m4_t acc = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
        for (int64_t i = 0; i < ne00;) {
            const size_t       vl = __riscv_vsetvl_e32m2(ne00 - i);
            const vfloat32m2_t v  = __riscv_vle32_v_f32m2(x + i, vl);
            acc                   = __riscv_vfwadd_wv_f64m4_tu(acc, acc, __riscv_vfmul_vv_f32m2(v, v, vl), vl);
            i += (int64_t) vl;
        }
        vfloat64m1_t s   = __riscv_vfmv_v_f_f64m1(0.0, 1);
        s                = __riscv_vfredusum_vs_f64m4_f64m1(acc, s, vlmax);
        const double sum = __riscv_vfmv_f_s_f64m1_f64(s);

        const float scale = 1.0f / fmaxf(sqrtf((float) sum), eps);
        for (int64_t i = 0; i < ne00;) {
            const size_t vl = __riscv_vsetvl_e32m2(ne00 - i);
            __riscv_vse32_v_f32m2(y + i, __riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(x + i, vl), scale, vl), vl);
            i += (int64_t) vl;
        }
    }
}

// SpacemiT's forward_ssm_conv_f32 for d_conv 4 (false for other widths, which take the reference): the channels in
// vector lanes, the 4-wide window kept in registers, one strided load per token, the channels split over the tiles as
// the reference splits them. flagos: the first tap is a fused multiply-add onto zero instead of a multiply, so a lane
// is fma(x3, w3, fma(x2, w2, fma(x1, w1, fma(x0, w0, 0)))), the reference's loop once the compiler contracts it (the
// two differ only in the sign of an all-zero sum otherwise)
bool ssm_conv_f32_rvv(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // conv_x {d_conv - 1 + n_t, d_inner, n_s}
    const ggml_tensor * src1 = dst->src[1];  // conv1d.weight {d_conv, d_inner}

    const int64_t nc  = src1->ne[0];
    const int64_t nr  = src0->ne[1];
    const int64_t n_t = dst->ne[1];
    const int64_t n_s = dst->ne[2];

    if (nc != 4) {
        return false;
    }

    const ptrdiff_t xs = (ptrdiff_t) src0->nb[1];  // channel stride of x
    const ptrdiff_t ws = (ptrdiff_t) src1->nb[1];  // channel stride of w

    int64_t ir0, ir1;
    tile_range(nr, tile, ir0, ir1);
    for (int64_t i3 = 0; i3 < n_s; ++i3) {
        for (int64_t c = ir0; c < ir1;) {
            const size_t vl = __riscv_vsetvl_e32m4(ir1 - c);

            const char *       wp = (const char *) src1->data + c * ws;
            const vfloat32m4_t w0 = __riscv_vlse32_v_f32m4((const float *) wp + 0, ws, vl);
            const vfloat32m4_t w1 = __riscv_vlse32_v_f32m4((const float *) wp + 1, ws, vl);
            const vfloat32m4_t w2 = __riscv_vlse32_v_f32m4((const float *) wp + 2, ws, vl);
            const vfloat32m4_t w3 = __riscv_vlse32_v_f32m4((const float *) wp + 3, ws, vl);

            const float * xp = (const float *) ((const char *) src0->data + c * xs + i3 * src0->nb[2]);
            vfloat32m4_t  x0 = __riscv_vlse32_v_f32m4(xp + 0, xs, vl);
            vfloat32m4_t  x1 = __riscv_vlse32_v_f32m4(xp + 1, xs, vl);
            vfloat32m4_t  x2 = __riscv_vlse32_v_f32m4(xp + 2, xs, vl);

            for (int64_t t = 0; t < n_t; ++t) {
                const vfloat32m4_t x3 = __riscv_vlse32_v_f32m4(xp + t + 3, xs, vl);

                vfloat32m4_t acc = __riscv_vfmacc_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.0f, vl), x0, w0, vl);
                acc              = __riscv_vfmacc_vv_f32m4(acc, x1, w1, vl);
                acc              = __riscv_vfmacc_vv_f32m4(acc, x2, w2, vl);
                acc              = __riscv_vfmacc_vv_f32m4(acc, x3, w3, vl);

                float * y = (float *) ((char *) dst->data + c * sizeof(float) + t * dst->nb[1] + i3 * dst->nb[2]);
                __riscv_vse32_v_f32m4(y, acc, vl);

                x0 = x1;
                x1 = x2;
                x2 = x3;
            }
            c += (int64_t) vl;
        }
    }
    return true;
}

// SpacemiT's forward_concat_dim0_transposed_f32 with col0 0, for its form only (false otherwise, and the reference
// runs): the first source's elements contiguous, the second a transposed view whose dimension 1 is contiguous. A run of
// channels is read with one load per column and stored down the destination's rows. flagos: the tile's rows are the
// reference's (dimensions 1 to 3 flattened), cut where a (i2, i3) plane ends, not the fork's channels per plane
bool concat_dim0_transposed_f32_rvv(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    GGML_TENSOR_BINARY_OP_LOCALS

    if (nb00 != sizeof(float) || nb11 != sizeof(float) || nb10 == sizeof(float)) {
        return false;
    }

    int64_t ir0, ir1;
    tile_range(ne1 * ne2 * ne3, tile, ir0, ir1);
    for (int64_t ir = ir0; ir < ir1;) {
        const int64_t i3 = ir / (ne2 * ne1);
        const int64_t i2 = (ir / ne1) % ne2;
        const int64_t c0 = ir % ne1;
        const int64_t c1 = std::min(ne1, c0 + (ir1 - ir));

        const char * s0 = (const char *) src0->data + i2 * nb02 + i3 * nb03;
        const char * s1 = (const char *) src1->data + i2 * nb12 + i3 * nb13;
        char *       d  = (char *) dst->data + i2 * nb2 + i3 * nb3;

        for (int64_t c = c0; c < c1;) {
            const size_t vl = __riscv_vsetvl_e32m4(c1 - c);
            for (int64_t i0 = 0; i0 < ne00; i0++) {
                const vfloat32m4_t v =
                    __riscv_vlse32_v_f32m4((const float *) (s0 + i0 * nb00 + c * nb01), (ptrdiff_t) nb01, vl);
                __riscv_vsse32_v_f32m4((float *) (d + i0 * nb0 + c * nb1), (ptrdiff_t) nb1, v, vl);
            }
            for (int64_t t = 0; t < ne10; t++) {
                const vfloat32m4_t v = __riscv_vle32_v_f32m4((const float *) (s1 + t * nb10 + c * nb11), vl);
                __riscv_vsse32_v_f32m4((float *) (d + (ne00 + t) * nb0 + c * nb1), (ptrdiff_t) nb1, v, vl);
            }
            c += (int64_t) vl;
        }
        ir += c1 - c0;
    }
    return true;
}

#endif  // GGML_FLAGOS_SPACEMIT_RVV

}  // namespace

bool spacemit_kernel_silu_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference()) {
        unary_f32_rvv<rvv_silu>(*tile_p, dst);
        return true;
    }
#endif
    unary_f32_ref<op_silu>(*tile_p, dst);
    return true;
}

bool spacemit_kernel_sigmoid_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference()) {
        unary_f32_rvv<rvv_sigmoid>(*tile_p, dst);
        return true;
    }
#endif
    unary_f32_ref<op_sigmoid>(*tile_p, dst);
    return true;
}

bool spacemit_kernel_softplus_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference()) {
        unary_f32_rvv<rvv_softplus>(*tile_p, dst);
        return true;
    }
#endif
    unary_f32_ref<op_softplus>(*tile_p, dst);
    return true;
}

bool spacemit_kernel_l2_norm_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference()) {
        l2_norm_f32_rvv(*tile_p, dst);
        return true;
    }
#endif
    l2_norm_f32_ref(*tile_p, dst);
    return true;
}

bool spacemit_kernel_ssm_conv_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference() && ssm_conv_f32_rvv(*tile_p, dst)) {
        return true;
    }
#endif
    ssm_conv_f32_ref(*tile_p, dst);
    return true;
}

bool spacemit_kernel_concat_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference() && concat_dim0_transposed_f32_rvv(*tile_p, dst)) {
        return true;
    }
#endif
    concat_dim0_f32_ref(*tile_p, dst);
    return true;
}

bool spacemit_kernel_mul_mat_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
    mul_mat_f32_ref(*tile_p, dst);
    return true;
}
