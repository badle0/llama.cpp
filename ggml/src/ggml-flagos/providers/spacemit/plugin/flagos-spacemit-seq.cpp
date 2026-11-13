#include "flagos-spacemit-seq.h"

#include "../../../../ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// Reference kernels for the Gated DeltaNet layers' remaining ops (plan.md "Step 2 spec", pieces D to G). Each follows
// ggml-cpu's own code (ggml/src/ggml-cpu @ ba360ef) with the rows split over the tiles, so the arithmetic order is
// ggml-cpu's where ggml-cpu is scalar there. No RVV variant: see the note on SSM_CONV below.

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
// the kernel's row, summed in float in ggml-cpu's order, so the result is ggml-cpu's bit for bit.
// No RVV variant: SpacemiT's own forward_ssm_conv_f32 that we can read (the fork's PR 36, ggml-spacemit's
// rvv_kernels.cpp) is this same scalar loop split over rows, and the vectorized and CONCAT-fused forms their newer
// header declares are in none of the fork's published branches; the fused one also needs graph fusion (a concat that
// materializes only its trailing columns), which this provider does not do
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

}  // namespace

bool spacemit_kernel_silu_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
    unary_f32_ref<op_silu>(*tile_p, dst);
    return true;
}

bool spacemit_kernel_sigmoid_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
    unary_f32_ref<op_sigmoid>(*tile_p, dst);
    return true;
}

bool spacemit_kernel_softplus_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
    unary_f32_ref<op_softplus>(*tile_p, dst);
    return true;
}

bool spacemit_kernel_l2_norm_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
    l2_norm_f32_ref(*tile_p, dst);
    return true;
}

bool spacemit_kernel_ssm_conv_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
    ssm_conv_f32_ref(*tile_p, dst);
    return true;
}

bool spacemit_kernel_concat_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
    concat_dim0_f32_ref(*tile_p, dst);
    return true;
}

bool spacemit_kernel_mul_mat_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
    mul_mat_f32_ref(*tile_p, dst);
    return true;
}
