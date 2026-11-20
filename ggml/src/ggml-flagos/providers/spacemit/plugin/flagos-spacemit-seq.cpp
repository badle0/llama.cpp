#include "flagos-spacemit-seq.h"

#include "flagos-spacemit-kernels.h"  // spacemit_use_reference

#if defined(GGML_FLAGOS_SPACEMIT_RVV)
#    include <riscv_vector.h>
#endif

#include "../../../../ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>

// Reference kernels for the Gated DeltaNet layers' remaining ops (plan.md "Step 2 spec", pieces D to G). Each follows
// ggml-cpu's own code (ggml/src/ggml-cpu @ ba360ef) with the rows split over the tiles, so the arithmetic order is
// ggml-cpu's where ggml-cpu is scalar there.
//
// RVV variants (on the K3, unless GGML_FLAGOS_KERNEL_VARIANTS=0 keeps the references) for SILU, SIGMOID, SOFTPLUS,
// SSM_CONV and CONCAT are ported from SpacemiT's fork (ggml-spacemit's rvv_kernels.cpp, branch mtmd-backend @
// ddfb41ddc); the ones for L2_NORM and MUL_MAT are ours. Each takes the same rows of the same tile as its reference.
// Changes to the ported code are marked "flagos".

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

// The convolution pair (core/flagos-spacemit-ops.h spacemit_conv_pair): a CONCAT [n + 3, C, 1] of the state's 3
// columns and n tokens whose only consumers are one SSM_CONV and views of its last 3 columns. The CONCAT step
// (concat.f32.tail) writes the true last 3 columns, and in each tile's rows of the rest of its buffer the convolution's
// input token-major: in[j][c] for j = 0..n-1 (outer), c = c0..c1-1 (inner), a stream that fills each row's first n
// floats and skips its last 3. The SSM_CONV step (ssm_conv.f32.stashed) reads that stream, and the last 3 tokens'
// inputs past n from the true columns, on the same tile split, so a tile reads only what it wrote

// in[j][c] of the pair: the state's column j for j < 3, else token j - 3 of the second source, read with the sources'
// strides
inline float conv_pair_in(const ggml_tensor * a, const ggml_tensor * b, int64_t j, int64_t c) {
    return j < a->ne[0] ? *(const float *) ((const char *) a->data + j * a->nb[0] + c * a->nb[1]) :
                          *(const float *) ((const char *) b->data + (j - a->ne[0]) * b->nb[0] + c * b->nb[1]);
}

void concat_tail_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    const int64_t       L = dst->ne[0];
    const int64_t       n = L - a->ne[0];
    float *             B = (float *) dst->data;

    int64_t c0, c1;
    tile_range(dst->ne[1], tile, c0, c1);
    for (int64_t c = c0; c < c1; ++c) {
        for (int64_t j = n; j < L; ++j) {
            B[c * L + j] = conv_pair_in(a, b, j, c);
        }
    }
    float * row = B + c0 * L;
    int64_t r   = 0;  // the stream's place in its row
    for (int64_t j = 0; j < n && c1 > c0; ++j) {
        for (int64_t c = c0; c < c1; ++c) {
            row[r] = conv_pair_in(a, b, j, c);
            if (++r == n) {
                r = 0;
                row += L;
            }
        }
    }
}

// ssm_conv_f32_ref's sum, in its order, on the stashed input: one sequence, d_conv 4
void ssm_conv_stashed_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // the paired CONCAT [n + 3, C, 1]
    const ggml_tensor * src1 = dst->src[1];  // conv1d.weight {4, C}
    const int64_t       L    = src0->ne[0];
    const int64_t       n    = dst->ne[1];
    const float *       B    = (const float *) src0->data;

    int64_t c0, c1;
    tile_range(src0->ne[1], tile, c0, c1);
    const int64_t W = c1 - c0;
    if (W <= 0) {
        return;
    }
    for (int64_t t = 0; t < n; ++t) {
        int64_t q[4] = {}, r[4] = {};  // per tap in the stream: the row past c0 and the place in it
        for (int k = 0; k < 4; ++k) {
            if (t + k < n) {
                q[k] = (t + k) * W / n;
                r[k] = (t + k) * W % n;
            }
        }
        float * y = (float *) ((char *) dst->data + t * dst->nb[1]);
        for (int64_t c = c0; c < c1; ++c) {
            const float * cw   = (const float *) ((const char *) src1->data + c * src1->nb[1]);
            float         sumf = 0.0f;
            for (int k = 0; k < 4; ++k) {
                const int64_t j = t + k;
                const float   x = j < n ? B[(c0 + q[k]) * L + r[k]] : B[c * L + j];
                sumf += x * cw[k];
            }
            y[c] = sumf;
            for (int k = 0; k < 4; ++k) {
                if (t + k < n && ++r[k] == n) {
                    r[k] = 0;
                    q[k]++;
                }
            }
        }
    }
}

// FLAGOS_SPACEMIT_MM_F32: blocked (unset or empty, the default): the tiles split the larger of the weight rows and
// the token rows, and a tile takes its token rows in blocks of 16; rows: the split over the token rows and the order
// before it, kept for the comparison run and removed with it (plan.md "The float matmul: all tiles for one token,
// blocks for a prompt"). Parsed once (spacemit_cached_switch; the check program resets between its settings);
// anything else warns once per parse generation and keeps the default
enum class mm_f32_setting { blocked, rows };

int parse_mm_f32_setting() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_MM_F32");
    if (v == nullptr || *v == '\0' || std::strcmp(v, "blocked") == 0) {
        return (int) mm_f32_setting::blocked;
    }
    if (std::strcmp(v, "rows") == 0) {
        return (int) mm_f32_setting::rows;
    }
    static std::atomic<uint32_t> warned_in{ 0 };
    const uint32_t               generation = spacemit_switches_generation();
    if (warned_in.exchange(generation) != generation) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_MM_F32=%s: not blocked or rows, the default (blocked) kept\n", __func__, v);
    }
    return (int) mm_f32_setting::blocked;
}

mm_f32_setting mm_f32_setting_from_env() {
    static std::atomic<int>      value{ 0 };
    static std::atomic<uint32_t> seen{ 0 };
    return static_cast<mm_f32_setting>(spacemit_cached_switch(value, seen, parse_mm_f32_setting));
}

// MUL_MAT of an F32 2-D weight by F32 activations (spacemit_mul_mat_f32_supported): one dot product per output
// element, a weight row against an activation row, summed in float; the weight broadcasts over the activations'
// higher dimensions. ggml-cpu adds in vector lanes (ggml_vec_dot_f32) or through llamafile's sgemm, so these sums are
// not its bits, only within test-backend-ops' bound.
//
// The default (blocked) splits the larger of the N weight rows and the R token rows over the tiles, as ggml-cpu does
// for the op (ggml-cpu.c:1413), and takes a tile's token rows in blocks of 16: one token then uses every tile, and a
// group of 4 weight rows is read once per block instead of once per token row. The rows setting keeps the split over
// the token rows, where one token was one tile's work and a prompt read the whole weight once per token row; it and
// its functions go once the K3 comparison run has decided. No output depends on another and each keeps its operations
// in its order, so both forms give the same bytes

// a tile's share: the weight rows in units of 4, all token rows, where the weight rows outnumber them; else all weight
// rows and today's split of the token rows
void mm_f32_share(int64_t n_weight, int64_t n_token, const spacemit_tile & tile, int64_t & w0, int64_t & w1,
                  int64_t & t0, int64_t & t1) {
    if (n_weight > n_token) {
        int64_t u0, u1;
        tile_range((n_weight + 3) / 4, tile, u0, u1);
        w0 = std::min(n_weight, 4 * u0);
        w1 = std::min(n_weight, 4 * u1);
        t0 = 0;
        t1 = n_token;
        return;
    }
    w0 = 0;
    w1 = n_weight;
    tile_range(n_token, tile, t0, t1);
}

constexpr int64_t mm_f32_block = 16;  // token rows per block, ggml-cpu's blck_1 (ggml-cpu.c:1203)

// one output element, the reference's sum in its order
float mm_f32_dot_ref(const float * w, const float * x, int64_t k_end) {
    float sum = 0.0f;
    for (int64_t k = 0; k < k_end; ++k) {
        sum += w[k] * x[k];
    }
    return sum;
}

void mul_mat_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // the weight [K, N]
    const ggml_tensor * src1 = dst->src[1];  // the activations [K, rows, ...]
    GGML_TENSOR_BINARY_OP_LOCALS

    int64_t w0, w1, t0, t1;
    mm_f32_share(ne0, ne1 * ne2 * ne3, tile, w0, w1, t0, t1);
    for (int64_t tb = t0; tb < t1; tb += mm_f32_block) {
        const int64_t n_rows = std::min(mm_f32_block, t1 - tb);
        const float * xs[mm_f32_block];
        float *       ys[mm_f32_block];
        for (int64_t j = 0; j < n_rows; ++j) {
            const int64_t ir = tb + j;
            const int64_t i3 = ir / (ne2 * ne1);
            const int64_t i2 = (ir / ne1) % ne2;
            const int64_t i1 = ir % ne1;

            xs[j] = (const float *) ((const char *) src1->data + i1 * nb11 + i2 * nb12 + i3 * nb13);
            ys[j] = (float *) ((char *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3);
        }
        for (int64_t r = w0; r < w1; r += 4) {
            const int64_t n_w = std::min((int64_t) 4, w1 - r);
            for (int64_t j = 0; j < n_rows; ++j) {
                for (int64_t i = 0; i < n_w; ++i) {
                    const float * w = (const float *) ((const char *) src0->data + (r + i) * nb01);
                    ys[j][r + i]    = mm_f32_dot_ref(w, xs[j], ne00);
                }
            }
        }
    }
}

// the rows setting's reference: the output's rows split over the tiles, each token row through every weight row
void mul_mat_f32_rows_ref(const spacemit_tile & tile, ggml_tensor * dst) {
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

// ours: concat_tail_f32_ref with the second source's channels contiguous (false otherwise, and the reference runs).
// The true columns by strided stores down the rows; the stream in runs of channels, each cut where a row's piece ends,
// read with one load (the state's columns strided) and stored unit-stride
bool concat_tail_f32_rvv(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    const int64_t       L = dst->ne[0];
    const int64_t       S = a->ne[0];
    const int64_t       n = L - S;
    if (b->nb[1] != sizeof(float) || n < S) {
        return false;
    }
    float * B = (float *) dst->data;

    int64_t c0, c1;
    tile_range(dst->ne[1], tile, c0, c1);
    for (int64_t j = n; j < L; ++j) {
        const float * x = (const float *) ((const char *) b->data + (j - S) * b->nb[0]);
        for (int64_t c = c0; c < c1;) {
            const size_t vl = __riscv_vsetvl_e32m4(c1 - c);
            __riscv_vsse32_v_f32m4(B + c * L + j, (ptrdiff_t) (L * sizeof(float)), __riscv_vle32_v_f32m4(x + c, vl),
                                   vl);
            c += (int64_t) vl;
        }
    }
    float * row = B + c0 * L;
    int64_t r   = 0;  // the stream's place in its row
    for (int64_t j = 0; j < n && c1 > c0; ++j) {
        const char * col = j < S ? (const char *) a->data + j * a->nb[0] : (const char *) b->data + (j - S) * b->nb[0];
        const size_t cs  = j < S ? a->nb[1] : sizeof(float);
        for (int64_t c = c0; c < c1;) {
            const size_t       vl = __riscv_vsetvl_e32m4(std::min(c1 - c, n - r));
            const float *      p  = (const float *) (col + c * cs);
            const vfloat32m4_t v  = j < S ? __riscv_vlse32_v_f32m4(p, (ptrdiff_t) cs, vl) : __riscv_vle32_v_f32m4(p, vl);
            __riscv_vse32_v_f32m4(row + r, v, vl);
            c += (int64_t) vl;
            r += (int64_t) vl;
            if (r == n) {
                r = 0;
                row += L;
            }
        }
    }
    return true;
}

// ours: ssm_conv_f32_rvv's lanes (the same fused multiply-adds in the same order, so its bits) on the stashed input.
// Per token the tile's channels in runs cut where a tap's stream piece ends: unit-stride loads of the stream and of the
// weights, de-interleaved once per call into the tile's 4 * W floats of the workspace (offset 4 * c0); the taps past
// n (the last 3 tokens) by strided loads from the true columns; unit-stride stores. false when the workspace is short
bool ssm_conv_stashed_f32_rvv(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // the paired CONCAT [n + 3, C, 1]
    const ggml_tensor * src1 = dst->src[1];  // conv1d.weight {4, C}
    const int64_t       C    = src0->ne[1];
    const int64_t       L    = src0->ne[0];
    const int64_t       n    = dst->ne[1];
    const float *       B    = (const float *) src0->data;

    int64_t c0, c1;
    tile_range(C, tile, c0, c1);
    const int64_t W = c1 - c0;
    if (W <= 0) {
        return true;
    }
    float * wd = (float *) tile.workspace + 4 * c0;
    for (int k = 0; k < 4; ++k) {
        for (int64_t i = 0; i < W;) {
            const size_t  vl = __riscv_vsetvl_e32m4(W - i);
            const float * wp = (const float *) ((const char *) src1->data + (c0 + i) * src1->nb[1]) + k;
            __riscv_vse32_v_f32m4(wd + k * W + i, __riscv_vlse32_v_f32m4(wp, (ptrdiff_t) src1->nb[1], vl), vl);
            i += (int64_t) vl;
        }
    }
    const ptrdiff_t ls = (ptrdiff_t) (L * sizeof(float));
    for (int64_t t = 0; t < n; ++t) {
        float * y = (float *) ((char *) dst->data + t * dst->nb[1]) + c0;
        for (int64_t i = 0; i < W;) {
            int64_t len = W - i;
            for (int k = 0; k < 4; ++k) {
                if (t + k < n) {
                    len = std::min(len, n - ((t + k) * W + i) % n);
                }
            }
            const size_t vl  = __riscv_vsetvl_e32m4(len);
            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            for (int k = 0; k < 4; ++k) {
                const int64_t      j = t + k;
                const int64_t      p = j * W + i;
                const vfloat32m4_t x = j < n ? __riscv_vle32_v_f32m4(B + (c0 + p / n) * L + p % n, vl) :
                                               __riscv_vlse32_v_f32m4(B + (c0 + i) * L + j, ls, vl);
                acc = __riscv_vfmacc_vv_f32m4(acc, x, __riscv_vle32_v_f32m4(wd + k * W + i, vl), vl);
            }
            __riscv_vse32_v_f32m4(y + i, acc, vl);
            i += (int64_t) vl;
        }
    }
    return true;
}

// the sum of an m2 group's vlmax lanes
float rvv_sum_f32m2(vfloat32m2_t acc, size_t vlmax) {
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vlmax));
}

// ours, not a port: each dot product in m2 lanes over strips of K (a short last strip keeps the lanes past its vl,
// tail undisturbed) and summed across the lanes at the end, so the sums are not the reference's bits. Four weight rows
// share each strip of x: 4 accumulators, x and 4 weight strips take 18 of the 32 registers. A row's sum does not
// depend on its group, so the rows past the last 4 take the same order one at a time
void rvv_dot4_f32m2(const float * w0, const float * w1, const float * w2, const float * w3, const float * x,
                    int64_t k_end, size_t vlmax, float * y) {
    vfloat32m2_t a0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t a1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t a2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t a3 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    for (int64_t k = 0; k < k_end;) {
        const size_t       vl = __riscv_vsetvl_e32m2(k_end - k);
        const vfloat32m2_t xv = __riscv_vle32_v_f32m2(x + k, vl);
        a0 = __riscv_vfmacc_vv_f32m2_tu(a0, __riscv_vle32_v_f32m2(w0 + k, vl), xv, vl);
        a1 = __riscv_vfmacc_vv_f32m2_tu(a1, __riscv_vle32_v_f32m2(w1 + k, vl), xv, vl);
        a2 = __riscv_vfmacc_vv_f32m2_tu(a2, __riscv_vle32_v_f32m2(w2 + k, vl), xv, vl);
        a3 = __riscv_vfmacc_vv_f32m2_tu(a3, __riscv_vle32_v_f32m2(w3 + k, vl), xv, vl);
        k += (int64_t) vl;
    }
    y[0] = rvv_sum_f32m2(a0, vlmax);
    y[1] = rvv_sum_f32m2(a1, vlmax);
    y[2] = rvv_sum_f32m2(a2, vlmax);
    y[3] = rvv_sum_f32m2(a3, vlmax);
}

void rvv_dot1_f32m2(const float * w, const float * x, int64_t k_end, size_t vlmax, float * y) {
    vfloat32m2_t a = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    for (int64_t k = 0; k < k_end;) {
        const size_t       vl = __riscv_vsetvl_e32m2(k_end - k);
        const vfloat32m2_t xv = __riscv_vle32_v_f32m2(x + k, vl);
        a                     = __riscv_vfmacc_vv_f32m2_tu(a, __riscv_vle32_v_f32m2(w + k, vl), xv, vl);
        k += (int64_t) vl;
    }
    y[0] = rvv_sum_f32m2(a, vlmax);
}

// the reference's share and blocks with the dots above
void mul_mat_f32_rvv(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // the weight [K, N]
    const ggml_tensor * src1 = dst->src[1];  // the activations [K, rows, ...]
    GGML_TENSOR_BINARY_OP_LOCALS

    const size_t vlmax = __riscv_vsetvlmax_e32m2();
    int64_t      w0, w1, t0, t1;
    mm_f32_share(ne0, ne1 * ne2 * ne3, tile, w0, w1, t0, t1);
    for (int64_t tb = t0; tb < t1; tb += mm_f32_block) {
        const int64_t n_rows = std::min(mm_f32_block, t1 - tb);
        const float * xs[mm_f32_block];
        float *       ys[mm_f32_block];
        for (int64_t j = 0; j < n_rows; ++j) {
            const int64_t ir = tb + j;
            const int64_t i3 = ir / (ne2 * ne1);
            const int64_t i2 = (ir / ne1) % ne2;
            const int64_t i1 = ir % ne1;

            xs[j] = (const float *) ((const char *) src1->data + i1 * nb11 + i2 * nb12 + i3 * nb13);
            ys[j] = (float *) ((char *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3);
        }
        for (int64_t r = w0; r < w1; r += 4) {
            if (r + 4 <= w1) {
                const float * p0 = (const float *) ((const char *) src0->data + (r + 0) * nb01);
                const float * p1 = (const float *) ((const char *) src0->data + (r + 1) * nb01);
                const float * p2 = (const float *) ((const char *) src0->data + (r + 2) * nb01);
                const float * p3 = (const float *) ((const char *) src0->data + (r + 3) * nb01);
                for (int64_t j = 0; j < n_rows; ++j) {
                    rvv_dot4_f32m2(p0, p1, p2, p3, xs[j], ne00, vlmax, ys[j] + r);
                }
                continue;
            }
            for (int64_t rr = r; rr < w1; ++rr) {
                const float * p = (const float *) ((const char *) src0->data + rr * nb01);
                for (int64_t j = 0; j < n_rows; ++j) {
                    rvv_dot1_f32m2(p, xs[j], ne00, vlmax, ys[j] + rr);
                }
            }
        }
    }
}

// the rows setting's kernel: the output's rows split over the tiles, each token row through every weight row
void mul_mat_f32_rows_rvv(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // the weight [K, N]
    const ggml_tensor * src1 = dst->src[1];  // the activations [K, rows, ...]
    GGML_TENSOR_BINARY_OP_LOCALS

    const size_t vlmax = __riscv_vsetvlmax_e32m2();
    int64_t      ir0, ir1;
    tile_range(ne1 * ne2 * ne3, tile, ir0, ir1);
    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i3 = ir / (ne2 * ne1);
        const int64_t i2 = (ir / ne1) % ne2;
        const int64_t i1 = ir % ne1;

        const float * x = (const float *) ((const char *) src1->data + i1 * nb11 + i2 * nb12 + i3 * nb13);
        float *       y = (float *) ((char *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3);
        int64_t       r = 0;
        for (; r + 4 <= ne0; r += 4) {
            const float * w0 = (const float *) ((const char *) src0->data + (r + 0) * nb01);
            const float * w1 = (const float *) ((const char *) src0->data + (r + 1) * nb01);
            const float * w2 = (const float *) ((const char *) src0->data + (r + 2) * nb01);
            const float * w3 = (const float *) ((const char *) src0->data + (r + 3) * nb01);
            rvv_dot4_f32m2(w0, w1, w2, w3, x, ne00, vlmax, y + r);
        }
        for (; r < ne0; ++r) {
            const float * w = (const float *) ((const char *) src0->data + r * nb01);
            rvv_dot1_f32m2(w, x, ne00, vlmax, y + r);
        }
    }
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

// one token takes the reference: measured on the K3 at one token only (the RVV SSM_CONV 2.8 ms a token against the
// reference's 2.0, the RVV CONCAT 1.3 against 1.1)
bool spacemit_kernel_ssm_conv_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference() && dst->ne[1] != 1 && ssm_conv_f32_rvv(*tile_p, dst)) {
        return true;
    }
#endif
    ssm_conv_f32_ref(*tile_p, dst);
    return true;
}

bool spacemit_kernel_concat_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference() && dst->src[1]->ne[0] != 1 && concat_dim0_transposed_f32_rvv(*tile_p, dst)) {
        return true;
    }
#endif
    concat_dim0_f32_ref(*tile_p, dst);
    return true;
}

bool spacemit_kernel_concat_f32_tail(const spacemit_tile * tile_p, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference() && concat_tail_f32_rvv(*tile_p, dst)) {
        return true;
    }
#endif
    concat_tail_f32_ref(*tile_p, dst);
    return true;
}

size_t spacemit_ssm_conv_stashed_workspace(const ggml_tensor * node) {
    return (size_t) (4 * node->src[0]->ne[1]) * sizeof(float);
}

bool spacemit_kernel_ssm_conv_f32_stashed(const spacemit_tile * tile_p, ggml_tensor * dst) {
    const spacemit_tile & tile   = *tile_p;
    const size_t          needed = spacemit_ssm_conv_stashed_workspace(dst);
    if (tile.workspace == nullptr || tile.workspace_size < needed) {
        GGML_LOG_ERROR("FlagOS SpacemiT: ssm_conv.f32.stashed needs %zu workspace bytes, has %zu\n", needed,
                       tile.workspace_size);
        return false;
    }
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference()) {
        return ssm_conv_stashed_f32_rvv(tile, dst);
    }
#endif
    ssm_conv_stashed_f32_ref(tile, dst);
    return true;
}

void spacemit_mm_f32_read_switch() {
    (void) mm_f32_setting_from_env();
}

bool spacemit_kernel_mul_mat_f32(const spacemit_tile * tile_p, ggml_tensor * dst) {
    const bool rows = mm_f32_setting_from_env() == mm_f32_setting::rows;
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference()) {
        if (rows) {
            mul_mat_f32_rows_rvv(*tile_p, dst);
        } else {
            mul_mat_f32_rvv(*tile_p, dst);
        }
        return true;
    }
#endif
    if (rows) {
        mul_mat_f32_rows_ref(*tile_p, dst);
    } else {
        mul_mat_f32_ref(*tile_p, dst);
    }
    return true;
}
