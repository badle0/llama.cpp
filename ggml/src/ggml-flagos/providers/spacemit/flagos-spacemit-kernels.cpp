#include "flagos-spacemit-kernels.h"

#if defined(GGML_FLAGOS_SPACEMIT_RVV)
#    include "flagos-spacemit-rvv-kernels.h"
#endif

#include "../../../ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

bool spacemit_use_reference() {
    static const bool reference = [] {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
        const char * value = std::getenv("FLAGOS_SPACEMIT_TEST_REFERENCE");
        return value != nullptr && std::strcmp(value, "0") != 0;
#else
        return true;
#endif
    }();
    return reference;
}

namespace {

// [begin, end) of n items for one tile, the way ggml-cpu splits rows over its threads
void tile_range(int64_t n, const spacemit_tile & tile, int64_t & begin, int64_t & end) {
    const int64_t per = (n + tile.nth - 1) / tile.nth;
    begin             = std::min(n, per * tile.ith);
    end               = std::min(n, begin + per);
}

//
// portable references: the arithmetic of ggml-cpu (ggml/src/ggml-cpu @ ba360ef), split over tiles
//

float op_add(float a, float b) {
    return a + b;
}

float op_mul(float a, float b) {
    return a * b;
}

// binary-ops.cpp apply_binary_op, F32: src1 repeats over src0 in every dimension
template <float (*op)(float, float)> void binary_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
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
                    z[r * ne10 + i] = op(x[r * ne10 + i], ((const float *) y)[i]);
                }
            }
        } else {
            for (int64_t i = 0; i < ne0; ++i) {
                z[i] = op(x[i], *(const float *) (y + (i % ne10) * nb10));
            }
        }
    }
}

// ops.cpp ggml_compute_forward_rms_norm_f32: sum of squares in double
void rms_norm_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
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
        const float mean  = sum / ne00;
        const float scale = 1.0f / sqrtf(mean + eps);
        for (int64_t i00 = 0; i00 < ne00; i00++) {
            y[i00] = x[i00] * scale;
        }
    }
}

// ops.cpp rope_yarn_ramp, rope_yarn, ggml_rope_cache_init and rotate_pairs (F32), copied
float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
    return 1 - MIN(1, MAX(0, y));
}

void rope_yarn(float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
               float * cos_theta, float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta        = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta          = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

void ggml_rope_cache_init(float theta_base, float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0,
                          float ext_factor, float mscale, float * cache, float sin_sign, float theta_scale) {
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]);
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

void rotate_pairs(const int64_t n, const int64_t n_offset, const float * cache, const float * src_data, float * dst_data,
                  const int scale = 2) {
    for (int64_t i0 = 0; i0 < n; i0 += 2) {
        const int64_t ic = i0 / scale;  // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

        const float cos_theta = cache[i0 + 0];
        const float sin_theta = cache[i0 + 1];

        const float * const src = src_data + ic;
        float *             dst = dst_data + ic;

        const float x0 = src[0];
        const float x1 = src[n_offset];

        dst[0]        = x0 * cos_theta - x1 * sin_theta;
        dst[n_offset] = x0 * sin_theta + x1 * cos_theta;
    }
}

// floats of cos/sin values each tile keeps in the workspace (as ggml-cpu, padded to a cache line)
int64_t rope_cache_floats(const ggml_tensor * node) {
    return node->ne[0] + 64 / (int64_t) sizeof(float);
}

// ops.cpp ggml_compute_forward_rope_flt<float>, forward, modes NORMAL and NEOX
void rope_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;

    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    std::memcpy(&freq_base, (int32_t *) dst->op_params + 5, sizeof(float));
    std::memcpy(&freq_scale, (int32_t *) dst->op_params + 6, sizeof(float));
    std::memcpy(&ext_factor, (int32_t *) dst->op_params + 7, sizeof(float));
    std::memcpy(&attn_factor, (int32_t *) dst->op_params + 8, sizeof(float));
    std::memcpy(&beta_fast, (int32_t *) dst->op_params + 9, sizeof(float));
    std::memcpy(&beta_slow, (int32_t *) dst->op_params + 10, sizeof(float));

    GGML_TENSOR_UNARY_OP_LOCALS

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const float *   freq_factors = src2 != nullptr ? (const float *) src2->data : nullptr;
    const int32_t * pos          = (const int32_t *) src1->data;
    float *         cache        = (float *) tile.workspace + rope_cache_floats(dst) * tile.ith;

    int64_t ir0, ir1;
    tile_range(ggml_nrows(dst), tile, ir0, ir1);
    int64_t last_i2 = -1;
    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i3 = ir / (ne2 * ne1);
        const int64_t i2 = (ir / ne1) % ne2;
        const int64_t i1 = ir % ne1;

        if (i2 != last_i2) {
            // flagos: filled to n_dims, not ne0 as in ggml-cpu, which reads the n_dims/2 frequency factors past their
            // end when n_dims < ne0 (the entries past n_dims are never used)
            ggml_rope_cache_init(pos[i2], freq_scale, freq_factors, corr_dims, n_dims, ext_factor, attn_factor, cache,
                                 1.0f, theta_scale);
            last_i2 = i2;
        }

        const float * src      = (const float *) ((const char *) src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
        float *       dst_data = (float *) ((char *) dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
        if (mode == GGML_ROPE_TYPE_NORMAL) {
            rotate_pairs(n_dims, 1, cache, src, dst_data, 1);
        } else {
            rotate_pairs(n_dims, n_dims / 2, cache, src, dst_data);
        }
        // the remaining channels are copied from the source
        for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
            dst_data[i0]     = src[i0];
            dst_data[i0 + 1] = src[i0 + 1];
        }
    }
}

// ops.cpp ggml_compute_forward_set_rows_impl; tiles split the columns, not the rows, so repeated indices (last row
// wins) cannot make two tiles write the same element (ggml-spacemit's split)
bool set_rows_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    GGML_TENSOR_BINARY_OP_LOCALS

    int64_t c0, c1;
    tile_range(ne00, tile, c0, c1);
    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            const int64_t i12 = i03 % ne12;
            const int64_t i11 = i02 % ne11;
            for (int64_t i = 0; i < ne01; ++i) {
                const char *  p  = (const char *) src1->data + i * nb10 + i11 * nb11 + i12 * nb12;
                const int64_t i1 = src1->type == GGML_TYPE_I64 ? *(const int64_t *) p : *(const int32_t *) p;
                if (i1 < 0 || i1 >= ne1) {
                    return false;
                }
                const char * s = (const char *) src0->data + i * nb01 + i02 * nb02 + i03 * nb03;
                char *       d = (char *) dst->data + i1 * nb1 + i02 * nb2 + i03 * nb3;
                for (int64_t c = c0; c < c1; ++c) {
                    const float v = src0->type == GGML_TYPE_F32 ? ((const float *) s)[c]
                                                                : GGML_FP16_TO_FP32(((const ggml_fp16_t *) s)[c]);
                    if (dst->type == GGML_TYPE_F32) {
                        ((float *) d)[c] = v;
                    } else {
                        ((ggml_fp16_t *) d)[c] = GGML_FP32_TO_FP16(v);
                    }
                }
            }
        }
    }
    return true;
}

// ops.cpp ggml_compute_forward_get_rows_f32; a single row is split by columns instead
bool get_rows_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nr = ggml_nelements(src1);
    int64_t       ir0 = 0, ir1 = nr, c0 = 0, c1 = ne00;
    if (nr == 1) {
        tile_range(ne00, tile, c0, c1);
    } else {
        tile_range(nr, tile, ir0, ir1);
    }
    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i / (ne11 * ne10);
        const int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
        const int64_t i10 = i - i12 * ne11 * ne10 - i11 * ne10;
        const int64_t i01 = *(const int32_t *) ((const char *) src1->data + i10 * nb10 + i11 * nb11 + i12 * nb12);
        if (i01 < 0 || i01 >= ne01) {
            return false;
        }
        const float * s = (const float *) ((const char *) src0->data + i01 * nb01 + i11 * nb02 + i12 * nb03);
        float *       d = (float *) ((char *) dst->data + i10 * nb1 + i11 * nb2 + i12 * nb3);
        for (int64_t c = c0; c < c1; ++c) {
            d[c] = s[c];
        }
    }
    return true;
}

// ops.cpp ggml_compute_forward_swiglu_f32 with vec.h ggml_silu_f32: silu(x) * g, split or single-tensor form; tiles
// split the elements (rows x columns) evenly
void swiglu_f32_ref(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int64_t nc      = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int64_t nr      = ggml_nrows(src0);
    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    int64_t e0, e1;
    tile_range(nr * nc, tile, e0, e1);
    for (int64_t e = e0; e < e1;) {
        const int64_t r   = e / nc;
        const int64_t c   = e % nc;
        const int64_t run = std::min(nc - c, e1 - e);

        const float * x = (const float *) ((const char *) src0->data + r * src0->nb[1]);
        const float * g = src1 ? (const float *) ((const char *) src1->data + r * src1->nb[1]) : x;
        if (src1 == nullptr) {
            x += swapped ? nc : 0;
            g += swapped ? 0 : nc;
        }
        float * y = (float *) ((char *) dst->data + r * dst->nb[1]);
        for (int64_t i = c; i < c + run; ++i) {
            y[i] = x[i] / (1.0f + expf(-x[i])) * g[i];
        }
        e += run;
    }
}

#if defined(GGML_FLAGOS_SPACEMIT_RVV)
spacemit_rvv::context rvv_context(const spacemit_tile & tile) {
    return { (int) tile.ith, (int) tile.nth, tile.workspace, tile.workspace_size, { tile.tcm, tile.tcm_size } };
}
#endif

}  // namespace

size_t spacemit_rope_workspace(const ggml_tensor * node) {
    return (size_t) rope_cache_floats(node) * SPACEMIT_MAX_TILES * sizeof(float);
}

//
// kernels: the ported RVV version where it applies, else the reference
//

bool spacemit_kernel_add_f32(const spacemit_tile & tile, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference() && spacemit_rvv::binary_applies(dst)) {
        auto ctx = rvv_context(tile);
        spacemit_rvv::forward_binary<GGML_OP_ADD, float>(ctx, dst);
        return true;
    }
#endif
    binary_f32_ref<op_add>(tile, dst);
    return true;
}

bool spacemit_kernel_mul_f32(const spacemit_tile & tile, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference() && spacemit_rvv::binary_applies(dst)) {
        auto ctx = rvv_context(tile);
        spacemit_rvv::forward_binary<GGML_OP_MUL, float>(ctx, dst);
        return true;
    }
#endif
    binary_f32_ref<op_mul>(tile, dst);
    return true;
}

bool spacemit_kernel_rms_norm_f32(const spacemit_tile & tile, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference()) {
        auto ctx = rvv_context(tile);
        spacemit_rvv::forward_rms_norm_f32(ctx, dst);
        return true;
    }
#endif
    rms_norm_f32_ref(tile, dst);
    return true;
}

bool spacemit_kernel_rope_f32(const spacemit_tile & tile, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference()) {
        auto ctx = rvv_context(tile);
        spacemit_rvv::forward_rope_f32(ctx, dst);
        return true;
    }
#endif
    rope_f32_ref(tile, dst);
    return true;
}

bool spacemit_kernel_set_rows(const spacemit_tile & tile, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference()) {
        auto ctx = rvv_context(tile);
        return spacemit_rvv::forward_set_rows(ctx, dst);
    }
#endif
    return set_rows_ref(tile, dst);
}

bool spacemit_kernel_get_rows_f32(const spacemit_tile & tile, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference()) {
        auto ctx = rvv_context(tile);
        return spacemit_rvv::forward_get_rows_f32(ctx, dst);
    }
#endif
    return get_rows_f32_ref(tile, dst);
}

bool spacemit_kernel_swiglu_f32(const spacemit_tile & tile, ggml_tensor * dst) {
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference()) {
        auto ctx = rvv_context(tile);
        spacemit_rvv::forward_glu_swiglu_f32(ctx, dst);
        return true;
    }
#endif
    swiglu_f32_ref(tile, dst);
    return true;
}
