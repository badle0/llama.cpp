// FLASH_ATTN_EXT on the AI cores (plan.md, M2e design): the claim, the split of the rows over the tiles, and the
// portable reference.

#include "flagos-spacemit-attn.h"

#include "flagos-spacemit-kernels.h"
#include "flagos-spacemit-rvv-attn.h"

#include "../../../ggml-impl.h"

#if defined(GGML_FLAGOS_SPACEMIT_RVV)
#    include <riscv_vector.h>
#endif

#include <algorithm>
#include <cmath>

namespace {

// largest DK and DV: one register group of the RVV kernels at VLEN 1024, and the reference's local arrays
constexpr int64_t max_head_size = 128;

#if defined(GGML_FLAGOS_SPACEMIT_RVV)
// from this many tokens per head the tiled kernel runs (ggml-cpu's ggml_fa_tile_config::Q, as in ime.cpp)
constexpr int64_t tiled_min_rows = 64;
#endif

bool divides(int64_t d, int64_t n) {
    return d > 0 && n % d == 0;
}

// [begin, end) of n items for one tile, the way ggml-cpu splits rows over its threads (flagos-spacemit-kernels.cpp)
void tile_range(int64_t n, const spacemit_tile & tile, int64_t & begin, int64_t & end) {
    const int64_t per = (n + tile.nth - 1) / tile.nth;
    begin             = std::min(n, per * tile.ith);
    end               = std::min(n, begin + per);
}

// ops.cpp ggml_compute_forward_flash_attn_ext_f16_one_chunk (ggml-cpu @ ba360ef) for F16 K and V, rows [ir0, ir1):
// q rounded to F16 and dotted with K in F32, online softmax, sinks at the end. flagos: V accumulated in F32, not F16
void flash_attn_ref(const ggml_tensor * dst, int64_t ir0, int64_t ir1) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const int64_t DK   = k->ne[0];
    const int64_t DV   = v->ne[0];
    const int64_t n_kv = k->ne[1];
    const int64_t neq1 = q->ne[1];
    const int64_t neq2 = q->ne[2];
    const int64_t ne1  = dst->ne[1];
    const int64_t ne2  = dst->ne[2];

    // broadcast factors
    const int64_t rk2 = neq2 / k->ne[2];
    const int64_t rk3 = q->ne[3] / k->ne[3];
    const int64_t rv2 = neq2 / v->ne[2];
    const int64_t rv3 = q->ne[3] / v->ne[3];

    float       scale         = ggml_get_op_params_f32(dst, 0);
    const float max_bias      = ggml_get_op_params_f32(dst, 1);
    const float logit_softcap = ggml_get_op_params_f32(dst, 2);
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(neq2));
    const float    m0          = powf(2.0f, -(max_bias) / n_head_log2);
    const float    m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t iq3 = ir / (neq2 * neq1);
        const int64_t iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        const int64_t iq1 = ir - iq3 * neq2 * neq1 - iq2 * neq1;

        const uint32_t h = iq2;  // head index
        const float    slope =
            (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

        const ggml_fp16_t * mp =
            mask ? (const ggml_fp16_t *) ((const char *) mask->data + iq1 * mask->nb[1] +
                                          (iq2 % mask->ne[2]) * mask->nb[2] + (iq3 % mask->ne[3]) * mask->nb[3]) :
                   nullptr;
        const char * k_head = (const char *) k->data + (iq2 / rk2) * k->nb[2] + (iq3 / rk3) * k->nb[3];
        const char * v_head = (const char *) v->data + (iq2 / rv2) * v->nb[2] + (iq3 / rv3) * v->nb[3];

        const float * pq = (const float *) ((const char *) q->data + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);
        float         q16[max_head_size];
        for (int64_t d = 0; d < DK; ++d) {
            q16[d] = GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(pq[d]));
        }

        float VKQ[max_head_size] = {};

        float S = 0.0f;       // sum
        float M = -INFINITY;  // maximum KQ value

        for (int64_t ic = 0; ic < n_kv; ++ic) {
            const float mv = mp ? slope * GGML_FP16_TO_FP32(mp[ic]) : 0.0f;
            if (mv == -INFINITY) {
                continue;
            }

            const ggml_fp16_t * kr = (const ggml_fp16_t *) (k_head + ic * k->nb[1]);
            float               s  = 0.0f;
            for (int64_t d = 0; d < DK; ++d) {
                s += q16[d] * GGML_FP16_TO_FP32(kr[d]);
            }
            s = s * scale;
            if (logit_softcap != 0.0f) {
                s = logit_softcap * tanhf(s);
            }
            s += mv;

            float ms = 1.0f;  // upon a new maximum, scales VKQ and S
            float vs = 1.0f;  // expf(s - M)
            if (s > M) {
                ms = expf(M - s);
                M  = s;
                for (int64_t d = 0; d < DV; ++d) {
                    VKQ[d] *= ms;
                }
            } else {
                vs = expf(s - M);
            }

            const ggml_fp16_t * vr = (const ggml_fp16_t *) (v_head + ic * v->nb[1]);
            for (int64_t d = 0; d < DV; ++d) {
                VKQ[d] += GGML_FP16_TO_FP32(vr[d]) * vs;
            }
            S = S * ms + vs;
        }

        if (sinks) {
            const float s  = ((const float *) sinks->data)[h];
            float       ms = 1.0f;
            float       vs = 1.0f;
            if (s > M) {
                ms = expf(M - s);
                M  = s;
                for (int64_t d = 0; d < DV; ++d) {
                    VKQ[d] *= ms;
                }
            } else {
                vs = expf(s - M);
            }
            S = S * ms + vs;
        }

        // V /= S, written permuted (0, 2, 1, 3)
        const float S_inv = S == 0.0f ? 0.0f : 1.0f / S;
        float *     out   = (float *) ((char *) dst->data + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * dst->nb[1]);
        for (int64_t d = 0; d < DV; ++d) {
            out[d] = VKQ[d] * S_inv;
        }
    }
}

}  // namespace

// FLASH_ATTN_EXT: F32 q and F16 K and V, each with contiguous rows; DK (q's and K's row length) and DV in 1..128;
// query heads broadcast over the K and V heads (GQA), streams do not; mask absent or F16 covering n_kv and the rows,
// broadcast over heads and streams as ggml allows; sinks absent or F32, one per query head; dst as ggml_flash_attn_ext
// makes it (the kernels write whole rows of it, permuted). ALiBi and softcap included: the kernels and the reference
// handle them, and the K3 run of test-backend-ops decides whether they stay (plan.md, M2e design)
bool spacemit_flash_attn_supported(const ggml_tensor * op) {
    const ggml_tensor * q     = op->src[0];
    const ggml_tensor * k     = op->src[1];
    const ggml_tensor * v     = op->src[2];
    const ggml_tensor * mask  = op->src[3];
    const ggml_tensor * sinks = op->src[4];

    const auto f16_rows = [](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F16 && t->nb[0] == sizeof(ggml_fp16_t);
    };
    const int64_t DK   = k->ne[0];
    const int64_t DV   = v->ne[0];
    const int32_t prec = ggml_get_op_params_i32(op, 3);

    const bool qkv = q->type == GGML_TYPE_F32 && q->nb[0] == sizeof(float) && f16_rows(k) && f16_rows(v) &&
                     q->ne[0] == DK && DK >= 1 && DK <= max_head_size && DV >= 1 && DV <= max_head_size &&
                     k->ne[1] == v->ne[1] && divides(k->ne[2], q->ne[2]) && divides(v->ne[2], q->ne[2]) &&
                     k->ne[3] == q->ne[3] && v->ne[3] == q->ne[3];
    const bool out = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->ne[0] == DV && op->ne[1] == q->ne[2] &&
                     op->ne[2] == q->ne[1] && op->ne[3] == q->ne[3];
    const bool mask_ok = mask == nullptr ||
                         (mask->type == GGML_TYPE_F16 && ggml_is_contiguous(mask) && mask->ne[0] >= k->ne[1] &&
                          mask->ne[1] >= q->ne[1] && divides(mask->ne[2], q->ne[2]) && divides(mask->ne[3], q->ne[3]));
    const bool sinks_ok =
        sinks == nullptr || (sinks->type == GGML_TYPE_F32 && ggml_is_contiguous(sinks) && sinks->ne[0] >= q->ne[2]);
    return qkv && out && mask_ok && sinks_ok && (prec == GGML_PREC_F32 || prec == GGML_PREC_DEFAULT);
}

// one step: each tile takes a contiguous range of rows (query head x token x stream, ggml-cpu's order) and writes
// their dst rows only, so tiles neither share scratch nor wait for each other
bool spacemit_kernel_flash_attn(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    int64_t             ir0, ir1;
    tile_range(q->ne[1] * q->ne[2] * q->ne[3], tile, ir0, ir1);
    if (ir0 >= ir1) {
        return true;
    }
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    // the copied kernels need VLEN 1024, read on the core that runs them (ime.cpp checks the same); the tiled kernel
    // keeps its scratch in TCM, the decode kernels use TCM when there is some and otherwise go row by row
    const int64_t DK = dst->src[1]->ne[0];
    const int64_t DV = dst->src[2]->ne[0];
    if (!spacemit_use_reference() && __riscv_vlenb() == 128 && spacemit_rvv_attn::supported_shape(DK, DV)) {
        if (q->ne[1] >= tiled_min_rows && tile.tcm != nullptr &&
            tile.tcm_size >= spacemit_rvv_attn::tiled_scratch_bytes(DK, DV)) {
            spacemit_rvv_attn::forward_tiled(dst, ir0, ir1, tile.tcm, tile.tcm_size);
        } else {
            spacemit_rvv_attn::forward_one_chunk(dst, ir0, ir1, tile.tcm, tile.tcm_size);
        }
        return true;
    }
#endif
    flash_attn_ref(dst, ir0, ir1);
    return true;
}
