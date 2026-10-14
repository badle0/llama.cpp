// FLASH_ATTN_EXT on the AI cores (plan.md, M2e design): the claim, the split of the rows over the tiles, and the
// portable reference; for one token per call from n_kv 512, the split of the KV positions over the tiles and the merge
// of their partial results (plan.md, M2e.3 round 2).

#include "flagos-spacemit-attn.h"

#include "flagos-spacemit-kernels.h"
#include "flagos-spacemit-rvv-attn.h"

#include "../../../ggml-impl.h"

#if defined(GGML_FLAGOS_SPACEMIT_RVV)
#    include <riscv_vector.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// the split (M2e.3 round 2) from this many positions, as ggml-cpu's for one token (ops.cpp:9115): 8 blocks of 64
constexpr int64_t split_min_kv = 512;

// how the calls the split takes run (plan.md M2e.3, round 2, adopted 2026-10-14): 2 the split with the turns of the two
// tiles of a pair (the default; tg32 at depth 4096 +20%), 1 the split without the turns (the same outputs), 0 the
// one-chunk kernels as before. test-only: FLAGOS_SPACEMIT_ATTN=split or one-chunk, for comparisons (paired names the
// default), read whenever spacemit_find_op looks at a node, so a benchmark can switch; anything else is reported once
// and ignored
int split_setting_from_env() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_ATTN");
    if (v == nullptr || *v == '\0' || std::strcmp(v, "paired") == 0) {
        return 2;
    }
    if (std::strcmp(v, "split") == 0) {
        return 1;
    }
    if (std::strcmp(v, "one-chunk") == 0) {
        return 0;
    }
    static std::atomic<bool> warned{ false };
    if (!warned.exchange(true)) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_ATTN=%s: unknown setting ignored\n", __func__, v);
    }
    return 2;
}

// the shapes the split takes: one token and one stream, from split_min_kv positions, F16 K and V in llama.cpp's cache
// layout (a position's cell holds every KV head, one after another), no ALiBi and no softcap, the mask (if any) one row
// for every head
bool split_shape(const ggml_tensor * op) {
    const ggml_tensor * q     = op->src[0];
    const ggml_tensor * k     = op->src[1];
    const ggml_tensor * v     = op->src[2];
    const ggml_tensor * mask  = op->src[3];
    const auto          cells = [](const ggml_tensor * t) {
        return t->nb[2] == (size_t) t->ne[0] * sizeof(ggml_fp16_t) && t->nb[1] == (size_t) t->ne[2] * t->nb[2];
    };
    return q->ne[1] == 1 && q->ne[3] == 1 && k->ne[1] >= split_min_kv && k->ne[2] == v->ne[2] && cells(k) && cells(v) &&
           ggml_get_op_params_f32(op, 1) == 0.0f && ggml_get_op_params_f32(op, 2) == 0.0f &&
           (mask == nullptr || (mask->ne[2] == 1 && mask->ne[3] == 1));
}

// the reference of the split's step 1, also its path without the RVV kernels or without TCM: blocks of 64
// positions, every query head, ggml-cpu's one-row algorithm with V summed in F32; the partial result as
// spacemit_rvv_attn::split_end writes it (per query head the maximum, the sum, then DV floats)
struct split_ref {
    static constexpr int64_t block = 64;

    const ggml_tensor *  dst;
    const ggml_tensor *  q, *k, *v, *mask;
    int64_t              DK, DV, n_kv, n_head, rk2;
    float                scale;
    std::vector<float>   q16, acc, M, S;
    const ggml_fp16_t *  mp;

    explicit split_ref(const ggml_tensor * dst) :
        dst(dst),
        q(dst->src[0]),
        k(dst->src[1]),
        v(dst->src[2]),
        mask(dst->src[3]),
        DK(k->ne[0]),
        DV(v->ne[0]),
        n_kv(k->ne[1]),
        n_head(q->ne[2]),
        rk2(q->ne[2] / k->ne[2]),
        scale(ggml_get_op_params_f32(dst, 0)),
        q16(n_head * DK),
        acc(n_head * DV, 0.0f),
        M(n_head, -INFINITY),
        S(n_head, 0.0f),
        mp(mask != nullptr ? (const ggml_fp16_t *) mask->data : nullptr) {
        for (int64_t h = 0; h < n_head; ++h) {
            const float * pq = (const float *) ((const char *) q->data + h * q->nb[2]);
            for (int64_t d = 0; d < DK; ++d) {
                q16[h * DK + d] = GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(pq[d]));
            }
        }
    }

    int64_t n_blocks() const { return (n_kv + block - 1) / block; }

    float mask_value(int64_t ic) const { return mp != nullptr ? GGML_FP16_TO_FP32(mp[ic]) : 0.0f; }

    // whether block b has a visible position
    bool stage(int64_t b) const {
        for (int64_t ic = b * block; ic < std::min(n_kv, (b + 1) * block); ++ic) {
            if (mask_value(ic) != -INFINITY) {
                return true;
            }
        }
        return false;
    }

    void compute(int64_t b) {
        for (int64_t ic = b * block; ic < std::min(n_kv, (b + 1) * block); ++ic) {
            const float mv = mask_value(ic);
            if (mv == -INFINITY) {
                continue;
            }
            for (int64_t h = 0; h < n_head; ++h) {
                const ggml_fp16_t * kr =
                    (const ggml_fp16_t *) ((const char *) k->data + ic * k->nb[1] + (h / rk2) * k->nb[2]);
                float s = 0.0f;
                for (int64_t d = 0; d < DK; ++d) {
                    s += q16[h * DK + d] * GGML_FP16_TO_FP32(kr[d]);
                }
                s = s * scale + mv;

                float ms = 1.0f;
                float vs = 1.0f;
                if (s > M[h]) {
                    ms   = expf(M[h] - s);
                    M[h] = s;
                    for (int64_t d = 0; d < DV; ++d) {
                        acc[h * DV + d] *= ms;
                    }
                } else {
                    vs = expf(s - M[h]);
                }
                const ggml_fp16_t * vr =
                    (const ggml_fp16_t *) ((const char *) v->data + ic * v->nb[1] + (h / rk2) * v->nb[2]);
                for (int64_t d = 0; d < DV; ++d) {
                    acc[h * DV + d] += GGML_FP16_TO_FP32(vr[d]) * vs;
                }
                S[h] = S[h] * ms + vs;
            }
        }
    }

    void end(float * partial) const {
        for (int64_t h = 0; h < n_head; ++h) {
            float * p = partial + h * (DV + 2);
            p[0]      = M[h];
            p[1]      = S[h];
            std::copy(acc.begin() + h * DV, acc.begin() + (h + 1) * DV, p + 2);
        }
    }
};

// a tile's blocks of the split: b = ith, ith + nth, ..., each staged (stage(b): false when the mask hides all of it)
// and then computed. With paired, as the 1-row matmuls' paired (flagos-spacemit-ime.cpp): the even tile of a pair
// stages its first block while the odd one waits, then in every round one computes while the other stages. Both wait
// 1 + rounds times, rounds being the even tile's number of blocks (the odd tile may have one fewer and still waits in
// the last round); a tile without a partner runs as without the turns. False when a wait gave up
template <typename Stage, typename Compute>
bool split_blocks(const spacemit_tile & tile, int64_t n_blocks, bool paired, Stage && stage, Compute && compute) {
    const int64_t ith     = tile.ith;
    const int64_t nth     = tile.nth;
    bool          visible = false;  // the staged block has a visible position
    const auto    stage_b = [&](int64_t b) { visible = b < n_blocks && stage(b); };
    const auto    comp_b  = [&](int64_t b) {
        if (visible) {
            compute(b);
        }
    };
    if (!paired) {
        for (int64_t b = ith; b < n_blocks; b += nth) {
            stage_b(b);
            comp_b(b);
        }
        return true;
    }
    const int64_t first   = ith & ~int64_t(1);
    const int64_t rounds  = first < n_blocks ? (n_blocks - first + nth - 1) / nth : 0;
    const bool    even    = ith % 2 == 0;
    const bool    partner = (ith ^ 1) < nth;
    const auto    wait    = [&] { return !partner || spacemit_pair_wait(tile); };
    if (rounds == 0) {
        return true;
    }
    int64_t b = ith;
    if (even) {
        stage_b(b);
    }
    if (!wait()) {
        return false;
    }
    if (!even) {
        stage_b(b);
    }
    for (int64_t r = 0; r < rounds; r++, b += nth) {
        if (!even && !wait()) {
            return false;
        }
        comp_b(b);
        if (even && !wait()) {
            return false;
        }
        stage_b(b + nth);
    }
    return true;
}

// step 1 of the split: this tile's partial result into its slice of the workspace
bool split_partial(const spacemit_tile & tile, ggml_tensor * dst, bool paired) {
    const int64_t n_head = dst->src[0]->ne[2];
    const int64_t DV     = dst->src[2]->ne[0];
    GGML_ASSERT(tile.nth <= SPACEMIT_MAX_TILES && tile.workspace != nullptr &&
                tile.workspace_size >= spacemit_flash_attn_split_workspace(dst));
    float * partial = (float *) tile.workspace + (size_t) tile.ith * n_head * (DV + 2);
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    const int64_t DK    = dst->src[1]->ne[0];
    const int64_t block = tile.tcm != nullptr ? spacemit_rvv_attn::split_block_positions(dst, tile.tcm_size) : 0;
    if (!spacemit_use_reference() && __riscv_vlenb() == 128 && spacemit_rvv_attn::supported_shape(DK, DV) &&
        block >= 8) {
        spacemit_rvv_attn::split_begin(dst, tile.tcm, tile.tcm_size);
        const bool ok = split_blocks(
            tile, (dst->src[1]->ne[1] + block - 1) / block, paired,
            [&](int64_t b) { return spacemit_rvv_attn::split_stage(dst, b, tile.tcm, tile.tcm_size); },
            [&](int64_t b) { spacemit_rvv_attn::split_compute(dst, b, tile.tcm, tile.tcm_size); });
        if (!ok) {
            return false;
        }
        spacemit_rvv_attn::split_end(dst, tile.tcm, tile.tcm_size, partial);
        return true;
    }
#endif
    split_ref  ref(dst);
    const auto stage   = [&](int64_t b) { return ref.stage(b); };
    const auto compute = [&](int64_t b) { ref.compute(b); };
    if (!split_blocks(tile, ref.n_blocks(), paired, stage, compute)) {
        return false;
    }
    ref.end(partial);
    return true;
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
    // keeps its scratch in TCM, the one-chunk kernels use TCM when there is some and otherwise go row by row
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

int spacemit_flash_attn_split(const ggml_tensor * op) {
    const int setting = split_setting_from_env();
    return setting != 0 && split_shape(op) ? setting : 0;
}

// a slice per tile: per query head the maximum, the sum and DV floats
size_t spacemit_flash_attn_split_workspace(const ggml_tensor * node) {
    return (size_t) SPACEMIT_MAX_TILES * (size_t) node->src[0]->ne[2] * (size_t) (node->src[2]->ne[0] + 2) *
           sizeof(float);
}

bool spacemit_kernel_flash_attn_split_partial(const spacemit_tile & tile, ggml_tensor * dst) {
    return split_partial(tile, dst, false);
}

bool spacemit_kernel_flash_attn_split_partial_paired(const spacemit_tile & tile, ggml_tensor * dst) {
    return split_partial(tile, dst, true);
}

// step 2 of the split: each tile merges the partial results of its query heads, the tiles' maxima brought to the
// largest, adds the sinks as ggml-cpu does after its positions, and writes dst (one token: query head h's row at h)
bool spacemit_kernel_flash_attn_split_merge(const spacemit_tile & tile, ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * sinks = dst->src[4];
    const int64_t       n_head = q->ne[2];
    const int64_t       DV     = dst->src[2]->ne[0];
    const int64_t       stride = DV + 2;
    const float *       part   = (const float *) tile.workspace;
    int64_t             h0, h1;
    tile_range(n_head, tile, h0, h1);
    for (int64_t h = h0; h < h1; ++h) {
        float M = -INFINITY;
        for (uint32_t t = 0; t < tile.nth; ++t) {
            M = std::max(M, part[(t * n_head + h) * stride]);
        }
        float S                  = 0.0f;
        float out[max_head_size] = {};
        if (M != -INFINITY) {
            for (uint32_t t = 0; t < tile.nth; ++t) {
                const float * p = part + (t * n_head + h) * stride;
                if (p[1] == 0.0f) {
                    continue;  // no visible position in this tile's blocks
                }
                const float w = expf(p[0] - M);
                S += p[1] * w;
                for (int64_t d = 0; d < DV; ++d) {
                    out[d] += p[2 + d] * w;
                }
            }
        }
        if (sinks != nullptr) {
            const float s  = ((const float *) sinks->data)[h];
            float       ms = 1.0f;
            float       vs = 1.0f;
            if (s > M) {
                ms = expf(M - s);
                for (int64_t d = 0; d < DV; ++d) {
                    out[d] *= ms;
                }
            } else {
                vs = expf(s - M);
            }
            S = S * ms + vs;
        }
        const float S_inv = S == 0.0f ? 0.0f : 1.0f / S;
        float *     o     = (float *) ((char *) dst->data + h * dst->nb[1]);
        for (int64_t d = 0; d < DV; ++d) {
            o[d] = out[d] * S_inv;
        }
    }
    return true;
}
