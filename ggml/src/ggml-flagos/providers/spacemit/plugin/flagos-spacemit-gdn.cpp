// The Gated DeltaNet step (GGML_OP_GATED_DELTA_NET, the contract gdn.f32), Qwen3.5's recurrent layers in one node
// (plan.md "Step 2 spec", piece A).
//
// The reference follows ggml-cpu's ggml_compute_forward_gated_delta_net_one_chunk (ggml/src/ggml-cpu/ops.cpp:10735)
// with the same operations in the same order, scalar where ggml-cpu calls ggml_vec_mul_f32, ggml_vec_scale_f32,
// ggml_vec_dot_f32 and ggml_vec_mad_f32. Those are SIMD in ggml-cpu (vec.h, vec.cpp: several float accumulators and
// fused multiply-adds, the ISA deciding how many), so the results are close but not bit for bit ggml-cpu's; the check
// program says exactly what it compares against ggml-cpu within a bound and what it compares bit for bit.
//
// A row is a (head, sequence) pair, H * n_seqs of them, split over the tiles as ggml-cpu splits them over its
// threads. One tile computes a row whole, and the rows write disjoint parts of the output, so the result does not
// depend on the number of tiles.
//
// The RVV kernel is ported from SpacemiT's (upstream llama.cpp master, ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp,
// rvv_dot_f32 and forward_gated_delta_net_one_chunk in namespace spacemit_kernels::rvv; fetched 2026-11-13). Changes,
// each marked "flagos": the row range and the scratch come from the tile instead of ggml's compute params, and the
// asserts are gone (the contract refuses what they checked, core/flagos-spacemit-ops.cpp). Its vector reductions make
// it a different rounding from the reference's, so GGML_FLAGOS_KERNEL_VARIANTS=0 keeps the reference.

#include "flagos-spacemit-gdn.h"

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

// the node's facts: the state side, the heads, the tokens, the sequences and the snapshot slots
struct gdn_shape {
    int64_t S_v      = 0;
    int64_t H        = 0;
    int64_t n_tokens = 0;
    int64_t n_seqs   = 0;
    int64_t K        = 0;
};

gdn_shape gdn_shape_of(const ggml_tensor * node) {
    const ggml_tensor * v = node->src[2];
    gdn_shape           sh;
    sh.S_v      = v->ne[0];
    sh.H        = v->ne[1];
    sh.n_tokens = v->ne[2];
    sh.n_seqs   = v->ne[3];
    sh.K        = ggml_get_op_params_i32(node, 0);
    return sh;
}

// the floats one tile takes of the workspace: ggml-cpu's per-thread scratch (delta, and the working state when K > 1),
// padded to a cache line so two tiles never write one line
int64_t gdn_tile_floats(const gdn_shape & sh) {
    const int64_t floats = sh.S_v + (sh.K > 1 ? sh.S_v * sh.S_v : 0);
    return GGML_PAD(floats, 64 / (int64_t) sizeof(float));
}

// ops.cpp ggml_compute_forward_gated_delta_net_one_chunk, its ggml_vec_* calls written out element by element
void gdn_ref(ggml_tensor * dst, float * delta, float * state_work, int64_t ir0, int64_t ir1) {
    const ggml_tensor * src_q     = dst->src[0];
    const ggml_tensor * src_k     = dst->src[1];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_g     = dst->src[3];
    const ggml_tensor * src_beta  = dst->src[4];
    const ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, src_q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, src_k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, src_v, nb)
    GGML_TENSOR_LOCALS(int64_t, neg, src_g, ne)
    GGML_TENSOR_LOCALS(size_t, nbg, src_g, nb)
    GGML_TENSOR_LOCALS(size_t, nbb, src_beta, nb)

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool    kda              = (neg0 == S_v);
    const int64_t K                = ggml_get_op_params_i32(dst, 0);
    const int64_t state_seq_stride = (int64_t) (src_state->nb[3] / sizeof(float));

    // the output is the attention scores, then K state snapshots (slot 0 the newest)
    const int64_t attn_score_elems    = S_v * H * n_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H * n_seqs;
    float *       attn_out_base       = (float *) dst->data;
    float *       state_out_base      = (float *) dst->data + attn_score_elems;

    const float * state_in_base = (const float *) src_state->data;

    const int64_t rq3 = nev3 / neq3;
    const int64_t rk3 = nev3 / nek3;

    const float scale = 1.0f / sqrtf((float) S_v);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t iv1 = ir % H;  // head
        const int64_t iv3 = ir / H;  // sequence

        const int64_t iq1 = iv1 % neq1;  // q and k repeat over the heads
        const int64_t ik1 = iv1 % nek1;

        const int64_t iq3 = iv3 / rq3;  // and over the sequences
        const int64_t ik3 = iv3 / rk3;

        // K = 1: the one output slot is the working state; K > 1: scratch, copied into a slot per token
        float * s_out = (K > 1) ? state_work : state_out_base + (iv3 * H + iv1) * S_v * S_v;

        const float * s_in = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        std::memcpy(s_out, s_in, (size_t) (S_v * S_v) * sizeof(float));

        float * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;

        for (int64_t t = 0; t < n_tokens; t++) {
            const float * q_d = (const float *) ((const char *) src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            const float * k_d = (const float *) ((const char *) src_k->data + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            const float * v_d = (const float *) ((const char *) src_v->data + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            const float beta_val = *(const float *) ((const char *) src_beta->data + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            const float * g_d    = (const float *) ((const char *) src_g->data + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // the state is stored transposed: s_out[j * S_v + i] is S[i][j], so row j is column j of S
            if (kda) {
                for (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                for (int64_t j = 0; j < S_v; ++j) {
                    float * row = s_out + j * S_v;
                    for (int64_t i = 0; i < S_v; ++i) {
                        row[i] = row[i] * delta[i];
                    }
                }
            } else {
                const float gate = expf(g_d[0]);
                for (int64_t i = 0; i < S_v * S_v; ++i) {
                    s_out[i] *= gate;
                }
            }

            // delta[j] = (v[j] - dot(row j, k)) * beta
            for (int64_t j = 0; j < S_v; ++j) {
                const float * row = s_out + j * S_v;
                float         sum = 0.0f;
                for (int64_t i = 0; i < S_v; ++i) {
                    sum += row[i] * k_d[i];
                }
                delta[j] = (v_d[j] - sum) * beta_val;
            }

            // the outer product: row j += delta[j] * k
            for (int64_t j = 0; j < S_v; ++j) {
                float *     row = s_out + j * S_v;
                const float dj  = delta[j];
                for (int64_t i = 0; i < S_v; ++i) {
                    row[i] += k_d[i] * dj;
                }
            }

            // the attention score: dot(row j, q) * scale
            for (int64_t j = 0; j < S_v; ++j) {
                const float * row = s_out + j * S_v;
                float         sum = 0.0f;
                for (int64_t i = 0; i < S_v; ++i) {
                    sum += row[i] * q_d[i];
                }
                attn_data[j] = sum * scale;
            }

            attn_data += S_v * H;  // the next token

            if (K > 1) {
                const int64_t target_slot = n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * curr_state_o =
                        state_out_base + target_slot * state_size_per_snap + (iv3 * H + iv1) * S_v * S_v;
                    std::memcpy(curr_state_o, s_out, (size_t) (S_v * S_v) * sizeof(float));
                }
            }
        }
    }
}

#if defined(GGML_FLAGOS_SPACEMIT_RVV)

// SpacemiT's rvv_dot_f32, copied
float rvv_dot_f32(const float * x, const float * y, int64_t n) {
    size_t       vl   = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t vsum = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    for (int64_t i = 0; i < n; i += (int64_t) vl) {
        vl                = __riscv_vsetvl_e32m4(n - i);
        vfloat32m4_t ax   = __riscv_vle32_v_f32m4(x + i, vl);
        vfloat32m4_t ay   = __riscv_vle32_v_f32m4(y + i, vl);
        vsum              = __riscv_vfmacc_vv_f32m4_tu(vsum, ax, ay, vl);
    }
    vl              = __riscv_vsetvlmax_e32m4();
    vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vs              = __riscv_vfredusum_vs_f32m4_f32m1(vsum, vs, vl);
    return __riscv_vfmv_f_s_f32m1_f32(vs);
}

// SpacemiT's forward_gated_delta_net_one_chunk, copied; flagos: the row range and the scratch are the tile's, the
// asserts are the contract's
void gdn_rvv(ggml_tensor * dst, float * delta, float * state_work, int64_t ir0, int64_t ir1) {
    const ggml_tensor * src_q     = dst->src[0];
    const ggml_tensor * src_k     = dst->src[1];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_g     = dst->src[3];
    const ggml_tensor * src_beta  = dst->src[4];
    const ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, src_q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, src_k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, src_v, nb)
    GGML_TENSOR_LOCALS(int64_t, neg, src_g, ne)
    GGML_TENSOR_LOCALS(size_t, nbg, src_g, nb)
    GGML_TENSOR_LOCALS(size_t, nbb, src_beta, nb)

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool    kda              = (neg0 == S_v);
    const int64_t K                = ggml_get_op_params_i32(dst, 0);
    const int64_t state_seq_stride = (int64_t) (src_state->nb[3] / sizeof(float));

    const int64_t attn_score_elems    = S_v * H * n_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H * n_seqs;
    float *       attn_out_base       = (float *) dst->data;
    float *       state_out_base      = (float *) dst->data + attn_score_elems;

    const float * state_in_base = (const float *) src_state->data;

    const int64_t rq3 = nev3 / neq3;
    const int64_t rk3 = nev3 / nek3;

    const float scale = 1.0f / sqrtf((float) S_v);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t iv1 = ir % H;
        const int64_t iv3 = ir / H;

        const int64_t iq1 = iv1 % neq1;
        const int64_t ik1 = iv1 % nek1;

        const int64_t iq3 = iv3 / rq3;
        const int64_t ik3 = iv3 / rk3;

        float * s_out = (K > 1) ? state_work : state_out_base + (iv3 * H + iv1) * S_v * S_v;

        const float * s_in = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        std::memcpy(s_out, s_in, (size_t) (S_v * S_v) * sizeof(float));

        float * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;

        for (int64_t t = 0; t < n_tokens; t++) {
            const float * q_d = (const float *) ((const char *) src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            const float * k_d = (const float *) ((const char *) src_k->data + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            const float * v_d = (const float *) ((const char *) src_v->data + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            const float beta_val = *(const float *) ((const char *) src_beta->data + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            const float * g_d    = (const float *) ((const char *) src_g->data + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            if (kda) {
                for (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                for (int64_t j = 0; j < S_v; ++j) {
                    float *       row       = s_out + j * S_v;
                    const float * d         = delta;
                    int64_t       remaining = S_v;
                    while (remaining > 0) {
                        const size_t vl = __riscv_vsetvl_e32m4(remaining);
                        vfloat32m4_t vr = __riscv_vle32_v_f32m4(row, vl);
                        vfloat32m4_t vd = __riscv_vle32_v_f32m4(d, vl);
                        vr              = __riscv_vfmul_vv_f32m4(vr, vd, vl);
                        __riscv_vse32_v_f32m4(row, vr, vl);
                        row += vl;
                        d += vl;
                        remaining -= (int64_t) vl;
                    }
                }
            } else {
                const float gate      = expf(g_d[0]);
                float *     p         = s_out;
                int64_t     remaining = S_v * S_v;
                while (remaining > 0) {
                    const size_t vl = __riscv_vsetvl_e32m4(remaining);
                    vfloat32m4_t v  = __riscv_vle32_v_f32m4(p, vl);
                    v               = __riscv_vfmul_vf_f32m4(v, gate, vl);
                    __riscv_vse32_v_f32m4(p, v, vl);
                    p += vl;
                    remaining -= (int64_t) vl;
                }
            }

            for (int64_t j = 0; j < S_v; ++j) {
                const float sum = rvv_dot_f32(s_out + j * S_v, k_d, S_v);
                delta[j]        = (v_d[j] - sum) * beta_val;
            }

            for (int64_t j = 0; j < S_v; ++j) {
                float *       row       = s_out + j * S_v;
                const float * k         = k_d;
                const float   dj        = delta[j];
                int64_t       remaining = S_v;
                while (remaining > 0) {
                    const size_t vl = __riscv_vsetvl_e32m4(remaining);
                    vfloat32m4_t vr = __riscv_vle32_v_f32m4(row, vl);
                    vfloat32m4_t vk = __riscv_vle32_v_f32m4(k, vl);
                    vr              = __riscv_vfmacc_vf_f32m4(vr, dj, vk, vl);
                    __riscv_vse32_v_f32m4(row, vr, vl);
                    row += vl;
                    k += vl;
                    remaining -= (int64_t) vl;
                }
            }

            for (int64_t j = 0; j < S_v; ++j) {
                const float sum = rvv_dot_f32(s_out + j * S_v, q_d, S_v);
                attn_data[j]    = sum * scale;
            }

            attn_data += S_v * H;

            if (K > 1) {
                const int64_t target_slot = n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * curr_state_o =
                        state_out_base + target_slot * state_size_per_snap + (iv3 * H + iv1) * S_v * S_v;
                    std::memcpy(curr_state_o, s_out, (size_t) (S_v * S_v) * sizeof(float));
                }
            }
        }
    }
}

#endif  // GGML_FLAGOS_SPACEMIT_RVV

}  // namespace

size_t spacemit_gdn_workspace(const ggml_tensor * node) {
    return (size_t) gdn_tile_floats(gdn_shape_of(node)) * SPACEMIT_MAX_TILES * sizeof(float);
}

bool spacemit_kernel_gated_delta_net(const spacemit_tile * tile_p, ggml_tensor * dst) {
    const spacemit_tile & tile     = *tile_p;
    const gdn_shape       sh       = gdn_shape_of(dst);
    const int64_t         per_tile = gdn_tile_floats(sh);
    const size_t          needed   = (size_t) per_tile * tile.nth * sizeof(float);
    if (tile.nth == 0 || tile.nth > SPACEMIT_MAX_TILES || tile.workspace == nullptr ||
        tile.workspace_size < needed) {
        GGML_LOG_ERROR("FlagOS SpacemiT: gdn.f32 needs %zu scratch bytes for %u tiles, has %zu\n", needed, tile.nth,
                       tile.workspace_size);
        return false;
    }

    float * delta      = (float *) tile.workspace + per_tile * tile.ith;
    float * state_work = sh.K > 1 ? delta + sh.S_v : nullptr;

    int64_t ir0, ir1;
    tile_range(sh.H * sh.n_seqs, tile, ir0, ir1);
    if (ir0 >= ir1) {
        return true;
    }
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference()) {
        gdn_rvv(dst, delta, state_work, ir0, ir1);
        return true;
    }
#endif
    gdn_ref(dst, delta, state_work, ir0, ir1);
    return true;
}
