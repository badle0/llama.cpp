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
// The RVV kernel is ported from SpacemiT's fork (ggml-spacemit's rvv_kernels.cpp, branch mtmd-backend @ ddfb41ddc,
// forward_gated_delta_net and its row helpers): the token recurrence with a fused one-pass row update for up to 4
// tokens and a two-pass one beyond. Changes, each marked "flagos": the row range and the scratch come from the tile
// instead of the fork's context and std::vectors, the asserts are gone (the contract refuses what they checked,
// core/flagos-spacemit-ops.cpp), and its chunkwise prefill and TCM-transposed paths are not ported. It computes a
// row's attention score from the decayed row (S q + delta k.q) and adds in vector reductions, a different rounding
// from the reference's, so GGML_FLAGOS_KERNEL_VARIANTS=0 keeps the reference.

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

// the floats one tile takes of the workspace: three vectors of S_v (the RVV kernel's delta, decay and attention sums;
// the reference's delta is the first), and the working state when K > 1, padded to a cache line so two tiles never
// write one line
int64_t gdn_tile_floats(const gdn_shape & sh) {
    const int64_t floats = 3 * sh.S_v + (sh.K > 1 ? sh.S_v * sh.S_v : 0);
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

// SpacemiT's rvv_gdn_fused_row_f32, both forms, copied: one pass over a state row decays it, takes its dot products
// with k and q, applies the delta-rule correction and stores it; returns the row's attention score, (S q + delta k.q)
// times scale, the updated row's dot with q by algebra. src and dst alias only as the same row
inline float rvv_gdn_fused_row_f32(float *            dst,
                                   const float *      src,
                                   const vfloat32m8_t kv,
                                   const vfloat32m8_t qv,
                                   const vfloat32m8_t dv,
                                   const float        v_j,
                                   const float        beta,
                                   const float        kq,
                                   const float        scale,
                                   const size_t       vl) {
    vfloat32m8_t s      = __riscv_vle32_v_f32m8(src, vl);
    s                   = __riscv_vfmul_vv_f32m8(s, dv, vl);
    vfloat32m1_t sum_k  = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t sum_q  = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    sum_k               = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(s, kv, vl), sum_k, vl);
    sum_q               = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(s, qv, vl), sum_q, vl);
    const float state_k = __riscv_vfmv_f_s_f32m1_f32(sum_k);
    const float state_q = __riscv_vfmv_f_s_f32m1_f32(sum_q);
    const float delta_j = (v_j - state_k) * beta;
    s                   = __riscv_vfmacc_vf_f32m8(s, delta_j, kv, vl);
    __riscv_vse32_v_f32m8(dst, s, vl);
    return (state_q + delta_j * kq) * scale;
}

// a row that fits in m4 leaves registers for the state and the reductions beside k, q and the decay
inline float rvv_gdn_fused_row_f32(float *            dst,
                                   const float *      src,
                                   const vfloat32m4_t kv,
                                   const vfloat32m4_t qv,
                                   const vfloat32m4_t dv,
                                   const float        v_j,
                                   const float        beta,
                                   const float        kq,
                                   const float        scale,
                                   const size_t       vl) {
    vfloat32m4_t s      = __riscv_vle32_v_f32m4(src, vl);
    s                   = __riscv_vfmul_vv_f32m4(s, dv, vl);
    vfloat32m1_t sum_k  = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t sum_q  = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    sum_k               = __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(s, kv, vl), sum_k, vl);
    sum_q               = __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(s, qv, vl), sum_q, vl);
    const float state_k = __riscv_vfmv_f_s_f32m1_f32(sum_k);
    const float state_q = __riscv_vfmv_f_s_f32m1_f32(sum_q);
    const float delta_j = (v_j - state_k) * beta;
    s                   = __riscv_vfmacc_vf_f32m4(s, delta_j, kv, vl);
    __riscv_vse32_v_f32m4(dst, s, vl);
    return (state_q + delta_j * kq) * scale;
}

// SpacemiT's rvv_gdn_dot_f32, copied
inline float rvv_gdn_dot_f32(const float * x, const float * y, int64_t n) {
    vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (int64_t i = 0; i < n;) {
        const size_t       vl   = __riscv_vsetvl_e32m8(n - i);
        const vfloat32m8_t xv   = __riscv_vle32_v_f32m8(x + i, vl);
        const vfloat32m8_t yv   = __riscv_vle32_v_f32m8(y + i, vl);
        const vfloat32m8_t prod = __riscv_vfmul_vv_f32m8(xv, yv, vl);
        sum                     = __riscv_vfredusum_vs_f32m8_f32m1(prod, sum, vl);
        i += (int64_t) vl;
    }
    return __riscv_vfmv_f_s_f32m1_f32(sum);
}

struct rvv_gdn_decay_dots_f32 {
    float state_k;
    float state_q;
};

// SpacemiT's two-pass row update, copied: this pass decays the row, stores it and reduces both dot products; the
// second (rvv_gdn_update_f32) applies the correction. The early store keeps the row loads pipelined, which SpacemiT
// measured faster than the fused form from about 8 tokens a call
inline rvv_gdn_decay_dots_f32 rvv_gdn_decay_dots_f32_impl(float *       state,
                                                          const float * decay,
                                                          float         decay_scalar,
                                                          const float * k,
                                                          const float * q,
                                                          int64_t       n,
                                                          bool          kda) {
    vfloat32m1_t sum_k = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t sum_q = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (int64_t i = 0; i < n;) {
        const size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t s  = __riscv_vle32_v_f32m8(state + i, vl);
        if (kda) {
            const vfloat32m8_t d = __riscv_vle32_v_f32m8(decay + i, vl);
            s                    = __riscv_vfmul_vv_f32m8(s, d, vl);
        } else {
            s = __riscv_vfmul_vf_f32m8(s, decay_scalar, vl);
        }
        __riscv_vse32_v_f32m8(state + i, s, vl);

        const vfloat32m8_t kv = __riscv_vle32_v_f32m8(k + i, vl);
        const vfloat32m8_t qv = __riscv_vle32_v_f32m8(q + i, vl);
        sum_k                 = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(s, kv, vl), sum_k, vl);
        sum_q                 = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(s, qv, vl), sum_q, vl);
        i += (int64_t) vl;
    }
    return { __riscv_vfmv_f_s_f32m1_f32(sum_k), __riscv_vfmv_f_s_f32m1_f32(sum_q) };
}

inline void rvv_gdn_update_f32(float * state, const float * k, float delta, int64_t n) {
    for (int64_t i = 0; i < n;) {
        const size_t       vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t       s  = __riscv_vle32_v_f32m8(state + i, vl);
        const vfloat32m8_t kv = __riscv_vle32_v_f32m8(k + i, vl);
        s                     = __riscv_vfmacc_vf_f32m8(s, delta, kv, vl);
        __riscv_vse32_v_f32m8(state + i, s, vl);
        i += (int64_t) vl;
    }
}

// SpacemiT's gdn_recurrent_rows_lookahead, copied: one token, a row of exactly one m4 register group, two rows at a
// time with the next two loaded before this pair is updated; d even
__attribute__((noinline)) void gdn_recurrent_rows_lookahead(float *       dst,
                                                            const float * src,
                                                            const float * k,
                                                            const float * q,
                                                            const float * v,
                                                            float         decay,
                                                            float         beta,
                                                            float         kq,
                                                            float         scale,
                                                            int64_t       d,
                                                            float *       out) {
    const size_t vl = __riscv_vsetvl_e32m4(d);
    const auto   kv = __riscv_vle32_v_f32m4(k, vl);
    const auto   qv = __riscv_vle32_v_f32m4(q, vl);
    auto         s0 = __riscv_vle32_v_f32m4(src, vl);
    auto         s1 = __riscv_vle32_v_f32m4(src + d, vl);
    for (int64_t j = 0; j < d; j += 2) {
        auto next0 = s0;
        auto next1 = s1;
        if (j + 2 < d) {
            next0 = __riscv_vle32_v_f32m4(src + (j + 2) * d, vl);
            next1 = __riscv_vle32_v_f32m4(src + (j + 3) * d, vl);
            // keep both loads before the arithmetic below in the generated code
            asm volatile("" : "+vr"(next0), "+vr"(next1));
        }
        s0                 = __riscv_vfmul_vf_f32m4(s0, decay, vl);
        s1                 = __riscv_vfmul_vf_f32m4(s1, decay, vl);
        const auto  zero   = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        const auto  pk0    = __riscv_vfmul_vv_f32m4(s0, kv, vl);
        const auto  pk1    = __riscv_vfmul_vv_f32m4(s1, kv, vl);
        const auto  sk0    = __riscv_vfredusum_vs_f32m4_f32m1(pk0, zero, vl);
        const auto  sk1    = __riscv_vfredusum_vs_f32m4_f32m1(pk1, zero, vl);
        const auto  pq0    = __riscv_vfmul_vv_f32m4(s0, qv, vl);
        const auto  pq1    = __riscv_vfmul_vv_f32m4(s1, qv, vl);
        const auto  sq0    = __riscv_vfredusum_vs_f32m4_f32m1(pq0, zero, vl);
        const auto  sq1    = __riscv_vfredusum_vs_f32m4_f32m1(pq1, zero, vl);
        const float delta0 = (v[j] - __riscv_vfmv_f_s_f32m1_f32(sk0)) * beta;
        const float delta1 = (v[j + 1] - __riscv_vfmv_f_s_f32m1_f32(sk1)) * beta;
        s0                 = __riscv_vfmacc_vf_f32m4(s0, delta0, kv, vl);
        s1                 = __riscv_vfmacc_vf_f32m4(s1, delta1, kv, vl);
        __riscv_vse32_v_f32m4(dst + j * d, s0, vl);
        __riscv_vse32_v_f32m4(dst + (j + 1) * d, s1, vl);
        const float y0 = (__riscv_vfmv_f_s_f32m1_f32(sq0) + delta0 * kq) * scale;
        const float y1 = (__riscv_vfmv_f_s_f32m1_f32(sq1) + delta1 * kq) * scale;
        out[j]         = y0;
        out[j + 1]     = y1;
        if (j + 2 < d) {
            s0 = next0;
            s1 = next1;
        }
    }
}

// SpacemiT's forward_gated_delta_net, its token recurrence: up to 4 tokens with a row in one register group the fused
// one-pass row update (the first token reads the input state and writes the output, no copy), else the two-pass form.
// flagos: the row range and the scratch are the tile's (its kda_decay, delta, decay and state_q vectors are the
// workspace slice's three S_v floats, its working state for K > 1 the slice's S_v * S_v); the asserts are the
// contract's; its chunkwise prefill (16 tokens and up, thread_local scratch) and its TCM-transposed recurrence are not
// ported, so those calls take the two-pass form
void gdn_rvv(ggml_tensor * dst, float * scratch, float * state_work, int64_t ir0, int64_t ir1) {
    const ggml_tensor * src_q     = dst->src[0];
    const ggml_tensor * src_k     = dst->src[1];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_g     = dst->src[3];
    const ggml_tensor * src_beta  = dst->src[4];
    const ggml_tensor * src_state = dst->src[5];

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];
    const int64_t K        = ggml_get_op_params_i32(dst, 0);

    const int64_t state_seq_stride    = (int64_t) (src_state->nb[3] / sizeof(float));
    const int64_t attn_score_elems    = S_v * H * n_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H * n_seqs;
    float *       attn_out_base       = (float *) dst->data;
    float *       state_out_base      = attn_out_base + attn_score_elems;
    const float * state_in_base       = (const float *) src_state->data;

    const float scale = 1.0f / sqrtf((float) S_v);
    // the scalar gate uses g[0]; KDA gives one gate per state column (S_v == 1: the two coincide, KDA kept)
    const bool  kda   = src_g->ne[0] == S_v;

    const bool fused     = n_tokens <= 4 && S_v <= (int64_t) __riscv_vsetvlmax_e32m8();
    float *    kda_decay = scratch;
    float *    delta     = scratch;
    float *    decay     = scratch + S_v;
    float *    state_q   = scratch + 2 * S_v;

    const int64_t rq3 = src_v->ne[3] / src_q->ne[3];
    const int64_t rk3 = src_v->ne[3] / src_k->ne[3];

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t iv1 = ir % H;
        const int64_t iv3 = ir / H;
        const int64_t iq1 = iv1 % src_q->ne[1];
        const int64_t ik1 = iv1 % src_k->ne[1];
        const int64_t iq3 = iv3 / rq3;
        const int64_t ik3 = iv3 / rk3;

        float *       s_out = K > 1 ? state_work : state_out_base + (iv3 * H + iv1) * S_v * S_v;
        const float * s_in  = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        if (!fused) {
            std::memcpy(s_out, s_in, (size_t) (S_v * S_v) * sizeof(float));
        }
        // the fused form reads the input state for its first token and writes the output, then works in place
        const float * s_cur = s_in;

        float * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;
        for (int64_t t = 0; t < n_tokens; ++t) {
            const float * q    = (const float *) ((const char *) src_q->data + iq3 * src_q->nb[3] + t * src_q->nb[2] +
                                               iq1 * src_q->nb[1]);
            const float * k    = (const float *) ((const char *) src_k->data + ik3 * src_k->nb[3] + t * src_k->nb[2] +
                                               ik1 * src_k->nb[1]);
            const float * v    = (const float *) ((const char *) src_v->data + iv3 * src_v->nb[3] + t * src_v->nb[2] +
                                               iv1 * src_v->nb[1]);
            const float   beta = *(const float *) ((const char *) src_beta->data + iv3 * src_beta->nb[3] +
                                                 t * src_beta->nb[2] + iv1 * src_beta->nb[1]);
            const float * g    = (const float *) ((const char *) src_g->data + iv3 * src_g->nb[3] + t * src_g->nb[2] +
                                               iv1 * src_g->nb[1]);

            const float kq = rvv_gdn_dot_f32(k, q, S_v);

            if (fused) {
                float decay_scalar = 0.0f;
                if (kda) {
                    for (int64_t i = 0; i < S_v; ++i) {
                        kda_decay[i] = expf(g[i]);
                    }
                } else {
                    decay_scalar = expf(g[0]);
                }

                if (n_tokens == 1 && K == 1 && !kda && S_v % 2 == 0 && S_v == (int64_t) __riscv_vsetvlmax_e32m4()) {
                    gdn_recurrent_rows_lookahead(s_out, s_cur, k, q, v, decay_scalar, beta, kq, scale, S_v, attn_data);
                } else if (S_v <= (int64_t) __riscv_vsetvlmax_e32m4()) {
                    const size_t       vl = __riscv_vsetvl_e32m4(S_v);
                    const vfloat32m4_t kv = __riscv_vle32_v_f32m4(k, vl);
                    const vfloat32m4_t qv = __riscv_vle32_v_f32m4(q, vl);
                    const vfloat32m4_t dv =
                        kda ? __riscv_vle32_v_f32m4(kda_decay, vl) : __riscv_vfmv_v_f_f32m4(decay_scalar, vl);
                    for (int64_t j = 0; j < S_v; ++j) {
                        attn_data[j] = rvv_gdn_fused_row_f32(s_out + j * S_v, s_cur + j * S_v, kv, qv, dv, v[j], beta,
                                                             kq, scale, vl);
                    }
                } else {
                    const size_t       vl = __riscv_vsetvl_e32m8(S_v);
                    const vfloat32m8_t kv = __riscv_vle32_v_f32m8(k, vl);
                    const vfloat32m8_t qv = __riscv_vle32_v_f32m8(q, vl);
                    const vfloat32m8_t dv =
                        kda ? __riscv_vle32_v_f32m8(kda_decay, vl) : __riscv_vfmv_v_f_f32m8(decay_scalar, vl);
                    for (int64_t j = 0; j < S_v; ++j) {
                        attn_data[j] = rvv_gdn_fused_row_f32(s_out + j * S_v, s_cur + j * S_v, kv, qv, dv, v[j], beta,
                                                             kq, scale, vl);
                    }
                }
                s_cur = s_out;
            } else {
                float decay_scalar = 0.0f;
                if (kda) {
                    for (int64_t i = 0; i < S_v; ++i) {
                        decay[i] = expf(g[i]);
                    }
                } else {
                    decay_scalar = expf(g[0]);
                }

                for (int64_t j = 0; j < S_v; ++j) {
                    const auto dots = rvv_gdn_decay_dots_f32_impl(s_out + j * S_v, decay, decay_scalar, k, q, S_v, kda);
                    delta[j]        = (v[j] - dots.state_k) * beta;
                    state_q[j]      = dots.state_q;
                }
                for (int64_t j = 0; j < S_v; ++j) {
                    rvv_gdn_update_f32(s_out + j * S_v, k, delta[j], S_v);
                    attn_data[j] = (state_q[j] + delta[j] * kq) * scale;
                }
            }
            attn_data += S_v * H;

            if (K > 1) {
                const int64_t target_slot = n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * state_snapshot =
                        state_out_base + target_slot * state_size_per_snap + (iv3 * H + iv1) * S_v * S_v;
                    std::memcpy(state_snapshot, s_out, (size_t) (S_v * S_v) * sizeof(float));
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

    float * scratch    = (float *) tile.workspace + per_tile * tile.ith;
    float * state_work = sh.K > 1 ? scratch + 3 * sh.S_v : nullptr;

    int64_t ir0, ir1;
    tile_range(sh.H * sh.n_seqs, tile, ir0, ir1);
    if (ir0 >= ir1) {
        return true;
    }
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference()) {
        gdn_rvv(dst, scratch, state_work, ir0, ir1);
        return true;
    }
#endif
    gdn_ref(dst, scratch, state_work, ir0, ir1);
    return true;
}
