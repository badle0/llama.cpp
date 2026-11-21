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
// tokens and a two-pass one beyond. Changes, each marked "flagos": the row range, the scratch and the TCM come from the
// tile instead of the fork's context, std::vectors and thread_local buffer, the asserts are gone (the contract refuses
// what they checked, core/flagos-spacemit-ops.cpp), and its two prefill forms (K 1, scalar gate, 16 tokens and up),
// the chunkwise form and the TCM-transposed recurrence, run by default in the fork's order (FLAGOS_SPACEMIT_GDN_PREFILL
// auto: the TCM form, else the chunkwise one, else the recurrence); recurrence, chunk or tcm pick one form. The
// chunkwise form is not bit for bit the recurrence. It computes a row's attention score from the decayed row
// (S q + delta k.q) and adds in vector reductions, a different rounding from the reference's, so
// GGML_FLAGOS_KERNEL_VARIANTS=0 keeps the reference.

#include "flagos-spacemit-gdn.h"

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

namespace {

enum gdn_prefill : int { gdn_prefill_recurrence = 0, gdn_prefill_chunk = 1, gdn_prefill_tcm = 2, gdn_prefill_auto = 3 };

// FLAGOS_SPACEMIT_GDN_PREFILL: auto (unset or empty, the default): the fork's order, the TCM-transposed recurrence
// where its conditions hold, else the chunkwise form where its conditions hold, else the token recurrence; recurrence:
// the token recurrence for every call; chunk or tcm: that form where its conditions hold, else the recurrence. Parsed
// once (spacemit_cached_switch); any other value warns once per parse generation and keeps the default
int parse_gdn_prefill() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_GDN_PREFILL");
    if (v == nullptr || *v == '\0' || std::strcmp(v, "auto") == 0) {
        return gdn_prefill_auto;
    }
    if (std::strcmp(v, "recurrence") == 0) {
        return gdn_prefill_recurrence;
    }
    if (std::strcmp(v, "chunk") == 0) {
        return gdn_prefill_chunk;
    }
    if (std::strcmp(v, "tcm") == 0) {
        return gdn_prefill_tcm;
    }
    static std::atomic<uint32_t> warned_in{ 0 };
    const uint32_t               generation = spacemit_switches_generation();
    if (warned_in.exchange(generation) != generation) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_GDN_PREFILL=%s: not auto, recurrence, chunk or tcm, the default (auto) kept\n",
                      __func__, v);
    }
    return gdn_prefill_auto;
}

int gdn_prefill_from_env() {
    static std::atomic<int>      value{ gdn_prefill_auto };
    static std::atomic<uint32_t> seen{ 0 };
    return spacemit_cached_switch(value, seen, parse_gdn_prefill);
}

// test-only (spacemit_gdn_paths_take): the tile calls that had rows, by the form they ran
std::atomic<uint64_t> g_gdn_reference{ 0 }, g_gdn_fused{ 0 }, g_gdn_two_pass{ 0 }, g_gdn_chunk{ 0 }, g_gdn_tcm{ 0 };

// the fork's chunk of the chunkwise form; also the fewest tokens either prefill form takes
constexpr int64_t GDN_CHUNK_SIZE = 16;

// the chunkwise form's buffer per head: the state, seven token tiles, three chunk matrices and four per-token vectors
int64_t gdn_chunk_floats(int64_t D) {
    const int64_t CS = GDN_CHUNK_SIZE;
    return D * D + 7 * CS * D + 3 * CS * CS + 4 * CS;
}

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
// write one line. A node the chunkwise form could take (K 1, 16 tokens and up) gets its buffer in the same place,
// whatever FLAGOS_SPACEMIT_GDN_PREFILL says (auto and chunk can take that form; the switch may change between the
// workspace's sizing and the launch only in a test)
int64_t gdn_tile_floats(const gdn_shape & sh) {
    int64_t floats = 3 * sh.S_v + (sh.K > 1 ? sh.S_v * sh.S_v : 0);
    if (sh.K == 1 && sh.n_tokens >= GDN_CHUNK_SIZE) {
        floats = std::max(floats, gdn_chunk_floats(sh.S_v));
    }
    return GGML_PAD(floats, 64 / (int64_t) sizeof(float));
}

// ops.cpp ggml_compute_forward_gated_delta_net_one_chunk, its ggml_vec_* calls written out element by element.
// flagos: the state comes from state_in_base and the new state goes to state_out_base, which with K 1 may be the same
// row (the state pass, core/flagos-spacemit-ops.h spacemit_state_pass): the only read of the input state is the copy
// into s_out, skipped when the two coincide
void gdn_ref(ggml_tensor * dst, const float * state_in_base, float * state_out_base, float * delta, float * state_work,
             int64_t ir0, int64_t ir1) {
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

    // the output starts with the attention scores; the K state snapshots after them (slot 0 the newest) are the
    // caller's state_out_base
    const int64_t state_size_per_snap = S_v * S_v * H * n_seqs;
    float *       attn_out_base       = (float *) dst->data;

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
        if (s_out != s_in) {  // the state pass gives the same row for both; the work below is in place
            std::memcpy(s_out, s_in, (size_t) (S_v * S_v) * sizeof(float));
        }

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

// SpacemiT's chunkwise prefill, copied (gdn_gemm, gdn_transpose, gdn_chunked_head). Per head, with S the k x v state
// (the op stores its transpose), chunk-local cumulative log decay gam_r = sum_{s<=r} g_s, Gam_r = exp(gam_r) and
// E_rs = exp(gam_r - gam_s) <= 1 for s <= r:
//   U   = (I + L)^-1 * beta (V - Gam K S0),  L_rs = beta_r E_rs (k_r . k_s), s < r   (forward substitution)
//   O   = scale (Gam Q S0 + (E o QK^T)_lower-incl U)
//   S1  = Gam_C S0 + (E_C. o K)^T U
// which equals the token recurrence up to float rounding. All products are 4-row blocked RVV matmuls. SpacemiT's
// numbers for chunk 16: Qwen3.6 pp512 GDN ms/token 1.29 against the recurrence's 2.56, KLD 0.004-0.005 against it

// 128 bytes is a suitable alignment size for the TCM state buffer.
constexpr size_t GDN_TCM_ALIGNMENT = 128;

// C[m x n] = (C0 ? c0s * C0 : 0) + A[m x k] B[k x n]. A rows strided by lda; B, C, C0 rows contiguous
// with leading dims ldb, ldc, ldc0; n <= vlmax(e32m4). C may alias C0 (row-wise in place).
void gdn_gemm(int64_t       m,
              int64_t       n,
              int64_t       k,
              const float * A,
              int64_t       lda,
              const float * B,
              int64_t       ldb,
              float *       C,
              int64_t       ldc,
              const float * C0,
              int64_t       ldc0,
              float         c0s) {
    const size_t vl = __riscv_vsetvl_e32m4(n);
    int64_t      r  = 0;
    for (; r + 4 <= m; r += 4) {
        vfloat32m4_t c0, c1, c2, c3;
        if (C0) {
            c0 = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(C0 + (r + 0) * ldc0, vl), c0s, vl);
            c1 = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(C0 + (r + 1) * ldc0, vl), c0s, vl);
            c2 = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(C0 + (r + 2) * ldc0, vl), c0s, vl);
            c3 = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(C0 + (r + 3) * ldc0, vl), c0s, vl);
        } else {
            c0 = c1 = c2 = c3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        }
        const float * a0 = A + (r + 0) * lda;
        const float * a1 = A + (r + 1) * lda;
        const float * a2 = A + (r + 2) * lda;
        const float * a3 = A + (r + 3) * lda;
        for (int64_t p = 0; p < k; ++p) {
            const vfloat32m4_t b = __riscv_vle32_v_f32m4(B + p * ldb, vl);
            c0                   = __riscv_vfmacc_vf_f32m4(c0, a0[p], b, vl);
            c1                   = __riscv_vfmacc_vf_f32m4(c1, a1[p], b, vl);
            c2                   = __riscv_vfmacc_vf_f32m4(c2, a2[p], b, vl);
            c3                   = __riscv_vfmacc_vf_f32m4(c3, a3[p], b, vl);
        }
        __riscv_vse32_v_f32m4(C + (r + 0) * ldc, c0, vl);
        __riscv_vse32_v_f32m4(C + (r + 1) * ldc, c1, vl);
        __riscv_vse32_v_f32m4(C + (r + 2) * ldc, c2, vl);
        __riscv_vse32_v_f32m4(C + (r + 3) * ldc, c3, vl);
    }
    for (; r < m; ++r) {
        vfloat32m4_t  c = C0 ? __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(C0 + r * ldc0, vl), c0s, vl) :
                               __riscv_vfmv_v_f_f32m4(0.0f, vl);
        const float * a = A + r * lda;
        for (int64_t p = 0; p < k; ++p) {
            c = __riscv_vfmacc_vf_f32m4(c, a[p], __riscv_vle32_v_f32m4(B + p * ldb, vl), vl);
        }
        __riscv_vse32_v_f32m4(C + r * ldc, c, vl);
    }
}

// dst[j * ld_dst + i] = src[i * ld_src + j] for an n x n block (n <= vlmax(e32m4))
void gdn_transpose(int64_t n_rows, int64_t n_cols, const float * src, int64_t ld_src, float * dst, int64_t ld_dst) {
    const size_t vl = __riscv_vsetvl_e32m4(n_cols);
    for (int64_t i = 0; i < n_rows; ++i) {
        __riscv_vsse32_v_f32m4(dst + i, ld_dst * sizeof(float), __riscv_vle32_v_f32m4(src + i * ld_src, vl), vl);
    }
}

struct gdn_chunk_args {
    const ggml_tensor *src_q, *src_k, *src_v, *src_g, *src_beta;
    int64_t            S_v, H, n_tokens;
    float              scale;
};

// One (head, seq): s_in / s_out are the transposed states (S_v x S_v), attn_data the first token's output.
// flagos: buf is the tile's workspace slice of gdn_chunk_floats(S_v), not a thread_local std::vector. s_in is read
// once, into buf, before anything is written to s_out, so the two may be the same row (the state pass)
void gdn_chunked_head(const gdn_chunk_args & a,
                      int64_t                iq1,
                      int64_t                ik1,
                      int64_t                iv1,
                      int64_t                iq3,
                      int64_t                ik3,
                      int64_t                iv3,
                      const float *          s_in,
                      float *                s_out,
                      float *                attn_data,
                      float *                buf) {
    const int64_t D  = a.S_v;
    const int64_t CS = GDN_CHUNK_SIZE;

    // State, seven token tiles, three chunk matrices, and four per-token vectors.
    // All accesses use their logical lengths; no extra tail padding is needed.
    float * S   = buf;
    float * Qc  = S + D * D;
    float * Kc  = Qc + CS * D;
    float * U   = Kc + CS * D;   // v rows, then U in place
    float * KT  = U + CS * D;    // D x CS
    float * KTs = KT + CS * D;   // D x CS, columns scaled by exp(gam_C - gam_s)
    float * P   = KTs + CS * D;  // K S0, then reused
    float * QS  = P + CS * D;    // Q S0 (rows scaled by Gam)
    float * KK  = QS + CS * D;   // CS x CS
    float * QK  = KK + CS * CS;
    float * E   = QK + CS * CS;
    float * gam = E + CS * CS;
    float * Gam = gam + CS;
    float * bet = Gam + CS;
    float * wst = bet + CS;

    const size_t vl = __riscv_vsetvl_e32m4(D);

    gdn_transpose(D, D, s_in, D, S, D);  // S[i][j] = s_in[j][i]

    GGML_TENSOR_LOCALS(size_t, nbq, a.src_q, nb)
    GGML_TENSOR_LOCALS(size_t, nbk, a.src_k, nb)
    GGML_TENSOR_LOCALS(size_t, nbv, a.src_v, nb)
    GGML_TENSOR_LOCALS(size_t, nbg, a.src_g, nb)
    GGML_TENSOR_LOCALS(size_t, nbb, a.src_beta, nb)

    for (int64_t t0 = 0; t0 < a.n_tokens; t0 += CS) {
        const int64_t Cn = std::min(CS, a.n_tokens - t0);

        float acc_g = 0.0f;
        for (int64_t r = 0; r < Cn; ++r) {
            const int64_t t = t0 + r;
            std::memcpy(Qc + r * D, (const char *) a.src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1,
                        D * sizeof(float));
            std::memcpy(Kc + r * D, (const char *) a.src_k->data + ik3 * nbk3 + t * nbk2 + ik1 * nbk1,
                        D * sizeof(float));
            std::memcpy(U + r * D, (const char *) a.src_v->data + iv3 * nbv3 + t * nbv2 + iv1 * nbv1,
                        D * sizeof(float));
            acc_g += *(const float *) ((const char *) a.src_g->data + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);
            gam[r] = acc_g;
            Gam[r] = expf(acc_g);
            bet[r] = *(const float *) ((const char *) a.src_beta->data + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
        }
        for (int64_t r = 0; r < Cn; ++r) {
            for (int64_t c = 0; c <= r; ++c) {
                E[r * CS + c] = expf(gam[r] - gam[c]);
            }
            wst[r] = expf(gam[Cn - 1] - gam[r]);
        }

        gdn_transpose(Cn, D, Kc, D, KT, CS);                           // KT[i][r] = K[r][i]
        gdn_gemm(Cn, Cn, D, Kc, D, KT, CS, KK, CS, nullptr, 0, 0.0f);  // K K^T
        gdn_gemm(Cn, Cn, D, Qc, D, KT, CS, QK, CS, nullptr, 0, 0.0f);  // Q K^T
        gdn_gemm(Cn, D, D, Kc, D, S, D, P, D, nullptr, 0, 0.0f);       // K S0
        gdn_gemm(Cn, D, D, Qc, D, S, D, QS, D, nullptr, 0, 0.0f);      // Q S0

        // U_r = beta_r (v_r - Gam_r P_r) - sum_{s<r} beta_r E_rs KK_rs U_s
        for (int64_t r = 0; r < Cn; ++r) {
            vfloat32m4_t u = __riscv_vle32_v_f32m4(U + r * D, vl);
            u              = __riscv_vfnmsac_vf_f32m4(u, Gam[r], __riscv_vle32_v_f32m4(P + r * D, vl), vl);
            u              = __riscv_vfmul_vf_f32m4(u, bet[r], vl);
            for (int64_t c = 0; c < r; ++c) {
                const float l = bet[r] * E[r * CS + c] * KK[r * CS + c];
                u             = __riscv_vfnmsac_vf_f32m4(u, l, __riscv_vle32_v_f32m4(U + c * D, vl), vl);
            }
            __riscv_vse32_v_f32m4(U + r * D, u, vl);
        }

        // O = scale (Gam Q S0 + A U), A_rs = E_rs QK_rs for s <= r (zero above the diagonal)
        for (int64_t r = 0; r < Cn; ++r) {
            float * qkr = QK + r * CS;
            for (int64_t c = 0; c <= r; ++c) {
                qkr[c] *= E[r * CS + c];
            }
            for (int64_t c = r + 1; c < Cn; ++c) {
                qkr[c] = 0.0f;
            }
            float * qs = QS + r * D;
            __riscv_vse32_v_f32m4(qs, __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(qs, vl), Gam[r], vl), vl);
        }
        gdn_gemm(Cn, D, Cn, QK, CS, U, D, P, D, QS, D, 1.0f);
        for (int64_t r = 0; r < Cn; ++r) {
            __riscv_vse32_v_f32m4(attn_data + (t0 + r) * D * a.H,
                                  __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(P + r * D, vl), a.scale, vl), vl);
        }

        // S = Gam_C S + KTs U
        for (int64_t i = 0; i < D; ++i) {
            for (int64_t c = 0; c < Cn; ++c) {
                KTs[i * CS + c] = KT[i * CS + c] * wst[c];
            }
        }
        gdn_gemm(D, D, Cn, KTs, CS, U, D, S, D, S, D, Gam[Cn - 1]);
    }

    gdn_transpose(D, D, S, D, s_out, D);
}

// SpacemiT's gdn_recurrent_transposed, copied. Transposed state makes each vector lane an independent output row. Four
// partial sums shorten the dot-product dependency chains without reducing across lanes. The four-way unroll requires
// d % 4 == 0, checked by the dispatcher.
__attribute__((noinline)) void gdn_recurrent_transposed(float *       state,
                                                        const float * k,
                                                        const float * q,
                                                        float         decay_scalar,
                                                        const float * v,
                                                        float         beta,
                                                        float         kq,
                                                        float         scale,
                                                        float *       out,
                                                        int64_t       d) {
    for (int64_t j = 0; j < d;) {
        const size_t vl = __riscv_vsetvl_e32m2(d - j);
        auto         k0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        auto         k1 = k0, k2 = k0, k3 = k0, q0 = k0, q1 = k0, q2 = k0, q3 = k0;
        for (int64_t i = 0; i < d; i += 4) {
            auto s0 = __riscv_vle32_v_f32m2(state + (i + 0) * d + j, vl);
            auto s1 = __riscv_vle32_v_f32m2(state + (i + 1) * d + j, vl);
            auto s2 = __riscv_vle32_v_f32m2(state + (i + 2) * d + j, vl);
            auto s3 = __riscv_vle32_v_f32m2(state + (i + 3) * d + j, vl);
            s0      = __riscv_vfmul_vf_f32m2(s0, decay_scalar, vl);
            s1      = __riscv_vfmul_vf_f32m2(s1, decay_scalar, vl);
            s2      = __riscv_vfmul_vf_f32m2(s2, decay_scalar, vl);
            s3      = __riscv_vfmul_vf_f32m2(s3, decay_scalar, vl);
            k0      = __riscv_vfadd_vv_f32m2(k0, __riscv_vfmul_vf_f32m2(s0, k[i + 0], vl), vl);
            k1      = __riscv_vfadd_vv_f32m2(k1, __riscv_vfmul_vf_f32m2(s1, k[i + 1], vl), vl);
            k2      = __riscv_vfadd_vv_f32m2(k2, __riscv_vfmul_vf_f32m2(s2, k[i + 2], vl), vl);
            k3      = __riscv_vfadd_vv_f32m2(k3, __riscv_vfmul_vf_f32m2(s3, k[i + 3], vl), vl);
            q0      = __riscv_vfadd_vv_f32m2(q0, __riscv_vfmul_vf_f32m2(s0, q[i + 0], vl), vl);
            q1      = __riscv_vfadd_vv_f32m2(q1, __riscv_vfmul_vf_f32m2(s1, q[i + 1], vl), vl);
            q2      = __riscv_vfadd_vv_f32m2(q2, __riscv_vfmul_vf_f32m2(s2, q[i + 2], vl), vl);
            q3      = __riscv_vfadd_vv_f32m2(q3, __riscv_vfmul_vf_f32m2(s3, q[i + 3], vl), vl);
        }
        const auto sk =
            __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(k0, k1, vl), __riscv_vfadd_vv_f32m2(k2, k3, vl), vl);
        const auto sq =
            __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(q0, q1, vl), __riscv_vfadd_vv_f32m2(q2, q3, vl), vl);
        auto delta = __riscv_vfsub_vv_f32m2(__riscv_vle32_v_f32m2(v + j, vl), sk, vl);
        delta      = __riscv_vfmul_vf_f32m2(delta, beta, vl);
        __riscv_vse32_v_f32m2(out + j, __riscv_vfmul_vf_f32m2(__riscv_vfmacc_vf_f32m2(sq, kq, delta, vl), scale, vl),
                              vl);
        for (int64_t i = 0; i < d; i += 4) {
            auto s0 = __riscv_vle32_v_f32m2(state + (i + 0) * d + j, vl);
            auto s1 = __riscv_vle32_v_f32m2(state + (i + 1) * d + j, vl);
            auto s2 = __riscv_vle32_v_f32m2(state + (i + 2) * d + j, vl);
            auto s3 = __riscv_vle32_v_f32m2(state + (i + 3) * d + j, vl);
            s0      = __riscv_vfmul_vf_f32m2(s0, decay_scalar, vl);
            s1      = __riscv_vfmul_vf_f32m2(s1, decay_scalar, vl);
            s2      = __riscv_vfmul_vf_f32m2(s2, decay_scalar, vl);
            s3      = __riscv_vfmul_vf_f32m2(s3, decay_scalar, vl);
            s0      = __riscv_vfmacc_vf_f32m2(s0, k[i + 0], delta, vl);
            s1      = __riscv_vfmacc_vf_f32m2(s1, k[i + 1], delta, vl);
            s2      = __riscv_vfmacc_vf_f32m2(s2, k[i + 2], delta, vl);
            s3      = __riscv_vfmacc_vf_f32m2(s3, k[i + 3], delta, vl);
            __riscv_vse32_v_f32m2(state + (i + 0) * d + j, s0, vl);
            __riscv_vse32_v_f32m2(state + (i + 1) * d + j, s1, vl);
            __riscv_vse32_v_f32m2(state + (i + 2) * d + j, s2, vl);
            __riscv_vse32_v_f32m2(state + (i + 3) * d + j, s3, vl);
        }
        j += vl;
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

// SpacemiT's forward_gated_delta_net: up to 4 tokens with a row in one register group the fused one-pass row update
// (the first token reads the input state and writes the output, no copy), else the two-pass form; for K 1, a scalar
// gate and 16 tokens and up its chunkwise form or its recurrence on a transposed state in TCM.
// flagos: the row range, the scratch and the TCM are the tile's (its kda_decay, delta, decay and state_q vectors are
// the workspace slice's three S_v floats, its working state for K > 1 the slice's S_v * S_v, the chunkwise form's
// buffer the slice's gdn_chunk_floats(S_v)); the asserts are the contract's; as the fork, auto (the default
// FLAGOS_SPACEMIT_GDN_PREFILL, prefill) takes the TCM form where it can and else the chunkwise one; chunk and tcm take
// only their form, recurrence neither, and a call outside a form's conditions, tcm without TCM too, takes the
// recurrence. The state is read from state_in_base and written to state_out_base, which with K 1 may be the same row
// (the state pass, core/flagos-spacemit-ops.h spacemit_state_pass): every form reads a row before it writes it
void gdn_rvv(ggml_tensor * dst,
             const float * state_in_base,
             float *       state_out_base,
             float *       scratch,
             float *       state_work,
             void *        tcm,
             size_t        tcm_size,
             int           prefill,
             int64_t       ir0,
             int64_t       ir1) {
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
    const int64_t state_size_per_snap = S_v * S_v * H * n_seqs;
    float *       attn_out_base       = (float *) dst->data;

    const float scale = 1.0f / sqrtf((float) S_v);
    // the scalar gate uses g[0]; KDA gives one gate per state column (S_v == 1: the two coincide, KDA kept)
    const bool  kda   = src_g->ne[0] == S_v;

    // Amortize the TCM transpose over prefill tokens; short tails use the row kernels.
    // flagos: the tile's TCM for the fork's ctx.shared, GGML_PAD for its align_up
    const uintptr_t shared_begin = reinterpret_cast<uintptr_t>(tcm);
    const size_t    tcm_offset   = GGML_PAD(shared_begin, GDN_TCM_ALIGNMENT) - shared_begin;
    const size_t    tcm_bytes    = (S_v * S_v + 3 * S_v) * sizeof(float);
    const bool      transposed   = (prefill == gdn_prefill_tcm || prefill == gdn_prefill_auto) && K == 1 && !kda &&
                            n_tokens >= GDN_CHUNK_SIZE && S_v % 4 == 0 && S_v >= (int64_t) __riscv_vsetvlmax_e32m2() &&
                            S_v <= (int64_t) __riscv_vsetvlmax_e32m4() && tcm != nullptr && tcm_offset <= tcm_size &&
                            tcm_bytes <= tcm_size - tcm_offset;
    float * tcm_state = transposed ? reinterpret_cast<float *>(shared_begin + tcm_offset) : nullptr;
    float * staged    = transposed ? tcm_state + S_v * S_v : nullptr;

    // prefill: chunkwise form (the token recurrence below handles the remaining cases)
    if ((prefill == gdn_prefill_chunk || prefill == gdn_prefill_auto) && !transposed && K == 1 && !kda &&
        n_tokens >= GDN_CHUNK_SIZE && S_v <= (int64_t) __riscv_vsetvlmax_e32m4() && src_q->type == GGML_TYPE_F32 &&
        src_k->type == GGML_TYPE_F32 && src_v->type == GGML_TYPE_F32 && src_g->type == GGML_TYPE_F32 &&
        src_beta->type == GGML_TYPE_F32 &&
        ggml_is_contiguous_rows(src_q) && ggml_is_contiguous_rows(src_k) && ggml_is_contiguous_rows(src_v) &&
        ggml_is_contiguous(src_g) && ggml_is_contiguous(src_beta) && src_beta->ne[0] == 1) {
        g_gdn_chunk.fetch_add(1, std::memory_order_relaxed);
        const int64_t        rq3c = n_seqs / src_q->ne[3];
        const int64_t        rk3c = n_seqs / src_k->ne[3];
        const gdn_chunk_args args = { src_q, src_k, src_v, src_g, src_beta, S_v, H, n_tokens, scale };
        for (int64_t ir = ir0; ir < ir1; ++ir) {
            const int64_t iv1 = ir % H;
            const int64_t iv3 = ir / H;
            gdn_chunked_head(args, iv1 % src_q->ne[1], iv1 % src_k->ne[1], iv1, iv3 / rq3c, iv3 / rk3c, iv3,
                             state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v,
                             state_out_base + (iv3 * H + iv1) * S_v * S_v,
                             attn_out_base + (iv3 * n_tokens * H + iv1) * S_v, scratch);
        }
        return;
    }

    const bool fused     = n_tokens <= 4 && S_v <= (int64_t) __riscv_vsetvlmax_e32m8();
    (transposed ? g_gdn_tcm : fused ? g_gdn_fused : g_gdn_two_pass).fetch_add(1, std::memory_order_relaxed);
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

        float *       s_out = transposed ? tcm_state :
                              K > 1      ? state_work :
                                           state_out_base + (iv3 * H + iv1) * S_v * S_v;
        const float * s_in  = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        if (transposed) {
            gdn_transpose(S_v, S_v, s_in, S_v, s_out, S_v);  // into TCM, read whole before the state is written back
        } else if (!fused && s_out != s_in) {  // the state pass gives the same row for both: work in place
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

            if (transposed) {
                std::memcpy(staged, k, S_v * sizeof(float));
                std::memcpy(staged + S_v, q, S_v * sizeof(float));
                std::memcpy(staged + 2 * S_v, v, S_v * sizeof(float));
                k = staged;
                q = staged + S_v;
                v = staged + 2 * S_v;
            }
            const float kq = rvv_gdn_dot_f32(k, q, S_v);

            if (transposed) {
                gdn_recurrent_transposed(s_out, k, q, expf(g[0]), v, beta, kq, scale, attn_data, S_v);
            } else if (fused) {
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
        if (transposed) {
            gdn_transpose(S_v, S_v, s_out, S_v, state_out_base + (iv3 * H + iv1) * S_v * S_v, S_v);
        }
    }
}

#endif  // GGML_FLAGOS_SPACEMIT_RVV

// one tile's rows of node, the state read from state_in_base and the new state written to state_out_base; name is the
// step's, for the workspace message
bool gdn_tile_run(const spacemit_tile & tile, ggml_tensor * node, const float * state_in_base, float * state_out_base,
                  const char * name) {
    const gdn_shape sh       = gdn_shape_of(node);
    const int64_t   per_tile = gdn_tile_floats(sh);
    const size_t    needed   = (size_t) per_tile * tile.nth * sizeof(float);
    if (tile.nth == 0 || tile.nth > SPACEMIT_MAX_TILES || tile.workspace == nullptr ||
        tile.workspace_size < needed) {
        GGML_LOG_ERROR("FlagOS SpacemiT: %s needs %zu scratch bytes for %u tiles, has %zu\n", name, needed, tile.nth,
                       tile.workspace_size);
        return false;
    }

    float * scratch    = (float *) tile.workspace + per_tile * tile.ith;
    float * state_work = sh.K > 1 ? scratch + 3 * sh.S_v : nullptr;
    // read on every build, so that an unknown value warns where only the reference runs too
    const int prefill  = gdn_prefill_from_env();

    int64_t ir0, ir1;
    tile_range(sh.H * sh.n_seqs, tile, ir0, ir1);
    if (ir0 >= ir1) {
        return true;
    }
#if defined(GGML_FLAGOS_SPACEMIT_RVV)
    if (!spacemit_use_reference()) {
        gdn_rvv(node, state_in_base, state_out_base, scratch, state_work, tile.tcm, tile.tcm_size, prefill, ir0, ir1);
        return true;
    }
#endif
    GGML_UNUSED(prefill);
    g_gdn_reference.fetch_add(1, std::memory_order_relaxed);
    gdn_ref(node, state_in_base, state_out_base, scratch, state_work, ir0, ir1);
    return true;
}

}  // namespace

size_t spacemit_gdn_workspace(const ggml_tensor * node) {
    return (size_t) gdn_tile_floats(gdn_shape_of(node)) * SPACEMIT_MAX_TILES * sizeof(float);
}

void spacemit_gdn_read_switch() {
    (void) gdn_prefill_from_env();
}

void spacemit_gdn_paths_take(uint64_t * reference, uint64_t * fused, uint64_t * two_pass, uint64_t * chunk,
                             uint64_t * tcm) {
    *reference = g_gdn_reference.exchange(0);
    *fused     = g_gdn_fused.exchange(0);
    *two_pass  = g_gdn_two_pass.exchange(0);
    *chunk     = g_gdn_chunk.exchange(0);
    *tcm       = g_gdn_tcm.exchange(0);
}

bool spacemit_kernel_gated_delta_net(const spacemit_tile * tile_p, ggml_tensor * dst) {
    const gdn_shape sh = gdn_shape_of(dst);
    return gdn_tile_run(*tile_p, dst, (const float *) dst->src[5]->data,
                        (float *) dst->data + sh.S_v * sh.H * sh.n_tokens * sh.n_seqs, "gdn.f32");
}

// the state pass's step runs on the CPY (core/flagos-spacemit-ops.h spacemit_state_pass): its source views the op's
// new state, the op's own state comes from the GET_ROWS' row of the persistent tensor, and the CPY's destination is
// the row to write. False, with the step named, when the row index or the rows are not as the pass promised
static bool gdn_direct_rows(ggml_tensor * cpy, const char * name, ggml_tensor ** node, const float ** state_in,
                            float ** state_out) {
    *node                    = cpy->src[0]->view_src;
    const ggml_tensor * st   = (*node)->src[5];
    const ggml_tensor * rows = st->view_src != nullptr ? st->view_src : st;
    const ggml_tensor * src  = rows->src[0];
    const int64_t       idx  = ((const int32_t *) rows->src[1]->data)[0];
    if (idx < 0 || idx >= src->ne[1]) {
        GGML_LOG_ERROR("FlagOS SpacemiT: %s: state row %lld of %lld\n", name, (long long) idx,
                       (long long) src->ne[1]);
        return false;
    }
    *state_in          = (const float *) ((const char *) src->data + (size_t) idx * src->nb[1]);
    *state_out         = (float *) cpy->data;
    const size_t bytes = ggml_nbytes(cpy);
    const char * in    = (const char *) *state_in;
    const char * out   = (const char *) *state_out;
    if (in != out && in < out + bytes && out < in + bytes) {  // whole rows: the same row or disjoint ones
        GGML_LOG_ERROR("FlagOS SpacemiT: %s: the state rows overlap in part\n", name);
        return false;
    }
    return true;
}

bool spacemit_kernel_gated_delta_net_direct(const spacemit_tile * tile_p, ggml_tensor * cpy) {
    const char * const name      = "gated_delta_net.f32.direct";
    ggml_tensor *      node      = nullptr;
    const float *      state_in  = nullptr;
    float *            state_out = nullptr;
    if (!gdn_direct_rows(cpy, name, &node, &state_in, &state_out)) {
        return false;
    }
    return gdn_tile_run(*tile_p, node, state_in, state_out, name);
}

bool spacemit_kernel_gated_delta_net_direct_cleared(const spacemit_tile * tile_p, ggml_tensor * cpy) {
    const char * const name      = "gated_delta_net.f32.direct.cleared";
    ggml_tensor *      node      = nullptr;
    const float *      state_in  = nullptr;
    float *            state_out = nullptr;
    if (!gdn_direct_rows(cpy, name, &node, &state_in, &state_out)) {
        return false;
    }
    // the clearing SCALE of a sequence's first graph folded in (spacemit_state_pass): this tile multiplies its own
    // heads' part of the row the CPY writes by 0, element by element as ggml_vec_scale_f32, so that a NaN, an
    // infinity or a negative value in the old state gives the CPU's bits
    const gdn_shape sh = gdn_shape_of(node);
    int64_t         ir0, ir1;
    tile_range(sh.H * sh.n_seqs, *tile_p, ir0, ir1);
    float * const y = state_out + ir0 * sh.S_v * sh.S_v;
    for (int64_t i = 0; i < (ir1 - ir0) * sh.S_v * sh.S_v; i++) {
        y[i] = y[i] * 0.0f;
    }
    return gdn_tile_run(*tile_p, node, state_in, state_out, name);
}
