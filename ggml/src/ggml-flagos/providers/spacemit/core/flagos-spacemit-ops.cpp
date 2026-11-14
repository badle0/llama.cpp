#include "flagos-spacemit-ops.h"

#include "flagos-spacemit-diagnostics.h"  // spacemit_op_disabled (GGML_FLAGOS_DISABLE_OPS)

#include "../../../../ggml-impl.h"

#include <cstdint>
#include <cstring>

// The core's contract table (plan.md "S1 step 2 spec"): per contract the ggml op, the structural check (operand types,
// shapes, modes and placement, the layout rule through the plugin's tensor_layout) and the step names in order. Every
// name is bound once, at load, to the plugin's kernel entry (spacemit_ops_bind), so this file calls no kernel code and
// includes no kernel header; the kernels' own shape facts are their entries' `supports`.

// an operand is usable when it is not allocated yet (placement time), or lives in our buffer or in host memory
static bool spacemit_buffer_usable(const ggml_tensor * t, ggml_backend_buffer_type_t own_buft) {
    const ggml_backend_buffer_t buffer = t->view_src != nullptr ? t->view_src->buffer : t->buffer;
    if (buffer == nullptr) {
        return true;
    }
    const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buffer);
    return buft == own_buft || ggml_backend_buft_is_host(buft);
}

// F32 with contiguous elements in each row; rows may have any stride
static bool spacemit_f32_rows(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float);
}

// F32 or F16 with contiguous elements in each row; rows may have any stride
static bool spacemit_f32_f16_rows(const ggml_tensor * t) {
    return (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) && t->nb[0] == ggml_type_size(t->type);
}

// ADD, MUL: src1 repeats over src0 in every dimension, as in ggml-cpu's binary-ops.cpp
static bool spacemit_binary_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                          ggml_backend_buffer_type_t) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    return spacemit_f32_rows(op) && spacemit_f32_rows(a) && b->type == GGML_TYPE_F32 && ggml_can_repeat(b, a) &&
           ggml_are_same_shape(a, op);
}

// RMS_NORM: any eps >= 0 (ggml-cpu asserts the same)
static bool spacemit_rms_norm_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                            ggml_backend_buffer_type_t) {
    float eps;
    std::memcpy(&eps, op->op_params, sizeof(float));
    return spacemit_f32_rows(op) && spacemit_f32_rows(op->src[0]) && ggml_are_same_shape(op->src[0], op) && eps >= 0.0f;
}

// ROPE (forward): F32 or F16 rows into rows of the same type (F16: the context shift's ROPE in place on the KV cache,
// src/llama-kv-cache.cpp:1878-1883); modes NORMAL and NEOX; I32 positions, one per token; optional F32 frequency
// factors; the channels past n_dims are copied in pairs, so ne0 is even. IMROPE (Qwen3.5, ggml_rope_multi): four
// positions per token, sections >= 0 with one of the first three > 0 and their sum at most ne0 (ggml-cpu's asserts)
static bool spacemit_rope_supported(const ggml_tensor * op, const spacemit_plugin_v1 *, ggml_backend_buffer_type_t) {
    const ggml_tensor * src0   = op->src[0];
    const ggml_tensor * pos    = op->src[1];
    const ggml_tensor * ff     = op->src[2];
    const int           n_dims = ggml_get_op_params_i32(op, 1);
    const int           mode   = ggml_get_op_params_i32(op, 2);
    const bool          imrope = mode == GGML_ROPE_TYPE_IMROPE;
    bool                sections_ok = !imrope;
    if (imrope) {
        const int32_t * s = (const int32_t *) op->op_params + 11;
        sections_ok       = s[0] >= 0 && s[1] >= 0 && s[2] >= 0 && s[3] >= 0 && (s[0] > 0 || s[1] > 0 || s[2] > 0) &&
                      (int64_t) s[0] + s[1] + s[2] + s[3] <= src0->ne[0];
    }
    return spacemit_f32_f16_rows(op) && spacemit_f32_f16_rows(src0) && op->type == src0->type &&
           ggml_are_same_shape(src0, op) &&
           (mode == GGML_ROPE_TYPE_NORMAL || mode == GGML_ROPE_TYPE_NEOX || imrope) && sections_ok && n_dims > 0 &&
           n_dims % 2 == 0 && n_dims <= src0->ne[0] && src0->ne[0] % 2 == 0 && pos->type == GGML_TYPE_I32 &&
           ggml_is_contiguous(pos) && pos->ne[0] >= (imrope ? 4 : 1) * src0->ne[2] &&
           (ff == nullptr || (ff->type == GGML_TYPE_F32 && ggml_is_contiguous(ff) && ff->ne[0] >= n_dims / 2));
}

// the KV cache's types besides F32 and F16 (-ctk, -ctv) that the provider writes from F32 rows, with ggml's reference
// quantizers (from_float_ref): whole blocks, ggml's layout
static bool spacemit_cache_block_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

// SET_ROWS (the KV cache writes, src/llama-kv-cache.cpp:1333, :1368): F32 or F16 rows into F32 or F16 rows, or F32 rows
// into rows of a block type above in whole blocks (an F16 source there stays unclaimed); I64 or I32 indices; never
// into a tensor in an IME layout (op is a view of the destination; the layout rule resolves its view_src)
static bool spacemit_set_rows_supported(const ggml_tensor * op, const spacemit_plugin_v1 * table,
                                        ggml_backend_buffer_type_t) {
    const ggml_tensor * src = op->src[0];
    const ggml_tensor * idx = op->src[1];
    if (op->ne[0] != src->ne[0] || (idx->type != GGML_TYPE_I64 && idx->type != GGML_TYPE_I32) ||
        table->tensor_layout(op) != 0) {
        return false;
    }
    if (spacemit_f32_f16_rows(op)) {
        return spacemit_f32_f16_rows(src);
    }
    return spacemit_cache_block_type(op->type) && spacemit_f32_rows(src) && op->nb[0] == ggml_type_size(op->type) &&
           op->ne[0] % ggml_blck_size(op->type) == 0;
}

// CPY: F32 rows into a contiguous tensor of a block type above, of the same shape, in whole blocks (the context shift's
// write-back into a quantized KV cache, src/llama-kv-cache.cpp:1863-1877); op is ggml_cpy's view of the destination,
// with its ne and nb (ggml.c:3526-3535), never in an IME layout. No other source types, no non-contiguous
// destinations; F32 into F32 is cpy.f32 below
static bool spacemit_cpy_quant_supported(const ggml_tensor * op, const spacemit_plugin_v1 * table,
                                         ggml_backend_buffer_type_t) {
    const ggml_tensor * src = op->src[0];
    return spacemit_cache_block_type(op->type) && spacemit_f32_rows(src) && ggml_are_same_shape(src, op) &&
           ggml_is_contiguous(op) && op->ne[0] % ggml_blck_size(op->type) == 0 && table->tensor_layout(op) == 0;
}

// CPY of F32 into F32, the writes into the recurrent state (src/llama-graph.cpp build_rs, src/models/delta-net-base.cpp),
// never into an IME layout. Source rows contiguous with any row strides, and either a contiguous destination of the same
// element count (the shapes may differ, as ggml allows) or a destination of the same shape with contiguous rows and any
// row strides (the rollback snapshots, delta-net-base.cpp:591-603, with n_rs_seq > 0). No other same-type copy
static bool spacemit_cpy_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 * table,
                                       ggml_backend_buffer_type_t) {
    const ggml_tensor * src = op->src[0];
    if (op->type != GGML_TYPE_F32 || !spacemit_f32_rows(src) || table->tensor_layout(op) != 0) {
        return false;
    }
    return (ggml_is_contiguous(op) && ggml_nelements(src) == ggml_nelements(op)) ||
           (spacemit_f32_rows(op) && ggml_are_same_shape(src, op));
}

// SCALE of F32 rows into F32 rows of the same shape, in place or not, bias 0 only (the clear of a recurrent state
// slot, src/llama-graph.cpp build_rs); a scale with a bias stays on the CPU
static bool spacemit_scale_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                         ggml_backend_buffer_type_t) {
    float bias;
    std::memcpy(&bias, (const float *) op->op_params + 1, sizeof(float));
    return spacemit_f32_rows(op) && spacemit_f32_rows(op->src[0]) && ggml_are_same_shape(op->src[0], op) &&
           bias == 0.0f;
}

// the largest state side (S_v) gdn.f32 claims: Qwen3.5 uses 128 (plan.md "Qwen3.5 probe"). The kernel's scratch is
// S_v * S_v floats per tile when K > 1, taken from the shared workspace, so a far larger state would ask for a far
// larger workspace; such a node stays on the CPU
static constexpr int64_t k_gdn_max_state = 256;

// GATED_DELTA_NET (the fused Gated DeltaNet step, src/models/delta-net-base.cpp:402 and :567): every operand and the
// output F32 and shaped exactly as ggml_gated_delta_net (ggml/src/ggml-cpu/ops.cpp's kernel reads them so):
//   v [S_v, H, n_tokens, n_seqs] with contiguous rows, the state [S_v, S_v, H, n_seqs] contiguous;
//   the gate g contiguous, scalar (g->ne[0] == 1) or one value per channel (KDA, g->ne[0] == S_v), and beta
//   contiguous with beta->ne[0] == 1, both [*, H, n_tokens, n_seqs];
//   q and k rows of S_v with contiguous rows, one row per (head, token, sequence), repeating over the heads and the
//   sequences as the kernel reads them (iq1 = iv1 % neq1, iq3 = iv3 / (nev3 / neq3)): their head and sequence counts
//   divide H and n_seqs, and they carry every token;
//   K (the snapshot slots, op_params 0) >= 1 and the output contiguous of ggml_gated_delta_net's shape,
//   [S_v * H, n_tokens * n_seqs + K * S_v * n_seqs].
// An F32 tensor never has an IME layout (the layout rule is by type, shape and name), so no layout check here.
// Anything else is refused and runs on the CPU
static bool spacemit_gdn_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                       ggml_backend_buffer_type_t) {
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * g = op->src[3];
    const ggml_tensor * b = op->src[4];
    const ggml_tensor * s = op->src[5];
    if (q == nullptr || k == nullptr || v == nullptr || g == nullptr || b == nullptr || s == nullptr) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 ||
        v->type != GGML_TYPE_F32 || g->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 ||
        s->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(b) || !ggml_is_contiguous(s) || !ggml_is_contiguous(op)) {
        return false;
    }
    const int64_t S_v      = v->ne[0];
    const int64_t H        = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs   = v->ne[3];
    const int64_t K        = ggml_get_op_params_i32(op, 0);
    if (S_v < 1 || S_v > k_gdn_max_state || H < 1 || n_tokens < 1 || n_seqs < 1 || K < 1) {
        return false;
    }
    if ((g->ne[0] != 1 && g->ne[0] != S_v) || g->ne[1] != H || g->ne[2] != n_tokens || g->ne[3] != n_seqs) {
        return false;
    }
    if (b->ne[0] != 1 || b->ne[1] != H || b->ne[2] != n_tokens || b->ne[3] != n_seqs) {
        return false;
    }
    if (s->ne[0] != S_v || s->ne[1] != S_v || s->ne[2] != H || s->ne[3] != n_seqs) {
        return false;
    }
    for (const ggml_tensor * t : { q, k }) {
        if (t->ne[0] != S_v || t->ne[1] < 1 || t->ne[3] < 1 || t->ne[2] != n_tokens || H % t->ne[1] != 0 ||
            n_seqs % t->ne[3] != 0) {
            return false;
        }
    }
    return op->ne[0] == S_v * H && op->ne[1] == n_tokens * n_seqs + K * S_v * n_seqs && op->ne[2] == 1 &&
           op->ne[3] == 1;
}

// SILU, SIGMOID and SOFTPLUS of F32 rows into F32 rows of the same shape (Qwen3.5's Gated DeltaNet gates and its
// convolution's activation, src/models/qwen35.cpp:253, :366, :374, :398): rows contiguous with any row strides, as
// ggml-cpu's apply_unary_op addresses them. The plain SILU here is not the SWIGLU already claimed (GGML_OP_GLU)
static bool spacemit_unary_f32_rows(const ggml_tensor * op) {
    return spacemit_f32_rows(op) && spacemit_f32_rows(op->src[0]) && ggml_are_same_shape(op->src[0], op);
}

static bool spacemit_silu_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                        ggml_backend_buffer_type_t) {
    return ggml_get_unary_op(op) == GGML_UNARY_OP_SILU && spacemit_unary_f32_rows(op);
}

static bool spacemit_sigmoid_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                           ggml_backend_buffer_type_t) {
    return ggml_get_unary_op(op) == GGML_UNARY_OP_SIGMOID && spacemit_unary_f32_rows(op);
}

static bool spacemit_softplus_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                            ggml_backend_buffer_type_t) {
    return ggml_get_unary_op(op) == GGML_UNARY_OP_SOFTPLUS && spacemit_unary_f32_rows(op);
}

// L2_NORM of F32 rows into F32 rows of the same shape (the convolved queries and keys, src/models/qwen35.cpp:432):
// rows contiguous with any row strides (the model's sources are views of the convolution's output), the op's eps from
// op_params, which ggml-cpu asserts to be >= 0
static bool spacemit_l2_norm_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                           ggml_backend_buffer_type_t) {
    float eps;
    std::memcpy(&eps, op->op_params, sizeof(float));
    return spacemit_unary_f32_rows(op) && eps >= 0.0f;
}

// SSM_CONV of F32 (src/models/qwen35.cpp:395): the shapes ggml_ssm_conv builds and ggml-cpu asserts - a 3-D input
// whose first two dimensions are contiguous, a 2-D contiguous kernel with as many rows as the input's, and a
// contiguous 3-D output [d_inner, n_t, n_s]. The kernel is read in ggml's layout, so it must have no IME layout
static bool spacemit_ssm_conv_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 * table,
                                            ggml_backend_buffer_type_t) {
    const ggml_tensor * sx = op->src[0];
    const ggml_tensor * c  = op->src[1];
    return op->type == GGML_TYPE_F32 && sx->type == GGML_TYPE_F32 && c->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(op) && table->tensor_layout(c) == 0 && table->tensor_layout(op) == 0 &&
           sx->nb[0] == sizeof(float) && sx->nb[1] == (size_t) sx->ne[0] * sizeof(float) && sx->ne[3] == 1 &&
           ggml_is_contiguous(c) && c->ne[2] == 1 && c->ne[3] == 1 && c->ne[1] == sx->ne[1] &&
           op->ne[0] == sx->ne[1] && op->ne[1] == sx->ne[0] - c->ne[0] + 1 && op->ne[2] == sx->ne[2] &&
           op->ne[3] == 1;
}

// CONCAT along dimension 0 of two F32 sources into a contiguous F32 destination (the convolution's input: the state's
// columns and the transposed projection, src/models/delta-net-base.cpp:470, so the second source is a transposed
// view). Every output element is one source element, read with ggml-cpu's strides, so any source strides work
static bool spacemit_concat_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 * table,
                                          ggml_backend_buffer_type_t) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    return ggml_get_op_params_i32(op, 0) == 0 && op->type == GGML_TYPE_F32 && a->type == GGML_TYPE_F32 &&
           b->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && table->tensor_layout(op) == 0 &&
           op->ne[0] == a->ne[0] + b->ne[0] && op->ne[1] == a->ne[1] && op->ne[2] == a->ne[2] &&
           op->ne[3] == a->ne[3] && a->ne[1] == b->ne[1] && a->ne[2] == b->ne[2] && a->ne[3] == b->ne[3];
}

// CONT of F32: a contiguous copy of a source whose rows are contiguous with any row strides (the attention layers'
// gate, src/models/qwen35.cpp:297). GGML_OP_CONT has src[0] only and the op is its own destination, so cpy.f32's
// kernel does it: the source's rows in ggml's order into the packed destination
static bool spacemit_cont_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 * table,
                                        ggml_backend_buffer_type_t) {
    const ggml_tensor * src = op->src[0];
    return op->type == GGML_TYPE_F32 && spacemit_f32_rows(src) && ggml_is_contiguous(op) &&
           ggml_nelements(src) == ggml_nelements(op) && table->tensor_layout(op) == 0;
}

// MUL_MAT of an F32 2-D weight by F32 activations (ssm_alpha and ssm_beta, src/models/qwen35.cpp:362, :369): these
// weights have no IME layout, so they are stored as the file holds them and read in ggml's layout, in our buffer or in
// a host buffer. The weight is a tensor of its own, never a view, so a view of an F32 KV cache stays refused; the
// activations and the output are contiguous, and the weight broadcasts over the activations' higher dimensions
static bool spacemit_mul_mat_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 * table,
                                           ggml_backend_buffer_type_t) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    return op->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 &&
           w->view_src == nullptr && table->tensor_layout(w) == 0 && ggml_is_contiguous(w) && w->ne[2] == 1 &&
           w->ne[3] == 1 && ggml_is_contiguous(x) && ggml_is_contiguous(op) && x->ne[0] == w->ne[0] &&
           op->ne[0] == w->ne[1] && op->ne[1] == x->ne[1] && op->ne[2] == x->ne[2] && op->ne[3] == x->ne[3];
}

// rows of elt-byte elements, evenly spaced over every dimension: ggml-cpu's ARGSORT and CLAMP address row i at
// i * nb[1], which is the row's place only when nb[2] and nb[3] follow from nb[1]
static bool spacemit_flat_rows(const ggml_tensor * t, size_t elt) {
    return t->nb[0] == elt && t->nb[2] == t->nb[1] * (size_t) t->ne[1] && t->nb[3] == t->nb[2] * (size_t) t->ne[2];
}

// The MoE router of Qwen3.6-35B-A3B (build_moe_ffn, src/llama-graph.cpp:1915; plan.md "Qwen3.6-35B-A3B header
// probe"): softmax over the experts, top-k by ARGSORT, the chosen weights summed, clamped and divided by their sum.
// SOFT_MAX of F32 without a mask, sinks or ALiBi (src/llama-graph.cpp:1965): a contiguous source (ggml_soft_max_impl
// asserts it) into a contiguous F32 destination of the same shape, in place or not; any scale
static bool spacemit_soft_max_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                            ggml_backend_buffer_type_t) {
    const ggml_tensor * src = op->src[0];
    float               max_bias;
    std::memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
    return op->type == GGML_TYPE_F32 && src->type == GGML_TYPE_F32 && op->src[1] == nullptr &&
           op->src[2] == nullptr && max_bias == 0.0f && ggml_is_contiguous(src) && ggml_is_contiguous(op) &&
           ggml_are_same_shape(src, op);
}

// ARGSORT of F32 rows into I32 rows of the same shape, ascending or descending (the expert top-k,
// ggml_argsort_top_k, src/llama-graph.cpp:2031; its view of the first k columns is not an op). Flat rows on both sides,
// as ggml-cpu addresses them
static bool spacemit_argsort_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                           ggml_backend_buffer_type_t) {
    const ggml_tensor * src   = op->src[0];
    const int32_t       order = ggml_get_op_params_i32(op, 0);
    return op->type == GGML_TYPE_I32 && src->type == GGML_TYPE_F32 && ggml_are_same_shape(src, op) &&
           spacemit_flat_rows(src, sizeof(float)) && spacemit_flat_rows(op, sizeof(int32_t)) &&
           (order == GGML_SORT_ORDER_ASC || order == GGML_SORT_ORDER_DESC) && src->ne[0] <= INT32_MAX;
}

// SUM_ROWS of F32 rows into F32 [1, ne1, ne2, ne3] (the chosen weights' sum, src/llama-graph.cpp:2059): rows
// contiguous with any row strides on both sides, as ggml-cpu reads them
static bool spacemit_sum_rows_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                            ggml_backend_buffer_type_t) {
    const ggml_tensor * src = op->src[0];
    return spacemit_f32_rows(op) && spacemit_f32_rows(src) && op->ne[0] == 1 && op->ne[1] == src->ne[1] &&
           op->ne[2] == src->ne[2] && op->ne[3] == src->ne[3];
}

// CLAMP of F32 into F32 of the same shape, in place (ggml_clamp's result is a view of its source,
// src/llama-graph.cpp:2063) or not; flat rows on both sides, as ggml-cpu addresses them
static bool spacemit_clamp_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                         ggml_backend_buffer_type_t) {
    const ggml_tensor * src = op->src[0];
    return op->type == GGML_TYPE_F32 && src->type == GGML_TYPE_F32 && ggml_are_same_shape(src, op) &&
           spacemit_flat_rows(src, sizeof(float)) && spacemit_flat_rows(op, sizeof(float));
}

// DIV: the forms ADD and MUL take (the weights by their clamped sum, [n_used, n_tokens] / [1, n_tokens],
// src/llama-graph.cpp:2066)
static bool spacemit_div_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 * table,
                                       ggml_backend_buffer_type_t own_buft) {
    return spacemit_binary_f32_supported(op, table, own_buft);
}

// GET_ROWS: F32 rows by I32 indices
static bool spacemit_get_rows_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                            ggml_backend_buffer_type_t) {
    return spacemit_f32_rows(op) && spacemit_f32_rows(op->src[0]) && op->src[1]->type == GGML_TYPE_I32 &&
           op->ne[0] == op->src[0]->ne[0];
}

// the block types the token embedding's GET_ROWS takes (ggml-cpu's get_rows_q, ggml's to_float per row)
static bool spacemit_get_rows_block_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

// GET_ROWS of quantized rows by I32 indices into F32 rows (the token embedding, src/llama-graph.cpp build_inp_embd):
// contiguous rows of whole blocks, shaped as ggml-cpu's get_rows_q asserts, stored as ggml's blocks. A host buffer
// holds them so; the layout rule answers for our buffer (and for a weight not allocated yet), where an IME layout is
// refused. FLAGOS_SPACEMIT_EMBD=cpu refuses every such lookup (spacemit_embd_on_cpu)
static bool spacemit_get_rows_quant_supported(const ggml_tensor * op, const spacemit_plugin_v1 * table,
                                              ggml_backend_buffer_type_t) {
    if (spacemit_embd_on_cpu()) {
        return false;
    }
    const ggml_tensor *         src    = op->src[0];
    const ggml_tensor *         idx    = op->src[1];
    const ggml_backend_buffer_t buffer = src->view_src != nullptr ? src->view_src->buffer : src->buffer;
    const bool plain = (buffer != nullptr && ggml_backend_buffer_is_host(buffer)) || table->tensor_layout(src) == 0;
    return spacemit_f32_rows(op) && idx->type == GGML_TYPE_I32 && spacemit_get_rows_block_type(src->type) &&
           ggml_is_contiguous(src) && src->nb[0] == ggml_type_size(src->type) &&
           src->ne[0] % ggml_blck_size(src->type) == 0 && op->ne[0] == src->ne[0] && src->ne[2] == idx->ne[1] &&
           ggml_nrows(op) == ggml_nelements(idx) && plain;
}

// GLU: SwiGLU only, split (two tensors) and single-tensor forms; rows addressed as in ggml-cpu (ggml_is_contiguous_1),
// dst shaped as ggml-cpu asserts (the kernels write nrows x nc elements of it)
static bool spacemit_swiglu_f32_supported(const ggml_tensor * op, const spacemit_plugin_v1 *,
                                          ggml_backend_buffer_type_t) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const bool          dst_shape =
        src1 != nullptr ? ggml_are_same_shape(src0, op) :
                          op->ne[0] == src0->ne[0] / 2 && ggml_nrows(op) == ggml_nrows(src0);
    return ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU && op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
           ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(op) && dst_shape &&
           (src1 == nullptr ||
            (src1->type == GGML_TYPE_F32 && ggml_is_contiguous_1(src1) && ggml_are_same_shape(src0, src1)));
}

// a weight in an IME layout (Q4_0 32x256; Q8_0, Q4_1, Q6_K and Q5_0 in q8_0 32x32), read in that layout: so it must
// sit in our buffer (or be unallocated at placement time), never in a host buffer. The layout comes from the weight's
// type, shape and name, the plugin's rule through the table (tensor_layout, which answers for a view's storage, so a
// view is excluded here): llama.cpp's cache tensors never have one, so a MUL_MAT never reads a quantized KV cache as a
// weight. The answer must not depend on the row count: llama.cpp asks once per weight, at load, with 512 rows
// (plan.md M2b design). A 3-D weight may have a layout (MUL_MAT_ID's experts) but is never a MUL_MAT weight here
static bool spacemit_mul_mat_ime_supported(const ggml_tensor * op, const spacemit_plugin_v1 * table,
                                           ggml_backend_buffer_type_t own_buft) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (w->view_src != nullptr || w->ne[2] != 1 || w->ne[3] != 1 || table->tensor_layout(w) == 0) {
        return false;
    }
    if (w->buffer != nullptr && ggml_backend_buffer_get_type(w->buffer) != own_buft) {
        return false;
    }
    return x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_is_contiguous(op);
}

// MUL_MAT_ID (the experts' matmuls, src/llama-graph.cpp build_moe_ffn): the experts' weight [K, N, n_expert] in an IME
// layout, so in our buffer, never a view (the layout rule gives a 3-D weight one when each expert's N rows are whole
// 32-row groups, and never to a 3-D Q4_1 weight); F32 activations [K, ne11, n_tokens] (ne11 1, broadcast, or
// n_used) and the F32 output contiguous; I32 ids [n_used, n_tokens] with any strides (build_moe_ffn's are a view of
// ARGSORT's output, ggml_argsort_top_k). ggml_mul_mat_id asserts the shapes. As for MUL_MAT the answer does not depend
// on the token count: llama.cpp asks once per weight, at load, with 512 tokens (src/llama-model-loader.cpp:938-946)
static bool spacemit_mul_mat_id_supported(const ggml_tensor * op, const spacemit_plugin_v1 * table,
                                          ggml_backend_buffer_type_t own_buft) {
    const ggml_tensor * w   = op->src[0];
    const ggml_tensor * x   = op->src[1];
    const ggml_tensor * ids = op->src[2];
    if (ids == nullptr || w->view_src != nullptr || w->ne[3] != 1 || table->tensor_layout(w) == 0) {
        return false;
    }
    if (w->buffer != nullptr && ggml_backend_buffer_get_type(w->buffer) != own_buft) {
        return false;
    }
    return x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 &&
           ggml_is_contiguous(x) && ggml_is_contiguous(op);
}

// the attention contracts' type and shape facts all sit in the kernels' supports today; moving their type part into
// the core is the Q4-vocabulary step (plan.md "S1 design" §1)
static bool spacemit_always(const ggml_tensor *, const spacemit_plugin_v1 *, ggml_backend_buffer_type_t) {
    return true;
}

// one contract: the ggml op it claims, its structural check, and its steps' names in order. Several contracts may
// claim one op (attention's three variants); spacemit_find_op takes the first whose checks pass
struct spacemit_contract {
    ggml_op op;
    bool (*check)(const ggml_tensor * op, const spacemit_plugin_v1 * table, ggml_backend_buffer_type_t own_buft);
    const char * steps[SPACEMIT_MAX_OP_STEPS];
    uint32_t     n_steps;
};

// M2e.3 round 2 (adopted 2026-10-14; the paired split is the default): one-token attention split over the tiles,
// partial results then their merge. The four FLASH_ATTN_EXT contracts are tried in this order and the kernels'
// supports pick one (spacemit_flash_attn_split, which reads the shape and FLAGOS_SPACEMIT_ATTN); first the KV-head split
// for a few tokens, 2 to 8, turned off by FLAGOS_SPACEMIT_ATTN_KVH=off (plan.md "The KV-head split for a
// few tokens, from SpacemiT's fork")
static const spacemit_contract k_contracts[] = {
    { GGML_OP_ADD,             spacemit_binary_f32_supported,   { "add.f32" },                                      1 },
    { GGML_OP_MUL,             spacemit_binary_f32_supported,   { "mul.f32" },                                      1 },
    { GGML_OP_RMS_NORM,        spacemit_rms_norm_f32_supported, { "rms_norm.f32" },                                  1 },
    { GGML_OP_ROPE,            spacemit_rope_supported,         { "rope" },                                          1 },
    { GGML_OP_SET_ROWS,        spacemit_set_rows_supported,     { "set_rows" },                                      1 },
    { GGML_OP_CPY,             spacemit_cpy_quant_supported,    { "cpy.quant" },                                     1 },
    { GGML_OP_CPY,             spacemit_cpy_f32_supported,      { "cpy.f32" },                                       1 },
    { GGML_OP_SCALE,           spacemit_scale_f32_supported,    { "scale.f32" },                                     1 },
    { GGML_OP_GET_ROWS,        spacemit_get_rows_f32_supported, { "get_rows.f32" },                                  1 },
    { GGML_OP_GET_ROWS,        spacemit_get_rows_quant_supported, { "get_rows.quant" },                              1 },
    { GGML_OP_GLU,             spacemit_swiglu_f32_supported,   { "swiglu.f32" },                                    1 },
    { GGML_OP_MUL_MAT,         spacemit_mul_mat_ime_supported,  { "mul_mat.quantize", "mul_mat.gemm" },              2 },
    { GGML_OP_MUL_MAT_ID,      spacemit_mul_mat_id_supported,   { "mul_mat_id.quantize", "mul_mat_id.gemm" },        2 },
    { GGML_OP_FLASH_ATTN_EXT,  spacemit_always, { "flash_attn.kvh.partial", "flash_attn.kvh.merge" },                2 },
    { GGML_OP_FLASH_ATTN_EXT,  spacemit_always,
      { "flash_attn.split.partial_paired", "flash_attn.split.merge" },                                               2 },
    { GGML_OP_FLASH_ATTN_EXT,  spacemit_always, { "flash_attn.split.partial", "flash_attn.split.merge" },            2 },
    { GGML_OP_FLASH_ATTN_EXT,  spacemit_always, { "flash_attn" },                                                    1 },
    { GGML_OP_GATED_DELTA_NET, spacemit_gdn_f32_supported,      { "gdn.f32" },                                       1 },
    // Qwen3.5's Gated DeltaNet layers, the ops beside the state update (plan.md "Step 2 spec", pieces D to G)
    { GGML_OP_UNARY,           spacemit_silu_f32_supported,     { "silu.f32" },                                      1 },
    { GGML_OP_UNARY,           spacemit_sigmoid_f32_supported,  { "sigmoid.f32" },                                   1 },
    { GGML_OP_UNARY,           spacemit_softplus_f32_supported, { "softplus.f32" },                                  1 },
    { GGML_OP_L2_NORM,         spacemit_l2_norm_f32_supported,  { "l2_norm.f32" },                                   1 },
    { GGML_OP_SSM_CONV,        spacemit_ssm_conv_f32_supported, { "ssm_conv.f32" },                                  1 },
    { GGML_OP_CONCAT,          spacemit_concat_f32_supported,   { "concat.f32" },                                    1 },
    { GGML_OP_CONT,            spacemit_cont_f32_supported,     { "cpy.f32" },                                       1 },
    { GGML_OP_MUL_MAT,         spacemit_mul_mat_f32_supported,  { "mul_mat.f32" },                                   1 },
    // the MoE router's ops (Qwen3.6-35B-A3B, build_moe_ffn)
    { GGML_OP_SOFT_MAX,        spacemit_soft_max_f32_supported, { "soft_max.f32" },                                  1 },
    { GGML_OP_ARGSORT,         spacemit_argsort_f32_supported,  { "argsort.f32" },                                   1 },
    { GGML_OP_SUM_ROWS,        spacemit_sum_rows_f32_supported, { "sum_rows.f32" },                                  1 },
    { GGML_OP_CLAMP,           spacemit_clamp_f32_supported,    { "clamp.f32" },                                     1 },
    { GGML_OP_DIV,             spacemit_div_f32_supported,      { "div.f32" },                                       1 },
};

static constexpr size_t k_n_contracts = sizeof(k_contracts) / sizeof(k_contracts[0]);

static_assert(k_n_contracts <= sizeof(spacemit_ops::ops) / sizeof(spacemit_ops::ops[0]),
              "spacemit_ops::ops must hold every contract");

// the contract names of the kernels the plugin offers, in its kernel table's order (plan.md "S1 step 1 spec" section 3)
const char * const spacemit_required_kernels[] = {
    "add.f32",          "mul.f32",          "rms_norm.f32",             "rope",
    "set_rows",         "cpy.quant",        "get_rows.f32",             "get_rows.quant",
    "swiglu.f32",       "mul_mat.quantize", "mul_mat.gemm",
    "mul_mat_id.quantize",                  "mul_mat_id.gemm",          "flash_attn",
    "flash_attn.split.partial",             "flash_attn.split.partial_paired",
    "flash_attn.split.merge",               "flash_attn.kvh.partial",   "flash_attn.kvh.merge",
    "scale.f32",        "cpy.f32",          "gdn.f32",
    "silu.f32",         "sigmoid.f32",      "softplus.f32",             "l2_norm.f32",
    "ssm_conv.f32",     "concat.f32",       "mul_mat.f32",
    "soft_max.f32",     "argsort.f32",      "sum_rows.f32",             "clamp.f32",
    "div.f32",
};
const size_t spacemit_n_required_kernels = sizeof(spacemit_required_kernels) / sizeof(spacemit_required_kernels[0]);

static bool spacemit_is_required_kernel(const char * name) {
    for (size_t i = 0; i < spacemit_n_required_kernels; i++) {
        if (std::strcmp(name, spacemit_required_kernels[i]) == 0) {
            return true;
        }
    }
    return false;
}

bool spacemit_ops_bind(const spacemit_plugin_v1 * table, spacemit_ops * out, std::string * detail) {
    const auto fail = [detail](const std::string & text) {
        if (detail != nullptr) {
            *detail = text;
        }
        return false;
    };
    if (table == nullptr || out == nullptr) {
        return fail("no plugin table");
    }
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    if (kernels == nullptr) {
        return fail("the plugin offers no kernels");
    }
    out->table = nullptr;
    out->n_ops = 0;
    for (size_t i = 0; i < k_n_contracts; i++) {
        const spacemit_contract & contract = k_contracts[i];
        spacemit_op &             op       = out->ops[i];
        op.n_steps                         = contract.n_steps;
        for (uint32_t s = 0; s < contract.n_steps; s++) {
            const char * name = contract.steps[s];
            if (!spacemit_is_required_kernel(name)) {
                return fail(std::string(name) + " is not a required kernel");
            }
            const spacemit_plugin_kernel * entry = nullptr;
            for (uint32_t k = 0; k < n_kernels && entry == nullptr; k++) {
                if (kernels[k].name != nullptr && std::strcmp(kernels[k].name, name) == 0) {
                    entry = &kernels[k];
                }
            }
            if (entry == nullptr) {
                return fail(std::string("no kernel ") + name);
            }
            op.steps[s] = entry;
        }
    }
    out->table = table;
    out->n_ops = (uint32_t) k_n_contracts;
    return true;
}

const spacemit_op * spacemit_find_op(const spacemit_ops & ops, const ggml_tensor * op,
                                     ggml_backend_buffer_type_t own_buft) {
    // GGML_FLAGOS_DISABLE_OPS names ops that go back to the CPU: refused here, so supports_op and graph_compute agree
    if (spacemit_op_disabled(op->op)) {
        return nullptr;
    }
    if (ops.table == nullptr || !spacemit_buffer_usable(op, own_buft)) {
        return nullptr;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] != nullptr && !spacemit_buffer_usable(op->src[i], own_buft)) {
            return nullptr;
        }
    }
    for (uint32_t i = 0; i < ops.n_ops; i++) {
        if (k_contracts[i].op != op->op || !k_contracts[i].check(op, ops.table, own_buft)) {
            continue;
        }
        const spacemit_op & impl    = ops.ops[i];
        bool                claimed = true;
        for (uint32_t s = 0; s < impl.n_steps && claimed; s++) {
            claimed = impl.steps[s]->supports == nullptr || impl.steps[s]->supports(op);
        }
        if (claimed) {
            return &impl;
        }
    }
    return nullptr;
}

size_t spacemit_op_workspace(const spacemit_op * op, const ggml_tensor * node) {
    if (op == nullptr) {
        return 0;
    }
    size_t bytes = 0;
    for (uint32_t s = 0; s < op->n_steps; s++) {
        size_t (*const rule)(const ggml_tensor *) = op->steps[s]->workspace_size;
        if (rule == nullptr) {
            continue;
        }
        bool asked = false;  // two steps may share one rule (both MUL_MAT steps, both split steps): ask it once
        for (uint32_t t = 0; t < s; t++) {
            asked = asked || op->steps[t]->workspace_size == rule;
        }
        if (!asked) {
            const size_t size = rule(node);
            bytes             = size > bytes ? size : bytes;
        }
    }
    return bytes;
}
