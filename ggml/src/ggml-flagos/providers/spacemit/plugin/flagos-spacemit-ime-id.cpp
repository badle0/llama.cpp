#include "flagos-spacemit-ime-id.h"

#include "flagos-spacemit-ime-format.h"
#include "flagos-spacemit-ime-kernels.h"

#include "../../../../ggml-impl.h"

#include <algorithm>

// MUL_MAT_ID (plan.md "Step 2 spec", Qwen3.6-35B-A3B): ggml-cpu's ggml_compute_forward_mul_mat_id (ggml-cpu.c) on the
// IME formats of MUL_MAT. Correctness first: the 1-row kernels, the weights read from DRAM. SpacemiT's fork
// (ggml-spacemit ime.cpp forward_mul_mat_id) groups an expert's rows by 4 or 8 and stages its weights through TCM;
// that is a later speed step. Its row map has a stride of n_tokens per expert, which overflows when an id repeats
// within a token; the counting sort below has room for every pair.

namespace {

constexpr int64_t nb_cols = spacemit_ime::row_tile;  // weight rows per IME tile, the unit of the column split

// one pair of ggml-cpu's mmid_row_mapping: the slot of the token's ids (i1: the output row, and the activation row
// i1 % ne11) and the token (i2)
struct mmid_row {
    int32_t i1;
    int32_t i2;
};

struct mmid_dims {
    const mm_format * f;            // the weight's layout
    int64_t           k;            // row length
    int64_t           n;            // rows per expert = output columns
    int64_t           n_as;         // experts
    int64_t           n_ids;        // ids per token (n_used)
    int64_t           ne11;         // activation rows per token: 1 (broadcast) or n_ids
    int64_t           ne12;         // tokens
    int64_t           k_blocks;
    int64_t           a_row_bytes;  // quantized activation row
    int64_t           b_row_bytes;  // repacked weight row
    size_t            act_bytes;    // the quantized activations, padded to a cache line
    size_t            start_bytes;  // a grouping's expert starts, n_as + 1 of them, padded
    size_t            group_bytes;  // one tile's grouping: the starts, then n_ids * ne12 pairs, padded
};

mmid_dims mmid_dims_of(const ggml_tensor * node) {
    const ggml_tensor * w   = node->src[0];
    const ggml_tensor * x   = node->src[1];
    const ggml_tensor * ids = node->src[2];
    mmid_dims           d;
    d.f           = &mm_format_of(w);
    d.k           = w->ne[0];
    d.n           = w->ne[1];
    d.n_as        = w->ne[2];
    d.n_ids       = ids->ne[0];
    d.ne11        = x->ne[1];
    d.ne12        = x->ne[2];
    d.k_blocks    = d.k / d.f->k_block;
    d.a_row_bytes = (int64_t) d.f->act_row_bytes((size_t) d.k);
    d.b_row_bytes = (int64_t) d.f->weight_row_bytes((size_t) d.k);
    d.act_bytes   = GGML_PAD((size_t) (d.ne11 * d.ne12 * d.a_row_bytes), 64);
    d.start_bytes = GGML_PAD((size_t) (d.n_as + 1) * sizeof(int64_t), 64);
    d.group_bytes = d.start_bytes + GGML_PAD((size_t) (d.n_ids * d.ne12) * sizeof(mmid_row), 64);
    return d;
}

int64_t div_up(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// ggml-cpu's grouping: every (token, slot) pair, the tokens in order and the slots in order within a token, appended
// to its expert's list. Here one list ordered by expert, a stable counting sort: expert e's pairs are rows[start[e]]
// to rows[start[e + 1] - 1], in ggml-cpu's order. It reads only the ids, so every tile that builds it holds the same
// lists. false when an id is outside [0, n_as), which ggml-cpu asserts (in debug builds only)
bool group_rows(const ggml_tensor * ids, const mmid_dims & d, int64_t * start, mmid_row * rows) {
    const auto id_at = [&](int64_t t, int64_t s) {
        return *(const int32_t *) ((const char *) ids->data + t * ids->nb[1] + s * ids->nb[0]);
    };
    std::fill(start, start + d.n_as + 1, int64_t(0));
    for (int64_t t = 0; t < d.ne12; t++) {
        for (int64_t s = 0; s < d.n_ids; s++) {
            const int32_t e = id_at(t, s);
            if (e < 0 || e >= d.n_as) {
                return false;
            }
            start[e + 1]++;
        }
    }
    for (int64_t e = 0; e < d.n_as; e++) {
        start[e + 1] += start[e];
    }
    for (int64_t t = 0; t < d.ne12; t++) {
        for (int64_t s = 0; s < d.n_ids; s++) {
            rows[start[id_at(t, s)]++] = { (int32_t) s, (int32_t) t };
        }
    }
    // each start[e] has moved to its list's end, the next expert's start
    for (int64_t e = d.n_as; e > 0; e--) {
        start[e] = start[e - 1];
    }
    start[0] = 0;
    return true;
}

}  // namespace

size_t spacemit_mul_mat_id_workspace(const ggml_tensor * node) {
    const mmid_dims d = mmid_dims_of(node);
    return d.act_bytes + SPACEMIT_MAX_TILES * d.group_bytes;
}

// step 1: every activation row (ne11 per token) quantized once, in the 1-row form, into the workspace's first part,
// row i11 + i12 * ne11. One row: the tiles split its K blocks, as mul_mat.quantize does; several: the rows
bool spacemit_mul_mat_id_quantize(const spacemit_tile * tile_p, ggml_tensor * node) {
    const spacemit_tile & tile      = *tile_p;
    const mmid_dims       d         = mmid_dims_of(node);
    const bool            reference = mm_use_reference();
    const ggml_tensor *   x         = node->src[1];
    uint8_t *             qa        = static_cast<uint8_t *>(tile.workspace);
    if (tile.workspace == nullptr || tile.workspace_size < d.act_bytes) {
        GGML_LOG_ERROR("FlagOS SpacemiT: mul_mat_id.quantize needs %zu workspace bytes, has %zu\n", d.act_bytes,
                       tile.workspace_size);
        return false;
    }
    const auto row_of = [&](int64_t r) {
        return (const float *) ((const char *) x->data + (r % d.ne11) * x->nb[1] + (r / d.ne11) * x->nb[2]);
    };

    const int64_t n_rows = d.ne11 * d.ne12;
    if (n_rows == 1) {
        const int64_t blk_bytes = (int64_t) d.f->act_row_bytes((size_t) d.f->k_block);
        const int64_t per_tile  = div_up(d.k_blocks, tile.nth);
        const int64_t b0        = tile.ith * per_tile;
        const int64_t b1        = std::min(b0 + per_tile, d.k_blocks);
        if (b0 < b1) {
            d.f->quantize_row(row_of(0) + b0 * d.f->k_block, (size_t) ((b1 - b0) * d.f->k_block), qa + b0 * blk_bytes,
                              reference);
        }
        return true;
    }
    const int64_t per_tile = div_up(n_rows, tile.nth);
    const int64_t r_end    = std::min((tile.ith + 1) * per_tile, n_rows);
    for (int64_t r = tile.ith * per_tile; r < r_end; r++) {
        d.f->quantize_row(row_of(r), (size_t) d.k, qa + r * d.a_row_bytes, reference);
    }
    return true;
}

// step 2: every tile groups the pairs by expert (group_rows) in its own part of the workspace, then takes its share
// of the units (pair j, 32 columns c), numbered j * (N / 32) + c in the grouping's order and dealt as contiguous ranges,
// so the split depends only on the ids and the tile count. Each run of one pair's columns is one 1-row kernel call on
// the expert's rows; every output element is one call's dot product, so the split cannot change it
bool spacemit_mul_mat_id_gemm(const spacemit_tile * tile_p, ggml_tensor * node) {
    const spacemit_tile & tile      = *tile_p;
    const mmid_dims       d         = mmid_dims_of(node);
    const bool            reference = mm_use_reference();
    const size_t          needed    = d.act_bytes + (size_t) tile.nth * d.group_bytes;
    if (tile.nth == 0 || tile.nth > SPACEMIT_MAX_TILES || tile.workspace == nullptr || tile.workspace_size < needed) {
        GGML_LOG_ERROR("FlagOS SpacemiT: mul_mat_id.gemm needs %zu workspace bytes for %u tiles, has %zu\n", needed,
                       tile.nth, tile.workspace_size);
        return false;
    }
    uint8_t * const       ws    = static_cast<uint8_t *>(tile.workspace);
    const uint8_t * const qa    = ws;
    uint8_t * const       group = ws + d.act_bytes + (size_t) tile.ith * d.group_bytes;
    int64_t * const       start = reinterpret_cast<int64_t *>(group);
    mmid_row * const      rows  = reinterpret_cast<mmid_row *>(group + d.start_bytes);
    if (!group_rows(node->src[2], d, start, rows)) {
        if (tile.ith == 0) {
            GGML_LOG_ERROR("FlagOS SpacemiT: MUL_MAT_ID (%s) has an id outside [0, %lld)\n", node->name,
                           (long long) d.n_as);
        }
        return false;
    }

    const uint8_t * w       = static_cast<const uint8_t *>(node->src[0]->data);
    const int64_t   n_pairs = d.n_ids * d.ne12;
    const int64_t   per     = d.n / nb_cols;  // units per pair
    const int64_t   units   = n_pairs * per;
    const int64_t   u_end   = units * (tile.ith + 1) / tile.nth;
    int64_t         e       = 0;  // the expert of pair j: pairs are in expert order, so it only moves forward
    for (int64_t u = units * tile.ith / tile.nth; u < u_end;) {
        const int64_t j  = u / per;
        const int64_t c0 = u % per;
        const int64_t c1 = std::min(per, c0 + (u_end - u));
        while (start[e + 1] <= j) {
            e++;
        }
        const mmid_row &    r   = rows[j];
        const uint8_t *     a   = qa + ((r.i1 % d.ne11) + (int64_t) r.i2 * d.ne11) * d.a_row_bytes;
        const uint8_t *     b   = w + (e * d.n + c0 * nb_cols) * d.b_row_bytes;
        float *             out = (float *) ((char *) node->data + r.i1 * node->nb[1] + r.i2 * node->nb[2]) +
                      c0 * nb_cols;
        d.f->gemm(a, b, out, 1, (size_t) ((c1 - c0) * nb_cols), (size_t) d.k_blocks, (size_t) d.n, reference);
        u += c1 - c0;
    }
    return true;
}
