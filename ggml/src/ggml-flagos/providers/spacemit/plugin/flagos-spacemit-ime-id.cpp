#include "flagos-spacemit-ime-id.h"

#include "flagos-spacemit-ime-format.h"
#include "flagos-spacemit-ime-kernels.h"
#include "flagos-spacemit-kernels.h"

#include "../../../../ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <new>

// MUL_MAT_ID (plan.md "Step 2 spec", Qwen3.6-35B-A3B): ggml-cpu's ggml_compute_forward_mul_mat_id (ggml-cpu.c) on the
// IME formats of MUL_MAT. One token keeps the 1-row path: one kernel call per (token, slot) pair, the weights read
// from DRAM. Several tokens group the pairs by expert, as SpacemiT's fork does (ggml-spacemit ime.cpp
// forward_mul_mat_id), so that an expert's weights are read once per node: a pair's activation row sits at its place in
// the sorted list, an expert's pairs go through the 4-row kernel in groups of 4 (the rest one by one) and a 32-column
// weight slab is staged through TCM for all of them. Its row map has a stride of n_tokens per expert, which overflows
// when an id repeats within a token; the counting sort below has room for every pair.

namespace {

constexpr int64_t nb_cols   = spacemit_ime::row_tile;  // weight rows per IME tile, the unit of the column split
constexpr int64_t row_align = 4;                       // rows of the 4-row kernel
// what a unit (expert, slab) costs beside its expert's rows: the slab's bytes, in units of one row's work. A guess, not
// measured: a slab is 32 weight rows, a row of the unit's work is one quantized activation row times those 32 rows
constexpr int64_t slab_weight = 4;

// one pair of ggml-cpu's mmid_row_mapping: the slot of the token's ids (i1: the output row, and the activation row
// i1 % ne11) and the token (i2)
struct mmid_row {
    int32_t i1;
    int32_t i2;
};

struct mmid_dims {
    const mm_format * f;             // the weight's layout
    int64_t           k;             // row length
    int64_t           n;             // rows per expert = output columns
    int64_t           n_as;          // experts
    int64_t           n_ids;         // ids per token (n_used)
    int64_t           ne11;          // activation rows per token: 1 (broadcast) or n_ids
    int64_t           ne12;          // tokens
    int64_t           k_blocks;
    int64_t           a_row_bytes;   // quantized activation row
    int64_t           b_row_bytes;   // repacked weight row
    size_t            act_bytes;     // the quantized activations, padded to a cache line
    size_t            start_bytes;   // a grouping's expert starts, n_as + 1 of them, padded
    size_t            group_bytes;   // one tile's grouping: the starts, then n_ids * ne12 pairs, padded
    size_t            gather_bytes;  // one tile's gathered 4 float rows, padded
    size_t            tile_bytes;    // one tile's part: its grouping, its gathered rows and its 4 x 32 result
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
    // a row per activation row (one token, or rows) or a row per pair, whichever is larger: the size must not depend on
    // the setting
    d.act_bytes    = GGML_PAD((size_t) (std::max(d.ne11, d.n_ids) * d.ne12 * d.a_row_bytes), 64);
    d.start_bytes  = GGML_PAD((size_t) (d.n_as + 1) * sizeof(int64_t), 64);
    d.group_bytes  = d.start_bytes + GGML_PAD((size_t) (d.n_ids * d.ne12) * sizeof(mmid_row), 64);
    d.gather_bytes = GGML_PAD((size_t) (row_align * d.k) * sizeof(float), 64);
    d.tile_bytes   = d.group_bytes + d.gather_bytes + GGML_PAD((size_t) (row_align * nb_cols) * sizeof(float), 64);
    return d;
}

int64_t div_up(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// FLAGOS_SPACEMIT_MMID: staged (unset or empty, the default): several tokens group the pairs by expert and stage each
// 32-column weight slab through TCM, with the expert's quantized rows beside it where they fit; grouped: the same
// grouping and the same kernel calls with the slabs read from DRAM, so the same bytes; rows: one 1-row kernel call per
// (pair, slab) from DRAM, the path before the grouping, for any token count. Parsed once (spacemit_cached_switch; the
// check program resets between its settings); anything else warns once per parse generation and keeps the default.
// Both steps read it, so a change between them (not something llama.cpp does) would leave the activations in a layout
// the multiply does not expect
enum class mmid_setting { staged, grouped, rows };

int parse_mmid_setting() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_MMID");
    if (v == nullptr || *v == '\0' || std::strcmp(v, "staged") == 0) {
        return (int) mmid_setting::staged;
    }
    if (std::strcmp(v, "grouped") == 0) {
        return (int) mmid_setting::grouped;
    }
    if (std::strcmp(v, "rows") == 0) {
        return (int) mmid_setting::rows;
    }
    static std::atomic<uint32_t> warned_in{ 0 };
    const uint32_t               generation = spacemit_switches_generation();
    if (warned_in.exchange(generation) != generation) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_MMID=%s: not staged, grouped or rows, the default (staged) kept\n", __func__,
                      v);
    }
    return (int) mmid_setting::staged;
}

mmid_setting mmid_setting_from_env() {
    static std::atomic<int>      value{ 0 };
    static std::atomic<uint32_t> seen{ 0 };
    return static_cast<mmid_setting>(spacemit_cached_switch(value, seen, parse_mmid_setting));
}

// test-only, for the comparison (plan.md "The expert matmul's claim queue"): FLAGOS_SPACEMIT_MMID_DEAL=queue (unset or
// empty, the default): several tokens' (expert, slab) units are cut into chunks that the tiles claim from one counter;
// static: the units dealt as contiguous ranges of cumulative weight before any work starts. Parsed once
// (spacemit_cached_switch); anything else warns once per parse generation and keeps the default
enum class mmid_deal { queue, fixed };

int parse_mmid_deal() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_MMID_DEAL");
    if (v == nullptr || *v == '\0' || std::strcmp(v, "queue") == 0) {
        return (int) mmid_deal::queue;
    }
    if (std::strcmp(v, "static") == 0) {
        return (int) mmid_deal::fixed;
    }
    static std::atomic<uint32_t> warned_in{ 0 };
    const uint32_t               generation = spacemit_switches_generation();
    if (warned_in.exchange(generation) != generation) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_MMID_DEAL=%s: not queue or static, the default (queue) kept\n", __func__, v);
    }
    return (int) mmid_deal::queue;
}

mmid_deal mmid_deal_from_env() {
    static std::atomic<int>      value{ 0 };
    static std::atomic<uint32_t> seen{ 0 };
    return static_cast<mmid_deal>(spacemit_cached_switch(value, seen, parse_mmid_deal));
}

// test-only: FLAGOS_SPACEMIT_MMID_CHUNKS=<n>, the chunks a tile's share of the queue's weight is cut into, n decimal
// digits only, 1 to 1024; unset or empty: 16. Parsed once (spacemit_cached_switch); anything else keeps the default
// with a warning, once per parse generation
constexpr int mmid_default_chunks = 16;
constexpr int mmid_max_chunks     = 1024;

int parse_mmid_chunks() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_MMID_CHUNKS");
    if (v == nullptr || *v == '\0') {
        return mmid_default_chunks;
    }
    int          n = 0;
    const char * c = v;
    for (; *c >= '0' && *c <= '9' && n <= mmid_max_chunks; ++c) {  // stops past 1024, so no overflow
        n = n * 10 + (*c - '0');
    }
    if (*c == '\0' && n >= 1 && n <= mmid_max_chunks) {
        return n;
    }
    static std::atomic<uint32_t> warned_in{ 0 };
    const uint32_t               generation = spacemit_switches_generation();
    if (warned_in.exchange(generation) != generation) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_MMID_CHUNKS=%s: not a chunk count from 1 to 1024, the default %d kept\n",
                      __func__, v, mmid_default_chunks);
    }
    return mmid_default_chunks;
}

int mmid_chunks_from_env() {
    static std::atomic<int>      value{ mmid_default_chunks };
    static std::atomic<uint32_t> seen{ 0 };
    return spacemit_cached_switch(value, seen, parse_mmid_chunks);
}

// the queue's counter: after the tiles' parts, at the workspace's end, on a cache line of its own when the workspace is
// 8-byte aligned (the core's is 64), else 8-byte aligned. Tile 0 of the first step constructs it at 0
using mmid_counter = std::atomic<int64_t>;
static_assert(mmid_counter::is_always_lock_free, "the claim counter must be lock-free");

void * mmid_counter_at(void * workspace, const mmid_dims & d) {
    const uintptr_t p = reinterpret_cast<uintptr_t>(workspace) + d.act_bytes + SPACEMIT_MAX_TILES * d.tile_bytes;
    uintptr_t       q = GGML_PAD(p, 64);
    if (q - p > 64 - sizeof(mmid_counter)) {
        q = GGML_PAD(p, alignof(mmid_counter));
    }
    return reinterpret_cast<void *>(q);
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

// an expert's r pairs, in list order, cut into groups: 4 while 4 remain, then singles. Group g holds group_size rows
// from pair group_start of the expert's list
int64_t group_count(int64_t r) {
    return r / row_align + r % row_align;
}

int64_t group_start(int64_t g, int64_t r) {
    const int64_t full = r / row_align;
    return g < full ? g * row_align : full * row_align + (g - full);
}

int64_t group_size(int64_t g, int64_t r) {
    return g < r / row_align ? row_align : 1;
}

// one tile's part of the workspace, after the activations
struct mmid_work {
    int64_t *  start;   // the grouping's expert starts
    mmid_row * rows;    // the grouping's pairs
    float *    gather;  // a group's 4 source rows, for quantize_4rows (row stride k)
    float *    result;  // a group's 4 x 32 output slab (ldc 32)
};

mmid_work mmid_work_of(const spacemit_tile & tile, const mmid_dims & d) {
    uint8_t * const p = static_cast<uint8_t *>(tile.workspace) + d.act_bytes + (size_t) tile.ith * d.tile_bytes;
    mmid_work       w;
    w.start  = reinterpret_cast<int64_t *>(p);
    w.rows   = reinterpret_cast<mmid_row *>(p + d.start_bytes);
    w.gather = reinterpret_cast<float *>(p + d.group_bytes);
    w.result = reinterpret_cast<float *>(p + d.group_bytes + d.gather_bytes);
    return w;
}

// one output row of a pair, at column n0
float * out_row(const ggml_tensor * node, const mmid_row & r, int64_t n0) {
    return (float *) ((char *) node->data + r.i1 * node->nb[1] + r.i2 * node->nb[2]) + n0;
}

}  // namespace

// the activations (one row per activation row, or one per pair, whichever is larger), then one grouping, one gathered
// group and one result slab per tile for SPACEMIT_MAX_TILES tiles, each padded to a cache line, then a cache line for
// the queue's counter. It must not depend on the setting: the steps of one node may not all see the same value
size_t spacemit_mul_mat_id_workspace(const ggml_tensor * node) {
    const mmid_dims d = mmid_dims_of(node);
    return d.act_bytes + SPACEMIT_MAX_TILES * d.tile_bytes + 64;
}

void spacemit_mul_mat_id_read_switch() {
    (void) mmid_deal_from_env();
    (void) mmid_chunks_from_env();
}

// step 1: quantize the activations into the workspace's first part. One token (or rows): every activation row (ne11 per
// token) once, in the 1-row form, at row i11 + i12 * ne11; one row: the tiles split its K blocks, as mul_mat.quantize
// does. Several tokens: one row per (token, slot) pair, at the pair's place in the grouping's order, so that an
// expert's rows are consecutive; the 4 rows of a group are stored interleaved for the 4-row kernel, which wants them
// contiguous, so the tile gathers them first. The groups are dealt to the tiles as contiguous ranges
bool spacemit_mul_mat_id_quantize(const spacemit_tile * tile_p, ggml_tensor * node) {
    const spacemit_tile & tile      = *tile_p;
    const mmid_dims       d         = mmid_dims_of(node);
    const bool            reference = mm_use_reference();
    const ggml_tensor *   x         = node->src[1];
    uint8_t *             qa        = static_cast<uint8_t *>(tile.workspace);
    const bool            by_pair   = d.ne12 > 1 && mmid_setting_from_env() != mmid_setting::rows;

    if (by_pair) {
        const size_t needed = spacemit_mul_mat_id_workspace(node);
        if (tile.nth == 0 || tile.nth > SPACEMIT_MAX_TILES || tile.workspace == nullptr ||
            tile.workspace_size < needed) {
            GGML_LOG_ERROR("FlagOS SpacemiT: mul_mat_id.quantize needs %zu workspace bytes for %u tiles, has %zu\n",
                           needed, tile.nth, tile.workspace_size);
            return false;
        }
        // the queue's reset, made visible to the second step's claims by the barrier between the steps
        if (tile.ith == 0) {
            new (mmid_counter_at(tile.workspace, d)) mmid_counter(0);
        }
        mmid_work w = mmid_work_of(tile, d);
        if (!group_rows(node->src[2], d, w.start, w.rows)) {
            if (tile.ith == 0) {
                GGML_LOG_ERROR("FlagOS SpacemiT: MUL_MAT_ID (%s) has an id outside [0, %lld)\n", node->name,
                               (long long) d.n_as);
            }
            return false;
        }
        const auto pair_row = [&](const mmid_row & r) {
            return (const float *) ((const char *) x->data + (r.i1 % d.ne11) * x->nb[1] + r.i2 * x->nb[2]);
        };
        int64_t n_groups = 0;
        for (int64_t e = 0; e < d.n_as; e++) {
            n_groups += group_count(w.start[e + 1] - w.start[e]);
        }
        const int64_t g0 = n_groups * tile.ith / tile.nth;
        const int64_t g1 = n_groups * (tile.ith + 1) / tile.nth;
        int64_t       g  = 0;
        for (int64_t e = 0; e < d.n_as && g < g1; e++) {
            const int64_t r     = w.start[e + 1] - w.start[e];
            const int64_t count = group_count(r);
            if (g + count <= g0) {
                g += count;
                continue;
            }
            for (int64_t i = 0; i < count && g + i < g1; i++) {
                if (g + i < g0) {
                    continue;
                }
                const int64_t j = w.start[e] + group_start(i, r);
                if (group_size(i, r) == row_align) {
                    for (int64_t t = 0; t < row_align; t++) {
                        std::memcpy(w.gather + t * d.k, pair_row(w.rows[j + t]), (size_t) d.k * sizeof(float));
                    }
                    d.f->quantize_4rows(w.gather, (size_t) d.k, qa + j * d.a_row_bytes, reference);
                } else {
                    d.f->quantize_row(pair_row(w.rows[j]), (size_t) d.k, qa + j * d.a_row_bytes, reference);
                }
            }
            g += count;
        }
        return true;
    }

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

// step 2: multiply. Every tile groups the pairs by expert (group_rows) in its own part of the workspace, from the ids
// alone, so all tiles hold the same lists.
// One token (or rows): the units are (pair j, 32 columns c), numbered j * (N / 32) + c in the grouping's order and
// dealt as contiguous ranges; each run of one pair's columns is one 1-row kernel call on the expert's rows.
// Several tokens: the units are (active expert e, 32-column slab s), expert-major, each weighted by the expert's rows
// plus the slab (slab_weight). By default (FLAGOS_SPACEMIT_MMID_DEAL=queue) an expert's slabs are cut into chunks of
// about a 16th of a tile's share of the weight, which the tiles claim in order from the counter that tile 0 of step 1
// reset; static deals the units as contiguous ranges of cumulative weight. A slab belongs to one tile either way, so an
// expert's weights are read once per node. For a unit the tile multiplies the slab by every group of the expert: a
// group of 4 into a 4 x 32 temporary whose rows are then scattered to their output rows, a single straight to its
// output row. With TCM (staged) the slab is copied there once per unit and the expert's quantized rows once per run of
// the expert's units where they fit beside it; a slab is copied whatever the group count, since staging a slab is also
// faster for one kernel call (plan.md "1-row matmul bandwidth").
// Either way every output element comes from one kernel call on one (group or pair, slab), so neither the tile count
// nor the staging can change it
bool spacemit_mul_mat_id_gemm(const spacemit_tile * tile_p, ggml_tensor * node) {
    const spacemit_tile & tile      = *tile_p;
    const mmid_dims       d         = mmid_dims_of(node);
    const bool            reference = mm_use_reference();
    const mmid_setting    setting   = mmid_setting_from_env();
    const size_t          needed    = spacemit_mul_mat_id_workspace(node);
    if (tile.nth == 0 || tile.nth > SPACEMIT_MAX_TILES || tile.workspace == nullptr || tile.workspace_size < needed) {
        GGML_LOG_ERROR("FlagOS SpacemiT: mul_mat_id.gemm needs %zu workspace bytes for %u tiles, has %zu\n", needed,
                       tile.nth, tile.workspace_size);
        return false;
    }
    const uint8_t * const qa = static_cast<const uint8_t *>(tile.workspace);
    mmid_work             w  = mmid_work_of(tile, d);
    if (!group_rows(node->src[2], d, w.start, w.rows)) {
        if (tile.ith == 0) {
            GGML_LOG_ERROR("FlagOS SpacemiT: MUL_MAT_ID (%s) has an id outside [0, %lld)\n", node->name,
                           (long long) d.n_as);
        }
        return false;
    }

    const uint8_t * const weights = static_cast<const uint8_t *>(node->src[0]->data);
    const int64_t         nth     = tile.nth;
    const int64_t         per     = d.n / nb_cols;  // 32-column slabs of an expert

    if (d.ne12 == 1 || setting == mmid_setting::rows) {
        const int64_t n_pairs = d.n_ids * d.ne12;
        const int64_t units   = n_pairs * per;
        const int64_t u_end   = units * (tile.ith + 1) / nth;
        int64_t       e       = 0;  // the expert of pair j: pairs are in expert order, so it only moves forward
        for (int64_t u = units * tile.ith / nth; u < u_end;) {
            const int64_t j  = u / per;
            const int64_t c0 = u % per;
            const int64_t c1 = std::min(per, c0 + (u_end - u));
            while (w.start[e + 1] <= j) {
                e++;
            }
            const mmid_row & r = w.rows[j];
            const uint8_t *  a = qa + ((r.i1 % d.ne11) + (int64_t) r.i2 * d.ne11) * d.a_row_bytes;
            const uint8_t *  b = weights + (e * d.n + c0 * nb_cols) * d.b_row_bytes;
            d.f->gemm(a, b, out_row(node, r, c0 * nb_cols), 1, (size_t) ((c1 - c0) * nb_cols), (size_t) d.k_blocks,
                      (size_t) d.n, reference);
            u += c1 - c0;
        }
        return true;
    }

    int64_t total = 0;  // the weight of every unit
    for (int64_t e = 0; e < d.n_as; e++) {
        const int64_t r = w.start[e + 1] - w.start[e];
        total += r > 0 ? per * (r + slab_weight) : 0;
    }
    uint8_t * const tcm        = static_cast<uint8_t *>(tile.tcm);
    const int64_t   slab_bytes = nb_cols * d.b_row_bytes;
    const bool      stage      = setting == mmid_setting::staged && tcm != nullptr &&
                       (int64_t) tile.tcm_size >= slab_bytes;

    // the expert at hand: its rows, read from the workspace or, once held, from TCM, and where its slabs are staged
    int64_t         e = -1, r = 0, rows_bytes = 0, count = 0;
    bool            hold_act = false, held = false;
    const uint8_t * acts     = nullptr;
    uint8_t *       slab_buf = nullptr;
    const auto      enter    = [&](int64_t expert) {
        e          = expert;
        r          = w.start[e + 1] - w.start[e];
        rows_bytes = r * d.a_row_bytes;  // the expert's quantized rows, consecutive in the workspace
        hold_act   = stage && (int64_t) GGML_PAD((size_t) rows_bytes, 64) + slab_bytes <= (int64_t) tile.tcm_size;
        acts       = qa + w.start[e] * d.a_row_bytes;
        slab_buf   = stage ? tcm + (hold_act ? GGML_PAD((size_t) rows_bytes, 64) : 0) : nullptr;
        count      = group_count(r);
        held       = false;
    };
    // the unit (e, slab s): the slab staged, then every group of the expert
    const auto unit = [&](int64_t s) {
        const uint8_t * b = weights + (e * d.n + s * nb_cols) * d.b_row_bytes;
        if (stage) {
            if (hold_act && !held) {
                spacemit_ime::copy(tcm, acts, (size_t) rows_bytes);
                acts = tcm;
                held = true;
            }
            spacemit_ime::copy(slab_buf, b, (size_t) slab_bytes);
            b = slab_buf;
        }
        for (int64_t g = 0; g < count; g++) {
            const int64_t   off = group_start(g, r);
            const uint8_t * a   = acts + off * d.a_row_bytes;
            if (group_size(g, r) == row_align) {
                d.f->gemm(a, b, w.result, row_align, (size_t) nb_cols, (size_t) d.k_blocks, (size_t) nb_cols,
                          reference);
                for (int64_t t = 0; t < row_align; t++) {
                    std::memcpy(out_row(node, w.rows[w.start[e] + off + t], s * nb_cols), w.result + t * nb_cols,
                                (size_t) nb_cols * sizeof(float));
                }
            } else {
                d.f->gemm(a, b, out_row(node, w.rows[w.start[e] + off], s * nb_cols), 1, (size_t) nb_cols,
                          (size_t) d.k_blocks, (size_t) d.n, reference);
            }
        }
    };

    if (mmid_deal_from_env() == mmid_deal::fixed) {
        const int64_t lo   = (int64_t) tile.ith * total;  // this tile takes the units with lo <= u * nth < lo + total
        const int64_t hi   = lo + total;
        int64_t       u    = 0;  // the weight of the units before the one at hand
        bool          done = false;
        for (int64_t x = 0; x < d.n_as && !done; x++) {
            const int64_t rx = w.start[x + 1] - w.start[x];
            if (rx == 0) {
                continue;
            }
            const int64_t weight = rx + slab_weight;
            if ((u + per * weight) * nth <= lo) {
                u += per * weight;  // the whole expert lies before this tile's range
                continue;
            }
            enter(x);
            for (int64_t s = 0; s < per; s++, u += weight) {
                if (u * nth < lo) {
                    continue;
                }
                if (u * nth >= hi) {
                    done = true;
                    break;
                }
                unit(s);
            }
        }
        return true;
    }

    // the queue: an expert's slabs in chunks of c, c about a chunks-th of a tile's share, numbered expert-major
    const int64_t target      = std::max<int64_t>(1, total / (nth * mmid_chunks_from_env()));
    const auto    chunk_slabs = [&](int64_t rx) { return std::clamp(target / (rx + slab_weight), int64_t(1), per); };
    int64_t       n_chunks    = 0;
    for (int64_t x = 0; x < d.n_as; x++) {
        const int64_t rx = w.start[x + 1] - w.start[x];
        n_chunks += rx > 0 ? div_up(per, chunk_slabs(rx)) : 0;
    }
    mmid_counter & next = *std::launder(static_cast<mmid_counter *>(mmid_counter_at(tile.workspace, d)));
    int64_t        x = -1, first = 0, n_x = 0, c = 1;  // the last claim's expert, its first chunk, its chunks, its c
    for (;;) {
        const int64_t i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= n_chunks) {
            break;
        }
        while (i >= first + n_x) {  // a tile's claims only grow, so its walk only moves forward
            first += n_x;
            do {
                x++;
            } while (w.start[x + 1] == w.start[x]);
            c   = chunk_slabs(w.start[x + 1] - w.start[x]);
            n_x = div_up(per, c);
        }
        if (x != e) {
            enter(x);
        }
        const int64_t s0 = (i - first) * c;
        const int64_t s1 = std::min(per, s0 + c);
        for (int64_t s = s0; s < s1; s++) {
            unit(s);
        }
    }
    return true;
}
