#include "flagos-spacemit-ime.h"

#include "flagos-spacemit-ime-format.h"
#include "flagos-spacemit-ime-kernels.h"
#include "flagos-spacemit-kernels.h"
#include "flagos-spacemit-weights.h"

#include "../../../../ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int64_t nb_cols   = spacemit_ime::row_tile;  // weight rows per IME tile
constexpr int64_t row_align = 4;                       // rows of the 4-row kernel

const mm_format k_q4_0_32x256 = { (int64_t) spacemit_ime::q4_0_k_block, spacemit_ime::q4_0_act_row_bytes,
                                  spacemit_ime::q4_0_weight_row_bytes,  spacemit_ime::q4_0_quantize_row,
                                  spacemit_ime::q4_0_quantize_4rows,    spacemit_ime::q4_0_gemm };
// Q8_0 weights, Q4_1 and Q6_K weights requantized to Q8_0 at load, and Q5_0 weights stored as Q8_0 exactly: the same
// layout and kernels
const mm_format k_q8_0_32x32  = { (int64_t) spacemit_ime::q8_0_k_block, spacemit_ime::i8_act_row_bytes,
                                  spacemit_ime::q8_0_weight_row_bytes,  spacemit_ime::i8_quantize_row,
                                  spacemit_ime::i8_quantize_4rows,      spacemit_ime::q8_0_gemm };

}  // namespace

const mm_format & mm_format_of(const ggml_tensor * w) {
    switch (spacemit_weight_layout(w)) {
        case spacemit_layout::q4_0_32x256:
            return k_q4_0_32x256;
        case spacemit_layout::q4_1_q8_0_32x32:
        case spacemit_layout::q8_0_32x32:
        case spacemit_layout::q6_k_q8_0_32x32:
        case spacemit_layout::q5_0_q8_0_32x32:
            return k_q8_0_32x32;
        case spacemit_layout::plain:
            break;
    }
    GGML_ABORT("MUL_MAT weight without an IME layout");
}

bool mm_use_reference() {
    return spacemit_use_reference() || !spacemit_ime::have_ime();
}

namespace {

struct mm_dims {
    const mm_format * f;  // the weight's layout
    int64_t           m;  // activation rows (all batch dimensions flattened; the weight is 2-D)
    int64_t           k;  // row length
    int64_t           n;  // weight rows = output columns
    int64_t           k_blocks;
    int64_t           a_row_bytes;  // quantized activation row
    int64_t           b_row_bytes;  // repacked weight row
};

mm_dims mm_dims_of(const ggml_tensor * node) {
    const ggml_tensor * w = node->src[0];
    const ggml_tensor * x = node->src[1];
    mm_dims             d;
    d.f           = &mm_format_of(w);
    d.m           = x->ne[1] * x->ne[2] * x->ne[3];
    d.k           = x->ne[0];
    d.n           = w->ne[1];
    d.k_blocks    = d.k / d.f->k_block;
    d.a_row_bytes = (int64_t) d.f->act_row_bytes(d.k);
    d.b_row_bytes = (int64_t) d.f->weight_row_bytes(d.k);
    return d;
}

int64_t div_up(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// FLAGOS_SPACEMIT_MM (fusion-and-splitting.md §7.3): fit,staged, the default since 2026-10-25, takes path B where TCM
// holds the batch's activations next to a 32-column slab, otherwise path A where every tile gets 4 rows, otherwise
// path C, and on path B every tile copies and multiplies its own slabs (gemm_slabs; §7.3, Qwen3-0.6B's run). test-only:
// split, the default until then, for comparisons: ime.cpp's choice of path A and path C where ime.cpp takes path B
// (below 113 rows, and for matrices more than 64 times wider than the rows). The same bytes under both. Parsed once
// (spacemit_cached_switch; the check program resets between its settings); anything else is reported once and ignored
enum class mm_setting { split, fit_staged };

mm_setting parse_mm_setting() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_MM");
    if (v == nullptr || *v == '\0' || std::strcmp(v, "fit,staged") == 0) {
        return mm_setting::fit_staged;
    }
    if (std::strcmp(v, "split") == 0) {
        return mm_setting::split;
    }
    static std::atomic<bool> warned{ false };
    if (!warned.exchange(true)) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_MM=%s: unknown setting ignored\n", __func__, v);
    }
    return mm_setting::fit_staged;
}

mm_setting mm_setting_from_env() {
    static std::atomic<int>      value{ 0 };
    static std::atomic<uint32_t> seen{ 0 };
    return static_cast<mm_setting>(
        spacemit_cached_switch(value, seen, [] { return static_cast<int>(parse_mm_setting()); }));
}

// Path A's last m % 4 rows, which the quantize step stores one by one and the kernels multiply one call each
// (gemm_rows): dealt to the tiles that have no group in path A's last round (path_a_tail_tile), a tile's rows in one
// call of multiply. In ime.cpp (406-433) they all go to the tile after the last group, which multiplies them one after
// another while the others wait: on the K3 a tile with 2 or 3 such rows takes about 1.4 or 2.05 times as long as a
// full round of path A, one row 0.68 (Qwen3-8B's op counters, fusion-and-splitting.md §7.3). Two rows in two calls
// stream the weights twice: with groups on the other tiles 1.58 rounds, in one call 1.47 (the same counters). The
// rows' bytes and their kernel calls stay the same, so the output does.
// the tile that multiplies leftover row j of path A's r when dealt. g is the tile after the last group, which holds
// them in ime.cpp; tiles g to nth - 1 have no group in the last round. Each takes one row, the first of them two
// consecutive ones when fewer are idle than rows (two rows in one call take less time than a group and a row,
// measured above); a row beyond that goes to tile 0 after its group (with 8 tiles only when m % 32 = 31: tile 7 takes
// two rows, tile 0 the third). A tile's rows are consecutive, so it multiplies them in one call
int64_t path_a_tail_tile(int64_t j, int64_t r, int64_t g, int64_t nth) {
    const int64_t idle  = nth - g;
    const int64_t pairs = std::min(std::max(r - idle, (int64_t) 0), idle);  // the idle tiles that take two rows
    if (j < 2 * pairs) {
        return g + j / 2;
    }
    return j < std::min(r, 2 * idle) ? g + j - pairs : j - 2 * idle;
}

// test-only (spacemit_mm_paths_take)
std::atomic<uint64_t> g_path_a{ 0 }, g_path_b{ 0 }, g_path_c{ 0 };
// test-only (spacemit_mm_tail_take): path A's leftover rows each tile multiplied, and in how many calls
std::atomic<uint64_t> g_tail_rows[SPACEMIT_MAX_TILES], g_tail_calls[SPACEMIT_MAX_TILES];

// rows [0, rows) of a block: 4 rows per kernel call while 4 remain, then 1
void gemm_rows(const uint8_t * qa, const uint8_t * wb, float * c, int64_t rows, int64_t cols, const mm_dims & d,
               bool reference) {
    while (rows > 0) {
        const auto done = (int64_t) d.f->gemm(qa, wb, c, rows, cols, d.k_blocks, d.n, reference);
        qa += done * d.a_row_bytes;
        c += done * d.n;
        rows -= done;
    }
}

// How a single activation row (generation) is multiplied: the weights staged through TCM 32 columns at a time
// (memcpy1d), each slab multiplied there, slabs dealt round-robin to the tiles, and the two tiles of each pair taking
// turns as the in-tree path B does: while one copies its next slab, the other multiplies. Measured on the K3 (plan.md,
// "1-row matmul bandwidth"): staging takes 10% less time per Qwen3-4B token than reading the weights straight from
// DRAM (memcpy1d reads DRAM faster than the kernels' own loads do; the kernels multiply a slab from TCM about three
// times faster than DRAM delivers it), and the turns 5% less again (tg128 8.7 -> 9.2), as fast as the in-tree path.
// Without room in TCM for a 32-column slab the weights are read straight from DRAM, 128 columns per kernel call.
// One activation row, already quantized into the start of the tile's TCM, times all weight rows; false when a pair
// wait failed
bool gemv(const spacemit_tile & tile, const uint8_t * w, float * out, const mm_dims & d, bool reference) {
    const int64_t ith   = tile.ith;
    const int64_t nth   = tile.nth;
    uint8_t *     tcm   = static_cast<uint8_t *>(tile.tcm);
    const int64_t a_off = GGML_PAD(d.a_row_bytes, 64);
    const int64_t fit   = (int64_t) tile.tcm_size > a_off ?
                              ((int64_t) tile.tcm_size - a_off) / d.b_row_bytes / nb_cols * nb_cols :
                              0;
    const int64_t slab  = std::min(nb_cols, fit);

    if (slab < nb_cols) {
        const int64_t chunk = 4 * nb_cols;
        for (int64_t n0 = ith * chunk; n0 < d.n; n0 += chunk * nth) {
            d.f->gemm(tcm, w + n0 * d.b_row_bytes, out + n0, 1, std::min(d.n - n0, chunk), d.k_blocks, d.n, reference);
        }
        return true;
    }

    // the slab starting at column n0, if there is one: copied into TCM, or multiplied there
    uint8_t *  buf      = tcm + a_off;
    const auto stage    = [&](int64_t n0) {
        if (n0 < d.n) {
            spacemit_ime::copy(buf, w + n0 * d.b_row_bytes, std::min(d.n - n0, slab) * d.b_row_bytes);
        }
    };
    const auto multiply = [&](int64_t n0) {
        if (n0 < d.n) {
            d.f->gemm(tcm, buf, out + n0, 1, std::min(d.n - n0, slab), d.k_blocks, d.n, reference);
        }
    };

    // the in-tree path B (ggml-cpu/spacemit/ime.cpp:435-494): the two tiles of a pair take turns. The even one copies
    // its first slab while the odd one waits; then in every round one multiplies while the other copies. Both wait
    // 1 + rounds times, rounds being the even tile's number of slabs (the odd tile may have one slab fewer and still
    // waits in that last round); a tile without a partner (an odd number of tiles) copies and multiplies alone
    const int64_t step    = slab * nth;
    const int64_t first   = (ith & ~int64_t(1)) * slab;  // the even tile's first slab
    const int64_t rounds  = first < d.n ? div_up(d.n - first, step) : 0;
    const bool    even    = ith % 2 == 0;
    const bool    partner = (ith ^ 1) < nth;
    const auto    wait    = [&] { return !partner || spacemit_pair_wait(tile); };
    if (rounds == 0) {
        return true;
    }
    int64_t n0 = ith * slab;
    if (even) {
        stage(n0);
    }
    if (!wait()) {
        return false;
    }
    if (!even) {
        stage(n0);
    }
    for (int64_t r = 0; r < rounds; r++, n0 += step) {
        if (!even && !wait()) {
            return false;
        }
        multiply(n0);
        if (even && !wait()) {
            return false;
        }
        stage(n0 + step);
    }
    return true;
}

// whether gemm_slabs' TCM holds the batch's quantized activations next to a 32-column slab (ime.cpp:438)
bool slabs_hold_activations(const mm_dims & d, size_t tcm_size) {
    return GGML_PAD(d.m * d.a_row_bytes, 64) + nb_cols * d.b_row_bytes <= (int64_t) tcm_size;
}

// path B for several rows (the default where TCM holds the batch's activations next to a slab): the in-tree path B
// (ggml-cpu/spacemit/ime.cpp:435-494; ggml-spacemit's ime.cpp @ 4e782bc likewise) without its pair turns, which leave
// the odd tile one slab's multiply behind the even one (fusion-and-splitting.md §7.3). 32-column weight slabs, dealt
// round-robin to the tiles, are copied into TCM next to the batch's quantized activations and multiplied there for all
// the batch's rows: every tile copies the activations and its first slab, then multiplies each slab and copies its
// next. Every slab is multiplied by gemm_rows over rows 0 to m, the same kernel calls on the same groups of 4 rows and
// blocks of 32 columns as paths A and C
void gemm_slabs(const spacemit_tile & tile, const uint8_t * qa, const uint8_t * w, float * out, const mm_dims & d,
                bool reference) {
    const int64_t ith   = tile.ith;
    const int64_t nth   = tile.nth;
    uint8_t *     tcm   = static_cast<uint8_t *>(tile.tcm);
    uint8_t *     buf   = tcm + GGML_PAD(d.m * d.a_row_bytes, 64);
    const int64_t step  = nb_cols * nth;
    const int64_t n_beg = ith * nb_cols;
    if (n_beg >= d.n) {
        return;
    }
    spacemit_ime::copy(tcm, qa, d.m * d.a_row_bytes);
    spacemit_ime::copy(buf, w + n_beg * d.b_row_bytes, std::min(d.n - n_beg, nb_cols) * d.b_row_bytes);
    for (int64_t n0 = n_beg; n0 < d.n; n0 += step) {
        gemm_rows(tcm, buf, out + n0, d.m, std::min(d.n - n0, nb_cols), d, reference);
        if (n0 + step < d.n) {
            spacemit_ime::copy(buf, w + (n0 + step) * d.b_row_bytes, std::min(d.n - n0 - step, nb_cols) * d.b_row_bytes);
        }
    }
}

}  // namespace

size_t spacemit_mul_mat_workspace(const ggml_tensor * node) {
    const mm_dims d = mm_dims_of(node);
    return GGML_PAD((size_t) (d.m * d.a_row_bytes), 64);
}

void spacemit_mm_paths_take(uint64_t * path_a, uint64_t * path_b, uint64_t * path_c) {
    *path_a = g_path_a.exchange(0);
    *path_b = g_path_b.exchange(0);
    *path_c = g_path_c.exchange(0);
}

void spacemit_mm_tail_take(uint64_t * rows, uint64_t * calls, uint32_t n_tiles) {
    for (uint32_t i = 0; i < SPACEMIT_MAX_TILES; i++) {
        const uint64_t v = g_tail_rows[i].exchange(0);
        const uint64_t c = g_tail_calls[i].exchange(0);
        if (i < n_tiles) {
            rows[i]  = v;
            calls[i] = c;
        }
    }
}

// the quantize step writes only what its activations and its layout's quantizers give (the 32x32 layouts share theirs)
bool spacemit_mul_mat_same_activations(const ggml_tensor * a, const ggml_tensor * b) {
    const ggml_tensor * xa = a->src[1];
    const ggml_tensor * xb = b->src[1];
    const mm_dims       da = mm_dims_of(a);
    const mm_dims       db = mm_dims_of(b);
    return xa->data == xb->data && xa->type == xb->type && ggml_are_same_shape(xa, xb) && ggml_are_same_stride(xa, xb) &&
           da.f->k_block == db.f->k_block && da.f->act_row_bytes == db.f->act_row_bytes &&
           da.f->quantize_row == db.f->quantize_row && da.f->quantize_4rows == db.f->quantize_4rows;
}

// step 1: quantize the activations into the workspace. One row: the tiles split its K blocks. Several rows: the tiles
// split blocks of 4 rows; full blocks are stored interleaved for the 4-row kernel, the last partial block row by row.
bool spacemit_mul_mat_quantize(const spacemit_tile * tile_p, ggml_tensor * node) {
    const spacemit_tile & tile = *tile_p;
    const mm_dims d         = mm_dims_of(node);
    const bool    reference = mm_use_reference();
    const float * x         = static_cast<const float *>(node->src[1]->data);
    uint8_t *     qa        = static_cast<uint8_t *>(tile.workspace);

    if (d.m == 1) {
        const int64_t blk_bytes = (int64_t) d.f->act_row_bytes(d.f->k_block);
        const int64_t per_tile  = div_up(d.k_blocks, tile.nth);
        const int64_t b0        = tile.ith * per_tile;
        const int64_t b1        = std::min(b0 + per_tile, d.k_blocks);
        if (b0 < b1) {
            d.f->quantize_row(x + b0 * d.f->k_block, (b1 - b0) * d.f->k_block, qa + b0 * blk_bytes, reference);
        }
        return true;
    }

    const int64_t row_blocks = div_up(d.m, row_align);
    const int64_t per_tile   = div_up(row_blocks, tile.nth);
    const int64_t rb_end     = std::min((tile.ith + 1) * per_tile, row_blocks);
    for (int64_t rb = tile.ith * per_tile; rb < rb_end; rb++) {
        const int64_t r0   = rb * row_align;
        const int64_t rows = std::min(row_align, d.m - r0);
        if (rows == row_align) {
            d.f->quantize_4rows(x + r0 * d.k, d.k, qa + r0 * d.a_row_bytes, reference);
        } else {
            for (int64_t r = r0; r < r0 + rows; r++) {
                d.f->quantize_row(x + r * d.k, d.k, qa + r * d.a_row_bytes, reference);
            }
        }
    }
    return true;
}

// step 2: multiply. The path depends only on the shape and the TCM size, which all tiles agree on.
bool spacemit_mul_mat_gemm(const spacemit_tile * tile_p, ggml_tensor * node) {
    const spacemit_tile & tile = *tile_p;
    const mm_dims   d         = mm_dims_of(node);
    const bool      reference = mm_use_reference();
    const uint8_t * qa        = static_cast<const uint8_t *>(tile.workspace);
    const uint8_t * w         = static_cast<const uint8_t *>(node->src[0]->data);
    float *         out       = static_cast<float *>(node->data);
    uint8_t *       tcm       = static_cast<uint8_t *>(tile.tcm);
    const int64_t   ith       = tile.ith;
    const int64_t   nth       = tile.nth;

    // one row (generation): the quantized row in TCM, the weights staged through TCM (gemv above)
    if (d.m == 1 && tcm != nullptr && (size_t) d.a_row_bytes <= tile.tcm_size) {
        spacemit_ime::copy(tcm, qa, d.a_row_bytes);
        return gemv(tile, w, out, d, reference);
    }

    // ime.cpp's split: blocks of rows when there are enough for every tile, otherwise columns
    const int64_t m_stride     = d.n / d.m > 64 ? d.m : 16;
    const int64_t m_blocks     = div_up(d.m, m_stride);
    const int64_t max_n_stride = div_up(d.n * m_blocks, nth);
    int64_t       n_stride     = d.n;
    if (max_n_stride < d.n) {
        n_stride = std::min(n_stride, div_up(max_n_stride, nb_cols) * nb_cols);
    }

    // path A (batches with 4 rows for every tile): each tile stages 4 quantized rows in TCM and streams all weights
    // past them, the groups of 4 dealt round-robin to the tiles, the last m % 4 rows one per idle tile, a tile's in
    // one call (path_a_tail_tile). By default whenever every tile gets 4 rows, unless path B holds the activations in
    // TCM; with split only where ime.cpp's split gives every tile all columns
    const bool fit    = mm_setting_from_env() == mm_setting::fit_staged;
    const bool fit_b  = fit && tcm != nullptr && slabs_hold_activations(d, tile.tcm_size);
    const bool rows_a = fit && !fit_b && d.m >= row_align * nth;
    if (!fit_b && (n_stride == d.n || rows_a) && tcm != nullptr &&
        (size_t) (row_align * d.a_row_bytes) <= tile.tcm_size) {
        if (ith == 0) {
            g_path_a.fetch_add(1, std::memory_order_relaxed);
        }
        const auto multiply = [&](int64_t m0, int64_t rows) {
            spacemit_ime::copy(tcm, qa + m0 * d.a_row_bytes, rows * d.a_row_bytes);
            for (int64_t n0 = 0; n0 < d.n; n0 += nb_cols) {
                gemm_rows(tcm, w + n0 * d.b_row_bytes, out + m0 * d.n + n0, rows, std::min(d.n - n0, nb_cols), d,
                          reference);
            }
        };
        const int64_t m4 = d.m - d.m % row_align;  // the rows in groups of 4
        const int64_t r  = d.m - m4;               // the rows left over
        const int64_t g  = m4 / row_align % nth;   // the tile after the last group
        for (int64_t m0 = ith * row_align; m0 < m4; m0 += row_align * nth) {
            multiply(m0, row_align);
        }
        // the leftover rows in runs of the same tile, each run in one call (with the rule above, one run per tile)
        for (int64_t j = 0; j < r;) {
            const int64_t t     = path_a_tail_tile(j, r, g, nth);
            int64_t       count = 1;
            while (j + count < r && path_a_tail_tile(j + count, r, g, nth) == t) {
                count++;
            }
            if (t == ith) {
                multiply(m4 + j, count);
                g_tail_rows[ith].fetch_add(count, std::memory_order_relaxed);
                g_tail_calls[ith].fetch_add(1, std::memory_order_relaxed);
            }
            j += count;
        }
        return true;
    }

    // path B (gemm_slabs): by default where TCM holds the activations next to a 32-column slab
    if (fit_b) {
        if (ith == 0) {
            g_path_b.fetch_add(1, std::memory_order_relaxed);
        }
        gemm_slabs(tile, qa, w, out, d, reference);
        return true;
    }

    // path C: (rows, columns) tasks straight from DRAM, ime.cpp's choice only without TCM. By default below 32 rows
    // when TCM cannot hold the activations next to a slab, and without TCM; with split (test-only) it stands in for
    // path B, as it did by default from M2b, which deferred path B, to 2026-10-25 (plan.md, M2b design and its review
    // of 2026-10-21)
    if (ith == 0) {
        g_path_c.fetch_add(1, std::memory_order_relaxed);
    }
    const int64_t tasks_m = div_up(d.m, m_stride);
    const int64_t tasks   = tasks_m * div_up(d.n, n_stride);
    const int64_t per     = div_up(tasks, nth);
    const int64_t t_end   = std::min((ith + 1) * per, tasks);
    for (int64_t t = ith * per; t < t_end; t++) {
        const int64_t m0    = (t % tasks_m) * m_stride;
        const int64_t rows  = std::min(d.m - m0, m_stride);
        const int64_t n0    = (t / tasks_m) * n_stride;
        const int64_t cols  = std::min(d.n - n0, n_stride);
        const int64_t n_blk = rows == 1 ? cols : nb_cols;
        for (int64_t ni = 0; ni < cols; ni += n_blk) {
            gemm_rows(qa + m0 * d.a_row_bytes, w + (n0 + ni) * d.b_row_bytes, out + m0 * d.n + n0 + ni, rows,
                      std::min(cols - ni, n_blk), d, reference);
        }
    }
    return true;
}
