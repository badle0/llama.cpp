#include "flagos-spacemit-ime.h"

#include "flagos-spacemit-ime-kernels.h"
#include "flagos-spacemit-kernels.h"
#include "flagos-spacemit-weights.h"

#include "../../../ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int64_t nb_cols   = spacemit_ime::row_tile;  // weight rows per IME tile
constexpr int64_t row_align = 4;                       // rows of the 4-row kernel

// one IME weight layout: its K block, its row sizes and its kernels (flagos-spacemit-ime-kernels.h)
struct mm_format {
    int64_t k_block;  // K values per block, in the activations and in the weights
    size_t (*act_row_bytes)(size_t k);
    size_t (*weight_row_bytes)(size_t k);
    void (*quantize_row)(const float * a, size_t count_k, uint8_t * qa, bool reference);
    void (*quantize_4rows)(const float * a, size_t count_k, uint8_t * qa, bool reference);
    size_t (*gemm)(const uint8_t * qa, const uint8_t * qb, float * c, size_t count_m, size_t count_n, size_t k_blocks,
                   size_t ldc, bool reference);
};

const mm_format k_q4_0_32x256 = { (int64_t) spacemit_ime::q4_0_k_block, spacemit_ime::q4_0_act_row_bytes,
                                  spacemit_ime::q4_0_weight_row_bytes,  spacemit_ime::q4_0_quantize_row,
                                  spacemit_ime::q4_0_quantize_4rows,    spacemit_ime::q4_0_gemm };
const mm_format k_q4_1_32x32  = { (int64_t) spacemit_ime::q4_1_k_block, spacemit_ime::i8_act_row_bytes,
                                  spacemit_ime::q4_1_weight_row_bytes,  spacemit_ime::i8_quantize_row,
                                  spacemit_ime::i8_quantize_4rows,      spacemit_ime::q4_1_gemm };
// Q8_0 weights, and Q4_1 and Q6_K weights requantized to Q8_0 at load: the same layout and kernels
const mm_format k_q8_0_32x32  = { (int64_t) spacemit_ime::q8_0_k_block, spacemit_ime::i8_act_row_bytes,
                                  spacemit_ime::q8_0_weight_row_bytes,  spacemit_ime::i8_quantize_row,
                                  spacemit_ime::i8_quantize_4rows,      spacemit_ime::q8_0_gemm };

// spacemit_find_op admits only weights with an IME layout
const mm_format & mm_format_of(const ggml_tensor * w) {
    switch (spacemit_weight_layout(w)) {
        case spacemit_layout::q4_0_32x256:
            return k_q4_0_32x256;
        case spacemit_layout::q4_1_32x32:
            return k_q4_1_32x32;
        case spacemit_layout::q4_1_q8_0_32x32:
        case spacemit_layout::q8_0_32x32:
        case spacemit_layout::q6_k_q8_0_32x32:
            return k_q8_0_32x32;
        case spacemit_layout::plain:
            break;
    }
    GGML_ABORT("MUL_MAT weight without an IME layout");
}

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

// test-only FLAGOS_SPACEMIT_TEST_REFERENCE (flagos-spacemit-kernels.h); without the IME kernels the scalar references
// are always used
bool use_reference() {
    return spacemit_use_reference() || !spacemit_ime::have_ime();
}

// FLAGOS_SPACEMIT_MM (fusion-and-splitting.md §7.3): fit,staged, the default since 2026-10-25, takes path B where TCM
// holds the batch's activations next to a 32-column slab, otherwise path A where every tile gets 4 rows, otherwise
// path C (from the steps measured per shape under split, rows and spacemit), and on path B every tile copies and
// multiplies its own slabs without the pair turns (gemm_slabs; §7.3, Qwen3-0.6B's run). The others are test-only, for
// comparisons: split, the default until then, keeps ime.cpp's choice of path A and takes path C where ime.cpp takes
// path B: below 113 rows, and for matrices more than 64 times wider than the rows; rows takes path A for every call
// that gives each tile at least one group of 4 rows; spacemit takes path B there instead of path C whenever TCM holds
// a slab, as ime.cpp and ggml-spacemit do; fit takes fit,staged's paths with the turns on path B; fit,mixed takes them
// with the turns only while the batch's activations take fewer bytes than a slab (slabs_paired). The same bytes in all
// six. Read at every call with several rows, so a benchmark can switch; anything else is reported once and ignored
enum class mm_setting { split, rows, spacemit, fit, fit_staged, fit_mixed };

mm_setting mm_setting_from_env() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_MM");
    if (v == nullptr || *v == '\0' || std::strcmp(v, "fit,staged") == 0) {
        return mm_setting::fit_staged;
    }
    if (std::strcmp(v, "split") == 0) {
        return mm_setting::split;
    }
    if (std::strcmp(v, "rows") == 0) {
        return mm_setting::rows;
    }
    if (std::strcmp(v, "spacemit") == 0) {
        return mm_setting::spacemit;
    }
    if (std::strcmp(v, "fit") == 0) {
        return mm_setting::fit;
    }
    if (std::strcmp(v, "fit,mixed") == 0) {
        return mm_setting::fit_mixed;
    }
    static std::atomic<bool> warned{ false };
    if (!warned.exchange(true)) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_MM=%s: unknown setting ignored\n", __func__, v);
    }
    return mm_setting::fit_staged;
}

// Path A's last m % 4 rows, which the quantize step stores one by one and the kernels multiply one call each
// (gemm_rows): dealt to the tiles that have no group in path A's last round (path_a_tail_tile), a tile's rows in one
// call of multiply. In ime.cpp (406-433) they all go to the tile after the last group, which multiplies them one after
// another while the others wait: on the K3 a tile with 2 or 3 such rows takes about 1.4 or 2.05 times as long as a
// full round of path A, one row 0.68 (Qwen3-8B's op counters, fusion-and-splitting.md §7.3). Two rows in two calls
// stream the weights twice: with groups on the other tiles 1.58 rounds, in one call 1.47 (the same counters). The
// rows' bytes and their kernel calls stay the same, so the output does. test-only: FLAGOS_SPACEMIT_TAIL=one-tile keeps
// ime.cpp's assignment, for comparisons; read at every call with several rows, so a benchmark can switch; anything
// else is reported once and ignored
bool path_a_tail_one_tile() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_TAIL");
    if (v == nullptr || *v == '\0' || std::strcmp(v, "dealt") == 0) {
        return false;
    }
    if (std::strcmp(v, "one-tile") == 0) {
        return true;
    }
    static std::atomic<bool> warned{ false };
    if (!warned.exchange(true)) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_TAIL=%s: unknown setting ignored\n", __func__, v);
    }
    return false;
}

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

// test-only (spacemit_mm_paths_take); g_path_b_staged: path B steps whose tile 0 ran without the pair turns
std::atomic<uint64_t> g_path_a{ 0 }, g_path_b{ 0 }, g_path_c{ 0 }, g_path_b_staged{ 0 };
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
// turns as the in-tree path B does: while one copies its next slab, the other multiplies (gemv below). Measured on the
// K3 (plan.md, "1-row matmul bandwidth"): staging takes 10% less time per Qwen3-4B token than reading the weights
// straight from DRAM (memcpy1d reads DRAM faster than the kernels' own loads do; the kernels multiply a slab from TCM
// about three times faster than DRAM delivers it), and the turns 5% less again (tg128 8.7 -> 9.2), as fast as the
// in-tree path. Without room in TCM for a 32-column slab the weights are read straight from DRAM.
// test-only: FLAGOS_SPACEMIT_GEMV=<strategy>[:<columns>][@<tiles>], read at every call, for comparisons: paired[:c]
// (c = 32, the default); staged[:c] (c = 32): the same without the turns; direct[:c] (c = 128): straight from DRAM, c
// columns per kernel call; copy[:c]: staging's copies alone, wrong results (the control of flagos-check-spacemit's
// strategy check); @t: only the first t tiles take part, the others wait (also gives a tile without a partner)
struct gemv_strategy {
    enum kind_t { paired, staged, direct, copy_only };
    kind_t  kind  = paired;
    int64_t cols  = nb_cols;
    int64_t tiles = 0;  // 0: all
};

gemv_strategy gemv_strategy_from_env() {
    static const struct {
        const char *          name;
        gemv_strategy::kind_t kind;
        int64_t               cols;
    } names[] = {
        { "paired", gemv_strategy::paired, nb_cols },
        { "staged", gemv_strategy::staged, nb_cols },
        { "direct", gemv_strategy::direct, 4 * nb_cols },
        { "copy", gemv_strategy::copy_only, nb_cols },
    };
    gemv_strategy s;
    const char *  v = std::getenv("FLAGOS_SPACEMIT_GEMV");
    if (v == nullptr) {
        return s;
    }
    const size_t len = std::strcspn(v, ":@");
    for (const auto & n : names) {
        if (std::strlen(n.name) == len && std::strncmp(v, n.name, len) == 0) {
            s.kind = n.kind;
            s.cols = n.cols;
        }
    }
    const char * cols  = std::strchr(v, ':');
    const char * tiles = std::strchr(v, '@');
    if (cols != nullptr && std::atoll(cols + 1) >= nb_cols) {
        s.cols = std::atoll(cols + 1) / nb_cols * nb_cols;
    }
    if (tiles != nullptr && std::atoll(tiles + 1) > 0) {
        s.tiles = std::atoll(tiles + 1);
    }
    return s;
}

// one activation row, already quantized into the start of the tile's TCM, times all weight rows (gemv_strategy); false
// when a pair wait failed
bool gemv(const gemv_strategy & s, const spacemit_tile & tile, const uint8_t * w, float * out, const mm_dims & d,
          bool reference) {
    const int64_t ith = tile.ith;
    int64_t       nth = tile.nth;
    if (s.tiles > 0) {
        nth = std::min(nth, s.tiles);
        if (ith >= nth) {
            return true;
        }
    }
    uint8_t *     tcm   = static_cast<uint8_t *>(tile.tcm);
    const int64_t a_off = GGML_PAD(d.a_row_bytes, 64);
    const int64_t fit   = (int64_t) tile.tcm_size > a_off ?
                              ((int64_t) tile.tcm_size - a_off) / d.b_row_bytes / nb_cols * nb_cols :
                              0;
    const int64_t slab  = std::min(s.cols, fit);

    if (s.kind == gemv_strategy::direct || slab < nb_cols) {
        const int64_t chunk = s.kind == gemv_strategy::direct ? s.cols : 4 * nb_cols;
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

    if (s.kind != gemv_strategy::paired) {  // staged, or its copies alone
        for (int64_t n0 = ith * slab; n0 < d.n; n0 += slab * nth) {
            stage(n0);
            if (s.kind == gemv_strategy::staged) {
                multiply(n0);
            }
        }
        return true;
    }

    // paired, the in-tree path B (ggml-cpu/spacemit/ime.cpp:435-494): the two tiles of a pair take turns. The even one
    // copies its first slab while the odd one waits; then in every round one multiplies while the other copies. Both
    // wait 1 + rounds times, rounds being the even tile's number of slabs (the odd tile may have one slab fewer and
    // still waits in that last round); a tile without a partner (an odd number of tiles) runs as staged
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

// fit and its two variants choose the paths alike
bool takes_fit_paths(mm_setting setting) {
    return setting == mm_setting::fit || setting == mm_setting::fit_staged || setting == mm_setting::fit_mixed;
}

// whether gemm_slabs' pairs take turns: never under fit,staged (the default), under fit,mixed only while the batch's
// activations take fewer bytes than a 32-column slab, always otherwise. With few rows a step is mostly copies, which
// the turns keep apart; with more the multiplies dominate and the turns' tail costs more than the copies' collisions
// (measured on the K3 at 2-320 rows, crossing between 8 and 16 rows for Q4_0 and Q4_1: fusion-and-splitting.md §7.3).
// The row length cancels, so the threshold depends on the layout only: 16 rows for Q4_0 and Q4_1, 29 for Q8_0 (not
// measured)
bool slabs_paired(const mm_dims & d, mm_setting setting) {
    switch (setting) {
        case mm_setting::fit_staged:
            return false;
        case mm_setting::fit_mixed:
            return d.m * d.a_row_bytes < nb_cols * d.b_row_bytes;
        default:
            return true;
    }
}

// path B for several rows (by default, and with the test-only FLAGOS_SPACEMIT_MM=spacemit, fit and fit,mixed): the
// in-tree path B (ggml-cpu/spacemit/ime.cpp:435-494; ggml-spacemit's ime.cpp @ 4e782bc likewise). 32-column weight
// slabs, dealt round-robin to the tiles, are copied into TCM and multiplied there for all the batch's rows. The batch's
// quantized activations go into TCM too when they fit next to a slab (ime.cpp:438-441); otherwise the kernels read
// them from the workspace. The two tiles of a pair take turns as in gemv's paired (the even tile copies its first slab
// and the activations while the odd one waits, then in every round one multiplies while the other copies; both wait
// 1 + rounds times, rounds being the even tile's number of slabs; a tile without a partner runs alone). The turns leave
// the odd tile one slab's multiply behind the even one (fusion-and-splitting.md §7.3); without them (paired false: the
// default fit,staged, and fit,mixed from its threshold, slabs_paired) every tile runs alone, as gemv's staged: it
// copies its first slab and the activations, then multiplies each slab and copies its next. Every slab is multiplied by
// gemm_rows over rows 0 to m, the same kernel calls on the same groups of 4 rows and blocks of 32 columns as paths A
// and C. False when a pair wait failed
bool gemm_slabs(const spacemit_tile & tile, const uint8_t * qa, const uint8_t * w, float * out, const mm_dims & d,
                bool reference, bool paired) {
    const int64_t   ith   = tile.ith;
    const int64_t   nth   = tile.nth;
    uint8_t *       tcm   = static_cast<uint8_t *>(tile.tcm);
    const int64_t   a_off = GGML_PAD(d.m * d.a_row_bytes, 64);
    const bool      a_tcm = slabs_hold_activations(d, tile.tcm_size);
    const uint8_t * acts  = a_tcm ? tcm : qa;
    uint8_t *       buf   = a_tcm ? tcm + a_off : tcm;

    const auto stage = [&](int64_t n0) {
        if (n0 < d.n) {
            spacemit_ime::copy(buf, w + n0 * d.b_row_bytes, std::min(d.n - n0, nb_cols) * d.b_row_bytes);
        }
    };
    const auto stage_acts = [&](int64_t n0) {
        if (a_tcm && n0 < d.n) {
            spacemit_ime::copy(tcm, qa, d.m * d.a_row_bytes);
        }
    };
    const auto multiply = [&](int64_t n0) {
        if (n0 < d.n) {
            gemm_rows(acts, buf, out + n0, d.m, std::min(d.n - n0, nb_cols), d, reference);
        }
    };

    const int64_t step    = nb_cols * nth;
    const int64_t first   = (ith & ~int64_t(1)) * nb_cols;  // the even tile's first slab
    const int64_t rounds  = first < d.n ? div_up(d.n - first, step) : 0;
    const bool    even    = ith % 2 == 0;
    const bool    partner = paired && (ith ^ 1) < nth;
    const auto    wait    = [&] { return !partner || spacemit_pair_wait(tile); };
    if (ith == 0 && !partner) {
        g_path_b_staged.fetch_add(1, std::memory_order_relaxed);
    }
    if (rounds == 0) {
        return true;
    }
    int64_t n0 = ith * nb_cols;
    if (even) {
        stage(n0);
        stage_acts(n0);
    }
    if (!wait()) {
        return false;
    }
    if (!even) {
        stage_acts(n0);
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

}  // namespace

size_t spacemit_mul_mat_workspace(const ggml_tensor * node) {
    const mm_dims d = mm_dims_of(node);
    return GGML_PAD((size_t) (d.m * d.a_row_bytes), 64);
}

void spacemit_mm_paths_take(uint64_t & path_a, uint64_t & path_b, uint64_t & path_c, uint64_t & path_b_staged) {
    path_a        = g_path_a.exchange(0);
    path_b        = g_path_b.exchange(0);
    path_c        = g_path_c.exchange(0);
    path_b_staged = g_path_b_staged.exchange(0);
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
bool spacemit_mul_mat_quantize(const spacemit_tile & tile, ggml_tensor * node) {
    const mm_dims d         = mm_dims_of(node);
    const bool    reference = use_reference();
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
bool spacemit_mul_mat_gemm(const spacemit_tile & tile, ggml_tensor * node) {
    const mm_dims   d         = mm_dims_of(node);
    const bool      reference = use_reference();
    const uint8_t * qa        = static_cast<const uint8_t *>(tile.workspace);
    const uint8_t * w         = static_cast<const uint8_t *>(node->src[0]->data);
    float *         out       = static_cast<float *>(node->data);
    uint8_t *       tcm       = static_cast<uint8_t *>(tile.tcm);
    const int64_t   ith       = tile.ith;
    const int64_t   nth       = tile.nth;

    // one row (generation): the quantized row in TCM, the weights staged through TCM (gemv_strategy above)
    if (d.m == 1 && tcm != nullptr && (size_t) d.a_row_bytes <= tile.tcm_size) {
        spacemit_ime::copy(tcm, qa, d.a_row_bytes);
        return gemv(gemv_strategy_from_env(), tile, w, out, d, reference);
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
    // one call (path_a_tail_tile). By default (fit,staged; and with fit and fit,mixed) whenever every tile gets 4
    // rows, unless path B holds the activations in TCM; with FLAGOS_SPACEMIT_MM=rows whenever every tile gets 4 rows;
    // with split only where ime.cpp's split gives every tile all columns
    const mm_setting setting = mm_setting_from_env();
    const bool       fit     = takes_fit_paths(setting);
    const bool       fit_b   = fit && tcm != nullptr && slabs_hold_activations(d, tile.tcm_size);
    const bool       rows_a  = d.m >= row_align * nth && (setting == mm_setting::rows || (fit && !fit_b));
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
        if (path_a_tail_one_tile()) {
            for (int64_t m0 = ith * row_align; m0 < d.m; m0 += row_align * nth) {
                multiply(m0, std::min(d.m - m0, row_align));
            }
            if (ith == g && r > 0) {
                g_tail_rows[ith].fetch_add(r, std::memory_order_relaxed);
                g_tail_calls[ith].fetch_add(1, std::memory_order_relaxed);
            }
            return true;
        }
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

    // path B (gemm_slabs): by default (fit,staged) where TCM holds the activations next to a 32-column slab, without
    // the turns; with fit there too, with the turns; with fit,mixed there too, with the turns while the activations
    // take fewer bytes than a slab (slabs_paired); with FLAGOS_SPACEMIT_MM=spacemit where ime.cpp takes it, whenever
    // TCM holds a slab
    if ((setting == mm_setting::spacemit || fit_b) && tcm != nullptr &&
        nb_cols * d.b_row_bytes <= (int64_t) tile.tcm_size) {
        if (ith == 0) {
            g_path_b.fetch_add(1, std::memory_order_relaxed);
        }
        return gemm_slabs(tile, qa, w, out, d, reference, slabs_paired(d, setting));
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
