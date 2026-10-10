#include "flagos-spacemit-ime.h"

#include "flagos-spacemit-ime-kernels.h"
#include "flagos-spacemit-kernels.h"
#include "flagos-spacemit-weights.h"

#include "../../../ggml-impl.h"

#include <algorithm>
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
// Q8_0 weights, and Q6_K weights requantized to Q8_0 at load: the same layout and kernels
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

}  // namespace

size_t spacemit_mul_mat_workspace(const ggml_tensor * node) {
    const mm_dims d = mm_dims_of(node);
    return GGML_PAD((size_t) (d.m * d.a_row_bytes), 64);
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

    // one row (generation): the quantized row in TCM, weights read straight from DRAM, 128 columns per call
    // (ggml-spacemit ime.cpp:421, which found this faster than staging Q4_0 weights in TCM; it stages the weights of
    // the other types in TCM, its path B, which we do not have: plan.md, M2c and M2c.2 designs)
    if (d.m == 1 && tcm != nullptr && (size_t) d.a_row_bytes <= tile.tcm_size) {
        spacemit_ime::copy(tcm, qa, d.a_row_bytes);
        constexpr int64_t tile_cols = 4 * nb_cols;
        for (int64_t n0 = ith * tile_cols; n0 < d.n; n0 += tile_cols * nth) {
            d.f->gemm(tcm, w + n0 * d.b_row_bytes, out + n0, 1, std::min(d.n - n0, tile_cols), d.k_blocks, d.n,
                      reference);
        }
        return true;
    }

    // ime.cpp's split: blocks of rows when there are enough for every tile, otherwise columns
    const int64_t m_stride     = d.n / d.m > 64 ? d.m : 16;
    const int64_t m_blocks     = div_up(d.m, m_stride);
    const int64_t max_n_stride = div_up(d.n * m_blocks, nth);
    int64_t       n_stride     = d.n;
    if (max_n_stride < d.n) {
        n_stride = std::min(n_stride, div_up(max_n_stride, nb_cols) * nb_cols);
    }

    // path A (prefill): each tile stages 4 quantized rows in TCM and streams all weights past them
    if (n_stride == d.n && tcm != nullptr && (size_t) (row_align * d.a_row_bytes) <= tile.tcm_size) {
        for (int64_t m0 = ith * row_align; m0 < d.m; m0 += row_align * nth) {
            const int64_t rows = std::min(d.m - m0, row_align);
            spacemit_ime::copy(tcm, qa + m0 * d.a_row_bytes, rows * d.a_row_bytes);
            for (int64_t n0 = 0; n0 < d.n; n0 += nb_cols) {
                gemm_rows(tcm, w + n0 * d.b_row_bytes, out + m0 * d.n + n0, rows, std::min(d.n - n0, nb_cols), d,
                          reference);
            }
        }
        return true;
    }

    // path C: (rows, columns) tasks straight from DRAM. ime.cpp's path B (weight slabs staged in TCM, core pairs
    // staggered by a pair barrier) would go here; it is deferred (plan.md, M2b design).
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
