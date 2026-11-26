#include "flagos-spacemit-f16.h"

#include "flagos-spacemit-f16-kernels.h"  // spacemit_f16: the IME2 leaves
#include "flagos-spacemit-ime-kernels.h"  // row_tile, q4_0_k_block
#include "flagos-spacemit-kernels.h"      // spacemit_cached_switch, spacemit_use_reference
#include "flagos-spacemit-weights.h"      // spacemit_weight_layout

#include "../../../../ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// The FP16 path (plan.md "The dense Q4_0 matmuls in FP16: the port", stage 2): SpacemiT's plain split-K route
// (its ime.cpp 2271-2414). K is cut in 8 slices, ks = K / 8, slice s one per tile (tile ith takes ith, ith + nth, ...).
// The products step takes a slice's rows in passes of m_block; per pass it casts each 16-row tile of the rows into the
// stage, A [ks / 8][16][8] fp16 per tile, then for each weight tile (N rotated by ith, as the row tiles) dequantizes its
// slice into the stage's B [ks / 8][32][8] and multiplies every row tile by it, storing each 16 x 32 result as fp16
// partials. The sum step adds the 8 slices of each output in slice order in float, from +0.0f.
// The stage is the tile's TCM, else its scratch in the workspace. Workspace: the partials, 64-byte aligned (each fp16
// row store stays in one DDR bank, as the fork), [mt][nt][16 rows][8 slices][32 columns] with mt = m / 16, nt = n / 32;
// then 8 tiles' scratch of stage_default bytes. Rows from M to GGML_PAD(M, 16) are never written nor read into the
// output.
// The leaves: the IME2 kernels (flagos-spacemit-f16-kernels.cpp) on the K3 unless GGML_FLAGOS_KERNEL_VARIANTS=0, else
// the scalar references below. cast, dequant and reduce match their kernels bit for bit; products_ref sums each group of
// 8 products in a fixed tree, and vfwmadot in the hardware's own order, so the two may differ by the rounding.
// Overflow fails the step (no clamp): the references test each value, the kernels read the OF flag after their work.

// as flagos-spacemit-seq.cpp: GCC's vector code of scalar loops ran slowly on the A100 cores; the multiply stays scalar,
// with the multiply and the add apart, out of line
#if defined(__GNUC__) && !defined(__clang__)
#    define SPACEMIT_SCALAR_REF __attribute__((noinline, optimize("no-tree-vectorize", "fp-contract=off")))
#else
#    define SPACEMIT_SCALAR_REF
#endif

namespace {

constexpr int64_t n_slices      = 8;
constexpr int64_t out_rows      = 16;  // an output tile's rows, an A tile's
constexpr int64_t out_cols      = (int64_t) spacemit_ime::row_tile;  // its columns, a weight tile's rows
constexpr int64_t part_tile     = out_rows * n_slices * out_cols;    // fp16 partials of one output tile
constexpr int64_t sum_chunk     = 4;   // output tiles a sum step's chunk takes
constexpr int64_t scratch_tiles = 8;   // tiles with a slice: ith < 8
constexpr size_t  stage_default = 393216;  // the stage's bytes without TCM (384 KiB, the K3's TCM a tile)
constexpr int64_t max_ks        = 4096;    // so that 16 rows and a weight tile fit in stage_default
constexpr int64_t max_threshold = 4096;
constexpr int64_t max_elements  = int64_t(1) << 24;  // GGML_PAD(M, 16) * GGML_PAD(N, 32)
// a 32-row tile's bytes per 32 values of K in q4_0 32x256: [fp16 d x 32][16 bytes x 32]; a tile's blocks follow each
// other along K, across the 256-value blocks too
constexpr int64_t q4_0_sub_blk  = 32 * 2 + 32 * 16;

// FLAGOS_SPACEMIT_DENSE_Q4_0: int8, unset or empty (the default): today's path. f16: this path from 64 rows;
// f16:<rows>: from that many rows (1 to 4096). Parsed once (spacemit_cached_switch); anything else warns once per parse
// generation and keeps int8
int parse_dense_q4_0() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_DENSE_Q4_0");
    if (v == nullptr || *v == '\0' || std::strcmp(v, "int8") == 0) {
        return 0;
    }
    if (std::strcmp(v, "f16") == 0) {
        return 64;
    }
    if (std::strncmp(v, "f16:", 4) == 0) {
        const char * digits = v + 4;
        const size_t len    = std::strlen(digits);
        bool         ok     = len >= 1 && len <= 4;
        for (size_t i = 0; ok && i < len; i++) {
            ok = digits[i] >= '0' && digits[i] <= '9';
        }
        const int rows = ok ? std::atoi(digits) : 0;
        if (rows >= 1 && rows <= max_threshold) {
            return rows;
        }
    }
    static std::atomic<uint32_t> warned_in{ 0 };
    const uint32_t               generation = spacemit_switches_generation();
    if (warned_in.exchange(generation) != generation) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_DENSE_Q4_0=%s: not int8, f16 or f16:<rows>, the default (int8) kept\n",
                      __func__, v);
    }
    return 0;
}

// FLAGOS_SPACEMIT_F16_STATS=<path> (test-only): the file the stats are appended to; the path kept beside the flag
std::mutex  g_stats_path_mutex;
std::string g_stats_path;

int parse_f16_stats() {
    const char *                v = std::getenv("FLAGOS_SPACEMIT_F16_STATS");
    std::lock_guard<std::mutex> lock(g_stats_path_mutex);
    g_stats_path = v != nullptr ? v : "";
    return g_stats_path.empty() ? 0 : 1;
}

// what the products steps saw since the last flush (FLAGOS_SPACEMIT_F16_STATS); the largest values only from the
// references (act_weight empty when none recorded)
struct f16_stats {
    std::mutex                                     mutex;
    uint64_t                                       matmuls   = 0;
    uint64_t                                       overflows = 0;
    float                                          max_act   = 0.0f;
    float                                          max_sum   = 0.0f;
    std::string                                    act_weight, sum_weight;
    std::map<std::pair<int64_t, int64_t>, uint64_t> shapes;  // (K, N): matmuls
};

f16_stats & stats() {
    static f16_stats * s = new f16_stats;  // never destroyed: a stream may be destroyed during exit
    return *s;
}

struct f16_dims {
    int64_t m, k, n, ks;
    int64_t m_pad;    // GGML_PAD(m, 16)
    int64_t m_tiles;  // m_pad / 16
    int64_t n_tiles;  // n / 32 (the layout's rows are whole 32-row tiles)
};

f16_dims dims_of(const ggml_tensor * node) {
    const ggml_tensor * w = node->src[0];
    const ggml_tensor * x = node->src[1];
    f16_dims            d;
    d.m       = x->ne[1] * x->ne[2] * x->ne[3];
    d.k       = x->ne[0];
    d.n       = w->ne[1];
    d.ks      = d.k / n_slices;
    d.m_pad   = GGML_PAD(d.m, out_rows);
    d.m_tiles = d.m_pad / out_rows;
    d.n_tiles = d.n / out_cols;
    return d;
}

size_t partials_bytes(const f16_dims & d) {
    return GGML_PAD((size_t) (d.m_tiles * d.n_tiles * part_tile) * sizeof(uint16_t), 64);
}

// the workspace's start rounded up to 64 bytes: the partials, then the tiles' scratch
uint8_t * workspace_base(void * workspace) {
    return reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(workspace) + 63) & ~uintptr_t{ 63 });
}

// the rows a pass takes: the stage holds m_block rows of the slice's activations and one weight tile's slice
int64_t pass_rows(const f16_dims & d, size_t stage_bytes) {
    const int64_t room = ((int64_t) stage_bytes - out_cols * d.ks * 2) / (d.ks * 2);
    return std::min(d.m_pad, room & ~(out_rows - 1));
}

// what a reference leaf saw: the largest finite magnitudes, and the first finite value that became infinite in fp16
struct ref_seen {
    float max_act = 0.0f, max_sum = 0.0f;
    bool  act_overflow = false, sum_overflow = false;
    float act_value = 0.0f, sum_value = 0.0f;
};

// the references, each the arithmetic of its kernel (flagos-spacemit-f16-kernels.h)

// fp16 of the values into a[g][r][j] = x[r][8 g + j], rows from `rows` on zero; as vfncvt, bit for bit
void cast_ref(const float * x, int64_t x_stride, int64_t rows, int64_t ks, uint16_t * a, ref_seen & seen) {
    for (int64_t g = 0; g < ks / 8; g++) {
        for (int64_t r = 0; r < out_rows; r++) {
            for (int64_t j = 0; j < 8; j++) {
                ggml_fp16_t h = 0;
                if (r < rows) {
                    const float v = x[r * x_stride + 8 * g + j];
                    h             = GGML_FP32_TO_FP16(v);
                    if (std::isfinite(v)) {
                        if (std::isinf(GGML_FP16_TO_FP32(h)) && !seen.act_overflow) {
                            seen.act_overflow = true;
                            seen.act_value    = v;
                        }
                        seen.max_act = std::max(seen.max_act, std::fabs(v));
                    }
                }
                a[(g * out_rows + r) * 8 + j] = h;
            }
        }
    }
}

// one slice of a 32-row tile, from its first 32-value block w, into b[g][c][j] = fp16(d_c (q - 8)): the product exact
// in float, so rounded once, as fp16 of ggml's own dequantization and as dequant_tile, bit for bit
void dequant_ref(const uint8_t * w, int64_t ks, uint16_t * b) {
    for (int64_t i = 0; i < ks / 32; i++) {
        const uint8_t * blk = w + i * q4_0_sub_blk;
        for (int64_t c = 0; c < out_cols; c++) {
            ggml_fp16_t dh;
            std::memcpy(&dh, blk + 2 * c, sizeof(dh));
            const float     d  = GGML_FP16_TO_FP32(dh);
            const uint8_t * qs = blk + 2 * out_cols + 16 * c;  // value e in byte e / 2, low nibble first
            for (int64_t e = 0; e < 32; e++) {
                const int q                                     = e % 2 == 0 ? qs[e / 2] & 0x0F : qs[e / 2] >> 4;
                b[((4 * i + e / 8) * out_cols + c) * 8 + e % 8] = GGML_FP32_TO_FP16(d * (float) (q - 8));
            }
        }
    }
}

// rows x cols outputs over ks: per output a float from +0.0f, and per group of 8 in order acc + the tree of its 8
// products (each exact in float), stored as fp16. vfwmadot adds a group in the hardware's order, so this is not the
// kernel bit for bit; flagos-check-spacemit bounds the difference
SPACEMIT_SCALAR_REF void products_ref(int64_t ks, const uint16_t * b, const uint16_t * a, uint16_t * c, int64_t rows,
                                      int64_t cols, int64_t c_stride, ref_seen & seen) {
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t n = 0; n < cols; n++) {
            float acc = 0.0f;
            for (int64_t g = 0; g < ks / 8; g++) {
                const uint16_t * ag = a + (g * out_rows + r) * 8;
                const uint16_t * bg = b + (g * out_cols + n) * 8;
                float            p[8];
                for (int j = 0; j < 8; j++) {
                    p[j] = GGML_FP16_TO_FP32(ag[j]) * GGML_FP16_TO_FP32(bg[j]);
                }
                acc = acc + (((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7])));
            }
            const ggml_fp16_t h = GGML_FP32_TO_FP16(acc);
            if (std::isfinite(acc)) {
                if (std::isinf(GGML_FP16_TO_FP32(h)) && !seen.sum_overflow) {
                    seen.sum_overflow = true;
                    seen.sum_value    = acc;
                }
                seen.max_sum = std::max(seen.max_sum, std::fabs(acc));
            }
            c[r * c_stride + n] = h;
        }
    }
}

// a partial tile [16][8][32] into rows x cols floats: from +0.0f the 8 slices in order, as the reduce kernels
void reduce_ref(const uint16_t * partials, float * out, int64_t out_stride_bytes, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; r++) {
        float * o = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(out) + r * out_stride_bytes);
        for (int64_t n = 0; n < cols; n++) {
            float acc = 0.0f;
            for (int64_t s = 0; s < n_slices; s++) {
                acc = acc + GGML_FP16_TO_FP32(partials[(r * n_slices + s) * out_cols + n]);
            }
            o[n] = acc;
        }
    }
}

// the leaves: the kernels on the K3 (GGML_FLAGOS_SPACEMIT_IME2) unless GGML_FLAGOS_KERNEL_VARIANTS=0
bool use_kernels() {
    return spacemit_f16::have() && !spacemit_use_reference();
}

// has_value false: the kernels' line for a partial sum, which has no value
bool overflow(const ggml_tensor * node, const char * what, float value, bool has_value, bool record) {
    if (has_value) {
        GGML_LOG_ERROR("mul_mat.f16: overflow in %s (weight %s): %s %g exceeds fp16\n", node->name, node->src[0]->name,
                       what, (double) value);
    } else {
        GGML_LOG_ERROR("mul_mat.f16: overflow in %s (weight %s): a %s exceeds fp16\n", node->name, node->src[0]->name,
                       what);
    }
    if (record) {
        f16_stats &                 st = stats();
        std::lock_guard<std::mutex> lock(st.mutex);
        st.overflows++;
    }
    return false;
}

// the kernels raised OF: the tile's slices of the activations scanned for a finite value infinite in fp16 (failure
// path only), else a partial sum overflowed
bool overflow_of_kernels(const ggml_tensor * node, const f16_dims & d, const float * x, const spacemit_tile & tile,
                         bool record) {
    for (int64_t s = tile.ith; s < n_slices; s += tile.nth) {
        for (int64_t m = 0; m < d.m; m++) {
            const float * xr = x + m * d.k + s * d.ks;
            for (int64_t i = 0; i < d.ks; i++) {
                if (std::isfinite(xr[i]) && std::isinf(GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(xr[i])))) {
                    return overflow(node, "activation", xr[i], true, record);
                }
            }
        }
    }
    return overflow(node, "partial sum", 0.0f, false, record);
}

// tile ith takes slices ith, ith + nth, ...; each slice in passes of m_block rows: the pass's row tiles cast into the
// stage, then for every weight tile in the fork's order its slice dequantized into the stage and multiplied by every
// row tile, the partials stored
bool f16_products(const spacemit_tile & tile, ggml_tensor * node, bool kernels) {
    if (tile.ith >= n_slices) {
        return true;
    }
    const f16_dims      d        = dims_of(node);
    const bool          record   = spacemit_f16_stats_on() != 0;
    const ggml_tensor * w        = node->src[0];
    const uint8_t *     wb       = static_cast<const uint8_t *>(w->data);
    const float *       x        = static_cast<const float *>(node->src[1]->data);
    uint8_t *           base     = workspace_base(tile.workspace);
    uint16_t *          partials = reinterpret_cast<uint16_t *>(base);
    uint16_t *          stage    = reinterpret_cast<uint16_t *>(base + partials_bytes(d) + tile.ith * stage_default);
    size_t              stage_b  = stage_default;
    if (tile.tcm != nullptr && pass_rows(d, tile.tcm_size) >= out_rows) {  // else the scratch, which always holds 16
        stage   = static_cast<uint16_t *>(tile.tcm);
        stage_b = tile.tcm_size;
    }
    const int64_t m_block = pass_rows(d, stage_b);
    GGML_ASSERT(m_block >= out_rows);
    uint16_t * const a_stage = stage;
    uint16_t * const b_stage = stage + m_block * d.ks;
    // a 32-row tile's bytes along K
    const int64_t    w_tile  = d.k / 32 * q4_0_sub_blk;

    if (record && tile.ith == 0) {
        f16_stats &                 st = stats();
        std::lock_guard<std::mutex> lock(st.mutex);
        st.matmuls++;
        st.shapes[{ d.k, d.n }]++;
    }
    ref_seen seen;
#if defined(GGML_FLAGOS_SPACEMIT_IME2)
    if (kernels) {
        spacemit_f16::clear_flags();
    }
#endif
    for (int64_t s = tile.ith; s < n_slices; s += tile.nth) {
        const int64_t k0 = s * d.ks;
        for (int64_t m0 = 0; m0 < d.m; m0 += m_block) {
            const int64_t m_tiles = (std::min(m_block, d.m - m0) + out_rows - 1) / out_rows;
            for (int64_t mt = 0; mt < m_tiles; mt++) {
                const int64_t tm0  = m0 + mt * out_rows;
                const int64_t rows = std::min(out_rows, d.m - tm0);
                const float * xt   = x + tm0 * d.k + k0;
                uint16_t *    at   = a_stage + mt * out_rows * d.ks;
#if defined(GGML_FLAGOS_SPACEMIT_IME2)
                if (kernels) {
                    spacemit_f16::cast_tile(xt, d.k, rows, d.ks, at);
                    continue;
                }
#endif
                cast_ref(xt, d.k, rows, d.ks, at, seen);
                if (seen.act_overflow) {
                    return overflow(node, "activation", seen.act_value, true, record);
                }
            }
            // as the fork: N and the row tiles rotated by ith, to spread the tiles' stores
            for (int64_t j = 0; j < d.n_tiles; j++) {
                const int64_t   nt = (j + tile.ith) % d.n_tiles;
                const uint8_t * wt = wb + nt * w_tile + k0 / 32 * q4_0_sub_blk;
#if defined(GGML_FLAGOS_SPACEMIT_IME2)
                if (kernels) {
                    spacemit_f16::dequant_tile(wt, d.ks, b_stage);
                } else
#endif
                {
                    dequant_ref(wt, d.ks, b_stage);
                }
                for (int64_t i = 0; i < m_tiles; i++) {
                    const int64_t    mt   = (i + tile.ith) % m_tiles;
                    const int64_t    tm0  = m0 + mt * out_rows;
                    const int64_t    rows = std::min(out_rows, d.m - tm0);
                    uint16_t *       c    = partials + ((tm0 / out_rows) * d.n_tiles + nt) * part_tile + s * out_cols;
                    const uint16_t * at   = a_stage + mt * out_rows * d.ks;
#if defined(GGML_FLAGOS_SPACEMIT_IME2)
                    if (kernels) {
                        spacemit_f16::products(d.ks, b_stage, at, c, rows, out_cols, n_slices * out_cols);
                        continue;
                    }
#endif
                    products_ref(d.ks, b_stage, at, c, rows, out_cols, n_slices * out_cols, seen);
                    if (seen.sum_overflow) {
                        return overflow(node, "partial sum", seen.sum_value, true, record);
                    }
                }
            }
        }
    }
#if defined(GGML_FLAGOS_SPACEMIT_IME2)
    if (kernels && spacemit_f16::overflow_raised()) {
        return overflow_of_kernels(node, d, x, tile, record);
    }
#endif
    if (record && !kernels) {
        f16_stats &                 st = stats();
        std::lock_guard<std::mutex> lock(st.mutex);
        if (st.act_weight.empty() || seen.max_act > st.max_act) {
            st.max_act    = seen.max_act;
            st.act_weight = w->name;
        }
        if (st.sum_weight.empty() || seen.max_sum > st.max_sum) {
            st.max_sum    = seen.max_sum;
            st.sum_weight = w->name;
        }
    }
    return true;
}

// the output's 16 x 32 tiles t = mt * n_tiles + nt in chunks of 4 consecutive ones, chunk c to tile c % nth; rows from
// M on are not written
bool f16_sum(const spacemit_tile & tile, ggml_tensor * node, bool kernels) {
    const f16_dims   d        = dims_of(node);
    const uint16_t * partials = reinterpret_cast<const uint16_t *>(workspace_base(tile.workspace));
    float *          out      = static_cast<float *>(node->data);
    const int64_t    n_out    = d.m_tiles * d.n_tiles;
    const int64_t    chunks   = (n_out + sum_chunk - 1) / sum_chunk;
    const int64_t    stride   = d.n * (int64_t) sizeof(float);
    GGML_UNUSED(kernels);  // without GGML_FLAGOS_SPACEMIT_IME2
    for (int64_t c = tile.ith; c < chunks; c += tile.nth) {
        for (int64_t t = c * sum_chunk; t < std::min(n_out, (c + 1) * sum_chunk); t++) {
            const int64_t    mt   = t / d.n_tiles;
            const int64_t    nt   = t % d.n_tiles;
            const int64_t    rows = std::min(out_rows, d.m - mt * out_rows);
            const uint16_t * pt   = partials + t * part_tile;
            float *          ot   = out + mt * out_rows * d.n + nt * out_cols;
#if defined(GGML_FLAGOS_SPACEMIT_IME2)
            if (kernels) {
                spacemit_f16::reduce_tile(pt, ot, stride, rows, out_cols);
                continue;
            }
#endif
            reduce_ref(pt, ot, stride, rows, out_cols);
        }
    }
    return true;
}

}  // namespace

uint32_t spacemit_dense_q4_0_threshold() {
    static std::atomic<int>      value{ 0 };
    static std::atomic<uint32_t> seen{ 0 };
    return (uint32_t) spacemit_cached_switch(value, seen, parse_dense_q4_0);
}

uint32_t spacemit_f16_stats_on() {
    static std::atomic<int>      value{ 0 };
    static std::atomic<uint32_t> seen{ 0 };
    return (uint32_t) spacemit_cached_switch(value, seen, parse_f16_stats);
}

// the same answer for both steps; never TCM or the tile count
bool spacemit_mul_mat_f16_supported(const ggml_tensor * op) {
    const uint32_t threshold = spacemit_dense_q4_0_threshold();
    if (threshold == 0 || op->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (w == nullptr || x == nullptr || w->type != GGML_TYPE_Q4_0 || w->ne[2] != 1 || w->ne[3] != 1 ||
        spacemit_weight_layout(w) != spacemit_layout::q4_0_32x256) {
        return false;
    }
    const int64_t k = x->ne[0];
    const int64_t m = x->ne[1] * x->ne[2] * x->ne[3];
    const int64_t n = w->ne[1];
    return k % (int64_t) spacemit_ime::q4_0_k_block == 0 && k / n_slices <= max_ks && m >= (int64_t) threshold &&
           GGML_PAD(m, out_rows) * GGML_PAD(n, out_cols) <= max_elements;
}

size_t spacemit_mul_mat_f16_workspace(const ggml_tensor * node) {
    return partials_bytes(dims_of(node)) + 64 + scratch_tiles * stage_default;
}

bool spacemit_mul_mat_f16_products(const spacemit_tile * tile, ggml_tensor * node) {
    return f16_products(*tile, node, use_kernels());
}

bool spacemit_mul_mat_f16_sum(const spacemit_tile * tile, ggml_tensor * node) {
    return f16_sum(*tile, node, use_kernels());
}

bool spacemit_mul_mat_f16_products_reference(const spacemit_tile * tile, ggml_tensor * node) {
    return f16_products(*tile, node, false);
}

bool spacemit_mul_mat_f16_sum_reference(const spacemit_tile * tile, ggml_tensor * node) {
    return f16_sum(*tile, node, false);
}

void spacemit_mul_mat_f16_stats_flush() {
    f16_stats &                 st = stats();
    std::lock_guard<std::mutex> lock(st.mutex);
    if (st.matmuls == 0) {
        return;
    }
    std::string path;
    if (spacemit_f16_stats_on() != 0) {
        std::lock_guard<std::mutex> path_lock(g_stats_path_mutex);
        path = g_stats_path;
    }
    if (!path.empty()) {
        if (FILE * f = std::fopen(path.c_str(), "a")) {
            std::fprintf(f, "mul_mat.f16: matmuls %llu, overflows %llu, threshold %u\n", (unsigned long long) st.matmuls,
                         (unsigned long long) st.overflows, spacemit_dense_q4_0_threshold());
            if (!st.act_weight.empty()) {
                std::fprintf(f, "mul_mat.f16: largest activation %.6g in %s\n", (double) st.max_act,
                             st.act_weight.c_str());
                std::fprintf(f, "mul_mat.f16: largest partial sum %.6g in %s\n", (double) st.max_sum,
                             st.sum_weight.c_str());
            }
            for (const auto & shape : st.shapes) {
                std::fprintf(f, "mul_mat.f16: K %lld N %lld: %llu matmuls\n", (long long) shape.first.first,
                             (long long) shape.first.second, (unsigned long long) shape.second);
            }
            std::fclose(f);
        } else {
            GGML_LOG_WARN("mul_mat.f16: cannot append the stats to %s\n", path.c_str());
        }
    }
    st.matmuls   = 0;
    st.overflows = 0;
    st.max_act   = 0.0f;
    st.max_sum   = 0.0f;
    st.act_weight.clear();
    st.sum_weight.clear();
    st.shapes.clear();
}

// dequant_ref's B tile read back as [32 rows][ks]
void spacemit_mul_mat_f16_stage_weights(const ggml_tensor * w, int64_t slice, int64_t col_tile, uint16_t * out) {
    const int64_t         k  = w->ne[0];
    const int64_t         ks = k / n_slices;
    std::vector<uint16_t> b((size_t) (out_cols * ks));
    dequant_ref(static_cast<const uint8_t *>(w->data) + (col_tile * (k / 32) + slice * ks / 32) * q4_0_sub_blk, ks,
                b.data());
    for (int64_t g = 0; g < ks / 8; g++) {
        for (int64_t c = 0; c < out_cols; c++) {
            for (int64_t j = 0; j < 8; j++) {
                out[c * ks + 8 * g + j] = b[(g * out_cols + c) * 8 + j];
            }
        }
    }
}
