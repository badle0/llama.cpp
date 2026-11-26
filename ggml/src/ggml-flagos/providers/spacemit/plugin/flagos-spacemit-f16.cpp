#include "flagos-spacemit-f16.h"

#include "flagos-spacemit-ime-kernels.h"  // row_tile, q4_0_k_block
#include "flagos-spacemit-kernels.h"      // spacemit_cached_switch
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

// Stage 1 of the FP16 path (plan.md "The dense Q4_0 matmuls in FP16: the port"): every piece scalar, to measure the
// arithmetic before any kernel is ported. For slice s of 8 (k in [s * ks, (s + 1) * ks), ks = K / 8), row m, column n:
// a16 = fp16(activation) and w16 = fp16(d (q - 8)), both rounded to nearest even once; P_s = the float sum of
// a16 * w16 in 8 lanes (lane j takes k - s * ks = j mod 8, in increasing k), then ((l0 + l1) + (l2 + l3)) + ((l4 + l5)
// + (l6 + l7)); stored as fp16(P_s) (or as float under FLAGOS_SPACEMIT_F16_SUM=f32); out = p_0 + p_1 + ... + p_7 in
// float, in slice order. A finite activation or partial sum that becomes infinite in fp16 fails the step (no clamp).
//
// Workspace: the partials, then 8 tiles' scratch. Partial (m, n, s) is at ((t * 8 + s) * 16 + m % 16) * 32 + n % 32,
// t = (n / 32) * GGML_PAD(M, 16) / 16 + m / 16 the output's 16 x 32 tile (column tiles outer), so each output tile's 8
// slices are 8 consecutive 16 x 32 blocks; rows from M to GGML_PAD(M, 16) are never written nor read into the output.
// A tile's scratch is the stage, [m_block x ks fp16 activations][32 x ks fp16 weights], as it would sit in TCM in
// stage 2 (here ordinary memory: scalar reads of TCM are slow on the K3), and exact float copies of the same values,
// which the multiply reads.

// as flagos-spacemit-seq.cpp: GCC's vector code of scalar loops ran slowly on the A100 cores; the multiply stays scalar,
// with the multiply and the add apart, out of line
#if defined(__GNUC__) && !defined(__clang__)
#    define SPACEMIT_SCALAR_REF __attribute__((noinline, optimize("no-tree-vectorize", "fp-contract=off")))
#else
#    define SPACEMIT_SCALAR_REF
#endif

namespace {

constexpr int64_t n_slices      = 8;
constexpr int64_t out_rows      = 16;  // an output tile's rows
constexpr int64_t out_cols      = (int64_t) spacemit_ime::row_tile;  // its columns, a weight tile's rows
constexpr int64_t sum_chunk     = 4;   // output tiles a sum step's chunk takes
constexpr int64_t scratch_tiles = 8;   // tiles with a slice: ith < 8
constexpr size_t  stage_default = 393216;  // the stage's bytes without TCM (384 KiB, the K3's TCM a tile)
constexpr int64_t max_ks        = 4096;    // so that 16 rows and a weight tile fit in stage_default
constexpr int64_t max_threshold = 4096;
constexpr int64_t max_elements  = int64_t(1) << 24;  // GGML_PAD(M, 16) * GGML_PAD(N, 32)
// a 32-row tile's bytes per 256 values of K in q4_0 32x256: 8 blocks of [fp16 d x 32][16 bytes x 32]
constexpr int64_t q4_0_tile_blk = 8 * (32 * 2 + 32 * 16);
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

// FLAGOS_SPACEMIT_F16_SUM (test-only): f16, unset or empty: the partials stored as fp16; f32: as float
int parse_f16_sum() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_F16_SUM");
    if (v == nullptr || *v == '\0' || std::strcmp(v, "f16") == 0) {
        return 0;
    }
    if (std::strcmp(v, "f32") == 0) {
        return 1;
    }
    static std::atomic<uint32_t> warned_in{ 0 };
    const uint32_t               generation = spacemit_switches_generation();
    if (warned_in.exchange(generation) != generation) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_F16_SUM=%s: not f16 or f32, f16 kept\n", __func__, v);
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

// what the products steps saw since the last flush (FLAGOS_SPACEMIT_F16_STATS)
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

size_t partials_bytes(const f16_dims & d, bool f32_sum) {
    return GGML_PAD((size_t) (d.m_pad * d.n_tiles * out_cols * n_slices) * (f32_sum ? 4 : 2), 64);
}

// a tile's fp16 stage and the float copies, each for up to all rows
size_t stage_f16_bytes(const f16_dims & d) {
    return GGML_PAD((size_t) ((d.m_pad + out_cols) * d.ks) * 2, 64);
}

size_t scratch_bytes(const f16_dims & d) {
    return stage_f16_bytes(d) + GGML_PAD((size_t) ((d.m_pad + out_cols) * d.ks) * 4, 64);
}

size_t partial_index(const f16_dims & d, int64_t m, int64_t n, int64_t s) {
    const int64_t t = (n / out_cols) * d.m_tiles + m / out_rows;
    return (size_t) (((t * n_slices + s) * out_rows + m % out_rows) * out_cols + n % out_cols);
}

// the rows a pass takes: the stage holds m_block rows of the slice's activations and one weight tile's slice
int64_t pass_rows(const f16_dims & d, size_t stage_bytes) {
    const int64_t room = ((int64_t) stage_bytes - out_cols * d.ks * 2) / (d.ks * 2);
    return std::min(d.m_pad, room & ~(out_rows - 1));
}

// slice s of weight tile nt (32 rows), [32][ks], from the q4_0 32x256 bytes as stored: fp16(d (q - 8)), the product
// exact in float, so rounded once, as fp16 of ggml's own dequantization. wf (if not null) gets the same values as floats
void stage_weights(const uint8_t * w, int64_t k, int64_t s, int64_t nt, uint16_t * w16, float * wf) {
    const int64_t   ks   = k / n_slices;
    const uint8_t * tile = w + nt * (k / (int64_t) spacemit_ime::q4_0_k_block) * q4_0_tile_blk;
    for (int64_t k0 = s * ks; k0 < (s + 1) * ks; k0 += 32) {
        const uint8_t * blk = tile + (k0 / (int64_t) spacemit_ime::q4_0_k_block) * q4_0_tile_blk +
                              (k0 % (int64_t) spacemit_ime::q4_0_k_block) / 32 * q4_0_sub_blk;
        for (int64_t r = 0; r < out_cols; r++) {
            ggml_fp16_t dh;
            std::memcpy(&dh, blk + 2 * r, sizeof(dh));
            const float     d   = GGML_FP16_TO_FP32(dh);
            const uint8_t * qs  = blk + 2 * out_cols + 16 * r;  // value e in byte e / 2, low nibble first
            const int64_t   at  = r * ks + (k0 - s * ks);
            for (int64_t e = 0; e < 32; e++) {
                const int         q = e % 2 == 0 ? qs[e / 2] & 0x0F : qs[e / 2] >> 4;
                const ggml_fp16_t h = GGML_FP32_TO_FP16(d * (float) (q - 8));
                w16[at + e]         = h;
                if (wf != nullptr) {
                    wf[at + e] = GGML_FP16_TO_FP32(h);
                }
            }
        }
    }
}

// one partial sum: 8 lanes, lane j the k = j mod 8, then the lanes in a fixed tree. a and b hold fp16 values, so each
// product is exact in float
SPACEMIT_SCALAR_REF float dot_f16(const float * a, const float * b, int64_t n) {
    float l[8] = {};
    for (int64_t k = 0; k < n; k += 8) {
        for (int j = 0; j < 8; j++) {
            l[j] += a[k + j] * b[k + j];
        }
    }
    return ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
}

bool overflow(const ggml_tensor * node, const char * what, float value, bool record) {
    GGML_LOG_ERROR("mul_mat.f16: overflow in %s (weight %s): %s %g exceeds fp16\n", node->name, node->src[0]->name,
                   what, (double) value);
    if (record) {
        f16_stats &                 st = stats();
        std::lock_guard<std::mutex> lock(st.mutex);
        st.overflows++;
    }
    return false;
}

}  // namespace

uint32_t spacemit_dense_q4_0_threshold() {
    static std::atomic<int>      value{ 0 };
    static std::atomic<uint32_t> seen{ 0 };
    return (uint32_t) spacemit_cached_switch(value, seen, parse_dense_q4_0);
}

uint32_t spacemit_f16_sum_f32() {
    static std::atomic<int>      value{ 0 };
    static std::atomic<uint32_t> seen{ 0 };
    return (uint32_t) spacemit_cached_switch(value, seen, parse_f16_sum);
}

uint32_t spacemit_f16_stats_on() {
    static std::atomic<int>      value{ 0 };
    static std::atomic<uint32_t> seen{ 0 };
    return (uint32_t) spacemit_cached_switch(value, seen, parse_f16_stats);
}

// the same answer for both steps, and for any F16_SUM setting; never TCM or the tile count
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
    const f16_dims d = dims_of(node);
    return partials_bytes(d, spacemit_f16_sum_f32() != 0) + scratch_tiles * scratch_bytes(d);
}

// tile ith takes slices ith, ith + nth, ...; each slice in passes of m_block rows: the pass's activations cast into the
// stage, then for every weight tile in turn its slice dequantized into the stage, multiplied and the partials stored
bool spacemit_mul_mat_f16_products(const spacemit_tile * tile_p, ggml_tensor * node) {
    const spacemit_tile & tile = *tile_p;
    if (tile.ith >= n_slices) {
        return true;
    }
    const f16_dims      d       = dims_of(node);
    const bool          f32_sum = spacemit_f16_sum_f32() != 0;
    const bool          record  = spacemit_f16_stats_on() != 0;
    const ggml_tensor * w       = node->src[0];
    const uint8_t *     wb      = static_cast<const uint8_t *>(w->data);
    const float *       x       = static_cast<const float *>(node->src[1]->data);
    uint8_t *           ws      = static_cast<uint8_t *>(tile.workspace);
    uint16_t *          p16     = reinterpret_cast<uint16_t *>(ws);
    float *             p32     = reinterpret_cast<float *>(ws);
    uint8_t *           scratch = ws + partials_bytes(d, f32_sum) + tile.ith * scratch_bytes(d);
    const size_t        stage   = tile.tcm != nullptr ? tile.tcm_size : stage_default;
    const int64_t       m_block = pass_rows(d, stage);
    GGML_ASSERT(m_block >= out_rows);
    uint16_t * a16 = reinterpret_cast<uint16_t *>(scratch);
    uint16_t * w16 = a16 + m_block * d.ks;
    float *    af  = reinterpret_cast<float *>(scratch + stage_f16_bytes(d));
    float *    wf  = af + m_block * d.ks;

    if (record && tile.ith == 0) {
        f16_stats &                 st = stats();
        std::lock_guard<std::mutex> lock(st.mutex);
        st.matmuls++;
        st.shapes[{ d.k, d.n }]++;
    }
    float max_act = 0.0f, max_sum = 0.0f;
    for (int64_t s = tile.ith; s < n_slices; s += tile.nth) {
        const int64_t k0 = s * d.ks;
        for (int64_t m0 = 0; m0 < d.m; m0 += m_block) {
            const int64_t rows = std::min(m_block, d.m - m0);
            for (int64_t r = 0; r < rows; r++) {
                const float * xr = x + (m0 + r) * d.k + k0;
                for (int64_t i = 0; i < d.ks; i++) {
                    const ggml_fp16_t h  = GGML_FP32_TO_FP16(xr[i]);
                    const float       hv = GGML_FP16_TO_FP32(h);
                    if (std::isfinite(xr[i])) {
                        if (std::isinf(hv)) {
                            return overflow(node, "activation", xr[i], record);
                        }
                        max_act = std::max(max_act, std::fabs(xr[i]));
                    }
                    a16[r * d.ks + i] = h;
                    af[r * d.ks + i]  = hv;
                }
            }
            for (int64_t nt = 0; nt < d.n_tiles; nt++) {
                stage_weights(wb, d.k, s, nt, w16, wf);
                for (int64_t r = 0; r < rows; r++) {
                    for (int64_t c = 0; c < out_cols; c++) {
                        const float  p  = dot_f16(af + r * d.ks, wf + c * d.ks, d.ks);
                        const size_t at = partial_index(d, m0 + r, nt * out_cols + c, s);
                        if (std::isfinite(p)) {
                            max_sum = std::max(max_sum, std::fabs(p));
                        }
                        if (f32_sum) {
                            p32[at] = p;
                            continue;
                        }
                        const ggml_fp16_t h = GGML_FP32_TO_FP16(p);
                        if (std::isfinite(p) && std::isinf(GGML_FP16_TO_FP32(h))) {
                            return overflow(node, "partial sum", p, record);
                        }
                        p16[at] = h;
                    }
                }
            }
        }
    }
    if (record) {
        f16_stats &                 st = stats();
        std::lock_guard<std::mutex> lock(st.mutex);
        if (st.act_weight.empty() || max_act > st.max_act) {
            st.max_act    = max_act;
            st.act_weight = w->name;
        }
        if (st.sum_weight.empty() || max_sum > st.max_sum) {
            st.max_sum    = max_sum;
            st.sum_weight = w->name;
        }
    }
    return true;
}

// the output's 16 x 32 tiles in chunks of 4 consecutive ones, chunk c to tile c % nth; each output the 8 partials
// added in slice order in float. Rows from M on are not written
bool spacemit_mul_mat_f16_sum(const spacemit_tile * tile_p, ggml_tensor * node) {
    const spacemit_tile & tile    = *tile_p;
    const f16_dims        d       = dims_of(node);
    const bool            f32_sum = spacemit_f16_sum_f32() != 0;
    const uint16_t *      p16     = static_cast<const uint16_t *>(tile.workspace);
    const float *         p32     = static_cast<const float *>(tile.workspace);
    float *               out     = static_cast<float *>(node->data);
    const int64_t         n_out   = d.n_tiles * d.m_tiles;
    const int64_t         chunks  = (n_out + sum_chunk - 1) / sum_chunk;
    const auto partial = [&](size_t at) { return f32_sum ? p32[at] : GGML_FP16_TO_FP32(p16[at]); };
    for (int64_t c = tile.ith; c < chunks; c += tile.nth) {
        for (int64_t t = c * sum_chunk; t < std::min(n_out, (c + 1) * sum_chunk); t++) {
            const int64_t n0 = t / d.m_tiles * out_cols;
            const int64_t m0 = t % d.m_tiles * out_rows;
            for (int64_t m = m0; m < std::min(d.m, m0 + out_rows); m++) {
                for (int64_t n = n0; n < n0 + out_cols; n++) {
                    float acc = partial(partial_index(d, m, n, 0));
                    for (int64_t s = 1; s < n_slices; s++) {
                        acc = acc + partial(partial_index(d, m, n, s));
                    }
                    out[m * d.n + n] = acc;
                }
            }
        }
    }
    return true;
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
            std::fprintf(f, "mul_mat.f16: matmuls %llu, overflows %llu, sum %s, threshold %u\n",
                         (unsigned long long) st.matmuls, (unsigned long long) st.overflows,
                         spacemit_f16_sum_f32() != 0 ? "f32" : "f16", spacemit_dense_q4_0_threshold());
            std::fprintf(f, "mul_mat.f16: largest activation %.6g in %s\n", (double) st.max_act,
                         st.act_weight.empty() ? "-" : st.act_weight.c_str());
            std::fprintf(f, "mul_mat.f16: largest partial sum %.6g in %s\n", (double) st.max_sum,
                         st.sum_weight.empty() ? "-" : st.sum_weight.c_str());
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

void spacemit_mul_mat_f16_stage_weights(const ggml_tensor * w, int64_t slice, int64_t col_tile, uint16_t * out) {
    stage_weights(static_cast<const uint8_t *>(w->data), w->ne[0], slice, col_tile, out, nullptr);
}
