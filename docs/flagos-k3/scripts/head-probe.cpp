// head-probe: two numbers for the model's output layer (token_embd, Q6_K in the GGUF, read once per generated token).
// Part A: is there DRAM read bandwidth beyond what the 8 A100 tiles take? A 1 GiB read-only sum on the A100 stream
//         (1, 2, 4, 8 tiles), on spine-runtime's spacemit-k3-x100 backend (4 tiles, 4 tiles + the caller), on
//         std::thread (4, 8), and on both backends at once.
// Part B: can a 1-row matmul read the file's Q6_K layout directly (26.25 bytes per 32 weights) instead of the Q8_0
//         copy made at load (34 bytes per 32)? A scalar reference, an RVV kernel for VLEN 1024, compute-only and
//         streaming rates on the A100 stream, and the read-only sum of the Q8_0 head's bytes for comparison.
// Build with -ffp-contract=off: the checks compare the kernel's float accumulation with the reference's bit for bit,
// so neither may be fused into an FMA by chance.
// Built and run by head-probe.sh. One fact per line, "<part> <what>: key=value ..."; a failed step prints the failure
// and the probe goes on. The lines that start with "head-probe:" are the summary; "check <name>:" lines are PASS/FAIL.
// rc 0, or 1 when a check failed, or 3 when a stream grant was left waiting.
//
// usage: head-probe [part ...] [--small]   parts: a b all (default all; "a,b" also works)
// env:   PROBE_MIB (1024: part A buffer), REPS (5), PROBE_GRANT_S (20: seconds to wait for one stream before it is
//        reported blocked; 0 creates it on the calling thread with no wait limit), PROBE_ROWS1 (248320),
//        PROBE_ROWS2 (151936), PROBE_COLS (2560), PROBE_COMPUTE_MS (50), PROBE_SAMPLE (512), PROBE_SEED (1)
//        --small: PROBE_MIB 64, 4096 and 2048 rows, REPS 3, PROBE_COMPUTE_MS 5 (a host run of seconds)
//
// The Q6_K kernel's scheme, in one place: a group of 32 weights per vector operation (e8 at LMUL 1/4 and e32 at
// LMUL 1 with VLEN 1024), the four groups of a half block taken straight from ql and qh as
// ggml-quants.c dequantize_row_q6_K reads them, so no lane crossing (no vrgather, no vslideup) is needed and the
// weight of lane i always meets its own activation. The group's two scales (one per 16 weights) come from a vmerge of
// two scalars, the product scale * (q - 32) goes to int16, and a widening multiply-accumulate with the sign-extended
// activations gives int32 lanes. Two epilogues are measured: the block's lanes reduced in integers (exact, the
// streaming pass uses it) and the lanes converted to float and accumulated over the row's blocks (one reduce a row).

#if defined(HEAD_PROBE_SPERT)
#    include <spert.hpp>
#endif

#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#ifdef __linux__
#    include <sched.h>
#endif

#if defined(__riscv_v_intrinsic) || defined(HEAD_PROBE_RVV_EMU)
#    define HP_RVV 1
#    if !defined(HEAD_PROBE_RVV_EMU)
#        include <riscv_vector.h>
#    endif
#else
#    define HP_RVV 0
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;

#if defined(HEAD_PROBE_SPERT)
static const char * k_x100_name = "spacemit-k3-x100";
#endif

// ---- small helpers ----

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static int64_t now_ns() {
    return (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count();
}

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    if (!v || !*v) {
        return def;
    }
    const int n = atoi(v);
    return n >= 0 ? n : def;
}

static int cur_cpu() {
#ifdef __linux__
    return sched_getcpu();
#else
    return -1;
#endif
}

static double cpu_seconds() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

static std::string proc_threads() {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) {
        return "n/a";
    }
    std::string n = "?";
    char        line[256];
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "Threads:", 8) == 0) {
            n = std::to_string(atoi(line + 8));
            break;
        }
    }
    fclose(f);
    return n;
}

static const char * proc_tcm() {
    FILE * f = fopen("/proc/self/maps", "r");
    if (!f) {
        return "n/a";
    }
    const char * r = "no";
    char         line[4096];
    while (fgets(line, sizeof line, f)) {
        if (strstr(line, "/dev/tcm")) {
            r = "yes";
            break;
        }
    }
    fclose(f);
    return r;
}

static void maps(const char * label) {
    printf("maps %s: tcm=%s threads=%s cpu_s=%.3f\n", label, proc_tcm(), proc_threads().c_str(), cpu_seconds());
}

template <typename T> static std::string join(const std::vector<T> & v) {
    std::string s;
    for (size_t i = 0; i < v.size(); i++) {
        s += (i ? "," : "") + std::to_string(v[i]);
    }
    return s.empty() ? "-" : s;
}

static std::string fmt_list(const std::vector<double> & v, const char * f) {
    std::string s;
    char        b[64];
    for (size_t i = 0; i < v.size(); i++) {
        snprintf(b, sizeof b, f, v[i]);
        s += (i ? "," : "") + std::string(b);
    }
    return s.empty() ? "-" : s;
}

static double median(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double vmin(const std::vector<double> & v) {
    return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end());
}

static double vmax(const std::vector<double> & v) {
    return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end());
}

static double gb_s(double bytes, double ms) {
    return ms > 0.0 ? bytes / (ms / 1000.0) / 1e9 : 0.0;
}

static void part_range(size_t n, uint32_t np, uint32_t i, size_t & b, size_t & e) {
    b = n * i / np;
    e = n * (i + 1) / np;
}

static void part_rows(int64_t n, uint32_t np, uint32_t i, int64_t & b, int64_t & e) {
    b = n * (int64_t) i / (int64_t) np;
    e = n * (int64_t) (i + 1) / (int64_t) np;
}

// ---- checks ----

static int g_checks = 0;
static int g_fails  = 0;

static void check(const char * name, bool ok, const std::string & detail) {
    g_checks++;
    if (!ok) {
        g_fails++;
    }
    printf("check %s: %s %s\n", name, ok ? "PASS" : "FAIL", detail.c_str());
}

// ---- fp16, local (no ggml) ----

static float fp16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t man  = h & 0x3ffu;
    uint32_t       bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            uint32_t m = man;
            int      e = -1;
            do {
                m <<= 1;
                e++;
            } while (!(m & 0x400u));
            bits = sign | ((uint32_t) (127 - 15 - e) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

// round to nearest even; the probe stores only small positive values, so large ones clamp to the largest finite fp16
static uint16_t f32_to_fp16(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof b);
    const uint32_t sign = (b >> 16) & 0x8000u;
    const int32_t  e    = (int32_t) ((b >> 23) & 0xffu) - 127 + 15;
    uint32_t       m    = b & 0x7fffffu;
    if (((b >> 23) & 0xffu) == 0xffu) {
        return (uint16_t) (sign | 0x7bffu);
    }
    if (e >= 31) {
        return (uint16_t) (sign | 0x7bffu);
    }
    if (e <= 0) {
        if (e < -10) {
            return (uint16_t) sign;
        }
        m |= 0x800000u;
        const int      shift = 14 - e;
        const uint32_t lower = m & ((1u << shift) - 1);
        uint32_t       r     = m >> shift;
        const uint32_t half  = 1u << (shift - 1);
        if (lower > half || (lower == half && (r & 1))) {
            r++;
        }
        return (uint16_t) (sign | r);
    }
    uint32_t       r     = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t lower = m & 0x1fffu;
    if (lower > 0x1000u || (lower == 0x1000u && (r & 1))) {
        r++;
    }
    return (uint16_t) (sign | r);
}

// ---- part A: the read-only sum ----

// the scalar reference: a 64-bit wrapping add of the u64 words in [b, e)
static uint64_t sum_words_ref(const uint64_t * p, size_t b, size_t e) {
    uint64_t s = 0;
    for (size_t i = b; i < e; i++) {
        s += p[i];
    }
    return s;
}

#if HP_RVV
// the e64 intrinsics take the target's own 64-bit word type (unsigned long on riscv64 lp64d), which need not be the
// same type as uint64_t for the compiler
using hp_vu64 = unsigned long;
static_assert(sizeof(hp_vu64) == 8, "the e64 word type is not 8 bytes");

// the read-only sum with RVV at the running core's own VLEN (vsetvlmax), two accumulators, a scalar tail
static uint64_t sum_words(const uint64_t * p, size_t b, size_t e) {
    const size_t vl = __riscv_vsetvlmax_e64m8();
    vuint64m8_t  a0 = __riscv_vmv_v_x_u64m8(0, vl);
    vuint64m8_t  a1 = __riscv_vmv_v_x_u64m8(0, vl);
    size_t       i  = b;
    for (; i + 2 * vl <= e; i += 2 * vl) {
        a0 = __riscv_vadd_vv_u64m8(a0, __riscv_vle64_v_u64m8((const hp_vu64 *) (p + i), vl), vl);
        a1 = __riscv_vadd_vv_u64m8(a1, __riscv_vle64_v_u64m8((const hp_vu64 *) (p + i + vl), vl), vl);
    }
    a0             = __riscv_vadd_vv_u64m8(a0, a1, vl);
    const uint64_t r = __riscv_vmv_x_s_u64m1_u64(
        __riscv_vredsum_vs_u64m8_u64m1(a0, __riscv_vmv_v_x_u64m1(0, 1), vl));
    uint64_t s = r;
    for (; i < e; i++) {
        s += p[i];
    }
    return s;
}
#else
static uint64_t sum_words(const uint64_t * p, size_t b, size_t e) {
    return sum_words_ref(p, b, e);
}
#endif

// ---- part B: the Q6_K layout ----

static const int    k_qk        = 256;  // weights per Q6_K block
static const size_t k_q6k_bytes = 210;  // ql[128] + qh[64] + scales[16] + fp16 d
static const size_t k_q8_bytes  = 34;   // the Q8_0 copy made at load, per 32 weights

struct q6k_block {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint8_t d[2];
};

static_assert(sizeof(q6k_block) == k_q6k_bytes, "q6k_block is not the file's 210 bytes");

static float block_d(const q6k_block * x) {
    uint16_t h;
    memcpy(&h, x->d, sizeof h);
    return fp16_to_f32(h);
}

// y[r] = scale_x * sum over the row's blocks of d_b * (int32 sum over the block of scales[g] * (q - 32) * a), with q
// unpacked as ggml-quants.c dequantize_row_q6_K reads it. bsum, when given, takes the per-block integer sums.
static void q6k_row_ref(const uint8_t * row, const int8_t * act, int64_t nblk, float scale_x, float * y,
                        int64_t * bsum) {
    float acc = 0.0f;
    for (int64_t b = 0; b < nblk; b++) {
        const q6k_block * x = (const q6k_block *) (row + (size_t) b * k_q6k_bytes);
        int32_t           s = 0;
        for (int n = 0; n < 2; n++) {
            const uint8_t * ql = x->ql + n * 64;
            const uint8_t * qh = x->qh + n * 32;
            const int8_t *  sc = x->scales + n * 8;
            const int8_t *  av = act + b * k_qk + n * 128;
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int q1 = (int) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                s += sc[is + 0] * q1 * av[l];
                s += sc[is + 2] * q2 * av[l + 32];
                s += sc[is + 4] * q3 * av[l + 64];
                s += sc[is + 6] * q4 * av[l + 96];
            }
        }
        if (bsum) {
            bsum[b] = s;
        }
        acc += block_d(x) * (float) s;
    }
    *y = scale_x * acc;
}

// the second reference: ggml's own float decode (its index arithmetic replicated here) dotted with the dequantized
// int8 activations
static float q6k_row_decode_dot(const uint8_t * row, const int8_t * act, int64_t nblk, float scale_x) {
    float dot = 0.0f;
    for (int64_t b = 0; b < nblk; b++) {
        const q6k_block * x  = (const q6k_block *) (row + (size_t) b * k_q6k_bytes);
        const float       d  = block_d(x);
        const int8_t *    av = act + b * k_qk;
        for (int n = 0; n < 2; n++) {
            const uint8_t * ql = x->ql + n * 64;
            const uint8_t * qh = x->qh + n * 32;
            const int8_t *  sc = x->scales + n * 8;
            const int       o  = n * 128;
            for (int l = 0; l < 32; l++) {
                const int   is = l / 16;
                const float q1 = (float) ((int) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                const float q2 = (float) ((int) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                const float q3 = (float) ((int) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                const float q4 = (float) ((int) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                dot += (d * (float) sc[is + 0] * q1) * (scale_x * (float) av[o + l]);
                dot += (d * (float) sc[is + 2] * q2) * (scale_x * (float) av[o + l + 32]);
                dot += (d * (float) sc[is + 4] * q3) * (scale_x * (float) av[o + l + 64]);
                dot += (d * (float) sc[is + 6] * q4) * (scale_x * (float) av[o + l + 96]);
            }
        }
    }
    return dot;
}

#if HP_RVV
// one group of 32 weights into the block's int32 lanes: q6 the 6-bit values in the file's order, s_lo and s_hi the
// group's two scales (one per 16 weights), av the group's 32 activations
static inline vint32m1_t q6k_group(vint32m1_t acc, vuint8mf4_t q6, int8_t s_lo, int8_t s_hi, vbool32_t hi,
                                   const int8_t * av) {
    const vint8mf4_t  q  = __riscv_vsub_vx_i8mf4(__riscv_vreinterpret_v_u8mf4_i8mf4(q6), 32, 32);
    const vint8mf4_t  s  = __riscv_vmerge_vxm_i8mf4(__riscv_vmv_v_x_i8mf4(s_lo, 32), s_hi, hi, 32);
    const vint16mf2_t p  = __riscv_vwmul_vv_i16mf2(q, s, 32);
    const vint16mf2_t aw = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(av, 32), 32);
    return __riscv_vwmacc_vv_i32m1(acc, p, aw, 32);
}

// VAR 0: the block's lanes reduced in integers (exact). VAR 1: the lanes converted to float, scaled by d and
// accumulated over the row's blocks, reduced once a row. VAR 2: VAR 0 and the per-block integer sums written to bsum
// (one row at a time, for the checks). The caller checks vlenb == 128.
template <int VAR>
static void q6k_rows_rvv(const uint8_t * tab, const int8_t * act, float * y, int64_t nblk, float scale_x, int64_t r0,
                         int64_t r1, int64_t * bsum = nullptr) {
    const size_t    vl   = 32;
    const vbool32_t hi   = __riscv_vmsgeu_vx_u32m1_b32(__riscv_vid_v_u32m1(vl), 16, vl);  // lanes 16-31
    const size_t    strd = (size_t) nblk * k_q6k_bytes;
    for (int64_t r = r0; r < r1; r++) {
        const uint8_t * row  = tab + (size_t) r * strd;
        float           acc  = 0.0f;
        vfloat32m1_t    facc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        for (int64_t b = 0; b < nblk; b++) {
            const q6k_block * x  = (const q6k_block *) (row + (size_t) b * k_q6k_bytes);
            vint32m1_t        ia = __riscv_vmv_v_x_i32m1(0, vl);
            for (int n = 0; n < 2; n++) {
                const uint8_t * ql = x->ql + n * 64;
                const uint8_t * qh = x->qh + n * 32;
                const int8_t *  sc = x->scales + n * 8;
                const int8_t *  av = act + b * k_qk + n * 128;

                const vuint8mf4_t l0 = __riscv_vle8_v_u8mf4(ql, vl);
                const vuint8mf4_t l1 = __riscv_vle8_v_u8mf4(ql + 32, vl);
                const vuint8mf4_t h  = __riscv_vle8_v_u8mf4(qh, vl);
                // the high 2 bits at bits 4-5: qh bits 0-1, 2-3, 4-5 and 6-7 for the four groups
                const vuint8mf4_t h0 = __riscv_vand_vx_u8mf4(__riscv_vsll_vx_u8mf4(h, 4, vl), 0x30, vl);
                const vuint8mf4_t h1 = __riscv_vand_vx_u8mf4(__riscv_vsll_vx_u8mf4(h, 2, vl), 0x30, vl);
                const vuint8mf4_t h2 = __riscv_vand_vx_u8mf4(h, 0x30, vl);
                const vuint8mf4_t h3 = __riscv_vand_vx_u8mf4(__riscv_vsrl_vx_u8mf4(h, 2, vl), 0x30, vl);

                ia = q6k_group(ia, __riscv_vor_vv_u8mf4(__riscv_vand_vx_u8mf4(l0, 0xF, vl), h0, vl), sc[0], sc[1], hi,
                               av);
                ia = q6k_group(ia, __riscv_vor_vv_u8mf4(__riscv_vand_vx_u8mf4(l1, 0xF, vl), h1, vl), sc[2], sc[3], hi,
                               av + 32);
                ia = q6k_group(ia, __riscv_vor_vv_u8mf4(__riscv_vsrl_vx_u8mf4(l0, 4, vl), h2, vl), sc[4], sc[5], hi,
                               av + 64);
                ia = q6k_group(ia, __riscv_vor_vv_u8mf4(__riscv_vsrl_vx_u8mf4(l1, 4, vl), h3, vl), sc[6], sc[7], hi,
                               av + 96);
            }
            const float d = block_d(x);
            if (VAR == 1) {
                facc = __riscv_vfmacc_vf_f32m1(facc, d, __riscv_vfcvt_f_x_v_f32m1(ia, vl), vl);
            } else {
                const int32_t s = __riscv_vmv_x_s_i32m1_i32(
                    __riscv_vredsum_vs_i32m1_i32m1(ia, __riscv_vmv_v_x_i32m1(0, 1), vl));
                acc += d * (float) s;
                if (VAR == 2) {
                    bsum[b] = s;
                }
            }
        }
        if (VAR != 1) {
            y[r] = scale_x * acc;
        } else {
            y[r] = scale_x * __riscv_vfmv_f_s_f32m1_f32(
                                  __riscv_vfredusum_vs_f32m1_f32m1(facc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl));
        }
    }
}
#endif

// the same scheme in plain C, one lane at a time: the host stand-in for q6k_rows_rvv and, on the board, a check of the
// scheme's index arithmetic against q6k_row_ref
template <int VAR>
static void q6k_rows_scheme(const uint8_t * tab, const int8_t * act, float * y, int64_t nblk, float scale_x,
                            int64_t r0, int64_t r1, int64_t * bsum = nullptr) {
    const size_t strd = (size_t) nblk * k_q6k_bytes;
    for (int64_t r = r0; r < r1; r++) {
        const uint8_t * row = tab + (size_t) r * strd;
        float           acc = 0.0f;
        float           fl[32];
        for (int i = 0; i < 32; i++) {
            fl[i] = 0.0f;
        }
        for (int64_t b = 0; b < nblk; b++) {
            const q6k_block * x = (const q6k_block *) (row + (size_t) b * k_q6k_bytes);
            int32_t           ia[32];
            for (int i = 0; i < 32; i++) {
                ia[i] = 0;
            }
            for (int n = 0; n < 2; n++) {
                const uint8_t * ql = x->ql + n * 64;
                const uint8_t * qh = x->qh + n * 32;
                const int8_t *  sc = x->scales + n * 8;
                const int8_t *  av = act + b * k_qk + n * 128;
                for (int l = 0; l < 32; l++) {
                    const uint8_t h  = qh[l];
                    const uint8_t h0 = (uint8_t) ((uint8_t) (h << 4) & 0x30);
                    const uint8_t h1 = (uint8_t) ((uint8_t) (h << 2) & 0x30);
                    const uint8_t h2 = (uint8_t) (h & 0x30);
                    const uint8_t h3 = (uint8_t) ((uint8_t) (h >> 2) & 0x30);
                    const uint8_t g[4] = { (uint8_t) ((ql[l] & 0xF) | h0), (uint8_t) ((ql[l + 32] & 0xF) | h1),
                                           (uint8_t) ((ql[l] >> 4) | h2), (uint8_t) ((ql[l + 32] >> 4) | h3) };
                    for (int k = 0; k < 4; k++) {
                        const int     q = (int) g[k] - 32;
                        const int8_t  s = sc[2 * k + (l < 16 ? 0 : 1)];
                        const int16_t p = (int16_t) (s * q);
                        ia[l] += (int32_t) p * (int32_t) av[l + 32 * k];
                    }
                }
            }
            const float d = block_d(x);
            if (VAR == 1) {
                for (int i = 0; i < 32; i++) {
                    fl[i] += d * (float) ia[i];
                }
            } else {
                int32_t s = 0;
                for (int i = 0; i < 32; i++) {
                    s += ia[i];
                }
                acc += d * (float) s;
                if (VAR == 2) {
                    bsum[b] = s;
                }
            }
        }
        if (VAR != 1) {
            y[r] = scale_x * acc;
        } else {
            float s = 0.0f;
            for (int i = 0; i < 32; i++) {
                s += fl[i];
            }
            y[r] = scale_x * s;
        }
    }
}

static int hp_vlenb() {
#if HP_RVV
    return (int) __riscv_vlenb();
#else
    return 0;
#endif
}

static void q6k_rows(int variant, const uint8_t * tab, const int8_t * act, float * y, int64_t nblk, float scale_x,
                     int64_t r0, int64_t r1, int64_t * bsum = nullptr) {
#if HP_RVV
    if (variant == 1) {
        q6k_rows_rvv<1>(tab, act, y, nblk, scale_x, r0, r1);
    } else if (variant == 2) {
        q6k_rows_rvv<2>(tab, act, y, nblk, scale_x, r0, r1, bsum);
    } else {
        q6k_rows_rvv<0>(tab, act, y, nblk, scale_x, r0, r1);
    }
#else
    if (variant == 1) {
        q6k_rows_scheme<1>(tab, act, y, nblk, scale_x, r0, r1);
    } else if (variant == 2) {
        q6k_rows_scheme<2>(tab, act, y, nblk, scale_x, r0, r1, bsum);
    } else {
        q6k_rows_scheme<0>(tab, act, y, nblk, scale_x, r0, r1);
    }
#endif
}

// ---- jobs and their parts ----

struct parts_out {
    std::vector<uint64_t> sums;
    std::vector<int64_t>  t0, t1, rows;
    std::vector<int>      cpu0, cpu1, vlenb;

    void init(uint32_t n) {
        sums.assign(n, 0);
        t0.assign(n, 0);
        t1.assign(n, 0);
        rows.assign(n, 0);
        cpu0.assign(n, -2);
        cpu1.assign(n, -2);
        vlenb.assign(n, 0);
    }
};

// the span the parts covered: from the first start to the last end
static double parts_span_ms(const parts_out & o) {
    int64_t lo = 0, hi = 0;
    bool    any = false;
    for (size_t i = 0; i < o.t0.size(); i++) {
        if (o.t1[i] == 0) {
            continue;
        }
        if (!any) {
            lo  = o.t0[i];
            hi  = o.t1[i];
            any = true;
        } else {
            lo = std::min(lo, o.t0[i]);
            hi = std::max(hi, o.t1[i]);
        }
    }
    return any ? (double) (hi - lo) / 1e6 : 0.0;
}

struct sum_job {
    const uint64_t * p      = nullptr;
    size_t           off    = 0;
    size_t           n      = 0;
    uint32_t         nparts = 1;
    uint32_t         first  = 0;
    parts_out *      out    = nullptr;
};

struct q6k_job {
    const uint8_t * tab     = nullptr;
    const int8_t *  act     = nullptr;
    float *         y       = nullptr;
    int64_t         nrows   = 0;
    int64_t         nblk    = 0;
    float           scale_x = 0.0f;
    int             variant = 0;
    int             reps    = 0;  // > 0: compute-only, each part loops over its own first rows_cache rows
    int64_t         rows_cache = 128;
    uint32_t        nparts  = 1;
    uint32_t        first   = 0;
    parts_out *     out     = nullptr;
    int64_t         row_lo  = -1;  // >= 0: part 0 runs exactly [row_lo, row_hi), for the checks
    int64_t         row_hi  = -1;
    int64_t *       bsum    = nullptr;  // variant 2 only
};

static void sum_part(sum_job * j, uint32_t idx) {
    if (idx >= j->nparts) {
        return;
    }
    size_t b, e;
    part_range(j->n, j->nparts, idx, b, e);
    j->out->cpu0[idx] = cur_cpu();
    j->out->t0[idx]   = now_ns();
    j->out->sums[idx] = sum_words(j->p, j->off + b, j->off + e);
    j->out->t1[idx]   = now_ns();
    j->out->cpu1[idx] = cur_cpu();
}

static void q6k_part(q6k_job * j, uint32_t idx) {
    if (idx >= j->nparts) {
        return;
    }
    int64_t r0, r1;
    part_rows(j->nrows, j->nparts, idx, r0, r1);
    j->out->cpu0[idx]  = cur_cpu();
    j->out->vlenb[idx] = hp_vlenb();
    j->out->t0[idx]    = now_ns();
#if HP_RVV
    if (j->out->vlenb[idx] != 128) {
        j->out->rows[idx] = -1;  // the kernel needs VLEN 1024; the caller reports the skip
        j->out->t1[idx]   = now_ns();
        return;
    }
#endif
    if (j->row_lo >= 0) {
        if (idx == 0) {
            q6k_rows(j->variant, j->tab, j->act, j->y, j->nblk, j->scale_x, j->row_lo, j->row_hi, j->bsum);
        }
        j->out->t1[idx]   = now_ns();
        j->out->rows[idx] = j->row_hi - j->row_lo;
        return;
    }
    int64_t rows = 0;
    if (j->reps > 0) {
        const int64_t n = std::min<int64_t>(j->rows_cache, r1 - r0);
        for (int rep = 0; rep < j->reps; rep++) {
            q6k_rows(j->variant, j->tab, j->act, j->y, j->nblk, j->scale_x, r0, r0 + n);
            rows += n;
        }
    } else {
        q6k_rows(j->variant, j->tab, j->act, j->y, j->nblk, j->scale_x, r0, r1);
        rows = r1 - r0;
    }
    j->out->t1[idx]   = now_ns();
    j->out->cpu1[idx] = cur_cpu();
    j->out->rows[idx] = rows;
}

// ---- backends ----

enum bk { BK_A100, BK_X100, BK_THREADS };

static const char * bk_name(bk k) {
    return k == BK_A100 ? "a100" : (k == BK_X100 ? "x100" : "threads");
}

static int g_grant_s   = 20;
static int g_abandoned = 0;

#if defined(HEAD_PROBE_SPERT)
static spert::Stream * g_a100    = nullptr;
static spert::Stream * g_x100    = nullptr;
static bool            g_subgrid = true;  // a grid smaller than the stream's core_count works

struct made_stream {
    spert::Stream * s       = nullptr;
    double          ms      = 0.0;
    bool            blocked = false;
};

// a stream constructor may wait for its grant; wait at most g_grant_s, then leave it to its thread and go on
static made_stream make_stream(const std::function<spert::Stream *()> & make) {
    made_stream r;
    if (g_grant_s == 0) {
        const auto t0 = clk::now();
        r.s           = make();
        r.ms          = ms_since(t0);
        return r;
    }
    struct shared_state {
        std::mutex              m;
        std::condition_variable cv;
        bool                    done      = false;
        bool                    abandoned = false;
        spert::Stream *         s         = nullptr;
        double                  ms        = 0.0;
    };
    auto        sh = std::make_shared<shared_state>();
    std::thread t([sh, make] {
        const auto      t0 = clk::now();
        spert::Stream * s  = make();
        const double    ms = ms_since(t0);
        std::lock_guard<std::mutex> lk(sh->m);
        if (sh->abandoned) {
            return;  // leaked on purpose: its grant may be live
        }
        sh->s    = s;
        sh->ms   = ms;
        sh->done = true;
        sh->cv.notify_one();
    });
    std::unique_lock<std::mutex> lk(sh->m);
    if (sh->cv.wait_for(lk, std::chrono::seconds(g_grant_s), [&] { return sh->done; })) {
        r.s  = sh->s;
        r.ms = sh->ms;
        lk.unlock();
        t.join();
    } else {
        sh->abandoned = true;
        r.blocked     = true;
        r.ms          = g_grant_s * 1000.0;
        lk.unlock();
        t.detach();
        g_abandoned++;
    }
    return r;
}

static void sum_tile(spert::Context * ctx, sum_job * j) {
    sum_part(j, j->first + ctx->program_id(0));
}

static void q6k_tile(spert::Context * ctx, q6k_job * j) {
    q6k_part(j, j->first + ctx->program_id(0));
}

static void empty_tile(spert::Context *) {}

static spert::Stream * stream_of(bk k) {
    return k == BK_A100 ? g_a100 : (k == BK_X100 ? g_x100 : nullptr);
}
#endif

static uint32_t g_a100_cores = 0;
static uint32_t g_x100_cores = 0;

// the grid for a launch that works on nparts tiles: where a grid below the stream's core_count is refused, the whole
// grid is launched and the tiles above nparts do nothing
static uint32_t a100_grid(uint32_t nparts) {
#if defined(HEAD_PROBE_SPERT)
    if (g_a100 && nparts < g_a100_cores && !g_subgrid) {
        return g_a100_cores;
    }
#endif
    return nparts;
}

// started work: either one spert launch or a set of threads
struct pending {
    bk       kind     = BK_THREADS;
    uint32_t grid     = 0;
    bool     ok       = false;
#if defined(HEAD_PROBE_SPERT)
    std::vector<spert::Future> fut;
#endif
    std::vector<std::thread> th;
};

template <typename JOB, void (*PART)(JOB *, uint32_t)>
static void start_threads(pending & p, uint32_t grid, JOB * j) {
    p.th.reserve(grid);
    for (uint32_t i = 0; i < grid; i++) {
        const uint32_t idx = j->first + i;
        p.th.emplace_back([j, idx] { PART(j, idx); });
    }
    p.ok = true;
}

static pending start_sum(bk k, uint32_t grid, sum_job * j) {
    pending p;
    p.kind = k;
    p.grid = grid;
#if defined(HEAD_PROBE_SPERT)
    spert::Stream * s = stream_of(k);
    if (s) {
        p.fut.push_back(s->launch(spert::Grid{ grid }, sum_tile, j));
        p.ok = p.fut.back().valid();
        if (!p.ok) {
            printf("launch %s: the sum launch returned an invalid future (grid=%u)\n", bk_name(k), grid);
        }
        return p;
    }
    if (k != BK_THREADS) {
        printf("launch %s: no stream, the sum runs on %u threads instead\n", bk_name(k), grid);
    }
#endif
    start_threads<sum_job, sum_part>(p, grid, j);
    return p;
}

static pending start_q6k(bk k, uint32_t grid, q6k_job * j) {
    pending p;
    p.kind = k;
    p.grid = grid;
#if defined(HEAD_PROBE_SPERT)
    spert::Stream * s = stream_of(k);
    if (s) {
        p.fut.push_back(s->launch(spert::Grid{ grid }, q6k_tile, j));
        p.ok = p.fut.back().valid();
        if (!p.ok) {
            printf("launch %s: the q6_K launch returned an invalid future (grid=%u)\n", bk_name(k), grid);
        }
        return p;
    }
    if (k != BK_THREADS) {
        printf("launch %s: no stream, the q6_K rows run on %u threads instead\n", bk_name(k), grid);
    }
#endif
    start_threads<q6k_job, q6k_part>(p, grid, j);
    return p;
}

static bool finish(pending & p) {
    bool ok = p.ok;
#if defined(HEAD_PROBE_SPERT)
    for (spert::Future & f : p.fut) {
        if (!f.valid()) {
            ok = false;
            continue;
        }
        const spert::Status st = f.sync();
        if (st != spert::Status::Ok) {
            printf("launch %s: sync status %s\n", bk_name(p.kind), spert::to_string(st));
            ok = false;
        }
    }
#endif
    for (std::thread & t : p.th) {
        t.join();
    }
    return ok;
}

// ---- the buffer ----

static uint8_t * map_buf(size_t n, const char *& advice) {
    void * p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        advice = "-";
        return nullptr;
    }
#ifdef MADV_HUGEPAGE
    advice = madvise(p, n, MADV_HUGEPAGE) == 0 ? "ok" : "failed";
#else
    advice = "unavailable";
#endif
    return static_cast<uint8_t *>(p);
}

static uint64_t g_rng = 1;

static uint64_t rng_next() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}

// ---- settings ----

static size_t  g_mib        = 1024;
static int     g_reps       = 5;
static int64_t g_rows1      = 248320;
static int64_t g_rows2      = 151936;
static int64_t g_cols       = 2560;
static int     g_compute_ms = 50;
static int64_t g_sample     = 512;

// part A's figures, kept for the summary
static double g_a100_solo[4] = { 0, 0, 0, 0 };  // 1, 2, 4, 8 tiles
static double g_x100_solo[2] = { 0, 0 };        // 4 tiles, 4 + caller
static double g_thr_solo[2]  = { 0, 0 };        // 4, 8 threads
static std::vector<std::string> g_conc_lines;
static std::vector<std::string> g_b_lines;
static std::string              g_b_compute;

// ---- part A ----

struct sum_meas {
    std::vector<double> ms;
    double              med   = 0.0;
    uint64_t            sum   = 0;
    bool                equal = false;
    bool                ran   = false;
};

static uint64_t total_sum(const parts_out & o, uint32_t nparts) {
    uint64_t s = 0;
    for (uint32_t i = 0; i < nparts; i++) {
        s += o.sums[i];
    }
    return s;
}

// one configuration of the read-only sum over the whole buffer, reps passes, median reported
static sum_meas run_sum_cfg(const char * label, bk k, uint32_t grid, uint32_t nparts, bool caller, const uint64_t * p,
                            size_t words, uint64_t ref, int reps) {
    sum_meas  m;
    parts_out out;
    sum_job   j;
    j.p      = p;
    j.off    = 0;
    j.n      = words;
    j.nparts = nparts;
    j.first  = caller ? 1 : 0;
    j.out    = &out;
    bool equal = true;
    for (int r = 0; r < reps; r++) {
        out.init(nparts);
        const auto t0 = clk::now();
        pending    pd = start_sum(k, grid, &j);
        if (caller) {
            sum_part(&j, 0);
        }
        const bool ok = finish(pd);
        const double ms = ms_since(t0);
        if (!ok) {
            printf("a %s: the pass failed\n", label);
            return m;
        }
        m.ms.push_back(ms);
        m.sum = total_sum(out, nparts);
        equal = equal && m.sum == ref;
    }
    m.ran   = true;
    m.med   = median(m.ms);
    m.equal = equal;
    printf("a %s: backend=%s grid=%u parts=%u median_ms=%.2f min_ms=%.2f max_ms=%.2f gb_s=%.2f (%.2f-%.2f) ms=%s "
           "checksum=0x%016llx%s cpu_start=%s cpu_end=%s\n",
           label, bk_name(k), grid, nparts, m.med, vmin(m.ms), vmax(m.ms), gb_s((double) words * 8, m.med),
           gb_s((double) words * 8, vmax(m.ms)), gb_s((double) words * 8, vmin(m.ms)), fmt_list(m.ms, "%.2f").c_str(),
           (unsigned long long) m.sum, m.equal ? "" : " MISMATCH", join(out.cpu0).c_str(), join(out.cpu1).c_str());
    check((std::string("a_checksum_") + label).c_str(), m.equal,
          "checksum = the scalar reference's over 1 pass of the buffer");
    return m;
}

// the A100's 8 tiles over the first f of the buffer while the other backend reads the rest, both launched before
// either is synced
static void run_concurrent(const char * label, bk other, uint32_t other_grid, uint32_t other_parts, bool other_caller,
                           double f, const uint64_t * p, size_t words, uint64_t ref, int reps) {
    const size_t cut = std::min(words, (size_t) ((double) words * f));
    if (cut == 0 || cut == words) {
        printf("a %s: skip, the split f=%.3f leaves one side empty\n", label, f);
        return;
    }
    parts_out oa, ob;
    sum_job   ja, jb;
    ja.p      = p;
    ja.off    = 0;
    ja.n      = cut;
    ja.nparts = g_a100_cores ? g_a100_cores : 8;
    ja.first  = 0;
    ja.out    = &oa;
    jb.p      = p;
    jb.off    = cut;
    jb.n      = words - cut;
    jb.nparts = other_parts;
    jb.first  = other_caller ? 1 : 0;
    jb.out    = &ob;

    std::vector<double> wall, span_a, span_b;
    bool                equal = true;
    for (int r = 0; r < reps; r++) {
        oa.init(ja.nparts);
        ob.init(jb.nparts);
        const auto t0 = clk::now();
        pending    pa = start_sum(BK_A100, ja.nparts, &ja);
        pending    pb = start_sum(other, other_grid, &jb);
        if (other_caller) {
            sum_part(&jb, 0);
        }
        const bool ok_b = finish(pb);
        const bool ok_a = finish(pa);
        const double ms = ms_since(t0);
        if (!ok_a || !ok_b) {
            printf("a %s: the pass failed (a100=%d other=%d)\n", label, ok_a ? 1 : 0, ok_b ? 1 : 0);
            return;
        }
        wall.push_back(ms);
        span_a.push_back(parts_span_ms(oa));
        span_b.push_back(parts_span_ms(ob));
        equal = equal && total_sum(oa, ja.nparts) + total_sum(ob, jb.nparts) == ref;
    }
    const double ba = (double) cut * 8, bb = (double) (words - cut) * 8;
    const double ga = gb_s(ba, median(span_a)), gbb = gb_s(bb, median(span_b));
    const double gc    = gb_s(ba + bb, median(wall));
    const double ratio = g_a100_solo[3] > 0 ? gc / g_a100_solo[3] : 0.0;
    printf("a %s: f=%.3f a100_ms=%.2f a100_gb_s=%.2f other_ms=%.2f other_gb_s=%.2f wall_ms=%.2f combined_gb_s=%.2f "
           "ratio_to_a100_solo=%.2f wall=%s%s\n",
           label, (double) cut / (double) words, median(span_a), ga, median(span_b), gbb, median(wall), gc, ratio,
           fmt_list(wall, "%.2f").c_str(), equal ? "" : " MISMATCH");
    check((std::string("a_checksum_") + label).c_str(), equal, "both sides' checksums sum to the scalar reference's");
    char line[320];
    snprintf(line, sizeof line,
             "head-probe: concurrent A100+%s (f=%.2f): A100 %.2f GB/s, %s %.2f GB/s, combined %.2f GB/s (ratio to "
             "A100 solo %.2f)",
             other == BK_X100 ? "X100" : "threads", (double) cut / (double) words, ga,
             other == BK_X100 ? "X100" : "threads", gbb, gc, ratio);
    g_conc_lines.push_back(line);
}

static void part_a() {
    const size_t n     = g_mib << 20;
    const char * adv   = "-";
    const auto   t0    = clk::now();
    uint8_t *    buf   = map_buf(n, adv);
    if (!buf) {
        printf("a buffer: mmap of %zu MiB failed\n", g_mib);
        return;
    }
    for (size_t i = 0; i < n; i += 8) {
        const uint64_t w = rng_next();
        memcpy(buf + i, &w, 8);
    }
    const uint64_t * p     = (const uint64_t *) buf;
    const size_t     words = n / 8;
    printf("a buffer: mib=%zu hugepage_advice=%s aligned_64=%d alloc_touch_ms=%.1f\n", g_mib, adv,
           ((uintptr_t) buf % 64) == 0 ? 1 : 0, ms_since(t0));
    const auto     tr  = clk::now();
    const uint64_t ref = sum_words_ref(p, 0, words);
    printf("a reference: scalar_ms=%.2f gb_s=%.2f checksum=0x%016llx\n", ms_since(tr),
           gb_s((double) n, ms_since(tr)), (unsigned long long) ref);

    // 1. the A100 stream at 1, 2, 4 and 8 tiles
    const uint32_t a100_cores = g_a100_cores ? g_a100_cores : 8;
    for (uint32_t k = 1; k <= 8; k *= 2) {
        if (k > a100_cores) {
            printf("a a100_%u: skip, the stream has %u cores\n", k, a100_cores);
            continue;
        }
        char label[32];
        snprintf(label, sizeof label, "a100_%u", k);
        const uint32_t grid = a100_grid(k);
        if (grid != k) {
            printf("a %s: a grid of %u is refused, launching %u tiles of which %u work\n", label, k, grid, k);
        }
        const sum_meas m = run_sum_cfg(label, BK_A100, grid, k, false, p, words, ref, g_reps);
        const int      s = k == 1 ? 0 : (k == 2 ? 1 : (k == 4 ? 2 : 3));
        if (m.ran) {
            g_a100_solo[s] = gb_s((double) n, m.med);
        }
    }

    // 2. the x100 backend: its tiles, and its tiles plus the calling thread
    if (g_x100_cores) {
        const sum_meas m1 = run_sum_cfg("x100_tiles", BK_X100, g_x100_cores, g_x100_cores, false, p, words, ref,
                                        g_reps);
        const sum_meas m2 = run_sum_cfg("x100_tiles_caller", BK_X100, g_x100_cores, g_x100_cores + 1, true, p, words,
                                        ref, g_reps);
        if (m1.ran) {
            g_x100_solo[0] = gb_s((double) n, m1.med);
        }
        if (m2.ran) {
            g_x100_solo[1] = gb_s((double) n, m2.med);
        }
    } else {
        printf("a x100_tiles: skip, no x100 backend\n");
    }

    // 3. unpinned std::thread
    const sum_meas t4 = run_sum_cfg("threads_4", BK_THREADS, 4, 4, false, p, words, ref, g_reps);
    const sum_meas t8 = run_sum_cfg("threads_8", BK_THREADS, 8, 8, false, p, words, ref, g_reps);
    if (t4.ran) {
        g_thr_solo[0] = gb_s((double) n, t4.med);
    }
    if (t8.ran) {
        g_thr_solo[1] = gb_s((double) n, t8.med);
    }

    // 4. both at once: f from the solo rates, then f = 0.75
    if (g_a100_solo[3] > 0 && g_x100_solo[1] > 0 && g_x100_cores) {
        const double f = g_a100_solo[3] / (g_a100_solo[3] + g_x100_solo[1]);
        run_concurrent("conc_a100_x100_auto", BK_X100, g_x100_cores, g_x100_cores + 1, true, f, p, words, ref, g_reps);
        run_concurrent("conc_a100_x100_075", BK_X100, g_x100_cores, g_x100_cores + 1, true, 0.75, p, words, ref,
                       g_reps);
    } else {
        printf("a conc_a100_x100: skip, need both solo rates (a100_8=%.2f x100_caller=%.2f cores=%u)\n",
               g_a100_solo[3], g_x100_solo[1], g_x100_cores);
    }
    if (g_a100_solo[3] > 0 && g_thr_solo[1] > 0) {
        const double f = g_a100_solo[3] / (g_a100_solo[3] + g_thr_solo[1]);
        run_concurrent("conc_a100_threads_auto", BK_THREADS, 8, 8, false, f, p, words, ref, g_reps);
        run_concurrent("conc_a100_threads_075", BK_THREADS, 8, 8, false, 0.75, p, words, ref, g_reps);
    } else {
        printf("a conc_a100_threads: skip, need both solo rates\n");
    }

    munmap(buf, n);
    printf("a buffer: freed %zu MiB\n", g_mib);
}

// ---- part B ----

// the read-only sum of the Q8_0 head's bytes on the A100 stream, for comparison with the Q6_K streaming pass
static double sum_q8_bytes(int64_t rows, size_t bytes, uint64_t & checksum_ok) {
    const char * adv = "-";
    uint8_t *    buf = map_buf(bytes, adv);
    if (!buf) {
        printf("b q8_sum_%lld: mmap of %zu MiB failed\n", (long long) rows, bytes >> 20);
        return 0.0;
    }
    for (size_t i = 0; i + 8 <= bytes; i += 8) {
        const uint64_t w = rng_next();
        memcpy(buf + i, &w, 8);
    }
    const size_t     words = bytes / 8;
    const uint64_t * p     = (const uint64_t *) buf;
    const uint64_t   ref   = sum_words_ref(p, 0, words);
    const uint32_t   np    = g_a100_cores ? g_a100_cores : 8;
    parts_out        out;
    sum_job          j;
    j.p      = p;
    j.n      = words;
    j.nparts = np;
    j.out    = &out;
    std::vector<double> ms;
    bool                equal = true;
    for (int r = 0; r < g_reps; r++) {
        out.init(np);
        const auto t0 = clk::now();
        pending    pd = start_sum(BK_A100, np, &j);
        const bool ok = finish(pd);
        ms.push_back(ms_since(t0));
        equal = equal && ok && total_sum(out, np) == ref;
    }
    const double med = median(ms);
    printf("b q8_sum_%lld_rows: mib=%zu tiles=%u median_ms=%.2f min_ms=%.2f max_ms=%.2f gb_s=%.2f ms=%s %s\n",
           (long long) rows, bytes >> 20, np, med, vmin(ms), vmax(ms), gb_s((double) bytes, med),
           fmt_list(ms, "%.2f").c_str(), equal ? "checksum equal" : "checksum MISMATCH");
    checksum_ok = equal ? 1 : 0;
    munmap(buf, bytes);
    return med;
}

struct table {
    uint8_t * p     = nullptr;
    size_t    bytes = 0;
    int64_t   rows  = 0;
    int64_t   nblk  = 0;
};

static bool make_table(table & t, int64_t rows, int64_t nblk) {
    t.rows        = rows;
    t.nblk        = nblk;
    t.bytes       = (size_t) rows * (size_t) nblk * k_q6k_bytes;
    const char * adv = "-";
    const auto   t0  = clk::now();
    t.p              = map_buf(t.bytes, adv);
    if (!t.p) {
        printf("b table: mmap of %zu MiB failed\n", t.bytes >> 20);
        return false;
    }
    // random ql, qh and scales; d a small positive fp16 around 1/1024
    for (size_t off = 0; off < t.bytes; off += k_q6k_bytes) {
        q6k_block * x = (q6k_block *) (t.p + off);
        uint8_t *   q = (uint8_t *) x;
        for (size_t i = 0; i + 8 <= 208; i += 8) {
            const uint64_t w = rng_next();
            memcpy(q + i, &w, 8);
        }
        const float    s = (1.0f / 1024.0f) * (1.0f + 0.5f * (float) (rng_next() >> 40) / 16777216.0f);
        const uint16_t h = f32_to_fp16(s);
        memcpy(x->d, &h, sizeof h);
    }
    printf("b table: rows=%lld blocks_per_row=%lld mib=%zu hugepage_advice=%s aligned_64=%d fill_ms=%.1f\n",
           (long long) rows, (long long) nblk, t.bytes >> 20, adv, ((uintptr_t) t.p % 64) == 0 ? 1 : 0,
           ms_since(t0));
    return true;
}

// one measured pass over the rows of the table on the given backend
struct q6k_meas {
    std::vector<double> ms;
    double              med   = 0.0;
    int64_t             rows  = 0;
    bool                ran   = false;
    int                 vlenb = 0;
};

static q6k_meas run_q6k(const char * label, const table & t, const int8_t * act, float * y, float scale_x, bk k,
                        uint32_t grid, uint32_t nparts, int variant, int reps_inner, int64_t rows_cache, int reps) {
    q6k_meas  m;
    parts_out out;
    q6k_job   j;
    j.tab        = t.p;
    j.act        = act;
    j.y          = y;
    j.nrows      = t.rows;
    j.nblk       = t.nblk;
    j.scale_x    = scale_x;
    j.variant    = variant;
    j.reps       = reps_inner;
    j.rows_cache = rows_cache;
    j.nparts     = nparts;
    j.out        = &out;
    for (int r = 0; r < reps; r++) {
        out.init(nparts);
        const auto t0 = clk::now();
        pending    pd = start_q6k(k, grid, &j);
        const bool ok = finish(pd);
        const double ms = ms_since(t0);
        if (!ok) {
            printf("b %s: the pass failed\n", label);
            return m;
        }
        int64_t rows = 0;
        for (uint32_t i = 0; i < nparts; i++) {
            if (out.rows[i] < 0) {
                printf("b %s: skip, part %u ran on a core with vlenb=%d (the kernel needs 128)\n", label, i,
                       out.vlenb[i]);
                m.vlenb = out.vlenb[i];
                return m;
            }
            rows += out.rows[i];
        }
        m.rows = rows;
        m.ms.push_back(ms);
    }
    m.ran   = true;
    m.med   = median(m.ms);
    m.vlenb = out.vlenb[0];
    const double weights = (double) m.rows * (double) t.nblk * (double) k_qk;
    const double q6bytes = (double) m.rows * (double) t.nblk * (double) k_q6k_bytes;
    printf("b %s: variant=%s grid=%u parts=%u rows=%lld median_ms=%.2f min_ms=%.2f max_ms=%.2f "
           "mweights_s=%.0f q6k_gb_s=%.2f per_tile_mweights_s=%.0f per_tile_q6k_gb_s=%.2f ms=%s vlenb=%s cpus=%s\n",
           label, variant == 0 ? "int" : "float", grid, nparts, (long long) m.rows, m.med, vmin(m.ms), vmax(m.ms),
           weights / m.med / 1e3, gb_s(q6bytes, m.med), weights / m.med / 1e3 / nparts,
           gb_s(q6bytes, m.med) / nparts, fmt_list(m.ms, "%.2f").c_str(), join(out.vlenb).c_str(),
           join(out.cpu0).c_str());
    return m;
}

// rows [lo, hi) on one A100 tile: the kernel needs the AI cores' VLEN 1024, so the checks never run it on the caller
static bool run_rows_tile(const table & t, const int8_t * act, float * y, float scale_x, int variant, int64_t lo,
                          int64_t hi, int64_t * bsum, int & vlenb) {
    parts_out out;
    q6k_job   j;
    j.tab     = t.p;
    j.act     = act;
    j.y       = y;
    j.nrows   = t.rows;
    j.nblk    = t.nblk;
    j.scale_x = scale_x;
    j.variant = variant;
    j.nparts  = 1;
    j.out     = &out;
    j.row_lo  = lo;
    j.row_hi  = hi;
    j.bsum    = bsum;
    out.init(1);
    pending    pd = start_q6k(BK_A100, a100_grid(1), &j);
    const bool ok = finish(pd);
    vlenb         = out.vlenb[0];
    return ok && out.rows[0] >= 0;
}

// NMSE of a against b over n values
static double nmse(const float * a, const float * b, int64_t n) {
    double num = 0.0, den = 0.0;
    for (int64_t i = 0; i < n; i++) {
        const double d = (double) a[i] - (double) b[i];
        num += d * d;
        den += (double) b[i] * (double) b[i];
    }
    return den > 0.0 ? num / den : num;
}

static std::vector<int64_t> sample_rows(int64_t rows, int64_t want) {
    std::vector<int64_t> s;
    const int64_t        n = std::min(rows, want);
    for (int64_t i = 0; i < n; i++) {
        s.push_back(n == 1 ? 0 : (rows - 1) * i / (n - 1));  // the first, the last and a spread between them
    }
    return s;
}

static void part_b_table(int64_t rows, double q8_ms_model) {
    const int64_t nblk = g_cols / k_qk;
    const size_t  q8   = (size_t) rows * (size_t) (g_cols / 32) * k_q8_bytes;

    // the comparison first, so the two buffers never live at the same time
    uint64_t     q8_ok  = 0;
    const double q8_sum = sum_q8_bytes(rows, q8, q8_ok);
    check((std::string("b_q8_sum_checksum_") + std::to_string(rows)).c_str(), q8_ok == 1,
          "the read-only sum of the Q8_0 head's bytes agrees with the scalar reference");

    table t;
    if (!make_table(t, rows, nblk)) {
        return;
    }

    // activations: random floats, quantized once with one scale
    std::vector<float> x((size_t) g_cols);
    float              amax = 0.0f;
    for (int64_t i = 0; i < g_cols; i++) {
        x[i] = (float) ((double) (rng_next() >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
        amax = std::max(amax, std::fabs(x[i]));
    }
    const float        scale_x = amax / 127.0f;
    std::vector<int8_t> act((size_t) g_cols);
    for (int64_t i = 0; i < g_cols; i++) {
        act[i] = (int8_t) std::lround(x[i] * 127.0f / (amax > 0.0f ? amax : 1.0f));
    }
    printf("b activations: cols=%lld max_abs=%.6f scale_x=%.9f\n", (long long) g_cols, amax, scale_x);

    std::vector<float> y((size_t) rows, 0.0f);

    // compute-only, on the first table only (the rate does not depend on the row count)
    if (g_b_compute.empty()) {
        const uint32_t np = g_a100_cores ? g_a100_cores : 8;
        int            inner = 1;
        q6k_meas       cal   = run_q6k("q6k_compute_cal", t, act.data(), y.data(), scale_x, BK_A100, np, np, 0, 1, 128,
                                       1);
        if (cal.ran && cal.med > 0.0) {
            inner = (int) std::max(1.0, std::ceil((double) g_compute_ms / cal.med));
        }
        printf("b q6k_compute: inner_reps=%d rows_per_tile=128 target_ms=%d\n", inner, g_compute_ms);
        const q6k_meas c8i = run_q6k("q6k_compute_8_int", t, act.data(), y.data(), scale_x, BK_A100, np, np, 0, inner,
                                     128, g_reps);
        const q6k_meas c8f = run_q6k("q6k_compute_8_float", t, act.data(), y.data(), scale_x, BK_A100, np, np, 1,
                                     inner, 128, g_reps);
        const q6k_meas c1i = run_q6k("q6k_compute_1_int", t, act.data(), y.data(), scale_x, BK_A100, a100_grid(1), 1,
                                     0, inner, 128, g_reps);
        const q6k_meas c1f = run_q6k("q6k_compute_1_float", t, act.data(), y.data(), scale_x, BK_A100, a100_grid(1), 1,
                                     1, inner, 128, g_reps);
        if (c8i.ran) {
            const double w = (double) c8i.rows * (double) nblk * (double) k_qk;
            const double b = (double) c8i.rows * (double) nblk * (double) k_q6k_bytes;
            char         line[256];
            snprintf(line, sizeof line,
                     "head-probe: q6_K compute-only %u tiles: %.2f GB/s of Q6_K bytes (%.0f Mweights/s)", np,
                     gb_s(b, c8i.med), w / c8i.med / 1e3);
            g_b_compute = line;
        }
        (void) c8f;
        (void) c1i;
        (void) c1f;
    }

    // streaming: the whole table once, on 8, 4 and 1 tiles
    const uint32_t np = g_a100_cores ? g_a100_cores : 8;
    double         stream_ms = 0.0;
    for (uint32_t k = np; k >= 1; k /= 2) {
        if (k != np && k != 4 && k != 1) {
            continue;
        }
        char label[48];
        snprintf(label, sizeof label, "q6k_stream_%u", k);
        const uint32_t grid = a100_grid(k);
        std::fill(y.begin(), y.end(), 0.0f);
        const q6k_meas m = run_q6k(label, t, act.data(), y.data(), scale_x, BK_A100, grid, k, 0, 0, 0, g_reps);
        if (k == np && m.ran) {
            stream_ms = m.med;
        }
        if (k == 1) {
            break;
        }
    }

    // the references on the sampled rows, against the kernel's own outputs
    const std::vector<int64_t> s = sample_rows(rows, g_sample);
    std::vector<float>         yr(s.size()), yd(s.size()), ys(s.size()), yki(s.size()), ykf(s.size());
    std::vector<int64_t>       bref((size_t) nblk), bker((size_t) nblk);
    double                     max_int = 0.0, max_scheme = 0.0, max_bsum = 0.0, rms = 0.0;
    bool                       kernel_ran = true;
    int                        kernel_vlenb = 0;
    for (size_t i = 0; i < s.size(); i++) {
        const uint8_t * row = t.p + (size_t) s[i] * (size_t) nblk * k_q6k_bytes;
        q6k_row_ref(row, act.data(), nblk, scale_x, &yr[i], bref.data());
        yd[i] = q6k_row_decode_dot(row, act.data(), nblk, scale_x);
        q6k_rows_scheme<0>(t.p, act.data(), y.data(), nblk, scale_x, s[i], s[i] + 1);
        ys[i] = y[s[i]];
        kernel_ran = kernel_ran && run_rows_tile(t, act.data(), y.data(), scale_x, 0, s[i], s[i] + 1, nullptr,
                                                 kernel_vlenb);
        yki[i]     = y[s[i]];
        kernel_ran = kernel_ran && run_rows_tile(t, act.data(), y.data(), scale_x, 1, s[i], s[i] + 1, nullptr,
                                                 kernel_vlenb);
        ykf[i]     = y[s[i]];
        kernel_ran = kernel_ran && run_rows_tile(t, act.data(), y.data(), scale_x, 2, s[i], s[i] + 1, bker.data(),
                                                 kernel_vlenb);
        if (!kernel_ran) {
            printf("b verify_%lld: the kernel did not run on a tile (vlenb=%d); the kernel checks fail\n",
                   (long long) rows, kernel_vlenb);
            break;
        }
        for (int64_t b = 0; b < nblk; b++) {
            max_bsum = std::max(max_bsum, (double) std::llabs(bker[(size_t) b] - bref[(size_t) b]));
        }
        max_int    = std::max(max_int, std::fabs((double) yki[i] - (double) yr[i]));
        max_scheme = std::max(max_scheme, std::fabs((double) ys[i] - (double) yr[i]));
        rms += (double) yr[i] * (double) yr[i];
    }
    rms                    = std::sqrt(rms / (double) s.size());
    const double nm_decode = nmse(yd.data(), yr.data(), (int64_t) s.size());
    const double nm_float  = nmse(ykf.data(), yr.data(), (int64_t) s.size());
    printf("b verify_%lld: sample=%zu first=%lld last=%lld nmse_decode=%.3e nmse_kernel_float=%.3e "
           "max_abs_kernel_int=%.3e max_abs_scheme_int=%.3e max_block_sum_diff=%.0f y_ref_rms=%.6f\n",
           (long long) rows, s.size(), (long long) s.front(), (long long) s.back(), nm_decode, nm_float, max_int,
           max_scheme, max_bsum, rms);
    const std::string tag = std::to_string(rows);
    char              det[128];
    snprintf(det, sizeof det, "NMSE %.3e <= 1e-10", nm_decode);
    check((std::string("b_ref_vs_ggml_decode_") + tag).c_str(), nm_decode <= 1e-10, det);
    check((std::string("b_scheme_int_exact_") + tag).c_str(), max_scheme == 0.0,
          "the plain-C scheme equals q6k_row_ref bit for bit");
    const std::string no_run = "the kernel did not run on a tile (vlenb=" + std::to_string(kernel_vlenb) + ")";
    check((std::string("b_kernel_int_exact_") + tag).c_str(), kernel_ran && max_int == 0.0,
          kernel_ran ? "the integer kernel equals q6k_row_ref bit for bit" : no_run);
    check((std::string("b_kernel_block_sums_") + tag).c_str(), kernel_ran && max_bsum == 0.0,
          kernel_ran ? "every per-block integer sum equals q6k_row_ref's" : no_run);
    snprintf(det, sizeof det, "NMSE %.3e <= 1e-10", nm_float);
    check((std::string("b_kernel_float_nmse_") + tag).c_str(), kernel_ran && nm_float <= 1e-10,
          kernel_ran ? std::string(det) : no_run);

    // the streaming pass itself: rerun on 8 tiles and compare the sampled rows
    std::fill(y.begin(), y.end(), 0.0f);
    const q6k_meas sv = run_q6k("q6k_stream_verify", t, act.data(), y.data(), scale_x, BK_A100, np, np, 0, 0, 0, 1);
    double         stream_max = 0.0;
    for (size_t i = 0; i < s.size(); i++) {
        stream_max = std::max(stream_max, std::fabs((double) y[s[i]] - (double) yr[i]));
    }
    check((std::string("b_stream_rows_") + tag).c_str(), sv.ran && stream_max == 0.0,
          "the 8-tile streaming pass equals q6k_row_ref on " + std::to_string(s.size()) + " sampled rows");

    char line[320];
    snprintf(line, sizeof line,
             "head-probe: q6_K streaming %lld rows: %.2f ms (today's Q8_0 head in the model: %.1f ms; read-only sum "
             "of %zu MiB here: %.2f ms)",
             (long long) rows, stream_ms, q8_ms_model, q8 >> 20, q8_sum);
    g_b_lines.push_back(line);

    munmap(t.p, t.bytes);
    printf("b table: freed %zu MiB\n", t.bytes >> 20);
}

static void part_b() {
    printf("b sizes: rows1=%lld rows2=%lld cols=%lld q6k_bytes_per_32=%.2f q8_bytes_per_32=%zu\n",
           (long long) g_rows1, (long long) g_rows2, (long long) g_cols, 32.0 * k_q6k_bytes / k_qk, k_q8_bytes);
    if (g_cols % k_qk != 0) {
        printf("b sizes: skip, cols=%lld is not a multiple of %d\n", (long long) g_cols, k_qk);
        return;
    }
    part_b_table(g_rows1, 22.4);
    part_b_table(g_rows2, 13.6);
}

// ---- main ----

static void summary() {
    printf("head-probe: A100 solo GB/s at 1/2/4/8 tiles: %.2f/%.2f/%.2f/%.2f\n", g_a100_solo[0], g_a100_solo[1],
           g_a100_solo[2], g_a100_solo[3]);
    printf("head-probe: X100 solo GB/s (4 tiles, 4+caller): %.2f/%.2f\n", g_x100_solo[0], g_x100_solo[1]);
    printf("head-probe: threads GB/s (4, 8): %.2f/%.2f\n", g_thr_solo[0], g_thr_solo[1]);
    for (const std::string & s : g_conc_lines) {
        printf("%s\n", s.c_str());
    }
    if (!g_b_compute.empty()) {
        printf("%s\n", g_b_compute.c_str());
    }
    for (const std::string & s : g_b_lines) {
        printf("%s\n", s.c_str());
    }
    printf("head-probe: checks %d/%d %s\n", g_checks - g_fails, g_checks, g_fails ? "FAIL" : "PASS");
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    std::set<std::string> want;
    bool                  small = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--small") {
            small = true;
            continue;
        }
        size_t p = 0;
        while (p <= a.size()) {
            const size_t q = std::min(a.find(',', p), a.size());
            if (q > p) {
                want.insert(a.substr(p, q - p));
            }
            p = q + 1;
        }
    }
    for (const std::string & w : want) {
        if (w != "a" && w != "b" && w != "all") {
            fprintf(stderr, "unknown part %s; parts: a b all\n", w.c_str());
            return 2;
        }
    }
    const bool all    = want.empty() || want.count("all") > 0;
    const bool do_a   = all || want.count("a") > 0;
    const bool do_b   = all || want.count("b") > 0;

    g_mib        = (size_t) std::max(1, env_int("PROBE_MIB", small ? 64 : 1024));
    g_reps       = std::max(1, env_int("REPS", small ? 3 : 5));
    g_rows1      = std::max(1, env_int("PROBE_ROWS1", small ? 4096 : 248320));
    g_rows2      = std::max(1, env_int("PROBE_ROWS2", small ? 2048 : 151936));
    g_cols       = std::max(k_qk, env_int("PROBE_COLS", 2560));
    g_compute_ms = std::max(1, env_int("PROBE_COMPUTE_MS", small ? 5 : 50));
    g_sample     = std::max(1, env_int("PROBE_SAMPLE", 512));
    g_grant_s    = env_int("PROBE_GRANT_S", 20);
    g_rng        = (uint64_t) std::max(1, env_int("PROBE_SEED", 1));

    printf("start: pid=%d parts=%s%s small=%d probe_mib=%zu reps=%d rows=%lld,%lld cols=%lld compute_ms=%d "
           "sample=%lld grant_wait_s=%d rvv=%d spert=%d vlenb=%d\n",
           (int) getpid(), do_a ? "a" : "", do_b ? "b" : "", small ? 1 : 0, g_mib, g_reps, (long long) g_rows1,
           (long long) g_rows2, (long long) g_cols, g_compute_ms, (long long) g_sample, g_grant_s, HP_RVV,
#if defined(HEAD_PROBE_SPERT)
           1,
#else
           0,
#endif
           hp_vlenb());
    maps("start");

#if defined(HEAD_PROBE_SPERT)
    {
        const spert::BackendInfo di = spert::backend_info(spert::default_backend());
        printf("info default: valid=%d shared_mem_size=%zu vlen=%zu num_cores=%zu\n",
               spert::default_backend().valid() ? 1 : 0, di.shared_mem_size, di.vlen, di.num_cores);
        made_stream a = make_stream([] { return new spert::Stream(8u); });
        if (a.blocked) {
            printf("info a100: Stream(8) blocked over %d s (left waiting)\n", g_grant_s);
        } else if (a.s->valid() && a.s->core_count() > 0) {
            g_a100       = a.s;
            g_a100_cores = a.s->core_count();
            printf("info a100: Stream(8) valid=1 core_count=%u create_ms=%.2f\n", g_a100_cores, a.ms);
            const spert::Future f = g_a100->launch(spert::Grid{ 1 }, empty_tile);
            g_subgrid              = f.valid() && f.sync() == spert::Status::Ok;
            printf("info a100 subgrid: a grid of 1 on an 8-core stream %s\n", g_subgrid ? "works" : "is refused");
        } else {
            printf("info a100: Stream(8) valid=%d core_count=%u\n", a.s->valid() ? 1 : 0, a.s->core_count());
            delete a.s;
        }
        const spert::BackendHandle h = spert::backend(k_x100_name);
        const spert::BackendInfo   xi = spert::backend_info(h);
        printf("info x100: name=%s valid=%d shared_mem_size=%zu vlen=%zu num_cores=%zu\n", k_x100_name,
               h.valid() ? 1 : 0, xi.shared_mem_size, xi.vlen, xi.num_cores);
        if (h.valid() && xi.num_cores > 0) {
            spert::StreamConfig cfg;
            cfg.n_cores   = (uint32_t) xi.num_cores;
            made_stream x = make_stream([h, cfg] { return new spert::Stream(h, cfg); });
            if (x.blocked) {
                printf("info x100 stream: blocked over %d s (left waiting)\n", g_grant_s);
            } else if (x.s->valid() && x.s->core_count() > 0) {
                g_x100       = x.s;
                g_x100_cores = x.s->core_count();
                printf("info x100 stream: valid=1 core_count=%u create_ms=%.2f\n", g_x100_cores, x.ms);
            } else {
                printf("info x100 stream: valid=%d core_count=%u (both backends at once not tested)\n",
                       x.s->valid() ? 1 : 0, x.s->core_count());
                delete x.s;
            }
        }
    }
#else
    g_a100_cores = 8;
    g_x100_cores = 4;
    printf("info backends: no spine-runtime in this build, std::thread stands in for both (a100=%u x100=%u)\n",
           g_a100_cores, g_x100_cores);
#endif
    maps("after backends");

    if (do_a) {
        const auto t0 = clk::now();
        part_a();
        printf("part a: wall_ms=%.1f\n", ms_since(t0));
        maps("after part a");
    }
    if (do_b) {
        const auto t0 = clk::now();
        part_b();
        printf("part b: wall_ms=%.1f\n", ms_since(t0));
        maps("after part b");
    }

    summary();

#if defined(HEAD_PROBE_SPERT)
    if (g_x100) {
        delete g_x100;
        g_x100 = nullptr;
    }
    if (g_a100) {
        delete g_a100;
        g_a100 = nullptr;
    }
#endif
    printf("done: checks=%d fails=%d abandoned_grants=%d\n", g_checks, g_fails, g_abandoned);
    fflush(stdout);
    if (g_abandoned) {
        _exit(g_fails ? 1 : 3);  // a thread still waits inside spert; skip the runtime's exit-time teardown
    }
    return g_fails ? 1 : 0;
}
