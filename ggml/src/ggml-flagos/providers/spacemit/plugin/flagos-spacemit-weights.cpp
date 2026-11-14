#include "flagos-spacemit-weights.h"

#include "flagos-spacemit-ime-kernels.h"
#include "flagos-spacemit-kernels.h"
#include "flagos-spacemit-x100.h"

#include "../../../../ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <system_error>
#include <thread>
#include <vector>

// llama.cpp's KV cache and recurrent state, named before they are allocated (src/llama-kv-cache.cpp:231-235,
// src/llama-memory-recurrent.cpp:102-103)
static bool spacemit_is_cache_name(const char * name) {
    for (const char * prefix : { "cache_k_", "cache_v_", "cache_r_", "cache_s_" }) {
        if (std::strncmp(name, prefix, std::strlen(prefix)) == 0) {
            return true;
        }
    }
    return false;
}

// a 3-D tensor (MUL_MAT_ID's experts, [K, N, n_expert]) is stored as its ggml_nrows rows in 32-row groups, each
// expert's N rows whole groups (N % 32 == 0), so expert e's rows start at row e * N. Q4_1 takes its layout in 2-D only:
// the requantization's read-back is the nearest Q4_1, on which test-backend-ops' MUL_MAT_ID cases fail under imatrix
// quantization (NMSE about 6e-4 against 5e-4), so a 3-D Q4_1 weight stays plain and MUL_MAT_ID refuses it
spacemit_layout spacemit_weight_layout(const ggml_tensor * t) {
    if (t->view_src != nullptr || t->ne[1] % spacemit_ime::row_tile != 0 || t->ne[3] != 1 ||
        !ggml_is_contiguous(t) || spacemit_is_cache_name(t->name)) {
        return spacemit_layout::plain;
    }
    if (t->type == GGML_TYPE_Q4_0 && t->ne[0] % spacemit_ime::q4_0_k_block == 0) {
        return spacemit_layout::q4_0_32x256;
    }
    if (t->type == GGML_TYPE_Q4_1 && t->ne[0] % spacemit_ime::q8_0_k_block == 0 && t->ne[2] == 1) {
        return spacemit_layout::q4_1_q8_0_32x32;
    }
    if (t->type == GGML_TYPE_Q8_0 && t->ne[0] % spacemit_ime::q8_0_k_block == 0) {
        return spacemit_layout::q8_0_32x32;
    }
    if (t->type == GGML_TYPE_Q6_K && t->ne[0] % ggml_blck_size(GGML_TYPE_Q6_K) == 0) {  // always true for Q6_K
        return spacemit_layout::q6_k_q8_0_32x32;
    }
    if (t->type == GGML_TYPE_Q5_0 && t->ne[0] % spacemit_ime::q8_0_k_block == 0) {  // always true for Q5_0
        return spacemit_layout::q5_0_q8_0_32x32;
    }
    return spacemit_layout::plain;
}

// the layouts requantized to Q8_0 at load: reads give the nearest weight of the tensor's type, writes must be whole
static bool spacemit_requantized(spacemit_layout layout) {
    return layout == spacemit_layout::q4_1_q8_0_32x32 || layout == spacemit_layout::q6_k_q8_0_32x32;
}

size_t spacemit_weight_alloc_size(const ggml_tensor * t) {
    const spacemit_layout layout = spacemit_weight_layout(t);
    if (spacemit_requantized(layout) || layout == spacemit_layout::q5_0_q8_0_32x32) {
        return (size_t) ggml_nrows(t) * spacemit_ime::q8_0_weight_row_bytes((size_t) t->ne[0]);
    }
    return ggml_nbytes(t);
}

// the tensor that owns t's bytes, and t's byte offset inside it
static ggml_tensor * spacemit_storage(const ggml_tensor * t, size_t & offset) {
    if (t->view_src == nullptr) {
        offset = 0;
        return const_cast<ggml_tensor *>(t);
    }
    offset = static_cast<size_t>(static_cast<const char *>(t->data) - static_cast<const char *>(t->view_src->data));
    return t->view_src;
}

namespace {

constexpr int    repack_max_threads     = 16;
constexpr int    repack_default_threads = 8;  // T6's rule passed on the K3, 2026-11-12
constexpr size_t repack_min_bytes       = size_t(1) << 20;  // a smaller weight repacks on the calling thread

// FLAGOS_SPACEMIT_REPACK_THREADS=<n> (plan.md "T6 spec"): the host threads that repack one weight, n decimal digits
// only, 1 to 16; unset or empty: 8. Parsed once (spacemit_cached_switch); anything else keeps the default with a
// warning, once per parse generation
int parse_repack_threads() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_REPACK_THREADS");
    if (v == nullptr || *v == '\0') {
        return repack_default_threads;
    }
    int          n = 0;
    const char * c = v;
    for (; *c >= '0' && *c <= '9' && n <= repack_max_threads; ++c) {  // stops past 16, so no overflow
        n = n * 10 + (*c - '0');
    }
    if (*c == '\0' && n >= 1 && n <= repack_max_threads) {
        return n;
    }
    static std::atomic<uint32_t> warned_in{ 0 };  // the generation last warned in; generations start at 1
    const uint32_t               generation = spacemit_switches_generation();
    if (warned_in.exchange(generation) != generation) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_REPACK_THREADS=%s: not a thread count from 1 to 16, the default %d kept\n",
                      __func__, v, repack_default_threads);
    }
    return repack_default_threads;
}

int repack_threads_from_env() {
    static std::atomic<int>      value{ repack_default_threads };
    static std::atomic<uint32_t> seen{ 0 };
    return spacemit_cached_switch(value, seen, parse_repack_threads);
}

enum repack_via : int { repack_via_threads = 0, repack_via_x100 = 1 };

#if defined(GGML_FLAGOS_SPACEMIT_SPERT)
constexpr int repack_via_default = repack_via_x100;  // user, 2026-11-14
#else
constexpr int repack_via_default = repack_via_threads;
#endif

// FLAGOS_SPACEMIT_REPACK_VIA (plan.md "Repack on spert's X100 backend"): threads: std::thread (T6); x100: the tiles of
// spine-runtime's X100 backend. Unset or empty: x100 with spine-runtime in the build, threads without. Parsed once
// (spacemit_cached_switch); x100 without spine-runtime and any other value warn once per parse generation
int parse_repack_via() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_REPACK_VIA");
    if (v == nullptr || *v == '\0') {
        return repack_via_default;
    }
    if (std::strcmp(v, "threads") == 0) {
        return repack_via_threads;
    }
#if defined(GGML_FLAGOS_SPACEMIT_SPERT)
    if (std::strcmp(v, "x100") == 0) {
        return repack_via_x100;
    }
#endif
    static std::atomic<uint32_t> warned_in{ 0 };
    const uint32_t               generation = spacemit_switches_generation();
    if (warned_in.exchange(generation) != generation) {
        if (std::strcmp(v, "x100") == 0) {
            GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_REPACK_VIA=x100: no spine-runtime in this build, threads kept\n",
                          __func__);
        } else {
            GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_REPACK_VIA=%s: not threads or x100, the default kept\n", __func__, v);
        }
    }
    return repack_via_default;
}

int repack_via_from_env() {
    static std::atomic<int>      value{ repack_via_default };
    static std::atomic<uint32_t> seen{ 0 };
    return spacemit_cached_switch(value, seen, parse_repack_via);
}

// test-only (spacemit_repack_take): the repacks that ran on several threads, and the most ranges one of them used
std::atomic<uint64_t> g_repack_threaded{ 0 }, g_repack_max_ranges{ 0 };
// test-only (spacemit_repack_x100_take): under x100 the weights repacked through an admitted launch, those that fell
// back to the calling thread, and the ranges run again on the calling thread
std::atomic<uint64_t> g_repack_x100_through{ 0 }, g_repack_x100_not_through{ 0 }, g_repack_x100_reran{ 0 };

int spacemit_repack(spacemit_layout layout, ggml_tensor * t, const void * data, size_t size) {
    switch (layout) {
        case spacemit_layout::q4_0_32x256:
            return spacemit_ime::repack_q4_0(t, data, size);
        case spacemit_layout::q4_1_q8_0_32x32:
            return spacemit_ime::repack_q4_1_q8_0(t, data, size);
        case spacemit_layout::q8_0_32x32:
            return spacemit_ime::repack_q8_0(t, data, size);
        case spacemit_layout::q6_k_q8_0_32x32:
            return spacemit_ime::repack_q6_k(t, data, size);
        case spacemit_layout::q5_0_q8_0_32x32:
            return spacemit_ime::repack_q5_0_q8_0(t, data, size);
        case spacemit_layout::plain:
            break;
    }
    return -1;
}

// the ranges of one split weight, for spacemit_x100_run
struct pack_ranges {
    spacemit_layout      layout;
    ggml_tensor *        parts;
    const char * const * srcs;
    const size_t *       sizes;
    int *                rcs;
};

void pack_range(void * arg, uint32_t i) {
    const pack_ranges & r = *static_cast<const pack_ranges *>(arg);
    r.rcs[i]              = spacemit_repack(r.layout, &r.parts[i], r.srcs[i], r.sizes[i]);
}

}  // namespace

uint32_t spacemit_repack_threads() {
    return (uint32_t) repack_threads_from_env();
}

uint32_t spacemit_repack_via() {
    return (uint32_t) repack_via_from_env();
}

void spacemit_repack_take(uint64_t * threaded, uint64_t * max_ranges) {
    *threaded   = g_repack_threaded.exchange(0);
    *max_ranges = g_repack_max_ranges.exchange(0);
}

void spacemit_repack_x100_take(uint64_t * through_stream, uint64_t * not_through, uint64_t * reran) {
    *through_stream = g_repack_x100_through.exchange(0);
    *not_through    = g_repack_x100_not_through.exchange(0);
    *reran          = g_repack_x100_reran.exchange(0);
}

// A weight of several 32-row groups and at least repack_min_bytes is cut into up to FLAGOS_SPACEMIT_REPACK_THREADS
// ranges of whole groups, each repacked by the same routine on a stand-in tensor of its rows (plan.md "T6 spec"): every
// routine works group by group, so the bytes are the one-thread bytes. A 3-D weight is cut the same way, on its
// ggml_nrows rows: a range may span experts, as its rows are one flat list of groups. The first range runs on the
// calling thread. Under FLAGOS_SPACEMIT_REPACK_VIA=x100 the other ranges are tiles on the X100 backend, at most its
// cores + 1 ranges, and no thread is started
static void spacemit_pack(ggml_tensor * t, const void * data) {
    const spacemit_layout layout = spacemit_weight_layout(t);
    const size_t          size   = ggml_nbytes(t);
    const int64_t         nrows  = ggml_nrows(t);
    const int64_t         groups = nrows / (int64_t) spacemit_ime::row_tile;
    const bool            split  = !(layout == spacemit_layout::plain || size < repack_min_bytes || groups <= 1);
    int64_t               n      = split ? std::min<int64_t>(repack_threads_from_env(), groups) : 1;
    const bool            x100   = n > 1 && repack_via_from_env() == repack_via_x100;
    if (x100) {  // the backend starts (workers, /dev/tcm) at the first such weight
        const uint32_t cores = spacemit_x100_cores();
        if (cores == 0) {
            static std::atomic<bool> warned{ false };
            if (!warned.exchange(true)) {
                GGML_LOG_WARN("FlagOS SpacemiT: FLAGOS_SPACEMIT_REPACK_VIA=x100: the X100 backend has no cores; "
                              "weights repack on the calling thread\n");
            }
            g_repack_x100_not_through.fetch_add(1, std::memory_order_relaxed);
            n = 1;
        } else {
            n = std::min<int64_t>(n, (int64_t) cores + 1);
        }
    }
    if (n == 1) {
        GGML_ASSERT(spacemit_repack(layout, t, data, size) == 0 && "tensor does not fit its IME layout");
        return;
    }

    // a weight layout has ne[1] % 32 == 0, ne[3] == 1 and no view; each range's stand-in is a 2-D tensor of its rows.
    // The Q8_0 layouts store 34 bytes per 32 weights, Q4_0's its own row size
    const size_t src_row = ggml_row_size(t->type, t->ne[0]);
    const size_t dst_row = layout == spacemit_layout::q4_0_32x256 ? src_row :
                                                                    spacemit_ime::q8_0_weight_row_bytes((size_t) t->ne[0]);
    std::vector<ggml_tensor>  parts((size_t) n);
    std::vector<const char *> srcs((size_t) n);
    std::vector<size_t>       sizes((size_t) n);
    std::vector<int>          rcs((size_t) n, -1);
    int64_t                   row0 = 0;
    for (int64_t i = 0; i < n; i++) {
        const int64_t rows = (groups / n + (i < groups % n ? 1 : 0)) * (int64_t) spacemit_ime::row_tile;
        ggml_tensor & p    = parts[i];
        p                  = *t;
        p.ne[1]            = rows;
        p.ne[2]            = 1;
        p.ne[3]            = 1;
        p.nb[2]            = p.nb[1] * (size_t) rows;
        p.nb[3]            = p.nb[2];
        p.data             = static_cast<char *>(t->data) + (size_t) row0 * dst_row;
        srcs[i]            = static_cast<const char *>(data) + (size_t) row0 * src_row;
        sizes[i]           = (size_t) rows * src_row;
        row0 += rows;
    }
    GGML_ASSERT(row0 == nrows);

    if (x100) {
        pack_ranges ranges = { layout, parts.data(), srcs.data(), sizes.data(), rcs.data() };
        uint32_t    reran  = 0;
        const bool  ran    = spacemit_x100_run((uint32_t) n, pack_range, &ranges, &reran);
        (ran ? g_repack_x100_through : g_repack_x100_not_through).fetch_add(1, std::memory_order_relaxed);
        g_repack_x100_reran.fetch_add(reran, std::memory_order_relaxed);
    } else {
        std::vector<std::thread> workers;
        workers.reserve((size_t) n - 1);
        for (int64_t i = 1; i < n; i++) {
            try {
                workers.emplace_back([&, i] { rcs[i] = spacemit_repack(layout, &parts[i], srcs[i], sizes[i]); });
            } catch (const std::system_error &) {
                rcs[i] = spacemit_repack(layout, &parts[i], srcs[i], sizes[i]);  // no thread: this range here
            }
        }
        rcs[0] = spacemit_repack(layout, &parts[0], srcs[0], sizes[0]);
        for (std::thread & w : workers) {
            w.join();
        }
    }

    g_repack_threaded.fetch_add(1, std::memory_order_relaxed);
    uint64_t most = g_repack_max_ranges.load(std::memory_order_relaxed);
    while (most < (uint64_t) n && !g_repack_max_ranges.compare_exchange_weak(most, (uint64_t) n)) {
    }
    for (const int rc : rcs) {
        GGML_ASSERT(rc == 0 && "tensor does not fit its IME layout");
    }
}

static void spacemit_unpack(const ggml_tensor * t, void * data) {
    switch (spacemit_weight_layout(t)) {
        case spacemit_layout::q4_0_32x256:
            spacemit_ime::unpack_q4_0(t, data);
            return;
        case spacemit_layout::q4_1_q8_0_32x32:
            spacemit_ime::unpack_q4_1_q8_0(t, data);
            return;
        case spacemit_layout::q8_0_32x32:
            spacemit_ime::unpack_q8_0(t, data);
            return;
        case spacemit_layout::q6_k_q8_0_32x32:
            spacemit_ime::unpack_q6_k(t, data);
            return;
        case spacemit_layout::q5_0_q8_0_32x32:
            spacemit_ime::unpack_q5_0_q8_0(t, data);
            return;
        case spacemit_layout::plain:
            break;
    }
    GGML_ABORT("tensor has no IME layout");
}

// a partial write into a lossy layout goes through the read-back of the whole tensor. Requantized to Q8_0 (Q4_1, Q6_K):
// the read-back is approximate, so writing it back would requantize every block the write leaves alone; only a write
// of the whole tensor is allowed. llama.cpp writes weights whole on this device.
static void spacemit_check_partial(const ggml_tensor * base, size_t offset, size_t size) {
    const spacemit_layout layout = spacemit_weight_layout(base);
    GGML_ASSERT((!spacemit_requantized(layout) || (offset == 0 && size == ggml_nbytes(base))) &&
                "a partial write to a Q4_1 or Q6_K weight must cover the whole tensor");
}

spacemit_layout spacemit_storage_layout(const ggml_tensor * t) {
    size_t offset = 0;
    return spacemit_weight_layout(spacemit_storage(t, offset));
}

bool spacemit_tensor_is_repacked(const ggml_tensor * t) {
    return spacemit_storage_layout(t) != spacemit_layout::plain;
}

void spacemit_tensor_write(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    size_t        base_offset = 0;
    ggml_tensor * base        = spacemit_storage(t, base_offset);
    if (spacemit_weight_layout(base) == spacemit_layout::plain) {
        std::memcpy(static_cast<char *>(t->data) + offset, data, size);
        return;
    }
    offset += base_offset;
    const size_t nbytes = ggml_nbytes(base);
    if (offset == 0 && size == nbytes) {
        spacemit_pack(base, data);
        return;
    }
    spacemit_check_partial(base, offset, size);
    std::vector<uint8_t> whole(nbytes);
    spacemit_unpack(base, whole.data());
    std::memcpy(whole.data() + offset, data, size);
    spacemit_pack(base, whole.data());
}

void spacemit_tensor_read(const ggml_tensor * t, void * data, size_t offset, size_t size) {
    size_t              base_offset = 0;
    const ggml_tensor * base        = spacemit_storage(t, base_offset);
    if (spacemit_weight_layout(base) == spacemit_layout::plain) {
        std::memcpy(data, static_cast<const char *>(t->data) + offset, size);
        return;
    }
    offset += base_offset;
    const size_t nbytes = ggml_nbytes(base);
    if (offset == 0 && size == nbytes) {
        spacemit_unpack(base, data);
        return;
    }
    std::vector<uint8_t> whole(nbytes);
    spacemit_unpack(base, whole.data());
    std::memcpy(data, whole.data() + offset, size);
}

void spacemit_tensor_fill(ggml_tensor * t, uint8_t value, size_t offset, size_t size) {
    size_t        base_offset = 0;
    ggml_tensor * base        = spacemit_storage(t, base_offset);
    if (spacemit_weight_layout(base) == spacemit_layout::plain) {
        std::memset(static_cast<char *>(t->data) + offset, value, size);
        return;
    }
    spacemit_check_partial(base, base_offset + offset, size);
    std::vector<uint8_t> whole(ggml_nbytes(base));
    spacemit_unpack(base, whole.data());
    std::memset(whole.data() + base_offset + offset, value, size);
    spacemit_pack(base, whole.data());
}
