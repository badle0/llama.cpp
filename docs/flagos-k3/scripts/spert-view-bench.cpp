// E7 (structural-experiments.md): the shared-buffer view across suspensions, with no provider and no ggml. spine-runtime's
// header says of Context::shared_buffer() (spert.hpp:474-478, 0.6.3 on the board, read 2026-10-30): "Returns the entire
// shared scratch buffer of the current worker. The view is non-owning and remains valid only while this tile executes
// on that worker." Our tile loop (flagos-spacemit-exec.cpp, spacemit_tile_main) takes the view once per launch and uses
// it through every barrier; E3d saw no tile change core inside one launch, E3b-diag saw tiles move and share a TCM
// when launches are ordered by join (F16). This bench measures three forms of TCM use under both launch shapes:
//   hold     the view taken once, every canary written and read through that first pointer (today's loop; E3b-diag)
//   retake   the view taken again after every barrier; the canary written through the view before the barrier and
//            read through the first pointer and through the fresh view after it; counts view changes against core
//            changes (the header's rule: the view changes exactly when the tile left its worker)
//   slice    a tile-private slice from Context::alloc_shared held across the barriers, the canary through it; after
//            each barrier, whether the slice still lies inside the current worker's buffer; free_shared at the end
// under two shapes:
//   single   one launch, K barriers (today's executor: one launch per split with a barrier per step)
//   joined   N launches queued, every tile first joining its own copy of the previous launch's Future (E3b's
//            ordering), then M barriers, the host waiting once
// Counts per form and shape: core changes (a tile's core after a barrier differs from before), tiles that changed core
// at all, view changes with and without a core change and core changes without a view change (retake), canaries lost
// across a barrier through the old pointer and through the fresh view, canaries torn with no suspension point in
// between (another tile using this buffer at the same time), slice allocation failures and slices found outside the
// current worker's buffer, order violations, launches with two tiles on one core or on one TCM at the end.
// The joined cases run last: if one hangs (no new line for a minute), stop it with Ctrl-C and run tcmrelease --apply
// Build on the K3: g++ -O2 -std=c++17 -I$HOME/spine-runtime/include spert-view-bench.cpp -L$HOME/spine-runtime/lib \
//                      -lspert -o spert-view-bench
// Run (inside tmux, nothing else on the AI cores): ~/tcmtest/tcmrelease --apply
//     LD_LIBRARY_PATH=$HOME/spine-runtime/lib ./spert-view-bench [joined launches per case, default 4000] \
//                                                                  [barriers per joined launch, default 4] [slice KiB, default 16]
#include <spert.hpp>

#if defined(__linux__)
#    include <sched.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <vector>

namespace {

using clk                    = std::chrono::steady_clock;
constexpr uint32_t MAX_TILES = 64;
constexpr size_t   WORDS     = 1024;  // canary words, scalar loads and stores

double us_between(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

int cpu_now() {
#if defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}

bool shares_core(const int * cpus, uint32_t n) {
    for (uint32_t a = 0; a < n; a++) {
        for (uint32_t b = a + 1; b < n; b++) {
            if (cpus[a] >= 0 && cpus[a] == cpus[b]) {
                return true;
            }
        }
    }
    return false;
}

std::set<int> g_cores;
int           g_failed_status = 0;

int count_failed(spert::Status st) {
    if (st == spert::Status::Ok) {
        return 0;
    }
    if (g_failed_status == 0) {
        g_failed_status = spert::status_code(st);
    }
    return 1;
}

enum class form { hold, retake, slice };

const char * form_name(form f) {
    return f == form::hold ? "hold" : f == form::retake ? "retake" : "slice";
}

struct view_arg {
    std::vector<spert::Future>    previous;                 // one copy per tile; none for a single launch or a batch's first
    const std::atomic<uint32_t> * previous_done = nullptr;  // the previous launch's finished tiles
    std::atomic<uint32_t>         done{ 0 };
    uint32_t                      nth         = 0;
    uint32_t                      id          = 0;  // the launch's number, in the canaries
    form                          f           = form::hold;
    int                           barriers    = 1;
    size_t                        slice_bytes = 0;
    std::atomic<int> *            failed      = nullptr;
    std::atomic<int> *            violations  = nullptr;
    // per tile
    int          core_changes[MAX_TILES]       = {};  // barriers after which the tile was on another core
    int          view_with_core[MAX_TILES]     = {};  // retake: the fresh view differs and the core changed
    int          view_without_core[MAX_TILES]  = {};  // retake: the fresh view differs, the core the same
    int          core_without_view[MAX_TILES]  = {};  // retake: the core changed, the fresh view the same
    int          lost_old[MAX_TILES]           = {};  // canary changed across a barrier, read through the old pointer
    int          lost_fresh[MAX_TILES]         = {};  // retake: read through the fresh view; slice: through the slice
    int          torn[MAX_TILES]               = {};  // canary changed with no suspension point in between
    int          alloc_failed[MAX_TILES]       = {};  // slice: alloc_shared returned an empty view
    int          slice_outside[MAX_TILES]      = {};  // slice: after a barrier, the slice is not inside the current worker's buffer
    int          cpu_end[MAX_TILES]            = {};
    const void * tcm_end[MAX_TILES]            = {};
};

// writes the canary for barrier b through w, reads it back four times with no suspension point in between
bool write_and_check(volatile uint32_t * w, size_t words, uint32_t pattern) {
    for (size_t j = 0; j < words; j++) {
        w[j] = pattern;
    }
    bool intact = true;
    for (int pass = 0; pass < 4; pass++) {
        for (size_t j = 0; j < words; j++) {
            intact = intact && w[j] == pattern;
        }
    }
    return intact;
}

bool check(const volatile uint32_t * w, size_t words, uint32_t pattern) {
    bool kept = true;
    for (size_t j = 0; j < words; j++) {
        kept = kept && w[j] == pattern;
    }
    return kept;
}

void tile_view(spert::Context * ctx, view_arg * a) noexcept {
    const uint32_t ith = ctx->program_id(0);
    if (!a->previous.empty()) {
        if (ctx->join(a->previous[ith]) != spert::Status::Ok) {
            a->failed->fetch_add(1);
        } else if (a->previous_done->load() != a->nth) {
            a->violations->fetch_add(1);
        }
    }
    int                     cpu  = cpu_now();
    spert::SharedBufferView view = ctx->shared_buffer();  // the first view: hold uses it throughout
    spert::SharedBufferView sl;
    if (a->f == form::slice) {
        sl = ctx->alloc_shared(a->slice_bytes, 64);  // may suspend until space is available (the header)
        if (sl.data == nullptr || sl.size == 0) {
            a->alloc_failed[ith]++;
        }
        cpu = cpu_now();  // the allocation may have suspended the tile
    }
    for (int b = 0; b < a->barriers; b++) {
        // the region this form writes through before the barrier
        volatile uint32_t * w = static_cast<volatile uint32_t *>(a->f == form::slice ? sl.data : view.data);
        const size_t        bytes = a->f == form::slice ? sl.size : view.size;
        const size_t        words = w != nullptr ? std::min<size_t>(bytes / 4, WORDS) : 0;
        const uint32_t      pattern = (a->id << 12) ^ (ith << 8) ^ static_cast<uint32_t>(b) ^ 0xa5a50000u;
        if (words > 0 && !write_and_check(w, words, pattern)) {
            a->torn[ith]++;
        }
        if (ctx->sync() != spert::Status::Ok) {
            a->failed->fetch_add(1);
            break;
        }
        const int  c            = cpu_now();
        const bool core_changed = c != cpu;
        cpu                     = c;
        a->core_changes[ith] += core_changed;
        if (words > 0 && !check(w, words, pattern)) {  // through the pointer from before the barrier
            a->lost_old[ith]++;
        }
        if (a->f == form::retake) {
            const spert::SharedBufferView fresh        = ctx->shared_buffer();
            const bool                    view_changed = fresh.data != view.data || fresh.size != view.size;
            a->view_with_core[ith] += view_changed && core_changed;
            a->view_without_core[ith] += view_changed && !core_changed;
            a->core_without_view[ith] += !view_changed && core_changed;
            const volatile uint32_t * fw     = static_cast<const volatile uint32_t *>(fresh.data);
            const size_t              fwords = fw != nullptr ? std::min<size_t>(fresh.size / 4, WORDS) : 0;
            if (fwords > 0 && !check(fw, fwords, pattern)) {
                a->lost_fresh[ith]++;
            }
            view = fresh;
        } else if (a->f == form::slice) {
            const spert::SharedBufferView fresh = ctx->shared_buffer();
            const auto *                  lo    = static_cast<const uint8_t *>(fresh.data);
            const auto *                  s     = static_cast<const uint8_t *>(sl.data);
            if (s != nullptr && (lo == nullptr || s < lo || s + sl.size > lo + fresh.size)) {
                a->slice_outside[ith]++;
            }
            if (words > 0 && !check(w, words, pattern)) {  // the slice is both the old and the fresh pointer
                a->lost_fresh[ith]++;
            }
        }
    }
    if (a->f == form::slice && sl.data != nullptr && sl.size != 0) {
        ctx->free_shared(sl);
    }
    a->cpu_end[ith] = cpu_now();
    a->tcm_end[ith] = a->f == form::slice ? sl.data : view.data;
    a->done.fetch_add(1);
}

struct totals {
    int launches = 0, failed = 0, violations = 0, moved_tiles = 0, core_changes = 0, view_with_core = 0,
        view_without_core = 0, core_without_view = 0, lost_old = 0, lost_fresh = 0, torn = 0, alloc_failed = 0,
        slice_outside = 0, shared_core = 0, shared_tcm = 0;
    double us = 0.0;
};

void add(totals & t, const view_arg & a) {
    t.launches++;
    for (uint32_t x = 0; x < a.nth; x++) {
        t.moved_tiles += a.core_changes[x] > 0;
        t.core_changes += a.core_changes[x];
        t.view_with_core += a.view_with_core[x];
        t.view_without_core += a.view_without_core[x];
        t.core_without_view += a.core_without_view[x];
        t.lost_old += a.lost_old[x];
        t.lost_fresh += a.lost_fresh[x];
        t.torn += a.torn[x];
        t.alloc_failed += a.alloc_failed[x];
        t.slice_outside += a.slice_outside[x];
        g_cores.insert(a.cpu_end[x]);
    }
    t.shared_core += shares_core(a.cpu_end, a.nth);
    bool same = false;
    for (uint32_t x = 0; x < a.nth; x++) {
        for (uint32_t y = x + 1; y < a.nth; y++) {
            same = same || (a.tcm_end[x] != nullptr && a.tcm_end[x] == a.tcm_end[y]);
        }
    }
    t.shared_tcm += same;
}

void print(const char * shape, form f, int batch, int barriers, uint32_t nth, const totals & t) {
    const int tiles = t.launches * static_cast<int>(nth), points = tiles * barriers;
    std::printf("%-7s %-7s", shape, form_name(f));
    if (batch > 0) {
        std::printf(" %2d at a time", batch);
    } else {
        std::printf("             ");
    }
    std::printf(": %d launches x %d barriers: %.2f us per launch; failed %d; order violations %d; tiles that changed "
                "core %d of %d (core changes %d of %d barriers)",
                t.launches, barriers, t.us / t.launches, t.failed, t.violations, t.moved_tiles, tiles, t.core_changes,
                points);
    if (f == form::retake) {
        std::printf("; view changed with the core %d, without it %d; core changed with the view the same %d",
                    t.view_with_core, t.view_without_core, t.core_without_view);
    }
    std::printf("; canaries lost across a barrier: old pointer %d", t.lost_old);
    if (f == form::retake) {
        std::printf(", fresh view %d", t.lost_fresh);
    }
    if (f == form::slice) {
        std::printf(" (the slice); slices that failed to allocate %d, found outside the current worker's buffer %d",
                    t.alloc_failed, t.slice_outside);
    }
    std::printf("; torn with no suspension point %d; launches ending with two tiles on one core %d, on one TCM %d\n",
                t.torn, t.shared_core, t.shared_tcm);
}

}  // namespace

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const int    reps        = argc > 1 ? std::max(64, std::atoi(argv[1])) : 4000;
    const int    m_barriers  = argc > 2 ? std::max(1, std::atoi(argv[2])) : 4;
    const size_t slice_bytes = (argc > 3 ? std::max(1, std::atoi(argv[3])) : 16) * static_cast<size_t>(1024);

    const spert::BackendInfo bi = spert::backend_info();
    std::printf("spine-runtime: %zu cores, VLEN %zu, shared buffer %zu bytes, arch 0x%" PRIx64 "\n", bi.num_cores,
                bi.vlen * 8, bi.shared_mem_size, static_cast<uint64_t>(bi.core_arch_id));
    const uint32_t n = bi.num_cores > 0 && bi.num_cores <= MAX_TILES ? static_cast<uint32_t>(bi.num_cores) : 1;
    spert::Stream  s(n);
    if (!s.valid() || s.core_count() == 0) {
        std::printf("could not get %u AI cores from spine-runtime\n", n);
        return 1;
    }
    const uint32_t nth = static_cast<uint32_t>(s.core_count());
    std::printf("stream: %u cores; joined: %d launches per case, %d barriers each; slice %zu bytes\n", nth, reps,
                m_barriers, slice_bytes);

    std::atomic<int> failed{ 0 }, violations{ 0 };

    // single: one launch, K barriers, 20 launches per form (E3d's barriers case with the canaries)
    {
        const int k = 1000, launches = 20;
        for (const form f : { form::hold, form::retake, form::slice }) {
            totals t;
            for (int l = 0; l < launches; l++) {
                auto a         = std::make_unique<view_arg>();
                a->nth         = nth;
                a->id          = static_cast<uint32_t>(l);
                a->f           = f;
                a->barriers    = k;
                a->slice_bytes = slice_bytes;
                a->failed      = &failed;
                a->violations  = &violations;
                const clk::time_point t0 = clk::now();
                t.failed += count_failed(s.launch(spert::Grid{ nth }, tile_view, a.get()).sync());
                t.us += us_between(t0, clk::now());
                add(t, *a);
            }
            t.failed += failed.exchange(0);
            t.violations += violations.exchange(0);
            print("single", f, 0, k, nth, t);
        }
    }

    // joined: N launches queued and ordered in the cores, M barriers each
    std::printf("joined: running (a hang here: Ctrl-C, then tcmrelease --apply)\n");
    for (const int batch : { 8, 64 }) {
        for (const form f : { form::hold, form::retake, form::slice }) {
            const int rounds = std::max(1, reps / batch);
            totals    t;
            for (int r = 0; r < rounds; r++) {
                std::vector<std::unique_ptr<view_arg>> args;
                std::vector<spert::Future>             futures;
                args.reserve(batch);
                futures.reserve(batch);
                const clk::time_point t0 = clk::now();
                for (int i = 0; i < batch; i++) {
                    auto a         = std::make_unique<view_arg>();
                    a->nth         = nth;
                    a->id          = static_cast<uint32_t>(r * batch + i);
                    a->f           = f;
                    a->barriers    = m_barriers;
                    a->slice_bytes = slice_bytes;
                    a->failed      = &failed;
                    a->violations  = &violations;
                    if (!futures.empty()) {
                        a->previous.assign(nth, futures.back());
                        a->previous_done = &args.back()->done;
                    }
                    futures.push_back(s.launch(spert::Grid{ nth }, tile_view, a.get()));
                    args.push_back(std::move(a));
                }
                t.failed += count_failed(spert::Future::sync_all(futures));
                t.us += us_between(t0, clk::now());
                for (const auto & a : args) {
                    add(t, *a);
                }
            }
            t.failed += failed.exchange(0);
            t.violations += violations.exchange(0);
            print("joined", f, batch, m_barriers, nth, t);
        }
    }

    std::printf("cores the tiles ended on:");
    for (const int c : g_cores) {
        std::printf(" %d", c);
    }
    std::printf("\n");
    if (g_failed_status != 0) {
        std::printf("first failed status: %d\n", g_failed_status);
    }
    return g_failed_status != 0 ? 1 : 0;
}
