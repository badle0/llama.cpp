// E3d (structural-experiments.md): what spine-runtime itself costs per launch, with no provider and no ggml, and
// whether its launches keep one tile per core. Every case runs tiles that do (almost) nothing, on all the AI cores:
//   launch+sync      one launch, the host waiting for it (Future::sync); E3a pays this per step, plus our own setup
//   launch+tcm+sync  the same, each tile taking its shared buffer (TCM) and passing one barrier: our per-launch setup
//   barriers         one launch, every tile passing K barriers (Context::sync): today's executor's cost per step;
//                    counts tiles found on another core than at the start, and launches with two tiles on one core
//   queued           N launches queued, the host waiting once (Future::sync_all): launches left unordered
//   joined           N launches queued, every tile first joining its own copy of the previous launch's Future
//                    (Context::join), the host waiting once: E3b's ordering, with tiles that cannot break it; counts
//                    order violations (a tile that found the previous launch unfinished after its join) and launches
//                    with two tiles on one core
//   joined+tcm       as joined, every tile then taking its shared buffer and passing one barrier, as E3b's tiles do;
//                    also counts tiles left without TCM and launches whose tiles got the same TCM
//   joined+canary    E3b-diag: as joined+tcm, every tile writing a pattern of its own into its TCM before the barrier
//                    and checking it after (did another tile use this TCM while this one waited?), then writing it again
//                    and checking it four times with no suspension point in between (did another tile use it at the
//                    same time?); also counts tiles that changed core between the join, taking TCM, the barrier and
//                    the end, and launches whose two tiles on different cores held one TCM
// The joined cases run last: if one hangs (no new line for a minute), stop it with Ctrl-C and run tcmrelease --apply
// Build on the K3: g++ -O2 -std=c++17 -I$HOME/spine-runtime/include spert-launch-bench.cpp -L$HOME/spine-runtime/lib \
//                      -lspert -o spert-launch-bench
// Run (inside tmux, nothing else on the AI cores): ~/tcmtest/tcmrelease --apply
//                  LD_LIBRARY_PATH=$HOME/spine-runtime/lib ./spert-launch-bench [launches per case, default 20000]
#include <spert.hpp>

#if defined(__linux__)
#    include <sched.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <vector>

namespace {

using clk                    = std::chrono::steady_clock;
constexpr uint32_t MAX_TILES = 64;

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

// two tiles of one launch on the same core
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

std::set<int> g_cores;  // every core a tile reported
int           g_failed_status = 0;

void note_cores(const int * cpus, uint32_t n) {
    g_cores.insert(cpus, cpus + n);
}

int count_failed(spert::Status st) {
    if (st == spert::Status::Ok) {
        return 0;
    }
    if (g_failed_status == 0) {
        g_failed_status = spert::status_code(st);
    }
    return 1;
}

void print_times(const char * name, uint32_t grid, std::vector<double> t, int failed) {
    std::sort(t.begin(), t.end());
    double sum = 0.0;
    for (double x : t) {
        sum += x;
    }
    auto at = [&](double q) { return t[std::min(t.size() - 1, static_cast<size_t>(q * (t.size() - 1) + 0.5))]; };
    std::printf("%-16s grid %u: %zu launches: median %.2f us (p10 %.2f, p90 %.2f, mean %.2f, max %.2f); failed %d\n", name,
                grid, t.size(), at(0.5), at(0.1), at(0.9), sum / t.size(), t.back(), failed);
}

void tile_empty(spert::Context *, int *) noexcept {}

void tile_tcm(spert::Context * ctx, std::atomic<int> * failed) noexcept {
    const spert::SharedBufferView buf = ctx->shared_buffer();
    static_cast<void>(buf);
    if (ctx->sync() != spert::Status::Ok) {
        failed->fetch_add(1);
    }
}

struct barrier_arg {
    int                k          = 0;
    int                start[MAX_TILES] = {};
    int                mid[MAX_TILES]   = {};
    int                end[MAX_TILES]   = {};
    double             us         = 0.0;  // tile 0: from the first barrier to the last
    std::atomic<int> * failed     = nullptr;
};

void tile_barriers(spert::Context * ctx, barrier_arg * a) noexcept {
    const uint32_t ith = ctx->program_id(0);
    a->start[ith]      = cpu_now();
    if (ctx->sync() != spert::Status::Ok) {
        a->failed->fetch_add(1);
        return;
    }
    const clk::time_point t0 = clk::now();
    for (int i = 0; i < a->k; i++) {
        if (ctx->sync() != spert::Status::Ok) {
            a->failed->fetch_add(1);
            return;
        }
        if (i == a->k / 2) {
            a->mid[ith] = cpu_now();
        }
    }
    if (ith == 0) {
        a->us = us_between(t0, clk::now());
    }
    a->end[ith] = cpu_now();
}

struct queued_arg {
    int cpus[MAX_TILES] = {};
};

void tile_queued(spert::Context * ctx, queued_arg * a) noexcept {
    a->cpus[ctx->program_id(0)] = cpu_now();
}

struct joined_arg {
    std::vector<spert::Future>    previous;                // one copy per tile; none for the first launch
    const std::atomic<uint32_t> * previous_done = nullptr;  // the previous launch's finished tiles
    std::atomic<uint32_t>         done{ 0 };
    uint32_t                      nth      = 0;
    uint32_t                      id       = 0;  // the launch's number, in the canaries
    bool                          take_tcm = false;
    bool                          canary   = false;
    int                           cpus[4][MAX_TILES]  = {};  // after the join, after taking TCM, after the barrier, at the end
    const void *                  tcm[MAX_TILES]      = {};
    size_t                        tcm_size[MAX_TILES] = {};
    std::atomic<int> *            violations = nullptr;
    std::atomic<int> *            failed     = nullptr;
    std::atomic<int> *            lost       = nullptr;  // canary written before the barrier, changed after it
    std::atomic<int> *            torn       = nullptr;  // canary changed with no suspension point in between
};

void tile_joined(spert::Context * ctx, joined_arg * a) noexcept {
    const uint32_t ith = ctx->program_id(0);
    if (!a->previous.empty()) {
        if (ctx->join(a->previous[ith]) != spert::Status::Ok) {
            a->failed->fetch_add(1);
        } else if (a->previous_done->load() != a->nth) {
            a->violations->fetch_add(1);
        }
    }
    a->cpus[0][ith] = cpu_now();
    if (a->take_tcm) {
        const spert::SharedBufferView buf = ctx->shared_buffer();
        a->tcm[ith]                       = buf.data;
        a->tcm_size[ith]                  = buf.data != nullptr ? buf.size : 0;
        a->cpus[1][ith]                   = cpu_now();
        // canaries: up to 1024 words of the tile's TCM, a pattern of its own, read back through scalar loads
        volatile uint32_t * w     = static_cast<volatile uint32_t *>(buf.data);
        const size_t        words = a->canary && w != nullptr ? std::min<size_t>(buf.size / 4, 1024) : 0;
        const uint32_t      mine  = (a->id << 8) | ith;
        for (size_t j = 0; j < words; j++) {
            w[j] = mine ^ 0xa5a5a5a5u;
        }
        if (ctx->sync() != spert::Status::Ok) {
            a->failed->fetch_add(1);
        }
        a->cpus[2][ith] = cpu_now();
        bool kept = true;  // across the barrier, where the tile may have been suspended
        for (size_t j = 0; j < words; j++) {
            kept = kept && w[j] == (mine ^ 0xa5a5a5a5u);
        }
        for (size_t j = 0; j < words; j++) {
            w[j] = mine;
        }
        bool intact = true;  // four passes with no suspension point: only a tile running at the same time can change it
        for (int pass = 0; pass < 4; pass++) {
            for (size_t j = 0; j < words; j++) {
                intact = intact && w[j] == mine;
            }
        }
        if (words > 0 && !kept) {
            a->lost->fetch_add(1);
        }
        if (words > 0 && !intact) {
            a->torn->fetch_add(1);
        }
    }
    a->cpus[3][ith] = cpu_now();
    a->done.fetch_add(1);
}

}  // namespace

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // each line out at once, also into a pipe: a hang shows where
    const int reps = argc > 1 ? std::max(64, std::atoi(argv[1])) : 20000;

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
    std::printf("stream: %u cores; %d launches per case\n", nth, reps);

    int dummy = 0;
    for (int i = 0; i < 100; i++) {  // warm-up
        s.launch(spert::Grid{ nth }, tile_empty, &dummy).sync();
    }

    for (const uint32_t grid : { nth, 1u }) {
        std::vector<double> t;
        t.reserve(reps);
        int failed = 0;
        for (int i = 0; i < reps; i++) {
            const clk::time_point t0 = clk::now();
            failed += count_failed(s.launch(spert::Grid{ grid }, tile_empty, &dummy).sync());
            t.push_back(us_between(t0, clk::now()));
        }
        print_times("launch+sync", grid, t, failed);
    }

    {
        std::atomic<int>    failed{ 0 };
        std::vector<double> t;
        t.reserve(reps);
        for (int i = 0; i < reps; i++) {
            const clk::time_point t0 = clk::now();
            failed += count_failed(s.launch(spert::Grid{ nth }, tile_tcm, &failed).sync());
            t.push_back(us_between(t0, clk::now()));
        }
        print_times("launch+tcm+sync", nth, t, failed.load());
    }

    {
        const int           k = 20000, launches = 20;
        std::atomic<int>    failed{ 0 };
        std::vector<double> per_barrier;
        int                 moved = 0, shared = 0;
        for (int l = 0; l < launches; l++) {
            barrier_arg a;
            a.k      = k;
            a.failed = &failed;
            failed += count_failed(s.launch(spert::Grid{ nth }, tile_barriers, &a).sync());
            per_barrier.push_back(a.us / k);
            for (uint32_t t = 0; t < nth; t++) {
                moved += a.start[t] != a.mid[t] || a.start[t] != a.end[t];
            }
            shared += shares_core(a.start, nth) || shares_core(a.mid, nth) || shares_core(a.end, nth);
            note_cores(a.start, nth);
        }
        std::sort(per_barrier.begin(), per_barrier.end());
        std::printf("%-16s grid %u: %d barriers x %d launches: median %.3f us per barrier (min %.3f, max %.3f); "
                    "failed %d; tiles that changed core %d of %u; launches with two tiles on one core %d of %d\n",
                    "barriers", nth, k, launches, per_barrier[per_barrier.size() / 2], per_barrier.front(),
                    per_barrier.back(), failed.load(), moved, nth * launches, shared, launches);
    }

    for (const int batch : { 8, 64 }) {
        const int               rounds = std::max(1, reps / batch);
        std::vector<queued_arg> args(batch);
        double                  total = 0.0, calls = 0.0;
        int                     failed = 0, shared = 0;
        for (int r = 0; r < rounds; r++) {
            std::vector<spert::Future> futures;
            futures.reserve(batch);
            const clk::time_point t0 = clk::now();
            for (int i = 0; i < batch; i++) {
                futures.push_back(s.launch(spert::Grid{ nth }, tile_queued, &args[i]));
            }
            const clk::time_point t1 = clk::now();
            failed += count_failed(spert::Future::sync_all(futures));
            const clk::time_point t2 = clk::now();
            calls += us_between(t0, t1);
            total += us_between(t0, t2);
            for (const queued_arg & a : args) {
                shared += shares_core(a.cpus, nth);
                note_cores(a.cpus, nth);
            }
        }
        const int n_launches = rounds * batch;
        std::printf("%-16s grid %u, %d at a time: %d launches: %.2f us per launch (the host's launch calls %.2f us "
                    "each); failed %d; launches with two tiles on one core %d\n",
                    "queued", nth, batch, n_launches, total / n_launches, calls / n_launches, failed, shared);
    }

    // joined: the tiles only join; +tcm: they also take their TCM and pass a barrier, as E3b's do; +canary: they also
    // check their TCM (E3b-diag)
    for (const int mode : { 0, 1, 2 }) {
        const bool take_tcm = mode > 0, canary = mode == 2;
        if (take_tcm) {
            std::printf("%s: running (a hang here: Ctrl-C, then tcmrelease --apply)\n", canary ? "joined+canary" : "joined+tcm");
        }
        for (const int batch : { 8, 64 }) {
            const int        rounds = std::max(1, reps / batch);
            std::atomic<int> violations{ 0 }, failed{ 0 }, lost{ 0 }, torn{ 0 };
            double           total = 0.0, calls = 0.0;
            int              shared = 0, no_tcm = 0, same_tcm = 0, moved = 0, elsewhere = 0;
            for (int r = 0; r < rounds; r++) {
                std::vector<std::unique_ptr<joined_arg>> args;
                std::vector<spert::Future>               futures;
                args.reserve(batch);
                futures.reserve(batch);
                const clk::time_point t0 = clk::now();
                for (int i = 0; i < batch; i++) {
                    auto a        = std::make_unique<joined_arg>();
                    a->nth        = nth;
                    a->id         = static_cast<uint32_t>(r * batch + i);
                    a->take_tcm   = take_tcm;
                    a->canary     = canary;
                    a->violations = &violations;
                    a->failed     = &failed;
                    a->lost       = &lost;
                    a->torn       = &torn;
                    if (!futures.empty()) {
                        a->previous.assign(nth, futures.back());
                        a->previous_done = &args.back()->done;
                    }
                    futures.push_back(s.launch(spert::Grid{ nth }, tile_joined, a.get()));
                    args.push_back(std::move(a));
                }
                const clk::time_point t1 = clk::now();
                failed += count_failed(spert::Future::sync_all(futures));
                const clk::time_point t2 = clk::now();
                calls += us_between(t0, t1);
                total += us_between(t0, t2);
                // the points each tile noted: after the join and at the end; with TCM also after taking it and after
                // the barrier
                const std::vector<int> points = take_tcm ? std::vector<int>{ 0, 1, 2, 3 } : std::vector<int>{ 0, 3 };
                for (const auto & a : args) {
                    bool twice = false;
                    for (const int p : points) {
                        twice = twice || shares_core(a->cpus[p], nth);
                        note_cores(a->cpus[p], nth);
                    }
                    shared += twice;
                    for (uint32_t x = 0; x < nth; x++) {
                        for (size_t p = 1; p < points.size(); p++) {
                            if (a->cpus[points[p]][x] != a->cpus[points[p - 1]][x]) {
                                moved++;
                                break;
                            }
                        }
                    }
                    if (!take_tcm) {
                        continue;
                    }
                    bool same = false, other_core = false;
                    for (uint32_t x = 0; x < nth; x++) {
                        no_tcm += a->tcm[x] == nullptr || a->tcm_size[x] == 0;
                        for (uint32_t y = x + 1; y < nth; y++) {
                            const bool one_tcm = a->tcm[x] != nullptr && a->tcm[x] == a->tcm[y];
                            same               = same || one_tcm;
                            other_core         = other_core || (one_tcm && a->cpus[2][x] != a->cpus[2][y]);
                        }
                    }
                    same_tcm += same;
                    elsewhere += other_core;
                }
            }
            const int n_launches = rounds * batch;
            std::printf("%-16s grid %u, %d at a time: %d launches: %.2f us per launch (the host's launch calls %.2f us "
                        "each); failed %d; order violations %d; launches with two tiles on one core %d; tiles that "
                        "changed core %d",
                        mode == 0 ? "joined" : canary ? "joined+canary" : "joined+tcm", nth, batch, n_launches,
                        total / n_launches, calls / n_launches, failed.load(), violations.load(), shared, moved);
            if (take_tcm) {
                std::printf("; tiles without TCM %d; launches with two tiles on the same TCM %d (on two cores %d)",
                            no_tcm, same_tcm, elsewhere);
            }
            if (canary) {
                std::printf("; tiles whose TCM changed across their barrier %d, with no suspension point %d", lost.load(),
                            torn.load());
            }
            std::printf("\n");
        }
    }

    std::printf("cores the tiles ran on:");
    for (const int c : g_cores) {
        std::printf(" %d", c);
    }
    std::printf("\n");
    if (g_failed_status != 0) {
        std::printf("first failed status: %d\n", g_failed_status);
    }
    return g_failed_status != 0 ? 1 : 0;
}
