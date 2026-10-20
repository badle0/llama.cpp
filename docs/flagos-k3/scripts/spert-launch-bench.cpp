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
//                    also counts tiles left without TCM and launches whose tiles got the same TCM. Runs last: if it
//                    hangs (no new line for a minute), stop it with Ctrl-C and run tcmrelease --apply
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
    bool                          take_tcm = false;
    int                           cpus[MAX_TILES]     = {};
    const void *                  tcm[MAX_TILES]      = {};
    size_t                        tcm_size[MAX_TILES] = {};
    std::atomic<int> *            violations = nullptr;
    std::atomic<int> *            failed     = nullptr;
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
    a->cpus[ith] = cpu_now();
    if (a->take_tcm) {
        const spert::SharedBufferView buf = ctx->shared_buffer();
        a->tcm[ith]                       = buf.data;
        a->tcm_size[ith]                  = buf.data != nullptr ? buf.size : 0;
        if (ctx->sync() != spert::Status::Ok) {
            a->failed->fetch_add(1);
        }
    }
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

    for (const bool take_tcm : { false, true }) {
        if (take_tcm) {
            std::printf("joined+tcm: running (a hang here: Ctrl-C, then tcmrelease --apply)\n");
        }
        for (const int batch : { 8, 64 }) {
            const int        rounds = std::max(1, reps / batch);
            std::atomic<int> violations{ 0 }, failed{ 0 };
            double           total = 0.0, calls = 0.0;
            int              shared = 0, no_tcm = 0, same_tcm = 0;
            for (int r = 0; r < rounds; r++) {
                std::vector<std::unique_ptr<joined_arg>> args;
                std::vector<spert::Future>               futures;
                args.reserve(batch);
                futures.reserve(batch);
                const clk::time_point t0 = clk::now();
                for (int i = 0; i < batch; i++) {
                    auto a        = std::make_unique<joined_arg>();
                    a->nth        = nth;
                    a->take_tcm   = take_tcm;
                    a->violations = &violations;
                    a->failed     = &failed;
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
                for (const auto & a : args) {
                    shared += shares_core(a->cpus, nth);
                    note_cores(a->cpus, nth);
                    if (!take_tcm) {
                        continue;
                    }
                    bool same = false;
                    for (uint32_t x = 0; x < nth; x++) {
                        no_tcm += a->tcm[x] == nullptr || a->tcm_size[x] == 0;
                        for (uint32_t y = x + 1; y < nth; y++) {
                            same = same || (a->tcm[x] != nullptr && a->tcm[x] == a->tcm[y]);
                        }
                    }
                    same_tcm += same;
                }
            }
            const int n_launches = rounds * batch;
            std::printf("%-16s grid %u, %d at a time: %d launches: %.2f us per launch (the host's launch calls %.2f us "
                        "each); failed %d; order violations %d; launches with two tiles on one core %d",
                        take_tcm ? "joined+tcm" : "joined", nth, batch, n_launches, total / n_launches,
                        calls / n_launches, failed.load(), violations.load(), shared);
            if (take_tcm) {
                std::printf("; tiles without TCM %d; launches with two tiles on the same TCM %d", no_tcm, same_tcm);
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
