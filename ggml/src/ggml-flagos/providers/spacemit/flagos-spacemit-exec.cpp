#include "flagos-spacemit-exec.h"

#include "../../../ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(GGML_FLAGOS_SPACEMIT_SPERT)
#    include <spert.hpp>
#    if defined(__linux__)
#        include <sched.h>
#    endif
#else
#    include <condition_variable>
#    include <thread>
#endif

// the pair waits of one launch (spacemit_pair_wait). The counts only grow during the launch, so the waits of all its
// steps continue one count
struct spacemit_pairs {
    struct alignas(64) counter {  // one cache line each: the two tiles of a pair spin on theirs
        std::atomic<uint32_t> calls{ 0 };
    };
    counter                   by_pair[SPACEMIT_MAX_TILES / 2];  // the calls of both tiles of the pair
    uint32_t                  by_tile[SPACEMIT_MAX_TILES] = {};  // each tile's own calls, written only by that tile
    const std::atomic<bool> * failed     = nullptr;             // the launch's failure flag
    bool                      concurrent = true;                // false: the tiles run one after another; only count
    // test-only (structural-experiments.md E7b): the waits yield the core instead of spinning, so that two tiles of one
    // launch on one worker can take turns (ordered launches, where the runtime may put them there); null: spin
    void (*yield)(void * arg)                        = nullptr;
    void *                    yield_arg[SPACEMIT_MAX_TILES] = {};  // each tile's spert::Context, set at its entry
};

bool spacemit_pair_wait(const spacemit_tile & tile) {
    GGML_ASSERT(tile.pairs != nullptr);
    spacemit_pairs & p = *tile.pairs;
    const uint32_t   n = ++p.by_tile[tile.ith];
    if (!p.concurrent) {
        return true;
    }
    std::atomic<uint32_t> & both = p.by_pair[tile.ith / 2].calls;
    both.fetch_add(1, std::memory_order_acq_rel);
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t spin = 1; both.load(std::memory_order_acquire) < 2 * n; spin++) {
        if (p.failed->load(std::memory_order_relaxed)) {
            return false;
        }
        if (spin % 65536 == 0 && std::chrono::steady_clock::now() - start > std::chrono::seconds(10)) {
            GGML_LOG_ERROR("FlagOS SpacemiT: tile %u waited 10 s for tile %u at pair wait %u\n", tile.ith, tile.ith ^ 1u, n);
            return false;
        }
        if (p.yield != nullptr) {
            p.yield(p.yield_arg[tile.ith]);
            continue;
        }
#if defined(__riscv_zihintpause)
        __asm__ __volatile__("pause");
#endif
    }
    return true;
}

namespace {

// test-only: FLAGOS_SPACEMIT_TEST_FAIL_NODE=n makes tile 0 report a failure at step n
int64_t spacemit_test_fail_step() {
    const char * value = std::getenv("FLAGOS_SPACEMIT_TEST_FAIL_NODE");
    return value != nullptr ? std::atoll(value) : -1;
}

// test-only (structural-experiments.md E3b-diag): each tile's core after its join, after taking its TCM, after the TCM
// agreement and after its last step, and the TCM it took
struct spacemit_tile_trace {
    int          cpu[4][SPACEMIT_MAX_TILES] = {};
    const void * tcm[SPACEMIT_MAX_TILES]    = {};
};

std::mutex              g_ordered_mutex;
spacemit_ordered_counts g_ordered{};
std::atomic<uint64_t>   g_steps_launched{ 0 };  // test-only (spacemit_steps_take)

struct spacemit_launch {
    const spacemit_plugin_step * steps          = nullptr;
    size_t                       n_steps        = 0;
    void *                       workspace      = nullptr;
    size_t                       workspace_size = 0;
    int64_t                      fail_step      = -1;
    std::atomic<bool>            failed{ false };
    bool                         retake_view    = true;  // the view taken again after every barrier (spacemit_tile_main)
    size_t                       tcm_slice      = 0;     // test-only (E7b): each tile's alloc_shared slice in bytes; 0: the whole buffer
    size_t                tcm_sizes[SPACEMIT_MAX_TILES] = {};  // each tile's TCM, so all can agree on one size
    spacemit_pairs        pairs;
    spacemit_run_stats *  stats      = nullptr;  // test-only (E3c): where the tiles note their times, if anywhere
    size_t                stats_base = 0;        // the first step of this launch in stats
    spacemit_tile_trace * trace      = nullptr;  // test-only (E3b-diag): where ordered launches note their cores

    spacemit_launch() { pairs.failed = &failed; }
};

bool spacemit_run_step(const spacemit_launch & launch, size_t i, const spacemit_tile & tile) {
    const spacemit_plugin_step & step = launch.steps[i];
    spacemit_tile                part = tile;  // the step's part of the workspace
    part.workspace                    = static_cast<uint8_t *>(tile.workspace) + step.workspace_offset;
    part.workspace_size               = tile.workspace_size - step.workspace_offset;
    const bool ok                     = step.fn(&part, step.node);
    return ok && !(static_cast<int64_t>(i) == launch.fail_step && tile.ith == 0);
}

// test-only (structural-experiments.md E3c): the tiles read the clock only when their launch keeps statistics. A tile
// writes its kernel times into its own row of stats->kernel_ns and its barrier time once, at the end of the launch, so
// the tiles do not share cache lines step by step
using spacemit_clock = std::chrono::steady_clock;

spacemit_clock::time_point spacemit_stats_now(const spacemit_launch & launch) {
    return launch.stats != nullptr ? spacemit_clock::now() : spacemit_clock::time_point{};
}

uint64_t spacemit_ns(spacemit_clock::duration d) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
}

// tile ith spent t0 to t1 in step i's kernel and t1 to t2 at the barrier after it; returns the barrier time
uint64_t spacemit_stats_step(const spacemit_launch & launch, size_t i, uint32_t ith, spacemit_clock::time_point t0,
                             spacemit_clock::time_point t1, spacemit_clock::time_point t2) {
    spacemit_run_stats * stats   = launch.stats;
    const size_t         at      = ith * stats->stride + launch.stats_base + i;
    const uint64_t       barrier = spacemit_ns(t2 - t1);
    stats->kernel_ns[at]         = static_cast<uint32_t>(spacemit_ns(t1 - t0));
    stats->step_barrier_ns[at]   = static_cast<uint32_t>(barrier);
    return barrier;
}

// before a launch of nth tiles: its steps follow those of the graph's earlier launches
void spacemit_stats_begin(spacemit_launch & launch, spacemit_run_stats * stats, uint32_t nth) {
    launch.stats = stats;
    if (stats == nullptr) {
        return;
    }
    GGML_ASSERT(stats->n_steps + launch.n_steps <= stats->stride && nth <= SPACEMIT_MAX_TILES);
    stats->n_tiles    = nth;
    launch.stats_base = stats->n_steps;
    stats->n_steps += launch.n_steps;
}

// after the host's wait for the launch, which it started at t
void spacemit_stats_end(const spacemit_launch & launch, spacemit_clock::time_point t) {
    if (launch.stats != nullptr) {
        launch.stats->launches++;
        launch.stats->launch_us += std::chrono::duration<double, std::micro>(spacemit_clock::now() - t).count();
    }
}

#if defined(GGML_FLAGOS_SPACEMIT_SPERT)

int spacemit_cpu_now() {
#    if defined(__linux__)
    return sched_getcpu();
#    else
    return -1;
#    endif
}

// every tile reaches every barrier, also after a failure, when it only skips the remaining kernels: a tile that left
// early would leave the others waiting at the next barrier forever. (Leaving when the flag is set after a barrier is
// not enough: a faster tile may already have failed in the next step, so tiles would disagree.)
void spacemit_tile_main(spert::Context * ctx, spacemit_launch * launch) noexcept {
    const uint32_t ith = ctx->program_id(0);
    const uint32_t nth = ctx->grid_dim(0);
    launch->pairs.yield_arg[ith] = ctx;
    // the tile's TCM: the worker's whole shared buffer, which the header grants only while the tile runs on that
    // worker (spert.hpp:474-478), so it is taken again after every barrier below (E7; FLAGOS_SPACEMIT_TEST_VIEW=once
    // keeps the first view throughout, the loop until 2026-10-30); or, test-only (E7b, FLAGOS_SPACEMIT_TEST_TCM_SLICE),
    // a slice from alloc_shared, the tile's own until free_shared at its end
    const bool              slice = launch->tcm_slice != 0;
    spert::SharedBufferView tcm   = slice ? ctx->alloc_shared(launch->tcm_slice, 64) : ctx->shared_buffer();
    if (slice && (tcm.data == nullptr || tcm.size == 0)) {
        launch->failed.store(true);  // no slice: the launch fails; the tile still reaches every barrier
        tcm = spert::SharedBufferView{};
    }
    const auto release = [&] {
        if (slice && tcm.data != nullptr && tcm.size != 0) {
            ctx->free_shared(tcm);
        }
    };
    if (launch->trace != nullptr) {  // test-only (E3b-diag)
        launch->trace->cpu[1][ith] = spacemit_cpu_now();
        launch->trace->tcm[ith]    = tcm.data;
    }

    // agree on one TCM size, the smallest any tile has, so every tile takes the same kernel path
    launch->tcm_sizes[ith] = tcm.data != nullptr ? tcm.size : 0;
    if (ctx->sync() != spert::Status::Ok) {
        launch->failed.store(true);  // spine-runtime's barrier itself failed: nothing can keep the tiles together
        release();
        return;
    }
    if (launch->trace != nullptr) {
        launch->trace->cpu[2][ith] = spacemit_cpu_now();
    }
    size_t tcm_size = launch->tcm_sizes[0];
    for (uint32_t i = 1; i < nth; i++) {
        tcm_size = std::min(tcm_size, launch->tcm_sizes[i]);
    }
    spacemit_tile tile = { ith, nth, tcm_size != 0 ? tcm.data : nullptr, tcm_size, launch->workspace,
                           launch->workspace_size, &launch->pairs };

    uint64_t barrier_ns = 0;  // test-only (E3c)
    for (size_t i = 0; i < launch->n_steps; i++) {
        const spacemit_clock::time_point t0 = spacemit_stats_now(*launch);
        if (!launch->failed.load() && !spacemit_run_step(*launch, i, tile)) {
            launch->failed.store(true);
        }
        const spacemit_clock::time_point t1     = spacemit_stats_now(*launch);
        const spert::Status              status = ctx->sync();
        if (launch->stats != nullptr) {
            barrier_ns += spacemit_stats_step(*launch, i, ith, t0, t1, spacemit_clock::now());
        }
        if (status != spert::Status::Ok) {
            launch->failed.store(true);
            release();
            return;
        }
        // the barrier suspended the tile: the view is valid only while the tile runs on the worker it took it from,
        // so it is taken again (a slice needs no re-take: it is the tile's own). A worker whose buffer is smaller than
        // the agreed size cannot run the kernels' paths: the launch fails and the tile skips its kernels
        if (tile.tcm != nullptr && !slice && launch->retake_view) {
            const spert::SharedBufferView fresh = ctx->shared_buffer();
            if (fresh.data == nullptr || fresh.size < tcm_size) {
                if (!launch->failed.exchange(true)) {
                    GGML_LOG_ERROR("FlagOS SpacemiT: tile %u: the shared buffer after step %zu holds %zu bytes, %zu "
                                   "agreed at the launch's start\n",
                                   ith, i, fresh.data != nullptr ? fresh.size : (size_t) 0, tcm_size);
                }
            } else {
                tile.tcm = fresh.data;
            }
        }
    }
    release();
    if (launch->stats != nullptr) {
        launch->stats->barrier_ns[ith] += barrier_ns;
    }
    if (launch->trace != nullptr) {
        launch->trace->cpu[3][ith] = spacemit_cpu_now();
    }
}

// test-only (structural-experiments.md E3b): one launch per step, ordered inside the cores. spine-runtime does not
// order the launches of a stream (dependent kernels wait for each other's Future), so every tile first joins its own
// copy of the previous launch's Future, before it takes its core's TCM. A tile that waits is suspended and its core
// may run another tile; two tiles of one launch on one core could deadlock at the TCM agreement or a pair wait, so
// every tile notes its cores and its TCM (E3b-diag), and the host counts what they show
struct spacemit_ordered_launch : spacemit_launch {
    std::vector<spert::Future> previous;  // one copy per tile; none for a graph's first launch
    spacemit_tile_trace        tile_trace;
    bool                       fence = false;  // E3b-diag: FLAGOS_SPACEMIT_TEST_JOIN_FENCE=1, full fences around it
};

void spacemit_tile_main_ordered(spert::Context * ctx, spacemit_ordered_launch * launch) noexcept {
    const uint32_t ith = ctx->program_id(0);
    if (!launch->previous.empty() && ctx->join(launch->previous[ith]) != spert::Status::Ok) {
        launch->failed.store(true);  // the tile still reaches every barrier, skipping its kernel
    }
    if (launch->fence) {
        std::atomic_thread_fence(std::memory_order_seq_cst);  // after the join: the previous launch's writes
    }
    launch->tile_trace.cpu[0][ith] = spacemit_cpu_now();
    spacemit_tile_main(ctx, launch);
    if (launch->fence) {
        std::atomic_thread_fence(std::memory_order_seq_cst);  // this launch's writes, before its Future completes
    }
}

#    if defined(__linux__)
// what the tiles of each ordered launch noted (spacemit_ordered_counts), added to the counts since the last take;
// returns the launches with two tiles on one core
size_t spacemit_count_ordered(const std::vector<std::unique_ptr<spacemit_ordered_launch>> & launches, uint32_t nth) {
    spacemit_ordered_counts c{};
    for (const auto & launch : launches) {
        const spacemit_tile_trace & t = launch->tile_trace;
        bool                        shared = false, pair = false, elsewhere = false;
        for (uint32_t a = 0; a < nth; a++) {
            for (int p = 1; p < 4; p++) {
                if (t.cpu[p][a] != t.cpu[p - 1][a]) {
                    c.moved_tiles++;
                    break;
                }
            }
            for (uint32_t b = a + 1; b < nth; b++) {
                bool same_core = false;
                for (int p = 0; p < 4; p++) {
                    same_core = same_core || t.cpu[p][a] == t.cpu[p][b];
                }
                shared    = shared || same_core;
                pair      = pair || (same_core && b == (a ^ 1u));
                elsewhere = elsewhere || (t.tcm[a] != nullptr && t.tcm[a] == t.tcm[b] && t.cpu[2][a] != t.cpu[2][b]);
            }
        }
        c.launches++;
        c.shared_core += shared;
        c.pairs_on_one_core += pair;
        c.tcm_elsewhere += elsewhere;
    }
    std::lock_guard<std::mutex> lock(g_ordered_mutex);
    g_ordered.launches += c.launches;
    g_ordered.shared_core += c.shared_core;
    g_ordered.moved_tiles += c.moved_tiles;
    g_ordered.tcm_elsewhere += c.tcm_elsewhere;
    g_ordered.pairs_on_one_core += c.pairs_on_one_core;
    return c.shared_core;
}
#    endif

// test-only (E7b): the pair waits of ordered launches yield the core (spacemit_pairs::yield)
void spacemit_yield_core(void * arg) {
    static_cast<spert::Context *>(arg)->yield();
}

class spert_executor final : public spacemit_executor {
  public:
    spert_executor(spacemit_stream_policy policy, uint32_t n_cores, int64_t fail_step) :
        policy(policy), n_cores(n_cores), fail_step(fail_step) {
        // FLAGOS_SPACEMIT_TEST_VIEW=once: the first view kept through every barrier (the loop until 2026-10-30, for the
        // comparison); step, or unset: the view taken again after every barrier (spacemit_tile_main)
        const char * view = std::getenv("FLAGOS_SPACEMIT_TEST_VIEW");
        retake_view       = view == nullptr || std::strcmp(view, "once") != 0;
        if (view != nullptr && std::strcmp(view, "once") != 0 && std::strcmp(view, "step") != 0) {
            GGML_LOG_WARN("FlagOS SpacemiT: FLAGOS_SPACEMIT_TEST_VIEW=%s: expected once or step; the view is taken again "
                          "after every barrier\n", view);
        }
        // test-only (E7b): FLAGOS_SPACEMIT_TEST_TCM_SLICE=<bytes>: each tile takes a slice of that size from its worker's
        // shared buffer (alloc_shared) as its TCM instead of the whole buffer, in every launch mode
        const char * slice = std::getenv("FLAGOS_SPACEMIT_TEST_TCM_SLICE");
        tcm_slice          = slice != nullptr ? (size_t) std::max(0LL, std::atoll(slice)) : 0;
        if (tcm_slice != 0) {
            GGML_LOG_WARN("FlagOS SpacemiT: test-only: each tile takes a %zu-byte alloc_shared slice as its TCM "
                          "(FLAGOS_SPACEMIT_TEST_TCM_SLICE)\n", tcm_slice);
        }
    }

    bool run(const spacemit_plugin_step * steps, size_t n_steps, void * workspace, size_t workspace_size) override {
        std::unique_ptr<spert::Stream> per_call_stream;
        spert::Stream *                stream = stream_for_run(per_call_stream);
        if (stream == nullptr) {
            return false;
        }

        spacemit_launch launch;
        launch.steps          = steps;
        launch.n_steps        = n_steps;
        launch.workspace      = workspace;
        launch.workspace_size = workspace_size;
        launch.fail_step      = fail_step;
        launch.retake_view    = retake_view;
        launch.tcm_slice      = tcm_slice;
        g_steps_launched += n_steps;
        spacemit_stats_begin(launch, stats, static_cast<uint32_t>(stream->core_count()));
        const spacemit_clock::time_point t      = spacemit_stats_now(launch);
        const spert::Future              future = stream->launch(spert::Grid{ stream->core_count() }, spacemit_tile_main,
                                                                 &launch);
        const spert::Status              status = future.sync();
        spacemit_stats_end(launch, t);
        if (status != spert::Status::Ok) {
            GGML_LOG_ERROR("FlagOS SpacemiT: spine-runtime launch failed with status %d\n", spert::status_code(status));
            return false;
        }
        return !launch.failed.load();
    }

    // every step its own launch, queued without the host waiting, each ordered after the previous by its tiles; the
    // host waits once, for all. Each launch's arguments stay alive until then
    bool run_ordered(const spacemit_plugin_step * steps, size_t n_steps, void * workspace,
                     size_t workspace_size) override {
        std::unique_ptr<spert::Stream> per_call_stream;
        spert::Stream *                stream = stream_for_run(per_call_stream);
        if (stream == nullptr) {
            return false;
        }
        const uint32_t nth = static_cast<uint32_t>(stream->core_count());
        const char *   fence_value = std::getenv("FLAGOS_SPACEMIT_TEST_JOIN_FENCE");  // test-only (E3b-diag)
        const bool     fence       = fence_value != nullptr && std::strcmp(fence_value, "0") != 0;

        std::vector<std::unique_ptr<spacemit_ordered_launch>> launches;
        std::vector<spert::Future>                             futures;
        launches.reserve(n_steps);
        futures.reserve(n_steps);
        g_steps_launched += n_steps;
        for (size_t s = 0; s < n_steps; s++) {
            auto launch            = std::make_unique<spacemit_ordered_launch>();
            launch->steps          = steps + s;
            launch->n_steps        = 1;
            launch->workspace      = workspace;
            launch->workspace_size = workspace_size;
            launch->fail_step      = fail_step;
            launch->retake_view    = retake_view;
            launch->tcm_slice      = tcm_slice;
            launch->pairs.yield    = spacemit_yield_core;  // E7b: two tiles of one launch on one worker take turns
            launch->trace          = &launch->tile_trace;
            launch->fence          = fence;
            if (!futures.empty()) {
                launch->previous.assign(nth, futures.back());
            }
            futures.push_back(stream->launch(spert::Grid{ stream->core_count() }, spacemit_tile_main_ordered,
                                             launch.get()));
            launches.push_back(std::move(launch));
        }
        // a launch the stream could not admit has an invalid Future, which this reports as well
        const spert::Status status = spert::Future::sync_all(futures);
        if (status != spert::Status::Ok) {
            GGML_LOG_ERROR("FlagOS SpacemiT: %zu ordered launches: spine-runtime status %d\n", futures.size(),
                           spert::status_code(status));
            return false;
        }

        bool ok = true;
        for (const auto & launch : launches) {
            ok = ok && !launch->failed.load();
        }
#if defined(__linux__)
        const size_t             shared = spacemit_count_ordered(launches, nth);
        static std::atomic<bool> warned{ false };
        if (shared > 0 && !warned.exchange(true)) {
            GGML_LOG_WARN("FlagOS SpacemiT: ordered launches: %zu of %zu had two tiles on one core (reported once)\n",
                          shared, launches.size());
        }
#endif
        return ok;
    }

    const char * name() const override {
        return policy == spacemit_stream_policy::per_call ? "spine-runtime, stream per call" : "spine-runtime, persistent stream";
    }

  private:
    // the stream for one run: the backend's, or one of its own (FLAGOS_SPACEMIT_STREAM=per-call) kept in per_call
    spert::Stream * stream_for_run(std::unique_ptr<spert::Stream> & per_call) {
        spert::Stream * stream = nullptr;
        if (policy == spacemit_stream_policy::per_call) {
            per_call = make_stream();
            stream   = per_call.get();
        } else {
            if (persistent_stream == nullptr) {
                persistent_stream = make_stream();
            }
            stream = persistent_stream.get();
        }
        if (stream != nullptr && static_cast<uint32_t>(stream->core_count()) > SPACEMIT_MAX_TILES) {
            GGML_LOG_ERROR("FlagOS SpacemiT: %u AI cores, at most %u supported\n",
                           static_cast<uint32_t>(stream->core_count()), SPACEMIT_MAX_TILES);
            return nullptr;
        }
        return stream;
    }

    std::unique_ptr<spert::Stream> make_stream() const {
        auto stream = std::make_unique<spert::Stream>(n_cores);
        if (!stream->valid() || stream->core_count() == 0) {
            GGML_LOG_ERROR("FlagOS SpacemiT: could not get %u AI cores from spine-runtime\n", n_cores);
            return nullptr;
        }
        return stream;
    }

    spacemit_stream_policy         policy;
    uint32_t                       n_cores;
    int64_t                        fail_step;
    bool                           retake_view = true;
    size_t                         tcm_slice   = 0;
    std::unique_ptr<spert::Stream> persistent_stream;
};

#else

// with its tiles one after another, the host executor can only count their pair waits. On the AI cores a pair whose
// tiles waited unequally often in a step would wait until the launch fails
bool spacemit_pairs_balanced(const spacemit_pairs & p, uint32_t n_tiles) {
    for (uint32_t t = 0; t + 1 < n_tiles; t += 2) {
        if (p.by_tile[t] != p.by_tile[t + 1]) {
            GGML_LOG_ERROR("FlagOS SpacemiT: tiles %u and %u made %u and %u pair waits\n", t, t + 1, p.by_tile[t],
                           p.by_tile[t + 1]);
            return false;
        }
    }
    return true;
}

// the barrier after each step when the host executor runs its tiles on threads
class host_barrier {
  public:
    explicit host_barrier(uint32_t n) : n(n) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        const uint64_t               round = rounds;
        if (++arrived == n) {
            arrived = 0;
            rounds++;
            done.notify_all();
        } else {
            done.wait(lock, [&] { return rounds != round; });
        }
    }

  private:
    std::mutex              mutex;
    std::condition_variable done;
    const uint32_t          n;
    uint32_t                arrived = 0;
    uint64_t                rounds  = 0;
};

// runs the tiles on the calling thread, one after another; lets the full path be tested on hosts without the AI cores.
// test-only: FLAGOS_SPACEMIT_TEST_TCM=<bytes> gives the tiles a stand-in for TCM in heap memory, so that the paths that
// need TCM (the 1-row path, path A) run there too. FLAGOS_SPACEMIT_TEST_THREADS=1 runs the tiles of a launch at the same
// time, on threads, as on the AI cores: each with its own TCM stand-in, a barrier after each step, spacemit_tile_main's
// failure protocol, and pair waits that wait
class host_executor final : public spacemit_executor {
  public:
    host_executor(uint32_t n_tiles, int64_t fail_step) : n_tiles(n_tiles), fail_step(fail_step) {
        const char * threads_value = std::getenv("FLAGOS_SPACEMIT_TEST_THREADS");
        threaded                   = threads_value != nullptr && std::strcmp(threads_value, "0") != 0;
        const char * value         = std::getenv("FLAGOS_SPACEMIT_TEST_TCM");
        const size_t size          = value != nullptr ? (size_t) std::max(0LL, std::atoll(value)) : 0;
        tcm_stride                 = GGML_PAD(size, 64);
        tcm_bytes                  = tcm_stride * (threaded ? n_tiles : 1);  // one stand-in per tile on threads
        tcm                        = size > 0 ? ggml_aligned_malloc(tcm_bytes) : nullptr;
        tcm_size                   = tcm != nullptr ? size : 0;
    }

    ~host_executor() override {
        if (tcm != nullptr) {
            ggml_aligned_free(tcm, tcm_bytes);
        }
    }

    bool run(const spacemit_plugin_step * steps, size_t n_steps, void * workspace, size_t workspace_size) override {
        spacemit_launch launch;
        launch.steps     = steps;
        launch.n_steps   = n_steps;
        launch.fail_step = fail_step;
        g_steps_launched += n_steps;
        spacemit_stats_begin(launch, stats, n_tiles);
        const spacemit_clock::time_point start = spacemit_stats_now(launch);
        if (threaded) {
            host_barrier             barrier(n_tiles);
            std::vector<std::thread> threads;
            for (uint32_t ith = 0; ith < n_tiles; ith++) {
                threads.emplace_back([&, ith] {
                    const spacemit_tile tile = { ith,       n_tiles,        tcm_of(ith), tcm_size,
                                                 workspace, workspace_size, &launch.pairs };
                    uint64_t            barrier_ns = 0;
                    for (size_t i = 0; i < launch.n_steps; i++) {
                        const spacemit_clock::time_point t0 = spacemit_stats_now(launch);
                        if (!launch.failed.load() && !spacemit_run_step(launch, i, tile)) {
                            launch.failed.store(true);
                        }
                        const spacemit_clock::time_point t1 = spacemit_stats_now(launch);
                        barrier.wait();
                        if (launch.stats != nullptr) {
                            barrier_ns += spacemit_stats_step(launch, i, ith, t0, t1, spacemit_clock::now());
                        }
                    }
                    if (launch.stats != nullptr) {
                        launch.stats->barrier_ns[ith] += barrier_ns;
                    }
                });
            }
            for (std::thread & t : threads) {
                t.join();
            }
            spacemit_stats_end(launch, start);
            return !launch.failed.load();
        }

        launch.pairs.concurrent = false;
        for (size_t i = 0; i < n_steps && !launch.failed.load(); i++) {
            for (uint32_t ith = 0; ith < n_tiles; ith++) {
                const spacemit_clock::time_point t0 = spacemit_stats_now(launch);
                if (!spacemit_run_step(launch, i,
                                       { ith, n_tiles, tcm_of(ith), tcm_size, workspace, workspace_size, &launch.pairs })) {
                    launch.failed.store(true);
                }
                if (launch.stats != nullptr) {
                    const spacemit_clock::time_point t1 = spacemit_clock::now();
                    spacemit_stats_step(launch, i, ith, t0, t1, t1);  // no barriers between tiles run in turn
                }
            }
            if (!spacemit_pairs_balanced(launch.pairs, n_tiles)) {
                launch.failed.store(true);
            }
        }
        spacemit_stats_end(launch, start);
        return !launch.failed.load();
    }

    const char * name() const override {
        return threaded ? "host threads (no spine-runtime)" : "serial (no spine-runtime)";
    }

  private:
    void * tcm_of(uint32_t ith) const {
        return tcm == nullptr ? nullptr : static_cast<uint8_t *>(tcm) + (threaded ? ith * tcm_stride : 0);
    }

    uint32_t n_tiles;
    int64_t  fail_step;
    bool     threaded   = false;
    void *   tcm        = nullptr;
    size_t   tcm_size   = 0;
    size_t   tcm_stride = 0;
    size_t   tcm_bytes  = 0;
};

#endif

}  // namespace

spacemit_ordered_counts spacemit_ordered_counts_take() {
    std::lock_guard<std::mutex> lock(g_ordered_mutex);
    const spacemit_ordered_counts c = g_ordered;
    g_ordered                       = spacemit_ordered_counts{};
    return c;
}

uint64_t spacemit_steps_take() {
    return g_steps_launched.exchange(0);
}

std::unique_ptr<spacemit_executor> spacemit_executor_create(spacemit_stream_policy policy, uint32_t n_cores) {
#if defined(GGML_FLAGOS_SPACEMIT_SPERT)
    return std::make_unique<spert_executor>(policy, n_cores, spacemit_test_fail_step());
#else
    GGML_UNUSED(policy);
    return std::make_unique<host_executor>(n_cores, spacemit_test_fail_step());
#endif
}
