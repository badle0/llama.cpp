#include "flagos-spacemit-exec.h"

#include "../../../ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>

#if defined(GGML_FLAGOS_SPACEMIT_SPERT)
#    include <spert.hpp>
#else
#    include <condition_variable>
#    include <cstring>
#    include <mutex>
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

struct spacemit_launch {
    const spacemit_step * steps          = nullptr;
    size_t                n_steps        = 0;
    void *                workspace      = nullptr;
    size_t                workspace_size = 0;
    int64_t               fail_step      = -1;
    std::atomic<bool>     failed{ false };
    size_t                tcm_sizes[SPACEMIT_MAX_TILES] = {};  // each tile's TCM, so all can agree on one size
    spacemit_pairs        pairs;

    spacemit_launch() { pairs.failed = &failed; }
};

bool spacemit_run_step(const spacemit_launch & launch, size_t i, const spacemit_tile & tile) {
    const bool ok = launch.steps[i].kernel(tile, launch.steps[i].node);
    return ok && !(static_cast<int64_t>(i) == launch.fail_step && tile.ith == 0);
}

#if defined(GGML_FLAGOS_SPACEMIT_SPERT)

// every tile reaches every barrier, also after a failure, when it only skips the remaining kernels: a tile that left
// early would leave the others waiting at the next barrier forever. (Leaving when the flag is set after a barrier is
// not enough: a faster tile may already have failed in the next step, so tiles would disagree.)
void spacemit_tile_main(spert::Context * ctx, spacemit_launch * launch) noexcept {
    const uint32_t                ith = ctx->program_id(0);
    const uint32_t                nth = ctx->grid_dim(0);
    const spert::SharedBufferView tcm = ctx->shared_buffer();

    // agree on one TCM size, the smallest any tile has, so every tile takes the same kernel path
    launch->tcm_sizes[ith] = tcm.data != nullptr ? tcm.size : 0;
    if (ctx->sync() != spert::Status::Ok) {
        launch->failed.store(true);  // spine-runtime's barrier itself failed: nothing can keep the tiles together
        return;
    }
    size_t tcm_size = launch->tcm_sizes[0];
    for (uint32_t i = 1; i < nth; i++) {
        tcm_size = std::min(tcm_size, launch->tcm_sizes[i]);
    }
    const spacemit_tile tile = { ith, nth, tcm_size != 0 ? tcm.data : nullptr, tcm_size, launch->workspace,
                                 launch->workspace_size, &launch->pairs };

    for (size_t i = 0; i < launch->n_steps; i++) {
        if (!launch->failed.load() && !spacemit_run_step(*launch, i, tile)) {
            launch->failed.store(true);
        }
        if (ctx->sync() != spert::Status::Ok) {
            launch->failed.store(true);
            return;
        }
    }
}

class spert_executor final : public spacemit_executor {
  public:
    spert_executor(spacemit_stream_policy policy, uint32_t n_cores, int64_t fail_step) :
        policy(policy), n_cores(n_cores), fail_step(fail_step) {}

    bool run(const std::vector<spacemit_step> & steps, void * workspace, size_t workspace_size) override {
        std::unique_ptr<spert::Stream> per_call_stream;
        spert::Stream *                stream = nullptr;
        if (policy == spacemit_stream_policy::per_call) {
            per_call_stream = make_stream();
            stream          = per_call_stream.get();
        } else {
            if (persistent_stream == nullptr) {
                persistent_stream = make_stream();
            }
            stream = persistent_stream.get();
        }
        if (stream == nullptr) {
            return false;
        }
        if (static_cast<uint32_t>(stream->core_count()) > SPACEMIT_MAX_TILES) {
            GGML_LOG_ERROR("FlagOS SpacemiT: %u AI cores, at most %u supported\n",
                           static_cast<uint32_t>(stream->core_count()), SPACEMIT_MAX_TILES);
            return false;
        }

        spacemit_launch launch;
        launch.steps          = steps.data();
        launch.n_steps        = steps.size();
        launch.workspace      = workspace;
        launch.workspace_size = workspace_size;
        launch.fail_step      = fail_step;
        const spert::Future future = stream->launch(spert::Grid{ stream->core_count() }, spacemit_tile_main, &launch);
        const spert::Status status = future.sync();
        if (status != spert::Status::Ok) {
            GGML_LOG_ERROR("FlagOS SpacemiT: spine-runtime launch failed with status %d\n", spert::status_code(status));
            return false;
        }
        return !launch.failed.load();
    }

    const char * name() const override {
        return policy == spacemit_stream_policy::per_call ? "spine-runtime, stream per call" : "spine-runtime, persistent stream";
    }

  private:
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

    bool run(const std::vector<spacemit_step> & steps, void * workspace, size_t workspace_size) override {
        spacemit_launch launch;
        launch.steps     = steps.data();
        launch.n_steps   = steps.size();
        launch.fail_step = fail_step;
        if (threaded) {
            host_barrier             barrier(n_tiles);
            std::vector<std::thread> threads;
            for (uint32_t ith = 0; ith < n_tiles; ith++) {
                threads.emplace_back([&, ith] {
                    const spacemit_tile tile = { ith,       n_tiles,        tcm_of(ith), tcm_size,
                                                 workspace, workspace_size, &launch.pairs };
                    for (size_t i = 0; i < launch.n_steps; i++) {
                        if (!launch.failed.load() && !spacemit_run_step(launch, i, tile)) {
                            launch.failed.store(true);
                        }
                        barrier.wait();
                    }
                });
            }
            for (std::thread & t : threads) {
                t.join();
            }
            return !launch.failed.load();
        }

        launch.pairs.concurrent = false;
        for (size_t i = 0; i < steps.size() && !launch.failed.load(); i++) {
            for (uint32_t ith = 0; ith < n_tiles; ith++) {
                if (!spacemit_run_step(launch, i,
                                       { ith, n_tiles, tcm_of(ith), tcm_size, workspace, workspace_size, &launch.pairs })) {
                    launch.failed.store(true);
                }
            }
            if (!spacemit_pairs_balanced(launch.pairs, n_tiles)) {
                launch.failed.store(true);
            }
        }
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

std::unique_ptr<spacemit_executor> spacemit_executor_create(spacemit_stream_policy policy, uint32_t n_cores) {
#if defined(GGML_FLAGOS_SPACEMIT_SPERT)
    return std::make_unique<spert_executor>(policy, n_cores, spacemit_test_fail_step());
#else
    GGML_UNUSED(policy);
    return std::make_unique<host_executor>(n_cores, spacemit_test_fail_step());
#endif
}
