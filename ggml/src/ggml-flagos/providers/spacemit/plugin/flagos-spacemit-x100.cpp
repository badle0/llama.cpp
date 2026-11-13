#include "flagos-spacemit-x100.h"

#include "../../../../ggml-impl.h"

#if defined(GGML_FLAGOS_SPACEMIT_SPERT)

#    include <spert.hpp>

#    include <atomic>
#    include <memory>

namespace {

spert::BackendHandle x100_backend() {
    static const spert::BackendHandle handle = spert::backend("spacemit-k3-x100");
    return handle;
}

struct x100_launch {
    void (*fn)(void * arg, uint32_t i);
    void *              arg;
    std::atomic<bool> * done;  // per index, set when its tile returned
};

// tile t runs index t + 1; index 0 is the calling thread's
void x100_tile(spert::Context * ctx, x100_launch * launch) noexcept {
    const uint32_t i = ctx->program_id(0) + 1;
    launch->fn(launch->arg, i);
    launch->done[i].store(true, std::memory_order_release);
}

void run_here(uint32_t n, void (*fn)(void * arg, uint32_t i), void * arg) {
    for (uint32_t i = 0; i < n; i++) {
        fn(arg, i);
    }
}

void warn_once(std::atomic<bool> & warned, const char * what) {
    if (!warned.exchange(true)) {
        GGML_LOG_WARN("FlagOS SpacemiT: X100 backend: %s; the work runs on the calling thread\n", what);
    }
}

}  // namespace

uint32_t spacemit_x100_cores() {
    static const uint32_t cores = [] {
        const spert::BackendHandle handle = x100_backend();
        return handle.valid() ? (uint32_t) spert::backend_info(handle).num_cores : 0u;
    }();
    return cores;
}

bool spacemit_x100_run(uint32_t n, void (*fn)(void * arg, uint32_t i), void * arg, uint32_t * reran) {
    static std::atomic<bool> warned_stream{ false }, warned_launch{ false }, warned_status{ false };
    GGML_ASSERT(n >= 2);
    *reran                            = 0;
    const spert::BackendHandle handle = x100_backend();
    if (!handle.valid()) {
        run_here(n, fn, arg);
        return false;
    }
    std::unique_ptr<std::atomic<bool>[]> done(new std::atomic<bool>[n]);
    for (uint32_t i = 0; i < n; i++) {
        done[i].store(false, std::memory_order_relaxed);
    }
    x100_launch launch = { fn, arg, done.get() };
    {
        spert::StreamConfig config;
        config.n_cores = n - 1;
        spert::Stream stream(handle, config);
        if (!stream.valid()) {
            warn_once(warned_stream, "no stream");
            run_here(n, fn, arg);
            return false;
        }
        const spert::Future future = stream.launch(spert::Grid{ n - 1 }, x100_tile, &launch);
        if (!future.valid()) {
            warn_once(warned_launch, "launch refused");
            run_here(n, fn, arg);
            return false;
        }
        fn(arg, 0);
        const spert::Status status = future.sync();
        if (status != spert::Status::Ok && !warned_status.exchange(true)) {
            GGML_LOG_WARN("FlagOS SpacemiT: X100 backend: launch ended with status %s; unfinished ranges run on the "
                          "calling thread\n",
                          spert::to_string(status));
        }
    }
    // after the release, which cancels the launch's outstanding work (spert.hpp:522-524)
    for (uint32_t i = 1; i < n; i++) {
        if (!done[i].load(std::memory_order_acquire)) {
            fn(arg, i);
            ++*reran;
        }
    }
    return true;
}

#else

uint32_t spacemit_x100_cores() {
    return 0;
}

bool spacemit_x100_run(uint32_t n, void (*fn)(void * arg, uint32_t i), void * arg, uint32_t * reran) {
    *reran = 0;
    for (uint32_t i = 0; i < n; i++) {
        fn(arg, i);
    }
    return false;
}

#endif
