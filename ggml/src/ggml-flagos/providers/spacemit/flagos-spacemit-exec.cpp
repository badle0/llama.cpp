#include "flagos-spacemit-exec.h"

#include "../../../ggml-impl.h"

#include <atomic>
#include <cstdlib>

#if defined(GGML_FLAGOS_SPACEMIT_SPERT)
#    include <spert.hpp>
#endif

namespace {

// test-only: FLAGOS_SPACEMIT_TEST_FAIL_NODE=n makes tile 0 report a failure at step n
int64_t spacemit_test_fail_step() {
    const char * value = std::getenv("FLAGOS_SPACEMIT_TEST_FAIL_NODE");
    return value != nullptr ? std::atoll(value) : -1;
}

struct spacemit_launch {
    const spacemit_step * steps   = nullptr;
    size_t                n_steps = 0;
    int64_t               fail_step = -1;
    std::atomic<bool>     failed{ false };
};

bool spacemit_run_step(const spacemit_launch & launch, size_t i, const spacemit_tile & tile) {
    const bool ok = launch.steps[i].kernel(tile, launch.steps[i].node);
    return ok && !(static_cast<int64_t>(i) == launch.fail_step && tile.ith == 0);
}

#if defined(GGML_FLAGOS_SPACEMIT_SPERT)

// every tile reaches every barrier and reads the failure flag right after it, so all tiles stop together:
// a tile that left early would leave the others waiting at the next barrier forever
void spacemit_tile_main(spert::Context * ctx, spacemit_launch * launch) noexcept {
    const spert::SharedBufferView tcm  = ctx->shared_buffer();
    const spacemit_tile           tile = { ctx->program_id(0), ctx->grid_dim(0), tcm.data, tcm.size };
    for (size_t i = 0; i < launch->n_steps; i++) {
        if (!spacemit_run_step(*launch, i, tile)) {
            launch->failed.store(true);
        }
        if (ctx->sync() != spert::Status::Ok) {
            launch->failed.store(true);
        }
        if (launch->failed.load()) {
            break;
        }
    }
}

class spert_executor final : public spacemit_executor {
  public:
    spert_executor(spacemit_stream_policy policy, uint32_t n_cores, int64_t fail_step) :
        policy(policy), n_cores(n_cores), fail_step(fail_step) {}

    bool run(const std::vector<spacemit_step> & steps) override {
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

        spacemit_launch launch;
        launch.steps     = steps.data();
        launch.n_steps   = steps.size();
        launch.fail_step = fail_step;
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

// runs the tiles one after another on the calling thread; lets the full path be tested on hosts without the AI cores
class serial_executor final : public spacemit_executor {
  public:
    serial_executor(uint32_t n_tiles, int64_t fail_step) : n_tiles(n_tiles), fail_step(fail_step) {}

    bool run(const std::vector<spacemit_step> & steps) override {
        spacemit_launch launch;
        launch.steps     = steps.data();
        launch.n_steps   = steps.size();
        launch.fail_step = fail_step;
        for (size_t i = 0; i < steps.size() && !launch.failed.load(); i++) {
            for (uint32_t ith = 0; ith < n_tiles; ith++) {
                if (!spacemit_run_step(launch, i, { ith, n_tiles, nullptr, 0 })) {
                    launch.failed.store(true);
                }
            }
        }
        return !launch.failed.load();
    }

    const char * name() const override { return "serial (no spine-runtime)"; }

  private:
    uint32_t n_tiles;
    int64_t  fail_step;
};

#endif

}  // namespace

std::unique_ptr<spacemit_executor> spacemit_executor_create(spacemit_stream_policy policy, uint32_t n_cores) {
#if defined(GGML_FLAGOS_SPACEMIT_SPERT)
    return std::make_unique<spert_executor>(policy, n_cores, spacemit_test_fail_step());
#else
    GGML_UNUSED(policy);
    return std::make_unique<serial_executor>(n_cores, spacemit_test_fail_step());
#endif
}
