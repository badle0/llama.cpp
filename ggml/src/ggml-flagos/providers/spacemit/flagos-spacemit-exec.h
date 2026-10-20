#pragma once

#include "flagos-spacemit-ops.h"

#include <cstdint>
#include <memory>
#include <vector>

struct spacemit_step {
    spacemit_kernel_fn kernel;
    ggml_tensor *      node;
};

// persistent: one spine-runtime stream (core grant) per backend, kept until the backend is freed
// per_call:   a new stream for every run, releasing the AI cores in between (ggml-spacemit's choice)
enum class spacemit_stream_policy {
    persistent,
    per_call,
};

class spacemit_executor {
  public:
    virtual ~spacemit_executor() = default;

    // runs the steps in order on every tile, with a barrier after each step, all tiles sharing the workspace;
    // false if any step or the runtime failed
    virtual bool run(const std::vector<spacemit_step> & steps, void * workspace, size_t workspace_size) = 0;

    // test-only (structural-experiments.md E3b): every step a launch of its own, ordered where the runtime can order
    // them without the host; here, as a runtime without that would do, the host waits for each
    virtual bool run_ordered(const std::vector<spacemit_step> & steps, void * workspace, size_t workspace_size) {
        std::vector<spacemit_step> one(1);
        for (const spacemit_step & step : steps) {
            one[0] = step;
            if (!run(one, workspace, workspace_size)) {
                return false;
            }
        }
        return true;
    }

    virtual const char * name() const = 0;
};

// spine-runtime executor on the AI cores; without spine-runtime (non-riscv64 builds) a serial stand-in for tests
std::unique_ptr<spacemit_executor> spacemit_executor_create(spacemit_stream_policy policy, uint32_t n_cores);
