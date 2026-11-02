#pragma once

#include "flagos-spacemit-ops.h"
#include "core/flagos-spacemit-stats.h"  // spacemit_run_stats (the core's; S1 step 5)

#include <cstdint>
#include <memory>

// a step is spacemit_plugin_step (the C interface): the kernel, its node and where its part of the shared workspace
// starts

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
    virtual bool run(const spacemit_plugin_step * steps, size_t n_steps, void * workspace, size_t workspace_size) = 0;

    virtual const char * name() const = 0;

    // test-only (E3c): set by graph_compute around the runs of a graph it counts; run() fills it
    spacemit_run_stats * stats = nullptr;
};

// spine-runtime executor on the AI cores; without spine-runtime (non-riscv64 builds) a serial stand-in for tests
std::unique_ptr<spacemit_executor> spacemit_executor_create(spacemit_stream_policy policy, uint32_t n_cores);

// test-only: the steps the executors have launched since the last call (flagos-check-spacemit counts the quantize steps
// that FLAGOS_SPACEMIT_QUANTIZE=shared leaves out)
uint64_t spacemit_steps_take();
