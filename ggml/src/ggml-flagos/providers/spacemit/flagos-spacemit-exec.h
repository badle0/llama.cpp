#pragma once

#include "flagos-spacemit-ops.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

struct spacemit_step {
    spacemit_kernel_fn kernel;
    ggml_tensor *      node;
    size_t             workspace_offset = 0;  // where the step's part of the shared workspace starts
};

// test-only (structural-experiments.md E3c, fusion-and-splitting.md §7): what one graph's runs spent, step by step and
// tile by tile. The executor fills it during run() when it has one; the runs of a graph append their steps one after
// another
struct spacemit_run_stats {
    size_t                stride    = 0;  // steps one tile's row of kernel_ns holds (the graph's steps)
    size_t                n_steps   = 0;  // steps recorded so far
    uint32_t              n_tiles   = 0;  // tiles of the runs
    uint64_t              launches  = 0;
    double                launch_us = 0;  // host: from each launch to the end of the wait for it, summed
    std::vector<uint32_t> kernel_ns;      // [tile * stride + step]: the tile's time in the step's kernel
    std::vector<uint32_t> step_barrier_ns;  // [tile * stride + step]: the tile's time at the barrier after the step
    uint64_t              barrier_ns[SPACEMIT_MAX_TILES] = {};  // per tile: at the barriers after its steps, summed

    // before a graph of graph_steps steps
    void reset(size_t graph_steps) {
        stride   = graph_steps;
        n_steps  = 0;
        n_tiles  = 0;
        launches = 0;
        launch_us = 0;
        if (kernel_ns.size() < SPACEMIT_MAX_TILES * stride) {
            kernel_ns.resize(SPACEMIT_MAX_TILES * stride);
            step_barrier_ns.resize(SPACEMIT_MAX_TILES * stride);
        }
        std::fill(std::begin(barrier_ns), std::end(barrier_ns), 0);
    }
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

    // test-only (E3c): set by graph_compute around the runs of a graph it counts; run() fills it, run_ordered() not
    spacemit_run_stats * stats = nullptr;
};

// spine-runtime executor on the AI cores; without spine-runtime (non-riscv64 builds) a serial stand-in for tests
std::unique_ptr<spacemit_executor> spacemit_executor_create(spacemit_stream_policy policy, uint32_t n_cores);

// test-only (structural-experiments.md E3b-diag): what the launches ordered in the cores (FLAGOS_SPACEMIT_TEST_LAUNCH=
// per-step-join) did since the last call. Each tile notes its core after its join, after taking its TCM, after the TCM
// agreement and after its step, and which TCM it took. All zero without spine-runtime
struct spacemit_ordered_counts {
    uint64_t launches          = 0;
    uint64_t shared_core       = 0;  // launches with two tiles on one core, at any of the points
    uint64_t moved_tiles       = 0;  // tiles found on another core than at the point before
    uint64_t tcm_elsewhere     = 0;  // launches where two tiles on different cores held the same TCM
    uint64_t pairs_on_one_core = 0;  // launches where both tiles of a pair (2i, 2i+1) were on one core
};
spacemit_ordered_counts spacemit_ordered_counts_take();

// test-only: the steps the executors have launched since the last call (flagos-check-spacemit counts the quantize steps
// that FLAGOS_SPACEMIT_QUANTIZE=shared leaves out)
uint64_t spacemit_steps_take();
