#pragma once

// The core's per-step clock reads (plan.md "S1 step 5 spec" §1): the struct the diagnostics fill and the plugin's
// tiles write into, so that the core's diagnostics need no plugin header. Moved here unchanged from
// flagos-spacemit-exec.h 2026-11-02 (S1 step 5); the executor includes this file.

#include "../flagos-spacemit-ops.h"  // SPACEMIT_MAX_TILES

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

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
