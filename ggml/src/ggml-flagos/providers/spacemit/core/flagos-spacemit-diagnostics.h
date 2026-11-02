#pragma once

// The core's diagnostics (plan.md "S1 design" §1 and §7, "S1 step 5 spec"): the switches the core acts on, under the
// deck's names (page 7), and the two counter files. Moved here from flagos-spacemit.cpp 2026-11-02 (S1 step 5), where
// graph_compute had them inline; the core's own code, built without libspert and without Common.
//
// The names (S1 step 5 §2; the old FLAGOS_SPACEMIT_* name of each, when still set, is reported once and otherwise
// ignored, spacemit_renamed):
//   GGML_FLAGOS_STATS=<file>|-|1          the per-graph counters (E3c's per-token budget); - or 1: stderr
//   GGML_FLAGOS_OP_STATS=<file>|-|1       the op counters (fusion-and-splitting.md §7)
//   GGML_FLAGOS_DISABLE_OPS=<op>[,<op>]   named ggml ops back to the CPU (spacemit_find_op refuses them)
//   GGML_FLAGOS_TRACE_LAUNCH=1            the steps logged by name before each replay, the failing one named
//   GGML_FLAGOS_SYNC_LAUNCH=1             one launch per step, the host waiting for each (E3a)
// GGML_FLAGOS_KERNEL_VARIANTS=0 is the plugin's (flagos-spacemit-kernels.cpp) and GGML_FLAGOS_PROVIDER_PATH the
// loader's. The core's other switches keep their names (FLAGOS_SPACEMIT_DISABLE, PLAN, QUANTIZE, TEST_DEVICE_TYPE):
// the deck names none of them.

#include "flagos-spacemit-plan.h"    // spacemit_plan_result
#include "flagos-spacemit-plugin.h"  // spacemit_plugin_step
#include "flagos-spacemit-stats.h"   // spacemit_run_stats

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

// an old switch name that was replaced: when it is still set, reported once with the new name and otherwise ignored.
// Called where the new name is read. Read by tests too (the check program's check_diagnostics)
void spacemit_renamed(const char * old_name, const char * new_name);

// the switches parsed once (GGML_FLAGOS_DISABLE_OPS, GGML_FLAGOS_TRACE_LAUNCH) and the warnings of
// spacemit_renamed parse and warn again after this call; the counter files' switches are read once per process and
// are not affected. For tests that change a switch inside one process (the check program's set_env)
void spacemit_diagnostics_reset();

enum class spacemit_launch_kind {
    per_split,      // the default: a split's steps in one launch, a barrier after each
    per_step,       // E3a: every step its own launch, the host waiting for each
};

// GGML_FLAGOS_SYNC_LAUNCH, any value but 0 or empty: every step launches on its own, as a runtime without graph
// capture launches each kernel (structural-experiments.md E3a); unset names the default. Denglin reads the same name
// the same way (flagos-denglin.cpp:371). Read at every graph, so a benchmark can switch
spacemit_launch_kind spacemit_launch_mode();

// test-only until adopted (fusion-and-splitting.md §5, candidate 6): FLAGOS_SPACEMIT_QUANTIZE=shared leaves out a
// MUL_MAT's quantize step when the matmul that last quantized took the same activations into the same layout
bool spacemit_quantize_shared();

// GGML_FLAGOS_DISABLE_OPS=<op>[,<op>...]: whether this ggml op is named there, so spacemit_find_op refuses it and the
// scheduler leaves it on the CPU. The list is parsed once (an unknown name is reported once and ignored)
bool spacemit_op_disabled(enum ggml_op op);

// GGML_FLAGOS_TRACE_LAUNCH=1: the steps of every replay logged by name. Read once
bool spacemit_trace_launch();

// one GGML_LOG_INFO line per step: its number, its contract name, its node, op and shape, and its first operand.
// names must be parallel to steps (graph_compute fills it when the trace or the op counters are on) or empty
void spacemit_trace_steps(const std::vector<spacemit_plugin_step> & steps, const std::vector<const char *> & names);

// whether node's output overlaps t's data
bool spacemit_overlaps(const ggml_tensor * node, const ggml_tensor * t);

// test-only (structural-experiments.md E3c): GGML_FLAGOS_STATS=<file> counts, per kind of graph (launch mode, rows of
// its first matmul, KV length class of its first attention: 0 below 1024, 1024 below 4096, else 4096), the graphs,
// steps and launches and the time spent in graph_compute, in the launches as the host waits for them, in the kernels
// (summed over the tiles; per step the slowest tile, "busy") and at the barriers (summed over the tiles), and the plan
// cache's answers (FLAGOS_SPACEMIT_PLAN: hits by uid, misses) with the time to the step list in hand (plan_lookup_us,
// the check and the expansion included on a miss and under none) and from after the check (plan_key_us: on a miss the
// store; on a hit, which skips the check, equal to plan_lookup_us). When the process ends it appends one JSON line per
// kind to <file>, or to stderr when the value is - or 1. The switch is read once
class spacemit_stats_file {
  public:
    // nullptr when the switch is not set
    static spacemit_stats_file * get();

    // plan, lookup_us, key_us: the plan cache's answer for this graph (FLAGOS_SPACEMIT_PLAN), the time from the top of
    // graph_compute to the step list in hand (the check and the expansion on a miss and under none) and the time from
    // after the contract check to it (the check expands as it goes, so on a miss the store and under none about 0; on
    // a hit, which skips the check, from the top, as lookup_us)
    void add(const char * launch, int64_t rows, int64_t n_kv, const spacemit_run_stats & run, double graph_us,
             spacemit_plan_result plan, double lookup_us, double key_us);

    // writes the lines
    ~spacemit_stats_file();

    // the switch's value, as get() reads it (nullptr or empty: nothing is written). Public for tests only, which have
    // no way to re-read the switch: the process reads it once
    explicit spacemit_stats_file(const char * value);

    spacemit_stats_file(const spacemit_stats_file &)             = delete;
    spacemit_stats_file & operator=(const spacemit_stats_file &) = delete;

  private:
    // whether anything is written: a path or stderr
    bool writing() const { return to_stderr || !path.empty(); }

    struct totals {
        uint64_t graphs = 0, steps = 0, launches = 0;
        uint32_t tiles  = 0;
        double   graph_us = 0.0, launch_us = 0.0, kernel_us = 0.0, busy_us = 0.0, barrier_us = 0.0;
        uint64_t plan_hits_uid = 0, plan_misses = 0;
        double   plan_lookup_us = 0.0, plan_key_us = 0.0;
    };

    std::string                                                 path;  // empty with stderr and when off
    bool                                                        to_stderr = false;
    std::mutex                                                  mutex;
    std::map<std::tuple<std::string, int64_t, int64_t>, totals> kinds;
};

// test-only (fusion-and-splitting.md §7): GGML_FLAGOS_OP_STATS=<file> times the steps by what they do, from E3c's
// clock reads (spacemit_run_stats). Per kind of graph (launch mode, rows of its first matmul, KV length of its first
// attention, and the kernels and shapes of its steps in order) it sums each step's times over the graphs, leaving out
// the kind's first graph (it may create the stream and touch new memory), and names each step when the kind is first
// seen. When the process ends it appends to <file> (or stderr when the value is - or 1) one JSON line per kind and
// name, and one for all of the kind's steps, all per graph: the steps, the mean tile's kernel time (kernel_us) and its
// kernel and barrier time (wall_us), each step's slowest tile summed (busy_us), each tile's kernel time (tile_us) and
// how often each was a step's slowest (slowest, over all graphs). The switch is read once
class spacemit_op_stats_file {
  public:
    // nullptr when the switch is not set
    static spacemit_op_stats_file * get();

    void add(const char * launch, int64_t rows, int64_t n_kv, const std::vector<spacemit_plugin_step> & steps,
             const std::vector<const char *> & names, const spacemit_run_stats & run);

    ~spacemit_op_stats_file();

    // as spacemit_stats_file's: the switch's value, public for tests only
    explicit spacemit_op_stats_file(const char * value);

    spacemit_op_stats_file(const spacemit_op_stats_file &)             = delete;
    spacemit_op_stats_file & operator=(const spacemit_op_stats_file &) = delete;

  private:
    bool writing() const { return to_stderr || !path.empty(); }

    struct totals {
        uint32_t                 tiles     = 0;
        uint64_t                 graphs    = 0;  // counted ones: all but the first
        double                   launch_us = 0.0;
        std::vector<std::string> labels;
        std::vector<double>      kernel_ns, busy_ns, wall_ns;  // per step, summed over the tiles (busy: the slowest)
        std::vector<double>      tile_ns;                      // [step * tiles + tile]
        std::vector<uint64_t>    slowest;                      // [step * tiles + tile]: graphs where it was slowest
    };

    std::string                                                           path;  // empty with stderr and when off
    bool                                                                  to_stderr = false;
    std::mutex                                                            mutex;
    std::map<std::tuple<std::string, int64_t, int64_t, uint64_t>, totals> kinds;
};
