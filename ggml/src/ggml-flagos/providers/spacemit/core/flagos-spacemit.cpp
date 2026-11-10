// FlagOS provider for the SpacemiT K3 AI cores (A100, CPUs 8-15).
// One IGPU device (D3, 2026-10-26; ACCEL before) with a buffer type and a backend; ops run on the AI cores through
// spine-runtime (M2a: ADD; M2b, M2c: Q4_0, Q4_1, Q8_0 and Q6_K MUL_MAT on the IME, weights repacked in the buffer, Q6_K
// requantized to Q8_0; M2d: the other ops of a layer except attention).

#include "flagos-spacemit-api.h"
#include "flagos-spacemit-ops.h"
#include "flagos-spacemit-diagnostics.h"  // the switches and the counter files (S1 step 5)
#include "flagos-spacemit-loader.h"
#include "flagos-spacemit-plan.h"
#include "flagos-spacemit-stats.h"  // spacemit_run_stats (test-only, E3c)

#include "../../../../ggml-backend-impl.h"
#include "../../../../ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr uint64_t SPACEMIT_PROVIDER_ID       = 0x53504D5400000001ULL;  // "SPMT", 1
constexpr uint64_t SPACEMIT_DEVICE_UUID_HI    = 0x53504D544B330000ULL;  // "SPMTK3"
constexpr uint64_t SPACEMIT_DEVICE_UUID_LO    = 0x0000000041000002ULL;
constexpr uint64_t SPACEMIT_MEMORY_DOMAIN_ID  = 0x53504D544B330001ULL;  // K3 DRAM, shared with the X100 cores
constexpr size_t   SPACEMIT_BUFFER_ALIGNMENT  = 64;
constexpr uint32_t SPACEMIT_A100_VECTOR_BITS  = 1024;

struct spacemit_device_context {
    std::string                name;
    std::string                description;  // from the plugin's props
    std::string                runtime;      // from the plugin's props (the FlagOS profile)
    uint32_t                   n_ai_cores = 0;
    spacemit_plugin *          plugin     = nullptr;  // kept for the process lifetime, as ggml keeps its backends
    const spacemit_plugin_v1 * table      = nullptr;
    spacemit_plugin_device *   dev        = nullptr;
    spacemit_ops               ops;  // the core's contracts bound to this plugin's kernels (spacemit_ops_bind)
    ggml_backend_device        device     = {};
    ggml_backend_buffer_type   buffer_type = {};
    std::mutex                 run_mutex;  // this device's backends share one executor: one run at a time
};

struct spacemit_buffer_context {
    const spacemit_plugin_v1 * table = nullptr;
    spacemit_plugin_buffer *   buf   = nullptr;  // the plugin's buffer
    void *                     data  = nullptr;  // its base
    size_t                     size  = 0;
};

struct spacemit_backend_context {
    const spacemit_plugin_v1 *        table  = nullptr;
    spacemit_plugin_stream *          stream = nullptr;
    spacemit_run_stats_attach_fn      attach = nullptr;  // test-only (E3c), nullptr when the plugin has no counters
    // test-only (FLAGOS_SPACEMIT_QUANTIZE=shared), nullptr when the plugin does not offer the fact
    spacemit_mul_mat_same_activations_fn same_activations = nullptr;
    // the plugin's switch generation, in the plan cache's word (moved only by a test's switch reset); nullptr: 0
    spacemit_switches_generation_fn   switches_generation = nullptr;
    std::vector<spacemit_plugin_step> steps;
    std::vector<const char *>         step_names;  // the steps' contract names, parallel to steps
    spacemit_plan_cache               plan;        // the plan cache (FLAGOS_SPACEMIT_PLAN; plan.md "S1 step 3c spec")
    void *                            workspace      = nullptr;  // shared by the tiles; grows to the largest split
    size_t                            workspace_size = 0;
    spacemit_run_stats                run_stats;  // test-only (E3c): the current graph's, with GGML_FLAGOS_STATS

    ~spacemit_backend_context() {
        if (stream != nullptr) {
            table->stream_destroy(stream);
        }
        if (workspace != nullptr) {
            ggml_aligned_free(workspace, workspace_size);
        }
    }
};

std::mutex              g_mutex;
bool                    g_probed  = false;
bool                    g_present = false;
spacemit_device_context g_device;

ggml_guid g_backend_guid = { 0x46, 0x6c, 0x61, 0x67, 0x4f, 0x53, 0x2d, 0x53, 0x50, 0x4d, 0x54, 0x00, 0x00, 0x00, 0x00, 0x01 };

//
// buffer: the plugin's memory (host memory on huge pages where the system has them, else 64-byte aligned); Q4_0, Q4_1,
// Q8_0 and Q6_K matmul weights are stored in IME layouts (flagos-spacemit-weights.h), a Q6_K weight in more than
// ggml_nbytes (get_alloc_size). set, get, memset and cpy work in ggml's layout within ggml_nbytes; only the repack and
// its inverse touch the stored bytes beyond it, and clear covers the whole buffer.
//

// a status the buffer functions cannot report to ggml, which expects them to succeed
void spacemit_buffer_status(const char * what, enum spacemit_status status) {
    if (status != SPACEMIT_STATUS_SUCCESS) {
        GGML_LOG_ERROR("FlagOS SpacemiT: %s failed: %s\n", what, spacemit_status_name(status));
    }
}

void spacemit_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    ctx->table->buffer_free(ctx->buf);
    delete ctx;
}

void * spacemit_buffer_base(ggml_backend_buffer_t buffer) {
    return static_cast<spacemit_buffer_context *>(buffer->context)->data;
}

void spacemit_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset,
                                   size_t size) {
    auto * ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    spacemit_buffer_status("memset_tensor", ctx->table->tensor_fill(tensor, value, offset, size));
}

// repacks; reads undo the repack (Q4_1: the converted weights; Q6_K: the nearest Q6_K), so the scheduler can still copy
// a weight out if the provider refuses an op on it
void spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset,
                                size_t size) {
    auto * ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    spacemit_buffer_status("set_tensor", ctx->table->tensor_set(tensor, data, offset, size));
}

void spacemit_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset,
                                size_t size) {
    auto * ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    spacemit_buffer_status("get_tensor", ctx->table->tensor_get(tensor, data, offset, size));
}

// a raw copy would skip the repack; returning false makes ggml use set_tensor instead
bool spacemit_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    auto * ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    if (src->buffer == nullptr || !ggml_backend_buffer_is_host(src->buffer) || ctx->table->tensor_layout(dst) != 0) {
        return false;
    }
    std::memcpy(dst->data, src->data, ggml_nbytes(src));
    return true;
}

void spacemit_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    std::memset(ctx->data, value, ctx->size);
}

const ggml_backend_buffer_i g_buffer_iface = {
    /* .free_buffer   = */ spacemit_buffer_free,
    /* .get_base      = */ spacemit_buffer_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ spacemit_buffer_memset_tensor,
    /* .set_tensor    = */ spacemit_buffer_set_tensor,
    /* .get_tensor    = */ spacemit_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ spacemit_buffer_cpy_tensor,
    /* .clear         = */ spacemit_buffer_clear,
    /* .reset         = */ nullptr,
};

const char * spacemit_buffer_type_name(ggml_backend_buffer_type_t) {
    return "FlagOS:SpacemiT";
}

// ggml_backend_buft_alloc_buffer() handles size 0 itself and never calls this with it. The plugin decides the pages
// (huge pages from 2 MiB)
ggml_backend_buffer_t spacemit_buffer_type_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto *                   dev = static_cast<spacemit_device_context *>(buft->context);
    spacemit_plugin_buffer * buf = nullptr;
    const enum spacemit_status status = dev->table->buffer_alloc(dev->dev, size, &buf);
    if (status != SPACEMIT_STATUS_SUCCESS) {
        spacemit_buffer_status("alloc_buffer", status);
        return nullptr;
    }
    auto * ctx = new spacemit_buffer_context{ dev->table, buf, dev->table->buffer_base(buf), size };
    return ggml_backend_buffer_init(buft, g_buffer_iface, ctx, size);
}

size_t spacemit_buffer_type_alignment(ggml_backend_buffer_type_t) {
    return SPACEMIT_BUFFER_ALIGNMENT;
}

// a Q6_K weight is stored requantized to Q8_0: 34 bytes per 32 weights instead of 26.25
size_t spacemit_buffer_type_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return static_cast<spacemit_device_context *>(buft->context)->table->tensor_alloc_size(tensor);
}

const ggml_backend_buffer_type_i g_buffer_type_iface = {
    /* .get_name       = */ spacemit_buffer_type_name,
    /* .alloc_buffer   = */ spacemit_buffer_type_alloc,
    /* .get_alignment  = */ spacemit_buffer_type_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ spacemit_buffer_type_alloc_size,
    // repacked weights are not in ggml's layout, so the CPU backend must not read this buffer directly
    /* .is_host        = */ [](ggml_backend_buffer_type_t) { return false; },
};

//
// backend
//

const char * spacemit_backend_name(ggml_backend_t backend) {
    return static_cast<spacemit_device_context *>(backend->device->context)->name.c_str();
}

void spacemit_backend_free(ggml_backend_t backend) {
    delete static_cast<spacemit_backend_context *>(backend->context);
    delete backend;
}

bool spacemit_data_ready(const ggml_tensor * node) {
    if (node->data == nullptr) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] != nullptr && node->src[i]->data == nullptr) {
            return false;
        }
    }
    return true;
}

// every node is checked before its graph first runs (on every graph when the cache is off, at its store in the plan
// cache otherwise) and the workspace before every launch, so a graph either runs completely or fails with nothing
// written; a hit trusts the check made at its store under the uid, which llama.cpp keeps only while the graph is reused
// unchanged (its tensors and their allocations kept; a rebuilt graph gets a new uid, ggml-backend.cpp:1538)
ggml_status spacemit_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = static_cast<spacemit_backend_context *>(backend->context);
    auto * dev = static_cast<spacemit_device_context *>(backend->device->context);

    spacemit_stats_file * const    stats    = spacemit_stats_file::get();     // test-only (E3c)
    spacemit_op_stats_file * const op_stats = spacemit_op_stats_file::get();  // test-only (fusion-and-splitting.md §7)
    const bool                     timed    = stats != nullptr || op_stats != nullptr;
    // GGML_FLAGOS_TRACE_LAUNCH: the steps logged by name before the replay. Both it and the op counters need the
    // steps' contract names, so both the expansion below and the plan cache's entry must carry them (an entry stored
    // without names serves no graph that wants them, core/flagos-spacemit-plan.cpp:60-66)
    const bool                     trace      = spacemit_trace_launch();
    const bool                     want_names = op_stats != nullptr || trace;
    // the plan cache (plan.md "S1 step 3c spec"): by default (FLAGOS_SPACEMIT_PLAN unset or uid-trust) the step list
    // and the workspace sizes come from the cache on a hit, keyed by the split's uid, the node objects and the
    // configuration word; the entry is taken before the contract check and a hit skips it, trusting the uid as CUDA
    // and Hexagon do: a change to a node's operands or placement under a kept uid and the same objects is not seen
    // (llama.cpp never does this: a reused graph keeps its tensors and their allocations, and the uid changes when the
    // graph is rebuilt). The check still runs at every store, that is on every miss, and on every graph the scheduler
    // did not stamp (uid 0, a miss). The test-only none checks and expands every graph, nothing cached
    const spacemit_plan_mode                    plan_mode = spacemit_plan_mode_get();
    const bool                                  clocked   = timed || plan_mode != spacemit_plan_mode::none;
    const std::chrono::steady_clock::time_point start =
        clocked ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    int64_t rows = -1, n_kv = -1;  // the counters' kind of graph: its first matmul's rows, its first attention's KV length

    // a MUL_MAT quantizes its activations in step 0 and multiplies them in step 1 (spacemit_find_op). They have their
    // own part of the workspace, after the other ops' scratch, so that the ops between two matmuls of one input leave
    // them, and with FLAGOS_SPACEMIT_QUANTIZE=shared the second matmul runs only its step 1
    bool share = spacemit_quantize_shared();
    if (share && ctx->same_activations == nullptr) {  // the fact is not contract, so a plugin may not offer it (A2)
        static std::atomic<bool> warned{ false };
        if (!warned.exchange(true)) {
            GGML_LOG_WARN("FlagOS SpacemiT: the plugin has no activation-sharing fact; "
                          "FLAGOS_SPACEMIT_QUANTIZE=shared stays off\n");
        }
        share = false;
    }
    // no plan cache under FLAGOS_SPACEMIT_QUANTIZE=shared (test-only): its sharing decisions read data addresses
    // (spacemit_overlaps), which the key does not cover
    const bool          cached = plan_mode == spacemit_plan_mode::uid_trust && !share;
    const ggml_tensor * held = nullptr;  // the MUL_MAT whose quantized activations that part holds
    size_t              scratch_size = 0, act_size = 0;
    ctx->steps.clear();
    ctx->step_names.clear();
    // the configuration word is the activation-sharing fact (bit 0) and, above it, the plugin's switch generation,
    // which moves whenever a kernel switch is reset (so FLAGOS_SPACEMIT_ATTN reaches the key without the core reading
    // a plugin switch); a different word is a miss
    uint64_t word = 0;
    if (cached) {
        const uint64_t generation = ctx->switches_generation != nullptr ? ctx->switches_generation() : 0;
        word                      = (generation << 1) | (share ? 1 : 0);
    }
    // the key alone, before the contract check
    spacemit_plan_result plan = spacemit_plan_result::off;
    if (cached) {
        plan = ctx->plan.probe(cgraph, word, want_names, ctx->steps, ctx->step_names, scratch_size, act_size);
    }
    const bool trusted = plan == spacemit_plan_result::hit_uid;
    // a checked node's steps and its part of the workspace: every graph when the cache is off (none, or
    // FLAGOS_SPACEMIT_QUANTIZE=shared), a miss of the plan cache otherwise (expanded as the check goes, then stored)
    const auto expand = [&](ggml_tensor * node, const spacemit_op * op) {
        uint32_t first = 0;
        if (share) {
            if (held != nullptr && spacemit_overlaps(node, held->src[1])) {
                held = nullptr;  // the node writes over the activations held
            }
            if (node->op == GGML_OP_MUL_MAT) {
                if (held != nullptr && ctx->same_activations(held, node)) {
                    first = 1;
                } else {
                    held = node;
                }
            }
        }
        for (uint32_t s = first; s < op->n_steps; s++) {
            ctx->steps.push_back({ op->steps[s]->fn, node, 0 });
            if (want_names) {  // the step names, kept parallel to steps only for the counters or the trace
                ctx->step_names.push_back(op->steps[s]->name);
            }
        }
        const size_t workspace = spacemit_op_workspace(op, node);
        if (workspace != 0) {
            size_t & size = node->op == GGML_OP_MUL_MAT ? act_size : scratch_size;
            size          = std::max(size, workspace);
        }
    };
    const int n_nodes = ggml_graph_n_nodes(cgraph);
    if (trusted) {
        // a trusted hit: no contract check, no expansion. The counters' kind of graph is still read, by the counters
        // and only for them, with the check's filter
        for (int i = 0; timed && i < n_nodes && (rows < 0 || n_kv < 0); i++) {
            const ggml_tensor * node = ggml_graph_node(cgraph, i);
            if (ggml_op_is_empty(node->op) || ggml_is_empty(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                continue;
            }
            if (rows < 0 && node->op == GGML_OP_MUL_MAT) {
                rows = node->src[1]->ne[1];
            }
            if (n_kv < 0 && node->op == GGML_OP_FLASH_ATTN_EXT) {
                n_kv = node->src[1]->ne[1];
            }
        }
    } else {
        // the contract check, every graph but a trusted hit: a graph either runs completely or fails with nothing
        // written
        for (int i = 0; i < n_nodes; i++) {
            ggml_tensor * node = ggml_graph_node(cgraph, i);
            // as ggml-cpu: nothing to do for empty tensors, and nodes without the compute flag are branches the graph
            // left unselected (ggml_build_forward_select), whose inputs may never have been set
            if (ggml_op_is_empty(node->op) || ggml_is_empty(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                continue;
            }
            if (timed && rows < 0 && node->op == GGML_OP_MUL_MAT) {
                rows = node->src[1]->ne[1];
            }
            if (timed && n_kv < 0 && node->op == GGML_OP_FLASH_ATTN_EXT) {
                n_kv = node->src[1]->ne[1];
            }
            const spacemit_op * op = spacemit_find_op(dev->ops, node, &dev->buffer_type);
            if (op == nullptr) {
                GGML_LOG_ERROR("FlagOS SpacemiT: no kernel for %s (%s)\n", ggml_op_desc(node), node->name);
                return GGML_STATUS_FAILED;
            }
            if (!spacemit_data_ready(node)) {
                GGML_LOG_ERROR("FlagOS SpacemiT: %s (%s) has an unallocated operand\n", ggml_op_desc(node), node->name);
                return GGML_STATUS_FAILED;
            }
            expand(node, op);
        }
    }
    // after the contract check; start when a trusted hit skipped it
    const std::chrono::steady_clock::time_point checked =
        clocked && !trusted ? std::chrono::steady_clock::now() : start;
    if (cached && !trusted) {
        // the probe missed: the checked graph's expansion stored, without a second lookup
        ctx->plan.store(cgraph, word, ctx->steps, ctx->step_names, scratch_size, act_size);
    }
    const std::chrono::steady_clock::time_point in_hand =
        clocked ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const double lookup_us = std::chrono::duration<double, std::micro>(in_hand - start).count();
    const double key_us    = std::chrono::duration<double, std::micro>(in_hand - checked).count();
    if (cached) {
        spacemit_plan_counters_add(plan, lookup_us, key_us);
    }
    if (ctx->steps.empty()) {
        return GGML_STATUS_SUCCESS;
    }
    const size_t act_offset     = GGML_PAD(scratch_size, 64);
    const size_t workspace_size = act_offset + act_size;
    for (spacemit_plugin_step & step : ctx->steps) {
        step.workspace_offset = step.node->op == GGML_OP_MUL_MAT ? act_offset : 0;
    }
    if (workspace_size > ctx->workspace_size) {
        if (ctx->workspace != nullptr) {
            ggml_aligned_free(ctx->workspace, ctx->workspace_size);
        }
        ctx->workspace      = ggml_aligned_malloc(workspace_size);
        ctx->workspace_size = ctx->workspace != nullptr ? workspace_size : 0;
        if (ctx->workspace == nullptr) {
            GGML_LOG_ERROR("FlagOS SpacemiT: failed to allocate a %zu-byte workspace\n", workspace_size);
            return GGML_STATUS_FAILED;
        }
    }

    std::lock_guard<std::mutex> lock(dev->run_mutex);
    const spacemit_launch_kind  mode = spacemit_launch_mode();
    if (timed && ctx->attach == nullptr) {
        static std::atomic<bool> warned{ false };
        if (!warned.exchange(true)) {
            GGML_LOG_WARN("FlagOS SpacemiT: the plugin has no step counters; the test statistics stay off\n");
        }
    }
    const bool counted = timed && ctx->attach != nullptr;
    if (counted) {  // the counters attach to the stream for this graph only; an uncounted graph leaves them detached
        ctx->run_stats.reset(ctx->steps.size());
        ctx->attach(ctx->stream, &ctx->run_stats);
    }
    // test-only: GGML_FLAGOS_SYNC_LAUNCH, one launch per step, is a flag of the replay
    const uint32_t flags   = mode == spacemit_launch_kind::per_step ? SPACEMIT_REPLAY_SYNC_LAUNCH : 0u;
    const size_t   n_steps = ctx->steps.size();
    if (trace) {
        spacemit_trace_steps(ctx->steps, ctx->step_names);
        GGML_LOG_INFO("FlagOS SpacemiT: launch %zu steps\n", n_steps);
    }
    enum spacemit_status status = SPACEMIT_STATUS_SUCCESS;
    if (trace && mode == spacemit_launch_kind::per_step) {
        // the trace with one launch per step: the core issues the replays itself, one step each (the same launches the
        // plugin would make under the flag), so that it names the step whose launch failed
        for (size_t i = 0; i < n_steps && status == SPACEMIT_STATUS_SUCCESS; i++) {
            status = ctx->table->replay(ctx->stream, ctx->steps.data() + i, 1, ctx->workspace, ctx->workspace_size,
                                        flags);
            GGML_LOG_INFO("FlagOS SpacemiT: launch step %zu/%zu: %s\n", i + 1, n_steps,
                          spacemit_status_name(status));
        }
    } else {
        status = ctx->table->replay(ctx->stream, ctx->steps.data(), static_cast<uint32_t>(n_steps), ctx->workspace,
                                    ctx->workspace_size, flags);
    }
    if (trace) {
        GGML_LOG_INFO("FlagOS SpacemiT: launch %zu steps: %s\n", n_steps, spacemit_status_name(status));
    }
    const bool ok = status == SPACEMIT_STATUS_SUCCESS;
    if (counted) {
        ctx->attach(ctx->stream, nullptr);
    }
    if (ok && counted) {
        const char * launch   = mode == spacemit_launch_kind::per_step ? "per-step" : "per-split";
        const double graph_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        if (stats != nullptr) {
            stats->add(launch, rows, n_kv, ctx->run_stats, graph_us, plan, lookup_us, key_us);
        }
        if (op_stats != nullptr) {
            op_stats->add(launch, rows, n_kv, ctx->steps, ctx->step_names, ctx->run_stats);
        }
    }
    return ok ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

const ggml_backend_i g_backend_iface = {
    /* .get_name            = */ spacemit_backend_name,
    /* .free                = */ spacemit_backend_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ nullptr,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ spacemit_backend_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

//
// device
//

spacemit_device_context * spacemit_device_from_dev(ggml_backend_dev_t dev) {
    return static_cast<spacemit_device_context *>(dev->context);
}

const char * spacemit_device_name(ggml_backend_dev_t dev) {
    return spacemit_device_from_dev(dev)->name.c_str();
}

const char * spacemit_device_description(ggml_backend_dev_t dev) {
    return spacemit_device_from_dev(dev)->description.c_str();
}

// the AI cores use the same DRAM as the CPU cores (the plugin reads it)
void spacemit_device_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * ctx = spacemit_device_from_dev(dev);
    *free      = 0;
    *total     = 0;
    spacemit_buffer_status("get_memory", ctx->table->device_memory(ctx->dev, free, total));
}

// IGPU (D3): llama.cpp gives the device whole layers and their KV cache, which lives in our buffer
enum ggml_backend_dev_type spacemit_device_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_IGPU;
}

void spacemit_device_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = spacemit_device_name(dev);
    props->description = spacemit_device_description(dev);
    spacemit_device_memory(dev, &props->memory_free, &props->memory_total);
    props->type      = spacemit_device_type(dev);
    props->device_id = nullptr;
    props->caps      = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ true,
    };
}

// one stream handle per backend context; the plugin shares one executor and core grant among a device's handles until
// the last is destroyed
ggml_backend_t spacemit_device_init(ggml_backend_dev_t dev, const char *) {
    auto *                   dctx   = spacemit_device_from_dev(dev);
    spacemit_plugin_stream * stream = nullptr;
    const enum spacemit_status status = dctx->table->stream_create(dctx->dev, dctx->n_ai_cores, &stream);
    if (status != SPACEMIT_STATUS_SUCCESS) {
        GGML_LOG_ERROR("FlagOS SpacemiT: the plugin could not create a stream: %s\n", spacemit_status_name(status));
        return nullptr;
    }
    auto * ctx  = new spacemit_backend_context;
    ctx->table  = dctx->table;
    ctx->stream = stream;
    // test-only (E3c, FLAGOS_SPACEMIT_QUANTIZE=shared), and the plugin's switch generation, in the plan cache's word
    // (moved only by a test's switch reset); nullptr: 0. Neither the per-tile clock reads, the activation-sharing fact
    // nor the switch generation is contract, so all three come by name (plan.md "S1 design" A2)
    ctx->attach = reinterpret_cast<spacemit_run_stats_attach_fn>(dctx->table->get_proc_address("spacemit_run_stats_attach"));
    ctx->same_activations = reinterpret_cast<spacemit_mul_mat_same_activations_fn>(
        dctx->table->get_proc_address("spacemit_mul_mat_same_activations"));
    ctx->switches_generation = reinterpret_cast<spacemit_switches_generation_fn>(
        dctx->table->get_proc_address("spacemit_switches_generation"));
    return new ggml_backend{
        /* .guid    = */ &g_backend_guid,
        /* .iface   = */ g_backend_iface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
}

ggml_backend_buffer_type_t spacemit_device_buffer_type(ggml_backend_dev_t dev) {
    return &spacemit_device_from_dev(dev)->buffer_type;
}

// inputs (NONE) and views need no kernel; claiming them lets graphs that contain them run here (design §7.4)
bool spacemit_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    spacemit_device_context * dctx = spacemit_device_from_dev(dev);
    return ggml_op_is_empty(op->op) || spacemit_find_op(dctx->ops, op, &dctx->buffer_type) != nullptr;
}

// operands may also live in CPU buffers: the activations of the CPU's splits
bool spacemit_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == &spacemit_device_from_dev(dev)->buffer_type || ggml_backend_buft_is_host(buft);
}

const ggml_backend_device_i g_device_iface = {
    /* .get_name             = */ spacemit_device_name,
    /* .get_description      = */ spacemit_device_description,
    /* .get_memory           = */ spacemit_device_memory,
    /* .get_type             = */ spacemit_device_type,
    /* .get_props            = */ spacemit_device_props,
    /* .init_backend         = */ spacemit_device_init,
    /* .get_buffer_type      = */ spacemit_device_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ spacemit_device_supports_op,
    /* .supports_buft        = */ spacemit_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

//
// probe: the plugin is loaded, checked and asked for its device; the K3 itself is detected inside the plugin (its A100
// cores in /proc/cpuinfo and the AI-thread gate)
//

void spacemit_host_log(int level, const char * text) {
    ggml_log_internal((enum ggml_log_level) level, "%s\n", text);
}

bool spacemit_probe_impl(ggml_backend_reg_t owner_reg) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_probed) {
        return g_present;
    }
    g_probed = true;

    // test switch: compare model output with and without the provider
    const char * disable = std::getenv("FLAGOS_SPACEMIT_DISABLE");
    if (disable != nullptr && std::strcmp(disable, "0") != 0) {
        GGML_LOG_INFO("FlagOS SpacemiT: provider disabled by FLAGOS_SPACEMIT_DISABLE\n");
        return false;
    }

    // the host table the loader hands to the entry, with the directory the plugin came from; static so the strings
    // outlive the call whatever a plugin does with them (ours copies what it keeps)
    static std::string      plugin_dir;
    static spacemit_host_v1 host;
    const std::string       path = spacemit_plugin_default_path();  // the same path the loader resolves
    const size_t            cut  = path.find_last_of('/');
    plugin_dir                   = cut != std::string::npos ? path.substr(0, cut) : std::string(".");
    host.struct_size             = sizeof(spacemit_host_v1);
    host.log                     = spacemit_host_log;
    host.plugin_dir              = plugin_dir.c_str();

    spacemit_plugin * plugin = nullptr;
    std::string       detail;
    const spacemit_load_result loaded = spacemit_plugin_load(nullptr, host, spacemit_required_kernels,
                                                             spacemit_n_required_kernels, &plugin, &detail);
    if (loaded != spacemit_load_result::ok) {
        if (loaded == spacemit_load_result::no_device) {
            // the ordinary case on a host without the AI cores
            GGML_LOG_DEBUG("FlagOS SpacemiT: no device: %s (%s)\n", detail.c_str(), path.c_str());
        } else {
            GGML_LOG_WARN("FlagOS SpacemiT: the plugin %s was not loaded: %s\n", path.c_str(), detail.c_str());
        }
        return false;
    }

    const spacemit_plugin_v1 * table = spacemit_plugin_table(plugin);
    spacemit_plugin_device *   pdev  = nullptr;
    enum spacemit_status       status = table->device_open(0, &pdev);
    if (status != SPACEMIT_STATUS_SUCCESS) {
        GGML_LOG_WARN("FlagOS SpacemiT: the plugin's device 0 did not open: %s\n", spacemit_status_name(status));
        spacemit_plugin_unload(plugin);
        return false;
    }
    spacemit_plugin_device_props props = {};
    props.struct_size                  = sizeof(props);
    status                             = table->device_props(pdev, &props);
    if (status != SPACEMIT_STATUS_SUCCESS) {
        GGML_LOG_WARN("FlagOS SpacemiT: the plugin's device properties are not available: %s\n",
                      spacemit_status_name(status));
        table->device_close(pdev);
        spacemit_plugin_unload(plugin);
        return false;
    }

    // the core's contracts bound to this plugin's kernels, once: a plugin whose kernels do not answer every contract
    // cannot run a graph, so it registers no device
    if (!spacemit_ops_bind(table, &g_device.ops, &detail)) {
        GGML_LOG_WARN("FlagOS SpacemiT: the plugin's kernels do not bind to the core's contracts: %s\n", detail.c_str());
        table->device_close(pdev);
        spacemit_plugin_unload(plugin);
        return false;
    }

    g_device.name                 = "FlagOS:SpacemiT:0";
    g_device.description          = props.description != nullptr ? props.description : "";
    g_device.runtime              = props.runtime != nullptr ? props.runtime : "none";
    g_device.n_ai_cores           = props.n_tiles;
    g_device.plugin               = plugin;
    g_device.table                = table;
    g_device.dev                  = pdev;
    g_device.device.iface         = g_device_iface;
    g_device.device.reg           = owner_reg;
    g_device.device.context       = &g_device;
    g_device.buffer_type.iface    = g_buffer_type_iface;
    g_device.buffer_type.device   = &g_device.device;
    g_device.buffer_type.context  = &g_device;

    // llama.cpp would first read this switch inside common_fit_params's trial load, which passes lines below ERROR
    // on as DEBUG (common/fit.cpp:49-53), so it is read here, at registration, and its once-only warnings show
    (void) spacemit_op_disabled(GGML_OP_NONE);
    g_present = true;
    return true;
}

//
// flagos_provider_v1 callbacks
//

bool spacemit_probe(ggml_backend_reg_t owner_reg) {
    return spacemit_probe_impl(owner_reg);
}

size_t spacemit_device_count() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_present ? 1 : 0;
}

ggml_backend_dev_t spacemit_device_get(size_t index) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_present && index == 0 ? &g_device.device : nullptr;
}

bool spacemit_device_identity(size_t index, flagos_device_identity * identity) {
    if (identity == nullptr || index != 0) {
        return false;
    }
    *identity = {
        /* .provider_id      = */ SPACEMIT_PROVIDER_ID,
        /* .uuid_hi          = */ SPACEMIT_DEVICE_UUID_HI,
        /* .uuid_lo          = */ SPACEMIT_DEVICE_UUID_LO,
        /* .memory_domain_id = */ SPACEMIT_MEMORY_DOMAIN_ID,
        /* .ordinal          = */ 0,
    };
    return true;
}

bool spacemit_device_caps(size_t index, flagos_device_caps * caps) {
    if (caps == nullptr || index != 0) {
        return false;
    }
    *caps = {
        // proposed D8 (plan.md): the A100 cores run RISC-V code with RVV and IME
        /* .kind       = */ flagos_provider_kind::cpu_accelerator,
        /* .memory     = */ FLAGOS_MEMORY_HOST_VISIBLE | FLAGOS_MEMORY_UNIFIED_COHERENT,
        /* .execution  = */ 0,
        /* .aot_format = */ nullptr,
    };
    return true;
}

bool spacemit_device_profile(size_t index, flagos_device_profile * profile) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (profile == nullptr || index != 0 || !g_present) {
        return false;
    }
    *profile = {};
    profile->struct_size       = sizeof(flagos_device_profile);
    profile->version           = 1;
    profile->engine            = flagos_engine_kind::cpu;
    profile->features          = FLAGOS_FEATURE_SIMD | FLAGOS_FEATURE_FP16 | FLAGOS_FEATURE_INT8 |
                                 FLAGOS_FEATURE_MATRIX | FLAGOS_FEATURE_UNIFIED_MEMORY;
    profile->vector_bits       = SPACEMIT_A100_VECTOR_BITS;
    profile->lane_count        = SPACEMIT_A100_VECTOR_BITS / 32;
    profile->concurrency       = g_device.n_ai_cores;
    profile->memory_domain_id  = SPACEMIT_MEMORY_DOMAIN_ID;
    profile->vendor            = "SpacemiT";
    profile->architecture      = "a100";
    profile->microarchitecture = "rvv1024-ime2";
    profile->target            = "riscv64-a100-ime2";
    profile->runtime           = g_device.runtime.c_str();  // the plugin's props
    profile->aot_format        = nullptr;
    return flagos_device_profile_is_valid(profile);
}

bool spacemit_is_backend(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, &g_backend_guid);
}

bool spacemit_set_device(size_t index) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_present && index == 0;
}

int spacemit_get_device() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_present ? 0 : -1;
}

int spacemit_score() {
    return spacemit_probe_impl(nullptr) ? 1 : 0;
}

}  // namespace

const spacemit_plugin * spacemit_device_plugin(ggml_backend_dev_t dev) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return dev != nullptr && dev->context == &g_device ? g_device.plugin : nullptr;
}

const flagos_provider_v1 * flagos_spacemit_provider() {
    static const flagos_provider_v1 provider = {
        /* .api_version        = */ FLAGOS_PROVIDER_API_VERSION,
        /* .struct_size        = */ sizeof(flagos_provider_v1),
        /* .identity           = */ { SPACEMIT_PROVIDER_ID, "SpacemiT", 1 },
        /* .probe              = */ spacemit_probe,
        /* .device_count       = */ spacemit_device_count,
        /* .device_get         = */ spacemit_device_get,
        /* .device_identity    = */ spacemit_device_identity,
        /* .device_caps        = */ spacemit_device_caps,
        /* .is_backend         = */ spacemit_is_backend,
        /* .set_device         = */ spacemit_set_device,
        /* .get_device         = */ spacemit_get_device,
        /* .score              = */ spacemit_score,
        /* .get_device_profile = */ spacemit_device_profile,
    };
    return &provider;
}
