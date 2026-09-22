#include "flagos-amd-api.h"
#include "flagos-amd-aot.h"

#include "../../../ggml-backend-impl.h"
#include "../../../ggml-impl.h"
#include "../../flagos-graph-plan.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace {

// Legacy fallback for packages generated before the multidimensional tile
// fields were added to manifest.json. New packages carry tile_m/tile_n and
// override these values at launch time.
constexpr unsigned AMD_F16_MATMUL_BLOCK_M = 64;
constexpr unsigned AMD_F16_MATMUL_BLOCK_N = 32;
constexpr unsigned AMD_QUANT_TILE_BLOCK_M = 16;
constexpr unsigned AMD_QUANT_TILE_BLOCK_N = 32;
constexpr size_t AMD_DEQUANT_ARENA_ALIGNMENT = 256;
constexpr size_t AMD_DEQUANT_ARENA_CHUNK_BYTES = size_t(512) * 1024 * 1024;

static bool amd_hip_check(hipError_t result, const char * expression, bool log_error = true) {
    if (result == hipSuccess) {
        return true;
    }
    if (log_error) {
        GGML_LOG_ERROR("FlagOS AMD: %s failed: %s\n", expression, hipGetErrorString(result));
    }
    return false;
}

static void amd_log_device_access_hint() {
#if !defined(_WIN32)
    if (access("/dev/kfd", F_OK) == 0 && access("/dev/kfd", R_OK | W_OK) != 0) {
        GGML_LOG_ERROR("FlagOS AMD: /dev/kfd is present but inaccessible; add the user to the render group and start a new login session\n");
    }
#endif
}

struct amd_device_context;

struct amd_buffer_context {
    amd_device_context * device = nullptr;
    void * data = nullptr;
    size_t size = 0;
    uint64_t allocation_id = 0;
    std::shared_ptr<uint8_t> lifetime = std::make_shared<uint8_t>(0);
    std::atomic<uint64_t> generation { 1 };

    amd_buffer_context(amd_device_context * device_, void * data_, size_t size_, uint64_t allocation_id_):
        device(device_), data(data_), size(size_), allocation_id(allocation_id_) {}
};

static bool amd_buffer_is_local_storage(ggml_backend_buffer_t buffer);

static amd_buffer_context * amd_buffer_from_buffer(ggml_backend_buffer_t buffer) {
    // Tensor-facing callbacks can be reached through scheduler fallback paths.
    // Never reinterpret a foreign backend's opaque buffer context as AMD
    // state; owning buffer callbacks still pass this identity check.
    return amd_buffer_is_local_storage(buffer)
        ? static_cast<amd_buffer_context *>(buffer->context) : nullptr;
}

static ggml_backend_buffer_t amd_tensor_buffer(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return nullptr;
    }
    // Resolve nested views to their backing allocation.  The bound prevents a
    // malformed view cycle from turning validation into an infinite loop.
    const ggml_tensor * storage = tensor;
    for (int depth = 0; depth <= GGML_MAX_SRC; ++depth) {
        if (storage->view_src == nullptr) {
            return storage->buffer;
        }
        storage = storage->view_src;
        if (storage == nullptr) {
            return nullptr;
        }
    }
    return nullptr;
}

static const amd_buffer_context * amd_buffer_from_tensor(const ggml_tensor * tensor) {
    return amd_buffer_from_buffer(amd_tensor_buffer(tensor));
}

static void amd_mark_buffer_modified(amd_buffer_context * buffer) {
    if (buffer != nullptr) {
        buffer->generation.fetch_add(1, std::memory_order_release);
    }
}

struct amd_event_context {
    int device = -1;
    hipEvent_t event = nullptr;
};

struct amd_graph_capture_entry {
    uint64_t fingerprint = 0;
    flagos_graph_binding_snapshot bindings;
    hipGraphExec_t executable = nullptr;
    uint64_t last_used = 0;
};

struct amd_graph_capture_candidate {
    uint64_t fingerprint = 0;
    flagos_graph_binding_snapshot bindings;
    uint64_t last_used = 0;
};

struct amd_graph_capture_scope {
    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    bool active = false;

    explicit amd_graph_capture_scope(hipStream_t stream_) : stream(stream_) {}

    ~amd_graph_capture_scope() {
        if (active) {
            hipGraph_t abandoned = nullptr;
            amd_hip_check(hipStreamEndCapture(stream, &abandoned), "hipStreamEndCapture (abort)");
            if (abandoned != nullptr) {
                amd_hip_check(hipGraphDestroy(abandoned), "hipGraphDestroy (abort)");
            }
        }
        if (graph != nullptr) {
            amd_hip_check(hipGraphDestroy(graph), "hipGraphDestroy (scope)");
        }
    }

    bool begin() {
        if (stream == nullptr || !amd_hip_check(hipStreamBeginCapture(
                stream, hipStreamCaptureModeRelaxed), "hipStreamBeginCapture")) {
            return false;
        }
        active = true;
        return true;
    }

    bool finish(hipGraphExec_t * executable) {
        if (executable == nullptr || !active) {
            return false;
        }
        *executable = nullptr;
        active = false;
        if (!amd_hip_check(hipStreamEndCapture(stream, &graph), "hipStreamEndCapture") ||
            graph == nullptr) {
            return false;
        }
        if (!amd_hip_check(hipGraphInstantiate(executable, graph, nullptr, nullptr, 0),
                "hipGraphInstantiate")) {
            *executable = nullptr;
            return false;
        }
        if (!amd_hip_check(hipGraphDestroy(graph), "hipGraphDestroy")) {
            amd_hip_check(hipGraphExecDestroy(*executable), "hipGraphExecDestroy after graph destroy failure");
            *executable = nullptr;
            return false;
        }
        graph = nullptr;
        return true;
    }
};

struct amd_dequant_cache_entry {
    const ggml_tensor * tensor = nullptr;
    const void * source = nullptr;
    const amd_buffer_context * source_buffer = nullptr;
    uint64_t source_allocation_id = 0;
    std::weak_ptr<uint8_t> source_lifetime;
    uint64_t source_generation = 0;
    flagos_quantized_matmul_kind kind = flagos_quantized_matmul_kind::none;
    int k = 0;
    int rows = 0;
    void * f16_data = nullptr;
    size_t bytes = 0;
    // A buffer write can make the private F16 copy stale before this weight is
    // encountered again by the ordinary execution path.  Keep that state
    // separate from source_generation so graph replay can be invalidated at
    // graph entry without pretending the F16 allocation has been refreshed.
    bool valid = true;
};

struct amd_dequant_arena_block {
    void * data = nullptr;
    size_t capacity = 0;
    size_t used = 0;
};

struct amd_backend_context {
    amd_device_context * device = nullptr;
    hipStream_t stream = nullptr;
    std::mutex execution_mutex;
    flagos_graph_plan_cache graph_plans;
    uint64_t graph_plan_config_key = 0;
    std::vector<amd_graph_capture_entry> graph_captures;
    std::vector<amd_graph_capture_candidate> graph_capture_candidates;
    std::vector<amd_dequant_cache_entry> dequant_cache;
    std::vector<amd_dequant_arena_block> dequant_arena;
    size_t dequant_cache_bytes = 0;
    size_t dequant_arena_bytes = 0;
    size_t dequant_cache_limit_bytes = 0;
    uint64_t dequant_cache_epoch = 0;
    bool dequant_cache_limit_logged = false;
    uint64_t graph_capture_tick = 0;
    uint64_t profile_dump_sync_interval = 0;
    uint64_t profile_sync_count = 0;

    struct stats_t {
        std::atomic<uint64_t> kernel_launches { 0 };
        std::atomic<uint64_t> direct_ops { 0 };
        std::atomic<uint64_t> fusion_steps { 0 };
        std::atomic<uint64_t> fusion_rms_norm_mul { 0 };
        std::atomic<uint64_t> fusion_rms_norm_mul_narrow { 0 };
        std::atomic<uint64_t> fusion_rms_norm_mul_rope { 0 };
        std::atomic<uint64_t> fusion_rms_norm_mul_rope_kv_store { 0 };
        std::atomic<uint64_t> fusion_add_rms_norm_mul { 0 };
        std::atomic<uint64_t> fusion_rope_kv_store { 0 };
        std::atomic<uint64_t> fusion_flash_attn_decode { 0 };
        std::atomic<uint64_t> fusion_flash_attn_prefill { 0 };
        std::atomic<uint64_t> fusion_gated_delta_net_cache { 0 };
        std::atomic<uint64_t> fusion_gated_delta_net_cache_only { 0 };
        std::atomic<uint64_t> fusion_gated_delta_net_cache_only_decode { 0 };
        std::atomic<uint64_t> fusion_ssm_conv_silu { 0 };
        std::atomic<uint64_t> fusion_attention_output_gate { 0 };
        std::atomic<uint64_t> fusion_ffn_swiglu { 0 };
        std::atomic<uint64_t> fusion_ffn_swiglu_q40_staged { 0 };
        std::atomic<uint64_t> fusion_ffn_swiglu_down { 0 };
        std::atomic<uint64_t> q40_matmul { 0 };
        std::atomic<uint64_t> q41_matmul { 0 };
        std::atomic<uint64_t> q80_matmul { 0 };
        std::atomic<uint64_t> q5_matmul { 0 };
        std::atomic<uint64_t> q4_matmul { 0 };
        std::atomic<uint64_t> q6_matmul { 0 };
        std::atomic<uint64_t> q40_matmul_batched { 0 };
        std::atomic<uint64_t> q41_matmul_batched { 0 };
        std::atomic<uint64_t> q80_matmul_batched { 0 };
        std::atomic<uint64_t> q5_matmul_batched { 0 };
        std::atomic<uint64_t> q4_matmul_batched { 0 };
        std::atomic<uint64_t> q6_matmul_batched { 0 };
        std::atomic<uint64_t> q4_matmul_tiled { 0 };
        std::atomic<uint64_t> q6_matmul_tiled { 0 };
        std::atomic<uint64_t> weight_dequantizations { 0 };
        std::atomic<uint64_t> f16_matmul_batched { 0 };
        std::atomic<uint64_t> q40_get_rows { 0 };
        std::atomic<uint64_t> q41_get_rows { 0 };
        std::atomic<uint64_t> q80_get_rows { 0 };
        std::atomic<uint64_t> q5_get_rows { 0 };
        std::atomic<uint64_t> q4_get_rows { 0 };
        std::atomic<uint64_t> q6_get_rows { 0 };
        std::atomic<uint64_t> mul { 0 };
        std::atomic<uint64_t> scale { 0 };
        std::atomic<uint64_t> rope { 0 };
        std::atomic<uint64_t> rope_kv_store { 0 };
        std::atomic<uint64_t> flash_attn_decode { 0 };
        std::atomic<uint64_t> flash_attn_prefill { 0 };
        std::atomic<uint64_t> soft_max { 0 };
        std::atomic<uint64_t> host_to_device_copies { 0 };
        std::atomic<uint64_t> device_to_device_copies { 0 };
        std::atomic<uint64_t> graph_plans_built { 0 };
        std::atomic<uint64_t> graph_plan_direct_steps { 0 };
        std::atomic<uint64_t> graph_plan_pattern_steps { 0 };
        std::atomic<uint64_t> graph_captures { 0 };
        std::atomic<uint64_t> graph_replays { 0 };
        std::atomic<uint64_t> graph_replay_failures { 0 };
        std::atomic<uint64_t> graph_capture_failures { 0 };
        std::atomic<uint64_t> graph_capture_requests { 0 };
        std::atomic<uint64_t> graph_capture_cache_waits { 0 };
        std::atomic<uint64_t> graph_capture_prefill_skips { 0 };
        std::atomic<uint64_t> graph_capture_unsafe_plans { 0 };
        std::atomic<uint64_t> graph_capture_small_graphs { 0 };
        std::atomic<uint64_t> graph_capture_eligible { 0 };
        std::atomic<uint64_t> graph_capture_exec_misses { 0 };
        std::atomic<uint64_t> graph_capture_pointer_misses { 0 };
        std::atomic<uint64_t> graph_capture_warmups { 0 };
        std::atomic<uint64_t> graph_capture_attempts { 0 };
        std::atomic<uint64_t> graph_capture_exec_evictions { 0 };
        std::atomic<uint64_t> graph_capture_candidate_evictions { 0 };
    } stats;

    amd_backend_context(amd_device_context * device_, hipStream_t stream_):
        device(device_), stream(stream_), graph_plans(128) {
        graph_captures.reserve(8);
        graph_capture_candidates.reserve(16);
        dequant_cache.reserve(64);
        dequant_arena.reserve(8);
        // Bound F16 cache residency on larger models.
        const char * limit_mb = std::getenv("FLAGOS_AMD_DEQUANT_CACHE_MAX_MB");
        if (limit_mb == nullptr || limit_mb[0] == '\0') {
            dequant_cache_limit_bytes = size_t(8192) * 1024 * 1024;
        } else {
            char * end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(limit_mb, &end, 10);
            if (limit_mb[0] >= '0' && limit_mb[0] <= '9' &&
                end != limit_mb && *end == '\0' && errno != ERANGE &&
                parsed <= SIZE_MAX / (1024ULL * 1024ULL)) {
                dequant_cache_limit_bytes = static_cast<size_t>(parsed) * 1024 * 1024;
            } else {
                GGML_LOG_WARN("FlagOS AMD: invalid FLAGOS_AMD_DEQUANT_CACHE_MAX_MB=%s; using 8192 MiB\n",
                    limit_mb);
                dequant_cache_limit_bytes = size_t(8192) * 1024 * 1024;
            }
        }

        // llama-bench keeps the provider registry alive until process exit,
        // so backend destruction is not a reliable profile dump point there.
        // Keep synchronization diagnostics explicitly opt-in and allow an
        // interval so long decode runs do not flood stderr.  The dump is
        // cumulative; the last snapshot therefore covers the full window.
        const char * dump_interval = std::getenv("FLAGOS_PROFILE_DUMP_SYNC_INTERVAL");
        if (dump_interval != nullptr && dump_interval[0] != '\0') {
            char * end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(dump_interval, &end, 10);
            if (dump_interval[0] >= '0' && dump_interval[0] <= '9' &&
                end != dump_interval && *end == '\0' && errno != ERANGE &&
                parsed <= std::numeric_limits<uint64_t>::max()) {
                profile_dump_sync_interval = static_cast<uint64_t>(parsed);
            } else {
                GGML_LOG_WARN("FlagOS AMD: invalid FLAGOS_PROFILE_DUMP_SYNC_INTERVAL=%s; disabling sync profile dumps\n",
                    dump_interval);
            }
        }
    }
};

static bool amd_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool amd_auto_tuning_enabled(const amd_device_context * device);
static bool amd_qwen35_v2_profile(const amd_device_context * device);

static bool amd_graph_capture_enabled() {
    return amd_env_enabled("FLAGOS_AMD_GRAPH_CAPTURE");
}

static bool amd_graph_capture_prefill_enabled() {
    const char * value = std::getenv("FLAGOS_AMD_GRAPH_CAPTURE");
    return value != nullptr && std::strcmp(value, "all") == 0;
}

static bool amd_graph_is_decode(const ggml_cgraph * cgraph) {
    if (cgraph == nullptr) {
        return false;
    }
    bool found_projection = false;
    for (int index = 0; index < cgraph->n_nodes; ++index) {
        flagos_quantized_matmul_signature signature;
        if (!flagos_describe_quantized_matmul(cgraph->nodes[index], &signature)) {
            continue;
        }
        found_projection = true;
        if (signature.columns != 1) {
            return false;
        }
    }
    return found_projection;
}

// A tuned package enables its validated fusion subset.
static bool amd_experimental_fusions_enabled(const amd_device_context * device) {
    return std::getenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS") != nullptr
        ? amd_env_enabled("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS")
        : amd_auto_tuning_enabled(device);
}

// During bring-up, allow individual fusions to be enabled without changing
// the provider ABI or rebuilding the AOT package.  An unset selector keeps
// the historical behaviour (all experimental fusions); a comma-separated
// selector makes numerical bisects reproducible, for example:
//   FLAGOS_AMD_FUSIONS=rope_kv_store,flash_attn_decode
static bool amd_fusion_enabled(const amd_device_context * device, const char * name) {
    if (!amd_experimental_fusions_enabled(device) || name == nullptr) {
        return false;
    }
    if (std::getenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS") == nullptr) {
        static constexpr std::string_view VALIDATED[] = {
            "rms_norm_mul_rope", "rms_norm_mul", "add_rms_norm_mul", "rope_kv_store",
            "flash_attn_decode", "flash_attn_prefill", "ffn_swiglu", "ffn_swiglu_down",
        };
        const bool validated_gdn = std::strcmp(name, "gated_delta_net_cache") == 0 &&
            amd_qwen35_v2_profile(device);
        if (!validated_gdn &&
            std::find(std::begin(VALIDATED), std::end(VALIDATED), name) == std::end(VALIDATED)) {
            return false;
        }
    }
    const char * selected = std::getenv("FLAGOS_AMD_FUSIONS");
    if (selected == nullptr || selected[0] == '\0') {
        return true;
    }
    std::string_view list(selected);
    size_t begin = 0;
    while (begin < list.size()) {
        const size_t end = list.find(',', begin);
        const std::string_view token = list.substr(begin,
            end == std::string_view::npos ? list.size() - begin : end - begin);
        if (token == "all" || token == name) {
            return true;
        }
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1;
    }
    return false;
}

static bool amd_candidate_outputs_are(
        const flagos_pattern_candidate & candidate,
        std::initializer_list<size_t> positions) {
    if (candidate.required_output_node_indices.size() != positions.size()) {
        return false;
    }
    size_t output = 0;
    for (const size_t position : positions) {
        if (position >= candidate.node_indices.size() ||
            candidate.required_output_node_indices[output++] != candidate.node_indices[position]) {
            return false;
        }
    }
    return true;
}

static uint64_t amd_fusion_config_key(const amd_backend_context * context) {
    // The graph-plan cache is provider-local, but its lowering decisions are
    // controlled by process configuration. Include all lowering-related
    // switches in a small generation key so tests or embedding applications
    // that change the environment at runtime cannot reuse a plan built under
    // another policy.
    uint64_t hash = 1469598103934665603ULL;
    const auto add = [&hash](const char * value) {
        if (value == nullptr) {
            const uint8_t marker = 0;
            hash ^= marker;
            hash *= 1099511628211ULL;
            return;
        }
        for (const unsigned char * cursor = reinterpret_cast<const unsigned char *>(value);
             *cursor != '\0'; ++cursor) {
            hash ^= *cursor;
            hash *= 1099511628211ULL;
        }
        hash ^= 0xff;
        hash *= 1099511628211ULL;
    };
    add(std::getenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS"));
    add(std::getenv("FLAGOS_AMD_FUSIONS"));
    add(std::getenv("FLAGOS_AMD_PREFILL_F16_GEMM"));
    add(std::getenv("FLAGOS_AMD_GROUPED_F16_GEMM"));
    add(std::getenv("FLAGOS_AMD_FFN_DOWN_F16"));
    add(std::getenv("FLAGOS_AMD_Q4_GEMV_NARROW"));
    add(std::getenv("FLAGOS_AMD_Q5_GEMV_NARROW"));
    add(std::getenv("FLAGOS_AMD_Q40_GEMV_NARROW"));
    add(std::getenv("FLAGOS_AMD_Q40_FFN_DECODE_STAGED"));
    add(std::getenv("FLAGOS_AMD_TUNING_PROFILE"));
    // Query results also depend on whether persistent provider-private weight
    // caches are cold, warm, or invalidated. Fold their generation into the
    // same provider-neutral configuration ID used by the common plan cache.
    // This lets a cold FFN plan be capture-unsafe for its allocating launch,
    // then rebuild as capture-safe once all pointers are stable.
    const uint64_t epoch = context == nullptr ? 0 : context->dequant_cache_epoch;
    for (size_t byte = 0; byte < sizeof(epoch); ++byte) {
        hash ^= static_cast<uint8_t>(epoch >> (byte * 8));
        hash *= 1099511628211ULL;
    }
    return hash == 0 ? 1 : hash;
}

static void amd_trace_op(const amd_backend_context * context, const ggml_tensor * node,
                         const char * path) {
    if (context != nullptr && node != nullptr && amd_env_enabled("FLAGOS_TRACE_OPS")) {
        GGML_LOG_INFO("FlagOS AMD: op=%s path=%s ne=[%lld,%lld,%lld,%lld]\n",
            ggml_op_name(node->op), path == nullptr ? "direct" : path,
            (long long) node->ne[0], (long long) node->ne[1],
            (long long) node->ne[2], (long long) node->ne[3]);
    }
}

static void amd_clear_graph_captures(amd_backend_context * context);

static amd_graph_capture_entry * amd_find_graph_capture(
        amd_backend_context * context, uint64_t fingerprint, uint64_t binding_fingerprint,
        const ggml_cgraph * cgraph, const flagos_graph_plan & plan, bool * structure_found) {
    if (context == nullptr) {
        return nullptr;
    }
    if (structure_found != nullptr) {
        *structure_found = false;
    }
    for (auto & entry : context->graph_captures) {
        if (entry.fingerprint != fingerprint || entry.executable == nullptr) {
            continue;
        }
        if (structure_found != nullptr) {
            *structure_found = true;
        }
        if (entry.bindings.fingerprint == binding_fingerprint &&
            flagos_graph_binding_snapshot_matches(cgraph, plan, entry.bindings)) {
            entry.last_used = ++context->graph_capture_tick;
            return &entry;
        }
    }
    return nullptr;
}

static bool amd_store_graph_capture(
        amd_backend_context * context, uint64_t fingerprint,
        flagos_graph_binding_snapshot bindings, hipGraphExec_t executable) {
    if (context == nullptr || executable == nullptr) {
        return false;
    }
    constexpr size_t capacity = 8;
    if (context->graph_captures.size() >= capacity) {
        auto lru = std::min_element(context->graph_captures.begin(), context->graph_captures.end(),
            [](const amd_graph_capture_entry & left, const amd_graph_capture_entry & right) {
                return left.last_used < right.last_used;
            });
        if (lru != context->graph_captures.end()) {
            // HIP does not provide a provider-independent guarantee that an
            // executable may be destroyed while an earlier launch is still
            // using it.  Eviction is rare; synchronize the owning stream so
            // the bounded cache never trades safety for one asynchronous
            // teardown.
            if (!amd_hip_check(hipStreamSynchronize(context->stream),
                    "hipStreamSynchronize before graph exec eviction")) {
                return false;
            }
            amd_hip_check(hipGraphExecDestroy(lru->executable), "hipGraphExecDestroy");
            context->graph_captures.erase(lru);
            context->stats.graph_capture_exec_evictions.fetch_add(1, std::memory_order_relaxed);
        }
    }
    context->graph_captures.push_back({
        fingerprint, std::move(bindings), executable, ++context->graph_capture_tick});
    return true;
}

static void amd_discard_graph_capture(
        amd_backend_context * context, amd_graph_capture_entry * entry) {
    if (context == nullptr || entry == nullptr) {
        return;
    }
    const auto found = std::find_if(
        context->graph_captures.begin(), context->graph_captures.end(),
        [entry](const amd_graph_capture_entry & candidate) { return &candidate == entry; });
    if (found == context->graph_captures.end()) {
        return;
    }
    hipGraphExec_t executable = found->executable;
    found->executable = nullptr;
    context->graph_captures.erase(found);
    if (executable != nullptr) {
        if (amd_hip_check(hipStreamSynchronize(context->stream),
                "hipStreamSynchronize before failed graph exec destroy")) {
            amd_hip_check(hipGraphExecDestroy(executable), "hipGraphExecDestroy after replay failure");
        } else {
            GGML_LOG_ERROR("FlagOS AMD: cannot prove failed graph executable is idle; leaking it safely\n");
        }
    }
}

static bool amd_take_graph_capture_candidate(
        amd_backend_context * context, uint64_t fingerprint, uint64_t binding_fingerprint,
        const ggml_cgraph * cgraph, const flagos_graph_plan & plan,
        flagos_graph_binding_snapshot * bindings) {
    if (context == nullptr || bindings == nullptr) {
        return false;
    }
    for (auto candidate = context->graph_capture_candidates.begin();
         candidate != context->graph_capture_candidates.end(); ++candidate) {
        if (candidate->fingerprint == fingerprint &&
            candidate->bindings.fingerprint == binding_fingerprint &&
            flagos_graph_binding_snapshot_matches(cgraph, plan, candidate->bindings)) {
            *bindings = std::move(candidate->bindings);
            context->graph_capture_candidates.erase(candidate);
            return true;
        }
    }
    return false;
}

static void amd_store_graph_capture_candidate(
        amd_backend_context * context, uint64_t fingerprint,
        flagos_graph_binding_snapshot bindings) {
    if (context == nullptr) {
        return;
    }
    constexpr size_t capacity = 16;
    if (context->graph_capture_candidates.size() >= capacity) {
        const auto lru = std::min_element(
            context->graph_capture_candidates.begin(), context->graph_capture_candidates.end(),
            [](const amd_graph_capture_candidate & left, const amd_graph_capture_candidate & right) {
                return left.last_used < right.last_used;
            });
        if (lru != context->graph_capture_candidates.end()) {
            context->graph_capture_candidates.erase(lru);
            context->stats.graph_capture_candidate_evictions.fetch_add(1, std::memory_order_relaxed);
        }
    }
    context->graph_capture_candidates.push_back({
        fingerprint, std::move(bindings), ++context->graph_capture_tick});
}

static void amd_log_stats(const amd_backend_context * context) {
    if (context == nullptr || !amd_env_enabled("FLAGOS_LOG_KERNELS")) {
        return;
    }
    const auto & s = context->stats;
    GGML_LOG_INFO(
        "FlagOS AMD kernel stats: launches=%llu direct=%llu fusions=%llu "
        "q40_matmul=%llu q41_matmul=%llu q80_matmul=%llu q5_matmul=%llu q4_matmul=%llu q6_matmul=%llu "
        "q40_batched=%llu q41_batched=%llu q80_batched=%llu q5_batched=%llu q4_batched=%llu q6_batched=%llu "
        "q4_tiled=%llu q6_tiled=%llu "
        "weight_dequantizations=%llu f16_batched=%llu "
        "q40_get_rows=%llu q41_get_rows=%llu q80_get_rows=%llu q5_get_rows=%llu q4_get_rows=%llu q6_get_rows=%llu rope=%llu rope_kv_store=%llu "
        "mul=%llu scale=%llu flash_decode=%llu flash_prefill=%llu soft_max=%llu "
        "dequant_cache_mib=%zu/%zu arena_mib=%zu blocks=%zu "
        "host_to_device=%llu device_to_device=%llu "
        "plan_builds=%llu plan_direct=%llu plan_patterns=%llu "
        "plan_hits=%llu plan_misses=%llu plan_evictions=%llu "
        "fusion_rms_mul=%llu fusion_rms_mul_narrow=%llu fusion_rms_mul_rope=%llu "
        "fusion_rms_mul_rope_store=%llu "
        "fusion_add_rms_mul=%llu fusion_rope_store=%llu "
        "fusion_flash_decode=%llu fusion_flash_prefill=%llu "
        "fusion_gdn_cache=%llu fusion_gdn_cache_only=%llu fusion_gdn_cache_only_decode=%llu "
        "fusion_ssm_conv_silu=%llu fusion_attention_output_gate=%llu "
        "fusion_ffn_swiglu=%llu fusion_ffn_q40_staged=%llu fusion_ffn_swiglu_down=%llu "
        "graph_captures=%llu graph_replays=%llu graph_replay_failures=%llu graph_capture_failures=%llu\n",
        (unsigned long long) s.kernel_launches.load(),
        (unsigned long long) s.direct_ops.load(),
        (unsigned long long) s.fusion_steps.load(),
        (unsigned long long) s.q40_matmul.load(),
        (unsigned long long) s.q41_matmul.load(),
        (unsigned long long) s.q80_matmul.load(),
        (unsigned long long) s.q5_matmul.load(),
        (unsigned long long) s.q4_matmul.load(),
        (unsigned long long) s.q6_matmul.load(),
        (unsigned long long) s.q40_matmul_batched.load(),
        (unsigned long long) s.q41_matmul_batched.load(),
        (unsigned long long) s.q80_matmul_batched.load(),
        (unsigned long long) s.q5_matmul_batched.load(),
        (unsigned long long) s.q4_matmul_batched.load(),
        (unsigned long long) s.q6_matmul_batched.load(),
        (unsigned long long) s.q4_matmul_tiled.load(),
        (unsigned long long) s.q6_matmul_tiled.load(),
        (unsigned long long) s.weight_dequantizations.load(),
        (unsigned long long) s.f16_matmul_batched.load(),
        (unsigned long long) s.q40_get_rows.load(),
        (unsigned long long) s.q41_get_rows.load(),
        (unsigned long long) s.q80_get_rows.load(),
        (unsigned long long) s.q5_get_rows.load(),
        (unsigned long long) s.q4_get_rows.load(),
        (unsigned long long) s.q6_get_rows.load(),
        (unsigned long long) s.rope.load(),
        (unsigned long long) s.rope_kv_store.load(),
        (unsigned long long) s.mul.load(),
        (unsigned long long) s.scale.load(),
        (unsigned long long) s.flash_attn_decode.load(),
        (unsigned long long) s.flash_attn_prefill.load(),
        (unsigned long long) s.soft_max.load(),
        context->dequant_cache_bytes / (1024 * 1024),
        context->dequant_cache_limit_bytes / (1024 * 1024),
        context->dequant_arena_bytes / (1024 * 1024),
        context->dequant_arena.size(),
        (unsigned long long) s.host_to_device_copies.load(),
        (unsigned long long) s.device_to_device_copies.load(),
        (unsigned long long) s.graph_plans_built.load(),
        (unsigned long long) s.graph_plan_direct_steps.load(),
        (unsigned long long) s.graph_plan_pattern_steps.load(),
        (unsigned long long) context->graph_plans.hits(),
        (unsigned long long) context->graph_plans.misses(),
        (unsigned long long) context->graph_plans.evictions(),
        (unsigned long long) s.fusion_rms_norm_mul.load(),
        (unsigned long long) s.fusion_rms_norm_mul_narrow.load(),
        (unsigned long long) s.fusion_rms_norm_mul_rope.load(),
        (unsigned long long) s.fusion_rms_norm_mul_rope_kv_store.load(),
        (unsigned long long) s.fusion_add_rms_norm_mul.load(),
        (unsigned long long) s.fusion_rope_kv_store.load(),
        (unsigned long long) s.fusion_flash_attn_decode.load(),
        (unsigned long long) s.fusion_flash_attn_prefill.load(),
        (unsigned long long) s.fusion_gated_delta_net_cache.load(),
        (unsigned long long) s.fusion_gated_delta_net_cache_only.load(),
        (unsigned long long) s.fusion_gated_delta_net_cache_only_decode.load(),
        (unsigned long long) s.fusion_ssm_conv_silu.load(),
        (unsigned long long) s.fusion_attention_output_gate.load(),
        (unsigned long long) s.fusion_ffn_swiglu.load(),
        (unsigned long long) s.fusion_ffn_swiglu_q40_staged.load(),
        (unsigned long long) s.fusion_ffn_swiglu_down.load(),
        (unsigned long long) s.graph_captures.load(),
        (unsigned long long) s.graph_replays.load(),
        (unsigned long long) s.graph_replay_failures.load(),
        (unsigned long long) s.graph_capture_failures.load());
    GGML_LOG_INFO(
        "FlagOS AMD graph capture stats: requests=%llu cache_waits=%llu prefill_skips=%llu "
        "unsafe_plans=%llu small_graphs=%llu eligible=%llu exec_misses=%llu "
        "pointer_misses=%llu warmups=%llu attempts=%llu captures=%llu "
        "replays=%llu replay_failures=%llu failures=%llu exec_evictions=%llu candidate_evictions=%llu "
        "cached_exec=%zu pending=%zu\n",
        (unsigned long long) s.graph_capture_requests.load(),
        (unsigned long long) s.graph_capture_cache_waits.load(),
        (unsigned long long) s.graph_capture_prefill_skips.load(),
        (unsigned long long) s.graph_capture_unsafe_plans.load(),
        (unsigned long long) s.graph_capture_small_graphs.load(),
        (unsigned long long) s.graph_capture_eligible.load(),
        (unsigned long long) s.graph_capture_exec_misses.load(),
        (unsigned long long) s.graph_capture_pointer_misses.load(),
        (unsigned long long) s.graph_capture_warmups.load(),
        (unsigned long long) s.graph_capture_attempts.load(),
        (unsigned long long) s.graph_captures.load(),
        (unsigned long long) s.graph_replays.load(),
        (unsigned long long) s.graph_replay_failures.load(),
        (unsigned long long) s.graph_capture_failures.load(),
        (unsigned long long) s.graph_capture_exec_evictions.load(),
        (unsigned long long) s.graph_capture_candidate_evictions.load(),
        context->graph_captures.size(),
        context->graph_capture_candidates.size());
}

struct amd_device_context {
    int ordinal = -1;
    hipDeviceProp_t props {};
    std::array<uint8_t, 16> uuid {};
    std::string arch;
    std::string name;
    std::string description;
    std::string device_id;
    ggml_backend_device device_iface {};
    ggml_backend_buffer_type buffer_type_iface {};
    std::unique_ptr<flagos_amd::kernel_registry> aot;
    std::unique_ptr<std::mutex> backends_mutex = std::make_unique<std::mutex>();
    std::vector<amd_backend_context *> backends;
    bool aot_attempted = false;
    bool tuned_package = false;
    flagos_device_profile profile {};
};

static bool amd_qwen35_v2_profile(const amd_device_context * device) {
    return device != nullptr && device->aot != nullptr &&
        device->aot->tuning_profile() == flagos_amd::tuning_profile_gfx1150_qwen35_q4km_v2;
}

class amd_device_execution_guard {
public:
    explicit amd_device_execution_guard(amd_device_context * device):
        device_(device),
        registry_lock_(device != nullptr && device->backends_mutex != nullptr
            ? std::unique_lock<std::mutex>(*device->backends_mutex)
            : std::unique_lock<std::mutex>()) {
        if (!registry_lock_.owns_lock()) {
            return;
        }
        contexts_ = device_->backends;
        std::sort(contexts_.begin(), contexts_.end(), std::less<amd_backend_context *>());
        locks_.reserve(contexts_.size());
        for (amd_backend_context * context : contexts_) {
            if (context != nullptr) {
                locks_.emplace_back(context->execution_mutex);
            }
        }
    }

    bool valid() const {
        return device_ != nullptr && registry_lock_.owns_lock();
    }

    const std::vector<amd_backend_context *> & contexts() const {
        return contexts_;
    }

    bool synchronize() const {
        if (!valid() || !amd_hip_check(hipSetDevice(device_->ordinal), "hipSetDevice")) {
            return false;
        }
        for (amd_backend_context * context : contexts_) {
            if (context != nullptr &&
                !amd_hip_check(hipStreamSynchronize(context->stream),
                    "hipStreamSynchronize before buffer operation")) {
                return false;
            }
        }
        return true;
    }

private:
    amd_device_context * device_ = nullptr;
    std::unique_lock<std::mutex> registry_lock_;
    std::vector<amd_backend_context *> contexts_;
    std::vector<std::unique_lock<std::mutex>> locks_;
};

static bool amd_auto_tuning_enabled(const amd_device_context * device) {
    if (device == nullptr || !device->tuned_package) {
        return false;
    }
    const char * configured = std::getenv("FLAGOS_AMD_TUNING_PROFILE");
    return configured == nullptr ||
        (configured[0] != '\0' && std::strcmp(configured, "0") != 0 && std::strcmp(configured, "off") != 0);
}

static uint64_t amd_hash_device_identity(
        const amd_device_context & device, uint64_t seed) {
    uint64_t hash = seed;
    const auto add = [&hash](const void * data, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(data);
        for (size_t index = 0; index < size; ++index) {
            hash ^= bytes[index];
            hash *= 1099511628211ULL;
        }
    };
    static constexpr char provider_namespace[] = "flagos:amd:";
    add(provider_namespace, sizeof(provider_namespace) - 1);

    bool has_uuid = false;
    for (const uint8_t byte : device.uuid) {
        has_uuid = has_uuid || byte != 0;
    }
    if (has_uuid) {
        add(device.uuid.data(), device.uuid.size());
    } else if (!device.device_id.empty()) {
        // hipGetDevicePCIBusId is stable across process restarts and does not
        // depend on HIP's enumeration order.
        add(device.device_id.data(), device.device_id.size());
    } else {
        // The ordinal is a last-resort process-local discriminator only.
        const uint64_t ordinal = static_cast<uint64_t>(device.ordinal + 1);
        add(&ordinal, sizeof(ordinal));
    }
    return hash == 0 ? seed | 1ULL : hash;
}

static uint64_t amd_memory_domain_id(const amd_device_context & device) {
    return amd_hash_device_identity(device, 1469598103934665603ULL);
}

static void amd_fill_device_profile(
        const amd_device_context & device, flagos_device_profile * profile) {
    if (profile == nullptr) {
        return;
    }
    *profile = {};
    profile->struct_size = sizeof(flagos_device_profile);
    profile->version = 1;
    profile->engine = flagos_engine_kind::gpu;
    profile->features = FLAGOS_FEATURE_SIMD;
    profile->lane_count = static_cast<uint32_t>(std::max(device.props.warpSize, 0));
    if (profile->lane_count == 32) {
        profile->features |= FLAGOS_FEATURE_WAVE32;
    } else if (profile->lane_count == 64) {
        profile->features |= FLAGOS_FEATURE_WAVE64;
    }
    if (device.props.integrated) {
        profile->features |= FLAGOS_FEATURE_UNIFIED_MEMORY;
    }
    profile->concurrency = static_cast<uint32_t>(std::max(device.props.multiProcessorCount, 0));
    profile->memory_domain_id = amd_memory_domain_id(device);
    profile->vendor = "AMD";
    profile->architecture = "GCN-compatible GPU";
    profile->target = device.arch.c_str();
    profile->runtime = "HIP";
    profile->aot_format = device.aot != nullptr ? "hsaco" : nullptr;
}

static void amd_clear_graph_captures(amd_backend_context * context) {
    if (context == nullptr) {
        return;
    }
    if (context->device != nullptr) {
        amd_hip_check(hipSetDevice(context->device->ordinal), "hipSetDevice");
    }
    for (auto & entry : context->graph_captures) {
        if (entry.executable != nullptr) {
            amd_hip_check(hipGraphExecDestroy(entry.executable), "hipGraphExecDestroy");
            entry.executable = nullptr;
        }
    }
    context->graph_captures.clear();
    context->graph_capture_candidates.clear();
}

static bool amd_profile_switch(const amd_device_context * device, const char * name) {
    return std::getenv(name) != nullptr ? amd_env_enabled(name) : amd_auto_tuning_enabled(device);
}

static bool amd_prefill_f16_gemm_enabled(const amd_device_context * device) {
    return amd_profile_switch(device, "FLAGOS_AMD_PREFILL_F16_GEMM");
}

static bool amd_ffn_down_f16_enabled(const amd_device_context * device) {
    return amd_profile_switch(device, "FLAGOS_AMD_FFN_DOWN_F16");
}

static const flagos_kernel_variant AMD_TUNED_PACKAGE_VARIANT = {
    flagos_amd::tuning_profile_gfx1150_q4ffn_v1,
    { FLAGOS_FEATURE_WAVE32, 0, 0, 32, "gfx1150", flagos_engine_bit(flagos_engine_kind::gpu) },
    0, 0, 0, 0, 0, 0, 0, flagos_support_state::tuned,
};

static const flagos_kernel_variant AMD_GROUPED_F16_GEMM_VARIANT = {
    "f16-grouped",
    { FLAGOS_FEATURE_WAVE32, 0, 0, 32, "gfx1150", flagos_engine_bit(flagos_engine_kind::gpu) },
    // The minimum N is read from the loaded kernel metadata below.  Keeping
    // it out of this provider capability record lets a package tune a
    // different tile without changing the common variant contract.
    0, 0, 0, 0, 0, 0, 0, flagos_support_state::validated,
};

static const flagos_kernel_variant AMD_GROUPED_F16_FFN_VARIANT = {
    "ffn-swiglu-f16-grouped",
    { FLAGOS_FEATURE_WAVE32, 0, 0, 32, "gfx1150", flagos_engine_bit(flagos_engine_kind::gpu) },
    0, 0, 32, 0, 0, 0, 0, flagos_support_state::validated,
};

static bool amd_is_tuned_package(const amd_device_context * device) {
    if (device == nullptr || device->aot == nullptr) {
        return false;
    }
    const std::string & profile = device->aot->tuning_profile();
    const flagos_kernel_shape shape {};
    return (profile == AMD_TUNED_PACKAGE_VARIANT.name ||
            profile == flagos_amd::tuning_profile_gfx1150_qwen35_q4km_v2) &&
        flagos_kernel_variant_matches(&device->profile, &AMD_TUNED_PACKAGE_VARIANT, &shape);
}

static bool amd_grouped_f16_gemm_enabled(const amd_device_context * device, int columns) {
    if (device == nullptr || columns <= 0 || device->aot == nullptr) {
        return false;
    }
    const char * configured = std::getenv("FLAGOS_AMD_GROUPED_F16_GEMM");
    if (configured != nullptr && (configured[0] == '\0' || std::strcmp(configured, "0") == 0)) {
        return false;
    }
    // The grouped schedule is a one-dimensional tile mapping.  Its minimum
    // useful batch is the compiled N tile, not a global model-size threshold;
    // using the manifest keeps this selection valid for smaller transformers
    // such as MiniCPM5 while preserving the provider-owned kernel contract.
    const auto * metadata = device->aot->find("flagos_mul_mat_f16_f32_grouped");
    const int minimum_columns = metadata != nullptr && metadata->tile_n > 0
        ? metadata->tile_n : 256;
    if (columns < minimum_columns) {
        return false;
    }
    const flagos_kernel_shape shape { 0, columns, 0 };
    // An explicit enable only opts into a validated provider-local variant; it
    // never bypasses the common target/feature matcher.
    return flagos_kernel_variant_matches(
        &device->profile, &AMD_GROUPED_F16_GEMM_VARIANT, &shape);
}

static bool amd_grouped_f16_ffn_enabled(const amd_device_context * device, int columns) {
    if (device == nullptr || columns < 32) {
        return false;
    }
    const char * configured = std::getenv("FLAGOS_AMD_GROUPED_F16_GEMM");
    if (configured != nullptr && (configured[0] == '\0' || std::strcmp(configured, "0") == 0)) {
        return false;
    }
    const flagos_kernel_shape shape { 0, columns, 0 };
    // Keep the override fail-closed on unsupported AMD architectures.
    return flagos_kernel_variant_matches(
        &device->profile, &AMD_GROUPED_F16_FFN_VARIANT, &shape);
}

static unsigned int amd_q4_gemv_narrow_row_tile(const amd_device_context * device) {
    const char * value = std::getenv("FLAGOS_AMD_Q4_GEMV_NARROW");
    if (value == nullptr) {
        return amd_auto_tuning_enabled(device) ? 8U : 1U;
    }
    if (std::strcmp(value, "8") == 0) {
        return 8U;
    }
    return amd_env_enabled("FLAGOS_AMD_Q4_GEMV_NARROW") ? 4U : 1U;
}

static unsigned int amd_q40_gemv_narrow_row_tile(const amd_device_context * device) {
    if (!amd_env_enabled("FLAGOS_AMD_Q40_GEMV_NARROW") ||
        device == nullptr || device->aot == nullptr) {
        return 1U;
    }
    const auto * metadata = device->aot->find("flagos_mul_mat_q4_0_f32_narrow");
    if (metadata == nullptr || metadata->block_size <= 1 || metadata->block_size > 32 ||
        (metadata->block_size & (metadata->block_size - 1)) != 0) {
        return 1U;
    }
    return static_cast<unsigned int>(metadata->block_size);
}

static unsigned int amd_q5_gemv_narrow_row_tile(const amd_device_context * device) {
    const char * configured = std::getenv("FLAGOS_AMD_Q5_GEMV_NARROW");
    const bool enabled = configured != nullptr
        ? amd_env_enabled("FLAGOS_AMD_Q5_GEMV_NARROW")
        : amd_auto_tuning_enabled(device) && amd_qwen35_v2_profile(device);
    if (!enabled || device == nullptr || device->aot == nullptr) {
        return 1U;
    }
    const auto * metadata = device->aot->find("flagos_mul_mat_q5_k_f32_narrow16");
    if (metadata == nullptr || metadata->block_size != 16) {
        return 1U;
    }
    return 16U;
}

static bool amd_quant_tiled_enabled() {
    return amd_env_enabled("FLAGOS_AMD_QUANT_TILED_GEMM");
}

static int64_t amd_quantized_block_size(flagos_quantized_matmul_kind kind) {
    switch (kind) {
        case flagos_quantized_matmul_kind::q4_0: return 32;
        case flagos_quantized_matmul_kind::q4_1:
        case flagos_quantized_matmul_kind::q8_0: return 32;
        case flagos_quantized_matmul_kind::q5_k:
        case flagos_quantized_matmul_kind::q4_k:
        case flagos_quantized_matmul_kind::q6_k: return 256;
        case flagos_quantized_matmul_kind::none: return 0;
    }
    return 0;
}

static flagos_quantized_matmul_kind amd_quantized_kind(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return flagos_quantized_matmul_kind::q4_0;
        case GGML_TYPE_Q4_1: return flagos_quantized_matmul_kind::q4_1;
        case GGML_TYPE_Q5_K: return flagos_quantized_matmul_kind::q5_k;
        case GGML_TYPE_Q4_K: return flagos_quantized_matmul_kind::q4_k;
        case GGML_TYPE_Q6_K: return flagos_quantized_matmul_kind::q6_k;
        case GGML_TYPE_Q8_0: return flagos_quantized_matmul_kind::q8_0;
        default: return flagos_quantized_matmul_kind::none;
    }
}

static const char * amd_quantized_dequant_kernel(flagos_quantized_matmul_kind kind) {
    switch (kind) {
        case flagos_quantized_matmul_kind::q4_0: return "flagos_dequant_q4_0_f16";
        case flagos_quantized_matmul_kind::q4_1: return "flagos_dequant_q4_1_f16";
        case flagos_quantized_matmul_kind::q5_k: return "flagos_dequant_q5_k_f16";
        case flagos_quantized_matmul_kind::q4_k: return "flagos_dequant_q4_k_f16";
        case flagos_quantized_matmul_kind::q6_k: return "flagos_dequant_q6_k_f16";
        case flagos_quantized_matmul_kind::q8_0: return "flagos_dequant_q8_0_f16";
        case flagos_quantized_matmul_kind::none: return nullptr;
    }
    return nullptr;
}

static const char * amd_quantized_gemv_kernel(flagos_quantized_matmul_kind kind) {
    switch (kind) {
        case flagos_quantized_matmul_kind::q4_0: return "flagos_mul_mat_q4_0_f32";
        case flagos_quantized_matmul_kind::q4_1: return "flagos_mul_mat_q4_1_f32";
        case flagos_quantized_matmul_kind::q5_k: return "flagos_mul_mat_q5_k_f32";
        case flagos_quantized_matmul_kind::q4_k: return "flagos_mul_mat_q4_k_f32";
        case flagos_quantized_matmul_kind::q6_k: return "flagos_mul_mat_q6_k_f32";
        case flagos_quantized_matmul_kind::q8_0: return "flagos_mul_mat_q8_0_f32";
        case flagos_quantized_matmul_kind::none: return nullptr;
    }
    return nullptr;
}

static const char * amd_quantized_get_rows_kernel(flagos_quantized_matmul_kind kind) {
    switch (kind) {
        case flagos_quantized_matmul_kind::q4_0: return "flagos_get_rows_q4_0_f32";
        case flagos_quantized_matmul_kind::q4_1: return "flagos_get_rows_q4_1_f32";
        case flagos_quantized_matmul_kind::q5_k: return "flagos_get_rows_q5_k_f32";
        case flagos_quantized_matmul_kind::q4_k: return "flagos_get_rows_q4_k_f32";
        case flagos_quantized_matmul_kind::q6_k: return "flagos_get_rows_q6_k_f32";
        case flagos_quantized_matmul_kind::q8_0: return "flagos_get_rows_q8_0_f32";
        case flagos_quantized_matmul_kind::none: return nullptr;
    }
    return nullptr;
}

static const char * amd_quantized_batched_kernel(const amd_device_context * device,
                                                 flagos_quantized_matmul_kind kind) {
    if (kind == flagos_quantized_matmul_kind::q4_0) {
        return "flagos_mul_mat_q4_0_f32_batched";
    }
    if (kind == flagos_quantized_matmul_kind::q4_1) {
        return "flagos_mul_mat_q4_1_f32_batched";
    }
    if (kind == flagos_quantized_matmul_kind::q8_0) {
        return "flagos_mul_mat_q8_0_f32_batched";
    }
    if (kind != flagos_quantized_matmul_kind::q5_k &&
        kind != flagos_quantized_matmul_kind::q4_k &&
        kind != flagos_quantized_matmul_kind::q6_k) {
        return nullptr;
    }
    const bool q4 = kind == flagos_quantized_matmul_kind::q4_k;
    if (amd_quant_tiled_enabled()) {
        const char * tiled = q4 ? "flagos_mul_mat_q4_k_f32_tiled" :
            kind == flagos_quantized_matmul_kind::q5_k ? nullptr : "flagos_mul_mat_q6_k_f32_tiled";
        if (tiled != nullptr && device != nullptr && device->aot != nullptr &&
            device->aot->find(tiled) != nullptr) {
            return tiled;
        }
    }
    return q4 ? "flagos_mul_mat_q4_k_f32_batched" :
        kind == flagos_quantized_matmul_kind::q5_k ? "flagos_mul_mat_q5_k_f32_batched" :
        "flagos_mul_mat_q6_k_f32_batched";
}

static void amd_release_dequant_cache(amd_backend_context * context) {
    if (context == nullptr || context->device == nullptr) {
        return;
    }
    amd_hip_check(hipSetDevice(context->device->ordinal), "hipSetDevice");
    context->dequant_cache.clear();
    for (auto & block : context->dequant_arena) {
        if (block.data != nullptr) {
            amd_hip_check(hipFree(block.data), "hipFree dequant arena");
            block.data = nullptr;
        }
    }
    context->dequant_arena.clear();
    context->dequant_cache_bytes = 0;
    context->dequant_arena_bytes = 0;
    ++context->dequant_cache_epoch;
    context->dequant_cache_limit_logged = false;
}

static bool amd_revalidate_dequant_cache(amd_backend_context * context) {
    if (context == nullptr) {
        return false;
    }
    bool expired = false;
    bool newly_stale = false;
    for (auto & entry : context->dequant_cache) {
        if (entry.source_lifetime.expired()) {
            expired = true;
            continue;
        }
        // The live lifetime token makes source_buffer safe to inspect.  A
        // coarse allocation generation is intentional: any write into a model
        // weight buffer invalidates every private dequantized view sourced from
        // that allocation.
        if (entry.valid &&
            (entry.source_buffer == nullptr ||
             entry.source_buffer->allocation_id != entry.source_allocation_id ||
             entry.source_buffer->generation.load(std::memory_order_acquire) !=
                entry.source_generation)) {
            entry.valid = false;
            newly_stale = true;
        }
    }
    if (!expired && !newly_stale) {
        return true;
    }
    // Finish prior users before destroying executables that may embed private
    // F16 pointers.  This check runs before graph replay lookup, so a buffer
    // generation change cannot bypass normal cache refresh through an old HIP
    // Graph executable.
    if (!amd_hip_check(hipStreamSynchronize(context->stream),
            "hipStreamSynchronize before dequant cache invalidation")) {
        return false;
    }
    amd_clear_graph_captures(context);
    if (expired) {
        // A model buffer was released while this backend survived (for example
        // a server unloading one model before loading another).  Arena
        // allocations are monotonic, so individual dead entries cannot be
        // reclaimed safely; reset the arena as one transaction.
        amd_release_dequant_cache(context);
    } else {
        // Keep the private allocations for an in-place weight refresh, but make
        // every old capture key obsolete even before the affected weight is
        // encountered and re-dequantized.
        ++context->dequant_cache_epoch;
    }
    return true;
}

static bool amd_bindings_reference_buffer(
        const flagos_graph_binding_snapshot & bindings, const void * data, size_t size) {
    if (data == nullptr || size == 0) {
        return false;
    }
    const uintptr_t begin = reinterpret_cast<uintptr_t>(data);
    if (begin > UINTPTR_MAX - size) {
        return !bindings.pointers.empty();
    }
    const uintptr_t end = begin + size;
    return std::any_of(bindings.pointers.begin(), bindings.pointers.end(),
        [begin, end](uintptr_t pointer) {
            return pointer >= begin && pointer < end;
        });
}

static void amd_release_buffer_storage(amd_buffer_context * buffer) {
    if (buffer == nullptr || buffer->device == nullptr || buffer->allocation_id == 0) {
        return;
    }
    amd_device_execution_guard guard(buffer->device);
    if (!guard.synchronize()) {
        GGML_LOG_ERROR("FlagOS AMD: cannot prove buffer is idle before release; leaking allocation %llu\n",
            static_cast<unsigned long long>(buffer->allocation_id));
        return;
    }
    for (amd_backend_context * context : guard.contexts()) {
        if (context == nullptr) {
            continue;
        }
        const bool cache_reference = std::any_of(
            context->dequant_cache.begin(), context->dequant_cache.end(),
            [buffer](const amd_dequant_cache_entry & entry) {
                return entry.source_allocation_id == buffer->allocation_id;
            });
        const bool capture_reference = std::any_of(
            context->graph_captures.begin(), context->graph_captures.end(),
            [buffer](const amd_graph_capture_entry & entry) {
                return amd_bindings_reference_buffer(entry.bindings, buffer->data, buffer->size);
            });
        const bool candidate_reference = std::any_of(
            context->graph_capture_candidates.begin(), context->graph_capture_candidates.end(),
            [buffer](const amd_graph_capture_candidate & candidate) {
                return amd_bindings_reference_buffer(candidate.bindings, buffer->data, buffer->size);
            });
        // The guard has already made every backend stream idle. Captured
        // executables bind raw GGML addresses, so remove any reference before
        // the allocator can reuse this range.
        if (cache_reference || capture_reference) {
            amd_clear_graph_captures(context);
        } else if (candidate_reference) {
            context->graph_capture_candidates.clear();
        }
        if (cache_reference) {
            amd_release_dequant_cache(context);
        }
    }
    if (buffer->data != nullptr &&
        amd_hip_check(hipFree(buffer->data), "hipFree")) {
        buffer->data = nullptr;
    }
}

static void * amd_allocate_dequant_cache(
        amd_backend_context * context, size_t bytes,
        size_t * allocation_index, size_t * previous_used) {
    if (context == nullptr || allocation_index == nullptr || previous_used == nullptr ||
        bytes == 0 || bytes > SIZE_MAX - (AMD_DEQUANT_ARENA_ALIGNMENT - 1)) {
        return nullptr;
    }
    const size_t aligned_bytes = (bytes + AMD_DEQUANT_ARENA_ALIGNMENT - 1) &
        ~(AMD_DEQUANT_ARENA_ALIGNMENT - 1);
    for (size_t index = 0; index < context->dequant_arena.size(); ++index) {
        auto & block = context->dequant_arena[index];
        if (block.used <= block.capacity && aligned_bytes <= block.capacity - block.used) {
            *allocation_index = index;
            *previous_used = block.used;
            void * result = static_cast<uint8_t *>(block.data) + block.used;
            block.used += aligned_bytes;
            return result;
        }
    }
    size_t chunk_bytes = AMD_DEQUANT_ARENA_CHUNK_BYTES;
    if (context->dequant_cache_limit_bytes != 0) {
        if (context->dequant_cache_bytes >= context->dequant_cache_limit_bytes) {
            return nullptr;
        }
        const size_t remaining = context->dequant_cache_limit_bytes - context->dequant_cache_bytes;
        if (aligned_bytes > remaining) {
            return nullptr;
        }
        chunk_bytes = std::min(chunk_bytes, remaining);
    }
    const size_t capacity = std::max(aligned_bytes, chunk_bytes);
    if (context->dequant_arena_bytes > SIZE_MAX - capacity) {
        return nullptr;
    }
    void * data = nullptr;
    if (!amd_hip_check(hipMalloc(&data, capacity), "hipMalloc dequant arena")) {
        return nullptr;
    }
    *allocation_index = context->dequant_arena.size();
    *previous_used = 0;
    context->dequant_arena.push_back({ data, capacity, aligned_bytes });
    context->dequant_arena_bytes += capacity;
    return data;
}

static void amd_rollback_dequant_cache_allocation(
        amd_backend_context * context, size_t allocation_index, size_t previous_used) {
    if (context == nullptr || allocation_index >= context->dequant_arena.size()) {
        return;
    }
    auto & block = context->dequant_arena[allocation_index];
    if (previous_used > block.used) {
        return;
    }
    block.used = previous_used;
    if (previous_used == 0 && allocation_index + 1 == context->dequant_arena.size()) {
        amd_hip_check(hipFree(block.data), "hipFree unused dequant arena");
        context->dequant_arena_bytes -= block.capacity;
        context->dequant_arena.pop_back();
    }
}

static bool amd_dequant_cache_contains(
        const amd_backend_context * context, const ggml_tensor * weight,
        const flagos_quantized_matmul_signature & signature) {
    if (context == nullptr || weight == nullptr) {
        return false;
    }
    const ggml_backend_buffer_t storage = amd_tensor_buffer(weight);
    const auto * source_buffer = amd_buffer_from_buffer(storage);
    if (!amd_buffer_is_local_storage(storage) || source_buffer == nullptr ||
        ggml_backend_buffer_get_usage(storage) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        (weight->flags & GGML_TENSOR_FLAG_PARAM) != 0) {
        return false;
    }
    const uint64_t generation = source_buffer->generation.load(std::memory_order_acquire);
    for (const auto & entry : context->dequant_cache) {
        if (entry.valid && entry.tensor == weight && entry.source == weight->data &&
            entry.source_buffer == source_buffer &&
            entry.source_allocation_id == source_buffer->allocation_id &&
            entry.source_generation == generation &&
            entry.kind == signature.weight_kind && entry.k == signature.k &&
            entry.rows == signature.rows) {
            return true;
        }
    }
    return false;
}

static bool amd_dequant_cache_entry_bytes(
        const flagos_quantized_matmul_signature & signature, size_t * bytes) {
    const int64_t block_size = amd_quantized_block_size(signature.weight_kind);
    if (bytes == nullptr || signature.k <= 0 || signature.rows <= 0 ||
        block_size == 0 || signature.k % block_size != 0) {
        return false;
    }
    const uint64_t elements = static_cast<uint64_t>(signature.k) *
        static_cast<uint64_t>(signature.rows);
    if (elements > SIZE_MAX / sizeof(ggml_fp16_t)) {
        return false;
    }
    *bytes = static_cast<size_t>(elements) * sizeof(ggml_fp16_t);
    return true;
}

static bool amd_dequant_cache_ready(
        const amd_backend_context * context,
        const ggml_tensor * const * projections,
        const flagos_quantized_matmul_signature * signatures,
        size_t projection_count) {
    if (context == nullptr || projections == nullptr || signatures == nullptr ||
        projection_count == 0 || projection_count > 3) {
        return false;
    }
    for (size_t index = 0; index < projection_count; ++index) {
        if (projections[index] == nullptr || projections[index]->src[0] == nullptr ||
            !amd_dequant_cache_contains(
                context, projections[index]->src[0], signatures[index])) {
            return false;
        }
    }
    return true;
}

static bool amd_f16_graph_cache_ready(
        const amd_backend_context * context, const ggml_cgraph * cgraph) {
    if (context == nullptr || cgraph == nullptr || !amd_prefill_f16_gemm_enabled(context->device)) {
        return true;
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr || node->op != GGML_OP_MUL_MAT ||
            node->src[0] == nullptr || node->src[1] == nullptr ||
            node->src[1]->ne[1] <= 1 || node->src[0]->ne[1] < 64 ||
            node->src[0]->ne[0] < 1024) {
            continue;
        }
        flagos_quantized_matmul_signature signature;
        if (!flagos_describe_quantized_matmul(node, &signature) ||
            !amd_dequant_cache_contains(context, node->src[0], signature)) {
            return false;
        }
    }
    return true;
}

static void * amd_get_dequantized_weight(
        amd_backend_context * context, const ggml_tensor * weight,
        const flagos_quantized_matmul_signature & signature) {
    const int64_t block_size = amd_quantized_block_size(signature.weight_kind);
    if (context == nullptr || context->device == nullptr || context->device->aot == nullptr ||
        weight == nullptr || weight->data == nullptr || signature.k <= 0 || signature.rows <= 0 ||
        block_size == 0 || signature.k % block_size != 0 || !ggml_is_contiguous(weight)) {
        return nullptr;
    }
    const ggml_backend_buffer_t storage = amd_tensor_buffer(weight);
    const auto * source_buffer = amd_buffer_from_buffer(storage);
    if (!amd_buffer_is_local_storage(storage) || source_buffer == nullptr ||
        ggml_backend_buffer_get_usage(storage) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        (weight->flags & GGML_TENSOR_FLAG_PARAM) != 0) {
        return nullptr;
    }
    const uint64_t source_generation = source_buffer->generation.load(std::memory_order_acquire);
    amd_dequant_cache_entry * reusable = nullptr;
    for (auto & entry : context->dequant_cache) {
        if (entry.valid && entry.tensor == weight && entry.source == weight->data &&
            entry.source_buffer == source_buffer &&
            entry.source_allocation_id == source_buffer->allocation_id &&
            entry.source_generation == source_generation &&
            entry.kind == signature.weight_kind &&
            entry.k == signature.k && entry.rows == signature.rows) {
            return entry.f16_data;
        }
        // A normal in-place model-weight update changes the buffer generation
        // but not the dequantized shape.  Reuse its private F16 allocation
        // instead of leaking dead entries against the bounded cache limit.
        // allocation_id closes the allocator-ABA case where a freed buffer
        // context is later recreated at the same host address.
        if (entry.tensor == weight && entry.source_buffer == source_buffer &&
            entry.source_allocation_id == source_buffer->allocation_id &&
            entry.kind == signature.weight_kind && entry.k == signature.k &&
            entry.rows == signature.rows) {
            reusable = &entry;
        }
    }
    const char * kernel_name = amd_quantized_dequant_kernel(signature.weight_kind);
    if (kernel_name == nullptr || context->device->aot->find(kernel_name) == nullptr) {
        return nullptr;
    }
    size_t bytes = 0;
    if (!amd_dequant_cache_entry_bytes(signature, &bytes) ||
        (reusable != nullptr && reusable->bytes != bytes)) {
        return nullptr;
    }
    const uint64_t elements = static_cast<uint64_t>(signature.k) *
        static_cast<uint64_t>(signature.rows);
    if (reusable == nullptr && context->dequant_cache_limit_bytes != 0 &&
        (bytes > context->dequant_cache_limit_bytes ||
         context->dequant_cache_bytes > context->dequant_cache_limit_bytes - bytes)) {
        if (!context->dequant_cache_limit_logged) {
            GGML_LOG_WARN("FlagOS AMD: F16 dequant cache limit reached (%zu/%zu MiB); "
                "falling back to quantized GEMM for subsequent weights\n",
                context->dequant_cache_bytes / (1024 * 1024),
                context->dequant_cache_limit_bytes / (1024 * 1024));
            context->dequant_cache_limit_logged = true;
        }
        return nullptr;
    }
    amd_hip_check(hipSetDevice(context->device->ordinal), "hipSetDevice");
    size_t allocation_index = 0;
    size_t previous_used = 0;
    void * f16_data = reusable != nullptr ? reusable->f16_data :
        amd_allocate_dequant_cache(context, bytes, &allocation_index, &previous_used);
    if (f16_data == nullptr) {
        return nullptr;
    }
    void * weights_u8 = weight->data;
    void * weights_f16 = weight->data;
    int blocks = static_cast<int>(elements / static_cast<uint64_t>(block_size));
    flagos_amd::kernel_arguments arguments = { &weights_u8, &weights_f16, &f16_data };
    if (!context->device->aot->launch(kernel_name, context->stream,
            static_cast<unsigned int>(blocks), 1, 1, arguments)) {
        if (reusable == nullptr) {
            amd_rollback_dequant_cache_allocation(context, allocation_index, previous_used);
        }
        return nullptr;
    }
    if (reusable != nullptr) {
        reusable->source = weight->data;
        reusable->source_generation = source_generation;
        reusable->valid = true;
    } else {
        context->dequant_cache.push_back({ weight, weight->data, source_buffer,
            source_buffer->allocation_id, source_buffer->lifetime, source_generation, signature.weight_kind,
            static_cast<int>(signature.k), static_cast<int>(signature.rows), f16_data, bytes });
        context->dequant_cache_bytes += bytes;
    }
    ++context->dequant_cache_epoch;
    context->stats.weight_dequantizations.fetch_add(1, std::memory_order_relaxed);
    return f16_data;
}

static void amd_prepare_dequant_cache(
        amd_backend_context * context, const ggml_cgraph * cgraph) {
    if (context == nullptr || cgraph == nullptr ||
        !amd_prefill_f16_gemm_enabled(context->device)) {
        return;
    }
    struct request {
        const ggml_tensor * weight;
        flagos_quantized_matmul_signature signature;
        size_t bytes;
    };
    std::vector<request> requests;
    requests.reserve(static_cast<size_t>(cgraph->n_nodes));
    for (int index = 0; index < cgraph->n_nodes; ++index) {
        const ggml_tensor * node = cgraph->nodes[index];
        if (node == nullptr || node->op != GGML_OP_MUL_MAT ||
            node->src[0] == nullptr || node->src[1] == nullptr ||
            node->src[1]->ne[1] <= 1 || node->src[0]->ne[1] < 64 ||
            node->src[0]->ne[0] < 1024) {
            continue;
        }
        flagos_quantized_matmul_signature signature;
        if (!flagos_describe_quantized_matmul(node, &signature) ||
            amd_dequant_cache_contains(context, node->src[0], signature)) {
            continue;
        }
        const bool duplicate = std::any_of(requests.begin(), requests.end(),
            [node, &signature](const request & item) {
                return item.weight == node->src[0] &&
                    item.signature.weight_kind == signature.weight_kind &&
                    item.signature.k == signature.k && item.signature.rows == signature.rows;
            });
        size_t bytes = 0;
        if (!duplicate && amd_dequant_cache_entry_bytes(signature, &bytes)) {
            requests.push_back({ node->src[0], signature, bytes });
        }
    }
    std::stable_sort(requests.begin(), requests.end(),
        [](const request & left, const request & right) {
            return left.bytes > right.bytes;
        });
    for (const auto & item : requests) {
        amd_get_dequantized_weight(context, item.weight, item.signature);
    }
}

static std::mutex g_mutex;
static bool g_probed = false;
static std::vector<amd_device_context> g_devices;
static std::atomic<uint64_t> g_buffer_allocation_id { 1 };

static amd_device_context * amd_device_from_dev(ggml_backend_dev_t dev) {
    return dev == nullptr ? nullptr : static_cast<amd_device_context *>(dev->context);
}

static amd_device_context * amd_device_from_buft(ggml_backend_buffer_type_t buft) {
    return buft == nullptr ? nullptr : static_cast<amd_device_context *>(buft->context);
}

static bool amd_buft_is_local(ggml_backend_buffer_type_t buft, const amd_device_context * device) {
    return amd_device_from_buft(buft) == device;
}

static bool amd_tensor_is_view_op(const ggml_tensor * op) {
    if (op == nullptr) {
        return false;
    }
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

static const char * amd_buffer_type_name(ggml_backend_buffer_type_t) {
    return "FlagOS_AMD";
}

static ggml_backend_buffer_t amd_buffer_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * device = amd_device_from_buft(buft);
    if (device == nullptr || !amd_hip_check(hipSetDevice(device->ordinal), "hipSetDevice")) {
        return nullptr;
    }

    void * data = nullptr;
    if (size != 0 && !amd_hip_check(hipMalloc(&data, size), "hipMalloc")) {
        return nullptr;
    }

    uint64_t allocation_id = g_buffer_allocation_id.fetch_add(1, std::memory_order_relaxed);
    if (allocation_id == 0) {
        allocation_id = g_buffer_allocation_id.fetch_add(1, std::memory_order_relaxed);
    }
    auto * context = new amd_buffer_context(device, data, size, allocation_id);
    static const ggml_backend_buffer_i iface = {
        /* .free_buffer    = */ [](ggml_backend_buffer_t buffer) {
            auto * context = amd_buffer_from_buffer(buffer);
            if (context != nullptr) {
                amd_release_buffer_storage(context);
                delete context;
            }
        },
        /* .get_base       = */ [](ggml_backend_buffer_t buffer) -> void * {
            auto * context = amd_buffer_from_buffer(buffer);
            return context == nullptr ? nullptr : context->data;
        },
        /* .init_tensor    = */ nullptr,
        /* .memset_tensor  = */ [](ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
            auto * context = amd_buffer_from_buffer(buffer);
            GGML_ASSERT(context != nullptr && tensor != nullptr && offset <= ggml_nbytes(tensor) &&
                size <= ggml_nbytes(tensor) - offset && (size == 0 || tensor->data != nullptr));
            if (size == 0) {
                return;
            }
            amd_device_execution_guard guard(context->device);
            GGML_ASSERT(guard.synchronize());
            amd_hip_check(hipSetDevice(context->device->ordinal), "hipSetDevice");
            GGML_ASSERT(amd_hip_check(hipMemset(static_cast<char *>(tensor->data) + offset, value, size), "hipMemset"));
            // Backend streams are created with hipStreamNonBlocking, so a
            // default-stream memset is not ordered before later provider
            // launches. Make this synchronous buffer callback complete the
            // clear before releasing the device execution guard.
            GGML_ASSERT(amd_hip_check(
                hipDeviceSynchronize(), "hipDeviceSynchronize after hipMemset"));
            amd_mark_buffer_modified(context);
        },
        /* .set_tensor     = */ [](ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
            auto * context = amd_buffer_from_buffer(buffer);
            GGML_ASSERT(context != nullptr && tensor != nullptr && offset <= ggml_nbytes(tensor) &&
                size <= ggml_nbytes(tensor) - offset && (size == 0 || (tensor->data != nullptr && data != nullptr)));
            if (size == 0) {
                return;
            }
            amd_device_execution_guard guard(context->device);
            GGML_ASSERT(guard.synchronize());
            amd_hip_check(hipSetDevice(context->device->ordinal), "hipSetDevice");
            GGML_ASSERT(amd_hip_check(hipMemcpy(
                static_cast<char *>(tensor->data) + offset, data, size, hipMemcpyHostToDevice), "hipMemcpy H2D"));
            amd_mark_buffer_modified(context);
        },
        /* .get_tensor     = */ [](ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
            auto * context = amd_buffer_from_buffer(buffer);
            GGML_ASSERT(context != nullptr && tensor != nullptr && offset <= ggml_nbytes(tensor) &&
                size <= ggml_nbytes(tensor) - offset && (size == 0 || (tensor->data != nullptr && data != nullptr)));
            if (size == 0) {
                return;
            }
            amd_device_execution_guard guard(context->device);
            GGML_ASSERT(guard.synchronize());
            amd_hip_check(hipSetDevice(context->device->ordinal), "hipSetDevice");
            GGML_ASSERT(amd_hip_check(hipMemcpy(
                data, static_cast<const char *>(tensor->data) + offset, size, hipMemcpyDeviceToHost), "hipMemcpy D2H"));
        },
        /* .set_tensor_2d  = */ nullptr,
        /* .get_tensor_2d  = */ nullptr,
        /* .cpy_tensor     = */ [](ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
            auto * dst_context = amd_buffer_from_buffer(buffer);
            const ggml_backend_buffer_t src_storage = amd_tensor_buffer(src);
            if (dst_context == nullptr || src == nullptr || dst == nullptr ||
                !amd_buffer_is_local_storage(src_storage) ||
                ggml_nbytes(src) != ggml_nbytes(dst)) {
                return false;
            }
            const auto * src_context = amd_buffer_from_buffer(src_storage);
            if (src_context == nullptr || dst_context->device != src_context->device) {
                return false;
            }
            const size_t size = ggml_nbytes(src);
            if (size == 0) {
                return true;
            }
            if (src->data == nullptr || dst->data == nullptr) {
                return false;
            }
            amd_device_execution_guard guard(dst_context->device);
            if (!guard.synchronize()) {
                return false;
            }
            amd_hip_check(hipSetDevice(dst_context->device->ordinal), "hipSetDevice");
            const bool copied = amd_hip_check(
                hipMemcpy(dst->data, src->data, size, hipMemcpyDeviceToDevice), "hipMemcpy D2D");
            if (copied) {
                amd_mark_buffer_modified(dst_context);
            }
            return copied;
        },
        /* .clear         = */ [](ggml_backend_buffer_t buffer, uint8_t value) {
            auto * context = amd_buffer_from_buffer(buffer);
            GGML_ASSERT(context != nullptr);
            if (context->size == 0) {
                return;
            }
            GGML_ASSERT(context->data != nullptr);
            amd_device_execution_guard guard(context->device);
            GGML_ASSERT(guard.synchronize());
            amd_hip_check(hipSetDevice(context->device->ordinal), "hipSetDevice");
            GGML_ASSERT(amd_hip_check(hipMemset(context->data, value, context->size), "hipMemset buffer"));
            GGML_ASSERT(amd_hip_check(
                hipDeviceSynchronize(), "hipDeviceSynchronize after buffer clear"));
            amd_mark_buffer_modified(context);
        },
        /* .reset         = */ nullptr,
    };
    return ggml_backend_buffer_init(buft, iface, context, size);
}

static size_t amd_buffer_alignment(ggml_backend_buffer_type_t) {
    return 256;
}

static size_t amd_buffer_max_size(ggml_backend_buffer_type_t) {
    return SIZE_MAX;
}

static const ggml_backend_buffer_type_i g_buffer_type_iface = {
    /* .get_name       = */ amd_buffer_type_name,
    /* .alloc_buffer   = */ amd_buffer_alloc,
    /* .get_alignment  = */ amd_buffer_alignment,
    /* .get_max_size   = */ amd_buffer_max_size,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ [](ggml_backend_buffer_type_t) { return false; },
};

static bool amd_buffer_is_local_storage(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && ggml_backend_buffer_get_type(buffer) != nullptr &&
        ggml_backend_buffer_get_type(buffer)->iface.get_name == g_buffer_type_iface.get_name;
}

static const char * amd_backend_name(ggml_backend_t) {
    return "FlagOS_AMD";
}

static void amd_backend_free(ggml_backend_t backend) {
    auto * context = backend == nullptr ? nullptr : static_cast<amd_backend_context *>(backend->context);
    if (context != nullptr) {
        auto * device = context->device;
        GGML_ASSERT(device != nullptr && device->backends_mutex != nullptr);
        // Buffer release takes these locks in the same order. Keep this backend
        // registered until its stream and private resources are idle so a
        // concurrent buffer free cannot miss queued work during teardown.
        std::unique_lock<std::mutex> backends_lock(*device->backends_mutex);
        std::lock_guard<std::mutex> execution_lock(context->execution_mutex);
        amd_log_stats(context);
        if (device->aot != nullptr) {
            device->aot->log_profile();
        }
        // Cached F16 weights and captured graph executables may still be
        // referenced by queued work.  Finish the backend stream before
        // releasing either resource; hipStreamDestroy alone happens too late
        // because both caches are torn down first.
        const bool idle = amd_hip_check(hipSetDevice(device->ordinal), "hipSetDevice") &&
            amd_hip_check(hipStreamSynchronize(context->stream),
                "hipStreamSynchronize before backend teardown");
        if (!idle) {
            // Keep the retired context registered. Buffer release must still
            // see and synchronize this stream; removing it here would turn a
            // failed teardown into a possible device use-after-free. There is
            // no owner left that can reclaim it, so this is an intentional
            // process-lifetime leak on an already failing device.
            GGML_LOG_ERROR("FlagOS AMD: cannot prove backend stream is idle; retaining its resources safely\n");
            backends_lock.unlock();
            delete backend;
            return;
        }
        amd_clear_graph_captures(context);
        amd_release_dequant_cache(context);
        amd_hip_check(hipStreamDestroy(context->stream), "hipStreamDestroy");
        auto & backends = device->backends;
        backends.erase(std::remove(backends.begin(), backends.end(), context), backends.end());
        backends_lock.unlock();
        delete context;
    }
    delete backend;
}

static void amd_backend_set_tensor_async(
        ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend != nullptr && tensor != nullptr);
    auto * backend_context = static_cast<amd_backend_context *>(backend->context);
    auto * buffer_context = const_cast<amd_buffer_context *>(amd_buffer_from_tensor(tensor));
    GGML_ASSERT(backend_context != nullptr && buffer_context != nullptr &&
        buffer_context->device == backend_context->device &&
        offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset &&
        (size == 0 || (tensor->data != nullptr && data != nullptr)));
    if (size == 0) {
        return;
    }
    std::lock_guard<std::mutex> execution_lock(backend_context->execution_mutex);
    amd_hip_check(hipSetDevice(backend_context->device->ordinal), "hipSetDevice");
    GGML_ASSERT(amd_hip_check(hipMemcpyAsync(
        static_cast<char *>(tensor->data) + offset, data, size,
        hipMemcpyHostToDevice, backend_context->stream), "hipMemcpyAsync H2D"));
    amd_mark_buffer_modified(buffer_context);
}

static void amd_backend_get_tensor_async(
        ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend != nullptr && tensor != nullptr);
    auto * backend_context = static_cast<amd_backend_context *>(backend->context);
    const auto * buffer_context = amd_buffer_from_tensor(tensor);
    GGML_ASSERT(backend_context != nullptr && buffer_context != nullptr &&
        buffer_context->device == backend_context->device &&
        offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset &&
        (size == 0 || (tensor->data != nullptr && data != nullptr)));
    if (size == 0) {
        return;
    }
    std::lock_guard<std::mutex> execution_lock(backend_context->execution_mutex);
    amd_hip_check(hipSetDevice(backend_context->device->ordinal), "hipSetDevice");
    GGML_ASSERT(amd_hip_check(hipMemcpyAsync(
        data, static_cast<const char *>(tensor->data) + offset, size,
        hipMemcpyDeviceToHost, backend_context->stream), "hipMemcpyAsync D2H"));
}

static bool amd_backend_copy_tensor_async(
        ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    if (backend_dst == nullptr || src == nullptr || dst == nullptr) {
        return false;
    }
    auto * dst_context = static_cast<amd_backend_context *>(backend_dst->context);
    const auto * dst_buffer = amd_buffer_from_tensor(dst);
    const size_t bytes = ggml_nbytes(src);
    if (dst_context == nullptr || dst_buffer == nullptr || dst_buffer->device != dst_context->device ||
        bytes != ggml_nbytes(dst) ||
        (bytes != 0 && (src->data == nullptr || dst->data == nullptr))) {
        return false;
    }
    if (bytes == 0) {
        return true;
    }
    if (!amd_hip_check(hipSetDevice(dst_context->device->ordinal), "hipSetDevice")) {
        return false;
    }

    // The scheduler uses this callback for partition boundaries as well as
    // same-device copies.  CPU fallback nodes expose host-backed tensors, so
    // blindly casting their buffer context to amd_buffer_context would either
    // reject a valid transfer or interpret foreign state as a HIP device.
    // Inspect the buffer type first and select the correct HIP direction.
    const ggml_backend_buffer_t src_storage = amd_tensor_buffer(src);
    if (src_storage == nullptr) {
        return false;
    }
    const ggml_backend_buffer_type_t src_buft = ggml_backend_buffer_get_type(src_storage);
    const bool src_is_local = src_buft == &dst_context->device->buffer_type_iface;
    const bool src_is_host = src_buft != nullptr && ggml_backend_buft_is_host(src_buft);
    if (!src_is_local && !src_is_host) {
        return false;
    }
    if (src_is_local) {
        const auto * src_buffer = amd_buffer_from_buffer(src_storage);
        auto * src_context = backend_src == nullptr
            ? nullptr : static_cast<amd_backend_context *>(backend_src->context);
        if (backend_src == nullptr || backend_src->device != &dst_context->device->device_iface ||
            src_context == nullptr || src_context->device != dst_context->device ||
            src_buffer == nullptr || src_buffer->device != dst_context->device) {
            return false;
        }
        const auto copy = [&]() {
            // Separate backend instances have separate HIP streams. Complete
            // the producer before enqueueing on the destination stream.
            if (src_context != dst_context &&
                !amd_hip_check(hipStreamSynchronize(src_context->stream),
                    "hipStreamSynchronize before cross-backend D2D")) {
                return false;
            }
            const bool copied = amd_hip_check(hipMemcpyAsync(
                dst->data, src->data, bytes, hipMemcpyDeviceToDevice, dst_context->stream),
                "hipMemcpyAsync D2D");
            if (copied) {
                amd_mark_buffer_modified(const_cast<amd_buffer_context *>(dst_buffer));
                dst_context->stats.device_to_device_copies.fetch_add(1, std::memory_order_relaxed);
            }
            return copied;
        };
        if (src_context == dst_context) {
            std::lock_guard<std::mutex> execution_lock(dst_context->execution_mutex);
            return copy();
        }
        std::scoped_lock execution_locks(
            src_context->execution_mutex, dst_context->execution_mutex);
        return copy();
    }

    // CPU graph temporaries may be recycled as soon as their split returns.
    // Order the copy after the CPU producer and do not return until pageable
    // source memory has been consumed.
    if (backend_src != nullptr) {
        ggml_backend_synchronize(backend_src);
    }
    std::lock_guard<std::mutex> execution_lock(dst_context->execution_mutex);
    if (!amd_hip_check(hipMemcpyAsync(
            dst->data, src->data, bytes, hipMemcpyHostToDevice, dst_context->stream),
            "hipMemcpyAsync H2D") ||
        !amd_hip_check(hipStreamSynchronize(dst_context->stream),
            "hipStreamSynchronize after pageable H2D")) {
        return false;
    }
    dst_context->stats.host_to_device_copies.fetch_add(1, std::memory_order_relaxed);
    amd_mark_buffer_modified(const_cast<amd_buffer_context *>(dst_buffer));
    return true;
}

static void amd_backend_synchronize(ggml_backend_t backend) {
    auto * context = static_cast<amd_backend_context *>(backend->context);
    std::lock_guard<std::mutex> execution_lock(context->execution_mutex);
    amd_hip_check(hipSetDevice(context->device->ordinal), "hipSetDevice");
    GGML_ASSERT(amd_hip_check(hipStreamSynchronize(context->stream), "hipStreamSynchronize"));
    if (context->profile_dump_sync_interval != 0 && context->device->aot != nullptr &&
        ++context->profile_sync_count % context->profile_dump_sync_interval == 0) {
        GGML_LOG_INFO("FlagOS AMD kernel profile snapshot: synchronize=%llu\n",
            static_cast<unsigned long long>(context->profile_sync_count));
        context->device->aot->log_profile();
        amd_log_stats(context);
    }
}

static void amd_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event);
static void amd_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event);

static bool amd_tensor_is_contiguous_f32(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(tensor);
}

static bool amd_tensor_element_offsets_fit_i32(
        const ggml_tensor * tensor, size_t element_size) {
    if (tensor == nullptr || element_size == 0 ||
        ggml_nbytes(tensor) > static_cast<size_t>(INT32_MAX) * element_size) {
        return false;
    }
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (tensor->nb[dim] % element_size != 0 ||
            tensor->nb[dim] / element_size > INT32_MAX) {
            return false;
        }
    }
    return true;
}

static bool amd_tensor_element_strides_are_aligned(
        const ggml_tensor * tensor, size_t element_size,
        size_t element_alignment, int first_dimension) {
    if (tensor == nullptr || element_size == 0 || element_alignment == 0 ||
        first_dimension < 0 || first_dimension >= GGML_MAX_DIMS) {
        return false;
    }
    for (int dim = first_dimension; dim < GGML_MAX_DIMS; ++dim) {
        if (tensor->nb[dim] % element_size != 0 ||
            (tensor->nb[dim] / element_size) % element_alignment != 0) {
            return false;
        }
    }
    return true;
}

static bool amd_tensor_data_is_aligned(
        const ggml_tensor * tensor, size_t alignment) {
    // supports_op can run before scheduler allocation. Revalidate the same
    // condition during execution, when data is bound, without rejecting an
    // otherwise supported unallocated graph.
    return tensor != nullptr && alignment != 0 &&
        (tensor->data == nullptr ||
         reinterpret_cast<uintptr_t>(tensor->data) % alignment == 0);
}

static bool amd_tensor_bindings_ready(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->data == nullptr ||
        (tensor->view_src != nullptr && tensor->view_src->data == nullptr)) {
        return false;
    }
    for (int source = 0; source < GGML_MAX_SRC; ++source) {
        if (tensor->src[source] != nullptr && tensor->src[source]->data == nullptr) {
            return false;
        }
    }
    return true;
}

static bool amd_candidate_bindings_ready(
        const ggml_cgraph * cgraph, const flagos_pattern_candidate & candidate) {
    if (cgraph == nullptr || candidate.node_indices.empty()) {
        return false;
    }
    for (const int index : candidate.node_indices) {
        if (index < 0 || index >= cgraph->n_nodes ||
            !amd_tensor_bindings_ready(cgraph->nodes[index])) {
            return false;
        }
    }
    return true;
}

static bool amd_tensor_data_overlaps(const ggml_tensor * left, const ggml_tensor * right) {
    if (left == nullptr || right == nullptr || left->data == nullptr || right->data == nullptr) {
        return false;
    }
    const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left->data);
    const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right->data);
    const size_t left_size = ggml_nbytes(left);
    const size_t right_size = ggml_nbytes(right);
    if (left_size == 0 || right_size == 0 || left_begin > UINTPTR_MAX - left_size ||
        right_begin > UINTPTR_MAX - right_size) {
        return true;
    }
    return left_begin < right_begin + right_size && right_begin < left_begin + left_size;
}

static bool amd_tensor_output_can_reuse_input(
        const ggml_tensor * output, const ggml_tensor * input) {
    return !amd_tensor_data_overlaps(output, input) ||
        (output->data == input->data && ggml_nbytes(output) == ggml_nbytes(input));
}

static bool amd_supports_unary_f32(
        const amd_device_context * device, const ggml_tensor * op,
        ggml_unary_op unary, const char * kernel_name) {
    return device != nullptr && device->aot != nullptr && kernel_name != nullptr &&
        device->aot->find(kernel_name) != nullptr && op != nullptr &&
        op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == unary &&
        op->src[0] != nullptr && amd_tensor_is_contiguous_f32(op) &&
        amd_tensor_is_contiguous_f32(op->src[0]) &&
        ggml_are_same_shape(op, op->src[0]) && ggml_nelements(op) > 0 &&
        ggml_nelements(op) <= INT32_MAX &&
        amd_tensor_output_can_reuse_input(op, op->src[0]);
}

static bool amd_supports_l2_norm(const amd_device_context * device, const ggml_tensor * op) {
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_L2_NORM || op->src[0] == nullptr ||
        device->aot->find("flagos_l2_norm_strided_f32") == nullptr ||
        !amd_tensor_is_contiguous_f32(op) || op->src[0]->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(op, op->src[0]) || op->src[0]->nb[0] != sizeof(float) ||
        !amd_tensor_element_offsets_fit_i32(op->src[0], sizeof(float)) ||
        ggml_nelements(op) <= 0 || ggml_nelements(op) > INT32_MAX ||
        op->ne[0] <= 0 || op->ne[0] > INT32_MAX ||
        op->ne[1] <= 0 || op->ne[1] > INT32_MAX ||
        op->ne[2] <= 0 || op->ne[2] > INT32_MAX ||
        op->ne[3] <= 0 || op->ne[3] > INT32_MAX) {
        return false;
    }
    const auto * metadata = device->aot->find("flagos_l2_norm_strided_f32");
    float eps = 0.0f;
    std::memcpy(&eps, op->op_params, sizeof(eps));
    return metadata->block_size >= op->ne[0] && std::isfinite(eps) && eps >= 0.0f &&
        // The strided source may map several logical rows onto a larger
        // backing allocation.  The kernel writes a separate contiguous
        // output layout, so even an exact base-pointer alias is unsafe unless
        // the source is proven identical row-major storage.  Keep this
        // reduction path disjoint and let GGML handle in-place cases.
        !amd_tensor_data_overlaps(op, op->src[0]);
}

static bool amd_supports_ssm_conv(
        const amd_device_context * device, const ggml_tensor * op,
        const char * kernel_name = "flagos_ssm_conv_f32") {
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_SSM_CONV || op->src[0] == nullptr || op->src[1] == nullptr ||
        kernel_name == nullptr || device->aot->find(kernel_name) == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
        op->src[1]->type != GGML_TYPE_F32 || !ggml_is_contiguous(op) ||
        op->src[0]->nb[0] != sizeof(float) || op->src[1]->nb[0] != sizeof(float) ||
        !amd_tensor_element_offsets_fit_i32(op->src[0], sizeof(float)) ||
        !amd_tensor_element_offsets_fit_i32(op->src[1], sizeof(float)) ||
        op->src[1]->ne[0] <= 0 || op->src[1]->ne[0] > INT32_MAX ||
        op->src[1]->ne[1] <= 0 || op->src[1]->ne[1] > INT32_MAX ||
        op->src[1]->ne[2] != 1 || op->src[1]->ne[3] != 1 ||
        op->src[0]->ne[0] != op->src[1]->ne[0] - 1 + op->ne[1] ||
        op->src[0]->ne[1] != op->src[1]->ne[1] ||
        op->src[0]->ne[2] != op->ne[2] || op->src[0]->ne[3] != 1 ||
        op->ne[0] != op->src[1]->ne[1] || op->ne[1] <= 0 ||
        op->ne[2] <= 0 || op->ne[3] != 1 ||
        op->ne[0] > INT32_MAX || op->ne[1] > INT32_MAX || op->ne[2] > INT32_MAX ||
        ggml_nelements(op) <= 0 || ggml_nelements(op) > INT32_MAX) {
        return false;
    }
    return !amd_tensor_data_overlaps(op, op->src[0]) &&
        !amd_tensor_data_overlaps(op, op->src[1]);
}

static bool amd_supports_ssm_conv_silu(
        const amd_device_context * device,
        const ggml_tensor * conv,
        const ggml_tensor * activation) {
    return amd_supports_ssm_conv(device, conv, "flagos_ssm_conv_silu_f32") &&
        activation != nullptr && activation->op == GGML_OP_UNARY &&
        ggml_get_unary_op(activation) == GGML_UNARY_OP_SILU &&
        amd_tensor_is_contiguous_f32(activation) &&
        ggml_are_same_shape(activation, conv) && ggml_nelements(activation) > 0 &&
        ggml_nelements(activation) <= INT32_MAX &&
        activation->src[0] == conv &&
        amd_tensor_output_can_reuse_input(activation, conv) &&
        !amd_tensor_data_overlaps(activation, conv->src[0]) &&
        !amd_tensor_data_overlaps(activation, conv->src[1]);
}

static const ggml_tensor * amd_attention_output_gate_other(
        const ggml_tensor * activation, const ggml_tensor * mul) {
    if (activation == nullptr || mul == nullptr || mul->op != GGML_OP_MUL) {
        return nullptr;
    }
    if (mul->src[0] == activation && mul->src[1] != activation) {
        return mul->src[1];
    }
    if (mul->src[1] == activation && mul->src[0] != activation) {
        return mul->src[0];
    }
    return nullptr;
}

static bool amd_supports_attention_output_gate(
        const amd_device_context * device,
        const ggml_tensor * activation,
        const ggml_tensor * mul) {
    const ggml_tensor * other = amd_attention_output_gate_other(activation, mul);
    if (device == nullptr || device->aot == nullptr || activation == nullptr ||
        activation->src[0] == nullptr || other == nullptr ||
        activation->op != GGML_OP_UNARY ||
        ggml_get_unary_op(activation) != GGML_UNARY_OP_SILU ||
        device->aot->find("flagos_silu_mul_f32") == nullptr ||
        !amd_tensor_is_contiguous_f32(activation) ||
        !amd_tensor_is_contiguous_f32(activation->src[0]) ||
        !amd_tensor_is_contiguous_f32(other) || !amd_tensor_is_contiguous_f32(mul) ||
        !ggml_are_same_shape(activation, activation->src[0]) ||
        !ggml_are_same_shape(mul, activation) || !ggml_are_same_shape(mul, other) ||
        ggml_nelements(mul) <= 0 || ggml_nelements(mul) > INT32_MAX) {
        return false;
    }
    // The fused elementwise kernel may overwrite either live input only at
    // the exact same base and size. Reject partial overlap just as the direct
    // unary and MUL paths do.
    return amd_tensor_output_can_reuse_input(mul, activation->src[0]) &&
        amd_tensor_output_can_reuse_input(mul, other);
}

static bool amd_launch_ssm_conv(
        amd_backend_context * context,
        const ggml_tensor * conv,
        ggml_tensor * output,
        const char * kernel_name) {
    if (context == nullptr || context->device == nullptr || context->device->aot == nullptr ||
        conv == nullptr || conv->src[0] == nullptr || conv->src[1] == nullptr ||
        output == nullptr || kernel_name == nullptr) {
        return false;
    }
    const auto * metadata = context->device->aot->find(kernel_name);
    if (metadata == nullptr) {
        return false;
    }
    int d_conv = static_cast<int>(conv->src[1]->ne[0]);
    int d_inner = static_cast<int>(conv->ne[0]);
    int n_tokens = static_cast<int>(conv->ne[1]);
    int n_seqs = static_cast<int>(conv->ne[2]);
    int ss1 = static_cast<int>(conv->src[0]->nb[1] / sizeof(float));
    int ss2 = static_cast<int>(conv->src[0]->nb[2] / sizeof(float));
    int sc1 = static_cast<int>(conv->src[1]->nb[1] / sizeof(float));
    int so0 = static_cast<int>(output->nb[1] / sizeof(float));
    int so1 = static_cast<int>(output->nb[2] / sizeof(float));
    void * state_data = conv->src[0]->data;
    void * weight_data = conv->src[1]->data;
    void * output_data = output->data;
    flagos_amd::kernel_arguments arguments = {
        &state_data, &weight_data, &output_data,
        &d_conv, &d_inner, &n_tokens, &n_seqs,
        &ss1, &ss2, &sc1, &so0, &so1,
    };
    const unsigned int grid_z = static_cast<unsigned int>(
        (static_cast<uint64_t>(d_inner) + metadata->block_size - 1) /
        metadata->block_size);
    return context->device->aot->launch(
        kernel_name, context->stream,
        static_cast<unsigned int>(n_tokens),
        static_cast<unsigned int>(n_seqs), grid_z, arguments);
}

static bool amd_supports_concat(const amd_device_context * device, const ggml_tensor * op) {
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_CONCAT || op->src[0] == nullptr || op->src[1] == nullptr ||
        device->aot->find("flagos_concat_f32") == nullptr ||
        !amd_tensor_is_contiguous_f32(op) || op->src[0]->type != GGML_TYPE_F32 ||
        op->src[1]->type != GGML_TYPE_F32 ||
        !amd_tensor_element_offsets_fit_i32(op->src[0], sizeof(float)) ||
        !amd_tensor_element_offsets_fit_i32(op->src[1], sizeof(float)) ||
        ggml_nelements(op) <= 0 || ggml_nelements(op) > INT32_MAX) {
        return false;
    }
    const int dim = ggml_get_op_params_i32(op, 0);
    if (dim < 0 || dim >= GGML_MAX_DIMS) {
        return false;
    }
    for (int index = 0; index < GGML_MAX_DIMS; ++index) {
        const int64_t expected = index == dim
            ? op->src[0]->ne[index] + op->src[1]->ne[index]
            : op->src[0]->ne[index];
        if (op->src[0]->ne[index] <= 0 || op->src[1]->ne[index] <= 0 ||
            op->src[0]->ne[index] > INT32_MAX || op->src[1]->ne[index] > INT32_MAX ||
            op->ne[index] != expected ||
            (index != dim && op->src[0]->ne[index] != op->src[1]->ne[index])) {
            return false;
        }
    }
    return !amd_tensor_data_overlaps(op, op->src[0]) &&
        !amd_tensor_data_overlaps(op, op->src[1]);
}

static bool amd_supports_copy_f32(const amd_device_context * device, const ggml_tensor * op) {
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        (op->op != GGML_OP_CPY && op->op != GGML_OP_CONT && op->op != GGML_OP_DUP) ||
        op->src[0] == nullptr || device->aot->find("flagos_copy_strided_f32") == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
        op->nb[0] != sizeof(float) || op->src[0]->nb[0] != sizeof(float) ||
        !amd_tensor_element_offsets_fit_i32(op, sizeof(float)) ||
        !amd_tensor_element_offsets_fit_i32(op->src[0], sizeof(float)) ||
        ggml_nelements(op) > INT32_MAX ||
        ggml_nelements(op) != ggml_nelements(op->src[0])) {
        return false;
    }
    if (op->op == GGML_OP_CPY && (op->src[1] == nullptr ||
        op->src[1]->type != GGML_TYPE_F32 || op->data != op->src[1]->data)) {
        return false;
    }
    if (ggml_nelements(op) == 0) {
        return true;
    }
    return !amd_tensor_data_overlaps(op, op->src[0]);
}

static bool amd_supports_gated_delta_net(
        const amd_device_context * device, const ggml_tensor * op) {
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_GATED_DELTA_NET ||
        device->aot->find("flagos_gated_delta_net_scalar_f32") == nullptr ||
        !amd_tensor_is_contiguous_f32(op) || ggml_nelements(op) <= 0 ||
        ggml_nelements(op) > INT32_MAX) {
        return false;
    }
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * gate = op->src[3];
    const ggml_tensor * beta = op->src[4];
    const ggml_tensor * state = op->src[5];
    if (q == nullptr || k == nullptr || v == nullptr || gate == nullptr ||
        beta == nullptr || state == nullptr ||
        q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 ||
        v->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 ||
        beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) ||
        !ggml_is_contiguous_rows(v) || !ggml_is_contiguous(gate) ||
        !ggml_is_contiguous(beta) || !ggml_is_contiguous(state)) {
        return false;
    }

    const int64_t state_size = v->ne[0];
    const int64_t heads = v->ne[1];
    const int64_t tokens = v->ne[2];
    const int64_t sequences = v->ne[3];
    const int64_t q_heads = q->ne[1];
    const int64_t q_sequences = q->ne[3];
    const int64_t snapshots = ggml_get_op_params_i32(op, 0);
    const auto * metadata = device->aot->find("flagos_gated_delta_net_scalar_f32");
    if (state_size <= 0 || state_size > INT32_MAX || heads <= 0 || heads > INT32_MAX ||
        tokens <= 0 || tokens > INT32_MAX || sequences <= 0 || sequences > INT32_MAX ||
        q_heads <= 0 || q_heads > heads || heads % q_heads != 0 ||
        q_sequences <= 0 || q_sequences > sequences || sequences % q_sequences != 0 ||
        snapshots <= 0 || state_size % 16 != 0 ||
        metadata->block_size < state_size || metadata->tile_n <= 0 ||
        state_size > INT32_MAX / state_size ||
        state_size * state_size > INT32_MAX / heads ||
        state_size * state_size * heads > INT32_MAX / sequences ||
        tokens > INT32_MAX / sequences ||
        snapshots > INT32_MAX / (state_size * sequences) ||
        q->ne[0] != state_size || q->ne[2] != tokens ||
        k->ne[0] != state_size || k->ne[1] != q_heads ||
        k->ne[2] != tokens || k->ne[3] != q_sequences ||
        gate->ne[0] != 1 || gate->ne[1] != heads ||
        gate->ne[2] != tokens || gate->ne[3] != sequences ||
        beta->ne[0] != 1 || beta->ne[1] != heads ||
        beta->ne[2] != tokens || beta->ne[3] != sequences ||
        state->ne[0] != state_size || state->ne[1] != state_size ||
        state->ne[2] != heads || state->ne[3] != sequences) {
        return false;
    }
    const uint64_t expected_ne0 = static_cast<uint64_t>(state_size) * heads;
    const uint64_t expected_ne1 = static_cast<uint64_t>(tokens) * sequences +
        static_cast<uint64_t>(snapshots) * state_size * sequences;
    if (expected_ne0 > INT64_MAX || expected_ne1 > INT64_MAX ||
        op->ne[0] != static_cast<int64_t>(expected_ne0) ||
        op->ne[1] != static_cast<int64_t>(expected_ne1) ||
        op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (q->ne[dim] != k->ne[dim] || q->nb[dim] != k->nb[dim] ||
            gate->ne[dim] != beta->ne[dim] || gate->nb[dim] != beta->nb[dim]) {
            return false;
        }
    }
    const ggml_tensor * inputs[] = { q, k, v, gate, beta, state };
    for (const ggml_tensor * input : inputs) {
        if (!amd_tensor_element_offsets_fit_i32(input, sizeof(float)) ||
            !amd_tensor_data_is_aligned(input, 16) ||
            amd_tensor_data_overlaps(op, input)) {
            return false;
        }
    }
    // The model-shape AOT package attaches tt.divisibility=16 to the Q/K/V
    // strides. ggml_is_contiguous_rows() proves only nb[0], so make the
    // stronger compiler contract explicit at the provider boundary.
    return amd_tensor_data_is_aligned(op, 16) &&
        amd_tensor_element_strides_are_aligned(q, sizeof(float), 16, 1) &&
        amd_tensor_element_strides_are_aligned(v, sizeof(float), 16, 1);
}

static bool amd_supports_gated_delta_net_cache(
        const amd_device_context * device,
        const ggml_tensor * gdn,
        const ggml_tensor * snapshot_view,
        const ggml_tensor * copy,
        const char * kernel_name) {
    if (!amd_supports_gated_delta_net(device, gdn) || snapshot_view == nullptr ||
        copy == nullptr || kernel_name == nullptr || device->aot->find(kernel_name) == nullptr ||
        snapshot_view->op != GGML_OP_VIEW || snapshot_view->type != GGML_TYPE_F32 ||
        snapshot_view->src[0] != gdn || snapshot_view->view_src != gdn ||
        copy->op != GGML_OP_CPY || copy->type != GGML_TYPE_F32 ||
        copy->src[0] != snapshot_view || copy->src[1] == nullptr ||
        copy->src[1]->op != GGML_OP_VIEW || copy->src[1]->type != GGML_TYPE_F32 ||
        copy->data == nullptr || copy->data != copy->src[1]->data ||
        gdn->data == nullptr || snapshot_view->data == nullptr ||
        !ggml_is_contiguous(snapshot_view)) {
        return false;
    }
    const ggml_tensor * query = gdn->src[0];
    const ggml_tensor * value = gdn->src[2];
    const int64_t state_size = value->ne[0];
    const int64_t heads = value->ne[1];
    const int64_t tokens = value->ne[2];
    const int64_t sequences = value->ne[3];
    const int64_t q_heads = query->ne[1];
    const int64_t q_sequences = query->ne[3];
    const int64_t snapshots = ggml_get_op_params_i32(gdn, 0);
    const bool decode_only = std::strcmp(
        kernel_name, "flagos_gated_delta_net_scalar_f32_cache_only_decode") == 0;
    const int64_t state_elements = state_size * state_size * heads;
    const int64_t written = std::min(tokens, snapshots);
    const int64_t attention_elements = state_size * heads * tokens * sequences;
    const size_t attention_bytes = ggml_row_size(GGML_TYPE_F32, attention_elements);
    const ggml_tensor * destination = copy->src[1];
    const std::array<int64_t, GGML_MAX_DIMS> expected = {
        state_elements, sequences, written, 1,
    };
    const auto * metadata = device->aot->find(kernel_name);
    const uintptr_t gdn_data = reinterpret_cast<uintptr_t>(gdn->data);
    if (gdn_data > UINTPTR_MAX - attention_bytes ||
        reinterpret_cast<uintptr_t>(snapshot_view->data) != gdn_data + attention_bytes ||
        snapshot_view->view_offs != attention_bytes ||
        ggml_nelements(snapshot_view) != state_elements * sequences * written ||
        !std::equal(expected.begin(), expected.end(), destination->ne) ||
        !std::equal(expected.begin(), expected.end(), copy->ne) ||
        destination->nb[0] != sizeof(float) ||
        destination->nb[1] != static_cast<size_t>(state_elements) * sizeof(float) ||
        (written > 1 && (destination->nb[2] % sizeof(float) != 0 ||
                         destination->nb[2] / sizeof(float) > INT32_MAX ||
                         destination->nb[2] / sizeof(float) <
                             static_cast<size_t>(state_elements * sequences))) ||
        metadata->block_size < state_size || metadata->tile_n <= 0 ||
        !amd_tensor_data_is_aligned(destination, 16) ||
        (decode_only &&
         (tokens != 1 || snapshots != 1 || !metadata->exact_block_size ||
          (q_heads & (q_heads - 1)) != 0 || q_sequences != sequences)) ||
        (metadata->exact_block_size &&
         (metadata->block_size != state_size ||
          state_size % metadata->tile_n != 0)) ||
        (written > 1 &&
         (destination->nb[2] / sizeof(float)) % 16 != 0) ||
        amd_tensor_data_overlaps(gdn, destination)) {
        return false;
    }
    // The fused kernel writes the cache while other workgroups may still be
    // loading the initial recurrent state.  Even though an exact in-place tile
    // can look safe locally, arbitrary overlap is not ordered across workgroups.
    for (int source = 0; source < 6; ++source) {
        if (amd_tensor_data_overlaps(destination, gdn->src[source])) {
            return false;
        }
    }
    return true;
}

static bool amd_gated_delta_net_cache_only_outputs_are_safe(
        const ggml_cgraph * cgraph,
        const flagos_pattern_candidate & candidate,
        const ggml_tensor * gdn,
        const ggml_tensor * copy) {
    if (cgraph == nullptr || gdn == nullptr || copy == nullptr || gdn->data == nullptr ||
        ggml_get_op_params_i32(gdn, 0) != 1 ||
        candidate.node_indices.empty() || candidate.node_indices.front() < 0 ||
        candidate.node_indices.front() >= cgraph->n_nodes ||
        cgraph->nodes[candidate.node_indices.front()] != gdn ||
        candidate.node_indices.back() < 0 ||
        candidate.node_indices.back() >= cgraph->n_nodes ||
        cgraph->nodes[candidate.node_indices.back()] != copy) {
        return false;
    }
    const ggml_tensor * value = gdn->src[2];
    if (value == nullptr || value->ne[0] <= 0 || value->ne[1] <= 0 ||
        value->ne[2] <= 0 || value->ne[3] <= 0) {
        return false;
    }
    uint64_t attention_elements = 1;
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        const uint64_t extent = static_cast<uint64_t>(value->ne[dimension]);
        if (extent > UINT64_MAX / attention_elements) {
            return false;
        }
        attention_elements *= extent;
    }
    if (attention_elements > SIZE_MAX / sizeof(float)) {
        return false;
    }
    const uintptr_t attention_begin = reinterpret_cast<uintptr_t>(gdn->data);
    const size_t attention_bytes = static_cast<size_t>(attention_elements) * sizeof(float);
    if (attention_begin > UINTPTR_MAX - attention_bytes) {
        return false;
    }
    const uintptr_t attention_end = attention_begin + attention_bytes;
    const int copy_index = candidate.node_indices.back();
    const ggml_tensor * destination = copy->src[1];
    if (destination == nullptr) {
        return false;
    }
    std::vector<bool> covered(static_cast<size_t>(cgraph->n_nodes), false);
    for (const int index : candidate.node_indices) {
        if (index < 0 || index >= cgraph->n_nodes || covered[static_cast<size_t>(index)]) {
            return false;
        }
        covered[static_cast<size_t>(index)] = true;
    }
    const auto alias_is_in_attention = [&](const ggml_tensor * tensor) {
        if (tensor == nullptr || tensor->data == nullptr || tensor->view_src != gdn) {
            return false;
        }
        const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor->data);
        const size_t bytes = ggml_nbytes(tensor);
        return bytes > 0 && begin >= attention_begin && begin <= UINTPTR_MAX - bytes &&
            begin + bytes <= attention_end;
    };
    // A zero-work view may be absent from cgraph->nodes and appear only as a
    // later node's source.  Required-output bookkeeping is node based, so
    // audit those implicit aliases here before omitting the snapshot suffix.
    // The covered CPY is allowed to read the suffix because this kernel writes
    // its destination directly; every other consumer must stay in the
    // materialized attention prefix.
    for (int consumer_index = 0; consumer_index < cgraph->n_nodes; ++consumer_index) {
        if (covered[static_cast<size_t>(consumer_index)]) {
            continue;
        }
        const ggml_tensor * consumer = cgraph->nodes[consumer_index];
        if (consumer == nullptr ||
            (consumer->view_src == gdn && !alias_is_in_attention(consumer))) {
            return false;
        }
        for (int source = 0; source < GGML_MAX_SRC; ++source) {
            const ggml_tensor * input = consumer->src[source];
            const bool attention_view_input = input == gdn &&
                consumer->view_src == gdn && alias_is_in_attention(consumer);
            if ((input == gdn && !attention_view_input) ||
                (input != nullptr && input->view_src == gdn && !alias_is_in_attention(input))) {
                return false;
            }
        }
    }
    bool copy_required = false;
    for (const int index : candidate.required_output_node_indices) {
        if (index == copy_index) {
            if (copy_required) {
                return false;
            }
            copy_required = true;
            continue;
        }
        if (index < 0 || index >= cgraph->n_nodes ||
            std::find(candidate.node_indices.begin(), candidate.node_indices.end(), index) ==
                candidate.node_indices.end()) {
            return false;
        }
        const ggml_tensor * output = cgraph->nodes[index];
        if (output == nullptr || output->data == nullptr) {
            return false;
        }
        // A scheduled destination VIEW and the terminal CPY expose the same
        // cache storage.  The cache-only kernel materializes both contracts
        // with its direct cache write.
        if (output == destination) {
            continue;
        }
        // Common records a producer node, not the byte range observed through
        // a later view.  The external-alias audit above proves that every
        // consumer of this GDN output stays inside the attention prefix.
        if (output == gdn && (gdn->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
            continue;
        }
        const uintptr_t begin = reinterpret_cast<uintptr_t>(output->data);
        const size_t bytes = ggml_nbytes(output);
        if (bytes == 0 || begin < attention_begin || begin > UINTPTR_MAX - bytes ||
            begin + bytes > attention_end) {
            return false;
        }
    }
    return copy_required;
}

static bool amd_supports_rope_neox(const amd_device_context * device, const ggml_tensor * op) {
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_ROPE || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->src[2] != nullptr || device->aot->find("flagos_rope_neox_f32") == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
        op->src[1]->type != GGML_TYPE_I32 || !ggml_are_same_shape(op, op->src[0]) ||
        !ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) ||
        !ggml_is_contiguous(op->src[1]) ||
        op->src[1]->ne[1] != 1 || op->src[1]->ne[2] != 1 || op->src[1]->ne[3] != 1 ||
        op->ne[0] <= 0 || op->ne[0] % 2 != 0 ||
        op->ne[1] <= 0 || op->ne[2] <= 0 || op->ne[3] != 1 ||
        op->src[0]->ne[2] != op->src[1]->ne[0] || op->src[0]->ne[3] != 1 ||
        ggml_nelements(op) <= 0 || ggml_nelements(op) > INT32_MAX ||
        op->ne[0] > INT32_MAX || op->ne[1] > INT32_MAX || op->ne[2] > INT32_MAX) {
        return false;
    }
    if (!amd_tensor_output_can_reuse_input(op, op->src[0]) ||
        amd_tensor_data_overlaps(op, op->src[1])) {
        return false;
    }

    const int32_t * params = static_cast<const int32_t *>(op->op_params);
    float freq_scale = 0.0f;
    float freq_base = 0.0f;
    float ext_factor = 0.0f;
    float attn_factor = 0.0f;
    std::memcpy(&freq_base, params + 5, sizeof(freq_base));
    std::memcpy(&freq_scale, params + 6, sizeof(freq_scale));
    std::memcpy(&ext_factor, params + 7, sizeof(ext_factor));
    std::memcpy(&attn_factor, params + 8, sizeof(attn_factor));
    return params[2] == GGML_ROPE_TYPE_NEOX &&
        params[1] > 0 && params[1] % 2 == 0 && params[1] == op->ne[0] &&
        std::isfinite(freq_base) && freq_base > 0.0f &&
        std::isfinite(freq_scale) && freq_scale == 1.0f &&
        std::isfinite(ext_factor) && ext_factor == 0.0f &&
        std::isfinite(attn_factor) && attn_factor == 1.0f;
}

static bool amd_supports_mrope(const amd_device_context * device, const ggml_tensor * op) {
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_ROPE || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->src[2] != nullptr || device->aot->find("flagos_mrope_f32") == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
        op->src[1]->type != GGML_TYPE_I32 || !ggml_are_same_shape(op, op->src[0]) ||
        !ggml_is_contiguous(op) || op->src[0]->nb[0] != sizeof(float) ||
        !ggml_is_contiguous(op->src[1]) ||
        !amd_tensor_element_offsets_fit_i32(op, sizeof(float)) ||
        !amd_tensor_element_offsets_fit_i32(op->src[0], sizeof(float)) ||
        op->src[1]->ne[0] != op->ne[2] * 4 ||
        op->src[1]->ne[1] != 1 || op->src[1]->ne[2] != 1 || op->src[1]->ne[3] != 1 ||
        op->ne[0] <= 0 || op->ne[0] > INT32_MAX || op->ne[0] % 2 != 0 ||
        op->ne[1] <= 0 || op->ne[1] > INT32_MAX ||
        op->ne[2] <= 0 || op->ne[2] > INT32_MAX ||
        op->ne[3] <= 0 || op->ne[3] > INT32_MAX ||
        ggml_nrows(op) > std::numeric_limits<unsigned int>::max()) {
        return false;
    }

    const int32_t * params = static_cast<const int32_t *>(op->op_params);
    float freq_base = 0.0f;
    float freq_scale = 0.0f;
    float ext_factor = 0.0f;
    float attn_factor = 0.0f;
    std::memcpy(&freq_base, params + 5, sizeof(freq_base));
    std::memcpy(&freq_scale, params + 6, sizeof(freq_scale));
    std::memcpy(&ext_factor, params + 7, sizeof(ext_factor));
    std::memcpy(&attn_factor, params + 8, sizeof(attn_factor));
    int64_t section_total = 0;
    for (int section = 0; section < GGML_MROPE_SECTIONS; ++section) {
        if (params[11 + section] < 0) {
            return false;
        }
        section_total += params[11 + section];
    }
    const auto * metadata = device->aot->find("flagos_mrope_f32");
    return params[2] == GGML_ROPE_TYPE_MROPE &&
        params[1] > 0 && params[1] <= op->ne[0] && params[1] % 2 == 0 &&
        section_total == params[1] / 2 && metadata->block_size >= op->ne[0] &&
        std::isfinite(freq_base) && freq_base > 0.0f &&
        std::isfinite(freq_scale) && freq_scale == 1.0f &&
        std::isfinite(ext_factor) && ext_factor == 0.0f &&
        std::isfinite(attn_factor) && attn_factor == 1.0f &&
        amd_tensor_output_can_reuse_input(op, op->src[0]) &&
        !amd_tensor_data_overlaps(op, op->src[1]);
}

static bool amd_supports_silu(const amd_device_context * device, const ggml_tensor * op) {
    return device != nullptr && device->aot != nullptr && op != nullptr &&
        op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_SILU &&
        device->aot->find("flagos_silu_f32") != nullptr && op->src[0] != nullptr &&
        amd_tensor_is_contiguous_f32(op) && amd_tensor_is_contiguous_f32(op->src[0]) &&
        ggml_are_same_shape(op, op->src[0]) && ggml_nelements(op) > 0 &&
        ggml_nelements(op) <= INT32_MAX &&
        amd_tensor_output_can_reuse_input(op, op->src[0]);
}

static bool amd_supports_set_rows(const amd_device_context * device, const ggml_tensor * op) {
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_SET_ROWS || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->src[2] == nullptr || device->aot->find("flagos_set_rows_f32_f16") == nullptr ||
        op->type != GGML_TYPE_F16 || op->src[0]->type != GGML_TYPE_F32 ||
        op->src[1]->type != GGML_TYPE_I64 || op->src[2]->type != GGML_TYPE_F16 ||
        op->ne[0] <= 0 || op->ne[1] <= 0 || op->ne[2] != 1 || op->ne[3] != 1 ||
        op->src[0]->ne[0] != op->ne[0] || op->src[0]->ne[1] <= 0 ||
        op->src[0]->ne[2] != 1 || op->src[0]->ne[3] != 1 ||
        op->src[1]->ne[0] != op->src[0]->ne[1] || op->src[1]->ne[1] != 1 ||
        op->src[1]->ne[2] != 1 || op->src[1]->ne[3] != 1 ||
        !ggml_are_same_shape(op, op->src[2]) ||
        !ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) ||
        !ggml_is_contiguous(op->src[1]) || !ggml_is_contiguous(op->src[2]) ||
        ggml_nelements(op) > INT32_MAX || ggml_nelements(op->src[0]) > INT32_MAX ||
        op->ne[0] > INT32_MAX || op->ne[1] > INT32_MAX) {
        return false;
    }
    // SET_ROWS updates the destination tensor supplied as src[2].  Reject a
    // scheduler layout where GGML's output pointer and that destination
    // diverge; the Triton ABI has one destination pointer and cannot preserve
    // both interpretations.
    if (op->data != nullptr && op->src[2]->data != nullptr &&
        op->data != op->src[2]->data) {
        return false;
    }
    return !amd_tensor_data_overlaps(op, op->src[0]) &&
        !amd_tensor_data_overlaps(op, op->src[1]);
}

static bool amd_supports_get_rows(const amd_device_context * device, const ggml_tensor * op) {
    const flagos_quantized_matmul_kind kind = op == nullptr || op->src[0] == nullptr
        ? flagos_quantized_matmul_kind::none : amd_quantized_kind(op->src[0]->type);
    const int64_t block_size = amd_quantized_block_size(kind);
    const char * kernel = amd_quantized_get_rows_kernel(kind);
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_GET_ROWS || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_I32 ||
        block_size == 0 || kernel == nullptr ||
        op->ne[0] <= 0 || op->ne[0] % block_size != 0 || op->ne[1] <= 0 ||
        op->src[0]->ne[0] != op->ne[0] ||
        op->ne[2] != 1 || op->ne[3] != 1 || op->src[0]->ne[1] <= 0 ||
        op->src[0]->ne[2] != 1 || op->src[0]->ne[3] != 1 ||
        ggml_nelements(op->src[1]) != op->ne[1] || !ggml_is_contiguous(op) ||
        !ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1]) ||
        ggml_nelements(op) > INT32_MAX ||
        op->ne[0] > INT32_MAX || op->ne[1] > INT32_MAX) {
        return false;
    }
    // The gather kernels form packed-byte addresses as
    // ``row_index * blocks_per_row * block_bytes`` using Triton's i32 scalar
    // arithmetic.  Bounding only the F32 output is insufficient for a large
    // embedding table: a valid row index could still wrap the packed source
    // address.  The source is contiguous, so its byte span is the exact bound
    // required by both the u8 and aliased fp16 pointer views.
    if (!amd_tensor_element_offsets_fit_i32(op->src[0], 1)) {
        return false;
    }
    return device->aot->find(kernel) != nullptr &&
        !amd_tensor_data_overlaps(op, op->src[0]) &&
        !amd_tensor_data_overlaps(op, op->src[1]);
}

static bool amd_i32_matrix_offsets_fit(
        const flagos_quantized_matmul_signature & signature) {
    const int64_t limit = INT32_MAX;
    return signature.k > 0 && signature.rows > 0 && signature.columns > 0 &&
        signature.k <= limit && signature.rows <= limit && signature.columns <= limit &&
        signature.k <= limit / signature.rows &&
        signature.k <= limit / signature.columns &&
        signature.rows <= limit / signature.columns;
}

static bool amd_supports_quantized_mul_mat(const amd_device_context * device, const ggml_tensor * op) {
    if (device == nullptr || device->aot == nullptr || op == nullptr) {
        return false;
    }
    flagos_quantized_matmul_signature signature;
    if (!flagos_describe_quantized_matmul(op, &signature) ||
        !amd_i32_matrix_offsets_fit(signature)) {
        return false;
    }
    const char * kernel = signature.columns == 1
        ? amd_quantized_gemv_kernel(signature.weight_kind)
        : amd_quantized_batched_kernel(device, signature.weight_kind);
    return kernel != nullptr && device->aot->find(kernel) != nullptr &&
        !amd_tensor_data_overlaps(op, op->src[0]) &&
        !amd_tensor_data_overlaps(op, op->src[1]);
}

// The FFN fusion consumes the same quantized projections accepted by the
// opt-in F16 dequant-cache path, but combines both projections and the
// terminal split SwiGLU into one launch.  Keep this capability deliberately
// narrow: the Triton ABI assumes two row-major F16 weight caches, one
// contiguous F32 activation matrix, and a contiguous F32 output matrix.
static bool amd_supports_ffn_swiglu_f16_layout(const amd_device_context * device,
                                               const ggml_tensor * gate,
                                               const ggml_tensor * up,
                                               const ggml_tensor * glu) {
    if (device == nullptr || device->aot == nullptr || gate == nullptr || up == nullptr || glu == nullptr ||
        gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT ||
        glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU ||
        glu->src[0] == nullptr || glu->src[1] == nullptr ||
        !((glu->src[0] == gate && glu->src[1] == up) ||
          (glu->src[0] == up && glu->src[1] == gate))) {
        return false;
    }

    flagos_quantized_matmul_signature gate_signature;
    flagos_quantized_matmul_signature up_signature;
    if (!flagos_describe_quantized_matmul(gate, &gate_signature) ||
        !flagos_describe_quantized_matmul(up, &up_signature) ||
        !amd_i32_matrix_offsets_fit(gate_signature) ||
        !amd_i32_matrix_offsets_fit(up_signature)) {
        return false;
    }
    const ggml_tensor * gate_weights = gate->src[0];
    const ggml_tensor * up_weights = up->src[0];
    const ggml_tensor * activation = gate->src[1];
    if (activation == nullptr || activation != up->src[1] || activation->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(activation) || !ggml_is_contiguous(gate) ||
        !ggml_is_contiguous(up) || !ggml_is_contiguous(glu) ||
        gate_signature.k != up_signature.k || gate_signature.rows != up_signature.rows ||
        gate_signature.columns != up_signature.columns || gate_signature.columns <= 1 ||
        gate_signature.rows < 64 || gate_signature.k < 1024 ||
        glu->type != GGML_TYPE_F32 || glu->ne[0] != gate_signature.rows ||
        glu->ne[1] != gate_signature.columns || glu->ne[2] != 1 || glu->ne[3] != 1 ||
        gate->ne[0] != gate_signature.rows || gate->ne[1] != gate_signature.columns ||
        up->ne[0] != up_signature.rows || up->ne[1] != up_signature.columns ||
        activation->ne[0] != gate_signature.k || activation->ne[1] != gate_signature.columns ||
        activation->ne[2] != 1 || activation->ne[3] != 1) {
        return false;
    }
    const auto has_dequant = [device](flagos_quantized_matmul_kind kind) {
        const char * kernel = amd_quantized_dequant_kernel(kind);
        return kernel != nullptr && device->aot->find(kernel) != nullptr;
    };
    if (!has_dequant(gate_signature.weight_kind) || !has_dequant(up_signature.weight_kind)) {
        return false;
    }

    // The fused launch overwrites only the terminal GLU allocation.  Reject
    // any known alias with an input, an intermediate projection, or a weight;
    // otherwise the first tile could clobber data still needed by a later
    // tile.  Null data is accepted during early graph construction, while
    // execution-time tensors are always checked by the launcher below.
    if (amd_tensor_data_overlaps(glu, activation) ||
        amd_tensor_data_overlaps(glu, gate_weights) ||
        amd_tensor_data_overlaps(glu, up_weights)) {
        return false;
    }
    if (glu->data != nullptr &&
        (glu->data == gate->data || glu->data == up->data)) {
        // The projection outputs are covered graph intermediates and are not
        // read by the fused kernel.  Retain the existing exact-alias guard until
        // that allocator-reuse layout has been validated numerically.
        return false;
    }
    if (gate->data != nullptr && up->data != nullptr && gate->data == up->data) {
        return false;
    }
    return true;
}

static bool amd_supports_ffn_swiglu_f16(const amd_device_context * device,
                                        const ggml_tensor * gate,
                                        const ggml_tensor * up,
                                        const ggml_tensor * glu) {
    return device != nullptr && device->aot != nullptr &&
        amd_fusion_enabled(device, "ffn_swiglu") && amd_prefill_f16_gemm_enabled(device) &&
        device->aot->find("flagos_ffn_swiglu_f16_f32_batched") != nullptr &&
        amd_supports_ffn_swiglu_f16_layout(device, gate, up, glu);
}

// Store the graph-private GLU result as F16 scratch for the down projection.
static bool amd_supports_ffn_swiglu_down_f16(const amd_device_context * device,
                                             const ggml_tensor * gate,
                                             const ggml_tensor * up,
                                             const ggml_tensor * glu,
                                             const ggml_tensor * down) {
    if (device == nullptr || device->aot == nullptr || down == nullptr ||
        !amd_fusion_enabled(device, "ffn_swiglu_down") || !amd_prefill_f16_gemm_enabled(device) ||
        !amd_ffn_down_f16_enabled(device) ||
        !amd_supports_ffn_swiglu_f16_layout(device, gate, up, glu) ||
        down->op != GGML_OP_MUL_MAT || down->src[1] != glu) {
        return false;
    }

    const auto * ffn_metadata = device->aot->find("flagos_ffn_swiglu_f16_f16_grouped");
    const auto * down_metadata = device->aot->find("flagos_mul_mat_f16_f16_grouped");
    if (ffn_metadata == nullptr || down_metadata == nullptr ||
        ffn_metadata->tile_m <= 0 || ffn_metadata->tile_n <= 0 || ffn_metadata->tile_k <= 0 ||
        down_metadata->tile_m <= 0 || down_metadata->tile_n <= 0 || down_metadata->tile_k <= 0) {
        return false;
    }

    flagos_quantized_matmul_signature gate_signature;
    flagos_quantized_matmul_signature up_signature;
    flagos_quantized_matmul_signature down_signature;
    if (!flagos_describe_quantized_matmul(gate, &gate_signature) ||
        !flagos_describe_quantized_matmul(up, &up_signature) ||
        !flagos_describe_quantized_matmul(down, &down_signature) ||
        gate_signature.columns <= 1 || gate_signature.columns != up_signature.columns ||
        gate_signature.columns != down_signature.columns ||
        gate_signature.rows != up_signature.rows || gate_signature.rows != down_signature.k ||
        down_signature.rows < 64 || down_signature.k < 1024 ||
        !amd_i32_matrix_offsets_fit(gate_signature) ||
        !amd_i32_matrix_offsets_fit(up_signature) ||
        !amd_i32_matrix_offsets_fit(down_signature) ||
        down->ne[0] != down_signature.rows || down->ne[1] != down_signature.columns ||
        !ggml_is_contiguous(down) || !ggml_is_contiguous(down->src[0]) ||
        !amd_grouped_f16_ffn_enabled(device, static_cast<int>(gate_signature.columns))) {
        return false;
    }
    const char * down_dequant = amd_quantized_dequant_kernel(down_signature.weight_kind);
    if (down_dequant == nullptr || device->aot->find(down_dequant) == nullptr) {
        return false;
    }

    const auto grid_fits = [](int64_t rows, int64_t columns,
                              int tile_m, int tile_n) {
        const uint64_t blocks_m = (static_cast<uint64_t>(rows) + tile_m - 1) / tile_m;
        const uint64_t blocks_n = (static_cast<uint64_t>(columns) + tile_n - 1) / tile_n;
        return blocks_m != 0 && blocks_n != 0 &&
            blocks_m <= std::numeric_limits<unsigned int>::max() / blocks_n;
    };
    if (!grid_fits(gate_signature.rows, gate_signature.columns,
                   ffn_metadata->tile_m, ffn_metadata->tile_n) ||
        !grid_fits(down_signature.rows, down_signature.columns,
                   down_metadata->tile_m, down_metadata->tile_n)) {
        return false;
    }

    const uint64_t scratch_elements = static_cast<uint64_t>(gate_signature.rows) *
        static_cast<uint64_t>(gate_signature.columns);
    if (scratch_elements > SIZE_MAX / sizeof(ggml_fp16_t) ||
        ggml_nbytes(glu) < static_cast<size_t>(scratch_elements) * sizeof(ggml_fp16_t)) {
        return false;
    }

    // Private scratch cannot overlap inputs used by either launch.
    const ggml_tensor * scratch_inputs[] = {
        gate->src[0], up->src[0], gate->src[1], down->src[0], down,
    };
    for (const ggml_tensor * input : scratch_inputs) {
        if (amd_tensor_data_overlaps(glu, input)) {
            return false;
        }
    }

    // The terminal output may reuse dead activations, but not weights or live scratch.
    const ggml_tensor * persistent_inputs[] = {
        gate->src[0], up->src[0], down->src[0],
    };
    for (const ggml_tensor * input : persistent_inputs) {
        if (amd_tensor_data_overlaps(down, input)) {
            return false;
        }
    }
    return !amd_tensor_data_overlaps(down, glu);
}

// Decode uses the same provider-neutral FFN/SwiGLU graph pattern as the F16
// prefill path, but keeps packed Q4_K or Q4_0 weights in place. The AMD
// implementation fuses both quantized GEMVs and the terminal activation,
// shares the single F32 activation vector, and materializes only the terminal
// GLU output.
static const char * amd_ffn_swiglu_q4_decode_kernel(
        const amd_device_context * device, flagos_quantized_matmul_kind kind) {
    if (device == nullptr || device->aot == nullptr) {
        return nullptr;
    }
    if (kind == flagos_quantized_matmul_kind::q4_0 &&
        amd_env_enabled("FLAGOS_AMD_Q40_FFN_DECODE_STAGED")) {
        static constexpr char staged[] =
            "flagos_ffn_swiglu_q4_0_f32_decode_staged";
        return device->aot->find(staged) != nullptr ? staged : nullptr;
    }
    const char * name = kind == flagos_quantized_matmul_kind::q4_0
        ? "flagos_ffn_swiglu_q4_0_f32_decode"
        : kind == flagos_quantized_matmul_kind::q4_k
        ? "flagos_ffn_swiglu_q4_k_f32_decode" : nullptr;
    return name != nullptr && device->aot->find(name) != nullptr ? name : nullptr;
}

static bool amd_supports_ffn_swiglu_q4_decode(const amd_device_context * device,
                                              const ggml_tensor * gate,
                                              const ggml_tensor * up,
                                              const ggml_tensor * glu) {
    if (device == nullptr || device->aot == nullptr || gate == nullptr || up == nullptr || glu == nullptr ||
        !amd_fusion_enabled(device, "ffn_swiglu") ||
        gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT ||
        glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU ||
        glu->src[0] == nullptr || glu->src[1] == nullptr ||
        !((glu->src[0] == gate && glu->src[1] == up) ||
          (glu->src[0] == up && glu->src[1] == gate))) {
        return false;
    }
    flagos_quantized_matmul_signature gate_signature;
    flagos_quantized_matmul_signature up_signature;
    if (!flagos_describe_quantized_matmul(gate, &gate_signature) ||
        !flagos_describe_quantized_matmul(up, &up_signature) ||
        gate_signature.weight_kind != up_signature.weight_kind ||
        (gate_signature.weight_kind != flagos_quantized_matmul_kind::q4_0 &&
         gate_signature.weight_kind != flagos_quantized_matmul_kind::q4_k) ||
        !amd_i32_matrix_offsets_fit(gate_signature) ||
        !amd_i32_matrix_offsets_fit(up_signature)) {
        return false;
    }
    const char * kernel_name = amd_ffn_swiglu_q4_decode_kernel(
        device, gate_signature.weight_kind);
    const auto * metadata = kernel_name == nullptr ? nullptr : device->aot->find(kernel_name);
    if (metadata == nullptr || metadata->block_size <= 0) {
        return false;
    }
    const ggml_tensor * activation = gate->src[1];
    const ggml_tensor * gate_weights = gate->src[0];
    const ggml_tensor * up_weights = up->src[0];
    const int64_t row_tile = metadata->block_size;
    if (activation == nullptr || activation != up->src[1] || activation->type != GGML_TYPE_F32 ||
        gate_weights == nullptr || up_weights == nullptr ||
        !ggml_is_contiguous(activation) || !ggml_is_contiguous(gate_weights) ||
        !ggml_is_contiguous(up_weights) || !ggml_is_contiguous(gate) ||
        !ggml_is_contiguous(up) || !ggml_is_contiguous(glu) ||
        gate_signature.k != up_signature.k || gate_signature.rows != up_signature.rows ||
        gate_signature.columns != 1 || up_signature.columns != 1 ||
        gate_signature.rows <= 0 || gate_signature.rows % row_tile != 0 ||
        gate_signature.rows > INT32_MAX || gate_signature.k <= 0 || gate_signature.k > INT32_MAX ||
        glu->type != GGML_TYPE_F32 || glu->ne[0] != gate_signature.rows ||
        glu->ne[1] != 1 || glu->ne[2] != 1 || glu->ne[3] != 1 ||
        gate->ne[0] != gate_signature.rows || gate->ne[1] != 1 ||
        up->ne[0] != up_signature.rows || up->ne[1] != 1 ||
        activation->ne[0] != gate_signature.k || activation->ne[1] != 1 ||
        activation->ne[2] != 1 || activation->ne[3] != 1) {
        return false;
    }

    if (amd_tensor_data_overlaps(glu, activation) ||
        amd_tensor_data_overlaps(glu, gate_weights) ||
        amd_tensor_data_overlaps(glu, up_weights)) {
        return false;
    }
    if (glu->data != nullptr &&
        (glu->data == gate->data || glu->data == up->data)) {
        return false;
    }
    return gate->data == nullptr || up->data == nullptr || gate->data != up->data;
}

static bool amd_supports_rope_kv_store(const amd_device_context * device,
                                       const ggml_tensor * rope,
                                       const ggml_tensor * view,
                                       const ggml_tensor * set_rows) {
    return device != nullptr && device->aot != nullptr &&
        device->aot->find("flagos_rope_kv_store_f32_f16") != nullptr &&
        rope != nullptr && view != nullptr && set_rows != nullptr &&
        amd_supports_rope_neox(device, rope) &&
        view->op == GGML_OP_VIEW && view->src[0] == rope &&
        // The fused kernel reads the original RoPE allocation directly.  A
        // non-zero view offset (or a scheduler-rewritten view pointer) would
        // make the flattened view and the pointer passed to the kernel refer
        // to different elements, so keep that layout out of the capability
        // boundary until an offset is carried in the ABI.
        view->view_offs == 0 && view->data == rope->data &&
        view->ne[0] == rope->ne[0] * rope->ne[1] && view->ne[1] == rope->ne[2] &&
        view->ne[2] == 1 && view->ne[3] == 1 && ggml_is_contiguous(view) &&
        set_rows->src[0] == view &&
        !amd_tensor_data_overlaps(set_rows, rope->src[0]) &&
        !amd_tensor_data_overlaps(set_rows, rope->src[1]) &&
        amd_supports_set_rows(device, set_rows);
}

static bool amd_supports_rms_norm_mul_rope(
        const amd_device_context * device,
        const ggml_tensor * norm,
        const ggml_tensor * mul,
        const ggml_tensor * rope,
        const ggml_tensor * view = nullptr,
        const ggml_tensor * set_rows = nullptr) {
    const bool store = set_rows != nullptr;
    const char * kernel_name = store
        ? "flagos_rms_norm_mul_rope_kv_store_neox_f32_f16"
        : "flagos_rms_norm_mul_rope_neox_f32";
    if (device == nullptr || device->aot == nullptr || norm == nullptr || mul == nullptr || rope == nullptr ||
        device->aot->find(kernel_name) == nullptr ||
        norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE ||
        norm->src[0] == nullptr || mul->src[0] == nullptr || mul->src[1] == nullptr ||
        rope->src[0] != mul || norm->type != GGML_TYPE_F32 || norm->src[0]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || !ggml_are_same_shape(norm, norm->src[0]) ||
        !ggml_are_same_shape(mul, norm) || !ggml_are_same_shape(rope, mul) ||
        !ggml_is_contiguous(norm->src[0]) || !ggml_is_contiguous(norm) ||
        !ggml_is_contiguous(mul) || !ggml_is_contiguous(rope) ||
        norm->ne[0] <= 0 || norm->ne[0] > INT32_MAX || norm->ne[1] <= 0 ||
        norm->ne[1] > INT32_MAX || norm->ne[2] <= 0 || norm->ne[2] > INT32_MAX ||
        norm->ne[3] != 1 || !amd_supports_rope_neox(device, rope)) {
        return false;
    }
    const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] :
        (mul->src[1] == norm ? mul->src[0] : nullptr);
    if (weight == nullptr || weight->type != GGML_TYPE_F32 ||
        weight->ne[0] != norm->ne[0] || weight->ne[1] != 1 ||
        weight->ne[2] != 1 || weight->ne[3] != 1 || !ggml_is_contiguous(weight)) {
        return false;
    }
    const int32_t * params = static_cast<const int32_t *>(rope->op_params);
    const auto * metadata = device->aot->find(kernel_name);
    float eps = 0.0f;
    std::memcpy(&eps, norm->op_params, sizeof(eps));
    if (params[1] != norm->ne[0] || metadata == nullptr || metadata->block_size < norm->ne[0] ||
        !std::isfinite(eps) || eps < 0.0f || norm->src[0]->data == nullptr ||
        weight->data == nullptr || rope->src[1]->data == nullptr) {
        return false;
    }
    const ggml_tensor * output = store ? set_rows : rope;
    if (output == nullptr || output->data == nullptr ||
        amd_tensor_data_overlaps(output, norm->src[0]) ||
        amd_tensor_data_overlaps(output, weight) ||
        amd_tensor_data_overlaps(output, rope->src[1])) {
        return false;
    }
    if (!store) {
        return true;
    }
    return view != nullptr && amd_supports_rope_kv_store(device, rope, view, set_rows) &&
        set_rows->src[1]->data != nullptr && !amd_tensor_data_overlaps(set_rows, set_rows->src[1]);
}

static bool amd_supports_swiglu(const amd_device_context * device, const ggml_tensor * op) {
    return device != nullptr && device->aot != nullptr && op != nullptr &&
        op->op == GGML_OP_GLU && ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU &&
        op->src[0] != nullptr && op->src[1] != nullptr &&
        device->aot->find("flagos_swiglu_split_f32") != nullptr &&
        op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
        op->src[1]->type == GGML_TYPE_F32 && ggml_are_same_shape(op, op->src[0]) &&
        ggml_are_same_shape(op, op->src[1]) && ggml_is_contiguous(op) &&
        ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) &&
        ggml_nelements(op) > 0 && ggml_nelements(op) <= INT32_MAX &&
        amd_tensor_output_can_reuse_input(op, op->src[0]) &&
        amd_tensor_output_can_reuse_input(op, op->src[1]);
}

static bool amd_supports_flash_attn_decode(const amd_device_context * device, const ggml_tensor * op) {
    float scale = 0.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    if (!amd_fusion_enabled(device, "flash_attn_decode") || device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_FLASH_ATTN_EXT || device->aot->find("flagos_flash_attn_decode_f32_f16") == nullptr) {
        return false;
    }
    const auto * op_params_bytes = reinterpret_cast<const uint8_t *>(op->op_params);
    std::memcpy(&scale, op_params_bytes, sizeof(scale));
    std::memcpy(&max_bias, op_params_bytes + sizeof(float), sizeof(max_bias));
    std::memcpy(&logit_softcap, op_params_bytes + 2 * sizeof(float), sizeof(logit_softcap));
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * mask = op->src[3];
    return q && k && v && mask && op->src[4] == nullptr &&
        op->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 &&
        k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && mask->type == GGML_TYPE_F16 &&
        op->ne[0] == 128 && op->ne[1] > 0 && op->ne[1] <= INT32_MAX && op->ne[2] == 1 && op->ne[3] == 1 &&
        q->ne[0] == 128 && q->ne[1] == 1 && q->ne[2] == op->ne[1] && q->ne[3] == 1 &&
        k->ne[0] == 128 && k->ne[1] > 0 && k->ne[1] <= INT32_MAX && k->ne[2] > 0 && k->ne[2] <= INT32_MAX && k->ne[3] == 1 &&
        q->ne[2] % k->ne[2] == 0 && ggml_are_same_shape(k, v) &&
        mask->ne[0] == k->ne[1] && mask->ne[1] == 1 && mask->ne[2] == 1 && mask->ne[3] == 1 &&
        q->nb[0] == sizeof(float) && k->nb[0] == sizeof(ggml_fp16_t) && v->nb[0] == sizeof(ggml_fp16_t) &&
        k->nb[1] >= 256 * sizeof(ggml_fp16_t) && k->nb[2] == 128 * sizeof(ggml_fp16_t) &&
        v->nb[1] == k->nb[1] && v->nb[2] == k->nb[2] &&
        amd_tensor_element_offsets_fit_i32(q, sizeof(float)) &&
        amd_tensor_element_offsets_fit_i32(k, sizeof(ggml_fp16_t)) &&
        amd_tensor_element_offsets_fit_i32(v, sizeof(ggml_fp16_t)) &&
        amd_tensor_element_offsets_fit_i32(mask, sizeof(ggml_fp16_t)) &&
        amd_tensor_element_offsets_fit_i32(op, sizeof(float)) &&
        q->nb[1] / sizeof(float) <= INT32_MAX && q->nb[2] / sizeof(float) <= INT32_MAX &&
        k->nb[1] / sizeof(ggml_fp16_t) <= INT32_MAX && k->nb[2] / sizeof(ggml_fp16_t) <= INT32_MAX &&
        v->nb[1] / sizeof(ggml_fp16_t) <= INT32_MAX && v->nb[2] / sizeof(ggml_fp16_t) <= INT32_MAX &&
        op->nb[1] / sizeof(float) <= INT32_MAX && op->nb[2] / sizeof(float) <= INT32_MAX &&
        ggml_is_contiguous(mask) && ggml_is_contiguous(op) &&
        std::isfinite(scale) && max_bias == 0.0f && logit_softcap == 0.0f &&
        !amd_tensor_data_overlaps(op, q) && !amd_tensor_data_overlaps(op, k) &&
        !amd_tensor_data_overlaps(op, v) && !amd_tensor_data_overlaps(op, mask);
}

static bool amd_supports_flash_attn_prefill(const amd_device_context * device, const ggml_tensor * op) {
    float scale = 0.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    if (!amd_fusion_enabled(device, "flash_attn_prefill") || device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_FLASH_ATTN_EXT ||
        device->aot->find("flagos_flash_attn_prefill_f32_f16") == nullptr) {
        return false;
    }
    const auto * op_params_bytes = reinterpret_cast<const uint8_t *>(op->op_params);
    std::memcpy(&scale, op_params_bytes, sizeof(scale));
    std::memcpy(&max_bias, op_params_bytes + sizeof(float), sizeof(max_bias));
    std::memcpy(&logit_softcap, op_params_bytes + 2 * sizeof(float), sizeof(logit_softcap));
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * mask = op->src[3];
    // The first prefill implementation deliberately uses one contiguous
    // F16 mask plane shared by all heads/batches.  More general broadcast and
    // ALiBi variants remain provider capabilities that can be added later
    // without changing the common pattern ABI.
    return q && k && v && mask && op->src[4] == nullptr &&
        op->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 &&
        k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 &&
        mask->type == GGML_TYPE_F16 && q->ne[1] > 1 && q->ne[1] <= INT32_MAX &&
        op->ne[0] == 128 && q->ne[0] == 128 && k->ne[0] == 128 && v->ne[0] == 128 &&
        op->ne[1] == q->ne[2] && op->ne[2] == q->ne[1] && op->ne[3] == q->ne[3] &&
        q->ne[2] > 0 && q->ne[2] <= INT32_MAX && q->ne[3] == 1 &&
        k->ne[1] > 0 && k->ne[1] <= INT32_MAX && k->ne[2] > 0 && k->ne[2] <= INT32_MAX &&
        k->ne[3] == 1 && v->ne[1] == k->ne[1] && v->ne[2] == k->ne[2] && v->ne[3] == 1 &&
        q->ne[2] % k->ne[2] == 0 &&
        mask->ne[0] >= k->ne[1] && mask->ne[1] >= q->ne[1] &&
        mask->ne[2] == 1 && mask->ne[3] == 1 &&
        q->nb[0] == sizeof(float) && k->nb[0] == sizeof(ggml_fp16_t) &&
        v->nb[0] == sizeof(ggml_fp16_t) && mask->nb[0] == sizeof(ggml_fp16_t) &&
        op->nb[0] == sizeof(float) && q->nb[1] / sizeof(float) <= INT32_MAX &&
        q->nb[2] / sizeof(float) <= INT32_MAX && k->nb[1] / sizeof(ggml_fp16_t) <= INT32_MAX &&
        k->nb[2] / sizeof(ggml_fp16_t) <= INT32_MAX &&
        v->nb[1] / sizeof(ggml_fp16_t) <= INT32_MAX && v->nb[2] / sizeof(ggml_fp16_t) <= INT32_MAX &&
        mask->nb[1] / sizeof(ggml_fp16_t) <= INT32_MAX &&
        amd_tensor_element_offsets_fit_i32(q, sizeof(float)) &&
        amd_tensor_element_offsets_fit_i32(k, sizeof(ggml_fp16_t)) &&
        amd_tensor_element_offsets_fit_i32(v, sizeof(ggml_fp16_t)) &&
        amd_tensor_element_offsets_fit_i32(mask, sizeof(ggml_fp16_t)) &&
        amd_tensor_element_offsets_fit_i32(op, sizeof(float)) &&
        op->nb[1] / sizeof(float) <= INT32_MAX && op->nb[2] / sizeof(float) <= INT32_MAX &&
        ggml_is_contiguous(mask) && ggml_is_contiguous(op) &&
        std::isfinite(scale) && max_bias == 0.0f && logit_softcap == 0.0f &&
        !amd_tensor_data_overlaps(op, q) && !amd_tensor_data_overlaps(op, k) &&
        !amd_tensor_data_overlaps(op, v) && !amd_tensor_data_overlaps(op, mask);
}

static bool amd_supports_soft_max(const amd_device_context * device, const ggml_tensor * op) {
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_SOFT_MAX || op->src[0] == nullptr ||
        op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(op, op->src[0]) || !ggml_is_contiguous(op) ||
        !ggml_is_contiguous(op->src[0]) || op->ne[0] <= 0 || op->ne[0] > INT32_MAX ||
        op->ne[1] <= 0 || op->ne[2] <= 0 || op->ne[3] <= 0 ||
        ggml_nelements(op) > INT32_MAX || op->src[2] != nullptr) {
        return false;
    }
    // The current AMD Triton reduction is validated for power-of-two row
    // widths.  Non-power-of-two masked reductions can produce materially wrong
    // results on gfx1150 due to the backend's masked warp-reduction lowering;
    // keep those shapes on the reference scheduler until a dedicated kernel
    // variant is available.
    if ((op->ne[0] & (op->ne[0] - 1)) != 0) {
        return false;
    }
    float scale = 1.0f;
    float max_bias = 0.0f;
    std::memcpy(&scale, op->op_params, sizeof(scale));
    const auto * op_params_bytes = reinterpret_cast<const uint8_t *>(op->op_params);
    std::memcpy(&max_bias, op_params_bytes + sizeof(float), sizeof(max_bias));
    if (!std::isfinite(scale) || scale == 0.0f || max_bias != 0.0f) {
        return false;
    }
    const ggml_tensor * mask = op->src[1];
    const char * kernel = mask == nullptr
        ? "flagos_soft_max_unmasked_f32" : "flagos_soft_max_masked_f32_f16";
    if (device->aot->find(kernel) == nullptr) {
        return false;
    }
    if (mask != nullptr &&
        (mask->type != GGML_TYPE_F16 ||
         mask->ne[0] != op->ne[0] || mask->ne[1] != op->ne[1] ||
         mask->ne[2] != op->ne[2] || mask->ne[3] != op->ne[3] ||
         !ggml_is_contiguous(mask))) {
        return false;
    }
    const auto * metadata = device->aot->find(kernel);
    return metadata != nullptr && metadata->block_size >= op->ne[0] &&
        amd_tensor_output_can_reuse_input(op, op->src[0]) &&
        (mask == nullptr || !amd_tensor_data_overlaps(op, mask));
}

static bool amd_supports_rms_norm_mul(const amd_device_context * device,
                                      const ggml_tensor * norm,
                                      const ggml_tensor * mul) {
    if (device == nullptr || device->aot == nullptr || norm == nullptr || mul == nullptr ||
        device->aot->find("flagos_rms_norm_mul_f32") == nullptr ||
        norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || norm->src[0] == nullptr ||
        norm->type != GGML_TYPE_F32 || norm->src[0]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(norm, norm->src[0]) || !ggml_are_same_shape(mul, norm) ||
        norm->ne[0] <= 0 || norm->ne[0] > INT32_MAX || norm->ne[0] % 4 != 0 ||
        norm->ne[0] > 4096 || ggml_nelements(norm) > INT32_MAX ||
        !ggml_is_contiguous(norm) || !ggml_is_contiguous(norm->src[0]) || !ggml_is_contiguous(mul)) {
        return false;
    }
    const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] :
        (mul->src[1] == norm ? mul->src[0] : nullptr);
    if (weight == nullptr || weight->type != GGML_TYPE_F32 ||
        weight->ne[0] != norm->ne[0] || weight->ne[1] != 1 ||
        weight->ne[2] != 1 || weight->ne[3] != 1 ||
        !ggml_is_contiguous(weight) || norm->data == nullptr || mul->data == nullptr ||
        norm->src[0]->data == nullptr || weight->data == nullptr) {
        return false;
    }
    if (amd_tensor_data_overlaps(norm, norm->src[0]) || amd_tensor_data_overlaps(norm, weight) ||
        amd_tensor_data_overlaps(mul, norm->src[0]) || amd_tensor_data_overlaps(mul, weight)) {
        return false;
    }
    float eps = 0.0f;
    std::memcpy(&eps, norm->op_params, sizeof(eps));
    if (!std::isfinite(eps) || eps < 0.0f) {
        return false;
    }
    const auto * metadata = device->aot->find("flagos_rms_norm_mul_f32");
    return metadata->block_size >= norm->ne[0];
}

static bool amd_supports_add_rms_norm_mul(const amd_device_context * device,
                                          const ggml_tensor * add,
                                          const ggml_tensor * norm,
                                          const ggml_tensor * mul);

static bool amd_supports_add_rms_norm_mul(const amd_device_context * device,
                                          const ggml_tensor * add,
                                          const ggml_tensor * norm,
                                          const ggml_tensor * mul) {
    if (device == nullptr || device->aot == nullptr || add == nullptr ||
        !amd_supports_rms_norm_mul(device, norm, mul) ||
        device->aot->find("flagos_add_rms_norm_mul_f32") == nullptr ||
        add->op != GGML_OP_ADD || add->type != GGML_TYPE_F32 ||
        add->src[0] == nullptr || add->src[1] == nullptr ||
        norm->src[0] != add ||
        !ggml_are_same_shape(add, norm) || !ggml_is_contiguous(add) ||
        add->src[0]->type != GGML_TYPE_F32 || add->src[1]->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(add->src[0], norm) || !ggml_are_same_shape(add->src[1], norm) ||
        !ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous(add->src[1]) ||
        add->src[0]->data == nullptr || add->src[1]->data == nullptr) {
        return false;
    }
    if (amd_tensor_data_overlaps(norm, add->src[0]) ||
        amd_tensor_data_overlaps(norm, add->src[1]) ||
        amd_tensor_data_overlaps(mul, add->src[0]) ||
        amd_tensor_data_overlaps(mul, add->src[1])) {
        return false;
    }
    const auto * metadata = device->aot->find("flagos_add_rms_norm_mul_f32");
    return metadata->block_size >= norm->ne[0];
}

static const char * amd_rms_norm_mul_inplace_kernel(
        const amd_device_context * device, int64_t n_cols) {
    if (device == nullptr || device->aot == nullptr || n_cols <= 0) {
        return nullptr;
    }
    if (const auto * narrow = device->aot->find(
            "flagos_rms_norm_mul_inplace_f32_narrow")) {
        if (narrow->exact_block_size && narrow->block_size == n_cols) {
            return "flagos_rms_norm_mul_inplace_f32_narrow";
        }
    }
    const auto * generic = device->aot->find("flagos_rms_norm_mul_inplace_f32");
    return generic != nullptr && generic->block_size >= n_cols
        ? "flagos_rms_norm_mul_inplace_f32" : nullptr;
}

static bool amd_supports_rms_norm_mul_inplace(const amd_device_context * device,
                                              const ggml_tensor * norm,
                                              const ggml_tensor * mul) {
    const ggml_tensor * weight = mul != nullptr && mul->src[0] == norm ? mul->src[1] :
        (mul != nullptr ? mul->src[0] : nullptr);
    return device != nullptr && device->aot != nullptr && norm != nullptr && mul != nullptr &&
        norm->data != nullptr && norm->data == mul->data &&
        amd_rms_norm_mul_inplace_kernel(device, norm->ne[0]) != nullptr &&
        amd_supports_rms_norm_mul(device, norm, mul) &&
        norm->src[0]->data != norm->data && weight != nullptr && weight->data != norm->data;
}

static bool amd_supports_add_rms_norm_mul_inplace(const amd_device_context * device,
                                                  const ggml_tensor * add,
                                                  const ggml_tensor * norm,
                                                  const ggml_tensor * mul) {
    if (device == nullptr || add == nullptr || norm == nullptr || mul == nullptr ||
        norm->data == nullptr || norm->data != mul->data ||
        device->aot == nullptr || device->aot->find("flagos_add_rms_norm_mul_inplace_f32") == nullptr ||
        !amd_supports_add_rms_norm_mul(device, add, norm, mul) ||
        add->src[0]->data == norm->data || add->src[1]->data == norm->data) {
        return false;
    }
    const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    return weight != nullptr && weight->data != norm->data;
}

static const char * amd_add_rms_norm_mul_residual_kernel(
        const amd_device_context * device, int64_t n_cols) {
    if (device == nullptr || device->aot == nullptr || n_cols <= 0) {
        return nullptr;
    }
    const char * best_name = nullptr;
    int best_capacity = INT_MAX;
    for (const char * name : {
            "flagos_add_rms_norm_mul_residual_f32_narrow",
            "flagos_add_rms_norm_mul_residual_f32" }) {
        const auto * metadata = device->aot->find(name);
        if (metadata != nullptr && metadata->block_size >= n_cols &&
            metadata->block_size < best_capacity) {
            best_name = name;
            best_capacity = metadata->block_size;
        }
    }
    return best_name;
}

static bool amd_supports_add_rms_norm_mul_residual(const amd_device_context * device,
                                                   const ggml_tensor * add,
                                                   const ggml_tensor * norm,
                                                   const ggml_tensor * mul) {
    if (device == nullptr || device->aot == nullptr || add == nullptr || norm == nullptr || mul == nullptr ||
        add->op != GGML_OP_ADD || norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL ||
        add->src[0] == nullptr || add->src[1] == nullptr || norm->src[0] != add ||
        add->type != GGML_TYPE_F32 || norm->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        add->src[0]->type != GGML_TYPE_F32 || add->src[1]->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(add, add->src[0]) || !ggml_are_same_shape(add, add->src[1]) ||
        !ggml_are_same_shape(norm, add) || !ggml_are_same_shape(mul, norm) ||
        !ggml_is_contiguous(add) || !ggml_is_contiguous(norm) || !ggml_is_contiguous(mul) ||
        !ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous(add->src[1]) ||
        add->ne[0] <= 0 || add->ne[0] > 4096 || add->ne[0] % 4 != 0 ||
        ggml_nelements(add) > INT32_MAX || add->data == nullptr || mul->data == nullptr ||
        add->src[0]->data == nullptr || add->src[1]->data == nullptr) {
        return false;
    }
    const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] :
        (mul->src[1] == norm ? mul->src[0] : nullptr);
    float eps = 0.0f;
    std::memcpy(&eps, norm->op_params, sizeof(eps));
    return weight != nullptr && weight->type == GGML_TYPE_F32 &&
        weight->ne[0] == norm->ne[0] && weight->ne[1] == 1 &&
        weight->ne[2] == 1 && weight->ne[3] == 1 && ggml_is_contiguous(weight) &&
        weight->data != nullptr &&
        amd_add_rms_norm_mul_residual_kernel(device, norm->ne[0]) != nullptr &&
        std::isfinite(eps) && eps >= 0.0f &&
        !amd_tensor_data_overlaps(add, mul) &&
        !amd_tensor_data_overlaps(add, weight) &&
        !amd_tensor_data_overlaps(mul, weight) &&
        amd_tensor_output_can_reuse_input(add, add->src[0]) &&
        amd_tensor_output_can_reuse_input(add, add->src[1]) &&
        amd_tensor_output_can_reuse_input(mul, add->src[0]) &&
        amd_tensor_output_can_reuse_input(mul, add->src[1]);
}

static flagos_lowering_choice amd_query_fusion(
        void * user_data, const ggml_cgraph * cgraph, const flagos_pattern_candidate & candidate) {
    auto * context = static_cast<amd_backend_context *>(user_data);
    flagos_lowering_choice choice;
    if (context == nullptr || cgraph == nullptr || candidate.node_indices.empty()) {
        return choice;
    }
    if (candidate.id == flagos_pattern_id::rms_norm_mul_rope && candidate.node_indices.size() == 3) {
        const ggml_tensor * norm = cgraph->nodes[candidate.node_indices[0]];
        const ggml_tensor * mul = cgraph->nodes[candidate.node_indices[1]];
        const ggml_tensor * rope = cgraph->nodes[candidate.node_indices[2]];
        if (amd_candidate_outputs_are(candidate, { 2 }) &&
            amd_fusion_enabled(context->device, "rms_norm_mul_rope") &&
            amd_supports_rms_norm_mul_rope(context->device, norm, mul, rope)) {
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = 8;
        }
    } else if (candidate.id == flagos_pattern_id::rms_norm_mul_rope_kv_store &&
               candidate.node_indices.size() == 5) {
        const ggml_tensor * norm = cgraph->nodes[candidate.node_indices[0]];
        const ggml_tensor * mul = cgraph->nodes[candidate.node_indices[1]];
        const ggml_tensor * rope = cgraph->nodes[candidate.node_indices[2]];
        const ggml_tensor * view = cgraph->nodes[candidate.node_indices[3]];
        const ggml_tensor * set_rows = cgraph->nodes[candidate.node_indices[4]];
        if (amd_candidate_outputs_are(candidate, { 4 }) &&
            amd_fusion_enabled(context->device, "rms_norm_mul_rope") &&
            amd_supports_rms_norm_mul_rope(context->device, norm, mul, rope, view, set_rows)) {
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = 9;
        }
    } else if (candidate.id == flagos_pattern_id::rms_norm_mul && candidate.node_indices.size() == 2) {
        const ggml_tensor * norm = cgraph->nodes[candidate.node_indices[0]];
        const ggml_tensor * mul = cgraph->nodes[candidate.node_indices[1]];
        const bool terminal_only = amd_candidate_outputs_are(candidate, { 1 });
        const bool norm_and_terminal = amd_candidate_outputs_are(candidate, { 0, 1 });
        if (amd_fusion_enabled(context->device, "rms_norm_mul") &&
            ((norm_and_terminal && !amd_tensor_data_overlaps(norm, mul) &&
              amd_supports_rms_norm_mul(context->device, norm, mul)) ||
             (terminal_only &&
              ((amd_supports_rms_norm_mul(context->device, norm, mul) &&
                !amd_tensor_data_overlaps(norm, mul)) ||
               amd_supports_rms_norm_mul_inplace(context->device, norm, mul))))) {
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = 1;
        }
    } else if (candidate.id == flagos_pattern_id::add_rms_norm_mul && candidate.node_indices.size() == 3) {
        const ggml_tensor * add = cgraph->nodes[candidate.node_indices[0]];
        const ggml_tensor * norm = cgraph->nodes[candidate.node_indices[1]];
        const ggml_tensor * mul = cgraph->nodes[candidate.node_indices[2]];
        const bool terminal_only = amd_candidate_outputs_are(candidate, { 2 });
        const bool norm_and_terminal = amd_candidate_outputs_are(candidate, { 1, 2 });
        const bool residual_and_terminal = amd_candidate_outputs_are(candidate, { 0, 2 });
        if (amd_fusion_enabled(context->device, "add_rms_norm_mul") &&
            ((residual_and_terminal &&
              amd_supports_add_rms_norm_mul_residual(context->device, add, norm, mul)) ||
             (norm_and_terminal && !amd_tensor_data_overlaps(norm, mul) &&
              amd_supports_add_rms_norm_mul(context->device, add, norm, mul)) ||
             (terminal_only &&
              ((amd_supports_add_rms_norm_mul(context->device, add, norm, mul) &&
                !amd_tensor_data_overlaps(norm, mul)) ||
               amd_supports_add_rms_norm_mul_inplace(context->device, add, norm, mul))))) {
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = residual_and_terminal ? 10 : 2;
        }
    } else if (candidate.id == flagos_pattern_id::ssm_conv_silu &&
               candidate.node_indices.size() == 2) {
        const ggml_tensor * conv = cgraph->nodes[candidate.node_indices[0]];
        const ggml_tensor * activation = cgraph->nodes[candidate.node_indices[1]];
        if (amd_candidate_outputs_are(candidate, { 1 }) &&
            amd_fusion_enabled(context->device, "ssm_conv_silu") &&
            amd_supports_ssm_conv_silu(context->device, conv, activation)) {
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = 15;
        }
    } else if (candidate.id == flagos_pattern_id::attention_output_gate &&
               candidate.node_indices.size() == 2) {
        const ggml_tensor * activation = cgraph->nodes[candidate.node_indices[0]];
        const ggml_tensor * mul = cgraph->nodes[candidate.node_indices[1]];
        if (amd_candidate_outputs_are(candidate, { 1 }) &&
            amd_fusion_enabled(context->device, "attention_output_gate") &&
            amd_supports_attention_output_gate(context->device, activation, mul)) {
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = 16;
        }
    } else if (candidate.id == flagos_pattern_id::rope_kv_store && candidate.node_indices.size() == 3) {
        const ggml_tensor * rope = cgraph->nodes[candidate.node_indices[0]];
        const ggml_tensor * view = cgraph->nodes[candidate.node_indices[1]];
        const ggml_tensor * set_rows = cgraph->nodes[candidate.node_indices[2]];
        if (amd_candidate_outputs_are(candidate, { 2 }) && amd_fusion_enabled(context->device, "rope_kv_store") &&
            amd_supports_rope_kv_store(context->device, rope, view, set_rows)) {
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = 4;
        }
    } else if ((candidate.id == flagos_pattern_id::gated_delta_net_decode ||
                candidate.id == flagos_pattern_id::gated_delta_net_prefill) &&
               candidate.node_indices.size() >= 3) {
        const ggml_tensor * gdn = cgraph->nodes[candidate.node_indices.front()];
        const ggml_tensor * copy = cgraph->nodes[candidate.node_indices.back()];
        const ggml_tensor * snapshot_view = copy->src[0];
        if (amd_fusion_enabled(context->device, "gated_delta_net_cache")) {
            const bool cache_only_outputs =
                amd_gated_delta_net_cache_only_outputs_are_safe(
                    cgraph, candidate, gdn, copy);
            const bool cache_only_decode = cache_only_outputs &&
                amd_supports_gated_delta_net_cache(
                    context->device, gdn, snapshot_view, copy,
                    "flagos_gated_delta_net_scalar_f32_cache_only_decode");
            const bool cache_only = cache_only_decode ||
                (cache_only_outputs && amd_supports_gated_delta_net_cache(
                    context->device, gdn, snapshot_view, copy,
                    "flagos_gated_delta_net_scalar_f32_cache_only"));
            if (cache_only || amd_supports_gated_delta_net_cache(
                context->device, gdn, snapshot_view, copy,
                "flagos_gated_delta_net_scalar_f32_cache")) {
                choice.supported = true;
                choice.capture_safe = true;
                choice.implementation_id = cache_only_decode ? 14 : cache_only ? 13 : 12;
            }
        }
    } else if (candidate.id == flagos_pattern_id::flash_attn_decode && candidate.node_indices.size() == 1) {
        const ggml_tensor * attention = cgraph->nodes[candidate.node_indices[0]];
        if (amd_candidate_outputs_are(candidate, { 0 }) && amd_fusion_enabled(context->device, "flash_attn_decode") &&
            amd_supports_flash_attn_decode(context->device, attention)) {
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = 3;
        }
    } else if (candidate.id == flagos_pattern_id::flash_attn_prefill && candidate.node_indices.size() == 1) {
        const ggml_tensor * attention = cgraph->nodes[candidate.node_indices[0]];
        if (amd_candidate_outputs_are(candidate, { 0 }) && amd_fusion_enabled(context->device, "flash_attn_prefill") &&
            amd_supports_flash_attn_prefill(context->device, attention)) {
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = 5;
        }
    } else if (candidate.id == flagos_pattern_id::ffn_swiglu_down && candidate.node_indices.size() == 4) {
        const ggml_tensor * first_projection = cgraph->nodes[candidate.node_indices[0]];
        const ggml_tensor * second_projection = cgraph->nodes[candidate.node_indices[1]];
        const ggml_tensor * glu = cgraph->nodes[candidate.node_indices[2]];
        const ggml_tensor * down_projection = cgraph->nodes[candidate.node_indices[3]];
        flagos_quantized_matmul_signature first_signature;
        flagos_quantized_matmul_signature second_signature;
        flagos_quantized_matmul_signature down_signature;
        const bool described =
            flagos_describe_quantized_matmul(first_projection, &first_signature) &&
            flagos_describe_quantized_matmul(second_projection, &second_signature) &&
            flagos_describe_quantized_matmul(down_projection, &down_signature);
        if (amd_candidate_outputs_are(candidate, { 3 }) &&
            amd_supports_ffn_swiglu_down_f16(
                context->device, first_projection, second_projection, glu, down_projection) &&
            described) {
            const ggml_tensor * projections[] = {
                first_projection, second_projection, down_projection,
            };
            const flagos_quantized_matmul_signature signatures[] = {
                first_signature, second_signature, down_signature,
            };
            if (!amd_dequant_cache_ready(context, projections, signatures, 3)) {
                return choice;
            }
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = 11;
        }
    } else if (candidate.id == flagos_pattern_id::ffn_swiglu && candidate.node_indices.size() == 3) {
        const ggml_tensor * first_projection = cgraph->nodes[candidate.node_indices[0]];
        const ggml_tensor * second_projection = cgraph->nodes[candidate.node_indices[1]];
        const ggml_tensor * glu = cgraph->nodes[candidate.node_indices[2]];
        flagos_quantized_matmul_signature first_signature;
        flagos_quantized_matmul_signature second_signature;
        const bool described =
            flagos_describe_quantized_matmul(first_projection, &first_signature) &&
            flagos_describe_quantized_matmul(second_projection, &second_signature);
        if (amd_candidate_outputs_are(candidate, { 2 }) && amd_supports_ffn_swiglu_q4_decode(
                context->device, first_projection, second_projection, glu)) {
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = 7;
        } else if (amd_candidate_outputs_are(candidate, { 2 }) &&
            amd_supports_ffn_swiglu_f16(context->device, first_projection, second_projection, glu) &&
            described) {
            const ggml_tensor * projections[] = {
                first_projection, second_projection,
            };
            const flagos_quantized_matmul_signature signatures[] = {
                first_signature, second_signature,
            };
            if (!amd_dequant_cache_ready(context, projections, signatures, 2)) {
                return choice;
            }
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = 6;
        }
    }
    return choice;
}

static bool amd_execute_fusion(void * user_data, ggml_cgraph * cgraph, const flagos_plan_step & step) {
    auto * context = static_cast<amd_backend_context *>(user_data);
    if (context == nullptr || !amd_candidate_bindings_ready(cgraph, step.candidate)) {
        return false;
    }
    const bool gdn_cache =
        (step.candidate.id == flagos_pattern_id::gated_delta_net_decode ||
         step.candidate.id == flagos_pattern_id::gated_delta_net_prefill) &&
        step.candidate.node_indices.size() >= 3;
    if (gdn_cache) {
        ggml_tensor * gdn = cgraph->nodes[step.candidate.node_indices.front()];
        ggml_tensor * copy = cgraph->nodes[step.candidate.node_indices.back()];
        ggml_tensor * snapshot_view = copy->src[0];
        if (step.implementation_id != 12 && step.implementation_id != 13 &&
            step.implementation_id != 14) {
            return false;
        }
        const bool cache_only = step.implementation_id != 12;
        const bool cache_only_decode = step.implementation_id == 14;
        const char * kernel_name = cache_only_decode
            ? "flagos_gated_delta_net_scalar_f32_cache_only_decode"
            : cache_only ? "flagos_gated_delta_net_scalar_f32_cache_only"
                         : "flagos_gated_delta_net_scalar_f32_cache";
        if (!amd_supports_gated_delta_net_cache(
                context->device, gdn, snapshot_view, copy, kernel_name)) {
            return false;
        }
        if (cache_only && !amd_gated_delta_net_cache_only_outputs_are_safe(
                cgraph, step.candidate, gdn, copy)) {
            return false;
        }
        const ggml_tensor * q = gdn->src[0];
        const ggml_tensor * v = gdn->src[2];
        const ggml_tensor * beta = gdn->src[4];
        const ggml_tensor * destination = copy->src[1];
        const auto * metadata = context->device->aot->find(kernel_name);
        int state_size = static_cast<int>(v->ne[0]);
        int n_heads = static_cast<int>(v->ne[1]);
        int n_tokens = static_cast<int>(v->ne[2]);
        int n_seqs = static_cast<int>(v->ne[3]);
        int sq1 = static_cast<int>(q->nb[1] / sizeof(float));
        int sq2 = static_cast<int>(q->nb[2] / sizeof(float));
        int sq3 = static_cast<int>(q->nb[3] / sizeof(float));
        int sv1 = static_cast<int>(v->nb[1] / sizeof(float));
        int sv2 = static_cast<int>(v->nb[2] / sizeof(float));
        int sv3 = static_cast<int>(v->nb[3] / sizeof(float));
        int sb1 = static_cast<int>(beta->nb[1] / sizeof(float));
        int sb2 = static_cast<int>(beta->nb[2] / sizeof(float));
        int sb3 = static_cast<int>(beta->nb[3] / sizeof(float));
        int q_heads = static_cast<int>(q->ne[1]);
        int q_seq_ratio = static_cast<int>(v->ne[3] / q->ne[3]);
        int snapshot_count = ggml_get_op_params_i32(gdn, 0);
        int cache_slot_stride = std::min(n_tokens, snapshot_count) > 1
            ? static_cast<int>(destination->nb[2] / sizeof(float)) : 0;
        float scale = 1.0f / std::sqrt(static_cast<float>(state_size));
        void * q_data = gdn->src[0]->data;
        void * k_data = gdn->src[1]->data;
        void * v_data = gdn->src[2]->data;
        void * gate_data = gdn->src[3]->data;
        void * beta_data = gdn->src[4]->data;
        void * state_data = gdn->src[5]->data;
        void * output_data = gdn->data;
        void * cache_data = destination->data;
        flagos_amd::kernel_arguments arguments = {
            &q_data, &k_data, &v_data, &gate_data, &beta_data, &state_data,
            &output_data, &cache_data,
            &state_size, &n_heads, &n_tokens, &n_seqs,
            &sq1, &sq2, &sq3, &sv1, &sv2, &sv3, &sb1, &sb2, &sb3,
            &q_heads, &q_seq_ratio, &snapshot_count, &scale, &cache_slot_stride,
        };
        const unsigned int grid_z = static_cast<unsigned int>(
            (static_cast<uint64_t>(state_size) + metadata->tile_n - 1) /
            metadata->tile_n);
        const bool launched = context->device->aot->launch(
            kernel_name, context->stream,
            static_cast<unsigned int>(n_heads),
            static_cast<unsigned int>(n_seqs), grid_z, arguments);
        if (launched) {
            context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_steps.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_gated_delta_net_cache.fetch_add(1, std::memory_order_relaxed);
            if (cache_only) {
                context->stats.fusion_gated_delta_net_cache_only.fetch_add(1, std::memory_order_relaxed);
            }
            if (cache_only_decode) {
                context->stats.fusion_gated_delta_net_cache_only_decode.fetch_add(
                    1, std::memory_order_relaxed);
            }
            amd_trace_op(context, gdn,
                cache_only_decode ? "gated_delta_net_cache_only_decode" :
                cache_only ? "gated_delta_net_cache_only" : "gated_delta_net_cache");
        }
        return launched;
    }
    if (step.candidate.id == flagos_pattern_id::ssm_conv_silu &&
        step.candidate.node_indices.size() == 2) {
        if (step.implementation_id != 15 ||
            !amd_candidate_outputs_are(step.candidate, { 1 })) {
            return false;
        }
        ggml_tensor * conv = cgraph->nodes[step.candidate.node_indices[0]];
        ggml_tensor * activation = cgraph->nodes[step.candidate.node_indices[1]];
        if (!amd_supports_ssm_conv_silu(context->device, conv, activation)) {
            return false;
        }
        const bool launched = amd_launch_ssm_conv(
            context, conv, activation, "flagos_ssm_conv_silu_f32");
        if (launched) {
            context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_steps.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_ssm_conv_silu.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(context, activation, "ssm_conv_silu");
        }
        return launched;
    }
    if (step.candidate.id == flagos_pattern_id::attention_output_gate &&
        step.candidate.node_indices.size() == 2) {
        if (step.implementation_id != 16 ||
            !amd_candidate_outputs_are(step.candidate, { 1 })) {
            return false;
        }
        ggml_tensor * activation = cgraph->nodes[step.candidate.node_indices[0]];
        ggml_tensor * mul = cgraph->nodes[step.candidate.node_indices[1]];
        const ggml_tensor * other = amd_attention_output_gate_other(activation, mul);
        if (!amd_supports_attention_output_gate(context->device, activation, mul) ||
            other == nullptr) {
            return false;
        }
        const auto * metadata = context->device->aot->find("flagos_silu_mul_f32");
        int n = static_cast<int>(ggml_nelements(mul));
        void * input_data = activation->src[0]->data;
        void * other_data = other->data;
        void * output_data = mul->data;
        flagos_amd::kernel_arguments arguments = {
            &input_data, &other_data, &output_data, &n,
        };
        const unsigned int grid_x = static_cast<unsigned int>(
            (static_cast<uint64_t>(n) + metadata->block_size - 1) /
            metadata->block_size);
        const bool launched = context->device->aot->launch(
            "flagos_silu_mul_f32", context->stream, grid_x, 1, 1, arguments);
        if (launched) {
            context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_steps.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_attention_output_gate.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(context, mul, "attention_output_gate");
        }
        return launched;
    }
    const bool rms_rope = step.candidate.id == flagos_pattern_id::rms_norm_mul_rope &&
        step.candidate.node_indices.size() == 3;
    const bool rms_rope_store =
        step.candidate.id == flagos_pattern_id::rms_norm_mul_rope_kv_store &&
        step.candidate.node_indices.size() == 5;
    if (rms_rope || rms_rope_store) {
        if ((rms_rope && !amd_candidate_outputs_are(step.candidate, { 2 })) ||
            (rms_rope_store && !amd_candidate_outputs_are(step.candidate, { 4 }))) {
            return false;
        }
        ggml_tensor * norm = cgraph->nodes[step.candidate.node_indices[0]];
        ggml_tensor * mul = cgraph->nodes[step.candidate.node_indices[1]];
        ggml_tensor * rope = cgraph->nodes[step.candidate.node_indices[2]];
        ggml_tensor * view = rms_rope_store
            ? cgraph->nodes[step.candidate.node_indices[3]] : nullptr;
        ggml_tensor * set_rows = rms_rope_store
            ? cgraph->nodes[step.candidate.node_indices[4]] : nullptr;
        if (!amd_supports_rms_norm_mul_rope(
                context->device, norm, mul, rope, view, set_rows)) {
            return false;
        }
        const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] : mul->src[0];
        const int32_t * params = static_cast<const int32_t *>(rope->op_params);
        int n_cols = static_cast<int>(norm->ne[0]);
        int n_heads = static_cast<int>(norm->ne[1]);
        int n_tokens = static_cast<int>(norm->ne[2]);
        int n_dims = params[1];
        float eps = 0.0f;
        float freq_base = 0.0f;
        float freq_scale = 0.0f;
        std::memcpy(&eps, norm->op_params, sizeof(eps));
        std::memcpy(&freq_base, params + 5, sizeof(freq_base));
        std::memcpy(&freq_scale, params + 6, sizeof(freq_scale));
        void * x_data = norm->src[0]->data;
        void * weight_data = weight->data;
        void * positions_data = rope->src[1]->data;
        void * output_data = rms_rope_store ? set_rows->data : rope->data;
        const char * kernel_name = rms_rope_store
            ? "flagos_rms_norm_mul_rope_kv_store_neox_f32_f16"
            : "flagos_rms_norm_mul_rope_neox_f32";
        flagos_amd::kernel_arguments arguments = {
            &x_data, &weight_data, &positions_data,
        };
        int n_dst_rows = 0;
        void * row_index_data = nullptr;
        if (rms_rope_store) {
            row_index_data = set_rows->src[1]->data;
            n_dst_rows = static_cast<int>(set_rows->ne[1]);
            arguments.push_back(&row_index_data);
        }
        arguments.push_back(&output_data);
        arguments.push_back(&n_cols);
        arguments.push_back(&n_heads);
        arguments.push_back(&n_tokens);
        if (rms_rope_store) {
            arguments.push_back(&n_dst_rows);
        }
        arguments.push_back(&n_dims);
        arguments.push_back(&eps);
        arguments.push_back(&freq_base);
        arguments.push_back(&freq_scale);
        const unsigned int grid_x = static_cast<unsigned int>(
            static_cast<uint64_t>(n_heads) * static_cast<uint64_t>(n_tokens));
        const bool launched = context->device->aot->launch(
            kernel_name, context->stream, grid_x, 1, 1, arguments);
        if (launched) {
            context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_steps.fetch_add(1, std::memory_order_relaxed);
            if (rms_rope_store) {
                context->stats.fusion_rms_norm_mul_rope_kv_store.fetch_add(1, std::memory_order_relaxed);
                context->stats.rope_kv_store.fetch_add(1, std::memory_order_relaxed);
                amd_trace_op(context, set_rows, "rms_norm_mul_rope_kv_store");
            } else {
                context->stats.fusion_rms_norm_mul_rope.fetch_add(1, std::memory_order_relaxed);
                context->stats.rope.fetch_add(1, std::memory_order_relaxed);
                amd_trace_op(context, rope, "rms_norm_mul_rope");
            }
        }
        return launched;
    }
    if (step.candidate.id == flagos_pattern_id::ffn_swiglu_down &&
        step.candidate.node_indices.size() == 4) {
        if (!amd_candidate_outputs_are(step.candidate, { 3 })) {
            return false;
        }
        ggml_tensor * first_projection = cgraph->nodes[step.candidate.node_indices[0]];
        ggml_tensor * second_projection = cgraph->nodes[step.candidate.node_indices[1]];
        ggml_tensor * glu = cgraph->nodes[step.candidate.node_indices[2]];
        ggml_tensor * down_projection = cgraph->nodes[step.candidate.node_indices[3]];
        ggml_tensor * gate_projection = glu->src[0] == first_projection
            ? first_projection : second_projection;
        ggml_tensor * up_projection = glu->src[0] == first_projection
            ? second_projection : first_projection;
        if (!amd_supports_ffn_swiglu_down_f16(
                context->device, gate_projection, up_projection, glu, down_projection)) {
            return false;
        }

        flagos_quantized_matmul_signature gate_signature;
        flagos_quantized_matmul_signature up_signature;
        flagos_quantized_matmul_signature down_signature;
        if (!flagos_describe_quantized_matmul(gate_projection, &gate_signature) ||
            !flagos_describe_quantized_matmul(up_projection, &up_signature) ||
            !flagos_describe_quantized_matmul(down_projection, &down_signature)) {
            return false;
        }
        void * gate_weights = amd_get_dequantized_weight(
            context, gate_projection->src[0], gate_signature);
        void * up_weights = amd_get_dequantized_weight(
            context, up_projection->src[0], up_signature);
        void * down_weights = amd_get_dequantized_weight(
            context, down_projection->src[0], down_signature);
        void * activation_data = gate_projection->src[1] == nullptr
            ? nullptr : gate_projection->src[1]->data;
        void * scratch_data = glu->data;
        void * output_data = down_projection->data;
        if (gate_weights == nullptr || up_weights == nullptr || down_weights == nullptr ||
            activation_data == nullptr || scratch_data == nullptr || output_data == nullptr) {
            return false;
        }

        const char * ffn_kernel = "flagos_ffn_swiglu_f16_f16_grouped";
        const auto * ffn_metadata = context->device->aot->find(ffn_kernel);
        if (ffn_metadata == nullptr || ffn_metadata->tile_m <= 0 || ffn_metadata->tile_n <= 0) {
            return false;
        }
        int ffn_k = static_cast<int>(gate_signature.k);
        int ffn_rows = static_cast<int>(gate_signature.rows);
        int columns = static_cast<int>(gate_signature.columns);
        flagos_amd::kernel_arguments ffn_arguments = {
            &gate_weights, &up_weights, &activation_data, &scratch_data,
            &ffn_k, &ffn_rows, &columns,
        };
        const unsigned int ffn_tile_m = static_cast<unsigned int>(ffn_metadata->tile_m);
        const unsigned int ffn_tile_n = static_cast<unsigned int>(ffn_metadata->tile_n);
        const unsigned int ffn_grid =
            ((static_cast<unsigned int>(ffn_rows) + ffn_tile_m - 1) / ffn_tile_m) *
            ((static_cast<unsigned int>(columns) + ffn_tile_n - 1) / ffn_tile_n);
        if (!context->device->aot->launch(
                ffn_kernel, context->stream, ffn_grid, 1, 1, ffn_arguments)) {
            return false;
        }
        context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);

        const char * down_kernel = "flagos_mul_mat_f16_f16_grouped";
        const auto * down_metadata = context->device->aot->find(down_kernel);
        if (down_metadata == nullptr || down_metadata->tile_m <= 0 || down_metadata->tile_n <= 0) {
            return false;
        }
        int down_k = static_cast<int>(down_signature.k);
        int down_rows = static_cast<int>(down_signature.rows);
        flagos_amd::kernel_arguments down_arguments = {
            &down_weights, &scratch_data, &output_data,
            &down_k, &down_rows, &columns,
        };
        const unsigned int down_tile_m = static_cast<unsigned int>(down_metadata->tile_m);
        const unsigned int down_tile_n = static_cast<unsigned int>(down_metadata->tile_n);
        const unsigned int down_grid =
            ((static_cast<unsigned int>(down_rows) + down_tile_m - 1) / down_tile_m) *
            ((static_cast<unsigned int>(columns) + down_tile_n - 1) / down_tile_n);
        if (!context->device->aot->launch(
                down_kernel, context->stream, down_grid, 1, 1, down_arguments)) {
            return false;
        }
        context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
        context->stats.fusion_steps.fetch_add(1, std::memory_order_relaxed);
        context->stats.fusion_ffn_swiglu.fetch_add(1, std::memory_order_relaxed);
        context->stats.fusion_ffn_swiglu_down.fetch_add(1, std::memory_order_relaxed);
        context->stats.f16_matmul_batched.fetch_add(1, std::memory_order_relaxed);
        amd_trace_op(context, down_projection, "ffn_swiglu_down_f16");
        return true;
    }
    if (step.candidate.id == flagos_pattern_id::ffn_swiglu && step.candidate.node_indices.size() == 3) {
        ggml_tensor * first_projection = cgraph->nodes[step.candidate.node_indices[0]];
        ggml_tensor * second_projection = cgraph->nodes[step.candidate.node_indices[1]];
        ggml_tensor * glu = cgraph->nodes[step.candidate.node_indices[2]];
        // GGML permits the two GLU inputs to be presented in either order.
        // The first input is the SiLU/gate projection; do not infer that role
        // from graph node order, which is only a topological property.
        ggml_tensor * gate_projection = glu->src[0] == first_projection
            ? first_projection : second_projection;
        ggml_tensor * up_projection = glu->src[0] == first_projection
            ? second_projection : first_projection;
        if (amd_supports_ffn_swiglu_q4_decode(
                context->device, gate_projection, up_projection, glu)) {
            flagos_quantized_matmul_signature gate_signature;
            flagos_quantized_matmul_signature up_signature;
            if (!flagos_describe_quantized_matmul(gate_projection, &gate_signature) ||
                !flagos_describe_quantized_matmul(up_projection, &up_signature) ||
                gate_projection->src[0] == nullptr || up_projection->src[0] == nullptr ||
                gate_projection->src[1] == nullptr || glu->data == nullptr) {
                return false;
            }
            const char * kernel_name = amd_ffn_swiglu_q4_decode_kernel(
                context->device, gate_signature.weight_kind);
            const auto * metadata = kernel_name == nullptr
                ? nullptr : context->device->aot->find(kernel_name);
            if (metadata == nullptr || metadata->block_size <= 0) {
                return false;
            }
            int k = static_cast<int>(gate_signature.k);
            int rows = static_cast<int>(gate_signature.rows);
            void * gate_weights_u8 = gate_projection->src[0]->data;
            void * gate_weights_f16 = gate_projection->src[0]->data;
            void * up_weights_u8 = up_projection->src[0]->data;
            void * up_weights_f16 = up_projection->src[0]->data;
            void * activation_data = gate_projection->src[1]->data;
            void * output_data = glu->data;
            if (gate_weights_u8 == nullptr || up_weights_u8 == nullptr || activation_data == nullptr) {
                return false;
            }
            flagos_amd::kernel_arguments arguments = {
                &gate_weights_u8, &gate_weights_f16,
                &up_weights_u8, &up_weights_f16,
                &activation_data, &output_data, &k, &rows,
            };
            const unsigned int row_tile = static_cast<unsigned int>(metadata->block_size);
            const bool launched = context->device->aot->launch(
                kernel_name, context->stream,
                (static_cast<unsigned int>(rows) + row_tile - 1) / row_tile,
                1, 1, arguments);
            if (launched) {
                context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
                context->stats.fusion_steps.fetch_add(1, std::memory_order_relaxed);
                context->stats.fusion_ffn_swiglu.fetch_add(1, std::memory_order_relaxed);
                const bool q40_staged = std::strcmp(
                    kernel_name, "flagos_ffn_swiglu_q4_0_f32_decode_staged") == 0;
                if (q40_staged) {
                    context->stats.fusion_ffn_swiglu_q40_staged.fetch_add(
                        1, std::memory_order_relaxed);
                }
                amd_trace_op(context, glu,
                    gate_signature.weight_kind == flagos_quantized_matmul_kind::q4_0
                    ? (q40_staged
                        ? "ffn_swiglu_q40_decode_staged" : "ffn_swiglu_q40_decode")
                    : "ffn_swiglu_q4_decode");
            }
            return launched;
        }
        if (!amd_supports_ffn_swiglu_f16(context->device, gate_projection, up_projection, glu)) {
            return false;
        }
        flagos_quantized_matmul_signature first_signature;
        flagos_quantized_matmul_signature second_signature;
        if (!flagos_describe_quantized_matmul(gate_projection, &first_signature) ||
            !flagos_describe_quantized_matmul(up_projection, &second_signature)) {
            return false;
        }
        void * first_weights = amd_get_dequantized_weight(
            context, gate_projection->src[0], first_signature);
        void * second_weights = amd_get_dequantized_weight(
            context, up_projection->src[0], second_signature);
        if (first_weights == nullptr || second_weights == nullptr ||
            gate_projection->src[1] == nullptr || glu->data == nullptr) {
            return false;
        }
        const bool grouped = amd_grouped_f16_ffn_enabled(context->device,
                static_cast<int>(first_signature.columns)) &&
            context->device->aot->find("flagos_ffn_swiglu_f16_f32_grouped") != nullptr;
        const char * kernel_name = grouped
            ? "flagos_ffn_swiglu_f16_f32_grouped" : "flagos_ffn_swiglu_f16_f32_batched";
        const auto * metadata = context->device->aot->find(kernel_name);
        if (metadata == nullptr) {
            return false;
        }
        int k = static_cast<int>(first_signature.k);
        int rows = static_cast<int>(first_signature.rows);
        int columns = static_cast<int>(first_signature.columns);
        void * activation_data = gate_projection->src[1]->data;
        void * output_data = glu->data;
        if (activation_data == nullptr) {
            return false;
        }
        flagos_amd::kernel_arguments arguments = {
            &first_weights, &second_weights, &activation_data, &output_data,
            &k, &rows, &columns,
        };
        const unsigned int tile_m = metadata->tile_m > 0
            ? static_cast<unsigned int>(metadata->tile_m) : AMD_F16_MATMUL_BLOCK_M;
        const unsigned int tile_n = metadata->tile_n > 0
            ? static_cast<unsigned int>(metadata->tile_n) : AMD_F16_MATMUL_BLOCK_N;
        const unsigned int blocks_m = (static_cast<unsigned int>(rows) + tile_m - 1) / tile_m;
        const unsigned int blocks_n = (static_cast<unsigned int>(columns) + tile_n - 1) / tile_n;
        const unsigned int grid_x = grouped ? blocks_m * blocks_n : blocks_m;
        const unsigned int grid_y = grouped ? 1U : blocks_n;
        const bool launched = context->device->aot->launch(
            kernel_name, context->stream,
            grid_x, grid_y, 1, arguments);
        if (launched) {
            context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_steps.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_ffn_swiglu.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(context, glu,
                grouped ? "ffn_swiglu_f16_grouped" : "ffn_swiglu_f16");
        }
        return launched;
    }
    if (step.candidate.id == flagos_pattern_id::flash_attn_decode && step.candidate.node_indices.size() == 1) {
        ggml_tensor * node = cgraph->nodes[step.candidate.node_indices[0]];
        if (!amd_supports_flash_attn_decode(context->device, node) ||
            !amd_tensor_bindings_ready(node)) return false;
        const ggml_tensor * q = node->src[0];
        const ggml_tensor * k = node->src[1];
        const ggml_tensor * v = node->src[2];
        const ggml_tensor * mask = node->src[3];
        int key_length = static_cast<int>(k->ne[1]);
        int q_per_kv = static_cast<int>(q->ne[2] / k->ne[2]);
        int stride_q_token = static_cast<int>(q->nb[1] / sizeof(float));
        int stride_q_head = static_cast<int>(q->nb[2] / sizeof(float));
        int stride_k_token = static_cast<int>(k->nb[1] / sizeof(ggml_fp16_t));
        int stride_k_head = static_cast<int>(k->nb[2] / sizeof(ggml_fp16_t));
        int stride_v_token = static_cast<int>(v->nb[1] / sizeof(ggml_fp16_t));
        int stride_v_head = static_cast<int>(v->nb[2] / sizeof(ggml_fp16_t));
        int stride_output_token = static_cast<int>(node->nb[2] / sizeof(float));
        int stride_output_head = static_cast<int>(node->nb[1] / sizeof(float));
        float scale = 0.0f;
        std::memcpy(&scale, node->op_params, sizeof(scale));
        void * q_data = q->data;
        void * k_data = k->data;
        void * v_data = v->data;
        void * mask_data = mask->data;
        void * output_data = node->data;
        flagos_amd::kernel_arguments arguments = {
            &q_data, &k_data, &v_data, &mask_data, &output_data,
            &key_length, &q_per_kv, &stride_q_token, &stride_q_head,
            &stride_k_token, &stride_k_head, &stride_v_token, &stride_v_head,
            &stride_output_token, &stride_output_head, &scale,
        };
        const bool launched = context->device->aot->launch("flagos_flash_attn_decode_f32_f16", context->stream,
            static_cast<unsigned int>(q->ne[2]), 1, 1, arguments);
        if (launched) {
            context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_steps.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_flash_attn_decode.fetch_add(1, std::memory_order_relaxed);
            context->stats.flash_attn_decode.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(context, node, "flash_attn_decode");
        }
        return launched;
    }

    if (step.candidate.id == flagos_pattern_id::flash_attn_prefill && step.candidate.node_indices.size() == 1) {
        ggml_tensor * node = cgraph->nodes[step.candidate.node_indices[0]];
        if (!amd_supports_flash_attn_prefill(context->device, node) ||
            !amd_tensor_bindings_ready(node)) return false;
        const ggml_tensor * q = node->src[0];
        const ggml_tensor * k = node->src[1];
        const ggml_tensor * v = node->src[2];
        const ggml_tensor * mask = node->src[3];
        const auto * metadata = context->device->aot->find("flagos_flash_attn_prefill_f32_f16");
        int query_length = static_cast<int>(q->ne[1]);
        int key_length = static_cast<int>(k->ne[1]);
        int q_per_kv = static_cast<int>(q->ne[2] / k->ne[2]);
        int stride_q_token = static_cast<int>(q->nb[1] / sizeof(float));
        int stride_q_head = static_cast<int>(q->nb[2] / sizeof(float));
        int stride_k_token = static_cast<int>(k->nb[1] / sizeof(ggml_fp16_t));
        int stride_k_head = static_cast<int>(k->nb[2] / sizeof(ggml_fp16_t));
        int stride_v_token = static_cast<int>(v->nb[1] / sizeof(ggml_fp16_t));
        int stride_v_head = static_cast<int>(v->nb[2] / sizeof(ggml_fp16_t));
        int stride_mask_token = static_cast<int>(mask->nb[0] / sizeof(ggml_fp16_t));
        int stride_mask_query = static_cast<int>(mask->nb[1] / sizeof(ggml_fp16_t));
        int stride_output_token = static_cast<int>(node->nb[2] / sizeof(float));
        int stride_output_head = static_cast<int>(node->nb[1] / sizeof(float));
        float scale = 0.0f;
        std::memcpy(&scale, node->op_params, sizeof(scale));
        void * q_data = q->data;
        void * k_data = k->data;
        void * v_data = v->data;
        void * mask_data = mask->data;
        void * output_data = node->data;
        flagos_amd::kernel_arguments arguments = {
            &q_data, &k_data, &v_data, &mask_data, &output_data,
            &query_length, &key_length, &q_per_kv,
            &stride_q_token, &stride_q_head,
            &stride_k_token, &stride_k_head,
            &stride_v_token, &stride_v_head,
            &stride_mask_token, &stride_mask_query,
            &stride_output_token, &stride_output_head, &scale,
        };
        const unsigned int grid_y = static_cast<unsigned int>(
            (static_cast<uint64_t>(query_length) + metadata->block_size - 1) /
            metadata->block_size);
        const bool launched = context->device->aot->launch(
            "flagos_flash_attn_prefill_f32_f16", context->stream,
            static_cast<unsigned int>(q->ne[2]), grid_y, 1, arguments);
        if (launched) {
            context->stats.fusion_steps.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_flash_attn_prefill.fetch_add(1, std::memory_order_relaxed);
            context->stats.flash_attn_prefill.fetch_add(1, std::memory_order_relaxed);
            context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(context, node, "flash_attn_prefill");
        }
        return launched;
    }

    if (step.candidate.id == flagos_pattern_id::rope_kv_store && step.candidate.node_indices.size() == 3) {
        ggml_tensor * rope = cgraph->nodes[step.candidate.node_indices[0]];
        ggml_tensor * view = cgraph->nodes[step.candidate.node_indices[1]];
        ggml_tensor * set_rows = cgraph->nodes[step.candidate.node_indices[2]];
        if (!amd_supports_rope_kv_store(context->device, rope, view, set_rows)) {
            return false;
        }
        const int32_t * params = static_cast<const int32_t *>(rope->op_params);
        int n_cols = static_cast<int>(view->ne[0]);
        int n_rows = static_cast<int>(view->ne[1]);
        int n_dst_rows = static_cast<int>(set_rows->ne[1]);
        int head_dim = static_cast<int>(rope->src[0]->ne[0]);
        int n_heads = static_cast<int>(rope->src[0]->ne[1]);
        int n_dims = params[1];
        float freq_base = 0.0f;
        float freq_scale = 0.0f;
        std::memcpy(&freq_base, params + 5, sizeof(freq_base));
        std::memcpy(&freq_scale, params + 6, sizeof(freq_scale));
        void * x_data = rope->src[0]->data;
        void * positions_data = rope->src[1]->data;
        void * row_index_data = set_rows->src[1]->data;
        void * output_data = set_rows->data;
        flagos_amd::kernel_arguments arguments = {
            &x_data, &positions_data, &row_index_data, &output_data,
            &n_cols, &n_rows, &n_dst_rows, &head_dim, &n_heads, &n_dims,
            &freq_base, &freq_scale,
        };
        const auto * metadata = context->device->aot->find("flagos_rope_kv_store_f32_f16");
        const unsigned int grid_x = static_cast<unsigned int>(
            (static_cast<uint64_t>(n_cols) * n_rows + metadata->block_size - 1) /
            metadata->block_size);
        const bool launched = context->device->aot->launch("flagos_rope_kv_store_f32_f16", context->stream,
            grid_x, 1, 1, arguments);
        if (launched) {
            context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_steps.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_rope_kv_store.fetch_add(1, std::memory_order_relaxed);
            context->stats.rope_kv_store.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(context, set_rows, "rope_kv_store");
        }
        return launched;
    }

    ggml_tensor * add = nullptr;
    ggml_tensor * norm = nullptr;
    ggml_tensor * mul = nullptr;
    const char * kernel_name = nullptr;
    bool write_residual = false;
    if (step.candidate.id == flagos_pattern_id::rms_norm_mul && step.candidate.node_indices.size() == 2) {
        norm = cgraph->nodes[step.candidate.node_indices[0]];
        mul = cgraph->nodes[step.candidate.node_indices[1]];
        const bool terminal_only = amd_candidate_outputs_are(step.candidate, { 1 });
        const bool norm_and_terminal = amd_candidate_outputs_are(step.candidate, { 0, 1 });
        if (!terminal_only && !norm_and_terminal) {
            return false;
        }
        const bool inplace = terminal_only && amd_supports_rms_norm_mul_inplace(context->device, norm, mul);
        if (!inplace &&
            (amd_tensor_data_overlaps(norm, mul) || !amd_supports_rms_norm_mul(context->device, norm, mul))) {
            return false;
        }
        kernel_name = inplace
            ? amd_rms_norm_mul_inplace_kernel(context->device, norm->ne[0])
            : "flagos_rms_norm_mul_f32";
        if (kernel_name == nullptr) {
            return false;
        }
    } else if (step.candidate.id == flagos_pattern_id::add_rms_norm_mul && step.candidate.node_indices.size() == 3) {
        add = cgraph->nodes[step.candidate.node_indices[0]];
        norm = cgraph->nodes[step.candidate.node_indices[1]];
        mul = cgraph->nodes[step.candidate.node_indices[2]];
        const bool terminal_only = amd_candidate_outputs_are(step.candidate, { 2 });
        const bool norm_and_terminal = amd_candidate_outputs_are(step.candidate, { 1, 2 });
        write_residual = amd_candidate_outputs_are(step.candidate, { 0, 2 });
        if (!terminal_only && !norm_and_terminal && !write_residual) {
            return false;
        }
        if (write_residual) {
            if (!amd_supports_add_rms_norm_mul_residual(context->device, add, norm, mul)) {
                return false;
            }
            kernel_name = amd_add_rms_norm_mul_residual_kernel(
                context->device, norm->ne[0]);
            if (kernel_name == nullptr) {
                return false;
            }
        }
        const bool inplace = terminal_only &&
            amd_supports_add_rms_norm_mul_inplace(context->device, add, norm, mul);
        if (!write_residual) {
            if (!inplace &&
                (amd_tensor_data_overlaps(norm, mul) ||
                 !amd_supports_add_rms_norm_mul(context->device, add, norm, mul))) {
                return false;
            }
            kernel_name = inplace ? "flagos_add_rms_norm_mul_inplace_f32" : "flagos_add_rms_norm_mul_f32";
        }
    } else return false;
    const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    int n_cols = static_cast<int>(norm->ne[0]);
    const unsigned int rows = static_cast<unsigned int>(ggml_nelements(norm) / norm->ne[0]);
    float eps = 0.0f;
    std::memcpy(&eps, norm->op_params, sizeof(eps));
    void * norm_data = norm->data;
    void * mul_data = mul->data;
    void * src_data = add == nullptr ? norm->src[0]->data : add->src[0]->data;
    void * bias_data = add == nullptr ? nullptr : add->src[1]->data;
    void * weight_data = weight->data;
    const bool narrow_rms =
        std::strcmp(kernel_name, "flagos_rms_norm_mul_inplace_f32_narrow") == 0;
    const bool inplace = narrow_rms ||
        std::strcmp(kernel_name, "flagos_rms_norm_mul_inplace_f32") == 0 ||
        std::strcmp(kernel_name, "flagos_add_rms_norm_mul_inplace_f32") == 0;
    if (write_residual) {
        void * residual_data = add->data;
        flagos_amd::kernel_arguments residual_arguments = {
            &residual_data, &mul_data, &src_data, &bias_data, &weight_data, &n_cols, &eps,
        };
        const bool launched = context->device->aot->launch(kernel_name, context->stream,
            rows, 1, 1, residual_arguments);
        if (launched) {
            context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_steps.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_add_rms_norm_mul.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(context, mul, "add_rms_norm_mul_residual");
        }
        return launched;
    }
    if (inplace) {
        flagos_amd::kernel_arguments inplace_arguments;
        if (add == nullptr) {
            inplace_arguments.assign(&mul_data, &src_data, &weight_data, &n_cols, &eps);
        } else {
            inplace_arguments.assign(&mul_data, &src_data, &bias_data, &weight_data, &n_cols, &eps);
        }
        const unsigned int inplace_rows = rows;
        const bool launched = context->device->aot->launch(kernel_name, context->stream,
            inplace_rows, 1, 1, inplace_arguments);
        if (launched) {
            context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_steps.fetch_add(1, std::memory_order_relaxed);
            if (step.candidate.id == flagos_pattern_id::rms_norm_mul) {
                context->stats.fusion_rms_norm_mul.fetch_add(1, std::memory_order_relaxed);
                if (narrow_rms) {
                    context->stats.fusion_rms_norm_mul_narrow.fetch_add(
                        1, std::memory_order_relaxed);
                }
            } else {
                context->stats.fusion_add_rms_norm_mul.fetch_add(1, std::memory_order_relaxed);
            }
            amd_trace_op(context, mul, narrow_rms ? "rms_norm_mul_inplace_narrow" :
                step.candidate.id == flagos_pattern_id::rms_norm_mul
                    ? "rms_norm_mul_inplace" : "add_rms_norm_mul_inplace");
        }
        return launched;
    }
    flagos_amd::kernel_arguments arguments = { &norm_data, &mul_data, &src_data };
    if (add != nullptr) arguments.push_back(&bias_data);
    arguments.push_back(&weight_data);
    arguments.push_back(&n_cols);
    arguments.push_back(&eps);
    const bool launched = context->device->aot->launch(kernel_name, context->stream, rows, 1, 1, arguments);
    if (launched) {
        context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
        context->stats.fusion_steps.fetch_add(1, std::memory_order_relaxed);
        if (step.candidate.id == flagos_pattern_id::rms_norm_mul) {
            context->stats.fusion_rms_norm_mul.fetch_add(1, std::memory_order_relaxed);
        } else {
            context->stats.fusion_add_rms_norm_mul.fetch_add(1, std::memory_order_relaxed);
        }
        amd_trace_op(context, mul, step.candidate.id == flagos_pattern_id::rms_norm_mul
            ? "rms_norm_mul" : "add_rms_norm_mul");
    }
    return launched;
}

static bool amd_supports_add(const amd_device_context * device, const ggml_tensor * op) {
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_ADD || op->src[0] == nullptr || op->src[1] == nullptr ||
        !amd_tensor_is_contiguous_f32(op) ||
        !amd_tensor_is_contiguous_f32(op->src[0]) ||
        !amd_tensor_is_contiguous_f32(op->src[1]) ||
        !ggml_are_same_shape(op, op->src[0]) ||
        !ggml_can_repeat(op->src[1], op->src[0]) ||
        ggml_nelements(op) <= 0 || ggml_nelements(op) > INT32_MAX ||
        ggml_nelements(op->src[1]) <= 0 || ggml_nelements(op->src[1]) > INT32_MAX) {
        return false;
    }
    const bool same_shape = ggml_are_same_shape(op->src[0], op->src[1]);
    const bool flat_vector_repeat = op->src[1]->ne[0] == op->src[0]->ne[0] &&
        op->src[1]->ne[1] == 1 && op->src[1]->ne[2] == 1 && op->src[1]->ne[3] == 1;
    const char * kernel = same_shape ? "flagos_add_f32" : "flagos_add_repeat_f32";
    return (same_shape || flat_vector_repeat) && device->aot->find(kernel) != nullptr &&
        amd_tensor_output_can_reuse_input(op, op->src[0]) &&
        (same_shape ? amd_tensor_output_can_reuse_input(op, op->src[1])
                    : !amd_tensor_data_overlaps(op, op->src[1]));
}

static bool amd_supports_mul(const amd_device_context * device, const ggml_tensor * op) {
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_MUL || op->src[0] == nullptr || op->src[1] == nullptr ||
        device->aot->find("flagos_mul_f32") == nullptr || op->type != GGML_TYPE_F32 ||
        op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32 ||
        !amd_tensor_is_contiguous_f32(op) || !amd_tensor_is_contiguous_f32(op->src[0]) ||
        !amd_tensor_is_contiguous_f32(op->src[1]) || ggml_nelements(op) <= 0 ||
        ggml_nelements(op) > INT32_MAX) {
        return false;
    }
    // ggml's binary-op contract keeps src[0] in the output shape and repeats
    // src[1] along dimensions for which it has extent one.  The AMD Triton
    // kernel uses a flat ``index % src1_elements`` broadcast, which is only
    // equivalent to ggml's multidimensional repeat when src[1] is either the
    // complete output or a single leading-dimension vector.  Do not advertise
    // generic shapes: an incorrect GPU result is worse than a CPU fallback.
    if (!ggml_are_same_shape(op, op->src[0]) || !ggml_can_repeat(op->src[1], op->src[0]) ||
        ggml_nelements(op->src[1]) <= 0 || ggml_nelements(op->src[1]) > INT32_MAX) {
        return false;
    }
    if (ggml_are_same_shape(op->src[0], op->src[1])) {
        return amd_tensor_output_can_reuse_input(op, op->src[0]) &&
            amd_tensor_output_can_reuse_input(op, op->src[1]);
    }
    return op->src[1]->ne[0] == op->src[0]->ne[0] &&
        op->src[1]->ne[1] == 1 && op->src[1]->ne[2] == 1 && op->src[1]->ne[3] == 1 &&
        amd_tensor_output_can_reuse_input(op, op->src[0]) &&
        !amd_tensor_data_overlaps(op, op->src[1]);
}

static bool amd_supports_scale(const amd_device_context * device, const ggml_tensor * op) {
    return device != nullptr && device->aot != nullptr && op != nullptr &&
        op->op == GGML_OP_SCALE && op->src[0] != nullptr &&
        device->aot->find("flagos_scale_f32") != nullptr &&
        amd_tensor_is_contiguous_f32(op) && amd_tensor_is_contiguous_f32(op->src[0]) &&
        ggml_are_same_shape(op, op->src[0]) &&
        ggml_nelements(op) <= INT32_MAX &&
        amd_tensor_output_can_reuse_input(op, op->src[0]);
}

static bool amd_supports_rms_norm(const amd_device_context * device, const ggml_tensor * op) {
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_RMS_NORM || op->src[0] == nullptr ||
        device->aot->find("flagos_rms_norm_f32") == nullptr ||
        !amd_tensor_is_contiguous_f32(op) ||
        !amd_tensor_is_contiguous_f32(op->src[0]) ||
        !ggml_are_same_shape(op, op->src[0]) || ggml_nelements(op) <= 0 ||
        ggml_nelements(op) > INT32_MAX ||
        op->ne[0] <= 0 || op->ne[0] > INT32_MAX || op->ne[0] % 4 != 0) {
        return false;
    }

    float eps = 0.0f;
    std::memcpy(&eps, op->op_params, sizeof(eps));
    if (!std::isfinite(eps) || eps < 0.0f ||
        !amd_tensor_output_can_reuse_input(op, op->src[0])) {
        return false;
    }

    // The Triton kernel uses one program per row and a compile-time BLOCK.
    // A package may choose a wider block than the model's hidden size, but it
    // must never be narrower or the masked load/store would be incomplete.
    const auto * metadata = device->aot->find("flagos_rms_norm_f32");
    return metadata->block_size >= op->ne[0];
}

static enum ggml_status amd_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    if (backend == nullptr || cgraph == nullptr || cgraph->n_nodes < 0 ||
        (cgraph->n_nodes > 0 && cgraph->nodes == nullptr)) {
        return GGML_STATUS_FAILED;
    }
    auto * backend_context = static_cast<amd_backend_context *>(backend->context);
    if (backend_context == nullptr || backend_context->device == nullptr) {
        return GGML_STATUS_FAILED;
    }
    for (int index = 0; index < cgraph->n_nodes; ++index) {
        if (cgraph->nodes[index] == nullptr) {
            return GGML_STATUS_FAILED;
        }
    }
    std::lock_guard<std::mutex> execution_lock(backend_context->execution_mutex);
    if (!amd_hip_check(hipSetDevice(backend_context->device->ordinal), "hipSetDevice")) {
        return GGML_STATUS_FAILED;
    }
    if (!amd_revalidate_dequant_cache(backend_context)) {
        return GGML_STATUS_FAILED;
    }
    if (cgraph->n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }
    amd_prepare_dequant_cache(backend_context, cgraph);
    const flagos_fusion_interface fusion_interface {
        /* .query_lowering = */ amd_query_fusion,
        /* .validate_lowering = */ amd_query_fusion,
        /* .execute_fusion = */ amd_execute_fusion,
        /* .user_data      = */ backend_context,
        /* .configuration_id = */ amd_fusion_config_key(backend_context),
    };
    const uint64_t config_key = fusion_interface.configuration_id;
    if (backend_context->graph_plan_config_key != 0 &&
        backend_context->graph_plan_config_key != config_key) {
        // Configuration changes are unusual in production, but are common in
        // bring-up tests.  Drop both plans and executable graphs together so
        // no old fusion choice can survive a selector change.
        if (!amd_hip_check(hipStreamSynchronize(backend_context->stream),
                "hipStreamSynchronize before plan invalidation")) {
            return GGML_STATUS_FAILED;
        }
        backend_context->graph_plans.clear();
        amd_clear_graph_captures(backend_context);
    }
    backend_context->graph_plan_config_key = config_key;
    bool plan_created = false;
    const flagos_graph_plan & plan = backend_context->graph_plans.get_or_create(
        cgraph, fusion_interface, &plan_created);
    if (plan_created) {
        backend_context->stats.graph_plans_built.fetch_add(1, std::memory_order_relaxed);
        for (const auto & step : plan.steps) {
            if (step.kind == flagos_execution_kind::pattern) {
                backend_context->stats.graph_plan_pattern_steps.fetch_add(1, std::memory_order_relaxed);
            } else {
                backend_context->stats.graph_plan_direct_steps.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    if (plan_created && std::getenv("FLAGOS_LOG_GRAPH_PLAN") != nullptr) {
        size_t pattern_count = 0;
        for (const auto & step : plan.steps) {
            if (step.kind == flagos_execution_kind::pattern) {
                ++pattern_count;
                GGML_LOG_INFO("FlagOS AMD: plan %016llx selected %s (%s) at node %d\n",
                    static_cast<unsigned long long>(plan.structural_fingerprint),
                    flagos_pattern_name(step.candidate.id),
                    step.candidate.scope == flagos_fusion_scope::single_operator ? "operator" : "graph",
                    step.candidate.node_indices[0]);
            }
        }
        GGML_LOG_INFO("FlagOS AMD: built plan %016llx with %zu steps and %zu patterns for %d nodes\n",
            static_cast<unsigned long long>(plan.structural_fingerprint),
            plan.steps.size(), pattern_count, cgraph->n_nodes);
        if (pattern_count == 0) {
            for (const auto & step : plan.steps) {
                if (step.kind == flagos_execution_kind::direct && !step.candidate.node_indices.empty()) {
                    GGML_LOG_DEBUG("FlagOS AMD: direct node %d op=%s\n",
                        step.candidate.node_indices[0],
                        ggml_op_name(cgraph->nodes[step.candidate.node_indices[0]]->op));
                }
            }
        }
    }

    const bool capture_requested = amd_graph_capture_enabled();
    bool capture_eligible = false;
    if (capture_requested) {
        backend_context->stats.graph_capture_requests.fetch_add(1, std::memory_order_relaxed);
        if (!amd_graph_capture_prefill_enabled() && !amd_graph_is_decode(cgraph)) {
            backend_context->stats.graph_capture_prefill_skips.fetch_add(1, std::memory_order_relaxed);
        } else if (!amd_f16_graph_cache_ready(backend_context, cgraph)) {
            backend_context->stats.graph_capture_cache_waits.fetch_add(1, std::memory_order_relaxed);
        } else if (!plan.capture_safe()) {
            backend_context->stats.graph_capture_unsafe_plans.fetch_add(1, std::memory_order_relaxed);
        } else if (cgraph->n_nodes < 4) {
            backend_context->stats.graph_capture_small_graphs.fetch_add(1, std::memory_order_relaxed);
        } else {
            capture_eligible = true;
            backend_context->stats.graph_capture_eligible.fetch_add(1, std::memory_order_relaxed);
        }
    }
    uint64_t capture_binding_fingerprint = 0;
    if (capture_eligible && !flagos_graph_binding_fingerprint(
            cgraph, plan, &capture_binding_fingerprint)) {
        capture_eligible = false;
        backend_context->stats.graph_capture_failures.fetch_add(1, std::memory_order_relaxed);
    } else if (capture_eligible) {
        // A captured launch may contain provider-private F16 cache pointers
        // that are not GGML tensor bindings.  Fold the cache epoch into the
        // binding key so an in-place model-weight update cannot replay an
        // executable bound to a retired dequantized allocation.
        capture_binding_fingerprint ^= backend_context->dequant_cache_epoch;
        capture_binding_fingerprint *= 1099511628211ULL;
    }
    flagos_graph_binding_snapshot capture_bindings;
    bool capture_ready = false;
    if (capture_eligible) {
        bool structure_found = false;
        if (amd_graph_capture_entry * entry = amd_find_graph_capture(
                backend_context, plan.structural_fingerprint,
                capture_binding_fingerprint, cgraph, plan, &structure_found)) {
            if (amd_hip_check(hipGraphLaunch(entry->executable, backend_context->stream), "hipGraphLaunch")) {
                backend_context->stats.graph_replays.fetch_add(1, std::memory_order_relaxed);
                return GGML_STATUS_SUCCESS;
            }
            backend_context->stats.graph_replay_failures.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.graph_capture_failures.fetch_add(1, std::memory_order_relaxed);
            amd_discard_graph_capture(backend_context, entry);
            return GGML_STATUS_FAILED;
        } else {
            backend_context->stats.graph_capture_exec_misses.fetch_add(1, std::memory_order_relaxed);
            if (structure_found) {
                backend_context->stats.graph_capture_pointer_misses.fetch_add(1, std::memory_order_relaxed);
            }
            capture_ready = amd_take_graph_capture_candidate(
                backend_context, plan.structural_fingerprint,
                capture_binding_fingerprint, cgraph, plan, &capture_bindings);
            if (capture_ready) {
                backend_context->stats.graph_capture_attempts.fetch_add(1, std::memory_order_relaxed);
            } else {
                flagos_graph_binding_snapshot candidate_bindings;
                if (flagos_graph_binding_snapshot_create(
                        cgraph, plan, &candidate_bindings)) {
                    candidate_bindings.fingerprint = capture_binding_fingerprint;
                    amd_store_graph_capture_candidate(
                        backend_context, plan.structural_fingerprint,
                        std::move(candidate_bindings));
                    backend_context->stats.graph_capture_warmups.fetch_add(1, std::memory_order_relaxed);
                } else {
                    backend_context->stats.graph_capture_failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    amd_graph_capture_scope capture_scope(backend_context->stream);
    bool capturing = false;
    if (capture_ready) {
        // Capture only after all work queued before this graph has completed.
        // This avoids accidentally importing scheduler copies or an earlier
        // graph into the captured dependency graph.
        if (!amd_hip_check(hipStreamSynchronize(backend_context->stream), "hipStreamSynchronize before capture")) {
            backend_context->stats.graph_capture_failures.fetch_add(1, std::memory_order_relaxed);
            return GGML_STATUS_FAILED;
        }
        capturing = capture_scope.begin();
        if (!capturing) {
            backend_context->stats.graph_capture_failures.fetch_add(1, std::memory_order_relaxed);
            return GGML_STATUS_FAILED;
        }
    }

    for (const auto & step : plan.steps) {
        if (step.kind == flagos_execution_kind::pattern) {
            if (!flagos_execute_fusion_step(fusion_interface, cgraph, step)) {
                GGML_LOG_ERROR("FlagOS AMD: fusion execution failed for %s\n",
                    flagos_pattern_name(step.candidate.id));
                return GGML_STATUS_FAILED;
            }
            continue;
        }
        const int i = step.candidate.node_indices[0];
        ggml_tensor * node = cgraph->nodes[i];
        if (amd_tensor_is_view_op(node)) {
            continue;
        }
        if (ggml_nelements(node) == 0 &&
            (amd_supports_copy_f32(backend_context->device, node) ||
             amd_supports_scale(backend_context->device, node))) {
            continue;
        }
        if (!amd_tensor_bindings_ready(node)) {
            GGML_LOG_ERROR("FlagOS AMD: missing tensor binding for %s\n", ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
        }
        if (amd_supports_copy_f32(backend_context->device, node)) {
            const auto * metadata = backend_context->device->aot->find("flagos_copy_strided_f32");
            int src_ne[GGML_MAX_DIMS];
            int dst_ne[GGML_MAX_DIMS];
            int src_stride[GGML_MAX_DIMS];
            int dst_stride[GGML_MAX_DIMS];
            for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                src_ne[dim] = static_cast<int>(node->src[0]->ne[dim]);
                dst_ne[dim] = static_cast<int>(node->ne[dim]);
                src_stride[dim] = static_cast<int>(node->src[0]->nb[dim] / sizeof(float));
                dst_stride[dim] = static_cast<int>(node->nb[dim] / sizeof(float));
            }
            int n_elements = static_cast<int>(ggml_nelements(node));
            void * src_data = node->src[0]->data;
            void * dst_data = node->data;
            flagos_amd::kernel_arguments arguments = {
                &src_data, &dst_data,
                &src_ne[0], &src_ne[1], &src_ne[2], &src_ne[3],
                &dst_ne[0], &dst_ne[1], &dst_ne[2], &dst_ne[3],
                &src_stride[0], &src_stride[1], &src_stride[2], &src_stride[3],
                &dst_stride[0], &dst_stride[1], &dst_stride[2], &dst_stride[3],
                &n_elements,
            };
            const unsigned int grid_x = static_cast<unsigned int>(
                (static_cast<uint64_t>(n_elements) + metadata->block_size - 1) /
                metadata->block_size);
            if (!backend_context->device->aot->launch(
                    "flagos_copy_strided_f32", backend_context->stream,
                    grid_x, 1, 1, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "copy_strided");
            continue;
        }
        if (node->op == GGML_OP_CONCAT && amd_supports_concat(backend_context->device, node)) {
            const auto * metadata = backend_context->device->aot->find("flagos_concat_f32");
            int dim = ggml_get_op_params_i32(node, 0);
            int dst_ne[GGML_MAX_DIMS];
            int a_ne[GGML_MAX_DIMS];
            int a_stride[GGML_MAX_DIMS];
            int b_stride[GGML_MAX_DIMS];
            for (int index = 0; index < GGML_MAX_DIMS; ++index) {
                dst_ne[index] = static_cast<int>(node->ne[index]);
                a_ne[index] = static_cast<int>(node->src[0]->ne[index]);
                a_stride[index] = static_cast<int>(node->src[0]->nb[index] / sizeof(float));
                b_stride[index] = static_cast<int>(node->src[1]->nb[index] / sizeof(float));
            }
            int n_elements = static_cast<int>(ggml_nelements(node));
            void * a_data = node->src[0]->data;
            void * b_data = node->src[1]->data;
            void * output_data = node->data;
            flagos_amd::kernel_arguments arguments = {
                &a_data, &b_data, &output_data, &dim,
                &dst_ne[0], &dst_ne[1], &dst_ne[2], &dst_ne[3],
                &a_ne[0], &a_ne[1], &a_ne[2], &a_ne[3],
                &a_stride[0], &a_stride[1], &a_stride[2], &a_stride[3],
                &b_stride[0], &b_stride[1], &b_stride[2], &b_stride[3],
                &n_elements,
            };
            const unsigned int grid_x = static_cast<unsigned int>(
                (static_cast<uint64_t>(n_elements) + metadata->block_size - 1) /
                metadata->block_size);
            if (!backend_context->device->aot->launch(
                    "flagos_concat_f32", backend_context->stream,
                    grid_x, 1, 1, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "concat");
            continue;
        }
        if (node->op == GGML_OP_SSM_CONV && amd_supports_ssm_conv(backend_context->device, node)) {
            if (!amd_launch_ssm_conv(
                    backend_context, node, node, "flagos_ssm_conv_f32")) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "ssm_conv");
            continue;
        }
        if (node->op == GGML_OP_L2_NORM && amd_supports_l2_norm(backend_context->device, node)) {
            int n_cols = static_cast<int>(node->ne[0]);
            int stride_row = static_cast<int>(node->src[0]->nb[1] / sizeof(float));
            int stride_channel = static_cast<int>(node->src[0]->nb[2] / sizeof(float));
            int stride_sample = static_cast<int>(node->src[0]->nb[3] / sizeof(float));
            float eps = 0.0f;
            std::memcpy(&eps, node->op_params, sizeof(eps));
            int n_rows = static_cast<int>(node->ne[1]);
            int n_channels = static_cast<int>(node->ne[2]);
            void * input_data = node->src[0]->data;
            void * output_data = node->data;
            flagos_amd::kernel_arguments arguments = {
                &input_data, &output_data, &n_cols,
                &stride_row, &stride_channel, &stride_sample, &eps,
                &n_rows, &n_channels,
            };
            if (!backend_context->device->aot->launch(
                    "flagos_l2_norm_strided_f32", backend_context->stream,
                    static_cast<unsigned int>(node->ne[1]),
                    static_cast<unsigned int>(node->ne[2]),
                    static_cast<unsigned int>(node->ne[3]), arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "l2_norm");
            continue;
        }
        if (node->op == GGML_OP_GATED_DELTA_NET &&
            amd_supports_gated_delta_net(backend_context->device, node)) {
            const ggml_tensor * q = node->src[0];
            const ggml_tensor * v = node->src[2];
            const ggml_tensor * beta = node->src[4];
            const auto * metadata = backend_context->device->aot->find(
                "flagos_gated_delta_net_scalar_f32");
            int state_size = static_cast<int>(v->ne[0]);
            int n_heads = static_cast<int>(v->ne[1]);
            int n_tokens = static_cast<int>(v->ne[2]);
            int n_seqs = static_cast<int>(v->ne[3]);
            int sq1 = static_cast<int>(q->nb[1] / sizeof(float));
            int sq2 = static_cast<int>(q->nb[2] / sizeof(float));
            int sq3 = static_cast<int>(q->nb[3] / sizeof(float));
            int sv1 = static_cast<int>(v->nb[1] / sizeof(float));
            int sv2 = static_cast<int>(v->nb[2] / sizeof(float));
            int sv3 = static_cast<int>(v->nb[3] / sizeof(float));
            int sb1 = static_cast<int>(beta->nb[1] / sizeof(float));
            int sb2 = static_cast<int>(beta->nb[2] / sizeof(float));
            int sb3 = static_cast<int>(beta->nb[3] / sizeof(float));
            int q_heads = static_cast<int>(q->ne[1]);
            int q_seq_ratio = static_cast<int>(v->ne[3] / q->ne[3]);
            int snapshot_count = ggml_get_op_params_i32(node, 0);
            float scale = 1.0f / std::sqrt(static_cast<float>(state_size));
            void * q_data = node->src[0]->data;
            void * k_data = node->src[1]->data;
            void * v_data = node->src[2]->data;
            void * gate_data = node->src[3]->data;
            void * beta_data = node->src[4]->data;
            void * state_data = node->src[5]->data;
            void * output_data = node->data;
            flagos_amd::kernel_arguments arguments = {
                &q_data, &k_data, &v_data, &gate_data, &beta_data, &state_data, &output_data,
                &state_size, &n_heads, &n_tokens, &n_seqs,
                &sq1, &sq2, &sq3, &sv1, &sv2, &sv3, &sb1, &sb2, &sb3,
                &q_heads, &q_seq_ratio, &snapshot_count, &scale,
            };
            const unsigned int grid_z = static_cast<unsigned int>(
                (static_cast<uint64_t>(state_size) + metadata->tile_n - 1) /
                metadata->tile_n);
            if (!backend_context->device->aot->launch(
                    "flagos_gated_delta_net_scalar_f32", backend_context->stream,
                    static_cast<unsigned int>(n_heads),
                    static_cast<unsigned int>(n_seqs), grid_z, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "gated_delta_net_scalar");
            continue;
        }
        if (node->op == GGML_OP_GET_ROWS && amd_supports_get_rows(backend_context->device, node)) {
            const flagos_quantized_matmul_kind kind = amd_quantized_kind(node->src[0]->type);
            const char * kernel_name = amd_quantized_get_rows_kernel(kind);
            const int64_t quant_block_size = amd_quantized_block_size(kind);
            int n_cols = static_cast<int>(node->ne[0]);
            int n_tokens = static_cast<int>(ggml_nelements(node->src[1]));
            int blocks_per_row = static_cast<int>(n_cols / quant_block_size);
            void * weights_u8 = node->src[0]->data;
            void * weights_f16 = node->src[0]->data;
            void * index_data = node->src[1]->data;
            void * output_data = node->data;
            flagos_amd::kernel_arguments arguments = {
                &weights_u8, &weights_f16, &index_data, &output_data, &n_cols,
            };
            if (!backend_context->device->aot->launch(kernel_name, backend_context->stream,
                    static_cast<unsigned int>(n_tokens), static_cast<unsigned int>(blocks_per_row), 1, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            switch (kind) {
                case flagos_quantized_matmul_kind::q4_0:
                    backend_context->stats.q40_get_rows.fetch_add(1, std::memory_order_relaxed);
                    break;
                case flagos_quantized_matmul_kind::q4_1:
                    backend_context->stats.q41_get_rows.fetch_add(1, std::memory_order_relaxed);
                    break;
                case flagos_quantized_matmul_kind::q5_k:
                    backend_context->stats.q5_get_rows.fetch_add(1, std::memory_order_relaxed);
                    break;
                case flagos_quantized_matmul_kind::q4_k:
                    backend_context->stats.q4_get_rows.fetch_add(1, std::memory_order_relaxed);
                    break;
                case flagos_quantized_matmul_kind::q6_k:
                    backend_context->stats.q6_get_rows.fetch_add(1, std::memory_order_relaxed);
                    break;
                case flagos_quantized_matmul_kind::q8_0:
                    backend_context->stats.q80_get_rows.fetch_add(1, std::memory_order_relaxed);
                    break;
                case flagos_quantized_matmul_kind::none:
                    return GGML_STATUS_FAILED;
            }
            amd_trace_op(backend_context, node, "get_rows");
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT && amd_supports_quantized_mul_mat(backend_context->device, node)) {
            const flagos_quantized_matmul_kind kind = amd_quantized_kind(node->src[0]->type);
            const bool q4 = kind == flagos_quantized_matmul_kind::q4_k;
            int k = static_cast<int>(node->src[0]->ne[0]);
            int rows = static_cast<int>(node->src[0]->ne[1]);
            int columns = static_cast<int>(node->src[1]->ne[1]);
            void * weights_u8 = node->src[0]->data;
            void * weights_f16 = node->src[0]->data;
            void * activation_data = node->src[1]->data;
            void * output_data = node->data;
            // The dequant cache is intentionally limited to production-scale
            // projection matrices.  The backend-ops harness reuses small
            // tensor objects with different random payloads across cases;
            // keeping those shapes on the reference AOT path avoids making a
            // cache entry depend on test-only mutation semantics.
            if (columns > 1 && rows >= 64 && k >= 1024 &&
                amd_prefill_f16_gemm_enabled(backend_context->device)) {
                flagos_quantized_matmul_signature signature;
                const bool described = flagos_describe_quantized_matmul(node, &signature);
                void * dequantized_weights = described
                    ? amd_get_dequantized_weight(backend_context, node->src[0], signature) : nullptr;
                const auto * f16_metadata = backend_context->device->aot->find(
                    "flagos_mul_mat_f16_f32_batched");
                if (dequantized_weights != nullptr && f16_metadata != nullptr) {
                    flagos_amd::kernel_arguments f16_arguments = {
                        &dequantized_weights, &activation_data, &output_data,
                        &k, &rows, &columns,
                    };
                    const bool grouped = amd_grouped_f16_gemm_enabled(backend_context->device, columns) &&
                        backend_context->device->aot->find("flagos_mul_mat_f16_f32_grouped") != nullptr;
                    const char * f16_kernel = grouped
                        ? "flagos_mul_mat_f16_f32_grouped" : "flagos_mul_mat_f16_f32_batched";
                    if (grouped) {
                        f16_metadata = backend_context->device->aot->find(f16_kernel);
                    }
                    const unsigned int tile_m = f16_metadata->tile_m > 0
                        ? static_cast<unsigned int>(f16_metadata->tile_m) : AMD_F16_MATMUL_BLOCK_M;
                    const unsigned int tile_n = f16_metadata->tile_n > 0
                        ? static_cast<unsigned int>(f16_metadata->tile_n) : AMD_F16_MATMUL_BLOCK_N;
                    const unsigned int blocks_m =
                        (static_cast<unsigned int>(rows) + tile_m - 1) / tile_m;
                    const unsigned int blocks_n =
                        (static_cast<unsigned int>(columns) + tile_n - 1) / tile_n;
                    const unsigned int grid_x = grouped ? blocks_m * blocks_n : blocks_m;
                    const unsigned int grid_y = grouped ? 1U : blocks_n;
                    if (!backend_context->device->aot->launch(
                            f16_kernel, backend_context->stream,
                            grid_x, grid_y, 1, f16_arguments)) {
                        return GGML_STATUS_FAILED;
                    }
                    backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
                    backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
                    backend_context->stats.f16_matmul_batched.fetch_add(1, std::memory_order_relaxed);
                    amd_trace_op(backend_context, node,
                        grouped ? "f16_matmul_grouped" : "f16_matmul_batched");
                    continue;
                }
            }
            if (columns == 1) {
                const unsigned int requested_narrow_tile =
                    amd_q4_gemv_narrow_row_tile(backend_context->device);
                const unsigned int requested_q40_narrow_tile =
                    amd_q40_gemv_narrow_row_tile(backend_context->device);
                const unsigned int requested_q5_narrow_tile =
                    amd_q5_gemv_narrow_row_tile(backend_context->device);
                const bool narrow_q4 = q4 && requested_narrow_tile > 1U;
                const char * narrow_kernel = nullptr;
                unsigned int row_tile = 1U;
                if (kind == flagos_quantized_matmul_kind::q4_0 &&
                    requested_q40_narrow_tile > 1U &&
                    rows % requested_q40_narrow_tile == 0) {
                    narrow_kernel = "flagos_mul_mat_q4_0_f32_narrow";
                    row_tile = requested_q40_narrow_tile;
                } else if (kind == flagos_quantized_matmul_kind::q5_k &&
                           requested_q5_narrow_tile == 16U && rows % 16 == 0) {
                    narrow_kernel = "flagos_mul_mat_q5_k_f32_narrow16";
                    row_tile = 16U;
                } else if (narrow_q4 && requested_narrow_tile >= 8U && rows % 8 == 0 &&
                    backend_context->device->aot->find("flagos_mul_mat_q4_k_f32_narrow8") != nullptr) {
                    narrow_kernel = "flagos_mul_mat_q4_k_f32_narrow8";
                    row_tile = 8U;
                } else if (narrow_q4 && rows % 4 == 0 &&
                           backend_context->device->aot->find("flagos_mul_mat_q4_k_f32_narrow") != nullptr) {
                    // An 8-row request safely falls back to the validated
                    // four-row symbol when an older package is loaded.
                    narrow_kernel = "flagos_mul_mat_q4_k_f32_narrow";
                    row_tile = 4U;
                }
                const char * kernel_name = narrow_kernel != nullptr ? narrow_kernel
                    : amd_quantized_gemv_kernel(kind);
                flagos_amd::kernel_arguments arguments = {
                    &weights_u8, &weights_f16, &activation_data, &output_data, &k, &rows,
                };
                const bool launched = backend_context->device->aot->launch(kernel_name, backend_context->stream,
                    (static_cast<unsigned int>(rows) + row_tile - 1) / row_tile, 1, 1, arguments);
                if (!launched) {
                    return GGML_STATUS_FAILED;
                }
                backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
                backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
                if (kind == flagos_quantized_matmul_kind::q4_0) {
                    backend_context->stats.q40_matmul.fetch_add(1, std::memory_order_relaxed);
                } else if (kind == flagos_quantized_matmul_kind::q5_k) {
                    backend_context->stats.q5_matmul.fetch_add(1, std::memory_order_relaxed);
                } else if (kind == flagos_quantized_matmul_kind::q4_1) {
                    backend_context->stats.q41_matmul.fetch_add(1, std::memory_order_relaxed);
                } else if (kind == flagos_quantized_matmul_kind::q8_0) {
                    backend_context->stats.q80_matmul.fetch_add(1, std::memory_order_relaxed);
                } else if (q4) {
                    backend_context->stats.q4_matmul.fetch_add(1, std::memory_order_relaxed);
                } else {
                    backend_context->stats.q6_matmul.fetch_add(1, std::memory_order_relaxed);
                }
                amd_trace_op(backend_context, node, narrow_kernel != nullptr
                    ? (kind == flagos_quantized_matmul_kind::q4_0
                        ? "q40_matmul_narrow" : kind == flagos_quantized_matmul_kind::q5_k
                        ? "q5_matmul_narrow16" : row_tile == 8U
                        ? "q4_matmul_narrow8" : "q4_matmul_narrow4")
                    : kind == flagos_quantized_matmul_kind::q4_0
                    ? "q40_matmul" : kind == flagos_quantized_matmul_kind::q5_k
                    ? "q5_matmul" : q4 ? "q4_matmul" : "q6_matmul");
            } else {
                const char * kernel_name = amd_quantized_batched_kernel(
                    backend_context->device, kind);
                // Execute must remain fail-closed even if a future kind or
                // runtime policy diverges from the earlier supports_op query.
                if (kernel_name == nullptr) {
                    return GGML_STATUS_FAILED;
                }
                const bool tiled = std::strstr(kernel_name, "_tiled") != nullptr;
                const auto * metadata = backend_context->device->aot->find(kernel_name);
                if (metadata == nullptr) {
                    return GGML_STATUS_FAILED;
                }
                flagos_amd::kernel_arguments arguments = {
                    &weights_u8, &weights_f16, &activation_data, &output_data,
                    &k, &rows, &columns,
                };
                const unsigned int tile_m = tiled && metadata->tile_m > 0
                    ? static_cast<unsigned int>(metadata->tile_m) : 1;
                const unsigned int tile_n = tiled && metadata->tile_n > 0
                    ? static_cast<unsigned int>(metadata->tile_n)
                    : static_cast<unsigned int>(metadata->block_size);
                const unsigned int grid_x = tiled
                    ? (static_cast<unsigned int>(rows) + tile_m - 1) / tile_m
                    : static_cast<unsigned int>(rows);
                const unsigned int grid_y = (static_cast<unsigned int>(columns) + tile_n - 1) / tile_n;
                if (!backend_context->device->aot->launch(kernel_name, backend_context->stream,
                        grid_x, grid_y, 1, arguments)) {
                    return GGML_STATUS_FAILED;
                }
                backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
                backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
                if (kind == flagos_quantized_matmul_kind::q4_0) {
                    backend_context->stats.q40_matmul_batched.fetch_add(1, std::memory_order_relaxed);
                } else if (kind == flagos_quantized_matmul_kind::q5_k) {
                    backend_context->stats.q5_matmul_batched.fetch_add(1, std::memory_order_relaxed);
                } else if (kind == flagos_quantized_matmul_kind::q4_1) {
                    backend_context->stats.q41_matmul_batched.fetch_add(1, std::memory_order_relaxed);
                } else if (kind == flagos_quantized_matmul_kind::q8_0) {
                    backend_context->stats.q80_matmul_batched.fetch_add(1, std::memory_order_relaxed);
                } else if (q4) {
                    backend_context->stats.q4_matmul_batched.fetch_add(1, std::memory_order_relaxed);
                } else {
                    backend_context->stats.q6_matmul_batched.fetch_add(1, std::memory_order_relaxed);
                }
                if (tiled) {
                    (q4 ? backend_context->stats.q4_matmul_tiled : backend_context->stats.q6_matmul_tiled)
                        .fetch_add(1, std::memory_order_relaxed);
                }
                amd_trace_op(backend_context, node,
                    kind == flagos_quantized_matmul_kind::q4_0
                    ? "q40_matmul_batched" : kind == flagos_quantized_matmul_kind::q4_1
                    ? "q41_matmul_batched" : kind == flagos_quantized_matmul_kind::q8_0
                    ? "q80_matmul_batched" : kind == flagos_quantized_matmul_kind::q5_k
                    ? "q5_matmul_batched" : q4 ? "q4_matmul_batched" : "q6_matmul_batched");
            }
            continue;
        }
        if (node->op == GGML_OP_MUL && amd_supports_mul(backend_context->device, node)) {
            const ggml_tensor * full = ggml_nelements(node->src[0]) == ggml_nelements(node)
                ? node->src[0] : node->src[1];
            const ggml_tensor * broadcast = full == node->src[0] ? node->src[1] : node->src[0];
            const auto * metadata = backend_context->device->aot->find("flagos_mul_f32");
            int n_elements = static_cast<int>(ggml_nelements(node));
            int y_elements = static_cast<int>(ggml_nelements(broadcast));
            void * x_data = full->data;
            void * y_data = broadcast->data;
            void * output_data = node->data;
            flagos_amd::kernel_arguments arguments = { &x_data, &y_data, &output_data,
                &n_elements, &y_elements };
            const unsigned int grid_x = static_cast<unsigned int>(
                (static_cast<uint64_t>(n_elements) + metadata->block_size - 1) /
                metadata->block_size);
            if (!backend_context->device->aot->launch("flagos_mul_f32", backend_context->stream,
                    grid_x, 1, 1, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.mul.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "mul");
            continue;
        }
        if (node->op == GGML_OP_SCALE && amd_supports_scale(backend_context->device, node)) {
            const auto * metadata = backend_context->device->aot->find("flagos_scale_f32");
            float scale = 1.0f;
            float bias = 0.0f;
            std::memcpy(&scale, reinterpret_cast<const float *>(node->op_params), sizeof(float));
            std::memcpy(&bias, reinterpret_cast<const float *>(node->op_params) + 1, sizeof(float));
            int n_elements = static_cast<int>(ggml_nelements(node));
            void * input_data = node->src[0]->data;
            void * output_data = node->data;
            flagos_amd::kernel_arguments arguments = {
                &input_data, &output_data, &scale, &bias, &n_elements,
            };
            const unsigned int grid_x = static_cast<unsigned int>(
                (static_cast<uint64_t>(n_elements) + metadata->block_size - 1) /
                metadata->block_size);
            if (!backend_context->device->aot->launch("flagos_scale_f32", backend_context->stream,
                    grid_x, 1, 1, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.scale.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "scale");
            continue;
        }
        if (node->op == GGML_OP_SET_ROWS && amd_supports_set_rows(backend_context->device, node)) {
            const auto * metadata = backend_context->device->aot->find("flagos_set_rows_f32_f16");
            int n_cols = static_cast<int>(node->src[0]->ne[0]);
            int n_rows = static_cast<int>(node->src[0]->ne[1]);
            int n_dst_rows = static_cast<int>(node->ne[1]);
            void * src_data = node->src[0]->data;
            void * index_data = node->src[1]->data;
            void * dst_data = node->data;
            flagos_amd::kernel_arguments arguments = {
                &src_data, &index_data, &dst_data, &n_cols, &n_rows, &n_dst_rows,
            };
            const unsigned int grid_x = static_cast<unsigned int>(
                (static_cast<uint64_t>(n_cols) * n_rows + metadata->block_size - 1) /
                metadata->block_size);
            if (!backend_context->device->aot->launch("flagos_set_rows_f32_f16", backend_context->stream,
                    grid_x, 1, 1, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "set_rows");
            continue;
        }
        if (node->op == GGML_OP_ADD && amd_supports_add(backend_context->device, node)) {
            const bool same_shape = ggml_are_same_shape(node->src[0], node->src[1]);
            const char * kernel_name = same_shape ? "flagos_add_f32" : "flagos_add_repeat_f32";
            const auto * metadata = backend_context->device->aot->find(kernel_name);
            const unsigned int grid_x = static_cast<unsigned int>(
                (ggml_nelements(node) + metadata->block_size - 1) / metadata->block_size);
            int n = static_cast<int>(ggml_nelements(node));
            flagos_amd::kernel_arguments arguments = {
                &node->src[0]->data,
                &node->src[1]->data,
                &node->data,
                &n,
            };
            int repeated = static_cast<int>(ggml_nelements(node->src[1]));
            if (!same_shape) {
                arguments.push_back(&repeated);
            }
            if (!backend_context->device->aot->launch(kernel_name, backend_context->stream,
                    grid_x, 1, 1, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, same_shape ? "add" : "add_repeat");
            continue;
        }
        if (node->op == GGML_OP_SOFT_MAX && amd_supports_soft_max(backend_context->device, node)) {
            const ggml_tensor * mask = node->src[1];
            const char * kernel_name = mask == nullptr
                ? "flagos_soft_max_unmasked_f32" : "flagos_soft_max_masked_f32_f16";
            int n_cols = static_cast<int>(node->ne[0]);
            int rows = static_cast<int>(ggml_nelements(node) / node->ne[0]);
            float scale = 1.0f;
            std::memcpy(&scale, node->op_params, sizeof(scale));
            void * mask_data = mask == nullptr ? nullptr : mask->data;
            flagos_amd::kernel_arguments arguments = {
                &node->src[0]->data, &mask_data, &node->data, &n_cols, &scale,
            };
            const unsigned int grid_x = static_cast<unsigned int>(rows);
            const bool launched = backend_context->device->aot->launch(
                kernel_name, backend_context->stream, grid_x, 1, 1, arguments);
            if (!launched) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.soft_max.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "soft_max");
            continue;
        }
        if (node->op == GGML_OP_RMS_NORM && amd_supports_rms_norm(backend_context->device, node)) {
            const unsigned int rows = static_cast<unsigned int>(ggml_nelements(node) / node->ne[0]);
            int n_cols = static_cast<int>(node->ne[0]);
            float eps = 0.0f;
            std::memcpy(&eps, node->op_params, sizeof(eps));
            flagos_amd::kernel_arguments arguments = {
                &node->data,
                &node->src[0]->data,
                &n_cols,
                &eps,
            };
            if (!backend_context->device->aot->launch("flagos_rms_norm_f32", backend_context->stream,
                    rows, 1, 1, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "rms_norm");
            continue;
        }
        if (node->op == GGML_OP_ROPE && amd_supports_mrope(backend_context->device, node)) {
            int ne0 = static_cast<int>(node->ne[0]);
            int ne1 = static_cast<int>(node->ne[1]);
            int ne2 = static_cast<int>(node->ne[2]);
            int ne3 = static_cast<int>(node->ne[3]);
            int stride_x1 = static_cast<int>(node->src[0]->nb[1] / sizeof(float));
            int stride_x2 = static_cast<int>(node->src[0]->nb[2] / sizeof(float));
            int stride_x3 = static_cast<int>(node->src[0]->nb[3] / sizeof(float));
            int stride_y1 = static_cast<int>(node->nb[1] / sizeof(float));
            int stride_y2 = static_cast<int>(node->nb[2] / sizeof(float));
            int stride_y3 = static_cast<int>(node->nb[3] / sizeof(float));
            const int32_t * params = static_cast<const int32_t *>(node->op_params);
            int n_dims = params[1];
            int sections[GGML_MROPE_SECTIONS] = {
                params[11], params[12], params[13], params[14],
            };
            float freq_base = 0.0f;
            float freq_scale = 0.0f;
            std::memcpy(&freq_base, params + 5, sizeof(freq_base));
            std::memcpy(&freq_scale, params + 6, sizeof(freq_scale));
            void * src_data = node->src[0]->data;
            void * positions_data = node->src[1]->data;
            void * output_data = node->data;
            flagos_amd::kernel_arguments arguments = {
                &src_data, &positions_data, &output_data,
                &ne0, &ne1, &ne2, &ne3, &n_dims,
                &stride_x1, &stride_x2, &stride_x3,
                &stride_y1, &stride_y2, &stride_y3,
                &sections[0], &sections[1], &sections[2], &sections[3],
                &freq_base, &freq_scale,
            };
            if (!backend_context->device->aot->launch(
                    "flagos_mrope_f32", backend_context->stream,
                    static_cast<unsigned int>(ggml_nrows(node)), 1, 1, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.rope.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "mrope");
            continue;
        }
        if (node->op == GGML_OP_ROPE && amd_supports_rope_neox(backend_context->device, node)) {
            const auto * metadata = backend_context->device->aot->find("flagos_rope_neox_f32");
            int n_elements = static_cast<int>(ggml_nelements(node));
            int ne0 = static_cast<int>(node->src[0]->ne[0]);
            int ne1 = static_cast<int>(node->src[0]->ne[1]);
            int ne2 = static_cast<int>(node->src[0]->ne[2]);
            const int32_t * params = static_cast<const int32_t *>(node->op_params);
            int n_dims = params[1];
            float freq_base = 0.0f;
            float freq_scale = 0.0f;
            std::memcpy(&freq_base, params + 5, sizeof(freq_base));
            std::memcpy(&freq_scale, params + 6, sizeof(freq_scale));
            void * src_data = node->src[0]->data;
            void * positions_data = node->src[1]->data;
            void * dst_data = node->data;
            flagos_amd::kernel_arguments arguments = {
                &src_data, &positions_data, &dst_data, &n_elements,
                &ne0, &ne1, &ne2, &n_dims, &freq_base, &freq_scale,
            };
            const unsigned int grid_x = static_cast<unsigned int>(
                (static_cast<uint64_t>(n_elements) / 2 + metadata->block_size - 1) /
                metadata->block_size);
            if (!backend_context->device->aot->launch("flagos_rope_neox_f32", backend_context->stream,
                    grid_x, 1, 1, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.rope.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "rope");
            continue;
        }
        if (node->op == GGML_OP_UNARY &&
            (amd_supports_unary_f32(backend_context->device, node,
                 GGML_UNARY_OP_SIGMOID, "flagos_sigmoid_f32") ||
             amd_supports_unary_f32(backend_context->device, node,
                 GGML_UNARY_OP_SOFTPLUS, "flagos_softplus_f32"))) {
            const bool sigmoid = ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID;
            const char * kernel_name = sigmoid ? "flagos_sigmoid_f32" : "flagos_softplus_f32";
            const auto * metadata = backend_context->device->aot->find(kernel_name);
            int n = static_cast<int>(ggml_nelements(node));
            void * src_data = node->src[0]->data;
            void * dst_data = node->data;
            flagos_amd::kernel_arguments arguments = { &src_data, &dst_data, &n };
            const unsigned int grid_x = static_cast<unsigned int>(
                (static_cast<uint64_t>(n) + metadata->block_size - 1) /
                metadata->block_size);
            if (!backend_context->device->aot->launch(
                    kernel_name, backend_context->stream, grid_x, 1, 1, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, sigmoid ? "sigmoid" : "softplus");
            continue;
        }
        if (node->op == GGML_OP_UNARY && amd_supports_silu(backend_context->device, node)) {
            const auto * metadata = backend_context->device->aot->find("flagos_silu_f32");
            int n = static_cast<int>(ggml_nelements(node));
            void * src_data = node->src[0]->data;
            void * dst_data = node->data;
            flagos_amd::kernel_arguments arguments = { &src_data, &dst_data, &n };
            const unsigned int grid_x = static_cast<unsigned int>(
                (static_cast<uint64_t>(n) + metadata->block_size - 1) / metadata->block_size);
            if (!backend_context->device->aot->launch("flagos_silu_f32", backend_context->stream,
                    grid_x, 1, 1, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "silu");
            continue;
        }
        if (node->op == GGML_OP_GLU && amd_supports_swiglu(backend_context->device, node)) {
            const auto * metadata = backend_context->device->aot->find("flagos_swiglu_split_f32");
            int n = static_cast<int>(ggml_nelements(node));
            void * gate_data = node->src[0]->data;
            void * up_data = node->src[1]->data;
            void * output_data = node->data;
            flagos_amd::kernel_arguments arguments = { &gate_data, &up_data, &output_data, &n };
            const unsigned int grid_x = static_cast<unsigned int>(
                (static_cast<uint64_t>(n) + metadata->block_size - 1) / metadata->block_size);
            if (!backend_context->device->aot->launch("flagos_swiglu_split_f32", backend_context->stream,
                    grid_x, 1, 1, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "swiglu");
            continue;
        }
        if (node->op == GGML_OP_FLASH_ATTN_EXT && amd_supports_flash_attn_decode(backend_context->device, node)) {
            const ggml_tensor * q = node->src[0];
            const ggml_tensor * k = node->src[1];
            const ggml_tensor * v = node->src[2];
            const ggml_tensor * mask = node->src[3];
            int key_length = static_cast<int>(k->ne[1]);
            int q_per_kv = static_cast<int>(q->ne[2] / k->ne[2]);
            int stride_q_token = static_cast<int>(q->nb[1] / sizeof(float));
            int stride_q_head = static_cast<int>(q->nb[2] / sizeof(float));
            int stride_k_token = static_cast<int>(k->nb[1] / sizeof(ggml_fp16_t));
            int stride_k_head = static_cast<int>(k->nb[2] / sizeof(ggml_fp16_t));
            int stride_v_token = static_cast<int>(v->nb[1] / sizeof(ggml_fp16_t));
            int stride_v_head = static_cast<int>(v->nb[2] / sizeof(ggml_fp16_t));
            int stride_output_token = static_cast<int>(node->nb[2] / sizeof(float));
            int stride_output_head = static_cast<int>(node->nb[1] / sizeof(float));
            float scale = 0.0f;
            std::memcpy(&scale, node->op_params, sizeof(scale));
            void * q_data = q->data;
            void * k_data = k->data;
            void * v_data = v->data;
            void * mask_data = mask->data;
            void * output_data = node->data;
            flagos_amd::kernel_arguments arguments = {
                &q_data, &k_data, &v_data, &mask_data, &output_data,
                &key_length, &q_per_kv, &stride_q_token, &stride_q_head,
                &stride_k_token, &stride_k_head, &stride_v_token, &stride_v_head,
                &stride_output_token, &stride_output_head, &scale,
            };
            if (!backend_context->device->aot->launch("flagos_flash_attn_decode_f32_f16", backend_context->stream,
                    static_cast<unsigned int>(q->ne[2]), 1, 1, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.flash_attn_decode.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "flash_attn_decode");
            continue;
        }
        if (node->op == GGML_OP_FLASH_ATTN_EXT && amd_supports_flash_attn_prefill(backend_context->device, node)) {
            const ggml_tensor * q = node->src[0];
            const ggml_tensor * k = node->src[1];
            const ggml_tensor * v = node->src[2];
            const ggml_tensor * mask = node->src[3];
            const auto * metadata = backend_context->device->aot->find("flagos_flash_attn_prefill_f32_f16");
            int query_length = static_cast<int>(q->ne[1]);
            int key_length = static_cast<int>(k->ne[1]);
            int q_per_kv = static_cast<int>(q->ne[2] / k->ne[2]);
            int stride_q_token = static_cast<int>(q->nb[1] / sizeof(float));
            int stride_q_head = static_cast<int>(q->nb[2] / sizeof(float));
            int stride_k_token = static_cast<int>(k->nb[1] / sizeof(ggml_fp16_t));
            int stride_k_head = static_cast<int>(k->nb[2] / sizeof(ggml_fp16_t));
            int stride_v_token = static_cast<int>(v->nb[1] / sizeof(ggml_fp16_t));
            int stride_v_head = static_cast<int>(v->nb[2] / sizeof(ggml_fp16_t));
            int stride_mask_token = static_cast<int>(mask->nb[0] / sizeof(ggml_fp16_t));
            int stride_mask_query = static_cast<int>(mask->nb[1] / sizeof(ggml_fp16_t));
            int stride_output_token = static_cast<int>(node->nb[2] / sizeof(float));
            int stride_output_head = static_cast<int>(node->nb[1] / sizeof(float));
            float scale = 0.0f;
            std::memcpy(&scale, node->op_params, sizeof(scale));
            void * q_data = q->data;
            void * k_data = k->data;
            void * v_data = v->data;
            void * mask_data = mask->data;
            void * output_data = node->data;
            flagos_amd::kernel_arguments arguments = {
                &q_data, &k_data, &v_data, &mask_data, &output_data,
                &query_length, &key_length, &q_per_kv,
                &stride_q_token, &stride_q_head,
                &stride_k_token, &stride_k_head,
                &stride_v_token, &stride_v_head,
                &stride_mask_token, &stride_mask_query,
                &stride_output_token, &stride_output_head, &scale,
            };
            const unsigned int grid_y = static_cast<unsigned int>(
                (static_cast<uint64_t>(query_length) + metadata->block_size - 1) /
                metadata->block_size);
            if (!backend_context->device->aot->launch(
                    "flagos_flash_attn_prefill_f32_f16", backend_context->stream,
                    static_cast<unsigned int>(q->ne[2]), grid_y, 1, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.flash_attn_prefill.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "flash_attn_prefill");
            continue;
        }
        GGML_LOG_ERROR("FlagOS AMD: unsupported op reached graph_compute: %s\n", ggml_op_name(node->op));
        {
            return GGML_STATUS_FAILED;
        }
    }

    if (capturing) {
        hipGraphExec_t executable = nullptr;
        if (!capture_scope.finish(&executable)) {
            backend_context->stats.graph_capture_failures.fetch_add(1, std::memory_order_relaxed);
            return GGML_STATUS_FAILED;
        }
        if (!amd_hip_check(hipGraphLaunch(executable, backend_context->stream), "hipGraphLaunch captured")) {
            backend_context->stats.graph_capture_failures.fetch_add(1, std::memory_order_relaxed);
            if (amd_hip_check(hipStreamSynchronize(backend_context->stream),
                    "hipStreamSynchronize after captured launch failure")) {
                amd_hip_check(hipGraphExecDestroy(executable),
                    "hipGraphExecDestroy after captured launch failure");
            } else {
                GGML_LOG_ERROR("FlagOS AMD: cannot prove failed captured launch is idle; leaking its executable safely\n");
            }
            return GGML_STATUS_FAILED;
        }
        if (!amd_store_graph_capture(
                backend_context, plan.structural_fingerprint,
                std::move(capture_bindings), executable)) {
            backend_context->stats.graph_capture_failures.fetch_add(1, std::memory_order_relaxed);
            if (amd_hip_check(hipStreamSynchronize(backend_context->stream),
                    "hipStreamSynchronize before uncached graph exec destroy")) {
                amd_hip_check(hipGraphExecDestroy(executable),
                    "hipGraphExecDestroy after capture cache failure");
            } else {
                GGML_LOG_ERROR("FlagOS AMD: cannot prove uncached graph executable is idle; leaking it safely\n");
            }
            return GGML_STATUS_FAILED;
        }
        backend_context->stats.graph_captures.fetch_add(1, std::memory_order_relaxed);
        GGML_LOG_DEBUG("FlagOS AMD: captured HIP graph %016llx with %d nodes\n",
            static_cast<unsigned long long>(plan.structural_fingerprint), cgraph->n_nodes);
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i g_backend_iface = {
    /* .get_name            = */ amd_backend_name,
    /* .free                = */ amd_backend_free,
    /* .set_tensor_async    = */ amd_backend_set_tensor_async,
    /* .get_tensor_async    = */ amd_backend_get_tensor_async,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ amd_backend_copy_tensor_async,
    /* .synchronize         = */ amd_backend_synchronize,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ amd_backend_graph_compute,
    /* .event_record        = */ amd_backend_event_record,
    /* .event_wait          = */ amd_backend_event_wait,
    /* .graph_optimize      = */ nullptr,
};

static const char * amd_device_name(ggml_backend_dev_t dev) {
    return amd_device_from_dev(dev)->name.c_str();
}

static const char * amd_device_description(ggml_backend_dev_t dev) {
    return amd_device_from_dev(dev)->description.c_str();
}

static void amd_device_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * device = amd_device_from_dev(dev);
    if (device == nullptr || !amd_hip_check(hipSetDevice(device->ordinal), "hipSetDevice") ||
        !amd_hip_check(hipMemGetInfo(free, total), "hipMemGetInfo")) {
        *free = 0;
        *total = 0;
    }
}

static enum ggml_backend_dev_type amd_device_type(ggml_backend_dev_t dev) {
    const auto * device = amd_device_from_dev(dev);
    return device != nullptr && device->props.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void amd_device_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    auto * device = amd_device_from_dev(dev);
    props->name = device->name.c_str();
    props->description = device->description.c_str();
    amd_device_memory(dev, &props->memory_free, &props->memory_total);
    props->type = amd_device_type(dev);
    props->device_id = device->device_id.empty() ? nullptr : device->device_id.c_str();
    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ true,
        /* .mmap_support         = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t amd_device_init(ggml_backend_dev_t dev, const char *) {
    auto * device = amd_device_from_dev(dev);
    if (device == nullptr || !amd_hip_check(hipSetDevice(device->ordinal), "hipSetDevice")) {
        return nullptr;
    }
    hipStream_t stream = nullptr;
    if (!amd_hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags")) {
        return nullptr;
    }
    auto * context = new amd_backend_context(device, stream);
    auto * backend = new ggml_backend {
        /* .guid    = */ []() -> ggml_guid_t {
            static ggml_guid guid = { 0x46, 0x6c, 0x61, 0x67, 0x4f, 0x53, 0x2d, 0x41, 0x4d, 0x44, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01 };
            return &guid;
        }(),
        /* .iface   = */ g_backend_iface,
        /* .device  = */ dev,
        /* .context = */ context,
    };
    {
        std::lock_guard<std::mutex> lock(*device->backends_mutex);
        device->backends.push_back(context);
    }
    return backend;
}

static ggml_backend_buffer_type_t amd_device_buffer_type(ggml_backend_dev_t dev) {
    return &amd_device_from_dev(dev)->buffer_type_iface;
}

static bool amd_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // Only advertise compute nodes with a validated AOT lowering.  Standalone
    // MUL is intentionally restricted to the exact contiguous/vector-repeat
    // layouts implemented by flagos_mul_f32; all other broadcasts stay on CPU.
    if (amd_tensor_is_view_op(op)) {
        return true;
    }
    return op != nullptr &&
        (amd_supports_copy_f32(amd_device_from_dev(dev), op) ||
         (op->op == GGML_OP_CONCAT && amd_supports_concat(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_SSM_CONV && amd_supports_ssm_conv(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_L2_NORM && amd_supports_l2_norm(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_GATED_DELTA_NET &&
          amd_supports_gated_delta_net(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_GET_ROWS && amd_supports_get_rows(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_MUL_MAT && amd_supports_quantized_mul_mat(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_ADD && amd_supports_add(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_MUL && amd_supports_mul(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_SCALE && amd_supports_scale(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_SET_ROWS && amd_supports_set_rows(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_RMS_NORM && amd_supports_rms_norm(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_ROPE &&
          (amd_supports_rope_neox(amd_device_from_dev(dev), op) ||
           amd_supports_mrope(amd_device_from_dev(dev), op))) ||
         (op->op == GGML_OP_UNARY &&
          (amd_supports_silu(amd_device_from_dev(dev), op) ||
           amd_supports_unary_f32(amd_device_from_dev(dev), op,
               GGML_UNARY_OP_SIGMOID, "flagos_sigmoid_f32") ||
           amd_supports_unary_f32(amd_device_from_dev(dev), op,
               GGML_UNARY_OP_SOFTPLUS, "flagos_softplus_f32"))) ||
         (op->op == GGML_OP_GLU && amd_supports_swiglu(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_SOFT_MAX && amd_supports_soft_max(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_FLASH_ATTN_EXT &&
          (amd_supports_flash_attn_decode(amd_device_from_dev(dev), op) ||
           amd_supports_flash_attn_prefill(amd_device_from_dev(dev), op))));
}

static bool amd_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return amd_buft_is_local(buft, amd_device_from_dev(dev));
}

static bool amd_device_offload_op(ggml_backend_dev_t, const ggml_tensor *) {
    return false;
}

static ggml_backend_event_t amd_event_new(ggml_backend_dev_t dev) {
    auto * device = amd_device_from_dev(dev);
    if (device == nullptr || !amd_hip_check(hipSetDevice(device->ordinal), "hipSetDevice")) {
        return nullptr;
    }
    hipEvent_t event = nullptr;
    if (!amd_hip_check(hipEventCreateWithFlags(&event, hipEventDisableTiming), "hipEventCreateWithFlags")) {
        return nullptr;
    }
    return new ggml_backend_event { dev, new amd_event_context { device->ordinal, event } };
}

static void amd_event_free(ggml_backend_dev_t, ggml_backend_event_t event) {
    auto * context = event == nullptr ? nullptr : static_cast<amd_event_context *>(event->context);
    if (context != nullptr) {
        amd_hip_check(hipSetDevice(context->device), "hipSetDevice");
        amd_hip_check(hipEventDestroy(context->event), "hipEventDestroy");
        delete context;
    }
    delete event;
}

static void amd_event_synchronize(ggml_backend_dev_t, ggml_backend_event_t event) {
    auto * context = event == nullptr ? nullptr : static_cast<amd_event_context *>(event->context);
    GGML_ASSERT(context != nullptr && context->event != nullptr);
    GGML_ASSERT(amd_hip_check(hipSetDevice(context->device), "hipSetDevice"));
    GGML_ASSERT(amd_hip_check(hipEventSynchronize(context->event), "hipEventSynchronize"));
}

static void amd_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * backend_context = backend == nullptr
        ? nullptr : static_cast<amd_backend_context *>(backend->context);
    auto * event_context = event == nullptr
        ? nullptr : static_cast<amd_event_context *>(event->context);
    GGML_ASSERT(backend_context != nullptr && backend_context->device != nullptr &&
        event_context != nullptr && event_context->event != nullptr &&
        event_context->device == backend_context->device->ordinal);
    std::lock_guard<std::mutex> execution_lock(backend_context->execution_mutex);
    GGML_ASSERT(amd_hip_check(hipSetDevice(backend_context->device->ordinal), "hipSetDevice"));
    GGML_ASSERT(amd_hip_check(hipEventRecord(event_context->event, backend_context->stream), "hipEventRecord"));
}

static void amd_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * backend_context = backend == nullptr
        ? nullptr : static_cast<amd_backend_context *>(backend->context);
    auto * event_context = event == nullptr
        ? nullptr : static_cast<amd_event_context *>(event->context);
    GGML_ASSERT(backend_context != nullptr && backend_context->device != nullptr &&
        event_context != nullptr && event_context->event != nullptr &&
        event_context->device == backend_context->device->ordinal);
    std::lock_guard<std::mutex> execution_lock(backend_context->execution_mutex);
    GGML_ASSERT(amd_hip_check(hipSetDevice(backend_context->device->ordinal), "hipSetDevice"));
    GGML_ASSERT(amd_hip_check(hipStreamWaitEvent(backend_context->stream, event_context->event, 0), "hipStreamWaitEvent"));
}

static const ggml_backend_device_i g_device_iface = {
    /* .get_name             = */ amd_device_name,
    /* .get_description      = */ amd_device_description,
    /* .get_memory           = */ amd_device_memory,
    /* .get_type             = */ amd_device_type,
    /* .get_props            = */ amd_device_props,
    /* .init_backend         = */ amd_device_init,
    /* .get_buffer_type      = */ amd_device_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ amd_device_supports_op,
    /* .supports_buft        = */ amd_device_supports_buft,
    /* .offload_op           = */ amd_device_offload_op,
    /* .event_new            = */ amd_event_new,
    /* .event_free           = */ amd_event_free,
    /* .event_synchronize    = */ amd_event_synchronize,
};

static std::filesystem::path amd_resolve_kernel_directory(
        const std::filesystem::path & root, const char * arch) {
    if (root.empty()) {
        return {};
    }
    if (std::filesystem::is_regular_file(root / "manifest.json")) {
        return root;
    }
    const std::filesystem::path target = root / arch;
    return std::filesystem::is_regular_file(target / "manifest.json") ? target : std::filesystem::path {};
}

static std::filesystem::path amd_module_directory() {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&g_device_iface), &module) == 0) {
        return {};
    }
    std::vector<wchar_t> path(512);
    for (;;) {
        const DWORD length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
        if (length == 0) {
            return {};
        }
        if (length < path.size()) {
            return std::filesystem::path(path.data(), path.data() + length).parent_path();
        }
        path.resize(path.size() * 2);
    }
#else
    Dl_info module_info {};
    if (dladdr(static_cast<const void *>(&g_device_iface), &module_info) == 0 ||
        module_info.dli_fname == nullptr) {
        return {};
    }
    return std::filesystem::path(module_info.dli_fname).parent_path();
#endif
}

static std::filesystem::path amd_kernel_directory(const char * arch) {
    const char * configured = std::getenv("FLAGOS_AMD_KERNEL_DIR");
    if (configured != nullptr && configured[0] != '\0') {
        const std::filesystem::path root(configured);
        if (const auto directory = amd_resolve_kernel_directory(root, arch); !directory.empty()) {
            return directory;
        }
        return root;
    }
#if defined(FLAGOS_AMD_RELOCATABLE_KERNEL_DIR)
    if (const auto module_directory = amd_module_directory(); !module_directory.empty()) {
        if (const auto directory = amd_resolve_kernel_directory(
                module_directory / FLAGOS_AMD_RELOCATABLE_KERNEL_DIR, arch); !directory.empty()) {
            return directory;
        }
    }
#endif
#if defined(FLAGOS_AMD_CONFIGURED_KERNEL_DIR)
    if (const auto directory = amd_resolve_kernel_directory(FLAGOS_AMD_CONFIGURED_KERNEL_DIR, arch);
            !directory.empty()) {
        return directory;
    }
#endif
#if defined(FLAGOS_AMD_INSTALLED_KERNEL_DIR)
    return amd_resolve_kernel_directory(FLAGOS_AMD_INSTALLED_KERNEL_DIR, arch);
#else
    return {};
#endif
}

static void amd_try_load_aot(amd_device_context & device) {
    if (device.aot_attempted) {
        return;
    }
    device.aot_attempted = true;
    const std::filesystem::path kernel_dir = amd_kernel_directory(device.arch.c_str());
    if (kernel_dir.empty()) {
        return;
    }
    auto registry = std::make_unique<flagos_amd::kernel_registry>();
    if (registry->initialize(kernel_dir, device.ordinal, device.arch)) {
        device.aot = std::move(registry);
        GGML_LOG_INFO("FlagOS AMD: loaded %zu AOT kernels for %s\n",
            device.aot->size(), device.arch.c_str());
    } else {
        GGML_LOG_WARN("FlagOS AMD: AOT kernel package is unavailable; compute ops remain disabled\n");
    }
}

static bool amd_probe_impl(ggml_backend_reg_t, bool log_device_count_error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_probed) {
        return !g_devices.empty();
    }
    int count = 0;
    const hipError_t device_count_result = hipGetDeviceCount(&count);
    if (!amd_hip_check(device_count_result, "hipGetDeviceCount", log_device_count_error) || count <= 0) {
        if (device_count_result != hipSuccess && log_device_count_error) {
            amd_log_device_access_hint();
        }
        return false;
    }

    g_probed = true;

    g_devices.reserve(static_cast<size_t>(count));
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        if (!amd_hip_check(hipSetDevice(ordinal), "hipSetDevice")) {
            continue;
        }
        amd_device_context device;
        device.ordinal = ordinal;
        if (!amd_hip_check(hipGetDeviceProperties(&device.props, ordinal), "hipGetDeviceProperties")) {
            continue;
        }
        device.arch = std::string(flagos_amd::base_gcn_arch(device.props.gcnArchName));
        hipUUID uuid {};
        if (amd_hip_check(hipDeviceGetUuid(&uuid, ordinal), "hipDeviceGetUuid")) {
            std::memcpy(device.uuid.data(), uuid.bytes, device.uuid.size());
        }
        char pci_id[64] = {};
        if (hipDeviceGetPCIBusId(pci_id, sizeof(pci_id), ordinal) == hipSuccess) {
            device.device_id = pci_id;
        }
        device.name = "FlagOS:AMD:" + std::to_string(ordinal);
        device.description = std::string(device.props.name) + " (" + device.props.gcnArchName + ")";
        device.device_iface.iface = g_device_iface;
        device.device_iface.context = nullptr;
        device.buffer_type_iface.iface = g_buffer_type_iface;
        device.buffer_type_iface.device = &device.device_iface;
        device.buffer_type_iface.context = nullptr;
        g_devices.push_back(std::move(device));
    }

    // std::vector reallocation is complete after reserve; wire contexts to the
    // stable elements and to the common device/buffer objects.
    for (auto & device : g_devices) {
        device.device_iface.context = &device;
        device.buffer_type_iface.device = &device.device_iface;
        device.buffer_type_iface.context = &device;
        amd_try_load_aot(device);
        amd_fill_device_profile(device, &device.profile);
        device.tuned_package = amd_is_tuned_package(&device);
        if (amd_auto_tuning_enabled(&device)) {
            GGML_LOG_INFO("FlagOS AMD: selected tuned profile %s for %s\n",
                device.aot->tuning_profile().c_str(), device.arch.c_str());
        }
    }
    return !g_devices.empty();
}

static bool amd_probe(ggml_backend_reg_t owner_reg) {
    return amd_probe_impl(owner_reg, true);
}

static size_t amd_device_count() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_devices.size();
}

static ggml_backend_dev_t amd_device_get(size_t index) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return index < g_devices.size() ? &g_devices[index].device_iface : nullptr;
}

static bool amd_device_identity(size_t index, flagos_device_identity * identity) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (identity == nullptr || index >= g_devices.size()) {
        return false;
    }
    const auto & device = g_devices[index];
    uint64_t uuid_hi = 0;
    uint64_t uuid_lo = 0;
    std::memcpy(&uuid_hi, device.uuid.data(), sizeof(uuid_hi));
    std::memcpy(&uuid_lo, device.uuid.data() + sizeof(uuid_hi), sizeof(uuid_lo));
    if (uuid_hi == 0 && uuid_lo == 0) {
        uuid_hi = amd_hash_device_identity(device, 1099511628211ULL);
        uuid_lo = amd_hash_device_identity(device, 0x9e3779b97f4a7c15ULL);
    }
    *identity = {
        /* .provider_id      = */ 0x414D440000000001ULL,
        /* .uuid_hi          = */ uuid_hi,
        /* .uuid_lo          = */ uuid_lo,
        /* .memory_domain_id = */ amd_memory_domain_id(device),
        /* .ordinal          = */ static_cast<uint32_t>(index),
    };
    return true;
}

static bool amd_device_caps(size_t index, flagos_device_caps * caps) {
    if (caps == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (index >= g_devices.size()) {
        return false;
    }
    const auto & device = g_devices[index];
    uint64_t execution = FLAGOS_EXECUTION_ASYNC_QUEUE |
        FLAGOS_EXECUTION_EVENTS | FLAGOS_EXECUTION_NATIVE_GRAPH;
    if (device.aot != nullptr) {
        execution |= FLAGOS_EXECUTION_AOT_MODULE;
    }
    *caps = {
        /* .kind      = */ flagos_provider_kind::gpu,
        // hipMalloc remains the allocation primitive for both dGPUs and APUs.
        // On an integrated AMD device the physical backing is system memory,
        // but the allocation is still a device pointer and is not host
        // dereferenceable, so do not advertise HOST_VISIBLE.
        /* .memory    = */ FLAGOS_MEMORY_DEVICE_LOCAL,
        /* .execution = */ execution,
        /* .aot_format= */ device.aot != nullptr ? "hsaco" : nullptr,
    };
    return true;
}

static bool amd_device_profile(size_t index, flagos_device_profile * profile) {
    if (profile == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (index >= g_devices.size()) {
        return false;
    }
    amd_fill_device_profile(g_devices[index], profile);
    return flagos_device_profile_is_valid(profile);
}

static bool amd_is_backend(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    static ggml_guid guid = { 0x46, 0x6c, 0x61, 0x67, 0x4f, 0x53, 0x2d, 0x41, 0x4d, 0x44, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01 };
    return ggml_guid_matches(backend->guid, &guid);
}

static bool amd_set_device(size_t index) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return index < g_devices.size() && amd_hip_check(hipSetDevice(g_devices[index].ordinal), "hipSetDevice");
}

static int amd_get_device() {
    int ordinal = -1;
    if (!amd_hip_check(hipGetDevice(&ordinal), "hipGetDevice")) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    for (size_t index = 0; index < g_devices.size(); ++index) {
        if (g_devices[index].ordinal == ordinal) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

static int amd_score() {
    // Dynamic backend loading calls the score entry point before the backend
    // registry is initialized.  Probe here so an AMD device can make the
    // FlagOS plugin selectable in GGML_BACKEND_DL mode.
    return amd_probe_impl(nullptr, false) ? 1 : 0;
}

} // namespace

const flagos_provider_v1 * flagos_amd_provider() {
    static const flagos_provider_v1 provider = {
        /* .api_version = */ FLAGOS_PROVIDER_API_VERSION,
        /* .struct_size = */ sizeof(flagos_provider_v1),
        /* .identity    = */ { 0x414D440000000001ULL, "amd", 1 },
        /* .probe       = */ amd_probe,
        /* .device_count= */ amd_device_count,
        /* .device_get  = */ amd_device_get,
        /* .device_identity = */ amd_device_identity,
        /* .device_caps = */ amd_device_caps,
        /* .is_backend  = */ amd_is_backend,
        /* .set_device  = */ amd_set_device,
        /* .get_device  = */ amd_get_device,
        /* .score       = */ amd_score,
        /* .get_device_profile = */ amd_device_profile,
    };
    return &provider;
}
