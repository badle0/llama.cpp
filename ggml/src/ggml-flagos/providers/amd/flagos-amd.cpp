#include "flagos-amd-api.h"
#include "flagos-amd-aot.h"

#include "../../../ggml-backend-impl.h"
#include "../../../ggml-impl.h"
#include "../../flagos-graph-plan.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#if !defined(_WIN32)
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
};

struct amd_event_context {
    int device = -1;
    hipEvent_t event = nullptr;
};

struct amd_graph_capture_entry {
    uint64_t fingerprint = 0;
    std::vector<uintptr_t> pointers;
    hipGraphExec_t executable = nullptr;
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
        amd_hip_check(hipGraphDestroy(graph), "hipGraphDestroy");
        graph = nullptr;
        return true;
    }
};

struct amd_dequant_cache_entry {
    const ggml_tensor * tensor = nullptr;
    const void * source = nullptr;
    flagos_quantized_matmul_kind kind = flagos_quantized_matmul_kind::none;
    int k = 0;
    int rows = 0;
    void * f16_data = nullptr;
    size_t bytes = 0;
};

struct amd_backend_context {
    amd_device_context * device = nullptr;
    hipStream_t stream = nullptr;
    flagos_graph_plan_cache graph_plans;
    uint64_t graph_plan_config_key = 0;
    std::vector<amd_graph_capture_entry> graph_captures;
    std::vector<amd_dequant_cache_entry> dequant_cache;
    size_t dequant_cache_bytes = 0;
    size_t dequant_cache_limit_bytes = 0;
    bool dequant_cache_limit_logged = false;
    uint64_t graph_capture_tick = 0;
    // A scheduler graph may keep its structure while rotating temporary
    // tensor allocations between executions.  Do not pay capture cost until
    // the exact data-pointer set has been observed twice consecutively.
    uint64_t graph_capture_candidate_fingerprint = 0;
    std::vector<uintptr_t> graph_capture_candidate_pointers;
    bool graph_capture_candidate_valid = false;
    uint64_t profile_dump_sync_interval = 0;
    uint64_t profile_sync_count = 0;

    struct stats_t {
        std::atomic<uint64_t> kernel_launches { 0 };
        std::atomic<uint64_t> direct_ops { 0 };
        std::atomic<uint64_t> fusion_steps { 0 };
        std::atomic<uint64_t> fusion_rms_norm_mul { 0 };
        std::atomic<uint64_t> fusion_add_rms_norm_mul { 0 };
        std::atomic<uint64_t> fusion_rope_kv_store { 0 };
        std::atomic<uint64_t> fusion_flash_attn_decode { 0 };
        std::atomic<uint64_t> fusion_flash_attn_prefill { 0 };
        std::atomic<uint64_t> fusion_ffn_swiglu { 0 };
        std::atomic<uint64_t> q4_matmul { 0 };
        std::atomic<uint64_t> q6_matmul { 0 };
        std::atomic<uint64_t> q4_matmul_batched { 0 };
        std::atomic<uint64_t> q6_matmul_batched { 0 };
        std::atomic<uint64_t> q4_matmul_tiled { 0 };
        std::atomic<uint64_t> q6_matmul_tiled { 0 };
        std::atomic<uint64_t> weight_dequantizations { 0 };
        std::atomic<uint64_t> f16_matmul_batched { 0 };
        std::atomic<uint64_t> q4_get_rows { 0 };
        std::atomic<uint64_t> q6_get_rows { 0 };
        std::atomic<uint64_t> mul { 0 };
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
        std::atomic<uint64_t> graph_capture_failures { 0 };
    } stats;

    amd_backend_context(amd_device_context * device_, hipStream_t stream_):
        device(device_), stream(stream_), graph_plans(32) {
        graph_captures.reserve(8);
        dequant_cache.reserve(64);
        // F16 weights are a deliberately opt-in optimization.  Keep its
        // residency bounded by default so enabling it on a larger model does
        // not silently consume all UMA/VRAM.  Set the variable to 0 to retain
        // the historical unlimited behavior, or provide a larger value when
        // the device has enough memory.
        const char * limit_mb = std::getenv("FLAGOS_AMD_DEQUANT_CACHE_MAX_MB");
        if (limit_mb == nullptr || limit_mb[0] == '\0') {
            dequant_cache_limit_bytes = size_t(8192) * 1024 * 1024;
        } else {
            char * end = nullptr;
            const unsigned long long parsed = std::strtoull(limit_mb, &end, 10);
            if (end != limit_mb && *end == '\0' && parsed <= SIZE_MAX / (1024ULL * 1024ULL)) {
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
            const unsigned long long parsed = std::strtoull(dump_interval, &end, 10);
            if (end != dump_interval && *end == '\0') {
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

static bool amd_graph_capture_enabled() {
    return amd_env_enabled("FLAGOS_AMD_GRAPH_CAPTURE");
}

// Multi-node and composite attention lowerings remain opt-in until their
// layouts have been validated against every llama graph variant.  Direct
// AOT operators stay enabled by default, while this switch provides a safe
// staging point for hardware-specific fusion bring-up.
static bool amd_experimental_fusions_enabled() {
    return amd_env_enabled("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS");
}

// During bring-up, allow individual fusions to be enabled without changing
// the provider ABI or rebuilding the AOT package.  An unset selector keeps
// the historical behaviour (all experimental fusions); a comma-separated
// selector makes numerical bisects reproducible, for example:
//   FLAGOS_AMD_FUSIONS=rope_kv_store,flash_attn_decode
static bool amd_fusion_enabled(const char * name) {
    if (!amd_experimental_fusions_enabled() || name == nullptr) {
        return false;
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

static uint64_t amd_fusion_config_key() {
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
    add(std::getenv("FLAGOS_AMD_Q4_GEMV_NARROW"));
    return hash;
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

static std::vector<uintptr_t> amd_graph_capture_pointers(const ggml_cgraph * cgraph) {
    std::vector<uintptr_t> pointers;
    if (cgraph == nullptr) {
        return pointers;
    }
    pointers.reserve(static_cast<size_t>(cgraph->n_nodes) * (GGML_MAX_SRC + 1));
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        pointers.push_back(reinterpret_cast<uintptr_t>(node == nullptr ? nullptr : node->data));
        if (node == nullptr) {
            continue;
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            pointers.push_back(reinterpret_cast<uintptr_t>(node->src[j] == nullptr ? nullptr : node->src[j]->data));
        }
    }
    return pointers;
}

static void amd_clear_graph_captures(amd_backend_context * context);

static amd_graph_capture_entry * amd_find_graph_capture(
        amd_backend_context * context, uint64_t fingerprint, const std::vector<uintptr_t> & pointers) {
    if (context == nullptr) {
        return nullptr;
    }
    for (auto & entry : context->graph_captures) {
        if (entry.fingerprint == fingerprint && entry.pointers == pointers && entry.executable != nullptr) {
            entry.last_used = ++context->graph_capture_tick;
            return &entry;
        }
    }
    return nullptr;
}

static void amd_store_graph_capture(
        amd_backend_context * context, uint64_t fingerprint, std::vector<uintptr_t> pointers,
        hipGraphExec_t executable) {
    if (context == nullptr || executable == nullptr) {
        return;
    }
    constexpr size_t capacity = 8;
    if (context->graph_captures.size() >= capacity) {
        auto lru = std::min_element(context->graph_captures.begin(), context->graph_captures.end(),
            [](const amd_graph_capture_entry & left, const amd_graph_capture_entry & right) {
                return left.last_used < right.last_used;
            });
        if (lru != context->graph_captures.end()) {
            amd_hip_check(hipGraphExecDestroy(lru->executable), "hipGraphExecDestroy");
            context->graph_captures.erase(lru);
        }
    }
    context->graph_captures.push_back({
        fingerprint, std::move(pointers), executable, ++context->graph_capture_tick});
}

static void amd_log_stats(const amd_backend_context * context) {
    if (context == nullptr || !amd_env_enabled("FLAGOS_LOG_KERNELS")) {
        return;
    }
    const auto & s = context->stats;
    GGML_LOG_INFO(
        "FlagOS AMD kernel stats: launches=%llu direct=%llu fusions=%llu "
        "q4_matmul=%llu q6_matmul=%llu q4_batched=%llu q6_batched=%llu "
        "q4_tiled=%llu q6_tiled=%llu "
        "weight_dequantizations=%llu f16_batched=%llu "
        "q4_get_rows=%llu q6_get_rows=%llu rope=%llu rope_kv_store=%llu "
        "mul=%llu flash_decode=%llu flash_prefill=%llu soft_max=%llu "
        "dequant_cache_mib=%zu/%zu "
        "host_to_device=%llu device_to_device=%llu "
        "plan_builds=%llu plan_direct=%llu plan_patterns=%llu "
        "fusion_rms_mul=%llu fusion_add_rms_mul=%llu fusion_rope_store=%llu "
        "fusion_flash_decode=%llu fusion_flash_prefill=%llu "
        "fusion_ffn_swiglu=%llu "
        "graph_captures=%llu graph_replays=%llu graph_capture_failures=%llu\n",
        (unsigned long long) s.kernel_launches.load(),
        (unsigned long long) s.direct_ops.load(),
        (unsigned long long) s.fusion_steps.load(),
        (unsigned long long) s.q4_matmul.load(),
        (unsigned long long) s.q6_matmul.load(),
        (unsigned long long) s.q4_matmul_batched.load(),
        (unsigned long long) s.q6_matmul_batched.load(),
        (unsigned long long) s.q4_matmul_tiled.load(),
        (unsigned long long) s.q6_matmul_tiled.load(),
        (unsigned long long) s.weight_dequantizations.load(),
        (unsigned long long) s.f16_matmul_batched.load(),
        (unsigned long long) s.q4_get_rows.load(),
        (unsigned long long) s.q6_get_rows.load(),
        (unsigned long long) s.rope.load(),
        (unsigned long long) s.rope_kv_store.load(),
        (unsigned long long) s.mul.load(),
        (unsigned long long) s.flash_attn_decode.load(),
        (unsigned long long) s.flash_attn_prefill.load(),
        (unsigned long long) s.soft_max.load(),
        context->dequant_cache_bytes / (1024 * 1024),
        context->dequant_cache_limit_bytes / (1024 * 1024),
        (unsigned long long) s.host_to_device_copies.load(),
        (unsigned long long) s.device_to_device_copies.load(),
        (unsigned long long) s.graph_plans_built.load(),
        (unsigned long long) s.graph_plan_direct_steps.load(),
        (unsigned long long) s.graph_plan_pattern_steps.load(),
        (unsigned long long) s.fusion_rms_norm_mul.load(),
        (unsigned long long) s.fusion_add_rms_norm_mul.load(),
        (unsigned long long) s.fusion_rope_kv_store.load(),
        (unsigned long long) s.fusion_flash_attn_decode.load(),
        (unsigned long long) s.fusion_flash_attn_prefill.load(),
        (unsigned long long) s.fusion_ffn_swiglu.load(),
        (unsigned long long) s.graph_captures.load(),
        (unsigned long long) s.graph_replays.load(),
        (unsigned long long) s.graph_capture_failures.load());
}

struct amd_device_context {
    int ordinal = -1;
    hipDeviceProp_t props {};
    std::array<uint8_t, 16> uuid {};
    std::string name;
    std::string description;
    std::string device_id;
    ggml_backend_device device_iface {};
    ggml_backend_buffer_type buffer_type_iface {};
    std::unique_ptr<flagos_amd::kernel_registry> aot;
    bool aot_attempted = false;
    flagos_device_profile profile {};
};

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
    profile->memory_domain_id = 0x414D440000000001ULL;
    profile->vendor = "AMD";
    profile->architecture = "GCN-compatible GPU";
    profile->target = device.props.gcnArchName;
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
    context->graph_capture_candidate_pointers.clear();
    context->graph_capture_candidate_fingerprint = 0;
    context->graph_capture_candidate_valid = false;
}

static bool amd_prefill_f16_gemm_enabled() {
    return amd_env_enabled("FLAGOS_AMD_PREFILL_F16_GEMM");
}

static const flagos_kernel_variant AMD_GROUPED_F16_GEMM_VARIANT = {
    "f16-grouped",
    { FLAGOS_FEATURE_WAVE32, 0, 0, 32, "gfx1150", flagos_engine_bit(flagos_engine_kind::gpu) },
    0, 0, 256, 0, 0, 0, 0, flagos_support_state::validated,
};

static const flagos_kernel_variant AMD_GROUPED_F16_FFN_VARIANT = {
    "ffn-swiglu-f16-grouped",
    { FLAGOS_FEATURE_WAVE32, 0, 0, 32, "gfx1150", flagos_engine_bit(flagos_engine_kind::gpu) },
    0, 0, 32, 0, 0, 0, 0, flagos_support_state::validated,
};

static bool amd_grouped_f16_gemm_enabled(const amd_device_context * device, int columns) {
    if (device == nullptr || columns < 256) {
        return false;
    }
    const char * configured = std::getenv("FLAGOS_AMD_GROUPED_F16_GEMM");
    if (configured != nullptr && (configured[0] == '\0' || std::strcmp(configured, "0") == 0)) {
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

static bool amd_q4_gemv_narrow_enabled() {
    return amd_env_enabled("FLAGOS_AMD_Q4_GEMV_NARROW");
}

static unsigned int amd_q4_gemv_narrow_row_tile() {
    const char * value = std::getenv("FLAGOS_AMD_Q4_GEMV_NARROW");
    return value != nullptr && std::strcmp(value, "8") == 0 ? 8U : 4U;
}

static bool amd_quant_tiled_enabled() {
    return amd_env_enabled("FLAGOS_AMD_QUANT_TILED_GEMM");
}

static const char * amd_quantized_batched_kernel(const amd_device_context * device,
                                                 flagos_quantized_matmul_kind kind) {
    const bool q4 = kind == flagos_quantized_matmul_kind::q4_k;
    if (amd_quant_tiled_enabled()) {
        const char * tiled = q4 ? "flagos_mul_mat_q4_k_f32_tiled" : "flagos_mul_mat_q6_k_f32_tiled";
        if (device != nullptr && device->aot != nullptr && device->aot->find(tiled) != nullptr) {
            return tiled;
        }
    }
    return q4 ? "flagos_mul_mat_q4_k_f32_batched" : "flagos_mul_mat_q6_k_f32_batched";
}

static void amd_release_dequant_cache(amd_backend_context * context) {
    if (context == nullptr || context->device == nullptr) {
        return;
    }
    amd_hip_check(hipSetDevice(context->device->ordinal), "hipSetDevice");
    for (auto & entry : context->dequant_cache) {
        if (entry.f16_data != nullptr) {
            amd_hip_check(hipFree(entry.f16_data), "hipFree dequantized weight");
            entry.f16_data = nullptr;
        }
    }
    context->dequant_cache.clear();
    context->dequant_cache_bytes = 0;
    context->dequant_cache_limit_logged = false;
}

static bool amd_dequant_cache_contains(
        const amd_backend_context * context, const ggml_tensor * weight,
        const flagos_quantized_matmul_signature & signature) {
    if (context == nullptr || weight == nullptr) {
        return false;
    }
    for (const auto & entry : context->dequant_cache) {
        if (entry.tensor == weight && entry.source == weight->data &&
            entry.kind == signature.weight_kind && entry.k == signature.k &&
            entry.rows == signature.rows) {
            return true;
        }
    }
    return false;
}

static bool amd_dequant_cache_entry_bytes(
        const flagos_quantized_matmul_signature & signature, size_t * bytes) {
    if (bytes == nullptr || signature.k <= 0 || signature.rows <= 0 ||
        signature.k % 256 != 0) {
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

// Query-time capacity check.  The FFN plan covers all three nodes, so an
// allocation failure during execution would otherwise turn a recoverable
// cache miss into a graph-wide failure.  Declining the pattern before the
// plan is cached lets the normal quantized projection and standalone GLU
// paths execute instead.
static bool amd_ffn_dequant_cache_can_fit(
        const amd_backend_context * context,
        const ggml_tensor * first_projection,
        const flagos_quantized_matmul_signature & first_signature,
        const ggml_tensor * second_projection,
        const flagos_quantized_matmul_signature & second_signature) {
    if (context == nullptr) {
        return false;
    }
    size_t additional = 0;
    const auto account = [&](const ggml_tensor * projection,
                             const flagos_quantized_matmul_signature & signature) {
        if (projection == nullptr || projection->src[0] == nullptr) {
            return false;
        }
        if (amd_dequant_cache_contains(context, projection->src[0], signature)) {
            return true;
        }
        size_t bytes = 0;
        if (!amd_dequant_cache_entry_bytes(signature, &bytes) ||
            additional > SIZE_MAX - bytes) {
            return false;
        }
        additional += bytes;
        return true;
    };
    if (!account(first_projection, first_signature)) {
        return false;
    }
    const bool same_weight = first_projection != nullptr && second_projection != nullptr &&
        first_projection->src[0] == second_projection->src[0] &&
        first_signature.weight_kind == second_signature.weight_kind &&
        first_signature.k == second_signature.k && first_signature.rows == second_signature.rows;
    if (!same_weight && !account(second_projection, second_signature)) {
        return false;
    }
    if (context->dequant_cache_limit_bytes == 0) {
        return true;
    }
    return additional <= context->dequant_cache_limit_bytes &&
        context->dequant_cache_bytes <=
            context->dequant_cache_limit_bytes - additional;
}

static bool amd_f16_graph_cache_ready(
        const amd_backend_context * context, const ggml_cgraph * cgraph) {
    if (context == nullptr || cgraph == nullptr || !amd_prefill_f16_gemm_enabled()) {
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
    if (context == nullptr || context->device == nullptr || context->device->aot == nullptr ||
        weight == nullptr || weight->data == nullptr || signature.k <= 0 || signature.rows <= 0 ||
        signature.k % 256 != 0 || !ggml_is_contiguous(weight)) {
        return nullptr;
    }
    for (const auto & entry : context->dequant_cache) {
        if (entry.tensor == weight && entry.source == weight->data &&
            entry.kind == signature.weight_kind &&
            entry.k == signature.k && entry.rows == signature.rows) {
            return entry.f16_data;
        }
    }
    const char * kernel_name = signature.weight_kind == flagos_quantized_matmul_kind::q4_k
        ? "flagos_dequant_q4_k_f16" : signature.weight_kind == flagos_quantized_matmul_kind::q6_k
        ? "flagos_dequant_q6_k_f16" : nullptr;
    if (kernel_name == nullptr || context->device->aot->find(kernel_name) == nullptr) {
        return nullptr;
    }
    size_t bytes = 0;
    if (!amd_dequant_cache_entry_bytes(signature, &bytes)) {
        return nullptr;
    }
    const uint64_t elements = static_cast<uint64_t>(signature.k) *
        static_cast<uint64_t>(signature.rows);
    if (context->dequant_cache_limit_bytes != 0 &&
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
    void * f16_data = nullptr;
    if (!amd_hip_check(hipMalloc(&f16_data, bytes),
            "hipMalloc dequantized weight")) {
        return nullptr;
    }
    void * weights_u8 = weight->data;
    void * weights_f16 = weight->data;
    int blocks = static_cast<int>(elements / 256);
    flagos_amd::kernel_arguments arguments = { &weights_u8, &weights_f16, &f16_data };
    if (!context->device->aot->launch(kernel_name, context->stream,
            static_cast<unsigned int>(blocks), 1, 1, arguments)) {
        amd_hip_check(hipFree(f16_data), "hipFree failed dequantized weight");
        return nullptr;
    }
    context->dequant_cache.push_back({ weight, weight->data, signature.weight_kind,
        static_cast<int>(signature.k), static_cast<int>(signature.rows), f16_data, bytes });
    context->dequant_cache_bytes += bytes;
    context->stats.weight_dequantizations.fetch_add(1, std::memory_order_relaxed);
    return f16_data;
}

static std::mutex g_mutex;
static bool g_probed = false;
static std::vector<amd_device_context> g_devices;

static amd_device_context * amd_device_from_dev(ggml_backend_dev_t dev) {
    return dev == nullptr ? nullptr : static_cast<amd_device_context *>(dev->context);
}

static amd_device_context * amd_device_from_buft(ggml_backend_buffer_type_t buft) {
    return buft == nullptr ? nullptr : static_cast<amd_device_context *>(buft->context);
}

static amd_buffer_context * amd_buffer_from_buffer(ggml_backend_buffer_t buffer) {
    return buffer == nullptr ? nullptr : static_cast<amd_buffer_context *>(buffer->context);
}

static const amd_buffer_context * amd_buffer_from_tensor(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return nullptr;
    }
    return amd_buffer_from_buffer(tensor->buffer);
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

    auto * context = new amd_buffer_context { device, data, size };
    static const ggml_backend_buffer_i iface = {
        /* .free_buffer    = */ [](ggml_backend_buffer_t buffer) {
            auto * context = amd_buffer_from_buffer(buffer);
            if (context != nullptr) {
                if (context->data != nullptr) {
                    amd_hip_check(hipSetDevice(context->device->ordinal), "hipSetDevice");
                    amd_hip_check(hipFree(context->data), "hipFree");
                }
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
            GGML_ASSERT(context != nullptr && tensor != nullptr && offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset);
            amd_hip_check(hipSetDevice(context->device->ordinal), "hipSetDevice");
            GGML_ASSERT(amd_hip_check(hipMemset(static_cast<char *>(tensor->data) + offset, value, size), "hipMemset"));
        },
        /* .set_tensor     = */ [](ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
            auto * context = amd_buffer_from_buffer(buffer);
            GGML_ASSERT(context != nullptr && tensor != nullptr && offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset);
            amd_hip_check(hipSetDevice(context->device->ordinal), "hipSetDevice");
            GGML_ASSERT(amd_hip_check(hipMemcpy(
                static_cast<char *>(tensor->data) + offset, data, size, hipMemcpyHostToDevice), "hipMemcpy H2D"));
        },
        /* .get_tensor     = */ [](ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
            auto * context = amd_buffer_from_buffer(buffer);
            GGML_ASSERT(context != nullptr && tensor != nullptr && offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset);
            amd_hip_check(hipSetDevice(context->device->ordinal), "hipSetDevice");
            GGML_ASSERT(amd_hip_check(hipMemcpy(
                data, static_cast<const char *>(tensor->data) + offset, size, hipMemcpyDeviceToHost), "hipMemcpy D2H"));
        },
        /* .set_tensor_2d  = */ nullptr,
        /* .get_tensor_2d  = */ nullptr,
        /* .cpy_tensor     = */ [](ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
            auto * dst_context = amd_buffer_from_buffer(buffer);
            const auto * src_context = amd_buffer_from_tensor(src);
            if (dst_context == nullptr || src_context == nullptr || dst_context->device != src_context->device ||
                src == nullptr || dst == nullptr || src->data == nullptr || dst->data == nullptr) {
                return false;
            }
            const size_t size = ggml_nbytes(src);
            amd_hip_check(hipSetDevice(dst_context->device->ordinal), "hipSetDevice");
            return amd_hip_check(hipMemcpy(dst->data, src->data, size, hipMemcpyDeviceToDevice), "hipMemcpy D2D");
        },
        /* .clear         = */ [](ggml_backend_buffer_t buffer, uint8_t value) {
            auto * context = amd_buffer_from_buffer(buffer);
            GGML_ASSERT(context != nullptr);
            amd_hip_check(hipSetDevice(context->device->ordinal), "hipSetDevice");
            GGML_ASSERT(amd_hip_check(hipMemset(context->data, value, context->size), "hipMemset buffer"));
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

static const char * amd_backend_name(ggml_backend_t) {
    return "FlagOS_AMD";
}

static void amd_backend_free(ggml_backend_t backend) {
    auto * context = backend == nullptr ? nullptr : static_cast<amd_backend_context *>(backend->context);
    if (context != nullptr) {
        amd_hip_check(hipSetDevice(context->device->ordinal), "hipSetDevice");
        amd_log_stats(context);
        if (context->device->aot != nullptr) {
            context->device->aot->log_profile();
        }
        amd_clear_graph_captures(context);
        amd_release_dequant_cache(context);
        amd_hip_check(hipStreamDestroy(context->stream), "hipStreamDestroy");
        delete context;
    }
}

static void amd_backend_set_tensor_async(
        ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * backend_context = static_cast<amd_backend_context *>(backend->context);
    auto * buffer_context = amd_buffer_from_buffer(tensor->buffer);
    GGML_ASSERT(buffer_context != nullptr && buffer_context->device == backend_context->device);
    amd_hip_check(hipSetDevice(backend_context->device->ordinal), "hipSetDevice");
    GGML_ASSERT(amd_hip_check(hipMemcpyAsync(
        static_cast<char *>(tensor->data) + offset, data, size,
        hipMemcpyHostToDevice, backend_context->stream), "hipMemcpyAsync H2D"));
}

static void amd_backend_get_tensor_async(
        ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * backend_context = static_cast<amd_backend_context *>(backend->context);
    const auto * buffer_context = amd_buffer_from_buffer(tensor->buffer);
    GGML_ASSERT(buffer_context != nullptr && buffer_context->device == backend_context->device);
    amd_hip_check(hipSetDevice(backend_context->device->ordinal), "hipSetDevice");
    GGML_ASSERT(amd_hip_check(hipMemcpyAsync(
        data, static_cast<const char *>(tensor->data) + offset, size,
        hipMemcpyDeviceToHost, backend_context->stream), "hipMemcpyAsync D2H"));
}

static bool amd_backend_copy_tensor_async(
        ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    auto * dst_context = static_cast<amd_backend_context *>(backend_dst->context);
    const auto * dst_buffer = amd_buffer_from_tensor(dst);
    if (dst_context == nullptr || dst_buffer == nullptr || dst_buffer->device != dst_context->device ||
        src == nullptr || dst == nullptr || src->data == nullptr || dst->data == nullptr ||
        ggml_nbytes(src) != ggml_nbytes(dst)) {
        GGML_UNUSED(backend_src);
        return false;
    }
    amd_hip_check(hipSetDevice(dst_context->device->ordinal), "hipSetDevice");

    // The scheduler uses this callback for partition boundaries as well as
    // same-device copies.  CPU fallback nodes expose host-backed tensors, so
    // blindly casting their buffer context to amd_buffer_context would either
    // reject a valid transfer or interpret foreign state as a HIP device.
    // Inspect the buffer type first and select the correct HIP direction.
    const ggml_backend_buffer_type_t src_buft = ggml_backend_buffer_get_type(src->buffer);
    const bool src_is_local = src_buft == &dst_context->device->buffer_type_iface;
    const bool src_is_host = src_buft != nullptr && ggml_backend_buft_is_host(src_buft);
    if (!src_is_local && !src_is_host) {
        return false;
    }
    const hipMemcpyKind kind = src_is_local ? hipMemcpyDeviceToDevice : hipMemcpyHostToDevice;
    // CPU graph temporaries may be recycled as soon as its split returns.  A
    // HIP async H2D copy must therefore be ordered after the source backend,
    // otherwise the DMA engine can observe a buffer that the CPU has already
    // reused.  Same-device copies are already ordered by the shared stream.
    if (!src_is_local && backend_src != nullptr) {
        ggml_backend_synchronize(backend_src);
    }
    if (!src_is_local) {
        // Host model/compute buffers are ordinary pageable allocations on the
        // current GGML CPU provider.  HIP's async pageable path may return
        // before its staging copy has consumed the source, so use a blocking
        // H2D transfer after synchronizing the CPU producer.
        const bool copied = amd_hip_check(hipMemcpy(dst->data, src->data, ggml_nbytes(src), kind),
            "hipMemcpy H2D");
        if (copied) {
            dst_context->stats.host_to_device_copies.fetch_add(1, std::memory_order_relaxed);
        }
        return copied;
    }
    const bool copied = amd_hip_check(hipMemcpyAsync(
        dst->data, src->data, ggml_nbytes(src), kind, dst_context->stream),
        "hipMemcpyAsync D2D");
    if (copied) {
        dst_context->stats.device_to_device_copies.fetch_add(1, std::memory_order_relaxed);
    }
    return copied;
}

static void amd_backend_synchronize(ggml_backend_t backend) {
    auto * context = static_cast<amd_backend_context *>(backend->context);
    amd_hip_check(hipSetDevice(context->device->ordinal), "hipSetDevice");
    GGML_ASSERT(amd_hip_check(hipStreamSynchronize(context->stream), "hipStreamSynchronize"));
    if (context->profile_dump_sync_interval != 0 && context->device->aot != nullptr &&
        ++context->profile_sync_count % context->profile_dump_sync_interval == 0) {
        GGML_LOG_INFO("FlagOS AMD kernel profile snapshot: synchronize=%llu\n",
            static_cast<unsigned long long>(context->profile_sync_count));
        context->device->aot->log_profile();
    }
}

static void amd_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event);
static void amd_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event);

static bool amd_tensor_is_contiguous_f32(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(tensor);
}

static bool amd_supports_rope_neox(const amd_device_context * device, const ggml_tensor * op) {
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_ROPE || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->src[2] != nullptr || device->aot->find("flagos_rope_neox_f32") == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
        op->src[1]->type != GGML_TYPE_I32 || !ggml_are_same_shape(op, op->src[0]) ||
        !ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) ||
        !ggml_is_contiguous(op->src[1]) || op->ne[0] <= 0 || op->ne[0] % 2 != 0 ||
        op->ne[1] <= 0 || op->ne[2] <= 0 || op->ne[3] != 1 ||
        op->src[0]->ne[2] != op->src[1]->ne[0] || op->src[0]->ne[3] != 1 ||
        ggml_nelements(op) <= 0 || ggml_nelements(op) > INT32_MAX ||
        op->ne[0] > INT32_MAX || op->ne[1] > INT32_MAX || op->ne[2] > INT32_MAX) {
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

static bool amd_supports_silu(const amd_device_context * device, const ggml_tensor * op) {
    return device != nullptr && device->aot != nullptr && op != nullptr &&
        op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_SILU &&
        device->aot->find("flagos_silu_f32") != nullptr && op->src[0] != nullptr &&
        amd_tensor_is_contiguous_f32(op) && amd_tensor_is_contiguous_f32(op->src[0]) &&
        ggml_are_same_shape(op, op->src[0]) && ggml_nelements(op) <= INT32_MAX;
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
        !ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) ||
        !ggml_is_contiguous(op->src[1]) || ggml_nelements(op->src[0]) > INT32_MAX ||
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
    return true;
}

static bool amd_supports_get_rows(const amd_device_context * device, const ggml_tensor * op) {
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_GET_ROWS || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_I32 ||
        (op->src[0]->type != GGML_TYPE_Q4_K && op->src[0]->type != GGML_TYPE_Q6_K) ||
        op->ne[0] <= 0 || op->ne[0] % 256 != 0 || op->ne[1] <= 0 ||
        op->ne[2] != 1 || op->ne[3] != 1 || op->src[0]->ne[1] <= 0 ||
        op->src[0]->ne[2] != 1 || op->src[0]->ne[3] != 1 ||
        ggml_nelements(op->src[1]) != op->ne[1] || !ggml_is_contiguous(op) ||
        !ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1]) ||
        op->ne[0] > INT32_MAX || op->ne[1] > INT32_MAX) {
        return false;
    }
    const char * kernel = op->src[0]->type == GGML_TYPE_Q4_K
        ? "flagos_get_rows_q4_k_f32" : "flagos_get_rows_q6_k_f32";
    return device->aot->find(kernel) != nullptr;
}

static bool amd_supports_quantized_mul_mat(const amd_device_context * device, const ggml_tensor * op) {
    if (device == nullptr || device->aot == nullptr || op == nullptr) {
        return false;
    }
    flagos_quantized_matmul_signature signature;
    if (!flagos_describe_quantized_matmul(op, &signature) || signature.k > INT32_MAX ||
        signature.rows > INT32_MAX || signature.columns > INT32_MAX ||
        signature.k > INT32_MAX / std::max<int64_t>(signature.columns, 1)) {
        return false;
    }
    const char * kernel = nullptr;
    if (signature.weight_kind == flagos_quantized_matmul_kind::q4_k) {
        kernel = signature.columns == 1 ? "flagos_mul_mat_q4_k_f32"
            : amd_quantized_batched_kernel(device, signature.weight_kind);
    } else {
        kernel = signature.columns == 1 ? "flagos_mul_mat_q6_k_f32"
            : amd_quantized_batched_kernel(device, signature.weight_kind);
    }
    return device->aot->find(kernel) != nullptr;
}

// The FFN fusion consumes the same quantized projections accepted by the
// opt-in F16 dequant-cache path, but combines both projections and the
// terminal split SwiGLU into one launch.  Keep this capability deliberately
// narrow: the Triton ABI assumes two row-major F16 weight caches, one
// contiguous F32 activation matrix, and a contiguous F32 output matrix.
static bool amd_supports_ffn_swiglu_f16(const amd_device_context * device,
                                        const ggml_tensor * gate,
                                        const ggml_tensor * up,
                                        const ggml_tensor * glu) {
    if (device == nullptr || device->aot == nullptr || gate == nullptr || up == nullptr || glu == nullptr ||
        !amd_fusion_enabled("ffn_swiglu") || !amd_prefill_f16_gemm_enabled() ||
        device->aot->find("flagos_ffn_swiglu_f16_f32_batched") == nullptr ||
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
        !flagos_describe_quantized_matmul(up, &up_signature)) {
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
        return device->aot->find(kind == flagos_quantized_matmul_kind::q4_k
            ? "flagos_dequant_q4_k_f16" : "flagos_dequant_q6_k_f16") != nullptr;
    };
    if (!has_dequant(gate_signature.weight_kind) || !has_dequant(up_signature.weight_kind)) {
        return false;
    }

    // The fused launch overwrites only the terminal GLU allocation.  Reject
    // any known alias with an input, an intermediate projection, or a weight;
    // otherwise the first tile could clobber data still needed by a later
    // tile.  Null data is accepted during early graph construction, while
    // execution-time tensors are always checked by the launcher below.
    if (glu->data != nullptr) {
        if (glu->data == activation->data || glu->data == gate->data || glu->data == up->data ||
            glu->data == gate_weights->data || glu->data == up_weights->data) {
            return false;
        }
    }
    if (gate->data != nullptr && up->data != nullptr && gate->data == up->data) {
        return false;
    }
    return true;
}

// Decode uses the same provider-neutral FFN/SwiGLU graph pattern as the F16
// prefill path, but keeps the Q4_K weights packed.  The AMD implementation
// fuses both quantized GEMVs and the terminal activation, shares the single
// F32 activation vector, and materializes only the terminal GLU output.
static bool amd_supports_ffn_swiglu_q4_decode(const amd_device_context * device,
                                              const ggml_tensor * gate,
                                              const ggml_tensor * up,
                                              const ggml_tensor * glu) {
    if (device == nullptr || device->aot == nullptr || gate == nullptr || up == nullptr || glu == nullptr ||
        !amd_fusion_enabled("ffn_swiglu") ||
        gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT ||
        glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU ||
        glu->src[0] == nullptr || glu->src[1] == nullptr ||
        !((glu->src[0] == gate && glu->src[1] == up) ||
          (glu->src[0] == up && glu->src[1] == gate))) {
        return false;
    }
    const auto * metadata = device->aot->find("flagos_ffn_swiglu_q4_k_f32_decode");
    if (metadata == nullptr || metadata->block_size <= 0) {
        return false;
    }

    flagos_quantized_matmul_signature gate_signature;
    flagos_quantized_matmul_signature up_signature;
    if (!flagos_describe_quantized_matmul(gate, &gate_signature) ||
        !flagos_describe_quantized_matmul(up, &up_signature) ||
        gate_signature.weight_kind != flagos_quantized_matmul_kind::q4_k ||
        up_signature.weight_kind != flagos_quantized_matmul_kind::q4_k) {
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

    if (glu->data != nullptr &&
        (glu->data == activation->data || glu->data == gate->data || glu->data == up->data ||
         glu->data == gate_weights->data || glu->data == up_weights->data)) {
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
        set_rows->data != rope->data && set_rows->data != rope->src[0]->data &&
        amd_supports_set_rows(device, set_rows);
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
        ggml_nelements(op) > 0 && ggml_nelements(op) <= INT32_MAX;
}

static bool amd_supports_flash_attn_decode(const amd_device_context * device, const ggml_tensor * op) {
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    if (!amd_fusion_enabled("flash_attn_decode") || device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_FLASH_ATTN_EXT || device->aot->find("flagos_flash_attn_decode_f32_f16") == nullptr) {
        return false;
    }
    const auto * op_params_bytes = reinterpret_cast<const uint8_t *>(op->op_params);
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
        q->nb[1] / sizeof(float) <= INT32_MAX && q->nb[2] / sizeof(float) <= INT32_MAX &&
        k->nb[1] / sizeof(ggml_fp16_t) <= INT32_MAX && k->nb[2] / sizeof(ggml_fp16_t) <= INT32_MAX &&
        v->nb[1] / sizeof(ggml_fp16_t) <= INT32_MAX && v->nb[2] / sizeof(ggml_fp16_t) <= INT32_MAX &&
        op->nb[1] / sizeof(float) <= INT32_MAX && op->nb[2] / sizeof(float) <= INT32_MAX &&
        ggml_is_contiguous(mask) && ggml_is_contiguous(op) && max_bias == 0.0f && logit_softcap == 0.0f;
}

static bool amd_supports_flash_attn_prefill(const amd_device_context * device, const ggml_tensor * op) {
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    if (!amd_fusion_enabled("flash_attn_prefill") || device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_FLASH_ATTN_EXT ||
        device->aot->find("flagos_flash_attn_prefill_f32_f16") == nullptr) {
        return false;
    }
    const auto * op_params_bytes = reinterpret_cast<const uint8_t *>(op->op_params);
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
        op->nb[1] / sizeof(float) <= INT32_MAX && op->nb[2] / sizeof(float) <= INT32_MAX &&
        ggml_is_contiguous(mask) && ggml_is_contiguous(op) &&
        max_bias == 0.0f && logit_softcap == 0.0f;
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
    return metadata != nullptr && metadata->block_size >= op->ne[0];
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
    const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    if (weight == nullptr || weight->type != GGML_TYPE_F32 || ggml_nelements(weight) != norm->ne[0] ||
        !ggml_is_contiguous(weight)) {
        return false;
    }
    if ((norm->data != nullptr && (norm->data == norm->src[0]->data || norm->data == weight->data)) ||
        (mul->data != nullptr && (mul->data == norm->src[0]->data || mul->data == weight->data))) {
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
        !ggml_are_same_shape(add, norm) || !ggml_is_contiguous(add) ||
        add->src[0]->type != GGML_TYPE_F32 || add->src[1]->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(add->src[0], norm) || !ggml_are_same_shape(add->src[1], norm) ||
        !ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous(add->src[1]) ||
        // The AMD add+RMS+MUL kernel is currently validated for decode rows.
        // Prefill residual layouts remain on the provider-neutral fallback
        // until their full stride/broadcast ABI is implemented.
        norm->ne[1] != 1 || norm->ne[2] != 1 || norm->ne[3] != 1) {
        return false;
    }
    const auto * metadata = device->aot->find("flagos_add_rms_norm_mul_f32");
    return metadata->block_size >= norm->ne[0];
}

static bool amd_supports_rms_norm_mul_inplace(const amd_device_context * device,
                                              const ggml_tensor * norm,
                                              const ggml_tensor * mul) {
    const ggml_tensor * weight = mul != nullptr && mul->src[0] == norm ? mul->src[1] :
        (mul != nullptr ? mul->src[0] : nullptr);
    return device != nullptr && device->aot != nullptr && norm != nullptr && mul != nullptr &&
        norm->data != nullptr && norm->data == mul->data &&
        device->aot->find("flagos_rms_norm_mul_inplace_f32") != nullptr &&
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

static flagos_lowering_choice amd_query_fusion(
        void * user_data, const ggml_cgraph * cgraph, const flagos_pattern_candidate & candidate) {
    auto * context = static_cast<amd_backend_context *>(user_data);
    flagos_lowering_choice choice;
    if (context == nullptr || cgraph == nullptr || candidate.node_indices.empty()) {
        return choice;
    }
    if (candidate.id == flagos_pattern_id::rms_norm_mul && candidate.node_indices.size() == 2) {
        const ggml_tensor * norm = cgraph->nodes[candidate.node_indices[0]];
        const ggml_tensor * mul = cgraph->nodes[candidate.node_indices[1]];
        if (amd_fusion_enabled("rms_norm_mul") &&
            ((amd_supports_rms_norm_mul(context->device, norm, mul) && norm->data != mul->data) ||
             amd_supports_rms_norm_mul_inplace(context->device, norm, mul))) {
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = 1;
        }
    } else if (candidate.id == flagos_pattern_id::add_rms_norm_mul && candidate.node_indices.size() == 3) {
        // Deliberately declined; see the model-graph audit note above.
    } else if (candidate.id == flagos_pattern_id::rope_kv_store && candidate.node_indices.size() == 3) {
        const ggml_tensor * rope = cgraph->nodes[candidate.node_indices[0]];
        const ggml_tensor * view = cgraph->nodes[candidate.node_indices[1]];
        const ggml_tensor * set_rows = cgraph->nodes[candidate.node_indices[2]];
        if (amd_fusion_enabled("rope_kv_store") && amd_supports_rope_kv_store(context->device, rope, view, set_rows)) {
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = 4;
        }
    } else if (candidate.id == flagos_pattern_id::flash_attn_decode && candidate.node_indices.size() == 1) {
        const ggml_tensor * attention = cgraph->nodes[candidate.node_indices[0]];
        if (amd_fusion_enabled("flash_attn_decode") && amd_supports_flash_attn_decode(context->device, attention)) {
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = 3;
        }
    } else if (candidate.id == flagos_pattern_id::flash_attn_prefill && candidate.node_indices.size() == 1) {
        const ggml_tensor * attention = cgraph->nodes[candidate.node_indices[0]];
        if (amd_fusion_enabled("flash_attn_prefill") && amd_supports_flash_attn_prefill(context->device, attention)) {
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = 5;
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
        if (amd_supports_ffn_swiglu_q4_decode(
                context->device, first_projection, second_projection, glu)) {
            choice.supported = true;
            choice.capture_safe = true;
            choice.implementation_id = 7;
        } else if (amd_supports_ffn_swiglu_f16(context->device, first_projection, second_projection, glu) &&
            described && amd_ffn_dequant_cache_can_fit(
                context, first_projection, first_signature,
                second_projection, second_signature)) {
            choice.supported = true;
            // The first use may allocate and populate the persistent F16
            // dequant caches; keep this fusion out of HIP Graph capture until
            // cache warm-up is made an explicit scheduler phase.
            choice.capture_safe = false;
            choice.implementation_id = 6;
        }
    }
    return choice;
}

static bool amd_execute_fusion(void * user_data, ggml_cgraph * cgraph, const flagos_plan_step & step) {
    auto * context = static_cast<amd_backend_context *>(user_data);
    if (context == nullptr || cgraph == nullptr) {
        return false;
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
            const auto * metadata = context->device->aot->find(
                "flagos_ffn_swiglu_q4_k_f32_decode");
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
                "flagos_ffn_swiglu_q4_k_f32_decode", context->stream,
                (static_cast<unsigned int>(rows) + row_tile - 1) / row_tile,
                1, 1, arguments);
            if (launched) {
                context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
                context->stats.fusion_steps.fetch_add(1, std::memory_order_relaxed);
                context->stats.fusion_ffn_swiglu.fetch_add(1, std::memory_order_relaxed);
                amd_trace_op(context, glu, "ffn_swiglu_q4_decode");
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
        if (!amd_supports_flash_attn_decode(context->device, node)) return false;
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
        if (!amd_supports_flash_attn_prefill(context->device, node)) return false;
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
    if (step.candidate.id == flagos_pattern_id::rms_norm_mul && step.candidate.node_indices.size() == 2) {
        norm = cgraph->nodes[step.candidate.node_indices[0]];
        mul = cgraph->nodes[step.candidate.node_indices[1]];
        const bool inplace = amd_supports_rms_norm_mul_inplace(context->device, norm, mul);
        if (!inplace && (norm->data == mul->data || !amd_supports_rms_norm_mul(context->device, norm, mul))) {
            return false;
        }
        kernel_name = inplace ? "flagos_rms_norm_mul_inplace_f32" : "flagos_rms_norm_mul_f32";
    } else if (step.candidate.id == flagos_pattern_id::add_rms_norm_mul && step.candidate.node_indices.size() == 3) {
        add = cgraph->nodes[step.candidate.node_indices[0]];
        norm = cgraph->nodes[step.candidate.node_indices[1]];
        mul = cgraph->nodes[step.candidate.node_indices[2]];
        const bool inplace = amd_supports_add_rms_norm_mul_inplace(context->device, add, norm, mul);
        if (!inplace && (norm->data == mul->data || !amd_supports_add_rms_norm_mul(context->device, add, norm, mul))) {
            return false;
        }
        kernel_name = inplace ? "flagos_add_rms_norm_mul_inplace_f32" : "flagos_add_rms_norm_mul_f32";
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
    const bool inplace = std::strcmp(kernel_name, "flagos_rms_norm_mul_inplace_f32") == 0 ||
        std::strcmp(kernel_name, "flagos_add_rms_norm_mul_inplace_f32") == 0;
    if (inplace) {
        flagos_amd::kernel_arguments inplace_arguments;
        if (add == nullptr) {
            inplace_arguments = { &mul_data, &src_data, &weight_data, &n_cols, &eps };
        } else {
            inplace_arguments = { &mul_data, &src_data, &bias_data, &weight_data, &n_cols, &eps };
        }
        const unsigned int inplace_rows = rows;
        const bool launched = context->device->aot->launch(kernel_name, context->stream,
            inplace_rows, 1, 1, inplace_arguments);
        if (launched) {
            context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            context->stats.fusion_steps.fetch_add(1, std::memory_order_relaxed);
            if (step.candidate.id == flagos_pattern_id::rms_norm_mul) {
                context->stats.fusion_rms_norm_mul.fetch_add(1, std::memory_order_relaxed);
            } else {
                context->stats.fusion_add_rms_norm_mul.fetch_add(1, std::memory_order_relaxed);
            }
            amd_trace_op(context, mul, step.candidate.id == flagos_pattern_id::rms_norm_mul
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
    return device != nullptr && device->aot != nullptr &&
        device->aot->find("flagos_add_f32") != nullptr && op != nullptr &&
        op->src[0] != nullptr && op->src[1] != nullptr &&
        amd_tensor_is_contiguous_f32(op) &&
        amd_tensor_is_contiguous_f32(op->src[0]) &&
        amd_tensor_is_contiguous_f32(op->src[1]) &&
        ggml_are_same_shape(op, op->src[0]) &&
        ggml_are_same_shape(op, op->src[1]) &&
        ggml_nelements(op) > 0 && ggml_nelements(op) <= INT32_MAX;
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
        return true;
    }
    return op->src[1]->ne[0] == op->src[0]->ne[0] &&
        op->src[1]->ne[1] == 1 && op->src[1]->ne[2] == 1 && op->src[1]->ne[3] == 1;
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

    // The Triton kernel uses one program per row and a compile-time BLOCK.
    // A package may choose a wider block than the model's hidden size, but it
    // must never be narrower or the masked load/store would be incomplete.
    const auto * metadata = device->aot->find("flagos_rms_norm_f32");
    return metadata->block_size >= op->ne[0];
}

static enum ggml_status amd_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * backend_context = static_cast<amd_backend_context *>(backend->context);
    if (backend_context == nullptr || backend_context->device == nullptr) {
        return GGML_STATUS_FAILED;
    }
    const flagos_fusion_interface fusion_interface {
        /* .query_lowering = */ amd_query_fusion,
        /* .execute_fusion = */ amd_execute_fusion,
        /* .user_data      = */ backend_context,
    };
    const uint64_t config_key = amd_fusion_config_key();
    if (backend_context->graph_plan_config_key != 0 &&
        backend_context->graph_plan_config_key != config_key) {
        // Configuration changes are unusual in production, but are common in
        // bring-up tests.  Drop both plans and executable graphs together so
        // no old fusion choice can survive a selector change.
        amd_hip_check(hipStreamSynchronize(backend_context->stream),
            "hipStreamSynchronize before plan invalidation");
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

    // The scheduler may split a transformer layer into several backend
    // subgraphs at CPU-fallback boundaries; on Qwen3 the largest stable
    // subgraph is currently eight nodes.  Requiring a 16-node graph would
    // therefore make capture permanently unreachable.  Keep a small lower
    // bound to amortize HIP graph launch/instantiate overhead while allowing
    // the planner's actual subgraph granularity to participate.
    // The opt-in prefill path may lazily allocate/dequantize a weight cache on
    // its first use. HIP forbids those allocations inside stream capture; once
    // every production-scale projection in this graph has a resident cache,
    // capture is safe again and can amortize the many small AOT launches.
    const bool capture_eligible = amd_graph_capture_enabled() &&
        amd_f16_graph_cache_ready(backend_context, cgraph) && plan.capture_safe() &&
        cgraph->n_nodes >= 4;
    const std::vector<uintptr_t> capture_pointers = capture_eligible
        ? amd_graph_capture_pointers(cgraph) : std::vector<uintptr_t>();
    bool capture_ready = false;
    if (capture_eligible) {
        if (amd_graph_capture_entry * entry = amd_find_graph_capture(
                backend_context, plan.structural_fingerprint, capture_pointers)) {
            if (amd_hip_check(hipGraphLaunch(entry->executable, backend_context->stream), "hipGraphLaunch")) {
                backend_context->stats.graph_replays.fetch_add(1, std::memory_order_relaxed);
                return GGML_STATUS_SUCCESS;
            }
            backend_context->stats.graph_capture_failures.fetch_add(1, std::memory_order_relaxed);
            // A failed replay may leave the executable or stream in an
            // implementation-specific error state.  Fall back to direct
            // launches for this invocation and let the next invocation retry
            // through the normal warm-up path.
        } else if (backend_context->graph_capture_candidate_valid &&
                   backend_context->graph_capture_candidate_fingerprint == plan.structural_fingerprint &&
                   backend_context->graph_capture_candidate_pointers == capture_pointers) {
            capture_ready = true;
            backend_context->graph_capture_candidate_valid = false;
            backend_context->graph_capture_candidate_pointers.clear();
        } else {
            backend_context->graph_capture_candidate_fingerprint = plan.structural_fingerprint;
            backend_context->graph_capture_candidate_pointers = capture_pointers;
            backend_context->graph_capture_candidate_valid = true;
        }
    }

    amd_graph_capture_scope capture_scope(backend_context->stream);
    bool capturing = false;
    if (capture_ready) {
        // Capture only after all work queued before this graph has completed.
        // This avoids accidentally importing scheduler copies or an earlier
        // graph into the captured dependency graph.
        amd_hip_check(hipStreamSynchronize(backend_context->stream), "hipStreamSynchronize before capture");
        capturing = capture_scope.begin();
        if (!capturing) {
            backend_context->stats.graph_capture_failures.fetch_add(1, std::memory_order_relaxed);
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
        if (node->op == GGML_OP_GET_ROWS && amd_supports_get_rows(backend_context->device, node)) {
            const char * kernel_name = node->src[0]->type == GGML_TYPE_Q4_K
                ? "flagos_get_rows_q4_k_f32" : "flagos_get_rows_q6_k_f32";
            int n_cols = static_cast<int>(node->ne[0]);
            int n_tokens = static_cast<int>(ggml_nelements(node->src[1]));
            int blocks_per_row = n_cols / 256;
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
            if (node->src[0]->type == GGML_TYPE_Q4_K) {
                backend_context->stats.q4_get_rows.fetch_add(1, std::memory_order_relaxed);
            } else {
                backend_context->stats.q6_get_rows.fetch_add(1, std::memory_order_relaxed);
            }
            amd_trace_op(backend_context, node, "get_rows");
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT && amd_supports_quantized_mul_mat(backend_context->device, node)) {
            const bool q4 = node->src[0]->type == GGML_TYPE_Q4_K;
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
            if (columns > 1 && rows >= 64 && k >= 1024 && amd_prefill_f16_gemm_enabled()) {
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
                const unsigned int requested_narrow_tile = amd_q4_gemv_narrow_row_tile();
                const bool narrow_q4 = q4 && amd_q4_gemv_narrow_enabled();
                const char * narrow_kernel = nullptr;
                unsigned int row_tile = 1U;
                if (narrow_q4 && requested_narrow_tile == 8U && rows % 8 == 0 &&
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
                    : q4 ? "flagos_mul_mat_q4_k_f32" : "flagos_mul_mat_q6_k_f32";
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
                (q4 ? backend_context->stats.q4_matmul : backend_context->stats.q6_matmul)
                    .fetch_add(1, std::memory_order_relaxed);
                amd_trace_op(backend_context, node,
                    narrow_kernel != nullptr ? (row_tile == 8U ? "q4_matmul_narrow8" : "q4_matmul_narrow4")
                    : q4 ? "q4_matmul" : "q6_matmul");
            } else {
                const char * kernel_name = amd_quantized_batched_kernel(
                    backend_context->device,
                    q4 ? flagos_quantized_matmul_kind::q4_k : flagos_quantized_matmul_kind::q6_k);
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
                (q4 ? backend_context->stats.q4_matmul_batched : backend_context->stats.q6_matmul_batched)
                    .fetch_add(1, std::memory_order_relaxed);
                if (tiled) {
                    (q4 ? backend_context->stats.q4_matmul_tiled : backend_context->stats.q6_matmul_tiled)
                        .fetch_add(1, std::memory_order_relaxed);
                }
                amd_trace_op(backend_context, node, q4 ? "q4_matmul_batched" : "q6_matmul_batched");
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
            const auto * metadata = backend_context->device->aot->find("flagos_add_f32");
            const unsigned int grid_x = static_cast<unsigned int>(
                (ggml_nelements(node) + metadata->block_size - 1) / metadata->block_size);
            int n = static_cast<int>(ggml_nelements(node));
            flagos_amd::kernel_arguments arguments = {
                &node->src[0]->data,
                &node->src[1]->data,
                &node->data,
                &n,
            };
            if (!backend_context->device->aot->launch("flagos_add_f32", backend_context->stream,
                    grid_x, 1, 1, arguments)) {
                return GGML_STATUS_FAILED;
            }
            backend_context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
            backend_context->stats.direct_ops.fetch_add(1, std::memory_order_relaxed);
            amd_trace_op(backend_context, node, "add");
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
        if (capture_scope.finish(&executable)) {
            amd_store_graph_capture(backend_context, plan.structural_fingerprint,
                capture_pointers, executable);
            backend_context->stats.graph_captures.fetch_add(1, std::memory_order_relaxed);
            GGML_LOG_DEBUG("FlagOS AMD: captured HIP graph %016llx with %d nodes\n",
                static_cast<unsigned long long>(plan.structural_fingerprint), cgraph->n_nodes);
        } else {
            backend_context->stats.graph_capture_failures.fetch_add(1, std::memory_order_relaxed);
        }
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
    return new ggml_backend {
        /* .guid    = */ []() -> ggml_guid_t {
            static ggml_guid guid = { 0x46, 0x6c, 0x61, 0x67, 0x4f, 0x53, 0x2d, 0x41, 0x4d, 0x44, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01 };
            return &guid;
        }(),
        /* .iface   = */ g_backend_iface,
        /* .device  = */ dev,
        /* .context = */ context,
    };
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
        ((op->op == GGML_OP_GET_ROWS && amd_supports_get_rows(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_MUL_MAT && amd_supports_quantized_mul_mat(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_ADD && amd_supports_add(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_MUL && amd_supports_mul(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_SET_ROWS && amd_supports_set_rows(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_RMS_NORM && amd_supports_rms_norm(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_ROPE && amd_supports_rope_neox(amd_device_from_dev(dev), op)) ||
         (op->op == GGML_OP_UNARY && amd_supports_silu(amd_device_from_dev(dev), op)) ||
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
    auto * context = static_cast<amd_event_context *>(event->context);
    amd_hip_check(hipSetDevice(context->device), "hipSetDevice");
    GGML_ASSERT(amd_hip_check(hipEventSynchronize(context->event), "hipEventSynchronize"));
}

static void amd_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * backend_context = static_cast<amd_backend_context *>(backend->context);
    auto * event_context = static_cast<amd_event_context *>(event->context);
    amd_hip_check(hipSetDevice(backend_context->device->ordinal), "hipSetDevice");
    GGML_ASSERT(amd_hip_check(hipEventRecord(event_context->event, backend_context->stream), "hipEventRecord"));
}

static void amd_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * backend_context = static_cast<amd_backend_context *>(backend->context);
    auto * event_context = static_cast<amd_event_context *>(event->context);
    amd_hip_check(hipSetDevice(backend_context->device->ordinal), "hipSetDevice");
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

static void amd_try_load_aot(amd_device_context & device) {
    if (device.aot_attempted) {
        return;
    }
    device.aot_attempted = true;
    const char * kernel_dir = std::getenv("FLAGOS_AMD_KERNEL_DIR");
    if (kernel_dir == nullptr || kernel_dir[0] == '\0') {
        return;
    }
    auto registry = std::make_unique<flagos_amd::kernel_registry>();
    if (registry->initialize(std::filesystem::path(kernel_dir), device.ordinal,
            device.props.gcnArchName)) {
        device.aot = std::move(registry);
        GGML_LOG_INFO("FlagOS AMD: loaded %zu AOT kernels for %s\n",
            device.aot->size(), device.props.gcnArchName);
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
        uuid_lo = static_cast<uint64_t>(index + 1);
    }
    *identity = {
        /* .provider_id      = */ 0x414D440000000001ULL,
        /* .uuid_hi          = */ uuid_hi,
        /* .uuid_lo          = */ uuid_lo,
        /* .memory_domain_id = */ 0x414D440000000001ULL,
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
