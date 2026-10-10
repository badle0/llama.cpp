// FlagOS provider for the SpacemiT K3 AI cores (A100, CPUs 8-15).
// One ACCEL device with a buffer type and a backend; ops run on the AI cores through spine-runtime
// (M2a: ADD; M2b, M2c: Q4_0 and Q4_1 MUL_MAT on the IME, weights repacked in the buffer; M2d: the other ops of a
// layer except attention).

#include "flagos-spacemit-api.h"
#include "flagos-spacemit-exec.h"
#include "flagos-spacemit-ops.h"
#include "flagos-spacemit-weights.h"

#include "../../../ggml-backend-impl.h"
#include "../../../ggml-impl.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(__linux__)
#    include <unistd.h>
#endif
#if defined(GGML_FLAGOS_SPACEMIT_HOST_TEST_DEVICE) && defined(__APPLE__)
#    include <sys/sysctl.h>
#endif

namespace {

constexpr uint64_t SPACEMIT_PROVIDER_ID       = 0x53504D5400000001ULL;  // "SPMT", 1
constexpr uint64_t SPACEMIT_DEVICE_UUID_HI    = 0x53504D544B330000ULL;  // "SPMTK3"
constexpr uint64_t SPACEMIT_DEVICE_UUID_LO    = 0x0000000041000002ULL;
constexpr uint64_t SPACEMIT_MEMORY_DOMAIN_ID  = 0x53504D544B330001ULL;  // K3 DRAM, shared with the X100 cores
constexpr size_t   SPACEMIT_BUFFER_ALIGNMENT  = 64;
constexpr uint32_t SPACEMIT_A100_VECTOR_BITS  = 1024;

struct spacemit_device_context {
    std::string                name;
    std::string                description;
    uint32_t                   n_ai_cores = 0;
    ggml_backend_device        device     = {};
    ggml_backend_buffer_type   buffer_type = {};
    std::mutex                 run_mutex;  // backends of this device share the same AI cores
};

struct spacemit_buffer_context {
    void * data = nullptr;
    size_t size = 0;
};

struct spacemit_backend_context {
    std::unique_ptr<spacemit_executor> executor;
    std::vector<spacemit_step>         steps;
    void *                             workspace      = nullptr;  // shared by the tiles; grows to the largest split
    size_t                             workspace_size = 0;

    ~spacemit_backend_context() {
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
// buffer: 64-byte aligned host memory; Q4_0 and Q4_1 matmul weights are stored in IME layouts (flagos-spacemit-weights.h)
//

void spacemit_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    ggml_aligned_free(ctx->data, ctx->size);
    delete ctx;
}

void * spacemit_buffer_base(ggml_backend_buffer_t buffer) {
    return static_cast<spacemit_buffer_context *>(buffer->context)->data;
}

void spacemit_buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    spacemit_tensor_fill(tensor, value, offset, size);
}

// repacks; reads undo the repack (Q4_1: the converted weights), so the scheduler can still copy a weight out if the
// provider refuses an op on it
void spacemit_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    spacemit_tensor_write(tensor, data, offset, size);
}

void spacemit_buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    spacemit_tensor_read(tensor, data, offset, size);
}

// a raw copy would skip the repack; returning false makes ggml use set_tensor instead
bool spacemit_buffer_cpy_tensor(ggml_backend_buffer_t, const ggml_tensor * src, ggml_tensor * dst) {
    if (src->buffer == nullptr || !ggml_backend_buffer_is_host(src->buffer) || spacemit_tensor_is_repacked(dst)) {
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

// ggml_backend_buft_alloc_buffer() handles size 0 itself and never calls this with it
ggml_backend_buffer_t spacemit_buffer_type_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);
    if (data == nullptr) {
        GGML_LOG_ERROR("FlagOS SpacemiT: failed to allocate a %zu-byte buffer\n", size);
        return nullptr;
    }
    auto * ctx = new spacemit_buffer_context{ data, size };
    return ggml_backend_buffer_init(buft, g_buffer_iface, ctx, size);
}

size_t spacemit_buffer_type_alignment(ggml_backend_buffer_type_t) {
    return SPACEMIT_BUFFER_ALIGNMENT;
}

const ggml_backend_buffer_type_i g_buffer_type_iface = {
    /* .get_name       = */ spacemit_buffer_type_name,
    /* .alloc_buffer   = */ spacemit_buffer_type_alloc,
    /* .get_alignment  = */ spacemit_buffer_type_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ nullptr,
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

// every node and the workspace are checked before the launch, so a graph either runs completely or fails with
// nothing written
ggml_status spacemit_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = static_cast<spacemit_backend_context *>(backend->context);
    auto * dev = static_cast<spacemit_device_context *>(backend->device->context);

    ctx->steps.clear();
    size_t workspace_size = 0;
    for (int i = 0; i < ggml_graph_n_nodes(cgraph); i++) {
        ggml_tensor * node = ggml_graph_node(cgraph, i);
        // as ggml-cpu: nothing to do for empty tensors, and nodes without the compute flag are branches the graph
        // left unselected (ggml_build_forward_select), whose inputs may never have been set
        if (ggml_op_is_empty(node->op) || ggml_is_empty(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        const spacemit_op * op = spacemit_find_op(node, &dev->buffer_type);
        if (op == nullptr) {
            GGML_LOG_ERROR("FlagOS SpacemiT: no kernel for %s (%s)\n", ggml_op_desc(node), node->name);
            return GGML_STATUS_FAILED;
        }
        if (!spacemit_data_ready(node)) {
            GGML_LOG_ERROR("FlagOS SpacemiT: %s (%s) has an unallocated operand\n", ggml_op_desc(node), node->name);
            return GGML_STATUS_FAILED;
        }
        for (uint32_t s = 0; s < op->n_steps; s++) {
            ctx->steps.push_back({ op->steps[s], node });
        }
        if (op->workspace_size != nullptr) {
            workspace_size = std::max(workspace_size, op->workspace_size(node));
        }
    }
    if (ctx->steps.empty()) {
        return GGML_STATUS_SUCCESS;
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
    return ctx->executor->run(ctx->steps, ctx->workspace, ctx->workspace_size) ? GGML_STATUS_SUCCESS
                                                                                  : GGML_STATUS_FAILED;
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

// the AI cores use the same DRAM as the CPU cores
void spacemit_device_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;
#if defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string   line;
    while (std::getline(meminfo, line)) {
        const size_t value_kib = std::strtoull(line.c_str() + line.find(':') + 1, nullptr, 10);
        if (line.rfind("MemTotal:", 0) == 0) {
            *total = value_kib * 1024;
        } else if (line.rfind("MemAvailable:", 0) == 0) {
            *free = value_kib * 1024;
        }
    }
#elif defined(GGML_FLAGOS_SPACEMIT_HOST_TEST_DEVICE) && defined(__APPLE__)
    // simulated device on a Mac: the host's memory size; free memory is not tracked
    uint64_t bytes = 0;
    size_t   len   = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &len, nullptr, 0) == 0) {
        *free  = bytes;
        *total = bytes;
    }
#endif
}

enum ggml_backend_dev_type spacemit_device_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
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

// experiment switch: FLAGOS_SPACEMIT_STREAM=per-call releases the AI cores after every graph (M2a measures both)
spacemit_stream_policy spacemit_stream_policy_from_env() {
    const char * value = std::getenv("FLAGOS_SPACEMIT_STREAM");
    return value != nullptr && std::strcmp(value, "per-call") == 0 ? spacemit_stream_policy::per_call
                                                                    : spacemit_stream_policy::persistent;
}

ggml_backend_t spacemit_device_init(ggml_backend_dev_t dev, const char *) {
    auto * ctx     = new spacemit_backend_context;
    ctx->executor  = spacemit_executor_create(spacemit_stream_policy_from_env(), spacemit_device_from_dev(dev)->n_ai_cores);
    GGML_LOG_DEBUG("FlagOS SpacemiT: backend uses %s\n", ctx->executor->name());
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
    return ggml_op_is_empty(op->op) || spacemit_find_op(op, &spacemit_device_from_dev(dev)->buffer_type) != nullptr;
}

// operands may also live in CPU buffers: under ACCEL the KV cache and activations stay there
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
// probe: a K3 is detected by its A100 cores in /proc/cpuinfo and the AI-thread gate
//

uint32_t spacemit_count_ai_cores() {
#if defined(GGML_FLAGOS_SPACEMIT_HOST_TEST_DEVICE)
    return 8;  // simulated device (tests only, non-riscv64 builds): the K3's 8 AI cores
#elif defined(__linux__) && defined(__riscv)
    constexpr uint64_t a100_marchid = 0x8000000041000002ULL;
    std::ifstream      cpuinfo("/proc/cpuinfo");
    std::string   line;
    uint32_t      count = 0;
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("marchid", 0) != 0) {
            continue;
        }
        const size_t colon = line.find(':');
        if (colon != std::string::npos && std::strtoull(line.c_str() + colon + 1, nullptr, 16) == a100_marchid) {
            count++;
        }
    }
    return count;
#else
    return 0;
#endif
}

bool spacemit_ai_gate_available() {
#if defined(GGML_FLAGOS_SPACEMIT_HOST_TEST_DEVICE)
    return true;
#elif defined(__linux__)
    return access("/proc/set_ai_thread", W_OK) == 0;
#else
    return false;
#endif
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

    const uint32_t n_ai_cores = spacemit_count_ai_cores();
    if (n_ai_cores == 0) {
        return false;
    }
    if (!spacemit_ai_gate_available()) {
        GGML_LOG_WARN("FlagOS SpacemiT: %u A100 cores found but /proc/set_ai_thread is not writable\n", n_ai_cores);
        return false;
    }

    g_device.name                 = "FlagOS:SpacemiT:0";
#if defined(GGML_FLAGOS_SPACEMIT_HOST_TEST_DEVICE)
    g_device.description          = "simulated SpacemiT K3 device for tests (" + std::to_string(n_ai_cores) + " tiles, serial)";
#else
    g_device.description          = "SpacemiT K3 A100 AI cores (" + std::to_string(n_ai_cores) + ")";
#endif
    g_device.n_ai_cores           = n_ai_cores;
    g_device.device.iface         = g_device_iface;
    g_device.device.reg           = owner_reg;
    g_device.device.context       = &g_device;
    g_device.buffer_type.iface    = g_buffer_type_iface;
    g_device.buffer_type.device   = &g_device.device;
    g_device.buffer_type.context  = &g_device;
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
#if defined(GGML_FLAGOS_SPACEMIT_SPERT)
    profile->runtime           = "spine-runtime";
#else
    profile->runtime           = "none";
#endif
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
