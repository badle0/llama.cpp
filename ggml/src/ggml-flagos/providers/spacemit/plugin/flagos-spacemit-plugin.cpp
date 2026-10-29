// The SpacemiT K3 plugin's table (plan.md "S1 step 1 spec" §3): the device, the buffers on huge pages with the repack
// inside set and get, the stream, the capture replay and the kernels, each member a thin wrapper over the code that
// was in flagos-spacemit.cpp or is in the files beside this one. In this step the plugin is still compiled into
// libggml-flagos and the core loads that library through its entry symbol (S2 builds it as
// libflagos-provider-spacemit.so).

#include "../core/flagos-spacemit-plugin.h"

#include "../flagos-spacemit-attn.h"
#include "../flagos-spacemit-exec.h"
#include "../flagos-spacemit-ime.h"
#include "../flagos-spacemit-kernels.h"
#include "../flagos-spacemit-ops.h"
#include "../flagos-spacemit-weights.h"

#include "../../../../ggml-impl.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

#if defined(__linux__)
#    include <unistd.h>
#endif
#if !defined(_WIN32)
#    include <sys/mman.h>
#endif
#if defined(GGML_FLAGOS_SPACEMIT_HOST_TEST_DEVICE) && defined(__APPLE__)
#    include <sys/sysctl.h>
#endif

// the handles the core holds opaquely
struct spacemit_plugin_device {
    uint32_t    n_tiles = 0;
    std::string description;
};

struct spacemit_plugin_stream {
    std::unique_ptr<spacemit_executor> executor;
};

struct spacemit_plugin_buffer {
    void * data   = nullptr;
    size_t size   = 0;
    size_t mapped = 0;  // > 0: data is a mapping of this many bytes (spacemit_map_huge), else from ggml_aligned_malloc
};

namespace {

constexpr uint32_t SPACEMIT_A100_VECTOR_BITS = 1024;

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

//
// memory: host memory on huge pages where the system has them, else 64-byte aligned; Q4_0, Q4_1, Q8_0 and Q6_K matmul
// weights are stored in IME layouts (flagos-spacemit-weights.h), a Q6_K weight in more than ggml_nbytes
// (tensor_alloc_size). set, get and fill work in ggml's layout within ggml_nbytes; only the repack and its inverse
// touch the stored bytes beyond it.
//

#if defined(MADV_HUGEPAGE)
constexpr size_t SPACEMIT_HUGE_PAGE = size_t(2) << 20;

// memory on 2 MiB transparent huge pages, as the in-tree path keeps its weights (ggml-cpu/spacemit/spine_mem_pool.cpp,
// HPAGE): the K3 gives them only on request (madvise). Measured on the K3 (plan.md, "1-row matmul bandwidth"): with the
// weights on huge pages, Qwen3-4B generates about 1% faster (the in-tree path: 2%). Whole huge pages from a huge page
// boundary; nullptr if the mapping fails. If madvise is refused, the memory keeps the default pages.
void * spacemit_map_huge(size_t size, size_t & mapped) {
    constexpr size_t huge = SPACEMIT_HUGE_PAGE;
    const size_t     len  = GGML_PAD(size, huge);
    void *           raw  = mmap(nullptr, len + huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
        return nullptr;
    }
    // one huge page more than needed, trimmed at both ends to start on a huge page boundary
    uint8_t * base = static_cast<uint8_t *>(raw);
    uint8_t * data = reinterpret_cast<uint8_t *>(GGML_PAD(reinterpret_cast<uintptr_t>(base), huge));
    if (data > base) {
        munmap(base, (size_t) (data - base));
    }
    munmap(data + len, huge - (size_t) (data - base));
    if (madvise(data, len, MADV_HUGEPAGE) != 0) {
        GGML_LOG_DEBUG("FlagOS SpacemiT: madvise(MADV_HUGEPAGE) failed (errno %d); default pages\n", errno);
    }
    mapped = len;
    return data;
}
#endif

//
// the table's members
//

uint32_t plugin_device_count(void) {
    const uint32_t n_ai_cores = spacemit_count_ai_cores();
    if (n_ai_cores == 0) {
        return 0;
    }
    if (!spacemit_ai_gate_available()) {
        GGML_LOG_WARN("FlagOS SpacemiT: %u A100 cores found but /proc/set_ai_thread is not writable\n", n_ai_cores);
        return 0;
    }
    return 1;
}

enum spacemit_status plugin_device_open(uint32_t index, spacemit_plugin_device ** dev) {
    if (index != 0 || dev == nullptr) {
        return SPACEMIT_STATUS_INVALID_ARGUMENT;
    }
    const uint32_t n_ai_cores = spacemit_count_ai_cores();
    auto *         d          = new spacemit_plugin_device;
    d->n_tiles                = n_ai_cores;
#if defined(GGML_FLAGOS_SPACEMIT_HOST_TEST_DEVICE)
    d->description = "simulated SpacemiT K3 device for tests (" + std::to_string(n_ai_cores) + " tiles, serial)";
#else
    d->description = "SpacemiT K3 A100 AI cores (" + std::to_string(n_ai_cores) + ")";
#endif
    *dev = d;
    return SPACEMIT_STATUS_SUCCESS;
}

void plugin_device_close(spacemit_plugin_device * dev) {
    delete dev;
}

// the TCM one tile gets: the K3's block per core (k3-hardware.md §2), or the host executor's stand-in
// (FLAGOS_SPACEMIT_TEST_TCM). Informational in S1: the tiles still agree their TCM size at a launch's first barrier
size_t plugin_tcm_bytes_per_tile() {
#if defined(GGML_FLAGOS_SPACEMIT_SPERT)
    return 384 * 1024;
#else
    const char * value = std::getenv("FLAGOS_SPACEMIT_TEST_TCM");
    return value != nullptr ? (size_t) std::max(0LL, std::atoll(value)) : 0;
#endif
}

enum spacemit_status plugin_device_props(spacemit_plugin_device * dev, spacemit_plugin_device_props * out) {
    if (dev == nullptr || out == nullptr || out->struct_size < sizeof(spacemit_plugin_device_props)) {
        return SPACEMIT_STATUS_INVALID_ARGUMENT;
    }
    out->n_tiles            = dev->n_tiles;
    out->vector_bits        = SPACEMIT_A100_VECTOR_BITS;
    out->tcm_bytes_per_tile = plugin_tcm_bytes_per_tile();
    out->target             = "riscv64-a100-ime2";
#if defined(GGML_FLAGOS_SPACEMIT_SPERT)
    out->runtime = "spine-runtime";
#else
    out->runtime = "none";
#endif
    out->description = dev->description.c_str();
    return SPACEMIT_STATUS_SUCCESS;
}

// the AI cores use the same DRAM as the CPU cores
enum spacemit_status plugin_device_memory(spacemit_plugin_device * dev, size_t * free, size_t * total) {
    if (dev == nullptr || free == nullptr || total == nullptr) {
        return SPACEMIT_STATUS_INVALID_ARGUMENT;
    }
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
    return SPACEMIT_STATUS_SUCCESS;
}

// Buffers of at least one huge page go on huge pages; a smaller one would gain nothing and still take a whole one
// (about 0.75 ms to fault in on the K3). test-only: FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1 keeps the default pages, for
// comparison
enum spacemit_status plugin_buffer_alloc(spacemit_plugin_device * dev, size_t size, spacemit_plugin_buffer ** buf) {
    if (dev == nullptr || buf == nullptr) {
        return SPACEMIT_STATUS_INVALID_ARGUMENT;
    }
    void * data   = nullptr;
    size_t mapped = 0;
    if (size > 0) {
#if defined(MADV_HUGEPAGE)
        const char * no_huge = std::getenv("FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES");
        if (size >= SPACEMIT_HUGE_PAGE && (no_huge == nullptr || std::strcmp(no_huge, "0") == 0)) {
            data = spacemit_map_huge(size, mapped);
        }
#endif
        if (data == nullptr) {
            data = ggml_aligned_malloc(size);
        }
        if (data == nullptr) {
            GGML_LOG_ERROR("FlagOS SpacemiT: failed to allocate a %zu-byte buffer\n", size);
            return SPACEMIT_STATUS_ALLOC_FAILED;
        }
    }
    auto * b  = new spacemit_plugin_buffer;
    b->data   = data;
    b->size   = size;
    b->mapped = mapped;
    *buf      = b;
    return SPACEMIT_STATUS_SUCCESS;
}

void plugin_buffer_free(spacemit_plugin_buffer * buf) {
    if (buf == nullptr) {
        return;
    }
    if (buf->mapped > 0) {
#if defined(MADV_HUGEPAGE)
        munmap(buf->data, buf->mapped);
#endif
    } else if (buf->data != nullptr) {
        ggml_aligned_free(buf->data, buf->size);
    }
    delete buf;
}

void * plugin_buffer_base(spacemit_plugin_buffer * buf) {
    return buf != nullptr ? buf->data : nullptr;
}

uint32_t plugin_tensor_layout(const struct ggml_tensor * t) {
    return static_cast<uint32_t>(spacemit_storage_layout(t));  // a view answers for its source
}

size_t plugin_tensor_alloc_size(const struct ggml_tensor * t) {
    return spacemit_weight_alloc_size(t);
}

enum spacemit_status plugin_tensor_set(struct ggml_tensor * t, const void * data, size_t offset, size_t size) {
    spacemit_tensor_write(t, data, offset, size);
    return SPACEMIT_STATUS_SUCCESS;
}

enum spacemit_status plugin_tensor_get(const struct ggml_tensor * t, void * data, size_t offset, size_t size) {
    spacemit_tensor_read(t, data, offset, size);
    return SPACEMIT_STATUS_SUCCESS;
}

enum spacemit_status plugin_tensor_fill(struct ggml_tensor * t, uint8_t value, size_t offset, size_t size) {
    spacemit_tensor_fill(t, value, offset, size);
    return SPACEMIT_STATUS_SUCCESS;
}

// experiment switch: FLAGOS_SPACEMIT_STREAM=per-call releases the AI cores after every graph (M2a measures both)
spacemit_stream_policy spacemit_stream_policy_from_env() {
    const char * value = std::getenv("FLAGOS_SPACEMIT_STREAM");
    return value != nullptr && std::strcmp(value, "per-call") == 0 ? spacemit_stream_policy::per_call
                                                                    : spacemit_stream_policy::persistent;
}

enum spacemit_status plugin_stream_create(spacemit_plugin_device * dev, uint32_t n_tiles,
                                          spacemit_plugin_stream ** stream) {
    if (dev == nullptr || stream == nullptr) {
        return SPACEMIT_STATUS_INVALID_ARGUMENT;
    }
    auto executor = spacemit_executor_create(spacemit_stream_policy_from_env(), n_tiles);
    if (executor == nullptr) {
        return SPACEMIT_STATUS_ALLOC_FAILED;
    }
    GGML_LOG_DEBUG("FlagOS SpacemiT: backend uses %s\n", executor->name());
    auto * s    = new spacemit_plugin_stream;
    s->executor = std::move(executor);
    *stream     = s;
    return SPACEMIT_STATUS_SUCCESS;
}

void plugin_stream_destroy(spacemit_plugin_stream * stream) {
    delete stream;
}

// the replay waits for its launches, so there is nothing left to synchronize
enum spacemit_status plugin_stream_synchronize(spacemit_plugin_stream * stream) {
    return stream != nullptr ? SPACEMIT_STATUS_SUCCESS : SPACEMIT_STATUS_INVALID_ARGUMENT;
}

const spacemit_plugin_kernel * plugin_kernels(uint32_t * count) {
    static const spacemit_plugin_kernel kernels[] = {
        { "add.f32",          spacemit_kernel_add_f32,       nullptr,                     nullptr                              },
        { "mul.f32",          spacemit_kernel_mul_f32,       nullptr,                     nullptr                              },
        { "rms_norm.f32",     spacemit_kernel_rms_norm_f32,  nullptr,                     nullptr                              },
        { "rope",             spacemit_kernel_rope,          nullptr,                     spacemit_rope_workspace              },
        { "set_rows",         spacemit_kernel_set_rows,      nullptr,                     nullptr                              },
        { "cpy.quant",        spacemit_kernel_cpy_quant,     nullptr,                     nullptr                              },
        { "get_rows.f32",     spacemit_kernel_get_rows_f32,  nullptr,                     nullptr                              },
        { "swiglu.f32",       spacemit_kernel_swiglu_f32,    nullptr,                     nullptr                              },
        { "mul_mat.quantize", spacemit_mul_mat_quantize,     nullptr,                     spacemit_mul_mat_workspace           },
        { "mul_mat.gemm",     spacemit_mul_mat_gemm,         nullptr,                     spacemit_mul_mat_workspace           },
        { "flash_attn",       spacemit_kernel_flash_attn,    spacemit_flash_attn_supported, nullptr                             },
        { "flash_attn.split.partial", spacemit_kernel_flash_attn_split_partial, spacemit_flash_attn_supported,
          spacemit_flash_attn_split_workspace                                                                                  },
        { "flash_attn.split.partial_paired", spacemit_kernel_flash_attn_split_partial_paired, spacemit_flash_attn_supported,
          spacemit_flash_attn_split_workspace                                                                                  },
        { "flash_attn.split.merge", spacemit_kernel_flash_attn_split_merge, spacemit_flash_attn_supported,
          spacemit_flash_attn_split_workspace                                                                                  },
    };
    if (count != nullptr) {
        *count = sizeof(kernels) / sizeof(kernels[0]);
    }
    return kernels;
}

// the captured steps of one split: one launch with a barrier after every step, each tile running every step
// (plan.md §2.4). The test-only flags run one launch per step instead (E3a, E3b)
enum spacemit_status plugin_replay(spacemit_plugin_stream * stream, const spacemit_plugin_step * steps,
                                   uint32_t n_steps, void * workspace, size_t workspace_size, uint32_t flags) {
    if (stream == nullptr || (steps == nullptr && n_steps > 0)) {
        return SPACEMIT_STATUS_INVALID_ARGUMENT;
    }
    bool ok = true;
    try {
        if ((flags & SPACEMIT_REPLAY_ORDERED_LAUNCHES) != 0) {
            ok = stream->executor->run_ordered(steps, n_steps, workspace, workspace_size);
        } else if ((flags & SPACEMIT_REPLAY_SYNC_LAUNCH) != 0) {
            // the same steps in the same order on the same workspace, so the same outputs: a launch ends on every
            // tile before the next starts, as the barrier after a step does inside one launch
            for (uint32_t i = 0; i < n_steps && ok; i++) {
                ok = stream->executor->run(steps + i, 1, workspace, workspace_size);
            }
        } else {
            ok = stream->executor->run(steps, n_steps, workspace, workspace_size);
        }
    } catch (...) {
        GGML_LOG_ERROR("FlagOS SpacemiT: the launch threw an exception\n");
        return SPACEMIT_STATUS_FAILED;
    }
    return ok ? SPACEMIT_STATUS_SUCCESS : SPACEMIT_STATUS_FAILED;
}

// test-only (E3c): where the tiles of this stream's launches note their times, or nullptr
void spacemit_run_stats_attach(spacemit_plugin_stream * stream, struct spacemit_run_stats * stats) {
    if (stream != nullptr) {
        stream->executor->stats = stats;
    }
}

// what is not contract (plan.md "S1 design" §2, A2): the test counters, by their current names
void * plugin_get_proc_address(const char * name) {
    if (name == nullptr) {
        return nullptr;
    }
    if (std::strcmp(name, "spacemit_steps_take") == 0) {
        return reinterpret_cast<void *>(static_cast<spacemit_steps_take_fn>(spacemit_steps_take));
    }
    if (std::strcmp(name, "spacemit_ordered_counts_take") == 0) {
        return reinterpret_cast<void *>(static_cast<spacemit_ordered_counts_take_fn>(spacemit_ordered_counts_take));
    }
    if (std::strcmp(name, "spacemit_mm_paths_take") == 0) {
        return reinterpret_cast<void *>(static_cast<spacemit_mm_paths_take_fn>(spacemit_mm_paths_take));
    }
    if (std::strcmp(name, "spacemit_mm_tail_take") == 0) {
        return reinterpret_cast<void *>(static_cast<spacemit_mm_tail_take_fn>(spacemit_mm_tail_take));
    }
    if (std::strcmp(name, "spacemit_run_stats_attach") == 0) {
        return reinterpret_cast<void *>(static_cast<spacemit_run_stats_attach_fn>(spacemit_run_stats_attach));
    }
    return nullptr;
}

// kept for S2, where the plugin logs through the core; copied, never a pointer to the caller's table, which may be a
// local of the caller (the check program's second load). Only the first call sets them: the core's own load at
// registration is always the first caller
void (*g_log)(int level, const char * text) = nullptr;
std::string g_host_dir;

}  // namespace

const spacemit_plugin_v1 * flagos_spacemit_plugin_v1(const spacemit_host_v1 * host) {
    if (host == nullptr || host->struct_size < sizeof(spacemit_host_v1)) {
        return nullptr;
    }
    if (g_log == nullptr) {
        g_log      = host->log;
        g_host_dir = host->plugin_dir != nullptr ? host->plugin_dir : "";
    }

    static const spacemit_plugin_v1 plugin = {
        /* .api_version        = */ FLAGOS_SPACEMIT_PLUGIN_API_VERSION,
        /* .struct_size        = */ sizeof(spacemit_plugin_v1),
        /* .device_count       = */ plugin_device_count,
        /* .device_open        = */ plugin_device_open,
        /* .device_close       = */ plugin_device_close,
        /* .device_props       = */ plugin_device_props,
        /* .device_memory      = */ plugin_device_memory,
        /* .buffer_alloc       = */ plugin_buffer_alloc,
        /* .buffer_free        = */ plugin_buffer_free,
        /* .buffer_base        = */ plugin_buffer_base,
        /* .tensor_layout      = */ plugin_tensor_layout,
        /* .tensor_alloc_size  = */ plugin_tensor_alloc_size,
        /* .tensor_set         = */ plugin_tensor_set,
        /* .tensor_get         = */ plugin_tensor_get,
        /* .tensor_fill        = */ plugin_tensor_fill,
        /* .stream_create      = */ plugin_stream_create,
        /* .stream_destroy     = */ plugin_stream_destroy,
        /* .stream_synchronize = */ plugin_stream_synchronize,
        /* .kernels            = */ plugin_kernels,
        /* .replay             = */ plugin_replay,
        /* .get_proc_address   = */ plugin_get_proc_address,
        /* .register_host_ptr  = */ nullptr,  // zero copy: S2 (plan.md "S1 design" A6)
        /* .tensor_prepare     = */ nullptr,
    };
    return &plugin;
}
