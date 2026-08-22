#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

enum class flagos_provider_kind : uint32_t {
    gpu,
    ai_accelerator,
    cpu_accelerator,
};

enum flagos_memory_cap : uint64_t {
    FLAGOS_MEMORY_DEVICE_LOCAL     = 1ull << 0,
    FLAGOS_MEMORY_HOST_PINNED      = 1ull << 1,
    FLAGOS_MEMORY_UNIFIED_COHERENT = 1ull << 2,
    FLAGOS_MEMORY_HOST_VISIBLE     = 1ull << 3,
};

enum flagos_execution_cap : uint64_t {
    FLAGOS_EXECUTION_ASYNC_QUEUE  = 1ull << 0,
    FLAGOS_EXECUTION_EVENTS       = 1ull << 1,
    FLAGOS_EXECUTION_NATIVE_GRAPH = 1ull << 2,
    FLAGOS_EXECUTION_GRAPH_UPDATE = 1ull << 3,
    FLAGOS_EXECUTION_AOT_MODULE   = 1ull << 4,
};

struct flagos_provider_identity {
    uint64_t id;
    const char * name;
    uint32_t version;
};

struct flagos_device_identity {
    uint64_t provider_id;
    uint64_t uuid_hi;
    uint64_t uuid_lo;
    uint64_t memory_domain_id;
    uint32_t ordinal;
};

struct flagos_device_caps {
    flagos_provider_kind kind;
    uint64_t memory;
    uint64_t execution;
    const char * aot_format;
};

struct flagos_provider_v1 {
    uint32_t api_version;
    uint32_t struct_size;
    flagos_provider_identity identity;

    bool (*probe)(ggml_backend_reg_t owner_reg);
    size_t (*device_count)();
    ggml_backend_dev_t (*device_get)(size_t index);
    bool (*device_identity)(size_t index, flagos_device_identity * identity);
    bool (*device_caps)(size_t index, flagos_device_caps * caps);

    bool (*is_backend)(ggml_backend_t backend);
    bool (*set_device)(size_t index);
    int (*get_device)();
    int (*score)();
};

static constexpr uint32_t FLAGOS_PROVIDER_API_VERSION = 1;

bool flagos_provider_is_valid(const flagos_provider_v1 * provider);
bool flagos_device_identity_is_valid(
    const flagos_provider_v1 * provider,
    size_t local_index,
    const flagos_device_identity * identity);
bool flagos_device_caps_are_valid(const flagos_device_caps * caps);

#ifdef GGML_FLAGOS_HAVE_DENGLIN
const flagos_provider_v1 * flagos_denglin_provider();
#endif

#ifdef GGML_FLAGOS_HAVE_AMD
const flagos_provider_v1 * flagos_amd_provider();
#endif
