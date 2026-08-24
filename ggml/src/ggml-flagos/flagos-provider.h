#pragma once

#include "ggml-backend.h"
#include "flagos-target.h"

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
    // The UUID identifies an execution engine as exposed by this provider.
    // A SoC with CPU/GPU/NPU engines may therefore use one UUID per engine,
    // while memory_domain_id remains shared when the engines use one pool.
    uint64_t uuid_hi;
    uint64_t uuid_lo;
    // Shared physical memory domain. Zero is valid when the provider cannot
    // describe sharing; it is not used for device de-duplication.
    uint64_t memory_domain_id;
    uint32_t ordinal;
};

struct flagos_device_caps {
    flagos_provider_kind kind;
    uint64_t memory;
    uint64_t execution;
    const char * aot_format;
};

using flagos_get_device_profile_fn = bool (*) (
        size_t index, flagos_device_profile * profile);

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

    // Optional ABI tail. Providers built against the original v1 header have
    // a shorter struct_size and are still valid; the registry derives a
    // conservative profile from the standard GGML device properties.
    flagos_get_device_profile_fn get_device_profile;
};

static constexpr uint32_t FLAGOS_PROVIDER_API_VERSION = 1;
static constexpr size_t FLAGOS_PROVIDER_V1_REQUIRED_SIZE =
    offsetof(flagos_provider_v1, get_device_profile);

GGML_BACKEND_API bool flagos_provider_is_valid(const flagos_provider_v1 * provider);
GGML_BACKEND_API bool flagos_device_identity_is_valid(
    const flagos_provider_v1 * provider,
    size_t local_index,
    const flagos_device_identity * identity);
GGML_BACKEND_API bool flagos_device_caps_are_valid(const flagos_device_caps * caps);
GGML_BACKEND_API bool flagos_provider_get_device_profile(
    const flagos_provider_v1 * provider,
    size_t local_index,
    flagos_device_profile * profile);
