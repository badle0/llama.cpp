#pragma once

#include "flagos-provider.h"

#include <cstddef>
#include <cstdint>

// Stable snapshot of one globally registered execution engine. Pointer fields
// are owned by the provider and remain valid while the registry is alive.
struct flagos_device_info {
    uint32_t struct_size = 0;
    uint32_t version = 1;
    size_t global_index = 0;
    size_t local_index = 0;
    const char * name = nullptr;
    const char * description = nullptr;
    flagos_provider_identity provider {};
    flagos_device_identity identity {};
    flagos_device_caps caps {};
    flagos_device_profile profile {};
};

enum flagos_status : uint32_t {
    FLAGOS_STATUS_SUCCESS = 0,
    FLAGOS_STATUS_INVALID_ARGUMENT,
    FLAGOS_STATUS_NOT_FOUND,
    FLAGOS_STATUS_AMBIGUOUS,
    FLAGOS_STATUS_NO_DEVICE,
    FLAGOS_STATUS_UNSUPPORTED_POLICY,
};

enum flagos_selection_policy_v1 : uint32_t {
    FLAGOS_SELECT_EXPLICIT = 0,
    FLAGOS_SELECT_CONFIGURED = 1,
    FLAGOS_SELECT_AUTO_SINGLE = 2,
};

static constexpr uint32_t FLAGOS_DEVICE_INDEX_UNSPECIFIED = UINT32_MAX;

// Selector fields are optional filters.  An explicit or configured selection
// must provide at least one filter; auto-single accepts an empty selector only
// when exactly one FlagOS device is available.  A UUID is unspecified only
// when both halves are zero; a provider may legitimately expose a UUID with
// one zero half.
struct flagos_provider_selector_v1 {
    uint32_t struct_size = 0;
    uint32_t version = 1;
    uint64_t provider_id = 0;
    const char * provider_name = nullptr;
    uint64_t device_uuid_hi = 0;
    uint64_t device_uuid_lo = 0;
    uint32_t local_device_index = FLAGOS_DEVICE_INDEX_UNSPECIFIED;
    uint32_t policy = FLAGOS_SELECT_EXPLICIT;
    uint32_t flags = 0;
};

struct flagos_provider_selection_v1 {
    uint32_t struct_size = 0;
    uint32_t version = 1;
    uint64_t provider_id = 0;
    const char * provider_name = nullptr;
    size_t global_device_index = 0;
    size_t local_device_index = 0;
    flagos_device_identity device {};
    flagos_device_caps caps {};
    flagos_device_profile profile {};
};

#ifdef __cplusplus
extern "C" {
#endif

// Register an out-of-tree provider before the FlagOS registry is first used.
// This is the extension point for future CPU/ARM, NPU, DSP, or vendor
// providers; the common registry does not need a vendor-specific preprocessor
// branch for them. Registration after initialization is rejected.  C linkage
// is intentional: a dynamic provider can resolve this symbol without knowing
// the C++ mangled name.
GGML_BACKEND_API bool flagos_registry_register_provider(const flagos_provider_v1 * provider);

// Copy the immutable profile for a registered global FlagOS device. The
// returned strings remain owned by the provider and are valid while the
// registry is alive.
GGML_BACKEND_API bool flagos_registry_get_device_profile(
        size_t global_index, flagos_device_profile * profile);

GGML_BACKEND_API bool flagos_registry_get_device_info(
        size_t global_index, flagos_device_info * info);

GGML_BACKEND_API const char * flagos_status_name(flagos_status status);

// Resolve one execution target before model/session creation.  This performs
// provider/device selection only; it never compares graph candidates or
// runtime performance across providers.
GGML_BACKEND_API flagos_status flagos_select_provider_v1(
        const flagos_provider_selector_v1 * selector,
        flagos_provider_selection_v1 * selection);

#ifdef __cplusplus
}
#endif
