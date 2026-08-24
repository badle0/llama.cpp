#pragma once

#include "flagos-provider.h"

#include <cstddef>

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

// Register an out-of-tree provider before the FlagOS registry is first used.
// This is the extension point for future CPU/ARM, NPU, DSP, or vendor
// providers; the common registry does not need a vendor-specific preprocessor
// branch for them. Registration after initialization is rejected.
GGML_BACKEND_API bool flagos_registry_register_provider(const flagos_provider_v1 * provider);

// Copy the immutable profile for a registered global FlagOS device. The
// returned strings remain owned by the provider and are valid while the
// registry is alive.
GGML_BACKEND_API bool flagos_registry_get_device_profile(
        size_t global_index, flagos_device_profile * profile);

GGML_BACKEND_API bool flagos_registry_get_device_info(
        size_t global_index, flagos_device_info * info);
