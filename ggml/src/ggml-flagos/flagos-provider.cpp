#include "flagos-provider.h"

#include "../ggml-backend-impl.h"

bool flagos_provider_is_valid(const flagos_provider_v1 * provider) {
    return provider != nullptr &&
        provider->api_version == FLAGOS_PROVIDER_API_VERSION &&
        provider->struct_size >= FLAGOS_PROVIDER_V1_REQUIRED_SIZE &&
        provider->identity.id != 0 &&
        provider->identity.name != nullptr &&
        provider->identity.name[0] != '\0' &&
        provider->probe != nullptr &&
        provider->device_count != nullptr &&
        provider->device_get != nullptr &&
        provider->device_identity != nullptr &&
        provider->device_caps != nullptr &&
        provider->is_backend != nullptr &&
        provider->set_device != nullptr &&
        provider->get_device != nullptr &&
        provider->score != nullptr;
}

bool flagos_device_identity_is_valid(
        const flagos_provider_v1 * provider,
        size_t local_index,
        const flagos_device_identity * identity) {
    return provider != nullptr &&
        identity != nullptr &&
        local_index <= UINT32_MAX &&
        identity->provider_id == provider->identity.id &&
        identity->ordinal == local_index &&
        (identity->uuid_hi != 0 || identity->uuid_lo != 0);
}

bool flagos_device_caps_are_valid(const flagos_device_caps * caps) {
    if (caps == nullptr || caps->memory == 0) {
        return false;
    }
    switch (caps->kind) {
        case flagos_provider_kind::gpu:
        case flagos_provider_kind::ai_accelerator:
        case flagos_provider_kind::cpu_accelerator:
            break;
        default:
            return false;
    }
    return (caps->execution & FLAGOS_EXECUTION_AOT_MODULE) == 0 ||
        (caps->aot_format != nullptr && caps->aot_format[0] != '\0');
}

static flagos_engine_kind flagos_engine_from_kind(flagos_provider_kind kind) {
    switch (kind) {
        case flagos_provider_kind::gpu:
            return flagos_engine_kind::gpu;
        case flagos_provider_kind::ai_accelerator:
            return flagos_engine_kind::npu;
        case flagos_provider_kind::cpu_accelerator:
            return flagos_engine_kind::cpu;
    }
    return flagos_engine_kind::other;
}

bool flagos_provider_get_device_profile(
        const flagos_provider_v1 * provider,
        size_t local_index,
        flagos_device_profile * profile) {
    if (provider == nullptr || profile == nullptr || !flagos_provider_is_valid(provider) ||
        local_index > UINT32_MAX) {
        return false;
    }
    *profile = {};
    profile->struct_size = sizeof(flagos_device_profile);
    profile->version = 1;

    if (provider->struct_size >= sizeof(flagos_provider_v1) &&
        provider->get_device_profile != nullptr &&
        provider->get_device_profile(local_index, profile)) {
        return flagos_device_profile_is_valid(profile);
    }

    flagos_device_caps caps {};
    if (!provider->device_caps(local_index, &caps) ||
        !flagos_device_caps_are_valid(&caps)) {
        return false;
    }
    ggml_backend_dev_t device = provider->device_get(local_index);
    if (device == nullptr) {
        return false;
    }
    ggml_backend_dev_props props {};
    const bool has_props = device->iface.get_props != nullptr;
    if (has_props) {
        ggml_backend_dev_get_props(device, &props);
    }
    static const char unknown_arch[] = "unknown";
    static const char unknown_target[] = "unknown";
    profile->engine = flagos_engine_from_kind(caps.kind);
    profile->features = 0;
    if ((caps.memory & FLAGOS_MEMORY_UNIFIED_COHERENT) != 0) {
        profile->features |= FLAGOS_FEATURE_UNIFIED_MEMORY;
    }
    flagos_device_identity identity {};
    if (provider->device_identity(local_index, &identity)) {
        profile->memory_domain_id = identity.memory_domain_id;
    }
    profile->vendor = provider->identity.name;
    // The provider kind is authoritative when an older device does not expose
    // standard GGML properties.  Properties refine the label when available,
    // but must not turn a GPU/NPU provider into a CPU profile due to a zeroed
    // compatibility struct.
    if (caps.kind == flagos_provider_kind::cpu_accelerator) {
        profile->architecture = "cpu-accelerator";
    } else if (caps.kind == flagos_provider_kind::ai_accelerator) {
        profile->architecture = "ai-accelerator";
    } else if (has_props && props.type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
        profile->architecture = "integrated-gpu";
    } else if (caps.kind == flagos_provider_kind::gpu) {
        profile->architecture = "gpu";
    } else {
        profile->architecture = "accelerator";
    }
    profile->microarchitecture = unknown_arch;
    profile->target = props.description != nullptr ? props.description : unknown_target;
    profile->runtime = nullptr;
    profile->aot_format = caps.aot_format;
    return flagos_device_profile_is_valid(profile);
}
