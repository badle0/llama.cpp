#include "flagos-provider.h"

bool flagos_provider_is_valid(const flagos_provider_v1 * provider) {
    return provider != nullptr &&
        provider->api_version == FLAGOS_PROVIDER_API_VERSION &&
        provider->struct_size >= sizeof(flagos_provider_v1) &&
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
