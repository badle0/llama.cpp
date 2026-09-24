#include "ggml-flagos.h"
#include "flagos-provider.h"
#include "flagos-registry.h"

#include "../ggml-backend-impl.h"
#include "../ggml-impl.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

#ifdef GGML_FLAGOS_HAVE_DENGLIN
const flagos_provider_v1 * flagos_denglin_provider();
#endif

#ifdef GGML_FLAGOS_HAVE_AMD
const flagos_provider_v1 * flagos_amd_provider();
#endif

struct flagos_registered_device {
    const flagos_provider_v1 * provider;
    size_t local_index;
    ggml_backend_dev_t device;
    flagos_device_identity identity;
    flagos_device_caps caps;
    flagos_device_profile profile;
};

static std::vector<const flagos_provider_v1 *> flagos_providers;
static std::vector<const flagos_provider_v1 *> flagos_extra_providers;
static std::vector<flagos_registered_device> flagos_devices;
static std::atomic<int> flagos_active_device { -1 };
static std::atomic<bool> flagos_registry_initialized { false };
static std::mutex flagos_provider_mutex;

const char * flagos_status_name(flagos_status status) {
    switch (status) {
        case FLAGOS_STATUS_SUCCESS:             return "success";
        case FLAGOS_STATUS_INVALID_ARGUMENT:    return "invalid_argument";
        case FLAGOS_STATUS_NOT_FOUND:           return "not_found";
        case FLAGOS_STATUS_AMBIGUOUS:           return "ambiguous";
        case FLAGOS_STATUS_NO_DEVICE:           return "no_device";
        case FLAGOS_STATUS_UNSUPPORTED_POLICY:  return "unsupported_policy";
    }
    return "unknown";
}

static bool flagos_provider_id_is_registered(uint64_t provider_id) {
    for (const flagos_provider_v1 * provider : flagos_providers) {
        if (provider->identity.id == provider_id) {
            return true;
        }
    }
    return false;
}

static bool flagos_device_identity_is_registered(const flagos_device_identity & identity) {
    for (const flagos_registered_device & registered : flagos_devices) {
        if (registered.identity.provider_id == identity.provider_id &&
            registered.identity.uuid_hi == identity.uuid_hi &&
            registered.identity.uuid_lo == identity.uuid_lo) {
            return true;
        }
    }
    return false;
}

static std::vector<const flagos_provider_v1 *> flagos_compiled_providers() {
    std::vector<const flagos_provider_v1 *> providers;
#ifdef GGML_FLAGOS_HAVE_DENGLIN
    providers.push_back(flagos_denglin_provider());
#endif
#ifdef GGML_FLAGOS_HAVE_AMD
    providers.push_back(flagos_amd_provider());
#endif
    return providers;
}

static std::vector<const flagos_provider_v1 *> flagos_all_providers() {
    std::vector<const flagos_provider_v1 *> providers = flagos_compiled_providers();
    {
        std::lock_guard<std::mutex> lock(flagos_provider_mutex);
        providers.insert(providers.end(), flagos_extra_providers.begin(), flagos_extra_providers.end());
    }
    return providers;
}

static const char * ggml_backend_flagos_reg_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "FlagOS";
}

static size_t ggml_backend_flagos_reg_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return flagos_devices.size();
}

static ggml_backend_dev_t ggml_backend_flagos_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_UNUSED(reg);
    GGML_ASSERT(index < flagos_devices.size());
    return flagos_devices[index].device;
}

static void * ggml_backend_flagos_reg_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (std::strcmp(name, "ggml_backend_flagos_init") == 0) {
        return reinterpret_cast<void *>(ggml_backend_flagos_init);
    }
    if (std::strcmp(name, "ggml_backend_flagos_buffer_type") == 0) {
        return reinterpret_cast<void *>(ggml_backend_flagos_buffer_type);
    }
    if (std::strcmp(name, "flagos_registry_get_device_profile") == 0) {
        return reinterpret_cast<void *>(flagos_registry_get_device_profile);
    }
    if (std::strcmp(name, "flagos_registry_get_device_info") == 0) {
        return reinterpret_cast<void *>(flagos_registry_get_device_info);
    }
    if (std::strcmp(name, "flagos_select_provider_v1") == 0) {
        return reinterpret_cast<void *>(flagos_select_provider_v1);
    }
    if (std::strcmp(name, "flagos_status_name") == 0) {
        return reinterpret_cast<void *>(flagos_status_name);
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_flagos_reg_interface = {
    /* .get_name         = */ ggml_backend_flagos_reg_name,
    /* .get_device_count = */ ggml_backend_flagos_reg_device_count,
    /* .get_device       = */ ggml_backend_flagos_reg_device_get,
    /* .get_proc_address = */ ggml_backend_flagos_reg_proc_address,
};

ggml_backend_reg_t ggml_backend_flagos_reg() {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_flagos_reg_interface,
        /* .context     = */ nullptr,
    };
    static std::once_flag once;
    std::call_once(once, [&]() {
        // Mark the registry closed before taking the provider snapshot. This
        // prevents a concurrent out-of-tree registration from being accepted
        // after enumeration has already started.
        flagos_registry_initialized.store(true, std::memory_order_release);
        for (const flagos_provider_v1 * provider : flagos_all_providers()) {
            if (!flagos_provider_is_valid(provider)) {
                GGML_LOG_ERROR("FlagOS: invalid provider descriptor\n");
                continue;
            }
            if (flagos_provider_id_is_registered(provider->identity.id)) {
                GGML_LOG_ERROR("FlagOS: duplicate provider id for %s\n", provider->identity.name);
                continue;
            }
            if (!provider->probe(&reg)) {
                GGML_LOG_WARN("FlagOS: provider %s probe failed\n", provider->identity.name);
                continue;
            }
            flagos_providers.push_back(provider);
            const size_t count = provider->device_count();
            for (size_t local_index = 0; local_index < count; ++local_index) {
                ggml_backend_dev_t device = provider->device_get(local_index);
                flagos_device_identity identity {};
                flagos_device_caps caps {};
                flagos_device_profile profile {};
                if (device == nullptr ||
                    !provider->device_identity(local_index, &identity) ||
                    !flagos_device_identity_is_valid(provider, local_index, &identity) ||
                    flagos_device_identity_is_registered(identity) ||
                    !provider->device_caps(local_index, &caps) ||
                    !flagos_device_caps_are_valid(&caps) ||
                    !flagos_provider_get_device_profile(provider, local_index, &profile)) {
                    GGML_LOG_ERROR("FlagOS: provider %s returned an invalid device at index %zu\n", provider->identity.name, local_index);
                    continue;
                }
                device->reg = &reg;
                flagos_devices.push_back({ provider, local_index, device, identity, caps, profile });
            }
        }
        if (!flagos_devices.empty()) {
            flagos_active_device.store(0);
        }
    });
    return &reg;
}

bool flagos_registry_register_provider(const flagos_provider_v1 * provider) {
    if (!flagos_provider_is_valid(provider) ||
        flagos_registry_initialized.load(std::memory_order_acquire)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(flagos_provider_mutex);
    if (flagos_registry_initialized.load(std::memory_order_relaxed)) {
        return false;
    }
    for (const flagos_provider_v1 * existing : flagos_compiled_providers()) {
        if (existing != nullptr && existing->identity.id == provider->identity.id) {
            return false;
        }
    }
    for (const flagos_provider_v1 * existing : flagos_extra_providers) {
        if (existing->identity.id == provider->identity.id) {
            return false;
        }
    }
    flagos_extra_providers.push_back(provider);
    return true;
}

bool flagos_registry_get_device_profile(
        size_t global_index, flagos_device_profile * profile) {
    flagos_device_info info {};
    if (!flagos_registry_get_device_info(global_index, &info) || profile == nullptr) {
        return false;
    }
    *profile = info.profile;
    return true;
}

bool flagos_registry_get_device_info(
        size_t global_index, flagos_device_info * info) {
    ggml_backend_flagos_reg();
    if (info == nullptr || global_index >= flagos_devices.size()) {
        return false;
    }
    const flagos_registered_device & registered = flagos_devices[global_index];
    *info = {};
    info->struct_size = sizeof(flagos_device_info);
    info->version = 1;
    info->global_index = global_index;
    info->local_index = registered.local_index;
    if (registered.device->iface.get_name != nullptr) {
        info->name = registered.device->iface.get_name(registered.device);
    }
    if (registered.device->iface.get_description != nullptr) {
        info->description = registered.device->iface.get_description(registered.device);
    }
    info->provider = registered.provider->identity;
    info->identity = registered.identity;
    info->caps = registered.caps;
    info->profile = registered.profile;
    return true;
}

static bool flagos_selector_has_filter(const flagos_provider_selector_v1 * selector) {
    return selector != nullptr &&
        (selector->provider_id != 0 ||
         selector->provider_name != nullptr ||
         selector->device_uuid_hi != 0 ||
         selector->device_uuid_lo != 0 ||
         selector->local_device_index != FLAGOS_DEVICE_INDEX_UNSPECIFIED);
}

static bool flagos_selector_matches(
        const flagos_registered_device & device,
        const flagos_provider_selector_v1 * selector) {
    if (selector->provider_id != 0 &&
        device.identity.provider_id != selector->provider_id) {
        return false;
    }
    if (selector->provider_name != nullptr &&
        std::strcmp(device.provider->identity.name, selector->provider_name) != 0) {
        return false;
    }
    if ((selector->device_uuid_hi != 0 || selector->device_uuid_lo != 0) &&
        (device.identity.uuid_hi != selector->device_uuid_hi ||
         device.identity.uuid_lo != selector->device_uuid_lo)) {
        return false;
    }
    return selector->local_device_index == FLAGOS_DEVICE_INDEX_UNSPECIFIED ||
        device.local_index == selector->local_device_index;
}

flagos_status flagos_select_provider_v1(
        const flagos_provider_selector_v1 * selector,
        flagos_provider_selection_v1 * selection) {
    if (selector == nullptr || selection == nullptr ||
        selector->struct_size < sizeof(flagos_provider_selector_v1) ||
        selector->version != 1 || selection->struct_size < sizeof(flagos_provider_selection_v1) ||
        selection->version != 1 || selector->flags != 0 ||
        (selector->provider_name != nullptr && selector->provider_name[0] == '\0')) {
        return FLAGOS_STATUS_INVALID_ARGUMENT;
    }
    if (selector->policy != FLAGOS_SELECT_EXPLICIT &&
        selector->policy != FLAGOS_SELECT_CONFIGURED &&
        selector->policy != FLAGOS_SELECT_AUTO_SINGLE) {
        return FLAGOS_STATUS_UNSUPPORTED_POLICY;
    }
    if ((selector->policy == FLAGOS_SELECT_EXPLICIT ||
         selector->policy == FLAGOS_SELECT_CONFIGURED) &&
        !flagos_selector_has_filter(selector)) {
        return FLAGOS_STATUS_INVALID_ARGUMENT;
    }

    ggml_backend_flagos_reg();
    size_t match_count = 0;
    size_t match_index = 0;
    for (size_t index = 0; index < flagos_devices.size(); ++index) {
        if (flagos_selector_matches(flagos_devices[index], selector)) {
            match_index = index;
            ++match_count;
        }
    }
    if (match_count == 0) {
        return flagos_devices.empty() ? FLAGOS_STATUS_NO_DEVICE : FLAGOS_STATUS_NOT_FOUND;
    }
    if (match_count != 1) {
        return FLAGOS_STATUS_AMBIGUOUS;
    }

    const flagos_registered_device & registered = flagos_devices[match_index];
    *selection = {};
    selection->struct_size = sizeof(flagos_provider_selection_v1);
    selection->version = 1;
    selection->provider_id = registered.provider->identity.id;
    selection->provider_name = registered.provider->identity.name;
    selection->global_device_index = match_index;
    selection->local_device_index = registered.local_index;
    selection->device = registered.identity;
    selection->caps = registered.caps;
    selection->profile = registered.profile;
    return FLAGOS_STATUS_SUCCESS;
}

ggml_backend_t ggml_backend_flagos_init(int device) {
    ggml_backend_reg_t reg = ggml_backend_flagos_reg();
    if (device < 0 || static_cast<size_t>(device) >= flagos_devices.size()) {
        return nullptr;
    }
    const flagos_registered_device & selected = flagos_devices[device];
    if (!selected.provider->set_device(selected.local_index)) {
        return nullptr;
    }
    flagos_active_device.store(device);
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, device), nullptr);
}

bool ggml_backend_is_flagos(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    ggml_backend_flagos_reg();
    for (const flagos_provider_v1 * provider : flagos_providers) {
        if (provider->is_backend(backend)) {
            return true;
        }
    }
    return false;
}

ggml_backend_buffer_type_t ggml_backend_flagos_buffer_type(int device) {
    ggml_backend_reg_t reg = ggml_backend_flagos_reg();
    if (device < 0 || static_cast<size_t>(device) >= flagos_devices.size()) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, device));
}

void ggml_backend_flagos_set_device(int device) {
    ggml_backend_flagos_reg();
    if (device < 0 || static_cast<size_t>(device) >= flagos_devices.size()) {
        GGML_ABORT("FlagOS: invalid device %d", device);
    }
    const flagos_registered_device & selected = flagos_devices[device];
    if (!selected.provider->set_device(selected.local_index)) {
        GGML_ABORT("FlagOS: failed to select device %d", device);
    }
    flagos_active_device.store(device);
}

int ggml_backend_flagos_get_device() {
    ggml_backend_flagos_reg();
    const int active = flagos_active_device.load();
    if (active >= 0 && static_cast<size_t>(active) < flagos_devices.size()) {
        const flagos_registered_device & selected = flagos_devices[active];
        if (selected.provider->get_device() == static_cast<int>(selected.local_index)) {
            return active;
        }
    }
    return -1;
}

void ggml_backend_flagos_reg_devices() {
    // Registration is done by ggml-backend-reg.cpp (GGML_USE_FLAGOS) or by the
    // dynamic loader. Calling ggml_backend_register() from here would make this
    // backend library depend on libggml, which links it (undefined on macOS).
}

static int ggml_backend_flagos_score() {
    int score = 0;
    for (const flagos_provider_v1 * provider : flagos_all_providers()) {
        if (flagos_provider_is_valid(provider)) {
            score += provider->score();
        }
    }
    return score;
}

GGML_BACKEND_DL_IMPL(ggml_backend_flagos_reg)
GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_flagos_score)
