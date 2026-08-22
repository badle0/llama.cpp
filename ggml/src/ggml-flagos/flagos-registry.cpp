#include "ggml-flagos.h"
#include "flagos-provider.h"

#include "../ggml-backend-impl.h"
#include "../ggml-impl.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

struct flagos_registered_device {
    const flagos_provider_v1 * provider;
    size_t local_index;
    ggml_backend_dev_t device;
    flagos_device_identity identity;
    flagos_device_caps caps;
};

static std::vector<const flagos_provider_v1 *> flagos_providers;
static std::vector<flagos_registered_device> flagos_devices;
static std::atomic<int> flagos_active_device { -1 };

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

static std::vector<const flagos_provider_v1 *> flagos_builtin_providers() {
    std::vector<const flagos_provider_v1 *> providers;
#ifdef GGML_FLAGOS_HAVE_DENGLIN
    providers.push_back(flagos_denglin_provider());
#endif
#ifdef GGML_FLAGOS_HAVE_AMD
    providers.push_back(flagos_amd_provider());
#endif
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
        for (const flagos_provider_v1 * provider : flagos_builtin_providers()) {
            if (!flagos_provider_is_valid(provider)) {
                GGML_LOG_ERROR("FlagOS: invalid built-in provider descriptor\n");
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
                if (device == nullptr ||
                    !provider->device_identity(local_index, &identity) ||
                    !flagos_device_identity_is_valid(provider, local_index, &identity) ||
                    flagos_device_identity_is_registered(identity) ||
                    !provider->device_caps(local_index, &caps) ||
                    !flagos_device_caps_are_valid(&caps)) {
                    GGML_LOG_ERROR("FlagOS: provider %s returned an invalid device at index %zu\n", provider->identity.name, local_index);
                    continue;
                }
                device->reg = &reg;
                flagos_devices.push_back({ provider, local_index, device, identity, caps });
            }
        }
        if (!flagos_devices.empty()) {
            flagos_active_device.store(0);
        }
    });
    return &reg;
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
    ggml_backend_register(ggml_backend_flagos_reg());
}

static int ggml_backend_flagos_score() {
    int score = 0;
    for (const flagos_provider_v1 * provider : flagos_builtin_providers()) {
        if (flagos_provider_is_valid(provider)) {
            score += provider->score();
        }
    }
    return score;
}

GGML_BACKEND_DL_IMPL(ggml_backend_flagos_reg)
GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_flagos_score)
