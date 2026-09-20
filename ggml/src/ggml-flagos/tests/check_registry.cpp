#include "flagos-provider.h"
#include "flagos-registry.h"
#include "ggml-flagos.h"

#include "../../ggml-backend-impl.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

static void check(bool condition, const char * expression, int line) {
    if (!condition) {
        std::fprintf(stderr, "check failed at line %d: %s\n", line, expression);
        std::abort();
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

const flagos_provider_v1 * flagos_denglin_provider();
const flagos_provider_v1 * flagos_amd_provider();

static constexpr uint64_t DENGLIN_ID = 1;
static constexpr uint64_t AMD_ID = 2;
static constexpr uint64_t ARM_ACCEL_ID = 3;

static ggml_backend_device denglin_devices[2] {};
static ggml_backend_device amd_devices[1] {};
static ggml_backend_device arm_accel_devices[1] {};
static int denglin_current_device = 0;
static int amd_current_device = 0;
static int arm_accel_current_device = 0;

static bool mock_probe(ggml_backend_reg_t) {
    return true;
}

static size_t denglin_device_count() {
    return 2;
}

static ggml_backend_dev_t denglin_device_get(size_t index) {
    return index < 2 ? &denglin_devices[index] : nullptr;
}

static bool denglin_device_identity(size_t index, flagos_device_identity * identity) {
    if (index >= 2 || identity == nullptr) {
        return false;
    }
    *identity = { DENGLIN_ID, DENGLIN_ID, index + 1, index + 10, static_cast<uint32_t>(index) };
    return true;
}

static bool mock_device_caps(size_t, flagos_device_caps * caps) {
    if (caps == nullptr) {
        return false;
    }
    *caps = {
        flagos_provider_kind::gpu,
        FLAGOS_MEMORY_DEVICE_LOCAL,
        FLAGOS_EXECUTION_ASYNC_QUEUE,
        nullptr,
    };
    return true;
}

static bool mock_is_backend(ggml_backend_t) {
    return false;
}

static bool denglin_set_device(size_t index) {
    if (index >= 2) {
        return false;
    }
    denglin_current_device = static_cast<int>(index);
    return true;
}

static int denglin_get_device() {
    return denglin_current_device;
}

static size_t amd_device_count() {
    return 1;
}

static ggml_backend_dev_t amd_device_get(size_t index) {
    return index == 0 ? &amd_devices[0] : nullptr;
}

static bool amd_device_identity(size_t index, flagos_device_identity * identity) {
    if (index != 0 || identity == nullptr) {
        return false;
    }
    *identity = { AMD_ID, AMD_ID, 1, 20, 0 };
    return true;
}

static bool amd_set_device(size_t index) {
    if (index != 0) {
        return false;
    }
    amd_current_device = 0;
    return true;
}

static int amd_get_device() {
    return amd_current_device;
}

static size_t arm_accel_device_count() {
    return 1;
}

static ggml_backend_dev_t arm_accel_device_get(size_t index) {
    return index == 0 ? &arm_accel_devices[0] : nullptr;
}

static bool arm_accel_device_identity(size_t index, flagos_device_identity * identity) {
    if (index != 0 || identity == nullptr) {
        return false;
    }
    *identity = { ARM_ACCEL_ID, ARM_ACCEL_ID, 1, ARM_ACCEL_ID, 0 };
    return true;
}

static bool arm_accel_device_caps(size_t index, flagos_device_caps * caps) {
    if (index != 0 || caps == nullptr) {
        return false;
    }
    *caps = {
        flagos_provider_kind::cpu_accelerator,
        FLAGOS_MEMORY_DEVICE_LOCAL,
        FLAGOS_EXECUTION_ASYNC_QUEUE,
        nullptr,
    };
    return true;
}

static bool arm_accel_set_device(size_t index) {
    if (index != 0) {
        return false;
    }
    arm_accel_current_device = 0;
    return true;
}

static int arm_accel_get_device() {
    return arm_accel_current_device;
}

static int mock_score() {
    return 0;
}

static bool denglin_device_profile(size_t index, flagos_device_profile * profile) {
    if (index >= 2 || profile == nullptr) {
        return false;
    }
    *profile = {};
    profile->struct_size = sizeof(flagos_device_profile);
    profile->version = 1;
    profile->engine = flagos_engine_kind::gpu;
    profile->features = FLAGOS_FEATURE_SIMD;
    profile->vendor = "Denglin";
    profile->architecture = "gpu";
    profile->target = "ks20";
    return true;
}

static bool amd_device_profile(size_t index, flagos_device_profile * profile) {
    if (index != 0 || profile == nullptr) {
        return false;
    }
    *profile = {};
    profile->struct_size = sizeof(flagos_device_profile);
    profile->version = 1;
    profile->engine = flagos_engine_kind::gpu;
    profile->features = FLAGOS_FEATURE_SIMD | FLAGOS_FEATURE_WAVE32 |
        FLAGOS_FEATURE_UNIFIED_MEMORY;
    profile->lane_count = 32;
    profile->vendor = "AMD";
    profile->architecture = "GCN-compatible GPU";
    profile->target = "gfx1150";
    return true;
}

static bool arm_accel_device_profile(size_t index, flagos_device_profile * profile) {
    if (index != 0 || profile == nullptr) {
        return false;
    }
    *profile = {};
    profile->struct_size = sizeof(flagos_device_profile);
    profile->version = 1;
    profile->engine = flagos_engine_kind::cpu;
    profile->features = FLAGOS_FEATURE_NEON | FLAGOS_FEATURE_SVE2 |
        FLAGOS_FEATURE_DOTPROD | FLAGOS_FEATURE_I8MM;
    profile->vector_bits = 128;
    profile->lane_count = 4;
    profile->memory_domain_id = ARM_ACCEL_ID;
    profile->vendor = "mock-arm";
    profile->architecture = "cpu-accelerator";
    profile->target = "arm-v2";
    return true;
}

const flagos_provider_v1 * flagos_denglin_provider() {
    static const flagos_provider_v1 provider = {
        FLAGOS_PROVIDER_API_VERSION,
        sizeof(flagos_provider_v1),
        { DENGLIN_ID, "mock-denglin", 1 },
        mock_probe,
        denglin_device_count,
        denglin_device_get,
        denglin_device_identity,
        mock_device_caps,
        mock_is_backend,
        denglin_set_device,
        denglin_get_device,
        mock_score,
        denglin_device_profile,
    };
    return &provider;
}

const flagos_provider_v1 * flagos_amd_provider() {
    static const flagos_provider_v1 provider = {
        FLAGOS_PROVIDER_API_VERSION,
        sizeof(flagos_provider_v1),
        { AMD_ID, "mock-amd", 1 },
        mock_probe,
        amd_device_count,
        amd_device_get,
        amd_device_identity,
        mock_device_caps,
        mock_is_backend,
        amd_set_device,
        amd_get_device,
        mock_score,
        amd_device_profile,
    };
    return &provider;
}

static const flagos_provider_v1 * mock_arm_accel_provider() {
    static const flagos_provider_v1 provider = {
        FLAGOS_PROVIDER_API_VERSION,
        sizeof(flagos_provider_v1),
        { ARM_ACCEL_ID, "mock-arm-accel", 1 },
        mock_probe,
        arm_accel_device_count,
        arm_accel_device_get,
        arm_accel_device_identity,
        arm_accel_device_caps,
        mock_is_backend,
        arm_accel_set_device,
        arm_accel_get_device,
        mock_score,
        arm_accel_device_profile,
    };
    return &provider;
}

int main() {
    CHECK(flagos_registry_register_provider(mock_arm_accel_provider()));
    CHECK(!flagos_registry_register_provider(mock_arm_accel_provider()));
    ggml_backend_reg_t reg = ggml_backend_flagos_reg();
    CHECK(!flagos_registry_register_provider(mock_arm_accel_provider()));
    CHECK(ggml_backend_reg_get_proc_address(reg, "flagos_registry_get_device_profile") != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, "flagos_registry_get_device_info") != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, "flagos_select_provider_v1") != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, "flagos_status_name") != nullptr);
    CHECK(ggml_backend_reg_dev_count(reg) == 4);
    CHECK(ggml_backend_reg_dev_get(reg, 0) == &denglin_devices[0]);
    CHECK(ggml_backend_reg_dev_get(reg, 1) == &denglin_devices[1]);
    CHECK(ggml_backend_reg_dev_get(reg, 2) == &amd_devices[0]);
    CHECK(ggml_backend_reg_dev_get(reg, 3) == &arm_accel_devices[0]);
    CHECK(ggml_backend_dev_backend_reg(&denglin_devices[0]) == reg);
    CHECK(ggml_backend_dev_backend_reg(&amd_devices[0]) == reg);
    CHECK(ggml_backend_dev_backend_reg(&arm_accel_devices[0]) == reg);

    CHECK(ggml_backend_flagos_get_device() == 0);
    ggml_backend_flagos_set_device(1);
    CHECK(denglin_current_device == 1);
    CHECK(ggml_backend_flagos_get_device() == 1);

    CHECK(denglin_set_device(0));
    CHECK(ggml_backend_flagos_get_device() == -1);

    ggml_backend_flagos_set_device(2);
    CHECK(ggml_backend_flagos_get_device() == 2);
    CHECK(denglin_current_device == 0);
    CHECK(amd_current_device == 0);

    flagos_device_profile profile {};
    CHECK(flagos_registry_get_device_profile(2, &profile));
    CHECK(profile.engine == flagos_engine_kind::gpu);
    CHECK(profile.lane_count == 32);
    CHECK((profile.features & FLAGOS_FEATURE_UNIFIED_MEMORY) != 0);
    CHECK(flagos_registry_get_device_profile(3, &profile));
    CHECK(profile.engine == flagos_engine_kind::cpu);
    CHECK((profile.features & FLAGOS_FEATURE_SVE2) != 0);
    CHECK(profile.architecture != nullptr);
    CHECK(std::strcmp(profile.architecture, "cpu-accelerator") == 0);
    flagos_device_info info {};
    CHECK(flagos_registry_get_device_info(3, &info));
    CHECK(info.struct_size == sizeof(flagos_device_info));
    CHECK(info.global_index == 3);
    CHECK(info.local_index == 0);
    CHECK(info.name == nullptr);
    CHECK(info.provider.name != nullptr);
    CHECK(std::strcmp(info.provider.name, "mock-arm-accel") == 0);
    CHECK(info.identity.memory_domain_id == ARM_ACCEL_ID);
    CHECK(info.caps.kind == flagos_provider_kind::cpu_accelerator);
    CHECK(!flagos_registry_get_device_profile(4, &profile));
    CHECK(!flagos_registry_get_device_info(4, &info));

    flagos_provider_selection_v1 selection {};
    selection.struct_size = sizeof(selection);
    selection.version = 1;
    flagos_provider_selector_v1 selector {};
    selector.struct_size = sizeof(selector);
    selector.version = 1;
    selector.provider_id = AMD_ID;
    selector.local_device_index = 0;
    CHECK(flagos_select_provider_v1(&selector, &selection) == FLAGOS_STATUS_SUCCESS);
    CHECK(selection.global_device_index == 2);
    CHECK(selection.local_device_index == 0);
    CHECK(selection.provider_id == AMD_ID);
    CHECK(std::strcmp(selection.provider_name, "mock-amd") == 0);
    CHECK(selection.profile.target != nullptr);
    CHECK(std::strcmp(selection.profile.target, "gfx1150") == 0);

    selector = {};
    selector.struct_size = sizeof(selector);
    selector.version = 1;
    selector.device_uuid_hi = AMD_ID;
    selector.device_uuid_lo = 1;
    CHECK(flagos_select_provider_v1(&selector, &selection) == FLAGOS_STATUS_SUCCESS);
    CHECK(selection.global_device_index == 2);
    CHECK(selection.device.uuid_hi == AMD_ID);
    CHECK(selection.device.uuid_lo == 1);

    selector = {};
    selector.struct_size = sizeof(selector);
    selector.version = 1;
    selector.provider_name = "mock-arm-accel";
    selector.policy = FLAGOS_SELECT_CONFIGURED;
    CHECK(flagos_select_provider_v1(&selector, &selection) == FLAGOS_STATUS_SUCCESS);
    CHECK(selection.global_device_index == 3);
    CHECK(selection.provider_id == ARM_ACCEL_ID);

    selector = {};
    selector.struct_size = sizeof(selector);
    selector.version = 1;
    selector.provider_id = DENGLIN_ID;
    CHECK(flagos_select_provider_v1(&selector, &selection) == FLAGOS_STATUS_AMBIGUOUS);

    selector.local_device_index = 0;
    CHECK(flagos_select_provider_v1(&selector, &selection) == FLAGOS_STATUS_SUCCESS);
    CHECK(selection.global_device_index == 0);

    selector = {};
    selector.struct_size = sizeof(selector);
    selector.version = 1;
    selector.policy = FLAGOS_SELECT_AUTO_SINGLE;
    selector.provider_id = ARM_ACCEL_ID;
    CHECK(flagos_select_provider_v1(&selector, &selection) == FLAGOS_STATUS_SUCCESS);
    CHECK(selection.global_device_index == 3);

    selector.provider_id = 999;
    CHECK(flagos_select_provider_v1(&selector, &selection) == FLAGOS_STATUS_NOT_FOUND);

    selector = {};
    selector.struct_size = sizeof(selector);
    selector.version = 1;
    selector.policy = FLAGOS_SELECT_AUTO_SINGLE;
    selector.provider_name = "";
    CHECK(flagos_select_provider_v1(&selector, &selection) == FLAGOS_STATUS_INVALID_ARGUMENT);

    selector = {};
    selector.struct_size = sizeof(selector);
    selector.version = 1;
    selector.policy = FLAGOS_SELECT_AUTO_SINGLE;
    selector.device_uuid_lo = 1;
    CHECK(flagos_select_provider_v1(&selector, &selection) == FLAGOS_STATUS_NOT_FOUND);

    selector = {};
    selector.struct_size = sizeof(selector);
    selector.version = 1;
    selector.policy = FLAGOS_SELECT_EXPLICIT;
    CHECK(flagos_select_provider_v1(&selector, &selection) == FLAGOS_STATUS_INVALID_ARGUMENT);

    selector.policy = 99;
    selector.provider_id = AMD_ID;
    CHECK(flagos_select_provider_v1(&selector, &selection) == FLAGOS_STATUS_UNSUPPORTED_POLICY);

    CHECK(std::strcmp(flagos_status_name(FLAGOS_STATUS_SUCCESS), "success") == 0);
    CHECK(std::strcmp(flagos_status_name(FLAGOS_STATUS_AMBIGUOUS), "ambiguous") == 0);

    CHECK(flagos_select_provider_v1(nullptr, &selection) == FLAGOS_STATUS_INVALID_ARGUMENT);
    CHECK(flagos_select_provider_v1(&selector, nullptr) == FLAGOS_STATUS_INVALID_ARGUMENT);
    selector = {};
    selector.struct_size = sizeof(selector) - 1;
    selector.version = 1;
    selector.provider_id = AMD_ID;
    CHECK(flagos_select_provider_v1(&selector, &selection) == FLAGOS_STATUS_INVALID_ARGUMENT);
    selector.struct_size = sizeof(selector);
    selector.version = 2;
    CHECK(flagos_select_provider_v1(&selector, &selection) == FLAGOS_STATUS_INVALID_ARGUMENT);
    selector.version = 1;
    selector.flags = 1;
    CHECK(flagos_select_provider_v1(&selector, &selection) == FLAGOS_STATUS_INVALID_ARGUMENT);
    selector.flags = 0;
    selection.struct_size = sizeof(selection) - 1;
    CHECK(flagos_select_provider_v1(&selector, &selection) == FLAGOS_STATUS_INVALID_ARGUMENT);

    std::puts("FlagOS multi-provider registry checks passed");
    return 0;
}
