#include "flagos-provider.h"

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

static bool probe(ggml_backend_reg_t) {
    return true;
}

static size_t device_count() {
    return 0;
}

static ggml_backend_dev_t device_get(size_t) {
    return nullptr;
}

static bool device_identity(size_t, flagos_device_identity *) {
    return false;
}

static bool device_caps(size_t, flagos_device_caps *) {
    return false;
}

static bool is_backend(ggml_backend_t) {
    return false;
}

static bool set_device(size_t) {
    return true;
}

static int get_device() {
    return 0;
}

static int score() {
    return 0;
}

static ggml_backend_device legacy_device {};

static size_t legacy_device_count() {
    return 1;
}

static ggml_backend_dev_t legacy_device_get(size_t index) {
    return index == 0 ? &legacy_device : nullptr;
}

static bool legacy_device_identity(size_t index, flagos_device_identity * identity) {
    if (index != 0 || identity == nullptr) {
        return false;
    }
    *identity = { 7, 7, 1, 7, 0 };
    return true;
}

static bool legacy_device_caps(size_t index, flagos_device_caps * caps) {
    if (index != 0 || caps == nullptr) {
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

static bool device_profile(size_t, flagos_device_profile * profile) {
    if (profile == nullptr) {
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
    profile->vendor = "mock";
    profile->architecture = "arm64";
    profile->target = "mock-arm-v2";
    return true;
}

int main() {
    flagos_provider_v1 provider = {
        FLAGOS_PROVIDER_API_VERSION,
        sizeof(flagos_provider_v1),
        { 1, "test", 1 },
        probe,
        device_count,
        device_get,
        device_identity,
        device_caps,
        is_backend,
        set_device,
        get_device,
        score,
        device_profile,
    };

    CHECK(flagos_provider_is_valid(&provider));

    provider.api_version += 1;
    CHECK(!flagos_provider_is_valid(&provider));
    provider.api_version = FLAGOS_PROVIDER_API_VERSION;

    provider.struct_size = FLAGOS_PROVIDER_V1_REQUIRED_SIZE - 1;
    CHECK(!flagos_provider_is_valid(&provider));
    provider.struct_size = FLAGOS_PROVIDER_V1_REQUIRED_SIZE;
    CHECK(flagos_provider_is_valid(&provider));
    provider.struct_size = sizeof(flagos_provider_v1);

    flagos_device_profile profile {};
    CHECK(flagos_provider_get_device_profile(&provider, 0, &profile));
    CHECK(profile.engine == flagos_engine_kind::cpu);
    CHECK((profile.features & FLAGOS_FEATURE_SVE2) != 0);

    const flagos_kernel_variant sve_variant {
        "sve2-dotprod",
        { FLAGOS_FEATURE_SVE2 | FLAGOS_FEATURE_DOTPROD, FLAGOS_FEATURE_SME, 128, 4, "mock-arm-v2" },
        1, 0, 1, 0, 1024, 0, 50,
    };
    const flagos_kernel_shape shape { 32, 16, 4096 };
    CHECK(flagos_kernel_variant_matches(&profile, &sve_variant, &shape));
    CHECK(flagos_kernel_variant_score(&profile, &sve_variant, &shape) == 50);
    const flagos_variant_match match = flagos_check_kernel_variant(&profile, &sve_variant, &shape);
    CHECK(match.state == flagos_support_state::available);
    CHECK(match.reason == flagos_support_reason::none);

    const flagos_kernel_variant wrong_target {
        "other-target",
        { FLAGOS_FEATURE_SVE2, 0, 0, 0, "other" },
    };
    CHECK(!flagos_kernel_variant_matches(&profile, &wrong_target, &shape));
    CHECK(flagos_check_kernel_variant(&profile, &wrong_target, &shape).reason ==
        flagos_support_reason::target);

    provider.identity.id = 0;
    CHECK(!flagos_provider_is_valid(&provider));
    provider.identity.id = 1;

    provider.probe = nullptr;
    CHECK(!flagos_provider_is_valid(&provider));
    provider.probe = probe;

    flagos_device_identity identity = { 1, 2, 3, 4, 0 };
    CHECK(flagos_device_identity_is_valid(&provider, 0, &identity));
    identity.provider_id = 2;
    CHECK(!flagos_device_identity_is_valid(&provider, 0, &identity));
    identity.provider_id = 1;
    identity.uuid_hi = 0;
    identity.uuid_lo = 0;
    CHECK(!flagos_device_identity_is_valid(&provider, 0, &identity));

    flagos_device_caps caps = {
        flagos_provider_kind::gpu,
        FLAGOS_MEMORY_DEVICE_LOCAL,
        FLAGOS_EXECUTION_AOT_MODULE,
        "test-aot",
    };
    CHECK(flagos_device_caps_are_valid(&caps));
    caps.aot_format = nullptr;
    CHECK(!flagos_device_caps_are_valid(&caps));
    caps.execution = FLAGOS_EXECUTION_ASYNC_QUEUE;
    CHECK(flagos_device_caps_are_valid(&caps));
    caps.memory = 0;
    CHECK(!flagos_device_caps_are_valid(&caps));

    const flagos_provider_v1 legacy_provider = {
        FLAGOS_PROVIDER_API_VERSION,
        FLAGOS_PROVIDER_V1_REQUIRED_SIZE,
        { 7, "legacy", 1 },
        probe,
        legacy_device_count,
        legacy_device_get,
        legacy_device_identity,
        legacy_device_caps,
        is_backend,
        set_device,
        get_device,
        score,
        nullptr,
    };
    CHECK(flagos_provider_is_valid(&legacy_provider));
    flagos_device_profile legacy_profile {};
    CHECK(flagos_provider_get_device_profile(&legacy_provider, 0, &legacy_profile));
    CHECK(legacy_profile.engine == flagos_engine_kind::gpu);
    CHECK(legacy_profile.architecture != nullptr);
    CHECK(std::strcmp(legacy_profile.architecture, "gpu") == 0);

    std::puts("FlagOS provider ABI checks passed");
    return 0;
}
