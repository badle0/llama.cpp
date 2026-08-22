#include "flagos-provider.h"

#include <cstdlib>
#include <cstdio>

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
    };

    CHECK(flagos_provider_is_valid(&provider));

    provider.api_version += 1;
    CHECK(!flagos_provider_is_valid(&provider));
    provider.api_version = FLAGOS_PROVIDER_API_VERSION;

    provider.struct_size = sizeof(flagos_provider_v1) - 1;
    CHECK(!flagos_provider_is_valid(&provider));
    provider.struct_size = sizeof(flagos_provider_v1);

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

    std::puts("FlagOS provider ABI checks passed");
    return 0;
}
