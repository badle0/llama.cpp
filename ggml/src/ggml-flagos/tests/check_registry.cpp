#include "flagos-provider.h"
#include "ggml-flagos.h"

#include "../../ggml-backend-impl.h"

#include <cstdlib>
#include <cstdio>

static void check(bool condition, const char * expression, int line) {
    if (!condition) {
        std::fprintf(stderr, "check failed at line %d: %s\n", line, expression);
        std::abort();
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

static constexpr uint64_t DENGLIN_ID = 1;
static constexpr uint64_t AMD_ID = 2;

static ggml_backend_device denglin_devices[2] {};
static ggml_backend_device amd_devices[1] {};
static int denglin_current_device = 0;
static int amd_current_device = 0;

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

static int mock_score() {
    return 0;
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
    };
    return &provider;
}

int main() {
    ggml_backend_reg_t reg = ggml_backend_flagos_reg();
    CHECK(ggml_backend_reg_dev_count(reg) == 3);
    CHECK(ggml_backend_reg_dev_get(reg, 0) == &denglin_devices[0]);
    CHECK(ggml_backend_reg_dev_get(reg, 1) == &denglin_devices[1]);
    CHECK(ggml_backend_reg_dev_get(reg, 2) == &amd_devices[0]);
    CHECK(ggml_backend_dev_backend_reg(&denglin_devices[0]) == reg);
    CHECK(ggml_backend_dev_backend_reg(&amd_devices[0]) == reg);

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

    std::puts("FlagOS multi-provider registry checks passed");
    return 0;
}
