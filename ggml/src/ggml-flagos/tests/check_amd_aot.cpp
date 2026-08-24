#include "../providers/amd/flagos-amd-aot.h"

#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static void check(bool condition, const char * expression, int line) {
    if (!condition) {
        std::fprintf(stderr, "check failed at line %d: %s\n", line, expression);
        std::abort();
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

int main() {
    const char * package_dir = std::getenv("FLAGOS_AMD_AOT_TEST_DIR");
    if (package_dir == nullptr || package_dir[0] == '\0') {
        std::puts("FlagOS AMD AOT checks skipped: FLAGOS_AMD_AOT_TEST_DIR is unset");
        return 0;
    }

    int device = -1;
    CHECK(hipGetDevice(&device) == hipSuccess);
    hipDeviceProp_t props {};
    CHECK(hipGetDeviceProperties(&props, device) == hipSuccess);

    flagos_amd::kernel_registry registry;
    CHECK(registry.initialize(std::filesystem::path(package_dir), device, props.gcnArchName));
    CHECK(registry.size() >= 1);

    const char * smoke_kernel = registry.find("double") != nullptr ? "double" : "flagos_add_f32";
    const auto * metadata = registry.find(smoke_kernel);
    CHECK(metadata != nullptr);
    CHECK(metadata->symbol == smoke_kernel);
    CHECK(metadata->threads > 0);
    CHECK(metadata->block_size > 0);
    if (const auto * f16 = registry.find("flagos_mul_mat_f16_f32_batched")) {
        CHECK(f16->tile_m > 0);
        CHECK(f16->tile_n > 0);
        CHECK(f16->tile_k > 0);
    }
    if (const auto * grouped = registry.find("flagos_mul_mat_f16_f32_grouped")) {
        CHECK(grouped->tile_m > 0);
        CHECK(grouped->tile_n > 0);
        CHECK(grouped->tile_k > 0);
    }
    if (const auto * grouped_ffn = registry.find("flagos_ffn_swiglu_f16_f32_grouped")) {
        CHECK(grouped_ffn->tile_m > 0);
        CHECK(grouped_ffn->tile_n > 0);
        CHECK(grouped_ffn->tile_k > 0);
    }

    hipStream_t stream = nullptr;
    CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) == hipSuccess);
    void * x = nullptr;
    void * y = nullptr;
    CHECK(hipMalloc(&x, 256 * sizeof(float)) == hipSuccess);
    CHECK(hipMalloc(&y, 256 * sizeof(float)) == hipSuccess);
    std::vector<float> host_x(256);
    std::vector<float> host_y(256, -1.0f);
    for (size_t i = 0; i < host_x.size(); ++i) {
        host_x[i] = static_cast<float>(i) * 0.5f;
    }
    CHECK(hipMemcpyAsync(x, host_x.data(), host_x.size() * sizeof(float),
        hipMemcpyHostToDevice, stream) == hipSuccess);
    int n = static_cast<int>(host_x.size());
    std::vector<void *> args;
    if (std::strcmp(smoke_kernel, "double") == 0) {
        args = { &x, &y, &n };
    } else {
        args = { &x, &x, &y, &n };
    }
    CHECK(registry.launch(smoke_kernel, stream, 2, 1, 1, args));
    CHECK(hipMemcpyAsync(host_y.data(), y, host_y.size() * sizeof(float),
        hipMemcpyDeviceToHost, stream) == hipSuccess);
    CHECK(hipStreamSynchronize(stream) == hipSuccess);
    for (size_t i = 0; i < host_y.size(); ++i) {
        const float expected = std::strcmp(smoke_kernel, "double") == 0
            ? host_x[i] * 2.0f : host_x[i] + host_x[i];
        CHECK(host_y[i] == expected);
    }
    CHECK(hipFree(x) == hipSuccess);
    CHECK(hipFree(y) == hipSuccess);
    CHECK(hipStreamDestroy(stream) == hipSuccess);
    std::printf("FlagOS AMD AOT checks passed: %s (%zu kernels)\n", props.gcnArchName, registry.size());
    return 0;
}
