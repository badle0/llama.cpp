// flagos-check-spacemit: M1 checks of the SpacemiT provider through the built ggml-flagos library.
// Usage: flagos-check-spacemit [--expect-device | --expect-none]
//   --expect-device  fail if the provider exposes no device (use on the K3)
//   --expect-none    fail if it exposes one (use with FLAGOS_SPACEMIT_DISABLE=1, or on another host)
// Without a flag, a host without the device skips the device checks.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-flagos.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL (line %d): %s\n", __LINE__, #cond);   \
            g_failures++;                                                    \
        }                                                                    \
    } while (0)

#define REQUIRE(cond)                                                                 \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "FAIL (line %d, stopping): %s\n", __LINE__, #cond);  \
            return 1;                                                                 \
        }                                                                             \
    } while (0)

static const char * k_device_name = "FlagOS:SpacemiT:0";

static int check_registry(ggml_backend_dev_t dev, size_t global_index) {
    CHECK(ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL);
    CHECK(ggml_backend_dev_description(dev) != nullptr);

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    CHECK(props.memory_total > 0);
    CHECK(props.memory_free > 0 && props.memory_free <= props.memory_total);
    CHECK(!props.caps.async && !props.caps.events && !props.caps.host_buffer);

    flagos_device_info info;
    REQUIRE(flagos_registry_get_device_info(global_index, &info));
    CHECK(std::strcmp(info.provider.name, "SpacemiT") == 0);
    CHECK(info.identity.provider_id == info.provider.id);
    CHECK(info.caps.kind == flagos_provider_kind::cpu_accelerator);
    CHECK(info.caps.execution == 0);
    CHECK(info.profile.vector_bits == 1024);
    CHECK(info.profile.concurrency > 0);

    std::printf("device   %s: %s, type ACCEL, memory %.1f / %.1f GiB free\n", props.name, props.description,
                props.memory_free / 1073741824.0, props.memory_total / 1073741824.0);
    std::printf("flagos   provider %s (id 0x%016llx), engine cpu, %u AI cores, VLEN %u, target %s, runtime %s\n",
                info.provider.name, (unsigned long long) info.provider.id, info.profile.concurrency,
                info.profile.vector_bits, info.profile.target, info.profile.runtime);
    return 0;
}

static int check_buffer_and_backend(ggml_backend_dev_t dev) {
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    REQUIRE(buft != nullptr);
    CHECK(std::strcmp(ggml_backend_buft_name(buft), "FlagOS:SpacemiT") == 0);
    CHECK(ggml_backend_buft_get_alignment(buft) == 64);
    CHECK(!ggml_backend_buft_is_host(buft));
    CHECK(ggml_backend_dev_supports_buft(dev, buft));
    CHECK(ggml_backend_dev_supports_buft(dev, ggml_backend_cpu_buffer_type()));

    // llama.cpp probes every weight buffer type with a zero-size buffer
    ggml_backend_buffer_t empty = ggml_backend_buft_alloc_buffer(buft, 0);
    CHECK(empty != nullptr && ggml_backend_buffer_get_size(empty) == 0);
    ggml_backend_buffer_free(empty);

    const int64_t           n      = 1000;
    ggml_init_params        params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead() * 4, nullptr, true };
    ggml_context *          ctx    = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * sum  = ggml_add(ctx, a, b);
    ggml_tensor * view = ggml_reshape_2d(ctx, a, 100, 10);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    REQUIRE(buf != nullptr);
    CHECK(reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buf)) % 64 == 0);
    CHECK(view->data == a->data);

    std::vector<float> in(n), out(n, -1.0f);
    for (int64_t i = 0; i < n; i++) {
        in[i] = 0.5f * (float) i - 7.0f;
    }
    ggml_backend_tensor_set(a, in.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_get(a, out.data(), 0, ggml_nbytes(a));
    CHECK(std::memcmp(in.data(), out.data(), ggml_nbytes(a)) == 0);

    ggml_backend_tensor_get(a, out.data() + 10, 40 * sizeof(float), 5 * sizeof(float));
    CHECK(std::memcmp(out.data() + 10, in.data() + 40, 5 * sizeof(float)) == 0);

    ggml_backend_tensor_memset(a, 0, 0, ggml_nbytes(a));
    ggml_backend_tensor_get(a, out.data(), 0, ggml_nbytes(a));
    CHECK(out[0] == 0.0f && out[n - 1] == 0.0f);

    ggml_backend_buffer_clear(buf, 0xff);
    ggml_backend_tensor_get(b, out.data(), 0, sizeof(float));
    uint32_t bits = 0;
    std::memcpy(&bits, out.data(), sizeof(bits));
    CHECK(bits == 0xffffffffu);

    // M1 claims no ops
    CHECK(!ggml_backend_dev_supports_op(dev, sum));

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    REQUIRE(backend != nullptr);
    CHECK(std::strcmp(ggml_backend_name(backend), k_device_name) == 0);
    CHECK(ggml_backend_is_flagos(backend));

    ggml_cgraph * empty_graph = ggml_new_graph(ctx);
    CHECK(ggml_backend_graph_compute(backend, empty_graph) == GGML_STATUS_SUCCESS);

    ggml_cgraph * view_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(view_graph, view);
    CHECK(ggml_backend_graph_compute(backend, view_graph) == GGML_STATUS_SUCCESS);

    // a compute node must fail cleanly, not crash and not report success
    std::fprintf(stderr, "(the next error line is expected)\n");
    ggml_cgraph * add_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(add_graph, sum);
    CHECK(ggml_backend_graph_compute(backend, add_graph) == GGML_STATUS_FAILED);

    ggml_backend_free(backend);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    std::printf("buffer   alloc/free (incl. size 0), set/get, memset, clear: checked\n");
    std::printf("backend  empty and view-only graphs succeed, ADD fails as unclaimed: checked\n");
    return 0;
}

int main(int argc, char ** argv) {
    bool expect_device = false;
    bool expect_none   = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--expect-device") == 0) {
            expect_device = true;
        } else if (std::strcmp(argv[i], "--expect-none") == 0) {
            expect_none = true;
        } else {
            std::fprintf(stderr, "usage: %s [--expect-device | --expect-none]\n", argv[0]);
            return 2;
        }
    }

    ggml_backend_reg_t reg = ggml_backend_flagos_reg();
    REQUIRE(reg != nullptr);

    ggml_backend_dev_t dev          = nullptr;
    size_t             global_index = 0;
    size_t             n_spacemit   = 0;
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
        ggml_backend_dev_t d = ggml_backend_reg_dev_get(reg, i);
        if (std::strncmp(ggml_backend_dev_name(d), "FlagOS:SpacemiT:", 16) == 0) {
            n_spacemit++;
            if (std::strcmp(ggml_backend_dev_name(d), k_device_name) == 0) {
                dev          = d;
                global_index = i;
            }
        }
    }
    std::printf("registry %zu FlagOS device(s), %zu SpacemiT\n", ggml_backend_reg_dev_count(reg), n_spacemit);

    if (dev == nullptr) {
        if (expect_device) {
            std::fprintf(stderr, "FAIL: no %s device\n", k_device_name);
            return 1;
        }
        std::printf("flagos-check-spacemit: no SpacemiT device on this host; device checks skipped\n");
        return 0;
    }
    REQUIRE(!expect_none);
    CHECK(n_spacemit == 1);

    if (check_registry(dev, global_index) != 0 || check_buffer_and_backend(dev) != 0) {
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "flagos-check-spacemit: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("flagos-check-spacemit: all checks passed\n");
    return 0;
}
