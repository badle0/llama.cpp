// flagos-check-spacemit: checks of the SpacemiT provider through the built ggml-flagos library.
// Usage: flagos-check-spacemit [--expect-device | --expect-none] [--bench]
//   --expect-device  fail if the provider exposes no device (use on the K3)
//   --expect-none    fail if it exposes one (use with FLAGOS_SPACEMIT_DISABLE=1, or on another host)
//   --bench          also measure launch cost, per-node cost and bandwidth for both stream policies
// Without a flag, a host without the device skips the device checks.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-flagos.h"
#include "ggml.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
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

static void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

static std::vector<float> ramp(int64_t n, float scale, float offset) {
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; i++) {
        v[i] = scale * (float) (i % 1000) + offset;
    }
    return v;
}

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

static int check_buffer(ggml_backend_dev_t dev) {
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

    const int64_t    n      = 1000;
    ggml_init_params params = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_tensor *         a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor *         b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    REQUIRE(buf != nullptr);
    CHECK(reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buf)) % 64 == 0);

    std::vector<float> in = ramp(n, 0.5f, -7.0f), out(n, -1.0f);
    ggml_backend_tensor_set(a, in.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_get(a, out.data(), 0, ggml_nbytes(a));
    CHECK(std::memcmp(in.data(), out.data(), ggml_nbytes(a)) == 0);
    ggml_backend_tensor_get(a, out.data() + 10, 40 * sizeof(float), 5 * sizeof(float));
    CHECK(std::memcmp(out.data() + 10, in.data() + 40, 5 * sizeof(float)) == 0);
    ggml_backend_tensor_memset(a, 0, 0, ggml_nbytes(a));
    ggml_backend_tensor_get(a, out.data(), 0, ggml_nbytes(a));
    CHECK(out[0] == 0.0f && out[n - 1] == 0.0f);
    ggml_backend_buffer_clear(buf, 0xff);
    uint32_t bits = 0;
    ggml_backend_tensor_get(b, &bits, 0, sizeof(bits));
    CHECK(bits == 0xffffffffu);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    std::printf("buffer   alloc/free (incl. size 0), set/get, memset, clear: checked\n");
    return 0;
}

// checks that ADD is claimed exactly, computes correctly, and that failures are reported without hanging
static int check_ops(ggml_backend_dev_t dev) {
    ggml_backend_buffer_type_t buft   = ggml_backend_dev_buffer_type(dev);
    const int64_t              n      = 1000;
    const int64_t              n_big  = 4000037;  // not a multiple of 8 tiles x 16 floats
    ggml_init_params           params = { ggml_tensor_overhead() * 32 + ggml_graph_overhead() * 8, nullptr, true };
    ggml_context *             ctx    = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * sum   = ggml_add(ctx, a, b);
    ggml_tensor * sum2  = ggml_add(ctx, sum, b);
    ggml_tensor * sum3  = ggml_add(ctx, sum2, b);
    ggml_tensor * view  = ggml_reshape_2d(ctx, a, 100, 10);
    ggml_tensor * prod  = ggml_mul(ctx, a, b);
    ggml_tensor * row   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100);
    ggml_tensor * bcast = ggml_add(ctx, view, row);
    ggml_tensor * tview = ggml_transpose(ctx, view);
    ggml_tensor * strided = ggml_add(ctx, tview, tview);
    ggml_tensor * h1    = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n);
    ggml_tensor * half  = ggml_add(ctx, h1, h1);
    ggml_tensor * big_a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_big);
    ggml_tensor * big_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_big);
    ggml_tensor * big   = ggml_add(ctx, big_a, big_b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    REQUIRE(buf != nullptr);

    // claimed exactly: inputs and views (no kernel), and same-shape contiguous F32 ADD
    CHECK(ggml_backend_dev_supports_op(dev, a));
    CHECK(ggml_backend_dev_supports_op(dev, view));
    CHECK(ggml_backend_dev_supports_op(dev, tview));
    CHECK(ggml_backend_dev_supports_op(dev, sum));
    CHECK(!ggml_backend_dev_supports_op(dev, prod));
    CHECK(!ggml_backend_dev_supports_op(dev, bcast));
    CHECK(!ggml_backend_dev_supports_op(dev, strided));
    CHECK(!ggml_backend_dev_supports_op(dev, half));

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    REQUIRE(backend != nullptr);
    CHECK(std::strcmp(ggml_backend_name(backend), k_device_name) == 0);
    CHECK(ggml_backend_is_flagos(backend));

    std::vector<float> va = ramp(n, 0.25f, -3.0f), vb = ramp(n, -0.125f, 1.5f), out(n);
    ggml_backend_tensor_set(a, va.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, vb.data(), 0, ggml_nbytes(b));

    ggml_cgraph * empty_graph = ggml_new_graph(ctx);
    CHECK(ggml_backend_graph_compute(backend, empty_graph) == GGML_STATUS_SUCCESS);
    ggml_cgraph * view_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(view_graph, view);
    CHECK(ggml_backend_graph_compute(backend, view_graph) == GGML_STATUS_SUCCESS);

    // three dependent ADDs in one launch: (a + b) + b + b
    ggml_cgraph * chain = ggml_new_graph(ctx);
    ggml_build_forward_expand(chain, sum3);
    CHECK(ggml_backend_graph_compute(backend, chain) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(sum3, out.data(), 0, ggml_nbytes(sum3));
    int64_t wrong = 0;
    for (int64_t i = 0; i < n; i++) {
        wrong += out[i] != ((va[i] + vb[i]) + vb[i]) + vb[i];
    }
    CHECK(wrong == 0);

    // large ADD split across all tiles, with a tail
    std::vector<float> ba = ramp(n_big, 0.5f, 0.0f), bb = ramp(n_big, -2.0f, 3.0f), bout(n_big);
    ggml_backend_tensor_set(big_a, ba.data(), 0, ggml_nbytes(big_a));
    ggml_backend_tensor_set(big_b, bb.data(), 0, ggml_nbytes(big_b));
    ggml_cgraph * big_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(big_graph, big);
    CHECK(ggml_backend_graph_compute(backend, big_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(big, bout.data(), 0, ggml_nbytes(big));
    int64_t wrong_big = 0;
    for (int64_t i = 0; i < n_big; i++) {
        wrong_big += bout[i] != ba[i] + bb[i];
    }
    CHECK(wrong_big == 0);

    // an unclaimed op is rejected before anything runs
    std::fprintf(stderr, "(the next error line is expected)\n");
    ggml_cgraph * mul_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(mul_graph, prod);
    CHECK(ggml_backend_graph_compute(backend, mul_graph) == GGML_STATUS_FAILED);
    ggml_backend_free(backend);

    // a kernel failure inside the launch stops every tile and is reported (test-only switch)
    set_env("FLAGOS_SPACEMIT_TEST_FAIL_NODE", "1");
    ggml_backend_t failing = ggml_backend_dev_init(dev, nullptr);
    set_env("FLAGOS_SPACEMIT_TEST_FAIL_NODE", nullptr);
    REQUIRE(failing != nullptr);
    CHECK(ggml_backend_graph_compute(failing, chain) == GGML_STATUS_FAILED);
    ggml_backend_free(failing);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    std::printf("ops      ADD claimed exactly (not MUL, broadcast, strided or F16); 3-node chain and %lld-element ADD exact: checked\n",
                (long long) n_big);
    std::printf("backend  empty and view-only graphs succeed; unclaimed op and forced kernel failure return FAILED: checked\n");
    return 0;
}

static double time_us(const std::function<void()> & fn, int iterations) {
    for (int i = 0; i < iterations / 10 + 1; i++) {
        fn();
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
        fn();
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;
}

// busy share of CPUs 8-15 (the AI cores) over one second, from /proc/stat; -1 where unavailable
static double ai_core_load() {
    auto sample = [](uint64_t & busy, uint64_t & total) {
        std::ifstream stat("/proc/stat");
        std::string   line;
        busy = total = 0;
        while (std::getline(stat, line)) {
            int cpu = -1;
            unsigned long long v[8] = {};
            if (std::sscanf(line.c_str(), "cpu%d %llu %llu %llu %llu %llu %llu %llu %llu", &cpu, &v[0], &v[1], &v[2],
                            &v[3], &v[4], &v[5], &v[6], &v[7]) == 9 && cpu >= 8 && cpu <= 15) {
                for (unsigned long long x : v) {
                    total += x;
                }
                busy += v[0] + v[1] + v[2] + v[5] + v[6] + v[7];
            }
        }
    };
    uint64_t b0, t0, b1, t1;
    sample(b0, t0);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    sample(b1, t1);
    return t1 > t0 ? 100.0 * (double) (b1 - b0) / (double) (t1 - t0) : -1.0;
}

static int bench(ggml_backend_dev_t dev) {
    const int64_t    n_small = 64;
    const int64_t    n_big   = 16 * 1024 * 1024;
    const int        chain   = 64;
    ggml_init_params params  = { ggml_tensor_overhead() * (chain + 16) + ggml_graph_overhead() * 4, nullptr, true };
    ggml_context *   ctx     = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_small);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_small);
    ggml_tensor * one = ggml_add(ctx, a, b);
    ggml_tensor * x   = a;
    for (int i = 0; i < chain; i++) {
        x = ggml_add(ctx, x, b);
    }
    ggml_tensor * big_a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_big);
    ggml_tensor * big_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_big);
    ggml_tensor * big   = ggml_add(ctx, big_a, big_b);
    ggml_cgraph * g_one   = ggml_new_graph(ctx);
    ggml_cgraph * g_chain = ggml_new_graph(ctx);
    ggml_cgraph * g_big   = ggml_new_graph(ctx);
    ggml_build_forward_expand(g_one, one);
    ggml_build_forward_expand(g_chain, x);
    ggml_build_forward_expand(g_big, big);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev));
    REQUIRE(buf != nullptr);
    ggml_backend_buffer_clear(buf, 0);
    const double big_bytes = 3.0 * n_big * sizeof(float);

    for (const char * policy : { "persistent", "per-call" }) {
        set_env("FLAGOS_SPACEMIT_STREAM", policy);
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        set_env("FLAGOS_SPACEMIT_STREAM", nullptr);
        REQUIRE(backend != nullptr);
        bool ok = true;
        const double t_one   = time_us([&] { ok &= ggml_backend_graph_compute(backend, g_one) == GGML_STATUS_SUCCESS; }, 500);
        const double t_chain = time_us([&] { ok &= ggml_backend_graph_compute(backend, g_chain) == GGML_STATUS_SUCCESS; }, 100);
        const double t_big   = time_us([&] { ok &= ggml_backend_graph_compute(backend, g_big) == GGML_STATUS_SUCCESS; }, 10);
        const double idle    = ai_core_load();
        CHECK(ok);
        std::printf("bench    %-10s launch %.1f us, per node %.2f us, 16M-float ADD %.2f GB/s, idle AI-core load %.0f%%\n",
                    policy, t_one, (t_chain - t_one) / (chain - 1), big_bytes / t_big / 1000.0, idle);
        ggml_backend_free(backend);
    }

    // CPU reference on the X100 cores for the same large ADD, in a CPU buffer
    ggml_init_params cpu_params = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true };
    ggml_context *   cpu_ctx    = ggml_init(cpu_params);
    ggml_tensor *    ca         = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, n_big);
    ggml_tensor *    cb         = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, n_big);
    ggml_cgraph *    cg         = ggml_new_graph(cpu_ctx);
    ggml_build_forward_expand(cg, ggml_add(cpu_ctx, ca, cb));
    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors_from_buft(cpu_ctx, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_clear(cbuf, 0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 8);
    const double t_cpu = time_us([&] { ggml_backend_graph_compute(cpu, cg); }, 10);
    std::printf("bench    cpu        16M-float ADD %.2f GB/s (8 X100 threads)\n", big_bytes / t_cpu / 1000.0);
    ggml_backend_free(cpu);
    ggml_backend_buffer_free(cbuf);
    ggml_free(cpu_ctx);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    bool expect_device = false;
    bool expect_none   = false;
    bool run_bench     = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--expect-device") == 0) {
            expect_device = true;
        } else if (std::strcmp(argv[i], "--expect-none") == 0) {
            expect_none = true;
        } else if (std::strcmp(argv[i], "--bench") == 0) {
            run_bench = true;
        } else {
            std::fprintf(stderr, "usage: %s [--expect-device | --expect-none] [--bench]\n", argv[0]);
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

    if (check_registry(dev, global_index) != 0 || check_buffer(dev) != 0 || check_ops(dev) != 0) {
        return 1;
    }
    if (run_bench && bench(dev) != 0) {
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "flagos-check-spacemit: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("flagos-check-spacemit: all checks passed\n");
    return 0;
}
