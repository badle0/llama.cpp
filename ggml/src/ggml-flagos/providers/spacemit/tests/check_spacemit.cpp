// flagos-check-spacemit: checks of the SpacemiT provider through the built ggml-flagos library.
// Usage: flagos-check-spacemit [--expect-device | --expect-none] [--full] [--bench]
//   --expect-device  fail if the provider exposes no device (use on the K3)
//   --expect-none    fail if it exposes one (use with FLAGOS_SPACEMIT_DISABLE=1, or on another host)
//   --full           also check Q4_0 and Q4_1 matmuls at Qwen3-4B's shapes (up to 512 x 9728 x 2560), and the Q6_K
//                    output heads (run as Q8_0) at 128 rows besides 1 and 7 (up to 128 x 2560 x 151936); slow without
//                    the IME
//   --bench          also measure launch cost, per-node cost, bandwidth, and Q4_0, Q4_1 and Q6_K (run as Q8_0) matmul
//                    time against the CPU
// Without a flag, a host without the device skips the device checks.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-flagos.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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

// reproducible values in [-scale, scale): a stream that continues across calls, and n values from a new one
struct random_stream {
    uint32_t s;

    explicit random_stream(uint32_t seed) : s(seed * 2654435761u + 1) {}

    void fill(float * v, int64_t n, float scale) {
        for (int64_t i = 0; i < n; i++) {
            s    = s * 1664525u + 1013904223u;
            v[i] = scale * ((float) (s >> 8) / 8388608.0f - 1.0f);
        }
    }
};

static std::vector<float> random_values(int64_t n, uint32_t seed, float scale) {
    std::vector<float> v(n);
    random_stream(seed).fill(v.data(), n, scale);
    return v;
}

// checks that ADD is claimed exactly, computes correctly, and that failures are reported without hanging
static int check_ops(ggml_backend_dev_t dev) {
    ggml_backend_buffer_type_t buft   = ggml_backend_dev_buffer_type(dev);
    const int64_t              n      = 1000;
    const int64_t              n_big  = 4000037;  // one row: the RVV kernel splits its elements (the reference, rows)
    const int64_t              n_rows = 4001;     // rows not a multiple of 8 tiles
    ggml_init_params           params = { ggml_tensor_overhead() * 32 + ggml_graph_overhead() * 8, nullptr, true };
    ggml_context *             ctx    = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * sum   = ggml_add(ctx, a, b);
    ggml_tensor * sum2  = ggml_add(ctx, sum, b);
    ggml_tensor * sum3  = ggml_add(ctx, sum2, b);
    ggml_tensor * view  = ggml_reshape_2d(ctx, a, 100, 10);
    ggml_tensor * diff  = ggml_sub(ctx, a, b);
    ggml_tensor * row   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100);
    ggml_tensor * bcast = ggml_add(ctx, view, row);
    ggml_tensor * tview = ggml_transpose(ctx, view);
    ggml_tensor * strided = ggml_add(ctx, tview, tview);
    ggml_tensor * h1    = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n);
    ggml_tensor * half  = ggml_add(ctx, h1, h1);
    ggml_tensor * big_a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_big);
    ggml_tensor * big_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_big);
    ggml_tensor * big   = ggml_add(ctx, big_a, big_b);
    ggml_tensor * mat_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n_rows);
    ggml_tensor * mat_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n_rows);
    ggml_tensor * mat   = ggml_add(ctx, mat_a, mat_b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    REQUIRE(buf != nullptr);

    // claimed exactly: inputs and views (no kernel), F32 ADD with broadcast (M2d); not SUB, strided elements or F16
    CHECK(ggml_backend_dev_supports_op(dev, a));
    CHECK(ggml_backend_dev_supports_op(dev, view));
    CHECK(ggml_backend_dev_supports_op(dev, tview));
    CHECK(ggml_backend_dev_supports_op(dev, sum));
    CHECK(ggml_backend_dev_supports_op(dev, bcast));
    CHECK(!ggml_backend_dev_supports_op(dev, diff));
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

    // large one-row ADD (the RVV kernel splits its elements over the tiles, with a tail)
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

    // ADD over rows that do not divide evenly among the tiles
    std::vector<float> ma = random_values(n * n_rows, 21, 4.0f), mb = random_values(n * n_rows, 22, 4.0f), mout(n * n_rows);
    ggml_backend_tensor_set(mat_a, ma.data(), 0, ggml_nbytes(mat_a));
    ggml_backend_tensor_set(mat_b, mb.data(), 0, ggml_nbytes(mat_b));
    ggml_cgraph * mat_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(mat_graph, mat);
    CHECK(ggml_backend_graph_compute(backend, mat_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(mat, mout.data(), 0, ggml_nbytes(mat));
    int64_t wrong_mat = 0;
    for (int64_t i = 0; i < n * n_rows; i++) {
        wrong_mat += mout[i] != ma[i] + mb[i];
    }
    CHECK(wrong_mat == 0);

    // an unclaimed op is rejected before anything runs
    std::fprintf(stderr, "(the next error line is expected)\n");
    ggml_cgraph * sub_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(sub_graph, diff);
    CHECK(ggml_backend_graph_compute(backend, sub_graph) == GGML_STATUS_FAILED);
    ggml_backend_free(backend);

    // a kernel failure inside the launch stops every tile and is reported (test-only switch). Repeated, in the
    // first step and in a later one: on the AI cores a failure seen at the wrong barrier would hang a launch
    constexpr int k_fail_runs = 100;
    int           fail_ok     = 0;
    for (const char * fail_node : { "0", "1" }) {
        set_env("FLAGOS_SPACEMIT_TEST_FAIL_NODE", fail_node);
        ggml_backend_t failing = ggml_backend_dev_init(dev, nullptr);
        set_env("FLAGOS_SPACEMIT_TEST_FAIL_NODE", nullptr);
        REQUIRE(failing != nullptr);
        for (int r = 0; r < k_fail_runs; r++) {
            fail_ok += ggml_backend_graph_compute(failing, chain) == GGML_STATUS_FAILED;
        }
        ggml_backend_free(failing);
    }
    CHECK(fail_ok == 2 * k_fail_runs);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    std::printf("ops      ADD claimed exactly (with broadcast; not SUB, strided or F16); 3-node chain, %lld-element row and "
                "%lldx%lld ADD exact: checked\n",
                (long long) n_big, (long long) n_rows, (long long) n);
    std::printf("backend  empty and view-only graphs succeed; unclaimed op returns FAILED; forced kernel failure in step 0 "
                "and 1 returns FAILED %d/%d times: checked\n",
                fail_ok, 2 * k_fail_runs);
    return 0;
}

//
// Q4_0, Q4_1, Q8_0 and Q6_K MUL_MAT (M2b, M2c): weights in IME layouts, compared with the CPU backend
//

// the values of random_values(k * n, seed, 1.0f) quantized to type: generated a few MB at a time and quantized on 8
// threads, since an output head (2560 x 151936) is 1.6 GB as floats and takes 14 s to quantize to Q6_K on one Mac core
static std::vector<uint8_t> quantize_weights(ggml_type type, int64_t k, int64_t n, uint32_t seed) {
    const int64_t        n_threads = 8;
    const int64_t        slab      = std::max<int64_t>(1, (1 << 20) / k);  // rows per thread and round
    const size_t         row_size  = ggml_row_size(type, k);
    std::vector<uint8_t> q(row_size * n);
    std::vector<float>   w(k * std::min(n, slab * n_threads));
    random_stream        values(seed);
    for (int64_t r0 = 0; r0 < n; r0 += slab * n_threads) {
        const int64_t nr = std::min(slab * n_threads, n - r0);
        values.fill(w.data(), k * nr, 1.0f);
        std::vector<std::thread> workers;
        for (int64_t r = 0; r < nr; r += slab) {
            workers.emplace_back([&, r] {
                ggml_quantize_chunk(type, w.data() + r * k, q.data() + (r0 + r) * row_size, 0, std::min(slab, nr - r), k,
                                    nullptr);
            });
        }
        for (std::thread & t : workers) {
            t.join();
        }
    }
    return q;
}

// the values of quantized bytes
static std::vector<float> dequantize(ggml_type type, const std::vector<uint8_t> & q) {
    std::vector<float> v(q.size() / ggml_type_size(type) * ggml_blck_size(type));
    ggml_get_type_traits(type)->to_float(q.data(), v.data(), (int64_t) v.size());
    return v;
}

// the values of n rows of k quantized weights, quantized to Q8_0 by ggml (per 32 values d = max|w| / 127 in fp16,
// q = round(w / d)): what the provider's Q6_K requantization stores, up to how ties round (ggml rounds them away from
// zero, the requantization to even). 32 rows at a time, as an output head is 1.6 GB as floats
static std::vector<uint8_t> requantize_q8_0(ggml_type type, const std::vector<uint8_t> & q, int64_t k, int64_t n) {
    const size_t         row_in = ggml_row_size(type, k), row_out = ggml_row_size(GGML_TYPE_Q8_0, k);
    std::vector<uint8_t> out(row_out * n);
    std::vector<float>   v(k * 32);
    for (int64_t r = 0; r < n; r += 32) {
        const int64_t nr = std::min<int64_t>(32, n - r);
        ggml_get_type_traits(type)->to_float(q.data() + r * row_in, v.data(), nr * k);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, v.data(), out.data() + r * row_out, 0, nr, k, nullptr);
    }
    return out;
}

// any NaN or Inf gives +inf, which fails every bound (a NaN would pass `e > bound`) and survives std::max
static double nmse(const float * out, const float * ref, int64_t n) {
    double err = 0.0, norm = 0.0;
    for (int64_t i = 0; i < n; i++) {
        err += ((double) out[i] - ref[i]) * ((double) out[i] - ref[i]);
        norm += (double) ref[i] * ref[i];
    }
    if (!std::isfinite(err) || !std::isfinite(norm)) {
        return HUGE_VAL;
    }
    return norm > 0.0 ? err / norm : err;
}

// one quantized weight [k x n], held twice: in the provider's buffer (repacked) and in a CPU buffer (the reference).
// The reference gets the provider's read-back: the GGUF bytes for Q4_0 and Q8_0, the converted weights for Q4_1, so a
// comparison measures the kernels and the tiling; check_weight_round_trip checks the conversion. With cpu_gguf (Q6_K)
// it gets the GGUF bytes instead: a Q6_K read is only the nearest Q6_K to the kernels' Q8_0 weights, so the comparison
// covers what the model sees, the requantization and the kernels; its error is mostly the requantization's own, so it
// only catches gross faults. w_q8 then holds ggml's Q8_0 of the GGUF values (requantize_q8_0), which a correct
// requantization matches up to ties: a comparison with it measures the requantization's faults, and the kernels'.
struct mm_weight {
    int64_t               k = 0, n = 0;
    ggml_context *        ctx_dev = nullptr, * ctx_cpu = nullptr;
    ggml_backend_buffer_t buf_dev = nullptr, buf_cpu = nullptr;
    ggml_tensor *         w_dev = nullptr, * w_cpu = nullptr, * w_q8 = nullptr;

    bool init(ggml_backend_dev_t dev, int64_t k_, int64_t n_, uint32_t seed, ggml_type type = GGML_TYPE_Q4_0,
              bool cpu_gguf = false) {
        return init_with(dev, type, k_, n_, quantize_weights(type, k_, n_, seed), cpu_gguf);
    }

    bool init_with(ggml_backend_dev_t dev, ggml_type type, int64_t k_, int64_t n_, std::vector<uint8_t> q,
                   bool cpu_gguf = false) {
        k                       = k_;
        n                       = n_;
        ggml_init_params params = { ggml_tensor_overhead() * 2, nullptr, true };
        ctx_dev                 = ggml_init(params);
        ctx_cpu                 = ggml_init(params);
        w_dev                   = ggml_new_tensor_2d(ctx_dev, type, k, n);
        w_cpu                   = ggml_new_tensor_2d(ctx_cpu, type, k, n);
        w_q8                    = cpu_gguf ? ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_Q8_0, k, n) : nullptr;
        buf_dev = ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev, ggml_backend_dev_buffer_type(dev));
        buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cpu, ggml_backend_cpu_buffer_type());
        if (buf_dev == nullptr || buf_cpu == nullptr) {
            return false;
        }
        ggml_backend_tensor_set(w_dev, q.data(), 0, q.size());
        if (cpu_gguf) {
            const std::vector<uint8_t> q8 = requantize_q8_0(type, q, k, n);
            ggml_backend_tensor_set(w_q8, q8.data(), 0, q8.size());
        } else {
            ggml_backend_tensor_get(w_dev, q.data(), 0, q.size());
        }
        ggml_backend_tensor_set(w_cpu, q.data(), 0, q.size());
        return true;
    }

    ~mm_weight() {
        ggml_backend_buffer_free(buf_dev);
        ggml_backend_buffer_free(buf_cpu);
        ggml_free(ctx_dev);
        ggml_free(ctx_cpu);
    }
};

// y = W x for m rows of x, on the provider (x in a CPU buffer, y in the provider's) and on the CPU backend (also with
// w.w_q8 if the weight has one, in its own graph)
struct mm_run {
    ggml_context *        ctx_host = nullptr, * ctx_dev = nullptr, * ctx_graph = nullptr;
    ggml_backend_buffer_t buf_host = nullptr, buf_dev = nullptr;
    ggml_tensor *         x = nullptr, * y = nullptr, * y_ref = nullptr, * y_q8 = nullptr;
    ggml_cgraph *         g_dev = nullptr, * g_cpu = nullptr, * g_q8 = nullptr;

    bool init(ggml_backend_dev_t dev, const mm_weight & w, int64_t m, uint32_t seed) {
        ggml_init_params params = { ggml_tensor_overhead() * 4, nullptr, true };
        ctx_host                = ggml_init(params);
        ctx_dev                 = ggml_init(params);
        ggml_init_params gparams = { ggml_graph_overhead() * 3, nullptr, true };
        ctx_graph                = ggml_init(gparams);
        x                        = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F32, w.k, m);
        y_ref                    = ggml_mul_mat(ctx_host, w.w_cpu, x);
        y_q8                     = w.w_q8 != nullptr ? ggml_mul_mat(ctx_host, w.w_q8, x) : nullptr;
        y                        = ggml_mul_mat(ctx_dev, w.w_dev, x);
        buf_host = ggml_backend_alloc_ctx_tensors_from_buft(ctx_host, ggml_backend_cpu_buffer_type());
        buf_dev  = ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev, ggml_backend_dev_buffer_type(dev));
        if (buf_host == nullptr || buf_dev == nullptr) {
            return false;
        }
        const std::vector<float> xv = random_values(w.k * m, seed, 2.0f);
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        g_dev = ggml_new_graph(ctx_graph);
        g_cpu = ggml_new_graph(ctx_graph);
        ggml_build_forward_expand(g_dev, y);
        ggml_build_forward_expand(g_cpu, y_ref);
        if (y_q8 != nullptr) {
            g_q8 = ggml_new_graph(ctx_graph);
            ggml_build_forward_expand(g_q8, y_q8);
        }
        return true;
    }

    ~mm_run() {
        ggml_backend_buffer_free(buf_host);
        ggml_backend_buffer_free(buf_dev);
        ggml_free(ctx_host);
        ggml_free(ctx_dev);
        ggml_free(ctx_graph);
    }

    // NMSE of the provider's result against the CPU's; -1 if the provider refused the op or failed. With e_q8, also the
    // NMSE against the CPU's result with w.w_q8 (-1 without one)
    double compare(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu, double * e_q8 = nullptr) {
        if (!ggml_backend_dev_supports_op(dev, y) || ggml_backend_graph_compute(backend, g_dev) != GGML_STATUS_SUCCESS ||
            ggml_backend_graph_compute(cpu, g_cpu) != GGML_STATUS_SUCCESS) {
            return -1.0;
        }
        std::vector<float> out(ggml_nelements(y));
        ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
        if (e_q8 != nullptr) {
            *e_q8 = g_q8 != nullptr && ggml_backend_graph_compute(cpu, g_q8) == GGML_STATUS_SUCCESS ?
                        nmse(out.data(), static_cast<const float *>(y_q8->data), ggml_nelements(y)) :
                        -1.0;
        }
        return nmse(out.data(), static_cast<const float *>(y_ref->data), ggml_nelements(y));
    }
};

constexpr double k_mul_mat_max_nmse = 5e-4;  // test-backend-ops' bound for MUL_MAT
// a Q6_K matmul against the CPU's on ggml's Q8_0 of the same values (mm_weight::w_q8): the ties that round apart give
// 4-5e-7 (Mac); faults in the requantization give 9e-6 (a stored scale 0.3% too large) to 2e-4 (rounding up by half)
constexpr double k_q6_k_requant_max_nmse = 5e-6;

// which tensors the provider claims for MUL_MAT: exactly Q4_0, Q4_1, Q8_0 and Q6_K weights in their IME layouts, in its
// buffer; and what its buffer type allocates for them (Q6_K is stored as Q8_0)
static int check_mul_mat_claims(ggml_backend_dev_t dev) {
    ggml_backend_buffer_type_t buft   = ggml_backend_dev_buffer_type(dev);
    ggml_init_params           params = { ggml_tensor_overhead() * 80, nullptr, true };
    ggml_context *             ctx    = ggml_init(params);
    ggml_context *             ctx_cpu = ggml_init(params);
    REQUIRE(ctx != nullptr && ctx_cpu != nullptr);

    ggml_tensor * w    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 64);
    ggml_tensor * x1   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
    ggml_tensor * x512 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 512);
    ggml_tensor * x4d  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 3, 2, 2);
    ggml_tensor * x16  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 4);
    ggml_tensor * xt   = ggml_transpose(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 256));
    ggml_tensor * w_k  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 2880, 64);
    ggml_tensor * x_k  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2880, 4);
    ggml_tensor * w_n  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 16);
    ggml_tensor * w_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, 256, 32, 2);
    ggml_tensor * x_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 4, 2);
    ggml_tensor * w_q8 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 256, 64);
    ggml_tensor * w_v  = ggml_view_2d(ctx, w, 256, 32, w->nb[1], 0);
    ggml_tensor * w_c  = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_Q4_0, 256, 64);
    ggml_tensor * v1   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, 256, 64);
    ggml_tensor * v1_k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, 2880, 64);
    ggml_tensor * v1_n = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, 256, 16);
    ggml_tensor * v1_3 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_1, 256, 32, 2);
    ggml_tensor * v1_v = ggml_view_2d(ctx, v1, 256, 32, v1->nb[1], 0);
    ggml_tensor * v1_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_Q4_1, 256, 64);
    ggml_tensor * w8_k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2880, 64);
    ggml_tensor * w8_n = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 256, 16);
    ggml_tensor * w6   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2560, 64);
    ggml_tensor * w6_n = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2560, 16);
    ggml_tensor * w6_o = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 17);
    ggml_tensor * w6_3 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q6_K, 2560, 32, 2);
    ggml_tensor * w6_v = ggml_view_2d(ctx, w6, 2560, 32, w6->nb[1], 0);
    ggml_tensor * w6_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_Q6_K, 2560, 64);
    ggml_tensor * x6   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 1);
    ggml_tensor * x6_m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 512);
    ggml_tensor * x6_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2560, 4);
    ggml_tensor * x6_3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2560, 4, 2);
    ggml_tensor * x6_o = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);

    // claimed for any row count, before allocation (test-backend-ops) and with a zero-size buffer of our type
    // standing in for the weight's (what llama.cpp does at load, src/llama-model-loader.cpp:907-947)
    CHECK(ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w, x1)));
    CHECK(ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w, x512)));
    CHECK(ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w, x4d)));
    CHECK(ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, v1, x1)));      // Q4_1: any row length % 32 (blocks of 32)
    CHECK(ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, v1, x512)));
    CHECK(ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, v1_k, x_k)));
    CHECK(ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w_q8, x1)));    // Q8_0: any row length % 32
    CHECK(ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w8_k, x_k)));
    CHECK(ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w6, x6)));      // Q6_K (row length % 256, as every Q6_K)
    CHECK(ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w6, x6_m)));
    ggml_tensor * probe = ggml_mul_mat(ctx, w, x512);
    w->buffer           = ggml_backend_buft_alloc_buffer(buft, 0);
    CHECK(ggml_backend_dev_supports_op(dev, probe));
    ggml_backend_buffer_free(w->buffer);
    w->buffer = nullptr;
    ggml_tensor * head = ggml_mul_mat(ctx, w6, x6);  // how llama.cpp places the output head
    w6->buffer         = ggml_backend_buft_alloc_buffer(buft, 0);
    CHECK(ggml_backend_dev_supports_op(dev, head));
    ggml_backend_buffer_free(w6->buffer);
    w6->buffer = nullptr;

    // refused: other layouts and types, views, non-contiguous or F16 activations, a weight in a CPU buffer
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w_k, x_k)));   // row length not a multiple of 256
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w_n, x1)));    // rows not a multiple of 32
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w_3d, x_3d))); // 3-D weight
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, v1_n, x1)));   // Q4_1, rows not a multiple of 32
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, v1_3, x_3d))); // Q4_1, 3-D weight
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, v1_v, x1)));   // Q4_1, view of a weight
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, v1, x16)));    // Q4_1, F16 activations
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w8_n, x1)));   // Q8_0, rows not a multiple of 32
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w6_n, x6)));   // Q6_K, rows not a multiple of 32
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w6_o, x6_o))); // Q6_K, 17 rows (as in test-backend-ops)
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w6_3, x6_3))); // Q6_K, 3-D weight
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w6_v, x6)));   // Q6_K, view of a weight
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w6, x6_h)));   // Q6_K, F16 activations
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w_v, x1)));    // view of a weight
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w, x16)));     // F16 activations
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w, xt)));      // non-contiguous activations
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cpu, ggml_backend_cpu_buffer_type());
    REQUIRE(buf_cpu != nullptr);
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w_c, x1)));    // plain Q4_0 in a CPU buffer
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, v1_c, x1)));   // plain Q4_1 in a CPU buffer
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w6_c, x6)));   // plain Q6_K in a CPU buffer

    // a Q6_K weight takes 34 bytes per 32 values (Q8_0) instead of 26.25, the others their own size. Required: the
    // round trip and the matmuls below would overrun a smaller allocation
    REQUIRE(ggml_backend_buft_get_alloc_size(buft, w6) == 64 * (2560 / 32) * 34);
    CHECK(ggml_backend_buft_get_alloc_size(buft, w8_k) == ggml_nbytes(w8_k));
    CHECK(ggml_backend_buft_get_alloc_size(buft, w) == ggml_nbytes(w));

    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx);
    ggml_free(ctx_cpu);
    return 0;
}

// a Q4_1 block as GGUF stores it: a weight is d * q + m
struct q4_1_block {
    ggml_fp16_t d, m;
    uint8_t     qs[16];
};
static_assert(sizeof(q4_1_block) == 20, "Q4_1 block size");

// the conversion of the q4_1 32x32 layout (plan.md, M2c design; make_block_q4_1x32 in the provider): the zero point
// zp = clamp(round(-m / d), 0, 15) replaces the minimum, so a read gives back m = -zp * d; d and the quants are kept
static std::vector<uint8_t> q4_1_converted(std::vector<uint8_t> bytes) {
    auto * b = reinterpret_cast<q4_1_block *>(bytes.data());
    for (size_t i = 0; i < bytes.size() / sizeof(q4_1_block); i++) {
        const float d  = ggml_fp16_to_fp32(b[i].d);
        const float zp = std::min(15.0f, std::max(0.0f, -std::nearbyintf(ggml_fp16_to_fp32(b[i].m) / d)));
        const float m  = -zp * d;
        b[i].m         = ggml_fp32_to_fp16(m == 0.0f ? 0.0f : m);
    }
    return bytes;
}

// the largest change of the minimum, in steps d: every weight of a block moves by that much
static double q4_1_max_shift(const std::vector<uint8_t> & from, const std::vector<uint8_t> & to) {
    const auto * a = reinterpret_cast<const q4_1_block *>(from.data());
    const auto * b = reinterpret_cast<const q4_1_block *>(to.data());
    double       worst = 0.0;
    for (size_t i = 0; i < from.size() / sizeof(q4_1_block); i++) {
        const double d = ggml_fp16_to_fp32(a[i].d);
        worst = std::max(worst, std::fabs((double) ggml_fp16_to_fp32(b[i].m) - ggml_fp16_to_fp32(a[i].m)) / d);
    }
    return worst;
}

// set_tensor repacks; get_tensor restores the GGUF bytes (Q4_0, Q8_0), returns the converted weights (Q4_1) or the
// nearest Q6_K to the stored Q8_0 values (Q6_K); partial access works (whole writes only for Q6_K)
static int check_weight_round_trip(ggml_backend_dev_t dev) {
    ggml_init_params params = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_tensor *         w   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 512, 64);
    ggml_tensor *         w1  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, 512, 64);
    ggml_tensor *         w8  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 512, 64);
    ggml_tensor *         w6  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 512, 64);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev));
    REQUIRE(buf != nullptr);

    std::vector<uint8_t>       q = quantize_weights(GGML_TYPE_Q4_0, 512, 64, 7), out(q.size());
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
    CHECK(std::memcmp(w->data, q.data(), q.size()) != 0);  // stored repacked (the buffer is host memory underneath)
    ggml_backend_tensor_get(w, out.data(), 0, out.size());
    CHECK(out == q);
    ggml_backend_tensor_get(w, out.data(), 1000, 300);
    CHECK(std::memcmp(out.data(), q.data() + 1000, 300) == 0);

    std::vector<uint8_t> patch(100, 0x5a);
    ggml_backend_tensor_set(w, patch.data(), 500, patch.size());
    std::copy(patch.begin(), patch.end(), q.begin() + 500);
    ggml_backend_tensor_memset(w, 0x11, 36, 18);
    std::fill(q.begin() + 36, q.begin() + 54, (uint8_t) 0x11);
    ggml_backend_tensor_get(w, out.data(), 0, out.size());
    CHECK(out == q);

    // Q4_1: reads return the converted weights, every weight within half a step of the GGUF one (random blocks hold
    // values of both signs, so no zero point is clamped); a second round trip, and partial access, convert nothing again
    const std::vector<uint8_t> q1 = quantize_weights(GGML_TYPE_Q4_1, 512, 64, 9);
    std::vector<uint8_t>       c1(q1.size()), again(q1.size());
    ggml_backend_tensor_set(w1, q1.data(), 0, q1.size());
    ggml_backend_tensor_get(w1, c1.data(), 0, c1.size());
    CHECK(c1 == q4_1_converted(q1));
    CHECK(c1 != q1);
    CHECK(q4_1_max_shift(q1, c1) <= 0.5 + 1e-2);  // plus the fp16 rounding of m
    ggml_backend_tensor_set(w1, c1.data(), 0, c1.size());
    ggml_backend_tensor_get(w1, again.data(), 0, again.size());
    CHECK(again == c1);
    ggml_backend_tensor_get(w1, again.data(), 1000, 300);
    CHECK(std::memcmp(again.data(), c1.data() + 1000, 300) == 0);
    ggml_backend_tensor_set(w1, patch.data(), 500, patch.size());  // inside blocks 25-29: their d, m and quants
    std::copy(patch.begin(), patch.end(), c1.begin() + 500);
    ggml_backend_tensor_get(w1, again.data(), 0, again.size());
    CHECK(again == q4_1_converted(c1));

    // Q8_0: the layout permutes the bytes, so it reads back as Q4_0 does
    std::vector<uint8_t> q8 = quantize_weights(GGML_TYPE_Q8_0, 512, 64, 11), out8(q8.size());
    ggml_backend_tensor_set(w8, q8.data(), 0, q8.size());
    CHECK(std::memcmp(w8->data, q8.data(), q8.size()) != 0);
    ggml_backend_tensor_get(w8, out8.data(), 0, out8.size());
    CHECK(out8 == q8);
    ggml_backend_tensor_get(w8, out8.data(), 1000, 300);
    CHECK(std::memcmp(out8.data(), q8.data() + 1000, 300) == 0);
    ggml_backend_tensor_set(w8, patch.data(), 500, patch.size());
    std::copy(patch.begin(), patch.end(), q8.begin() + 500);
    ggml_backend_tensor_memset(w8, 0x11, 36, 18);
    std::fill(q8.begin() + 36, q8.begin() + 54, (uint8_t) 0x11);
    ggml_backend_tensor_get(w8, out8.data(), 0, out8.size());
    CHECK(out8 == q8);

    // Q6_K: stored requantized to Q8_0, so a read is the nearest Q6_K to the stored values, within NMSE 1e-5 of the
    // GGUF weights (these uniform random weights give 1.0e-6; normal ones with outliers would give about 6e-5); writing
    // it back and reading again stays as close; a whole-tensor memset reads back exactly. check_mul_mat checks the
    // stored values themselves
    const std::vector<uint8_t> q6 = quantize_weights(GGML_TYPE_Q6_K, 512, 64, 13);
    std::vector<uint8_t>       r6(q6.size()), again6(q6.size());
    ggml_backend_tensor_set(w6, q6.data(), 0, q6.size());
    CHECK(std::memcmp(w6->data, q6.data(), q6.size()) != 0);
    ggml_backend_tensor_get(w6, r6.data(), 0, r6.size());
    const std::vector<float> f6 = dequantize(GGML_TYPE_Q6_K, q6), f6_read = dequantize(GGML_TYPE_Q6_K, r6);
    CHECK(nmse(f6_read.data(), f6.data(), (int64_t) f6.size()) <= 1e-5);
    ggml_backend_tensor_set(w6, r6.data(), 0, r6.size());
    ggml_backend_tensor_get(w6, again6.data(), 0, again6.size());
    const std::vector<float> f6_again = dequantize(GGML_TYPE_Q6_K, again6);
    CHECK(nmse(f6_again.data(), f6_read.data(), (int64_t) f6_read.size()) <= 1e-5);
    // a 32-row group of zeros, as ggml quantizes an output head's unused vocabulary rows (zero bytes): it stays zero
    // (the requantization's zero guard), and the other group reads back as before, since rows are requantized alone
    const size_t         group6 = 32 * ggml_row_size(GGML_TYPE_Q6_K, 512);
    std::vector<uint8_t> z6(q6.begin(), q6.begin() + group6);
    z6.resize(q6.size(), 0);
    ggml_backend_tensor_set(w6, z6.data(), 0, z6.size());
    ggml_backend_tensor_get(w6, again6.data(), 0, again6.size());
    CHECK(std::equal(again6.begin(), again6.begin() + group6, r6.begin()));
    CHECK(std::all_of(again6.begin() + group6, again6.end(), [](uint8_t b) { return b == 0; }));
    ggml_backend_tensor_memset(w6, 0, 0, ggml_nbytes(w6));
    ggml_backend_tensor_get(w6, again6.data(), 0, again6.size());
    CHECK(std::all_of(again6.begin(), again6.end(), [](uint8_t b) { return b == 0; }));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return 0;
}

// MUL_MAT then ADD in one launch (3 steps): the residual pattern of every layer
static int check_mul_mat_add(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu) {
    mm_weight w;
    REQUIRE(w.init(dev, 512, 96, 11));
    const int64_t    m      = 5;
    ggml_init_params params = { ggml_tensor_overhead() * 8 + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context *   ctx_h  = ggml_init(params);
    ggml_context *   ctx_d  = ggml_init(params);
    ggml_tensor *    x      = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F32, 512, m);
    ggml_tensor *    r      = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F32, 96, m);
    ggml_tensor *    z_ref  = ggml_add(ctx_h, ggml_mul_mat(ctx_h, w.w_cpu, x), r);
    ggml_tensor *    z      = ggml_add(ctx_d, ggml_mul_mat(ctx_d, w.w_dev, x), r);
    ggml_backend_buffer_t bh = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_t bd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, ggml_backend_dev_buffer_type(dev));
    REQUIRE(bh != nullptr && bd != nullptr);
    const std::vector<float> xv = random_values(512 * m, 3, 2.0f), rv = random_values(96 * m, 4, 1.0f);
    ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(r, rv.data(), 0, ggml_nbytes(r));
    ggml_cgraph * g  = ggml_new_graph(ctx_d);
    ggml_cgraph * gc = ggml_new_graph(ctx_h);
    ggml_build_forward_expand(g, z);
    ggml_build_forward_expand(gc, z_ref);
    CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_graph_compute(cpu, gc) == GGML_STATUS_SUCCESS);
    std::vector<float> out(ggml_nelements(z));
    ggml_backend_tensor_get(z, out.data(), 0, ggml_nbytes(z));
    CHECK(nmse(out.data(), static_cast<const float *>(z_ref->data), ggml_nelements(z)) < k_mul_mat_max_nmse);
    ggml_backend_buffer_free(bh);
    ggml_backend_buffer_free(bd);
    ggml_free(ctx_h);
    ggml_free(ctx_d);
    return 0;
}

// how much the zero-point conversion alone changes a matmul: the CPU backend with the GGUF Q4_1 weights against the
// CPU backend with the converted ones, random weights at Qwen3-0.6B's ffn_down shape, 16 rows (informational: the
// accuracy budget is decided by perplexity and KLD on the model, plan.md M2c design)
static double q4_1_conversion_nmse(ggml_backend_dev_t dev, ggml_backend_t cpu) {
    const int64_t k = 3072, n = 1024, m = 16;
    mm_weight     w;
    if (!w.init(dev, k, n, 21, GGML_TYPE_Q4_1)) {
        return -1.0;
    }
    ggml_init_params params = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    ggml_tensor *    w_gguf = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, k, n);
    ggml_tensor *    x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    ggml_tensor *    y_conv = ggml_mul_mat(ctx, w.w_cpu, x);
    ggml_tensor *    y_gguf = ggml_mul_mat(ctx, w_gguf, x);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    double                e   = -1.0;
    if (buf != nullptr) {
        const std::vector<uint8_t> q  = quantize_weights(GGML_TYPE_Q4_1, k, n, 21);
        const std::vector<float>   xv = random_values(k * m, 22, 2.0f);
        ggml_backend_tensor_set(w_gguf, q.data(), 0, q.size());
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, y_conv);
        ggml_build_forward_expand(g, y_gguf);
        if (ggml_backend_graph_compute(cpu, g) == GGML_STATUS_SUCCESS) {
            e = nmse(static_cast<const float *>(y_conv->data), static_cast<const float *>(y_gguf->data), n * m);
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return e;
}

// zero point times activation sum beyond 16 bits: blocks with zero point 15 (values 0 down to -1) against 32 equal
// activations need 15 x 32 x 127 = 60960. Upstream's active branch of the 1-row kernel (gemm_kernel_i8i4_m1) multiplies
// in 16 bits and would wrap; the provider runs its exact branch, as the 4-row kernel is (plan.md R12). Returns the NMSE
// against the CPU of 1 row and of 4 rows
static std::pair<double, double> q4_1_zero_point_probe(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu) {
    const int64_t      k = 256, n = 32;
    std::vector<float> wv(k * n);
    for (int64_t i = 0; i < k * n; i++) {
        wv[i] = -(float) (i % 16) / 15.0f;  // per block of 32: d = 1/15, m = -1, so zero point 15
    }
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q4_1, k) * n);
    ggml_quantize_chunk(GGML_TYPE_Q4_1, wv.data(), q.data(), 0, n, k, nullptr);
    mm_weight w;
    if (!w.init_with(dev, GGML_TYPE_Q4_1, k, n, q)) {
        return { -1.0, -1.0 };
    }
    double e[2] = { -1.0, -1.0 };
    for (int i = 0; i < 2; i++) {
        const int64_t m = i == 0 ? 1 : 4;
        mm_run        run;
        if (run.init(dev, w, m, 1)) {
            const std::vector<float> ones(k * m, 1.0f);
            ggml_backend_tensor_set(run.x, ones.data(), 0, ggml_nbytes(run.x));
            e[i] = run.compare(dev, backend, cpu);
        }
    }
    return { e[0], e[1] };
}

// the provider against the CPU backend for shapes and row counts that take every path: 1 row (GEMV), 113 and more
// rows with n <= 64 rows (path A), the rest (path C); 4-row blocks with partial tails. Q4_1 is compared with its
// converted weights, Q6_K with the GGUF ones (mm_weight)
static int check_mul_mat(ggml_backend_dev_t dev, bool full) {
    if (check_mul_mat_claims(dev) != 0 || check_weight_round_trip(dev) != 0) {
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 8);

    // the output heads also at 128 rows with --full: a batch that needs every token's logits (perplexity, KLD)
    const std::vector<int64_t> head_rows = full ? std::vector<int64_t>{ 1, 7, 128 } : std::vector<int64_t>{ 1, 7 };
    struct shape { ggml_type type; int64_t k, n; std::vector<int64_t> rows; };
    std::vector<shape> shapes = {
        { GGML_TYPE_Q4_0, 256, 32, { 1, 2, 4, 5, 16, 64, 113, 128 } },
        { GGML_TYPE_Q4_0, 512, 96, { 1, 2, 4, 5, 16, 64, 113, 128 } },
        { GGML_TYPE_Q4_0, 1024, 2048, { 1, 4, 7, 64, 128 } },  // Qwen3-0.6B q projection
        { GGML_TYPE_Q4_0, 2560, 1024, { 1, 5, 128 } },         // Qwen3-4B k/v projection
        { GGML_TYPE_Q4_1, 256, 32, { 1, 2, 4, 5, 16, 64, 113, 128 } },
        { GGML_TYPE_Q4_1, 2880, 64, { 1, 5, 113 } },           // 90 K blocks, not a multiple of 16
        { GGML_TYPE_Q4_1, 3072, 1024, { 1, 4, 7, 64, 128 } },  // Qwen3-0.6B ffn down
        { GGML_TYPE_Q8_0, 256, 32, { 1, 2, 4, 5, 16, 64, 113, 128 } },
        { GGML_TYPE_Q8_0, 2880, 64, { 1, 5, 113 } },           // 90 K blocks
        { GGML_TYPE_Q8_0, 2560, 1024, { 1, 7, 128 } },         // a Qwen3-4B k/v projection's shape
        { GGML_TYPE_Q6_K, 256, 32, { 1, 2, 4, 5, 16, 64, 113, 128 } },
        { GGML_TYPE_Q6_K, 2560, 151936, head_rows },           // Qwen3-4B output head
        { GGML_TYPE_Q6_K, 1024, 151936, head_rows },           // Qwen3-0.6B output head
    };
    if (full) {  // Qwen3-4B q, attention output, ffn gate/up, ffn down, and its Q4_1 ffn down
        for (const auto & s : { std::pair<int64_t, int64_t>{ 2560, 4096 }, { 4096, 2560 }, { 2560, 9728 }, { 9728, 2560 } }) {
            shapes.push_back({ GGML_TYPE_Q4_0, s.first, s.second, { 1, 7, 128, 512 } });
        }
        shapes.push_back({ GGML_TYPE_Q4_1, 9728, 2560, { 1, 7, 128, 512 } });
    }

    const ggml_type types[4]    = { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K };
    int             n_cases[4]  = { 0, 0, 0, 0 };
    int64_t         max_rows[4] = { 0, 0, 0, 0 };
    double          max_nmse[4] = { 0.0, 0.0, 0.0, 0.0 };
    double          max_nmse_q8 = 0.0;  // Q6_K against the CPU's Q8_0 of the same values
    for (const shape & s : shapes) {
        const int t = (int) (std::find(types, types + 4, s.type) - types);
        mm_weight w;
        REQUIRE(w.init(dev, s.k, s.n, (uint32_t) (s.k + s.n), s.type, s.type == GGML_TYPE_Q6_K));
        for (int64_t m : s.rows) {
            mm_run run;
            REQUIRE(run.init(dev, w, m, (uint32_t) m));
            double       e_q8 = -1.0;
            const double e    = run.compare(dev, backend, cpu, &e_q8);
            if (e < 0.0 || e > k_mul_mat_max_nmse) {
                std::fprintf(stderr, "FAIL: %s matmul k=%lld n=%lld rows=%lld: %s %.3e\n", ggml_type_name(s.type),
                             (long long) s.k, (long long) s.n, (long long) m, e < 0.0 ? "refused or failed" : "NMSE", e);
                g_failures++;
            } else if (s.type == GGML_TYPE_Q6_K && (e_q8 < 0.0 || e_q8 > k_q6_k_requant_max_nmse)) {
                std::fprintf(stderr, "FAIL: Q6_K matmul k=%lld n=%lld rows=%lld against the CPU's Q8_0 of its values: "
                             "NMSE %.3e (bound %.0e)\n",
                             (long long) s.k, (long long) s.n, (long long) m, e_q8, k_q6_k_requant_max_nmse);
                g_failures++;
            }
            max_nmse[t] = std::max(max_nmse[t], e);
            max_nmse_q8 = std::max(max_nmse_q8, e_q8);
            max_rows[t] = std::max(max_rows[t], m);
            n_cases[t]++;
        }
    }
    check_mul_mat_add(dev, backend, cpu);
    const double conversion = q4_1_conversion_nmse(dev, cpu);
    const auto   probe      = q4_1_zero_point_probe(dev, backend, cpu);
    CHECK(conversion >= 0.0);  // ran; the value is informational
    if (probe.first < 0.0 || probe.first > k_mul_mat_max_nmse || probe.second < 0.0 || probe.second > k_mul_mat_max_nmse) {
        std::fprintf(stderr, "FAIL: Q4_1 matmul, zero point 15 against 32 equal activations: NMSE 1 row %.3e, 4 rows %.3e\n",
                     probe.first, probe.second);
        g_failures++;
    }

    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    const char * kernels = std::getenv("FLAGOS_SPACEMIT_TEST_REFERENCE") != nullptr ? ", reference kernels" : "";
    std::printf("matmul   Q4_0 (IME layout 32x256) vs CPU: %d cases, rows 1-%d, max NMSE %.2e (bound %.0e)%s\n",
                n_cases[0], (int) max_rows[0], max_nmse[0], k_mul_mat_max_nmse, kernels);
    std::printf("matmul   Q4_1 (IME layout 32x32) vs CPU on the converted weights: %d cases, rows 1-%d, max NMSE %.2e "
                "(bound %.0e); zero point 15 against 32 equal activations: NMSE 1 row %.2e, 4 rows %.2e%s\n",
                n_cases[1], (int) max_rows[1], max_nmse[1], k_mul_mat_max_nmse, probe.first, probe.second, kernels);
    std::printf("matmul   Q8_0 (IME layout 32x32) vs CPU: %d cases, rows 1-%d, max NMSE %.2e (bound %.0e)%s\n",
                n_cases[2], (int) max_rows[2], max_nmse[2], k_mul_mat_max_nmse, kernels);
    std::printf("matmul   Q6_K (requantized to Q8_0, IME layout 32x32) vs CPU's Q6_K: %d cases, rows 1-%d, max NMSE %.2e "
                "(bound %.0e); vs CPU's Q8_0 of the same values (the requantization): max NMSE %.2e (bound %.0e)%s\n",
                n_cases[3], (int) max_rows[3], max_nmse[3], k_mul_mat_max_nmse, max_nmse_q8, k_q6_k_requant_max_nmse,
                kernels);
    std::printf("matmul   claims exact (any row count; not other layouts, types, views, F16 or strided activations, "
                "CPU-buffer weights); Q6_K allocation size; repack round trip (Q4_1 converted, within half a step; Q6_K "
                "read as the nearest Q6_K, within NMSE 1e-5); MUL_MAT->ADD graph: checked\n");
    std::printf("info     Q4_1 conversion alone (CPU, GGUF against converted weights), random weights 3072x1024, 16 rows: "
                "NMSE %.2e\n",
                conversion);
    return 0;
}

//
// M2d: the ops of a layer together in one launch, against the CPU backend
//

struct layer_weights {
    ggml_tensor *attn_norm, *q_norm, *k_norm, *ffn_norm, *wq, *wk, *wv, *wo, *wg, *wu, *wd;
};

// a Qwen3-like layer (all dimensions small): d hidden, nh query and nkv key/value heads of hd, ff in the FFN
constexpr int64_t k_d = 256, k_hd = 128, k_nh = 2, k_nkv = 1, k_ff = 512, k_slots = 16;

// down: the FFN down projection's type (Qwen3-4B Q4_0 has Q4_1 there in its first 4 layers)
static layer_weights new_layer_weights(ggml_context * ctx, ggml_type down) {
    layer_weights w;
    w.attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k_d);
    w.q_norm    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k_hd);
    w.k_norm    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k_hd);
    w.ffn_norm  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k_d);
    w.wq        = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k_d, k_nh * k_hd);
    w.wk        = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k_d, k_nkv * k_hd);
    w.wv        = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k_d, k_nkv * k_hd);
    w.wo        = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k_nh * k_hd, k_d);
    w.wg        = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k_d, k_ff);
    w.wu        = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k_d, k_ff);
    w.wd        = ggml_new_tensor_2d(ctx, down, k_ff, k_d);
    return w;
}

static std::vector<ggml_tensor *> layer_tensors(const layer_weights & w) {
    return { w.attn_norm, w.q_norm, w.k_norm, w.ffn_norm, w.wq, w.wk, w.wv, w.wo, w.wg, w.wu, w.wd };
}

static void set_layer_weights(const layer_weights & w) {
    uint32_t seed = 100;
    for (ggml_tensor * t : { w.attn_norm, w.q_norm, w.k_norm, w.ffn_norm }) {
        std::vector<float> v = random_values(t->ne[0], seed++, 0.2f);
        for (float & x : v) {
            x += 1.0f;
        }
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    }
    for (ggml_tensor * t : { w.wq, w.wk, w.wv, w.wo, w.wg, w.wu, w.wd }) {
        const std::vector<uint8_t> q = quantize_weights(t->type, t->ne[0], t->ne[1], seed++);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
}

// the CPU computes with the weights the provider holds: the same bytes, a Q4_1 weight after its conversion (mm_weight)
static void copy_layer_weights(const layer_weights & from, const layer_weights & to) {
    const std::vector<ggml_tensor *> a = layer_tensors(from), b = layer_tensors(to);
    for (size_t i = 0; i < a.size(); i++) {
        std::vector<uint8_t> bytes(ggml_nbytes(a[i]));
        ggml_backend_tensor_get(a[i], bytes.data(), 0, bytes.size());
        ggml_backend_tensor_set(b[i], bytes.data(), 0, bytes.size());
    }
}

struct layer_out {
    ggml_tensor *full, *out, *set_k, *set_v;  // full: the layer output of every token; out: the last token's row
};

// norm, Q/K/V, Q and K norms, RoPE (NEOX), KV writes, output projection, residual, FFN norm, SwiGLU FFN, residual,
// and the last token's row: every op of a Qwen3 layer except attention, whose output q stands in for here
static layer_out build_layer(ggml_context * ctx, const layer_weights & w, ggml_tensor * x, ggml_tensor * pos,
                             ggml_tensor * kv_idx, ggml_tensor * out_ids, ggml_tensor * k_cache, ggml_tensor * v_cache,
                             int64_t n_tokens) {
    const float   eps = 1e-6f;
    ggml_tensor * h   = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w.attn_norm);
    ggml_tensor * q   = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w.wq, h), k_hd, k_nh, n_tokens);
    q                 = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps), w.q_norm);
    q = ggml_rope_ext(ctx, q, pos, nullptr, k_hd, GGML_ROPE_TYPE_NEOX, 0, 1e6f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w.wk, h), k_hd, k_nkv, n_tokens);
    k               = ggml_mul(ctx, ggml_rms_norm(ctx, k, eps), w.k_norm);
    k = ggml_rope_ext(ctx, k, pos, nullptr, k_hd, GGML_ROPE_TYPE_NEOX, 0, 1e6f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * v = ggml_mul_mat(ctx, w.wv, h);

    layer_out o;
    o.set_k = ggml_set_rows(ctx, k_cache, ggml_reshape_2d(ctx, k, k_hd * k_nkv, n_tokens), kv_idx);
    o.set_v = ggml_set_rows(ctx, v_cache, v, kv_idx);

    ggml_tensor * f   = ggml_add(ctx, ggml_mul_mat(ctx, w.wo, ggml_reshape_2d(ctx, q, k_hd * k_nh, n_tokens)), x);
    ggml_tensor * g   = ggml_mul(ctx, ggml_rms_norm(ctx, f, eps), w.ffn_norm);
    ggml_tensor * s   = ggml_swiglu_split(ctx, ggml_mul_mat(ctx, w.wg, g), ggml_mul_mat(ctx, w.wu, g));
    o.full            = ggml_add(ctx, ggml_mul_mat(ctx, w.wd, s), f);
    o.out             = ggml_get_rows(ctx, o.full, out_ids);
    return o;
}

static std::vector<float> f16_to_f32(const std::vector<uint8_t> & bytes) {
    std::vector<float> out(bytes.size() / sizeof(ggml_fp16_t));
    ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(bytes.data()), out.data(), (int64_t) out.size());
    return out;
}

// a layer chains three quantized matmuls, each quantizing its activations to 8 bits; on the Mac the whole output's NMSE
// against the CPU is about 4.0e-4 at 7 tokens, above a single matmul's bound
constexpr double k_layer_max_nmse = 1e-3;

// every contiguous F32 node of a computed graph is finite. In a layer a NaN from a norm, RoPE or SwiGLU would
// otherwise reach the output as finite values: the next matmul's activation quantization ignores it in its scale
static bool graph_finite(ggml_cgraph * g) {
    for (int i = 0; i < ggml_graph_n_nodes(g); i++) {
        ggml_tensor * t = ggml_graph_node(g, i);
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
            continue;
        }
        std::vector<float> v(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        if (!std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); })) {
            return false;
        }
    }
    return true;
}

// the layer on the provider, with the KV cache in a CPU buffer and the norm and quantized weights in the provider's
// buffer (where llama.cpp puts them), against the same layer on the CPU backend; returns the larger NMSE of the whole
// output and the caches, or -1 when refused, failed, or the last-token row is not an exact copy of the provider's output
static double layer_nmse(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu, int64_t n_tokens,
                         ggml_type down) {
    ggml_init_params params = { ggml_tensor_overhead() * 128 + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context *   ctx_h  = ggml_init(params);  // inputs and caches: CPU buffer
    ggml_context *   ctx_wd = ggml_init(params);  // weights for the provider: its buffer
    ggml_context *   ctx_wc = ggml_init(params);  // weights for the CPU: CPU buffer
    ggml_context *   ctx_gd = ggml_init(params);  // the provider's graph
    ggml_context *   ctx_gc = ggml_init(params);  // the CPU's graph

    ggml_tensor * x       = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F32, k_d, n_tokens);
    ggml_tensor * pos     = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, n_tokens);
    ggml_tensor * kv_idx  = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I64, n_tokens);
    ggml_tensor * out_ids = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, 1);
    ggml_tensor * kc_d    = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F16, k_hd * k_nkv, k_slots);
    ggml_tensor * vc_d    = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F16, k_hd * k_nkv, k_slots);
    ggml_tensor * kc_c    = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F16, k_hd * k_nkv, k_slots);
    ggml_tensor * vc_c    = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F16, k_hd * k_nkv, k_slots);
    const layer_weights wd = new_layer_weights(ctx_wd, down);
    const layer_weights wc = new_layer_weights(ctx_wc, down);

    ggml_backend_buffer_t bh  = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_t bwd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wd, ggml_backend_dev_buffer_type(dev));
    ggml_backend_buffer_t bwc = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wc, ggml_backend_cpu_buffer_type());
    if (bh == nullptr || bwd == nullptr || bwc == nullptr) {
        return -1.0;
    }
    ggml_backend_buffer_clear(bh, 0);
    set_layer_weights(wd);
    copy_layer_weights(wd, wc);

    const std::vector<float> xv = random_values(k_d * n_tokens, 5, 1.0f);
    std::vector<int32_t>     pv(n_tokens);
    std::vector<int64_t>     iv(n_tokens);
    for (int64_t t = 0; t < n_tokens; t++) {
        pv[t] = (int32_t) (40 + t);
        iv[t] = (3 + 5 * t) % k_slots;  // distinct slots
    }
    const int32_t last = (int32_t) (n_tokens - 1);
    ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(pos, pv.data(), 0, ggml_nbytes(pos));
    ggml_backend_tensor_set(kv_idx, iv.data(), 0, ggml_nbytes(kv_idx));
    ggml_backend_tensor_set(out_ids, &last, 0, sizeof(last));

    const layer_out od = build_layer(ctx_gd, wd, x, pos, kv_idx, out_ids, kc_d, vc_d, n_tokens);
    const layer_out oc = build_layer(ctx_gc, wc, x, pos, kv_idx, out_ids, kc_c, vc_c, n_tokens);
    ggml_backend_buffer_t bgd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_gd, ggml_backend_dev_buffer_type(dev));
    ggml_backend_buffer_t bgc = ggml_backend_alloc_ctx_tensors_from_buft(ctx_gc, ggml_backend_cpu_buffer_type());
    ggml_cgraph *         gd  = ggml_new_graph(ctx_gd);
    ggml_cgraph *         gc  = ggml_new_graph(ctx_gc);
    for (ggml_tensor * t : { od.out, od.set_k, od.set_v }) {
        ggml_build_forward_expand(gd, t);
    }
    for (ggml_tensor * t : { oc.out, oc.set_k, oc.set_v }) {
        ggml_build_forward_expand(gc, t);
    }

    double worst = -1.0;
    if (bgd != nullptr && bgc != nullptr && ggml_backend_graph_compute(backend, gd) == GGML_STATUS_SUCCESS &&
        ggml_backend_graph_compute(cpu, gc) == GGML_STATUS_SUCCESS) {
        std::vector<float> full(ggml_nelements(od.full)), out(ggml_nelements(od.out));
        ggml_backend_tensor_get(od.full, full.data(), 0, ggml_nbytes(od.full));
        ggml_backend_tensor_get(od.out, out.data(), 0, ggml_nbytes(od.out));
        worst = nmse(full.data(), static_cast<const float *>(oc.full->data), (int64_t) full.size());
        if (std::memcmp(out.data(), full.data() + (n_tokens - 1) * k_d, k_d * sizeof(float)) != 0 || !graph_finite(gd)) {
            worst = -1.0;  // GET_ROWS is a copy; a non-finite intermediate is a failure
        }
        for (const auto & caches : { std::pair<ggml_tensor *, ggml_tensor *>{ kc_d, kc_c }, { vc_d, vc_c } }) {
            std::vector<uint8_t> a(ggml_nbytes(caches.first)), b(a.size());
            ggml_backend_tensor_get(caches.first, a.data(), 0, a.size());
            ggml_backend_tensor_get(caches.second, b.data(), 0, b.size());
            const std::vector<float> fa = f16_to_f32(a), fb = f16_to_f32(b);
            worst = worst < 0.0 ? worst : std::max(worst, nmse(fa.data(), fb.data(), (int64_t) fa.size()));
        }
    }

    for (ggml_backend_buffer_t b : { bh, bwd, bwc, bgd, bgc }) {
        ggml_backend_buffer_free(b);
    }
    for (ggml_context * c : { ctx_h, ctx_wd, ctx_wc, ctx_gd, ctx_gc }) {
        ggml_free(c);
    }
    return worst;
}

// ROPE NEOX with a head size above 128, the shape that takes ggml-spacemit's RVV rotation (smaller NEOX heads use its
// scalar loop): -o ROPE has none (test-backend-ops reaches it only in RMS_NORM_MUL_ROPE, rows of 768 and 8192, whole
// rows rotated), here at Qwen3-like base and positions, whole and half rotated
static double rope_wide_nmse(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu, int n_dims) {
    ggml_init_params params = { ggml_tensor_overhead() * 8 + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context *   ctx_h  = ggml_init(params);
    ggml_context *   ctx_d  = ggml_init(params);
    ggml_tensor *    x      = ggml_new_tensor_3d(ctx_h, GGML_TYPE_F32, 256, 4, 3);
    ggml_tensor *    pos    = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, 3);
    ggml_tensor * y_ref = ggml_rope_ext(ctx_h, x, pos, nullptr, n_dims, GGML_ROPE_TYPE_NEOX, 0, 1e6f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * y     = ggml_rope_ext(ctx_d, x, pos, nullptr, n_dims, GGML_ROPE_TYPE_NEOX, 0, 1e6f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_backend_buffer_t bh = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_t bd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, ggml_backend_dev_buffer_type(dev));
    double                e  = -1.0;
    if (bh != nullptr && bd != nullptr) {
        const std::vector<float> xv = random_values(ggml_nelements(x), 9, 1.0f);
        const int32_t            pv[3] = { 0, 17, 4095 };
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(pos, pv, 0, sizeof(pv));
        ggml_cgraph * gd = ggml_new_graph(ctx_d);
        ggml_cgraph * gc = ggml_new_graph(ctx_h);
        ggml_build_forward_expand(gd, y);
        ggml_build_forward_expand(gc, y_ref);
        if (ggml_backend_dev_supports_op(dev, y) && ggml_backend_graph_compute(backend, gd) == GGML_STATUS_SUCCESS &&
            ggml_backend_graph_compute(cpu, gc) == GGML_STATUS_SUCCESS) {
            std::vector<float> out(ggml_nelements(y));
            ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
            e = nmse(out.data(), static_cast<const float *>(y_ref->data), (int64_t) out.size());
        }
    }
    ggml_backend_buffer_free(bh);
    ggml_backend_buffer_free(bd);
    ggml_free(ctx_h);
    ggml_free(ctx_d);
    return e;
}

// GET_ROWS with one index: the tiles split the columns, and with few columns some tiles get none (the ported kernel
// overran there before its flagos guard). llama.cpp's last-token rows (one row of 2560) and MoE router weights (one
// column) are such cases. Inputs in CPU buffers, output in ours.
static double get_rows_nmse(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu, int64_t nc) {
    ggml_init_params params = { ggml_tensor_overhead() * 8 + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context *   ctx_h  = ggml_init(params);
    ggml_context *   ctx_d  = ggml_init(params);
    ggml_tensor *    src    = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F32, nc, 8);
    ggml_tensor *    idx    = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, 1);
    ggml_tensor *    y_ref  = ggml_get_rows(ctx_h, src, idx);
    ggml_tensor *    y      = ggml_get_rows(ctx_d, src, idx);
    ggml_backend_buffer_t bh = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_t bd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, ggml_backend_dev_buffer_type(dev));
    double                e  = -1.0;
    if (bh != nullptr && bd != nullptr) {
        const std::vector<float> sv = random_values(ggml_nelements(src), 11, 1.0f);
        const int32_t            row = 5;
        ggml_backend_tensor_set(src, sv.data(), 0, ggml_nbytes(src));
        ggml_backend_tensor_set(idx, &row, 0, sizeof(row));
        ggml_cgraph * gd = ggml_new_graph(ctx_d);
        ggml_cgraph * gc = ggml_new_graph(ctx_h);
        ggml_build_forward_expand(gd, y);
        ggml_build_forward_expand(gc, y_ref);
        if (ggml_backend_dev_supports_op(dev, y) && ggml_backend_graph_compute(backend, gd) == GGML_STATUS_SUCCESS &&
            ggml_backend_graph_compute(cpu, gc) == GGML_STATUS_SUCCESS) {
            std::vector<float> out(ggml_nelements(y));
            ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
            e = nmse(out.data(), static_cast<const float *>(y_ref->data), (int64_t) out.size());
        }
    }
    ggml_backend_buffer_free(bh);
    ggml_backend_buffer_free(bd);
    ggml_free(ctx_h);
    ggml_free(ctx_d);
    return e;
}

// SET_ROWS of nr F32 rows into an F16 cache in a CPU buffer, as llama.cpp's KV writes; nc 1 is the transposed V
// cache written when flash attention is off. Returns the NMSE of the provider's cache against the CPU backend's.
static double set_rows_nmse(ggml_backend_t backend, ggml_backend_t cpu, int64_t nc, int64_t nr) {
    const int64_t    slots  = 2 * nr + 3;
    ggml_init_params params = { ggml_tensor_overhead() * 8 + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    ggml_tensor *    src    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nr);
    ggml_tensor *    idx    = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nr);
    ggml_tensor *    c_d    = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nc, slots);
    ggml_tensor *    c_c    = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nc, slots);
    ggml_tensor *    y_d    = ggml_set_rows(ctx, c_d, src, idx);  // views of the caches: no buffer of ours involved
    ggml_tensor *    y_c    = ggml_set_rows(ctx, c_c, src, idx);
    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    double                e = -1.0;
    if (b != nullptr) {
        ggml_backend_buffer_clear(b, 0);
        const std::vector<float> sv = random_values(ggml_nelements(src), 13, 1.0f);
        std::vector<int64_t>     iv(nr);
        for (int64_t r = 0; r < nr; r++) {
            iv[r] = (2 * r + 1) % slots;
        }
        ggml_backend_tensor_set(src, sv.data(), 0, ggml_nbytes(src));
        ggml_backend_tensor_set(idx, iv.data(), 0, ggml_nbytes(idx));
        ggml_cgraph * gd = ggml_new_graph(ctx);
        ggml_cgraph * gc = ggml_new_graph(ctx);
        ggml_build_forward_expand(gd, y_d);
        ggml_build_forward_expand(gc, y_c);
        if (ggml_backend_supports_op(backend, y_d) && ggml_backend_graph_compute(backend, gd) == GGML_STATUS_SUCCESS &&
            ggml_backend_graph_compute(cpu, gc) == GGML_STATUS_SUCCESS) {
            std::vector<uint8_t> a(ggml_nbytes(c_d)), r(ggml_nbytes(c_c));
            ggml_backend_tensor_get(c_d, a.data(), 0, a.size());
            ggml_backend_tensor_get(c_c, r.data(), 0, r.size());
            const std::vector<float> fa = f16_to_f32(a), fr = f16_to_f32(r);
            e = nmse(fa.data(), fr.data(), (int64_t) fa.size());
        }
    }
    ggml_backend_buffer_free(b);
    ggml_free(ctx);
    return e;
}

// a GET_ROWS with an index out of range fails the launch (the CPU would abort), as the first step and after an ADD,
// repeatedly (see check_ops); an unselected node (no compute flag, as ggml_build_forward_select leaves) is skipped
// as on the CPU, so its never-set indices do no harm. Returns the number of launches that behaved, of k_bad_runs * 2 + 1.
constexpr int k_bad_runs = 100;

static int bad_index_runs(ggml_backend_dev_t dev, ggml_backend_t backend) {
    ggml_init_params params = { ggml_tensor_overhead() * 8 + ggml_graph_overhead() * 4, nullptr, true };
    ggml_context *   ctx_h  = ggml_init(params);
    ggml_context *   ctx_d  = ggml_init(params);
    ggml_tensor *    src    = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F32, 64, 8);
    ggml_tensor *    idx    = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, 3);
    ggml_tensor *    first  = ggml_get_rows(ctx_d, src, idx);
    ggml_tensor *    later  = ggml_get_rows(ctx_d, ggml_add(ctx_d, src, src), idx);
    ggml_backend_buffer_t bh = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_t bd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, ggml_backend_dev_buffer_type(dev));
    int                   ok = 0;
    if (bh != nullptr && bd != nullptr) {
        const std::vector<float> sv       = random_values(ggml_nelements(src), 17, 1.0f);
        const int32_t            good[3]  = { 1, 7, 2 };
        const int32_t            bad[3]   = { 1, 9, 2 };  // 9 >= 8 rows; with 3 rows split over the tiles, tile 1 fails
        ggml_backend_tensor_set(src, sv.data(), 0, ggml_nbytes(src));
        ggml_backend_tensor_set(idx, good, 0, sizeof(good));
        ggml_cgraph * g_first = ggml_new_graph(ctx_d);
        ggml_cgraph * g_later = ggml_new_graph(ctx_d);
        ggml_build_forward_expand(g_first, first);
        ggml_build_forward_expand(g_later, later);
        // with valid indices both graphs run, so the failures below come from the kernel, not a refusal
        if (ggml_backend_graph_compute(backend, g_first) != GGML_STATUS_SUCCESS ||
            ggml_backend_graph_compute(backend, g_later) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "FAIL: GET_ROWS graphs with valid indices did not run\n");
            g_failures++;
            ok = -1;
        }
        ggml_backend_tensor_set(idx, bad, 0, sizeof(bad));
        for (int r = 0; r < k_bad_runs && ok >= 0; r++) {
            ok += ggml_backend_graph_compute(backend, g_first) == GGML_STATUS_FAILED;
            ok += ggml_backend_graph_compute(backend, g_later) == GGML_STATUS_FAILED;
        }
        first->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
        ok += ggml_backend_graph_compute(backend, g_first) == GGML_STATUS_SUCCESS;
    }
    ggml_backend_buffer_free(bh);
    ggml_backend_buffer_free(bd);
    ggml_free(ctx_h);
    ggml_free(ctx_d);
    return ok;
}

static int check_rows(ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);

    double worst_get = 0.0, worst_set = 0.0;
    for (int64_t nc : { 1, 10, 17, 41, 2560 }) {
        const double e = get_rows_nmse(dev, backend, cpu, nc);
        if (e < 0.0 || e > 0.0) {
            std::fprintf(stderr, "FAIL: GET_ROWS one row of %lld: %s %.3e\n", (long long) nc,
                         e < 0.0 ? "refused or failed" : "NMSE", e);
            g_failures++;
        }
        worst_get = std::max(worst_get, e);
    }
    for (const auto & [nc, nr] : { std::pair<int64_t, int64_t>{ 1024, 1 }, { 1024, 7 }, { 1024, 512 }, { 1, 512 } }) {
        const double e = set_rows_nmse(backend, cpu, nc, nr);
        if (e < 0.0 || e > 1e-7) {
            std::fprintf(stderr, "FAIL: SET_ROWS %lld rows of %lld into a CPU-buffer F16 cache: %s %.3e\n",
                         (long long) nr, (long long) nc, e < 0.0 ? "refused or failed" : "NMSE", e);
            g_failures++;
        }
        worst_set = std::max(worst_set, e);
    }

    const int bad_ok = bad_index_runs(dev, backend);
    if (bad_ok != 2 * k_bad_runs + 1) {
        std::fprintf(stderr, "FAIL: index out of range / unselected node: %d of %d launches as expected\n", bad_ok,
                     2 * k_bad_runs + 1);
        g_failures++;
    }

    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    std::printf("rows     GET_ROWS one row of 1-2560 columns vs CPU: max NMSE %.2e (bound 0); SET_ROWS into a CPU-buffer "
                "F16 cache (1-512 rows of 1024, 512 rows of 1): max NMSE %.2e; index out of range fails the launch and "
                "an unselected node is skipped: %d/%d\n",
                worst_get, worst_set, bad_ok, 2 * k_bad_runs + 1);
    return 0;
}

static int check_layer(ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 8);

    double worst = 0.0;
    for (ggml_type down : { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1 }) {
        for (int64_t n_tokens : { 1, 7 }) {
            const double e = layer_nmse(dev, backend, cpu, n_tokens, down);
            if (e < 0.0 || e > k_layer_max_nmse) {
                std::fprintf(stderr, "FAIL: layer with %lld tokens, down projection %s: %s %.3e\n", (long long) n_tokens,
                             ggml_type_name(down), e < 0.0 ? "refused or failed" : "NMSE", e);
                g_failures++;
            }
            worst = std::max(worst, e);
        }
    }
    // the whole head rotated, and half of it (the rest copied)
    const double e_rope_full = rope_wide_nmse(dev, backend, cpu, 256);
    const double e_rope_half = rope_wide_nmse(dev, backend, cpu, 128);
    const double e_rope      = e_rope_full < 0.0 || e_rope_half < 0.0 ? -1.0 : std::max(e_rope_full, e_rope_half);
    if (e_rope < 0.0 || e_rope > 1e-7) {
        std::fprintf(stderr, "FAIL: ROPE NEOX head 256 (n_dims 256, 128): %s %.3e\n", e_rope < 0.0 ? "refused or failed" : "NMSE",
                     e_rope);
        g_failures++;
    }

    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    std::printf("layer    Qwen3-like layer in one launch vs CPU (1 and 7 tokens, down projection Q4_0 and Q4_1, all rows; "
                "KV cache in a CPU buffer, weights in ours): max NMSE %.2e (bound %.0e); ROPE NEOX head 256 NMSE %.2e%s\n",
                worst, k_layer_max_nmse, e_rope, std::getenv("FLAGOS_SPACEMIT_TEST_REFERENCE") != nullptr ? ", reference kernels" : "");
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

    // Q4_0 matmul at Qwen3-4B's FFN shapes, and Q4_1 at its ffn_down shape (its first 4 layers): 1 row (GEMV), small
    // batches (path C), 128 rows (path A). Q6_K (run as Q8_0, against the CPU's Q6_K) at its output head, whose 1 row
    // is a generated token's: up to 16 rows and fewer iterations, as the head holds 15x an FFN matrix's weights
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu8    = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu8 != nullptr);
    ggml_backend_cpu_set_n_threads(cpu8, 8);
    struct bench_shape { ggml_type type; int64_t k, n; std::vector<int64_t> rows; };
    const std::vector<int64_t> ffn_rows = { 1, 4, 16, 64, 128 };
    for (const bench_shape & s : { bench_shape{ GGML_TYPE_Q4_0, 2560, 9728, ffn_rows },
                                   bench_shape{ GGML_TYPE_Q4_0, 9728, 2560, ffn_rows },
                                   bench_shape{ GGML_TYPE_Q4_1, 9728, 2560, ffn_rows },
                                   bench_shape{ GGML_TYPE_Q6_K, 2560, 151936, { 1, 4, 16 } } }) {
        const bool head = s.type == GGML_TYPE_Q6_K;
        mm_weight  w;
        REQUIRE(w.init(dev, s.k, s.n, 1, s.type, head));
        for (int64_t m : s.rows) {
            mm_run run;
            REQUIRE(run.init(dev, w, m, 1));
            bool         ok    = true;
            const int    iters = head ? 5 : m <= 16 ? 20 : 5;
            const double t_dev = time_us([&] { ok &= ggml_backend_graph_compute(backend, run.g_dev) == GGML_STATUS_SUCCESS; }, iters);
            const double t_cpu = time_us([&] { ggml_backend_graph_compute(cpu8, run.g_cpu); }, iters);
            CHECK(ok);
            const double flop = 2.0 * s.k * s.n * m;
            std::printf("bench    %s matmul %lldx%lld rows %3lld: provider %8.0f us (%6.1f GFLOP/s), cpu %8.0f us (%6.1f GFLOP/s)\n",
                        ggml_type_name(s.type), (long long) s.k, (long long) s.n, (long long) m, t_dev,
                        flop / t_dev / 1e3, t_cpu, flop / t_cpu / 1e3);
        }
    }
    ggml_backend_free(backend);
    ggml_backend_free(cpu8);
    return 0;
}

int main(int argc, char ** argv) {
    bool expect_device = false;
    bool expect_none   = false;
    bool full          = false;
    bool run_bench     = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--expect-device") == 0) {
            expect_device = true;
        } else if (std::strcmp(argv[i], "--expect-none") == 0) {
            expect_none = true;
        } else if (std::strcmp(argv[i], "--full") == 0) {
            full = true;
        } else if (std::strcmp(argv[i], "--bench") == 0) {
            run_bench = true;
        } else {
            std::fprintf(stderr, "usage: %s [--expect-device | --expect-none] [--full] [--bench]\n", argv[0]);
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

    if (check_registry(dev, global_index) != 0 || check_buffer(dev) != 0 || check_ops(dev) != 0 ||
        check_mul_mat(dev, full) != 0 || check_rows(dev) != 0 || check_layer(dev) != 0) {
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
