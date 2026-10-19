// flagos-check-spacemit: checks of the SpacemiT provider through the built ggml-flagos library.
// Usage: flagos-check-spacemit [--expect-device | --expect-none] [--full] [--bench | --bench-gemv | --bench-attn]
//   --expect-device  fail if the provider exposes no device (use on the K3)
//   --expect-none    fail if it exposes one (use with FLAGOS_SPACEMIT_DISABLE=1, or on another host)
//   --full           also check Q4_0 and Q4_1 matmuls at Qwen3-4B's shapes (up to 512 x 9728 x 2560), and the Q6_K
//                    output heads (run as Q8_0) at 128 rows besides 1 and 7 (up to 128 x 2560 x 151936); slow without
//                    the IME
//   --bench          also measure launch cost, per-node cost, bandwidth, Q4_0, Q4_1 and Q6_K (run as Q8_0) matmul time
//                    and flash attention time (Qwen3-4B's heads: a token at n_kv 256-8192, 128 tokens at n_kv 128
//                    and 4096, 512 at n_kv 512 and 4608) against the CPU, and the 1-row strategies of
//                    FLAGOS_SPACEMIT_GEMV at a Qwen3-4B token's shapes
//   --bench-gemv     of the benchmarks only the 1-row strategies
//   --bench-attn     of the benchmarks only flash attention (at one token also FLAGOS_SPACEMIT_ATTN=split and
//                    one-chunk, against the default)
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
#include <memory>
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

// sets an environment variable (nullptr: unsets it) and restores the caller's value when it goes out of scope, so that
// a setting given to the whole run (FLAGOS_SPACEMIT_TEST_TCM, FLAGOS_SPACEMIT_TEST_THREADS) holds for the later checks
class env_override {
  public:
    env_override(const char * name, const char * value) : name(name) {
        const char * old = std::getenv(name);
        had              = old != nullptr;
        saved            = had ? old : "";
        set_env(name, value);
    }

    ~env_override() { set_env(name.c_str(), had ? saved.c_str() : nullptr); }

    env_override(const env_override &)             = delete;
    env_override & operator=(const env_override &) = delete;

  private:
    std::string name;
    std::string saved;
    bool        had = false;
};

static std::vector<float> ramp(int64_t n, float scale, float offset) {
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; i++) {
        v[i] = scale * (float) (i % 1000) + offset;
    }
    return v;
}

// ACCEL, unless the test-only FLAGOS_SPACEMIT_TEST_DEVICE_TYPE names GPU or IGPU (structural-experiments.md E2)
static enum ggml_backend_dev_type expected_device_type() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_TEST_DEVICE_TYPE");
    if (v != nullptr && std::strcmp(v, "gpu") == 0) {
        return GGML_BACKEND_DEVICE_TYPE_GPU;
    }
    if (v != nullptr && std::strcmp(v, "igpu") == 0) {
        return GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static int check_registry(ggml_backend_dev_t dev, size_t global_index) {
    CHECK(ggml_backend_dev_type(dev) == expected_device_type());
    CHECK(ggml_backend_dev_description(dev) != nullptr);

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    CHECK(props.type == expected_device_type());
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

    const char * type = props.type == GGML_BACKEND_DEVICE_TYPE_GPU  ? "GPU" :
                        props.type == GGML_BACKEND_DEVICE_TYPE_IGPU ? "IGPU" :
                                                                      "ACCEL";
    std::printf("device   %s: %s, type %s, memory %.1f / %.1f GiB free\n", props.name, props.description, type,
                props.memory_free / 1073741824.0, props.memory_total / 1073741824.0);
    std::printf("flagos   provider %s (id 0x%016llx), engine cpu, %u AI cores, VLEN %u, target %s, runtime %s\n",
                info.provider.name, (unsigned long long) info.provider.id, info.profile.concurrency,
                info.profile.vector_bits, info.profile.target, info.profile.runtime);
    return 0;
}

// bytes of [base, base + size) on huge pages, from the kernel's count per mapping (AnonHugePages in /proc/self/smaps);
// -1 without that file (not Linux)
static double huge_page_bytes(const void * base, size_t size) {
    std::ifstream smaps("/proc/self/smaps");
    if (!smaps) {
        return -1.0;
    }
    const uintptr_t b = reinterpret_cast<uintptr_t>(base), e = b + size;
    std::string     line;
    bool            overlaps = false;
    double          bytes    = 0.0;
    while (std::getline(smaps, line)) {
        unsigned long long lo = 0, hi = 0;
        char               dash = 0;
        if (std::sscanf(line.c_str(), "%llx%c%llx", &lo, &dash, &hi) == 3 && dash == '-') {
            overlaps = lo < e && hi > b;  // a mapping's first line: its address range
        } else if (overlaps && line.rfind("AnonHugePages:", 0) == 0) {
            bytes += 1024.0 * std::atof(line.c_str() + std::strlen("AnonHugePages:"));
        }
    }
    return bytes;
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

    // where the system has them, the buffers lie on huge pages (flagos-spacemit.cpp, spacemit_map_huge): what the
    // kernel counts for a 64 MiB buffer once a clear has touched it
    const size_t          big_size = size_t(64) << 20;
    ggml_backend_buffer_t big      = ggml_backend_buft_alloc_buffer(buft, big_size);
    REQUIRE(big != nullptr);
    ggml_backend_buffer_clear(big, 1);
    const double huge = huge_page_bytes(ggml_backend_buffer_get_base(big), big_size);
    ggml_backend_buffer_free(big);
    std::string thp;
    std::getline(std::ifstream("/sys/kernel/mm/transparent_hugepage/enabled"), thp);
    const char * no_huge = std::getenv("FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES");
    if (huge < 0.0) {
        std::printf("info     huge pages: not counted (no /proc/self/smaps)\n");
    } else {
        std::printf("info     huge pages: %.0f of 64 MiB of a provider buffer%s (transparent huge pages: %s)\n", huge / 1048576.0,
                    no_huge != nullptr && std::strcmp(no_huge, "0") != 0 ? ", FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES set" : "",
                    thp.empty() ? "?" : thp.c_str());
    }
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
    // first step and in a later one: on the AI cores a failure seen at the wrong barrier would hang a launch. The step
    // counts within a launch, so the launches are set here: one per split, and one per step (structural-experiments.md
    // E3), where a failure in the first launch must fail the graph (a launch of one step has no step 1)
    constexpr int k_fail_runs = 100;
    int           fail_ok     = 0;
    for (const char * launch : { "per-split", "per-step" }) {
        env_override launch_switch("FLAGOS_SPACEMIT_TEST_LAUNCH", launch);
        for (const char * fail_node : { "0", "1" }) {
            if (std::strcmp(launch, "per-step") == 0 && std::strcmp(fail_node, "1") == 0) {
                continue;
            }
            set_env("FLAGOS_SPACEMIT_TEST_FAIL_NODE", fail_node);
            ggml_backend_t failing = ggml_backend_dev_init(dev, nullptr);
            set_env("FLAGOS_SPACEMIT_TEST_FAIL_NODE", nullptr);
            REQUIRE(failing != nullptr);
            for (int r = 0; r < k_fail_runs; r++) {
                fail_ok += ggml_backend_graph_compute(failing, chain) == GGML_STATUS_FAILED;
            }
            ggml_backend_free(failing);
        }
    }
    CHECK(fail_ok == 3 * k_fail_runs);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    std::printf("ops      ADD claimed exactly (with broadcast; not SUB, strided or F16); 3-node chain, %lld-element row and "
                "%lldx%lld ADD exact: checked\n",
                (long long) n_big, (long long) n_rows, (long long) n);
    std::printf("backend  empty and view-only graphs succeed; unclaimed op returns FAILED; forced kernel failure in step 0 "
                "and 1 (and in step 0 with one launch per step) returns FAILED %d/%d times: checked\n",
                fail_ok, 3 * k_fail_runs);
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

// the 1-row strategies of FLAGOS_SPACEMIT_GEMV (plan.md, "1-row matmul bandwidth"): the default (paired:32, unset), the
// others, and some on fewer tiles (@t); bench_gemv() times paired, staged and direct. For paired the shapes below also
// give a pair whose odd tile has one slab fewer (480 and 1056 columns), pairs without work (paired:256) and a tile
// without a partner (@5)
static const char * const k_gemv_strategies[] = { "paired",    "paired:64",  "paired:256", "paired@5",   "staged",
                                                  "staged:64", "staged:128", "staged:256", "staged@5",   "direct",
                                                  "direct:32", "direct:512", "direct:32@6" };

// every strategy gives the same bytes as reading the weights straight from DRAM: the same kernel on the same data, only
// the weights' path (DRAM or a copy in TCM) and the tile of each column differ. The output is filled with NaN first, so
// skipped columns show. The strategies need TCM: on a host without the AI cores the host executor gets a stand-in of
// the K3's 384 KiB (spine-runtime ignores the variable), and the check runs twice there: with the tiles one after
// another, which counts the pair waits, and on threads (FLAGOS_SPACEMIT_TEST_THREADS), where the waits wait
static int check_gemv_strategies(ggml_backend_dev_t dev, bool host) {
    env_override                gemv("FLAGOS_SPACEMIT_GEMV", nullptr);
    std::vector<ggml_backend_t> backends;
    for (const char * threads : { (const char *) nullptr, "1" }) {
        if (threads != nullptr && !host) {
            break;
        }
        env_override tcm("FLAGOS_SPACEMIT_TEST_TCM", "393216");
        env_override on_threads("FLAGOS_SPACEMIT_TEST_THREADS", threads);
        backends.push_back(ggml_backend_dev_init(dev, nullptr));
        REQUIRE(backends.back() != nullptr);
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 8);

    struct shape { ggml_type type; int64_t k, n; };
    const shape shapes[] = {
        { GGML_TYPE_Q4_0, 2560, 1024 },  // Qwen3-4B k/v projection
        { GGML_TYPE_Q4_0, 512, 480 },    // 15 groups of 32 columns: partial chunks, ranges and slabs
        { GGML_TYPE_Q4_0, 9728, 512 },   // a long row: a 128-column slab does not fit the TCM and shrinks to 64
        { GGML_TYPE_Q4_1, 9728, 512 },
        { GGML_TYPE_Q8_0, 2560, 2048 },
        { GGML_TYPE_Q6_K, 2560, 1056 },  // 33 groups
    };
    int    n_same = 0, n_runs = 0;
    double worst  = 0.0;
    for (const shape & s : shapes) {
        mm_weight w;
        REQUIRE(w.init(dev, s.k, s.n, (uint32_t) (3 * s.k + s.n), s.type, s.type == GGML_TYPE_Q6_K));
        mm_run run;
        REQUIRE(run.init(dev, w, 1, 9));
        set_env("FLAGOS_SPACEMIT_GEMV", "direct");
        const double e = run.compare(dev, backends[0], cpu);
        if (e < 0.0 || e > k_mul_mat_max_nmse) {
            std::fprintf(stderr, "FAIL: %s 1-row matmul k=%lld n=%lld with TCM: %s %.3e\n", ggml_type_name(s.type),
                         (long long) s.k, (long long) s.n, e < 0.0 ? "refused or failed" : "NMSE", e);
            g_failures++;
        }
        worst = std::max(worst, e);
        std::vector<float> ref(s.n), out(s.n);
        const std::vector<float> nan(s.n, NAN);
        ggml_backend_tensor_get(run.y, ref.data(), 0, ggml_nbytes(run.y));
        for (ggml_backend_t backend : backends) {
            const auto run_strategy = [&](const char * strategy) {
                set_env("FLAGOS_SPACEMIT_GEMV", strategy);
                ggml_backend_tensor_set(run.y, nan.data(), 0, ggml_nbytes(run.y));
                const bool ok = ggml_backend_graph_compute(backend, run.g_dev) == GGML_STATUS_SUCCESS;
                ggml_backend_tensor_get(run.y, out.data(), 0, ggml_nbytes(run.y));
                return ok && std::memcmp(out.data(), ref.data(), ggml_nbytes(run.y)) == 0;
            };
            for (const char * strategy : k_gemv_strategies) {
                if (!run_strategy(strategy) || (strategy == k_gemv_strategies[0] && !run_strategy(nullptr))) {
                    std::fprintf(stderr, "FAIL: %s 1-row matmul k=%lld n=%lld: strategy %s differs from direct%s\n",
                                 ggml_type_name(s.type), (long long) s.k, (long long) s.n, strategy,
                                 backend != backends[0] ? " (tiles on threads)" : "");
                    g_failures++;
                } else {
                    n_same++;
                }
                n_runs++;
            }
            // control: copy only stages the weights and never multiplies, so its output must differ; if it does not,
            // the strategies were not used (no TCM: path C instead) and the comparisons above prove nothing
            if (run_strategy("copy")) {
                std::fprintf(stderr, "FAIL: %s 1-row matmul k=%lld n=%lld: the 1-row strategies are not in use (no TCM?)\n",
                             ggml_type_name(s.type), (long long) s.k, (long long) s.n);
                g_failures++;
            }
        }
        set_env("FLAGOS_SPACEMIT_GEMV", nullptr);
    }

    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }
    ggml_backend_free(cpu);
    std::printf("gemv     1-row strategies (the default paired:32, paired:64/256, paired@5, staged:32/64/128/256, staged@5, "
                "direct:32/128/512, direct:32@6) at 6 shapes with TCM%s: %d/%d identical to direct (copy-only control "
                "differs); direct vs CPU max NMSE %.2e (bound %.0e)%s\n",
                host ? ", tiles in turn and on threads" : "", n_same, n_runs, worst, k_mul_mat_max_nmse,
                std::getenv("FLAGOS_SPACEMIT_TEST_REFERENCE") != nullptr ? ", reference kernels" : "");
    return 0;
}

//
// M2e.1: FLASH_ATTN_EXT over the KV cache, against the CPU backend
//

// FLASH_ATTN_EXT as llama.cpp builds it over caches that hold every KV head of a position in one cell
// (src/llama-kv-cache.cpp:1249-1300, V not transposed; src/llama-graph.cpp:2500-2545): views of the first n_kv cells
// and q [d, n_head, n_tokens], each permuted (0, 2, 1, 3); scale 1/sqrt(d); precision F32
static ggml_tensor * build_flash_attn(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k_cache, ggml_tensor * v_cache,
                                      ggml_tensor * mask, int64_t n_head_kv, int64_t n_kv, float max_bias) {
    const int64_t d = q->ne[0];
    ggml_tensor * k = ggml_view_4d(ctx, k_cache, d, n_head_kv, n_kv, 1, ggml_row_size(k_cache->type, d), k_cache->nb[1],
                                   ggml_nbytes(k_cache), 0);
    ggml_tensor * v = ggml_view_4d(ctx, v_cache, d, n_head_kv, n_kv, 1, ggml_row_size(v_cache->type, d), v_cache->nb[1],
                                   ggml_nbytes(v_cache), 0);
    ggml_tensor * a = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), ggml_permute(ctx, k, 0, 2, 1, 3),
                                          ggml_permute(ctx, v, 0, 2, 1, 3), mask, 1.0f / std::sqrt((float) d), max_bias,
                                          0.0f);
    ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
    return a;
}

static std::vector<ggml_fp16_t> random_f16(int64_t n, uint32_t seed, float scale) {
    const std::vector<float> f = random_values(n, seed, scale);
    std::vector<ggml_fp16_t> h(n);
    ggml_fp32_to_fp16_row(f.data(), h.data(), n);
    return h;
}

// n_head query and n_head_kv KV heads of d; n_tokens queries at positions n_past.. over the first n_kv cells, cell c
// holding position c (one sequence, as llama.cpp fills its cache)
struct attn_shape {
    int64_t n_head, n_head_kv, d, n_kv, n_tokens, n_past;
};

// the mask's cells that a token sees (positions up to its own) read 0 (causal), or minus their distance to the token
// as llama.cpp writes them for ALiBi (src/llama-kv-cache.cpp:1673); the others -inf. none: no mask, every cell seen
enum class attn_mask { none, causal, alibi };

// one FLASH_ATTN_EXT (build_flash_attn) on the provider, its output in the provider's buffer, and on the CPU backend,
// both reading the same inputs in a CPU buffer: caches of n_kv + 32 cells (cells beyond the view hold data too, as when
// the context is longer than n_kv), random K and V in [-1, 1); q in [-4, 4), which spreads the scores over about 1.3
// standard deviations at d 128; the mask F16 [n_kv, n_tokens, 1, 1], `visible` added to every cell a token sees
struct attn_run {
    ggml_context *        ctx_host = nullptr, * ctx_dev = nullptr;
    ggml_backend_buffer_t buf_host = nullptr, buf_dev = nullptr;
    ggml_tensor *         out = nullptr, * out_ref = nullptr;
    ggml_cgraph *         g_dev = nullptr, * g_cpu = nullptr;

    bool init(ggml_backend_dev_t dev, const attn_shape & s, attn_mask mode, float visible, float max_bias,
              uint32_t seed, float q_scale = 4.0f) {
        ggml_init_params params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        ctx_host                = ggml_init(params);
        ctx_dev                 = ggml_init(params);
        const int64_t cells     = s.n_kv + 32;
        ggml_tensor * k_cache   = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F16, s.d * s.n_head_kv, cells);
        ggml_tensor * v_cache   = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F16, s.d * s.n_head_kv, cells);
        ggml_tensor * q         = ggml_new_tensor_3d(ctx_host, GGML_TYPE_F32, s.d, s.n_head, s.n_tokens);
        ggml_tensor * mask =
            mode == attn_mask::none ? nullptr : ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, s.n_kv, s.n_tokens, 1, 1);
        out_ref  = build_flash_attn(ctx_host, q, k_cache, v_cache, mask, s.n_head_kv, s.n_kv, max_bias);
        out      = build_flash_attn(ctx_dev, q, k_cache, v_cache, mask, s.n_head_kv, s.n_kv, max_bias);
        buf_host = ggml_backend_alloc_ctx_tensors_from_buft(ctx_host, ggml_backend_cpu_buffer_type());
        buf_dev  = ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev, ggml_backend_dev_buffer_type(dev));
        if (buf_host == nullptr || buf_dev == nullptr) {
            return false;
        }
        const std::vector<float> qv = random_values(ggml_nelements(q), seed, q_scale);
        ggml_backend_tensor_set(q, qv.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(k_cache, random_f16(ggml_nelements(k_cache), seed + 1, 1.0f).data(), 0,
                                ggml_nbytes(k_cache));
        ggml_backend_tensor_set(v_cache, random_f16(ggml_nelements(v_cache), seed + 2, 1.0f).data(), 0,
                                ggml_nbytes(v_cache));
        if (mask != nullptr) {
            std::vector<ggml_fp16_t> m(ggml_nelements(mask));
            for (int64_t t = 0; t < s.n_tokens; t++) {
                const int64_t p = s.n_past + t;
                for (int64_t c = 0; c < s.n_kv; c++) {
                    const float value = mode == attn_mask::alibi ? visible - (float) (p - c) : visible;
                    m[t * s.n_kv + c] = ggml_fp32_to_fp16(c <= p ? value : -INFINITY);
                }
            }
            ggml_backend_tensor_set(mask, m.data(), 0, ggml_nbytes(mask));
        }
        g_dev = ggml_new_graph(ctx_dev);
        g_cpu = ggml_new_graph(ctx_host);
        ggml_build_forward_expand(g_dev, out);
        ggml_build_forward_expand(g_cpu, out_ref);
        return true;
    }

    ~attn_run() {
        ggml_backend_buffer_free(buf_host);
        ggml_backend_buffer_free(buf_dev);
        ggml_free(ctx_host);
        ggml_free(ctx_dev);
    }

    // the provider's output, filled with NaN first so rows that no tile writes show; false if it failed
    bool dev_output(ggml_backend_t backend, std::vector<float> & o) {
        o.assign(ggml_nelements(out), NAN);
        ggml_backend_tensor_set(out, o.data(), 0, ggml_nbytes(out));
        if (ggml_backend_graph_compute(backend, g_dev) != GGML_STATUS_SUCCESS) {
            return false;
        }
        ggml_backend_tensor_get(out, o.data(), 0, ggml_nbytes(out));
        return true;
    }

    // NMSE of the provider's output against the CPU's (+inf if not finite); -1 if the provider refused the op or
    // failed. worst_row: the largest NMSE of one row (ne[0] floats: one query head and token), since a fault in a few
    // rows of the thousands of a batch hardly moves the first. The output is filled with NaN first, so rows that no
    // tile writes show
    double compare(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu, double & worst_row) {
        worst_row = -1.0;
        if (!ggml_backend_dev_supports_op(dev, out)) {
            return -1.0;
        }
        std::vector<float> o(ggml_nelements(out), NAN);
        ggml_backend_tensor_set(out, o.data(), 0, ggml_nbytes(out));
        if (ggml_backend_graph_compute(backend, g_dev) != GGML_STATUS_SUCCESS ||
            ggml_backend_graph_compute(cpu, g_cpu) != GGML_STATUS_SUCCESS) {
            return -1.0;
        }
        ggml_backend_tensor_get(out, o.data(), 0, ggml_nbytes(out));
        const float * ref = static_cast<const float *>(out_ref->data);
        worst_row         = 0.0;
        for (int64_t r = 0; r < (int64_t) o.size(); r += out->ne[0]) {
            worst_row = std::max(worst_row, nmse(o.data() + r, ref + r, out->ne[0]));
        }
        return nmse(o.data(), ref, (int64_t) o.size());
    }
};

constexpr double k_attn_max_nmse = 5e-4;  // test-backend-ops' bound for FLASH_ATTN_EXT
// a row's own bound: rows of the reference reach 4e-7 in the tiled cases and 2e-5 for one token over 4096 cells (the
// CPU's 1-row kernel accumulates V in F16); a row given the wrong KV head or missing a KV tile 0.3 or more, while the
// whole output of a tiled case stayed below 5e-4
constexpr double k_attn_max_row_nmse = 5e-3;
// the tiled kernel's rows have their own, tighter bound. It accumulates V in F32 as the CPU's tiled kernel does, and
// its rows matched to 8.4e-7 on the K3 (4.5e-7 with the reference). A row that misses one visible position, as a wrong
// mask count would make it (plan.md M2e.3, L4), is off in NMSE by a median 1.9e-4 at 4096 positions (6e-5 at least)
// when the scores are flat (q's values scaled by 1, as in the case at depth), and by a median 1.3e-3 at 512
// (simulated); the general bound lets those pass. The other cases' peaked scores (q scaled by 4) can hide one position
// in a long row even from this bound (median 5e-6 at 4096), not in rows of up to a few hundred. From 64 tokens per
// call, with TCM, the provider runs the tiled kernel
constexpr double  k_attn_max_row_nmse_tiled = 1e-5;
constexpr int64_t k_attn_tiled_tokens       = 64;

// over the whole output and the worst row (attn_run::compare); all < 0: refused or failed
struct attn_error {
    double all = -1.0, row = -1.0;
};

// o against ref, over all n values and per row of row_len values
static attn_error attn_errors(const float * o, const float * ref, int64_t n, int64_t row_len) {
    attn_error e;
    e.all = nmse(o, ref, n);
    e.row = 0.0;
    for (int64_t r = 0; r < n; r += row_len) {
        e.row = std::max(e.row, nmse(o + r, ref + r, row_len));
    }
    return e;
}

static attn_error attn_nmse(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu, const attn_shape & s,
                            attn_mask mode, float visible, float max_bias, uint32_t seed, float q_scale = 4.0f) {
    attn_run   run;
    attn_error e;
    if (run.init(dev, s, mode, visible, max_bias, seed, q_scale)) {
        e.all = run.compare(dev, backend, cpu, e.row);
    }
    return e;
}

static void attn_report(const attn_error & e, const char * what, const attn_shape & s,
                        double row_bound = k_attn_max_row_nmse) {
    if (e.all < 0.0 || e.all > k_attn_max_nmse || e.row > row_bound) {
        std::fprintf(stderr,
                     "FAIL: attention %s (%lld query and %lld KV heads of %lld, n_kv %lld, %lld tokens after %lld): "
                     "%s %.3e (bound %.0e), worst row %.3e (bound %.0e)\n",
                     what, (long long) s.n_head, (long long) s.n_head_kv, (long long) s.d, (long long) s.n_kv,
                     (long long) s.n_tokens, (long long) s.n_past,
                     e.all < 0.0       ? "refused or failed" :
                     std::isinf(e.all) ? "output not finite" :
                                         "NMSE",
                     e.all, k_attn_max_nmse, e.row, row_bound);
        g_failures++;
    }
}

// FLASH_ATTN_EXT against the CPU backend, which runs the same algorithm (q rounded to F16, F32 scores, online softmax;
// V accumulated in F16 by one row, in F32 by its 64 x 64 tiles from 64 tokens), so the NMSE measures rounding. On the
// K3, with TCM, the cases select each of the provider's kernels: the 4-row (Qwen3-4B) and 2-row (0.6B) one-chunk
// kernels (one token), the 1-row one (7 tokens), the tiled kernel (99 and 128 tokens). 128 tokens from position 0 fill
// whole 64 x 64 tiles; 99 after 37 also take the tiled kernel's branches for rows that see none of a KV tile that other
// rows of their tile see (54 rows per head, 2 of them in mixed 4-row groups) and its 1-row tail (3 rows of a 35-row
// tile); 128 after 3968, with flat scores, give it rows that see about 4000 positions, where the tiled cases' tighter
// row bound catches a single position its mask counts miss. Each probe gates one of the faults fixed in the copied
// kernels (plan.md, M2e design): (a) every score far below 0, where rows whose maximum started at 0 came out as zeros
// (the one-chunk kernels accumulate V in F16, where a weight exp(s - 0) is 0 below about -17; below about -104 it is 0
// in F32 too, which -120 shows in the F32 reference); (b) no mask, which crashed; (c) ALiBi with GQA, where the grouped
// rows took the first row's slope
static int check_flash_attn(ggml_backend_dev_t dev) {
    // claimed exactly: llama.cpp's graph (here before allocation) at head sizes up to 128, without a mask too; not head
    // size 256, F32 or quantized K and V, or K and V of different types
    ggml_init_params params = { ggml_tensor_overhead() * 128, nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    REQUIRE(ctx != nullptr);
    const auto fa = [&](int64_t d, ggml_type tk, ggml_type tv, bool with_mask) {
        ggml_tensor * q  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, 32, 7);
        ggml_tensor * kc = ggml_new_tensor_2d(ctx, tk, d * 8, 256);
        ggml_tensor * vc = ggml_new_tensor_2d(ctx, tv, d * 8, 256);
        ggml_tensor * m  = with_mask ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 256, 7, 1, 1) : nullptr;
        return build_flash_attn(ctx, q, kc, vc, m, 8, 256, 0.0f);
    };
    CHECK(ggml_backend_dev_supports_op(dev, fa(128, GGML_TYPE_F16, GGML_TYPE_F16, true)));
    CHECK(ggml_backend_dev_supports_op(dev, fa(64, GGML_TYPE_F16, GGML_TYPE_F16, true)));
    CHECK(ggml_backend_dev_supports_op(dev, fa(128, GGML_TYPE_F16, GGML_TYPE_F16, false)));
    CHECK(!ggml_backend_dev_supports_op(dev, fa(256, GGML_TYPE_F16, GGML_TYPE_F16, true)));
    CHECK(!ggml_backend_dev_supports_op(dev, fa(128, GGML_TYPE_F32, GGML_TYPE_F32, true)));
    CHECK(!ggml_backend_dev_supports_op(dev, fa(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true)));
    CHECK(!ggml_backend_dev_supports_op(dev, fa(128, GGML_TYPE_F16, GGML_TYPE_Q8_0, true)));
    ggml_free(ctx);

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 8);

    struct attn_case { const char * what; attn_shape s; float q_scale = 4.0f; };
    const attn_case cases[] = {
        { "Qwen3-4B 1 token, 37 of 256 cells valid", { 32, 8, 128, 256, 1, 36 } },
        { "Qwen3-4B 1 token", { 32, 8, 128, 1024, 1, 1023 } },
        { "Qwen3-4B 1 token", { 32, 8, 128, 4096, 1, 4095 } },
        { "Qwen3-4B batch", { 32, 8, 128, 256, 7, 100 } },
        { "Qwen3-4B batch (tiled)", { 32, 8, 128, 256, 128, 0 } },
        { "Qwen3-4B batch (tiled), partly masked KV tiles", { 32, 8, 128, 256, 99, 37 } },
        { "Qwen3-0.6B 1 token, 300 of 512 cells valid", { 16, 8, 128, 512, 1, 299 } },
        { "Qwen3-0.6B batch (tiled)", { 16, 8, 128, 256, 128, 0 } },
        { "Qwen3-4B batch (tiled) at depth, flat scores", { 32, 8, 128, 4096, 128, 3968 }, 1.0f },
    };
    // runs and reports one case, returns its NMSE over the whole output (HUGE_VAL if refused or failed); worst_row and
    // worst_row_tiled over the cases and the probes, below and from k_attn_tiled_tokens tokens
    double     worst_row = 0.0, worst_row_tiled = 0.0;
    const auto attn      = [&](const char * what, const attn_shape & s, attn_mask mode, float visible, float max_bias,
                          uint32_t seed, float q_scale = 4.0f) {
        const attn_error e     = attn_nmse(dev, backend, cpu, s, mode, visible, max_bias, seed, q_scale);
        const bool       tiled = s.n_tokens >= k_attn_tiled_tokens;
        attn_report(e, what, s, tiled ? k_attn_max_row_nmse_tiled : k_attn_max_row_nmse);
        double & worst_of = tiled ? worst_row_tiled : worst_row;
        worst_of          = e.all < 0.0 ? HUGE_VAL : std::max(worst_of, e.row);
        return e.all < 0.0 ? HUGE_VAL : e.all;
    };
    // the cases and probes check the one-chunk and tiled kernels: one-token calls from n_kv 512 take the split by
    // default (M2e.3 round 2, checked below, where each run sets the switch itself)
    env_override one_chunk_switch("FLAGOS_SPACEMIT_ATTN", "one-chunk");
    double       worst = 0.0;
    uint32_t     seed  = 40;
    for (const attn_case & c : cases) {
        worst = std::max(worst, attn(c.what, c.s, attn_mask::causal, 0.0f, 0.0f, seed++, c.q_scale));
    }

    const attn_shape one_token = { 32, 8, 128, 256, 1, 36 };
    double           e_low  = 0.0;
    for (float visible : { -30.0f, -120.0f }) {
        e_low = std::max(e_low,
                         attn(visible > -100.0f ? "probe (a), every score far below 0 (mask -30 on the valid cells)" :
                                                  "probe (a), every score far below 0 (mask -120 on the valid cells)",
                              one_token, attn_mask::causal, visible, 0.0f, 60));
    }
    const attn_shape all_cells = { 32, 8, 128, 256, 1, 255 };
    const double     e_none    = attn("probe (b), no mask", all_cells, attn_mask::none, 0.0f, 0.0f, 61);
    const double e_alibi =
        attn("probe (c), ALiBi (max_bias 8) with GQA 4", one_token, attn_mask::alibi, 0.0f, 8.0f, 62);

    // M2e.3 round 2 (plan.md; the default since 2026-10-14): one-token calls split over the tiles, with the turns of
    // each pair (the default) and without them (FLAGOS_SPACEMIT_ATTN=split), against the CPU. V summed in F32 and the
    // tiles' partial results merged change the rounding, so the outputs differ from the one-chunk kernels'
    // (FLAGOS_SPACEMIT_ATTN=one-chunk; counted: that the split ran) and must stay within the bounds. Both settings deal
    // each tile the same blocks in the same order, so their outputs must be identical (a turn that skips or repeats a
    // block breaks that). The cases take its edges: a last block of 40 positions (n_kv 1000), blocks the mask hides
    // (4000 of 4096 valid), n_kv 8192, GQA 2 and 1, head size 64, no mask. Without the RVV kernels (the Mac,
    // FLAGOS_SPACEMIT_TEST_REFERENCE) the split's reference runs
    struct split_case {
        const char * what;
        attn_shape   s;
        attn_mask    mode;
    };
    const split_case split_cases[] = {
        { "Qwen3-4B 1 token", { 32, 8, 128, 1024, 1, 1023 }, attn_mask::causal },
        { "Qwen3-4B 1 token, n_kv 1000", { 32, 8, 128, 1000, 1, 999 }, attn_mask::causal },
        { "Qwen3-4B 1 token, 4000 of 4096 cells valid", { 32, 8, 128, 4096, 1, 3999 }, attn_mask::causal },
        { "Qwen3-4B 1 token", { 32, 8, 128, 8192, 1, 8191 }, attn_mask::causal },
        { "Qwen3-0.6B 1 token, 300 of 512 cells valid", { 16, 8, 128, 512, 1, 299 }, attn_mask::causal },
        { "GQA 1, 1 token", { 8, 8, 128, 1024, 1, 1023 }, attn_mask::causal },
        { "head size 64, 1 token", { 32, 8, 64, 1024, 1, 1023 }, attn_mask::causal },
        { "Qwen3-4B 1 token, no mask", { 32, 8, 128, 1024, 1, 1023 }, attn_mask::none },
    };
    // the switch's value per setting (nullptr: the default) and its name in the reports
    const char * const split_settings[][2] = { { nullptr, "the default, with the turns" },
                                               { "split", "FLAGOS_SPACEMIT_ATTN=split, without the turns" } };
    int                n_split = 0, n_split_differ = 0, n_turns_same = 0, n_launch_same = 0;
    double             e_split = 0.0, e_split_row = 0.0;
    for (const split_case & c : split_cases) {
        attn_run run;
        REQUIRE(run.init(dev, c.s, c.mode, 0.0f, 0.0f, seed++));
        REQUIRE(ggml_backend_graph_compute(cpu, run.g_cpu) == GGML_STATUS_SUCCESS);
        std::vector<float> one_chunk, split[2];
        {
            env_override attn_switch("FLAGOS_SPACEMIT_ATTN", "one-chunk");
            REQUIRE(run.dev_output(backend, one_chunk));
        }
        for (int k = 0; k < 2; k++) {
            env_override attn_switch("FLAGOS_SPACEMIT_ATTN", split_settings[k][0]);
            REQUIRE(run.dev_output(backend, split[k]));
            const int64_t     n    = (int64_t) split[k].size();
            const attn_error  e    = attn_errors(split[k].data(), static_cast<const float *>(run.out_ref->data), n,
                                                 run.out->ne[0]);
            const std::string what = std::string("split (") + split_settings[k][1] + "), " + c.what;
            attn_report(e, what.c_str(), c.s);
            e_split     = std::max(e_split, e.all);
            e_split_row = std::max(e_split_row, e.row);
            n_split_differ += std::memcmp(split[k].data(), one_chunk.data(), n * sizeof(float)) != 0;
            n_split++;
        }
        const bool same = std::memcmp(split[0].data(), split[1].data(), split[0].size() * sizeof(float)) == 0;
        if (!same) {
            std::fprintf(stderr, "FAIL: attention split, %s: the outputs with and without the turns differ\n", c.what);
            g_failures++;
        }
        n_turns_same += same;
        // structural-experiments.md E3: the default split with its two steps (the tiles' partial results, then their
        // merge, passed through the workspace) in launches of their own, against both in one launch
        std::vector<float> launched[2];
        for (int l = 0; l < 2; l++) {
            env_override attn_switch("FLAGOS_SPACEMIT_ATTN", nullptr);
            env_override launch_switch("FLAGOS_SPACEMIT_TEST_LAUNCH", l == 0 ? "per-split" : "per-step");
            REQUIRE(run.dev_output(backend, launched[l]));
        }
        const bool same_launch =
            std::memcmp(launched[0].data(), launched[1].data(), launched[0].size() * sizeof(float)) == 0;
        if (!same_launch) {
            std::fprintf(stderr, "FAIL: attention split, %s: one launch per step differs from one launch\n", c.what);
            g_failures++;
        }
        n_launch_same += same_launch;
    }

    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    std::printf(
        "attn     FLASH_ATTN_EXT vs CPU, one-chunk and tiled kernels (F16 KV cache in a CPU buffer, viewed as "
        "llama.cpp does; Qwen3-4B and 0.6B heads): %zu cases (1 token at n_kv 256-4096; 7, 99 and 128 tokens; 128 "
        "after 3968), max NMSE %.2e (bound %.0e), worst row %.2e (bound %.0e), of the tiled cases %.2e (bound %.0e); "
        "probes: every score far below 0 (mask -30, -120) %.2e, no mask %.2e, ALiBi with GQA 4 %.2e; claims exact "
        "(F16 K and V, head sizes to 128): checked%s\n",
        sizeof(cases) / sizeof(cases[0]), worst, k_attn_max_nmse, worst_row, k_attn_max_row_nmse, worst_row_tiled,
        k_attn_max_row_nmse_tiled, e_low, e_none, e_alibi,
        std::getenv("FLAGOS_SPACEMIT_TEST_REFERENCE") != nullptr ? ", reference kernels" : "");
    std::printf("attn     split over the tiles (the default; FLAGOS_SPACEMIT_ATTN=split without the turns) on %d "
                "one-token cases: max NMSE %.2e (bound %.0e), worst row %.2e (bound %.0e); %d/%d outputs differ from "
                "the one-chunk kernels' (the split ran); with and without the turns identical in %d/%zu cases; one "
                "launch per step (E3) identical in %d/%zu\n",
                n_split, e_split, k_attn_max_nmse, e_split_row, k_attn_max_row_nmse, n_split_differ, n_split,
                n_turns_same, sizeof(split_cases) / sizeof(split_cases[0]), n_launch_same,
                sizeof(split_cases) / sizeof(split_cases[0]));
    return 0;
}

//
// M2d: the ops of a layer together in one launch, against the CPU backend
//

struct layer_weights {
    ggml_tensor *attn_norm, *q_norm, *k_norm, *ffn_norm, *wq, *wk, *wv, *wo, *wg, *wu, *wd;
};

// a Qwen3-like layer (all dimensions small): d hidden, nh query and nkv key/value heads of hd, ff in the FFN; a KV
// cache of k_slots cells, the batch's positions from k_layer_pos on
constexpr int64_t k_d = 256, k_hd = 128, k_nh = 2, k_nkv = 1, k_ff = 512, k_slots = 64;
constexpr int32_t k_layer_pos = 40;

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
    ggml_tensor *full, *out;  // full: the layer output of every token; out: the last token's row
};

// norm, Q/K/V, Q and K norms, RoPE (NEOX), KV writes, attention, output projection, residual, FFN norm, SwiGLU FFN,
// residual, and the last token's row: every op of a Qwen3 layer. The attention is flash attention over every cell of
// the caches (build_flash_attn), so it reads the KV writes through views of the caches, not through the nodes: only
// the graph's order puts the writes first. Expanded into gf in llama.cpp's order (src/llama-graph.cpp:2770-2790): q, v
// and k, the KV writes, then the attention and the rest
static layer_out build_layer(ggml_context * ctx, ggml_cgraph * gf, const layer_weights & w, ggml_tensor * x,
                             ggml_tensor * pos, ggml_tensor * kv_idx, ggml_tensor * out_ids, ggml_tensor * k_cache,
                             ggml_tensor * v_cache, ggml_tensor * mask, int64_t n_tokens) {
    const float   eps = 1e-6f;
    ggml_tensor * h   = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w.attn_norm);
    ggml_tensor * q   = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w.wq, h), k_hd, k_nh, n_tokens);
    q                 = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps), w.q_norm);
    q = ggml_rope_ext(ctx, q, pos, nullptr, k_hd, GGML_ROPE_TYPE_NEOX, 0, 1e6f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w.wk, h), k_hd, k_nkv, n_tokens);
    k               = ggml_mul(ctx, ggml_rms_norm(ctx, k, eps), w.k_norm);
    k = ggml_rope_ext(ctx, k, pos, nullptr, k_hd, GGML_ROPE_TYPE_NEOX, 0, 1e6f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * v = ggml_mul_mat(ctx, w.wv, h);
    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, v);
    ggml_build_forward_expand(gf, k);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_cache, ggml_reshape_2d(ctx, k, k_hd * k_nkv, n_tokens), kv_idx));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_cache, v, kv_idx));

    layer_out     o;
    ggml_tensor * a = build_flash_attn(ctx, q, k_cache, v_cache, mask, k_nkv, k_slots, 0.0f);
    ggml_tensor * f = ggml_add(ctx, ggml_mul_mat(ctx, w.wo, ggml_reshape_2d(ctx, a, k_hd * k_nh, n_tokens)), x);
    ggml_tensor * g = ggml_mul(ctx, ggml_rms_norm(ctx, f, eps), w.ffn_norm);
    ggml_tensor * s = ggml_swiglu_split(ctx, ggml_mul_mat(ctx, w.wg, g), ggml_mul_mat(ctx, w.wu, g));
    o.full          = ggml_add(ctx, ggml_mul_mat(ctx, w.wd, s), f);
    o.out           = ggml_get_rows(ctx, o.full, out_ids);
    ggml_build_forward_expand(gf, o.out);
    return o;
}

static std::vector<float> f16_to_f32(const std::vector<uint8_t> & bytes) {
    std::vector<float> out(bytes.size() / sizeof(ggml_fp16_t));
    ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(bytes.data()), out.data(), (int64_t) out.size());
    return out;
}

// a layer chains three quantized matmuls, each quantizing its activations to 8 bits; on the Mac the whole output's NMSE
// against the CPU is about 5.4e-4 at 7 tokens (4.0e-4 with q standing in for the attention, before M2e.1), above a
// single matmul's bound. The FFN gives most of it: the attention's own output differs by 4e-5
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

// the layer on the provider, with the KV cache and the mask in a CPU buffer and the norm and quantized weights in the
// provider's buffer (where llama.cpp puts them), against the same layer on the CPU backend; returns the larger NMSE of
// the whole output and the caches, or -1 when refused, failed, or the last-token row is not an exact copy of the
// provider's output. The batch's tokens go to scattered cells (3 + 5t), the positions before it fill the other cells in
// order and the rest stay empty; every cell first holds random K and V (stale data in the empty ones), and the causal
// mask lets a token see the cells of its own position and the earlier ones
// kv_in_provider: the provider's KV cache in its own buffer, as llama.cpp places it for a GPU or IGPU device
// (structural-experiments.md E2), instead of a CPU buffer, as for ACCEL; full_out: the provider's layer output
static double layer_nmse(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu, int64_t n_tokens,
                         ggml_type down, bool kv_in_provider = false, std::vector<float> * full_out = nullptr) {
    ggml_init_params params = { ggml_tensor_overhead() * 128 + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context *   ctx_h  = ggml_init(params);  // inputs and caches: CPU buffer
    ggml_context *   ctx_wd = ggml_init(params);  // weights for the provider: its buffer
    ggml_context *   ctx_wc = ggml_init(params);  // weights for the CPU: CPU buffer
    ggml_context *   ctx_gd = ggml_init(params);  // the provider's graph
    ggml_context *   ctx_gc = ggml_init(params);  // the CPU's graph
    ggml_context *   ctx_kd = ggml_init(params);  // the provider's caches with kv_in_provider: its buffer

    ggml_tensor * x       = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F32, k_d, n_tokens);
    ggml_tensor * pos     = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, n_tokens);
    ggml_tensor * kv_idx  = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I64, n_tokens);
    ggml_tensor * out_ids = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, 1);
    ggml_tensor * kc_d    = ggml_new_tensor_2d(kv_in_provider ? ctx_kd : ctx_h, GGML_TYPE_F16, k_hd * k_nkv, k_slots);
    ggml_tensor * vc_d    = ggml_new_tensor_2d(kv_in_provider ? ctx_kd : ctx_h, GGML_TYPE_F16, k_hd * k_nkv, k_slots);
    ggml_tensor * kc_c    = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F16, k_hd * k_nkv, k_slots);
    ggml_tensor * vc_c    = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F16, k_hd * k_nkv, k_slots);
    ggml_tensor * mask    = ggml_new_tensor_4d(ctx_h, GGML_TYPE_F16, k_slots, n_tokens, 1, 1);
    const layer_weights wd = new_layer_weights(ctx_wd, down);
    const layer_weights wc = new_layer_weights(ctx_wc, down);

    ggml_backend_buffer_t bh  = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_t bwd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wd, ggml_backend_dev_buffer_type(dev));
    ggml_backend_buffer_t bwc = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wc, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_t bkd =
        kv_in_provider ? ggml_backend_alloc_ctx_tensors_from_buft(ctx_kd, ggml_backend_dev_buffer_type(dev)) : nullptr;
    if (bh == nullptr || bwd == nullptr || bwc == nullptr || (kv_in_provider && bkd == nullptr)) {
        return -1.0;
    }
    ggml_backend_buffer_clear(bh, 0);
    if (bkd != nullptr) {
        ggml_backend_buffer_clear(bkd, 0);
    }
    set_layer_weights(wd);
    copy_layer_weights(wd, wc);

    const std::vector<float> xv = random_values(k_d * n_tokens, 5, 1.0f);
    std::vector<int32_t>     pv(n_tokens);
    std::vector<int64_t>     iv(n_tokens);
    std::vector<int32_t>     cell_pos(k_slots, -1);  // -1: empty
    for (int64_t t = 0; t < n_tokens; t++) {
        pv[t]           = (int32_t) (k_layer_pos + t);
        iv[t]           = (3 + 5 * t) % k_slots;  // distinct cells
        cell_pos[iv[t]] = pv[t];
    }
    for (int32_t c = 0, p = 0; c < k_slots && p < k_layer_pos; c++) {
        if (cell_pos[c] < 0) {
            cell_pos[c] = p++;
        }
    }
    std::vector<ggml_fp16_t> mv(k_slots * n_tokens);
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t c = 0; c < k_slots; c++) {
            mv[t * k_slots + c] = ggml_fp32_to_fp16(cell_pos[c] >= 0 && cell_pos[c] <= pv[t] ? 0.0f : -INFINITY);
        }
    }
    const int32_t last = (int32_t) (n_tokens - 1);
    ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(pos, pv.data(), 0, ggml_nbytes(pos));
    ggml_backend_tensor_set(kv_idx, iv.data(), 0, ggml_nbytes(kv_idx));
    ggml_backend_tensor_set(out_ids, &last, 0, sizeof(last));
    ggml_backend_tensor_set(mask, mv.data(), 0, ggml_nbytes(mask));
    uint32_t cache_seed = 6;
    for (const auto & caches : { std::pair<ggml_tensor *, ggml_tensor *>{ kc_d, kc_c }, { vc_d, vc_c } }) {
        const std::vector<ggml_fp16_t> h = random_f16(ggml_nelements(caches.first), cache_seed++, 1.0f);
        ggml_backend_tensor_set(caches.first, h.data(), 0, ggml_nbytes(caches.first));
        ggml_backend_tensor_set(caches.second, h.data(), 0, ggml_nbytes(caches.second));
    }

    ggml_cgraph *         gd  = ggml_new_graph(ctx_gd);
    ggml_cgraph *         gc  = ggml_new_graph(ctx_gc);
    const layer_out       od  = build_layer(ctx_gd, gd, wd, x, pos, kv_idx, out_ids, kc_d, vc_d, mask, n_tokens);
    const layer_out       oc  = build_layer(ctx_gc, gc, wc, x, pos, kv_idx, out_ids, kc_c, vc_c, mask, n_tokens);
    ggml_backend_buffer_t bgd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_gd, ggml_backend_dev_buffer_type(dev));
    ggml_backend_buffer_t bgc = ggml_backend_alloc_ctx_tensors_from_buft(ctx_gc, ggml_backend_cpu_buffer_type());

    double worst = -1.0;
    if (bgd != nullptr && bgc != nullptr && ggml_backend_graph_compute(backend, gd) == GGML_STATUS_SUCCESS &&
        ggml_backend_graph_compute(cpu, gc) == GGML_STATUS_SUCCESS) {
        std::vector<float> full(ggml_nelements(od.full)), out(ggml_nelements(od.out));
        ggml_backend_tensor_get(od.full, full.data(), 0, ggml_nbytes(od.full));
        ggml_backend_tensor_get(od.out, out.data(), 0, ggml_nbytes(od.out));
        if (full_out != nullptr) {
            *full_out = full;
        }
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

    for (ggml_backend_buffer_t b : { bh, bwd, bwc, bgd, bgc, bkd }) {
        ggml_backend_buffer_free(b);
    }
    for (ggml_context * c : { ctx_h, ctx_wd, ctx_wc, ctx_gd, ctx_gc, ctx_kd }) {
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
    // structural-experiments.md E2: the KV cache in the provider's buffer, as llama.cpp places it for a GPU or IGPU
    // device: the same kernels, so the same bound
    double worst_kv = 0.0;
    for (int64_t n_tokens : { 1, 7 }) {
        const double e = layer_nmse(dev, backend, cpu, n_tokens, GGML_TYPE_Q4_0, true);
        if (e < 0.0 || e > k_layer_max_nmse) {
            std::fprintf(stderr, "FAIL: layer with %lld tokens, KV cache in the provider's buffer: %s %.3e\n",
                         (long long) n_tokens, e < 0.0 ? "refused or failed" : "NMSE", e);
            g_failures++;
        }
        worst_kv = std::max(worst_kv, e);
    }
    // E3: one launch per step (quantization and matmul, the norms, RoPE, the cache writes, attention, each launched on
    // its own) gives the same outputs as one launch for the layer; set both ways, whatever the run's setting
    int n_launch_same = 0, n_launch_cases = 0;
    for (ggml_type down : { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1 }) {
        for (int64_t n_tokens : { 1, 7 }) {
            std::vector<float> out[2];
            double             e[2];
            for (int l = 0; l < 2; l++) {
                env_override launch_switch("FLAGOS_SPACEMIT_TEST_LAUNCH", l == 0 ? "per-split" : "per-step");
                e[l] = layer_nmse(dev, backend, cpu, n_tokens, down, false, &out[l]);
            }
            const bool same = e[0] >= 0.0 && e[1] >= 0.0 && out[0].size() == out[1].size() &&
                              std::memcmp(out[0].data(), out[1].data(), out[0].size() * sizeof(float)) == 0;
            if (!same) {
                std::fprintf(stderr, "FAIL: layer with %lld tokens, down projection %s: one launch per step %s\n",
                             (long long) n_tokens, ggml_type_name(down),
                             e[0] < 0.0 || e[1] < 0.0 ? "refused or failed" : "differs from one per layer");
                g_failures++;
            }
            n_launch_same += same;
            n_launch_cases++;
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
    std::printf("layer    Qwen3-like layer with flash attention in one launch vs CPU (1 and 7 tokens after %d "
                "positions, down projection Q4_0 and Q4_1, all rows; KV cache of %lld cells and mask in a CPU buffer, "
                "weights in ours): max NMSE %.2e (bound %.0e); ROPE NEOX head 256 NMSE %.2e; KV cache in our buffer "
                "(E2) max NMSE %.2e; one launch per step (E3) identical in %d/%d cases%s\n",
                k_layer_pos, (long long) k_slots, worst, k_layer_max_nmse, e_rope, worst_kv, n_launch_same,
                n_launch_cases, std::getenv("FLAGOS_SPACEMIT_TEST_REFERENCE") != nullptr ? ", reference kernels" : "");
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

// bytes of one weight row in the provider's layout (a Q6_K weight is stored as Q8_0)
static double ime_row_bytes(ggml_type type, int64_t k) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return (double) (k / 32 * 18);
        case GGML_TYPE_Q4_1:
            return (double) (k / 32 * 19);
        default:
            return (double) (k / 32 * 34);
    }
}

// the 1-row strategies (FLAGOS_SPACEMIT_GEMV; plan.md, "1-row matmul bandwidth") at every 1-row matmul shape of a
// Qwen3-4B token, and the time each gives a whole token: the shapes weighted by how often a token runs them (q; k and v;
// attention output; gate and up; down in Q4_0 and in Q4_1; the head): the default (paired), the same without the turns
// (staged) and direct reads. As in the model, where a token reads each matrix once, every run reads its weights from
// DRAM: it takes the next of enough copies of the weight (32 MB together) to leave the caches (before 2026-10-11 k and v
// stayed cached). Every copy runs once before the timing, so that no run pays for faulting in its output. Each run is
// its own launch, so per token these include about 250 launches the model does not pay
static int bench_gemv(ggml_backend_dev_t dev) {
    env_override   gemv("FLAGOS_SPACEMIT_GEMV", nullptr);
    ggml_backend_t backend = nullptr;
    {
        env_override tcm("FLAGOS_SPACEMIT_TEST_TCM", "393216");  // the host executor's TCM (check_gemv_strategies)
        backend = ggml_backend_dev_init(dev, nullptr);
    }
    REQUIRE(backend != nullptr);
    struct gemv_shape { ggml_type type; int64_t k, n; int per_token; };
    const gemv_shape shapes[] = {
        { GGML_TYPE_Q4_0, 2560, 4096, 36 }, { GGML_TYPE_Q4_0, 2560, 1024, 72 }, { GGML_TYPE_Q4_0, 4096, 2560, 36 },
        { GGML_TYPE_Q4_0, 2560, 9728, 72 }, { GGML_TYPE_Q4_0, 9728, 2560, 32 }, { GGML_TYPE_Q4_1, 9728, 2560, 4 },
        { GGML_TYPE_Q6_K, 2560, 151936, 1 },
    };
    const std::vector<std::string> strategies = { "paired", "staged", "direct" };
    constexpr double    k_rotation_bytes = 32e6;
    std::vector<double> per_token(strategies.size(), 0.0);
    double              token_bytes = 0.0;
    for (const gemv_shape & s : shapes) {
        const double                            bytes  = (double) s.n * ime_row_bytes(s.type, s.k);
        const size_t                            copies = (size_t) std::min(32.0, std::ceil(k_rotation_bytes / bytes));
        const std::vector<uint8_t>              q      = quantize_weights(s.type, s.k, s.n, 1);
        std::vector<std::unique_ptr<mm_weight>> ws;
        std::vector<std::unique_ptr<mm_run>>    runs;
        for (size_t c = 0; c < copies; c++) {
            ws.push_back(std::make_unique<mm_weight>());
            REQUIRE(ws.back()->init_with(dev, s.type, s.k, s.n, q, s.type == GGML_TYPE_Q6_K));
            runs.push_back(std::make_unique<mm_run>());
            REQUIRE(runs.back()->init(dev, *ws.back(), 1, 1));
            CHECK(ggml_backend_graph_compute(backend, runs.back()->g_dev) == GGML_STATUS_SUCCESS);
        }
        token_bytes += bytes * s.per_token;
        for (size_t i = 0; i < strategies.size(); i++) {
            set_env("FLAGOS_SPACEMIT_GEMV", strategies[i].c_str());
            bool         ok   = true;
            size_t       next = 0;
            const double t    = time_us(
                [&] { ok &= ggml_backend_graph_compute(backend, runs[next++ % copies]->g_dev) == GGML_STATUS_SUCCESS; },
                s.n > 100000 ? 5 : 30);
            CHECK(ok);
            per_token[i] += t * s.per_token;
            std::printf("bench    gemv %-4s %5lldx%-6lld %-15s %8.0f us %6.1f GB/s\n", ggml_type_name(s.type),
                        (long long) s.k, (long long) s.n, strategies[i].c_str(), t, bytes / t / 1e3);
        }
        set_env("FLAGOS_SPACEMIT_GEMV", nullptr);
    }
    for (size_t i = 0; i < strategies.size(); i++) {
        std::printf("bench    gemv per Qwen3-4B token (%.0f MB of weights) %-15s %6.1f ms %6.1f GB/s\n", token_bytes / 1e6,
                    strategies[i].c_str(), per_token[i] / 1e3, token_bytes / per_token[i] / 1e3);
    }
    ggml_backend_free(backend);
    return 0;
}

// flash attention at Qwen3-4B's heads (32 query, 8 KV of 128; attn_run): a token at n_kv 256, 512, 1024, 4096 and
// 8192, every cell valid, and prompt batches, causal: 128 tokens at n_kv 128 and 4096, 512 at n_kv 512 and 4608 (depth
// 0 and 4096); the provider against the CPU backend (8 threads) on the same graph. As in the model, where each layer
// has its own cache, every call reads its cache from DRAM: it takes the next of enough copies (32 MB together, as
// bench_gemv). GB/s counts the n_kv cells' K and V bytes once; a token (or batch) is 36 layers' calls, each its own
// launch here. At one token the default (from n_kv 512 the split over the tiles with the turns; plan.md M2e.3, round 2)
// also against FLAGOS_SPACEMIT_ATTN=split (the split without the turns) and one-chunk (the kernels before round 2)
static int bench_attn(ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 8);
    struct attn_bench { int64_t n_kv, n_tokens; int iterations; };
    for (const attn_bench & b : { attn_bench{ 256, 1, 30 }, { 512, 1, 25 }, { 1024, 1, 20 }, { 4096, 1, 10 },
                                  { 8192, 1, 5 }, { 128, 128, 10 }, { 4096, 128, 2 }, { 512, 512, 4 },
                                  { 4608, 512, 2 } }) {
        const attn_shape s        = { 32, 8, 128, b.n_kv, b.n_tokens, b.n_kv - b.n_tokens };
        const double     kv_bytes = 2.0 * (double) (s.n_kv * s.n_head_kv * s.d) * sizeof(ggml_fp16_t);
        const size_t     copies   = (size_t) std::min(32.0, std::ceil(32e6 / kv_bytes));
        std::vector<std::unique_ptr<attn_run>> runs;
        for (size_t c = 0; c < copies; c++) {
            runs.push_back(std::make_unique<attn_run>());
            REQUIRE(runs.back()->init(dev, s, attn_mask::causal, 0.0f, 0.0f, (uint32_t) c));
            CHECK(ggml_backend_graph_compute(backend, runs.back()->g_dev) == GGML_STATUS_SUCCESS);
        }
        bool       ok   = true;
        size_t     next = 0;
        const auto dev_us = [&](const char * attn) {
            env_override attn_switch("FLAGOS_SPACEMIT_ATTN", attn);
            return time_us(
                [&] { ok &= ggml_backend_graph_compute(backend, runs[next++ % copies]->g_dev) == GGML_STATUS_SUCCESS; },
                b.iterations);
        };
        // at one token the default against the split without the turns and the one-chunk kernels (M2e.3 round 2):
        // each takes two turns and keeps the faster
        static const char * const other_settings[] = { "split", "one-chunk" };
        double                    t_dev            = dev_us(nullptr);
        double                    t_other[2]       = { HUGE_VAL, HUGE_VAL };
        if (b.n_tokens == 1) {
            for (int turn = 0; turn < 2; turn++) {
                for (int k = 0; k < 2; k++) {
                    t_other[k] = std::min(t_other[k], dev_us(other_settings[k]));
                }
                t_dev = std::min(t_dev, dev_us(nullptr));
            }
        }
        const double t_cpu = time_us(
            [&] { ok &= ggml_backend_graph_compute(cpu, runs[next++ % copies]->g_cpu) == GGML_STATUS_SUCCESS; },
            b.iterations);
        CHECK(ok);
        if (b.n_tokens == 1) {
            std::printf("bench    attn 1 token n_kv %4lld: provider %7.0f us (%6.1f ms per token, %5.1f GB/s of K and "
                        "V), cpu %7.0f us (%6.1f ms, %5.1f GB/s)\n",
                        (long long) s.n_kv, t_dev, 36 * t_dev / 1e3, kv_bytes / t_dev / 1e3, t_cpu, 36 * t_cpu / 1e3,
                        kv_bytes / t_cpu / 1e3);
            std::printf("bench    attn 1 token n_kv %4lld, FLAGOS_SPACEMIT_ATTN: split %.0f us (%+.1f%%, %.1f GB/s); "
                        "one-chunk %.0f us (%+.1f%%, %.1f GB/s)%s\n",
                        (long long) s.n_kv, t_other[0], 100.0 * (t_other[0] / t_dev - 1.0), kv_bytes / t_other[0] / 1e3,
                        t_other[1], 100.0 * (t_other[1] / t_dev - 1.0), kv_bytes / t_other[1] / 1e3,
                        s.n_kv < 512 ? " (below the split's 512 positions: all three ran the one-chunk kernels)" : "");
        } else {
            std::printf("bench    attn %lld tokens n_kv %4lld: provider %8.0f us (%7.1f ms per batch), cpu %8.0f us "
                        "(%7.1f ms)\n",
                        (long long) s.n_tokens, (long long) s.n_kv, t_dev, 36 * t_dev / 1e3, t_cpu, 36 * t_cpu / 1e3);
        }
    }
    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    return 0;
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
    return bench_attn(dev) != 0 ? 1 : bench_gemv(dev);
}

int main(int argc, char ** argv) {
    bool expect_device = false;
    bool expect_none   = false;
    bool full          = false;
    bool run_bench     = false;
    bool run_gemv      = false;
    bool run_attn      = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--expect-device") == 0) {
            expect_device = true;
        } else if (std::strcmp(argv[i], "--expect-none") == 0) {
            expect_none = true;
        } else if (std::strcmp(argv[i], "--full") == 0) {
            full = true;
        } else if (std::strcmp(argv[i], "--bench") == 0) {
            run_bench = true;
        } else if (std::strcmp(argv[i], "--bench-gemv") == 0) {
            run_gemv = true;
        } else if (std::strcmp(argv[i], "--bench-attn") == 0) {
            run_attn = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--expect-device | --expect-none] [--full] [--bench | --bench-gemv | --bench-attn]\n",
                         argv[0]);
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

    // without spine-runtime the provider runs on the host executor (simulated device)
    flagos_device_info info;
    const bool         host = flagos_registry_get_device_info(global_index, &info) && info.profile.runtime != nullptr &&
                      std::strcmp(info.profile.runtime, "none") == 0;
    if (check_registry(dev, global_index) != 0 || check_buffer(dev) != 0 || check_ops(dev) != 0 ||
        check_mul_mat(dev, full) != 0 || check_gemv_strategies(dev, host) != 0 || check_rows(dev) != 0 ||
        check_flash_attn(dev) != 0 || check_layer(dev) != 0) {
        return 1;
    }
    if (run_bench ? bench(dev) != 0 : (run_gemv && bench_gemv(dev) != 0) || (run_attn && bench_attn(dev) != 0)) {
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "flagos-check-spacemit: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("flagos-check-spacemit: all checks passed\n");
    return 0;
}
