// flagos-check-spacemit: checks of the SpacemiT provider through the built ggml-flagos library.
// Usage: flagos-check-spacemit [--expect-device | --expect-none] [--full] [--bench | --bench-attn | --bench-mm]
//   --expect-device  fail if the provider exposes no device (use on the K3)
//   --expect-none    fail if it exposes one (use with FLAGOS_SPACEMIT_DISABLE=1, or on another host)
//   --full           also check Q4_0 and Q4_1 matmuls at Qwen3-4B's shapes (up to 512 x 9728 x 2560), and the Q6_K
//                    output heads (run as Q8_0) at 128 rows besides 1 and 7 (up to 128 x 2560 x 151936); slow without
//                    the IME
//   --bench          also measure launch cost, per-node cost, bandwidth, Q4_0, Q4_1 and Q6_K (run as Q8_0) matmul time
//                    and flash attention time (Qwen3-4B's heads: a token at n_kv 256-8192, 128 tokens at n_kv 128
//                    and 4096, 512 at n_kv 512 and 4608) against the CPU, and the default matmul setting (fit,staged)
//                    against split at the row counts of --bench-mm
//   --bench-attn     of the benchmarks only flash attention (at one token also FLAGOS_SPACEMIT_ATTN=split and
//                    one-chunk, against the default; at 1, 2, 3, 4 and 8 tokens FLAGOS_SPACEMIT_ATTN_KVH=off
//                    against 1-8)
//   --bench-mm       of the benchmarks only the matmuls of a Qwen3-4B batch of 2-512 rows under the default
//                    (fit,staged) against FLAGOS_SPACEMIT_MM=split (the default until 2026-10-25)
// Without a flag, a host without the device skips the device checks. Q4_1 weights are checked as the provider stores
// them: requantized to Q8_0.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-flagos.h"
#include "ggml.h"
#include "../ggml-impl.h"  // ggml_cgraph's uid and ggml_graph_view (the plan cache's probes)
#include "providers/spacemit/core/flagos-spacemit-diagnostics.h"  // the diagnostics under the deck's names (S1 step 5)
#include "providers/spacemit/core/flagos-spacemit-loader.h"  // the plugin loader and its results (S1 step 1)
#include "providers/spacemit/core/flagos-spacemit-plan.h"    // the plan cache's counters (S1 step 3)
#include "providers/spacemit/flagos-spacemit-api.h"   // spacemit_device_plugin

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#    include <unistd.h>  // dup, dup2 (the counters' lines read back from stderr)
#endif

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

// the plugin's test names this program calls (plan.md "S1 design" A2), looked up once from the device's table
// (bind_plugin_names); nullptr with no device, where only the "device checks skipped" path runs
static spacemit_steps_take_fn     g_steps_take     = nullptr;
static spacemit_mm_paths_take_fn  g_mm_paths_take  = nullptr;
static spacemit_mm_tail_take_fn   g_mm_tail_take   = nullptr;
static spacemit_switches_reset_fn g_switches_reset = nullptr;

static int bind_plugin_names(ggml_backend_dev_t dev) {
    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    g_steps_take     = reinterpret_cast<spacemit_steps_take_fn>(table->get_proc_address("spacemit_steps_take"));
    g_mm_paths_take  = reinterpret_cast<spacemit_mm_paths_take_fn>(table->get_proc_address("spacemit_mm_paths_take"));
    g_mm_tail_take   = reinterpret_cast<spacemit_mm_tail_take_fn>(table->get_proc_address("spacemit_mm_tail_take"));
    g_switches_reset = reinterpret_cast<spacemit_switches_reset_fn>(table->get_proc_address("spacemit_switches_reset"));
    REQUIRE(g_steps_take != nullptr && g_mm_paths_take != nullptr && g_mm_tail_take != nullptr &&
            g_switches_reset != nullptr);
    return 0;
}

// sets or unsets a switch for this process. The kernels parse their switches once (spacemit_cached_switch), so every
// change tells them to parse again (spacemit_switches_reset, by name from the plugin; no device, nothing to tell), and
// so do the core's parsed-once diagnostics (GGML_FLAGOS_DISABLE_OPS, TRACE_LAUNCH and the replaced switches' warnings:
// spacemit_diagnostics_reset). Every switch change in this program goes through here, directly or through env_override
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
    if (g_switches_reset != nullptr) {
        g_switches_reset();
    }
    spacemit_diagnostics_reset();
}

// sets an environment variable (nullptr: unsets it) and restores the caller's value when it goes out of scope, so that
// a setting given to the whole run (FLAGOS_SPACEMIT_TEST_TCM, FLAGOS_SPACEMIT_TEST_THREADS) holds for the later checks
// sets a switch for a scope and restores it after (set_env resets the kernels' cached switches both times)
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

// ", reference kernels" when GGML_FLAGOS_KERNEL_VARIANTS=0 asks for them (flagos-spacemit-kernels.cpp), else ""
static const char * reference_label() {
    const char * v = std::getenv("GGML_FLAGOS_KERNEL_VARIANTS");
    return v != nullptr && std::strcmp(v, "0") == 0 ? ", reference kernels" : "";
}

static std::vector<float> ramp(int64_t n, float scale, float offset) {
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; i++) {
        v[i] = scale * (float) (i % 1000) + offset;
    }
    return v;
}

// IGPU (D3), unless the test-only FLAGOS_SPACEMIT_TEST_DEVICE_TYPE names ACCEL or GPU (structural-experiments.md E1, E2)
static enum ggml_backend_dev_type expected_device_type() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_TEST_DEVICE_TYPE");
    if (v != nullptr && std::strcmp(v, "accel") == 0) {
        return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    }
    if (v != nullptr && std::strcmp(v, "gpu") == 0) {
        return GGML_BACKEND_DEVICE_TYPE_GPU;
    }
    return GGML_BACKEND_DEVICE_TYPE_IGPU;
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

static bool readable_file(const char * path) {
    FILE * f = std::fopen(path, "rb");
    if (f == nullptr) {
        return false;
    }
    std::fclose(f);
    return true;
}

static void loader_log(int level, const char * text) {  // the plugin logs through GGML_LOG_* in this build
    (void) level;
    (void) text;
}

// S1 step 1 (plan.md "S1 step 1 spec" §7): the plugin the core loaded through the loader, its table, the names that
// are not contract, and the loader's negative matrix on the fake plugins of tests/plugins
static int check_loader(ggml_backend_dev_t dev, size_t global_index) {
    const spacemit_plugin * loaded = spacemit_device_plugin(dev);
    REQUIRE(loaded != nullptr);
    const char * path = spacemit_plugin_path(loaded);
    REQUIRE(path != nullptr && *path != '\0');
    CHECK(readable_file(path));

    spacemit_host_v1 host = {};
    host.struct_size      = sizeof(host);
    host.log              = loader_log;
    host.plugin_dir       = FLAGOS_SPACEMIT_TEST_PLUGIN_DIR;

    // the same library loads again: one dlopen more on an already loaded library
    spacemit_plugin * again  = nullptr;
    std::string       detail;
    REQUIRE(spacemit_plugin_load(path, host, spacemit_required_kernels, spacemit_n_required_kernels, &again,
                                 &detail) == spacemit_load_result::ok);
    const spacemit_plugin_v1 * table = spacemit_plugin_table(again);
    REQUIRE(table != nullptr);
    CHECK(table->api_version == FLAGOS_SPACEMIT_PLUGIN_API_VERSION);
    CHECK(table->struct_size == sizeof(spacemit_plugin_v1));
    CHECK(table->register_host_ptr == nullptr && table->tensor_prepare == nullptr);  // zero copy: not built

    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    REQUIRE(kernels != nullptr);
    CHECK(n_kernels == spacemit_n_required_kernels);
    for (size_t i = 0; i < spacemit_n_required_kernels; i++) {
        bool found = false;
        for (uint32_t k = 0; k < n_kernels; k++) {
            found = found || (kernels[k].name != nullptr && kernels[k].fn != nullptr &&
                              std::strcmp(kernels[k].name, spacemit_required_kernels[i]) == 0);
        }
        CHECK(found);
    }

    // what is not contract comes by name (plan.md "S1 design" A2): every name answers, and the behaviour behind each is
    // checked where it is used
    const char * names[] = { "spacemit_steps_take", "spacemit_mm_paths_take", "spacemit_mm_tail_take",
                             "spacemit_run_stats_attach", "spacemit_mul_mat_same_activations",
                             "spacemit_switches_reset", "spacemit_switches_generation" };
    size_t       n_names = 0;
    for (const char * name : names) {
        n_names += table->get_proc_address(name) != nullptr ? 1 : 0;
    }
    CHECK(n_names == sizeof(names) / sizeof(names[0]));
    CHECK(table->get_proc_address("nonexistent") == nullptr);

    // the device facts the core reports come from the plugin's props
    size_t                   tcm  = 0;
    spacemit_plugin_device * pdev = nullptr;
    spacemit_plugin_device * second = nullptr;
    CHECK(table->device_open(0, &pdev) == SPACEMIT_STATUS_SUCCESS);
    CHECK(table->device_open(1, &second) == SPACEMIT_STATUS_INVALID_ARGUMENT);
    CHECK(second == nullptr);
    if (pdev != nullptr) {
        spacemit_plugin_device_props props = {};
        props.struct_size                  = sizeof(props);
        CHECK(table->device_props(pdev, &props) == SPACEMIT_STATUS_SUCCESS);
        CHECK(props.n_tiles > 0 && props.vector_bits == 1024);
        CHECK(props.target != nullptr && props.runtime != nullptr && props.description != nullptr);
        tcm = props.tcm_bytes_per_tile;
        flagos_device_info info;
        CHECK(flagos_registry_get_device_info(global_index, &info) && info.profile.runtime != nullptr &&
              std::strcmp(info.profile.runtime, props.runtime) == 0);
        CHECK(info.profile.concurrency == props.n_tiles);
        props.struct_size = sizeof(props) - 1;  // a props struct the plugin cannot fill
        CHECK(table->device_props(pdev, &props) == SPACEMIT_STATUS_INVALID_ARGUMENT);
        table->device_close(pdev);
    }
    const uint32_t api = table->api_version, struct_size = table->struct_size;
    spacemit_plugin_unload(again);

    // the negative matrix: each fake plugin's one fault, and a path that is not there
    const struct {
        const char *         name;
        spacemit_load_result want;
    } cases[] = {
        { "bad-version",    spacemit_load_result::bad_version    },
        { "short-struct",   spacemit_load_result::short_struct   },
        { "null-member",    spacemit_load_result::null_member    },
        { "missing-kernel", spacemit_load_result::missing_kernel },
        { "no-device",      spacemit_load_result::no_device      },
        { "no-entry",       spacemit_load_result::no_entry       },
        { "absent",         spacemit_load_result::not_found      },
    };
    for (const auto & c : cases) {
        const std::string     bad_path = std::string(FLAGOS_SPACEMIT_TEST_PLUGIN_DIR) + "/" + c.name +
                                     FLAGOS_SPACEMIT_TEST_PLUGIN_SUFFIX;
        spacemit_plugin *    bad = nullptr;
        std::string          why;
        const spacemit_load_result got =
            spacemit_plugin_load(bad_path.c_str(), host, spacemit_required_kernels, spacemit_n_required_kernels, &bad,
                                 &why);
        if (got != c.want) {
            std::fprintf(stderr, "FAIL: %s gave result %d, expected %d (%s)\n", c.name, (int) got, (int) c.want,
                         why.c_str());
            g_failures++;
        }
        CHECK(bad == nullptr);
        CHECK(!why.empty());
        if (bad != nullptr) {
            spacemit_plugin_unload(bad);
        }
    }

    // GGML_FLAGOS_PROVIDER_PATH names the path and has no fallback; it is read at every call
    {
        const std::string wanted = std::string(FLAGOS_SPACEMIT_TEST_PLUGIN_DIR) + "/absent" +
                                   FLAGOS_SPACEMIT_TEST_PLUGIN_SUFFIX;
        env_override      override_path("GGML_FLAGOS_PROVIDER_PATH", wanted.c_str());
        CHECK(spacemit_plugin_default_path() == wanted);
    }

    std::printf("loader   %s: api %u, struct_size %u, %u kernels, %zu names by get_proc_address, tcm %zu per tile\n",
                path, api, struct_size, n_kernels, n_names, tcm);
    return 0;
}

// the attention checks' builder, below: check_contracts builds its FLASH_ATTN_EXT nodes the same way
static ggml_tensor * build_flash_attn(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k_cache, ggml_tensor * v_cache,
                                      ggml_tensor * mask, int64_t n_head_kv, int64_t n_kv, float max_bias,
                                      float logit_softcap = 0.0f, ggml_tensor * sinks = nullptr);

// the real table, for the kernels() below: check_contracts sets it before the one call that needs it
static const spacemit_plugin_v1 * g_real_table = nullptr;

// the real table's kernels except mul_mat.gemm, built once from g_real_table: a kernels() for a copy of the real
// table that lacks one kernel the core's contracts name
static const spacemit_plugin_kernel * kernels_without_gemm(uint32_t * count) {
    static std::vector<spacemit_plugin_kernel> kept;
    if (kept.empty() && g_real_table != nullptr) {
        uint32_t                       n   = 0;
        const spacemit_plugin_kernel * all = g_real_table->kernels(&n);
        for (uint32_t k = 0; all != nullptr && k < n; k++) {
            if (all[k].name == nullptr || std::strcmp(all[k].name, "mul_mat.gemm") != 0) {
                kept.push_back(all[k]);
            }
        }
    }
    if (count != nullptr) {
        *count = (uint32_t) kept.size();
    }
    return kept.data();
}

// S1 step 2 (plan.md "S1 step 2 spec" §7): the core's contracts bound to the plugin's kernels by name, and the answers
// spacemit_find_op gives through them - the variant choice of attention by the kernels' supports, the layout rule by
// the table, the workspace by the entries
static int check_contracts(ggml_backend_dev_t dev) {
    const spacemit_plugin * loaded = spacemit_device_plugin(dev);
    REQUIRE(loaded != nullptr);
    const spacemit_plugin_v1 * table = spacemit_plugin_table(loaded);
    REQUIRE(table != nullptr);

    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    REQUIRE(kernels != nullptr && n_kernels > 0);
    const auto entry_named = [&](const char * name) -> const spacemit_plugin_kernel * {
        for (uint32_t k = 0; k < n_kernels; k++) {
            if (kernels[k].name != nullptr && std::strcmp(kernels[k].name, name) == 0) {
                return &kernels[k];
            }
        }
        return nullptr;
    };

    // every contract's steps are the entries of their names
    spacemit_ops ops;
    std::string  detail;
    REQUIRE(spacemit_ops_bind(table, &ops, &detail));
    CHECK(ops.table == table);
    CHECK(ops.n_ops == 13);
    for (uint32_t i = 0; i < ops.n_ops; i++) {
        REQUIRE(ops.ops[i].n_steps >= 1 && ops.ops[i].n_steps <= SPACEMIT_MAX_OP_STEPS);
        for (uint32_t s = 0; s < ops.ops[i].n_steps; s++) {
            const spacemit_plugin_kernel * step = ops.ops[i].steps[s];
            REQUIRE(step != nullptr && step->name != nullptr && step->fn != nullptr);
            const spacemit_plugin_kernel * want = entry_named(step->name);
            REQUIRE(want != nullptr);
            CHECK(want == step && want->fn == step->fn && std::strcmp(want->name, step->name) == 0);
        }
    }

    // the core's own guard, tested on a copy of the real table that lacks one kernel (the loader refuses such a
    // plugin before the core sees it)
    {
        g_real_table                   = table;
        spacemit_plugin_v1 short_table = *table;
        short_table.kernels            = kernels_without_gemm;
        uint32_t n_short               = 0;
        CHECK(short_table.kernels(&n_short) != nullptr && n_short == n_kernels - 1);
        spacemit_ops fake_ops;
        std::string  why;
        CHECK(!spacemit_ops_bind(&short_table, &fake_ops, &why));
        CHECK(why.find("mul_mat.gemm") != std::string::npos);
    }

    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    // attention: three contracts in order, the kernels' supports choosing. One token at n_kv 512 takes the paired
    // split by default and the other two under FLAGOS_SPACEMIT_ATTN; 64 tokens take the one-step kernel in all three
    {
        ggml_init_params params  = { ggml_tensor_overhead() * 32, nullptr, true };
        ggml_context *   ctx     = ggml_init(params);
        const int64_t    d = 128, n_head = 8, n_head_kv = 2, n_kv = 512;
        ggml_tensor *    k_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d * n_head_kv, n_kv + 32);
        ggml_tensor *    v_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d * n_head_kv, n_kv + 32);
        ggml_tensor *    one =
            build_flash_attn(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, n_head, 1), k_cache, v_cache, nullptr,
                             n_head_kv, n_kv, 0.0f);
        ggml_tensor * batch =
            build_flash_attn(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, n_head, 64), k_cache, v_cache, nullptr,
                             n_head_kv, n_kv, 0.0f);
        const struct {
            const char * setting;
            const char * first;
            uint32_t     n_steps;
        } wanted[] = {
            { nullptr,     "flash_attn.split.partial_paired", 2 },
            { "split",     "flash_attn.split.partial",        2 },
            { "one-chunk", "flash_attn",                      1 },
        };
        for (const auto & w : wanted) {
            env_override        attn_switch("FLAGOS_SPACEMIT_ATTN", w.setting);
            const spacemit_op * got = spacemit_find_op(ops, one, buft);
            REQUIRE(got != nullptr);
            CHECK(got->n_steps == w.n_steps);
            CHECK(got->steps[0] == entry_named(w.first));
            if (got->n_steps == 2) {
                CHECK(got->steps[1] == entry_named("flash_attn.split.merge"));
            }
            const spacemit_op * many = spacemit_find_op(ops, batch, buft);
            REQUIRE(many != nullptr);
            CHECK(many->n_steps == 1 && many->steps[0] == entry_named("flash_attn"));
        }
        ggml_free(ctx);
    }

    // the KV-head split (plan.md "The KV-head split for a few tokens, from SpacemiT's fork"): the first attention
    // contract, taken for the token counts FLAGOS_SPACEMIT_ATTN_KVH names (unset or empty: 2 to 8), with 2, 4 or 8 query
    // heads per KV head, without ALiBi and from 128 positions per chunk (2 chunks with 4 KV heads); otherwise the calls
    // go as above
    {
        ggml_init_params params = { ggml_tensor_overhead() * 512, nullptr, true };
        ggml_context *   ctx    = ggml_init(params);
        const int64_t    d      = 128;
        const struct {
            const char * setting;
            int64_t      lo, hi;  // the token counts it takes; 0, 0: none
        } settings[] = {
            { nullptr, 2, 8 }, { "", 2, 8 }, { "off", 0, 0 }, { "2-4", 2, 4 }, { "1-4", 1, 4 }, { "2-8", 2, 8 },
        };
        const struct {
            const char * what;
            int64_t      n_head, n_head_kv, n_kv;
            float        max_bias;
            bool         kvh;  // a shape the split takes
        } shapes[] = {
            { "Qwen3-4B", 32, 8, 512, 0.0f, true },
            { "GQA 8, n_kv 256", 32, 4, 256, 0.0f, true },
            { "GQA 8, n_kv 255", 32, 4, 255, 0.0f, false },
            { "GQA 1", 8, 8, 512, 0.0f, false },
            { "ALiBi", 32, 8, 512, 8.0f, false },
        };
        const int64_t token_counts[] = { 1, 2, 3, 4, 5, 8, 9, 64 };
        ggml_tensor * nodes[5][8];
        for (int i = 0; i < 5; i++) {
            ggml_tensor * k_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d * shapes[i].n_head_kv, shapes[i].n_kv + 32);
            ggml_tensor * v_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d * shapes[i].n_head_kv, shapes[i].n_kv + 32);
            for (int j = 0; j < 8; j++) {
                ggml_tensor * q    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, shapes[i].n_head, token_counts[j]);
                ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, shapes[i].n_kv, token_counts[j], 1, 1);
                nodes[i][j]        = build_flash_attn(ctx, q, k_cache, v_cache, mask, shapes[i].n_head_kv,
                                                      shapes[i].n_kv, shapes[i].max_bias);
            }
        }
        int n_checked = 0;
        for (const auto & w : settings) {
            env_override kvh_switch("FLAGOS_SPACEMIT_ATTN_KVH", w.setting);
            env_override attn_switch("FLAGOS_SPACEMIT_ATTN", nullptr);
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 8; j++) {
                    const auto &        sh       = shapes[i];
                    const int64_t       n_tokens = token_counts[j];
                    const spacemit_op * got      = spacemit_find_op(ops, nodes[i][j], buft);
                    REQUIRE(got != nullptr);
                    const bool   kvh   = sh.kvh && w.lo <= n_tokens && n_tokens <= w.hi;
                    const bool   split = n_tokens == 1 && sh.max_bias == 0.0f && sh.n_kv >= 512;
                    const char * first = kvh   ? "flash_attn.kvh.partial" :
                                         split ? "flash_attn.split.partial_paired" :
                                                 "flash_attn";
                    const bool   ok    = got->steps[0] == entry_named(first) &&
                                    (kvh   ? got->n_steps == 2 && got->steps[1] == entry_named("flash_attn.kvh.merge") :
                                     split ? got->n_steps == 2 :
                                             got->n_steps == 1);
                    if (!ok) {
                        std::fprintf(stderr,
                                     "FAIL: contracts, FLAGOS_SPACEMIT_ATTN_KVH=%s, %s, %lld tokens: %s, not %s\n",
                                     w.setting != nullptr ? w.setting : "(unset)", sh.what, (long long) n_tokens,
                                     got->steps[0]->name, first);
                        g_failures++;
                    }
                    n_checked++;
                }
            }
        }
        std::printf("contracts FLASH_ATTN_EXT under FLAGOS_SPACEMIT_ATTN_KVH (unset, empty, off, 2-4, 1-4, 2-8; "
                    "1-64 tokens; Qwen3-4B, GQA 8 at n_kv 256 and 255, GQA 1, ALiBi): %d selections checked\n",
                    n_checked);
        ggml_free(ctx);
    }

    // MUL_MAT: the layout rule through the table, so a view of a repacked weight is not one; the workspace from the
    // entries, the two steps sharing one rule
    {
        ggml_init_params params = { ggml_tensor_overhead() * 8, nullptr, true };
        ggml_context *   ctx    = ggml_init(params);
        ggml_tensor *    w      = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 32);
        ggml_tensor *    x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
        ggml_tensor *    view   = ggml_view_2d(ctx, w, 256, 32, w->nb[1], 0);
        CHECK(spacemit_find_op(ops, ggml_mul_mat(ctx, view, x), buft) == nullptr);
        ggml_tensor *       node = ggml_mul_mat(ctx, w, x);
        const spacemit_op * mm   = spacemit_find_op(ops, node, buft);
        REQUIRE(mm != nullptr);
        CHECK(mm->n_steps == 2);
        CHECK(mm->steps[0] == entry_named("mul_mat.quantize"));
        CHECK(mm->steps[1] == entry_named("mul_mat.gemm"));
        const spacemit_plugin_kernel * gemm = entry_named("mul_mat.gemm");
        REQUIRE(gemm != nullptr && gemm->workspace_size != nullptr);
        CHECK(spacemit_op_workspace(mm, node) == gemm->workspace_size(node));
        ggml_free(ctx);
    }

    std::printf("contracts %u bound to %u kernels: checked\n", ops.n_ops, n_kernels);
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
    // counts within a launch, so the launches are set here: one per split, and one per step with the host waiting
    // (structural-experiments.md E3a), where a failure in a step's launch must fail the graph (a launch of one step has
    // no step 1)
    constexpr int k_fail_runs = 100;
    int           fail_ok     = 0;
    for (const char * launch : { "per-split", "per-step" }) {
        // GGML_FLAGOS_SYNC_LAUNCH: unset is one launch per split, 1 one launch per step
        env_override launch_switch("GGML_FLAGOS_SYNC_LAUNCH",
                                   std::strcmp(launch, "per-step") == 0 ? "1" : nullptr);
        for (const char * fail_node : { "0", "1" }) {
            if (std::strcmp(launch, "per-split") != 0 && std::strcmp(fail_node, "1") == 0) {
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

// the weights the provider stores requantized to Q8_0 (Q4_1, Q6_K): their reads are approximate, so the CPU reference
// takes the GGUF bytes and, for the tight bound, ggml's Q8_0 of them (mm_weight)
static bool stored_as_q8_0(ggml_type type) {
    return type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q4_1;
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
// The reference gets the provider's read-back: the GGUF bytes for Q4_0 and Q8_0, so a comparison measures the kernels
// and the tiling. With cpu_gguf (stored_as_q8_0: Q4_1, Q6_K) it gets the GGUF bytes instead: a read is only
// the nearest Q6_K or Q4_1 to the kernels' Q8_0 weights, so the comparison covers what the model sees, the
// requantization and the kernels; its error is mostly the requantization's own, so it only catches gross faults. w_q8
// then holds ggml's Q8_0 of the GGUF values (requantize_q8_0), which a correct requantization matches (Q6_K: up to
// ties; Q4_1: bit for bit, as the provider requantizes it with that quantizer): a comparison with it measures the
// requantization's faults, and the kernels'.
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
// a Q6_K or Q4_1 matmul against the CPU's on ggml's Q8_0 of the same values (mm_weight::w_q8): Q6_K's ties that round
// apart give 4-5e-7 (Mac); faults in the requantization give 9e-6 (a stored scale 0.3% too large) to 2e-4 (rounding up
// by half)
constexpr double k_requant_max_nmse = 5e-6;

// which tensors the provider claims for MUL_MAT: exactly Q4_0, Q4_1, Q8_0 and Q6_K weights in their IME layouts, in its
// buffer; and what its buffer type allocates for them (Q4_1 and Q6_K are stored as Q8_0)
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

    // a Q6_K weight takes 34 bytes per 32 values (Q8_0) instead of 26.25, a Q4_1 weight 34 instead of 20, the others
    // their own size. Required: the round trip and the matmuls below would overrun a smaller allocation
    REQUIRE(ggml_backend_buft_get_alloc_size(buft, w6) == 64 * (2560 / 32) * 34);
    REQUIRE(ggml_backend_buft_get_alloc_size(buft, v1) == (size_t) 64 * (256 / 32) * 34);
    CHECK(ggml_backend_buft_get_alloc_size(buft, w8_k) == ggml_nbytes(w8_k));
    CHECK(ggml_backend_buft_get_alloc_size(buft, w) == ggml_nbytes(w));

    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx);
    ggml_free(ctx_cpu);
    return 0;
}

// how far a read of a Q4_1 weight stored as Q8_0 may lie from the GGUF weights: these random weights give 5.4e-6 (Mac)
constexpr double k_q4_1_read_max_nmse = 5e-5;

// set_tensor repacks; get_tensor restores the GGUF bytes (Q4_0, Q8_0) or returns the nearest Q4_1 or Q6_K to the
// stored Q8_0 values (Q4_1, Q6_K); partial access works (whole writes only for the weights stored as Q8_0)
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

    const std::vector<uint8_t> q1 = quantize_weights(GGML_TYPE_Q4_1, 512, 64, 9);
    std::vector<uint8_t>       c1(q1.size()), again(q1.size());
    // Q4_1 stored requantized to Q8_0: as Q6_K below, a read is the nearest Q4_1 to the stored
    // values, within NMSE k_q4_1_read_max_nmse of the GGUF weights; writing it back and reading again stays as
    // close; a partial read is that part of a whole read. check_mul_mat checks the stored values themselves
    ggml_backend_tensor_set(w1, q1.data(), 0, q1.size());
    CHECK(std::memcmp(w1->data, q1.data(), q1.size()) != 0);
    ggml_backend_tensor_get(w1, c1.data(), 0, c1.size());
    const std::vector<float> f1 = dequantize(GGML_TYPE_Q4_1, q1), f1_read = dequantize(GGML_TYPE_Q4_1, c1);
    CHECK(nmse(f1_read.data(), f1.data(), (int64_t) f1.size()) <= k_q4_1_read_max_nmse);
    ggml_backend_tensor_get(w1, again.data(), 1000, 300);
    CHECK(std::memcmp(again.data(), c1.data() + 1000, 300) == 0);
    ggml_backend_tensor_set(w1, c1.data(), 0, c1.size());
    ggml_backend_tensor_get(w1, again.data(), 0, again.size());
    const std::vector<float> f1_again = dequantize(GGML_TYPE_Q4_1, again);
    CHECK(nmse(f1_again.data(), f1_read.data(), (int64_t) f1_read.size()) <= k_q4_1_read_max_nmse);

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

// how much the provider's form of Q4_1 weights alone changes a matmul: the CPU backend with the GGUF Q4_1 weights
// against the CPU backend with the weights the provider computes with, ggml's Q8_0 of them; random weights at
// Qwen3-0.6B's ffn_down shape, 16 rows
// (informational: the accuracy budget is decided by perplexity and KLD on the model, plan.md M2c design)
static double q4_1_conversion_nmse(ggml_backend_dev_t dev, ggml_backend_t cpu) {
    const int64_t k = 3072, n = 1024, m = 16;
    mm_weight     w;
    if (!w.init(dev, k, n, 21, GGML_TYPE_Q4_1, true)) {
        return -1.0;
    }
    ggml_init_params params = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    ggml_tensor *    w_gguf = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, k, n);
    ggml_tensor *    x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    ggml_tensor *    y_conv = ggml_mul_mat(ctx, w.w_q8, x);
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

// the provider against the CPU backend for shapes and row counts that take every path: 1 row (GEMV) and, by the
// default rule, path B where TCM holds the activations next to a 32-column slab, else path A from 32 rows, else path C
// (on the K3 Qwen3-4B's Q4_1 down at 7 rows, with --full; without TCM every case); 4-row blocks with partial tails.
// split's paths give the same bytes (check_mm_paths). Q4_1 and Q6_K (stored as Q8_0) are compared with the GGUF
// weights and with ggml's Q8_0 of them (mm_weight)
static int check_mul_mat(ggml_backend_dev_t dev, bool full) {
    if (check_mul_mat_claims(dev) != 0 || check_weight_round_trip(dev) != 0) {
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);  // a correctness reference: single-threaded (PROJECT.md §6, item 28)

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

    const ggml_type types[4]       = { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K };
    int             n_cases[4]     = { 0, 0, 0, 0 };
    int64_t         max_rows[4]    = { 0, 0, 0, 0 };
    double          max_nmse[4]    = { 0.0, 0.0, 0.0, 0.0 };
    double          max_nmse_q8[4] = { 0.0, 0.0, 0.0, 0.0 };  // against the CPU's Q8_0 of the same values
    for (const shape & s : shapes) {
        const int t = (int) (std::find(types, types + 4, s.type) - types);
        mm_weight w;
        REQUIRE(w.init(dev, s.k, s.n, (uint32_t) (s.k + s.n), s.type, stored_as_q8_0(s.type)));
        for (int64_t m : s.rows) {
            mm_run run;
            REQUIRE(run.init(dev, w, m, (uint32_t) m));
            double       e_q8 = -1.0;
            const double e    = run.compare(dev, backend, cpu, &e_q8);
            if (e < 0.0 || e > k_mul_mat_max_nmse) {
                std::fprintf(stderr, "FAIL: %s matmul k=%lld n=%lld rows=%lld: %s %.3e\n", ggml_type_name(s.type),
                             (long long) s.k, (long long) s.n, (long long) m, e < 0.0 ? "refused or failed" : "NMSE", e);
                g_failures++;
            } else if (stored_as_q8_0(s.type) && (e_q8 < 0.0 || e_q8 > k_requant_max_nmse)) {
                std::fprintf(stderr, "FAIL: %s matmul k=%lld n=%lld rows=%lld against the CPU's Q8_0 of its values: "
                             "NMSE %.3e (bound %.0e)\n",
                             ggml_type_name(s.type), (long long) s.k, (long long) s.n, (long long) m, e_q8,
                             k_requant_max_nmse);
                g_failures++;
            }
            max_nmse[t]    = std::max(max_nmse[t], e);
            max_nmse_q8[t] = std::max(max_nmse_q8[t], e_q8);
            max_rows[t]    = std::max(max_rows[t], m);
            n_cases[t]++;
        }
    }
    check_mul_mat_add(dev, backend, cpu);
    const double conversion = q4_1_conversion_nmse(dev, cpu);
    CHECK(conversion >= 0.0);  // ran; the value is informational

    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    const char * kernels = reference_label();
    std::printf("matmul   Q4_0 (IME layout 32x256) vs CPU: %d cases, rows 1-%d, max NMSE %.2e (bound %.0e)%s\n",
                n_cases[0], (int) max_rows[0], max_nmse[0], k_mul_mat_max_nmse, kernels);
    std::printf("matmul   Q4_1 (requantized to Q8_0, IME layout 32x32) vs CPU's Q4_1: %d cases, rows 1-%d, max NMSE "
                "%.2e (bound %.0e); vs CPU's Q8_0 of the same values (the requantization): max NMSE %.2e (bound "
                "%.0e)%s\n",
                n_cases[1], (int) max_rows[1], max_nmse[1], k_mul_mat_max_nmse, max_nmse_q8[1], k_requant_max_nmse,
                kernels);
    std::printf("matmul   Q8_0 (IME layout 32x32) vs CPU: %d cases, rows 1-%d, max NMSE %.2e (bound %.0e)%s\n",
                n_cases[2], (int) max_rows[2], max_nmse[2], k_mul_mat_max_nmse, kernels);
    std::printf("matmul   Q6_K (requantized to Q8_0, IME layout 32x32) vs CPU's Q6_K: %d cases, rows 1-%d, max NMSE %.2e "
                "(bound %.0e); vs CPU's Q8_0 of the same values (the requantization): max NMSE %.2e (bound %.0e)%s\n",
                n_cases[3], (int) max_rows[3], max_nmse[3], k_mul_mat_max_nmse, max_nmse_q8[3], k_requant_max_nmse,
                kernels);
    std::printf("matmul   claims exact (any row count; not other layouts, types, views, F16 or strided activations, "
                "CPU-buffer weights); Q4_1 and Q6_K allocation sizes; repack round trip (Q4_1 read as the nearest Q4_1, "
                "within NMSE 5e-5; Q6_K read as the nearest Q6_K, within NMSE 1e-5); MUL_MAT->ADD graph: checked\n");
    std::printf("info     Q4_1 requantization to Q8_0 alone (CPU, GGUF against ggml's Q8_0 of the values), random weights "
                "3072x1024, 16 rows: NMSE %.2e\n",
                conversion);
    return 0;
}

// the 1-row path (generation; plan.md, "1-row matmul bandwidth"): the weights staged through TCM in 32-column slabs, the
// two tiles of a pair taking turns (gemv, flagos-spacemit-ime.cpp). With TCM against the CPU, the output filled with NaN
// first so skipped columns show. On a host without the AI cores the host executor gets a stand-in of the K3's 384 KiB
// (spine-runtime ignores the variable), with the tiles one after another, which counts the pair waits, and on threads
// (FLAGOS_SPACEMIT_TEST_THREADS), where the waits wait; there each must give the same bytes as a backend without TCM,
// which reads the weights straight from DRAM with the same kernel. The shapes give a pair whose odd tile has one slab
// fewer (480 and 1056 columns) and a long row
static int check_gemv(ggml_backend_dev_t dev, bool host) {
    std::vector<ggml_backend_t> backends;  // with TCM: in turn, then (host) on threads
    for (const char * threads : { (const char *) nullptr, "1" }) {
        if (threads != nullptr && !host) {
            break;
        }
        env_override tcm("FLAGOS_SPACEMIT_TEST_TCM", "393216");
        env_override on_threads("FLAGOS_SPACEMIT_TEST_THREADS", threads);
        backends.push_back(ggml_backend_dev_init(dev, nullptr));
        REQUIRE(backends.back() != nullptr);
    }
    ggml_backend_t direct = nullptr;  // host only: no TCM, the weights read straight from DRAM
    if (host) {
        env_override tcm("FLAGOS_SPACEMIT_TEST_TCM", nullptr);
        direct = ggml_backend_dev_init(dev, nullptr);
        REQUIRE(direct != nullptr);
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);  // a correctness reference: single-threaded (PROJECT.md §6, item 28)

    struct shape { ggml_type type; int64_t k, n; };
    const shape shapes[] = {
        { GGML_TYPE_Q4_0, 2560, 1024 },  // Qwen3-4B k/v projection
        { GGML_TYPE_Q4_0, 512, 480 },    // 15 groups of 32 columns: partial ranges and slabs
        { GGML_TYPE_Q4_0, 9728, 512 },   // a long row
        { GGML_TYPE_Q4_1, 9728, 512 },
        { GGML_TYPE_Q8_0, 2560, 2048 },
        { GGML_TYPE_Q6_K, 2560, 1056 },  // 33 groups
    };
    int    n_same = 0, n_runs = 0;
    double worst  = 0.0;
    for (const shape & s : shapes) {
        mm_weight w;
        REQUIRE(w.init(dev, s.k, s.n, (uint32_t) (3 * s.k + s.n), s.type, stored_as_q8_0(s.type)));
        mm_run run;
        REQUIRE(run.init(dev, w, 1, 9));
        std::vector<float>       ref(s.n), out(s.n);
        const std::vector<float> nan(s.n, NAN);
        if (direct != nullptr) {
            REQUIRE(ggml_backend_graph_compute(direct, run.g_dev) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(run.y, ref.data(), 0, ggml_nbytes(run.y));
        }
        for (ggml_backend_t backend : backends) {
            ggml_backend_tensor_set(run.y, nan.data(), 0, ggml_nbytes(run.y));
            const double e = run.compare(dev, backend, cpu);
            ggml_backend_tensor_get(run.y, out.data(), 0, ggml_nbytes(run.y));
            const bool same = direct == nullptr || std::memcmp(out.data(), ref.data(), ggml_nbytes(run.y)) == 0;
            if (e < 0.0 || e > k_mul_mat_max_nmse || !same) {
                std::fprintf(stderr, "FAIL: %s 1-row matmul k=%lld n=%lld with TCM%s: %s %.3e\n", ggml_type_name(s.type),
                             (long long) s.k, (long long) s.n, backend != backends[0] ? " (tiles on threads)" : "",
                             e < 0.0 ? "refused or failed" : !same ? "differs from the read without TCM, NMSE" : "NMSE",
                             e);
                g_failures++;
            } else {
                n_same++;
            }
            worst = std::max(worst, e);
            n_runs++;
        }
    }

    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }
    if (direct != nullptr) {
        ggml_backend_free(direct);
    }
    ggml_backend_free(cpu);
    std::printf("gemv     the 1-row path at 6 shapes with TCM%s: %d/%d %s; vs CPU max NMSE %.2e (bound %.0e)%s\n",
                host ? ", tiles in turn and on threads" : "", n_same, n_runs,
                host ? "identical to the read without TCM" : "within the bound", worst, k_mul_mat_max_nmse,
                reference_label());
    return 0;
}

static double ime_row_bytes(ggml_type type, int64_t k);static double ime_row_bytes(ggml_type type, int64_t k);

// bytes of one quantized activation row in the provider's layout for a weight type: the Q4_0 32x256 layout's 290 per
// 256 values, the 32x32 layouts' 38 per 32 (flagos-spacemit-ime-kernels.h)
static int64_t ime_act_row_bytes(ggml_type type, int64_t k) {
    return type == GGML_TYPE_Q4_0 ? k / 256 * 290 : k / 32 * 38;
}

// fusion-and-splitting.md §7.3: the default (fit,staged: path B where TCM holds the activations next to a slab, else
// path A from 32 rows, else path C; path B without the pair turns) against FLAGOS_SPACEMIT_MM=split (the default until
// 2026-10-25): the same bytes, as both run the same kernels on the same groups of 4 rows and blocks of 32 columns. The
// paths taken are counted (spacemit_mm_paths_take) and compared with what the rules give, worked out here again; the
// output is filled with NaN before every run, so columns a path skips show, and split's must come out finite. Path A's
// last m % 4 rows are counted by the tile that multiplied them, with its calls (spacemit_mm_tail_take: one per tile
// that has any, as a tile's rows are consecutive and two rows in two calls stream the weights twice), and compared
// with their dealing, worked out here again. Paths A and B need TCM: as check_gemv, a stand-in of the K3's 384 KiB on a
// host without the AI cores, there also with tiles on threads
static int check_mm_paths(ggml_backend_dev_t dev, bool host) {
    env_override                mm("FLAGOS_SPACEMIT_MM", nullptr);
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

    struct mm_case { ggml_type type; int64_t k, n, m; };
    const mm_case cases[] = {
        { GGML_TYPE_Q4_0, 512, 4096, 32 },   // the fewest rows that give each of 8 tiles 4
        { GGML_TYPE_Q4_0, 512, 1024, 37 },   // a last group of 1 row
        { GGML_TYPE_Q4_0, 256, 8320, 128 },  // more than 64 times wider than the rows, as gate and up at 128 rows
        { GGML_TYPE_Q4_1, 512, 1024, 66 },
        { GGML_TYPE_Q8_0, 512, 1024, 64 },
        { GGML_TYPE_Q6_K, 512, 1024, 33 },
        // below 32 rows: path B where the activations fit, else path C
        { GGML_TYPE_Q4_0, 512, 1024, 2 },   // the fewest rows that are not the 1-row path
        { GGML_TYPE_Q4_0, 512, 416, 17 },   // 13 slabs: tiles 4 and 5 get 2 and 1; a last group of 1 row
        { GGML_TYPE_Q4_0, 512, 96, 6 },     // 3 slabs: tile 3 gets none while tile 2 gets one; tiles without slabs
        { GGML_TYPE_Q4_0, 9728, 512, 21 },  // the activations do not fit in TCM next to a slab: path C
        { GGML_TYPE_Q4_1, 9728, 256, 19 },  // the same in Q4_1
        { GGML_TYPE_Q8_0, 512, 1024, 3 },
        { GGML_TYPE_Q6_K, 512, 1024, 16 },
        // the default's other paths
        { GGML_TYPE_Q4_0, 9728, 512, 33 },   // the activations do not fit, every tile gets 4 rows: path A
        { GGML_TYPE_Q4_0, 512, 1024, 113 },  // the split gives path A, the activations fit: path B
        // path A's last m % 4 rows, dealt to the tiles without a group in the last round (37, 66 and 113 above too)
        { GGML_TYPE_Q4_0, 512, 1024, 34 },   // no group in the last round: tiles 0 and 1 take a row each
        { GGML_TYPE_Q4_0, 512, 1024, 35 },   // tiles 0, 1 and 2
        { GGML_TYPE_Q4_0, 512, 1024, 47 },   // tiles 0-2 hold the last round's groups: tiles 3, 4 and 5
        { GGML_TYPE_Q4_0, 512, 1024, 59 },   // only tiles 6 and 7 idle: tile 6 two rows, tile 7 one
        { GGML_TYPE_Q4_0, 512, 1024, 62 },   // only tile 7 idle: it keeps both rows
        { GGML_TYPE_Q4_0, 512, 1024, 63 },   // tile 7 two rows, tile 0 the third after its group
        { GGML_TYPE_Q8_0, 512, 1024, 59 },
        { GGML_TYPE_Q4_1, 512, 1024, 63 },
        { GGML_TYPE_Q6_K, 512, 1024, 47 },
        { GGML_TYPE_Q4_0, 9728, 512, 35 },   // the default's path A (the activations do not fit next to a slab)
        { GGML_TYPE_Q4_0, 512, 1024, 115 },  // the split's path A: tiles 4, 5 and 6
    };
    constexpr int      n_set           = 2;
    const char * const settings[n_set] = { "split", nullptr };
    const char * const names[n_set]    = { "split", "the default" };
    int                n_same = 0, n_runs = 0;
    for (const mm_case & c : cases) {
        mm_weight w;
        REQUIRE(w.init(dev, c.k, c.n, (uint32_t) (c.k + c.n + c.m), c.type, stored_as_q8_0(c.type)));
        mm_run run;
        REQUIRE(run.init(dev, w, c.m, 12));
        const std::vector<float> nan(ggml_nelements(run.y), NAN);
        // the path each setting must take with 8 tiles and 384 KiB of TCM (spacemit_mul_mat_gemm): split path A where
        // ime.cpp's split gives every tile all columns, else path C; the default path B where TCM holds the
        // activations next to a 32-column slab, else path A from 32 rows, else path C
        const int64_t m_stride = c.n / c.m > 64 ? c.m : 16;
        const bool    split_a  = (c.n * ((c.m + m_stride - 1) / m_stride) + 7) / 8 >= c.n;
        const bool    fit_b    = (int64_t) GGML_PAD(c.m * ime_act_row_bytes(c.type, c.k), 64) +
                                  32 * (int64_t) ime_row_bytes(c.type, c.k) <=
                              393216;
        const char    expect[n_set] = { split_a ? 'A' : 'C', fit_b ? 'B' : c.m >= 32 ? 'A' : 'C' };
        // path A's last m % 4 rows by tile, dealt: one to each tile from g, the tile after the last group, to tile 7,
        // then once more over those tiles, then to tile 0 onwards
        constexpr int tiles = 8;
        const int64_t left  = c.m % 4;
        const int64_t g     = (c.m - left) / 4 % tiles;
        uint64_t      tail_dealt[tiles] = {};
        for (int64_t j = 0, t = g, pass = 0, extra = 0; j < left; j++) {
            if (pass < 2) {
                tail_dealt[t]++;
                if (++t == tiles) {
                    t = g;
                    pass++;
                }
            } else {
                tail_dealt[extra++]++;
            }
        }
        for (ggml_backend_t backend : backends) {
            std::vector<float> out[n_set];
            uint64_t           path_a[n_set] = {}, path_b[n_set] = {}, path_c[n_set] = {};
            uint64_t           tail_rows[n_set][tiles] = {}, tail_calls[n_set][tiles] = {};
            bool               ok            = true;
            for (int s = 0; s < n_set; s++) {
                set_env("FLAGOS_SPACEMIT_MM", settings[s]);
                ggml_backend_tensor_set(run.y, nan.data(), 0, ggml_nbytes(run.y));
                g_mm_paths_take(&path_a[s], &path_b[s], &path_c[s]);
                g_mm_tail_take(tail_rows[s], tail_calls[s], tiles);
                ok = ok && ggml_backend_graph_compute(backend, run.g_dev) == GGML_STATUS_SUCCESS;
                g_mm_paths_take(&path_a[s], &path_b[s], &path_c[s]);
                g_mm_tail_take(tail_rows[s], tail_calls[s], tiles);
                out[s].resize(ggml_nelements(run.y));
                ggml_backend_tensor_get(run.y, out[s].data(), 0, ggml_nbytes(run.y));
            }
            set_env("FLAGOS_SPACEMIT_MM", nullptr);
            bool        switched = true;
            int         misdealt = -1;  // the first setting whose leftover rows are not on the expected tiles
            std::string paths;
            for (int s = 0; s < n_set; s++) {
                const char p = expect[s];
                switched = switched && path_a[s] == (uint64_t) (p == 'A') && path_b[s] == (uint64_t) (p == 'B') &&
                           path_c[s] == (uint64_t) (p == 'C');
                for (int i = 0; i < tiles && misdealt < 0; i++) {
                    const uint64_t want = p != 'A' ? 0 : tail_dealt[i];
                    if (tail_rows[s][i] != want || tail_calls[s][i] != (uint64_t) (want > 0)) {
                        misdealt = s;
                    }
                }
                paths += std::string(s == 0 ? "" : ", ") + names[s] + " " + std::to_string(path_a[s]) + "/" +
                         std::to_string(path_b[s]) + "/" + std::to_string(path_c[s]) + " (" + p + ")";
            }
            switched = switched && misdealt < 0;
            std::string dealt;  // the leftover rows by tile, of the first misdealt setting and as expected
            if (misdealt >= 0) {
                dealt = std::string(", ") + names[misdealt] + "'s leftover rows by tile";
                for (int i = 0; i < tiles; i++) {
                    dealt += (i == 0 ? " " : ",") + std::to_string(tail_rows[misdealt][i]);
                }
                dealt += " in calls";
                for (int i = 0; i < tiles; i++) {
                    dealt += (i == 0 ? " " : ",") + std::to_string(tail_calls[misdealt][i]);
                }
                dealt += ", expected";
                for (int i = 0; i < tiles; i++) {
                    dealt += (i == 0 ? " " : ",") + std::to_string(expect[misdealt] != 'A' ? 0 : tail_dealt[i]);
                }
                dealt += ", one call each";
            }
            paths += dealt;
            const bool finite = std::all_of(out[0].begin(), out[0].end(), [](float v) { return std::isfinite(v); });
            const bool differs = std::memcmp(out[0].data(), out[1].data(), out[0].size() * sizeof(float)) != 0;
            const bool same    = ok && finite && !differs;
            if (!same || !switched) {
                const std::string what = !ok           ? std::string("a run failed") :
                                         !finite       ? std::string("split's output is not finite") :
                                         differs       ? std::string("the default differs from split") :
                                         misdealt >= 0 ? std::string("path A's leftover rows not as expected") :
                                                         std::string("a setting did not take its path");
                std::fprintf(stderr,
                             "FAIL: %s matmul k=%lld n=%lld rows=%lld: %s (paths A/B/C taken, expected: %s)%s\n",
                             ggml_type_name(c.type), (long long) c.k, (long long) c.n, (long long) c.m, what.c_str(),
                             paths.c_str(), backend != backends[0] ? " (tiles on threads)" : "");
                g_failures++;
            }
            n_same += same && switched;
            n_runs++;
        }
    }

    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }
    std::printf("mm       the default (fit,staged: path B where TCM holds the activations, else A from 32 rows, else C) "
                "against FLAGOS_SPACEMIT_MM=split: Q4_0, Q4_1, Q8_0 and Q6_K at 2-128 rows with TCM%s: %d/%d identical, "
                "paths and path A's leftover rows by tile, a tile's in one call, as expected%s\n",
                host ? ", tiles in turn and on threads" : "", n_same, n_runs,
                reference_label());
    return 0;
}

//
// M2e.1: FLASH_ATTN_EXT over the KV cache, against the CPU backend
//

// FLASH_ATTN_EXT as llama.cpp builds it over caches that hold every KV head of a position in one cell
// (src/llama-kv-cache.cpp:1249-1300, V not transposed; src/llama-graph.cpp:2500-2545): views of the first n_kv cells
// and q [d, n_head, n_tokens], each permuted (0, 2, 1, 3); scale 1/sqrt(d); precision F32
static ggml_tensor * build_flash_attn(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k_cache, ggml_tensor * v_cache,
                                      ggml_tensor * mask, int64_t n_head_kv, int64_t n_kv, float max_bias,
                                      float logit_softcap, ggml_tensor * sinks) {
    const int64_t d = q->ne[0];
    ggml_tensor * k = ggml_view_4d(ctx, k_cache, d, n_head_kv, n_kv, 1, ggml_row_size(k_cache->type, d), k_cache->nb[1],
                                   ggml_nbytes(k_cache), 0);
    ggml_tensor * v = ggml_view_4d(ctx, v_cache, d, n_head_kv, n_kv, 1, ggml_row_size(v_cache->type, d), v_cache->nb[1],
                                   ggml_nbytes(v_cache), 0);
    ggml_tensor * a = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), ggml_permute(ctx, k, 0, 2, 1, 3),
                                          ggml_permute(ctx, v, 0, 2, 1, 3), mask, 1.0f / std::sqrt((float) d), max_bias,
                                          logit_softcap);
    ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
    if (sinks != nullptr) {
        ggml_flash_attn_ext_add_sinks(a, sinks);
    }
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
// standard deviations at d 128; the mask F16 [n_kv, n_tokens, 1, 1], `visible` added to every cell a token sees.
// logit_softcap as ggml_flash_attn_ext takes it; with_sinks: one sink per query head, in [-4, 4)
struct attn_run {
    ggml_context *        ctx_host = nullptr, * ctx_dev = nullptr;
    ggml_backend_buffer_t buf_host = nullptr, buf_dev = nullptr;
    ggml_tensor *         out = nullptr, * out_ref = nullptr;
    ggml_cgraph *         g_dev = nullptr, * g_cpu = nullptr;

    bool init(ggml_backend_dev_t dev, const attn_shape & s, attn_mask mode, float visible, float max_bias,
              uint32_t seed, float q_scale = 4.0f, float logit_softcap = 0.0f, bool with_sinks = false) {
        ggml_init_params params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        ctx_host                = ggml_init(params);
        ctx_dev                 = ggml_init(params);
        const int64_t cells     = s.n_kv + 32;
        ggml_tensor * k_cache   = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F16, s.d * s.n_head_kv, cells);
        ggml_tensor * v_cache   = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F16, s.d * s.n_head_kv, cells);
        ggml_tensor * q         = ggml_new_tensor_3d(ctx_host, GGML_TYPE_F32, s.d, s.n_head, s.n_tokens);
        ggml_tensor * mask =
            mode == attn_mask::none ? nullptr : ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, s.n_kv, s.n_tokens, 1, 1);
        ggml_tensor * sinks = with_sinks ? ggml_new_tensor_1d(ctx_host, GGML_TYPE_F32, s.n_head) : nullptr;
        out_ref  = build_flash_attn(ctx_host, q, k_cache, v_cache, mask, s.n_head_kv, s.n_kv, max_bias, logit_softcap,
                                    sinks);
        out      = build_flash_attn(ctx_dev, q, k_cache, v_cache, mask, s.n_head_kv, s.n_kv, max_bias, logit_softcap,
                                    sinks);
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
        if (sinks != nullptr) {
            ggml_backend_tensor_set(sinks, random_values(s.n_head, seed + 3, 4.0f).data(), 0, ggml_nbytes(sinks));
        }
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
    ggml_backend_cpu_set_n_threads(cpu, 1);  // a correctness reference: single-threaded (PROJECT.md §6, item 28)

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
    // default (M2e.3 round 2, checked below, where each run sets the switch itself), and 2 to 8 tokens the KV-head split
    // (checked below, where each run sets FLAGOS_SPACEMIT_ATTN_KVH itself)
    env_override one_chunk_switch("FLAGOS_SPACEMIT_ATTN", "one-chunk");
    env_override kvh_off_switch("FLAGOS_SPACEMIT_ATTN_KVH", "off");
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
    // GGML_FLAGOS_KERNEL_VARIANTS=0) the split's reference runs
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
        // structural-experiments.md E3a: the default split with its two steps (the tiles' partial results, then their
        // merge, passed through the workspace) in launches of their own, the host waiting for each, against both in
        // one launch
        std::vector<float> launched[2];  // l 0: one launch per split, l 1: GGML_FLAGOS_SYNC_LAUNCH=1
        for (int l = 0; l < 2; l++) {
            env_override attn_switch("FLAGOS_SPACEMIT_ATTN", nullptr);
            env_override launch_switch("GGML_FLAGOS_SYNC_LAUNCH", l == 1 ? "1" : nullptr);
            REQUIRE(run.dev_output(backend, launched[l]));
        }
        const bool same_launch =
            std::memcmp(launched[0].data(), launched[1].data(), launched[0].size() * sizeof(float)) == 0;
        if (!same_launch) {
            std::fprintf(stderr, "FAIL: attention split, %s: GGML_FLAGOS_SYNC_LAUNCH=1 differs from one launch\n",
                         c.what);
            g_failures++;
        }
        n_launch_same += same_launch;
    }

    // the KV-head split for a few tokens (plan.md "The KV-head split for a few tokens, from SpacemiT's fork") under
    // FLAGOS_SPACEMIT_ATTN_KVH=1-8, against the CPU within the same bounds. Each case must take its contract
    // (spacemit_find_op) and repeat bit for bit in a second run; that its outputs differ from those under
    // FLAGOS_SPACEMIT_ATTN_KVH=off is counted only, since with one chunk per KV head the reference (the Mac) sums as the
    // other kernels' do. The check_contracts table holds the default's token counts. The cases take
    // its edges: 3 tokens on Qwen3-4B (3 passes of 4 rows), GQA 2 (8 tokens in one pass) and 8 (2 chunks), 2 KV heads (4
    // chunks) with the last chunk hidden from tokens 0 and 1, n_kv 1000 and 999 (neither a multiple of 64 positions
    // nor, at 999, of the chunk), head size 64, no mask, sinks, softcap, every cell hidden, one token. The two steps also
    // run directly on the host with 3 and 5 tiles (the reference there), which must agree bit for bit
    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    spacemit_ops ops;
    std::string  detail;
    REQUIRE(spacemit_ops_bind(table, &ops, &detail));
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    const spacemit_plugin_kernel * kvh_steps[2] = { nullptr, nullptr };
    for (uint32_t k = 0; kernels != nullptr && k < n_kernels; k++) {
        for (int i = 0; i < 2; i++) {
            if (kernels[k].name != nullptr &&
                std::strcmp(kernels[k].name, i == 0 ? "flash_attn.kvh.partial" : "flash_attn.kvh.merge") == 0) {
                kvh_steps[i] = &kernels[k];
            }
        }
    }
    REQUIRE(kvh_steps[0] != nullptr && kvh_steps[1] != nullptr && kvh_steps[0]->workspace_size != nullptr);
    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    struct kvh_case {
        const char * what;
        attn_shape   s;
        attn_mask    mode    = attn_mask::causal;
        float        softcap = 0.0f;
        bool         sinks   = false;
        float        visible = 0.0f;
    };
    std::vector<kvh_case> kvh_cases;
    for (int64_t n_tokens : { 2, 3, 4, 8 }) {
        for (int64_t n_kv : { 256, 512, 1000, 4096 }) {
            kvh_cases.push_back({ "Qwen3-4B", { 32, 8, 128, n_kv, n_tokens, n_kv - n_tokens } });
        }
    }
    kvh_cases.push_back({ "Qwen3-0.6B", { 16, 8, 128, 512, 2, 510 } });
    kvh_cases.push_back({ "Qwen3-0.6B", { 16, 8, 128, 4096, 2, 4094 } });
    kvh_cases.push_back({ "Qwen3-0.6B", { 16, 8, 128, 512, 3, 509 } });
    kvh_cases.push_back({ "Qwen3-0.6B", { 16, 8, 128, 512, 4, 508 } });
    kvh_cases.push_back({ "Qwen3-0.6B", { 16, 8, 128, 512, 8, 504 } });
    kvh_cases.push_back({ "GQA 8, 2 chunks", { 32, 4, 128, 1024, 4, 1020 } });
    kvh_cases.push_back({ "2 KV heads, 4 chunks, the last hidden from tokens 0 and 1", { 8, 2, 128, 1024, 4, 766 } });
    kvh_cases.push_back({ "2 KV heads, chunks of 250 and 249 positions", { 8, 2, 128, 999, 3, 996 } });
    kvh_cases.push_back({ "head size 64", { 32, 8, 64, 1024, 4, 1020 } });
    kvh_cases.push_back({ "no mask", { 32, 8, 128, 1024, 4, 1020 }, attn_mask::none });
    kvh_cases.push_back({ "sinks", { 32, 8, 128, 1024, 4, 1020 }, attn_mask::causal, 0.0f, true });
    kvh_cases.push_back({ "sinks, GQA 8, 2 chunks", { 32, 4, 128, 1024, 4, 1020 }, attn_mask::causal, 0.0f, true });
    kvh_cases.push_back({ "softcap 2", { 32, 8, 128, 1024, 4, 1020 }, attn_mask::causal, 2.0f });
    kvh_cases.push_back(
        { "every cell hidden, sinks", { 32, 4, 128, 1024, 4, 1020 }, attn_mask::causal, 0.0f, true, -INFINITY });
    kvh_cases.push_back({ "1 token", { 32, 8, 128, 1024, 1, 1023 } });
    int    n_kvh_taken = 0, n_kvh_differ = 0, n_kvh_repeat = 0, n_kvh_direct = 0, n_kvh_direct_same = 0;
    double e_kvh = 0.0, e_kvh_row = 0.0;
    for (const kvh_case & c : kvh_cases) {
        attn_run run;
        REQUIRE(run.init(dev, c.s, c.mode, c.visible, 0.0f, seed++, 4.0f, c.softcap, c.sinks));
        REQUIRE(ggml_backend_graph_compute(cpu, run.g_cpu) == GGML_STATUS_SUCCESS);
        const int64_t            n = ggml_nelements(run.out);
        const float *            r = static_cast<const float *>(run.out_ref->data);
        const std::vector<float> ref(r, r + n);
        std::vector<float>       off, kvh[2];
        {
            env_override attn_switch("FLAGOS_SPACEMIT_ATTN", nullptr);
            env_override kvh_switch("FLAGOS_SPACEMIT_ATTN_KVH", "off");
            REQUIRE(run.dev_output(backend, off));
        }
        env_override        attn_switch("FLAGOS_SPACEMIT_ATTN", nullptr);
        env_override        kvh_switch("FLAGOS_SPACEMIT_ATTN_KVH", "1-8");
        const spacemit_op * got   = spacemit_find_op(ops, run.out, buft);
        const bool          taken = got != nullptr && got->steps[0] == kvh_steps[0];
        if (!taken) {
            std::fprintf(stderr, "FAIL: attention KV-head split, %s, %lld tokens at n_kv %lld: its contract not taken\n",
                         c.what, (long long) c.s.n_tokens, (long long) c.s.n_kv);
            g_failures++;
        }
        n_kvh_taken += taken;
        for (int k = 0; k < 2; k++) {
            REQUIRE(run.dev_output(backend, kvh[k]));
        }
        const attn_error  e    = attn_errors(kvh[0].data(), ref.data(), n, run.out->ne[0]);
        const std::string what = std::string("KV-head split, ") + c.what;
        attn_report(e, what.c_str(), c.s);
        e_kvh     = std::max(e_kvh, e.all);
        e_kvh_row = std::max(e_kvh_row, e.row);
        n_kvh_differ += std::memcmp(kvh[0].data(), off.data(), n * sizeof(float)) != 0;
        const bool repeat = std::memcmp(kvh[0].data(), kvh[1].data(), n * sizeof(float)) == 0;
        if (!repeat) {
            std::fprintf(stderr, "FAIL: attention KV-head split, %s: two runs differ\n", c.what);
            g_failures++;
        }
        n_kvh_repeat += repeat;

        // the steps called directly on the CPU-buffer node, with 3 and 5 tiles (the shorter cases only)
        if (c.s.n_kv > 1024) {
            continue;
        }
        ggml_tensor * node = run.out_ref;
        CHECK(kvh_steps[0]->supports(node) && kvh_steps[1]->supports(node));
        std::vector<float> direct[2];
        for (int k = 0; k < 2; k++) {
            const uint32_t     nth = k == 0 ? 3 : 5;
            const size_t       ws  = kvh_steps[0]->workspace_size(node);
            std::vector<float> workspace(ws / sizeof(float), NAN);
            float *            out = static_cast<float *>(node->data);
            std::fill(out, out + n, NAN);
            bool ok = true;
            for (int step = 0; step < 2; step++) {
                for (uint32_t ith = 0; ith < nth; ith++) {
                    const spacemit_tile tile = { ith, nth, nullptr, 0, workspace.data(), ws, nullptr };
                    ok &= kvh_steps[step]->fn(&tile, node);
                }
            }
            CHECK(ok);
            direct[k].assign(out, out + n);
            const attn_error  ed = attn_errors(direct[k].data(), ref.data(), n, run.out->ne[0]);
            const std::string wd = std::string("KV-head split with ") + std::to_string(nth) + " tiles, " + c.what;
            attn_report(ed, wd.c_str(), c.s);
        }
        const bool same = std::memcmp(direct[0].data(), direct[1].data(), n * sizeof(float)) == 0;
        if (!same) {
            std::fprintf(stderr, "FAIL: attention KV-head split, %s: 3 and 5 tiles differ\n", c.what);
            g_failures++;
        }
        n_kvh_direct++;
        n_kvh_direct_same += same;
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
        reference_label());
    std::printf("attn     split over the tiles (the default; FLAGOS_SPACEMIT_ATTN=split without the turns) on %d "
                "one-token cases: max NMSE %.2e (bound %.0e), worst row %.2e (bound %.0e); %d/%d outputs differ from "
                "the one-chunk kernels' (the split ran); with and without the turns identical in %d/%zu cases; one "
                "launch per step identical in %d/%zu (E3a)\n",
                n_split, e_split, k_attn_max_nmse, e_split_row, k_attn_max_row_nmse, n_split_differ, n_split,
                n_turns_same, sizeof(split_cases) / sizeof(split_cases[0]), n_launch_same,
                sizeof(split_cases) / sizeof(split_cases[0]));
    std::printf("attn     KV-head split for a few tokens (FLAGOS_SPACEMIT_ATTN_KVH=1-8) on %zu cases (1-8 tokens, n_kv "
                "256-4096, GQA 2, 4 and 8, 1, 2 and 4 chunks, sinks, softcap, no mask, every cell hidden): max NMSE "
                "%.2e (bound %.0e), worst row %.2e (bound %.0e); its contract taken in %d/%zu; %d/%zu outputs differ "
                "from those under off; two runs identical in %d/%zu; 3 and 5 tiles called directly identical in "
                "%d/%d%s\n",
                kvh_cases.size(), e_kvh, k_attn_max_nmse, e_kvh_row, k_attn_max_row_nmse, n_kvh_taken,
                kvh_cases.size(), n_kvh_differ, kvh_cases.size(), n_kvh_repeat, kvh_cases.size(), n_kvh_direct_same,
                n_kvh_direct, reference_label());
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
    // named as llama.cpp names a layer's weights, which the op counters read (GGML_FLAGOS_OP_STATS)
    const std::pair<ggml_tensor *, const char *> names[] = {
        { w.attn_norm, "attn_norm" }, { w.q_norm, "attn_q_norm" }, { w.k_norm, "attn_k_norm" }, { w.ffn_norm, "ffn_norm" },
        { w.wq, "attn_q" },           { w.wk, "attn_k" },          { w.wv, "attn_v" },          { w.wo, "attn_output" },
        { w.wg, "ffn_gate" },         { w.wu, "ffn_up" },          { w.wd, "ffn_down" },
    };
    for (const auto & [t, name] : names) {
        ggml_format_name(t, "blk.0.%s.weight", name);
    }
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

// the CPU computes with the weights the provider holds: the same bytes. A Q4_1 weight stored as Q8_0 reads back only
// approximately; layer_nmse gives the CPU its GGUF bytes instead
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
// and k, the KV writes, then the attention and the rest. freq_base: RoPE's base (check_plan's probe (c) changes it)
static layer_out build_layer(ggml_context * ctx, ggml_cgraph * gf, const layer_weights & w, ggml_tensor * x,
                             ggml_tensor * pos, ggml_tensor * kv_idx, ggml_tensor * out_ids, ggml_tensor * k_cache,
                             ggml_tensor * v_cache, ggml_tensor * mask, int64_t n_tokens, float freq_base = 1e6f) {
    const float   eps = 1e-6f;
    ggml_tensor * h   = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w.attn_norm);
    ggml_tensor * q   = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w.wq, h), k_hd, k_nh, n_tokens);
    q                 = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps), w.q_norm);
    q = ggml_rope_ext(ctx, q, pos, nullptr, k_hd, GGML_ROPE_TYPE_NEOX, 0, freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w.wk, h), k_hd, k_nkv, n_tokens);
    k               = ggml_mul(ctx, ggml_rms_norm(ctx, k, eps), w.k_norm);
    k = ggml_rope_ext(ctx, k, pos, nullptr, k_hd, GGML_ROPE_TYPE_NEOX, 0, freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
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
    for (ggml_tensor * c : { kc_d, kc_c }) {
        ggml_set_name(c, "cache_k_l0");
    }
    for (ggml_tensor * c : { vc_d, vc_c }) {
        ggml_set_name(c, "cache_v_l0");
    }

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
    if (stored_as_q8_0(down)) {
        set_layer_weights(wc);  // the same GGUF bytes (fixed seeds): the bound covers the requantization's error too
    } else {
        copy_layer_weights(wd, wc);
    }

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

//
// D3 (IGPU): llama.cpp's KV cache in the provider's buffer, written by the provider (the cache writes' SET_ROWS, the
// context shift's ROPE in place and its CPY back into a quantized cache), against the CPU backend
//

// the cache types besides F16 and F32 that the provider writes from F32 rows (spacemit_cache_block_type)
static const ggml_type k_cache_block_types[] = { GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,   GGML_TYPE_Q4_1, GGML_TYPE_Q5_0,
                                                  GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL, GGML_TYPE_BF16 };

// the types whose quantizer in ggml-cpu is ggml's reference one (ggml-cpu/quants.c), so the CPU's bytes must be the
// provider's; ggml-cpu's Q8_0 rounds ties to even on some hosts (ggml-cpu/arch/arm/quants.c), the reference away from 0
static bool cpu_quantizes_as_ref(ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1 ||
           type == GGML_TYPE_IQ4_NL;
}

constexpr double k_cache_max_nmse = 1e-6;

// values in type's bytes by ggml's reference conversion (F32: as they are); v.size() a multiple of the block size
static std::vector<uint8_t> to_type_ref(ggml_type type, const std::vector<float> & v) {
    std::vector<uint8_t> q(ggml_row_size(type, (int64_t) v.size()));
    if (type == GGML_TYPE_F32) {
        std::memcpy(q.data(), v.data(), q.size());
    } else {
        ggml_get_type_traits(type)->from_float_ref(v.data(), q.data(), (int64_t) v.size());
    }
    return q;
}

// the values of bytes of type (F32: as they are)
static std::vector<float> type_values(ggml_type type, const std::vector<uint8_t> & bytes) {
    if (type != GGML_TYPE_F32) {
        return dequantize(type, bytes);
    }
    std::vector<float> v(bytes.size() / sizeof(float));
    std::memcpy(v.data(), bytes.data(), bytes.size());
    return v;
}

// one op that writes a cache, on the provider and on the CPU backend: the cache [ne] of type twice, in the provider's
// buffer and in a CPU buffer, both named name and holding init. cache_k_l0 is a KV cache by the layout rule, which keeps
// it in ggml's layout where its type and shape would make it a matmul weight. build makes the op on a cache, after its
// allocation, from inputs the caller keeps in a CPU buffer. Returns the NMSE of the provider's cache against the CPU's,
// both read back with ggml_backend_tensor_get; -1 when the provider refuses the op or a graph fails. out_d and out_c
// get the caches' bytes
static double cache_op_nmse(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu, ggml_type type,
                            const std::array<int64_t, 4> & ne, const std::vector<uint8_t> & init,
                            const std::function<ggml_tensor *(ggml_context *, ggml_tensor *)> & build,
                            std::vector<uint8_t> * out_d = nullptr, std::vector<uint8_t> * out_c = nullptr,
                            const char * name = "cache_k_l0") {
    ggml_init_params params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context *   ctx_d  = ggml_init(params);  // the provider's cache: its buffer
    ggml_context *   ctx_c  = ggml_init(params);  // the CPU's cache: a CPU buffer
    ggml_context *   ctx_g  = ggml_init(params);  // the ops and the graphs
    ggml_tensor *    c_d    = ggml_new_tensor(ctx_d, type, 4, ne.data());
    ggml_tensor *    c_c    = ggml_new_tensor(ctx_c, type, 4, ne.data());
    ggml_set_name(c_d, name);
    ggml_set_name(c_c, name);
    ggml_backend_buffer_t b_d = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, ggml_backend_dev_buffer_type(dev));
    ggml_backend_buffer_t b_c = ggml_backend_alloc_ctx_tensors_from_buft(ctx_c, ggml_backend_cpu_buffer_type());
    double                e   = -1.0;
    if (b_d != nullptr && b_c != nullptr && init.size() == ggml_nbytes(c_d)) {
        ggml_backend_tensor_set(c_d, init.data(), 0, init.size());
        ggml_backend_tensor_set(c_c, init.data(), 0, init.size());
        ggml_tensor * o_d = build(ctx_g, c_d);
        ggml_tensor * o_c = build(ctx_g, c_c);
        ggml_cgraph * g_d = ggml_new_graph(ctx_g);
        ggml_cgraph * g_c = ggml_new_graph(ctx_g);
        ggml_build_forward_expand(g_d, o_d);
        ggml_build_forward_expand(g_c, o_c);
        if (ggml_backend_dev_supports_op(dev, o_d) && ggml_backend_graph_compute(backend, g_d) == GGML_STATUS_SUCCESS &&
            ggml_backend_graph_compute(cpu, g_c) == GGML_STATUS_SUCCESS) {
            std::vector<uint8_t> a(ggml_nbytes(c_d)), r(ggml_nbytes(c_c));
            ggml_backend_tensor_get(c_d, a.data(), 0, a.size());
            ggml_backend_tensor_get(c_c, r.data(), 0, r.size());
            const std::vector<float> va = type_values(type, a), vr = type_values(type, r);
            e = nmse(va.data(), vr.data(), (int64_t) va.size());
            if (out_d != nullptr) {
                *out_d = std::move(a);
            }
            if (out_c != nullptr) {
                *out_c = std::move(r);
            }
        }
    }
    ggml_backend_buffer_free(b_d);
    ggml_backend_buffer_free(b_c);
    ggml_free(ctx_d);
    ggml_free(ctx_c);
    ggml_free(ctx_g);
    return e;
}

// SET_ROWS on the host with ggml's reference quantizer, in ggml-cpu's order (the last of repeated rows wins): row i of
// plane (i2, i3) of src (F32, dense [ne0, nr, ne2, ne3]) into row idx[i, i2 % ni1, i3 % ni2] (dense [nr, ni1, ni2]) of
// the cache (bytes of [ne0, ne1, ne2, ne3] in a block type)
static void set_rows_host(ggml_type type, std::vector<uint8_t> & cache, const std::array<int64_t, 4> & ne,
                          const std::vector<float> & src, int64_t nr, const std::vector<int64_t> & idx, int64_t ni1,
                          int64_t ni2) {
    const size_t row = ggml_row_size(type, ne[0]);
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i = 0; i < nr; i++) {
                const int64_t r = idx[i + nr * ((i2 % ni1) + ni1 * (i3 % ni2))];
                ggml_get_type_traits(type)->from_float_ref(src.data() + ne[0] * (i + nr * (i2 + ne[2] * i3)),
                                                           cache.data() + row * (r + ne[1] * (i2 + ne[2] * i3)), ne[0]);
            }
        }
    }
}

// what a cache_set_rows case gave: the NMSE against the CPU (-1: refused or failed), whether the provider's bytes are
// the CPU's and those of ggml's reference quantizer applied on the host (set_rows_host)
struct cache_rows_result {
    double e         = -1.0;
    bool   exact_cpu = false;
    bool   exact_ref = false;
};

// SET_ROWS of F32 rows [ne0, nr, ne2, ne3] into a cache [ne] of a block type named name, by indices idx (dense
// [nr, ni1, ni2], I64 or I32), as cache_op_nmse; with strided the rows are every other row of a source with twice the
// rows, and the indices every other plane of an index tensor with twice the planes (nb1 doubled in both), whose other
// rows and indices (out of range) the op must not read
static cache_rows_result cache_set_rows(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu,
                                        ggml_type type, const std::array<int64_t, 4> & ne, int64_t nr,
                                        const std::vector<int64_t> & idx, int64_t ni1, int64_t ni2, ggml_type idx_type,
                                        bool strided, uint32_t seed, const char * name = "cache_k_l0") {
    const int64_t    s      = strided ? 2 : 1;
    ggml_init_params params = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context *   ctx_h  = ggml_init(params);
    ggml_tensor *    src_a  = ggml_new_tensor_4d(ctx_h, GGML_TYPE_F32, ne[0], nr * s, ne[2], ne[3]);
    ggml_tensor *    idx_a  = ggml_new_tensor_3d(ctx_h, idx_type, nr, ni1 * s, ni2);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
    cache_rows_result     res;
    if (buf != nullptr) {
        ggml_tensor * src = strided ? ggml_view_4d(ctx_h, src_a, ne[0], nr, ne[2], ne[3], 2 * src_a->nb[1], src_a->nb[2],
                                                   src_a->nb[3], 0) :
                                      src_a;
        ggml_tensor * ix  = strided ? ggml_view_3d(ctx_h, idx_a, nr, ni1, ni2, 2 * idx_a->nb[1], idx_a->nb[2], 0) : idx_a;

        const std::vector<float> sv = random_values(ggml_nelements(src_a), seed, 1.0f);
        std::vector<float>       dense(ne[0] * nr * ne[2] * ne[3]);  // the rows the op takes
        for (int64_t p = 0; p < ne[2] * ne[3]; p++) {
            for (int64_t i = 0; i < nr; i++) {
                std::memcpy(dense.data() + ne[0] * (i + nr * p), sv.data() + ne[0] * (i * s + nr * s * p),
                            ne[0] * sizeof(float));
            }
        }
        std::vector<int64_t> iv(ggml_nelements(idx_a), ne[1] + 7);  // out of range where the op must not read
        for (int64_t k = 0; k < ni2; k++) {
            for (int64_t j = 0; j < ni1; j++) {
                for (int64_t i = 0; i < nr; i++) {
                    iv[i + nr * (j * s + ni1 * s * k)] = idx[i + nr * (j + ni1 * k)];
                }
            }
        }
        ggml_backend_tensor_set(src_a, sv.data(), 0, ggml_nbytes(src_a));
        if (idx_type == GGML_TYPE_I64) {
            ggml_backend_tensor_set(idx_a, iv.data(), 0, ggml_nbytes(idx_a));
        } else {
            std::vector<int32_t> i32(iv.begin(), iv.end());
            ggml_backend_tensor_set(idx_a, i32.data(), 0, ggml_nbytes(idx_a));
        }

        const std::vector<uint8_t> init = to_type_ref(type, random_values(ne[0] * ne[1] * ne[2] * ne[3], seed + 1, 1.0f));
        std::vector<uint8_t>       out_d, out_c;
        res.e = cache_op_nmse(
            dev, backend, cpu, type, ne, init,
            [&](ggml_context * ctx, ggml_tensor * cache) { return ggml_set_rows(ctx, cache, src, ix); }, &out_d, &out_c,
            name);
        if (res.e >= 0.0) {
            std::vector<uint8_t> expect = init;
            set_rows_host(type, expect, ne, dense, nr, idx, ni1, ni2);
            res.exact_cpu = out_d == out_c;
            res.exact_ref = out_d == expect;
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx_h);
    return res;
}

// the context shift on a Q8_0 cache as llama.cpp builds it, without the Hadamard rotation (src/llama-kv-cache.cpp:
// 1863-1877): a view [128, 8, 256] of the cache [1024, 256] cast to F32, rotated (NEOX) and copied back, through
// ggml_backend_sched with the provider and the CPU, the cache in the provider's buffer; against a scheduler with only
// the CPU on a cache in a CPU buffer. The copy back writes our buffer, so the scheduler must give it to the provider
// (else it aborts); the cast the provider refuses, so the CPU gets a copy of the view. Returns the NMSE of the caches,
// -1 when a graph fails; on_provider: the scheduler gave the copy back to the provider; n_splits: its splits
static double shift_sched_nmse(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu, bool & on_provider,
                               int & n_splits) {
    const ggml_type  type    = GGML_TYPE_Q8_0;
    ggml_init_params params  = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_init_params gparams = { ggml_tensor_overhead() * 32 + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context *   ctx_d   = ggml_init(params);  // the provider's cache: its buffer
    ggml_context *   ctx_c   = ggml_init(params);  // the CPU's cache and the positions: a CPU buffer
    ggml_context *   ctx_g   = ggml_init(gparams);
    ggml_tensor *    c_d     = ggml_new_tensor_2d(ctx_d, type, 1024, 256);
    ggml_tensor *    c_c     = ggml_new_tensor_2d(ctx_c, type, 1024, 256);
    ggml_tensor *    pos     = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, 256);
    ggml_set_name(c_d, "cache_k_l0");
    ggml_set_name(c_c, "cache_k_l0");
    ggml_set_name(pos, "k_shift");
    ggml_backend_buffer_t b_d = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, ggml_backend_dev_buffer_type(dev));
    ggml_backend_buffer_t b_c = ggml_backend_alloc_ctx_tensors_from_buft(ctx_c, ggml_backend_cpu_buffer_type());
    double                e   = -1.0;
    on_provider               = false;
    n_splits                  = 0;
    if (b_d != nullptr && b_c != nullptr) {
        const std::vector<uint8_t> init = to_type_ref(type, random_values(1024 * 256, 41, 1.0f));
        const std::vector<float>   pf   = random_values(256, 42, 1.0f);
        std::vector<int32_t>       pv(256);
        for (size_t i = 0; i < pv.size(); i++) {
            pv[i] = std::min(63, (int32_t) ((pf[i] + 1.0f) * 32.0f));  // a shift in [0, 64)
        }
        ggml_backend_tensor_set(c_d, init.data(), 0, init.size());
        ggml_backend_tensor_set(c_c, init.data(), 0, init.size());
        ggml_backend_tensor_set(pos, pv.data(), 0, ggml_nbytes(pos));
        const auto build = [&](ggml_tensor * cache) {
            ggml_tensor * view =
                ggml_view_3d(ctx_g, cache, 128, 8, 256, ggml_row_size(type, 128), ggml_row_size(type, 1024), 0);
            ggml_tensor * tmp = ggml_cast(ctx_g, view, GGML_TYPE_F32);
            tmp = ggml_rope_ext(ctx_g, tmp, pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                                0.0f);
            ggml_tensor * out = ggml_cpy(ctx_g, tmp, view);
            ggml_set_output(out);
            return out;
        };
        ggml_tensor * o_d = build(c_d);
        ggml_tensor * o_c = build(c_c);
        ggml_cgraph * g_d = ggml_new_graph(ctx_g);
        ggml_cgraph * g_c = ggml_new_graph(ctx_g);
        ggml_build_forward_expand(g_d, o_d);
        ggml_build_forward_expand(g_c, o_c);

        ggml_backend_t       both[2] = { backend, cpu };
        ggml_backend_sched_t s_d     = ggml_backend_sched_new(both, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
        ggml_backend_sched_t s_c     = ggml_backend_sched_new(&cpu, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
        if (s_d != nullptr && s_c != nullptr && ggml_backend_sched_graph_compute(s_d, g_d) == GGML_STATUS_SUCCESS &&
            ggml_backend_sched_graph_compute(s_c, g_c) == GGML_STATUS_SUCCESS) {
            on_provider = ggml_backend_sched_get_tensor_backend(s_d, o_d) == backend;
            n_splits    = ggml_backend_sched_get_n_splits(s_d);
            std::vector<uint8_t> a(ggml_nbytes(c_d)), r(ggml_nbytes(c_c));
            ggml_backend_tensor_get(c_d, a.data(), 0, a.size());
            ggml_backend_tensor_get(c_c, r.data(), 0, r.size());
            const std::vector<float> va = dequantize(type, a), vr = dequantize(type, r);
            e = nmse(va.data(), vr.data(), (int64_t) va.size());
        }
        ggml_backend_sched_free(s_d);
        ggml_backend_sched_free(s_c);
    }
    ggml_backend_buffer_free(b_d);
    ggml_backend_buffer_free(b_c);
    ggml_free(ctx_d);
    ggml_free(ctx_c);
    ggml_free(ctx_g);
    return e;
}

// the KV cache in the provider's buffer, as llama.cpp places it for an IGPU device: every op that writes it claimed
// exactly and computed as the CPU does. Caches of 256 cells (a multiple of 32, so only the name keeps a quantized one
// in ggml's layout), [1024, 256] but where noted, named cache_k_l0, from random bytes; inputs in a CPU buffer; the CPU
// backend on one thread, whose SET_ROWS keeps the order of repeated indices only so. The attention over quantized K and
// V stays unclaimed (check_flash_attn)
static int check_cache_ops(ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);

    const std::array<int64_t, 4> cache_ne = { 1024, 256, 1, 1 };
    double                       worst_set = 0.0, worst_cpy = 0.0, worst_rope = 0.0, rope_max_abs = 0.0;
    int                          n_ref = 0, n_ref_ok = 0, n_cpu = 0, n_cpu_ok = 0;
    uint32_t                     seed = 300;

    // a SET_ROWS case: within the bound, the bytes of ggml's reference quantizer, and the CPU's where it runs that one
    const auto rows_case = [&](const char * what, ggml_type type, const cache_rows_result & r, bool exact_cpu) {
        n_ref++;
        n_ref_ok += r.exact_ref;
        n_cpu += exact_cpu;
        n_cpu_ok += exact_cpu && r.exact_cpu;
        if (r.e < 0.0) {
            std::fprintf(stderr, "FAIL: SET_ROWS %s into a %s cache: refused or failed\n", what, ggml_type_name(type));
            g_failures++;
        } else if (r.e > k_cache_max_nmse || !r.exact_ref || (exact_cpu && !r.exact_cpu)) {
            std::fprintf(stderr, "FAIL: SET_ROWS %s into a %s cache: NMSE %.3e; bytes %s ggml's reference quantizer%s\n",
                         what, ggml_type_name(type), r.e, r.exact_ref ? "equal to" : "differ from",
                         exact_cpu ? (r.exact_cpu ? ", equal to the CPU's" : ", differ from the CPU's") : "");
            g_failures++;
        }
        worst_set = std::max(worst_set, r.e);
    };

    // a. 1, 7 and 100 rows of 1024 into every block type, at cells 2r + 1; one case with I32 indices
    for (ggml_type type : k_cache_block_types) {
        for (int64_t nr : { 1, 7, 100 }) {
            std::vector<int64_t> idx(nr);
            for (int64_t r = 0; r < nr; r++) {
                idx[r] = (2 * r + 1) % cache_ne[1];
            }
            char what[64];
            std::snprintf(what, sizeof(what), "%lld rows", (long long) nr);
            rows_case(what, type,
                      cache_set_rows(dev, backend, cpu, type, cache_ne, nr, idx, 1, 1, GGML_TYPE_I64, false, seed++),
                      cpu_quantizes_as_ref(type));
        }
    }
    {
        std::vector<int64_t> idx(7);
        for (int64_t r = 0; r < 7; r++) {
            idx[r] = 2 * r + 1;
        }
        rows_case("7 rows, I32 indices", GGML_TYPE_Q8_0,
                  cache_set_rows(dev, backend, cpu, GGML_TYPE_Q8_0, cache_ne, 7, idx, 1, 1, GGML_TYPE_I32, false, seed++),
                  false);
    }
    // b. repeated indices: the last row wins
    rows_case("3 rows at cells 3, 3, 5", GGML_TYPE_Q8_0,
              cache_set_rows(dev, backend, cpu, GGML_TYPE_Q8_0, cache_ne, 3, { 3, 3, 5 }, 1, 1, GGML_TYPE_I64, false,
                             seed++),
              false);
    // c. 4-D: rows [1024, 5, 2, 3] into a cache [1024, 64, 2, 3], indices for every plane [5, 2, 3] and broadcast over
    // the planes [5, 1, 1]; then strided rows and index planes (nb1 doubled)
    {
        const std::array<int64_t, 4> ne4 = { 1024, 64, 2, 3 };
        std::vector<int64_t>         idx(5 * 2 * 3), idx_b(5);
        for (int64_t k = 0; k < 3; k++) {
            for (int64_t j = 0; j < 2; j++) {
                for (int64_t i = 0; i < 5; i++) {
                    idx[i + 5 * (j + 2 * k)] = (2 * i + 1 + 7 * j + 13 * k) % ne4[1];
                }
            }
        }
        for (int64_t i = 0; i < 5; i++) {
            idx_b[i] = 3 * i + 2;
        }
        rows_case("4-D [1024, 5, 2, 3], indices [5, 2, 3]", GGML_TYPE_Q8_0,
                  cache_set_rows(dev, backend, cpu, GGML_TYPE_Q8_0, ne4, 5, idx, 2, 3, GGML_TYPE_I64, false, seed++),
                  false);
        rows_case("4-D [1024, 5, 2, 3], indices [5, 1, 1]", GGML_TYPE_Q8_0,
                  cache_set_rows(dev, backend, cpu, GGML_TYPE_Q8_0, ne4, 5, idx_b, 1, 1, GGML_TYPE_I64, false, seed++),
                  false);
        rows_case("4-D strided rows and indices", GGML_TYPE_Q8_0,
                  cache_set_rows(dev, backend, cpu, GGML_TYPE_Q8_0, ne4, 5, idx, 2, 3, GGML_TYPE_I64, true, seed++),
                  false);
    }
    // d. refused: F16 rows into a quantized cache; a Q8_0 [1024, 64] not named as a cache is a matmul weight by the
    // layout rule (refused allocated in our buffer too), and named cache_k_l0 it is a cache like the others
    {
        ggml_init_params params = { ggml_tensor_overhead() * 32, nullptr, true };
        ggml_context *   ctx    = ggml_init(params);
        REQUIRE(ctx != nullptr);
        ggml_tensor * cache  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 1024, 256);
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 1024, 64);
        ggml_tensor * rows16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 1024, 7);
        ggml_tensor * rows32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 7);
        ggml_tensor * idx    = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 7);
        ggml_set_name(cache, "cache_k_l0");
        ggml_set_name(weight, "w");
        CHECK(ggml_backend_dev_supports_op(dev, ggml_set_rows(ctx, cache, rows32, idx)));
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_set_rows(ctx, cache, rows16, idx)));
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_set_rows(ctx, weight, rows32, idx)));
        // CPY: F32 into a contiguous view of the cache; not into a non-contiguous one, not from F16
        const size_t  r128 = ggml_row_size(GGML_TYPE_Q8_0, 128), r1024 = ggml_row_size(GGML_TYPE_Q8_0, 1024);
        ggml_tensor * view = ggml_view_3d(ctx, cache, 128, 8, 256, r128, r1024, 0);
        ggml_tensor * part = ggml_view_3d(ctx, cache, 64, 8, 256, r128, r1024, 0);
        CHECK(ggml_backend_dev_supports_op(dev, ggml_cpy(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 8, 256), view)));
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_cpy(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 8, 256), part)));
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_cpy(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, 8, 256), view)));
        ggml_free(ctx);
        std::vector<int64_t> idx7(7);
        for (int64_t r = 0; r < 7; r++) {
            idx7[r] = 9 * r;
        }
        const std::array<int64_t, 4> ne_w = { 1024, 64, 1, 1 };
        const cache_rows_result      w    = cache_set_rows(dev, backend, cpu, GGML_TYPE_Q8_0, ne_w, 7, idx7, 1, 1,
                                                           GGML_TYPE_I64, false, seed++, "w");
        if (w.e >= 0.0) {
            std::fprintf(stderr, "FAIL: SET_ROWS into a Q8_0 [1024, 64] named w (a weight) was claimed\n");
            g_failures++;
        }
        rows_case("7 rows into [1024, 64]", GGML_TYPE_Q8_0,
                  cache_set_rows(dev, backend, cpu, GGML_TYPE_Q8_0, ne_w, 7, idx7, 1, 1, GGML_TYPE_I64, false, seed++),
                  false);
    }

    // e. CPY of F32 rows into the shift's view [128, 8, 256] of a Q8_0 and of a Q4_0 cache (Q4_0: the CPU's bytes),
    // from contiguous and from strided rows; and Q4_0 [96] (3 blocks), test-backend-ops' case of a block count not a
    // multiple of the kernels'
    {
        ggml_init_params params = { ggml_tensor_overhead() * 8, nullptr, true };
        ggml_context *   ctx_h  = ggml_init(params);
        REQUIRE(ctx_h != nullptr);
        ggml_tensor *         x    = ggml_new_tensor_3d(ctx_h, GGML_TYPE_F32, 128, 8, 256);
        ggml_tensor *         x2   = ggml_new_tensor_3d(ctx_h, GGML_TYPE_F32, 128, 16, 256);
        ggml_tensor *         x96  = ggml_new_tensor_1d(ctx_h, GGML_TYPE_F32, 96);
        ggml_backend_buffer_t buf  = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
        REQUIRE(buf != nullptr);
        ggml_tensor * xs = ggml_view_3d(ctx_h, x2, 128, 8, 256, 2 * x2->nb[1], x2->nb[2], 0);  // every other row
        for (ggml_tensor * t : { x, x2, x96 }) {
            const std::vector<float> v = random_values(ggml_nelements(t), seed++, 1.0f);
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        }
        const auto cpy_case = [&](const char * what, ggml_type type, const std::array<int64_t, 4> & ne,
                                  ggml_tensor * src, bool view) {
            std::vector<uint8_t> out_d, out_c;
            const double         e = cache_op_nmse(
                dev, backend, cpu, type, ne,
                to_type_ref(type, random_values(ne[0] * ne[1] * ne[2] * ne[3], seed++, 1.0f)),
                [&](ggml_context * ctx, ggml_tensor * cache) {
                    ggml_tensor * dst = view ? ggml_view_3d(ctx, cache, 128, 8, 256, ggml_row_size(type, 128),
                                                            ggml_row_size(type, 1024), 0) :
                                               cache;
                    return ggml_cpy(ctx, src, dst);
                },
                &out_d, &out_c);
            const bool exact = !cpu_quantizes_as_ref(type) || out_d == out_c;
            if (e < 0.0 || e > k_cache_max_nmse || !exact) {
                std::fprintf(stderr, "FAIL: CPY %s into a %s cache: %s %.3e%s\n", what, ggml_type_name(type),
                             e < 0.0 ? "refused or failed" : "NMSE", e, exact ? "" : "; bytes differ from the CPU's");
                g_failures++;
            }
            worst_cpy = std::max(worst_cpy, e);
        };
        cpy_case("F32 [128, 8, 256]", GGML_TYPE_Q8_0, cache_ne, x, true);
        cpy_case("F32 [128, 8, 256]", GGML_TYPE_Q4_0, cache_ne, x, true);
        cpy_case("strided F32 rows", GGML_TYPE_Q8_0, cache_ne, xs, true);
        cpy_case("F32 [96]", GGML_TYPE_Q4_0, { 96, 1, 1, 1 }, x96, false);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx_h);
    }

    // f. ROPE in place on the shift's view [128, 8, 256] of an F16 and of an F32 cache (on the K3 F32 takes the RVV
    // kernel), NORMAL and NEOX, with and without frequency factors, positions in [0, 64)
    {
        ggml_init_params params = { ggml_tensor_overhead() * 4, nullptr, true };
        ggml_context *   ctx_h  = ggml_init(params);
        REQUIRE(ctx_h != nullptr);
        ggml_tensor *         pos = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, 256);
        ggml_tensor *         ff  = ggml_new_tensor_1d(ctx_h, GGML_TYPE_F32, 64);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
        REQUIRE(buf != nullptr);
        const std::vector<float> pf = random_values(256, seed++, 1.0f);
        std::vector<int32_t>     pv(256);
        for (size_t i = 0; i < pv.size(); i++) {
            pv[i] = std::min(63, (int32_t) ((pf[i] + 1.0f) * 32.0f));
        }
        std::vector<float> fv = random_values(64, seed++, 0.1f);
        for (float & f : fv) {
            f += 1.0f;  // [0.9, 1.1), as test-backend-ops
        }
        ggml_backend_tensor_set(pos, pv.data(), 0, ggml_nbytes(pos));
        ggml_backend_tensor_set(ff, fv.data(), 0, ggml_nbytes(ff));
        for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
            for (int mode : { GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX }) {
                for (bool with_ff : { false, true }) {
                    std::vector<uint8_t> out_d, out_c;
                    const double         e = cache_op_nmse(
                        dev, backend, cpu, type, cache_ne, to_type_ref(type, random_values(1024 * 256, seed++, 1.0f)),
                        [&](ggml_context * ctx, ggml_tensor * cache) {
                            ggml_tensor * view = ggml_view_3d(ctx, cache, 128, 8, 256, ggml_row_size(type, 128),
                                                              ggml_row_size(type, 1024), 0);
                            return ggml_rope_ext_inplace(ctx, view, pos, with_ff ? ff : nullptr, 128, mode, 0, 10000.0f,
                                                         1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                        },
                        &out_d, &out_c);
                    if (e >= 0.0) {
                        const std::vector<float> a = type_values(type, out_d), r = type_values(type, out_c);
                        for (size_t i = 0; i < a.size(); i++) {
                            rope_max_abs = std::max(rope_max_abs, (double) std::fabs(a[i] - r[i]));
                        }
                    }
                    if (e < 0.0 || e > k_cache_max_nmse) {
                        std::fprintf(stderr, "FAIL: ROPE %s in place on a %s cache view%s: %s %.3e\n",
                                     mode == GGML_ROPE_TYPE_NEOX ? "NEOX" : "NORMAL", ggml_type_name(type),
                                     with_ff ? " with frequency factors" : "", e < 0.0 ? "refused or failed" : "NMSE", e);
                        g_failures++;
                    }
                    worst_rope = std::max(worst_rope, e);
                }
            }
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx_h);
    }

    // g. the context shift on a Q8_0 cache through the scheduler
    bool         shift_here = false;
    int          n_splits   = 0;
    const double e_shift    = shift_sched_nmse(dev, backend, cpu, shift_here, n_splits);
    if (e_shift < 0.0 || e_shift > k_cache_max_nmse || !shift_here) {
        std::fprintf(stderr, "FAIL: context shift on a Q8_0 cache through the scheduler: %s %.3e%s\n",
                     e_shift < 0.0 ? "failed" : "NMSE", e_shift, shift_here ? "" : "; the copy back ran elsewhere");
        g_failures++;
    }

    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    std::printf("cache    caches named cache_k_l0 in our buffer vs CPU: SET_ROWS of 1-100 F32 rows into Q8_0, Q4_0, Q4_1, "
                "Q5_0, Q5_1, IQ4_NL and BF16 (I64 and I32 indices, repeated, 4-D, broadcast, strided) max NMSE %.2e "
                "(bound %.0e), bytes as ggml's reference quantizers %d/%d, as the CPU's %d/%d; CPY of F32 rows into "
                "Q8_0 and Q4_0 max NMSE %.2e; ROPE in place on F16 and F32 views max NMSE %.2e, max abs difference "
                "%.2e; context shift on a Q8_0 cache through the scheduler (%d splits) NMSE %.2e; F16 rows into Q8_0, "
                "a Q8_0 weight, a non-contiguous or F16 CPY refused: checked\n",
                worst_set, k_cache_max_nmse, n_ref_ok, n_ref, n_cpu_ok, n_cpu, worst_cpy, worst_rope, rope_max_abs,
                n_splits, e_shift);
    return 0;
}

static int check_layer(ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);  // a correctness reference: single-threaded (PROJECT.md §6, item 28)

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
    // E3a: one launch per step (quantization and matmul, the norms, RoPE, the cache writes, attention, each launched on
    // its own), the host waiting for each, gives the same outputs as one launch for the layer; both settings given here,
    // whatever the run's
    int n_launch_same = 0, n_launch_cases = 0;  // l 0: one launch per split, l 1: GGML_FLAGOS_SYNC_LAUNCH=1
    for (ggml_type down : { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1 }) {
        for (int64_t n_tokens : { 1, 7 }) {
            std::vector<float> out[2];
            double             e[2];
            for (int l = 0; l < 2; l++) {
                env_override launch_switch("GGML_FLAGOS_SYNC_LAUNCH", l == 1 ? "1" : nullptr);
                e[l] = layer_nmse(dev, backend, cpu, n_tokens, down, false, &out[l]);
            }
            const bool same = e[0] >= 0.0 && e[1] >= 0.0 && out[0].size() == out[1].size() &&
                              std::memcmp(out[0].data(), out[1].data(), out[0].size() * sizeof(float)) == 0;
            if (!same) {
                std::fprintf(stderr, "FAIL: layer with %lld tokens, down projection %s: GGML_FLAGOS_SYNC_LAUNCH=1 %s\n",
                             (long long) n_tokens, ggml_type_name(down),
                             e[0] < 0.0 || e[1] < 0.0 ? "refused or failed" : "differs from one launch per layer");
                g_failures++;
            }
            n_launch_same += same;
            n_launch_cases++;
        }
    }
    // fusion-and-splitting.md §5, candidate 6: with FLAGOS_SPACEMIT_QUANTIZE=shared, Q, V and K, and gate and up, multiply
    // one quantization of their input: 3 steps fewer per layer, and the same bytes, in one launch and in one launch per
    // step (where the quantized activations outlast the launches between)
    int n_shared_same = 0, n_shared_fewer = 0, n_shared_cases = 0;
    for (ggml_type down : { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1 }) {
        for (int64_t n_tokens : { 1, 7 }) {
            // the second column is GGML_FLAGOS_SYNC_LAUNCH: unset one launch per split, 1 one per step
            const char * const settings[3][2] = { { "separate", nullptr }, { "shared", nullptr },
                                                  { "shared", "1" } };
            std::vector<float> out[3];
            double             e[3];
            uint64_t           steps[3];
            for (int s = 0; s < 3; s++) {
                env_override quantize_switch("FLAGOS_SPACEMIT_QUANTIZE", settings[s][0]);
                env_override launch_switch("GGML_FLAGOS_SYNC_LAUNCH", settings[s][1]);
                g_steps_take();
                e[s]     = layer_nmse(dev, backend, cpu, n_tokens, down, false, &out[s]);
                steps[s] = g_steps_take();
            }
            bool same = e[0] >= 0.0;
            for (int s = 1; s < 3; s++) {
                same = same && e[s] >= 0.0 && out[s].size() == out[0].size() &&
                       std::memcmp(out[0].data(), out[s].data(), out[0].size() * sizeof(float)) == 0;
            }
            const bool fewer = steps[1] + 3 == steps[0] && steps[2] == steps[1];
            if (!same || !fewer) {
                std::fprintf(stderr,
                             "FAIL: layer with %lld tokens, down projection %s: FLAGOS_SPACEMIT_QUANTIZE=shared %s; "
                             "steps separate %llu, shared %llu, shared one launch per step %llu\n",
                             (long long) n_tokens, ggml_type_name(down),
                             !same ? "refused, failed or differs from separate" : "has not 3 steps fewer",
                             (unsigned long long) steps[0], (unsigned long long) steps[1], (unsigned long long) steps[2]);
                g_failures++;
            }
            n_shared_same += same;
            n_shared_fewer += fewer;
            n_shared_cases++;
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
                "(E2) max NMSE %.2e; one launch per step identical in %d/%d cases (E3a); one quantization for Q/K/V "
                "and for gate/up (FLAGOS_SPACEMIT_QUANTIZE=shared) identical in %d/%d cases, 3 steps fewer in %d/%d%s\n",
                k_layer_pos, (long long) k_slots, worst, k_layer_max_nmse, e_rope, worst_kv, n_launch_same,
                n_launch_cases, n_shared_same, n_shared_cases, n_shared_fewer, n_shared_cases,
                reference_label());
    return 0;
}

//
// S1 steps 3, 3b and 3c (plan.md "S1 step 3 spec" §3, "S1 step 3b spec" §3, "S1 step 3c spec" §3): the plan cache's
// probes under FLAGOS_SPACEMIT_PLAN=none and uid-trust (the default)
//

// one Qwen3-like layer (build_layer, down projection Q4_0) built once on the provider and on the CPU, computed as often
// as a probe asks: the inputs, caches and weights of layer_nmse (fixed seeds), so two layers of one shape built apart
// hold the same values and the provider must give them the same bytes
struct plan_layer {
    ggml_context *        ctx_h = nullptr, * ctx_wd = nullptr, * ctx_wc = nullptr, * ctx_gd = nullptr, * ctx_gc = nullptr;
    ggml_backend_buffer_t bh = nullptr, bwd = nullptr, bwc = nullptr, bgd = nullptr, bgc = nullptr;
    ggml_cgraph *         gd = nullptr, * gc = nullptr;
    layer_out             od = {}, oc = {};
    layer_weights         wd = {};  // the provider's graph's weights
    bool                  cpu_done = false;

    // weights_on_cpu: the weights in a plain CPU buffer instead of the provider's (probe (j))
    bool init(ggml_backend_dev_t dev, int64_t n_tokens, float freq_base, bool weights_on_cpu = false) {
        ggml_init_params params = { ggml_tensor_overhead() * 128 + ggml_graph_overhead() * 2, nullptr, true };
        ctx_h                   = ggml_init(params);
        ctx_wd                  = ggml_init(params);
        ctx_wc                  = ggml_init(params);
        ctx_gd                  = ggml_init(params);
        ctx_gc                  = ggml_init(params);
        ggml_tensor * x         = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F32, k_d, n_tokens);
        ggml_tensor * pos       = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, n_tokens);
        ggml_tensor * kv_idx    = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I64, n_tokens);
        ggml_tensor * out_ids   = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, 1);
        ggml_tensor * kc_d      = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F16, k_hd * k_nkv, k_slots);
        ggml_tensor * vc_d      = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F16, k_hd * k_nkv, k_slots);
        ggml_tensor * kc_c      = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F16, k_hd * k_nkv, k_slots);
        ggml_tensor * vc_c      = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F16, k_hd * k_nkv, k_slots);
        ggml_tensor * mask      = ggml_new_tensor_4d(ctx_h, GGML_TYPE_F16, k_slots, n_tokens, 1, 1);
        wd                      = new_layer_weights(ctx_wd, GGML_TYPE_Q4_0);
        const layer_weights wc  = new_layer_weights(ctx_wc, GGML_TYPE_Q4_0);
        for (ggml_tensor * c : { kc_d, kc_c }) {
            ggml_set_name(c, "cache_k_l0");
        }
        for (ggml_tensor * c : { vc_d, vc_c }) {
            ggml_set_name(c, "cache_v_l0");
        }
        bh  = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
        bwd = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx_wd, weights_on_cpu ? ggml_backend_cpu_buffer_type() : ggml_backend_dev_buffer_type(dev));
        bwc = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wc, ggml_backend_cpu_buffer_type());
        if (bh == nullptr || bwd == nullptr || bwc == nullptr) {
            return false;
        }
        ggml_backend_buffer_clear(bh, 0);
        set_layer_weights(wd);
        copy_layer_weights(wd, wc);

        const std::vector<float> xv = random_values(k_d * n_tokens, 5, 1.0f);
        std::vector<int32_t>     pv(n_tokens);
        std::vector<int64_t>     iv(n_tokens);
        std::vector<int32_t>     cell_pos(k_slots, -1);  // -1: empty
        for (int64_t t = 0; t < n_tokens; t++) {
            pv[t]           = (int32_t) (k_layer_pos + t);
            iv[t]           = (3 + 5 * t) % k_slots;
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
        gd  = ggml_new_graph(ctx_gd);
        gc  = ggml_new_graph(ctx_gc);
        od  = build_layer(ctx_gd, gd, wd, x, pos, kv_idx, out_ids, kc_d, vc_d, mask, n_tokens, freq_base);
        oc  = build_layer(ctx_gc, gc, wc, x, pos, kv_idx, out_ids, kc_c, vc_c, mask, n_tokens, freq_base);
        bgd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_gd, ggml_backend_dev_buffer_type(dev));
        bgc = ggml_backend_alloc_ctx_tensors_from_buft(ctx_gc, ggml_backend_cpu_buffer_type());
        return bgd != nullptr && bgc != nullptr;
    }

    ~plan_layer() {
        for (ggml_backend_buffer_t b : { bh, bwd, bwc, bgd, bgc }) {
            ggml_backend_buffer_free(b);
        }
        for (ggml_context * c : { ctx_h, ctx_wd, ctx_wc, ctx_gd, ctx_gc }) {
            ggml_free(c);
        }
    }

    // the provider's layer output of graph g (default: gd), filled with NaN first, so steps bound to another graph's
    // nodes leave it NaN; false if the provider failed
    bool run(ggml_backend_t backend, std::vector<float> & out, ggml_cgraph * g = nullptr) {
        out.assign(ggml_nelements(od.full), NAN);
        ggml_backend_tensor_set(od.full, out.data(), 0, ggml_nbytes(od.full));
        if (ggml_backend_graph_compute(backend, g != nullptr ? g : gd) != GGML_STATUS_SUCCESS) {
            return false;
        }
        ggml_backend_tensor_get(od.full, out.data(), 0, ggml_nbytes(od.full));
        return true;
    }

    // out's NMSE against the CPU's layer (computed once); +inf if not finite
    double cpu_nmse(ggml_backend_t cpu, const std::vector<float> & out) {
        if (!cpu_done) {
            cpu_done = ggml_backend_graph_compute(cpu, gc) == GGML_STATUS_SUCCESS;
        }
        return cpu_done ? nmse(out.data(), static_cast<const float *>(oc.full->data), (int64_t) out.size()) : HUGE_VAL;
    }
};

static bool same_bytes(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// the key probe's stand-in kernels: never run, only compared as the cached steps a hit hands back
static bool plan_fake_kernel_1(const spacemit_tile *, ggml_tensor *) {
    return true;
}

static bool plan_fake_kernel_2(const spacemit_tile *, ggml_tensor *) {
    return true;
}

// probe (j)'s second part, the key on the cache itself through probe(), on g (stamped with a uid): two stand-in steps
// per node with an op stored with sizes 4096 and 8192 and no names, then asked with a copy of g, each case one clause
// of the key: (1) the same uid, nodes and word: a hit with the stored sizes and steps; (2) one node pointer replaced in
// a copy of the node array: a miss (the pointer walk); (3) one node fewer and (6) one node more (the first again): a
// miss each (the node count; without it (6) would walk only the stored nodes and hit, and (3) would read past the
// copy); (4) another word: a miss (the word); (5) another stamped uid with the same nodes: a miss (the uid's equality;
// "uid not 0" is (h)'s "the view again"); (7) the same uid, nodes and word asked with the op counters' names: a miss
// (an entry stored without names serves no graph that wants them). Returns false and reports when the cache does
// otherwise
static bool plan_key_probe(const ggml_cgraph * g) {
    ggml_cgraph *                     gm = const_cast<ggml_cgraph *>(g);
    const int                         n  = ggml_graph_n_nodes(gm);
    std::vector<spacemit_plugin_step> full;
    for (int i = 0; i < n; i++) {
        ggml_tensor * node = ggml_graph_node(gm, i);
        if (ggml_op_is_empty(node->op)) {
            continue;
        }
        full.push_back({ plan_fake_kernel_1, node, 0 });
        full.push_back({ plan_fake_kernel_2, node, 0 });
    }
    const uint64_t                    word = 2;
    std::vector<spacemit_plugin_step> steps;
    std::vector<const char *>         names;
    size_t                            scratch = 0, act = 0;
    bool                              ok = !full.empty() && g->uid != 0 && n > 1;
    if (!ok) {
        std::fprintf(stderr, "FAIL: plan key probe: the graph needs a uid and two nodes, one with an op\n");
        g_failures++;
        return false;
    }
    spacemit_plan_cache cache;
    cache.store(g, word, full, {}, 4096, 8192);
    // a copy of g whose nodes are `nodes` and whose uid is `uid`, asked through probe() with `w` (and the op counters'
    // names when want_names); a hit hands back the stored list and sizes
    const auto ask = [&](const char * what, std::vector<ggml_tensor *> nodes, uint64_t uid, uint64_t w,
                         spacemit_plan_result want, bool want_names = false) {
        ggml_cgraph copy = *g;
        copy.nodes       = nodes.data();
        copy.n_nodes     = static_cast<int>(nodes.size());
        copy.uid         = uid;
        scratch = act = 0;
        steps.clear();
        names.clear();
        const spacemit_plan_result got  = cache.probe(&copy, w, want_names, steps, names, scratch, act);
        bool                       good = got == want;
        if (got != spacemit_plan_result::miss) {
            good = good && scratch == 4096 && act == 8192 && steps.size() == full.size();
            for (size_t j = 0; good && j < steps.size(); j++) {
                good = steps[j].fn == full[j].fn && steps[j].node == full[j].node;
            }
        }
        if (!good) {
            std::fprintf(stderr,
                         "FAIL: plan key probe, %s: probe gave %d (1 a hit by uid, 2 a miss), expected %d; sizes %zu "
                         "and %zu, %zu steps\n",
                         what, (int) got, (int) want, scratch, act, steps.size());
            g_failures++;
        }
        ok = ok && good;
    };
    const std::vector<ggml_tensor *> same(g->nodes, g->nodes + n);
    std::vector<ggml_tensor *>       replaced = same;
    replaced[n / 2]                           = replaced[0];  // another of g's objects
    const std::vector<ggml_tensor *> fewer(g->nodes, g->nodes + n - 1);
    std::vector<ggml_tensor *>       more = same;
    more.push_back(same[0]);
    ask("(1) the same uid, nodes and word", same, g->uid, word, spacemit_plan_result::hit_uid);
    ask("(2) one node pointer replaced", replaced, g->uid, word, spacemit_plan_result::miss);
    ask("(3) one node fewer", fewer, g->uid, word, spacemit_plan_result::miss);
    ask("(4) another word", same, g->uid, word + 2, spacemit_plan_result::miss);
    ask("(5) another stamped uid with the same nodes", same, g->uid + 1, word, spacemit_plan_result::miss);
    ask("(6) one node more", more, g->uid, word, spacemit_plan_result::miss);
    ask("(7) the names wanted, none stored", same, g->uid, word, spacemit_plan_result::miss, true);
    return ok;
}

// The probes of plan.md "S1 design" §3 ("S1 step 3b spec" §3, "S1 step 3c spec" §3) under each setting of
// FLAGOS_SPACEMIT_PLAN, none and uid-trust, each on a fresh backend (an empty cache and workspace), with the counters
// read around every graph (hits by uid / misses under uid-trust): (a) one layer graph computed twice as the same object
// (stamped with a uid): a miss, then a hit; (e) the same structure built again as new objects with the same uid: a
// miss (the key's pointer walk); (b) the same structure as new objects, uid 0: a miss; (c) RoPE's base changed and (d)
// 7 tokens: new objects, uid 0, a miss each (the key reads neither op_params nor shapes: these two check the outputs,
// and (d) grows the workspace); (g) the first layer again after the larger graph grew the workspace: a miss (the core
// keeps one entry, the last plan), then its plan replayed in the grown workspace (a hit); (h) a graph view of it (uid
// 0): a miss, and a miss again when computed a second time, the entry then its own (a uid of 0 never hits); (f)
// FLAGOS_SPACEMIT_ATTN flipped between two one-token attention graphs (n_kv 1024, the graph stamped with its own uid):
// a miss each, the one-chunk graph a hit when computed again, the step list the other variant's (1 step one-chunk, 2
// the split), counted by the steps the executor ran; (l) the switch set again to the value it has: the generation moves
// and the contract stays, so a miss by the word alone, 1 step; (j) the first layer's structure with its weights in a
// CPU buffer (new objects under the same uid), right after the first layer is the entry: refused by the contract check,
// nothing counted, the first layer a hit after it; and the cache itself (plan_key_probe: the key alone through
// probe(), seven cases, under uid-trust); (o) a node unflagged after the store: one MUL_MAT of the first layer's
// object loses GGML_TENSOR_FLAG_COMPUTE, so under none the check's filter skips it (2 steps fewer) and under uid-trust
// the hit runs the stored steps (as many as before): the hit skips the check; (m) the first layer's own object with
// another uid, the entry its own: a miss by the uid alone; (k) under FLAGOS_SPACEMIT_QUANTIZE=shared no cache, so a
// layer graph twice counts nothing; (i) under none every counter stays 0; (n), made once before the settings and
// counted in each line: the switch unset gives uid-trust (the default), none gives none, and the old uid falls to
// uid-trust with one warning. Under uid-trust every hit's key time equals its lookup time (the counters' definition: a
// hit reads `checked` as `start`; that the hit skips the check is (o)). Every output against the CPU, the layers'
// against each other byte for byte, and every output of uid-trust against none's byte for byte. 15 probes under none,
// 14 under uid-trust
static int check_plan(ggml_backend_dev_t dev) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);  // a correctness reference: single-threaded (PROJECT.md §6, item 28)
    constexpr uint64_t k_uid = 0x504c414e00000001ULL;  // "PLAN", stamped as the scheduler stamps a split's graph
    std::map<std::string, std::vector<float>> none_outputs;  // none's outputs by name, the reference of the others

    constexpr uint64_t k_attn_uid = 0x504c414e00000002ULL;  // (f) and (l)'s attention graph's

    // (n) the default: the switch unset gives uid-trust, none gives none, the old uid gives uid-trust (each override
    // restores the switch after)
    bool default_ok = true;
    {
        env_override             unset("FLAGOS_SPACEMIT_PLAN", nullptr);
        const spacemit_plan_mode got = spacemit_plan_mode_get();
        if (got != spacemit_plan_mode::uid_trust) {
            std::fprintf(stderr, "FAIL: plan, probe (n): FLAGOS_SPACEMIT_PLAN unset gave mode %d, not uid-trust\n",
                         (int) got);
            g_failures++;
            default_ok = false;
        }
    }
    {
        env_override             none("FLAGOS_SPACEMIT_PLAN", "none");
        const spacemit_plan_mode got = spacemit_plan_mode_get();
        if (got != spacemit_plan_mode::none) {
            std::fprintf(stderr, "FAIL: plan, probe (n): FLAGOS_SPACEMIT_PLAN=none gave mode %d, not none\n", (int) got);
            g_failures++;
            default_ok = false;
        }
    }
    {
        // the setting removed in step 3c: unknown now, so the default, with one warning
        env_override             old("FLAGOS_SPACEMIT_PLAN", "uid");
        const spacemit_plan_mode got = spacemit_plan_mode_get();
        if (got != spacemit_plan_mode::uid_trust) {
            std::fprintf(stderr, "FAIL: plan, probe (n): FLAGOS_SPACEMIT_PLAN=uid gave mode %d, not uid-trust\n",
                         (int) got);
            g_failures++;
            default_ok = false;
        }
    }

    for (const char * setting : { "none", "uid-trust" }) {
        env_override   plan_switch("FLAGOS_SPACEMIT_PLAN", setting);
        env_override   quantize_switch("FLAGOS_SPACEMIT_QUANTIZE", nullptr);  // separate; (k) sets shared itself
        const bool     on      = std::strcmp(setting, "none") != 0;  // uid-trust: the cache on, a hit trusted
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        REQUIRE(backend != nullptr);
        spacemit_plan_counters_take();
        spacemit_plan_counters total;
        int                    n_graphs = 0, n_probes = 0, n_passed = 0, n_compared = 0;
        double                 worst_layer = 0.0, worst_attn = 0.0;

        // the counters of the graphs since the last call, added to the total
        const auto take = [&]() {
            const spacemit_plan_counters d = spacemit_plan_counters_take();
            total.hits_uid += d.hits_uid;
            total.misses += d.misses;
            total.lookup_us += d.lookup_us;
            total.key_us += d.key_us;
            return d;
        };
        // one graph's counters against what the probe expects with the cache on (all 0 under none); ran: whether the
        // graph ran (a refused one is not counted among the graphs)
        const auto expect = [&](const char * probe, uint64_t uid_hits, uint64_t misses, bool ran_graph = true) {
            n_graphs += ran_graph;
            const spacemit_plan_counters d = take();
            if (!on) {
                uid_hits = misses = 0;
            }
            bool ok = d.hits_uid == uid_hits && d.misses == misses;
            if (!ok) {
                std::fprintf(stderr, "FAIL: plan %s, probe %s: hits by uid %llu, misses %llu; expected %llu, %llu\n",
                             setting, probe, (unsigned long long) d.hits_uid, (unsigned long long) d.misses,
                             (unsigned long long) uid_hits, (unsigned long long) misses);
                g_failures++;
            }
            // the counters' definition: on a hit graph_compute takes `checked` as `start`, so its key time is its
            // lookup time, the same clock reads (that a hit really skips the contract check is probe (o))
            if (on && uid_hits == 1 && d.hits_uid == 1 && d.lookup_us != d.key_us) {
                std::fprintf(stderr, "FAIL: plan %s, probe %s: a trusted hit took %.1f us to the step list but %.1f us "
                             "from after the check: the check ran\n", setting, probe, d.lookup_us, d.key_us);
                g_failures++;
                ok = false;
            }
            return ok;
        };
        // an output kept under none and compared byte for byte under uid-trust
        const auto as_none = [&](const char * name, const std::vector<float> & out) {
            if (!on) {
                none_outputs[name] = out;
                return true;
            }
            n_compared++;
            const auto it = none_outputs.find(name);
            if (it == none_outputs.end() || !same_bytes(it->second, out)) {
                std::fprintf(stderr, "FAIL: plan %s: output %s differs from FLAGOS_SPACEMIT_PLAN=none's\n", setting, name);
                g_failures++;
                return false;
            }
            return true;
        };
        // a probe's outcome: its counters and its outputs
        const auto probe = [&](const char * name, bool counters, bool outputs) {
            n_probes++;
            if (!outputs) {
                std::fprintf(stderr, "FAIL: plan %s, probe %s: the outputs are wrong or the provider failed\n", setting,
                             name);
                g_failures++;
            }
            n_passed += counters && outputs;
        };
        const auto layer_ok = [&](plan_layer & l, const std::vector<float> & out) {
            const double e = l.cpu_nmse(cpu, out);
            worst_layer    = std::max(worst_layer, e);
            return e <= k_layer_max_nmse;
        };

        plan_layer a, e, b, c, d, w;
        REQUIRE(a.init(dev, 1, 1e6f) && e.init(dev, 1, 1e6f) && b.init(dev, 1, 1e6f) && c.init(dev, 1, 1e4f) &&
                d.init(dev, 7, 1e6f) && w.init(dev, 1, 1e6f, true));
        a.gd->uid = k_uid;
        e.gd->uid = k_uid;  // (e): the uid reused by other node objects
        w.gd->uid = k_uid;  // (j): likewise, so that its miss is the pointer walk's, not a uid of 0

        // (a) the same object twice
        std::vector<float> out_a, out_a2;
        bool               ran = a.run(backend, out_a);
        bool               cnt = expect("(a) first", 0, 1);
        ran                    = ran && a.run(backend, out_a2);
        cnt                    = expect("(a) second", 1, 0) && cnt;
        probe("(a)", cnt, ran && layer_ok(a, out_a) && same_bytes(out_a, out_a2) && as_none("a", out_a));

        // (e) the uid stamped on other objects of the same structure: a miss by the key's pointer walk
        std::vector<float> out_e;
        ran = e.run(backend, out_e);
        probe("(e)", expect("(e)", 0, 1), ran && same_bytes(out_e, out_a) && as_none("e", out_e));

        // (b) the same structure as new objects, uid 0
        std::vector<float> out_b;
        ran = b.run(backend, out_b);
        probe("(b)", expect("(b)", 0, 1), ran && same_bytes(out_b, out_a) && as_none("b", out_b));

        // (c) one op_params value changed: RoPE's base
        std::vector<float> out_c;
        ran = c.run(backend, out_c);
        probe("(c)", expect("(c)", 0, 1),
              ran && layer_ok(c, out_c) && !same_bytes(out_c, out_a) && as_none("c", out_c));

        // (d) one shape changed: 7 tokens, a larger workspace
        std::vector<float> out_d;
        ran = d.run(backend, out_d);
        probe("(d)", expect("(d)", 0, 1), ran && layer_ok(d, out_d) && as_none("d", out_d));

        // (g) the first layer after the larger graph: a miss (one entry), then its plan replayed in the grown workspace
        std::vector<float> out_g, out_g2;
        ran = a.run(backend, out_g);
        cnt = expect("(g) first", 0, 1);
        ran = ran && a.run(backend, out_g2);
        cnt = expect("(g) second", 1, 0) && cnt;
        probe("(g)", cnt, ran && same_bytes(out_g, out_a) && same_bytes(out_g2, out_a) && as_none("g", out_g2));

        // (h) a graph view of the first layer: uid 0, a miss
        ggml_cgraph        view = ggml_graph_view(a.gd, 0, ggml_graph_n_nodes(a.gd));
        std::vector<float> out_h;
        std::vector<float> out_h2;
        ran = view.uid == 0 && a.run(backend, out_h, &view);
        cnt = expect("(h) the view", 0, 1);
        // the view again: the entry is now its own (uid 0, the same objects and word), and a uid of 0 never hits
        ran = ran && a.run(backend, out_h2, &view);
        cnt = expect("(h) the view again", 0, 1) && cnt;
        probe("(h)", cnt, ran && same_bytes(out_h, out_a) && same_bytes(out_h2, out_a) && as_none("h", out_h));

        // (f) FLAGOS_SPACEMIT_ATTN flipped between two one-token attention graphs, then (l) set again to its value
        {
            const attn_shape s = { 32, 8, 128, 1024, 1, 1023 };
            attn_run         r;
            REQUIRE(r.init(dev, s, attn_mask::causal, 0.0f, 0.0f, 70));
            REQUIRE(ggml_backend_graph_compute(cpu, r.g_cpu) == GGML_STATUS_SUCCESS);
            r.g_dev->uid = k_attn_uid;  // stamped, so that only the word tells the variants apart
            const auto attn_ok = [&](const std::vector<float> & o) {
                const attn_error err =
                    attn_errors(o.data(), static_cast<const float *>(r.out_ref->data), (int64_t) o.size(), r.out->ne[0]);
                worst_attn = std::max(worst_attn, err.all);
                return err.all <= k_attn_max_nmse && err.row <= k_attn_max_row_nmse;
            };
            std::vector<float> one_chunk, one_chunk2, split;
            uint64_t           steps_one = 0, steps_one2 = 0, steps_split = 0;
            {
                env_override attn_switch("FLAGOS_SPACEMIT_ATTN", "one-chunk");
                take();
                g_steps_take();
                ran       = r.dev_output(backend, one_chunk);
                steps_one = g_steps_take();
                cnt       = expect("(f) one-chunk", 0, 1);
                ran       = ran && r.dev_output(backend, one_chunk2);
                steps_one2 = g_steps_take();
                cnt        = expect("(f) one-chunk again", 1, 0) && cnt;
            }
            {
                env_override attn_switch("FLAGOS_SPACEMIT_ATTN", nullptr);  // the default: the split, with the turns
                take();
                ran         = ran && r.dev_output(backend, split);
                steps_split = g_steps_take();
                cnt         = expect("(f) split", 0, 1) && cnt;
            }
            const bool steps_ok = steps_one == 1 && steps_one2 == 1 && steps_split == 2;
            if (!steps_ok) {
                std::fprintf(stderr,
                             "FAIL: plan %s, probe (f): steps run one-chunk %llu, again %llu, split %llu; expected 1, 1, "
                             "2\n",
                             setting, (unsigned long long) steps_one, (unsigned long long) steps_one2,
                             (unsigned long long) steps_split);
                g_failures++;
            }
            probe("(f)", cnt && steps_ok,
                  ran && attn_ok(one_chunk) && same_bytes(one_chunk, one_chunk2) && attn_ok(split) &&
                      as_none("f one-chunk", one_chunk) && as_none("f split", split));

            // (l) back to one-chunk (a miss: the contract changes), then the same value set again through set_env:
            // the switch generation moves, the contract stays, so only the word tells the plan apart: a miss, 1 step
            std::vector<float> again, same;
            uint64_t           steps_again = 0, steps_same = 0;
            {
                env_override attn_switch("FLAGOS_SPACEMIT_ATTN", "one-chunk");
                take();
                g_steps_take();
                ran         = r.dev_output(backend, again);
                steps_again = g_steps_take();
                cnt         = expect("(l) one-chunk after the split", 0, 1);
                set_env("FLAGOS_SPACEMIT_ATTN", "one-chunk");
                ran        = ran && r.dev_output(backend, same);
                steps_same = g_steps_take();
                cnt        = expect("(l) the same value set again", 0, 1) && cnt;
            }
            const bool steps_l = steps_again == 1 && steps_same == 1;
            if (!steps_l) {
                std::fprintf(stderr, "FAIL: plan %s, probe (l): steps run %llu and %llu; expected 1 and 1\n", setting,
                             (unsigned long long) steps_again, (unsigned long long) steps_same);
                g_failures++;
            }
            probe("(l)", cnt && steps_l,
                  ran && same_bytes(again, one_chunk) && same_bytes(same, one_chunk) && as_none("l", same));
        }
        take();

        // (j) a refusal right after the first layer is the entry: the same structure with its weights in a plain CPU
        // buffer (the contract check refuses its MUL_MATs: an IME weight must sit in our buffer; new objects, so the
        // probe misses and the check runs); it fails with nothing counted (a refused graph returns before the
        // counters), and the first layer is a hit after it. The first layer's own graph object with a MUL_MAT pointed
        // at such a weight is not run: the probe would hit and the IME kernel would read an unrepacked weight; that is
        // exactly the check the trust gives up (llama.cpp's reuse never changes a reused graph's tensors: the uid
        // changes when the graph is rebuilt). Then the cache itself (plan_key_probe: the key alone through probe())
        {
            std::vector<float> out_j, out_w, out_j2;
            ran = a.run(backend, out_j);  // a miss: the entry was (l)'s
            cnt = expect("(j) the first layer", 0, 1);
            const bool refused_w = !w.run(backend, out_w);
            cnt = expect("(j) weights in a CPU buffer", 0, 0, false) && cnt;
            ran = ran && a.run(backend, out_j2);
            cnt = expect("(j) the first layer after the refusal", 1, 0) && cnt;
            const bool key = !on || plan_key_probe(a.gd);
            probe("(j)", cnt && key,
                  refused_w && ran && same_bytes(out_j, out_a) && same_bytes(out_j2, out_a) && as_none("j", out_j2));
        }

        // (o) a node unflagged after the store, a probe of the trust itself: the entry is the first layer's (its hit
        // in (j)); one of its MUL_MATs loses GGML_TENSOR_FLAG_COMPUTE, the graph runs, the flag is restored. Under none
        // the contract check's filter skips the node, its 2 steps fewer than the graph's; under uid-trust the probe
        // hits and the stored steps run, as many as before, so the check was not made. The bytes are the same under
        // both: the skipped node's output keeps the previous run's value. Not a scenario: llama.cpp sets the flag only
        // when it builds a graph (ggml.c:7100), and a reused graph skips the build (src/llama-context.cpp:1339)
        {
            ggml_tensor * unflagged = nullptr;
            for (int i = 0; i < ggml_graph_n_nodes(a.gd) && unflagged == nullptr; i++) {
                ggml_tensor * node = ggml_graph_node(a.gd, i);
                if (node->op == GGML_OP_MUL_MAT && (node->flags & GGML_TENSOR_FLAG_COMPUTE) != 0) {
                    unflagged = node;
                }
            }
            REQUIRE(unflagged != nullptr);
            std::vector<float> out_o, out_o2;
            take();
            g_steps_take();
            ran                      = a.run(backend, out_o);
            const uint64_t steps_all = g_steps_take();
            cnt                      = expect("(o) the first layer", 1, 0);
            unflagged->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            ran                      = ran && a.run(backend, out_o2);
            const uint64_t steps_now = g_steps_take();
            unflagged->flags |= GGML_TENSOR_FLAG_COMPUTE;
            cnt                    = expect("(o) a node unflagged after the store", 1, 0) && cnt;
            const uint64_t want    = on ? steps_all : steps_all - 2;
            const bool     steps_o = steps_now == want;
            if (!steps_o) {
                std::fprintf(stderr, "FAIL: plan %s, probe (o): steps run %llu with the node unflagged, %llu before; "
                             "expected %llu\n", setting, (unsigned long long) steps_now,
                             (unsigned long long) steps_all, (unsigned long long) want);
                g_failures++;
            }
            probe("(o)", cnt && steps_o,
                  ran && same_bytes(out_o, out_a) && same_bytes(out_o2, out_a) && as_none("o", out_o2));
        }

        // (m) the first layer's own object with another uid: the entry is its own (the same objects and word), so only
        // the uid decides, a miss. llama.cpp rebuilds every graph in the same buffer (llama-graph.cpp:1316-1326), so a
        // rebuilt graph of another shape can get the same node addresses and count; its new uid is what tells it apart
        {
            std::vector<float> out_m;
            a.gd->uid = k_uid + 2;
            ran       = a.run(backend, out_m);
            a.gd->uid = k_uid;
            probe("(m)", expect("(m) another uid on the same objects", 0, 1),
                  ran && same_bytes(out_m, out_a) && as_none("m", out_m));
        }

        // (k) FLAGOS_SPACEMIT_QUANTIZE=shared: no plan cache, so a layer graph twice counts nothing
        {
            env_override       shared("FLAGOS_SPACEMIT_QUANTIZE", "shared");
            std::vector<float> out_k, out_k2;
            ran = a.run(backend, out_k);
            cnt = expect("(k) shared, first", 0, 0);
            ran = ran && a.run(backend, out_k2);
            cnt = expect("(k) shared, second", 0, 0) && cnt;
            probe("(k)", cnt, ran && same_bytes(out_k, out_a) && same_bytes(out_k2, out_a) && as_none("k", out_k2));
        }
        take();

        // (i) none: nothing counted
        if (!on) {
            const bool zero =
                total.hits_uid == 0 && total.misses == 0 && total.lookup_us == 0.0 && total.key_us == 0.0;
            if (!zero) {
                std::fprintf(stderr, "FAIL: plan none, probe (i): a counter moved\n");
                g_failures++;
            }
            probe("(i)", zero, true);
        }
        probe("(n)", default_ok, true);
        ggml_backend_free(backend);
        std::printf("plan     FLAGOS_SPACEMIT_PLAN=%s: probes %d/%d passed (a same object twice, b same structure as "
                    "new objects, c RoPE base changed, d 7 tokens, e a stamped uid on other objects, f "
                    "FLAGOS_SPACEMIT_ATTN flipped, l the same value set again, g replay after a larger graph, h a graph "
                    "view, j a refusal%s, o a node unflagged after the store, m another uid on the same objects, k no "
                    "cache under QUANTIZE=shared%s, n the default uid-trust); %d graphs: %llu "
                    "hits by uid, %llu misses, lookup %.1f us and key %.1f us per graph; %s; layer NMSE %.2e (bound "
                    "%.0e), attention NMSE %.2e (bound %.0e)\n",
                    setting, n_passed, n_probes, on ? " after the entry and the key alone (seven cases)" : "",
                    on ? "" : ", i no counter moved", n_graphs, (unsigned long long) total.hits_uid,
                    (unsigned long long) total.misses, n_graphs > 0 ? total.lookup_us / n_graphs : 0.0,
                    n_graphs > 0 ? total.key_us / n_graphs : 0.0,
                    on ? (std::to_string(n_compared) + " outputs identical to none's").c_str() :
                         (std::to_string(none_outputs.size()) + " outputs kept").c_str(),
                    worst_layer, k_layer_max_nmse, worst_attn, k_attn_max_nmse);
    }
    ggml_backend_free(cpu);
    return 0;
}

//
// S1 step 5 (plan.md "S1 step 5 spec" §4): the diagnostics under the deck's names. GGML_FLAGOS_DISABLE_OPS sends the
// ops it names back to the CPU (placement and execution together), GGML_FLAGOS_TRACE_LAUNCH logs the steps by name and
// names the step whose launch failed, GGML_FLAGOS_STATS=- writes the counters to stderr, and every replaced
// FLAGOS_SPACEMIT_* name is reported once with the new one and otherwise ignored
//

// the GGML log lines while it lives, kept and still printed by ggml's default callback
class log_capture {
  public:
    log_capture() { ggml_log_set(sink, this); }

    ~log_capture() { ggml_log_set(nullptr, nullptr); }

    log_capture(const log_capture &)             = delete;
    log_capture & operator=(const log_capture &) = delete;

    // the lines since the last call
    std::string take() {
        std::lock_guard<std::mutex> lock(mutex);
        std::string                 out;
        out.swap(text);
        return out;
    }

  private:
    static void sink(enum ggml_log_level level, const char * line, void * user) {
        auto * self = static_cast<log_capture *>(user);
        {
            std::lock_guard<std::mutex> lock(self->mutex);
            self->text += line;
        }
        if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
            std::fputs(line, stderr);  // a warning the checks do not expect stays visible
        }
    }

    std::mutex  mutex;
    std::string text;
};

// lines of text that start with prefix
static int count_lines(const std::string & text, const char * prefix) {
    const size_t len = std::strlen(prefix);
    int          n   = 0;
    size_t       at  = 0;
    while (at <= text.size()) {
        const size_t end  = text.find('\n', at);
        const size_t stop = end == std::string::npos ? text.size() : end;
        if (stop - at >= len && text.compare(at, len, prefix) == 0) {
            n++;
        }
        if (end == std::string::npos) {
            break;
        }
        at = end + 1;
    }
    return n;
}

// a Qwen3-like layer (build_layer, down projection Q4_0) with its weights in the provider's buffer and its inputs,
// caches and mask in CPU buffers, its graph left for a scheduler to allocate: an op the provider refuses then has its
// output in a CPU buffer, which is what GGML_FLAGOS_DISABLE_OPS asks of the scheduler (a graph allocated in our buffer
// beforehand would instead give every node to the buffer's backend). The values are plan_layer's
struct diag_layer {
    ggml_context *        ctx_h = nullptr, * ctx_w = nullptr, * ctx_g = nullptr;
    ggml_backend_buffer_t bh = nullptr, bw = nullptr;
    ggml_cgraph *         g = nullptr;
    layer_out             o = {};

    bool init(ggml_backend_dev_t dev, int64_t n_tokens) {
        ggml_init_params params = { ggml_tensor_overhead() * 128 + ggml_graph_overhead(), nullptr, true };
        ctx_h                   = ggml_init(params);
        ctx_w                   = ggml_init(params);
        ctx_g                   = ggml_init(params);
        ggml_tensor * x         = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F32, k_d, n_tokens);
        ggml_tensor * pos       = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, n_tokens);
        ggml_tensor * kv_idx    = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I64, n_tokens);
        ggml_tensor * out_ids   = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, 1);
        ggml_tensor * kc        = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F16, k_hd * k_nkv, k_slots);
        ggml_tensor * vc        = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F16, k_hd * k_nkv, k_slots);
        ggml_tensor * mask      = ggml_new_tensor_4d(ctx_h, GGML_TYPE_F16, k_slots, n_tokens, 1, 1);
        ggml_set_name(kc, "cache_k_l0");
        ggml_set_name(vc, "cache_v_l0");
        const layer_weights w = new_layer_weights(ctx_w, GGML_TYPE_Q4_0);
        bh                    = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
        bw                    = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, ggml_backend_dev_buffer_type(dev));
        if (bh == nullptr || bw == nullptr) {
            return false;
        }
        ggml_backend_buffer_clear(bh, 0);
        set_layer_weights(w);

        const std::vector<float> xv = random_values(k_d * n_tokens, 5, 1.0f);
        std::vector<int32_t>     pv(n_tokens);
        std::vector<int64_t>     iv(n_tokens);
        std::vector<int32_t>     cell_pos(k_slots, -1);  // -1: empty
        for (int64_t t = 0; t < n_tokens; t++) {
            pv[t]           = (int32_t) (k_layer_pos + t);
            iv[t]           = (3 + 5 * t) % k_slots;
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
        for (ggml_tensor * cache : { kc, vc }) {
            const std::vector<ggml_fp16_t> h = random_f16(ggml_nelements(cache), cache_seed++, 1.0f);
            ggml_backend_tensor_set(cache, h.data(), 0, ggml_nbytes(cache));
        }
        g = ggml_new_graph(ctx_g);
        o = build_layer(ctx_g, g, w, x, pos, kv_idx, out_ids, kc, vc, mask, n_tokens);
        ggml_set_output(o.full);
        return true;
    }

    ~diag_layer() {
        for (ggml_backend_buffer_t b : { bh, bw }) {
            ggml_backend_buffer_free(b);
        }
        for (ggml_context * c : { ctx_h, ctx_w, ctx_g }) {
            ggml_free(c);
        }
    }

    // the layer once, through a scheduler of its own: the output, the splits the graph took and the backend the
    // scheduler gave the SwiGLU. A layer is run once, since its nodes then hold the scheduler's allocations, which the
    // scheduler owns; two settings are compared on two layers of the same values. false if the graph failed
    bool run(ggml_backend_t backend, ggml_backend_t cpu, std::vector<float> & out, int & n_splits,
             ggml_backend_t & glu_backend) {
        ggml_tensor * glu = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(g) && glu == nullptr; i++) {
            if (ggml_graph_node(g, i)->op == GGML_OP_GLU) {
                glu = ggml_graph_node(g, i);
            }
        }
        ggml_backend_t       both[2] = { backend, cpu };
        ggml_backend_sched_t sched   = ggml_backend_sched_new(both, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
        bool                 ok      = sched != nullptr && glu != nullptr &&
                  ggml_backend_sched_graph_compute(sched, g) == GGML_STATUS_SUCCESS;
        if (ok) {
            n_splits    = ggml_backend_sched_get_n_splits(sched);
            glu_backend = ggml_backend_sched_get_tensor_backend(sched, glu);
            out.assign(ggml_nelements(o.full), NAN);
            ggml_backend_tensor_get(o.full, out.data(), 0, ggml_nbytes(o.full));
        }
        ggml_backend_sched_free(sched);
        return ok;
    }
};

static int check_diagnostics(ggml_backend_dev_t dev) {
    const spacemit_plugin * loaded = spacemit_device_plugin(dev);
    REQUIRE(loaded != nullptr);
    const spacemit_plugin_v1 * table = spacemit_plugin_table(loaded);
    REQUIRE(table != nullptr);
    spacemit_ops ops;
    std::string  detail;
    REQUIRE(spacemit_ops_bind(table, &ops, &detail));
    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    // three nodes the provider claims without the switch: a SwiGLU, a one-token attention and an ADD
    ggml_init_params params  = { ggml_tensor_overhead() * 32, nullptr, true };
    ggml_context *   ctx     = ggml_init(params);
    ggml_tensor *    gate    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_ff, 1);
    ggml_tensor *    up      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_ff, 1);
    ggml_tensor *    glu     = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor *    k_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k_hd * 2, 544);
    ggml_tensor *    v_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k_hd * 2, 544);
    ggml_tensor *    attn =
        build_flash_attn(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k_hd, 8, 1), k_cache, v_cache, nullptr, 2, 512,
                         0.0f);
    ggml_tensor * add = ggml_add(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_d, 1),
                                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_d, 1));

    // the switch's three settings: unset, two ops named, an unknown name. find_op and the device's supports_op must
    // answer alike, since both call find_op (placement and execution cannot disagree)
    const struct {
        const char * value;
        bool         glu, attn, add;
    } cases[] = {
        { nullptr,                 true,  true,  true },
        { "GLU, FLASH_ATTN_EXT ",  false, false, true },  // spaces trimmed, case as given
        { "NOPE",                  true,  true,  true },
    };
    int disable_ok = 0;
    for (const auto & c : cases) {
        env_override switch_off("GGML_FLAGOS_DISABLE_OPS", c.value);
        log_capture  log;
        bool         ok = true;
        for (const auto & want : { std::pair<ggml_tensor *, bool>{ glu, c.glu }, { attn, c.attn }, { add, c.add } }) {
            const bool found    = spacemit_find_op(ops, want.first, buft) != nullptr;
            const bool claimed  = ggml_backend_dev_supports_op(dev, want.first);
            ok                  = ok && found == want.second && claimed == want.second;
        }
        // an unknown name is reported once, however often the list is read, and disables nothing
        const std::string text = log.take();
        const int          said = count_lines(text, "FlagOS SpacemiT: GGML_FLAGOS_DISABLE_OPS: NOPE");
        ok = ok && said == (c.value != nullptr && std::strcmp(c.value, "NOPE") == 0 ? 1 : 0);
        if (!ok) {
            std::fprintf(stderr, "FAIL: diagnostics, GGML_FLAGOS_DISABLE_OPS=%s: the ops claimed are wrong, or the "
                                 "unknown name was reported %d times\n", c.value != nullptr ? c.value : "(unset)", said);
            g_failures++;
        }
        disable_ok += ok;
    }
    ggml_free(ctx);

    // a layer through a scheduler of the provider and the CPU: with GLU disabled the scheduler gives the SwiGLU to the
    // CPU and splits the layer around it, and the layer's output stays within the layer bound of the run that keeps it
    // on the provider (the CPU's vectorized SiLU is not the reference kernel's expf, so the bytes differ)
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);
    bool   sched_ok = false;
    int    splits[2] = { 0, 0 };
    double e_sched   = -1.0;
    {
        std::vector<float> out[2];
        ggml_backend_t     glu_backend[2] = { nullptr, nullptr };
        bool               ran            = true;
        for (int i = 0; i < 2; i++) {
            env_override switch_off("GGML_FLAGOS_DISABLE_OPS", i == 1 ? "GLU" : nullptr);
            diag_layer   l;  // one layer per setting, of the same values: its nodes keep its scheduler's allocations
            REQUIRE(l.init(dev, 1));
            ran = ran && l.run(backend, cpu, out[i], splits[i], glu_backend[i]);
        }
        if (ran) {
            e_sched  = nmse(out[1].data(), out[0].data(), (int64_t) out[0].size());
            sched_ok = glu_backend[0] == backend && glu_backend[1] == cpu && splits[1] > splits[0] &&
                       e_sched >= 0.0 && e_sched <= k_layer_max_nmse;
        }
        if (!sched_ok) {
            std::fprintf(stderr,
                         "FAIL: diagnostics, GGML_FLAGOS_DISABLE_OPS=GLU through the scheduler: %s; the SwiGLU went to "
                         "%s (without the switch %s), splits %d against %d, NMSE %.3e\n",
                         ran ? "wrong placement or output" : "a graph failed",
                         glu_backend[1] != nullptr ? ggml_backend_name(glu_backend[1]) : "nowhere",
                         glu_backend[0] != nullptr ? ggml_backend_name(glu_backend[0]) : "nowhere", splits[1],
                         splits[0], e_sched);
            g_failures++;
        }
    }

    // GGML_FLAGOS_TRACE_LAUNCH: one line per step before the replay, then the two launch lines; with
    // GGML_FLAGOS_SYNC_LAUNCH as well one status line per step, the core issuing a replay per step
    int      trace_ok = 0;
    uint64_t n_steps = 0, n_steps_sync = 0;
    int      step_lines = 0, status_lines = 0;
    {
        env_override switch_off("GGML_FLAGOS_DISABLE_OPS", nullptr);
        plan_layer   l;
        REQUIRE(l.init(dev, 1, 1e6f));
        for (int sync = 0; sync < 2; sync++) {
            env_override       trace_switch("GGML_FLAGOS_TRACE_LAUNCH", "1");
            env_override       sync_switch("GGML_FLAGOS_SYNC_LAUNCH", sync == 1 ? "1" : nullptr);
            std::vector<float> out;
            log_capture        log;
            g_steps_take();
            const bool        ran   = l.run(backend, out);
            const uint64_t    steps = g_steps_take();
            const std::string text  = log.take();
            const int         lines = count_lines(text, "FlagOS SpacemiT: step ");
            const int         status = count_lines(text, "FlagOS SpacemiT: launch step ");
            const int         launch = count_lines(text, "FlagOS SpacemiT: launch ") - status;
            const bool        ok = ran && steps > 0 && lines == (int) steps && launch == 2 &&
                            status == (sync == 1 ? (int) steps : 0);
            if (!ok) {
                std::fprintf(stderr,
                             "FAIL: diagnostics, GGML_FLAGOS_TRACE_LAUNCH=1%s: %d step lines and %d launch lines "
                             "(%d with a status) for %llu steps\n", sync == 1 ? " with SYNC_LAUNCH=1" : "", lines,
                             launch, status, (unsigned long long) steps);
                g_failures++;
            }
            trace_ok += ok;
            if (sync == 1) {
                n_steps_sync = steps;
                status_lines = status;
            } else {
                n_steps    = steps;
                step_lines = lines;
            }
        }
    }

    // the step whose launch failed, named by the core: FLAGOS_SPACEMIT_TEST_FAIL_NODE is the step's index inside a
    // launch, so with one launch per step only the first step of a launch can be forced to fail, and the first failing
    // step is step 1. With one launch per split the whole graph's launch fails, and the trace names no step
    int fail_ok = 0, named_step = 0;
    {
        env_override switch_off("GGML_FLAGOS_DISABLE_OPS", nullptr);
        plan_layer   l;
        REQUIRE(l.init(dev, 1, 1e6f));
        for (int sync = 0; sync < 2; sync++) {
            env_override trace_switch("GGML_FLAGOS_TRACE_LAUNCH", "1");
            env_override sync_switch("GGML_FLAGOS_SYNC_LAUNCH", sync == 1 ? "1" : nullptr);
            set_env("FLAGOS_SPACEMIT_TEST_FAIL_NODE", sync == 1 ? "0" : "3");
            ggml_backend_t failing = ggml_backend_dev_init(dev, nullptr);
            set_env("FLAGOS_SPACEMIT_TEST_FAIL_NODE", nullptr);
            REQUIRE(failing != nullptr);
            std::vector<float> out;
            std::fprintf(stderr, "(the next error lines are expected)\n");
            log_capture log;
            out.assign(ggml_nelements(l.od.full), NAN);
            const bool        failed = ggml_backend_graph_compute(failing, l.gd) == GGML_STATUS_FAILED;
            const std::string text   = log.take();
            // the last status line of a failing sync run names the step: "launch step 1/<n>: failed"
            const int  status = count_lines(text, "FlagOS SpacemiT: launch step ");
            const bool named  = text.find("launch step 1/") != std::string::npos &&
                               text.find(": failed") != std::string::npos;
            const bool ok = failed && (sync == 1 ? status == 1 && named : status == 0) &&
                            text.find("steps: failed") != std::string::npos;
            if (!ok) {
                std::fprintf(stderr,
                             "FAIL: diagnostics, a forced failure under GGML_FLAGOS_TRACE_LAUNCH=1%s: compute %s, %d "
                             "status lines, the failing step %s\n", sync == 1 ? " with SYNC_LAUNCH=1" : "",
                             failed ? "failed as it should" : "did not fail", status, named ? "named" : "not named");
                g_failures++;
            }
            fail_ok += ok;
            named_step += sync == 1 && named;
            ggml_backend_free(failing);
        }
    }
    ggml_backend_free(backend);
    ggml_backend_free(cpu);

    // GGML_FLAGOS_STATS=-: the counters' JSON lines go to stderr when the counter file is destroyed. The switch is
    // read once per process, so the probe makes a counter file of its own from the switch's value, as get() does, and
    // reads stderr from a temporary file
    bool stats_stderr = false;
#if !defined(_WIN32)
    {
        const std::string path = std::string(FLAGOS_SPACEMIT_TEST_PLUGIN_DIR) + "/diag-stats.jsonl";
        std::fflush(stderr);
        const int saved = dup(fileno(stderr));
        FILE *    f     = std::fopen(path.c_str(), "w+");
        if (f != nullptr && saved >= 0) {
            dup2(fileno(f), fileno(stderr));
            {
                spacemit_stats_file counters("-");
                spacemit_run_stats  run;
                run.reset(2);
                run.n_steps = 2;
                run.n_tiles = 1;
                counters.add("per-split", 1, -1, run, 123.0, spacemit_plan_result::off, 1.0, 1.0);
            }  // the destructor writes
            std::fflush(stderr);
            dup2(saved, fileno(stderr));
            close(saved);
            std::fclose(f);
            std::ifstream written(path);
            std::string   line, all;
            int           lines = 0;  // the JSON lines; the file also holds the warning that named stderr
            while (std::getline(written, line)) {
                if (!line.empty() && line[0] == '{') {
                    lines++;
                    all = line;
                }
            }
            stats_stderr = lines == 1 && all.rfind("{\"launch\":\"per-split\",\"rows\":1,\"kv\":-1,\"graphs\":1,", 0) == 0;
        }
        if (!stats_stderr) {
            std::fprintf(stderr, "FAIL: diagnostics, GGML_FLAGOS_STATS=-: no single JSON line on stderr (%s)\n",
                         path.c_str());
            g_failures++;
        }
    }
#else
    stats_stderr = true;
#endif

    // every replaced name: set, it is reported once with the new name and changes nothing
    const struct {
        const char * old_name;
        const char * new_name;
    } renamed[] = {
        { "FLAGOS_SPACEMIT_TEST_STATS",     "GGML_FLAGOS_STATS"           },
        { "FLAGOS_SPACEMIT_TEST_OP_STATS",  "GGML_FLAGOS_OP_STATS"        },
        { "FLAGOS_SPACEMIT_TEST_REFERENCE", "GGML_FLAGOS_KERNEL_VARIANTS" },
        { "FLAGOS_SPACEMIT_TEST_LAUNCH",    "GGML_FLAGOS_SYNC_LAUNCH"     },
        { "FLAGOS_SPACEMIT_PLUGIN",         "GGML_FLAGOS_PROVIDER_PATH"   },
    };
    int renamed_ok = 0;
    for (const auto & r : renamed) {
        // every switch change resets the warnings, so each name is reported again here
        env_override old_set(r.old_name, "per-step");
        env_override sync_unset("GGML_FLAGOS_SYNC_LAUNCH", nullptr);
        log_capture  log;
        spacemit_renamed(r.old_name, r.new_name);
        spacemit_renamed(r.old_name, r.new_name);
        const std::string text = log.take();
        const std::string said = std::string("FlagOS SpacemiT: ") + r.old_name + " is now " + r.new_name;
        // "changes nothing": the launch mode is the default although the old launch switch says per-step
        const bool ok = count_lines(text, said.c_str()) == 1 && count_lines(text, "FlagOS SpacemiT: FLAGOS_") == 1 &&
                        spacemit_launch_mode() == spacemit_launch_kind::per_split;
        if (!ok) {
            std::fprintf(stderr, "FAIL: diagnostics, %s set: not reported exactly once as %s, or it still acts\n",
                         r.old_name, r.new_name);
            g_failures++;
        }
        renamed_ok += ok;
    }

    std::printf("diag     DISABLE_OPS (%d/%zu settings: GLU and FLASH_ATTN_EXT refused by find_op and supports_op, an "
                "unknown name reported once and ignored; a layer's SwiGLU on the CPU side of the scheduler, %d splits "
                "against %d, NMSE %.2e, bound %.0e), TRACE_LAUNCH (%d/2: %d step lines for %llu steps and the two "
                "launch lines, with one launch per step %d status lines for %llu; %d/2 forced failures reported, the "
                "failing step named %d/1), STATS=- one JSON line to stderr: %s, renamed switches %d/%zu: checked\n",
                disable_ok, sizeof(cases) / sizeof(cases[0]), splits[1], splits[0], e_sched, k_layer_max_nmse,
                trace_ok, step_lines, (unsigned long long) n_steps, status_lines, (unsigned long long) n_steps_sync,
                fail_ok, named_step, stats_stderr ? "yes" : "no", renamed_ok,
                sizeof(renamed) / sizeof(renamed[0]));
    return 0;
}

// eight 1-row matmuls of one weight on one input (sixteen steps: the activations quantized, then the matmul, each
// time), their outputs one after another; false if the provider failed
static bool matmuls_out(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_type type, std::vector<float> & out) {
    mm_weight w;
    if (!w.init(dev, k_ff, k_d, 21, type)) {
        return false;
    }
    ggml_init_params params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx_h  = ggml_init(params);  // the input: CPU buffer
    ggml_context *   ctx_d  = ggml_init(params);  // the graph and the outputs: the provider's buffer
    ggml_tensor *    x      = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F32, k_ff, 1);
    ggml_cgraph *    g      = ggml_new_graph(ctx_d);
    std::vector<ggml_tensor *> ys;
    for (int i = 0; i < 8; i++) {
        ys.push_back(ggml_mul_mat(ctx_d, w.w_dev, x));
        ggml_build_forward_expand(g, ys.back());
    }
    ggml_backend_buffer_t bh = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_t bd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, ggml_backend_dev_buffer_type(dev));
    bool                  ok = bh != nullptr && bd != nullptr;
    if (ok) {
        const std::vector<float> xv = random_values(k_ff, 22, 2.0f);
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ok = ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS;
    }
    out.clear();
    for (ggml_tensor * y : ys) {
        if (ok) {
            std::vector<float> v(ggml_nelements(y));
            ggml_backend_tensor_get(y, v.data(), 0, ggml_nbytes(y));
            out.insert(out.end(), v.begin(), v.end());
        }
    }
    ggml_backend_buffer_free(bh);
    ggml_backend_buffer_free(bd);
    ggml_free(ctx_h);
    ggml_free(ctx_d);
    return ok;
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

// bytes of one weight row in the provider's layout (Q4_1 and Q6_K weights are stored as Q8_0)
static double ime_row_bytes(ggml_type type, int64_t k) {
    return type == GGML_TYPE_Q4_0 ? (double) (k / 32 * 18) : (double) (k / 32 * 34);
}

// fusion-and-splitting.md §7.3: the matmuls of a Qwen3-4B batch under the default (fit,staged) against
// FLAGOS_SPACEMIT_MM=split, at the row counts of short prompts, of a prompt's last batch, of llama-bench's batches and
// with 2 or 3 rows left over after the groups of 4: each shape alone with the path each setting took, and per batch the
// shapes weighted by how often a batch runs them (36 q and attention outputs, 72 k and v, 72 gate and up, 32 + 4 down;
// the head runs on one row). As in the model, where a batch reads each matrix once, every run reads its weights from
// DRAM: it takes the next of enough copies of the weight (32 MB together)
static int bench_mm(ggml_backend_dev_t dev) {
    env_override   mm("FLAGOS_SPACEMIT_MM", nullptr);
    ggml_backend_t backend = nullptr;
    {
        env_override tcm("FLAGOS_SPACEMIT_TEST_TCM", "393216");  // the host executor's TCM (check_mm_paths)
        backend = ggml_backend_dev_init(dev, nullptr);
    }
    REQUIRE(backend != nullptr);
    struct mm_shape { const char * what; ggml_type type; int64_t k, n; int per_batch; };
    const mm_shape shapes[] = {
        { "q", GGML_TYPE_Q4_0, 2560, 4096, 36 },        { "k, v", GGML_TYPE_Q4_0, 2560, 1024, 72 },
        { "attn out", GGML_TYPE_Q4_0, 4096, 2560, 36 }, { "gate, up", GGML_TYPE_Q4_0, 2560, 9728, 72 },
        { "down", GGML_TYPE_Q4_0, 9728, 2560, 32 },     { "down Q4_1", GGML_TYPE_Q4_1, 9728, 2560, 4 },
    };
    const std::vector<int64_t> rows = { 2,  4,  8,  16,  17,  24,  31,  32,  33,  34,  35,  36,  40,  44,  48,  51, 59,
                                        62, 63, 64, 65, 79, 96, 100, 112, 113, 115, 128, 129, 131, 149, 160, 256, 512 };
    const char * const settings[] = { "split", "fit,staged" };
    const int          n_set      = 2;
    std::vector<std::vector<double>> batch(n_set, std::vector<double>(rows.size(), 0.0));
    for (const mm_shape & s : shapes) {
        const double               bytes  = (double) s.n * ime_row_bytes(s.type, s.k);
        const size_t               copies = (size_t) std::min(32.0, std::ceil(32e6 / bytes));
        const std::vector<uint8_t> q      = quantize_weights(s.type, s.k, s.n, 1);
        std::vector<std::unique_ptr<mm_weight>> ws;
        for (size_t c = 0; c < copies; c++) {
            ws.push_back(std::make_unique<mm_weight>());
            REQUIRE(ws.back()->init_with(dev, s.type, s.k, s.n, q));
        }
        for (size_t r = 0; r < rows.size(); r++) {
            const int64_t                        m = rows[r];
            std::vector<std::unique_ptr<mm_run>> runs;
            for (size_t c = 0; c < copies; c++) {
                runs.push_back(std::make_unique<mm_run>());
                REQUIRE(runs.back()->init(dev, *ws[c], m, 1));
                CHECK(ggml_backend_graph_compute(backend, runs.back()->g_dev) == GGML_STATUS_SUCCESS);
            }
            const int   iters    = m <= 64 ? 10 : m <= 160 ? 5 : 3;
            std::vector<double>      t(n_set, 0.0);
            std::vector<std::string> path(n_set);
            for (int k = 0; k < n_set; k++) {
                set_env("FLAGOS_SPACEMIT_MM", settings[k]);
                bool     ok   = true;
                size_t   next = 0;
                uint64_t path_a, path_b, path_c;
                g_mm_paths_take(&path_a, &path_b, &path_c);
                t[k] = time_us(
                    [&] { ok &= ggml_backend_graph_compute(backend, runs[next++ % copies]->g_dev) == GGML_STATUS_SUCCESS; },
                    iters);
                g_mm_paths_take(&path_a, &path_b, &path_c);
                const std::pair<uint64_t, const char *> taken[] = { { path_a, "A" }, { path_b, "B" }, { path_c, "C" } };
                for (const auto & [count, name] : taken) {
                    if (count > 0) {
                        path[k] += path[k].empty() ? name : std::string("+") + name;
                    }
                }
                CHECK(ok);
            }
            set_env("FLAGOS_SPACEMIT_MM", nullptr);
            const double flop = 2.0 * s.k * s.n * m;
            std::string  line;
            for (int k = 0; k < n_set; k++) {
                char cell[128];
                std::snprintf(cell, sizeof(cell), "%s%s (path %s) %8.0f us %6.1f GFLOP/s", k == 0 ? "" : ", ",
                              settings[k], path[k].c_str(), t[k], flop / t[k] / 1e3);
                line += cell;
                if (k > 0) {
                    std::snprintf(cell, sizeof(cell), " (%+.1f%%)", 100.0 * (t[k] / t[0] - 1.0));
                    line += cell;
                }
                batch[k][r] += t[k] * s.per_batch;
            }
            std::printf("bench    mm %-9s %s %5lldx%-5lld rows %3lld: %s\n", s.what, ggml_type_name(s.type),
                        (long long) s.k, (long long) s.n, (long long) m, line.c_str());
        }
    }
    for (size_t r = 0; r < rows.size(); r++) {
        std::string line;
        for (int k = 0; k < n_set; k++) {
            char cell[96];
            std::snprintf(cell, sizeof(cell), "%s%s %8.1f ms", k == 0 ? "" : ", ", settings[k], batch[k][r] / 1e3);
            line += cell;
            if (k > 0) {
                std::snprintf(cell, sizeof(cell), " (%+.1f%%)", 100.0 * (batch[k][r] / batch[0][r] - 1.0));
                line += cell;
            }
        }
        std::printf("bench    mm per Qwen3-4B batch of %3lld rows (its matmuls but the head): %s\n",
                    (long long) rows[r], line.c_str());
    }
    ggml_backend_free(backend);
    return 0;
}

// flash attention at Qwen3-4B's heads (32 query, 8 KV of 128; attn_run): a token at n_kv 256, 512, 1024, 4096 and
// 8192, every cell valid, and prompt batches, causal: 128 tokens at n_kv 128 and 4096, 512 at n_kv 512 and 4608 (depth
// 0 and 4096); the provider against the CPU backend (8 threads) on the same graph. As in the model, where each layer
// has its own cache, every call reads its cache from DRAM: it takes the next of enough copies (32 MB together, as
// bench_mm). GB/s counts the n_kv cells' K and V bytes once; a token (or batch) is 36 layers' calls, each its own
// launch here. At one token the default (from n_kv 512 the split over the tiles with the turns; plan.md M2e.3, round 2)
// also against FLAGOS_SPACEMIT_ATTN=split (the split without the turns) and one-chunk (the kernels before round 2); a
// few tokens FLAGOS_SPACEMIT_ATTN_KVH=off against 1-8 (below)
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

    // the KV-head split for a few tokens (plan.md "The KV-head split for a few tokens, from SpacemiT's fork"): 1, 2, 3,
    // 4 and 8 tokens at n_kv 256-8192, FLAGOS_SPACEMIT_ATTN_KVH=off against FLAGOS_SPACEMIT_ATTN_KVH=1-8,
    // the two taking turns twice, the faster turn of each kept; the copies and GB/s as above
    for (const int64_t n_kv : { 256, 512, 1024, 4096, 8192 }) {
        const int iterations = n_kv <= 256 ? 30 : n_kv <= 512 ? 25 : n_kv <= 1024 ? 20 : n_kv <= 4096 ? 10 : 5;
        for (const int64_t n_tokens : { 1, 2, 3, 4, 8 }) {
            const attn_shape s        = { 32, 8, 128, n_kv, n_tokens, n_kv - n_tokens };
            const double     kv_bytes = 2.0 * (double) (s.n_kv * s.n_head_kv * s.d) * sizeof(ggml_fp16_t);
            const size_t     copies   = (size_t) std::min(32.0, std::ceil(32e6 / kv_bytes));
            std::vector<std::unique_ptr<attn_run>> runs;
            for (size_t c = 0; c < copies; c++) {
                runs.push_back(std::make_unique<attn_run>());
                REQUIRE(runs.back()->init(dev, s, attn_mask::causal, 0.0f, 0.0f, (uint32_t) c));
                CHECK(ggml_backend_graph_compute(backend, runs.back()->g_dev) == GGML_STATUS_SUCCESS);
            }
            bool       ok     = true;
            size_t     next   = 0;
            const auto dev_us = [&](const char * kvh) {
                env_override attn_switch("FLAGOS_SPACEMIT_ATTN", nullptr);
                env_override kvh_switch("FLAGOS_SPACEMIT_ATTN_KVH", kvh);
                return time_us(
                    [&] {
                        ok &= ggml_backend_graph_compute(backend, runs[next++ % copies]->g_dev) == GGML_STATUS_SUCCESS;
                    },
                    iterations);
            };
            double t_off = HUGE_VAL, t_kvh = HUGE_VAL;
            for (int turn = 0; turn < 2; turn++) {
                t_off = std::min(t_off, dev_us("off"));
                t_kvh = std::min(t_kvh, dev_us("1-8"));
            }
            CHECK(ok);
            std::printf("bench    attn %lld token%s n_kv %4lld, FLAGOS_SPACEMIT_ATTN_KVH: off %7.0f us (%5.1f GB/s); "
                        "1-8 %7.0f us (%+.1f%%, %5.1f GB/s)\n",
                        (long long) n_tokens, n_tokens == 1 ? " " : "s", (long long) n_kv, t_off,
                        kv_bytes / t_off / 1e3, t_kvh, 100.0 * (t_kvh / t_off - 1.0), kv_bytes / t_kvh / 1e3);
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

    // Q4_0 matmul at Qwen3-4B's FFN shapes, and Q4_1 at its ffn_down shape (its first 4 layers): 1 row (the 1-row path),
    // small batches and 64 rows (by default path B where TCM holds the activations next to a slab, here at 4-64 rows
    // for gate and up, 4 and 16 for down, 4 for the Q4_1 down; else path A from 32 rows, else path C), 128 rows (path
    // A; split, the default until 2026-10-25, kept gate and up on path C there, fusion-and-splitting.md §7.3,
    // bench_mm). Q6_K (run as Q8_0, against the CPU's Q6_K) at its output head, whose 1 row is a generated token's: up
    // to 16 rows (path B) and fewer iterations, as the head holds 15x an FFN matrix's weights
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
        REQUIRE(w.init(dev, s.k, s.n, 1, s.type, stored_as_q8_0(s.type)));
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
    return bench_attn(dev) != 0 ? 1 : bench_mm(dev);
}

int main(int argc, char ** argv) {
    bool expect_device = false;
    bool expect_none   = false;
    bool full          = false;
    bool run_bench     = false;
    bool run_attn      = false;
    bool run_mm        = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--expect-device") == 0) {
            expect_device = true;
        } else if (std::strcmp(argv[i], "--expect-none") == 0) {
            expect_none = true;
        } else if (std::strcmp(argv[i], "--full") == 0) {
            full = true;
        } else if (std::strcmp(argv[i], "--bench") == 0) {
            run_bench = true;
        } else if (std::strcmp(argv[i], "--bench-attn") == 0) {
            run_attn = true;
        } else if (std::strcmp(argv[i], "--bench-mm") == 0) {
            run_mm = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--expect-device | --expect-none] [--full] [--bench | --bench-attn | --bench-mm]\n",
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
    if (bind_plugin_names(dev) != 0) {
        return 1;
    }

    // without spine-runtime the provider runs on the host executor (simulated device)
    flagos_device_info info;
    const bool         host = flagos_registry_get_device_info(global_index, &info) && info.profile.runtime != nullptr &&
                      std::strcmp(info.profile.runtime, "none") == 0;
    if (check_registry(dev, global_index) != 0 || check_loader(dev, global_index) != 0 || check_contracts(dev) != 0 ||
        check_buffer(dev) != 0 || check_ops(dev) != 0 ||
        check_mul_mat(dev, full) != 0 || check_gemv(dev, host) != 0 || check_mm_paths(dev, host) != 0 ||
        check_rows(dev) != 0 || check_cache_ops(dev) != 0 || check_flash_attn(dev) != 0 || check_layer(dev) != 0 ||
        check_plan(dev) != 0 || check_diagnostics(dev) != 0) {
        return 1;
    }
    if (run_bench ? bench(dev) != 0 : (run_attn && bench_attn(dev) != 0) || (run_mm && bench_mm(dev) != 0)) {
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "flagos-check-spacemit: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("flagos-check-spacemit: all checks passed\n");
    return 0;
}
