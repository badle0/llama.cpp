// flagos-check-spacemit: checks of the SpacemiT provider through the built ggml-flagos library.
// Usage: flagos-check-spacemit [--expect-device | --expect-none] [--full]
//                              [--bench | --bench-attn | --bench-mm | --bench-f32 | --bench-embd | --bench-repack |
//                               --bench-x100 | --bench-mmid | --bench-pairwait]
//   --expect-device  fail if the provider exposes no device (use on the K3)
//   --expect-none    fail if it exposes one (use with FLAGOS_SPACEMIT_DISABLE=1, or on another host)
//   --full           also check Q4_0 and Q4_1 matmuls at Qwen3-4B's shapes (up to 512 x 9728 x 2560), and the Q6_K
//                    output heads (run as Q8_0) at 128 rows besides 1 and 7 (up to 128 x 2560 x 151936), and
//                    MUL_MAT_ID at Qwen3.6-35B-A3B's expert shapes for Q8_0, Q6_K and Q5_0 too; slow without the IME
//   --bench          also measure launch cost, per-node cost, bandwidth, Q4_0, Q4_1 and Q6_K (run as Q8_0) matmul time
//                    and flash attention time (Qwen3-4B's heads: a token at n_kv 256-8192, 128 tokens at n_kv 128
//                    and 4096, 512 at n_kv 512 and 4608) against the CPU (Q4_1 on the provider only), and the default
//                    matmul setting (fit,staged) against split at the row counts of --bench-mm, and the F32 matmul
//                    (--bench-f32)
//   --bench-attn     of the benchmarks only flash attention (at one token also FLAGOS_SPACEMIT_ATTN=split and
//                    one-chunk, against the default; at 2, 3, 4 and 8 tokens FLAGOS_SPACEMIT_ATTN_KVH=off
//                    against the default) (not run by --bench)
//   --bench-mm       of the benchmarks only the matmuls of a Qwen3-4B batch of 2-512 rows under the default
//                    (fit,staged) against FLAGOS_SPACEMIT_MM=split (the default until 2026-10-25)
//   --bench-f32      of the benchmarks only the F32 matmul (the router's [2048, 256] at 1, 32, 128 and 501 token rows
//                    and [2048, 32] at 1 and 501)
//   --bench-embd     of the benchmarks only the token embedding lookup: one call of 1-512 rows of 1024 and 2560
//                    columns from 32768, Q6_K by ggml's dequantizers, and the same values as F32
//                    copied by get_rows.f32 (not run by --bench)
//   --bench-repack   of the benchmarks only the weights' repack at load: one tensor of 2560 columns per layout (Q4_0,
//                    Q4_1 and Q6_K to Q8_0, Q8_0) on the build's route (8 threads; with spine-runtime the X100
//                    stream, its ranges printed) (not run by --bench)
//   --bench-x100     of the benchmarks only the 1-row matmul beside the X100 cores (head Q6_K 2560x151936, ffn_up
//                    and ffn_down Q4_0): timed while 4 or 8 threads read another 1 GiB buffer, and split by rows
//                    between the provider and the CPU backend on 8 or 4 threads; lines "x100bench ...", then
//                    "x100bench: checks n/n PASS|FAIL". X100BENCH_REPS (5), X100BENCH_SMALL=1 (small shapes) (not run
//                    by --bench)
//   --bench-mmid     of the benchmarks only one MUL_MAT_ID node of Q4_0 experts at Qwen3.6-35B-A3B's two shapes
//                    ([2048, 512, 256] and [512, 2048, 256], 8 ids a token): 1 token, 4 tokens of 32 experts, 4 and 8
//                    tokens of the same 8 experts, 32 tokens of random ids, FLAGOS_SPACEMIT_EXPERT_Q4_0=plain against
//                    the default taking turns twice; median and minimum of 60 computes (not run by --bench)
//   --bench-pairwait of the benchmarks only the pair wait: one launch a variant on all tiles, each making 1,000,000
//                    rendezvous with its partner, spacemit_pair_wait against SpacemiT's barrier (spine_barrier.h)
//                    without and with our two exits, three rounds in rotated order; lines "pairwait ...", skipped on
//                    the serial host executor (not run by --bench)
// Without a flag, a host without the device skips the device checks. Q4_1 weights are checked as the provider stores
// them: requantized to Q8_0.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-flagos.h"
#include "ggml.h"
#include "../ggml-impl.h"  // ggml_cgraph's uid and ggml_graph_view (the plan cache's probes)
#define GGML_COMMON_DECL_CPP
#include "../ggml-common.h"  // block_q6_K (the edge blocks of the Q6_K lookup)
#include "providers/spacemit/core/flagos-spacemit-diagnostics.h"  // the diagnostics under the deck's names (S1 step 5)
#include "providers/spacemit/core/flagos-spacemit-loader.h"  // the plugin loader and its results (S1 step 1)
#include "providers/spacemit/core/flagos-spacemit-plan.h"    // the plan cache's counters (S1 step 3)
#include "providers/spacemit/core/flagos-spacemit-api.h"   // spacemit_device_plugin

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#    include <sys/mman.h>  // mmap, madvise (--bench-x100's buffer)
#    include <unistd.h>    // dup, dup2 (the counters' lines read back from stderr)
#endif
#ifdef __linux__
#    include <sched.h>  // sched_getcpu (--bench-x100's readers)
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
static spacemit_steps_take_fn      g_steps_take      = nullptr;
static spacemit_executors_take_fn  g_executors_take  = nullptr;
static spacemit_mm_paths_take_fn   g_mm_paths_take   = nullptr;
static spacemit_mm_tail_take_fn    g_mm_tail_take    = nullptr;
static spacemit_switches_reset_fn  g_switches_reset  = nullptr;
static spacemit_attn_paths_take_fn g_attn_paths_take = nullptr;
static spacemit_embd_paths_take_fn g_embd_paths_take = nullptr;
static spacemit_gdn_paths_take_fn  g_gdn_paths_take  = nullptr;
static spacemit_repack_take_fn     g_repack_take     = nullptr;
static spacemit_repack_x100_take_fn g_repack_x100_take = nullptr;

static int bind_plugin_names(ggml_backend_dev_t dev) {
    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    g_steps_take     = reinterpret_cast<spacemit_steps_take_fn>(table->get_proc_address("spacemit_steps_take"));
    g_executors_take = reinterpret_cast<spacemit_executors_take_fn>(table->get_proc_address("spacemit_executors_take"));
    g_mm_paths_take  = reinterpret_cast<spacemit_mm_paths_take_fn>(table->get_proc_address("spacemit_mm_paths_take"));
    g_mm_tail_take   = reinterpret_cast<spacemit_mm_tail_take_fn>(table->get_proc_address("spacemit_mm_tail_take"));
    g_switches_reset = reinterpret_cast<spacemit_switches_reset_fn>(table->get_proc_address("spacemit_switches_reset"));
    g_attn_paths_take =
        reinterpret_cast<spacemit_attn_paths_take_fn>(table->get_proc_address("spacemit_attn_paths_take"));
    g_embd_paths_take =
        reinterpret_cast<spacemit_embd_paths_take_fn>(table->get_proc_address("spacemit_embd_paths_take"));
    g_gdn_paths_take =
        reinterpret_cast<spacemit_gdn_paths_take_fn>(table->get_proc_address("spacemit_gdn_paths_take"));
    g_repack_take = reinterpret_cast<spacemit_repack_take_fn>(table->get_proc_address("spacemit_repack_take"));
    g_repack_x100_take =
        reinterpret_cast<spacemit_repack_x100_take_fn>(table->get_proc_address("spacemit_repack_x100_take"));
    REQUIRE(g_steps_take != nullptr && g_executors_take != nullptr && g_mm_paths_take != nullptr &&
            g_mm_tail_take != nullptr && g_switches_reset != nullptr && g_attn_paths_take != nullptr &&
            g_embd_paths_take != nullptr && g_gdn_paths_take != nullptr && g_repack_take != nullptr &&
            g_repack_x100_take != nullptr);
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

// the device is an IGPU (D3)
static int check_registry(ggml_backend_dev_t dev, size_t global_index) {
    CHECK(ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU);
    CHECK(ggml_backend_dev_description(dev) != nullptr);

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    CHECK(props.type == GGML_BACKEND_DEVICE_TYPE_IGPU);
    CHECK(props.memory_total > 0);
    CHECK(props.memory_free > 0 && props.memory_free <= props.memory_total);
    CHECK(!props.caps.async && !props.caps.events && !props.caps.host_buffer);
    CHECK(!props.caps.mmap_support);  // llama.cpp's default load mode then reads the file, it does not map it

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
    const char * names[] = { "spacemit_steps_take", "spacemit_executors_take", "spacemit_mm_paths_take",
                             "spacemit_mm_tail_take", "spacemit_run_stats_attach", "spacemit_mul_mat_same_activations",
                             "spacemit_switches_reset", "spacemit_switches_generation",
                             "spacemit_attn_paths_take", "spacemit_embd_paths_take", "spacemit_gdn_paths_take",
                             "spacemit_repack_take", "spacemit_repack_x100_take" };
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
    CHECK(ops.n_ops == 32);
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
    // contract, taken for 2 to 8 tokens (unset, empty or 2-8; off and any other value: none), with 2, 4 or 8 query
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
            { nullptr, 2, 8 }, { "", 2, 8 }, { "off", 0, 0 }, { "2-8", 2, 8 },
            { "1-8", 0, 0 },  // a former range, taken as off
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
        std::printf("contracts FLASH_ATTN_EXT under FLAGOS_SPACEMIT_ATTN_KVH (unset, empty, off, 2-8, 1-8; "
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
    if (huge < 0.0) {
        std::printf("info     huge pages: not counted (no /proc/self/smaps)\n");
    } else {
        std::printf("info     huge pages: %.0f of 64 MiB of a provider buffer (transparent huge pages: %s)\n", huge / 1048576.0,
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
    ggml_init_params           params = { ggml_tensor_overhead() * 128, nullptr, true };
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
    ggml_tensor * w8_3 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, 256, 32, 2);
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w8_3, x_3d))); // Q8_0, 3-D weight
    // a 3-D weight has a layout (MUL_MAT_ID's experts) and stays refused in our buffer too
    ggml_tensor * probe_3d = ggml_mul_mat(ctx, w_3d, x_3d);
    w_3d->buffer           = ggml_backend_buft_alloc_buffer(buft, 0);
    CHECK(!ggml_backend_dev_supports_op(dev, probe_3d));
    ggml_backend_buffer_free(w_3d->buffer);
    w_3d->buffer = nullptr;
    CHECK(ggml_backend_buft_get_alloc_size(buft, v1_3) == ggml_nbytes(v1_3));  // 3-D Q4_1: the q4_1 min layout
    CHECK(ggml_backend_buft_get_alloc_size(buft, w_3d) == ggml_nbytes(w_3d));
    // Q5_0, stored as Q8_0 exactly: any row length % 32, rows % 32, 2-D only, 34 bytes per 32 values instead of 22
    ggml_tensor * w5   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_0, 256, 64);
    ggml_tensor * w5_k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_0, 2880, 64);
    ggml_tensor * w5_3 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q5_0, 256, 32, 2);
    CHECK(ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w5, x1)));
    CHECK(ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w5, x512)));
    CHECK(ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w5_k, x_k)));
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_0, 256, 16), x1)));
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w5_3, x_3d)));
    CHECK(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, ggml_view_2d(ctx, w5, 256, 32, w5->nb[1], 0), x1)));
    CHECK(ggml_backend_buft_get_alloc_size(buft, w5) == (size_t) 64 * (256 / 32) * 34);
    CHECK(ggml_backend_buft_get_alloc_size(buft, w5_3) == (size_t) 64 * (256 / 32) * 34);
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

    // Q5_0, stored as Q8_0 exactly: the stored bytes are those of the Q8_0 weight with the same d and qs = q - 16, which
    // holds the same values bit for bit, and reads restore the GGUF bytes, whole, in part and after partial writes
    {
        const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        ggml_init_params                 p5   = { ggml_tensor_overhead() * 2, nullptr, true };
        ggml_context *                   ctx5 = ggml_init(p5);
        REQUIRE(ctx5 != nullptr);
        ggml_tensor *         w5   = ggml_new_tensor_2d(ctx5, GGML_TYPE_Q5_0, 512, 64);
        ggml_tensor *         w58  = ggml_new_tensor_2d(ctx5, GGML_TYPE_Q8_0, 512, 64);
        ggml_backend_buffer_t buf5 = ggml_backend_alloc_ctx_tensors_from_buft(ctx5, buft);
        REQUIRE(buf5 != nullptr);
        std::vector<uint8_t> q5 = quantize_weights(GGML_TYPE_Q5_0, 512, 64, 19), r5(q5.size());
        std::vector<uint8_t> q58(ggml_row_size(GGML_TYPE_Q8_0, 512) * 64);
        const block_q5_0 *   b5 = reinterpret_cast<const block_q5_0 *>(q5.data());
        block_q8_0 *         b8 = reinterpret_cast<block_q8_0 *>(q58.data());
        for (int64_t i = 0; i < 64 * 512 / 32; i++) {
            uint32_t qh;
            std::memcpy(&qh, b5[i].qh, sizeof(qh));
            b8[i].d = b5[i].d;
            for (int j = 0; j < 16; j++) {
                b8[i].qs[j]      = (int8_t) (((b5[i].qs[j] & 0x0F) | (((qh >> j) & 1) << 4)) - 16);
                b8[i].qs[j + 16] = (int8_t) (((b5[i].qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16);
            }
        }
        CHECK(dequantize(GGML_TYPE_Q5_0, q5) == dequantize(GGML_TYPE_Q8_0, q58));
        const size_t alloc = ggml_backend_buft_get_alloc_size(buft, w5);
        CHECK(alloc == ggml_backend_buft_get_alloc_size(buft, w58) && alloc > ggml_nbytes(w5));
        ggml_backend_tensor_set(w5, q5.data(), 0, q5.size());
        ggml_backend_tensor_set(w58, q58.data(), 0, q58.size());
        CHECK(std::memcmp(w5->data, w58->data, alloc) == 0);
        ggml_backend_tensor_get(w5, r5.data(), 0, r5.size());
        CHECK(r5 == q5);
        ggml_backend_tensor_get(w5, r5.data(), 1000, 300);
        CHECK(std::memcmp(r5.data(), q5.data() + 1000, 300) == 0);
        ggml_backend_tensor_set(w5, patch.data(), 500, patch.size());
        std::copy(patch.begin(), patch.end(), q5.begin() + 500);
        ggml_backend_tensor_memset(w5, 0x11, 22, 22);
        std::fill(q5.begin() + 22, q5.begin() + 44, (uint8_t) 0x11);
        ggml_backend_tensor_get(w5, r5.data(), 0, r5.size());
        CHECK(r5 == q5);
        ggml_backend_buffer_free(buf5);
        ggml_free(ctx5);
    }

    // a 3-D weight (MUL_MAT_ID's experts, 3 of 64 rows) is stored as the 2-D weight of its 192 rows: the same alloc
    // size, stored bytes and read-back, whole and in part; for the lossless layouts that is the GGUF bytes, and a
    // partial write inside the third expert reads back patched. A 3-D Q4_1 weight takes the q4_1 min layout (2-D Q4_1
    // the Q8_0 route): it is stored repacked in ggml_nbytes and reads back exactly, whole, in part and after a partial
    // write
    {
        const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        ggml_init_params                 p1   = { ggml_tensor_overhead(), nullptr, true };
        ggml_context *                   ctx1 = ggml_init(p1);
        REQUIRE(ctx1 != nullptr);
        ggml_tensor *         t1   = ggml_new_tensor_3d(ctx1, GGML_TYPE_Q4_1, 512, 64, 3);
        ggml_backend_buffer_t buf1 = ggml_backend_alloc_ctx_tensors_from_buft(ctx1, buft);
        REQUIRE(buf1 != nullptr);
        const std::vector<uint8_t> q1_3 = quantize_weights(GGML_TYPE_Q4_1, 512, 192, 17);
        std::vector<uint8_t>       r1_3(q1_3.size());
        CHECK(ggml_backend_buft_get_alloc_size(buft, t1) == ggml_nbytes(t1));
        ggml_backend_tensor_set(t1, q1_3.data(), 0, q1_3.size());
        CHECK(std::memcmp(t1->data, q1_3.data(), q1_3.size()) != 0);  // stored repacked
        ggml_backend_tensor_get(t1, r1_3.data(), 0, r1_3.size());
        CHECK(r1_3 == q1_3);
        const size_t         at1 = ggml_nbytes(t1) / 3 + 1000;  // inside the second expert
        std::vector<uint8_t> part1(300);
        ggml_backend_tensor_get(t1, part1.data(), at1, part1.size());
        CHECK(std::memcmp(part1.data(), q1_3.data() + at1, part1.size()) == 0);
        std::vector<uint8_t> want1 = q1_3;
        std::copy(patch.begin(), patch.end(), want1.begin() + 2 * ggml_nbytes(t1) / 3 + 700);  // the third expert
        ggml_backend_tensor_set(t1, patch.data(), 2 * ggml_nbytes(t1) / 3 + 700, patch.size());
        ggml_backend_tensor_get(t1, r1_3.data(), 0, r1_3.size());
        CHECK(r1_3 == want1);
        ggml_backend_buffer_free(buf1);
        ggml_free(ctx1);
    }
    // FLAGOS_SPACEMIT_EXPERT_Q4_0=plain: by default a 3-D Q4_0 weight has a layout of its own (q4_0s, its round trip
    // in check_mul_mat_id_q4_0s)
    env_override q4_0_plain("FLAGOS_SPACEMIT_EXPERT_Q4_0", "plain");
    for (const ggml_type type : { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K }) {
        const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        ggml_init_params                 p3   = { ggml_tensor_overhead() * 2, nullptr, true };
        ggml_context *                   ctx3 = ggml_init(p3);
        REQUIRE(ctx3 != nullptr);
        ggml_tensor *         t3       = ggml_new_tensor_3d(ctx3, type, 512, 64, 3);
        ggml_tensor *         t2       = ggml_new_tensor_2d(ctx3, type, 512, 192);
        ggml_backend_buffer_t buf3     = ggml_backend_alloc_ctx_tensors_from_buft(ctx3, buft);
        const bool            lossless = !stored_as_q8_0(type);
        REQUIRE(buf3 != nullptr);
        std::vector<uint8_t> q3 = quantize_weights(type, 512, 192, 17), r3(q3.size()), r2(q3.size()), part(300);
        const size_t         alloc = ggml_backend_buft_get_alloc_size(buft, t3);
        CHECK(alloc == ggml_backend_buft_get_alloc_size(buft, t2));
        ggml_backend_tensor_set(t3, q3.data(), 0, q3.size());
        ggml_backend_tensor_set(t2, q3.data(), 0, q3.size());
        CHECK(std::memcmp(t3->data, q3.data(), q3.size()) != 0);  // stored repacked
        CHECK(std::memcmp(t3->data, t2->data, alloc) == 0);
        ggml_backend_tensor_get(t3, r3.data(), 0, r3.size());
        ggml_backend_tensor_get(t2, r2.data(), 0, r2.size());
        CHECK(r3 == r2);
        CHECK(!lossless || r3 == q3);
        const size_t at = ggml_nbytes(t3) / 3 + 1000;  // inside the second expert
        ggml_backend_tensor_get(t3, part.data(), at, part.size());
        CHECK(std::memcmp(part.data(), r3.data() + at, part.size()) == 0);
        if (lossless) {
            const std::vector<uint8_t> patch3(100, 0x5a);
            const size_t               at3 = 2 * ggml_nbytes(t3) / 3 + 700;  // inside the third expert
            ggml_backend_tensor_set(t3, patch3.data(), at3, patch3.size());
            std::copy(patch3.begin(), patch3.end(), q3.begin() + at3);
            ggml_backend_tensor_get(t3, r3.data(), 0, r3.size());
            CHECK(r3 == q3);
        }
        ggml_backend_buffer_free(buf3);
        ggml_free(ctx3);
    }
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
    // the IME int8 kernels against the CPU: Q4_0 at 32 rows and up would take the f16 path (check_mul_mat_f16's)
    env_override   dense("FLAGOS_SPACEMIT_DENSE_Q4_0", "int8");
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
        { GGML_TYPE_Q5_0, 256, 32, { 1, 2, 4, 5, 16, 64, 113, 128 } },
        { GGML_TYPE_Q5_0, 2880, 64, { 1, 5, 113 } },           // 90 K blocks
        { GGML_TYPE_Q5_0, 2048, 512, { 1, 7, 128 } },          // Qwen3.6-35B-A3B's shared expert gate and up
    };
    if (full) {  // Qwen3-4B q, attention output, ffn gate/up, ffn down, and its Q4_1 ffn down
        for (const auto & s : { std::pair<int64_t, int64_t>{ 2560, 4096 }, { 4096, 2560 }, { 2560, 9728 }, { 9728, 2560 } }) {
            shapes.push_back({ GGML_TYPE_Q4_0, s.first, s.second, { 1, 7, 128, 512 } });
        }
        shapes.push_back({ GGML_TYPE_Q4_1, 9728, 2560, { 1, 7, 128, 512 } });
    }

    const ggml_type types[5]       = { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_0 };
    int             n_cases[5]     = { 0, 0, 0, 0, 0 };
    int64_t         max_rows[5]    = { 0, 0, 0, 0, 0 };
    double          max_nmse[5]    = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    double          max_nmse_q8[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };  // against the CPU's Q8_0 of the same values
    for (const shape & s : shapes) {
        const int t = (int) (std::find(types, types + 5, s.type) - types);
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
    std::printf("matmul   Q5_0 (as Q8_0 exactly, IME layout 32x32) vs CPU: %d cases, rows 1-%d, max NMSE %.2e (bound "
                "%.0e)%s\n",
                n_cases[4], (int) max_rows[4], max_nmse[4], k_mul_mat_max_nmse, kernels);
    std::printf("matmul   claims exact (any row count; not other layouts, types, views, 3-D weights, F16 or strided "
                "activations, CPU-buffer weights); Q4_1, Q6_K and Q5_0 allocation sizes; repack round trip (Q4_1 read "
                "as the nearest Q4_1, within NMSE 5e-5; Q6_K read as the nearest Q6_K, within NMSE 1e-5; Q5_0 stored "
                "as the Q8_0 of its values, read exactly; 3-D weights stored and read as their 2-D rows, a 3-D Q4_1 "
                "one in the q4_1 min layout, read exactly); MUL_MAT->ADD graph: checked\n");
    std::printf("info     Q4_1 requantization to Q8_0 alone (CPU, GGUF against ggml's Q8_0 of the values), random weights "
                "3072x1024, 16 rows: NMSE %.2e\n",
                conversion);
    return 0;
}

// a backend with a stand-in of the K3's 384 KiB of TCM, its tiles in turn or (threads "1") on threads. The executor
// reads both settings when it is created and a device's live backends share it, so a backend with other settings
// must wait until this one is freed
static ggml_backend_t tcm_backend(ggml_backend_dev_t dev, const char * threads) {
    env_override tcm("FLAGOS_SPACEMIT_TEST_TCM", "393216");
    env_override on_threads("FLAGOS_SPACEMIT_TEST_THREADS", threads);
    return ggml_backend_dev_init(dev, nullptr);
}

// the 1-row path (generation; plan.md, "1-row matmul bandwidth"): the weights staged through TCM in 32-column slabs, the
// two tiles of a pair taking turns (gemv, flagos-spacemit-ime.cpp). With TCM against the CPU, the output filled with NaN
// first so skipped columns show. On a host without the AI cores the host executor gets a stand-in of the K3's 384 KiB
// (spine-runtime ignores the variable), with the tiles one after another, which counts the pair waits, and on threads
// (FLAGOS_SPACEMIT_TEST_THREADS), where the waits wait; there each must give the same bytes as a backend without TCM,
// which reads the weights straight from DRAM with the same kernel. The shapes give a pair whose odd tile has one slab
// fewer (480 and 1056 columns) and a long row
static int check_gemv(ggml_backend_dev_t dev, bool host) {
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
        if (host) {
            ggml_backend_t direct = nullptr;  // no TCM, the weights read straight from DRAM
            {
                env_override tcm("FLAGOS_SPACEMIT_TEST_TCM", nullptr);
                direct = ggml_backend_dev_init(dev, nullptr);
            }
            REQUIRE(direct != nullptr);
            REQUIRE(ggml_backend_graph_compute(direct, run.g_dev) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(run.y, ref.data(), 0, ggml_nbytes(run.y));
            ggml_backend_free(direct);
        }
        for (const char * threads : { (const char *) nullptr, "1" }) {  // with TCM: in turn, then (host) on threads
            if (threads != nullptr && !host) {
                break;
            }
            ggml_backend_t backend = tcm_backend(dev, threads);
            REQUIRE(backend != nullptr);
            ggml_backend_tensor_set(run.y, nan.data(), 0, ggml_nbytes(run.y));
            const double e = run.compare(dev, backend, cpu);
            ggml_backend_tensor_get(run.y, out.data(), 0, ggml_nbytes(run.y));
            const bool same = !host || std::memcmp(out.data(), ref.data(), ggml_nbytes(run.y)) == 0;
            if (e < 0.0 || e > k_mul_mat_max_nmse || !same) {
                std::fprintf(stderr, "FAIL: %s 1-row matmul k=%lld n=%lld with TCM%s: %s %.3e\n", ggml_type_name(s.type),
                             (long long) s.k, (long long) s.n, threads != nullptr ? " (tiles on threads)" : "",
                             e < 0.0 ? "refused or failed" : !same ? "differs from the read without TCM, NMSE" : "NMSE",
                             e);
                g_failures++;
            } else {
                n_same++;
            }
            worst = std::max(worst, e);
            n_runs++;
            ggml_backend_free(backend);
        }
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
    env_override mm("FLAGOS_SPACEMIT_MM", nullptr);
    env_override dense("FLAGOS_SPACEMIT_DENSE_Q4_0", "int8");  // the int8 path's counters: no f16 steps from 32 rows

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
        for (const char * threads : { (const char *) nullptr, "1" }) {
            if (threads != nullptr && !host) {
                break;
            }
            ggml_backend_t backend = tcm_backend(dev, threads);
            REQUIRE(backend != nullptr);
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
                             paths.c_str(), threads != nullptr ? " (tiles on threads)" : "");
                g_failures++;
            }
            n_same += same && switched;
            n_runs++;
            ggml_backend_free(backend);
        }
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
// call the CPU runs its tiled kernel too (the provider, with TCM, from 9)
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
static int check_flash_attn(ggml_backend_dev_t dev, bool host) {
    // claimed exactly: llama.cpp's graph (here before allocation) at head sizes up to 256, without a mask too; not head
    // size 257, F32 or quantized K and V, or K and V of different types
    ggml_init_params params = { ggml_tensor_overhead() * 256, nullptr, true };
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
    CHECK(ggml_backend_dev_supports_op(dev, fa(129, GGML_TYPE_F16, GGML_TYPE_F16, true)));
    CHECK(ggml_backend_dev_supports_op(dev, fa(192, GGML_TYPE_F16, GGML_TYPE_F16, true)));
    CHECK(ggml_backend_dev_supports_op(dev, fa(256, GGML_TYPE_F16, GGML_TYPE_F16, true)));
    CHECK(ggml_backend_dev_supports_op(dev, fa(256, GGML_TYPE_F16, GGML_TYPE_F16, false)));
    CHECK(!ggml_backend_dev_supports_op(dev, fa(257, GGML_TYPE_F16, GGML_TYPE_F16, true)));
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
    // the default (2 to 8 tokens), against the CPU within the same bounds. Each case must take its contract
    // (spacemit_find_op) and repeat bit for bit in a second run; that its outputs differ from those under
    // FLAGOS_SPACEMIT_ATTN_KVH=off is counted only, since with one chunk per KV head the reference (the Mac) sums as the
    // other kernels' do. The cases take its edges: 3 tokens on Qwen3-4B (3 passes of 4 rows), GQA 2 (8 tokens in one
    // pass) and 8 (2 chunks), 2 KV heads (4 chunks) with the last chunk hidden from tokens 0 and 1, n_kv 1000 and 999
    // (neither a multiple of 64 positions nor, at 999, of the chunk), head size 64, no mask, sinks, softcap, every cell
    // hidden. The two steps also run directly on the host with 3 and 5 tiles (the reference there), which must agree
    // bit for bit
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
        env_override        kvh_switch("FLAGOS_SPACEMIT_ATTN_KVH", nullptr);
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

    // T1 (plan.md "T1 spec"): the tiled kernel below 64 tokens, from 9 (the pins above, KVH off, change nothing at
    // these token counts). Against two references: the CPU on the same call, within the general bounds (below 64 tokens
    // the CPU runs its one-chunk kernel, which sums V in F16), and the CPU's tiled kernel on the same call made 64
    // tokens long, whose first rows are this call's (attn_run draws K and V cell by cell and q token by token from the
    // seed; the mask hides the added cells from these rows), within the tiled bound, which catches a missed visible
    // position. The kernel by the counter: the tiled one on the K3, the reference on the host or under
    // GGML_FLAGOS_KERNEL_VARIANTS=0. A second run must repeat bit for bit. The cases take its edges: 108 rows over 8
    // tiles (tiles start inside a head), rows 0-13 of 26 after 50 seeing none of the second KV tile (one mixed group of
    // 4)
    const bool         expect_reference = host || reference_label()[0] != '\0';
    const char * const kernel_names[3]  = { "tiled", "one-chunk", "reference" };
    const int          expected_kernel  = expect_reference ? 2 : 0;
    struct tiled_case {
        const char * what;
        attn_shape   s;
        attn_mask    mode     = attn_mask::causal;
        float        max_bias = 0.0f;
        float        q_scale  = 4.0f;
        float        softcap  = 0.0f;
        bool         sinks    = false;
        float        visible  = 0.0f;
    };
    std::vector<tiled_case> tiled_cases;
    for (int64_t n_tokens : { 9, 12, 26, 38, 63 }) {
        for (int64_t n_kv : { 256, 1000, 4096 }) {
            tiled_cases.push_back({ "Qwen3-4B", { 32, 8, 128, n_kv, n_tokens, n_kv - n_tokens } });
        }
    }
    tiled_cases.push_back({ "Qwen3-0.6B", { 16, 8, 128, 512, 9, 503 } });
    tiled_cases.push_back({ "Qwen3-0.6B", { 16, 8, 128, 512, 63, 449 } });
    tiled_cases.push_back({ "GQA 8", { 32, 4, 128, 1024, 26, 998 } });
    tiled_cases.push_back({ "GQA 1", { 8, 8, 128, 1024, 26, 998 } });
    tiled_cases.push_back({ "108 rows over 8 tiles, tiles starting inside a head", { 12, 4, 128, 512, 9, 503 } });
    tiled_cases.push_back({ "26 after 50, one group of 4 mixed over the second KV tile", { 32, 8, 128, 256, 26, 50 } });
    tiled_cases.push_back({ "head size 64", { 32, 8, 64, 1024, 26, 998 } });
    tiled_cases.push_back({ "no mask", { 32, 8, 128, 1024, 26, 998 }, attn_mask::none });
    tiled_cases.push_back({ "sinks", { 32, 8, 128, 1024, 26, 998 }, attn_mask::causal, 0.0f, 4.0f, 0.0f, true });
    tiled_cases.push_back({ "softcap 2", { 32, 8, 128, 1024, 26, 998 }, attn_mask::causal, 0.0f, 4.0f, 2.0f });
    tiled_cases.push_back({ "ALiBi (max_bias 8) with GQA 4", { 32, 8, 128, 1024, 26, 998 }, attn_mask::alibi, 8.0f });
    tiled_cases.push_back({ "every cell hidden, sinks", { 32, 8, 128, 1024, 26, 998 }, attn_mask::causal, 0.0f, 4.0f,
                            0.0f, true, -INFINITY });
    tiled_cases.push_back({ "flat scores at depth", { 32, 8, 128, 4096, 26, 4070 }, attn_mask::causal, 0.0f, 1.0f });
    int    n_tiled_ran = 0, n_tiled_repeat = 0;
    double e_tiled = 0.0, e_tiled_row = 0.0, e_tiled_row64 = 0.0;
    for (const tiled_case & c : tiled_cases) {
        const uint32_t case_seed = seed++;
        attn_run       run;
        REQUIRE(run.init(dev, c.s, c.mode, c.visible, c.max_bias, case_seed, c.q_scale, c.softcap, c.sinks));
        REQUIRE(ggml_backend_graph_compute(cpu, run.g_cpu) == GGML_STATUS_SUCCESS);
        // the second reference: 64 tokens from the same n_past, the cells raised so that the mask has room
        attn_shape s64 = c.s;
        s64.n_tokens   = 64;
        s64.n_kv       = c.mode == attn_mask::none ? c.s.n_kv : std::max(c.s.n_kv, c.s.n_past + 64);
        attn_run run64;
        REQUIRE(run64.init(dev, s64, c.mode, c.visible, c.max_bias, case_seed, c.q_scale, c.softcap, c.sinks));
        REQUIRE(ggml_backend_graph_compute(cpu, run64.g_cpu) == GGML_STATUS_SUCCESS);
        const int64_t      n = ggml_nelements(run.out);
        std::vector<float> tiled[2];
        uint64_t           count[3];
        g_attn_paths_take(&count[0], &count[1], &count[2]);
        REQUIRE(run.dev_output(backend, tiled[0]));
        g_attn_paths_take(&count[0], &count[1], &count[2]);
        REQUIRE(run.dev_output(backend, tiled[1]));
        const std::string what = std::string("tiled below 64 tokens, ") + c.what;
        const attn_error  e    = attn_errors(tiled[0].data(), static_cast<const float *>(run.out_ref->data), n,
                                             run.out->ne[0]);
        attn_report(e, (what + ", against the CPU").c_str(), c.s);
        const attn_error e64 = attn_errors(tiled[0].data(), static_cast<const float *>(run64.out_ref->data), n,
                                           run.out->ne[0]);
        attn_report(e64, (what + ", against the CPU's tiled kernel on 64 tokens").c_str(), c.s,
                    k_attn_max_row_nmse_tiled);
        e_tiled       = std::max(e_tiled, e.all);
        e_tiled_row   = std::max(e_tiled_row, e.row);
        e_tiled_row64 = std::max(e_tiled_row64, e64.row);
        bool ran = true;
        for (int k = 0; k < 3; k++) {
            ran = ran && count[k] == (k == expected_kernel ? 1u : 0u);
        }
        if (!ran) {
            std::fprintf(stderr,
                         "FAIL: attention tiled below 64 tokens, %s, %lld tokens at n_kv %lld: counted tiled %llu, "
                         "one-chunk %llu, reference %llu (expected one %s)\n",
                         c.what, (long long) c.s.n_tokens, (long long) c.s.n_kv, (unsigned long long) count[0],
                         (unsigned long long) count[1], (unsigned long long) count[2], kernel_names[expected_kernel]);
            g_failures++;
        }
        n_tiled_ran += ran;
        const bool repeat = std::memcmp(tiled[0].data(), tiled[1].data(), n * sizeof(float)) == 0;
        if (!repeat) {
            std::fprintf(stderr,
                         "FAIL: attention tiled below 64 tokens, %s, %lld tokens at n_kv %lld: two runs differ\n",
                         c.what, (long long) c.s.n_tokens, (long long) c.s.n_kv);
            g_failures++;
        }
        n_tiled_repeat += repeat;
    }

    // head sizes above 128 (plan.md "Step 2 spec", B: Qwen3.5's 256; 129 and 192 for the edges). On the K3 the backend
    // runs the RVV kernels (above 128 in segments of 128) on the AI cores, on the host and under
    // GGML_FLAGOS_KERNEL_VARIANTS=0 the references. Each case must take its contract (one-step, paired split, KV-head
    // split) under its switches, a one-step case its kernel by the counter (tiled or one-chunk; the reference on the
    // host), and run on 8 tiles within the bounds against the CPU. Its steps called directly on the CPU-buffer node on 1
    // and 8 tiles (the split without the turns) run on the calling core, so the references (VLEN below 1024 on the
    // K3's host cores too): within the bounds, and where the backend ran the references too, on 8 tiles the backend's
    // bytes, on 1 tile too but for the split, whose partial sums depend on the tile count. The cases select, on the K3
    // with 8 tiles: the 4-row one-chunk kernel (32 query heads: 4 rows a tile), the 2-row one (16 query heads: 2 rows a
    // tile), both over several packed K/V tiles at n_kv 1024, with softcap and sinks and without a mask; the 1-row one
    // (ALiBi; 7 tokens with KV-head split off, whose rows share no mask row), with sinks and softcap; the tiled kernel
    // (26, 99 after 37 and 128 tokens, softcap and sinks, no mask); the split (n_kv 1000 and 1024, 4096, no mask); the
    // KV-head split with 2, 4, 8 and 16 rows a pass (GQA 2 with 3, 2 and 4 tokens; GQA 4 with 4), softcap and sinks
    struct wide_case {
        const char * what;
        attn_shape   s;
        const char * first;            // the contract's first step
        int          kernel   = -1;    // a one-step case's kernel by the counter: 0 tiled, 1 one-chunk
        attn_mask    mode     = attn_mask::causal;
        float        max_bias = 0.0f;
        float        softcap  = 0.0f;
        bool         sinks    = false;
        const char * attn     = nullptr;  // FLAGOS_SPACEMIT_ATTN, FLAGOS_SPACEMIT_ATTN_KVH (nullptr: the default)
        const char * kvh      = nullptr;
    };
    const char * const     one_step = "flash_attn";
    const char * const     split_p  = "flash_attn.split.partial_paired";
    const char * const     kvh_p    = "flash_attn.kvh.partial";
    const attn_mask        causal   = attn_mask::causal;
    std::vector<wide_case> wide_cases;
    for (int64_t d : { 129, 192, 256 }) {
        wide_cases.push_back({ "1 token, 37 of 256 cells valid", { 16, 4, d, 256, 1, 36 }, one_step, 1 });
        wide_cases.push_back({ "1 token", { 16, 4, d, 1024, 1, 1023 }, split_p });
        wide_cases.push_back({ "4 tokens", { 16, 4, d, 1024, 4, 1020 }, kvh_p });
        wide_cases.push_back({ "7 tokens", { 16, 4, d, 256, 7, 249 }, kvh_p });
        wide_cases.push_back({ "batch of 26", { 16, 4, d, 256, 26, 100 }, one_step, 0 });
        wide_cases.push_back({ "batch of 128", { 16, 4, d, 256, 128, 0 }, one_step, 0 });
        wide_cases.push_back({ "1 token, 4 rows a tile", { 32, 8, d, 256, 1, 36 }, one_step, 1 });
        wide_cases.push_back({ "1 token, GQA 2, one-chunk", { 16, 8, d, 1024, 1, 1023 }, one_step, 1, causal, 0.0f, 0.0f,
                               false, "one-chunk" });
        wide_cases.push_back({ "1 token, 4 rows a tile, softcap 2, sinks", { 32, 8, d, 1024, 1, 1023 }, one_step, 1,
                               causal, 0.0f, 2.0f, true });
        wide_cases.push_back({ "1 token, 4 rows a tile, no mask, one-chunk", { 32, 8, d, 1024, 1, 1023 }, one_step, 1,
                               attn_mask::none, 0.0f, 0.0f, false, "one-chunk" });
        wide_cases.push_back({ "1 token, ALiBi (max_bias 8) with GQA 4", { 16, 4, d, 256, 1, 255 }, one_step, 1,
                               attn_mask::alibi, 8.0f });
        wide_cases.push_back({ "7 tokens, KV-head split off", { 16, 4, d, 256, 7, 249 }, one_step, 1, causal, 0.0f,
                               0.0f, false, nullptr, "off" });
        wide_cases.push_back({ "7 tokens, softcap 2, sinks, KV-head split off", { 16, 4, d, 256, 7, 249 }, one_step, 1,
                               causal, 0.0f, 2.0f, true, nullptr, "off" });
        wide_cases.push_back({ "batch of 26, softcap 2, sinks", { 16, 4, d, 256, 26, 100 }, one_step, 0, causal, 0.0f,
                               2.0f, true });
        wide_cases.push_back({ "batch of 26, no mask", { 16, 4, d, 256, 26, 100 }, one_step, 0, attn_mask::none });
        wide_cases.push_back({ "99 after 37, partly masked KV tiles", { 16, 4, d, 256, 99, 37 }, one_step, 0 });
        wide_cases.push_back({ "1 token, n_kv 1000, no mask", { 16, 4, d, 1000, 1, 999 }, split_p, -1, attn_mask::none });
        wide_cases.push_back({ "2 tokens, GQA 2", { 16, 8, d, 1024, 2, 1022 }, kvh_p });
        wide_cases.push_back({ "3 tokens, GQA 2", { 16, 8, d, 1024, 3, 1021 }, kvh_p });
        wide_cases.push_back({ "4 tokens, GQA 2", { 16, 8, d, 1024, 4, 1020 }, kvh_p });
        wide_cases.push_back({ "4 tokens, softcap 2, sinks", { 16, 4, d, 1024, 4, 1020 }, kvh_p, -1, causal, 0.0f, 2.0f,
                               true });
    }
    wide_cases.push_back({ "1 token at 4096", { 16, 4, 256, 4096, 1, 4095 }, split_p });
    wide_cases.push_back({ "Qwen3.5-0.8B-like, 4 tokens, 4 chunks", { 8, 2, 256, 1024, 4, 1020 }, kvh_p });
    const auto named = [&](const char * name) -> const spacemit_plugin_kernel * {
        for (uint32_t k = 0; kernels != nullptr && k < n_kernels; k++) {
            if (kernels[k].name != nullptr && std::strcmp(kernels[k].name, name) == 0) {
                return &kernels[k];
            }
        }
        return nullptr;
    };
    int    n_wide_taken = 0, n_wide_same8 = 0, n_wide_same1 = 0, n_wide_same1_cases = 0;
    int    n_wide_one_step = 0, n_wide_kernel = 0;
    double e_wide = 0.0, e_wide_row = 0.0, e_wide_row_tiled = 0.0;
    for (const wide_case & c : wide_cases) {
        env_override attn_default("FLAGOS_SPACEMIT_ATTN", c.attn);
        env_override kvh_default("FLAGOS_SPACEMIT_ATTN_KVH", c.kvh);
        attn_run     run;
        REQUIRE(run.init(dev, c.s, c.mode, 0.0f, c.max_bias, seed++, 4.0f, c.softcap, c.sinks));
        REQUIRE(ggml_backend_graph_compute(cpu, run.g_cpu) == GGML_STATUS_SUCCESS);
        const int64_t            n = ggml_nelements(run.out);
        const float *            r = static_cast<const float *>(run.out_ref->data);
        const std::vector<float> ref(r, r + n);
        const bool               split = std::strncmp(c.first, "flash_attn.split", 16) == 0;
        // the tiled row bound where the CPU runs its tiled kernel: 64 tokens or more and DV a multiple of its vector
        // width (ops.cpp, use_tiled: 4 floats on NEON and RVV, at most 16), so not at 129
        const bool   cpu_tiled = c.s.n_tokens >= k_attn_tiled_tokens && c.s.d % 16 == 0;
        const double row_bound = cpu_tiled ? k_attn_max_row_nmse_tiled : k_attn_max_row_nmse;

        const spacemit_op * got   = spacemit_find_op(ops, run.out, buft);
        const bool          taken = got != nullptr && got->steps[0] == named(c.first);
        if (!taken) {
            std::fprintf(stderr, "FAIL: attention head size %lld, %s: contract %s not taken\n", (long long) c.s.d,
                         c.what, c.first);
            g_failures++;
        }
        n_wide_taken += taken;
        std::vector<float> o;
        uint64_t           count[3];
        g_attn_paths_take(&count[0], &count[1], &count[2]);
        REQUIRE(run.dev_output(backend, o));
        g_attn_paths_take(&count[0], &count[1], &count[2]);
        const std::string what = std::string("head size ") + std::to_string(c.s.d) + ", " + c.what;
        if (c.kernel >= 0) {
            const int kernel = expect_reference ? 2 : c.kernel;
            bool      ran    = true;
            for (int k = 0; k < 3; k++) {
                ran = ran && count[k] == (k == kernel ? 1u : 0u);
            }
            if (!ran) {
                std::fprintf(stderr,
                             "FAIL: attention %s: counted tiled %llu, one-chunk %llu, reference %llu (expected one "
                             "%s)\n",
                             what.c_str(), (unsigned long long) count[0], (unsigned long long) count[1],
                             (unsigned long long) count[2], kernel_names[kernel]);
                g_failures++;
            }
            n_wide_one_step++;
            n_wide_kernel += ran;
        }
        const attn_error  e    = attn_errors(o.data(), ref.data(), n, run.out->ne[0]);
        attn_report(e, (what + ", 8 tiles").c_str(), c.s, row_bound);
        e_wide                = std::max(e_wide, e.all);
        double & worst_row_of = cpu_tiled ? e_wide_row_tiled : e_wide_row;
        worst_row_of          = std::max(worst_row_of, e.row);

        ggml_tensor * node = run.out_ref;
        for (uint32_t nth : { 1u, 8u }) {
            env_override        no_turns("FLAGOS_SPACEMIT_ATTN", split ? "split" : c.attn);
            const spacemit_op * op = spacemit_find_op(ops, node, buft);
            REQUIRE(op != nullptr);
            const size_t       ws = spacemit_op_workspace(op, node);
            std::vector<float> workspace(ws / sizeof(float) + 1, NAN);
            float *            out = static_cast<float *>(node->data);
            std::fill(out, out + n, NAN);
            bool ok = true;
            for (uint32_t step = 0; step < op->n_steps; step++) {
                for (uint32_t ith = 0; ith < nth; ith++) {
                    const spacemit_tile tile = { ith, nth, nullptr, 0, workspace.data(), ws, nullptr };
                    ok &= op->steps[step]->fn(&tile, node);
                }
            }
            CHECK(ok);
            const attn_error ed = attn_errors(out, ref.data(), n, run.out->ne[0]);
            attn_report(ed, (what + ", directly on " + std::to_string(nth) + " tiles").c_str(), c.s, row_bound);
            const bool same = std::memcmp(out, o.data(), n * sizeof(float)) == 0;
            if (expect_reference && (nth == 8 || !split)) {
                if (!same) {
                    std::fprintf(stderr, "FAIL: attention %s: directly on %u tiles differs from the backend's run\n",
                                 what.c_str(), nth);
                    g_failures++;
                }
                (nth == 8 ? n_wide_same8 : n_wide_same1) += same;
                n_wide_same1_cases += nth == 1;
            }
        }
    }

    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    std::printf(
        "attn     FLASH_ATTN_EXT vs CPU, one-chunk and tiled kernels (F16 KV cache in a CPU buffer, viewed as "
        "llama.cpp does; Qwen3-4B and 0.6B heads): %zu cases (1 token at n_kv 256-4096; 7, 99 and 128 tokens; 128 "
        "after 3968), max NMSE %.2e (bound %.0e), worst row %.2e (bound %.0e), of the tiled cases %.2e (bound %.0e); "
        "probes: every score far below 0 (mask -30, -120) %.2e, no mask %.2e, ALiBi with GQA 4 %.2e; claims exact "
        "(F16 K and V, head sizes to 256): checked%s\n",
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
    std::printf("attn     KV-head split for a few tokens (the default, 2-8 tokens) on %zu cases (n_kv "
                "256-4096, GQA 2, 4 and 8, 1, 2 and 4 chunks, sinks, softcap, no mask, every cell hidden): max NMSE "
                "%.2e (bound %.0e), worst row %.2e (bound %.0e); its contract taken in %d/%zu; %d/%zu outputs differ "
                "from those under off; two runs identical in %d/%zu; 3 and 5 tiles called directly identical in "
                "%d/%d%s\n",
                kvh_cases.size(), e_kvh, k_attn_max_nmse, e_kvh_row, k_attn_max_row_nmse, n_kvh_taken,
                kvh_cases.size(), n_kvh_differ, kvh_cases.size(), n_kvh_repeat, kvh_cases.size(), n_kvh_direct_same,
                n_kvh_direct, reference_label());
    std::printf("attn     tiled kernel below 64 tokens (9-63 tokens, the default) on %zu cases (Qwen3-4B 9-63 tokens "
                "at n_kv 256-4096, Qwen3-0.6B, GQA 8 and 1, tiles starting inside a head, a mixed group of 4, head "
                "size 64, no mask, sinks, softcap, ALiBi with GQA 4, every cell hidden, flat scores at depth): vs CPU "
                "max NMSE %.2e (bound %.0e), worst row %.2e (bound %.0e); vs the CPU's tiled kernel on 64 tokens worst "
                "row %.2e (bound %.0e); the %s ran in %d/%zu; two runs identical in %d/%zu%s\n",
                tiled_cases.size(), e_tiled, k_attn_max_nmse, e_tiled_row, k_attn_max_row_nmse, e_tiled_row64,
                k_attn_max_row_nmse_tiled, expect_reference ? "reference" : "tiled kernel", n_tiled_ran,
                tiled_cases.size(), n_tiled_repeat, tiled_cases.size(), reference_label());
    std::printf("attn     head sizes 129, 192 and 256 (16 or 32 query and 4 or 8 KV heads; 8 and 2 at 256) on %zu "
                "cases, the backend on the %s (1 token at n_kv 256-4096: one-chunk 4- and 2-row, ALiBi 1-row, split; "
                "2-7 tokens: KV-head split 2-16 rows a pass, 1-row with it off; batches of 26, 99 and 128; softcap, "
                "sinks, no mask): its contract taken in %d/%zu, the one-step kernel by the counter in %d/%d; vs CPU "
                "max NMSE %.2e (bound %.0e), worst row %.2e (bound %.0e), of the 99- and 128-token batches at 192 and "
                "256 %.2e (bound %.0e), on 8 tiles and directly on 1 and 8; directly on 8 tiles the backend's bytes in "
                "%d/%d, on 1 tile in %d/%d (the split's left out; compared only where the backend ran the "
                "references)\n",
                wide_cases.size(), expect_reference ? "references" : "RVV kernels", n_wide_taken, wide_cases.size(),
                n_wide_kernel, n_wide_one_step, e_wide, k_attn_max_nmse, e_wide_row, k_attn_max_row_nmse,
                e_wide_row_tiled, k_attn_max_row_nmse_tiled, n_wide_same8, expect_reference ? (int) wide_cases.size() : 0,
                n_wide_same1, n_wide_same1_cases);
    return 0;
}

//
// T6: a weight's repack on several host threads (plan.md "T6 spec"), against the one-thread bytes
//

static int count_lines_with(const std::string & text, const char * what) {
    int n = 0;
    for (size_t at = 0; at < text.size();) {
        const size_t end = std::min(text.find('\n', at), text.size());
        n += text.substr(at, end - at).find(what) != std::string::npos;
        at = end + 1;
    }
    return n;
}

// per layout, weights of 32, 64, 96, 1056 and 4096 rows of 2560 columns, 64 and 96 rows of 32768, one group of 65536
// columns (over 1 MiB in one group) and three 3-D weights (3 experts of 32 rows of 2560, 8 of 64 and 4 of 512 rows of
// 2048; Q4_1's in the q4_1 min layout), each written whole: by the counters the
// ranges ran, exactly for the weights of at least 1 MiB and several groups, on min(8, groups) threads without
// spine-runtime (spert) and through the X100 stream with it (at most its cores + 1 ranges), and for those the stored
// bytes (the whole alloc size, past ggml_nbytes for the layouts requantized to Q8_0) and the read-back are the
// one-thread ones: the same rows written piece by piece as weights of one group (3-D: of 2 experts of 32 rows), which
// repack on the calling thread. Then a partial write (Q4_0, Q8_0, Q5_0) on the same route against the whole write of
// the patched bytes
static int check_repack_threads(ggml_backend_dev_t dev, bool spert) {
    ggml_backend_buffer_type_t buft    = ggml_backend_dev_buffer_type(dev);
    const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_0 };
    // mats > 1: a 3-D weight of mats experts (MUL_MAT_ID's), cut on its ggml_nrows rows
    const struct {
        int64_t rows, cols, mats;
    } shapes[] = { { 32, 2560, 1 },  { 64, 2560, 1 },  { 96, 2560, 1 },  { 1056, 2560, 1 }, { 4096, 2560, 1 },
                   { 64, 32768, 1 }, { 96, 32768, 1 }, { 32, 65536, 1 }, { 32, 2560, 3 },   { 64, 2048, 8 },
                   { 512, 2048, 4 } };
    const size_t  min_bytes = size_t(1) << 20;
    const int64_t n_threads = 8;
    int           n_cmp = 0, n_raw_same = 0, n_read_same = 0, n_path = 0, n_path_ok = 0, n_threaded = 0;
    bool          seen_ranges[9] = {};
    uint32_t      seed           = 71;
    // the route of a write: with spert through the X100 stream, 2 to want ranges; else on want threads
    const auto route_ok = [&](bool split, uint64_t want, uint64_t threaded, uint64_t ranges, uint64_t through,
                              uint64_t not_through, uint64_t reran) {
        if (not_through != 0 || reran != 0) {
            return false;
        }
        if (!split) {
            return threaded == 0 && ranges == 0 && through == 0;
        }
        return threaded == 1 &&
               (spert ? through == 1 && ranges >= 2 && ranges <= want : through == 0 && ranges == want);
    };
    for (const ggml_type type : types) {
        for (const auto & s : shapes) {
            ggml_init_params params = { 2 * ggml_tensor_overhead(), nullptr, true };
            ggml_context *   ctx    = ggml_init(params);
            REQUIRE(ctx != nullptr);
            const int64_t piece_mats = s.mats > 1 ? 2 : 1;
            ggml_tensor * w          = ggml_new_tensor_3d(ctx, type, s.cols, s.rows, s.mats);
            ggml_tensor * piece      = ggml_new_tensor_3d(ctx, type, s.cols, 32, piece_mats);
            ggml_set_name(w, s.mats > 1 ? "blk.0.ffn_up_exps.weight" : "blk.0.ffn_up.weight");
            ggml_set_name(piece, ggml_get_name(w));
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            REQUIRE(buf != nullptr);
            const std::vector<uint8_t> q      = quantize_weights(type, s.cols, s.rows * s.mats, seed++);
            const size_t               alloc  = ggml_backend_buft_get_alloc_size(buft, w);
            const int64_t              groups = s.rows * s.mats / 32;
            const bool                 big    = ggml_nbytes(w) >= min_bytes && groups > 1;
            const uint8_t *            raw    = static_cast<const uint8_t *>(w->data);  // host memory underneath
            std::vector<uint8_t>       read(q.size());

            ggml_backend_buffer_clear(buf, 0xa5);  // a range left out shows
            uint64_t threaded = 0, ranges = 0, through = 0, not_through = 0, reran = 0;
            g_repack_take(&threaded, &ranges);
            g_repack_x100_take(&through, &not_through, &reran);
            ggml_backend_tensor_set(w, q.data(), 0, q.size());
            g_repack_take(&threaded, &ranges);
            g_repack_x100_take(&through, &not_through, &reran);
            ggml_backend_tensor_get(w, read.data(), 0, read.size());
            const uint64_t want    = big ? (uint64_t) std::min(n_threads, groups) : 0;
            const bool     path_ok = route_ok(big, want, threaded, ranges, through, not_through, reran);
            n_path++;
            n_path_ok += path_ok;
            n_threaded += threaded != 0;
            if (path_ok && big) {
                seen_ranges[ranges] = true;
            }
            if (!path_ok) {
                std::fprintf(stderr,
                             "FAIL: repack %s %lldx%lld: %llu threaded with %llu ranges, %llu through the X100 stream, "
                             "%llu not, %llu rerun (expected %s)\n",
                             ggml_type_name(type), (long long) (s.rows * s.mats), (long long) s.cols,
                             (unsigned long long) threaded, (unsigned long long) ranges, (unsigned long long) through,
                             (unsigned long long) not_through, (unsigned long long) reran,
                             !big ? "none" : spert ? "1 through the stream" : "1 threaded");
                g_failures++;
            }
            CHECK(std::memcmp(raw, q.data(), q.size()) != 0);  // stored repacked
            if (big) {
                // the one-thread bytes: the same rows, one piece after another
                const size_t  piece_alloc = ggml_backend_buft_get_alloc_size(buft, piece);
                const size_t  piece_bytes = ggml_nbytes(piece);
                const int64_t n_pieces    = groups / piece_mats;
                REQUIRE(groups % piece_mats == 0 && piece_alloc * (size_t) n_pieces == alloc &&
                        piece_bytes * (size_t) n_pieces == q.size());
                const std::vector<uint8_t> write_raw(raw, raw + alloc);
                std::vector<uint8_t>       raw_ref(alloc), read_ref(q.size());
                g_repack_take(&threaded, &ranges);
                for (int64_t p = 0; p < n_pieces; p++) {
                    ggml_backend_tensor_set(piece, q.data() + (size_t) p * piece_bytes, 0, piece_bytes);
                    std::memcpy(raw_ref.data() + (size_t) p * piece_alloc, piece->data, piece_alloc);
                    ggml_backend_tensor_get(piece, read_ref.data() + (size_t) p * piece_bytes, 0, piece_bytes);
                }
                g_repack_take(&threaded, &ranges);
                CHECK(threaded == 0);  // every piece on the calling thread
                const bool raw_same  = write_raw == raw_ref;
                const bool read_same = read == read_ref;
                n_cmp++;
                n_raw_same += raw_same;
                n_read_same += read_same;
                if (!raw_same || !read_same) {
                    std::fprintf(stderr, "FAIL: repack %s %lldx%lld in %llu ranges: stored bytes %s, read-back %s\n",
                                 ggml_type_name(type), (long long) (s.rows * s.mats), (long long) s.cols,
                                 (unsigned long long) want, raw_same ? "same" : "differ",
                                 read_same ? "same" : "differs");
                    g_failures++;
                }
            }
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }

    // a partial write repacks the whole weight again (unpack, patch, pack) on the same route: the bytes of a whole
    // write of the patched bytes (these layouts read back the GGUF bytes)
    int n_partial = 0, n_partial_same = 0;
    for (const ggml_type type : { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0 }) {
        ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
        ggml_context *   ctx    = ggml_init(params);
        REQUIRE(ctx != nullptr);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, type, 2560, 4096);
        ggml_set_name(w, "blk.0.ffn_up.weight");
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        REQUIRE(buf != nullptr);
        const std::vector<uint8_t> q = quantize_weights(type, 2560, 4096, seed++);
        const std::vector<uint8_t> patch(100000, 0x3c);
        std::vector<uint8_t>       patched = q;
        std::memcpy(patched.data() + 12345, patch.data(), patch.size());
        const size_t    alloc    = ggml_backend_buft_get_alloc_size(buft, w);
        const uint8_t * raw      = static_cast<const uint8_t *>(w->data);
        uint64_t        threaded = 0, ranges = 0, through = 0, not_through = 0, reran = 0;
        ggml_backend_buffer_clear(buf, 0xa5);
        ggml_backend_tensor_set(w, patched.data(), 0, patched.size());
        const std::vector<uint8_t> raw_ref(raw, raw + alloc);
        ggml_backend_buffer_clear(buf, 0xa5);
        ggml_backend_tensor_set(w, q.data(), 0, q.size());
        g_repack_take(&threaded, &ranges);
        g_repack_x100_take(&through, &not_through, &reran);
        ggml_backend_tensor_set(w, patch.data(), 12345, patch.size());
        g_repack_take(&threaded, &ranges);
        g_repack_x100_take(&through, &not_through, &reran);
        const bool same = std::memcmp(raw, raw_ref.data(), alloc) == 0 &&
                          route_ok(true, (uint64_t) n_threads, threaded, ranges, through, not_through, reran);
        n_partial++;
        n_partial_same += same;
        CHECK(same);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    std::string range_list;
    for (int r = 8; r >= 2; r--) {
        if (seen_ranges[r]) {
            const char * sep = range_list.empty() ? "" : range_list.find(" and ") == std::string::npos ? " and " : ", ";
            range_list       = std::to_string(r) + sep + range_list;
        }
    }
    std::printf("repack   whole writes (Q4_0, Q4_1 and Q6_K to Q8_0, Q8_0, Q5_0 as Q8_0; 32, 64, 96, 1056 and 4096 "
                "rows of 2560, 64 and 96 of 32768, 32 of 65536; 3-D, Q4_1 in the q4_1 min layout: 3 x 32 of 2560, 8 x "
                "64 and 4 x 512 of 2048): the route by the counters as expected in %d/%d (%d split, %s, with %s "
                "ranges: at least 1 MiB and several groups only); of those the stored bytes identical to the "
                "one-thread ones (the rows written as weights of one group, 3-D of 2 experts of 32 rows) in %d/%d, "
                "read-back identical in %d/%d; a partial write on the same route identical to the whole write of the "
                "patched bytes in %d/%d\n",
                n_path_ok, n_path, n_threaded, spert ? "through the X100 stream" : "on threads", range_list.c_str(),
                n_raw_same, n_cmp, n_read_same, n_cmp, n_partial_same, n_partial);
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

// ROPE in IMROPE mode (plan.md "Step 2 spec", C), forward, as Qwen3.5 builds it (src/models/qwen35.cpp,
// ggml_rope_multi): head 256 rotated over n_dims 64, sections 11, 11, 10, 0, base 1e7, n_ctx_orig 262144, no YaRN; q
// [256, 16, T] and k [256, 4, T] at T 1 and 7, text positions (t = h = w, e = 0) and distinct ones; then a whole head
// rotated with YaRN and frequency factors, F16 rows, strided rows. Against the CPU within the wide NEOX case's bound,
// on 8 tiles (the output in our buffer) and the kernel entry called directly on 1 and 8 tiles, which must give the
// backend's bytes; the claims: modes MROPE and VISION, IMROPE with no section or sections past the row refused
struct rope_imrope_stats {
    int    n_cases = 0, n_ok = 0, n_identical = 0, n_direct_same = 0;
    double worst   = 0.0;
};

static void rope_imrope_case(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu,
                             const spacemit_plugin_kernel * kernel, const char * what, ggml_type type,
                             const std::array<int64_t, 3> & ne, int n_dims, const std::array<int, 4> & sections,
                             float freq_base, float ext_factor, bool with_ff, bool strided, bool text_pos,
                             uint32_t seed, rope_imrope_stats & st) {
    ggml_init_params params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context *   ctx_h  = ggml_init(params);
    ggml_context *   ctx_d  = ggml_init(params);
    // strided: rows of ne[0] in a tensor of rows of ne[0] + 6
    ggml_tensor *    base = ggml_new_tensor_3d(ctx_h, type, ne[0] + (strided ? 6 : 0), ne[1], ne[2]);
    ggml_tensor *    x    = strided ? ggml_view_3d(ctx_h, base, ne[0], ne[1], ne[2], base->nb[1], base->nb[2], 0) : base;
    ggml_tensor *    pos  = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, 4 * ne[2]);
    ggml_tensor *    ff   = with_ff ? ggml_new_tensor_1d(ctx_h, GGML_TYPE_F32, n_dims / 2) : nullptr;
    int              sec[4] = { sections[0], sections[1], sections[2], sections[3] };
    const auto       rope = [&](ggml_context * ctx) {
        return ggml_rope_multi(ctx, x, pos, ff, n_dims, sec, GGML_ROPE_TYPE_IMROPE, 262144, freq_base, 1.0f,
                                     ext_factor, 1.0f, 32.0f, 1.0f);
    };
    ggml_tensor *         y_ref = rope(ctx_h);
    ggml_tensor *         y     = rope(ctx_d);
    ggml_backend_buffer_t bh    = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_t bd    = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, ggml_backend_dev_buffer_type(dev));
    bool                  ok    = false;
    double                e     = -1.0;
    if (bh != nullptr && bd != nullptr) {
        const std::vector<float> bv = random_values(ggml_nelements(base), seed, 1.0f);
        if (type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> h(bv.size());
            ggml_fp32_to_fp16_row(bv.data(), h.data(), (int64_t) bv.size());
            ggml_backend_tensor_set(base, h.data(), 0, ggml_nbytes(base));
        } else {
            ggml_backend_tensor_set(base, bv.data(), 0, ggml_nbytes(base));
        }
        std::vector<int32_t> pv(4 * ne[2]);
        for (int64_t t = 0; t < ne[2]; t++) {
            const int32_t p = (int32_t) (t * 37 + 5);
            pv[t]           = p;
            pv[t + ne[2]]     = text_pos ? p : p + 1000;
            pv[t + 2 * ne[2]] = text_pos ? p : 3 * p + 7;
            pv[t + 3 * ne[2]] = text_pos ? 0 : 50000 - p;
        }
        ggml_backend_tensor_set(pos, pv.data(), 0, ggml_nbytes(pos));
        if (ff != nullptr) {
            std::vector<float> fv = random_values(n_dims / 2, seed + 1, 1.0f);
            for (float & f : fv) {
                f = 1.5f + f;  // in [0.5, 2.5)
            }
            ggml_backend_tensor_set(ff, fv.data(), 0, ggml_nbytes(ff));
        }
        ggml_cgraph * gd = ggml_new_graph(ctx_d);
        ggml_cgraph * gc = ggml_new_graph(ctx_h);
        ggml_build_forward_expand(gd, y);
        ggml_build_forward_expand(gc, y_ref);
        if (ggml_backend_dev_supports_op(dev, y) && ggml_backend_graph_compute(backend, gd) == GGML_STATUS_SUCCESS &&
            ggml_backend_graph_compute(cpu, gc) == GGML_STATUS_SUCCESS) {
            const int64_t        n = ggml_nelements(y);
            std::vector<uint8_t> got(ggml_nbytes(y)), want(ggml_nbytes(y_ref));
            ggml_backend_tensor_get(y, got.data(), 0, got.size());
            ggml_backend_tensor_get(y_ref, want.data(), 0, want.size());
            const auto as_f32 = [&](const std::vector<uint8_t> & b) {
                return type == GGML_TYPE_F16 ? f16_to_f32(b) :
                                               std::vector<float>((const float *) b.data(), (const float *) b.data() + n);
            };
            const std::vector<float> fg = as_f32(got), fw = as_f32(want);
            e                           = nmse(fg.data(), fw.data(), n);
            st.n_identical += got == want;
            // the kernel entry directly on the CPU-buffer node, its output first filled with junk
            bool same_direct = true;
            for (uint32_t nth : { 1u, 8u }) {
                const size_t       ws = kernel->workspace_size(y_ref);
                std::vector<float> workspace(ws / sizeof(float) + 1, NAN);
                std::vector<uint8_t> junk(want.size(), 0xA5);
                ggml_backend_tensor_set(y_ref, junk.data(), 0, junk.size());
                bool direct_ok = true;
                for (uint32_t ith = 0; ith < nth; ith++) {
                    const spacemit_tile tile = { ith, nth, nullptr, 0, workspace.data(), ws, nullptr };
                    direct_ok &= kernel->fn(&tile, y_ref);
                }
                std::vector<uint8_t> direct(want.size());
                ggml_backend_tensor_get(y_ref, direct.data(), 0, direct.size());
                same_direct = same_direct && direct_ok && direct == got;
            }
            st.n_direct_same += same_direct;
            ok = e <= 1e-7 && same_direct;
        }
    }
    st.n_cases++;
    st.n_ok += ok;
    st.worst = std::max(st.worst, e < 0.0 ? HUGE_VAL : e);
    if (!ok) {
        std::fprintf(stderr, "FAIL: ROPE IMROPE %s: %s %.3e (bound 1e-7)\n", what,
                     e < 0.0 ? "refused or failed" : "NMSE or direct calls differ from the backend's run,", e);
        g_failures++;
    }
    ggml_backend_buffer_free(bh);
    ggml_backend_buffer_free(bd);
    ggml_free(ctx_h);
    ggml_free(ctx_d);
}

static int rope_imrope_check(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu) {
    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    const spacemit_plugin_kernel * kernel    = nullptr;
    for (uint32_t k = 0; kernels != nullptr && k < n_kernels; k++) {
        if (kernels[k].name != nullptr && std::strcmp(kernels[k].name, "rope") == 0) {
            kernel = &kernels[k];
        }
    }
    REQUIRE(kernel != nullptr && kernel->workspace_size != nullptr);

    rope_imrope_stats         st;
    const std::array<int, 4>  q35 = { 11, 11, 10, 0 };
    uint32_t                  seed = 700;
    for (int64_t t : { 1, 7 }) {
        for (bool text : { true, false }) {
            const std::string pos_kind = text ? "text positions" : "distinct t, h, w, e positions";
            rope_imrope_case(dev, backend, cpu, kernel, ("q [256, 16, " + std::to_string(t) + "], " + pos_kind).c_str(),
                             GGML_TYPE_F32, { 256, 16, t }, 64, q35, 1e7f, 0.0f, false, false, text, seed++, st);
            rope_imrope_case(dev, backend, cpu, kernel, ("k [256, 4, " + std::to_string(t) + "], " + pos_kind).c_str(),
                             GGML_TYPE_F32, { 256, 4, t }, 64, q35, 1e7f, 0.0f, false, false, text, seed++, st);
        }
    }
    rope_imrope_case(dev, backend, cpu, kernel, "whole head, YaRN, frequency factors", GGML_TYPE_F32, { 256, 4, 7 },
                     256, { 24, 20, 20, 0 }, 1e6f, 1.0f, true, false, false, seed++, st);
    rope_imrope_case(dev, backend, cpu, kernel, "F16 rows", GGML_TYPE_F16, { 256, 4, 7 }, 64, q35, 1e7f, 0.0f, false,
                     false, false, seed++, st);
    rope_imrope_case(dev, backend, cpu, kernel, "strided rows", GGML_TYPE_F32, { 256, 4, 7 }, 64, q35, 1e7f, 0.0f,
                     false, true, false, seed++, st);

    // the claims, before allocation
    ggml_init_params p   = { ggml_tensor_overhead() * 16, nullptr, true };
    ggml_context *   ctx = ggml_init(p);
    REQUIRE(ctx != nullptr);
    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 4, 7);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * 7);
    const auto    multi = [&](int mode, std::array<int, 4> s) {
        return ggml_rope_multi(ctx, x, pos, nullptr, 64, s.data(), mode, 262144, 1e7f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    };
    const bool claims = ggml_backend_dev_supports_op(dev, multi(GGML_ROPE_TYPE_IMROPE, q35)) &&
                        !ggml_backend_dev_supports_op(dev, multi(GGML_ROPE_TYPE_MROPE, q35)) &&
                        !ggml_backend_dev_supports_op(dev, multi(GGML_ROPE_TYPE_VISION, { 32, 32, 0, 0 })) &&
                        !ggml_backend_dev_supports_op(dev, multi(GGML_ROPE_TYPE_IMROPE, { 0, 0, 0, 0 })) &&
                        !ggml_backend_dev_supports_op(dev, multi(GGML_ROPE_TYPE_IMROPE, { 0, 0, 0, 11 })) &&
                        !ggml_backend_dev_supports_op(dev, multi(GGML_ROPE_TYPE_IMROPE, { 200, 57, 0, 0 }));
    ggml_free(ctx);
    if (!claims) {
        std::fprintf(stderr, "FAIL: ROPE IMROPE claims: Qwen3.5's refused, or MROPE, VISION, no section or sections "
                             "past the row taken\n");
        g_failures++;
    }
    std::printf("rope     IMROPE forward vs CPU (Qwen3.5's q and k at 1 and 7 tokens, text positions and distinct t, h, w, e; a "
                "whole head with YaRN and frequency factors, F16 rows, strided rows): %d/%d within NMSE 1e-7 (max "
                "%.2e), %d/%d bit for bit; directly on 1 and 8 tiles the backend's bytes in %d/%d; MROPE, VISION, "
                "IMROPE with no section or sections past the row refused: %s%s\n",
                st.n_ok, st.n_cases, st.worst, st.n_identical, st.n_cases, st.n_direct_same, st.n_cases,
                claims ? "checked" : "FAILED", reference_label());
    return 0;
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
// the token embedding on the provider (plan.md "Embedding lookup spec"): GET_ROWS of quantized rows in a CPU buffer, as
// llama.cpp keeps token_embd, by ggml's own dequantizer per row, so the output is the CPU's byte for byte
//

// rows of the test vocabularies: a multiple of 32, so by the layout rule alone Q6_K and Q4_0 would have an IME layout
// (as token_embd has, 151936 rows); in a CPU buffer they keep ggml's
constexpr int64_t k_vocab = 4096;

// a vocabulary of k_vocab rows of nc random values quantized to type, in a CPU buffer marked as weights, as llama.cpp
// keeps token_embd (the scheduler places a node by its weights' buffer); with edges, Q6_K blocks built by hand
// (q6_K_edge_rows) instead
struct embd_table {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         w   = nullptr;

    bool init(ggml_type type, int64_t nc, uint32_t seed, bool edges = false);

    ~embd_table() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

// Q6_K blocks by hand for the edges of ggml's arithmetic, (d * scale) * q in F32: block p takes by the digits of p one
// of 12 values of d (0, -0, 1, -1, a negative one, the smallest subnormal, the largest negative subnormal, the
// smallest normal, 65504, -65504, a small one, a random normal), one of 5 sets of scales (all -128, all 127, all 0,
// cycling -128, 127, 0, -1 and 1, random) and one of 4 sets of 6-bit values (all 0, all 63, alternating 0 and 63,
// random), so every combination occurs among 240 consecutive blocks
static std::vector<uint8_t> q6_K_edge_rows(int64_t nc, int64_t n, uint32_t seed) {
    static_assert(sizeof(block_q6_K) == 210, "block_q6_K");
    const int64_t           nb = nc / QK_K * n;
    std::vector<block_q6_K> x(nb);
    uint64_t                state = seed;
    const auto              next  = [&state] {  // splitmix64
        uint64_t z = (state += 0x9e3779b97f4a7c15ull);
        z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z          = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    };
    static const uint16_t d_bits[] = { 0x0000, 0x8000, 0x3c00, 0xbc00, 0xb555, 0x0001,
                                       0x83ff, 0x0400, 0x7bff, 0xfbff, 0x1234, 0 };
    static const int8_t   cycle[]  = { -128, 127, 0, -1, 1 };
    for (int64_t p = 0; p < nb; p++) {
        block_q6_K & b       = x[p];
        const int    d_mode  = (int) (p % 12);
        const int    sc_mode = (int) (p / 12 % 5);
        const int    q_mode  = (int) (p / 60 % 4);
        uint16_t     d       = d_bits[d_mode];
        if (d_mode == 11) {  // a random normal: exponent 1-30, any sign and mantissa
            const uint64_t r = next();
            d                = (uint16_t) ((r & 0x83ff) | ((1 + (r >> 16) % 30) << 10));
        }
        std::memcpy(&b.d, &d, sizeof(d));
        for (int j = 0; j < QK_K / 16; j++) {
            b.scales[j] = sc_mode == 0 ? (int8_t) -128 : sc_mode == 1 ? (int8_t) 127 : sc_mode == 2 ? (int8_t) 0 :
                          sc_mode == 3 ? cycle[j % 5] : (int8_t) (next() & 0xff);
        }
        uint8_t q[QK_K];  // the 6-bit values in output order
        for (int k = 0; k < QK_K; k++) {
            q[k] = q_mode == 0 ? 0 : q_mode == 1 ? 63 : q_mode == 2 ? (k % 2 ? 63 : 0) : (uint8_t) (next() & 63);
        }
        std::memset(b.ql, 0, sizeof(b.ql));
        std::memset(b.qh, 0, sizeof(b.qh));
        for (int h = 0; h < 2; h++) {  // dequantize_row_q6_K's layout, inverted
            for (int l = 0; l < 32; l++) {
                const uint8_t * v = q + 128 * h + l;
                b.ql[64 * h + l]      = (uint8_t) ((v[0] & 0xf) | ((v[64] & 0xf) << 4));
                b.ql[64 * h + l + 32] = (uint8_t) ((v[32] & 0xf) | ((v[96] & 0xf) << 4));
                b.qh[32 * h + l] = (uint8_t) ((v[0] >> 4) | ((v[32] >> 4) << 2) | ((v[64] >> 4) << 4) | ((v[96] >> 4) << 6));
            }
        }
    }
    std::vector<uint8_t> bytes(nb * sizeof(block_q6_K));
    std::memcpy(bytes.data(), x.data(), bytes.size());
    return bytes;
}

bool embd_table::init(ggml_type type, int64_t nc, uint32_t seed, bool edges) {
    ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
    ctx                     = ggml_init(params);
    w                       = ggml_new_tensor_2d(ctx, type, nc, k_vocab);
    ggml_set_name(w, "token_embd.weight");
    buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    if (buf == nullptr || (edges && type != GGML_TYPE_Q6_K)) {
        return false;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const std::vector<uint8_t> q = edges ? q6_K_edge_rows(nc, k_vocab, seed) : quantize_weights(type, nc, k_vocab, seed);
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
    return true;
}

// the kernel the quantized lookups since the last call ran, by the counter (spacemit_embd_paths_take): ggml's
// dequantizers, or none
static const char * embd_kernel_taken() {
    uint64_t calls = 0;
    g_embd_paths_take(&calls);
    return calls != 0 ? "ggml" : "none";
}

// n token ids, unordered, every fifth a repeat of an earlier one; from 7 on also the first and the last row
static std::vector<int32_t> token_ids(int64_t n, uint32_t seed) {
    std::vector<int32_t> ids(n);
    for (int64_t i = 0; i < n; i++) {
        ids[i] = i % 5 == 4 ? ids[i / 2] : (int32_t) (((uint64_t) i * 2654435761u + seed * 40503u) % k_vocab);
    }
    if (n >= 7) {
        ids[1] = 0;
        ids[2] = (int32_t) k_vocab - 1;
    }
    return ids;
}

// the rows of n ids on the provider (output in our buffer, first filled with NaNs) and on the CPU backend: 1 if the
// bytes are the same, 0 if not, -1 if refused or failed
static int lookup_identical(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu, const embd_table & t,
                            int64_t n, uint32_t seed) {
    ggml_init_params params = { ggml_tensor_overhead() * 4 + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context *   ctx_h  = ggml_init(params);
    ggml_context *   ctx_d  = ggml_init(params);
    ggml_tensor *    idx    = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, n);
    ggml_tensor *    y_ref  = ggml_get_rows(ctx_h, t.w, idx);
    ggml_tensor *    y      = ggml_get_rows(ctx_d, t.w, idx);
    ggml_backend_buffer_t bh = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_t bd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, ggml_backend_dev_buffer_type(dev));
    int                   same = -1;
    if (bh != nullptr && bd != nullptr) {
        ggml_backend_buffer_clear(bd, 0xff);
        const std::vector<int32_t> ids = token_ids(n, seed);
        ggml_backend_tensor_set(idx, ids.data(), 0, ggml_nbytes(idx));
        ggml_cgraph * gd = ggml_new_graph(ctx_d);
        ggml_cgraph * gc = ggml_new_graph(ctx_h);
        ggml_build_forward_expand(gd, y);
        ggml_build_forward_expand(gc, y_ref);
        if (ggml_backend_dev_supports_op(dev, y) && ggml_backend_graph_compute(backend, gd) == GGML_STATUS_SUCCESS &&
            ggml_backend_graph_compute(cpu, gc) == GGML_STATUS_SUCCESS) {
            std::vector<uint8_t> out(ggml_nbytes(y));
            ggml_backend_tensor_get(y, out.data(), 0, out.size());
            same = std::memcmp(out.data(), y_ref->data, out.size()) == 0 ? 1 : 0;
        }
    }
    ggml_backend_buffer_free(bh);
    ggml_backend_buffer_free(bd);
    ggml_free(ctx_h);
    ggml_free(ctx_d);
    return same;
}

// which GET_ROWS the provider claims, and by which contract: F32 rows by get_rows.f32 as before; Q4_0, Q4_1, Q5_0,
// Q5_1, Q8_0, Q4_K, Q5_K and Q6_K rows by get_rows.quant from a CPU buffer whatever the layout rule says, from our buffer
// (or not allocated, as test-backend-ops asks) only where the rule gives ggml's layout; no other type, no I64 indices,
// no strided rows, no partial block. Returns the number of nodes checked
static int lookup_claims(ggml_backend_dev_t dev) {
    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    spacemit_ops               ops;
    std::string                detail;
    if (table == nullptr || !spacemit_ops_bind(table, &ops, &detail)) {
        std::fprintf(stderr, "FAIL: GET_ROWS claims: no contracts (%s)\n", detail.c_str());
        g_failures++;
        return 0;
    }
    const ggml_backend_buffer_type_t buft      = ggml_backend_dev_buffer_type(dev);
    int                              n_checked = 0;
    // want: the contract's step name, or "" for refused; find_op and supports_op must agree
    const auto expect = [&](const std::string & what, const ggml_tensor * node, const char * want) {
        const spacemit_op * op      = spacemit_find_op(ops, node, buft);
        const std::string   got     = op != nullptr ? op->steps[0]->name : "";
        const bool          claimed = ggml_backend_dev_supports_op(dev, node);
        if (got != want || claimed != (*want != '\0')) {
            std::fprintf(stderr, "FAIL: GET_ROWS claims, %s: %s (supports_op %d), expected %s\n", what.c_str(),
                         got.empty() ? "refused" : got.c_str(), claimed ? 1 : 0, *want != '\0' ? want : "refused");
            g_failures++;
        }
        n_checked++;
    };

    ggml_init_params params  = { ggml_tensor_overhead() * 256, nullptr, true };
    ggml_context *   ctx     = ggml_init(params);  // not allocated, or in a zero-size buffer of ours
    ggml_context *   ctx_cpu = ggml_init(params);
    ggml_tensor *    idx     = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 7);
    ggml_tensor *    idx64   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 7);
    ggml_tensor *    q6_cpu  = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_Q6_K, 1024, 64);
    ggml_tensor *    q4_cpu  = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_Q4_0, 1024, 64);
    ggml_tensor *    f32_cpu = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, 1024, 64);
    ggml_tensor *    f16_cpu = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F16, 1024, 64);
    ggml_tensor *    q6_ime  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 1024, 64);  // the layout rule: q6_k_q8_0_32x32
    ggml_tensor *    q4_ime  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 1024, 64);  // q4_0_32x256
    ggml_tensor *    q6_few  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 1024, 5);   // 5 rows: ggml's layout
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cpu, ggml_backend_cpu_buffer_type());
    if (buf_cpu == nullptr) {
        std::fprintf(stderr, "FAIL: GET_ROWS claims: no CPU buffer\n");
        g_failures++;
        ggml_free(ctx);
        ggml_free(ctx_cpu);
        return 0;
    }

    expect("Q6_K in a CPU buffer", ggml_get_rows(ctx, q6_cpu, idx), "get_rows.quant");
    expect("Q4_0 in a CPU buffer", ggml_get_rows(ctx, q4_cpu, idx), "get_rows.quant");
    expect("F32 in a CPU buffer", ggml_get_rows(ctx, f32_cpu, idx), "get_rows.f32");
    expect("F16 in a CPU buffer", ggml_get_rows(ctx, f16_cpu, idx), "");
    expect("the first 32 rows of Q6_K in a CPU buffer",
           ggml_get_rows(ctx, ggml_view_2d(ctx, q6_cpu, 1024, 32, q6_cpu->nb[1], 0), idx), "get_rows.quant");
    expect("every other row of Q6_K in a CPU buffer",
           ggml_get_rows(ctx, ggml_view_2d(ctx, q6_cpu, 1024, 32, 2 * q6_cpu->nb[1], 0), idx), "");
    // ggml_get_rows asserts I32 indices: the node is built with them, then given I64 ones
    ggml_tensor * by_i64 = ggml_get_rows(ctx, q6_cpu, idx);
    by_i64->src[1]       = idx64;
    expect("I64 indices", by_i64, "");
    // ggml_new_tensor makes no partial block: a Q4_0 row of 2 blocks is cut to 48 values, with the strides of one block
    ggml_tensor * cut    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 64, 5);
    ggml_tensor * by_cut = ggml_get_rows(ctx, cut, idx);
    cut->ne[0] = by_cut->ne[0] = 48;
    cut->nb[1]                 = ggml_type_size(GGML_TYPE_Q4_0);
    cut->nb[2] = cut->nb[3] = cut->nb[1] * 5;
    expect("Q4_0 rows of 48 values", by_cut, "");

    // our buffer: an IME layout refused, ggml's taken; not allocated, the same answers
    for (const bool allocated : { false, true }) {
        const std::string where = allocated ? " in our buffer" : " not allocated";
        for (ggml_tensor * w : { q6_ime, q4_ime, q6_few }) {
            w->buffer = allocated ? ggml_backend_buft_alloc_buffer(buft, 0) : nullptr;
        }
        expect("Q6_K of 64 rows (IME layout)" + where, ggml_get_rows(ctx, q6_ime, idx), "");
        expect("Q4_0 of 64 rows (IME layout)" + where, ggml_get_rows(ctx, q4_ime, idx), "");
        expect("a view of Q6_K of 64 rows (IME layout)" + where,
               ggml_get_rows(ctx, ggml_view_2d(ctx, q6_ime, 1024, 32, q6_ime->nb[1], 0), idx), "");
        expect("Q6_K of 5 rows (ggml's layout)" + where, ggml_get_rows(ctx, q6_few, idx), "get_rows.quant");
        for (ggml_tensor * w : { q6_ime, q4_ime, q6_few }) {
            if (w->buffer != nullptr) {
                ggml_backend_buffer_free(w->buffer);
                w->buffer = nullptr;
            }
        }
    }

    // every type, 5 rows not allocated (ggml's layout): exactly the listed block types and F32
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const ggml_type type = (ggml_type) t;
        if (ggml_blck_size(type) == 0 || ggml_type_size(type) == 0 || 1024 % ggml_blck_size(type) != 0) {
            continue;
        }
        const bool   listed = type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 ||
                            type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_K ||
                            type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
        const char * want   = listed ? "get_rows.quant" : type == GGML_TYPE_F32 ? "get_rows.f32" : "";
        expect(std::string(ggml_type_name(type)) + " of 5 rows", ggml_get_rows(ctx, ggml_new_tensor_2d(ctx, type, 1024, 5), idx),
               want);
    }

    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx);
    ggml_free(ctx_cpu);
    return n_checked;
}

// an index out of range fails the launch (the CPU asserts), past the end and below 0, among 7 rows and as the one row
// (tile 0 alone), and the node after the lookup is not run: its output keeps what it held. Returns the launches that
// behaved, of 4
static int lookup_bad_index_runs(ggml_backend_dev_t dev, ggml_backend_t backend, const embd_table & t) {
    ggml_init_params params = { ggml_tensor_overhead() * 8 + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context *   ctx_h  = ggml_init(params);
    ggml_context *   ctx_d  = ggml_init(params);
    ggml_tensor *    idx7   = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, 7);
    ggml_tensor *    idx1   = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, 1);
    ggml_tensor *    n7     = ggml_rms_norm(ctx_d, ggml_get_rows(ctx_d, t.w, idx7), 1e-6f);
    ggml_tensor *    n1     = ggml_rms_norm(ctx_d, ggml_get_rows(ctx_d, t.w, idx1), 1e-6f);
    ggml_backend_buffer_t bh = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_t bd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, ggml_backend_dev_buffer_type(dev));
    int                   ok = 0;
    if (bh != nullptr && bd != nullptr) {
        ggml_cgraph * g7 = ggml_new_graph(ctx_d);
        ggml_cgraph * g1 = ggml_new_graph(ctx_d);
        ggml_build_forward_expand(g7, n7);
        ggml_build_forward_expand(g1, n1);
        const std::vector<int32_t> good = token_ids(7, 3);
        ggml_backend_tensor_set(idx7, good.data(), 0, ggml_nbytes(idx7));
        ggml_backend_tensor_set(idx1, good.data(), 0, ggml_nbytes(idx1));
        // with valid indices both graphs run, so the failures below come from the kernel, not a refusal
        if (ggml_backend_graph_compute(backend, g7) != GGML_STATUS_SUCCESS ||
            ggml_backend_graph_compute(backend, g1) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "FAIL: GET_ROWS of Q6_K with valid indices did not run\n");
            g_failures++;
            ok = -1;
        }
        const struct {
            ggml_tensor * idx, * out;
            ggml_cgraph * g;
            int64_t       at;
            int32_t       id;
        } cases[] = {
            { idx7, n7, g7, 3, (int32_t) k_vocab }, { idx7, n7, g7, 6, -1 },
            { idx1, n1, g1, 0, (int32_t) k_vocab }, { idx1, n1, g1, 0, -5 },
        };
        for (const auto & c : cases) {
            if (ok < 0) {
                break;
            }
            std::vector<int32_t> ids = good;
            ids[c.at]                = c.id;
            ggml_backend_tensor_set(c.idx, ids.data(), 0, ggml_nbytes(c.idx));
            ggml_backend_tensor_memset(c.out, 0x5a, 0, ggml_nbytes(c.out));
            const bool           failed = ggml_backend_graph_compute(backend, c.g) == GGML_STATUS_FAILED;
            std::vector<uint8_t> after(ggml_nbytes(c.out));
            ggml_backend_tensor_get(c.out, after.data(), 0, after.size());
            ok += failed && std::all_of(after.begin(), after.end(), [](uint8_t b) { return b == 0x5a; });
        }
    }
    ggml_backend_buffer_free(bh);
    ggml_backend_buffer_free(bd);
    ggml_free(ctx_h);
    ggml_free(ctx_d);
    return ok;
}

// the lookup and the first op of a layer (build_inp_embd, then attn_norm) through a scheduler of the provider and the
// CPU, the ids in a CPU buffer, the norm weight in ours: inputs and weights allocated beforehand, the graph left to the
// scheduler, as diag_layer
struct lookup_graph {
    ggml_context *        ctx_h = nullptr, * ctx_w = nullptr, * ctx_g = nullptr;
    ggml_backend_buffer_t bh = nullptr, bw = nullptr;
    ggml_tensor *         ids = nullptr, * x = nullptr, * norm = nullptr, * y = nullptr;
    ggml_cgraph *         g = nullptr;

    bool init(ggml_backend_dev_t dev, const embd_table & t, int64_t n) {
        ggml_init_params params = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ctx_h                   = ggml_init(params);
        ctx_w                   = ggml_init(params);
        ctx_g                   = ggml_init(params);
        ids                     = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, n);
        ggml_tensor * nw        = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, t.w->ne[0]);
        ggml_set_name(nw, "blk.0.attn_norm.weight");
        bh = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
        bw = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, ggml_backend_dev_buffer_type(dev));
        if (bh == nullptr || bw == nullptr) {
            return false;
        }
        ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        const std::vector<int32_t> iv = token_ids(n, 11);
        const std::vector<float>   wv = random_values(nw->ne[0], 12, 1.0f);
        ggml_backend_tensor_set(ids, iv.data(), 0, ggml_nbytes(ids));
        ggml_backend_tensor_set(nw, wv.data(), 0, ggml_nbytes(nw));
        x    = ggml_get_rows(ctx_g, t.w, ids);
        norm = ggml_rms_norm(ctx_g, x, 1e-6f);
        y    = ggml_mul(ctx_g, norm, nw);
        ggml_set_output(x);
        ggml_set_output(y);
        g = ggml_new_graph(ctx_g);
        ggml_build_forward_expand(g, y);
        return true;
    }

    ~lookup_graph() {
        for (ggml_backend_buffer_t b : { bh, bw }) {
            ggml_backend_buffer_free(b);
        }
        for (ggml_context * c : { ctx_h, ctx_w, ctx_g }) {
            ggml_free(c);
        }
    }
};

static int check_embedding(ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);
    REQUIRE(g_embd_paths_take != nullptr);

    // Q6_K as Qwen3's token_embd (1024 and 2560 columns), one type each of the legacy and the K family; Q6_K of one
    // block per row and with the edge blocks (q6_K_edge_rows), at 1-3 ids (a row's blocks over the tiles, or a few rows)
    // and from 9; ggml's dequantizers by the counter
    const std::vector<int64_t> ids_first = { 1, 7, 64, 512 };
    const std::vector<int64_t> ids_few   = { 1, 2, 3, 9, 64, 512 };
    const struct {
        ggml_type                    type;
        int64_t                      nc;
        bool                         edges;
        const std::vector<int64_t> & ids;
    } tables[] = {
        { GGML_TYPE_Q6_K, 1024, false, ids_first }, { GGML_TYPE_Q6_K, 2560, false, ids_first },
        { GGML_TYPE_Q4_0, 1024, false, ids_first }, { GGML_TYPE_Q4_K, 1024, false, ids_first },
        { GGML_TYPE_Q6_K, 256, false, ids_few },    { GGML_TYPE_Q6_K, 2560, false, ids_few },
        { GGML_TYPE_Q6_K, 2560, true, ids_few },
    };
    int n_cases = 0, n_same = 0, n_kernel = 0, bad_ok = 0, n_bad = 0;
    for (const auto & tb : tables) {
        embd_table t;
        REQUIRE(t.init(tb.type, tb.nc, 21, tb.edges));
        {
            const char * want = "ggml";
            for (int64_t n : tb.ids) {
                (void) embd_kernel_taken();
                const int    same   = lookup_identical(dev, backend, cpu, t, n, (uint32_t) (n + tb.nc));
                const char * kernel = embd_kernel_taken();
                if (same != 1 || std::strcmp(kernel, want) != 0) {
                    std::fprintf(stderr,
                                 "FAIL: GET_ROWS of %lld %s rows of %lld%s from a CPU buffer: %s, kernel %s (expected "
                                 "%s)\n",
                                 (long long) n, ggml_type_name(tb.type), (long long) tb.nc,
                                 tb.edges ? " (edge blocks)" : "",
                                 same == 1 ? "identical" : same < 0 ? "refused or failed" : "bytes differ from the CPU's",
                                 kernel, want);
                    g_failures++;
                }
                n_cases++;
                n_same += same == 1;
                n_kernel += std::strcmp(kernel, want) == 0;
            }
            if (tb.type == GGML_TYPE_Q6_K && tb.nc == 1024) {
                (void) embd_kernel_taken();
                const int    bad    = lookup_bad_index_runs(dev, backend, t);
                const char * kernel = embd_kernel_taken();  // a failed launch's tile 0 may skip the step: not "mixed"
                if (bad != 4 || std::strcmp(kernel, want) != 0) {
                    std::fprintf(stderr,
                                 "FAIL: GET_ROWS of Q6_K, index out of range: %d of 4 launches as expected, kernel %s "
                                 "(expected %s)\n",
                                 bad, kernel, want);
                    g_failures++;
                }
                bad_ok += std::strcmp(kernel, want) == 0 ? std::max(bad, 0) : 0;
                n_bad += 4;
            }
        }
    }
    const int n_claims = lookup_claims(dev);

    // through the scheduler: 1 split with the lookup on the provider, 2 with it on the CPU (GGML_FLAGOS_DISABLE_OPS),
    // the same bytes. Then, under uid-trust, new ids on the kept graph (a decode reusing its
    // graph): one hit by uid, and the rows of the new ids as the CPU's
    embd_table t;
    REQUIRE(t.init(GGML_TYPE_Q6_K, 1024, 22));
    int                  splits[2] = { 0, 0 };
    bool                 placed[2] = { false, false };
    std::vector<uint8_t> out[2];
    bool                 ran = true, reused = false;
    uint64_t             hits = 0;
    for (int i = 0; i < 2; i++) {
        env_override         switch_off("GGML_FLAGOS_DISABLE_OPS", i == 1 ? "GET_ROWS" : nullptr);
        env_override         plan_switch("FLAGOS_SPACEMIT_PLAN", "uid-trust");
        env_override         quantize_switch("FLAGOS_SPACEMIT_QUANTIZE", nullptr);
        lookup_graph         l;
        REQUIRE(l.init(dev, t, 7));
        ggml_backend_t       both[2] = { backend, cpu };
        ggml_backend_sched_t sched   = ggml_backend_sched_new(both, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
        const bool ok = sched != nullptr && ggml_backend_sched_graph_compute(sched, l.g) == GGML_STATUS_SUCCESS;
        ran           = ran && ok;
        if (ok) {
            splits[i] = ggml_backend_sched_get_n_splits(sched);
            placed[i] = ggml_backend_sched_get_tensor_backend(sched, l.x) == (i == 0 ? backend : cpu) &&
                        ggml_backend_sched_get_tensor_backend(sched, l.norm) == backend;
            out[i].resize(ggml_nbytes(l.x) + ggml_nbytes(l.y));
            ggml_backend_tensor_get(l.x, out[i].data(), 0, ggml_nbytes(l.x));
            ggml_backend_tensor_get(l.y, out[i].data() + ggml_nbytes(l.x), 0, ggml_nbytes(l.y));
        }
        if (ok && i == 0) {
            ggml_init_params params = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true };
            ggml_context *   ctx_r  = ggml_init(params);
            ggml_tensor *    ids_r  = ggml_new_tensor_1d(ctx_r, GGML_TYPE_I32, 7);
            ggml_tensor *    x_r    = ggml_get_rows(ctx_r, t.w, ids_r);
            ggml_backend_buffer_t br = ggml_backend_alloc_ctx_tensors_from_buft(ctx_r, ggml_backend_cpu_buffer_type());
            const std::vector<int32_t> next = token_ids(7, 99);
            ggml_backend_tensor_set(l.ids, next.data(), 0, ggml_nbytes(l.ids));
            ggml_backend_tensor_set(ids_r, next.data(), 0, ggml_nbytes(ids_r));
            ggml_cgraph * gr = ggml_new_graph(ctx_r);
            ggml_build_forward_expand(gr, x_r);
            spacemit_plan_counters_take();
            const bool again = ggml_backend_sched_graph_compute(sched, l.g) == GGML_STATUS_SUCCESS;
            hits             = spacemit_plan_counters_take().hits_uid;
            if (br != nullptr && again && ggml_backend_graph_compute(cpu, gr) == GGML_STATUS_SUCCESS) {
                std::vector<uint8_t> x_new(ggml_nbytes(l.x));
                ggml_backend_tensor_get(l.x, x_new.data(), 0, x_new.size());
                reused = std::memcmp(x_new.data(), x_r->data, x_new.size()) == 0;
            }
            ggml_backend_buffer_free(br);
            ggml_free(ctx_r);
        }
        ggml_backend_sched_free(sched);
    }
    const bool same     = out[0] == out[1];
    const bool sched_ok = ran && splits[0] == 1 && splits[1] == 2 && placed[0] && placed[1] && same && reused &&
                          hits == 1;
    if (!sched_ok) {
        std::fprintf(stderr,
                     "FAIL: GET_ROWS of Q6_K through the scheduler: %s; splits %d (GGML_FLAGOS_DISABLE_OPS=GET_ROWS "
                     "%d), placement as expected %d/%d, outputs %s, new ids on the kept graph %s, %llu hits by uid\n",
                     ran ? "ran" : "a graph failed", splits[0], splits[1], placed[0] ? 1 : 0, placed[1] ? 1 : 0,
                     same ? "identical" : "differ",
                     reused ? "as the CPU's" : "wrong", (unsigned long long) hits);
        g_failures++;
    }

    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    std::printf("rows     GET_ROWS of a quantized token embedding in a CPU buffer vs CPU (Q6_K of 256, 1024 and 2560 "
                "columns and of 2560 with edge blocks, Q4_0 and Q4_K of 1024, %lld rows; 1-512 ids, repeated and "
                "unordered): %d/%d identical (bound 0, memcmp), ggml's kernel by the counter %d/%d; claims exact (%d "
                "nodes: the eight block types, not other types, I64 indices, strided rows, a partial block or an IME "
                "layout; F32 rows still get_rows.f32); index out of range fails the launch and the next node is not "
                "run: %d/%d\n",
                (long long) k_vocab, n_same, n_cases, n_kernel, n_cases, n_claims, bad_ok, n_bad);
    std::printf("rows     the lookup and RMS_NORM through the scheduler: %d split against %d with "
                "GGML_FLAGOS_DISABLE_OPS=GET_ROWS, outputs %s; new ids on the kept graph %s, %llu hit by uid\n",
                splits[0], splits[1], same ? "identical" : "differ", reused ? "as the CPU's" : "wrong",
                (unsigned long long) hits);
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

//
// the recurrent state's writes (plan.md "Step 1 spec: Qwen3.5-4B loads on the default path"): SCALE by scale.f32 (the
// clear of a slot) and CPY of F32 into F32 by cpy.f32 (the copies into the state), bit for bit against the CPU
//

// what a state_op_run case gave: claimed by the provider; the state's and the op's bytes equal the reference's after
// the backend's run (8 tiles) and after the kernel entry called directly on 1 and on 8 tiles
struct state_op_result {
    bool claimed = false, same = false, same_1 = false, same_8 = false;
};

// one op on an F32 state [ne] named cache_r_l0 that holds init: on the provider with the state in its buffer (in a CPU
// buffer with host), and directly through kernel on a copy in a CPU buffer. build makes the op on a state from inputs
// the caller keeps in a CPU buffer; an op that is not a view gets its own tensor beside the state. The reference is
// the CPU backend (one thread) on a third copy, or expect (the state's bytes) where ggml-cpu cannot run the op
static state_op_result state_op_run(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu,
                                    const spacemit_plugin_kernel * kernel, bool host, const std::array<int64_t, 4> & ne,
                                    const std::vector<float> &                                          init,
                                    const std::function<ggml_tensor *(ggml_context *, ggml_tensor *)> & build,
                                    const std::vector<float> * expect = nullptr) {
    ggml_init_params      params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context *        ctx[3];  // the provider's run, the CPU backend's, the direct calls
    ggml_tensor *         state[3];
    ggml_tensor *         node[3];
    ggml_backend_buffer_t buf[3];
    for (int k = 0; k < 3; k++) {
        ctx[k]   = ggml_init(params);
        state[k] = ggml_new_tensor(ctx[k], GGML_TYPE_F32, 4, ne.data());
        ggml_set_name(state[k], "cache_r_l0");
        node[k] = build(ctx[k], state[k]);
        buf[k]  = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx[k], k == 0 && !host ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type());
    }
    const auto bytes = [](const ggml_tensor * t) {
        std::vector<uint8_t> b(ggml_nbytes(t));
        if (!b.empty()) {
            ggml_backend_tensor_get(t, b.data(), 0, b.size());
        }
        return b;
    };
    state_op_result r;
    if (buf[0] != nullptr && buf[1] != nullptr && buf[2] != nullptr &&
        (int64_t) init.size() == ggml_nelements(state[0])) {
        for (int k = 0; k < 3; k++) {
            ggml_backend_tensor_set(state[k], init.data(), 0, ggml_nbytes(state[k]));
        }
        r.claimed           = ggml_backend_dev_supports_op(dev, node[0]);
        ggml_cgraph * g_dev = ggml_new_graph(ctx[0]);
        ggml_cgraph * g_cpu = ggml_new_graph(ctx[1]);
        ggml_build_forward_expand(g_dev, node[0]);
        ggml_build_forward_expand(g_cpu, node[1]);
        std::vector<uint8_t> want_state, want_node;
        bool                 ref_ok = true;
        if (expect != nullptr) {
            want_state.resize(expect->size() * sizeof(float));
            std::memcpy(want_state.data(), expect->data(), want_state.size());
        } else {
            ref_ok     = ggml_backend_graph_compute(cpu, g_cpu) == GGML_STATUS_SUCCESS;
            want_state = bytes(state[1]);
            want_node  = bytes(node[1]);
        }
        const bool own_node = node[0]->view_src == nullptr;  // compared besides the state
        if (r.claimed && ref_ok && ggml_backend_graph_compute(backend, g_dev) == GGML_STATUS_SUCCESS) {
            r.same = bytes(state[0]) == want_state && (!own_node || expect != nullptr || bytes(node[0]) == want_node);
            for (uint32_t nth : { 1u, 8u }) {
                ggml_backend_tensor_set(state[2], init.data(), 0, ggml_nbytes(state[2]));
                if (own_node && ggml_nbytes(node[2]) > 0) {
                    const std::vector<uint8_t> junk(ggml_nbytes(node[2]), 0xA5);  // rows a tile skips stay wrong
                    ggml_backend_tensor_set(node[2], junk.data(), 0, junk.size());
                }
                bool ok = true;
                for (uint32_t ith = 0; ith < nth; ith++) {
                    const spacemit_tile tile = { ith, nth, nullptr, 0, nullptr, 0, nullptr };
                    ok &= kernel->fn(&tile, node[2]);
                }
                const bool same = ok && bytes(state[2]) == want_state &&
                                  (!own_node || expect != nullptr || bytes(node[2]) == want_node);
                (nth == 1 ? r.same_1 : r.same_8) = same;
            }
        }
    }
    for (int k = 0; k < 3; k++) {
        ggml_backend_buffer_free(buf[k]);
        ggml_free(ctx[k]);
    }
    return r;
}

static int check_state_ops(ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);

    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    const auto                     entry     = [&](const char * name) -> const spacemit_plugin_kernel * {
        for (uint32_t k = 0; k < n_kernels; k++) {
            if (kernels[k].name != nullptr && std::strcmp(kernels[k].name, name) == 0) {
                return &kernels[k];
            }
        }
        return nullptr;
    };
    const spacemit_plugin_kernel * scale = entry("scale.f32");
    const spacemit_plugin_kernel * copy  = entry("cpy.f32");
    REQUIRE(scale != nullptr && copy != nullptr);
    spacemit_ops ops;
    std::string  detail;
    REQUIRE(spacemit_ops_bind(table, &ops, &detail));
    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    // the sources: a convolution input [7, 100, 2] (its last 3 columns are the state's), states [300, 2], rows
    // [16, 16, 3, 2] (every other row taken), a new SSM state [65536], an op output [100 + 512 * 2 * 3] holding 3
    // snapshots of 2 states of 512 after 100 other values (the rollback snapshots, delta-net-base.cpp:591-603)
    ggml_init_params params = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context *   ctx_h  = ggml_init(params);
    REQUIRE(ctx_h != nullptr);
    ggml_tensor *         conv_in = ggml_new_tensor_3d(ctx_h, GGML_TYPE_F32, 7, 100, 2);
    ggml_tensor *         rows    = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F32, 300, 2);
    ggml_tensor *         rows4   = ggml_new_tensor_4d(ctx_h, GGML_TYPE_F32, 16, 16, 3, 2);
    ggml_tensor *         ssm     = ggml_new_tensor_1d(ctx_h, GGML_TYPE_F32, 65536);
    ggml_tensor *         gdn     = ggml_new_tensor_1d(ctx_h, GGML_TYPE_F32, 100 + 512 * 2 * 3);
    ggml_backend_buffer_t buf_h   = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
    REQUIRE(buf_h != nullptr);
    uint32_t seed = 500;
    for (ggml_tensor * t : { conv_in, rows, rows4, ssm, gdn }) {
        const std::vector<float> v = random_values(ggml_nelements(t), seed++, 1.0f);
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    }

    // a state's values: random, with infinities, NaNs and signed zeros in every row
    const auto state_init = [&](const std::array<int64_t, 4> & ne) {
        std::vector<float> v = random_values(ne[0] * ne[1] * ne[2] * ne[3], seed++, 2.0f);
        for (size_t i = 0; i + 4 < v.size(); i += (size_t) ne[0]) {
            v[i]     = INFINITY;
            v[i + 1] = -INFINITY;
            v[i + 2] = NAN;
            v[i + 3] = -0.0f;
        }
        return v;
    };

    int        n_cases = 0, n_same = 0;
    const auto run_case = [&](const char * what, const spacemit_plugin_kernel * kernel,
                              const std::array<int64_t, 4> & ne,
                              const std::function<ggml_tensor *(ggml_context *, ggml_tensor *)> & build,
                              const std::vector<float> * expect = nullptr) {
        const std::vector<float> init = state_init(ne);
        std::vector<float>       want;
        if (expect != nullptr) {  // expect: the multiplier of each element of init, NAN where it stays
            want = init;
            for (size_t i = 0; i < want.size(); i++) {
                want[i] = std::isnan((*expect)[i]) ? init[i] : init[i] * (*expect)[i];
            }
        }
        for (bool host : { false, true }) {
            const state_op_result r =
                state_op_run(dev, backend, cpu, kernel, host, ne, init, build, expect != nullptr ? &want : nullptr);
            n_cases++;
            const bool ok = r.claimed && r.same && r.same_1 && r.same_8;
            n_same += ok;
            if (!ok) {
                std::fprintf(stderr,
                             "FAIL: %s, the state in %s: %s; bytes as the reference: 8 tiles %s, direct 1 tile %s, "
                             "8 tiles %s\n",
                             what, host ? "a CPU buffer" : "our buffer", r.claimed ? "claimed" : "refused",
                             r.same ? "yes" : "no", r.same_1 ? "yes" : "no", r.same_8 ? "yes" : "no");
                g_failures++;
            }
        }
    };

    // SCALE: the clear (s 0 in place on one row of 1000, a 1-D view, over infinities and NaNs), the empty view of a step
    // that clears none, s 0.37 in place on rows 1-3, s -1.5 into a new tensor, s 0 over 33 rows of 7 (tiles of 5 and 3
    // rows), and s 2 in place on rows of 500 of 1000 (strided; ggml-cpu takes contiguous tensors only, so the host's
    // product is the reference)
    const std::array<int64_t, 4> s_ne = { 1000, 4, 1, 1 };
    run_case("SCALE 0 in place on row 2 of a state [1000, 4]", scale, s_ne, [](ggml_context * ctx, ggml_tensor * st) {
        return ggml_scale_inplace(ctx, ggml_view_1d(ctx, st, 1000, 2 * st->nb[1]), 0.0f);
    });
    run_case("SCALE 0 in place on an empty view", scale, s_ne, [](ggml_context * ctx, ggml_tensor * st) {
        return ggml_scale_inplace(ctx, ggml_view_1d(ctx, st, 0, 0), 0.0f);
    });
    run_case("SCALE 0.37 in place on rows 1-3", scale, s_ne, [](ggml_context * ctx, ggml_tensor * st) {
        return ggml_scale_inplace(ctx, ggml_view_2d(ctx, st, 1000, 3, st->nb[1], st->nb[1]), 0.37f);
    });
    run_case("SCALE -1.5 into a new tensor", scale, s_ne,
             [](ggml_context * ctx, ggml_tensor * st) { return ggml_scale(ctx, st, -1.5f); });
    run_case("SCALE 0 in place on [7, 33]", scale, { 7, 33, 1, 1 },
             [](ggml_context * ctx, ggml_tensor * st) { return ggml_scale_inplace(ctx, st, 0.0f); });
    {
        std::vector<float> mult(1000 * 4, NAN);
        for (int64_t r = 0; r < 4; r++) {
            for (int64_t c = 0; c < 500; c++) {
                mult[r * 1000 + c] = 2.0f;
            }
        }
        run_case("SCALE 2 in place on strided rows [500, 4] of [1000, 4]", scale, s_ne,
                 [](ggml_context * ctx, ggml_tensor * st) {
                     return ggml_scale_inplace(ctx, ggml_view_2d(ctx, st, 500, 4, st->nb[1], 0), 2.0f);
                 },
                 &mult);
    }

    // CPY: the convolution state's last columns (a 3-D view [3, 100, 2] with rows of 3 of 7) into rows 1-2 of a state
    // [300, 4]; the extra states [300, 2] into rows 2-3; an empty copy (no extra state); every other row of
    // [16, 16, 3, 2] into row 1 of a state [768, 2] (a 1-D view); the new SSM state [65536] into row 1 of 3
    run_case("CPY of conv [3, 100, 2] (rows of 3 of 7) into rows 1-2 of [300, 4]", copy, { 300, 4, 1, 1 },
             [&](ggml_context * ctx, ggml_tensor * st) {
                 ggml_tensor * src =
                     ggml_view_3d(ctx, conv_in, 3, 100, 2, conv_in->nb[1], conv_in->nb[2], 4 * sizeof(float));
                 return ggml_cpy(ctx, src, ggml_view_2d(ctx, st, 300, 2, st->nb[1], st->nb[1]));
             });
    run_case("CPY of [300, 2] into rows 2-3 of [300, 4]", copy, { 300, 4, 1, 1 },
             [&](ggml_context * ctx, ggml_tensor * st) {
                 return ggml_cpy(ctx, rows, ggml_view_2d(ctx, st, 300, 2, st->nb[1], 2 * st->nb[1]));
             });
    run_case("CPY of [300, 0] into an empty view", copy, { 300, 4, 1, 1 }, [&](ggml_context * ctx, ggml_tensor * st) {
        return ggml_cpy(ctx, ggml_view_2d(ctx, rows, 300, 0, rows->nb[1], 0),
                        ggml_view_2d(ctx, st, 300, 0, st->nb[1], 3 * st->nb[1]));
    });
    run_case("CPY of every other row of [16, 16, 3, 2] into row 1 of [768, 2]", copy, { 768, 2, 1, 1 },
             [&](ggml_context * ctx, ggml_tensor * st) {
                 ggml_tensor * src = ggml_view_4d(ctx, rows4, 16, 8, 3, 2, 2 * rows4->nb[1], rows4->nb[2],
                                                  rows4->nb[3], 0);
                 return ggml_cpy(ctx, src, ggml_view_1d(ctx, st, 768, st->nb[1]));
             });
    run_case("CPY of [65536] into row 1 of [65536, 3]", copy, { 65536, 3, 1, 1 },
             [&](ggml_context * ctx, ggml_tensor * st) {
                 return ggml_cpy(ctx, ssm, ggml_view_1d(ctx, st, 65536, st->nb[1]));
             });
    // the same shape into a non-contiguous destination: the rollback snapshots [512, 2, 3] into a state [512, 12] of 3
    // slots of 4 cells at cell 1 (nb2 4 rows), and every other row of [16, 16, 3, 2] into half rows of a state
    // [32, 48] at column 4 (nb1 1 row, nb2 8, nb3 24)
    run_case("CPY of snapshots [512, 2, 3] into slots of 4 cells of [512, 12]", copy, { 512, 12, 1, 1 },
             [&](ggml_context * ctx, ggml_tensor * st) {
                 ggml_tensor * src = ggml_view_3d(ctx, gdn, 512, 2, 3, 512 * sizeof(float), 1024 * sizeof(float),
                                                  100 * sizeof(float));
                 return ggml_cpy(ctx, src, ggml_view_3d(ctx, st, 512, 2, 3, st->nb[1], 4 * st->nb[1], st->nb[1]));
             });
    run_case("CPY of every other row of [16, 16, 3, 2] into half rows of [32, 48]", copy, { 32, 48, 1, 1 },
             [&](ggml_context * ctx, ggml_tensor * st) {
                 ggml_tensor * src = ggml_view_4d(ctx, rows4, 16, 8, 3, 2, 2 * rows4->nb[1], rows4->nb[2],
                                                  rows4->nb[3], 0);
                 return ggml_cpy(ctx, src, ggml_view_4d(ctx, st, 16, 8, 3, 2, st->nb[1], 8 * st->nb[1],
                                                        24 * st->nb[1], 4 * sizeof(float)));
             });

    // the contracts each claim takes, and what stays refused: a scale with a bias, F16 scales; CPY of F16 into F16 or
    // F32, of F32 into F16, into a non-contiguous F32 view of another shape or with strided elements, from a transposed
    // source; the same shape into a non-contiguous view is taken
    {
        ggml_init_params p   = { ggml_tensor_overhead() * 32, nullptr, true };
        ggml_context *   ctx = ggml_init(p);
        REQUIRE(ctx != nullptr);
        ggml_tensor * st   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 300, 4);
        ggml_tensor * f32  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 300, 2);
        ggml_tensor * f16  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 300, 2);
        ggml_tensor * f16d = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 300, 2);
        ggml_tensor * sq   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_set_name(st, "cache_r_l0");
        const auto taken = [&](ggml_tensor * node, const spacemit_plugin_kernel * want) {
            const spacemit_op * got = spacemit_find_op(ops, node, buft);
            return got != nullptr && got->n_steps == 1 && got->steps[0] == want;
        };
        CHECK(taken(ggml_scale_inplace(ctx, ggml_view_1d(ctx, st, 300, st->nb[1]), 0.0f), scale));
        CHECK(taken(ggml_cpy(ctx, f32, ggml_view_2d(ctx, st, 300, 2, st->nb[1], 0)), copy));
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_scale_bias(ctx, f32, 0.5f, 0.25f)));
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_scale_bias(ctx, f32, 0.0f, 1.0f)));
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_scale(ctx, f16, 0.0f)));
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_cpy(ctx, f16, f16d)));
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_cpy(ctx, f16, f32)));
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_cpy(ctx, f32, f16d)));
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_cpy(ctx, f32, ggml_view_2d(ctx, st, 150, 4, st->nb[1], 0))));
        CHECK(taken(ggml_cpy(ctx, f32, ggml_view_2d(ctx, st, 300, 2, 2 * st->nb[1], 0)), copy));
        ggml_tensor * sq_dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_cpy(ctx, ggml_transpose(ctx, sq), sq_dst)));
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_cpy(ctx, sq, ggml_transpose(ctx, sq_dst))));  // nb0 a row
        ggml_free(ctx);
    }

    ggml_backend_buffer_free(buf_h);
    ggml_free(ctx_h);
    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    std::printf("state    the recurrent state's writes, F32 in our buffer and in a CPU buffer: SCALE (0 over infinities "
                "and NaNs, in place on views, empty, into a new tensor, strided rows) and CPY of F32 (3-D strided into "
                "2-D, contiguous, empty, 4-D every other row, 65536 floats; the same shape into rollback snapshot slots "
                "and 4-D half rows) %d/%d bit for bit as the CPU on 8 tiles and directly on 1 and 8; a scale with a "
                "bias, F16 scales and copies, a non-contiguous destination of another shape or with strided "
                "elements, a transposed source refused: checked\n",
                n_same, n_cases);
    return 0;
}

//
// the Gated DeltaNet step (plan.md "Step 2 spec", piece A): GATED_DELTA_NET by gdn.f32, Qwen3.5's recurrent layers
//
// What is compared against ggml-cpu (one thread) within a bound: every output value, as the NMSE of the whole output
// and the largest absolute difference. Bit equality with ggml-cpu is not attainable: its kernel adds its dot products
// with ggml_vec_dot_f32 and its outer product with ggml_vec_mad_f32, which are SIMD there (several float accumulators
// and fused multiply-adds, their number the host ISA's), while the provider's reference adds them one by one.
// What is compared bit for bit: the provider's two runs of the same node, and the kernel entry called directly on 1
// and on 8 tiles (a row, one (head, sequence) pair, is computed by one tile alone, so the tile count must not change
// a byte), each in the provider's buffer and in a CPU buffer.
constexpr double k_gdn_max_nmse = 1e-6;

// a GATED_DELTA_NET node's operands and shape: q and k may have fewer heads (H_qk) and sequences (seqs_qk) than v,
// and repeat over them; the gate is scalar or one value per channel (kda); K is the snapshot slot count
struct gdn_case {
    const char * what;
    int64_t      S_v, H, n_tokens, n_seqs;
    int64_t      H_qk, seqs_qk;
    bool         kda;
    int64_t      K;
};

// one copy of a case: the six operands and the node in one buffer
struct gdn_run {
    ggml_context *        ctx   = nullptr;
    ggml_backend_buffer_t buf   = nullptr;
    ggml_tensor *         node  = nullptr;
    ggml_tensor *         in[6] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };

    gdn_run() = default;

    ~gdn_run() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    gdn_run(const gdn_run &)             = delete;
    gdn_run & operator=(const gdn_run &) = delete;

    bool init(const gdn_case & c, ggml_backend_buffer_type_t buft) {
        ggml_init_params params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        ctx                     = ggml_init(params);
        if (ctx == nullptr) {
            return false;
        }
        in[0] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H_qk, c.n_tokens, c.seqs_qk);  // q
        in[1] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H_qk, c.n_tokens, c.seqs_qk);  // k
        in[2] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H, c.n_tokens, c.n_seqs);      // v
        in[3] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.kda ? c.S_v : 1, c.H, c.n_tokens, c.n_seqs);  // g
        in[4] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, c.H, c.n_tokens, c.n_seqs);                  // beta
        in[5] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.S_v, c.H, c.n_seqs);                    // the state
        ggml_set_name(in[5], "cache_s_l0");  // llama.cpp's name for it, as the provider's buffer holds it
        node = ggml_gated_delta_net(ctx, in[0], in[1], in[2], in[3], in[4], in[5], c.K);
        buf  = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        return buf != nullptr;
    }

    // the same values from the same seed in every copy: q, k and v in [-1, 1), the gate <= 0 (the model's -softplus),
    // beta in [0, 1) (its sigmoid), the state in [-0.5, 0.5)
    void fill(uint32_t seed) {
        for (int i = 0; i < 6; i++) {
            std::vector<float> v = random_values(ggml_nelements(in[i]), seed + (uint32_t) i, 1.0f);
            for (float & x : v) {
                x = i == 3 ? -std::fabs(x) : i == 4 ? 0.5f * (x + 1.0f) : i == 5 ? 0.5f * x : x;
            }
            ggml_backend_tensor_set(in[i], v.data(), 0, ggml_nbytes(in[i]));
        }
    }

    // the model's inputs: q and k L2-normalized per head and token (Qwen3.5's graph normalizes both before the op), v
    // in [-1, 1), the gate (a log decay) in (-2, 0], beta in [0.01, 0.99), the state in [-0.5, 0.5)
    void fill_model(uint32_t seed) {
        for (int i = 0; i < 6; i++) {
            std::vector<float> v = random_values(ggml_nelements(in[i]), seed + (uint32_t) i, 1.0f);
            if (i <= 1) {
                const size_t d = (size_t) in[i]->ne[0];
                for (size_t r = 0; r < v.size(); r += d) {
                    double sum = 0.0;
                    for (size_t j = 0; j < d; j++) {
                        sum += (double) v[r + j] * v[r + j];
                    }
                    const float inv = sum > 0.0 ? (float) (1.0 / std::sqrt(sum)) : 1.0f;
                    for (size_t j = 0; j < d; j++) {
                        v[r + j] *= inv;
                    }
                }
            } else {
                for (float & x : v) {
                    x = i == 3 ? -(x + 1.0f) : i == 4 ? 0.01f + 0.49f * (x + 1.0f) : i == 5 ? 0.5f * x : x;
                }
            }
            ggml_backend_tensor_set(in[i], v.data(), 0, ggml_nbytes(in[i]));
        }
    }
};

// a GATED_DELTA_NET node built by hand, for what ggml_gated_delta_net's asserts forbid the test to build (an F16
// operand, a gate of neither 1 nor S_v, a state with gaps, an output of the wrong length)
static ggml_tensor * gdn_raw(ggml_context * ctx, ggml_type dst_type, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                             ggml_tensor * g, ggml_tensor * b, ggml_tensor * s, int64_t K, int64_t ne1 = 0) {
    const int64_t S_v   = v->ne[0], H = v->ne[1], n_tokens = v->ne[2], n_seqs = v->ne[3];
    const int64_t ne[4] = { S_v * H, ne1 > 0 ? ne1 : n_tokens * n_seqs + K * S_v * n_seqs, 1, 1 };
    ggml_tensor * r     = ggml_new_tensor(ctx, dst_type, 4, ne);
    ggml_set_op_params_i32(r, 0, (int32_t) K);
    r->op     = GGML_OP_GATED_DELTA_NET;
    r->src[0] = q;
    r->src[1] = k;
    r->src[2] = v;
    r->src[3] = g;
    r->src[4] = b;
    r->src[5] = s;
    return r;
}

//
// Qwen3.5's Gated DeltaNet layers, the ops beside the state update (plan.md "Step 2 spec", pieces D to G): SILU,
// SIGMOID, SOFTPLUS, L2_NORM, SSM_CONV, CONCAT, CONT and MUL_MAT on the F32 ssm_alpha and ssm_beta weights, against
// ggml-cpu on one thread, at the model's shapes (one token and a 512-token batch) and at edge cases.
//
// What is compared how, and why:
// - SSM_CONV, CONCAT and CONT: bit for bit. ggml-cpu computes these scalar in a fixed order (ops.cpp
//   ggml_compute_forward_ssm_conv_f32 and _concat_f32), and the kernels are those loops with the rows split over the
//   tiles, or on the K3 their RVV ports: CONCAT's moves the same floats. SSM_CONV's RVV port adds the four taps with
//   fused multiply-adds, which is not how GCC built ggml-cpu's loop on the K3, so with the RVV kernels it is held to a
//   rounding bound instead (k_ssm_conv in check_seq_ops). Each output element comes from one row, so the split cannot
//   change a result. The convolution pair's stashed SSM_CONV (check_conv_pair) is held to that bound against the CPU
//   with the reference kernels too, as a compiler may contract its multiply and add (seen with Apple clang 21 on
//   2026-11-18, NMSE 3e-15); its copies and its own paths stay bit for bit.
// - SILU, SIGMOID, SOFTPLUS and L2_NORM: within an NMSE and a largest |difference| / max(1, |CPU's value|) (the
//   absolute error up to 1, the relative one above). The references are bit for bit ggml-cpu's scalar code for
//   SIGMOID, SOFTPLUS and L2_NORM, but on the K3 the RVV kernels use SpacemiT's vector exp (ARM's optimized routine,
//   1.954 ulp at worst over all 2^32 floats by the lane model below), the fork's log1p for SOFTPLUS (no cut-off at
//   20), and lane sums for L2_NORM. The bounds are the worst differences from ggml-cpu's scalar formulas the lane
//   model gave over all floats on the Mac (sigmoid 1.2e-7, silu 2.3e-7, softplus 2.4e-7 of max(1, |y|)) with a margin
//   of about 4 for another libm, and two ulps for L2_NORM's scale; NMSE 1e-12 is a few ulps squared. SILU's ggml-cpu uses a vector exp approximation where the host has one
//   (on the K3 the RVV kernel's own), so its references differ from it too.
// - MUL_MAT: within test-backend-ops' bound for MUL_MAT (5e-4), not bit for bit: ggml-cpu adds a dot product in
//   vector lanes (ggml_vec_dot_f32) or runs llamafile's sgemm, which a plain row dot cannot match; nor does the RVV
//   kernel's lanes on the K3, whose order depends on the vector length too.

// an NMSE and a largest |difference| / max(1, |CPU's value|); { 0, 0 } asks for the CPU's bytes
struct seq_bound {
    double nmse, abs;
};

constexpr seq_bound k_seq_exact    = { 0.0, 0.0 };
constexpr seq_bound k_seq_silu     = { 1e-12, 1e-6 };
constexpr seq_bound k_seq_sigmoid  = { 1e-12, 5e-7 };
constexpr seq_bound k_seq_softplus = { 1e-12, 1e-6 };
constexpr seq_bound k_seq_l2_norm  = { 1e-12, 5e-7 };
constexpr seq_bound k_seq_mul_mat  = { k_mul_mat_max_nmse, HUGE_VAL };

// what one seq_op_run case gave: claimed by the provider, the op ran, its bytes equal the CPU's after the backend's
// run (8 tiles) and after the kernel entry called directly on 1 and on 8 tiles, and the worst NMSE and the worst
// |difference| / max(1, |CPU's value|) of those three (HUGE_VAL where one is non-finite and the other not the same)
struct seq_op_result {
    bool   claimed = false, ran = false, exact = false;
    double err = -1.0, abs = -1.0;
};

// builds the node under test in ctx and pushes the tensors to fill (bases, never views) into inputs
using seq_build = std::function<ggml_tensor *(ggml_context *, std::vector<ggml_tensor *> &)>;

// one op built three times: on the provider with every tensor in its buffer (in a CPU buffer with host), on the CPU
// backend (the reference, one thread), and a third copy for the kernel entry called directly on 1 and 8 tiles. The
// inputs get the same values in all three: random in [-scale, scale], from seed on
static seq_op_result seq_op_run(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu,
                                const spacemit_plugin_kernel * kernel, bool host, const seq_build & build,
                                uint32_t seed, float scale) {
    ggml_init_params           params = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    ggml_context *             ctx[3];
    ggml_tensor *              node[3];
    std::vector<ggml_tensor *> inputs[3];
    ggml_backend_buffer_t      buf[3];
    for (int k = 0; k < 3; k++) {
        ctx[k]  = ggml_init(params);
        node[k] = build(ctx[k], inputs[k]);
        buf[k]  = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx[k], k == 0 && !host ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type());
    }
    const auto bytes = [](const ggml_tensor * t) {
        std::vector<uint8_t> b(ggml_nbytes(t));
        if (!b.empty()) {
            ggml_backend_tensor_get(t, b.data(), 0, b.size());
        }
        return b;
    };
    seq_op_result r;
    if (buf[0] != nullptr && buf[1] != nullptr && buf[2] != nullptr) {
        for (int k = 0; k < 3; k++) {
            uint32_t s = seed;
            for (ggml_tensor * t : inputs[k]) {
                const std::vector<float> v = random_values(ggml_nelements(t), s++, scale);
                if (!v.empty()) {
                    ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
                }
            }
        }
        r.claimed           = ggml_backend_dev_supports_op(dev, node[0]);
        ggml_cgraph * g_dev = ggml_new_graph(ctx[0]);
        ggml_cgraph * g_cpu = ggml_new_graph(ctx[1]);
        ggml_build_forward_expand(g_dev, node[0]);
        ggml_build_forward_expand(g_cpu, node[1]);
        if (r.claimed && ggml_backend_graph_compute(cpu, g_cpu) == GGML_STATUS_SUCCESS &&
            ggml_backend_graph_compute(backend, g_dev) == GGML_STATUS_SUCCESS) {
            const std::vector<uint8_t> want = bytes(node[1]);
            const int64_t              n    = ggml_nelements(node[1]);
            const auto                 compare = [&](const std::vector<uint8_t> & got) {
                r.exact = r.exact && got == want;
                if (n > 0) {
                    const float * g = (const float *) got.data();
                    const float * w = (const float *) want.data();
                    r.err           = std::max(r.err, nmse(g, w, n));
                    for (int64_t i = 0; i < n; i++) {
                        double d = HUGE_VAL;
                        if (g[i] == w[i]) {
                            d = 0.0;
                        } else if (std::isfinite(g[i]) && std::isfinite(w[i])) {
                            d = std::fabs((double) g[i] - w[i]) / std::max(1.0, std::fabs((double) w[i]));
                        }
                        r.abs = std::max(r.abs, d);
                    }
                } else {
                    r.err = std::max(r.err, 0.0);
                    r.abs = std::max(r.abs, 0.0);
                }
            };
            r.ran   = true;
            r.exact = true;
            compare(bytes(node[0]));
            for (uint32_t nth : { 1u, 8u }) {
                if (ggml_nbytes(node[2]) > 0) {
                    const std::vector<uint8_t> junk(ggml_nbytes(node[2]), 0xA5);  // what a tile skips stays wrong
                    ggml_backend_tensor_set(node[2], junk.data(), 0, junk.size());
                }
                bool ok = true;
                for (uint32_t ith = 0; ith < nth; ith++) {
                    const spacemit_tile tile = { ith, nth, nullptr, 0, nullptr, 0, nullptr };
                    ok &= kernel->fn(&tile, node[2]);
                }
                r.ran = r.ran && ok;
                compare(bytes(node[2]));
            }
        }
    }
    for (int k = 0; k < 3; k++) {
        ggml_backend_buffer_free(buf[k]);
        ggml_free(ctx[k]);
    }
    return r;
}

//
// The MoE router of Qwen3.6-35B-A3B (build_moe_ffn, src/llama-graph.cpp:1915): SOFT_MAX, ARGSORT, SUM_ROWS, CLAMP and
// DIV against ggml-cpu on one thread, at the model's shapes (256 experts, 8 used, 2048 embedding channels; one token
// and a 512-token batch) and at edge cases, with the router's other ops on the contracts that already take them
// (GET_ROWS by the top-k view, MUL by the weights, ADD of the experts' views).
//
// What is compared how, and why:
// - ARGSORT, CLAMP, DIV and the router's GET_ROWS, MUL and ADD: bit for bit. ggml-cpu sorts with std::sort and the
//   same comparisons (so ties come out in its order), clamps with its MIN and MAX macros, and divides once per element
//   (IEEE, so also on the RVV path); the others copy or apply one IEEE operation.
// - SUM_ROWS: bit for bit against the program's own double sum (vec.h ggml_vec_sum_f32 without Accelerate) where the
//   case has one and the reference kernel runs, and against ggml-cpu bit for bit where ggml-cpu sums in double (not on
//   Apple hosts, whose ggml-cpu uses vDSP_sve) with the reference kernel; else (the RVV kernel's float lanes, or
//   Accelerate) within k_moe_sum_max_nmse.
// - SOFT_MAX: within test-backend-ops' bound (1e-7): ggml-cpu's ggml_vec_soft_max_f32 uses a vector exp on most hosts.

constexpr double k_moe_soft_max_nmse = 1e-7;   // test-backend-ops' default bound
constexpr double k_moe_sum_max_nmse  = 1e-10;  // float sums against double ones, rows of up to a few thousand

// what one moe_op_run case gave: claimed, ran, its bytes as ggml-cpu's after the backend's run (8 tiles) and after the
// kernel called directly on 1 and 8 tiles, its F32 bytes as the program's own expectation where it has one, the worst
// NMSE against ggml-cpu (0 for equal bytes)
struct moe_op_result {
    bool   claimed = false, ran = false, exact = false, host_exact = true;
    double err     = -1.0;
};

// the values of input i (in the order the build pushed it) from seed; an I32 input takes them truncated
using moe_values = std::function<std::vector<float>(int64_t n, size_t i, uint32_t seed)>;
// the node's F32 output as the program computes it from the inputs' values (empty function: no such check)
using moe_expect = std::function<std::vector<float>(const std::vector<std::vector<float>> & inputs)>;

// seq_op_run with chosen input values (ties, infinities, NaNs) and I32 inputs, nodes in place on an input (their
// inputs are set again before every run), and an optional expectation of the program's own
static moe_op_result moe_op_run(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_t cpu,
                                const spacemit_plugin_kernel * kernel, bool host, const seq_build & build,
                                const moe_values & values, const moe_expect & expect, uint32_t seed) {
    ggml_init_params           params = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    ggml_context *             ctx[3];
    ggml_tensor *              node[3];
    std::vector<ggml_tensor *> inputs[3];
    ggml_backend_buffer_t      buf[3];
    for (int k = 0; k < 3; k++) {
        ctx[k]  = ggml_init(params);
        node[k] = build(ctx[k], inputs[k]);
        buf[k]  = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx[k], k == 0 && !host ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type());
    }
    const auto bytes = [](const ggml_tensor * t) {
        std::vector<uint8_t> b(ggml_nbytes(t));
        if (!b.empty()) {
            ggml_backend_tensor_get(t, b.data(), 0, b.size());
        }
        return b;
    };
    std::vector<std::vector<float>> vals;
    const auto                      set_inputs = [&](int k) {
        for (size_t i = 0; i < inputs[k].size(); i++) {
            ggml_tensor * t = inputs[k][i];
            if (ggml_nelements(t) == 0) {
                continue;
            }
            if (t->type == GGML_TYPE_I32) {
                std::vector<int32_t> iv(vals[i].size());
                for (size_t j = 0; j < iv.size(); j++) {
                    iv[j] = (int32_t) vals[i][j];
                }
                ggml_backend_tensor_set(t, iv.data(), 0, ggml_nbytes(t));
            } else {
                ggml_backend_tensor_set(t, vals[i].data(), 0, ggml_nbytes(t));
            }
        }
    };
    moe_op_result r;
    if (buf[0] != nullptr && buf[1] != nullptr && buf[2] != nullptr) {
        for (size_t i = 0; i < inputs[0].size(); i++) {
            vals.push_back(values(ggml_nelements(inputs[0][i]), i, seed + (uint32_t) i));
        }
        for (int k = 0; k < 3; k++) {
            set_inputs(k);
        }
        r.claimed           = ggml_backend_dev_supports_op(dev, node[0]);
        ggml_cgraph * g_dev = ggml_new_graph(ctx[0]);
        ggml_cgraph * g_cpu = ggml_new_graph(ctx[1]);
        ggml_build_forward_expand(g_dev, node[0]);
        ggml_build_forward_expand(g_cpu, node[1]);
        if (r.claimed && ggml_backend_graph_compute(cpu, g_cpu) == GGML_STATUS_SUCCESS &&
            ggml_backend_graph_compute(backend, g_dev) == GGML_STATUS_SUCCESS) {
            const std::vector<uint8_t> want = bytes(node[1]);
            const int64_t              n    = ggml_nelements(node[1]);
            const bool                 f32  = node[1]->type == GGML_TYPE_F32;
            const std::vector<float>   mine = expect ? expect(vals) : std::vector<float>();
            const auto                 compare = [&](const std::vector<uint8_t> & got) {
                const bool same = got == want;
                r.exact         = r.exact && same;
                double e        = same ? 0.0 : HUGE_VAL;
                if (!same && f32 && n > 0) {
                    e = nmse((const float *) got.data(), (const float *) want.data(), n);
                }
                r.err = std::max(r.err, e);
                if (expect) {
                    r.host_exact = r.host_exact && got.size() == mine.size() * sizeof(float) &&
                                   (mine.empty() || std::memcmp(got.data(), mine.data(), got.size()) == 0);
                }
            };
            r.ran   = true;
            r.exact = true;
            compare(bytes(node[0]));
            for (uint32_t nth : { 1u, 8u }) {
                if (ggml_nbytes(node[2]) > 0) {
                    const std::vector<uint8_t> junk(ggml_nbytes(node[2]), 0xA5);  // what a tile skips stays wrong
                    ggml_backend_tensor_set(node[2], junk.data(), 0, junk.size());
                }
                set_inputs(2);  // a node in place on an input reads that input again
                bool ok = true;
                for (uint32_t ith = 0; ith < nth; ith++) {
                    const spacemit_tile tile = { ith, nth, nullptr, 0, nullptr, 0, nullptr };
                    ok &= kernel->fn(&tile, node[2]);
                }
                r.ran = r.ran && ok;
                compare(bytes(node[2]));
            }
        }
    }
    for (int k = 0; k < 3; k++) {
        ggml_backend_buffer_free(buf[k]);
        ggml_free(ctx[k]);
    }
    return r;
}

static int check_moe_ops(ggml_backend_dev_t dev) {
    env_override router_f32("FLAGOS_SPACEMIT_ROUTER", "f32");  // the F32 matmul path: its weights carry the router's name
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);  // a correctness reference: single-threaded (PROJECT.md section 6, item 28)

    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    const auto                     entry     = [&](const char * name) -> const spacemit_plugin_kernel * {
        for (uint32_t k = 0; k < n_kernels; k++) {
            if (kernels[k].name != nullptr && std::strcmp(kernels[k].name, name) == 0) {
                return &kernels[k];
            }
        }
        return nullptr;
    };
    const spacemit_plugin_kernel * k_soft_max = entry("soft_max.f32");
    const spacemit_plugin_kernel * k_argsort  = entry("argsort.f32");
    const spacemit_plugin_kernel * k_sum_rows = entry("sum_rows.f32");
    const spacemit_plugin_kernel * k_clamp    = entry("clamp.f32");
    const spacemit_plugin_kernel * k_div      = entry("div.f32");
    const spacemit_plugin_kernel * k_get_rows = entry("get_rows.f32");
    const spacemit_plugin_kernel * k_mul      = entry("mul.f32");
    const spacemit_plugin_kernel * k_add      = entry("add.f32");
    REQUIRE(k_soft_max != nullptr && k_argsort != nullptr && k_sum_rows != nullptr && k_clamp != nullptr &&
            k_div != nullptr && k_get_rows != nullptr && k_mul != nullptr && k_add != nullptr);
    spacemit_ops ops;
    std::string  detail;
    REQUIRE(spacemit_ops_bind(table, &ops, &detail));
    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    // the reference kernels run here: always off the K3, with GGML_FLAGOS_KERNEL_VARIANTS=0 on it
#if defined(__riscv)
    const bool reference = std::strlen(reference_label()) > 0;
#else
    const bool reference = true;
#endif
#if defined(__APPLE__)
    const bool cpu_sums_in_double = false;  // ggml-cpu's ggml_vec_sum_f32 is vDSP_sve with Accelerate
#else
    const bool cpu_sums_in_double = true;
#endif

    const moe_values random_in = [](float scale) {
        return [scale](int64_t n, size_t, uint32_t s) { return random_values(n, s, scale); };
    }(1.0f);
    const auto random_scaled = [](float scale) -> moe_values {
        return [scale](int64_t n, size_t, uint32_t s) { return random_values(n, s, scale); };
    };
    // input 0 F32 in [-1, 1); input 1 indices in [0, n_idx)
    const auto values_and_ids = [](int64_t n_idx) -> moe_values {
        return [n_idx](int64_t n, size_t i, uint32_t s) {
            std::vector<float> v = random_values(n, s, 1.0f);
            if (i == 1) {
                for (float & x : v) {
                    x = (float) std::min<int64_t>(n_idx - 1, (int64_t) ((x + 1.0f) * 0.5f * (float) n_idx));
                }
            }
            return v;
        };
    };
    // the row sums of a contiguous [ne0, rows] input in double, as ggml_vec_sum_f32 without Accelerate
    const auto row_sums = [](int64_t ne0) -> moe_expect {
        return [ne0](const std::vector<std::vector<float>> & in) {
            const int64_t      rows = ne0 > 0 ? (int64_t) in[0].size() / ne0 : 0;
            std::vector<float> out((size_t) rows);
            for (int64_t r = 0; r < rows; r++) {
                double sum = 0.0;
                for (int64_t i = 0; i < ne0; i++) {
                    sum += (double) in[0][(size_t) (r * ne0 + i)];
                }
                out[(size_t) r] = (float) sum;
            }
            return out;
        };
    };

    uint32_t   seed = 1300;
    int        n_cases = 0, n_ok = 0, n_exact = 0, n_exact_cases = 0, n_mine = 0;
    double     worst_soft_max = -1.0, worst_bound = -1.0;
    // bound 0: bit for bit as ggml-cpu; sum: SUM_ROWS' rule (above); else within bound
    const auto run_case = [&](const char * what, const spacemit_plugin_kernel * kernel, double bound,
                              const seq_build & build, const moe_values & values, const moe_expect & expect = nullptr,
                              bool sum = false) {
        for (bool host : { false, true }) {
            const moe_op_result r = moe_op_run(dev, backend, cpu, kernel, host, build, values, expect, seed);
            n_cases++;
            const bool exact_wanted = sum ? reference && cpu_sums_in_double : bound == 0.0;
            const bool mine_wanted  = expect && reference;
            const double limit      = sum && !exact_wanted ? k_moe_sum_max_nmse : bound;
            const bool ok = r.claimed && r.ran && r.err >= 0.0 && r.err <= limit && (!exact_wanted || r.exact) &&
                            (!mine_wanted || r.host_exact);
            n_ok += ok;
            n_mine += mine_wanted && r.host_exact ? 1 : 0;
            if (exact_wanted) {
                n_exact_cases++;
                n_exact += r.exact ? 1 : 0;
            } else if (kernel == k_soft_max) {
                worst_soft_max = std::max(worst_soft_max, r.err);
            } else {
                worst_bound = std::max(worst_bound, r.err);
            }
            if (!ok) {
                std::fprintf(stderr,
                             "FAIL: %s, the tensors in %s: %s, %s; NMSE %.3e (bound %.0e), bytes as the CPU's: %s, as "
                             "the program's own: %s\n",
                             what, host ? "a CPU buffer" : "our buffer", r.claimed ? "claimed" : "refused",
                             r.ran ? "ran" : "failed", r.err, limit, r.exact ? "yes" : "no",
                             !expect ? "-" : r.host_exact ? "yes" : "no");
                g_failures++;
            }
        }
        seed += 8;
    };

    // Qwen3.6-35B-A3B's router for one token and a 512-token batch: the logits [256, n_t] of the F32 ffn_gate_inp
    for (int64_t n_t : { (int64_t) 1, (int64_t) 512 }) {
        run_case("SOFT_MAX of the router logits [256, n_t]", k_soft_max, k_moe_soft_max_nmse,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, n_t);
                     in.push_back(x);
                     return ggml_soft_max(ctx, x);
                 },
                 random_scaled(4.0f));
        // the probabilities' top 8: ARGSORT descending (ggml_argsort_top_k's view of its first 8 columns is no op)
        run_case("ARGSORT descending of the probabilities [256, n_t]", k_argsort, 0.0,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * p = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, n_t);
                     in.push_back(p);
                     return ggml_argsort(ctx, p, GGML_SORT_ORDER_DESC);
                 },
                 [](int64_t n, size_t, uint32_t s) {
                     std::vector<float> v = random_values(n, s, 1.0f);
                     for (float & x : v) {
                         x = std::fabs(x) / 256.0f;
                     }
                     return v;
                 });
        // the chosen weights: GET_ROWS of the probabilities as [1, 256, n_t] by the top-k view [8, n_t] of the sort
        run_case("GET_ROWS of [1, 256, n_t] by the view [8, n_t] of an I32 [256, n_t]", k_get_rows, 0.0,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * p   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, n_t);
                     ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 256, n_t);
                     in.push_back(p);
                     in.push_back(ids);
                     ggml_tensor * top = ggml_view_2d(ctx, ids, 8, n_t, ids->nb[1], 0);
                     return ggml_get_rows(ctx, ggml_reshape_3d(ctx, p, 1, 256, n_t), top);
                 },
                 values_and_ids(256));
        // their sum, clamped in place, and the division (src/llama-graph.cpp:2057-2069)
        run_case("SUM_ROWS of the weights [8, n_t] (a reshape of [1, 8, n_t])", k_sum_rows, 0.0,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 8, n_t);
                     in.push_back(w);
                     return ggml_sum_rows(ctx, ggml_reshape_2d(ctx, w, 8, n_t));
                 },
                 random_in, row_sums(8), true);
        run_case("CLAMP in place of the sums [1, n_t] to [6.1e-5, inf]", k_clamp, 0.0,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_t);
                     in.push_back(s);
                     return ggml_clamp(ctx, s, 6.103515625e-5f, INFINITY);
                 },
                 [](int64_t n, size_t, uint32_t s) {
                     std::vector<float> v = random_values(n, s, 1e-3f);
                     return v;
                 });
        run_case("DIV of the weights [8, n_t] by their sums [1, n_t]", k_div, 0.0,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 8, n_t);
                     ggml_tensor * s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_t);
                     in.push_back(w);
                     in.push_back(s);
                     return ggml_div(ctx, ggml_reshape_2d(ctx, w, 8, n_t), s);
                 },
                 [](int64_t n, size_t i, uint32_t s) {
                     std::vector<float> v = random_values(n, s, 1.0f);
                     for (float & x : v) {
                         x = i == 1 ? 0.5f + std::fabs(x) : std::fabs(x);
                     }
                     return v;
                 });
        // the experts' outputs by their weights, and the sum of the 8 experts' views (src/llama-graph.cpp:2227-2254)
        run_case("MUL of the experts [2048, 8, n_t] by the weights [1, 8, n_t]", k_mul, 0.0,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * e = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 8, n_t);
                     ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 8, n_t);
                     in.push_back(e);
                     in.push_back(w);
                     return ggml_mul(ctx, e, w);
                 },
                 random_in);
        run_case("ADD of two views [2048, n_t] of the experts [2048, 8, n_t]", k_add, 0.0,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * e = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 8, n_t);
                     in.push_back(e);
                     ggml_tensor * a = ggml_view_2d(ctx, e, 2048, n_t, e->nb[2], 0);
                     ggml_tensor * b = ggml_view_2d(ctx, e, 2048, n_t, e->nb[2], 3 * e->nb[1]);
                     return ggml_add(ctx, a, b);
                 },
                 random_in);
        // the shared expert by its sigmoid gate [1, n_t] (qwen35moe.cpp:541)
        run_case("MUL of the shared expert [2048, n_t] by its gate [1, n_t]", k_mul, 0.0,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, n_t);
                     ggml_tensor * g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_t);
                     in.push_back(x);
                     in.push_back(g);
                     return ggml_mul(ctx, x, g);
                 },
                 random_in);
    }

    // edge cases
    run_case("SOFT_MAX in place [256, 7]", k_soft_max, k_moe_soft_max_nmse,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 7);
                 in.push_back(x);
                 return ggml_soft_max_inplace(ctx, x);
             },
             random_scaled(4.0f));
    run_case("SOFT_MAX with scale -0.5 [37, 5]", k_soft_max, k_moe_soft_max_nmse,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 37, 5);
                 in.push_back(x);
                 return ggml_soft_max_ext(ctx, x, nullptr, -0.5f, 0.0f);
             },
             random_scaled(8.0f));
    run_case("SOFT_MAX with -inf in every row [64, 9]", k_soft_max, k_moe_soft_max_nmse,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 9);
                 in.push_back(x);
                 return ggml_soft_max(ctx, x);
             },
             [](int64_t n, size_t, uint32_t s) {
                 std::vector<float> v = random_values(n, s, 4.0f);
                 for (int64_t i = 0; i < n; i += 3) {
                     v[(size_t) i] = -INFINITY;
                 }
                 return v;
             });
    run_case("SOFT_MAX with values to +-80 [256, 4]", k_soft_max, k_moe_soft_max_nmse,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
                 in.push_back(x);
                 return ggml_soft_max(ctx, x);
             },
             random_scaled(80.0f));
    run_case("SOFT_MAX of one column [1, 5]", k_soft_max, k_moe_soft_max_nmse,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 5);
                 in.push_back(x);
                 return ggml_soft_max(ctx, x);
             },
             random_in);
    run_case("SOFT_MAX 4-D [17, 5, 3, 2]", k_soft_max, k_moe_soft_max_nmse,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 17, 5, 3, 2);
                 in.push_back(x);
                 return ggml_soft_max(ctx, x);
             },
             random_in);
    run_case("SOFT_MAX of an empty view [256, 0]", k_soft_max, k_moe_soft_max_nmse,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
                 in.push_back(x);
                 x = ggml_view_2d(ctx, x, 256, 0, x->nb[1], 0);
                 return ggml_soft_max(ctx, x);
             },
             random_in);
    // ties: 7 levels over 256 columns, so std::sort's order among equal keys is compared (descending and ascending)
    for (ggml_sort_order order : { GGML_SORT_ORDER_DESC, GGML_SORT_ORDER_ASC }) {
        run_case("ARGSORT with ties [256, 33]", k_argsort, 0.0,
                 [order](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 33);
                     in.push_back(x);
                     return ggml_argsort(ctx, x, order);
                 },
                 [](int64_t n, size_t, uint32_t s) {
                     std::vector<float> v = random_values(n, s, 1.0f);
                     for (float & x : v) {
                         x = std::round(x * 3.0f) / 3.0f;
                     }
                     return v;
                 });
    }
    run_case("ARGSORT of equal rows [100, 3]", k_argsort, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 100, 3);
                 in.push_back(x);
                 return ggml_argsort(ctx, x, GGML_SORT_ORDER_DESC);
             },
             [](int64_t n, size_t, uint32_t) { return std::vector<float>((size_t) n, 0.25f); });
    run_case("ARGSORT with +-inf [64, 4]", k_argsort, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
                 in.push_back(x);
                 return ggml_argsort(ctx, x, GGML_SORT_ORDER_DESC);
             },
             [](int64_t n, size_t, uint32_t s) {
                 std::vector<float> v = random_values(n, s, 1.0f);
                 for (int64_t i = 0; i < n; i += 5) {
                     v[(size_t) i] = i % 2 == 0 ? INFINITY : -INFINITY;
                 }
                 return v;
             });
    run_case("ARGSORT ascending of long rows [5000, 3]", k_argsort, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5000, 3);
                 in.push_back(x);
                 return ggml_argsort(ctx, x, GGML_SORT_ORDER_ASC);
             },
             random_in);
    run_case("ARGSORT of one column [1, 9]", k_argsort, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 9);
                 in.push_back(x);
                 return ggml_argsort(ctx, x, GGML_SORT_ORDER_DESC);
             },
             random_in);
    run_case("ARGSORT 4-D [16, 3, 2, 2]", k_argsort, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 3, 2, 2);
                 in.push_back(x);
                 return ggml_argsort(ctx, x, GGML_SORT_ORDER_DESC);
             },
             random_in);
    run_case("ARGSORT of an empty view [256, 0]", k_argsort, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
                 in.push_back(x);
                 x = ggml_view_2d(ctx, x, 256, 0, x->nb[1], 0);
                 return ggml_argsort(ctx, x, GGML_SORT_ORDER_DESC);
             },
             random_in);
    run_case("SUM_ROWS of long rows [3000, 5]", k_sum_rows, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3000, 5);
                 in.push_back(x);
                 return ggml_sum_rows(ctx, x);
             },
             random_in, row_sums(3000), true);
    run_case("SUM_ROWS of strided rows [100, 7] of [1000, 7]", k_sum_rows, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1000, 7);
                 in.push_back(x);
                 return ggml_sum_rows(ctx, ggml_view_2d(ctx, x, 100, 7, x->nb[1], 13 * sizeof(float)));
             },
             random_in, nullptr, true);
    run_case("SUM_ROWS 4-D [10, 3, 2, 2]", k_sum_rows, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 10, 3, 2, 2);
                 in.push_back(x);
                 return ggml_sum_rows(ctx, x);
             },
             random_in, row_sums(10), true);
    run_case("SUM_ROWS of empty rows [0, 4]", k_sum_rows, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
                 in.push_back(x);
                 return ggml_sum_rows(ctx, ggml_view_2d(ctx, x, 0, 4, x->nb[1], 0));
             },
             random_in, nullptr, true);
    // NaN (ggml-cpu's macros give the upper bound), infinities and values beyond both bounds
    run_case("CLAMP in place to [-1, 1] with NaN and +-inf [64, 3]", k_clamp, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 3);
                 in.push_back(x);
                 return ggml_clamp(ctx, x, -1.0f, 1.0f);
             },
             [](int64_t n, size_t, uint32_t s) {
                 std::vector<float> v = random_values(n, s, 4.0f);
                 for (int64_t i = 0; i < n; i += 7) {
                     v[(size_t) i] = NAN;
                 }
                 for (int64_t i = 3; i < n; i += 11) {
                     v[(size_t) i] = i % 2 == 0 ? INFINITY : -INFINITY;
                 }
                 return v;
             });
    run_case("CLAMP 3-D [10, 4, 3]", k_clamp, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 10, 4, 3);
                 in.push_back(x);
                 return ggml_clamp(ctx, x, -0.5f, 0.25f);
             },
             random_in);
    run_case("DIV of the same shape [64, 9]", k_div, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 9);
                 ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 9);
                 in.push_back(a);
                 in.push_back(b);
                 return ggml_div(ctx, a, b);
             },
             random_in);
    run_case("DIV of [64, 9] by one row [64, 1]", k_div, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 9);
                 ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 1);
                 in.push_back(a);
                 in.push_back(b);
                 return ggml_div(ctx, a, b);
             },
             random_in);
    run_case("DIV of one row [37] by [37]", k_div, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 37);
                 ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 37);
                 in.push_back(a);
                 in.push_back(b);
                 return ggml_div(ctx, a, b);
             },
             random_in);
    run_case("DIV of [16, 4] by zeros and infinities [1, 4]", k_div, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 4);
                 ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 4);
                 in.push_back(a);
                 in.push_back(b);
                 return ggml_div(ctx, a, b);
             },
             [](int64_t n, size_t i, uint32_t s) {
                 std::vector<float> v = random_values(n, s, 1.0f);
                 if (i == 1) {
                     v = { 0.0f, -0.0f, INFINITY, 0.75f };
                 } else {
                     for (float & x : v) {
                         x = x >= 0.0f ? x + 0.5f : x - 0.5f;  // no 0 / 0
                     }
                 }
                 return v;
             });
    // src1's elements strided and repeated inside a row: ggml-cpu's non-contiguous path, so the reference runs
    run_case("DIV of [64, 8] by a transposed [8, 8] (strided src1 elements, repeated in a row)", k_div, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 8);
                 ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8);
                 in.push_back(a);
                 in.push_back(b);
                 return ggml_div(ctx, a, ggml_transpose(ctx, b));
             },
             random_in);
    run_case("DIV 4-D [8, 4, 3, 2] by [8, 1, 3, 1]", k_div, 0.0,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 4, 3, 2);
                 ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 1, 3, 1);
                 in.push_back(a);
                 in.push_back(b);
                 return ggml_div(ctx, a, b);
             },
             random_in);

    // the contracts: the router's forms taken by their kernels (the new five, and the router's other ops on the
    // contracts that already take them), and the forms that stay refused
    int n_claims = 0, n_claims_ok = 0;
    {
        ggml_init_params p   = { ggml_tensor_overhead() * 96, nullptr, true };
        ggml_context *   ctx = ggml_init(p);
        REQUIRE(ctx != nullptr);
        const int64_t n_t      = 512;
        ggml_tensor * logits   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, n_t);
        ggml_tensor * f16      = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, n_t);
        ggml_tensor * mask     = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, n_t);
        ggml_tensor * sinks    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_tensor * x3d      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 4, 3);
        ggml_tensor * gate_inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 256);
        ggml_tensor * shexp_g  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
        ggml_tensor * cur      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, n_t);
        ggml_tensor * experts  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 8, n_t);
        ggml_tensor * gate_up  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, 8, n_t);
        ggml_set_name(gate_inp, "blk.0.ffn_gate_inp.weight");
        ggml_set_name(shexp_g, "blk.0.ffn_gate_inp_shexp.weight");
        const auto taken = [&](ggml_tensor * node, const spacemit_plugin_kernel * want) {
            const spacemit_op * got = spacemit_find_op(ops, node, buft);
            n_claims++;
            const bool ok = got != nullptr && got->n_steps == 1 && got->steps[0] == want;
            n_claims_ok += ok;
            if (!ok) {
                std::fprintf(stderr, "FAIL: contracts, %s not taken by %s\n", ggml_op_desc(node), want->name);
                g_failures++;
            }
        };
        const auto refused = [&](ggml_tensor * node, const char * what) {
            n_claims++;
            const bool ok = !ggml_backend_dev_supports_op(dev, node);
            n_claims_ok += ok;
            if (!ok) {
                std::fprintf(stderr, "FAIL: contracts, %s claimed but must be refused\n", what);
                g_failures++;
            }
        };
        // build_moe_ffn's chain as the model builds it
        ggml_tensor * probs = ggml_soft_max(ctx, logits);
        taken(probs, k_soft_max);
        ggml_tensor * top = ggml_argsort_top_k(ctx, probs, 8);
        taken(top->view_src, k_argsort);
        ggml_tensor * weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, 256, n_t), top);
        taken(weights, k_get_rows);
        ggml_tensor * w2  = ggml_reshape_2d(ctx, weights, 8, n_t);
        ggml_tensor * sum = ggml_sum_rows(ctx, w2);
        taken(sum, k_sum_rows);
        ggml_tensor * clamped = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
        taken(clamped, k_clamp);
        ggml_tensor * normed = ggml_div(ctx, w2, clamped);
        taken(normed, k_div);
        ggml_tensor * weighted = ggml_mul(ctx, experts, ggml_reshape_3d(ctx, normed, 1, 8, n_t));
        taken(weighted, k_mul);
        taken(ggml_add(ctx, ggml_view_2d(ctx, weighted, 2048, n_t, weighted->nb[2], 0),
                       ggml_view_2d(ctx, weighted, 2048, n_t, weighted->nb[2], weighted->nb[1])),
              k_add);
        taken(ggml_swiglu_split(ctx, gate_up, gate_up), entry("swiglu.f32"));
        taken(ggml_mul_mat(ctx, gate_inp, cur), entry("mul_mat.f32"));
        ggml_tensor * shared_gate = ggml_mul_mat(ctx, shexp_g, cur);
        taken(shared_gate, entry("mul_mat.f32"));
        ggml_tensor * shared_sig = ggml_sigmoid(ctx, shared_gate);
        taken(shared_sig, entry("sigmoid.f32"));
        taken(ggml_mul(ctx, cur, shared_sig), k_mul);
        // refused
        refused(ggml_soft_max_ext(ctx, logits, mask, 1.0f, 0.0f), "SOFT_MAX with a mask");
        refused(ggml_soft_max_ext(ctx, logits, mask, 1.0f, 8.0f), "SOFT_MAX with ALiBi");
        {
            ggml_tensor * with_sinks = ggml_soft_max(ctx, logits);
            ggml_soft_max_add_sinks(with_sinks, sinks);
            refused(with_sinks, "SOFT_MAX with sinks");
        }
        refused(ggml_soft_max(ctx, f16), "SOFT_MAX of F16");
        refused(ggml_argsort(ctx, f16, GGML_SORT_ORDER_DESC), "ARGSORT of F16");
        refused(ggml_argsort(ctx, ggml_view_3d(ctx, x3d, 16, 2, 3, x3d->nb[1], x3d->nb[2], 0), GGML_SORT_ORDER_DESC),
                "ARGSORT of rows not evenly spaced");
        refused(ggml_argsort(ctx, ggml_transpose(ctx, logits), GGML_SORT_ORDER_DESC), "ARGSORT of a transposed source");
        refused(ggml_top_k(ctx, logits, 8), "TOP_K (not built by the model)");
        refused(ggml_sum_rows(ctx, f16), "SUM_ROWS of F16");
        refused(ggml_sum_rows(ctx, ggml_transpose(ctx, logits)), "SUM_ROWS of a transposed source");
        refused(ggml_clamp(ctx, f16, 0.0f, 1.0f), "CLAMP of F16");
        refused(ggml_clamp(ctx, ggml_view_3d(ctx, x3d, 16, 2, 3, x3d->nb[1], x3d->nb[2], 0), 0.0f, 1.0f),
                "CLAMP of rows not evenly spaced");
        refused(ggml_div(ctx, f16, f16), "DIV of F16");
        refused(ggml_div(ctx, logits, f16), "DIV of F32 by F16");
        refused(ggml_div(ctx, ggml_transpose(ctx, logits), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_t, 256)),
                "DIV of a transposed src0");
        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    std::printf("moe      the MoE router's ops, F32 in our buffer and in a CPU buffer, at Qwen3.6-35B-A3B's shapes (256 "
                "experts, 8 used, 2048 channels; 1 and 512 tokens) and edge cases (in place, negative scale, -inf, "
                "+-80, ties, equal rows, +-inf, NaN, long rows, one column, 3-D and 4-D, empty, strided, division by "
                "0 and inf, broadcast)%s: %d/%d cases as the CPU on 8 tiles and directly on 1 and 8; bit for bit in "
                "%d/%d, SUM_ROWS as the program's double sums in %d, SOFT_MAX max NMSE %.2e (bound %.0e), other "
                "bounded max NMSE %.2e (bound %.0e); the router's 13 claims and the 15 refused forms (a mask, ALiBi, "
                "sinks, F16, rows not evenly spaced, transposed sources, TOP_K): %d/%d\n",
                reference_label(), n_ok, n_cases, n_exact, n_exact_cases, n_mine, worst_soft_max, k_moe_soft_max_nmse,
                worst_bound, k_moe_sum_max_nmse, n_claims_ok, n_claims);
    return 0;
}

// test-only: the RVV kernels' exp, log1p, SILU, SIGMOID and SOFTPLUS (flagos-spacemit-seq.cpp, from SpacemiT's fork)
// lane by lane in scalar C++: the same constants and steps, a fused multiply-add (std::fma) where the RVV code has
// vfmacc, vfnmsac or vfmadd, the exponent built in the bits. A lane depends only on its own input, so this is the
// kernels' arithmetic on any host; checked over a sweep of all floats against libm, not the intrinsics themselves
static float seq_lane_bits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static float seq_lane_expf(float x) {
    const float    r  = 0x1.8p23f;
    const float    z  = std::fma(0x1.715476p+0f, x, r);
    const float    n  = z - r;
    const float    b0 = std::fma(-0x1.62e4p-1f, n, x);
    const float    b  = std::fma(-0x1.7f7d1cp-20f, n, b0);
    uint32_t       zb;
    std::memcpy(&zb, &z, sizeof(zb));
    const uint32_t e  = zb << 23;
    const float    k  = seq_lane_bits(e + 0x3f800000u);
    const float    u  = b * b;
    const float    jb = b * 0x1.ffffecp-1f;
    const float    p0 = std::fma(0x1.555e66p-3f, b, 0x1.fffdb6p-2f);
    const float    p1 = std::fma(0x1.0e4020p-7f, b, 0x1.573e2ep-5f);
    const float    p2 = std::fma(p1, u, p0);
    const float    j  = std::fma(p2, u, jb);
    if (!(std::fabs(n) > 126.0f)) {
        return std::fma(k, j, k);
    }
    const uint32_t d  = n <= 0.0f ? 0x82000000u : 0u;
    const float    s1 = seq_lane_bits(d + 0x7f000000u);
    const float    s2 = seq_lane_bits(e - d);
    if (std::fabs(n) > 192.0f) {
        return s1 * s1;
    }
    const float t = std::fma(s2, j, s2);
    return t * s1;
}

static float seq_lane_log1pf_01(float t) {
    const float u   = t + 1.0f;
    const float um1 = u - 1.0f;
    const float err = (t - um1) / u;
    const bool  big = u > 0x1.6a09e6p+0f;
    const float m   = big ? u * 0.5f : u;
    const float e   = big ? 1.0f : 0.0f;
    const float f   = m - 1.0f;
    const float z   = f * f;
    float       p   = 7.0376836292E-2f;
    for (const float c : { -1.1514610310E-1f, 1.1676998740E-1f, -1.2420140846E-1f, 1.4249322787E-1f,
                           -1.6668057665E-1f, 2.0000714765E-1f, -2.4999993993E-1f, 3.3333331174E-1f }) {
        p = std::fma(p, f, c);
    }
    const float pf = p * f;
    float       y  = pf * z;
    y              = std::fma(-2.12194440e-4f, e, y);
    y              = std::fma(-0.5f, z, y);
    y              = y + err;
    const float fy = f + y;
    return std::fma(0.693359375f, e, fy);
}

static float seq_lane_silu(float x) {
    const float d = seq_lane_expf(-x) + 1.0f;
    return x / d;
}

static float seq_lane_sigmoid(float x) {
    const float d = seq_lane_expf(-x) + 1.0f;
    return 1.0f / d;
}

static float seq_lane_softplus(float x) {
    const float m = x > 0.0f ? x : 0.0f;  // vfmax_vf(x, 0)
    const float l = seq_lane_log1pf_01(seq_lane_expf(-std::fabs(x)));
    return m + l;
}

// one lane model over the sweep: the worst ulps against the exact value in double where that is a normal float (and
// x is inside the formula's range, in_range), the worst |difference| / max(1, |y|) against ggml-cpu's scalar formula
// (libm), and the inputs where one of the two is finite and the other not, or both non-finite but unequal
struct seq_lane_stat {
    double max_ulp = 0.0, max_cpu = 0.0;
    int64_t n_nonfinite = 0;

    void add(double exact, bool in_range, float got, float cpu) {
        if (in_range && std::isfinite(got) && std::isfinite(exact) && std::fabs(exact) >= 0x1p-126) {
            int ex;
            std::frexp(exact, &ex);
            max_ulp = std::max(max_ulp, std::fabs((double) got - exact) / std::ldexp(1.0, ex - 24));
        }
        if (std::isfinite(got) && std::isfinite(cpu)) {
            max_cpu = std::max(max_cpu, std::fabs((double) got - cpu) / std::max(1.0, std::fabs((double) cpu)));
        } else if (!(got == cpu) && !(std::isnan(got) && std::isnan(cpu))) {
            n_nonfinite++;
        }
    }
};

// plan.md "The float matmul: all tiles for one token, blocks for a prompt": the shapes of the F32 matmul's split rule
// and its blocks of 16 token rows. The router's [2048, 256] and the Gated DeltaNet's [2048, 32] at the token counts
// where the rule turns over and around a block, rows left over after the groups of 4 with fewer units than tiles
// ([64, 7]), one weight row, and a 3-D batch (the weight broadcast)
struct mm_f32_shape {
    int64_t k, n, rows, ne2;
};

static const mm_f32_shape k_mm_f32_shapes[] = {
    { 2048, 256, 1, 1 },  { 2048, 256, 2, 1 },  { 2048, 256, 31, 1 }, { 2048, 256, 255, 1 },
    { 2048, 256, 256, 1 }, { 2048, 256, 501, 1 }, { 2048, 32, 1, 1 },  { 2048, 32, 31, 1 },
    { 2048, 32, 32, 1 },  { 2048, 32, 33, 1 },  { 64, 7, 1, 1 },      { 64, 7, 3, 1 },
    { 64, 7, 9, 1 },      { 2048, 1, 1, 1 },    { 2048, 1, 40, 1 },   { 2048, 256, 5, 3 },
};

// the shape's MUL_MAT node, its weight named as the model's router weight is
static ggml_tensor * mm_f32_build(const mm_f32_shape & s, ggml_context * ctx, std::vector<ggml_tensor *> & in) {
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, s.n);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.k, s.rows, s.ne2);
    ggml_set_name(w, "blk.0.ffn_gate_inp.weight");
    in.push_back(w);
    in.push_back(x);
    return ggml_mul_mat(ctx, w, x);
}

static std::string mm_f32_what(const mm_f32_shape & s) {
    std::string what = "MUL_MAT of an F32 [" + std::to_string(s.k) + ", " + std::to_string(s.n) + "] weight by [" +
                       std::to_string(s.k) + ", " + std::to_string(s.rows);
    if (s.ne2 != 1) {
        what += ", " + std::to_string(s.ne2);
    }
    return what + "]";
}

static int check_seq_ops(ggml_backend_dev_t dev) {
    env_override router_f32("FLAGOS_SPACEMIT_ROUTER", "f32");  // the F32 matmul path: its weights carry the router's name
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);
    // SSM_CONV: the reference is ggml-cpu's bytes; the RVV kernel's fused taps are not (K3 2026-11-14: NMSE 2.7e-15,
    // 3.3e-07 of max(1, |y|) against the ggml-cpu GCC built), so on the K3 it gets a rounding bound
#if defined(__riscv)
    const seq_bound k_ssm_conv = *reference_label() != '\0' ? k_seq_exact : seq_bound{ 1e-12, 1e-6 };
#else
    const seq_bound k_ssm_conv = k_seq_exact;
#endif
    ggml_backend_cpu_set_n_threads(cpu, 1);  // a correctness reference: single-threaded (PROJECT.md section 6, item 28)

    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    const auto                     entry     = [&](const char * name) -> const spacemit_plugin_kernel * {
        for (uint32_t k = 0; k < n_kernels; k++) {
            if (kernels[k].name != nullptr && std::strcmp(kernels[k].name, name) == 0) {
                return &kernels[k];
            }
        }
        return nullptr;
    };
    const spacemit_plugin_kernel * k_silu     = entry("silu.f32");
    const spacemit_plugin_kernel * k_sigmoid  = entry("sigmoid.f32");
    const spacemit_plugin_kernel * k_softplus = entry("softplus.f32");
    const spacemit_plugin_kernel * k_l2       = entry("l2_norm.f32");
    const spacemit_plugin_kernel * k_conv     = entry("ssm_conv.f32");
    const spacemit_plugin_kernel * k_concat   = entry("concat.f32");
    const spacemit_plugin_kernel * k_copy     = entry("cpy.f32");  // CONT's kernel
    const spacemit_plugin_kernel * k_mm       = entry("mul_mat.f32");
    REQUIRE(k_silu != nullptr && k_sigmoid != nullptr && k_softplus != nullptr && k_l2 != nullptr &&
            k_conv != nullptr && k_concat != nullptr && k_copy != nullptr && k_mm != nullptr);
    spacemit_ops ops;
    std::string  detail;
    REQUIRE(spacemit_ops_bind(table, &ops, &detail));
    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    uint32_t   seed = 900;
    int        n_cases = 0, n_ok = 0, n_exact = 0, n_exact_cases = 0;
    double     worst_exact = 0.0;
    // the worst NMSE and |difference| / max(1, |y|) of each bounded op, by kernel
    std::map<const spacemit_plugin_kernel *, std::pair<double, double>> worst;
    const auto run_case = [&](const char * what, const spacemit_plugin_kernel * kernel, seq_bound bound,
                              const seq_build & build, float scale = 1.0f) {
        for (bool host : { false, true }) {
            const seq_op_result r = seq_op_run(dev, backend, cpu, kernel, host, build, seed, scale);
            n_cases++;
            const bool exact_wanted = bound.nmse == 0.0 && bound.abs == 0.0;
            const bool ok           = r.claimed && r.ran && r.err >= 0.0 && r.err <= bound.nmse && r.abs >= 0.0 &&
                            r.abs <= bound.abs && (!exact_wanted || r.exact);
            n_ok += ok;
            if (exact_wanted) {
                n_exact_cases++;
                n_exact += r.exact ? 1 : 0;
                worst_exact = std::max(worst_exact, r.err);
            } else {
                auto it = worst.emplace(kernel, std::make_pair(-1.0, -1.0)).first;
                it->second.first  = std::max(it->second.first, r.err);
                it->second.second = std::max(it->second.second, r.abs);
            }
            if (!ok) {
                std::fprintf(stderr,
                             "FAIL: %s, the tensors in %s: %s, %s; NMSE %.3e (bound %.0e), |difference| / max(1, |y|) "
                             "%.3e (bound %.0e), bytes as the CPU's: %s\n",
                             what, host ? "a CPU buffer" : "our buffer", r.claimed ? "claimed" : "refused",
                             r.ran ? "ran" : "failed", r.err, bound.nmse, r.abs, bound.abs, r.exact ? "yes" : "no");
                g_failures++;
            }
        }
        seed += 8;
    };

    // Qwen3.5-4B's shapes: 2560 embedding channels, 8192 convolution channels (2 x 16 x 128 key and value channels +
    // 4096 value channels), 128 per head, 16 key and 32 value heads, a 4-tap convolution; one token and a 512-token
    // batch, with one sequence (the model's conv_input is [d_conv - 1 + n_t, 8192, n_seqs])
    for (int64_t n_t : { (int64_t) 1, (int64_t) 512 }) {
        // D. SILU on the convolution's output [8192, n_t] and on the gate z [128, 32, n_t] (qwen35.cpp:398, :253)
        run_case("SILU on the convolution output [8192, n_t]", k_silu, k_seq_silu,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8192, n_t);
                     in.push_back(x);
                     return ggml_silu(ctx, x);
                 });
        run_case("SILU on the gate [128, 32, n_t]", k_silu, k_seq_silu,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * z = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 32, n_t, 1);
                     in.push_back(z);
                     return ggml_silu(ctx, z);
                 });
        // D. SIGMOID on beta [1, 32, n_t] (qwen35.cpp:366) and on the attention gate [4096, n_t] (:327)
        run_case("SIGMOID on beta [1, 32, n_t]", k_sigmoid, k_seq_sigmoid,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 32, n_t, 1);
                     in.push_back(b);
                     return ggml_sigmoid(ctx, b);
                 });
        run_case("SIGMOID on the attention gate [4096, n_t]", k_sigmoid, k_seq_sigmoid,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, n_t);
                     in.push_back(g);
                     return ggml_sigmoid(ctx, g);
                 });
        // D. SOFTPLUS on the biased alpha [32, n_t] (qwen35.cpp:374)
        run_case("SOFTPLUS on alpha [32, n_t]", k_softplus, k_seq_softplus,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, n_t, 1);
                     in.push_back(a);
                     return ggml_softplus(ctx, a);
                 });
        // E. L2_NORM on the convolved queries and keys: views [128, 16, n_t] of the convolution's output with rows
        // 8192 floats apart, the keys 2048 floats in (qwen35.cpp:410-432)
        for (int64_t offset : { (int64_t) 0, (int64_t) 2048 }) {
            run_case("L2_NORM on a strided view [128, 16, n_t] of [8192, n_t]", k_l2, k_seq_l2_norm,
                     [n_t, offset](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                         ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8192, n_t);
                         in.push_back(c);
                         ggml_tensor * v = ggml_view_4d(ctx, c, 128, 16, n_t, 1, 128 * sizeof(float),
                                                        8192 * sizeof(float), 8192 * n_t * sizeof(float),
                                                        offset * sizeof(float));
                         return ggml_l2_norm(ctx, v, 1e-6f);
                     });
        }
        // F. SSM_CONV on [3 + n_t, 8192, 1] with a [4, 8192] kernel (qwen35.cpp:395)
        run_case("SSM_CONV [3 + n_t, 8192, 1] with a [4, 8192] kernel", k_conv, k_ssm_conv,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * sx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 + n_t, 8192, 1);
                     ggml_tensor * c  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8192);
                     in.push_back(sx);
                     in.push_back(c);
                     return ggml_ssm_conv(ctx, sx, c);
                 });
        // F. CONCAT along dimension 0 of the state's 3 columns and the transposed projection (delta-net-base.cpp:470)
        run_case("CONCAT dim 0 of [3, 8192, 1] and a transposed [n_t, 8192, 1]", k_concat, k_seq_exact,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * st   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 8192, 1);
                     ggml_tensor * qkv  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8192, n_t, 1);
                     in.push_back(st);
                     in.push_back(qkv);
                     return ggml_concat(ctx, st, ggml_transpose(ctx, qkv), 0);
                 });
        // F. CONT of the attention layers' gate: a view [256, 16, n_t] of [8192, n_t] with rows 512 floats apart
        // (qwen35.cpp:293-297)
        run_case("CONT of a strided view [256, 16, n_t] into [4096, n_t]", k_copy, k_seq_exact,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * qg = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8192, n_t);
                     in.push_back(qg);
                     ggml_tensor * v = ggml_view_3d(ctx, qg, 256, 16, n_t, 512 * sizeof(float), 8192 * sizeof(float),
                                                    256 * sizeof(float));
                     return ggml_cont_2d(ctx, v, 4096, n_t);
                 });
        // G. MUL_MAT of the F32 ssm_alpha / ssm_beta weight [2560, 32] by the layer's input [2560, n_t]
        // (qwen35.cpp:362, :369)
        run_case("MUL_MAT of an F32 [2560, 32] weight by [2560, n_t]", k_mm, k_seq_mul_mat,
                 [n_t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 32);
                     ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, n_t);
                     ggml_set_name(w, "blk.0.ssm_alpha.weight");
                     in.push_back(w);
                     in.push_back(x);
                     return ggml_mul_mat(ctx, w, x);
                 });
    }

    // edge cases: an empty tensor, one row, strided rows where the contract allows them, and more sequences than one
    run_case("SILU on an empty view [8192, 0]", k_silu, k_seq_silu,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8192, 4);
                 in.push_back(x);
                 return ggml_silu(ctx, ggml_view_2d(ctx, x, 8192, 0, x->nb[1], 0));
             });
    run_case("SIGMOID on one row [1000]", k_sigmoid, k_seq_sigmoid, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000);
        in.push_back(x);
        return ggml_sigmoid(ctx, x);
    });
    run_case("SOFTPLUS on strided rows [100, 33] of [1000, 33]", k_softplus, k_seq_softplus,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1000, 33);
                 in.push_back(x);
                 return ggml_softplus(ctx, ggml_view_2d(ctx, x, 100, 33, x->nb[1], 7 * sizeof(float)));
             });
    run_case("SILU on strided rows [100, 33] of [1000, 33]", k_silu, k_seq_silu,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1000, 33);
                 in.push_back(x);
                 return ggml_silu(ctx, ggml_view_2d(ctx, x, 100, 33, x->nb[1], 7 * sizeof(float)));
             });
    // values up to +-40, over SOFTPLUS' 20 cut-off (unary-ops.cpp op_softplus) and into SILU's and SIGMOID's tails
    run_case("SOFTPLUS on [32, 33] with values to +-40", k_softplus, k_seq_softplus,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 33);
                 in.push_back(x);
                 return ggml_softplus(ctx, x);
             },
             40.0f);
    run_case("SIGMOID on [32, 33] with values to +-40", k_sigmoid, k_seq_sigmoid,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 33);
                 in.push_back(x);
                 return ggml_sigmoid(ctx, x);
             },
             40.0f);
    run_case("SILU on [32, 33] with values to +-40", k_silu, k_seq_silu,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 33);
                 in.push_back(x);
                 return ggml_silu(ctx, x);
             },
             40.0f);
    // the ranges that stress the exp: near 0, subnormal inputs (exp(-x) is 1), to +-88 (exp's overflow at 88.72 and
    // its two-step scaling from |x| ~ 87.3) and to +-1000 (exp inf or 0, the results still finite)
    const struct {
        const char * what;
        float        scale;
    } exp_ranges[] = {
        { "near 0 (to +-1e-6)", 1e-6f },
        { "subnormal (to +-1e-38)", 1e-38f },
        { "to +-88", 88.0f },
        { "to +-1000", 1000.0f },
    };
    for (const auto & er : exp_ranges) {
        const struct {
            const char *                   name;
            const spacemit_plugin_kernel * kernel;
            seq_bound                      bound;
            ggml_tensor * (*op)(ggml_context *, ggml_tensor *);
        } unary[] = {
            { "SILU", k_silu, k_seq_silu, ggml_silu },
            { "SIGMOID", k_sigmoid, k_seq_sigmoid, ggml_sigmoid },
            { "SOFTPLUS", k_softplus, k_seq_softplus, ggml_softplus },
        };
        for (const auto & u : unary) {
            const std::string what = std::string(u.name) + " on [1024, 8] with values " + er.what;
            run_case(what.c_str(), u.kernel, u.bound,
                     [&u](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                         ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 8);
                         in.push_back(x);
                         return u.op(ctx, x);
                     },
                     er.scale);
        }
    }
    // L2_NORM where the squares fall below float's normal range and where they reach 1e36
    for (const float scale : { 1e-20f, 1e18f }) {
        run_case(scale < 1.0f ? "L2_NORM on [128, 64] with values to +-1e-20" : "L2_NORM on [128, 64] with values to +-1e18",
                 k_l2, k_seq_l2_norm,
                 [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                     ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 64);
                     in.push_back(x);
                     return ggml_l2_norm(ctx, x, 1e-6f);
                 },
                 scale);
    }
    run_case("L2_NORM on one row [128], eps 0", k_l2, k_seq_l2_norm, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
        in.push_back(x);
        return ggml_l2_norm(ctx, x, 0.0f);
    });
    run_case("L2_NORM on an empty view [128, 0]", k_l2, k_seq_l2_norm, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 4);
        in.push_back(x);
        return ggml_l2_norm(ctx, ggml_view_2d(ctx, x, 128, 0, x->nb[1], 0), 1e-6f);
    });
    run_case("SSM_CONV [7, 64, 3] with a [4, 64] kernel (3 sequences)", k_conv, k_ssm_conv,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * sx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 64, 3);
                 ggml_tensor * c  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 64);
                 in.push_back(sx);
                 in.push_back(c);
                 return ggml_ssm_conv(ctx, sx, c);
             });
    run_case("SSM_CONV [4, 1, 1] with a [4, 1] kernel (one channel)", k_conv, k_ssm_conv,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * sx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 1, 1);
                 ggml_tensor * c  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
                 in.push_back(sx);
                 in.push_back(c);
                 return ggml_ssm_conv(ctx, sx, c);
             });
    run_case("SSM_CONV [3, 64, 1] with a [4, 64] kernel (no token)", k_conv, k_ssm_conv,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * sx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 64, 1);
                 ggml_tensor * c  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 64);
                 in.push_back(sx);
                 in.push_back(c);
                 return ggml_ssm_conv(ctx, sx, c);
             });
    run_case("CONCAT dim 0 of [3, 64, 3] and a transposed [4, 64, 3] (3 sequences)", k_concat, k_seq_exact,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * st  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 64, 3);
                 ggml_tensor * qkv = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 4, 3);
                 in.push_back(st);
                 in.push_back(qkv);
                 return ggml_concat(ctx, st, ggml_transpose(ctx, qkv), 0);
             });
    run_case("CONCAT dim 0 of an empty [0, 64, 1] and [4, 64, 1]", k_concat, k_seq_exact,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 64, 1);
                 ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 64, 1);
                 in.push_back(a);
                 in.push_back(b);
                 return ggml_concat(ctx, ggml_view_3d(ctx, a, 0, 64, 1, a->nb[1], a->nb[2], 0), b, 0);
             });
    run_case("CONT of a permuted [16, 8, 3, 2] into [768]", k_copy, k_seq_exact,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 8, 3, 2);
                 in.push_back(x);
                 return ggml_cont_1d(ctx, ggml_permute(ctx, x, 0, 2, 1, 3), 768);
             });
    run_case("CONT of an empty view [128, 0]", k_copy, k_seq_exact, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 4);
        in.push_back(x);
        return ggml_cont(ctx, ggml_view_2d(ctx, x, 128, 0, x->nb[1], 0));
    });
    run_case("MUL_MAT of an F32 [2560, 48] weight by [2560, 1] (the 27B's head count)", k_mm, k_seq_mul_mat,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 48);
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 1);
                 in.push_back(w);
                 in.push_back(x);
                 return ggml_mul_mat(ctx, w, x);
             });
    run_case("MUL_MAT of an F32 [7, 3] weight by [7, 5, 2, 2] (broadcast)", k_mm, k_seq_mul_mat,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 7, 3);
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 7, 5, 2, 2);
                 in.push_back(w);
                 in.push_back(x);
                 return ggml_mul_mat(ctx, w, x);
             });
    // the RVV kernel's blocks of 4 weight rows and its strips of K: N whole blocks, a remainder and both; K under, at
    // and over a strip of 16, 32 and 64 lanes (VLEN 256, 512 and 1024 at m2) and the model's 2560; 1, 2 and 9 rows
    {
        const int64_t mm_rows[] = { 1, 2, 9 };
        int           i         = 0;
        for (const int64_t N : { 1, 3, 4, 5, 16 }) {
            for (const int64_t K : { 1, 7, 31, 32, 33, 65, 128, 2560 }) {
                const int64_t     M    = mm_rows[i++ % 3];
                const std::string what = "MUL_MAT of an F32 [" + std::to_string(K) + ", " + std::to_string(N) +
                                         "] weight by [" + std::to_string(K) + ", " + std::to_string(M) + "]";
                run_case(what.c_str(), k_mm, k_seq_mul_mat, [N, K, M](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
                    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
                    in.push_back(w);
                    in.push_back(x);
                    return ggml_mul_mat(ctx, w, x);
                });
            }
        }
    }
    run_case("MUL_MAT of an F32 [33, 5] weight by [33, 2, 3, 2] (broadcast)", k_mm, k_seq_mul_mat,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 33, 5);
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 33, 2, 3, 2);
                 in.push_back(w);
                 in.push_back(x);
                 return ggml_mul_mat(ctx, w, x);
             });
    run_case("MUL_MAT of an F32 [2560, 32] weight by an empty [2560, 0]", k_mm, k_seq_mul_mat,
             [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                 ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 32);
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 4);
                 in.push_back(w);
                 in.push_back(x);
                 return ggml_mul_mat(ctx, w, ggml_view_2d(ctx, x, 2560, 0, x->nb[1], 0));
             });
    // the split rule's and the blocks' shapes (check_mul_mat_f32_split compares the splits with each other)
    for (const mm_f32_shape & s : k_mm_f32_shapes) {
        run_case(mm_f32_what(s).c_str(), k_mm, k_seq_mul_mat,
                 [&s](ggml_context * ctx, std::vector<ggml_tensor *> & in) { return mm_f32_build(s, ctx, in); });
    }

    // the contracts: what each claim takes, and the forms that stay refused - one per op
    int        n_claims = 0, n_claims_ok = 0;
    {
        ggml_init_params p   = { ggml_tensor_overhead() * 64, nullptr, true };
        ggml_context *   ctx = ggml_init(p);
        REQUIRE(ctx != nullptr);
        ggml_tensor * f32   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 16);
        ggml_tensor * f16   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 128, 16);
        ggml_tensor * w32   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 32);
        ggml_tensor * w16   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 128, 32);
        ggml_tensor * w3d   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 32, 2);
        ggml_tensor * q4    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 32);
        ggml_tensor * x256  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
        ggml_tensor * sx    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 64, 1);
        ggml_tensor * sxw   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 9, 64, 1);
        ggml_tensor * cw    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 64);
        ggml_tensor * cw16  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4, 64);
        ggml_tensor * cat_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 64, 1);
        ggml_tensor * cat_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 64, 1);
        ggml_set_name(w32, "blk.0.ssm_alpha.weight");
        const auto taken = [&](ggml_tensor * node, const spacemit_plugin_kernel * want) {
            const spacemit_op * got = spacemit_find_op(ops, node, buft);
            n_claims++;
            const bool ok = got != nullptr && got->n_steps == 1 && got->steps[0] == want;
            n_claims_ok += ok;
            if (!ok) {
                std::fprintf(stderr, "FAIL: contracts, %s not taken by %s\n", ggml_op_desc(node), want->name);
                g_failures++;
            }
        };
        const auto refused = [&](ggml_tensor * node, const char * what) {
            n_claims++;
            const bool ok = !ggml_backend_dev_supports_op(dev, node);
            n_claims_ok += ok;
            if (!ok) {
                std::fprintf(stderr, "FAIL: contracts, %s claimed but must be refused\n", what);
                g_failures++;
            }
        };
        taken(ggml_silu(ctx, f32), k_silu);
        taken(ggml_sigmoid(ctx, f32), k_sigmoid);
        taken(ggml_softplus(ctx, f32), k_softplus);
        taken(ggml_l2_norm(ctx, f32, 1e-6f), k_l2);
        taken(ggml_ssm_conv(ctx, sx, cw), k_conv);
        taken(ggml_concat(ctx, cat_a, cat_b, 0), k_concat);
        taken(ggml_cont(ctx, ggml_view_2d(ctx, f32, 64, 16, f32->nb[1], 0)), k_copy);
        taken(ggml_mul_mat(ctx, w32, f32), k_mm);
        // the IME matmul keeps its own contract: a quantized weight in a layout is not mul_mat.f32
        {
            const spacemit_op * ime = spacemit_find_op(ops, ggml_mul_mat(ctx, q4, x256), buft);
            n_claims++;
            const bool ok = ime != nullptr && ime->n_steps == 2 && ime->steps[0] == entry("mul_mat.quantize");
            n_claims_ok += ok;
            if (!ok) {
                std::fprintf(stderr, "FAIL: contracts, a Q4_0 MUL_MAT no longer takes the IME contract\n");
                g_failures++;
            }
        }
        // the unary ops take no non-contiguous row: ggml_unary_impl asserts it, so only the type can be refused
        refused(ggml_silu(ctx, f16), "SILU of F16");
        refused(ggml_sigmoid(ctx, f16), "SIGMOID of F16");
        refused(ggml_softplus(ctx, f16), "SOFTPLUS of F16");
        refused(ggml_l2_norm(ctx, f16, 1e-6f), "L2_NORM of F16");
        refused(ggml_l2_norm(ctx, ggml_transpose(ctx, f32), 1e-6f), "L2_NORM of a transposed source");
        refused(ggml_ssm_conv(ctx, sx, cw16), "SSM_CONV with an F16 kernel");
        refused(ggml_ssm_conv(ctx, ggml_view_3d(ctx, sxw, 7, 64, 1, sxw->nb[1], sxw->nb[2], 0), cw),
                "SSM_CONV with strided input rows");
        refused(ggml_concat(ctx, cat_b, cat_b, 1), "CONCAT along dimension 1");
        refused(ggml_concat(ctx, f16, f16, 0), "CONCAT of F16");
        refused(ggml_cont(ctx, ggml_transpose(ctx, f32)), "CONT of a transposed source");
        refused(ggml_cont(ctx, ggml_transpose(ctx, f16)), "CONT of F16");
        refused(ggml_mul_mat(ctx, w16, f32), "MUL_MAT with an F16 weight");
        refused(ggml_mul_mat(ctx, w3d, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 16, 2)),
                "MUL_MAT with a 3-D F32 weight");
        refused(ggml_mul_mat(ctx, ggml_view_2d(ctx, w32, 128, 16, w32->nb[1], 0), f32),
                "MUL_MAT with a view of an F32 weight (an F32 KV cache's form)");
        refused(ggml_mul_mat(ctx, w32, ggml_transpose(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 128))),
                "MUL_MAT with non-contiguous activations");
        refused(ggml_mul_mat(ctx, w32, f16), "MUL_MAT with F16 activations");
        ggml_free(ctx);
    }

    // the lane model of the RVV exp and activations against libm over every 509th float of each sign (all exponents,
    // subnormals to inf; 509 a prime, so the mantissas vary): the exp within 2 ulp of the exact value and the
    // activations within 4.5 (over all floats the worst were 1.954 and silu's 4.015), each within its bound above of
    // ggml-cpu's scalar formula, every non-finite result as libm's
    seq_lane_stat lane_exp, lane_silu, lane_sigmoid, lane_softplus;
    int64_t       n_lanes = 0;
    for (uint64_t u = 0; u <= 0x7f800000u; u += 509) {
        for (const uint32_t sign : { 0u, 0x80000000u }) {
            const float  x  = seq_lane_bits((uint32_t) u | sign);
            const double xd = x;
            // sigmoid and silu as ggml-cpu writes them overflow where exp(-x) does; softplus by its two branches
            const bool   neg_finite = std::isfinite(expf(-x));
            lane_exp.add(std::exp(xd), true, seq_lane_expf(x), expf(x));
            lane_silu.add(xd / (1.0 + std::exp(-xd)), neg_finite, seq_lane_silu(x), x / (1.0f + expf(-x)));
            lane_sigmoid.add(1.0 / (1.0 + std::exp(-xd)), neg_finite, seq_lane_sigmoid(x), 1.0f / (1.0f + expf(-x)));
            lane_softplus.add(xd > 0.0 ? xd + std::log1p(std::exp(-xd)) : std::log1p(std::exp(xd)), true,
                              seq_lane_softplus(x), x > 20.0f ? x : logf(1.0f + expf(x)));
            n_lanes++;
        }
    }
    const struct {
        const char *          name;
        const seq_lane_stat & st;
        double                max_ulp, max_cpu;
    } lanes[] = {
        { "exp", lane_exp, 2.0, HUGE_VAL },
        { "silu", lane_silu, 4.5, k_seq_silu.abs },
        { "sigmoid", lane_sigmoid, 4.5, k_seq_sigmoid.abs },
        { "softplus", lane_softplus, 4.5, k_seq_softplus.abs },
    };
    for (const auto & l : lanes) {
        if (l.st.max_ulp > l.max_ulp || l.st.max_cpu > l.max_cpu || l.st.n_nonfinite != 0) {
            std::fprintf(stderr,
                         "FAIL: seq lane model of the RVV %s: %.3f ulp from the exact value (bound %.2f), %.3e from "
                         "ggml-cpu's formula (bound %.0e), %lld non-finite results unlike libm's\n",
                         l.name, l.st.max_ulp, l.max_ulp, l.st.max_cpu, l.max_cpu, (long long) l.st.n_nonfinite);
            g_failures++;
        }
    }

    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    const auto worst_of = [&](const spacemit_plugin_kernel * k) {
        const auto it = worst.find(k);
        return it == worst.end() ? std::make_pair(-1.0, -1.0) : it->second;
    };
    const auto w_silu = worst_of(k_silu), w_sigmoid = worst_of(k_sigmoid), w_softplus = worst_of(k_softplus);
    const auto w_l2 = worst_of(k_l2), w_mm = worst_of(k_mm);
    std::printf("seq      the Gated DeltaNet layers' other ops, F32 in our buffer and in a CPU buffer, at Qwen3.5-4B's "
                "shapes (1 and 512 tokens) and edge cases (empty, one row, strided rows, 3 sequences, no token, "
                "broadcast, the 27B's 48 heads, values to +-40, near 0, subnormal, to +-88 and +-1000, L2_NORM to "
                "1e-20 and 1e18): %d/%d cases as the CPU on 8 tiles and directly on 1 and 8%s; SSM_CONV, CONCAT and "
                "CONT bit for bit in %d/%d (max NMSE %.2e); max NMSE and |difference| / max(1, |y|): SILU %.2e "
                "%.2e, SIGMOID %.2e %.2e, SOFTPLUS %.2e %.2e, L2_NORM %.2e %.2e (bounds 1e-12 and 1e-6, 5e-7, 1e-6, "
                "5e-7), MUL_MAT NMSE %.2e (bound %.0e, ggml-cpu's vector sums) and %.2e, N 1, 3, 4, 5 and 16 by K "
                "1, 7, 31, 32, 33, 65, 128 and 2560 and the split rule's shapes ([2048, 256] by 1 to 501 rows, "
                "[2048, 32] by 1 to 33, [64, 7], [2048, 1], a 3-D batch); the lane model of the RVV exp and "
                "activations over %lld floats: exp %.3f ulp, silu %.3f, sigmoid %.3f, softplus %.3f from the exact "
                "value, from ggml-cpu's formulas silu %.2e, sigmoid %.2e, softplus %.2e; the nine claims (the IME "
                "matmul still its own) and the refused forms (F16 everywhere, a transposed L2_NORM source, strided "
                "SSM_CONV input rows, an F16 kernel, CONCAT along dimension 1, a transposed CONT source, a 3-D or "
                "viewed F32 weight, non-contiguous or F16 activations): %d/%d\n",
                n_ok, n_cases, reference_label(), n_exact, n_exact_cases, worst_exact, w_silu.first, w_silu.second,
                w_sigmoid.first, w_sigmoid.second, w_softplus.first, w_softplus.second, w_l2.first, w_l2.second,
                w_mm.first, k_mul_mat_max_nmse, w_mm.second, (long long) n_lanes, lane_exp.max_ulp, lane_silu.max_ulp,
                lane_sigmoid.max_ulp, lane_softplus.max_ulp, lane_silu.max_cpu, lane_sigmoid.max_cpu,
                lane_softplus.max_cpu, n_claims_ok, n_claims);
    return 0;
}

//
// The F32 matmul's split and blocks (plan.md "The float matmul: all tiles for one token, blocks for a prompt").
// check_seq_ops compares every run with the CPU and not runs with each other, so the same-bytes comparisons live here.
// Each shape's node is built as seq_op_run builds its third copy (every tensor in a CPU buffer) and the kernel entry
// called directly, the output filled with 0xA5 bytes first so that a share no tile took stays visible: 1, 3 and 8 tiles
// must give the same bytes. Who computes what is read from the floats one tile alone changed, so the plugin keeps its
// interface, and checked against the share rule computed here
//

// one MUL_MAT node of a shape, every tensor in one buffer (the CPU's, or the device's for a backend run), its inputs
// random from seed as seq_op_run fills them
struct mm_f32_node {
    ggml_context *        ctx = nullptr, * ctx_g = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         y   = nullptr;
    ggml_cgraph *         g   = nullptr;

    bool init(ggml_backend_dev_t dev, const mm_f32_shape & s, bool device, uint32_t seed) {
        ggml_init_params p  = { ggml_tensor_overhead() * 8, nullptr, true };
        ggml_init_params gp = { ggml_graph_overhead(), nullptr, true };
        ctx                 = ggml_init(p);
        ctx_g               = ggml_init(gp);
        std::vector<ggml_tensor *> in;
        y   = mm_f32_build(s, ctx, in);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx, device ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type());
        if (buf == nullptr) {
            return false;
        }
        uint32_t next = seed;
        for (ggml_tensor * t : in) {
            const std::vector<float> v = random_values(ggml_nelements(t), next++, 1.0f);
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        }
        g = ggml_new_graph(ctx_g);
        ggml_build_forward_expand(g, y);
        return true;
    }

    ~mm_f32_node() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_free(ctx_g);
    }
};

static std::vector<uint8_t> mm_f32_out(const ggml_tensor * y) {
    std::vector<uint8_t> b(ggml_nbytes(y));
    ggml_backend_tensor_get(y, b.data(), 0, b.size());
    return b;
}

// FLAGOS_SPACEMIT_ROUTER=q8, the default (SpacemiT's fork's router route): the F32 MoE router weight quantized to Q8_0
// at load in the q8_0 32x32 layout; f32 the way back. Checked: the rule by the plugin's layout answers (unless f32,
// only the router's name, 2-D, rows % 32, K % 32; another type keeps its layout; a bad value warns once and keeps q8;
// the alloc size ggml_nbytes); the read-back is ggml's Q8_0 of the floats dequantized, bit for bit, whole and in part,
// small and split over the repack's ranges; MUL_MAT at 1, 4, 32 and 129 rows (without and with TCM), q8 and unset,
// claimed by the two 8-bit steps, the bytes of the same graph on a Q8_0 weight of ggml's Q8_0 of the floats, within the
// MUL_MAT bound of the CPU on the floats; f32, mul_mat.f32 within that bound; the core's guard: no contract but
// MUL_MAT's weight reads the router unless f32
static int check_router_q8(ggml_backend_dev_t dev) {
    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    REQUIRE(kernels != nullptr);
    const auto entry = [&](const char * name) -> const spacemit_plugin_kernel * {
        for (uint32_t k = 0; k < n_kernels; k++) {
            if (kernels[k].name != nullptr && std::strcmp(kernels[k].name, name) == 0) {
                return &kernels[k];
            }
        }
        return nullptr;
    };
    const spacemit_plugin_kernel * k_quantize = entry("mul_mat.quantize");
    const spacemit_plugin_kernel * k_gemm     = entry("mul_mat.gemm");
    const spacemit_plugin_kernel * k_f32      = entry("mul_mat.f32");
    REQUIRE(k_quantize != nullptr && k_gemm != nullptr && k_f32 != nullptr);
    spacemit_ops ops;
    std::string  detail;
    REQUIRE(spacemit_ops_bind(table, &ops, &detail));
    const ggml_backend_buffer_type_t buft   = ggml_backend_dev_buffer_type(dev);
    const char *                     router = "blk.0.ffn_gate_inp.weight";

    int        n_checks = 0, n_ok = 0;
    const auto expect   = [&](bool ok, const std::string & what) {
        n_checks++;
        n_ok += ok;
        if (!ok) {
            std::fprintf(stderr, "FAIL: router q8, %s\n", what.c_str());
            g_failures++;
        }
        return ok;
    };
    // ggml's Q8_0 of n rows of k floats, dequantized
    const auto q8_values = [](const std::vector<float> & v, int64_t k, int64_t n) {
        std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q8_0, k) * n);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, v.data(), q.data(), 0, n, k, nullptr);
        return dequantize(GGML_TYPE_Q8_0, q);
    };

    // A. the rule
    uint32_t router_layout = 0;
    for (const char * setting : { (const char *) nullptr, "", "f32", "q8", "x" }) {
        env_override     sw("FLAGOS_SPACEMIT_ROUTER", setting);
        log_capture      log;
        const bool       q8  = setting == nullptr || std::strcmp(setting, "f32") != 0;  // x keeps the default
        const std::string at = std::string("FLAGOS_SPACEMIT_ROUTER=") + (setting != nullptr ? setting : "(unset)");
        ggml_init_params p   = { ggml_tensor_overhead() * 16, nullptr, true };
        ggml_context *   ctx = ggml_init(p);
        REQUIRE(ctx != nullptr);
        const auto named = [&](ggml_tensor * t, const char * name) {
            ggml_set_name(t, name);
            return t;
        };
        ggml_tensor *  w      = named(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 32), router);
        const uint32_t l      = table->tensor_layout(w);
        const uint32_t l_again = table->tensor_layout(w);
        expect(q8 ? l != 0 && l_again == l : l == 0 && l_again == 0, at + ": the router's layout " + std::to_string(l));
        expect(ggml_backend_buft_get_alloc_size(buft, w) == ggml_nbytes(w), at + ": the alloc size is not ggml_nbytes");
        if (q8) {
            router_layout = l;
        }
        const struct {
            ggml_tensor * t;
            const char *  what;
        } plain[] = {
            { named(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 32), "blk.0.ffn_gate_inp_shexp.weight"), "shexp" },
            { named(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 32), "blk.0.ffn_gate_inp"), "a name without .weight" },
            { named(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 32), "blk.0.ssm_alpha.weight"), "ssm_alpha" },
            { named(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 48, 32), router), "[48, 32]" },
            { named(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 31), router), "[64, 31]" },
            { named(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 32, 2), router), "[64, 32, 2]" },
        };
        for (const auto & c : plain) {
            expect(table->tensor_layout(c.t) == 0, at + ": " + c.what + " has layout " +
                                                       std::to_string(table->tensor_layout(c.t)));
        }
        ggml_tensor *  w8         = named(ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 32), router);
        const uint32_t q8_0_named = table->tensor_layout(w8);
        const uint32_t q8_0_other = table->tensor_layout(ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 32));
        expect(q8_0_named != 0 && q8_0_named == q8_0_other && (!q8 || q8_0_named != l),
               at + ": a Q8_0 weight under the router's name has layout " + std::to_string(q8_0_named));
        const int warnings = count_lines_with(log.take(), "FLAGOS_SPACEMIT_ROUTER=");
        const int want     = setting != nullptr && std::strcmp(setting, "x") == 0 ? 1 : 0;
        expect(warnings == want, at + ": " + std::to_string(warnings) + " warnings, not " + std::to_string(want));
        ggml_free(ctx);
    }

    // B. the round trip: whole and partial reads, small (one thread) and split over the repack's ranges
    uint64_t big_ranges = 0;
    uint32_t seed       = 7100;
    for (const std::array<int64_t, 2> & s : std::vector<std::array<int64_t, 2>>{ { 256, 64 }, { 2048, 256 } }) {
        env_override     sw("FLAGOS_SPACEMIT_ROUTER", "q8");
        const std::string at  = "[" + std::to_string(s[0]) + ", " + std::to_string(s[1]) + "]";
        ggml_init_params p   = { ggml_tensor_overhead(), nullptr, true };
        ggml_context *   ctx = ggml_init(p);
        REQUIRE(ctx != nullptr);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s[0], s[1]);
        ggml_set_name(w, router);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        REQUIRE(buf != nullptr);
        const std::vector<float> v    = random_values(s[0] * s[1], seed++, 1.0f);
        const std::vector<float> want = q8_values(v, s[0], s[1]);
        std::vector<float>       read(v.size()), part(300);
        uint64_t                 threaded = 0, ranges = 0;
        g_repack_take(&threaded, &ranges);
        ggml_backend_tensor_set(w, v.data(), 0, ggml_nbytes(w));
        g_repack_take(&threaded, &ranges);
        const bool big = ggml_nbytes(w) >= ((size_t) 1 << 20);
        expect(table->tensor_layout(w) == router_layout && router_layout != 0, at + ": not in the router's layout");
        expect(big ? threaded == 1 && ranges >= 2 : threaded == 0,
               at + ": " + std::to_string(threaded) + " threaded repacks of " + std::to_string(ranges) + " ranges");
        if (big) {
            big_ranges = ranges;
        }
        ggml_backend_tensor_get(w, read.data(), 0, ggml_nbytes(w));
        expect(std::memcmp(read.data(), want.data(), ggml_nbytes(w)) == 0,
               at + ": the read-back is not ggml's Q8_0 of the floats");
        const size_t at_byte = ggml_nbytes(w) / 3 + 4 * 37;
        ggml_backend_tensor_get(w, part.data(), at_byte, part.size() * sizeof(float));
        expect(std::memcmp(part.data(), reinterpret_cast<const uint8_t *>(want.data()) + at_byte,
                           part.size() * sizeof(float)) == 0,
               at + ": a partial read is not that part of ggml's Q8_0 of the floats");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // C. MUL_MAT: y = W x with x in a CPU buffer and y in ours, through be (nullptr: the CPU backend, y in a CPU
    // buffer); got is the contract spacemit_find_op chose
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);  // a correctness reference: single-threaded (PROJECT.md item 28)
    const auto mm = [&](ggml_backend_t be, ggml_tensor * w, const std::vector<float> & xv, int64_t m,
                        std::vector<float> & out, const spacemit_op ** got) {
        ggml_init_params p  = { ggml_tensor_overhead() * 2 + ggml_graph_overhead(), nullptr, true };
        ggml_context *   ch = ggml_init(p);
        ggml_context *   cd = ggml_init(p);
        ggml_tensor *    x  = ggml_new_tensor_2d(ch, GGML_TYPE_F32, w->ne[0], m);
        ggml_tensor *    y  = ggml_mul_mat(cd, w, x);
        ggml_backend_buffer_t bh = ggml_backend_alloc_ctx_tensors_from_buft(ch, ggml_backend_cpu_buffer_type());
        ggml_backend_buffer_t bd =
            ggml_backend_alloc_ctx_tensors_from_buft(cd, be != nullptr ? buft : ggml_backend_cpu_buffer_type());
        bool ok = bh != nullptr && bd != nullptr;
        if (ok) {
            ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
            if (got != nullptr) {
                *got = spacemit_find_op(ops, y, buft);
            }
            ggml_cgraph * g = ggml_new_graph(cd);
            ggml_build_forward_expand(g, y);
            ok = (be == nullptr || ggml_backend_dev_supports_op(dev, y)) &&
                 ggml_backend_graph_compute(be != nullptr ? be : cpu, g) == GGML_STATUS_SUCCESS;
            out.assign((size_t) ggml_nelements(y), NAN);
            ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
        }
        ggml_backend_buffer_free(bh);
        ggml_backend_buffer_free(bd);
        ggml_free(ch);
        ggml_free(cd);
        return ok;
    };
    const int64_t            k = 256, n = 64;
    const std::vector<float> wv = random_values(k * n, seed++, 1.0f);
    double                   max_q8 = 0.0, max_f32 = 0.0;
    for (const char * setting : { "q8", (const char *) nullptr, "f32" }) {
        env_override      sw("FLAGOS_SPACEMIT_ROUTER", setting);
        const bool        q8 = setting == nullptr || std::strcmp(setting, "q8") == 0;
        const std::string at = setting != nullptr ? setting : "unset";
        ggml_init_params  p   = { ggml_tensor_overhead() * 3, nullptr, true };
        ggml_context *    ctx = ggml_init(p);
        ggml_context *    ctc = ggml_init(p);
        REQUIRE(ctx != nullptr && ctc != nullptr);
        ggml_tensor * w   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
        ggml_tensor * w8  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, n);
        ggml_tensor * w_c = ggml_new_tensor_2d(ctc, GGML_TYPE_F32, k, n);
        ggml_set_name(w, router);
        ggml_set_name(w_c, router);
        ggml_backend_buffer_t buf  = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        ggml_backend_buffer_t bufc = ggml_backend_alloc_ctx_tensors_from_buft(ctc, ggml_backend_cpu_buffer_type());
        REQUIRE(buf != nullptr && bufc != nullptr);
        std::vector<uint8_t> q8w(ggml_nbytes(w8));
        ggml_quantize_chunk(GGML_TYPE_Q8_0, wv.data(), q8w.data(), 0, n, k, nullptr);
        ggml_backend_tensor_set(w, wv.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(w8, q8w.data(), 0, q8w.size());
        ggml_backend_tensor_set(w_c, wv.data(), 0, ggml_nbytes(w_c));
        for (bool tcm : { false, true }) {
            ggml_backend_t be = tcm ? tcm_backend(dev, nullptr) : ggml_backend_dev_init(dev, nullptr);
            REQUIRE(be != nullptr);
            for (int64_t m : { (int64_t) 1, (int64_t) 4, (int64_t) 32, (int64_t) 129 }) {
                const std::string   what = at + (tcm ? ", TCM, " : ", ") + std::to_string(m) + " rows";
                const std::vector<float> xv = random_values(k * m, seed++, 2.0f);
                std::vector<float>  out, out8, ref;
                const spacemit_op * got = nullptr;
                const bool ran = mm(be, w, xv, m, out, &got) && mm(nullptr, w_c, xv, m, ref, nullptr);
                const bool two = got != nullptr && got->n_steps == 2 && got->steps[0] == k_quantize &&
                                 got->steps[1] == k_gemm;
                const bool one = got != nullptr && got->n_steps == 1 && got->steps[0] == k_f32;
                expect(ran, what + ": refused or failed");
                const std::string by = got == nullptr ? "nothing" : got->steps[0]->name;
                expect(q8 ? two : one, what + ": claimed by " + by);
                const double e = ran ? nmse(out.data(), ref.data(), (int64_t) out.size()) : HUGE_VAL;
                expect(e <= k_mul_mat_max_nmse, what + ": NMSE against the CPU " + std::to_string(e));
                (q8 ? max_q8 : max_f32) = std::max(q8 ? max_q8 : max_f32, e);
                if (q8) {
                    expect(mm(be, w8, xv, m, out8, nullptr) && out8.size() == out.size() &&
                               std::memcmp(out.data(), out8.data(), out.size() * sizeof(float)) == 0,
                           what + ": not the bytes of the Q8_0 weight");
                }
            }
            ggml_backend_free(be);
        }

        // D. the core's guard: under q8 (and unset) only MUL_MAT's weight reads the router; under f32, ADD takes it
        ggml_init_params pg = { ggml_tensor_overhead() * 8, nullptr, true };
        ggml_context *   cg = ggml_init(pg);
        REQUIRE(cg != nullptr);
        ggml_tensor * a  = ggml_new_tensor_2d(cg, GGML_TYPE_F32, k, n);
        ggml_tensor * x4 = ggml_new_tensor_2d(cg, GGML_TYPE_F32, k, 4);
        expect(ggml_backend_dev_supports_op(dev, ggml_add(cg, w, a)) == !q8, at + ": ADD of the router");
        if (q8) {
            expect(!ggml_backend_dev_supports_op(dev, ggml_mul(cg, a, w)), at + ": MUL by the router claimed");
            expect(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(cg, a, w)), at + ": MUL_MAT with the router as "
                                                                                   "src[1] claimed");
            expect(ggml_backend_dev_supports_op(dev, ggml_mul_mat(cg, w, x4)), at + ": MUL_MAT of the router refused");
        }
        ggml_free(cg);
        ggml_backend_buffer_free(buf);
        ggml_backend_buffer_free(bufc);
        ggml_free(ctx);
        ggml_free(ctc);
    }
    ggml_backend_free(cpu);

    std::printf("router   FLAGOS_SPACEMIT_ROUTER (unset, empty, f32, q8, x): the F32 router weight in a layout (%u) "
                "unless f32, not the shexp gate, other names, [48, 32], [64, 31] or 3-D; Q8_0 its own; x warned once, "
                "q8 kept; alloc ggml_nbytes. q8 read-back ggml's Q8_0 of the floats bit for bit, whole and in part, "
                "[256, 64] and [2048, 256] (%llu ranges). MUL_MAT [256, 64] by 1, 4, 32 and 129 rows, without and with "
                "TCM: q8 and unset by mul_mat.quantize and mul_mat.gemm, the bytes of the Q8_0 weight, max NMSE vs the "
                "CPU's F32 %.2e; f32 by mul_mat.f32, max NMSE %.2e (bound %.0e). Guard: ADD, MUL and MUL_MAT src[1] of "
                "the q8 router refused, MUL_MAT src[0] taken, ADD taken under f32: %d/%d\n",
                router_layout, (unsigned long long) big_ranges, max_q8, max_f32, k_mul_mat_max_nmse, n_ok, n_checks);
    return 0;
}

// test-only (plugin/flagos-spacemit-f16.cpp): the fp16 weights the reference dequantization stages for one slice of K
// and one 32-column tile, [32 rows][K / 8]. Declared here: the core's header is left alone
typedef void (*spacemit_mul_mat_f16_stage_fn)(const ggml_tensor * w, int64_t slice, int64_t col_tile, uint16_t * out);
// test-only (plugin/flagos-spacemit-f16.cpp): the route the two steps take for a node on nth tiles with tcm_size bytes
// of TCM, under FLAGOS_SPACEMIT_F16_ROUTE as parsed now: out[0] slices, out[1] weight slots, out[2] row passes, out[3]
// the stage in TCM (all but out[0] 0 on the plain route)
typedef void (*spacemit_mul_mat_f16_route_fn)(const ggml_tensor * node, uint32_t nth, size_t tcm_size, int64_t * out);

static uint32_t device_tiles(ggml_backend_dev_t dev);

static float f16_round(float v) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
}

// the f16 path's arithmetic written again, for m rows of x by the n rows of w (both [*, k] floats), a16 and w16 fp16 of
// the values, K cut in `slices` (8 on the plain route, 4 on the pairs route). out: per slice of k / slices a float from
// +0.0f, per group of 8 in order acc + ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)), stored as fp16; the slices
// added in order in float from +0.0f: the references bit for bit.
// [lo, hi] bounds the kernels, whose vfwmadot adds a group (the accumulator and 8 exact products) in its own order: at
// most 8 additions a group, each within one float ulp (2^-23 relative: to nearest, truncated or wider), so a slice's
// float sum is within eps = 8 2^-23 sum_g (|P_(g-1)| + sum_j |p_gj|) (1 + 2^-10) of its exact sum e (P_g the exact
// prefix, in double, with 2^-52 ks sum |p| more for that). fp16 rounding and the float sum of the slices are monotonic:
// the output lies between the sums of fp16(e - eps) and of fp16(e + eps). Mostly both ends round to the same fp16 value
// and the output must be `out` exactly; a slice may round either way only when its sum lies within eps (a few % of an
// fp16 step) of a midpoint. So it catches a wrong product, layout or group, and an error that moves a partial over a
// midpoint (a weight or activation one fp16 ulp off does so in some rows); not an error that moves none, nor an order
// inside a group. The exact cases in check_mul_mat_f16 cover the fp16 values and the layouts bit for bit
struct f16_model {
    std::vector<float> out, lo, hi;

    void init(const std::vector<float> & x, const std::vector<float> & w, int64_t k, int64_t n, int64_t m,
              int64_t slices) {
        const int64_t      ks = k / slices;
        std::vector<float> a16(x.size()), w16(w.size());
        for (size_t i = 0; i < x.size(); i++) {
            a16[i] = f16_round(x[i]);
        }
        for (size_t i = 0; i < w.size(); i++) {
            w16[i] = f16_round(w[i]);
        }
        out.assign((size_t) (m * n), 0.0f);
        lo.assign((size_t) (m * n), 0.0f);
        hi.assign((size_t) (m * n), 0.0f);
        for (int64_t r = 0; r < m; r++) {
            for (int64_t c = 0; c < n; c++) {
                float o = 0.0f, o_lo = 0.0f, o_hi = 0.0f;
                for (int64_t s = 0; s < slices; s++) {
                    float  acc = 0.0f;
                    double e = 0.0, nodes = 0.0, abs_sum = 0.0;
                    for (int64_t g = s * ks; g < (s + 1) * ks; g += 8) {
                        float  p[8];
                        double group = 0.0;
                        for (int j = 0; j < 8; j++) {
                            p[j] = a16[r * k + g + j] * w16[c * k + g + j];
                            group += std::fabs((double) p[j]);
                        }
                        nodes += std::fabs(e) + group;
                        abs_sum += group;
                        for (int j = 0; j < 8; j++) {
                            e += (double) p[j];
                        }
                        acc = acc + (((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7])));
                    }
                    const double eps = std::ldexp(8.0 * nodes, -23) * (1.0 + std::ldexp(1.0, -10)) +
                                       std::ldexp((double) ks * abs_sum, -52);
                    o    = o + f16_round(acc);
                    o_lo = o_lo + f16_round((float) (e - eps));
                    o_hi = o_hi + f16_round((float) (e + eps));
                }
                out[r * n + c] = o;
                lo[r * n + c]  = o_lo;
                hi[r * n + c]  = o_hi;
            }
        }
    }
};

// the outputs outside [lo, hi] (a NaN is outside); adds the outputs bit-equal to the model's to `equal` and those whose
// bound is one value to `pinned`, and keeps the largest |out - model| / max(model - lo, hi - model) in `worst`
static int64_t f16_outside(const std::vector<float> & out, const f16_model & model, int64_t & equal, int64_t & pinned,
                           double & worst) {
    int64_t bad = 0;
    for (size_t i = 0; i < out.size() && i < model.out.size(); i++) {
        bad += !(out[i] >= model.lo[i] && out[i] <= model.hi[i]);
        equal += std::memcmp(&out[i], &model.out[i], sizeof(float)) == 0;
        pinned += model.lo[i] == model.hi[i];
        const double e = std::fabs((double) out[i] - (double) model.out[i]);
        const double b = std::max((double) model.out[i] - model.lo[i], (double) model.hi[i] - model.out[i]);
        worst          = std::max(worst, b > 0.0 ? e / b : (e > 0.0 ? HUGE_VAL : 0.0));
    }
    return bad + (int64_t) (out.size() != model.out.size());
}

// GGUF Q4_0 bytes of n rows of k values: d(row, block of 32) and the nibble q(row, k)
static std::vector<uint8_t> q4_0_raw(int64_t k, int64_t n, const std::function<float(int64_t, int64_t)> & d,
                                     const std::function<int(int64_t, int64_t)> & q) {
    static_assert(sizeof(block_q4_0) == 18, "Q4_0 block");
    std::vector<block_q4_0> blocks((size_t) (n * k / 32));
    for (int64_t r = 0; r < n; r++) {
        for (int64_t b = 0; b < k / 32; b++) {
            block_q4_0 & blk = blocks[r * (k / 32) + b];
            blk.d            = ggml_fp32_to_fp16(d(r, b));
            for (int64_t j = 0; j < 16; j++) {
                blk.qs[j] = (uint8_t) (q(r, 32 * b + j) | (q(r, 32 * b + j + 16) << 4));
            }
        }
    }
    std::vector<uint8_t> bytes(blocks.size() * sizeof(block_q4_0));
    std::memcpy(bytes.data(), blocks.data(), bytes.size());
    return bytes;
}

// FLAGOS_SPACEMIT_DENSE_Q4_0 (plan.md "The dense Q4_0 matmuls in FP16: the port"): MUL_MAT on 2-D Q4_0 weights in fp16
// arithmetic by mul_mat.f16.products and mul_mat.f16.sum, the IME2 kernels on the K3 (unless GGML_FLAGOS_KERNEL_VARIANTS
// =0), the references elsewhere. Checked: (A) the claim by the contract spacemit_find_op binds, through the switch's
// settings and thresholds, never a Q8_0 or 3-D weight, a weight over the 2^24 cap or K over 32768; (B) the reference's
// staged weights are fp16 of ggml's dequantization bit for bit; (C) the outputs against a model of the arithmetic
// written here (the references bit for bit, the kernels within its bound) and against the CPU, the same bytes with a
// TCM stand-in that forces several row passes and on threads, and the reference steps called directly on 1 and 3
// tiles the model's bytes; (C2) exact cases, bit for bit in both modes, and the fp16 subnormal probes; (D) an overflow
// fails the graph with one line, an infinite input does not; (E) unset is f16, int8 the integer path's steps; (G)
// QUANTIZE=shared; (F) FLAGOS_SPACEMIT_F16_STATS. Kernels run only through a launch: the direct calls take the
// reference steps (spacemit_mul_mat_f16_*_reference), which never run IME code, so they are safe on the test's thread
// (an X100 core on the K3)
static int check_mul_mat_f16(ggml_backend_dev_t dev, bool host) {
    const auto                 t_start = std::chrono::steady_clock::now();
    const bool                 kernels = !host && *reference_label() == '\0';
    const char *               mode    = kernels ? "kernels" : "references";
    const spacemit_plugin_v1 * table   = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels_t = table->kernels(&n_kernels);
    REQUIRE(kernels_t != nullptr);
    const auto entry = [&](const char * name) -> const spacemit_plugin_kernel * {
        for (uint32_t k = 0; k < n_kernels; k++) {
            if (kernels_t[k].name != nullptr && std::strcmp(kernels_t[k].name, name) == 0) {
                return &kernels_t[k];
            }
        }
        return nullptr;
    };
    const spacemit_plugin_kernel * k_quantize = entry("mul_mat.quantize");
    const spacemit_plugin_kernel * k_gemm     = entry("mul_mat.gemm");
    const spacemit_plugin_kernel * k_products = entry("mul_mat.f16.products");
    const spacemit_plugin_kernel * k_sum      = entry("mul_mat.f16.sum");
    const spacemit_plugin_kernel * k_id       = entry("mul_mat_id.quantize");
    REQUIRE(k_quantize != nullptr && k_gemm != nullptr && k_products != nullptr && k_sum != nullptr && k_id != nullptr);
    REQUIRE(k_products->workspace_size != nullptr && k_products->workspace_size == k_sum->workspace_size);
    const auto stage_weights =
        reinterpret_cast<spacemit_mul_mat_f16_stage_fn>(table->get_proc_address("spacemit_mul_mat_f16_stage_weights"));
    const auto products_ref =
        reinterpret_cast<spacemit_step_fn>(table->get_proc_address("spacemit_mul_mat_f16_products_reference"));
    const auto sum_ref = reinterpret_cast<spacemit_step_fn>(table->get_proc_address("spacemit_mul_mat_f16_sum_reference"));
    const auto route_of =
        reinterpret_cast<spacemit_mul_mat_f16_route_fn>(table->get_proc_address("spacemit_mul_mat_f16_route"));
    REQUIRE(stage_weights != nullptr && products_ref != nullptr && sum_ref != nullptr && route_of != nullptr);
    spacemit_ops ops;
    std::string  detail;
    REQUIRE(spacemit_ops_bind(table, &ops, &detail));
    const ggml_backend_buffer_type_t buft  = ggml_backend_dev_buffer_type(dev);
    const char *                     dense = "FLAGOS_SPACEMIT_DENSE_Q4_0";
    const char *                     route = "FLAGOS_SPACEMIT_F16_ROUTE";
    const uint32_t                   dev_tiles = device_tiles(dev);
    // the slices the steps take for node on the device's tiles (TCM as the K3's or none: the same stage size)
    const auto slices_of = [&](const ggml_tensor * node) {
        int64_t r[4] = {};
        route_of(node, dev_tiles, 0, r);
        return r[0];
    };
    const int64_t f16_default = 32;  // the threshold under unset, empty, f16 or a bad value

    int        n_checks = 0, n_ok = 0;
    const auto expect   = [&](bool ok, const std::string & what) {
        n_checks++;
        n_ok += ok;
        if (!ok) {
            std::fprintf(stderr, "FAIL: mul_mat f16, %s\n", what.c_str());
            g_failures++;
        }
        return ok;
    };
    const auto is_f16 = [&](const spacemit_op * op) {
        return op != nullptr && op->n_steps == 2 && op->steps[0] == k_products && op->steps[1] == k_sum;
    };
    const auto is_int8 = [&](const spacemit_op * op) {
        return op != nullptr && op->n_steps == 2 && op->steps[0] == k_quantize && op->steps[1] == k_gemm;
    };
    const auto by = [](const spacemit_op * op) { return std::string(op == nullptr ? "nothing" : op->steps[0]->name); };

    // A. the claim: weights in our buffer (a zero-size one of our type, as llama.cpp places weights at load), the
    // activations unallocated. A Q4_0 weight whose K is not a multiple of 256 has no q4_0 32x256 layout, so neither
    // MUL_MAT contract takes it (check_mul_mat_claims, [2880, 64]); not repeated here
    {
        ggml_init_params p   = { ggml_tensor_overhead() * 512, nullptr, true };
        ggml_context *   ctx = ggml_init(p);
        REQUIRE(ctx != nullptr);
        std::vector<ggml_tensor *> placed;
        const auto weight = [&](ggml_type type, int64_t k, int64_t n, int64_t ne2) {
            ggml_tensor * w = ne2 > 1 ? ggml_new_tensor_3d(ctx, type, k, n, ne2) : ggml_new_tensor_2d(ctx, type, k, n);
            w->buffer       = ggml_backend_buft_alloc_buffer(buft, 0);
            placed.push_back(w);
            return w;
        };
        const auto rows = [&](ggml_tensor * w, int64_t m) {
            return spacemit_find_op(ops, ggml_mul_mat(ctx, w, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w->ne[0], m)),
                                    buft);
        };
        ggml_tensor * w = weight(GGML_TYPE_Q4_0, 512, 64, 1);
        const std::vector<std::pair<int64_t, bool>> by_default = { { f16_default - 1, false }, { f16_default, true },
                                                                   { 200, true } };
        const struct {
            const char *                          setting;
            std::vector<std::pair<int64_t, bool>> rows;  // (M, taken by f16)
            bool                                  warns;
        } cases[] = {
            { nullptr,    by_default,                                       false },
            { "",         by_default,                                       false },
            { "int8",     { { 16, false }, { 32, false }, { 200, false } }, false },
            { "f16",      by_default,                                       false },
            { "f16:16",   { { 15, false }, { 16, true } },                  false },
            { "f16:128",  { { 127, false }, { 128, true } },                false },
            { "x",        by_default,                                       true  },
            { "f16:0",    by_default,                                       true  },
            { "f16:abc",  by_default,                                       true  },
            { "f16:5000", by_default,                                       true  },
        };
        for (const auto & c : cases) {
            env_override      sw(dense, c.setting);
            log_capture       log;
            const std::string at = std::string(dense) + "=" + (c.setting != nullptr ? c.setting : "(unset)");
            for (const auto & r : c.rows) {
                const spacemit_op * op = rows(w, r.first);
                expect(r.second ? is_f16(op) : is_int8(op),
                       at + ", [512, 64] by " + std::to_string(r.first) + " rows: claimed by " + by(op));
            }
            const int warnings = count_lines_with(log.take(), "FLAGOS_SPACEMIT_DENSE_Q4_0=");
            expect(warnings == (c.warns ? 1 : 0), at + ": " + std::to_string(warnings) + " warnings");
        }

        // never f16 even when on
        env_override  sw(dense, "f16");
        ggml_tensor * w8    = weight(GGML_TYPE_Q8_0, 512, 64, 1);
        ggml_tensor * w3    = weight(GGML_TYPE_Q4_0, 512, 64, 4);
        ggml_tensor * w_cap = weight(GGML_TYPE_Q4_0, 512, 80672, 1);  // GGML_PAD(200, 16) * 80672 = 2^24 + 2560
        ggml_tensor * w_in  = weight(GGML_TYPE_Q4_0, 512, 80640, 1);  // 2^24 - 4096
        ggml_tensor * w_kx  = weight(GGML_TYPE_Q4_0, 33024, 32, 1);   // K over 32768
        ggml_tensor * w_k   = weight(GGML_TYPE_Q4_0, 32768, 32, 1);
        expect(is_int8(rows(w8, 200)), "f16, Q8_0 [512, 64] by 200 rows: claimed by " + by(rows(w8, 200)));
        ggml_tensor *       x3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, 2, 200);
        ggml_tensor *       id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 200);
        const spacemit_op * mi = spacemit_find_op(ops, ggml_mul_mat_id(ctx, w3, x3, id), buft);
        expect(mi != nullptr && mi->steps[0] == k_id, "f16, MUL_MAT_ID of 3-D Q4_0: claimed by " + by(mi));
        const spacemit_op * m3 =
            spacemit_find_op(ops, ggml_mul_mat(ctx, w3, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, 200, 4)), buft);
        expect(m3 == nullptr, "f16, MUL_MAT of a 3-D Q4_0 weight: claimed by " + by(m3));
        expect(is_int8(rows(w_cap, 200)), "f16, [512, 80672] by 200 rows (over 2^24): " + by(rows(w_cap, 200)));
        expect(is_f16(rows(w_in, 200)), "f16, [512, 80640] by 200 rows (under 2^24): " + by(rows(w_in, 200)));
        expect(is_int8(rows(w_kx, 64)), "f16, [33024, 32] by 64 rows: claimed by " + by(rows(w_kx, 64)));
        expect(is_f16(rows(w_k, 64)), "f16, [32768, 32] by 64 rows: claimed by " + by(rows(w_k, 64)));
        const spacemit_op * b3 =
            spacemit_find_op(ops, ggml_mul_mat(ctx, w, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, 8, 25)), buft);
        const spacemit_op * b2 =
            spacemit_find_op(ops, ggml_mul_mat(ctx, w, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, 5, 6)), buft);
        expect(is_f16(b3) && is_int8(b2), "f16, [512, 8, 25] and [512, 5, 6] activations: " + by(b3) + ", " + by(b2));
        for (ggml_tensor * t : placed) {
            ggml_backend_buffer_free(t->buffer);
            t->buffer = nullptr;
        }
        ggml_free(ctx);
    }

    // B. the staged weights, every (slice, 32-column tile), against fp16 of ggml's dequantization of the GGUF bytes
    uint32_t seed     = 7300;
    int      n_staged = 0;
    for (const std::array<int64_t, 2> & s : std::vector<std::array<int64_t, 2>>{ { 2048, 96 }, { 256, 32 } }) {
        const int64_t              k = s[0], n = s[1], ks = k / 8;
        const std::vector<uint8_t> q = quantize_weights(GGML_TYPE_Q4_0, k, n, seed++);
        mm_weight                  w;
        REQUIRE(w.init_with(dev, GGML_TYPE_Q4_0, k, n, q));
        const std::vector<float> deq = dequantize(GGML_TYPE_Q4_0, q);
        std::vector<uint16_t>    got((size_t) (32 * ks));
        for (int64_t sl = 0; sl < 8; sl++) {
            for (int64_t t = 0; t < n / 32; t++) {
                stage_weights(w.w_dev, sl, t, got.data());
                int64_t bad = 0;
                for (int64_t r = 0; r < 32; r++) {
                    for (int64_t i = 0; i < ks; i++) {
                        bad += got[r * ks + i] != ggml_fp32_to_fp16(deq[(t * 32 + r) * k + sl * ks + i]);
                    }
                }
                n_staged++;
                expect(bad == 0, "[" + std::to_string(k) + ", " + std::to_string(n) + "] slice " + std::to_string(sl) +
                                     " tile " + std::to_string(t) + ": " + std::to_string(bad) +
                                     " staged weights not fp16 of ggml's dequantization");
            }
        }
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);  // a correctness reference: single-threaded (PROJECT.md item 28)

    // C. the arithmetic on the plain route. Each shape's x, its CPU result and its model once; then the default
    // backend, a TCM stand-in of 98304 bytes (at K 2048 160 rows a pass, 4 passes at 501 rows; at K 4096 64 rows), the
    // same on threads, and the reference steps called directly on 1 and 3 tiles with that stage, there under
    // FLAGOS_SPACEMIT_F16_ROUTE=pairs, which needs 8 tiles. With spine-runtime the stand-in is ignored and the board's
    // TCM sets the passes (two at K 4096 and 700 rows); the direct calls still take the small stage
    const char * small_tcm = "98304";
    double       worst = 0.0, max_nmse = 0.0;
    int64_t      n_equal = 0, n_pinned = 0, n_outputs = 0;
    {
        env_override sw(dense, "f16:16");
        env_override rt(route, "plain");
        struct c_case {
            std::unique_ptr<mm_weight> w;
            std::unique_ptr<mm_run>    run;
            int64_t                    k, n, m;
            std::vector<float>         ref;
            f16_model                  model;
        };
        std::vector<c_case> cases;
        const struct {
            int64_t              k, n;
            std::vector<int64_t> ms;
        // [256, 320]: 10 column tiles, more than the 8 tiles, for the rotation of the column order
        } shapes[] = { { 2048, 96, { 16, 17, 64, 129, 501 } }, { 4096, 64, { 64, 700 } }, { 256, 32, { 64 } },
                       { 256, 320, { 40 } } };
        for (const auto & s : shapes) {
            for (int64_t m : s.ms) {
                c_case c;
                c.k = s.k;
                c.n = s.n;
                c.m = m;
                c.w = std::make_unique<mm_weight>();
                const std::vector<uint8_t> q = quantize_weights(GGML_TYPE_Q4_0, s.k, s.n, seed++);
                REQUIRE(c.w->init_with(dev, GGML_TYPE_Q4_0, s.k, s.n, q));
                c.run = std::make_unique<mm_run>();
                REQUIRE(c.run->init(dev, *c.w, m, seed++));
                std::vector<float> x((size_t) (s.k * m));
                ggml_backend_tensor_get(c.run->x, x.data(), 0, ggml_nbytes(c.run->x));
                REQUIRE(ggml_backend_graph_compute(cpu, c.run->g_cpu) == GGML_STATUS_SUCCESS);
                c.ref.resize((size_t) (s.n * m));
                ggml_backend_tensor_get(c.run->y_ref, c.ref.data(), 0, ggml_nbytes(c.run->y_ref));
                c.model.init(x, dequantize(GGML_TYPE_Q4_0, q), s.k, s.n, m, 8);
                cases.push_back(std::move(c));
            }
        }
        const auto what = [](const c_case & c) {
            return "K " + std::to_string(c.k) + " N " + std::to_string(c.n) + " M " + std::to_string(c.m);
        };
        const char * variants[] = { "default", "TCM 98304", "TCM 98304 on threads", "1 tile direct", "3 tiles direct" };
        std::vector<std::vector<std::vector<float>>> outs(cases.size(), std::vector<std::vector<float>>(5));
        for (int v = 0; v < 3; v++) {
            std::unique_ptr<env_override> tcm, threads;
            if (v > 0) {
                tcm     = std::make_unique<env_override>("FLAGOS_SPACEMIT_TEST_TCM", small_tcm);
                threads = std::make_unique<env_override>("FLAGOS_SPACEMIT_TEST_THREADS", v == 2 ? "1" : "0");
            }
            g_executors_take();
            ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(be != nullptr);
            expect(g_executors_take() == 1, std::string(variants[v]) + ": the backend shares an older executor");
            for (size_t i = 0; i < cases.size(); i++) {
                c_case &            c  = cases[i];
                const spacemit_op * op = spacemit_find_op(ops, c.run->y, buft);
                expect(is_f16(op), what(c) + ": claimed by " + by(op));
                const bool ran = ggml_backend_graph_compute(be, c.run->g_dev) == GGML_STATUS_SUCCESS;
                expect(ran, what(c) + ", " + variants[v] + ": failed");
                outs[i][v].assign((size_t) (c.n * c.m), NAN);
                if (ran) {
                    ggml_backend_tensor_get(c.run->y, outs[i][v].data(), 0, ggml_nbytes(c.run->y));
                }
            }
            ggml_backend_free(be);
        }
        std::vector<uint8_t> stage(std::stoul(small_tcm));
        env_override         direct_pairs(route, "pairs");  // on 1 and 3 tiles the plain route all the same
        for (int v = 3; v < 5; v++) {
            const uint32_t nth = v == 3 ? 1 : 3;
            for (size_t i = 0; i < cases.size(); i++) {
                c_case &           c = cases[i];
                std::vector<float> nan((size_t) (c.n * c.m), NAN);
                std::vector<float> ws(k_products->workspace_size(c.run->y) / sizeof(float) + 1);
                ggml_backend_tensor_set(c.run->y, nan.data(), 0, ggml_nbytes(c.run->y));
                bool ok = true;
                for (spacemit_step_fn step : { products_ref, sum_ref }) {
                    for (uint32_t ith = 0; ith < nth; ith++) {
                        const spacemit_tile tile = { ith, nth, stage.data(), stage.size(), ws.data(),
                                                     ws.size() * sizeof(float), nullptr };
                        ok = step(&tile, c.run->y) && ok;
                    }
                }
                expect(ok, what(c) + ", " + variants[v] + ": a step failed");
                outs[i][v].assign((size_t) (c.n * c.m), NAN);
                ggml_backend_tensor_get(c.run->y, outs[i][v].data(), 0, ggml_nbytes(c.run->y));
            }
        }
        for (size_t i = 0; i < cases.size(); i++) {
            const c_case &    c     = cases[i];
            const std::string at    = what(c);
            const size_t      bytes = outs[i][0].size() * sizeof(float);
            for (int v = 1; v < 3; v++) {
                expect(std::memcmp(outs[i][v].data(), outs[i][0].data(), bytes) == 0,
                       at + ", " + variants[v] + ": not the bytes of the default backend");
            }
            for (int v = 3; v < 5; v++) {
                expect(std::memcmp(outs[i][v].data(), c.model.out.data(), bytes) == 0,
                       at + ", " + variants[v] + ": not the model's bytes");
            }
            int64_t       equal = 0;
            const int64_t bad   = f16_outside(outs[i][0], c.model, equal, n_pinned, worst);
            expect(bad == 0, at + ": " + std::to_string(bad) + " outputs outside the model's bound (" + mode + ")");
            expect(kernels || equal == (int64_t) outs[i][0].size(),
                   at + ": " + std::to_string(equal) + " of " + std::to_string(outs[i][0].size()) +
                       " outputs the model's bytes (references)");
            n_equal += equal;
            n_outputs += (int64_t) outs[i][0].size();
            const double e = nmse(outs[i][0].data(), c.ref.data(), (int64_t) c.ref.size());
            max_nmse       = std::max(max_nmse, e);
            expect(e <= k_mul_mat_max_nmse, at + ": NMSE against the CPU " + std::to_string(e));
        }
    }

    // P. the pairs route (FLAGOS_SPACEMIT_F16_ROUTE=pairs): with 8 tiles 4 slices of K / 4, tiles 2 g and 2 g + 1 on
    // slice g. First the route the plugin takes (spacemit_mul_mat_f16_route) against the fork's rules worked out for a
    // stage of 384 KiB (the scratch, the K3's TCM and the stand-in below): m_fit for 1 and 2 slots 352 and 320 at K
    // 2048, 160 and 128 at K 4096, 3040 and 3008 at K 256; 2 slots when there are 2 chunks or more and they cost no
    // pass more; a pass 2 m_fit rows; K 8192 (64 rows) keeps the plain route's 8 slices; 1 or 3 tiles the plain route.
    // Then each shape against its model (4 slices; 8 at K 8192) and the CPU: tiles in turn, where a tile dequantizes
    // its partner's weight tile itself (order B, the waits counted, so unequal counts fail the launch), and on threads,
    // where it reads its partner's stage (order A with 2 slots, B with 1), each without TCM (the stage the scratch) and
    // with a TCM stand-in of 393216 bytes (the stage the TCM): the same bytes in all four
    int64_t p_equal = 0, p_pinned = 0, p_outputs = 0;
    double  p_worst = 0.0;
    {
        env_override sw(dense, "f16:16");
        env_override rt(route, "pairs");
        struct p_case {
            std::unique_ptr<mm_weight> w;
            std::unique_ptr<mm_run>    run;
            int64_t                    k, n, m, slices;
            std::vector<float>         ref;
            f16_model                  model;
        };
        std::vector<p_case> cases;
        // per row count: the slices, slots and passes expected
        const struct {
            int64_t                             k, n;
            std::vector<std::array<int64_t, 4>> ms;
        } shapes[] = {
            // 3 column tiles: 2 chunks, the second with one tile; at 16 rows the odd tile has none
            { 2048, 96, { { 16, 4, 2, 1 }, { 64, 4, 2, 1 }, { 129, 4, 2, 1 }, { 501, 4, 2, 1 } } },
            // 1 chunk: 1 slot; 3 and 4 passes
            { 4096, 64, { { 64, 4, 1, 1 }, { 700, 4, 1, 3 }, { 1100, 4, 1, 4 } } },
            // 2 chunks: 2 slots and 3 passes at 700; at 1100 2 slots would take 5 passes, 1 slot 4
            { 4096, 128, { { 64, 4, 2, 1 }, { 700, 4, 2, 3 }, { 1100, 4, 1, 4 } } },
            // 10 column tiles: 5 chunks
            { 256, 320, { { 40, 4, 2, 1 } } },
            { 8192, 32, { { 64, 8, 0, 0 } } },
        };
        for (const auto & s : shapes) {
            for (const std::array<int64_t, 4> & e : s.ms) {
                p_case c;
                c.k      = s.k;
                c.n      = s.n;
                c.m      = e[0];
                c.slices = e[1];
                c.w      = std::make_unique<mm_weight>();
                const std::vector<uint8_t> q = quantize_weights(GGML_TYPE_Q4_0, s.k, s.n, seed++);
                REQUIRE(c.w->init_with(dev, GGML_TYPE_Q4_0, s.k, s.n, q));
                c.run = std::make_unique<mm_run>();
                REQUIRE(c.run->init(dev, *c.w, c.m, seed++));
                const std::string at = "pairs, K " + std::to_string(s.k) + " N " + std::to_string(s.n) + " M " +
                                       std::to_string(c.m);
                for (size_t tcm : { (size_t) 0, (size_t) 393216 }) {
                    int64_t r[4] = {};
                    route_of(c.run->y, 8, tcm, r);
                    expect(r[0] == e[1] && r[1] == e[2] && r[2] == e[3] && r[3] == (int64_t) (tcm > 0 && e[1] == 4),
                           at + ", TCM " + std::to_string(tcm) + ": slices " + std::to_string(r[0]) + ", slots " +
                               std::to_string(r[1]) + ", passes " + std::to_string(r[2]) + ", in TCM " +
                               std::to_string(r[3]) + " (expected " + std::to_string(e[1]) + ", " +
                               std::to_string(e[2]) + ", " + std::to_string(e[3]) + ")");
                }
                for (uint32_t few : { 1u, 3u }) {
                    int64_t r[4] = {};
                    route_of(c.run->y, few, 0, r);
                    expect(r[0] == 8 && r[1] == 0, at + ": " + std::to_string(r[0]) + " slices on " +
                                                       std::to_string(few) + " tiles");
                }
                std::vector<float> x((size_t) (s.k * c.m));
                ggml_backend_tensor_get(c.run->x, x.data(), 0, ggml_nbytes(c.run->x));
                REQUIRE(ggml_backend_graph_compute(cpu, c.run->g_cpu) == GGML_STATUS_SUCCESS);
                c.ref.resize((size_t) (s.n * c.m));
                ggml_backend_tensor_get(c.run->y_ref, c.ref.data(), 0, ggml_nbytes(c.run->y_ref));
                c.model.init(x, dequantize(GGML_TYPE_Q4_0, q), s.k, s.n, c.m, c.slices);
                cases.push_back(std::move(c));
            }
        }
        const char * variants[4] = { "in turn", "in turn with TCM", "on threads", "on threads with TCM" };
        std::vector<std::vector<std::vector<float>>> outs(cases.size(), std::vector<std::vector<float>>(4));
        for (int v = 0; v < 4; v++) {
            env_override tcm("FLAGOS_SPACEMIT_TEST_TCM", v % 2 == 1 ? "393216" : nullptr);
            env_override threads("FLAGOS_SPACEMIT_TEST_THREADS", v >= 2 ? "1" : "0");
            g_executors_take();
            ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(be != nullptr);
            expect(g_executors_take() == 1,
                   std::string("pairs ") + variants[v] + ": the backend shares an older executor");
            for (size_t i = 0; i < cases.size(); i++) {
                p_case &            c  = cases[i];
                const spacemit_op * op = spacemit_find_op(ops, c.run->y, buft);
                const std::vector<float> nan((size_t) (c.n * c.m), NAN);
                ggml_backend_tensor_set(c.run->y, nan.data(), 0, ggml_nbytes(c.run->y));
                const bool ran = is_f16(op) && ggml_backend_graph_compute(be, c.run->g_dev) == GGML_STATUS_SUCCESS;
                expect(ran, "pairs, K " + std::to_string(c.k) + " N " + std::to_string(c.n) + " M " +
                                std::to_string(c.m) + ", " + variants[v] + ": claimed by " + by(op) + ", " +
                                (ran ? "ran" : "failed"));
                outs[i][v].assign((size_t) (c.n * c.m), NAN);
                if (ran) {
                    ggml_backend_tensor_get(c.run->y, outs[i][v].data(), 0, ggml_nbytes(c.run->y));
                }
            }
            ggml_backend_free(be);
        }
        for (size_t i = 0; i < cases.size(); i++) {
            const p_case &    c  = cases[i];
            const std::string at = "pairs, K " + std::to_string(c.k) + " N " + std::to_string(c.n) + " M " +
                                   std::to_string(c.m) + " (" + std::to_string(c.slices) + " slices)";
            const size_t bytes = outs[i][0].size() * sizeof(float);
            for (int v = 1; v < 4; v++) {
                expect(std::memcmp(outs[i][v].data(), outs[i][0].data(), bytes) == 0,
                       at + ", " + variants[v] + ": not the bytes " + variants[0]);
            }
            int64_t       equal = 0;
            const int64_t bad   = f16_outside(outs[i][0], c.model, equal, p_pinned, p_worst);
            expect(bad == 0, at + ": " + std::to_string(bad) + " outputs outside the model's bound (" + mode + ")");
            expect(kernels || equal == (int64_t) outs[i][0].size(),
                   at + ": " + std::to_string(equal) + " of " + std::to_string(outs[i][0].size()) +
                       " outputs the model's bytes (references)");
            p_equal += equal;
            p_outputs += (int64_t) outs[i][0].size();
            const double e = nmse(outs[i][0].data(), c.ref.data(), (int64_t) c.ref.size());
            max_nmse       = std::max(max_nmse, e);
            expect(e <= k_mul_mat_max_nmse, at + ": NMSE against the CPU " + std::to_string(e));
        }
    }

    // C2. exact cases, on the plain route's 8 slices and the pairs route's 4 (ks = K / slices, each case's route
    // checked by spacemit_mul_mat_f16_route): every product and sum in them is exact in any order, so the outputs are
    // expected bit for bit in both modes (on the K3 the kernels, through a launch, there with the pairs' real waits).
    // (1) One 1.0 per slice in each row (row r at r mod ks) reads the fp16 weights back: out = the float sum in slice
    // order from +0.0f of fp16(w[c][s ks + r mod ks]). (2) A weight of d = 1 and nibble 9 at one place per slice and
    // column (8 elsewhere) reads the fp16 activations back, of mixed magnitudes: ties, the smallest normals, negative
    // zeros, large ones; no value is an fp16 subnormal. (3) Integers in [-16, 16] by d (q - 8) with d a power of two:
    // each slice's sum is exact in float, then rounded to fp16 once. (4) fp16 subnormals as an activation, a weight and
    // a partial sum: kept by the references (IEEE); what the K3's kernels do is reported, not required (the first K3
    // run, 2026-11-26: counted as zero somewhere on the hardware path)
    int         n_exact = 0, p_exact = 0;
    int64_t     n_rounded = 0;      // case 3's slice sums that fp16 rounds, on the plain route
    std::string subnormal[2][4];  // case 4's outcomes by route, for the summary
    for (int ri = 0; ri < 2; ri++) {
        const int64_t     slices = ri == 0 ? 8 : 4;
        const std::string on     = ri == 0 ? "" : ", pairs";
        env_override      sw(dense, "f16:16");
        env_override      rt(route, ri == 0 ? "plain" : "pairs");
        int64_t           rounded = 0;
        ggml_backend_t    be      = ggml_backend_dev_init(dev, nullptr);
        REQUIRE(be != nullptr);
        const auto run_exact = [&](const std::string & at, const std::vector<uint8_t> & q, int64_t k, int64_t n,
                                   const std::vector<float> & x, const std::vector<float> & want) {
            const int64_t m = (int64_t) x.size() / k;
            mm_weight     w;
            mm_run        run;
            if (!w.init_with(dev, GGML_TYPE_Q4_0, k, n, q) || !run.init(dev, w, m, 0)) {
                return expect(false, at + ": not set up");
            }
            ggml_backend_tensor_set(run.x, x.data(), 0, ggml_nbytes(run.x));
            const spacemit_op * op  = spacemit_find_op(ops, run.y, buft);
            const bool          ran = ggml_backend_graph_compute(be, run.g_dev) == GGML_STATUS_SUCCESS;
            std::vector<float>  out((size_t) (n * m), NAN);
            if (ran) {
                ggml_backend_tensor_get(run.y, out.data(), 0, ggml_nbytes(run.y));
            }
            int64_t bad = 0;
            for (size_t i = 0; i < out.size(); i++) {
                bad += std::memcmp(&out[i], &want[i], sizeof(float)) != 0;
            }
            (ri == 0 ? n_exact : p_exact)++;
            const int64_t taken = slices_of(run.y);
            return expect(is_f16(op) && taken == slices && ran && bad == 0,
                          at + on + ": claimed by " + by(op) + ", " + std::to_string(taken) + " slices" +
                              (ran ? ", " + std::to_string(bad) + " outputs not bit for bit" : ", failed"));
        };
        const auto shape = [](int64_t k, int64_t n, int64_t m) {
            return " [" + std::to_string(k) + ", " + std::to_string(n) + "] by " + std::to_string(m) + " rows";
        };

        // (1) a row count of ks at K 2048 and K 4096 takes every position of a slice (on the plain route 256 and 512,
        // two row passes at 4096 with 384 KiB of TCM; on the pairs route 512 and 1024, four passes at 4096); 77 rows
        // end in a partial row tile
        const int64_t f = 8 / slices;
        for (const std::array<int64_t, 3> & s : std::vector<std::array<int64_t, 3>>{
                 { 2048, 96, 256 * f }, { 4096, 64, 512 * f }, { 2048, 96, 77 } }) {
            const int64_t              k = s[0], n = s[1], m = s[2], ks = k / slices;
            const std::vector<uint8_t> q   = quantize_weights(GGML_TYPE_Q4_0, k, n, seed++);
            const std::vector<float>   deq = dequantize(GGML_TYPE_Q4_0, q);
            std::vector<float>         x((size_t) (k * m), 0.0f), want((size_t) (n * m));
            for (int64_t r = 0; r < m; r++) {
                for (int64_t sl = 0; sl < slices; sl++) {
                    x[r * k + sl * ks + r % ks] = 1.0f;
                }
                for (int64_t c = 0; c < n; c++) {
                    float o = 0.0f;
                    for (int64_t sl = 0; sl < slices; sl++) {
                        o = o + f16_round(deq[c * k + sl * ks + r % ks]);
                    }
                    want[r * n + c] = o;
                }
            }
            run_exact("weights read back," + shape(k, n, m), q, k, n, x, want);
        }

        // (2) the position read in slice s by column c varies with both
        for (const std::array<int64_t, 3> & s : std::vector<std::array<int64_t, 3>>{ { 2048, 96, 129 }, { 4096, 64, 64 } }) {
            const int64_t k = s[0], n = s[1], m = s[2], ks = k / slices;
            const auto    pos = [ks](int64_t c, int64_t sl) { return (c * 37 + sl * 11) % ks; };
            const std::vector<uint8_t> q = q4_0_raw(
                k, n, [](int64_t, int64_t) { return 1.0f; },
                [&](int64_t c, int64_t kk) { return kk % ks == pos(c, kk / ks) ? 9 : 8; });
            const std::vector<float> u = random_values(2 * k * m, seed++, 1.0f);
            std::vector<float>       x((size_t) (k * m)), want((size_t) (n * m));
            for (int64_t i = 0; i < k * m; i++) {
                const float a = u[2 * i], b = u[2 * i + 1];
                switch ((int) ((b + 1.0f) * 4.0f) % 8) {
                    case 0:
                        x[i] = -0.0f;
                        break;
                    case 1:
                        x[i] = std::copysign(6.2e-5f + std::fabs(a) * 1e-3f, a);  // fp16's smallest normals
                        break;
                    case 2:
                        x[i] = a * 3e4f;
                        break;
                    case 3:  // a midpoint between two fp16 values: rounds to the even one
                        x[i] = std::copysign(
                            std::ldexp(1.0f + (float) (2 * (int) (std::fabs(a) * 1023.0f) + 1) / 2048.0f, (int) (a * 10.0f)),
                            a);
                        break;
                    default:
                        x[i] = a * 2.0f;
                        break;
                }
                // no fp16 subnormal here (fp16's smallest normal instead): they are probed in (4)
                if (x[i] != 0.0f && std::fabs(f16_round(x[i])) < 0x1p-14f) {
                    x[i] = std::copysign(0x1p-14f, x[i]);
                }
            }
            for (int64_t r = 0; r < m; r++) {
                for (int64_t c = 0; c < n; c++) {
                    float o = 0.0f;
                    for (int64_t sl = 0; sl < slices; sl++) {
                        o = o + f16_round(x[r * k + sl * ks + pos(c, sl)]);
                    }
                    want[r * n + c] = o;
                }
            }
            run_exact("activations read back," + shape(k, n, m), q, k, n, x, want);
        }

        // (3) even rows non-negative and every third column's nibbles at least 8, so that many slice sums pass 2048,
        // where fp16 rounds
        for (const std::array<int64_t, 3> & s : std::vector<std::array<int64_t, 3>>{ { 2048, 64, 64 }, { 4096, 64, 48 } }) {
            const int64_t            k = s[0], n = s[1], m = s[2], ks = k / slices;
            const std::vector<float> ux = random_values(k * m, seed++, 1.0f);
            const std::vector<float> uq = random_values(k * n, seed++, 1.0f);
            const std::vector<float> ud = random_values(k / 32 * n, seed++, 1.0f);
            std::vector<float>       x((size_t) (k * m)), want((size_t) (n * m));
            std::vector<int>         qv((size_t) (k * n));
            std::vector<float>       dv((size_t) (k / 32 * n));
            for (int64_t r = 0; r < m; r++) {
                for (int64_t i = 0; i < k; i++) {
                    const int v = std::min(16, (int) ((ux[r * k + i] + 1.0f) * 16.5f) - 16);
                    x[r * k + i] = (float) (r % 2 == 0 ? std::abs(v) : v);
                }
            }
            for (int64_t c = 0; c < n; c++) {
                for (int64_t i = 0; i < k; i++) {
                    const int v      = std::min(15, (int) ((uq[c * k + i] + 1.0f) * 8.0f));
                    qv[c * k + i] = c % 3 == 0 ? 8 + v / 2 : v;
                }
                for (int64_t b = 0; b < k / 32; b++) {
                    dv[c * (k / 32) + b] = std::ldexp(1.0f, -std::min(3, (int) ((ud[c * (k / 32) + b] + 1.0f) * 2.0f)));
                }
            }
            const std::vector<uint8_t> q = q4_0_raw(
                k, n, [&](int64_t c, int64_t b) { return dv[c * (k / 32) + b]; },
                [&](int64_t c, int64_t kk) { return qv[c * k + kk]; });
            double largest = 0.0;
            for (int64_t r = 0; r < m; r++) {
                for (int64_t c = 0; c < n; c++) {
                    float o = 0.0f;
                    for (int64_t sl = 0; sl < slices; sl++) {
                        double e = 0.0;
                        for (int64_t i = sl * ks; i < (sl + 1) * ks; i++) {
                            e += (double) x[r * k + i] * (double) dv[c * (k / 32) + i / 32] * (double) (qv[c * k + i] - 8);
                        }
                        const float p = f16_round((float) e);
                        rounded += (double) p != e;
                        largest = std::max(largest, std::fabs(e));
                        o       = o + p;
                    }
                    want[r * n + c] = o;
                }
            }
            REQUIRE(largest < 65504.0);  // no slice sum overflows fp16
            run_exact("small integers," + shape(k, n, m), q, k, n, x, want);
        }
        expect(rounded > 0, "small integers" + on + ": no slice sum rounded by fp16");
        n_rounded += ri == 0 ? rounded : 0;

        // (4) a weight of d at one place per slice and column as in (2), every activation v: each output is 8 v d
        const struct {
            const char * name;
            float        v, d;
        } probes[4] = {
            { "an activation by 1", 0x1p-16f, 1.0f },        // the activation and the slice's partial sum
            { "an activation by 1024", 0x1p-16f, 1024.0f },  // the activation only (product 2^-6)
            { "a weight by 1024", 1024.0f, 0x1p-16f },       // the weight's scale only
            { "a partial sum", 0x1p-8f, 0x1p-8f },           // the slice's partial sum only (2^-16)
        };
        for (int p = 0; p < 4; p++) {
            const int64_t k = 2048, n = 64, m = 24, ks = k / slices;
            const auto    pos = [&](int64_t c, int64_t sl) { return (c * 37 + sl * 11) % ks; };
            const float   v = probes[p].v, d = probes[p].d;
            const std::vector<uint8_t> q = q4_0_raw(
                k, n, [d](int64_t, int64_t) { return d; },
                [&](int64_t c, int64_t kk) { return kk % ks == pos(c, kk / ks) ? 9 : 8; });
            const std::vector<float> x((size_t) (k * m), v);
            float                    want = 0.0f;
            for (int64_t sl = 0; sl < slices; sl++) {
                want = want + v * d;
            }
            mm_weight           w;
            mm_run              run;
            const spacemit_op * op  = nullptr;
            bool                ran = false;
            std::vector<float>  out((size_t) (n * m), NAN);
            if (w.init_with(dev, GGML_TYPE_Q4_0, k, n, q) && run.init(dev, w, m, 0)) {
                ggml_backend_tensor_set(run.x, x.data(), 0, ggml_nbytes(run.x));
                op  = spacemit_find_op(ops, run.y, buft);
                ran = ggml_backend_graph_compute(be, run.g_dev) == GGML_STATUS_SUCCESS;
                if (ran) {
                    ggml_backend_tensor_get(run.y, out.data(), 0, ggml_nbytes(run.y));
                }
            }
            const float   zero_f = 0.0f;
            const int64_t total  = (int64_t) out.size();
            int64_t       kept = 0, zero = 0, other = 0;
            for (size_t i = 0; i < out.size(); i++) {
                if (std::memcmp(&out[i], &want, sizeof(float)) == 0) {
                    kept++;
                } else if (std::memcmp(&out[i], &zero_f, sizeof(float)) == 0) {
                    zero++;
                } else {
                    other++;
                }
            }
            // the references are IEEE and keep every output; on the K3 the hardware decides, zeros are reported
            expect(is_f16(op) && ran && other == 0 && (kernels || zero == 0),
                   std::string("fp16 subnormals") + on + ", " + probes[p].name + ": claimed by " + by(op) + ", " +
                       std::to_string(kept) + " kept, " + std::to_string(zero) + " zero, " + std::to_string(other) +
                       " other of " + std::to_string(total));
            subnormal[ri][p] = kept == total ? "kept" :
                               zero == total ? "zero" :
                                               "kept in " + std::to_string(kept) + " of " + std::to_string(total) +
                                                   ", zero in " + std::to_string(zero);
        }
        ggml_backend_free(be);
    }

    // D. overflow fails the graph with one line; an infinite input passes through. On the plain route, and on the pairs
    // route (N 96: 2 slots) in turn and on threads, where the failing tile's partner gives up its pair wait when the
    // launch fails: the graph returns in under 5 s, not after the wait's 10 s
    for (int ri = 0; ri < 3; ri++) {
        const bool         pairs = ri > 0;
        const std::string  at    = ri == 0 ? "" : ri == 1 ? "pairs in turn, " : "pairs on threads, ";
        env_override       sw(dense, "f16");
        env_override       rt(route, pairs ? "pairs" : "plain");
        const int64_t      k = 512, n = pairs ? 96 : 64, m = 64, slices = pairs ? 4 : 8;
        std::vector<float> wv = random_values(k * n, seed++, 1.0f);
        std::fill(wv.begin(), wv.begin() + k, 0.5f);  // weight row 0: Q4_0 keeps 0.5 exactly
        std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q4_0, k) * n);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, wv.data(), q.data(), 0, n, k, nullptr);
        mm_weight w;
        REQUIRE(w.init_with(dev, GGML_TYPE_Q4_0, k, n, q));
        ggml_set_name(w.w_dev, "f16.overflow.weight");
        mm_run run;
        REQUIRE(run.init(dev, w, m, seed++));
        ggml_set_name(run.y, "f16.overflow");
        expect(slices_of(run.y) == slices, at + "overflow case: " + std::to_string(slices_of(run.y)) + " slices");
        const std::vector<float> x0 = random_values(k * m, seed++, 2.0f);
        ggml_backend_t           be = nullptr;
        {
            std::unique_ptr<env_override> threads;
            if (pairs) {
                threads = std::make_unique<env_override>("FLAGOS_SPACEMIT_TEST_THREADS", ri == 2 ? "1" : "0");
            }
            be = ggml_backend_dev_init(dev, nullptr);
        }
        REQUIRE(be != nullptr);
        // x0 with one change; then the status, the overflow lines and the seconds taken
        double     secs     = 0.0;
        const auto run_with = [&](const std::vector<float> & x, int & lines, std::string & text) {
            ggml_backend_tensor_set(run.x, x.data(), 0, ggml_nbytes(run.x));
            log_capture log;
            const auto  t0 = std::chrono::steady_clock::now();
            const bool  ok = ggml_backend_graph_compute(be, run.g_dev) == GGML_STATUS_SUCCESS;
            secs           = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            text           = log.take();
            lines          = count_lines_with(text, "mul_mat.f16: overflow");
            return ok;
        };
        int                lines = 0;
        std::string        text;
        std::vector<float> x = x0;
        x[5 * k + 3]         = 1e5f;
        const bool ok        = run_with(x, lines, text);
        expect(!ok && lines == 1 && secs < 5.0 &&
                   count_lines_with(text, "mul_mat.f16: overflow in f16.overflow (weight f16.overflow.weight): "
                                          "activation 100000 exceeds fp16") == 1,
               at + "an activation of 1e5: " + std::string(ok ? "success" : "failure") + ", " +
                   std::to_string(lines) + " overflow lines, " + std::to_string(secs) + " s");
        x                 = x0;
        x[0]              = INFINITY;
        const bool inf_ok = run_with(x, lines, text);
        expect(inf_ok && lines == 0, at + "an activation of +inf: " + std::string(inf_ok ? "success" : "failure") +
                                         ", " + std::to_string(lines) + " overflow lines");
        // row 0's slice 0 by weight row 0: K / slices * 3e4 * 0.5 (960000, 1.92e+06 on the pairs route); the kernels'
        // line has no value
        x = x0;
        std::fill(x.begin(), x.begin() + k / slices, 3e4f);
        const bool        sum_ok = run_with(x, lines, text);
        const std::string value  = pairs ? "1.92e+06" : "960000";
        const std::string line   = kernels ? "mul_mat.f16: overflow in f16.overflow (weight f16.overflow.weight): a "
                                             "partial sum exceeds fp16" :
                                             "mul_mat.f16: overflow in f16.overflow (weight f16.overflow.weight): "
                                             "partial sum " + value + " exceeds fp16";
        expect(!sum_ok && lines == 1 && secs < 5.0 && count_lines_with(text, line.c_str()) == 1,
               at + "a slice's sum of " + value + " (" + mode + "): " + (sum_ok ? "success" : "failure") + ", " +
                   std::to_string(lines) + " overflow lines, " + std::to_string(secs) + " s");
        ggml_backend_free(be);
    }

    // E. the default is f16: unset, empty and f16 give the f16 steps and the same bytes; int8 is today's integer path,
    // quantize and gemm with the bytes of f16:4096 (below its threshold, so the int8 contract)
    {
        mm_weight w;
        REQUIRE(w.init(dev, 512, 64, seed++));
        for (int64_t m : { (int64_t) 64, (int64_t) 200 }) {
            mm_run run;
            REQUIRE(run.init(dev, w, m, seed++));
            const char * settings[5] = { nullptr, "", "f16", "int8", "f16:4096" };
            const char * names[5]    = { "unset", "empty", "f16", "int8", "f16:4096" };
            std::vector<float> out[5];
            for (int i = 0; i < 5; i++) {
                env_override        sw(dense, settings[i]);
                const spacemit_op * op = spacemit_find_op(ops, run.y, buft);
                ggml_backend_t      be = ggml_backend_dev_init(dev, nullptr);
                REQUIRE(be != nullptr);
                const bool ran = ggml_backend_graph_compute(be, run.g_dev) == GGML_STATUS_SUCCESS;
                out[i].assign((size_t) (64 * m), NAN);
                ggml_backend_tensor_get(run.y, out[i].data(), 0, ggml_nbytes(run.y));
                ggml_backend_free(be);
                expect(ran && (i < 3 ? is_f16(op) : is_int8(op)),
                       std::string(names[i]) + ", " + std::to_string(m) + " rows: " +
                           (ran ? "claimed by " + by(op) : "failed"));
            }
            for (int i : { 1, 2, 4 }) {
                const int with = i == 4 ? 3 : 0;
                expect(std::memcmp(out[i].data(), out[with].data(), out[0].size() * sizeof(float)) == 0,
                       std::string(names[i]) + " and " + names[with] + " at " + std::to_string(m) +
                           " rows: not the same bytes");
            }
            expect(std::memcmp(out[0].data(), out[3].data(), out[0].size() * sizeof(float)) != 0,
                   "unset and int8 at " + std::to_string(m) + " rows: the same bytes (the f16 path not taken?)");
        }
    }

    // H. FLAGOS_SPACEMIT_F16_ROUTE (test-only), K 512 N 96 by 64 rows: unset, empty and pairs the pairs route and the
    // same bytes, a bad value warned once and pairs kept, plain the plain route. The two routes must differ here: some
    // outputs are pinned by both models' bounds to different values
    {
        env_override               sw(dense, "f16");
        const char * const         settings[5] = { nullptr, "", "pairs", "x", "plain" };
        const char * const         names[5]    = { "unset", "empty", "pairs", "x", "plain" };
        const std::vector<uint8_t> q           = quantize_weights(GGML_TYPE_Q4_0, 512, 96, seed++);
        mm_weight                  w;
        REQUIRE(w.init_with(dev, GGML_TYPE_Q4_0, 512, 96, q));
        mm_run run;
        REQUIRE(run.init(dev, w, 64, seed++));
        std::vector<float> x((size_t) (512 * 64));
        ggml_backend_tensor_get(run.x, x.data(), 0, ggml_nbytes(run.x));
        f16_model model4, model8;
        model4.init(x, dequantize(GGML_TYPE_Q4_0, q), 512, 96, 64, 4);
        model8.init(x, dequantize(GGML_TYPE_Q4_0, q), 512, 96, 64, 8);
        int64_t apart = 0;
        for (size_t i = 0; i < model4.out.size(); i++) {
            apart += model4.lo[i] == model4.hi[i] && model8.lo[i] == model8.hi[i] && model4.out[i] != model8.out[i];
        }
        expect(apart > 0, "FLAGOS_SPACEMIT_F16_ROUTE: no output pinned apart by the two routes' models");
        std::vector<float> out[5];
        for (int i = 0; i < 5; i++) {
            log_capture    log;
            env_override   rt(route, settings[i]);
            const int64_t  taken = slices_of(run.y);
            ggml_backend_t be    = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(be != nullptr);
            const bool ran = ggml_backend_graph_compute(be, run.g_dev) == GGML_STATUS_SUCCESS;
            ggml_backend_free(be);
            out[i].assign((size_t) (96 * 64), NAN);
            if (ran) {
                ggml_backend_tensor_get(run.y, out[i].data(), 0, ggml_nbytes(run.y));
            }
            const int warnings = count_lines_with(log.take(), "FLAGOS_SPACEMIT_F16_ROUTE=");
            expect(ran && taken == (i == 4 ? 8 : 4) && warnings == (i == 3 ? 1 : 0),
                   std::string("FLAGOS_SPACEMIT_F16_ROUTE ") + names[i] + ": " + (ran ? "ran" : "failed") + ", " +
                       std::to_string(taken) + " slices, " + std::to_string(warnings) + " warnings");
        }
        for (int i = 1; i < 4; i++) {
            expect(std::memcmp(out[i].data(), out[0].data(), out[0].size() * sizeof(float)) == 0,
                   std::string("FLAGOS_SPACEMIT_F16_ROUTE ") + names[i] + ": not the bytes of unset");
        }
        expect(std::memcmp(out[4].data(), out[0].data(), out[0].size() * sizeof(float)) != 0,
               "FLAGOS_SPACEMIT_F16_ROUTE plain: the bytes of unset (" + std::to_string(apart) +
                   " outputs pinned apart)");
    }

    // G. FLAGOS_SPACEMIT_QUANTIZE=shared (test-only) skips no f16 step: two f16 matmuls of one input give the bytes and
    // the step count of separate
    {
        env_override sw(dense, "f16");
        mm_weight    wa, wb;
        REQUIRE(wa.init(dev, 512, 64, seed++) && wb.init(dev, 512, 64, seed++));
        ggml_init_params p  = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true };
        ggml_context *   ch = ggml_init(p);
        ggml_context *   cd = ggml_init(p);
        REQUIRE(ch != nullptr && cd != nullptr);
        ggml_tensor * x  = ggml_new_tensor_2d(ch, GGML_TYPE_F32, 512, 64);
        ggml_tensor * ya = ggml_mul_mat(cd, wa.w_dev, x);
        ggml_tensor * yb = ggml_mul_mat(cd, wb.w_dev, x);
        ggml_backend_buffer_t bh = ggml_backend_alloc_ctx_tensors_from_buft(ch, ggml_backend_cpu_buffer_type());
        ggml_backend_buffer_t bd = ggml_backend_alloc_ctx_tensors_from_buft(cd, buft);
        REQUIRE(bh != nullptr && bd != nullptr);
        const std::vector<float> xv = random_values(512 * 64, seed++, 2.0f);
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_cgraph * g = ggml_new_graph(cd);
        ggml_build_forward_expand(g, ya);
        ggml_build_forward_expand(g, yb);
        std::vector<float> out[2];
        uint64_t           steps[2] = {};
        bool               ran      = true;
        for (int i = 0; i < 2; i++) {
            env_override   share("FLAGOS_SPACEMIT_QUANTIZE", i == 0 ? "separate" : "shared");
            ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(be != nullptr);
            g_steps_take();
            ran      = ggml_backend_graph_compute(be, g) == GGML_STATUS_SUCCESS && ran;
            steps[i] = g_steps_take();
            ggml_backend_free(be);
            out[i].assign(2 * 64 * 64, NAN);
            ggml_backend_tensor_get(ya, out[i].data(), 0, ggml_nbytes(ya));
            ggml_backend_tensor_get(yb, out[i].data() + 64 * 64, 0, ggml_nbytes(yb));
        }
        expect(ran && steps[0] == 4 && steps[1] == 4 &&
                   std::memcmp(out[0].data(), out[1].data(), out[0].size() * sizeof(float)) == 0,
               "FLAGOS_SPACEMIT_QUANTIZE=shared: steps " + std::to_string(steps[1]) + " (separate " +
                   std::to_string(steps[0]) + "), " + (ran ? "bytes compared" : "failed"));
        ggml_backend_buffer_free(bh);
        ggml_backend_buffer_free(bd);
        ggml_free(ch);
        ggml_free(cd);
    }

    // F. FLAGOS_SPACEMIT_F16_STATS: appended when a backend that took f16 matmuls is freed; unset, nothing. The largest
    // values only from the references (the kernels do not scan)
    {
        const std::string path      = std::string(FLAGOS_SPACEMIT_TEST_PLUGIN_DIR) + "/f16-stats.txt";
        const auto        read_back = [&] {
            std::ifstream in(path);
            return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        };
        mm_weight wa, wb;
        REQUIRE(wa.init(dev, 512, 64, seed++) && wb.init(dev, 256, 32, seed++));
        ggml_set_name(wa.w_dev, "f16.stats_a.weight");
        ggml_set_name(wb.w_dev, "f16.stats_b.weight");
        mm_run ra, rb, rc;
        REQUIRE(ra.init(dev, wa, 64, seed++) && rb.init(dev, wa, 100, seed++) && rc.init(dev, wb, 64, seed++));
        for (int pass = 0; pass < 3; pass++) {  // off; on, the plain route; on, the pairs route
            const bool on    = pass > 0;
            const bool pairs = pass == 2;
            FILE *     f     = std::fopen(path.c_str(), "w");
            REQUIRE(f != nullptr);
            std::fclose(f);
            env_override   sw(dense, "f16");
            env_override   rt(route, pairs ? "pairs" : "plain");
            env_override   sw_stats("FLAGOS_SPACEMIT_F16_STATS", on ? path.c_str() : nullptr);
            ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(be != nullptr);
            bool ran = true;
            for (mm_run * r : { &ra, &rb, &rc }) {
                ran = ggml_backend_graph_compute(be, r->g_dev) == GGML_STATUS_SUCCESS && ran;
            }
            ggml_backend_free(be);
            const std::string text = read_back();
            if (!on) {
                expect(ran && text.empty(), "stats unset: " + std::to_string(text.size()) + " bytes written");
                continue;
            }
            // tile 0's pair waits on the pairs route (products_pairs): order A (2 slots, the tiles at the same time)
            // 1 + chunks a pass, order B 2 chunks a pass; the route under the backend's TCM, 384 KiB on the K3
            const char * tcm_env    = std::getenv("FLAGOS_SPACEMIT_TEST_TCM");
            const char * thr_env    = std::getenv("FLAGOS_SPACEMIT_TEST_THREADS");
            const size_t host_tcm   = tcm_env != nullptr ? (size_t) std::max(0LL, std::atoll(tcm_env)) : 0;
            const size_t tcm        = host ? host_tcm : 393216;
            const bool   concurrent = !host || (thr_env != nullptr && std::strcmp(thr_env, "0") != 0);
            int64_t      waits      = 0;
            for (const mm_run * r : { &ra, &rb, &rc }) {
                int64_t q[4] = {};
                route_of(r->y, dev_tiles, tcm, q);
                const int64_t chunks = (r->y->ne[0] / 32 + 1) / 2;
                waits += q[2] * (q[1] == 2 && concurrent ? 1 + chunks : 2 * chunks);  // 0 passes on the plain route
            }
            const int         largest = kernels ? 0 : 1;
            const std::string first   = std::string("mul_mat.f16: matmuls 3 (pairs ") + (pairs ? "3" : "0") +
                                      "), overflows 0, threshold " + std::to_string(f16_default) +
                                      ", pair waits " + std::to_string(waits);
            const bool        lines   = (waits > 0) == pairs && text.substr(0, text.find('\n')) == first &&
                               count_lines_with(text, "mul_mat.f16: matmuls ") == 1 &&
                               count_lines_with(text, "mul_mat.f16: largest activation ") == largest &&
                               count_lines_with(text, "mul_mat.f16: largest partial sum ") == largest &&
                               count_lines_with(text, "mul_mat.f16: K 512 N 64: 2 matmuls") == 1 &&
                               count_lines_with(text, "mul_mat.f16: K 256 N 32: 1 matmuls") == 1 &&
                               count_lines_with(text, " in f16.stats_") == 2 * largest &&
                               count_lines_with(text, "mul_mat.f16: ") == 3 + 2 * largest;
            expect(ran && lines, std::string("stats (") + mode + (pairs ? ", pairs" : "") + "): " + text);
        }
    }
    ggml_backend_free(cpu);

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("f16mm    %s: FLAGOS_SPACEMIT_DENSE_Q4_0 (unset, empty, int8, f16, f16:16, f16:128, x, f16:0, f16:abc, "
                "f16:5000): Q4_0 [512, 64] taken by mul_mat.f16.products and .sum from the threshold (32 the default: "
                "unset, empty, f16 or a bad value; 16, 128), else and under int8 quantize and gemm; bad values warned "
                "once; never a Q8_0 or 3-D weight, over 2^24 or K 33024. %d staged weight tiles fp16 of ggml's "
                "dequantization bit for bit. On the plain route K 2048 N 96 at 16-501 rows, K 4096 N 64 at 64 and "
                "700, K 256 N 32 at 64, K 256 N 320 at 40: default, TCM %s serial and on threads the same bytes, the "
                "reference steps on 1 and 3 tiles direct the model's; %lld of %lld outputs the model's bytes (%lld "
                "pinned to it by the bound), worst |out - model| / bound %.3f; max NMSE vs the CPU (both routes) %.2e "
                "(bound %.0e). %d exact cases bit for bit (weights and activations read back, small integers with %lld "
                "slice sums rounded). fp16 subnormals: an activation by 1 %s, an activation by 1024 %s, a weight by "
                "1024 %s, a partial sum %s on the plain route, on pairs %s, %s, %s and %s. pairs: %lld of %lld outputs "
                "the model's bytes (%lld pinned), %d exact cases bit for bit, in turn and on threads the same bytes. "
                "The pairs route (the default, 8 tiles, 4 slices) at K 2048 N 96 at 16-501 rows, K 4096 N 64 and N 128 "
                "at 64-1100, K 256 N 320 at 40, slots and passes as the fork's rules give, K 8192 the plain route's 8 "
                "slices, with and without TCM, worst |out - model| / bound %.3f; unset, empty, pairs and a bad value "
                "the same bytes, the bad value warned once, plain 8 slices and other bytes. Overflow of an activation and of a partial "
                "sum fail with one line, on pairs in turn and on threads in under 5 s, +inf passes; unset, empty and "
                "f16 the same bytes by the f16 steps, int8 and f16:4096 the same bytes by quantize and gemm; "
                "QUANTIZE=shared skips no f16 step; F16_STATS appended at the backend's free, nothing unset: %d/%d in "
                "%.1f s\n",
                mode, n_staged, small_tcm, (long long) n_equal, (long long) n_outputs, (long long) n_pinned, worst, max_nmse,
                k_mul_mat_max_nmse, n_exact, (long long) n_rounded, subnormal[0][0].c_str(), subnormal[0][1].c_str(),
                subnormal[0][2].c_str(), subnormal[0][3].c_str(), subnormal[1][0].c_str(), subnormal[1][1].c_str(),
                subnormal[1][2].c_str(), subnormal[1][3].c_str(), (long long) p_equal, (long long) p_outputs,
                (long long) p_pinned, p_exact, p_worst, n_ok, n_checks, secs);
    return 0;
}

static int check_mul_mat_f32_split(ggml_backend_dev_t dev) {
    env_override router_f32("FLAGOS_SPACEMIT_ROUTER", "f32");  // the F32 matmul path: its weights carry the router's name
    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    const spacemit_plugin_kernel * k_mm      = nullptr;
    for (uint32_t i = 0; i < n_kernels && kernels != nullptr; i++) {
        if (kernels[i].name != nullptr && std::strcmp(kernels[i].name, "mul_mat.f32") == 0) {
            k_mm = &kernels[i];
        }
    }
    REQUIRE(k_mm != nullptr);
    constexpr uint32_t n_tiles = 8;
    constexpr uint8_t  junk[4] = { 0xA5, 0xA5, 0xA5, 0xA5 };

    // [begin, end) of n items for one tile, as the kernel's tile_range splits them
    const auto range = [](int64_t n, uint32_t ith, uint32_t nth, int64_t & begin, int64_t & end) {
        const int64_t per = (n + nth - 1) / nth;
        begin             = std::min(n, per * (int64_t) ith);
        end               = std::min(n, begin + per);
    };
    // the output floats a tile's share holds: the weight rows in units of 4 where they outnumber the token rows, else
    // the token rows
    const auto share_floats = [&](const mm_f32_shape & s, uint32_t ith) {
        const int64_t n_weight = s.n, n_token = s.rows * s.ne2;
        int64_t       begin, end;
        if (n_weight > n_token) {
            range((n_weight + 3) / 4, ith, n_tiles, begin, end);
            return (std::min(n_weight, 4 * end) - std::min(n_weight, 4 * begin)) * n_token;
        }
        range(n_token, ith, n_tiles, begin, end);
        return n_weight * (end - begin);
    };

    int        n_checks = 0, n_ok = 0, n_shapes = 0, n_shapes_ok = 0;
    const auto expect = [&](bool ok, const mm_f32_shape & s, const std::string & what) {
        n_checks++;
        n_ok += ok;
        if (!ok) {
            std::fprintf(stderr, "FAIL: mm.f32 split, %s: %s\n", mm_f32_what(s).c_str(), what.c_str());
            g_failures++;
        }
    };

    uint32_t seed = 1700;
    for (const mm_f32_shape & s : k_mm_f32_shapes) {
        const int before = n_ok - n_checks;
        mm_f32_node host;
        REQUIRE(host.init(dev, s, false, seed));
        seed += 8;
        const int64_t n_floats = ggml_nelements(host.y);
        // tiles [first, last) of nth computed on the junk-filled output
        const auto tiles = [&](uint32_t first, uint32_t last, uint32_t nth) {
            const std::vector<uint8_t> filled(ggml_nbytes(host.y), 0xA5);
            ggml_backend_tensor_set(host.y, filled.data(), 0, filled.size());
            for (uint32_t ith = first; ith < last; ith++) {
                const spacemit_tile tile = { ith, nth, nullptr, 0, nullptr, 0, nullptr };
                CHECK(k_mm->fn(&tile, host.y));
            }
            return mm_f32_out(host.y);
        };

        const std::vector<uint8_t> want = tiles(0, 1, 1);
        for (const uint32_t nth : { 3u, 8u }) {
            expect(tiles(0, nth, nth) == want, s,
                   "the direct calls on " + std::to_string(nth) + " tiles are not the bytes of 1 tile");
        }

        // who computes what: one tile at a time of 8, its floats against the share rule, the eight sets disjoint and
        // the output covered
        std::vector<int64_t> counts(n_tiles, 0);
        std::vector<int>     by(n_floats, 0);
        for (uint32_t ith = 0; ith < n_tiles; ith++) {
            const std::vector<uint8_t> got = tiles(ith, ith + 1, n_tiles);
            for (int64_t i = 0; i < n_floats; i++) {
                const bool changed = std::memcmp(got.data() + i * sizeof(float), junk, sizeof(junk)) != 0;
                counts[ith] += changed;
                by[i] += changed;
            }
            expect(counts[ith] == share_floats(s, ith), s,
                   "tile " + std::to_string(ith) + " of 8 computed " + std::to_string(counts[ith]) +
                       " floats, the share rule " + std::to_string(share_floats(s, ith)));
        }
        int64_t n_once = 0;
        for (const int c : by) {
            n_once += c == 1;
        }
        expect(n_once == n_floats, s,
               std::to_string(n_once) + " of " + std::to_string(n_floats) +
                   " floats were computed by exactly one of the 8 tiles");
        // plan.md's own figures for three shapes
        const auto literal = [&](int64_t first, int64_t last) {
            bool ok = true;
            for (uint32_t ith = 0; ith < n_tiles; ith++) {
                ok = ok && counts[ith] == (ith + 1 < n_tiles ? first : last);
            }
            expect(ok, s,
                   "the counts are not " + std::to_string(first) + " floats on tiles 0-6 and " +
                       std::to_string(last) + " on tile 7");
        };
        if (s.k == 2048 && s.n == 256 && s.rows == 1 && s.ne2 == 1) {
            literal(32, 32);
        } else if (s.k == 2048 && s.n == 32 && s.rows == 1) {
            literal(4, 4);
        } else if (s.k == 2048 && s.n == 256 && s.rows == 501) {
            literal(63 * 256, 60 * 256);
        }
        n_shapes++;
        n_shapes_ok += n_ok - n_checks == before;
    }

    std::printf("mmf32    the F32 matmul split over the larger of the weight rows and the token rows, in blocks of 16 "
                "token rows, at %d shapes ([2048, 256] by 1, 2, 31, 255, 256 and 501 token rows, [2048, 32] by 1, 31, "
                "32 and 33, [64, 7] by 1, 3 and 9, [2048, 1] by 1 and 40, a 3-D batch): %d/%d shapes, the direct "
                "calls on 1, 3 and 8 tiles the same bytes, every tile's floats the share rule's and the 8 tiles "
                "disjoint and whole (32 a tile at [2048, 256] by 1, 4 at [2048, 32] by 1, 63 and 60 rows of 256 at "
                "501 rows): %d/%d\n",
                (int) (sizeof(k_mm_f32_shapes) / sizeof(k_mm_f32_shapes[0])), n_shapes_ok, n_shapes, n_ok, n_checks);
    return 0;
}

//
// The convolution pair (core/flagos-spacemit-ops.h spacemit_conv_pair, plugin/flagos-spacemit-seq.cpp): a CONCAT whose
// only consumers are one SSM_CONV and views of its last 3 columns runs concat.f32.tail and ssm_conv.f32.stashed. The
// graphs as delta-net-base.cpp build_conv_state and qwen35.cpp build them: concat(state [3, C, n_s], transpose(x)),
// the view of its last 3 columns copied into a state tensor, ssm_conv(concat, kernel). Through the device against
// ggml-cpu on one thread: the convolution's output within the rounding bound of check_seq_ops' SSM_CONV on the K3
// (NMSE 1e-12, |difference| / max(1, |y|) 1e-6), with the reference kernels too, as a compiler may contract the
// multiply and add (seen with Apple clang 21 on 2026-11-18, NMSE 3e-15); the copied state (and a case's other output)
// bit for bit, fused or not by the pair counter (spacemit_conv_pairs_take). A fused case's two
// kernels also called directly on 1, 3 and 8 tiles on a CPU-buffer copy (on the K3 the calling core, another vector
// length): the device's bytes, since each element is one sum in one order whatever the split and the vector length.
// Then one fused graph through the scheduler (its allocator reusing memory) under the plan cache: a miss, then a
// trusted hit counting no pair, both outputs the same bytes, within the bound of the CPU's and the copied state bit for
// bit. GGML_FLAGOS_KERNEL_VARIANTS=0 is read once per process,
// so the program's reference pass covers the portable kernels; the device always runs its 8 tiles
//

struct conv_pair_case {
    const char * what;
    int64_t      n, C, n_s, kw;
    int          variant;  // 0 the model's graph, 1 a second consumer (SCALE of the concat), 2 a view from column
                           // n - 1 copied out (CONT), 3 the concat flagged as an output
    bool         fused;
};

// one case's graph; with ops_unallocated only the inputs are allocated (the scheduler allocates the rest)
struct conv_pair_graph {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         st = nullptr, *x = nullptr, *w = nullptr, *state = nullptr;  // state: the copy's target
    ggml_tensor *         cat = nullptr, *out = nullptr, *extra = nullptr;  // extra: variants 1 and 2's output
    ggml_cgraph *         g = nullptr;

    conv_pair_graph() = default;
    conv_pair_graph(const conv_pair_graph &) = delete;
    conv_pair_graph & operator=(const conv_pair_graph &) = delete;

    ~conv_pair_graph() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    bool init(const conv_pair_case & c, ggml_backend_buffer_type_t buft, uint32_t seed, bool ops_unallocated = false) {
        ggml_init_params params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        ctx                     = ggml_init(params);
        if (ctx == nullptr) {
            return false;
        }
        st    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, c.C, c.n_s);
        x     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.C, c.n, c.n_s);
        w     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.kw, c.C);
        state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, c.C, c.n_s);
        if (ops_unallocated) {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }
        cat = ggml_concat(ctx, st, ggml_transpose(ctx, x), 0);
        if (c.variant == 3) {
            ggml_set_output(cat);
        }
        ggml_tensor * last = ggml_view_3d(ctx, cat, 3, c.C, c.n_s, cat->nb[1], cat->nb[2], c.n * sizeof(float));
        ggml_tensor * cp   = ggml_cpy(ctx, last, state);
        out                = ggml_ssm_conv(ctx, cat, w);
        if (c.variant == 1) {
            extra = ggml_scale(ctx, cat, 2.0f);
        } else if (c.variant == 2) {
            extra = ggml_cont(ctx, ggml_view_3d(ctx, cat, 3, c.C, c.n_s, cat->nb[1], cat->nb[2],
                                                (c.n - 1) * sizeof(float)));
        }
        ggml_set_output(out);
        g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, cp);
        ggml_build_forward_expand(g, out);
        if (extra != nullptr) {
            ggml_set_output(extra);
            ggml_build_forward_expand(g, extra);
        }
        if (!ops_unallocated) {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }
        if (buf == nullptr) {
            return false;
        }
        uint32_t s = seed;
        for (ggml_tensor * t : { st, x, w }) {
            const std::vector<float> v = random_values(ggml_nelements(t), s++, 1.0f);
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        }
        return true;
    }

    // the outputs a run writes, set to junk before it so that a missed write shows
    void junk() const {
        for (ggml_tensor * t : { cat, out, state, extra }) {
            if (t != nullptr && t->data != nullptr) {
                const std::vector<uint8_t> j(ggml_nbytes(t), 0xA5);
                ggml_backend_tensor_set(t, j.data(), 0, j.size());
            }
        }
    }

    static std::vector<uint8_t> bytes(const ggml_tensor * t) {
        std::vector<uint8_t> b(t != nullptr ? ggml_nbytes(t) : 0);
        if (!b.empty()) {
            ggml_backend_tensor_get(t, b.data(), 0, b.size());
        }
        return b;
    }

    // the concat's last 3 columns, the true ones, of every row
    std::vector<uint8_t> tail() const {
        const std::vector<uint8_t> all = bytes(cat);
        const int64_t              L   = cat->ne[0];
        std::vector<uint8_t>       t;
        for (int64_t r = 0; r < ggml_nrows(cat); r++) {
            const uint8_t * p = all.data() + (r * L + L - 3) * sizeof(float);
            t.insert(t.end(), p, p + 3 * sizeof(float));
        }
        return t;
    }
};

static int check_conv_pair(ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);  // a correctness reference: single-threaded (PROJECT.md section 6, item 28)
    // check_seq_ops' SSM_CONV bound with the RVV kernels, for every kernel: a compiler may contract the stashed
    // kernel's multiply and add, the RVV port fuses its taps
    const seq_bound bound = { 1e-12, 1e-6 };
    // the convolution's output against the CPU's: its NMSE and its largest |difference| / max(1, |y|); -1 if the sizes
    // differ
    const auto rounding = [](const std::vector<uint8_t> & got, const void * want, size_t want_bytes, double & e,
                             double & a) {
        e = a = -1.0;
        if (got.size() != want_bytes) {
            return;
        }
        const float * g = (const float *) got.data();
        const float * r = (const float *) want;
        const int64_t n = (int64_t) (got.size() / sizeof(float));
        e               = nmse(g, r, n);
        a               = 0.0;
        for (int64_t i = 0; i < n; i++) {
            double d = HUGE_VAL;
            if (g[i] == r[i]) {
                d = 0.0;
            } else if (std::isfinite(g[i]) && std::isfinite(r[i])) {
                d = std::fabs((double) g[i] - r[i]) / std::max(1.0, std::fabs((double) r[i]));
            }
            a = std::max(a, d);
        }
    };

    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    const auto                     entry     = [&](const char * name) -> const spacemit_plugin_kernel * {
        for (uint32_t k = 0; k < n_kernels; k++) {
            if (kernels[k].name != nullptr && std::strcmp(kernels[k].name, name) == 0) {
                return &kernels[k];
            }
        }
        return nullptr;
    };
    const spacemit_plugin_kernel * k_tail  = entry("concat.f32.tail");
    const spacemit_plugin_kernel * k_stash = entry("ssm_conv.f32.stashed");
    REQUIRE(k_tail != nullptr && k_stash != nullptr && k_stash->workspace_size != nullptr);
    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    env_override                     plan_default("FLAGOS_SPACEMIT_PLAN", nullptr);
    env_override                     quantize_default("FLAGOS_SPACEMIT_QUANTIZE", nullptr);

    const conv_pair_case cases[] = {
  // what,                                          n,    C, n_s, kw, variant, fused
        { "32 tokens of 8 channels (1 per tile)",       32,    8,   1,  4,       0, true  },
        { "33 tokens of 96",                            33,   96,   1,  4,       0, true  },
        { "100 tokens of 1000",                        100, 1000,   1,  4,       0, true  },
        { "501 tokens of 8192 (Qwen3.5-4B)",           501, 8192,   1,  4,       0, true  },
        { "64 tokens of 7 (an idle tile)",              64,    7,   1,  4,       0, true  },
        { "31 tokens (below 32)",                       31,   96,   1,  4,       0, false },
        { "a second consumer (SCALE of the concat)",    64,   96,   1,  4,       1, false },
        { "a view from column n - 1 copied out",        64,   96,   1,  4,       2, false },
        { "2 sequences",                                64,   96,   2,  4,       0, false },
        { "a kernel of width 3",                        64,   96,   1,  3,       0, false },
        { "the concat flagged as an output",            64,   96,   1,  4,       3, false },
    };
    int      n_cases = 0, n_ok = 0, n_fused = 0, n_direct = 0, n_direct_ok = 0;
    double   worst_nmse = 0.0, worst_abs = 0.0;
    uint32_t seed = 1300;
    for (const conv_pair_case & c : cases) {
        n_cases++;
        conv_pair_graph dv, ref;
        if (!dv.init(c, buft, seed) || !ref.init(c, ggml_backend_cpu_buffer_type(), seed)) {
            std::fprintf(stderr, "FAIL: conv pair %s: allocation failed\n", c.what);
            g_failures++;
            seed += 8;
            continue;
        }
        const bool ref_ok = ggml_backend_graph_compute(cpu, ref.g) == GGML_STATUS_SUCCESS;
        const std::vector<uint8_t> want_out = conv_pair_graph::bytes(ref.out), want_state = conv_pair_graph::bytes(ref.state),
                                   want_extra = conv_pair_graph::bytes(ref.extra), want_tail = ref.tail();

        // one device run: ran, the pairs it fused, its outputs
        const auto run = [&](uint64_t & pairs, std::vector<uint8_t> & out, std::vector<uint8_t> & state,
                             std::vector<uint8_t> & extra) {
            dv.junk();
            spacemit_conv_pairs_take();  // drops earlier graphs'
            const bool ok = ggml_backend_graph_compute(backend, dv.g) == GGML_STATUS_SUCCESS;
            pairs         = spacemit_conv_pairs_take();
            out           = conv_pair_graph::bytes(dv.out);
            state         = conv_pair_graph::bytes(dv.state);
            extra         = conv_pair_graph::bytes(dv.extra);
            return ok;
        };
        uint64_t             pairs = 0;
        std::vector<uint8_t> out, state, extra;
        const bool           dev_ok = ref_ok && run(pairs, out, state, extra);
        double               e = -1.0, a = -1.0;
        bool                 exact = false;
        if (dev_ok) {
            rounding(out, want_out.data(), want_out.size(), e, a);
            exact = out == want_out;
        }
        const bool in_bound = e >= 0.0 && e <= bound.nmse && a >= 0.0 && a <= bound.abs;
        worst_nmse          = std::max(worst_nmse, e);
        worst_abs           = std::max(worst_abs, a);
        bool direct_ok = true;
        if (c.fused && dev_ok) {
            // the two kernels directly, on a CPU-buffer copy
            conv_pair_graph dir;
            direct_ok = dir.init(c, ggml_backend_cpu_buffer_type(), seed);
            const size_t       ws = k_stash->workspace_size(dir.out);
            std::vector<float> workspace(ws / sizeof(float) + 1, NAN);
            for (uint32_t nth : { 1u, 3u, 8u }) {
                if (!direct_ok) {
                    break;
                }
                n_direct++;
                dir.junk();
                bool ok = true;
                for (uint32_t ith = 0; ith < nth; ith++) {
                    const spacemit_tile tile = { ith, nth, nullptr, 0, nullptr, 0, nullptr };
                    ok &= k_tail->fn(&tile, dir.cat);
                }
                for (uint32_t ith = 0; ith < nth; ith++) {
                    const spacemit_tile tile = { ith, nth, nullptr, 0, workspace.data(), ws, nullptr };
                    ok &= k_stash->fn(&tile, dir.out);
                }
                const bool same = ok && conv_pair_graph::bytes(dir.out) == out && dir.tail() == want_tail;
                n_direct_ok += same;
                if (!same) {
                    std::fprintf(stderr, "FAIL: conv pair %s, the kernels called directly on %u tiles: %s, output %s\n",
                                 c.what, nth, ok ? "ran" : "failed", same ? "equal" : "differs from the device's");
                }
                direct_ok = direct_ok && same;
            }
        }
        const bool ok = ref_ok && dev_ok && pairs == (c.fused ? 1u : 0u) && in_bound && state == want_state &&
                        extra == want_extra && direct_ok;
        n_ok += ok;
        n_fused += pairs == 1;
        if (!ok) {
            std::fprintf(stderr,
                         "FAIL: conv pair %s (n %lld, C %lld, %lld sequences, kernel width %lld): the CPU reference %s, "
                         "the run %s, %llu pairs fused (expected %d); NMSE %.3e, |difference| / max(1, |y|) %.3e "
                         "(bounds %.0e, %.0e), bytes as the CPU's %s; the copied state %s, the other output %s; the "
                         "direct calls %s\n",
                         c.what, (long long) c.n, (long long) c.C, (long long) c.n_s, (long long) c.kw,
                         ref_ok ? "ran" : "failed", dev_ok ? "ran" : "failed", (unsigned long long) pairs,
                         c.fused ? 1 : 0, e, a, bound.nmse, bound.abs, exact ? "yes" : "no",
                         state == want_state ? "as the CPU's" : "differs", extra == want_extra ? "as the CPU's" : "differs",
                         direct_ok ? "as the device's" : "differ");
            g_failures++;
        }
        seed += 8;
    }

    // through the scheduler (inputs in our buffer, the rest in its compute buffer) under the plan cache's uid trust
    bool     sched_ok = false;
    uint64_t sched_pairs[2] = {}, sched_hits[2] = {}, sched_misses[2] = {};
    {
        const conv_pair_case c = { "the scheduler", 100, 1000, 1, 4, 0, true };
        conv_pair_graph      sg, ref;
        if (sg.init(c, buft, seed, true) && ref.init(c, ggml_backend_cpu_buffer_type(), seed)) {
            ggml_backend_t       both[2] = { backend, cpu };
            ggml_backend_sched_t sched   = ggml_backend_sched_new(both, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
            bool                 ok      = sched != nullptr && ggml_backend_graph_compute(cpu, ref.g) == GGML_STATUS_SUCCESS;
            const std::vector<uint8_t> want_state = conv_pair_graph::bytes(ref.state);
            std::vector<uint8_t>       first;
            for (int i = 0; i < 2 && ok; i++) {
                spacemit_conv_pairs_take();
                spacemit_plan_counters_take();
                ok = ggml_backend_sched_graph_compute(sched, sg.g) == GGML_STATUS_SUCCESS;
                const spacemit_plan_counters pc = spacemit_plan_counters_take();
                sched_pairs[i]                  = spacemit_conv_pairs_take();
                sched_hits[i]                   = pc.hits_uid;
                sched_misses[i]                 = pc.misses;
                const std::vector<uint8_t> out  = ok ? conv_pair_graph::bytes(sg.out) : std::vector<uint8_t>();
                if (i == 0) {
                    first = out;
                }
                ok = ok && out == first && conv_pair_graph::bytes(sg.state) == want_state;
            }
            const bool on_device = sched != nullptr && ggml_backend_sched_get_tensor_backend(sched, sg.cat) == backend &&
                                   ggml_backend_sched_get_tensor_backend(sched, sg.out) == backend;
            // the output within the bound of ggml-cpu's
            double e = -1.0, a = -1.0;
            if (ok) {
                rounding(first, ref.out->data, ggml_nbytes(ref.out), e, a);
            }
            const bool e_ok = e >= 0.0 && e <= bound.nmse && a >= 0.0 && a <= bound.abs;
            sched_ok = ok && on_device && e_ok && sched_pairs[0] == 1 && sched_misses[0] == 1 && sched_hits[1] == 1 &&
                       sched_pairs[1] == 0;
            if (sched != nullptr) {
                ggml_backend_sched_free(sched);
            }
        }
        if (!sched_ok) {
            std::fprintf(stderr,
                         "FAIL: conv pair through the scheduler: pairs %llu/%llu (expected 1/0), plan hits %llu/%llu "
                         "(expected 0/1), misses %llu/%llu (expected 1/0), or the outputs, the placement or the bound "
                         "failed\n",
                         (unsigned long long) sched_pairs[0], (unsigned long long) sched_pairs[1],
                         (unsigned long long) sched_hits[0], (unsigned long long) sched_hits[1],
                         (unsigned long long) sched_misses[0], (unsigned long long) sched_misses[1]);
            g_failures++;
        }
    }

    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    std::printf("convpair CONCAT + SSM_CONV as the model builds them (n 32, 33, 100, 501, 64 tokens of 8, 96, 1000, "
                "8192, 7 channels fused; 31 tokens, a second consumer, a view from column n - 1, 2 sequences, a kernel "
                "of width 3, an output concat not)%s: %d/%d cases, %d fused by the counter; the output against the CPU "
                "(a rounding bound) max NMSE %.2e, |difference| / max(1, |y|) %.2e (bounds %.0e, %.0e), the copied "
                "state and other outputs bit for bit; the kernels directly on 1, 3 and 8 tiles as the device %d/%d; "
                "through the scheduler a miss, a hit by uid (no pair), the same bytes, within the bound: %s\n",
                reference_label(), n_ok, n_cases, n_fused, worst_nmse, worst_abs, bound.nmse, bound.abs, n_direct_ok,
                n_direct, sched_ok ? "checked" : "FAILED");
    return 0;
}

// the many-row path of MUL_MAT_ID: from 2 tokens the (token, slot) pairs are sorted by expert, an expert's rows go in
// groups of 4 through the 4-row kernel (the rest one by one) and each 32-column weight slab is multiplied by all of
// them, so that the expert's weights are read once per node. With TCM the slab is copied there, with the expert's
// quantized rows beside it where they fit; without TCM it is read where it lies, with the same kernel calls on the same
// bytes. One token (ne12 == 1) keeps the 1-row path.
// The ids of a case are built from a wanted row count per expert, so one node has experts with 0, 1, 2, 3, 4, 5, 7, 8,
// 9 and 40 rows (groups of 4 with and without a tail, and experts no token uses), in that order and shuffled. Per
// case: with TCM and without bit for bit, the two steps called directly on 1 and on 8 tiles likewise, every output
// against the CPU backend on one thread within MUL_MAT's bound. TCM: a stand-in of the K3's 384 KiB, as check_gemv and
// check_mm_paths get one; the direct calls get a buffer of their own, zeroed first, so that a staged call must leave
// bytes in it
// weights quantized from random normal values, with the other checks' uniform generator through a Box-Muller
// transform: a real weight's values are near normal
static std::vector<uint8_t> quantize_normal_weights(ggml_type type, int64_t k, int64_t n, uint32_t seed) {
    const size_t       count = (size_t) (k * n);
    std::vector<float> u(count + 1);
    random_stream(seed).fill(u.data(), (int64_t) u.size(), 1.0f);  // uniform in [-1, 1)
    std::vector<float> v(count);
    for (size_t i = 0; i < count; i += 2) {
        const float u1 = std::max(1e-7f, 0.5f * (u[i] + 1.0f));  // (0, 1)
        const float u2 = 0.5f * (u[i + 1] + 1.0f);
        const float r  = std::sqrt(-2.0f * std::log(u1));
        v[i]           = r * std::cosf(6.2831853f * u2);
        if (i + 1 < count) {
            v[i + 1] = r * std::sinf(6.2831853f * u2);
        }
    }
    std::vector<uint8_t> q(ggml_row_size(type, k) * (size_t) n);
    ggml_quantize_chunk(type, v.data(), q.data(), 0, n, k, nullptr);
    return q;
}

// MUL_MAT_ID with 3-D Q4_1 experts in the q4_1 min layout (the model's five blk.{0..4}.ffn_down_exps of
// Qwen3.6-35B-A3B). The layout is lossless, so both comparisons carry the
// bound: against the CPU on the weight's read-back (what the model and test-backend-ops see) and on the GGUF bytes.
// Checked: with TCM and without against each other and the steps directly on 1 and on 8 tiles,
// experts of 0 to 40 rows, one token and ten, broadcast activations and a row per id, the model's down shape with 4
// experts, repack-unpack-repack, and that the read-back is the GGUF bytes
static int check_mul_mat_id_q4_1_min(ggml_backend_dev_t dev, bool direct, ggml_backend_t cpu,
                                     const spacemit_plugin_kernel * step_q, const spacemit_plugin_kernel * step_g) {
    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    struct q41_case {
        const char *         what;
        int64_t              k, n, n_used, ne11, n_tok;
        std::vector<int64_t> counts;  // the rows wanted per expert; its size is the expert count
    };
    // 10 tokens of 8 ids: 80 pairs over 8 experts, one unused. Expert 7's 40 pairs fill several tokens' slots, so an
    // id repeats within a token
    const std::vector<int64_t> many  = { 0, 1, 3, 4, 5, 8, 19, 40 };
    const std::vector<int64_t> one   = { 0, 2, 1, 0, 3, 0, 0, 2 };  // 1 token of 8 ids, two experts repeated
    const std::vector<int64_t> model = { 0, 4, 7, 9 };              // 5 tokens of 4 ids
    const std::vector<q41_case> cases = {
        { "experts of 0-40 rows, one slab", 256, 32, 8, 1, 10, many },
        { "experts of 0-40 rows, 8 slabs", 256, 256, 8, 8, 10, many },
        { "experts of 0-40 rows, a row per id", 256, 32, 8, 8, 10, many },
        { "one token", 256, 96, 8, 1, 1, one },
        { "one token, a row per id", 256, 32, 8, 8, 1, one },
        { "the model's down shape", 512, 2048, 4, 4, 5, model },
        { "the model's down shape, broadcast", 512, 2048, 4, 1, 5, model },
    };

    // the case's ids: each expert e repeated counts[e] times, filling the tokens' slots in order
    const auto id_list = [](const q41_case & c) {
        std::vector<int32_t> list;
        for (size_t e = 0; e < c.counts.size(); e++) {
            list.insert(list.end(), (size_t) c.counts[e], (int32_t) e);
        }
        return list;
    };

    int      n_cases = 0, n_ok = 0, n_same = 0, n_stable = 0, n_bits = 0;
    double   worst = 0.0, worst_gguf = 0.0;
    uint32_t seed = 4100;
    for (const q41_case & c : cases) {
        const int64_t              n_mats = (int64_t) c.counts.size();
        const std::vector<int32_t> ids_in = id_list(c);
        REQUIRE((int64_t) ids_in.size() == c.n_used * c.n_tok);
        ggml_init_params pw     = { ggml_tensor_overhead() * 4, nullptr, true };
        ggml_context *   ctx_wd = ggml_init(pw);
        ggml_context *   ctx_wc = ggml_init(pw);
        REQUIRE(ctx_wd != nullptr && ctx_wc != nullptr);
        ggml_tensor * w_dev = ggml_new_tensor_3d(ctx_wd, GGML_TYPE_Q4_1, c.k, c.n, n_mats);
        ggml_tensor * w_cpu = ggml_new_tensor_3d(ctx_wc, GGML_TYPE_Q4_1, c.k, c.n, n_mats);
        ggml_tensor * w_rb  = ggml_new_tensor_3d(ctx_wc, GGML_TYPE_Q4_1, c.k, c.n, n_mats);
        ggml_set_name(w_dev, "blk.0.ffn_down_exps.weight");
        ggml_backend_buffer_t buf_wd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wd, buft);
        ggml_backend_buffer_t buf_wc = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wc, ggml_backend_cpu_buffer_type());
        REQUIRE(buf_wd != nullptr && buf_wc != nullptr);
        const std::vector<uint8_t> q = quantize_normal_weights(GGML_TYPE_Q4_1, c.k, c.n * n_mats, seed++);
        std::vector<uint8_t>       read(q.size()), read2(q.size());
        ggml_backend_tensor_set(w_dev, q.data(), 0, q.size());
        ggml_backend_tensor_get(w_dev, read.data(), 0, read.size());
        ggml_backend_tensor_set(w_dev, read.data(), 0, read.size());  // repack, unpack, repack: the same bytes
        ggml_backend_tensor_get(w_dev, read2.data(), 0, read2.size());
        const bool stable = read2 == read;
        const bool bits   = read == q;
        ggml_backend_tensor_set(w_cpu, q.data(), 0, q.size());
        ggml_backend_tensor_set(w_rb, read.data(), 0, read.size());

        ggml_init_params p     = { ggml_tensor_overhead() * 8 + ggml_graph_overhead() * 2, nullptr, true };
        ggml_context *   ctx_h = ggml_init(p);
        ggml_context *   ctx_d = ggml_init(p);
        REQUIRE(ctx_h != nullptr && ctx_d != nullptr);
        ggml_tensor * x     = ggml_new_tensor_3d(ctx_h, GGML_TYPE_F32, c.k, c.ne11, c.n_tok);
        ggml_tensor * ids   = ggml_new_tensor_2d(ctx_h, GGML_TYPE_I32, c.n_used, c.n_tok);
        ggml_tensor * y_ref = ggml_mul_mat_id(ctx_h, w_rb, x, ids);
        ggml_tensor * y_q41 = ggml_mul_mat_id(ctx_h, w_cpu, x, ids);
        ggml_tensor * y     = ggml_mul_mat_id(ctx_d, w_dev, x, ids);
        ggml_backend_buffer_t bh = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
        ggml_backend_buffer_t bd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, buft);
        REQUIRE(bh != nullptr && bd != nullptr);
        const std::vector<float> xv = random_values(ggml_nelements(x), seed++, 2.0f);
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(ids, ids_in.data(), 0, ids_in.size() * sizeof(int32_t));

        const int64_t            n_out = ggml_nelements(y);
        const std::vector<float> junk((size_t) n_out, NAN);
        ggml_cgraph *            g_cpu = ggml_new_graph(ctx_h);
        ggml_build_forward_expand(g_cpu, y_ref);
        ggml_build_forward_expand(g_cpu, y_q41);
        ggml_cgraph * g_dev = ggml_new_graph(ctx_d);
        ggml_build_forward_expand(g_dev, y);
        bool ok = ggml_backend_dev_supports_op(dev, y) && ggml_backend_graph_compute(cpu, g_cpu) == GGML_STATUS_SUCCESS;

        const auto run = [&](ggml_backend_t backend) {
            ggml_backend_tensor_set(y, junk.data(), 0, ggml_nbytes(y));
            ok = ok && ggml_backend_graph_compute(backend, g_dev) == GGML_STATUS_SUCCESS;
            std::vector<float> out((size_t) n_out, NAN);
            ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
            return out;
        };
        std::vector<float> out, staged_plain;
        {
            ggml_backend_t backend = tcm_backend(dev, nullptr);
            REQUIRE(backend != nullptr);
            out = run(backend);
            ggml_backend_free(backend);
        }
        {  // without TCM the slabs are read where they lie
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(backend != nullptr);
            staged_plain = run(backend);
            ggml_backend_free(backend);
        }

        // the two steps directly on 1 and on 8 tiles, with a TCM buffer of their own and without
        const size_t         ws = step_q->workspace_size(y);
        std::vector<uint8_t> workspace(ws);
        std::vector<uint8_t> tcm(393216);
        std::vector<float>   d_out[4];
        for (int r = 0; r < 4 && direct; r++) {
            const uint32_t nth    = (r % 2) == 0 ? 1u : 8u;
            const bool     staged = r < 2;
            ggml_backend_tensor_set(y, junk.data(), 0, ggml_nbytes(y));
            for (const spacemit_plugin_kernel * step : { step_q, step_g }) {
                for (uint32_t ith = 0; ith < nth; ith++) {
                    const spacemit_tile tile = { ith,  nth, staged ? tcm.data() : nullptr, staged ? tcm.size() : 0,
                                                 workspace.data(), ws, nullptr };
                    ok &= step->fn(&tile, y);
                }
            }
            d_out[r].assign((size_t) n_out, NAN);
            ggml_backend_tensor_get(y, d_out[r].data(), 0, ggml_nbytes(y));
        }

        const size_t out_bytes = (size_t) n_out * sizeof(float);
        bool         same      = ok && std::memcmp(out.data(), staged_plain.data(), out_bytes) == 0;
        for (int r = 0; r < 4 && direct && same; r++) {
            same = std::memcmp(out.data(), d_out[r].data(), out_bytes) == 0;
        }
        const double e        = ok ? nmse(out.data(), (const float *) y_ref->data, n_out) : -1.0;
        const double e_gguf   = ok ? nmse(out.data(), (const float *) y_q41->data, n_out) : -1.0;
        const bool   in_bound = e >= 0.0 && e_gguf >= 0.0 && e <= k_mul_mat_max_nmse && e_gguf <= k_mul_mat_max_nmse;
        const bool   case_ok  = ok && same && in_bound && stable && bits;
        n_cases++;
        n_ok += case_ok;
        n_same += same;
        n_stable += stable;
        n_bits += bits;
        worst      = std::max(worst, e);
        worst_gguf = std::max(worst_gguf, e_gguf);
        if (!case_ok) {
            std::fprintf(stderr,
                         "FAIL: MUL_MAT_ID Q4_1, %s, [%lld, %lld, %lld], %lld ids, %lld rows per token, %lld "
                         "tokens: %s; with TCM, without and the tile counts %s; repack-unpack-repack %s; the "
                         "read-back %s the GGUF bytes; NMSE against the read-back %.3e, against the GGUF bytes %.3e "
                         "(bound %.0e)\n",
                         c.what, (long long) c.k, (long long) c.n, (long long) n_mats, (long long) c.n_used,
                         (long long) c.ne11, (long long) c.n_tok, ok ? "ran" : "failed", same ? "equal" : "differ",
                         stable ? "stable" : "changed the bytes", bits ? "is" : "is not", e, e_gguf,
                         k_mul_mat_max_nmse);
            g_failures++;
        }
        ggml_backend_buffer_free(bh);
        ggml_backend_buffer_free(bd);
        ggml_free(ctx_h);
        ggml_free(ctx_d);
        ggml_backend_buffer_free(buf_wd);
        ggml_backend_buffer_free(buf_wc);
        ggml_free(ctx_wd);
        ggml_free(ctx_wc);
    }

    std::printf("mmid     3-D Q4_1 experts (the q4_1 min layout, the default) at experts of 0, 1, 3, 4, 5, 8, 19 and "
                "40 rows a node, 1 and 8 column slabs, broadcast and a row per id, 10 tokens and 1, the model's down "
                "shape [512, 2048, 4]: %d/%d cases; with TCM and without%s bit for bit in %d/%d; repack-unpack-repack "
                "keeps the bytes in %d/%d; the read-back is the GGUF bytes in %d/%d; max NMSE against the CPU on one "
                "thread computing on the read-back weights %.2e, on the GGUF bytes %.2e (bound %.0e)%s\n",
                n_ok, n_cases, direct ? ", with the steps directly on 1 and on 8 tiles" : "", n_same, n_cases,
                n_stable, n_cases, n_bits, n_cases, worst, worst_gguf, k_mul_mat_max_nmse, reference_label());
    return 0;
}

// the claims of 3-D Q4_1 weights, and their layout (check_mul_mat_id_q4_1_min)
static int check_mul_mat_id_q4_1(ggml_backend_dev_t dev, bool host) {
    // the direct calls run on the calling core: on the K3 that is an X100 core, which has no IME instructions
    const bool     direct = host || *reference_label() != '\0';
    ggml_backend_t cpu    = ggml_backend_cpu_init();
    REQUIRE(cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);  // a correctness reference: single-threaded (PROJECT.md item 28)

    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    REQUIRE(kernels != nullptr);
    const spacemit_plugin_kernel * step_q = nullptr, * step_g = nullptr;
    for (uint32_t i = 0; i < n_kernels; i++) {
        if (kernels[i].name != nullptr && std::strcmp(kernels[i].name, "mul_mat_id.quantize") == 0) {
            step_q = &kernels[i];
        }
        if (kernels[i].name != nullptr && std::strcmp(kernels[i].name, "mul_mat_id.gemm") == 0) {
            step_g = &kernels[i];
        }
    }
    REQUIRE(step_q != nullptr && step_g != nullptr && step_q->workspace_size != nullptr);

    // a 3-D Q4_1 weight has a layout and MUL_MAT_ID takes it; 2-D Q4_1 keeps its Q8_0 route. The row rule stays
    // (N % 32), and K must be a multiple of 256
    int n_claims = 0, n_claims_ok = 0;
    {
        ggml_init_params p   = { ggml_tensor_overhead() * 32, nullptr, true };
        ggml_context *   ctx = ggml_init(p);
        REQUIRE(ctx != nullptr);
        const auto expect = [&](bool wanted, ggml_tensor * node, const char * what) {
            n_claims++;
            const bool got = ggml_backend_dev_supports_op(dev, node);
            n_claims_ok += got == wanted;
            if (got != wanted) {
                std::fprintf(stderr, "FAIL: MUL_MAT_ID Q4_1 claims: %s: %s, expected %s\n", what,
                             got ? "claimed" : "refused", wanted ? "claimed" : "refused");
                g_failures++;
            }
        };
        ggml_tensor * x1   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 512);
        ggml_tensor * ids2 = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 512);
        ggml_tensor * w    = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_1, 256, 64, 4);
        ggml_tensor * w2d  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, 256, 64);
        expect(true, ggml_mul_mat_id(ctx, w, x1, ids2), "a 3-D Q4_1 weight");
        expect(true, ggml_mul_mat(ctx, w2d, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 7)), "MUL_MAT, 2-D Q4_1");
        n_claims += 2;
        n_claims_ok += table->tensor_layout(w) != 0;
        n_claims_ok += table->tensor_layout(w2d) != 0;  // 2-D Q4_1: the Q8_0 route
        if (table->tensor_layout(w) == 0 || table->tensor_layout(w2d) == 0) {
            std::fprintf(stderr, "FAIL: MUL_MAT_ID Q4_1 claims: the 3-D layout is %u (wanted a layout), the 2-D layout "
                         "%u\n", table->tensor_layout(w), table->tensor_layout(w2d));
            g_failures++;
        }
        expect(false,
               ggml_mul_mat_id(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_1, 128, 64, 4),
                               ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 1, 512), ids2),
               "Q4_1 rows of 128 (not a multiple of 256)");
        expect(false, ggml_mul_mat_id(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_1, 256, 48, 4), x1, ids2),
               "48 rows per expert (not a multiple of 32)");
        ggml_free(ctx);
    }

    check_mul_mat_id_q4_1_min(dev, direct, cpu, step_q, step_g);

    ggml_backend_free(cpu);
    std::printf("mmid     claims of 3-D Q4_1 (a 3-D Q4_1 weight has a layout and MUL_MAT_ID takes it; 2-D Q4_1 keeps "
                "its layout and MUL_MAT; rows per expert not a multiple of 32 and K not a multiple of 256 refused) "
                "%d/%d\n",
                n_claims_ok, n_claims);
    return 0;
}

static int check_mul_mat_id_groups(ggml_backend_dev_t dev, bool host) {
    // the direct calls run on the calling core: on the K3 that is an X100 core, which has no IME instructions
    const bool     direct = host || *reference_label() != '\0';
    ggml_backend_t cpu    = ggml_backend_cpu_init();
    REQUIRE(cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);  // a correctness reference: single-threaded (PROJECT.md item 28)
    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    REQUIRE(kernels != nullptr);
    const spacemit_plugin_kernel * step_q = nullptr, * step_g = nullptr;
    for (uint32_t i = 0; i < n_kernels; i++) {
        if (kernels[i].name != nullptr && std::strcmp(kernels[i].name, "mul_mat_id.quantize") == 0) {
            step_q = &kernels[i];
        }
        if (kernels[i].name != nullptr && std::strcmp(kernels[i].name, "mul_mat_id.gemm") == 0) {
            step_g = &kernels[i];
        }
    }
    REQUIRE(step_q != nullptr && step_g != nullptr && step_q->workspace_size != nullptr);

    struct group_case {
        const char *         what;
        ggml_type            type;
        int64_t              k, n, n_used, ne11, n_tok;
        bool                 shuffled;
        std::vector<int64_t> counts;  // the rows wanted per expert; its size is the expert count
    };
    // 10 tokens of 8 ids: 80 pairs over 12 experts, two of them unused. Expert 9's 40 pairs fill several tokens'
    // slots, so an id repeats within a token whatever the order
    const std::vector<int64_t> many = { 0, 1, 2, 3, 4, 5, 7, 8, 9, 40, 1, 0 };
    const std::vector<int64_t> one  = { 0, 2, 1, 0, 3, 0, 0, 2, 0, 0, 0, 0 };  // 1 token of 8 ids, two repeated
    std::vector<group_case>    cases;
    for (const ggml_type type : { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0 }) {
        for (const int64_t n : { (int64_t) 32, (int64_t) 96 }) {  // one slab, and three
            for (const int64_t ne11 : { (int64_t) 1, (int64_t) 8 }) {
                for (const bool shuffled : { false, true }) {
                    cases.push_back({ shuffled ? "0-40 rows an expert, the ids shuffled" : "0-40 rows an expert", type,
                                      256, n, 8, ne11, 10, shuffled, many });
                }
            }
        }
        cases.push_back({ "one token", type, 256, 96, 8, 1, 1, false, one });
        cases.push_back({ "one token, a row per id", type, 256, 32, 8, 8, 1, false, one });
    }
    // 8 pairs of 9728 values: the slab fits in 384 KiB of TCM (330,752 bytes), an expert's quantized rows do not
    cases.push_back({ "a long row: the rows do not fit beside the slab", GGML_TYPE_Q8_0, 9728, 32, 4, 4, 2, false,
                      { 0, 1, 2, 0, 1, 0, 3, 1, 0, 0, 0, 0 } });

    // the case's ids: each expert e repeated counts[e] times, filling the tokens' slots in order, shuffled first where
    // the case asks (a partial Fisher-Yates with the other checks' generator)
    const auto id_list = [](const group_case & c, uint32_t seed) {
        std::vector<int32_t> list;
        for (size_t e = 0; e < c.counts.size(); e++) {
            list.insert(list.end(), (size_t) c.counts[e], (int32_t) e);
        }
        if (c.shuffled) {
            uint32_t s = seed * 2654435761u + 1;
            for (size_t i = list.size(); i > 1; i--) {
                s = s * 1664525u + 1013904223u;
                std::swap(list[i - 1], list[(s >> 8) % i]);
            }
        }
        return list;
    };

    int      n_cases = 0, n_ok = 0, n_same = 0;
    double   worst = 0.0;
    uint32_t seed  = 2600;
    for (const group_case & c : cases) {
        const int64_t              n_mats = (int64_t) c.counts.size();
        const std::vector<int32_t> ids_in = id_list(c, seed);
        REQUIRE((int64_t) ids_in.size() == c.n_used * c.n_tok);
        ggml_init_params pw     = { ggml_tensor_overhead() * 4, nullptr, true };
        ggml_context *   ctx_wd = ggml_init(pw);
        ggml_context *   ctx_wc = ggml_init(pw);
        REQUIRE(ctx_wd != nullptr && ctx_wc != nullptr);
        ggml_tensor * w_dev = ggml_new_tensor_3d(ctx_wd, c.type, c.k, c.n, n_mats);
        ggml_tensor * w_cpu = ggml_new_tensor_3d(ctx_wc, c.type, c.k, c.n, n_mats);
        ggml_set_name(w_dev, "blk.0.ffn_down_exps.weight");
        ggml_backend_buffer_t buf_wd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wd, buft);
        ggml_backend_buffer_t buf_wc = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wc, ggml_backend_cpu_buffer_type());
        REQUIRE(buf_wd != nullptr && buf_wc != nullptr);
        const std::vector<uint8_t> q = quantize_weights(c.type, c.k, c.n * n_mats, seed++);
        ggml_backend_tensor_set(w_dev, q.data(), 0, q.size());
        ggml_backend_tensor_set(w_cpu, q.data(), 0, q.size());

        ggml_init_params p     = { ggml_tensor_overhead() * 8 + ggml_graph_overhead() * 2, nullptr, true };
        ggml_context *   ctx_h = ggml_init(p);
        ggml_context *   ctx_d = ggml_init(p);
        REQUIRE(ctx_h != nullptr && ctx_d != nullptr);
        ggml_tensor * x       = ggml_new_tensor_3d(ctx_h, GGML_TYPE_F32, c.k, c.ne11, c.n_tok);
        ggml_tensor * ids     = ggml_new_tensor_2d(ctx_h, GGML_TYPE_I32, c.n_used, c.n_tok);
        ggml_tensor * y_ref   = ggml_mul_mat_id(ctx_h, w_cpu, x, ids);
        ggml_tensor * y       = ggml_mul_mat_id(ctx_d, w_dev, x, ids);
        ggml_backend_buffer_t bh = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
        ggml_backend_buffer_t bd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, buft);
        REQUIRE(bh != nullptr && bd != nullptr);
        const std::vector<float> xv = random_values(ggml_nelements(x), seed++, 2.0f);
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(ids, ids_in.data(), 0, ids_in.size() * sizeof(int32_t));

        const int64_t            n_out = ggml_nelements(y);
        const std::vector<float> junk((size_t) n_out, NAN);
        ggml_cgraph *            g_cpu = ggml_new_graph(ctx_h);
        ggml_build_forward_expand(g_cpu, y_ref);
        ggml_cgraph * g_dev = ggml_new_graph(ctx_d);
        ggml_build_forward_expand(g_dev, y);
        bool ok = ggml_backend_dev_supports_op(dev, y) && ggml_backend_graph_compute(cpu, g_cpu) == GGML_STATUS_SUCCESS;

        // through the backend, the output filled with NaN first so an element no tile writes shows
        const auto run = [&](ggml_backend_t backend) {
            ggml_backend_tensor_set(y, junk.data(), 0, ggml_nbytes(y));
            ok = ok && ggml_backend_graph_compute(backend, g_dev) == GGML_STATUS_SUCCESS;
            std::vector<float> out((size_t) n_out, NAN);
            ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
            return out;
        };
        std::vector<float> out, staged_plain, staged_threads;
        {
            ggml_backend_t backend = tcm_backend(dev, nullptr);
            REQUIRE(backend != nullptr);
            out = run(backend);
            ggml_backend_free(backend);
        }
        if (host) {  // the tiles on threads, each staging into a TCM stand-in of its own
            ggml_backend_t backend = tcm_backend(dev, "1");
            REQUIRE(backend != nullptr);
            staged_threads = run(backend);
            ggml_backend_free(backend);
        }
        {  // without TCM the slabs are read where they lie
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(backend != nullptr);
            staged_plain = run(backend);
            ggml_backend_free(backend);
        }

        // the two steps directly on 1 and on 8 tiles, with a TCM buffer of their own and without
        const size_t         ws = step_q->workspace_size(y);
        std::vector<uint8_t> workspace(ws);
        std::vector<uint8_t> tcm(393216);
        std::vector<float>   d_out[4];
        bool                 tcm_used = true;
        for (int r = 0; r < 4 && direct; r++) {
            const uint32_t nth    = (r % 2) == 0 ? 1u : 8u;
            const bool     staged = r < 2;
            std::fill(tcm.begin(), tcm.end(), uint8_t(0));
            ggml_backend_tensor_set(y, junk.data(), 0, ggml_nbytes(y));
            for (const spacemit_plugin_kernel * step : { step_q, step_g }) {
                for (uint32_t ith = 0; ith < nth; ith++) {
                    const spacemit_tile tile = { ith,  nth, staged ? tcm.data() : nullptr, staged ? tcm.size() : 0,
                                                 workspace.data(), ws, nullptr };
                    ok &= step->fn(&tile, y);
                }
            }
            d_out[r].assign((size_t) n_out, NAN);
            ggml_backend_tensor_get(y, d_out[r].data(), 0, ggml_nbytes(y));
            if (staged && c.n_tok > 1) {  // a staged call copies at least one slab into TCM
                tcm_used = tcm_used && std::any_of(tcm.begin(), tcm.end(), [](uint8_t b) { return b != 0; });
            }
        }

        // with TCM and without, on 8 tiles and on 1: one kernel call per (group, slab) either way
        const size_t out_bytes = (size_t) n_out * sizeof(float);
        bool         same      = ok && std::memcmp(out.data(), staged_plain.data(), out_bytes) == 0 && tcm_used &&
                     (!host || std::memcmp(out.data(), staged_threads.data(), out_bytes) == 0);
        for (int r = 0; r < 4 && direct && same; r++) {
            same = std::memcmp(out.data(), d_out[r].data(), out_bytes) == 0;
        }
        const double e       = ok ? nmse(out.data(), (const float *) y_ref->data, n_out) : -1.0;
        const bool   case_ok = ok && same && e >= 0.0 && e <= k_mul_mat_max_nmse;
        n_cases++;
        n_ok += case_ok;
        n_same += same;
        worst = std::max(worst, e);
        if (!case_ok) {
            std::fprintf(stderr,
                         "FAIL: MUL_MAT_ID many rows, %s, %s [%lld, %lld, %lld], %lld ids, %lld rows per token, %lld "
                         "tokens: %s; with TCM, without and the tile counts %s; NMSE %.3e (bound %.0e)%s\n",
                         c.what, ggml_type_name(c.type), (long long) c.k, (long long) c.n, (long long) n_mats,
                         (long long) c.n_used, (long long) c.ne11, (long long) c.n_tok, ok ? "ran" : "failed",
                         same ? "equal" : "differ", e, k_mul_mat_max_nmse,
                         tcm_used ? "" : ", a staged run left TCM untouched");
            g_failures++;
        }
        ggml_backend_buffer_free(bh);
        ggml_backend_buffer_free(bd);
        ggml_free(ctx_h);
        ggml_free(ctx_d);
        ggml_backend_buffer_free(buf_wd);
        ggml_backend_buffer_free(buf_wc);
        ggml_free(ctx_wd);
        ggml_free(ctx_wc);
    }

    ggml_backend_free(cpu);
    std::printf("mmid     the many-row path (the pairs grouped by expert, the slabs staged through TCM) at experts of "
                "0, 1, 2, 3, 4, 5, 7, 8, 9 and 40 rows a node, ids designed and shuffled with repeats within a token, "
                "Q4_0 and Q8_0, 1 and 3 column slabs, broadcast and a row per id, 10 tokens and 1, a long row whose "
                "rows do not fit beside the slab: %d/%d cases; with TCM%s and without TCM bit for bit%s in %d/%d; "
                "against the CPU on one thread max NMSE %.2e (bound %.0e)%s\n",
                n_ok, n_cases, host ? " (tiles in turn and on threads)" : "",
                direct ? ", with the steps directly on 1 and on 8 tiles" : "", n_same, n_cases, worst,
                k_mul_mat_max_nmse, reference_label());
    return 0;
}

// the many-row path's claim queue (plan.md "The expert matmul's claim queue"). Every output element comes from one
// kernel call whoever claims its chunk, so the bytes are the first backend run's (with TCM) for any tile count, claim
// order and staging. Through the backend: without TCM, twice in a row on one backend (the second compute
// needs step 1's reset), on threads 20 times. Directly: on 1 tile (it claims every chunk in order) and 8, in reverse,
// one tile first, on threads 20 times; after each the counter (at the workspace's end, which the test aligns to 64)
// holds the chunk count plus the tile count, each tile's last claim being past the end. The two steps' labels in the
// op counters
static int check_mul_mat_id_queue(ggml_backend_dev_t dev, bool host) {
    const bool     direct = host || *reference_label() != '\0';
    ggml_backend_t cpu    = ggml_backend_cpu_init();
    REQUIRE(cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);
    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    REQUIRE(kernels != nullptr);
    const spacemit_plugin_kernel * step_q = nullptr, * step_g = nullptr;
    for (uint32_t i = 0; i < n_kernels; i++) {
        if (kernels[i].name != nullptr && std::strcmp(kernels[i].name, "mul_mat_id.quantize") == 0) {
            step_q = &kernels[i];
        }
        if (kernels[i].name != nullptr && std::strcmp(kernels[i].name, "mul_mat_id.gemm") == 0) {
            step_g = &kernels[i];
        }
    }
    REQUIRE(step_q != nullptr && step_g != nullptr && step_q->workspace_size != nullptr);

    struct queue_case {
        const char *         what;
        ggml_type            type;
        int64_t              n, n_used, ne11, n_tok;
        bool                 shuffled;
        std::vector<int64_t> counts;  // the rows wanted per expert; its size is the expert count
    };
    // the skewed draw (half the pairs on expert 1) cuts its experts into chunks of 1 to 4 slabs on 8 tiles
    const std::vector<int64_t>    skew  = { 1, 160, 1, 2, 30, 0, 50, 4, 72 };
    const std::vector<queue_case> cases = {
        { "every pair on one expert", GGML_TYPE_Q4_0, 256, 4, 1, 6, false, { 0, 0, 24, 0 } },
        { "one row an expert", GGML_TYPE_Q4_0, 256, 2, 2, 8, false, std::vector<int64_t>(16, 1) },
        { "a skewed draw", GGML_TYPE_Q4_0, 256, 8, 1, 40, true, skew },
        { "a skewed draw, Q8_0 and a row per id", GGML_TYPE_Q8_0, 96, 8, 8, 40, true, skew },
        { "fewer chunks than tiles", GGML_TYPE_Q4_0, 32, 1, 1, 2, false, { 0, 1, 0, 1 } },
        { "experts of 0 rows between active ones", GGML_TYPE_Q4_0, 256, 2, 1, 8, false,
         { 0, 3, 0, 0, 5, 0, 0, 0, 2, 0, 6, 0 } },
    };
    const int64_t  k        = 256;
    const size_t   tcm_size = 393216;  // the K3's 384 KiB
    const uint32_t k_runs   = 20;      // the runs with the tiles at the same time

    // the queue's chunk count as the plugin cuts it (slab_weight 4 and mmid_chunks 16, flagos-spacemit-ime-id.cpp)
    const auto chunk_count = [](const std::vector<int64_t> & counts, int64_t per, int64_t nth) {
        int64_t total = 0;
        for (const int64_t r : counts) {
            total += r > 0 ? per * (r + 4) : 0;
        }
        const int64_t target = std::max<int64_t>(1, total / (nth * 16));
        int64_t       n      = 0;
        for (const int64_t r : counts) {
            if (r > 0) {
                const int64_t c = std::clamp(target / (r + 4), int64_t(1), per);
                n += (per + c - 1) / c;
            }
        }
        return n;
    };
    const auto id_list = [](const queue_case & c, uint32_t seed) {
        std::vector<int32_t> list;
        for (size_t e = 0; e < c.counts.size(); e++) {
            list.insert(list.end(), (size_t) c.counts[e], (int32_t) e);
        }
        if (c.shuffled) {
            uint32_t s = seed * 2654435761u + 1;
            for (size_t i = list.size(); i > 1; i--) {
                s = s * 1664525u + 1013904223u;
                std::swap(list[i - 1], list[(s >> 8) % i]);
            }
        }
        return list;
    };

    int      n_cases = 0, n_ok = 0, n_backend = 0, n_backend_ok = 0, n_direct = 0, n_direct_ok = 0;
    int      n_design = 0, n_design_ok = 0;
    double   worst = 0.0;
    uint32_t seed = 3100;
    for (size_t ci = 0; ci < cases.size(); ci++) {
        const queue_case &         c      = cases[ci];
        const int64_t              n_mats = (int64_t) c.counts.size();
        const int64_t              per    = c.n / 32;
        const std::vector<int32_t> ids_in = id_list(c, seed);
        REQUIRE((int64_t) ids_in.size() == c.n_used * c.n_tok);
        ggml_init_params pw     = { ggml_tensor_overhead() * 4, nullptr, true };
        ggml_context *   ctx_wd = ggml_init(pw);
        ggml_context *   ctx_wc = ggml_init(pw);
        REQUIRE(ctx_wd != nullptr && ctx_wc != nullptr);
        ggml_tensor * w_dev = ggml_new_tensor_3d(ctx_wd, c.type, k, c.n, n_mats);
        ggml_tensor * w_cpu = ggml_new_tensor_3d(ctx_wc, c.type, k, c.n, n_mats);
        ggml_set_name(w_dev, "blk.0.ffn_down_exps.weight");
        ggml_backend_buffer_t buf_wd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wd, buft);
        ggml_backend_buffer_t buf_wc = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wc, ggml_backend_cpu_buffer_type());
        REQUIRE(buf_wd != nullptr && buf_wc != nullptr);
        const std::vector<uint8_t> q = quantize_weights(c.type, k, c.n * n_mats, seed++);
        ggml_backend_tensor_set(w_dev, q.data(), 0, q.size());
        ggml_backend_tensor_set(w_cpu, q.data(), 0, q.size());

        ggml_init_params p     = { ggml_tensor_overhead() * 8 + ggml_graph_overhead() * 2, nullptr, true };
        ggml_context *   ctx_h = ggml_init(p);
        ggml_context *   ctx_d = ggml_init(p);
        REQUIRE(ctx_h != nullptr && ctx_d != nullptr);
        ggml_tensor *         x     = ggml_new_tensor_3d(ctx_h, GGML_TYPE_F32, k, c.ne11, c.n_tok);
        ggml_tensor *         ids   = ggml_new_tensor_2d(ctx_h, GGML_TYPE_I32, c.n_used, c.n_tok);
        ggml_tensor *         y_ref = ggml_mul_mat_id(ctx_h, w_cpu, x, ids);
        ggml_tensor *         y     = ggml_mul_mat_id(ctx_d, w_dev, x, ids);
        ggml_backend_buffer_t bh    = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
        ggml_backend_buffer_t bd    = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, buft);
        REQUIRE(bh != nullptr && bd != nullptr);
        const std::vector<float> xv = random_values(ggml_nelements(x), seed++, 2.0f);
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(ids, ids_in.data(), 0, ids_in.size() * sizeof(int32_t));

        const int64_t            n_out = ggml_nelements(y);
        const size_t             bytes = (size_t) n_out * sizeof(float);
        const std::vector<float> junk((size_t) n_out, NAN);
        ggml_cgraph *            g_cpu = ggml_new_graph(ctx_h);
        ggml_build_forward_expand(g_cpu, y_ref);
        ggml_cgraph * g_dev = ggml_new_graph(ctx_d);
        ggml_build_forward_expand(g_dev, y);
        bool ok = ggml_backend_dev_supports_op(dev, y) && ggml_backend_graph_compute(cpu, g_cpu) == GGML_STATUS_SUCCESS;

        std::string bad;  // the comparisons that failed
        const auto  note = [&](bool good, const char * what) {
            if (!good) {
                bad += bad.empty() ? "" : ", ";
                bad += what;
            }
            return good;
        };
        const auto same = [&](const std::vector<float> & a, const std::vector<float> & b) {
            return a.size() == b.size() && std::memcmp(a.data(), b.data(), bytes) == 0;
        };
        // through the backend, the output filled with NaN first so an element no tile writes shows
        const auto run = [&](ggml_backend_t backend) {
            ggml_backend_tensor_set(y, junk.data(), 0, ggml_nbytes(y));
            ok = ok && ggml_backend_graph_compute(backend, g_dev) == GGML_STATUS_SUCCESS;
            std::vector<float> out((size_t) n_out, NAN);
            ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
            return out;
        };
        const auto backend_same = [&](bool good, const char * what) {
            n_backend++;
            n_backend_ok += note(good, what);
        };

        // the design: one expert's chunks are single slabs; fewer chunks than tiles
        if (ci == 0 || ci == 4) {
            const int64_t n8 = chunk_count(c.counts, per, 8);
            const bool    as = ci == 0 ? n8 == per : n8 < 8;
            n_design++;
            n_design_ok += note(as, ci == 0 ? "not single-slab chunks" : "not fewer chunks than tiles");
        }

        std::vector<float> ref;
        {
            ggml_backend_t backend = tcm_backend(dev, nullptr);
            REQUIRE(backend != nullptr);
            ref = run(backend);
            ggml_backend_free(backend);
        }
        {  // without TCM, and the same graph twice on one backend: the second compute finds the first's counter
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(backend != nullptr);
            const std::vector<float> first  = run(backend);
            const std::vector<float> second = run(backend);
            backend_same(same(first, ref), "queue without TCM");
            backend_same(same(second, ref), "the second compute in a row");
            ggml_backend_free(backend);
        }
        if (host) {  // the tiles on threads, each staging into a TCM stand-in of its own
            ggml_backend_t backend = tcm_backend(dev, "1");
            REQUIRE(backend != nullptr);
            bool all = true;
            for (uint32_t i = 0; i < k_runs; i++) {
                all = same(run(backend), ref) && all;
            }
            backend_same(all, "queue on threads");
            ggml_backend_free(backend);
        }

        // the two steps directly. The workspace is kept from run to run, so every run after the first needs the reset
        const size_t         ws = step_q->workspace_size(y);
        std::vector<uint8_t> raw(ws + 64);
        uint8_t * const      wsp = raw.data() + (64 - reinterpret_cast<uintptr_t>(raw.data()) % 64) % 64;
        const auto * const   counter = reinterpret_cast<const std::atomic<int64_t> *>(wsp + ws - 64);
        std::vector<uint8_t> tcm(8 * tcm_size);
        // order: 0 the tiles in turn, 1 in reverse, 2 tile nth / 2 first and then the others in turn, 3 on threads
        const auto direct_run = [&](uint32_t nth, int order, bool staged, int64_t & claims) {
            ggml_backend_tensor_set(y, junk.data(), 0, ggml_nbytes(y));
            std::atomic<bool> failed{ false };
            const auto        call = [&](const spacemit_plugin_kernel * step, uint32_t ith) {
                uint8_t * const   own  = tcm.data() + (order == 3 ? ith * tcm_size : 0);
                const spacemit_tile tile = { ith, nth, staged ? own : nullptr, staged ? tcm_size : 0, wsp, ws, nullptr };
                if (!step->fn(&tile, y)) {
                    failed = true;
                }
            };
            for (const spacemit_plugin_kernel * step : { step_q, step_g }) {
                if (order == 3) {
                    std::vector<std::thread> threads;
                    for (uint32_t ith = 0; ith < nth; ith++) {
                        threads.emplace_back([&, ith] { call(step, ith); });
                    }
                    for (std::thread & t : threads) {
                        t.join();
                    }
                    continue;
                }
                for (uint32_t i = 0; i < nth; i++) {
                    const uint32_t half = nth / 2;
                    call(step, order == 1 ? nth - 1 - i : order == 2 ? (i == 0 ? half : i <= half ? i - 1 : i) : i);
                }
            }
            claims = counter->load();
            ok     = ok && !failed;
            std::vector<float> out((size_t) n_out, NAN);
            ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
            return out;
        };
        struct direct_form {
            const char * what;
            uint32_t     nth;
            int          order;
            bool         staged;
            uint32_t     runs;
        };
        const std::vector<direct_form> forms = {
            { "1 tile", 1, 0, true, 1 },
            { "8 tiles", 8, 0, true, 1 },
            { "8 tiles in reverse", 8, 1, true, 1 },
            { "8 tiles, one first", 8, 2, true, 1 },
            { "8 tiles without TCM", 8, 0, false, 1 },
            { "8 tiles on threads", 8, 3, true, k_runs },
            { "8 tiles on threads without TCM", 8, 3, false, k_runs },
        };
        for (size_t f = 0; f < forms.size() && direct; f++) {
            const direct_form & form   = forms[f];
            const int64_t       want   = chunk_count(c.counts, per, form.nth) + form.nth;
            bool                good   = true;
            int64_t             claims = -1;
            for (uint32_t i = 0; i < form.runs; i++) {
                good = same(direct_run(form.nth, form.order, form.staged, claims), ref) && claims == want && good;
            }
            n_direct++;
            n_direct_ok += note(good, form.what);
            if (!good) {
                std::fprintf(stderr, "FAIL: MUL_MAT_ID queue, %s, directly on %s: counter %lld (wanted %lld)\n",
                             c.what, form.what, (long long) claims, (long long) want);
                g_failures++;
            }
        }

        const double e       = ok ? nmse(ref.data(), (const float *) y_ref->data, n_out) : -1.0;
        const bool   case_ok = ok && bad.empty() && e >= 0.0 && e <= k_mul_mat_max_nmse;
        n_cases++;
        n_ok += case_ok;
        worst = std::max(worst, e);
        if (!case_ok) {
            std::fprintf(stderr,
                         "FAIL: MUL_MAT_ID queue, %s, %s [%lld, %lld, %lld], %lld ids, %lld rows per token, %lld tokens: "
                         "%s; not the first run's bytes or not as designed: %s; NMSE %.3e (bound %.0e)\n",
                         c.what, ggml_type_name(c.type), (long long) k, (long long) c.n, (long long) n_mats,
                         (long long) c.n_used, (long long) c.ne11, (long long) c.n_tok, ok ? "ran" : "failed",
                         bad.empty() ? "none" : bad.c_str(), e, k_mul_mat_max_nmse);
            g_failures++;
        }
        ggml_backend_buffer_free(bh);
        ggml_backend_buffer_free(bd);
        ggml_free(ctx_h);
        ggml_free(ctx_d);
        ggml_backend_buffer_free(buf_wd);
        ggml_backend_buffer_free(buf_wc);
        ggml_free(ctx_wd);
        ggml_free(ctx_wc);
    }
    ggml_backend_free(cpu);

    // the op counters name the node's two steps by their contract names
    int n_labels = 0, n_old = 0;
    {
        const std::string path = std::string(FLAGOS_SPACEMIT_TEST_PLUGIN_DIR) + "/mmid-op-stats.jsonl";
        if (FILE * f = std::fopen(path.c_str(), "w")) {  // the destructor appends
            std::fclose(f);
        }
        ggml_init_params p   = { ggml_tensor_overhead() * 8, nullptr, true };
        ggml_context *   ctx = ggml_init(p);
        REQUIRE(ctx != nullptr);
        ggml_tensor * y = ggml_mul_mat_id(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, 256, 64, 4),
                                          ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 10),
                                          ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 10));
        const std::vector<spacemit_plugin_step> steps = { { step_q->fn, y, 0 }, { step_g->fn, y, 0 } };
        const std::vector<const char *>         names = { step_q->name, step_g->name };
        {
            spacemit_op_stats_file counters(path.c_str());
            spacemit_run_stats     run;
            run.reset(steps.size());
            run.n_steps = steps.size();
            run.n_tiles = 1;
            counters.add("per-split", 10, 0, steps, names, run);
        }  // the destructor writes
        ggml_free(ctx);
        int           quantize = 0, gemm = 0;
        std::ifstream written(path);
        std::string   line;
        while (std::getline(written, line)) {
            if (line.empty() || line[0] != '{' || line.find("\"first\":1") != std::string::npos) {
                continue;
            }
            quantize += line.find("\"label\":\"MUL_MAT_ID.quantize r20\"") != std::string::npos;
            gemm += line.find("\"label\":\"MUL_MAT_ID.gemm r20\"") != std::string::npos;
            n_old += line.find("\"label\":\"MUL_MAT_ID r") != std::string::npos;
        }
        n_labels = (quantize == 1) + (gemm == 1);
        if (n_labels != 2 || n_old != 0) {
            std::fprintf(stderr,
                         "FAIL: MUL_MAT_ID labels: %d MUL_MAT_ID.quantize, %d MUL_MAT_ID.gemm, %d unnamed steps (wanted "
                         "1, 1, 0; %s)\n",
                         quantize, gemm, n_old, path.c_str());
            g_failures++;
        }
    }

    std::printf("mmid     the claim queue (16 chunks a tile's share) at every pair on one expert, one row an expert, a "
                "skewed draw (Q4_0, and Q8_0 with a row per id), fewer chunks than tiles, experts of 0 rows between "
                "active ones: %d/%d cases; through the backend (without TCM, twice in a row%s) the first run's bytes "
                "in %d/%d; directly (1 and 8 tiles, in reverse, one tile first, without TCM%s) the first run's bytes "
                "and the counter at chunks + tiles in %d/%d; as designed %d/%d; the two steps' labels %d/2; against "
                "the CPU on one thread max NMSE %.2e (bound %.0e)%s\n",
                n_ok, n_cases, host ? ", on threads 20 times" : "", n_backend_ok, n_backend,
                direct ? ", on threads 20 times" : "", n_direct_ok, n_direct, n_design_ok, n_design, n_labels, worst,
                k_mul_mat_max_nmse, reference_label());
    return 0;
}

// the q4_0s kernels against each other on rows of equal magnitude, where a group's shared scale is each row's own and
// only the fp16 rounding differs (the references 7e-7 apart on the Mac; a wrong lane or scale gives about 1e-1), with
// room for the IME's own fp16 order; one output row against the CPU, which a padded row's zeros (NMSE 1) exceed
constexpr double k_q4_0s_kernels_max_nmse = 5e-5;
constexpr double k_q4_0s_row_max_nmse     = 1e-2;

// one MUL_MAT_ID node on 3-D Q4_0 experts for check_mul_mat_id_q4_0s: the weight in our buffer (its layout is decided
// by the FLAGOS_SPACEMIT_EXPERT_Q4_0 in force when it is made, so it is made inside the setting's scope) and the CPU
// backend's result on the dequantized weights in F32, computed once
struct q40s_node {
    ggml_context *        ctx_w = nullptr, * ctx_h = nullptr, * ctx_d = nullptr;
    ggml_backend_buffer_t buf_w = nullptr, buf_h = nullptr, buf_d = nullptr;
    ggml_tensor *         w = nullptr, * y = nullptr, * y_ref = nullptr;
    ggml_cgraph *         g_dev = nullptr;
    int64_t               n_out = 0;
    bool                  ok    = false;  // claimed, and the CPU's result computed

    q40s_node(ggml_backend_dev_t dev, ggml_backend_t cpu, const std::vector<uint8_t> & q, int64_t k, int64_t n,
              int64_t n_mats, int64_t n_used, int64_t ne11, int64_t n_tok, const std::vector<int32_t> & id_values,
              const std::vector<float> & xv) {
        ggml_init_params pw = { ggml_tensor_overhead() * 2, nullptr, true };
        ggml_init_params p  = { ggml_tensor_overhead() * 8 + ggml_graph_overhead() * 2, nullptr, true };
        ctx_w               = ggml_init(pw);
        ctx_h               = ggml_init(p);
        ctx_d               = ggml_init(p);
        if (ctx_w == nullptr || ctx_h == nullptr || ctx_d == nullptr) {
            return;
        }
        w = ggml_new_tensor_3d(ctx_w, GGML_TYPE_Q4_0, k, n, n_mats);
        ggml_set_name(w, "blk.0.ffn_gate_exps.weight");
        ggml_tensor * w_f32 = ggml_new_tensor_3d(ctx_h, GGML_TYPE_F32, k, n, n_mats);
        ggml_tensor * x     = ggml_new_tensor_3d(ctx_h, GGML_TYPE_F32, k, ne11, n_tok);
        ggml_tensor * ids   = ggml_new_tensor_2d(ctx_h, GGML_TYPE_I32, n_used, n_tok);
        y_ref               = ggml_mul_mat_id(ctx_h, w_f32, x, ids);
        y                   = ggml_mul_mat_id(ctx_d, w, x, ids);
        buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, ggml_backend_dev_buffer_type(dev));
        buf_h = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
        buf_d = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, ggml_backend_dev_buffer_type(dev));
        if (buf_w == nullptr || buf_h == nullptr || buf_d == nullptr || q.size() != ggml_nbytes(w) ||
            (int64_t) xv.size() != ggml_nelements(x) || (int64_t) id_values.size() != n_used * n_tok) {
            return;
        }
        ggml_backend_tensor_set(w, q.data(), 0, q.size());
        const std::vector<float> wf = dequantize(GGML_TYPE_Q4_0, q);
        ggml_backend_tensor_set(w_f32, wf.data(), 0, ggml_nbytes(w_f32));
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(ids, id_values.data(), 0, ggml_nbytes(ids));
        n_out               = ggml_nelements(y);
        ggml_cgraph * g_cpu = ggml_new_graph(ctx_h);
        ggml_build_forward_expand(g_cpu, y_ref);
        g_dev = ggml_new_graph(ctx_d);
        ggml_build_forward_expand(g_dev, y);
        ok = ggml_backend_dev_supports_op(dev, y) && ggml_backend_graph_compute(cpu, g_cpu) == GGML_STATUS_SUCCESS;
    }

    ~q40s_node() {
        ggml_backend_buffer_free(buf_d);
        ggml_backend_buffer_free(buf_h);
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx_d);
        ggml_free(ctx_h);
        ggml_free(ctx_w);
    }

    q40s_node(const q40s_node &)             = delete;
    q40s_node & operator=(const q40s_node &) = delete;

    // through the backend, the output filled with NaN first so an element no tile writes shows; ok turns false when
    // the compute fails
    std::vector<float> run(ggml_backend_t backend) {
        const std::vector<float> junk((size_t) n_out, NAN);
        ggml_backend_tensor_set(y, junk.data(), 0, ggml_nbytes(y));
        ok = ok && ggml_backend_graph_compute(backend, g_dev) == GGML_STATUS_SUCCESS;
        std::vector<float> out((size_t) n_out, NAN);
        ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
        return out;
    }

    const float * ref() const { return static_cast<const float *>(y_ref->data); }
};

// the q4_0s layout and the m8 arm (the default; FLAGOS_SPACEMIT_EXPERT_Q4_0=plain the fallback; plan.md "The Q4_0
// experts: the fork's signed layout and 8-row kernel", its tests on the Mac). Checked: the switch's values (the layouts
// of 3-D and 2-D Q4_0 and of 3-D Q4_1, the claims, an unknown value and pad8 warning once); repack and unpack restore
// the GGUF bytes at the model's two shapes and small ones; each group kernel (8, 4 and 1 rows) against the CPU on the
// dequantized weights; the kernels against each other on rows of equal magnitude; the groups (1 to 17, 24, 31 and 32
// rows an expert, experts of 0 rows between active ones, every pair on one expert) against the CPU, each output row on
// its own, and the claim queue check's forms the same bytes; one token within the bound of plain. Reported, not
// bounded: the NMSE of a small row grouped with 3 and with 7 rows a hundred times larger, under plain and m8. The
// direct calls run on the calling core, so on the K3 only with the reference kernels
static int check_mul_mat_id_q4_0s(ggml_backend_dev_t dev, bool host) {
    const bool     direct = host || *reference_label() != '\0';
    ggml_backend_t cpu    = ggml_backend_cpu_init();
    REQUIRE(cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);  // a correctness reference: single-threaded (PROJECT.md item 28)
    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    REQUIRE(kernels != nullptr);
    const spacemit_plugin_kernel * step_q = nullptr, * step_g = nullptr;
    for (uint32_t i = 0; i < n_kernels; i++) {
        if (kernels[i].name != nullptr && std::strcmp(kernels[i].name, "mul_mat_id.quantize") == 0) {
            step_q = &kernels[i];
        }
        if (kernels[i].name != nullptr && std::strcmp(kernels[i].name, "mul_mat_id.gemm") == 0) {
            step_g = &kernels[i];
        }
    }
    REQUIRE(step_q != nullptr && step_g != nullptr && step_q->workspace_size != nullptr);

    const auto fail = [](bool good, const std::string & what) {
        if (!good) {
            std::fprintf(stderr, "FAIL: MUL_MAT_ID q4_0s, %s\n", what.c_str());
            g_failures++;
        }
        return good;
    };
    // expert e repeated counts[e] times, filling the tokens' slots in order
    const auto id_list = [](const std::vector<int64_t> & counts) {
        std::vector<int32_t> list;
        for (size_t e = 0; e < counts.size(); e++) {
            list.insert(list.end(), (size_t) counts[e], (int32_t) e);
        }
        return list;
    };
    // the NMSE of rows [r0, r1) of n columns of a against b
    const auto rows_nmse = [](const std::vector<float> & a, const float * b, int64_t n, int64_t r0, int64_t r1) {
        return nmse(a.data() + r0 * n, b + r0 * n, (r1 - r0) * n);
    };
    const auto sci = [](double v) {
        char text[32];
        std::snprintf(text, sizeof(text), "%.3e", v);
        return std::string(text);
    };
    uint32_t seed = 5200;

    // the switch: the layouts plain gives, then each value
    uint32_t plain2 = 0, plain3 = 0, q41 = 0;
    {
        env_override     sw("FLAGOS_SPACEMIT_EXPERT_Q4_0", "plain");
        ggml_init_params p   = { ggml_tensor_overhead() * 4, nullptr, true };
        ggml_context *   ctx = ggml_init(p);
        REQUIRE(ctx != nullptr);
        plain2 = table->tensor_layout(ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 64));
        plain3 = table->tensor_layout(ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, 256, 64, 4));
        q41    = table->tensor_layout(ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_1, 256, 64, 4));
        ggml_free(ctx);
        REQUIRE(plain2 != 0 && plain3 == plain2 && q41 != 0);
    }
    int n_switch = 0, n_switch_ok = 0;
    for (const char * setting : { (const char *) nullptr, "", "m8", "plain", "pad8", "bogus" }) {
        env_override     sw("FLAGOS_SPACEMIT_EXPERT_Q4_0", setting);
        log_capture      log;
        const bool       named = setting != nullptr && *setting != '\0';
        const bool       bogus = named && (std::strcmp(setting, "pad8") == 0 || std::strcmp(setting, "bogus") == 0);
        const bool       want  = !named || std::strcmp(setting, "plain") != 0;
        ggml_init_params p     = { ggml_tensor_overhead() * 16, nullptr, true };
        ggml_context *   ctx   = ggml_init(p);
        REQUIRE(ctx != nullptr);
        ggml_tensor *  w3 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, 256, 64, 4);
        ggml_tensor *  w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 64);
        ggml_tensor *  wg = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 2048, 512);  // the model's gate shape in 2-D
        const uint32_t l3 = table->tensor_layout(w3);
        const uint32_t l2 = table->tensor_layout(w2);
        const uint32_t lg = table->tensor_layout(wg);
        const uint32_t l1 = table->tensor_layout(ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_1, 256, 64, 4));
        const bool     layouts =
            l2 == plain2 && lg == plain2 && l1 == q41 && (want ? l3 != plain3 && l3 != 0 : l3 == plain3);
        ggml_tensor * x1     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 512);
        ggml_tensor * ids2   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 512);
        ggml_tensor * x7     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 7);
        const bool    claims = ggml_backend_dev_supports_op(dev, ggml_mul_mat_id(ctx, w3, x1, ids2)) &&
                            ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w2, x7));
        const int warnings = count_lines_with(log.take(), "FLAGOS_SPACEMIT_EXPERT_Q4_0=");
        n_switch++;
        n_switch_ok += fail(layouts && claims && warnings == (bogus ? 1 : 0),
                            std::string("EXPERT_Q4_0=") + (setting != nullptr ? setting : "unset") +
                                ": layouts 3-D " + std::to_string(l3) + ", 2-D " + std::to_string(l2) + " and " +
                                std::to_string(lg) + ", 3-D Q4_1 " + std::to_string(l1) + " (plain's 3-D and 2-D " +
                                std::to_string(plain2) + ", 3-D Q4_1 " + std::to_string(q41) + "); claims " +
                                (claims ? "taken" : "refused") + "; " + std::to_string(warnings) + " warnings");
        ggml_free(ctx);
    }

    // repack and unpack restore the GGUF bytes, and a second repack keeps them; the model's two shapes repack on
    // several threads (over 1 MiB)
    int n_trip = 0, n_trip_ok = 0;
    {
        env_override sw("FLAGOS_SPACEMIT_EXPERT_Q4_0", "m8");
        for (const std::array<int64_t, 3> & s : std::vector<std::array<int64_t, 3>>{
                 { 2048, 512, 4 }, { 512, 2048, 4 }, { 256, 32, 2 }, { 512, 64, 3 } }) {
            ggml_init_params p   = { ggml_tensor_overhead() * 2, nullptr, true };
            ggml_context *   ctx = ggml_init(p);
            REQUIRE(ctx != nullptr);
            ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, s[0], s[1], s[2]);
            ggml_set_name(w, "blk.0.ffn_up_exps.weight");
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            REQUIRE(buf != nullptr);
            const std::vector<uint8_t> q = quantize_weights(GGML_TYPE_Q4_0, s[0], s[1] * s[2], seed++);
            std::vector<uint8_t>       read(q.size()), read2(q.size());
            ggml_backend_tensor_set(w, q.data(), 0, q.size());
            ggml_backend_tensor_get(w, read.data(), 0, read.size());
            ggml_backend_tensor_set(w, read.data(), 0, read.size());
            ggml_backend_tensor_get(w, read2.data(), 0, read2.size());
            const bool signed_layout = table->tensor_layout(w) != plain3;
            n_trip++;
            n_trip_ok += fail(signed_layout && read == q && read2 == q,
                              std::string("[") + std::to_string(s[0]) + ", " + std::to_string(s[1]) + ", " +
                                  std::to_string(s[2]) + "]: " + (signed_layout ? "" : "not the q4_0s layout, ") +
                                  "the read-back " + (read == q ? "is" : "is not") + " the GGUF bytes, after a second " +
                                  "repack " + (read2 == q ? "is" : "is not"));
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }

    // each group kernel against the CPU on the dequantized weights: 16 tokens of one id, experts of 8, 4, 1 and 3 rows,
    // so tokens 0-7 take the 8-row kernel, 8-11 the 4-row one, 12-15 the 1-row one
    double e_k8 = HUGE_VAL, e_k4 = HUGE_VAL, e_k1 = HUGE_VAL;
    bool   kernels_ok = true;
    {
        const int64_t              k = 2048, n = 64;
        const std::vector<int64_t> counts = { 8, 4, 1, 3 };
        const std::vector<uint8_t> q      = quantize_weights(GGML_TYPE_Q4_0, k, n * 4, seed++);
        const std::vector<float>   xv     = random_values(k * 16, seed++, 2.0f);
        env_override               sw("FLAGOS_SPACEMIT_EXPERT_Q4_0", "m8");
        q40s_node                  nd(dev, cpu, q, k, n, 4, 1, 1, 16, id_list(counts), xv);
        ggml_backend_t             backend = tcm_backend(dev, nullptr);
        REQUIRE(backend != nullptr);
        const std::vector<float> out = nd.run(backend);
        ggml_backend_free(backend);
        if (nd.ok) {
            e_k8 = rows_nmse(out, nd.ref(), n, 0, 8);
            e_k4 = rows_nmse(out, nd.ref(), n, 8, 12);
            e_k1 = rows_nmse(out, nd.ref(), n, 12, 16);
        }
        kernels_ok = fail(nd.ok && e_k8 <= k_mul_mat_max_nmse && e_k4 <= k_mul_mat_max_nmse && e_k1 <= k_mul_mat_max_nmse,
                          "the kernels against the CPU: 8 rows " + sci(e_k8) + ", 4 rows " + sci(e_k4) + ", 1 row " +
                              sci(e_k1));
    }

    // the kernels against each other on rows of equal magnitude: every row's largest value of a subblock the same, so
    // a group's shared scale is each row's own. All experts hold the same weights; 8 tokens of 2 ids: slot 0 on expert
    // 0 (a group of 8 under m8), slot 1 on expert 1 for tokens 0-3 (a group of 4), 2 for 4-6 and 3 for 7 (single rows);
    // each token alone (one token: the 1-row path) multiplies its pairs with the 1-row kernel
    double e_84 = HUGE_VAL, e_81 = HUGE_VAL, e_41 = HUGE_VAL;
    {
        const int64_t        k = 512, n = 64, n_tok = 8;
        std::vector<uint8_t> one = quantize_weights(GGML_TYPE_Q4_0, k, n, seed++), q;
        for (int e = 0; e < 4; e++) {
            q.insert(q.end(), one.begin(), one.end());
        }
        std::vector<float> xv = random_values(k * n_tok, seed++, 0.9f);
        for (int64_t t = 0; t < n_tok; t++) {
            for (int64_t b = 0; b < k / 32; b++) {
                const float m = 0.5f + 0.25f * (float) (b % 8);
                for (int64_t i = 0; i < 32; i++) {
                    xv[(size_t) (t * k + b * 32 + i)] *= m;
                }
                xv[(size_t) (t * k + b * 32 + (t + b) % 32)] = t % 2 == 0 ? m : -m;
            }
        }
        std::vector<int32_t> ids;
        for (int64_t t = 0; t < n_tok; t++) {
            ids.push_back(0);
            ids.push_back(t < 4 ? 1 : t < 7 ? 2 : 3);
        }
        env_override   sw("FLAGOS_SPACEMIT_EXPERT_Q4_0", "m8");
        q40s_node      nd(dev, cpu, q, k, n, 4, 2, 1, n_tok, ids, xv);
        ggml_backend_t backend = tcm_backend(dev, nullptr);
        REQUIRE(backend != nullptr);
        const std::vector<float> staged = nd.run(backend);
        std::vector<float>       rows;  // the tokens one by one, as the grouped node's output rows
        bool                     alone_ok = true;
        for (int64_t t = 0; t < n_tok; t++) {
            const std::vector<float> xt(xv.begin() + t * k, xv.begin() + (t + 1) * k);
            q40s_node                nt(dev, cpu, q, k, n, 4, 2, 1, 1, { ids[t * 2], ids[t * 2 + 1] }, xt);
            const std::vector<float> out = nt.run(backend);
            alone_ok                     = alone_ok && nt.ok;
            rows.insert(rows.end(), out.begin(), out.end());
        }
        ggml_backend_free(backend);
        if (nd.ok && alone_ok) {
            // the output row of (token t, slot s) is t * 2 + s
            std::vector<float> s8, s4, r8, r4;
            for (int64_t t = 0; t < n_tok; t++) {
                const auto at = [&](const std::vector<float> & v, int64_t s) { return v.begin() + (t * 2 + s) * n; };
                s8.insert(s8.end(), at(staged, 0), at(staged, 0) + n);
                r8.insert(r8.end(), at(rows, 0), at(rows, 0) + n);
                if (t < 4) {
                    s4.insert(s4.end(), at(staged, 1), at(staged, 1) + n);
                    r4.insert(r4.end(), at(rows, 1), at(rows, 1) + n);
                }
            }
            e_84 = nmse(s8.data(), s4.data(), (int64_t) s4.size());
            e_81 = nmse(s8.data(), r8.data(), (int64_t) s8.size());
            e_41 = nmse(s4.data(), r4.data(), (int64_t) s4.size());
        }
    }
    const bool equal_ok = fail(e_84 <= k_q4_0s_kernels_max_nmse && e_81 <= k_q4_0s_kernels_max_nmse &&
                                   e_41 <= k_q4_0s_kernels_max_nmse,
                               "the kernels against each other on rows of equal magnitude: 8 vs 4 rows " +
                                   sci(e_84) + ", 8 vs 1 " + sci(e_81) + ", 4 vs 1 " +
                                   sci(e_41));

    // the cost of the shared scale, reported: token 0 and token 4 a hundred times smaller than the others; tokens 0-3
    // on expert 0 (token 0 with 3 others), 4-11 on expert 1 (token 4 with 7 others; plain: in two groups of 4)
    double e_small[2][3] = {};  // [plain, m8][with 3, with 7, alone (one token)]
    {
        const int64_t              k = 512, n = 64, n_tok = 12;
        const std::vector<uint8_t> q  = quantize_weights(GGML_TYPE_Q4_0, k, n * 2, seed++);
        std::vector<float>         xv = random_values(k * n_tok, seed++, 1.0f);
        for (const int64_t t : { 0, 4 }) {
            for (int64_t i = 0; i < k; i++) {
                xv[(size_t) (t * k + i)] *= 0.01f;
            }
        }
        const char * const settings[2] = { "plain", "m8" };
        for (int a = 0; a < 2; a++) {
            env_override   sw("FLAGOS_SPACEMIT_EXPERT_Q4_0", settings[a]);
            q40s_node      nd(dev, cpu, q, k, n, 2, 1, 1, n_tok, id_list({ 4, 8 }), xv);
            q40s_node      alone(dev, cpu, q, k, n, 2, 1, 1, 1, { 0 }, std::vector<float>(xv.begin(), xv.begin() + k));
            ggml_backend_t backend = tcm_backend(dev, nullptr);
            REQUIRE(backend != nullptr);
            const std::vector<float> staged = nd.run(backend);
            const std::vector<float> one    = alone.run(backend);
            ggml_backend_free(backend);
            e_small[a][0] = nd.ok ? rows_nmse(staged, nd.ref(), n, 0, 1) : HUGE_VAL;
            e_small[a][1] = nd.ok ? rows_nmse(staged, nd.ref(), n, 4, 5) : HUGE_VAL;
            e_small[a][2] = alone.ok ? rows_nmse(one, alone.ref(), n, 0, 1) : HUGE_VAL;
            fail(nd.ok && alone.ok, std::string("the shared scale's cost under ") + settings[a] + ": the node failed");
        }
    }

    // the groups. Each case against the CPU (the whole output, and its worst row: a row written with another pair's
    // output has NMSE about 1), and the first backend run's bytes (with TCM) without TCM, twice in a row, on threads 20
    // times, and the steps directly on 1 tile and 8 (in reverse, one tile first, without TCM, on threads 20 times)
    struct shape_case {
        const char *         what;
        int64_t              n_used, ne11;
        std::vector<int64_t> counts;  // the rows per expert; its size is the expert count
    };
    std::vector<int64_t> one_to_17;
    for (int64_t r = 1; r <= 17; r++) {
        one_to_17.push_back(r);
    }
    const std::vector<shape_case> shapes = {
        { "1 to 17 rows an expert", 1, 1, one_to_17 },
        { "24, 31 and 32 rows an expert", 1, 1, { 24, 31, 32 } },
        { "experts of 0 rows between active ones", 1, 1, { 0, 3, 0, 0, 9, 0, 0, 0, 2, 0, 13, 0 } },
        { "every pair on one expert, a row per id", 4, 4, { 0, 0, 40, 0 } },
    };
    struct direct_form {
        const char * what;
        uint32_t     nth;
        int          order;  // 0 the tiles in turn, 1 in reverse, 2 tile nth / 2 first, 3 on threads
        bool         staged;
        uint32_t     runs;
    };
    const std::vector<direct_form> forms = {
        { "1 tile", 1, 0, true, 1 },
        { "8 tiles", 8, 0, true, 1 },
        { "8 tiles in reverse", 8, 1, true, 1 },
        { "8 tiles, one first", 8, 2, true, 1 },
        { "8 tiles without TCM", 8, 0, false, 1 },
        { "8 tiles on threads", 8, 3, true, 20 },
        { "8 tiles on threads without TCM", 8, 3, false, 20 },
    };
    const size_t        tcm_size = 393216;  // the K3's 384 KiB
    std::vector<uint8_t> tcm(8 * tcm_size);
    // the two steps directly on nth tiles, the workspace aligned to 64 as the core's
    const auto direct_run = [&](q40s_node & nd, uint32_t nth, int order, bool staged) {
        const size_t         ws = step_q->workspace_size(nd.y);
        std::vector<uint8_t> raw(ws + 64);
        uint8_t * const      wsp = raw.data() + (64 - reinterpret_cast<uintptr_t>(raw.data()) % 64) % 64;
        const std::vector<float> junk((size_t) nd.n_out, NAN);
        ggml_backend_tensor_set(nd.y, junk.data(), 0, ggml_nbytes(nd.y));
        std::atomic<bool> failed{ false };
        const auto        call = [&](const spacemit_plugin_kernel * step, uint32_t ith) {
            uint8_t * const     own  = tcm.data() + (order == 3 ? ith * tcm_size : 0);
            const spacemit_tile tile = { ith, nth, staged ? own : nullptr, staged ? tcm_size : 0, wsp, ws, nullptr };
            if (!step->fn(&tile, nd.y)) {
                failed = true;
            }
        };
        for (const spacemit_plugin_kernel * step : { step_q, step_g }) {
            if (order == 3) {
                std::vector<std::thread> threads;
                for (uint32_t ith = 0; ith < nth; ith++) {
                    threads.emplace_back([&, ith] { call(step, ith); });
                }
                for (std::thread & t : threads) {
                    t.join();
                }
                continue;
            }
            for (uint32_t i = 0; i < nth; i++) {
                const uint32_t half = nth / 2;
                call(step, order == 1 ? nth - 1 - i : order == 2 ? (i == 0 ? half : i <= half ? i - 1 : i) : i);
            }
        }
        nd.ok = nd.ok && !failed;
        std::vector<float> out((size_t) nd.n_out, NAN);
        ggml_backend_tensor_get(nd.y, out.data(), 0, ggml_nbytes(nd.y));
        return out;
    };
    int    n_shapes = 0, n_shapes_ok = 0, n_same = 0, n_same_ok = 0;
    double worst_shape = 0.0, worst_row = 0.0;
    for (const shape_case & c : shapes) {
        const int64_t              k = 256, n = 96, n_mats = (int64_t) c.counts.size();
        const std::vector<int32_t> ids   = id_list(c.counts);
        const int64_t              n_tok = (int64_t) ids.size() / c.n_used;
        const std::vector<uint8_t> q     = quantize_weights(GGML_TYPE_Q4_0, k, n * n_mats, seed++);
        const std::vector<float>   xv    = random_values(k * c.ne11 * n_tok, seed++, 2.0f);
        for (const char * arm : { "m8" }) {
            env_override       sw("FLAGOS_SPACEMIT_EXPERT_Q4_0", arm);
            q40s_node          nd(dev, cpu, q, k, n, n_mats, c.n_used, c.ne11, n_tok, ids, xv);
            const size_t       bytes = (size_t) nd.n_out * sizeof(float);
            std::string        differ;
            const auto same = [&](const std::vector<float> & a, const std::vector<float> & b, const char * what) {
                n_same++;
                const bool good = a.size() == b.size() && std::memcmp(a.data(), b.data(), bytes) == 0;
                n_same_ok += good;
                if (!good) {
                    differ += differ.empty() ? what : std::string(", ") + what;
                }
            };
            if (!nd.ok) {
                n_shapes++;
                fail(false, std::string(arm) + ", " + c.what + ": not claimed, or the CPU failed");
                continue;
            }
            std::vector<float> ref;
            {
                ggml_backend_t backend = tcm_backend(dev, nullptr);
                REQUIRE(backend != nullptr);
                ref = nd.run(backend);
                ggml_backend_free(backend);
            }
            {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                REQUIRE(backend != nullptr);
                same(nd.run(backend), ref, "without TCM");
                same(nd.run(backend), ref, "the second compute in a row");
                ggml_backend_free(backend);
            }
            if (host) {
                ggml_backend_t backend = tcm_backend(dev, "1");
                REQUIRE(backend != nullptr);
                bool all = true;
                for (int i = 0; i < 20; i++) {
                    const std::vector<float> out = nd.run(backend);
                    all = all && std::memcmp(out.data(), ref.data(), bytes) == 0;
                }
                same(all ? ref : std::vector<float>(), ref, "on threads");
                ggml_backend_free(backend);
            }
            for (size_t f = 0; f < forms.size() && direct; f++) {
                bool all = true;
                for (uint32_t i = 0; i < forms[f].runs; i++) {
                    const std::vector<float> out = direct_run(nd, forms[f].nth, forms[f].order, forms[f].staged);
                    all = all && std::memcmp(out.data(), ref.data(), bytes) == 0;
                }
                same(all ? ref : std::vector<float>(), ref, forms[f].what);
            }
            const int64_t n_rows = nd.n_out / n;
            double        e = HUGE_VAL, e_row = HUGE_VAL;
            if (nd.ok) {
                e     = nmse(ref.data(), nd.ref(), nd.n_out);
                e_row = 0.0;
                for (int64_t r = 0; r < n_rows; r++) {
                    e_row = std::max(e_row, rows_nmse(ref, nd.ref(), n, r, r + 1));
                }
            }
            worst_shape = std::max(worst_shape, e);
            worst_row   = std::max(worst_row, e_row);
            n_shapes++;
            n_shapes_ok += fail(nd.ok && differ.empty() && e <= k_mul_mat_max_nmse && e_row <= k_q4_0s_row_max_nmse,
                                std::string(arm) + ", " + c.what + ": " + (nd.ok ? "ran" : "failed") +
                                    "; not the first run's bytes: " + (differ.empty() ? "none" : differ) +
                                    "; NMSE " + sci(e) + ", worst row " + sci(e_row));
        }
    }

    // one token: the 1-row pack and the signed 1-row kernel, against plain and the CPU within the bound
    int    n_one = 0, n_one_ok = 0;
    double worst_one = 0.0;
    {
        const int64_t              k = 2048, n = 512, n_mats = 16;
        const std::vector<uint8_t> q   = quantize_weights(GGML_TYPE_Q4_0, k, n * n_mats, seed++);
        const std::vector<float>   xv  = random_values(k, seed++, 2.0f);
        const std::vector<int32_t> ids = { 3, 14, 0, 7, 9, 1, 12, 5 };
        const char * const         settings[2] = { "plain", "m8" };
        std::vector<float>         out[2];
        bool                       ran = true;
        double                     e_cpu = 0.0;
        for (int a = 0; a < 2; a++) {
            env_override   sw("FLAGOS_SPACEMIT_EXPERT_Q4_0", settings[a]);
            q40s_node      nd(dev, cpu, q, k, n, n_mats, 8, 1, 1, ids, xv);
            ggml_backend_t backend = tcm_backend(dev, nullptr);
            REQUIRE(backend != nullptr);
            out[a] = nd.run(backend);
            ggml_backend_free(backend);
            ran   = ran && nd.ok;
            e_cpu = std::max(e_cpu, nd.ok ? nmse(out[a].data(), nd.ref(), nd.n_out) : HUGE_VAL);
        }
        worst_one = ran ? nmse(out[1].data(), out[0].data(), (int64_t) out[0].size()) : HUGE_VAL;
        n_one++;
        n_one_ok += fail(ran && worst_one <= k_mul_mat_max_nmse && e_cpu <= k_mul_mat_max_nmse,
                         "one token: against plain NMSE " + sci(worst_one) + ", against the CPU " + sci(e_cpu));
    }

    ggml_backend_free(cpu);
    std::printf("mmid     3-D Q4_0 experts in the q4_0s layout, the m8 arm (the default): the switch (unset, empty and "
                "m8 give m8, plain gives plain, pad8 and another unknown value warn once and keep m8; 2-D Q4_0 and 3-D "
                "Q4_1 keep their layouts) %d/%d; repack and unpack the GGUF bytes at [2048, 512, 4], [512, 2048, 4] "
                "and two small shapes %d/%d; against the CPU on the dequantized weights max NMSE the 8-row kernel "
                "%.2e, 4-row %.2e, 1-row %.2e (bound %.0e)%s; on rows of equal magnitude 8 vs 4 rows %.2e, 8 vs 1 "
                "%.2e, 4 vs 1 %.2e (bound %.0e)%s; the groups (1 to 17, 24, 31 and 32 rows an expert, experts of 0 "
                "rows between active ones, every pair on one expert) %d/%d cases, max NMSE %.2e, worst row %.2e "
                "(bound %.0e); the first run's bytes (without TCM, twice in a row%s%s) in %d/%d; "
                "one token against plain within the bound %d/%d (NMSE %.2e)%s\n",
                n_switch_ok, n_switch, n_trip_ok, n_trip, e_k8, e_k4, e_k1, k_mul_mat_max_nmse,
                kernels_ok ? "" : " FAILED", e_84, e_81, e_41, k_q4_0s_kernels_max_nmse, equal_ok ? "" : " FAILED",
                n_shapes_ok, n_shapes, worst_shape, worst_row, k_q4_0s_row_max_nmse,
                host ? ", on threads 20 times" : "",
                direct ? ", the steps directly on 1 and 8 tiles, in reverse, one first, without TCM, on threads" : "",
                n_same_ok, n_same, n_one_ok, n_one, worst_one, reference_label());
    std::printf("info     MUL_MAT_ID q4_0s, the cost of the shared activation scale (reported, not bounded): NMSE of a "
                "row a hundred times smaller than the rows it is grouped with, with 3 / with 7 / alone (one token): "
                "plain %.2e / %.2e / %.2e, m8 %.2e / %.2e / %.2e%s\n",
                e_small[0][0], e_small[0][1], e_small[0][2], e_small[1][0], e_small[1][1], e_small[1][2],
                reference_label());
    return 0;
}

// MUL_MAT_ID, the experts' matmuls (Qwen3.6-35B-A3B): the claims and the refused forms, then the provider against the
// CPU backend on one thread, for the IME layouts' types at the model's shapes (gate and up [2048, 512, 256] with one
// activation row per token, down [512, 2048, 256] with one per id, 8 ids per token) and at small ones, 1 to 512
// tokens, 1 to 8 ids per token. The ids: distinct within a token as ARGSORT gives them (repeating across tokens), the
// same for every token, repeating within a token; a view of a wider I32 tensor (ARGSORT's output, its unused entries
// out of range) or a tensor of their own. The activations: broadcast (one row per token), one row per id, one per two
// ids. Q6_K, stored requantized to Q8_0, is compared with the GGUF weights (bound 5e-4) and with ggml's Q8_0 of them
// (5e-6), as check_mul_mat does; a 3-D Q4_1 weight is claimed in the q4_1 min layout (check_mul_mat_id_q4_1 checks
// its results), but not with K 128 (no layout). Every case runs
// twice through the backend (8 tiles) and its two kernel entries directly on 1 and on 8 tiles: the four outputs bit
// for bit. An id outside the experts fails the graph.
// --full adds the model's shapes for Q8_0, Q6_K and Q5_0
static int check_mul_mat_id(ggml_backend_dev_t dev, bool host, bool full) {
    // the direct calls run on the calling core: on the K3 that is an X100 core, which has no IME instructions, so
    // they run there only with the reference kernels
    const bool direct = host || *reference_label() != '\0';
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);  // a correctness reference: single-threaded (PROJECT.md item 28)
    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    REQUIRE(kernels != nullptr);
    const spacemit_plugin_kernel * step_q = nullptr, * step_g = nullptr;
    for (uint32_t i = 0; i < n_kernels; i++) {
        if (kernels[i].name != nullptr && std::strcmp(kernels[i].name, "mul_mat_id.quantize") == 0) {
            step_q = &kernels[i];
        }
        if (kernels[i].name != nullptr && std::strcmp(kernels[i].name, "mul_mat_id.gemm") == 0) {
            step_g = &kernels[i];
        }
    }
    REQUIRE(step_q != nullptr && step_g != nullptr && step_q->workspace_size != nullptr &&
            step_q->workspace_size == step_g->workspace_size);

    // what the contract takes and what it refuses
    int n_claims = 0, n_claims_ok = 0;
    {
        ggml_init_params p       = { ggml_tensor_overhead() * 64, nullptr, true };
        ggml_context *   ctx     = ggml_init(p);
        ggml_context *   ctx_cpu = ggml_init(p);
        REQUIRE(ctx != nullptr && ctx_cpu != nullptr);
        const auto expect = [&](bool want, ggml_tensor * node, const char * what) {
            n_claims++;
            const bool got = ggml_backend_dev_supports_op(dev, node);
            n_claims_ok += got == want;
            if (got != want) {
                std::fprintf(stderr, "FAIL: MUL_MAT_ID %s: %s, expected %s\n", what, got ? "claimed" : "refused",
                             want ? "claimed" : "refused");
                g_failures++;
            }
        };
        ggml_tensor * w        = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, 256, 64, 4);
        ggml_tensor * x1       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 512);
        ggml_tensor * x2       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, 512);
        ggml_tensor * ids2     = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 512);
        ggml_tensor * ids_full = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 512);
        ggml_tensor * ids_v    = ggml_view_2d(ctx, ids_full, 2, 512, ids_full->nb[1], 0);
        expect(true, ggml_mul_mat_id(ctx, w, x1, ids2), "Q4_0, broadcast activations, 512 tokens");
        expect(true, ggml_mul_mat_id(ctx, w, x2, ids_v), "Q4_0, a row per id, the ids a view");
        expect(true,
               ggml_mul_mat_id(ctx, w, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 1),
                               ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1)),
               "Q4_0, 1 token");
        for (const ggml_type type : { GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_0 }) {
            expect(true, ggml_mul_mat_id(ctx, ggml_new_tensor_3d(ctx, type, 256, 64, 4), x1, ids2),
                   ggml_type_name(type));
        }
        ggml_tensor * placed = ggml_mul_mat_id(ctx, w, x1, ids2);  // how llama.cpp places the weight
        w->buffer            = ggml_backend_buft_alloc_buffer(buft, 0);
        expect(true, placed, "Q4_0 in a zero-size buffer of our type");
        ggml_backend_buffer_free(w->buffer);
        w->buffer = nullptr;
        // a 3-D Q4_1 weight takes the q4_1 min layout and is claimed, also in our buffer; with K 128 (not a multiple
        // of 256) it has none and is refused, also in our buffer
        ggml_tensor * w1   = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_1, 256, 64, 4);
        ggml_tensor * w1_k = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_1, 128, 64, 4);
        ggml_tensor * w12  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, 256, 64);
        n_claims += 3;
        n_claims_ok += table->tensor_layout(w1) != 0;
        n_claims_ok += table->tensor_layout(w1_k) == 0;
        n_claims_ok += table->tensor_layout(w12) != 0;
        CHECK(table->tensor_layout(w1) != 0 && table->tensor_layout(w1_k) == 0 && table->tensor_layout(w12) != 0);
        expect(true, ggml_mul_mat_id(ctx, w1, x1, ids2), "a 3-D Q4_1 weight");
        ggml_tensor * placed1 = ggml_mul_mat_id(ctx, w1, x2, ids_v);
        w1->buffer            = ggml_backend_buft_alloc_buffer(buft, 0);
        expect(true, placed1, "a 3-D Q4_1 weight in a zero-size buffer of our type");
        ggml_backend_buffer_free(w1->buffer);
        w1->buffer         = nullptr;
        ggml_tensor * x1_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 1, 512);
        expect(false, ggml_mul_mat_id(ctx, w1_k, x1_k, ids2), "a 3-D Q4_1 weight of K 128 (no layout)");
        ggml_tensor * placed1_k = ggml_mul_mat_id(ctx, w1_k, x1_k, ids2);
        w1_k->buffer            = ggml_backend_buft_alloc_buffer(buft, 0);
        expect(false, placed1_k, "a 3-D Q4_1 weight of K 128 in a zero-size buffer of our type");
        ggml_backend_buffer_free(w1_k->buffer);
        w1_k->buffer = nullptr;

        expect(false, ggml_mul_mat_id(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, 256, 48, 4), x1, ids2),
               "48 rows per expert (not a multiple of 32)");
        expect(false,
               ggml_mul_mat_id(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, 96, 64, 4),
                               ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 96, 1, 512), ids2),
               "Q4_0 rows of 96 (not a multiple of 256)");
        expect(false, ggml_mul_mat_id(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 64, 4), x1, ids2),
               "an F32 weight");
        expect(false, ggml_mul_mat_id(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, 256, 64, 4), x1, ids2),
               "a Q4_K weight (no layout)");
        expect(false, ggml_mul_mat_id(ctx, ggml_view_3d(ctx, w, 256, 64, 2, w->nb[1], w->nb[2], 0), x1, ids2),
               "a view of a weight");
        expect(false, ggml_mul_mat_id(ctx, w, ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 256, 1, 512), ids2),
               "F16 activations");
        ggml_tensor * wide = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, 1, 512);
        expect(false, ggml_mul_mat_id(ctx, w, ggml_view_3d(ctx, wide, 256, 1, 512, wide->nb[1], wide->nb[2], 0), ids2),
               "non-contiguous activations");
        ggml_tensor *         w_c   = ggml_new_tensor_3d(ctx_cpu, GGML_TYPE_Q4_0, 256, 64, 4);
        ggml_backend_buffer_t buf_c = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cpu, ggml_backend_cpu_buffer_type());
        REQUIRE(buf_c != nullptr);
        expect(false, ggml_mul_mat_id(ctx, w_c, x1, ids2), "a weight in a CPU buffer");
        expect(false, ggml_mul_mat(ctx, w, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 4, 4)),
               "MUL_MAT with the 3-D weight");
        ggml_backend_buffer_free(buf_c);
        ggml_free(ctx);
        ggml_free(ctx_cpu);
    }

    enum ids_kind { ids_distinct, ids_same, ids_repeat };
    struct mmid_case {
        const char *         what;
        ggml_type            type;
        int64_t              k, n, n_mats, n_used, ne11;
        ids_kind             ids;
        bool                 view;  // the ids a view of [n_mats, tokens], as ARGSORT's output
        std::vector<int64_t> tokens;
        bool                 imatrix = false;  // quantized with an importance matrix of ones, as test-backend-ops does
    };
    const std::vector<int64_t> all_tokens = { 1, 4, 5, 17, 512 };
    std::vector<mmid_case>     cases      = {
        { "the model's gate and up", GGML_TYPE_Q4_0, 2048, 512, 256, 8, 1, ids_distinct, true, all_tokens },
        { "the model's down", GGML_TYPE_Q4_0, 512, 2048, 256, 8, 8, ids_distinct, true, all_tokens },
        { "ids the same for every token", GGML_TYPE_Q4_0, 256, 64, 8, 4, 4, ids_same, true, { 17 } },
        { "ids repeating within a token", GGML_TYPE_Q8_0, 256, 64, 8, 4, 1, ids_repeat, true, { 1, 17 } },
        { "ids repeating within a token, a row per id", GGML_TYPE_Q4_0, 256, 64, 8, 4, 4, ids_repeat, true, { 5 } },
        { "a row per two ids", GGML_TYPE_Q8_0, 256, 64, 8, 4, 2, ids_distinct, true, { 5, 17 } },
        { "96 rows per expert, every expert used", GGML_TYPE_Q8_0, 512, 96, 4, 4, 4, ids_distinct, false, { 1, 5, 17 } },
        // test-backend-ops' three Q6_K shapes, quantized with an imatrix as it does half the time
        { "test-backend-ops' shape, imatrix", GGML_TYPE_Q6_K, 256, 512, 4, 2, 2, ids_distinct, true, { 1, 32 }, true },
        { "test-backend-ops' shape, imatrix", GGML_TYPE_Q6_K, 768, 64, 4, 2, 2, ids_distinct, true, { 16 }, true },
    };
    for (const ggml_type type : { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_0 }) {
        for (const int64_t n_used : { 1, 2, 4, 8 }) {
            for (const int64_t ne11 : { (int64_t) 1, n_used }) {
                if (ne11 == 1 && n_used == 1 && cases.back().type == type && cases.back().n_used == 1) {
                    continue;  // one id: one row per token either way
                }
                cases.push_back({ ne11 == 1 ? "small, broadcast" : "small, a row per id", type, 256, 64, 8, n_used, ne11,
                                  ids_distinct, n_used < 8, all_tokens });
            }
        }
    }
    if (full) {
        for (const ggml_type type : { GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_0 }) {
            cases.push_back({ "the model's gate and up", type, 2048, 512, 256, 8, 1, ids_distinct, true, all_tokens });
            cases.push_back({ "the model's down", type, 512, 2048, 256, 8, 8, ids_distinct, true, all_tokens });
        }
    }

    const ggml_type types[4]    = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_0 };
    int             n_runs[4]   = { 0, 0, 0, 0 }, n_ok = 0, n_total = 0, n_same = 0;
    double          worst[4]    = { 0.0, 0.0, 0.0, 0.0 };
    double          worst_q8[4] = { 0.0, 0.0, 0.0, 0.0 };
    // Q6_K: the CPU on the weight's read-back (the nearest Q6_K to the stored Q8_0), what test-backend-ops' CPU computes
    // with; quantized without and with an imatrix
    double          worst_readback[2] = { 0.0, 0.0 };
    uint32_t        seed          = 900;
    for (const mmid_case & c : cases) {
        const int  t        = (int) (std::find(types, types + 4, c.type) - types);
        const bool requant  = stored_as_q8_0(c.type);
        // the weight: in our buffer, and in a CPU buffer with the GGUF bytes (for the lossless layouts the read-back,
        // which must be them), its Q8_0 for the layout requantized to Q8_0 (Q6_K), and its read-back
        ggml_init_params      pw     = { ggml_tensor_overhead() * 4, nullptr, true };
        ggml_context *        ctx_wd = ggml_init(pw);
        ggml_context *        ctx_wc = ggml_init(pw);
        ggml_tensor *         w_dev  = ggml_new_tensor_3d(ctx_wd, c.type, c.k, c.n, c.n_mats);
        ggml_tensor *         w_cpu  = ggml_new_tensor_3d(ctx_wc, c.type, c.k, c.n, c.n_mats);
        ggml_tensor *         w_q8   = requant ? ggml_new_tensor_3d(ctx_wc, GGML_TYPE_Q8_0, c.k, c.n, c.n_mats) : nullptr;
        ggml_tensor *         w_rb   = requant ? ggml_new_tensor_3d(ctx_wc, c.type, c.k, c.n, c.n_mats) : nullptr;
        ggml_set_name(w_dev, "blk.0.ffn_down_exps.weight");
        ggml_backend_buffer_t buf_wd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wd, buft);
        ggml_backend_buffer_t buf_wc = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wc, ggml_backend_cpu_buffer_type());
        REQUIRE(buf_wd != nullptr && buf_wc != nullptr);
        {
            std::vector<uint8_t> q;
            if (c.imatrix) {
                const std::vector<float> v = random_values(c.k * c.n * c.n_mats, seed++, 1.0f);
                const std::vector<float> ones((size_t) c.k, 1.0f);
                q.resize(ggml_row_size(c.type, c.k) * (size_t) (c.n * c.n_mats));
                ggml_quantize_chunk(c.type, v.data(), q.data(), 0, c.n * c.n_mats, c.k, ones.data());
            } else {
                q = quantize_weights(c.type, c.k, c.n * c.n_mats, seed++);
            }
            std::vector<uint8_t> read(q.size());
            ggml_backend_tensor_set(w_dev, q.data(), 0, q.size());
            ggml_backend_tensor_get(w_dev, read.data(), 0, read.size());
            CHECK(requant || read == q);
            ggml_backend_tensor_set(w_cpu, q.data(), 0, q.size());
            if (w_q8 != nullptr) {
                const std::vector<uint8_t> q8 = requantize_q8_0(c.type, q, c.k, c.n * c.n_mats);
                ggml_backend_tensor_set(w_q8, q8.data(), 0, q8.size());
            }
            if (w_rb != nullptr) {
                ggml_backend_tensor_set(w_rb, read.data(), 0, read.size());
            }
        }

        for (const int64_t n_tok : c.tokens) {
            ggml_init_params p     = { ggml_tensor_overhead() * 12 + ggml_graph_overhead() * 4, nullptr, true };
            ggml_context *   ctx_h = ggml_init(p);
            ggml_context *   ctx_d = ggml_init(p);
            REQUIRE(ctx_h != nullptr && ctx_d != nullptr);
            ggml_tensor * x        = ggml_new_tensor_3d(ctx_h, GGML_TYPE_F32, c.k, c.ne11, n_tok);
            ggml_tensor * ids_full = ggml_new_tensor_2d(ctx_h, GGML_TYPE_I32, c.view ? c.n_mats : c.n_used, n_tok);
            ggml_tensor * ids = c.view ? ggml_view_2d(ctx_h, ids_full, c.n_used, n_tok, ids_full->nb[1], 0) : ids_full;
            ggml_tensor * y_ref  = ggml_mul_mat_id(ctx_h, w_cpu, x, ids);
            ggml_tensor * y_q8   = w_q8 != nullptr ? ggml_mul_mat_id(ctx_h, w_q8, x, ids) : nullptr;
            ggml_tensor * y_rb   = w_rb != nullptr ? ggml_mul_mat_id(ctx_h, w_rb, x, ids) : nullptr;
            ggml_tensor * y      = ggml_mul_mat_id(ctx_d, w_dev, x, ids);
            ggml_backend_buffer_t bh = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
            ggml_backend_buffer_t bd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, buft);
            REQUIRE(bh != nullptr && bd != nullptr);

            const std::vector<float> xv = random_values(ggml_nelements(x), seed++, 2.0f);
            ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
            // the ids: every entry of ids_full out of range first, so a read past the view fails the case
            std::vector<int32_t> iv((size_t) ggml_nelements(ids_full));
            for (size_t i = 0; i < iv.size(); i++) {
                iv[i] = (int32_t) (c.n_mats + 1000 + (int64_t) i);
            }
            uint32_t        s     = seed++ * 2654435761u + 1;
            const auto      next  = [&s] { return s = s * 1664525u + 1013904223u, s >> 8; };
            const int64_t   width = ids_full->ne[0];
            std::vector<int32_t> perm((size_t) c.n_mats);
            for (int64_t tk = 0; tk < n_tok; tk++) {
                if (c.ids == ids_same && tk > 0) {  // token 0's experts
                    std::copy(iv.begin(), iv.begin() + c.n_used, iv.begin() + tk * width);
                    continue;
                }
                for (int64_t e = 0; e < c.n_mats; e++) {
                    perm[(size_t) e] = (int32_t) e;
                }
                for (int64_t e = 0; e < c.n_used; e++) {  // a partial Fisher-Yates: distinct experts
                    std::swap(perm[(size_t) e], perm[(size_t) (e + next() % (uint32_t) (c.n_mats - e))]);
                }
                for (int64_t sl = 0; sl < c.n_used; sl++) {  // ids_repeat: each expert twice in a row
                    iv[(size_t) (tk * width + sl)] = perm[(size_t) (c.ids == ids_repeat ? sl / 2 : sl)];
                }
            }
            ggml_backend_tensor_set(ids_full, iv.data(), 0, iv.size() * sizeof(int32_t));

            const int64_t n_out   = ggml_nelements(y);
            const bool    claimed = ggml_backend_dev_supports_op(dev, y);
            ggml_cgraph * g_cpu   = ggml_new_graph(ctx_h);
            ggml_build_forward_expand(g_cpu, y_ref);
            if (y_q8 != nullptr) {
                ggml_build_forward_expand(g_cpu, y_q8);
            }
            if (y_rb != nullptr) {
                ggml_build_forward_expand(g_cpu, y_rb);
            }
            const bool cpu_ok = ggml_backend_graph_compute(cpu, g_cpu) == GGML_STATUS_SUCCESS;

            // two runs through the backend, then the two entries directly on 1 and on 8 tiles; the output filled with
            // NaN before each, so an element no tile writes fails the comparison
            const std::vector<float> junk((size_t) n_out, NAN);
            std::vector<float>       out[4];
            ggml_cgraph *            g_dev = ggml_new_graph(ctx_d);
            ggml_build_forward_expand(g_dev, y);
            bool dev_ok = claimed && cpu_ok;
            for (int r = 0; r < 2 && dev_ok; r++) {
                ggml_backend_tensor_set(y, junk.data(), 0, ggml_nbytes(y));
                dev_ok = ggml_backend_graph_compute(backend, g_dev) == GGML_STATUS_SUCCESS;
                out[r].assign((size_t) n_out, NAN);
                ggml_backend_tensor_get(y, out[r].data(), 0, ggml_nbytes(y));
            }
            const size_t         ws = step_q->workspace_size(y);
            std::vector<uint8_t> workspace(ws);
            for (int r = 2; r < 4 && dev_ok && direct; r++) {
                const uint32_t nth = r == 2 ? 1u : 8u;
                ggml_backend_tensor_set(y, junk.data(), 0, ggml_nbytes(y));
                for (const spacemit_plugin_kernel * step : { step_q, step_g }) {
                    for (uint32_t ith = 0; ith < nth; ith++) {
                        const spacemit_tile tile = { ith, nth, nullptr, 0, workspace.data(), ws, nullptr };
                        dev_ok &= step->fn(&tile, y);
                    }
                }
                out[r].assign((size_t) n_out, NAN);
                ggml_backend_tensor_get(y, out[r].data(), 0, ggml_nbytes(y));
            }
            const size_t out_bytes = (size_t) n_out * sizeof(float);
            const bool   same      = dev_ok && std::memcmp(out[0].data(), out[1].data(), out_bytes) == 0 &&
                              (!direct || (std::memcmp(out[0].data(), out[2].data(), out_bytes) == 0 &&
                                           std::memcmp(out[0].data(), out[3].data(), out_bytes) == 0));
            const double e    = dev_ok ? nmse(out[0].data(), (const float *) y_ref->data, n_out) : -1.0;
            const double e_q8 = dev_ok && y_q8 != nullptr ? nmse(out[0].data(), (const float *) y_q8->data, n_out) : 0.0;
            if (dev_ok && y_rb != nullptr) {
                double & wr = worst_readback[c.imatrix ? 1 : 0];
                wr          = std::max(wr, nmse(out[0].data(), (const float *) y_rb->data, n_out));
            }
            const bool ok = dev_ok && same && e >= 0.0 && e <= k_mul_mat_max_nmse && e_q8 <= k_requant_max_nmse;
            n_total++;
            n_ok += ok;
            n_same += same;
            n_runs[t]++;
            worst[t]    = std::max(worst[t], e);
            worst_q8[t] = std::max(worst_q8[t], e_q8);
            if (!ok) {
                std::fprintf(stderr,
                             "FAIL: MUL_MAT_ID %s, %s [%lld, %lld, %lld], %lld ids, %lld rows per token, %lld tokens: "
                             "%s, %s; the runs and 1 and 8 tiles %s; NMSE %.3e (bound %.0e)%s\n",
                             c.what, ggml_type_name(c.type), (long long) c.k, (long long) c.n, (long long) c.n_mats,
                             (long long) c.n_used, (long long) c.ne11, (long long) n_tok,
                             claimed ? "claimed" : "refused", dev_ok ? "ran" : "failed", same ? "equal" : "differ", e,
                             k_mul_mat_max_nmse, requant ? (e_q8 <= k_requant_max_nmse ? "" : ", vs Q8_0 over bound") : "");
                g_failures++;
            }
            ggml_backend_buffer_free(bh);
            ggml_backend_buffer_free(bd);
            ggml_free(ctx_h);
            ggml_free(ctx_d);
        }
        ggml_backend_buffer_free(buf_wd);
        ggml_backend_buffer_free(buf_wc);
        ggml_free(ctx_wd);
        ggml_free(ctx_wc);
    }

    // an id outside the experts (past the last, and -1) fails the graph, and the backend runs the next graph
    int n_bad = 0, n_bad_ok = 0;
    {
        ggml_init_params p     = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context *   ctx_w = ggml_init(p);
        ggml_context *   ctx_h = ggml_init(p);
        ggml_tensor *    w     = ggml_new_tensor_3d(ctx_w, GGML_TYPE_Q8_0, 256, 32, 4);
        ggml_tensor *    x     = ggml_new_tensor_3d(ctx_h, GGML_TYPE_F32, 256, 1, 3);
        ggml_tensor *    ids   = ggml_new_tensor_2d(ctx_h, GGML_TYPE_I32, 2, 3);
        ggml_tensor *    y     = ggml_mul_mat_id(ctx_w, w, x, ids);
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
        ggml_backend_buffer_t bh = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
        REQUIRE(bw != nullptr && bh != nullptr);
        const std::vector<uint8_t> q  = quantize_weights(GGML_TYPE_Q8_0, 256, 128, 31);
        const std::vector<float>   xv = random_values(256 * 3, 32, 1.0f);
        ggml_backend_tensor_set(w, q.data(), 0, q.size());
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_cgraph * g = ggml_new_graph(ctx_w);
        ggml_build_forward_expand(g, y);
        for (const int32_t bad : { 4, -1, 1000, INT32_MIN }) {
            const int32_t good[6] = { 0, 1, 2, 3, 1, 2 };
            int32_t       iv[6];
            std::memcpy(iv, good, sizeof(iv));
            iv[3] = bad;
            ggml_backend_tensor_set(ids, iv, 0, sizeof(iv));
            const bool failed = ggml_backend_graph_compute(backend, g) != GGML_STATUS_SUCCESS;
            ggml_backend_tensor_set(ids, good, 0, sizeof(good));
            const bool again = ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS;
            n_bad++;
            n_bad_ok += failed && again;
            if (!failed || !again) {
                std::fprintf(stderr, "FAIL: MUL_MAT_ID with id %d: the graph %s, the next one %s\n", (int) bad,
                             failed ? "failed" : "ran", again ? "ran" : "failed");
                g_failures++;
            }
        }
        ggml_backend_buffer_free(bw);
        ggml_backend_buffer_free(bh);
        ggml_free(ctx_w);
        ggml_free(ctx_h);
    }

    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    const char * kernels_used = reference_label();
    std::printf("mmid     MUL_MAT_ID vs the CPU on one thread (the model's gate/up and down at 1-512 tokens, 8 ids; small "
                "shapes with 1, 2, 4 and 8 ids, broadcast and a row per id; ids as a view, the same for every token, "
                "repeating within a token; a row per two ids): %d/%d cases, the two runs and 1 and 8 tiles bit for "
                "bit in %d/%d; max NMSE Q4_0 %.2e (%d), Q8_0 %.2e (%d), Q6_K %.2e (%d; vs CPU's Q8_0 of its values "
                "%.2e), Q5_0 %.2e (%d) (bounds %.0e and %.0e)%s\n",
                n_ok, n_total, n_same, n_total, worst[0], n_runs[0], worst[1], n_runs[1], worst[2], n_runs[2],
                worst_q8[2], worst[3], n_runs[3], k_mul_mat_max_nmse, k_requant_max_nmse, kernels_used);
    std::printf("mmid     claims and refused forms (a 3-D Q4_1 weight claimed, but not of K 128, also in our buffer; "
                "rows per expert not a multiple of 32, Q4_0 rows not a multiple of 256, F32 and Q4_K weights, a viewed "
                "weight, F16 or strided activations, a CPU-buffer weight, MUL_MAT on a 3-D weight) %d/%d; ids outside "
                "the experts fail the graph %d/%d\n",
                n_claims_ok, n_claims, n_bad_ok, n_bad);
    std::printf("info     MUL_MAT_ID Q6_K against the CPU on the weight's read-back (the nearest Q6_K to the stored "
                "Q8_0, what test-backend-ops' CPU computes with): max NMSE %.2e, at test-backend-ops' three Q6_K "
                "shapes quantized with an imatrix %.2e (bound %.0e)\n",
                worst_readback[0], worst_readback[1], k_mul_mat_max_nmse);
    return check_mul_mat_id_groups(dev, host) != 0 || check_mul_mat_id_queue(dev, host) != 0 ||
                   check_mul_mat_id_q4_1(dev, host) != 0 || check_mul_mat_id_q4_0s(dev, host) != 0 ?
               1 :
               0;
}

static int check_gdn(ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);  // a correctness reference: single-threaded (PROJECT.md item 28)

    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    REQUIRE(kernels != nullptr);
    const spacemit_plugin_kernel * gdn = nullptr;
    for (uint32_t i = 0; i < n_kernels; i++) {
        if (kernels[i].name != nullptr && std::strcmp(kernels[i].name, "gdn.f32") == 0) {
            gdn = &kernels[i];
        }
    }
    REQUIRE(gdn != nullptr && gdn->fn != nullptr && gdn->workspace_size != nullptr);
    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    const gdn_case cases[] = {
  // what,                                                 S_v,   H, tok, seq, H_qk, seq_qk,   kda, K
        { "scalar gate, 1 token",                            16,   4,   1,   1,    4,      1, false, 1 },
        { "KDA gate, 1 token",                               16,   4,   1,   1,    4,      1,  true, 1 },
        { "scalar gate, 16 tokens",                          16,   4,  16,   1,    4,      1, false, 1 },
        { "KDA gate, 16 tokens",                             16,   4,  16,   1,    4,      1,  true, 1 },
        { "q and k over 2 heads of 8, 3 tokens",             16,   8,   3,   1,    2,      1, false, 1 },
        { "2 sequences, 2 tokens",                           16,   4,   2,   2,    4,      2, false, 1 },
        { "2 sequences, q and k repeating over both, KDA",   16,   8,   2,   2,    2,      1,  true, 1 },
        { "K 4, 2 tokens (fewer than K)",                    16,   4,   2,   1,    4,      1, false, 4 },
        { "K 4, 4 tokens (K)",                               16,   4,   4,   1,    4,      1, false, 4 },
        { "K 4, 6 tokens (more than K)",                     16,   4,   6,   1,    4,      1, false, 4 },
        { "K 2, 2 sequences, 3 tokens, KDA",                 16,   4,   3,   2,    4,      2,  true, 2 },
        { "9 rows of 7 (not a multiple of the tiles)",         7,   9,   5,   1,    3,      1, false, 1 },
        { "Qwen3.5-4B: 32 heads, S_v 128, 1 token",          128,  32,   1,   1,   16,      1, false, 1 },
        { "Qwen3.5-4B: 32 heads, S_v 128, 16 tokens",        128,  32,  16,   1,   16,      1, false, 1 },
        { "Qwen3.5-4B: 32 heads, S_v 128, KDA, K 2",         128,  32,   2,   1,   16,      1,  true, 2 },
        // the RVV kernel's forms by the vector length: one token with a row of one m4 group at 256 bits (the
        // look-ahead), m8 rows at 256 bits, a side of 256 (fused in m8 at 1024 bits, two-pass at 256), 5 tokens (past
        // the fused form's 4)
        { "S_v 32, 1 token",                                  32,   4,   1,   1,    4,      1, false, 1 },
        { "S_v 64, 3 tokens, KDA",                            64,   4,   3,   1,    4,      1,  true, 1 },
        { "S_v 256, 1 token",                                256,   2,   1,   1,    2,      1, false, 1 },
        { "S_v 256, 4 tokens, K 2",                          256,   2,   4,   1,    2,      1, false, 2 },
        { "S_v 128, 5 tokens, KDA",                          128,   4,   5,   1,    4,      1,  true, 1 },
    };

    int      n_cases = 0, n_ok = 0;
    double   worst_nmse = 0.0, worst_abs = 0.0;
    uint32_t seed = 700;
    for (const gdn_case & c : cases) {
        for (bool host : { false, true }) {
            n_cases++;
            gdn_run dv, ref, dir;
            if (!dv.init(c, host ? ggml_backend_cpu_buffer_type() : buft) ||
                !ref.init(c, ggml_backend_cpu_buffer_type()) || !dir.init(c, ggml_backend_cpu_buffer_type())) {
                std::fprintf(stderr, "FAIL: GATED_DELTA_NET %s: allocation failed\n", c.what);
                g_failures++;
                continue;
            }
            dv.fill(seed);
            ref.fill(seed);
            dir.fill(seed);

            const int64_t n       = ggml_nelements(dv.node);
            const bool    claimed = ggml_backend_dev_supports_op(dev, dv.node);
            // the op writes the attention scores and the newest min(n_tokens, K) snapshot slots; the older slots of a
            // batch shorter than K belong to the caller (ggml-cpu leaves them too), so only this prefix is compared
            const int64_t n_cmp = c.S_v * c.H * c.n_tokens * c.n_seqs +
                                  std::min(c.n_tokens, c.K) * c.S_v * c.S_v * c.H * c.n_seqs;

            ggml_cgraph * g_ref = ggml_new_graph(ref.ctx);
            ggml_build_forward_expand(g_ref, ref.node);
            const bool ref_ok = ggml_backend_graph_compute(cpu, g_ref) == GGML_STATUS_SUCCESS;
            std::vector<float> want((size_t) n, NAN);
            if (ref_ok) {
                ggml_backend_tensor_get(ref.node, want.data(), 0, (size_t) n * sizeof(float));
            }

            // the provider's run, twice: the second must give the same bytes
            ggml_cgraph * g_dev = ggml_new_graph(dv.ctx);
            ggml_build_forward_expand(g_dev, dv.node);
            std::vector<float> got[2];
            bool               dev_ok = claimed && ref_ok;
            for (int r = 0; r < 2 && dev_ok; r++) {
                const std::vector<float> junk((size_t) n, (float) (-r - 1));  // rows no tile wrote stay wrong
                ggml_backend_tensor_set(dv.node, junk.data(), 0, junk.size() * sizeof(float));
                dev_ok = ggml_backend_graph_compute(backend, g_dev) == GGML_STATUS_SUCCESS;
                if (dev_ok) {
                    got[r].assign((size_t) n, NAN);
                    ggml_backend_tensor_get(dv.node, got[r].data(), 0, (size_t) n * sizeof(float));
                }
            }
            const bool   repeat =
                dev_ok && std::memcmp(got[0].data(), got[1].data(), (size_t) n_cmp * sizeof(float)) == 0;
            const double e_dev   = dev_ok ? nmse(got[0].data(), want.data(), n_cmp) : -1.0;
            double       abs_dev = 0.0;
            for (int64_t i = 0; dev_ok && i < n_cmp; i++) {
                abs_dev = std::max(abs_dev, (double) std::fabs(got[0][i] - want[i]));
            }

            // the kernel entry called directly on 1 and on 8 tiles, on the CPU-buffer copy
            const size_t       ws = gdn->workspace_size(dir.node);
            std::vector<float> workspace(ws / sizeof(float), NAN);
            std::vector<float> direct[2];
            bool               direct_ok = ws > 0;
            double             e_direct  = -1.0;
            for (int t = 0; t < 2 && direct_ok; t++) {
                const uint32_t nth = t == 0 ? 1u : 8u;
                float *        out = static_cast<float *>(dir.node->data);
                std::fill(out, out + n, NAN);
                bool ok = true;
                for (uint32_t ith = 0; ith < nth; ith++) {
                    const spacemit_tile tile = { ith, nth, nullptr, 0, workspace.data(), ws, nullptr };
                    ok &= gdn->fn(&tile, dir.node);
                }
                direct_ok = ok;
                direct[t].assign(out, out + n);
            }
            if (direct_ok && ref_ok) {
                e_direct = nmse(direct[0].data(), want.data(), n_cmp);
                for (int64_t i = 0; i < n_cmp; i++) {
                    worst_abs = std::max(worst_abs, (double) std::fabs(direct[0][i] - want[i]));
                }
            }
            const bool tiles_same =
                direct_ok && std::memcmp(direct[0].data(), direct[1].data(), (size_t) n_cmp * sizeof(float)) == 0;

            worst_nmse = std::max(worst_nmse, std::max(e_dev, e_direct));
            worst_abs  = std::max(worst_abs, abs_dev);
            const bool ok = claimed && ref_ok && dev_ok && repeat && direct_ok && tiles_same && e_dev >= 0.0 &&
                            e_dev <= k_gdn_max_nmse && e_direct >= 0.0 && e_direct <= k_gdn_max_nmse;
            n_ok += ok;
            if (!ok) {
                std::fprintf(stderr,
                             "FAIL: GATED_DELTA_NET %s, in %s: %s; the CPU reference %s, the run %s, its repeat %s, "
                             "the direct calls %s, 1 and 8 tiles %s; NMSE %.3e (direct %.3e, bound %.0e), "
                             "max |difference| %.3e\n",
                             c.what, host ? "a CPU buffer" : "our buffer", claimed ? "claimed" : "refused",
                             ref_ok ? "ran" : "failed", dev_ok ? "ran" : "failed", repeat ? "equal" : "differs",
                             direct_ok ? "ran" : "failed", tiles_same ? "equal" : "differ", e_dev, e_direct,
                             k_gdn_max_nmse, abs_dev);
                g_failures++;
            }
        }
        seed += 8;
    }

    // the prefill forms at Qwen3.5-4B's side (S_v 128, K 1, scalar gate, the model's inputs) from the chunk's 16
    // tokens, through the device only: a direct call runs on the calling thread, on the K3 an X100 core (VLEN 256, no
    // TCM), where neither prefill form is taken. Against ggml-cpu, the scores and the final state apart, within the
    // chunkwise form's bound (another order of operations); each run repeated bit for bit. The tiles count the form
    // they ran (spacemit_gdn_paths_take): with the RVV kernels the TCM form where its conditions hold (S_v % 4 == 0,
    // vlmax(e32m2) <= S_v <= vlmax(e32m4) and a tile's TCM holds it), else the chunkwise form (S_v <= vlmax(e32m4)), so
    // S_v 32 takes the chunkwise one; without them (another host, or GGML_FLAGOS_KERNEL_VARIANTS=0) the reference
    const gdn_case prefill_cases[] = {
  // what,                                                 S_v,   H, tok, seq, H_qk, seq_qk,   kda, K
        { "16 heads, 16 tokens (one chunk)",                128,  16,  16,   1,   16,      1, false, 1 },
        { "16 heads, 17 tokens (a 1-token last chunk)",     128,  16,  17,   1,   16,      1, false, 1 },
        { "16 heads, 33 tokens",                            128,  16,  33,   1,   16,      1, false, 1 },
        { "16 heads, 100 tokens",                           128,  16, 100,   1,   16,      1, false, 1 },
        { "4 heads, q and k over 2, 33 tokens (idle tiles)", 128,  4,  33,   1,    2,      1, false, 1 },
        { "S_v 32, 16 heads, 33 tokens",                     32,  16,  33,   1,   16,      1, false, 1 },
    };
    constexpr double k_gdn_prefill_max_nmse = 1e-6;  // on the scores and on the state
    int              n_prefill = 0, n_prefill_ok = 0;
    double           worst_prefill[2] = {};  // the scores' NMSE, the state's
    // the counters' forms, in spacemit_gdn_paths_take's order; the cases in which the tiles took each
    const char * const form_names[5] = { "reference", "fused", "two-pass", "chunk", "tcm" };
    int                n_took[5]     = {};
#if defined(__riscv)
    const bool rvv = std::strlen(reference_label()) == 0;
#else
    const bool rvv = false;
#endif
    size_t   tcm_per_tile = 0;
    uint32_t vector_bits  = 0;
    {
        spacemit_plugin_device * pdev = nullptr;
        if (table->device_open(0, &pdev) == SPACEMIT_STATUS_SUCCESS && pdev != nullptr) {
            spacemit_plugin_device_props props = {};
            props.struct_size                  = sizeof(props);
            if (table->device_props(pdev, &props) == SPACEMIT_STATUS_SUCCESS) {
                tcm_per_tile = props.tcm_bytes_per_tile;
                vector_bits  = props.vector_bits;
            }
            table->device_close(pdev);
        }
    }
    const int64_t vlmax_m2 = vector_bits / 32 * 2, vlmax_m4 = vector_bits / 32 * 4;  // e32 at the AI cores' VLEN
    size_t        tcm_needed = 0;  // the TCM form's state, k, q and v, and up to 128 bytes to align them
    for (const gdn_case & c : prefill_cases) {
        gdn_run dv, ref;
        if (!dv.init(c, buft) || !ref.init(c, ggml_backend_cpu_buffer_type())) {
            std::fprintf(stderr, "FAIL: GATED_DELTA_NET prefill %s: allocation failed\n", c.what);
            g_failures++;
            continue;
        }
        dv.fill_model(seed);
        ref.fill_model(seed);
        seed += 8;

        const int64_t n       = ggml_nelements(dv.node);
        const int64_t n_attn  = c.S_v * c.H * c.n_tokens * c.n_seqs;
        const int64_t n_cmp   = n_attn + c.S_v * c.S_v * c.H * c.n_seqs;
        const bool    claimed = ggml_backend_dev_supports_op(dev, dv.node);

        ggml_cgraph * g_ref = ggml_new_graph(ref.ctx);
        ggml_build_forward_expand(g_ref, ref.node);
        const bool         ref_ok = ggml_backend_graph_compute(cpu, g_ref) == GGML_STATUS_SUCCESS;
        std::vector<float> want((size_t) n, NAN);
        if (ref_ok) {
            ggml_backend_tensor_get(ref.node, want.data(), 0, (size_t) n * sizeof(float));
        }

        ggml_cgraph * g_dev = ggml_new_graph(dv.ctx);
        ggml_build_forward_expand(g_dev, dv.node);
        const size_t tcm_bytes = (size_t) (c.S_v * c.S_v + 3 * c.S_v) * sizeof(float) + 128;
        tcm_needed             = std::max(tcm_needed, tcm_bytes);
        n_prefill++;
        std::vector<float> got[2];
        bool               dev_ok = claimed && ref_ok;
        uint64_t           taken[5];
        g_gdn_paths_take(&taken[0], &taken[1], &taken[2], &taken[3], &taken[4]);  // drops earlier calls
        for (int r = 0; r < 2 && dev_ok; r++) {
            const std::vector<float> junk((size_t) n, (float) (-r - 1));
            ggml_backend_tensor_set(dv.node, junk.data(), 0, junk.size() * sizeof(float));
            dev_ok = ggml_backend_graph_compute(backend, g_dev) == GGML_STATUS_SUCCESS;
            if (dev_ok) {
                got[r].assign((size_t) n, NAN);
                ggml_backend_tensor_get(dv.node, got[r].data(), 0, (size_t) n * sizeof(float));
            }
        }
        g_gdn_paths_take(&taken[0], &taken[1], &taken[2], &taken[3], &taken[4]);
        // the form the tiles must take: tcm 4, chunk 3, -1 a recurrence form (fused or two-pass), 0 the reference
        const bool tcm_form  = c.S_v % 4 == 0 && c.S_v >= vlmax_m2 && c.S_v <= vlmax_m4 && tcm_per_tile >= tcm_bytes;
        const int  want_form = !rvv ? 0 : tcm_form ? 4 : c.S_v <= vlmax_m4 ? 3 : -1;
        bool       forms_ok  = want_form >= 0 ? taken[want_form] > 0 : taken[1] + taken[2] > 0;
        for (int f = 0; f < 5; f++) {
            forms_ok = forms_ok && (taken[f] == 0 || f == want_form || (want_form < 0 && (f == 1 || f == 2)));
            n_took[f] += dev_ok && taken[f] > 0;
        }
        const bool   repeat  = dev_ok && std::memcmp(got[0].data(), got[1].data(), (size_t) n_cmp * sizeof(float)) == 0;
        const double e_out   = dev_ok ? nmse(got[0].data(), want.data(), n_attn) : -1.0;
        const double e_state = dev_ok ? nmse(got[0].data() + n_attn, want.data() + n_attn, n_cmp - n_attn) : -1.0;
        const bool   in_bound = e_out >= 0.0 && e_out <= k_gdn_prefill_max_nmse && e_state >= 0.0 &&
                              e_state <= k_gdn_prefill_max_nmse;
        const bool ok = claimed && ref_ok && dev_ok && repeat && in_bound && forms_ok;
        n_prefill_ok += ok;
        worst_prefill[0] = std::max(worst_prefill[0], e_out);
        worst_prefill[1] = std::max(worst_prefill[1], e_state);
        if (!ok) {
            std::fprintf(stderr,
                         "FAIL: GATED_DELTA_NET prefill %s: %s; the CPU reference %s, the run %s, its repeat %s; NMSE "
                         "scores %.3e, state %.3e (bound %.0e); tile calls by form: reference %llu, fused %llu, "
                         "two-pass %llu, chunk %llu, tcm %llu (expected %s; TCM %zu bytes per tile, the form needs "
                         "%zu)\n",
                         c.what, claimed ? "claimed" : "refused", ref_ok ? "ran" : "failed", dev_ok ? "ran" : "failed",
                         repeat ? "equal" : "differs", e_out, e_state, k_gdn_prefill_max_nmse,
                         (unsigned long long) taken[0], (unsigned long long) taken[1], (unsigned long long) taken[2],
                         (unsigned long long) taken[3], (unsigned long long) taken[4],
                         want_form >= 0 ? form_names[want_form] : "fused or two-pass", tcm_per_tile, tcm_bytes);
            g_failures++;
        }
    }

    // what the contract takes and what it refuses
    {
        ggml_init_params p   = { ggml_tensor_overhead() * 64, nullptr, true };
        ggml_context *   ctx = ggml_init(p);
        REQUIRE(ctx != nullptr);
        const auto f32 = [&](int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
            return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
        };
        // the taken shape: S_v 8, 4 heads, 2 tokens, 1 sequence
        ggml_tensor * q = f32(8, 4, 2, 1), * k = f32(8, 4, 2, 1), * v = f32(8, 4, 2, 1);
        ggml_tensor * g = f32(1, 4, 2, 1), * b = f32(1, 4, 2, 1), * s = f32(8, 8, 4, 1);
        CHECK(ggml_backend_dev_supports_op(dev, ggml_gated_delta_net(ctx, q, k, v, g, b, s, 1)));
        CHECK(ggml_backend_dev_supports_op(dev, ggml_gated_delta_net(ctx, q, k, v, f32(8, 4, 2, 1), b, s, 3)));
        // refused: a state side over the contract's bound (256)
        {
            ggml_tensor * qb = f32(512, 1, 2, 1), * vb = f32(512, 1, 2, 1), * sb = f32(512, 512, 1, 1);
            CHECK(!ggml_backend_dev_supports_op(
                dev, ggml_gated_delta_net(ctx, qb, qb, vb, f32(1, 1, 2, 1), f32(1, 1, 2, 1), sb, 1)));
        }
        // refused: q heads that do not divide H, q missing a token, q sequences that do not divide n_seqs
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_gated_delta_net(ctx, f32(8, 3, 2, 1), k, v, g, b, s, 1)));
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_gated_delta_net(ctx, f32(8, 4, 1, 1), k, v, g, b, s, 1)));
        {
            ggml_tensor * v2 = f32(8, 4, 2, 2), * s2 = f32(8, 8, 4, 2);
            ggml_tensor * g2 = f32(1, 4, 2, 2), * b2 = f32(1, 4, 2, 2);
            CHECK(ggml_backend_dev_supports_op(dev, ggml_gated_delta_net(ctx, q, k, v2, g2, b2, s2, 1)));  // q repeats
            CHECK(!ggml_backend_dev_supports_op(
                dev, ggml_gated_delta_net(ctx, f32(8, 4, 2, 3), k, v2, g2, b2, s2, 1)));
        }
        // refused: a gate or a beta of the wrong head count
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_gated_delta_net(ctx, q, k, v, f32(1, 1, 2, 1), b, s, 1)));
        CHECK(!ggml_backend_dev_supports_op(dev, ggml_gated_delta_net(ctx, q, k, v, g, f32(1, 1, 2, 1), s, 1)));
        // refused, through nodes built by hand: an F16 operand or output, a gate of neither 1 nor S_v, a state whose
        // rows have gaps, an output one row short
        ggml_tensor * q16 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 8, 4, 2, 1);
        CHECK(ggml_backend_dev_supports_op(dev, gdn_raw(ctx, GGML_TYPE_F32, q, k, v, g, b, s, 1)));  // the builder
        CHECK(!ggml_backend_dev_supports_op(dev, gdn_raw(ctx, GGML_TYPE_F32, q16, k, v, g, b, s, 1)));
        CHECK(!ggml_backend_dev_supports_op(dev, gdn_raw(ctx, GGML_TYPE_F16, q, k, v, g, b, s, 1)));
        CHECK(!ggml_backend_dev_supports_op(dev, gdn_raw(ctx, GGML_TYPE_F32, q, k, v, f32(2, 4, 2, 1), b, s, 1)));
        ggml_tensor * wide   = f32(16, 8, 4, 1);
        ggml_tensor * s_gaps = ggml_view_4d(ctx, wide, 8, 8, 4, 1, wide->nb[1], wide->nb[2], wide->nb[3], 0);
        CHECK(!ggml_backend_dev_supports_op(dev, gdn_raw(ctx, GGML_TYPE_F32, q, k, v, g, b, s_gaps, 1)));
        CHECK(!ggml_backend_dev_supports_op(dev, gdn_raw(ctx, GGML_TYPE_F32, q, k, v, g, b, s, 1, 9)));
        ggml_free(ctx);
    }

    std::string took;  // the forms the tiles took and in how many cases
    for (int f = 0; f < 5; f++) {
        if (n_took[f] > 0) {
            took += std::string(took.empty() ? "" : ", ") + form_names[f] + " form in " + std::to_string(n_took[f]) +
                    "/" + std::to_string(std::size(prefill_cases));
        }
    }
    took = took.empty() ? "none" : took;

    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    std::printf("gdn      GATED_DELTA_NET in our buffer and in a CPU buffer (scalar gate and KDA, K 1, 2 and 4 with "
                "fewer, as many and more tokens than K, 1 and 2 sequences, q and k repeating over heads and "
                "sequences, 1 to 16 tokens, Qwen3.5-4B's 32 heads of 128, sides of 32, 64 and 256 for the RVV "
                "kernel's row forms) %d/%d: max NMSE against the CPU on one "
                "thread %.2e (bound %.0e), max |difference| %.2e, the run's repeat and 1 against 8 tiles bit for "
                "bit (over the scores and the newest min(n_tokens, K) snapshot slots, the only part the op writes); "
                "an F16 operand or output, a state side over 256, a state with gaps, q heads or sequences that "
                "do not divide v's, a gate of neither 1 nor S_v, a gate or beta of the wrong head count, a short "
                "output refused: checked\n",
                n_ok, n_cases, worst_nmse, k_gdn_max_nmse, worst_abs);
    std::printf("gdn      the prefill forms (the TCM form, else the chunkwise one, else the recurrence) on the device, "
                "K 1, the model's inputs, S_v 128: 16, 17, 33 and 100 tokens of 16 heads and 33 of 4 (idle tiles), "
                "S_v 32: 33 tokens of 16 heads%s: %d/%d; max NMSE against the CPU on one thread, scores / state %.2e / "
                "%.2e (bound %.0e); each run's repeat bit for bit; the tiles' forms by the counters (TCM %zu bytes per "
                "tile, the tcm form needs %zu): %s\n",
                reference_label(), n_prefill_ok, n_prefill, worst_prefill[0], worst_prefill[1],
                k_gdn_prefill_max_nmse, tcm_per_tile, tcm_needed, took.c_str());
    return 0;
}

//
// The state pass (core/flagos-spacemit-ops.h spacemit_state_pass, plugin/flagos-spacemit-gdn.cpp): a GATED_DELTA_NET
// with K 1 whose state comes from a GET_ROWS of one row of the persistent state tensor and whose new state one CPY
// writes back into that tensor runs as one step, gated_delta_net.f32.direct, on the CPY node; the GET_ROWS and the CPY
// get no steps. The graphs are the model's (src/llama-graph.cpp build_rs with src/models/delta-net-base.cpp
// build_recurrent_attn, the !keep branch): the states tensor of several rows in its own context and buffer, an I32
// index input, build_rs' zero-sized SCALE and its empty extra GET_ROWS and CPY, the reshape, the op, a CONT of the
// attention scores (the layer's next op) and the CPY of the new state into row head.
//
// Through the device against ggml-cpu on one thread: the scores and the written row of the states within check_gdn's
// bound, the rows the CPY does not write bit for bit the inputs, the pass counted (spacemit_state_passes_take). The
// step is also called directly on 1, 3 and 8 tiles on a CPU-buffer copy: a row is one tile's whole work, so the three
// must agree bit for bit and stay within the bound; they equal the device's bytes only where both run the same kernels
// (the reference ones, or any host but the K3, whose calling core has another vector length - as check_gdn's direct
// calls).
// Then one graph through the scheduler (its allocator reusing memory) under the plan cache: a miss, then a trusted hit
// counting no pass, the same bytes; and two computes in a row with the state carried over, against the CPU doing the
// same.
//
// A case with zrow >= 0 is a sequence's first graph: build_rs' SCALE by 0 really clears row zrow there (clear says
// how). Where it clears exactly the row the CPY writes, the pass folds it into its step
// (gated_delta_net.f32.direct.cleared, by the trace's step names, with no scale.f32 of its own); otherwise the SCALE
// keeps its step and the pass takes the graph all the same, except with a write of the states between the two copies,
// which refuses it.
//
struct state_case {
    const char * what;
    int64_t      S_v, H, n_tokens, n_seqs, K;
    int64_t      slots, idx, head;
    int          variant;  // 0 the model's graph, 1 a second consumer of the row read in, 2 another view of the new
                           // state, 3 a read of the states between the two copies, 4 the CPY into another tensor
    bool         pass;
    int64_t      zrow    = -1;     // the row build_rs' SCALE clears; -1 its zero-sized view, as a later graph has it
    int          clear   = 0;      // how that SCALE is built (state_clear)
    bool         cleared = false;  // the fold expected: the step clears the row itself and the SCALE emits none
    bool         wild    = false;  // NaN, infinities, negative values and -0.0 in the cleared row (then bytes, not
                                   // NMSE, against the CPU: with a NaN in the state every value computed is a NaN)
};

// how a case builds build_rs' clearing SCALE (zrow >= 0)
enum state_clear {
    clear_whole = 0,  // the whole row zrow in place by 0, before the GET_ROWS, as build_rs has it
    clear_half,       // its first half only
    clear_two,        // rows zrow and zrow + 1
    clear_factor,     // the whole row by 0.5
    clear_reader,     // the whole row, with a read of the states between it and the GET_ROWS
    clear_late,       // the whole row, between the GET_ROWS and the CPY
};

// one copy of a case: the persistent states in their own context and buffer, the inputs and the graph in another
struct state_graph {
    ggml_context *        pctx = nullptr, * ctx = nullptr;
    ggml_backend_buffer_t pbuf = nullptr, buf = nullptr;
    ggml_tensor *         persist = nullptr;  // the states, one row per slot
    ggml_tensor *         idx     = nullptr;  // build_rs' s_copy
    ggml_tensor *         in[5]   = { nullptr, nullptr, nullptr, nullptr, nullptr };  // q, k, v, the gate, beta
    ggml_tensor *         rows = nullptr, * gdn = nullptr, * cpy = nullptr;
    ggml_tensor *         scores = nullptr;  // the CONT of the attention scores
    ggml_tensor *         extra  = nullptr;  // a variant's other output
    ggml_cgraph *         g      = nullptr;
    std::vector<uint8_t>  persist_in;  // the states as the case filled them

    state_graph() = default;
    state_graph(const state_graph &) = delete;
    state_graph & operator=(const state_graph &) = delete;

    ~state_graph() {
        ggml_backend_buffer_free(buf);
        ggml_backend_buffer_free(pbuf);
        ggml_free(ctx);
        ggml_free(pctx);
    }

    // with ops_unallocated only the inputs and the states are allocated (the scheduler allocates the rest)
    bool init(const state_case & c, ggml_backend_buffer_type_t buft, ggml_backend_buffer_type_t pbuft,
              bool ops_unallocated = false) {
        ggml_init_params params = { ggml_tensor_overhead() * 48 + ggml_graph_overhead(), nullptr, true };
        pctx                    = ggml_init(params);
        ctx                     = ggml_init(params);
        if (pctx == nullptr || ctx == nullptr) {
            return false;
        }
        const int64_t state_size = c.S_v * c.S_v * c.H;
        const size_t  row_size   = (size_t) state_size * sizeof(float);
        persist                  = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, state_size, c.slots);
        ggml_set_name(persist, "cache_s_l0");  // llama.cpp's name for it
        pbuf = ggml_backend_alloc_ctx_tensors_from_buft(pctx, pbuft);

        in[0] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H, c.n_tokens, c.n_seqs);
        in[1] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H, c.n_tokens, c.n_seqs);
        in[2] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H, c.n_tokens, c.n_seqs);
        in[3] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, c.H, c.n_tokens, c.n_seqs);
        in[4] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, c.H, c.n_tokens, c.n_seqs);
        idx   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.n_seqs);
        ggml_set_input(idx);
        ggml_tensor * target = c.variant == 4 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_size, c.n_seqs) : nullptr;
        if (ops_unallocated) {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }

        // build_rs: the reshape, the clear of one slot (zero-sized with rs_zero -1, as a generation graph has it; one
        // row of it at a sequence's start), the row of this sequence, and the extra rows (none here) copied back
        ggml_tensor * states = ggml_reshape_2d(ctx, persist, state_size, c.slots);
        const int64_t zn     = c.zrow < 0            ? 0 :
                               c.clear == clear_half ? state_size / 2 :
                               c.clear == clear_two  ? 2 * state_size : state_size;
        ggml_tensor * zero   = ggml_view_1d(ctx, states, zn, c.zrow < 0 ? 0 : (size_t) c.zrow * states->nb[1]);
        ggml_tensor * clear  = ggml_scale_inplace(ctx, zero, c.clear == clear_factor ? 0.5f : 0.0f);
        rows                 = ggml_get_rows(ctx, states, ggml_view_1d(ctx, idx, c.n_seqs, 0));
        ggml_tensor * rows_x = ggml_get_rows(ctx, states,
                                             ggml_view_1d(ctx, idx, 0, (size_t) c.n_seqs * ggml_element_size(idx)));
        ggml_tensor * cpy_x  = ggml_cpy(ctx, rows_x,
                                        ggml_view_2d(ctx, persist, state_size, 0, persist->nb[1],
                                                     (size_t) (c.head + c.n_seqs) * row_size));

        ggml_tensor * state4d = ggml_reshape_4d(ctx, rows, c.S_v, c.S_v, c.H, c.n_seqs);
        gdn                   = ggml_gated_delta_net(ctx, in[0], in[1], in[2], in[3], in[4], state4d, c.K);
        const int64_t attn    = c.S_v * c.H * c.n_tokens * c.n_seqs;
        ggml_tensor * out     = ggml_view_4d(ctx, gdn, c.S_v, c.H, c.n_tokens, c.n_seqs,
                                             ggml_row_size(gdn->type, c.S_v), ggml_row_size(gdn->type, c.S_v * c.H),
                                             ggml_row_size(gdn->type, c.S_v * c.H * c.n_tokens), 0);
        ggml_tensor * new_st  = ggml_view_4d(ctx, gdn, c.S_v, c.S_v, c.H, c.n_seqs,
                                             ggml_row_size(gdn->type, c.S_v), ggml_row_size(gdn->type, c.S_v * c.S_v),
                                             ggml_row_size(gdn->type, c.S_v * c.S_v * c.H),
                                             ggml_row_size(gdn->type, attn));
        scores                = ggml_cont(ctx, out);
        ggml_set_output(scores);
        cpy = ggml_cpy(ctx, new_st,
                       c.variant == 4 ? target :
                                        ggml_view_2d(ctx, persist, state_size, c.n_seqs, persist->nb[1],
                                                     (size_t) c.head * row_size));
        extra = c.variant == 1          ? ggml_cont(ctx, rows) :
                c.variant == 2          ? ggml_cont(ctx, new_st) :
                c.variant == 3          ? ggml_cont(ctx, persist) :
                c.variant == 4          ? target :
                c.clear == clear_reader ? ggml_cont(ctx, persist) :
                                          nullptr;
        if (extra != nullptr) {
            ggml_set_output(extra);
        }

        g = ggml_new_graph(ctx);
        if (c.clear != clear_late) {
            ggml_build_forward_expand(g, clear);
        }
        if (c.clear == clear_reader) {
            ggml_build_forward_expand(g, extra);  // between the clear and the GET_ROWS
        }
        ggml_build_forward_expand(g, rows);
        ggml_build_forward_expand(g, cpy_x);
        ggml_build_forward_expand(g, scores);
        if (c.clear == clear_late) {
            ggml_build_forward_expand(g, clear);  // a write of the states between the two copies
        }
        if (c.variant == 3) {
            ggml_build_forward_expand(g, extra);  // between the op and the CPY: it must read the row as it was
        }
        ggml_build_forward_expand(g, cpy);
        if (extra != nullptr && c.variant != 3 && c.clear != clear_reader) {
            ggml_build_forward_expand(g, extra);
        }
        if (!ops_unallocated) {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }
        return pbuf != nullptr && buf != nullptr;
    }

    // the model's inputs, as check_gdn's fill_model: q and k L2-normalized per head and token, v in [-1, 1), the gate
    // (a log decay) in (-2, 0], beta in [0.01, 0.99); every row of the states in [-0.5, 0.5), so a row the CPY must
    // not touch shows if it changes
    void fill(const state_case & c, uint32_t seed) {
        for (int i = 0; i < 5; i++) {
            std::vector<float> v = random_values(ggml_nelements(in[i]), seed + (uint32_t) i, 1.0f);
            if (i <= 1) {
                const size_t d = (size_t) in[i]->ne[0];
                for (size_t r = 0; r < v.size(); r += d) {
                    double sum = 0.0;
                    for (size_t j = 0; j < d; j++) {
                        sum += (double) v[r + j] * v[r + j];
                    }
                    const float inv = sum > 0.0 ? (float) (1.0 / std::sqrt(sum)) : 1.0f;
                    for (size_t j = 0; j < d; j++) {
                        v[r + j] *= inv;
                    }
                }
            } else {
                for (float & x : v) {
                    x = i == 3 ? -(x + 1.0f) : i == 4 ? 0.01f + 0.49f * (x + 1.0f) : x;
                }
            }
            ggml_backend_tensor_set(in[i], v.data(), 0, ggml_nbytes(in[i]));
        }
        std::vector<float> s = random_values(ggml_nelements(persist), seed + 5, 1.0f);
        for (float & x : s) {
            x *= 0.5f;
        }
        if (c.wild) {  // the old state of the row the clear must handle bit for bit
            const float wild[8] = { std::nanf(""),           std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity(), -0.0f,
                                    -3.5f,                   0.0f,
                                    1e30f,                   -1e-30f };
            float * row = s.data() + c.zrow * c.S_v * c.S_v * c.H;
            for (int64_t i = 0; i < c.S_v * c.S_v * c.H; i++) {
                row[i] = wild[i % 8];
            }
        }
        ggml_backend_tensor_set(persist, s.data(), 0, ggml_nbytes(persist));
        const uint8_t * p = (const uint8_t *) s.data();
        persist_in.assign(p, p + ggml_nbytes(persist));
        std::vector<int32_t> iv((size_t) c.n_seqs);
        for (int64_t i = 0; i < c.n_seqs; i++) {
            iv[(size_t) i] = (int32_t) (c.idx + i);
        }
        ggml_backend_tensor_set(idx, iv.data(), 0, iv.size() * sizeof(int32_t));
    }

    // before a run: the states back to the case's values, the outputs to junk so a missed write shows
    void reset() const {
        ggml_backend_tensor_set(persist, persist_in.data(), 0, persist_in.size());
        for (ggml_tensor * t : { gdn, scores, extra }) {
            if (t != nullptr && t->data != nullptr) {
                const std::vector<uint8_t> j(ggml_nbytes(t), 0xA5);
                ggml_backend_tensor_set(t, j.data(), 0, j.size());
            }
        }
    }

    static std::vector<float> floats(const ggml_tensor * t) {
        std::vector<float> v(t != nullptr ? (size_t) ggml_nelements(t) : 0);
        if (!v.empty()) {
            ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        }
        return v;
    }

    // the op's attention scores, the only part of its output a state pass writes
    std::vector<float> gdn_scores(const state_case & c) const {
        const std::vector<float> all = floats(gdn);
        const size_t             n   = (size_t) (c.S_v * c.H * c.n_tokens * c.n_seqs);
        return std::vector<float>(all.begin(), all.begin() + std::min(n, all.size()));
    }
};

// the same bytes, so that a NaN counts equal to the same NaN (float == does not) and +0.0 differs from -0.0
static bool same_bytes(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

static int check_state_pass(ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr && cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 1);  // a correctness reference: single-threaded (PROJECT.md item 28)

    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    REQUIRE(table != nullptr);
    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    const spacemit_plugin_kernel * k_direct  = nullptr;
    const spacemit_plugin_kernel * k_cleared = nullptr;
    for (uint32_t k = 0; k < n_kernels; k++) {
        if (kernels[k].name == nullptr) {
            continue;
        }
        if (std::strcmp(kernels[k].name, "gated_delta_net.f32.direct") == 0) {
            k_direct = &kernels[k];
        }
        if (std::strcmp(kernels[k].name, "gated_delta_net.f32.direct.cleared") == 0) {
            k_cleared = &kernels[k];
        }
    }
    REQUIRE(k_direct != nullptr && k_direct->fn != nullptr && k_direct->workspace_size != nullptr);
    REQUIRE(k_cleared != nullptr && k_cleared->fn != nullptr && k_cleared->workspace_size != nullptr);
    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    env_override                     plan_default("FLAGOS_SPACEMIT_PLAN", nullptr);
    env_override                     quantize_default("FLAGOS_SPACEMIT_QUANTIZE", nullptr);
    // the direct calls run on the calling core: its bytes are the device's only where both run the same kernels
#if defined(__riscv)
    const bool exact_direct = *reference_label() != '\0';
#else
    const bool exact_direct = true;
#endif

    const state_case cases[] = {
  // what,                                           S_v,  H, tok, seq, K, slots, idx, head, variant, pass
        { "1 token",                                   16,  4,   1,   1, 1,     4,   2,    2,       0, true  },
        { "1 token, the row read not the row written", 16,  4,   1,   1, 1,     4,   1,    3,       0, true  },
        { "2 tokens",                                  16,  4,   2,   1, 1,     4,   0,    0,       0, true  },
        { "33 tokens",                                 16,  4,  33,   1, 1,     3,   1,    1,       0, true  },
        { "501 tokens",                                16,  4, 501,   1, 1,     3,   2,    0,       0, true  },
        { "Qwen3.5-4B: S_v 128, 16 heads, 33 tokens", 128, 16,  33,   1, 1,     3,   1,    1,       0, true  },
        { "S_v 128, 16 heads, 1 token",               128, 16,   1,   1, 1,     3,   0,    2,       0, true  },
        { "a second consumer of the row read in",      16,  4,   2,   1, 1,     4,   1,    1,       1, false },
        { "another view of the new state",             16,  4,   2,   1, 1,     4,   1,    1,       2, false },
        { "2 sequences",                               16,  4,   2,   2, 1,     4,   0,    2,       0, false },
        { "K 2",                                       16,  4,   2,   1, 2,     4,   1,    1,       0, false },
        { "a read of the states between the copies",   16,  4,   2,   1, 1,     4,   1,    1,       3, false },
        { "the CPY into another tensor",               16,  4,   2,   1, 1,     4,   1,    1,       4, false },
  // a sequence's first graph: the clear folded into the step (zrow the row the CPY writes)
  // what,                                            S_v,  H, tok, seq, K, slots, idx, head, variant, pass, zrow, clear, cleared, wild
        { "a sequence's start, 1 token",               16,  4,   1,   1, 1,     4,   2,    2,       0, true,    2, clear_whole,  true  },
        { "a sequence's start, 2 tokens",              16,  4,   2,   1, 1,     4,   1,    1,       0, true,    1, clear_whole,  true  },
        { "a sequence's start, 32 tokens",             16,  4,  32,   1, 1,     4,   0,    0,       0, true,    0, clear_whole,  true  },
        { "a sequence's start, 501 tokens",            16,  4, 501,   1, 1,     4,   3,    3,       0, true,    3, clear_whole,  true  },
        { "a sequence's start, 3 rows, the cleared row read",
                                                       16,  4,   2,   1, 1,     3,   1,    1,       0, true,    1, clear_whole,  true  },
        { "a sequence's start, 3 rows, another row read",
                                                       16,  4,   2,   1, 1,     3,   0,    2,       0, true,    2, clear_whole,  true  },
        { "a sequence's start, the old state wild",    16,  4,   2,   1, 1,     3,   1,    1,       0, true,    1, clear_whole,  true, true },
        { "Qwen3.6-35B: S_v 128, 32 heads, 1 token",  128, 32,   1,   1, 1,     3,   1,    1,       0, true,    1, clear_whole,  true  },
        { "Qwen3.6-35B: S_v 128, 32 heads, 32 tokens",
                                                      128, 32,  32,   1, 1,     3,   1,    1,       0, true,    1, clear_whole,  true  },
  // the pass without the fold: the clear keeps its own step
        { "a clear by 0.5",                            16,  4,   2,   1, 1,     4,   1,    1,       0, true,    1, clear_factor, false },
        { "a clear over half a row",                   16,  4,   2,   1, 1,     4,   1,    1,       0, true,    1, clear_half,   false },
        { "a clear over two rows",                     16,  4,   2,   1, 1,     4,   1,    1,       0, true,    1, clear_two,    false },
        { "the cleared row not the row written",       16,  4,   2,   1, 1,     4,   1,    1,       0, true,    2, clear_whole,  false },
        { "a read of the states between the clear and the GET_ROWS",
                                                       16,  4,   2,   1, 1,     4,   1,    1,       0, true,    1, clear_reader, false },
  // refused: a write of the states between the two copies
        { "a write of the states between the copies",  16,  4,   2,   1, 1,     4,   1,    1,       0, false,   2, clear_late,   false },
    };
    int      n_cases = 0, n_ok = 0, n_passed = 0, n_direct = 0, n_direct_ok = 0;
    double   worst_nmse = 0.0;
    uint32_t seed       = 1500;
    for (const state_case & c : cases) {
        state_graph dv, ref;
        if (!dv.init(c, buft, buft) || !ref.init(c, ggml_backend_cpu_buffer_type(), ggml_backend_cpu_buffer_type())) {
            std::fprintf(stderr, "FAIL: state pass %s: allocation failed\n", c.what);
            g_failures++;
            seed += 8;
            continue;
        }
        dv.fill(c, seed);
        ref.fill(c, seed);
        const bool               ref_ok    = ggml_backend_graph_compute(cpu, ref.g) == GGML_STATUS_SUCCESS;
        const std::vector<float> want      = state_graph::floats(ref.scores);
        const std::vector<float> want_st   = state_graph::floats(ref.persist);
        const std::vector<float> want_more = state_graph::floats(ref.extra);

        // one device run: the passes it took and its outputs
        const auto run = [&](uint64_t & passes, std::vector<float> & out, std::vector<float> & st,
                             std::vector<float> & more) {
            dv.reset();
            spacemit_state_passes_take();  // drops earlier graphs'
            const bool ok = ggml_backend_graph_compute(backend, dv.g) == GGML_STATUS_SUCCESS;
            passes        = spacemit_state_passes_take();
            out           = state_graph::floats(dv.scores);
            st            = state_graph::floats(dv.persist);
            more          = state_graph::floats(dv.extra);
            return ok;
        };
        {
            n_cases++;
            uint64_t           passes = 0;
            std::vector<float> out, st, more;
            const bool         dev_ok = ref_ok && run(passes, out, st, more);
            std::vector<float> direct_out, direct_st;
            bool               direct_ok = true;
            // the scores and the state against ggml-cpu, and the rows the CPY does not write bit for bit the inputs
            const double e_out = dev_ok && out.size() == want.size() ? nmse(out.data(), want.data(), (int64_t) out.size()) : -1.0;
            const double e_st  = dev_ok && st.size() == want_st.size() ? nmse(st.data(), want_st.data(), (int64_t) st.size()) : -1.0;
            const double e_more = dev_ok && more.size() == want_more.size() && !more.empty() ?
                                      nmse(more.data(), want_more.data(), (int64_t) more.size()) : 0.0;
            // with a wild old state every value the kernels compute is a NaN (a NaN or an infinity times 0 is one), so
            // there the bytes must be the CPU's and NMSE says nothing (nmse answers HUGE_VAL on a NaN)
            const bool wild_ok = !c.wild || (dev_ok && same_bytes(out, want) && same_bytes(st, want_st) &&
                                             same_bytes(more, want_more));
            const bool close   = c.wild || (e_out >= 0.0 && e_out <= k_gdn_max_nmse && e_st >= 0.0 &&
                                          e_st <= k_gdn_max_nmse && e_more <= k_gdn_max_nmse);
            bool         kept  = dev_ok, clear_ok = dev_ok;
            {
                const size_t row = (size_t) (c.S_v * c.S_v * c.H) * sizeof(float);
                const size_t lo  = c.variant == 4 ? 0 : (size_t) c.head * row;
                const size_t hi  = c.variant == 4 ? 0 : lo + (size_t) c.n_seqs * row;
                // what build_rs' clear writes: inside the row the CPY writes, or rows of its own, which must then
                // hold the CPU's bytes (x * 0 and x * 0.5 are exact on both sides, -0.0 and NaN among them)
                const size_t zlo = c.zrow < 0 ? 0 : (size_t) c.zrow * row;
                const size_t zhi = c.zrow < 0 ? 0 :
                                   zlo + (c.clear == clear_half ? row / 2 : c.clear == clear_two ? 2 * row : row);
                const uint8_t * got_bytes  = (const uint8_t *) st.data();
                const uint8_t * want_bytes = (const uint8_t *) want_st.data();
                const bool      sized      = st.size() * sizeof(float) == dv.persist_in.size() &&
                                        want_st.size() == st.size();
                kept = kept && sized;
                clear_ok = clear_ok && sized;
                for (size_t i = 0; kept && i < dv.persist_in.size(); i++) {
                    kept = (i >= lo && i < hi) || (i >= zlo && i < zhi) || got_bytes[i] == dv.persist_in[i];
                }
                for (size_t i = zlo; clear_ok && i < zhi; i++) {
                    clear_ok = (i >= lo && i < hi) || got_bytes[i] == want_bytes[i];
                }
            }
            // the step called directly on 1, 3 and 8 tiles, on a CPU-buffer copy: the folding step where the fold
            // applies, which clears the row itself, so its result is the CPU's whole graph. Where the clear keeps its
            // own step it is left out: the step alone does not do that node's work
            if (dev_ok && c.pass && (c.zrow < 0 || c.cleared)) {
                const spacemit_plugin_kernel * entry = c.cleared ? k_cleared : k_direct;
                state_graph                    dir;
                direct_ok = dir.init(c, ggml_backend_cpu_buffer_type(), ggml_backend_cpu_buffer_type());
                if (direct_ok) {
                    dir.fill(c, seed);
                    const size_t       ws = entry->workspace_size(dir.gdn);
                    std::vector<float> workspace(ws / sizeof(float) + 1, NAN);
                    for (uint32_t nth : { 1u, 3u, 8u }) {
                        n_direct++;
                        dir.reset();
                        bool ok = true;
                        for (uint32_t ith = 0; ith < nth; ith++) {
                            const spacemit_tile tile = { ith, nth, nullptr, 0, workspace.data(), ws, nullptr };
                            ok &= entry->fn(&tile, dir.cpy);
                        }
                        const std::vector<float> got    = dir.gdn_scores(c);
                        const std::vector<float> got_st = state_graph::floats(dir.persist);
                        const double             e      = ok && got.size() == want.size() ?
                                                              nmse(got.data(), want.data(), (int64_t) got.size()) : -1.0;
                        const double e_s = ok && got_st.size() == want_st.size() ?
                                               nmse(got_st.data(), want_st.data(), (int64_t) got_st.size()) : -1.0;
                        if (nth == 1u) {
                            direct_out = got;
                            direct_st  = got_st;
                        }
                        const bool tiles = same_bytes(got, direct_out) && same_bytes(got_st, direct_st);
                        const bool near  = c.wild ? same_bytes(got, want) && same_bytes(got_st, want_st) :
                                                   e >= 0.0 && e <= k_gdn_max_nmse && e_s >= 0.0 &&
                                                       e_s <= k_gdn_max_nmse;
                        const bool same  = ok && tiles && near &&
                                          (!exact_direct || (same_bytes(got, dv.gdn_scores(c)) && same_bytes(got_st, st)));
                        if (!c.wild) {
                            worst_nmse = std::max(worst_nmse, std::max(e, e_s));
                        }
                        n_direct_ok += same;
                        direct_ok = direct_ok && same;
                        if (!same) {
                            std::fprintf(stderr,
                                         "FAIL: state pass %s, the step called directly on %u tiles: %s, NMSE scores "
                                         "%.3e state %.3e (bound %.0e), the tile counts %s%s\n",
                                         c.what, nth, ok ? "ran" : "failed", e, e_s, k_gdn_max_nmse,
                                         tiles ? "equal" : "differ",
                                         exact_direct ? ", against the device's bytes checked" : "");
                            g_failures++;
                        }
                    }
                }
            }
            // the step names (GGML_FLAGOS_TRACE_LAUNCH): the folding step where the fold applies and no scale.f32 for
            // the clearing node, the plain step and that node's own step otherwise
            bool names_ok = true;
            int  n_names[3] = {};
            if (dev_ok && c.zrow >= 0) {
                env_override trace("GGML_FLAGOS_TRACE_LAUNCH", "1");
                log_capture  log;
                dv.reset();
                const bool        ran  = ggml_backend_graph_compute(backend, dv.g) == GGML_STATUS_SUCCESS;
                const std::string text = log.take();
                n_names[0]             = count_lines_with(text, " gated_delta_net.f32.direct.cleared ");
                n_names[1]             = count_lines_with(text, " gated_delta_net.f32.direct ");
                n_names[2]             = count_lines_with(text, " scale.f32 ");
                names_ok = ran && n_names[0] == (c.cleared ? 1 : 0) &&
                           n_names[1] == (c.pass && !c.cleared ? 1 : 0) && n_names[2] == (c.cleared ? 0 : 1);
            }
            if (!c.wild) {
                worst_nmse = std::max(worst_nmse, std::max(e_out, e_st));
            }
            const bool ok = dev_ok && names_ok && kept && clear_ok && direct_ok &&
                            passes == (c.pass ? 1u : 0u) && close && wild_ok;
            n_ok += ok;
            n_passed += passes == 1;
            if (!ok) {
                std::fprintf(stderr,
                             "FAIL: state pass %s (S_v %lld, %lld heads, %lld tokens, %lld sequences, K %lld, row "
                             "%lld to row %lld of %lld, variant %d, clear %d of row %lld): the CPU reference %s, the "
                             "run %s, %llu passes (expected "
                             "%d); NMSE scores %.3e, states %.3e, other output %.3e (bound %.0e), the bytes with a "
                             "wild old state %s; the rows the CPY does not write %s, the cleared rows %s; the direct "
                             "calls %s; the step names %d cleared, %d direct, %d scale (expected %d/%d/%d)\n",
                             c.what, (long long) c.S_v, (long long) c.H, (long long) c.n_tokens, (long long) c.n_seqs,
                             (long long) c.K, (long long) c.idx, (long long) c.head, (long long) c.slots, c.variant,
                             c.clear, (long long) c.zrow, ref_ok ? "ran" : "failed", dev_ok ? "ran" : "failed",
                             (unsigned long long) passes, c.pass ? 1 : 0, e_out, e_st, e_more, k_gdn_max_nmse,
                             wild_ok ? "as the CPU's" : "other", kept ? "unchanged" : "changed",
                             clear_ok ? "as the CPU's" : "other", direct_ok ? "as expected" : "failed",
                             n_names[0], n_names[1], n_names[2], c.cleared ? 1 : 0, c.pass && !c.cleared ? 1 : 0,
                             c.cleared ? 0 : 1);
                g_failures++;
            }
        }
        seed += 8;
    }

    // two computes in a row on the same states, the second starting from the first's state, against the CPU
    bool carried_ok = false;
    {
        const state_case c = { "carried over", 16, 4, 2, 1, 1, 4, 1, 1, 0, true };
        state_graph      dv, ref;
        if (dv.init(c, buft, buft) && ref.init(c, ggml_backend_cpu_buffer_type(), ggml_backend_cpu_buffer_type())) {
            dv.fill(c, seed);
            ref.fill(c, seed);
            bool     ok     = true;
            uint64_t passes = 0;
            spacemit_state_passes_take();
            for (int r = 0; r < 2; r++) {
                ok = ok && ggml_backend_graph_compute(cpu, ref.g) == GGML_STATUS_SUCCESS &&
                     ggml_backend_graph_compute(backend, dv.g) == GGML_STATUS_SUCCESS;
            }
            passes                             = spacemit_state_passes_take();
            const std::vector<float> got       = state_graph::floats(dv.scores);
            const std::vector<float> want      = state_graph::floats(ref.scores);
            const std::vector<float> got_st    = state_graph::floats(dv.persist);
            const std::vector<float> want_st   = state_graph::floats(ref.persist);
            const double             e         = ok && got.size() == want.size() ?
                                                     nmse(got.data(), want.data(), (int64_t) got.size()) : -1.0;
            const double             e_st      = ok && got_st.size() == want_st.size() ?
                                                     nmse(got_st.data(), want_st.data(), (int64_t) got_st.size()) : -1.0;
            worst_nmse = std::max(worst_nmse, std::max(e, e_st));
            carried_ok = ok && passes == 2 && e >= 0.0 && e <= k_gdn_max_nmse && e_st >= 0.0 && e_st <= k_gdn_max_nmse;
            if (!carried_ok) {
                std::fprintf(stderr,
                             "FAIL: state pass, two computes with the state carried over: %s, %llu passes (expected "
                             "2); NMSE scores %.3e, states %.3e (bound %.0e)\n",
                             ok ? "ran" : "failed", (unsigned long long) passes, e, e_st, k_gdn_max_nmse);
                g_failures++;
            }
        } else {
            std::fprintf(stderr, "FAIL: state pass, two computes with the state carried over: allocation failed\n");
            g_failures++;
        }
        seed += 8;
    }

    // through the scheduler (the inputs and the states in our buffer, the rest in its compute buffer) under the plan
    // cache's uid trust
    bool     sched_ok = false;
    uint64_t sched_passes[2] = {}, sched_hits[2] = {}, sched_misses[2] = {};
    {
        const state_case c = { "the scheduler", 16, 4, 2, 1, 1, 4, 1, 1, 0, true };
        state_graph      sg, ref;
        if (sg.init(c, buft, buft, true) &&
            ref.init(c, ggml_backend_cpu_buffer_type(), ggml_backend_cpu_buffer_type())) {
            sg.fill(c, seed);
            ref.fill(c, seed);
            ggml_backend_t       both[2] = { backend, cpu };
            ggml_backend_sched_t sched   = ggml_backend_sched_new(both, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
            bool                 ok      = sched != nullptr && ggml_backend_graph_compute(cpu, ref.g) == GGML_STATUS_SUCCESS;
            const std::vector<float> want    = state_graph::floats(ref.scores);
            const std::vector<float> want_st = state_graph::floats(ref.persist);
            std::vector<float>       first, first_st;
            for (int i = 0; i < 2 && ok; i++) {
                sg.reset();
                spacemit_state_passes_take();
                spacemit_plan_counters_take();
                ok = ggml_backend_sched_graph_compute(sched, sg.g) == GGML_STATUS_SUCCESS;
                const spacemit_plan_counters pc = spacemit_plan_counters_take();
                sched_passes[i]                 = spacemit_state_passes_take();
                sched_hits[i]                   = pc.hits_uid;
                sched_misses[i]                 = pc.misses;
                const std::vector<float> out    = ok ? state_graph::floats(sg.scores) : std::vector<float>();
                const std::vector<float> out_st = ok ? state_graph::floats(sg.persist) : std::vector<float>();
                if (i == 0) {
                    first    = out;
                    first_st = out_st;
                }
                ok = ok && same_bytes(out, first) && same_bytes(out_st, first_st);
            }
            const bool on_device = sched != nullptr && ggml_backend_sched_get_tensor_backend(sched, sg.gdn) == backend &&
                                   ggml_backend_sched_get_tensor_backend(sched, sg.cpy) == backend;
            const double e    = ok && first.size() == want.size() ?
                                    nmse(first.data(), want.data(), (int64_t) first.size()) : -1.0;
            const double e_st = ok && first_st.size() == want_st.size() ?
                                    nmse(first_st.data(), want_st.data(), (int64_t) first_st.size()) : -1.0;
            worst_nmse = std::max(worst_nmse, std::max(e, e_st));
            sched_ok   = ok && on_device && e >= 0.0 && e <= k_gdn_max_nmse && e_st >= 0.0 && e_st <= k_gdn_max_nmse &&
                       sched_passes[0] == 1 && sched_misses[0] == 1 && sched_hits[1] == 1 && sched_passes[1] == 0;
            if (sched != nullptr) {
                ggml_backend_sched_free(sched);
            }
        }
        if (!sched_ok) {
            std::fprintf(stderr,
                         "FAIL: state pass through the scheduler: passes %llu/%llu (expected 1/0), plan hits %llu/%llu "
                         "(expected 0/1), misses %llu/%llu (expected 1/0), or the outputs, the placement or the bound "
                         "failed\n",
                         (unsigned long long) sched_passes[0], (unsigned long long) sched_passes[1],
                         (unsigned long long) sched_hits[0], (unsigned long long) sched_hits[1],
                         (unsigned long long) sched_misses[0], (unsigned long long) sched_misses[1]);
            g_failures++;
        }
    }

    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    std::printf("statepass GATED_DELTA_NET + GET_ROWS + CPY as the model builds them (1, 2, 33 and 501 tokens of S_v "
                "16, S_v 128 with 16 heads at 1 and 33 tokens, the row read in the row written or not; refused: a "
                "second consumer of the row read in, another view of the new state, 2 sequences, K 2, a read of the "
                "states between the copies, the CPY into another tensor), and a sequence's first graph with build_rs' "
                "clear of a whole row (1, 2, 32 and 501 tokens of S_v 16, Qwen3.6-35B's S_v 128 with 32 heads at 1 "
                "and 32 tokens, 3 rows with the cleared row read or not, a wild old state) folding it into the step, "
                "without the fold (a factor 0.5, half a row, two rows, the cleared row not the written one, a read of "
                "the states before the GET_ROWS) and refused (a write of the states between the copies)%s: %d/%d "
                "cases, %d passes by the counter; against the CPU on one thread max NMSE %.2e (bound %.0e), the rows "
                "the CPY does not write bit for bit and the cleared rows the CPU's bytes, the step names the fold's "
                "where it applies and no scale.f32 there; the step directly on 1, 3 and 8 tiles %d/%d; two computes "
                "with the state carried over: %s; through the scheduler a miss, a hit by uid (no pass), the same "
                "bytes: %s\n",
                reference_label(), n_ok, n_cases, n_passed, worst_nmse, k_gdn_max_nmse, n_direct_ok, n_direct,
                carried_ok ? "checked" : "FAILED", sched_ok ? "checked" : "FAILED");
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
    if (rope_imrope_check(dev, backend, cpu) != 0) {
        return 1;
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
// uid-trust with one warning. Every output against the CPU, the layers'
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
            const bool ok = d.hits_uid == uid_hits && d.misses == misses;
            if (!ok) {
                std::fprintf(stderr, "FAIL: plan %s, probe %s: hits by uid %llu, misses %llu; expected %llu, %llu\n",
                             setting, probe, (unsigned long long) d.hits_uid, (unsigned long long) d.misses,
                             (unsigned long long) uid_hits, (unsigned long long) misses);
                g_failures++;
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
            const bool zero = total.hits_uid == 0 && total.misses == 0;
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
                    "cache under QUANTIZE=shared%s, n the default uid-trust); %d graphs: %llu hits by uid, %llu "
                    "misses; %s; layer NMSE %.2e (bound %.0e), attention NMSE %.2e (bound %.0e)\n",
                    setting, n_passed, n_probes, on ? " after the entry and the key alone (seven cases)" : "",
                    on ? "" : ", i no counter moved", n_graphs, (unsigned long long) total.hits_uid,
                    (unsigned long long) total.misses,
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

// the number after "key": in a JSON line, NAN when absent
static double json_number(const std::string & line, const char * key) {
    const std::string k  = std::string("\"") + key + "\":";
    const size_t      at = line.find(k);
    return at == std::string::npos ? NAN : std::strtod(line.c_str() + at + k.size(), nullptr);
}

// GGML_FLAGOS_OP_STATS keeps a kind's first graph apart: one kind seen once (rows 7) and one seen three times (rows 9),
// two steps on two tiles, each graph's clock reads made up. The counted lines of rows 9 hold graphs 2 and 3 only, the
// "first":1 lines graph 1 only; rows 7 has all-zero counted lines (graphs 0) and first lines with its one graph
static bool check_op_stats_first() {
    const std::string path = std::string(FLAGOS_SPACEMIT_TEST_PLUGIN_DIR) + "/diag-op-stats.jsonl";
    if (FILE * f = std::fopen(path.c_str(), "w")) {  // the destructor appends
        std::fclose(f);
    }
    ggml_init_params params = { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "FAIL: diagnostics, OP_STATS: no context\n");
        g_failures++;
        return false;
    }
    ggml_tensor *                           a     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor *                           x     = ggml_scale(ctx, a, 2.0f);
    ggml_tensor *                           y     = ggml_add(ctx, x, a);
    const std::vector<spacemit_plugin_step> steps = { { plan_fake_kernel_1, x, 0 }, { plan_fake_kernel_2, y, 0 } };
    const std::vector<const char *>         names = { "scale", "add" };
    const size_t                            n     = steps.size();
    const uint32_t                          tiles = 2;
    // graph g's (0-based) clock reads
    const auto kernel = [](int g, size_t s, uint32_t t) -> uint32_t {
        return static_cast<uint32_t>(1000 * (g + 1) * (s + 1) + 500 * t);
    };
    const auto barrier = [](int g) -> uint32_t { return static_cast<uint32_t>(200 * (g + 1)); };
    const auto launch  = [](int g) { return 10.0 * (g + 1); };
    {
        spacemit_op_stats_file counters(path.c_str());
        spacemit_run_stats     run;
        const auto             graph = [&](int64_t rows, int g) {
            run.reset(n);
            run.n_steps   = n;
            run.n_tiles   = tiles;
            run.launch_us = launch(g);
            for (uint32_t t = 0; t < tiles; t++) {
                for (size_t s = 0; s < n; s++) {
                    run.kernel_ns[t * run.stride + s]       = kernel(g, s, t);
                    run.step_barrier_ns[t * run.stride + s] = barrier(g);
                }
            }
            counters.add("per-split", rows, 512, steps, names, run);
        };
        graph(7, 0);
        for (int g = 0; g < 3; g++) {
            graph(9, g);
        }
    }  // the destructor writes
    ggml_free(ctx);

    // the (all) line's expected values per graph over graphs [g0, g1)
    struct want {
        double wall, kernel, busy, launch;
    };
    const auto expect = [&](int g0, int g1) {
        want w = {};
        for (int g = g0; g < g1; g++) {
            for (size_t s = 0; s < n; s++) {
                uint32_t slowest = 0;
                for (uint32_t t = 0; t < tiles; t++) {
                    w.kernel += kernel(g, s, t);
                    w.wall += kernel(g, s, t) + barrier(g);
                    slowest = std::max(slowest, kernel(g, s, t));
                }
                w.busy += slowest;
            }
            w.launch += launch(g);
        }
        const double per = 1e3 * std::max(g1 - g0, 1);
        return want{ w.wall / tiles / per, w.kernel / tiles / per, w.busy / per, 1e3 * w.launch / per };
    };
    // rows -> first -> the (all) line; and the lines per (rows, first)
    std::map<std::pair<int, bool>, std::string> all;
    std::map<std::pair<int, bool>, int>         lines;
    bool                                        format = true;
    std::ifstream                               written(path);
    std::string                                 line;
    while (std::getline(written, line)) {
        if (line.empty() || line[0] != '{') {
            continue;
        }
        const bool first = line.find("\"first\":1") != std::string::npos;
        const int  rows  = static_cast<int>(json_number(line, "rows"));
        lines[{ rows, first }]++;
        const int graphs = static_cast<int>(json_number(line, "graphs"));
        char      head[64];  // the fields around "first" in their order
        std::snprintf(head, sizeof(head), "\"graphs\":%d,%s\"label\":", graphs, first ? "\"first\":1," : "");
        format = format && line.find(head) != std::string::npos;
        if (line.find("\"label\":\"(all)\"") != std::string::npos) {
            all[{ rows, first }] = line;
        }
    }
    const auto matches = [&](const std::string & l, int graphs, const want & w) {
        const auto near = [](double got, double v) { return std::fabs(got - v) <= 1e-3 * std::max(1.0, std::fabs(v)); };
        return static_cast<int>(json_number(l, "graphs")) == graphs && near(json_number(l, "wall_us"), w.wall) &&
               near(json_number(l, "kernel_us"), w.kernel) && near(json_number(l, "busy_us"), w.busy) &&
               near(json_number(l, "launch_us"), w.launch);
    };
    const want zero = {};
    const bool once = all.count({ 7, false }) && all.count({ 7, true }) && matches(all[{ 7, false }], 0, zero) &&
                      matches(all[{ 7, true }], 1, expect(0, 1)) && json_number(all[{ 7, true }], "wall_us") > 0.0;
    const bool thrice = all.count({ 9, false }) && all.count({ 9, true }) &&
                        matches(all[{ 9, false }], 2, expect(1, 3)) && matches(all[{ 9, true }], 1, expect(0, 1));
    // two names and (all) per set
    const bool count = lines.size() == 4 && lines[{ 7, false }] == 3 && lines[{ 7, true }] == 3 &&
                       lines[{ 9, false }] == 3 && lines[{ 9, true }] == 3;
    const bool ok = once && thrice && count && format;
    if (!ok) {
        std::fprintf(stderr,
                     "FAIL: diagnostics, OP_STATS: the first graph kept apart: a kind seen once %s, seen three times "
                     "%s, lines %s, fields %s (%s)\n",
                     once ? "ok" : "wrong", thrice ? "ok" : "wrong", count ? "ok" : "wrong", format ? "ok" : "wrong",
                     path.c_str());
        g_failures++;
    }
    return ok;
}

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

    const bool op_stats = check_op_stats_first();

    std::printf("diag     DISABLE_OPS (%d/%zu settings: GLU and FLASH_ATTN_EXT refused by find_op and supports_op, an "
                "unknown name reported once and ignored; a layer's SwiGLU on the CPU side of the scheduler, %d splits "
                "against %d, NMSE %.2e, bound %.0e), TRACE_LAUNCH (%d/2: %d step lines for %llu steps and the two "
                "launch lines, with one launch per step %d status lines for %llu; %d/2 forced failures reported, the "
                "failing step named %d/1), STATS=- one JSON line to stderr (%s), OP_STATS the first graph kept "
                "apart (%s): checked\n",
                disable_ok, sizeof(cases) / sizeof(cases[0]), splits[1], splits[0], e_sched, k_layer_max_nmse,
                trace_ok, step_lines, (unsigned long long) n_steps, status_lines, (unsigned long long) n_steps_sync,
                fail_ok, named_step, stats_stderr ? "yes" : "no", op_stats ? "yes" : "no");
    return 0;
}

// how many of rounds runs of l on backend failed or differ from ref; it holds no check, so a thread may call it
static int bad_runs(plan_layer & l, ggml_backend_t backend, const std::vector<float> & ref, int rounds) {
    int                bad = 0;
    std::vector<float> out;
    for (int r = 0; r < rounds; r++) {
        bad += !(l.run(backend, out) && same_bytes(out, ref));
    }
    return bad;
}

// plan.md "S2 spec", step 2: a device's backend contexts share one executor, as spacemit_executors_take counts, and two
// graphs on two contexts give the bytes of each alone: interleaved, with the contexts freed in any order, with one
// context failing, and from two threads, the core's run_mutex keeping the runs one at a time. Called last, so that a
// backend an earlier check leaked shows as a count of 0 at the first backend made here
static int check_shared_stream(ggml_backend_dev_t dev, bool host) {
    plan_layer a, b, c;  // c: only for the failing graph, whose outputs are left unfinished
    REQUIRE(a.init(dev, 1, 1e6f));
    REQUIRE(b.init(dev, 7, 1e4f));
    REQUIRE(c.init(dev, 1, 1e6f));
    constexpr int k_rounds = 500;  // of the two-thread case: 1 to 2 s on a Mac
    int           n_same = 0, n_runs = 0;
    for (const char * threads : { (const char *) nullptr, "1" }) {
        if (threads != nullptr && !host) {
            break;
        }
        env_override tcm("FLAGOS_SPACEMIT_TEST_TCM", "393216");
        env_override on_threads("FLAGOS_SPACEMIT_TEST_THREADS", threads);
        const char * how = threads != nullptr ? " (tiles on threads)" : "";

        const auto count = [&](const char * what, uint64_t want) {
            const uint64_t got = g_executors_take();
            if (got != want) {
                std::fprintf(stderr, "FAIL: shared stream, %s%s: %llu executors created, expected %llu\n", what, how,
                             (unsigned long long) got, (unsigned long long) want);
                g_failures++;
            }
        };
        const auto same = [&](const char * what, plan_layer & l, ggml_backend_t backend, const std::vector<float> & ref) {
            const int bad = bad_runs(l, backend, ref, 1);
            if (bad != 0) {
                std::fprintf(stderr, "FAIL: shared stream, %s%s: a run failed or differs from its run alone\n", what, how);
                g_failures++;
            }
            n_same += 1 - bad;
            n_runs++;
        };
        const auto bad_in_thread = [&](const char * what, int bad) {
            if (bad != 0) {
                std::fprintf(stderr, "FAIL: shared stream, %s%s: %d runs failed or differ from their run alone\n", what,
                             how, bad);
                g_failures++;
            }
        };
        const auto finite = [](const std::vector<float> & v) {
            return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
        };
        g_executors_take();

        // each graph alone: the references
        std::vector<float> ref_a, ref_b;
        {
            ggml_backend_t x = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(x != nullptr);
            count("the first backend, alone", 1);
            REQUIRE(a.run(x, ref_a) && finite(ref_a));
            ggml_backend_free(x);
            ggml_backend_t y = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(y != nullptr);
            count("the second backend, alone", 1);
            REQUIRE(b.run(y, ref_b) && finite(ref_b));
            ggml_backend_free(y);
        }

        // two live, then their lifetimes
        {
            ggml_backend_t ctx_a = ggml_backend_dev_init(dev, nullptr);
            ggml_backend_t ctx_b = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(ctx_a != nullptr && ctx_b != nullptr);
            count("two live backends", 1);
            for (int r = 0; r < 20; r++) {
                same("two live, interleaved", a, ctx_a, ref_a);
                same("two live, interleaved", b, ctx_b, ref_b);
            }
            for (int r = 0; r < 5; r++) {
                same("two live, swapped", b, ctx_a, ref_b);
                same("two live, swapped", a, ctx_b, ref_a);
            }

            ggml_backend_free(ctx_a);  // the first made
            for (int r = 0; r < 5; r++) {
                same("the first freed", b, ctx_b, ref_b);
            }
            ggml_backend_t ctx_c = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(ctx_c != nullptr);
            count("a backend made while another lives", 0);
            same("a backend made while another lives", a, ctx_c, ref_a);
            ggml_backend_free(ctx_c);  // the newest
            same("the newest freed", b, ctx_b, ref_b);
            ggml_backend_free(ctx_b);
            ggml_backend_t ctx_d = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(ctx_d != nullptr);
            count("a backend after the last was freed", 1);
            same("a backend after the last was freed", a, ctx_d, ref_a);
            ggml_backend_free(ctx_d);
        }

        // one context's failure stays its own
        {
            ggml_backend_t ctx_a = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(ctx_a != nullptr);
            count("a backend beside a failing one", 1);
            set_env("FLAGOS_SPACEMIT_TEST_FAIL_NODE", "0");
            ggml_backend_t failing = ggml_backend_dev_init(dev, nullptr);
            set_env("FLAGOS_SPACEMIT_TEST_FAIL_NODE", nullptr);
            REQUIRE(failing != nullptr);
            count("the failing backend", 0);
            int n_failed = 0;
            for (int r = 0; r < 10; r++) {
                n_failed += ggml_backend_graph_compute(failing, c.gd) == GGML_STATUS_FAILED;
                same("beside a failing backend", a, ctx_a, ref_a);
            }
            if (n_failed != 10) {
                std::fprintf(stderr, "FAIL: shared stream, a forced failure%s: %d of 10 runs failed\n", how, n_failed);
                g_failures++;
            }
            ggml_backend_free(failing);
            same("after the failing backend was freed", a, ctx_a, ref_a);
            ggml_backend_t ctx_g = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(ctx_g != nullptr);
            count("a backend after the failure", 0);
            same("a backend after the failure", b, ctx_g, ref_b);
            ggml_backend_free(ctx_g);
            ggml_backend_free(ctx_a);
        }

        // one thread per context
        {
            ggml_backend_t ctx_a = ggml_backend_dev_init(dev, nullptr);
            ggml_backend_t ctx_b = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(ctx_a != nullptr && ctx_b != nullptr);
            count("two backends on two threads", 1);
            int         bad_a = 0, bad_b = 0;
            std::thread on_a([&] { bad_a = bad_runs(a, ctx_a, ref_a, k_rounds); });
            std::thread on_b([&] { bad_b = bad_runs(b, ctx_b, ref_b, k_rounds); });
            on_a.join();
            on_b.join();
            bad_in_thread("two threads, the first", bad_a);
            bad_in_thread("two threads, the second", bad_b);
            n_same += 2 * k_rounds - bad_a - bad_b;
            n_runs += 2 * k_rounds;
            ggml_backend_free(ctx_a);
            ggml_backend_free(ctx_b);
        }

        // a context made and freed while another runs; that thread's rounds vary, so they are not in the line's count
        {
            ggml_backend_t ctx_a = ggml_backend_dev_init(dev, nullptr);
            REQUIRE(ctx_a != nullptr);
            count("a backend running on a thread", 1);
            std::atomic<bool> stop{ false };
            int               bad_a = 0;
            std::thread       on_a([&] {
                do {
                    bad_a += bad_runs(a, ctx_a, ref_a, 1);
                } while (!stop.load());
            });
            int n_made = 0;
            for (int cycle = 0; cycle < 20; cycle++) {
                ggml_backend_t ctx_b = ggml_backend_dev_init(dev, nullptr);
                if (ctx_b != nullptr) {  // no REQUIRE: the thread must be joined
                    n_made++;
                    same("made and freed while another runs", b, ctx_b, ref_b);
                    ggml_backend_free(ctx_b);
                }
            }
            stop.store(true);
            on_a.join();
            count("backends made and freed while another runs", 0);
            CHECK(n_made == 20);
            bad_in_thread("the backend running on a thread", bad_a);
            ggml_backend_free(ctx_a);
        }
    }
    std::printf("stream   one executor per device (backends alone, two live interleaved and swapped, freed in any order, "
                "the last free releasing it, a forced failure on one, one thread per backend, one made and freed while "
                "another runs)%s: %d/%d runs identical to their run alone%s\n",
                host ? ", tiles in turn and on threads" : "", n_same, n_runs, reference_label());
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
    env_override   dense("FLAGOS_SPACEMIT_DENSE_Q4_0", "int8");  // the int8 paths: no f16 steps at 32 rows and up
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

// the tile count the device reports, for the per-tile rates below
static uint32_t device_tiles(ggml_backend_dev_t dev) {
    const spacemit_plugin_v1 * table = spacemit_plugin_table(spacemit_device_plugin(dev));
    spacemit_plugin_device *   pdev  = nullptr;
    uint32_t                   tiles = 1;
    if (table != nullptr && table->device_open(0, &pdev) == SPACEMIT_STATUS_SUCCESS && pdev != nullptr) {
        spacemit_plugin_device_props props = {};
        props.struct_size                  = sizeof(props);
        if (table->device_props(pdev, &props) == SPACEMIT_STATUS_SUCCESS && props.n_tiles > 0) {
            tiles = props.n_tiles;
        }
        table->device_close(pdev);
    }
    return tiles;
}

// plan.md "The float matmul: all tiles for one token, blocks for a prompt", --bench-f32 and --bench: one MUL_MAT node
// of an F32 weight through the backend, at the router's [2048, 256] and the Gated DeltaNet's [2048, 32], two turns,
// the faster kept. As in the model, where a token reads each weight once, every call reads its weight from DRAM: it
// takes the next of enough copies of the weight (32 MB together, as bench_mm)
static int bench_mm_f32(ggml_backend_dev_t dev) {
    env_override router_f32("FLAGOS_SPACEMIT_ROUTER", "f32");  // the F32 matmul path: its weights carry the router's name
    const uint32_t tiles   = device_tiles(dev);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    REQUIRE(backend != nullptr);
    for (const mm_f32_shape & s : { mm_f32_shape{ 2048, 256, 1, 1 }, mm_f32_shape{ 2048, 256, 32, 1 },
                                    mm_f32_shape{ 2048, 256, 128, 1 }, mm_f32_shape{ 2048, 256, 501, 1 },
                                    mm_f32_shape{ 2048, 32, 1, 1 }, mm_f32_shape{ 2048, 32, 501, 1 } }) {
        const double weight_bytes = (double) s.k * s.n * sizeof(float);
        const size_t copies       = (size_t) std::ceil(32e6 / weight_bytes);  // 16 of the router's, 123 of [2048, 32]
        std::vector<std::unique_ptr<mm_f32_node>> runs;
        for (size_t c = 0; c < copies; c++) {
            runs.push_back(std::make_unique<mm_f32_node>());
            REQUIRE(runs.back()->init(dev, s, true, (uint32_t) (70 + c)));
            CHECK(ggml_backend_graph_compute(backend, runs.back()->g) == GGML_STATUS_SUCCESS);
        }
        bool       ok   = true;
        size_t     next = 0;
        const auto call = [&] {
            ok &= ggml_backend_graph_compute(backend, runs[next++ % copies]->g) == GGML_STATUS_SUCCESS;
        };
        const double probe = time_us(call, 1);
        const int    iters = std::max(20, (int) std::min(1000.0, 2e5 / std::max(probe, 1.0)));  // about 0.2 s a turn
        const double t     = std::min(time_us(call, iters), time_us(call, iters));
        CHECK(ok);
        const double mac = (double) s.rows * s.ne2 * s.k * s.n;
        std::printf("f32mm [%lld, %4lld] by %3lld rows: %9.3f ms (%6.3f Gmac/s per tile, %6.2f GB/s of weights)\n",
                    (long long) s.k, (long long) s.n, (long long) (s.rows * s.ne2), t / 1e3, mac / t / tiles / 1e3,
                    weight_bytes / t / 1e3);
        std::fflush(stdout);
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
// few tokens FLAGOS_SPACEMIT_ATTN_KVH=off against the default (below)
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

    // the KV-head split for a few tokens (plan.md "The KV-head split for a few tokens, from SpacemiT's fork"): 2, 3, 4
    // and 8 tokens at n_kv 256-8192, FLAGOS_SPACEMIT_ATTN_KVH=off against the default, the two taking turns twice, the
    // faster turn of each kept; the copies and GB/s as above
    for (const int64_t n_kv : { 256, 512, 1024, 4096, 8192 }) {
        const int iterations = n_kv <= 256 ? 30 : n_kv <= 512 ? 25 : n_kv <= 1024 ? 20 : n_kv <= 4096 ? 10 : 5;
        for (const int64_t n_tokens : { 2, 3, 4, 8 }) {
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
                t_kvh = std::min(t_kvh, dev_us(nullptr));
            }
            CHECK(ok);
            std::printf("bench    attn %lld tokens n_kv %4lld, FLAGOS_SPACEMIT_ATTN_KVH: off %7.0f us (%5.1f GB/s); "
                        "default %7.0f us (%+.1f%%, %5.1f GB/s)\n",
                        (long long) n_tokens, (long long) n_kv, t_off,
                        kv_bytes / t_off / 1e3, t_kvh, 100.0 * (t_kvh / t_off - 1.0), kv_bytes / t_kvh / 1e3);
        }
    }
    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    return 0;
}

// plan.md "Embedding lookup spec", step 2, --bench-embd only: one lookup call (a graph of the one GET_ROWS node on the
// provider, the output in our buffer) from a vocabulary of 32768 rows in a CPU buffer, at 1024 and 2560 columns and 1,
// 8, 128 and 512 random ids (fixed seed): Q6_K rows by ggml's dequantizers, and the same values as F32 rows
// (dequantized once by ggml's to_float) copied by get_rows.f32, the ceiling of a table kept dequantized. The two take
// turns twice (in order, then reversed), the faster turn of each kept; the Q6_K lookup's kernel by the counter
// (spacemit_embd_paths_take). The same ids every call: after the warm-up their rows are in the caches
static int bench_embd(ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    REQUIRE(backend != nullptr);
    constexpr int64_t n_vocab = 32768;
    for (const int64_t nc : { 1024, 2560 }) {
        ggml_init_params params_w = { ggml_tensor_overhead() * 2, nullptr, true };
        ggml_context *   ctx_w    = ggml_init(params_w);
        ggml_tensor *    wq       = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q6_K, nc, n_vocab);
        ggml_tensor *    wf       = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, nc, n_vocab);
        ggml_set_name(wq, "token_embd.weight");
        ggml_set_name(wf, "token_embd.weight");
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, ggml_backend_cpu_buffer_type());
        REQUIRE(bw != nullptr);
        ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        {
            const std::vector<uint8_t> q = quantize_weights(GGML_TYPE_Q6_K, nc, n_vocab, 41);
            ggml_backend_tensor_set(wq, q.data(), 0, q.size());
        }
        ggml_get_type_traits(GGML_TYPE_Q6_K)->to_float(wq->data, (float *) wf->data, nc * n_vocab);  // host memory
        for (const int64_t n : { 1, 8, 128, 512 }) {
            ggml_init_params params = { ggml_tensor_overhead() * 4 + ggml_graph_overhead() * 2, nullptr, true };
            ggml_context *   ctx_h  = ggml_init(params);
            ggml_context *   ctx_d  = ggml_init(params);
            ggml_tensor *    idx    = ggml_new_tensor_1d(ctx_h, GGML_TYPE_I32, n);
            ggml_tensor *    yq     = ggml_get_rows(ctx_d, wq, idx);
            ggml_tensor *    yf     = ggml_get_rows(ctx_d, wf, idx);
            ggml_backend_buffer_t bh = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
            ggml_backend_buffer_t bd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, ggml_backend_dev_buffer_type(dev));
            REQUIRE(bh != nullptr && bd != nullptr);
            std::vector<int32_t> ids(n);
            uint32_t             r = 1000003u * (uint32_t) (n + nc);
            for (int32_t & id : ids) {
                r  = r * 1664525u + 1013904223u;
                id = (int32_t) ((r >> 8) % n_vocab);
            }
            ggml_backend_tensor_set(idx, ids.data(), 0, ggml_nbytes(idx));
            ggml_cgraph * gq = ggml_new_graph(ctx_d);
            ggml_cgraph * gf = ggml_new_graph(ctx_d);
            ggml_build_forward_expand(gq, yq);
            ggml_build_forward_expand(gf, yf);
            CHECK(ggml_backend_dev_supports_op(dev, yq) && ggml_backend_dev_supports_op(dev, yf));
            const int     iterations = n <= 8 ? 200 : n <= 128 ? 50 : 20;
            bool          ok         = true;
            double        t[2]       = { HUGE_VAL, HUGE_VAL };
            uint64_t      count      = 0;  // the Q6_K lookup's calls
            const auto    run = [&](int k) {
                ggml_cgraph * g     = k == 1 ? gf : gq;
                uint64_t      taken = 0;
                g_embd_paths_take(&taken);
                t[k] = std::min(
                    t[k], time_us([&] { ok &= ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS; }, iterations));
                g_embd_paths_take(&taken);
                if (k == 0) {
                    count += taken;
                }
            };
            for (const int k : { 0, 1, 1, 0 }) {
                run(k);
            }
            CHECK(ok);
            const char * kernel = count != 0 ? "ggml" : "none";
            std::printf("bench    embd %lld rows of %lld: ggml %.1f us, f32 copy %.1f us; kernel %s\n", (long long) n,
                        (long long) nc, t[0], t[1], kernel);
            std::fflush(stdout);
            ggml_backend_buffer_free(bh);
            ggml_backend_buffer_free(bd);
            ggml_free(ctx_h);
            ggml_free(ctx_d);
        }
        ggml_backend_buffer_free(bw);
        ggml_free(ctx_w);
    }
    ggml_backend_free(backend);
    return 0;
}

// plan.md "T6 spec", step 0, --bench-repack only: one whole-weight write (set_tensor, the repack at load) per layout of
// 2560 columns, Qwen3-4B's width (Q4_0 and Q4_1 98304 rows, Q8_0 65536, Q6_K 32768: 65-150 MiB of GGUF bytes), on the
// build's route (8 threads; with spine-runtime the X100 stream, at most its cores + 1 ranges), best of 3; MiB/s of the
// GGUF bytes
static int bench_repack(ggml_backend_dev_t dev, bool spert) {
    const int64_t cols = 2560;
    const struct {
        ggml_type    type;
        const char * name;
        int64_t      rows;
    } shapes[] = {
        { GGML_TYPE_Q4_0, "Q4_0", 98304 },
        { GGML_TYPE_Q4_1, "Q4_1->Q8_0", 98304 },
        { GGML_TYPE_Q8_0, "Q8_0", 65536 },
        { GGML_TYPE_Q6_K, "Q6_K->Q8_0", 32768 },
    };
    for (const auto & s : shapes) {
        ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
        ggml_context *   ctx    = ggml_init(params);
        REQUIRE(ctx != nullptr);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, s.type, cols, s.rows);
        ggml_set_name(w, "blk.0.ffn_up.weight");
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev));
        REQUIRE(buf != nullptr);
        const std::vector<uint8_t> q    = quantize_weights(s.type, cols, s.rows, 43);
        double                     best = HUGE_VAL;
        uint64_t                   used = 0;
        for (int round = 0; round < 3; round++) {
            uint64_t threaded = 0, ranges = 0, through = 0, not_through = 0, reran = 0;
            g_repack_take(&threaded, &ranges);
            g_repack_x100_take(&through, &not_through, &reran);
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_tensor_set(w, q.data(), 0, q.size());
            const auto t1 = std::chrono::steady_clock::now();
            g_repack_take(&threaded, &ranges);
            g_repack_x100_take(&through, &not_through, &reran);
            CHECK(threaded == 1 && not_through == 0 && reran == 0 && (spert ? through == 1 : ranges == 8));
            used = ranges;
            best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
        }
        const double mib = q.size() / 1048576.0;
        std::printf("bench    repack %s %lldx%lld %.1f MiB: %s, %llu ranges, %.0f MiB/s\n", s.name, (long long) s.rows,
                    (long long) cols, mib, spert ? "the X100 stream" : "threads", (unsigned long long) used,
                    mib / best);
        std::fflush(stdout);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
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

    {
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        REQUIRE(backend != nullptr);
        bool ok = true;
        const double t_one   = time_us([&] { ok &= ggml_backend_graph_compute(backend, g_one) == GGML_STATUS_SUCCESS; }, 500);
        const double t_chain = time_us([&] { ok &= ggml_backend_graph_compute(backend, g_chain) == GGML_STATUS_SUCCESS; }, 100);
        const double t_big   = time_us([&] { ok &= ggml_backend_graph_compute(backend, g_big) == GGML_STATUS_SUCCESS; }, 10);
        const double idle    = ai_core_load();
        CHECK(ok);
        std::printf("bench    launch %.1f us, per node %.2f us, 16M-float ADD %.2f GB/s, idle AI-core load %.0f%%\n",
                    t_one, (t_chain - t_one) / (chain - 1), big_bytes / t_big / 1000.0, idle);
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
    // small batches (by default path B where TCM holds the activations next to a slab, here at 4 and 16 rows for gate,
    // up and down, 4 for the Q4_1 down; else path A from 32 rows, else path C), 64 and 128 rows (Q4_0 on the FP16 path,
    // with 8 tiles the pairs route for gate and up, K 2560, the plain route for down, K 9728; the Q4_1 down on path A,
    // bench_mm). Q6_K (run as Q8_0, against the CPU's Q6_K) at its output head, whose 1 row is a generated token's: up
    // to 16 rows (path B) and fewer iterations, as the head holds 15x an FFN matrix's weights. Q4_1 has no CPU timing:
    // ggml-cpu's Q4_1 matmul under 8 threads is where question 8's fault strikes (build.md section 7, the eighth
    // dump), and no check reads the figure
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
            const double flop  = 2.0 * s.k * s.n * m;
            if (s.type == GGML_TYPE_Q4_1) {
                CHECK(ok);
                std::printf("bench    %s matmul %lldx%lld rows %3lld: provider %8.0f us (%6.1f GFLOP/s)\n",
                            ggml_type_name(s.type), (long long) s.k, (long long) s.n, (long long) m, t_dev,
                            flop / t_dev / 1e3);
                continue;
            }
            const double t_cpu = time_us([&] { ggml_backend_graph_compute(cpu8, run.g_cpu); }, iters);
            CHECK(ok);
            std::printf("bench    %s matmul %lldx%lld rows %3lld: provider %8.0f us (%6.1f GFLOP/s), cpu %8.0f us (%6.1f GFLOP/s)\n",
                        ggml_type_name(s.type), (long long) s.k, (long long) s.n, (long long) m, t_dev,
                        flop / t_dev / 1e3, t_cpu, flop / t_cpu / 1e3);
        }
    }
    ggml_backend_free(backend);
    ggml_backend_free(cpu8);
    return bench_attn(dev) != 0 || bench_mm(dev) != 0 ? 1 : bench_mm_f32(dev);
}

//
// --bench-x100 (plan.md "Output layer probe"): is there DRAM bandwidth for the X100 cores beside the AI cores' 1-row
// matmul? Part 1 times the real matmul while ordinary threads read another buffer; part 2 splits one matmul's weight
// rows between the provider and the CPU backend, both released at the same instant
//

// X100BENCH_<name> as a positive integer, else def
static int x100_env(const char * name, int def) {
    const char * v = std::getenv(name);
    const int    n = v != nullptr ? std::atoi(v) : 0;
    return n > 0 ? n : def;
}

static int x100_cpu() {
#ifdef __linux__
    return sched_getcpu();
#else
    return -1;
#endif
}

struct x100_stat {
    double med = 0.0, lo = 0.0, hi = 0.0;
};

static x100_stat x100_stats(std::vector<double> v) {
    x100_stat s;
    if (!v.empty()) {
        std::sort(v.begin(), v.end());
        s.med = v[v.size() / 2];
        s.lo  = v.front();
        s.hi  = v.back();
    }
    return s;
}

// one tensor of rows x k in a buffer of buft, from bytes (GGUF rows for weights, floats for the activation row)
struct x100_tensor {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         t   = nullptr;

    bool init(ggml_backend_buffer_type_t buft, ggml_type type, int64_t k, int64_t rows, const void * bytes) {
        ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
        ctx                     = ggml_init(params);
        t                       = ggml_new_tensor_2d(ctx, type, k, rows);
        buf                     = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (buf == nullptr) {
            return false;
        }
        ggml_backend_tensor_set(t, bytes, 0, ggml_nbytes(t));
        return true;
    }

    ~x100_tensor() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

// y = W x for one activation row, y in a buffer of buft
struct x100_mm {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         y   = nullptr;
    ggml_cgraph *         g   = nullptr;

    bool init(ggml_backend_buffer_type_t buft, ggml_tensor * w, ggml_tensor * x) {
        ggml_init_params params = { ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ctx                     = ggml_init(params);
        y                       = ggml_mul_mat(ctx, w, x);
        g                       = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, y);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        return buf != nullptr;
    }

    std::vector<float> out() const {
        std::vector<float> v(ggml_nelements(y));
        ggml_backend_tensor_get(y, v.data(), 0, ggml_nbytes(y));
        return v;
    }

    ~x100_mm() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

// the readers' buffer: 64-byte aligned, huge pages asked for on Linux (as head-probe.cpp), every page touched
struct x100_buffer {
    uint64_t * p     = nullptr;
    size_t     bytes = 0;

    explicit x100_buffer(size_t n) : bytes(n) {
#if defined(_WIN32)
        p = static_cast<uint64_t *>(::operator new(n, std::align_val_t(64)));
#else
        void * m = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) {
            return;
        }
#    ifdef __linux__
        madvise(m, n, MADV_HUGEPAGE);
#    endif
        p = static_cast<uint64_t *>(m);
#endif
        std::memset(p, 1, n);
    }

    ~x100_buffer() {
        if (p == nullptr) {
            return;
        }
#if defined(_WIN32)
        ::operator delete(p, std::align_val_t(64));
#else
        munmap(p, bytes);
#endif
    }

    x100_buffer(const x100_buffer &)             = delete;
    x100_buffer & operator=(const x100_buffer &) = delete;
};

// plain u64 adds in 4 independent sums, a loop the compiler can vectorize
static uint64_t x100_sum(const uint64_t * p, size_t n) {
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (size_t i = 0; i < n; i += 4) {
        s0 += p[i];
        s1 += p[i + 1];
        s2 += p[i + 2];
        s3 += p[i + 3];
    }
    return s0 + s1 + s2 + s3;
}

// n ordinary threads, each summing its own contiguous slice of the buffer in 8 MiB steps until stopped. Each counts its
// bytes every 256 KiB in its own atomic, so a sample is off by at most 256 KiB a thread
class x100_readers {
  public:
    x100_readers(const x100_buffer & b, int n) : slots(n) {
        const size_t words = b.bytes / sizeof(uint64_t) / n;
        for (int t = 0; t < n; t++) {
            threads.emplace_back([this, t, p = b.p + t * words, words] { run(slots[t], p, words); });
        }
    }

    ~x100_readers() { stop(); }

    uint64_t bytes() const {
        uint64_t n = 0;
        for (const slot & s : slots) {
            n += s.bytes.load(std::memory_order_relaxed);
        }
        return n;
    }

    // joins the threads; each thread's CPU at its start and end, "start:end,..."
    std::string stop() {
        halt.store(true, std::memory_order_relaxed);
        for (std::thread & t : threads) {
            t.join();
        }
        threads.clear();
        std::string cpus;
        for (const slot & s : slots) {
            cpus += (cpus.empty() ? "" : ",") + std::to_string(s.cpu0) + ":" + std::to_string(s.cpu1);
        }
        return cpus;
    }

  private:
    struct alignas(64) slot {
        std::atomic<uint64_t> bytes{ 0 };
        int                   cpu0 = -1, cpu1 = -1;
        uint64_t              sum  = 0;
    };

    void run(slot & s, const uint64_t * p, size_t words) {
        s.cpu0                = x100_cpu();
        const size_t step     = (8u << 20) / sizeof(uint64_t);
        const size_t chunk    = (256u << 10) / sizeof(uint64_t);
        uint64_t     sum = 0, n = 0;
        size_t       off = 0;
        while (!halt.load(std::memory_order_relaxed)) {
            for (size_t c = 0; c < step; c += chunk) {
                sum += x100_sum(p + off + c, chunk);
                n += chunk * sizeof(uint64_t);
                s.bytes.store(n, std::memory_order_relaxed);
            }
            off = off + 2 * step <= words ? off + step : 0;
        }
        s.sum  = sum;
        s.cpu1 = x100_cpu();
    }

    std::vector<slot>        slots;
    std::vector<std::thread> threads;
    std::atomic<bool>        halt{ false };
};

static double x100_ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ms per call of g on backend: 1 warm-up repetition, then before(), then reps repetitions of calls computes each
static std::vector<double> x100_time(ggml_backend_t backend, ggml_cgraph * g, int calls, int reps, bool & ok,
                                     const std::function<void()> & before = nullptr) {
    std::vector<double> ms;
    for (int r = 0; r <= reps; r++) {
        if (r == 1 && before) {
            before();
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < calls; i++) {
            ok &= ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS;
        }
        if (r > 0) {
            ms.push_back(x100_ms(t0, std::chrono::steady_clock::now()) / calls);
        }
    }
    return ms;
}

// part 2's timing: per matmul the provider's graph and the CPU's graph released at the same instant to two
// already-running host threads that spin on a start counter (nullptr leaves a side and its thread out); per
// repetition, after 1 warm-up, ms per matmul over calls matmuls of the wall time (release to both done) and of each
// side's own compute
struct x100_split_times {
    std::vector<double> wall, prov, cpu;
    bool                ok = true;
};

static x100_split_times x100_shared(ggml_backend_t backend, ggml_cgraph * g_prov, ggml_backend_t cpu,
                                    ggml_cgraph * g_cpu, int calls, int reps) {
    using clock = std::chrono::steady_clock;
    struct side {
        ggml_backend_t    be;
        ggml_cgraph *     g;
        double            ms = 0.0;
        clock::time_point end;
        bool              ok = true;
    };
    std::vector<side> sides;
    if (g_prov != nullptr) {
        sides.push_back({ backend, g_prov, 0.0, clock::time_point(), true });
    }
    if (g_cpu != nullptr) {
        sides.push_back({ cpu, g_cpu, 0.0, clock::time_point(), true });
    }
    std::atomic<int>         go{ 0 }, done{ 0 };
    std::atomic<bool>        quit{ false };
    std::vector<std::thread> threads;
    for (side & s : sides) {
        threads.emplace_back([&go, &done, &quit, &s] {
            int seen = 0;
            for (;;) {
                int v;
                while ((v = go.load(std::memory_order_acquire)) == seen) {
                }
                seen = v;
                if (quit.load(std::memory_order_relaxed)) {
                    break;
                }
                const clock::time_point t0 = clock::now();
                s.ok &= ggml_backend_graph_compute(s.be, s.g) == GGML_STATUS_SUCCESS;
                s.end = clock::now();
                s.ms  = x100_ms(t0, s.end);
                done.fetch_add(1, std::memory_order_release);
            }
        });
    }
    x100_split_times t;
    for (int r = 0; r <= reps; r++) {
        double wall = 0.0, ms[2] = { 0.0, 0.0 };
        for (int i = 0; i < calls; i++) {
            done.store(0, std::memory_order_relaxed);
            const clock::time_point release = clock::now();
            go.fetch_add(1, std::memory_order_release);
            while (done.load(std::memory_order_acquire) < (int) sides.size()) {
                std::this_thread::yield();
            }
            clock::time_point end = release;
            for (size_t k = 0; k < sides.size(); k++) {
                end = std::max(end, sides[k].end);
                ms[k] += sides[k].ms;
            }
            wall += x100_ms(release, end);
        }
        if (r > 0) {
            t.wall.push_back(wall / calls);
            t.prov.push_back(g_prov != nullptr ? ms[0] / calls : 0.0);
            t.cpu.push_back(g_cpu != nullptr ? ms[g_prov != nullptr ? 1 : 0] / calls : 0.0);
        }
    }
    quit.store(true, std::memory_order_relaxed);
    go.fetch_add(1, std::memory_order_release);
    for (std::thread & th : threads) {
        th.join();
    }
    for (const side & s : sides) {
        t.ok &= s.ok;
    }
    return t;
}

// whether this build's CPU backend runs its threads on the AI cores: ggml-cpu built with GGML_CPU_RISCV64_SPACEMIT
// defines GGML_USE_CPU_RISCV64_SPACEMIT for itself only (ggml-cpu/CMakeLists.txt), lists its IME buffer type among
// its extra buffer types (ggml-cpu.cpp) and binds every compute thread to the AI cores (ggml-cpu.c,
// ggml_graph_compute_thread), so this program sees the macro through that buffer type
static bool x100_cpu_uses_ai_cores() {
#if defined(GGML_USE_CPU_RISCV64_SPACEMIT)
    return true;
#else
    ggml_backend_reg_t reg = ggml_backend_cpu_reg();
    auto get = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts"));
    ggml_backend_buffer_type_t * b = get != nullptr ? get(ggml_backend_reg_dev_get(reg, 0)) : nullptr;
    for (; b != nullptr && *b != nullptr; b++) {
        if (std::strcmp(ggml_backend_buft_name(*b), "CPU_RISCV64_SPACEMIT") == 0) {
            return true;
        }
    }
    return false;
#endif
}

static int bench_x100(ggml_backend_dev_t dev) {
    const int  reps  = x100_env("X100BENCH_REPS", 5);
    const bool small = x100_env("X100BENCH_SMALL", 0) == 1;
    struct x100_shape {
        const char * name;
        ggml_type    type;
        int64_t      k, n;
        bool         part1;
        int          calls1, calls2;  // matmuls per repetition in parts 1 and 2
    };
    const x100_shape shapes[] = {
        { "head", GGML_TYPE_Q6_K, small ? 256 : 2560, small ? 4096 : 151936, true, 1, 1 },
        { "ffn_up", GGML_TYPE_Q4_0, small ? 256 : 2560, small ? 1024 : 9728, true, small ? 20 : 200, small ? 10 : 100 },
        { "ffn_down", GGML_TYPE_Q4_0, small ? 1024 : 9728, small ? 256 : 2560, false, 0, small ? 10 : 100 },
    };
    const double fracs[]  = { 0.85, 0.75, 0.65, 0.5 };
    const int    cs[]     = { 8, 4 };
    int          n_checks = 0, n_passed = 0;
    const auto   check    = [&](bool ok, const char * shape, const char * what, double f) {
        n_checks++;
        n_passed += ok;
        if (!ok) {
            std::fprintf(stderr, "FAIL (x100bench %s f=%.2f): %s\n", shape, f, what);
            g_failures++;
        }
    };

    ggml_backend_t backend = nullptr;
    {
        env_override tcm("FLAGOS_SPACEMIT_TEST_TCM", "393216");  // the host executor's TCM: the 1-row path (bench_mm)
        backend = ggml_backend_dev_init(dev, nullptr);
    }
    REQUIRE(backend != nullptr);
    const bool part2 = !x100_cpu_uses_ai_cores();
    if (!part2) {
        std::printf("x100bench: the CPU backend uses the AI cores in this build, part 2 skipped\n");
    }
    ggml_backend_t           cpu[2] = { nullptr, nullptr };
    ggml_threadpool *        pool[2] = { nullptr, nullptr };
    for (int i = 0; part2 && i < 2; i++) {
        ggml_threadpool_params tp = ggml_threadpool_params_default(cs[i]);
        cpu[i]                    = ggml_backend_cpu_init();
        pool[i]                   = ggml_threadpool_new(&tp);
        REQUIRE(cpu[i] != nullptr && pool[i] != nullptr);
        ggml_backend_cpu_set_n_threads(cpu[i], cs[i]);
        ggml_backend_cpu_set_threadpool(cpu[i], pool[i]);
    }
    x100_buffer rbuf((size_t) (small ? 64 : 1024) << 20);
    REQUIRE(rbuf.p != nullptr);

    ggml_backend_buffer_type_t buft_dev = ggml_backend_dev_buffer_type(dev);
    ggml_backend_buffer_type_t buft_cpu = ggml_backend_cpu_buffer_type();
    for (const x100_shape & s : shapes) {
        if (!s.part1 && !part2) {
            continue;
        }
        const char *               name  = s.name;
        const size_t               row   = ggml_row_size(s.type, s.k);
        const double               bytes = (double) s.n * ime_row_bytes(s.type, s.k);  // as bench_mm: 34/32 or 18/32
        const std::vector<uint8_t> q     = quantize_weights(s.type, s.k, s.n, 61);
        const std::vector<float>   xv    = random_values(s.k, 62, 2.0f);
        x100_tensor                x;
        REQUIRE(x.init(buft_cpu, GGML_TYPE_F32, s.k, 1, xv.data()));
        auto w_full  = std::make_unique<x100_tensor>();
        auto mm_full = std::make_unique<x100_mm>();
        REQUIRE(w_full->init(buft_dev, s.type, s.k, s.n, q.data()) && mm_full->init(buft_dev, w_full->t, x.t));
        uint64_t paths[3];
        g_mm_paths_take(&paths[0], &paths[1], &paths[2]);

        // part 1: a. the matmul alone; per T b. the readers alone, c. both; d. the matmul alone again
        if (s.part1) {
            bool            ok    = true;
            const x100_stat alone = x100_stats(x100_time(backend, mm_full->g, s.calls1, reps, ok));
            for (const int n_thr : { 4, 8 }) {
                std::vector<double> gbs;
                {
                    x100_readers rd(rbuf, n_thr);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    for (int r = 0; r <= reps; r++) {
                        const uint64_t b0 = rd.bytes();
                        const auto     t0 = std::chrono::steady_clock::now();
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        const uint64_t b1 = rd.bytes();
                        const double   ms = x100_ms(t0, std::chrono::steady_clock::now());
                        if (r > 0) {
                            gbs.push_back((double) (b1 - b0) / 1e6 / ms);
                        }
                    }
                }
                const x100_stat     rd_alone = x100_stats(gbs);
                uint64_t            b0 = 0, b1 = 0;
                double              window = 0.0;
                std::vector<double> with;
                std::string         cpus;
                {
                    x100_readers rd(rbuf, n_thr);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    std::chrono::steady_clock::time_point t0;
                    with = x100_time(backend, mm_full->g, s.calls1, reps, ok, [&] {
                        b0 = rd.bytes();
                        t0 = std::chrono::steady_clock::now();
                    });
                    b1     = rd.bytes();
                    window = x100_ms(t0, std::chrono::steady_clock::now());
                    cpus   = rd.stop();
                }
                const x100_stat ws      = x100_stats(with);
                const double    rd_with = (double) (b1 - b0) / 1e6 / window;
                const double    mm_a    = bytes / 1e6 / alone.med;
                const double    mm_w    = bytes / 1e6 / ws.med;
                std::printf("x100bench part1 %s T=%d: matmul alone %.3f (%.3f-%.3f), with readers %.3f (%.3f-%.3f), "
                            "ratio %.3f; readers alone %.1f, with matmul %.1f; weight bytes %.1f -> matmul %.1f alone / "
                            "%.1f with readers; sum with readers %.1f; cpus %s\n",
                            name, n_thr, alone.med, alone.lo, alone.hi, ws.med, ws.lo, ws.hi, ws.med / alone.med,
                            rd_alone.med, rd_with, bytes / 1048576.0, mm_a, mm_w, mm_w + rd_with, cpus.c_str());
                std::fflush(stdout);
            }
            const x100_stat again = x100_stats(x100_time(backend, mm_full->g, s.calls1, reps, ok));
            std::printf("x100bench part1 %s: matmul alone again %.3f (%.3f-%.3f), ratio to first %.3f\n", name,
                        again.med, again.lo, again.hi, again.med / alone.med);
            std::fflush(stdout);
            check(ok, name, "part 1 computes", 1.0);
        }
        if (!part2) {
            continue;
        }

        // part 2: f = 1.0 (the provider alone) and f = 0 (the CPU alone, C = 8), then each split
        const auto line = [&](double f, int c, const x100_split_times & t, double ref) {
            const x100_stat w = x100_stats(t.wall);
            std::printf("x100bench part2 %s f=%.2f C=%d: wall %.3f (%.3f-%.3f), provider side %.3f, cpu side %.3f, "
                        "ratio to f=1.0 %.3f\n",
                        name, f, c, w.med, w.lo, w.hi, x100_stats(t.prov).med, x100_stats(t.cpu).med,
                        ref > 0.0 ? w.med / ref : 1.0);
            std::fflush(stdout);
            return w.med;
        };
        x100_split_times t1 = x100_shared(backend, mm_full->g, nullptr, nullptr, s.calls2, reps);
        const double     ref = line(1.0, 0, t1, 0.0);
        check(t1.ok, name, "provider alone computes", 1.0);
        const std::vector<float> y_prov = mm_full->out();
        mm_full.reset();
        w_full.reset();
        std::vector<float> y_cpu;
        {
            x100_tensor w_cpu;
            x100_mm     mm_cpu;
            REQUIRE(w_cpu.init(buft_cpu, s.type, s.k, s.n, q.data()) && mm_cpu.init(buft_cpu, w_cpu.t, x.t));
            const x100_split_times t0 = x100_shared(nullptr, nullptr, cpu[0], mm_cpu.g, s.calls2, reps);
            line(0.0, cs[0], t0, ref);
            check(t0.ok, name, "cpu alone computes", 0.0);
            y_cpu = mm_cpu.out();
        }
        for (const double f : fracs) {
            const int64_t nf = (int64_t) std::llround(f * (double) s.n / 32.0) * 32;
            x100_tensor   w_p, w_c;
            x100_mm       mm_p, mm_c;
            REQUIRE(nf > 0 && nf < s.n);
            REQUIRE(w_p.init(buft_dev, s.type, s.k, nf, q.data()) && mm_p.init(buft_dev, w_p.t, x.t));
            REQUIRE(w_c.init(buft_cpu, s.type, s.k, s.n - nf, q.data() + nf * row) && mm_c.init(buft_cpu, w_c.t, x.t));
            bool ok = true;
            for (int i = 0; i < 2; i++) {
                const x100_split_times t = x100_shared(backend, mm_p.g, cpu[i], mm_c.g, s.calls2, reps);
                line(f, cs[i], t, ref);
                ok &= t.ok;
            }
            check(ok, name, "split computes", f);
            // the 1-row path gives each weight row's result from that row alone (the activation row quantized by K
            // blocks, then 32-row slabs dealt to the tiles: ime.cpp gemv), so the first nf rows match bit for bit
            const std::vector<float> yp = mm_p.out();
            const std::vector<float> yc = mm_c.out();
            check(std::memcmp(yp.data(), y_prov.data(), nf * sizeof(float)) == 0, name,
                  "provider rows differ from the provider's full matmul", f);
            check(nmse(yc.data(), y_cpu.data() + nf, s.n - nf) <= k_mul_mat_max_nmse, name,
                  "cpu rows outside the MUL_MAT bound against the cpu's full matmul", f);
        }
        g_mm_paths_take(&paths[0], &paths[1], &paths[2]);
        check(paths[0] == 0 && paths[1] == 0 && paths[2] == 0, name, "a provider matmul left the 1-row path", 1.0);
    }
    for (int i = 0; i < 2; i++) {
        if (cpu[i] != nullptr) {
            ggml_backend_free(cpu[i]);
            ggml_threadpool_free(pool[i]);
        }
    }
    ggml_backend_free(backend);
    std::printf("x100bench: checks %d/%d %s\n", n_passed, n_checks, n_passed == n_checks ? "PASS" : "FAIL");
    return 0;
}

// --bench-mmid only (plan.md "The Q4_0 experts: the fork's signed layout and 8-row kernel"): one MUL_MAT_ID node of
// Q4_0 experts through the backend at Qwen3.6-35B-A3B's two shapes (gate and up [2048, 512, 256], one activation row a
// token; down [512, 2048, 256], one a slot), 8 ids a token, for five row patterns: 1 token (the 1-row path), 4 tokens
// of 32 different experts (single rows), 4 and 8 tokens of the same 8 experts (groups of 4 and of 8), 32 tokens drawn
// at random (seed fixed). FLAGOS_SPACEMIT_EXPERT_Q4_0=plain and the default (m8) take turns twice, the weight written
// again at each turn (its layout is decided at load); per cell 5 warm-up computes and 60 timed, the median and the
// minimum in us a node, then per shape and pattern the default's median over plain's (all of a setting's timed
// computes). The host executor runs the reference kernels: there the numbers mean nothing
static int bench_mmid(ggml_backend_dev_t dev) {
    ggml_backend_t backend = nullptr;
    {
        env_override tcm("FLAGOS_SPACEMIT_TEST_TCM", "393216");  // the host executor's TCM, as bench_mm
        backend = ggml_backend_dev_init(dev, nullptr);
    }
    REQUIRE(backend != nullptr);
    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    constexpr int64_t n_as = 256, n_used = 8;
    constexpr int     n_warm = 5, n_timed = 60;
    struct mmid_shape {
        const char * what;
        int64_t      k, n, ne11;
    };
    const mmid_shape shapes[] = { { "gate, up", 2048, 512, 1 }, { "down", 512, 2048, n_used } };
    struct mmid_pattern {
        const char *         what;
        int64_t              n_tok;
        std::vector<int32_t> ids;
    };
    std::vector<mmid_pattern> patterns;
    {
        mmid_pattern a = { "1 token", 1, {} }, b = { "4 tokens, 32 experts", 4, {} };
        mmid_pattern c = { "4 tokens, the same 8 experts", 4, {} }, e = { "8 tokens, the same 8 experts", 8, {} };
        mmid_pattern r = { "32 tokens, random ids", 32, {} };
        for (int64_t s = 0; s < n_used; s++) {
            a.ids.push_back((int32_t) (s * 37 % n_as));
        }
        for (int64_t j = 0; j < 4 * n_used; j++) {
            b.ids.push_back((int32_t) (j * 7 % n_as));  // 7 is prime to 256: 32 different experts
        }
        for (int64_t t = 0; t < 8; t++) {
            for (int64_t s = 0; s < n_used; s++) {
                if (t < 4) {
                    c.ids.push_back((int32_t) (s * 31 % n_as));
                }
                e.ids.push_back((int32_t) (s * 31 % n_as));
            }
        }
        uint32_t             seed = 4711;
        std::vector<int32_t> perm((size_t) n_as);
        for (int64_t t = 0; t < 32; t++) {  // distinct experts within a token, as ARGSORT gives them
            for (int64_t x = 0; x < n_as; x++) {
                perm[(size_t) x] = (int32_t) x;
            }
            for (int64_t s = 0; s < n_used; s++) {
                seed = seed * 1664525u + 1013904223u;
                std::swap(perm[(size_t) s], perm[(size_t) (s + (seed >> 8) % (uint32_t) (n_as - s))]);
                r.ids.push_back(perm[(size_t) s]);
            }
        }
        patterns = { a, b, c, e, r };
    }

    const char * const settings[2] = { "plain", nullptr };  // nullptr: unset, the default (m8)
    for (const mmid_shape & sh : shapes) {
        const std::vector<uint8_t> q = quantize_weights(GGML_TYPE_Q4_0, sh.k, sh.n * n_as, 1);
        ggml_init_params           pw    = { ggml_tensor_overhead(), nullptr, true };
        ggml_context *             ctx_w = ggml_init(pw);
        REQUIRE(ctx_w != nullptr);
        ggml_tensor * w = ggml_new_tensor_3d(ctx_w, GGML_TYPE_Q4_0, sh.k, sh.n, n_as);
        ggml_set_name(w, "blk.0.ffn_gate_exps.weight");
        ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
        REQUIRE(buf_w != nullptr);

        // the patterns' nodes, built once: activations and ids on the host, the output in our buffer
        std::vector<ggml_context *>         ctxs;
        std::vector<ggml_backend_buffer_t>  bufs;
        std::vector<ggml_cgraph *>          graphs;
        for (const mmid_pattern & pt : patterns) {
            ggml_init_params p     = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true };
            ggml_context *   ctx_h = ggml_init(p);
            ggml_context *   ctx_d = ggml_init(p);
            REQUIRE(ctx_h != nullptr && ctx_d != nullptr);
            ggml_tensor * x   = ggml_new_tensor_3d(ctx_h, GGML_TYPE_F32, sh.k, sh.ne11, pt.n_tok);
            ggml_tensor * ids = ggml_new_tensor_2d(ctx_h, GGML_TYPE_I32, n_used, pt.n_tok);
            ggml_tensor * y   = ggml_mul_mat_id(ctx_d, w, x, ids);
            ggml_backend_buffer_t bh = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
            ggml_backend_buffer_t bd = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, buft);
            REQUIRE(bh != nullptr && bd != nullptr);
            REQUIRE((int64_t) pt.ids.size() == n_used * pt.n_tok);
            const std::vector<float> xv = random_values(ggml_nelements(x), 2, 1.0f);
            ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(ids, pt.ids.data(), 0, ggml_nbytes(ids));
            ggml_cgraph * g = ggml_new_graph(ctx_d);
            ggml_build_forward_expand(g, y);
            ctxs.insert(ctxs.end(), { ctx_h, ctx_d });
            bufs.insert(bufs.end(), { bh, bd });
            graphs.push_back(g);
        }

        std::vector<std::vector<double>> all(patterns.size() * 2);  // [pattern][setting]: every timed compute, us
        for (int turn = 0; turn < 4; turn++) {
            const int    k = turn % 2;
            env_override sw("FLAGOS_SPACEMIT_EXPERT_Q4_0", settings[k]);
            ggml_backend_tensor_set(w, q.data(), 0, q.size());  // the repack in this setting's layout
            for (size_t pi = 0; pi < patterns.size(); pi++) {
                bool ok = ggml_backend_dev_supports_op(dev, ggml_graph_node(graphs[pi], -1));
                for (int i = 0; i < n_warm && ok; i++) {
                    ok = ggml_backend_graph_compute(backend, graphs[pi]) == GGML_STATUS_SUCCESS;
                }
                std::vector<double> t;
                for (int i = 0; i < n_timed && ok; i++) {
                    const auto t0 = std::chrono::steady_clock::now();
                    ok            = ggml_backend_graph_compute(backend, graphs[pi]) == GGML_STATUS_SUCCESS;
                    const auto t1 = std::chrono::steady_clock::now();
                    t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                }
                CHECK(ok && !t.empty());
                std::vector<double> sorted = t;
                std::sort(sorted.begin(), sorted.end());
                all[pi * 2 + k].insert(all[pi * 2 + k].end(), t.begin(), t.end());
                std::printf("bench    mmid %-8s [%lld, %lld, %lld] %-28s %-7s turn %d: median %9.1f us, min %9.1f us a "
                            "node\n",
                            sh.what, (long long) sh.k, (long long) sh.n, (long long) n_as, patterns[pi].what,
                            settings[k] != nullptr ? settings[k] : "default", turn + 1, sorted[sorted.size() / 2],
                            sorted.front());
            }
        }
        for (size_t pi = 0; pi < patterns.size(); pi++) {
            double med[2];
            for (int k = 0; k < 2; k++) {
                std::vector<double> & v = all[pi * 2 + k];
                std::sort(v.begin(), v.end());
                med[k] = v[v.size() / 2];
            }
            std::printf("bench    mmid %-8s [%lld, %lld, %lld] %-28s default over plain: %.3f (medians %.1f and %.1f "
                        "us, %d computes each)\n",
                        sh.what, (long long) sh.k, (long long) sh.n, (long long) n_as, patterns[pi].what,
                        med[1] / med[0], med[1], med[0], 2 * n_timed);
        }
        for (ggml_backend_buffer_t b : bufs) {
            ggml_backend_buffer_free(b);
        }
        for (ggml_context * c : ctxs) {
            ggml_free(c);
        }
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx_w);
    }
    ggml_backend_free(backend);
    return 0;
}

// the pair-wait launches (spacemit_pairwait_launch) on a stream of the plugin device the core opened: a backend alive
// would share its executor and core grant. Made where no backend is alive, so the stream makes the one executor (1
// created, else a FAIL), which reads FLAGOS_SPACEMIT_TEST_THREADS then; the launches' steps are taken off the counter
struct pairwait_stream {
    const spacemit_plugin_v1 *  table  = nullptr;
    spacemit_plugin_stream *    stream = nullptr;
    spacemit_pairwait_launch_fn launch = nullptr;

    bool init(ggml_backend_dev_t dev) {
        table                         = spacemit_plugin_table(spacemit_device_plugin(dev));
        spacemit_plugin_device * pdev = spacemit_device_plugin_device(dev);
        launch = table != nullptr ? reinterpret_cast<spacemit_pairwait_launch_fn>(
                                        table->get_proc_address("spacemit_pairwait_launch")) :
                                    nullptr;
        if (pdev == nullptr || launch == nullptr) {
            std::fprintf(stderr, "FAIL: pairwait: no plugin device or no spacemit_pairwait_launch\n");
            g_failures++;
            return false;
        }
        g_executors_take();
        const bool     made = table->stream_create(pdev, device_tiles(dev), &stream) == SPACEMIT_STATUS_SUCCESS;
        const uint64_t own  = g_executors_take();
        if (!made || own != 1) {
            std::fprintf(stderr, "FAIL: pairwait: stream %s, %llu executors created (1 expected: no backend alive)\n",
                         made ? "made" : "not made", (unsigned long long) own);
            g_failures++;
            return false;
        }
        return true;
    }

    ~pairwait_stream() {
        if (stream != nullptr) {
            table->stream_destroy(stream);
        }
        g_steps_take();
    }
};

static const char * const k_pairwait_names[3] = { "ours", "spine", "spine+exit" };

// the pair-wait launches of --bench-pairwait, small: with the tiles at the same time (the AI cores, the host's threads)
// every variant makes its 10,000 rendezvous on every tile; on the serial executor every variant returns at once. On
// the AI cores spine, which has no exit, is left to --bench-pairwait: a launch gone wrong would hang this check
static int check_pairwait(ggml_backend_dev_t dev, bool host) {
    constexpr uint32_t n     = 10000;
    const uint32_t     tiles = device_tiles(dev);
    int                n_checks = 0, n_ok = 0;
    for (const char * threads : { "0", "1" }) {
        if (!host && threads[0] == '1') {
            break;  // the AI cores: one pass
        }
        env_override    on_threads("FLAGOS_SPACEMIT_TEST_THREADS", threads);
        const bool      concurrent = !host || threads[0] == '1';
        pairwait_stream s;
        if (!s.init(dev)) {
            continue;
        }
        for (uint32_t v = 0; v < 3; v++) {
            if (!host && v == SPACEMIT_PAIRWAIT_SPINE) {
                continue;
            }
            spacemit_pairwait_result r   = {};
            const bool               ran = s.launch(s.stream, v, n, &r);
            bool ok = ran && r.n_tiles == tiles && r.concurrent == concurrent && r.executor != nullptr;
            for (uint32_t t = 0; t < r.n_tiles; t++) {
                ok = ok && r.waits[t] == (concurrent ? n : 0);
            }
            n_checks++;
            n_ok += ok;
            if (!ok) {
                std::fprintf(stderr, "FAIL: pairwait %s, %s: %s, %u tiles, concurrent %d, tile 0 %u waits\n",
                             k_pairwait_names[v], r.executor != nullptr ? r.executor : "?", ran ? "ran" : "failed",
                             r.n_tiles, (int) r.concurrent, r.waits[0]);
                g_failures++;
            }
        }
    }
    std::printf("pairchk  %s, 10000 rendezvous a tile%s: %d/%d\n",
                host ? "ours, spine and spine+exit" : "ours and spine+exit", host ? " on threads, none in turn" : "",
                n_ok, n_checks);
    return 0;
}

// --bench-pairwait: what a pair rendezvous costs on the device's tiles. A launch a variant on all tiles, each tile
// making n rendezvous with tile ith ^ 1 back to back after 1,000 untimed, timing its own loop: ours
// (spacemit_pair_wait), spine (SpacemiT's barrier, ggml-cpu/spacemit/spine_barrier.h, no exit) and spine+exit (the
// same with our two exits). Three rounds, the order rotated each round; a round's figure the mean over the tiles of
// tile ns / n, a variant's the median of its rounds. Skipped where the tiles run in turn (the serial host executor)
static int bench_pairwait(ggml_backend_dev_t dev) {
    constexpr uint32_t n        = 1000000;
    constexpr int      n_rounds = 3;
    pairwait_stream    s;
    if (!s.init(dev)) {
        return 1;
    }
    spacemit_pairwait_result probe = {};
    REQUIRE(s.launch(s.stream, SPACEMIT_PAIRWAIT_OURS, 1000, &probe));
    std::printf("pairwait: %u tiles, pause in ours: %s (__riscv_zihintpause), executor %s\n", probe.n_tiles,
                probe.pause_in_ours ? "yes" : "no", probe.executor);
    if (!probe.concurrent) {
        std::printf("pairwait: skipped, the tiles run in turn\n");
        return 0;
    }
    std::vector<double> rounds[3];
    for (int round = 0; round < n_rounds; round++) {
        for (int j = 0; j < 3; j++) {
            const int                v   = (round + j) % 3;
            spacemit_pairwait_result r   = {};
            const bool               ran = s.launch(s.stream, (uint32_t) v, n, &r);
            bool                     all = ran && r.n_tiles > 0;
            std::vector<double>      per;
            for (uint32_t t = 0; t < r.n_tiles; t++) {
                all = all && r.waits[t] == n;
                per.push_back((double) r.ns[t] / n);
            }
            if (!all) {
                std::fprintf(stderr, "FAIL: pairwait round %d %s: %s, tile 0 made %u of %u waits\n", round + 1,
                             k_pairwait_names[v], ran ? "ran" : "the launch failed", r.waits[0], n);
                g_failures++;
                continue;
            }
            std::sort(per.begin(), per.end());
            double mean = 0;
            for (double x : per) {
                mean += x / per.size();
            }
            const double median = (per[(per.size() - 1) / 2] + per[per.size() / 2]) / 2;
            std::printf("pairwait round %d %s: %.1f ns a wait (tiles min %.1f, median %.1f, max %.1f), %u waits a "
                        "tile\n",
                        round + 1, k_pairwait_names[v], mean, per.front(), median, per.back(), n);
            rounds[v].push_back(mean);
        }
    }
    for (int v = 0; v < 3; v++) {
        std::vector<double> & m = rounds[v];
        if (!m.empty()) {
            std::sort(m.begin(), m.end());
            std::printf("pairwait %s: median of %zu rounds %.1f ns a wait\n", k_pairwait_names[v], m.size(),
                        (m[(m.size() - 1) / 2] + m[m.size() / 2]) / 2);
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
    bool expect_device = false;
    bool expect_none   = false;
    bool full          = false;
    bool run_bench     = false;
    bool run_attn      = false;
    bool run_mm        = false;
    bool run_f32       = false;
    bool run_embd      = false;
    bool run_repack    = false;
    bool run_x100      = false;
    bool run_mmid      = false;
    bool run_pairwait  = false;
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
        } else if (std::strcmp(argv[i], "--bench-f32") == 0) {
            run_f32 = true;
        } else if (std::strcmp(argv[i], "--bench-embd") == 0) {
            run_embd = true;
        } else if (std::strcmp(argv[i], "--bench-repack") == 0) {
            run_repack = true;
        } else if (std::strcmp(argv[i], "--bench-x100") == 0) {
            run_x100 = true;
        } else if (std::strcmp(argv[i], "--bench-mmid") == 0) {
            run_mmid = true;
        } else if (std::strcmp(argv[i], "--bench-pairwait") == 0) {
            run_pairwait = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--expect-device | --expect-none] [--full] [--bench | --bench-attn | --bench-mm | "
                         "--bench-f32 | --bench-embd | --bench-repack | --bench-x100 | --bench-mmid | "
                         "--bench-pairwait]\n"
                         "  --bench-attn: flash attention only\n"
                         "  --bench-f32: the F32 matmul only\n"
                         "  --bench-embd: the token embedding lookup only, ggml's dequantizers against an F32 copy\n"
                         "  --bench-repack: the weights' repack at load only, on the build's route\n"
                         "  --bench-x100: the 1-row matmul beside readers on the X100 cores, and split between the "
                         "provider and the CPU backend\n"
                         "  --bench-mmid: one MUL_MAT_ID node of Q4_0 experts, FLAGOS_SPACEMIT_EXPERT_Q4_0=plain against "
                         "the default\n"
                         "  --bench-pairwait: the pair wait against SpacemiT's barrier, without and with our exits\n",
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
        if (run_x100) {
            std::printf("x100bench: no SpacemiT device, skipped\n");
        }
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
        check_mul_mat(dev, full) != 0 || check_repack_threads(dev, !host) != 0 || check_gemv(dev, host) != 0 || check_mm_paths(dev, host) != 0 ||
        check_rows(dev) != 0 || check_embedding(dev) != 0 || check_cache_ops(dev) != 0 ||
        check_state_ops(dev) != 0 || check_mul_mat_id(dev, host, full) != 0 || check_gdn(dev) != 0 ||
        check_state_pass(dev) != 0 || check_seq_ops(dev) != 0 || check_mul_mat_f32_split(dev) != 0 ||
        check_router_q8(dev) != 0 || check_mul_mat_f16(dev, host) != 0 || check_conv_pair(dev) != 0 || check_moe_ops(dev) != 0 ||
        check_flash_attn(dev, host) != 0 ||
        check_layer(dev) != 0 || check_plan(dev) != 0 || check_diagnostics(dev) != 0 ||
        check_pairwait(dev, host) != 0 || check_shared_stream(dev, host) != 0) {
        return 1;
    }
    if (run_bench ? bench(dev) != 0 :
                    (run_attn && bench_attn(dev) != 0) ||
                        (run_mm && bench_mm(dev) != 0) || (run_f32 && bench_mm_f32(dev) != 0) ||
                        (run_embd && bench_embd(dev) != 0) ||
                        (run_repack && bench_repack(dev, !host) != 0) || (run_x100 && bench_x100(dev) != 0) ||
                        (run_mmid && bench_mmid(dev) != 0) || (run_pairwait && bench_pairwait(dev) != 0)) {
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "flagos-check-spacemit: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("flagos-check-spacemit: all checks passed\n");
    return 0;
}
