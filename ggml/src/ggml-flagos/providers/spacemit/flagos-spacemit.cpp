// FlagOS provider for the SpacemiT K3 AI cores (A100, CPUs 8-15).
// One IGPU device (D3, 2026-10-26; ACCEL before) with a buffer type and a backend; ops run on the AI cores through
// spine-runtime (M2a: ADD; M2b, M2c: Q4_0, Q4_1, Q8_0 and Q6_K MUL_MAT on the IME, weights repacked in the buffer, Q6_K
// requantized to Q8_0; M2d: the other ops of a layer except attention).

#include "flagos-spacemit-api.h"
#include "flagos-spacemit-attn.h"
#include "flagos-spacemit-exec.h"
#include "flagos-spacemit-ime.h"
#include "flagos-spacemit-ops.h"
#include "core/flagos-spacemit-loader.h"

#include "../../../ggml-backend-impl.h"
#include "../../../ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr uint64_t SPACEMIT_PROVIDER_ID       = 0x53504D5400000001ULL;  // "SPMT", 1
constexpr uint64_t SPACEMIT_DEVICE_UUID_HI    = 0x53504D544B330000ULL;  // "SPMTK3"
constexpr uint64_t SPACEMIT_DEVICE_UUID_LO    = 0x0000000041000002ULL;
constexpr uint64_t SPACEMIT_MEMORY_DOMAIN_ID  = 0x53504D544B330001ULL;  // K3 DRAM, shared with the X100 cores
constexpr size_t   SPACEMIT_BUFFER_ALIGNMENT  = 64;
constexpr uint32_t SPACEMIT_A100_VECTOR_BITS  = 1024;

struct spacemit_device_context {
    std::string                name;
    std::string                description;  // from the plugin's props
    std::string                runtime;      // from the plugin's props (the FlagOS profile)
    uint32_t                   n_ai_cores = 0;
    spacemit_plugin *          plugin     = nullptr;  // kept for the process lifetime, as ggml keeps its backends
    const spacemit_plugin_v1 * table      = nullptr;
    spacemit_plugin_device *   dev        = nullptr;
    ggml_backend_device        device     = {};
    ggml_backend_buffer_type   buffer_type = {};
    std::mutex                 run_mutex;  // backends of this device share the same AI cores
};

struct spacemit_buffer_context {
    const spacemit_plugin_v1 * table = nullptr;
    spacemit_plugin_buffer *   buf   = nullptr;  // the plugin's buffer
    void *                     data  = nullptr;  // its base
    size_t                     size  = 0;
};

struct spacemit_backend_context {
    const spacemit_plugin_v1 *        table  = nullptr;
    spacemit_plugin_stream *          stream = nullptr;
    spacemit_run_stats_attach_fn      attach = nullptr;  // test-only (E3c), nullptr when the plugin has no counters
    std::vector<spacemit_plugin_step> steps;
    void *                            workspace      = nullptr;  // shared by the tiles; grows to the largest split
    size_t                            workspace_size = 0;
    spacemit_run_stats                run_stats;  // test-only (E3c): the current graph's, with FLAGOS_SPACEMIT_TEST_STATS

    ~spacemit_backend_context() {
        if (stream != nullptr) {
            table->stream_destroy(stream);
        }
        if (workspace != nullptr) {
            ggml_aligned_free(workspace, workspace_size);
        }
    }
};

std::mutex              g_mutex;
bool                    g_probed  = false;
bool                    g_present = false;
spacemit_device_context g_device;

ggml_guid g_backend_guid = { 0x46, 0x6c, 0x61, 0x67, 0x4f, 0x53, 0x2d, 0x53, 0x50, 0x4d, 0x54, 0x00, 0x00, 0x00, 0x00, 0x01 };

//
// buffer: the plugin's memory (host memory on huge pages where the system has them, else 64-byte aligned); Q4_0, Q4_1,
// Q8_0 and Q6_K matmul weights are stored in IME layouts (flagos-spacemit-weights.h), a Q6_K weight in more than
// ggml_nbytes (get_alloc_size). set, get, memset and cpy work in ggml's layout within ggml_nbytes; only the repack and
// its inverse touch the stored bytes beyond it, and clear covers the whole buffer.
//

// a status the buffer functions cannot report to ggml, which expects them to succeed
void spacemit_buffer_status(const char * what, enum spacemit_status status) {
    if (status != SPACEMIT_STATUS_SUCCESS) {
        GGML_LOG_ERROR("FlagOS SpacemiT: %s failed: %s\n", what, spacemit_status_name(status));
    }
}

void spacemit_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    ctx->table->buffer_free(ctx->buf);
    delete ctx;
}

void * spacemit_buffer_base(ggml_backend_buffer_t buffer) {
    return static_cast<spacemit_buffer_context *>(buffer->context)->data;
}

void spacemit_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset,
                                   size_t size) {
    auto * ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    spacemit_buffer_status("memset_tensor", ctx->table->tensor_fill(tensor, value, offset, size));
}

// repacks; reads undo the repack (Q4_1: the converted weights; Q6_K: the nearest Q6_K), so the scheduler can still copy
// a weight out if the provider refuses an op on it
void spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset,
                                size_t size) {
    auto * ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    spacemit_buffer_status("set_tensor", ctx->table->tensor_set(tensor, data, offset, size));
}

void spacemit_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset,
                                size_t size) {
    auto * ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    spacemit_buffer_status("get_tensor", ctx->table->tensor_get(tensor, data, offset, size));
}

// a raw copy would skip the repack; returning false makes ggml use set_tensor instead
bool spacemit_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    auto * ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    if (src->buffer == nullptr || !ggml_backend_buffer_is_host(src->buffer) || ctx->table->tensor_layout(dst) != 0) {
        return false;
    }
    std::memcpy(dst->data, src->data, ggml_nbytes(src));
    return true;
}

void spacemit_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    std::memset(ctx->data, value, ctx->size);
}

const ggml_backend_buffer_i g_buffer_iface = {
    /* .free_buffer   = */ spacemit_buffer_free,
    /* .get_base      = */ spacemit_buffer_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ spacemit_buffer_memset_tensor,
    /* .set_tensor    = */ spacemit_buffer_set_tensor,
    /* .get_tensor    = */ spacemit_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ spacemit_buffer_cpy_tensor,
    /* .clear         = */ spacemit_buffer_clear,
    /* .reset         = */ nullptr,
};

const char * spacemit_buffer_type_name(ggml_backend_buffer_type_t) {
    return "FlagOS:SpacemiT";
}

// ggml_backend_buft_alloc_buffer() handles size 0 itself and never calls this with it. The plugin decides the pages
// (huge pages from 2 MiB; test-only: FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1 keeps the default pages, for comparison)
ggml_backend_buffer_t spacemit_buffer_type_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto *                   dev = static_cast<spacemit_device_context *>(buft->context);
    spacemit_plugin_buffer * buf = nullptr;
    const enum spacemit_status status = dev->table->buffer_alloc(dev->dev, size, &buf);
    if (status != SPACEMIT_STATUS_SUCCESS) {
        spacemit_buffer_status("alloc_buffer", status);
        return nullptr;
    }
    auto * ctx = new spacemit_buffer_context{ dev->table, buf, dev->table->buffer_base(buf), size };
    return ggml_backend_buffer_init(buft, g_buffer_iface, ctx, size);
}

size_t spacemit_buffer_type_alignment(ggml_backend_buffer_type_t) {
    return SPACEMIT_BUFFER_ALIGNMENT;
}

// a Q6_K weight is stored requantized to Q8_0: 34 bytes per 32 weights instead of 26.25
size_t spacemit_buffer_type_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return static_cast<spacemit_device_context *>(buft->context)->table->tensor_alloc_size(tensor);
}

const ggml_backend_buffer_type_i g_buffer_type_iface = {
    /* .get_name       = */ spacemit_buffer_type_name,
    /* .alloc_buffer   = */ spacemit_buffer_type_alloc,
    /* .get_alignment  = */ spacemit_buffer_type_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ spacemit_buffer_type_alloc_size,
    // repacked weights are not in ggml's layout, so the CPU backend must not read this buffer directly
    /* .is_host        = */ [](ggml_backend_buffer_type_t) { return false; },
};

//
// backend
//

const char * spacemit_backend_name(ggml_backend_t backend) {
    return static_cast<spacemit_device_context *>(backend->device->context)->name.c_str();
}

void spacemit_backend_free(ggml_backend_t backend) {
    delete static_cast<spacemit_backend_context *>(backend->context);
    delete backend;
}

bool spacemit_data_ready(const ggml_tensor * node) {
    if (node->data == nullptr) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] != nullptr && node->src[i]->data == nullptr) {
            return false;
        }
    }
    return true;
}

enum class spacemit_launch_kind {
    per_split,      // the default: a split's steps in one launch, a barrier after each
    per_step,       // E3a: every step its own launch, the host waiting for each
    per_step_join,  // E3b: every step its own launch, ordered inside the cores, the host waiting once
};

// test-only: FLAGOS_SPACEMIT_TEST_LAUNCH=per-step or per-step-join launches every step on its own, as a runtime
// without graph capture launches each kernel (structural-experiments.md E3a, E3b); per-split names the default. Read
// at every graph, so a benchmark can switch; anything else is reported once and ignored
spacemit_launch_kind spacemit_launch_mode() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_TEST_LAUNCH");
    if (v == nullptr || *v == '\0' || std::strcmp(v, "per-split") == 0) {
        return spacemit_launch_kind::per_split;
    }
    if (std::strcmp(v, "per-step") == 0) {
        return spacemit_launch_kind::per_step;
    }
    if (std::strcmp(v, "per-step-join") == 0) {
        return spacemit_launch_kind::per_step_join;
    }
    static std::atomic<bool> warned{ false };
    if (!warned.exchange(true)) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_TEST_LAUNCH=%s: unknown setting ignored\n", __func__, v);
    }
    return spacemit_launch_kind::per_split;
}

// test-only until adopted (fusion-and-splitting.md §5, candidate 6): FLAGOS_SPACEMIT_QUANTIZE=shared leaves out a
// MUL_MAT's quantize step when the matmul that last quantized took the same activations into the same layout (Q, V and
// K of a layer; gate and up), so it multiplies what that step wrote; separate (the default) quantizes for every matmul.
// The same bytes either way. Read at every graph, so a benchmark can switch; anything else is reported once and ignored
bool spacemit_quantize_shared() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_QUANTIZE");
    if (v == nullptr || *v == '\0' || std::strcmp(v, "separate") == 0) {
        return false;
    }
    if (std::strcmp(v, "shared") == 0) {
        return true;
    }
    static std::atomic<bool> warned{ false };
    if (!warned.exchange(true)) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_QUANTIZE=%s: unknown setting ignored\n", __func__, v);
    }
    return false;
}

// whether node's output overlaps t's data
bool spacemit_overlaps(const ggml_tensor * node, const ggml_tensor * t) {
    const char * a = static_cast<const char *>(node->data);
    const char * b = static_cast<const char *>(t->data);
    return a < b + ggml_nbytes(t) && b < a + ggml_nbytes(node);
}

// test-only (structural-experiments.md E3c): FLAGOS_SPACEMIT_TEST_STATS=<file> counts, per kind of graph (launch mode,
// rows of its first matmul, KV length class of its first attention: 0 below 1024, 1024 below 4096, else 4096), the
// graphs, steps and launches and the time spent in graph_compute, in the launches as the host waits for them, in the
// kernels (summed over the tiles; per step the slowest tile, "busy") and at the barriers (summed over the tiles). When
// the process ends it appends one JSON line per kind to <file>. Read once; one launch per step joined in the cores
// (E3b) is not counted
class spacemit_stats_file {
  public:
    // nullptr when the switch is not set
    static spacemit_stats_file * get() {
        static spacemit_stats_file file(std::getenv("FLAGOS_SPACEMIT_TEST_STATS"));
        return file.path.empty() ? nullptr : &file;
    }

    void add(const char * launch, int64_t rows, int64_t n_kv, const spacemit_run_stats & run, double graph_us) {
        double kernel_ns = 0.0, busy_ns = 0.0, barrier_ns = 0.0;
        for (size_t s = 0; s < run.n_steps; s++) {
            uint32_t slowest = 0;
            for (uint32_t t = 0; t < run.n_tiles; t++) {
                const uint32_t ns = run.kernel_ns[t * run.stride + s];
                kernel_ns += ns;
                slowest = std::max(slowest, ns);
            }
            busy_ns += slowest;
        }
        for (uint32_t t = 0; t < run.n_tiles; t++) {
            barrier_ns += static_cast<double>(run.barrier_ns[t]);
        }
        const int64_t               kv_class = n_kv < 0 ? -1 : n_kv < 1024 ? 0 : n_kv < 4096 ? 1024 : 4096;
        std::lock_guard<std::mutex> lock(mutex);
        totals & k = kinds[std::make_tuple(std::string(launch), rows, kv_class)];
        k.graphs++;
        k.steps += run.n_steps;
        k.launches += run.launches;
        k.tiles = run.n_tiles;
        k.graph_us += graph_us;
        k.launch_us += run.launch_us;
        k.kernel_us += kernel_ns / 1e3;
        k.busy_us += busy_ns / 1e3;
        k.barrier_us += barrier_ns / 1e3;
    }

    ~spacemit_stats_file() {
        if (path.empty() || kinds.empty()) {
            return;
        }
        FILE * f = std::fopen(path.c_str(), "a");
        if (f == nullptr) {
            GGML_LOG_ERROR("FlagOS SpacemiT: cannot write the test statistics to %s\n", path.c_str());
            return;
        }
        for (const auto & [key, k] : kinds) {
            std::fprintf(f,
                         "{\"launch\":\"%s\",\"rows\":%lld,\"kv\":%lld,\"graphs\":%llu,\"steps\":%llu,\"launches\":%llu,"
                         "\"tiles\":%u,\"graph_us\":%.3f,\"launch_us\":%.3f,\"kernel_us\":%.3f,\"busy_us\":%.3f,"
                         "\"barrier_us\":%.3f}\n",
                         std::get<0>(key).c_str(), (long long) std::get<1>(key), (long long) std::get<2>(key),
                         (unsigned long long) k.graphs, (unsigned long long) k.steps, (unsigned long long) k.launches,
                         k.tiles, k.graph_us, k.launch_us, k.kernel_us, k.busy_us, k.barrier_us);
        }
        std::fclose(f);
    }

  private:
    explicit spacemit_stats_file(const char * p) : path(p != nullptr ? p : "") {
        if (!path.empty()) {
            GGML_LOG_WARN("FlagOS SpacemiT: test statistics (FLAGOS_SPACEMIT_TEST_STATS) go to %s at exit\n", p);
        }
    }

    struct totals {
        uint64_t graphs = 0, steps = 0, launches = 0;
        uint32_t tiles  = 0;
        double   graph_us = 0.0, launch_us = 0.0, kernel_us = 0.0, busy_us = 0.0, barrier_us = 0.0;
    };

    std::string                                                     path;
    std::mutex                                                      mutex;
    std::map<std::tuple<std::string, int64_t, int64_t>, totals> kinds;
};

// the role in a tensor's name, for the op counters: blk.12.attn_q.weight -> attn_q, cache_k_l12 -> cache_k; "-" without
std::string spacemit_role(const ggml_tensor * t) {
    if (t == nullptr || t->name[0] == '\0') {
        return "-";
    }
    std::string n = t->name;
    n             = n.substr(0, n.find(' '));  // ggml's " (view)" and the like
    if (n.compare(0, 4, "blk.") == 0 && n.find('.', 4) != std::string::npos) {
        n = n.substr(n.find('.', 4) + 1);
    }
    const std::string weight = ".weight";
    if (n.size() > weight.size() && n.compare(n.size() - weight.size(), weight.size(), weight) == 0) {
        n.resize(n.size() - weight.size());
    }
    const size_t l = n.rfind("_l");
    if (l != std::string::npos && l + 2 < n.size() && n.find_first_not_of("0123456789", l + 2) == std::string::npos) {
        n.resize(l);
    }
    for (char & c : n) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '.' && c != '-') {
            c = '_';  // the name goes into JSON as it is
        }
    }
    return n.empty() ? "-" : n;
}

// a step's name in the op counters: the op (and the step of a MUL_MAT or a split attention), the role its tensors'
// names give, and its rows (r: the output's; m: a matmul's activations', after its weight type)
std::string spacemit_step_label(const std::vector<spacemit_plugin_step> & steps, size_t i) {
    const spacemit_plugin_step & step = steps[i];
    const ggml_tensor *          n    = step.node;
    const long long              r    = static_cast<long long>(ggml_nrows(n));
    char                         buf[192];
    switch (n->op) {
        case GGML_OP_MUL_MAT:
            std::snprintf(buf, sizeof(buf), "MUL_MAT.%s %s %s m%lld",
                          step.fn == spacemit_mul_mat_quantize ? "quantize" : "gemm", spacemit_role(n->src[0]).c_str(),
                          ggml_type_name(n->src[0]->type), static_cast<long long>(ggml_nrows(n->src[1])));
            break;
        case GGML_OP_RMS_NORM:
            {
                std::string role = "-";  // the norm weight of the MUL that takes it
                for (size_t j = i + 1; j < steps.size() && j <= i + 8; j++) {
                    if (steps[j].node->op == GGML_OP_MUL && steps[j].node->src[0] == n) {
                        role = spacemit_role(steps[j].node->src[1]);
                        break;
                    }
                }
                std::snprintf(buf, sizeof(buf), "RMS_NORM %s r%lld", role.c_str(), r);
                break;
            }
        case GGML_OP_MUL:
            std::snprintf(buf, sizeof(buf), "MUL %s r%lld", spacemit_role(n->src[1]).c_str(), r);
            break;
        case GGML_OP_ROPE:  // the norm weight of the MUL before it
            std::snprintf(buf, sizeof(buf), "ROPE %s r%lld",
                          n->src[0]->op == GGML_OP_MUL ? spacemit_role(n->src[0]->src[1]).c_str() : "-", r);
            break;
        case GGML_OP_SET_ROWS:  // the cache it writes
            std::snprintf(buf, sizeof(buf), "SET_ROWS %s r%lld", spacemit_role(n->src[2]).c_str(),
                          static_cast<long long>(ggml_nrows(n->src[0])));
            break;
        case GGML_OP_ADD:
            {
                std::string role = "-";  // the weight of the matmul whose output it adds
                for (int k = 0; k < 2 && role == "-"; k++) {
                    const ggml_tensor * s = n->src[k];
                    if (s->op == GGML_OP_GET_ROWS) {
                        s = s->src[0];
                    }
                    if (s->op == GGML_OP_MUL_MAT) {
                        role = "after " + spacemit_role(s->src[0]);
                    }
                }
                std::snprintf(buf, sizeof(buf), "ADD %s r%lld", role.c_str(), r);
                break;
            }
        case GGML_OP_GLU:
            std::snprintf(buf, sizeof(buf), "GLU %s r%lld", ggml_glu_op_name(ggml_get_glu_op(n)), r);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            std::snprintf(buf, sizeof(buf), "FLASH_ATTN_EXT.%s q%lld kv%lld",
                          step.fn == spacemit_kernel_flash_attn                 ? "one-step" :
                          step.fn == spacemit_kernel_flash_attn_split_merge     ? "merge" :
                                                                                  "partial",
                          static_cast<long long>(n->src[0]->ne[1]), static_cast<long long>(n->src[1]->ne[1]));
            break;
        default:
            std::snprintf(buf, sizeof(buf), "%s r%lld", ggml_op_name(n->op), r);
            break;
    }
    return buf;
}

// test-only (fusion-and-splitting.md §7): FLAGOS_SPACEMIT_TEST_OP_STATS=<file> times the steps by what they do, from
// E3c's clock reads (spacemit_run_stats). Per kind of graph (launch mode, rows of its first matmul, KV length of its
// first attention, and the kernels and shapes of its steps in order) it sums each step's times over the graphs, leaving
// out the kind's first graph (it may create the stream and touch new memory), and names each step when the kind is first
// seen (spacemit_step_label). When the process ends it appends to <file> one JSON line per kind and name, and one for
// all of the kind's steps, all per graph: the steps, the mean tile's kernel time (kernel_us) and its kernel and barrier
// time (wall_us), each step's slowest tile summed (busy_us), each tile's kernel time (tile_us) and how often each was
// a step's slowest (slowest, over all graphs). Read once; E3b's launches (per-step-join) are not counted
class spacemit_op_stats_file {
  public:
    // nullptr when the switch is not set
    static spacemit_op_stats_file * get() {
        static spacemit_op_stats_file file(std::getenv("FLAGOS_SPACEMIT_TEST_OP_STATS"));
        return file.path.empty() ? nullptr : &file;
    }

    void add(const char * launch, int64_t rows, int64_t n_kv, const std::vector<spacemit_plugin_step> & steps,
             const spacemit_run_stats & run) {
        const size_t   n     = steps.size();
        const uint32_t tiles = run.n_tiles;
        if (run.n_steps != n || tiles == 0) {
            return;
        }
        const uint64_t              shape = structure_of(steps);
        std::lock_guard<std::mutex> lock(mutex);
        totals &                    k = kinds[std::make_tuple(std::string(launch), rows, n_kv, shape)];
        if (k.labels.empty()) {  // the kind's first graph: its names only
            k.tiles = tiles;
            for (size_t s = 0; s < n; s++) {
                k.labels.push_back(spacemit_step_label(steps, s));
            }
            k.kernel_ns.assign(n, 0.0);
            k.busy_ns.assign(n, 0.0);
            k.wall_ns.assign(n, 0.0);
            k.tile_ns.assign(n * tiles, 0.0);
            k.slowest.assign(n * tiles, 0);
            return;
        }
        if (k.tiles != tiles) {
            return;
        }
        k.graphs++;
        k.launch_us += run.launch_us;
        for (size_t s = 0; s < n; s++) {
            uint32_t slowest = 0, at = 0;
            for (uint32_t t = 0; t < tiles; t++) {
                const uint32_t kernel = run.kernel_ns[t * run.stride + s];
                k.kernel_ns[s] += kernel;
                k.wall_ns[s] += static_cast<double>(kernel) + run.step_barrier_ns[t * run.stride + s];
                k.tile_ns[s * tiles + t] += kernel;
                if (kernel > slowest) {
                    slowest = kernel;
                    at      = t;
                }
            }
            k.busy_ns[s] += slowest;
            k.slowest[s * tiles + at]++;
        }
    }

    ~spacemit_op_stats_file() {
        if (path.empty() || kinds.empty()) {
            return;
        }
        FILE * f = std::fopen(path.c_str(), "a");
        if (f == nullptr) {
            GGML_LOG_ERROR("FlagOS SpacemiT: cannot write the op counters to %s\n", path.c_str());
            return;
        }
        for (const auto & [key, k] : kinds) {
            // each name's steps, the names in the order of their first step; then all steps
            std::vector<std::string>                    names;
            std::map<std::string, std::vector<size_t>> steps_of;
            std::vector<size_t>                         all(k.labels.size());
            for (size_t s = 0; s < k.labels.size(); s++) {
                std::vector<size_t> & of = steps_of[k.labels[s]];
                if (of.empty()) {
                    names.push_back(k.labels[s]);
                }
                of.push_back(s);
                all[s] = s;
            }
            names.push_back("(all)");
            steps_of["(all)"] = all;
            const double per = 1e3 * static_cast<double>(std::max<uint64_t>(k.graphs, 1));  // ns summed -> us per graph
            for (const std::string & name : names) {
                double              kernel = 0.0, busy = 0.0, wall = 0.0;
                std::vector<double> tile(k.tiles, 0.0);
                std::vector<uint64_t> slowest(k.tiles, 0);
                for (size_t s : steps_of[name]) {
                    kernel += k.kernel_ns[s];
                    busy += k.busy_ns[s];
                    wall += k.wall_ns[s];
                    for (uint32_t t = 0; t < k.tiles; t++) {
                        tile[t] += k.tile_ns[s * k.tiles + t];
                        slowest[t] += k.slowest[s * k.tiles + t];
                    }
                }
                std::string tile_us, slowest_n;
                for (uint32_t t = 0; t < k.tiles; t++) {
                    char v[48];
                    std::snprintf(v, sizeof(v), "%s%.3f", t > 0 ? "," : "", tile[t] / per);
                    tile_us += v;
                    std::snprintf(v, sizeof(v), "%s%llu", t > 0 ? "," : "", static_cast<unsigned long long>(slowest[t]));
                    slowest_n += v;
                }
                std::fprintf(f,
                             "{\"launch\":\"%s\",\"rows\":%lld,\"n_kv\":%lld,\"structure\":\"%016llx\",\"steps\":%zu,"
                             "\"tiles\":%u,\"graphs\":%llu,\"label\":\"%s\",\"n\":%zu,\"wall_us\":%.3f,\"kernel_us\":%.3f,"
                             "\"busy_us\":%.3f,\"launch_us\":%.3f,\"tile_us\":[%s],\"slowest\":[%s]}\n",
                             std::get<0>(key).c_str(), static_cast<long long>(std::get<1>(key)),
                             static_cast<long long>(std::get<2>(key)), static_cast<unsigned long long>(std::get<3>(key)),
                             k.labels.size(), k.tiles, static_cast<unsigned long long>(k.graphs), name.c_str(),
                             steps_of[name].size(), wall / k.tiles / per, kernel / k.tiles / per, busy / per,
                             name == "(all)" ? 1e3 * k.launch_us / per : 0.0, tile_us.c_str(), slowest_n.c_str());
            }
        }
        std::fclose(f);
    }

  private:
    explicit spacemit_op_stats_file(const char * p) : path(p != nullptr ? p : "") {
        if (!path.empty()) {
            GGML_LOG_WARN("FlagOS SpacemiT: op counters (FLAGOS_SPACEMIT_TEST_OP_STATS) go to %s at exit\n", p);
        }
    }

    // what tells two graphs' steps apart: kernels, ops and shapes, not addresses, which may change when llama.cpp
    // builds the same graph again
    static uint64_t structure_of(const std::vector<spacemit_plugin_step> & steps) {
        uint64_t   h   = 0xcbf29ce484222325ULL;
        const auto mix = [&h](uint64_t v) { h = (h ^ v) * 0x100000001b3ULL; };
        for (const spacemit_plugin_step & s : steps) {
            mix(reinterpret_cast<uintptr_t>(s.fn));
            mix(static_cast<uint64_t>(s.node->op));
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                mix(static_cast<uint64_t>(s.node->ne[d]));
            }
            if (s.node->src[0] != nullptr) {
                mix(static_cast<uint64_t>(s.node->src[0]->type));
                mix(static_cast<uint64_t>(s.node->src[0]->ne[0]));
            }
        }
        return h;
    }

    struct totals {
        uint32_t                 tiles     = 0;
        uint64_t                 graphs    = 0;  // counted ones: all but the first
        double                   launch_us = 0.0;
        std::vector<std::string> labels;
        std::vector<double>      kernel_ns, busy_ns, wall_ns;  // per step, summed over the tiles (busy: the slowest)
        std::vector<double>      tile_ns;                      // [step * tiles + tile]
        std::vector<uint64_t>    slowest;                      // [step * tiles + tile]: graphs where it was slowest
    };

    std::string                                                                path;
    std::mutex                                                                 mutex;
    std::map<std::tuple<std::string, int64_t, int64_t, uint64_t>, totals> kinds;
};

// every node and the workspace are checked before the launch, so a graph either runs completely or fails with
// nothing written
ggml_status spacemit_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = static_cast<spacemit_backend_context *>(backend->context);
    auto * dev = static_cast<spacemit_device_context *>(backend->device->context);

    spacemit_stats_file * const    stats    = spacemit_stats_file::get();     // test-only (E3c)
    spacemit_op_stats_file * const op_stats = spacemit_op_stats_file::get();  // test-only (fusion-and-splitting.md §7)
    const bool                     timed    = stats != nullptr || op_stats != nullptr;
    const std::chrono::steady_clock::time_point start =
        timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    int64_t rows = -1, n_kv = -1;  // the counters' kind of graph: its first matmul's rows, its first attention's KV length

    // a MUL_MAT quantizes its activations in step 0 and multiplies them in step 1 (spacemit_find_op). They have their
    // own part of the workspace, after the other ops' scratch, so that the ops between two matmuls of one input leave
    // them, and with FLAGOS_SPACEMIT_QUANTIZE=shared the second matmul runs only its step 1
    const bool          share = spacemit_quantize_shared();
    const ggml_tensor * held  = nullptr;  // the MUL_MAT whose quantized activations that part holds
    size_t              scratch_size = 0, act_size = 0;
    ctx->steps.clear();
    for (int i = 0; i < ggml_graph_n_nodes(cgraph); i++) {
        ggml_tensor * node = ggml_graph_node(cgraph, i);
        // as ggml-cpu: nothing to do for empty tensors, and nodes without the compute flag are branches the graph
        // left unselected (ggml_build_forward_select), whose inputs may never have been set
        if (ggml_op_is_empty(node->op) || ggml_is_empty(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (timed && rows < 0 && node->op == GGML_OP_MUL_MAT) {
            rows = node->src[1]->ne[1];
        }
        if (timed && n_kv < 0 && node->op == GGML_OP_FLASH_ATTN_EXT) {
            n_kv = node->src[1]->ne[1];
        }
        const spacemit_op * op = spacemit_find_op(node, &dev->buffer_type);
        if (op == nullptr) {
            GGML_LOG_ERROR("FlagOS SpacemiT: no kernel for %s (%s)\n", ggml_op_desc(node), node->name);
            return GGML_STATUS_FAILED;
        }
        if (!spacemit_data_ready(node)) {
            GGML_LOG_ERROR("FlagOS SpacemiT: %s (%s) has an unallocated operand\n", ggml_op_desc(node), node->name);
            return GGML_STATUS_FAILED;
        }
        uint32_t first = 0;
        if (share) {
            if (held != nullptr && spacemit_overlaps(node, held->src[1])) {
                held = nullptr;  // the node writes over the activations held
            }
            if (node->op == GGML_OP_MUL_MAT) {
                if (held != nullptr && spacemit_mul_mat_same_activations(held, node)) {
                    first = 1;
                } else {
                    held = node;
                }
            }
        }
        for (uint32_t s = first; s < op->n_steps; s++) {
            ctx->steps.push_back({ op->steps[s], node, 0 });
        }
        if (op->workspace_size != nullptr) {
            size_t & size = node->op == GGML_OP_MUL_MAT ? act_size : scratch_size;
            size          = std::max(size, op->workspace_size(node));
        }
    }
    if (ctx->steps.empty()) {
        return GGML_STATUS_SUCCESS;
    }
    const size_t act_offset     = GGML_PAD(scratch_size, 64);
    const size_t workspace_size = act_offset + act_size;
    for (spacemit_plugin_step & step : ctx->steps) {
        step.workspace_offset = step.node->op == GGML_OP_MUL_MAT ? act_offset : 0;
    }
    if (workspace_size > ctx->workspace_size) {
        if (ctx->workspace != nullptr) {
            ggml_aligned_free(ctx->workspace, ctx->workspace_size);
        }
        ctx->workspace      = ggml_aligned_malloc(workspace_size);
        ctx->workspace_size = ctx->workspace != nullptr ? workspace_size : 0;
        if (ctx->workspace == nullptr) {
            GGML_LOG_ERROR("FlagOS SpacemiT: failed to allocate a %zu-byte workspace\n", workspace_size);
            return GGML_STATUS_FAILED;
        }
    }

    std::lock_guard<std::mutex> lock(dev->run_mutex);
    const spacemit_launch_kind  mode = spacemit_launch_mode();
    if (timed && ctx->attach == nullptr) {
        static std::atomic<bool> warned{ false };
        if (!warned.exchange(true)) {
            GGML_LOG_WARN("FlagOS SpacemiT: the plugin has no step counters; the test statistics stay off\n");
        }
    }
    const bool counted = timed && mode != spacemit_launch_kind::per_step_join && ctx->attach != nullptr;
    if (counted) {
        ctx->run_stats.reset(ctx->steps.size());
    }
    if (ctx->attach != nullptr) {
        ctx->attach(ctx->stream, counted ? &ctx->run_stats : nullptr);
    }
    // test-only: the two launch modes of E3a and E3b, one launch per step, are flags of the replay
    uint32_t flags = 0;
    if (mode == spacemit_launch_kind::per_step) {
        flags |= SPACEMIT_REPLAY_SYNC_LAUNCH;
    } else if (mode == spacemit_launch_kind::per_step_join) {
        flags |= SPACEMIT_REPLAY_ORDERED_LAUNCHES;
    }
    const bool ok = ctx->table->replay(ctx->stream, ctx->steps.data(), static_cast<uint32_t>(ctx->steps.size()),
                                       ctx->workspace, ctx->workspace_size, flags) == SPACEMIT_STATUS_SUCCESS;
    if (ctx->attach != nullptr) {
        ctx->attach(ctx->stream, nullptr);
    }
    if (ok && counted) {
        const char * launch   = mode == spacemit_launch_kind::per_step ? "per-step" : "per-split";
        const double graph_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        if (stats != nullptr) {
            stats->add(launch, rows, n_kv, ctx->run_stats, graph_us);
        }
        if (op_stats != nullptr) {
            op_stats->add(launch, rows, n_kv, ctx->steps, ctx->run_stats);
        }
    }
    return ok ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

const ggml_backend_i g_backend_iface = {
    /* .get_name            = */ spacemit_backend_name,
    /* .free                = */ spacemit_backend_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ nullptr,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ spacemit_backend_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

//
// device
//

spacemit_device_context * spacemit_device_from_dev(ggml_backend_dev_t dev) {
    return static_cast<spacemit_device_context *>(dev->context);
}

const char * spacemit_device_name(ggml_backend_dev_t dev) {
    return spacemit_device_from_dev(dev)->name.c_str();
}

const char * spacemit_device_description(ggml_backend_dev_t dev) {
    return spacemit_device_from_dev(dev)->description.c_str();
}

// the AI cores use the same DRAM as the CPU cores (the plugin reads it)
void spacemit_device_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * ctx = spacemit_device_from_dev(dev);
    *free      = 0;
    *total     = 0;
    spacemit_buffer_status("get_memory", ctx->table->device_memory(ctx->dev, free, total));
}

// IGPU (D3, 2026-10-26): llama.cpp gives the device whole layers and their KV cache, which lives in our buffer.
// test-only: FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=accel or gpu reports that type instead, for comparisons (ACCEL: the type
// until 2026-10-26, the KV cache in a CPU buffer; structural-experiments.md E1, E2); igpu names the default. Read once,
// reported once when it changes the type, anything else reported and ignored
enum ggml_backend_dev_type spacemit_device_type(ggml_backend_dev_t) {
    static const enum ggml_backend_dev_type type = [] {
        const char * v = std::getenv("FLAGOS_SPACEMIT_TEST_DEVICE_TYPE");
        if (v == nullptr || *v == '\0' || std::strcmp(v, "igpu") == 0) {
            return GGML_BACKEND_DEVICE_TYPE_IGPU;
        }
        if (std::strcmp(v, "accel") == 0 || std::strcmp(v, "gpu") == 0) {
            const bool accel = v[0] == 'a';
            GGML_LOG_WARN("spacemit_device_type: FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=%s: reporting %s instead of IGPU\n", v,
                          accel ? "ACCEL" : "GPU");
            return accel ? GGML_BACKEND_DEVICE_TYPE_ACCEL : GGML_BACKEND_DEVICE_TYPE_GPU;
        }
        GGML_LOG_WARN("spacemit_device_type: FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=%s: unknown setting ignored\n", v);
        return GGML_BACKEND_DEVICE_TYPE_IGPU;
    }();
    return type;
}

void spacemit_device_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = spacemit_device_name(dev);
    props->description = spacemit_device_description(dev);
    spacemit_device_memory(dev, &props->memory_free, &props->memory_total);
    props->type      = spacemit_device_type(dev);
    props->device_id = nullptr;
    props->caps      = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ true,
    };
}

// one stream per backend context; the plugin holds the core grant until it is destroyed (FLAGOS_SPACEMIT_STREAM is the
// plugin's switch now)
ggml_backend_t spacemit_device_init(ggml_backend_dev_t dev, const char *) {
    auto *                   dctx   = spacemit_device_from_dev(dev);
    spacemit_plugin_stream * stream = nullptr;
    const enum spacemit_status status = dctx->table->stream_create(dctx->dev, dctx->n_ai_cores, &stream);
    if (status != SPACEMIT_STATUS_SUCCESS) {
        GGML_LOG_ERROR("FlagOS SpacemiT: the plugin could not create a stream: %s\n", spacemit_status_name(status));
        return nullptr;
    }
    auto * ctx  = new spacemit_backend_context;
    ctx->table  = dctx->table;
    ctx->stream = stream;
    // test-only (E3c): the per-tile clock reads are not contract, so they come by name (plan.md "S1 design" A2)
    ctx->attach = reinterpret_cast<spacemit_run_stats_attach_fn>(dctx->table->get_proc_address("spacemit_run_stats_attach"));
    return new ggml_backend{
        /* .guid    = */ &g_backend_guid,
        /* .iface   = */ g_backend_iface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
}

ggml_backend_buffer_type_t spacemit_device_buffer_type(ggml_backend_dev_t dev) {
    return &spacemit_device_from_dev(dev)->buffer_type;
}

// inputs (NONE) and views need no kernel; claiming them lets graphs that contain them run here (design §7.4)
bool spacemit_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return ggml_op_is_empty(op->op) || spacemit_find_op(op, &spacemit_device_from_dev(dev)->buffer_type) != nullptr;
}

// operands may also live in CPU buffers: the activations of the CPU's splits, and under ACCEL (test-only) the KV cache;
// under IGPU, the default, the KV cache lives in our buffer
bool spacemit_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == &spacemit_device_from_dev(dev)->buffer_type || ggml_backend_buft_is_host(buft);
}

const ggml_backend_device_i g_device_iface = {
    /* .get_name             = */ spacemit_device_name,
    /* .get_description      = */ spacemit_device_description,
    /* .get_memory           = */ spacemit_device_memory,
    /* .get_type             = */ spacemit_device_type,
    /* .get_props            = */ spacemit_device_props,
    /* .init_backend         = */ spacemit_device_init,
    /* .get_buffer_type      = */ spacemit_device_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ spacemit_device_supports_op,
    /* .supports_buft        = */ spacemit_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

//
// probe: the plugin is loaded, checked and asked for its device; the K3 itself is detected inside the plugin (its A100
// cores in /proc/cpuinfo and the AI-thread gate)
//

void spacemit_host_log(int level, const char * text) {
    ggml_log_internal((enum ggml_log_level) level, "%s\n", text);
}

bool spacemit_probe_impl(ggml_backend_reg_t owner_reg) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_probed) {
        return g_present;
    }
    g_probed = true;

    // test switch: compare model output with and without the provider
    const char * disable = std::getenv("FLAGOS_SPACEMIT_DISABLE");
    if (disable != nullptr && std::strcmp(disable, "0") != 0) {
        GGML_LOG_INFO("FlagOS SpacemiT: provider disabled by FLAGOS_SPACEMIT_DISABLE\n");
        return false;
    }

    // the host table the loader hands to the entry, with the directory the plugin came from; static so the strings
    // outlive the call whatever a plugin does with them (ours copies what it keeps)
    static std::string      plugin_dir;
    static spacemit_host_v1 host;
    bool                    self = false;                               // the path is the core's own image (A5)
    const std::string       path = spacemit_plugin_default_path(&self);  // the same path the loader resolves
    const size_t            cut  = path.find_last_of('/');
    plugin_dir                   = cut != std::string::npos ? path.substr(0, cut) : std::string(".");
    host.struct_size             = sizeof(spacemit_host_v1);
    host.log                     = spacemit_host_log;
    host.plugin_dir              = plugin_dir.c_str();

    spacemit_plugin * plugin = nullptr;
    std::string       detail;
    const spacemit_load_result loaded = spacemit_plugin_load(nullptr, host, spacemit_required_kernels,
                                                             spacemit_n_required_kernels, &plugin, &detail);
    if (loaded != spacemit_load_result::ok) {
        if (loaded == spacemit_load_result::no_device) {
            // the ordinary case on a host without the AI cores
            GGML_LOG_DEBUG("FlagOS SpacemiT: no device: %s (%s)\n", detail.c_str(), path.c_str());
        } else if (loaded == spacemit_load_result::not_found && self) {
            // a statically linked executable cannot be dlopened: A5's documented consequence, not a fault
            GGML_LOG_DEBUG("FlagOS SpacemiT: no plugin in this image: %s (%s)\n", detail.c_str(), path.c_str());
        } else {
            GGML_LOG_WARN("FlagOS SpacemiT: the plugin %s was not loaded: %s\n", path.c_str(), detail.c_str());
        }
        return false;
    }

    const spacemit_plugin_v1 * table = spacemit_plugin_table(plugin);
    spacemit_plugin_device *   pdev  = nullptr;
    enum spacemit_status       status = table->device_open(0, &pdev);
    if (status != SPACEMIT_STATUS_SUCCESS) {
        GGML_LOG_WARN("FlagOS SpacemiT: the plugin's device 0 did not open: %s\n", spacemit_status_name(status));
        spacemit_plugin_unload(plugin);
        return false;
    }
    spacemit_plugin_device_props props = {};
    props.struct_size                  = sizeof(props);
    status                             = table->device_props(pdev, &props);
    if (status != SPACEMIT_STATUS_SUCCESS) {
        GGML_LOG_WARN("FlagOS SpacemiT: the plugin's device properties are not available: %s\n",
                      spacemit_status_name(status));
        table->device_close(pdev);
        spacemit_plugin_unload(plugin);
        return false;
    }

    g_device.name                 = "FlagOS:SpacemiT:0";
    g_device.description          = props.description != nullptr ? props.description : "";
    g_device.runtime              = props.runtime != nullptr ? props.runtime : "none";
    g_device.n_ai_cores           = props.n_tiles;
    g_device.plugin               = plugin;
    g_device.table                = table;
    g_device.dev                  = pdev;
    g_device.device.iface         = g_device_iface;
    g_device.device.reg           = owner_reg;
    g_device.device.context       = &g_device;
    g_device.buffer_type.iface    = g_buffer_type_iface;
    g_device.buffer_type.device   = &g_device.device;
    g_device.buffer_type.context  = &g_device;
    g_present = true;
    return true;
}

//
// flagos_provider_v1 callbacks
//

bool spacemit_probe(ggml_backend_reg_t owner_reg) {
    return spacemit_probe_impl(owner_reg);
}

size_t spacemit_device_count() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_present ? 1 : 0;
}

ggml_backend_dev_t spacemit_device_get(size_t index) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_present && index == 0 ? &g_device.device : nullptr;
}

bool spacemit_device_identity(size_t index, flagos_device_identity * identity) {
    if (identity == nullptr || index != 0) {
        return false;
    }
    *identity = {
        /* .provider_id      = */ SPACEMIT_PROVIDER_ID,
        /* .uuid_hi          = */ SPACEMIT_DEVICE_UUID_HI,
        /* .uuid_lo          = */ SPACEMIT_DEVICE_UUID_LO,
        /* .memory_domain_id = */ SPACEMIT_MEMORY_DOMAIN_ID,
        /* .ordinal          = */ 0,
    };
    return true;
}

bool spacemit_device_caps(size_t index, flagos_device_caps * caps) {
    if (caps == nullptr || index != 0) {
        return false;
    }
    *caps = {
        // proposed D8 (plan.md): the A100 cores run RISC-V code with RVV and IME
        /* .kind       = */ flagos_provider_kind::cpu_accelerator,
        /* .memory     = */ FLAGOS_MEMORY_HOST_VISIBLE | FLAGOS_MEMORY_UNIFIED_COHERENT,
        /* .execution  = */ 0,
        /* .aot_format = */ nullptr,
    };
    return true;
}

bool spacemit_device_profile(size_t index, flagos_device_profile * profile) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (profile == nullptr || index != 0 || !g_present) {
        return false;
    }
    *profile = {};
    profile->struct_size       = sizeof(flagos_device_profile);
    profile->version           = 1;
    profile->engine            = flagos_engine_kind::cpu;
    profile->features          = FLAGOS_FEATURE_SIMD | FLAGOS_FEATURE_FP16 | FLAGOS_FEATURE_INT8 |
                                 FLAGOS_FEATURE_MATRIX | FLAGOS_FEATURE_UNIFIED_MEMORY;
    profile->vector_bits       = SPACEMIT_A100_VECTOR_BITS;
    profile->lane_count        = SPACEMIT_A100_VECTOR_BITS / 32;
    profile->concurrency       = g_device.n_ai_cores;
    profile->memory_domain_id  = SPACEMIT_MEMORY_DOMAIN_ID;
    profile->vendor            = "SpacemiT";
    profile->architecture      = "a100";
    profile->microarchitecture = "rvv1024-ime2";
    profile->target            = "riscv64-a100-ime2";
    profile->runtime           = g_device.runtime.c_str();  // the plugin's props
    profile->aot_format        = nullptr;
    return flagos_device_profile_is_valid(profile);
}

bool spacemit_is_backend(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, &g_backend_guid);
}

bool spacemit_set_device(size_t index) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_present && index == 0;
}

int spacemit_get_device() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_present ? 0 : -1;
}

int spacemit_score() {
    return spacemit_probe_impl(nullptr) ? 1 : 0;
}

}  // namespace

const spacemit_plugin * spacemit_device_plugin(ggml_backend_dev_t dev) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return dev != nullptr && dev->context == &g_device ? g_device.plugin : nullptr;
}

const flagos_provider_v1 * flagos_spacemit_provider() {
    static const flagos_provider_v1 provider = {
        /* .api_version        = */ FLAGOS_PROVIDER_API_VERSION,
        /* .struct_size        = */ sizeof(flagos_provider_v1),
        /* .identity           = */ { SPACEMIT_PROVIDER_ID, "SpacemiT", 1 },
        /* .probe              = */ spacemit_probe,
        /* .device_count       = */ spacemit_device_count,
        /* .device_get         = */ spacemit_device_get,
        /* .device_identity    = */ spacemit_device_identity,
        /* .device_caps        = */ spacemit_device_caps,
        /* .is_backend         = */ spacemit_is_backend,
        /* .set_device         = */ spacemit_set_device,
        /* .get_device         = */ spacemit_get_device,
        /* .score              = */ spacemit_score,
        /* .get_device_profile = */ spacemit_device_profile,
    };
    return &provider;
}
