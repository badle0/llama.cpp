// The core's diagnostics (core/flagos-spacemit-diagnostics.h; plan.md "S1 step 5 spec"). The counter classes, the role
// and step labels and the launch and quantize switches moved here unchanged from flagos-spacemit.cpp 2026-11-02, under
// the deck's names.

#include "flagos-spacemit-diagnostics.h"

#include "../../../../ggml-impl.h"  // GGML_LOG_*, ggml_nrows

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

std::mutex g_diag_mutex;  // the parsed-once switches

std::atomic<bool> g_ops_parsed{ false };  // GGML_FLAGOS_DISABLE_OPS, parsed once (and again after a reset)
std::atomic<bool> g_ops_any{ false };     // whether any op is disabled, the fast path of spacemit_op_disabled
bool              g_ops_disabled[GGML_OP_COUNT] = {};

std::atomic<bool> g_trace_parsed{ false };  // GGML_FLAGOS_TRACE_LAUNCH, read once (and again after a reset)
bool              g_trace = false;

// the switch's value: a path, or stderr for - and 1 (the deck's STATS=1 prints counts; ours write JSON lines)
bool value_is_stderr(const char * value) {
    return value != nullptr && (std::strcmp(value, "-") == 0 || std::strcmp(value, "1") == 0);
}

// where the lines of a counter file go, and the name of that destination for the warning at start
FILE * counter_stream(bool to_stderr, const std::string & path, const char * what) {
    if (to_stderr) {
        return stderr;
    }
    FILE * f = std::fopen(path.c_str(), "a");
    if (f == nullptr) {
        GGML_LOG_ERROR("FlagOS SpacemiT: cannot write the %s to %s\n", what, path.c_str());
    }
    return f;
}

// GGML_FLAGOS_DISABLE_OPS: a comma list of ggml op names as ggml_op_name spells them (case as given, spaces
// trimmed); an unknown name is reported once and ignored. Called under g_diag_mutex
void parse_disabled_ops() {
    std::fill(std::begin(g_ops_disabled), std::end(g_ops_disabled), false);
    bool         any   = false;
    const char * value = std::getenv("GGML_FLAGOS_DISABLE_OPS");
    for (const char * p = value; p != nullptr && *p != '\0';) {
        const char * comma = std::strchr(p, ',');
        const char * end   = comma != nullptr ? comma : p + std::strlen(p);
        while (p < end && std::isspace(static_cast<unsigned char>(*p))) {
            p++;
        }
        while (end > p && std::isspace(static_cast<unsigned char>(end[-1]))) {
            end--;
        }
        const std::string name(p, static_cast<size_t>(end - p));
        if (!name.empty()) {
            bool found = false;
            for (int op = 0; op < GGML_OP_COUNT && !found; op++) {
                if (name == ggml_op_name(static_cast<enum ggml_op>(op))) {
                    g_ops_disabled[op] = true;
                    found = any = true;
                }
            }
            if (!found) {
                GGML_LOG_WARN("FlagOS SpacemiT: GGML_FLAGOS_DISABLE_OPS: %s is no ggml op name; ignored\n",
                              name.c_str());
            }
        }
        p = comma != nullptr ? comma + 1 : end;
    }
    if (any) {
        GGML_LOG_WARN("FlagOS SpacemiT: GGML_FLAGOS_DISABLE_OPS=%s: those ops stay on the CPU\n", value);
    }
    g_ops_any.store(any, std::memory_order_relaxed);
}

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

// a step's name in the op counters: the op (and the step of a MUL_MAT or a split attention, by the step's contract
// name), the role its tensors' names give, and its rows (r: the output's; m: a matmul's activations', after its weight
// type)
std::string spacemit_step_label(const std::vector<spacemit_plugin_step> & steps,
                                const std::vector<const char *> & names, size_t i) {
    const spacemit_plugin_step & step = steps[i];
    const char * const           name = names[i];
    const ggml_tensor *          n    = step.node;
    const long long              r    = static_cast<long long>(ggml_nrows(n));
    char                         buf[192];
    switch (n->op) {
        case GGML_OP_MUL_MAT:
            std::snprintf(buf, sizeof(buf), "MUL_MAT.%s %s %s m%lld",
                          std::strcmp(name, "mul_mat.quantize") == 0 ? "quantize" : "gemm",
                          spacemit_role(n->src[0]).c_str(), ggml_type_name(n->src[0]->type),
                          static_cast<long long>(ggml_nrows(n->src[1])));
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
        case GGML_OP_GET_ROWS:  // the lookup of quantized rows (the token embedding) by its weight and type
            if (std::strcmp(name, "get_rows.quant") == 0) {
                std::snprintf(buf, sizeof(buf), "GET_ROWS %s %s r%lld", spacemit_role(n->src[0]).c_str(),
                              ggml_type_name(n->src[0]->type), r);
            } else {
                std::snprintf(buf, sizeof(buf), "GET_ROWS r%lld", r);
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            std::snprintf(buf, sizeof(buf), "FLASH_ATTN_EXT.%s q%lld kv%lld",
                          std::strcmp(name, "flash_attn") == 0             ? "one-step" :
                          std::strcmp(name, "flash_attn.split.merge") == 0 ? "merge" :
                          std::strcmp(name, "flash_attn.kvh.partial") == 0 ? "kvh-partial" :
                          std::strcmp(name, "flash_attn.kvh.merge") == 0   ? "kvh-merge" :
                                                                             "partial",
                          static_cast<long long>(n->src[0]->ne[1]), static_cast<long long>(n->src[1]->ne[1]));
            break;
        default:
            std::snprintf(buf, sizeof(buf), "%s r%lld", ggml_op_name(n->op), r);
            break;
    }
    return buf;
}

// what tells two graphs' steps apart: kernels, ops and shapes, not addresses, which may change when llama.cpp
// builds the same graph again
uint64_t structure_of(const std::vector<spacemit_plugin_step> & steps) {
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

}  // namespace

void spacemit_diagnostics_reset() {
    std::lock_guard<std::mutex> lock(g_diag_mutex);
    g_ops_parsed.store(false, std::memory_order_release);
    g_trace_parsed.store(false, std::memory_order_release);
}

spacemit_launch_kind spacemit_launch_mode() {
    const char * v = std::getenv("GGML_FLAGOS_SYNC_LAUNCH");
    const bool   on = v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
    return on ? spacemit_launch_kind::per_step : spacemit_launch_kind::per_split;
}

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

bool spacemit_embd_on_cpu() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_EMBD");
    if (v == nullptr || *v == '\0') {
        return false;
    }
    if (std::strcmp(v, "cpu") == 0) {
        return true;
    }
    static std::atomic<bool> warned{ false };
    if (!warned.exchange(true)) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_EMBD=%s: unknown setting ignored\n", __func__, v);
    }
    return false;
}

bool spacemit_op_disabled(enum ggml_op op) {
    if (!g_ops_parsed.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(g_diag_mutex);
        if (!g_ops_parsed.load(std::memory_order_relaxed)) {
            parse_disabled_ops();
            g_ops_parsed.store(true, std::memory_order_release);
        }
    }
    return g_ops_any.load(std::memory_order_relaxed) && op >= 0 && op < GGML_OP_COUNT && g_ops_disabled[op];
}

bool spacemit_trace_launch() {
    if (!g_trace_parsed.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(g_diag_mutex);
        if (!g_trace_parsed.load(std::memory_order_relaxed)) {
            const char * v = std::getenv("GGML_FLAGOS_TRACE_LAUNCH");
            g_trace        = v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
            g_trace_parsed.store(true, std::memory_order_release);
        }
    }
    return g_trace;
}

void spacemit_trace_steps(const std::vector<spacemit_plugin_step> & steps, const std::vector<const char *> & names) {
    const size_t n = steps.size();
    for (size_t i = 0; i < n; i++) {
        const ggml_tensor * node = steps[i].node;
        const ggml_tensor * src0 = node->src[0];
        GGML_LOG_INFO("FlagOS SpacemiT: step %zu/%zu %s %s %s [%lld,%lld,%lld,%lld] %s %s\n", i + 1, n,
                      i < names.size() && names[i] != nullptr ? names[i] : "-",
                      node->name[0] != '\0' ? node->name : "-", ggml_op_name(node->op),
                      static_cast<long long>(node->ne[0]), static_cast<long long>(node->ne[1]),
                      static_cast<long long>(node->ne[2]), static_cast<long long>(node->ne[3]),
                      src0 != nullptr && src0->name[0] != '\0' ? src0->name : "-",
                      src0 != nullptr ? ggml_type_name(src0->type) : "-");
    }
}

bool spacemit_overlaps(const ggml_tensor * node, const ggml_tensor * t) {
    const char * a = static_cast<const char *>(node->data);
    const char * b = static_cast<const char *>(t->data);
    return a < b + ggml_nbytes(t) && b < a + ggml_nbytes(node);
}

//
// GGML_FLAGOS_STATS: the per-graph counters
//

spacemit_stats_file::spacemit_stats_file(const char * value) {
    to_stderr = value_is_stderr(value);
    if (!to_stderr && value != nullptr) {
        path = value;
    }
    if (writing()) {
        GGML_LOG_WARN("FlagOS SpacemiT: the counters (GGML_FLAGOS_STATS) go to %s at exit\n",
                      to_stderr ? "stderr" : path.c_str());
    }
}

spacemit_stats_file * spacemit_stats_file::get() {
    static spacemit_stats_file file(std::getenv("GGML_FLAGOS_STATS"));
    return file.writing() ? &file : nullptr;
}

void spacemit_stats_file::add(const char * launch, int64_t rows, int64_t n_kv, const spacemit_run_stats & run,
                              double graph_us, spacemit_plan_result plan, double lookup_us, double key_us) {
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
    k.plan_hits_uid += plan == spacemit_plan_result::hit_uid;
    k.plan_misses += plan == spacemit_plan_result::miss;
    k.plan_lookup_us += lookup_us;
    k.plan_key_us += key_us;
}

spacemit_stats_file::~spacemit_stats_file() {
    if (!writing() || kinds.empty()) {
        return;
    }
    FILE * f = counter_stream(to_stderr, path, "counters");
    if (f == nullptr) {
        return;
    }
    for (const auto & [key, k] : kinds) {
        std::fprintf(f,
                     "{\"launch\":\"%s\",\"rows\":%lld,\"kv\":%lld,\"graphs\":%llu,\"steps\":%llu,\"launches\":%llu,"
                     "\"tiles\":%u,\"graph_us\":%.3f,\"launch_us\":%.3f,\"kernel_us\":%.3f,\"busy_us\":%.3f,"
                     "\"barrier_us\":%.3f,\"plan_hits_uid\":%llu,\"plan_misses\":%llu,\"plan_lookup_us\":%.3f,"
                     "\"plan_key_us\":%.3f}\n",
                     std::get<0>(key).c_str(), (long long) std::get<1>(key), (long long) std::get<2>(key),
                     (unsigned long long) k.graphs, (unsigned long long) k.steps, (unsigned long long) k.launches,
                     k.tiles, k.graph_us, k.launch_us, k.kernel_us, k.busy_us, k.barrier_us,
                     (unsigned long long) k.plan_hits_uid, (unsigned long long) k.plan_misses, k.plan_lookup_us,
                     k.plan_key_us);
    }
    if (to_stderr) {
        std::fflush(f);
    } else {
        std::fclose(f);
    }
}

//
// GGML_FLAGOS_OP_STATS: the op counters
//

spacemit_op_stats_file::spacemit_op_stats_file(const char * value) {
    to_stderr = value_is_stderr(value);
    if (!to_stderr && value != nullptr) {
        path = value;
    }
    if (writing()) {
        GGML_LOG_WARN("FlagOS SpacemiT: the op counters (GGML_FLAGOS_OP_STATS) go to %s at exit\n",
                      to_stderr ? "stderr" : path.c_str());
    }
}

spacemit_op_stats_file * spacemit_op_stats_file::get() {
    static spacemit_op_stats_file file(std::getenv("GGML_FLAGOS_OP_STATS"));
    return file.writing() ? &file : nullptr;
}

void spacemit_op_stats_file::add(const char * launch, int64_t rows, int64_t n_kv,
                                 const std::vector<spacemit_plugin_step> & steps,
                                 const std::vector<const char *> & names, const spacemit_run_stats & run) {
    const size_t   n     = steps.size();
    const uint32_t tiles = run.n_tiles;
    if (run.n_steps != n || tiles == 0) {
        return;
    }
    GGML_ASSERT(names.size() == n);  // the two vectors are filled together in graph_compute
    const uint64_t              shape = structure_of(steps);
    std::lock_guard<std::mutex> lock(mutex);
    totals &                    k = kinds[std::make_tuple(std::string(launch), rows, n_kv, shape)];
    if (k.labels.empty()) {  // the kind's first graph: its names only
        k.tiles = tiles;
        for (size_t s = 0; s < n; s++) {
            k.labels.push_back(spacemit_step_label(steps, names, s));
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

spacemit_op_stats_file::~spacemit_op_stats_file() {
    if (!writing() || kinds.empty()) {
        return;
    }
    FILE * f = counter_stream(to_stderr, path, "op counters");
    if (f == nullptr) {
        return;
    }
    for (const auto & [key, k] : kinds) {
        // each name's steps, the names in the order of their first step; then all steps
        std::vector<std::string>                   names;
        std::map<std::string, std::vector<size_t>> steps_of;
        std::vector<size_t>                        all(k.labels.size());
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
            double                kernel = 0.0, busy = 0.0, wall = 0.0;
            std::vector<double>   tile(k.tiles, 0.0);
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
    if (to_stderr) {
        std::fflush(f);
    } else {
        std::fclose(f);
    }
}
