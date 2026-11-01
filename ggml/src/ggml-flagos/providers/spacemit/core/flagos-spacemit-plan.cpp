// The core's plan cache (flagos-spacemit-plan.h; plan.md "S1 step 3b spec", "S1 step 3c spec").

#include "flagos-spacemit-plan.h"

#include "../../../../ggml-impl.h"  // ggml_cgraph: uid, n_nodes, nodes

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>

namespace {

std::atomic<uint64_t> g_hits_uid{ 0 };
std::atomic<uint64_t> g_misses{ 0 };
std::atomic<uint64_t> g_lookup_ns{ 0 };
std::atomic<uint64_t> g_key_ns{ 0 };

}  // namespace

spacemit_plan_mode spacemit_plan_mode_get() {
    const char * v = std::getenv("FLAGOS_SPACEMIT_PLAN");
    if (v == nullptr || *v == '\0' || std::strcmp(v, "uid-trust") == 0) {
        return spacemit_plan_mode::uid_trust;
    }
    if (std::strcmp(v, "none") == 0) {
        return spacemit_plan_mode::none;
    }
    static std::atomic<bool> warned{ false };
    if (!warned.exchange(true)) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_PLAN=%s: unknown setting ignored (uid-trust)\n", __func__, v);
    }
    return spacemit_plan_mode::uid_trust;
}

spacemit_plan_counters spacemit_plan_counters_take() {
    spacemit_plan_counters c;
    c.hits_uid  = g_hits_uid.exchange(0);
    c.misses    = g_misses.exchange(0);
    c.lookup_us = static_cast<double>(g_lookup_ns.exchange(0)) / 1e3;
    c.key_us    = static_cast<double>(g_key_ns.exchange(0)) / 1e3;
    return c;
}

void spacemit_plan_counters_add(spacemit_plan_result result, double lookup_us, double key_us) {
    switch (result) {
        case spacemit_plan_result::off:
            return;
        case spacemit_plan_result::hit_uid:
            g_hits_uid++;
            break;
        case spacemit_plan_result::miss:
            g_misses++;
            break;
    }
    g_lookup_ns += static_cast<uint64_t>(lookup_us * 1e3);
    g_key_ns += static_cast<uint64_t>(key_us * 1e3);
}

bool spacemit_plan_cache::key_matches(const ggml_cgraph * cgraph, uint64_t word, bool want_names) const {
    // a plan stored without the op counters' names cannot serve a graph that wants them (the counters' switch is read
    // once per process, so this does not happen; a guard). The pointer walk: CUDA asserts only the node count and
    // Hexagon checks nothing; within a trusted hit's 18 us key on Qwen3-4B (run 12), and it catches a uid reused on
    // other objects
    return valid_ && cgraph->uid != 0 && cgraph->uid == uid_ && word == word_ &&
           (!want_names || names_.size() == steps_.size()) && static_cast<size_t>(cgraph->n_nodes) == nodes_.size() &&
           std::equal(nodes_.begin(), nodes_.end(), cgraph->nodes);
}

void spacemit_plan_cache::fill(bool want_names, std::vector<spacemit_plugin_step> & steps,
                               std::vector<const char *> & names, size_t & scratch_size, size_t & act_size) const {
    steps.assign(steps_.begin(), steps_.end());
    if (want_names) {
        names.assign(names_.begin(), names_.end());
    }
    scratch_size = scratch_size_;
    act_size     = act_size_;
}

spacemit_plan_result spacemit_plan_cache::probe(const ggml_cgraph * cgraph, uint64_t word, bool want_names,
                                                std::vector<spacemit_plugin_step> & steps,
                                                std::vector<const char *> & names, size_t & scratch_size,
                                                size_t & act_size) const {
    if (!key_matches(cgraph, word, want_names)) {
        return spacemit_plan_result::miss;
    }
    fill(want_names, steps, names, scratch_size, act_size);
    return spacemit_plan_result::hit_uid;
}

void spacemit_plan_cache::store(const ggml_cgraph * cgraph, uint64_t word,
                                const std::vector<spacemit_plugin_step> & steps,
                                const std::vector<const char *> & names, size_t scratch_size, size_t act_size) {
    valid_ = true;
    word_  = word;
    uid_   = cgraph->uid;
    nodes_.assign(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    steps_.assign(steps.begin(), steps.end());
    names_.assign(names.begin(), names.end());
    scratch_size_ = scratch_size;
    act_size_     = act_size;
}
