// The core's plan cache (flagos-spacemit-plan.h; plan.md "S1 step 3b spec", "S1 step 3c spec").

#include "flagos-spacemit-plan.h"

#include "ggml-impl.h"  // ggml_cgraph: uid, n_nodes, nodes

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>

namespace {

std::atomic<uint64_t> g_hits_uid{ 0 };
std::atomic<uint64_t> g_misses{ 0 };

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
    c.hits_uid = g_hits_uid.exchange(0);
    c.misses   = g_misses.exchange(0);
    return c;
}

void spacemit_plan_counters_add(spacemit_plan_result result) {
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
}

spacemit_plan_result spacemit_plan_cache::probe(const ggml_cgraph * cgraph, uint64_t word, bool want_names) const {
    // a plan expanded without the op counters' names cannot serve a graph that wants them (the counters' switch is
    // read once per process, so this does not happen; a guard). The pointer walk: CUDA asserts only the node count and
    // Hexagon checks nothing; within a trusted hit's 18 us key on Qwen3-4B (run 12), and it catches a uid reused on
    // other objects
    const bool hit = valid_ && cgraph->uid != 0 && cgraph->uid == uid_ && word == word_ && (!want_names || named_) &&
                     static_cast<size_t>(cgraph->n_nodes) == nodes_.size() &&
                     std::equal(nodes_.begin(), nodes_.end(), cgraph->nodes);
    return hit ? spacemit_plan_result::hit_uid : spacemit_plan_result::miss;
}

void spacemit_plan_cache::reset() {
    valid_ = false;
    steps.clear();
    names.clear();
    workspace_size = 0;
}

void spacemit_plan_cache::set_key(const ggml_cgraph * cgraph, uint64_t word) {
    uid_   = cgraph->uid;
    word_  = word;
    named_ = names.size() == steps.size();
    nodes_.assign(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    valid_ = uid_ != 0;
}
