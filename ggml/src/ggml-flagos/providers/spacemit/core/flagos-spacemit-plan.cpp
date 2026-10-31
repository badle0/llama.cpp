// The core's plan cache (flagos-spacemit-plan.h; plan.md "S1 step 3 spec", "S1 step 3b spec").

#include "flagos-spacemit-plan.h"

#include "../flagos-spacemit-ops.h"  // spacemit_op: a contract's bound steps (the hit guard)

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
    if (v == nullptr || *v == '\0' || std::strcmp(v, "none") == 0) {
        return spacemit_plan_mode::none;
    }
    if (std::strcmp(v, "uid") == 0) {
        return spacemit_plan_mode::uid;
    }
    if (std::strcmp(v, "uid-trust") == 0) {
        return spacemit_plan_mode::uid_trust;
    }
    static std::atomic<bool> warned{ false };
    if (!warned.exchange(true)) {
        GGML_LOG_WARN("%s: FLAGOS_SPACEMIT_PLAN=%s: unknown setting ignored\n", __func__, v);
    }
    return spacemit_plan_mode::none;
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
    // Hexagon checks nothing; an estimate of a few us on 830 nodes (the K3's key column measures it), and it catches a
    // uid reused on other objects
    return valid_ && cgraph->uid != 0 && cgraph->uid == uid_ && word == word_ &&
           (!want_names || names_.size() == steps_.size()) && static_cast<size_t>(cgraph->n_nodes) == nodes_.size() &&
           std::equal(nodes_.begin(), nodes_.end(), cgraph->nodes);
}

bool spacemit_plan_cache::steps_match(const std::vector<const spacemit_op *> & node_ops) const {
    size_t n_contract = 0;  // nodes with a contract now
    for (const spacemit_op * op : node_ops) {
        n_contract += op != nullptr;
    }
    if (n_contract != n_contract_nodes_) {
        return false;
    }
    // the cached steps node by node (a node's steps one after another): each node's exactly its contract's steps
    size_t n_groups = 0;
    for (size_t j = 0; j < steps_.size();) {
        const size_t i = static_cast<size_t>(index_[j]);
        if (i >= node_ops.size() || node_ops[i] == nullptr) {
            return false;
        }
        const spacemit_op * op = node_ops[i];
        size_t              n  = 0;
        while (j + n < steps_.size() && static_cast<size_t>(index_[j + n]) == i) {
            n++;
        }
        if (n != op->n_steps) {
            return false;
        }
        for (uint32_t s = 0; s < op->n_steps; s++) {
            if (steps_[j + s].fn != op->steps[s]->fn) {
                return false;
            }
        }
        j += n;
        n_groups++;
    }
    return n_groups == n_contract_nodes_;
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

spacemit_plan_result spacemit_plan_cache::lookup(spacemit_plan_mode mode, const ggml_cgraph * cgraph, uint64_t word,
                                                 bool want_names, const std::vector<const spacemit_op *> & node_ops,
                                                 std::vector<spacemit_plugin_step> & steps,
                                                 std::vector<const char *> & names, size_t & scratch_size,
                                                 size_t & act_size) const {
    if (mode == spacemit_plan_mode::none) {
        return spacemit_plan_result::off;
    }
    if (!key_matches(cgraph, word, want_names) || !steps_match(node_ops)) {
        return spacemit_plan_result::miss;
    }
    fill(want_names, steps, names, scratch_size, act_size);
    return spacemit_plan_result::hit_uid;
}

void spacemit_plan_cache::store(const ggml_cgraph * cgraph, uint64_t word,
                                const std::vector<const spacemit_op *> & node_ops,
                                const std::vector<spacemit_plugin_step> & steps,
                                const std::vector<const char *> & names, size_t scratch_size, size_t act_size) {
    const size_t n_nodes = static_cast<size_t>(cgraph->n_nodes);
    valid_               = true;
    word_                = word;
    uid_                 = cgraph->uid;
    nodes_.assign(cgraph->nodes, cgraph->nodes + n_nodes);
    steps_.assign(steps.begin(), steps.end());
    names_.assign(names.begin(), names.end());
    scratch_size_     = scratch_size;
    act_size_         = act_size;
    n_contract_nodes_ = 0;
    for (const spacemit_op * op : node_ops) {
        n_contract_nodes_ += op != nullptr;
    }
    // each step's node index: the steps follow the nodes in order, a node's steps one after another
    index_.resize(steps_.size());
    size_t i = 0;
    for (size_t j = 0; j < steps_.size(); j++) {
        while (i < n_nodes && cgraph->nodes[i] != steps_[j].node) {
            i++;
        }
        if (i == n_nodes) {
            valid_ = false;  // not the graph's step list: never served
            return;
        }
        index_[j] = static_cast<int>(i);
    }
}
