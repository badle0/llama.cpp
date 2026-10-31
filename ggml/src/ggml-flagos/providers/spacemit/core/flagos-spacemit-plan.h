#pragma once

// The core's plan cache (plan.md "S1 design" §3, "S1 step 3 spec"): the draft of the deck's L1. On a hit graph_compute
// takes a split's step list and its two workspace sizes from here instead of expanding every node again; the contract
// check stays per graph. Both candidates for the key of A4 are built, behind one test switch, until the K3 decides:
//   structure  candidate S: Common's structural cache (flagos_graph_plan_cache, used as it is) finds the graph's plan
//              object, and the core's one entry holds the step list of the last plan it saw
//   uid        candidate U: the split's uid (ggml-backend.cpp:1538) and the same node objects take the entry directly;
//              a uid of 0, a new uid or other node objects go through S's path
// The core's own code: built without libspert.

#include "flagos-spacemit-plugin.h"  // spacemit_plugin_step

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;
struct spacemit_op;
struct flagos_graph_plan;
class flagos_graph_plan_cache;

enum class spacemit_plan_mode {
    none,       // today's path: every graph expanded, nothing cached (the default until A4 is decided)
    structure,  // candidate S
    uid,        // candidate U
};

// test-only until A4 is decided: FLAGOS_SPACEMIT_PLAN=none|structure|uid. Read at every graph, as the core's other
// per-graph switches, so a test can switch; anything else is reported once and ignored (none)
spacemit_plan_mode spacemit_plan_mode_get();

enum class spacemit_plan_result {
    off,            // FLAGOS_SPACEMIT_PLAN=none: no lookup
    hit_uid,        // U: the same uid and node objects
    hit_structure,  // S's path: Common returned the entry's plan object, not created, with the same word
    miss,           // expanded and stored
};

// test-only: what the lookups did since the last take, over all backends (none: nothing is counted). lookup_us: the
// time from the top of graph_compute to the step list in hand, summed; key_us: the lookup alone, from after the
// contract check to the step list in hand (on a miss with the expansion and the store), summed
struct spacemit_plan_counters {
    uint64_t hits_uid       = 0;
    uint64_t hits_structure = 0;
    uint64_t misses         = 0;
    uint64_t rebinds        = 0;  // hits by structure whose step nodes were other objects than the entry's
    double   lookup_us      = 0.0;
    double   key_us         = 0.0;
};

spacemit_plan_counters spacemit_plan_counters_take();
void spacemit_plan_counters_add(spacemit_plan_result result, bool rebound, double lookup_us, double key_us);

// one backend context's cache: Common's structural cache and the core's one entry, the last plan
class spacemit_plan_cache {
  public:
    spacemit_plan_cache();
    ~spacemit_plan_cache();
    spacemit_plan_cache(const spacemit_plan_cache &)             = delete;
    spacemit_plan_cache & operator=(const spacemit_plan_cache &) = delete;

    // word: the configuration word (bit 0 the activation-sharing fact, above it the plugin's switch generation).
    // node_ops: the contract the per-graph check found for each node of cgraph (nullptr for a node it skipped); a hit
    // needs the cached steps of every node to be its contract's steps, all of them and in order, and as many nodes
    // with a contract as when the entry was stored, so a node that now takes another contract (from a fact outside
    // the structure and the word) rebuilds. On a hit fills steps (workspace offsets 0;
    // graph_compute sets them from the sizes), names when want_names, and the two workspace sizes, and sets rebound
    // when a step's node is another object than the entry's. On a miss leaves them and remembers the plan object for
    // store()
    spacemit_plan_result lookup(spacemit_plan_mode mode, const ggml_cgraph * cgraph, uint64_t word, bool want_names,
                                const std::vector<const spacemit_op *> & node_ops,
                                std::vector<spacemit_plugin_step> & steps, std::vector<const char *> & names,
                                size_t & scratch_size, size_t & act_size, bool & rebound);

    // after a miss: the expansion of cgraph (steps in node order, names parallel to them or empty), the contracts the
    // check found (node_ops, as for lookup) and the sizes
    void store(spacemit_plan_mode mode, const ggml_cgraph * cgraph, uint64_t word,
               const std::vector<const spacemit_op *> & node_ops, const std::vector<spacemit_plugin_step> & steps, const std::vector<const char *> & names,
               size_t scratch_size, size_t act_size);

  private:
    // the hit guard: each node's cached steps exactly its contract's steps in node_ops, by position and count, and as
    // many nodes with a contract as at the store
    bool steps_match(const std::vector<const spacemit_op *> & node_ops) const;

    std::unique_ptr<flagos_graph_plan_cache> common_;
    const flagos_graph_plan *                pending_ = nullptr;  // the plan object of the last miss

    // the entry
    bool                              valid_ = false;
    spacemit_plan_mode                mode_  = spacemit_plan_mode::none;
    const flagos_graph_plan *         key_   = nullptr;
    uint64_t                          word_  = 0;
    uint64_t                          uid_   = 0;
    std::vector<const ggml_tensor *>  nodes_;  // the graph's nodes, for U's check
    std::vector<spacemit_plugin_step> steps_;  // bound to the last graph that took the entry
    std::vector<int>                  index_;  // each step's node index in the graph
    size_t                            n_contract_nodes_ = 0;  // nodes with a contract at the store
    std::vector<const char *>         names_;
    size_t                            scratch_size_ = 0, act_size_ = 0;
};
