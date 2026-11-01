#pragma once

// The core's plan cache (plan.md "S1 design" §3, "S1 step 3b spec", "S1 step 3c spec"): the draft of the deck's L1. On
// a hit graph_compute takes a split's step list and its two workspace sizes from here instead of checking and expanding
// every node again. One entry, the last plan, keyed by the split's uid (ggml-backend.cpp:1538), the very node objects
// it recorded and the configuration word; a uid of 0 or a new uid is a miss (the contract check, the expansion and a
// store), as Hexagon. The default since step 3c (FLAGOS_SPACEMIT_PLAN unset or uid-trust): the entry is taken before
// the contract check (probe), and a hit skips the check, trusting the one made at the store: the cached step list runs
// on the current node objects without their contracts being checked again, so a change to a node's operands or
// placement under a kept uid and the same objects is not seen. llama.cpp never does this (a reused graph keeps its
// tensors and their allocations; the uid changes when the graph is rebuilt), and the per-graph fail-closed rule of
// "S1 design" §3 (i) then holds at every store, that is at every miss, and for every graph the scheduler did not stamp.
// CUDA's and Hexagon's form, the deck's "③ and ④ redone only when the graph's structure changes". The test-only
// FLAGOS_SPACEMIT_PLAN=none is the full path: the contract check and the expansion on every graph, nothing cached.
// The core's own code: built without libspert and without Common.

#include "flagos-spacemit-plugin.h"  // spacemit_plugin_step

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;

enum class spacemit_plan_mode {
    none,       // test-only: every graph checked and expanded, nothing cached
    uid_trust,  // the default: the uid key taken before the contract check; a hit skips it
};

// FLAGOS_SPACEMIT_PLAN unset or empty: uid_trust; none (test-only): none; uid-trust: uid_trust; anything else is
// reported once and ignored (uid_trust). Read at every graph, as the core's other per-graph switches, so a test can
// switch
spacemit_plan_mode spacemit_plan_mode_get();

enum class spacemit_plan_result {
    off,      // FLAGOS_SPACEMIT_PLAN=none (or no cache under FLAGOS_SPACEMIT_QUANTIZE=shared): no lookup
    hit_uid,  // the same uid, node objects and word
    miss,     // expanded and stored
};

// read by tests only; added on every graph the cache serves or stores, the default path included (three clock reads
// on a miss, two on a hit, and three atomic adds per graph); nothing under none or QUANTIZE=shared. What the lookups
// did since the last take, over all backends. lookup_us: the time from the top of graph_compute to the step list in
// hand, summed; key_us: from after the contract check (on a hit, which skips it, from the top of graph_compute) to the
// step list in hand (on a miss the store; a miss's probe lies before the check and the expansion runs inside it, so
// both are in lookup_us only), summed
struct spacemit_plan_counters {
    uint64_t hits_uid  = 0;
    uint64_t misses    = 0;
    double   lookup_us = 0.0;
    double   key_us    = 0.0;
};

spacemit_plan_counters spacemit_plan_counters_take();
void spacemit_plan_counters_add(spacemit_plan_result result, double lookup_us, double key_us);

// one backend context's cache: one entry, the last plan
class spacemit_plan_cache {
  public:
    // the key: cgraph's uid not 0 and the entry's, word the entry's, and cgraph's nodes the very objects the entry
    // recorded, all of them and in order. word: the configuration word (bit 0 the activation-sharing fact, above it
    // the plugin's switch generation)

    // the key alone, before the contract check. On a hit fills steps (the entry's, whose nodes are this graph's;
    // workspace offsets 0, graph_compute sets them from the sizes), names when want_names, and the two workspace
    // sizes, and returns hit_uid; else leaves them and returns miss
    spacemit_plan_result probe(const ggml_cgraph * cgraph, uint64_t word, bool want_names,
                               std::vector<spacemit_plugin_step> & steps, std::vector<const char *> & names,
                               size_t & scratch_size, size_t & act_size) const;

    // after every miss, a uid of 0 included (such an entry never hits; the entry is always the last plan): the
    // expansion of cgraph after its contract check (steps in node order, names parallel to them or empty) and the sizes
    void store(const ggml_cgraph * cgraph, uint64_t word, const std::vector<spacemit_plugin_step> & steps,
               const std::vector<const char *> & names, size_t scratch_size, size_t act_size);

  private:
    // the key (above), and the entry able to serve a graph that wants the op counters' names
    bool key_matches(const ggml_cgraph * cgraph, uint64_t word, bool want_names) const;

    // the outputs of a hit
    void fill(bool want_names, std::vector<spacemit_plugin_step> & steps, std::vector<const char *> & names,
              size_t & scratch_size, size_t & act_size) const;

    // the entry
    bool                              valid_ = false;
    uint64_t                          uid_   = 0;
    uint64_t                          word_  = 0;
    std::vector<const ggml_tensor *>  nodes_;  // the graph's nodes, the key's pointer walk
    std::vector<spacemit_plugin_step> steps_;  // bound to the graph stored, whose node objects a hit requires
    std::vector<const char *>         names_;
    size_t                            scratch_size_ = 0, act_size_ = 0;
};
