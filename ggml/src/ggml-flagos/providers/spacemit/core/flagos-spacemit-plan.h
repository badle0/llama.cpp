#pragma once

// The core's plan cache (plan.md "S1 design" section 3, "S1 step 3b spec", "S1 step 3c spec"): the draft of the
// deck's L1. It holds the backend's current plan, the step list graph_compute replays and its workspace size, and a key
// that says which graph the plan was expanded for. On a hit graph_compute replays the plan as it is instead of checking
// and expanding every node again. One plan, the last, keyed by the split's uid (ggml-backend.cpp:1538), the very node
// objects it recorded and the configuration word; a uid of 0 or a new uid is a miss (the contract check, the expansion
// and a new key), as Hexagon. The default since step 3c (FLAGOS_SPACEMIT_PLAN unset or uid-trust): the key is taken
// before the contract check (probe), and a hit skips the check, trusting the one made when the key was set: the plan
// runs on the current node objects without their contracts being checked again, so a change to a node's operands or
// placement under a kept uid and the same objects is not seen. llama.cpp never does this (a reused graph keeps its
// tensors and their allocations; the uid changes when the graph is rebuilt), and the per-graph fail-closed rule of
// "S1 design" section 3 (i) then holds at every miss and for every graph the scheduler did not stamp.
// CUDA's and Hexagon's form, the deck's "③ and ④ redone only when the graph's structure changes". The test-only
// FLAGOS_SPACEMIT_PLAN=none is the full path: the contract check and the expansion on every graph, the key never set.
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
    off,      // FLAGOS_SPACEMIT_PLAN=none: no lookup
    hit_uid,  // the same uid, node objects and word
    miss,     // checked and expanded
};

// read by tests only; added on every graph the cache serves or expands, the default path included (one atomic add per
// graph); nothing under none. What the lookups did since the last take, over all backends
struct spacemit_plan_counters {
    uint64_t hits_uid = 0;
    uint64_t misses   = 0;
};

spacemit_plan_counters spacemit_plan_counters_take();
void spacemit_plan_counters_add(spacemit_plan_result result);

// one backend context's plan: the steps in node order, their contract names (parallel to them, or empty) and the
// largest workspace one of their ops needs, filled by graph_compute's expansion between reset() and set_key()
class spacemit_plan_cache {
  public:
    std::vector<spacemit_plugin_step> steps;
    std::vector<const char *>         names;
    size_t                            workspace_size = 0;

    // the key alone, before the contract check: hit_uid when the key is set and cgraph's uid is not 0 and the key's,
    // word the key's, cgraph's nodes the very objects the key recorded, all of them and in order, and the plan has
    // names when want_names; else miss. word: the configuration word, the plugin's switch generation
    spacemit_plan_result probe(const ggml_cgraph * cgraph, uint64_t word, bool want_names) const;

    // before an expansion: the key cleared, the plan empty
    void reset();

    // after the contract check and the expansion of a stamped cgraph (uid not 0): the plan is cgraph's under word
    void set_key(const ggml_cgraph * cgraph, uint64_t word);

  private:
    bool                             valid_ = false;
    uint64_t                         uid_   = 0;
    uint64_t                         word_  = 0;
    bool                             named_ = false;  // names parallel to steps when the key was set
    std::vector<const ggml_tensor *> nodes_;          // the graph's nodes, the key's pointer walk
};
