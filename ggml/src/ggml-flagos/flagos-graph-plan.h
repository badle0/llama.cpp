#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;

enum class flagos_pattern_id : uint32_t {
    none = 0,
    // Keep existing numeric IDs stable for out-of-tree providers. New pattern
    // kinds are appended even when they are conceptually adjacent.
    rms_norm_mul         = 1,
    add_rms_norm_mul     = 2,
    ssm_conv_silu        = 3,
    gated_delta_net_decode  = 4,
    gated_delta_net_prefill = 5,
    gated_rms_norm       = 6,
    ffn_swiglu           = 7,
    rope_kv_store        = 8,
    qkv_mrope_kv_store   = 9,
    flash_attn_decode    = 10,
    flash_attn_prefill   = 11,
    attention_output_gate = 12,
    rms_norm_mul_rope    = 13,
    rms_norm_mul_rope_kv_store = 14,
    // Gate/up projections, SwiGLU, and down projection with private GLU scratch.
    ffn_swiglu_down      = 15,
    // Parallel GDN alpha/beta projections and their decode gate epilogues.
    gdn_gate_projections = 16,
};

const char * flagos_pattern_name(flagos_pattern_id id);

enum class flagos_quantized_matmul_kind : uint32_t {
    none = 0,
    q4_k = 1,
    q6_k = 2,
    // Appended values preserve the numeric ABI of the original K-quant kinds.
    q4_0 = 3,
    q5_k = 4,
    q4_1 = 5,
    q8_0 = 6,
};

// Provider-neutral description of a GGML quantized matrix multiply. The
// descriptor deliberately contains only graph/layout facts; a provider still
// decides which AOT, vendor-library, or fallback implementation to use.
struct flagos_quantized_matmul_signature {
    flagos_quantized_matmul_kind weight_kind = flagos_quantized_matmul_kind::none;
    int64_t k = 0;
    int64_t rows = 0;
    int64_t columns = 0;
    int weight_type = -1;
    int activation_type = -1;
    int output_type = -1;
};

bool flagos_describe_quantized_matmul(
        const ggml_tensor * op,
        flagos_quantized_matmul_signature * signature);

// A pattern can either describe one composite GGML operator (for example
// FLASH_ATTN_EXT) or a subgraph made of several ordinary operators.  Keeping
// this classification in the common planner lets providers share one query
// and execution ABI without inferring intent from node_indices.size().
enum class flagos_fusion_scope : uint32_t {
    single_operator,
    graph,
};

flagos_fusion_scope flagos_pattern_scope(flagos_pattern_id id);

struct flagos_pattern_candidate {
    flagos_pattern_id id = flagos_pattern_id::none;
    flagos_fusion_scope scope = flagos_fusion_scope::graph;
    std::vector<int> node_indices;
    // The provider must write every listed output.
    std::vector<int> required_output_node_indices;
    uint64_t eliminated_read_bytes = 0;
    uint64_t eliminated_write_bytes = 0;
    uint32_t eliminated_launches = 0;
};

struct flagos_lowering_choice {
    bool supported = false;
    bool capture_safe = false;
    uint64_t implementation_id = 0;
    int64_t score = 0;
};

struct flagos_plan_step;

using flagos_resolve_tensor_data_fn = const void * (*)(
        void * user_data,
        const ggml_tensor * tensor);

using flagos_query_lowering_fn = flagos_lowering_choice (*)(
        void * user_data,
        const ggml_cgraph * cgraph,
        const flagos_pattern_candidate & candidate);

// Provider-neutral execution hook for a selected fused pattern. The provider
// owns its stream, kernel registry, and temporary-storage policy in user_data.
// Returning false means the selected implementation could not be launched and
// the backend must fail the graph rather than silently execute a partial fuse.
using flagos_execute_fusion_fn = bool (*) (
        void * user_data,
        ggml_cgraph * cgraph,
        const struct flagos_plan_step & step);

bool flagos_execute_fusion_step(
        flagos_execute_fusion_fn execute,
        void * user_data,
        ggml_cgraph * cgraph,
        const struct flagos_plan_step & step);

// Optional bundled form for providers that expose the planner through a
// single vtable-like object.  The callbacks remain plain function pointers so
// this interface is usable from providers with C or C++ runtimes alike.
struct flagos_fusion_interface {
    flagos_query_lowering_fn query_lowering = nullptr;
    flagos_query_lowering_fn validate_lowering = nullptr;
    flagos_execute_fusion_fn execute_fusion = nullptr;
    void * user_data = nullptr;
    // Change this value when provider policy changes accepted lowerings.
    uint64_t configuration_id = 0;
    // Optional provider address resolver. Providers that substitute private
    // allocations expose the actual address here so cache validation and
    // schedule-motion checks see the same binding as execute_fusion.
    flagos_resolve_tensor_data_fn resolve_tensor_data = nullptr;
};

bool flagos_execute_fusion_step(
        const flagos_fusion_interface & interface,
        ggml_cgraph * cgraph,
        const struct flagos_plan_step & step);

enum class flagos_execution_kind : uint32_t {
    direct,
    pattern,
};

struct flagos_plan_step {
    flagos_execution_kind kind = flagos_execution_kind::direct;
    flagos_pattern_candidate candidate;
    bool capture_safe = true;
    uint64_t implementation_id = 0;
};

struct flagos_graph_binding_slot {
    int node_index = -1;
    int source_index = -1;
    bool view_source = false;
};

struct flagos_graph_binding_snapshot {
    uint64_t fingerprint = 0;
    std::vector<uintptr_t> pointers;
};

struct flagos_canonical_tensor {
    bool present = false;
    // Non-negative values name graph nodes. Values below -1 name external aliases.
    int producer = -1;
    int type = -1;
    int op = -1;
    uint32_t flags = 0;
    size_t view_offset = 0;
    int view_source = -1;
    std::array<int64_t, 4> ne {};
    std::array<size_t, 4> nb {};
    std::array<uint8_t, 64> op_params {};

    bool operator==(const flagos_canonical_tensor & other) const;
};

struct flagos_canonical_node {
    flagos_canonical_tensor output;
    std::array<flagos_canonical_tensor, 10> sources;

    bool operator==(const flagos_canonical_node & other) const;
};

struct flagos_graph_plan {
    uint64_t structural_fingerprint = 0;
    bool binding_sensitive = false;
    std::vector<flagos_canonical_node> canonical_nodes;
    std::vector<flagos_plan_step> steps;
    std::vector<flagos_graph_binding_slot> binding_slots;

    bool matches(const ggml_cgraph * cgraph) const;
    bool capture_safe() const;
};

// The graph must match the plan's canonical structure. Null resolved addresses fail closed.
bool flagos_graph_binding_fingerprint(
        const ggml_cgraph * cgraph,
        const flagos_graph_plan & plan,
        uint64_t * fingerprint,
        flagos_resolve_tensor_data_fn resolve = nullptr,
        void * user_data = nullptr);

bool flagos_graph_binding_snapshot_create(
        const ggml_cgraph * cgraph,
        const flagos_graph_plan & plan,
        flagos_graph_binding_snapshot * snapshot,
        flagos_resolve_tensor_data_fn resolve = nullptr,
        void * user_data = nullptr);

bool flagos_graph_binding_snapshot_matches(
        const ggml_cgraph * cgraph,
        const flagos_graph_plan & plan,
        const flagos_graph_binding_snapshot & snapshot,
        flagos_resolve_tensor_data_fn resolve = nullptr,
        void * user_data = nullptr);

std::unique_ptr<flagos_graph_plan> flagos_build_graph_plan(
        const ggml_cgraph * cgraph,
        flagos_query_lowering_fn query_lowering,
        void * user_data);

std::unique_ptr<flagos_graph_plan> flagos_build_graph_plan(
        const ggml_cgraph * cgraph,
        const flagos_fusion_interface & interface);

class flagos_graph_plan_cache {
public:
    explicit flagos_graph_plan_cache(size_t capacity = 32);
    ~flagos_graph_plan_cache();

    const flagos_graph_plan & get_or_create(
            const ggml_cgraph * cgraph,
            flagos_query_lowering_fn query_lowering,
            void * user_data,
            bool * created = nullptr,
            uint64_t configuration_id = 0,
            flagos_query_lowering_fn validate_lowering = nullptr,
            flagos_resolve_tensor_data_fn resolve_tensor_data = nullptr);

    const flagos_graph_plan & get_or_create(
            const ggml_cgraph * cgraph,
            const flagos_fusion_interface & interface,
            bool * created = nullptr);

    void clear();
    size_t size() const;
    uint64_t hits() const;
    uint64_t misses() const;
    uint64_t evictions() const;

private:
    struct entry;
    struct scratch;

    static bool plan_is_valid(entry & item, const ggml_cgraph * cgraph);

    size_t capacity_;
    uint64_t tick_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t evictions_ = 0;
    entry * last_entry_ = nullptr;
    flagos_graph_plan * last_plan_ = nullptr;
    std::vector<std::unique_ptr<entry>> entries_;
    std::unique_ptr<scratch> scratch_;
};
