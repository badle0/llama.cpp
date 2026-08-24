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
    rms_norm_mul,
    add_rms_norm_mul,
    ssm_conv_silu,
    gated_delta_net_decode,
    gated_delta_net_prefill,
    gated_rms_norm,
    ffn_swiglu,
    rope_kv_store,
    qkv_mrope_kv_store,
    flash_attn_decode,
    flash_attn_prefill,
    attention_output_gate,
};

const char * flagos_pattern_name(flagos_pattern_id id);

enum class flagos_quantized_matmul_kind : uint32_t {
    none = 0,
    q4_k,
    q6_k,
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
    flagos_execute_fusion_fn execute_fusion = nullptr;
    void * user_data = nullptr;
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

struct flagos_canonical_tensor {
    bool present = false;
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
    std::vector<flagos_canonical_node> canonical_nodes;
    std::vector<flagos_plan_step> steps;

    bool matches(const ggml_cgraph * cgraph) const;
    bool capture_safe() const;
};

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
            bool * created = nullptr);

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

    size_t capacity_;
    uint64_t tick_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t evictions_ = 0;
    uint64_t last_uid_ = 0;
    flagos_graph_plan * last_plan_ = nullptr;
    std::vector<std::unique_ptr<entry>> entries_;
};
