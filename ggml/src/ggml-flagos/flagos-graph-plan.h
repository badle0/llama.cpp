#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct ggml_cgraph;

enum class flagos_pattern_id : uint32_t {
    none = 0,
    rms_norm_mul,
    add_rms_norm_mul,
    ssm_conv_silu,
    gated_delta_net_decode,
    gated_delta_net_prefill,
    gated_rms_norm,
    ffn_swiglu,
    qkv_mrope_kv_store,
    flash_attn_decode,
    flash_attn_prefill,
    attention_output_gate,
};

const char * flagos_pattern_name(flagos_pattern_id id);

struct flagos_pattern_candidate {
    flagos_pattern_id id = flagos_pattern_id::none;
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

using flagos_query_lowering_fn = flagos_lowering_choice (*)(
        void * user_data,
        const ggml_cgraph * cgraph,
        const flagos_pattern_candidate & candidate);

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

class flagos_graph_plan_cache {
public:
    explicit flagos_graph_plan_cache(size_t capacity = 32);
    ~flagos_graph_plan_cache();

    const flagos_graph_plan & get_or_create(
            const ggml_cgraph * cgraph,
            flagos_query_lowering_fn query_lowering,
            void * user_data,
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
