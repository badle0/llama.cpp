#include "flagos-graph-plan.h"

#include "../ggml-impl.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>

static_assert(GGML_MAX_DIMS == 4, "update canonical tensor dimensions");
static_assert(GGML_MAX_SRC == 10, "update canonical tensor sources");
static_assert(GGML_MAX_OP_PARAMS == 64, "update canonical op parameters");

const char * flagos_pattern_name(flagos_pattern_id id) {
    switch (id) {
        case flagos_pattern_id::none:                    return "direct";
        case flagos_pattern_id::rms_norm_mul:            return "rms_norm_mul";
        case flagos_pattern_id::add_rms_norm_mul:        return "add_rms_norm_mul";
        case flagos_pattern_id::ssm_conv_silu:           return "ssm_conv_silu";
        case flagos_pattern_id::gated_delta_net_decode:  return "gated_delta_net_decode";
        case flagos_pattern_id::gated_delta_net_prefill: return "gated_delta_net_prefill";
        case flagos_pattern_id::gated_rms_norm:          return "gated_rms_norm";
        case flagos_pattern_id::ffn_swiglu:              return "ffn_swiglu";
        case flagos_pattern_id::qkv_mrope_kv_store:      return "qkv_mrope_kv_store";
        case flagos_pattern_id::flash_attn_decode:       return "flash_attn_decode";
        case flagos_pattern_id::flash_attn_prefill:      return "flash_attn_prefill";
        case flagos_pattern_id::attention_output_gate:   return "attention_output_gate";
    }
    return "unknown";
}

bool flagos_canonical_tensor::operator==(const flagos_canonical_tensor & other) const {
    return present == other.present && producer == other.producer && type == other.type && op == other.op &&
        flags == other.flags && view_offset == other.view_offset && view_source == other.view_source &&
        ne == other.ne && nb == other.nb && op_params == other.op_params;
}

bool flagos_canonical_node::operator==(const flagos_canonical_node & other) const {
    return output == other.output && sources == other.sources;
}

static void flagos_hash_bytes(uint64_t & hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
}

template <typename T>
static void flagos_hash_value(uint64_t & hash, const T & value) {
    flagos_hash_bytes(hash, &value, sizeof(value));
}

static flagos_canonical_tensor flagos_canonicalize_tensor(
        const ggml_tensor * tensor,
        const std::unordered_map<const ggml_tensor *, int> & producers) {
    flagos_canonical_tensor result;
    if (tensor == nullptr) {
        return result;
    }

    result.present = true;
    const auto producer = producers.find(tensor);
    result.producer = producer == producers.end() ? -1 : producer->second;
    result.type = tensor->type;
    result.op = tensor->op;
    result.flags = tensor->flags;
    result.view_offset = tensor->view_offs;
    const auto view_source = producers.find(tensor->view_src);
    result.view_source = view_source == producers.end() ? -1 : view_source->second;
    std::copy_n(tensor->ne, GGML_MAX_DIMS, result.ne.begin());
    std::copy_n(tensor->nb, GGML_MAX_DIMS, result.nb.begin());
    std::memcpy(result.op_params.data(), tensor->op_params, result.op_params.size());
    return result;
}

static std::vector<flagos_canonical_node> flagos_canonicalize_graph(const ggml_cgraph * cgraph) {
    std::unordered_map<const ggml_tensor *, int> producers;
    producers.reserve(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        producers.emplace(cgraph->nodes[i], i);
    }

    std::vector<flagos_canonical_node> result(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        result[i].output = flagos_canonicalize_tensor(node, producers);
        for (int src = 0; src < GGML_MAX_SRC; ++src) {
            result[i].sources[src] = flagos_canonicalize_tensor(node->src[src], producers);
        }
    }
    return result;
}

static uint64_t flagos_fingerprint(const std::vector<flagos_canonical_node> & nodes) {
    uint64_t hash = 1469598103934665603ULL;
    flagos_hash_value(hash, nodes.size());
    for (const auto & node : nodes) {
        const auto hash_tensor = [&hash](const flagos_canonical_tensor & tensor) {
            flagos_hash_value(hash, tensor.present);
            flagos_hash_value(hash, tensor.producer);
            flagos_hash_value(hash, tensor.type);
            flagos_hash_value(hash, tensor.op);
            flagos_hash_value(hash, tensor.flags);
            flagos_hash_value(hash, tensor.view_offset);
            flagos_hash_value(hash, tensor.view_source);
            flagos_hash_bytes(hash, tensor.ne.data(), tensor.ne.size() * sizeof(tensor.ne[0]));
            flagos_hash_bytes(hash, tensor.nb.data(), tensor.nb.size() * sizeof(tensor.nb[0]));
            flagos_hash_bytes(hash, tensor.op_params.data(), tensor.op_params.size());
        };
        hash_tensor(node.output);
        for (const auto & source : node.sources) {
            hash_tensor(source);
        }
    }
    return hash;
}

static uint64_t flagos_tensor_bytes(const ggml_tensor * tensor) {
    return tensor == nullptr ? 0 : static_cast<uint64_t>(ggml_nbytes(tensor));
}

static bool flagos_is_mul_of(const ggml_tensor * mul, const ggml_tensor * input) {
    return mul != nullptr && mul->op == GGML_OP_MUL &&
        (mul->src[0] == input || mul->src[1] == input) && ggml_are_same_shape(mul, input);
}

static std::vector<flagos_pattern_candidate> flagos_enumerate_candidates(const ggml_cgraph * cgraph) {
    std::vector<flagos_pattern_candidate> candidates;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];

        if (i + 2 < cgraph->n_nodes && node->op == GGML_OP_ADD) {
            const ggml_tensor * norm = cgraph->nodes[i + 1];
            const ggml_tensor * mul = cgraph->nodes[i + 2];
            if (norm->op == GGML_OP_RMS_NORM && norm->src[0] == node && flagos_is_mul_of(mul, norm)) {
                flagos_pattern_candidate candidate;
                candidate.id = flagos_pattern_id::add_rms_norm_mul;
                candidate.node_indices = { i, i + 1, i + 2 };
                candidate.eliminated_read_bytes = flagos_tensor_bytes(node) + flagos_tensor_bytes(norm);
                candidate.eliminated_launches = 2;
                candidates.push_back(candidate);
            }
        }

        if (i + 1 < cgraph->n_nodes && node->op == GGML_OP_RMS_NORM) {
            const ggml_tensor * mul = cgraph->nodes[i + 1];
            if (flagos_is_mul_of(mul, node)) {
                flagos_pattern_candidate candidate;
                candidate.id = flagos_pattern_id::rms_norm_mul;
                candidate.node_indices = { i, i + 1 };
                candidate.eliminated_read_bytes = flagos_tensor_bytes(node);
                candidate.eliminated_launches = 1;
                candidates.push_back(candidate);
            }
        }

        if (i + 1 < cgraph->n_nodes && node->op == GGML_OP_SSM_CONV) {
            const ggml_tensor * activation = cgraph->nodes[i + 1];
            if (activation->op == GGML_OP_UNARY && activation->src[0] == node &&
                ggml_get_unary_op(activation) == GGML_UNARY_OP_SILU) {
                flagos_pattern_candidate candidate;
                candidate.id = flagos_pattern_id::ssm_conv_silu;
                candidate.node_indices = { i, i + 1 };
                candidate.eliminated_read_bytes = flagos_tensor_bytes(node);
                candidate.eliminated_launches = 1;
                candidates.push_back(candidate);
            }
        }
    }
    return candidates;
}

static int64_t flagos_default_score(const flagos_pattern_candidate & candidate) {
    const uint64_t bytes = candidate.eliminated_read_bytes + candidate.eliminated_write_bytes;
    const uint64_t launch_score = static_cast<uint64_t>(candidate.eliminated_launches) * 1024 * 1024;
    const uint64_t score = std::min<uint64_t>(bytes + launch_score, std::numeric_limits<int64_t>::max());
    return static_cast<int64_t>(score);
}

bool flagos_graph_plan::matches(const ggml_cgraph * cgraph) const {
    return canonical_nodes == flagos_canonicalize_graph(cgraph);
}

bool flagos_graph_plan::capture_safe() const {
    return std::all_of(steps.begin(), steps.end(), [](const flagos_plan_step & step) {
        return step.capture_safe;
    });
}

std::unique_ptr<flagos_graph_plan> flagos_build_graph_plan(
        const ggml_cgraph * cgraph,
        flagos_query_lowering_fn query_lowering,
        void * user_data) {
    auto plan = std::make_unique<flagos_graph_plan>();
    plan->canonical_nodes = flagos_canonicalize_graph(cgraph);
    plan->structural_fingerprint = flagos_fingerprint(plan->canonical_nodes);

    struct selected_candidate {
        flagos_pattern_candidate candidate;
        flagos_lowering_choice choice;
    };

    std::vector<selected_candidate> supported;
    if (query_lowering != nullptr) {
        for (const auto & candidate : flagos_enumerate_candidates(cgraph)) {
            auto choice = query_lowering(user_data, cgraph, candidate);
            if (!choice.supported) {
                continue;
            }
            if (choice.score == 0) {
                choice.score = flagos_default_score(candidate);
            }
            if (choice.score > 0) {
                supported.push_back({ candidate, choice });
            }
        }
    }

    std::sort(supported.begin(), supported.end(), [](const selected_candidate & left, const selected_candidate & right) {
        if (left.choice.score != right.choice.score) {
            return left.choice.score > right.choice.score;
        }
        return left.candidate.node_indices.size() > right.candidate.node_indices.size();
    });

    std::vector<int> selected_by(cgraph->n_nodes, -1);
    std::vector<selected_candidate> selected;
    for (const auto & item : supported) {
        bool overlaps = false;
        for (const int node_index : item.candidate.node_indices) {
            if (selected_by[node_index] != -1) {
                overlaps = true;
                break;
            }
        }
        if (overlaps) {
            continue;
        }
        const int selected_index = static_cast<int>(selected.size());
        selected.push_back(item);
        for (const int node_index : item.candidate.node_indices) {
            selected_by[node_index] = selected_index;
        }
    }

    for (int node_index = 0; node_index < cgraph->n_nodes;) {
        const int selected_index = selected_by[node_index];
        if (selected_index >= 0 && selected[selected_index].candidate.node_indices[0] == node_index) {
            const auto & item = selected[selected_index];
            flagos_plan_step step;
            step.kind = flagos_execution_kind::pattern;
            step.candidate = item.candidate;
            step.capture_safe = item.choice.capture_safe;
            step.implementation_id = item.choice.implementation_id;
            plan->steps.push_back(step);
            node_index += static_cast<int>(item.candidate.node_indices.size());
            continue;
        }

        flagos_plan_step step;
        step.kind = flagos_execution_kind::direct;
        step.candidate.node_indices = { node_index };
        plan->steps.push_back(step);
        ++node_index;
    }

    return plan;
}

struct flagos_graph_plan_cache::entry {
    std::unique_ptr<flagos_graph_plan> plan;
    uint64_t last_used = 0;
};

flagos_graph_plan_cache::flagos_graph_plan_cache(size_t capacity) : capacity_(std::max<size_t>(capacity, 1)) {}

flagos_graph_plan_cache::~flagos_graph_plan_cache() = default;

const flagos_graph_plan & flagos_graph_plan_cache::get_or_create(
        const ggml_cgraph * cgraph,
        flagos_query_lowering_fn query_lowering,
        void * user_data,
        bool * created) {
    ++tick_;
    if (created != nullptr) {
        *created = false;
    }

    if (cgraph->uid != 0 && cgraph->uid == last_uid_ && last_plan_ != nullptr) {
        ++hits_;
        return *last_plan_;
    }

    const auto canonical_nodes = flagos_canonicalize_graph(cgraph);
    const uint64_t fingerprint = flagos_fingerprint(canonical_nodes);
    for (const auto & item : entries_) {
        if (item->plan->structural_fingerprint == fingerprint && item->plan->canonical_nodes == canonical_nodes) {
            item->last_used = tick_;
            last_uid_ = cgraph->uid;
            last_plan_ = item->plan.get();
            ++hits_;
            return *item->plan;
        }
    }

    if (entries_.size() >= capacity_) {
        const auto lru = std::min_element(entries_.begin(), entries_.end(), [](const auto & left, const auto & right) {
            return left->last_used < right->last_used;
        });
        if ((*lru)->plan.get() == last_plan_) {
            last_uid_ = 0;
            last_plan_ = nullptr;
        }
        entries_.erase(lru);
        ++evictions_;
    }

    auto item = std::make_unique<entry>();
    item->plan = flagos_build_graph_plan(cgraph, query_lowering, user_data);
    item->last_used = tick_;
    last_uid_ = cgraph->uid;
    last_plan_ = item->plan.get();
    entries_.push_back(std::move(item));
    ++misses_;
    if (created != nullptr) {
        *created = true;
    }
    return *last_plan_;
}

void flagos_graph_plan_cache::clear() {
    entries_.clear();
    last_uid_ = 0;
    last_plan_ = nullptr;
}

size_t flagos_graph_plan_cache::size() const {
    return entries_.size();
}

uint64_t flagos_graph_plan_cache::hits() const {
    return hits_;
}

uint64_t flagos_graph_plan_cache::misses() const {
    return misses_;
}

uint64_t flagos_graph_plan_cache::evictions() const {
    return evictions_;
}
