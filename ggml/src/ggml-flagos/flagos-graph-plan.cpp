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
        case flagos_pattern_id::rms_norm_mul_rope:       return "rms_norm_mul_rope";
        case flagos_pattern_id::rms_norm_mul_rope_kv_store:
                                                            return "rms_norm_mul_rope_kv_store";
        case flagos_pattern_id::add_rms_norm_mul:        return "add_rms_norm_mul";
        case flagos_pattern_id::ssm_conv_silu:           return "ssm_conv_silu";
        case flagos_pattern_id::gated_delta_net_decode:  return "gated_delta_net_decode";
        case flagos_pattern_id::gated_delta_net_prefill: return "gated_delta_net_prefill";
        case flagos_pattern_id::gated_rms_norm:          return "gated_rms_norm";
        case flagos_pattern_id::ffn_swiglu:              return "ffn_swiglu";
        case flagos_pattern_id::ffn_swiglu_down:         return "ffn_swiglu_down";
        case flagos_pattern_id::rope_kv_store:           return "rope_kv_store";
        case flagos_pattern_id::qkv_mrope_kv_store:      return "qkv_mrope_kv_store";
        case flagos_pattern_id::flash_attn_decode:       return "flash_attn_decode";
        case flagos_pattern_id::flash_attn_prefill:      return "flash_attn_prefill";
        case flagos_pattern_id::attention_output_gate:   return "attention_output_gate";
    }
    return "unknown";
}

bool flagos_describe_quantized_matmul(
        const ggml_tensor * op,
        flagos_quantized_matmul_signature * signature) {
    if (signature != nullptr) {
        *signature = {};
    }
    if (op == nullptr || signature == nullptr || op->op != GGML_OP_MUL_MAT ||
        op->src[0] == nullptr || op->src[1] == nullptr ||
        (op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_1 &&
         op->src[0]->type != GGML_TYPE_Q5_K &&
         op->src[0]->type != GGML_TYPE_Q4_K &&
         op->src[0]->type != GGML_TYPE_Q6_K && op->src[0]->type != GGML_TYPE_Q8_0) ||
        op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
        op->src[0]->ne[0] <= 0 ||
        ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_1 ||
          op->src[0]->type == GGML_TYPE_Q8_0) ? op->src[0]->ne[0] % 32 != 0
                                               : op->src[0]->ne[0] % 256 != 0) ||
        op->src[0]->ne[0] != op->src[1]->ne[0] || op->src[0]->ne[1] <= 0 ||
        op->src[1]->ne[1] <= 0 || op->src[0]->ne[2] != 1 || op->src[0]->ne[3] != 1 ||
        op->src[1]->ne[2] != 1 || op->src[1]->ne[3] != 1 ||
        op->ne[0] != op->src[0]->ne[1] || op->ne[1] != op->src[1]->ne[1] ||
        op->ne[2] != 1 || op->ne[3] != 1 || !ggml_is_contiguous(op->src[0]) ||
        !ggml_is_contiguous(op->src[1]) || !ggml_is_contiguous(op)) {
        return false;
    }
    signature->weight_kind = op->src[0]->type == GGML_TYPE_Q4_0
        ? flagos_quantized_matmul_kind::q4_0
        : op->src[0]->type == GGML_TYPE_Q4_1
        ? flagos_quantized_matmul_kind::q4_1
        : op->src[0]->type == GGML_TYPE_Q5_K
        ? flagos_quantized_matmul_kind::q5_k
        : op->src[0]->type == GGML_TYPE_Q4_K
        ? flagos_quantized_matmul_kind::q4_k
        : op->src[0]->type == GGML_TYPE_Q6_K
        ? flagos_quantized_matmul_kind::q6_k
        : flagos_quantized_matmul_kind::q8_0;
    signature->k = op->src[0]->ne[0];
    signature->rows = op->src[0]->ne[1];
    signature->columns = op->src[1]->ne[1];
    signature->weight_type = op->src[0]->type;
    signature->activation_type = op->src[1]->type;
    signature->output_type = op->type;
    return true;
}

flagos_fusion_scope flagos_pattern_scope(flagos_pattern_id id) {
    switch (id) {
        // These are already composite GGML operators.  They still go through
        // the common plan so a provider can replace the operator with an AOT
        // implementation, but they are not multi-node graph rewrites.
        case flagos_pattern_id::flash_attn_decode:
        case flagos_pattern_id::flash_attn_prefill:
            return flagos_fusion_scope::single_operator;
        case flagos_pattern_id::none:
        case flagos_pattern_id::rms_norm_mul:
        case flagos_pattern_id::rms_norm_mul_rope:
        case flagos_pattern_id::rms_norm_mul_rope_kv_store:
        case flagos_pattern_id::add_rms_norm_mul:
        case flagos_pattern_id::ssm_conv_silu:
        case flagos_pattern_id::gated_delta_net_decode:
        case flagos_pattern_id::gated_delta_net_prefill:
        case flagos_pattern_id::gated_rms_norm:
        case flagos_pattern_id::ffn_swiglu:
        case flagos_pattern_id::ffn_swiglu_down:
        case flagos_pattern_id::rope_kv_store:
        case flagos_pattern_id::qkv_mrope_kv_store:
        case flagos_pattern_id::attention_output_gate:
            return flagos_fusion_scope::graph;
    }
    return flagos_fusion_scope::graph;
}

bool flagos_execute_fusion_step(
        flagos_execute_fusion_fn execute,
        void * user_data,
        ggml_cgraph * cgraph,
        const flagos_plan_step & step) {
    return execute != nullptr && execute(user_data, cgraph, step);
}

bool flagos_execute_fusion_step(
        const flagos_fusion_interface & interface,
        ggml_cgraph * cgraph,
        const flagos_plan_step & step) {
    return flagos_execute_fusion_step(
        interface.execute_fusion, interface.user_data, cgraph, step);
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

struct flagos_producer_map {
    std::vector<const ggml_tensor *> keys;
    std::vector<int> values;

    void rebuild(const ggml_cgraph * cgraph) {
        const size_t minimum_size = std::max<size_t>(1, static_cast<size_t>(cgraph->n_nodes) * 2);
        const size_t table_size = ggml_hash_size(minimum_size);
        if (keys.size() < table_size) {
            keys.resize(table_size);
            values.resize(table_size);
        }
        std::fill(keys.begin(), keys.end(), nullptr);
        for (int index = 0; index < cgraph->n_nodes; ++index) {
            const ggml_tensor * tensor = cgraph->nodes[index];
            size_t slot = ggml_hash(tensor) % keys.size();
            while (keys[slot] != nullptr && keys[slot] != tensor) {
                slot = (slot + 1) % keys.size();
            }
            if (keys[slot] == nullptr) {
                keys[slot] = tensor;
                values[slot] = index;
            }
        }
    }

    int find(const ggml_tensor * tensor) const {
        if (tensor == nullptr || keys.empty()) {
            return -1;
        }
        const size_t first = ggml_hash(tensor) % keys.size();
        size_t slot = first;
        while (keys[slot] != nullptr) {
            if (keys[slot] == tensor) {
                return values[slot];
            }
            slot = (slot + 1) % keys.size();
            if (slot == first) {
                break;
            }
        }
        return -1;
    }
};

struct flagos_tensor_identity_map {
    std::unordered_map<const ggml_tensor *, int> external;

    void reset(size_t reserve) {
        external.clear();
        external.reserve(reserve);
    }

    int find(const ggml_tensor * tensor, const flagos_producer_map & producers) {
        if (tensor == nullptr) {
            return -1;
        }
        const int producer = producers.find(tensor);
        if (producer >= 0) {
            return producer;
        }
        const auto [entry, inserted] = external.emplace(
            tensor, -2 - static_cast<int>(external.size()));
        GGML_UNUSED(inserted);
        return entry->second;
    }
};

static flagos_canonical_tensor flagos_canonicalize_tensor(
        const ggml_tensor * tensor,
        const flagos_producer_map & producers,
        flagos_tensor_identity_map & identities) {
    flagos_canonical_tensor result;
    if (tensor == nullptr) {
        return result;
    }

    result.present = true;
    result.producer = identities.find(tensor, producers);
    result.type = tensor->type;
    result.op = tensor->op;
    result.flags = tensor->flags;
    result.view_offset = tensor->view_offs;
    result.view_source = identities.find(tensor->view_src, producers);
    std::copy_n(tensor->ne, GGML_MAX_DIMS, result.ne.begin());
    std::copy_n(tensor->nb, GGML_MAX_DIMS, result.nb.begin());
    std::memcpy(result.op_params.data(), tensor->op_params, result.op_params.size());
    return result;
}

static void flagos_prepare_producers(
        const ggml_cgraph * cgraph,
        flagos_producer_map & producers,
        std::vector<const ggml_tensor *> & producer_nodes) {
    if (cgraph->n_nodes == 0) {
        producer_nodes.clear();
        producers.rebuild(cgraph);
        return;
    }
    const bool same_producers = producer_nodes.size() == static_cast<size_t>(cgraph->n_nodes) &&
        std::equal(producer_nodes.begin(), producer_nodes.end(), cgraph->nodes);
    if (!same_producers) {
        producer_nodes.assign(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
        producers.rebuild(cgraph);
    }
}

static void flagos_canonicalize_graph(
        const ggml_cgraph * cgraph,
        flagos_producer_map & producers,
        flagos_tensor_identity_map & identities,
        std::vector<const ggml_tensor *> & producer_nodes,
        std::vector<flagos_canonical_node> & result) {
    flagos_prepare_producers(cgraph, producers, producer_nodes);
    identities.reset(static_cast<size_t>(cgraph->n_nodes) * 2);

    result.resize(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        result[i].output = flagos_canonicalize_tensor(node, producers, identities);
        for (int src = 0; src < GGML_MAX_SRC; ++src) {
            result[i].sources[src] = flagos_canonicalize_tensor(node->src[src], producers, identities);
        }
    }
}

static bool flagos_tensor_matches_canonical(
        const ggml_tensor * tensor,
        const flagos_producer_map & producers,
        flagos_tensor_identity_map & identities,
        const flagos_canonical_tensor & canonical) {
    if (canonical.present != (tensor != nullptr)) {
        return false;
    }
    if (tensor == nullptr) {
        return true;
    }

    const int producer_index = identities.find(tensor, producers);
    const int view_source_index = identities.find(tensor->view_src, producers);
    return canonical.producer == producer_index && canonical.type == tensor->type && canonical.op == tensor->op &&
        canonical.flags == static_cast<uint32_t>(tensor->flags) && canonical.view_offset == tensor->view_offs &&
        canonical.view_source == view_source_index &&
        std::equal(canonical.ne.begin(), canonical.ne.end(), tensor->ne) &&
        std::equal(canonical.nb.begin(), canonical.nb.end(), tensor->nb) &&
        std::memcmp(canonical.op_params.data(), tensor->op_params, canonical.op_params.size()) == 0;
}

static bool flagos_graph_matches_canonical(
        const ggml_cgraph * cgraph,
        const flagos_producer_map & producers,
        flagos_tensor_identity_map & identities,
        const std::vector<flagos_canonical_node> & canonical_nodes) {
    if (canonical_nodes.size() != static_cast<size_t>(cgraph->n_nodes)) {
        return false;
    }
    identities.reset(static_cast<size_t>(cgraph->n_nodes) * 2);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        const flagos_canonical_node & canonical = canonical_nodes[static_cast<size_t>(i)];
        if (!flagos_tensor_matches_canonical(node, producers, identities, canonical.output)) {
            return false;
        }
        for (int src = 0; src < GGML_MAX_SRC; ++src) {
            if (!flagos_tensor_matches_canonical(
                    node->src[src], producers, identities, canonical.sources[src])) {
                return false;
            }
        }
    }
    return true;
}

static std::vector<flagos_canonical_node> flagos_canonicalize_graph(const ggml_cgraph * cgraph) {
    flagos_producer_map producers;
    flagos_tensor_identity_map identities;
    std::vector<const ggml_tensor *> producer_nodes;
    std::vector<flagos_canonical_node> result;
    flagos_canonicalize_graph(cgraph, producers, identities, producer_nodes, result);
    return result;
}

static uint64_t flagos_fingerprint(const std::vector<flagos_canonical_node> & nodes) {
    uint64_t hash = 1469598103934665603ULL;
    flagos_hash_value(hash, nodes.size());
    for (const auto & node : nodes) {
        const auto hash_tensor = [&hash](const flagos_canonical_tensor & tensor) {
            flagos_hash_value(hash, tensor.present);
            if (!tensor.present) {
                return;
            }
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
    if (mul == nullptr || input == nullptr || mul->op != GGML_OP_MUL ||
        (mul->src[0] != input && mul->src[1] != input) ||
        !ggml_are_same_shape(mul, input)) {
        return false;
    }
    // The second operand may be a broadcast weight (the common Transformer
    // RMSNorm scale layout is [hidden, 1, 1, 1]).  Providers still validate
    // the exact stride/type ABI in their lowering query; the planner only
    // records the dependency when the operation is semantically repeatable.
    const ggml_tensor * weight = mul->src[0] == input ? mul->src[1] : mul->src[0];
    return weight != nullptr && ggml_can_repeat(weight, input);
}

static bool flagos_is_graph_barrier(const ggml_tensor * node) {
    if (node == nullptr) {
        return false;
    }
    // These operations mutate or copy externally visible storage.  Do not
    // move a fusion across one merely because its tensor dependencies are
    // disjoint; preserving GGML's side-effect order is more important than a
    // speculative launch reduction.
    switch (node->op) {
        case GGML_OP_ACC:
        case GGML_OP_SET:
        case GGML_OP_CPY:
        case GGML_OP_SET_ROWS:
        case GGML_OP_MAP_CUSTOM1:
        case GGML_OP_MAP_CUSTOM2:
        case GGML_OP_MAP_CUSTOM3:
        case GGML_OP_CUSTOM:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            return true;
        default:
            return false;
    }
}

static bool flagos_is_view_or_noop(const ggml_tensor * node) {
    return node != nullptr && (ggml_is_empty(node) || node->op == GGML_OP_NONE ||
        node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW ||
        node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE);
}

static bool flagos_is_gated_delta_net_cache_copy(
        const ggml_tensor * gdn,
        const ggml_tensor * snapshot_view,
        const ggml_tensor * copy) {
    if (gdn == nullptr || snapshot_view == nullptr || copy == nullptr ||
        gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        snapshot_view->op != GGML_OP_VIEW || snapshot_view->type != GGML_TYPE_F32 ||
        snapshot_view->src[0] != gdn || snapshot_view->view_src != gdn ||
        copy->op != GGML_OP_CPY || copy->type != GGML_TYPE_F32 ||
        copy->src[0] != snapshot_view || copy->src[1] == nullptr ||
        copy->src[1]->op != GGML_OP_VIEW || copy->src[1]->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(snapshot_view)) {
        return false;
    }

    const ggml_tensor * value = gdn->src[2];
    if (value == nullptr || value->ne[0] <= 0 || value->ne[1] <= 0 ||
        value->ne[2] <= 0 || value->ne[3] <= 0) {
        return false;
    }
    const int64_t state_size = value->ne[0];
    const int64_t heads = value->ne[1];
    const int64_t tokens = value->ne[2];
    const int64_t sequences = value->ne[3];
    const int64_t snapshots = ggml_get_op_params_i32(gdn, 0);
    if (snapshots <= 0 || state_size > INT64_MAX / state_size ||
        state_size * state_size > INT64_MAX / heads) {
        return false;
    }
    const int64_t state_elements = state_size * state_size * heads;
    if (state_elements > INT64_MAX / sequences ||
        state_size > INT64_MAX / heads || state_size * heads > INT64_MAX / tokens ||
        state_size * heads * tokens > INT64_MAX / sequences) {
        return false;
    }
    const int64_t snapshot_stride = state_elements * sequences;
    const int64_t attention_elements = state_size * heads * tokens * sequences;
    const int64_t written = std::min(tokens, snapshots);
    const size_t element_size = ggml_type_size(GGML_TYPE_F32);
    if (snapshot_stride > INT64_MAX / written ||
        static_cast<uint64_t>(attention_elements) > SIZE_MAX / element_size ||
        static_cast<uint64_t>(state_elements) > SIZE_MAX / element_size ||
        snapshot_view->view_offs != ggml_row_size(GGML_TYPE_F32, attention_elements) ||
        ggml_nelements(snapshot_view) != snapshot_stride * written) {
        return false;
    }

    const ggml_tensor * destination = copy->src[1];
    const std::array<int64_t, GGML_MAX_DIMS> expected = {
        state_elements, sequences, written, 1,
    };
    return std::equal(expected.begin(), expected.end(), destination->ne) &&
        std::equal(expected.begin(), expected.end(), copy->ne) &&
        destination->nb[0] == ggml_type_size(GGML_TYPE_F32) &&
        destination->nb[1] == ggml_row_size(GGML_TYPE_F32, state_elements) &&
        ggml_nelements(copy) == ggml_nelements(snapshot_view);
}

// A graph-private tensor has one consumer and is not externally visible.
static bool flagos_tensor_is_private_to_consumer(
        const ggml_cgraph * cgraph,
        const ggml_tensor * producer,
        int expected_consumer_index) {
    if (cgraph == nullptr || producer == nullptr || expected_consumer_index < 0 ||
        expected_consumer_index >= cgraph->n_nodes ||
        (producer->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }
    bool found_expected = false;
    for (int consumer_index = 0; consumer_index < cgraph->n_nodes; ++consumer_index) {
        const ggml_tensor * consumer = cgraph->nodes[consumer_index];
        bool consumes = consumer->view_src == producer;
        for (int source_index = 0; !consumes && source_index < GGML_MAX_SRC; ++source_index) {
            consumes = consumer->src[source_index] == producer;
        }
        if (!consumes) {
            continue;
        }
        if (consumer_index != expected_consumer_index) {
            return false;
        }
        found_expected = true;
    }
    return found_expected;
}

static bool flagos_populate_required_outputs(
        const ggml_cgraph * cgraph, flagos_pattern_candidate & candidate) {
    if (cgraph == nullptr || candidate.node_indices.empty()) {
        return false;
    }
    std::vector<bool> covered(static_cast<size_t>(cgraph->n_nodes), false);
    for (const int index : candidate.node_indices) {
        if (index < 0 || index >= cgraph->n_nodes || covered[static_cast<size_t>(index)]) {
            return false;
        }
        covered[static_cast<size_t>(index)] = true;
    }

    candidate.required_output_node_indices.clear();
    for (size_t position = 0; position < candidate.node_indices.size(); ++position) {
        const int producer_index = candidate.node_indices[position];
        const ggml_tensor * producer = cgraph->nodes[producer_index];
        bool required = position + 1 == candidate.node_indices.size() ||
            (producer->flags & GGML_TENSOR_FLAG_OUTPUT) != 0;
        for (int consumer_index = 0; consumer_index < cgraph->n_nodes; ++consumer_index) {
            if (required || consumer_index == producer_index || covered[static_cast<size_t>(consumer_index)]) {
                continue;
            }
            const ggml_tensor * consumer = cgraph->nodes[consumer_index];
            for (int source_index = 0; source_index < GGML_MAX_SRC; ++source_index) {
                if (consumer->src[source_index] == producer) {
                    required = true;
                    break;
                }
            }
            if (consumer->view_src == producer) {
                required = true;
            }
        }
        if (required) {
            candidate.required_output_node_indices.push_back(producer_index);
        }
    }
    return true;
}

// A non-contiguous pattern executes at its first node.  External inputs used
// by later nodes therefore must already exist at that point.  An interleaved
// producer cannot be skipped over merely because it is outside the candidate:
// doing so would launch the fused kernel before that input has been computed.
static bool flagos_candidate_inputs_available_at_entry(
        const ggml_cgraph * cgraph, const flagos_pattern_candidate & candidate) {
    if (cgraph == nullptr || candidate.node_indices.empty()) {
        return false;
    }
    const int entry = candidate.node_indices.front();
    if (entry < 0 || entry >= cgraph->n_nodes) {
        return false;
    }
    std::unordered_map<const ggml_tensor *, int> producers;
    producers.reserve(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        producers.emplace(cgraph->nodes[i], i);
    }
    std::vector<bool> covered(static_cast<size_t>(cgraph->n_nodes), false);
    for (const int index : candidate.node_indices) {
        if (index < entry || index >= cgraph->n_nodes) {
            return false;
        }
        covered[static_cast<size_t>(index)] = true;
    }
    const auto available = [&](const ggml_tensor * source) {
        if (source == nullptr) {
            return true;
        }
        const auto producer = producers.find(source);
        return producer == producers.end() || producer->second < entry ||
            covered[static_cast<size_t>(producer->second)];
    };
    for (const int index : candidate.node_indices) {
        const ggml_tensor * node = cgraph->nodes[index];
        for (int source = 0; source < GGML_MAX_SRC; ++source) {
            if (!available(node->src[source])) {
                return false;
            }
        }
        if (!available(node->view_src)) {
            return false;
        }
    }
    return true;
}

static bool flagos_tensor_storage_overlaps(
        const ggml_tensor * left, const ggml_tensor * right,
        flagos_resolve_tensor_data_fn resolve = nullptr,
        void * user_data = nullptr) {
    if (left == nullptr || right == nullptr) {
        return false;
    }
    const void * left_data = resolve == nullptr ? left->data : resolve(user_data, left);
    const void * right_data = resolve == nullptr ? right->data : resolve(user_data, right);
    if (left_data == nullptr || right_data == nullptr) {
        return false;
    }
    const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left_data);
    const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right_data);
    const size_t left_size = ggml_nbytes(left);
    const size_t right_size = ggml_nbytes(right);
    if (left_size == 0 || right_size == 0) {
        return false;
    }
    if (left_begin > UINTPTR_MAX - left_size || right_begin > UINTPTR_MAX - right_size) {
        return true;
    }
    return left_begin < right_begin + right_size && right_begin < left_begin + left_size;
}

// A non-contiguous fusion runs at its first node.  Consequently every output
// that the provider is allowed to materialize can be written earlier than in
// the original schedule.  GGML's allocator may reuse that future output range
// for an unrelated temporary whose lifetime ends before the original producer.
// Reject such a candidate before asking a provider to lower it; otherwise an
// early fused write can silently clobber an intervening node or one of its
// inputs.  Checking every covered output is intentionally conservative because
// the common execution ABI does not limit a provider to required outputs.
static bool flagos_candidate_early_writes_are_safe(
        const ggml_cgraph * cgraph, const flagos_pattern_candidate & candidate,
        flagos_resolve_tensor_data_fn resolve = nullptr,
        void * user_data = nullptr) {
    if (cgraph == nullptr || candidate.node_indices.empty()) {
        return false;
    }
    const int entry = candidate.node_indices.front();
    if (entry < 0 || entry >= cgraph->n_nodes) {
        return false;
    }
    std::vector<bool> covered(static_cast<size_t>(cgraph->n_nodes), false);
    for (const int index : candidate.node_indices) {
        if (index < entry || index >= cgraph->n_nodes || covered[static_cast<size_t>(index)]) {
            return false;
        }
        covered[static_cast<size_t>(index)] = true;
    }
    for (const int producer_index : candidate.node_indices) {
        const ggml_tensor * early_output = cgraph->nodes[producer_index];
        if (early_output == nullptr) {
            return false;
        }
        for (int index = entry; index < producer_index; ++index) {
            if (covered[static_cast<size_t>(index)]) {
                continue;
            }
            const ggml_tensor * intervening = cgraph->nodes[index];
            if (intervening == nullptr ||
                flagos_tensor_storage_overlaps(early_output, intervening, resolve, user_data) ||
                flagos_tensor_storage_overlaps(early_output, intervening->view_src, resolve, user_data)) {
                return false;
            }
            for (int source = 0; source < GGML_MAX_SRC; ++source) {
                if (flagos_tensor_storage_overlaps(
                        early_output, intervening->src[source], resolve, user_data)) {
                    return false;
                }
            }
        }
    }
    return true;
}

// A later node's external input is also read early when a non-contiguous
// pattern runs at its first node. Tensor dependencies alone are insufficient:
// GGML may give an intervening output and that later input distinct tensor
// identities backed by the same storage. The original schedule writes before
// reading; the fused schedule would otherwise observe stale data.
static bool flagos_candidate_early_reads_are_safe(
        const ggml_cgraph * cgraph, const flagos_pattern_candidate & candidate,
        flagos_resolve_tensor_data_fn resolve = nullptr,
        void * user_data = nullptr) {
    if (cgraph == nullptr || candidate.node_indices.empty()) {
        return false;
    }
    const int entry = candidate.node_indices.front();
    if (entry < 0 || entry >= cgraph->n_nodes) {
        return false;
    }
    std::unordered_map<const ggml_tensor *, int> producers;
    producers.reserve(cgraph->n_nodes);
    for (int index = 0; index < cgraph->n_nodes; ++index) {
        if (cgraph->nodes[index] == nullptr) {
            return false;
        }
        producers.emplace(cgraph->nodes[index], index);
    }
    std::vector<bool> covered(static_cast<size_t>(cgraph->n_nodes), false);
    for (const int index : candidate.node_indices) {
        if (index < entry || index >= cgraph->n_nodes || covered[static_cast<size_t>(index)]) {
            return false;
        }
        covered[static_cast<size_t>(index)] = true;
    }
    const auto external_to_candidate = [&producers, &covered](const ggml_tensor * source) {
        const auto producer = producers.find(source);
        return source != nullptr &&
            (producer == producers.end() || !covered[static_cast<size_t>(producer->second)]);
    };
    const auto crosses_write = [&](const ggml_tensor * source, int consumer_index) {
        if (!external_to_candidate(source)) {
            return false;
        }
        for (int index = entry + 1; index < consumer_index; ++index) {
            if (!covered[static_cast<size_t>(index)] &&
                flagos_tensor_storage_overlaps(
                    source, cgraph->nodes[index], resolve, user_data)) {
                return true;
            }
        }
        return false;
    };
    for (const int consumer_index : candidate.node_indices) {
        if (consumer_index == entry) {
            continue;
        }
        const ggml_tensor * consumer = cgraph->nodes[consumer_index];
        for (int source = 0; source < GGML_MAX_SRC; ++source) {
            if (crosses_write(consumer->src[source], consumer_index)) {
                return false;
            }
        }
        if (crosses_write(consumer->view_src, consumer_index)) {
            return false;
        }
    }
    return true;
}

static std::vector<flagos_pattern_candidate> flagos_enumerate_candidates(const ggml_cgraph * cgraph) {
    std::vector<flagos_pattern_candidate> candidates;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_FLASH_ATTN_EXT && node->src[4] == nullptr) {
            flagos_pattern_candidate candidate;
            // FLASH_ATTN_EXT is a composite GGML operator in both cases, but
            // its query length determines which provider ABI is appropriate:
            // the decode kernel has one query row, while prefill kernels need
            // a tiled query dimension.  Keep both in the common catalog so a
            // provider can independently accept either variant.
            candidate.id = node->src[0] == nullptr || node->src[0]->ne[1] == 1
                ? flagos_pattern_id::flash_attn_decode
                : flagos_pattern_id::flash_attn_prefill;
            candidate.scope = flagos_pattern_scope(candidate.id);
            candidate.node_indices = { i };
            candidate.eliminated_launches = 1;
            candidates.push_back(candidate);
        }

        if (node->op == GGML_OP_GATED_DELTA_NET) {
            int copy_index = -1;
            for (int j = i + 1; j < cgraph->n_nodes; ++j) {
                if (flagos_is_view_or_noop(cgraph->nodes[j])) {
                    continue;
                }
                if (cgraph->nodes[j]->op == GGML_OP_CPY) {
                    copy_index = j;
                }
                break;
            }
            if (copy_index >= 0) {
                const ggml_tensor * copy = cgraph->nodes[copy_index];
                const ggml_tensor * snapshot_view = copy->src[0];
                // Zero-work views are not guaranteed to appear in cgraph->nodes.
                // llama.cpp's Qwen3.5 graph keeps the snapshot view only as the
                // CPY source, so validate the tensor edge itself and cover the
                // view node only when the graph materializes one.
                const bool cache_copy =
                    flagos_is_gated_delta_net_cache_copy(node, snapshot_view, copy);
                if (cache_copy) {
                    flagos_pattern_candidate candidate;
                    candidate.id = node->src[2]->ne[2] == 1
                        ? flagos_pattern_id::gated_delta_net_decode
                        : flagos_pattern_id::gated_delta_net_prefill;
                    candidate.scope = flagos_pattern_scope(candidate.id);
                    candidate.node_indices.push_back(i);
                    for (int j = i + 1; j < copy_index; ++j) {
                        // GGML canonicalizes nested views to their backing tensor.
                        // Cover every alias of the GDN allocation so the provider's
                        // promise to materialize the full GDN output also satisfies
                        // any externally consumed attention or snapshot view.  A
                        // mere src[0] dependency is not sufficient: a zero-launch
                        // node may still own distinct storage.
                        if (cgraph->nodes[j]->view_src == node ||
                            cgraph->nodes[j] == copy->src[1]) {
                            candidate.node_indices.push_back(j);
                        }
                    }
                    candidate.node_indices.push_back(copy_index);
                    candidate.eliminated_read_bytes = flagos_tensor_bytes(copy->src[0]);
                    candidate.eliminated_launches = 1;
                    candidates.push_back(candidate);
                }
            }
        }

        // Transformer KV-cache update: RoPE output is flattened through a
        // view and then written into the cache by SET_ROWS.  The view carries
        // no computation, so the provider may replace all three nodes with a
        // single rope-and-store kernel.
        if (node->op == GGML_OP_ROPE) {
            int view_index = -1;
            int set_rows_index = -1;
            const ggml_tensor * view = nullptr;
            for (int j = i + 1; j < cgraph->n_nodes && view_index < 0; ++j) {
                if (flagos_is_graph_barrier(cgraph->nodes[j])) {
                    break;
                }
                if (cgraph->nodes[j]->op == GGML_OP_VIEW && cgraph->nodes[j]->src[0] == node) {
                    view_index = j;
                }
            }
            if (view_index >= 0) {
                view = cgraph->nodes[view_index];
                for (int j = view_index + 1; j < cgraph->n_nodes; ++j) {
                    if (cgraph->nodes[j]->op == GGML_OP_SET_ROWS && cgraph->nodes[j]->src[0] == view) {
                        set_rows_index = j;
                        break;
                    }
                    if (flagos_is_graph_barrier(cgraph->nodes[j])) {
                        break;
                    }
                }
            }
            if (view_index >= 0 && set_rows_index >= 0) {
                flagos_pattern_candidate candidate;
                candidate.id = flagos_pattern_id::rope_kv_store;
                candidate.scope = flagos_pattern_scope(candidate.id);
                candidate.node_indices = { i, view_index, set_rows_index };
                candidate.eliminated_read_bytes = flagos_tensor_bytes(node) + flagos_tensor_bytes(view);
                candidate.eliminated_launches = 2;
                candidates.push_back(candidate);
            }
        }

        if (node->op == GGML_OP_ADD) {
            int norm_index = -1;
            int mul_index = -1;
            for (int j = i + 1; j < cgraph->n_nodes && norm_index < 0; ++j) {
                const ggml_tensor * norm = cgraph->nodes[j];
                if (norm->op == GGML_OP_RMS_NORM && norm->src[0] == node) {
                    norm_index = j;
                }
                if (flagos_is_graph_barrier(norm)) {
                    break;
                }
            }
            if (norm_index >= 0) {
                const ggml_tensor * norm = cgraph->nodes[norm_index];
                for (int j = norm_index + 1; j < cgraph->n_nodes; ++j) {
                    if (flagos_is_mul_of(cgraph->nodes[j], norm)) {
                        mul_index = j;
                        break;
                    }
                    if (flagos_is_graph_barrier(cgraph->nodes[j])) {
                        break;
                    }
                }
            }
            if (norm_index >= 0 && mul_index >= 0) {
                const ggml_tensor * norm = cgraph->nodes[norm_index];
                flagos_pattern_candidate candidate;
                candidate.id = flagos_pattern_id::add_rms_norm_mul;
                candidate.scope = flagos_pattern_scope(candidate.id);
                candidate.node_indices = { i, norm_index, mul_index };
                candidate.eliminated_read_bytes = flagos_tensor_bytes(node) + flagos_tensor_bytes(norm);
                candidate.eliminated_launches = 2;
                candidates.push_back(candidate);
            }
        }

        if (node->op == GGML_OP_RMS_NORM) {
            int mul_index = -1;
            for (int j = i + 1; j < cgraph->n_nodes; ++j) {
                if (flagos_is_mul_of(cgraph->nodes[j], node)) {
                    mul_index = j;
                    break;
                }
                if (flagos_is_graph_barrier(cgraph->nodes[j])) {
                    break;
                }
            }
            if (mul_index >= 0) {
                flagos_pattern_candidate candidate;
                candidate.id = flagos_pattern_id::rms_norm_mul;
                candidate.scope = flagos_pattern_scope(candidate.id);
                candidate.node_indices = { i, mul_index };
                candidate.eliminated_read_bytes = flagos_tensor_bytes(node);
                candidate.eliminated_launches = 1;
                candidates.push_back(candidate);

                const ggml_tensor * mul = cgraph->nodes[mul_index];
                int rope_index = -1;
                for (int j = mul_index + 1; j < cgraph->n_nodes; ++j) {
                    const ggml_tensor * rope = cgraph->nodes[j];
                    if (rope->op == GGML_OP_ROPE && rope->src[0] == mul) {
                        rope_index = j;
                        break;
                    }
                    if (flagos_is_graph_barrier(rope)) {
                        break;
                    }
                }
                if (rope_index >= 0) {
                    const ggml_tensor * rope = cgraph->nodes[rope_index];
                    flagos_pattern_candidate rope_candidate;
                    rope_candidate.id = flagos_pattern_id::rms_norm_mul_rope;
                    rope_candidate.scope = flagos_pattern_scope(rope_candidate.id);
                    rope_candidate.node_indices = { i, mul_index, rope_index };
                    rope_candidate.eliminated_read_bytes =
                        flagos_tensor_bytes(node) + flagos_tensor_bytes(mul);
                    rope_candidate.eliminated_write_bytes = flagos_tensor_bytes(node);
                    rope_candidate.eliminated_launches = 2;
                    candidates.push_back(rope_candidate);

                    int view_index = -1;
                    int set_rows_index = -1;
                    const ggml_tensor * view = nullptr;
                    for (int j = rope_index + 1; j < cgraph->n_nodes && view_index < 0; ++j) {
                        if (flagos_is_graph_barrier(cgraph->nodes[j])) {
                            break;
                        }
                        if (cgraph->nodes[j]->op == GGML_OP_VIEW && cgraph->nodes[j]->src[0] == rope) {
                            view_index = j;
                        }
                    }
                    if (view_index >= 0) {
                        view = cgraph->nodes[view_index];
                        for (int j = view_index + 1; j < cgraph->n_nodes; ++j) {
                            if (cgraph->nodes[j]->op == GGML_OP_SET_ROWS && cgraph->nodes[j]->src[0] == view) {
                                set_rows_index = j;
                                break;
                            }
                            if (flagos_is_graph_barrier(cgraph->nodes[j])) {
                                break;
                            }
                        }
                    }
                    if (view_index >= 0 && set_rows_index >= 0) {
                        flagos_pattern_candidate store_candidate;
                        store_candidate.id = flagos_pattern_id::rms_norm_mul_rope_kv_store;
                        store_candidate.scope = flagos_pattern_scope(store_candidate.id);
                        store_candidate.node_indices = {
                            i, mul_index, rope_index, view_index, set_rows_index };
                        store_candidate.eliminated_read_bytes =
                            flagos_tensor_bytes(node) + flagos_tensor_bytes(mul) +
                            flagos_tensor_bytes(rope) + flagos_tensor_bytes(view);
                        store_candidate.eliminated_write_bytes =
                            flagos_tensor_bytes(node) + flagos_tensor_bytes(mul) +
                            flagos_tensor_bytes(rope);
                        store_candidate.eliminated_launches = 4;
                        candidates.push_back(store_candidate);
                    }
                }
            }
        }

        if (node->op == GGML_OP_SSM_CONV) {
            int activation_index = -1;
            for (int j = i + 1; j < cgraph->n_nodes; ++j) {
                const ggml_tensor * activation = cgraph->nodes[j];
                if (activation->op == GGML_OP_UNARY && activation->src[0] == node &&
                    ggml_get_unary_op(activation) == GGML_UNARY_OP_SILU) {
                    activation_index = j;
                    break;
                }
                if (flagos_is_graph_barrier(activation)) {
                    break;
                }
            }
            if (activation_index >= 0) {
                flagos_pattern_candidate candidate;
                candidate.id = flagos_pattern_id::ssm_conv_silu;
                candidate.scope = flagos_pattern_scope(candidate.id);
                candidate.node_indices = { i, activation_index };
                candidate.eliminated_read_bytes = flagos_tensor_bytes(node);
                candidate.eliminated_launches = 1;
                candidates.push_back(candidate);
            }
        }

        // Gated residual/output paths commonly materialize an activation first
        // and multiply it by an already available tensor. These are the same
        // three unary variants fused by the CUDA backend. Keep any producer
        // outside this candidate: a side-plan executes at its first covered
        // node, so only inputs available at the unary entry may be consumed by
        // the fused lowering.
        const bool is_gated_unary = node->op == GGML_OP_UNARY &&
            (ggml_get_unary_op(node) == GGML_UNARY_OP_SILU ||
             ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID ||
             ggml_get_unary_op(node) == GGML_UNARY_OP_SOFTPLUS);
        if (is_gated_unary) {
            int mul_index = -1;
            for (int j = i + 1; j < cgraph->n_nodes; ++j) {
                if (flagos_is_mul_of(cgraph->nodes[j], node)) {
                    mul_index = j;
                    break;
                }
                if (flagos_is_graph_barrier(cgraph->nodes[j])) {
                    break;
                }
            }
            if (mul_index >= 0) {
                flagos_pattern_candidate candidate;
                candidate.id = flagos_pattern_id::attention_output_gate;
                candidate.scope = flagos_pattern_scope(candidate.id);
                candidate.node_indices = { i, mul_index };
                candidate.eliminated_read_bytes = flagos_tensor_bytes(node);
                candidate.eliminated_launches = 1;
                candidates.push_back(candidate);
            }
        }

        // Parallel FFN projections followed by a SwiGLU.  Providers may
        // lower this as a true fused projection/activation kernel, or decline
        // it and execute the individual nodes through their normal paths.
        if (node->op == GGML_OP_MUL_MAT) {
            int gate_index = -1;
            int glu_index = -1;
            for (int j = i + 1; j < cgraph->n_nodes; ++j) {
                const ggml_tensor * glu = cgraph->nodes[j];
                if (glu->op == GGML_OP_GLU && ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
                    (glu->src[0] == node || glu->src[1] == node)) {
                    const ggml_tensor * other = glu->src[0] == node ? glu->src[1] : glu->src[0];
                    for (int producer = i + 1; producer < j; ++producer) {
                        if (other != nullptr && other->op == GGML_OP_MUL_MAT &&
                            cgraph->nodes[producer] == other) {
                            gate_index = producer;
                            glu_index = j;
                            break;
                        }
                    }
                    if (glu_index >= 0) {
                        break;
                    }
                }
                if (flagos_is_graph_barrier(glu)) {
                    break;
                }
            }
            if (gate_index >= 0 && glu_index >= 0) {
                const ggml_tensor * glu = cgraph->nodes[glu_index];
                flagos_pattern_candidate candidate;
                candidate.id = flagos_pattern_id::ffn_swiglu;
                candidate.scope = flagos_pattern_scope(candidate.id);
                candidate.node_indices = { i, gate_index, glu_index };
                candidate.eliminated_read_bytes = flagos_tensor_bytes(glu);
                candidate.eliminated_launches = 2;
                candidates.push_back(candidate);

                // A private GLU output lets the provider choose its scratch format.
                int down_index = -1;
                for (int j = glu_index + 1; j < cgraph->n_nodes; ++j) {
                    const ggml_tensor * down = cgraph->nodes[j];
                    if (down->op == GGML_OP_MUL_MAT && down->src[1] == glu) {
                        down_index = j;
                        break;
                    }
                    if (flagos_is_graph_barrier(down)) {
                        break;
                    }
                }
                if (down_index >= 0 &&
                    flagos_tensor_is_private_to_consumer(cgraph, glu, down_index)) {
                    flagos_pattern_candidate down_candidate;
                    down_candidate.id = flagos_pattern_id::ffn_swiglu_down;
                    down_candidate.scope = flagos_pattern_scope(down_candidate.id);
                    down_candidate.node_indices = { i, gate_index, glu_index, down_index };
                    // Equal scores let the node-count tie-break prefer this superset.
                    down_candidate.eliminated_read_bytes = flagos_tensor_bytes(glu);
                    down_candidate.eliminated_launches = 2;
                    candidates.push_back(down_candidate);
                }
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

template <typename Visitor>
static bool flagos_visit_graph_bindings(
        const ggml_cgraph * cgraph,
        const flagos_graph_plan & plan,
        flagos_resolve_tensor_data_fn resolve,
        void * user_data,
        Visitor visitor) {
    if (cgraph == nullptr || cgraph->n_nodes < 0 ||
        static_cast<size_t>(cgraph->n_nodes) != plan.canonical_nodes.size() ||
        (cgraph->n_nodes > 0 && cgraph->nodes == nullptr)) {
        return false;
    }
    for (const auto & slot : plan.binding_slots) {
        if (slot.node_index < 0 || slot.node_index >= cgraph->n_nodes ||
            slot.source_index < -1 || slot.source_index >= GGML_MAX_SRC) {
            return false;
        }
        const ggml_tensor * node = cgraph->nodes[slot.node_index];
        if (node == nullptr) {
            return false;
        }
        const ggml_tensor * tensor = slot.source_index < 0
            ? node : node->src[slot.source_index];
        if (slot.view_source && tensor != nullptr) {
            tensor = tensor->view_src;
        }
        if (tensor == nullptr) {
            return false;
        }
        const void * data = resolve == nullptr ? tensor->data : resolve(user_data, tensor);
        if (data == nullptr || !visitor(reinterpret_cast<uintptr_t>(data))) {
            return false;
        }
    }
    return true;
}

bool flagos_graph_binding_fingerprint(
        const ggml_cgraph * cgraph,
        const flagos_graph_plan & plan,
        uint64_t * fingerprint,
        flagos_resolve_tensor_data_fn resolve,
        void * user_data) {
    if (fingerprint == nullptr) {
        return false;
    }
    *fingerprint = 0;
    uint64_t result = 1469598103934665603ULL;
    if (!flagos_visit_graph_bindings(cgraph, plan, resolve, user_data,
            [&result](uintptr_t pointer) {
                result ^= static_cast<uint64_t>(pointer);
                result *= 1099511628211ULL;
                return true;
            })) {
        return false;
    }
    *fingerprint = result;
    return true;
}

bool flagos_graph_binding_snapshot_create(
        const ggml_cgraph * cgraph,
        const flagos_graph_plan & plan,
        flagos_graph_binding_snapshot * snapshot,
        flagos_resolve_tensor_data_fn resolve,
        void * user_data) {
    if (snapshot == nullptr) {
        return false;
    }
    snapshot->pointers.clear();
    snapshot->pointers.reserve(plan.binding_slots.size());
    uint64_t fingerprint = 1469598103934665603ULL;
    if (!flagos_visit_graph_bindings(cgraph, plan, resolve, user_data,
            [&snapshot, &fingerprint](uintptr_t pointer) {
                snapshot->pointers.push_back(pointer);
                fingerprint ^= static_cast<uint64_t>(pointer);
                fingerprint *= 1099511628211ULL;
                return true;
            })) {
        snapshot->pointers.clear();
        snapshot->fingerprint = 0;
        return false;
    }
    snapshot->fingerprint = fingerprint;
    return true;
}

bool flagos_graph_binding_snapshot_matches(
        const ggml_cgraph * cgraph,
        const flagos_graph_plan & plan,
        const flagos_graph_binding_snapshot & snapshot,
        flagos_resolve_tensor_data_fn resolve,
        void * user_data) {
    if (snapshot.pointers.size() != plan.binding_slots.size()) {
        return false;
    }
    size_t index = 0;
    const bool matched = flagos_visit_graph_bindings(cgraph, plan, resolve, user_data,
        [&snapshot, &index](uintptr_t pointer) {
            return index < snapshot.pointers.size() &&
                snapshot.pointers[index++] == pointer;
        });
    return matched && index == snapshot.pointers.size();
}

static std::unique_ptr<flagos_graph_plan> flagos_build_graph_plan_from_canonical(
        const ggml_cgraph * cgraph,
        flagos_query_lowering_fn query_lowering,
        void * user_data,
        const std::vector<flagos_canonical_node> & canonical_nodes,
        uint64_t structural_fingerprint,
        flagos_resolve_tensor_data_fn resolve_tensor_data) {
    auto plan = std::make_unique<flagos_graph_plan>();
    plan->canonical_nodes = canonical_nodes;
    plan->structural_fingerprint = structural_fingerprint;
    plan->binding_slots.reserve(canonical_nodes.size() * 2);
    for (size_t node = 0; node < canonical_nodes.size(); ++node) {
        plan->binding_slots.push_back({ static_cast<int>(node), -1 });
        if (canonical_nodes[node].output.view_source < -1) {
            plan->binding_slots.push_back({ static_cast<int>(node), -1, true });
        }
        for (int source = 0; source < GGML_MAX_SRC; ++source) {
            const auto & canonical = canonical_nodes[node].sources[source];
            if (canonical.present && canonical.producer < 0) {
                plan->binding_slots.push_back({ static_cast<int>(node), source });
            }
            if (canonical.present && canonical.view_source < -1) {
                plan->binding_slots.push_back({ static_cast<int>(node), source, true });
            }
        }
    }

    struct selected_candidate {
        flagos_pattern_candidate candidate;
        flagos_lowering_choice choice;
    };

    std::vector<selected_candidate> supported;
    if (query_lowering != nullptr) {
        const auto candidates = flagos_enumerate_candidates(cgraph);
        plan->binding_sensitive = !candidates.empty();
        for (auto candidate : candidates) {
            if (!flagos_populate_required_outputs(cgraph, candidate) ||
                !flagos_candidate_inputs_available_at_entry(cgraph, candidate) ||
                !flagos_candidate_early_writes_are_safe(
                    cgraph, candidate, resolve_tensor_data, user_data) ||
                !flagos_candidate_early_reads_are_safe(
                    cgraph, candidate, resolve_tensor_data, user_data)) {
                continue;
            }
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
        if (left.candidate.node_indices.size() != right.candidate.node_indices.size()) {
            return left.candidate.node_indices.size() > right.candidate.node_indices.size();
        }
        if (left.candidate.node_indices != right.candidate.node_indices) {
            return left.candidate.node_indices < right.candidate.node_indices;
        }
        return static_cast<uint32_t>(left.candidate.id) < static_cast<uint32_t>(right.candidate.id);
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

    for (int node_index = 0; node_index < cgraph->n_nodes; ++node_index) {
        const int selected_index = selected_by[node_index];
        if (selected_index >= 0 && selected[selected_index].candidate.node_indices.front() == node_index) {
            const auto & item = selected[selected_index];
            flagos_plan_step step;
            step.kind = flagos_execution_kind::pattern;
            step.candidate = item.candidate;
            step.capture_safe = item.choice.capture_safe;
            step.implementation_id = item.choice.implementation_id;
            plan->steps.push_back(step);
        } else if (selected_index >= 0) {
            // This node is covered by a previously emitted non-contiguous
            // pattern.  It must not be emitted as a direct step as well.
            continue;
        } else {
            flagos_plan_step step;
            step.kind = flagos_execution_kind::direct;
            step.candidate.node_indices = { node_index };
            plan->steps.push_back(step);
        }
    }

    return plan;
}

std::unique_ptr<flagos_graph_plan> flagos_build_graph_plan(
        const ggml_cgraph * cgraph,
        flagos_query_lowering_fn query_lowering,
        void * user_data) {
    const auto canonical_nodes = flagos_canonicalize_graph(cgraph);
    return flagos_build_graph_plan_from_canonical(
        cgraph, query_lowering, user_data, canonical_nodes,
        flagos_fingerprint(canonical_nodes), nullptr);
}

std::unique_ptr<flagos_graph_plan> flagos_build_graph_plan(
        const ggml_cgraph * cgraph,
        const flagos_fusion_interface & interface) {
    const auto canonical_nodes = flagos_canonicalize_graph(cgraph);
    return flagos_build_graph_plan_from_canonical(
        cgraph, interface.query_lowering, interface.user_data, canonical_nodes,
        flagos_fingerprint(canonical_nodes), interface.resolve_tensor_data);
}

struct flagos_graph_plan_cache::entry {
    std::unique_ptr<flagos_graph_plan> plan;
    flagos_query_lowering_fn query_lowering = nullptr;
    flagos_query_lowering_fn validate_lowering = nullptr;
    void * user_data = nullptr;
    uint64_t configuration_id = 0;
    flagos_resolve_tensor_data_fn resolve_tensor_data = nullptr;
    flagos_graph_binding_snapshot validated_bindings;
    bool bindings_captured = false;
    uint64_t last_used = 0;
};

bool flagos_graph_plan_cache::plan_is_valid(entry & entry, const ggml_cgraph * cgraph) {
    if (entry.bindings_captured && flagos_graph_binding_snapshot_matches(
            cgraph, *entry.plan, entry.validated_bindings,
            entry.resolve_tensor_data, entry.user_data)) {
        return true;
    }
    // A graph with no recognized fusion candidate cannot change its plan when
    // scheduler buffers rotate. Direct steps consume the current tensor
    // bindings at execution time, so rebuilding the canonical plan adds no
    // safety and can thrash on multi-buffer decode schedules.
    if (!entry.plan->binding_sensitive) {
        return true;
    }
    if (entry.validate_lowering == nullptr) {
        if (entry.bindings_captured) {
            return false;
        }
        flagos_graph_binding_snapshot bindings;
        const bool bindings_available = flagos_graph_binding_snapshot_create(
            cgraph, *entry.plan, &bindings,
            entry.resolve_tensor_data, entry.user_data);
        // Before the scheduler assigns storage, structural reuse is valid.
        // Once bindings become available, force one rebuild so the new plan
        // captures addresses and rechecks alias-sensitive fusion hazards.
        return !bindings_available;
    }
    bool has_pattern = false;
    for (const auto & step : entry.plan->steps) {
        if (step.kind != flagos_execution_kind::pattern) {
            continue;
        }
        has_pattern = true;
        if (!flagos_candidate_early_writes_are_safe(
                cgraph, step.candidate, entry.resolve_tensor_data, entry.user_data) ||
            !flagos_candidate_early_reads_are_safe(
                cgraph, step.candidate, entry.resolve_tensor_data, entry.user_data)) {
            return false;
        }
        const flagos_lowering_choice choice = entry.validate_lowering(
            entry.user_data, cgraph, step.candidate);
        if (!choice.supported || choice.capture_safe != step.capture_safe ||
            choice.implementation_id != step.implementation_id) {
            return false;
        }
    }
    if (!has_pattern) {
        return false;
    }
    entry.bindings_captured = flagos_graph_binding_snapshot_create(
        cgraph, *entry.plan, &entry.validated_bindings,
        entry.resolve_tensor_data, entry.user_data);
    return entry.bindings_captured;
}

struct flagos_graph_plan_cache::scratch {
    flagos_producer_map producers;
    flagos_tensor_identity_map identities;
    std::vector<const ggml_tensor *> producer_nodes;
    std::vector<flagos_canonical_node> canonical_nodes;
};

flagos_graph_plan_cache::flagos_graph_plan_cache(size_t capacity) :
    capacity_(std::max<size_t>(capacity, 1)),
    scratch_(std::make_unique<scratch>()) {
}

flagos_graph_plan_cache::~flagos_graph_plan_cache() = default;

const flagos_graph_plan & flagos_graph_plan_cache::get_or_create(
        const ggml_cgraph * cgraph,
        flagos_query_lowering_fn query_lowering,
        void * user_data,
        bool * created,
        uint64_t configuration_id,
        flagos_query_lowering_fn validate_lowering,
        flagos_resolve_tensor_data_fn resolve_tensor_data) {
    ++tick_;
    if (created != nullptr) {
        *created = false;
    }

    // Match structure directly because scheduler UIDs do not identify shapes or edges.
    if (last_plan_ != nullptr && last_entry_->query_lowering == query_lowering &&
        last_entry_->validate_lowering == validate_lowering &&
        last_entry_->resolve_tensor_data == resolve_tensor_data &&
        last_entry_->user_data == user_data && last_entry_->configuration_id == configuration_id) {
        flagos_prepare_producers(cgraph, scratch_->producers, scratch_->producer_nodes);
        if (flagos_graph_matches_canonical(
                cgraph, scratch_->producers, scratch_->identities, last_plan_->canonical_nodes) &&
            plan_is_valid(*last_entry_, cgraph)) {
            last_entry_->last_used = tick_;
            ++hits_;
            return *last_plan_;
        }
        for (const auto & item : entries_) {
            if (item.get() == last_entry_ || item->query_lowering != query_lowering ||
                item->validate_lowering != validate_lowering ||
                item->resolve_tensor_data != resolve_tensor_data ||
                item->user_data != user_data || item->configuration_id != configuration_id ||
                !flagos_graph_matches_canonical(
                    cgraph, scratch_->producers, scratch_->identities, item->plan->canonical_nodes) ||
                !plan_is_valid(*item, cgraph)) {
                continue;
            }
            item->last_used = tick_;
            last_entry_ = item.get();
            last_plan_ = item->plan.get();
            ++hits_;
            return *last_plan_;
        }
    }
    flagos_canonicalize_graph(
        cgraph, scratch_->producers, scratch_->identities, scratch_->producer_nodes,
        scratch_->canonical_nodes);
    const auto & canonical_nodes = scratch_->canonical_nodes;
    const uint64_t fingerprint = flagos_fingerprint(canonical_nodes);
    for (const auto & item : entries_) {
        if (item->query_lowering == query_lowering && item->validate_lowering == validate_lowering &&
            item->resolve_tensor_data == resolve_tensor_data &&
            item->user_data == user_data &&
            item->configuration_id == configuration_id &&
            item->plan->structural_fingerprint == fingerprint && item->plan->canonical_nodes == canonical_nodes &&
            plan_is_valid(*item, cgraph)) {
            item->last_used = tick_;
            last_entry_ = item.get();
            last_plan_ = item->plan.get();
            ++hits_;
            return *item->plan;
        }
    }

    if (entries_.size() >= capacity_) {
        const auto lru = std::min_element(entries_.begin(), entries_.end(), [](const auto & left, const auto & right) {
            return left->last_used < right->last_used;
        });
        if (lru->get() == last_entry_) {
            last_entry_ = nullptr;
            last_plan_ = nullptr;
        }
        entries_.erase(lru);
        ++evictions_;
    }

    auto item = std::make_unique<entry>();
    item->plan = flagos_build_graph_plan_from_canonical(
        cgraph, query_lowering, user_data, canonical_nodes, fingerprint, resolve_tensor_data);
    item->query_lowering = query_lowering;
    item->validate_lowering = validate_lowering;
    item->user_data = user_data;
    item->configuration_id = configuration_id;
    item->resolve_tensor_data = resolve_tensor_data;
    item->bindings_captured = flagos_graph_binding_snapshot_create(
        cgraph, *item->plan, &item->validated_bindings,
        resolve_tensor_data, user_data);
    item->last_used = tick_;
    last_entry_ = item.get();
    last_plan_ = item->plan.get();
    entries_.push_back(std::move(item));
    ++misses_;
    if (created != nullptr) {
        *created = true;
    }
    return *last_plan_;
}

const flagos_graph_plan & flagos_graph_plan_cache::get_or_create(
        const ggml_cgraph * cgraph,
        const flagos_fusion_interface & interface,
        bool * created) {
    return get_or_create(
        cgraph, interface.query_lowering, interface.user_data,
        created, interface.configuration_id, interface.validate_lowering,
        interface.resolve_tensor_data);
}

void flagos_graph_plan_cache::clear() {
    entries_.clear();
    last_entry_ = nullptr;
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
