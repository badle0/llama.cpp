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
        (op->src[0]->type != GGML_TYPE_Q4_K && op->src[0]->type != GGML_TYPE_Q6_K) ||
        op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
        op->src[0]->ne[0] <= 0 || op->src[0]->ne[0] % 256 != 0 ||
        op->src[0]->ne[0] != op->src[1]->ne[0] || op->src[0]->ne[1] <= 0 ||
        op->src[1]->ne[1] <= 0 || op->src[0]->ne[2] != 1 || op->src[0]->ne[3] != 1 ||
        op->src[1]->ne[2] != 1 || op->src[1]->ne[3] != 1 ||
        op->ne[0] != op->src[0]->ne[1] || op->ne[1] != op->src[1]->ne[1] ||
        op->ne[2] != 1 || op->ne[3] != 1 || !ggml_is_contiguous(op->src[0]) ||
        !ggml_is_contiguous(op->src[1]) || !ggml_is_contiguous(op)) {
        return false;
    }
    signature->weight_kind = op->src[0]->type == GGML_TYPE_Q4_K
        ? flagos_quantized_matmul_kind::q4_k : flagos_quantized_matmul_kind::q6_k;
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
        case flagos_pattern_id::add_rms_norm_mul:
        case flagos_pattern_id::ssm_conv_silu:
        case flagos_pattern_id::gated_delta_net_decode:
        case flagos_pattern_id::gated_delta_net_prefill:
        case flagos_pattern_id::gated_rms_norm:
        case flagos_pattern_id::ffn_swiglu:
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
        case GGML_OP_CPY:
        case GGML_OP_SET_ROWS:
            return true;
        default:
            return false;
    }
}

// A multi-node lowering may only elide intermediate tensors that have no
// consumers outside the candidate.  Without this check, a graph such as
// RMS_NORM -> {MUL, ADD} could select RMS_NORM+MUL, skip materializing the
// norm output, and leave the ADD branch reading uninitialized data.
static bool flagos_candidate_has_no_intermediate_fanout(
        const ggml_cgraph * cgraph, const flagos_pattern_candidate & candidate) {
    if (cgraph == nullptr || candidate.node_indices.size() <= 1) {
        return true;
    }
    std::vector<bool> covered(static_cast<size_t>(cgraph->n_nodes), false);
    for (const int index : candidate.node_indices) {
        if (index < 0 || index >= cgraph->n_nodes) {
            return false;
        }
        covered[static_cast<size_t>(index)] = true;
    }
    // The terminal node's output is the externally visible fused result.  All
    // earlier outputs must be consumed only by nodes in this candidate.
    for (size_t position = 0; position + 1 < candidate.node_indices.size(); ++position) {
        const int producer_index = candidate.node_indices[position];
        const ggml_tensor * producer = cgraph->nodes[producer_index];
        for (int consumer_index = 0; consumer_index < cgraph->n_nodes; ++consumer_index) {
            if (consumer_index == producer_index || covered[static_cast<size_t>(consumer_index)]) {
                continue;
            }
            const ggml_tensor * consumer = cgraph->nodes[consumer_index];
            for (int source_index = 0; source_index < GGML_MAX_SRC; ++source_index) {
                if (consumer->src[source_index] == producer) {
                    return false;
                }
            }
            if (consumer->view_src == producer) {
                return false;
            }
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

        // Qwen-style KV-cache update: RoPE output is flattened through a
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

        // Parallel FFN projections followed by a SwiGLU.  Providers may
        // lower this as a true fused projection/activation kernel, or decline
        // it and execute the individual nodes through their normal paths.
        if (node->op == GGML_OP_MUL_MAT) {
            int gate_index = -1;
            int glu_index = -1;
            for (int j = i + 1; j < cgraph->n_nodes && gate_index < 0; ++j) {
                if (cgraph->nodes[j]->op == GGML_OP_MUL_MAT) {
                    gate_index = j;
                }
                if (flagos_is_graph_barrier(cgraph->nodes[j])) {
                    break;
                }
            }
            if (gate_index >= 0) {
                const ggml_tensor * gate = cgraph->nodes[gate_index];
                for (int j = gate_index + 1; j < cgraph->n_nodes; ++j) {
                    const ggml_tensor * glu = cgraph->nodes[j];
                    if (glu->op == GGML_OP_GLU && ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
                        ((glu->src[0] == node && glu->src[1] == gate) ||
                         (glu->src[0] == gate && glu->src[1] == node))) {
                        glu_index = j;
                        break;
                    }
                    if (flagos_is_graph_barrier(glu)) {
                        break;
                    }
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
            if (!flagos_candidate_has_no_intermediate_fanout(cgraph, candidate) ||
                !flagos_candidate_inputs_available_at_entry(cgraph, candidate)) {
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
        const flagos_fusion_interface & interface) {
    return flagos_build_graph_plan(cgraph, interface.query_lowering, interface.user_data);
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

    const auto canonical_nodes = flagos_canonicalize_graph(cgraph);
    const uint64_t fingerprint = flagos_fingerprint(canonical_nodes);
    // GGML schedulers may reuse a graph UID while changing tensor shapes or
    // strides between prompt chunks (for example, a 512-token ubatch followed
    // by another chunk with a longer KV length).  UID alone therefore cannot
    // prove structural identity.  Validate the canonical graph before taking
    // the hot last-plan path; otherwise a plan can retain stale fusion choices
    // and launch an ABI for the previous shape.
    if (cgraph->uid != 0 && cgraph->uid == last_uid_ && last_plan_ != nullptr &&
        last_plan_->structural_fingerprint == fingerprint &&
        last_plan_->canonical_nodes == canonical_nodes) {
        ++hits_;
        return *last_plan_;
    }
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

const flagos_graph_plan & flagos_graph_plan_cache::get_or_create(
        const ggml_cgraph * cgraph,
        const flagos_fusion_interface & interface,
        bool * created) {
    return get_or_create(cgraph, interface.query_lowering, interface.user_data, created);
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
