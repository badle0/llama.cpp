#include "flagos-graph-plan.h"

#include "../../ggml-impl.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>

static void check(bool condition, const char * expression, int line) {
    if (!condition) {
        std::fprintf(stderr, "check failed at line %d: %s\n", line, expression);
        std::abort();
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

struct test_graph {
    ggml_tensor input {};
    ggml_tensor residual {};
    ggml_tensor weight {};
    ggml_tensor add {};
    ggml_tensor norm {};
    ggml_tensor mul {};
    ggml_tensor * nodes[3] {};
    ggml_cgraph graph {};
};

static void init_tensor(ggml_tensor & tensor, ggml_op op, int64_t columns, int64_t rows = 1) {
    tensor.type = GGML_TYPE_F32;
    tensor.op = op;
    tensor.flags = GGML_TENSOR_FLAG_COMPUTE;
    tensor.ne[0] = columns;
    tensor.ne[1] = rows;
    tensor.ne[2] = 1;
    tensor.ne[3] = 1;
    tensor.nb[0] = sizeof(float);
    tensor.nb[1] = columns * sizeof(float);
    tensor.nb[2] = rows * tensor.nb[1];
    tensor.nb[3] = tensor.nb[2];
}

static void make_norm_mul_graph(test_graph & result, bool with_add) {
    init_tensor(result.input, GGML_OP_NONE, 1536);
    init_tensor(result.residual, GGML_OP_NONE, 1536);
    init_tensor(result.weight, GGML_OP_NONE, 1536);
    init_tensor(result.add, GGML_OP_ADD, 1536);
    init_tensor(result.norm, GGML_OP_RMS_NORM, 1536);
    init_tensor(result.mul, GGML_OP_MUL, 1536);
    float eps = 1e-6f;
    std::memcpy(result.norm.op_params, &eps, sizeof(eps));

    result.add.src[0] = &result.input;
    result.add.src[1] = &result.residual;
    result.norm.src[0] = with_add ? &result.add : &result.input;
    result.mul.src[0] = &result.norm;
    result.mul.src[1] = &result.weight;

    result.nodes[0] = with_add ? &result.add : &result.norm;
    result.nodes[1] = with_add ? &result.norm : &result.mul;
    result.nodes[2] = with_add ? &result.mul : nullptr;
    result.graph.n_nodes = with_add ? 3 : 2;
    result.graph.nodes = result.nodes;
}

static flagos_lowering_choice query_pattern(
        void * user_data,
        const ggml_cgraph *,
        const flagos_pattern_candidate & candidate) {
    const auto supported = *static_cast<const flagos_pattern_id *>(user_data);
    flagos_lowering_choice result;
    result.supported = candidate.id == supported;
    result.capture_safe = result.supported;
    result.implementation_id = result.supported ? 17 : 0;
    return result;
}

static flagos_lowering_choice query_pattern_not_capture_safe(
        void * user_data,
        const ggml_cgraph * cgraph,
        const flagos_pattern_candidate & candidate) {
    auto result = query_pattern(user_data, cgraph, candidate);
    result.capture_safe = false;
    return result;
}

static flagos_lowering_choice query_pattern_terminal_only(
        void * user_data,
        const ggml_cgraph * cgraph,
        const flagos_pattern_candidate & candidate) {
    auto result = query_pattern(user_data, cgraph, candidate);
    if (result.supported &&
        (candidate.required_output_node_indices.size() != 1 ||
         candidate.required_output_node_indices[0] != candidate.node_indices.back())) {
        result = {};
    }
    return result;
}

struct interface_state {
    flagos_pattern_id supported = flagos_pattern_id::none;
    int execute_count = 0;
    const ggml_tensor * resolved_tensor = nullptr;
    const void * resolved_data = nullptr;
};

static flagos_lowering_choice query_pattern_interface(
        void * user_data,
        const ggml_cgraph * cgraph,
        const flagos_pattern_candidate & candidate) {
    const auto * state = static_cast<const interface_state *>(user_data);
    GGML_UNUSED(cgraph);
    flagos_lowering_choice result;
    result.supported = candidate.id == state->supported;
    result.capture_safe = result.supported;
    result.implementation_id = result.supported ? 17 : 0;
    return result;
}

static bool execute_probe(void * user_data, ggml_cgraph *, const flagos_plan_step & step) {
    if (step.kind != flagos_execution_kind::pattern) {
        return false;
    }
    ++static_cast<interface_state *>(user_data)->execute_count;
    return true;
}

static const void * resolve_interface_binding(void * user_data, const ggml_tensor * tensor) {
    const auto * state = static_cast<const interface_state *>(user_data);
    return state->resolved_tensor == tensor ? state->resolved_data : tensor->data;
}

struct binding_resolver_state {
    const ggml_tensor * tensor = nullptr;
    const void * data = nullptr;
};

static const void * resolve_binding(void * user_data, const ggml_tensor * tensor) {
    const auto * state = static_cast<const binding_resolver_state *>(user_data);
    return tensor == state->tensor ? state->data : tensor->data;
}

int main() {
    static_assert(static_cast<uint32_t>(flagos_pattern_id::rms_norm_mul) == 1);
    static_assert(static_cast<uint32_t>(flagos_pattern_id::attention_output_gate) == 12);
    static_assert(static_cast<uint32_t>(flagos_pattern_id::rms_norm_mul_rope) == 13);
    static_assert(static_cast<uint32_t>(flagos_quantized_matmul_kind::q4_k) == 1);
    static_assert(static_cast<uint32_t>(flagos_quantized_matmul_kind::q6_k) == 2);
    static_assert(static_cast<uint32_t>(flagos_quantized_matmul_kind::q4_0) == 3);
    static_assert(static_cast<uint32_t>(flagos_quantized_matmul_kind::q5_k) == 4);
    static_assert(static_cast<uint32_t>(flagos_quantized_matmul_kind::q4_1) == 5);
    static_assert(static_cast<uint32_t>(flagos_quantized_matmul_kind::q8_0) == 6);
    ggml_tensor q40_weight {};
    q40_weight.type = GGML_TYPE_Q4_0;
    q40_weight.ne[0] = 64;
    q40_weight.ne[1] = 96;
    q40_weight.ne[2] = 1;
    q40_weight.ne[3] = 1;
    q40_weight.nb[0] = ggml_type_size(GGML_TYPE_Q4_0);
    q40_weight.nb[1] = q40_weight.nb[0] * q40_weight.ne[0] / ggml_blck_size(GGML_TYPE_Q4_0);
    q40_weight.nb[2] = q40_weight.nb[1] * q40_weight.ne[1];
    q40_weight.nb[3] = q40_weight.nb[2];
    ggml_tensor q40_activation {};
    init_tensor(q40_activation, GGML_OP_NONE, 64, 3);
    ggml_tensor q40_output {};
    init_tensor(q40_output, GGML_OP_MUL_MAT, 96, 3);
    q40_output.src[0] = &q40_weight;
    q40_output.src[1] = &q40_activation;
    flagos_quantized_matmul_signature q40_signature;
    CHECK(flagos_describe_quantized_matmul(&q40_output, &q40_signature));
    CHECK(q40_signature.weight_kind == flagos_quantized_matmul_kind::q4_0);
    CHECK(q40_signature.k == 64);
    CHECK(q40_signature.rows == 96);
    CHECK(q40_signature.columns == 3);

    ggml_tensor q41_weight = q40_weight;
    q41_weight.type = GGML_TYPE_Q4_1;
    q41_weight.nb[0] = ggml_type_size(GGML_TYPE_Q4_1);
    q41_weight.nb[1] = q41_weight.nb[0] * q41_weight.ne[0] / ggml_blck_size(GGML_TYPE_Q4_1);
    ggml_tensor q41_output = q40_output;
    q41_output.src[0] = &q41_weight;
    flagos_quantized_matmul_signature q41_signature;
    CHECK(flagos_describe_quantized_matmul(&q41_output, &q41_signature));
    CHECK(q41_signature.weight_kind == flagos_quantized_matmul_kind::q4_1);

    ggml_tensor q80_weight = q40_weight;
    q80_weight.type = GGML_TYPE_Q8_0;
    q80_weight.nb[0] = ggml_type_size(GGML_TYPE_Q8_0);
    q80_weight.nb[1] = q80_weight.nb[0] * q80_weight.ne[0] / ggml_blck_size(GGML_TYPE_Q8_0);
    ggml_tensor q80_output = q40_output;
    q80_output.src[0] = &q80_weight;
    flagos_quantized_matmul_signature q80_signature;
    CHECK(flagos_describe_quantized_matmul(&q80_output, &q80_signature));
    CHECK(q80_signature.weight_kind == flagos_quantized_matmul_kind::q8_0);

    ggml_tensor q5_weight {};
    q5_weight.type = GGML_TYPE_Q5_K;
    q5_weight.ne[0] = 256;
    q5_weight.ne[1] = 96;
    q5_weight.ne[2] = 1;
    q5_weight.ne[3] = 1;
    q5_weight.nb[0] = ggml_type_size(GGML_TYPE_Q5_K);
    q5_weight.nb[1] = q5_weight.nb[0] * q5_weight.ne[0] / ggml_blck_size(GGML_TYPE_Q5_K);
    q5_weight.nb[2] = q5_weight.nb[1] * q5_weight.ne[1];
    q5_weight.nb[3] = q5_weight.nb[2];
    ggml_tensor q5_activation {};
    init_tensor(q5_activation, GGML_OP_NONE, 256, 3);
    ggml_tensor q5_output {};
    init_tensor(q5_output, GGML_OP_MUL_MAT, 96, 3);
    q5_output.src[0] = &q5_weight;
    q5_output.src[1] = &q5_activation;
    flagos_quantized_matmul_signature q5_signature;
    CHECK(flagos_describe_quantized_matmul(&q5_output, &q5_signature));
    CHECK(q5_signature.weight_kind == flagos_quantized_matmul_kind::q5_k);
    CHECK(q5_signature.k == 256);
    CHECK(q5_signature.rows == 96);
    CHECK(q5_signature.columns == 3);

    q40_weight.ne[0] = 48;
    q40_activation.ne[0] = 48;
    CHECK(!flagos_describe_quantized_matmul(&q40_output, &q40_signature));

    test_graph graph;
    make_norm_mul_graph(graph, false);
    auto supported = flagos_pattern_id::rms_norm_mul;
    auto plan = flagos_build_graph_plan(&graph.graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].kind == flagos_execution_kind::pattern);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::rms_norm_mul);
    CHECK(plan->steps[0].candidate.scope == flagos_fusion_scope::graph);
    CHECK(plan->steps[0].candidate.required_output_node_indices.size() == 1);
    CHECK(plan->steps[0].candidate.required_output_node_indices[0] == 1);
    CHECK(plan->steps[0].candidate.eliminated_write_bytes == 0);
    CHECK(plan->steps[0].candidate.eliminated_read_bytes == 1536 * sizeof(float));
    CHECK(plan->capture_safe());

    uint8_t norm_data = 0;
    uint8_t mul_data = 0;
    uint8_t input_data = 0;
    uint8_t weight_data = 0;
    graph.norm.data = &norm_data;
    graph.mul.data = &mul_data;
    graph.input.data = &input_data;
    graph.weight.data = &weight_data;
    flagos_graph_binding_snapshot bindings;
    CHECK(flagos_graph_binding_snapshot_create(&graph.graph, *plan, &bindings));
    CHECK(bindings.pointers.size() == 4);
    uint64_t binding_fingerprint = 0;
    CHECK(flagos_graph_binding_fingerprint(&graph.graph, *plan, &binding_fingerprint));
    CHECK(binding_fingerprint == bindings.fingerprint);
    CHECK(flagos_graph_binding_snapshot_matches(&graph.graph, *plan, bindings));
    graph.weight.data = &input_data;
    CHECK(!flagos_graph_binding_snapshot_matches(&graph.graph, *plan, bindings));
    graph.weight.data = &weight_data;

    binding_resolver_state resolver { &graph.weight, &input_data };
    flagos_graph_binding_snapshot resolved_bindings;
    CHECK(flagos_graph_binding_snapshot_create(
        &graph.graph, *plan, &resolved_bindings, resolve_binding, &resolver));
    CHECK(flagos_graph_binding_snapshot_matches(
        &graph.graph, *plan, resolved_bindings, resolve_binding, &resolver));
    resolver.data = &weight_data;
    CHECK(!flagos_graph_binding_snapshot_matches(
        &graph.graph, *plan, resolved_bindings, resolve_binding, &resolver));

    graph.weight.data = nullptr;
    flagos_graph_binding_snapshot invalid_bindings;
    CHECK(!flagos_graph_binding_snapshot_create(&graph.graph, *plan, &invalid_bindings));
    CHECK(invalid_bindings.pointers.empty());
    binding_fingerprint = 1;
    CHECK(!flagos_graph_binding_fingerprint(&graph.graph, *plan, &binding_fingerprint));
    CHECK(binding_fingerprint == 0);
    CHECK(!flagos_graph_binding_snapshot_matches(&graph.graph, *plan, bindings));
    graph.weight.data = &weight_data;

    ggml_cgraph invalid_graph = graph.graph;
    invalid_graph.nodes = nullptr;
    CHECK(!flagos_graph_binding_snapshot_create(&invalid_graph, *plan, &invalid_bindings));
    CHECK(!flagos_graph_binding_snapshot_create(nullptr, *plan, &invalid_bindings));

    ggml_tensor view_backing {};
    init_tensor(view_backing, GGML_OP_NONE, 1536);
    uint8_t view_backing_data = 0;
    uint8_t other_view_backing_data = 0;
    view_backing.data = &view_backing_data;
    graph.input.view_src = &view_backing;
    auto view_plan = flagos_build_graph_plan(&graph.graph, query_pattern, &supported);
    flagos_graph_binding_snapshot view_bindings;
    CHECK(flagos_graph_binding_snapshot_create(&graph.graph, *view_plan, &view_bindings));
    CHECK(view_bindings.pointers.size() == 5);
    view_backing.data = &other_view_backing_data;
    CHECK(!flagos_graph_binding_snapshot_matches(&graph.graph, *view_plan, view_bindings));
    view_backing.data = nullptr;
    CHECK(!flagos_graph_binding_snapshot_create(&graph.graph, *view_plan, &invalid_bindings));
    graph.input.view_src = nullptr;
    view_backing.data = &view_backing_data;

    interface_state state { supported, 0 };
    flagos_fusion_interface interface {
        /* .query_lowering  = */ query_pattern_interface,
        /* .validate_lowering = */ query_pattern_interface,
        /* .execute_fusion  = */ execute_probe,
        /* .user_data       = */ &state,
        /* .configuration_id = */ 1,
        /* .resolve_tensor_data = */ resolve_interface_binding,
    };
    auto interface_plan = flagos_build_graph_plan(&graph.graph, interface);
    CHECK(interface_plan->steps.size() == 1);
    CHECK(flagos_execute_fusion_step(interface, &graph.graph, interface_plan->steps[0]));
    CHECK(state.execute_count == 1);

    plan = flagos_build_graph_plan(&graph.graph, query_pattern_not_capture_safe, &supported);
    CHECK(!plan->capture_safe());

    supported = flagos_pattern_id::none;
    plan = flagos_build_graph_plan(&graph.graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 2);
    CHECK(plan->steps[0].kind == flagos_execution_kind::direct);
    CHECK(plan->steps[1].kind == flagos_execution_kind::direct);

    test_graph three_node_graph;
    make_norm_mul_graph(three_node_graph, true);
    supported = flagos_pattern_id::add_rms_norm_mul;
    plan = flagos_build_graph_plan(&three_node_graph.graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::add_rms_norm_mul);
    CHECK(plan->steps[0].candidate.required_output_node_indices.size() == 1);
    CHECK(plan->steps[0].candidate.required_output_node_indices[0] == 2);

    ggml_tensor residual_next_input {};
    ggml_tensor residual_next_add {};
    init_tensor(residual_next_input, GGML_OP_NONE, 1536);
    init_tensor(residual_next_add, GGML_OP_ADD, 1536);
    residual_next_add.src[0] = &three_node_graph.add;
    residual_next_add.src[1] = &residual_next_input;
    ggml_tensor * residual_nodes[] = {
        &three_node_graph.add, &three_node_graph.norm, &three_node_graph.mul, &residual_next_add,
    };
    ggml_cgraph residual_graph {};
    residual_graph.n_nodes = 4;
    residual_graph.nodes = residual_nodes;
    plan = flagos_build_graph_plan(&residual_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 2);
    CHECK(plan->steps[0].kind == flagos_execution_kind::pattern);
    CHECK(plan->steps[0].candidate.required_output_node_indices.size() == 2);
    CHECK(plan->steps[0].candidate.required_output_node_indices[0] == 0);
    CHECK(plan->steps[0].candidate.required_output_node_indices[1] == 2);
    CHECK(plan->steps[1].kind == flagos_execution_kind::direct);
    CHECK(plan->steps[1].candidate.node_indices[0] == 3);

    plan = flagos_build_graph_plan(&residual_graph, query_pattern_terminal_only, &supported);
    CHECK(plan->steps.size() == 4);
    for (const auto & step : plan->steps) {
        CHECK(step.kind == flagos_execution_kind::direct);
    }

    supported = flagos_pattern_id::rms_norm_mul;
    plan = flagos_build_graph_plan(&three_node_graph.graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 2);
    CHECK(plan->steps[0].kind == flagos_execution_kind::direct);
    CHECK(plan->steps[1].candidate.id == flagos_pattern_id::rms_norm_mul);

    test_graph equivalent_graph;
    make_norm_mul_graph(equivalent_graph, false);
    supported = flagos_pattern_id::rms_norm_mul;
    plan = flagos_build_graph_plan(&graph.graph, query_pattern, &supported);
    CHECK(plan->matches(&equivalent_graph.graph));
    CHECK(plan->structural_fingerprint ==
        flagos_build_graph_plan(&equivalent_graph.graph, query_pattern, &supported)->structural_fingerprint);

    equivalent_graph.norm.ne[0] = 1024;
    CHECK(!plan->matches(&equivalent_graph.graph));

    test_graph aliased_external_graph;
    make_norm_mul_graph(aliased_external_graph, false);
    aliased_external_graph.mul.src[1] = &aliased_external_graph.input;
    CHECK(!plan->matches(&aliased_external_graph.graph));

    flagos_graph_plan_cache cache(2);
    bool created = false;
    cache.get_or_create(&graph.graph, query_pattern, &supported, &created);
    CHECK(created);
    test_graph same_structure;
    make_norm_mul_graph(same_structure, false);
    same_structure.input.data = graph.input.data;
    same_structure.weight.data = graph.weight.data;
    same_structure.norm.data = graph.norm.data;
    same_structure.mul.data = graph.mul.data;
    cache.get_or_create(&same_structure.graph, query_pattern, &supported, &created);
    CHECK(!created);
    CHECK(cache.hits() == 1);
    CHECK(cache.misses() == 1);

    cache.get_or_create(&aliased_external_graph.graph, query_pattern, &supported, &created);
    CHECK(created);
    CHECK(cache.misses() == 2);

    ggml_tensor direct_node {};
    init_tensor(direct_node, GGML_OP_ADD, 16);
    uint8_t direct_data_a = 0;
    uint8_t direct_data_b = 0;
    direct_node.data = &direct_data_a;
    ggml_tensor * direct_nodes[] = { &direct_node };
    ggml_cgraph direct_graph {};
    direct_graph.n_nodes = 1;
    direct_graph.nodes = direct_nodes;
    interface_state direct_state;
    flagos_fusion_interface direct_interface {
        query_pattern_interface,
        query_pattern_interface,
        nullptr,
        &direct_state,
        1,
        resolve_interface_binding,
    };
    flagos_graph_plan_cache direct_cache(2);
    direct_cache.get_or_create(&direct_graph, direct_interface, &created);
    CHECK(created);
    direct_node.data = &direct_data_b;
    direct_cache.get_or_create(&direct_graph, direct_interface, &created);
    CHECK(!created);
    CHECK(direct_cache.hits() == 1);
    CHECK(direct_cache.misses() == 1);

    flagos_graph_plan_cache policy_cache(4);
    const auto & capture_safe_plan = policy_cache.get_or_create(
        &graph.graph, query_pattern, &supported, &created);
    CHECK(created);
    CHECK(capture_safe_plan.capture_safe());
    const auto & capture_unsafe_plan = policy_cache.get_or_create(
        &graph.graph, query_pattern_not_capture_safe, &supported, &created);
    CHECK(created);
    CHECK(!capture_unsafe_plan.capture_safe());
    supported = flagos_pattern_id::none;
    const auto & changed_configuration_plan = policy_cache.get_or_create(
        &graph.graph, query_pattern, &supported, &created, 2);
    CHECK(created);
    CHECK(changed_configuration_plan.steps.size() == 2);
    supported = flagos_pattern_id::rms_norm_mul;

    flagos_graph_plan_cache binding_cache(4);
    state.supported = flagos_pattern_id::rms_norm_mul;
    const auto & original_binding_plan = binding_cache.get_or_create(
        &graph.graph, interface, &created);
    CHECK(created);
    CHECK(original_binding_plan.steps.size() == 1);
    uint8_t other_weight_data = 0;
    graph.weight.data = &other_weight_data;
    state.supported = flagos_pattern_id::none;
    const auto & changed_binding_plan = binding_cache.get_or_create(
        &graph.graph, interface, &created);
    CHECK(created);
    CHECK(changed_binding_plan.steps.size() == 2);
    graph.weight.data = &weight_data;
    state.supported = flagos_pattern_id::rms_norm_mul;
    const auto & restored_binding_plan = binding_cache.get_or_create(
        &graph.graph, interface, &created);
    CHECK(!created);
    CHECK(restored_binding_plan.steps.size() == 1);

    // A scheduler UID is not a structural identity: prompt chunks can reuse
    // it while changing tensor shapes/strides.  The cache must not return the
    // previous plan solely because the UID matches.
    graph.graph.uid = 17;
    same_structure.graph.uid = 17;
    same_structure.norm.ne[0] = 1024;
    cache.get_or_create(&same_structure.graph, query_pattern, &supported, &created);
    CHECK(created);
    CHECK(cache.misses() == 3);

    cache.get_or_create(&same_structure.graph, query_pattern, &supported, &created);
    CHECK(!created);
    CHECK(cache.hits() == 2);
    same_structure.norm.ne[0] = 512;
    cache.get_or_create(&same_structure.graph, query_pattern, &supported, &created);
    CHECK(created);
    CHECK(cache.misses() == 4);

    ggml_cgraph empty_graph {};
    cache.get_or_create(&empty_graph, query_pattern, &supported, &created);
    CHECK(created);
    CHECK(cache.get_or_create(&empty_graph, query_pattern, &supported, &created).steps.empty());
    CHECK(!created);

    // A structural cache hit must refresh LRU state. Otherwise an actively reused plan can be evicted before a colder entry when a third structure is inserted.
    flagos_graph_plan_cache lru_cache(2);
    test_graph lru_a;
    test_graph lru_b;
    test_graph lru_c;
    make_norm_mul_graph(lru_a, false);
    make_norm_mul_graph(lru_b, false);
    make_norm_mul_graph(lru_c, false);
    lru_a.graph.uid = 101;
    lru_b.graph.uid = 102;
    lru_c.graph.uid = 103;
    lru_b.norm.ne[0] = 1024;
    lru_c.norm.ne[0] = 2048;
    lru_cache.get_or_create(&lru_a.graph, query_pattern, &supported, &created);
    CHECK(created);
    lru_cache.get_or_create(&lru_b.graph, query_pattern, &supported, &created);
    CHECK(created);
    lru_cache.get_or_create(&lru_a.graph, query_pattern, &supported, &created);
    CHECK(!created);
    lru_cache.get_or_create(&lru_c.graph, query_pattern, &supported, &created);
    CHECK(created);
    CHECK(lru_cache.evictions() == 1);
    lru_cache.get_or_create(&lru_b.graph, query_pattern, &supported, &created);
    CHECK(created);
    CHECK(lru_cache.evictions() == 2);

    // Pattern matching is dependency-based rather than tied to adjacent
    // scheduler node indices.  An unrelated branch may be interleaved between
    // the producer and consumer in a real GGML graph.
    ggml_tensor inter_input {};
    ggml_tensor inter_weight {};
    ggml_tensor inter_norm {};
    ggml_tensor inter_mul {};
    ggml_tensor inter_unrelated {};
    init_tensor(inter_input, GGML_OP_NONE, 1536);
    init_tensor(inter_weight, GGML_OP_NONE, 1536);
    init_tensor(inter_norm, GGML_OP_RMS_NORM, 1536);
    init_tensor(inter_mul, GGML_OP_MUL, 1536);
    init_tensor(inter_unrelated, GGML_OP_ADD, 8);
    inter_norm.src[0] = &inter_input;
    inter_mul.src[0] = &inter_norm;
    inter_mul.src[1] = &inter_weight;
    ggml_tensor * inter_nodes[] = { &inter_norm, &inter_unrelated, &inter_mul };
    ggml_cgraph inter_graph {};
    inter_graph.n_nodes = 3;
    inter_graph.nodes = inter_nodes;
    supported = flagos_pattern_id::rms_norm_mul;
    plan = flagos_build_graph_plan(&inter_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 2);
    CHECK(plan->steps[0].kind == flagos_execution_kind::pattern);
    CHECK(plan->steps[0].candidate.node_indices.size() == 2);
    CHECK(plan->steps[0].candidate.node_indices[0] == 0);
    CHECK(plan->steps[0].candidate.node_indices[1] == 2);
    CHECK(plan->steps[1].kind == flagos_execution_kind::direct);
    CHECK(plan->steps[1].candidate.node_indices[0] == 1);

    for (const ggml_op barrier : {
            GGML_OP_ACC,
            GGML_OP_SET,
            GGML_OP_CPY,
            GGML_OP_SET_ROWS,
            GGML_OP_MAP_CUSTOM1,
            GGML_OP_MAP_CUSTOM2,
            GGML_OP_MAP_CUSTOM3,
            GGML_OP_CUSTOM,
            GGML_OP_OPT_STEP_ADAMW,
            GGML_OP_OPT_STEP_SGD }) {
        inter_unrelated.op = barrier;
        plan = flagos_build_graph_plan(&inter_graph, query_pattern, &supported);
        CHECK(plan->steps.size() == 3);
        for (const auto & step : plan->steps) {
            CHECK(step.kind == flagos_execution_kind::direct);
        }
    }
    inter_unrelated.op = GGML_OP_ADD;

    // A fused terminal is written at the pattern entry.  If GGML has reused
    // that future output range for an intervening temporary, the
    // non-contiguous fusion must be declined and revalidated when scheduler
    // bindings rotate.
    uint8_t inter_mul_storage[1536 * sizeof(float)] {};
    uint8_t inter_norm_storage[1536 * sizeof(float)] {};
    uint8_t inter_input_storage[1536 * sizeof(float)] {};
    uint8_t inter_unrelated_storage[8 * sizeof(float)] {};
    inter_input.data = inter_input_storage;
    inter_norm.data = inter_norm_storage;
    inter_mul.data = inter_mul_storage;
    inter_unrelated.data = inter_unrelated_storage;
    plan = flagos_build_graph_plan(&inter_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 2);
    CHECK(plan->steps[0].kind == flagos_execution_kind::pattern);

    flagos_graph_plan_cache early_write_cache(4);
    early_write_cache.get_or_create(
        &inter_graph, query_pattern_interface, &state, &created,
        3, query_pattern_interface);
    CHECK(created);
    inter_unrelated.data = inter_mul_storage;
    const auto & alias_unsafe_plan = early_write_cache.get_or_create(
        &inter_graph, query_pattern_interface, &state, &created,
        3, query_pattern_interface);
    CHECK(created);
    CHECK(alias_unsafe_plan.steps.size() == 3);
    for (const auto & step : alias_unsafe_plan.steps) {
        CHECK(step.kind == flagos_execution_kind::direct);
    }
    inter_unrelated.data = inter_unrelated_storage;

    // A later fused node also moves its reads to the pattern entry. Distinct
    // tensor identities may still alias an intervening output allocation, so
    // the common planner must reject this write-before-read dependency.
    uint8_t inter_weight_storage[1536 * sizeof(float)] {};
    inter_weight.data = inter_weight_storage;
    plan = flagos_build_graph_plan(&inter_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 2);
    inter_weight.data = inter_unrelated_storage;
    plan = flagos_build_graph_plan(&inter_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 3);
    for (const auto & step : plan->steps) {
        CHECK(step.kind == flagos_execution_kind::direct);
    }

    // Binding rotation must invalidate a previously safe cached plan too.
    inter_weight.data = inter_weight_storage;
    flagos_graph_plan_cache early_read_cache(4);
    early_read_cache.get_or_create(
        &inter_graph, query_pattern_interface, &state, &created,
        4, query_pattern_interface);
    CHECK(created);
    inter_weight.data = inter_unrelated_storage;
    const auto & read_unsafe_plan = early_read_cache.get_or_create(
        &inter_graph, query_pattern_interface, &state, &created,
        4, query_pattern_interface);
    CHECK(created);
    CHECK(read_unsafe_plan.steps.size() == 3);
    for (const auto & step : read_unsafe_plan.steps) {
        CHECK(step.kind == flagos_execution_kind::direct);
    }
    inter_weight.data = inter_weight_storage;

    // The query-only cache API has no provider validator, but a captured
    // binding change must still rebuild the plan so storage hazards are
    // checked against the new allocation layout.
    flagos_graph_plan_cache query_only_binding_cache(4);
    query_only_binding_cache.get_or_create(
        &inter_graph, query_pattern, &supported, &created);
    CHECK(created);
    inter_weight.data = inter_unrelated_storage;
    const auto & query_only_alias_plan = query_only_binding_cache.get_or_create(
        &inter_graph, query_pattern, &supported, &created);
    CHECK(created);
    CHECK(query_only_alias_plan.steps.size() == 3);
    for (const auto & step : query_only_alias_plan.steps) {
        CHECK(step.kind == flagos_execution_kind::direct);
    }
    inter_weight.data = inter_weight_storage;

    // Early planning may run before the scheduler assigns storage. Reuse the
    // structural plan while bindings remain unavailable, then rebuild once
    // concrete addresses appear so storage hazards are checked.
    test_graph unbound_graph;
    make_norm_mul_graph(unbound_graph, false);
    flagos_graph_plan_cache unbound_cache(4);
    unbound_cache.get_or_create(
        &unbound_graph.graph, query_pattern, &supported, &created);
    CHECK(created);
    unbound_cache.get_or_create(
        &unbound_graph.graph, query_pattern, &supported, &created);
    CHECK(!created);
    unbound_graph.input.data = &input_data;
    unbound_graph.weight.data = &weight_data;
    unbound_graph.norm.data = &norm_data;
    unbound_graph.mul.data = &mul_data;
    unbound_cache.get_or_create(
        &unbound_graph.graph, query_pattern, &supported, &created);
    CHECK(created);
    unbound_cache.get_or_create(
        &unbound_graph.graph, query_pattern, &supported, &created);
    CHECK(!created);

    // Provider-private aliases participate in the same hazard check.
    state.resolved_tensor = &inter_weight;
    state.resolved_data = inter_unrelated_storage;
    plan = flagos_build_graph_plan(&inter_graph, interface);
    CHECK(plan->steps.size() == 3);
    state.resolved_tensor = nullptr;
    state.resolved_data = nullptr;

    // A non-contiguous match cannot jump over a producer needed by its later
    // node: the fused step executes at node 0, before node 1 has produced the
    // dynamic scale tensor consumed by the MUL at node 2.
    ggml_tensor late_input {};
    ggml_tensor late_weight_a {};
    ggml_tensor late_weight_b {};
    ggml_tensor late_norm {};
    ggml_tensor late_weight {};
    ggml_tensor late_mul {};
    init_tensor(late_input, GGML_OP_NONE, 1536);
    init_tensor(late_weight_a, GGML_OP_NONE, 1536);
    init_tensor(late_weight_b, GGML_OP_NONE, 1536);
    init_tensor(late_norm, GGML_OP_RMS_NORM, 1536);
    init_tensor(late_weight, GGML_OP_ADD, 1536);
    init_tensor(late_mul, GGML_OP_MUL, 1536);
    late_norm.src[0] = &late_input;
    late_weight.src[0] = &late_weight_a;
    late_weight.src[1] = &late_weight_b;
    late_mul.src[0] = &late_norm;
    late_mul.src[1] = &late_weight;
    ggml_tensor * late_nodes[] = { &late_norm, &late_weight, &late_mul };
    ggml_cgraph late_graph {};
    late_graph.n_nodes = 3;
    late_graph.nodes = late_nodes;
    supported = flagos_pattern_id::rms_norm_mul;
    plan = flagos_build_graph_plan(&late_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 3);
    CHECK(plan->steps[0].kind == flagos_execution_kind::direct);
    CHECK(plan->steps[1].kind == flagos_execution_kind::direct);
    CHECK(plan->steps[2].kind == flagos_execution_kind::direct);

    // A provider that only writes the terminal output must decline fanout.
    ggml_tensor fanout_input {};
    ggml_tensor fanout_weight {};
    ggml_tensor fanout_other {};
    ggml_tensor fanout_norm {};
    ggml_tensor fanout_mul {};
    ggml_tensor fanout_branch {};
    init_tensor(fanout_input, GGML_OP_NONE, 1536);
    init_tensor(fanout_weight, GGML_OP_NONE, 1536);
    init_tensor(fanout_other, GGML_OP_NONE, 1536);
    init_tensor(fanout_norm, GGML_OP_RMS_NORM, 1536);
    init_tensor(fanout_mul, GGML_OP_MUL, 1536);
    init_tensor(fanout_branch, GGML_OP_ADD, 1536);
    fanout_norm.src[0] = &fanout_input;
    fanout_mul.src[0] = &fanout_norm;
    fanout_mul.src[1] = &fanout_weight;
    fanout_branch.src[0] = &fanout_norm;
    fanout_branch.src[1] = &fanout_other;
    ggml_tensor * fanout_nodes[] = { &fanout_norm, &fanout_mul, &fanout_branch };
    ggml_cgraph fanout_graph {};
    fanout_graph.n_nodes = 3;
    fanout_graph.nodes = fanout_nodes;
    supported = flagos_pattern_id::rms_norm_mul;
    plan = flagos_build_graph_plan(&fanout_graph, query_pattern_terminal_only, &supported);
    CHECK(plan->steps.size() == 3);
    CHECK(plan->steps[0].kind == flagos_execution_kind::direct);
    CHECK(plan->steps[1].kind == flagos_execution_kind::direct);
    CHECK(plan->steps[2].kind == flagos_execution_kind::direct);

    // A provider with a two-output kernel can materialize the branched norm.
    plan = flagos_build_graph_plan(&fanout_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 2);
    CHECK(plan->steps[0].kind == flagos_execution_kind::pattern);
    CHECK(plan->steps[0].candidate.required_output_node_indices.size() == 2);
    CHECK(plan->steps[0].candidate.required_output_node_indices[0] == 0);
    CHECK(plan->steps[0].candidate.required_output_node_indices[1] == 1);
    CHECK(plan->steps[1].kind == flagos_execution_kind::direct);

    ggml_tensor rope_input {};
    ggml_tensor rope_weight {};
    ggml_tensor rope_norm {};
    ggml_tensor rope_mul {};
    ggml_tensor rope {};
    init_tensor(rope_input, GGML_OP_NONE, 128, 16);
    init_tensor(rope_weight, GGML_OP_NONE, 128);
    init_tensor(rope_norm, GGML_OP_RMS_NORM, 128, 16);
    init_tensor(rope_mul, GGML_OP_MUL, 128, 16);
    init_tensor(rope, GGML_OP_ROPE, 128, 16);
    rope_norm.src[0] = &rope_input;
    rope_mul.src[0] = &rope_norm;
    rope_mul.src[1] = &rope_weight;
    rope.src[0] = &rope_mul;
    ggml_tensor * rope_nodes[] = { &rope_norm, &rope_mul, &rope };
    ggml_cgraph rope_graph {};
    rope_graph.n_nodes = 3;
    rope_graph.nodes = rope_nodes;
    supported = flagos_pattern_id::rms_norm_mul_rope;
    plan = flagos_build_graph_plan(&rope_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::rms_norm_mul_rope);
    CHECK(plan->steps[0].candidate.node_indices.size() == 3);
    CHECK(plan->steps[0].candidate.eliminated_launches == 2);

    supported = flagos_pattern_id::none;
    plan = flagos_build_graph_plan(&rope_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 3);
    CHECK(plan->steps[0].kind == flagos_execution_kind::direct);
    CHECK(plan->steps[1].kind == flagos_execution_kind::direct);
    CHECK(plan->steps[2].kind == flagos_execution_kind::direct);

    rope_norm.flags |= GGML_TENSOR_FLAG_OUTPUT;
    supported = flagos_pattern_id::rms_norm_mul_rope;
    plan = flagos_build_graph_plan(&rope_graph, query_pattern_terminal_only, &supported);
    CHECK(plan->steps.size() == 3);
    CHECK(plan->steps[0].kind == flagos_execution_kind::direct);
    plan = flagos_build_graph_plan(&rope_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].candidate.required_output_node_indices.size() == 2);
    CHECK(plan->steps[0].candidate.required_output_node_indices[0] == 0);
    CHECK(plan->steps[0].candidate.required_output_node_indices[1] == 2);
    rope_norm.flags &= ~GGML_TENSOR_FLAG_OUTPUT;

    ggml_tensor rope_view {};
    ggml_tensor rope_indices {};
    ggml_tensor rope_cache {};
    ggml_tensor rope_set_rows {};
    init_tensor(rope_view, GGML_OP_VIEW, 128 * 16, 1);
    init_tensor(rope_indices, GGML_OP_NONE, 1);
    init_tensor(rope_cache, GGML_OP_NONE, 128 * 16, 32);
    init_tensor(rope_set_rows, GGML_OP_SET_ROWS, 128 * 16, 32);
    rope_view.src[0] = &rope;
    rope_view.view_src = &rope;
    rope_set_rows.src[0] = &rope_view;
    rope_set_rows.src[1] = &rope_indices;
    rope_set_rows.src[2] = &rope_cache;
    ggml_tensor * rope_store_nodes[] = {
        &rope_norm, &rope_mul, &rope, &rope_view, &rope_set_rows,
    };
    ggml_cgraph rope_store_graph {};
    rope_store_graph.n_nodes = 5;
    rope_store_graph.nodes = rope_store_nodes;
    supported = flagos_pattern_id::rms_norm_mul_rope_kv_store;
    plan = flagos_build_graph_plan(&rope_store_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::rms_norm_mul_rope_kv_store);
    CHECK(plan->steps[0].candidate.node_indices.size() == 5);
    CHECK(plan->steps[0].candidate.eliminated_launches == 4);

    ggml_tensor rope_branch_input {};
    ggml_tensor rope_branch {};
    init_tensor(rope_branch_input, GGML_OP_NONE, 128, 16);
    init_tensor(rope_branch, GGML_OP_ADD, 128, 16);
    rope_branch.src[0] = &rope;
    rope_branch.src[1] = &rope_branch_input;
    ggml_tensor * rope_fanout_nodes[] = {
        &rope_norm, &rope_mul, &rope, &rope_view, &rope_set_rows, &rope_branch,
    };
    ggml_cgraph rope_fanout_graph {};
    rope_fanout_graph.n_nodes = 6;
    rope_fanout_graph.nodes = rope_fanout_nodes;
    plan = flagos_build_graph_plan(&rope_fanout_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 2);
    CHECK(plan->steps[0].kind == flagos_execution_kind::pattern);
    CHECK(plan->steps[0].candidate.required_output_node_indices.size() == 2);
    CHECK(plan->steps[0].candidate.required_output_node_indices[0] == 2);
    CHECK(plan->steps[0].candidate.required_output_node_indices[1] == 4);
    CHECK(plan->steps[1].kind == flagos_execution_kind::direct);

    plan = flagos_build_graph_plan(&rope_fanout_graph, query_pattern_terminal_only, &supported);
    CHECK(plan->steps.size() == 6);
    for (const auto & step : plan->steps) {
        CHECK(step.kind == flagos_execution_kind::direct);
    }

    ggml_tensor ssm_input {};
    ggml_tensor ssm_conv {};
    ggml_tensor silu {};
    init_tensor(ssm_input, GGML_OP_NONE, 256);
    init_tensor(ssm_conv, GGML_OP_SSM_CONV, 256);
    init_tensor(silu, GGML_OP_UNARY, 256);
    ssm_conv.src[0] = &ssm_input;
    silu.src[0] = &ssm_conv;
    const int32_t unary_op = GGML_UNARY_OP_SILU;
    std::memcpy(silu.op_params, &unary_op, sizeof(unary_op));
    ggml_tensor * ssm_nodes[] = { &ssm_conv, &silu };
    ggml_cgraph ssm_graph {};
    ssm_graph.n_nodes = 2;
    ssm_graph.nodes = ssm_nodes;
    supported = flagos_pattern_id::ssm_conv_silu;
    plan = flagos_build_graph_plan(&ssm_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::ssm_conv_silu);
    CHECK(plan->steps[0].candidate.required_output_node_indices.size() == 1);
    CHECK(plan->steps[0].candidate.required_output_node_indices[0] == 1);

    ggml_tensor ssm_observer {};
    init_tensor(ssm_observer, GGML_OP_MUL, 256);
    ssm_observer.src[0] = &ssm_conv;
    ssm_observer.src[1] = &ssm_input;
    ggml_tensor * ssm_fanout_nodes[] = { &ssm_conv, &silu, &ssm_observer };
    ggml_cgraph ssm_fanout_graph {};
    ssm_fanout_graph.n_nodes = 3;
    ssm_fanout_graph.nodes = ssm_fanout_nodes;
    plan = flagos_build_graph_plan(&ssm_fanout_graph, query_pattern_terminal_only, &supported);
    CHECK(plan->steps.size() == 3);
    for (const auto & step : plan->steps) {
        CHECK(step.kind == flagos_execution_kind::direct);
    }

    ggml_tensor gate_input {};
    ggml_tensor gated_value {};
    ggml_tensor gate_silu {};
    ggml_tensor gated_output {};
    init_tensor(gate_input, GGML_OP_NONE, 128, 32);
    init_tensor(gated_value, GGML_OP_NONE, 128, 32);
    init_tensor(gate_silu, GGML_OP_UNARY, 128, 32);
    init_tensor(gated_output, GGML_OP_MUL, 128, 32);
    gate_silu.src[0] = &gate_input;
    std::memcpy(gate_silu.op_params, &unary_op, sizeof(unary_op));
    gated_output.src[0] = &gated_value;
    gated_output.src[1] = &gate_silu;
    ggml_tensor * gated_nodes[] = { &gate_silu, &gated_output };
    ggml_cgraph gated_graph {};
    gated_graph.n_nodes = 2;
    gated_graph.nodes = gated_nodes;
    supported = flagos_pattern_id::attention_output_gate;
    plan = flagos_build_graph_plan(&gated_graph, query_pattern_terminal_only, &supported);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::attention_output_gate);
    CHECK(plan->steps[0].candidate.node_indices.size() == 2);
    CHECK(plan->steps[0].candidate.required_output_node_indices.size() == 1);
    CHECK(plan->steps[0].candidate.required_output_node_indices[0] == 1);

    for (const int32_t gated_unary_op : {
            static_cast<int32_t>(GGML_UNARY_OP_SIGMOID),
            static_cast<int32_t>(GGML_UNARY_OP_SOFTPLUS) }) {
        std::memcpy(gate_silu.op_params, &gated_unary_op, sizeof(gated_unary_op));
        plan = flagos_build_graph_plan(
            &gated_graph, query_pattern_terminal_only, &supported);
        CHECK(plan->steps.size() == 1);
        CHECK(plan->steps[0].candidate.id == flagos_pattern_id::attention_output_gate);
    }
    const int32_t unsupported_gated_unary_op = GGML_UNARY_OP_TANH;
    std::memcpy(gate_silu.op_params, &unsupported_gated_unary_op,
        sizeof(unsupported_gated_unary_op));
    plan = flagos_build_graph_plan(
        &gated_graph, query_pattern_terminal_only, &supported);
    CHECK(plan->steps.size() == 2);
    CHECK(plan->steps[0].kind == flagos_execution_kind::direct);
    CHECK(plan->steps[1].kind == flagos_execution_kind::direct);
    std::memcpy(gate_silu.op_params, &unary_op, sizeof(unary_op));

    ggml_tensor alpha_input {};
    ggml_tensor alpha_bias {};
    ggml_tensor alpha_scale {};
    ggml_tensor alpha_add {};
    ggml_tensor alpha_softplus {};
    ggml_tensor alpha_output {};
    init_tensor(alpha_input, GGML_OP_NONE, 32, 4);
    init_tensor(alpha_bias, GGML_OP_NONE, 32);
    init_tensor(alpha_scale, GGML_OP_NONE, 32);
    init_tensor(alpha_add, GGML_OP_ADD, 32, 4);
    init_tensor(alpha_softplus, GGML_OP_UNARY, 32, 4);
    init_tensor(alpha_output, GGML_OP_MUL, 32, 4);
    alpha_add.src[0] = &alpha_input;
    alpha_add.src[1] = &alpha_bias;
    alpha_softplus.src[0] = &alpha_add;
    const int32_t softplus_op = GGML_UNARY_OP_SOFTPLUS;
    std::memcpy(alpha_softplus.op_params, &softplus_op, sizeof(softplus_op));
    alpha_output.src[0] = &alpha_softplus;
    alpha_output.src[1] = &alpha_scale;
    ggml_tensor * alpha_nodes[] = { &alpha_add, &alpha_softplus, &alpha_output };
    ggml_cgraph alpha_graph {};
    alpha_graph.n_nodes = 3;
    alpha_graph.nodes = alpha_nodes;
    plan = flagos_build_graph_plan(
        &alpha_graph, query_pattern_terminal_only, &supported);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::attention_output_gate);
    CHECK(plan->steps[0].candidate.node_indices.size() == 3);
    CHECK(plan->steps[0].candidate.required_output_node_indices.size() == 1);
    CHECK(plan->steps[0].candidate.required_output_node_indices[0] == 2);

    ggml_tensor alpha_add_observer {};
    init_tensor(alpha_add_observer, GGML_OP_MUL, 32, 4);
    alpha_add_observer.src[0] = &alpha_add;
    alpha_add_observer.src[1] = &alpha_input;
    ggml_tensor * alpha_fanout_nodes[] = {
        &alpha_add, &alpha_softplus, &alpha_output, &alpha_add_observer,
    };
    ggml_cgraph alpha_fanout_graph {};
    alpha_fanout_graph.n_nodes = 4;
    alpha_fanout_graph.nodes = alpha_fanout_nodes;
    plan = flagos_build_graph_plan(
        &alpha_fanout_graph, query_pattern_terminal_only, &supported);
    CHECK(plan->steps.size() == 3);
    CHECK(plan->steps[0].kind == flagos_execution_kind::direct);
    CHECK(plan->steps[1].kind == flagos_execution_kind::pattern);
    CHECK(plan->steps[1].candidate.node_indices.size() == 2);
    CHECK(plan->steps[2].kind == flagos_execution_kind::direct);

    ggml_tensor gate_observer {};
    init_tensor(gate_observer, GGML_OP_ADD, 128, 32);
    gate_observer.src[0] = &gate_silu;
    gate_observer.src[1] = &gated_value;
    ggml_tensor * gated_fanout_nodes[] = { &gate_silu, &gated_output, &gate_observer };
    ggml_cgraph gated_fanout_graph {};
    gated_fanout_graph.n_nodes = 3;
    gated_fanout_graph.nodes = gated_fanout_nodes;
    plan = flagos_build_graph_plan(
        &gated_fanout_graph, query_pattern_terminal_only, &supported);
    CHECK(plan->steps.size() == 3);
    for (const auto & step : plan->steps) {
        CHECK(step.kind == flagos_execution_kind::direct);
    }

    // A later producer cannot be pulled before the SiLU entry.  This is the
    // dependency that prevents the current side-plan from claiming an entire
    // RMSNorm -> gate matmul -> SiLU -> Mul region as one fused candidate.
    ggml_tensor late_value {};
    init_tensor(late_value, GGML_OP_ADD, 128, 32);
    late_value.src[0] = &gated_value;
    late_value.src[1] = &gated_value;
    gated_output.src[0] = &late_value;
    ggml_tensor * gated_late_input_nodes[] = {
        &gate_silu, &late_value, &gated_output,
    };
    ggml_cgraph gated_late_input_graph {};
    gated_late_input_graph.n_nodes = 3;
    gated_late_input_graph.nodes = gated_late_input_nodes;
    plan = flagos_build_graph_plan(
        &gated_late_input_graph, query_pattern_terminal_only, &supported);
    CHECK(plan->steps.size() == 3);
    for (const auto & step : plan->steps) {
        CHECK(step.kind == flagos_execution_kind::direct);
    }

    ggml_tensor gdn_q {};
    ggml_tensor gdn_k {};
    ggml_tensor gdn_v {};
    ggml_tensor gdn_gate {};
    ggml_tensor gdn_beta {};
    ggml_tensor gdn_state {};
    ggml_tensor gdn {};
    ggml_tensor gdn_snapshot {};
    ggml_tensor gdn_cache {};
    ggml_tensor gdn_copy {};
    ggml_tensor gdn_attention {};
    ggml_tensor gdn_attention_consumer {};
    init_tensor(gdn_q, GGML_OP_NONE, 4, 2);
    init_tensor(gdn_k, GGML_OP_NONE, 4, 2);
    init_tensor(gdn_v, GGML_OP_NONE, 4, 2);
    init_tensor(gdn_gate, GGML_OP_NONE, 1, 2);
    init_tensor(gdn_beta, GGML_OP_NONE, 1, 2);
    init_tensor(gdn_state, GGML_OP_NONE, 4, 8);
    init_tensor(gdn, GGML_OP_GATED_DELTA_NET, 8, 5);
    init_tensor(gdn_snapshot, GGML_OP_VIEW, 32);
    init_tensor(gdn_cache, GGML_OP_VIEW, 32);
    init_tensor(gdn_copy, GGML_OP_CPY, 32);
    init_tensor(gdn_attention, GGML_OP_VIEW, 8);
    init_tensor(gdn_attention_consumer, GGML_OP_RMS_NORM, 8);
    gdn_snapshot.ne[0] = 4;
    gdn_snapshot.ne[1] = 4;
    gdn_snapshot.ne[2] = 2;
    gdn_snapshot.nb[1] = gdn_snapshot.nb[0] * gdn_snapshot.ne[0];
    gdn_snapshot.nb[2] = gdn_snapshot.nb[1] * gdn_snapshot.ne[1];
    gdn_snapshot.nb[3] = gdn_snapshot.nb[2] * gdn_snapshot.ne[2];
    gdn.src[0] = &gdn_q;
    gdn.src[1] = &gdn_k;
    gdn.src[2] = &gdn_v;
    gdn.src[3] = &gdn_gate;
    gdn.src[4] = &gdn_beta;
    gdn.src[5] = &gdn_state;
    const int32_t gdn_snapshot_count = 1;
    std::memcpy(gdn.op_params, &gdn_snapshot_count, sizeof(gdn_snapshot_count));
    gdn_snapshot.src[0] = &gdn;
    gdn_snapshot.view_src = &gdn;
    gdn_snapshot.view_offs = 8 * sizeof(float);
    gdn_copy.src[0] = &gdn_snapshot;
    gdn_copy.src[1] = &gdn_cache;
    gdn_attention.src[0] = &gdn;
    gdn_attention.view_src = &gdn;
    gdn_attention_consumer.src[0] = &gdn_attention;
    uint8_t gdn_storage[128 * sizeof(float)] {};
    uint8_t gdn_cache_storage[96 * sizeof(float)] {};
    gdn.data = gdn_storage;
    gdn_attention.data = gdn_storage;
    gdn_snapshot.data = gdn_storage + gdn_snapshot.view_offs;
    gdn_cache.data = gdn_cache_storage;
    gdn_copy.data = gdn_cache_storage;
    ggml_tensor * gdn_nodes[] = {
        &gdn, &gdn_attention, &gdn_snapshot, &gdn_copy, &gdn_attention_consumer,
    };
    ggml_cgraph gdn_graph {};
    gdn_graph.n_nodes = 5;
    gdn_graph.nodes = gdn_nodes;
    supported = flagos_pattern_id::gated_delta_net_decode;
    plan = flagos_build_graph_plan(&gdn_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 2);
    CHECK(plan->steps[0].kind == flagos_execution_kind::pattern);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::gated_delta_net_decode);
    CHECK(plan->steps[0].candidate.node_indices.size() == 4);
    CHECK(plan->steps[0].candidate.node_indices[0] == 0);
    CHECK(plan->steps[0].candidate.node_indices[1] == 1);
    CHECK(plan->steps[0].candidate.node_indices[2] == 2);
    CHECK(plan->steps[0].candidate.node_indices[3] == 3);
    CHECK(plan->steps[0].candidate.required_output_node_indices.size() == 2);
    CHECK(plan->steps[0].candidate.required_output_node_indices[0] == 1);
    CHECK(plan->steps[0].candidate.required_output_node_indices[1] == 3);
    CHECK(plan->steps[0].candidate.eliminated_launches == 1);
    CHECK(plan->steps[0].candidate.eliminated_read_bytes == 32 * sizeof(float));
    CHECK(plan->steps[1].candidate.node_indices[0] == 4);

    // llama.cpp may omit zero-work view tensors from cgraph->nodes while
    // retaining them as CPY and consumer sources.  The cache-write pattern
    // must follow those tensor edges instead of requiring an explicit VIEW
    // node in the schedule.
    ggml_tensor * gdn_implicit_view_nodes[] = {
        &gdn, &gdn_copy, &gdn_attention_consumer,
    };
    ggml_cgraph gdn_implicit_view_graph {};
    gdn_implicit_view_graph.n_nodes = 3;
    gdn_implicit_view_graph.nodes = gdn_implicit_view_nodes;
    plan = flagos_build_graph_plan(&gdn_implicit_view_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 2);
    CHECK(plan->steps[0].kind == flagos_execution_kind::pattern);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::gated_delta_net_decode);
    CHECK(plan->steps[0].candidate.node_indices.size() == 2);
    CHECK(plan->steps[0].candidate.node_indices[0] == 0);
    CHECK(plan->steps[0].candidate.node_indices[1] == 1);
    CHECK(plan->steps[1].candidate.node_indices[0] == 2);

    ggml_tensor * gdn_scheduled_destination_nodes[] = {
        &gdn, &gdn_cache, &gdn_copy, &gdn_attention_consumer,
    };
    ggml_cgraph gdn_scheduled_destination_graph {};
    gdn_scheduled_destination_graph.n_nodes = 4;
    gdn_scheduled_destination_graph.nodes = gdn_scheduled_destination_nodes;
    plan = flagos_build_graph_plan(
        &gdn_scheduled_destination_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 2);
    CHECK(plan->steps[0].kind == flagos_execution_kind::pattern);
    CHECK(plan->steps[0].candidate.node_indices.size() == 3);
    CHECK(plan->steps[0].candidate.node_indices[0] == 0);
    CHECK(plan->steps[0].candidate.node_indices[1] == 1);
    CHECK(plan->steps[0].candidate.node_indices[2] == 2);
    CHECK(plan->steps[1].candidate.node_indices[0] == 3);

    gdn.flags |= GGML_TENSOR_FLAG_OUTPUT;
    plan = flagos_build_graph_plan(&gdn_graph, query_pattern, &supported);
    CHECK(plan->steps[0].candidate.required_output_node_indices.size() == 3);
    CHECK(plan->steps[0].candidate.required_output_node_indices[0] == 0);
    CHECK(plan->steps[0].candidate.required_output_node_indices[1] == 1);
    CHECK(plan->steps[0].candidate.required_output_node_indices[2] == 3);
    gdn.flags &= ~GGML_TENSOR_FLAG_OUTPUT;

    gdn_snapshot.flags |= GGML_TENSOR_FLAG_OUTPUT;
    plan = flagos_build_graph_plan(&gdn_graph, query_pattern, &supported);
    CHECK(plan->steps[0].candidate.required_output_node_indices.size() == 3);
    CHECK(plan->steps[0].candidate.required_output_node_indices[0] == 1);
    CHECK(plan->steps[0].candidate.required_output_node_indices[1] == 2);
    CHECK(plan->steps[0].candidate.required_output_node_indices[2] == 3);
    gdn_snapshot.flags &= ~GGML_TENSOR_FLAG_OUTPUT;

    gdn_v.ne[2] = 4;
    gdn_v.nb[3] = gdn_v.nb[2] * gdn_v.ne[2];
    const int32_t gdn_prefill_snapshot_count = 3;
    std::memcpy(gdn.op_params, &gdn_prefill_snapshot_count, sizeof(gdn_prefill_snapshot_count));
    gdn.ne[1] = 16;
    gdn.nb[2] = gdn.nb[1] * gdn.ne[1];
    gdn.nb[3] = gdn.nb[2];
    gdn_snapshot.ne[0] = 32;
    gdn_snapshot.ne[1] = 1;
    gdn_snapshot.ne[2] = 3;
    gdn_snapshot.nb[1] = gdn_snapshot.nb[0] * gdn_snapshot.ne[0];
    gdn_snapshot.nb[2] = gdn_snapshot.nb[1];
    gdn_snapshot.nb[3] = gdn_snapshot.nb[2] * gdn_snapshot.ne[2];
    gdn_snapshot.view_offs = 32 * sizeof(float);
    gdn_snapshot.data = gdn_storage + gdn_snapshot.view_offs;
    gdn_cache.ne[1] = 1;
    gdn_cache.ne[2] = 3;
    gdn_cache.nb[1] = gdn_cache.nb[0] * gdn_cache.ne[0];
    gdn_cache.nb[2] = gdn_cache.nb[1];
    gdn_cache.nb[3] = gdn_cache.nb[2] * gdn_cache.ne[2];
    gdn_copy.ne[1] = 1;
    gdn_copy.ne[2] = 3;
    gdn_copy.nb[1] = gdn_copy.nb[0] * gdn_copy.ne[0];
    gdn_copy.nb[2] = gdn_copy.nb[1];
    gdn_copy.nb[3] = gdn_copy.nb[2] * gdn_copy.ne[2];
    gdn_attention.ne[0] = 32;
    gdn_attention.nb[1] = gdn_attention.nb[0] * gdn_attention.ne[0];
    gdn_attention.nb[2] = gdn_attention.nb[1];
    gdn_attention.nb[3] = gdn_attention.nb[2];
    gdn_attention_consumer.ne[0] = 32;
    gdn_attention_consumer.nb[1] =
        gdn_attention_consumer.nb[0] * gdn_attention_consumer.ne[0];
    gdn_attention_consumer.nb[2] = gdn_attention_consumer.nb[1];
    gdn_attention_consumer.nb[3] = gdn_attention_consumer.nb[2];
    supported = flagos_pattern_id::gated_delta_net_prefill;
    plan = flagos_build_graph_plan(&gdn_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 2);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::gated_delta_net_prefill);

    gdn_snapshot.view_offs += sizeof(float);
    plan = flagos_build_graph_plan(&gdn_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 5);
    gdn_snapshot.view_offs -= sizeof(float);
    gdn_cache.nb[1] += sizeof(float);
    plan = flagos_build_graph_plan(&gdn_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 5);
    gdn_cache.nb[1] -= sizeof(float);

    // Reject a structurally plausible graph before converting an enormous
    // attention extent to a byte offset.  The element products still fit in
    // int64_t here, but the F32 byte count exceeds size_t on a 64-bit host.
    const int64_t gdn_tokens = gdn_v.ne[2];
    gdn_v.ne[2] = std::numeric_limits<int64_t>::max() / 16 + 1;
    plan = flagos_build_graph_plan(&gdn_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 5);
    gdn_v.ne[2] = gdn_tokens;

    ggml_tensor gdn_intervening {};
    init_tensor(gdn_intervening, GGML_OP_ADD, 1);
    ggml_tensor * gdn_intervening_nodes[] = {
        &gdn, &gdn_attention, &gdn_snapshot, &gdn_intervening, &gdn_copy,
        &gdn_attention_consumer,
    };
    ggml_cgraph gdn_intervening_graph {};
    gdn_intervening_graph.n_nodes = 6;
    gdn_intervening_graph.nodes = gdn_intervening_nodes;
    plan = flagos_build_graph_plan(&gdn_intervening_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 6);

    ggml_tensor gdn_snapshot_tap {};
    init_tensor(gdn_snapshot_tap, GGML_OP_VIEW, 32);
    gdn_snapshot_tap.src[0] = &gdn_snapshot;
    gdn_snapshot_tap.view_src = &gdn_snapshot;
    ggml_tensor * gdn_fanout_nodes[] = {
        &gdn, &gdn_attention, &gdn_snapshot, &gdn_copy,
        &gdn_attention_consumer, &gdn_snapshot_tap,
    };
    ggml_cgraph gdn_fanout_graph {};
    gdn_fanout_graph.n_nodes = 6;
    gdn_fanout_graph.nodes = gdn_fanout_nodes;
    plan = flagos_build_graph_plan(&gdn_fanout_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 3);
    CHECK(plan->steps[0].candidate.required_output_node_indices.size() == 3);
    CHECK(plan->steps[0].candidate.required_output_node_indices[0] == 1);
    CHECK(plan->steps[0].candidate.required_output_node_indices[1] == 2);
    CHECK(plan->steps[0].candidate.required_output_node_indices[2] == 3);
    plan = flagos_build_graph_plan(&gdn_fanout_graph, query_pattern_terminal_only, &supported);
    CHECK(plan->steps.size() == 6);

    ggml_tensor ffn_x {};
    ggml_tensor ffn_gate {};
    ggml_tensor ffn_up {};
    ggml_tensor ffn_glu {};
    init_tensor(ffn_x, GGML_OP_NONE, 512, 2);
    init_tensor(ffn_gate, GGML_OP_MUL_MAT, 512, 2);
    init_tensor(ffn_up, GGML_OP_MUL_MAT, 512, 2);
    init_tensor(ffn_glu, GGML_OP_GLU, 512, 2);
    ffn_gate.src[0] = &ffn_x;
    ffn_up.src[0] = &ffn_x;
    ffn_glu.src[0] = &ffn_gate;
    ffn_glu.src[1] = &ffn_up;
    const int32_t glu_op = GGML_GLU_OP_SWIGLU;
    std::memcpy(ffn_glu.op_params, &glu_op, sizeof(glu_op));
    ggml_tensor * ffn_nodes[] = { &ffn_gate, &ffn_up, &ffn_glu };
    ggml_cgraph ffn_graph {};
    ffn_graph.n_nodes = 3;
    ffn_graph.nodes = ffn_nodes;
    supported = flagos_pattern_id::ffn_swiglu;
    plan = flagos_build_graph_plan(&ffn_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::ffn_swiglu);

    ggml_tensor ffn_down_weight {};
    ggml_tensor ffn_down {};
    init_tensor(ffn_down_weight, GGML_OP_NONE, 512, 512);
    init_tensor(ffn_down, GGML_OP_MUL_MAT, 512, 2);
    ffn_down.src[0] = &ffn_down_weight;
    ffn_down.src[1] = &ffn_glu;
    ggml_tensor * ffn_down_nodes[] = { &ffn_gate, &ffn_up, &ffn_glu, &ffn_down };
    ggml_cgraph ffn_down_graph {};
    ffn_down_graph.n_nodes = 4;
    ffn_down_graph.nodes = ffn_down_nodes;
    supported = flagos_pattern_id::ffn_swiglu_down;
    plan = flagos_build_graph_plan(&ffn_down_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::ffn_swiglu_down);
    CHECK(plan->steps[0].candidate.node_indices.size() == 4);
    CHECK(plan->steps[0].candidate.required_output_node_indices.size() == 1);
    CHECK(plan->steps[0].candidate.required_output_node_indices[0] == 3);

    // The provider-neutral full-FFN pattern promises that the GLU
    // intermediate is graph-private.  An output flag or a second consumer
    // therefore prevents enumeration rather than relying on a provider to
    // discover the violation after planning.
    ffn_glu.flags |= GGML_TENSOR_FLAG_OUTPUT;
    plan = flagos_build_graph_plan(&ffn_down_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 4);
    ffn_glu.flags &= ~GGML_TENSOR_FLAG_OUTPUT;

    ggml_tensor ffn_glu_tap {};
    init_tensor(ffn_glu_tap, GGML_OP_ADD, 512, 2);
    ffn_glu_tap.src[0] = &ffn_glu;
    ggml_tensor * ffn_fanout_nodes[] = {
        &ffn_gate, &ffn_up, &ffn_glu, &ffn_down, &ffn_glu_tap,
    };
    ggml_cgraph ffn_fanout_graph {};
    ffn_fanout_graph.n_nodes = 5;
    ffn_fanout_graph.nodes = ffn_fanout_nodes;
    plan = flagos_build_graph_plan(&ffn_fanout_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 5);

    ggml_tensor ffn_barrier {};
    init_tensor(ffn_barrier, GGML_OP_CPY, 1);
    ggml_tensor * ffn_barrier_nodes[] = {
        &ffn_gate, &ffn_up, &ffn_glu, &ffn_barrier, &ffn_down,
    };
    ggml_cgraph ffn_barrier_graph {};
    ffn_barrier_graph.n_nodes = 5;
    ffn_barrier_graph.nodes = ffn_barrier_nodes;
    plan = flagos_build_graph_plan(&ffn_barrier_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 5);

    // If both FFN patterns are accepted with their default score, the strict
    // four-node superset wins the common node-count tie-break.
    const auto query_all_ffn = [](void *, const ggml_cgraph *,
                                  const flagos_pattern_candidate & candidate) {
        flagos_lowering_choice result;
        result.supported = candidate.id == flagos_pattern_id::ffn_swiglu ||
            candidate.id == flagos_pattern_id::ffn_swiglu_down;
        result.capture_safe = true;
        return result;
    };
    plan = flagos_build_graph_plan(&ffn_down_graph, query_all_ffn, nullptr);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::ffn_swiglu_down);

    // Match the second projection through the GLU dependency rather than by
    // choosing the first later MUL_MAT. Independent projections can be
    // interleaved by the scheduler without disabling a valid FFN fusion.
    ggml_tensor ffn_unrelated_weight {};
    ggml_tensor ffn_unrelated_input {};
    ggml_tensor ffn_unrelated {};
    init_tensor(ffn_unrelated_weight, GGML_OP_NONE, 512, 2);
    init_tensor(ffn_unrelated_input, GGML_OP_NONE, 512, 2);
    init_tensor(ffn_unrelated, GGML_OP_MUL_MAT, 512, 2);
    ffn_unrelated.src[0] = &ffn_unrelated_weight;
    ffn_unrelated.src[1] = &ffn_unrelated_input;
    ggml_tensor * ffn_interleaved_nodes[] = {
        &ffn_gate, &ffn_unrelated, &ffn_up, &ffn_glu,
    };
    ggml_cgraph ffn_interleaved_graph {};
    ffn_interleaved_graph.n_nodes = 4;
    ffn_interleaved_graph.nodes = ffn_interleaved_nodes;
    supported = flagos_pattern_id::ffn_swiglu;
    plan = flagos_build_graph_plan(&ffn_interleaved_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 2);
    CHECK(plan->steps[0].kind == flagos_execution_kind::pattern);
    CHECK(plan->steps[0].candidate.node_indices.size() == 3);
    CHECK(plan->steps[0].candidate.node_indices[0] == 0);
    CHECK(plan->steps[0].candidate.node_indices[1] == 2);
    CHECK(plan->steps[0].candidate.node_indices[2] == 3);
    CHECK(plan->steps[1].kind == flagos_execution_kind::direct);
    CHECK(plan->steps[1].candidate.node_indices[0] == 1);

    // The GLU sources may be topologically ordered either way; the provider
    // must inspect the GLU input slots to preserve the gate/up roles.
    ffn_glu.src[0] = &ffn_up;
    ffn_glu.src[1] = &ffn_gate;
    supported = flagos_pattern_id::ffn_swiglu;
    plan = flagos_build_graph_plan(&ffn_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::ffn_swiglu);
    supported = flagos_pattern_id::ffn_swiglu_down;
    plan = flagos_build_graph_plan(&ffn_down_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::ffn_swiglu_down);

    ggml_tensor flash_attn {};
    init_tensor(flash_attn, GGML_OP_FLASH_ATTN_EXT, 128, 1);
    ggml_tensor * flash_nodes[] = { &flash_attn };
    ggml_cgraph flash_graph {};
    flash_graph.n_nodes = 1;
    flash_graph.nodes = flash_nodes;
    supported = flagos_pattern_id::flash_attn_decode;
    plan = flagos_build_graph_plan(&flash_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::flash_attn_decode);
    CHECK(plan->steps[0].candidate.scope == flagos_fusion_scope::single_operator);
    CHECK(plan->steps[0].candidate.required_output_node_indices.size() == 1);
    CHECK(plan->steps[0].candidate.required_output_node_indices[0] == 0);

    CHECK(flagos_pattern_scope(flagos_pattern_id::ffn_swiglu) == flagos_fusion_scope::graph);
    CHECK(flagos_pattern_scope(flagos_pattern_id::ffn_swiglu_down) == flagos_fusion_scope::graph);
    CHECK(flagos_pattern_scope(flagos_pattern_id::flash_attn_prefill) == flagos_fusion_scope::single_operator);

    std::printf("FlagOS graph plan checks passed\n");
    return 0;
}
