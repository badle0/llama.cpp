#include "flagos-graph-plan.h"

#include "../../ggml-impl.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

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

struct interface_state {
    flagos_pattern_id supported = flagos_pattern_id::none;
    int execute_count = 0;
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

int main() {
    test_graph graph;
    make_norm_mul_graph(graph, false);
    auto supported = flagos_pattern_id::rms_norm_mul;
    auto plan = flagos_build_graph_plan(&graph.graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].kind == flagos_execution_kind::pattern);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::rms_norm_mul);
    CHECK(plan->steps[0].candidate.scope == flagos_fusion_scope::graph);
    CHECK(plan->steps[0].candidate.eliminated_write_bytes == 0);
    CHECK(plan->steps[0].candidate.eliminated_read_bytes == 1536 * sizeof(float));
    CHECK(plan->capture_safe());

    interface_state state { supported, 0 };
    flagos_fusion_interface interface {
        /* .query_lowering  = */ query_pattern_interface,
        /* .execute_fusion  = */ execute_probe,
        /* .user_data       = */ &state,
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

    flagos_graph_plan_cache cache(2);
    bool created = false;
    cache.get_or_create(&graph.graph, query_pattern, &supported, &created);
    CHECK(created);
    test_graph same_structure;
    make_norm_mul_graph(same_structure, false);
    cache.get_or_create(&same_structure.graph, query_pattern, &supported, &created);
    CHECK(!created);
    CHECK(cache.hits() == 1);
    CHECK(cache.misses() == 1);

    // A scheduler UID is not a structural identity: prompt chunks can reuse
    // it while changing tensor shapes/strides.  The cache must not return the
    // previous plan solely because the UID matches.
    graph.graph.uid = 17;
    same_structure.graph.uid = 17;
    same_structure.norm.ne[0] = 1024;
    cache.get_or_create(&same_structure.graph, query_pattern, &supported, &created);
    CHECK(created);
    CHECK(cache.misses() == 2);

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

    // Do not fuse an intermediate whose output fans out to an unrelated
    // consumer; the fused RMS+MUL kernel only materializes the terminal MUL
    // result and cannot satisfy the second branch.
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
    plan = flagos_build_graph_plan(&fanout_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 3);
    CHECK(plan->steps[0].kind == flagos_execution_kind::direct);
    CHECK(plan->steps[1].kind == flagos_execution_kind::direct);
    CHECK(plan->steps[2].kind == flagos_execution_kind::direct);

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

    // The GLU sources may be topologically ordered either way; the provider
    // must inspect the GLU input slots to preserve the gate/up roles.
    ffn_glu.src[0] = &ffn_up;
    ffn_glu.src[1] = &ffn_gate;
    plan = flagos_build_graph_plan(&ffn_graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::ffn_swiglu);

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

    CHECK(flagos_pattern_scope(flagos_pattern_id::ffn_swiglu) == flagos_fusion_scope::graph);
    CHECK(flagos_pattern_scope(flagos_pattern_id::flash_attn_prefill) == flagos_fusion_scope::single_operator);

    std::printf("FlagOS graph plan checks passed\n");
    return 0;
}
