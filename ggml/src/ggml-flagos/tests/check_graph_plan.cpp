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

int main() {
    test_graph graph;
    make_norm_mul_graph(graph, false);
    auto supported = flagos_pattern_id::rms_norm_mul;
    auto plan = flagos_build_graph_plan(&graph.graph, query_pattern, &supported);
    CHECK(plan->steps.size() == 1);
    CHECK(plan->steps[0].kind == flagos_execution_kind::pattern);
    CHECK(plan->steps[0].candidate.id == flagos_pattern_id::rms_norm_mul);
    CHECK(plan->steps[0].candidate.eliminated_write_bytes == 0);
    CHECK(plan->steps[0].candidate.eliminated_read_bytes == 1536 * sizeof(float));
    CHECK(plan->capture_safe());

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

    std::printf("FlagOS graph plan checks passed\n");
    return 0;
}
