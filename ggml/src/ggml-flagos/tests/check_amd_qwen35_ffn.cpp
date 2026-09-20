#include "ggml-impl.h"

#include "check_amd_qwen35_ffn.h"

#include "../flagos-provider.h"
#include "../ggml-flagos.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(bool condition, const char * expression, int line) {
    if (!condition) {
        std::fprintf(stderr, "check failed at line %d: %s\n", line, expression);
        std::abort();
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

void flagos_check_amd_qwen35_ffn_case(
        ggml_backend_t backend, ggml_backend_dev_t device, ggml_backend_buffer_type_t buft,
        ggml_context * context, ggml_type quant_type) {
    constexpr int64_t K = 256;
    constexpr int64_t ROWS = 16;
    ggml_tensor * gate_weights = ggml_new_tensor_2d(context, quant_type, K, ROWS);
    ggml_tensor * up_weights = ggml_new_tensor_2d(context, quant_type, K, ROWS);
    ggml_tensor * activation = ggml_new_tensor_2d(context, GGML_TYPE_F32, K, 1);
    ggml_tensor * gate = ggml_mul_mat(context, gate_weights, activation);
    ggml_tensor * up = ggml_mul_mat(context, up_weights, activation);
    ggml_tensor * glu = ggml_swiglu_split(context, gate, up);
    CHECK(gate_weights != nullptr && up_weights != nullptr && activation != nullptr &&
        gate != nullptr && up != nullptr && glu != nullptr);
    // Tests observe the provider's externally configured policy; they do not
    // enable experimental fusions or alter process-wide environment state.
    if (!ggml_backend_dev_supports_op(device, glu)) {
        std::fprintf(stdout, "FlagOS AMD FFN check skipped: fusion is not enabled for %s\n",
            ggml_type_name(quant_type));
        return;
    }

    ggml_backend_buffer_t buffers[] = {
        ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(gate_weights)),
        ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(up_weights)),
        ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(activation)),
        ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(gate)),
        ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(up)),
        ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(glu)),
    };
    for (ggml_backend_buffer_t buffer : buffers) {
        CHECK(buffer != nullptr);
    }
    ggml_tensor * tensors[] = { gate_weights, up_weights, activation, gate, up, glu };
    for (size_t index = 0; index < sizeof(tensors) / sizeof(tensors[0]); ++index) {
        CHECK(ggml_backend_tensor_alloc(buffers[index], tensors[index],
            ggml_backend_buffer_get_base(buffers[index])) == GGML_STATUS_SUCCESS);
    }

    std::vector<float> gate_host(static_cast<size_t>(K * ROWS));
    std::vector<float> up_host(gate_host.size());
    std::vector<float> activation_host(static_cast<size_t>(K));
    for (size_t index = 0; index < gate_host.size(); ++index) {
        gate_host[index] = static_cast<float>((index * 17) % 101) * 0.011f - 0.52f;
        up_host[index] = static_cast<float>((index * 29) % 89) * 0.013f - 0.61f;
    }
    for (size_t index = 0; index < activation_host.size(); ++index) {
        activation_host[index] = static_cast<float>((index * 13) % 79) * 0.017f - 0.63f;
    }
    std::vector<uint8_t> gate_quantized(ggml_nbytes(gate_weights));
    std::vector<uint8_t> up_quantized(ggml_nbytes(up_weights));
    ggml_quantize_chunk(quant_type, gate_host.data(), gate_quantized.data(), 0, ROWS, K, nullptr);
    ggml_quantize_chunk(quant_type, up_host.data(), up_quantized.data(), 0, ROWS, K, nullptr);
    ggml_backend_tensor_set_async(backend, gate_weights, gate_quantized.data(), 0, gate_quantized.size());
    ggml_backend_tensor_set_async(backend, up_weights, up_quantized.data(), 0, up_quantized.size());
    ggml_backend_tensor_set_async(backend, activation, activation_host.data(), 0, ggml_nbytes(activation));
    std::vector<float> gate_sentinel(static_cast<size_t>(ROWS), 1234.5f);
    std::vector<float> up_sentinel(static_cast<size_t>(ROWS), -987.25f);
    ggml_backend_tensor_set_async(backend, gate, gate_sentinel.data(), 0, ggml_nbytes(gate));
    ggml_backend_tensor_set_async(backend, up, up_sentinel.data(), 0, ggml_nbytes(up));
    ggml_backend_synchronize(backend);

    ggml_tensor * nodes[] = { gate, up, glu };
    ggml_cgraph graph {};
    graph.n_nodes = 3;
    graph.nodes = nodes;
    CHECK(ggml_backend_graph_compute(backend, &graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(static_cast<size_t>(ROWS));
    std::vector<float> gate_after(static_cast<size_t>(ROWS));
    std::vector<float> up_after(static_cast<size_t>(ROWS));
    ggml_backend_tensor_get_async(backend, glu, actual.data(), 0, ggml_nbytes(glu));
    ggml_backend_tensor_get_async(backend, gate, gate_after.data(), 0, ggml_nbytes(gate));
    ggml_backend_tensor_get_async(backend, up, up_after.data(), 0, ggml_nbytes(up));
    ggml_backend_synchronize(backend);
    CHECK(std::memcmp(gate_after.data(), gate_sentinel.data(), ggml_nbytes(gate)) == 0);
    CHECK(std::memcmp(up_after.data(), up_sentinel.data(), ggml_nbytes(up)) == 0);
    const size_t row_bytes = ggml_row_size(quant_type, K);
    const auto * traits = ggml_get_type_traits(quant_type);
    std::vector<float> gate_dequant(static_cast<size_t>(K));
    std::vector<float> up_dequant(static_cast<size_t>(K));
    for (int64_t row = 0; row < ROWS; ++row) {
        traits->to_float(gate_quantized.data() + static_cast<size_t>(row) * row_bytes,
            gate_dequant.data(), K);
        traits->to_float(up_quantized.data() + static_cast<size_t>(row) * row_bytes,
            up_dequant.data(), K);
        float gate_value = 0.0f;
        float up_value = 0.0f;
        for (int64_t col = 0; col < K; ++col) {
            gate_value += gate_dequant[static_cast<size_t>(col)] * activation_host[static_cast<size_t>(col)];
            up_value += up_dequant[static_cast<size_t>(col)] * activation_host[static_cast<size_t>(col)];
        }
        const float expected = gate_value / (1.0f + std::exp(-gate_value)) * up_value;
        CHECK(std::isfinite(actual[static_cast<size_t>(row)]));
        CHECK(std::fabs(actual[static_cast<size_t>(row)] - expected) < 3e-2f);
    }
    for (ggml_backend_buffer_t buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

#if defined(FLAGOS_AMD_QWEN35_FFN_STANDALONE)
int main() {
    ggml_backend_reg_t reg = ggml_backend_flagos_reg();
    CHECK(reg != nullptr);
    if (ggml_backend_reg_dev_count(reg) == 0) {
        std::puts("FlagOS AMD Qwen3.5 FFN checks skipped: no HIP device visible");
        return 0;
    }
    ggml_backend_dev_t device = ggml_backend_reg_dev_get(reg, 0);
    CHECK(device != nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_flagos_buffer_type(0);
    CHECK(buft != nullptr);
    ggml_init_params params { 64u << 20, nullptr, true };
    ggml_context * context = ggml_init(params);
    CHECK(context != nullptr);
    ggml_backend_t backend = ggml_backend_flagos_init(0);
    CHECK(backend != nullptr);
    flagos_check_amd_qwen35_ffn_case(backend, device, buft, context, GGML_TYPE_Q4_K);
    flagos_check_amd_qwen35_ffn_case(backend, device, buft, context, GGML_TYPE_Q4_0);
    ggml_backend_free(backend);
    ggml_free(context);
    std::puts("FlagOS AMD Qwen3.5 Q4_K/Q4_0 FFN checks passed");
    return 0;
}
#endif
