#include "flagos-provider.h"
#include "flagos-graph-plan.h"
#include "ggml-flagos.h"
#include "providers/amd/flagos-amd-aot.h"
#include "providers/amd/flagos-amd-api.h"
#include "tests/check_amd_qwen35_ffn.h"

#include "../../ggml-backend-impl.h"
#include "../../ggml-impl.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

static void check(bool condition, const char * expression, int line) {
    if (!condition) {
        std::fprintf(stderr, "check failed at line %d: %s\n", line, expression);
        std::abort();
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

int main(int argc, char ** argv) {
    bool require_ssm_conv_silu = false;
    bool require_attention_output_gate = false;
    for (int argument = 1; argument < argc; ++argument) {
        if (std::strcmp(argv[argument], "--require-ssm-conv-silu") == 0) {
            require_ssm_conv_silu = true;
        } else if (std::strcmp(argv[argument], "--require-attention-output-gate") == 0) {
            require_attention_output_gate = true;
        } else {
            std::fprintf(stderr,
                "usage: %s [--require-ssm-conv-silu] [--require-attention-output-gate]\n",
                argv[0]);
            return 2;
        }
    }

    ggml_backend_reg_t reg = ggml_backend_flagos_reg();
    CHECK(reg != nullptr);

    const size_t n_devices = ggml_backend_reg_dev_count(reg);
    if (n_devices == 0) {
        std::puts("FlagOS AMD checks skipped: no HIP device visible");
        return 0;
    }

    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    CHECK(dev != nullptr);
    CHECK(ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
        ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU);
    CHECK(ggml_backend_dev_name(dev) != nullptr);
    CHECK(ggml_backend_dev_description(dev) != nullptr);

    ggml_backend_dev_props props {};
    ggml_backend_dev_get_props(dev, &props);
    CHECK(props.name != nullptr);
    CHECK(props.description != nullptr);
    CHECK(props.caps.async);
    CHECK(props.caps.events);
    CHECK(props.type == ggml_backend_dev_type(dev));

    flagos_device_identity identity {};
    const flagos_provider_v1 * provider = flagos_amd_provider();
    CHECK(provider != nullptr);
    CHECK(provider->probe(reg));
    CHECK(provider->device_count() >= 1);
    CHECK(provider->device_identity(0, &identity));
    CHECK(flagos_device_identity_is_valid(provider, 0, &identity));
    CHECK(ggml_backend_flagos_get_device() == 0);

    ggml_backend_buffer_type_t buft = ggml_backend_flagos_buffer_type(0);
    CHECK(buft != nullptr);
    CHECK(ggml_backend_dev_supports_buft(dev, buft));
    CHECK(ggml_backend_buft_get_alignment(buft) >= 1);

    ggml_init_params params {
        /* .mem_size   = */ 1u << 20,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    CHECK(tensor != nullptr);

    const size_t tensor_size = ggml_nbytes(tensor);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, tensor_size);
    CHECK(buffer != nullptr);
    CHECK(ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);

    ggml_backend_buffer_clear(buffer, 0);
    std::vector<float> input(16);
    std::vector<float> output(16, -1.0f);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) + 0.25f;
    }
    // The backend owns a stream; synchronize before checking asynchronous copies.
    ggml_backend_t backend = ggml_backend_flagos_init(0);
    CHECK(backend != nullptr);
    ggml_backend_tensor_set_async(backend, tensor, input.data(), 0, tensor_size);
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get_async(backend, tensor, output.data(), 0, tensor_size);
    ggml_backend_synchronize(backend);
    for (size_t i = 0; i < output.size(); ++i) {
        CHECK(std::fabs(output[i] - input[i]) < 1e-6f);
    }

    ggml_backend_event_t event = ggml_backend_event_new(dev);
    CHECK(event != nullptr);
    ggml_backend_event_record(event, backend);
    ggml_backend_event_synchronize(event);
    ggml_backend_event_wait(backend, event);
    ggml_backend_synchronize(backend);
    ggml_backend_event_free(event);

    ggml_tensor view = *tensor;
    view.op = GGML_OP_VIEW;
    ggml_tensor add = *tensor;
    add.op = GGML_OP_ADD;
    ggml_tensor standalone_mul = *tensor;
    standalone_mul.op = GGML_OP_MUL;
    // Check an operation for which this provider intentionally has no
    // lowering; package availability is handled by the compile-time fixture
    // below rather than a process environment variable.
    ggml_tensor unsupported_unary = *tensor;
    unsupported_unary.op = GGML_OP_UNARY;
    unsupported_unary.src[0] = tensor;
    ggml_set_op_params_i32(&unsupported_unary, 0, GGML_UNARY_OP_TANH);
    CHECK(ggml_backend_dev_supports_op(dev, &view));
    CHECK(!ggml_backend_dev_supports_op(dev, &add));
    CHECK(!ggml_backend_dev_supports_op(dev, &standalone_mul));

    CHECK(!ggml_backend_dev_supports_op(dev, &unsupported_unary));

    ggml_tensor * empty_src = ggml_view_1d(ctx, tensor, 0, 0);
    ggml_tensor * empty_dst = ggml_view_1d(ctx, tensor, 0, 0);
    ggml_tensor * empty_scale = ggml_scale_inplace(ctx, empty_src, 0.0f);
    ggml_tensor * empty_copy = ggml_cpy(ctx, empty_src, empty_dst);
    CHECK(empty_src != nullptr && empty_dst != nullptr &&
        empty_scale != nullptr && empty_copy != nullptr);
    CHECK(ggml_backend_dev_supports_op(dev, empty_scale));
    CHECK(ggml_backend_dev_supports_op(dev, empty_copy));
    ggml_tensor * empty_nodes[] = { empty_src, empty_scale, empty_dst, empty_copy };
    ggml_cgraph empty_graph {};
    empty_graph.n_nodes = 4;
    empty_graph.nodes = empty_nodes;
    CHECK(ggml_backend_graph_compute(backend, &empty_graph) == GGML_STATUS_SUCCESS);
#if defined(FLAGOS_AMD_CONFIGURED_KERNEL_DIR)
    {
        flagos_device_profile device_profile {};
        CHECK(flagos_provider_get_device_profile(provider, 0, &device_profile));
        CHECK(device_profile.target != nullptr && device_profile.target[0] != '\0');
        std::filesystem::path package_dir(FLAGOS_AMD_CONFIGURED_KERNEL_DIR);
        if (!std::filesystem::exists(package_dir / "manifest.json")) {
            package_dir /= device_profile.target;
        }
        flagos_amd::kernel_registry package_metadata;
        CHECK(package_metadata.initialize(package_dir, -1, device_profile.target, false));
        const bool q40_ffn_available =
            package_metadata.find("flagos_ffn_swiglu_q4_0_f32_decode") != nullptr ||
            package_metadata.find("flagos_ffn_swiglu_q4_0_f32_decode_staged") != nullptr;
        if (q40_ffn_available) {
            flagos_check_amd_qwen35_ffn_case(
                backend, dev, buft, ctx, GGML_TYPE_Q4_0);
        }

        ggml_tensor * lhs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        ggml_tensor * rhs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        ggml_tensor * sum = ggml_add(ctx, lhs, rhs);
        CHECK(lhs != nullptr && rhs != nullptr && sum != nullptr);
        ggml_backend_buffer_t lhs_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(lhs));
        ggml_backend_buffer_t rhs_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(rhs));
        ggml_backend_buffer_t sum_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(sum));
        CHECK(lhs_buffer != nullptr && rhs_buffer != nullptr && sum_buffer != nullptr);
        CHECK(ggml_backend_tensor_alloc(lhs_buffer, lhs, ggml_backend_buffer_get_base(lhs_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(rhs_buffer, rhs, ggml_backend_buffer_get_base(rhs_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(sum_buffer, sum, ggml_backend_buffer_get_base(sum_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_dev_supports_op(dev, sum));

        std::vector<float> lhs_host(256);
        std::vector<float> rhs_host(256);
        std::vector<float> sum_host(256, -1.0f);
        for (size_t i = 0; i < lhs_host.size(); ++i) {
            lhs_host[i] = static_cast<float>(i) * 0.5f;
            rhs_host[i] = static_cast<float>(i) * -0.25f + 3.0f;
        }
        ggml_backend_tensor_set_async(backend, lhs, lhs_host.data(), 0, ggml_nbytes(lhs));
        ggml_backend_tensor_set_async(backend, rhs, rhs_host.data(), 0, ggml_nbytes(rhs));
        ggml_backend_synchronize(backend);
        ggml_tensor * nodes[] = { sum };
        ggml_cgraph graph {};
        graph.n_nodes = 1;
        graph.nodes = nodes;
        CHECK(ggml_backend_graph_compute(backend, &graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(backend, sum, sum_host.data(), 0, ggml_nbytes(sum));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < sum_host.size(); ++i) {
            CHECK(std::fabs(sum_host[i] - (lhs_host[i] + rhs_host[i])) < 1e-5f);
        }
        ggml_backend_buffer_free(lhs_buffer);
        ggml_backend_buffer_free(rhs_buffer);
        ggml_backend_buffer_free(sum_buffer);

        if (package_metadata.find("flagos_scale_f32") != nullptr) {
            constexpr float scale_value = 0.75f;
            constexpr float bias_value = -0.25f;
            ggml_tensor * scale_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
            ggml_tensor * scale_output = ggml_scale_bias(
                ctx, scale_input, scale_value, bias_value);
            CHECK(scale_input != nullptr && scale_output != nullptr);
            ggml_backend_buffer_t scale_input_buffer =
                ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(scale_input));
            ggml_backend_buffer_t scale_output_buffer =
                ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(scale_output));
            CHECK(scale_input_buffer != nullptr && scale_output_buffer != nullptr);
            CHECK(ggml_backend_tensor_alloc(scale_input_buffer, scale_input,
                ggml_backend_buffer_get_base(scale_input_buffer)) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_tensor_alloc(scale_output_buffer, scale_output,
                ggml_backend_buffer_get_base(scale_output_buffer)) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_dev_supports_op(dev, scale_output));

            std::vector<float> scale_input_host(256);
            std::vector<float> scale_output_host(256, -1.0f);
            for (size_t i = 0; i < scale_input_host.size(); ++i) {
                scale_input_host[i] = static_cast<float>(i) * 0.125f - 5.0f;
            }
            ggml_backend_tensor_set_async(
                backend, scale_input, scale_input_host.data(), 0, ggml_nbytes(scale_input));
            ggml_backend_synchronize(backend);
            ggml_tensor * scale_nodes[] = { scale_output };
            ggml_cgraph scale_graph {};
            scale_graph.n_nodes = 1;
            scale_graph.nodes = scale_nodes;
            CHECK(ggml_backend_graph_compute(backend, &scale_graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get_async(
                backend, scale_output, scale_output_host.data(), 0, ggml_nbytes(scale_output));
            ggml_backend_synchronize(backend);
            for (size_t i = 0; i < scale_output_host.size(); ++i) {
                CHECK(std::fabs(scale_output_host[i] -
                    (scale_input_host[i] * scale_value + bias_value)) < 1e-6f);
            }
            ggml_backend_buffer_free(scale_input_buffer);
            ggml_backend_buffer_free(scale_output_buffer);
        }

        const int64_t rms_rows = 3;
        const int64_t rms_cols = 1536;
        const float rms_eps = 1e-5f;
        ggml_tensor * rms_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rms_cols, rms_rows);
        ggml_tensor * rms_output = ggml_rms_norm(ctx, rms_input, rms_eps);
        CHECK(rms_input != nullptr && rms_output != nullptr);
        ggml_backend_buffer_t rms_input_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(rms_input));
        ggml_backend_buffer_t rms_output_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(rms_output));
        CHECK(rms_input_buffer != nullptr && rms_output_buffer != nullptr);
        CHECK(ggml_backend_tensor_alloc(rms_input_buffer, rms_input,
            ggml_backend_buffer_get_base(rms_input_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(rms_output_buffer, rms_output,
            ggml_backend_buffer_get_base(rms_output_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_dev_supports_op(dev, rms_output));

        std::vector<float> rms_input_host(static_cast<size_t>(rms_rows * rms_cols));
        std::vector<float> rms_output_host(rms_input_host.size(), -1.0f);
        std::vector<float> rms_expected(rms_input_host.size(), 0.0f);
        for (int64_t row = 0; row < rms_rows; ++row) {
            double sum_sq = 0.0;
            for (int64_t col = 0; col < rms_cols; ++col) {
                const size_t index = static_cast<size_t>(row * rms_cols + col);
                rms_input_host[index] = static_cast<float>((row + 1) * 0.03125 + col * 0.001 - 0.75);
                sum_sq += static_cast<double>(rms_input_host[index]) * rms_input_host[index];
            }
            const float scale = 1.0f / std::sqrt(static_cast<float>(sum_sq / rms_cols) + rms_eps);
            for (int64_t col = 0; col < rms_cols; ++col) {
                const size_t index = static_cast<size_t>(row * rms_cols + col);
                rms_expected[index] = rms_input_host[index] * scale;
            }
        }
        ggml_backend_tensor_set_async(backend, rms_input, rms_input_host.data(), 0, ggml_nbytes(rms_input));
        ggml_backend_synchronize(backend);
        ggml_tensor * rms_nodes[] = { rms_output };
        ggml_cgraph rms_graph {};
        rms_graph.n_nodes = 1;
        rms_graph.nodes = rms_nodes;
        CHECK(ggml_backend_graph_compute(backend, &rms_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(backend, rms_output, rms_output_host.data(), 0, ggml_nbytes(rms_output));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < rms_output_host.size(); ++i) {
            CHECK(std::fabs(rms_output_host[i] - rms_expected[i]) < 3e-5f);
        }
        ggml_backend_buffer_free(rms_input_buffer);
        ggml_backend_buffer_free(rms_output_buffer);

        ggml_tensor * fused_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rms_cols, rms_rows);
        ggml_tensor * fused_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rms_cols);
        ggml_tensor * fused_norm = ggml_rms_norm(ctx, fused_input, rms_eps);
        ggml_tensor * fused_mul = ggml_mul(ctx, fused_norm, fused_weight);
        CHECK(fused_input != nullptr && fused_weight != nullptr && fused_norm != nullptr && fused_mul != nullptr);
        ggml_backend_buffer_t fused_input_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(fused_input));
        ggml_backend_buffer_t fused_weight_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(fused_weight));
        ggml_backend_buffer_t fused_norm_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(fused_norm));
        ggml_backend_buffer_t fused_mul_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(fused_mul));
        CHECK(fused_input_buffer != nullptr && fused_weight_buffer != nullptr &&
            fused_norm_buffer != nullptr && fused_mul_buffer != nullptr);
        CHECK(ggml_backend_tensor_alloc(fused_input_buffer, fused_input,
            ggml_backend_buffer_get_base(fused_input_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(fused_weight_buffer, fused_weight,
            ggml_backend_buffer_get_base(fused_weight_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(fused_norm_buffer, fused_norm,
            ggml_backend_buffer_get_base(fused_norm_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(fused_mul_buffer, fused_mul,
            ggml_backend_buffer_get_base(fused_mul_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_dev_supports_op(dev, fused_norm));
        CHECK(ggml_backend_dev_supports_op(dev, fused_mul));
        std::vector<float> fused_input_host(static_cast<size_t>(rms_rows * rms_cols));
        std::vector<float> fused_weight_host(static_cast<size_t>(rms_cols));
        std::vector<float> fused_output_host(fused_input_host.size(), -1.0f);
        std::vector<float> fused_expected(fused_input_host.size(), 0.0f);
        for (int64_t col = 0; col < rms_cols; ++col) {
            fused_weight_host[static_cast<size_t>(col)] = 0.5f + static_cast<float>(col % 17) * 0.01f;
        }
        for (int64_t row = 0; row < rms_rows; ++row) {
            double sum_sq = 0.0;
            for (int64_t col = 0; col < rms_cols; ++col) {
                const size_t index = static_cast<size_t>(row * rms_cols + col);
                fused_input_host[index] = static_cast<float>((row + 1) * 0.07 + col * 0.0007 - 0.5);
                sum_sq += static_cast<double>(fused_input_host[index]) * fused_input_host[index];
            }
            const float scale = 1.0f / std::sqrt(static_cast<float>(sum_sq / rms_cols) + rms_eps);
            for (int64_t col = 0; col < rms_cols; ++col) {
                const size_t index = static_cast<size_t>(row * rms_cols + col);
                fused_expected[index] = fused_input_host[index] * scale * fused_weight_host[static_cast<size_t>(col)];
            }
        }
        ggml_backend_tensor_set_async(backend, fused_input, fused_input_host.data(), 0, ggml_nbytes(fused_input));
        ggml_backend_tensor_set_async(backend, fused_weight, fused_weight_host.data(), 0, ggml_nbytes(fused_weight));
        ggml_backend_synchronize(backend);
        ggml_tensor * fused_nodes[] = { fused_norm, fused_mul };
        ggml_cgraph fused_graph {};
        fused_graph.n_nodes = 2;
        fused_graph.nodes = fused_nodes;
        CHECK(ggml_backend_graph_compute(backend, &fused_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(backend, fused_mul, fused_output_host.data(), 0, ggml_nbytes(fused_mul));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < fused_output_host.size(); ++i) {
            CHECK(std::fabs(fused_output_host[i] - fused_expected[i]) < 4e-5f);
        }
        ggml_backend_buffer_free(fused_input_buffer);
        ggml_backend_buffer_free(fused_norm_buffer);
        ggml_backend_buffer_free(fused_mul_buffer);

        // Match the scheduler allocation used by Qwen3.5 GDN blocks: the
        // dead RMSNorm output and terminal scale result share one buffer.
        // This exercises the one-output ABI selected by the narrow kernel.
        constexpr int64_t narrow_rms_cols = 128;
        constexpr int64_t narrow_rms_rows = 33;
        ggml_tensor * narrow_rms_input = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, narrow_rms_cols, narrow_rms_rows);
        ggml_tensor * narrow_rms_weight = ggml_new_tensor_1d(
            ctx, GGML_TYPE_F32, narrow_rms_cols);
        ggml_tensor * narrow_rms_norm = ggml_rms_norm(
            ctx, narrow_rms_input, rms_eps);
        ggml_tensor * narrow_rms_mul = ggml_mul(
            ctx, narrow_rms_norm, narrow_rms_weight);
        CHECK(narrow_rms_input && narrow_rms_weight && narrow_rms_norm && narrow_rms_mul);
        ggml_backend_buffer_t narrow_rms_input_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(narrow_rms_input));
        ggml_backend_buffer_t narrow_rms_weight_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(narrow_rms_weight));
        ggml_backend_buffer_t narrow_rms_shared_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(narrow_rms_mul));
        CHECK(narrow_rms_input_buffer && narrow_rms_weight_buffer && narrow_rms_shared_buffer);
        CHECK(ggml_backend_tensor_alloc(narrow_rms_input_buffer, narrow_rms_input,
            ggml_backend_buffer_get_base(narrow_rms_input_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(narrow_rms_weight_buffer, narrow_rms_weight,
            ggml_backend_buffer_get_base(narrow_rms_weight_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(narrow_rms_shared_buffer, narrow_rms_norm,
            ggml_backend_buffer_get_base(narrow_rms_shared_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(narrow_rms_shared_buffer, narrow_rms_mul,
            ggml_backend_buffer_get_base(narrow_rms_shared_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(narrow_rms_norm->data == narrow_rms_mul->data);
        std::vector<float> narrow_rms_input_host(
            static_cast<size_t>(narrow_rms_cols * narrow_rms_rows));
        std::vector<float> narrow_rms_weight_host(static_cast<size_t>(narrow_rms_cols));
        std::vector<float> narrow_rms_output_host(narrow_rms_input_host.size(), 0.0f);
        std::vector<float> narrow_rms_expected(narrow_rms_input_host.size(), 0.0f);
        for (int64_t col = 0; col < narrow_rms_cols; ++col) {
            narrow_rms_weight_host[static_cast<size_t>(col)] =
                0.75f + static_cast<float>(col % 13) * 0.015f;
        }
        for (int64_t row = 0; row < narrow_rms_rows; ++row) {
            double sum_sq = 0.0;
            for (int64_t col = 0; col < narrow_rms_cols; ++col) {
                const size_t index = static_cast<size_t>(row * narrow_rms_cols + col);
                narrow_rms_input_host[index] =
                    static_cast<float>(static_cast<int>(index * 29 % 113) - 56) * 0.009f;
                sum_sq += static_cast<double>(narrow_rms_input_host[index]) *
                    narrow_rms_input_host[index];
            }
            const float scale = 1.0f / std::sqrt(
                static_cast<float>(sum_sq / narrow_rms_cols) + rms_eps);
            for (int64_t col = 0; col < narrow_rms_cols; ++col) {
                const size_t index = static_cast<size_t>(row * narrow_rms_cols + col);
                narrow_rms_expected[index] = narrow_rms_input_host[index] * scale *
                    narrow_rms_weight_host[static_cast<size_t>(col)];
            }
        }
        ggml_backend_tensor_set_async(backend, narrow_rms_input,
            narrow_rms_input_host.data(), 0, ggml_nbytes(narrow_rms_input));
        ggml_backend_tensor_set_async(backend, narrow_rms_weight,
            narrow_rms_weight_host.data(), 0, ggml_nbytes(narrow_rms_weight));
        ggml_backend_synchronize(backend);
        ggml_tensor * narrow_rms_nodes[] = { narrow_rms_norm, narrow_rms_mul };
        ggml_cgraph narrow_rms_graph {};
        narrow_rms_graph.n_nodes = 2;
        narrow_rms_graph.nodes = narrow_rms_nodes;
        CHECK(ggml_backend_graph_compute(backend, &narrow_rms_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(backend, narrow_rms_mul,
            narrow_rms_output_host.data(), 0, ggml_nbytes(narrow_rms_mul));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < narrow_rms_output_host.size(); ++i) {
            CHECK(std::fabs(narrow_rms_output_host[i] - narrow_rms_expected[i]) < 4e-5f);
        }
        ggml_backend_buffer_free(narrow_rms_input_buffer);
        ggml_backend_buffer_free(narrow_rms_weight_buffer);
        ggml_backend_buffer_free(narrow_rms_shared_buffer);

        const int64_t add_rows = 1;
        ggml_tensor * add_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rms_cols, add_rows);
        ggml_tensor * add_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rms_cols, add_rows);
        ggml_tensor * add_node = ggml_add(ctx, add_input, add_bias);
        ggml_tensor * add_norm = ggml_rms_norm(ctx, add_node, rms_eps);
        ggml_tensor * add_mul = ggml_mul(ctx, add_norm, fused_weight);
        CHECK(add_input != nullptr && add_bias != nullptr && add_node != nullptr && add_norm != nullptr && add_mul != nullptr);
        ggml_backend_buffer_t add_input_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(add_input));
        ggml_backend_buffer_t add_bias_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(add_bias));
        ggml_backend_buffer_t add_node_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(add_node));
        ggml_backend_buffer_t add_norm_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(add_norm));
        ggml_backend_buffer_t add_mul_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(add_mul));
        CHECK(add_input_buffer && add_bias_buffer && add_node_buffer && add_norm_buffer && add_mul_buffer);
        CHECK(ggml_backend_tensor_alloc(add_input_buffer, add_input, ggml_backend_buffer_get_base(add_input_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(add_bias_buffer, add_bias, ggml_backend_buffer_get_base(add_bias_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(add_node_buffer, add_node, ggml_backend_buffer_get_base(add_node_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(add_norm_buffer, add_norm, ggml_backend_buffer_get_base(add_norm_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(add_mul_buffer, add_mul, ggml_backend_buffer_get_base(add_mul_buffer)) == GGML_STATUS_SUCCESS);
        std::vector<float> add_input_host(static_cast<size_t>(rms_cols * add_rows));
        std::vector<float> add_bias_host(add_input_host.size());
        std::vector<float> add_output_host(add_input_host.size(), -1.0f);
        std::vector<float> add_expected(add_input_host.size());
        for (size_t i = 0; i < add_input_host.size(); ++i) {
            add_input_host[i] = static_cast<float>(i % 101) * 0.003f - 0.2f;
            add_bias_host[i] = static_cast<float>(i % 37) * 0.002f - 0.05f;
        }
        for (int64_t row = 0; row < add_rows; ++row) {
            double sum_sq = 0.0;
            for (int64_t col = 0; col < rms_cols; ++col) {
                const size_t index = static_cast<size_t>(row * rms_cols + col);
                const float value = add_input_host[index] + add_bias_host[index];
                sum_sq += static_cast<double>(value) * value;
            }
            const float scale = 1.0f / std::sqrt(static_cast<float>(sum_sq / rms_cols) + rms_eps);
            for (int64_t col = 0; col < rms_cols; ++col) {
                const size_t index = static_cast<size_t>(row * rms_cols + col);
                add_expected[index] = (add_input_host[index] + add_bias_host[index]) * scale * fused_weight_host[static_cast<size_t>(col)];
            }
        }
        ggml_backend_tensor_set_async(backend, add_input, add_input_host.data(), 0, ggml_nbytes(add_input));
        ggml_backend_tensor_set_async(backend, add_bias, add_bias_host.data(), 0, ggml_nbytes(add_bias));
        ggml_backend_tensor_set_async(backend, fused_weight, fused_weight_host.data(), 0, ggml_nbytes(fused_weight));
        ggml_backend_synchronize(backend);
        ggml_tensor * add_nodes[] = { add_node, add_norm, add_mul };
        ggml_cgraph add_graph {};
        add_graph.n_nodes = 3;
        add_graph.nodes = add_nodes;
        CHECK(ggml_backend_graph_compute(backend, &add_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(backend, add_mul, add_output_host.data(), 0, ggml_nbytes(add_mul));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < add_output_host.size(); ++i) {
            const float diff = std::fabs(add_output_host[i] - add_expected[i]);
            CHECK(diff < 2e-4f);
        }
        ggml_backend_buffer_free(add_input_buffer);
        ggml_backend_buffer_free(add_bias_buffer);
        ggml_backend_buffer_free(add_node_buffer);
        ggml_backend_buffer_free(add_norm_buffer);
        ggml_backend_buffer_free(add_mul_buffer);
        ggml_backend_buffer_free(fused_weight_buffer);

        // SET_ROWS is the KV-cache write used by Qwen-style decode graphs.
        // The destination is a view of the cache tensor, so initialize the
        // view through the backend rather than allocating a second buffer.
        const int64_t set_rows_cols = 256;
        const int64_t set_rows_src_count = 3;
        const int64_t set_rows_dst_count = 32;
        ggml_tensor * set_rows_dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,
            set_rows_cols, set_rows_dst_count);
        ggml_tensor * set_rows_src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
            set_rows_cols, set_rows_src_count);
        ggml_tensor * set_rows_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64,
            set_rows_src_count);
        ggml_tensor * set_rows_out = ggml_set_rows(ctx, set_rows_dst, set_rows_src, set_rows_indices);
        CHECK(set_rows_dst && set_rows_src && set_rows_indices && set_rows_out);
        ggml_backend_buffer_t set_rows_dst_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(set_rows_dst));
        ggml_backend_buffer_t set_rows_src_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(set_rows_src));
        ggml_backend_buffer_t set_rows_indices_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(set_rows_indices));
        CHECK(set_rows_dst_buffer && set_rows_src_buffer && set_rows_indices_buffer);
        CHECK(ggml_backend_tensor_alloc(set_rows_dst_buffer, set_rows_dst,
            ggml_backend_buffer_get_base(set_rows_dst_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_view_init(set_rows_out) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(set_rows_src_buffer, set_rows_src,
            ggml_backend_buffer_get_base(set_rows_src_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(set_rows_indices_buffer, set_rows_indices,
            ggml_backend_buffer_get_base(set_rows_indices_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_dev_supports_op(dev, set_rows_out));
        std::vector<float> set_rows_src_host(static_cast<size_t>(set_rows_cols * set_rows_src_count));
        std::vector<int64_t> set_rows_indices_host { 19, 7, 23 };
        std::vector<ggml_fp16_t> set_rows_dst_host(static_cast<size_t>(set_rows_cols * set_rows_dst_count),
            ggml_fp32_to_fp16(0.0f));
        for (size_t i = 0; i < set_rows_src_host.size(); ++i) {
            set_rows_src_host[i] = static_cast<float>(i % 37) * 0.03125f - 0.4f;
        }
        ggml_backend_buffer_clear(set_rows_dst_buffer, 0);
        ggml_backend_tensor_set_async(backend, set_rows_src, set_rows_src_host.data(),
            0, ggml_nbytes(set_rows_src));
        ggml_backend_tensor_set_async(backend, set_rows_indices, set_rows_indices_host.data(),
            0, ggml_nbytes(set_rows_indices));
        ggml_backend_synchronize(backend);
        ggml_tensor * set_rows_nodes[] = { set_rows_out };
        ggml_cgraph set_rows_graph {};
        set_rows_graph.n_nodes = 1;
        set_rows_graph.nodes = set_rows_nodes;
        CHECK(ggml_backend_graph_compute(backend, &set_rows_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(backend, set_rows_dst, set_rows_dst_host.data(),
            0, ggml_nbytes(set_rows_dst));
        ggml_backend_synchronize(backend);
        for (int64_t row = 0; row < set_rows_src_count; ++row) {
            const int64_t destination = set_rows_indices_host[static_cast<size_t>(row)];
            for (int64_t col = 0; col < set_rows_cols; ++col) {
                const size_t src_index = static_cast<size_t>(row * set_rows_cols + col);
                const size_t dst_index = static_cast<size_t>(destination * set_rows_cols + col);
                CHECK(std::fabs(ggml_fp16_to_fp32(set_rows_dst_host[dst_index]) - set_rows_src_host[src_index]) < 5e-3f);
            }
        }
        ggml_backend_buffer_free(set_rows_dst_buffer);
        ggml_backend_buffer_free(set_rows_src_buffer);
        ggml_backend_buffer_free(set_rows_indices_buffer);

        auto check_quantized_matmul = [&](ggml_type quant_type, int64_t rows, int64_t columns,
                                          int64_t k = 512) {
            const auto optional_mixed_quant = [](ggml_type type) {
                return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
                    type == GGML_TYPE_Q8_0;
            };
            ggml_tensor * quant_weights = ggml_new_tensor_2d(ctx, quant_type, k, rows);
            ggml_tensor * quant_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, columns);
            ggml_tensor * quant_output = ggml_mul_mat(ctx, quant_weights, quant_input);
            CHECK(quant_weights && quant_input && quant_output);
            ggml_backend_buffer_t quant_weights_buffer = ggml_backend_buft_alloc_buffer(
                buft, ggml_nbytes(quant_weights));
            ggml_backend_buffer_t quant_input_buffer = ggml_backend_buft_alloc_buffer(
                buft, ggml_nbytes(quant_input));
            ggml_backend_buffer_t quant_output_buffer = ggml_backend_buft_alloc_buffer(
                buft, ggml_nbytes(quant_output));
            CHECK(quant_weights_buffer && quant_input_buffer && quant_output_buffer);
            CHECK(ggml_backend_tensor_alloc(quant_weights_buffer, quant_weights,
                ggml_backend_buffer_get_base(quant_weights_buffer)) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_tensor_alloc(quant_input_buffer, quant_input,
                ggml_backend_buffer_get_base(quant_input_buffer)) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_tensor_alloc(quant_output_buffer, quant_output,
                ggml_backend_buffer_get_base(quant_output_buffer)) == GGML_STATUS_SUCCESS);
            if (!ggml_backend_dev_supports_op(dev, quant_output)) {
                if (optional_mixed_quant(quant_type)) {
                    std::fprintf(stderr,
                        "FlagOS AMD checks skipped optional %s MUL_MAT: AOT symbol is absent\n",
                        ggml_type_name(quant_type));
                    ggml_backend_buffer_free(quant_weights_buffer);
                    ggml_backend_buffer_free(quant_input_buffer);
                    ggml_backend_buffer_free(quant_output_buffer);
                    return;
                }
                CHECK(false);
            }
            flagos_quantized_matmul_signature quant_signature;
            CHECK(flagos_describe_quantized_matmul(quant_output, &quant_signature));
            CHECK(quant_signature.k == k && quant_signature.rows == rows &&
                quant_signature.columns == columns);

            std::vector<float> weights_host(static_cast<size_t>(rows * k));
            std::vector<uint8_t> weights_quantized(ggml_nbytes(quant_weights));
            std::vector<float> input_host(static_cast<size_t>(columns * k));
            for (size_t i = 0; i < weights_host.size(); ++i) {
                weights_host[i] = static_cast<float>((i * 17) % 101) * 0.011f - 0.52f;
            }
            for (size_t i = 0; i < input_host.size(); ++i) {
                input_host[i] = static_cast<float>((i * 13) % 79) * 0.017f - 0.63f;
            }
            ggml_quantize_chunk(quant_type, weights_host.data(), weights_quantized.data(),
                0, rows, k, nullptr);
            ggml_backend_tensor_set_async(backend, quant_weights, weights_quantized.data(),
                0, weights_quantized.size());
            ggml_backend_tensor_set_async(backend, quant_input, input_host.data(),
                0, ggml_nbytes(quant_input));
            ggml_backend_synchronize(backend);
            ggml_tensor * quant_nodes[] = { quant_output };
            ggml_cgraph quant_graph {};
            quant_graph.n_nodes = 1;
            quant_graph.nodes = quant_nodes;
            CHECK(ggml_backend_graph_compute(backend, &quant_graph) == GGML_STATUS_SUCCESS);
            std::vector<float> output_host(static_cast<size_t>(rows * columns), -1.0f);
            ggml_backend_tensor_get_async(backend, quant_output, output_host.data(),
                0, ggml_nbytes(quant_output));
            ggml_backend_synchronize(backend);

            const size_t row_bytes = ggml_row_size(quant_type, k);
            std::vector<float> dequantized(static_cast<size_t>(k));
            const auto * traits = ggml_get_type_traits(quant_type);
            for (int64_t row = 0; row < rows; ++row) {
                traits->to_float(weights_quantized.data() + static_cast<size_t>(row) * row_bytes,
                    dequantized.data(), k);
                for (int64_t column = 0; column < columns; ++column) {
                    float expected = 0.0f;
                    for (int64_t col = 0; col < k; ++col) {
                        expected += dequantized[static_cast<size_t>(col)] *
                            input_host[static_cast<size_t>(column * k + col)];
                    }
                    const float actual = output_host[static_cast<size_t>(column * rows + row)];
                    CHECK(std::fabs(actual - expected) < 3e-2f);
                }
            }
            ggml_backend_buffer_free(quant_weights_buffer);
            ggml_backend_buffer_free(quant_input_buffer);
            ggml_backend_buffer_free(quant_output_buffer);

        };
        check_quantized_matmul(GGML_TYPE_Q4_0, 7, 1);
        check_quantized_matmul(GGML_TYPE_Q4_0, 7, 5);
        check_quantized_matmul(GGML_TYPE_Q4_1, 7, 1);
        check_quantized_matmul(GGML_TYPE_Q4_1, 7, 5);
        check_quantized_matmul(GGML_TYPE_Q8_0, 7, 1);
        check_quantized_matmul(GGML_TYPE_Q8_0, 7, 5);
        check_quantized_matmul(GGML_TYPE_Q4_K, 7, 1);
        check_quantized_matmul(GGML_TYPE_Q4_K, 7, 5);
        check_quantized_matmul(GGML_TYPE_Q5_K, 7, 1);
        check_quantized_matmul(GGML_TYPE_Q5_K, 7, 5);
        check_quantized_matmul(GGML_TYPE_Q5_K, 16, 1);
        check_quantized_matmul(GGML_TYPE_Q6_K, 7, 1);
        check_quantized_matmul(GGML_TYPE_Q6_K, 7, 5);
        // Exercise the opt-in dequant-cache + dense F16 prefill path for both
        // quantized formats.  The production Qwen projections are much wider
        // than this regression case, but keeping K=1024 and rows=64 catches
        // cache reuse, Q6 decode, and the tiled GEMM ABI without a large test
        // allocation.
        check_quantized_matmul(GGML_TYPE_Q4_K, 64, 5, 1024);
        check_quantized_matmul(GGML_TYPE_Q5_K, 64, 5, 1024);
        check_quantized_matmul(GGML_TYPE_Q6_K, 64, 5, 1024);
        // Exercise the Qwen output projection geometry.  Small row counts can
        // hide launch/grid or address arithmetic bugs that only appear when a
        // vocabulary-sized Q6_K matrix is dispatched.
        check_quantized_matmul(GGML_TYPE_Q6_K, 151936, 1);

        auto check_quantized_get_rows = [&](ggml_type quant_type) {
            const auto optional_mixed_quant = [](ggml_type type) {
                return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
                    type == GGML_TYPE_Q8_0;
            };
            const int64_t k = 512;
            const int64_t table_rows = 9;
            const int64_t tokens = 3;
            ggml_tensor * table = ggml_new_tensor_2d(ctx, quant_type, k, table_rows);
            ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
            ggml_tensor * gathered = ggml_get_rows(ctx, table, indices);
            CHECK(table && indices && gathered);
            ggml_backend_buffer_t table_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(table));
            ggml_backend_buffer_t indices_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(indices));
            ggml_backend_buffer_t gathered_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(gathered));
            CHECK(table_buffer && indices_buffer && gathered_buffer);
            CHECK(ggml_backend_tensor_alloc(table_buffer, table,
                ggml_backend_buffer_get_base(table_buffer)) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_tensor_alloc(indices_buffer, indices,
                ggml_backend_buffer_get_base(indices_buffer)) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_tensor_alloc(gathered_buffer, gathered,
                ggml_backend_buffer_get_base(gathered_buffer)) == GGML_STATUS_SUCCESS);
            if (!ggml_backend_dev_supports_op(dev, gathered)) {
                if (optional_mixed_quant(quant_type)) {
                    std::fprintf(stderr,
                        "FlagOS AMD checks skipped optional %s GET_ROWS: AOT symbol is absent\n",
                        ggml_type_name(quant_type));
                    ggml_backend_buffer_free(table_buffer);
                    ggml_backend_buffer_free(indices_buffer);
                    ggml_backend_buffer_free(gathered_buffer);
                    return;
                }
                CHECK(false);
            }
            std::vector<float> table_host(static_cast<size_t>(table_rows * k));
            std::vector<uint8_t> table_quantized(ggml_nbytes(table));
            std::vector<int32_t> index_host { 7, 2, 5 };
            for (size_t i = 0; i < table_host.size(); ++i) {
                table_host[i] = static_cast<float>((i * 19) % 113) * 0.009f - 0.48f;
            }
            ggml_quantize_chunk(quant_type, table_host.data(), table_quantized.data(),
                0, table_rows, k, nullptr);
            ggml_backend_tensor_set_async(backend, table, table_quantized.data(), 0, table_quantized.size());
            ggml_backend_tensor_set_async(backend, indices, index_host.data(), 0, ggml_nbytes(indices));
            ggml_backend_synchronize(backend);
            ggml_tensor * gather_nodes[] = { gathered };
            ggml_cgraph gather_graph {};
            gather_graph.n_nodes = 1;
            gather_graph.nodes = gather_nodes;
            CHECK(ggml_backend_graph_compute(backend, &gather_graph) == GGML_STATUS_SUCCESS);
            std::vector<float> gathered_host(static_cast<size_t>(tokens * k), -1.0f);
            ggml_backend_tensor_get_async(backend, gathered, gathered_host.data(), 0, ggml_nbytes(gathered));
            ggml_backend_synchronize(backend);
            const size_t row_bytes = ggml_row_size(quant_type, k);
            std::vector<float> dequantized(static_cast<size_t>(k));
            const auto * traits = ggml_get_type_traits(quant_type);
            for (int64_t token = 0; token < tokens; ++token) {
                const int32_t source_row = index_host[static_cast<size_t>(token)];
                traits->to_float(table_quantized.data() + static_cast<size_t>(source_row) * row_bytes,
                    dequantized.data(), k);
                for (int64_t col = 0; col < k; ++col) {
                    CHECK(std::fabs(gathered_host[static_cast<size_t>(token * k + col)] -
                        dequantized[static_cast<size_t>(col)]) < 3e-2f);
                }
            }
            ggml_backend_buffer_free(table_buffer);
            ggml_backend_buffer_free(indices_buffer);
            ggml_backend_buffer_free(gathered_buffer);
        };
        check_quantized_get_rows(GGML_TYPE_Q4_0);
        check_quantized_get_rows(GGML_TYPE_Q4_1);
        check_quantized_get_rows(GGML_TYPE_Q8_0);
        check_quantized_get_rows(GGML_TYPE_Q4_K);
        check_quantized_get_rows(GGML_TYPE_Q5_K);
        check_quantized_get_rows(GGML_TYPE_Q6_K);

        const int64_t rope_ne0 = 128;
        const int64_t rope_ne1 = 3;
        const int64_t rope_ne2 = 4;
        ggml_tensor * rope_input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rope_ne0, rope_ne1, rope_ne2);
        ggml_tensor * rope_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rope_ne2);
        ggml_tensor * rope_output = ggml_rope(ctx, rope_input, rope_positions,
            static_cast<int>(rope_ne0), GGML_ROPE_TYPE_NEOX);
        CHECK(rope_input != nullptr && rope_positions != nullptr && rope_output != nullptr);
        ggml_backend_buffer_t rope_input_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(rope_input));
        ggml_backend_buffer_t rope_positions_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(rope_positions));
        ggml_backend_buffer_t rope_output_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(rope_output));
        CHECK(rope_input_buffer != nullptr && rope_positions_buffer != nullptr && rope_output_buffer != nullptr);
        CHECK(ggml_backend_tensor_alloc(rope_input_buffer, rope_input,
            ggml_backend_buffer_get_base(rope_input_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(rope_positions_buffer, rope_positions,
            ggml_backend_buffer_get_base(rope_positions_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(rope_output_buffer, rope_output,
            ggml_backend_buffer_get_base(rope_output_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_dev_supports_op(dev, rope_output));

        const size_t rope_elements = static_cast<size_t>(rope_ne0 * rope_ne1 * rope_ne2);
        std::vector<float> rope_input_host(rope_elements);
        std::vector<float> rope_output_host(rope_elements, -1.0f);
        std::vector<float> rope_expected(rope_elements, 0.0f);
        std::vector<int32_t> rope_positions_host { 0, 1, 5, 17 };
        const float rope_base = 10000.0f;
        const int64_t rope_half = rope_ne0 / 2;
        for (int64_t token = 0; token < rope_ne2; ++token) {
            const float position = static_cast<float>(rope_positions_host[static_cast<size_t>(token)]);
            for (int64_t head = 0; head < rope_ne1; ++head) {
                const size_t row_start = static_cast<size_t>((token * rope_ne1 + head) * rope_ne0);
                for (int64_t dim = 0; dim < rope_half; ++dim) {
                    const size_t first = row_start + static_cast<size_t>(dim);
                    const size_t second = row_start + static_cast<size_t>(dim + rope_half);
                    rope_input_host[first] = static_cast<float>(0.1 + first * 0.002);
                    rope_input_host[second] = static_cast<float>(-0.2 + second * 0.001);
                    const float theta = position * std::pow(rope_base,
                        -2.0f * static_cast<float>(dim) / static_cast<float>(rope_ne0));
                    const float c = std::cos(theta);
                    const float s = std::sin(theta);
                    rope_expected[first] = rope_input_host[first] * c - rope_input_host[second] * s;
                    rope_expected[second] = rope_input_host[first] * s + rope_input_host[second] * c;
                }
            }
        }
        ggml_backend_tensor_set_async(backend, rope_input, rope_input_host.data(), 0, ggml_nbytes(rope_input));
        ggml_backend_tensor_set_async(backend, rope_positions, rope_positions_host.data(), 0, ggml_nbytes(rope_positions));
        ggml_backend_synchronize(backend);
        ggml_tensor * rope_nodes[] = { rope_output };
        ggml_cgraph rope_graph {};
        rope_graph.n_nodes = 1;
        rope_graph.nodes = rope_nodes;
        CHECK(ggml_backend_graph_compute(backend, &rope_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(backend, rope_output, rope_output_host.data(), 0, ggml_nbytes(rope_output));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < rope_output_host.size(); ++i) {
            CHECK(std::fabs(rope_output_host[i] - rope_expected[i]) < 5e-5f);
        }

        ggml_tensor * rope_store_view = ggml_view_2d(ctx, rope_output,
            rope_ne0 * rope_ne1, rope_ne2, rope_output->nb[2], 0);
        ggml_tensor * rope_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,
            rope_ne0 * rope_ne1, 32);
        ggml_tensor * rope_store_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, rope_ne2);
        ggml_tensor * rope_store = ggml_set_rows(ctx, rope_cache, rope_store_view, rope_store_indices);
        CHECK(rope_store_view && rope_cache && rope_store_indices && rope_store);
        ggml_backend_buffer_t rope_cache_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(rope_cache));
        ggml_backend_buffer_t rope_store_indices_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(rope_store_indices));
        CHECK(rope_cache_buffer && rope_store_indices_buffer);
        CHECK(ggml_backend_tensor_alloc(rope_cache_buffer, rope_cache,
            ggml_backend_buffer_get_base(rope_cache_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_view_init(rope_store_view) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_view_init(rope_store) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(rope_store_indices_buffer, rope_store_indices,
            ggml_backend_buffer_get_base(rope_store_indices_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_dev_supports_op(dev, rope_store));
        const std::vector<int64_t> rope_store_indices_host { 3, 7, 11, 19 };
        ggml_backend_buffer_clear(rope_cache_buffer, 0);
        ggml_backend_tensor_set_async(backend, rope_store_indices, rope_store_indices_host.data(),
            0, ggml_nbytes(rope_store_indices));
        ggml_backend_synchronize(backend);
        ggml_tensor * rope_store_nodes[] = { rope_output, rope_store_view, rope_store };
        ggml_cgraph rope_store_graph {};
        rope_store_graph.n_nodes = 3;
        rope_store_graph.nodes = rope_store_nodes;
        CHECK(ggml_backend_graph_compute(backend, &rope_store_graph) == GGML_STATUS_SUCCESS);
        std::vector<ggml_fp16_t> rope_cache_host(static_cast<size_t>(rope_ne0 * rope_ne1 * 32),
            ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_get_async(backend, rope_cache, rope_cache_host.data(), 0, ggml_nbytes(rope_cache));
        ggml_backend_synchronize(backend);
        for (int64_t token = 0; token < rope_ne2; ++token) {
            const int64_t destination = rope_store_indices_host[static_cast<size_t>(token)];
            for (int64_t col = 0; col < rope_ne0 * rope_ne1; ++col) {
                const size_t expected_index = static_cast<size_t>(token * rope_ne0 * rope_ne1 + col);
                const size_t cache_index = static_cast<size_t>(destination * rope_ne0 * rope_ne1 + col);
                CHECK(std::fabs(ggml_fp16_to_fp32(rope_cache_host[cache_index]) - rope_expected[expected_index]) < 5e-3f);
            }
        }
        ggml_backend_buffer_free(rope_cache_buffer);
        ggml_backend_buffer_free(rope_store_indices_buffer);
        ggml_backend_buffer_free(rope_input_buffer);
        ggml_backend_buffer_free(rope_positions_buffer);
        ggml_backend_buffer_free(rope_output_buffer);

        const int64_t gdn_cases[][3] = {
            { 1, 3, 128 }, { 1, 1, 128 }, { 4, 3, 128 },
            { 4, 1, 128 }, { 4, 1, 64 },
        };
        // The 64-wide case must use the generic full-output kernel when the
        // package describes an exact-128 cache-only implementation.
        for (const auto & gdn_case : gdn_cases) {
            const int64_t gdn_tokens = gdn_case[0];
            const int64_t gdn_state_size = gdn_case[2];
            const int64_t gdn_q_heads = 2;
            const int64_t gdn_heads = 4;
            const int64_t gdn_sequences = 1;
            // Keep the reference cache shape identical to the provider ABI;
            // snapshots may be larger than the current token count.
            const int64_t gdn_snapshots = gdn_case[1];
            const int64_t gdn_written = std::min(gdn_tokens, gdn_snapshots);
            const int64_t gdn_state_elements = gdn_state_size * gdn_state_size * gdn_heads;
            const int64_t gdn_attention_elements =
                gdn_state_size * gdn_heads * gdn_tokens * gdn_sequences;
            ggml_tensor * gdn_q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                gdn_state_size, gdn_q_heads, gdn_tokens, gdn_sequences);
            ggml_tensor * gdn_k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                gdn_state_size, gdn_q_heads, gdn_tokens, gdn_sequences);
            ggml_tensor * gdn_v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                gdn_state_size, gdn_heads, gdn_tokens, gdn_sequences);
            ggml_tensor * gdn_gate = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                1, gdn_heads, gdn_tokens, gdn_sequences);
            ggml_tensor * gdn_beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                1, gdn_heads, gdn_tokens, gdn_sequences);
            ggml_tensor * gdn_state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                gdn_state_size, gdn_state_size, gdn_heads, gdn_sequences);
            ggml_tensor * gdn_output = ggml_gated_delta_net(
                ctx, gdn_q, gdn_k, gdn_v, gdn_gate, gdn_beta, gdn_state, gdn_snapshots);
            ggml_tensor * gdn_snapshot_view = ggml_view_3d(
                ctx, gdn_output, gdn_state_elements, gdn_sequences, gdn_written,
                gdn_state_elements * sizeof(float),
                gdn_state_elements * gdn_sequences * sizeof(float),
                gdn_attention_elements * sizeof(float));
            ggml_tensor * gdn_cache = ggml_new_tensor_3d(
                ctx, GGML_TYPE_F32, gdn_state_elements, gdn_sequences, gdn_written);
            ggml_tensor * gdn_cache_view = ggml_view_3d(
                ctx, gdn_cache, gdn_state_elements, gdn_sequences, gdn_written,
                gdn_cache->nb[1], gdn_cache->nb[2], 0);
            ggml_tensor * gdn_copy = ggml_cpy(ctx, gdn_snapshot_view, gdn_cache_view);
            ggml_tensor * gdn_attention_view = ggml_view_1d(
                ctx, gdn_output, gdn_attention_elements, 0);
            ggml_tensor * gdn_snapshot_observer = ggml_view_1d(
                ctx, gdn_output, gdn_state_elements,
                gdn_attention_elements * sizeof(float));
            CHECK(gdn_q && gdn_k && gdn_v && gdn_gate && gdn_beta && gdn_state &&
                gdn_output && gdn_snapshot_view && gdn_cache && gdn_cache_view &&
                gdn_copy && gdn_attention_view && gdn_snapshot_observer);

            ggml_tensor * gdn_allocated[] = {
                gdn_q, gdn_k, gdn_v, gdn_gate, gdn_beta, gdn_state, gdn_output, gdn_cache,
            };
            std::vector<ggml_backend_buffer_t> gdn_buffers;
            for (ggml_tensor * gdn_tensor : gdn_allocated) {
                ggml_backend_buffer_t gdn_buffer = ggml_backend_buft_alloc_buffer(
                    buft, ggml_nbytes(gdn_tensor));
                CHECK(gdn_buffer != nullptr);
                CHECK(ggml_backend_tensor_alloc(gdn_buffer, gdn_tensor,
                    ggml_backend_buffer_get_base(gdn_buffer)) == GGML_STATUS_SUCCESS);
                gdn_buffers.push_back(gdn_buffer);
            }
            CHECK(ggml_backend_view_init(gdn_snapshot_view) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_view_init(gdn_cache_view) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_view_init(gdn_copy) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_view_init(gdn_attention_view) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_view_init(gdn_snapshot_observer) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_dev_supports_op(dev, gdn_output));
            CHECK(ggml_backend_dev_supports_op(dev, gdn_copy));

            // The offline Triton package promises 16-element Q/K/V strides.
            // Row contiguity alone does not imply that stronger contract, so
            // make sure the provider rejects a layout that would invalidate
            // the AOT specialization before any kernel can be launched.
            const size_t saved_q_nb1 = gdn_q->nb[1];
            const size_t saved_k_nb1 = gdn_k->nb[1];
            gdn_q->nb[1] += sizeof(float);
            gdn_k->nb[1] += sizeof(float);
            CHECK(!ggml_backend_dev_supports_op(dev, gdn_output));
            gdn_q->nb[1] = saved_q_nb1;
            gdn_k->nb[1] = saved_k_nb1;
            CHECK(ggml_backend_dev_supports_op(dev, gdn_output));

            void * saved_q_data = gdn_q->data;
            gdn_q->data = static_cast<char *>(gdn_q->data) + sizeof(float);
            CHECK(!ggml_backend_dev_supports_op(dev, gdn_output));
            gdn_q->data = saved_q_data;
            CHECK(ggml_backend_dev_supports_op(dev, gdn_output));

            std::vector<float> gdn_q_host(ggml_nelements(gdn_q));
            std::vector<float> gdn_k_host(ggml_nelements(gdn_k));
            std::vector<float> gdn_v_host(ggml_nelements(gdn_v));
            std::vector<float> gdn_gate_host(ggml_nelements(gdn_gate));
            std::vector<float> gdn_beta_host(ggml_nelements(gdn_beta));
            std::vector<float> gdn_state_host(ggml_nelements(gdn_state));
            for (size_t i = 0; i < gdn_q_host.size(); ++i) {
                gdn_q_host[i] = static_cast<float>(i % 37) * 0.002f - 0.03f;
                gdn_k_host[i] = static_cast<float>(i % 29) * 0.0015f - 0.02f;
            }
            for (size_t i = 0; i < gdn_v_host.size(); ++i) {
                gdn_v_host[i] = static_cast<float>(i % 41) * 0.001f - 0.02f;
            }
            for (size_t i = 0; i < gdn_gate_host.size(); ++i) {
                gdn_gate_host[i] = static_cast<float>(i % 7) * 0.002f - 0.012f;
                gdn_beta_host[i] = 0.4f + static_cast<float>(i % 5) * 0.03f;
            }
            for (size_t i = 0; i < gdn_state_host.size(); ++i) {
                gdn_state_host[i] = static_cast<float>(i % 31) * 0.0005f - 0.0075f;
            }
            ggml_backend_tensor_set_async(backend, gdn_q, gdn_q_host.data(), 0, ggml_nbytes(gdn_q));
            ggml_backend_tensor_set_async(backend, gdn_k, gdn_k_host.data(), 0, ggml_nbytes(gdn_k));
            ggml_backend_tensor_set_async(backend, gdn_v, gdn_v_host.data(), 0, ggml_nbytes(gdn_v));
            ggml_backend_tensor_set_async(backend, gdn_gate, gdn_gate_host.data(), 0, ggml_nbytes(gdn_gate));
            ggml_backend_tensor_set_async(backend, gdn_beta, gdn_beta_host.data(), 0, ggml_nbytes(gdn_beta));
            ggml_backend_tensor_set_async(backend, gdn_state, gdn_state_host.data(), 0, ggml_nbytes(gdn_state));
            ggml_backend_synchronize(backend);

            std::vector<float> gdn_direct_output(ggml_nelements(gdn_output), -1.0f);
            std::vector<float> gdn_direct_cache(ggml_nelements(gdn_cache), -1.0f);
            std::vector<float> gdn_fused_output(gdn_direct_output.size(), -1.0f);
            std::vector<float> gdn_fused_cache(gdn_direct_cache.size(), -1.0f);

            // Compute one element of the final recurrent state independently.
            // Repeating clear -> compute catches missing ordering between the
            // synchronous buffer clear callback and the backend's nonblocking
            // HIP stream.
            std::vector<float> gdn_expected_state_row(
                gdn_state_host.begin(), gdn_state_host.begin() + gdn_state_size);
            for (int64_t token = 0; token < gdn_tokens; ++token) {
                const size_t qk_base = static_cast<size_t>(
                    token * gdn_q_heads * gdn_state_size);
                const size_t v_base = static_cast<size_t>(
                    token * gdn_heads * gdn_state_size);
                const size_t scalar_index = static_cast<size_t>(token * gdn_heads);
                const float decay = std::exp(gdn_gate_host[scalar_index]);
                float projected_key = 0.0f;
                for (int64_t lane = 0; lane < gdn_state_size; ++lane) {
                    gdn_expected_state_row[static_cast<size_t>(lane)] *= decay;
                    projected_key += gdn_expected_state_row[static_cast<size_t>(lane)] *
                        gdn_k_host[qk_base + static_cast<size_t>(lane)];
                }
                const float delta =
                    (gdn_v_host[v_base] - projected_key) * gdn_beta_host[scalar_index];
                for (int64_t lane = 0; lane < gdn_state_size; ++lane) {
                    gdn_expected_state_row[static_cast<size_t>(lane)] +=
                        delta * gdn_k_host[qk_base + static_cast<size_t>(lane)];
                }
            }

            // Build the reference in two direct graphs.  Keeping the cache
            // copy in a separate graph prevents the provider-neutral
            // GDN+VIEW+CPY candidate from matching, without mutating process
            // environment or provider policy from inside the test.
            ggml_backend_buffer_clear(gdn_buffers[7], 0);
            ggml_tensor * gdn_direct_nodes[] = { gdn_output };
            ggml_cgraph gdn_direct_graph {};
            gdn_direct_graph.n_nodes = 1;
            gdn_direct_graph.nodes = gdn_direct_nodes;
            const int clear_order_trials =
                gdn_tokens == 4 && gdn_snapshots == 3 && gdn_state_size == 128 ? 16 : 1;
            for (int trial = 0; trial < clear_order_trials; ++trial) {
                ggml_backend_buffer_clear(gdn_buffers[6], 0);
                CHECK(ggml_backend_graph_compute(backend, &gdn_direct_graph) == GGML_STATUS_SUCCESS);
                float first_snapshot_value = 0.0f;
                ggml_backend_tensor_get(
                    gdn_output, &first_snapshot_value,
                    static_cast<size_t>(gdn_attention_elements) * sizeof(float), sizeof(float));
                CHECK(std::fabs(first_snapshot_value - gdn_expected_state_row[0]) < 5e-4f);
            }
            ggml_tensor * gdn_copy_nodes[] = { gdn_snapshot_view, gdn_copy };
            ggml_cgraph gdn_copy_graph {};
            gdn_copy_graph.n_nodes = 2;
            gdn_copy_graph.nodes = gdn_copy_nodes;
            CHECK(ggml_backend_graph_compute(backend, &gdn_copy_graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get_async(backend, gdn_output, gdn_direct_output.data(),
                0, ggml_nbytes(gdn_output));
            ggml_backend_tensor_get_async(backend, gdn_cache, gdn_direct_cache.data(),
                0, ggml_nbytes(gdn_cache));
            ggml_backend_synchronize(backend);

            // The full graph is executed under the caller's provider policy.
            // On a tuned package this may select the fused cache ABI; on an
            // untuned or explicitly restricted package it remains a direct
            // conformance check.  Either way, numerical behavior must match
            // the direct reference above.
            ggml_backend_buffer_clear(gdn_buffers[6], 0);
            ggml_backend_buffer_clear(gdn_buffers.back(), 0);
            ggml_tensor * gdn_nodes[] = {
                gdn_output, gdn_attention_view, gdn_snapshot_view, gdn_copy,
            };
            ggml_cgraph gdn_graph {};
            gdn_graph.n_nodes = 4;
            gdn_graph.nodes = gdn_nodes;
            CHECK(ggml_backend_graph_compute(backend, &gdn_graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get_async(backend, gdn_output, gdn_fused_output.data(),
                0, ggml_nbytes(gdn_output));
            ggml_backend_tensor_get_async(backend, gdn_cache, gdn_fused_cache.data(),
                0, ggml_nbytes(gdn_cache));
            ggml_backend_synchronize(backend);
            // Cache-only fusion is allowed to leave the temporary snapshot
            // suffix unwritten because the graph exposes it only through the
            // copy destination. The attention prefix and cache are the
            // required observable outputs of this graph.
            for (size_t i = 0; i < static_cast<size_t>(gdn_attention_elements); ++i) {
                CHECK(std::fabs(gdn_fused_output[i] - gdn_direct_output[i]) < 5e-4f);
            }
            for (size_t i = 0; i < gdn_fused_cache.size(); ++i) {
                CHECK(std::fabs(gdn_fused_cache[i] - gdn_direct_cache[i]) < 5e-4f);
            }

            // Production llama.cpp graphs omit the zero-work snapshot source
            // view but schedule the cache destination view. Exercise that
            // exact form while retaining the source view as the CPY tensor
            // edge and the attention view as an external prefix consumer.
            std::fill(gdn_fused_output.begin(), gdn_fused_output.end(), -1.0f);
            std::fill(gdn_fused_cache.begin(), gdn_fused_cache.end(), -1.0f);
            ggml_backend_buffer_clear(gdn_buffers[6], 0);
            ggml_backend_buffer_clear(gdn_buffers.back(), 0);
            ggml_tensor * gdn_scheduled_view_nodes[] = {
                gdn_output, gdn_cache_view, gdn_copy, gdn_attention_view,
            };
            ggml_cgraph gdn_scheduled_view_graph {};
            gdn_scheduled_view_graph.n_nodes = 4;
            gdn_scheduled_view_graph.nodes = gdn_scheduled_view_nodes;
            CHECK(ggml_backend_graph_compute(
                backend, &gdn_scheduled_view_graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get_async(backend, gdn_output, gdn_fused_output.data(),
                0, ggml_nbytes(gdn_output));
            ggml_backend_tensor_get_async(backend, gdn_cache, gdn_fused_cache.data(),
                0, ggml_nbytes(gdn_cache));
            ggml_backend_synchronize(backend);
            for (size_t i = 0; i < static_cast<size_t>(gdn_attention_elements); ++i) {
                CHECK(std::fabs(gdn_fused_output[i] - gdn_direct_output[i]) < 5e-4f);
            }
            for (size_t i = 0; i < gdn_fused_cache.size(); ++i) {
                CHECK(std::fabs(gdn_fused_cache[i] - gdn_direct_cache[i]) < 5e-4f);
            }

            // A consumer of the snapshot suffix makes cache-only execution
            // illegal. The provider must select the full-output implementation
            // so the observer still sees the recurrent state materialized in
            // the GDN output tensor.
            std::fill(gdn_fused_output.begin(), gdn_fused_output.end(), -1.0f);
            std::fill(gdn_fused_cache.begin(), gdn_fused_cache.end(), -1.0f);
            ggml_backend_buffer_clear(gdn_buffers[6], 0);
            ggml_backend_buffer_clear(gdn_buffers.back(), 0);
            ggml_tensor * gdn_snapshot_observer_nodes[] = {
                gdn_output, gdn_cache_view, gdn_copy,
                gdn_attention_view, gdn_snapshot_observer,
            };
            ggml_cgraph gdn_snapshot_observer_graph {};
            gdn_snapshot_observer_graph.n_nodes = 5;
            gdn_snapshot_observer_graph.nodes = gdn_snapshot_observer_nodes;
            CHECK(ggml_backend_graph_compute(
                backend, &gdn_snapshot_observer_graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get_async(backend, gdn_output, gdn_fused_output.data(),
                0, ggml_nbytes(gdn_output));
            ggml_backend_tensor_get_async(backend, gdn_cache, gdn_fused_cache.data(),
                0, ggml_nbytes(gdn_cache));
            ggml_backend_synchronize(backend);
            for (size_t i = 0; i < gdn_fused_output.size(); ++i) {
                CHECK(std::fabs(gdn_fused_output[i] - gdn_direct_output[i]) < 5e-4f);
            }
            for (size_t i = 0; i < gdn_fused_cache.size(); ++i) {
                CHECK(std::fabs(gdn_fused_cache[i] - gdn_direct_cache[i]) < 5e-4f);
            }
            for (ggml_backend_buffer_t gdn_buffer : gdn_buffers) {
                ggml_backend_buffer_free(gdn_buffer);
            }
        }

        constexpr int64_t ssm_d_conv = 4;
        constexpr int64_t ssm_d_inner = 513;
        constexpr int64_t ssm_tokens = 3;
        constexpr int64_t ssm_sequences = 2;
        constexpr float ssm_intermediate_sentinel = -123.25f;
        ggml_tensor * ssm_state = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F32, ssm_d_conv - 1 + ssm_tokens,
            ssm_d_inner, ssm_sequences);
        ggml_tensor * ssm_weight = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, ssm_d_conv, ssm_d_inner);
        ggml_tensor * ssm_conv = ggml_ssm_conv(ctx, ssm_state, ssm_weight);
        ggml_tensor * ssm_silu = ggml_silu(ctx, ssm_conv);
        CHECK(ssm_state != nullptr && ssm_weight != nullptr &&
            ssm_conv != nullptr && ssm_silu != nullptr);
        ggml_backend_buffer_t ssm_state_buffer =
            ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(ssm_state));
        ggml_backend_buffer_t ssm_weight_buffer =
            ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(ssm_weight));
        ggml_backend_buffer_t ssm_conv_buffer =
            ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(ssm_conv));
        ggml_backend_buffer_t ssm_silu_buffer =
            ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(ssm_silu));
        CHECK(ssm_state_buffer != nullptr && ssm_weight_buffer != nullptr &&
            ssm_conv_buffer != nullptr && ssm_silu_buffer != nullptr);
        CHECK(ggml_backend_tensor_alloc(ssm_state_buffer, ssm_state,
            ggml_backend_buffer_get_base(ssm_state_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(ssm_weight_buffer, ssm_weight,
            ggml_backend_buffer_get_base(ssm_weight_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(ssm_conv_buffer, ssm_conv,
            ggml_backend_buffer_get_base(ssm_conv_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(ssm_silu_buffer, ssm_silu,
            ggml_backend_buffer_get_base(ssm_silu_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_dev_supports_op(dev, ssm_conv));
        CHECK(ggml_backend_dev_supports_op(dev, ssm_silu));

        std::vector<float> ssm_state_host(ggml_nelements(ssm_state));
        std::vector<float> ssm_weight_host(ggml_nelements(ssm_weight));
        std::vector<float> ssm_conv_host(
            ggml_nelements(ssm_conv), ssm_intermediate_sentinel);
        std::vector<float> ssm_silu_host(ggml_nelements(ssm_silu), 0.0f);
        std::vector<float> ssm_expected(ssm_silu_host.size(), 0.0f);
        for (size_t i = 0; i < ssm_state_host.size(); ++i) {
            ssm_state_host[i] = static_cast<float>(static_cast<int>(i * 17 % 43) - 21) * 0.013f;
        }
        for (size_t i = 0; i < ssm_weight_host.size(); ++i) {
            ssm_weight_host[i] = static_cast<float>(static_cast<int>(i * 11 % 31) - 15) * 0.017f;
        }
        const int64_t ssm_state_columns = ssm_d_conv - 1 + ssm_tokens;
        for (int64_t sequence = 0; sequence < ssm_sequences; ++sequence) {
            for (int64_t token = 0; token < ssm_tokens; ++token) {
                for (int64_t channel = 0; channel < ssm_d_inner; ++channel) {
                    float sum = 0.0f;
                    for (int64_t tap = 0; tap < ssm_d_conv; ++tap) {
                        const size_t state_index = static_cast<size_t>(
                            sequence * ssm_state_columns * ssm_d_inner +
                            channel * ssm_state_columns + token + tap);
                        const size_t weight_index = static_cast<size_t>(
                            channel * ssm_d_conv + tap);
                        sum += ssm_state_host[state_index] * ssm_weight_host[weight_index];
                    }
                    const size_t output_index = static_cast<size_t>(
                        sequence * ssm_tokens * ssm_d_inner +
                        token * ssm_d_inner + channel);
                    ssm_expected[output_index] = sum / (1.0f + std::exp(-sum));
                }
            }
        }
        ggml_backend_tensor_set_async(
            backend, ssm_state, ssm_state_host.data(), 0, ggml_nbytes(ssm_state));
        ggml_backend_tensor_set_async(
            backend, ssm_weight, ssm_weight_host.data(), 0, ggml_nbytes(ssm_weight));
        ggml_backend_tensor_set_async(
            backend, ssm_conv, ssm_conv_host.data(), 0, ggml_nbytes(ssm_conv));
        ggml_backend_synchronize(backend);
        ggml_tensor * ssm_nodes[] = { ssm_conv, ssm_silu };
        ggml_cgraph ssm_graph {};
        ssm_graph.n_nodes = 2;
        ssm_graph.nodes = ssm_nodes;
        CHECK(ggml_backend_graph_compute(backend, &ssm_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(
            backend, ssm_conv, ssm_conv_host.data(), 0, ggml_nbytes(ssm_conv));
        ggml_backend_tensor_get_async(
            backend, ssm_silu, ssm_silu_host.data(), 0, ggml_nbytes(ssm_silu));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < ssm_silu_host.size(); ++i) {
            CHECK(std::fabs(ssm_silu_host[i] - ssm_expected[i]) < 5e-5f);
            if (require_ssm_conv_silu) {
                CHECK(ssm_conv_host[i] == ssm_intermediate_sentinel);
            }
        }
        ggml_backend_buffer_free(ssm_state_buffer);
        ggml_backend_buffer_free(ssm_weight_buffer);
        ggml_backend_buffer_free(ssm_conv_buffer);
        ggml_backend_buffer_free(ssm_silu_buffer);

        ggml_tensor * silu_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 513);
        ggml_tensor * silu_output = ggml_silu(ctx, silu_input);
        CHECK(silu_input != nullptr && silu_output != nullptr);
        ggml_backend_buffer_t silu_input_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(silu_input));
        ggml_backend_buffer_t silu_output_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(silu_output));
        CHECK(silu_input_buffer != nullptr && silu_output_buffer != nullptr);
        CHECK(ggml_backend_tensor_alloc(silu_input_buffer, silu_input,
            ggml_backend_buffer_get_base(silu_input_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(silu_output_buffer, silu_output,
            ggml_backend_buffer_get_base(silu_output_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_dev_supports_op(dev, silu_output));
        std::vector<float> silu_input_host(513);
        std::vector<float> silu_output_host(513, -1.0f);
        for (size_t i = 0; i < silu_input_host.size(); ++i) {
            silu_input_host[i] = static_cast<float>(i) * 0.03125f - 8.0f;
        }
        ggml_backend_tensor_set_async(backend, silu_input, silu_input_host.data(), 0, ggml_nbytes(silu_input));
        ggml_backend_synchronize(backend);
        ggml_tensor * silu_nodes[] = { silu_output };
        ggml_cgraph silu_graph {};
        silu_graph.n_nodes = 1;
        silu_graph.nodes = silu_nodes;
        CHECK(ggml_backend_graph_compute(backend, &silu_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(backend, silu_output, silu_output_host.data(), 0, ggml_nbytes(silu_output));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < silu_output_host.size(); ++i) {
            const float x = silu_input_host[i];
            CHECK(std::fabs(silu_output_host[i] - x / (1.0f + std::exp(-x))) < 4e-5f);
        }
        ggml_backend_buffer_free(silu_input_buffer);
        ggml_backend_buffer_free(silu_output_buffer);

        constexpr int64_t gated_columns = 128;
        constexpr int64_t gated_rows = 33;
        constexpr float gated_intermediate_sentinel = -91.75f;
        ggml_tensor * gated_input = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, gated_columns, gated_rows);
        ggml_tensor * gated_other = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, gated_columns, gated_rows);
        ggml_tensor * gated_silu = ggml_silu(ctx, gated_input);
        ggml_tensor * gated_output = ggml_mul(ctx, gated_other, gated_silu);
        CHECK(gated_input && gated_other && gated_silu && gated_output);
        ggml_backend_buffer_t gated_input_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(gated_input));
        ggml_backend_buffer_t gated_other_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(gated_other));
        ggml_backend_buffer_t gated_silu_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(gated_silu));
        ggml_backend_buffer_t gated_output_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(gated_output));
        CHECK(gated_input_buffer && gated_other_buffer &&
            gated_silu_buffer && gated_output_buffer);
        CHECK(ggml_backend_tensor_alloc(gated_input_buffer, gated_input,
            ggml_backend_buffer_get_base(gated_input_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(gated_other_buffer, gated_other,
            ggml_backend_buffer_get_base(gated_other_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(gated_silu_buffer, gated_silu,
            ggml_backend_buffer_get_base(gated_silu_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(gated_output_buffer, gated_output,
            ggml_backend_buffer_get_base(gated_output_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_dev_supports_op(dev, gated_silu));
        CHECK(ggml_backend_dev_supports_op(dev, gated_output));
        const size_t gated_elements = static_cast<size_t>(ggml_nelements(gated_output));
        std::vector<float> gated_input_host(gated_elements);
        std::vector<float> gated_other_host(gated_elements);
        std::vector<float> gated_silu_host(gated_elements, gated_intermediate_sentinel);
        std::vector<float> gated_output_host(gated_elements, 0.0f);
        for (size_t i = 0; i < gated_elements; ++i) {
            gated_input_host[i] = static_cast<float>(static_cast<int>(i * 19 % 101) - 50) * 0.031f;
            gated_other_host[i] = static_cast<float>(static_cast<int>(i * 23 % 83) - 41) * 0.017f;
        }
        ggml_backend_tensor_set_async(
            backend, gated_input, gated_input_host.data(), 0, ggml_nbytes(gated_input));
        ggml_backend_tensor_set_async(
            backend, gated_other, gated_other_host.data(), 0, ggml_nbytes(gated_other));
        ggml_backend_tensor_set_async(
            backend, gated_silu, gated_silu_host.data(), 0, ggml_nbytes(gated_silu));
        ggml_backend_synchronize(backend);
        ggml_tensor * gated_nodes[] = { gated_silu, gated_output };
        ggml_cgraph gated_graph {};
        gated_graph.n_nodes = 2;
        gated_graph.nodes = gated_nodes;
        CHECK(ggml_backend_graph_compute(backend, &gated_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(
            backend, gated_silu, gated_silu_host.data(), 0, ggml_nbytes(gated_silu));
        ggml_backend_tensor_get_async(
            backend, gated_output, gated_output_host.data(), 0, ggml_nbytes(gated_output));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < gated_elements; ++i) {
            const float value = gated_input_host[i];
            const float expected = value / (1.0f + std::exp(-value)) * gated_other_host[i];
            CHECK(std::fabs(gated_output_host[i] - expected) < 6e-5f);
            if (require_attention_output_gate) {
                CHECK(gated_silu_host[i] == gated_intermediate_sentinel);
            }
        }
        ggml_backend_buffer_free(gated_input_buffer);
        ggml_backend_buffer_free(gated_other_buffer);
        ggml_backend_buffer_free(gated_silu_buffer);
        ggml_backend_buffer_free(gated_output_buffer);

        const auto check_unary_output_gate = [&](ggml_unary_op unary_op) {
            constexpr int64_t columns = 37;
            constexpr int64_t rows = 5;
            constexpr float intermediate_sentinel = -73.25f;
            ggml_tensor * input = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, columns, rows);
            ggml_tensor * other = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, columns, rows);
            ggml_tensor * activation = ggml_unary(ctx, input, unary_op);
            ggml_tensor * output = ggml_mul(ctx, activation, other);
            CHECK(input && other && activation && output);
            ggml_backend_buffer_t input_buffer = ggml_backend_buft_alloc_buffer(
                buft, ggml_nbytes(input));
            ggml_backend_buffer_t other_buffer = ggml_backend_buft_alloc_buffer(
                buft, ggml_nbytes(other));
            ggml_backend_buffer_t activation_buffer = ggml_backend_buft_alloc_buffer(
                buft, ggml_nbytes(activation));
            ggml_backend_buffer_t output_buffer = ggml_backend_buft_alloc_buffer(
                buft, ggml_nbytes(output));
            CHECK(input_buffer && other_buffer && activation_buffer && output_buffer);
            CHECK(ggml_backend_tensor_alloc(input_buffer, input,
                ggml_backend_buffer_get_base(input_buffer)) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_tensor_alloc(other_buffer, other,
                ggml_backend_buffer_get_base(other_buffer)) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_tensor_alloc(activation_buffer, activation,
                ggml_backend_buffer_get_base(activation_buffer)) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_tensor_alloc(output_buffer, output,
                ggml_backend_buffer_get_base(output_buffer)) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_dev_supports_op(dev, activation));
            CHECK(ggml_backend_dev_supports_op(dev, output));
            const size_t n = static_cast<size_t>(ggml_nelements(output));
            std::vector<float> input_host(n);
            std::vector<float> other_host(n);
            std::vector<float> activation_host(n, intermediate_sentinel);
            std::vector<float> output_host(n, 0.0f);
            for (size_t i = 0; i < n; ++i) {
                input_host[i] = static_cast<float>(static_cast<int>(i * 17 % 113) - 56) * 0.073f;
                other_host[i] = static_cast<float>(static_cast<int>(i * 29 % 89) - 44) * 0.019f;
            }
            ggml_backend_tensor_set_async(
                backend, input, input_host.data(), 0, ggml_nbytes(input));
            ggml_backend_tensor_set_async(
                backend, other, other_host.data(), 0, ggml_nbytes(other));
            ggml_backend_tensor_set_async(
                backend, activation, activation_host.data(), 0, ggml_nbytes(activation));
            ggml_backend_synchronize(backend);
            ggml_tensor * nodes[] = { activation, output };
            ggml_cgraph graph {};
            graph.n_nodes = 2;
            graph.nodes = nodes;
            CHECK(ggml_backend_graph_compute(backend, &graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get_async(
                backend, activation, activation_host.data(), 0, ggml_nbytes(activation));
            ggml_backend_tensor_get_async(
                backend, output, output_host.data(), 0, ggml_nbytes(output));
            ggml_backend_synchronize(backend);
            for (size_t i = 0; i < n; ++i) {
                const float value = input_host[i];
                const float activated = unary_op == GGML_UNARY_OP_SIGMOID
                    ? 1.0f / (1.0f + std::exp(-value))
                    : std::log1p(std::exp(value));
                CHECK(std::fabs(output_host[i] - activated * other_host[i]) < 8e-5f);
                if (require_attention_output_gate) {
                    CHECK(activation_host[i] == intermediate_sentinel);
                }
            }
            ggml_backend_buffer_free(input_buffer);
            ggml_backend_buffer_free(other_buffer);
            ggml_backend_buffer_free(activation_buffer);
            ggml_backend_buffer_free(output_buffer);
        };
        check_unary_output_gate(GGML_UNARY_OP_SIGMOID);
        check_unary_output_gate(GGML_UNARY_OP_SOFTPLUS);

        constexpr int64_t alpha_columns = 32;
        constexpr int64_t alpha_rows = 33;
        constexpr float alpha_add_sentinel = -61.5f;
        constexpr float alpha_softplus_sentinel = -62.5f;
        ggml_tensor * alpha_input = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, alpha_columns, alpha_rows);
        ggml_tensor * alpha_bias = ggml_new_tensor_1d(
            ctx, GGML_TYPE_F32, alpha_columns);
        ggml_tensor * alpha_scale = ggml_new_tensor_1d(
            ctx, GGML_TYPE_F32, alpha_columns);
        ggml_tensor * alpha_add = ggml_add(ctx, alpha_input, alpha_bias);
        ggml_tensor * alpha_softplus = ggml_softplus(ctx, alpha_add);
        ggml_tensor * alpha_output = ggml_mul(ctx, alpha_softplus, alpha_scale);
        CHECK(alpha_input && alpha_bias && alpha_scale && alpha_add &&
            alpha_softplus && alpha_output);
        ggml_backend_buffer_t alpha_input_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(alpha_input));
        ggml_backend_buffer_t alpha_bias_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(alpha_bias));
        ggml_backend_buffer_t alpha_scale_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(alpha_scale));
        ggml_backend_buffer_t alpha_add_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(alpha_add));
        ggml_backend_buffer_t alpha_softplus_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(alpha_softplus));
        ggml_backend_buffer_t alpha_output_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(alpha_output));
        CHECK(alpha_input_buffer && alpha_bias_buffer && alpha_scale_buffer &&
            alpha_add_buffer && alpha_softplus_buffer && alpha_output_buffer);
        CHECK(ggml_backend_tensor_alloc(alpha_input_buffer, alpha_input,
            ggml_backend_buffer_get_base(alpha_input_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(alpha_bias_buffer, alpha_bias,
            ggml_backend_buffer_get_base(alpha_bias_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(alpha_scale_buffer, alpha_scale,
            ggml_backend_buffer_get_base(alpha_scale_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(alpha_add_buffer, alpha_add,
            ggml_backend_buffer_get_base(alpha_add_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(alpha_softplus_buffer, alpha_softplus,
            ggml_backend_buffer_get_base(alpha_softplus_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(alpha_output_buffer, alpha_output,
            ggml_backend_buffer_get_base(alpha_output_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_dev_supports_op(dev, alpha_add));
        CHECK(ggml_backend_dev_supports_op(dev, alpha_softplus));
        CHECK(ggml_backend_dev_supports_op(dev, alpha_output));
        const size_t alpha_elements = static_cast<size_t>(ggml_nelements(alpha_output));
        std::vector<float> alpha_input_host(alpha_elements);
        std::vector<float> alpha_bias_host(alpha_columns);
        std::vector<float> alpha_scale_host(alpha_columns);
        std::vector<float> alpha_add_host(alpha_elements, alpha_add_sentinel);
        std::vector<float> alpha_softplus_host(alpha_elements, alpha_softplus_sentinel);
        std::vector<float> alpha_output_host(alpha_elements, 0.0f);
        for (size_t i = 0; i < alpha_elements; ++i) {
            alpha_input_host[i] = static_cast<float>(static_cast<int>(i * 13 % 97) - 48) * 0.061f;
        }
        alpha_input_host[0] = 100.0f;
        alpha_input_host[1] = -100.0f;
        for (size_t i = 0; i < static_cast<size_t>(alpha_columns); ++i) {
            alpha_bias_host[i] = static_cast<float>(static_cast<int>(i * 7 % 29) - 14) * 0.037f;
            alpha_scale_host[i] = static_cast<float>(static_cast<int>(i * 11 % 31) - 15) * 0.043f;
        }
        ggml_backend_tensor_set_async(
            backend, alpha_input, alpha_input_host.data(), 0, ggml_nbytes(alpha_input));
        ggml_backend_tensor_set_async(
            backend, alpha_bias, alpha_bias_host.data(), 0, ggml_nbytes(alpha_bias));
        ggml_backend_tensor_set_async(
            backend, alpha_scale, alpha_scale_host.data(), 0, ggml_nbytes(alpha_scale));
        ggml_backend_tensor_set_async(
            backend, alpha_add, alpha_add_host.data(), 0, ggml_nbytes(alpha_add));
        ggml_backend_tensor_set_async(
            backend, alpha_softplus, alpha_softplus_host.data(), 0,
            ggml_nbytes(alpha_softplus));
        ggml_backend_synchronize(backend);
        ggml_tensor * alpha_nodes[] = { alpha_add, alpha_softplus, alpha_output };
        ggml_cgraph alpha_graph {};
        alpha_graph.n_nodes = 3;
        alpha_graph.nodes = alpha_nodes;
        CHECK(ggml_backend_graph_compute(backend, &alpha_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(
            backend, alpha_add, alpha_add_host.data(), 0, ggml_nbytes(alpha_add));
        ggml_backend_tensor_get_async(
            backend, alpha_softplus, alpha_softplus_host.data(), 0,
            ggml_nbytes(alpha_softplus));
        ggml_backend_tensor_get_async(
            backend, alpha_output, alpha_output_host.data(), 0, ggml_nbytes(alpha_output));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < alpha_elements; ++i) {
            const size_t column = i % static_cast<size_t>(alpha_columns);
            const float biased = alpha_input_host[i] + alpha_bias_host[column];
            const float softplus = biased > 20.0f ? biased : std::log1p(std::exp(biased));
            const float expected = softplus * alpha_scale_host[column];
            CHECK(std::fabs(alpha_output_host[i] - expected) < 8e-5f);
            if (require_attention_output_gate) {
                CHECK(alpha_add_host[i] == alpha_add_sentinel);
                CHECK(alpha_softplus_host[i] == alpha_softplus_sentinel);
            }
        }
        ggml_backend_buffer_free(alpha_input_buffer);
        ggml_backend_buffer_free(alpha_bias_buffer);
        ggml_backend_buffer_free(alpha_scale_buffer);
        ggml_backend_buffer_free(alpha_add_buffer);
        ggml_backend_buffer_free(alpha_softplus_buffer);
        ggml_backend_buffer_free(alpha_output_buffer);

        ggml_tensor * gate_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 3);
        ggml_tensor * up_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 3);
        ggml_tensor * swiglu_tensor = ggml_swiglu_split(ctx, gate_tensor, up_tensor);
        CHECK(gate_tensor != nullptr && up_tensor != nullptr && swiglu_tensor != nullptr);
        ggml_backend_buffer_t gate_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(gate_tensor));
        ggml_backend_buffer_t up_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(up_tensor));
        ggml_backend_buffer_t swiglu_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(swiglu_tensor));
        CHECK(gate_buffer != nullptr && up_buffer != nullptr && swiglu_buffer != nullptr);
        CHECK(ggml_backend_tensor_alloc(gate_buffer, gate_tensor, ggml_backend_buffer_get_base(gate_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(up_buffer, up_tensor, ggml_backend_buffer_get_base(up_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(swiglu_buffer, swiglu_tensor, ggml_backend_buffer_get_base(swiglu_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_dev_supports_op(dev, swiglu_tensor));
        const size_t swiglu_elements = static_cast<size_t>(ggml_nelements(swiglu_tensor));
        std::vector<float> gate_host(swiglu_elements);
        std::vector<float> up_host(swiglu_elements);
        std::vector<float> swiglu_host(swiglu_elements, -1.0f);
        for (size_t i = 0; i < swiglu_elements; ++i) {
            gate_host[i] = static_cast<float>(i % 257) * 0.025f - 3.0f;
            up_host[i] = static_cast<float>(i % 113) * 0.017f - 0.8f;
        }
        ggml_backend_tensor_set_async(backend, gate_tensor, gate_host.data(), 0, ggml_nbytes(gate_tensor));
        ggml_backend_tensor_set_async(backend, up_tensor, up_host.data(), 0, ggml_nbytes(up_tensor));
        ggml_backend_synchronize(backend);
        ggml_tensor * swiglu_nodes[] = { swiglu_tensor };
        ggml_cgraph swiglu_graph {};
        swiglu_graph.n_nodes = 1;
        swiglu_graph.nodes = swiglu_nodes;
        CHECK(ggml_backend_graph_compute(backend, &swiglu_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(backend, swiglu_tensor, swiglu_host.data(), 0, ggml_nbytes(swiglu_tensor));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < swiglu_elements; ++i) {
            const float expected = gate_host[i] / (1.0f + std::exp(-gate_host[i])) * up_host[i];
            CHECK(std::fabs(swiglu_host[i] - expected) < 5e-5f);
        }
        ggml_backend_buffer_free(gate_buffer);
        ggml_backend_buffer_free(up_buffer);
        ggml_backend_buffer_free(swiglu_buffer);

        // Softmax is a direct FlagOS operator.  Exercise both ABI variants:
        // an unmasked F32 matrix and the F16 mask layout used by attention.
        const int64_t softmax_cols = 4096;
        const int64_t softmax_rows = 3;
        ggml_tensor * softmax_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
            softmax_cols, softmax_rows);
        ggml_tensor * softmax_output = ggml_soft_max(ctx, softmax_input);
        CHECK(softmax_input && softmax_output);
        ggml_backend_buffer_t softmax_input_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(softmax_input));
        ggml_backend_buffer_t softmax_output_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(softmax_output));
        CHECK(softmax_input_buffer && softmax_output_buffer);
        CHECK(ggml_backend_tensor_alloc(softmax_input_buffer, softmax_input,
            ggml_backend_buffer_get_base(softmax_input_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(softmax_output_buffer, softmax_output,
            ggml_backend_buffer_get_base(softmax_output_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_dev_supports_op(dev, softmax_output));
        std::vector<float> softmax_input_host(static_cast<size_t>(softmax_cols * softmax_rows));
        std::vector<float> softmax_output_host(softmax_input_host.size(), -1.0f);
        std::vector<float> softmax_expected(softmax_input_host.size(), 0.0f);
        for (int64_t row = 0; row < softmax_rows; ++row) {
            float max_value = -INFINITY;
            for (int64_t col = 0; col < softmax_cols; ++col) {
                const size_t index = static_cast<size_t>(row * softmax_cols + col);
                softmax_input_host[index] = static_cast<float>((col * 17 + row * 3) % 101) * 0.021f - 1.0f;
                max_value = std::max(max_value, softmax_input_host[index]);
            }
            float sum = 0.0f;
            for (int64_t col = 0; col < softmax_cols; ++col) {
                const size_t index = static_cast<size_t>(row * softmax_cols + col);
                softmax_expected[index] = std::exp(softmax_input_host[index] - max_value);
                sum += softmax_expected[index];
            }
            for (int64_t col = 0; col < softmax_cols; ++col) {
                softmax_expected[static_cast<size_t>(row * softmax_cols + col)] /= sum;
            }
        }
        ggml_backend_tensor_set_async(backend, softmax_input, softmax_input_host.data(),
            0, ggml_nbytes(softmax_input));
        ggml_backend_synchronize(backend);
        ggml_tensor * softmax_nodes[] = { softmax_output };
        ggml_cgraph softmax_graph {};
        softmax_graph.n_nodes = 1;
        softmax_graph.nodes = softmax_nodes;
        CHECK(ggml_backend_graph_compute(backend, &softmax_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(backend, softmax_output, softmax_output_host.data(),
            0, ggml_nbytes(softmax_output));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < softmax_output_host.size(); ++i) {
            CHECK(std::fabs(softmax_output_host[i] - softmax_expected[i]) < 5e-5f);
        }
        ggml_backend_buffer_free(softmax_input_buffer);
        ggml_backend_buffer_free(softmax_output_buffer);

        ggml_tensor * masked_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
            softmax_cols, softmax_rows);
        ggml_tensor * masked_values = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,
            softmax_cols, softmax_rows);
        ggml_tensor * masked_output = ggml_soft_max_ext(ctx, masked_input, masked_values,
            0.125f, 0.0f);
        CHECK(masked_input && masked_values && masked_output);
        ggml_backend_buffer_t masked_input_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(masked_input));
        ggml_backend_buffer_t masked_values_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(masked_values));
        ggml_backend_buffer_t masked_output_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(masked_output));
        CHECK(masked_input_buffer && masked_values_buffer && masked_output_buffer);
        CHECK(ggml_backend_tensor_alloc(masked_input_buffer, masked_input,
            ggml_backend_buffer_get_base(masked_input_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(masked_values_buffer, masked_values,
            ggml_backend_buffer_get_base(masked_values_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(masked_output_buffer, masked_output,
            ggml_backend_buffer_get_base(masked_output_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_dev_supports_op(dev, masked_output));
        std::vector<ggml_fp16_t> masked_values_host(static_cast<size_t>(softmax_cols * softmax_rows));
        std::vector<float> masked_input_host(softmax_input_host.size());
        std::vector<float> masked_output_host(masked_input_host.size(), -1.0f);
        std::vector<float> masked_expected(masked_input_host.size(), 0.0f);
        for (size_t i = 0; i < masked_input_host.size(); ++i) {
            masked_input_host[i] = static_cast<float>((i * 29) % 131) * 0.013f - 0.8f;
            const float bias = static_cast<float>((i * 7) % 19) * 0.011f - 0.1f;
            masked_values_host[i] = ggml_fp32_to_fp16(bias);
        }
        for (int64_t row = 0; row < softmax_rows; ++row) {
            float max_value = -INFINITY;
            for (int64_t col = 0; col < softmax_cols; ++col) {
                const size_t index = static_cast<size_t>(row * softmax_cols + col);
                const float value = masked_input_host[index] * 0.125f +
                    ggml_fp16_to_fp32(masked_values_host[index]);
                max_value = std::max(max_value, value);
                masked_expected[index] = value;
            }
            float sum = 0.0f;
            for (int64_t col = 0; col < softmax_cols; ++col) {
                const size_t index = static_cast<size_t>(row * softmax_cols + col);
                masked_expected[index] = std::exp(masked_expected[index] - max_value);
                sum += masked_expected[index];
            }
            for (int64_t col = 0; col < softmax_cols; ++col) {
                masked_expected[static_cast<size_t>(row * softmax_cols + col)] /= sum;
            }
        }
        ggml_backend_tensor_set_async(backend, masked_input, masked_input_host.data(),
            0, ggml_nbytes(masked_input));
        ggml_backend_tensor_set_async(backend, masked_values, masked_values_host.data(),
            0, ggml_nbytes(masked_values));
        ggml_backend_synchronize(backend);
        ggml_tensor * masked_nodes[] = { masked_output };
        ggml_cgraph masked_graph {};
        masked_graph.n_nodes = 1;
        masked_graph.nodes = masked_nodes;
        CHECK(ggml_backend_graph_compute(backend, &masked_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(backend, masked_output, masked_output_host.data(),
            0, ggml_nbytes(masked_output));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < masked_output_host.size(); ++i) {
            CHECK(std::fabs(masked_output_host[i] - masked_expected[i]) < 7e-5f);
        }
        ggml_backend_buffer_free(masked_input_buffer);
        ggml_backend_buffer_free(masked_values_buffer);
        ggml_backend_buffer_free(masked_output_buffer);

        const int64_t attn_dim = 128;
        const int64_t attn_q_heads = 4;
        const int64_t attn_kv_heads = 2;
        const int64_t attn_keys = 37;
        ggml_tensor * attn_q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, attn_dim, 1, attn_q_heads);
        ggml_tensor * attn_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, attn_dim, attn_keys, attn_kv_heads);
        ggml_tensor * attn_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, attn_dim, attn_keys, attn_kv_heads);
        ggml_tensor * attn_mask = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, attn_keys);
        CHECK(attn_q != nullptr && attn_k != nullptr && attn_v != nullptr && attn_mask != nullptr);
        // Match the permuted [head_dim, token, head] layout produced by the
        // llama graph before FLASH_ATTN_EXT.
        attn_q->nb[1] = attn_dim * attn_q_heads * sizeof(float);
        attn_q->nb[2] = attn_dim * sizeof(float);
        attn_k->nb[1] = attn_dim * attn_kv_heads * sizeof(ggml_fp16_t);
        attn_k->nb[2] = attn_dim * sizeof(ggml_fp16_t);
        attn_v->nb[1] = attn_dim * attn_kv_heads * sizeof(ggml_fp16_t);
        attn_v->nb[2] = attn_dim * sizeof(ggml_fp16_t);
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(attn_dim));
        ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, attn_q, attn_k, attn_v, attn_mask,
            attn_scale, 0.0f, 0.0f);
        CHECK(attn_out != nullptr);
        ggml_backend_buffer_t attn_q_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(attn_q));
        ggml_backend_buffer_t attn_k_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(attn_k));
        ggml_backend_buffer_t attn_v_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(attn_v));
        ggml_backend_buffer_t attn_mask_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(attn_mask));
        ggml_backend_buffer_t attn_out_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(attn_out));
        CHECK(attn_q_buffer && attn_k_buffer && attn_v_buffer && attn_mask_buffer && attn_out_buffer);
        CHECK(ggml_backend_tensor_alloc(attn_q_buffer, attn_q, ggml_backend_buffer_get_base(attn_q_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(attn_k_buffer, attn_k, ggml_backend_buffer_get_base(attn_k_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(attn_v_buffer, attn_v, ggml_backend_buffer_get_base(attn_v_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(attn_mask_buffer, attn_mask, ggml_backend_buffer_get_base(attn_mask_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(attn_out_buffer, attn_out, ggml_backend_buffer_get_base(attn_out_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_dev_supports_op(dev, attn_out));
        std::vector<float> attn_q_host(ggml_nbytes(attn_q) / sizeof(float), 0.0f);
        std::vector<ggml_fp16_t> attn_k_host(ggml_nbytes(attn_k) / sizeof(ggml_fp16_t));
        std::vector<ggml_fp16_t> attn_v_host(ggml_nbytes(attn_v) / sizeof(ggml_fp16_t));
        std::vector<ggml_fp16_t> attn_mask_host(attn_keys, ggml_fp32_to_fp16(0.0f));
        std::vector<float> attn_expected(static_cast<size_t>(attn_dim * attn_q_heads), 0.0f);
        std::vector<float> attn_output(attn_expected.size(), -1.0f);
        auto q_at = [&](int64_t dim, int64_t head) -> float & {
            return attn_q_host[static_cast<size_t>((dim * sizeof(float) + head * attn_q->nb[2]) / sizeof(float))];
        };
        auto kv_at = [&](std::vector<ggml_fp16_t> & data, int64_t dim, int64_t token, int64_t head) -> ggml_fp16_t & {
            const size_t byte_offset = static_cast<size_t>(dim * sizeof(ggml_fp16_t) + token * attn_k->nb[1] + head * attn_k->nb[2]);
            return data[byte_offset / sizeof(ggml_fp16_t)];
        };
        for (int64_t head = 0; head < attn_q_heads; ++head) {
            for (int64_t dim = 0; dim < attn_dim; ++dim) {
                q_at(dim, head) = static_cast<float>((dim + 3 * head) % 29) * 0.01f - 0.12f;
            }
        }
        for (int64_t head = 0; head < attn_kv_heads; ++head) {
            for (int64_t token = 0; token < attn_keys; ++token) {
                for (int64_t dim = 0; dim < attn_dim; ++dim) {
                    kv_at(attn_k_host, dim, token, head) = ggml_fp32_to_fp16(static_cast<float>((dim + token + head) % 23) * 0.01f - 0.1f);
                    kv_at(attn_v_host, dim, token, head) = ggml_fp32_to_fp16(static_cast<float>((2 * dim + token + 3 * head) % 31) * 0.008f - 0.09f);
                }
            }
        }
        for (int64_t head = 0; head < attn_q_heads; ++head) {
            const int64_t kv_head = head / (attn_q_heads / attn_kv_heads);
            std::vector<float> scores(attn_keys);
            float max_score = -INFINITY;
            for (int64_t token = 0; token < attn_keys; ++token) {
                float score = 0.0f;
                for (int64_t dim = 0; dim < attn_dim; ++dim) {
                    score += q_at(dim, head) * ggml_fp16_to_fp32(kv_at(attn_k_host, dim, token, kv_head));
                }
                scores[static_cast<size_t>(token)] = score * attn_scale;
                max_score = std::max(max_score, scores[static_cast<size_t>(token)]);
            }
            float sum = 0.0f;
            for (float & score : scores) { score = std::exp(score - max_score); sum += score; }
            for (int64_t dim = 0; dim < attn_dim; ++dim) {
                float value = 0.0f;
                for (int64_t token = 0; token < attn_keys; ++token) {
                    value += scores[static_cast<size_t>(token)] / sum * ggml_fp16_to_fp32(kv_at(attn_v_host, dim, token, kv_head));
                }
                attn_expected[static_cast<size_t>(head * attn_dim + dim)] = value;
            }
        }
        ggml_backend_tensor_set_async(backend, attn_q, attn_q_host.data(), 0, ggml_nbytes(attn_q));
        ggml_backend_tensor_set_async(backend, attn_k, attn_k_host.data(), 0, ggml_nbytes(attn_k));
        ggml_backend_tensor_set_async(backend, attn_v, attn_v_host.data(), 0, ggml_nbytes(attn_v));
        ggml_backend_tensor_set_async(backend, attn_mask, attn_mask_host.data(), 0, ggml_nbytes(attn_mask));
        ggml_backend_synchronize(backend);
        ggml_tensor * attn_nodes[] = { attn_out };
        ggml_cgraph attn_graph {};
        attn_graph.n_nodes = 1;
        attn_graph.nodes = attn_nodes;
        CHECK(ggml_backend_graph_compute(backend, &attn_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(backend, attn_out, attn_output.data(), 0, ggml_nbytes(attn_out));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < attn_output.size(); ++i) {
            CHECK(std::fabs(attn_output[i] - attn_expected[i]) < 4e-3f);
        }
        ggml_backend_buffer_free(attn_q_buffer);
        ggml_backend_buffer_free(attn_k_buffer);
        ggml_backend_buffer_free(attn_v_buffer);
        ggml_backend_buffer_free(attn_mask_buffer);
        ggml_backend_buffer_free(attn_out_buffer);

        // Prefill attention uses a tiled query dimension and the same
        // provider-neutral FLASH_ATTN_EXT pattern with a distinct selector.
        const int64_t prefill_query_length = 5;
        const int64_t prefill_key_length = 9;
        const int64_t prefill_q_heads = 4;
        const int64_t prefill_kv_heads = 2;
        ggml_tensor * prefill_q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
            attn_dim, prefill_query_length, prefill_q_heads);
        ggml_tensor * prefill_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,
            attn_dim, prefill_key_length, prefill_kv_heads);
        ggml_tensor * prefill_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,
            attn_dim, prefill_key_length, prefill_kv_heads);
        ggml_tensor * prefill_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,
            prefill_key_length, prefill_query_length);
        // Use the [token, head, dim] physical layout emitted by the llama
        // attention graph; the provider ABI carries explicit strides.
        prefill_q->nb[1] = attn_dim * prefill_q_heads * sizeof(float);
        prefill_q->nb[2] = attn_dim * sizeof(float);
        prefill_k->nb[1] = attn_dim * prefill_kv_heads * sizeof(ggml_fp16_t);
        prefill_k->nb[2] = attn_dim * sizeof(ggml_fp16_t);
        prefill_v->nb[1] = attn_dim * prefill_kv_heads * sizeof(ggml_fp16_t);
        prefill_v->nb[2] = attn_dim * sizeof(ggml_fp16_t);
        ggml_tensor * prefill_out = ggml_flash_attn_ext(ctx, prefill_q, prefill_k, prefill_v,
            prefill_mask, attn_scale, 0.0f, 0.0f);
        CHECK(prefill_q && prefill_k && prefill_v && prefill_mask && prefill_out);
        CHECK(prefill_out->ne[0] == attn_dim && prefill_out->ne[1] == prefill_q_heads &&
            prefill_out->ne[2] == prefill_query_length);
        ggml_backend_buffer_t prefill_q_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(prefill_q));
        ggml_backend_buffer_t prefill_k_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(prefill_k));
        ggml_backend_buffer_t prefill_v_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(prefill_v));
        ggml_backend_buffer_t prefill_mask_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(prefill_mask));
        ggml_backend_buffer_t prefill_out_buffer = ggml_backend_buft_alloc_buffer(
            buft, ggml_nbytes(prefill_out));
        CHECK(prefill_q_buffer && prefill_k_buffer && prefill_v_buffer &&
            prefill_mask_buffer && prefill_out_buffer);
        CHECK(ggml_backend_tensor_alloc(prefill_q_buffer, prefill_q,
            ggml_backend_buffer_get_base(prefill_q_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(prefill_k_buffer, prefill_k,
            ggml_backend_buffer_get_base(prefill_k_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(prefill_v_buffer, prefill_v,
            ggml_backend_buffer_get_base(prefill_v_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(prefill_mask_buffer, prefill_mask,
            ggml_backend_buffer_get_base(prefill_mask_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(prefill_out_buffer, prefill_out,
            ggml_backend_buffer_get_base(prefill_out_buffer)) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_dev_supports_op(dev, prefill_out));

        const size_t prefill_q_elements = static_cast<size_t>(attn_dim * prefill_query_length * prefill_q_heads);
        const size_t prefill_kv_elements = static_cast<size_t>(attn_dim * prefill_key_length * prefill_kv_heads);
        std::vector<float> prefill_q_host(prefill_q_elements);
        std::vector<ggml_fp16_t> prefill_k_host(prefill_kv_elements);
        std::vector<ggml_fp16_t> prefill_v_host(prefill_kv_elements);
        std::vector<ggml_fp16_t> prefill_mask_host(
            static_cast<size_t>(prefill_key_length * prefill_query_length));
        std::vector<float> prefill_output_host(
            static_cast<size_t>(attn_dim * prefill_q_heads * prefill_query_length), -1.0f);
        std::vector<float> prefill_expected(prefill_output_host.size(), 0.0f);
        auto prefill_q_at = [&](int64_t query, int64_t head, int64_t dim) -> float & {
            return prefill_q_host[static_cast<size_t>(
                query * prefill_q_heads * attn_dim + head * attn_dim + dim)];
        };
        auto prefill_kv_at = [&](std::vector<ggml_fp16_t> & data, int64_t token,
                                 int64_t head, int64_t dim) -> ggml_fp16_t & {
            return data[static_cast<size_t>(
                token * prefill_kv_heads * attn_dim + head * attn_dim + dim)];
        };
        for (int64_t query = 0; query < prefill_query_length; ++query) {
            for (int64_t head = 0; head < prefill_q_heads; ++head) {
                for (int64_t dim = 0; dim < attn_dim; ++dim) {
                    prefill_q_at(query, head, dim) =
                        static_cast<float>((query * 11 + head * 7 + dim) % 37) * 0.013f - 0.2f;
                }
            }
        }
        for (int64_t head = 0; head < prefill_kv_heads; ++head) {
            for (int64_t token = 0; token < prefill_key_length; ++token) {
                for (int64_t dim = 0; dim < attn_dim; ++dim) {
                    prefill_kv_at(prefill_k_host, token, head, dim) = ggml_fp32_to_fp16(
                        static_cast<float>((head * 5 + token * 3 + dim) % 43) * 0.009f - 0.15f);
                    prefill_kv_at(prefill_v_host, token, head, dim) = ggml_fp32_to_fp16(
                        static_cast<float>((head * 3 + token * 7 + dim * 2) % 47) * 0.008f - 0.12f);
                }
            }
        }
        for (int64_t query = 0; query < prefill_query_length; ++query) {
            const int64_t query_position = prefill_key_length - prefill_query_length + query;
            for (int64_t token = 0; token < prefill_key_length; ++token) {
                const float value = token > query_position ? -INFINITY : 0.0f;
                prefill_mask_host[static_cast<size_t>(token + query * prefill_key_length)] =
                    ggml_fp32_to_fp16(value);
            }
        }
        for (int64_t query = 0; query < prefill_query_length; ++query) {
            for (int64_t head = 0; head < prefill_q_heads; ++head) {
                const int64_t kv_head = head / (prefill_q_heads / prefill_kv_heads);
                std::vector<float> scores(static_cast<size_t>(prefill_key_length));
                float max_score = -INFINITY;
                for (int64_t token = 0; token < prefill_key_length; ++token) {
                    float score = 0.0f;
                    for (int64_t dim = 0; dim < attn_dim; ++dim) {
                        score += prefill_q_at(query, head, dim) * ggml_fp16_to_fp32(
                            prefill_kv_at(prefill_k_host, token, kv_head, dim));
                    }
                    score *= attn_scale;
                    score += ggml_fp16_to_fp32(prefill_mask_host[static_cast<size_t>(
                        token + query * prefill_key_length)]);
                    scores[static_cast<size_t>(token)] = score;
                    max_score = std::max(max_score, score);
                }
                float sum = 0.0f;
                for (float & score : scores) {
                    score = std::exp(score - max_score);
                    sum += score;
                }
                for (int64_t dim = 0; dim < attn_dim; ++dim) {
                    float value = 0.0f;
                    for (int64_t token = 0; token < prefill_key_length; ++token) {
                        value += scores[static_cast<size_t>(token)] / sum * ggml_fp16_to_fp32(
                            prefill_kv_at(prefill_v_host, token, kv_head, dim));
                    }
                    prefill_expected[static_cast<size_t>(
                        query * prefill_q_heads * attn_dim + head * attn_dim + dim)] = value;
                }
            }
        }
        ggml_backend_tensor_set_async(backend, prefill_q, prefill_q_host.data(),
            0, ggml_nbytes(prefill_q));
        ggml_backend_tensor_set_async(backend, prefill_k, prefill_k_host.data(),
            0, ggml_nbytes(prefill_k));
        ggml_backend_tensor_set_async(backend, prefill_v, prefill_v_host.data(),
            0, ggml_nbytes(prefill_v));
        ggml_backend_tensor_set_async(backend, prefill_mask, prefill_mask_host.data(),
            0, ggml_nbytes(prefill_mask));
        ggml_backend_synchronize(backend);
        ggml_tensor * prefill_nodes[] = { prefill_out };
        ggml_cgraph prefill_graph {};
        prefill_graph.n_nodes = 1;
        prefill_graph.nodes = prefill_nodes;
        CHECK(ggml_backend_graph_compute(backend, &prefill_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(backend, prefill_out, prefill_output_host.data(),
            0, ggml_nbytes(prefill_out));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < prefill_output_host.size(); ++i) {
            CHECK(std::fabs(prefill_output_host[i] - prefill_expected[i]) < 1e-2f);
        }
        ggml_backend_buffer_free(prefill_q_buffer);
        ggml_backend_buffer_free(prefill_k_buffer);
        ggml_backend_buffer_free(prefill_v_buffer);
        ggml_backend_buffer_free(prefill_mask_buffer);
        ggml_backend_buffer_free(prefill_out_buffer);
    }
#endif

    ggml_backend_free(backend);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::printf("FlagOS AMD checks passed: %s (%s)\n", props.name, props.description);
    return 0;
}
