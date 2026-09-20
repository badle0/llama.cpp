#include <hip/hip_runtime.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "triton_jit/device_ptr.h"
#include "triton_jit/triton_jit_function.h"

namespace {

void check_hip(hipError_t result) {
    if (result != hipSuccess) {
        throw std::runtime_error(hipGetErrorString(result));
    }
}

class device_buffer {
public:
    explicit device_buffer(size_t bytes) {
        check_hip(hipMalloc(&data_, bytes));
    }

    ~device_buffer() {
        if (data_ != nullptr) {
            (void) hipFree(data_);
        }
    }

    float * data() const {
        return static_cast<float *>(data_);
    }

private:
    void * data_ = nullptr;
};

} // namespace

int main() {
    constexpr int rows = 2;
    constexpr int n_cols = 64;
    constexpr int elements = rows * n_cols;
    constexpr int block_size = 64;
    constexpr float epsilon = 1.0e-6F;

    std::vector<float> x(elements);
    std::vector<float> bias(elements);
    std::vector<float> weight(n_cols);
    std::vector<float> residual(elements);
    std::vector<float> output(elements);
    for (int i = 0; i < elements; ++i) {
        x[i] = static_cast<float>((i % 17) - 8) * 0.125F;
        bias[i] = static_cast<float>((i % 5) - 2) * 0.0625F;
    }
    for (int i = 0; i < n_cols; ++i) {
        weight[i] = 0.5F + static_cast<float>(i) / 128.0F;
    }

    device_buffer device_residual(elements * sizeof(float));
    device_buffer device_output(elements * sizeof(float));
    device_buffer device_x(elements * sizeof(float));
    device_buffer device_bias(elements * sizeof(float));
    device_buffer device_weight(n_cols * sizeof(float));
    hipStream_t stream = nullptr;
    check_hip(hipStreamCreate(&stream));
    check_hip(hipMemcpyAsync(device_x.data(), x.data(), elements * sizeof(float), hipMemcpyHostToDevice, stream));
    check_hip(hipMemcpyAsync(device_bias.data(), bias.data(), elements * sizeof(float), hipMemcpyHostToDevice, stream));
    check_hip(hipMemcpyAsync(device_weight.data(), weight.data(), n_cols * sizeof(float), hipMemcpyHostToDevice, stream));

    const std::filesystem::path source = std::filesystem::path(FLAGOS_TRITON_KERNEL_DIR) / "transformer.py";
    const auto & kernel = triton_jit::TritonJITFunction::get_instance(
        source.string(), "flagos_add_rms_norm_mul_residual_f32");
    kernel(stream,
           rows, 1, 1,
           1, 1,
           triton_jit::device_ptr(device_residual.data()),
           triton_jit::device_ptr(device_output.data()),
           triton_jit::device_ptr(device_x.data()),
           triton_jit::device_ptr(device_bias.data()),
           triton_jit::device_ptr(device_weight.data()),
           n_cols,
           epsilon,
           block_size);

    check_hip(hipMemcpyAsync(residual.data(), device_residual.data(), elements * sizeof(float), hipMemcpyDeviceToHost, stream));
    check_hip(hipMemcpyAsync(output.data(), device_output.data(), elements * sizeof(float), hipMemcpyDeviceToHost, stream));
    check_hip(hipStreamSynchronize(stream));
    check_hip(hipStreamDestroy(stream));

    for (int row = 0; row < rows; ++row) {
        float sum_squares = 0.0F;
        for (int col = 0; col < n_cols; ++col) {
            const float value = x[row * n_cols + col] + bias[row * n_cols + col];
            sum_squares += value * value;
            if (std::abs(residual[row * n_cols + col] - value) > 1.0e-6F) {
                throw std::runtime_error("FlagOS JIT residual output mismatch");
            }
        }
        const float inverse_rms = 1.0F / std::sqrt(sum_squares / n_cols + epsilon);
        for (int col = 0; col < n_cols; ++col) {
            const float value = x[row * n_cols + col] + bias[row * n_cols + col];
            const float expected = value * inverse_rms * weight[col];
            if (std::abs(output[row * n_cols + col] - expected) > 2.0e-5F) {
                throw std::runtime_error("FlagOS JIT normalized output mismatch");
            }
        }
    }

    std::cout << "flagos-check-amd-jit: all checks passed\n";
    return 0;
}
