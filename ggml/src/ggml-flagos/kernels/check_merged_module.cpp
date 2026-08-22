// Load the merged FlagOS module once, resolve every kernel the backend needs,
// and exercise the scalar argument ABI of the strided-copy launcher.
#include <cuda.h>
#include <cmath>
#include <cstdio>
#include <vector>

static const char * en(CUresult r) {
    const char * n = nullptr; cuGetErrorName(r, &n); return n ? n : "unknown";
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::printf("usage: %s <merged.cubin>\n", argv[0]); return 2; }

    const char * kernels[] = {
        "flagos_add_f32",
        "flagos_mul_f32",
        "flagos_scale_f32",
        "flagos_copy_f32",
        "flagos_copy_strided_f32",
        "flagos_swiglu_split_f32",
        "flagos_set_rows_f32_f16",
        "flagos_flash_attn_decode_f32_f16",
        "flagos_rope_neox_f32",
        "flagos_rms_norm_f32",
        "flagos_rms_norm_mul_f32",
        "flagos_cast_f32_f16",
        "flagos_cast_f16_f32",
        "flagos_dequant_q4_k_f16",
        "flagos_dequant_q6_k_f16",
        "flagos_mul_mat_q4_k_f32",
        "flagos_mul_mat_q4_k_f32_batched",
        "flagos_mul_mat_q6_k_f32",
        "flagos_mul_mat_q6_k_f32_batched",
        "flagos_get_rows_q4_k_f32",
        "flagos_get_rows_q6_k_f32",
        "flagos_get_rows_f32",
        "flagos_ssm_conv_f32",
        "flagos_sub_f32",
        "flagos_div_f32",
        "flagos_sigmoid_f32",
        "flagos_exp_f32",
        "flagos_softplus_f32",
        "flagos_fill_f32",
        "flagos_sum_rows_f32",
        "flagos_l2_norm_f32",
        "flagos_norm_f32",
        "flagos_cumsum_f32",
        "flagos_soft_max_f32",
    };
    const int total = sizeof(kernels) / sizeof(kernels[0]);

    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) { std::printf("cuInit -> %s\n", en(r)); return 1; }
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    CUmodule mod = nullptr;
    r = cuModuleLoad(&mod, argv[1]);
    std::printf("cuModuleLoad -> %s  (1 load for %d kernels)\n", en(r), total);
    if (r != CUDA_SUCCESS) return 1;

    int ok = 0;
    for (int i = 0; i < total; ++i) {
        CUfunction fn = nullptr;
        CUresult fr = cuModuleGetFunction(&fn, mod, kernels[i]);
        std::printf("  %-36s %s\n", kernels[i], fr == CUDA_SUCCESS ? "OK" : en(fr));
        ok += (fr == CUDA_SUCCESS);
    }

    CUfunction copy = nullptr;
    r = cuModuleGetFunction(&copy, mod, "flagos_copy_strided_f32");
    if (r == CUDA_SUCCESS) {
        constexpr int ne0 = 1;
        constexpr int ne1 = 3;
        constexpr int ne2 = 2;
        constexpr int ne3 = 4;
        constexpr int n = ne0 * ne1 * ne2 * ne3;
        std::vector<float> input(n);
        std::vector<float> output(n, -1.0f);
        std::vector<float> expected(n);
        for (int i = 0; i < n; ++i) {
            input[i] = static_cast<float>(i);
        }
        for (int i3 = 0; i3 < ne3; ++i3) {
            for (int i2 = 0; i2 < ne2; ++i2) {
                for (int i1 = 0; i1 < ne1; ++i1) {
                    for (int i0 = 0; i0 < ne0; ++i0) {
                        const int logical = i0 + ne0 * (i1 + ne1 * (i2 + ne2 * i3));
                        const int physical = i0 + 2 * i1 + i2 + 6 * i3;
                        expected[logical] = input[physical];
                    }
                }
            }
        }

        CUdeviceptr input_device = 0;
        CUdeviceptr output_device = 0;
        r = cuMemAlloc(&input_device, input.size() * sizeof(float));
        if (r == CUDA_SUCCESS) r = cuMemAlloc(&output_device, output.size() * sizeof(float));
        if (r == CUDA_SUCCESS) r = cuMemcpyHtoD(input_device, input.data(), input.size() * sizeof(float));
        if (r == CUDA_SUCCESS) r = cuMemcpyHtoD(output_device, output.data(), output.size() * sizeof(float));

        int xe0 = ne0, xe1 = ne1, xe2 = ne2, xe3 = ne3;
        int ye0 = ne0, ye1 = ne1, ye2 = ne2, ye3 = ne3;
        int sx0 = 1, sx1 = 2, sx2 = 1, sx3 = 6;
        int sy0 = 1, sy1 = 1, sy2 = 3, sy3 = 6;
        int n_elements = n;
        void * args[] = {
            &input_device, &output_device,
            &xe0, &xe1, &xe2, &xe3,
            &ye0, &ye1, &ye2, &ye3,
            &sx0, &sx1, &sx2, &sx3,
            &sy0, &sy1, &sy2, &sy3,
            &n_elements,
        };
        if (r == CUDA_SUCCESS) {
            r = cuLaunchKernel(copy, 1, 1, 1, 128, 1, 1, 0, nullptr, args, nullptr);
        }
        if (r == CUDA_SUCCESS) r = cuCtxSynchronize();
        if (r == CUDA_SUCCESS) r = cuMemcpyDtoH(output.data(), output_device, output.size() * sizeof(float));

        bool matches = r == CUDA_SUCCESS;
        int first_mismatch = -1;
        for (int i = 0; matches && i < n; ++i) {
            if (std::fabs(output[i] - expected[i]) != 0.0f) {
                matches = false;
                first_mismatch = i;
            }
        }
        std::printf("\n  %-36s %s", "strided-copy launch ABI", matches ? "OK" : en(r));
        if (!matches && r == CUDA_SUCCESS) {
            std::printf(" (index %d: got %.1f, expected %.1f)",
                first_mismatch, output[first_mismatch], expected[first_mismatch]);
        }
        std::printf("\n");
        ok += matches;
        if (input_device != 0) cuMemFree(input_device);
        if (output_device != 0) cuMemFree(output_device);
    }

    cuModuleUnload(mod);
    std::printf("\n%d/%d module checks passed\n", ok, total + 1);
    return ok == total + 1 ? 0 : 1;
}
