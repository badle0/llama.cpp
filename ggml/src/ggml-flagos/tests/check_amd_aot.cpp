#include "../providers/amd/flagos-amd-aot.h"
#include "../providers/amd/flagos-amd-hsaco.h"

#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static void check(bool condition, const char * expression, int line) {
    if (!condition) {
        std::fprintf(stderr, "check failed at line %d: %s\n", line, expression);
        std::abort();
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

int main(int argc, char ** argv) {
    CHECK(flagos_amd::base_gcn_arch("gfx1150") == "gfx1150");
    CHECK(flagos_amd::base_gcn_arch("gfx1150:xnack-") == "gfx1150");
    CHECK(flagos_amd::base_gcn_arch("gfx90a:sramecc+:xnack-") == "gfx90a");
    bool metadata_only = false;
    std::string arch;
    std::string package_path;
    std::string hsaco_path;
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--metadata-only") == 0 && index + 1 < argc) {
            metadata_only = true;
            arch = flagos_amd::base_gcn_arch(argv[++index]);
        } else if (std::strcmp(argv[index], "--parse-hsaco") == 0 && index + 1 < argc) {
            hsaco_path = argv[++index];
        } else if (package_path.empty()) {
            package_path = argv[index];
        } else {
            std::fprintf(stderr,
                "usage: %s [PACKAGE_DIR] [--metadata-only ARCH] [--parse-hsaco FILE]\n",
                argv[0]);
            return 2;
        }
    }
    if (!hsaco_path.empty()) {
        std::ifstream source(hsaco_path, std::ios::binary);
        std::vector<uint8_t> image(
            (std::istreambuf_iterator<char>(source)), std::istreambuf_iterator<char>());
        if (source.bad() || image.empty()) {
            std::fprintf(stderr, "cannot read HSACO: %s\n", hsaco_path.c_str());
            return 2;
        }
        flagos_amd::hsaco_kernel_metadata metadata;
        std::string error;
        if (!flagos_amd::parse_hsaco_metadata(image, metadata, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        std::printf("%s %s %zu\n", metadata.target.c_str(), metadata.name.c_str(), metadata.arguments.size());
        return 0;
    }
#if defined(FLAGOS_AMD_CONFIGURED_KERNEL_DIR)
    if (package_path.empty()) {
        package_path = FLAGOS_AMD_CONFIGURED_KERNEL_DIR;
    }
#endif
    if (package_path.empty()) {
        std::puts("FlagOS AMD AOT checks skipped: no package directory configured");
        return 0;
    }

    int device = -1;
    hipDeviceProp_t props {};
    if (!metadata_only) {
        if (hipGetDevice(&device) != hipSuccess ||
            hipGetDeviceProperties(&props, device) != hipSuccess) {
            std::puts("FlagOS AMD AOT checks skipped: no HIP device visible");
            return 0;
        }
        arch = flagos_amd::base_gcn_arch(props.gcnArchName);
    }

    flagos_amd::kernel_registry registry;
    if (!registry.initialize(std::filesystem::path(package_path), device, arch, !metadata_only)) {
        std::fprintf(stderr, "FlagOS AMD AOT package validation failed: %s\n", package_path.c_str());
        return 1;
    }
    CHECK(registry.size() >= 1);
    const auto check_tuning_profile = [&registry](const auto & contracts) {
        CHECK(registry.size() == std::size(contracts));
        for (const auto & abi : contracts) {
            const auto * tuned = registry.find(abi.name);
            CHECK(tuned != nullptr);
            CHECK(tuned->argument_count == static_cast<int>(abi.argument_count));
            CHECK(tuned->block_size == abi.block_size);
            CHECK(tuned->exact_block_size == abi.exact_block_size);
            CHECK(tuned->tile_m == abi.tile_m);
            CHECK(tuned->tile_n == abi.tile_n);
            CHECK(tuned->tile_k == abi.tile_k);
            CHECK(tuned->num_warps == abi.num_warps);
            CHECK(tuned->warp_size == abi.warp_size);
        }
    };
    if (registry.tuning_profile() == flagos_amd::tuning_profile_gfx1150_q4ffn_v1) {
        check_tuning_profile(flagos_amd::tuning_profile_gfx1150_q4ffn_v1_kernels);
    } else if (registry.tuning_profile() == flagos_amd::tuning_profile_gfx1150_qwen35_q4km_v2) {
        check_tuning_profile(flagos_amd::tuning_profile_gfx1150_qwen35_q4km_v2_kernels);
    }

    const char * smoke_kernel = registry.find("double") != nullptr ? "double" :
        registry.find("flagos_add_f32") != nullptr ? "flagos_add_f32" : nullptr;
    const auto check_q4_ffn_abi = [&registry](const char * name) {
        if (const auto * kernel = registry.find(name)) {
            CHECK(kernel->argument_count == 8);
            CHECK(kernel->argument_abi.size() == 8);
            CHECK(kernel->block_size > 0 && kernel->block_size <= 32);
            CHECK((kernel->block_size & (kernel->block_size - 1)) == 0);
            void * gate_u8 = nullptr;
            void * gate_f16 = nullptr;
            void * up_u8 = nullptr;
            void * up_f16 = nullptr;
            void * activation = nullptr;
            void * output = nullptr;
            int k = 256;
            int rows = kernel->block_size;
            flagos_amd::kernel_arguments arguments {
                &gate_u8, &gate_f16, &up_u8, &up_f16,
                &activation, &output, &k, &rows,
            };
            CHECK(arguments.matches(kernel->argument_abi));
        }
    };
    check_q4_ffn_abi("flagos_ffn_swiglu_q4_0_f32_decode");
    check_q4_ffn_abi("flagos_ffn_swiglu_q4_0_f32_decode_staged");
    check_q4_ffn_abi("flagos_ffn_swiglu_q4_k_f32_decode");
    check_q4_ffn_abi("flagos_ffn_swiglu_q4_k_f32_decode_staged");
    // Compile-only packages may not contain an arithmetic smoke kernel.
    if (smoke_kernel == nullptr) {
        std::printf("FlagOS AMD AOT metadata checks passed: %s (%zu kernels)\n",
            arch.c_str(), registry.size());
        return 0;
    }
    const auto * metadata = registry.find(smoke_kernel);
    CHECK(metadata != nullptr);
    CHECK(metadata->symbol == smoke_kernel);
    CHECK(metadata->threads > 0);
    CHECK(metadata->block_size > 0);
    CHECK(metadata->argument_count >= 0);
    CHECK(metadata->argument_abi.size() == static_cast<size_t>(metadata->argument_count));
    CHECK(metadata->kernarg_segment_size > 0);
    struct expected_kernel_abi {
        const char * name;
        int argument_count;
    };
    const expected_kernel_abi qwen35_abis[] = {
        { "flagos_add_repeat_f32", 5 },
        { "flagos_copy_strided_f32", 19 },
        { "flagos_concat_f32", 21 },
        { "flagos_ssm_conv_f32", 12 },
        { "flagos_ssm_conv_silu_f32", 12 },
        { "flagos_silu_mul_f32", 4 },
        { "flagos_sigmoid_mul_f32", 4 },
        { "flagos_softplus_mul_f32", 4 },
        { "flagos_rms_norm_mul_inplace_f32_narrow", 5 },
        { "flagos_l2_norm_strided_f32", 9 },
        { "flagos_gated_delta_net_scalar_f32", 24 },
        { "flagos_gated_delta_net_scalar_f32_cache", 26 },
        { "flagos_gated_delta_net_scalar_f32_cache_only", 26 },
        { "flagos_gated_delta_net_scalar_f32_cache_only_decode", 26 },
        { "flagos_mrope_f32", 20 },
        { "flagos_sigmoid_f32", 3 },
        { "flagos_softplus_f32", 3 },
        { "flagos_get_rows_q4_0_f32", 5 },
        { "flagos_get_rows_q4_1_f32", 5 },
        { "flagos_get_rows_q5_k_f32", 5 },
        { "flagos_get_rows_q8_0_f32", 5 },
        { "flagos_dequant_q4_0_f16", 3 },
        { "flagos_dequant_q4_1_f16", 3 },
        { "flagos_dequant_q5_k_f16", 3 },
        { "flagos_dequant_q8_0_f16", 3 },
        { "flagos_mul_mat_q4_0_f32", 6 },
        { "flagos_mul_mat_q4_0_f32_narrow", 6 },
        { "flagos_mul_mat_q4_1_f32", 6 },
        { "flagos_mul_mat_q5_k_f32", 6 },
        { "flagos_mul_mat_q5_k_f32_narrow16", 6 },
        { "flagos_mul_mat_q8_0_f32", 6 },
        { "flagos_mul_mat_q4_0_f32_batched", 7 },
        { "flagos_mul_mat_q4_1_f32_batched", 7 },
        { "flagos_mul_mat_q5_k_f32_batched", 7 },
        { "flagos_mul_mat_q8_0_f32_batched", 7 },
        { "flagos_ffn_swiglu_q4_0_f32_decode", 8 },
        { "flagos_ffn_swiglu_q4_0_f32_decode_staged", 8 },
        { "flagos_ffn_swiglu_q4_k_f32_decode", 8 },
        { "flagos_ffn_swiglu_q4_k_f32_decode_staged", 8 },
    };
    for (const auto & expected : qwen35_abis) {
        if (const auto * kernel = registry.find(expected.name)) {
            CHECK(kernel->argument_count == expected.argument_count);
            CHECK(kernel->argument_abi.size() == static_cast<size_t>(expected.argument_count));
        }
    }
    if (const auto * gdn = registry.find("flagos_gated_delta_net_scalar_f32")) {
        CHECK(gdn->block_size == 128);
        CHECK(gdn->tile_n == 4);
    }
    if (const auto * narrow = registry.find(
            "flagos_rms_norm_mul_inplace_f32_narrow")) {
        CHECK(narrow->argument_count == 5);
        CHECK(narrow->block_size == 128);
        CHECK(narrow->exact_block_size);
        CHECK(narrow->num_warps == 1);
        CHECK(narrow->warp_size == 32);
    }
    if (const auto * gdn_cache = registry.find("flagos_gated_delta_net_scalar_f32_cache")) {
        CHECK(gdn_cache->block_size == 128);
        CHECK(gdn_cache->tile_n >= 1 && gdn_cache->tile_n <= 64);
        CHECK((gdn_cache->tile_n & (gdn_cache->tile_n - 1)) == 0);
        CHECK(gdn_cache->num_warps == 1 || gdn_cache->num_warps == 2 ||
            gdn_cache->num_warps == 4 || gdn_cache->num_warps == 8);
    }
    if (const auto * gdn_cache_only = registry.find("flagos_gated_delta_net_scalar_f32_cache_only")) {
        CHECK(gdn_cache_only->block_size == 128);
        CHECK(gdn_cache_only->tile_n >= 1 && gdn_cache_only->tile_n <= 64);
        CHECK((gdn_cache_only->tile_n & (gdn_cache_only->tile_n - 1)) == 0);
        CHECK(!gdn_cache_only->exact_block_size ||
            gdn_cache_only->block_size % gdn_cache_only->tile_n == 0);
        CHECK(gdn_cache_only->num_warps == 1 || gdn_cache_only->num_warps == 2 ||
            gdn_cache_only->num_warps == 4 || gdn_cache_only->num_warps == 8);
    }
    if (const auto * gdn_decode = registry.find(
            "flagos_gated_delta_net_scalar_f32_cache_only_decode")) {
        CHECK(gdn_decode->argument_count == 26);
        CHECK(gdn_decode->argument_abi.size() == 26);
        CHECK(gdn_decode->block_size == 128);
        CHECK(gdn_decode->exact_block_size);
        CHECK(gdn_decode->tile_n >= 1 && gdn_decode->tile_n <= 64);
        CHECK((gdn_decode->tile_n & (gdn_decode->tile_n - 1)) == 0);
        CHECK(gdn_decode->block_size % gdn_decode->tile_n == 0);
    }
    if (registry.find("flagos_dequant_q4_0_f16") != nullptr) {
        CHECK(registry.find("flagos_get_rows_q4_0_f32") != nullptr);
        CHECK(registry.find("flagos_mul_mat_q4_0_f32") != nullptr);
        CHECK(registry.find("flagos_mul_mat_q4_0_f32_batched") != nullptr);
    }
    if (registry.find("flagos_dequant_q4_1_f16") != nullptr) {
        CHECK(registry.find("flagos_get_rows_q4_1_f32") != nullptr);
        CHECK(registry.find("flagos_mul_mat_q4_1_f32") != nullptr);
        CHECK(registry.find("flagos_mul_mat_q4_1_f32_batched") != nullptr);
    }
    if (registry.find("flagos_dequant_q5_k_f16") != nullptr) {
        CHECK(registry.find("flagos_get_rows_q5_k_f32") != nullptr);
        CHECK(registry.find("flagos_mul_mat_q5_k_f32") != nullptr);
        CHECK(registry.find("flagos_mul_mat_q5_k_f32_batched") != nullptr);
    }
    if (registry.find("flagos_mul_mat_q5_k_f32_narrow16") != nullptr) {
        CHECK(registry.find("flagos_mul_mat_q5_k_f32") != nullptr);
    }
    if (registry.find("flagos_dequant_q8_0_f16") != nullptr) {
        CHECK(registry.find("flagos_get_rows_q8_0_f32") != nullptr);
        CHECK(registry.find("flagos_mul_mat_q8_0_f32") != nullptr);
        CHECK(registry.find("flagos_mul_mat_q8_0_f32_batched") != nullptr);
    }
    if (const auto * f16 = registry.find("flagos_mul_mat_f16_f32_batched")) {
        CHECK(f16->tile_m > 0);
        CHECK(f16->tile_n > 0);
        CHECK(f16->tile_k > 0);
    }
    if (const auto * grouped = registry.find("flagos_mul_mat_f16_f32_grouped")) {
        CHECK(grouped->tile_m > 0);
        CHECK(grouped->tile_n > 0);
        CHECK(grouped->tile_k > 0);
    }
    if (const auto * grouped_ffn = registry.find("flagos_ffn_swiglu_f16_f32_grouped")) {
        CHECK(grouped_ffn->tile_m > 0);
        CHECK(grouped_ffn->tile_n > 0);
        CHECK(grouped_ffn->tile_k > 0);
    }
    if (const auto * f16_scratch_ffn = registry.find("flagos_ffn_swiglu_f16_f16_grouped")) {
        CHECK(f16_scratch_ffn->tile_m > 0);
        CHECK(f16_scratch_ffn->tile_n > 0);
        CHECK(f16_scratch_ffn->tile_k > 0);
        const auto * f16_scratch_down = registry.find("flagos_mul_mat_f16_f16_grouped");
        CHECK(f16_scratch_down != nullptr);
        CHECK(f16_scratch_down->tile_m > 0);
        CHECK(f16_scratch_down->tile_n > 0);
        CHECK(f16_scratch_down->tile_k > 0);
    }
    if (registry.find("flagos_mul_mat_f16_f16_grouped") != nullptr) {
        CHECK(registry.find("flagos_ffn_swiglu_f16_f16_grouped") != nullptr);
    }
    if (const auto * residual_narrow = registry.find(
            "flagos_add_rms_norm_mul_residual_f32_narrow")) {
        const auto * residual = registry.find("flagos_add_rms_norm_mul_residual_f32");
        CHECK(residual != nullptr);
        CHECK(residual_narrow->block_size > 0);
        CHECK(residual_narrow->block_size < residual->block_size);
        CHECK(residual_narrow->threads == residual->threads);
    }
    void * x = nullptr;
    void * y = nullptr;
    int n = 256;
    flagos_amd::kernel_arguments args;
    if (std::strcmp(smoke_kernel, "double") == 0) {
        args.assign(&x, &y, &n);
    } else {
        args.assign(&x, &x, &y, &n);
    }
    CHECK(args.matches(metadata->argument_abi));
    int wrong_pointer_argument = 0;
    flagos_amd::kernel_arguments wrong_args;
    if (std::strcmp(smoke_kernel, "double") == 0) {
        wrong_args.assign(&wrong_pointer_argument, &y, &n);
    } else {
        wrong_args.assign(&wrong_pointer_argument, &x, &y, &n);
    }
    CHECK(!wrong_args.matches(metadata->argument_abi));
    if (metadata_only) {
        std::printf("FlagOS AMD AOT metadata checks passed: %s (%zu kernels)\n",
            arch.c_str(), registry.size());
        return 0;
    }

    hipStream_t stream = nullptr;
    CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) == hipSuccess);
    CHECK(hipMalloc(&x, 256 * sizeof(float)) == hipSuccess);
    CHECK(hipMalloc(&y, 256 * sizeof(float)) == hipSuccess);
    std::vector<float> host_x(256);
    std::vector<float> host_y(256, -1.0f);
    for (size_t i = 0; i < host_x.size(); ++i) {
        host_x[i] = static_cast<float>(i) * 0.5f;
    }
    CHECK(hipMemcpyAsync(x, host_x.data(), host_x.size() * sizeof(float),
        hipMemcpyHostToDevice, stream) == hipSuccess);
    n = static_cast<int>(host_x.size());
    CHECK(registry.launch(smoke_kernel, stream, 2, 1, 1, args));
    CHECK(hipMemcpyAsync(host_y.data(), y, host_y.size() * sizeof(float),
        hipMemcpyDeviceToHost, stream) == hipSuccess);
    CHECK(hipStreamSynchronize(stream) == hipSuccess);
    for (size_t i = 0; i < host_y.size(); ++i) {
        const float expected = std::strcmp(smoke_kernel, "double") == 0
            ? host_x[i] * 2.0f : host_x[i] + host_x[i];
        CHECK(host_y[i] == expected);
    }
    CHECK(hipFree(x) == hipSuccess);
    CHECK(hipFree(y) == hipSuccess);
    CHECK(hipStreamDestroy(stream) == hipSuccess);
    std::printf("FlagOS AMD AOT checks passed: %s (%zu kernels)\n", arch.c_str(), registry.size());
    return 0;
}
