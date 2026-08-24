#include "flagos-target.h"

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

static flagos_device_profile make_profile(
        flagos_engine_kind engine, uint64_t features, uint32_t vector_bits,
        uint32_t lane_count, const char * target) {
    flagos_device_profile profile {};
    profile.struct_size = sizeof(profile);
    profile.version = 1;
    profile.engine = engine;
    profile.features = features;
    profile.vector_bits = vector_bits;
    profile.lane_count = lane_count;
    profile.vendor = "mock";
    profile.architecture = engine == flagos_engine_kind::cpu ? "arm64" : "accelerator";
    profile.target = target;
    return profile;
}

static const flagos_kernel_variant * select(
        const flagos_device_profile & profile,
        const flagos_kernel_variant * variants,
        size_t count) {
    const flagos_kernel_shape shape { 4096, 32, 4096 };
    return flagos_select_kernel_variant(&profile, variants, count, &shape);
}

int main() {
    CHECK(std::strcmp(flagos_support_state_name(flagos_support_state::tuned), "tuned") == 0);
    CHECK(std::strcmp(flagos_support_reason_name(flagos_support_reason::shape_k), "shape_k") == 0);

    const uint32_t cpu = flagos_engine_bit(flagos_engine_kind::cpu);
    const uint32_t gpu = flagos_engine_bit(flagos_engine_kind::gpu);
    const uint32_t npu = flagos_engine_bit(flagos_engine_kind::npu);
    const flagos_kernel_variant variants[] = {
        {
            "cpu-neon-dotprod",
            { FLAGOS_FEATURE_NEON | FLAGOS_FEATURE_DOTPROD, 0, 128, 0, nullptr, cpu },
            1, 0, 1, 0, 1, 0, 10,
        },
        {
            "cpu-i8mm",
            { FLAGOS_FEATURE_NEON | FLAGOS_FEATURE_I8MM, 0, 128, 0, nullptr, cpu },
            1, 0, 1, 0, 1, 0, 20,
        },
        {
            "cpu-sme2-matrix",
            { FLAGOS_FEATURE_SME2 | FLAGOS_FEATURE_MATRIX, 0, 0, 0, nullptr, cpu },
            1, 0, 1, 0, 1, 0, 40,
        },
        {
            "gpu-wave32",
            { FLAGOS_FEATURE_WAVE32, 0, 0, 32, nullptr, gpu },
            1, 0, 1, 0, 1, 0, 30,
        },
        {
            "npu-int4",
            { FLAGOS_FEATURE_INT4 | FLAGOS_FEATURE_MATRIX, 0, 0, 0, nullptr, npu },
            1, 0, 1, 0, 1, 0, 50,
        },
    };

    const flagos_kernel_variant quality_tie[] = {
        {
            "available-tie",
            { FLAGOS_FEATURE_NEON, 0, 0, 0, nullptr, cpu },
            1, 0, 1, 0, 1, 0, 0, flagos_support_state::available,
        },
        {
            "validated-tie",
            { FLAGOS_FEATURE_NEON, 0, 0, 0, nullptr, cpu },
            1, 0, 1, 0, 1, 0, 0, flagos_support_state::validated,
        },
    };

    const auto apple_m4 = make_profile(flagos_engine_kind::cpu,
        FLAGOS_FEATURE_NEON | FLAGOS_FEATURE_DOTPROD | FLAGOS_FEATURE_I8MM |
        FLAGOS_FEATURE_SME2 | FLAGOS_FEATURE_MATRIX, 128, 4, "apple-m4-mock");
    const auto apple_m5_pro = make_profile(flagos_engine_kind::cpu,
        FLAGOS_FEATURE_NEON | FLAGOS_FEATURE_DOTPROD | FLAGOS_FEATURE_I8MM |
        FLAGOS_FEATURE_SVE2 | FLAGOS_FEATURE_SME2 | FLAGOS_FEATURE_MATRIX,
        256, 8, "apple-m5-pro-mock");
    const auto cix_p1 = make_profile(flagos_engine_kind::cpu,
        FLAGOS_FEATURE_NEON | FLAGOS_FEATURE_DOTPROD | FLAGOS_FEATURE_I8MM,
        128, 4, "cix-p1-mock");
    const auto x2_elite = make_profile(flagos_engine_kind::cpu,
        FLAGOS_FEATURE_NEON | FLAGOS_FEATURE_DOTPROD,
        128, 4, "qualcomm-x2-elite-mock");
    const auto gfx1150 = make_profile(flagos_engine_kind::gpu,
        FLAGOS_FEATURE_SIMD | FLAGOS_FEATURE_WAVE32 |
        FLAGOS_FEATURE_UNIFIED_MEMORY, 0, 32, "gfx1150");
    const auto ks20 = make_profile(flagos_engine_kind::gpu,
        FLAGOS_FEATURE_SIMD, 0, 0, "ks20");
    const auto mobile_npu = make_profile(flagos_engine_kind::npu,
        FLAGOS_FEATURE_INT4 | FLAGOS_FEATURE_INT8 | FLAGOS_FEATURE_MATRIX,
        0, 0, "mobile-npu-mock");

    CHECK(std::strcmp(select(apple_m4, variants, 5)->name, "cpu-sme2-matrix") == 0);
    CHECK(std::strcmp(select(apple_m5_pro, variants, 5)->name, "cpu-sme2-matrix") == 0);
    CHECK(std::strcmp(select(cix_p1, variants, 5)->name, "cpu-i8mm") == 0);
    CHECK(std::strcmp(select(x2_elite, variants, 5)->name, "cpu-neon-dotprod") == 0);
    CHECK(std::strcmp(select(gfx1150, variants, 5)->name, "gpu-wave32") == 0);
    CHECK(select(ks20, variants, 5) == nullptr);
    CHECK(std::strcmp(select(mobile_npu, variants, 5)->name, "npu-int4") == 0);
    const auto tie_profile = make_profile(flagos_engine_kind::cpu, FLAGOS_FEATURE_NEON, 128, 4, "tie");
    CHECK(std::strcmp(select(tie_profile, quality_tie, 2)->name, "validated-tie") == 0);
    flagos_kernel_variant invalid_quality = quality_tie[0];
    invalid_quality.quality = static_cast<flagos_support_state>(99);
    const flagos_kernel_shape shape { 4096, 32, 4096 };
    CHECK(flagos_check_kernel_variant(&tie_profile, &invalid_quality, &shape).reason ==
        flagos_support_reason::invalid_variant);

    std::puts("FlagOS target capability checks passed");
    return 0;
}
