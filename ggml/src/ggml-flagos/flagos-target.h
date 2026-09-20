#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

// A physical SoC may expose more than one execution engine.  The provider
// still enumerates each engine as a GGML device, while this descriptor keeps
// the selection logic independent of vendor product names.
enum class flagos_engine_kind : uint32_t {
    cpu = 0,
    gpu,
    npu,
    dsp,
    matrix,
    other,
};

enum flagos_target_feature : uint64_t {
    FLAGOS_FEATURE_SIMD       = 1ull << 0,
    FLAGOS_FEATURE_NEON       = 1ull << 1,
    FLAGOS_FEATURE_SVE        = 1ull << 2,
    FLAGOS_FEATURE_SVE2       = 1ull << 3,
    FLAGOS_FEATURE_SME        = 1ull << 4,
    FLAGOS_FEATURE_SME2       = 1ull << 5,
    FLAGOS_FEATURE_DOTPROD    = 1ull << 6,
    FLAGOS_FEATURE_I8MM       = 1ull << 7,
    FLAGOS_FEATURE_FP16       = 1ull << 8,
    FLAGOS_FEATURE_BF16       = 1ull << 9,
    FLAGOS_FEATURE_INT8       = 1ull << 10,
    FLAGOS_FEATURE_INT4       = 1ull << 11,
    FLAGOS_FEATURE_MATRIX     = 1ull << 12,
    FLAGOS_FEATURE_WAVE32     = 1ull << 13,
    FLAGOS_FEATURE_WAVE64     = 1ull << 14,
    FLAGOS_FEATURE_UNIFIED_MEMORY = 1ull << 15,
};

enum class flagos_support_state : uint32_t {
    unsupported = 0,
    fallback,
    available,
    validated,
    tuned,
};

GGML_BACKEND_API const char * flagos_support_state_name(flagos_support_state state);

enum class flagos_support_reason : uint32_t {
    none = 0,
    invalid_profile,
    invalid_variant,
    missing_engine,
    missing_feature,
    forbidden_feature,
    vector_width,
    lane_count,
    target,
    shape_m,
    shape_n,
    shape_k,
};

GGML_BACKEND_API const char * flagos_support_reason_name(flagos_support_reason reason);

struct flagos_device_profile {
    uint32_t struct_size = 0;
    uint32_t version = 1;
    flagos_engine_kind engine = flagos_engine_kind::other;
    uint64_t features = 0;
    uint32_t vector_bits = 0;
    uint32_t lane_count = 0;
    uint32_t matrix_m = 0;
    uint32_t matrix_n = 0;
    uint32_t matrix_k = 0;
    uint32_t concurrency = 0;
    uint64_t memory_domain_id = 0;
    const char * vendor = nullptr;
    const char * architecture = nullptr;
    const char * microarchitecture = nullptr;
    const char * target = nullptr;
    const char * runtime = nullptr;
    const char * aot_format = nullptr;
};

GGML_BACKEND_API bool flagos_device_profile_is_valid(const flagos_device_profile * profile);

struct flagos_variant_requirements {
    uint64_t required_features = 0;
    uint64_t forbidden_features = 0;
    uint32_t min_vector_bits = 0;
    uint32_t min_lane_count = 0;
    const char * target = nullptr;
    uint32_t engine_mask = 0;
};

struct flagos_kernel_variant {
    const char * name = nullptr;
    flagos_variant_requirements requirements;
    int64_t min_m = 0;
    int64_t max_m = 0;
    int64_t min_n = 0;
    int64_t max_n = 0;
    int64_t min_k = 0;
    int64_t max_k = 0;
    int64_t score = 0;
    flagos_support_state quality = flagos_support_state::available;
};

struct flagos_kernel_shape {
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
};

struct flagos_variant_match {
    flagos_support_state state = flagos_support_state::unsupported;
    flagos_support_reason reason = flagos_support_reason::invalid_variant;
    int64_t score = 0;
};

GGML_BACKEND_API flagos_variant_match flagos_check_kernel_variant(
        const flagos_device_profile * profile,
        const flagos_kernel_variant * variant,
        const flagos_kernel_shape * shape);

GGML_BACKEND_API bool flagos_kernel_variant_matches(
        const flagos_device_profile * profile,
        const flagos_kernel_variant * variant,
        const flagos_kernel_shape * shape);

GGML_BACKEND_API int64_t flagos_kernel_variant_score(
        const flagos_device_profile * profile,
        const flagos_kernel_variant * variant,
        const flagos_kernel_shape * shape);

inline constexpr uint32_t flagos_engine_bit(flagos_engine_kind engine) {
    const uint32_t value = static_cast<uint32_t>(engine);
    return value < 32 ? 1u << value : 0;
}

GGML_BACKEND_API const flagos_kernel_variant * flagos_select_kernel_variant(
        const flagos_device_profile * profile,
        const flagos_kernel_variant * variants,
        size_t variant_count,
        const flagos_kernel_shape * shape);
