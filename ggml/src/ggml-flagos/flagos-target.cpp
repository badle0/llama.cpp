#include "flagos-target.h"

#include <cstring>
#include <limits>

const char * flagos_support_state_name(flagos_support_state state) {
    switch (state) {
        case flagos_support_state::unsupported: return "unsupported";
        case flagos_support_state::fallback:    return "fallback";
        case flagos_support_state::available:   return "available";
        case flagos_support_state::validated:   return "validated";
        case flagos_support_state::tuned:       return "tuned";
    }
    return "unknown";
}

const char * flagos_support_reason_name(flagos_support_reason reason) {
    switch (reason) {
        case flagos_support_reason::none:             return "none";
        case flagos_support_reason::invalid_profile:  return "invalid_profile";
        case flagos_support_reason::invalid_variant:  return "invalid_variant";
        case flagos_support_reason::missing_engine:   return "missing_engine";
        case flagos_support_reason::missing_feature:  return "missing_feature";
        case flagos_support_reason::forbidden_feature: return "forbidden_feature";
        case flagos_support_reason::vector_width:     return "vector_width";
        case flagos_support_reason::lane_count:       return "lane_count";
        case flagos_support_reason::target:           return "target";
        case flagos_support_reason::shape_m:          return "shape_m";
        case flagos_support_reason::shape_n:          return "shape_n";
        case flagos_support_reason::shape_k:          return "shape_k";
    }
    return "unknown";
}

bool flagos_device_profile_is_valid(const flagos_device_profile * profile) {
    if (profile == nullptr || profile->struct_size < sizeof(flagos_device_profile) ||
        profile->version != 1) {
        return false;
    }
    switch (profile->engine) {
        case flagos_engine_kind::cpu:
        case flagos_engine_kind::gpu:
        case flagos_engine_kind::npu:
        case flagos_engine_kind::dsp:
        case flagos_engine_kind::matrix:
        case flagos_engine_kind::other:
            return true;
    }
    return false;
}

static bool flagos_dimension_matches(int64_t value, int64_t minimum, int64_t maximum) {
    return value >= 0 && (minimum <= 0 || value >= minimum) &&
        (maximum <= 0 || value <= maximum);
}

static bool flagos_support_state_is_valid(flagos_support_state state) {
    switch (state) {
        case flagos_support_state::unsupported:
        case flagos_support_state::fallback:
        case flagos_support_state::available:
        case flagos_support_state::validated:
        case flagos_support_state::tuned:
            return true;
    }
    return false;
}

flagos_variant_match flagos_check_kernel_variant(
        const flagos_device_profile * profile,
        const flagos_kernel_variant * variant,
        const flagos_kernel_shape * shape) {
    flagos_variant_match result;
    if (!flagos_device_profile_is_valid(profile)) {
        result.reason = flagos_support_reason::invalid_profile;
        return result;
    }
    if (variant == nullptr || variant->name == nullptr || variant->name[0] == '\0' ||
        shape == nullptr) {
        result.reason = flagos_support_reason::invalid_variant;
        return result;
    }
    if (!flagos_support_state_is_valid(variant->quality)) {
        result.reason = flagos_support_reason::invalid_variant;
        return result;
    }
    if (variant->requirements.engine_mask != 0 &&
        (variant->requirements.engine_mask & flagos_engine_bit(profile->engine)) == 0) {
        result.reason = flagos_support_reason::missing_engine;
        return result;
    }
    if ((profile->features & variant->requirements.required_features) !=
            variant->requirements.required_features) {
        result.reason = flagos_support_reason::missing_feature;
        return result;
    }
    if ((profile->features & variant->requirements.forbidden_features) != 0) {
        result.reason = flagos_support_reason::forbidden_feature;
        return result;
    }
    if (variant->requirements.min_vector_bits != 0 &&
        profile->vector_bits < variant->requirements.min_vector_bits) {
        result.reason = flagos_support_reason::vector_width;
        return result;
    }
    if (variant->requirements.min_lane_count != 0 &&
        profile->lane_count < variant->requirements.min_lane_count) {
        result.reason = flagos_support_reason::lane_count;
        return result;
    }
    if (variant->requirements.target != nullptr &&
        (profile->target == nullptr ||
         std::strcmp(profile->target, variant->requirements.target) != 0)) {
        result.reason = flagos_support_reason::target;
        return result;
    }
    if (!flagos_dimension_matches(shape->m, variant->min_m, variant->max_m)) {
        result.reason = flagos_support_reason::shape_m;
        return result;
    }
    if (!flagos_dimension_matches(shape->n, variant->min_n, variant->max_n)) {
        result.reason = flagos_support_reason::shape_n;
        return result;
    }
    if (!flagos_dimension_matches(shape->k, variant->min_k, variant->max_k)) {
        result.reason = flagos_support_reason::shape_k;
        return result;
    }
    result.state = variant->quality;
    result.reason = flagos_support_reason::none;
    result.score = variant->score;
    return result;
}

bool flagos_kernel_variant_matches(
        const flagos_device_profile * profile,
        const flagos_kernel_variant * variant,
        const flagos_kernel_shape * shape) {
    return flagos_check_kernel_variant(profile, variant, shape).state !=
        flagos_support_state::unsupported;
}

int64_t flagos_kernel_variant_score(
        const flagos_device_profile * profile,
        const flagos_kernel_variant * variant,
        const flagos_kernel_shape * shape) {
    const flagos_variant_match match = flagos_check_kernel_variant(profile, variant, shape);
    if (match.state == flagos_support_state::unsupported) {
        return std::numeric_limits<int64_t>::min();
    }
    return match.score;
}

const flagos_kernel_variant * flagos_select_kernel_variant(
        const flagos_device_profile * profile,
        const flagos_kernel_variant * variants,
        size_t variant_count,
        const flagos_kernel_shape * shape) {
    const flagos_kernel_variant * selected = nullptr;
    int64_t selected_score = std::numeric_limits<int64_t>::min();
    flagos_support_state selected_state = flagos_support_state::unsupported;
    if (variants == nullptr || shape == nullptr) {
        return nullptr;
    }
    for (size_t index = 0; index < variant_count; ++index) {
        const flagos_variant_match match = flagos_check_kernel_variant(
            profile, &variants[index], shape);
        if (match.state == flagos_support_state::unsupported) {
            continue;
        }
        if (match.score > selected_score ||
            (match.score == selected_score &&
             static_cast<uint32_t>(match.state) > static_cast<uint32_t>(selected_state))) {
            selected = &variants[index];
            selected_score = match.score;
            selected_state = match.state;
        }
    }
    return selected_score == std::numeric_limits<int64_t>::min() ? nullptr : selected;
}
