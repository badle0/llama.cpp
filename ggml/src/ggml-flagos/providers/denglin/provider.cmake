set(FLAGOS_DENGLIN_SDK_ROOT "$ENV{FLAGOS_DENGLIN_SDK_ROOT}" CACHE PATH "Denglin SDK root")
if (NOT FLAGOS_DENGLIN_SDK_ROOT AND FLAGOS_SDK_ROOT)
    set(FLAGOS_DENGLIN_SDK_ROOT "${FLAGOS_SDK_ROOT}" CACHE PATH "Denglin SDK root" FORCE)
endif()
if (NOT FLAGOS_DENGLIN_SDK_ROOT AND NOT "$ENV{FLAGOS_SDK_ROOT}" STREQUAL "")
    set(FLAGOS_DENGLIN_SDK_ROOT "$ENV{FLAGOS_SDK_ROOT}" CACHE PATH "Denglin SDK root" FORCE)
endif()
if (NOT FLAGOS_DENGLIN_SDK_ROOT)
    set(FLAGOS_DENGLIN_SDK_ROOT "/usr/local/dlgpu/sdk")
endif()

if (NOT FLAGOS_DENGLIN_CUDA_INCLUDE_DIR AND FLAGOS_CUDA_INCLUDE_DIR)
    set(FLAGOS_DENGLIN_CUDA_INCLUDE_DIR "${FLAGOS_CUDA_INCLUDE_DIR}" CACHE PATH "Denglin CUDA-compatible include directory" FORCE)
endif()
find_path(FLAGOS_DENGLIN_CUDA_INCLUDE_DIR
    NAMES cuda_runtime_api.h cuda.h
    HINTS "$ENV{FLAGOS_DENGLIN_CUDA_INCLUDE_DIR}" "${FLAGOS_DENGLIN_SDK_ROOT}/include"
    NO_DEFAULT_PATH)

find_library(FLAGOS_DENGLIN_CURT_LIBRARY
    NAMES curt
    HINTS "${FLAGOS_DENGLIN_SDK_ROOT}/lib"
    NO_DEFAULT_PATH)

find_library(FLAGOS_DENGLIN_BLAS_LIBRARY
    NAMES dlblas
    HINTS "${FLAGOS_DENGLIN_SDK_ROOT}/lib"
    NO_DEFAULT_PATH)

if (NOT FLAGOS_DENGLIN_CUDA_INCLUDE_DIR OR NOT FLAGOS_DENGLIN_CURT_LIBRARY OR NOT FLAGOS_DENGLIN_BLAS_LIBRARY)
    message(FATAL_ERROR "The FlagOS Denglin provider requires its SDK. Set FLAGOS_DENGLIN_SDK_ROOT or disable GGML_FLAGOS_DENGLIN.")
endif()

get_filename_component(FLAGOS_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(FLAGOS_DENGLIN_AOT_KERNEL_DIR "${FLAGOS_ROOT_DIR}/kernels/aot")
foreach(kernel
        flagos_kernels.cubin
        flagos_add_f32.cubin
        flagos_mul_f32.cubin
        flagos_scale_f32.cubin
        flagos_copy_f32.cubin
        flagos_copy_strided_f32.cubin
        flagos_swiglu_split_f32.cubin
        flagos_set_rows_f32_f16.cubin
        flagos_flash_attn_decode_f32_f16.cubin
        flagos_rope_neox_f32.cubin
        flagos_mul_mat_q4_k_f32.cubin
        flagos_mul_mat_q4_k_f32_batched.cubin
        flagos_mul_mat_q6_k_f32.cubin
        flagos_mul_mat_q6_k_f32_batched.cubin
        flagos_rms_norm_f32.cubin
        flagos_rms_norm_mul_f32.cubin
        flagos_cast_f32_f16.cubin
        flagos_cast_f16_f32.cubin
        flagos_dequant_q4_k_f16.cubin
        flagos_dequant_q6_k_f16.cubin
        flagos_get_rows_q4_k_f32.cubin
        flagos_ssm_conv_f32.cubin
        flagos_sub_f32.cubin
        flagos_div_f32.cubin
        flagos_sigmoid_f32.cubin
        flagos_exp_f32.cubin
        flagos_softplus_f32.cubin
        flagos_fill_f32.cubin
        flagos_sum_rows_f32.cubin
        flagos_l2_norm_f32.cubin
        flagos_norm_f32.cubin
        flagos_cumsum_f32.cubin
        flagos_soft_max_f32.cubin
        manifest.json)
    if (NOT EXISTS "${FLAGOS_DENGLIN_AOT_KERNEL_DIR}/${kernel}")
        message(FATAL_ERROR "Missing FlagOS Denglin AOT asset: ${FLAGOS_DENGLIN_AOT_KERNEL_DIR}/${kernel}")
    endif()
endforeach()

list(APPEND FLAGOS_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/flagos-denglin-api.h"
    "${CMAKE_CURRENT_LIST_DIR}/flagos-denglin.cpp")
list(APPEND FLAGOS_PRIVATE_INCLUDE_DIRS
    "${FLAGOS_DENGLIN_CUDA_INCLUDE_DIR}"
    "${FLAGOS_DENGLIN_SDK_ROOT}/include")
list(APPEND FLAGOS_PRIVATE_LIBRARIES
    "${FLAGOS_DENGLIN_CURT_LIBRARY}"
    "${FLAGOS_DENGLIN_BLAS_LIBRARY}")
list(APPEND FLAGOS_PRIVATE_DEFINITIONS
    GGML_FLAGOS_HAVE_DENGLIN
    FLAGOS_KERNEL_DIR="${FLAGOS_DENGLIN_AOT_KERNEL_DIR}")
