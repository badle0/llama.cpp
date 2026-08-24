if (NOT DEFINED ROCM_PATH)
    if (DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "$ENV{ROCM_PATH}")
    elseif (EXISTS "/opt/rocm")
        set(ROCM_PATH "/opt/rocm")
    else()
        set(ROCM_PATH "/usr")
    endif()
endif()

list(APPEND CMAKE_PREFIX_PATH
    "${ROCM_PATH}"
    "${ROCM_PATH}/lib64/cmake"
    "${ROCM_PATH}/lib/x86_64-linux-gnu/cmake")

find_package(hip CONFIG REQUIRED)

list(APPEND FLAGOS_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/flagos-amd-api.h"
    "${CMAKE_CURRENT_LIST_DIR}/flagos-amd.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/flagos-amd-aot.h"
    "${CMAKE_CURRENT_LIST_DIR}/flagos-amd-aot.cpp")
list(APPEND FLAGOS_PRIVATE_LIBRARIES hip::host)
list(APPEND FLAGOS_PRIVATE_DEFINITIONS GGML_FLAGOS_HAVE_AMD)

message(STATUS "FlagOS: enabling AMD HIP provider with ROCM_PATH=${ROCM_PATH}")
