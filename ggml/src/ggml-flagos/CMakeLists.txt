cmake_minimum_required(VERSION 3.18)

set(FLAGOS_SOURCES
    flagos-graph-plan.cpp
    flagos-graph-plan.h
    flagos-provider.cpp
    flagos-provider.h
    flagos-registry.h
    flagos-registry.cpp
    flagos-target.cpp
    flagos-target.h
    ggml-flagos.h)

set(FLAGOS_PRIVATE_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/vendor")
set(FLAGOS_PRIVATE_LIBRARIES)
set(FLAGOS_PRIVATE_DEFINITIONS)

set(FLAGOS_TRITON_JIT_SOURCE_DIR "" CACHE PATH "Path to a libtriton_jit source tree")
if (GGML_FLAGOS_TRITON_JIT)
    if (NOT GGML_FLAGOS_AMD)
        message(FATAL_ERROR "The initial FlagOS Triton JIT integration requires GGML_FLAGOS_AMD=ON")
    endif()
    if (NOT EXISTS "${FLAGOS_TRITON_JIT_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "Set FLAGOS_TRITON_JIT_SOURCE_DIR to a libtriton_jit source tree")
    endif()
    set(TRITON_JIT_BACKEND AMD CACHE STRING "libtriton_jit backend" FORCE)
    set(TRITON_JIT_WITH_TORCH OFF CACHE BOOL "Build libtriton_jit without its Torch C++ adapter" FORCE)
    set(TRITON_JIT_BUILD_TESTS OFF CACHE BOOL "Do not build libtriton_jit tests as part of llama.cpp" FORCE)
    set(TRITON_JIT_BUILD_OPERATORS OFF CACHE BOOL "Do not build libtriton_jit operators as part of llama.cpp" FORCE)
    set(TRITON_JIT_INSTALL OFF CACHE BOOL "Do not install libtriton_jit as a separate package" FORCE)
    add_subdirectory("${FLAGOS_TRITON_JIT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/libtriton_jit" EXCLUDE_FROM_ALL)
endif()

if (GGML_FLAGOS_DENGLIN)
    include(providers/denglin/provider.cmake)
endif()

if (GGML_FLAGOS_AMD)
    include(providers/amd/provider.cmake)
endif()

if (NOT GGML_FLAGOS_DENGLIN AND NOT GGML_FLAGOS_AMD)
    message(STATUS "FlagOS: building provider-neutral core with no device provider")
endif()

ggml_add_backend_library(ggml-flagos ${FLAGOS_SOURCES})
target_include_directories(ggml-flagos PRIVATE ${FLAGOS_PRIVATE_INCLUDE_DIRS})
target_link_libraries(ggml-flagos PRIVATE ${FLAGOS_PRIVATE_LIBRARIES})
target_compile_definitions(ggml-flagos PRIVATE ${FLAGOS_PRIVATE_DEFINITIONS})

# In static mode ggml-backend-reg.cpp includes the backend's public registry
# header while compiling the core ggml target.  Keep that include path private
# for the dynamic plugin, but propagate it to the core static target.
if (NOT GGML_BACKEND_DL)
    target_include_directories(ggml PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

if (GGML_FLAGOS_DENGLIN)
    add_executable(flagos-check-kernels EXCLUDE_FROM_ALL kernels/check_merged_module.cpp)
    target_include_directories(flagos-check-kernels PRIVATE "${FLAGOS_DENGLIN_CUDA_INCLUDE_DIR}")
    target_link_libraries(flagos-check-kernels PRIVATE "${FLAGOS_DENGLIN_CURT_LIBRARY}")
endif()

add_executable(flagos-check-graph-plan EXCLUDE_FROM_ALL
    flagos-graph-plan.cpp
    tests/check_graph_plan.cpp)
target_include_directories(flagos-check-graph-plan PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(flagos-check-graph-plan PRIVATE ggml-base)

add_executable(flagos-check-provider EXCLUDE_FROM_ALL
    flagos-provider.cpp
    flagos-target.cpp
    tests/check_provider.cpp)
target_include_directories(flagos-check-provider PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(flagos-check-provider PRIVATE ggml-base)

add_executable(flagos-check-target EXCLUDE_FROM_ALL
    flagos-target.cpp
    tests/check_target.cpp)
target_include_directories(flagos-check-target PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/ggml/include")

add_executable(flagos-check-registry EXCLUDE_FROM_ALL
    flagos-provider.cpp
    flagos-target.cpp
    flagos-registry.cpp
    tests/check_registry.cpp)
target_include_directories(flagos-check-registry PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_definitions(flagos-check-registry PRIVATE GGML_FLAGOS_HAVE_DENGLIN GGML_FLAGOS_HAVE_AMD)
target_link_libraries(flagos-check-registry PRIVATE ggml)

if (GGML_FLAGOS_AMD)
    add_executable(flagos-check-amd EXCLUDE_FROM_ALL
        flagos-provider.cpp
        flagos-target.cpp
        flagos-registry.cpp
        flagos-graph-plan.cpp
        providers/amd/flagos-amd.cpp
        providers/amd/flagos-amd-hsaco.cpp
        providers/amd/flagos-amd-aot.cpp
        tests/check_amd_qwen35_ffn.cpp
        tests/check_amd.cpp)
    target_include_directories(flagos-check-amd PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
        "${CMAKE_SOURCE_DIR}/vendor")
    target_compile_definitions(flagos-check-amd PRIVATE
        GGML_FLAGOS_HAVE_AMD
        ${FLAGOS_AMD_PRIVATE_DEFINITIONS})
    target_link_libraries(flagos-check-amd PRIVATE ggml hip::host ${CMAKE_DL_LIBS})

    add_executable(flagos-check-amd-aot EXCLUDE_FROM_ALL
        providers/amd/flagos-amd-hsaco.cpp
        providers/amd/flagos-amd-aot.cpp
        tests/check_amd_aot.cpp)
    target_include_directories(flagos-check-amd-aot PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
        "${CMAKE_SOURCE_DIR}/vendor")
    target_compile_definitions(flagos-check-amd-aot PRIVATE
        ${FLAGOS_AMD_PRIVATE_DEFINITIONS})
    target_link_libraries(flagos-check-amd-aot PRIVATE ggml-base hip::host)

    add_executable(flagos-check-amd-qwen35-ffn EXCLUDE_FROM_ALL
        flagos-provider.cpp
        flagos-target.cpp
        flagos-registry.cpp
        flagos-graph-plan.cpp
        providers/amd/flagos-amd.cpp
        providers/amd/flagos-amd-hsaco.cpp
        providers/amd/flagos-amd-aot.cpp
        tests/check_amd_qwen35_ffn.cpp)
    target_include_directories(flagos-check-amd-qwen35-ffn PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
        "${CMAKE_SOURCE_DIR}/vendor")
    target_compile_definitions(flagos-check-amd-qwen35-ffn PRIVATE
        GGML_FLAGOS_HAVE_AMD
        FLAGOS_AMD_QWEN35_FFN_STANDALONE
        ${FLAGOS_AMD_PRIVATE_DEFINITIONS})
    target_link_libraries(flagos-check-amd-qwen35-ffn PRIVATE ggml hip::host ${CMAKE_DL_LIBS})

    if (GGML_FLAGOS_TRITON_JIT)
        add_executable(flagos-check-amd-jit EXCLUDE_FROM_ALL tests/check_amd_jit.cpp)
        target_compile_definitions(flagos-check-amd-jit PRIVATE
            FLAGOS_TRITON_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels/triton")
        target_link_libraries(flagos-check-amd-jit PRIVATE TritonJIT::triton_jit hip::host)
    endif()
endif()

install(FILES
    ggml-flagos.h
    flagos-provider.h
    flagos-registry.h
    flagos-target.h
    DESTINATION include)
if (GGML_FLAGOS_DENGLIN)
    install(DIRECTORY "${FLAGOS_DENGLIN_AOT_KERNEL_DIR}/" DESTINATION share/ggml/flagos/denglin)
endif()
if (GGML_FLAGOS_AMD AND FLAGOS_AMD_AOT_KERNEL_DIR)
    install(DIRECTORY "${FLAGOS_AMD_AOT_KERNEL_DIR}/" DESTINATION "${CMAKE_INSTALL_DATADIR}/ggml/flagos/amd")
endif()
if (GGML_FLAGOS_TRITON_JIT)
    install(DIRECTORY kernels/triton/ DESTINATION "${CMAKE_INSTALL_DATADIR}/ggml/flagos/triton")
endif()
