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
        providers/amd/flagos-amd-aot.cpp
        tests/check_amd.cpp)
    target_include_directories(flagos-check-amd PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
        "${CMAKE_SOURCE_DIR}/vendor")
    target_compile_definitions(flagos-check-amd PRIVATE GGML_FLAGOS_HAVE_AMD)
    target_link_libraries(flagos-check-amd PRIVATE ggml hip::host)

    add_executable(flagos-check-amd-aot EXCLUDE_FROM_ALL
        providers/amd/flagos-amd-aot.cpp
        tests/check_amd_aot.cpp)
    target_include_directories(flagos-check-amd-aot PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
        "${CMAKE_SOURCE_DIR}/vendor")
    target_link_libraries(flagos-check-amd-aot PRIVATE ggml-base hip::host)
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
