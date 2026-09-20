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

if (NOT GGML_FLAGOS_DENGLIN)
    message(STATUS "FlagOS: building provider-neutral core with no device provider")
endif()

ggml_add_backend_library(ggml-flagos ${FLAGOS_SOURCES})
target_include_directories(ggml-flagos PRIVATE ${FLAGOS_PRIVATE_INCLUDE_DIRS})
target_link_libraries(ggml-flagos PRIVATE ${FLAGOS_PRIVATE_LIBRARIES})
target_compile_definitions(ggml-flagos PRIVATE ${FLAGOS_PRIVATE_DEFINITIONS})

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

install(FILES
    ggml-flagos.h
    flagos-provider.h
    flagos-registry.h
    flagos-target.h
    DESTINATION include)
if (GGML_FLAGOS_DENGLIN)
    install(DIRECTORY "${FLAGOS_DENGLIN_AOT_KERNEL_DIR}/" DESTINATION share/ggml/flagos/denglin)
endif()
