cmake_minimum_required(VERSION 3.18)

ggml_add_backend_library(ggml-flagos
    flagos-registry.cpp
    ggml-flagos.h)

if (NOT GGML_BACKEND_DL)
    target_include_directories(ggml PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

install(FILES ggml-flagos.h DESTINATION include)
