# The loader's negative matrix (plan.md "S1 step 1 spec" §7): six fake plugins, each a complete table with one fault,
# built as small shared modules that flagos-check-spacemit dlopens through the loader. Their C11 build with
# -Wall -Wextra -Werror is also the C check of core/flagos-spacemit-plugin.h.
# Included from ggml/src/ggml-flagos/CMakeLists.txt inside the flagos-check-spacemit block.

set(FLAGOS_SPACEMIT_TEST_PLUGIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/spacemit-test-plugins")

if (APPLE)
    set(FLAGOS_SPACEMIT_TEST_PLUGIN_SUFFIX ".dylib")
else()
    set(FLAGOS_SPACEMIT_TEST_PLUGIN_SUFFIX ".so")
endif()

foreach (name bad-version short-struct null-member missing-kernel no-device no-entry)
    add_library(flagos-spacemit-test-plugin-${name} MODULE EXCLUDE_FROM_ALL
                "${CMAKE_CURRENT_LIST_DIR}/tests/plugins/${name}.c")
    set_target_properties(flagos-spacemit-test-plugin-${name} PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${name}"
        SUFFIX "${FLAGOS_SPACEMIT_TEST_PLUGIN_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY "${FLAGOS_SPACEMIT_TEST_PLUGIN_DIR}")
    target_include_directories(flagos-spacemit-test-plugin-${name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/core")
    target_compile_options(flagos-spacemit-test-plugin-${name} PRIVATE -std=c11 -Wall -Wextra -Werror)
    add_dependencies(flagos-check-spacemit flagos-spacemit-test-plugin-${name})
endforeach()

target_compile_definitions(flagos-check-spacemit PRIVATE
    FLAGOS_SPACEMIT_TEST_PLUGIN_DIR="${FLAGOS_SPACEMIT_TEST_PLUGIN_DIR}"
    FLAGOS_SPACEMIT_TEST_PLUGIN_SUFFIX="${FLAGOS_SPACEMIT_TEST_PLUGIN_SUFFIX}")
