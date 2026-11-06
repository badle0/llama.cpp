# SpacemiT K3 provider, the core's side: the ggml vtables, the contracts, the plan cache, the diagnostics and the
# loader, built without libspert and without the IME2 toolchain. The plugin, which holds those, is its own library
# (plugin.cmake, included right after the ggml-flagos target).

set(FLAGOS_SPACEMIT_DIR "${CMAKE_CURRENT_LIST_DIR}")

# the core reaches the plugin with dlopen only, there is no linked-in form (plan.md "S1 design" A5)
if (NOT BUILD_SHARED_LIBS)
    message(FATAL_ERROR "FlagOS SpacemiT: needs BUILD_SHARED_LIBS=ON: the core loads the K3 plugin with dlopen and "
                        "its linked-in form is not built (plan.md \"S1 design\" A5)")
endif()

# the CPU backend's SpacemiT path would be a second TCM user in the same process
if (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64" AND GGML_CPU_RISCV64_SPACEMIT)
    message(FATAL_ERROR "FlagOS SpacemiT: GGML_FLAGOS_SPACEMIT and GGML_CPU_RISCV64_SPACEMIT cannot be combined; "
                        "build them in separate build directories")
endif()

list(APPEND FLAGOS_SOURCES
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-api.h"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit.cpp"
    "${FLAGOS_SPACEMIT_DIR}/core/flagos-spacemit-plugin.h"
    "${FLAGOS_SPACEMIT_DIR}/core/flagos-spacemit-stats.h"
    "${FLAGOS_SPACEMIT_DIR}/core/flagos-spacemit-diagnostics.h"
    "${FLAGOS_SPACEMIT_DIR}/core/flagos-spacemit-diagnostics.cpp"
    "${FLAGOS_SPACEMIT_DIR}/core/flagos-spacemit-loader.h"
    "${FLAGOS_SPACEMIT_DIR}/core/flagos-spacemit-loader.cpp"
    "${FLAGOS_SPACEMIT_DIR}/core/flagos-spacemit-plan.h"
    "${FLAGOS_SPACEMIT_DIR}/core/flagos-spacemit-plan.cpp"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-ops.h"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-ops.cpp")
list(APPEND FLAGOS_PRIVATE_DEFINITIONS GGML_FLAGOS_HAVE_SPACEMIT)
# the core's loader calls dlopen itself (ggml's dl_load_library lives in libggml, which a backend must not depend on)
list(APPEND FLAGOS_PRIVATE_LIBRARIES ${CMAKE_DL_LIBS})
