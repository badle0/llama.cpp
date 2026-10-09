# SpacemiT K3 provider. On riscv64 it needs spine-runtime (libspert) to run ops on the A100 AI cores;
# elsewhere it builds without it, finds no device, and its executor falls back to a serial stand-in used by tests.

set(FLAGOS_SPACEMIT_DIR "${CMAKE_CURRENT_LIST_DIR}")

list(APPEND FLAGOS_SOURCES
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-api.h"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit.cpp"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-exec.h"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-exec.cpp"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-ops.h"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-ops.cpp"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-kernels.h"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-kernels.cpp")
list(APPEND FLAGOS_PRIVATE_DEFINITIONS GGML_FLAGOS_HAVE_SPACEMIT)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    set(FLAGOS_SPACEMIT_SPERT_DIR "$ENV{HOME}/spine-runtime" CACHE PATH "spine-runtime release directory (with include/ and lib/)")
    find_library(FLAGOS_SPACEMIT_SPERT_LIB NAMES spert PATHS "${FLAGOS_SPACEMIT_SPERT_DIR}/lib" NO_DEFAULT_PATH)
    if (NOT FLAGOS_SPACEMIT_SPERT_LIB OR NOT EXISTS "${FLAGOS_SPACEMIT_SPERT_DIR}/include/spert.hpp")
        message(FATAL_ERROR "FlagOS SpacemiT: spine-runtime not found in ${FLAGOS_SPACEMIT_SPERT_DIR}; "
                            "set -DFLAGOS_SPACEMIT_SPERT_DIR=<extracted spine-runtime release>")
    endif()
    list(APPEND FLAGOS_PRIVATE_INCLUDE_DIRS "${FLAGOS_SPACEMIT_SPERT_DIR}/include")
    # linked by full path, so binaries in the build tree find libspert through their RPATH
    list(APPEND FLAGOS_PRIVATE_LIBRARIES "${FLAGOS_SPACEMIT_SPERT_LIB}")
    list(APPEND FLAGOS_PRIVATE_DEFINITIONS GGML_FLAGOS_SPACEMIT_SPERT)
    # only the kernels need the vector ISA; they run on the A100 cores (RVV 1024)
    set_source_files_properties("${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-kernels.cpp" PROPERTIES
        COMPILE_OPTIONS "-march=rv64gcv_zfh_zvfh_zba;-mabi=lp64d")
    message(STATUS "FlagOS: enabling SpacemiT K3 provider with spine-runtime at ${FLAGOS_SPACEMIT_SPERT_DIR}")
else()
    message(STATUS "FlagOS: enabling SpacemiT K3 provider without spine-runtime (not riscv64: no device, serial test executor)")
endif()
