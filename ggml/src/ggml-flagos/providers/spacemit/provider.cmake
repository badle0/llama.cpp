# SpacemiT K3 provider. On riscv64 it needs spine-runtime (libspert) to run ops on the A100 AI cores and a compiler
# with the SpacemiT IME2 instructions (GCC >= 15); elsewhere it builds without them, finds no device, and its executor
# falls back to a serial stand-in and the kernels to scalar references, used by tests.

set(FLAGOS_SPACEMIT_DIR "${CMAKE_CURRENT_LIST_DIR}")

list(APPEND FLAGOS_SOURCES
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-api.h"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit.cpp"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-exec.h"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-exec.cpp"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-ops.h"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-ops.cpp"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-kernels.h"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-kernels.cpp"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-weights.h"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-weights.cpp"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-ime.h"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-ime.cpp"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-ime-kernels.h"
    "${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-ime-kernels.cpp")
list(APPEND FLAGOS_PRIVATE_DEFINITIONS GGML_FLAGOS_HAVE_SPACEMIT)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    # the CPU backend's SpacemiT path would be a second TCM user in the same process
    if (GGML_CPU_RISCV64_SPACEMIT)
        message(FATAL_ERROR "FlagOS SpacemiT: GGML_FLAGOS_SPACEMIT and GGML_CPU_RISCV64_SPACEMIT cannot be combined; "
                            "build them in separate build directories")
    endif()

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

    # the instructions the copied IME2 kernels use (flagos-spacemit-ime-kernels.cpp)
    include(CheckCSourceCompiles)
    set(FLAGOS_SPACEMIT_IME_MARCH "-march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii")
    set(CMAKE_REQUIRED_FLAGS "${FLAGOS_SPACEMIT_IME_MARCH} -mabi=lp64d")
    check_c_source_compiles("int main(void) { __asm__ volatile(\"vmadotu.hp v16, v2, v4, v0, 0, i4\"); return 0; }" FLAGOS_SPACEMIT_HAS_VMADOTU_HP)
    check_c_source_compiles("int main(void) { __asm__ volatile(\"vmadotsu.hp v16, v3, v4, v0, 4, i4\"); return 0; }" FLAGOS_SPACEMIT_HAS_VMADOTSU_HP)
    check_c_source_compiles("int main(void) { __asm__ volatile(\"vpack.vv v2, v0, v1, 2\"); return 0; }" FLAGOS_SPACEMIT_HAS_VPACK)
    check_c_source_compiles("int main(void) { __asm__ volatile(\"vnpack4.vv v12, v0, v1, 3\"); return 0; }" FLAGOS_SPACEMIT_HAS_VNPACK4)
    check_c_source_compiles("int main(void) { __asm__ volatile(\"vupack.vv v2, v12, v12, 2\"); return 0; }" FLAGOS_SPACEMIT_HAS_VUPACK)
    unset(CMAKE_REQUIRED_FLAGS)
    if (NOT (FLAGOS_SPACEMIT_HAS_VMADOTU_HP AND FLAGOS_SPACEMIT_HAS_VMADOTSU_HP AND FLAGOS_SPACEMIT_HAS_VPACK AND
             FLAGOS_SPACEMIT_HAS_VNPACK4 AND FLAGOS_SPACEMIT_HAS_VUPACK))
        message(FATAL_ERROR "FlagOS SpacemiT: the compiler lacks the SpacemiT IME2 instructions (needs GCC >= 15); "
                            "see CMakeFiles/CMakeConfigureLog.yaml for the failed checks")
    endif()
    list(APPEND FLAGOS_PRIVATE_DEFINITIONS GGML_FLAGOS_SPACEMIT_IME2)

    # only the kernels need the vector ISA; they run on the A100 cores (RVV 1024)
    set_source_files_properties("${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-kernels.cpp" PROPERTIES
        COMPILE_OPTIONS "-march=rv64gcv_zfh_zvfh_zba;-mabi=lp64d")
    set_source_files_properties("${FLAGOS_SPACEMIT_DIR}/flagos-spacemit-ime-kernels.cpp" PROPERTIES
        COMPILE_OPTIONS "${FLAGOS_SPACEMIT_IME_MARCH};-mabi=lp64d")
    message(STATUS "FlagOS: enabling SpacemiT K3 provider with spine-runtime at ${FLAGOS_SPACEMIT_SPERT_DIR} and IME2 kernels")
else()
    message(STATUS "FlagOS: enabling SpacemiT K3 provider without spine-runtime (not riscv64: no device, serial test executor, reference kernels)")
endif()
