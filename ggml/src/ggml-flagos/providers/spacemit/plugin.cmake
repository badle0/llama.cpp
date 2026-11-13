# The SpacemiT K3 plugin (plan.md "S2 spec", step 1), its own shared library that nothing links: the core dlopens it
# from beside itself (core/flagos-spacemit-loader.h). On riscv64 it needs spine-runtime (libspert) to run ops on the
# A100 AI cores and a compiler with the SpacemiT IME2 instructions (GCC >= 15); elsewhere it builds without them, finds
# no device (unless FLAGOS_SPACEMIT_HOST_TEST_DEVICE is ON), and its executor falls back to a serial stand-in and the
# kernels to the references, used by tests.
# Included from ggml/src/ggml-flagos/CMakeLists.txt right after the ggml-flagos target, in that directory, whose source
# file properties these are.

add_library(flagos-provider-spacemit SHARED
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-plugin.cpp"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-exec.h"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-exec.cpp"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-kernels.h"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-kernels.cpp"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-rvv-kernels.h"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-rvv-kernels.cpp"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-weights.h"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-weights.cpp"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-x100.h"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-x100.cpp"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-ime.h"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-ime.cpp"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-ime-kernels.h"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-ime-kernels.cpp"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-attn.h"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-attn.cpp"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-rvv-attn.h"
    "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-rvv-attn.cpp")
target_link_libraries(flagos-provider-spacemit PRIVATE ggml-base)
target_include_directories(flagos-provider-spacemit PRIVATE ..)
# no version suffix, the name the loader opens; beside libggml-flagos, as ggml_add_backend_library places a module; only
# the entry symbol exported (FLAGOS_SPACEMIT_PLUGIN_EXPORT)
set_target_properties(flagos-provider-spacemit PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
# nothing links the plugin, and --target llama-bench must still build it
add_dependencies(ggml-flagos flagos-provider-spacemit)
# beside the core, as ggml_add_backend_library installs it
if (GGML_BACKEND_DL)
    if (GGML_BACKEND_DIR)
        install(TARGETS flagos-provider-spacemit LIBRARY DESTINATION ${GGML_BACKEND_DIR})
    else()
        install(TARGETS flagos-provider-spacemit LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
else()
    install(TARGETS flagos-provider-spacemit LIBRARY)
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    set(FLAGOS_SPACEMIT_SPERT_DIR "$ENV{HOME}/spine-runtime" CACHE PATH "spine-runtime release directory (with include/ and lib/)")
    find_library(FLAGOS_SPACEMIT_SPERT_LIB NAMES spert PATHS "${FLAGOS_SPACEMIT_SPERT_DIR}/lib" NO_DEFAULT_PATH)
    if (NOT FLAGOS_SPACEMIT_SPERT_LIB OR NOT EXISTS "${FLAGOS_SPACEMIT_SPERT_DIR}/include/spert.hpp")
        message(FATAL_ERROR "FlagOS SpacemiT: spine-runtime not found in ${FLAGOS_SPACEMIT_SPERT_DIR}; "
                            "set -DFLAGOS_SPACEMIT_SPERT_DIR=<extracted spine-runtime release>")
    endif()
    target_include_directories(flagos-provider-spacemit PRIVATE "${FLAGOS_SPACEMIT_SPERT_DIR}/include")
    # linked by full path, so binaries in the build tree find libspert through their RPATH
    target_link_libraries(flagos-provider-spacemit PRIVATE "${FLAGOS_SPACEMIT_SPERT_LIB}")
    target_compile_definitions(flagos-provider-spacemit PRIVATE GGML_FLAGOS_SPACEMIT_SPERT)

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
    target_compile_definitions(flagos-provider-spacemit PRIVATE GGML_FLAGOS_SPACEMIT_IME2 GGML_FLAGOS_SPACEMIT_RVV)

    # only the kernels need the vector ISA; they run on the A100 cores (RVV 1024)
    set_source_files_properties("${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-kernels.cpp"
                                "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-rvv-kernels.cpp"
                                "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-attn.cpp"
                                "${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-rvv-attn.cpp" PROPERTIES
        COMPILE_OPTIONS "-march=rv64gcv_zfh_zvfh_zba;-mabi=lp64d")
    set_source_files_properties("${FLAGOS_SPACEMIT_DIR}/plugin/flagos-spacemit-ime-kernels.cpp" PROPERTIES
        COMPILE_OPTIONS "${FLAGOS_SPACEMIT_IME_MARCH};-mabi=lp64d")
    message(STATUS "FlagOS: enabling SpacemiT K3 provider with spine-runtime at ${FLAGOS_SPACEMIT_SPERT_DIR} and IME2 kernels")
else()
    # test only: a simulated device on this host (8 tiles run one after another by the serial executor, reference
    # kernels), so test-backend-ops and flagos-check-spacemit can run the provider's code before it reaches the K3
    option(FLAGOS_SPACEMIT_HOST_TEST_DEVICE "FlagOS SpacemiT: expose a simulated device on non-riscv64 hosts (tests only)" OFF)
    if (FLAGOS_SPACEMIT_HOST_TEST_DEVICE)
        target_compile_definitions(flagos-provider-spacemit PRIVATE GGML_FLAGOS_SPACEMIT_HOST_TEST_DEVICE)
        message(STATUS "FlagOS: enabling SpacemiT K3 provider with a simulated test device (not riscv64: serial executor, reference kernels)")
    else()
        message(STATUS "FlagOS: enabling SpacemiT K3 provider without spine-runtime (not riscv64: no device, serial test executor, reference kernels)")
    endif()
endif()
