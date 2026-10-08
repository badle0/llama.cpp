# SpacemiT K3 provider. M1 has no external dependencies; spine-runtime and the IME kernels come with M2.
# The provider compiles on any host; it only finds a device on a K3 (riscv64 Linux with A100 cores).

list(APPEND FLAGOS_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/flagos-spacemit-api.h"
    "${CMAKE_CURRENT_LIST_DIR}/flagos-spacemit.cpp")
list(APPEND FLAGOS_PRIVATE_DEFINITIONS GGML_FLAGOS_HAVE_SPACEMIT)

message(STATUS "FlagOS: enabling SpacemiT K3 provider")
