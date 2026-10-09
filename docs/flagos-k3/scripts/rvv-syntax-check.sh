#!/bin/sh
# Syntax check of the SpacemiT provider's riscv64-only code (RVV and IME kernels) on macOS, which has no RISC-V
# toolchain. Apple clang parses for riscv64 with RVV, using rvv-syntax/riscv_vector.h (Apple clang ships without it)
# and the macOS SDK headers presented as Apple's. It checks types, intrinsic names and signatures and template
# instantiations; nothing is assembled or linked, so the K3 build (GCC 15) is still the real test.
# Usage (any directory): docs/flagos-k3/scripts/rvv-syntax-check.sh [file.cpp ...]   (default: the kernel files)
cd "$(dirname "$0")/../../.." || exit 1
INC="docs/flagos-k3/scripts/rvv-syntax"
SDK=$(xcrun --show-sdk-path) || exit 1
LOGS=${TMPDIR:-/tmp}
[ $# -gt 0 ] || set -- flagos-spacemit-rvv-kernels.cpp flagos-spacemit-kernels.cpp flagos-spacemit-ime-kernels.cpp flagos-spacemit-ime.cpp
status=0
for f in "$@"; do
  log="$LOGS/rvv-syntax-$f.log"
  clang++ --target=riscv64-unknown-linux-gnu -march=rv64gcv_zfh_zvfh_zba -mabi=lp64d -std=c++17 -fsyntax-only \
    -Wall -Wextra -Wpedantic -Wcast-qual -Wmissing-declarations -Wshadow -Wsign-compare -Wno-unused-function \
    -DGGML_FLAGOS_SPACEMIT_RVV -DGGML_FLAGOS_SPACEMIT_IME2 -I"$INC" -Iggml/include -Iggml/src -Iggml/src/ggml-flagos \
    -nostdinc++ -isystem "$SDK/usr/include/c++/v1" -isystem "$(clang -print-resource-dir)/include" -isystem "$SDK/usr/include" \
    -U__linux__ -U__linux -Ulinux -U__gnu_linux__ -U__unix__ -U__unix -Uunix -D__arm64__ -D__APPLE__ -D__MACH__ \
    "ggml/src/ggml-flagos/providers/spacemit/$f" > "$log" 2>&1 || status=1
  echo "$f: $(grep -c 'error:' "$log") errors, $(grep 'warning:' "$log" | grep -c 'providers/spacemit') warnings in provider code (log: $log)"
  grep -E 'error:|providers/spacemit.*warning:' "$log" | head -5
done
exit $status
