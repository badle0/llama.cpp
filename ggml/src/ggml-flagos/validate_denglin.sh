#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../../.." && pwd)"
build_dir="${FLAGOS_BUILD_DIR:-${repo_dir}/build-flagos}"
model="${1:-${repo_dir}/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"

if [[ ! -f "${model}" ]]; then
    echo "Model not found: ${model}" >&2
    exit 1
fi

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DGGML_FLAGOS=ON \
    -DGGML_BACKEND_DL=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_CURL=OFF \
    -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "${build_dir}" -j4 --target \
    flagos-check-kernels test-backend-ops llama-bench llama-cli llama-server

export FLAGOS_KERNEL_DIR="${script_dir}/kernels/aot"
export FLAGOS_DEQUANT_BLAS=1
export FLAGOS_GRAPH_CAPTURE=1

"${build_dir}/bin/flagos-check-kernels" "${FLAGOS_KERNEL_DIR}/flagos_kernels.cubin"

"${build_dir}/bin/test-backend-ops" test -b FlagOS0

"${build_dir}/bin/llama-bench" \
    -m "${model}" -p 128 -n 32 -t 4 -r 3 -ngl 99 \
    -ot 'blk\..*=FlagOS0;output.*=FlagOS0'

env -u FLAGOS_DEQUANT_BLAS -u FLAGOS_GRAPH_CAPTURE \
    "${build_dir}/bin/llama-bench" \
    -m "${model}" -p 128 -n 32 -t 4 -r 3 -ngl 0 -dev none

"${build_dir}/bin/llama-bench" \
    -m "${model}" -p 2048 -n 256 -b 512 -ub 512 -t 4 -r 3 -ngl 99 \
    -ot 'blk\..*=FlagOS0;output.*=FlagOS0'
