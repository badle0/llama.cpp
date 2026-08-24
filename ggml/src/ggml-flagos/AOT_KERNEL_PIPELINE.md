# FlagOS AOT Kernel Pipeline

This document explains how the current Denglin AOT kernels are produced, packaged, loaded, selected, and launched by `ggml-flagos`. It also defines the steps required to add a new direct GGML operator.

## Scope and terminology

The current integration is an AOT package consumed by the Denglin provider. It is not a runtime Python bridge and it does not call a central `libflagos_operator` library.

The `.cubin` suffix means a loadable device module for the Denglin CUDA-compatible Driver API in this provider. A module produced by the Denglin compiler is tied to the Denglin target and runtime. It is not an NVIDIA binary and must not be reused by an NVIDIA provider. AMD must use an HSACO or another ROCm code object, while a CPU provider must use an ELF object, a shared library, or statically linked functions.

The package contains:

- one merged module, `flagos_kernels.cubin`, which exports every AOT kernel;
- one module per kernel, such as `flagos_scale_f32.cubin`, for fallback and diagnosis;
- `manifest.json`, which records filenames, symbols, launch metadata, and the current schema version.

## End-to-end data flow

```text
Triton kernel source
    -> Denglin PyTorch and Triton JIT
    -> Triton cache: cubin + LLIR + JSON metadata
    -> copy per-kernel cubins into an AOT staging directory
    -> merge the per-kernel LLIR files
    -> Denglin dlcc compiles flagos_kernels.cubin
    -> generate manifest.json
    -> ggml-flagos loads the package during backend initialization
    -> supports_op accepts only compatible GGML signatures
    -> GGML Scheduler assigns accepted nodes to the FlagOS device
    -> the Denglin launcher binds tensor addresses and scalar arguments
    -> cuLaunchKernel submits the selected AOT function to the provider stream
```

CUDA Graph capture occurs after this lowering and binding. It records launches that the provider can already execute. CUDA Graph does not compile Triton source and does not replace the AOT package.

## Source and generated files

| File | Role |
| --- | --- |
| [`kernels/generate_flagos_kernels.py`](kernels/generate_flagos_kernels.py) | Triton kernel source, compile-time launch configurations, device-side numerical checks, artifact extraction, Manifest generation, and merged-module compilation |
| [`kernels/merge_flagos_module.py`](kernels/merge_flagos_module.py) | Merges Triton LLVM IR modules while preserving kernel annotations and renumbering module-local metadata |
| [`kernels/aot/manifest.json`](kernels/aot/manifest.json) | Runtime package catalog and launch metadata |
| `kernels/aot/flagos_kernels.cubin` | Preferred merged device module |
| `kernels/aot/flagos_*.cubin` | Per-kernel fallback modules |
| [`kernels/check_merged_module.cpp`](kernels/check_merged_module.cpp) | Loads the merged module, resolves every required symbol, and executes a scalar-launch ABI check |
| [`providers/denglin/flagos-denglin.cpp`](providers/denglin/flagos-denglin.cpp) | Manifest parser, module loader, GGML signature checks, typed argument binding, kernel launch, vendor-library dispatch, and CUDA Graph handling |
| [`providers/denglin/provider.cmake`](providers/denglin/provider.cmake) | Denglin SDK discovery, vendor-library linkage, AOT asset checks, and the default kernel directory |

Files under `kernels/aot/` are generated artifacts. The Python and C++ sources are the reviewable source of truth for semantics and argument binding. The current Manifest v2 is not yet a complete ABI source of truth because it does not describe argument types or all shape constraints.

## Build environment

Generating the package requires a real Denglin device and the same vendor software family used for deployment:

- a Denglin-compatible PyTorch build where `torch.cuda` selects the Denglin device;
- a Denglin Triton backend capable of emitting `.cubin`, `.llir`, and `.json` cache artifacts;
- the Denglin SDK with `bin/dlcc`, headers, and runtime libraries;
- Python packages imported by the generator: `torch` and `triton`;
- an empty `TRITON_CACHE_DIR` for each generation run.

Do not generate directly into the tracked `kernels/aot/` directory. Generate into a staging directory, validate it, review the Manifest and binary differences, and then update the tracked package as a separate deliberate action.

Record the toolchain identity before generation:

```sh
python - <<'PY'
import torch
import triton

print("torch", torch.__version__)
print("triton", triton.__version__)
print("device", torch.cuda.get_device_name(0))
PY

/path/to/denglin/sdk/bin/dlcc --version
```

The published package should record these versions together with the target product, driver/runtime version, and output hashes. Manifest v2 does not carry those fields, so they must currently be recorded in release evidence.

## Generate a package

Use an empty cache and a separate output directory:

```sh
repo_dir=/path/to/llama.cpp
sdk_root=/path/to/denglin/sdk
kernel_cache=$(mktemp -d /tmp/flagos-triton-cache.XXXXXX)
aot_output=$(mktemp -d /tmp/flagos-aot-output.XXXXXX)

export FLAGOS_SDK_ROOT="$sdk_root"
export TRITON_CACHE_DIR="$kernel_cache"
export LD_LIBRARY_PATH="$sdk_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python "$repo_dir/ggml/src/ggml-flagos/kernels/generate_flagos_kernels.py" \
    --sdk-root "$sdk_root" \
    --output-dir "$aot_output"
```

`compile_kernels()` allocates test tensors on the Denglin device and invokes every Triton kernel with an approved compile-time configuration. Triton JIT writes the device module, LLVM IR, and metadata into `TRITON_CACHE_DIR`. The generator compares each result against a PyTorch reference before accepting the artifact.

`copy_artifact()` requires exactly one cache artifact for each expected symbol. It copies the individual module into the staging directory and transfers the symbol name, shared-memory size, warp count, warp size, and logical block size into `manifest.json`.

Unless `--no-merge` is used, the generator finds each kernel's `.llir`, calls `merge_flagos_module.py`, verifies that no kernel symbol was lost, and invokes Denglin `dlcc` with target triple `dlgput64-unknown-cuda`. The result is `flagos_kernels.cubin`. With `--no-merge`, the Manifest omits the merged module and the runtime loads every individual module.

Inspect the staging package before using it:

```sh
python -m json.tool "$aot_output/manifest.json"
find "$aot_output" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
sha256sum "$aot_output"/* | sort
diff -qr "$aot_output" "$repo_dir/ggml/src/ggml-flagos/kernels/aot"
```

A binary difference is expected after a compiler or source change, but it must be accompanied by the corresponding source change, Manifest review, numerical validation, and target information.

## Why the merged module exists

The Denglin Triton backend emits one kernel per compilation unit. Loading every output separately causes one `cuModuleLoad` call and one live module handle per kernel.

The SDK link paths tested for finished cubins were not usable for this package. The working implementation merges the textual LLVM IR before the final `dlcc` invocation. `merge_flagos_module.py`:

- removes debug metadata that cannot be safely combined without full remapping;
- renumbers LLVM attribute groups and kernel annotations;
- deduplicates declarations and compatible module globals;
- preserves every callable kernel entry point;
- emits one LLVM module for the final Denglin compilation.

The runtime therefore prefers one load of `flagos_kernels.cubin`. It falls back to per-kernel modules only when the merged module is absent or unusable.

## Manifest and module loading

The current Manifest entry for the scale kernel is equivalent to:

```json
{
  "file": "flagos_scale_f32.cubin",
  "name": "flagos_scale_f32",
  "shared": 0,
  "num_warps": 4,
  "warp_size": 32,
  "block_size": 256
}
```

During backend initialization, `flagos_kernel_registry::initialize()` selects the package directory in this order:

1. `FLAGOS_KERNEL_DIR` from the process environment;
2. `FLAGOS_KERNEL_DIR` compiled by the Denglin provider CMake configuration.

It parses `manifest.json`, validates every expected entry, and tries:

```text
cuModuleLoad(flagos_kernels.cubin)
    -> cuModuleGetFunction for every Manifest symbol
```

If that fails cleanly, it tries each entry separately:

```text
cuModuleLoad(flagos_scale_f32.cubin)
    -> cuModuleGetFunction("flagos_scale_f32")
```

The module and every resolved `CUfunction` remain alive for the backend lifetime.

## `GGML_OP_SCALE` binding example

The Triton signature is:

```python
def flagos_scale_f32(x, output, scale, bias, n_elements, BLOCK: tl.constexpr)
```

The GGML operation computes:

```text
output[i] = input[i] * scale + bias
```

The runtime binding is:

| Triton argument | C++ value | Source |
| --- | --- | --- |
| `x` | `src0_data` | Resolved device address of `node->src[0]` |
| `output` | `dst->data` | Device address allocated for the GGML result |
| `scale` | `float scale` | First F32 value in `dst->op_params` |
| `bias` | `float bias` | Second F32 value in `dst->op_params` |
| `n_elements` | `int n_elements` | `ggml_nelements(dst)` |
| `BLOCK` | no runtime argument | Compile-time Triton constant recorded as `block_size` in the Manifest |

The C++ launcher constructs the runtime argument array in exactly the same order:

```cpp
void * arguments[] = {
    &src0_data,
    &dst_data,
    &scale,
    &bias,
    &n_elements,
};
```

It calculates `grid_x = ceil(n_elements / block_size)`. The physical launch width is `num_warps * warp_size`, while `block_size` is the number of logical elements handled by one Triton program. These are different values and must not be interchanged.

Before scheduling, `supports_op` accepts `GGML_OP_SCALE` only when the input and output are F32, have the same shape, are contiguous, and fit the 32-bit element-count ABI. Unsupported signatures stay on another GGML backend, normally CPU. The execution switch calls `launch_scale()` only for nodes that satisfied this contract.

## Launcher ABI rules

The Python signature, generated device entry point, Manifest metadata, C++ argument array, and `supports_op` constraints form one ABI. A mismatch can produce silent numerical errors, invalid memory access, or a launch failure.

Follow these rules:

- Preserve the exact runtime argument order.
- Match scalar widths and signedness. The current launchers use explicit `float`, `int`, and pointer values.
- Do not pass `tl.constexpr` arguments at runtime.
- Record every launch constant that affects the grid, layout, shared memory, or accepted shape.
- Prevent Triton from specializing away a scalar that must remain a runtime argument. Use `do_not_specialize` where required and verify the result through a real Driver API launch.
- Make `supports_op` no broader than the compiled variant. Shape, type, layout, stride, alignment, and integer-range constraints must match the launcher.
- Keep device addresses valid for asynchronous execution and CUDA Graph replay.
- Treat changing the kernel signature as an ABI change that requires regenerating the package and updating the launcher in the same change.

Manifest v2 does not encode the argument schema or an ABI hash. The current implementation relies on code review and launch tests to keep both sides synchronized. The planned Manifest v3 and generated typed bindings should replace this manual contract.

## Runtime relationship with FlagOS

The provider-neutral FlagOS layer owns provider registration, device identity, capability reporting, graph pattern discovery, and execution-plan caching. It does not interpret Denglin cubins.

The Denglin provider owns:

- the CUDA-compatible Runtime and Driver API;
- device memory, streams, events, copies, and synchronization;
- the AOT Manifest and module loader;
- C++ launch bindings and `supports_op` rules;
- Denglin BLAS dispatch and optional quantized-library paths;
- provider-specific CUDA Graph capture and replay.

The integration boundary is therefore:

```text
GGML graph
    -> common FlagOS planning and provider selection
    -> Denglin signature validation and lowering
    -> AOT kernel or Denglin library selection
    -> Denglin C++ launcher
    -> Denglin Runtime and Driver
```

The AOT package is currently stored inside the llama.cpp source tree and consumed directly by the provider. A future external FlagOS package service or installed operator library would need the same Manifest, ABI, target, and validation contract, but that service is not part of the current implementation.

## Build and validation

Configure the Denglin provider with the SDK and, when necessary, its CUDA-compatible header directory:

```sh
repo_dir=/path/to/llama.cpp
build_dir=/path/to/build-flagos

export FLAGOS_DENGLIN_SDK_ROOT=/path/to/denglin/sdk
export FLAGOS_DENGLIN_CUDA_INCLUDE_DIR=/path/to/denglin/cuda-compatible/include

cmake -S "$repo_dir" -B "$build_dir" \
    -DGGML_FLAGOS=ON \
    -DGGML_FLAGOS_DENGLIN=ON \
    -DGGML_BACKEND_DL=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_CURL=OFF

cmake --build "$build_dir" -j4 --target \
    ggml-flagos flagos-check-kernels test-backend-ops
```

Validate the merged module and launcher ABI:

```sh
export LD_LIBRARY_PATH="$FLAGOS_DENGLIN_SDK_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$build_dir/bin/flagos-check-kernels" \
    "$repo_dir/ggml/src/ggml-flagos/kernels/aot/flagos_kernels.cubin"
```

The current package reports 35 checks: 34 required symbols plus one real strided-copy launch ABI check.

Run the numerical backend suite with the package under test:

```sh
export FLAGOS_KERNEL_DIR=/path/to/validated/aot-package
"$build_dir/bin/test-backend-ops" test -b FlagOS0
```

An accepted operator must pass against the CPU reference. Module resolution alone does not prove numerical correctness. After the operator suite, run the target GGUF model and record backend placement, CPU fallbacks, output correctness, graph capture/replay counters, latency, throughput, and memory use.

## Add a direct AOT operator

Use this checklist for a new GGML operator or a new compiled variant:

1. Add the Triton kernel to `generate_flagos_kernels.py` with an explicit runtime signature and compile-time constants.
2. Exercise it in `compile_kernels()` on a real Denglin device and compare its output with a PyTorch reference, including boundary shapes.
3. Add `copy_artifact()` for the exported symbol so the generator emits its module and Manifest entry.
4. Add the expected AOT asset to `providers/denglin/provider.cmake` while the build still uses an explicit preflight list.
5. Add a `flagos_kernel` member and a registry entry with the exact exported symbol.
6. Add a typed C++ launcher that binds addresses and scalar values in the Triton signature order.
7. Add strict `supports_op` checks for every compiled type, shape, layout, stride, alignment, and range limit.
8. Add the GGML execution-switch case or provider lowering implementation.
9. Add focused `test-backend-ops` coverage for accepted and rejected signatures.
10. Regenerate into a staging directory, review the Manifest and binaries, run the merged-module test, run numerical tests, and complete an end-to-end model test.

For a fused pattern, also add the provider-neutral pattern description, dependency and overlap checks, provider lowering query, multi-output preservation rules, and fused execution binding. A fused kernel and CUDA Graph solve different problems: the fused kernel changes the device work, while CUDA Graph reduces repeated host submission overhead.

## Cross-provider packages

Source-level algorithms and canonical GGML signatures can be shared, but compiled device modules and launch APIs are provider-specific:

| Provider | Expected package | Runtime launch API |
| --- | --- | --- |
| Denglin | Denglin device module with the current `.cubin` convention | Denglin CUDA-compatible Driver API |
| NVIDIA | `sm_XX` cubin, optionally PTX as an explicit fallback | Native CUDA Driver API |
| AMD | `gfxXXXX` HSACO or ROCm code object | HIP Module API |
| ARM CPU | ELF object, shared library, or static function table | Direct CPU call or command plan |
| SpacemiT K3 | Vendor-defined AI Core AOT module | K3 provider launcher |

Do not select a package by filename suffix alone. A production Manifest must identify the provider, target architecture, runtime ABI, compiler version, kernel ABI hash, and binary hashes before a provider advertises support.

## Current limitations

The implementation is a validated vertical slice, not the final package system:

- The target matcher is available to providers, but the current AMD manifest
  still exposes a flat kernel catalog; variant requirements and tuning scores
  are not yet serialized in the package.
- Manifest v2 has no provider target, compiler identity, runtime ABI, argument schema, constraints, ABI hash, or file hashes.
- `supports_op` and C++ launch binding are manually synchronized with the generator.
- CMake maintains a separate AOT filename preflight list.
- The compiled default package path points into the source tree unless `FLAGOS_KERNEL_DIR` overrides it.
- Per-kernel modules duplicate symbols already present in the merged module, but they remain useful for fallback and diagnosis.
- AOT generation requires a real Denglin software and hardware environment and is not part of a normal llama.cpp build.

These limitations must be addressed before treating the package as a portable multi-vendor FlagOS ABI.
