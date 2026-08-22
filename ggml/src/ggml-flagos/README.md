# GGML FlagOS multi-provider backend

This directory contains the provider-neutral FlagOS registry, graph planner, and
the first production provider for Denglin GPGPU. llama.cpp remains responsible
for GGUF loading, graph construction, scheduling, and CPU fallback. Each FlagOS
provider owns its device buffers, queues, events, AOT dispatch, vendor libraries,
and native execution plan.

This is a target-first implementation for a KS20-A, validated with
Qwen2.5-1.5B and Qwen3-4B. Its
`supports_op` checks are intentionally strict. Unsupported operations and shapes
remain on the CPU through the normal GGML scheduler.

The execution boundary is:

```text
GGUF / KV cache / graph construction (llama.cpp)
    -> GGML scheduler (supports_op + supports_buft)
        -> ggml-flagos backend
            -> fused graph patterns when available
            -> Triton AOT cubin or Denglin BLAS
            -> CPU fallback for unsupported nodes
        -> optional CUDA-compatible graph capture and replay
```

## Build

The Denglin provider requires its CUDA-compatible SDK. The default SDK root is
`/usr/local/dlgpu/sdk`; set `FLAGOS_DENGLIN_SDK_ROOT` before configuration to
override it. The older `FLAGOS_SDK_ROOT` setting remains accepted for existing
build directories.

```sh
cmake -S . -B build-flagos \
    -DGGML_FLAGOS=ON \
    -DGGML_FLAGOS_DENGLIN=ON \
    -DGGML_BACKEND_DL=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_CURL=OFF \
    -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-flagos -j4 --target test-backend-ops llama-bench llama-cli llama-server
```

The provider-neutral core can be configured without the Denglin SDK by setting
`GGML_FLAGOS_DENGLIN=OFF`. It builds a zero-device registry plus the Provider ABI
and Graph Plan checks, which is the build-isolation baseline for AMD development.

The Denglin build links `libcurt` and `libdlblas`. It also checks that every required AOT
kernel in `kernels/aot` exists. `kernels/generate_flagos_kernels.py` is the source
used to compile and validate those Triton kernels in the Denglin PyTorch/Triton
environment.

The generated `manifest.json` is the source of truth for kernel filenames,
symbols, block sizes, warp counts, and shared-memory sizes. The registry loads
the merged `flagos_kernels.cubin` once and resolves all symbols from it, with the
per-kernel cubins as a fallback. The shape/stride scalars of the generic strided
copy are marked `do_not_specialize`, so Triton cannot silently remove parameters
whose build-time value happens to be one; its C++ driver launcher has a stable
ABI without disabling useful alignment specialization for the hot model kernels.

The source boundary is:

```text
flagos-provider.*              versioned provider contract and identity
flagos-registry.cpp            aggregate FlagOS registry and global device mapping
flagos-graph-plan.*            provider-neutral graph patterns and plan cache
providers/denglin/             Denglin runtime, memory, launcher, DLBLAS, and graph
providers/amd/                 HIP/gfx1150 contract and fail-closed build entry
```

Each provider owns its `provider.cmake`, SDK discovery, vendor libraries, and AOT
package checks. Enabling or modifying AMD therefore does not add HIP branches to
the common CMake file or the Denglin provider.

Build and run the opt-in module/launcher smoke test with:

```sh
cmake --build build-flagos --target flagos-check-kernels
LD_LIBRARY_PATH=/path/to/denglin/sdk/lib \
build-flagos/bin/flagos-check-kernels \
    ggml/src/ggml-flagos/kernels/aot/flagos_kernels.cubin
```

## Run

The Qwen3-4B Q4_K_M validation model can be downloaded reproducibly from
ModelScope. The include filter avoids downloading the other quantizations in the
repository:

```sh
modelscope download \
    --model unsloth/Qwen3-4B-GGUF \
    --include '*Q4_K_M*' \
    --local_dir models/qwen3-4b-q4_k_m \
    --max-workers 4
```

The file used for the target validation is
`models/qwen3-4b-q4_k_m/Qwen3-4B-Q4_K_M.gguf` (2,497,281,312 bytes,
SHA-256 `f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a`).

Run its memory-conservative direct-Q4_K path with:

```sh
export FLAGOS_KERNEL_DIR="$PWD/ggml/src/ggml-flagos/kernels/aot"
export FLAGOS_GRAPH_CAPTURE=1

build-flagos/bin/llama-cli \
    -m models/qwen3-4b-q4_k_m/Qwen3-4B-Q4_K_M.gguf \
    -p "你好，请用一句话介绍你自己。" -n 32 -t 4 -ngl 99 \
    -ot 'blk\..*=FlagOS0,output.*=FlagOS0' \
    --temp 0 -s 1 -no-cnv -st
```

The high-performance mode dequantizes Q4_K/Q6_K weights to an F16 device cache
and dispatches matrix multiplication through Denglin BLAS. Graph capture is
separate and can be enabled independently.

```sh
export FLAGOS_KERNEL_DIR="$PWD/ggml/src/ggml-flagos/kernels/aot"
export FLAGOS_DEQUANT_BLAS=1
export FLAGOS_GRAPH_CAPTURE=1

build-flagos/bin/llama-cli \
    -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    -p "Gravity is" -n 32 -t 4 -ngl 99 \
    -ot 'blk\..*=FlagOS0,output.*=FlagOS0' \
    --temp 0 -s 1 -no-cnv -st
```

For `llama-bench`, separate tensor override rules with a semicolon:

```sh
build-flagos/bin/llama-bench \
    -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    -p 128 -n 32 -t 4 -r 3 -ngl 99 \
    -ot 'blk\..*=FlagOS0;output.*=FlagOS0'
```

Use `-b 512 -ub 512` when the prompt is longer than 512 tokens. The current
multi-column BLAS path deliberately caps one graph batch at 512 columns. Also use
`-np 1` with `llama-server`; decode flash attention currently covers one sequence,
so a multi-slot server configuration disables that fused path.

## Implemented execution paths

The merged module contains 34 Triton AOT kernels:

- dense and strided copy, F32-to-F16 and F16-to-F32 casts, fill, and scale;
- F32 ADD, SUB, MUL, DIV, SIGMOID, EXP, SOFTPLUS, and split-input SwiGLU;
- SUM_ROWS, L2_NORM, NORM, CUMSUM, SOFT_MAX, RMS_NORM, and fused
  RMS_NORM plus MUL;
- Q4_K/Q6_K dequantization, packed one-warp decode GEMV, and batched projection
  kernels. The decode kernels unpack two Q4_K values or four Q6_K values per
  program lane to reduce launch occupancy and duplicated packed-weight loads;
- Q4_K/Q6_K/F32 GET_ROWS, F32-to-F16 SET_ROWS, NEOX RoPE, decode-only GQA flash
  attention, and SSM_CONV. Decode attention uses a runtime GQA head mapping:
  one AOT cubin covers any positive query/KV head counts where the query count
  is divisible by the KV count. It is numerically tested for Qwen2.5's 12/2 and
  Qwen3-4B's 32/8 configurations; head dimension 128 and one query token remain
  deliberate target constraints.

RESHAPE, VIEW, PERMUTE, and TRANSPOSE are handled as zero-launch view nodes.
`supports_op` validates dtype, shape, layout, stride range, and target-specific
limits before a node is assigned to FlagOS. It does not advertise generic GGML
coverage for signatures the AOT kernels have not proven.

With `FLAGOS_DEQUANT_BLAS=1`, Q4_K/Q6_K matrix multiplication supports one to 512
contiguous activation columns. Weights are dequantized once by Triton AOT kernels,
activations are cast to F16, and GEMM is sent to `libdlblas`. On the measured
Qwen2.5-1.5B model this cache contains 197 weights and consumes 3,087,138,816
bytes. Leave the option disabled when that memory tradeoff is not acceptable.

`FLAGOS_Q4_DLBLAS=1` enables a second, decode-only Q4_K variant. It repacks each
immutable GGUF weight once into the unsigned-INT4 group-quant layout consumed by
Denglin `dlblasGemmExV2`, stores F16 scale/zero parameters in
`[K/group_size, output_rows]` order, and uses group size 32. DLBLAS writes F16;
the added AOT cast writes the F32 tensor required by GGML. Unsupported shapes,
partial weight updates, or a failed optional vendor dispatch fall back to the
existing direct AOT implementation.

This variant is not enabled by default. Qwen3-4B caches 216 repacked Q4_K
weights and consumes 1,961,164,800 additional device bytes. The recovered SDK
also JIT-compiles its quantized GEMV on first use and currently requires a
matching Denglin compiler/CUDA-header toolchain. A production package should
ship those kernels AOT or provide a persistent vendor JIT cache before this
variant is selected by default.

The recovered `libdleol` aborts the process when JIT compilation cannot produce
the requested CU function. To fail safely, this backend ignores
`FLAGOS_Q4_DLBLAS` unless `DLEOL_JIT_USE_DLCC=1` and a non-empty
`DLEOL_CU_COMPILE_OPTIONS` are both present. This preflight protects the known
incomplete SDK environment; it is not a substitute for a vendor-supported AOT
package and versioned ABI.

Graph execution now goes through a provider-neutral structural planner before
launch. The planner canonicalizes the GGML graph without tensor pointer identity,
finds legal pattern candidates, asks the active provider whether it can lower each
candidate, and greedily selects non-overlapping candidates by provider cost. Plan
cache hits use a structural fingerprint followed by full canonical equality; a
nonzero GGML graph UID is only a last-plan fast path.

The first Denglin lowering is adjacent RMS_NORM plus channel-wise MUL. It launches
the existing Triton AOT macro-kernel and preserves both GGML outputs. The common
catalog also names later Qwen3.8 and hybrid-model targets, while the current
Denglin provider declines every pattern for which it has no executable. Candidate
recognition currently covers RMS_NORM+MUL, ADD+RMS_NORM+MUL, and SSM_CONV+SILU;
only RMS_NORM+MUL is enabled on Denglin. Repeated F32-to-F16 activation casts used
by QKV and FFN projections are also reused within a graph evaluation.

This is a first Graph Plan slice, not a Common IR pass: it does not rewrite the
GGML graph or reduce GGML allocation, and it only combines nodes that are already
adjacent. It establishes the CUDA-free Pattern/Plan and provider `can_lower`
boundary that a later ARM `ACCEL` provider can reuse with CPU AOT macro-kernels
and a cached threaded execution plan.

`FLAGOS_GRAPH_CAPTURE=1` captures stable device subgraphs with at least 16 nodes
after one warmup. A semantic snapshot checks node/source identity, type, shape,
stride, op parameters, and resolved device pointers before replay. Shape or
pointer changes invalidate and recapture the slot. The cache is bounded to 16
graphs and evicts the least-recently-used slot after stream synchronization.
The current target benefits primarily during decode. Prefill attention is still
a CPU fallback; the multi-column projection and FFN GEMMs run on the GPGPU.

## Validation

Run the numerical backend tests with the same performance mode used for inference:

```sh
FLAGOS_DEQUANT_BLAS=1 \
FLAGOS_KERNEL_DIR="$PWD/ggml/src/ggml-flagos/kernels/aot" \
build-flagos/bin/test-backend-ops test -b FlagOS0
```

The planner has a device-independent opt-in unit test. It verifies provider
accept/decline behavior, overlap selection, structural cache reuse across new
tensor addresses, shape invalidation, capture safety, and SSM_CONV+SILU catalog
recognition:

```sh
cmake --build build-flagos --target flagos-check-graph-plan
build-flagos/bin/flagos-check-graph-plan
```

The provider tests validate the versioned descriptor, nonzero stable identity,
memory and execution capabilities, aggregate registry ordering, and the mapping
between global FlagOS ordinals and provider-local ordinals. The registry test
uses mock Denglin and AMD providers, so it runs without either vendor SDK:

```sh
cmake --build build-flagos --target flagos-check-provider flagos-check-registry
build-flagos/bin/flagos-check-provider
build-flagos/bin/flagos-check-registry
```

The KS20-A direct-AOT target run on 2026-08-22 passed all 392 operations that the
backend declared supported. This includes both 12/2 and 32/8 decode-attention
layouts plus dense, reshaped, permuted, sliced, and strided-destination F32
CPY/CONT cases. End-to-end greedy output is semantically correct, but it is
not expected to remain token-for-token identical to the CPU indefinitely because
the high-performance path uses F16 dequantized weights and F16 activations.

A short graph-enabled Qwen2.5-1.5B run produced one capture and three replays;
the final kernel counters reported 352 FlagOS graph calls, 997 quantized matrix
multiplications, and 291 fused RMS_NORM plus MUL launches.

Measured with Qwen2.5-1.5B Q4_K_M, four CPU threads, three benchmark repetitions:

| Backend | PP128 (token/s) | TG32 (token/s) |
| --- | ---: | ---: |
| CPU (`-ngl 0 -dev none`) | 24.07 +/- 0.11 | 16.97 +/- 0.03 |
| FlagOS high-performance + graph | 503.79 +/- 2.62 | 17.28 +/- 0.01 |

Three repetitions of a longer `-p 2048 -n 256 -b 512 -ub 512` run produced
296.84 +/- 1.02 prompt tokens/s and 17.23 +/- 0.04 generation tokens/s. During
that run, `dlsmi` reported 4,478 MiB for the process and 4,767 / 16,384 MiB total
board memory usage.

Local `llama-server` SSE TTFT was measured with an exact 128-token prompt,
`-np 1`, prompt caching disabled, and model load excluded:

| Backend | First request TTFT | Warm weight-cache TTFT |
| --- | ---: | ---: |
| CPU (`-ngl 0 -dev none`) | 5,341.81 ms | 5,333.11 ms |
| FlagOS high-performance + graph | 425.03 ms | 276.48 ms |

The first FlagOS request includes lazy dequantization of the model weights. The
warm number represents steady-state serving after that cache is populated.

These numbers establish an end-to-end target baseline, not a general performance
claim for other models, shapes, or Denglin products.

Qwen3-4B Q4_K_M was also validated end to end on 2026-08-22. The dedicated
32-query-head/8-KV-head backend test passed against the CPU reference, as did the
existing 12/2 regression test. A synchronized four-token CLI run completed with
108 direct decode FlashAttention launches. With graph capture enabled, a
32-token `/no_think` CLI run completed a coherent answer and reported one capture
and 29 replays; `llama-bench` TG16 reported one capture and fifteen replays.

The following three-repetition numbers were measured while an unrelated process
occupied about 8 GiB of the 16 GiB board, so the F16 dequantization cache was
intentionally disabled:

| Backend/path | PP64 (token/s) | TG16 (token/s) |
| --- | ---: | ---: |
| CPU, 4 threads | 8.59 +/- 0.00 | 6.86 +/- 0.03 |
| FlagOS direct Q4_K AOT + graph | 7.80 +/- 0.00 | 4.73 +/- 0.08 |

This memory-conservative path proves model coverage, decode attention lowering,
and graph replay, but it is not yet a Qwen3 performance win. The direct decode
gain from the initial four-warp build is 113% (TG16 2.22 to 4.73 token/s). The
32-token end-to-end run generated at 4.8 token/s with one capture and 29 replays.
The remaining gap is dominated by quantized matrix-vector execution. Enabling
the current full F16 dequantization cache needs substantially more free device
memory; a memory-efficient vendor quantized GEMM or tiled/evictable
dequantization cache is the next performance path for this model.

That memory-efficient vendor path is now implemented experimentally with
DLBLAS. A varied-parameter standalone probe matched the CPU INT4 group-quant
reference exactly at 16x256 and 256x256; model-sized probes differed only by the
expected F16 output rounding. Qwen3-4B produced coherent output and the DLBLAS
path remained capture-safe in the tested decode graph (one capture, nine
replays). Three warmed TG16 repetitions on the same partially occupied board
measured:

| Decode path | CUDA Graph | TG16 (token/s) |
| --- | --- | ---: |
| Direct Q4_K AOT | on | 4.74 +/- 0.04 |
| Q4_K DLBLAS | off | 10.03 +/- 0.00 |
| Q4_K DLBLAS | on | 10.75 +/- 0.54 |

The DLBLAS plus Graph steady-state result is 127% above the same-run direct-AOT
control. Cold start remains unresolved: a no-warmup TG16 run measured 1.21
token/s because vendor JIT compilation occurs inside the first inference.
Consequently this is a successful performance variant, but not yet a suitable
default or a production cold-start result.

## Diagnostics

- `FLAGOS_LOG_KERNELS=1` prints kernel, cache, capture, and replay counters at
  backend destruction.
- `FLAGOS_LOG_GRAPH_PLAN=1` prints newly built structural plans and selected patterns.
- `FLAGOS_NO_GRAPH_FUSION=1` keeps the planner active but makes Denglin decline all
  fused patterns, providing a direct-op correctness and performance control.
- `FLAGOS_LOG_UNSUPPORTED_OPS=1` prints each unsupported operation signature once.
- `FLAGOS_TRACE_OPS=1` traces nodes accepted by graph execution.
- `FLAGOS_SYNC_EACH_OP=1` synchronizes after each launch to isolate device errors.
- `FLAGOS_LOG_COPY=1` prints dense/strided copy signatures.
- `FLAGOS_KERNEL_DIR=/path/to/aot` overrides the build-time AOT asset location.
- `FLAGOS_DEQUANT_BLAS=1` enables cached F16 dequantization and vendor BLAS.
- `FLAGOS_Q4_DLBLAS=1` enables the experimental decode-only repacked-Q4_K DLBLAS
  variant; it currently needs the vendor JIT compiler environment described
  above and about 1.83 GiB extra memory for Qwen3-4B.
- `FLAGOS_GRAPH_CAPTURE=1` enables stable-subgraph capture and replay.

The next optimization boundary is decode profiling and broader shape coverage.
Qwen3.8-27B is not yet an end-to-end supported target: its Gated DeltaNet path,
256-wide attention heads, wider RMS rows, and complete MRoPE/KV-store signatures
still need matching AOT kernels or vendor implementations before their catalog
patterns can be accepted. Common IR is not required by this backend and should
only be inserted after the profitable GGML subgraph patterns and memory lifetimes
are stable.

Run `bash ggml/src/ggml-flagos/validate_denglin.sh [model.gguf]` inside the
Denglin SDK container to rebuild and repeat the operator, short-context, CPU
baseline, and long-context validation sequence.
