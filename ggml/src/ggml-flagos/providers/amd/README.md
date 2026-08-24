# FlagOS AMD provider boundary

The AMD provider must implement `flagos_amd_provider()` from
`../../flagos-provider.h`. It is a peer of the Denglin provider, not a branch in
the Denglin launcher and not a wrapper around the upstream `ggml-hip` backend.

The first target is the Radeon 890M in Ryzen AI 9 HX 370 (`gfx1150`). Keep all
HIP and ROCm types inside this directory. The provider-neutral registry and
graph planner must continue to build with no ROCm installation.

The provider owns:

- HIP device discovery and a stable UUID or PCI-derived identity;
- device, pinned, and coherent host-visible memory where the APU supports it;
- HIP streams, events, copies, synchronization, and backend lifetime;
- HSACO loading, symbol lookup, packed argument ABI, and kernel launch;
- hipBLAS or rocBLAS dispatch and any provider-specific quantized library;
- `supports_op` validation matching the actual AOT package;
- lowering of accepted common graph patterns;
- HIP Graph capture, instantiate, replay, invalidation, and bounded caching.

The implementation must return a full GGML device from `device_get()`. The
common registry assigns its global FlagOS ordinal and preserves the provider's
local ordinal. Device names must be globally unambiguous, for example
`FlagOS:AMD:0`. Provider IDs and device UUIDs must be nonzero and unique.

## Current status

The first AMD enablement slice is now implemented for ROCm/HIP hosts. It
provides real device discovery, HIP device buffers, synchronous and
stream-ordered host/device copies, stream synchronization, and HIP-backed
events. On the target Radeon 890M it reports the `gfx1150` architecture and is
visible to `llama-cli --list-devices` when the process has access to the DRM
render group.

The provider now also has an AMD HSACO registry and HIP module launcher. Set
`FLAGOS_AMD_KERNEL_DIR` to a package containing `manifest.json` and per-kernel
`.hsaco` files. The manifest records the Triton symbol, workgroup metadata,
logical block size, and architecture. Triton's two hidden scratch arguments are
appended by the launcher; packages that require profiling scratch are rejected
until a provider scratch allocator is added.

The Qwen KV-cache path additionally recognizes `ROPE -> VIEW -> SET_ROWS` as a
portable graph fusion. When the package contains `flagos_rope_kv_store_f32_f16`,
AMD writes the rotated F32 query/key rows directly into the F16 cache using the
I64 cache indices, eliminating the intermediate flattened view and one launch.

The first graph bindings are `GGML_OP_ADD`, contiguous F32
`GGML_OP_RMS_NORM`, the non-M-RoPE Neox form of `GGML_OP_ROPE`, F32 SiLU
(`GGML_OP_UNARY` with `GGML_UNARY_OP_SILU`), and the Qwen KV-cache
`GGML_OP_SET_ROWS` path (F32 source to F16 cache with I64 row indices). They are
enabled only when the package contains the corresponding symbols
(`flagos_add_f32`, `flagos_rms_norm_f32`, `flagos_rope_neox_f32`,
`flagos_silu_f32`, and `flagos_set_rows_f32_f16`). RMSNorm
uses one Triton program per contiguous row and requires the manifest's
compile-time block to cover the hidden width. RoPE currently requires F32
input/output, I32 positions, no frequency-factor tensor, Neox mode, and the
default non-YaRN options. Without the matching package the provider remains
fail-closed and advertises only `NONE`, view, reshape, permute, and transpose
nodes. Other unary variants remain declined. Qwen3 can now run end to end with
the validated direct AOT operators and CPU fallback for unsupported work. The
optional F32 softmax lowering is additionally restricted to power-of-two row
widths, zero ALiBi bias, and the exact F16-mask/no-mask layouts validated by the
AMD Triton package; other softmax cases remain on CPU.
Multi-node/composite fusions are deliberately staged behind
`FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS=1`. The `RMS_NORM + MUL`,
`ROPE + VIEW + SET_ROWS`, `FLASH_ATTN_EXT` decode, and prefill paths have passed
the standalone AMD conformance graph and are exercised successfully in Qwen3
runs. They remain opt-in because their layouts are intentionally narrower than
the full GGML operator contract. `ADD + RMS_NORM + MUL` remains disabled in the
planner after its model-graph alias audit exposed incorrect logits; its kernels
are retained for further debugging. This keeps the default Qwen path correct
while preserving an explicit bring-up switch.

The AMD provider also supports the F32 `GGML_OP_GLU` SwiGLU split form using
`flagos_swiglu_split_f32`. This is the operator-fused FFN activation used by
the Qwen graph when the two projection results are already available.

Quantized Qwen projection and embedding paths are now available when the AOT
package contains the Q4_K/Q6_K symbols:

```text
GET_ROWS(Q4_K/Q6_K -> F32)
MUL_MAT(Q4_K/Q6_K, F32 -> F32), decode and multi-column prefill
```

The common planner exposes a provider-neutral
`flagos_quantized_matmul_signature` (weight kind, K, output rows, and columns).
AMD uses it to select the decode or batched FlagTree kernel while other
providers can map the same signature to rocBLAS, hipBLAS, vendor INT4, or a
different AOT implementation. Unsupported quantization layouts remain on the
CPU scheduler path.

The common graph planner recognizes the portable
`MUL_MAT + MUL_MAT + GLU(SWIGLU)` FFN pattern as `ffn_swiglu`. AMD currently
has two opt-in lowerings behind that same provider-neutral pattern. For
multi-column prefill, `flagos_ffn_swiglu_f16_f32_batched` consumes the two
Q4_K/Q6_K projections through the persistent F16 dequant caches, computes both
projection tiles, and applies SwiGLU before writing the terminal F32 output.
It is enabled with `FLAGOS_AMD_PREFILL_F16_GEMM=1` and by adding `ffn_swiglu`
to `FLAGOS_AMD_FUSIONS` while
`FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS=1` is set. The capability gate requires
matching contiguous K/rows/columns, a shared F32 activation source, more than
one column, production-scale projection shapes, and non-aliasing buffers.
Cache allocation remains outside HIP Graph capture.

For single-column decode, an optional
`flagos_ffn_swiglu_q4_k_f32_decode` kernel keeps both projection weights in
Q4_K form, shares the activation loads between the gate and up dot products,
and avoids materializing either projection before applying SwiGLU. It uses the
same graph pattern and fusion selector, but is emitted only with
`FLAGOS_AMD_EMIT_Q4_FFN_DECODE=1`. The current gfx1150 tile computes eight
rows with two waves; the manifest carries the row tile so the runtime can
reject incompatible shapes. In a 6144x2048 microbenchmark the fused kernel was
about 0.187 ms versus 0.204 ms for two narrow Q4 GEMVs, before accounting for
the eliminated standalone SwiGLU launch. On Qwen3-1.7B it raised `tg32` from
about 51.9 to 53.8 t/s and `tg128` from 51.8 to 53.5 t/s. A deterministic
64-token CLI comparison produced identical generated text. Providers without
either lowering continue to execute the individual nodes normally.

The planner also classifies sink-free `FLASH_ATTN_EXT` nodes as
`flash_attn_decode`. AMD lowers the validated F32-Q/F16-KV, head-dim-128,
zero-softcap form to `flagos_flash_attn_decode_f32_f16`; providers that do not
offer this implementation simply decline the candidate.

Fusion is routed through the common `flagos-graph-plan` layer rather than an
AMD-specific graph walker. The provider-neutral plan discovers patterns and
queries provider capabilities; each provider supplies a lowering choice and an
execution callback. Candidates are explicitly classified as either a
single-operator implementation (currently decode `FLASH_ATTN_EXT`) or a
multi-node graph implementation. When the experimental switch is enabled, the
AMD slice executes the validated `RMS_NORM + MUL` pattern. The package includes
both the ordinary two-output kernel and an alias-safe one-output
`flagos_rms_norm_mul_inplace_f32` variant because GGML may reuse the dead
RMSNorm allocation for the terminal MUL result. `ADD + RMS_NORM + MUL` has an
experimental ABI and kernels, but its planner query is deliberately disabled
until end-to-end residual-buffer alias coverage is complete; this prevents
silently corrupting logits on model graphs. Denglin uses the same
execution-hook interface for its existing fused kernel. Unsupported patterns
remain direct nodes and are handled by the normal backend scheduler or CPU
fallback. The AMD `supports_op` boundary admits standalone F32 `MUL` only for
contiguous full-shape or hidden-size-vector repeats matching the Triton ABI;
arbitrary multidimensional broadcasts remain on CPU. Leave the switch unset
for the validated Qwen3 path.

HIP Graph capture is currently an explicit experiment, not the default path:

```text
FLAGOS_AMD_GRAPH_CAPTURE=1
```

The scheduler splits Qwen3 into backend subgraphs of at most eight nodes. The
provider therefore uses a small eligibility threshold and a two-observation
pointer-stability warm-up; it will not repeatedly capture when temporary
tensor addresses rotate. In the current llama scheduler, decode attention
shapes grow with the KV length, so the structural fingerprint changes every
token and the Qwen run observes captures but no safe replays. This is an
intentional fail-safe result, not a production graph-acceleration claim.
The F16 prefill path is eligible after all production-scale projection weights
in a subgraph have been dequantized; the first pass warms those caches outside
capture, and later stable shapes may be captured. The FFN/SwiGLU fusion remains
capture-unsafe because it can still populate caches while lowering a graph.
Graph update/bucketing for dynamic KV lengths is still required before this
switch can improve steady-state decode.

For historical context, an earlier interactive comparison on the local Radeon
890M (`gfx1150`) with the same Qwen3-1.7B Q4_K_M model was:

| Path | Prompt (t/s) | Generation (t/s) |
| --- | ---: | ---: |
| Native ROCm/HIP | ~585 | ~61.5 |
| FlagOS AMD direct AOT + standalone MUL | ~178--183 | ~31--32 |
| FlagOS AMD `rms_norm_mul` + `flash_attn_decode` | ~206--212 | ~50--52 |
| FlagOS AMD all validated experiments | ~236--254 | ~51.7--52.3 |
| FlagOS AMD opt-in F16 prefill GEMM + validated fusions | ~1,100 | ~51.5 |
| FlagOS AMD opt-in F16 GEMM + fused FFN/SwiGLU | ~1,158 | ~51.5 |
| FlagOS AMD grouped F16 GEMM + grouped fused FFN/SwiGLU | ~1,747 | ~53.3 |

For a directly repeatable `llama-bench` comparison (`pp512/tg32`, three
repetitions), a native ROCm/HIP build compiled for the actual runtime target
`gfx1150` (no `HSA_OVERRIDE_GFX_VERSION`, `GGML_CUDA_FORCE_MMQ=ON`) measured
`1802/60.73` t/s on this machine. The validated FlagOS AOT path measured
`164/37` t/s with no experimental graph fusions; enabling the validated
`rms_norm_mul`, `rope_kv_store`, and FlashAttention fusions raises it to
`302/50.7` t/s. The current opt-in F16 cache path, using a
manifest-described `32x64x32` tile and two warps for the dense GEMM, measured
`1094/51.3` t/s; adding the FFN/SwiGLU projection fusion raises this to
`1158/51.5` t/s over three repetitions. A gfx1150 retune of the standalone
dense projection kernel to a `64x128x32` tile with four warps measures about
`1292/51.4` t/s with the same fused prefill FFN path. The previous 29-kernel
gfx1150 experiment additionally uses the eight-row Q4 decode GEMV and fused
Q4 decode FFN/SwiGLU; it measured `1308/53.80` t/s. In the combined benchmark
it reached `529.45` t/s versus `645.99` t/s for native ROCm. The current
31-kernel package adds target-gated grouped scheduling to the standalone F16
GEMM and fused FFN/SwiGLU. Five-repetition runs measure about `1747/53.3` t/s
for `pp512/tg128`, versus native ROCm MMQ at `1792/60.67` t/s. The combined
`pp512+tg32` test reaches `581.99` t/s versus native `644.86` t/s. This closes
prefill to roughly 97.5% of native while decode remains about 87.9% and the
combined workload about 90.3%.
The remaining decode gap is now concentrated in the standalone Q4/Q6 matvecs,
especially Q6_K, plus launch/scheduler overhead around the remaining nodes.

The native binary was configured with `GGML_CUDA_FORCE_MMQ=ON` because the
installed ROCm 5.1 rocBLAS package has no `gfx1150` Tensile library (a normal
gfx1150 run otherwise aborts during rocBLAS initialization). This is still a
native ROCm quantized-kernel comparison, and it avoids pretending that a
gfx1151 compatibility override is a fair gfx1150 result.

Long-prefill measurements with grouped scheduling are about `1659` t/s for
`pp1024` and `1523` t/s for `pp2048` with the normal `-ub 512` setting. During
this work a zero-length
intermediate ADD/RMSNorm graph exposed a scheduler edge case; the provider now
rejects zero-element ADD/RMSNorm tensors before launch, and both multi-chunk
long-prompt cases pass again.

The default FlagOS run is launch-bound rather than device-discovery-bound:
one representative run recorded 5,060 AOT launches and 1,440 host-to-device
copies for eight generated tokens, with no selected graph patterns. The fully
validated opt-in run recorded 17,238 launches, 5,712 fusion launches, 3,808
RMSNorm+MUL fusions, 952 RoPE+KV-store fusions, 868 decode-attention fusions,
84 prefill-attention fusions, and 102 H2D copies. The remaining gap to native
ROCm is primarily quantized matvec/AOT efficiency, per-subgraph launch
overhead, and the lack of a vendor quantized GEMM or HIP-Graph-update path.

For a repeatable `llama-bench` check (`pp512/tg32`), the direct-fusion package
measured approximately 301--303 t/s prefill and 50--51 t/s decode on this
Radeon 890M. The retuned F16 prefill package initially reached about 1,300
t/s; the grouped 31-kernel package now reaches about 1,747/53.3 t/s and keeps
fixed-seed generated text identical to the two-dimensional fallback. The
benchmark is more stable than an interactive `llama-cli` run and should be used for
regression tracking. The AOT call sites use fixed-capacity stack argument
lists, avoiding a host heap allocation for each kernel launch; the launcher
still packs the ABI pointers into HIP's launch array.

An additional, explicitly opt-in prefill path is available in the 27-kernel
package generated by the AMD tool:

```text
FLAGOS_AMD_PREFILL_F16_GEMM=1
FLAGOS_AMD_KERNEL_DIR=/tmp/flagos-amd-gfx1150-f16gemm
```

For production-scale Q4_K/Q6_K projection matrices (`rows >= 64`, `K >=
1024`, and more than one activation column), it decodes each weight tensor to
an F16 cache once and uses a Triton F16xF32 tiled GEMM for subsequent prefill
graphs. The path passed the current 260-case backend-op suite and raised the local `pp512`
result from roughly 302 to 1,098 t/s while leaving `tg32` near 51 t/s. The
tile dimensions are recorded in the AOT manifest (`tile_m`, `tile_n`,
`tile_k`); the C++ launcher reads them when constructing the grid, so changing
the generator tile cannot silently under-cover the output tensor.
Short prompts can be slower on the first pass because the one-time F16 decode
cost is then not amortized: in the same harness, `pp32` was about 432 t/s with
the F16 path versus 274 t/s for direct AOT. The benefit is targeted at
long-prefill and reuse workloads. It remains opt-in because the cache lifetime
is tied to stable GGML weight tensor objects and needs broader model/scheduler
coverage before becoming the default package. The cache is capped at 8192 MiB
per backend context by default; when the cap is reached, new weights
automatically fall back to the direct quantized AOT GEMM. Override the limit
with `FLAGOS_AMD_DEQUANT_CACHE_MAX_MB` (set it to `0` for unlimited residency)
when the device has enough memory. The FFN/SwiGLU planner performs the same
capacity check for both projection weights before selecting its three-node
fusion, so a constrained cache declines the fusion and preserves the normal
three-node fallback instead of failing an entire graph.

An additional research-only path emits MFMA-oriented tiled Q4_K/Q6_K
prefill kernels (`flagos_mul_mat_*_tiled`). Enable it with
`FLAGOS_AMD_QUANT_TILED_GEMM=1` and a package generated with
`FLAGOS_QUANT_TILE_BLOCK_M/N` (the validated experimental package used
`16x32`, four warps). It is capability-gated and falls back to the ordinary
batched AOT kernels when the package does not contain the tiled symbols. On
the Radeon 890M this first implementation is numerically validated by the
generator but currently slower than the existing direct and F16-cache paths
(about 172 t/s vs 302 t/s direct and 1,100 t/s F16-cache at `pp512`), so it is
not part of the default package. Its value is as an integration seam for a
future MFMA-optimized dequantization/GEMM implementation, not as a production
performance claim.

### Build and smoke test

On a ROCm installation, configure with:

```bash
cmake -S . -B build-flagos-amd \
  -DGGML_FLAGOS=ON -DGGML_FLAGOS_AMD=ON -DGGML_FLAGOS_DENGLIN=OFF \
  -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON
cmake --build build-flagos-amd -j4 --target flagos-check-amd llama-cli
```

The AMD package generator uses the installed FlagTree/Triton compiler and
emits the manifest plus 27 gfx1150 HSACO kernels, including the validated
four-row wave32 Q4 decode variant and the opt-in F16 prefill GEMM:

```bash
FLAGOS_AMD_ARCH=gfx1150 TRITON_CACHE_DIR=/tmp/flagos-amd-cache \
  python \
  ggml/src/ggml-flagos/providers/amd/tools/generate_flagos_amd_kernels.py \
  --output-dir /tmp/flagos-amd-gfx1150 \
  --cache-dir /tmp/flagos-amd-cache
FLAGOS_AMD_KERNEL_DIR=/tmp/flagos-amd-gfx1150 \
  ./build-flagos-amd/bin/flagos-check-amd
```

For the additional FFN projection/SwiGLU fusion, add
`FLAGOS_AMD_EMIT_FFN_FUSION=1` while generating the package.  This emits a
29-kernel package containing both the two-dimensional and grouped
`flagos_ffn_swiglu_f16_f32_*` variants; it is still
runtime opt-in through the fusion selector described above.

The gfx1150 decode experiments are emitted independently:

```text
FLAGOS_AMD_EMIT_Q4_GEMV_NARROW8=1
FLAGOS_AMD_EMIT_Q4_FFN_DECODE=1
```

Together with `FLAGOS_AMD_EMIT_FFN_FUSION=1`, these produce the validated
31-kernel experiment package. At runtime, set
`FLAGOS_AMD_Q4_GEMV_NARROW=8`; an older package without the eight-row symbol
automatically falls back to the four-row kernel. The decode FFN kernel is
selected through the existing `ffn_swiglu` fusion gate, for example:

```text
FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS=1
FLAGOS_AMD_FUSIONS=all
FLAGOS_AMD_PREFILL_F16_GEMM=1
FLAGOS_AMD_Q4_GEMV_NARROW=8
```

To additionally emit the research-only tiled Q4_K/Q6_K prefill kernels, add
`FLAGOS_AMD_EMIT_QUANT_TILED=1` to the generator environment and enable
`FLAGOS_AMD_QUANT_TILED_GEMM=1` at runtime. The ordinary command above remains
the stable 27-kernel package.

The common generator exposes `FLAGOS_GEMV_NUM_WARPS` for target experiments;
the validated gfx1150 package deliberately keeps the one-warp decode setting.
`FLAGOS_Q4_FFN_DECODE_BLOCK_M` and
`FLAGOS_Q4_FFN_DECODE_NUM_WARPS` tune the optional quantized decode fusion;
the validated Radeon 890M values are eight rows and two waves.
For the optional dense prefill kernel, `FLAGOS_F16_MATMUL_BLOCK_M/N/K` and
`FLAGOS_F16_MATMUL_NUM_WARPS` control the tile and warp shape. The validated
Radeon 890M defaults are `64/128/32` and four warps for the standalone dense
GEMM. On gfx1150, prefill widths of at least 256 columns automatically use an
additional one-dimensional `GROUP_M=8`, stage-1 schedule; smaller prefill
widths and other AMD targets keep the two-dimensional kernel. Set
`FLAGOS_AMD_GROUPED_F16_GEMM=0` to disable it, or `1` to opt in while tuning a
different target. The fused FFN/SwiGLU kernel keeps a lower-register
`32/64/32`, two-warp tile and uses a separate `GROUP_M=4`, stage-2 grouped
schedule on gfx1150 when the prefill has at least 32 columns. These knobs are
for target experiments and should be re-benchmarked on other AMD
architectures.

For operator-level attribution, set `FLAGOS_PROFILE_KERNELS=1`. The AMD AOT
registry wraps each non-captured launch with HIP timing events and reports per
symbol launch count, total device time, and average microseconds when the
backend or provider registry is destroyed. Symbols are ordered by cumulative
device time and split by launch grid so projections with different output
shapes remain distinguishable. This intentionally serializes launches, so use
it to identify expensive kernels rather than to report model throughput. An
optional `FLAGOS_PROFILE_DUMP_SYNC_INTERVAL=N` prints a cumulative snapshot
after every Nth backend synchronization. This is useful for programs such as
`llama-bench` that retain the provider registry until process exit; keep the
interval unset during ordinary benchmarks. If the application also bypasses
the backend synchronize callback, `FLAGOS_PROFILE_DUMP_LAUNCH_INTERVAL=N`
prints a cumulative snapshot after every N timed AOT launches instead.
An eight-token Qwen3-1.7B decode sample attributed about 98.3 ms to Q4_K GEMV,
47.8 ms to Q6_K GEMV, and 6.9 ms to FlashAttention decode; quantized GEMV was
therefore roughly 88 percent of the measured AOT device time in that sample.
In a later identical-length profile window, the eight-row Q4 kernel reduced
the aggregate Q4 time from about 371 to 336 ms. After enabling the quantized
decode FFN fusion, the remaining standalone Q4 launches accounted for about
148 ms, the fused gate/up/SwiGLU launches for 171 ms, and Q6_K remained about
184 ms. These event-timed runs serialize launches and are attribution data,
not throughput measurements.

Run device-facing checks with the account that owns the DRM render-device
permission (the example adds the group for a single command):

```bash
sudo -n setpriv --reuid="$USER" --regid="$USER" \
  --groups="$USER",render \
  env LD_LIBRARY_PATH="$PWD/build-flagos-amd/bin:/usr/lib/x86_64-linux-gnu" \
./build-flagos-amd/bin/flagos-check-amd
```

Set `FLAGOS_LOG_GRAPH_PLAN=1` together with `-v` when running `llama-cli` to
print the provider-neutral AMD plan and the selected operator/graph patterns.

`llama-cli --list-devices` loads `libggml-flagos.so` from its executable
directory. `GGML_BACKEND_PATH` may also point at that specific `.so` file.
Without the `render` permission HIP probing can legitimately return no
devices; this is an OS access issue rather than a provider compile failure.

Suggested source layout:

```text
providers/amd/
    provider.cmake
    flagos-amd.cpp
    flagos-amd-api.h
    kernels/
        manifest.json
        gfx1150/*.hsaco
```

`provider.cmake` discovers HIP only when `GGML_FLAGOS_AMD=ON`, appends the AMD
sources and `hip::host` to the parent `FLAGOS_*` lists, and defines
`GGML_FLAGOS_HAVE_AMD`. Vendor math libraries and AOT modules should be added
in later, capability-gated increments rather than linked speculatively.

Minimum enablement gates:

1. `GGML_FLAGOS_DENGLIN=OFF` and `GGML_FLAGOS_AMD=ON` configure on a ROCm host.
2. Core and Graph Plan tests pass without linking Denglin libraries.
3. Device enumeration, allocation, copies, events, and synchronization pass.
4. Every advertised op passes `test-backend-ops` numerically.
5. Qwen3-4B Q4_K_M runs end to end with unsupported nodes on CPU.
6. HIP Graph reports real capture and replay counters with output unchanged.
7. Cold-start, TTFT, decode throughput, memory, and fallback counts are recorded.
