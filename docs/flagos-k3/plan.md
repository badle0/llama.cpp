# SpacemiT K3 provider: build plan

Written 2026-10-07. Nothing in this plan is built yet.

- "Design" = `ggml-flagos-provider-design.pdf` V2.0 (review baseline, 2026-09-22). Section numbers like "Design §13.3" refer to it.
- "Mentor design" = `FLAGOS_BACKEND_MULTIPLATFORM_DESIGN.md` and `FLAGOS_BACKEND_MULTIPLATFORM_DESIGN_REVIEW_RESOLUTION.md` on `kevin/feature/flagos-multi-provider-backend` (v0.3, 2026-08-22). Read with `git show kevin/feature/flagos-multi-provider-backend:<file>`.
- Decisions marked *proposed* need the mentor's confirmation (§7). Facts marked UNVERIFIED have not been checked on the board or in code.

## 0. Summary

1. Build a **direct-op provider** (Design §4.1) at `ggml/src/ggml-flagos/providers/spacemit/`, compiled into `libggml-flagos`, that runs ggml ops on the 8 A100 AI cores (CPUs 8-15).
2. **Device type: ggml `ACCEL`** (mentor decision, 2026-10-08). llama.cpp then keeps model layers and the KV cache on the CPU side, tries the provider's buffer type first for each weight whose op the provider supports, and always creates the provider's backend ahead of the CPU backend (§2.8). Whole layers reach the AI cores only if the provider also reads CPU buffers directly: in E1, a backend that did not needed 280 backend switches per generated token.
3. **Memory**: the provider's buffer type holds weights in IME-repacked layout (`is_host = false`), stored once. The KV cache and other inputs stay in CPU buffers, which the provider reads directly (§2.3, D4).
4. Execution: **one spine-runtime launch per split**. Each of the 8 tiles walks the split's nodes, with a barrier after each node and the core's TCM as scratch. This is ggml-spacemit's model.
5. Kernels: reuse the in-tree IME GEMM and repack code (byte-identical to ggml-spacemit's) and port ggml-spacemit's tiling and RVV ops, each behind its own op tests. The mentor allows changing ggml-spacemit's code freely (it is AI-generated); its X0 failures must be fixed, not copied. Neither source is validated yet: the upstream 1198/1198 `MUL_MAT` run never used the IME weight layout (`test-backend-ops` allocates in the CPU backend's plain buffer), and ggml-spacemit fails part of the op tests on this board (`device-type.md` X0). FlagTree AOT kernels come later and only if the mentor wants them.
6. **Attention without flash attention must work.** Under ACCEL, llama.cpp's default `-fa auto` switches flash attention off whenever the provider runs attention, because each layer's device is the CPU (confirmed in E1). The default path is therefore `MUL_MAT` on the KV cache plus masked `SOFT_MAX`; flash attention is used with `-fa on` (§2.4).
7. Order follows Design §13.3: **M0** targets and baselines (no provider code) -> **M1** skeleton -> **M2** direct ops, quantized `MUL_MAT` first -> **M3** AOT package (conditional) -> **M4** fixed fusion (conditional). Design phases P5 (split compiler) and P6 (pre-placement partition) are not needed.

## 1. Facts this plan rests on

| Fact | Evidence |
|---|---|
| Under default device selection, an `ACCEL` device does not join `model->devices`, so layers and KV cache stay on the CPU when no GPU is selected. Its buffer type is tried first for CPU-side weights, and its backend is always added ahead of the CPU backend. Explicit device selection is an exception (section 2.8) | `src/llama.cpp:155-178, 220-223`, `src/llama-model.cpp:903-917`, `src/llama-context.cpp:340-350` |
| Mentor design v0.3: K3 = `GPU` (superseded: the mentor decided on `ACCEL`, 2026-10-08); co-build in `providers/spacemit`; no parallel `ggml-spacemit`; `ggml-cpu/spacemit` is a reference for kernels, repack, affinity and TCM only | mentor design lines 30-31, 38, 170, 179, 768-770; resolution lines 13, 24-26 |
| Mentor design split of work: FlagOS Core owns GGML integration, patterns, legality, planner, fallback, tests; the provider owns device, memory/TCM, AI-core scheduling, AOT loader and launcher, kernels, tuning | mentor design §14.2 |
| The KV cache of an offloaded layer goes into the device's default buffer type | `src/llama-kv-cache.cpp:213-217` |
| The CPU backend accepts any host buffer type | `ggml/src/ggml-cpu/ggml-cpu.cpp:480-481` |
| An op whose output already lives in a buffer (for example a KV-cache write) runs on the highest-priority backend that supports both that buffer type and the op; if none does, the scheduler aborts ("pre-allocated tensor ... cannot run the operation") | `ggml/src/ggml-backend.cpp:877-930` |
| The scheduler spreads non-CPU backends to neighbouring ops they support, and moves ops to a higher-priority backend when buffer types are compatible | `ggml/src/ggml-backend.cpp:1113-1193` |
| FlagOS Common is a registry only: every ggml vtable (supports_op, buffers, graph_compute) belongs to the provider, and `get_proc_address` forwards no provider function (no `set_n_threads`, no extra buffer types) | `ggml/src/ggml-flagos/flagos-registry.cpp:105-126, 135-186` |
| In-tree IME GEMM kernels and activation quantizers take plain pointers; the orchestration around them is tied to ggml-cpu's thread pool | `ggml/src/ggml-cpu/spacemit/ime_kernels.h:72-111`; `ime.cpp:69-70, 393, 707` |
| Upstream IME claims `MUL_MAT` (2-D) and `MUL_MAT_ID` (3-D) for Q2_K-Q6_K, Q4_0/1, Q5_0/1, Q8_0 with F32 activations | `ggml/src/ggml-cpu/spacemit/ime.cpp:1580-1612` |
| TCM works on this board when stale blocks are released and at most 8 compute threads are used | `k3-hardware.md` §4 |
| ggml-spacemit: `ACCEL`, one spert launch per `graph_compute`, a grid barrier after every node, whole per-core TCM per tile, IME only for `MUL_MAT`/`MUL_MAT_ID` on repacked weights; its IME and repack sources are byte-identical to ours | `spacemit-com/llama.cpp` @ `4e782bc`: `ggml/src/ggml-spacemit/ggml-spacemit.cpp:423-515, 590-593` |
| FlagTree SpacemiT: Triton -> riscv64 `.so`, launched through spine-runtime; no AOT export tool; 4 commits, no tests, CI only under QEMU | FlagTree `third_party/spacemit/backend/compiler.py:342-353`, `driver.py:82-107` |
| Reference numbers, upstream IME on A100, 8 threads: Qwen3-0.6B Q4_0 pp128 565.83 / tg128 55.77; Qwen3-4B Q4_0 pp128 79.74 / tg128 11.29 t/s | `docs/build-riscv64-spacemit.md:101-110` |

## 2. Architecture when M2 is done

### 2.1 Where it sits

```
llama.cpp   ACCEL device: backend created automatically, no -ngl
  ggml scheduler: placement, splits, copies, allocation        (unchanged)
    FlagOS Common: registry, profile; side-plan fusion from M4  (unchanged except 3 wiring lines)
      SpacemiT provider
        device + weight buffer type (IME-repacked, stored once) + backend
        reads CPU buffers directly: KV cache, inputs
        executor: spine-runtime stream, 8 tiles on CPUs 8-15, TCM per tile
        kernels: IME GEMM, RVV ops (later: FlagTree AOT .so)
  CPU backend (X100, CPUs 0-7): owns the KV cache; runs every op the provider does not claim
```

### 2.2 Descriptor and device (built in M1)

| Field | Value | Note |
|---|---|---|
| `identity` | fixed non-zero id, name `SpacemiT`, version 1 | Design §5.1 rule 7 |
| `probe` | true only on riscv64 Linux with A100 cores (arch id `0xA064` / marchid `0x8000000041000002`) and `/proc/set_ai_thread` present | no runtime start-up here; ggml-spacemit starts spert in a static constructor, which must not be copied |
| devices | 1, ggml name `FlagOS:SpacemiT:0` | name from mentor design line 170 |
| ggml type | `ACCEL` | D3, adopted (mentor, 2026-10-08); §2.8 |
| `caps.kind` | `cpu_accelerator` | *proposed* (D8), independent of the ggml device type. Kinds: `gpu`, `ai_accelerator`, `cpu_accelerator`, `dsp`, `matrix`, `other` (`flagos-provider.h:9-19`); `ai_accelerator` maps to engine `npu`, `cpu_accelerator` to `cpu` (`flagos-provider.cpp:54-69`). No real provider uses either yet |
| `caps.memory` | `HOST_VISIBLE` and `UNIFIED_COHERENT` | `flagos-provider.h:21-26` |
| `caps.execution` | 0 (synchronous, no events); `AOT_MODULE` from M3 | `flagos-provider.h:28-34` |
| UUID, `memory_domain_id` | fixed non-zero constants | Common never reads `memory_domain_id` today |
| profile | `vector_bits` 1024 (A100, M0.4), `concurrency` 8, vendor `SpacemiT`, architecture `a100`, target `riscv64-a100-ime2`, runtime `spine-runtime <version>`; `features` 0 until RVV/IME bits exist (Common change C2) | without `get_device_profile`, Common derives `vector_bits = 0` |
| ggml props | async false, events false, host_buffer false, buffer_from_host_ptr false | a synchronous ACCEL backend disables pipeline parallelism across several GPUs (mentor design line 723); irrelevant on a K3-only board |
| `get_memory` | `/proc/meminfo` MemAvailable / MemTotal | same RAM as the CPU |

### 2.3 Memory model under ACCEL (D4, *proposed*)

Under ACCEL, llama.cpp puts a weight into the provider's buffer type only when the provider supports the op that uses it (`src/llama-model.cpp:903-917`), and the KV cache always goes to CPU buffers (`src/llama-kv-cache.cpp:211-217`). The scheduler also allocates the provider's compute tensors from its default buffer type.

| Option | How | For | Against |
|---|---|---|---|
| **A2 (proposed)** | Default buffer type holds weights in IME-repacked layout, `is_host = false` (like ggml-spacemit and ggml-cpu's repack buffers). The provider accepts CPU buffers for every non-weight operand | weights stored once; the CPU never needs provider weights, since a weight only lands there when the provider claims its op | provider outputs read by CPU ops are copied (one `memcpy` per split boundary); `get_tensor` must undo the repack |
| A1 (earlier proposal) | Default buffer = plain host memory plus a private repacked copy of each quantized weight | the CPU can read every provider buffer | quantized weights stored twice (Qwen3-4B Q4_0: +2.2 GiB); its main benefit, CPU fallback on a provider-owned KV cache, does not exist under ACCEL |
| A3 | Host default buffer plus a second, repacked buffer type | 1x memory and host-readable activations | llama.cpp uses only an ACCEL device's default buffer type; a second one would need llama.cpp and Common changes (C3) |

A2 rules:
- Buffer type `FlagOS:SpacemiT`: 64-byte alignment, size-0 allocations allowed (llama.cpp probes with one).
- `supports_buft`: own buffer type, or any host buffer type.
- `supports_op` judges a tensor by `t->view_src ? t->view_src->buffer : t->buffer`. ggml-spacemit checks only `t->buffer`; views have none before allocation, so it claimed attention on CPU-resident KV views, which started the E1 failure chain (`device-type.md` §5). Weight operands (buffer usage `WEIGHTS`) must be in the provider's buffer; every other operand may be in the provider's buffer or any host buffer.
- Claims for weight ops must not depend on batch size: llama.cpp picks each weight's buffer once, at load time. If a later graph's op were refused, the scheduler would copy the repacked weight for the CPU.
- `set_tensor` repacks quantized weights (whole-tensor writes); `get_tensor` returns the original layout. Lossy repacks (Q4_1, Q4_K, Q6_K) cannot be inverted: keep their original bytes as well, or do not claim them (R4).
- `init_tensor` attaches repack data only to tensors in weight buffers; ggml-spacemit also attaches it to compute tensors.

### 2.4 Execution model (D5, *proposed*)

- `graph_compute(split)`:
  1. Re-check every node before any write: the same predicate as `supports_op`, data pointers non-null, alignment, workspace size (Design §7.3). Any failure returns `GGML_STATUS_FAILED` with nothing written.
  2. One launch of 8 tiles. Each tile walks the nodes; a barrier follows each node.
  3. A tile that fails sets a shared flag; tiles stop at the next barrier; the result is `GGML_STATUS_FAILED`. No exception crosses the runtime (spert has no try/catch around tiles). No fallback after a write (Design §3.3, §12.3).
- Executor interface (`flagos-spacemit-exec.h`): launch N tiles, barrier, per-tile `{ith, nth, tcm, tcm_size, workspace}`.
  - Implementation 1: spine-runtime (`spert::Stream`, `Grid`, `Context::sync`, `shared_buffer`). It also does the AI-thread opt-in and cross-process core arbitration, and FlagTree kernels (M3) need it anyway.
  - Implementation 2, only if M0 shows spert problems: own pinned thread pool + libspine_tcm, the upstream IME approach that now works on this board.
- Stream lifetime: one per backend (lower latency, holds the A100 cores) versus one per call (ggml-spacemit, commit `bd85a3c`). Measure in M2a.
- Cores: 8. An environment variable may override it for experiments only (Design §15.6 gate 9).
- TCM hygiene: release on every path. At probe, a read-only check warns (`GGML_LOG_WARN`) when blocks are held by dead threads and names `tcmrelease`.
- Never enable `GGML_CPU_RISCV64_SPACEMIT` in the same build: two TCM users in one process.
- Attention under ACCEL: with `-fa auto` (llama.cpp's default) flash attention is switched off whenever the provider runs attention, because each layer's device is the CPU (`src/llama-context.cpp:506-560`; confirmed in E1). The default path is therefore non-flash attention (`MUL_MAT` on F16/F32 KV, masked `SOFT_MAX`, `CONT`); it must be correct and fast (M2e). `-fa on` keeps flash attention. llama.cpp's own TODO at that check says the rule is wrong for other cases too; a fix that accepts an ACCEL backend running a CPU layer's attention is an upstream candidate (issue first).

### 2.5 Kernel sources (D6, *proposed*)

| Ops | Source | Milestone |
|---|---|---|
| quantized `MUL_MAT` | in-tree `ime1_kernels.cpp`, `ime2_kernels.cpp`, `repack.cpp`, RVV activation quantizers, compiled into the provider from their current location (no copy); tiling and TCM staging ported from ggml-spacemit `ime.cpp` (prefill path A, decode path B, no-TCM path C, direct GEMV for Q4_0) | M2b, M2c |
| `MUL_MAT_ID` | same, MoE kernels | M2c, only for an MoE target |
| `ADD`, `MUL`, `SCALE`, `RMS_NORM`, `ROPE`, `SOFT_MAX`, `GLU`, `UNARY`, `GET_ROWS`, `SET_ROWS`, `CPY`/`CONT` | port from ggml-spacemit `rvv_kernels.cpp` (kernels take a small context struct) | M2a, M2d |
| attention: non-flash path (`MUL_MAT` F16/F32 batched, masked `SOFT_MAX`) and `FLASH_ATTN_EXT` | RVV, ported from ggml-spacemit with its X0 failures fixed (both paths fail there); the fast flash-attention variant needs VLEN 1024, which the A100 cores have (M0.4) | M2e |
| ops from FlagOS's shared Triton kernels | FlagTree SpacemiT backend, AOT package | M3 |

`repack.cpp` includes `ggml-cpu.h`; check what it calls before M2b.

### 2.6 `supports_op` rules for the K3

- Design §7 applies in full: semantic, then tensor/layout, then artifact/runtime; fail closed; dispatch re-runs the same predicates.
- Do not copy ggml-spacemit's known over-claims: Q5_0 is repacked but has no compute case; F16 `MUL_MAT` ignores `ne2`/`ne3`; `ROPE` workspace is not counted; `GATED_DELTA_NET` is always accepted but asserts `S_v <= VLMAX`; overall it checks only that a kernel trait exists. The X0 failure list (`device-type.md` §5) is the negative checklist: each failing case is either fixed or refused.
- Buffer checks follow §2.3: judge views by their source buffer; weights only in the provider's buffer; other operands in the provider's or any host buffer.
- IME layout limits (for example weight rows `% 32 == 0` for the 32-row interleave) are checked exactly, per type.
- Lossy repacks: Q4_1 and Q4_K are converted to an integer zero point, and Q6_K is requantized to Q8_0. Claim these only after the perplexity check in M2c shows an acceptable loss; until then they stay on the CPU.

### 2.7 Files

New, in `ggml/src/ggml-flagos/providers/spacemit/` (layout from Design §13.2):

| File | Content | From |
|---|---|---|
| `flagos-spacemit-api.h` | `const flagos_provider_v1 * flagos_spacemit_provider();` | M1 |
| `flagos-spacemit.cpp` | descriptor, probe, device, buffer type, backend, `graph_compute` | M1 |
| `provider.cmake` | sources, `GGML_FLAGOS_HAVE_SPACEMIT`, kernel sources and IME flags only on riscv64, spine-runtime only where found | M1 |
| `flagos-spacemit-exec.{h,cpp}` | executor | M2a **Implemented 2026-10-09** (uncommitted; Mac-tested incl. a simulated K3, board run pending): `flagos-spacemit-exec` (spine-runtime executor, one launch per split, barrier per node, failure flag read after each barrier; serial stand-in without spine-runtime), `flagos-spacemit-ops` (op table shared by `supports_op` and `graph_compute`), `flagos-spacemit-kernels` (RVV `ADD`); stream policy switch `FLAGOS_SPACEMIT_STREAM=per-call`, test-only `FLAGOS_SPACEMIT_TEST_FAIL_NODE=n`; run `spacemit_check.py --milestone m2a --build` |
| `flagos-spacemit-ops.{h,cpp}` | predicates and dispatch | M2a |
| `flagos-spacemit-weights.{h,cpp}` | repack at `set_tensor`, inverse at `get_tensor` | M2b |
| `tests/check_spacemit.cpp` | `flagos-check-spacemit` | M1 |
| `tools/spacemit_check.py` | acceptance run on the K3 per milestone (`--milestone m1` or `m2a`; was `m1_check.py`) | M1 |
| `flagos-spacemit-exec.{h,cpp}`, `flagos-spacemit-ops.{h,cpp}`, `flagos-spacemit-kernels.{h,cpp}` | executor, op table, RVV kernels | M2a |
| `flagos-spacemit-aot.{h,cpp}`, `tools/` | package loader and generator | M3 |

Common files touched (C1, required, about 10 lines): `ggml/CMakeLists.txt` (option `GGML_FLAGOS_SPACEMIT`, default OFF, next to L201-204); `ggml/src/ggml-flagos/CMakeLists.txt` (include `provider.cmake`, update the "no provider" message, add the check target); `flagos-registry.cpp` (forward declaration and `push_back` under `#ifdef GGML_FLAGOS_HAVE_SPACEMIT`, L13-19 and L69-78).

The skeleton compiles on the Mac (probe returns false there). Kernels compile only on riscv64 with GCC 15.

### 2.8 Device type (D3: ACCEL, decided 2026-10-08)

The mentor decided on ACCEL. This section records what the types change and what E1 measured (`device-type.md` §5). Consequences for the provider: read CPU buffers directly (§2.3); ship correct non-flash attention (§2.4); no per-layer control with `-ngl`; do not name the device with `--device` (two backend instances, see the code check below).

Measured 2026-10-08 (`device-type.md` §5, X3/X3b): with the backend reading CPU buffers, flash attention on and llama.cpp's norm pinning disabled, ACCEL matches GPU type within 1-2% (Qwen3-0.6B tg128 57.1 vs 57.4 t/s, 2 splits per token, same perplexity). Two llama.cpp heuristics tie work to the layer's device and therefore to the CPU under ACCEL: the automatic flash-attention check (`src/llama-context.cpp:506-560`, flash attention off) and norm pinning (`:2499-2520`, 224 splits per token on Qwen3-0.6B, 288 on Qwen3-4B; generation 39% / 30% slower). Fixing them is change C5 (§6), for the mentor to decide.

| | `GPU` | `ACCEL` (default device selection) | Evidence |
|---|---|---|---|
| Model layers | offloaded with `-ngl` | not selected for layer offload: absent from `model->devices` | `src/llama.cpp:220-223` |
| KV cache | in the device's buffer for offloaded layers | CPU buffers | `src/llama-kv-cache.cpp:213-217` |
| Weights | device buffer first, CPU list as fallback | device buffer first in the CPU buffer list, for weights whose op the device supports | `src/llama-model.cpp:903-917` |
| Backend created | only for model devices | always, ahead of the CPU backend; no per-run off switch | `src/llama-context.cpp:340-350` |
| User control | `-ngl`, `-ot`, `--device` | initialized automatically; explicit `--device` has the caveat below | `common/arg.cpp:1058-1077`, `src/llama-context.cpp:330-350` |
| Next to a discrete GPU | competes with it for layers | complements the CPU layers only | `src/llama.cpp:220-262` |
| Other FlagOS providers | Denglin, AMD report `GPU`/`IGPU` | ggml-spacemit, BLAS, mentor's ARM reference use `ACCEL` | `flagos-denglin.cpp`, `flagos-amd.cpp`; mentor design line 713 |

Code check (2026-10-07): `--device` accepts an ACCEL device (`common/arg.cpp:1070-1073`), and explicit device selection adds it to `model->devices` (`src/llama.cpp:155-178`). The context then initializes it in both the model-device loop and the unconditional ACCEL loop (`src/llama-context.cpp:330-350`). This creates two backend instances for the same device; the runtime effect has not been tested. The table describes automatic selection. ACCEL does not restrict execution to matrix multiplication: actual placement depends on `supports_op`, `supports_buft`, and backend priority (`ggml/src/ggml-backend.cpp:883-889`).

E1 (M0.9) results are in `device-type.md` §5: IGPU behaves exactly like GPU on this board; ggml-spacemit as ACCEL needed 280 graph splits per generated token against 2 as GPU, and generated 2.4-2.8x slower; its wrong ACCEL output was a backend bug triggered by the automatic flash-attention rule (ACCEL with `-fa on` matches GPU's perplexity exactly). X3 measures ACCEL when the backend reads CPU buffers, the configuration this provider will use.

`IGPU` is a third existing candidate. ggml defines it as "integrated GPU device using host memory" (`ggml/include/ggml-backend.h:134-145`), which describes the K3 better than "GPU device using dedicated memory". llama.cpp treats it like `GPU` (layers, KV cache, `-ngl`) except that it drops integrated GPUs when a discrete GPU is present (`src/llama.cpp:272-276`); that is why the v0.3 resolution chose `GPU`. Measuring it needs one more value in the E1 switch.

A new ggml device type is possible (upstream added `IGPU` in PR #15797 and `META` in PR #19378) but costly: a ggml API change, deliberate handling in about 16 files that branch on device type (device selection, weight placement, backend creation, `common/fit.cpp`, tools, several backends), and a fork-only divergence unless upstream accepts it. It contradicts the design PDF's aim that ordinary providers need no ggml or scheduler change (§4.3). Consider it only if measurements show placement behavior that none of `GPU`, `IGPU`, `ACCEL` can express, and then raise it as an upstream issue first.

## 3. Decisions

| ID | Decision | Status | If rejected |
|---|---|---|---|
| D1 | Full ggml device/buffer/backend via FlagOS, not a ggml-cpu extra buffer type | adopted | - |
| D2 | In-tree provider in the mentor's fork | adopted | - |
| D3 | ggml device type `ACCEL` | **adopted** (mentor, 2026-10-08) | - |
| D4 | Weights in an IME-repacked buffer, stored once (A2); the provider reads CPU buffers for all other operands (§2.3) | *proposed*, re-evaluated for ACCEL | A1: host buffer plus a private repacked copy (twice the quantized-weight memory) |
| D5 | spine-runtime executor, one launch per split, barrier per node; decided after M0 | *proposed* | own pinned pool + libspine_tcm behind the same executor interface |
| D6 | Reuse in-tree IME/repack kernels in place; port ggml-spacemit tiling and RVV ops, fixing them as needed (the mentor allows changing that AI-generated code, 2026-10-08) | *proposed* | copy the files into `providers/spacemit/` (duplication) |
| D7 | First target Qwen3-0.6B / Qwen3-4B Q4_0, then the AMD target Qwen3.5-4B Q4_K_M | *proposed* | start with Qwen3.5: needs Q4_K/Q6_K (lossy repack) and `GATED_DELTA_NET`/`SSM_CONV` early |
| D8 | FlagOS `caps.kind = cpu_accelerator` (engine `cpu`): the A100 cores execute RISC-V code with RVV/IME. Independent of D3 (a provider can report ggml `GPU` and FlagOS `cpu_accelerator`) | *proposed*, open for the mentor | `ai_accelerator` makes FlagOS classify the K3 as an NPU (engine `npu`), matching the v0.3 "AI chip provider" wording. The kind changes no execution: only validation, the default profile labels and engine-filtered kernel variants (`flagos-target.cpp:87-89`), so there is nothing to benchmark |
| D9 | FlagTree AOT (M3) only if the mentor puts it in scope | *proposed* | hand-written kernels only |

## 4. Milestones

Each milestone is a separate, reviewable change, roughly under 1000 new lines excluding reused kernel files. Sizes are estimates.

### M0 - Targets and baselines (Design Phase 0, no provider code)

Design Phase 0 output: target descriptor, model and shapes, op matrix, fallback, CPU and native-runtime baselines.

| Step | Build or do | Result |
|---|---|---|
| M0.1 | Build ggml-spacemit with spine-runtime (`build.md` §5). Current SpacemiT CI uses branch `mtmd-backend` @ `64316cd` with spine-runtime 0.6.3; `agent/ggml-spacemit-backend` @ `4e782bc` with 0.6.0 also works. Record which | native-runtime baseline |
| M0.2 | Run it with `SPINE_TCM_RUNTIME_LOG=true`, `tcmrelease` first | does `shared_buffer()` get real TCM or the silent heap fallback? launch overhead of an empty graph |
| M0.3 | Download Qwen3-0.6B Q4_0 and Qwen3-4B Q4_0 (plus Qwen3.5-4B Q4_K_M if D7 changes) | models in `~/models` |
| M0.4 | Measure A100 VLEN correctly: in a fresh thread, opt in before any vector instruction, then read `vlenb`; also read `BackendInfo::vlen` (bytes) inside a spert tile | **done 2026-10-07: A100 1024, X100 256**; M0.2 also done: `shared_buffer()` is real TCM (`k3-hardware.md` §2, §4) |
| M0.5 | Baselines, interleaved, 5 runs each: plain CPU (`build/`), upstream IME + TCM (`build-ime/`), ggml-spacemit. Settings as SpacemiT's table: `-t 8 -p 128 -n 128 -ub 128 -fa 1 -mmp 0` | baseline table, mean and median |
| M0.6 | Perplexity of each baseline on a fixed text (`~/ppl.txt`, 8 x 512 tokens) | accuracy reference (the IME repacks are partly lossy). **M0.5 and M0.6 done 2026-10-08** on Qwen3-4B Q4_0: `build.md` §7 |
| M0.7 | Op inventory of one decode step and one 128-token prefill (`llama-eval-callback`; `GGML_SCHED_DEBUG=2` on the ggml-spacemit build to see its splits) | op matrix: op, types, shapes, count per token, which source, which milestone  **Done 2026-10-08**, below. |
| M0.8 | Settle D4-D9 with the mentor (D3 settled: ACCEL, 2026-10-08) | PROJECT.md decisions updated |
| M0.9 | E1: build ggml-spacemit with `scripts/patch-spacemit.py` (device-type and buffer-policy switches) and run `scripts/e1-device-type.sh` on Qwen3-0.6B and Qwen3-4B Q4_0; experiments X0-X6 in `device-type.md` | X0-X2, X4, X5 done on Qwen3-0.6B (`device-type.md` §5). X3/X3b done 2026-10-08: ACCEL with CPU-buffer reads, `-fa on` and norm pinning off equals GPU type within 1-2%; the remaining gaps are two llama.cpp heuristics (C5). Then Qwen3-4B |

Scripts for M0 are in `docs/flagos-k3/scripts/`: `spacemit-device-type.patch` and `e1-device-type.sh` (E1), `spert-info.cpp` (M0.2, M0.4: spine-runtime info, worker VLEN, whether `shared_buffer()` is inside a `/dev/tcm` mapping), `vlen-probe.c` (M0.4), `m0-baselines.sh` (M0.5, M0.6), `bench-summary.py`.

Exit: baseline table in `build.md` §7, op matrix appended to this file, decisions recorded. Commands for M0.5:
```bash
mkdir -p ~/bench; M=$HOME/models/Qwen3-4B-Q4_0.gguf
A="-m $M -t 8 -p 128 -n 128 -ub 128 -fa 1 -mmp 0 -r 1 -o csv"     # csv: header + one pp row + one tg row
for i in 1 2 3 4 5; do
  ./build/bin/llama-bench $A | tail -n +2 >> ~/bench/cpu.csv
  ~/tcmtest/tcmrelease --apply >/dev/null; ./build-ime/bin/llama-bench $A | tail -n +2 >> ~/bench/ime-tcm.csv
  LD_LIBRARY_PATH=$HOME/spine-runtime/lib ~/llama.cpp-spacemit/build/bin/llama-bench $A | tail -n +2 >> ~/bench/spacemit.csv
done
```

### M0.7 result: op matrix, Qwen3-4B Q4_0 (2026-10-08)

Source: `llama-eval-callback` dumps of one generation step (1 token) and one prefill (131 tokens), flash attention on and off, from our fork's CPU build; `scripts/op-inventory.py` joins them with the GGUF tensor table (`~/m07/inventory.txt` on the board). Model: 36 layers, hidden 2560, 32 query / 8 KV heads of 128, FFN 9728, vocabulary 151936, tied embedding/output. Per graph 834 compute ops (36 x 23 + 6) and 432 views; generation and prefill have the same op list, only `n_tokens` differs.

| Op | Count | Weights read per token | Milestone |
|---|---|---|---|
| `MUL_MAT` Q4_0 | 248 | 1895.6 MiB (84%) | M2b |
| `MUL_MAT` Q6_K (output head, tied to `token_embd`) | 1 | 304.3 MiB (13.5%) | M2c |
| `MUL_MAT` Q4_1 (`ffn_down` in 4 layers) | 4 | 59.4 MiB (2.6%) | M2c |
| `GET_ROWS` Q6_K (token embedding) | 1 | one row | stays on CPU (first split) |
| `RMS_NORM`, `MUL` (norm weight) | 145, 145 | - | M2a/M2d |
| `ADD` (residuals) | 72 | - | M2a/M2d |
| `ROPE` | 72 | - | M2d |
| `SET_ROWS` (KV write, F16) | 72 | - | M2d |
| `SWIGLU` | 36 | - | M2d |
| `GET_ROWS` F32 (output-token rows, last layer) | 2 | - | M2d |
| `FLASH_ATTN_EXT` (`-fa on`) | 36 | - | M2e |
| instead with `-fa off`: `MUL_MAT` F16 x F32, masked `SOFT_MAX`, `CONT` | 72, 36, 36 | - | M2e |

One layer, in order (23 compute ops, `-fa on`): `RMS_NORM`, `MUL` (attn norm); `MUL_MAT` Q; `RMS_NORM`, `MUL`, `ROPE` (Q norm and rotary); `MUL_MAT` V, `MUL_MAT` K; `RMS_NORM`, `MUL`, `ROPE` (K); `SET_ROWS` K, `SET_ROWS` V (into `cache_k_lN`/`cache_v_lN`, CPU-resident under ACCEL); `FLASH_ATTN_EXT`; `MUL_MAT` attention output; `ADD` (residual); `RMS_NORM`, `MUL` (FFN norm); `MUL_MAT` gate, `MUL_MAT` up; `SWIGLU`; `MUL_MAT` down; `ADD` (residual). The GGUF accounting is byte-exact (398 tensors: 248 Q4_0, 145 F32 norm weights, 4 Q4_1, 1 Q6_K).

Consequences:
- Q4_0 matmuls carry 84% of the bytes but are 248 of 834 ops; one split per token needs all 9 op kinds of the flash-attention path (or 11 kinds with `-fa off`).
- For this file M2c is Q6_K and Q4_1 (both converted lossily to the IME layout; the upstream IME path that does this measured perplexity 9.7680 vs 9.7740 on the CPU, `build.md` §7), not Q8_0/Q4_K.
- The `-fa off` path is exactly the set that ggml-spacemit gets wrong in X0 (F16 `MUL_MAT`, masked `SOFT_MAX`, `CONT`); under ACCEL it is the default path until C5.
- Every kernel needs a one-token (generation) and a batched (prefill) path.

### M1 - Skeleton (Design Phase 1)

**Done 2026-10-08** (commit `4389005`): `m1_check.py --build` (now `spacemit_check.py --milestone m1`) on the K3 passes 14/14. Device found (8 A100 cores, ACCEL, VLEN 1024, 31.3 GiB shared RAM), absent with `FLAGOS_SPACEMIT_DISABLE=1`; 0 of 19577 `test-backend-ops` cases claimed; Qwen3-0.6B output identical with the provider enabled and disabled, no provider buffer in use, 1 graph split in both. FlagOS's own check tools still pass.

Build: C1 wiring; descriptor, probe (with a test-only switch `FLAGOS_SPACEMIT_DISABLE=1` that makes it return false), ACCEL device, buffer type, backend whose `supports_op` claims nothing and whose `graph_compute` returns `GGML_STATUS_FAILED` for any compute node; profile; `flagos-check-spacemit`. Files:

| File | Role |
|---|---|
| `ggml/src/ggml-flagos/providers/spacemit/flagos-spacemit-api.h` | declares `flagos_spacemit_provider()` for the registry |
| `.../providers/spacemit/flagos-spacemit.cpp` | the provider: probe, device, buffer type, backend, FlagOS descriptor |
| `.../providers/spacemit/provider.cmake` | adds the sources and `GGML_FLAGOS_HAVE_SPACEMIT` |
| `.../providers/spacemit/tests/check_spacemit.cpp` | `flagos-check-spacemit`: registry, buffer and backend checks through the built library |
| `.../providers/spacemit/tools/spacemit_check.py` | the acceptance run on the K3 (build, checks, device list, op support, model comparison); `--milestone m1` |
| `ggml/CMakeLists.txt`, `ggml/src/ggml-flagos/CMakeLists.txt`, `flagos-registry.cpp` | C1 wiring: option `GGML_FLAGOS_SPACEMIT` (default OFF), include, check target, registry entry |

Test (K3), in a separate build directory so `build/` stays the plain-CPU baseline:
```bash
python3 ggml/src/ggml-flagos/providers/spacemit/tools/spacemit_check.py --milestone m1 --build   # builds build-flagos/, then all checks
python3 ggml/src/ggml-flagos/providers/spacemit/tools/spacemit_check.py --milestone m1           # checks only, after a rebuild
```
Exit: registry lists exactly one FlagOS device on the K3 and none on the Mac; all check tools pass; nothing is claimed, so nothing lands in the provider's buffers and the output equals the run with the provider disabled. With the option OFF, nothing changes.

### M2 - Direct ops (Design Phase 2)

In the table, every `test-backend-ops` run uses `-b FlagOS:SpacemiT:0`; model runs use default device selection (ACCEL: no `-ngl`, no `--device`), with `tcmrelease` before each. Attention is checked with the default `-fa auto` and with `-fa on`.

Design's order is simple ops first, then quantized matmul (§13.3). The K3 starts quantized `MUL_MAT` earlier because it is the reason this provider exists, and partial coverage is safe: under ACCEL the CPU backend runs every op the provider does not claim, and a weight only moves to the provider's buffer when its op is claimed. Each sub-step delivers the full per-op set from Design §13.3: supports contract, dispatch, CPU comparison, stride/tail/alignment cases, trace.

| Step | Build | Test | Exit |
|---|---|---|---|
| M2a | executor (spert), predicates/dispatch framework, error flag, TCM check at probe; first op `ADD` (F32, same shape) | `test-backend-ops -o ADD -b FlagOS:SpacemiT:0`; a forced kernel failure returns `GGML_STATUS_FAILED`; after the run `tcmtest info` shows 0 blocks held | `ADD` matches CPU; empty-launch cost and per-node barrier cost measured; stream-lifetime choice made |
| M2b | quantized `MUL_MAT` for Q4_0: repacking weight buffer, decode and prefill paths, TCM staging, direct GEMV | `test-backend-ops -o MUL_MAT`; perplexity vs M0.6; `llama-bench -m <model> -t 8 ...`; `graph splits` in the load log (`-lv 4`) | correct; perplexity within the M0.6 range; tg and pp recorded (expect below IME baseline: every other op still runs on X100, with a split switch around each matmul) |
| M2c | the target's other weight types: for Qwen3-4B Q4_0 these are Q6_K (output head) and Q4_1 (`ffn_down` in 4 layers), see the op matrix; Q8_0/Q4_K for later targets; `MUL_MAT_ID` for an MoE target | per type: `test-backend-ops`, perplexity | each claimed type within accuracy budget; lossy types decided with evidence |
| M2d | the remaining ops of a layer; for Qwen3-4B (op matrix): `RMS_NORM`, `MUL`, `ADD`, `ROPE`, `SET_ROWS` (KV write into the CPU-resident cache), `SWIGLU`, `GET_ROWS` F32, view ops; others (`SCALE`, `UNARY`, `CPY`) as later targets need them | per op `test-backend-ops`; split count | a decode graph has at most a few splits; tg at least the upstream IME baseline |
| M2e | attention, reading the CPU-resident KV cache directly: the non-flash path (`MUL_MAT` F16/F32 batched, masked `SOFT_MAX`, `CONT`) that the default `-fa auto` uses, and `FLASH_ATTN_EXT` for `-fa on` | `test-backend-ops -o MUL_MAT`, `-o SOFT_MAX`, `-o FLASH_ATTN_EXT`; perplexity with `-fa auto` and `-fa on`; long-context run | default flags give correct output; whole layers on the K3 (a few splits per token); **tg at least the upstream IME path's and pp at least ggml-spacemit's best** (Qwen3-4B Q4_0: tg128 11.10, pp128 81.94, `build.md` §7; Qwen3-0.6B: 57.3 / 594 t/s) |
| M2f | memory and weight round trip: THP for compute buffers | VmRSS about model + KV + compute (weights stored once); `get_tensor` returns the original bytes for every claimed type | no double storage |

### M3 - AOT package (Design Phase 3, only if D9 is accepted)

- Generator `tools/generate_flagos_spacemit_kernels.py`, modelled on the AMD generator: compile FlagOS's shared Triton kernels (`ggml/src/ggml-flagos/kernels/generate_flagos_kernels.py`) with `triton.compile(ASTSource(...), target=AICPUTarget(...))` and take `asm["so"]`. Cross-compile on x86 with SpacemiT's toolchain; FlagTree's documented flow is x86 + Docker.
- Manifest (Design §9.4 envelope): ABI `flagos-spacemit-aot-v1`, backend spert, format riscv64 ELF `.so`, arch id `0xA064`, `num_threads` (baked in at compile time), per-kernel symbol, argument signature, size, sha256, allowed undefined symbols, provenance.
- Strict loader: ELF machine check, exactly one exported entry, undefined symbols within the allowlist, hash, fail closed (Design §9.6). `libspert` must be loaded with `RTLD_GLOBAL` first, because kernel `.so` files do not declare it.
- Typed launcher for FlagTree's ABI: `fn(spert::Context*, {rank, memref desc} per pointer, scalars, gx, gy, gz)`, on the provider's existing stream.
- Start with one non-critical op (for example `RMS_NORM`); keep it only if it matches or beats the hand-written RVV kernel.
- Exit: Design §15.3 package/ABI negative matrix passes (wrong hash, missing or extra kernel, wrong arch, wrong thread count); one AOT op in the default path behind an exact profile.

### M4 - Fixed fusion (Design Phase 4, only from measured hotspots)

- Profile a decode step (per-node time) after M2e. Candidates from Common's catalog: `rms_norm_mul` (1), `add_rms_norm_mul` (2), `ffn_swiglu` (7), `rope_kv_store` (8), `flash_attn_decode` (10).
- Implement `query_lowering` and `execute_fusion` (`flagos-graph-plan.h:124-135`); every member op already has a direct path from M2d.
- Gate (Design §15.6): stable local gain, positive end-to-end, exact shape guards, required-output, alias and state tests.
- K3-specific fusions such as dequant + GEMM + epilogue tiled in TCM do not fit Common's fixed catalog (`upstream-review.md` A1). Raise them with the mentor instead of editing Common.

### Not planned

- Design Phase 5 (split compiler): the K3 has no vendor graph compiler. One launch per split is an executor detail, not a compiler.
- Design Phase 6 (pre-placement partition): every K3 op can run on its own; no fused-only constraint.

## 5. Verification and acceptance

| Area (Design §15) | K3 check | From |
|---|---|---|
| Registry, lifecycle | `flagos-check-*`, `flagos-check-spacemit`; repeated model load/free | M1 |
| `supports_op` negative matrix | `test-backend-ops support`: every unclaimed shape/type reported unsupported, never wrong | M2 |
| Numerics | `test-backend-ops test -b FlagOS:SpacemiT:0 -o <op>`; perplexity vs M0.6, with the default `-fa auto` and with `-fa on` | M2 |
| Memory | VmRSS; weights stored once; `get_tensor` round trip | M2f |
| State and alias | KV write (`SET_ROWS`) and in-place ops with views; context shift | M2d-M2e |
| Fail closed | forced kernel failure returns `GGML_STATUS_FAILED`; no partial writes before launch checks | M2a |
| TCM hygiene | 0 blocks held after every run, including failures | M2a |
| Performance | interleaved A/B vs the three M0 baselines; pp128, tg128; split count; first run vs steady state | M2b onward |
| Package/ABI | Design §15.3 matrix | M3 |
| Stability | long decode (thousands of tokens), repeated sessions | M2e |

Design §15.6 gates, as they apply to the K3: no `supports_op` false positives (1); the target model runs entirely through direct paths plus CPU fallback (2, 3); KV/state writes tested (4); AOT fails closed (5, M3 only); full-model correctness (7); interleaved pp/tg A/B (8); default path needs no development environment variables (9); errors never reported as success (11); lifecycle stress (12).

## 6. Risks

| ID | Risk | Mitigation |
|---|---|---|
| R1 | Resolved 2026-10-07: A100 VLEN is 1024, X100 256, and a thread's `vlenb` changes when it migrates. Remaining risk: code that reads VLEN on the main thread (X100) and runs on A100 workers picks the wrong path; ggml-cpu does this (`ggml-cpu.c:742`, `repack.cpp:4591-4712`) | provider kernels read VLEN on the worker thread; profile reports 1024 |
| R2 | spine-runtime is a closed binary: `shared_buffer()` falls back to heap silently if TCM is unavailable (on this board it gets real TCM, 2026-10-07), all-or-nothing core grants, no exception handling around tiles | log a warning when the buffer is outside `/dev/tcm`; executor interface keeps option 2 open |
| R3 | Stale TCM blocks after a crash block all later runs | probe-time warning; release on every path; `tcmrelease` documented |
| R4 | Lossy repacks (Q4_1, Q4_K, Q6_K) change results | claim only with perplexity evidence (M2c) |
| R5 | Under ACCEL, `-fa auto` switches flash attention off when the provider runs attention, so default runs use non-flash attention | correct non-flash kernels (M2e); document `-fa on`; upstream fix candidate (§2.4) |
| R6 | Under ACCEL, a provider op that refuses CPU buffers forces copies and splits (E1: 280 splits per generated token). Also many splits until M2d, while each matmul is surrounded by CPU ops | accept host buffers for non-weight operands (§2.3); X3 measures it; split count in every benchmark |
| R7 | ggml-spacemit is AI-generated and over-claims (X0: wrong F16 matmul, masked softmax, CPY; aborts in RMS_NORM, ROPE) | port piece by piece with fixes and tests; the mentor allows changing it |
| R8 | FlagTree SpacemiT is immature (no AOT tool, no tests, QEMU-only CI with VLEN 1024) | M3 is conditional and starts with one op |
| R9 | Common gaps: no `get_proc_address` forwarding (thread count, abort callback), no RVV/IME feature bits, no `LOCAL_SCRATCH` cap for TCM (mentor design §8.1, not in code) | work around in the provider; propose C2-C4 only when needed |

Optional Common changes, each needing separate approval: C2 RVV/IME/TCM feature bits in `flagos-target.h` (append-only); C3 forward provider functions through `get_proc_address`; C4 `LOCAL_SCRATCH` memory cap.

Outside FlagOS, needing the mentor's decision (fork patch or upstream llama.cpp issue): C5, in `src/llama-context.cpp`, (a) let the automatic flash-attention check accept an ACCEL backend that supports the op when the layer's device is the CPU (`:506-560`); (b) skip norm pinning when the layer's device is the CPU (`:2499-2520`, marked FIXME upstream). Without C5, ACCEL users need `-fa on` and get 30-39% slower generation (Qwen3-4B / 0.6B) (`device-type.md` §5, X3b).

## 7. Questions for the mentor

1. ~~Device type~~ decided: ACCEL (2026-10-08). Still open: FlagOS `caps.kind`, `cpu_accelerator` (proposed) or `ai_accelerator`?
2. Memory under ACCEL: weights in an IME-repacked buffer, stored once, with the provider reading CPU buffers for everything else (A2, §2.3). Agreed? The alternative (A1) doubles quantized-weight memory.
3. Is spine-runtime (closed `libspert`) acceptable as a hard dependency of the provider?
4. Kernel reuse: compile the in-tree `ggml-cpu/spacemit` kernel sources into the provider in place, or copy them into `providers/spacemit/`?
5. Target for Phase 0: Qwen3-4B Q4_0 (SpacemiT has reference numbers) or Qwen3.5-4B Q4_K_M (comparable with AMD, but needs lossy Q4_K/Q6_K repacks and gated-delta-net ops early)?
6. Is FlagTree AOT (M3) in scope, and should it be cross-compiled on x86 or built on the K3?
7. Is co-building in `providers/spacemit` agreed with SpacemiT, given they ship ggml-spacemit as an `ACCEL` backend?
8. Should we report the TCM issues to SpacemiT (no dead-owner recovery; `try_wait` timeout unit about 12 µs, not 1 µs)?
9. ACCEL gaps in llama.cpp (C5, §6): patch the automatic flash-attention check and norm pinning in our fork, or raise an upstream llama.cpp issue? Without them ACCEL needs `-fa on` and generates 30-39% slower (`device-type.md` §5, X3b).
