# SpacemiT K3 provider: build plan

Written 2026-10-07; updated as milestones land (M1 done 2026-10-08, M2a and M2b done 2026-10-09, M2d and M2c done 2026-10-10).

- "Design" = `ggml-flagos-provider-design.pdf` V2.0 (review baseline, 2026-09-22). Section numbers like "Design §13.3" refer to it.
- "Mentor design" = `FLAGOS_BACKEND_MULTIPLATFORM_DESIGN.md` and `FLAGOS_BACKEND_MULTIPLATFORM_DESIGN_REVIEW_RESOLUTION.md` on `kevin/feature/flagos-multi-provider-backend` (v0.3, 2026-08-22). Read with `git show kevin/feature/flagos-multi-provider-backend:<file>`.
- Decisions marked *proposed* need the mentor's confirmation (§7). Facts marked UNVERIFIED have not been checked on the board or in code.

## 0. Summary

1. Build a **direct-op provider** (Design §4.1) at `ggml/src/ggml-flagos/providers/spacemit/`, compiled into `libggml-flagos`, that runs ggml ops on the 8 A100 AI cores (CPUs 8-15).
2. **Device type: ggml `ACCEL`** (mentor decision, 2026-10-08). llama.cpp then keeps model layers and the KV cache on the CPU side, tries the provider's buffer type first for each weight whose op the provider supports, and always creates the provider's backend ahead of the CPU backend (§2.8). Whole layers reach the AI cores only if the provider also reads CPU buffers directly: in E1, a backend that did not needed 280 backend switches per generated token.
3. **Memory**: the provider's buffer type holds weights in IME-repacked layout (`is_host = false`), stored once. The KV cache and other inputs stay in CPU buffers, which the provider reads directly (§2.3, D4: building on this from M2b, 2026-10-09; mentor confirmation pending).
4. Execution: **one spine-runtime launch per split**. Each of the 8 tiles walks the split's steps (one or more per node), with a barrier after each step and the core's TCM as scratch. This is ggml-spacemit's model, except that its kernels also wait inside a node.
5. Kernels: copy into the provider, milestone by milestone, only the IME GEMM, quantizer and repack functions it uses from the in-tree `ggml-cpu/spacemit` sources (byte-identical to ggml-spacemit's), and port ggml-spacemit's tiling and RVV ops, each behind its own op tests (D6, decided 2026-10-09, §2.5). The mentor allows changing ggml-spacemit's code freely (it is AI-generated); its X0 failures must be fixed, not copied. Neither source is validated yet: the upstream 1198/1198 `MUL_MAT` run never used the IME weight layout (`test-backend-ops` allocates in the CPU backend's plain buffer), `test-backend-ops` has no Q4_0 case that fits the IME 32x256 layout at all (§1), and ggml-spacemit fails part of the op tests on this board (`device-type.md` X0). FlagTree AOT kernels come later and only if the mentor wants them.
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
| ggml-spacemit: `ACCEL`, one spert launch per `graph_compute`, a grid barrier after every node, whole per-core TCM per tile, IME only for `MUL_MAT`/`MUL_MAT_ID` on repacked weights; its IME and repack sources are byte-identical to ours. It copied the in-tree files into its own directory: `ime1/ime2_kernels.cpp`, `ime_kernels.h`, `repack.{h,cpp}`, `spine_barrier.h` unchanged; `ime.cpp` (+586/-396) and `rvv_kernels.cpp` (+1222/-119) rewritten. Its build still includes ggml-cpu headers and calls ggml-cpu functions (`ggml_get_type_traits_cpu`, `ggml_vec_*`) without linking ggml-cpu, and forces `GGML_CPU_RISCV64_SPACEMIT` off. Branch `mtmd-backend` @ `64316cd` (2026-09-24, used by SpacemiT's CI) has the same kernel, IME, RVV and repack files; it only adds core pinning (`SPACEMIT_PERFER_CORE_ID`) | `spacemit-com/llama.cpp` @ `4e782bc`: `ggml/src/ggml-spacemit/ggml-spacemit.cpp:423-515, 590-593`, `CMakeLists.txt`; compared 2026-10-09 |
| `test-backend-ops` has no Q4_0 `MUL_MAT` case that fits the IME 32x256 layout (row length % 256, rows % 32): the Q4_0 cases have 16 rows, or 2880 x 2880 (2880 % 256 != 0); the fused cases need ops the provider does not claim | `tests/test-backend-ops.cpp:8875, 8897-8968, 6082-6200` |
| llama.cpp places a weight once, at load, by asking `supports_op` about a 512-row `MUL_MAT` on it; it writes each weight whole (`ggml_backend_tensor_set(cur, data, 0, n_size)`) unless the device has async, host-buffer and event support | `src/llama-model-loader.cpp:907-947, 1447-1475, 1569, 1640` |
| If no backend accepts an op on a weight in a buffer, the scheduler copies the weight to another backend (through the buffer's `get_tensor`) | `ggml/src/ggml-backend.cpp:877-895` |
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

### 2.3 Memory model under ACCEL (D4: A2, building from M2b, 2026-10-09; mentor confirmation pending)

Under ACCEL, llama.cpp puts a weight into the provider's buffer type only when the provider supports the op that uses it (`src/llama-model.cpp:903-917`), and the KV cache always goes to CPU buffers (`src/llama-kv-cache.cpp:211-217`). The scheduler also allocates the provider's compute tensors from its default buffer type.

| Option | How | For | Against |
|---|---|---|---|
| **A2 (proposed)** | Default buffer type holds weights in IME-repacked layout, `is_host = false` (like ggml-spacemit and ggml-cpu's repack buffers). The provider accepts CPU buffers for every non-weight operand | weights stored once; the CPU never needs provider weights, since a weight only lands there when the provider claims its op | provider outputs read by CPU ops are copied (one `memcpy` per split boundary); `get_tensor` must undo the repack |
| A1 (earlier proposal) | Default buffer = plain host memory plus a private repacked copy of each quantized weight | the CPU can read every provider buffer | quantized weights stored twice (Qwen3-4B Q4_0: +2.2 GiB); its main benefit, CPU fallback on a provider-owned KV cache, does not exist under ACCEL |
| A3 | Host default buffer plus a second, repacked buffer type | 1x memory and host-readable activations | llama.cpp uses only an ACCEL device's default buffer type; a second one would need llama.cpp and Common changes (C3) |

A2 rules:
- Buffer type `FlagOS:SpacemiT`: 64-byte alignment, size-0 allocations allowed (llama.cpp probes with one).
- `supports_buft`: own buffer type, or any host buffer type.
- `supports_op` judges a tensor by `t->view_src ? t->view_src->buffer : t->buffer`. ggml-spacemit checks only `t->buffer`; views have none before allocation, so it claimed attention on CPU-resident KV views, which started the E1 failure chain (`device-type.md` §5). A weight operand that the kernel reads in repacked layout must be in the provider's buffer (or not yet allocated, at placement time), never in a host buffer; every other operand may be in the provider's buffer or any host buffer.
- Claims for weight ops must not depend on batch size: llama.cpp picks each weight's buffer once, at load time. If a later graph's op were refused, the scheduler would copy the repacked weight for the CPU.
- Whether a tensor is stored repacked is a pure function of its type and shape (one layout function, used by `supports_op`, `set_tensor`, `get_tensor` and the kernel), not of the buffer's usage flag: `test-backend-ops` allocates weights in an ordinary buffer (it sets `WEIGHTS` usage only for its separate weight context, `tests/test-backend-ops.cpp:1384-1393`). No `init_tensor` state is needed. (Corrected 2026-10-09; the earlier rule "repack only in weight buffers" would have left test weights unrepacked.)
- `set_tensor` repacks whole-tensor writes; partial access to a repacked tensor goes through a temporary copy of the whole tensor (llama.cpp writes weights whole on this device, §1; changed during M2b from aborting). `get_tensor` returns ggml's layout (lossless layouts: the original bytes; lossy ones: the converted tensor, next rule); this is needed for correctness, not only for the memory check in M2f: if the provider ever refuses an op on a weight it holds, the scheduler copies the weight out through `get_tensor` (§1). The in-tree buffer has no `get_tensor` (`ggml-cpu/spacemit/ime.cpp:1468`). `cpy_tensor` returns false for repacked tensors, so ggml falls back to `set_tensor` instead of copying raw bytes.
- Lossy repacks (Q4_1, Q4_K, Q6_K) cannot be inverted. Decided 2026-10-10 (M2c design; earlier rule: keep the original bytes as well, or do not claim them): `get_tensor` returns the converted tensor in ggml's type, the weights the kernels compute with, and repacking it gives the same layout again; the weight is stored once. A partial write must cover whole blocks (it re-converts the blocks it touches), else it aborts. A layout requantized to another type (Q6_K -> Q8_0, M2c.2) cannot be read back exactly in its own type: reads return the nearest tensor of that type to the stored values, and partial writes abort. Claimed only with model-level accuracy evidence (R4). The Q4_0 repack is a byte permutation (same size, lossless).

### 2.4 Execution model (D5, *proposed*; implemented in M2a)

- `graph_compute(split)`:
  1. Re-check every node before any write: the same predicate as `supports_op`, data pointers non-null, alignment, workspace size (Design §7.3). Any failure returns `GGML_STATUS_FAILED` with nothing written.
  2. One launch of 8 tiles. Each tile walks the steps; a barrier follows each step. An op takes one or more steps (`ADD`: one; `MUL_MAT`: quantize activations, then GEMM). Kernels never wait inside a step: barriers exist only between steps, which keeps the failure protocol below valid and lets the serial stand-in (no spine-runtime) run the same step list.
  3. A tile that fails sets a shared flag; from then on every tile skips its remaining kernels but still reaches every barrier, so none waits for a tile that left (changed in M2d: stopping at the next barrier let tiles disagree, see "M2d design"); only a failed barrier ends a tile early. The result is `GGML_STATUS_FAILED`. No exception crosses the runtime (spert has no try/catch around tiles). No fallback after a write (Design §3.3, §12.3).
- Executor interface (`flagos-spacemit-exec.h`): launch N tiles, barrier, per-tile `{ith, nth, tcm, tcm_size, workspace}` (M2a built it without `workspace`; M2b adds it). Workspace: one shared scratch buffer per backend, grown before launch to the largest step of the split; an allocation failure returns `GGML_STATUS_FAILED` with nothing written. TCM: from M2b the tiles agree on one TCM size at the start of each launch, so they all choose the same kernel path (mixed paths would leave outputs uncomputed).
  - Implementation 1: spine-runtime (`spert::Stream`, `Grid`, `Context::sync`, `shared_buffer`). It also does the AI-thread opt-in and cross-process core arbitration, and FlagTree kernels (M3) need it anyway.
  - Implementation 2, only if M0 shows spert problems: own pinned thread pool + libspine_tcm, the upstream IME approach that now works on this board.
- Stream lifetime: **decided 2026-10-09 from the M2a benchmark: persistent (one stream per backend) is the default.** Measured on the K3 (`flagos-check-spacemit --bench`): launch 9.9 us persistent vs 24.3 us per call; per-node barrier 0.79 vs 0.83 us; idle AI-core load 0% in both (a held stream does not spin). Cost: the backend keeps the 8-core grant while it exists, so another spine-runtime process waits until it is freed; `FLAGOS_SPACEMIT_STREAM=per-call` remains as an experiment switch. Bandwidth for reference: 16M-float `ADD` 21.3 GB/s on the 8 AI cores vs 9.6 GB/s on 8 X100 threads.
- Cores: 8. An environment variable may override it for experiments only (Design §15.6 gate 9).
- TCM hygiene: release on every path. At probe, a read-only check warns (`GGML_LOG_WARN`) when blocks are held by dead threads and names `tcmrelease`.
- Never enable `GGML_CPU_RISCV64_SPACEMIT` in the same build: two TCM users in one process.
- Attention under ACCEL: with `-fa auto` (llama.cpp's default) flash attention is switched off whenever the provider runs attention, because each layer's device is the CPU (`src/llama-context.cpp:506-560`; confirmed in E1). The default path is therefore non-flash attention (`MUL_MAT` on F16/F32 KV, masked `SOFT_MAX`, `CONT`); it must be correct and fast (M2e). `-fa on` keeps flash attention. llama.cpp's own TODO at that check says the rule is wrong for other cases too; a fix that accepts an ACCEL backend running a CPU layer's attention is an upstream candidate (issue first).

### 2.5 Kernel sources (D6: copy per milestone, decided 2026-10-09)

| Ops | Source | Milestone |
|---|---|---|
| quantized `MUL_MAT` | IME2 GEMM kernels, activation quantizers, `memcpy1d` and repack functions copied from in-tree `ggml-cpu/spacemit` (`ime2_kernels.cpp`, `rvv_kernels.cpp`, `repack.cpp` @ `ba360ef`), only the functions each milestone uses, bodies unchanged, origin noted; tiling and TCM staging ported from ggml-spacemit `ime.cpp` @ `4e782bc`: direct GEMV (1 row; ggml-spacemit uses it for Q4_0 only, we also for Q4_1, M2c design), path A (at least 113 rows and n <= 64 m, activations staged in TCM), path C (everything else). Path B (weight slabs in TCM, core-pair barrier) deferred (M2b design). `ime1_kernels.cpp` is not needed: IME1 serves X100/A60 cores, the A100 cores use IME2 (`ime_env.cpp:259-262`) | M2b, M2c |
| `MUL_MAT_ID` | same, MoE kernels | M2c, only for an MoE target |
| `ADD`, `MUL`, `SCALE`, `RMS_NORM`, `ROPE`, `SOFT_MAX`, `GLU`, `UNARY`, `GET_ROWS`, `SET_ROWS`, `CPY`/`CONT` | port from ggml-spacemit `rvv_kernels.cpp` (kernels take a small context struct). For `CPY`/`CONT` take upstream fix `f266648fa` (2026-09-16): permuted 2-byte (F16/BF16) copies called the 32-bit transpose; ggml-spacemit and our tree still have the bug, which likely explains X0's `CONT` failures (F16/BF16 only) | M2a, M2d |
| attention: non-flash path (`MUL_MAT` F16/F32 batched, masked `SOFT_MAX`) and `FLASH_ATTN_EXT` | RVV, ported from ggml-spacemit with its X0 failures fixed (both paths fail there); the fast flash-attention variant needs VLEN 1024, which the A100 cores have (M0.4) | M2e |
| ops from FlagOS's shared Triton kernels | FlagTree SpacemiT backend, AOT package | M3 |

Why copy (D6, decided 2026-10-09; options and evidence compared in the M2b review):
- The mentor design (§14.1) treats `ggml-cpu/spacemit` as a reference for kernels, repack, affinity and TCM, "not a formal architecture boundary". Compiling it in place would make the provider depend on ggml-cpu's internal headers (`rvv_kernels.h` includes `ggml-cpu-impl.h`, and `ime2_kernels.cpp` includes `rvv_kernels.h`).
- SpacemiT made the same choice for ggml-spacemit and never needed to change the kernel or repack files.
- Copies in the provider's own namespace cannot clash with ggml-cpu's symbols, and the scalar reference functions they include (`*_ref`, plain C++, disabled with `#if 0` upstream and so untested) can run the whole path on the Mac.
- Size: about 1,000 lines for Q4_0 (M2b) instead of compiling about 10,700.
- Cost: upstream fixes are taken by hand. At each upstream merge, check `git log` of `ggml/src/ggml-cpu/spacemit/`. Status 2026-10-09: the functions M2b copies are identical in our tree, ggml-spacemit (`4e782bc`, `mtmd-backend` `64316cd`) and upstream master (2026-10-08); upstream changed this directory 3 times since our base (IME1 Q8_0 for X60, `alloc_buffer_n`, the transpose fix above), none in M2b's code. The copied code calls only ggml core functions.

### 2.6 `supports_op` rules for the K3

- Design §7 applies in full: semantic, then tensor/layout, then artifact/runtime; fail closed; dispatch re-runs the same predicates.
- Do not copy ggml-spacemit's known over-claims: Q5_0 is repacked but has no compute case; F16 `MUL_MAT` ignores `ne2`/`ne3`; `ROPE` workspace is not counted; `GATED_DELTA_NET` is always accepted but asserts `S_v <= VLMAX`; overall it checks only that a kernel trait exists. The X0 failure list (`device-type.md` §5) is the negative checklist: each failing case is either fixed or refused.
- Buffer checks follow §2.3: judge views by their source buffer; weights only in the provider's buffer; other operands in the provider's or any host buffer.
- IME layout limits (for example weight rows `% 32 == 0` for the 32-row interleave) are checked exactly, per type.
- Lossy repacks: Q4_1 and Q4_K are converted to an integer zero point, and Q6_K is requantized to Q8_0. Claim these only after the perplexity check in M2c shows an acceptable loss; until then they stay on the CPU. Q4_1 is claimed from M2c.1, accepted on the K3 2026-10-10: mean KLD against the CPU 0.0032 (512-token batches) and 0.0020 (`-ub 1`), perplexity -0.10% (`build.md` §7).

### 2.7 Files

New, in `ggml/src/ggml-flagos/providers/spacemit/` (layout from Design §13.2):

| File | Content | From |
|---|---|---|
| `flagos-spacemit-api.h` | `const flagos_provider_v1 * flagos_spacemit_provider();` | M1 |
| `flagos-spacemit.cpp` | descriptor, probe, device, buffer type, backend, `graph_compute` | M1 |
| `provider.cmake` | sources, `GGML_FLAGOS_HAVE_SPACEMIT`, kernel sources and IME flags only on riscv64, spine-runtime only where found | M1 |
| `flagos-spacemit-exec.{h,cpp}` | executor | M2a **done 2026-10-09** (commits `f78022f`, `187239e`): `flagos-spacemit-exec` (spine-runtime executor, one launch per split, barrier per node, failure flag read after each barrier; serial stand-in without spine-runtime), `flagos-spacemit-ops` (op table shared by `supports_op` and `graph_compute`), `flagos-spacemit-kernels` (RVV `ADD`); stream policy switch `FLAGOS_SPACEMIT_STREAM=per-call`, test-only `FLAGOS_SPACEMIT_TEST_FAIL_NODE=n`; run `spacemit_check.py --milestone m2a --build`. First K3 run 2026-10-09: 14/15; the failure was the provider not claiming `NONE`/view tensors, so `test-backend-ops` test mode (which asks about every tensor) ran 0 `ADD` cases; fixed by claiming them as design §7.4 allows (`187239e`); on the rerun `test-backend-ops -o ADD` passes on the K3. Also found: in a real model `ADD` stays on the CPU (1 split, no provider buffer), because under ACCEL ops without weights only move to the provider next to ops it already holds; claimed weights (M2b) are what seed placement |
| `flagos-spacemit-ops.{h,cpp}` | predicates and dispatch | M2a |
| `flagos-spacemit-weights.{h,cpp}` | layout function; repack at `set_tensor`, inverse at `get_tensor` | M2b |
| `flagos-spacemit-ime.{h,cpp}`, `flagos-spacemit-ime-kernels.cpp` | Q4_0 matmul steps and paths (ported orchestration); copied IME2 kernel, quantizers, `memcpy1d`, repack (own namespace; IME code on riscv64 only, scalar references everywhere) | M2b |
| `tests/check_spacemit.cpp` | `flagos-check-spacemit` | M1 |
| `tools/spacemit_check.py` | acceptance run on the K3 per milestone (`--milestone m1`, `m2a`, `m2b` or `m2d`, the default; was `m1_check.py`) | M1 |
| `flagos-spacemit-exec.{h,cpp}`, `flagos-spacemit-ops.{h,cpp}`, `flagos-spacemit-kernels.{h,cpp}` | executor, op table, kernel entry points and the ggml-cpu-style references (M2a's hand-written RVV `ADD` was replaced in M2d) | M2a |
| `flagos-spacemit-rvv-kernels.{h,cpp}` | RVV kernels ported from ggml-spacemit `rvv_kernels.cpp` @ `4e782bc` (ranges and changes in its header); riscv64 only | M2d |
| `flagos-spacemit-aot.{h,cpp}`, `tools/` | package loader and generator | M3 |

Common files touched (C1, required, about 10 lines): `ggml/CMakeLists.txt` (option `GGML_FLAGOS_SPACEMIT`, default OFF, next to L201-204); `ggml/src/ggml-flagos/CMakeLists.txt` (include `provider.cmake`, update the "no provider" message, add the check target); `flagos-registry.cpp` (forward declaration and `push_back` under `#ifdef GGML_FLAGOS_HAVE_SPACEMIT`, L13-19 and L69-78).

The skeleton compiles on the Mac (probe returns false there). IME and RVV kernels compile only on riscv64 with GCC 15; from M2b the scalar reference kernels compile everywhere.

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
| D4 | Weights in an IME-repacked buffer, stored once (A2); the provider reads CPU buffers for all other operands (§2.3) | *proposed*; building on it from M2b (user, 2026-10-09), mentor confirmation pending | A1: host buffer plus a private repacked copy (twice the quantized-weight memory) |
| D5 | spine-runtime executor, one launch per split, barrier per step (an op may take several steps) | *proposed*; implemented in M2a | own pinned pool + libspine_tcm behind the same executor interface |
| D6 | Copy, per milestone, only the IME kernel, quantizer and repack functions used from in-tree `ggml-cpu/spacemit` (bodies unchanged, origin noted); port ggml-spacemit tiling and RVV ops, fixing them as needed (the mentor allows changing that AI-generated code, 2026-10-08) | **decided** (user, 2026-10-09; §2.5) | compile the in-tree sources in place (depends on ggml-cpu internal headers; clashes with `GGML_CPU_RISCV64_SPACEMIT`) |
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
| M2b | quantized `MUL_MAT` for Q4_0 in the IME 32x256 layout: repacking weight buffer, two-step execution, shared workspace, direct GEMV, path A, path C (design below) | `flagos-check-spacemit` against the CPU backend at the model's shapes; perplexity vs M0.6; `llama-bench -m <model> -t 8 ...`; `graph splits` in the load log (`-lv 4`); `spacemit_check.py --milestone m2b`. Not `test-backend-ops -o MUL_MAT`: no case fits the layout (§1) | correct; perplexity within the M0.6 range; tg and pp recorded (expect below IME baseline: every other op still runs on X100, with a split switch around each matmul) |
| M2c (after M2d, decided 2026-10-10; done 2026-10-10 for the Qwen3 targets: M2c.1 Q4_1, M2c.2 Q6_K and Q8_0) | the target's other weight types: for Qwen3-4B Q4_0 these are Q4_1 (`ffn_down` in 4 layers; M2c.1, first) and Q6_K (output head; M2c.2), see the op matrix; Q8_0/Q4_K for later targets; `MUL_MAT_ID` for an MoE target (design below) | per type: `flagos-check-spacemit` against the CPU (no `test-backend-ops` case fits the IME layouts), KLD against the CPU's logits, perplexity; `spacemit_check.py --milestone m2c` | each claimed type within accuracy budget (KLD at most the IME baseline's, perplexity within 1%); lossy types decided with evidence |
| M2d (before M2c, decided 2026-10-09) | the remaining ops of a layer except attention; for Qwen3-4B: `RMS_NORM`, `MUL`, `ADD`, `ROPE`, `SET_ROWS` (KV write into the CPU-resident cache), SwiGLU, `GET_ROWS` F32; plus the norm-pinning patch C5b if accepted (design below) | per op `test-backend-ops` on the provider, all claimed cases; `flagos-check-spacemit` (KV write into a CPU buffer, a Qwen3-like layer against the CPU); split count from `sched-summary.py`; perplexity; `spacemit_check.py --milestone m2d` | every claimed case passes; about 2 handoffs per layer in generation (with C5b) and in prefill; perplexity within 1% of the CPU; tg and pp recorded (IME-level speed needs M2c and M2e) |
| M2e | attention, reading the CPU-resident KV cache directly: the non-flash path (`MUL_MAT` F16/F32 batched, masked `SOFT_MAX`, `CONT`) that the default `-fa auto` uses, and `FLASH_ATTN_EXT` for `-fa on` | `test-backend-ops -o MUL_MAT`, `-o SOFT_MAX`, `-o FLASH_ATTN_EXT`; perplexity with `-fa auto` and `-fa on`; long-context run | default flags give correct output; whole layers on the K3 (a few splits per token); **tg at least the upstream IME path's and pp at least ggml-spacemit's best** (Qwen3-4B Q4_0: tg128 11.10, pp128 81.94, `build.md` §7; Qwen3-0.6B: 57.3 / 594 t/s) |
| M2f | memory and weight round trip: THP for compute buffers | VmRSS about model + KV + compute (weights stored once); `get_tensor` returns the original bytes for every lossless layout, the converted tensor for lossy ones, and the nearest tensor of the original type for requantized ones (§2.3) | no double storage |

### M2b design (2026-10-09)

**Done 2026-10-09** (commit `6671c59`; K3 results in `build.md` §7): `flagos-spacemit-weights`, `-ime`, `-ime-kernels` (copies verified byte-identical to the in-tree ranges, except the two marked zero guards in the reference quantizer), op table with steps and workspace, executor TCM agreement, `flagos-check-spacemit --full`, `spacemit_check.py --milestone m2b`. Mac, reference kernels on a simulated device: 40 matmul cases up to 512 rows at Qwen3-4B shapes, max NMSE 2.6e-5 (bound 5e-4); claims, round trip and `MUL_MAT`->`ADD` checked; 0 build warnings. The IME code compiles only on the K3.

K3 results (2026-10-09): `spacemit_check.py --milestone m2b` passes 17/17 on Qwen3-0.6B and 15/15 on Qwen3-4B (the CPU-only perplexity run crashed once inside the runner; the disassembly rules out a llama.cpp bug: either a register was changed from outside the program or the fault address was misreported, both in the kernel or core, `build.md` §7; rerun alone: rc 0, 9.7740, identical to M0.6). The IME Q4_0 kernel matches the CPU at all tested shapes (first op-level check, R10). Perplexity 9.7617 vs 9.7740 on the CPU (-0.13%). Placement as designed: all 248 Q4_0 matmuls and the 68 residual `ADD`s next to them run on the AI cores; the CPU keeps `RMS_NORM` 145, norm-weight `MUL` 144, `ROPE` 72, `SWIGLU` 36, attention, KV writes, the Q6_K output head and the 4 Q4_1 matmuls. Speed: pp128 2.4x the X100 cores, tg128 only +5% (4B), because the 355 handoffs per token (about 0.13 ms each) cost about as much as the matmuls save; numbers and budget in `build.md` §7. Consequences:
- M2d (the CPU ops above, except attention) cuts the handoffs to about 2 round trips per layer (about 72 splits per token) and is worth more for generation than M2c; proposed order M2d, then M2c (output head about 25 ms per token on the X100 cores), then M2e.
- Path B would recover about 3 ms per 128-row gate/up matmul (path C 648 vs path A 960 GFLOP/s), about 8-10% of pp128.
- Provider outputs read by CPU ops are copied out (6 per layer: Q, K, V, `ffn_inp`, gate, up), since the buffer type is not host-readable (A2); nothing is copied in. The copies disappear as M2d keeps those consumers on the provider.
- Open: whether the handoff cost is mostly OpenMP worker wake-ups on the X100 side (test: `OMP_WAIT_POLICY=ACTIVE`; ggml's `--poll` does not apply to this OpenMP build and showed no effect).

Scope: Q4_0 `MUL_MAT` in the IME 32x256 layout only, the layout the IME baseline used. Weight 2-D, rows % 32 == 0, row length % 256 == 0; activations F32; output F32. This covers every Q4_0 matmul of Qwen3-0.6B (1024, 2048, 3072) and Qwen3-4B (1024, 2560, 4096, 9728). Q6_K and Q4_1 stay on the CPU until M2c.

Weights (`flagos-spacemit-weights`, §2.3):
- `spacemit_weight_layout(t)` returns none or `q4_0_32x256` from type and shape; it is the only place that decides.
- `set_tensor` repacks whole writes, `get_tensor` undoes the repack, `cpy_tensor` refuses repacked tensors; partial reads or writes of a repacked tensor go through a temporary copy of the whole tensor (first designed to abort). No `get_alloc_size` (same size).

Op check for `MUL_MAT`: weight layout `q4_0_32x256`, 2-D, not a view, in the provider's buffer or unallocated (never host); activations F32 and contiguous; output F32 and contiguous. It must not depend on the row count: llama.cpp asks once, at load, with 512 rows (§1).

Execution:
- The op table returns, per op, its steps and workspace size (`ADD`: one step; `MUL_MAT`: quantize activations, then GEMM); `graph_compute` expands nodes into steps; the executor's barrier between steps is the only barrier (§2.4).
- Workspace: rows x (row length / 256) x 290 bytes for the quantized activations (5.6 MB at 512 rows x 9728).
- Paths, chosen per matmul from the row count and the agreed TCM size:

| Rows | Path | Used by |
|---|---|---|
| 1 | direct GEMV: activations in TCM, weights read from DRAM in 128-column tiles (ggml-spacemit `ime.cpp:421`) | generation (tg128) |
| at least 113, and at most 64 weight rows per activation row (n <= 64 m) | path A: 4-row activation blocks staged in TCM, weights from DRAM | perplexity (512); pp128 except Qwen3-4B's FFN gate/up (n = 9728 needs at least 150 rows) |
| otherwise, or no TCM | path C: DRAM only | short prompts, the last part of a prompt, `-np`, speculative drafts; Qwen3-4B's FFN gate/up at 128 rows |

  Largest TCM need for Qwen3-4B: about 186 KB of the 384 KiB per core.
- Path B is deferred (decided 2026-10-09; not needed as of M2d, 2026-10-10: with the Q4_1 matmuls on the AI cores, pp512 matches IME, 87.19 vs 87.56, and pp128 exceeds it, 83.0 vs 79.5, without path B; `build.md` §7). It stages 32-column weight slabs in TCM and staggers the two cores of each pair with a pair barrier, so one copies while the other computes. Reasons: tg128 and perplexity do not use it (pp128 with `-ub 128` does, for Qwen3-4B's FFN gate/up: ime.cpp takes path B whenever n/m > 64, here 9728/128; those 2 of 7 matmuls, about half of the FLOPs, run path C instead; corrected 2026-10-09, measured by the `--bench` line `2560x9728 rows 128`); the in-tree version deadlocks when a pair's slab counts differ (rows not a multiple of 256 with 8 cores, or an odd core count), which ggml-spacemit rewrote (`has_pair`, shared slab list); it needs a barrier inside a kernel, which our executor avoids; its gain over path C is unmeasured (ggml-spacemit found reading Q4_0 straight from DRAM faster than staging for 1 row). The M2b benchmark times path C at 4, 16 and 64 rows; if those are clearly slow, path B follows as its own step, ported from ggml-spacemit's loop, with deadlock tests (odd core count, rows not a multiple of 256).

Kernel code (D6): copied from in-tree @ `ba360ef`: `gemm_kernel_i8i4_hp` with `_m1`, `_m4` and the scalar `_mrow_ref`; `quantize_a_row_i8_hp`, `quantize_a_4row_i8_hp` and the scalar `quantize_a_nrow_i8_hp_ref`; `memcpy1d`; `repack_q4_0_to_q4_0_256_32_bl_ref` with `make_block_q4_0x32`. About 1,000 lines in the provider's own namespace. Orchestration ported from ggml-spacemit `ime.cpp` @ `4e782bc`. Build: IME code on riscv64 with `-march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii` (GCC >= 15) after the compiler checks of `FindSMTIME.cmake`; scalar references everywhere, so the Mac's serial stand-in runs the whole path; configuring with `GGML_CPU_RISCV64_SPACEMIT` also on stops with an error (two TCM users in one process).

Tests:
- `flagos-check-spacemit`: provider against the CPU backend for the model's matmul shapes at 1, 4, 16, 64, 113, 128 and 512 rows (all three paths), NMSE within `test-backend-ops`' `MUL_MAT` bound (5e-4); on the K3 also IME kernel against the scalar reference; refusals (row length % 256 != 0, rows % 32 != 0, 3-D weight, view, weight in a host buffer, non-contiguous or F16 activations); repack round trip (`set_tensor` then `get_tensor` returns the original bytes); a `MUL_MAT` then `ADD` graph; forced failure still returns `FAILED`; benchmark per path against the CPU.
- `spacemit_check.py --milestone m2b`: the checks above, model output, provider buffer size in the load log, graph splits, perplexity vs M0.6, pp128/tg128.

Expected: matmuls (and the residual `ADD`s next to them) on the AI cores, everything else on the CPU: about 10 splits per layer, roughly 360 per token on Qwen3-4B (estimate), each costing about 10 us of launch plus a small copy. Generation below the IME baseline's 11.1 t/s until M2d moves the rest of the layer.

### M2d design (2026-10-09)

**Done 2026-10-10** (commits `f1cf809` C5b, `7a2b897`, `e5f69ff`; K3 results in `build.md` §7). On the K3: `spacemit_check.py --milestone m2d` 26/26 on Qwen3-4B and 28/28 on Qwen3-0.6B; every claimed `test-backend-ops` case passes with the RVV kernels and with the references; 83 splits per token exactly as designed below; tg128 5.82 -> 7.16 t/s (each removed split saved about 0.118 ms), pp128 51.08 -> 53.00; perplexity +0.21% and mean KLD 0.0024 against the CPU (IME baseline 0.0034); correct with `-fa off` and at a context of 2048; 8,000 forced-failure launches without a hang. Consequences: (1) the 4 Q4_1 `ffn_down` matmuls on the X100 cores take 36% of prefill, and moving them (measured with a requantized model) gives pp128 83, past the prefill target, so M2c is next (question 10); (2) path B is not needed (pp512 equal to IME once those matmuls move); (3) attention on the X100 cores dominates generation beyond about 1k tokens of context (1.72 t/s at depth 4096, IME 6.35), which M2e addresses.

Mac, simulated device (`FLAGOS_SPACEMIT_HOST_TEST_DEVICE`, reference kernels): `test-backend-ops` passes every claimed case, ADD 54/54, MUL 46/46, RMS_NORM 52/52, ROPE 165/165, SET_ROWS 87/87, GET_ROWS 9/9, SWIGLU 12/12, and the multi-op graphs built from them, RMS_NORM_MUL_ADD 30/30, ADD_RMS_NORM 25/25, RMS_NORM_MUL_ROPE 144/144, ROPE_SET_ROWS 24/24 (F16 and MROPE cases reported as not supported). `flagos-check-spacemit` passes, including a Qwen3-like layer in one launch against the CPU backend (all rows: max NMSE 4.0e-4, bound 1e-3, from the three matmuls' activation quantization), ROPE NEOX with heads of 256 (NMSE 0), the row-op cases below (bit-exact) and 400 launches that must fail cleanly. 0 build warnings. The RVV and IME code compiles only on the K3; `scripts/rvv-syntax-check.sh` parses it for riscv64 on the Mac (0 errors; `build.md` §6). A review workflow (four areas, each finding checked by a second agent) found the executor race and the test gaps below; all fixed.

Goal: run every op of a Qwen3 layer except attention on the AI cores, so that a generated token crosses between the X100 and AI cores about twice per layer instead of about 10 times (M2b: 355 splits per token on Qwen3-4B, each handoff about 0.13 ms).

**Prerequisite for generation: norm pinning (C5b).** For batches under 32 tokens, llama.cpp pins every tensor named `norm` (the output of each RMS norm) to the backend of the layer's device (`src/llama-context.cpp:2509-2521`, marked FIXME upstream). Under ACCEL that device is the CPU, so in generation the 4 RMS norms of every layer stay on the CPU whatever the provider claims: 5 CPU splits per layer (4 norms and attention), still about 10 handoffs. Without C5b, M2d leaves generation nearly unchanged (X3b: pinning cost 30% of generation speed on this model). Prefill is pinned too when llama.cpp counts the model as fully offloaded (`full_offload` in the same rule): in this llama.cpp version the default `n_gpu_layers` of -1 counts as all layers (`src/llama-model.cpp:1732-1735`) unless the fit step lowers it; to be checked on the board. Patch, **implemented 2026-10-09** (user approved; a separate commit, for the mentor to keep or replace with an upstream issue, question 9): for layers whose device is the CPU, the rule no longer pins `norm` and `l_last` when an ACCEL backend supports the op; the scheduler places them with their neighbours, as `LLAMA_NO_NORM_PIN` did in X3b. 12 lines in `graph_get_cb`; nothing changes without an ACCEL device, for layers on a GPU, or when no ACCEL backend supports the op (for example BLAS, or this provider before M2d.1).

**Ops and exact claims** (counts per generated token on Qwen3-4B, from the M2b scheduler summary; F32 unless noted). Every operand may sit in the provider's buffer or in host memory (§2.3):

| Op | Count | Claim | Kernel on the K3 (ggml-spacemit `rvv_kernels.cpp` @ `4e782bc`) | Split over tiles |
|---|---|---|---|---|
| `RMS_NORM` | 145 | `eps` >= 0 (the CPU asserts the same); contiguous elements in a row, any row strides (views); in place | `forward_rms_norm_f32`: sum of squares in float vector lanes (the CPU sums in double: a rounding-level difference; the float sum overflows only for \|x\| above about sqrt(FLT_MAX / ne0)). Fixed: `eps` 0 allowed (X0 abort); tail-undisturbed accumulator (`_tu`), so rows that are not a multiple of the vector length do not depend on the core's tail policy | rows |
| `MUL` | 145 | `src1` repeatable into `src0` in every dimension (broadcast); dst shaped like `src0`; contiguous elements in the rows of `src0` and dst | `forward_binary`, when `src1`'s rows are contiguous; otherwise the reference. Claiming `MUL` makes llama.cpp load the norm weights into the provider's buffer (its load-time test is a same-shape `MUL`, `src/llama-model-loader.cpp:960-963`), which places these ops on the AI cores | rows; elements when both operands are one row of the same length (generation's norm `MUL`s and residual `ADD`s; the reference splits rows, so there one tile does them) |
| `ADD` | 72 | as `MUL` (M2a's same-shape claim, extended) | as `MUL` | as `MUL` |
| `ROPE` | 72 | forward; modes NORMAL and NEOX (Qwen3: NEOX); `n_dims` even, 0 < `n_dims` <= `ne0`, `ne0` even (the rest copied in pairs); positions I32, one per token; optional F32 frequency factors; YaRN parameters as the CPU; row strides; in place | `forward_rope_impl<float>`: per-position cos/sin table, for each tile a slice of the op's workspace (sized here: X0's abort); NEOX heads up to 128 (Qwen3) use SpacemiT's scalar loop with a stack table, which they measured faster; larger heads the RVV rotation | rows (heads x tokens) |
| `SET_ROWS` | 72 | F32 or F16 rows into F32 or F16 (the KV cache types); indices I64 or I32; dst may be the CPU-resident KV cache | `forward_set_rows`: K and V go straight into the cache, which is host memory, so no copy. Changed: an index out of range fails the op instead of aborting | columns (repeated indices cannot make two tiles write one element). With `-fa off` the V cache is transposed and written as rows of 1 element, which then all go to tile 0 (correct, slower in prefill); not the M2d default (`-fa auto` keeps flash attention on while attention runs on the CPU); to revisit in M2e |
| `GLU` | 36 | `SWIGLU` only; split and single-tensor (`swapped`) forms; rows as `ggml_is_contiguous_1`; dst shaped as the CPU asserts | `forward_glu_swiglu_f32`; its exp approximation is ggml-cpu's own RVV `ggml_v_expf_m2` | elements |
| `GET_ROWS` | 2 | `src` F32, indices I32 (the last layer's output rows; also admits MoE router weights, and llama.cpp's load-time test would move an F32 token embedding, not Qwen3's quantized one, into the provider's buffer) | `forward_get_rows`. Changed: an index out of range fails the op; a guard against a negative copy size when a tile gets no columns (one row of few columns overran memory) | rows; columns for one row |

Not in M2d: attention (`FLASH_ATTN_EXT`, and the `-fa off` path) stays on the CPU until M2e; the Q6_K output head and the 4 Q4_1 matmuls until M2c; the token-embedding `GET_ROWS` (Q6_K) stays on the CPU, where the graph starts. All these ops are covered well by `test-backend-ops` (default bound: NMSE 1e-7), unlike M2b's matmul.

**Kernels: ggml-spacemit's RVV ports with ggml-cpu references** (user decision 2026-10-09, replacing the portable-C++ proposal; D6, copy per milestone). `flagos-spacemit-rvv-kernels.cpp` copies the seven functions above from `rvv_kernels.cpp` @ `4e782bc` (ranges in its header), plus `rvv_expf_approx_f32m2`. Every change is marked `flagos`: the kernels take a small context (`ith`, `nth`, workspace) instead of ggml-spacemit's, which has no `sync()` here because a step cannot wait (§2.4); the fixes in the table; `copy()` from the M2b IME code, now `noinline` because its assembly does not declare the vector registers it uses. `flagos-spacemit-kernels.cpp` holds the entry points and a reference for each op, written from ggml-cpu's arithmetic (`ops.cpp`, `binary-ops.cpp`, `vec.h`). The reference runs off riscv64, on the K3 when `FLAGOS_SPACEMIT_TEST_REFERENCE=1` (tells a kernel bug from a tiling bug, as in M2b), and where the RVV version does not apply (binary ops with a non-contiguous `src1`). No hand-written RVV. ggml-cpu also has RVV code for some of these ops (its SwiGLU exp is the same algorithm), but much of it goes through its `GGML_SIMD` mapping, which works on 4 floats at a time (`simd-mappings.h:1284-1285`), an eighth of an A100 vector register (VLEN 1024); ggml-spacemit's kernels take the vector length from the hardware and were written for its A100 backend.

**Executor.** Every op is one step; `ROPE` asks for a workspace (one table per tile, `(ne0 + 16) x 4` bytes each), as `MUL_MAT` does. Two fixes:
- `graph_compute` skips empty nodes and nodes without the compute flag, as ggml-cpu and the other backends do: llama.cpp leaves unselected branches in the graph (`ggml_build_forward_select`), and a row op there would read indices that were never set. Empty nodes also kept M2b's GEMM from dividing by zero (rows 0, in prompt ubatches without an output token).
- Failure race (present since M2a, more likely now that `GET_ROWS`/`SET_ROWS` can fail on a bad index): a tile that read the shared failure flag after barrier i could see a failure from step i+1 and leave, while the others waited at barrier i+1 forever. A model of the loop with 8 spinning threads hung 11-13% of forced-failure launches. Now a tile that sees the flag skips its remaining kernels but still reaches every barrier; only a failed barrier (spine-runtime error) ends a tile early. `flagos-check-spacemit` runs 200 forced failures and 200 bad-index launches, so a hang on the K3 would show as a timeout in the runner.

**Placement after M2d (Qwen3-4B generation, with C5b):** one provider split per layer runs `attn_norm`, Q/K/V, the Q and K norms, RoPE and both KV writes; the CPU runs attention; the next provider split runs the output projection, the residual add, `ffn_norm`, gate, up, SwiGLU, down and the add, and continues into the next layer. About 75-85 splits per token (the 4 layers with a Q4_1 down projection keep 2 more each until M2c). One copy per layer remains (Q, 16 KB, to the CPU for attention); K and V go straight into the cache.

**Expected (estimates, to be measured):** Qwen3-4B generation about 125-135 ms per token (7.5-8 t/s, from 5.82), from about 280 fewer handoffs at about 0.13 ms each, with C5b; prefill pp128 about 60-65 t/s (from 51), with C5b. IME-level speed also needs M2c (the output head is about 25 ms per token on the X100 cores) and M2e (attention).

**Tests:**
- Mac: CMake option `FLAGOS_SPACEMIT_HOST_TEST_DEVICE` (non-riscv64 only, decided 2026-10-09 over an environment variable: a test-only switch should not exist in a normal build) exposes a simulated device with 8 tiles run one after another, so `test-backend-ops -b FlagOS:SpacemiT:0` and `flagos-check-spacemit --expect-device` run every claimed case with the reference kernels before the K3 (`build.md` §6).
- `flagos-check-spacemit`: a Qwen3-like layer (norm, Q/K/V, Q and K norms, NEOX RoPE, both KV writes, output projection, residual, FFN norm, SwiGLU FFN, residual, last-token `GET_ROWS`) in one launch, with 1 and 7 tokens, the KV cache in a CPU buffer (the llama.cpp case; `test-backend-ops` allocates every tensor in the backend's own buffer) and the weights in the provider's, against the same graph on the CPU backend: every row of the output (bound 1e-3 for three chained quantized matmuls), both caches, the last-token row as an exact copy, and every intermediate finite (a NaN before a matmul would otherwise come out finite); ROPE NEOX with heads of 256, whole and half rotated, which reaches the RVV rotation (`-o ROPE` has no head above 128; only `RMS_NORM_MUL_ROPE`'s NEOX rows of 768 and 8192 reach it in `test-backend-ops`); `GET_ROWS` of one row with 1 to 2560 columns (tiles without columns); `SET_ROWS` into an F16 cache in a CPU buffer, 1 to 512 rows of 1024 and 512 rows of 1 (the transposed V cache); an index out of range fails the launch, as the first step and after another, 100 times each, and the same node without the compute flag is skipped; forced failures in steps 0 and 1, 100 times each; a one-row ADD of 4M elements and a 4001-row ADD with different rows. Any NaN or Inf counts as a failure (an NMSE of NaN passed every bound before). Mutations confirmed that these checks fail when they should (a NaN in one norm row, a tile reading the wrong row, removing the compute-flag skip).
- K3: `spacemit_check.py --milestone m2d` (`build.md` §6): claims exactly these ops; `test-backend-ops` per op and for the four multi-op graphs, with the RVV and with the reference kernels; `flagos-check-spacemit --full` with the RVV and IME kernels, and without `--full` (smaller matmul shapes) with the reference kernels; graph splits; perplexity within 1% of the CPU; pp128/tg128 against the CPU and IME; `--segv` loads the crash reporter.

**Order:** M2d.1 the binary kernel (`MUL`, `ADD` with broadcast), `RMS_NORM`, SwiGLU; M2d.2 `ROPE`, `SET_ROWS`, `GET_ROWS`; M2d.3 C5b, model runs, runner. All three are implemented together; the K3 run checks them per op.

### M2c design (2026-10-10)

**M2c.1 done 2026-10-10** (commit `6577170`; K3 results in `build.md` §7). On the K3: `spacemit_check.py --milestone m2c` 28/28 on Qwen3-4B; Q4_1 against the CPU on the converted weights max NMSE 9.7e-7 with the IME kernels; the probe gives 3.0e-7 for 1 row and for 4, so the exact 1-row branch works on the board; 75 splits per token as designed; pp128 53.0 -> 83.5 (above `ime` 79.3 and the prefill target of 82), tg128 7.16 -> 7.22 (within the variation between days); perplexity -0.10%; mean KLD against the CPU 0.0032 with 512-token batches (M2d 0.0024, `ime` 0.0034) and 0.0020 with `-ub 1`. Consequences: (1) Q4_1 stays on the AI cores (R4, accepted with this evidence); (2) the prefill half of M2e's exit is met, and generation now waits for the head (M2c.2) and attention (M2e); (3) generation is the more accurate path, because in prefill the 4-row Q4_0 quantizer shares one activation scale across 4 tokens; (4) with 512-token batches M2c.1 is within one standard error of `ime`'s KLD, so M2c.2's accuracy is judged by its increase over M2c.1 (below). Qwen3-0.6B not run (expected 59 splits). M2c.2 (the Q6_K output head) follows.

**Order (user decision 2026-10-10, question 10):** M2c before M2e, the Q4_1 matmuls first. On the K3 (`build.md` §7, M2d) the 4 Q4_1 `ffn_down` matmuls of Qwen3-4B take 36% of prefill on the X100 cores, because ggml-cpu repacks only Q4_0; with them requantized to Q4_0 (speed only) pp128 went 53 -> 83, past the prefill target, and tg128 6.95 -> 7.26. Qwen3-0.6B has 3 such layers. (Side effect: in provider mode these matmuls leave `ggml_vec_dot_q4_1_q8_1` on the X100 cores, where M2b's one CPU-backend crash happened.)

**Scope (M2c.1):** Q4_1 `MUL_MAT` with the same op check as Q4_0: weight 2-D, rows % 32 == 0, not a view, in the provider's buffer (or unallocated); activations F32 contiguous; output F32 contiguous. Any row length (Q4_1 blocks are 32 values, which is also the layout's K block). Qwen3-4B: 4 x (9728 -> 2560); Qwen3-0.6B: 3 x (3072 -> 1024).

**Route: the in-tree IME path's Q4_1 route, unchanged.** In-tree `ggml-cpu/spacemit` and ggml-spacemit both choose `q4_1_32x32_q8_0` on the A100 cores (`ime.cpp:1299-1319`; a Q4_1 variant of the 32x256 layout is commented out there as TODO), and the IME baseline's perplexity on Qwen3-4B (9.7680, CPU 9.7740) includes it:
- weights, layout `q4_1 32x32`: 32 rows interleaved, K in blocks of 32; per row and block an fp16 scale d, a uint8 zero point zp and 32 4-bit values q, a weight being d x (q - zp). 19 bytes per 32 weights (GGUF Q4_1: 20). Repack `repack_q4_1_to_q4_1_32_bl_ref` with `make_block_q4_1x32`, the scalar version: upstream calls an RVV version that divides in fp16, so near .5 it can round a zero point differently; the scalar one gives the same bytes on the Mac and the K3.
- activations: int8 per 32 values with an fp32 scale and the negated int16 sum, 38 bytes per 32 values (`quantize_a_row_i8`, `quantize_a_4row_i8`).
- kernels: `gemm_kernel_i8i4_m1` (1 row) and `gemm_kernel_i8i4_m4` (4 rows), the zero-point variants.

**The conversion is lossy (R4).** Q4_1 stores a weight as d x q + m, m an fp16 minimum; the layout needs an integer zero point, so the repack sets zp = clamp(round(-m / d), 0, 15). All weights of a block move by the same amount, at most d/2 when -m/d is within [0, 15]. Outside that range (a block of one sign) zp is clamped and the block moves further; a block of equal values (d = 0) becomes 0. Real weights have both signs in nearly every block (with an imatrix, llama.cpp's Q4_1 even forces m <= 0, `ggml-quants.c:1012, 2167`). On random weights the conversion changes a matmul's output by NMSE 5.9e-3 (`flagos-check-spacemit` info line): the shift is as large as a rounding error but common to the 32 weights, so it about doubles Q4_1's own error. The model decides: KLD against the CPU's logits, against M2d (0.00238) and the IME baseline (0.00339, which includes this conversion), and perplexity within 1%.

**Reads of a lossy layout (decided 2026-10-10, §2.3).** `get_tensor` returns the converted Q4_1: the same d and quants, m = -zp x d, the weights the kernels compute with; repacking it gives the same zero points again (checked). A partial write must cover whole blocks: it re-converts the blocks it touches from the converted values, so a block's minimum split over two writes would get a different zero point than one whole write (found in review); such a write aborts. llama.cpp writes weights whole on this device. Alternatives: keeping the GGUF bytes as well costs 39 instead of 19 bytes per 32 weights for these tensors (Qwen3-4B: +59 MiB) against single storage (D4), and a weight the scheduler copies out would compute differently on the CPU than here; not claiming Q4_1 is M2d's state. `test-backend-ops` reads the tested backend's tensors through `get_tensor` when it copies the graph to the CPU, so it would compare the kernels on the converted weights; no `MUL_MAT` case fits the layout anyway (Q4_1 cases have 16-row weights).

**Execution:** M2b's two steps (quantize, GEMM) and three paths with the same thresholds, now chosen through a table per layout (K block, row sizes, kernels) in `flagos-spacemit-ime.cpp`. For 1 row (generation) Q4_1 also takes the direct GEMV path: activations in TCM, weights from DRAM, 128 columns per call. ggml-spacemit and the in-tree path instead stage Q4_1 weights in TCM for 1 row (their path B; ggml-spacemit took the direct path only for Q4_0, where it measured faster); the `--bench` line `q4_1 matmul 9728x2560 rows 1` measures ours against the CPU and against Q4_0 at the same shape. Workspace at 512 rows x 9728: 5.9 MB.

**Kernel code (D6):** copied from in-tree @ `ba360ef`, byte-identical to ggml-spacemit `4e782bc` and upstream master (fetched 2026-10-10): `gemm_kernel_i8i4_m1`, `_m4` and the scalar `gemm_kernel_i8i4_mrow_ref`; `quantize_a_row_i8`, `quantize_a_4row_i8` and the scalar `quantize_a_nrow_i8_ref`; `block_with_zp`, `make_block_q4_1x32`, `repack_q4_1_to_q4_1_32_bl_ref`; `q8_blk_size`. About 800 lines, in the M2b file. Changes, marked `flagos`: the reference GEMM asserted that the K blocks are a multiple of 16 and dropped a partial last group of 16 (it accumulates up to 16 blocks in fp16); it now adds that group, so the provider claims any row length (the IME kernels loop per block and have no such limit); a zero guard in the reference quantizer (an all-zero block divided by zero; the RVV versions guard). Unaligned fp32 scales in the references (1-row activation blocks are 38 bytes) are read and written with `memcpy` (UBSan, found in review). New: the inverse repack `unpack_q4_1`, and the dispatch.

**The 1-row kernel runs its exact branch (decided 2026-10-10, R12).** `_m4` sums each block exactly in 32-bit integers. Upstream's active branch of `_m1`, which every generated token uses, adds each block's products in fp16 (`vmadot*.hp`) and multiplies zero point and activation sum in 16 bits (`vmul.vx` at SEW 16, keeping the low 16 bits): that wraps when 32 activations of a block are near their maximum and of one sign and the zero point is above 8, and the fp16 sum loses precision when the zero-point term nearly cancels it. `_m1` also holds the exact branch, disabled with `#if 0`: 32-bit `vmadotsu`/`vmadotu`, the zero-point product widened (`vwmul.vx`), the same arithmetic as `_m4`. The provider enables it (one line, marked `flagos`). Why: perplexity and KLD run 512-row batches, which use only `_m4`, so they say nothing about `_m1`'s arithmetic; with the exact branch generation computes what prefill computes. It costs about 7 more vector instructions per block in a GEMV that is memory-bound by about 10 to 1 (the `--bench` line `q4_1 matmul 9728x2560 rows 1` measures it). Alternatives: keep upstream's branch and accept the wrap (wrong results for some valid inputs; SwiGLU outputs rarely look like that, but a claim covers every input); route 1 row through `_m4` with 3 zero rows (exact on shipped code, but needs a 4-row activation block, a scratch output and a copy in every path that ends with 1 row; the fallback if the exact branch fails on the board). The probe in `flagos-check-spacemit` (zero point 15 against 32 equal activations, 1 row against 4; NMSE 4.6 if the product is 16 bits, shown by emulating it in the reference) is a check: it fails the run.

The copied reference keeps upstream's numerics: activation scales converted to fp16 (blocks with max\|x\| below about 4e-6 contribute 0) and partial sums of up to 16 blocks in fp16 (inf above 65504). This affects only the Mac and `FLAGOS_SPACEMIT_TEST_REFERENCE` runs; the IME kernels use fp32 scales (found in review).

**Fallback if the accuracy is not acceptable:** keep the exact minimum. Store zp = 0 and the fp16 minima beside the weights, and after the IME GEMM add m x (the block's activation sum) for each block; the sums are already in the quantized activations. Exact like the CPU, 2 more bytes per 32 weights, and an extra RVV pass of 1/32 of the matmul's work. Not upstream code, so it needs its own tests.

**Tests** (Mac: simulated device, reference kernels, path C only since the serial stand-in has no TCM; K3: all paths, IME and reference kernels):
- claims: Q4_1 at any row count, row length 256 and 2880 (90 blocks); refused with 16 rows, 3-D, a view, F16 activations, in a CPU buffer.
- round trip: `get_tensor` returns exactly the documented conversion; on random weights no weight moves by more than half a step; a second round trip changes nothing; partial reads and writes.
- matmuls against the CPU on the converted weights (`mm_weight` copies the provider's read-back to the CPU): 256 x 32 and 2880 x 64 at up to 128 rows, 3072 x 1024 (0.6B `ffn_down`) at 1-128 rows; `--full` adds 9728 x 2560 (4B `ffn_down`) at 1, 7, 128, 512; bound 5e-4.
- layer: the Qwen3-like layer with a Q4_1 down projection, 1 and 7 tokens, against the CPU.
- the zero-point probe (above), a check; informational: the conversion's own effect (random weights).
- mutations confirmed: without the tail fix the reference returns 0 for K = 256 (NMSE 1.0); an inverse off by one step fails the round trip and the matmuls; a 16-bit zero-point product in the reference fails the probe (NMSE 4.6).
- K3: `spacemit_check.py --milestone m2c` (default; as m2d plus the lines above), then KLD (`build.md` §6) with 512-row batches (prefill: `_m4`, paths A and C) and with `-ub 1` (generation: `_m1`, the GEMV path, and the 1-row cases of every other provider op; measured for no milestone so far).

Mac results 2026-10-10: `flagos-check-spacemit --expect-device --full` passes: Q4_1 20 cases up to 512 rows, max NMSE 1.1e-6 (bound 5e-4; Q4_0 unchanged, 40 cases, 2.6e-5); the layer with a Q4_1 down projection within its bound (max NMSE of all layer cases 4.0e-4); conversion alone NMSE 5.9e-3; probe 6.3e-7 for 1 and 4 rows. `test-backend-ops -o MUL_MAT`: 0 claimed cases, as with Q4_0. 0 build warnings; `rvv-syntax-check.sh`: 0 errors and the same 2 warnings in the copied Q4_0 kernel.

**Expected (estimates, to be measured):** Qwen3-4B 83 -> 75 splits per token (the 4 layers no longer leave the AI cores for `ffn_down`), pp128 about 80 (requantized: 83 with Q4_0's 32x256 kernel; IME, running Q4_1 through this 32x32 route: 79.5), tg128 about 7.3 (requantized: 7.26), KLD between M2d's 0.0024 and IME's 0.0034. Qwen3-0.6B 65 -> 59 splits.

**Exit (M2c.1):** every `flagos-check-spacemit` case passes on the K3 with the IME and with the reference kernels, the probe included; KLD at most IME's 0.0034 with 512-row batches and with `-ub 1`, and perplexity within 1% of the CPU (else Q4_1 goes back to the CPU or the exact-minimum fallback follows); pp128, tg128 and splits recorded.

**M2c.2 (next): the Q6_K output head**, designed below ("M2c.2 design"). Accuracy: M2c.1 is already within one standard error of `ime`'s KLD at 512-token batches, so a fixed ceiling of 0.0034 would be decided by noise; M2c.2 is judged by its increase in mean KLD over M2c.1, against the same CPU base file, with both runs back to back, with 512-token batches and with `-ub 1`.

### M2c.2 design (2026-10-10)

**M2c.2 done 2026-10-10** (commit `72e0666`; K3 results in `build.md` §7). On the K3: `spacemit_check.py --milestone m2c2` 29/29 on Qwen3-4B; the 8-bit IME kernels match the references to three digits at every shape, both heads included; `test-backend-ops -o MUL_MAT` 3/3; 74 splits; the head takes 16.6 ms per token on the AI cores (24.9 GB/s, the same bandwidth as the Q4 1-row matmuls, so Q8_0 needs no path B) against 26.0 ms on the X100 cores; tg128 7.22 -> 7.84 (+8.6%), pp128 unchanged; perplexity -0.16%; mean KLD +0.00010 (512-token batches, 0.00326) and +0.00017 (`-ub 1`, 0.00220) over M2c.1, under the exit limit of 0.0005. This completes M2c for the Qwen3 targets: every matmul weight type of Qwen3-4B Q4_0 (Q4_0, Q4_1, Q6_K) runs on the AI cores, and Q8_0 as well; Q4_K and `MUL_MAT_ID` wait for a target that needs them (question 5). Consequence: generation is now 127.6 ms per token against `ime`'s 90.7; the gap is attention on the X100 cores (M2e) and the 1-row matmul bandwidth ("What M2c.2 leaves" below). Qwen3-0.6B not run (expected 58 splits).

**Implementation status, 2026-10-10:** implemented, Mac checks pass (below).

Mac results, simulated device (reference kernels, path C only): `flagos-check-spacemit --expect-device` and `--full` pass. Q8_0: 14 cases, max NMSE 4.5e-8. Q6_K against the CPU's Q6_K (requantization plus kernel, what the model sees): 12 cases, 5.9e-5 (bound 5e-4); 14 with `--full`, which adds both heads at 128 rows. Q6_K against the CPU's Q8_0 of the same values, built with ggml's own quantizer (the requantization alone): 5.1e-7 (bound 5e-6). Q6_K read-back within NMSE 1.0e-6 of the GGUF, a second round trip 9.7e-7 (bound 1e-5). `test-backend-ops -o MUL_MAT`: 3/3 claimed cases pass; the 11 M2d op sets unchanged; the whole `test-backend-ops` run on the provider: 662 cases pass, 0 fail. 0 build warnings; `rvv-syntax-check.sh` 0 errors; UBSan and ASan clean in provider code. Mutations caught: a wrong inverse permutation, a read-back that skips the requantization, wrong column or K offsets in the reference GEMM, `get_alloc_size` returning `ggml_nbytes` (stops the run before the heap is overrun), and in the requantization truncation, a scale 0.3% or 0.8% too large and rounding up by half a step. Removing the zero guard makes UBSan report the NaN-to-int8 cast in the new zero-row test.

Review (four lenses, each finding checked by three independent verifiers): one confirmed finding, fixed: the requantization was effectively unchecked (faults in it kept every check green, `test-backend-ops` included, since its CPU copy is our own read-back), now the tight Q8_0 comparison above. Three findings rejected. Confirmed correct: copies identical except the marked lines; R13 unreachable (every 4-row call gets at most 32 columns: GEMV uses 1 row, path A and path C at most 32 columns); every ggml allocation path uses `get_alloc_size`; no llama.cpp path writes a weight partially on this device; a small Qwen3 GGUF loads with the tied head in the provider's buffer and the embedding on the CPU; the runner's timeouts hold with the head shapes.

Implementation notes: the dispatch calls `_m4` once and asserts at most 32 columns (R13); the test quantizes weights on 8 threads (same bytes as before, the 4B head in 4.6 s instead of 14 s), so `flagos-check-spacemit` links `Threads::Threads`; the bench line is `q6_K matmul 2560x151936`.

**Goal:** the output head on the AI cores. Qwen3-4B: Q6_K, 2560 -> 151936, 304 MiB; Qwen3-0.6B: 1024 -> 151936, 122 MiB if Q6_K (expected from llama.cpp's rule below; to confirm in its load log). It is one 1-row matmul per generated token, and also per prompt batch: by default llama.cpp outputs only the last token of a batch (`src/llama-batch.cpp:120-128`; llama-bench's pp test uses that default, `tools/llama-bench/llama-bench.cpp:2131`). Only runs that ask for every token's logits (perplexity, KLD) multiply it by many rows. On the X100 cores the head costs about 25 ms per generated token on 4B (`build.md` §7, M2d: the `--pure` experiment), about 10 ms on 0.6B (estimate), plus one handoff.

**Facts (verified in code):**
- Placement: Qwen3 makes the head a duplicate of the token embedding (`src/models/qwen3.cpp:21-26`). The loader treats a duplicated `token_embd` as the output tensor, op `MUL_MAT`, chosen from the output layer's buffer list (`src/llama-model-loader.cpp:1101-1107`, `src/llama-arch.cpp:644, 648`), and creates a second tensor when the buffer type chosen differs from the embedding's (`:1298-1316`). So claiming Q6_K `MUL_MAT` moves only the head into the provider's buffer; the embedding's `GET_ROWS` stays on the CPU and reads the mapped GGUF, touching only the rows of the tokens seen. The two uses do not conflict.
- Types: llama.cpp gives a tied embedding the output head's type: Q6_K when the row length is a multiple of 256, else Q8_0 (`src/llama-quant.cpp:449-473`).
- In-tree route: Q6_K and Q8_0 both use the layout `q8_0 32x32` and `gemm_kernel_i8i8` (`ggml-cpu/spacemit/ime.cpp:1335-1350`). Q6_K is requantized to Q8_0 at load (`repack_q6_k_to_q8_0_32_bl_ref`, the scalar version, which upstream itself selects, `repack.cpp:1763-1768`): per 32 values, scale = max / 127, so every weight moves by at most half a Q8_0 step. Q8_0 is a byte permutation (`repack_q8_0_to_q8_0_32_bl_ref`). The IME baseline's perplexity and KLD on Qwen3-4B (9.7680, 0.0034) include this route for the head.
- Size: Q8_0 needs 34 bytes per 32 weights, Q6_K 26.25: the head grows 30% (4B 304 -> 394 MiB; 0.6B 122 -> 158 MiB). The buffer type needs `get_alloc_size` for the first time.
- Kernels: `gemm_kernel_i8i8_m1` and `_m4` multiply int8 by int8 with 32-bit `vmadot` and scale in fp32 (`ime2_kernels.cpp:4773-5004`): exact per block in both, with no fp16 or 16-bit branch (unlike Q4_1's 1-row kernel) and no `#if`. The activations use the per-row quantizer already copied for Q4_1. Copies identical in in-tree, ggml-spacemit `4e782bc` and upstream master (checked 2026-10-10).
- Two latent faults in the copied code, unreachable by construction: `_m4` decrements its input-only asm operand `%[BK]` (`:4981`) inside a loop over 32-column groups, so in a call with more than 32 columns a reused register could cut later groups' K loop short; every multi-row call passes at most 32 columns (here as upstream), and the dispatch asserts it (implementation: a loop calling it per 32 columns would not have helped, since inlined into that loop the decremented register could carry over between iterations). `_m1`'s asm declares no vector registers, as Q4_1's: called only from the dispatch, which holds no vector values.
- The Q6_K requantization divides by the block maximum, so an all-zero 32-value block (padding rows of a vocabulary) computes 0/0 and casts NaN to int8: undefined behaviour, harmless only because its scale is 0. Zero guard needed.
- X0: ggml-spacemit failed 46 Q8_0 cases (writes past the output) and 10 Q6_K cases (NMSE up to 4e15; `device-type.md` X0). Likely cause: the in-tree claim has no `rows % 32` check for these two types (unlike Q4_0/Q4_1), while `_m4` always stores 32 columns and the repack writes whole 32-row groups past `ggml_nbytes`. Our claim requires rows % 32 == 0 (every Qwen3 vocabulary: 151936 = 4748 x 32); the claimed `test-backend-ops` cases below test the rest.

**Route: the in-tree one, as for Q4_1:** layout `q8_0 32x32` for Q6_K (requantized) and Q8_0 (lossless), kernels `gemm_kernel_i8i8`. Alternatives:
- Exact Q6_K on the AI cores with ggml-cpu's RVV dot product for VLEN 1024 (`ggml-cpu/arch/riscv/quants.c:2618`, added upstream 2026-06-04): no requantization, the head stays 304 MiB in ggml's layout (exact reads, no `get_alloc_size`). Against: its 1-row speed is uncertain (it works on 32 bytes at a time on a 1024-bit core, about 90 vector instructions per 256 weights, so it may be compute-bound at 13-21 ms against Q8_0's memory-bound ~17 ms); batches that need every token's logits run about 15x slower than on the IME; that code path has probably never run on hardware (the CPU backend runs on the 256-bit X100 cores); and it needs a new execution shape (dot products per column). Kept as the fallback if the KLD cost below is too high.
- Requantize to Q4_0 (the `--pure` experiment: tg128 8.19): faster, but 4 bits instead of 8.
- Keep Q6_K bytes in memory and requantize slabs into TCM per token: exact reads and 304 MiB, but per-token requantization work and a new tiling path; only if memory becomes the constraint.
- Leave the head on the X100 cores (M2c.1's state).

**Reads of the requantized head (extends §2.3).** Q6_K cannot express the stored Q8_0 values, so `get_tensor` returns the nearest Q6_K: the stored values dequantized, then quantized with ggml's own Q6_K quantizer. Measured on the Mac (`ggml_quantize_chunk`, random weights quantized as `test-backend-ops` does): that read-back differs from the kernel's weights by NMSE 1.5e-5 (uniform) to 5.7e-5 (normal with 1% outliers), which is what `test-backend-ops` sees when it copies the weight to the CPU, 10-35x under its 5e-4 bound; the requantization itself moves the weights by 1.4e-5 to 5.5e-5. Alternatives: keep the GGUF bytes as well (exact reads, +304 MiB on 4B for a read llama.cpp never makes; `test-backend-ops` would then see the same magnitude, the CPU's Q6_K against the kernel's Q8_0); abort (breaks 2 claimed `test-backend-ops` cases). A partial write aborts for this layout (it would requantize the whole tensor through the approximate read); whole writes and whole-tensor memset work. M2f: "requantized layouts return the nearest tensor of their type".

**Scope:** `MUL_MAT` with weights Q6_K (row length % 256, as every Q6_K) or Q8_0 (% 32), 2-D, rows % 32 == 0, not a view, in the provider's buffer; activations F32 contiguous; output F32 contiguous. Q8_0 comes along because it shares the layout and kernel: lossless, it gives `test-backend-ops` an IME matmul on exact weights, and it covers heads with row lengths not divisible by 256 and Q8_0 models.

**Execution:** M2b's paths with a third table entry (K block 32, activation rows as Q4_1, weight rows K/32 x 34 bytes); `_m4` called with at most 32 columns, asserted. Generation takes the direct GEMV path. ggml-spacemit uses the direct path for Q4_0 only and states that for Q8_0 it "always loses to the staged TCM path" (its `ime.cpp:421-422`); the bench line `q6_K matmul 2560x151936 rows 1` (a Q6_K weight run as Q8_0, against the CPU's Q6_K) shows whether ours reaches about 24 GB/s (17 ms for 413 MB); if not, path B for 1 row comes next. All-logits batches take path C.

**Kernel code (D6):** `gemm_kernel_i8i8_m1`, `_m4`, `_mrow_ref`; `block_q8_0x32`, `make_block_q8_0x32`; `repack_q8_0_to_q8_0_32_bl_ref`; `repack_q6_k_to_q8_0_32_bl_ref` (about 350 lines). Changes, marked `flagos`: the zero guard in the Q6_K requantization; `memcpy` for the unaligned activation scale in `_mrow_ref` (as in M2c.1). New: `unpack_q8_0` (inverse permutation), the Q6_K read-back, `get_alloc_size`, the dispatch.

**Tests:**
- claims: Q6_K at any row count (row lengths 2560, 2048); Q8_0 (2880); refused: rows not % 32 (16, and 17 as in `test-backend-ops`), 3-D, a view, F16 activations, a CPU-buffer weight; `get_alloc_size` = rows x K/32 x 34 for Q6_K.
- round trip: Q8_0 exact; Q6_K: a read is the nearest Q6_K of the stored values (NMSE against the GGUF at most 1e-4 on random weights), and writing it back and reading again stays within that.
- matmuls against the CPU: Q8_0 on identical weights (the kernel alone); Q6_K against the CPU's Q6_K (requantization and kernel, bound 5e-4; measured 5.9e-5, the estimate was 2e-5) and against the CPU's Q8_0 of the same values (the requantization alone, bound 5e-6) at the head shapes 2560 x 151936 and 1024 x 151936, 1, 7 and 128 rows, plus small shapes.
- `test-backend-ops -o MUL_MAT`: 3 claimed cases, Q8_0 2880 x 32 rows x 2880 and Q6_K 2048 x 6144 and 151936 x 2048 (1 row; the latter is a Qwen3 vocabulary projection), with the IME and the reference kernels: the first `test-backend-ops` cases for an IME matmul (`tests/test-backend-ops.cpp:8876, 9016-9017`).
- bench: Q8_0 at the 4B head shape against the CPU's Q6_K.
- K3: `spacemit_check.py --milestone m2c2` (as m2c, plus `test-backend-ops -o MUL_MAT`), then both KLD runs; KLD is deterministic, so M2c.1's numbers are the baseline.

**Expected (estimates):** Qwen3-4B 75 -> 74 splits; tg128 +6% (head 25 -> about 17 ms); pp128 unchanged (the head is 1 row per batch); provider buffer +394 MiB; mean KLD +0.0002 (logit noise from the requantization, about one standard error; `ime`'s 0.0034 is M2c.1's 0.0032 plus this head). Qwen3-0.6B 59 -> 58 splits, tg128 about +10% (head about 10 -> 7 ms).

**Exit:** every claimed case passes (`flagos-check-spacemit`; `test-backend-ops -o MUL_MAT` 3/3) with the IME and the reference kernels; mean KLD at most 0.0005 above M2c.1's (0.00316 and 0.00203) at both batch sizes, same top token within noise; perplexity within 1% of the CPU; tg128, splits, buffer size and the head's GEMV bandwidth recorded.

**What M2c.2 leaves:** the X100 cores then run only attention and the token embedding lookup. Generation would be about 130 ms per token against `ime`'s 90.5. Besides attention (M2e), the 1-row matmuls: `ime` streams 2.46 GB of weights per token (Q4_0, Q4_1, Q8_0 head) in at most 90.5 ms, so its 1-row matmuls run at 27 GB/s or more, while our direct GEMV path measures 23.6-25.5 GB/s (`build.md` §7). Each GB/s is about 4 ms per token; reaching 30 GB/s would save about 20 ms, more than M2c.2. Worth measuring after M2c.2: path B for 1 row (weight slabs staged in TCM), which `ime` uses there.

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
| Numerics | `test-backend-ops test -b FlagOS:SpacemiT:0 -o <op>`; where no `test-backend-ops` case fits the claim (Q4_0 IME layout), `flagos-check-spacemit` compares with the CPU backend; perplexity vs M0.6, with the default `-fa auto` and with `-fa on` | M2 |
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
| R4 | Lossy repacks (Q4_1, Q4_K, Q6_K) change results. Q4_1 (M2c.1): on random weights the zero-point conversion changes a matmul by NMSE 5.9e-3, about doubling Q4_1's own error | claim only with KLD and perplexity evidence on the model (M2c); fallback for Q4_1: keep the exact minimum (M2c design). Q4_1 accepted 2026-10-10: on Qwen3-4B it raised the mean KLD against the CPU from 0.0024 to 0.0032 with 512-token batches (0.0020 with `-ub 1`), perplexity -0.10%. Q6_K (requantized to Q8_0) accepted 2026-10-10: +0.0001 (512-token batches) and +0.0002 (`-ub 1`), perplexity -0.16% |
| R5 | Under ACCEL, `-fa auto` switches flash attention off when the provider runs attention, so default runs use non-flash attention | correct non-flash kernels (M2e); document `-fa on`; upstream fix candidate (§2.4) |
| R6 | Under ACCEL, a provider op that refuses CPU buffers forces copies and splits (E1: 280 splits per generated token). Also many splits until M2d, while each matmul is surrounded by CPU ops | accept host buffers for non-weight operands (§2.3); X3 measures it; split count in every benchmark |
| R7 | ggml-spacemit is AI-generated and over-claims (X0: wrong F16 matmul, masked softmax, CPY; aborts in RMS_NORM, ROPE) | port piece by piece with fixes and tests; the mentor allows changing it |
| R8 | FlagTree SpacemiT is immature (no AOT tool, no tests, QEMU-only CI with VLEN 1024) | M3 is conditional and starts with one op |
| R9 | Common gaps: no `get_proc_address` forwarding (thread count, abort callback), no RVV/IME feature bits, no `LOCAL_SCRATCH` cap for TCM (mentor design §8.1, not in code) | work around in the provider; propose C2-C4 only when needed |
| R10 | The IME Q4_0 32x256 kernel has never been checked op by op: no `test-backend-ops` case fits the layout, the 1198/1198 run used plain buffers, and X0 ran only the cases ggml-spacemit claimed (its Q8_0 and Q6_K IME matmuls failed there) | M2b compares with the CPU backend at the model's shapes, and the IME kernel with its scalar reference |
| R11 | Copied kernel code (D6) misses later upstream fixes | at each upstream merge, check `git log` of `ggml/src/ggml-cpu/spacemit/`; copied functions keep their bodies unchanged and note their origin |
| R12 | Upstream's active branch of the 1-row Q4_1 IME kernel (`gemm_kernel_i8i4_m1`) multiplies zero point and activation sum in 16 bits (`vmul.vx` at SEW 16), which wraps for 32 activations near their block's maximum and of one sign against a zero point above 8, and adds each block in fp16. Found by reading the code (and by the review), from the ISA's definition; not run on the board | the provider enables the function's exact branch (`#if 0` upstream, M2c design); a probe in `flagos-check-spacemit` fails the run if the 1-row result is wrong; KLD with `-ub 1`; fallback: 1 row through the 4-row kernel. Confirmed on the board 2026-10-10: the probe gives 3.0e-7 for 1 row and for 4, mean KLD 0.0020 with `-ub 1` |
| R13 | The copied 4-row 8-bit kernel (`gemm_kernel_i8i8_m4`) decrements an input-only asm operand inside its loop over 32-column groups, so a call with more than 32 columns depends on register allocation | every multi-row call passes at most 32 columns; the M2c.2 dispatch asserts it (a loop over 32-column calls would not help once the kernel is inlined). On the K3 2026-10-10 the 4-row kernel matches the reference at every shape |

Optional Common changes, each needing separate approval: C2 RVV/IME/TCM feature bits in `flagos-target.h` (append-only); C3 forward provider functions through `get_proc_address`; C4 `LOCAL_SCRATCH` memory cap.

Outside FlagOS, needing the mentor's decision (fork patch or upstream llama.cpp issue): C5, in `src/llama-context.cpp`, (a) let the automatic flash-attention check accept an ACCEL backend that supports the op when the layer's device is the CPU (`:506-560`); (b) skip norm pinning when the layer's device is the CPU (`:2499-2520`, marked FIXME upstream; **implemented 2026-10-09** for M2d, skipping the pin only when an ACCEL backend supports the op, see M2d design). Without C5, ACCEL users need `-fa on` and get 30-39% slower generation (Qwen3-4B / 0.6B) (`device-type.md` §5, X3b).

## 7. Questions for the mentor

1. ~~Device type~~ decided: ACCEL (2026-10-08). Still open: FlagOS `caps.kind`, `cpu_accelerator` (proposed) or `ai_accelerator`?
2. Memory under ACCEL: weights in an IME-repacked buffer, stored once, with the provider reading CPU buffers for everything else (A2, §2.3). We are building M2b on A2 (2026-10-09); please confirm. The alternative (A1) doubles quantized-weight memory and would change only the buffer part.
3. Is spine-runtime (closed `libspert`) acceptable as a hard dependency of the provider?
4. Kernel reuse: decided 2026-10-09 to copy, per milestone, only the functions used into `providers/spacemit/` (§2.5), following the mentor design's "reference, not boundary" (§14.1) and SpacemiT's own ggml-spacemit. Objections? The alternative is compiling the in-tree sources in place.
5. Target for Phase 0: Qwen3-4B Q4_0 (SpacemiT has reference numbers) or Qwen3.5-4B Q4_K_M (comparable with AMD, but needs lossy Q4_K/Q6_K repacks and gated-delta-net ops early)?
6. Is FlagTree AOT (M3) in scope, and should it be cross-compiled on x86 or built on the K3?
7. Is co-building in `providers/spacemit` agreed with SpacemiT, given they ship ggml-spacemit as an `ACCEL` backend?
8. Should we report the TCM issues to SpacemiT (no dead-owner recovery; `try_wait` timeout unit about 12 µs, not 1 µs)? Also one CPU-backend crash in 8 runs whose register state the program cannot produce (`build.md` §7; targeted tests with about 80 million page faults did not reproduce it). And, found by reading: the 1-row Q4_1 IME kernel's active branch multiplies zero point and activation sum in 16 bits (R12).
9. ACCEL gaps in llama.cpp (C5, §6): patch the automatic flash-attention check and norm pinning in our fork, or raise an upstream llama.cpp issue? (b), norm pinning, is now patched in the fork as a separate commit (`f1cf809`, 2026-10-09, needed for M2d; on the K3 it gives the same 83 splits for 1-token and 512-token graphs); keep it, or replace it with an upstream issue? Without them ACCEL needs `-fa on` and generates 30-39% slower (`device-type.md` §5, X3b).
10. ~~Order after M2d~~ decided by the user 2026-10-10: M2c first, starting with the Q4_1 matmuls (M2c.1), then the Q6_K head (M2c.2), then M2e. The question was: M2c (Q4_1 and Q6_K on the AI cores), then M2e (attention)? Measured on the K3 (`build.md` §7, M2d): moving the 4 Q4_1 `ffn_down` matmuls alone raises pp128 from 53 to 83 t/s, past the prefill target, and moving the head saves about 12-25 ms per token; attention on the X100 cores costs +445 ms per token at a context of 4096 tokens (IME: +68 ms), so M2e decides generation speed at long context. M2e brings the flash-attention question of item 9 (C5a) back.
