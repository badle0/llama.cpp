# SpacemiT K3 provider: build plan

Written 2026-10-07; updated as milestones land (M1 done 2026-10-08, M2a and M2b done 2026-10-09, M2d and M2c done 2026-10-10, M2e.1 done 2026-10-13).

- "Design" = `ggml-flagos-provider-design.pdf` V2.0 (review baseline, 2026-09-22). Section numbers like "Design §13.3" refer to it.
- "Mentor design" = `FLAGOS_BACKEND_MULTIPLATFORM_DESIGN.md` and `FLAGOS_BACKEND_MULTIPLATFORM_DESIGN_REVIEW_RESOLUTION.md` on `kevin/feature/flagos-multi-provider-backend` (v0.3, 2026-08-22). Read with `git show kevin/feature/flagos-multi-provider-backend:<file>`.
- Decisions marked *proposed* need the mentor's confirmation (§7). Facts marked UNVERIFIED have not been checked on the board or in code.

## 0. Summary

1. Build a **direct-op provider** (Design §4.1) at `ggml/src/ggml-flagos/providers/spacemit/`, compiled into `libggml-flagos`, that runs ggml ops on the 8 A100 AI cores (CPUs 8-15).
2. **Device type: ggml `ACCEL`** (mentor decision, 2026-10-08). llama.cpp then keeps model layers and the KV cache on the CPU side, tries the provider's buffer type first for each weight whose op the provider supports, and always creates the provider's backend ahead of the CPU backend (§2.8). Whole layers reach the AI cores only if the provider also reads CPU buffers directly: in E1, a backend that did not needed 280 backend switches per generated token.
3. **Memory**: the provider's buffer type holds weights in IME-repacked layout (`is_host = false`), stored once. The KV cache and other inputs stay in CPU buffers, which the provider reads directly (§2.3, D4: building on this from M2b, 2026-10-09; mentor confirmation pending).
4. Execution: **one spine-runtime launch per split**. Each of the 8 tiles walks the split's steps (one or more per node), with a barrier after each step and the core's TCM as scratch. This is ggml-spacemit's model, except that its kernels also wait inside a node.
5. Kernels: copy into the provider, milestone by milestone, only the IME GEMM, quantizer and repack functions it uses from the in-tree `ggml-cpu/spacemit` sources (byte-identical to ggml-spacemit's), and port ggml-spacemit's tiling and RVV ops, each behind its own op tests (D6, decided 2026-10-09, §2.5). The mentor allows changing ggml-spacemit's code freely (it is AI-generated); its X0 failures must be fixed, not copied. Neither source is validated yet: the upstream 1198/1198 `MUL_MAT` run never used the IME weight layout (`test-backend-ops` allocates in the CPU backend's plain buffer), `test-backend-ops` has no Q4_0 case that fits the IME 32x256 layout at all (§1), and ggml-spacemit fails part of the op tests on this board (`device-type.md` X0). FlagTree AOT kernels come later and only if the mentor wants them.
6. **Attention without flash attention must work.** Under ACCEL, llama.cpp's default `-fa auto` switches flash attention off whenever the provider runs attention, because each layer's device is the CPU (confirmed in E1). The default path is therefore `MUL_MAT` on the KV cache plus masked `SOFT_MAX`; flash attention is used with `-fa on` (§2.4). With C5a (decided 2026-10-12 for M2e.1) `-fa auto` keeps flash attention when the provider runs it; attention without it stays correct on the CPU.
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
  2. One launch of 8 tiles. Each tile walks the steps; a barrier follows each step. An op takes one or more steps (`ADD`: one; `MUL_MAT`: quantize activations, then GEMM). Kernels never wait inside a step: barriers exist only between steps, which keeps the failure protocol below valid and lets the serial stand-in (no spine-runtime) run the same step list. One exception (2026-10-11, the 1-row matmuls, "1-row matmul bandwidth"): the two tiles of a pair wait for each other inside the GEMM step (`spacemit_pair_wait`); since 2026-10-14 the same turns run in the first step of one-token attention's split over the tiles (M2e.3 round 2). The failure protocol still holds because such a wait gives up once the launch has failed (or after 10 s, with an error). The host executor (the serial stand-in) counts these waits when it runs the tiles one after another, and fails the launch when the two tiles of a pair made different numbers; with `FLAGOS_SPACEMIT_TEST_THREADS=1` it runs the tiles on threads, where the waits wait.
  3. A tile that fails sets a shared flag; from then on every tile skips its remaining kernels but still reaches every barrier, so none waits for a tile that left (changed in M2d: stopping at the next barrier let tiles disagree, see "M2d design"); only a failed barrier ends a tile early. The result is `GGML_STATUS_FAILED`. No exception crosses the runtime (spert has no try/catch around tiles). No fallback after a write (Design §3.3, §12.3).
- Executor interface (`flagos-spacemit-exec.h`): launch N tiles, barrier, per-tile `{ith, nth, tcm, tcm_size, workspace}` (M2a built it without `workspace`; M2b adds it). Workspace: one shared scratch buffer per backend, grown before launch to the largest step of the split; an allocation failure returns `GGML_STATUS_FAILED` with nothing written. TCM: from M2b the tiles agree on one TCM size at the start of each launch, so they all choose the same kernel path (mixed paths would leave outputs uncomputed).
  - Implementation 1: spine-runtime (`spert::Stream`, `Grid`, `Context::sync`, `shared_buffer`). It also does the AI-thread opt-in and cross-process core arbitration, and FlagTree kernels (M3) need it anyway.
  - Implementation 2, only if M0 shows spert problems: own pinned thread pool + libspine_tcm, the upstream IME approach that now works on this board.
- Stream lifetime: **decided 2026-10-09 from the M2a benchmark: persistent (one stream per backend) is the default.** Measured on the K3 (`flagos-check-spacemit --bench`): launch 9.9 us persistent vs 24.3 us per call; per-node barrier 0.79 vs 0.83 us; idle AI-core load 0% in both (a held stream does not spin). Cost: the backend keeps the 8-core grant while it exists, so another spine-runtime process waits until it is freed; `FLAGOS_SPACEMIT_STREAM=per-call` remains as an experiment switch. Bandwidth for reference: 16M-float `ADD` 21.3 GB/s on the 8 AI cores vs 9.6 GB/s on 8 X100 threads.
- Cores: 8. An environment variable may override it for experiments only (Design §15.6 gate 9).
- TCM hygiene: release on every path. At probe, a read-only check warns (`GGML_LOG_WARN`) when blocks are held by dead threads and names `tcmrelease`.
- Never enable `GGML_CPU_RISCV64_SPACEMIT` in the same build: two TCM users in one process.
- Attention under ACCEL: with `-fa auto` (llama.cpp's default) flash attention is switched off whenever the provider runs attention, because each layer's device is the CPU (`src/llama-context.cpp:506-560`; confirmed in E1). The default path is therefore non-flash attention (`MUL_MAT` on F16/F32 KV, masked `SOFT_MAX`, `CONT`); it must be correct and fast (M2e). `-fa on` keeps flash attention. llama.cpp's own TODO at that check says the rule is wrong for other cases too; a fix that accepts an ACCEL backend running a CPU layer's attention is an upstream candidate (issue first). Decided 2026-10-12 (user): that fix goes into the fork as C5a, its own commit, with M2e.1, so the default path becomes flash attention on the provider; the path without flash attention moves to the provider only if the mentor rejects C5a (M2e.2; M2e design).

### 2.5 Kernel sources (D6: copy per milestone, decided 2026-10-09)

| Ops | Source | Milestone |
|---|---|---|
| quantized `MUL_MAT` | IME2 GEMM kernels, activation quantizers, `memcpy1d` and repack functions copied from in-tree `ggml-cpu/spacemit` (`ime2_kernels.cpp`, `rvv_kernels.cpp`, `repack.cpp` @ `ba360ef`), only the functions each milestone uses, bodies unchanged, origin noted; tiling and TCM staging ported from ggml-spacemit `ime.cpp` @ `4e782bc`: direct GEMV (1 row; ggml-spacemit uses it for Q4_0 only, we also for Q4_1, M2c design), path A (at least 113 rows and n <= 64 m, activations staged in TCM), path C (everything else). Path B (weight slabs in TCM, core-pair barrier) deferred (M2b design). `ime1_kernels.cpp` is not needed: IME1 serves X100/A60 cores, the A100 cores use IME2 (`ime_env.cpp:259-262`) | M2b, M2c |
| `MUL_MAT_ID` | same, MoE kernels | M2c, only for an MoE target |
| `ADD`, `MUL`, `SCALE`, `RMS_NORM`, `ROPE`, `SOFT_MAX`, `GLU`, `UNARY`, `GET_ROWS`, `SET_ROWS`, `CPY`/`CONT` | port from ggml-spacemit `rvv_kernels.cpp` (kernels take a small context struct). For `CPY`/`CONT` take upstream fix `f266648fa` (2026-09-16): permuted 2-byte (F16/BF16) copies called the 32-bit transpose; ggml-spacemit and our tree still have the bug, which likely explains X0's `CONT` failures (F16/BF16 only) | M2a, M2d |
| attention: non-flash path (`MUL_MAT` F16/F32 batched, masked `SOFT_MAX`) and `FLASH_ATTN_EXT` | RVV, ported from ggml-spacemit with its X0 failures fixed (both paths fail there); the fast flash-attention variant needs VLEN 1024, which the A100 cores have (M0.4). Decided 2026-10-12 (M2e design): `FLASH_ATTN_EXT` first (M2e.1), copied from in-tree `rvv_kernels.cpp`, whose one-chunk kernels ggml-spacemit shares, with three faults fixed; the non-flash path only if C5a is rejected (M2e.2) | M2e |
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
| D6 | Copy, per milestone, only the IME kernel, quantizer and repack functions used from in-tree `ggml-cpu/spacemit` (bodies unchanged, origin noted); port ggml-spacemit tiling and RVV ops, fixing them as needed (the mentor allows changing that code, 2026-10-08) | **decided** (user, 2026-10-09; §2.5) | compile the in-tree sources in place (depends on ggml-cpu internal headers; clashes with `GGML_CPU_RISCV64_SPACEMIT`) |
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
| M0.7 | Op inventory of one generation step and one 128-token prompt (`llama-eval-callback`; `GGML_SCHED_DEBUG=2` on the ggml-spacemit build to see its splits) | op matrix: op, types, shapes, count per token, which source, which milestone  **Done 2026-10-08**, below. |
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

Source: `llama-eval-callback` dumps of one generation step (1 token) and one prompt (131 tokens), flash attention on and off, from our fork's CPU build; `scripts/op-inventory.py` joins them with the GGUF tensor table (`~/m07/inventory.txt` on the board). Model: 36 layers, hidden 2560, 32 query / 8 KV heads of 128, FFN 9728, vocabulary 151936, tied embedding/output. Per graph 834 compute ops (36 x 23 + 6) and 432 views; generation and prompt processing have the same op list, only `n_tokens` differs.

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
- Every kernel needs a path for one token per call (generation) and one for batches (prompt processing).

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
| M2d (before M2c, decided 2026-10-09) | the remaining ops of a layer except attention; for Qwen3-4B: `RMS_NORM`, `MUL`, `ADD`, `ROPE`, `SET_ROWS` (KV write into the CPU-resident cache), SwiGLU, `GET_ROWS` F32; plus the norm-pinning patch C5b if accepted (design below) | per op `test-backend-ops` on the provider, all claimed cases; `flagos-check-spacemit` (KV write into a CPU buffer, a Qwen3-like layer against the CPU); split count from `sched-summary.py`; perplexity; `spacemit_check.py --milestone m2d` | every claimed case passes; about 2 handoffs per layer in generation (with C5b) and in prompt processing; perplexity within 1% of the CPU; tg and pp recorded (IME-level speed needs M2c and M2e) |
| M2e (split 2026-10-12, M2e design; M2e.1 done 2026-10-13; M2e.3 round 2 adopted 2026-10-14) | attention, reading the CPU-resident KV cache directly. M2e.1 (first): `FLASH_ATTN_EXT` with the in-tree kernels, three faults fixed, plus C5a so that the default `-fa auto` keeps flash attention. M2e.2: the non-flash path (`MUL_MAT` F16/F32 batched, masked `SOFT_MAX`, `CONT`), only if the mentor rejects C5a. M2e.3: attention during generation at streaming speed, by measurement (designed 2026-10-13; round 2, the KV positions split over the tiles, adopted 2026-10-14: tg32 at depth 4096 +20%) | `test-backend-ops -o FLASH_ATTN_EXT` (M2e.2 also `-o MUL_MAT`, `-o SOFT_MAX`); perplexity and KLD with `-fa auto` and `-fa on`; depth series to 4096 | default flags give correct output; whole layers on the K3 (a few splits per token); **tg at least the upstream IME path's and pp at least ggml-spacemit's best** (Qwen3-4B Q4_0: tg128 11.10, pp128 81.94, `build.md` §7; Qwen3-0.6B: 57.3 / 594 t/s) |
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

**Done 2026-10-10** (commits `f1cf809` C5b, `7a2b897`, `e5f69ff`; K3 results in `build.md` §7). On the K3: `spacemit_check.py --milestone m2d` 26/26 on Qwen3-4B and 28/28 on Qwen3-0.6B; every claimed `test-backend-ops` case passes with the RVV kernels and with the references; 83 splits per token exactly as designed below; tg128 5.82 -> 7.16 t/s (each removed split saved about 0.118 ms), pp128 51.08 -> 53.00; perplexity +0.21% and mean KLD 0.0024 against the CPU (IME baseline 0.0034); correct with `-fa off` and at a context of 2048; 8,000 forced-failure launches without a hang. Consequences: (1) the 4 Q4_1 `ffn_down` matmuls on the X100 cores take 36% of prompt processing, and moving them (measured with a requantized model) gives pp128 83, past the pp128 target, so M2c is next (question 10); (2) path B is not needed (pp512 equal to IME once those matmuls move); (3) attention on the X100 cores dominates generation beyond about 1k tokens of context (1.72 t/s at depth 4096, IME 6.35), which M2e addresses.

Mac, simulated device (`FLAGOS_SPACEMIT_HOST_TEST_DEVICE`, reference kernels): `test-backend-ops` passes every claimed case, ADD 54/54, MUL 46/46, RMS_NORM 52/52, ROPE 165/165, SET_ROWS 87/87, GET_ROWS 9/9, SWIGLU 12/12, and the multi-op graphs built from them, RMS_NORM_MUL_ADD 30/30, ADD_RMS_NORM 25/25, RMS_NORM_MUL_ROPE 144/144, ROPE_SET_ROWS 24/24 (F16 and MROPE cases reported as not supported). `flagos-check-spacemit` passes, including a Qwen3-like layer in one launch against the CPU backend (all rows: max NMSE 4.0e-4, bound 1e-3, from the three matmuls' activation quantization), ROPE NEOX with heads of 256 (NMSE 0), the row-op cases below (bit-exact) and 400 launches that must fail cleanly. 0 build warnings. The RVV and IME code compiles only on the K3; `scripts/rvv-syntax-check.sh` parses it for riscv64 on the Mac (0 errors; `build.md` §6). A review workflow (four areas, each finding checked by a second agent) found the executor race and the test gaps below; all fixed.

Goal: run every op of a Qwen3 layer except attention on the AI cores, so that a generated token crosses between the X100 and AI cores about twice per layer instead of about 10 times (M2b: 355 splits per token on Qwen3-4B, each handoff about 0.13 ms).

**Prerequisite for generation: norm pinning (C5b).** For batches under 32 tokens, llama.cpp pins every tensor named `norm` (the output of each RMS norm) to the backend of the layer's device (`src/llama-context.cpp:2509-2521`, marked FIXME upstream). Under ACCEL that device is the CPU, so in generation the 4 RMS norms of every layer stay on the CPU whatever the provider claims: 5 CPU splits per layer (4 norms and attention), still about 10 handoffs. Without C5b, M2d leaves generation nearly unchanged (X3b: pinning cost 30% of generation speed on this model). Prompt processing is pinned too when llama.cpp counts the model as fully offloaded (`full_offload` in the same rule): in this llama.cpp version the default `n_gpu_layers` of -1 counts as all layers (`src/llama-model.cpp:1732-1735`) unless the fit step lowers it; to be checked on the board. Patch, **implemented 2026-10-09** (user approved; a separate commit, for the mentor to keep or replace with an upstream issue, question 9): for layers whose device is the CPU, the rule no longer pins `norm` and `l_last` when an ACCEL backend supports the op; the scheduler places them with their neighbours, as `LLAMA_NO_NORM_PIN` did in X3b. 12 lines in `graph_get_cb`; nothing changes without an ACCEL device, for layers on a GPU, or when no ACCEL backend supports the op (for example BLAS, or this provider before M2d.1).

**Ops and exact claims** (counts per generated token on Qwen3-4B, from the M2b scheduler summary; F32 unless noted). Every operand may sit in the provider's buffer or in host memory (§2.3):

| Op | Count | Claim | Kernel on the K3 (ggml-spacemit `rvv_kernels.cpp` @ `4e782bc`) | Split over tiles |
|---|---|---|---|---|
| `RMS_NORM` | 145 | `eps` >= 0 (the CPU asserts the same); contiguous elements in a row, any row strides (views); in place | `forward_rms_norm_f32`: sum of squares in float vector lanes (the CPU sums in double: a rounding-level difference; the float sum overflows only for \|x\| above about sqrt(FLT_MAX / ne0)). Fixed: `eps` 0 allowed (X0 abort); tail-undisturbed accumulator (`_tu`), so rows that are not a multiple of the vector length do not depend on the core's tail policy | rows |
| `MUL` | 145 | `src1` repeatable into `src0` in every dimension (broadcast); dst shaped like `src0`; contiguous elements in the rows of `src0` and dst | `forward_binary`, when `src1`'s rows are contiguous; otherwise the reference. Claiming `MUL` makes llama.cpp load the norm weights into the provider's buffer (its load-time test is a same-shape `MUL`, `src/llama-model-loader.cpp:960-963`), which places these ops on the AI cores | rows; elements when both operands are one row of the same length (generation's norm `MUL`s and residual `ADD`s; the reference splits rows, so there one tile does them) |
| `ADD` | 72 | as `MUL` (M2a's same-shape claim, extended) | as `MUL` | as `MUL` |
| `ROPE` | 72 | forward; modes NORMAL and NEOX (Qwen3: NEOX); `n_dims` even, 0 < `n_dims` <= `ne0`, `ne0` even (the rest copied in pairs); positions I32, one per token; optional F32 frequency factors; YaRN parameters as the CPU; row strides; in place | `forward_rope_impl<float>`: per-position cos/sin table, for each tile a slice of the op's workspace (sized here: X0's abort); NEOX heads up to 128 (Qwen3) use SpacemiT's scalar loop with a stack table, which they measured faster; larger heads the RVV rotation | rows (heads x tokens) |
| `SET_ROWS` | 72 | F32 or F16 rows into F32 or F16 (the KV cache types); indices I64 or I32; dst may be the CPU-resident KV cache | `forward_set_rows`: K and V go straight into the cache, which is host memory, so no copy. Changed: an index out of range fails the op instead of aborting | columns (repeated indices cannot make two tiles write one element). With `-fa off` the V cache is transposed and written as rows of 1 element, which then all go to tile 0 (correct, slower in prompt processing); not the M2d default (`-fa auto` keeps flash attention on while attention runs on the CPU); to revisit in M2e |
| `GLU` | 36 | `SWIGLU` only; split and single-tensor (`swapped`) forms; rows as `ggml_is_contiguous_1`; dst shaped as the CPU asserts | `forward_glu_swiglu_f32`; its exp approximation is ggml-cpu's own RVV `ggml_v_expf_m2` | elements |
| `GET_ROWS` | 2 | `src` F32, indices I32 (the last layer's output rows; also admits MoE router weights, and llama.cpp's load-time test would move an F32 token embedding, not Qwen3's quantized one, into the provider's buffer) | `forward_get_rows`. Changed: an index out of range fails the op; a guard against a negative copy size when a tile gets no columns (one row of few columns overran memory) | rows; columns for one row |

Not in M2d: attention (`FLASH_ATTN_EXT`, and the `-fa off` path) stays on the CPU until M2e; the Q6_K output head and the 4 Q4_1 matmuls until M2c; the token-embedding `GET_ROWS` (Q6_K) stays on the CPU, where the graph starts. All these ops are covered well by `test-backend-ops` (default bound: NMSE 1e-7), unlike M2b's matmul.

**Kernels: ggml-spacemit's RVV ports with ggml-cpu references** (user decision 2026-10-09, replacing the portable-C++ proposal; D6, copy per milestone). `flagos-spacemit-rvv-kernels.cpp` copies the seven functions above from `rvv_kernels.cpp` @ `4e782bc` (ranges in its header), plus `rvv_expf_approx_f32m2`. Every change is marked `flagos`: the kernels take a small context (`ith`, `nth`, workspace) instead of ggml-spacemit's, which has no `sync()` here because a step cannot wait (§2.4); the fixes in the table; `copy()` from the M2b IME code, now `noinline` because its assembly does not declare the vector registers it uses. `flagos-spacemit-kernels.cpp` holds the entry points and a reference for each op, written from ggml-cpu's arithmetic (`ops.cpp`, `binary-ops.cpp`, `vec.h`). The reference runs off riscv64, on the K3 when `FLAGOS_SPACEMIT_TEST_REFERENCE=1` (tells a kernel bug from a tiling bug, as in M2b), and where the RVV version does not apply (binary ops with a non-contiguous `src1`). No hand-written RVV. ggml-cpu also has RVV code for some of these ops (its SwiGLU exp is the same algorithm), but much of it goes through its `GGML_SIMD` mapping, which works on 4 floats at a time (`simd-mappings.h:1284-1285`), an eighth of an A100 vector register (VLEN 1024); ggml-spacemit's kernels take the vector length from the hardware and were written for its A100 backend.

**Executor.** Every op is one step; `ROPE` asks for a workspace (one table per tile, `(ne0 + 16) x 4` bytes each), as `MUL_MAT` does. Two fixes:
- `graph_compute` skips empty nodes and nodes without the compute flag, as ggml-cpu and the other backends do: llama.cpp leaves unselected branches in the graph (`ggml_build_forward_select`), and a row op there would read indices that were never set. Empty nodes also kept M2b's GEMM from dividing by zero (rows 0, in prompt ubatches without an output token).
- Failure race (present since M2a, more likely now that `GET_ROWS`/`SET_ROWS` can fail on a bad index): a tile that read the shared failure flag after barrier i could see a failure from step i+1 and leave, while the others waited at barrier i+1 forever. A model of the loop with 8 spinning threads hung 11-13% of forced-failure launches. Now a tile that sees the flag skips its remaining kernels but still reaches every barrier; only a failed barrier (spine-runtime error) ends a tile early. `flagos-check-spacemit` runs 200 forced failures and 200 bad-index launches, so a hang on the K3 would show as a timeout in the runner.

**Placement after M2d (Qwen3-4B generation, with C5b):** one provider split per layer runs `attn_norm`, Q/K/V, the Q and K norms, RoPE and both KV writes; the CPU runs attention; the next provider split runs the output projection, the residual add, `ffn_norm`, gate, up, SwiGLU, down and the add, and continues into the next layer. About 75-85 splits per token (the 4 layers with a Q4_1 down projection keep 2 more each until M2c). One copy per layer remains (Q, 16 KB, to the CPU for attention); K and V go straight into the cache.

**Expected (estimates, to be measured):** Qwen3-4B generation about 125-135 ms per token (7.5-8 t/s, from 5.82), from about 280 fewer handoffs at about 0.13 ms each, with C5b; pp128 about 60-65 t/s (from 51), with C5b. IME-level speed also needs M2c (the output head is about 25 ms per token on the X100 cores) and M2e (attention).

**Tests:**
- Mac: CMake option `FLAGOS_SPACEMIT_HOST_TEST_DEVICE` (non-riscv64 only, decided 2026-10-09 over an environment variable: a test-only switch should not exist in a normal build) exposes a simulated device with 8 tiles run one after another, so `test-backend-ops -b FlagOS:SpacemiT:0` and `flagos-check-spacemit --expect-device` run every claimed case with the reference kernels before the K3 (`build.md` §6).
- `flagos-check-spacemit`: a Qwen3-like layer (norm, Q/K/V, Q and K norms, NEOX RoPE, both KV writes, output projection, residual, FFN norm, SwiGLU FFN, residual, last-token `GET_ROWS`) in one launch, with 1 and 7 tokens, the KV cache in a CPU buffer (the llama.cpp case; `test-backend-ops` allocates every tensor in the backend's own buffer) and the weights in the provider's, against the same graph on the CPU backend: every row of the output (bound 1e-3 for three chained quantized matmuls), both caches, the last-token row as an exact copy, and every intermediate finite (a NaN before a matmul would otherwise come out finite); ROPE NEOX with heads of 256, whole and half rotated, which reaches the RVV rotation (`-o ROPE` has no head above 128; only `RMS_NORM_MUL_ROPE`'s NEOX rows of 768 and 8192 reach it in `test-backend-ops`); `GET_ROWS` of one row with 1 to 2560 columns (tiles without columns); `SET_ROWS` into an F16 cache in a CPU buffer, 1 to 512 rows of 1024 and 512 rows of 1 (the transposed V cache); an index out of range fails the launch, as the first step and after another, 100 times each, and the same node without the compute flag is skipped; forced failures in steps 0 and 1, 100 times each; a one-row ADD of 4M elements and a 4001-row ADD with different rows. Any NaN or Inf counts as a failure (an NMSE of NaN passed every bound before). Mutations confirmed that these checks fail when they should (a NaN in one norm row, a tile reading the wrong row, removing the compute-flag skip).
- K3: `spacemit_check.py --milestone m2d` (`build.md` §6): claims exactly these ops; `test-backend-ops` per op and for the four multi-op graphs, with the RVV and with the reference kernels; `flagos-check-spacemit --full` with the RVV and IME kernels, and without `--full` (smaller matmul shapes) with the reference kernels; graph splits; perplexity within 1% of the CPU; pp128/tg128 against the CPU and IME; `--segv` loads the crash reporter.

**Order:** M2d.1 the binary kernel (`MUL`, `ADD` with broadcast), `RMS_NORM`, SwiGLU; M2d.2 `ROPE`, `SET_ROWS`, `GET_ROWS`; M2d.3 C5b, model runs, runner. All three are implemented together; the K3 run checks them per op.

### M2c design (2026-10-10)

**M2c.1 done 2026-10-10** (commit `6577170`; K3 results in `build.md` §7). On the K3: `spacemit_check.py --milestone m2c` 28/28 on Qwen3-4B; Q4_1 against the CPU on the converted weights max NMSE 9.7e-7 with the IME kernels; the probe gives 3.0e-7 for 1 row and for 4, so the exact 1-row branch works on the board; 75 splits per token as designed; pp128 53.0 -> 83.5 (above `ime` 79.3 and the pp128 target of 82), tg128 7.16 -> 7.22 (within the variation between days); perplexity -0.10%; mean KLD against the CPU 0.0032 with 512-token batches (M2d 0.0024, `ime` 0.0034) and 0.0020 with `-ub 1`. Consequences: (1) Q4_1 stays on the AI cores (R4, accepted with this evidence); (2) the pp128 half of M2e's exit is met, and generation now waits for the head (M2c.2) and attention (M2e); (3) generation is the more accurate path, because in batches the 4-row Q4_0 quantizer shares one activation scale across 4 tokens; (4) with 512-token batches M2c.1 is within one standard error of `ime`'s KLD, so M2c.2's accuracy is judged by its increase over M2c.1 (below). Qwen3-0.6B not run (expected 59 splits). M2c.2 (the Q6_K output head) follows.

**Order (user decision 2026-10-10, question 10):** M2c before M2e, the Q4_1 matmuls first. On the K3 (`build.md` §7, M2d) the 4 Q4_1 `ffn_down` matmuls of Qwen3-4B take 36% of prompt processing on the X100 cores, because ggml-cpu repacks only Q4_0; with them requantized to Q4_0 (speed only) pp128 went 53 -> 83, past the pp128 target, and tg128 6.95 -> 7.26. Qwen3-0.6B has 3 such layers. (Side effect: in provider mode these matmuls leave `ggml_vec_dot_q4_1_q8_1` on the X100 cores, where M2b's one CPU-backend crash happened.)

**Scope (M2c.1):** Q4_1 `MUL_MAT` with the same op check as Q4_0: weight 2-D, rows % 32 == 0, not a view, in the provider's buffer (or unallocated); activations F32 contiguous; output F32 contiguous. Any row length (Q4_1 blocks are 32 values, which is also the layout's K block). Qwen3-4B: 4 x (9728 -> 2560); Qwen3-0.6B: 3 x (3072 -> 1024).

**Route: the in-tree IME path's Q4_1 route, unchanged.** In-tree `ggml-cpu/spacemit` and ggml-spacemit both choose `q4_1_32x32_q8_0` on the A100 cores (`ime.cpp:1299-1319`; a Q4_1 variant of the 32x256 layout is commented out there as TODO), and the IME baseline's perplexity on Qwen3-4B (9.7680, CPU 9.7740) includes it:
- weights, layout `q4_1 32x32`: 32 rows interleaved, K in blocks of 32; per row and block an fp16 scale d, a uint8 zero point zp and 32 4-bit values q, a weight being d x (q - zp). 19 bytes per 32 weights (GGUF Q4_1: 20). Repack `repack_q4_1_to_q4_1_32_bl_ref` with `make_block_q4_1x32`, the scalar version: upstream calls an RVV version that divides in fp16, so near .5 it can round a zero point differently; the scalar one gives the same bytes on the Mac and the K3.
- activations: int8 per 32 values with an fp32 scale and the negated int16 sum, 38 bytes per 32 values (`quantize_a_row_i8`, `quantize_a_4row_i8`).
- kernels: `gemm_kernel_i8i4_m1` (1 row) and `gemm_kernel_i8i4_m4` (4 rows), the zero-point variants.

**The conversion is lossy (R4).** Q4_1 stores a weight as d x q + m, m an fp16 minimum; the layout needs an integer zero point, so the repack sets zp = clamp(round(-m / d), 0, 15). All weights of a block move by the same amount, at most d/2 when -m/d is within [0, 15]. Outside that range (a block of one sign) zp is clamped and the block moves further; a block of equal values (d = 0) becomes 0. Real weights have both signs in nearly every block (with an imatrix, llama.cpp's Q4_1 even forces m <= 0, `ggml-quants.c:1012, 2167`). On random weights the conversion changes a matmul's output by NMSE 5.9e-3 (`flagos-check-spacemit` info line): the shift is as large as a rounding error but common to the 32 weights, so it about doubles Q4_1's own error. The model decides: KLD against the CPU's logits, against M2d (0.00238) and the IME baseline (0.00339, which includes this conversion), and perplexity within 1%.

**Reads of a lossy layout (decided 2026-10-10, §2.3).** `get_tensor` returns the converted Q4_1: the same d and quants, m = -zp x d, the weights the kernels compute with; repacking it gives the same zero points again (checked). A partial write must cover whole blocks: it re-converts the blocks it touches from the converted values, so a block's minimum split over two writes would get a different zero point than one whole write (found in review); such a write aborts. llama.cpp writes weights whole on this device. Alternatives: keeping the GGUF bytes as well costs 39 instead of 19 bytes per 32 weights for these tensors (Qwen3-4B: +59 MiB) against single storage (D4), and a weight the scheduler copies out would compute differently on the CPU than here; not claiming Q4_1 is M2d's state. `test-backend-ops` reads the tested backend's tensors through `get_tensor` when it copies the graph to the CPU, so it would compare the kernels on the converted weights; no `MUL_MAT` case fits the layout anyway (Q4_1 cases have 16-row weights).

**Execution:** M2b's two steps (quantize, GEMM) and three paths with the same thresholds, now chosen through a table per layout (K block, row sizes, kernels) in `flagos-spacemit-ime.cpp`. For 1 row (generation) Q4_1 also takes the direct GEMV path: activations in TCM, weights from DRAM, 128 columns per call. ggml-spacemit and the in-tree path instead stage Q4_1 weights in TCM for 1 row (their path B; ggml-spacemit took the direct path only for Q4_0, where it measured faster); the `--bench` line `q4_1 matmul 9728x2560 rows 1` measures ours against the CPU and against Q4_0 at the same shape. Workspace at 512 rows x 9728: 5.9 MB.

**Kernel code (D6):** copied from in-tree @ `ba360ef`, byte-identical to ggml-spacemit `4e782bc` and upstream master (fetched 2026-10-10): `gemm_kernel_i8i4_m1`, `_m4` and the scalar `gemm_kernel_i8i4_mrow_ref`; `quantize_a_row_i8`, `quantize_a_4row_i8` and the scalar `quantize_a_nrow_i8_ref`; `block_with_zp`, `make_block_q4_1x32`, `repack_q4_1_to_q4_1_32_bl_ref`; `q8_blk_size`. About 800 lines, in the M2b file. Changes, marked `flagos`: the reference GEMM asserted that the K blocks are a multiple of 16 and dropped a partial last group of 16 (it accumulates up to 16 blocks in fp16); it now adds that group, so the provider claims any row length (the IME kernels loop per block and have no such limit); a zero guard in the reference quantizer (an all-zero block divided by zero; the RVV versions guard). Unaligned fp32 scales in the references (1-row activation blocks are 38 bytes) are read and written with `memcpy` (UBSan, found in review). New: the inverse repack `unpack_q4_1`, and the dispatch.

**The 1-row kernel runs its exact branch (decided 2026-10-10, R12).** `_m4` sums each block exactly in 32-bit integers. Upstream's active branch of `_m1`, which every generated token uses, adds each block's products in fp16 (`vmadot*.hp`) and multiplies zero point and activation sum in 16 bits (`vmul.vx` at SEW 16, keeping the low 16 bits): that wraps when 32 activations of a block are near their maximum and of one sign and the zero point is above 8, and the fp16 sum loses precision when the zero-point term nearly cancels it. `_m1` also holds the exact branch, disabled with `#if 0`: 32-bit `vmadotsu`/`vmadotu`, the zero-point product widened (`vwmul.vx`), the same arithmetic as `_m4`. The provider enables it (one line, marked `flagos`). Why: perplexity and KLD run 512-row batches, which use only `_m4`, so they say nothing about `_m1`'s arithmetic; with the exact branch generation computes what prompt processing computes. It costs about 7 more vector instructions per block in a GEMV that is memory-bound by about 10 to 1 (the `--bench` line `q4_1 matmul 9728x2560 rows 1` measures it). Alternatives: keep upstream's branch and accept the wrap (wrong results for some valid inputs; SwiGLU outputs rarely look like that, but a claim covers every input); route 1 row through `_m4` with 3 zero rows (exact on shipped code, but needs a 4-row activation block, a scratch output and a copy in every path that ends with 1 row; the fallback if the exact branch fails on the board). The probe in `flagos-check-spacemit` (zero point 15 against 32 equal activations, 1 row against 4; NMSE 4.6 if the product is 16 bits, shown by emulating it in the reference) is a check: it fails the run.

The copied reference keeps upstream's numerics: activation scales converted to fp16 (blocks with max\|x\| below about 4e-6 contribute 0) and partial sums of up to 16 blocks in fp16 (inf above 65504). This affects only the Mac and `FLAGOS_SPACEMIT_TEST_REFERENCE` runs; the IME kernels use fp32 scales (found in review).

**Fallback if the accuracy is not acceptable:** keep the exact minimum. Store zp = 0 and the fp16 minima beside the weights, and after the IME GEMM add m x (the block's activation sum) for each block; the sums are already in the quantized activations. Exact like the CPU, 2 more bytes per 32 weights, and an extra RVV pass of 1/32 of the matmul's work. Not upstream code, so it needs its own tests.

**Tests** (Mac: simulated device, reference kernels, path C only since the serial stand-in has no TCM; K3: all paths, IME and reference kernels):
- claims: Q4_1 at any row count, row length 256 and 2880 (90 blocks); refused with 16 rows, 3-D, a view, F16 activations, in a CPU buffer.
- round trip: `get_tensor` returns exactly the documented conversion; on random weights no weight moves by more than half a step; a second round trip changes nothing; partial reads and writes.
- matmuls against the CPU on the converted weights (`mm_weight` copies the provider's read-back to the CPU): 256 x 32 and 2880 x 64 at up to 128 rows, 3072 x 1024 (0.6B `ffn_down`) at 1-128 rows; `--full` adds 9728 x 2560 (4B `ffn_down`) at 1, 7, 128, 512; bound 5e-4.
- layer: the Qwen3-like layer with a Q4_1 down projection, 1 and 7 tokens, against the CPU.
- the zero-point probe (above), a check; informational: the conversion's own effect (random weights).
- mutations confirmed: without the tail fix the reference returns 0 for K = 256 (NMSE 1.0); an inverse off by one step fails the round trip and the matmuls; a 16-bit zero-point product in the reference fails the probe (NMSE 4.6).
- K3: `spacemit_check.py --milestone m2c` (default; as m2d plus the lines above), then KLD (`build.md` §6) with 512-row batches (batched: `_m4`, paths A and C) and with `-ub 1` (one token per call: `_m1`, the GEMV path, and the 1-row cases of every other provider op; measured for no milestone so far).

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

### 1-row matmul bandwidth (2026-10-10)

**Status:** done 2026-10-11 (tables in `build.md` §7). Round 1: staging the weights through TCM in 32-column slabs (`staged:32`) is 9.8% faster per token than direct reads and became the 1-row path (commit `507cb52`; tg128 7.84 -> 8.31). Round 2 (commit `2aa32d0`): the two tiles of each pair taking turns as the in-tree path B does (`paired`) is 5% faster again in generation and becomes the 1-row path, with the buffers on huge pages (+1%): tg128 8.72 -> 9.18 in one session; adopted in commit `b8187d2` and confirmed by the acceptance runner (all checks pass, perplexity unchanged, tg128 8.31 -> 8.68 against round 1's run of the runner, which reads about 5% below stand-alone llama-bench). The 1-row matmuls now match the in-tree path's (86.6 against 89.2 ms per token), so what remains to `ime` is M2e's.

**Question.** After M2c, Qwen3-4B generates at 127.6 ms per token against `ime`'s 90.7. About 99 ms of ours are 1-row matmuls streaming 2.46 GB of weights (Q4_0, Q4_1, the Q8_0 head), and every one of them measures 23.6-25.5 GB/s, whatever its format (`build.md` §7). `ime` streams the same 2.46 GB in at most 90.7 ms, so its 1-row matmuls run at 27.1 GB/s or more, about 29 if its other work takes about 5 ms (estimate). At 29 GB/s ours would take about 85 ms: about 14 ms less per token, +12% generation.

**Hypothesis.** Our 1-row kernels read the weights straight from DRAM and use each 512-byte vector load right after issuing it, so a core waits for memory about once per block: 3.1 GB/s per core is what one such load in flight gives at about 165 ns of latency (estimate). The in-tree path (path B) first copies 32-column weight slabs into TCM with `memcpy1d`, which keeps two 1 KB loads in flight, then multiplies from TCM, and staggers the two cores of each pair with a barrier so that one copies while the other multiplies. ggml-spacemit kept direct reads for Q4_0 (faster there, by its measurement) and staged the other types.

**Measurement** (test-only switch `FLAGOS_SPACEMIT_GEMV=<strategy>[:<columns>]`, read at every call; unset, the provider behaves as before):

| Strategy | What it does |
|---|---|
| `direct[:c]` (c = 128) | today's path: weights from DRAM, c columns per kernel call, calls dealt round-robin to the tiles |
| `range` | each tile one contiguous range of columns (longer sequential streams); removed 2026-10-11, no gain |
| `staged[:c]` (c = 64; 32 from 2026-10-11, the default in round 1), `staged-range[:c]` | c columns at a time copied into TCM with `memcpy1d`, then multiplied there (path B without its pair barrier); slabs shrink to fit the 384 KiB TCM. `staged-range` removed 2026-10-11 |
| `paired[:c]` (c = 32; added in round 2, the default since) | `staged` with path B's pair barrier: the two tiles of a pair take turns, one multiplying while the other copies |
| `copy[:c]`, `compute[:c]` | staging's copies alone (the ceiling for staging: DRAM to TCM) or its multiplications alone (the kernels' speed from TCM); wrong results, timing only. `compute` removed after round 2; `copy` stays as the strategy check's control |
| `prefetch[:c]` (c = 32), `prefetch-range[:c]` | direct, each call first hinting the tile's next chunk into the caches with Zicbop `prefetch.r`; removed 2026-10-11 (41-58% slower) |
| `@t` (added 2026-10-11) | only the first t tiles take part, the others wait: how the bandwidth depends on the number of tiles streaming at once |

- Correctness: every strategy returns the same bytes as direct, checked by `flagos-check-spacemit` at 6 shapes (partial chunks, ranges and slabs; slabs that shrink; all four weight types). On the Mac it gives the serial stand-in a TCM (new test-only `FLAGOS_SPACEMIT_TEST_TCM`), which also runs the 1-row path and path A there for the first time. A copy-only control must differ, proving the strategies ran. Mutations confirmed: a staged strategy that skips a slab fails; without the TCM stand-in the control fails.
- Speed: `flagos-check-spacemit --bench` times every strategy at the 7 shapes of a token's 1-row matmuls and prints per-token totals, weighted by how often a token runs each shape (36 q; 72 k and v; 36 attention outputs; 72 gate and up; 32 Q4_0 and 4 Q4_1 down; the head). Each run is its own launch, so the totals include about 250 launches the model does not pay; the comparison between strategies is what counts.
- Reference: `docs/flagos-k3/scripts/ime-gemv-bench.cpp` times the upstream IME path (`ime`) at the same shapes through ggml-cpu's `CPU_RISCV64_SPACEMIT` buffer type, each result also compared with ggml-cpu's plain path to show that the IME path ran.

**Decision rule.** Adopt the fastest strategy as the 1-row path if its per-token total is at least 5% below direct (its results are identical, so accuracy cannot change), confirm with tg128 against today's numbers, then remove the switch and the strategies that lost. If none wins: `copy` about as slow as direct means DRAM is the ceiling, `ime`'s advantage lies elsewhere (no handoffs; attention on the AI cores), and M2e is next; `ime-gemv-bench` clearly faster than every strategy points to path B's pair staggering, which needs a wait inside a step: decide then.

**Results (K3, 2026-10-11; per-token totals of the 7 shapes, table in `build.md` §7):** direct 102.8 ms (23.9 GB/s); `staged:32` 92.7 ms (26.5 GB/s, -9.8%), the only strategy past the 5% bar; `staged:64` 98.1; `staged:256` 114.8; `range` 103.8; `direct:512` 106.0; prefetch hints 145-163; staging's copies alone 94.5, its multiplications alone 30.4.
- `staged:32` gains on q (-10%), the attention output (-11%), gate and up (-14.5%; 41% of a token's weight bytes) and the head (-10%), and loses a little on the Q4_1 down projection (+4%). Longer slabs lose: a tile then spends longer in each copy.
- Staging is bound by its copies: `staged:32` takes about the time of the copies alone, and the kernels multiply a slab from TCM about three times faster than DRAM delivers it (30.4 ms per token), so their work hides while other tiles copy.
- Fewer tiles streaming at once get more bandwidth in total: `direct:512` leaves only 5 of 8 tiles busy at the 2560-column shapes and still beats balanced 8-tile direct reads (down projection 533 vs 607 us, 27 vs 23 GB/s; attention output 217 vs 253). Staging likely wins the same way: a tile multiplying for about a third of its time leaves about 5-6 tiles streaming. The in-tree path B staggers the two cores of each pair with a barrier. Measured next with `@t`: true for direct reads, hardly for copies ("Results after adoption" below).
- Zicbop prefetch hints cost 41-58%: dropped.
- k and v (1.5 MB each) stay in a cache across the benchmark's 30 repetitions (30 us, 49 GB/s); a token reads them once, so that column does not represent the model.

**Decision (2026-10-11, by the rule above):** `staged:32` is the 1-row path. Direct reads remain the fallback where a 32-column slab does not fit the TCM (none of Qwen3's shapes). `range`, `staged-range` and the prefetch strategies are removed; `FLAGOS_SPACEMIT_GEMV` keeps `staged`, `direct`, `copy`, `compute` and `@t` for measurements. Results are identical bytes, so accuracy cannot change; checked at 6 shapes, and on the Mac the whole `flagos-check-spacemit` and `test-backend-ops` also pass with a TCM stand-in, where the new path runs everywhere. Expected: about 10 ms less per token, tg128 7.84 -> about 8.5.

**Results after adoption (K3, 2026-10-11; tables in `build.md` §7):**
- In the model: tg128 7.84 -> 8.31 (+6.0%, 7.2 ms less per token against 10.4 predicted; k and v, cached in the benchmark, explain part of it); `llama-completion` 7.80 -> 8.64; perplexity and prompt processing unchanged.
- Fewer tiles: `staged@6` ties `staged` per token (92.9 ms). It is faster on the long matmuls (head -8%, gate and up -3%, down -3%) and slower on q, the attention output and the cached k and v; without k and v about 2.5% faster, under the 5% bar, so the path stays on 8 tiles. `@5` and `@4` lose (101.0, 111.3): each tile multiplies a larger share. Direct reads gain from fewer tiles (`direct:32` on 5 or 6 tiles: 94.9 against 101.7) but stay behind staging. Copies alone barely depend on the number of tiles: 2 tiles 25.1 GB/s, 6 26.7, 8 26.4 (4: 23.3, unexplained; tiles 0-3 are CPUs 8-11). So staging wins mainly because `memcpy1d` reads DRAM faster than the kernels' own loads, and only a little through fewer streams.
- The IME reference: 82.4 ms per token (29.9 GB/s), 11% below `staged`, at 28.4-30.4 GB/s on every shape where ours range from 23.3 to 27.5.
- Huge pages: upstream allocates its weights with `mmap` and `madvise(MADV_HUGEPAGE)` (`ggml-cpu/spacemit/spine_mem_pool.cpp:447-480`), the provider with `posix_memalign` (`flagos-spacemit.cpp:139`), and transparent huge pages are madvise-only on this board (`build.md` §7). Upstream on 4 KiB pages (`SPACEMIT_MEM_BACKEND=POSIX`) takes 84.4 ms per token (+2.0) and generates at 10.80 instead of 11.05 t/s (+2.1 ms per token): pages are worth about 2 ms per token, a fifth of the gap.
- On equal pages upstream is still 8.5 ms per token faster (84.4 vs 92.9): gate and up 3.8 ms, down 3.2, the head 1.3, the attention output 0.8; q equal. Ruled out as causes: the copy routine (the same `memcpy1d`, inline assembly for each vector length), the slabs (32 columns dealt round-robin in both), the activation (quantized by all tiles, then held in TCM, in both), and our TCM or runtime as such: with 6 tiles the head streams as fast as upstream's (29.5 vs 29.6 GB/s). The one known difference left is path B's pair barrier (`ggml-cpu/spacemit/ime.cpp:455, 471, 485`), which changes how the 8 cores share DRAM and how each matmul starts. Not proven.

**Round 2 (user decision 2026-10-11, the "pair staggering" case of the decision rule; commit `2aa32d0`):** one more round before M2e.
1. Huge pages: the provider's buffers are mapped from a 2 MiB boundary with `mmap` and `madvise(MADV_HUGEPAGE)` (`flagos-spacemit.cpp`, `spacemit_map_huge`; Linux only, other hosts keep `ggml_aligned_malloc`). Test-only `FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1` keeps the default pages, for a comparison in one session. `flagos-check-spacemit` reports how much of a 64 MiB buffer the kernel put on huge pages (line `info     huge pages`, from `/proc/self/smaps`). Expected about 2 ms less per token.
2. `paired[:c]`: path B's schedule (`ggml-cpu/spacemit/ime.cpp:435-494`). The two tiles of a pair (CPUs 8+2p and 9+2p, which share their TCM blocks' affinity) take turns: the even tile copies its first slab while the odd one waits, then in every round one multiplies while the other copies. Its waits (`spacemit_pair_wait`, `flagos-spacemit-exec.cpp`) are the first inside a step (§2.4): both tiles of a pair wait 1 + rounds times by construction (rounds = the even tile's number of slabs); a wait gives up once the launch has failed, so a partner that skipped the step cannot hang it, or after 10 s, with an error; the serial executor, whose tiles run one after another, only counts the waits and fails the launch when a pair's counts differ after a step. Expected up to 8.5 ms less per token, if the pair barrier explains the rest of the gap.
3. The benchmark reads every weight from DRAM, as the model does: each run takes the next of enough copies of the weight to leave the caches (32 MB together: 22 copies of k and v, 6 of q and the attention output, 3 of the FFN matrices, 1 head), in `flagos-check-spacemit` and in `ime-gemv-bench.cpp`, so that the per-token totals predict the model. New flag `--bench-gemv`: of the benchmarks only the 1-row table.

Decision rules, fixed before measuring:
- Huge pages stay the default unless tg128 is lower with them than without (same session). They cost nothing else (each buffer rounds up to whole 2 MiB pages).
- `paired` becomes the 1-row path if its per-token total is at least 5% below `staged`'s (both on huge pages) and tg128 confirms it (`FLAGOS_SPACEMIT_GEMV=paired` against the default, same session). Adopting it makes the pair wait permanent: §2.4 then allows exactly this wait, and the Mac needs a check with tiles running concurrently (the serial executor only counts).
- Otherwise: record the results, remove `paired` and the other test strategies that lost, and go to M2e.

Mac, 2026-10-11: the 1-row strategy check gives 78/78 identical to direct (13 strategies at 6 shapes; the four `paired` variants cover a pair whose odd tile has one slab fewer, pairs without work and a tile without a partner); `test-backend-ops`: the 12 op sets pass with and without the TCM stand-in. Mutations caught: the odd tile skipping its wait in a round without a slab (7 runs fail, "tiles 0 and 1 made 5 and 4 pair waits"; on the AI cores this would hang); the odd tile not copying its first slab (24 runs differ).

**Round 2 results (K3, 2026-10-11; tables in `build.md` §7):**
- Every check passes on the AI cores, `paired` with real pair waits (78/78 identical to direct, no timeouts). Huge pages as intended: 64 of 64 MiB of a test buffer, 0 with the switch.
- Generation, back to back, two rounds interleaved: `paired` 9.20, 9.17, 9.18 against `staged` 8.77, 8.69, 8.75 on huge pages, +5.0% (114.4 -> 108.9 ms per token); +4.6% on 4 KiB pages. Huge pages: +0.7% for `staged` (within the noise), +1.2% for `paired`. The model's weights do lie on huge pages (AnonHugePages 2374 MiB during generation; the weight buffer is 2350 MiB).
- Benchmark, every weight from DRAM: on 4 KiB pages `paired` takes 95.0 ms per token against `staged`'s 101.7 (-6.6%, past the 5% bar). On huge pages `paired` (85.0) is the fastest (`staged@6` 96.2, `direct` 94.8, the copies alone 90.0); the huge-page `staged` row is not valid, as its runs paid for faulting in their outputs' huge pages (`build.md` §7); the clean rows put it at about 92 (`paired` -7.5%). `paired:64` (86.9) and `paired@6` (87.0) do not beat `paired`. The largest gain is at the down projection (-13% on huge pages: long rows, 10 slabs per tile), which fits the reading that the turns help most where copies and multiplications would otherwise stay in step; that the two cores of a pair compete when both copy is a possible mechanism, not measured.
- The two page settings of the benchmark ran minutes apart, and on the shared board that moved the totals more than the pages do (`paired` 95.0 -> 85.0 there, +1.2% in generation); only the interleaved tg128 runs measure the pages.
- Upstream: its 1-row matmuls, weights rotated as ours, take 89.2 ms per token (82.4 unrotated: the old benchmark also served mid-sized shapes partly from caches). `paired` on huge pages takes 85.0, faster or equal at every shape except the Q4_1 down projection (+5%). The 1-row gap to the in-tree path is closed; what remains to `ime`'s tg128 (about 108.9 against 90.7 ms per token) is outside the 1-row matmuls: the 74 handoffs per token, attention on the X100 cores and the other ops (M2e).

**Decision (2026-10-11, by the rules above):** `paired` (32 columns, 8 tiles) is the 1-row path and huge pages stay. Implemented in commit `b8187d2`, confirmed on the K3 (`build.md` §7, "Round 2 adopted"):
- `flagos-spacemit-ime.cpp`: `paired` is the default. `FLAGOS_SPACEMIT_GEMV` keeps `staged` and `direct` for comparisons, `copy` as the strategy check's control and `@t` (which also gives a tile without a partner); `compute` is removed.
- The pair wait is permanent (§2.4). The host executor (formerly the serial executor) runs the tiles of each launch on threads with `FLAGOS_SPACEMIT_TEST_THREADS=1`: each tile with its own TCM stand-in, a barrier after each step, the AI cores' failure protocol, and waits that wait. `flagos-check-spacemit` runs its strategy check on the host both ways, so that the pair waits are counted and also really wait.
- Huge pages only for buffers of at least 2 MiB: a smaller one gains nothing and still takes a whole huge page. The benchmark runs every copy once before the timing, and times `paired`, `staged` and `direct`.
- `flagos-check-spacemit` restores the run's own settings after its strategy check and benchmark. Before, a `FLAGOS_SPACEMIT_TEST_TCM` given to the whole run was cleared there, so the row and layer checks after them ran without the TCM stand-in.

Mac (adoption, 2026-10-11): `flagos-check-spacemit --expect-device` passes as it is, with the TCM stand-in, and with the TCM stand-in and every launch on threads. The strategy check gives 156/156 (78 with the tiles in turn, 78 on threads); with the TCM stand-in the layer check also runs its 1-row matmuls on the paired path, several in one launch. ThreadSanitizer with every launch on threads: all checks pass, 0 reports. `test-backend-ops`: the 12 op sets pass in all three modes. Syntax: the spine-runtime executor, the huge-page branch and the riscv64 kernels, 0 errors, no new warnings.

**Open (2026-10-12): generation right after heavy load.** The runner's tg128 reads about 5% below stand-alone llama-bench (8.68 against 9.2). Not its flags or its order of tests (`build.md` §7): on a rested board its flags give 9.23 (9.25 with 30 s of rest before each test), a second context in one process 9.16 and 9.20, and after a prompt test the 1-row strategies rank as when run alone (`paired` 9.23, `staged` 8.76, `direct` 8.69), so the path has its TCM there. Both slow measurements started right after minutes of full load on the X100 cores (the runner's CPU-only perplexity, which runs right before its llama-bench; the `cpu` depth runs). The provider still runs attention, the embedding lookup and the thread that drives the AI cores there, while `ime`, all on the A100 cores, was not slowed. Most likely heat, possibly another user's job; to confirm: read the clocks and temperatures, load the X100 cores for about 3 minutes, then measure tg128 at once and after a minute's rest. If it is heat, the runner should run llama-bench before the perplexity runs (its tg128 then reads about 5% above earlier runner runs). M2e moves attention to the AI cores, which should make the provider much less sensitive to the X100 cores' state. Observed 2026-10-13 (M2e.1): the runner's tg128 (11.27) now matches `llama-completion`'s (11.34) in the same run, consistent with the X100 cores' state after the heavy runs as the cause.

**On the K3, confirming the adoption:**
```bash
cd ~/llama.cpp-flagos && git -c safe.directory=$PWD pull
python3 ggml/src/ggml-flagos/providers/spacemit/tools/spacemit_check.py --milestone m2c2 --build --model ~/models/Qwen3-4B-Q4_0.gguf --segv ~/crash/segv.so > ~/m2c2-paired.txt 2>&1; tail -1 ~/m2c2-paired.txt
grep -E '^\[FAIL\]|huge pages|^\[(INFO|PASS)\] (graph|generation|perplexity|llama-bench)' ~/m2c2-paired.txt | cut -c1-220
awk '/^bench    gemv q/ { k = $3 ":" $4; if (!(k in seen)) { seen[k] = 1; col[++nc] = k } if (!($5 in sseen)) { sseen[$5] = 1; row[++nr] = $5 } t[$5, k] = $6 } /^bench    gemv per/ { tot[$10] = $11 } END { printf "%-17s", "us"; for (i = 1; i <= nc; i++) printf " %15s", col[i]; printf " %9s\n", "token ms"; for (j = 1; j <= nr; j++) { printf "%-17s", row[j]; for (i = 1; i <= nc; i++) printf " %15s", t[row[j], col[i]]; printf " %9s\n", tot[row[j]] } }' build-flagos/m2c2-logs/bench.out; grep '^gemv' build-flagos/m2c2-logs/check-spacemit.out | cut -c1-220
```

**On the K3, round 2:**
```bash
cd ~/llama.cpp-flagos && git -c safe.directory=$PWD pull
cmake --build build-flagos --target ggml-flagos flagos-check-spacemit llama-bench -j 8 2>&1 | tail -1
# the checks (paired with real pair waits) and the 1-row table, on huge pages and on 4 KiB pages
~/tcmtest/tcmrelease --apply > /dev/null && ./build-flagos/bin/flagos-check-spacemit --expect-device --bench-gemv > ~/gemv2-huge.txt 2>&1; tail -1 ~/gemv2-huge.txt
FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1 ./build-flagos/bin/flagos-check-spacemit --expect-device --bench-gemv > ~/gemv2-small.txt 2>&1; tail -1 ~/gemv2-small.txt
grep -hE '^(gemv|info     huge)' ~/gemv2-huge.txt ~/gemv2-small.txt
# generation, interleaved: staged and paired, huge and 4 KiB pages
for r in 1 2; do for c in "" FLAGOS_SPACEMIT_GEMV=paired FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1 "FLAGOS_SPACEMIT_GEMV=paired FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1"; do echo "${c:-default}: $(env $c ./build-flagos/bin/llama-bench -m ~/models/Qwen3-4B-Q4_0.gguf -t 8 -p 0 -n 128 -fa 1 -mmp 0 -r 3 2>/dev/null | grep tg128 | awk -F'|' '{print $(NF-1)}')"; done; done
# the model's pages, read while it generates
cat /sys/kernel/mm/transparent_hugepage/defrag; ./build-flagos/bin/llama-bench -m ~/models/Qwen3-4B-Q4_0.gguf -t 8 -p 0 -n 512 -fa 1 -mmp 0 -r 1 > /dev/null 2>&1 & sleep 45; grep -E '^(Rss|AnonHugePages)' /proc/$!/smaps_rollup; wait
# the IME reference with the weights rotated (the IME build itself is unchanged)
g++ -O2 -std=c++17 -pthread -Iggml/include docs/flagos-k3/scripts/ime-gemv-bench.cpp build-ime/bin/libggml.so build-ime/bin/libggml-base.so build-ime/bin/libggml-cpu.so -Wl,-rpath,$PWD/build-ime/bin -o ~/ime-gemv-bench
~/tcmtest/tcmrelease --apply > /dev/null && OMP_THREAD_LIMIT=8 ~/ime-gemv-bench 2>&1 | grep '^ime' | tee ~/ime-gemv2.txt
```

**On the K3, round 1:**
```bash
cd ~/llama.cpp-flagos && git -c safe.directory=$PWD pull
cmake --build build-flagos --target ggml-flagos flagos-check-spacemit -j 8 2>&1 | tail -1
~/tcmtest/tcmrelease --apply > /dev/null && ./build-flagos/bin/flagos-check-spacemit --expect-device --bench > ~/gemv-bench.txt 2>&1; tail -1 ~/gemv-bench.txt
awk '/^bench    gemv q/ { k = $3 ":" $4; if (!(k in seen)) { seen[k] = 1; col[++nc] = k } if (!($5 in sseen)) { sseen[$5] = 1; row[++nr] = $5 } t[$5, k] = $6 } /^bench    gemv per/ { tot[$10] = $11 } END { printf "%-17s", "us"; for (i = 1; i <= nc; i++) printf " %15s", col[i]; printf " %9s\n", "token ms"; for (j = 1; j <= nr; j++) { printf "%-17s", row[j]; for (i = 1; i <= nc; i++) printf " %15s", t[row[j], col[i]]; printf " %9s\n", tot[row[j]] } }' ~/gemv-bench.txt; grep '^gemv' ~/gemv-bench.txt
cmake --build build-ime --target ggml ggml-cpu -j 8 2>&1 | tail -1      # the IME build, current tree
g++ -O2 -std=c++17 -pthread -Iggml/include docs/flagos-k3/scripts/ime-gemv-bench.cpp build-ime/bin/libggml.so build-ime/bin/libggml-base.so build-ime/bin/libggml-cpu.so -Wl,-rpath,$PWD/build-ime/bin -o ~/ime-gemv-bench
~/tcmtest/tcmrelease --apply > /dev/null && OMP_THREAD_LIMIT=8 ~/ime-gemv-bench | tee ~/ime-gemv.txt
# the same on 4 KiB pages, and upstream's generation with and without huge pages
~/tcmtest/tcmrelease --apply > /dev/null && SPACEMIT_MEM_BACKEND=POSIX OMP_THREAD_LIMIT=8 ~/ime-gemv-bench 2>&1 | grep -E '^ime|mem_backend'
for b in HPAGE POSIX; do ~/tcmtest/tcmrelease --apply > /dev/null; echo "$b $(SPACEMIT_MEM_BACKEND=$b ./build-ime/bin/llama-bench -m ~/models/Qwen3-4B-Q4_0.gguf -t 8 -p 0 -n 128 -fa 1 -mmp 0 -r 3 2>/dev/null | grep tg128)"; done
```

### M2e design (2026-10-12)

**Status:** designed 2026-10-12; both decisions taken by the user the same day (end of this section): C5a in the fork as its own commit, and M2e.1 first. The M2e row of the milestone table now lists M2e.1, M2e.2 (the graph without flash attention, only if the mentor rejects C5a) and M2e.3. **M2e.1 done on the K3 2026-10-13** (implemented 2026-10-12, commits `64e6cbd` (C5a) and `715b9bc`; results below and in `build.md` §7). With L3 and L4 (ggml-spacemit's tiled kernel, vector mask counts), the final run the same day (`build.md` §7, "M2e's final state"): 37/37, accuracy unchanged to the last digit, pp128 92.27 and pp512 at depth 4096 34.56 against `ime`'s 79.13 and 25.13, generation unchanged (tg32 at depth 4096 6.49, `ime` 6.40). Qwen3-0.6B the same day (34/34): pp128 654.10 and tg128 55.60 against `ime`'s 553.20 and 54.25 in the same session, so the exit holds for it too; M0's tg figure 57.3 is not reached, but `ime` measured 5% under its own. Perplexity +0.89% against the CPU (bound 1%), the thinnest margin so far; the upstream IME path has the same mean KLD on 0.6B (0.0140 against our 0.0147, within one standard error), so the gap is the IME approach's own. With 8 chunks the perplexity ratio on 0.6B carries ±0.74%, too coarse for a 1% bound: on 0.6B, KLD is the measure.

**M2e.1 implementation (2026-10-12):**
- Files:
  - `flagos-spacemit-rvv-attn.{h,cpp}`: the copied kernels (in-tree `rvv_kernels.cpp` @ `81b0d88`; every change marked `flagos`).
  - `flagos-spacemit-attn.{h,cpp}`: the claim, the per-tile dispatch and the portable reference.
  - `flagos-spacemit-ops.cpp`: the one-step op.
  - `provider.cmake`: the two files get the vector ISA, like the other kernels.
  - `tests/check_spacemit.cpp` and `tools/spacemit_check.py`: the checks and the runner below.
  - C5a in `src/llama-context.cpp:533-540`.
- Fixes in the copied code:
  - (a) to (c) as designed.
  - Three more found while copying and reviewing: (d) the tiled kernel applied the softcap to the padding of a short last KV tile, so without a mask the padding got weight (ggml-cpu's tiled kernel has the same fault; the provider then differs from the CPU there by an estimated 1e-5); (e) the K-transpose asm overwrote v0-v7 and memory without declaring them (now declared, the function not inlined); (f) the same asm advanced by VLMAX instead of the vl each strip gets, wrong for head sizes 65-127 if the A100 returns vl < VLMAX (a Python model of the asm checks all shapes under three legal vl policies).
- Deviations from the design:
  - Without TCM (or with less than the tiled kernel's 160 KiB) every row takes the 1-row kernel; there is no workspace slice.
  - ALiBi and softcap are claimed already. If any of their `test-backend-ops` cases fails on the K3, the claim drops `max_bias > 0` and `logit_softcap != 0`.
- Mac results (host test device, reference kernels; three modes: serial, with the TCM stand-in, and on threads):
  - `test-backend-ops -o FLASH_ATTN_EXT`: the provider claims 1324 of 5143 cases (exactly the F16-KV ones with head sizes up to 128) and passes all 1324; the 12 earlier op sets unchanged.
  - `flagos-check-spacemit`:
    - attention against the CPU with the cache's layout (Qwen3-4B and 0.6B heads; 1 token at n_kv 256-4096; 7, 99 and 128 tokens): max NMSE 1.2e-5 (bound 5e-4), worst row 1.7e-5 (bound 5e-3);
    - the probes for (a) (masks at -30 and -120; -30 catches the fault only in an F16 accumulator, as on the K3), (b) and (c) pass;
    - the layer check now runs attention on the provider (a whole layer in one launch): max NMSE 5.4e-4 (bound 1e-3; 4.0e-4 before, from the FFN getting different data, not from attention);
    - mutations of the reference are each caught (mask ignored, one ALiBi slope per group, maxima starting at 0, a tile skipped, the wrong K head, the wrong head for a last row).
  - ThreadSanitizer on threads: 0 reports. UBSan: none in provider code.
  - C5a, on a 2-layer random Qwen3 model (Q4_0): `-fa auto` logs "Flash Attention enabled" and all attention nodes run on the provider, reading the CPU cache in place. With C5a reverted (a scratch copy), llama.cpp logs "set to disabled" and the graph goes from 6 to 10 splits. The 6 come from the toy model's shapes (a Q4_0 embedding, `ffn_down` with K 384), not from attention. The provider's attention matches the CPU's attention there to NMSE 1e-12.
  - riscv64 syntax check: 0 errors. 7 distinct warnings in copied code (sign-compare, an unused variable), kept unchanged as M2b kept its 2.
- Review (five lenses, each finding checked by two verifiers): 6 findings, all confirmed and fixed. These were (f) above, the test bound per row besides the whole-tensor NMSE, an unaligned tiled case (99 tokens after 37), and three in the runner: the C5a check now also requires attention on the provider (splits below the layer count), `--depth` reports missing depths, and the KLD runs and pp512 at depth 4096 were added.
- Runner milestone m2e (now the default):
  - checks the claimed op set, `test-backend-ops` including `FLASH_ATTN_EXT` (also with the reference kernels), and the attention lines of `flagos-check-spacemit`;
  - the model run: C5a keeps flash attention on the provider, and a token takes 2 splits (FAIL otherwise);
  - perplexity with `-fa on` and `-fa auto`, each within 1% of the CPU;
  - mean KLD against the CPU at the default batch and at `-ub 1`, compared with M2c.2's by hand;
  - `--depth`: tg32 at depths 0, 1024 and 4096, and pp512 at depth 4096, for the provider, the CPU and the IME build.
- K3 results, 2026-10-13 (`spacemit_check.py --milestone m2e --depth` on Qwen3-4B: 37/37; `build.md` §7):
  - GCC assembled the changed asm. `test-backend-ops -o FLASH_ATTN_EXT` passes all 1324 claimed cases with the RVV kernels (ALiBi, softcap and head sizes 40-128 included, so fix (f) holds and nothing has to leave the claim). `flagos-check-spacemit`, its attention checks and the three fault probes pass with the RVV and with the reference kernels.
  - `-fa auto` keeps flash attention on the provider (C5a) and a token takes 2 graph splits, from 74.
  - Perplexity 9.782 against the CPU's 9.774 (+0.08%), the same with `-fa on` and `-fa auto`. Mean KLD 0.00335 (512-token batches) and 0.00205 (`-ub 1`), both within one standard error of M2c.2's.
  - tg128 11.27 (`ime` 11.03, before 8.68), pp128 87.70 (`ime` 79.37). tg32 at depth 0 / 1024 / 4096: 11.25 / 9.53 / 6.48 (`ime` 11.09 / 9.34 / 6.34; before 9.70 / 6.74 / 1.80). pp512 at depth 4096: 24.87 (`ime` 25.13).
  - Every exit criterion is met. So are the project's targets for Qwen3-4B (tg128 >= 11.1 and pp128 >= 82, M0), and the M2e row's target "tg at least the upstream IME path's and pp at least ggml-spacemit's best".
  - What the benchmark shows for what comes next:
    - In generation, attention (the one-chunk kernels) reads K and V at 9.7 GB/s at n_kv 4096 (62 ms per token), a third of the 1-row matmuls' 28.2 GB/s: M2e.3's headroom, about 62 -> 23 ms per token, tg32 at depth 4096 about 6.5 -> 8.7 (estimate).
    - In prompt processing at depth, attention (the tiled kernel) is compute-bound at about 88 GFLOP/s (128 tokens at n_kv 4096: 98 ms per layer), about 70% of pp512 at depth 4096. A faster tiled kernel is a candidate of its own: the copied kernel rereads and re-transposes K for every 64-token query tile and for each query head of a KV head, and the IME reaches 600-890 GFLOP/s on matmuls. Hardening suggested by the review, applied 2026-10-13 at the user's request: C5a also requires that the ACCEL device accepts the layer's buffer type (the KV cache's), so an ACCEL backend that could not read the cache in place keeps the old behaviour. Checked on the Mac's toy model: with the provider as is, flash attention stays on (6 splits); with the provider made to refuse host buffers (a temporary change), llama.cpp logs "set to disabled" again (30 splits).

**Goal:** attention on the AI cores, the last part of a Qwen3 layer on the X100 cores. It decides generation beyond about 1k tokens of context (`build.md` §7, 2026-10-12): from depth 0 to 4096 a Qwen3-4B token takes 452 ms more on the provider (attention on the X100 cores, about 110 us per position) and 66 ms more on `ime` (16 us); tg32 at depth 0 / 1024 / 4096: provider 9.70 / 6.74 / 1.80, `ime` 11.05 / 9.34 / 6.41. Attention on the CPU also splits each layer in two: 72 of the 74 handoffs per token.

**Facts (verified in code):**
- Two graphs (`src/llama-graph.cpp:2500-2633`). With flash attention: one `FLASH_ATTN_EXT` reading q (F32, a permuted view of the RoPE output), K and V (F16 views of the cache) and the mask (F16, `[n_kv, n_tokens, 1, n_stream]`, `:28-45`), precision F32; its output reaches the output projection through a reshape. Without: `MUL_MAT` (F16 K view x F32 q, broadcast over the query heads of each KV head), `SOFT_MAX` (F32 mask, scale), `MUL_MAT` (F16 V, which the cache then stores transposed, x F32), `CONT`. With an F16 cache nothing else sits between the KV writes and the output projection (the Hadamard rotations exist only for quantized caches, `src/llama-kv-cache.cpp:319-336`).
- `-fa auto` (`src/llama-context.cpp:504-562`) builds a 1-token graph and turns flash attention off when a `FLASH_ATTN_EXT` node's device differs from its layer's. Under ACCEL every layer's device is the CPU, so flash attention stays on only while the CPU runs it. Once the provider claims `FLASH_ATTN_EXT`, `-fa auto` switches to the graph without flash attention, which the provider does not run: default runs would get attention on the X100 cores without flash attention, slower than today. So M2e needs C5a or that graph on the provider.
- Cache (`src/llama-kv-cache.cpp:1249-1300`): a cell holds every KV head (Qwen3-4B: 8 x 128 F16, 2 KiB), so a head's K or V rows are 256 B long and 2 KiB apart; n_kv is padded to a multiple of 256 (`:1233-1247`), the padding masked with -inf.
- ggml (`ggml/src/ggml.c:5402`): the mask is F16, contiguous, and broadcasts over heads and streams; q and K have the same dimension 3; only heads broadcast (GQA).
- ggml-cpu (`ggml-cpu/ops.cpp:8468-9215`): for one token per call (n_kv >= 512) it splits the KV positions over the threads and merges partial results; its one-row kernel accumulates F16 V in F16; from 64 rows per head it uses 64 x 64 tiles accumulated in F32.
- The in-tree IME path runs its own kernels for VLEN 1024 (`ggml-cpu/spacemit/ime.cpp:1151-1227`; `rvv_kernels.cpp:33-1628`, about 1100 lines with their helpers; ggml-spacemit `4e782bc` has the same one-chunk kernels, its tiled kernel also trims masked tails): F32 q, F16 K and V, head sizes up to 128. Rows (query head x token) are split statically over the threads. With one token per call a thread takes the 4 (or 2) query heads of one KV head and stages 64 positions at a time in TCM (K transposed by strided stores): scores in F32, V accumulated in F16. From 64 rows per head, 64 x 64 tiles in TCM accumulated in F32. These kernels are what `ime`'s 66 ms at depth 4096 measures.
- Three faults in the one-chunk kernels, found by reading:
  - (a) `float M[QLEN] = { -INFINITY }` (`rvv_kernels.cpp:669`) starts rows 1-3 at 0 instead of -inf. A row whose scores all lie far below 0 is then exponentiated against 0, its weights underflow in the F16 accumulator and its output becomes 0.
  - (b) The dispatcher reads `mask->ne[2]` without a null check (`:1202-1210`), so a 1-token call without a mask crashes.
  - (c) The 2- and 4-row kernels apply the first row's ALiBi slope to every row (computed from its head, passed at `:1269, 1290`, applied at `:710, 833`), though each query head has its own: wrong with ALiBi and GQA.
  - (b) and (c) fit X0, where ggml-spacemit missed its ALiBi cases by NMSE about 0.006 and crashed right after the last head-size-40 ALiBi case, where `test-backend-ops` (`tests/test-backend-ops.cpp:9579`) starts the cases without a mask (`device-type.md` X0). Qwen3 meets none of them in practice: no ALiBi, always a mask, scores rarely all far below 0.
- Speed of that one-chunk kernel in generation: `ime`'s 16 us per position per token is 0.44 us per position per layer and thread, about 900 cycles. That is about one DRAM round trip per position: each position's K and V rows are loaded and then used at once, and `memcpy2d` copies row by row (estimate, not measured). Streaming the same 147 KB per position and token at the 27 GB/s our 1-row path reaches would take 5.5 us.
- `test-backend-ops` (`:6819-6935, 9579-9660`): NMSE bound 5e-4. With F16 K and V at head sizes 40-128: masks on and off, sinks, ALiBi, softcap, GQA 4 and 12, 3 streams, n_kv 113 / 512 / 1024, 1-75 rows, permuted layouts. Also the fork's Qwen3-4B one-token case (8 KV heads x 4, n_kv 256, the cache's layout) and n_kv 4096 and 16384 with 512 rows.

**M2e.1: `FLASH_ATTN_EXT` on the AI cores with the in-tree kernels, and C5a.**
- Kernels: copy the in-tree VLEN-1024 kernels and the helpers they use into `flagos-spacemit-rvv-attn.cpp`, origin noted, changes marked `flagos`:
  - the provider's interface (tile index, TCM, a workspace slice) instead of `ggml_compute_params`;
  - (a) every row's maximum starts at -inf;
  - (b) without a mask, rows group freely;
  - (c) rows are grouped only when their slopes are equal (no ALiBi), else the 1-row kernel runs.
  
  A portable reference following ggml-cpu's one-row kernel, accumulating in F32, runs on the Mac and with `FLAGOS_SPACEMIT_TEST_REFERENCE=1`, as the M2d references do.
- Op: one step, the rows split statically over the tiles as `ime` does.
  - With one token per call each tile takes the query heads of one KV head (Qwen3-4B: 4, 0.6B: 2) with the 4-row (2-row) one-chunk kernel. From 64 tokens per call the tiled kernel runs, otherwise the 1-row one-chunk kernel.
  - No reduction step and no wait inside the step.
  - Scratch in TCM (one-chunk about 35 KiB, tiled 160 KiB of the 384); without TCM, the 1-row kernel and a workspace slice.
  - The KV writes (`SET_ROWS`) run one step earlier in the same launch, so the barrier orders them before the reads.
- Claim (exact): q F32 with contiguous rows; K and V F16 with contiguous rows, DK equal to q's row length, DK and DV in 1..128; mask absent or F16 covering n_kv and the rows; sinks absent or F32, one value per query head; precision F32 or default; dst F32 and contiguous; every operand in host memory (the CPU-resident cache, read in place, §2.3) or in ours. ALiBi and softcap are claimed only if their `test-backend-ops` cases pass on the K3 (Qwen3 uses neither).
- C5a, a separate fork commit like C5b (`src/llama-context.cpp`, `resolve_fused_ops`): a fused node on an ACCEL device whose layer's device is the CPU does not count as a mismatch. `-fa auto` then keeps flash attention when the provider runs it (log: "Flash Attention enabled"). It covers llama.cpp's other fused ops of the same kind too; the TODO at that check already calls the rule wrong for other cases. Mentor to confirm (question 9).
- Placement (prediction): 2 splits per token, from 74: the token embedding on the CPU, everything else on the provider.

**Alternatives:**
- The graph without flash attention on the provider instead of C5a (the milestone table's M2e). It needs three more op kinds: F16 x F32 `MUL_MAT` over strided views with head broadcast, masked `SOFT_MAX`, and `CONT`. It materializes the score matrix (Qwen3-4B prompt processing at depth 4096: 64 MiB per layer), and these are the very ops ggml-spacemit got wrong in X0. It is needed only because of the automatic check. Kept as M2e.2, if the mentor rejects C5a or `-fa off` must be fast.
- Neither C5a nor M2e.2: only `-fa on` gains; `-fa auto` loses flash attention.
- A new kernel for generation first (M2e.3 below): faster at long context, but new code without a baseline on the board. M2e.1 gives that baseline and parity with `ime`.
- ggml-cpu's generic kernel compiled into the provider: the provider cannot call ggml-cpu, so it would be a copy of code without VLEN-1024 vectorization.

**Then M2e.3, by measurement (as the 1-row bandwidth work): attention during generation at streaming speed.** Designed in full 2026-10-13 ("M2e.3 design" below); the first list of candidates:
- (a) Split the KV positions over the tiles, as ggml-cpu does for one token per call. Each tile streams whole cells (every KV head) through TCM with `memcpy1d`, the pairs taking turns as in `paired`; a second step merges the 8 partial results per query head.
- (b) A KV head per tile as in M2e.1, but 64 positions' rows copied into TCM with several loads in flight before they are used.
- (c) The KV cache on huge pages: set aside by the user 2026-10-13. It is no kernel change: it needs llama.cpp to allocate the cache differently (an upstream change, `upstream-changes.md`) or a system setting the shared board rules out, and huge pages slowed parallel streams 2.2x in L1.

Expected (estimate): attention at n_kv 4096 from about 66 to 25-30 ms per token, so tg32 at depth 4096 from about 6.3 to 8-8.5 t/s (`ime` 6.41). Rule, fixed before measuring: adopt if attention at n_kv 4096 costs at least 20% less per token, confirmed by tg32 at depth 4096, with NMSE against the CPU within the bound (the outputs differ in rounding only).

**Inputs for M2e.3, measured 2026-10-13** (the ggml-spacemit comparison's first layer, L1; `build.md` §7, "One-chunk attention kernels across three backends"):
- ggml-spacemit's one-chunk kernels run at our speed (same code), so nothing to adopt there.
  - The only real difference in its attention is the tiled kernel, which trims the masked tail of each 64-position KV tile per query row.
  - Measured inside the provider in L2 and decided in L3 (next block).
- The cache layout costs about 45%. With the current kernel, each tile streaming its own contiguous region reaches 17.9 GB/s of K and V, against 9.7 GB/s on llama.cpp's cache layout, where every tile reads 256 B of every 2 KiB cell.
  - Candidate (a) removes that pattern. On that measurement alone it would take attention in generation at depth 4096 from about 62 to about 34 ms per token, tg32 at depth 4096 about 6.5 -> 7.9 (estimate).
  - Beyond that the stream-only / compute-only split decides: per position the kernel's arithmetic now takes at most 0.23 us (its whole time on contiguous streams), and 28 GB/s would need 0.15.
  - (b) keeps the strided pattern and only adds loads in flight, so it is expected to gain less than (a); it stays in the comparison.
- Huge pages cost 2.2x when the 8 tiles stream from addresses a multiple of 2 MiB apart (`k3-hardware.md` §2).
  - (a)'s split of the positions (contiguous ranges n_kv / 8 cells apart, 1 MiB at n_kv 4096) and candidate (c) (the cache on huge pages) must avoid that: stagger the range starts or deal blocks of cells round-robin.
  - Measure every strategy with the test cache on 4 KiB pages and on huge pages.

**L2 and L3: ggml-spacemit's tiled kernel inside the provider** (measured and adopted 2026-10-13; results below and in `build.md` §7):
- What runs: `forward_tiled_trim` in `flagos-spacemit-rvv-attn.cpp`, ggml-spacemit's tiled kernel (`4e782bc`, `rvv_kernels.cpp:1306-1641`).
  - Unchanged except where it meets the provider: its scratch is the tile's TCM, as `forward_tiled`'s; its `rvv_qk_dot_tile`, which takes the K tile's row stride apart from the column count, is renamed `rvv_qk_dot_tile_stride`.
  - Switch: `FLAGOS_SPACEMIT_ATTN=trim`, test-only, read at every call, so one process can time both kernels.
  - Fixes (e) and (f) reach it through the shared helpers; (d) cannot occur in it (it applies the softcap only to each row's visible columns); (a)-(c) are faults of the one-chunk kernels.
- The only difference from ours: per query row and KV tile it finds the last column the mask leaves visible.
  - Scores, mask, softmax and the V accumulation stop there; a group of 4 rows stops at the largest of its 4.
  - K is transposed and V copied only up to the tile's largest.
  - Both kernels skip fully masked tiles; the rest is the same code.
- Where that saves work: in the diagonal tiles of the causal mask only (aligned to 64 there, about 53% of their V loop runs). Of the tiles processed, these are:
  - 2 of 3 at 128 tokens from depth 0, 8 of 36 at 512 tokens from depth 0;
  - 2 of 127 at 128 tokens after 3968, 8 of 548 at 512 after 4096.
- Predictions (estimates):
  - Per call (`--bench-attn`):
    - 128 tokens at n_kv 128: 15-30% less time. The low end holds if a vector instruction takes as long with fewer columns; then only the V loop gains.
    - 512 tokens at n_kv 512: 5-10% less.
    - The two depth shapes: about 1% less (its mask scan stops at a row's last visible column; ours reads all 64).
  - Model: attention is 7% of pp128 at depth 0 (106 of 1460 ms per batch), so pp128 about +2%, pp512 at depth 0 about +1.5%, and at depth 4096 under +1%, inside the noise.
  - Outputs: equal to ours bit for bit, except perhaps the last bit where a diagonal tile's row sum is reduced over fewer lanes (an unordered vector sum). Within the NMSE bounds against the CPU either way.
- Checks and measurements on the K3:
  1. `flagos-check-spacemit --expect-device`: every check. A new line compares the tiled cases (Qwen3-4B 128 tokens and 99 after 37, Qwen3-0.6B 128 tokens) between the two kernels and against the CPU.
  2. `FLAGOS_SPACEMIT_ATTN=trim test-backend-ops -o FLASH_ATTN_EXT -b FlagOS:SpacemiT:0`: every case passes (1324 with ours).
  3. `flagos-check-spacemit --bench-attn`: both kernels per call at the four prompt shapes, taking turns twice, the faster turn of each.
  4. `llama-bench` pp128 and pp512 at depth 0, both settings taking turns, twice; pp512 at depth 4096 once per setting (slow).
- Rule (L3), fixed before measuring: adopt `trim` if all of these hold:
  - 1 and 2 pass;
  - 3 shows at least 5% less time at 128 or 512 tokens from depth 0, and no prompt shape more than 2% slower;
  - 4 shows no setting more than 1% below ours.
- Adopting:
  - `forward_tiled` takes the masked-tail handling (one tiled kernel; our comments and fixes kept);
  - `forward_tiled_trim` and the switch go;
  - test-backend-ops and the 512-token KLD run again (expected unchanged).
- Otherwise both go, and the result is recorded here.
- Results on the K3, 2026-10-13 (commit `68207bf`):
  - Correct: outputs bit-identical to ours in all three tiled cases; test-backend-ops passes every case with `trim`.
  - Per call: 128 tokens at depth 0 -26%, 512 at depth 0 -31%, 128 after 3968 -33%, 512 after 4096 -33%.
  - Model: pp128 87.17 -> 89.18 (+2.3%), pp512 85.11 -> 90.89 (+6.8%), pp512 at depth 4096 24.86 -> 33.22 (+34%; `ime` 25.13).
- The predictions held for pp128 only, and for the wrong reason: most of the gain is not the trimming.
  - The in-tree kernel read all 4096 mask values of every KV tile one at a time, to see whether it could skip the tile. `trim` reads one per row of a visible tile.
  - A fit gives about 16 ns (32 cycles) per value read plus 32 us per diagonal tile from the trimming. Fitted on two shapes, it predicts the other two within 2% and 17%, and the model's gains within 0.4 points.
  - So that check was a third of the in-tree kernel's time at every depth; the prediction had put it at 1%. Why a scalar read of the mask copy (in TCM) costs 32 cycles is not verified.
- Decision (L3), by the rule: adopted 2026-10-13. `forward_tiled` is now ggml-spacemit's kernel, with its `rvv_qk_dot_tile`; the in-tree tiled kernel, the copy and the switch are gone.
  - Accepted on the K3 2026-10-13 (commit `a832349`, runner m2e without `--depth`: 36/36). Perplexity 9.782 with `-fa on` and `-fa auto` and mean KLD 0.003354 / 0.002050 are M2e.1's to the last digit, and the greedy text leaves the CPU's at the same character (143), as bit-identical outputs predict; pp128 87.70 -> 89.41, tg128 11.27 -> 11.28 (`build.md` §7).

**L4: the scalar mask reads that remain** (measured and adopted 2026-10-13; results below and in `build.md` §7):
- What remains: the adopted kernel still reads the mask one value at a time wherever a row's tail is masked.
  - 4096 values per fully masked tile (skipped after the reads), about 2100 per diagonal tile, 64 per visible tile.
  - With n_kv padded to 256, a 128-token prompt from depth 0 has 5 fully masked tiles of the 8 per head, and a 512-token one 28 of 64.
- Change, behind `FLAGOS_SPACEMIT_ATTN=vscan` (test-only, read at every call): the same counts from vector instructions.
  - One strided load of the tile's last column tells whether every row's last column is visible (the case at depth); then every count is the tile's width.
  - Otherwise each row's count is one vector compare and a maximum over its 64 columns (`rvv_active_columns`).
  - The counts are the same integers as the scalar loop's, so the outputs must be bit-identical.
- Predictions, from L2's fit (16 ns per scalar read) and an estimated 2.5 us per tile for the vector counts (0.1 us when every row is visible):
  - Per call: 128 tokens at depth 0 about -23%, 512 at depth 0 about -32%, 128 after 3968 about -1.5%, 512 after 4096 about -3.4%.
  - Model: pp128 about +4% (89.2 -> 92.7), pp512 about +5.5% (90.9 -> 95.9), pp512 at depth 4096 about +2.4% (33.2 -> 34.0).
  - pp128 gains more than its benchmark shape suggests: in the model its n_kv is 256, not 128, so two of its four KV tiles are padding, all masked.
  - If 512 tokens from depth 0 gains much less than 20%, the scalar reads are not what L2 removed and the 16 ns figure is wrong.
- Checks and measurements on the K3: L2's 1-4 with `vscan` in place of `trim`; 1 now fails on any output that differs from the default's.
- Rule, fixed before measuring: adopt `vscan` if all of these hold:
  - 1 and 2 pass;
  - 3 shows at least 5% less time at 128 or 512 tokens from depth 0, and no prompt shape more than 2% slower;
  - 4 shows no setting more than 1% below the default.
- Adopting: the vector counts are the only path, and the scalar loop and the switch go. Otherwise the switch goes.
- Results on the K3, 2026-10-13 (commit `a832349`):
  - Correct: outputs bit-identical to the default's in all three tiled cases; test-backend-ops passes every case with `vscan`.
  - Per call: 128 tokens at depth 0 -20.4%, 512 at depth 0 -27.1%, 128 after 3968 -4.0%, 512 after 4096 -5.6% (predicted -23, -32, -1.5, -3.4).
  - Model, two rounds taking turns: pp128 89.31 -> 92.13 (+3.2%), pp512 91.78 -> 95.59 (+4.2%), pp512 at depth 4096 33.17 -> 34.49 (+4.0%). Predicted before measuring: +4, +5.5 and +2.4%.
- Why depth 0 gained less and depth more than predicted, from a fit over the four shapes:
  - Per tile, the vector counts save about 4.5 us on a fully visible tile, 28 us on a diagonal one and 50 us on a fully masked one.
  - So each of a visible tile's 64 scalar reads (one per row, 256 B apart) cost about 70 ns, against about 16 ns in sequence (L2).
  - The vector counts cost more than estimated: each row's count moves from a vector register to a scalar one, which waits for the vector unit.
  - The model's gains, predicted from these per-tile savings before llama-bench ran, matched within 0.5 points.
- Decision, by the rule: adopted 2026-10-13. The vector counts are the only path; the scalar loop and the switch are gone. The outputs are unchanged bit for bit, so L3's acceptance run covers L4 too.
- L3's acceptance run covers L4: the outputs are bit-identical, so perplexity and KLD stay M2e.1's.
- Guards, added after adoption (2026-10-13), because a wrong count could now pass the NMSE checks, the byte-for-byte comparison having gone with the switch:
  - the tiled cases' rows are held to 1e-5 in NMSE against the CPU (5e-3 for the others; the tiled rows matched to 8.4e-7 on the K3);
  - a tiled case at depth, 128 tokens after 3968 with flat scores (q's values scaled by 1): there a row that misses one visible position is off by a median 1.9e-4, at least 6e-5 (simulated), so the bound catches it; with the other cases' peaked scores one position in a 4096-position row could stay below it (median 5e-6);
  - `forward_tiled` asserts that a register group holds a tile's row (VLEN 1024), so a shorter VLEN fails instead of counting part of each row.
  - On the K3 (commit `fc1240a`): every check passes. The tiled rows' worst NMSE is 6.2e-7 with the case at depth, 16x under the bound and about 100x under the smallest simulated miss (6e-5); the one-chunk cases are unchanged (9.7e-5, worst row 1.4e-4).
- What remains of the check: an estimated 15 us per diagonal or fully masked tile, from the 64 vector-to-scalar moves. Counting all rows in one register would remove most of it (roughly 10% of the tiled kernel's time at depth 0, pp512 about +1.5%); not planned.

**Tests:**
- `test-backend-ops -o FLASH_ATTN_EXT` on the Mac (host test device, reference) and on the K3 (RVV kernels and references): every claimed case passes; the 12 earlier op sets unchanged.
- `flagos-check-spacemit`, against the CPU with the cache's layout (all KV heads in one cell of a CPU buffer), at Qwen3-4B's and 0.6B's head counts:
  - 1 token at n_kv 256 with 37 valid positions, at 1024 and at 4096;
  - 7 rows (1-row one-chunk kernel); 128 rows with a causal mask (tiled), and 128 after 3968 with flat scores (tiled, rows of about 4000 positions; added with L4's guards);
  - bound 5e-4; each row 5e-3, the tiled cases' rows 1e-5 (L4's guards); on the K3 also the RVV kernels against the reference.
- Probes for the three faults, each gating:
  - (a): one KV head whose scores all lie below -20 (q scaled), which the unfixed kernel returns as zeros for 3 of its 4 query heads;
  - (b): no mask;
  - (c): ALiBi with GQA.
- The layer check runs attention on the provider too, so a whole layer is one launch.
- Runner milestone m2e:
  - the above;
  - perplexity with `-fa auto` (C5a) and with `-fa on`;
  - KLD against the CPU for both kernels (512-token batches: tiled; `-ub 1`: one-chunk);
  - 2 splits per token;
  - llama-bench tg128 and pp128, the depth series tg32 at 0 / 1024 / 4096 for provider, `cpu` and `ime`, and pp512 at depth 4096.

**Expected (estimates):**
- About 9 ms less per token from 72 fewer handoffs, and attention at 16 instead of 110 us per position.
- tg32 at depth 0 / 1024 / 4096 about 10.9 / 9.2 / 6.3 (`ime` 11.05 / 9.34 / 6.41): parity with `ime` at every depth.
- tg128 about 10.5-11 (`ime` 11.0). The milestone's target of 11.10 may be missed by a few percent; the rest is per-op overhead, M4's area.
- pp128 a few percent higher; prompt processing at depth gains most.
- Accuracy: the same algorithms as the CPU's (F16 Q.K with F32 sums, softmax in F32, V in F16 in the one-chunk kernels like the CPU's one-row kernel), so KLD within the noise of M2c.2's (0.0033 / 0.0022).

**Exit (M2e.1):**
- Every claimed case passes on the K3.
- With C5a, `-fa auto` keeps flash attention on the provider and a token takes 2 splits.
- Perplexity within 1% of the CPU and mean KLD at most 0.0005 above M2c.2's.
- tg32 at depth 4096 at least 95% of `ime`'s.
- tg128 and pp128 not below today's (9.2 and 83).

**Risks:**
- Further faults in the copied code. The probes, the reference and NMSE checks at long n_kv guard against them, and ALiBi and softcap are claimed only once verified.
- C5a rejected: then M2e.2 is needed for default runs.
- No TCM in a launch: the slower kernels run, as for the 1-row matmuls.
- Runs right after heavy X100 load read low (the open note above). After M2e.1 the provider depends far less on the X100 cores.

**Decisions (user, 2026-10-12):**
1. C5a goes into the fork as its own commit, like C5b, so that `-fa auto` keeps flash attention on the provider. The mentor confirms it under question 9; if rejected, M2e.2 instead.
2. M2e.1 first, with the in-tree kernels; M2e.3 afterwards, by measurement; no new kernel for generation before that.

### M2e.3 design: attention during generation at streaming speed

**Status:** designed 2026-10-13. Round 1 implemented and measured 2026-10-14 (results below and in `build.md` §7): nothing adopted. Round 2 implemented 2026-10-14 (user: "now free to work round2 candidate a"), measured the same day and adopted by its rule (user: "yes apply the adoption plan"): one-token attention from n_kv 512 splits the KV positions over the tiles, the two tiles of a pair taking turns, -35% per call at n_kv 4096 and tg32 at depth 4096 +20% (6.52 -> 7.83), KLD unchanged (below and `build.md` §7). Acceptance run pending. Built on M2e's final state (`build.md` §7) and on what L1-L4 measured. The candidates are kernel changes inside the provider: the same op, inputs, outputs and math. None changes llama.cpp, the model or the KV cache. Inference-level options (a quantized KV cache, another cache layout, speculative decoding) are out of scope; (c), the cache on huge pages, is set aside (above).

**Where the time goes now (Qwen3-4B, depth 4096):**
- A token takes 154 ms: attention 65.6 ms (36 calls of 1822 us at n_kv 4096), the 1-row matmuls about 88 ms. Attention is 43% of a token, and its share grows with context; at depth 1024 it is about 17%.
- At one token per call each of the 8 tiles takes the 4 query heads of one KV head (2 for Qwen3-0.6B) and runs the grouped one-chunk kernel (`flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow`). For each of up to 64 positions it reads the mask value, loads the head's K row and V row (256 B each, from cells 2 KiB apart), and writes them into TCM (K transposed). Then it computes the scores of the chunk, the softmax update and the V accumulation (in F16), from TCM.
- A call reads 16 MiB of K and V at n_kv 4096. That is 9.2 GB/s, or 0.445 us per position for the 8 tiles together.
  - The same kernel on contiguous streams (each head's rows adjacent, L1) takes 0.228 us per position (17.9 GB/s).
  - The 1-row matmuls stream at 28 GB/s, which would be 0.146 us per position: the practical ceiling, 21.6 ms per token at depth 4096.
- So the strided pattern costs about half of attention's time: one 256-B row at a time from each 2 KiB cell, two dependent loads per position, staging and computing in turn.
  - The inner loops also take scalar operands from TCM: 8 q values per position in the scores, 4 probabilities per position in the V loop. Scalar reads of TCM were slow in L2-L4 (16-70 ns).
  - But these reads cannot be what limits the contiguous case. Its 0.228 us per position leaves no room for 12 reads at 16 ns besides the 0.146 us the bytes need, unless they overlap the loads. On the real layout the strided loads most likely dominate.
- Prefetch hints are not a candidate: Zicbop `prefetch.r` made the 1-row matmuls 41-58% slower (2026-10-11). Staging through TCM with copies worked there (`staged`, then `paired`).

**Round 1: measure inside the current kernel** (small code, one K3 session). A test switch `FLAGOS_SPACEMIT_ATTN` takes comma-separated settings, read at every call; it acts on the grouped one-chunk kernel only:
- `stream` (test-only, wrong outputs, like `FLAGOS_SPACEMIT_GEMV=copy`): stage K, V and the mask values as now, skip the scores, softmax and V accumulation. Its time is the staging.
- `compute` (test-only, wrong outputs): stage the first chunk only, and let every later chunk reuse it. Its time is the arithmetic, the mask reads and one chunk's staging.
- `cached`, candidate (e): q and the chunk's scores/probabilities in a stack array (2 KiB, L1-cached) instead of TCM, so the inner loops' scalar operands come from ordinary memory. The values and the order of operations are unchanged, so the outputs must be bit-identical.
- `burst`, candidate (b): stage a chunk's K and V rows with several rows' loads in flight (load 8 rows into separate vector registers, then store them), instead of one row at a time between stores. Bit-identical too.
- With `stream` and `compute` against the default, the round says whether the kernel is bound by staging, by arithmetic, or by both in turn: `stream` + `compute` close to the total means they never overlap.
- Implementation (2026-10-14):
  - `FLAGOS_SPACEMIT_ATTN` is read in `spacemit_kernel_flash_attn` at every call and passed as bits (`spacemit_rvv_attn::one_chunk_test`) through `forward_one_chunk` to the grouped kernel; the tiled and the 1-row kernels ignore it.
  - In the grouped kernel the default stages one position at a time as before. `burst` and `compute` collect the same visible positions per chunk, at most 8 at a time and never past the chunk's room, so every chunk holds what it holds by default.
  - `burst` stages 8 rows with `rvv_stage_rows_x8`: 8 K loads, their 8 stores, then the same for V. `compute` stages the first chunk only. `stream` skips the arithmetic, leaving the output at zero.
  - `cached` points q and the scores at 2 KiB on the stack, the TCM layout unchanged.
  - `flagos-check-spacemit` compares `cached`, `burst` and `cached,burst` byte for byte with the default on the four one-token cases (gating on the K3).
  - `--bench-attn` (and `--bench`) times `stream`, `stream,burst`, `compute`, `cached`, `burst` and `cached,burst` at one token, n_kv 256, 1024, 4096 and the new 8192, every setting and the default taking turns twice, the faster turn of each.

**Predictions (estimates):**
- `stream` close to the total on the real layout (strided loads dominate); `compute` well under half of it.
- `cached`: 0-10% less attention time (it can only help where the arithmetic is on the critical path).
- `burst`: 10-40% less. Same layout, more memory parallelism; the contiguous case shows what a better access pattern is worth (-49%).
- In the model, each 10% less attention is about 4% faster generation at depth 4096 and about 2% at depth 1024; nothing changes at depth 0.

**Rule for `cached` and `burst`** (each, and both together), fixed before measuring, as L4's:
- outputs identical to the default's in every 1-token check case; test-backend-ops passes every case with the setting;
- at least 5% less 1-token attention time at n_kv 4096 (`--bench-attn`, taking turns), and no 1-token shape more than 2% slower;
- tg32 at depth 4096 (llama-bench, taking turns) not more than 1% below the default.
- Adopting: the setting becomes the kernel's only path and the switch value goes. `stream` and `compute` never become paths; they leave with the round.

**Round 1 results (K3, 2026-10-14; `build.md` §7):**
- `cached`, `burst` and `cached,burst` keep the outputs byte for byte (four one-token cases, test-backend-ops).
- The kernel is bound by its staging, and staging and arithmetic run one after the other. At n_kv 4096, staging alone takes 80% of a call (1307 of 1633 us), the arithmetic alone 31%; at 8192, 75% and 29%.
- `cached`: -4 to -6% per call at n_kv 256 and 1024; at depth the model moves by +0.7% (tg32 at depth 4096). Its +7-9% at n_kv 4096 and 8192 in the benchmark is noise (not borne out by the model, nor consistent with `cached,burst`). The scalar TCM reads cost little in this kernel: independent of each other and interleaved with vector work.
- `burst`: staging -13 to -14% over 256-1024 positions, +16 to +26% over 4096-8192 (`stream,burst` against `stream`); in the model -1.4% at depth 4096.
- The predictions held for the decomposition (`stream` close to the total, `compute` well under half) and for `cached` (0-10%), not for `burst` (10-40% predicted).
- Decision, by the rule: nothing adopted. No setting reaches 5% less time at n_kv 4096, and the model moves by at most 1.4%.
- Removed 2026-10-14 (user), with their switch: the kernel, its header and the dispatcher are `fc1240a`'s again, and the check program keeps only the benchmark's one-token shape at n_kv 8192. Round 2 brings its own switch: (a) is a separate two-step op, so the choice sits where the op is selected, not in the grouped kernel.
- For round 2: attention at n_kv 4096 is 1.63 ms per call, above the 1.0 ms that calls for it. The staging's access pattern is the limit, which (a) replaces. Overlapping staging with arithmetic (double buffering, or the two tiles of a pair taking turns) is worth up to the smaller phase, about 30%. For effects of a few percent at long context the model, not the benchmark, decides.

**Round 2: (a), split the positions over the tiles** (only if round 1 leaves attention at n_kv 4096 above about 1.0 ms per call, that is, well short of contiguous-stream speed):
- At one token per call, from n_kv 512 (ggml-cpu's threshold for one token, `ops.cpp:9115`; the crossover is measured), each tile takes blocks of 64 positions dealt round-robin. Block b goes to tile b mod 8, so the tiles' streams are adjacent rather than 1 MiB apart (L1's power-of-two rule).
- Each tile copies whole cells into TCM with `spacemit_ime::copy`, the routine behind the 1-row path's 28 GB/s. A 64-position block is one contiguous range of K (128 KiB) and one of V (128 KiB); 32-position slabs leave room for double buffering, or the two tiles of a pair take turns as in `paired`.
- For each of the 8 KV heads in the slab: transpose its K rows (stride 2 KiB in TCM) with `rvv_transposed_s16_mn_to_nm`, score its 4 query heads, update their running maximum and sum, and accumulate V in F32 (V rows read at a 2 KiB stride, which needs a strided variant of the V helper). Mask values come by vector loads, and q and the scores live in cached memory (round 1's lesson).
- Step 1 writes each tile's partial results to the workspace: per query head the maximum, the sum and 128 floats, 8 x 32 x 130 floats = 133 KB. Step 2, after the executor's barrier, merges the 8 partials per query head (each tile takes 4 heads), applies the sinks, and writes dst.
- This uses the provider's existing machinery: an op of two steps with a workspace, as the matmul's quantize-and-multiply. `spacemit_find_op` picks it by shape; every other call keeps the current kernels.
- Outputs change in rounding: F32 instead of F16 accumulation, and the merge's order. They are checked against the CPU and with KLD.
- Prediction: at least the contiguous case's 0.23 us per position, about 1.0 ms per call at n_kv 4096 (-45 to -50%), attention 34-36 ms per token, tg32 at depth 4096 about 6.5 -> 8.1 (+25%). If its staging and arithmetic overlap, toward the 28 GB/s ceiling: about 0.6 ms per call, about 9 t/s at depth 4096.
- Implementation (2026-10-14):
  - Test switch `FLAGOS_SPACEMIT_ATTN=split` (and `split,paired` for the turns of each pair), read by `spacemit_find_op` at every graph. It takes the one-token calls the split handles (`split_shape`: one token and one stream, n_kv from 512, F16 K and V in the cache's cell layout, no ALiBi, no softcap, a mask row shared by all heads). Those get a two-step op, `k_flash_attn_split[_paired]`; every other call keeps the one-step op.
  - Step 1 (`spacemit_kernel_flash_attn_split_partial[_paired]`): blocks of `split_block_positions` positions (64 for Qwen3-4B) dealt round-robin. On the K3 each tile runs `split_begin`, then per block `split_stage` and `split_compute` (`flagos-spacemit-rvv-attn.cpp`), then `split_end`; `paired` interleaves them by the 1-row matmuls' protocol (`split_blocks`).
  - `split_stage` reads the block's mask by vector instructions and skips a block the mask hides entirely; otherwise it copies the block's whole cells of K and of V into TCM with `spacemit_ime::copy`. `split_compute` transposes each KV head's K rows, scores its query heads 4 (or 1) at a time with the tiled kernel's score helpers, updates their maxima and sums, and adds V in F32 by new strided helpers.
  - TCM per tile for Qwen3-4B: 128 KiB of K and 128 KiB of V, 16 KiB transposed K, 8 KiB q, 16 KiB V sums, small arrays (about 300 KiB of 384).
  - Without the RVV kernels or without TCM (the Mac, `FLAGOS_SPACEMIT_TEST_REFERENCE`) the same block loop runs a portable reference (`split_ref`), so the Mac checks the step structure, the dealing, the turns and the merge.
  - Step 2 (`spacemit_kernel_flash_attn_split_merge`, portable): each tile merges its query heads' partial results from the workspace (64 tiles x heads x (DV + 2) floats), adds the sinks as ggml-cpu does and writes dst.
  - Checks: `flagos-check-spacemit` runs 8 one-token cases with each setting against the CPU (n_kv 1000, 4000 of 4096 valid, 8192, GQA 2 and 1, head size 64, no mask, Qwen3-4B at 1024), and counts that their outputs differ from the one-chunk kernels' (that the split ran). `--bench-attn` times `split` and `split,paired` against the default at one token, taking turns.
  - On the Mac: all checks pass in the three host modes (the turns with real waits in the threads mode). The split's reference has max NMSE 2.3e-5 against the CPU, worst row 3.7e-5, 16/16 outputs differ from the default's. `test-backend-ops -o FLASH_ATTN_EXT` passes 1324/1324 with `split` and with `split,paired` (its permuted 512-position one-token cases, sinks included, take the split).
- Predictions refined at implementation, before measuring:
  - `split` stages then computes, in turn: copies of 16 MiB per call at n_kv 4096 at the 1-row path's 28 GB/s take about 0.6 ms, the arithmetic about 0.5 ms (round 1's `compute`), so about 1.1 ms per call, -30 to -40% against the default's 1.6-1.75.
  - `split,paired` overlaps one tile's copies with its partner's arithmetic: about 0.7-0.8 ms, -50 to -60%.
  - tg32 at depth 4096 accordingly about +15% (`split`) and about +25% (`split,paired`).
  - Outputs: NMSE against the CPU like the one-chunk kernels' or lower (V summed in F32); KLD at `-ub 1` about unchanged (at `-c 512` the split takes the positions from 256 on, where n_kv is 512).
- Rule, fixed before measuring:
  - at least 20% less 1-token attention time at n_kv 4096 than round 1's result, no 1-token shape slower;
  - tg32 at depth 4096 at least 8% higher, taking turns;
  - NMSE against the CPU within the bounds, perplexity within 1% of the CPU, mean KLD at `-ub 1` at most 0.0005 above M2e's 0.002050 (Qwen3-4B);
  - on Qwen3-0.6B too (added 2026-10-13, revised the same day, both before any measurement of (a)): over 32 chunks, M2e's state measured first, mean KLD at `-ub 1` at most 0.001 above M2e's, about 3 standard errors. No perplexity bound there: 8 chunks put ±0.74% on its ratio and the IME path itself measured +1.64%, and 32 chunks still leave about ±0.37%;
  - test-backend-ops passes every case.
  - Added 2026-10-14, before any measurement of (a): a one-token shape below n_kv 4096 that the split makes slower moves its threshold (`split_min_kv`) above that shape, to the crossover measured with added benchmark shapes, instead of rejecting the split. If both settings pass, the one with the higher tg32 at depth 4096 is adopted, `split` when the two are within 2% (it needs no pair waits).
- Results on the K3, 2026-10-14 (commit `e06cf90`; tables in `build.md` §7):
  - Correct: 16/16 one-token cases within the bounds (max NMSE 2.3e-5), each differing from the one-chunk kernels' output; test-backend-ops passes every case with `split` and with `split,paired`.
  - Per call against the one-chunk kernels: `split` -22.0 / -25.2 / -23.9% at n_kv 1024 / 4096 / 8192, `split,paired` -19.6 / -35.1 / -31.9%. Below the predictions (-30 to -40%, -50 to -60%): K and V read at 12.9 and 14.8 GB/s at n_kv 4096, half the 1-row matmuls' 28. Whether the arithmetic or the copies fall short of the estimate is not separated.
  - tg32, two rounds taking turns: at depth 4096 6.515 -> 7.515 (`split`, +15.3%) and 7.825 (`split,paired`, +20.1%); at depth 1024 +0.4% and +3.1%. At depth 1024 the round-robin deal likely leaves tile 0 with 3 of the 17 visible blocks while the others take 2 (n_kv 1280); not measured.
  - Accuracy, mean KLD at `-ub 1 -b 512`: Qwen3-4B 0.002050 -> 0.002001 (PPL ratio 1.0061 -> 1.0069), Qwen3-0.6B over 32 chunks 0.014032 -> 0.013979; both changes within one standard error.
- The `-ub 1` runs, corrected before the split's accuracy was measured: the first KLD runs, with llama-perplexity's default batch, gave both settings the one-chunk kernels' numbers to the last digit. llama-perplexity runs 4 sequences at once in separate cache streams, `split_equal` then puts a token of each of 2 sequences into every `-ub 1` ubatch, and the split takes one stream only (checked on the Mac, `build.md` §7). With `-b 512` (one sequence) every ubatch is one token, as in generation, and the split ran. So the rule's reference (M2e's 0.002050, Qwen3-4B) is the one-chunk kernels with one-token ubatches in the same run, which gave the same number. Every earlier `-ub 1` KLD had 2-token ubatches; on Qwen3-4B one-token ubatches reproduce its values to the last digit, so they stand. The runner and `build.md`'s recipe take `-b 512` since.
- Decision (2026-10-14, by the rule): adopted, `split,paired` (4.1% ahead of `split` at depth 4096, more than the 2% margin). Changes:
  - The split with the turns is the default for the calls `split_shape` takes; `FLAGOS_SPACEMIT_ATTN` keeps `split` (without the turns) and `one-chunk` (the kernels before) for comparisons, as `FLAGOS_SPACEMIT_GEMV` keeps `staged` and `direct`.
  - `flagos-check-spacemit` runs its one-chunk and tiled cases with `one-chunk`, and checks the split's cases with and without the turns against the CPU and against each other: the two settings deal each tile the same blocks in the same order, so their outputs must be identical (on the Mac 8/8; a mutation where the odd tile skips its first block fails both checks).
  - The benchmark compares the default with `split` and `one-chunk` at one token, and adds n_kv 512, where the split starts: if it is slower there, the threshold moves (the rule's added line).
  - `spacemit_check.py` runs its `-ub 1` KLD with `-b 512`.
- Not covered by the split: several sequences generating at once (several streams, or several tokens per call); ggml-cpu restricts its own split of the positions to one stream too (`ggml-cpu/ops.cpp:9115`).
- Possible later: deal the visible blocks evenly over the tiles (the depth-1024 imbalance above), and separate copies from arithmetic in the split (why 15 rather than 28 GB/s).

**Not candidates:** prefetch hints (above). Candidate (c), the KV cache on huge pages (set aside). The 1-row one-chunk kernel (`_m1`), which runs for batches of 2-63 tokens, ALiBi and launches without TCM: it is no part of generation for the Qwen3 targets.

**Tests:**
- `flagos-check-spacemit`: the 1-token cases (n_kv 256 with 37 valid, 1024, 4096; Qwen3-0.6B 300 of 512). For `cached` and `burst`, each output compared byte for byte with the default's. For (a), more cases:
  - n_kv 8192;
  - a padded tail (4000 valid of 4096);
  - n_kv not a multiple of 64 x 8;
  - ALiBi, softcap and sinks;
  - GQA 4, 2 and 1;
  - head sizes 64 and 128.
- `test-backend-ops -o FLASH_ATTN_EXT` with every setting that keeps the outputs (all claimed cases).
- `--bench-attn`: the 1-token shapes at n_kv 256, 1024, 4096 and 8192, every setting against the default, taking turns twice, the faster turn of each.
- llama-bench tg32 at depths 1024 and 4096, taking turns. For (a) also the runner (perplexity, KLD at `-ub 1`).

**Exit (M2e.3):** every adopted change passed its rule; attention per token at depth 4096 and the tg32 depth series recorded against M2e's final state; accuracy as M2e.1's exit; all checks pass. Status 2026-10-14: round 2 adopted by its rule; the acceptance run (runner m2e with `--depth`) records the rest.

**Risks:**
- The A100's vector unit may not keep several loads in flight (then `burst` shows nothing), and the scalar-read cost may not apply to these loops (then `cached` shows nothing). Both are cheap to measure.
- `stream` and `compute` produce wrong outputs by design. They stay out of every correctness check, like `FLAGOS_SPACEMIT_GEMV=copy`.
- (a) is the largest change: a second step per attention call (about 1 us per layer), a 133 KB workspace, new strided helpers, and changed rounding.
- The benchmark rotates copies of the cache so that reads come from DRAM; at small n_kv the model's cache may partly stay in the caches.

### M3 - AOT package (Design Phase 3, only if D9 is accepted)

- Generator `tools/generate_flagos_spacemit_kernels.py`, modelled on the AMD generator: compile FlagOS's shared Triton kernels (`ggml/src/ggml-flagos/kernels/generate_flagos_kernels.py`) with `triton.compile(ASTSource(...), target=AICPUTarget(...))` and take `asm["so"]`. Cross-compile on x86 with SpacemiT's toolchain; FlagTree's documented flow is x86 + Docker.
- Manifest (Design §9.4 envelope): ABI `flagos-spacemit-aot-v1`, backend spert, format riscv64 ELF `.so`, arch id `0xA064`, `num_threads` (baked in at compile time), per-kernel symbol, argument signature, size, sha256, allowed undefined symbols, provenance.
- Strict loader: ELF machine check, exactly one exported entry, undefined symbols within the allowlist, hash, fail closed (Design §9.6). `libspert` must be loaded with `RTLD_GLOBAL` first, because kernel `.so` files do not declare it.
- Typed launcher for FlagTree's ABI: `fn(spert::Context*, {rank, memref desc} per pointer, scalars, gx, gy, gz)`, on the provider's existing stream.
- Start with one non-critical op (for example `RMS_NORM`); keep it only if it matches or beats the hand-written RVV kernel.
- Exit: Design §15.3 package/ABI negative matrix passes (wrong hash, missing or extra kernel, wrong arch, wrong thread count); one AOT op in the default path behind an exact profile.

### M4 - Fixed fusion (Design Phase 4, only from measured hotspots)

- Profile a generation step (per-node time) after M2e. Candidates from Common's catalog: `rms_norm_mul` (1), `add_rms_norm_mul` (2), `ffn_swiglu` (7), `rope_kv_store` (8), `flash_attn_decode` (10).
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
| Stability | long generation (thousands of tokens), repeated sessions | M2e |

Design §15.6 gates, as they apply to the K3: no `supports_op` false positives (1); the target model runs entirely through direct paths plus CPU fallback (2, 3); KV/state writes tested (4); AOT fails closed (5, M3 only); full-model correctness (7); interleaved pp/tg A/B (8); default path needs no development environment variables (9); errors never reported as success (11); lifecycle stress (12).

## 6. Risks

| ID | Risk | Mitigation |
|---|---|---|
| R1 | Resolved 2026-10-07: A100 VLEN is 1024, X100 256, and a thread's `vlenb` changes when it migrates. Remaining risk: code that reads VLEN on the main thread (X100) and runs on A100 workers picks the wrong path; ggml-cpu does this (`ggml-cpu.c:742`, `repack.cpp:4591-4712`) | provider kernels read VLEN on the worker thread; profile reports 1024 |
| R2 | spine-runtime is a closed binary: `shared_buffer()` falls back to heap silently if TCM is unavailable (on this board it gets real TCM, 2026-10-07), all-or-nothing core grants, no exception handling around tiles | log a warning when the buffer is outside `/dev/tcm`; executor interface keeps option 2 open |
| R3 | Stale TCM blocks after a crash block all later runs | probe-time warning; release on every path; `tcmrelease` documented |
| R4 | Lossy repacks (Q4_1, Q4_K, Q6_K) change results. Q4_1 (M2c.1): on random weights the zero-point conversion changes a matmul by NMSE 5.9e-3, about doubling Q4_1's own error | claim only with KLD and perplexity evidence on the model (M2c); fallback for Q4_1: keep the exact minimum (M2c design). Q4_1 accepted 2026-10-10: on Qwen3-4B it raised the mean KLD against the CPU from 0.0024 to 0.0032 with 512-token batches (0.0020 with `-ub 1`), perplexity -0.10%. Q6_K (requantized to Q8_0) accepted 2026-10-10: +0.0001 (512-token batches) and +0.0002 (`-ub 1`), perplexity -0.16% |
| R5 | Under ACCEL, `-fa auto` switches flash attention off when the provider runs attention, so default runs use non-flash attention | C5a in the fork (decided 2026-10-12, its own commit with M2e.1; mentor to confirm); non-flash kernels on the provider (M2e.2) only if C5a is rejected; document `-fa on`; upstream fix candidate (§2.4) |
| R6 | Under ACCEL, a provider op that refuses CPU buffers forces copies and splits (E1: 280 splits per generated token). Also many splits until M2d, while each matmul is surrounded by CPU ops | accept host buffers for non-weight operands (§2.3); X3 measures it; split count in every benchmark |
| R7 | ggml-spacemit over-claims (X0: wrong F16 matmul, masked softmax, CPY; aborts in RMS_NORM, ROPE) | port piece by piece with fixes and tests; the mentor allows changing it |
| R8 | FlagTree SpacemiT is immature (no AOT tool, no tests, QEMU-only CI with VLEN 1024) | M3 is conditional and starts with one op |
| R9 | Common gaps: no `get_proc_address` forwarding (thread count, abort callback), no RVV/IME feature bits, no `LOCAL_SCRATCH` cap for TCM (mentor design §8.1, not in code) | work around in the provider; propose C2-C4 only when needed |
| R10 | The IME Q4_0 32x256 kernel has never been checked op by op: no `test-backend-ops` case fits the layout, the 1198/1198 run used plain buffers, and X0 ran only the cases ggml-spacemit claimed (its Q8_0 and Q6_K IME matmuls failed there) | M2b compares with the CPU backend at the model's shapes, and the IME kernel with its scalar reference |
| R11 | Copied kernel code (D6) misses later upstream fixes | at each upstream merge, check `git log` of `ggml/src/ggml-cpu/spacemit/`; copied functions keep their bodies unchanged and note their origin |
| R12 | Upstream's active branch of the 1-row Q4_1 IME kernel (`gemm_kernel_i8i4_m1`) multiplies zero point and activation sum in 16 bits (`vmul.vx` at SEW 16), which wraps for 32 activations near their block's maximum and of one sign against a zero point above 8, and adds each block in fp16. Found by reading the code (and by the review), from the ISA's definition; not run on the board | the provider enables the function's exact branch (`#if 0` upstream, M2c design); a probe in `flagos-check-spacemit` fails the run if the 1-row result is wrong; KLD with `-ub 1`; fallback: 1 row through the 4-row kernel. Confirmed on the board 2026-10-10: the probe gives 3.0e-7 for 1 row and for 4, mean KLD 0.0020 with `-ub 1` |
| R13 | The copied 4-row 8-bit kernel (`gemm_kernel_i8i8_m4`) decrements an input-only asm operand inside its loop over 32-column groups, so a call with more than 32 columns depends on register allocation | every multi-row call passes at most 32 columns; the M2c.2 dispatch asserts it (a loop over 32-column calls would not help once the kernel is inlined). On the K3 2026-10-10 the 4-row kernel matches the reference at every shape |

Optional Common changes, each needing separate approval: C2 RVV/IME/TCM feature bits in `flagos-target.h` (append-only); C3 forward provider functions through `get_proc_address`; C4 `LOCAL_SCRATCH` memory cap.

Outside FlagOS, needing the mentor's decision (fork patch or upstream llama.cpp issue): C5, in `src/llama-context.cpp`, (a) let the automatic flash-attention check accept an ACCEL backend that supports the op when the layer's device is the CPU (`:504-562`; **decided by the user 2026-10-12** for M2e.1, as its own commit like (b): once the provider claims `FLASH_ATTN_EXT`, `-fa auto` would otherwise turn flash attention off and leave attention on the X100 cores, see M2e design); (b) skip norm pinning when the layer's device is the CPU (`:2499-2520`, marked FIXME upstream; **implemented 2026-10-09** for M2d, skipping the pin only when an ACCEL backend supports the op, see M2d design). Without C5, ACCEL users need `-fa on` and get 30-39% slower generation (Qwen3-4B / 0.6B) (`device-type.md` §5, X3b).

## 7. Questions for the mentor

1. ~~Device type~~ decided: ACCEL (2026-10-08). Still open: FlagOS `caps.kind`, `cpu_accelerator` (proposed) or `ai_accelerator`?
2. Memory under ACCEL: weights in an IME-repacked buffer, stored once, with the provider reading CPU buffers for everything else (A2, §2.3). We are building M2b on A2 (2026-10-09); please confirm. The alternative (A1) doubles quantized-weight memory and would change only the buffer part.
3. Is spine-runtime (closed `libspert`) acceptable as a hard dependency of the provider?
4. Kernel reuse: decided 2026-10-09 to copy, per milestone, only the functions used into `providers/spacemit/` (§2.5), following the mentor design's "reference, not boundary" (§14.1) and SpacemiT's own ggml-spacemit. Objections? The alternative is compiling the in-tree sources in place.
5. Target for Phase 0: Qwen3-4B Q4_0 (SpacemiT has reference numbers) or Qwen3.5-4B Q4_K_M (comparable with AMD, but needs lossy Q4_K/Q6_K repacks and gated-delta-net ops early)?
6. Is FlagTree AOT (M3) in scope, and should it be cross-compiled on x86 or built on the K3?
7. Is co-building in `providers/spacemit` agreed with SpacemiT, given they ship ggml-spacemit as an `ACCEL` backend?
8. Should we report the TCM issues to SpacemiT (no dead-owner recovery; `try_wait` timeout unit about 12 µs, not 1 µs)? Also a CPU-backend crash whose register state the program cannot produce, now seen twice at the same instruction (2026-10-09 and 2026-10-13, both in long CPU-only runs; targeted tests with about 80 million page faults did not reproduce it). The second time the crash reporter captured the registers: every register matches the loop's state except `a2`, which is 0 where the code had just set it to a valid pointer, so something outside the program cleared it (kernel or core; `build.md` §7). And, found by reading: the 1-row Q4_1 IME kernel's active branch multiplies zero point and activation sum in 16 bits (R12).
9. ACCEL gaps in llama.cpp (C5, §6): patch the automatic flash-attention check and norm pinning in our fork, or raise an upstream llama.cpp issue? (b), norm pinning, is now patched in the fork as a separate commit (`f1cf809`, 2026-10-09, needed for M2d; on the K3 it gives the same 83 splits for 1-token and 512-token graphs); keep it, or replace it with an upstream issue? Without them ACCEL needs `-fa on` and generates 30-39% slower (`device-type.md` §5, X3b). (a) is decided by the user for M2e.1 the same way (2026-10-12, M2e design: a fork patch in its own commit): once the provider runs flash attention, `-fa auto` without (a) would turn it off. Keep it as well? If not, the graph without flash attention goes on the provider instead (M2e.2).
10. ~~Order after M2d~~ decided by the user 2026-10-10: M2c first, starting with the Q4_1 matmuls (M2c.1), then the Q6_K head (M2c.2), then M2e. The question was: M2c (Q4_1 and Q6_K on the AI cores), then M2e (attention)? Measured on the K3 (`build.md` §7, M2d): moving the 4 Q4_1 `ffn_down` matmuls alone raises pp128 from 53 to 83 t/s, past the pp128 target, and moving the head saves about 12-25 ms per token; attention on the X100 cores costs +445 ms per token at a context of 4096 tokens (IME: +68 ms), so M2e decides generation speed at long context. M2e brings the flash-attention question of item 9 (C5a) back.
