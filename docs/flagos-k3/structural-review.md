# Structural review: the SpacemiT provider against the deck's core / plugin / package design

Written 2026-10-26 (user request: a structural review of our backend, compared in particular with pages 6-8 of the
deck `FlagOS端侧与车载智算.pdf`, the mentor's "one core for every chip" design). Code at `65280e5` (the provider's last
change) and `adae43c` (HEAD). The deck's text is quoted from the PDF (its slides are images; the text was extracted
2026-10-26); `structural-experiments.md` already holds the mentor's requirements R1-R13 taken from it, the questions
Q1-Q10 and the findings F1-F18 that this review builds on. The mentor design v0.3
(`FLAGOS_BACKEND_MULTIPLATFORM_DESIGN.md` on `kevin/feature/flagos-multi-provider-backend`, 2026-08-22) is cited where
the deck changed it.

Labels: *measured* (on the K3 or the Mac, the run named), *code* (read at the lines given), *deck* (what a slide says),
*reading* (our interpretation of a slide), *estimate*. Line numbers are at `65280e5`; paths are relative to
`ggml/src/ggml-flagos/` unless given in full.

## 0. Summary

- The provider is a complete direct-op ggml backend compiled into `libggml-flagos`, built on the provider model of the
  mentor's `5794e12` (a `flagos_provider_v1` descriptor plus the ggml device, buffer and backend vtables; `upstream-review.md`
  B1, B4). It runs Qwen3-0.6B to 14B on the AI cores at or above the upstream IME path's speed with the KV cache and
  logits checks passing (`build.md` §7; `PROJECT.md` §1).
- The deck's pages 6-8 describe a different split of the same work: one shared core (`libggml-flagos.so`, layers L0-L2,
  no vendor SDK linked), a thin vendor runtime plugin (`libflagos-provider-<chip>.so`, a pure C ABI of 17 required
  functions) and a kernel package (`manifest.json` plus one binary per variant). None of that ABI, loader or package
  format is in any tree we have (F1): the deck is a target, and the K3 row on page 6 ("构建通过 · runtime 插件待做":
  builds, runtime plugin to do) is, by our reading, this provider's status as the mentor sees it.
- Against that split, our provider holds all three parts in one place (§2, §3): what the deck gives the core (the
  contract check, the expansion of an op into steps, the workspace, the variant choice, graph capture, diagnostics) is
  implemented inside the provider; what it gives the plugin (device, buffers, the stream, launches) is there too; and
  the kernels are compiled in, not packaged.
- Where we match the deck (§4): llama.cpp is unchanged at run time (E4, measured); placement is a contract check that
  execution repeats (③); an op expands into steps with a preceding step into the workspace (⑤'s "反量化进 workspace");
  the workspace is owned by the backend and grows to the largest split; graph capture with per-token replay is the
  execution model (Tier-2 ⑤); the acceptance criteria of page 8 are met in substance (§4.10).
- The gaps, in order of what they would take (§4, G1-G10): no plugin ABI (G1) and no kernel package (G2), both
  waiting on the mentor's header and schema; kernel choice by code where the deck has `when` (G3); an execution model
  that the deck treats as optional but the K3 needs (G4, measured: E3a, E3b, F15, F16); per-tile services the deck's
  launch does not carry (G5); weights repacked in the buffer where the deck assumes zero-copy weights (G6); no plan
  cache and no L1 (G7, 0.6% of a token); diagnostics with other names and two of the deck's five missing (G8);
  deployment and naming (G9); the device type and kind (G10).
- The M3 decision of 2026-10-26 (§6: FlagGems' kernels through FlagTree for the ops they have, the IME and RVV kernels
  for the rest) adds a second kernel source beside the compiled-in one; since the same day M3 and the FlagGems work are deferred
  until the structural work on the backend is done (user and mentor). Its structural cost is the launch model: a
  FlagTree kernel is one Triton program per spert tile launched on its own, and a launch of its own costs 13.4-15.0 µs
  in a model run (E3a, E3c, measured); every small op of a Qwen3-4B layer from the package as its own launch would cost
  about 8 ms per generated token, about 9% of tg128 (estimate, §6.2). Whether FlagTree can emit a kernel that runs as a
  step inside our one launch per split is the first thing M3.0 should test.

**Decided after this review (user and mentor, 2026-10-26; D10):** no separate `flagos-runtime.h`; the split is done
K3-only in our tree: first the core (S1: a draft of L1, the plugin loader, the rest of the core completed), then the
plugin (S2) with `flagos-spacemit-weights.{h,cpp}` in it; M3 and the FlagGems kernels stay deferred. §3's mapping is
the plan's basis with two changes (the weights and the per-tile services get plugin and interface slots); §7.1's ranks
1 and 3 are closed and rank 2 is S1's design input; §8 holds the order; `plan.md` S1 and S2 the content.

## 1. The deck's design, pages 6-8 (deck, with our reading)

| Element | What the deck says | Where it sits |
|---|---|---|
| llama.cpp | unchanged; uses the core through the ggml backend interface ("ggml backend 接口（llama.cpp 不用改）") | page 6, top |
| Core `libggml-flagos.so` | one for every chip, about 500 KB, no vendor SDK linked. L0: the ggml interface layer (device, buffer, stream, events; unified memory zero-copy). L1: Common planning (fusion pattern scan, required outputs and alias checks, an execution plan cache). L2: op adaptation, a ggml node to a kernel contract (the support decision, argument binding, the grid, the variant, one op expanded into several steps such as dequantize + GEMM). Also: plugin loading and package verification (dlopen, ABI, sha256, pruning of variants the device cannot run), workspace management (device scratch the core holds), graph capture and diagnostics (per-token replay; trace, sync, stats) | page 6, middle |
| Plugin `libflagos-provider-<chip>.so` | links only the vendor's SDK; the pure C header `flagos-runtime.h`; one export `flagos_provider_get_entry_v1(host)` returning the ABI version, kind RUNTIME, a name and a function table; the host provides logging and the plugin directory. Tier-1, 17 required functions: device 5 (`device_count`, `device_open`/`close`, `device_props`, `device_memory`), memory 3 (`buffer_alloc`, `buffer_free`, `buffer_base`; the core allocates its workspace with them), copy 2 (`memcpy`, `memset`, ordered on a stream; no stream means synchronous), stream 3 (`stream_create`/`destroy`/`synchronize`), kernel 4 (`module_load`/`unload`, `kernel_get`, `launch` with typed arguments PTR, I32, F32, ...). Optional events (5; without them the core runs synchronously). Tier-2, optional: graph capture (`graph_capture_begin`/`end`/`launch`/`destroy`, `end_update` for updates in place), zero-copy (`buffer_map`, `register_host_ptr`), a fusion switch (`fusion_supported(pattern)`). "Binary only" delivery, corrected by the user to source over a closed SDK (R3) | page 8, left |
| Kernel package | released apart from the `.so`, at `share/ggml/flagos/<hw>/<target>/`: `manifest.json` and one binary per variant; per variant `name`, `file`, `sha256`, `block_size`, `num_warps`, `argument_count`, `contract`, `priority`, `when` (example: `flagos_mul_mat_q4_0_f32_wide_b8_w1`, priority 20, `when` k a multiple of 256, align a multiple of 16). FlagGems sources, compiled per target by FlagTree ("gfx1150 HSACO · K3 ELF"; the vendor provides the compiler backend). Tuning changes only the package (a wide Q4_0 GEMV variant: 12.0 to 22.9 t/s in generation on AMD) | pages 6 and 8, right |
| The five stages | ① startup: the plugin loader scans `flagos-providers/`, dlopens, checks the ABI version, opens the package, checks sha256, prunes variants. ② model load: L0 allocates buffers; unified memory is zero-copy, weights need not be copied again. ③ placement, per node: L2 checks the kernel contract (is the kernel in the package; shape, alignment, resources); else the node stays on the CPU; the plugin is not involved. ④ planning: L1 scans fusion patterns (the provider enables them and the package has the fused kernel), caches the plan (a hit about 12 µs). ⑤ every token: L2 binds arguments, computes the grid, picks the variant whose `when` matches the facts, runs a preceding launch when needed (dequantize into the workspace); optionally captures the whole graph and replays it per token. ①② once; ③④ only when the graph's structure changes, reused in generation; ⑤ every token | page 7 |
| Diagnostics | `GGML_FLAGOS_STATS=1` (calls per op and variant), `KERNEL_VARIANTS=0` (variants off, for comparison), `DISABLE_OPS=...` (ops back to the CPU), `TRACE_LAUNCH=1` and `SYNC_LAUNCH=1` (find a kernel that fails or hangs) | page 7, footnote |
| Deployment and acceptance | the plugin in `flagos-providers/` next to the core or at `GGML_FLAGOS_PROVIDER_PATH`; the package at `GGML_FLAGOS_KERNEL_DIR` (sha256 checked when opened, variants pruned); `llama-cli --list-devices` shows `FlagOS-<chip>0`; accepted when `test-backend-ops` passes entirely, the whole model's KLD is below 0.01 and a llama-bench baseline is recorded | page 8, bottom |
| Reference plugins | AMD `amd-rt`: HIP, 35 KB, 488 lines, depends only on `libamdhip64`, 2238/2238 on the device. hostsim: kernels simulated on the CPU, no hardware, 2220/2220. SpacemiT K3: `libspert` runtime, native RVV/IME kept during the transition, builds, runtime plugin to do | page 6, bottom |

Three generations of the design exist, in this order (code and documents):
1. the mentor design v0.3 (2026-08-22, `kevin/feature/flagos-multi-provider-backend`): in-tree providers split into
   `runtime`, `memory`, `launcher`, `plan` and `libraries` files under `providers/<chip>/` (§17), a `core/` with op
   signatures, a kernel catalog, a selector and plans, manifest v3 with `semantic_op`, `argument_schema_id` and
   constraints (§10.1), diagnostics named `FLAGOS_*` (§18), and for the K3 the division of work of §14.2 (the core: ggml
   integration, patterns, planner, fallback, diagnostics, tests; SpacemiT: device, memory and TCM, AI-core scheduling,
   AOT loader and typed launcher, kernel variants and constraints, vendor libraries, the cost model);
2. the code we build on (`5794e12`, 2026-09-22) and its design document V2.0 (`ggml-flagos-provider-design.pdf`,
   "Design" in `plan.md`): providers compiled into `libggml-flagos`, each a full ggml backend behind a
   `flagos_provider_v1` descriptor (`flagos-provider.h:65-85`), which V2.0 §5.1 calls a source ABI and §1.2 lists
   among what the system is not ("a stable, hot-unloadable third-party binary plugin ABI"); Common a registry
   (`flagos-registry.cpp:142-193`) plus the fixed-fusion planner (`flagos-graph-plan.h`); the provider owning runtime,
   buffers, streams, workspace, private caches and replay (V2.0 §2.1, §2.3, §12.1); a pure C plugin ABI a P2 item on
   Common's roadmap, "an RFC for opaque handles, versions and unload" (§15.1, §17.3);
3. the deck (2026-10): the core as one binary for every chip, holding L0-L2, the workspace and the capture driver;
   the vendor's part a runtime plugin behind a C ABI; the kernels a package with `contract`, `when` and `priority`;
   diagnostics named `GGML_FLAGOS_*`.

Our provider is generation 2's code with generation 1's division of work for the K3 (we built what §14.2 gives
SpacemiT and left Common as it is), and nothing of generation 3 is in the tree. V2.0 read against the pending
questions: §7.2.

## 2. Our provider's structure today (code)

9,901 lines in `providers/spacemit/` outside the tests (`wc -l`, 2026-10-26), 5,950 of them (60%) the three kernel
files; the tests and tools add 3,634. Common (`flagos-*.{h,cpp}`, `ggml-flagos.h`) is 3,095 lines, untouched but for
the registry's three lines (`upstream-changes.md` M2) and the macOS link fix (M1 there). Each file, and the deck layer
it would belong to:

| File | Lines | Does | Deck layer (reading) |
|---|---|---|---|
| `flagos-spacemit.cpp` :102-226 | 125 | the buffer type: host memory on huge pages when at least 2 MiB (`spacemit_map_huge` :102), 64-byte alignment, `is_host` false (:225), `get_alloc_size` larger for the requantized layouts (:214); the buffer: `set_tensor` repacks, `get_tensor` unpacks, `cpy_tensor` refuses raw copies of repacked tensors (:146-160) | plugin memory 3 and copy 2, but with the repack (G6), which the deck has nowhere |
| `flagos-spacemit.cpp` :644-764 | 120 | `graph_compute`: every node to its op through `spacemit_find_op` (fail closed, `GGML_STATUS_FAILED` before any write), the op's steps appended with their workspace offsets, the workspace grown to the split's largest need, then one executor run per split under the device's mutex; the launch-mode switch (test-only) and the counters around it | core L2 (the multi-step expansion, the workspace) and the core's capture driver |
| `flagos-spacemit.cpp` :786-1101 | 315 | device: memory from `/proc/meminfo` (:802), the type (ACCEL, :830), props (no async, no events, no host buffer type, mmap :847), `init` creating an executor per backend (:869), `supports_op` = `spacemit_find_op` (:886), `supports_buft` own type or any host type (:891); probe by the A100 `marchid` in `/proc/cpuinfo` and the writable `/proc/set_ai_thread` (:917-1000); the descriptor, caps and profile (:1022-1101) | plugin device 5 (`device_count`, `open`, `props`, `memory`), plus Common's descriptor |
| `flagos-spacemit.cpp` :241-642 | 400 | test-only switches and the two counter classes (E3c's `FLAGOS_SPACEMIT_TEST_STATS`, the op counters `FLAGOS_SPACEMIT_TEST_OP_STATS`) | core diagnostics (G8) |
| `flagos-spacemit-exec.{h,cpp}` | 96 + 567 | the executor: one persistent `spert::Stream` holding the 8 cores (:388-395), one launch per run with `Grid{core_count}` and a host wait (:280-305); the tile's main loop: TCM taken once, one TCM size agreed across the tiles at a barrier, every step then a barrier, a shared failure flag honoured at every barrier (:160-213); the pair waits (:34-60); the test-only ordered launches (E3b, :220-366) and the host executor for the Mac (:448-556) | plugin stream 3, `launch`, and Tier-2 capture (G4, G5); the host executor is the deck's hostsim |
| `flagos-spacemit-ops.{h,cpp}` | 49 + 176 | the op table (:106-127: 8 one-step ops, `MUL_MAT` in 2 steps, the split attention in 2) and `spacemit_find_op` (:129-176): exact type, shape, layout and buffer predicates per op, the same function at placement and execution | core L2's contract check (③) |
| `flagos-spacemit-weights.{h,cpp}` | 42 + 190 | the layout rule from type and shape alone (:30-53), the allocation size (:55), repack and unpack on whole and partial writes and reads (:140-190) | nowhere in the deck (G6) |
| `flagos-spacemit-ime.{h,cpp}` | 31 + 612 | the matmul's two steps; the choice between the 1-row path with the weights staged through TCM and the pair turns (:508-512, :239-322), path B (slabs in TCM, :362-425), path A (4 rows per tile staged in TCM, :523-575) and path C (DRAM, :589-611), by rows, TCM size and layout; the leftover rows dealt (:140-167) | core L2's variant choice (⑤, `when`; G3) over package kernels, but using per-tile services (G5) |
| `flagos-spacemit-attn.{h,cpp}` | 24 + 528 | the claim (:401-429), the one-step kernel choosing tiled (64 rows or more, TCM) or one-chunk (:431-455), the split over the tiles from n_kv 512 for one token (:195-207, :361-399) with its merge (:478-528); the portable reference | the same as above; the reference is hostsim |
| `flagos-spacemit-kernels.{h,cpp}` | 22 + 421 | the element-wise and row ops: the RVV kernel where it applies, else the portable reference (:343-421); `FLAGOS_SPACEMIT_TEST_REFERENCE` (:14-24) | L2's variant choice; the references are hostsim |
| `flagos-spacemit-ime-kernels.{h,cpp}` | 106 + 3,195 | IME2 GEMM kernels, quantizers, `memcpy1d` and repack functions copied from `ggml-cpu/spacemit` with the changes listed at :1-28 | the kernel package's contents, compiled in (G2) |
| `flagos-spacemit-rvv-kernels.{h,cpp}` | 41 + 771 | RVV kernels copied from ggml-spacemit (:1-12) | the same |
| `flagos-spacemit-rvv-attn.{h,cpp}` | 51 + 1,786 | VLEN-1024 attention copied from the in-tree path and ggml-spacemit, six faults fixed, the split's kernels new (:1-32) | the same |
| `provider.cmake` | 85 | sources; on riscv64 `libspert` linked into `libggml-flagos` (:36-45), the IME2 instruction check needing GCC 15 (:47-62), per-file `-march` (:64-72); elsewhere the simulated device (:72-85) | the plugin's build (G1) |
| `tests/check_spacemit.cpp`, `tools/spacemit_check.py` | 3,146 + 488 | the provider checks and benchmarks; the acceptance runner per milestone | the deck's step 4 (acceptance) |

Not in the provider: fusion (`graph_optimize` null, no `flagos_fusion_interface`, Common's `flagos_graph_plan_cache`
unused; :766-783), events and async copies (:766-783, :847-861), a second buffer type, `offload_op`.

## 3. Where each part would go under the deck's split (reading, with estimates)

| Deck part | From our code | Lines (estimate) | Notes |
|---|---|---|---|
| core L0 | nothing new: the ggml vtables we implement would become the core's, calling the plugin | - | our buffer and device vtables are thin; the repack in `set_tensor` is not L0 work (G6) |
| core L1 | nothing: we have no fusion and no plan cache | 0 | G7 |
| core L2 | `flagos-spacemit-ops.cpp` (the contract check), `graph_compute`'s step expansion and workspace sizing, the path and kernel choices in `ime.cpp`, `attn.cpp`, `kernels.cpp` | about 300 of ops and `graph_compute`, plus the choice logic (about 150) | the choices need facts `when` does not express (G3) and services the launch ABI does not carry (G5) |
| plugin (17 functions, capture) | `exec.cpp` (the stream, the launch, the tile loop with its barriers and TCM agreement, the pair waits), the buffer allocation on huge pages, the probe, device memory and props | about 700-900 without the test-only code (E3b's ordered launches and E3c's clock reads, about 300) | against `amd-rt`'s 488 lines; the difference is the capture (the K3's main path, G4) |
| package | the three kernel files and the matmul and attention orchestration (paths, staging, the split), about 6,500 lines, plus the weight layouts | 6,500 + 230 | compiled in today; FlagTree-compiled kernels for the ops FlagGems has from M3 (§6); the orchestration is kernel code in the deck's sense only if a kernel may use the tile services (G5) |
| diagnostics | the two counter classes and the test switches | about 400 | names differ (G8) |
| hostsim | the host executor and the portable references | about 500 | the Mac's simulated device plays hostsim's role today (`provider.cmake:73-84`): the checks pass there before the K3 |
| tests | `check_spacemit.cpp`, `spacemit_check.py` | 3,634 | stay with the plugin |

Decided 2026-10-26 (D10): this mapping is the plan, K3-only, with two changes: the weight layouts go to the plugin
(the repack inside the buffer's set and get), and the per-tile services go into our own core–plugin interface, so the
two rows with no slot become plugin rows (`plan.md` S1, S2).

## 4. Gap analysis, box by box

Each gap: the deck, our code, what it costs today, what closes it, and whose it is.

### 4.1 L0, the ggml interface layer (device, buffer, stream, events, zero-copy)

- Fits: one ggml backend interface, llama.cpp unchanged at run time (E4, measured: stock `ba360ef` loading the provider
  as a module through `GGML_BACKEND_PATH` runs Qwen3-4B at today's speed with identical logits; `structural-experiments.md`
  §6 run 2). The ggml-side lines of the mentor's tree (F13) are build-time only.
- Fits: unified memory zero-copy for activations and the KV cache. The provider accepts any host buffer type for its
  operands (`flagos-spacemit.cpp:891-893`, `flagos-spacemit-ops.cpp:13-22`), so under ACCEL the KV cache in a CPU buffer
  is read in place, and under IGPU the cache lives in our buffer and nothing copies it (2 splits per token in both; E2).
- Differs, G6: weights are not zero-copy. `set_tensor` repacks into an IME layout and `is_host` is false
  (`flagos-spacemit.cpp:146-153, :225`; `flagos-spacemit-weights.cpp:140-190`), so a weight is copied once at load from
  the mapped file, and reads convert back (Q4_1 and Q6_K approximately, as the nearest tensor of their type). The deck's
  ② says weights need not be copied again. Every upstream accelerator with a layout of its own does what we do (zDNN,
  CANN, Hexagon, the CPU's repack buffers; F4, F10), and a package changes who owns the layout, not whether one is
  needed (F4). Options and evidence: Q2 (a)-(e), E5, E6. Owner: the mentor (which option the core supports); us for
  the measurements. Decided 2026-10-26: the plugin's (`flagos-spacemit-weights.{h,cpp}` part of it; D10): the repack
  inside the plugin's set and get.
- Fits: events optional. Ours reports no async and no events (`:847-861`, `:766-783`); the deck says the core then
  runs synchronously, which is what the scheduler does with us today.
- Fits, with a note: `mmap_support` true (`:858`): llama.cpp reads the file through its mapping and writes each weight
  whole into our buffer (`plan.md` §1), which the repack needs.

### 4.2 L1, Common planning (fusion scan, required outputs and aliases, plan cache)

- Gap, G7: none of it exists in the provider. `graph_optimize` is null and no `flagos_fusion_interface` is built
  (`:766-783`), so Common's planner and cache (`flagos-graph-plan.h`) never run for us; the step list and the workspace
  sizes are recomputed in every `graph_compute` (`:662-723`).
- Cost, measured (E3c, Qwen3-4B, tg128): host work inside `graph_compute` 0.55-0.64 ms per token, 0.6% of a token;
  the deck's cached plan hits in about 12 µs (deck). A cache keyed by the split's structure, as Common's
  `flagos_graph_plan` does, could remove most of the 0.55 ms (estimate, not measured: the time was not split between the
  node loop and the rest).
- Fusion: every candidate in our layer is estimated at 1-2% of tg128 and 5-6% of pp512 together
  (`fusion-and-splitting.md` §5), none adopted; the K3's valuable fusions (quantize + IME matmul + epilogue in TCM) are
  not in Common's closed catalog (`upstream-review.md` A1), and the mentor design's K3 fusion list (§14.3: RMSNorm+Mul,
  MatMul+residual, gate/up+SwiGLU, QKV, RoPE+KV store, an attention region) is the catalog's. The deck's L1 ("融合条件：
  provider 开启且包里有融合 kernel") would make a fused kernel in the package the condition: we have no package (G2).
- Owner: us (a step cache, if a measurement shows it worth 0.5%); the mentor (the catalog, A1-A6, M4).

### 4.3 L2, op adaptation (contract check, argument binding, grid, variant, multi-step)

- Fits: ③'s contract check is `spacemit_find_op` (`flagos-spacemit-ops.cpp:129-176`): exact op, types, shapes,
  layouts and buffers, true only for what a kernel handles, and the same function answers at placement and at execution
  (the rule in `flagos-spacemit-ops.h:45-49`), which is the deck's "只回答，不执行" at ③ and its re-check at ⑤. Also
  claimed: `NONE` and views (`flagos-spacemit.cpp:886-889`), as the Design's §7.4 allows.
- Fits: ⑤'s multi-step op. `MUL_MAT` is a quantize step into the workspace followed by the matmul
  (`flagos-spacemit-ops.cpp:115-116`), the deck's "反量化 + GEMM" in the other direction (our weights are already in
  kernel layout, the activations are quantized); the split attention is partial results then a merge (`:120-127`).
  The `FLAGOS_SPACEMIT_QUANTIZE=shared` experiment (`flagos-spacemit.cpp:284-300`) skips a repeated quantize step for
  Q, K and V, the kind of planning the deck gives L1 or L2.
- Differs, G3: the variant is chosen by code, not by `when`. The matmul's four paths are picked per call from the row
  count, the TCM size the tiles agreed on and the layout (`flagos-spacemit-ime.cpp:508-611`; the 1-row path's strategy
  and path B's turns from the settings at :104-128, :203-237); attention picks tiled, one-chunk or split from the query
  rows, n_kv, the cache's cell layout, ALiBi, softcap and TCM (`flagos-spacemit-attn.cpp:195-207, :431-455`); every
  element-wise kernel picks RVV or reference (`flagos-spacemit-kernels.cpp:343-421`). The deck's `when` shows
  `multiple_of` on `k` and `align`; our rules need ranges (rows 1, 2-31, 32 and more), equality (one token, one stream),
  strides (the cache's cells), the weight's layout and the TCM size, and every tile must reach the same choice (Q4). The
  mentor design's `shape_constraints` and `required_caps` (§10.3) are closer. Owner: the mentor (the vocabulary, Q4's
  options (a)-(c)); us (the Q4 table, still to do).
- Differs: the grid. The deck's core computes a grid per kernel from `block_size`; our grid is always the 8 tiles of
  one launch, and each kernel spreads its own work over them (`tile_range`, the slabs dealt round-robin, the KV blocks
  dealt round-robin). This follows from G4.
- Fits in part: argument binding is by the kernel reading the ggml tensor (`spacemit_kernel_fn(tile, node)`,
  `flagos-spacemit-ops.h:36`), not by a typed argument list. A package kernel would need the binding the deck's L2 does.

### 4.4 Plugin loading and package verification (dlopen, ABI, sha256, variant pruning)

- Gap, G1: there is no plugin ABI. The provider is compiled into `libggml-flagos` through `provider.cmake` and the
  registry's `#ifdef` (`flagos-registry.cpp:21-23, :81-83`), and implements the ggml vtables itself; `flagos_provider_v1`
  (`flagos-provider.h:65-85`) is a discovery descriptor, not a runtime ABI (`upstream-review.md` A6, B1). The deck's
  `flagos-runtime.h`, `flagos_provider_get_entry_v1`, the loader and `amd-rt` are in no branch we have (F1). The header's
  name is the deck's alone (page 8: "纯 C 接口 flagos-runtime.h"; page 6: `flagos_runtime_v1`); no document or branch we
  have shows the file (checked again 2026-10-26: not in the code, never in any fetched branch, and not in the mentor's newest branch of
  2026-10-10, `feature/flagos-amd-890-audit-20261010`, fetched that day; F1), and V2.0 names only a
  future pure-C plugin ABI RFC (§15.1, §17.3). Nothing that runs today needs it: the provider is a ggml backend on
  V2.0's source ABI, stock upstream loads it as a module (E4), and the split would change the build, not speed or
  correctness. It is needed only if the K3 is to be delivered as a plugin against that ABI, which is Q11. Closed 2026-10-26: no
  separate header; the K3-only split with our own interface (D10). Consequences
  today: with the provider on, `libggml-flagos` links `libspert` (`provider.cmake:36-45`) and needs GCC 15 with the
  IME2 instructions (`:47-62`), against the deck's core that links no vendor SDK; the Mac's host build of the core with
  the provider's references and no kernels is 296 KB (`build-host/bin/libggml-flagos.0.19.0.dylib`, measured
  2026-10-25), the K3 binary's size is not recorded. Owner: the mentor (the header, the loader, `amd-rt` as the
  reference); us (the split, §3, once the header exists). Who defines the ABI, by the documents: the mentor design's §21 has
  changes to the core and to the provider contract led by FlagOS and reviewed by the provider teams ("FlagOS 主导；
  Provider 团队共同评审；Provider Contract 变更必须有版本和兼容说明"), and its §14.2 gives SpacemiT's side the device,
  memory and TCM, the AI-core scheduling and the launcher; the deck keeps the core FlagOS-maintained and shared. So the
  ABI is the mentor's to define, ours to review for the K3's needs (§7.1 rank 2) and to implement the K3 plugin against. The plugin's delivery is source over the closed `libspert`
  (R3, Q8), which is what we have.
- Gap, G2: there is no package. The kernels are compiled in; `caps.execution` is 0 and `aot_format` null
  (`flagos-spacemit.cpp:1022-1034`); no manifest, no sha256, no `GGML_FLAGOS_KERNEL_DIR`. The deck itself keeps the
  native RVV/IME kernels in the plugin during the transition (R7), and Q6 decided the same for now (user, 2026-10-20),
  so this is a planned state. The package arrives with M3 (§6). Three manifest schemas exist and none is final: this
  tree's format 2 (`kernels/aot/manifest.json`; the AMD loader reads `format`, `abi`, `backend`, `binary_format`,
  `arch`, per kernel `name`, `symbol`, `file`, `sha256`, `size_bytes`, `argument_count`, `num_warps`, `warp_size`,
  `block_size`, `exact_block_size`, `tile_*`, `shared`, `global_scratch_size`; `flagos-amd-aot.cpp:414-516`; no
  `contract`, `when` or `priority`), the mentor design's v3 (`semantic_op`, `variant`, `argument_schema_id`,
  `required_caps`, `dtype_constraints`, `shape_constraints`, `workspace`, `capture_safe`; §10.1) and the deck's
  (`contract`, `when`, `priority`). Owner: the mentor (question 6); us (the loader, M3.1, with AMD's policy: transactional,
  hashes, exact profile, fail closed).
- Variant pruning at startup (①): ours is the TCM agreement at every launch (`flagos-spacemit-exec.cpp:170-180`) and
  the `__riscv_vlenb()` check on the core that runs the kernel (`flagos-spacemit-attn.cpp:443`): device facts checked
  at run time, not at load. With a package these belong in the startup filter (Q4).

### 4.5 Workspace management

- Fits: one workspace per backend, grown to the largest split, shared by the tiles, with the matmuls' quantized
  activations in a part of their own after the other ops' scratch (`flagos-spacemit.cpp:662-723`); an allocation
  failure fails the graph before any write. The deck's core "持有的设备临时内存" is this, allocated through the plugin's
  `buffer_alloc`; ours is `ggml_aligned_malloc` on the host side of the same DRAM (`:717`), which on the K3 is the same
  memory. TCM is per tile and never in the workspace (the deck has no local-memory notion; the mentor design had a
  `LOCAL_SCRATCH` cap, `plan.md` C4).

### 4.6 Graph capture and diagnostics

- Differs by K3 choice, G4 (R13: a departure with its reason). The deck's Tier-1 execution is one `launch` per kernel
  on a stream, capture an optional Tier-2 extra. On spine-runtime 0.6 that is not the K3's path: one launch per kernel
  with the host waiting for each costs tg128 14.2% on Qwen3-4B and 35.8% on Qwen3-0.6B (E3a, measured), and launches
  ordered inside the cores by `join` are not reliably correct with our kernels, because paused tiles move between
  cores and two of them use one TCM at once (E3b, E3b-diag, F16, measured). A bare launch and its wait cost 7.67 µs against a
  barrier's 0.435 µs (E3d). So our executor is the K3's capture: a split's steps replayed as one launch with a barrier
  after each (`flagos-spacemit-exec.cpp:160-213, :280-305`), the AI cores idle 1.3-1.8% of a token by the deck's
  measure (E3c), near the deck's 1.2% for AMD (R12). What differs from the deck's Tier-2: the capture is rebuilt every
  graph (no `graph_capture_begin`/`end` kept across tokens, no `end_update`; the step list is cheap, G7), and the
  "captured" unit is a step list of function pointers, not recorded launches.
- Gap, G5: the deck's `launch` carries a grid and typed arguments; ours carries a `spacemit_tile` with the tile's index
  and count, the agreed TCM, the shared workspace and the pair-wait handle (`flagos-spacemit-ops.h:13-25`), plus a
  launch-wide failure flag. The manifest's `block_size` and `num_warps` say nothing of a core per tile, its TCM or a
  partner tile, which our 1-row matmuls and the split attention rely on (and which E3b's queued launches broke). Q3's
  option (a): a standard first argument (the spine-runtime context) plus a small table of services the plugin exports,
  with the grid fixed at the core count for K3 contracts. Owner: the mentor (the ABI); us (the proposal).
- Failure reporting (Q7): a kernel returns false, the flag stops the other tiles at the next barrier, the graph
  returns `GGML_STATUS_FAILED` (`flagos-spacemit-exec.cpp:186-199`; `GET_ROWS` and `SET_ROWS` use it for a bad index).
  The deck shows only `SYNC_LAUNCH` and `TRACE_LAUNCH` for finding failures.
- Differs, G8: diagnostics. Ours are 18 `FLAGOS_SPACEMIT_*` variables, 11 of them `TEST_` (`grep getenv`, 2026-10-26);
  Common has none, and the only `GGML_FLAGOS_*` names in the tree are Denglin's `SYNC_LAUNCH` and `LOG_DECLINED`
  (`flagos-denglin.cpp:371, :2519`). Against the deck's five:

| Deck (`GGML_FLAGOS_`) | Ours today | Gap |
|---|---|---|
| `STATS=1`, calls per op and variant | `FLAGOS_SPACEMIT_TEST_OP_STATS=<file>`: per kind of graph, every step's time by op, role and shape, per tile (`flagos-spacemit.cpp:489-642`; `scripts/op-stats.py`); `FLAGOS_SPACEMIT_TEST_STATS` for the per-token budget (E3c) | more than the deck's counts, but a file at exit and other names; no variant names (the paths are counted only in the checks, `spacemit_mm_paths_take`) |
| `KERNEL_VARIANTS=0`, variants off for comparison | `FLAGOS_SPACEMIT_TEST_REFERENCE=1` (the portable references on the AI cores) and the per-op settings `FLAGOS_SPACEMIT_MM`, `ATTN`, `GEMV`, `Q4_1`, `TAIL`, `QUANTIZE` | the same purpose, one switch per op kind instead of one for all |
| `DISABLE_OPS=...`, named ops back to the CPU | `FLAGOS_SPACEMIT_DISABLE=1` disables the whole provider (`:958-962`) | no per-op fallback switch; Denglin has `FLAGOS_NO_<OP>` per op |
| `TRACE_LAUNCH=1`, find the failing kernel | `FLAGOS_SPACEMIT_TEST_FAIL_NODE=n` forces a failure at step n (`flagos-spacemit-exec.cpp:62-65`); a missing kernel and an unallocated operand are logged by op and name (`flagos-spacemit.cpp:677, :681`) | no trace of each launch or step by name |
| `SYNC_LAUNCH=1`, find the hanging kernel | `FLAGOS_SPACEMIT_TEST_LAUNCH=per-step`: every step its own launch with a host wait (E3a) | the same mechanism under a test name; the pair waits' 10 s give-up names the tile and the wait (`:49`) |

  Owner: us (cheap: a per-op fallback switch and a step trace; the names once the core owns them); the mentor (the
  names).

### 4.7 The plugin ABI, function by function

| Group (count) | Deck | Ours (code) | Fit |
|---|---|---|---|
| device (5) | `device_count`, `device_open`/`close`, `device_props`, `device_memory`; reports the target and memory properties | probe by `/proc/cpuinfo` and `/proc/set_ai_thread` (`flagos-spacemit.cpp:917-1000`), one device, props (`:847-861`), memory from `/proc/meminfo` (`:802`), the profile with target `riscv64-a100-ime2` and VLEN 1024 (`:1036-1066`) | fits; `open`/`close` is the backend's init and free with the stream |
| memory (3) | `buffer_alloc`, `buffer_free`, `buffer_base`; the core's workspace too | the buffer type (`:189-226`): huge pages from 2 MiB, else aligned malloc; the workspace is a separate `ggml_aligned_malloc` (`:717`) | fits |
| copy (2) | `memcpy`, `memset`, ordered on a stream; synchronous without one | `set_tensor`, `get_tensor`, `memset_tensor`, `cpy_tensor` (`:142-166`), synchronous; `set_tensor` repacks and `get_tensor` unpacks | differs: a plain `memcpy` cannot carry the layout (G6, Q2) |
| stream (3) | `stream_create`/`destroy`/`synchronize` | one persistent `spert::Stream` per backend, created at the first run and kept (`flagos-spacemit-exec.cpp:388-395`; `FLAGOS_SPACEMIT_STREAM=per-call` for comparison); every run waits on its `Future` (`:298`) | fits; spine-runtime orders nothing on a stream (F14), which the deck's copies assume |
| kernel (4) | `module_load`/`unload`, `kernel_get`, `launch` (typed arguments) | no modules: kernels are compiled-in functions in the op table; `launch` is `stream->launch(Grid{core_count}, spacemit_tile_main, &launch)` once per split (`:296`) | `module_load` and `kernel_get` arrive with M3; `launch` differs (G4, G5) |
| events (5, optional) | `event_create`/`destroy`/`record`/`wait`/`synchronize` | none (`:766-783`, `:847-861`) | fits (optional) |
| Tier-2 capture (5) | `graph_capture_begin`/`end`/`launch`/`destroy`, `end_update` | the executor's step list per split, rebuilt each graph (`flagos-spacemit.cpp:662-704`), replayed as one launch | the K3's main path, not optional (G4); no `end_update`, rebuilt instead |
| Tier-2 zero-copy (2) | `buffer_map`, `register_host_ptr` | the buffer is host memory, but hidden from the CPU (`is_host` false) because of the repack; host buffers accepted as operands (`:891`) | activations fit; weights do not (G6) |
| Tier-2 fusion switch | `fusion_supported(pattern)` | none | G7 |

### 4.8 The kernel package and manifest

- Our equivalent of a contract is an entry of the op table (`flagos-spacemit-ops.cpp:106-127`), of a variant a branch in
  the dispatch (G3), of `sha256` the compiled binary itself, of `priority` the order of the `if`s. The deck's `contract`
  names a FlagGems kernel signature (`flagos_mul_mat_q4_0_f32`), which our IME matmuls do not share: they read
  activations quantized to the layout's own blocks and weights in the 32-row interleaved layouts
  (`flagos-spacemit-ime-kernels.h:1-20`). A K3 package of native kernels would have to describe that layout, which no
  schema does (Q2's constraint: every variant of a contract must read the same layout, or the weight is stored twice).
- The deck's example of tuning by package alone (a wide Q4_0 GEMV variant, 12.0 to 22.9 t/s on AMD) has its K3
  counterpart in our 1-row path (staged weights and pair turns, tg128 7.84 to 9.18, `build.md` §7) and in `fit,staged`
  (pp128 +15-35%): tuning done in the dispatch and the orchestration, which under the deck would be code in the
  package's kernels or rules in the manifest, depending on where the tile services land (G5).

### 4.9 The five stages

| Stage | Deck | Ours (code, measured) |
|---|---|---|
| ① startup | scan `flagos-providers/`, dlopen, ABI check, open the package, sha256, prune variants | the compiled-in descriptor probed once (`:950-1000`): A100 cores counted, the AI-thread gate checked; no package; the `FLAGOS_SPACEMIT_DISABLE` switch |
| ② model load | buffers through L0; weights used in place | weights repacked into our buffer on huge pages, once (G6); the KV cache in a CPU buffer (ACCEL) or ours (IGPU), the model's embedding mapped on the CPU (E2: 2,350 MiB ours, 304 MiB CPU for Qwen3-4B) |
| ③ placement | the contract check per node, the plugin not asked | `spacemit_find_op` per node; the plugin and core are one, but nothing runs (fits) |
| ④ planning | fusion scan, plan cache; redone only when the structure changes | none; the step list is rebuilt per graph (G7, 0.55 ms per token) |
| ⑤ every token | bind, grid, `when`, a preceding launch, optional capture and replay | one launch per split: the steps bound to tensors by the kernels themselves, the grid the 8 tiles, the paths chosen inside the steps (G3), the quantize step first, the capture the main path (G4); 2 splits per token, the embedding lookup on the X100 cores |

### 4.10 Deployment and acceptance

- Deployment differs (G9): the provider is built in with `GGML_FLAGOS_SPACEMIT=ON` (`ggml/CMakeLists.txt:204`), or as a
  ggml backend module found through `GGML_BACKEND_PATH` (E4, measured); there is no `flagos-providers/` scan and no
  `GGML_FLAGOS_PROVIDER_PATH`; the device is `FlagOS:SpacemiT:0` (`:973`), the deck's `FlagOS-<chip>0`. The package
  directory, when there is one, is `share/ggml/flagos/spacemit/k3-a100/` (Q6).
- Acceptance fits in substance. `test-backend-ops` passes for every claimed op with the RVV and with the reference
  kernels (the runner's `m2e` milestone, `tools/spacemit_check.py`); it has no `MUL_MAT` case that fits the Q4_0 IME
  layout (`plan.md` §1), so `flagos-check-spacemit` compares those against the CPU at the models' shapes. The whole
  model's KLD against the CPU is 0.0025 on Qwen3-4B, 0.0047 on Qwen3-0.6B (measured 2026-10-24 and 26), below the
  deck's 0.01. llama-bench baselines are recorded per milestone (`build.md` §7), against the CPU and the upstream IME
  path in the same session.
- The deck's hostsim (2220/2220 with no hardware) is our host executor with the portable references on the Mac's
  simulated device (`provider.cmake:72-85`): the provider checks, `test-backend-ops` and the layer cases run there
  before every K3 run (`PROJECT.md` item 21, repeatedly).

### 4.11 Device type and kind (G10)

The deck's K3 row says nothing of the ggml device type. Ours reports ACCEL, IGPU behind a test switch
(`flagos-spacemit.cpp:830-845`); D3 is IGPU since 2026-10-26 and the flip waits on F8's two gaps. FlagOS `caps.kind` is
`cpu_accelerator` (`:1028`, D8 open). Under the deck's ABI `device_props` would carry the type; nothing changes
structurally.

## 5. What the review confirms, and what it changes

- Confirmed: the provider is the deck's "SpacemiT K3 plugin" at the stage the deck gives it: builds, runs, native
  kernels kept, the runtime-plugin split to do. Nothing in today's structure contradicts the deck's three-way split
  (F15 already said this for the execution model); every gap is either waiting on an artifact only the mentor has (the
  ABI header, the loader, the manifest schema: G1, G2, G3, G5) or is a K3 choice with measured reasons (G4, G6).
- Changed by this review: nothing in the code. Three cheap alignments inside the provider need no ABI and are proposed
  (§8): a per-op CPU fallback switch, a step trace by name, and a step-list cache measured against the 0.55 ms.
- Not changed: the decision to keep one launch per split (Q1, user 2026-10-20) and the weights' storage (D4).

## 6. The M3 decision (user, 2026-10-26) and its structural consequences

**Decision:** for the ops that FlagGems has a kernel for and FlagTree's SpacemiT backend can compile, the provider
uses that kernel; every other op keeps today's IME and RVV kernels as they are. Recorded in `PROJECT.md` (D9, question
6, item 19), `plan.md` (D9, M3) and `structural-experiments.md` (Q6). What question 6 still asks the mentor (which
compiler copy, which manifest schema, x86 or K3 builds, the success criterion) stays open. **Deferred (user and mentor, 2026-10-26):** M3 and the FlagGems kernel work
start only after the structural work on the backend (the IGPU switch, the plugin split, and what the mentor's answers in
§7 require) is done; nothing of M3.0 runs before then.

### 6.1 Which of our ops FlagGems has (code: `kernels/aot/manifest.json` and `plan.md` M3's list of the shared kernels)

| Op we claim | FlagGems shared kernel | Fit with our claim |
|---|---|---|
| `ADD`, `MUL` | `flagos_add_f32`, `flagos_mul_f32` | the broadcast forms the kernel handles only (AMD restricts `MUL` to contiguous repeats, `flagos-amd.cpp:5802-5804`); ours claims every `ggml_can_repeat` layout |
| `RMS_NORM` | `flagos_rms_norm_f32` (`BLOCK` 4096 covers Qwen3's rows), and `flagos_rms_norm_mul_f32` fused | yes |
| `ROPE` | `flagos_rope_neox_f32` | NEOX only; our claim includes NORMAL mode (Qwen3 uses NEOX) |
| `SET_ROWS` | `flagos_set_rows_f32_f16` | F32 into F16 only; ours claims the four F32/F16 pairs and I32 or I64 indices |
| `GET_ROWS` | `flagos_get_rows_f32`, `_q4_k_f32`, `_q6_k_f32` | F32 as ours; the Q6_K one is the token-embedding lookup, the last op on the X100 cores (`plan.md` M3's later list), a candidate for taking the token to 1 split |
| `GLU` SwiGLU | `flagos_swiglu_split_f32` | the split form; ours also claims the single-tensor form |
| `FLASH_ATTN_EXT` | `flagos_flash_attn_decode_f32_f16` | one-token calls only; prompt processing has no shared kernel (the AMD overlay was not checked), so the tiled kernel stays RVV |
| `MUL_MAT` | `flagos_mul_mat_q4_0/q4_k/q6_k_f32` (+ batched), on GGUF's block layout | **not usable on our weights**: the IME layouts (`flagos-spacemit-ime-kernels.h:1-20`) are not GGUF's, and the matmuls are 95% of a token (E3c). By the rule's second half they stay IME; this is the one op where both halves of the rule apply, so it should be confirmed (§7, Q13) |

So the AOT candidates are the small ops and one-token attention: 3.3 ms of a Qwen3-4B token for everything but the
matmuls and attention, and attention 1.4 ms at depth 0, 43% of a token at depth 4096 (E3c and the baseline,
`fusion-and-splitting.md` §4; measured).

### 6.2 The launch model is the structural question

- A FlagTree kernel is one Triton program per spert tile, launched as `stream.launch(Grid(gx, gy, gz), fn, args...)`,
  with `tl.program_id` read from `spine_grid(ctx, axis)` (`plan.md` M3 facts, FlagTree `driver.py:82-138, 182-201`,
  read 2026-10-14). Inside our one launch of 8 tiles a kernel compiled for "one program per row" would see a grid of 8
  and compute 8 rows. So either each AOT op is a launch of its own (M3.1's design: the executor cuts a launch at an AOT
  node), or the kernel loops over its rows inside a tile, a variant FlagGems does not have today (not verified; M3.0
  should ask FlagTree for it, or check `spine-triton`).
- The cost of a launch of its own in a model run is 13.4-15.0 µs (E3a, E3c; measured, every step launched). A Qwen3-4B
  layer has 16 ops FlagGems covers (4 `RMS_NORM`, 4 `MUL`, 2 `ROPE`, 2 `SET_ROWS`, 2 `ADD`, 1 SwiGLU, 1 attention),
  576 per token: about 8 ms more per generated token, about 9% of tg128 (estimate from E3a's per-launch cost; E3a
  itself measured all 1,086 steps as launches at +14.6 ms, -14.2%); in prompt processing the launches cost 0-2% (E3a).
  M3.2's rule (an AOT op in the default path only within 1% of tg128 and pp128) would then exclude every small op in
  generation unless the kernel runs inside the captured launch.
- What this means for the deck comparison: the user's decision is the deck's own path (FlagGems kernels through the
  package), and the K3's capture-only execution (G4) is what makes it expensive per kernel. The deck's Tier-1 cost model
  (a launch hidden behind the previous kernel) does not hold on spine-runtime 0.6 (F15), so the package's kernels need
  the same in-launch form as the native ones to be cheap: Q3's services for native kernels, and a looping program for
  Triton ones (estimate; nothing built).

### 6.3 Other consequences

- One contract, two sources: `spacemit_find_op` must answer the same whichever source will run, so an op's claim must
  not depend on the package (the native kernel stays the fallback for every claimed op; AMD's `supports_op` depends on
  the loaded package instead, `flagos-amd.cpp:5801`, and claims nothing without one, `:5990-6006`).
- The choice per op belongs to the deck's `KERNEL_VARIANTS` switch: M3.2's `FLAGOS_SPACEMIT_AOT=<ops>` is its form
  until the core owns the names (G8).
- The Mac's simulated device cannot run riscv64 `.so` kernels; the loader's negative matrix runs there, the kernels
  only on the K3 (M3's tests).
- The speed rule: `plan.md` M3 says an AOT kernel enters the default path only where it is at least as fast. The
  decision as stated ("use FlagGems' kernels for the ops they have") does not say whether that rule stands; §7, Q14.

## 7. Questions this review adds for the mentor (continuing `structural-experiments.md` §2)

- **Q11.** The deck's K3 row ("构建通过 · runtime 插件待做"): is that this provider, and is the runtime-plugin split
  expected from us, with the ABI header (the deck names it `flagos-runtime.h`), the loader and `amd-rt` as the reference? Nothing of them is in the branches
  we have (F1). Until then the split cannot start (G1). Closed 2026-10-26: no separate header; the split is ours,
  K3-only (D10).
- **Q12.** For a plugin whose kernels run inside one captured launch with per-tile services (G4, G5): will the core's
  capture policy treat capture as the main path for such a plugin, and can the launch ABI carry a standard context
  argument plus a service table (Q3 (a))? Also F15's question: how the core waits when a plugin has no events.
- **Q13.** The matmuls under the M3 decision: FlagGems' quantized matmuls read GGUF's layout and cannot read the IME
  layouts; we read the decision as keeping the IME matmuls (the rule's second half). Confirm, or choose Q2 (d) (kernels
  that read GGUF's layout) with its measurement (E5, M3.0's extra kernel).
- **Q14.** The speed rule for AOT kernels: the default path only where an AOT kernel is at least as fast (M3.2's 1%
  rule), or FlagGems' kernels by default where they exist, with the native ones kept for comparison? The first keeps R1's
  "without losing performance"; the second is the deck's path. With separate launches the difference is about 9% of
  tg128 on Qwen3-4B (estimate, §6.2).

### 7.1 Priority among all pending questions (2026-10-26)

Ranked for the mentor after the decision of the same day (user and mentor) that M3 and the FlagGems kernel work wait for
the structural work on the backend. Criteria, in order: what blocks the next structural work; what shapes the ABI or the
package while they are being defined (hard to change once other vendors' plugins exist against them); what changes
accuracy or delivery; reports and confirmations last. Q numbers are `structural-experiments.md` §2's and this file's,
"question n" is `PROJECT.md` §7's, D are decisions.

**Status after the mentor's decisions (2026-10-26 evening, D10):** rank 1 closed (no separate `flagos-runtime.h`; the
K3-only split is ours, the loader ours to write), rank 3 closed for ownership (`flagos-spacemit-weights.{h,cpp}` is the
plugin's), rank 2 turned into design inputs for our own interface (`plan.md` S1); ranks 4-9 stand.

| Rank | Ask the mentor | Why here | Ours meanwhile |
|---|---|---|---|
| 1 | Is the K3 to be delivered as a runtime plugin against the deck's C ABI (the deck's K3 row, "runtime 插件待做")? If yes: the ABI header (the deck's page 8 names it `flagos-runtime.h`; we have not seen the file), the core's loader, the `amd-rt` and hostsim sources, and the ABI's version negotiation and unload protocol (V2.0 §5.1; the K3 plugin holds a core grant and TCM). If no, the split leaves the list (Q9, Q11, Q10) | decides the largest structural item (G1). Nothing that runs today needs the ABI: the provider is a ggml backend on V2.0's source ABI and stock upstream loads it as a module (E4); the split changes the build (a core without `libspert`), not speed or correctness. The ABI exists on the mentor's side by the deck (`amd-rt` 2238/2238, hostsim 2220/2220) and in no tree we have (F1) | nothing until the answer; §3's mapping is ready |
| 2 | The K3's requirements on the ABI before it freezes: capture as the main path for a plugin like ours (the core's policy; Q1's core side, Q12), `launch` carrying a standard context plus a service table (tile index and count, the agreed TCM, the workspace, pair waits, a failure flag; Q3 (a)), a failure word (Q7 (b)); how the core waits without events is answered by V2.0 §6.4 and the deck (§7.2) | measured reasons (E3a, E3b, E3d, F16); cheap in a new ABI, costly once other plugins ship against it; the one place the deck's cost model does not hold on the K3; V2.0 keeps replay with the provider and plans it per split for AMD (§7.2) | one proposal page, no code |
| 3 | Weight layouts in the ABI and the package (Q2, question 2, D4): options (a)-(e); the Q4_1-to-Q8_0 requantization (accuracy against 1.0-2.4% of generation) | shapes `buffer_alloc` and `memcpy`, and every variant of a contract must read one layout; built on since M2b without confirmation; a documented departure from V2.0's weight model (§7.2) | E5, E6 if the mentor wants measurements |
| 4 | After the IGPU switch: C5a and C5b (U1, U2) kept for other ACCEL providers, proposed upstream, or dropped; `caps.kind` (question 1, D8) | the switch is our next code item (F8's two gaps); its tail needs the mentor's word; small | the switch itself |
| 5 | The `when` vocabulary and the manifest schema (Q4; part of question 6): is `when` Common's `flagos_kernel_variant` serialized (ranges and features, which cover our row counts), and will it carry strides, layout and scratch (§7.2); format 2, v3 or the deck's | input now, while the schema is being defined; our own use (native kernels in a package, M3) is deferred | the Q4 table, on paper |
| 6 | Co-building and ownership: where the plugin's source lives (FlagOS, SpacemiT, this fork), SpacemiT's parallel ggml-spacemit (question 7, Q8); the mentor design's §21 has it co-maintained | delivery, not the work | - |
| 7 | Reports to SpacemiT (question 8): the Q4_1 conversion's integer zero point, the 1-row Q4_1 kernel's wrap, the TCM library, the X100 crash, the NaN | SpacemiT's to fix, through the mentor; nothing of ours waits on them | the report text when asked |
| 8 | Confirmations of decisions in force: D5 (one launch per split, Q1), question 3 (closed `libspert`, acceptable by R3), question 4 (copied kernels), D7 and question 5 (the Qwen3 targets; Qwen3.5-4B Q4_K_M later?) | formalities | - |
| 9 | Parked until the structural work is done: M3's remaining parts (question 6: compiler copy, schema, x86 or K3 builds, success criterion; Q6's package build and shipping), Q13 (the matmuls stay on the IME), Q14 (the speed rule) | deferred (user and mentor, 2026-10-26) | - |

### 7.2 The design document V2.0 against the ranking (read 2026-10-26)

`ggml-flagos-provider-design.pdf` V2.0 (2026-09-22, the design document of the code we build on; "Design" in
`plan.md`), read against §7.1 at the user's request. It changes no rank. It confirms who owns the ABI, answers one
sub-question of rank 2, and sharpens the asks of ranks 1, 2, 3 and 5.

- **Rank 1, the owner confirmed.** V2.0 §1.2 lists "a stable, hot-unloadable third-party binary plugin ABI" among what
  the current system is not; §5.1 calls `flagos_provider_v1` a source ABI "not declared a cross-compiler stable plugin
  ABI" and says a real one "should be defined separately: a pure C POD descriptor, opaque handles, an allocator,
  version negotiation and unload"; §15.1 puts "pure C plugin ABI evaluation (an RFC for opaque handles, versions and
  unload)" at P2 on Common's roadmap, §17.3 as a future RFC conditional on binary distribution and unload. The deck's
  `flagos_runtime_v1` is that RFC realized on the mentor's side (inference from the deck's `amd-rt` and hostsim
  figures; F1). Two refinements of the ask: the ABI's version negotiation and unload protocol matter to the K3 plugin,
  which holds a core grant and TCM that unload must release; and since the plugin is delivered as source (R3), what the
  split buys the K3 is build separation (a core without `libspert` and without the IME toolchain), not binary delivery.
- **Rank 2, one sub-point answered, one sharpened, one strengthened.** How the core waits without events is answered by
  both documents: V2.0 §6.2 makes events optional, §6.4 has dependencies guaranteed by events or `synchronize`, §6.5
  serializes `graph_compute` inside the context when the runtime handle is not concurrent; the deck's required
  `stream_synchronize` and "没有时 core 退回同步执行" say the same. It leaves the ask. The TCM requirement has a slot:
  V2.0 §9.4's package envelope carries "static/dynamic shared/private/scratch" per kernel and §7.2 checks a launch's
  shared and scratch limits in `supports_op`, so the agreed TCM size can be a per-kernel scratch requirement against
  the device's per-core TCM; only a core per tile, the partner tile and a grid fixed at the core count remain new
  (Q3 (a)). And capture: V2.0 keeps replay with the provider (§2.4 "replay" with binding, epoch, capture-safe and
  invalidation rules; §12.1 the provider owns the native graph replay object; §8.3 `capture_safe`), which is today's
  executor; its AMD plan P3 (§14.6) is replay of larger regions "along real split boundaries", our unit. So the ask is
  the policy under the deck's core-driven capture (capture by default for this plugin; `end_update` for the arguments
  that change per token) and the service table, not whether capture is legitimate.
- **Rank 3, sharpened.** V2.0 documents the opposite weight model: GGML owns the GGUF and weight semantics and the
  visible buffer keeps GGUF's bytes, the provider keeps a private dequantized cache with its own epoch (§12.1; §12.4's
  "dequant weight" cache tier; §14.1 AMD's F16 cache), and kernels adapt to GGML's layouts at build time (§9.1 "GGML
  dtype/quant/layout adapter", §13.1 "quant decode 与 packing"). D4 is therefore a documented departure (Q2's (a)-(c)
  against the design's (d) and (e)), taken for memory (about +2.35 GB for (e) on Qwen3-4B) and for the IME kernels'
  interleaved layout; the ask is the same, with the departure named.
- **Rank 5, narrower.** Common already has a variant vocabulary: `flagos_kernel_variant` (`flagos-target.h:88-108`:
  ranges on m, n and k, required and forbidden features, minimum vector bits and lanes, target, engine mask, score)
  with `flagos_select_kernel_variant` (`:142`, `flagos-target.cpp:154`; V2.0 §17.2), used by the AMD provider
  (`flagos-amd.cpp:933-948`) and the Common check (`tests/check_provider.cpp:138-155`). It covers our row-count ranges
  and equalities (`min_m = max_m = 1`) and, with RVV and IME feature bits (`plan.md` C2), the device facts; it lacks
  strides, the cache's cell layout, the weight's layout and the local-memory size, which V2.0 §9.4's envelope foresees
  ("shape/layout constraints", "scratch"). The ask becomes: is the deck's `when` this struct serialized, and will it
  carry layout, stride and scratch conditions?
- **Unchanged:** ranks 4, 6, 7, 8 and 9. V2.0 §13.2's provider skeleton (`flagos-<vendor>-api.h`, `.cpp`, `-aot`,
  `-profile`, `provider.cmake`, `tools/`, `tests/`) is our layout but for the two M3 files; §13.4's deliverables are
  met except the AOT package (parked); §15.6's gates 9 (no development environment variable on the default path) and 11
  (no error reported as success) hold today (code: the defaults read no switch; a failed step returns
  `GGML_STATUS_FAILED`).
- **Also read there:** V2.0 §14.4 measured AMD at 17-24% behind native with about 700 launches per token; the deck's
  page 9 shows 9% (45.3 against 41.2 ms). Neither says what closed the gap; V2.0's P1 (the Q6_K vocabulary projection)
  and P3 (replay regions) are the candidates (reading). V2.0 §5.4's out-of-tree registration must precede the first
  `ggml_backend_flagos_reg()`, which llama.cpp's static build calls at startup (`upstream-review.md` B1); the deck's
  loader scanning `flagos-providers/` at startup is what resolves it.

## 8. Proposed next steps (no code until approved; `CLAUDE.md`'s workflow)

Reordered 2026-10-26 (evening) after the mentor's decisions (D10): the K3-only core first, then the plugin; M3 and the
FlagGems kernels deferred.

1. Record the decisions (done): D10 in `PROJECT.md` and `plan.md`, S1 and S2 in `plan.md`.
2. Proposed before S1 (the user's call): the IGPU switch (D3): close F8's two gaps, flip the default, rerun E2's checks
   and the acceptance runner; small, independent, and it touches code S1 and S2 move.
3. S1, the K3 core (`plan.md` S1): first the core–plugin interface as a design note (the required functions, the capture
   replay, the per-tile services: §7.1 rank 2's requirements), then L2 completed behind it, the L1 draft (the plan cache
   keyed by structure, measured against the 0.55 ms per token), the plugin loader, the workspace and capture driver, the
   diagnostics under the deck's names. Each step a spec for the coder, checked on the Mac's simulated device before the
   K3.
4. S2, the K3 plugin (`plan.md` S2): device, memory with the repack (`flagos-spacemit-weights.{h,cpp}`), copy, stream,
   launch, the capture replay, the native kernels, the tests; loaded through S1's loader; the acceptance runs again.
5. After S2: M3.0 on the launch question (§6.2), then the loader's package check and the package (M3.1); fusion (M4) by
   measurement.

## 9. Log

- 2026-10-26: written from the deck's pages 6-8 (text extracted), the provider at `65280e5`, Common at `5794e12` plus
  the fork's three lines, the mentor design v0.3 on `kevin/feature/flagos-multi-provider-backend`, and the findings in
  `structural-experiments.md`. The M3 decision recorded the same day in `PROJECT.md`, `plan.md` and
  `structural-experiments.md`.
- 2026-10-26: the user and the mentor defer M3 and the FlagGems kernel work until the structural work on the backend is
  done; §7.1 ranks the pending questions for the mentor, §8 reordered; the review's reference to a stand-alone
  reproduction removed (that series was stopped and reverted the same day; F16's evidence stands).
- 2026-10-26: the design document V2.0 read against the ranking (§7.2, user request): the order stands; the ABI
  confirmed as Common's roadmap item; rank 2's waiting sub-question answered; ranks 1, 2, 3 and 5 sharpened; §1's
  generations put in date order with V2.0 added.
- 2026-10-26, evening: decided with the mentor (D10): no separate `flagos-runtime.h`; the core / plugin split done
  K3-only in our tree, the core first (S1), the plugin after (S2) with the weights code in it; §0, §3, §4.1, §4.4, §7,
  §7.1 and §8 updated; the content in `plan.md` S1 and S2.
