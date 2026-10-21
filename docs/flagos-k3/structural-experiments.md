# Structural experiments: plugin ABI, device type, upstream footprint

Started 2026-10-19. This file holds the mentor's requirements for the provider's structure (how it fits FlagOS's plugin
design and llama.cpp with as little upstream change as possible), the open structural questions, what has been found,
and the experiments that answer the testable questions before they go to the mentor. Kernel work and milestones stay in
`plan.md`; measurements of the kernels stay in `build.md` §7. The experiments' code: two test-only switches in the
provider, checks in `flagos-check-spacemit`, and `scripts/structural-experiments.sh` (§5).

## 1. The mentor's requirements

| ID | Requirement | Source |
|---|---|---|
| R1 | Run llama.cpp inference on the K3 with the least change to upstream llama.cpp and ggml, without losing performance | mentor, in a meeting (reported 2026-10-19) |
| R2 | One FlagOS core for every chip (`libggml-flagos.so`: no vendor SDK linked, about 500 KB); vendors deliver only a runtime plugin and a kernel package; llama.cpp uses the core through the ggml backend interface and is not changed ("llama.cpp 不用改") | deck `FlagOS端侧与车载智算.pdf` (not in the repo; its slides are images), architecture slide |
| R3 | The plugin `libflagos-provider-<chip>.so`: links only the vendor's SDK, pure C header `flagos-runtime.h` (`flagos_runtime_v1`), may ship as a binary only. One export, `flagos_provider_get_entry_v1(host)`, returning the ABI version, kind RUNTIME, a name and a function table; the host gives it logging and the plugin directory. **Tier-1, required, 17 functions:** device (5: `device_count`, `device_open`/`close`, `device_props`, `device_memory`; reports the target and memory properties), memory (3: `buffer_alloc`, `buffer_free`, `buffer_base`; the core also allocates its workspace with them), copy (2: `memcpy`, `memset`; ordered on a stream, no stream means synchronous), stream (3: `stream_create`, `stream_destroy`, `stream_synchronize`), kernel (4: `module_load`/`unload`, `kernel_get`, `launch`; argument types PTR, I32, F32, ...). **Optional:** events (`event_create`/`destroy`/`record`/`wait`/`synchronize`; without them the core runs synchronously). **Tier-2, optional:** graph capture (`graph_capture_begin`/`end`/`launch`/`destroy`, `end_update` for updates in place), zero-copy (`buffer_map`, `register_host_ptr`, for APU and SoC unified memory), a fusion switch (`fusion_supported(pattern)`; the package must also hold the fused kernel) | deck, ABI slide |
| R4 | The kernel package, released apart from the `.so`: `share/ggml/flagos/<hw>/<target>/` with `manifest.json` and one binary per variant (AMD 890M: 140 kernels, 7 of them variants). Per variant: `name`, `file`, `sha256`, `block_size`, `num_warps`, `argument_count`, `contract`, `priority`, `when` (example: `flagos_mul_mat_q4_0_f32_wide_b8_w1`, priority 20, `when` k a multiple of 256 and align a multiple of 16). Kernels are FlagGems sources compiled per target by FlagTree (gfx1150 HSACO, K3 ELF). Tuning changes only the package: a wide Q4_0 GEMV variant took generation from 12.0 to 22.9 t/s (the deck says "decode") | deck |
| R5 | "判断和调度都在 core，厂商实现橙色这一层": the core judges and schedules, the vendor implements only the runtime layer. Per stage: ① startup: the plugin loader scans `flagos-providers/`, dlopens, checks the ABI version, opens the package, checks sha256, drops variants the device cannot run. ② model load: L0 allocates buffers; "统一内存可零拷贝（权重不必再复制）" (unified memory is zero-copy, weights need not be copied again). ③ placement, per node: L2 checks the kernel contract (is the kernel in the package; shape, alignment, resources?); if not, the node stays on the CPU (it only answers, runs nothing). ④ planning: L1 scans fusion patterns (the provider enables them and the package has the fused kernel), caches the plan (a hit costs about 12 µs). ⑤ every token: L2 binds the arguments, computes the grid, picks the variant whose `when` matches the facts, runs a preceding launch when needed (e.g. dequantizing into the workspace); optionally captures the whole graph and replays it per token. ①② run once; ③④ are redone only when the graph's structure changes; ⑤ runs every token | deck, flow slide |
| R6 | Deployment and acceptance: the plugin in `flagos-providers/` next to the core or at `GGML_FLAGOS_PROVIDER_PATH`; the package at `GGML_FLAGOS_KERNEL_DIR`; `llama-cli --list-devices` shows `FlagOS-<chip>0`; accepted when `test-backend-ops` passes entirely, the whole model's KLD is below 0.01, and a llama-bench baseline is recorded. Diagnostics (prefix `GGML_FLAGOS_`): `STATS=1` counts calls per op and variant, `KERNEL_VARIANTS=0` turns variants off for comparison, `DISABLE_OPS=...` sends ops back to the CPU, `TRACE_LAUNCH=1` and `SYNC_LAUNCH=1` find a kernel that fails or hangs | deck |
| R7 | The deck's status of the K3 plugin: "libspert runtime；过渡期保留原生 RVV/IME；构建通过 · runtime 插件待做" (spine-runtime; native RVV/IME kept during the transition; builds; runtime plugin still to do). AMD's `amd-rt` plugin: HIP, 35 KB, 488 lines, 2238/2238 on the device; the hostsim reference plugin (kernels simulated on the CPU): 2220/2220 | deck, architecture slide |
| R8 | Later, kernels and their files go to FlagOS upstream, the executor and runner files to llama.cpp upstream | user, 2026-10-16 |
| R9 | Settled earlier: device type ACCEL (D3, mentor, 2026-10-08); C5a and C5b approved for the fork (2026-10-14); the copied kernels may be changed (2026-10-08); the mentor's design §14.1: in-tree code is "reference, not boundary" | `PROJECT.md` §2, §7 |
| R10 | The AMD provider is a reference, not a template: it is still being developed | user, 2026-10-19 |
| R11 | Design review of ggml-flagos focused on kernel fusion and on the shim between Common and the providers | `PROJECT.md` §1, `upstream-review.md` |
| R12 | The measure of the framework's cost: "框架开销约 1%" (about 1%), defined as the time per token the device waits idle. AMD 890M, Qwen3.5-4B Q4_K_M generation: ggml-flagos 45.3 ms per token (kernels 44.7 ms, framework idle 0.6 ms, 1.2%) against the native HIP backend's 41.2 ms; the GEMVs (82% of generation) at about 1.0x native; the remaining gap is kernels (GDN 1.7x slower, SSM_CONV 1.18x), with 98-99% of native expected once they are fixed (90-92% today) | deck, page 9 (its text, extracted 2026-10-19) |
| R13 | The deck guides the macro design and is not a standard to follow to the letter: designs differ from chip to chip, so a K3 choice that departs from it stands when its reason is stated | user, 2026-10-20 |

## 2. Questions for the mentor

Each with the problem, the options, a recommendation, and what answers it. Q1, Q2, Q4 and Q5 can be tested first (§4);
Q3 and Q6-Q10 are design questions only. Related numbering in `PROJECT.md` §7 is given where it overlaps.

**Q1. Does a K3 plugin need Tier-2 graph capture?**
- Problem: today a split's steps run in one spine-runtime launch, a barrier between steps; a token takes 2 splits and
  about 1,060 steps (estimated; E3c counted 1,086, §6 run 4): about 1 ms of barriers at 0.94 µs each (`build.md` §7;
  E3c measured 0.6 ms), about 1.1% of a token's 88.5 ms (an estimate; E3c: the AI cores idle 1.3-1.8% of a token by the
  deck's measure, the barriers 0.7% of it), near the deck's 1.2% for AMD (R12). A plugin with the required
  functions alone gets one `launch` per kernel on a stream, which we read the ABI as ordering (the deck states the
  order only for copies, F15). spine-runtime's documentation promises no order between launches (F14): the plugin must
  order them itself, by waiting for each launch on the host (11.2-11.5 µs per launch against a barrier's 0.89-0.94 µs:
  about 12 ms per token, tg128 about -12%; estimate) or, if the board's spine-runtime has it, by making each launch's
  tiles wait inside the cores for the previous launch (`Context::join`; cost unknown).
- Options: (a) the K3 plugin implements Tier-2 capture: record the launches, replay them as one spine-runtime launch with
  a barrier between kernels, which is today's executor; (b) a "launch list" call added to Tier-1; (c) accept the cost.
- Recommendation: (a), unless E3b shows ordered launches cost about as little as the barriers. AMD's provider has
  replay (hipGraph) only as an opt-in experiment (F12).
- Measured 2026-10-20 (§6 run 3, log; Qwen3-4B Q4_0): the required functions alone (one launch per kernel) are either
  slow (E3a, the host waiting for each launch: tg128 -14%) or not reliably correct with our kernels on spine-runtime
  0.6 (E3b: the 1-token layer cases differed from one launch in 5 of 5 runs; in the same runs the runtime put two tiles
  of one launch on one core, which our tiles are not written for. E3b-diag found the cause, run 5: paused tiles resume on
  other cores and two of them, on two cores, use one TCM at the same time, which corrupts the 1-row matmuls' staged
  weights; F16).
  So by E3's agreed rule the K3 needs graph capture (for generation; the scope is in F15), and today's executor is its
  implementation: the captured launches replayed as one spine-runtime launch with a barrier between kernels, the
  arguments that change per token (the KV length) updated in place (`end_update`), the plugin's plain `launch` kept as
  a slow fallback (E3a). For the mentor: graph capture required on spine-runtime 0.6 (the one runtime tested); kernels
  launched one at a time cannot count on a core of their own (E3b saw two tiles of one launch on one core; FlagTree's
  kernels in M3 not tried; ours count on it, and today's one launch per split has shown no sign of breaking it, though
  it is not counted there); for SpacemiT: ordered launches on a stream, or tile i kept on core i, might make one launch
  per kernel cheap enough (the launch's own cost would then decide; not measured).
- Decided (user, 2026-10-20): E3b is not fixed. Fixing it would need (E3b-diag, run 5) tiles that never rely on owning
  a core or a TCM across a pause: TCM taken after a tile's last pause before using it, and no spin waits between tiles
  that may land on one core, so no pair turns (worth 3.5-5% of generation in the 1-row matmuls and about 4% at depth
  4096 in the split attention) or two tile models side by side, for a launch model the K3 does not need and whose
  launches cost 7.3-316 µs each here (E3d). Today's
  executor stays the K3's graph capture; SpacemiT is asked for ordered launches on a stream, or tile i kept on core i.
  E3c is not needed for this decision (it would measure today's overhead the deck's way, R12). E3 against the deck: F15.
- Where the barrier design comes from: ggml-cpu runs every node on every thread with a barrier after each node
  (`ggml/src/ggml-cpu/ggml-cpu.c:3088-3117`; the in-tree IME path runs inside that model); ggml-spacemit moved it onto
  the AI cores as one spine-runtime launch per `graph_compute` with a grid barrier after every node (`plan.md:34`); D5
  (M2a) took ggml-spacemit's model with nodes expanded into steps and no waits inside a step (the pair waits came later).
  It gives: a barrier (about 0.9 µs) instead of a launch (about 11 µs) per step; the order between steps without the
  runtime ordering launches (F14); one workspace for a multi-step op; per-launch setup (the TCM agreement) once per
  split; one failure flag. It costs every step's wait for its slowest tile. In the deck's terms it is the K3's form of
  Tier-2 "图录制 · 逐 token 重放", with the step list rebuilt each graph instead of captured once.
- Answered by: E3a, E3b (E3c splits today's overhead the deck's way).

**Q2. How does a plugin keep weights in a vendor layout?** (`PROJECT.md` question 2, D4)
- Problem: the IME kernels need weights repacked (Q4_0 interleaved 32 rows x 256; Q4_1 with an integer zero point,
  lossy; Q6_K requantized to Q8_0, more bytes than `ggml_nbytes`), reads must convert back, and the CPU must never read
  the raw bytes (`is_host = false`). The ABI's `buffer_alloc` and `memcpy` carry no layout, and the flow slide assumes
  weights are used in place (R5 ②). A kernel package changes who owns the layout, not whether one is needed (F4).
- Options: (a) repack and unpack kernels as contracts in the package, run by the core at upload and read-back, with the
  size declared in the manifest; (b) a Tier-2 `tensor_set/get_with_layout` plus `alloc_size`; (c) the core marks weight
  buffers at `buffer_alloc` and the plugin repacks inside `memcpy`; (d) no repacking: kernels read GGUF's layout (the
  1-row path could interleave while copying into TCM; paths A and C would need new kernels); (e) AMD's way: GGUF's bytes
  in the buffer ggml sees plus a private repacked copy (about +2.35 GB for Qwen3-4B).
- Constraint for every option: variants are picked per token, so all variants of a contract must read the same layout,
  or the weight is stored twice.
- Recommendation: (a) while native IME kernels stay; (d) if E5 and the Triton kernel of the M3.0 spike read GGUF's
  layout fast enough.
- Answered by: E6, E5, M3.0's extra kernel.

**Q3. What does a kernel get per tile, and who sets the K3's grid?**
- Problem: our kernels use the tile number and count, one TCM size all tiles agreed on, the shared workspace, pair waits
  and a launch-wide failure flag. The core computes the grid (R5 ⑤) from the manifest's `block_size`; FlagTree's launcher
  passes a `spert::Context*` first, which has no pair waits and no failure flag.
- Options: (a) a standard first argument (the spine-runtime context) plus a small table of services the plugin exports
  (pair wait, failure flag, agreed TCM size), and a grid fixed at the core count for K3 contracts; (b) each kernel keeps
  its own state between tiles; (c) native kernels stay inside the plugin during the transition (but then the plugin
  dispatches, against R5).
- Recommendation: (a); also confirm I64 as an argument type (byte strides, large offsets).

**Q4. Can `when` express the K3's kernel choices?**
- Known from the flow slide: `when` is checked every token against "facts" (⑤), variants the device cannot run are
  dropped at startup (①), and `KERNEL_VARIANTS=0` implies a generic variant per contract.
- Our choices: the matmuls' 1-row path or paths A and C by row count; attention's one-chunk kernels (fewer than 64
  query rows), tiled kernel (64 or more, with TCM) or split (1 row, from n_kv 512, llama.cpp's cache layout;
  `split_shape`, `flagos-spacemit-attn.cpp`). They need ranges, equality, strides and contiguity and the weight's layout;
  `multiple_of` alone covers none. TCM availability belongs in the startup filter.
- Options: (a) extend `when` (ranges, enums, types, strides); (b) a C check per contract in the core (L2), `when` only
  choosing among variants; (c) a plugin callback `kernel_supports(desc)`.
- Recommendation: (b) plus small extensions from (a); whichever is chosen must give placement and execution the same
  answer (the provider's `spacemit_find_op` rule).
- Answered by: the Q4 table (§4, no board).

**Q5. Device type and upstream footprint** (D3)
- Problem: ACCEL runs at full speed only with C5a and C5b (22 lines in `src/llama-context.cpp`); without them
  generation was 30% slower (X3b on ggml-spacemit). GPU and IGPU need no llama.cpp change. Upstream's accelerators that
  run whole layers report GPU or IGPU, and Huawei's Ascend reports GPU (F5, F10).
- Options: (a) ACCEL, with C5a and C5b proposed upstream; (b) IGPU: no patches, ggml's "host memory" definition fits
  the K3 (F9), dropped next to a discrete GPU, the KV cache in the provider's buffer (two provider gaps to fix, F8);
  (c) GPU, as Ascend: no patches, competes for layers next to a discrete GPU; (d) ACCEL without patches (`-fa on`,
  generation -30%).
- Recommendation: decide after E2 (with E2-hp and E2-neg) and E1. If E2 matches today's speed with identical logits, (b)
  fits R1 and ggml's definitions best.
- Measured 2026-10-19 (§6, run 1; Qwen3-4B Q4_0 only): as IGPU or GPU the K3 matches today's speed with identical
  logits and no llama.cpp change; ACCEL without C5a and C5b loses 24-95% in generation and 8-52% in prompt processing.
  So (b), once the provider fixes F8's two gaps (confirmed by E2-neg); (a) only if the K3 must keep accelerating next to
  a discrete GPU.
- Measured 2026-10-20 (§6, run 2, E4; Qwen3-4B Q4_0 only): as IGPU the provider also runs from stock upstream
  llama.cpp, loaded at run time as a module, with the same speed and identical logits: no upstream change at run time
  (building the module still uses the mentor's CMake lines, §6 run 2).
- Measured 2026-10-20 (§6, run 4, E2-0.6B): on Qwen3-0.6B Q4_0 too, IGPU and GPU match today's speed with identical
  logits (pp512 at depth 4096 3.8-3.9% faster, as on 4B). Two models of one family and one quantization so far; F8's
  two gaps were not rerun on 0.6B.
- Answered by: E1, E2, E2-hp, E2-neg, E4.

**Q6. During the transition, do native RVV/IME kernels go in the package or in the plugin?** The deck says both: the
architecture slide keeps them in the plugin (R7); the flow slide lists "HSACO / ELF / 厂商格式" (vendor format) as package
binaries. Recommendation: the package (K3 ELF `.so` files with FlagTree's calling convention), so the core picks them like
any variant and Triton versions replace them one by one.
- Decided (user, 2026-10-20): the native kernels stay in the plugin for now (as R7 says), and the other experiments come
  first. When the K3 gets a package, its directory is `share/ggml/flagos/spacemit/k3-a100/` (`<hw>/<target>` =
  `spacemit/k3-a100`). How the package is built and shipped (FlagTree's CI cross-compiling on x86 or a build on the K3;
  released apart from git) goes to the mentor (`PROJECT.md` question 6).
- Package locations already in the tree: Denglin installs to `share/ggml/flagos/denglin` (`ggml-flagos/CMakeLists.txt:169`;
  run-time override `FLAGOS_KERNEL_DIR`, `flagos-denglin.cpp:301-302`), AMD takes `FLAGOS_AMD_AOT_KERNEL_DIR` at build
  time (a `manifest.json` required, `providers/amd/provider.cmake:18-22`) and installs to `share/ggml/flagos/amd`
  (`CMakeLists.txt:172`); the deck's variable is `GGML_FLAGOS_KERNEL_DIR`.

**Q7. How does a kernel report a failure?** Today `GET_ROWS` and `SET_ROWS` with a bad index make the graph return
`GGML_STATUS_FAILED` instead of aborting; the deck shows only `SYNC_LAUNCH` and `TRACE_LAUNCH`. Options: (a) `launch`
returns a status; (b) an error word in a buffer the core provides, checked at synchronization; (c) no reporting.
Recommendation: (b), which works for Triton kernels too.

**Q8. Where does the plugin live, and is a closed `libspert` acceptable?** (`PROJECT.md` questions 3 and 7)

**Q9. Where is the plugin-split code?** `flagos-runtime.h`, the core's loader and `amd-rt` are in none of the branches we
have (F1).

**Q10. Where does each part of our code go under the deck's split?** R8 sends the executor and runner to llama.cpp, but
the deck keeps llama.cpp unchanged and puts the core in `libggml-flagos`, the runtime in the vendor's plugin and the
kernels in the package. Our executor (`flagos-spacemit-exec.cpp`) maps to the plugin's Tier-2 capture; the acceptance
runner to the plugin's tests.

## 3. Findings so far

**F1. The plugin ABI exists only in the deck** (checked 2026-10-19). No `flagos_runtime_v1`, `flagos-runtime.h` or
`flagos_provider_get_entry_v1` in any local branch, including `kevin/*` (newest 2026-09-22), in `~/flagos-kernels`, or
in Downloads, Documents and Desktop. The AMD provider in this tree is still compiled in (6,201 lines), not the deck's
488-line `amd-rt`.

**F2. Tiers.** Tier-1 is the 17 required functions; events are optional; Tier-2 is graph capture, zero-copy and the
fusion switch. There is no Tier-1 graph capture: without Tier-2, the core calls `launch` for every kernel every token.
The ABI's authors decide which tier a function is in; a plugin only chooses whether to implement Tier-2. The core most
likely detects Tier-2 by non-null entries in the function table (inferred; the deck does not say).

**F3. Our provider against the flow slide.** ③'s contract check is `spacemit_find_op` in `supports_op`; ⑤'s choice by
`when` is our choice by shape (row count, `split_shape`); ⑤'s preceding launch into the workspace is our multi-step ops
(activations quantized, then the matmul); ⑤'s optional capture and replay is our executor (one spine-runtime launch per
split). It differs in the weights repacked in place (Q2) and in the tiles and TCM (Q3).

**F4. Weights.** The IME layouts are why the provider keeps weights (`flagos-spacemit-weights.h`). Upstream does the
same: the ACCEL device's buffer type comes first for CPU-side weights (`src/llama-model.cpp`, `make_cpu_buft_list`);
zDNN (ACCEL) converts weights in `set_tensor` with `is_host = false`; CANN (GPU) converts Q4_0 and Q8_0 by type and F16
matmul weights to NZ by tensor name; Hexagon has a repack buffer with `is_host = false`; the CPU backend's repack buffers
(AMX, KleidiAI, SpacemiT IME) likewise. Moving kernels into a package changes who owns the layout (the core and the
package instead of the plugin), not whether one is needed, unless the kernels read GGUF's layout. AMD reads GGUF Q4_0 in
place for generation and keeps a private F16 dequantized cache for prompt processing (V2.0 §12.1); the flow slide's
"dequantize into the workspace" may replace that cache (ask).

**F5. ACCEL upstream** (`master` @ `31288c9c6`, 2026-10-18, 1,348 commits after `ba360ef`; fetched to the local ref
`refs/upstream/master`).
- Unchanged definition: "accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)"; AMX
  is now a CPU repack buffer, not a device.
- ACCEL backends: BLAS, zDNN, ZenDNN, all claiming matmuls and views only. None runs attention or norms, so the two rules
  C5a and C5b change never affected them.
- Accelerators that run whole layers on shared memory report GPU (Hexagon) or GPU and IGPU (OpenVINO's NPU, since #29852,
  2026-10-03); SYCL, Vulkan and CUDA report IGPU for integrated devices.
- Since our base, ACCEL is treated as part of the CPU side: RPC skips ACCEL devices (#29020); the MoE expert cache works
  on GPU and IGPU only (#29887, #30112); `--fit` and `llama_supports_gpu_offload` count GPU and IGPU only.
- The flash-attention check (`resolve_fused_ops`) and norm pinning are unchanged; C5a and C5b apply cleanly (95 and 347
  lines lower). A web search found no upstream issue or PR about ACCEL backends running attention or norms (not
  exhaustive).

**F6. GPU against IGPU** (upstream `master`). They differ only in llama.cpp's default device choice when there are
several devices: every GPU is used (one physical device seen by two backends counts once, by `device_id`); IGPUs are used
only if there is no GPU, and only those of the first backend that reports one (#23897); `ggml_backend_init_best` and
mtmd's vision encoder prefer a GPU. `--device` bypasses both rules. Placement, buffers, `-ngl`, `--fit`, the
flash-attention check and norm pinning are the same. FlagOS registers before Vulkan (`ggml-backend-reg.cpp`), so as an
IGPU it would be the first.

**F7. ACCEL against IGPU on the K3.** The same placement and speed on a K3 alone (measured on ggml-spacemit; E2 measures
our provider). IGPU needs no llama.cpp change and gets the newer features (MoE cache, RPC, `--fit`, mtmd); `-ngl`
becomes meaningful (llama.cpp's default, auto, offloaded every layer in M0's X4). ACCEL keeps accelerating CPU layers next
to a discrete GPU, where IGPU is dropped; ACCEL fits ggml's wording ("together with the CPU"). `--fit` counts our reported
memory and the CPU's separately though they are the same DRAM (possible double counting; unverified).

**F8. The KV cache in the provider's buffer** (IGPU or GPU; from reading the code; the two aborts confirmed by E2-neg
on 2026-10-19, §6).
- Hardware: no obstacle; the provider's buffer is ordinary DRAM (`mmap`/`malloc`).
- Works with the default F16 cache: `SET_ROWS` (F32 to F16), `FLASH_ATTN_EXT` on views in our buffer (`test-backend-ops`
  already allocates every tensor in the backend's buffer), `clear`, state save and load (plain copies), `-nkvo`.
- Aborts with a quantized cache (`-ctk`/`-ctv q8_0`, `q4_0`, `q4_1`): the K and V tensors are `[n_embd_gqa, kv_size,
  n_stream]` with `kv_size` a multiple of 256, which the layout rule (`spacemit_weight_layout`) takes for matmul weights
  when there is one stream; and `SET_ROWS` accepts F32 and F16 only, so the scheduler aborts ("pre-allocated tensor ...
  cannot run the operation", `ggml-backend.cpp:930`).
- Aborts at a context shift (`--context-shift`, `llama_memory_seq_add`): llama.cpp rotates the cached K with `ROPE` in
  place on the F16 cache (`llama-kv-cache.cpp`, `build_rope_shift`); ours handles F32 only.
- Slow with `-fa off`: attention's `MUL_MAT` on F16 K and V runs on the X100 cores, copying K and V out of our buffer in
  every layer.
- Huge pages: the cache would be on 2 MiB pages. L1 measured 2.1-2.2x slower attention when 8 tiles stream from
  addresses 2 MiB apart (`k3-hardware.md` §2); llama.cpp's layout keeps all heads in a row and the split hands tiles
  64-position blocks 128 KiB apart (Qwen3-4B), below that spacing. M2e.3 listed the cache on huge pages as candidate (c).
  Unmeasured.
- Under ACCEL all of these run on the CPU, because the cache stays in a CPU buffer.

**F9. The K3's memory** (board, 2026-10-19). No NUMA (`/sys/devices/system/node/` absent); one device-tree memory node,
`memory@102000000`; one `System RAM` range `0x102000000-0x8ffffffff` (32 GiB less 32 MiB). Inside it: CMA 512 MiB at
`0x140000000` (`CmaTotal` 524288 kB, about 7 MiB used), the framebuffer (32 MiB at `0x3fe000000`), and about 595 MiB at
the top of RAM marked "Reserved" (most likely the kernel's page descriptors, 64 bytes per 4 KiB page = 512 MiB, plus
other boot allocations; inferred from the size). Outside it: two real-time processors (`rcpu0`, `rcpu1`) with their
rpmsg buffers, the display controller, firmware (`mmode_resv0`, `rpmi0`), and TCM at physical 0 (3 MiB,
8 x 384 KiB). Nothing is reserved for the AI cores: they have no DRAM of their own, only TCM. `/dev/hugetlb_1g` and
`/dev/tcm_sync_mem` do not exist (as `k3-hardware.md` §3 records).

**F10. Huawei Ascend (CANN) upstream.** Reports GPU on every chip (910B, 310P; PCIe cards with their own HBM). Its buffer
is `is_host = false`; `supports_buft` accepts only its own buffer; `set_tensor` converts Q4_0 and Q8_0 into its layout
(by type) and F16/F32 matmul weights to the NZ format (chosen by tensor name, `is_matmul_weight`, `aclnn_ops.h`), and
`get_alloc_size` grows the allocation for NZ; the KV cache lives in its buffer; `SET_ROWS` writes F32, F16 and BF16;
`ROPE` takes any type, so in-place rotation of the cache works. Its weight handling matches our D4; its device type
follows from its dedicated memory.

**F11. M0's GPU-against-IGPU measurement** (`device-type.md` §5, X1 and X2, 2026-10-07): ggml-spacemit with a type
switch, Qwen3-0.6B Q4_0, one session, modes interleaved. Identical text and perplexity (14.5404), pp128 594.4 / 597.3,
tg128 57.3 / 57.4, at depth 1024 293.2 / 293.6 and 35.3 / 35.2 (GPU / IGPU), 2 splits per token. Gaps: SpacemiT's
backend, not ours; IGPU on 0.6B only; depths up to 1024; three single runs without spread; IGPU's split count only in
X3's combined column.

**F12. The AMD design against our plan** (2026-10-19). AMD: GPU or IGPU, no llama.cpp change; one launch per kernel on
one non-blocking HIP stream, hipGraph capture and replay only as an opt-in experiment (`FLAGOS_AMD_GRAPH_CAPTURE` set;
generation graphs only unless it is `all`; `flagos-amd.cpp:366-373, 4808-4812`; V2.0 p. 33 "HIP Graph replay 实验路径",
p. 34 a larger replay cache among the rejected experiments; corrected 2026-10-20); GGUF weights read in place for
generation plus a private F16 dequantized cache; all kernels Triton AOT with a manifest and a strict loader; variants
chosen in the provider by name and tuning profiles; workspace owned by the
provider. Our plan follows the deck more than AMD's code; the three real differences are the device type, the weights'
storage and where kernels come from (native, compiled in).

**F13. Upstream footprint against `ba360ef`** (2026-10-19, outside `ggml-flagos` and the docs). The mentor's: 5 lines in
`ggml/CMakeLists.txt`, 1 in `ggml/src/CMakeLists.txt`, 8 in `ggml/src/ggml-backend-reg.cpp`, 31 in
`tests/test-backend-ops.cpp` (tests only). Ours: `src/llama-context.cpp`, 21 lines added and 1 changed (C5a, C5b). The
core already builds as a loadable backend (`ggml_add_backend_library`; `GGML_BACKEND_DL_IMPL` in `flagos-registry.cpp`),
and stock ggml loads one from `GGML_BACKEND_PATH` (`ggml-backend-reg.cpp:597`), so the ggml-side lines may not be needed
(E4). The provider has no link dependency on the CPU backend (its mentions of `ggml-cpu` are comments). Confirmed
2026-10-20 (E4, §6 run 2): at run time none of these lines is needed.

**F14. spine-runtime does not order launches** (its documentation on GitHub, `spacemit-com/spine-runtime` `main`,
read 2026-10-19; the board's 0.6.x header not yet checked). `Stream::launch` "launches a Tile-SPMD task and returns a
`Future`"; `Future::sync` waits for it, `Future::sync_all` for a group; Futures are copyable shared handles. No ordering
between launches on one stream is stated: "Kernels with read/write dependencies should establish an explicit execution
order through `Future::sync()`, `Future::sync_all()`, or in-tile cooperation primitives"; "independent tasks can be
submitted consecutively and then waited on together". Each core has one resident worker, and a tile waiting at a barrier,
event, `join` or shared-buffer allocation is suspended so the worker can run other ready tiles, so tiles of queued
launches may interleave. The API reference lists `Context::join`, "cooperatively waits for a Future from the same Stream".
The ABI slide states an order on a stream for copies ("在 stream 上排序") and has `stream_synchronize`; that it assumes
ordered launches too is our reading (F15). On spine-runtime, by its documentation, the plugin must order dependent
launches itself. Today's executor needs none of it: one launch per split, barriers inside (`Context::sync`).
The board's header (`~/spine-runtime/include/spert.hpp`, read 2026-10-20) has what in-core ordering needs:
`Stream::launch` returns a `Future` and copies the arguments into storage the launch owns (`:579-590`; the Future is
invalid if the launch cannot be admitted); `Future` is a shared handle with `sync()` and `sync_all(const
std::vector<Future> &)` (`:202-278`); `Context::join(Future &)` (`:469`), `spawn` (`:466`), `yield()` (`:447-448`) and
the grid barrier `sync()` (`:453`); a stream sends its launches only to the workers of its granted cores (`:506-507`,
`:522`).

**F15. E3 against the deck** (2026-10-20). E3 contradicts neither the deck's split (one core for every chip; a runtime
plugin and a kernel package from the vendor; the core choosing the kernels and their order) nor its ABI: both forms the
K3 can use are built from the deck's functions. It contradicts an assumption beneath the deck's layering (our reading:
the ABI slide states the order only for copies, "在 stream 上排序", and the core's code is in no branch we have, F1): that
the vendor's runtime has a GPU-like stream, ordered and asynchronous, so that the required functions alone (one `launch`
per kernel) reach the deck's cost of about 1% (R12) and graph capture is an optional extra.
- AMD fits the assumption as far as this tree's provider shows (the deck's `amd-rt` is not available, F1): by default
  one launch per kernel on one non-blocking HIP stream with no wait between kernels (`flagos-amd.cpp:4896` on), so the
  host queues the next kernel while the GPU runs the last (V2.0 p. 34: about 700 launches per token); its replay is an
  opt-in experiment (F12). Whether the launch costs hide behind the kernels was not measured by us, and the deck does
  not say whether its 0.6 ms of idle per token was measured with capture.
- The K3 does not. spine-runtime 0.6 promises no order between launches (its documentation, F14; dependent launches
  were never run unordered). One launch per kernel with the host waiting for each is correct and costs Qwen3-4B 11-14%
  of generation and 0-2% of prompt processing (E3a). The cost is per kernel: a one-node graph takes about 11 µs through
  our whole path (`--bench`), E3a 13.4 µs per step over the 1,086 steps E3c counted (run 3; 14.3-15.0 µs with E3c's
  counters on, of which 11.7-12.3 µs is each launch's own overhead, run 4), so a smaller model loses more (Qwen3-0.6B:
  tg128 -35.8%, run 4) and a larger one less. Passing E3's 5% line needs a launch with its wait under about 5 µs
  (at 1,086 steps: 4.1 µs more than a barrier); spine-runtime alone takes 7.67 µs, 7.2 more than a barrier (E3d, run 4),
  so no plugin passes it on Qwen3-4B with this runtime. Launches queued and ordered inside the cores are not reliably
  correct with our kernels (E3b: the 1-token layers differ from one launch in 5 of 5 runs and one run failed; E3b-diag:
  NMSE up to 2.4 against one launch, because paused tiles move between cores and two on two cores use one TCM at the
  same time, F16) and cost 7.27-316 µs per launch even with tiles that do nothing (E3d). For generation the roles swap:
  graph
  capture is the K3's normal path, the required functions alone its slow fallback.
- Point by point. A plugin with the required functions alone is complete (R3): holds (E3a, identical logits); fast: not
  for Qwen3-4B's generation on the K3 (E3a). Capture optional (R3, R5 ⑤): in the ABI yes, for that speed no; today's
  executor implements it (Q1). The core decides and schedules (R5): holds; the order between kernels comes from the
  plugin's replay (barriers inside one launch), which is runtime work. The core computes each kernel's grid (R5 ⑤):
  inside one replayed launch every kernel runs on the same 8 tiles, so the plugin spreads each kernel's grid over them
  (Q3 (a)). The manifest (R4) describes a kernel by `block_size`, `num_warps`, `contract` and `when`; nothing in it says
  a kernel needs a core per tile, its core's TCM or a partner tile, the assumptions our tiles make and E3b's queued
  launches did not keep (two tiles of one launch on one core): a gap rather than a contradiction, and moot while the
  native kernels stay in the plugin (Q6).
- For the core, on the K3: its capture policy must treat a plugin's capture as the main path, for generation graphs at
  least (without capture, prompt processing loses 0-2%, E3a), not as an opt-in like AMD's. To ask the mentor: how the
  core waits when a plugin has no events, and whether the deck's 0.6 ms was measured with capture.
- For M3 (our plan, not the deck): an AOT kernel run as a launch of its own (`plan.md` M3.1) would pay a host-waited
  launch, 13.4-15.0 µs per launch in a model run (E3a, E3c) against the 12 µs M3.2 assumes (for a token's 145 RMS
  norms about 2 ms at the measured cost, 1.7 ms at M3.2's, estimates, either over M3.2's 1% rule); AOT kernels called
  as steps inside the
  replayed launch (Q3) would not (neither is built).

**F16. With launches ordered by `join`, spine-runtime moves paused tiles, and two can use one TCM at once** (measured
2026-10-20, E3d and E3b-diag, §6 run 5; spine-runtime 0.6 on the K3, one session). With 8 or 64 launches in flight, each
tile first joining the previous launch's Future:
- Tiles that pause after their join (taking their shared buffer, a barrier) often resume on another core: 67-78% of
  tiles changed core. Tiles that only joined and did not pause again did not move.
- Two tiles of one launch ran on one core in up to 83% of launches; two tiles on two different cores held the same
  shared buffer (TCM) in 24-77%. No tile was left without TCM.
- Tiles that do nothing but write a pattern into their TCM and read it back four times, with no pause in between, found
  it overwritten: 6,207 of 160,000 tiles with 8 launches in flight, 22,958 of 159,744 with 64. Two tiles wrote one TCM
  at the same time.
- The order between launches held: no violation in about 120,000 joined launches.
- None of it inside a single launch: E3d's 20 launches of 20,000 barriers each had no moved tile and no shared core.
- Not known: whether spine-runtime expects a tile to take its shared buffer again after every pause, and why tiles move
  (a question for SpacemiT). Our provider takes TCM once per launch, before the TCM agreement's barrier.

## 4. Experiments

All on Qwen3-4B Q4_0 unless named, in one session, every mode against the reference (today's build, default flags)
with the modes interleaved in each repetition: the board drifts between sessions.

| ID | Answers | What changes | Prediction | Decision rule | Status |
|---|---|---|---|---|---|
| E2 | Q5 | Build B (upstream's `llama-context.cpp`), `FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=igpu` and `gpu`, `-ngl 99`; once with the default `-ngl` (placement only) | 2 splits per token; flash attention enabled under `-fa auto`; logits identical to the reference (KLD 0, same top p 100%; the same kernels run the same nodes); speed within 1-2%; GPU = IGPU; the default `-ngl` offloads every layer | agreed (user, 2026-10-19): "IGPU is viable with no llama.cpp change" if logits are identical at both batch settings, a token takes 2 splits, and pp128, tg128, tg32 at depths 0/1024/4096 and pp512 at depth 4096 are within 2% of the reference | done 2026-10-19 (§6): passes; pp512 at depth 4096 +3.6% |
| E2-hp | Q5 | E2's IGPU and the reference, each with and without `FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1`, depth runs | unknown; the 128 KiB block spacing suggests no 2.2x slowdown | KV cache on huge pages "hurts" if IGPU's tg32 at depth 4096 loses at least 2% more from huge pages than the reference does, "helps" if it gains at least 2% more | done 2026-10-19 (§6): neither in generation; mixed in prompt processing at depth |
| E2-neg | F8 | Build B as IGPU, and the reference: a Q8_0 KV cache (`-ctk q8_0 -ctv q8_0 -fa on -c 512`), and a context shift (`-c 256 -n 400 --ignore-eos --context-shift`) | IGPU aborts in both ("pre-allocated tensor ... cannot run the operation": `SET_ROWS`, then `ROPE`); the reference runs both (the shift logs "context full, swapping") | confirms or refutes F8's two gaps | done 2026-10-19 (§6): both confirmed |
| E1 | Q5 | Build B as ACCEL: `-fa auto` (llama.cpp's default) and `-fa on` | `-fa auto`: flash attention turned off with llama.cpp's warning, attention on the X100 cores, generation at depth far below the reference (before M2e.1: tg32 at depth 4096 1.80). `-fa on`: 288 splits per token (36 layers x 4 norms x 2), tg128 about -30% (X3b: 7.52 against 10.75), prompt processing unchanged | descriptive: what ACCEL costs without the patches | done 2026-10-19 (§6): -24 to -95%; not repeated (user, 2026-10-20: M0's X3/X3b and run 1 cover it; out of the default `EXPS`) |
| E3a | Q1 | Build A, `FLAGOS_SPACEMIT_TEST_LAUNCH=per-step`: one spine-runtime launch per step, the host waiting for each (the order spine-runtime's documentation prescribes for dependent kernels, F14): a Tier-1 plugin without in-core ordering | tg128 about -12% (about 11.5 µs more per step at about 1,060 steps per token); pp128 about -1%; logits identical (checked on the K3: 4/4 layer and 8/8 split cases byte-identical) | agreed (user, 2026-10-20), anchored to R12: Tier-1 is enough for the K3 if its cheapest correct form (E3b, as E3a fails) keeps tg128 within 1% of `ref` (whose framework cost is already about 1.1%); Tier-2 graph capture is required at a tg128 loss of 5% or more; in between, report the numbers | done 2026-10-20 (§6, run 3): tg128 -14.2%, about 14.6 ms per token; identical logits |
| E3b | Q1 | One launch per step queued without host waits, each launch's tiles first waiting inside the cores for the previous launch (`Context::join`, if the board's spine-runtime has it): one launch per kernel on an ordered stream, as we read the ABI (F15) | unknown: between the reference and E3a | as E3a | implemented 2026-10-20 (`FLAGOS_SPACEMIT_TEST_LAUNCH=per-step-join`, below); **fails its checks on the K3** (2026-10-20, 5 of 5 runs): the 1-token layer cases differ from one launch (once failed), while E3a passes everything and E3b passes the 8 split-attention and the 7-token layer cases; spine-runtime ran two tiles of one launch on one core. Not measured: a computation that is not reliably correct. Not fixed (user, 2026-10-20; Q1). Cause found by E3b-diag (run 5, F16) |
| E3c | Q1, R12 | Build A with `FLAGOS_SPACEMIT_TEST_STATS=<file>` (§5): per graph the steps and launches, the time in `graph_compute`, in the launches (host), every tile's time in every step's kernel and at the barriers; per token: outside `graph_compute` (llama.cpp, the CPU's split, sampling), host work in it, launch overhead, the barriers' own cost, imbalance, kernels. Modes `e3c` (today's launches) and `e3c-a` (E3a's) against `ref` | steps per token about 1,060 (estimated, never counted); the barriers' own cost about 1 ms per token (0.94 µs each); the counters cost tg128 under 1% (estimate: 3 clock reads per step and tile); idle (the deck's measure) unknown: the 1.1% counted the barriers only | none for a design decision: it replaces Q1's estimates (steps, 1.1%) with counts, gives R12's number, and from `e3c-a` E3a's cost per step and per launch inside a run (13.8 µs per step estimated against the benchmark's 11.5). If `e3c` loses more than 1% of tg128 against `ref`, its times are reported as disturbed | done 2026-10-20 (§6 run 4): 1,086 steps per token (1,122 from n_kv 1024); idle 1.3-1.8% by the deck's measure; E3a 14.3-15.0 µs per step in a run; the counters cost tg128 0.5% |
| E3d | Q1, F15 | `scripts/spert-launch-bench.cpp`: spine-runtime alone, no provider and no ggml, tiles that do nothing: one launch and its wait (grid 8 and 1); the same with each tile taking its TCM and one barrier (our per-launch setup); 20,000 barriers in one launch, with each tile's core at its start, middle and end; 8 and 64 launches queued, unordered; the same ordered by `Context::join`, checking the order and the cores; and again with each tile taking its TCM and a barrier (E3b's tiles); `joined+canary` added for E3b-diag (below) | launch and wait: unknown, below our whole path's 11.0-11.5 µs (`--bench`); barriers about 0.9 µs; the rest unknown | fixed before the run: (1) launch and wait (grid 8, median) at 5 µs or more: one launch per kernel with a host wait cannot pass E3's 5% line on Qwen3-4B on spine-runtime 0.6 whatever the plugin does (at about 1,060 steps; E3c counts them); below, our own per-launch path is the larger part. (2) joined, 64 at a time, at 1.7 µs or less per launch with no order violation and no shared core: in-core ordering would pass E3's 1% line with tiles that need no core of their own; reported to the user (E3b's decision is theirs) and the mentor. Any order violation: `Context::join` does not order launches, for SpacemiT. (3) barriers: any tile that changed core or shared one contradicts today's executor's assumption; stop and look before other runs. (4) joined with TCM: tiles without TCM or on the same TCM test E3b's likely mechanism (log); reported, no decision | done 2026-10-20 (§6 run 4): rule 1 triggers (7.67 µs), rule 2 fails (7.27-295.6 µs per launch), rule 3 holds, rule 4: no tile without TCM, tiles on a shared core got the same TCM |
| E3a-0.6B | F15 | E3a (`e3`) on Qwen3-0.6B Q4_0, runner group `e3a` (E3a without E3b) | tg128 about -35 to -40% (estimate: E3a's 11.5-13.8 µs per step on 4B, about 820 steps for 0.6B's 28 layers against 4B's 36, on a token of about 17.9 ms); prompt processing a few %; logits identical | descriptive: replaces F15's estimate ("a third or more") | done 2026-10-20 (§6 run 4): tg128 -35.8%, as predicted; pp128 -5.2%; logits identical |
| E2-0.6B | Q5 | E2 (`e2`) on Qwen3-0.6B Q4_0 | as on 4B: identical logits, 2 splits per token, speeds within 2% (pp512 at depth 4096 possibly faster, as on 4B) | E2's | done 2026-10-20 (§6 run 4): passes as on 4B; pp512 at depth 4096 +3.8-3.9% |
| R12-hw | `PROJECT.md` question 8, `plan.md` R12 | Build R (`EXPS=r12 ... prepare`): build A's configuration with `FLAGOS_SPACEMIT_TEST_UPSTREAM_Q4_1_M1` defined, which compiles upstream's active branch of the 1-row Q4_1 kernel (16-bit zero-point product, fp16 sums); the provider checks once, the zero-point probe (zero point 15 against 32 equal activations, 1 row and 4) | 1-row NMSE about 4.6 (the reference's emulation of a 16-bit product, `plan.md` M2c design), 4 rows unchanged (3e-7), so the check fails; random-data Q4_1 cases unknown | 1-row NMSE above the matmul bound (5e-4): upstream's branch wraps on the board, R12 becomes a tested finding for question 8; within it: the reading was wrong and R12 is withdrawn | done 2026-10-20 (§6 run 4): 1 row 4.62, 4 rows 3.0e-7: upstream's branch wraps on the board |
| E3b-diag | the cause of E3b's differences (Q1; for SpacemiT) | Two parts, both test-only. (a) E3d's `joined+canary`: tiles that only join, take their TCM, pass a barrier and check a pattern of their own in their TCM, across the barrier and with no suspension point in between, noting their core at four points. (b) `flagos-check-spacemit --e3b[=runs]`: the 1-token layers (q4_0, q4_1), the 7-token layer and eight 1-row matmuls alone (q4_0, q4_1), each `runs` times ordered in the cores against one launch, in four variants: the default 1-row strategy (`paired`), `staged` (TCM, no pair turns), `direct` (weights from DRAM, no pair turns) and `paired` with full memory fences after every join and at every tile's end (`FLAGOS_SPACEMIT_TEST_JOIN_FENCE=1`); per run, what the ordered launches' tiles noted: two tiles on one core, tiles that moved, one TCM held on two cores, both tiles of a pair on one core | unknown. Candidates: H1, two tiles use one TCM at the same time (truly at once on one core, or a tile that moved still using its first core's TCM); H2, both tiles of a pair on one core, the first spinning on the second until the 10 s limit (would explain the one failed run, not the differences); H3, a tile reading data the previous launch wrote before it is visible (memory order across `join`); H4, something in the 1-row path itself (pair turns or TCM staging) | fixed before the run. H1: (a) shows a canary changed with no suspension point, or one TCM on two cores, and (b)'s differing runs show tiles that moved or one TCM on two cores. H2: (b)'s failed runs, and only they, show a pair on one core. H3: `paired+fence` has no differing run where `paired` has many. H4: the 1-row matmuls alone differ, and `staged` or `direct` changes how often (pair turns or TCM staging). Differing runs with no shared core, moved tile or shared TCM rule co-location out. If none fits, the cause stays open and the next step is comparing every node's output | done 2026-10-20 (§6 run 5): H1 and H4 by the rules (tiles moved by `join` use one TCM at the same time; only TCM staging of each tile's own weights turns it into wrong results); H2 explains the failed runs; H3 ruled out (F16) |
| E4 | Q5, F13, R1 | Stock `ba360ef` binaries built with `GGML_BACKEND_DL=ON`, `libggml-flagos.so` from this fork loaded through `GGML_BACKEND_PATH`, with E2's IGPU | the same as E2 | E2's rule; then "no upstream change needed at all" | done 2026-10-20 (§6, run 2): passes; pp512 at depth 4096 +4.0% |
| E6 | Q2 | GGUF's bytes in the buffer ggml sees plus a private repacked copy | about +2.35 GB, the same speed, exact read-back | evidence for option (e) | not implemented |
| E5 | Q2 | the 1-row path reads GGUF's layout, interleaving 32 rows while copying into TCM | unknown; generation only | how much repacking buys | not implemented |
| Q4 table | Q4 | on paper: our kernel choices as manifest entries (startup filter for device facts, `when` for per-token shapes, one generic variant per contract), each condition `when` cannot express marked | some inexpressible (ranges, equality, strides, layout) | input for Q4 | to do |
| M3.0+ | Q2 | the M3.0 spike (`plan.md` M3) plus a Triton Q4_0 matmul reading GGUF's layout, timed against ours | unknown | input for Q2 (d) | with M3.0 |

### How to run E1, E2, E2-hp, E2-neg and E3 (on the K3, inside tmux)

```bash
git -c safe.directory=$PWD pull                                    # the switches, checks and script
docs/flagos-k3/scripts/structural-experiments.sh prepare           # build B, then build A's tools and checks
# E2 first (its rule is agreed); E3a later with E3b, in one session, once E3's rule is settled
EXPS="e2 e1 e2hp e2neg" docs/flagos-k3/scripts/structural-experiments.sh run ~/models/Qwen3-4B-Q4_0.gguf 2>&1 | tee ~/sx-run.log
docs/flagos-k3/scripts/structural-experiments.sh summary ~/sx/Qwen3-4B-Q4_0     # the tables again
```

- The board's checkout belongs to uid 501 (copied from the Mac; its `.git` belongs to root), so git refuses it
  ("dubious ownership") unless a command allows it, as `git -c safe.directory=$PWD` does; the script passes the same
  exception on every git call and changes no git settings.
- `prepare` makes build B: a worktree of the repository's HEAD at `../llama.cpp-flagos-no-c5` whose
  `src/llama-context.cpp` has C5a and C5b reverse-applied (it then equals upstream's; the script checks that `src/`
  matches `ba360ef`), configured as `build-flagos/` is and built with `llama-completion`, `llama-bench` and
  `llama-perplexity` (a full build). It then builds build A's tools and runs `flagos-check-spacemit --expect-device`,
  printing the new E2 and E3 check results. Nothing is committed; rerunning resets the worktree to the new HEAD.
- `run` measures, per mode: placement and text (`llama-completion`: graph splits for the batch and 1-token graphs,
  flash attention under `-fa auto`, offloaded layers, where the KV cache is, the reported type), llama-bench (pp128 and
  tg128 with `-ub 128`; with `DEPTH=1` tg32 at depths 0, 1024 and 4096 and pp512 at depth 4096; `-r 1` per repetition,
  `REPS` repetitions interleaved), perplexity on `~/ppl.txt` (8 x 512 tokens) with the reference's logits as the KLD
  base, the KLD at one token per ubatch (`-ub 1 -b 512`) for E2's IGPU, and E2-neg's four runs. `tcmrelease --apply`
  before every run; 8 threads.
- Modes: `ref` (A), `e1-auto` and `e1-on` (B, ACCEL), `e2-igpu`, `e2-gpu`, `e2-igpu-auto` (B), `e3` (A), `ref-nohp`
  and `e2-igpu-nohp` (E2-hp), `neg-{q8,shift}-{igpu,ref}` (E2-neg); the script's header lists their settings.
- Options: `EXPS` (default `e2 e1 e3 e2hp e2neg`), `REPS` (3), `DEPTH` (1), `PPL_TEXT`, `CHUNKS`, `KLD_UB1`, `TIMEOUT`,
  `BUILD_A`, `WT_B`, `BUILD_B`, `DRY=1` (print the commands). A first check of the wiring:
  `REPS=1 DEPTH=0 PPL_TEXT= EXPS="e2 e3" ...run ...` (minutes).
- Time with the defaults: about 2-3 hours (estimate: the depth runs process 4096 tokens before each test, and `e1-auto`
  runs attention on the X100 cores).

### Run 4 (prepared 2026-10-20): E3d, R12-hw, E3c, then E3a and E2 on Qwen3-0.6B (on the K3, inside tmux)

```bash
git -c safe.directory=$PWD pull
# E3d, spine-runtime alone (seconds; nothing else on the AI cores)
g++ -O2 -std=c++17 -I$HOME/spine-runtime/include docs/flagos-k3/scripts/spert-launch-bench.cpp \
    -L$HOME/spine-runtime/lib -lspert -o ~/spert-launch-bench
~/tcmtest/tcmrelease --apply && LD_LIBRARY_PATH=$HOME/spine-runtime/lib ~/spert-launch-bench 2>&1 | tee ~/sx-e3d.log
# build A with the counters and its checks; build R and its checks (R12-hw)
EXPS="e3c r12" docs/flagos-k3/scripts/structural-experiments.sh prepare 2>&1 | tee ~/sx-prepare-r4.log
# E3c on Qwen3-4B: ref, e3c, e3c-a; no perplexity (the counters change no output)
EXPS="e3c" docs/flagos-k3/scripts/structural-experiments.sh run ~/models/Qwen3-4B-Q4_0.gguf ~/sx/e3c 2>&1 | tee ~/sx-e3c.log
# E3a and E2 on Qwen3-0.6B (build B again, from the new HEAD)
EXPS="e2 e3a" docs/flagos-k3/scripts/structural-experiments.sh prepare 2>&1 | tee ~/sx-prepare-06b.log
EXPS="e2 e3a" docs/flagos-k3/scripts/structural-experiments.sh run ~/models/Qwen3-0.6B-Q4_0.gguf ~/sx/0.6b 2>&1 | tee ~/sx-06b.log
```

- E3d prints one line per case; if a `joined+tcm` line does not come within a minute, the run hangs there (itself a
  result): Ctrl-C, then `~/tcmtest/tcmrelease --apply`.
- R12-hw's result is the `prepare` lines under "build R": the check's last line and the probe's NMSE for 1 row and 4.
- E3c's summary adds a table per generated token (µs): outside `graph_compute`, host work in it, launch overhead, the
  barriers' own cost, imbalance, kernels, idle (the deck's measure) and its share, steps and launches per token, and
  from `e3c-a` E3a's cost per step and per launch.
- Time (estimates): E3c about an hour (3 modes, 3 repetitions, depth runs); the 0.6B session about 40 minutes; build R
  a full build of the checks.

### Run 5 (prepared 2026-10-20): E3b-diag (on the K3 once run 4's runs have ended, inside tmux)

```bash
# the commits (bundle or pull), then:
cmake --build build-flagos --target flagos-check-spacemit 2>&1 | tail -1
g++ -O2 -std=c++17 -I$HOME/spine-runtime/include docs/flagos-k3/scripts/spert-launch-bench.cpp \
    -L$HOME/spine-runtime/lib -lspert -o ~/spert-launch-bench
~/tcmtest/tcmrelease --apply && LD_LIBRARY_PATH=$HOME/spine-runtime/lib ~/spert-launch-bench 2>&1 | tee ~/sx-e3d-2.log
~/tcmtest/tcmrelease --apply && ./build-flagos/bin/flagos-check-spacemit --expect-device --e3b 2>&1 | tee ~/sx-e3b-diag.log
```

- `--e3b` prints one `e3b-diag` line per case and variant (20 runs each by default, `--e3b=50` for more); it fails
  nothing. A few minutes, longer if pair waits run into their 10 s limit (each such run takes 10 s).
- The benchmark repeats run 4's cases and adds `joined+canary` at the end.

### E4: stock upstream llama.cpp with the provider as a run-time module (scripted 2026-10-20)

**Why:** E2 showed that as an IGPU the K3 needs no llama.cpp change. What remains of the fork's footprint is the mentor's
ggml lines (F13): 5 CMake options, `ggml_add_backend(FLAGOS)`, and in `ggml-backend-reg.cpp` the compiled-in
registration plus one line in the run-time loader's name list (`ggml_backend_load_best("flagos", ...)`). With
`GGML_BACKEND_DL=ON`, ggml builds every backend as a separate module and loads them at start (`ggml_backend_load_all`:
known names, then the one file `GGML_BACKEND_PATH` names); the compiled-in registration does not apply, and
`GGML_BACKEND_PATH` loads a module without any name. So stock upstream should run the provider, built elsewhere, with no
line changed. If it does at E2's speed, R1 is met with no upstream change at all; it is also the deck's model ("ggml
backend 接口，llama.cpp 不用改").

**Builds** (`EXPS=e4 ... prepare`):
- S: a worktree of upstream `ba360ef` at `<repo>-stock`, built in `build-dl` with `-DGGML_BACKEND_DL=ON
  -DBUILD_SHARED_LIBS=ON` and upstream's defaults otherwise (its SpacemiT CPU path stays off, as in A and B); the
  programs pull in every backend module (the CPU's), as ggml makes them depend on `ggml`.
- M: this repository built in `build-flagos-dl` the same way with the FlagOS options, target `ggml-flagos`: the module
  `build-flagos-dl/bin/libggml-flagos.so` (modules go to the programs' directory, `ggml_add_backend_library`).
- A: as before (the reference and the provider checks).
- Smoke check: the `libggml-base` names (soname) of S and M and the module's needs, which must agree so that the dynamic
  linker reuses the stock library (one ggml core in the process); then the stock `llama-completion --list-devices` with
  `GGML_BACKEND_PATH` and `FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=igpu`, which must list `FlagOS:SpacemiT:0` and report IGPU.

**Mode** `e4-igpu`: S's programs, `GGML_BACKEND_PATH=<M's module>`, `FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=igpu`, `-ngl 99`;
the same measurements as E2 (placement, pp128/tg128, the depth runs, perplexity and KLD against `ref`, also at one token
per ubatch). Rule: E2's.

```bash
EXPS="e4" docs/flagos-k3/scripts/structural-experiments.sh prepare 2>&1 | tee ~/sx-prepare-e4.log
EXPS="e4" docs/flagos-k3/scripts/structural-experiments.sh run ~/models/Qwen3-4B-Q4_0.gguf ~/sx/e4 2>&1 | tee ~/sx-e4.log
```

`EXPS="e2 e4"` adds `e2-igpu` (build B) to tell effects of loading the provider as a module from effects of the device
type. Expected cost: a full build of S, then about 1-1.5 hours of runs (estimate).

**Mac, 2026-10-20** (`build-host-dl`, the simulated device, backends as modules): the module builds and links needing only
`libggml-base` (`otool -L`; macOS refuses a module with unresolved symbols); this repository's own loader finds it by
name; moved out of `bin`, it is no longer found, and `GGML_BACKEND_PATH` alone loads it, reporting IGPU under the
switch. Not tried on the Mac: stock upstream's programs (the K3's smoke check does it).

## 5. Code for the experiments

Test-only; the defaults are unchanged.

| Switch | Where | What |
|---|---|---|
| `FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=gpu\|igpu` | `spacemit_device_type`, `flagos-spacemit.cpp` | reports that type instead of ACCEL; read once per process; logs one warning naming the type (the script reads it); anything else is reported and ignored |
| `FLAGOS_SPACEMIT_TEST_LAUNCH=per-step` | `spacemit_launch_mode` and `spacemit_backend_graph_compute`, `flagos-spacemit.cpp` | E3a: runs the executor once per step instead of once per split; the same steps, order and workspace, so the same outputs; read at every graph; `per-split` names the default |
| `FLAGOS_SPACEMIT_TEST_LAUNCH=per-step-join` | the executors' `run_ordered` (`flagos-spacemit-exec.h`, `flagos-spacemit-exec.cpp`) | E3b: every step its own spine-runtime launch, queued without host waits; each tile first joins its own copy of the previous launch's `Future` (`Context::join`), before it takes its core's TCM; the host waits once (`Future::sync_all`) and reports, once, launches whose tiles shared a core (`sched_getcpu`), where the TCM agreement or a pair wait could deadlock. Without spine-runtime (the Mac) the host waits for each launch, as E3a |
| `FLAGOS_SPACEMIT_TEST_STATS=<file>` | `spacemit_stats_file` and `spacemit_backend_graph_compute` (`flagos-spacemit.cpp`), `spacemit_run_stats` (`flagos-spacemit-exec.h`), the executors' `run` | E3c: per kind of graph (launch mode, the first matmul's rows, the first attention's KV length class: below 1024, below 4096, above), the graphs, steps and launches, the time in `graph_compute`, in the launches (host), in the kernels (summed over the tiles, and per step the slowest tile) and at the barriers; one JSON line per kind appended to `<file>` when the process ends. Read once; the tiles read the clock only with the switch set and write their times into their own rows. E3b's launches are not counted |
| `FLAGOS_SPACEMIT_TEST_UPSTREAM_Q4_1_M1` (compile-time) | `gemm_kernel_i8i4_m1`, `flagos-spacemit-ime-kernels.cpp` | R12-hw: compiles upstream's active branch of the 1-row Q4_1 kernel instead of the exact one; only in build R, through `CMAKE_CXX_FLAGS` |
| `FLAGOS_SPACEMIT_TEST_JOIN_FENCE=1` | `run_ordered` and `spacemit_tile_main_ordered`, `flagos-spacemit-exec.cpp` | E3b-diag: with E3b, a full memory fence after every tile's join and at its end. E3b's launches also always note each tile's core after its join, after taking its TCM, after the TCM agreement and at its end, and its TCM; `spacemit_ordered_counts_take()` (`flagos-spacemit-exec.h`) returns what they showed |
| `flagos-check-spacemit --e3b[=runs]` | `diagnose_e3b`, `tests/check_spacemit.cpp` | E3b-diag: only the E3b diagnosis (§4), nothing else checked |

The kernels are the same in every mode: the switches change only what llama.cpp is told (E2) or how the steps are
grouped into launches (E3).

Checks in `flagos-check-spacemit` (`tests/check_spacemit.cpp`):
- the device type follows `FLAGOS_SPACEMIT_TEST_DEVICE_TYPE` (`expected_device_type`), and the device line prints the
  type reported;
- the Qwen3-like layer with the KV cache in the provider's buffer (E2: `layer_nmse(..., kv_in_provider = true)`), within
  the layer's NMSE bound against the CPU;
- one launch per step against one per layer, byte for byte, for Q4_0 and Q4_1 down projections at 1 and 7 tokens
  (quantization and matmul, the norms, RoPE, the cache writes and attention each launched on their own);
- the split attention's two steps (partial results, then their merge through the workspace) launched separately, byte for
  byte against one launch, on its 8 one-token cases;
- the forced-failure test sets its launch mode: per split (failure in step 0 and step 1) and per step (step 0).

`scripts/structural-experiments.sh`: `prepare`, `run`, `summary` (§4). `scripts/spert-launch-bench.cpp`: E3d, built
and run by hand (its header, and run 4 in §4).

Verified on the Mac, 2026-10-19 (`build-host`, the simulated device, reference kernels): `flagos-check-spacemit
--expect-device` passes with the defaults, with `FLAGOS_SPACEMIT_TEST_DEVICE_TYPE` set to `igpu`, `gpu` and an unknown
value (reported type IGPU, GPU, ACCEL), and with `FLAGOS_SPACEMIT_TEST_LAUNCH=per-step` with tiles one after another and
on threads (`FLAGOS_SPACEMIT_TEST_THREADS=1`); the layer with the KV cache in our buffer has the same max NMSE as with it
in a CPU buffer (5.40e-04); per step identical in 4/4 layer cases and 8/8 split cases; `test-backend-ops` with one launch
per step: `MUL_MAT` 3/3, `FLASH_ATTN_EXT` 1324/1324, `ROPE_SET_ROWS` 24/24, `RMS_NORM_MUL_ADD` 30/30;
`llama-completion --list-devices` lists the device with the IGPU warning. The script: `bash -n`, a dry run of every
command, and its summary on sample logs (bash 3.2); `prepare`'s git steps (worktree, C5a and C5b reverse-applied,
`src/` equal to `ba360ef`, and a second run on the existing worktree) under `GIT_TEST_ASSUME_DIFFERENT_OWNER=1`, git's
own simulation of the board's ownership check, with `cmake` stubbed. Not yet on the K3.

## 6. Results

### Run 1: E1, E2, E2-hp, E2-neg (K3, 2026-10-19)

Qwen3-4B Q4_0; builds A and B from `0b201a6` (later commits changed only the runner and the docs);
`EXPS="e2 e1 e2hp e2neg"`, 3 repetitions, depth runs, perplexity on 8 x 512 tokens of `~/ppl.txt`, the one-token KLD
for `e2-igpu`; logs in `~/sx/Qwen3-4B-Q4_0`. Every run ended with exit code 0 except the two IGPU E2-neg runs (134,
aborted as predicted).

Placement (`llama-completion`; its default context is the model's 40,960 tokens):

| Mode | Splits (batch / 1 token) | Flash attention under `-fa auto` | Offloaded | KV cache |
|---|---|---|---|---|
| `ref` | 2 / 2 | enabled | - | CPU, 5760 MiB |
| `e2-igpu`, `e2-gpu`, `e2-igpu-auto` | 2 / 2 | enabled | 37/37 | provider, 5760 MiB |
| `e1-auto` | 360 / 360 | disabled | - | CPU |
| `e1-on` | 288 / 288 | set on | - | CPU |

The same text in every mode.

Speed (t/s, mean ± sd over 3 repetitions; change against `ref`):

| Mode | pp128 | tg128 | tg32 depth 0 | tg32 depth 1024 | tg32 depth 4096 | pp512 depth 4096 |
|---|---|---|---|---|---|---|
| `ref` | 92.44 ± 0.24 | 11.28 ± 0.04 | 11.37 ± 0.03 | 9.85 ± 0.04 | 7.84 ± 0.02 | 34.63 ± 0.12 |
| `e2-igpu` | 92.09 ± 0.14 (-0.4%) | 11.27 ± 0.00 (-0.1%) | 11.36 ± 0.04 (-0.0%) | 9.85 ± 0.03 (-0.0%) | 7.81 ± 0.03 (-0.3%) | 35.88 ± 0.11 (+3.6%) |
| `e2-gpu` | 92.13 ± 0.02 (-0.3%) | 11.27 ± 0.03 (-0.2%) | 11.40 ± 0.02 (+0.3%) | 9.82 ± 0.01 (-0.3%) | 7.82 ± 0.02 (-0.3%) | 35.91 ± 0.05 (+3.7%) |
| `e1-auto` | 64.57 ± 4.78 (-30.2%) | 3.99 ± 0.10 (-64.6%) | 4.04 ± 0.15 (-64.5%) | 1.38 ± 0.01 (-86.0%) | 0.36 ± 0.00 (-95.4%) | 16.72 ± 0.10 (-51.7%) |
| `e1-on` | 84.83 ± 0.74 (-8.2%) | 7.94 ± 0.29 (-29.6%) | 8.20 ± 0.34 (-27.8%) | 7.12 ± 0.34 (-27.7%) | 5.93 ± 0.24 (-24.3%) | 34.34 ± 0.07 (-0.8%) |
| `ref-nohp` | - | - | 11.03 ± 0.01 (-2.9%) | 9.61 ± 0.03 (-2.4%) | 7.67 ± 0.02 (-2.1%) | 35.70 ± 0.03 (+3.1%) |
| `e2-igpu-nohp` | - | - | 11.04 ± 0.02 (-2.9%) | 9.61 ± 0.00 (-2.4%) | 7.65 ± 0.02 (-2.3%) | 35.71 ± 0.01 (+3.1%) |

Accuracy (KLD against `ref`'s logits, which llama-perplexity stores as 16-bit log-probabilities,
`tools/perplexity/perplexity.cpp:86-104`, so identical logits give a mean of 0 and a maximum of about 5e-5):

| Run | Perplexity | Mean KLD | Max KLD | Same top token |
|---|---|---|---|---|
| `ref`, 512-token batches | 9.7820 | (base) | | |
| `e2-igpu` | 9.782019 | -0.000000 | 0.000053 | 100% |
| `e2-gpu` | 9.782019 | -0.000000 | 0.000053 | 100% |
| `ref`, one token per ubatch | 9.7566 | (base) | | |
| `e2-igpu`, one token per ubatch | 9.756647 | 0.000000 | 0.000055 | 100% |
| `e1-auto` | 9.767601 | 0.001285 | 0.072804 | 98.28% |
| `e1-on` | 9.774899 | 0.001339 | 0.124172 | 98.53% |

E2-neg:

| Run | Exit code | Result |
|---|---|---|
| `neg-q8-igpu` | 134 | aborted: "pre-allocated tensor (cache_k_l0 (view)) in a buffer (FlagOS:SpacemiT) that cannot run the operation (SET_ROWS)" |
| `neg-q8-ref` | 0 | ran |
| `neg-shift-igpu` | 134 | aborted at the shift: "pre-allocated tensor (cache_k_l0 (view) (view)) ... (R" (cut off in the summary; `ROPE` by the code path) |
| `neg-shift-ref` | 0 | 399 tokens generated in a 256-token context: shifted and went on (its "context full" line is a debug message, shown only from `-lv 5`) |

Reading:
- **E2, by the agreed rule:** logits identical at both batch settings (perplexity to six digits, the same top token
  every time, mean KLD 0); a token takes 2 splits; five of the six speeds within 0.4% of `ref`. pp512 at depth 4096 is
  3.6% faster as IGPU and 3.7% as GPU: outside "within 2%" only in the favorable direction, a pass if the rule means no
  loss beyond 2% (the user's reading pending). GPU and IGPU are the same. So the K3 runs Qwen3-4B Q4_0 at today's speed
  and accuracy as IGPU or GPU with no change to llama.cpp (other models not run).
- **E1:** stock llama.cpp with an ACCEL device needs C5a and C5b. With `-fa auto` flash attention goes off and attention
  runs on the X100 cores (360 splits per graph): generation at depth 4096 falls to 5% of `ref`. With `-fa on` the norms
  are pinned to the CPU (288 splits): generation -24 to -30%, as predicted (-30%, X3b). The pinning hits prompt
  processing too (pp128 -8%): llama.cpp counts its default `-ngl` (auto) as full offload, a negative setting meaning
  every layer plus the output (`src/llama-model.cpp:1732-1735`), and full offload pins norms at every batch size
  (`src/llama-context.cpp:2518-2519`). X3b's runs in M0 did not see this.
- **E2-hp:** the weights on huge pages make generation 2-3% faster (`ref` against `ref-nohp`, and `e2-igpu` against
  `e2-igpu-nohp`), more than the 1% measured on 2026-10-11. The KV cache on huge pages changes nothing in generation, at
  depth 4096 either (7.81 against 7.84; without huge pages 7.65 against 7.67): L1's 2.2x slowdown does not appear in
  llama.cpp's cache layout. Prompt processing at depth 4096: `ref`, the one arrangement with the weights on huge pages and
  the cache on 4 KiB pages, is the slowest; huge pages off everywhere +3.1%, on everywhere (IGPU) +3.6%. Two effects of
  opposite sign, not explained: candidates are address collisions of the batch matmuls' weights on huge pages and fewer
  TLB misses when the tiled kernel reads the cache; not measured.
- **E2-neg:** F8's two gaps are real: a quantized KV cache and a context shift both abort as IGPU, and both run as
  ACCEL.

**E2 in full** (from the run's `info.txt`, `bench.jsonl`, `load-*.err` and `kld-*.out`):
- Started 2026-10-19 07:52 UTC. `ref`: build A, `~/llama.cpp-flagos/build-flagos`, its binaries built by `prepare` from
  `0b201a6` (the repository was at `72d741f` when the run started; the commits between changed only the runner and this
  file); its llama-bench reports `build_commit` "unknown" because git refuses the board's checkout when cmake asks for
  the commit. `e2-igpu`, `e2-gpu`: build B, `~/llama.cpp-flagos-no-c5/build-flagos` at `0b201a6`, whose
  `src/llama-context.cpp` has C5a and C5b removed (1 line restored, 21 removed).
- llama-bench as it recorded itself: 8 threads, batch 2048, ubatch 128 in the pp128/tg128 run (its default 512 in the
  depth runs), flash attention auto, K and V cache F16, devices auto; `-ngl 99` for the E2 modes (`ref`: -1, auto).
- Memory (`llama-completion`, context 40,960 tokens), the same in every mode except the KV cache: model 2349.87 MiB in
  the provider's buffer and 304.28 MiB mapped on the CPU (the token embeddings); KV cache 5760 MiB in a CPU buffer
  (`ref`) or in the provider's (E2); compute buffers 301.75 MiB (provider) and 50.01 MiB (CPU); 2 graph splits; flash
  attention enabled; E2 offloads 37 of 37 layers.

Every repetition (t/s; repetitions 1, 2, 3):

| Test | `ref` | `e2-igpu` | `e2-gpu` |
|---|---|---|---|
| pp128 | 92.292, 92.720, 92.313 | 91.930, 92.174, 92.166 | 92.151, 92.105, 92.129 |
| tg128 | 11.258, 11.327, 11.270 | 11.268, 11.274, 11.277 | 11.292, 11.279, 11.231 |
| tg32 depth 0 | 11.395, 11.362, 11.342 | 11.355, 11.403, 11.334 | 11.403, 11.420, 11.385 |
| tg32 depth 1024 | 9.864, 9.806, 9.889 | 9.815, 9.852, 9.879 | 9.834, 9.829, 9.808 |
| tg32 depth 4096 | 7.817, 7.853, 7.840 | 7.783, 7.836, 7.815 | 7.816, 7.835, 7.800 |
| pp512 depth 4096 | 34.503, 34.748, 34.644 | 36.001, 35.782, 35.851 | 35.886, 35.969, 35.874 |

Only pp512 at depth 4096 separates the modes: every E2 repetition is above every `ref` one. In the other tests the
ranges overlap or touch; pp128's E2 values sit 0.3-0.4% below `ref`'s, whose second repetition (92.720) is its highest.

Accuracy in full:

| | `ref`'s perplexity | E2's perplexity | Mean KLD | Median | 99% | Max | Same top token |
|---|---|---|---|---|---|---|---|
| 512-token batches (`e2-igpu`, `e2-gpu` alike) | 9.7820 ± 0.70560 | 9.782019 ± 0.705601 | -0.000000 ± 0.000000 | 0.000000 | 0.000035 | 0.000053 | 100.000% |
| one token per ubatch (`e2-igpu`) | 9.7566 ± 0.70302 | 9.756647 ± 0.703021 | 0.000000 ± 0.000000 | 0.000000 | 0.000039 | 0.000055 | 100.000% |

llama-perplexity also prints "PPL(base)" 9.693236 (one token per ubatch: 9.670406) and a ratio of 1.009159 (1.008918).
That base is recomputed from the stored file, which keeps each token's log-probability only within 16 nats of the most
likely one and stores anything less likely at that floor (`tools/perplexity/perplexity.cpp:86`, `:213`), so it comes out
optimistic. `ref`'s own perplexity, from its full logits, is the comparison: equal to E2's.

### Run 2: E4 (K3, 2026-10-20)

Qwen3-4B Q4_0. `ref`: build A, ACCEL (provider code as `0b201a6`; later commits changed only the runner and the docs).
`e4-igpu`: upstream `ba360ef` with no FlagOS line, built with backends as run-time modules (build S), running the
provider module from build M (this repository at `53ebb49`, the same provider code) loaded through `GGML_BACKEND_PATH`,
with `FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=igpu` and `-ngl 99`. `EXPS="e4"`, 3 repetitions, depth runs, perplexity on 8 x 512
tokens of `~/ppl.txt` and at one token per ubatch; logs in `~/sx/e4`. Every run ended with exit code 0. Smoke check in
`prepare`: both builds name their core `libggml-base.so.0`, the module needs that name and `libspert.so.1`, and the stock
`llama-completion --list-devices` lists `FlagOS:SpacemiT:0: SpacemiT K3 A100 AI cores (8)` as IGPU.

| Mode | Splits (batch / 1 token) | Flash attention under `-fa auto` | Offloaded | KV cache |
|---|---|---|---|---|
| `ref` | 2 / 2 | enabled | - | CPU, 5760 MiB |
| `e4-igpu` | 2 / 2 | enabled | 37/37 | provider, 5760 MiB |

The same text in both.

| Mode | pp128 | tg128 | tg32 depth 0 | tg32 depth 1024 | tg32 depth 4096 | pp512 depth 4096 |
|---|---|---|---|---|---|---|
| `ref` | 92.44 ± 0.17 | 11.28 ± 0.01 | 11.38 ± 0.05 | 9.89 ± 0.03 | 7.84 ± 0.02 | 34.52 ± 0.03 |
| `e4-igpu` | 92.19 ± 0.19 (-0.3%) | 11.30 ± 0.01 (+0.2%) | 11.37 ± 0.05 (-0.1%) | 9.88 ± 0.03 (-0.2%) | 7.84 ± 0.01 (+0.0%) | 35.89 ± 0.12 (+4.0%) |

| | `ref`'s perplexity | `e4-igpu`'s perplexity | Mean KLD | Max KLD | Same top token |
|---|---|---|---|---|---|
| 512-token batches | 9.7820 | 9.782019 | -0.000000 | 0.000053 | 100% |
| one token per ubatch | 9.7566 | 9.756647 | 0.000000 | 0.000055 | 100% |

Reading: E4 passes E2's rule as E2 did: identical logits at both batch settings, 2 splits per token, five speeds within
0.3% of `ref` and pp512 at depth 4096 4.0% faster (E2: +3.6%), the KV cache on huge pages again. So with the provider
reporting IGPU, the K3 runs Qwen3-4B Q4_0 at today's speed and with identical results with no change at all to upstream
llama.cpp or ggml: neither C5a and C5b nor the mentor's ggml lines are needed at run time. What this does not settle:
- F8's two gaps hold here too (the KV cache is in the provider's buffer), so IGPU needs that provider fix before it can
  be the default.
- The type came from a test switch; making IGPU the provider's type is D3's question for the mentor.
- Delivery: the module still builds inside this repository (the mentor's CMake lines are needed to build it, not to run
  it), it must match the host's ggml version (`ba360ef` here), and stock programs find it only through
  `GGML_BACKEND_PATH`; finding it by name needs the one loader line (`ggml_backend_load_best("flagos", ...)`).

### Run 3: E3a (K3, 2026-10-20)

Qwen3-4B Q4_0, build A for both modes (provider code as `0b201a6`): `ref`, and `e3` with
`FLAGOS_SPACEMIT_TEST_LAUNCH=per-step` (one spine-runtime launch per step, the host waiting for each). `EXPS="e3"`, 3
repetitions, depth runs, perplexity on 8 x 512 tokens; logs in `~/sx/e3a`. Every run ended with exit code 0. Placement
the same in both (2 / 2 splits, flash attention enabled, ACCEL, KV cache in a CPU buffer); the same text.

| Mode | pp128 | tg128 | tg32 depth 0 | tg32 depth 1024 | tg32 depth 4096 | pp512 depth 4096 |
|---|---|---|---|---|---|---|
| `ref` | 92.47 ± 0.32 | 11.31 ± 0.01 | 11.41 ± 0.04 | 9.87 ± 0.03 | 7.85 ± 0.00 | 34.54 ± 0.05 |
| `e3` (E3a) | 90.93 ± 0.09 (-1.7%) | 9.71 ± 0.01 (-14.2%) | 9.78 ± 0.02 (-14.3%) | 8.62 ± 0.01 (-12.7%) | 6.99 ± 0.01 (-11.0%) | 34.47 ± 0.05 (-0.2%) |
| extra time | 23 ms per 128-token batch | 14.6 ms per token | 14.6 ms per token | 14.7 ms per token | 15.7 ms per token | about 30 ms per 512-token batch (within the noise) |

Accuracy: perplexity 9.782019 against `ref`'s 9.7820, mean KLD -0.000000, maximum 0.000053 (the base file's rounding),
the same top token 100%: identical logits, as the checks showed (byte-identical outputs).

Reading:
- A plugin with the required functions alone, ordering the kernels by waiting on the host, costs Qwen3-4B Q4_0 on the
  K3 11-14% of generation: about 14.6 ms more per token at depths 0-1024 (15.7 ms at 4096), about 12 times today's
  estimated 1.1% (and the deck's 1.2% for AMD). Prompt processing loses little (-1.7% at 128 tokens, nothing measurable
  at 512), most likely because each step carries more work (not measured).
  By the proposed rule (Tier-2 needed at 5% or more), E3a fails clearly, so the question moves to E3b.
- The cost per step is higher than predicted: 14.6 ms per token over the estimated 1,060 steps is about 13.8 µs per step,
  against about 11.5 µs from the launch and barrier benchmarks (predicted -12%, measured -14%). Either a launch costs
  more inside a model run than in the benchmark, or a token has more steps than estimated; the step count was never
  counted. E3c counts it.
- Correctness and placement do not change with the launch model (expected: the same steps in the same order).

### Run 4 (K3, 2026-10-20): E3d, R12-hw, E3c, and E3a and E2 on Qwen3-0.6B

Commits up to `060c96d1f` (by `git bundle`, the board's name lookup failing). Every program ended without a runtime
failure.

E3d, spine-runtime 0.6 alone (8 cores, VLEN 1024, shared buffer 384 KiB, arch 0xa064), tiles that do (almost)
nothing; medians over 20,000 launches per case, one session:

| Case | Result |
|---|---|
| launch and wait, 8 tiles / 1 tile | 7.67 µs (p10 7.46, p90 7.96) / 4.50 µs |
| launch, TCM and one barrier, and wait (our per-launch setup) | 8.79 µs |
| barriers inside one launch (20,000 x 20 launches) | 0.435 µs each; no tile changed core (0 of 160), no launch with two tiles on one core (0 of 20) |
| queued, unordered, 8 / 64 at a time | 5.86 / 5.36 µs per launch; the host's launch calls alone 5.49 / 5.28 µs; no shared core |
| ordered by `join`, 8 / 64 at a time | 7.27 / 295.64 µs per launch; no order violation; two tiles on one core in 44 of 20,000 / 11,345 of 19,968 launches |
| ordered by `join`, with TCM and a barrier, 8 / 64 at a time | 26.37 / 420.46 µs per launch; no order violation; two tiles on one core in 5,210 / 14,716 launches; no tile without TCM; two tiles on the same TCM in 5,210 / 14,716 launches |

Tiles ran on cores 8-15 only. By the rules fixed in §4:
- Rule 1 triggers: a launch and its wait cost 7.2 µs more than a barrier, so one launch per kernel with a host wait
  stays under E3's 5% line on Qwen3-4B (88.5 ms per token) only below about 610 steps per token (estimated 1,060;
  E3c counts them).
- Rule 2 fails: in-core ordering by `join` costs 7.27 µs per launch at 8 in flight and 295.64 µs at 64, against the
  1.7 µs it needed. It kept the order in all of about 80,000 launches.
- Rule 3 holds: within one launch, no tile moved and none shared a core, in 20 launches of 20,000 barriers.
- Rule 4: no tile was left without TCM, which contradicts the first half of E3b's likely mechanism in the log. Tiles
  that shared a core got the same TCM, in every such launch. Two tiles on one core came only with `join`, and also
  with tiles that do nothing: the runtime's doing, not our kernels'.
- Also measured: a bare barrier costs 0.435 µs, half the 0.89-0.94 µs per node of the provider's `--bench` (which
  includes a node's kernel and dispatch), so Q1's "about 1 ms of barriers per token" is likely about twice too high
  (estimate; E3c measures the barriers' own cost in a run). The host's launch call alone costs 5.3-5.5 µs, so queuing
  launches without waits still costs the host about that per launch.

R12-hw (build R): the probe's 1-row result has NMSE 4.62, the 4-row one 3.0e-7, so the check fails as predicted; its
random-data Q4_1 cases pass with upstream's branch. For this input every block's value is -30,480 and a 16-bit wrap
turns the zero-point term 60,960 into -4,576, so the kernel computes 35,056: off by 65,536 = 2^16 in every block, an NMSE
of (65,536 / 30,480)^2 = 4.62, the board's figure. By the rule: upstream's active branch of the 1-row Q4_1 kernel wraps
on the K3 (in the provider's copy, its body unchanged); R12 is a tested finding for question 8. Build A passes the probe.
How often real models produce such blocks is not measured.

E3c (Qwen3-4B Q4_0, build A, `ref`, `e3c` and `e3c-a` interleaved, 3 repetitions, depth runs; logs in `~/sx/e3c`).
Speed (t/s, change against `ref`):

| Mode | pp128 | tg128 | tg32 depth 0 | tg32 depth 1024 | tg32 depth 4096 | pp512 depth 4096 |
|---|---|---|---|---|---|---|
| `ref` | 92.43 ± 0.25 | 11.25 ± 0.01 | 11.39 ± 0.02 | 9.87 ± 0.04 | 7.81 ± 0.02 | 34.59 ± 0.12 |
| `e3c` (counters) | 92.01 (-0.4%) | 11.20 (-0.5%) | 11.27 (-1.1%) | 9.83 (-0.4%) | 7.79 (-0.3%) | 34.61 (+0.1%) |
| `e3c-a` (counters, one launch per step) | 90.88 (-1.7%) | 9.54 (-15.3%) | 9.57 (-16.0%) | 8.43 (-14.6%) | 6.90 (-11.8%) | 34.51 (-0.2%) |

One generated token with today's launches (`e3c`), µs, mean of the repetitions (the summary's columns: outside
`graph_compute`; host work in it outside the launch; launch, the launch less the mean tile's kernels and barrier waits;
the barriers' own cost; imbalance, each step's slowest tile less the mean tile, summed; kernels, the mean tile's; idle,
the deck's measure, the token less the busy time):

| Test | Token | Outside | Host | Launch | Barriers | Imbalance | Kernels | Idle | Steps |
|---|---|---|---|---|---|---|---|---|---|
| tg128 | 89,285 | 273 | 546 | 190 | 604 | 2,469 | 85,202 | 1,614 (1.81%) | 1,086 |
| tg32 depth 0 | 88,768 | -264 | 643 | 203 | 587 | 2,479 | 85,120 | 1,169 (1.32%) | 1,086 |
| tg32 depth 1024 | 101,758 | 369 | 548 | 190 | 620 | 3,711 | 96,321 | 1,726 (1.70%) | 1,122 |
| tg32 depth 4096 | 128,356 | 390 | 547 | 193 | 623 | 3,508 | 123,095 | 1,753 (1.37%) | 1,122 |

With one launch per step (`e3c-a`): idle 11.5-14.8% of a token, each launch's own overhead 11.73-12.25 µs; E3a's cost
per step, `e3c-a` less `e3c` over `e3c`'s steps, 14.34 µs (tg128), 14.51, 14.97 and 14.84 µs (tg32 at depths 0, 1024,
4096).

Reading:
- E3c's rule: the counters cost tg128 0.5% (tg32 at depth 0 1.1%, at the edge), so the times stand, with the counters'
  own clock reads in them.
- Steps: 1,086 per token below n_kv 1024 and 1,122 from it, 36 more: the split attention's merge step in each of the 36
  layers (it runs from n_kv 512; inferred from the design, the numbers match). Q1's estimate was 1,060.
- The framework's cost by the deck's measure, the AI cores idle per token: 1.2-1.8 ms, 1.3-1.8% of a token (with the
  counters on). Of it: the barriers' own cost 0.59-0.62 ms (0.55 µs each with a clock read; E3d's bare barrier 0.435
  µs), host work in `graph_compute` 0.55-0.64 ms, the one launch 0.19-0.20 ms (it also holds the clock reads the counters
  do not attribute, so it overstates spine-runtime's share; inference), outside `graph_compute` 0.27-0.39 ms
  (llama.cpp, the X100 cores' split, sampling). Depth 0's negative "outside" most likely comes from the run's first
  1-row graph, which also creates the spine-runtime stream: it is in the counters' average and not in llama-bench's
  timed tokens (inferred). The deck's AMD figure is 1.2% (their measurement on other hardware; whether the definitions
  match is not known). Q1's estimate was 1.1%, the barriers alone at twice their cost.
- Imbalance, the slowest tile's lead summed over the steps, is 2.5-3.7 ms per token (2.7-3.6%): more than the whole
  idle time. It counts as busy by the deck's measure; it is in how the kernels split their work, not in the launch
  model.
- E3a in a model run: 14.3-15.0 µs per step with the counters on (run 3, without them: 13.4 µs over the counted 1,086
  steps). Of it, each launch's own overhead 11.7-12.3 µs (E3d's runtime alone: 7.67 µs, 8.79 with TCM and a barrier),
  host work about 1.2 µs more per launch, and the kernels about 1.4 µs slower per step when each step is a launch
  (inference: each launch starting cold). Run 3's question, more steps or dearer launches: the count adds 2.5%; the
  launches cost more in a run than in either benchmark.

Build A's provider checks: rc=1, only from E3b's two 1-token layer cases (as since 2026-10-20); E3a identical in all
cases, the KV cache in our buffer max NMSE 5.24e-04. Builds A and R compile with E3c's counters against the board's
`spert.hpp`.

E3a-0.6B and E2-0.6B (Qwen3-0.6B Q4_0; build A for `ref` and `e3`, build B rebuilt at `060c96d1f` for the E2 modes;
`EXPS="e2 e3a"`, 3 repetitions, depth runs, perplexity on 8 x 512 tokens and at one token per ubatch; logs in
`~/sx/0.6b`). Every run ended with exit code 0. Placement: 2 / 2 splits and flash attention enabled in every mode; as
IGPU and GPU all 29 layers offloaded and the KV cache (4480 MiB at the default context) in the provider's buffer, also
under llama.cpp's automatic fit (`e2-igpu-auto`); `ref` and `e3` keep it in a CPU buffer. The generated text starts the
same in every mode (the pasted summary showed the first 80 characters; its right half was overwritten by another tmux
pane, and the cut values below were read back from their percentage columns, which agree).

| Mode | pp128 | tg128 | tg32 depth 0 | tg32 depth 1024 | tg32 depth 4096 | pp512 depth 4096 |
|---|---|---|---|---|---|---|
| `ref` | 656.39 ± 1.00 | 55.70 ± 0.13 | 57.70 ± 0.89 | 36.03 ± 0.11 | 20.40 ± 0.06 | 110.48 ± 0.14 |
| `e2-igpu` | 660.05 (+0.6%) | 55.73 (+0.1%) | 57.67 (-0.0%) | 35.94 (-0.3%) | 20.36 (-0.2%) | 114.77 (+3.9%) |
| `e2-gpu` | 654.26 (-0.3%) | 55.74 (+0.1%) | 57.95 (+0.4%) | 35.95 (-0.2%) | 20.34 (-0.3%) | 114.71 (+3.8%) |
| `e3` (E3a) | 622.59 (-5.2%) | 35.75 (-35.8%) | 36.72 (-36.3%) | 26.17 (-27.4%) | 16.77 (-17.8%) | 110.17 (-0.3%) |

Perplexity 14.5068 (`ref`) and 14.4825 at one token per ubatch; `e2-igpu`, `e2-gpu` and `e3` 14.506814, `e2-igpu` at one
token per ubatch 14.482462; mean KLD 0.000000 and maximum 0.000055 (the base file's rounding) in all four, the same top
token 100%: identical logits.

Reading:
- E2-0.6B passes E2's rule as on 4B: identical logits at both batch settings, 2 splits per token, five speeds within
  0.6% of `ref`, pp512 at depth 4096 3.8-3.9% faster (4B: 3.6-4.0%; the reading of "within 2%" for a gain is still the
  user's). The prediction held.
- E3a-0.6B: tg128 -35.8%, inside the predicted -35 to -40%: about 10.0 ms more per token (17.95 -> 27.97 ms). F15's
  estimate ("a third or more") is measured. About 12 µs per step if a 0.6B token has about 845 steps (the same per-layer
  graph as 4B's counted 1,086, for 28 layers instead of 36; inferred, not counted). The loss shrinks with depth (-27.4%
  at 1024, -17.8% at 4096: a fixed cost per token on longer tokens). Prompt processing loses 5.2% at 128 tokens (4B:
  1.7%), nothing measurable at 512 tokens after a 4096-token context.

### Run 5 (K3, 2026-10-20): E3b-diag

Build A at `a21d4f694` (by `git bundle`); E3d rebuilt from the same commit. Every program ended normally.

(a) E3d again, with `joined+canary` (tiles that do nothing; 20,000 launches per case, about 160,000 tiles):

| Case, launches in flight | µs per launch | Two tiles on one core (launches) | Tiles that changed core | One TCM held by two tiles (on two cores) | Canary changed across the tile's barrier / with no pause |
|---|---|---|---|---|---|
| joined, 8 | 7.40 | 122 | 0 | - | - |
| joined, 64 | 316.14 | 12,169 | 0 | - | - |
| joined+tcm, 8 | 27.21 | 13,313 | 109,843 | 5,220 (5,086) | - |
| joined+tcm, 64 | 420.99 | 16,554 | 121,187 | 14,737 (14,655) | - |
| joined+canary, 8 | 339.38 | 4,808 | 107,820 | 4,733 (4,722) | 11,738 / 6,207 tiles |
| joined+canary, 64 | 651.72 | 15,470 | 124,098 | 15,400 (15,384) | 42,799 / 22,958 tiles |

No failure, no order violation, no tile without TCM. The host's launch calls cost 5.0-7.2 µs each.

(b) `flagos-check-spacemit --e3b`: runs that differ from one launch per graph, out of 20 per case and variant (failed
runs in brackets):

| Case | `paired` | `staged` (no pair turns) | `direct` (weights not staged in TCM) | `paired+fence` |
|---|---|---|---|---|
| layer, 1 token, down q4_0 | 19 (1) | 20 | 0 | 16 (4) |
| layer, 1 token, down q4_1 | 20 | 20 | 0 | 18 (2) |
| layer, 7 tokens, down q4_0 | 0 | 0 | 0 | 0 |
| 8 matmuls, 1 row, q4_0 | 11 (1) | 19 | 0 | 18 |
| 8 matmuls, 1 row, q4_1 | 18 | 17 | 0 | 16 |

- Largest NMSE against one launch: 1.6-2.4 in the 1-token layers, 0.04-0.09 in the matmuls alone.
- Every failed run logged a tile that waited 10 s for its pair partner ("tile 3 waited 10 s for tile 2 at pair wait 1";
  8 such lines for the 8 failed runs).
- The ordered launches' tiles, in every variant alike, identical runs included:
  - in the layer cases (620 launches each): two tiles on one core in 525-562 launches, tiles moved 4,542-4,588 of 4,960,
    one TCM on two cores in 295-307 launches, both tiles of a pair on one core in 131-174;
  - in the matmuls alone (320 launches each): 229-263, 2,264-2,334 of 2,560, 36-54 and 24-70.
- The first ordered graph's warning: "25 of 31 had two tiles on one core".

Reading, by the rules fixed in §4:
- H1 holds: (a) shows tiles overwriting each other's TCM with no pause in between and one TCM held on two cores, with
  tiles that do nothing; (b)'s differing runs show moved tiles and one TCM on two cores (in the 1-token q4_0 layer with
  `paired`, 283 of the 297 such launches were in differing runs).
- H4 holds: the 1-row matmuls alone differ, and `direct` never does (0 of 80 runs in the four 1-row cases) while
  `staged` does, with the same moving and sharing in both. The ingredient is staging each tile's own weights in TCM;
  the pair turns are not needed.
- H2 explains the failed runs, not the differences: the failures each logged a partner's 10 s wait, while `staged`,
  without pair waits, differs as often. The rule's "only they" does not hold: pairs on one core also occur in runs that
  did not fail.
- H3 ruled out: the fences change nothing.
- So E3b's cause (F16): with launches ordered by `join`, paused tiles resume on other cores and two tiles on two cores
  use one TCM at the same time; the 1-row matmuls stage each tile's own weights there, so a tile multiplies weights
  another tile wrote. `direct` copies only the activation row into TCM, the same bytes in every tile, and the 7-token
  matmuls (path C) use no TCM, so they are unaffected. When both tiles of a pair land on one core, the first spins on a
  partner that cannot run until it gives up after 10 s.
- Today's provider is not exposed: one launch per split, where E3d saw no moved tile and no shared core; E3a's
  host-waited launches, one at a time, were byte-identical in every check, so `plan.md` M3.1's AOT nodes as host-waited
  launches are not either (inferred from E3a).

## 7. Log

- 2026-10-19: requirements, questions and findings collected from the deck, the AMD comparison, upstream `master`, the
  Ascend backend and the board's memory map; E1-E3 switches, checks and runner written and verified on the Mac.
- 2026-10-19: the first `git pull` on the board stopped at git's dubious-ownership check, which the board has had since
  2026-10-08 (handled with `git -c safe.directory=$PWD`); the runner now passes the same exception to every git call.
- 2026-10-19: `prepare` on the K3: builds A and B without errors; `flagos-check-spacemit` passes all checks with the AI
  cores (the layer with the KV cache in our buffer: max NMSE 5.24e-04; one launch per step byte-identical in 4/4 layer
  and 8/8 split cases). E2's rule agreed. E3's rule left open for a comparison with the deck: R12 (the deck's 1.2%) and
  F14 (spine-runtime leaves launches unordered) split E3 into E3a (host waits, built), E3b (in-core ordering, needs the
  board's header) and E3c (the deck's measure).
- 2026-10-19: the first wiring run hung at the first `llama-completion` (`ps`: state `Tl`, stopped, after 27 minutes).
  The cause, from the code: GNU `timeout` runs its command outside the terminal's foreground, and `llama-completion` sets the
  terminal up at start (`console::init`, `tcsetattr` on stdin), which makes the kernel stop it (SIGTTOU). The runner now
  uses `timeout --foreground` and gives every run `/dev/null` as input. E2 runs first; E3a waits for E3b.
- 2026-10-19: wiring check on the K3 (`REPS=1 DEPTH=0 PPL_TEXT= EXPS="e2"`: ref, e2-igpu, e2-gpu, e2-igpu-auto; no
  depth runs, perplexity or repetitions): every run exit code 0; as IGPU and GPU all 37 layers and the KV cache
  (5760 MiB at llama-completion's default context, the model's 40,960 tokens) are in the provider's buffer, also with
  llama.cpp's automatic fit; ref keeps the KV cache in a CPU buffer; flash attention enabled under `-fa auto` in every
  mode; the same text. pp128 / tg128 (one run each): ref 91.88 / 11.30, IGPU 92.15 / 11.28, GPU 92.06 / 11.29. The
  summary misread two columns (the split count, printed as one number when both graphs take the same; and the type
  under the automatic fit); fixed.
- 2026-10-19: run 1 (E1, E2, E2-hp, E2-neg) on the K3; results in §6. E2 passes on logits, splits and speed (pp512 at
  depth 4096 3.6% faster, the rule's reading to confirm); E1 shows ACCEL needs C5a and C5b, C5b for prompt processing
  too; E2-neg confirms F8's two gaps. The summary now reads negative zeros in the KLD and reports the tokens the shift
  runs generated (llama.cpp's "context full" line is a debug message).
- 2026-10-20: E4 scripted (no provider change): `prepare` builds S (upstream `ba360ef`, backends as modules) and M (the
  provider as a module) and smoke-checks the stock programs loading it through `GGML_BACKEND_PATH`; mode `e4-igpu`. On
  the Mac the module builds needing only `libggml-base` and loads through `GGML_BACKEND_PATH`. E1 leaves the default
  `EXPS` (user: M0 and run 1 cover it); the default is now `e2 e2hp e2neg e4`.
- 2026-10-20: E4's `prepare` on the K3 (the board's name lookup failing, the commits came by `git bundle` over ssh):
  builds S and M without errors; both name their core `libggml-base.so.0`, the name the module needs (with
  `libspert.so.1`); upstream `ba360ef`'s `llama-completion --list-devices`, with no FlagOS line in it, loads the module
  through `GGML_BACKEND_PATH` and lists `FlagOS:SpacemiT:0: SpacemiT K3 A100 AI cores (8)` as IGPU. Build A's checks
  all pass. So at run time the mentor's ggml lines are not needed; speed and logits next (run 2).
- 2026-10-20: run 2 (E4) on the K3; results in §6: stock upstream with the provider module as IGPU passes E2's rule
  (identical logits, 2 splits, speeds within 0.3%, pp512 at depth 4096 +4.0%): no upstream change needed at run time.
- 2026-10-20: decided by the user (Q6): native kernels stay in the plugin for now; the K3's package directory is
  `spacemit/k3-a100`; how the package is built and shipped goes to the mentor. Q1 notes where the barrier design comes
  from.
- 2026-10-20: run 3 (E3a) on the K3; results in §6: one launch per step with host waits costs tg128 14.2% (about 14.6 ms
  per token, about 13.8 µs per step if a token has 1,060 steps), prompt processing 0-2%, logits identical.
- 2026-10-20: E3's rule agreed (user): Tier-1 is enough if E3b keeps tg128 within 1% of `ref`, Tier-2 capture is required
  at 5% or more. E3b implemented (`FLAGOS_SPACEMIT_TEST_LAUNCH=per-step-join`); the checks compare it with one launch
  byte for byte and force a failure in it; the runner's `e3` group now measures `ref`, `e3` and `e3b` in one session and
  `prepare` runs the checks 5 times. Mac: built without warnings, all checks pass (by default and with the switch set for
  the whole program; the simulated executor runs E3b as E3a, so only the K3 tests the order); the spine-runtime code
  compiled against a stand-in for `spert.hpp` (scratchpad, not committed).
- 2026-10-20: E3b on the K3 (`EXPS=e3 prepare`: build A without errors, the checks 5 times): every run fails the same
  two checks, the 1-token layer with a Q4_0 and with a Q4_1 down projection under `per-step-join` (differs from one
  launch; once refused or failed); E3a is byte-identical in 4/4 layer and 8/8 split-attention cases, E3b in 8/8
  split-attention and the two 7-token layer cases. Each run reports "ordered launches: 1 of 3 had two tiles on one core"
  (printed once per run): with launches queued, spine-runtime parks a waiting tile and runs another on its core, and
  here two tiles of one launch shared a core. Our tiles assume one core each (their core's TCM, one TCM size agreed by
  all, the 1-row matmuls' pair turns across two cores); the 1-token layer is the case that uses the 1-row matmuls with
  TCM and pair turns. Likely mechanism (unverified): a tile left without usable TCM makes the launch agree on no TCM and
  take the matmul path without it (correct, other arithmetic: "differs"), or a pair wait between two tiles on one core
  gives up after 10 s ("failed"). E3b is not measured. By the agreed rule, the cheapest correct Tier-1 form on
  spine-runtime 0.6 is then E3a (-14%), so Tier-2 capture is required for the K3 unless the runtime orders the
  launches of a stream or keeps tile i on core i (a question for SpacemiT and the mentor).
- 2026-10-20: decided by the user: E3b is not fixed, today's executor stays the K3's graph capture (Q1); the deck guides
  the macro design and is not a standard (R13). F15 sets E3 against the deck. Corrected: AMD's hipGraph replay is an
  opt-in experiment, not something AMD depends on (F12, Q1; `flagos-amd.cpp:366-373, 4808-4812`).
- 2026-10-20: F15 and Q1 scoped (user: how certain is it?): what is measured, what is read in code and what is our
  reading of the deck; E3a's cost is per kernel, so its percentage depends on the model and the phase; E3b "not
  reliably correct" rather than "not correct" (its check prints no size for the difference).
- 2026-10-20: rule (user, `PROJECT.md` §6): conclusions never assert untested findings. Applied to this file's
  conclusions: Q1 (the step count and the 1.1% marked as estimates; the ABI's order as our reading; E3b's shared core
  not shown to cause its differences), F14, F15, Q5 and runs 1-3 (scoped to Qwen3-4B Q4_0), the E3b row.
- 2026-10-20: run 4 prepared (user: the cheap experiments that need no approval): E3d (`spert-launch-bench.cpp`),
  E3c implemented (`FLAGOS_SPACEMIT_TEST_STATS`, modes `e3c` and `e3c-a`), runner group `e3a` (E3a without E3b, for
  Qwen3-0.6B), R12-hw (`FLAGOS_SPACEMIT_TEST_UPSTREAM_Q4_1_M1`, build R in `prepare`). Predictions and rules fixed in
  §4 before any run. Mac: the checks pass with and without the counters (serial, threaded, per step); the runner's dry
  runs and its E3c table on fixture logs; E3d compiles against a stand-in for `spert.hpp`. Nothing run on the K3.
- 2026-10-20: run 4 on the K3, first part (§6): E3d (rules 1 and 3 as predicted, rule 2 fails, rule 4: shared cores
  get the same TCM, no tile without TCM) and R12-hw (upstream's branch wraps: NMSE 4.62, exactly a 2^16 error per
  block). E3c's runs still going.
- 2026-10-20: E3b-diag prepared (user: find E3b's cause): `joined+canary` in E3d, `flagos-check-spacemit --e3b`,
  `FLAGOS_SPACEMIT_TEST_JOIN_FENCE`, and the ordered launches' per-tile cores and TCM counted per run; hypotheses
  H1-H4 and what each would show fixed in §4 before any run. Mac: builds, the checks pass, `--e3b=3` runs (E3b as host
  waits there, so identical and nothing counted); the K3-only code compiles against a stand-in for `spert.hpp`.
  `plan.md` R12 and `PROJECT.md` question 8 still read "found by reading"; to update when run 4 is recorded in full.
- 2026-10-20: run 4, E3c (§6): 1,086 steps per token (1,122 with the split attention), the AI cores idle 1.3-1.8% of
  a token by the deck's measure (barriers 0.6 ms, host work 0.55 ms, the launch 0.2 ms, llama.cpp's side about 0.3 ms),
  imbalance 2.5-3.7 ms, E3a 14.3-15.0 µs per step in a run (11.7-12.3 µs of it each launch's overhead). Q1, F15 and the
  M3 note updated with the counts. The Qwen3-0.6B runs (E3a-0.6B, E2-0.6B) still going.
- 2026-10-20: run 4 ends with Qwen3-0.6B (§6): E2-0.6B passes as on 4B (identical logits, 2 splits, pp512 at depth
  4096 +3.8-3.9%); E3a-0.6B costs tg128 35.8%, as predicted (F15 and Q5 updated). `plan.md` R12 and `PROJECT.md`
  question 8 not yet updated with R12-hw: both files hold older uncommitted edits.
- 2026-10-20: run 5 (E3b-diag, §6) finds E3b's cause, by the rules fixed before the run: with launches ordered by
  `join`, paused tiles resume on other cores and two tiles on two cores use one TCM at the same time (shown with tiles
  that do nothing, F16); the 1-row matmuls' staging of each tile's own weights in TCM turns that into wrong results
  (`direct` never differs); pair partners on one core make the 10 s failures; fences change nothing. The log's earlier
  "likely mechanism" (a tile without TCM) did not occur. Q1, F15 and the E3b row updated; E3b stays unfixed.
