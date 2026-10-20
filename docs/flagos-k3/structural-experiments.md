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

## 2. Questions for the mentor

Each with the problem, the options, a recommendation, and what answers it. Q1, Q2, Q4 and Q5 can be tested first (§4);
Q3 and Q6-Q10 are design questions only. Related numbering in `PROJECT.md` §7 is given where it overlaps.

**Q1. Does a K3 plugin need Tier-2 graph capture?**
- Problem: today a split's steps run in one spine-runtime launch, a barrier between steps; a token takes 2 splits and
  about 1,060 steps (1 ms of barriers at 0.94 µs each, `build.md` §7), about 1.1% of a token's 88.5 ms: near the deck's
  1.2% for AMD (R12). A plugin with Tier-1 alone gets one `launch` per kernel on a stream the ABI treats as ordered. On
  spine-runtime nothing orders launches (F14): the plugin must order them itself, by waiting for each launch on the host
  (11.2-11.5 µs per launch against a barrier's 0.89-0.94 µs: about 12 ms per token, tg128 about -12%; estimate) or,
  if the board's spine-runtime has it, by making each launch's tiles wait inside the cores for the previous launch
  (`Context::join`; cost unknown).
- Options: (a) the K3 plugin implements Tier-2 capture: record the launches, replay them as one spine-runtime launch with
  a barrier between kernels, which is today's executor; (b) a "launch list" call added to Tier-1; (c) accept the cost.
- Recommendation: (a), unless E3b shows ordered launches cost about as little as the barriers. AMD's provider already
  depends on replay (hipGraph).
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
- Measured 2026-10-19 (§6, run 1): as IGPU or GPU the K3 matches today's speed with identical logits and no llama.cpp
  change; ACCEL without C5a and C5b loses 24-95% in generation and 8-52% in prompt processing. So (b), once the provider
  fixes F8's two gaps (confirmed by E2-neg); (a) only if the K3 must keep accelerating next to a discrete GPU.
- Measured 2026-10-20 (§6, run 2, E4): as IGPU the provider also runs from stock upstream llama.cpp, loaded at run time
  as a module, with the same speed and identical logits: no upstream change at all.
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

**F12. The AMD design against our plan** (2026-10-19). AMD: GPU or IGPU, no llama.cpp change; hipGraph capture and
replay; GGUF weights read in place for generation plus a private F16 dequantized cache; all kernels Triton AOT with a
manifest and a strict loader; variants chosen in the provider by name and tuning profiles; workspace owned by the
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
The ABI slide's Tier-1 assumes an ordered stream ("在 stream 上排序" for copies, `stream_synchronize`); on spine-runtime the
plugin must provide that order. Today's executor needs none of it: one launch per split, barriers inside (`Context::sync`).
The board's header (`~/spine-runtime/include/spert.hpp`, read 2026-10-20) has what in-core ordering needs:
`Stream::launch` returns a `Future` and copies the arguments into storage the launch owns (`:579-590`; the Future is
invalid if the launch cannot be admitted); `Future` is a shared handle with `sync()` and `sync_all(const
std::vector<Future> &)` (`:202-278`); `Context::join(Future &)` (`:469`), `spawn` (`:466`), `yield()` (`:447-448`) and
the grid barrier `sync()` (`:453`); a stream sends its launches only to the workers of its granted cores (`:506-507`,
`:522`).

## 4. Experiments

All on Qwen3-4B Q4_0 unless named, in one session, every mode against the reference (today's build, default flags)
with the modes interleaved in each repetition: the board drifts between sessions.

| ID | Answers | What changes | Prediction | Decision rule | Status |
|---|---|---|---|---|---|
| E2 | Q5 | Build B (upstream's `llama-context.cpp`), `FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=igpu` and `gpu`, `-ngl 99`; once with the default `-ngl` (placement only) | 2 splits per token; flash attention enabled under `-fa auto`; logits identical to the reference (KLD 0, same top p 100%; the same kernels run the same nodes); speed within 1-2%; GPU = IGPU; the default `-ngl` offloads every layer | agreed (user, 2026-10-19): "IGPU is viable with no llama.cpp change" if logits are identical at both batch settings, a token takes 2 splits, and pp128, tg128, tg32 at depths 0/1024/4096 and pp512 at depth 4096 are within 2% of the reference | done 2026-10-19 (§6): passes; pp512 at depth 4096 +3.6% |
| E2-hp | Q5 | E2's IGPU and the reference, each with and without `FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1`, depth runs | unknown; the 128 KiB block spacing suggests no 2.2x slowdown | KV cache on huge pages "hurts" if IGPU's tg32 at depth 4096 loses at least 2% more from huge pages than the reference does, "helps" if it gains at least 2% more | done 2026-10-19 (§6): neither in generation; mixed in prompt processing at depth |
| E2-neg | F8 | Build B as IGPU, and the reference: a Q8_0 KV cache (`-ctk q8_0 -ctv q8_0 -fa on -c 512`), and a context shift (`-c 256 -n 400 --ignore-eos --context-shift`) | IGPU aborts in both ("pre-allocated tensor ... cannot run the operation": `SET_ROWS`, then `ROPE`); the reference runs both (the shift logs "context full, swapping") | confirms or refutes F8's two gaps | done 2026-10-19 (§6): both confirmed |
| E1 | Q5 | Build B as ACCEL: `-fa auto` (llama.cpp's default) and `-fa on` | `-fa auto`: flash attention turned off with llama.cpp's warning, attention on the X100 cores, generation at depth far below the reference (before M2e.1: tg32 at depth 4096 1.80). `-fa on`: 288 splits per token (36 layers x 4 norms x 2), tg128 about -30% (X3b: 7.52 against 10.75), prompt processing unchanged | descriptive: what ACCEL costs without the patches | done 2026-10-19 (§6): -24 to -95%; not repeated (user, 2026-10-20: M0's X3/X3b and run 1 cover it; out of the default `EXPS`) |
| E3a | Q1 | Build A, `FLAGOS_SPACEMIT_TEST_LAUNCH=per-step`: one spine-runtime launch per step, the host waiting for each (the order spine-runtime's documentation prescribes for dependent kernels, F14): a Tier-1 plugin without in-core ordering | tg128 about -12% (about 11.5 µs more per step at about 1,060 steps per token); pp128 about -1%; logits identical (checked on the K3: 4/4 layer and 8/8 split cases byte-identical) | open (user, 2026-10-19: more investigation against the deck first). Proposed, anchored to R12: Tier-1 is enough for the K3 if its cheapest correct form (E3b if possible, else E3a) keeps the framework's cost within about 2% of a token (tg128 within about 1% of the reference, which already carries about 1.1%); Tier-2 capture is needed if it costs 5% or more; in between, report | ready (runs in the full run) |
| E3b | Q1 | One launch per step queued without host waits, each launch's tiles first waiting inside the cores for the previous launch (`Context::join`, if the board's spine-runtime has it): Tier-1 on an ordered stream, as the ABI assumes | unknown: between the reference and E3a | as E3a | feasible (2026-10-20: the board's `spert.hpp` has `Context::join`, `Future::sync_all`, copyable Futures; F14); design: each tile joins its own copy of the previous launch's Future before taking its TCM, every launch's arguments live until `sync_all`, a launch not admitted waits for those in flight and is retried once; not implemented |
| E3c | Q1, R12 | Test-only counters: per token, the time inside launches, the kernels' own time (tile 0) and the steps; the rest of a token is the AI cores waiting on llama.cpp, the CPU's split and sampling | today's cost the deck's way (device idle per token), split into inside and outside launches, for comparison with AMD's 0.6 ms of 45 ms | none: a measurement | proposed, not implemented |
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
| `FLAGOS_SPACEMIT_TEST_LAUNCH=per-step` | `spacemit_launch_per_step` and `spacemit_backend_graph_compute`, `flagos-spacemit.cpp` | runs the executor once per step instead of once per split; the same steps, order and workspace, so the same outputs; read at every graph; `per-split` names the default |

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

`scripts/structural-experiments.sh`: `prepare`, `run`, `summary` (§4).

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
  loss beyond 2% (the user's reading pending). GPU and IGPU are the same. So the K3 runs at today's speed and accuracy
  as IGPU or GPU with no change to llama.cpp.
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
reporting IGPU, the K3 runs llama.cpp at today's speed and with identical results with no change at all to upstream
llama.cpp or ggml: neither C5a and C5b nor the mentor's ggml lines are needed at run time. What this does not settle:
- F8's two gaps hold here too (the KV cache is in the provider's buffer), so IGPU needs that provider fix before it can
  be the default.
- The type came from a test switch; making IGPU the provider's type is D3's question for the mentor.
- Delivery: the module still builds inside this repository (the mentor's CMake lines are needed to build it, not to run
  it), it must match the host's ggml version (`ba360ef` here), and stock programs find it only through
  `GGML_BACKEND_PATH`; finding it by name needs the one loader line (`ggml_backend_load_best("flagos", ...)`).

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
