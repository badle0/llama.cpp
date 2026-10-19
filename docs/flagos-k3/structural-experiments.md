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

## 2. Questions for the mentor

Each with the problem, the options, a recommendation, and what answers it. Q1, Q2, Q4 and Q5 can be tested first (§4);
Q3 and Q6-Q10 are design questions only. Related numbering in `PROJECT.md` §7 is given where it overlaps.

**Q1. Does a K3 plugin need Tier-2 graph capture?**
- Problem: today a split's steps run in one spine-runtime launch, a barrier between steps; a token takes 2 splits and
  about 1,060 steps (1 ms of barriers at 0.94 µs each, `build.md` §7). A plugin with Tier-1 alone gets one `launch` per
  kernel: launch 11.2-11.5 µs against a barrier's 0.89-0.94 µs, about 12 ms of a token's 88.5 ms (tg128 about -12%;
  estimate).
- Options: (a) the K3 plugin implements Tier-2 capture: record the launches, replay them as one spine-runtime launch with
  a barrier between kernels, which is today's executor; (b) a "launch list" call added to Tier-1; (c) accept the cost.
- Recommendation: (a). AMD's provider already depends on replay (hipGraph).
- Answered by: E3.

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
- Answered by: E1, E2, E2-hp, E2-neg, E4.

**Q6. During the transition, do native RVV/IME kernels go in the package or in the plugin?** The deck says both: the
architecture slide keeps them in the plugin (R7); the flow slide lists "HSACO / ELF / 厂商格式" (vendor format) as package
binaries. Recommendation: the package (K3 ELF `.so` files with FlagTree's calling convention), so the core picks them like
any variant and Triton versions replace them one by one.

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

**F8. The KV cache in the provider's buffer** (IGPU or GPU; from reading the code).
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
(E4). The provider has no link dependency on the CPU backend (its mentions of `ggml-cpu` are comments).

## 4. Experiments

All on Qwen3-4B Q4_0 unless named, in one session, every mode against the reference (today's build, default flags)
with the modes interleaved in each repetition: the board drifts between sessions.

| ID | Answers | What changes | Prediction | Decision rule (proposed; confirm before the run) | Status |
|---|---|---|---|---|---|
| E2 | Q5 | Build B (upstream's `llama-context.cpp`), `FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=igpu` and `gpu`, `-ngl 99`; once with the default `-ngl` (placement only) | 2 splits per token; flash attention enabled under `-fa auto`; logits identical to the reference (KLD 0, same top p 100%; the same kernels run the same nodes); speed within 1-2%; GPU = IGPU; the default `-ngl` offloads every layer | "IGPU is viable with no llama.cpp change" if logits are identical at both batch settings, a token takes 2 splits, and pp128, tg128, tg32 at depths 0/1024/4096 and pp512 at depth 4096 are within 2% of the reference | ready |
| E2-hp | Q5 | E2's IGPU and the reference, each with and without `FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1`, depth runs | unknown; the 128 KiB block spacing suggests no 2.2x slowdown | KV cache on huge pages "hurts" if IGPU's tg32 at depth 4096 loses at least 2% more from huge pages than the reference does, "helps" if it gains at least 2% more | ready |
| E2-neg | F8 | Build B as IGPU, and the reference: a Q8_0 KV cache (`-ctk q8_0 -ctv q8_0 -fa on -c 512`), and a context shift (`-c 256 -n 400 --ignore-eos --context-shift`) | IGPU aborts in both ("pre-allocated tensor ... cannot run the operation": `SET_ROWS`, then `ROPE`); the reference runs both (the shift logs "context full, swapping") | confirms or refutes F8's two gaps | ready |
| E1 | Q5 | Build B as ACCEL: `-fa auto` (llama.cpp's default) and `-fa on` | `-fa auto`: flash attention turned off with llama.cpp's warning, attention on the X100 cores, generation at depth far below the reference (before M2e.1: tg32 at depth 4096 1.80). `-fa on`: 288 splits per token (36 layers x 4 norms x 2), tg128 about -30% (X3b: 7.52 against 10.75), prompt processing unchanged | descriptive: what ACCEL costs without the patches | ready |
| E3 | Q1 | Build A, `FLAGOS_SPACEMIT_TEST_LAUNCH=per-step`: one spine-runtime launch per step instead of per split | tg128 about -12% (about 11.5 µs more per step at about 1,060 steps per token); pp128 about -1%; logits identical | graph capture "needed" for the K3 if tg128 loses 5% or more; "not needed" at 2% or less; in between, report | ready |
| E4 | Q5, F13 | Stock `ba360ef` binaries built with `GGML_BACKEND_DL=ON`, `libggml-flagos.so` from this fork loaded through `GGML_BACKEND_PATH`, with E2's IGPU | the same as E2 | "no ggml lines needed" if it loads and matches E2 | manual (below) |
| E6 | Q2 | GGUF's bytes in the buffer ggml sees plus a private repacked copy | about +2.35 GB, the same speed, exact read-back | evidence for option (e) | not implemented |
| E5 | Q2 | the 1-row path reads GGUF's layout, interleaving 32 rows while copying into TCM | unknown; generation only | how much repacking buys | not implemented |
| Q4 table | Q4 | on paper: our kernel choices as manifest entries (startup filter for device facts, `when` for per-token shapes, one generic variant per contract), each condition `when` cannot express marked | some inexpressible (ranges, equality, strides, layout) | input for Q4 | to do |
| M3.0+ | Q2 | the M3.0 spike (`plan.md` M3) plus a Triton Q4_0 matmul reading GGUF's layout, timed against ours | unknown | input for Q2 (d) | with M3.0 |

### How to run E1, E2, E2-hp, E2-neg and E3 (on the K3, inside tmux)

```bash
git pull                                                           # the switches, checks and script
docs/flagos-k3/scripts/structural-experiments.sh prepare           # build B, then build A's tools and checks
docs/flagos-k3/scripts/structural-experiments.sh run ~/models/Qwen3-4B-Q4_0.gguf 2>&1 | tee ~/sx-run.log
docs/flagos-k3/scripts/structural-experiments.sh summary ~/sx/Qwen3-4B-Q4_0     # the tables again
```

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

### E4 (manual; not yet tried)

```bash
git -C ~/llama.cpp-flagos worktree add --detach ~/llama.cpp-stock ba360ef        # upstream's tree, no FlagOS lines
cmake -S ~/llama.cpp-stock -B ~/llama.cpp-stock/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON \
  -DBUILD_SHARED_LIBS=ON && cmake --build ~/llama.cpp-stock/build --target llama-bench llama-completion
cmake -S . -B build-flagos-dl -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON \
  -DGGML_FLAGOS=ON -DGGML_FLAGOS_DENGLIN=OFF -DGGML_FLAGOS_AMD=OFF -DGGML_FLAGOS_SPACEMIT=ON && \
  cmake --build build-flagos-dl --target ggml-flagos
GGML_BACKEND_PATH=$PWD/build-flagos-dl/bin/libggml-flagos.so FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=igpu \
  ~/llama.cpp-stock/build/bin/llama-bench -m ~/models/Qwen3-4B-Q4_0.gguf -t 8 -ngl 99 -p 128 -n 128 -ub 128 -mmp 0
```

Risks: the module's path and name in the build tree, and whether the stock binaries' `libggml-base.so` serves the module
(both at `ba360ef`, so the ABI should match).

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
command, and its summary on sample logs (bash 3.2). Not yet on the K3.

## 6. Results

To be filled from the board run (`summary` output), with the date and commit.

## 7. Log

- 2026-10-19: requirements, questions and findings collected from the deck, the AMD comparison, upstream `master`, the
  Ascend backend and the board's memory map; E1-E3 switches, checks and runner written and verified on the Mac.
