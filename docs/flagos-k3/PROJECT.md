# FlagOS × SpacemiT K3 — project context

Loaded at every Claude Code session, with `CLAUDE.md` beside it, through `.claude/CLAUDE.md` (the root `CLAUDE.md` and `AGENTS.md` stay upstream's). Details live in sibling files; read them when the task touches their topic:

- `docs/flagos-k3/build.md` — step-by-step builds (Mac, K3), verification, benchmarks, troubleshooting
- `docs/flagos-k3/k3-hardware.md` — board facts, AI cores, TCM investigation and its evidence
- `docs/flagos-k3/upstream-review.md` — structural concerns in ggml-flagos (fusion + provider shim), with file:line evidence
- `docs/flagos-k3/plan.md` — build plan for the SpacemiT provider: architecture, decisions D3–D9, milestones M0–M4 with tests and exit criteria, risks
- `docs/flagos-k3/device-type.md` — GPU vs IGPU vs ACCEL: what the type controls in llama.cpp, what "right behavior" means (R1–R6), predictions and experiments X0–X6
- `docs/flagos-k3/code-tour.md` — reading guide to the provider's code, stop by stop, each compared with the AMD provider, with links to the lines
- `docs/flagos-k3/upstream-changes.md` — every change this fork makes to upstream llama.cpp/ggml or to the mentor's FlagOS code, with commit, lines, reason and status
- `docs/flagos-k3/structural-experiments.md` — the mentor's structural requirements (FlagOS plugin ABI, kernel package, minimal upstream change), questions Q1–Q10, findings F1–F16 (ACCEL/IGPU/GPU upstream, Ascend, the K3's memory map, spine-runtime's unordered launches, E3 against the deck, tiles that `join` moves sharing one TCM), and experiments E1–E6, E3c, E3d, E3b-diag and R12-hw with their switches (`FLAGOS_SPACEMIT_TEST_DEVICE_TYPE`, `FLAGOS_SPACEMIT_TEST_LAUNCH`, `FLAGOS_SPACEMIT_TEST_STATS`) and runner `scripts/structural-experiments.sh`
- `docs/flagos-k3/structural-review.md` — structural review of the provider (2026-10-26) against the deck's core / runtime plugin / kernel package design (pages 6-8): the deck's design in one table, our files mapped to its layers, the gaps G1-G10 with evidence, the M3 decision's consequences (which ops FlagGems covers, the launch model), questions Q11-Q14 for the mentor, proposed next steps
- `docs/flagos-k3/fusion-and-splitting.md` — groundwork for operator fusion and graph splitting (from 2026-10-21): FlagOS's provider modes and Common's fusion catalog, our 30-step layer, the measured per-token budget, candidates ranked by estimate, splitting options; the measurements, one candidate at a time (§7): the op counters (`FLAGOS_SPACEMIT_TEST_OP_STATS`), rank 1 behind `FLAGOS_SPACEMIT_QUANTIZE=shared`, runner `scripts/fusion-experiments.sh` with `scripts/op-stats.py`; read first for any fusion or splitting work
- `docs/flagos-k3/progress.md` — the history: the status narrative, the decisions with their evidence, the questions' full record, the log of items 1-32 (moved out of this file 2026-10-31); read the section a task touches
- `docs/flagos-k3/upstream-precedents.md` — upstream llama.cpp master (commit `8ca4060b`, 2026-10-28) read against the S1 design: ggml's own backend loader (API version 3), Metal's op/device layering behind a C header, Hexagon's host/DSP split (op batches, VTCM, a host-solved kernel choice), ET's uberkernel step list, CUDA graphs keyed by the split's uid; the S1 alterations proposed (A1-A7); zero copy in ggml's terms (`is_host`, `buffer_from_host_ptr`, the loader's raw reads), what the provider has and lacks, where it goes (a mapped buffer type, two plugin entries); SpacemiT's in-tree IME changes since our base

## 1. Task

- Goal: run llama.cpp inference on the SpacemiT K3 through FlagOS — a **SpacemiT provider inside ggml-flagos**, analogous to how vllm-plugin-FL plugs FlagOS into vLLM.
- Deliverable shape: in-tree provider at `ggml/src/ggml-flagos/providers/spacemit/` (M1 skeleton done). ggml-flagos builds as one ggml backend (`libggml-flagos`); providers are compiled into it. Since 2026-10-26 (decided with the mentor, D10) the target shape is a K3-only split of that provider into a core layer (the deck's core, built here without the AMD part) and a plugin (`libflagos-provider-spacemit.so`) that the core loads; no separate `flagos-runtime.h` from FlagOS (`plan.md` S1 and S2; `structural-review.md` §3, §8).
- Mentor also asked for a design review of ggml-flagos, focused on **kernel fusion** (providers have different kernel structures) and the **shim between Common and providers** → `upstream-review.md`.
- Status (2026-11-01): M0-M2 done for the Qwen3 targets (every layer, the KV cache and attention on the AI cores; Qwen3-4B Q4_0 pp128 about 106 t/s and tg128 11.07 against the upstream IME path's 79 and 10.98 in one session; perplexity within 0.1% of the CPU, mean KLD 0.0025 / 0.0012 at 512-token batches / one token; the numbers of every acceptance run in `build.md` §7). The device type is IGPU (D3). The structural work (D10) is in S1: steps 1, 2, 2b and 3 to 3c accepted on the K3; the plan cache (A4) is the default since step 3c: a hit trusts the split's uid and skips the contract check (the host work per 4B token 629 to 46 µs; the full path 0.6% slower on tg128 on 4B and 2.3% on 0.6B; identical logits). Next: steps 4 and 5. M3 and fusion wait for S2. The full narrative: `progress.md` §A and §D.
- Terms (fixed 2026-10-13): the phases are *generation* (one new token per step) and *prompt processing* (the prompt, in ubatches of up to 512 tokens). Kernels and paths are named by structure, never by phase, since the call's shape picks them: attention's *one-chunk kernels* (`forward_one_chunk`: rows one at a time, or 2 or 4 that share K, V and a mask row) for calls of fewer than 64 tokens, and its *tiled kernel* (`forward_tiled`: 64 query rows x 64 KV positions) from 64 tokens with TCM; the matmuls' 1-row path and paths A and C. "Decode" and "prefill" are not used for either (llama.cpp's `llama_decode` runs any batch).

## 2. Decisions (proposed = still to confirm with mentor; the evidence behind each in `progress.md` §B)

| ID | Decision | Status |
|---|---|---|
| D1 | Standalone backend design (ggml device/buffer/backend), not a ggml-cpu extra buffer type | adopted via ggml-flagos |
| D2 | Work in the mentor's fork, in-tree; the earlier standalone `ggml-plugin-FL` repo is retired | adopted |
| D3 | ggml device type `IGPU` (was `ACCEL` 2026-10-08 to 26) | agreed by the mentor; in the code since 2026-10-27, accepted on the K3; `FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=accel|gpu` test-only; C5a and C5b reverted 2026-10-28 |
| D4 | Weights in an IME-repacked buffer, stored once (`is_host = false`); the provider reads CPU buffers for all other operands; Q4_1 and Q6_K requantized to Q8_0 at load, read back as the nearest Q4_1 or Q6_K | building on it since M2b; mentor confirmation pending |
| D5 | spine-runtime executor, one launch per split, a barrier per step | proposed; implemented in M2a; re-confirmed by E3r 2026-10-30 (one launch per step costs 14-35% of generation) |
| D6 | Copy into the provider only the IME kernel, quantizer and repack functions it uses; port ggml-spacemit's tiling and RVV ops, fixing them | decided (user, 2026-10-09) |
| D7 | First target Qwen3-0.6B / 4B Q4_0, then Qwen3.5-4B Q4_K_M | proposed |
| D8 | `flagos_provider_kind::cpu_accelerator`; `ai_accelerator` would classify the K3 as an NPU | proposed, open |
| D9 | FlagTree AOT package (M3): FlagGems' kernels through FlagTree for the ops they have; the IME and RVV kernels for the rest | decided (user, 2026-10-26); deferred until the structural work is done |
| D10 | The core / plugin split, K3-only, no separate `flagos-runtime.h`; the core first (S1), then the plugin (S2) with the weights code | decided (user and mentor, 2026-10-26); S1 in progress |

Caveat: ggml's scheduler runs splits sequentially (`ggml_backend_sched_compute_splits`); do not assume the X100 CPU backend and A100 provider run concurrently.

## 3. Code and references

| What | Where | Notes |
|---|---|---|
| This repo | fork of `kevinzs2048/llama.cpp`, branch `feature/flagos-spacemit` | = mentor's `feature/flagos-amd-890-opt` @ `5794e12` + macOS link fix `eaa25ff` |
| Mentor baseline | `kevinzs2048/llama.cpp` `feature/flagos-amd-890-opt` @ `5794e12` | exactly the design PDF's baseline; upstream base `ba360ef` (2026-08-11). The mentor's newest branch, `feature/flagos-amd-890-audit-20261010` (2026-10-10; fetched read-only 2026-10-26 as `kevin/...`), is AMD work on the same compiled-in provider model: no plugin ABI, loader or package vocabulary (`structural-experiments.md` F1); it removes `ggml_backend_flagos_reg_devices()` (`upstream-changes.md` M1) |
| Mentor design history | same repo, branch `feature/flagos-multi-provider-backend` | 4 `FLAGOS_BACKEND_*.md` design/review docs not on the AMD branch; §14 is the K3 provider design (co-built in `providers/spacemit`, no parallel ggml-spacemit; it chose K3 = GPU, which the mentor now considers open) |
| FlagOS Common | `ggml/src/ggml-flagos/flagos-{provider,registry,target,graph-plan}.*` | registry, profiles, fixed-fusion side-plan |
| Provider templates | `providers/denglin/flagos-denglin.cpp` (read first, 3.2k lines), `providers/amd/` (strict AOT loader, many fusions) | both report GPU/IGPU device types |
| Upstream SpacemiT CPU path | `ggml/src/ggml-cpu/spacemit/` (`ime.cpp`, `ime_env.cpp`, `spine_mem_pool.cpp`, vendored `spine_tcm.h`) | extra-buffer-type design; A100 binding, TCM, IME2 kernels |
| SpacemiT standalone backend | `spacemit-com/llama.cpp` branch `agent/ggml-spacemit-backend` @ `4e782bc`, `ggml/src/ggml-spacemit/` | ACCEL backend on spine-runtime; own repacking buffer type; ~35 ops. Primary K3-side reference. Copies the in-tree kernel/repack files unchanged; branch `mtmd-backend` @ `64316cd` (2026-09-24) has the same kernel, IME and repack code (checked 2026-10-09). Measured 2026-10-08 (`build.md` §7): the upstream IME path, not ggml-spacemit, is the fastest generation baseline (Qwen3-4B Q4_0 tg128 11.10 vs 6.82 t/s) |
| spine-runtime | `spacemit-com/spine-runtime` release 0.6.0 read offline (2026-09-25); the board runs 0.6.3 (`build.md` §5, installed 2026-10-07) (`libspert.so`, `spert.hpp`) | `spert::Stream::launch(Grid, fn)`, `Context::program_id/grid_dim/sync/shared_buffer`, `backend_info()`; does the `/proc/set_ai_thread` opt-in itself |
| FlagTree SpacemiT backend | `flagos-ai/FlagTree` `third_party/spacemit/` (read @ `a134c5c`, 2026-10-14) | Triton → linalg → spine-mlir → riscv64 `.so`; `AICPUTarget(...)`; launcher ABI in `backend/driver.py`; no AOT export, CI only under QEMU; SpacemiT's own copy, `spacemit-com/spine-triton`, is ahead (`plan.md` M3) |
| Design doc | `ggml-flagos-provider-design.pdf` V2.0 (AI-written summary of the AMD work) | treat claims as needing code verification |

## 4. K3 essentials (full detail in `k3-hardware.md`)

- Bianbu 4.0, kernel 6.18.3, riscv64; GCC 15.2 (assembles IME1+IME2), CMake 4.2, Ninja; 31 GiB RAM, no swap.
- CPUs 0–7: X100 @ 2.4 GHz (default affinity). CPUs 8–15: A100 AI cores @ 2.0 GHz, usable only after a thread writes `/proc/set_ai_thread`.
- RVV VLEN: X100 256 bits, A100 1024 bits (measured 2026-10-07; a thread's `vlenb` changes when it migrates); identical ISA strings. A100 extras: IME matrix extension, TCM (8 × 384 KiB via `/dev/tcm`, assigned per core pair). spine-runtime's `shared_buffer()` is real TCM on this board (`k3-hardware.md` §4).
- TCM works (resolved 2026-10-07, `k3-hardware.md` §4): upstream IME path passes 1198/1198 `MUL_MAT` with TCM on. That test uses the CPU backend's plain buffers, so it checks thread pinning and TCM handling, not the IME kernels. Two rules: run `~/tcmtest/tcmrelease --apply` before IME runs (a crashed/interrupted run leaves blocks stuck in `/dev/shm/tcm_sync_standalone`), and use ≤ 8 threads (`OMP_THREAD_LIMIT=8` for test-backend-ops, `-t 8` for llama-bench/cli), else `ime.cpp:1705` aborts.

## 5. Everyday commands (on the K3, repo root)

```bash
tmux attach -t build || tmux new -s build                  # long jobs always inside tmux
cmake --build build --target ggml-flagos flagos-check-provider flagos-check-target flagos-check-graph-plan flagos-check-registry test-backend-ops llama-bench llama-cli
for t in provider target graph-plan registry; do ./build/bin/flagos-check-$t; done
~/tcmtest/tcmrelease --apply && OMP_THREAD_LIMIT=8 ./build-ime/bin/test-backend-ops -o MUL_MAT -b CPU 2>&1 | tail -3
grep -nE 'error:|FAILED' build.log | head                  # after: cmake --build ... 2>&1 | tee build.log
```

## 6. Working rules for Claude Code

The delegation workflow (the `coder` subagent) and the rule that the root `CLAUDE.md` and `AGENTS.md` stay upstream's are in `CLAUDE.md` beside this file.

Before any code change:
1. Prefer the **smallest change** that achieves the goal; avoid touching upstream/mentor files unless required.
2. State briefly **why the chosen approach is optimal**, list **realistic alternatives**, and why they lose. Then make the change.
3. Verify facts in the code (file:line) instead of assuming; say when something is unverified.

While changing code:
- No redundant prints. Use `GGML_LOG_DEBUG` / `GGML_LOG_WARN` / `GGML_LOG_ERROR` for diagnostics worth keeping; remove ad-hoc `printf`s before finishing.
- Keep terminal output focused: pipe long builds to a log and grep for `error:|FAILED`.
- Never weaken `supports_op`: return true only for exactly what the kernel handles (see `upstream-review.md`).
- Any change outside `providers/spacemit/` and `docs/flagos-k3/` (upstream llama.cpp/ggml files, or the mentor's FlagOS code) gets an entry in `upstream-changes.md` in the same commit, ideally a commit of its own; removing such a change removes its entry (user, 2026-10-13).
- Do not change system configuration on the shared board (cgroups, `/proc`, drivers). Reading is fine.
- CPU-only checks on the K3 (a perplexity or KLD reference, any CPU-versus-provider comparison): a CPU run with several threads on the X100 cores is wrong at random for any model with Q4_1 tensors (item 28), and a wrong reference looks like a provider accuracy regression. Run the reference single-threaded with the provider disabled, twice, and keep it only if the two logit files are byte-identical: `FLAGOS_SPACEMIT_DISABLE=1 ./build-flagos/bin/llama-perplexity -m $M -f ~/ppl.txt -c 512 --chunks 8 -t 1 -fa on --kl-divergence-base base.bin`, the same again into `base2.bin`, then `cmp base.bin base2.bin`. Never use an 8-thread CPU run as a reference, and never run a CPU-only check without `FLAGOS_SPACEMIT_DISABLE=1`: without it the provider computes (the kernel log's `hmp_set_ai_thread` lines show it) and the run says nothing about the CPU. Same-session comparisons against one verified base keep their meaning. The acceptance runner's CPU reference is single-threaded since 2026-10-29 (`tools/spacemit_check.py`); its second, byte-compared run is still done by hand.

After any code change, always report:
1. **What changed** (files, functions, and why).
2. **How to build, run and test it** (exact commands for this board).
3. **Ways to play with it**: which parameters/env vars/shapes to change and what result to expect.

Conclusions, in chat, in these docs and in commit messages (user, 2026-10-20):
- Never assert an untested finding. State as found only what was tested: measured on the K3 or the Mac, checked by a test, or read in the code (file:line). Label everything else as what it is (estimate, prediction, inference, our reading of a document, a vendor's documentation not yet observed) and name the test that would settle it.
- A tested result names its setup (model, phase, depth, runtime version, number of runs) and is not stretched to setups that were not run. A cause or mechanism stays unverified until a test isolates it.

Git (from AGENTS.md, applies to anything that may go upstream):
- Do not commit or push without explicit approval each time. Commit messages for this fork carry no AI attribution line (no `Assisted-by:`, no `Co-authored-by:`; user decision 2026-10-08).
- Push only to `origin` (the user's fork). Never push to the mentor's repo.
- Never write the board's SSH hostname or credentials into the repo (the fork is public).

## 7. Open questions for the mentor

Same numbering as `plan.md` §7, which gives the context; the full record of each, with every date, in `progress.md` §C.

1. ~~Device type~~ decided: IGPU (D3). Still open: FlagOS `caps.kind`, `cpu_accelerator` (proposed) or `ai_accelerator` (D8)?
2. Memory: weights repacked and stored once, Q4_1 requantized to Q8_0 at load (D4): acceptable? The requantization's trade-off: 1.0-2.4% slower generation, KLD 0.00138 against the CPU's own Q4_1 path's 0.00109 on Llama-3.2-1B.
3. Is closed `libspert` (spine-runtime) acceptable as a hard dependency?
4. Kernel reuse by copying only the functions used (D6): objections?
5. Phase 0 target: Qwen3-4B Q4_0 or Qwen3.5-4B Q4_K_M?
6. ~~FlagTree AOT in scope~~ decided (D9). Still open, deferred with M3: cross-compile on x86 or build on the K3; FlagTree's copy or SpacemiT's `spine-triton`; the manifest schema; the success criterion.
7. Is co-building in `providers/spacemit` agreed with SpacemiT, who ship ggml-spacemit as an ACCEL backend?
8. The reports for SpacemiT (each with its evidence in `progress.md` §C and `build.md` §7): TCM (no dead-owner recovery, the `try_wait` unit); the CPU path wrong at random under several threads on the X100 cores for models with Q4_1 tensors (item 28); the 16-bit zero-point product in the 1-row Q4_1 IME kernel (R12); the in-tree Q4_1 conversion's integer zero point (+5.5% perplexity on Llama-3.2-1B); the Q4_0 scale fix's KLD cost; GCC 15.2.0's epilogue fault (item 29); spine-runtime's shared-buffer view, tile stacking and the `Deadlock` above a pool eighth (E7, E7b, item 31); and the questions: may a tile leave its worker inside one launch, is the stacking intended, can a launch reserve a worker's pool, is a graph or multi-kernel launch planned?
9. ~~ACCEL gaps in llama.cpp (C5)~~ approved 2026-10-14, then reverted 2026-10-28 with IGPU (U1, U2); the mentor to be told.
10. ~~Order after M2d~~ decided (M2c first, then M2e).

Zero copy: the provider reads host buffers in place and its buffers are host DRAM; what it lacks is the model file used in place (`buffer_from_host_ptr`), proposed as a mapped buffer type for S2 (`upstream-precedents.md` §5); `is_host` stays false.

Priority order for the mentor (2026-10-26, `structural-review.md` §7.1): (4) `caps.kind`; (5) the `when` vocabulary and the manifest schema; (6) co-building and where the plugin's source lives; (7) the reports to SpacemiT; (8) confirmations of D5, questions 3, 4, D7 and 5; (9) M3's remaining parts, parked until the structural work is done. Closed: the plugin ABI package and the weights' ownership (D10), capture as the main path (an input to our own interface).

## 8. Current state and next step (2026-11-01)

The log of every milestone and experiment, items 1-35 with their numbers, is `progress.md` §D; the acceptance runs are in `build.md` §7; the plan with its exit criteria is `plan.md` §4.

- Done: M0-M2 (items 1-18), the structural experiments E1-E7b and E3r (items 20, 31; `structural-experiments.md`), the matmul path work (item 21), the IGPU switch (item 22), S1 step 1 (the C interface header, the `dlopen` loader, item 29), step 2 (the contract table bound by name, item 30), step 2b (the default path pruned: the kernel switches parsed once, item 32). The K3-only core / plugin split is the current work (D10, `plan.md` S1 and S2).
- S1 step 3 (A4's plan-cache experiment, `FLAGOS_SPACEMIT_PLAN=none|structure|uid`) ran on the K3 2026-10-31 (item 33; `structural-experiments.md` run 11): both candidates hit every generation graph with identical logits, but neither passes the rule, since the per-node contract check (about 0.5 ms of a 4B token) runs on every graph under both and the expansion the cache replaces is about 0.1 ms; S's compare costs 0.8-1.0 ms on the X100.
- S1 step 3b (`FLAGOS_SPACEMIT_PLAN=none|uid|uid-trust`, candidate S removed) ran on the K3 2026-11-01 (item 34; run 12): the trusted-uid hit (`uid-trust`: no contract check on a hit, CUDA's and Hexagon's form and the deck's "redone only when the structure changes") hits every generation graph with identical logits and lowers the host work per generated token from 630 to 46 µs on 4B and from 446 to 39 on 0.6B; the checked form `uid` saves nothing. Adopted by the rule; A4 closed.
- S1 step 3c (`plan.md` "S1 step 3c spec"; item 35; run 13) accepted on the K3 2026-11-01: `uid-trust` the default, the switch test-only as `none|uid-trust`, the checked form removed; without counters the full path `none` is 0.6% below the default on tg128 on 4B and 2.3% on 0.6B (estimated +0.5% and +2%), pp128 within spread; the acceptance runner 39/39 and 40/40 with the logits of 2026-10-27 to the last digit. Found on the way: the runner's `-fa auto` attention check for the perplexity run had been skipped since 2026-10-29, now fixed; the CPU-only depth benchmark died twice at question 8's instruction, with registers (`build.md` §7).
- Next: steps 4 and 5 (`plan.md` "S1 design", exit of the note: the workspace and capture driver as the core's, then the diagnostics under the deck's names); then S2 (the plugin as its own library, the zero-copy entries); then M3 and fusion (D9, deferred).
- Also pending: the summary of runs 6-10 for the mentor and SpacemiT in the agreed form; the mentor's answers to §7.
