# FlagOS × SpacemiT K3 — project context

Loaded at every Claude Code session. Details live in sibling files; read them when the task touches their topic:

- `docs/flagos-k3/build.md` — step-by-step builds (Mac, K3), verification, benchmarks, troubleshooting
- `docs/flagos-k3/k3-hardware.md` — board facts, AI cores, TCM investigation and its evidence
- `docs/flagos-k3/upstream-review.md` — structural concerns in ggml-flagos (fusion + provider shim), with file:line evidence
- `docs/flagos-k3/plan.md` — build plan for the SpacemiT provider: architecture, decisions D3–D9, milestones M0–M4 with tests and exit criteria, risks
- `docs/flagos-k3/device-type.md` — GPU vs IGPU vs ACCEL: what the type controls in llama.cpp, what "right behavior" means (R1–R6), predictions and experiments X0–X6

## 1. Task

- Goal: run llama.cpp inference on the SpacemiT K3 through FlagOS — a **SpacemiT provider inside ggml-flagos**, analogous to how vllm-plugin-FL plugs FlagOS into vLLM.
- Deliverable shape: in-tree provider at `ggml/src/ggml-flagos/providers/spacemit/` (M1 skeleton done). ggml-flagos builds as one ggml backend (`libggml-flagos`); providers are compiled into it.
- Mentor also asked for a design review of ggml-flagos, focused on **kernel fusion** (providers have different kernel structures) and the **shim between Common and providers** → `upstream-review.md`.
- Status: FlagOS provider-neutral build and upstream SpacemiT IME baseline both build on the K3; IME + TCM passes on the board (2026-10-07). Build plan written (`plan.md`, 2026-10-07). M1 skeleton done 2026-10-08 (`providers/spacemit/`, commit `4389005`; 14/14 checks on the K3). M2a done 2026-10-09 (spine-runtime executor + `ADD`, commits `f78022f`, `187239e`; `test-backend-ops -o ADD` passes on the K3). M2b done 2026-10-09 (Q4_0 `MUL_MAT` on the IME, commit `6671c59`): accurate (perplexity -0.13% vs CPU on Qwen3-4B), pp128 2.4x the X100 cores, tg128 about equal to them because of 355 CPU/AI-core handoffs per token (`build.md` §7). M2d done 2026-10-10 (the rest of a layer except attention, commits `f1cf809`, `7a2b897`, `e5f69ff`): tg128 7.16, pp128 53.0, as accurate as the IME path (`build.md` §7); the Q4_1 `ffn_down` matmuls and attention on the X100 cores are what remains.

## 2. Decisions (proposed = still to confirm with mentor)

| ID | Decision | Status |
|---|---|---|
| D1 | Standalone backend design (ggml device/buffer/backend), not a ggml-cpu extra buffer type | adopted via ggml-flagos |
| D2 | Work in the mentor's fork, in-tree; the earlier standalone `ggml-plugin-FL` repo is retired | adopted |
| D3 | ggml device type `ACCEL` | **adopted** (mentor, 2026-10-08). Consequences: the provider must read CPU buffers directly (the KV cache stays on the CPU) and ship correct non-flash attention (`-fa auto` turns flash attention off when it runs attention). Evidence: `device-type.md` §5, `plan.md` §2.8 |
| D4 | Weights in an IME-repacked buffer, stored once (`is_host = false`); the provider reads CPU buffers for all other operands | **building on it from M2b** (user, 2026-10-09); mentor confirmation pending (`plan.md` §2.3) |
| D5 | spine-runtime executor, one launch per split, barrier per step (an op may take several steps) | proposed; implemented in M2a, persistent stream chosen by measurement (`plan.md` §2.4) |
| D6 | Copy into the provider, per milestone, only the IME kernel, quantizer and repack functions it uses from in-tree `ggml-cpu/spacemit` (bodies unchanged, origin noted); port ggml-spacemit tiling and RVV ops, fixing them (the mentor allows changing that AI-generated code, 2026-10-08) | **decided** (user, 2026-10-09); was "compile in place" (`plan.md` §2.5) |
| D7 | First target Qwen3-0.6B / 4B Q4_0, then Qwen3.5-4B Q4_K_M | proposed |
| D8 | `flagos_provider_kind::cpu_accelerator` (engine `cpu`); `ai_accelerator` would classify the K3 as an NPU. Independent of D3; affects no execution | proposed, open |
| D9 | FlagTree AOT package (M3) only if in scope | proposed |

Caveat: ggml's scheduler runs splits sequentially (`ggml_backend_sched_compute_splits`); do not assume the X100 CPU backend and A100 provider run concurrently.

## 3. Code and references

| What | Where | Notes |
|---|---|---|
| This repo | fork of `kevinzs2048/llama.cpp`, branch `feature/flagos-spacemit` | = mentor's `feature/flagos-amd-890-opt` @ `5794e12` + macOS link fix `eaa25ff` |
| Mentor baseline | `kevinzs2048/llama.cpp` `feature/flagos-amd-890-opt` @ `5794e12` | exactly the design PDF's baseline; upstream base `ba360ef` (2026-08-11) |
| Mentor design history | same repo, branch `feature/flagos-multi-provider-backend` | 4 `FLAGOS_BACKEND_*.md` design/review docs not on the AMD branch; §14 is the K3 provider design (co-built in `providers/spacemit`, no parallel ggml-spacemit; it chose K3 = GPU, which the mentor now considers open) |
| FlagOS Common | `ggml/src/ggml-flagos/flagos-{provider,registry,target,graph-plan}.*` | registry, profiles, fixed-fusion side-plan |
| Provider templates | `providers/denglin/flagos-denglin.cpp` (read first, 3.2k lines), `providers/amd/` (strict AOT loader, many fusions) | both report GPU/IGPU device types |
| Upstream SpacemiT CPU path | `ggml/src/ggml-cpu/spacemit/` (`ime.cpp`, `ime_env.cpp`, `spine_mem_pool.cpp`, vendored `spine_tcm.h`) | extra-buffer-type design; A100 binding, TCM, IME2 kernels |
| SpacemiT standalone backend | `spacemit-com/llama.cpp` branch `agent/ggml-spacemit-backend` @ `4e782bc`, `ggml/src/ggml-spacemit/` | ACCEL backend on spine-runtime; own repacking buffer type; ~35 ops. Primary K3-side reference. Copies the in-tree kernel/repack files unchanged; branch `mtmd-backend` @ `64316cd` (2026-09-24) has the same kernel, IME and repack code (checked 2026-10-09). Measured 2026-10-08 (`build.md` §7): the upstream IME path, not ggml-spacemit, is the fastest generation baseline (Qwen3-4B Q4_0 tg128 11.10 vs 6.82 t/s) |
| spine-runtime | `spacemit-com/spine-runtime` release 0.6.0 (`libspert.so`, `spert.hpp`) | `spert::Stream::launch(Grid, fn)`, `Context::program_id/grid_dim/sync/shared_buffer`, `backend_info()`; does the `/proc/set_ai_thread` opt-in itself |
| FlagTree SpacemiT backend | `flagos-ai/FlagTree` `third_party/spacemit/` | Triton → linalg → spine-mlir → riscv64 `.so`; `AICPUTarget(...)`; launcher ABI in `backend/driver.py` |
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

Before any code change:
1. Prefer the **smallest change** that achieves the goal; avoid touching upstream/mentor files unless required.
2. State briefly **why the chosen approach is optimal**, list **realistic alternatives**, and why they lose. Then make the change.
3. Verify facts in the code (file:line) instead of assuming; say when something is unverified.

While changing code:
- No redundant prints. Use `GGML_LOG_DEBUG` / `GGML_LOG_WARN` / `GGML_LOG_ERROR` for diagnostics worth keeping; remove ad-hoc `printf`s before finishing.
- Keep terminal output focused: pipe long builds to a log and grep for `error:|FAILED`.
- Never weaken `supports_op`: return true only for exactly what the kernel handles (see `upstream-review.md`).
- Do not change system configuration on the shared board (cgroups, `/proc`, drivers). Reading is fine.

After any code change, always report:
1. **What changed** (files, functions, and why).
2. **How to build, run and test it** (exact commands for this board).
3. **Ways to play with it**: which parameters/env vars/shapes to change and what result to expect.

Git (from AGENTS.md, applies to anything that may go upstream):
- Do not commit or push without explicit approval each time. Commit messages for this fork carry no AI attribution line (no `Assisted-by:`, no `Co-authored-by:`; user decision 2026-10-08).
- Push only to `origin` (the user's fork). Never push to the mentor's repo.
- Never write the board's SSH hostname or credentials into the repo (the fork is public).

## 7. Open questions for the mentor

Same numbering as `plan.md` §7, which gives the context for each.

1. ~~Device type~~ decided: ACCEL (2026-10-08). Still open: FlagOS `caps.kind`, `cpu_accelerator` (proposed) or `ai_accelerator`?
2. Memory under ACCEL: weights in an IME-repacked buffer, stored once, provider reading CPU buffers for everything else (`plan.md` §2.3, A2)? We are building M2b on A2 (2026-10-09); the alternative doubles quantized-weight memory and would change only the buffer part.
3. Is closed `libspert` (spine-runtime) acceptable as a hard dependency?
4. Kernel reuse: decided 2026-10-09 to copy, per milestone, only the functions used (`plan.md` §2.5: mentor design §14.1 "reference, not boundary"; ggml-spacemit did the same). Objections?
5. Phase 0 target: Qwen3-4B Q4_0 or Qwen3.5-4B Q4_K_M?
6. FlagTree AOT (M3) in scope? Cross-compile on x86 or build on the K3? (FlagTree's SpacemiT backend has no AOT tool; we would write generator, manifest, loader, launcher.)
7. Is co-building in `providers/spacemit` agreed with SpacemiT, who ship ggml-spacemit as an `ACCEL` backend?
8. TCM: report to SpacemiT (no dead-owner recovery; `try_wait` timeout unit ~12 µs) and install spacemit-tcm 3.0.1 on the board? Also one CPU-backend crash in 8 runs whose register state the program cannot produce (`build.md` §7; not reproduced by targeted tests).
9. ACCEL gaps in llama.cpp (C5, `plan.md` §6): fix the automatic flash-attention check and norm pinning in our fork, or raise an upstream issue? Norm pinning is now patched in the fork as a separate commit (`f1cf809`, 2026-10-09, needed for M2d); keep it? Without them ACCEL needs `-fa on` and generates 30-39% slower (Qwen3-4B / 0.6B).
10. Order after M2d: M2c, then M2e? Moving the Q4_1 matmuls alone takes pp128 from 53 to 83 t/s; attention on the X100 cores costs +445 ms per token at a context of 4096 (`plan.md` §7).

Zero-copy between same-memory devices (former question 6) is handled at the ggml level by host-visible buffers (`plan.md` §2.3); `memory_domain_id` stays unused.

## 8. Next steps (in order)

Full milestone list with tests and exit criteria: `plan.md` §4. Current milestone: M2c (proposed next, question 10; M2d done 2026-10-10).

1. ~~TCM isolation~~ done 2026-10-07: TCM works with the IME path (`k3-hardware.md` §4).
2. M0.1–M0.2: build and run `ggml-spacemit` with spine-runtime (`build.md` §5), with `scripts/spacemit-device-type.patch` applied; `scripts/spert-info.cpp` shows whether `shared_buffer()` is real TCM.
3. M0.9 (E1): `scripts/e1-device-type.sh`. Done on Qwen3-0.6B (`device-type.md` §5); D3 decided (ACCEL). X3/X3b done 2026-10-08: ACCEL with CPU-buffer reads, `-fa on` and llama.cpp's norm pinning off matches GPU type within 1-2% (Qwen3-0.6B and Qwen3-4B, `device-type.md` §5); the two llama.cpp heuristics are change C5 (`plan.md` §6), a question for the mentor.
4. ~~M0.3–M0.6~~ done 2026-10-08 on Qwen3-4B Q4_0 (`build.md` §7): provider targets tg128 >= 11.1 (upstream IME) and pp128 >= 82 (ggml-spacemit fixed); all modes within 0.3% perplexity.
5. ~~M0.4: re-measure A100 VLEN~~ done 2026-10-07: 1024 on A100, 256 on X100 (`k3-hardware.md` §2). ~~M0.2 TCM check~~ done: spine-runtime gets real TCM.
6. ~~M0.7~~ done 2026-10-08: op matrix of Qwen3-4B Q4_0 in `plan.md` (9 op kinds with flash attention; output head is Q6_K, 4 `ffn_down` are Q4_1).
7. M0.8: settle D4–D9 and C5 with the mentor (questions above; D3 settled).
8. ~~M1~~ done 2026-10-08: `providers/spacemit/` skeleton, 14/14 checks on the K3 (`plan.md` M1).
9. ~~M2a~~ done 2026-10-09: spine-runtime executor + `ADD` (`plan.md` M2a); persistent stream chosen by measurement.
10. ~~M2b~~ done 2026-10-09: Q4_0 `MUL_MAT` on the IME (A2 buffer, copied kernels, GEMV + path A + path C); results in `plan.md` "M2b design" and `build.md` §7. Next: M2d before M2c (decided 2026-10-09; removing the handoffs is worth more for generation than moving the output head). M2d implemented 2026-10-09 (`plan.md` "M2d design"): the rest of a layer except attention, with ggml-spacemit's RVV kernels ported and ggml-cpu-style references; the norm-pinning patch C5b (approved by the user; mentor to confirm, question 9); a CMake host test device for the Mac. All claimed cases pass on the Mac's simulated device.
11. ~~M2d~~ done 2026-10-10 on the K3 (`build.md` §7). Next, pending the mentor (question 10): M2c, the Q4_1 matmuls first (pp128 53 -> about 83, measured with a requantized model), then the Q6_K output head; then M2e (attention; decisive beyond about 1k tokens of context).
