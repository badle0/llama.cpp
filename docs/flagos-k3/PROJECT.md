# FlagOS × SpacemiT K3 — project context

Loaded at every Claude Code session. Details live in sibling files; read them when the task touches their topic:

- `docs/flagos-k3/build.md` — step-by-step builds (Mac, K3), verification, benchmarks, troubleshooting
- `docs/flagos-k3/k3-hardware.md` — board facts, AI cores, TCM investigation and its evidence
- `docs/flagos-k3/upstream-review.md` — structural concerns in ggml-flagos (fusion + provider shim), with file:line evidence

## 1. Task

- Goal: run llama.cpp inference on the SpacemiT K3 through FlagOS — a **SpacemiT provider inside ggml-flagos**, analogous to how vllm-plugin-FL plugs FlagOS into vLLM.
- Deliverable shape: in-tree provider at `ggml/src/ggml-flagos/providers/spacemit/` (not yet created). ggml-flagos builds as one ggml backend (`libggml-flagos`); providers are compiled into it.
- Mentor also asked for a design review of ggml-flagos, focused on **kernel fusion** (providers have different kernel structures) and the **shim between Common and providers** → `upstream-review.md`.
- Status: FlagOS provider-neutral build and upstream SpacemiT IME baseline both build on the K3; no provider code written yet.

## 2. Decisions (proposed = still to confirm with mentor)

| ID | Decision | Status |
|---|---|---|
| D1 | Standalone backend design (ggml device/buffer/backend), not a ggml-cpu extra buffer type | adopted via ggml-flagos |
| D2 | Work in the mentor's fork, in-tree; the earlier standalone `ggml-plugin-FL` repo is retired | adopted |
| D3 | ggml device type `ACCEL`, FlagOS `flagos_provider_kind::cpu_accelerator` | proposed; SpacemiT's own backend is also ACCEL |

Caveat on D3: ggml's scheduler runs splits sequentially (`ggml_backend_sched_compute_splits`); do not assume the X100 CPU backend and A100 provider run concurrently.

## 3. Code and references

| What | Where | Notes |
|---|---|---|
| This repo | fork of `kevinzs2048/llama.cpp`, branch `feature/flagos-spacemit` | = mentor's `feature/flagos-amd-890-opt` @ `5794e12` + macOS link fix `eaa25ff` |
| Mentor baseline | `kevinzs2048/llama.cpp` `feature/flagos-amd-890-opt` @ `5794e12` | exactly the design PDF's baseline; upstream base `ba360ef` (2026-08-11) |
| Mentor design history | same repo, branch `feature/flagos-multi-provider-backend` | 4 `FLAGOS_BACKEND_*.md` design/review docs not on the AMD branch |
| FlagOS Common | `ggml/src/ggml-flagos/flagos-{provider,registry,target,graph-plan}.*` | registry, profiles, fixed-fusion side-plan |
| Provider templates | `providers/denglin/flagos-denglin.cpp` (read first, 3.2k lines), `providers/amd/` (strict AOT loader, many fusions) | both report GPU/IGPU device types |
| Upstream SpacemiT CPU path | `ggml/src/ggml-cpu/spacemit/` (`ime.cpp`, `ime_env.cpp`, `spine_mem_pool.cpp`, vendored `spine_tcm.h`) | extra-buffer-type design; A100 binding, TCM, IME2 kernels |
| SpacemiT standalone backend | `spacemit-com/llama.cpp` branch `agent/ggml-spacemit-backend` @ `4e782bc`, `ggml/src/ggml-spacemit/` | ACCEL backend on spine-runtime; own repacking buffer type; ~35 ops. Primary K3-side reference and the performance baseline to beat |
| spine-runtime | `spacemit-com/spine-runtime` release 0.6.0 (`libspert.so`, `spert.hpp`) | `spert::Stream::launch(Grid, fn)`, `Context::program_id/grid_dim/sync/shared_buffer`, `backend_info()`; does the `/proc/set_ai_thread` opt-in itself |
| FlagTree SpacemiT backend | `flagos-ai/FlagTree` `third_party/spacemit/` | Triton → linalg → spine-mlir → riscv64 `.so`; `AICPUTarget(...)`; launcher ABI in `backend/driver.py` |
| Design doc | `ggml-flagos-provider-design.pdf` V2.0 (AI-written summary of the AMD work) | treat claims as needing code verification |

## 4. K3 essentials (full detail in `k3-hardware.md`)

- Bianbu 4.0, kernel 6.18.3, riscv64; GCC 15.2 (assembles IME1+IME2), CMake 4.2, Ninja; 31 GiB RAM, no swap.
- CPUs 0–7: X100 @ 2.4 GHz (default affinity). CPUs 8–15: A100 AI cores @ 2.0 GHz, usable only after a thread writes `/proc/set_ai_thread`.
- RVV VLEN 256 on both core types; identical ISA strings. A100 extras: IME matrix extension, per-core TCM (8 × 384 KiB via `/dev/tcm`).
- Known issue: upstream IME path aborts in TCM acquisition (`ime.cpp:1728`); runs with `SPACEMIT_DISABLE_TCM=1`. Root cause open.

## 5. Everyday commands (on the K3, repo root)

```bash
tmux attach -t build || tmux new -s build                  # long jobs always inside tmux
cmake --build build --target ggml-flagos flagos-check-provider flagos-check-target flagos-check-graph-plan test-backend-ops llama-bench llama-cli
for t in provider target graph-plan; do ./build/bin/flagos-check-$t; done
SPACEMIT_DISABLE_TCM=1 ./build-ime/bin/test-backend-ops -o MUL_MAT -b CPU 2>&1 | tail -3
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
- Do not commit or push without explicit approval each time. If asked to commit, add `Assisted-by: Claude Code`; never `Co-authored-by:`.
- Push only to `origin` (the user's fork). Never push to the mentor's repo.
- Never write the board's SSH hostname or credentials into the repo (the fork is public).

## 7. Open questions for the mentor

1. Scope: reuse `ggml-spacemit`'s runtime/buffer layer (spine-runtime launch, repacked buffers) and contribute FlagTree kernels + FlagOS integration, or write an independent provider?
2. Confirm D3 (ACCEL + `cpu_accelerator`) vs GPU/IGPU-style like AMD.
3. TCM on Bianbu 4.0: no `/dev/tcm_sync_mem`; `spine_tcm_mem_try_wait` fails in llama.cpp's IME path. Known? Newer image/driver?
4. AOT path for FlagTree SpacemiT kernels (riscv64 `.so` + stable C signature) usable from C++ without Python?
5. P0 target model and quant (AMD work used Qwen3.5-4B-Q4_K_M).
6. Is zero-copy between same-memory-domain devices (`memory_domain_id`) in scope for FlagOS Common?

## 8. Next steps (in order)

1. TCM isolation: `diff -w` of `spine_tcm.h` headers, then the standalone `libspine_tcm` test (`k3-hardware.md` §4).
2. Build and run `ggml-spacemit` with spine-runtime 0.6.0 (`build.md` §5) — does its TCM path work here?
3. Put a model on the board; first `llama-bench` numbers: plain CPU vs upstream IME vs ggml-spacemit (`build.md` §7).
4. Settle scope with the mentor (questions 1–2), then create `providers/spacemit/` (M1: device registers, claims nothing).
