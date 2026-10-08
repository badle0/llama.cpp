# FlagOS SpacemiT K3 provider - Codex guidance

This file supplements the repository-root `AGENTS.md` for FlagOS and SpacemiT K3 work. The root instructions still apply. Paths below are relative to the llama.cpp fork root.

## Read current context first

- Read `docs/flagos-k3/PROJECT.md` at the start of each K3 task. It is the source of truth for current status, decisions, open questions, and next steps. This file is a working guide, not a second status ledger.
- Read `docs/flagos-k3/build.md` for build, test, benchmark, and troubleshooting commands; `docs/flagos-k3/k3-hardware.md` for measured board facts and the TCM investigation; and `docs/flagos-k3/upstream-review.md` when changing Common, fusion, provider registration, or the provider shim.
- Check the current branch, `git status`, recent commits, and relevant source before relying on a dated statement in any document. Mark unverified claims as such. If code, board evidence, and docs disagree, resolve the discrepancy and update the affected document.

## Goal and current baseline

- Deliver llama.cpp inference on SpacemiT K3 through a provider compiled into `libggml-flagos` at `ggml/src/ggml-flagos/providers/spacemit/`. The old standalone `ggml-plugin-FL` scaffold is retired.
- The active fork is based on the mentor's `feature/flagos-amd-890-opt` branch at `5794e12`; K3 work is on `feature/flagos-spacemit`. Keep the mentor fork as a read-only reference and work in the user's fork.
- As recorded on 2026-09-24, the provider-neutral FlagOS build and upstream SpacemiT IME baseline built on the K3. No SpacemiT FlagOS provider code had been written. The separate SpacemiT `ggml-spacemit` backend had not yet been run on this board. Recheck `PROJECT.md` and the tree for newer progress.
- The mentor requested a design review of kernel fusion and the Common/provider shim. Findings and proposed directions are in `upstream-review.md`; do not treat proposed changes as approved design decisions.

## Architecture and decisions

- Adopted: implement a full ggml device/buffer/backend through the in-tree FlagOS provider model, rather than a `ggml-cpu` extra buffer type. `ggml-flagos` is one ggml backend with compiled-in providers.
- Adopted (mentor, 2026-10-08): the K3 is a ggml `ACCEL` device. Its buffers must accept CPU-resident operands (the KV cache stays on the CPU), and its non-flash attention must be correct, because `-fa auto` turns flash attention off when an ACCEL backend runs attention (`device-type.md` §5). Decisions D4-D9 are proposed in `plan.md`.
- FlagOS Common is in `ggml/src/ggml-flagos/flagos-{provider,registry,target,graph-plan}.*`. Read `providers/denglin/` for the smaller provider pattern and `providers/amd/` for AOT and fusion examples. The SpacemiT references are the fork's `ggml/src/ggml-cpu/spacemit/` and `spacemit-com/llama.cpp` branch `agent/ggml-spacemit-backend` at `4e782bc` (`ggml/src/ggml-spacemit/`). That backend is AI-generated and fails many op tests (X0); the mentor allows changing it freely, so port it piece by piece with fixes and tests. FlagTree's SpacemiT backend is a separate AOT route.
- The ggml scheduler runs graph splits sequentially. Do not assume the X100 CPU backend and A100 provider run concurrently.

## K3 constraints

- The measured board runs Bianbu 4.0 on riscv64 with GCC 15.2, 31 GiB RAM, and no swap. CPUs 0-7 are X100 cores; CPUs 8-15 are A100 AI cores. A100 access requires the per-thread `/proc/set_ai_thread` opt-in, normally handled by the relevant runtime. RVV VLEN is 256 bits on X100 and 1024 bits on A100; a thread's `vlenb` changes when it migrates, so read it on the thread that runs the kernel.
- TCM works on this board (resolved 2026-10-07, `k3-hardware.md` §4). Before IME runs, release blocks left by dead processes with `~/tcmtest/tcmrelease --apply`, and use at most 8 compute threads. Results with `SPACEMIT_DISABLE_TCM=1` are IME2 without TCM and without fixed A100 pinning; label them so.
- The board is shared. Do not manually change system-wide configuration, cgroups, `/proc` settings, or drivers. The runtime's per-thread AI-core opt-in is part of normal execution. Reading board state is fine.

## Rules when changing code

Before editing:

1. Inspect the relevant code and verify claims with file and line references. Distinguish observed behavior from assumptions.
2. Prefer the smallest change that achieves the task. Avoid modifying upstream or mentor code outside the needed scope.
3. Briefly explain the chosen approach, realistic alternatives, and why the alternatives are less suitable for this change.

During implementation:

- Preserve exact placement contracts. `supports_op` must return true only for the precise operations, tensor types, shapes, layouts, and buffer conditions the kernel can execute correctly. Do not claim an operation because a related kernel exists.
- Keep diagnostics worth retaining in `GGML_LOG_DEBUG`, `GGML_LOG_WARN`, or `GGML_LOG_ERROR`. Remove temporary `printf` output.
- Keep build output focused: use targeted build targets, capture long output in a log, and inspect `error:` or `FAILED`. Run long K3 jobs in `tmux`.
- Keep FlagOS Common responsible for scheduler-visible placement and copies. A provider should fail a graph after an execution error rather than silently falling back after visible writes; see `upstream-review.md` for the current design and its limits.

After editing, report:

1. Files and functions changed, what changed, and why.
2. Exact K3 commands to build, run, and test the change, plus results actually observed. Say clearly when hardware testing was unavailable.
3. Useful parameters, environment variables, or shapes to vary and the expected behavior.

When a build result, benchmark, hardware finding, design decision, or milestone changes project status, update the relevant `PROJECT.md`, `build.md`, `k3-hardware.md`, or `upstream-review.md` entry with date and evidence. Preserve the distinction between adopted and proposed decisions.

## Git and public-repository hygiene

- Follow the root `AGENTS.md` restrictions. Do not commit or push without explicit approval for each action. If asked to commit, use `Assisted-by: Codex`; never use `Co-authored-by:`.
- Push only to `origin`, the user's fork, when explicitly authorized. Never push to the mentor's repository.
- Do not write the board's SSH hostname or credentials into this public repository.

## Immediate work and unresolved choices

Follow the current ordering in `PROJECT.md` and the milestones in `plan.md` §4. As of 2026-10-07 the current milestone is M0: build and run SpacemiT's standalone backend, baselines and perplexity on a real model, A100 VLEN re-check, op inventory, then settle D3-D9 with the mentor before the M1 skeleton.

Questions still requiring mentor input include reuse of SpacemiT's runtime/buffer layer versus an independent implementation, the device-kind choice, TCM compatibility, the FlagTree AOT C++ launch path, the first model/quantization target, and whether same-memory-domain zero-copy belongs in FlagOS Common. Do not present these as settled.
