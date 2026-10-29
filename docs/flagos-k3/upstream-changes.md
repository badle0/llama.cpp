# Changes to upstream and mentor code

Every change this fork makes outside `ggml/src/ggml-flagos/providers/spacemit/` and `docs/flagos-k3/`: to upstream llama.cpp or ggml files, or to the mentor's FlagOS code. The baseline is the mentor's `feature/flagos-amd-890-opt` at `5794e12`, where this fork starts.

Rule: a change outside those two directories gets an entry here in the same commit, ideally in a commit of its own. Removing a change removes its entry.

Check the list against the tree:

```bash
git diff --stat 5794e12..HEAD -- . ':(exclude)ggml/src/ggml-flagos/providers/spacemit' ':(exclude)docs/flagos-k3'
```

Last checked 2026-10-28 after U1 and U2 were reverted and M1's function deleted: 7 files, 38 lines added, 8 removed; of these the four code files (U3, M1-M3: `ggml/CMakeLists.txt`, `ggml-flagos/CMakeLists.txt`, `flagos-registry.cpp`, `ggml-flagos.h`) 19 added and 8 removed, the three `.claude/` files 19 added (fork-only tooling); `src/llama-context.cpp` is the mentor's byte for byte again, and M1's hunks are his own commit's. No edit was made and later undone: every file touched shows its net change. The mentor's providers (`providers/amd`, `providers/denglin`), Common (`flagos-provider.*`, `flagos-target.*`, `flagos-graph-plan.*`, `ggml-flagos.cpp`) and the FlagOS check programs are untouched.

## Upstream llama.cpp and ggml

### U1. Norms of CPU layers left to the scheduler when an ACCEL backend runs them (C5b): reverted 2026-10-28

- Where: `src/llama-context.cpp`, `llama_context::graph_get_cb()`, lines 2522-2533 (12 lines added).
- Commit: `f1cf809` (2026-10-09), on its own.
- What: for batches under 32 tokens (and with full offload), llama.cpp pins every `norm` and `l_last` node to its layer's device. Under ACCEL's default device selection, a layer not on a GPU has the CPU as its device, so each norm was pinned to the CPU although the provider runs the ops around it. The change returns before pinning when the layer's device is the CPU and an ACCEL backend supports the node.
- Why: whole layers on the AI cores (M2d). Without it ACCEL generates 30-39% slower (Qwen3-4B / 0.6B, X3b, `device-type.md` §5).
- Status: approved by the user 2026-10-09 and by the mentor 2026-10-14 (`PROJECT.md` question 9). With D3 changed to IGPU (2026-10-26) and the provider reporting IGPU since 2026-10-27 (`PROJECT.md` item 22; the K3 run pending), the change is inert for this provider: the layer's device is the provider, so the condition (an ACCEL backend) no longer matches. It stays in the fork, its value for other ACCEL providers unchanged; keep or revert is the mentor's call.
- Reverted 2026-10-28 (user): with the device type IGPU, an ACCEL-only change has no reason to stay in the fork; `src/llama-context.cpp` restored to the mentor's `5794e12` with `git checkout`, the Mac build and `flagos-check-spacemit` unchanged. Consequence: the test-only `FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=accel` mode runs as stock llama.cpp runs an ACCEL device (norms pinned to the CPU, flash attention off under `-fa auto`), so it no longer reproduces 2026-10-26's ACCEL numbers; it remains a device-type comparison, with `-fa on` for speed. The known comment overstatement went with it.

### U2. A fused op that an ACCEL backend runs for a CPU layer is no mismatch (C5a): reverted 2026-10-28

- Where: `src/llama-context.cpp`, `llama_context::resolve_fused_ops()`, lines 533-541 (9 lines added, 1 changed).
- Commit: `64e6cbd` (2026-10-13), on its own, with the buffer-type check.
- What: under `-fa auto`, llama.cpp turns a fused op (flash attention) off when the device the scheduler gave it differs from its layer's device. With ACCEL the layer is on the CPU and the op on the ACCEL device, so flash attention was turned off. The change treats that case as no mismatch when the ACCEL device supports the layer's buffer type (the KV cache's), that is, reads those tensors in place.
- Why: flash attention on the provider under default flags (M2e.1): 2 graph splits per token instead of 74. Without it `-fa auto` would need M2e.2 (the graph without flash attention on the provider), or every run `-fa on`.
- Status: decided by the user 2026-10-12, as a commit of its own; approved by the mentor 2026-10-14 (question 9). Inert for this provider since it reports IGPU (D3; in the code 2026-10-27, accepted on the K3 the same day). Reverted 2026-10-28 (user) with U1, in the same `git checkout`; the mentor hears of both reverts with question 9's update.

### U3. Build option `GGML_FLAGOS_SPACEMIT`

- Where: `ggml/CMakeLists.txt:204`, 1 line beside the mentor's `GGML_FLAGOS_*` options.
- Commit: `4389005` (M1, 2026-10-08).
- What: builds the SpacemiT provider; off by default, no effect when off.

## The mentor's FlagOS code (outside `providers/spacemit`)

### M1. `ggml_backend_flagos_reg_devices()` removed (the macOS link fix, now the mentor's own removal)

- Where: `ggml/src/ggml-flagos/flagos-registry.cpp` (the 6-line function deleted) and `ggml/src/ggml-flagos/ggml-flagos.h` (its declaration deleted). From `eaa25ff` (2026-09-24) to 2026-10-28 the fork kept the function with an empty body; since 2026-10-28 (user) it is deleted exactly as the mentor's `f324584c8` deletes it: our diff against `5794e12` for the two files equals his commit's hunks, line for line, the only other lines in that file being M2's two hooks, so the rebase onto his audit branch merges cleanly.
- Commit: `eaa25ff` (2026-09-24), the fix this fork's base adds to `5794e12`; the deletion in the commit of 2026-10-28 with the revert of U1 and U2.
- What: in builds without `GGML_BACKEND_DL` the function called `ggml_backend_register(ggml_backend_flagos_reg())`, which made the backend library depend on libggml, the library that links it: an undefined symbol on macOS. `ggml-backend-reg.cpp` (`GGML_USE_FLAGOS`) or the dynamic loader registers the backend instead.
- Status: to report to the mentor. The function is public (`ggml-flagos.h:20`, `GGML_BACKEND_API`) and now does nothing on every platform. Nothing in this tree calls it; a program outside that relied on it to register the backend would no longer get it. The mentor's newer branch `feature/flagos-amd-890-audit-20261010` (2026-10-10; fetched read-only 2026-10-26) removes the function and its declaration altogether, so this fix converges with the mentor's own change at the next rebase.
- Verified 2026-10-28, both directions, on the Mac (`build-host`, Apple clang, arm64): with the mentor's original body restored, linking `libggml-flagos.dylib` fails with `Undefined symbols for architecture arm64: "_ggml_backend_register", referenced from: _ggml_backend_flagos_reg_devices in flagos-registry.cpp.o`; with the fork's body it links. The mechanism, read in the build files: `ggml_backend_register` lives in `ggml-backend-reg.cpp`, a source of the `ggml` library (`ggml/src/CMakeLists.txt:242-244`), and in a build without `GGML_BACKEND_DL` the `ggml` library links the backend library (`ggml_add_backend_library`, `:279`), so the call made the backend depend on the library that links it. macOS's linker rejects an undefined symbol in a dylib at link time; Linux's defers it to load time, where the executable's `libggml.so` satisfies it, which is why the mentor's build passed on Linux and the K3. So the cause is certain for the link failure; a runtime effect on Linux was never observed and is not claimed. The mentor's own removal: commit `f324584c8` on `kevin/feature/flagos-amd-890-audit-20261010` (2026-10-07, "flagos: use core registration and portable plugin test backends") deletes the function from `flagos-registry.cpp` and its declaration from `ggml-flagos.h`, and changes the two AMD check programs that called it (`tests/check_amd_gdn_value_tile.cpp`, `tests/check_amd_strided_rms.cpp`) and `CMakeLists.txt`; our fix keeps the empty function, so a rebase onto that branch drops it without conflict.

### M2. The SpacemiT provider in the registry

- Where: `ggml/src/ggml-flagos/flagos-registry.cpp:21-23` (declaration) and `:81-83` (`flagos_compiled_providers()`), under `GGML_FLAGOS_HAVE_SPACEMIT`; 7 lines.
- Commit: `4389005` (M1).
- What: compiles the provider into the registry as the Denglin and AMD providers are.

### M3. The provider's build and its check program

- Where: `ggml/src/ggml-flagos/CMakeLists.txt:42-44` (includes `providers/spacemit/provider.cmake`), `:46` (the "no provider" message also checks the option), `:156-161` (target `flagos-check-spacemit`: not built by default, builds without `GGML_BACKEND_DL` only; `:160` includes `providers/spacemit/tests.cmake`).
- Commits: `4389005` (M1); `72e0666` (M2c.2) links the check program with `Threads::Threads`, for the 8 threads that quantize the output head's reference weights; S1 step 1 adds `:160`, one line, so the fake plugins of the loader's negative matrix (`providers/spacemit/tests/plugins/`) are built beside the check program and their directory reaches it as a compile definition. Everything that line adds lives in the provider's own `tests.cmake`.

## Fork-only tooling (new files, no upstream file changed)

- `.claude/CLAUDE.md`: imports `docs/flagos-k3/PROJECT.md` and `docs/flagos-k3/CLAUDE.md`, so Claude Code loads them at every session; `.claude/agents/coder.md`: the `coder` subagent; `.claude/settings.json`: denies Claude Code's edits to the root `CLAUDE.md` and `AGENTS.md`. Upstream has no `.claude/` (2026-10-26). Tooling only, no effect on any build; leave it out of anything proposed upstream.

## Proposed, not made

- The KV cache on huge pages (M2e.3 candidate (c)): would need llama.cpp to allocate the cache differently, or a system setting the shared board rules out. Set aside by the user 2026-10-13.
- The in-tree IME path's tiled attention kernel checks every mask value one at a time (`ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp:1470-1474`), a third of its time (L2, `build.md` §7). Not changed here: for the report to SpacemiT.
- C5b's comment (U1).

## Not changed

- `CLAUDE.md` and `AGENTS.md` (repository root): upstream's, byte for byte, since 2026-10-26 (user). The import of `PROJECT.md` that `CLAUDE.md` carried from `213bbf7` (formerly U4) moved to `.claude/CLAUDE.md`; the user's uncommitted pointer in `AGENTS.md` was removed. Check: `git diff --stat ba360ef -- AGENTS.md CLAUDE.md` prints nothing.
- `ggml/src/ggml-cpu/spacemit/` (the in-tree IME path): copied from, never modified.
- `tests/test-backend-ops.cpp`: untouched; its Qwen3-4B one-token attention case is in the baseline.
