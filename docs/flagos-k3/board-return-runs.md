# When the board is back: the runs in detail

Written 2026-11-25. `board-return-plan.md` holds the targets, the decisions and the steps in brief; this file says what each paste on the board runs, what it must print, how long it should take and what each result decides. Times are estimates from earlier logs unless a figure is quoted as measured.

## 1. The rounds at a glance

| Round | Plan step | What it does | Board time | Result folder |
|---|---|---|---|---|
| A | 0 | status check, read-only | 1 minute | none, the output is pasted back |
| B | 1 | pull, build at HEAD, check program, our full table, text, KLD | about 40 minutes | `~/cleanup-confirm` |
| C | 2 | the fork's build if it is gone, the device check, both backends in turn three times, the fork's text and KLD, then the fork's profiling build | 1 to 2 hours | `~/fork-overall` |
| D | 3 | isolated op timings on both backends, then the whole model op by op on both | about 1 hour | `~/fork-per-op` |
| E | 5 | the chat run without checkpoints, only if `~/ck` is gone | 16 minutes | `~/chat-no-checkpoints` |

Four pastes for steps 0 to 3. Each block refuses to start when its folder exists, when a process holds `/dev/tcm`, or when a build or a model run is still going; a second try uses the folder name with `-retry`.

## 2. Before the board is back (on the Mac)

- Commit this file. Make the bundle of the commits the board lacks: `09c711eea..feature/flagos-spacemit` (the board's checkout was at `09c711eea` when it left; its build tree was built at `d5e9ae0e8`).
- Have `~/smt-b847.tar.gz` (the fork's source at `b847bf396`, 39,175,813 bytes) ready to copy if round A says it is gone.
- To build before round D: `docs/flagos-k3/scripts/fork-trace.py` and the comparison script (section 6). They are written against the trace format read from the fork's `ggml/src/ggml-profile.c` and tested on the Mac with a made-up trace; the real file is first seen on the board.

## 3. Round A: the status check

Prints: the date and kernel, free space, which of our files exist, where `~/spine-runtime` points (its folder name carries the version), the repo's commit and build folder, the fork's build, any process holding `/dev/tcm`.

| It shows | Then |
|---|---|
| everything present, the repo at `09c711eea`, spine-runtime 0.6.3 | round B as written |
| the repo gone | a bundle of the whole branch and a configure step from `build.md` section 5 before round B |
| the model, `~/q36-kld/base.bin`, `~/ppl.txt` or `~/segv.so` gone | restore from the Mac where a copy exists (`segv.so` is built from `scripts/segv.c`); the KLD base is remade on the board with the CPU, one thread (462.8 s a pass when it was made on 2026-11-17) |
| `~/f32/gen.out` gone | copy it back from `~/k3-logs/f32` on the Mac |
| a newer spine-runtime | round B's op tables give the launch and barrier cost again (13 to 15 us and 0.4 us before); both builds are remade against the new version |
| the date is right | `build-dates.sh` is no longer needed and leaves the blocks |
| the fork's build gone | round C unpacks and builds it again |

## 4. Round B: our build and the cleanup's confirmation

In order, each stage only if the one before it passed:

1. Pull the bundle (fast-forward only), print the commit.
2. `build-dates.sh check`, and `fix d5e9ae0e8` if dates are ahead of the clock. Build with the runner's build step. Must hold: build rc 0, and the build log shows the changed files compiled (`flagos-spacemit-weights.cpp`, `-ime-id.cpp`, `-gdn.cpp`, `-attn.cpp`, `check_spacemit.cpp`).
3. The check program with the device (`--expect-device`, under the crash reporter). Must print "all checks passed". This is the first run of the cleanup `fd757088f` on the real kernels; it covers what the Mac could not: the tiled attention at 9 to 63 tokens and the Gated DeltaNet's forms on RVV, and the conv pair against the rounding bound.
4. The full table, one run of three repetitions: pp32, pp128, pp501, pp2048, tg512, with the op tables. Expected from `e92642893`: 59.85, 82.55, 92.04, 84.50 and 9.56 t/s.
5. One completion (50 sentences, 32 tokens, seed 1) compared byte for byte with the float run's text.
6. The KLD of 2 chunks against the CPU base. Expected: mean 0.005115, maximum 0.089103, same top 94.510%.

Decides: the cleanup stands if 3 passes, 5 gives the float run's text, 6 gives the same three figures, and 4 is within 2% of the expected column. A figure outside 2% gets two more runs before it is believed. A failed check names its line; nothing after it runs.

Also read from this round: the barrier and launch cost per graph in the op tables, to see whether the upgrade changed the runtime's costs.

## 5. Round C: the fork against ours, overall

Setup, in order:

1. The fork's build: reuse `~/llama.cpp-smt-b847/build` if its four programs exist, else unpack `~/smt-b847.tar.gz` with `tar -xzmf`, touch every file, configure and build (`build.md` section 5 flags; 173 steps; the one earlier build's duration was not recorded, allow 45 minutes).
2. The device check, the stop condition of the whole round: with `GGML_SPACEMIT_BACKEND=1`, `test-backend-ops perf -o MUL_MAT -b SPACEMIT0 -p type_a=q4_0` must print `us/run` rows.
3. The offload check: one short completion with `-dev SPACEMIT0 -ngl 99 -lm none -lv 4`. The fork never prints an "offloaded N/N layers" line (llama.cpp prints it only for GPU-type devices, and the fork's is of type ACCEL), so the check reads the buffer lines: `SPACEMIT model buffer size` must hold the weights (18,918 MiB for this model; the CPU keeps 273 MiB, the token embedding) and `SPACEMIT KV buffer size` must be there. If the weights are not in the device's buffer, stop: the figures would again be its CPU's.

Measurements, the two backends in turn, three rounds, nothing else running:

| What | Ours | The fork |
|---|---|---|
| program | `build-flagos/bin/llama-bench` | `~/llama.cpp-smt-b847/build/bin/llama-bench` |
| flags | `-p 4,32,128,501 -n 64 -r 3 -t 8` | the same, plus `-dev SPACEMIT0 -ngl 99 -lm none` |
| environment | none | `GGML_SPACEMIT_BACKEND=1` |
| also recorded | op tables (`GGML_FLAGOS_OP_STATS`) in round 1 only | graph splits from the offload check |

Then on the fork: the same completion as round B (text against the float run's, and its prompt and per-token times), and the KLD of 2 chunks against our CPU base. The fork's CPU fallback gave 0.004140 against that base, so a figure near 0.004 to 0.005 is the expected order.

Last, with nothing being measured: the fork's second build with `-DGGML_BUILD_PROFILE=ON` in `build-profile` (target `llama-bench` only), for round D.

Decides, per size (pp4, pp32, pp128, pp501, tg64): "the fork is faster", "ours is faster" or "inside the spread", where the spread is the larger of the two backends' range over the three rounds. The op-by-op round runs whatever comes out.

What can go wrong, and what the block then does:

- The fork cannot run this model on its device (an op it does not take, an exception, a crash). The block keeps the error text and stops the fork's side; round D's isolated op timings still run, and the whole-model part is repeated on a model the fork can run.
- Its load takes long or its memory use is high (it repacks into its own buffers). The block prints the load time and the peak memory of the first run.
- The fork prints its own tuning variables (`GGML_SPACEMIT_WORKERS`, `SPACEMIT_PERFER_CORE_ID`, `SPACEMIT_MEM_BACKEND`). They stay unset: its defaults are what a user gets.

## 6. Round D: the fork against ours, op by op

Two views, because neither is enough alone: the isolated timings are the same test on both backends but not at this model's shapes; the whole-model tables are this model's shapes but come from two different profilers.

**D1. Isolated ops (`test-backend-ops perf`), both backends in turn, three rounds.**

The two trees share these timing cases (read from both sources on 2026-11-25):

| Op | Shared cases | Filter |
|---|---|---|
| dense Q4_0 matmul | 4096 x 14336 at 1, 2, 3, 4, 5, 8, 512 rows | `-o MUL_MAT -p type_a=q4_0` |
| expert multiply | 128 experts, 8 used, 768 to 2048, at 1, 4, 8, 32, 64, 128, 256, 512 tokens; 32 experts, 4 used, 1792 to 2048, same sizes | `-o MUL_MAT_ID -p type_a=q4_0` |
| attention | head size 64, 8 heads, depth 7680, at 1, 4 and 512 tokens with F16, Q4_0 and Q8_0 caches; head sizes 64 and 128 at depths 4096, 8192, 16384, one token | `-o FLASH_ATTN_EXT`, a pattern for these depths |
| Gated DeltaNet | 32 heads x 128 at 1, 64, 256, 512, 1024 tokens; 4 heads x 128 at 64 to 1024 | `-o GATED_DELTA_NET` |

Devices: `FlagOS:SpacemiT:0` for ours, `SPACEMIT0` for the fork. The fork has more attention cases than ours (depths to 65,536, head sizes to 576); the pattern keeps the run to the shared ones. A row counts only if both backends print it with the same label; the comparison script lists the rest apart. None of these cases has this model's widths (2048 and 512), which is why D2 exists.

**D2. The whole model, op by op.**

- Ours: `GGML_FLAGOS_OP_STATS` with `llama-bench`, as in every run so far. Per graph size it gives a row per step kind with its wall time.
- The fork: its profiling build with `GGML_TRACE=1`. It writes `ggml_trace.json` in the working folder: a begin and an end event for every graph node on every worker, with the op, the output's and operands' shapes, types and names. Each run starts in a folder of its own so the files do not overwrite each other.
- Sizes: 1 token (`-p 0 -n 6 -r 1`), then 4, 32, 128 and 501 tokens (`-n 0 -r 4`), one size a run. The last graph of a run is not used: each worker keeps its events in a 64 KB buffer of its own that is written out when full, and only the main thread's is written at exit (`ggml-profile.c`), so the end of the last graph can be missing.
- `fork-trace.py` turns a trace into the same kind of table as ours: per graph size, a row per op and weight (for example `MUL_MAT attn_qkv q4_0 m501`), the wall time from the first worker's begin to the last worker's end, and the time between one node's end and the next one's begin (the fork's wait between nodes). The first graph of each size is dropped as the warm-up.

The profiler's own cost is measured, not assumed: for each size the profiling build runs once with `GGML_TRACE` unset and once with it set, and the plain build's figure from round C stands beside them. The model's graph has 3,727 nodes by ggml's count (the fork's device may run fewer as its own); each event formats up to 1 KB of text. My estimate is that this is small against a 5-second graph and large against a 100 ms one, so the one-token table is reported with the measured cost per event taken off, and the isolated one-row cases of D1 are the firmer evidence there. A trace is about 20 MB per graph by estimate; the traces are compressed on the board and copied back, and the tables are made on the Mac. A begin event takes its timestamp after its text is formatted and an end event before, so the formatting falls between two nodes and not inside one (read in `ggml-profile.c`).

**The comparison table.** One row per family, for 1, 4, 32, 128 and 501 tokens: ours in ms, the fork's in ms, the ratio.

| Family | Ours (steps summed) | The fork (nodes summed) |
|---|---|---|
| dense quantized matmuls, by weight (`attn_qkv`, `attn_gate`, `ssm_out`, `attn_q`, `attn_output`, `attn_k`, `attn_v`, the shared expert's three, `output`) | `MUL_MAT.quantize` + `MUL_MAT.gemm` | `MUL_MAT` nodes with that weight |
| router | `MUL_MAT.gemm ffn_gate_inp` | `MUL_MAT` on `ffn_gate_inp` |
| expert multiply | `MUL_MAT_ID.quantize` + `MUL_MAT_ID.gemm` | `MUL_MAT_ID` nodes |
| Gated DeltaNet | `GATED_DELTA_NET.*` + `SSM_CONV.*` | the same two ops |
| attention | `FLASH_ATTN_EXT.*` | `FLASH_ATTN_EXT` |
| everything else | the remaining steps | the remaining nodes |
| waiting | barrier + imbalance | time between nodes (waits inside an op are part of the op's time and cannot be split out) |
| the graph | wall | first begin to last end |

Our pairs fused into one step (the conv with its bias and SiLU, for one) are compared as families, never node by node.

Decides: a family is a candidate for step 4 when the fork is faster in D2 by more than the spread, D1 agrees wherever it has a case for it, and the gain is at least 1% of a 501-token graph or of a one-token graph. Candidates are ordered by ms saved on the 501-token graph first (first token is the top priority), then on the one-token graph. Where ours is faster, nothing changes and the row is recorded.

## 7. After round D

- Step 4, per candidate: a spec in `plan.md` saying which of the fork's driver code is taken and how it sits behind our plugin's steps, built behind a switch by a coder, checked against the CPU on the Mac, then one board run of the two arms in one session with the KLD.
- Step 5: copy `~/ck` if round A found it, else round E; the expert quantize step's deal; a build script of its own.

## 8. What each round copies back

| Folder | Holds |
|---|---|
| `~/cleanup-confirm` | the build log, the check program's output, our full table with op tables, the completion, the KLD |
| `~/fork-overall` | the device and offload checks, three rounds of both backends' bench output, our op tables, the fork's completion and KLD, both builds' logs |
| `~/fork-per-op` | the isolated op timings of both backends, our op tables, the fork's trace tables (not the traces), the comparison table |

## 9. Round A's result (2026-11-25): the board came back empty

- Kernel 6.18.3, 216 GB free, the clock is right. Nothing of ours is left: the repo and its build, the model, spine-runtime, `segv.so`, the KLD text and base, the float run's text, the fork's build, every run folder. `~/ck` and `~/fork-reference*` were never copied and are lost.
- Round B becomes a setup round. First a probe of the tools, `/dev/tcm` and the network, which also fetches spine-runtime 0.6.3 and starts the model's download (ModelScope, 20,836,243,072 bytes, SHA-256 `52312daa5b2190c1f5723d33c3315c01c55af4206f6c6e6eb63f3d8dd52bb85e`). Then: the repo cloned from a bundle of the whole branch, `segv.so` built, ours configured and built from nothing, and the checks and the table as section 4 has them. The fork is built from its archive in round C.
- spine-runtime stays at 0.6.3, the version behind every earlier figure, so that a change in speed is the system's and not the runtime's.
- `build-dates.sh` is not needed while the clock is right; the blocks no longer call it.
- `~/ppl.txt` cannot be restored: its source was never recorded. The KLD text from now on is wikitext-2's test file (`wiki.test.raw`, 1,290,590 bytes, SHA-256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`, copied from the Mac), and the base is remade from it with the CPU on one thread. The figures 0.005115, 0.089103 and 94.510% belonged to the old text and are no longer the expected values. The check is the deck's bound, a mean below 0.01, and the new figures become the reference.
- The completion's text is still compared with the float run's (`~/k3-logs/f32/gen.out` on the Mac).
- If the speed table is outside 2% of the old one, the commit before the cleanup (`d5e9ae0e8`) is built in a second folder and the two run in turn, to tell the system's change from the cleanup's.
- Round E (the chat run without checkpoints) is needed again.

**The probe, the same day.** Bianbu 4.0.5, gcc 15.2.0, git 2.53.0, Python 3.14.4, make; no cmake, no ninja, no tmux. `/dev/tcm` is there. `/usr/lib` now carries `libspert.so.0.6.3` and `libspine_tcm.so.3.0.1` of the system's own. The CPUs are as before (`k3-hardware.md`): 16 online, 8 X100 and 8 A100, a normal process allowed on 0 to 7, so `nproc` says 8. This kernel's `/proc/cpuinfo` lists no vendor extensions in its `isa` lines; whether the runtime still reaches the AI cores is settled by the check program with `--expect-device`. GitHub, ModelScope and Hugging Face all answer with the certificate check on. Correction (the same evening): the kernel and the compiler are not new. `PROJECT.md` records the old system as Bianbu 4.0, kernel 6.18.3, GCC 15.2 and CMake 4.2; the reinstall is Bianbu 4.0.5 with the same kernel and compiler versions, so at most patch levels of packages differ. The model came from ModelScope at about 450 KiB/s on one connection, which would take 12 hours, so the download continues with `aria2c` over several connections from ModelScope and Hugging Face together (the same file on both by SHA-256).

**The setup, the same evening (`~/setup-build`).** cmake 4.2.3, ninja 1.13.2 and tmux installed from the board's archive; the repo cloned from the whole-branch bundle; the build from nothing: 388 steps, rc 0, 109 warning lines (80 in upstream's `ggml-cpu`, 29 in our provider, 25 of them in `plugin/flagos-spacemit-rvv-attn.cpp`, mostly narrowing conversions and sign comparisons; whether the old build printed the same ones was not recorded). The check program with `--expect-device` printed "all checks passed": the cleanup `fd757088f` holds on the real kernels, on the new kernel, with the runtime reaching the AI cores. The model finished downloading with `aria2c` on 16 connections. Next, one block: the model's SHA-256, the new KLD base, the table, the completion, the KLD, then the fork's two builds (plain and profiling) so that rounds C and D need no build.

**The confirmation run, the same evening (`~/cleanup-confirm`, `~/fork-build`).**

- The model's SHA-256 and the wiki text's are as recorded. The new KLD base (`~/kld-wiki/base.bin`, 2 chunks of 512 tokens, CPU, one thread): PPL 5.8619, 506 s a pass.
- The provider against it: mean KLD 0.003381 +- 0.000361, median 0.001238, 99.9% 0.096165, maximum 0.108020, same top token 97.647 +- 0.672%. These are the reference for this text. The completion's text is the float run's; 1 graph split.
- The prompt side is as before or slightly better: pp32 60.59, pp128 83.63, pp501 93.19, pp2048 85.97 t/s (59.85, 82.55, 92.04, 84.50 before the reinstall).
- **Generation is 25% slower: tg512 7.18 t/s against 9.56.** A one-token graph takes 137.2 ms against 102.4. Two steps carry all of it: `CONCAT r8192` 25.3 ms for its 30 calls against 1.4, and `SSM_CONV r1` 14.4 ms against 2.5. Both are the convolution's one-token steps, both run the plain C loops of `plugin/flagos-spacemit-seq.cpp`, a file the cleanup did not touch, and every tile takes the same time. A few small one-row f32 matmuls are 1.3 to 1.9 times slower too; the quantized matmuls and the expert multiply are unchanged. Cause not known. Next: the commit before the cleanup built beside this one and the two run in turn (`~/tg-regression`), with the THP setting, the CPU governors and the two loops' machine code printed.
- The fork built twice with no error line: 173 steps plain, 163 steps with `GGML_BUILD_PROFILE` (`llama-bench` only). Its configure took the system's `/usr/lib/libspert.so`, the same 0.6.3.

**The cleanup cleared of the slow generation (`~/tg-regression`, the same night).** The commit before the cleanup (`d5e9ae0e8`, a second checkout and build) and HEAD, in turn, two rounds, `-p 32 -n 64`:

| run | pp32 | tg64 | one-token graph | `CONCAT r8192` (30 calls) | `SSM_CONV r1` (30 calls) |
|---|---|---|---|---|---|
| HEAD, 1 | 60.54 | 5.66 | 174.4 ms | 62.3 ms | 14.4 ms |
| before the cleanup, 1 | 60.11 | 7.97 | 123.5 ms | 11.8 ms | 14.4 ms |
| HEAD, 2 | 60.36 | 7.94 | 123.9 ms | 11.8 ms | 14.4 ms |
| before the cleanup, 2 | 60.14 | 5.65 | 174.9 ms | 62.3 ms | 14.4 ms |

- Both commits are slow, so the cleanup is not the cause. `SSM_CONV r1` is 14.4 ms in every run of either. `CONCAT r8192` takes one of a few values a process (11.8, 14.8, 25.3, 62.3 ms seen), whichever commit runs: it follows the process, not the build.
- The machine code of the two kernels, read from the plugin: the plain C loops meant as the scalar reference were vectorized by the compiler. The convolution's sum is an ordered vector sum (`vfredosum`) over 4 elements a channel; the concat's copies are strided vector loads and stores of 3 and 1 elements a row.
- Not known: why the same loops were fast before the reinstall (the old binaries are gone; the compiler's package revision or its tuning default may differ), and why the concat's time changes from process to process.
- Next (`~/autovec-test`): the same source built with `-fno-tree-vectorize` beside the build as it is, the two run in turn at pp32, pp128, pp501 and tg64, with the system's settings saved to a file this time.

**It is a different board (the user, 2026-11-25 night).** SpacemiT did not reinstall the old unit: they handed over a new, upgraded one, which is why nothing of ours was on it. So the figures "before the reinstall" in this file, every earlier speed table and the facts in `k3-hardware.md` were measured on another unit. A difference from them can come from the hardware or its firmware as well as from the software. What stays valid: comparisons made on this board in one session (the cleanup against the commit before it). To do: read the new unit's identity (core ids, clocks, caches, firmware) against `k3-hardware.md`, and take this board's own figures as the reference from now on.

**The new unit's identity** is in `k3-hardware.md` section 0: the same core ids, maximum clocks of 2.2 and 1.8 GHz (2.4 and 2.0 on the old unit), 32 KiB first-level caches on the AI cores as sysfs reports them (64 before), more ISA extensions listed, governor `userspace` with both core types at their maximum.

**The slow generation found: the compiler's vector code in two scalar loops (`~/autovec-test`).** The same source built a third time with `-fno-tree-vectorize` for every file, run in turn with the build as it was, two rounds:

| build | pp32 | pp128 | pp501 | tg64 | one-token graph | `CONCAT r8192` | `SSM_CONV r1` |
|---|---|---|---|---|---|---|---|
| as built | 59.90, 60.40 | 83.98, 83.57 | 93.89, 93.85 | 7.91, 7.93 | 124.3, 124.0 ms | 11.8 ms | 14.5 ms |
| no auto-vectorization | 59.71, 59.72 | 83.40, 84.11 | 93.76, 93.94 | 9.83, 9.82 | 99.7, 99.6 ms | 1.4 ms | 0.6 ms |

- Generation is back, and a one-token graph is faster than the old unit's 102.4 ms. The prompt sizes are the same in both builds.
- Turning the vectorizer off everywhere is not the fix: a few plain C steps are slower without it in a prompt graph (`SCALE r1` 0.3 to 3.0 ms, `GLU SWIGLU` 15% and `L2_NORM` 12% at 128 tokens), which the whole graph does not show.
- The fix (in `plugin/flagos-spacemit-seq.cpp`): the two reference loops, `ssm_conv_f32_ref` and `concat_dim0_f32_ref`, carry a GCC attribute that keeps them out of line, not vectorized, and with the multiply and the add apart as ggml-cpu's own build has them. Apple clang ignores it, so the Mac's build and its check program are unchanged ("all checks passed"). Not yet built on the K3: whether GCC 15.2 takes the attribute as written, the loops' machine code and the speed are the next run's to show (`~/scalar-conv-loops`), with the full table as this board's reference, the completion and the KLD.
- Why GCC's version of these loops is so slow on the AI cores (and why the concat's time changed from process to process) is not explained; the loops run 3, 1 and 4 elements at a time, where a vector instruction's fixed cost has nothing to pay it back.

**The fix on the K3, and this board's reference table (`~/scalar-conv-loops`, build at `f0cff0e32`, 2026-11-26).** GCC 15.2 took the attribute without a warning: `ssm_conv_f32_ref` and `concat_dim0_f32_ref` are now functions of their own with no vector instruction and no fused multiply-add. The check program: "all checks passed".

| | pp32 | pp128 | pp501 | pp2048 | tg512 | one-token graph | `CONCAT r8192` | `SSM_CONV r1` |
|---|---|---|---|---|---|---|---|---|
| this board, with the fix | 60.21 | 84.33 | 93.39 | 86.60 | 9.71 | 100.9 ms | 1.26 ms | 0.77 ms |
| this board, before it | 60.59 | 83.63 | 93.19 | 85.97 | 7.18 | 137.2 ms | 25.3 ms | 14.4 ms |
| the old board (`e92642893`) | 59.85 | 82.55 | 92.04 | 84.50 | 9.56 | 102.4 ms | 1.39 ms | 2.52 ms |

- Three more processes at `-p 32 -n 64`: tg64 9.77, 9.75, 9.83; the concat 1.24 ms in each. The change from process to process is gone.
- The bits did not move: the completion's text is the float run's, and the KLD's figures are the same to the last digit (mean 0.003381, maximum 0.108020, same top 97.647%).
- Step 1 of `board-return-plan.md` is done: the cleanup and this fix stand on the new board, and the first row is its reference.

## 10. The fork on its device, first figures (`~/fork-overall`, `~/fork-per-op`, 2026-11-26)

**What ran.** The fork's device came up (`SPACEMIT0`, 8 cores, `use_ime2=1`) and ran the matmul test, the completion and the isolated op timings. The three whole-model rounds and the fork's KLD did not run: the block waited for an "offloaded N/N layers" line that this fork never prints (section 5, item 3). The completion's log shows the model was on the device: 18,918.80 MiB of weights, the KV cache, the recurrent state and a 497 MiB compute buffer in `SPACEMIT` buffers, 272.81 MiB of weights and a 22 MiB compute buffer on the CPU, 3,819 graph nodes in 3 splits (ours: 1).

**The one completion (501 prompt tokens, 31 generated; one run each, ours from `~/scalar-conv-loops` the day before, so an indication and not yet the three rounds in turn).**

| | prompt | generation | load |
|---|---|---|---|
| ours | 5,467 ms (91.64 t/s) | 104.58 ms a token (9.56 t/s) | 29.1 s |
| the fork | 4,262 ms (117.55 t/s) | 114.51 ms a token (8.73 t/s) | 59.9 s |

The fork's text is the float run's, as ours is. Its prompt is 22% shorter; its token takes 9% longer. The loads are not comparable (ours maps the file, the fork ran with `-lm none`).

**Isolated ops, three rounds in turn, the median of each (fork / ours; above 1 ours is faster).** A case counts as won only when the three rounds' ranges do not overlap.

| Op | Case | Ours | The fork | fork / ours |
|---|---|---|---|---|
| dense Q4_0 matmul, 4096 x 14336 | 1 row | 1.14 ms | 1.46 ms | 1.28 |
| | 2 to 8 rows | 1.50 to 1.82 ms | 2.70 to 2.79 ms | 1.51 to 1.81 |
| | 512 rows | 64.68 ms | 29.07 ms | 0.45 |
| expert multiply, 128 experts, 8 used | 1 token | 0.287 ms | 0.255 ms | 0.89 |
| | 4 to 512 tokens | 0.97 to 11.08 ms | 1.12 to 11.91 ms | 1.07 to 1.21 |
| expert multiply, 32 experts, 4 used | 1 and 4 tokens | | | 0.98, 1.10 (ranges overlap) |
| | 8 to 512 tokens | 1.51 to 8.91 ms | 1.93 to 9.85 ms | 1.11 to 1.41 |
| Gated DeltaNet | 10 of 12 cases | | | 0.99 to 1.07 (ranges overlap) |
| | 16 heads x 64 at 1 token; 4 heads x 128 at 64 tokens | | | 0.92; 0.88 |
| attention, F16 cache, 4 query rows | depths 4096 to 16384 | | | 1.48 to 1.93 |
| attention, 1 query row, head 64 | depth 4096; 8192; 16384 | | | 0.90; 1.27; 1.29 |
| attention, 1 query row, head 128 | depths 4096 to 16384 | | | 1.05; 1.01 and 1.02 (ranges overlap) |
| attention, 8 heads x 8 rows, depth 7680 | 1 and 4 tokens | | | 1.13; 1.01 |

- The fork wins clearly in one place: the dense Q4_0 matmul at 512 rows, 2.2 times faster. That is its FP16 path (the test's log shows the repack `q4_0_q41block_f16`), the candidate `board-return-plan.md` step 4 expected first.
- At 1 to 8 rows the same matmul is 1.3 to 1.8 times faster on ours; the fork's time is flat at 2.7 ms from 2 to 8 rows (its 16 rows a call).
- The expert multiply is faster on ours from 4 tokens up in the 128-expert case and from 8 in the 32-expert one (7 to 41%); at 1 token of the 128-expert case the fork is 11% faster. The Gated DeltaNet is the same on both. Attention is faster on ours in most cases and never more than 10% slower.
- Both sides refuse the Q4_0 and Q8_0 KV cache cases. The fork has 8-row cases at depths 4096 to 16384 that our tree's test does not have.
- Sizing the matmul's lead against the whole model (an inference, D2 is the measurement): in our 501-token graph the dense Q4_0 matmuls take 1,221 ms of 5,359, so at the ratio above they would save about 670 ms, about half of the 1,205 ms the fork's prompt was shorter by. All our dense matmuls together take 2,347 ms. Which of the fork's ops make up the rest is what the trace run shows.

**Next (`~/fork-whole-model`, `~/fork-trace`).** The three rounds in turn and the fork's KLD with the corrected check; ours once more with the op tables on, at 4, 32, 128 and 501 tokens and at one; the fork's profiling build with the trace off and then on, one size a run, the traces copied back compressed.
