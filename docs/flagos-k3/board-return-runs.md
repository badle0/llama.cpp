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

## 11. The fork against ours: the three rounds, its KLD and the op tables (`~/fork-whole-model`, `~/fork-trace`, 2026-11-26)

**The whole model, both backends in turn, three rounds (t/s; the range of the three).**

| size | ours | the fork | verdict |
|---|---|---|---|
| pp4 | 22.64 to 22.93 | 17.21 to 17.37 | ours 32% faster |
| pp32 | 60.16 to 60.45 | 56.79 to 57.84 | ours 6% faster |
| pp128 | 83.02 to 84.08 | 95.80 to 97.03 | the fork 16% faster |
| pp501 | 93.54 to 94.35 | 117.79 to 119.10 | the fork 27% faster |
| tg64 | 9.97 to 10.01 | 8.80 to 8.84 | ours 13% faster |

No ranges overlap. At 501 tokens the fork's graph is 1,137 ms shorter (4,214 against 5,351 ms); a token of ours is 13.4 ms shorter (100.1 against 113.5 ms). Our op tables cost 2% of a token (tg64 9.80 with them on) and nothing visible at the prompt sizes.

**The fork's KLD against our CPU base (2 chunks).** Mean 0.021611 +- 0.003827, median 0.004617, 99.9% 1.099585, maximum 1.546852, same top token 94.902 +- 0.975%. Ours against the same base: 0.003381 +- 0.000361, 0.001238, 0.096165, 0.108020, 97.647%. The fork's mean is 6.4 times ours and above our bound of 0.01. Which of its paths costs this is not isolated: it has no switch for any of them. Two of its paths are lossy by its own source and absent from ours: the dense Q4_0 products and partial sums in fp16, and the router weight `ffn_gate_inp` converted from f32 to Q8_0 at load (`ime.cpp:2622-2631`), which picks the experts.

**The trace.** The device runs 2,165 nodes a prompt graph (2,105 a token after the first) against our 2,516 steps; every worker logs each. As read in the source, the last graph of each file is cut short; the first is the warm-up; three clean graphs remain for each prompt size, four at one token. `scripts/fork-trace.py` (new; `show` and `compare`) makes the tables: a node's time is the slowest worker's end minus begin, and the time between nodes is counted apart. The profiler's cost, the same build with the trace off and on:

| | pp4 | pp32 | pp128 | pp501 | one token |
|---|---|---|---|---|---|
| trace off, t/s | 17.06 | 57.12 | 96.02 | 119.44 | 8.49 |
| trace on, t/s | 13.82 | 52.47 | 93.28 | 117.77 | 5.89 |
| time added | 23% | 9% | 3% | 1.4% | 44% |

So the tables at 128 and 501 tokens stand as they are. At 1 and 4 tokens the nodes' own times sum to more than the untraced graph (132.1 ms against 113.5 at one token): a begin's timestamp is taken before its text is written, so each node carries some of the cost, and the small ops' rows there (an `ADD` 12.6 us in the fork against 2.3 in ours) are the profiler's, not the fork's. There the three rounds and the isolated one-row cases are the evidence.

**By family at 501 tokens (ms a graph; ours from the op tables of the same session).**

| family | ours | the fork | fork / ours | the fork's gain |
|---|---|---|---|---|
| dense Q4_0 matmuls | 1,220.5 | 387.7 | 0.32 | 833 |
| the router, `ffn_gate_inp` (f32 in the file) | 353.2 | 43.0 | 0.12 | 310 |
| norms and their multiplies (`RMS_NORM`, `L2_NORM`, `MUL`) | 389.2 | 270.4 | 0.69 | 119 |
| expert multiply | 1,209.7 | 1,109.1 | 0.92 | 101 |
| convolution (`SSM_CONV`, `CONCAT`) | 141.3 | 102.2 | 0.72 | 39 |
| dense Q8_0 | 414.3 | 377.1 | 0.91 | 37 |
| `ADD` | 271.3 | 245.9 | 0.91 | 25 |
| `UNARY` | 116.2 | 93.4 | 0.80 | 23 |
| dense Q6_K and Q5_0 | 277.7 | 262.5 | 0.95 | 15 |
| Gated DeltaNet | 579.5 | 583.1 | 1.01 | -4 |
| the other f32 matmuls (`ssm_alpha`, `ssm_beta`, `ffn_gate_inp_shexp`) | 79.4 | 102.2 | 1.29 | -23 |
| the rest (`ARGSORT`, `GLU`, `ROPE`, `SCALE`, copies, ...) | 144.2 | 191.7 | 1.33 | -48 |
| attention | 158.1 | 443.8 | 2.81 | -286 |
| total | 5,354.6 | 4,212.1 | 0.79 | 1,143 |

The fork also spends 37.8 ms between nodes. Checked apart from the script: the dense Q4_0, f32, expert, attention, router and DeltaNet figures recomputed from the trace by a second program, the same to 0.1 ms.

- Two things make the fork's long prompt faster, and they are 1,143 ms of its 1,143 ms lead: the dense Q4_0 matmuls in FP16 (3.1 times faster here, more than the 2.2 of the isolated case at K 14,336) and the router run as Q8_0 (8 times faster). Everything else nets to zero: its gains in norms, experts and convolution against our 286 ms lead in attention.
- By weight, the Q4_0 gain is in every one of them (`attn_qkv` 565.7 to 180.9, `attn_gate` 284.2 to 95.2, `ssm_out` 157.9 to 46.0, `attn_q` 113.7 to 36.1, `attn_output` 99.1 to 29.4). The Q8_0, Q6_K and Q5_0 weights run on 8-bit paths in both and are within 2 to 10%.
- The norms row is not one kernel against another: the fork's graph has 151 `MUL` and 191 `RMS_NORM` nodes and no `L2_NORM`, ours 281, 131 and 60 steps. The two trees build this part differently.
- The expert multiply is 8% faster in the fork inside the model, where the isolated cases had ours ahead. The shapes differ (256 experts of 2048 to 512 here), so the two views do not agree and it is not a candidate by the rule.
- At 128 tokens the same two: dense Q4_0 273.0 to 133.6 ms, the router 94.7 to 11.7; attention 15.2 against 58.3. At 32 tokens the Q4_0 matmuls are level (75.3 against 64.0, with 9% of trace cost in the fork's figure) and the router is 24.6 to 5.1. At 4 tokens the fork's Q4_0 matmuls take twice ours (27.3 against 59.4), as the isolated 2 to 8 row cases said.
- The router is faster in the fork at every size, one token included (3.9 against 1.6 ms a token, the trace's cost inside the 1.6).

**Candidates for step 4, by the rule of section 6 (faster by more than the spread, both views agree where both exist, at least 1% of a graph), ordered by ms at 501 tokens.**

| candidate | gain at 501 tokens | isolated view | size of the port | open |
|---|---|---|---|---|
| dense Q4_0 in FP16, from about 16 rows | 833 ms, 15.6% | agrees (0.45 at 512 rows) | the largest so far (`plan.md`, Part B) | its KLD |
| the router as Q8_0 | 310 ms, 5.8%; 2 ms a token | no shared case | small: one more load-time conversion onto the Q8_0 path we have | its KLD: it decides which experts run |

With both at the fork's speed our 501-token graph would be about 4.2 s (119 t/s) by sum, the fork's own figure, with our leads at small sizes and in generation kept. Part A's rule for going on to Part B (`plan.md`) is met: the Q4_0 matmul is more than 15% faster at 512 rows and the fork's pp501 is above ours. Not candidates: the expert multiply (the views disagree), convolution, `ADD` and `UNARY` (each under 1%).


## 12. The router as Q8_0 on the board (`~/router-q8`, 2026-11-26)

The first port from section 11's candidates (`plan.md`, "The router as Q8_0, from SpacemiT's fork"; its rule was fixed before this run). Build `a8f92959a`: 53 steps, rc 0. The check program with the device passes under both settings of `FLAGOS_SPACEMIT_ROUTER`, so the new layout's matmul ran on the real Q8_0 kernels at the router's shape.

**Speed, both settings in turn, three rounds (t/s; the range of the three; `f32` first in rounds 1 and 3, `q8` first in round 2).**

| size | `f32` (the default) | `q8` | medians | verdict |
|---|---|---|---|---|
| pp4 | 21.72 to 23.38 | 23.15 to 23.80 | 22.88, 23.56 | ranges overlap; `q8` higher in each round |
| pp32 | 60.13 to 60.49 | 62.66 to 62.75 | 60.26, 62.72 | `q8` 4.1% faster |
| pp128 | 82.90 to 83.21 | 87.46 to 87.57 | 83.12, 87.56 | `q8` 5.3% faster |
| pp501 | 93.76 to 94.12 | 98.71 to 99.39 | 93.79, 99.17 | `q8` 5.7% faster (5.7, 6.0 and 4.9% by round) |
| tg64 | 9.95 to 10.00 | 10.22 to 10.24 | 9.95, 10.23 | `q8` 2.8% faster (100.5 to 97.8 ms a token) |

The estimates before the run were about 6% at pp501, about 5% at pp128 and level to 3% at one token.

**The router's steps, from the op tables of one more run of each setting (ms a graph, 40 matmuls).**

| rows | `f32`: one step | `q8`: gemm + quantize | the fork (section 11) | the graph, `f32` to `q8` |
|---|---|---|---|---|
| 1 | 3.9 | 1.1 + 0.2 | 1.6 | 100.1 to 97.7 |
| 4 | 5.0 | 1.2 + 0.9 | 2.0 | 178.2 to 167.4 |
| 32 | 24.6 | 3.4 + 1.1 | 5.1 | 529.7 to 509.7 |
| 128 | 94.8 | 10.9 + 3.5 | 11.7 | 1,539.0 to 1,463.3 |
| 501 | 353.2 | 35.4 + 17.4 | 43.0 | 5,364.3 to 5,062.2 |

At 501 tokens the router went from 353.2 to 52.8 ms and the graph is 302.1 ms shorter, with 40 more steps (2,556). A third of what is left is the activations' quantize step; sharing it with the shared experts' Q8_0 matmuls, which read the same input, is not done.

**Accuracy.** A new CPU base of 8 chunks was made for this: `~/kld-wiki/base-8.bin` on the board, by `FLAGOS_SPACEMIT_DISABLE=1 llama-perplexity -m <the model> -f ~/wiki.test.raw -c 512 --chunks 8 -t 1 --kl-divergence-base <file>`; PPL 6.2963 +- 0.3449, 33 minutes (991 s a pass of 4 chunks). Its first two chunks give the 2-chunk base's PPL (5.8619).

| | mean KLD | median | 99.9% | maximum | same top token |
|---|---|---|---|---|---|
| `f32`, 8 chunks | 0.006936 +- 0.000374 | 0.002501 | 0.191495 | 0.279836 | 97.010 +- 0.377% |
| `q8`, 8 chunks | 0.007150 +- 0.000383 | 0.002594 | 0.187847 | 0.315684 | 97.304 +- 0.359% |
| `f32`, 2 chunks (section 9) | 0.003381 +- 0.000361 | 0.001238 | 0.096165 | 0.108020 | 97.647 +- 0.672% |
| `q8`, 2 chunks | 0.003553 +- 0.000364 | 0.001162 | 0.098536 | 0.114114 | 97.647 +- 0.672% |

- The rule's bound is `f32`'s mean plus its error bar, 0.007310; `q8` is at 0.007150, 3.1% above `f32` and 0.6 of an error bar. The two runs score the same tokens, so the error bars are not independent and this is the rule as written, not a test of the difference.
- The completion of the 501-token prompt under `q8` is the float run's text, byte for byte; 1 graph split.
- Our own mean on 8 chunks is twice the 2-chunk figure: chunks 3 to 8 score about 0.0075 each against 0.003 for the first two. 0.006936 is this text's reference from now on; the bound of 0.01 is closer than the 2-chunk run said.
- The first two chunks score 0.00360 inside the 8-chunk run against 0.003381 alone: the decodes differ (4 chunks a batch against 2). Not looked into.
- The router is not what costs the fork its accuracy: on the same 2 chunks the fork's mean is 0.021611 and the router as Q8_0 moves ours from 0.003381 to 0.003553. Of the two lossy paths named in section 11 that leaves its FP16 dense matmuls, by elimination and not measured; the fork was not run on 8 chunks.

**Verdict by the rule:** all three points hold, so `q8` is proposed as the default, the user's decision.

**Against the fork with `q8` (the fork's figures are section 11's, another session of the same day).** pp501 98.71 to 99.39 against 117.79 to 119.10, the fork 19% ahead (27% before); pp128 87.46 to 87.57 against 95.80 to 97.03, the fork 10% ahead (16%); pp32 ours 9% ahead (6%); tg64 10.22 to 10.24 against 8.80 to 8.84, ours 16% ahead (13%). What is left of the fork's lead at long prompts is the dense Q4_0 matmuls.

**The default confirmed (`~/router-default`, build `196eb4aaf`, the same day).** On the user's word `q8` became the default, with `FLAGOS_SPACEMIT_ROUTER=f32` the way back. The check program passes with the device under the default and under `f32`.

| | pp4 | pp32 | pp128 | pp501 | pp2048 | tg64 | tg512 |
|---|---|---|---|---|---|---|---|
| the default (`q8`) | 22.41 | 62.62 | 87.57 | 98.77 | 88.79 | | 10.15 |
| `f32` | 22.40 | 60.29 | 83.09 | 93.91 | | 9.94 | |

- Both rows are inside the three rounds' ranges above (`q8` pp501 98.71 to 99.39, `f32` 93.76 to 94.12). The default's row is this board's reference from now on; it replaces section 9's (pp32 60.21, pp128 84.33, pp501 93.39, pp2048 86.60, tg512 9.71 on build `f0cff0e32`, another session).
- The op table at 501 tokens: under the default the router is the gemm and quantize rows (35.3 and 17.9 ms, 2,556 steps a graph), under `f32` the gemm row alone (353.2 ms, 2,516 steps).
- The KLD under the default is the `q8` arm's to the last digit, on 8 chunks (0.007150 +- 0.000383, maximum 0.315684, same top token 97.304%) and on 2 (0.003553 +- 0.000364); the 8-chunk output file is the same bytes. The completion's text is the float run's, 1 graph split, 501 tokens at 99.80 t/s and 99.55 ms a token.

## 13. The FP16 arithmetic on the board (`~/fp16-arithmetic-retry`, build `b92ccfcc7`, 2026-11-26)

Stage 1 of the FP16 port (`plan.md`, "The dense Q4_0 matmuls in FP16: the port"): the dense Q4_0 matmuls under `FLAGOS_SPACEMIT_DENSE_Q4_0=f16` on scalar reference code, no kernel yet. The run measures accuracy, not speed.

**The first attempt (`~/fp16-arithmetic`) stopped in the build,** at upstream's web UI step (`llama-ui-assets`), which only `llama-server` needs; the provider's new file had compiled. The step downloads the UI archive named after the build number (`b10646`, which does not exist for a fork's count), falls back to upstream's `latest`, and embeds it. `latest` now holds a newer UI (`_app/immutable/...`) than this tree's `llama-ui-embed` expects (`bundle[hash].js`, `bundle[hash].css`, `workbox[hash].js`), so the embed fails: "missing required asset(s)". The two builds of the same morning passed; upstream's archive changed in between (inferred from the two logs, not looked up). Until the tree is rebased or the step is configured away, `llama-server` cannot be rebuilt on the board; its binary there is build `196eb4aaf`'s. The retry built the named targets without it (rc 0, 22 steps).

**The check program** passes with the device under `int8` and under `f16` (0 FAIL lines, the two outputs the same bytes; the new check's 323 points among them). The loader line reads `tcm 393216 per tile`: 384 KiB on this unit too.

**The default did not move** (one run, `-r 3`): pp4 23.60, pp32 63.23, pp128 88.77, pp501 100.40, tg64 10.29 t/s. All are 1 to 2% above section 12's reference (pp501 98.71 to 99.39 over four runs); one run, not repeated, and nothing in the int8 path changed, so it is read as the session's drift. Its KLD file is the default's of section 12 line for line apart from the time estimate.

**The shapes of the five dense Q4_0 weights** (from the stats file's counts per shape and the op table's time per matmul; 92 a graph):

| weight | matmuls a graph | K | N |
|---|---|---|---|
| `attn_qkv` | 30 | 2,048 | 8,192 |
| `attn_q` | 6 | 2,048 | 8,192 |
| `attn_gate` | 30 | 2,048 | 4,096 |
| `ssm_out` | 16 | 4,096 | 2,048 |
| `attn_output` | 10 | 4,096 | 2,048 |

The stats file gives 36 matmuls at [2,048, 8,192], 30 at [2,048, 4,096] and 26 at [4,096, 2,048]; which name has which is from the times (`attn_qkv` and `attn_q` 2.39 s a matmul on the scalar code, the other three 1.19 to 1.20 s).

**The values.** Over the KLD run's 736 matmuls (8 graphs of 512 rows): largest activation 58.13 (`blk.20.attn_qkv`), largest partial sum 22.57 (`blk.38.attn_qkv`), no overflow. fp16 holds 65,504, so the margin is three orders of magnitude on this text.

**Accuracy, 8 chunks against `~/kld-wiki/base-8.bin`.**

| | mean KLD | median | 99.9% | maximum | same top token |
|---|---|---|---|---|---|
| `int8` (the default) | 0.007150 +- 0.000383 | 0.002594 | 0.187847 | 0.315684 | 97.304 +- 0.359% |
| `f16` | 0.006329 +- 0.000342 | 0.002324 | 0.192597 | 0.326131 | 97.598 +- 0.339% |
| `f16`, partials as f32 | 0.006655 +- 0.000376 | 0.002328 | 0.205979 | 0.345795 | 97.108 +- 0.371% |

- The rule's bound is `int8`'s mean plus its error bar, 0.007533. `f16` is at 0.006329 and `f16` with f32 partials at 0.006655: both hold.
- `f16` is 11.5% below `int8`, 2.1 of its error bars; the mean and the median are lower, the maximum and the 99.9% point slightly higher. The runs score the same tokens, so the bars are not independent; no test of the difference was made.
- Storing the partials as fp16 costs nothing that shows: that arm is the lower of the two FP16 arms, by less than one error bar. It is not shown to be better.
- The completion of the 501-token prompt under `f16` is the float run's text, byte for byte; 1 graph split.
- FP16 arithmetic of this kind is not what costs the fork its accuracy. The first two chunks inside this run score 0.00296 under `f16` (0.00375 under `int8`); the fork scored 0.021611 on those two chunks alone (section 11). What in the fork's path differs from ours (its dequantization, its other lossy paths) is not isolated.

**Speed is the scalar code's and means nothing:** one 501-token graph 163.7 s (3.06 t/s), the products step 152.7 s and the sum step 7.2 s of it; a KLD arm about 21 minutes.

**Verdict by the rule:** holds, with fp16 partials. Stage 2 (the kernels, the plain route) follows as specified.

## 14. The FP16 kernels on the board, the first attempt (`~/fp16-kernels`, build `a99a2f74f`, 2026-11-26)

Stage 2 of the FP16 port (`plan.md`, "Stage 2 as built"): SpacemiT's FP16 kernels under `FLAGOS_SPACEMIT_DENSE_Q4_0=f16`. Read from the block's printed lines; the folder was not copied back. The run stopped at the check program, as the block is written to, so it holds no speed and no KLD.

**The build passed** (rc 0, 38 steps, without `llama-server`): GCC assembles the copied text, the configure step found `vfwmadot`, no warning in the two FP16 files.

**The check program on the kernels failed in two of its points,** the same two under `int8` and under `f16`:

- Bit for bit on the hardware: the weights read back (three cases) and the small integers (two cases). These go through the written dequantization, the pack, the product kernel, the two reduce kernels and the real TCM as the stage.
- Not bit for bit: the activations read back, 3,056 of 12,384 outputs at [2048, 96] by 129 rows and 1,024 of 4,096 at [4096, 64] by 64 rows.
- For random inputs 132,700 of 133,536 outputs are the model's bytes and every one lies inside the model's bound; the largest NMSE against the CPU is 1.47e-05 (bound 5e-04).
- With `GGML_FLAGOS_KERNEL_VARIANTS=0` (the references, on the board) all checks pass.

**The reading: fp16 subnormals count as zero on the kernels' path** (inferred, not observed directly). The failing case is the only one that feeds fp16 subnormals (values below 2^-14 = 6.1e-05). Computed again on the Mac with every such activation taken as zero, the expected outputs differ from the IEEE ones in exactly 3,056 and 1,024 places. Six other candidates do not give these counts: ties rounded away from zero (3,796 and 1,253), to odd (6,394 and 2,076) or toward zero (3,794 and 1,230), a negative zero taken as positive (0 and 0), the slices summed in reverse (3,645 and 1,218) or in pairs (2,653 and 861). The counts agree; that the same outputs differ was not compared (the folder's outputs are not printed).

- Where the value is lost is not known: the pack's narrowing, `vfwmadot`'s inputs, the store of a slice's partial sum, or the sum's widening.
- The references keep subnormals (IEEE), so the Mac could not have shown this.
- For a model the effect is an activation below 6.1e-05 read as zero; the KLD of the retry measures whether it shows.

**What follows:** the bit-for-bit case no longer holds subnormals, and four probes report what the kernels do with a subnormal activation, weight and partial sum (`plan.md`, "Stage 2 on the K3, the first attempt"). The retry is `~/fp16-kernels-retry`.

## 15. The FP16 kernels on the board, the retry (`~/fp16-kernels-retry`, build `539c1c976`, 2026-11-26)

Stage 2 of the FP16 port with the check program's subnormal probes (section 14). Default arm `int8`, other arm `FLAGOS_SPACEMIT_DENSE_Q4_0=f16` (from 64 rows); the two in turn, three rounds.

**Build and checks.** Build rc 0 (34 steps, without `llama-server`). The check program passes on the kernels under `int8` and under `f16` (233 points, 0 FAIL lines) and on the references (`GGML_FLAGOS_KERNEL_VARIANTS=0`). All seven exact cases are bit for bit on the kernels, the activations read back among them now that they hold no subnormal. For random inputs 132,700 of 133,536 outputs are the model's bytes, all inside the bound.

**The subnormal probes on the kernels:** an activation by 1 zero, an activation by 1,024 zero, a weight by 1,024 zero, a partial sum kept (all four kept on the references). A subnormal that enters the product is lost on either side, and one that the product makes from normal inputs is stored and summed. One cause fits all four: `vfwmadot` reads an fp16 subnormal operand as zero (inferred; the narrowing that packs the activations is the instruction that stores the partial sum, which keeps its subnormal). Section 14's reading stands.

**Speed, t/s (`-p 4,32,128,501 -n 64 -r 3`).**

| size | `int8`, rounds 1 to 3 | `f16`, rounds 1 to 3 | `f16` against `int8` |
|---|---|---|---|
| pp4 | 23.64, 23.20, 22.79 | 23.22, 23.04, 22.48 | -0.7 to -1.8%, inside the runs' own spread (0.6 to 1.7); the path is not taken below 64 rows |
| pp32 | 62.44, 63.15, 62.34 | 62.85, 62.80, 62.53 | level |
| pp128 | 88.36, 88.13, 88.28 | 81.79, 81.65, 81.61 | 7.4 to 7.6% slower |
| pp501 | 99.04, 99.87, 99.44 | 108.48, 108.26, 108.19 | +9.5, +8.4, +8.8% |
| tg64 | 10.25, 10.28, 10.26 | 10.26, 10.26, 10.25 | level |
| pp2048 (one pair, `-r 2`) | 91.26 | 95.18 | +4.3% |

The threshold pair (`-p 16,32,64,128 -r 3`, `int8` against `f16:16`): 46.64 against 27.68 at 16 rows, 63.14 against 43.45 at 32, 75.46 against 62.14 at 64, 87.89 against 82.01 at 128. FP16 is slower at every size up to 128 rows.

**Accuracy.** KLD on 8 chunks: `f16` on the kernels 0.006689 +- 0.000337 (median 0.002378, 99.9% 0.181019, maximum 0.209118, same top token 97.108 +- 0.371%); `int8` 0.007150 +- 0.000383 as before. The bound is 0.007533: holds. Stage 1's scalar code gave 0.006329. 736 matmuls took the path, no overflow. The completion of the 501-token prompt is the float run's text, byte for byte; 1 graph split.

**The dense Q4_0 matmuls by the op tables (ms a graph).**

| rows | `int8` | `f16`: products + sum | the fork (section 11) |
|---|---|---|---|
| 128 | 269.5 | 381.3 = 333.2 + 48.2 | 133.6 |
| 501 | 1,211.7 | 822.3 = 622.4 + 200.2 | 387.7 |

The 501-token graph is 4,617 ms against 5,021 (404 ms shorter; the dense Q4_0 matmuls 389 of it).

**Verdict by stage 2's rule.** pp501 is at least 8% above `int8` in every round (narrowly: 8.4% in the second); pp4, pp32 and one token are not slower; pp2048 is not slower; the KLD holds. The rule is met. Outside the rule: pp128 is 7.5% slower, because the default threshold of 64 rows is below the break-even, which lies between 128 and 501 rows and was not measured (the sizes run were 16 to 128).

**The same kernels take 2.1 times the fork's time at 501 rows and 2.9 times at 128.** The kernels are the fork's text, so the difference is in what surrounds them. Read from the fork's driver after the run (`ime.cpp` 2093-2283): with 8 threads and K from 2,048 the fork does not take the plain split-K route that stage 2 follows; it takes the pairs (two cores share a slice of K/4 and each other's weight tile), and the plain route is its fallback. Three differences, sized from the two row counts (two points a line, so estimates):

- *A cost that does not grow with the rows:* about 234 ms a graph in our products step, against about 46 ms in the fork's whole matmul. Per weight it follows the weight's size at the same rate (`attn_qkv` 3.9 ms a matmul for 16.8 M values, `attn_gate` 2.0 ms for 8.4 M, `ssm_out` 2.0 ms a pass for 8.4 M: 4.2 G values a second on 8 cores). The only work of that kind is the dequantization of the weights, the one kernel written here and not copied (a strided load of 32 quads a group and a gather for the scales). The fork's, an assembler kernel on its own layout, runs near the memory's speed.
- *The sum step:* 200 ms at 501 rows, at the memory's speed (`attn_qkv` moves about 84 MB in 3.5 ms). We keep 8 slices of partial sums, the fork's pairs 4, so its sum reads half.
- *The products by the row:* 0.78 ms a row against 0.68 for the fork's whole matmul, its sum included: twice the partial sums written and twice the kernel calls at half the K each.

None of the three is the references or their switch. The first is our weight layout and a slow kernel of mine; the other two follow from 8 slices instead of the pairs' 4, and the pairs need two cores to wait on each other inside a step (stage 3). Corrected after section 16: such a wait exists in our executor since the 1-row path (`spacemit_pair_wait`); what the FP16 steps lack is its use and a view of the partner's TCM.

## 16. The fork's dequantization kernel on the board (`~/fp16-fork-dequant`, 2026-11-26)

The FP16 path with SpacemiT's dequantization kernel on our weight block in place of the one written here (`plan.md`, "The fork's dequantization kernel on our block"). Section 15's block again; the board's commit line was not sent back (the bundle's head was `3d003fee0`).

**Build and checks.** Build rc 0 (34 steps): the board's assembler takes the ten instructions new to the plugin. The check program passes on the kernels under `int8` and `f16` and on the references, 0 FAIL lines; the seven exact cases bit for bit, the three that read the weights back through the new kernel among them; 132,700 of 133,536 outputs the model's bytes and the four subnormal probes as in section 15.

**The same bits.** The KLD summary under `f16` is section 15's line for line (0.006689 +- 0.000337, maximum 0.209118, same top token 97.108%), and the completion is the same text, the float run's. 736 matmuls, no overflow.

**Speed, t/s (`-p 4,32,128,501 -n 64 -r 3`, the two arms in turn).**

| size | `int8`, rounds 1 to 3 | `f16`, rounds 1 to 3 | `f16` against `int8` | section 15 |
|---|---|---|---|---|
| pp4 | 22.89, 23.11, 23.18 | 23.49, 22.91, 23.22 | level | level |
| pp32 | 62.58, 62.60, 62.68 | 62.77, 62.88, 62.72 | level | level |
| pp128 | 87.59, 87.53, 87.57 | 95.04, 96.51, 94.64 | +8.5, +10.3, +8.1% | -7.5% |
| pp501 | 99.36, 99.21, 99.35 | 114.39, 114.80, 114.44 | +15.1, +15.7, +15.2% | +8.4 to +9.5% |
| tg64 | 10.27, 10.28, 10.25 | 10.29, 10.29, 10.24 | level | level |
| pp2048 (one pair, `-r 2`) | 91.21 | 101.40 | +11.2% | +4.3% |

**The threshold pair (`int8` against `f16:16`, one run each, `-r 3`).**

| rows | `int8` | `f16` | |
|---|---|---|---|
| 16 | 46.57 | 44.61 | 4.2% slower |
| 32 | 62.94 | 63.30 | level |
| 64 | 75.61 | 78.86 | +4.3% |
| 128 | 88.90 | 95.32 | +7.2% |
| 256 | 93.75 | 105.04 | +12.0% |

The two paths cross near 32 rows; from 64 rows, the threshold as built, FP16 is the faster one. One run an arm, so 32 against 64 is not settled finer than that.

**The dense Q4_0 matmuls by the op tables (ms a graph).**

| rows | `int8` | `f16` in section 15 | `f16` now: products + sum | the fork (section 11) |
|---|---|---|---|---|
| 128 | 268.6 | 381.3 | 156.7 = 110.9 + 45.8 | 133.6 |
| 501 | 1,210.7 | 822.3 | 538.3 = 357.2 + 181.2 | 387.7 |

- The kernel took 225 ms off a graph at 128 rows and 284 at 501, more than the 190 estimated. What does not grow with the rows is now about 26 ms a graph (two points a line), below the fork's 46.
- The 501-token graph is 4,344 ms against 4,993 on `int8` (the fork's was 4,214 in its own session). Our pp501 of 114.4 to 114.8 is 3 to 4% below the fork's 117.8 to 119.1 of section 11, measured in another session; at 128 tokens we are level with it (94.6 to 96.5 against 95.8 to 97.0), and ahead at 4 and 32 tokens and in generation as before.
- What is left to the fork is 151 ms at 501 rows (3.5% of the graph) and 23 ms at 128. It grows with the rows: the sum step, 181 ms at 501 rows over 8 slices where the fork's pairs sum 4, and the products' share of the same.

**Verdict.** Stage 2's rule is met with room: pp501 at least 15% above `int8` in every round, nothing slower, pp2048 11% above, the KLD unchanged inside its bound. The dense Q4_0 matmuls are at 538 ms at 501 rows, below the 600 that would call for stage 3.

## 17. The pairs on the board (`~/fp16-pairs`, 2026-11-27)

Stage 3 of the FP16 port (`plan.md`, "`f16` as the default, and stage 3 as specified"), with `f16` the default. Three arms in turn: `int8` (`FLAGOS_SPACEMIT_DENSE_Q4_0=int8`), `plain` (nothing set: the default, 8 slices) and `pairs` (`FLAGOS_SPACEMIT_F16_ROUTE=pairs`: 4 slices, a pair of cores sharing each other's weight tile). The board's commit line was not sent back (the bundle's head was `39f983f78`).

**Build and checks.** Build rc 0 (49 steps; one warning in `core/flagos-spacemit.cpp:482`, a file this change did not touch, recompiled for a header). The check program passes on the kernels unset, under `pairs` and under `int8`, and on the references under both routes, 0 FAIL lines in the five. Under `pairs` on the kernels the seven exact cases are bit for bit, so two cores of a pair read each other's TCM correctly through the pair waits; 438,902 of 440,896 outputs are the 4-slice model's bytes and all are inside its bound (the plain route's 132,700 of 133,536 as before). The subnormal probes read the same on both routes. No pair wait gave up in any run.

**Speed, t/s (`-p 4,32,128,501 -n 64 -r 3`, three rounds).**

| size | `int8` | `plain` | `pairs` | `pairs` against `plain` |
|---|---|---|---|---|
| pp4 | 23.12, 23.89, 23.08 | 23.39, 22.92, 23.36 | 23.04, 23.27, 23.26 | level |
| pp32 | 62.91, 62.76, 62.96 | 62.88, 62.88, 62.97 | 62.83, 62.58, 63.16 | level |
| pp128 | 87.67, 89.17, 88.85 | 95.23, 94.91, 96.11 | 97.87, 97.73, 97.90 | +2.8, +3.0, +1.9% |
| pp501 | 99.04, 100.03, 99.96 | 115.00, 114.86, 115.26 | 119.53, 119.62, 119.52 | +3.9, +4.1, +3.7% |
| tg64 | 10.28, 10.31, 10.34 | 10.30, 10.28, 10.30 | 10.33, 10.27, 10.28 | level |
| pp2048 (one run, `-r 2`) | 88.68 | 100.25 | 107.90 | +7.6% |

Against `int8` the pairs are 19.6 to 20.7% above at pp501, 9.6 to 11.6% at pp128 and 21.7% at pp2048. From 16 rows (`f16:16`, one run an arm) the pairs are level with the plain route or 1% above at 16, 32, 64, 128 and 256 rows (44.52 against 44.50, 64.11 against 63.61, 80.22 against 79.16, 97.50 against 96.40, 108.38 against 107.13), so the threshold of 64 holds for them too.

**Accuracy.** KLD on 8 chunks under `pairs`: 0.006362 +- 0.000344 (median 0.002430, 99.9% 0.188862, maximum 0.313382, same top token 97.108 +- 0.371%), inside the bound of 0.007533. The plain route's is section 16's line for line (0.006689 +- 0.000337). All 736 matmuls took the pairs, no overflow. The completion of the 501-token prompt is the float run's text; 118.62 t/s in that run.

**The dense Q4_0 matmuls by the op tables (ms a graph).**

| rows | `plain`: products + sum | `pairs`: products + sum | the fork (section 11) |
|---|---|---|---|
| 128 | 156.4 = 111.4 + 45.1 | 128.4 = 101.9 + 26.5 | 133.6 |
| 501 | 540.0 = 360.3 + 179.7 | 388.9 = 284.8 + 104.1 | 387.7 |

- The pairs take 151 ms off a graph at 501 rows, half of it in the sum (4 slices read where there were 8) and half in the products. They are at the fork's figure at 501 rows and below it at 128.
- The 501-token graph is 4,206 ms (the plain route 4,370; the fork's 4,214 in its own session). Our pp501 of 119.5 to 119.6 is at the top of the fork's 117.8 to 119.1 of section 11, measured in another session, so the two are level there; at 4 and 32 tokens and in generation we stay ahead as before.
- The waits cost nothing that shows: the products steps' imbalance between tiles is 2.9 ms of `attn_qkv`'s 128.0.

**Verdict by stage 3's rule.** `pairs` is at least 3% above `plain` at pp501 in every round (3.7 to 4.1%); nothing is slower at pp4, pp32, pp128, one token or pp2048; the KLD holds; the check program passes on the kernels and the references. The rule is met: the pairs become the default route.

**The default confirmed (`~/fp16-pairs-default`, the same day; one run an arm).** With the pairs as the default route the build passes (37 steps) and the check program passes on the kernels with nothing set, under `plain` and under `int8`, 0 FAIL lines, the same counts as above. With nothing set the KLD is the `pairs` arm's above line for line (0.006362 +- 0.000344), all 736 matmuls take the pairs, and the completion is the float run's text (118.42 t/s for the prompt, 10.12 for the 31 tokens). Speed: nothing set 23.62, 62.98, 97.40, 119.05 and 10.34 t/s at pp4, pp32, pp128, pp501 and one token; `plain` 23.65, 63.37, 96.58, 114.37 and 10.31. pp501 is 4.1% above `plain`, as in the three rounds; both arms read 0.4 to 0.6% below those rounds, one run each, taken as the session's drift.

## 18. The acceptance script on the small models under the new defaults (`~/acceptance-small-models`, 2026-11-27)

The defaults (`f16` from 64 rows, the pairs as its route) were decided on Qwen3.6-35B-A3B alone and apply to every model. This run is the acceptance script (`tools/spacemit_check.py`, with `--skip-server`: the board's `llama-server` is an older build's and cannot be rebuilt there) on Qwen3-0.6B and Qwen3-4B Q4_0, downloaded to the new board for it (sha256 as recorded from the old board), then three rounds of `FLAGOS_SPACEMIT_DENSE_Q4_0=int8` against nothing set. Build `6d8142b96`, no rebuild. The text is `wiki.test.raw` (8 chunks), not the old board's `ppl.txt`, so perplexities and KLDs are not comparable with earlier acceptance runs.

**The script.** 56/56 checks on both models, its first runs on this board.

| | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| perplexity, provider / CPU | 23.0355 / 23.0657 (-0.13%) | 14.9383 / 14.9376 (+0.00%) |
| KLD against the CPU, 512-token batches | 0.002826 +- 0.000146 | 0.001736 +- 0.000230 |
| KLD against the CPU, `-ub 1 -b 512` | 0.003130 +- 0.000102 | 0.001767 +- 0.000096 |
| greedy text against the CPU's | identical | differs after 143 characters, as before the FP16 port |
| the script's `llama-bench`, pp128 / tg128, t/s | 946.24 / 56.52 | 216.56 / 11.12 |
| the same on the CPU (8 X100 threads) | 152.57 / 28.99 | 21.23 / 5.41 |

The 512-token batches go through the FP16 path and `-ub 1` through the 1-row integer path; the two KLDs are within their errors of each other on the 4B and 10% apart on the 0.6B, the FP16 one the lower. No TCM block check (no `~/tcmtest` on this board).

**`int8` against nothing set, three rounds in turn (`-p 32,128,512 -n 32 -r 3`), t/s.**

| | `int8` | nothing set | nothing set against `int8` |
|---|---|---|---|
| Qwen3-0.6B pp32 | 598.04, 606.11, 599.20 | 602.10, 603.89, 602.65 | level (32 rows are below the threshold: the same path) |
| Qwen3-0.6B pp128 | 712.08, 703.04, 701.59 | 962.66, 943.11, 956.02 | +34 to +36% |
| Qwen3-0.6B pp512 | 528.68, 526.00, 525.59 | 741.78, 744.30, 743.68 | +40 to +42% |
| Qwen3-0.6B tg32 | 58.45, 58.21, 58.36 | 58.35, 57.98, 57.75 | level (the ranges overlap) |
| Qwen3-4B pp32 | 108.36, 108.44, 108.42 | 108.36, 108.91, 108.57 | level (the same path) |
| Qwen3-4B pp128 | 106.49, 106.61, 106.97 | 214.99, 213.41, 215.39 | 2.00 to 2.02 times |
| Qwen3-4B pp512 | 96.16, 96.28, 95.61 | 199.18, 195.12, 196.94 | 2.03 to 2.07 times |
| Qwen3-4B tg32 | 11.19, 11.21, 11.20 | 11.20, 11.21, 11.19 | level |

- The defaults regress nothing on the small models and gain far more than on the 35B (pp501 +20% there): in Qwen3 the dense Q4_0 matmuls are most of a prompt graph, in the 35B a quarter of it.
- The `int8` arm reads what the old board read in its last acceptance runs (Qwen3-4B pp128 106.2 to 106.8, pp512 96.3; Qwen3-0.6B about 700 and 529), so the two units agree on that path.
- Which route each weight took was not counted in this run. By the rule (pairs up to a row length of 4,864) Qwen3-4B's K 2,560 weights take the pairs and its K 9,728 projection the plain route; an inference.
- **Open, seen here:** below 64 rows both arms run the integer path, and on the 4B that path at 32 tokens (108 t/s) is half the FP16 path's speed at 128 (215). The threshold of 64 was set on the 35B, where 32 rows were level and 16 rows 4% slower in FP16. On Qwen3 a lower threshold may gain at 16 to 63 tokens; not measured. `FLAGOS_SPACEMIT_DENSE_Q4_0=f16:<rows>` measures it.

## 19. The fork against the new defaults, the pair wait and the threshold (`~/fork-against-pairs`, 2026-11-27)

Step 2 of the order agreed after the FP16 port: one session with both backends in turn, on build `df80f446d` (the pair-wait benchmark and the wait count; board at `bdf40b59c`). Build rc 0 (62 steps, without `llama-server`); the check program passes with 0 FAIL lines, `pairchk` 2/2, the pairs' exact cases bit for bit. The completion is the float run's text (119.89 t/s for the 501-token prompt, 10.06 for the 31 tokens).

**The whole model, Qwen3.6-35B-A3B, three rounds in turn (t/s).**

| size | ours | the fork | verdict |
|---|---|---|---|
| pp4 | 23.26, 23.30, 23.30 | 17.10, 17.33, 17.51 | ours 34% faster |
| pp32 | 63.10, 62.75, 62.79 | 57.42, 57.80, 58.23 | ours 9% faster |
| pp128 | 97.49, 96.81, 96.95 | 96.06, 97.17, 95.60 | level (the ranges overlap) |
| pp501 | 119.53, 118.50, 118.03 | 114.63, 118.56, 116.98 | level (the ranges overlap; ours the higher in two rounds of three) |
| tg64 | 10.26, 10.25, 10.26 | 8.88, 8.83, 8.82 | ours 16% faster |

Section 11 had the fork 16% ahead at pp128 and 27% at pp501. The fork leads nowhere now.

**KLD against our CPU base, the same 8 chunks.** Ours 0.006362 +- 0.000344 (maximum 0.313382, same top token 97.108%), the figures of section 17 to the last digit. The fork 0.023747 +- 0.001695 (median 0.007783, 99.9% 1.005546, maximum 2.038259, same top token 93.873 +- 0.531%): 3.7 times ours. Its 2-chunk figure of section 11 (0.021611) holds on 8.

**By family at 501 tokens (ms a graph; ours from the op tables, the fork from its trace, both this session).**

| family | ours | the fork | fork / ours | the fork's gain |
|---|---|---|---|---|
| expert multiply | 1,235.6 | 1,139.8 | 0.92 | 96 |
| norms and their multiplies (`RMS_NORM`, `L2_NORM`, `MUL`) | 369.4 | 274.3 | 0.74 | 95 |
| dense Q8_0 | 410.8 | 362.7 | 0.88 | 48 |
| `UNARY` | 116.9 | 94.7 | 0.81 | 22 |
| dense Q6_K and Q5_0 | 277.5 | 257.0 | 0.93 | 21 |
| the router, `ffn_gate_inp` | 52.6 | 40.1 | 0.76 | 13 |
| dense Q4_0 | 402.8 | 396.7 | 0.98 | 6 |
| convolution (`SSM_CONV`, `CONCAT`) | 138.5 | 133.5 | 0.96 | 5 |
| `ADD` | 242.6 | 240.9 | 0.99 | 2 |
| Gated DeltaNet | 582.6 | 582.2 | 1.00 | 0 |
| the other f32 matmuls | 78.0 | 113.0 | 1.45 | -35 |
| the rest | 136.5 | 189.6 | 1.39 | -53 |
| attention | 158.2 | 443.2 | 2.80 | -285 |
| total | 4,202.0 | 4,267.7 (4,307.7 with the time between nodes) | 1.02 | -66 |

- The dense Q4_0 matmuls and the router, 1,143 ms of the fork's lead in section 11, are level now (6 and 13 ms left).
- What the fork still does faster adds to 308 ms, 7% of our graph: the experts (96, though the isolated cases of section 10 had ours ahead, so the two views still disagree), the norms and their multiplies (95; the two trees build this part differently, 281 `MUL` steps against 151 nodes), the dense Q8_0 (48, of it `ssm_out` 27 and the shared experts 17). Each of the others is under 1% of a graph.
- At 128 tokens the graph is 1,299.8 against 1,342.1 ms: experts 487.6 against 483.8, dense Q4_0 133.6 against 128.3, Q8_0 103.2 against 91.2, attention 15.2 against 58.5.

**The pair wait (`--bench-pairwait`: 8 tiles, 1,000,000 rendezvous a tile, three rounds; spine-runtime's executor, `pause` compiled into ours).**

| wait | ns a rendezvous, the three rounds | spread between tiles |
|---|---|---|
| ours (`spacemit_pair_wait`) | 1,357.9, 1,353.0, 1,356.7 | 3 ns |
| SpacemiT's barrier, copied | 109.2, 109.2, 109.2 | 0.1 ns |
| SpacemiT's barrier with our two exits | 161.1, 160.8, 161.5 | 0.3 ns |

- Ours costs 12 times SpacemiT's back to back. The exits are not the reason: they add 52 ns to SpacemiT's.
- What it costs a prompt: tile 0 makes 8,310 pair waits in the 501-token graph's 92 FP16 matmuls, 90 a matmul (66,480 in the KLD run's 736). At the benchmark's figures that is at most 11.3 ms of a 4,202 ms graph with ours and 1.3 ms with SpacemiT's and our exits: 10 ms, 0.24%. An upper bound: in the benchmark all 8 tiles wait back to back, in a matmul a wait comes between kernels.
- The waits of generation (the 1-row matmuls, attention, the experts) are not counted by this run.
- Why ours is slow, found on the Mac the same day: each tile counts its own waits in `by_tile`, and the 8 counters lay in one cache line, so every wait of every tile took the line from the others. With a line a counter the Mac's threads read 196 ns for ours (420 before; SpacemiT's 149, with our exits 175). The K3's figure with the fix is not measured yet.

**The threshold, `f16` from 64, 32 and 16 rows, three rounds in turn (t/s).**

| | from 64 (the default) | from 32 | from 16 |
|---|---|---|---|
| Qwen3-0.6B pp16 | 463.78, 474.49, 464.21 | 474.22, 475.77, 475.45 | 382.67, 381.73, 381.98 |
| Qwen3-0.6B pp32 | 606.75, 605.28, 606.46 | 611.16, 612.26, 604.98 | 602.48, 607.89, 602.10 |
| Qwen3-0.6B pp48 | 652.86, 652.85, 650.69 | 722.16, 716.06, 721.55 | 722.41, 720.35, 723.99 |
| Qwen3-4B pp16 | 83.08, 83.08, 83.09 | 83.33, 83.26, 83.26 | 71.02, 71.13, 71.12 |
| Qwen3-4B pp32 | 108.80, 108.33, 108.63 | 126.96, 126.63, 127.23 | 126.48, 127.09, 126.63 |
| Qwen3-4B pp48 | 106.36, 106.51, 106.10 | 150.19, 150.03, 150.97 | 150.26, 149.75, 149.80 |
| Qwen3.6-35B pp16 | 46.38, 46.28, 46.77 | 46.71, 46.37, 46.14 | 44.14, 44.62, 44.69 |
| Qwen3.6-35B pp32 | 62.25, 63.13, 62.31 | 64.27, 63.72, 63.98 | 64.24, 64.11, 64.10 |
| Qwen3.6-35B pp48 | 69.00, 71.29, 68.87 | 74.25, 71.62, 74.10 | 74.14, 74.18, 74.19 |

- From 32 rows: Qwen3-4B +17% at 32 tokens and +41% at 48; Qwen3-0.6B level at 32 and +10% at 48; the 35B +1 to +3% at 32 and about +6% at 48 (one round of `from 32` reads 71.62 there). Nothing is slower.
- From 16 rows the 16-token prompt is slower on all three (-18%, -14%, -4.5%): the FP16 route dequantizes the whole weight whatever the rows, the integer path does not. Between 17 and 31 rows is not measured.
- So 32 is the better default on every model measured; the 64 came from the plain route on the 35B (section 16), where 32 rows were level.

**The fork on the small models, its first runs on them, three rounds in turn (t/s).** Its weights are in its device's buffers (407.55 and 2,455.18 MiB).

| | ours | the fork |
|---|---|---|
| Qwen3-0.6B pp32 | 604.18, 605.68, 599.72 | 603.56, 595.62, 587.80 |
| Qwen3-0.6B pp128 | 950.82, 954.57, 946.15 | 552.50, 550.12, 550.67 |
| Qwen3-0.6B pp512 | 743.18, 751.17, 748.54 | 371.45, 371.55, 371.21 |
| Qwen3-0.6B tg32 | 58.04, 58.06, 58.06 | 52.82, 53.21, 53.43 |
| Qwen3-4B pp32 | 108.88, 108.54, 108.55 | 121.58, 120.51, 121.38 |
| Qwen3-4B pp128 | 217.19, 216.49, 216.43 | 152.43, 152.92, 152.66 |
| Qwen3-4B pp512 | 202.64, 201.93, 202.97 | 118.80, 118.87, 118.79 |
| Qwen3-4B tg32 | 11.21, 11.21, 11.08 | 10.11, 10.09, 10.11 |

Ours is 1.4 to 2.0 times the fork at 128 and 512 tokens and 9 to 11% faster a token. The one place the fork leads is the 4B at 32 tokens (+11%), which `f16` from 32 rows passes (126.6 to 127.2).

**The fork's own `test-backend-ops` against the CPU (never run before).** It did not finish: 714 cases OK, 19 FAIL, 8,793 not supported, then an abort in a kernel (`rvv_kernels.cpp:2828: GGML_ASSERT(epsilon > 0.0f)`) at the case after `SILU_BACK`, before the norms, the matmuls and attention were reached. The failures: 13 `CPY` (NaN in an f16 copy with a permuted source), 5 `SCALE` with a bias (error 0.73 to 0.75), 1 in the first test's `iq3_xxs` case. So this run says nothing on what costs the fork its KLD.

## 20. The pair wait with a cache line a counter, and the threshold below 32 (`~/pair-wait-and-threshold`, 2026-11-27)

One session on the board at `5c63c8547` (the code of `4ff1c2cf8`: a cache line for each tile's wait count). The plugin of section 19's build was kept before the rebuild and loaded through `GGML_FLAGOS_PROVIDER_PATH` as the "before" arm, so both plugins ran in one session. Build rc 0 (34 steps, without `llama-server`), the rebuilt plugin differs from the kept one. The check program passes with both, 0 FAIL lines, `pairchk` 2/2; the "before" run's loader line names the kept plugin.

**The pair wait (`--bench-pairwait`, as section 19; ns a rendezvous, the three rounds).**

| wait | before | after | spread between tiles, after |
|---|---|---|---|
| ours (`spacemit_pair_wait`) | 1,445.1, 1,445.2, 1,445.2 | 377.0, 368.7, 366.6 | 347 to 397 |
| SpacemiT's barrier, copied | 109.2, 109.1, 109.2 | 109.1, 109.1, 109.1 | 0.1 |
| SpacemiT's barrier with our two exits | 174.6, 174.7, 174.6 | 161.3, 161.1, 161.0 | 0.1 |

- The cache line of section 19 was the cause on the K3 too: ours is 3.9 times faster. It is still 2.3 times SpacemiT's barrier with our exits (369 against 161), and its tiles differ by up to 50 ns where SpacemiT's are level; that rest is not explained.
- The same kept plugin read 1,357 and 161 ns in section 19's session, 1,445 and 175 here: 6 to 8% between sessions or load paths, so the two columns' 161 and 175 are not a difference of the code.

**The models before and after, three rounds in turn (t/s).**

| | before | after |
|---|---|---|
| 35B pp4 | 23.21, 23.69, 23.52 | 23.24, 22.82, 22.96 |
| 35B pp32 | 62.54, 63.08, 62.27 | 63.20, 62.11, 62.15 |
| 35B pp128 | 98.58, 97.40, 96.57 | 98.81, 98.81, 98.45 |
| 35B pp501 | 119.68, 120.19, 117.95 | 119.52, 119.87, 118.22 |
| 35B tg64 | 10.28, 10.30, 10.24 | 10.32, 10.29, 10.19 |
| Qwen3-4B pp32 | 108.59, 108.38, 108.40 | 108.63, 108.78, 108.92 |
| Qwen3-4B pp128 | 216.19, 213.82, 215.64 | 216.22, 214.08, 215.70 |
| Qwen3-4B pp512 | 200.73, 201.40, 201.66 | 199.46, 202.44, 203.97 |
| Qwen3-4B tg64 | 11.18, 11.17, 11.17 | 11.22, 11.21, 11.19 |

Seven of the nine rows' ranges overlap; the 4B's pp32 and tg64 are 0.3% higher after in every round, a size these rounds cannot tell from a plugin placed elsewhere in memory. So no speed moved by an amount three rounds can show. That is what the count predicted: 8,310 waits a 501-token graph times the 1,076 ns saved is 8.9 ms of about 4,190, 0.2%. Generation's waits (1-row matmuls, attention) are still not counted; its speed did not move either. The wait is settled as a cost: the 208 ns left to SpacemiT's would be 1.7 ms a 501-token graph.

**The threshold at 20, 24 and 28 tokens, three rounds in turn (t/s).** Under `f16:32` all three prompts take the integer path; under `:24` the 24- and 28-token prompts take FP16, under `:20` and `:16` all three.

| | `f16:32` (integer) | `f16:24` | `f16:20` | `f16:16` |
|---|---|---|---|---|
| Qwen3-0.6B pp20 | 520.05, 520.13, 519.79 | 519.99, 519.62, 519.36 (integer) | 439.20, 440.61, 442.97 | 438.05, 442.71, 437.64 |
| Qwen3-0.6B pp24 | 554.00, 546.12, 553.03 | 468.37, 470.62, 475.20 | 491.84, 492.65, 490.72 | 492.26, 488.34, 492.95 |
| Qwen3-0.6B pp28 | 581.28, 578.17, 586.68 | 530.50, 536.63, 532.09 | 538.95, 534.09, 532.31 | 529.08, 531.74, 535.19 |
| Qwen3-4B pp20 | 83.00, 83.11, 83.53 | 83.39, 83.28, 83.41 (integer) | 81.65, 81.72, 82.04 | 81.85, 81.62, (85.50) |
| Qwen3-4B pp24 | 87.07, 86.92, 87.21 | 92.95, 92.53, 92.82 | 91.91, 91.96, 92.36 | 92.99, 93.06, (100.66) |
| Qwen3-4B pp28 | 89.72, 89.56, 89.85 | 102.82, 102.68, 102.37 | 101.48, 101.42, 101.41 | 103.24, 103.32, (105.61) |
| 35B pp20 | 50.91, 50.83, 50.83 | 50.60, 50.57, 50.53 (integer) | 49.62, 49.41, 49.36 | 48.88, 49.39, 49.01 |
| 35B pp24 | 53.94, 53.73, 54.08 | 53.43, 53.92, 53.69 | 53.14, 53.33, 53.32 | 53.12, 53.38, 53.12 |
| 35B pp28 | 57.01, 57.72, 57.06 | 57.07, 57.24, 57.03 | 56.91, 57.32, 57.16 | 57.50, 57.15, 56.90 |

- FP16 against the integer path: Qwen3-0.6B slower at all three sizes (-15% at 20, -10 to -15% at 24, -8% at 28); Qwen3-4B -2% at 20, +6 to +7% at 24, +13 to +15% at 28; the 35B -3% at 20, level or 1% slower at 24, level at 28.
- So the rule for a threshold below 32 (faster at 20, 24 and 28 on all three models) is not met: only the 4B gains, the 0.6B loses up to 15%. With section 19's rows (32 tokens: 0.6B level, 4B +17%, 35B +1 to +3%) the crossover is at about 32 rows on the 0.6B, between 20 and 24 on the 4B and between 24 and 32 on the 35B. 32 is the lowest measured threshold that is slower nowhere.
- Two oddities, neither explained and neither changing the verdict. The 0.6B at 24 tokens reads 468 to 475 when the threshold is exactly 24 and 488 to 493 under `:20` and `:16`, the same FP16 path. The 4B's third round under `:16` (in brackets) is 2 to 8% above its other two with errors of 1 to 9 t/s.

**KLD in 48-token batches (`-b 512 -ub 48`, 8 chunks of `wiki.test.raw`, against the CPU on one thread).** Under the default no batch reaches 64 rows, so the default arm is the integer path (no FP16 matmul taken, no stats file); under `f16:32` every batch takes FP16 (the last of a chunk has 32 rows).

| | the default (integer at 48 rows) | `f16:32` |
|---|---|---|
| 35B mean KLD | 0.006506 +- 0.000305 | 0.006497 +- 0.000381 |
| 35B 99.9% / maximum | 0.141692 / 0.191571 | 0.279354 / 0.333546 |
| 35B same top token | 96.912 +- 0.383% | 97.353 +- 0.356% |
| 35B FP16 matmuls | 0 | 8,096, all on the pairs, 0 overflows, 655,776 waits |
| Qwen3-4B mean KLD | 0.003400 +- 0.000155 | 0.001736 +- 0.000230 |
| Qwen3-4B 99.9% / maximum | 0.091462 / 0.120833 | 0.021535 / 0.456467 |
| Qwen3-4B same top token | 97.941 +- 0.314% | 98.137 +- 0.299% |
| Qwen3-4B FP16 matmuls | 0 | 21,704 (18,928 on the pairs), 0 overflows, 1,400,688 waits |

- The rule for 32 as the default (the mean under `f16:32` at most the default's mean plus its error) is met on both: 0.006497 against 0.006811 on the 35B, 0.001736 against 0.003555 on the 4B.
- On the 4B FP16 halves the mean, and its mean and error are section 18's 512-token figures to the last digit. On the 35B the means are equal. The top token agrees a little more often under FP16 on both. The single worst token is worse under FP16 on both (0.33 against 0.19, 0.46 against 0.12), and the 35B's 99.9% point doubles while the 4B's falls to a quarter.
- The tool's own time for a pass, one reading each: 35B 8.14 s (integer) against 7.95 s, 4B 5.62 against 4.25 s.
- The 4B's CPU logits (one thread, 29.5 minutes, perplexity 14.9376 +- 1.14749) stay on the board as `~/kld-wiki/base-8-qwen3-4b.bin`.

## 21. 32 rows as the default on the board, and 512 tokens beside 501 (`~/f16-from-32-rows`, 2026-11-27)

One session on the board with the build of `2cec2e84d` (`default_threshold` 32). Build rc 0 (37 steps, without `llama-server`), the plugin rebuilt and the check program's binary naming 32. The check program passes, 0 FAIL lines, `pairchk` 2/2, and its f16 line reads "32 the default". `FLAGOS_SPACEMIT_DENSE_Q4_0=f16:64` is the arm for the default before.

**`f16:64` against nothing set, three rounds in turn (`-p 16,32,48 -n 0 -r 3`, t/s).**

| | `f16:64` | nothing set | nothing set against `f16:64`, round by round |
|---|---|---|---|
| Qwen3-0.6B pp16 | 472.91, 466.36, 475.96 | 475.34, 474.25, 473.99 | level (the integer path in both) |
| Qwen3-0.6B pp32 | 606.82, 607.24, 598.14 | 612.61, 619.83, 618.85 | +1.0, +2.1, +3.5% |
| Qwen3-0.6B pp48 | 653.03, 652.56, 651.65 | 725.91, 722.34, 725.06 | +11% |
| Qwen3-4B pp16 | 82.71, 83.19, 82.64 | 83.03, 82.75, 83.39 | level |
| Qwen3-4B pp32 | 108.53, 108.50, 108.65 | 128.11, 127.89, 128.45 | +18% |
| Qwen3-4B pp48 | 106.17, 106.60, 106.22 | 151.07, 150.47, 151.06 | +41 to +42% |
| 35B pp16 | 46.40, 46.10, 46.44 | 46.67, 46.03, 45.99 | level |
| 35B pp32 | 63.06, 62.52, 63.71 | 64.24, 64.87, 64.37 | +1.9, +3.8, +1.0% |
| 35B pp48 | 70.98, 71.07, 71.70 | 74.37, 72.01, 74.27 | +4.8, +1.3, +3.6% |

- The default reads what `f16:32` read in section 19 (Qwen3-4B 126.6 to 127.2 and 150.0 to 151.0 there, the 35B 63.7 to 64.3 and 71.6 to 74.3, Qwen3-0.6B 605 to 612 and 716 to 722), and nothing is slower. The 0.6B at 32 tokens, level there, is 1 to 3.5% faster here; the two arms' ranges do not overlap in this run.
- The 35B at 48 tokens reads 72.0 in one round of three, with an error of 0.16 inside that process, as one round did in section 19 (71.62). So it is a process that is 3% slower as a whole, not a noisy repetition; not explained.

**One standard bench a model, nothing set (`-n 64 -r 3`, t/s).**

| | this run | before (section 20, "after"; section 18 for the 0.6B) |
|---|---|---|
| 35B pp4 / pp32 / pp128 / pp501 / tg64 | 22.56 / 63.86 / 98.71 / 119.35 / 10.28 | 22.8 to 23.2 / 62.1 to 63.2 / 98.5 to 98.8 / 118.2 to 119.9 / 10.19 to 10.32 |
| Qwen3-4B pp32 / pp128 / pp512 / tg64 | 127.96 / 212.50 / 196.65 / 11.20 | 108.6 to 108.9 / 214.1 to 216.2 / 199.5 to 204.0 / 11.19 to 11.22 |
| Qwen3-0.6B pp32 / pp128 / pp512 / tg64 | 603.50 +- 11.60 / 965.39 / 750.77 / 58.45 | 602 to 604 / 943 to 963 / 742 to 744 / tg32 57.8 to 58.4 |

Only pp32 is on another path than before (the sizes from 64 rows took FP16 already, generation and 4 tokens stay on the integer path). The rest are one reading each: the 4B's pp128 and pp512 are 1 to 2% under the earlier ranges and the 35B's pp4 is under its range by its own error (0.30), on code that did not change. The 0.6B's pp32 has an error of 11.6 in this reading, so the three rounds above are the figure for it.

**512 tokens beside 501 on the 35B (`-p 501,512 -n 0 -r 3`, one process, after the run).**

| | t/s | one prompt |
|---|---|---|
| pp501 | 118.49 +- 0.14 | 4.228 s |
| pp512 | 113.50 +- 0.30 | 4.511 s |

- 512 tokens are 4.2% slower a token than 501: the 11 more tokens cost 283 ms, 25.7 ms each where the mean token costs 8.4. Both are one graph (`llama-bench` decodes up to 512 tokens in one). One process, so the size is not established and the cause is not known: a step at 512 rows, or a slope from 501. Attention's square law cannot make it (about 0.6 ms a token more at this length, from section 19's 158 ms; an estimate).
- pp512 is the size other projects quote, so 113.5 is the figure to quote for the 35B until this is explained. The fork was measured at 501 only; its pp512 is not known.

**KLD in 48- and 32-token batches (`-b 512 -ub 48` and `-ub 32`, 8 chunks of `wiki.test.raw`, against the CPU on one thread).** Under `f16:64` every batch takes the integer path (no stats file is written); with nothing set every batch takes FP16 (the last of a chunk in 48-token batches has 32 rows).

| | `f16:64` (integer) | nothing set (FP16) |
|---|---|---|
| 35B, 48: mean KLD | 0.006506 +- 0.000305 | 0.006497 +- 0.000381 |
| 35B, 48: 99.9% / maximum / same top token | 0.141692 / 0.191571 / 96.912 +- 0.383% | 0.279354 / 0.333546 / 97.353 +- 0.356% |
| 35B, 48: FP16 matmuls | 0 | 8,096, all on the pairs, 0 overflows, 655,776 waits |
| 35B, 32: mean KLD | 0.006899 +- 0.000391 | 0.006377 +- 0.000450 |
| 35B, 32: 99.9% / maximum / same top token | 0.245147 / 0.364111 / 97.304 +- 0.359% | 0.195795 / 0.645058 / 97.451 +- 0.349% |
| 35B, 32: FP16 matmuls | 0 | 11,776, all on the pairs, 0 overflows, 953,856 waits |
| Qwen3-4B, 48 and 32: mean KLD | 0.003400 +- 0.000155 | 0.001736 +- 0.000230 |
| Qwen3-4B, 48 and 32: 99.9% / maximum / same top token | 0.091462 / 0.120833 / 97.941 +- 0.314% | 0.021535 / 0.456467 / 98.137 +- 0.299% |
| Qwen3-4B: FP16 matmuls, 48 / 32 | 0 | 21,704 (18,928 on the pairs) / 31,552 (27,520), 0 overflows, 1,400,688 / 2,035,584 waits |

- The four runs in 48-token batches print section 20's figures to the last digit (the default here against its `f16:32`, `f16:64` here against its default): the change of the constant changed no arithmetic, and the runs repeat exactly.
- In 32-token batches, new here, the rule holds on the 35B too: 0.006377 against 0.006899 + 0.000391. The single worst token is again worse under FP16 (0.65 against 0.36) while the 99.9% point is lower this time (0.20 against 0.25); the top token agrees as often within the errors.
- On Qwen3-4B both arms print the same figures in 48- and 32-token batches, perplexity included, though the batches differ (11 and 16 a chunk; the FP16 count rises from 21,704 to 31,552). So on that model neither path's result depends on the batch between those two sizes, to the printed digits, and FP16's is the 512-token figure of section 18 as well. The 35B's does depend on it in both arms; its expert matmuls share an activation scale between the rows of a batch, which would do that (an inference, not traced).
- The tool's own time for a pass, one reading each, integer against FP16: 35B 8.05 against 7.76 s (48) and 8.75 against 8.61 s (32); 4B 5.63 against 4.25 s and 5.51 against 4.78 s.

**The acceptance script (`--skip-server --ppl-text ~/wiki.test.raw`).** 56/56 on Qwen3-0.6B and 56/56 on Qwen3-4B.

| | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| perplexity, provider / CPU | 23.0355 / 23.0657 (-0.13%) | 14.9383 / 14.9376 (+0.00%) |
| KLD against the CPU, 512-token batches | 0.002826 +- 0.000146 | 0.001736 +- 0.000230 |
| KLD against the CPU, `-ub 1 -b 512` | 0.003130 +- 0.000102 | 0.001767 +- 0.000096 |
| the script's `llama-bench`, pp128 / tg128, t/s | 956.91 / 56.94 | 213.62 / 11.12 |
| the same on the CPU (8 X100 threads) | 153.51 / 27.62 | 21.00 / 5.43 |

Perplexities and KLDs are section 18's to the last digit, as they should be: the script's batches are 512 tokens and 1, on neither side of the change. So the script shows that nothing else moved; the 32 to 63 rows themselves are covered by the check program (31 and 32 rows on both sides) and by the KLDs above.

**So 32 rows stay the default.** Open from this run: the 4% between 501 and 512 tokens on the 35B.

## 22. The dip at 512 tokens (`~/prompt-512-step`, 2026-11-27)

No rebuild (the build of `2cec2e84d`; only notes changed since). Three rounds in turn on the 35B: ours at eight sizes up to 512, ours at five sizes around 512 in one graph (`-ub 1024`), the fork at three sizes; then one run of ours with the op tables on, then the two small models. All `-n 0 -r 3`, every rc 0.

**The 35B, t/s, the three rounds.**

| tokens | ours | against its 501 | ours, `-ub 1024` | against its 501 | the fork | against its 501 |
|---|---|---|---|---|---|---|
| 448 | 119.25, 118.31, 117.71 | +0.4, -1.2, -0.4% | | | | |
| 480 | 119.40, 121.24, 120.51 | +0.6, +1.2, +2.0% | | | 120.23, 116.20, 117.29 | -0.4, 0.0, -1.5% |
| 496 | 118.29, 119.90, 119.60 | -0.4, +0.1, +1.2% | | | | |
| 501 | 118.72, 119.75, 118.15 | | 118.01, 118.48, 118.48 | | 120.73, 116.18, 119.08 | |
| 504 | 119.21, 120.73, 119.29 | +0.4, +0.8, +1.0% | | | | |
| 508 | 116.65, 118.75, 117.39 | -1.7, -0.8, -0.6% | | | | |
| 511 | 113.30, 117.02, 114.62 | -4.6, -2.3, -3.0% | | | | |
| 512 | 112.41, 117.24, 112.30 | -5.3, -2.1, -5.0% | 112.54, 114.97, 112.97 | -4.6, -3.0, -4.7% | 116.63, 113.74, 113.94 | -3.4, -2.1, -4.3% |
| 513 | | | 115.85, 116.54, 115.49 | -1.8, -1.6, -2.5% | | |
| 528 | | | 119.54, 118.96, 118.94 | +1.3, +0.4, +0.4% | | |
| 576 | | | 119.96, 119.80, 119.59 | +1.7, +1.1, +0.9% | | |

- It is a dip, not a step: speed is level from 448 to 504, falls from 508, is lowest at 512 and is back by 528. In the `-ub 1024` rounds a 512-token prompt takes 4.45 to 4.55 s and a 528-token one 4.42 to 4.44 s.
- The fork has the same dip (2 to 4% at 512). So section 21's 4% is not ours alone, and at 512 tokens the two are level as they are at 501: ours 112.3 to 117.2, the fork 113.7 to 116.6.
- The depth differs between processes: about 5% in four of ours and 2 to 3% in two, each steady inside its process (errors of 0.15 to 0.3).
- Graphs of 513, 528 and 576 rows, the first of more than 512 rows on the board, ran without a failure.

**Where the time goes: the op tables at 496, 501 and 512 rows (one process; ms a graph).** That process read pp512 118.18 against pp501 119.18, so it is one of the shallow ones: 36 ms over what 501 rows scale to (4,326.3 against 4,290.1), where the deep ones lose 210 to 240.

| steps | 501 rows | 512 rows | 512 against 501 scaled by rows |
|---|---|---|---|
| `UNARY` (170) | 116.3 | 160.7 | +42 ms, 1.35 times |
| `GLU SWIGLU` (80) | 50.6 | 80.9 | +29 ms, 1.56 times |
| `MUL` on the expert rows (30 `node_*` steps) | 31.1 | 46.5 | +15 ms, 1.47 times |
| `ADD` (420) | 249.6 | 260.8 | +6 ms |
| `CPY` (30) | 1.6 | 6.9 | +5 ms |
| the matmuls, attention, the Gated DeltaNet | | | at or under the scaled figure |
| `SSM_CONV` (30) | 102.4 | 80.0 | -25 ms (88.9 at 496: its 501 is the odd one) |

- The loss is in the elementwise steps, not in the matmuls. Inside them the tiles differ at 512 rows where they are level at 501: `UNARY` on 512 rows 74.9 to 101.6 ms between tiles (69.7 to 73.1 at 501), a `MUL` on the expert rows 0.67 to 1.89 ms (1.057 to 1.058), and the graph's imbalance rises from 102.5 to 138.5 ms.
- What a deep process loses, and where, is not in this table.

**The small models, ours (t/s).**

| tokens | Qwen3-4B | Qwen3-0.6B |
|---|---|---|
| 448 | 211.57, 210.32, 209.02 | 813.47, 802.70, 815.63 |
| 480 | 209.07, 209.22, 208.98 | 789.92, 729.04, 795.51 |
| 501 | 204.86, 205.23, 203.95 | 764.37, 704.99, 761.73 |
| 511 | 201.27, 200.23, 201.30 | 819.23, 700.67, 739.44 |
| 512 | 200.15, 198.09, 201.15 | 814.22, 749.37, 740.94 |

- Qwen3-4B slopes down evenly (5% from 448 to 512, 511 and 512 alike): no dip of its own at 512 shows, though sizes past 512 were not run. Qwen3-0.6B varies by up to 10% between processes at these sizes (one round reads 511 and 512 above 501), so it shows nothing either way.

**The likely cause (an inference from these figures and an older measurement; not tested).** The elementwise kernels, ours copied from the fork's, give each tile one block of rows (`dr * ith` in `plugin/flagos-spacemit-rvv-kernels.cpp`). At 512 rows of the 35B's width (2,048 floats) a tensor is exactly 4 MiB and a tile's block 512 KiB, so the 8 tiles stream from addresses a power of two apart; the expert steps' tensors are 8 and 32 times as many rows. That is the case `k3-hardware.md` records from 2026-10-13 on the old unit ("Huge pages and power-of-two strides": 8 AI cores streaming from starts a multiple of 2 MiB apart ran 2.1 to 2.2 times slower on huge pages) with its rule to stagger the tiles' starts. It fits the dip being in both backends, being at one size, and the 4B (width 2,560: 640 KiB blocks at 512 rows) not showing it. Not explained by it: why 511 and 513 rows lose too, and why processes differ.

**Why it matters beyond the benchmark.** `llama-bench`, `llama-server` and the other tools decode a long prompt in batches of 512 tokens by default, so every 512-token batch of a long prompt is at the bottom of the dip; a batch size beside it (`-ub 528` or 576) would not be. Not measured yet.

## 23. A batch size beside 512, and the dip at 1,024 (`~/long-prompt-batch-sizes`, 2026-11-27)

No rebuild (the build of `2cec2e84d`). Every rc 0; `-n 0 -r 3` throughout.

**A 2,048-token prompt by batch size, three rounds in turn, the 35B (t/s).**

| batch (`-ub`) | ours | against its 512 | the fork | against its 512 |
|---|---|---|---|---|
| 512 (the tools' default) | 107.83, 107.25, 107.12 | | 96.46, 93.83, 92.48 | |
| 528 | 108.68, 109.58, 109.23 | +0.8, +2.2, +2.0% | | |
| 576 | 108.94, 108.65, 109.69 | +1.0, +1.3, +2.4% | 98.77, 99.97, 100.10 | +2.4, +6.5, +8.2% |

- A batch beside 512 is a small gain for us on a long prompt, 1 to 2% (18.7 to 18.9 s against 19.0 to 19.1 s), and a larger one for the fork, 2 to 8%.
- On a 2,048-token prompt ours is 12 to 16% faster than the fork at the default batch and 9 to 10% at 576: the first long-prompt comparison with the fork on this board.

**Around 256 and 1,024 tokens in one graph (`-ub 1100`), ours, the 35B (t/s).**

| tokens | the three rounds | against the mean of its neighbours |
|---|---|---|
| 248 | 111.64, 111.08, 110.48 | |
| 256 | 108.13, 110.21, 111.20 | -2.7, -0.6, +0.7% |
| 264 | 110.53, 110.70, 110.48 | |
| 1,008 | 120.05, 119.66, 120.06 | |
| 1,024 | 114.77, 113.22, 112.61 | -4.3, -5.5, -6.0% |
| 1,040 | 119.76, 119.92, 119.66 | |

- 1,024 tokens dip in every round, as section 22's likely cause predicted before the run; 256 dips in one round of three. 1,008 and 1,040 rows in one graph ran for the first time and read what 501 does (120 t/s).

**The op tables at 501 and 512 rows in three processes (ms a graph).** Two of the three are deep ones (pp512 6.2 and 4.5% under pp501), one shallow (2.2%).

| | process 1 (deep) | process 2 (shallow) | process 3 (deep) |
|---|---|---|---|
| 512 rows over 501 scaled by rows, the whole graph | +281 | +97 | +201 |
| the elementwise steps | +324 | +120 | +220 |
| the matmuls | -37 | -1 | -11 |
| `UNARY` (170), against scaled | 1.68 times | 1.38 | 1.54 |
| `GLU SWIGLU` (80) | 2.41 times | 1.66 | 2.17 |
| `MUL` on the expert rows (30) | 2.08 times | 1.59 | 2.01 |
| `RMS_NORM ssm_norm` (30) | 1.98 times | at scale | 1.85 |

- The whole dip is in the elementwise steps in the deep processes too; the matmuls, attention and the Gated DeltaNet stay at or under their scaled figures. The factors of about 2 are the size `k3-hardware.md` records for the power-of-two strides.
- A deep process differs from a shallow one by more of the same steps slowing (the recurrent layers' norm only in the deep ones), not by another kind of step.

**Qwen3-4B past 512 in one graph (`-ub 1024`, t/s).**

| tokens | the three rounds | against its 501 |
|---|---|---|
| 501 | 203.64, 204.07, 204.17 | |
| 512 | 195.12, 197.01, 198.05 | -4.2, -3.5, -3.0% |
| 528 | 199.40, 199.56, 200.39 | -2.1, -2.2, -1.9% |
| 576 | 192.94, 192.40, 192.69 | -5.3, -5.7, -5.6% |

- The 4B dips at 512 as well, by 2 to 3% under the line from 501 to 528 (528 tokens are faster a token than 512), on top of its slope. Section 22 read no dip there; it had no sizes past 512. Its blocks at 512 rows are 640 KiB, not a power of two, so the cause as section 22 words it (blocks a power of two apart) is too narrow: a multiple of a smaller power of two is enough to lose some.

**What stands.** The dip is real, at 512 and 1,024 rows on the 35B and smaller on the 4B, in the elementwise steps, in both backends. The cause is still an inference: the direct test is the staggered blocks, not built. On a long prompt it costs us 1 to 2%.

## 24. `llama-server` rebuilt on this unit, the server checks and the TTFT harness (`~/llama-server-rebuilt`, 2026-11-27)

No code change (`59a1aaea7`, documents only since the build of `2cec2e84d`). The first run of `llama-server` on this unit.

**The build.**

| step | result |
|---|---|
| `llama-server` as the board was configured | rc 0, 31 steps. `UI: downloading from b10669`, "archive verified and extracted": the fork's commit count is now 10,669, upstream has a build of that number, and its page is the old layout |
| the base's archive (`b10375`) downloaded on the board | 3,076,825 bytes, the recorded sha256 (read again on the Mac's copy) |
| rebuilt with it in `tools/ui/dist` | rc 0, 5 steps. `UI: using pre-built assets from /root/llama.cpp-flagos/tools/ui/dist` |
| built once more | 1 step (the UI step looks and does nothing): no compile, no link, no download |

- The failure of 2026-11-26 did not come back, and could not: it belongs to a commit count upstream has no build for (10,646 then), and the count has moved on. That it passed at 10,669 on another build's page is what item 139's cause says would happen. The cause itself stays shown on the Mac only.
- Before the fix the board's server carried the page of upstream's build 10,669, which is not this tree's. With the archive in place the page is the base's at every commit count, and the build asks the network for nothing.
- The block's own line for the page's name in the binary was printed on the board and is not in the copied files; the acceptance runs below were gated on it.

**The acceptance script with its server checks: 62/62 on Qwen3-0.6B and on Qwen3-4B** (56 of them the checks of section 18, passing as before). The six server checks:

| | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| up, every layer and the KV cache on the provider, 4 slots, 1 split | 3.0 s, 29/29 | 6.0 s, 37/37 |
| the greedy reply against `llama-completion` | 153 characters equal | 150 equal |
| a follow-up on its cache against from scratch (KLD, limit 0.1) | 0.000392 | 0.005875 |
| interleaved conversations from the prompt cache (KLD) | 0.000179, 0.000069 | 0.000002, 0.000005 |
| KV shifting (`--cache-reuse`) | 40 tokens moved, `cache_n` 71 of 72 | the same |
| a follow-up ending in 4 tokens (KLD) | 0.000073 | 0.000002 |
| a fresh 500-token prompt to the first token (one request) | 674 ms | 2,470 ms |
| a generated token | 24.0 ms | 95.6 ms |
| a 12-token follow-up with the cache, and its 59 tokens from scratch | 34 and 84 ms | 166 and 371 ms |
| a 4-token follow-up with the cache, and its 23 tokens from scratch | 25 and 51 ms | 113 and 318 ms |

- The 4B's 2.47 s is inside the target of `batch-one.md` section 5 (a fresh 500-token prompt at most 4 s). The old unit's baseline was 5.21 s (2026-11-09, older code and another board, so not a like-for-like pair).
- `llama-bench` in the same runs: 968.61 and 57.31 t/s (pp128, tg128) on the 0.6B, 214.69 and 11.16 on the 4B, as in sections 18 and 21.

**The TTFT harness, `--quick`, Qwen3-0.6B: 11/11.** Client ms, p50 of 2:

| tail (tokens) | depth 0 | depth 256 |
|---|---|---|
| 1 | 21.2 | 27.3 |
| 4 | 26.1 | 31.9 |
| 9, 12 | 32.2 | 42.9 |
| 63 | | 113.4 |
| 64 | 82.6 | 110.2 |
| 65 | 89.5 | |

- 64 generated tokens: 56.93 t/s from an empty cache, 42.74 from depth 256 (2 runs each).
- Chat, three turns: thinking off, the first content after 98, 228 and 200 ms; thinking on, the first reasoning after 131, 130 and 138 ms.
- `restart`: after a `kill -9` one second into a request the next server was up in 4.0 s with no "could not get" line and the same speed (55.93 against 55.89 t/s).
- `second`: a second server beside the first came up after 63.1 s, its request was refused after 30.0 s with "Compute error." in the stream, and it answered with the first one's text once the first had stopped. The old unit did the same (65.1 s, 30.0 s).
- `cancel`: a request closed after 0.5 s is not stopped, and the next one waited 406 and 413 ms for it. Cancel is not in the design (decided 2026-11-05).
- Against the old unit's baseline (2026-11-08, older code, another board): depth 256 with a 12-token tail 96.3 then, 42.9 ms now; a 63-token tail 467.0 then, 113.4 now; the 1-token tail 27.1 then, 27.3 now.

**Not run on this unit yet.** The harness's `control` and `alongside` (there is no `build-ime` here), the read of marked TCM blocks after a kill (no `~/tcmtest`), the harness at full size on either model, and the 35B through the server.

**Before the 35B: the six checks on the Mac with a small model of its family** (Qwen3.5-0.8B, the simulated device). Five pass, with the follow-ups and the prompt cache reused (`cache_n` 50, 51, 52 and 21). KV shifting cannot: llama-server prints "cache_reuse is not supported by this context, it will be disabled" at its start, since this family gives a token several positions (`llama_kv_cache::get_can_shift`). `spacemit_check.py` now reports that check as not applicable when the server prints the line, and counts five. Whether the 35B's server prints it is an inference from the family until its run.
