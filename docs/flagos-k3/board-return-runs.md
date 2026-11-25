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
3. The offload check: one short completion with `-dev SPACEMIT0 -ngl 99 -lm none -lv 4`. Report the lines that say how many layers went to the device, the graph nodes and the graph splits. If the layers are not on the device, stop: the figures would again be its CPU's.

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
- Sizes: 1 token (`-p 0 -n 16`), then 4, 32, 128 and 501 tokens (`-n 0`), each `-r 2`, one size a run.
- `fork-trace.py` turns a trace into the same kind of table as ours: per graph size, a row per op and weight (for example `MUL_MAT attn_qkv q4_0 m501`), the wall time from the first worker's begin to the last worker's end, and the time between one node's end and the next one's begin (the fork's wait between nodes). The first graph of each size is dropped as the warm-up.

The profiler's own cost is measured, not assumed: for each size the profiling build runs once with `GGML_TRACE` unset and once with it set, and the plain build's figure from round C stands beside them. The model's graph has 3,727 nodes by ggml's count (the fork's device may run fewer as its own); each event formats up to 1 KB of text. My estimate is that this is small against a 5-second graph and large against a 100 ms one, so the one-token table is reported with the measured cost per event taken off, and the isolated one-row cases of D1 are the firmer evidence there. A 501-token trace is about 20 MB per graph by estimate; the traces stay on the board and only the tables are copied.

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
