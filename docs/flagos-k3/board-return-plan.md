# When the board is back: the revisit plan

Written 2026-11-24, while the K3 is away for an upgrade. It holds what was found and decided on 2026-11-23 and 24 and the steps to take in order. Quote a step to start it. The detail behind each line is in `plan.md` ("The dense Q4_0 matmuls: the fork's FP16 path, a reference run first", "The third cleanup", "What came from the documents before the deck") and `progress.md` items 111 to 114.

## 1. Targets

1. Know whether SpacemiT's own backend is faster than our provider on Qwen3.6-35B-A3B, overall and op by op.
2. Where it is faster, bring that driver's way of working into our plugin, keeping the FlagOS core and plugin split.
3. Confirm on the real kernels the cleanup that so far ran only on the Mac.
4. Settle the server's checkpoint setting for chat from the run already made.

Speed targets, agreed for Qwen3 on 2026-11-06 and not yet set for the 35B: a fresh 500-token prompt in at most 4 s (5.4 s now), a chat turn at depth 4096 in at most 1 s.

## 2. Decisions made so far

- The only design reference is the deck, `FlagOS端侧与车载智算.pdf` pages 6 to 8. The mentor's August document is obsolete and not to be cited. The V2.0 design PDF is a reference, not a rule book.
- The device type is IGPU. It is not part of the comparison.
- The comparison op by op is done whatever the overall result is.
- If SpacemiT's driver is faster on the board, our plugin uses it or learns from it. The Mac's checks are for arithmetic, not for choosing a design by speed.
- The 8-row expert path is the default; its 3 ms cost on a 4-token graph is accepted for now, and `FLAGOS_SPACEMIT_EXPERT_Q4_0=plain` stays as the way back.
- The cleanup is committed (`fd757088f`): nine switches of settled questions gone, the conv pair check on a rounding bound. Kept: the plugin table's unused members.
- Three specs are written and not approved beyond the reference run: the checkpoint setting, the dense FP16 path (its Part A approved), the expert quantize step's deal.

## 3. What we know about the fork before measuring

- Version: `spacemit-com/llama.cpp`, branch `mtmd-backend`, commit `b847bf396`. Its source archive is on the Mac as `~/smt-b847.tar.gz`.
- It builds on the board with spine-runtime 0.6.3 (173 steps, done once on 2026-11-23).
- Its backend is off unless `GGML_SPACEMIT_BACKEND=1` is set.
- Its device is of type ACCEL and takes only its own buffers. Without `-dev SPACEMIT0 -ngl 99` its layers and KV cache stay on the CPU. Both were missing or wrong in the first blocks.
- Its tools take `-lm none`, not `-mmp 0`.
- Like ours it makes one launch per graph. Its unit is the graph node, with waits inside an op; ours is the step, with waits between steps.
- Where the two differ most: its dense prompt matmuls run in FP16 with 16 rows a call and paired cores sharing TCM; ours run in 8-bit on paths A, B and C. Its expert kernels are the ones we already took.
- Not yet measured on this version: anything. The first attempt measured its CPU fallback by mistake.

## 4. Checklist, in order

**Step 0. What survived the upgrade (read-only, minutes).**
- Run the status check: the date, the free space, which of our files exist, the repo's commit, any process holding `/dev/tcm`.
- Note the spine-runtime version. If it changed, rerun the launch-cost test before trusting the old figures (a launch 13 to 15 us, a barrier 0.4 us).
- Decide from the result: restore what is missing (the model, the KLD base, `segv.so`, `ppl.txt`), or go on.

**Step 1. Our own build and the cleanup's confirmation (about 40 minutes).**
- One bundle of the whole branch if the repo is gone, else of the commits since `d5e9ae0e8`.
- Build, then the check program with the device: "all checks passed" with the real kernels.
- The full table: pp32, pp128, pp501, pp2048, tg512; one completion against the float run's text; the KLD of 2 chunks (0.005115 expected).
- The two mutations the Mac cannot catch are covered by this run's checks: the tiled attention at 9 to 63 tokens and the Gated DeltaNet's forms.

**Step 2. The fork against ours, overall (about 1.5 hours with its build).**
- Build the fork in a folder of its own, or reuse `~/llama.cpp-smt-b847` if it survived.
- Run it with `GGML_SPACEMIT_BACKEND=1 -dev SPACEMIT0 -ngl 99 -lm none`; stop at once if its device does not come up.
- Both backends in turn, three times: pp4, pp32, pp128, pp501, tg64. Report each one's graph splits.
- On the fork: one completion and the KLD of 2 chunks.

**Step 3. The fork against ours, op by op (about 1 hour; done whatever step 2 shows).**
- The op test's timing mode on both backends: dense Q4_0 matmul at 1, 4, 32, 128 and 512 rows; the expert multiply at the same sizes; attention at 1, 26 and 128 tokens over a few depths; Gated DeltaNet.
- A table of the two, with the ratio for each cell.

**Step 4. Decide and port (one op at a time).**
- For each op where the fork wins by more than the runs differ: a spec, then its driver behind a switch, then a run of the two in one session with the KLD.
- Expected first candidate: the dense FP16 matmul. Its port needs waits inside a step or two more steps, and a way to unpack our weight layout to FP16.
- The rule that no kernel waits inside a step is open to change here; the check before the first write and the failure status stay.

**Step 5. The smaller items, as board time allows.**
- Copy `~/ck` if it survived and judge the checkpoint setting; else rerun that chat stage (16 minutes).
- The expert quantize step's deal by rows (spec written).
- A build script of its own in place of `t1-experiments.sh`'s build step, whose other steps name a removed switch.

## 5. Files to build when a step is quoted

| Step | To build | Where |
|---|---|---|
| 0 | the status check (ready, in the message of 2026-11-23) | given inline |
| 1 | the build and confirmation block | given inline |
| 2 | the overall comparison block, with the device selected | given inline |
| 3 | the op-by-op timing block and a script that tabulates both backends' output | `docs/flagos-k3/scripts/` |
| 4 | per op: the spec in `plan.md`, the driver in `plugin/`, its check in `tests/check_spacemit.cpp` | by a coder, verified |
| 5 | the build script | `docs/flagos-k3/scripts/` |

Result folders on the board are named for what they hold (`~/cleanup-confirm`, `~/fork-overall`, `~/fork-per-op`), one name a purpose.

## 6. Open, for the user

- Targets for the 35B itself.
- Whether `--reasoning-preserve` becomes the documented chat setting (it changes the answers).
- Whether the checkpoint setting is wanted at all, given that an edited or regenerated message then reprocesses the conversation.
