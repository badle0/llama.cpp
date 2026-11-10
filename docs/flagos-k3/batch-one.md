# Batch-one inference on the K3: what to test and what to build

Written 2026-11-04 on the user's request. This is analysis: nothing here is implemented, and the implementation waits until S2 is finished (user, 2026-11-04). Figures are Qwen3-4B Q4_0 on the K3 unless marked. "Measured" means a recorded run, "by reading" means the code was read but not run, "estimate" is my arithmetic from measured rates and each estimate names the test that would settle it.

**Term (user, 2026-11-04).** Batch-one inference is one active request or sequence at a time. It says nothing about tokens per kernel call: a prompt batch of 512 rows and a generated token of 1 row are both batch-one. Older documents' "batch size 1" usually means one row per graph.

**Criteria, in the user's order:** lowest TTFT first, then tg512 and pp.

## 1. Conclusions

1. **A batch-one request crosses three regimes with three different bounds** (§2), so one optimization does not serve all three criteria: tg512 is bound by DRAM bandwidth, a chat turn's TTFT by one-chunk attention at depth, and a fresh prompt's TTFT and pp by compute.
2. **TTFT is not monotonic in prompt length today.** At 4096 cached positions 63 tokens take 17.2 s and 64 tokens 1.67 s (measured rates). Below 64 tokens, decoding less can cost more. Measured as the time to the first token by the TTFT harness, 2026-11-09: 17.6 s and 1.67 s (`plan.md` "TTFT harness spec", full K3 runs).
3. **A real Qwen3 chat turn does not decode "12 tokens".** Qwen3's template drops the think block from earlier assistant turns, so the cached prefix breaks at the start of the last reply and the server re-decodes that whole reply (measured on the Mac, §7): with thinking off 109 tokens where the new turn is 26, with it on 138 where the new turn is 22. Earlier notes' "12-token turn, 3.3 s" holds only when the prefix is kept. The runner's follow-ups use `/completion`, so no acceptance check sees this.
4. **The tails that matter are about 20 to 130 tokens, and the KV-head split does not reach most of them.** On Qwen3-4B it takes 1, 2 or 4 tokens in one read per call, and 3 and 5 to 8 with one read per token, a quarter of today's reads. The threshold experiment for 9 to 63 tokens is therefore the top kernel item for TTFT, ahead of the KV-head result.
5. **The largest TTFT wins need no kernel work.** Restoring a saved system prompt turned 21.6 s into 1.73 s on the Mac's simulated device with the same reply text (§7).
6. **tg512 follows one law:** 2.51 GB of weights per token at 28.7 GB/s. Only fewer bytes, more bandwidth, more rows per read (speculation) or cheaper attention at depth can move it. Step overheads total 1.8% of a token.
7. **Two items touch S2's design** (§5): how the 11.0 s start-up divides, which decides whether the zero-copy entries are worth building, and a cancel check in the replay loop, which S2 moves into the plugin.

## 2. Where a request spends its time

TTFT = process start (when cold) + template and tokenize + cache restore + the sum over ubatches of a graph's time at (rows, n_kv) + first sampling. The server adds nothing measurable (`llama-completion` 5242 ms, server 5247 ms on the same prompt), and sampling takes about 0.5 ms per token.

| Regime | Rows per graph | Bound by | Evidence | Serves |
|---|---|---|---|---|
| Generation | 1 | DRAM: 2.51 GB at 28.7 GB/s, 87.3 ms; attention adds 8.6 µs per cached position | bottleneck suite, `build.md` §7 | tg512 |
| Few-token | 2 to 63 | at depth, one-chunk attention reading a KV head once per token and query head, about 0.27 s per token at n_kv 4352; shallow, the matmuls | pp12 and pp63 at depth 4096 | TTFT of chat turns |
| Prefill | 64 and up | compute: kernels at about 0.96 int8 TOPS, attention far slower than the matmuls | rows table below | TTFT of fresh prompts, pp |

Time per graph at depth 0 (measured):

| Rows | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|
| ms | 87.3 | 105.8 | 111.0 | 131.6 | 197.6 | 309 | 549 | 1,198 | 2,476 | 5,310 |
| t/s | 11.45 | 18.91 | 36.05 | 60.79 | 80.99 | 103.55 | 116.55 | 106.85 | 103.37 | 96.39 |

TTFT today:

| Request | Decoded | Cached before | TTFT | Basis |
|---|---|---|---|---|
| Process start, added to any request below | | | +11.04 s (0.6B 4.29 s) | measured, server up (`plan.md` "The start-up split") |
| Fresh prompt | 500 | 0 | 5.25 s | measured, server |
| Fresh prompt | 4,096 | 0 | about 75 s | estimate, interpolated between pp512 at depth 0 and 4096 |
| Follow-up, prefix kept | 12 | 47 | 182 ms | measured, server |
| From scratch | 59 | 0 | 638 ms | measured, server |
| Regenerate | 1 | 4,096 | 129 ms | measured, tg32 |
| Follow-up, prefix kept | 12 | 4,096 | 3.27 s | measured rate, pp12 |
| Follow-up, prefix kept | 26 | 4,096 | about 7 s | estimate at pp12's rate |
| Chat turn after a short reply, default server | 38 | 4,096 | about 10 s | estimate at pp12's rate |
| Prompt tail | 63 | 4,096 | 17.2 s | measured rate, pp63 |
| Prompt tail | 64 | 4,096 | 1.67 s | measured rate, pp64 |
| Chat turn after a longer reply, default server | 110 | 4,096 | about 2.9 s | estimate at pp64's rate |

Measured by the TTFT harness, 2026-11-09 (client ms p50; `plan.md` "TTFT harness spec", full K3 runs):
- fresh prompts: 500 tokens 5.21 s, 4,095 tokens 73.2 s;
- at depth 4096: 1 token 136 ms, 12 3.34 s, 26 7.25 s, 38 10.56 s, 63 17.58 s, 64 1.67 s, 110 3.00 s.

Every estimate in the table is within 6% of these figures.

Generation against depth (tg32, measured): 89.3, 102.6, 111.4, 129.0 and 164.2 ms per token at 0, 1,024, 2,048, 4,096 and 8,192 cached positions. tg512 itself has never been run; from depth 0 it should be about 10.5 t/s and from depth 4096 about 7.6 (estimates, integrating this line). Measured 2026-11-09: 10.56 and 7.53 t/s, the p50 of ten runs each.

Depth's cost, measured by the op counters (2026-11-05, `plan.md` "attention at depth by the op counters"): attention is 13.9%, 31.5% and 46.3% of a one-row graph at n_kv 1,280, 4,352 and 8,448, and 37.0%, 67.4% and 79.8% of a 512-row graph at n_kv 1,536, 4,608 and 8,704; everything else stays at 88 ms and about 4.65 s. The KV-head split, the default since 2026-11-05, takes 2 to 8 tokens at depth 4096 from 3.18 to 3.51 t/s to 9.44 to 17.71 t/s (pp4: a graph of 1,166 to 226 ms; `plan.md`); the table above predates it, and 9 to 63 tokens stay as they are.

## 3. Cases that test and challenge batch-one

None of these is an acceptance check today except where noted. Grid sizes are proposals.

**Cold and idle**

| # | Case | What it challenges | Known today |
|---|---|---|---|
| C1 | Process start to first token, page cache warm and cold (evict only the model file, no system setting) | Load, repack, warm-up, core grant | Measured (`plan.md` "The start-up split"): warm 11.04 s and 4.29 s, cold 13.37 s and 4.70 s; the weights stage 8.47 s and 2.11 s, 97% user time; the fit and vocabulary 1.78 s on both; a first request adds 3 ms |
| C2 | First request after minutes idle | Cold caches, frequency, what idle workers cost | Not measured |

**Fresh prompts**

| # | Case | What it challenges | Known today |
|---|---|---|---|
| C3 | TTFT against prompt length: 1 to 5, 8, 9, 16, 32, 63, 64, 65, 128, 256, 511 to 513, 575, 576, 1024, 2048, 4095 | Path thresholds, n_kv steps at multiples of 256, ubatch tails | Measured 2026-11-09 by the TTFT harness (`plan.md`). At depth 0 on Qwen3-4B: 1 token 94 ms, 63 676 ms against 556 ms for 64, 500 5.21 s, 4,095 73.2 s. On Qwen3-0.6B: 23 ms, 121 against 98 ms, 955 ms, 19.6 s |
| C4 | New conversations sharing a system prompt, decoded against restored | The snapshot lever | Mac only, §7 |

**Depth and chat**

| # | Case | What it challenges | Known today |
|---|---|---|---|
| C5 | Grid: tails of 1, 2, 3, 4, 8, 12, 26, 38, 63, 64, 110 × depths 0, 256, 1024, 2048, 4096, 8192 | The few-token regime; the main TTFT map | Measured 2026-11-09 on both models (the harness; `plan.md`). At depth 4096 on Qwen3-4B: 12 tokens 3.34 s, 26 7.25 s, 38 10.56 s, 63 17.58 s, against 1.67 s for 64 and 3.00 s for 110. Per cached position, 12 to 63 tokens cost about 16 times what 64 and 110 do, on both models. 3 tokens on both, and 8 on Qwen3-4B, take one pass per token (the KV-head split's rule) |
| C6 | A real client: 20 turns through `/v1/chat/completions`, thinking on and off, reading `cache_n` and `prompt_n` per turn | Prefix breaks, TTFT growth over a conversation | Measured on the K3 2026-11-09 (`plan.md`). Thinking off decodes the last reply again every turn (533 to 537 tokens on Qwen3-4B); the first event reached 36.7 s at turn 17 on Qwen3-4B and 12.5 s at turn 19 on Qwen3-0.6B. Qwen3-4B's run stopped at turn 18 on the harness's context. Run again the same day with `-c 12288`: 20 of 20 turns in both modes, turn 19 at 42.0 s. A long turn's last 21 to 25 tokens, the tail after its 512-token ubatch, take 32 to 40% of its time in turns 7 to 19 (inference from the grid's rates, which match turns 2 to 20 within 4%). Thinking on, turns decode 16 to 335 tokens, few-token tails at depth: 19 tokens after 624 took 935 ms on Qwen3-4B. Outside `prompt_ms`: 35 ms at turn 1 and 109 to 148 ms at turn 20, growing 4 to 7 ms a turn; on turns keeping under half the cache, 44 to 90 ms more for the prompt cache's save |
| C7 | Regenerate; edit an early message; branch back | Prefix loss, `--cache-reuse` under edits | One shift check, depth about 100 |
| C8 | Two conversations interleaved at depth | Prompt-cache save and load of up to 576 MiB of KV | Only at about 60 positions |
| C9 | Context full | Shift, truncation or refusal: a TTFT spike or an error | Not measured in the server |
| C10 | Cancel during prefill at depth, then a new request | Time the next request waits | By reading, no abort path: the provider exposes no `ggml_backend_set_abort_callback`, the server sets none, and it reads its queue between `llama_decode` calls of up to `-b` 2048 tokens, so up to about 57 s at depth 4096 (estimate). Measured 2026-11-09 (`plan.md`): 61.4 s on Qwen3-4B and 18.4 s on Qwen3-0.6B, after a close 2 s into a 2048-token tail at depth 4096 |

**Generation**

| # | Case | What it challenges | Known today |
|---|---|---|---|
| C11 | tg512 from depths 0, 500, 4096, ten runs back to back, speed per 100 tokens, reading the thermal sensors | The criterion itself, sustained load, throttling | Measured 2026-11-09 (`plan.md`): p50 10.56, 9.83 and 7.53 t/s from depths 0, 500 and 4096 on Qwen3-4B; 46.58, 36.98 and 19.16 on Qwen3-0.6B. The ten runs lie within 0.4%, at 64 to 78 °C: no throttling seen |
| C12 | Sampling: top-k and top-p, repetition penalty, grammar, `n_probs` | Host work per token on 151,936 logits | Default sampling only |
| C13 | Speculation by prompt type (code, prose, question, repetitive), n-gram and draft model | Whether a gain survives real text and depth | Draft model only: 1.30 to 1.61 on code, at most 1.07 on prose |

**System**

| # | Case | What it challenges | Known today |
|---|---|---|---|
| C14 | `kill -9` during prefill, restart, first token | Whether anything outlives a killed process. Not the TCM mark, which the provider does not hold while computing (2026-11-06, `plan.md` "S2's exit against the results"), so read what the next run gets: the cores, `/dev/tcm` mapped, the same speed, `libspert` falling back to the heap silently | Measured 2026-11-08/09 (`plan.md`): 3 rounds on each model. The new server came up after 5.0 s and 11.0 s, with no block marked, nine `/dev/tcm` lines, no "could not get", and tg64 within 0.8% of before: nothing outlived the killed process |
| C15 | A second process wanting the AI cores (a second context needs none since S2's step 2). Proposed 2026-11-06: an IME process as the second, which does not ask spine-runtime and takes a TCM mark the provider ignores | Each failure costs about 30 s, then an abort, retried on the next run; replies going wrong with no error (inference, `plan.md`) | Seen with speculation; the wait is inferred to be spine-runtime's. On the K3 2026-11-08 (`--quick`; `plan.md` "TTFT harness spec", first K3 run): a refused decode took 30.0 s and reached the streaming client as "Compute error." after a 200, and once A stopped B answered with A's text. Beside the IME build's `llama-bench`, 0 of 2 replies at temperature 0 equal the reference, with no error. The cause is TCM (the full runs, 2026-11-09): with the IME loop's TCM off, 5 of 5 equal on both models, both at about half speed (Qwen3-4B 5.56 to 5.65 t/s against 11.1 alone); with it on, 0 of 5 on both models, garbled from the first characters and right again once the loop stopped. The full runs repeated the 30.0 s refusal on both models |
| C15b | 1, 2, 4 and 8 AI cores on the one-row matmuls and on attention at depth | Whether the weight stream saturates before 8 cores (Codex's point, §8) | Not measured; needs a test-only core count |
| C16 | A fixed memory load on the X100 cores | The weight stream shares DRAM with a shared board's other work | Not measured |
| C17 | `-c` 8192 to 32768 | KV at 144 KiB per position, prompt cache 8 GiB by default, 31 GiB RAM and no swap | Attention extrapolates to about 280 ms per token at 32768 (estimate) |
| C18 | Models: 0.6B (fixed costs weigh about 5 times more), 8B and 14B (1.9 and 3.4 times the bytes), head size 256 (D7) | Whether a lever generalizes; attention leaving the provider | Bandwidth holds at 28 GB/s across sizes; head size 256 unsupported |
| C19 | Accuracy on every path a request crosses: 1 row, 2 to 63, 64 and up, and reused against scratch | Different rounding per path: ggml-cpu's own KLD goes from 0.001896 at `-ub 63` to 0.000000 at `-ub 64` (Mac) | Partly: `-ub 1`, 512-token batches, reuse KL |

## 4. Techniques

Gains are estimates unless marked. Each needs a rule set before its run, as the KV-head split has.

**For TTFT, in the order I would try them**

| # | Technique | Gain | Basis | Effort, risk |
|---|---|---|---|---|
| T1 | Tiled kernel below 64 rows (threshold by rows, perhaps n_kv) | 26-token tail at 4096: about 7 s to 0.7-1.4 s | pp64 is 10.5 times pp63; the kernel already takes fewer rows. A long chat turn ends in such a tail, 21 to 25 tokens after its 512-token ubatch: 32 to 40% of the turn on Qwen3-4B in turns 7 to 19 (2026-11-09, `plan.md`; inference) | One constant behind a test switch. May lose at n_kv 256, where a head stays in the core's cache. Its F32 sums might also improve accuracy (inference). Spec for review 2026-11-09 (`plan.md` "T1 spec") |
| T2 | Keep the prefix across chat turns: `--cache-reuse 16`, or a template keeping the empty think block | Measured on the Mac: decoded 109 to 26 and 138 to 22 | §7 | Settings only. **Replies change** (measured), so first-token KL against scratch must pass. Needs T1 first at depth, where fewer tokens are slower today |
| T3 | System prompt decoded once, saved, restored at start or per conversation (`--slot-save-path`) | Measured on the Mac: 21.6 s to 1.73 s. K3: about 3.7 s to 0.3-0.7 s | §7; restore 4.7 ms, 39.9 MB | No provider code. The file is tied to model and build. Replies equal over 24 tokens only |
| T4 | Prefill while the user types or between turns (`n_predict` 0 worked on the Mac) | Leaves the last few template tokens, the shapes T5 serves | §7 | Client work; wasted when the text changes |
| T5 | KV-head split for 1 to 8 tokens | 4 tokens at n_kv 4096: about 28 ms to 3-5 ms per call | Blocks 2 to 6, running now | Built; decided by its rule |
| T6 | Measure start-up, then parallel repack or a repacked-weights file | Every start: the repack is one thread at about 280 MiB/s, 8.24 s of user time on Qwen3-4B | C1: ceiling 8.47 s of 11.04 s. A parallel repack needs no interface; `perf` can rank the head against the Q4 repack | Moot for a server kept running |
| T7 | Prefill attention at depth: more rows per K and V read, or positions split over tiles for many rows | Attention halved: pp512 at 4096 from 36 to about 54 t/s, a fresh 4,096-token prompt from about 75 to 55 s | Tiled kernel compute-bound, about 88 GFLOP/s before L3 and L4, against 600-890 for matmuls | Block 6 gives the per-op split first |
| T8 | Head-contiguous KV layout inside the provider's buffer | Attention at 4096 from 40.8 to at best 21.6 ms per token, tg there +17% at most; few-token tails too | One-chunk kernels read 17.9 GB/s contiguous, 9.7 GB/s on the real layout (measured) | High: a third layout through state save, restore and shift. On huge pages the heads must not sit a power of two apart (`k3-hardware.md` §2) |
| T9 | Q8_0 KV cache with rotation, new kernels | Half the bytes read and half the memory; perhaps IME for the scores (theory, untested) | Accuracy usable on both models: 0.0042 and 0.0013 (Mac); 4-bit ruled out | High: new kernels, rotation at store |
| T10 | `-b 64 -ub 64` | pp +21% at depth 0, +7% at 4096; cancel wait at most about 1.7 s at 4096 | Rows table; pp64 against pp512 | One flag, but only after T1: every tail is then below 64 |
| T11 | Cancel check in the replay loop | Cancel within a step instead of a whole decode | By reading, §5 | Small in the plugin; the server needs an abort callback and must handle an aborted decode, a TODO today (`tools/server/server-context.cpp:3679`; `upstream-changes.md` entries) |

**For pp**

T10, T7, and these smaller ones: elementwise and quantize steps at 512 rows hold 14% of a batch, with fusion's ceiling 5-6% (`fusion-and-splitting.md` §5); the requantized Q8_0 `ffn_down` of 4 layers cost about 2%; tile imbalance is largest at 8 rows, 6.6%.

**For tg512**

| # | Technique | Gain | Basis | Effort, risk |
|---|---|---|---|---|
| G1 | Test whether X100 and AI cores together read faster than 28.7 GB/s | Upper bound about 44.6 GB/s, 17 t/s, if they add (they may not) | Open since `fusion-and-splitting.md` §6; X100 cores alone read the head at 15.9 GB/s | One microbenchmark. Only if it adds: a matmul shared across core types, which needs scheduler work, and question 8's X100 fault applies |
| G2 | Fewer bytes: head in Q4_0 instead of Q8_0 | About 7 ms per token, +8% | Head 413 MB, 13.2-16.6 ms, 15% of a token | KLD on the Mac first; the head may be too sensitive |
| G3 | N-gram self-speculation (`--spec-type ngram-*`) | Code and repetitive text; no check runs when nothing matches, so little cost elsewhere (by reading, untested) | Up to 8 rows cost at most 1.5 times one; no draft context, so no second core grant | Settings. Needs T1 or T5 at depth. Text may differ at near-ties |
| G4 | Cheaper attention at depth: T8, T9, the one-token KV-head if it clears 2% | At depth only | Above | Above |
| G5 | Tile imbalance in the matmuls | At most 2.3% | 2.0 ms per token (0.6 pair tail, 1.4 spread) | Modest |
| G6 | Fusion | 1-2% in all | `fusion-and-splitting.md` §5 | After everything above |

**Not worth pursuing for batch-one**

| Idea | Why |
|---|---|
| Launch, barriers, host work | 1.6 ms of 89.3 together with llama.cpp's own (measured) |
| Zicbop prefetch | 41-58% slower on the 1-row matmuls (measured) |
| More loads in flight (`burst`) | Faster over short caches, 16-26% slower over long ones (measured) |
| Sampling and server overhead | About 0.5 ms per token; the server adds nothing measurable |
| Attending to less of the cache | Leaves the target's distribution |

## 5. Order after S2, and what S2 needs now

**Inputs to S2's design, cheap and worth having before its spec:**

1. **C1, the split of the 11.0 s** (no code: `scripts/startup-split.py`; `plan.md` "The start-up split"). My reading rated the zero-copy entries low value, and 3 s per GB might have changed that. The bytes settle that part (2026-11-05): plain tensors are 0.06% of what the provider loads on Qwen3-0.6B, so option A cannot shorten start-up. The split, measured the same day, keeps T6 and rules out the flags.
2. **Cancel (T11).** S2 moves the replay loop and the plugin table. If cancel is wanted, one flag read at each barrier belongs in that design, not in a later interface change. The alternative is T10's small `-b`, with no code. This is the user's decision when the S2 spec is written. Decided 2026-11-05: out of S2 for now (`plan.md` "S2 spec", decision 2, with what the server would need).
3. **A short tail in the server check.** S2's baseline ran the KV-head path only on an almost empty cache (the warm-up and request A), with only A's text compared (corrected 2026-11-05; `plan.md`). A follow-up ending in 2 to 8 tokens is S2's step 0, done 2026-11-05: 4 tokens after 19 cached, passing on the K3 on both models, with new baselines (`plan.md`).

**After S2:**

1. A TTFT harness: C3, C5, C6, C10 and C14, with C11 and C15 added (the user, 2026-11-06; tg512 has never been run), client-side and server-side, as the baseline. Built 2026-11-08 as a script beside the runner, `tools/spacemit_ttft.py` (`plan.md` "TTFT harness spec" and its result; usage in `build.md` §6). First K3 run the same day, `--quick`, 11/11. The full runs followed that night: 13/13 on Qwen3-0.6B and 12/13 on Qwen3-4B, its one failure the harness's chat context (fixed 2026-11-09). That case and Qwen3-4B's `alongside` without TCM ran again the same day, 3/3, which completes the baseline.
2. T1, with T5's result beside it. Next (the user, 2026-11-09, ahead of the embedding lookup); its spec in `plan.md` "T1 spec".
3. T2, T3 and T10, each against the baseline with its accuracy check.
4. Block 6's per-op figures, then T7.
5. G1 (could run any time, it needs no provider change), G3 and G2's KLD.
6. T6, if cold start matters to the deployment.
7. T8 or T9, only if attention at depth is still the largest cost at the depths that matter.

**Targets, agreed by the user 2026-11-06:** chat turn at depth 4096 at most 1 s; new conversation with a saved system prompt at most 0.5 s; fresh 500-token prompt at most 4 s; tg512 from depth 0 no worse than today, and from depth 4096 at least 9 t/s. Against the baseline (2026-11-09, `plan.md`), Qwen3-4B: a chat turn at depth 4096 takes 3.00 to 10.56 s by tail, a fresh 500 tokens 5.21 s, and tg512 from depth 4096 runs at 7.53 t/s. Qwen3-0.6B meets every target but the chat turns of 26 and 38 tokens (2.80 and 4.11 s).

## 6. How to measure so the results hold

- Same session, settings interleaved, at least 5 repeats, p50 and p95, and the IME build as a control (between sessions tg128 moves 4-5%).
- TTFT both from the client to the first streamed chunk and as the server's `prompt_ms`, with rows, n_kv, page-cache state and board load noted.
- Each lever against a rule fixed beforehand, with accuracy in the same rule: mean KLD by the CPU-reference rule of `PROJECT.md` §6 per path, reused against scratch by the server check's KL, replies against `--server-baseline` where they should not change.
- One lever at a time; `tcmrelease --apply` before each run.
- With thinking on, record both the first generated token and the first visible answer token (Codex's point, §8): the think block runs at generation speed and can outweigh every prefill lever.

## 7. The Mac tests behind this analysis

Throwaway scripts, 2026-11-04, `build-host`, Qwen3-0.6B Q4_0, whose tokenizer and template the 4B shares; token counts on the 4B will differ with its replies. None of this has run on the K3.

**Chat turns**, `/v1/chat/completions`, the CPU, `-c 4096 -t 8`, temperature 0, the client resending the returned content:

| Setting | Turn 2: prompt, cached, decoded | Turn 3: prompt, cached, decoded | Turn 1 reply |
|---|---|---|---|
| Thinking off, server defaults | 142, 33, **109** | 245, 138, 107 | 84 tokens |
| Thinking off, `--cache-reuse 16` | 142, 116, **26** | 254, 228, 26 | 84 |
| Thinking off, template keeping the empty think block in history | 146, 120, **26** | 243, 217, 26 | 84 |
| Thinking on, server defaults | 171, 33, **138** | 292, 171, 121 | 268 |
| Thinking on, `--cache-reuse 16` | 171, 149, **22** | 277, 255, 22 | 268 |

Turn 2's and turn 3's replies differ from the defaults' in every variant: the template variant's turn 2 keeps the same first 8 words in what I read, and turn 2's length moves from 82 to 72 or 91 tokens with thinking off. First-token KL was not measured. With replies under 16 tokens `--cache-reuse 16` changes nothing, as its minimum chunk predicts. The template's text is `tokenizer.chat_template` in the GGUF: earlier assistant turns are rendered as `<|im_start|>assistant\n` plus the content, without the think block generation left in the cache.

**Saved system prompt**, the simulated device (`FlagOS:SpacemiT` KV buffer, 29/29 layers, `graph splits = 2`), `-c 1024 -np 1 --slot-save-path`, a raw prompt of 348 tokens plus 24:

| Path | Cached | Decoded | Time |
|---|---|---|---|
| From scratch | 0 | 372 | 21,600 ms |
| Prefill with `n_predict` 0 | 0 | 348 | 19,897 ms |
| The rest, after that prefill | 348 | 24 | 1,732 ms |
| The rest, a fresh process after `action=restore` | 348 | 24 | 1,726 ms |

Save 42 ms, restore 4.7 ms, 39.9 MB. The 24-token reply is identical in all three. The simulated device is serial, so only the ratios mean anything.

## 8. Codex's general roadmap against this board (2026-11-04)

The user asked Codex for general advice on batch-one inference (eight points: K3 constraints, backend overhead, separate prefill and decode kernels, quantized weight access, fusion, scratchpad and core topology, attention by context length, benchmarking) and asked that any idea not yet considered be recorded here. Each point against what the project has measured; "done" means built and measured, with the record named.

| Codex's point | State here | Evidence |
|---|---|---|
| 1. Constraints: 51 GB/s peak, 1.5 MB scratchpad per cluster, 60 TOPS sparse INT4; model a token as the sum over ops of max(compute, bytes) plus dispatch | The achieved figures are what count: 28.7 GB/s for the weight stream, about 0.96 int8 TOPS on the 512-row matmuls, TCM 384 KiB per core (1.5 MB per cluster of 4). The per-op model is the bottleneck suite's method; it puts 87.3 of a token's 89.3 ms in the weight stream | `build.md` §7 bottleneck suite; `k3-hardware.md` §2 and §4 |
| 2. Backend overhead: per-op placement and timing; X100 fallback, repacking per call, a sync per op, allocation or compilation in the decode loop, host copies, Python per op | Done and closed: every layer on the AI cores, 2 graph splits per token (the token embedding and sampling on the CPU); weights repacked once at load; one launch per split with a barrier per step; the plan cache replays a split's step list after its first graph (host work 46 µs per token); the workspace fixed; no Python anywhere. Launch, barriers and host work together 1.6 ms of 89.3, about 1.8% | `fusion-and-splitting.md` §7.1; `plan.md` S1 step 3c; §4 of this document, "not worth pursuing" |
| 3. Separate prefill and decode linear kernels; partition output rows across cores; matrix extension against vector; the FFN holds about 70% of a block's weights (31.5 M attention, 74.7 M FFN) | Already separate: the 1-row kernel (exact branch, columns split over the tiles in pairs, no cross-core reduction) and the 4-row kernels (path A from 113 rows, path C below); every matmul on the IME, the RVV Q6_K dot product considered and rejected for the head. A better decode GEMV cannot gain: the stream is at the memory ceiling, so only fewer bytes can (G2). Correction: the four attention projections hold 26.2 M weights, not 31.5 M, so the FFN is 74% | `plan.md` M2b to M2c, R12; `fusion-and-splitting.md` §7.3 |
| 4. Weight format: pack once at load in the kernel's tile layout, keep it compressed, dequantize tiles into registers or scratchpad, integer dot products with quantized activations, counting the quantization cost | Done: Q4_0 and Q8_0 packed once into the IME layouts, read compressed, int8 products with the activations quantized per step (1.04 ms per token, 1.2%); Q4_1 and Q6_K requantized to Q8_0 at load, accepted by KLD. One layout serves both regimes. Q4_K is not supported; it is a target item (D7, Qwen3.5-4B Q4_K_M), not a batch-one one | `plan.md` M2c, D4 |
| 5. Fusion: residual add and norm, Q/K norm and RoPE, gate/up and SiLU, attention with the cache, head and sampling | Measured ceilings: 1-2% of a token in all, 5-6% of pp512; attention is already one fused op; the head's 15% is its bytes, sampling 0.5 ms. Ranked last here (G6) for that reason, not from doubt about the patterns | `fusion-and-splitting.md` §5 and §7.1 |
| 6. Scratchpad: double buffering in prefill; a worker sweep 1, 2, 4, 8; X100 workers only after measuring | TCM stages path A's rows and attention's tiles; more loads in flight were faster over short caches and 16-26% slower over long ones. **Not done: a sweep of the AI-core count** for bandwidth saturation (the stream takes a core count; a test-only switch would be small); nor double buffering of the weight stream on path A. X100 and AI cores together: G1, open | `k3-hardware.md` §2; §4 G1 |
| 7. Attention: 144 KiB of KV per token; tiled prefill with online softmax; a separate decode kernel; the KV head reused over its 4 query heads; KV quantization when its traffic matters; no cache copying on append; chunked prefill does not by itself lower TTFT | The same 144 KiB. All built: the tiled kernel (64 x 64, online softmax), the one-chunk and split kernels for one row, the grouped read of a KV head for its query heads (the KV-head split extends it to several tokens), one-row writes on append. KV quantization measured on the Mac (T9: Q8_0 with rotation usable, 4-bit not). On chunking the board disagrees with the general rule: 64-row ubatches beat 512-row ones by 21% at depth 0 (T10), because pp peaks at 64 rows | `plan.md` M2e; §4 T5, T7 to T10 |
| 8. Benchmark: same inputs and quality checks; cold against warm; prompt lengths 128 to 4096; 512 tokens after short and long prompts; worker counts; formats; fusion on and off; check what tg512 excludes; **first generated token against first visible answer token with thinking on** | Covered by §3 and §6 except the worker sweep (above) and the thinking point, which I had missed: with thinking on, the first visible answer waits for the whole think block at generation speed, 150-250 tokens in the Mac test (678-984 characters of reasoning), about 15-25 s on the 4B at 11 t/s (estimate), more than any prefill lever saves. So for the TTFT criterion the first question is whether thinking is on, and a reasoning budget or `enable_thinking: false` is a lever above T1 | §7 chat turns |

Taken into this document from Codex: the AI-core count sweep (added to C11's family as a system case) and the visible-token rule for thinking on (added to §6). Not applicable here: Python per op, the datasheet's 60 TOPS and 51 GB/s as roofs, repacking per call, a sync per op, and decode GEMV tuning as a lever.
