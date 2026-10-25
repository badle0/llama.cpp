# Fusion and graph splitting: groundwork for the SpacemiT provider

Started 2026-10-21 (user: explore operator fusion and graph splitting while the mentor's answers are pending; pruning
left aside). Exploration first; since the same day the candidates are prepared for measurement one at a time, by rank
(§7; user request), each behind a test-only switch. Measured numbers cite `structural-experiments.md` §6 (runs 4-5),
`build.md` §7 and `plan.md`; everything else is labelled as an estimate or an inference. `plan.md` M4 (fixed fusion,
"only from measured hotspots") is the milestone this feeds.

## 1. The two angles, in this codebase's terms

**Operator fusion:** several consecutive ops run as one kernel, so the data between them stays in registers or TCM
instead of going back to DRAM, and the steps between them need no barrier. Cost: one kernel per fused pattern, tests for
every shape it claims, and the separate ops kept as a fallback.

**Graph splitting:** deciding which device runs which part of the graph and where to cut it. llama.cpp's scheduler
places each node on a backend whose `supports_op` accepts it, cuts the graph wherever the backend changes ("splits"),
copies data across the cuts, and runs the splits one after another (`ggml_backend_sched_compute_splits`,
`ggml/src/ggml-backend.cpp:1594`). "Based on the model" means choosing the chunks from the model's structure (layers,
attention against FFN, prompt processing against generation) instead of letting support decide op by op.

**The FlagOS design (V2.0, `ggml-flagos-provider-design.pdf`, not in the repo; see the memory note on the design PDF)**
names four ways a provider can work (pp. 7-8, §4):

| Mode | What it does | Here |
|---|---|---|
| direct-op | each ggml op is its own kernel | **our provider today** |
| fixed-fusion | Common scans the graph for a fixed catalog of patterns; the provider accepts or rejects each candidate and runs a fused kernel | not used: `graph_optimize` is `nullptr` (`flagos-spacemit.cpp:466`), no side-plan |
| split compiler | the provider takes the split it was given and compiles it as a unit: internal partition and fusion, layout and memory planning, compile cache; needs no placement API | not used |
| pre-placement partition | redraws the cuts before placement; only for hardware that can run nothing but whole fused blocks, and only with real hardware evidence (§4.4, phase P6 p. 12) | not needed: our cores run every op alone (`plan.md` §1 drops P5 and P6) |

Rules from the same document:
- `supports_op` true means the node runs correctly on this backend *alone*, even if no fusion happens (§7).
- The call order is placement, then splits and copies, then `graph_optimize(split)`, then `graph_compute(split)`
  (p. 6, §3.1): fusion comes after placement, but "a split may still fuse, lower to a compiler and replay internally".
- The current system is not an arbitrary-subgraph compiler partitioner nor a global partition solver; its fuser is a
  greedy, non-overlapping pick among fixed patterns (pp. 2, §1.2).
- A provider does not copy llama.cpp's model layers or keep its own model topology (p. 2): model knowledge enters as
  patterns over ggml's graph.

**Common's fixed-fusion mechanism** (read in `upstream-review.md` §A at `5794e12`): `flagos_enumerate_candidates()`
finds candidates in the split, the provider's `query_lowering` (`flagos-graph-plan.h:101`) accepts or rejects each, a
greedy non-overlapping plan is formed, and the provider's `execute_fusion` (`:110`) runs the chosen steps. Providers
build the `flagos_fusion_interface` inside `graph_compute` (AMD `flagos-amd.cpp:4748`, Denglin `flagos-denglin.cpp:2261`).
The catalog has 16 pattern IDs (`flagos-graph-plan.h:12-33`): `rms_norm_mul`, `add_rms_norm_mul`, `ssm_conv_silu`,
`gated_delta_net_decode`, `gated_delta_net_prefill`, `gated_rms_norm`, `ffn_swiglu`, `rope_kv_store`,
`qkv_mrope_kv_store`, `flash_attn_decode`, `flash_attn_prefill`, `attention_output_gate`, `rms_norm_mul_rope`,
`rms_norm_mul_rope_kv_store`, `ffn_swiglu_down` (gate and up projections, SwiGLU and down projection with private
scratch), `gdn_gate_projections`. AMD's provider uses 14 of them (all but `gated_rms_norm` and `qkv_mrope_kv_store`,
by a search of `flagos-amd.cpp`); Denglin only `rms_norm_mul`.

Known concerns with it (`upstream-review.md` A1-A6): the catalog is closed and owned by Common, so a provider cannot
propose a pattern cut at another boundary (the K3's likely valuable fusions, such as quantize + IME matmul + epilogue
tiled in TCM, are not in it); pattern layouts are positional; each provider re-validates; nothing binds pattern
tensors to kernel arguments; fusion happens after placement, so every member op must also be supported alone; the
descriptor declares no patterns. `plan.md` M4's rule stands: K3-specific fusions go to the mentor, not into Common.

## 2. What fusion and splitting act on: our execution model

- One spine-runtime launch per split; `graph_compute` expands the split's nodes into steps (`flagos-spacemit.cpp`
  `spacemit_backend_graph_compute`); the op table gives each op its steps (`flagos-spacemit-ops.cpp:115-130`):
  `MUL_MAT` two (quantize the activations, then the matmul), the split attention two (partial results, then the
  merge), every other op one.
- Every tile (8, one per AI core, each with its core's 384 KiB TCM) runs every step, with a barrier after each step
  (`flagos-spacemit-exec.cpp`, `spacemit_tile_main`). Inside a step the 1-row matmuls and the split attention also wait
  for their pair partner (spin waits).
- Rules from the structural experiments: keep one launch in flight per stream; never order launches with `join`
  (F16: tiles then move and share TCM); a tile takes its TCM once per launch, safe because tiles do not move within a
  single launch (0 of 160 tiles moved, E3d). Any fused kernel that uses TCM or pair waits inherits these assumptions.
- Placement today: 2 splits per token, the token-embedding lookup (`GET_ROWS` Q6_K) on the X100 cores and everything
  else on the AI cores, as ACCEL with C5a and C5b and as IGPU or GPU without them (E2). The KV cache is read in place from
  a CPU buffer (ACCEL) or lives in our buffer (IGPU, GPU).

## 3. The graph per token (Qwen3-4B Q4_0)

`plan.md` M0.7: 36 layers of 23 ops plus 6 = 834 ops; generation and prompt processing have the same op list, only the
number of tokens differs. With our step expansion a layer is 30 steps:

```
RMS_NORM, MUL                         attention norm
quantize, MUL_MAT Q                   query projection
RMS_NORM, MUL, ROPE                   query norm and rotary
quantize, MUL_MAT V; quantize, MUL_MAT K
RMS_NORM, MUL, ROPE                   key norm and rotary
SET_ROWS K, SET_ROWS V                writes into the KV cache
FLASH_ATTN_EXT                        (two steps, partials and merge, from n_kv 512 for one token)
quantize, MUL_MAT O                   attention output
ADD                                   residual
RMS_NORM, MUL                         FFN norm
quantize, MUL_MAT gate; quantize, MUL_MAT up
SWIGLU
quantize, MUL_MAT down
ADD                                   residual
```

36 x 30 + 6 (output norm, the two output-row `GET_ROWS`, the head's quantize and matmul) = **1,086 steps**, the number
E3c counted (1,122 from n_kv 512, the split attention's merge in each layer). Weights read per token: Q4_0 1,895.6 MiB
(248 matmuls), the Q6_K head 304.3 MiB (run as Q8_0), Q4_1 59.4 MiB: 2,259 MiB in all.

## 4. The measured budget of a generated token

E3c, Qwen3-4B, tg128, counters on (they cost 0.5%), mean of 3 repetitions (`structural-experiments.md` §6 run 4):

| Part | Per token | Share |
|---|---|---|
| kernels, mean tile | 85.2 ms | 95.4% |
| imbalance (each step's slowest tile less the mean, summed) | 2.5 ms (2.5-3.7 at depth) | 2.8% |
| barriers' own cost | 0.6 ms (0.55 µs each with a clock read; a bare barrier 0.435 µs, E3d) | 0.7% |
| host work in `graph_compute` | 0.55 ms | 0.6% |
| the one launch | 0.19 ms | 0.2% |
| outside `graph_compute` (llama.cpp, the X100 split, sampling) | 0.27 ms | 0.3% |
| token | 89.3 ms | |

Other measured points: the output head takes 16.6 ms per token on the AI cores (26.0 on the X100 cores; M2c.2);
attention is 43% of a token at depth 4096 (M2e.3); attention was about 70% of pp512 at depth 4096 (M2e.1, before the
tiled kernel became a third faster in M2e's L3 and L4); the AI cores stream weights at up to about 28 GB/s (M2e.3).

**Measured 2026-10-21** (§7.1, the baseline; until then the main gap): time per op kind. Per tg128 token the matmuls'
multiply steps take 83.46 ms, attention 1.44 ms and everything else 3.26 ms; the matmuls hold 2.0 of the 2.44 ms of
imbalance: the pair turns' tail about 0.6 ms of it, a spread between pairs about 1.4 ms (§7.1).

Reading (inference): fusion cannot reduce the weight traffic, since every weight is read once per token. In
generation these angles can only win back the per-step costs (barriers, imbalance, launch, host work) and the small ops'
own time. Prompt processing has more room: its activations are up to 512 tokens wide (a 512 x 2560 F32 activation is
5 MiB per read or write).

## 5. Fusion candidates in our layer (estimates; nothing measured per op)

How to find one: for each pair of adjacent steps, ask whether the second uses only the first's output, whether the data
between them fits in registers or TCM, and whether the barrier between them exists only because of our one-op-one-step
rule.

| # | Candidate | Steps now -> after | In Common's catalog | Saves per token (estimate) |
|---|---|---|---|---|
| 1 | `RMS_NORM` -> `MUL` (4 per layer, and the output norm) | 2 -> 1 | `rms_norm_mul` (1) | 145 steps |
| 2 | residual `ADD` -> `RMS_NORM` -> `MUL` | 3 -> 1 | `add_rms_norm_mul` (2) | 2 steps per site |
| 3 | query: `RMS_NORM` -> `MUL` -> `ROPE` | 3 -> 1 | `rms_norm_mul_rope` (13) | 72 steps |
| 4 | key: `RMS_NORM` -> `MUL` -> `ROPE` -> `SET_ROWS` | 4 -> 1 | `rms_norm_mul_rope_kv_store` (14) | 108 steps |
| 5 | gate, up -> `SWIGLU` (-> down) | 5-7 -> fewer | `ffn_swiglu` (7), `ffn_swiglu_down` (15) | 1-3 steps per layer |
| 6 | Q, K and V each quantize the same input; so do gate and up | 5 quantize steps -> 2 | no (our internal step) | 108 steps, and the repeated work |
| 7 | one token: each tile quantizes the whole activation row itself (2,560 values) instead of splitting it and waiting at a barrier | quantize step -> part of the matmul step | no | up to 7 barriers per layer (1-row graphs only); costs each tile a redundant small quantization |
| 8 | barriers only where a step reads another tile's output: Q/K/V matmuls, gate/up, the two cache writes run without barriers between them | about 4-5 barriers fewer per layer | no (the split-compiler idea, small) | about 150-180 barriers |
| 9 | matmul epilogues: the attention-output and down matmuls add the residual as they write | 2 -> 1 per site | no (`upstream-review.md` A1) | 72 steps, one pass over the activation each |
| 10 | RoPE's sin/cos once per position: at head size 128 every tile computes 64 `sinf` and 64 `cosf` in each of the 72 RoPE steps of a token, for the same positions (`legacy_small_f32`, `flagos-spacemit-rvv-kernels.cpp:485-524`) | no step fewer; the trig of 71 of 72 RoPE steps | no (inside a kernel) | 0.15-0.45 ms (the trig's cost on the AI cores not measured) |

Notes:
- Combined, my estimate: removing 10-15 steps per layer removes 360-540 barriers per token, at about 0.55 µs each plus a
  share of the imbalance: about 0.5-1.5 ms per token, 0.5-1.5% of generation. Prompt processing probably gains more
  (wider activations), not estimated.
- Catalog patterns (1-5) fit FlagOS's fixed-fusion mode: implement `query_lowering` and `execute_fusion`; every member op
  already has a direct path (M2d), as A5 requires. `plan.md` M4 lists 1, 2, `ffn_swiglu`, `rope_kv_store` and
  `flash_attn_decode` (already one op here).
- Internal ones (6-9) live inside `graph_compute(split)`, which the design allows (p. 6). 8 needs per-step dependency
  information in the executor; 7 changes the work split, so outputs must stay byte-identical or be checked against the
  CPU.
- Not fusion, but the largest overhead measured: the imbalance, 2.5-3.7 ms per token, which comes from how the kernels
  split work across the 8 tiles.

Found by reading (2026-10-21), all per generated token: a one-row `RMS_NORM` runs on tile 0 alone, 73 steps
(`flagos-spacemit-rvv-kernels.cpp:94-98`); a one-row Q4_0 quantize at K 2560 has 10 blocks of 256 values, so tiles 5-7
get none (`flagos-spacemit-ime.cpp:254-262`); in the 1-row matmuls' pair turns the odd tile does its last multiply after
the last pair wait, so it ends one slab-multiply after the even one in every matmul (`flagos-spacemit-ime.cpp:216-236`),
about 0.8 ms of the imbalance by estimate. Removing barriers does not help a tile that is late every time.

**Ranking (2026-10-21, estimates; the op counters of §7.1 replace them).** Generation saves per removed step about 1 µs
(`--bench`'s 0.89-0.94 µs per node, barrier included); prompt processing saves passes over activations at the 16M-float
`ADD`'s 22.5 GB/s (less if the caches hold them). Why the phases differ: the same 1,086 steps carry 1 row in generation
and 512 in pp512; a token reads every weight in generation (89 ms), while 512 tokens share that read in pp512 (10.5 ms a
token), and the activations between ops grow from 10-38 KiB to 5-19 MiB.

| Rank | Candidate | Route | tg128 | pp512 | Effort |
|---|---|---|---|---|---|
| 1 | #6 one quantization for Q/K/V and for gate/up | internal | 0.1-0.2 ms (0.1-0.2%) | ~30 ms (0.6%) | low |
| 2 | #10 RoPE's sin/cos once per position | internal | 0.15-0.45 ms (0.2-0.5%) | 12-24 ms (0.2-0.45%) | low |
| 3 | #2 residual add + norm + mul (#1 its subset) | Common's catalog | 0.2-0.35 ms (0.2-0.4%) | ~50 ms (0.9%) | medium: the planner's first use |
| 4 | #3, #4 the Q and K chains | Common's catalog | 0.15-0.25 ms (0.2-0.3%) | ~75 ms (1.4%) | medium |
| 5 | #5 gate and up in one matmul step, SwiGLU (and the down quantize) in its epilogue | Common's catalog | 0.2-0.4 ms (0.2-0.45%) | 95-190 ms (1.8-3.5%) | high |
| 6 | #8 barriers by data dependency | internal | 0.15-0.5 ms | ~0 | medium-high |
| 7 | #9 generalized: residual, norm, mul and quantize in the matmul's epilogue (path A gives a tile whole rows) | K3-specific | ~0.1 ms | up to 0.3-0.4 s, instead of ranks 3-5 | high, mentor |
| 8 | #7 each tile quantizes the whole row | internal | likely a loss | n/a | low |

All together about 1-2% of tg128 and 5-6% of pp512. Every candidate can keep outputs bit-identical (the same float
operations in the same order), so identical logits can accept them. Risk to ranks 3-4: Common's plan cache checks the
whole graph at every call; at 0.1-0.3 ms per token (not measured) it would cancel their generation gain.

## 6. Graph splitting on the K3

- Today: 2 splits per token; the X100 cores run only the embedding lookup, so they are nearly idle during generation
  (inferred from the split structure; their use was not measured).
- Using the X100 cores at the same time, e.g. for the output head (16.6 ms on the AI cores): within one sequence every
  layer needs the previous one and the head needs the last layer, so nothing runs in parallel; parallel work exists
  only across sequences (batches, speculative decoding). ggml's scheduler runs splits one after another, so overlap
  needs scheduler changes, against R1 (least upstream change). Both core types read the same DRAM, and generation is
  bound by weight streaming: whether X100 and AI cores together read faster than the AI cores alone (about 28 GB/s) is
  not measured (§7 item 2).
- Split compiler: our one split compiled as a unit (fusion, barriers placed by data dependency, memory planning) fits
  the design (§4.3, no placement API) and our executor, which already holds the split as a step list. It is §5's
  items 6-9 generalized; a larger version would lower whole layers with a compiler (spine-mlir or FlagTree, M3-related,
  maturity unknown).
- Pre-placement partition: not needed (§1).
- Device type does not change the cut: ACCEL with C5a and C5b, IGPU and GPU all give 2 splits per token (E2).
- Reading: the realistic splitting angle on the K3 is the split compiler, which overlaps with fusion; using the X100
  cores alongside needs the bandwidth measurement first and upstream changes after.

## 7. Measurements

Candidate by candidate, by rank (user, 2026-10-21). Each candidate gets a test-only switch, checks that compare it with
the defaults byte for byte, predictions and rules fixed before its K3 run, and a mode in `scripts/fusion-experiments.sh`.

### 7.1 The op counters (built 2026-10-21; not yet run on the K3)

`FLAGOS_SPACEMIT_TEST_OP_STATS=<file>` (test-only; `spacemit_op_stats_file`, `flagos-spacemit.cpp`) groups E3c's times by
what each step does. It is M4's first item ("profile a generation step").
- Recorded per step and tile: the kernel time (E3c's `kernel_ns`) and, new, the wait at the barrier after the step
  (`step_barrier_ns`), from the same clock reads.
- Per kind of graph (launch mode, rows of the first matmul, n_kv of the first attention, and a hash of the steps' kernels,
  ops and shapes) every step's times are summed over the graphs; the kind's first graph is left out (it may create the
  stream and touch new memory). Host work: two passes over the steps per graph (estimate 20-40 µs).
- Names, made once per kind (`spacemit_step_label`): the op (with `quantize`/`gemm` for a matmul, `one-step`/`partial`/
  `merge` for attention), the role from the tensors' names (`blk.N.attn_q.weight` -> `attn_q`; an `RMS_NORM` by the norm
  weight of the `MUL` that takes it, a `ROPE` by the one before it, a `SET_ROWS` by its cache, an `ADD` by the matmul
  whose output it adds), the weight type and the rows. Nothing in execution depends on them.
- Written when the process ends, one JSON line per kind and name and one per kind for all steps (`(all)`), per graph:
  `wall_us` (the mean tile's kernel and barrier time; sums to the launch), `kernel_us` (the mean tile's), `busy_us`
  (each step's slowest tile, summed), each tile's kernel time, and how often each tile was a step's slowest. Imbalance =
  busy - kernel and barrier = wall - busy, E3c's definitions.
- A file of its own: E3c's summary takes the first line whose rows and KV class match (`structural-experiments.sh:402`),
  so the two switches write separately and can be used together.
- Summary: `scripts/op-stats.py show <files>` (per kind, names by wall time) and `diff <files> -- <files>` (the second
  set against the first, name by name). A set's files (one per run) are merged, weighted by their graphs.
- On the Mac (host executor on threads with a 384 KiB TCM stand-in): the layer check's 31 steps get the expected names
  in llama.cpp's order (Q with its norm and RoPE, V, K with its norm and RoPE, the cache writes, attention, ...). Times
  there mean nothing (the host's barriers take about 20 µs).
- Limits: a step shorter than a few clock ticks is not resolved per step (the board's timebase is not read yet:
  `/proc/device-tree/cpus/timebase-frequency`); a pair wait inside a step counts as kernel time; the barrier part also
  holds the tiles' skew leaving the barrier.

**The baseline run (prepared 2026-10-21; user: run it before any candidate, as the foundation).** `EXPS=base` runs only
`ref` and `ops`, interleaved; `ops` writes the op counters and E3c's counters from the same clock reads, so every run
checks the one against the other. No logits check: the counters change no output. It checks the counters on the board
(their cost, their totals against E3c's, the names), measures where a token's and a prompt batch's time goes op by op at
depths 0, 1024 and 4096, and gives the candidates' ceilings before anything more is built.

Predictions (estimates, fixed before the run; Qwen3-4B Q4_0, mode `ops`):

| Test | Prediction |
|---|---|
| tg128, per token | the matmuls' gemm steps 82-85 ms (busy); attention 2-3 ms; the quantize steps 0.25-0.5 ms; `RMS_NORM` 0.2-0.45 ms, mostly the 73 one-row ones; `ROPE` 0.2-0.5 ms; `MUL`, `ADD`, `SET_ROWS`, `GLU` and `GET_ROWS` 0.3-0.5 ms together; a barrier part of about 0.55 µs per step in every name |
| its imbalance (2.5 ms) | at least 1.5 ms in the gemm steps; the odd tiles slowest in most 1-row gemm steps (the pair turns), tile 0 in every one-row `RMS_NORM`, one of tiles 0-4 in the quantize steps at K 2560 |
| tg32 at depth 1024, 4096 | attention (partial and merge steps) about 13.5 and 40 ms per token; `ROPE` dearer at 4096 (glibc's `sinf` and `cosf` take a slower path for large arguments); the other names as at depth 0 |
| pp512, per batch | the gemm steps 3.9-4.2 s; attention (tiled) about 0.67 s; the elementwise and quantize steps 0.3-0.6 s: `GLU` 80-130 ms, quantize 80-120 ms, the `RMS_NORM` and `MUL` steps 100-150 ms, `ROPE` 40-70 ms, `ADD` 30-50 ms, `SET_ROWS` 10-15 ms |
| pp128, per batch | the gemm steps about 1.0-1.1 s of 1.39 s (gate and up take path C at 128 rows, `flagos-spacemit-ime.cpp:299-306`); attention about 60 ms; the rest 0.15-0.3 s |

Valid if (fixed before the run): every run ends with rc 0; the counters cost at most 1% in tg128 and pp512 (`ops`
against `ref`); in generation the op counters' kernels and busy per graph are within 1% of E3c's from the same runs (E3c
counts each kind's first graph, the op counters do not); every step of a layer is named by its role, as on the Mac.

What it decides: each candidate's ceiling from fixed names, an upper bound (a fused kernel still does the work of the
steps it takes in):
- #6: the wall of `MUL_MAT.quantize attn_v`, `attn_k` and `ffn_up`.
- #10: the kernel time of `ROPE attn_q_norm` and `ROPE attn_k_norm` (the trig's share in it is not separated).
- #2: the wall of `ADD after attn_output`, `ADD after ffn_down`, `MUL attn_norm`, `MUL ffn_norm` and `MUL output_norm`,
  and the imbalance of the one-row `RMS_NORM` steps.
- #3, #4: the wall of `MUL attn_q_norm`, `ROPE attn_q_norm`, `MUL attn_k_norm`, `ROPE attn_k_norm` and `SET_ROWS cache_k`.
- #5: the wall of `MUL_MAT.quantize ffn_up` and `GLU SWIGLU`, and the imbalance of `MUL_MAT.gemm ffn_gate`.
- #8: the barrier part of the steps whose barrier would go (in the Q and K chains, after V's and gate's gemm steps,
  after `SET_ROWS cache_k`).
- #9: the wall of the two `ADD after` names.

The candidates are then re-ranked by ceiling and effort; the order is the user's decision.

```bash
cd /root/llama.cpp-flagos && git -c safe.directory=$PWD pull --ff-only ~/sx-<tag>.bundle feature/flagos-spacemit
docs/flagos-k3/scripts/fusion-experiments.sh prepare 2>&1 | tee ~/fx-prepare.log
od -An -tu4 --endian=big /proc/device-tree/cpus/timebase-frequency   # the clock's ticks per second, if present
EXPS=base DEPTH=1 REPS=3 docs/flagos-k3/scripts/fusion-experiments.sh run ~/models/Qwen3-4B-Q4_0.gguf ~/fx/base > ~/fx-base.log 2>&1
docs/flagos-k3/scripts/fusion-experiments.sh summary ~/fx/base > ~/fx-base-summary.txt; head -60 ~/fx-base-summary.txt
```

About 35 minutes (estimate: 2 modes, 3 repetitions, the depth runs). The summary: llama-bench per mode and the counters'
cost; E3c's budget of a generated token with the op counters' totals against E3c's; the `ops` table per kind of graph
(tg128 at n_kv 256, tg32 at n_kv 1280 and 4352, pp128, pp512, pp512 at n_kv 4608). The same run on Qwen3-0.6B is
optional: its token takes about 18 ms, so a step's fixed cost weighs about 4 times more there.

**Baseline results (K3, 2026-10-21; build `eb7a53962`; Qwen3-4B Q4_0, `EXPS=base DEPTH=1 REPS=3`; 25 of 25 runs rc 0;
logs `~/fx/base` on the board).** Valid by the rules fixed before the run: the counters cost tg128 -0.83% and pp512
+0.06% (tg32 at depth 0 -1.38%, outside the rule; E3c's own counters cost 1.1% there); in every generation test the op
counters' kernels, busy and barriers lie within 0.5% of E3c's from the same runs (tg128: -0.02%, -0.02%, +0.00%); tg128's
budget is run 4's (kernels 85.2 ms, imbalance 2.45 ms, 1,086 steps); every step is named.

| Group | tg128, per token | pp128, per batch | pp512, per batch |
|---|---|---|---|
| matmul multiply steps | 83.46 ms (94.7%) | 1,187 ms (85.6%) | 3,964 ms (74.0%) |
| attention | 1.44 ms (1.6%) | 62 ms (4.5%) | 666 ms (12.4%) |
| quantize steps | 1.04 ms (1.2%) | 39 ms (2.8%) | 167 ms (3.1%) |
| `RMS_NORM` | 0.43 ms (0.5%) | 29 ms (2.1%) | 197 ms (3.7%) |
| `ROPE` | 0.59 ms (0.7%) | 22 ms (1.6%) | 90 ms (1.7%) |
| `MUL` | 0.40 ms (0.5%) | 11 ms (0.8%) | 84 ms (1.6%) |
| `ADD` | 0.32 ms (0.4%) | 9 ms (0.7%) | 70 ms (1.3%) |
| `GLU` (SwiGLU) | 0.30 ms (0.3%) | 25 ms (1.8%) | 111 ms (2.1%) |
| `SET_ROWS` | 0.19 ms (0.2%) | 3 ms (0.2%) | 12 ms (0.2%) |
| all (wall) | 88.16 ms | 1,387 ms | 5,360 ms |

At depth (tg32) attention takes 14.26 ms per token at n_kv 1280 (14.1%) and 40.83 ms at n_kv 4352 (32.1%); the other
groups stay as at depth 0, RoPE 6-9% dearer.

Against the predictions: the matmuls (82-85 ms), the one-row norms' slowest tile (tile 0, 100%) and pp512's attention
(0.67 s) as predicted, and the matmuls hold 2.0 of the 2.44 ms of imbalance (predicted at least 1.5). Missed: the rest of
a generated token is 3.26 ms, about twice the upper estimate (the quantize steps 1.04 ms against 0.25-0.5; `MUL`, `ADD`,
`SET_ROWS` and `GLU` 1.2 ms against 0.3-0.5); attention at depth 0 1.44 ms (2-3 predicted); pp512's elementwise and
quantize steps 0.73 s (0.3-0.6); pp128's matmuls 1.19 s (1.0-1.1).

Findings (measured unless marked):
- The 1-row matmuls hold 2.0 ms of imbalance per token, the largest overhead after the weights, and in all 9 matmul roles
  the slowest tile was an odd one in 100% of the steps. The pair turns explain which tile: after a pair's last
  handshake the even tile has nothing left and the odd one still multiplies its last slab
  (`flagos-spacemit-ime.cpp:225-256`, the order of the in-tree `ggml-cpu/spacemit/ime.cpp:447-485`, a two-tile barrier
  per pair). They do not explain all of the time (corrected 2026-10-21, from the tiles' kernel times): the odd tiles'
  lag per matmul grows with the row length as one slab's multiply should, 3.2-3.3 µs at K 2560, 5.2 at 4096, 12.2 at
  9728 (18.1 for Q4_1), which leaves the even tiles idle about 0.6 ms per token. The other 1.4 ms is a spread between
  pairs (some pairs finish later; cause not isolated, most likely uneven memory speed), which also lands on an odd tile
  and moves between them (for Q, tiles 1, 3, 5 and 7 were the slowest in 18, 27, 22 and 33% of the steps). So letting a
  pair's tiles finish together would remove about 0.6 ms; the spread varies from step to step, so steps that do not
  depend on each other could let it average out without a barrier between them (#8: only V and K, gate and up).
- A small step costs 2.5-4.5 µs of wall time in generation. Its kernel takes 1.7-1.9 µs on every tile even with almost
  no work: the idle tiles 5-7 of a K-2560 quantize step 1.67 µs, `MUL attn_k_norm` (128 floats of its own row per tile)
  1.85 µs. `--bench`'s chain of tiny `ADD`s needs about 0.45 µs per kernel. Cause not isolated. Hypothesis: each step's
  tensor metadata and inputs have left the caches while the matmuls streamed their weights. Test: the `--bench` chain
  with a weight stream between its `ADD`s, or touching the next step's tensors during the barrier.
- RoPE: Q's 9.7 µs per step and K's 5.2 µs fit 3.7 µs of trig plus 1.5 µs per row of scalar rotation (inferred from the
  two points), about 0.27 ms of each per token.
- At 128 rows the gate and up projections take path C (`flagos-spacemit-ime.cpp:299-306` splits columns when n/m > 64):
  9.56 ms per layer, against 6.63 ms for the down projection of the same size on path A (667 against 962 GFLOP/s), so
  about 205 ms of pp128's 1.39 s. If path A ran them, pp128 would gain about 17% (estimate). pp512 takes path A everywhere.
- In pp512 the head-row norms are latency-bound: `RMS_NORM attn_q_norm` (16,384 rows of 128) takes 98 ms per batch, 3.6
  times what its bytes need at the `ADD`'s rate; the Q chain (norm, mul, RoPE) takes 199 ms.
- At n_kv 1280 the split attention's partial step leaves tile 0 the slowest in 80% of the steps (3 of the 17 visible
  blocks against 2): 1.28 ms of imbalance per token.

Ceilings from the fixed names (upper bounds: a fused kernel still does the steps' work):

| Candidate | tg128, per token | pp512, per batch |
|---|---|---|
| #5 gate and up in one step with SwiGLU | 0.65 ms (0.74%); with the down quantize 0.84 ms (0.96%) | 126 ms (2.3%); 174 ms (3.2%) |
| #2 residual add + norm + mul | 0.62 ms (0.70%) | 113 ms (2.1%) |
| #10 RoPE's sin/cos | 0.54 ms (0.61%), the trig about half | 89 ms (1.7%) |
| #6 one quantization for Q/K/V and gate/up | 0.40 ms (0.45%) | 46 ms (0.86%) |
| #3, #4 the Q and K chains | 0.87 ms (0.99%), of which the steps' fixed costs about 0.5-0.6 | 137 ms (2.6%) |
| #9 residual add in the matmul's epilogue | 0.32 ms (0.36%), overlapping #2 | 70 ms (1.3%) |
| #8 barriers by data dependency | 0.14 ms (0.16%), and part of the spread between pairs where two independent matmuls follow each other | 0.3 ms |

Not fusion, but larger: the matmuls' imbalance 2.0 ms per token (2.3%: the pair turns' tail about 0.6 ms, a spread
between pairs about 1.4 ms); pp128's path C about 205 ms per batch (14.8%); the split attention's imbalance 1.1-1.3 ms
per token at depth.

Gaps: pp512 at depth 4096 has no table: with `-r 1` each process runs that graph once, and the counters leave each kind's
first graph out (to fix: keep it apart instead of dropping it). The board's timebase is not read yet. "Structures" in the
tables counts processes, because the hash holds kernel addresses, which differ from one process to the next.

### 7.2 Rank 1, candidate #6: one quantization for Q/K/V and for gate/up (prepared 2026-10-21; user request)

What it does. `FLAGOS_SPACEMIT_QUANTIZE=shared` (test-only until adopted; read at every graph; `separate` is the default)
makes `graph_compute` leave out a `MUL_MAT`'s quantize step when the matmul that last quantized took the same activations
(same data, shape and strides) into the same layout (`spacemit_mul_mat_same_activations`, `flagos-spacemit-ime.cpp`),
and no node since wrote over them; the matmul's multiply step reads what that step wrote. On Qwen3, V and K use Q's
quantization and up uses gate's: 3 steps fewer per layer, 108 per token. The same bytes by construction.

What it needed (in the default path too). The matmuls' quantized activations have their own part of the workspace,
after the other ops' scratch (`spacemit_step::workspace_offset`, applied in `spacemit_run_step`). Before, every
workspace user started at offset 0, and between Q's matmul and V's runs RoPE, which uses the workspace in its reference
and wide-head paths (not in the K3's head-128 path). The workspace is now the sum of the two parts, not the larger.
Alternatives not taken: skipping only where no op between uses the workspace (K from V and up from gate: 72 of the 108
steps, no layout change, but it misses Q's share and depends on which RoPE path runs); a second workspace pointer in
`spacemit_tile` (every executor's interface changes).

Checks (`flagos-check-spacemit`, layer check): `shared` against `separate` byte for byte, down projections Q4_0 and Q4_1,
1 and 7 tokens, in one launch and in one launch per step (the quantized activations then outlast the launches between),
and 3 steps fewer per layer (`spacemit_steps_take`, a test-only count of launched steps). On the Mac, 2026-10-21:
identical in 4/4 cases and 3 steps fewer in 4/4, serial and on threads with the TCM stand-in; `test-backend-ops` with
`shared`: `MUL_MAT` 3/3, `ROPE` 165/165, `RMS_NORM_MUL_ADD` 30/30, `ROPE_SET_ROWS` 24/24, `FLASH_ATTN_EXT` 1324/1324. The
spine-runtime executor's changed code compiles against a stub of `spert.hpp`; the K3's `prepare` builds it for real.
On the K3, 2026-10-21 (`fusion-experiments.sh prepare`): build A without errors; `shared` identical in 4/4 cases and 3
steps fewer in 4/4; the layer's max NMSE 5.24e-04 as in run 4; the checks' rc 1 only from E3b's two 1-token layer cases
(launches ordered by `join`), as since 2026-10-20 (`structural-experiments.md` F16). The default checks no longer run
E3b's launches since the same day (user decision; `structural-experiments.md` §5), so their rc is 0 when all pass.

Predictions (estimates, fixed before the run; Qwen3-4B Q4_0):

| What | Prediction |
|---|---|
| steps | 1,086 -> 978 per generated token (1,122 -> 1,014 from n_kv 512); prompt graphs also 108 fewer; `MUL_MAT.quantize attn_v`, `attn_k` and `ffn_up` gone, no other name's count changed |
| outputs | identical logits (KLD 0, same top p 100%) at 512-token batches and at one token per ubatch; the same greedy text |
| removed time, generation | the three names' wall in `ops` together 0.10-0.20 ms per token: 108 steps of about 1-1.5 µs with their barrier |
| tg128 | +0.1-0.2%, within llama-bench's spread: the counters decide |
| pp512 | +0.3-0.6%: 108 quantize passes of 5 MiB read and 1.3 MiB written, 0.15-0.3 ms each (16-32 ms of 5.36 s per ubatch) |
| pp128 | +0.2-0.6% (1.25 MiB activations; how much the caches hold is not known) |
| side effects | V's, K's and up's multiply steps no slower in generation; in prompt graphs at most about 0.1 ms each per ubatch (their quantized rows may come from DRAM rather than the caches) |
| the counters' cost | `ops` against `ref` at most 1% (E3c's counters cost 0.5% on tg128) |

Rules (proposed 2026-10-21; the user fixes them before the run):
- Valid if the outputs are identical, the steps and names change as predicted, and the counters cost at most 1%.
- Adopt `shared` as the default if valid, the counters show a total saving of at least 0 per graph in generation and in
  pp512 with no name slower beyond the run's spread, and no llama-bench test is slower than `ref` by more than the larger
  of the two modes' spreads. The ranking's thresholds (0.5% of tg128, 1% of pp512) decide whether to build a candidate;
  #6 is built, and identical outputs make keeping it cheap.
- Either way, the `ops` tables (the defaults) re-rank the remaining candidates by measured ceilings.

How to run (on the K3, inside tmux, after the baseline of §7.1, from the same build):

```bash
EXPS=q6 docs/flagos-k3/scripts/fusion-experiments.sh run ~/models/Qwen3-4B-Q4_0.gguf ~/fx/q6 > ~/fx-q6.log 2>&1
docs/flagos-k3/scripts/fusion-experiments.sh summary ~/fx/q6 > ~/fx-q6-summary.txt; head -40 ~/fx-q6-summary.txt
```

- `prepare` (run once, before the baseline) builds build A's tools and `flagos-check-spacemit` and runs the checks; its
  layer line ends with "one quantization for Q/K/V and for gate/up (FLAGOS_SPACEMIT_QUANTIZE=shared) identical in 4/4
  cases, 3 steps fewer in 4/4".
- `run`: the greedy text for `ref` and `q6`; 5 repetitions of pp128 with tg128 (`-ub 128`) and of pp512, the modes `ref`,
  `ops`, `q6` and `q6-ops` interleaved; then the logits (8 chunks) at 512-token batches and with `-ub 1 -b 512`. About
  40 minutes (estimate); `DEPTH=1` adds tg32 at depths 0, 1024 and 4096 and pp512 at depth 4096 (about an hour more).
- The summary: the texts; llama-bench per mode against `ref`, and the counters' cost; E3c's budget of a generated token
  for `ops` and `q6-ops`; the `ops` table per kind of graph; `q6-ops` against `ops` name by name; the logits.

### 7.3 Path A for small prompt batches (not fusion; prepared and measured 2026-10-21; user request, from the baseline)

The baseline found gate and up at 128 rows on path C, 9.56 ms per layer, against 6.63 ms for the down projection of the
same size on path A (§7.1). The split copied from the in-tree IME path (`flagos-spacemit-ime.cpp:299-306`) takes path A
only when there are at least as many blocks of 16 rows as tiles and the matrix is at most 64 times wider than the rows:
so every batch below 113 rows, and gate and up up to 149 rows, run on path C. That covers llama-bench's pp128 at
`-ub 128`, every prompt of fewer than 113 tokens, and a prompt's last partial batch.

What it does. `FLAGOS_SPACEMIT_MM=rows` (test-only until adopted; read at every call that gives each tile 4 rows;
`split` is the default) takes path A whenever each tile gets at least one group of 4 rows, 32 rows or more on 8 tiles,
whatever the shape. The same bytes by construction: both paths run the 4-row kernel on the same groups of 4 rows and
blocks of 32 columns. Path A keeps the 4 rows in TCM and streams the weights past them; path C reads both from the
caches and DRAM. Alternatives not taken: a new threshold in place of 64 (it would guess the rule before measuring it);
path A for every batch of several rows (below 32 rows some tiles would have none).

Checks (`flagos-check-spacemit`, `check_mm_paths`): `rows` against `split` byte for byte for Q4_0, Q4_1, Q8_0 and Q6_K at
32 to 128 rows, including a last group of 1 row and a matrix 65 times wider than its 128 rows, with the paths taken
counted (`spacemit_mm_paths_take`), so a case that does not switch fails. On the Mac, 2026-10-21: 12 of 12 identical
with the path switched, serial and on threads with the TCM stand-in; `test-backend-ops -o MUL_MAT` with `rows` 3/3.

Measurement:
- `flagos-check-spacemit --expect-device --bench-mm`: Qwen3-4B's 6 matmul shapes at 16 to 512 rows under both settings,
  each run reading its weights from DRAM, with the path each setting took, and per batch the shapes weighted as a batch
  runs them. 16 rows (path C both ways) and 160 to 512 rows (path A both ways) are controls.
- `EXPS=mm` in `fusion-experiments.sh`: modes `ref`, `ops`, `mm`, `mm-ops`; pp128 with tg128 (`-ub 128`), pp512, and
  prompts of 32, 64 and 96 tokens (one batch each); the logits at batches of 128 and of 64 rows, where the paths differ.

Predictions (estimates, fixed before the run; Qwen3-4B Q4_0):

| What | Prediction |
|---|---|
| outputs | identical: KLD 0 and the same top token at `-ub 128` and `-ub 64`; the same greedy text |
| bench, gate and up at 128 rows | 9.6 ms on path C, 6.6-7.5 ms on path A: down's rate if K 2560's shorter rows cost nothing, up to 15% below it if they do |
| bench, every shape at 64 and 96 rows | path A 15-35% faster |
| bench, every shape at 32 rows | between 20% faster and 20% slower: each tile streams all the weights for its one group of 4 rows |
| bench controls | 16 rows and 160-512 rows within 3% |
| pp128 (`-ub 128`) | +12-17% (92.1 t/s to 103-108): the 70 gate and up steps 2.0-2.9 ms faster each |
| pp64, pp96 | +15-30% |
| pp32 | -10% to +20% |
| pp512, tg128 | unchanged within 0.5% (path A and the 1-row path already) |

Rules (proposed 2026-10-21; the user fixes them before the run):
- Valid if the outputs are identical and the bench's controls stay within 3%.
- Adopt a new split, path A whenever every tile gets 4 rows, if the bench finds path A faster or within 3% at every shape
  from 32 rows, and llama-bench finds pp32, pp64, pp96 and pp128 no slower than `ref` beyond the larger spread and pp128
  at least 5% faster. If path A loses somewhere, adopt it only from the measured crossover (per shape if needed).

How to run (on the K3, inside tmux, after a bundle of the commit):

```bash
cd /root/llama.cpp-flagos && git -c safe.directory=$PWD pull --ff-only ~/sx-fx3.bundle feature/flagos-spacemit
docs/flagos-k3/scripts/fusion-experiments.sh prepare 2>&1 | tee ~/fx-prepare2.log
~/tcmtest/tcmrelease --apply && build-flagos/bin/flagos-check-spacemit --expect-device --bench-mm > ~/fx-bench-mm.log 2>&1
grep -E "^bench    mm" ~/fx-bench-mm.log
EXPS=mm REPS=3 docs/flagos-k3/scripts/fusion-experiments.sh run ~/models/Qwen3-4B-Q4_0.gguf ~/fx/mm > ~/fx-mm.log 2>&1
```

- `prepare` builds and runs the checks: rc 0 expected (E3b left them), its `mm` line "12/12" on the Mac becomes "6/6" on
  the K3 (no threaded host executor there).
- The bench takes about a minute (estimate); the run about 20 minutes (4 modes, 3 repetitions, 4 logits runs).
- Then on the Mac: `scp -r k3:/root/fx/mm ~/fx-mm && scp k3:/root/fx-bench-mm.log ~/`. To rebuild the summary there,
  give the run's experiments: `EXPS=mm docs/flagos-k3/scripts/fusion-experiments.sh summary ~/fx-mm` (without `EXPS`
  the script assumes `q6` and leaves out the `mm-ops` against `ops` tables).

**Bench results (K3, 2026-10-21, `--bench-mm`; the q lines and the 16-row k and v line added from the log the same
day).** Valid: the controls stay within 3% (16 rows, path C both ways: -2.8 to +0.9%, k and v the -2.8; where the
split already takes path A, 128-512 rows: -2.4 to +0.8%, k and v and once down at -1.5 to -2.4%). Where the paths
differ, path A is faster at every shape and row count:

| Rows | q | k, v | attention output | gate, up | down | down Q4_1 | per batch, matmuls but the head |
|---|---|---|---|---|---|---|---|
| 32 | -33.2% | -25.0% | -36.2% | -33.5% | -34.7% | -35.4% | 398 -> 264 ms (-33.6%) |
| 48 | -16.8% | -47.6% | -49.9% | -16.3% | -53.8% | -53.3% | 803 -> 512 ms (-36.3%) |
| 64 | -19.6% | -21.6% | -26.2% | -34.0% | -26.1% | -24.7% | 745 -> 527 ms (-29.2%) |
| 96 | -58.8% | -57.3% | -60.5% | -31.2% | -62.2% | -62.5% | 1,598 -> 790 ms (-50.6%) |
| 128 | path A both | path A both | path A both | -26.3% (9.72 -> 7.17 ms) | path A both | path A both | 1,248 -> 1,063 ms (-14.8%) |

- Rates: path A reaches 870-920 GFLOP/s for Q4_0 (k and v 770-820, Q4_1 610) wherever every tile gets the same
  number of 4-row groups, 620-710 at 48 rows (12 groups for 8 tiles: four tiles take two; Q4_1 460). Path C reaches
  550-710 at 32 and 64 rows (Q4_1 390-460) and only 320-370 at 48 and 96 rows (Q4_1 215-230), except where it splits
  only columns (below): gate and up at both, q at 48 rows, 570-630.
- Path C's dips come from its task split (read in code, `flagos-spacemit-ime.cpp:337-343` and the task loop after it):
  for matrices at most 64 times wider than the rows it cuts blocks of 16 rows and column chunks aiming at 8 tasks; at 48
  rows that gives 9 tasks (3 x 3) and at 96 rows 12 tasks of two sizes, so the busiest tile does twice its share; at 32
  and 64 rows exactly 8. A matrix more than 64 times wider than the rows has only its columns split, into 8 balanced
  tasks: gate and up up to 149 rows, q up to 63, the attention output and down up to 39; so q at 48 rows gains only
  17%, as gate and up do. So today a 96-row batch's matmuls take longer than a 128-row batch's (1,598 against 1,248 ms).
- Against the predictions: gate and up at 128 rows 7.17 ms (6.6-7.5 predicted); 64 rows 22-34% faster (15-35); 96 rows
  31-63% (15-35, beyond); 32 rows 25-36% faster (between 20% faster and 20% slower predicted, beyond).
- The bench part of the adoption rule is met; the llama-bench and logits parts wait for `EXPS=mm`. Estimates from these
  numbers (the other ops at 32-96 rows scaled from the baseline's 128-row batch): pp128 about +15% (gate and up 2.56 ms
  faster in each of 70 steps, 179 ms of 1.39 s); pp32 about +40%, pp64 about +35%, pp96 about +85%, the last because
  path C's split makes today's pp96 slower per token than pp128.

**How SpacemiT's code handles small batches (read 2026-10-21: ggml-spacemit `ime.cpp` @ `4e782bc`, lines 372-548; the
in-tree `ggml-cpu/spacemit/ime.cpp:395-500`).** Both use the same split rule and the same path A, 4-row groups dealt
round-robin. But where path A does not apply and TCM holds a 32-column slab, both take path B, not path C: slabs dealt
round-robin to the cores, each core multiplying all of the batch's rows against its slab, the two cores of a pair taking
turns. Splitting by columns keeps path B balanced at any row count; path C is only their fallback without TCM.
ggml-spacemit's rewrite of path B (`has_pair` for an odd core count; both cores of a pair stepping through the same
slab bases) makes it safe, not more balanced. So the collapse of path C at 33-112 rows is the provider's own: M2b
deferred path B and runs path C in its place. Path A's rounding is the same in SpacemiT's code, but they take path A only
from 113 rows, where it stays at 82-100% balance by group count (modelled from the code; the worst, 82.5%, at 129 rows:
33 groups in 5 rounds; corrected 2026-10-21 from 85-100%). Real prompts are rarely a multiple of 32 rows
(a prompt under 512 tokens is one batch of that many rows; a longer one ends with a partial batch), so path A from 32
rows would bring its 56-94% balance into 33-102 rows. Open: which serves 33-112 rows best, path A with a balanced split,
or path B for several rows (the provider's 1-row path is path B for one row); estimates for both below.

**`EXPS=mm` results (K3, 2026-10-21; build `90cf54f5f`, Qwen3-4B Q4_0, 3 runs per mode, the modes interleaved; all 42
runs rc 0).** Valid by the rules: the same greedy text; at `-ub 128` and `-ub 64` mean KLD 0.000000, maximum 0.000055
and the same top token 100%, the floor that the base file's 16-bit log-probabilities leave for identical logits
(`tools/perplexity/perplexity.cpp:79-93`; the same values for identical logits in `structural-experiments.md`, runs 1
and 4); the bench's controls within 3%. The counters cost at most 0.9% (tg128), as in the baseline.

| t/s, mean ± sd of 3 | pp32 | pp64 | pp96 | pp128 (`-ub 128`) | pp512 | tg128 |
|---|---|---|---|---|---|---|
| `ref` (`split`) | 71.55 ± 0.20 | 79.46 ± 0.03 | 57.05 ± 0.03 | 92.06 ± 0.11 | 94.93 ± 0.05 | 11.27 ± 0.01 |
| `mm` (`rows`) | 101.47 ± 0.44 | 108.74 ± 0.66 | 107.84 ± 0.15 | 106.20 ± 0.17 | 95.27 ± 0.66 | 11.29 ± 0.01 |
| change | +41.8% | +36.9% | +89.0% | +15.4% | +0.4% | +0.1% |
| predicted before the bench | -10 to +20% | +15-30% | +15-30% | +12-17% | within 0.5% | within 0.5% |
| estimated from the bench | about +40% | about +35% | about +85% | about +15% | | |

- The predictions fixed before the bench hold for pp128, pp512 and tg128 and fall short for pp32, pp64 and pp96: they
  expected every tile streaming all the weights for its group to cost path A at 32 rows, and missed path C's split at
  96 rows. The estimates from the bench, made before llama-bench ran, are within 4 points of every result.
- Op counters per batch (`mm-ops` against `ops`): 32 rows 447 -> 315 ms (-29.6%), 64 rows 803 -> 589 (-26.6%), 96
  rows 1,680 -> 888 (-47.2%), 128 rows 1,391 -> 1,203 (-13.5%); where no path changes, 512 rows +0.4% (the runs'
  spread) and a generated token 88.27 -> 88.28 ms. At 128 rows only gate and up move (9.58 -> 6.89 ms per step, 94 ms
  each per batch); the matmuls that already took path A move by less than 0.2%. At 96 rows 509 of the 792 ms saved
  are path C's imbalance (512 -> 3 ms per batch), the rest path A's rate; at 32 and 64 rows path C was balanced
  (imbalance 9.5 and 8.1 ms per batch, with `rows` 1.8 and 2.9) and the gain is mostly rate. The other ops together
  move by -0.5 to +0.1 ms per batch.
- With `rows`, prompts of 32-128 tokens run at 101-109 t/s, above pp512's 95 (whose matmuls already take path A;
  attention takes 0.67 s of its 5.4). Today a 96-token prompt runs at 57 t/s against 92 for 128 tokens; with `rows`,
  108 and 106.
- By the rule proposed above, the conditions for adoption are met at every row count measured. But those are path A's
  best case: 32, 64, 96 and 128 rows give every tile the same number of groups (48 rows, only in the bench, gives four
  tiles two). Not adopted (below).

**What the measured row counts leave out (a model; estimates, not measured).** Path A deals the 4-row groups
round-robin (`flagos-spacemit-ime.cpp:353`), so a batch that is not a multiple of 32 rows ends with a round in which
some tiles wait: at 36 rows tile 0 multiplies a second group while the other seven wait. Path C splits only the
columns of a matrix more than 64 times wider than the rows, into 8 balanced tasks (above): below 40 rows that is every
shape but k and v. A model fitted to the bench (path C: the busiest tile's rows x columns at the per-tile rate the
bench measured for that kind of task; path A: one group's time per round, as the 48-row bench's second round took)
gives the five measured row counts within 5%, and llama-bench's `split` at pp32-pp128 within 2% and `rows`' gains
within 5 points. What the bench cannot give is the cost of a round with few tiles. At 32 rows each tile reads all of
gate's 14.0 MB in 1.73 ms, 65 GB/s of loads together against the 28 GB/s DRAM ceiling, so most of path A's reads must
hit a cache shared by tiles that read the same weights at the same time (inference from the times and the loop,
`:353-359`). A last round of 1-3 tiles has little to share: the optimistic case prices its groups as with 8 tiles, the
pessimistic one at 3.1 GB/s per tile (one load in flight; `plan.md`, "1-row matmul bandwidth", an estimate; 3.0 GB/s
measured per tile with all 8 streaming). A partial group (1-3 rows, the 1-row kernel row by row, `:112-120`) is priced
as a full one. Per batch, the matmuls but the head, against today (llama-bench t/s in brackets, from the same model):

| Rows | Today | `rows`, optimistic | `rows`, pessimistic | Balanced split (not built) |
|---|---|---|---|---|
| 33-39 | 448-519 ms | +2 to +18% (pp33 -13%, pp36 -7%) | +78 to +106% (pp33 -48%, pp36 -44%) | -22 to -33% (pp36 +31%) |
| 40-44 | 728-756 ms | -27 to -30% (pp40 +32%) | +22 to +27% (pp40 -20%) | -51 to -52% |
| 45-64 | 644-785 ms | -18 to -33% (pp48 +41%) | the same | -28 to -49% |
| 65-76 | 1,289-1,353 ms | -38 to -41% (pp65 +55%) | -8 to -12% (pp65 +8%) | -52 to -54% |
| 77-96 | 1,359-1,652 ms | -42 to -52% | the same | -51 to -56% |
| 97-108 | 1,839-1,904 ms | -43 to -44% (pp100 +66%) | -21 to -24% (pp100 +25%) | -52 to -53% |
| 109-128 | 1,220-1,927 ms | -13 to -45% | the same | -18 to -52% |
| 129-140 | 1,454-1,518 ms (pessimistic 1,650-1,714) | -9 to -13% (pp129 +9%) | 0 to +4% (pp129 -4%) | -21 to -24% |
| 141-149 | 1,524-1,571 ms | -13 to -16% | the same | -20 to -23% |

- So `rows` as tested is predicted to make prompts of 33-39 tokens slower than today even if a round of few tiles
  costs nothing extra (path C is balanced there for every shape but k and v), and prompts of 33-44 tokens much slower
  if it does; faster everywhere else. Today's default has the same last rounds wherever it already takes path A, from
  113 rows: 129-140, 161-172, ... rows, every batch whose groups leave 1-3 over (a 512-row batch leaves none).
- The balanced split (not built): after the full rounds, the leftover groups multiplied by all 8 tiles, each on an
  eighth of the columns in whole 32-column blocks: the same kernel calls on the same groups and blocks, so the same
  bytes. Modelled as the full rounds plus the larger of the weights read once at 24 GB/s and the leftover work spread
  over 8 tiles: 18-56% faster than today at every row count from 33 to 149.

**SpacemiT's rule in full has never run in the provider (history read 2026-10-21 in git and `plan.md`).** The split,
path A and path C came from ime.cpp with M2b (commit `6671c59`, 2026-10-09) and have not changed since. Path B was
deferred (`plan.md`, M2b design): tg128 and perplexity do not use it; the in-tree version deadlocks when a pair's slab
counts differ; it needs a wait inside a step; its gain over path C was unmeasured; it was to follow if the M2b bench
found path C clearly slow at 4-64 rows. M2d called it not needed (pp128 83.0 against the IME path's 79.5, a whole-model
comparison). Path B came for 1 row only, by measurement: weight slabs staged in TCM (`507cb52`, 2026-10-11; 10% less
per token than the direct reads M2b had copied from ggml-spacemit) and the pair turns (`b8187d2`, 5% less again), which
brought the first wait inside a step (`spacemit_pair_wait`) and handle pairs with unequal slab counts. So at 2-112 rows
(gate and up 2-149) the provider has always run path C where both SpacemiT versions run path B, and no test has
compared the two: `scripts/ime-gemv-bench.cpp` times the IME path at 1 row only, and the IME path's whole-model runs
differ in every other op as well. Two of the four reasons no longer hold (the wait inside a step, unequal slab counts),
and the bench above meets the trigger (path C at 32-112 rows 320-710 GFLOP/s against path A's 870-920). Read in their
path B besides: when the batch's quantized activations fit in TCM next to a slab they are copied there too
(`ggml-cpu/spacemit/ime.cpp:438-441`), so both operands come from TCM: at Qwen3-4B's shapes up to 119 rows at K 2560,
68 for the attention output, 19 for down (18 for Q4_1). Its 32-column slabs are dealt round-robin, 4, 10, 16 and 38 per
tile at Qwen3-4B's widths, so it is balanced at any row count.

A model of their rule (estimates): path B per tile as the activation copy plus slabs x (slab copy + multiply) plus the
pair waits, since in the turn schedule each tile copies its own slabs (`flagos-spacemit-ime.cpp:239-257`); calibrated on
the baseline's 1-row `paired` kernel times (0.265 us per KB of slab, about three quarters of it copying) and on path A's
per-call time; copies at 0.10-0.20 us per KB, multiplies with the activations outside TCM at 1.0-1.4 times. Per batch,
against today: 16-31 rows -11 to -24% (pessimistic -10 to +2%), 32-112 rows -21 to -55% (-10 to -48%), 113-149 rows,
where only gate and up change, -16 to -18% (0 to -16%); the same paths at 1 row and from 150 rows. Against `rows`
slower at the multiples of 32 that favour path A (32 rows +14%, 64 +8%, 96 +5%, 128 +2%; pessimistic +22 to +34%),
faster at the others; against the balanced split -9 to +15%. Below 16 rows both paths are bound by reading the weights
(at least 70 ms per batch at the 1-row path's rate, 86 ms at the kernels' own loads), and the model does not apply.

**Next, as proposed before the odd-length run (item 1 ran the same day; results below).**
1. No code change, one session: llama-bench at prompts of 16, 32, 33, 36, 37, 40, 44, 48, 64, 65, 96, 100, 128 and 129
   tokens under `split` and `rows`, and the IME path (SpacemiT's rule end to end) at the same prompts. Ours separates
   the two cases (pp36 -7% or -44%; pp40 +32% or -20%) and tests today's last rounds (pp129 against pp128 under
   `split`: -8% optimistic, -18% pessimistic). The IME path's curve shows whether path B leaves out our steps
   (predicted: no dip at 40-48 or 96 rows, pp33-pp40 within 7% of pp32) and whether its path A has the same last round
   (predicted: pp129 at least 4% below pp128).
2. Then test-only settings, each checked byte for byte like `rows`: path B for several rows, SpacemiT's rule
   (`FLAGOS_SPACEMIT_MM=spacemit`: the 1-row `paired` code extended to the batch's rows, activations in TCM where they
   fit), first by the estimate, since it is SpacemiT's own design and also covers 2-31 rows; the balanced split
   (`balanced`) if path B falls short. The bench at every row count from 2 to 160 under all settings, then llama-bench
   at the prompts of 1. Adopt a setting if it is within 3% of the fastest at every row count and llama-bench agrees;
   whether path A should keep the multiples of 32 where it wins is a later question, settled by measurement.

**Odd prompt lengths (K3, 2026-10-21; build `90cf54f5f`, Qwen3-4B Q4_0, llama-bench `-r 3 -n 0`, every prompt one
batch; ours with `-fa auto`, the IME path `build-ime` with `-fa 1`).** t/s, standard deviations at most 0.8:

| Prompt | `split` | `rows` | IME path | `rows` against `split` (predicted: optimistic / pessimistic) | IME against `split` | IME against `rows` |
|---|---|---|---|---|---|---|
| 16 | 69.71 | 70.24 | 86.65 | +0.8% (the same path) | +24.3% | +23.4% |
| 32 | 71.23 | 101.20 | 85.88 | +42.1% (+39%) | +20.6% | -15.1% |
| 33 | 63.53 | 67.32 | 81.15 | +6.0% (-13% / -48%) | +27.7% | +20.5% |
| 36 | 65.64 | 64.94 | 83.93 | -1.1% (-7% / -44%) | +27.9% | +29.2% |
| 37 | 64.21 | 66.11 | 80.74 | +3.0% (-5% / -43%) | +25.7% | +22.1% |
| 40 | 49.45 | 70.66 | 86.28 | +42.9% (+32% / -20%) | +74.5% | +22.1% |
| 44 | 51.94 | 75.84 | 85.23 | +46.0% (+37% / -17%) | +64.1% | +12.4% |
| 48 | 54.77 | 80.56 | 88.09 | +47.1% (+41%) | +60.8% | +9.3% |
| 64 | 79.60 | 108.69 | 90.74 | +36.5% (+33%) | +14.0% | -16.5% |
| 65 | 48.63 | 84.91 | 87.57 | +74.6% (+55% / +8%) | +80.1% | +3.1% |
| 96 | 57.25 | 108.15 | 87.08 | +88.9% (+91%) | +52.1% | -19.5% |
| 100 | 53.17 | 88.77 | 85.41 | +67.0% (+66% / +25%) | +60.6% | -3.8% |
| 128 | 92.35 | 106.53 | 79.49 | +15.4% (+20%) | -13.9% | -25.4% |
| 129 | 85.97 | 93.42 | 74.04 | +8.7% (+9% / -4%) | -13.9% | -20.7% |

- Our two settings reproduce `EXPS=mm` within 0.5% at 32, 64, 96 and 128 tokens; the model of `split` is within 5% at
  13 of the 14 lengths (pp16 +7%).
- `rows` is never more than 1.1% slower than `split` (pp36) and 3-89% faster at 12 of the 14 lengths. The pessimistic
  case is refuted wherever it differed. The optimistic case holds from 40 tokens (the gains within -5 to +20 points of
  it) but not at 33-37 tokens: predicted 5-13% slower, measured -1.1 to +6.0%. The predicted loss at 33-39 tokens did
  not appear.
- What a round of path A costs, from the differences of `rows`' times per prompt: one more round with a full group on
  one tile adds 238-239 ms per batch (32 -> 36 tokens, 96 -> 100), with a 1-row group 174-179 ms (32 -> 33, 64 -> 65,
  128 -> 129); both include the other ops' growth, about 1.5 ms per row (op counters at 32 rows). A round in which all 8
  tiles work costs about 243 ms (the multiply steps of the 32-row matmuls with `rows`). So a round costs about the same
  whether one tile works in it or eight. A tile alone reads its weights about as fast as eight do together (gate's 14
  MB in about 1.7 ms), so each tile's 4-row kernel sets path A's pace, not DRAM (inference), and the pessimistic case's
  3.1 GB/s does not apply to that kernel.
- Under `split`, pp129 is 6.9% below pp128 (predicted -8% optimistic, -18% pessimistic), and so is the IME path's
  (-6.9%): the same last round of the same path A in both.
- The IME path, SpacemiT's rule end to end: no dip at 40-48 or 96 tokens, pp33-pp40 within 6% of pp32, pp129 6.9%
  below pp128; the three predictions hold. From 16 to 100 tokens it runs at 80.7-90.7 t/s, faster than our `split` at
  every length (by 14-80%) and than `rows` at every length below 100 that is not a multiple of 32 (by 3-29%); `rows`
  wins at 32, 64, 96 and 128 tokens (the IME path 15-25% slower), at 100 and at 129.
- How much of that is path B (inference, not isolated): below 64 tokens the IME path's attention is the same code as
  ours (the one-chunk kernels; M2e.3's L1, 2026-10-13), and from 64 its tiled kernel is slower than ours (L3). If its
  other ops cost at least what ours do (op counters: attention and the other ops 47 ms of a 32-row batch, 63 of a
  64-row one, 104 of a 96-row one), its matmuls take at most 326, 642 and 999 ms against our path C's 400, 740 and
  1,577 (multiply and quantize steps, `ops`) and our path A's 268, 527 and 784 (`mm-ops`). Path B's matmuls then take
  at least 19%, 13% and 37% less time than path C's there, and more than path A's at those multiples of 32 unless the
  IME path's other ops are slower than ours. At 36 tokens the same bound, with our other ops scaled from 32 rows, puts
  path B at least 24% below both of our settings.
- Our attention costs less at 64 rows (22.9 ms per batch, the tiled kernel) than at 32 (27.6 ms, the one-chunk
  kernels), which most likely is why a 64-token prompt takes `rows` less time than a 48-token one (589 against 596 ms;
  48 rows were not counted).

**Next (proposed; waits for the user).** Path B for several rows (`FLAGOS_SPACEMIT_MM=spacemit`, item 2 above), now
first: the IME path puts it ahead of both our settings at 16-48 tokens. Predicted for the port, to fix before its run:
from 16 to 48 tokens at least the IME path's t/s (80.7-88.1), since below 64 tokens our attention is the same code and
our other ops are assumed no slower; below `rows` at 32, 64 and 96 tokens. Then a rule per row count from the
measurements: path A where its last round is full (the multiples of 32), path B elsewhere, or the balanced split for
path A's last round. Meanwhile `rows` is never more than 1.1% slower than today in this run and could become the
default before path B (the user's choice).

**Path B for several rows (prepared 2026-10-21; the user's decision after the review of its deferral, `plan.md` M2b
design).** `FLAGOS_SPACEMIT_MM=spacemit` (test-only until adopted; read at every call with several rows) takes path B
where the split gives no path A and TCM holds a 32-column slab, which is SpacemiT's rule (`gemm_slabs`,
`flagos-spacemit-ime.cpp`). It is upstream's schedule (`ggml-cpu/spacemit/ime.cpp:435-494`): slabs dealt round-robin,
copied into TCM and multiplied there for all the batch's rows; the batch's quantized activations in TCM too when they
fit next to a slab (`:438-441`), otherwise read from the workspace; the two tiles of a pair taking turns as the 1-row
`paired` code does, whose waits (`spacemit_pair_wait`) it uses. The same bytes as paths A and C: `gemm_rows` over rows 0
to m for each block of 32 columns, the same kernel calls on the same groups of 4 rows. `split` stays the default with
path C, which also remains the path without TCM; `rows` and the 1-row path are unchanged.

Checks (`flagos-check-spacemit`, `check_mm_paths`, now all three settings): 13 cases at 2-128 rows in Q4_0, Q4_1, Q8_0
and Q6_K, among them a pair with uneven slab counts (13 slabs), tiles without a slab (3 slabs) and activations that do
not fit in TCM (K 9728, Q4_0 and Q4_1); each setting's paths counted; the output filled with NaN before every run (the
earlier check read back whatever the previous run had left in place) and split's required finite. On the Mac,
2026-10-21: 26/26 identical with the expected paths, with the tiles in turn and on threads (the TCM stand-in); two
mutations caught (skipping each tile's last slab: 26/26 fail; the slab over the activations: 22/26 fail, the 4 others
being the cases whose activations stay in the workspace); ThreadSanitizer over the whole check program: 0 reports;
`test-backend-ops -o MUL_MAT` 3/3 under each setting, in turn and on threads.

Measurement:
- `flagos-check-spacemit --expect-device --bench-mm`: Qwen3-4B's 6 matmul shapes at 25 row counts from 2 to 512 under
  `split`, `rows` and `spacemit`, each with the path it took, and per batch. Controls: where the settings take the
  same path (`rows` below 32 rows; all three from 150 rows, path A).
- `EXPS="mm pb"` in `fusion-experiments.sh`: modes `ref`, `ops`, `mm`, `mm-ops`, `pb`, `pb-ops`; pp128 with tg128
  (`-ub 128`), pp512, and prompts of 2, 4, 8, 16, 32, 33, 36, 37, 40, 44, 48, 64, 65, 96, 100, 128 and 129 tokens (one
  batch each); the logits at batches of 128, 64 and 16 rows.

Predictions (estimates, fixed before any K3 run; Qwen3-4B Q4_0; from the model and the IME path's totals above):

| What | Prediction |
|---|---|
| outputs | identical: KLD at the floor, the same top token at `-ub 128`, `-ub 64` and `-ub 16`; the same greedy text |
| bench, path B against path C | faster at every shape from 16 rows; per batch 10-55% faster at 16-149 rows (model), at least 13-37% at 32-96 rows (IME bound) |
| bench, path B against path A (`rows`) | slower at 32, 64 and 96 rows; faster at 33-44 rows |
| bench controls | within 3% |
| pp16-pp48 | at least the IME path's 81-88 t/s (today 49-71) |
| pp32, pp64, pp96 | below `rows` (101, 109, 108 t/s) |
| pp65, pp100 | within 10% of `rows` |
| pp2-pp8 | -5% to +20% against `split` (bound by reading the weights; the model does not apply) |
| pp512, tg128 | unchanged within 0.5% (path A and the 1-row path) |

Rules (proposed 2026-10-21; the user fixes them before the run):
- Valid if the outputs are identical and the bench's controls stay within 3%.
- Adopt path B in place of path C (`spacemit` as the default) if llama-bench finds every tested length no slower than
  `split` beyond the larger spread, and the lengths where `split` runs path C (16-100 tokens) at least 5% faster.
- Path A against path B at 32-112 rows (whether the default should take path A where its rounds come out even): a
  separate decision, from the same data.

How to run (on the K3, inside tmux, after a bundle of the commit):

```bash
cd /root/llama.cpp-flagos && git -c safe.directory=$PWD pull --ff-only ~/sx-fx4.bundle feature/flagos-spacemit
docs/flagos-k3/scripts/fusion-experiments.sh prepare 2>&1 | tee ~/fx-prepare4.log
~/tcmtest/tcmrelease --apply && build-flagos/bin/flagos-check-spacemit --expect-device --bench-mm > ~/fx-bench-pb.log 2>&1
grep -E "^bench    mm per" ~/fx-bench-pb.log
EXPS="mm pb" REPS=3 docs/flagos-k3/scripts/fusion-experiments.sh run ~/models/Qwen3-4B-Q4_0.gguf ~/fx/pb > ~/fx-pb.log 2>&1
```

- `prepare`: rc 0 expected; its `mm` line "26/26" on the Mac becomes "13/13" on the K3 (no threaded host executor).
- The bench takes a few minutes (estimate); the run about 40 minutes (6 modes, 3 repetitions, 9 logits runs).
- Then on the Mac: `scp -r k3:/root/fx/pb ~/fx-pb && scp k3:/root/fx-bench-pb.log ~/`, and
  `EXPS="mm pb" docs/flagos-k3/scripts/fusion-experiments.sh summary ~/fx-pb` (with many prompt lengths the llama-bench
  table has a row per test).

**Bench results (K3, 2026-10-21, build `17413e6`, `--bench-mm`; per batch, the matmuls but the head, ms; the lines per
shape are in the log, not yet read).** `prepare`: rc 0, all checks passed. Valid: the controls stay within 3% (160-512
rows, path A in all three settings: -0.2 to +1.0%; below 32 rows `rows` takes `split`'s path C: -1.0 to 0.0%).

| Rows | `split` | `rows` | `spacemit` | fastest, against `split` |
|---|---|---|---|---|
| 2 | 95.7 | 95.5 | 83.2 | path B, -13% |
| 4 | 96.4 | 95.4 | 79.9 | path B, -17% |
| 8 | 119.2 | 119.0 | 98.0 | path B, -18% |
| 16 | 204.5 | 203.2 | 153.9 | path B, -25% |
| 17 | 237.1 | 235.6 | 172.5 | path B, -27% |
| 24 | 299.1 | 299.1 | 249.4 | path B, -17% |
| 31 | 447.4 | 447.2 | 356.4 | path B, -20% |
| 32 | 399.6 | 262.7 | 318.2 | path A, -34% |
| 33 | 470.0 | 437.4 | 346.9 | path B, -26% |
| 36 | 489.1 | 497.7 | 355.4 | path B, -27% |
| 40 | 754.4 | 501.3 | 390.5 | path B, -48% |
| 44 | 780.6 | 505.8 | 425.9 | path B, -45% |
| 48 | 801.7 | 510.9 | 459.3 | path B, -43% |
| 64 | 748.7 | 525.7 | 590.9 | path A, -30% |
| 65 | 1,287.7 | 700.0 | 619.6 | path B, -52% |
| 96 | 1,599.2 | 791.1 | 926.1 | path A, -51% |
| 100 | 1,792.2 | 1,022.4 | 964.1 | path B, -46% |
| 112 | 1,827.4 | 1,036.2 | 1,069.6 | path A, -43% |
| 113 | 1,178.4 | 1,042.3 | 928.0 | path B, -21% |
| 128 | 1,245.0 | 1,052.4 | 1,374.5 | path A, -15% |
| 129 | 1,354.0 | 1,230.8 | 1,496.8 | path A, -9% |
| 149 | 1,486.8 | 1,293.8 | 1,643.3 | path A, -13% |
| 160, 256, 512 | 1,312.9, 2,102.3, 4,186.5 | +0.1, +0.0, +1.0% | -0.2, +0.2, +1.0% | the same path A |

- Against the predictions: path B is faster than path C at every row count from 2 to 113 (13-52% per batch; predicted
  10-55% at 16-149 rows and at least 13-37% at 32-96, which holds up to 113). At 128, 129 and 149 rows it is 10.4-10.5%
  slower, beyond the pessimistic case (0 to -16%): there only gate and up differ, and their activations (2,900 bytes a
  row) no longer fit in TCM next to a slab, which they do up to 119 rows; at 113 rows path B is still 21% faster. Path
  B against path A as predicted: slower at 32, 64 and 96 rows (+21%, +12%, +17%), faster at 33-44 (-16% to -29%);
  besides, faster at 48, 65, 100 and 113 rows, about level at 112 (+3%), slower from 128 (+22% to +31%).
- So by the bench, path B wins where path A's rounds do not come out even and the activations fit in TCM; path A wins
  at the multiples of 32 and where path B's activations would come from the workspace; `split` is never the fastest.
  SpacemiT's rule as built makes a batch of 120-149 rows slower than today (gate and up), so `spacemit` alone likely
  fails the adoption rule at pp128 and pp129 (a prediction; the llama-bench run tests it). The IME path's drop at 128
  tokens (79.5 t/s against 85-87 at 96-100) fits the same mechanism: from 120 rows it runs gate and up on path B with
  the activations outside TCM (inference).
- Next, from the lines per shape: whether down (activations outside TCM from 20 rows) also loses on path B; then a rule
  per call, path A where its rounds come out even or path B's activations do not fit, path B otherwise, designed with
  the llama-bench results.

**`EXPS="mm pb"` results (K3, 2026-10-21; build `17413e6`, Qwen3-4B Q4_0, 3 runs per mode, the modes interleaved;
all 66 runs rc 0).** Valid by the rules: the same greedy text in all three; at `-ub 128`, `-ub 64` and `-ub 16` mean KLD
0.000000, maximum 0.000055 and the same top token 100% for both `rows` and `spacemit` (the floor of identical logits);
the same perplexity, 9.7573, at every batch size; the counters cost at most 1% (pp2 and pp4 up to 3%, within their
spread). `split` reproduces the odd-length run within 0.5%.

| t/s, mean of 3 | `split` | `rows` | `spacemit` | `spacemit` against `split` | IME path (the odd-length run) |
|---|---|---|---|---|---|
| pp2 | 18.23 | 18.56 | 20.45 | +12.2% | |
| pp4 | 37.31 | 36.67 | 41.35 | +10.8% | |
| pp8 | 58.35 | 58.57 | 68.02 | +16.6% | |
| pp16 | 69.47 | 69.74 | 87.48 | +25.9% | 86.65 |
| pp32 | 71.41 | 101.45 | 86.81 | +21.6% | 85.88 |
| pp33 | 63.54 | 67.26 | 82.12 | +29.3% | 81.15 |
| pp36 | 65.89 | 64.77 | 86.68 | +31.6% | 83.93 |
| pp37 | 64.21 | 66.09 | 83.72 | +30.4% | 80.74 |
| pp40 | 49.39 | 70.64 | 87.69 | +77.5% | 86.28 |
| pp44 | 51.96 | 75.78 | 88.26 | +69.9% | 85.23 |
| pp48 | 54.83 | 80.39 | 88.86 | +62.1% | 88.09 |
| pp64 | 79.62 | 108.39 | 98.31 | +23.5% | 90.74 |
| pp65 | 48.68 | 84.93 | 94.98 | +95.1% | 87.57 |
| pp96 | 57.33 | 107.93 | 94.11 | +64.1% | 87.08 |
| pp100 | 53.22 | 88.76 | 93.66 | +76.0% | 85.41 |
| pp128 | 92.05 | 106.36 | 84.36 | -8.4% | 79.49 |
| pp129 | 86.04 | 93.33 | 78.86 | -8.3% | 74.04 |
| pp512 | 95.00 | 94.89 | 95.11 | +0.1% | |
| tg128 | 11.24 | 11.23 | 11.22 | -0.2% | |

- Against the predictions fixed before the run: pp16-pp48 at least the IME path's (82.1-88.9 against 80.7-88.1:
  holds, by 1-4%); below `rows` at 32, 64 and 96 tokens (holds); within 10% of `rows` at 65 and 100 (holds at 100,
  +5.5%; at 65 `spacemit` is 11.8% faster); pp2-pp8 -5% to +20% against `split` (holds, +11% to +17%); pp512 and tg128
  within 0.5% (hold). As the bench predicted, `spacemit` is 8% slower than `split` at 128 and 129 tokens. With
  SpacemiT's rule the provider runs 1-10% faster than the IME path at every length measured in both runs (two sessions
  the same day; `split` moved less than 0.5% between them).
- By the rule proposed above, `spacemit` is not adopted: pp128 and pp129 are slower than `split`. At 16-100 tokens it is
  22-95% faster.
- Per shape (the op counters of the same runs, microseconds per multiply step) path B wins where its activations fit in
  TCM next to a slab, even at the multiples of 32 for gate and up (32 rows: 1,609 against path A's 1,682 and path C's
  2,572), and loses where they do not: down from 32 rows (40 rows: path B 3,841, path A 3,196), the attention output at
  96 and 100 rows (path B 3,806, path A 2,117 at 96), gate and up at 128 and 129 (path B 11,375, path A 6,892 at 128).
  Where the activations fit, path A beats path B only for k and v at 32, 64 and 96 rows and for q and the attention
  output at 32 (by 2-22%), mostly the batch's small matrices.
- A rule per call from these numbers (an estimate composed of the measured steps, not run): path B where its activations
  fit in TCM next to a slab, else path A where every tile gets 4 rows, else path C. Its matmuls per batch come within 0-3%
  of the fastest path for each shape at every row count measured, never slower than `rows` or `spacemit`, and below both
  at sizes like 64 and 96 rows (against `rows` -6% and -8%). With the other ops as measured (the same in all three
  settings, within 1 ms), predicted t/s: pp16 87.5, pp32 101.7, pp36 88.8, pp40 91.9, pp48 96.9, pp64 114.9, pp65 103.9,
  pp96 115.5, pp100 105.7, pp128 106.5, pp129 93.6 (`split` 69.5-92.1). Open before building it: down at 20-31 rows,
  where its activations no longer fit and path A does not apply (path C or path B from the workspace; the bench's lines
  per shape at 24 and 31 rows decide).
- Settled by the bench's lines per shape (the same K3 session and build): down at 24 and 31 rows, its activations
  outside TCM and no path A, takes 2,190 and 3,094 us on path C against 2,588 and 3,737 on path B (Q4_1: 3,135 and
  4,464 against 3,391 and 4,855), so the rule's last branch is path C. Elsewhere the lines agree with the op counters,
  e.g. gate and up at 113 rows on path B 5,360 us against path A's 6,940.

**The rule per call as a fourth setting (prepared 2026-10-21; user decision).** `FLAGOS_SPACEMIT_MM=fit` (test-only
until adopted) takes path B where TCM holds the batch's activations next to a 32-column slab (the test `gemm_slabs`
itself uses, `slabs_hold_activations`), otherwise path A where every tile gets 4 rows, otherwise path C. At Qwen3-4B's
shapes: path B up to 119 rows for q, k, v, gate and up, up to 68 for the attention output, up to 19 for down (Q4_1:
18); path A beyond those from 32 rows; path C only for down at 20-31 rows. The 1-row path is unchanged; the same bytes
as the other three settings.

Checks (`check_mm_paths`, now four settings and 15 cases; the path each setting must take is worked out in the test
from the rules, so a case may expect any path): the two new cases are K 9728 at 33 rows (fit takes path A) and 113
rows, where the split gives path A (fit takes path B). On the Mac, 2026-10-21: 30/30 identical with the expected paths,
tiles in turn and on threads; three mutations of the rule caught (path B whatever the activations: the three cases
where they do not fit fail; no path A: the 33-row case; the split's path A first: the 113-row case);
`test-backend-ops -o MUL_MAT` 3/3 under `fit`, in turn and on threads; ThreadSanitizer over the whole check program: 0
reports.

Measurement: `--bench-mm` with a fourth column; `EXPS="mm fit"`: modes `ref`, `ops`, `mm`, `mm-ops`, `fit`, `fit-ops`;
pp128 with tg128 (`-ub 128`), pp512, the prompts of 2-129 tokens; the logits at batches of 128, 64 and 16 rows.

Predictions (estimates fixed before the run, from the steps measured per shape in the `EXPS="mm pb"` run):

| What | Prediction |
|---|---|
| outputs | identical: KLD at the floor and the same top token at `-ub 128`, `-ub 64` and `-ub 16`; the same text |
| bench, per batch | within 3% of the fastest of `split`, `rows` and `spacemit` at every row count; below all three at 64 and 96 rows |
| llama-bench t/s (each within 3%) | pp16 87.5, pp32 101.7, pp36 88.8, pp40 91.9, pp48 96.9, pp64 114.9, pp65 103.9, pp96 115.5, pp100 105.7, pp128 106.5, pp129 93.6 |
| pp2, pp4, pp8 | as `spacemit` (20.5, 41.4, 68.0) within 3% |
| pp512, tg128 | unchanged within 0.5% |

Rules (proposed 2026-10-21; the user fixes them before the run):
- Valid if the outputs are identical and the bench's controls stay within 3% (from 160 rows all four take path A).
- Adopt `fit` as the default if llama-bench finds every tested length no more than 1% slower than either `split` or
  `rows`, and the lengths where `split` runs path C (2-100 tokens) at least 5% faster than `split`.

How to run (on the K3, inside tmux, after a bundle of the commit):

```bash
cd /root/llama.cpp-flagos && git -c safe.directory=$PWD pull --ff-only ~/sx-fx5.bundle feature/flagos-spacemit
docs/flagos-k3/scripts/fusion-experiments.sh prepare 2>&1 | tee ~/fx-prepare5.log
~/tcmtest/tcmrelease --apply && build-flagos/bin/flagos-check-spacemit --expect-device --bench-mm > ~/fx-bench-fit.log 2>&1
grep -E "^bench    mm per" ~/fx-bench-fit.log
EXPS="mm fit" REPS=3 docs/flagos-k3/scripts/fusion-experiments.sh run ~/models/Qwen3-4B-Q4_0.gguf ~/fx/fit > ~/fx-fit.log 2>&1
```

- `prepare`: rc 0 expected; its `mm` line "30/30" on the Mac becomes "15/15" on the K3.
- The bench takes a few minutes more than before (a fourth setting); the run about 40 minutes (6 modes, 3
  repetitions, 9 logits runs). Progress from a second window: `grep -c 'rc=' ~/fx-fit.log` (66 runs in all).
- Then on the Mac: `scp -rq k3:/root/fx/fit ~/fx-fit && scp -q k3:/root/fx-fit.log k3:/root/fx-bench-fit.log ~/`, and
  `EXPS="mm fit" docs/flagos-k3/scripts/fusion-experiments.sh summary ~/fx-fit`.

**Bench results (K3, 2026-10-21, build `1d7e20d`, `--bench-mm`; per batch, the matmuls but the head, ms).** `prepare`:
rc 0, all checks passed (the paths check: 15/15 identical, paths as expected); in the lines per shape (q and down at
all 25 row counts) `fit` takes the path its rule gives in all 50. Valid by the bench's part of the rule: the controls
stay within 3% (160-512 rows, path A in all four settings: -0.8 to +0.1%). Elsewhere, settings that take the same paths
differ by -0.9 to +3.6% (`rows` against `split` below 32 rows, the +3.6% at 8 rows; `fit` against `spacemit` at 2-17
rows and against `rows` at 128-149). `split`, `rows` and `spacemit` reproduce the previous session (build `17413e6`)
within 1.4% from 16 rows up and within 5% at 2-8 rows.

| Rows | `split` | `rows` | `spacemit` | `fit` | `fit` against the fastest of the other three | against `split` |
|---|---|---|---|---|---|---|
| 2 | 93.8 | 93.2 | 80.8 | 80.1 | -0.9% (`spacemit`) | -15% |
| 4 | 91.9 | 91.6 | 78.8 | 78.7 | -0.1% (`spacemit`) | -14% |
| 8 | 118.6 | 122.9 | 97.1 | 96.9 | -0.2% (`spacemit`) | -18% |
| 16 | 201.9 | 201.3 | 153.3 | 152.8 | -0.3% (`spacemit`) | -24% |
| 17 | 237.1 | 235.0 | 172.4 | 171.8 | -0.3% (`spacemit`) | -28% |
| 24 | 298.3 | 298.1 | 246.0 | 233.2 | -5.2% (`spacemit`) | -22% |
| 31 | 447.1 | 447.3 | 358.2 | 334.9 | -6.5% (`spacemit`) | -25% |
| 32 | 399.9 | 263.2 | 318.4 | 262.5 | -0.3% (`rows`) | -34% |
| 33 | 470.7 | 441.1 | 348.0 | 325.2 | -6.6% (`spacemit`) | -31% |
| 36 | 490.0 | 496.8 | 354.7 | 346.3 | -2.4% (`spacemit`) | -29% |
| 40 | 752.8 | 499.5 | 389.3 | 368.0 | -5.5% (`spacemit`) | -51% |
| 44 | 780.3 | 504.0 | 424.6 | 390.5 | -8.0% (`spacemit`) | -50% |
| 48 | 802.1 | 513.2 | 459.2 | 412.6 | -10.1% (`spacemit`) | -49% |
| 64 | 748.8 | 526.8 | 591.4 | 496.5 | -5.8% (`rows`) | -34% |
| 65 | 1,285.0 | 697.3 | 618.9 | 558.0 | -9.8% (`spacemit`) | -57% |
| 96 | 1,600.9 | 791.6 | 925.6 | 735.1 | -7.1% (`rows`) | -54% |
| 100 | 1,798.7 | 1,025.9 | 965.6 | 840.7 | -12.9% (`spacemit`) | -53% |
| 112 | 1,830.0 | 1,037.0 | 1,070.4 | 895.4 | -13.7% (`rows`) | -51% |
| 113 | 1,172.9 | 1,038.3 | 924.2 | 908.4 | -1.7% (`spacemit`) | -23% |
| 128 | 1,249.9 | 1,050.4 | 1,372.0 | 1,050.1 | -0.0% (`rows`) | -16% |
| 129 | 1,350.0 | 1,223.6 | 1,487.2 | 1,234.9 | +0.9% (`rows`) | -9% |
| 149 | 1,492.4 | 1,297.8 | 1,639.9 | 1,292.0 | -0.4% (`rows`) | -13% |
| 160, 256, 512 | 1,315.9, 2,104.8, 4,227.6 | -0.0, -0.3, +0.1% | -0.0, -0.3, -0.8% | -0.1, -0.3, -0.8% | the same path A | |

- Against the predictions fixed before the run: within 3% of the fastest of the other three at every row count (holds:
  never more than 0.9% slower, at 129 rows, where `fit` and `rows` take the same paths); below all three at 64 and 96
  rows (holds: -5.8% and -7.1% against `rows`, estimated -6% and -8%).
- Where the shapes' fastest paths differ, `fit` beats every single-path setting: 5-14% below all three at 24, 31, 33,
  40-48, 65, 100 and 112 rows, 2% at 36 and 113. Where it takes the fastest setting's paths (2-17 rows: `spacemit`'s;
  128-149: `rows`'s) it is level with it within 1%, and at 32 rows path B for six shapes and path A for down add up to
  `rows`'s time (-0.3%). Down at 24 and 31 rows on path C instead of `spacemit`'s path B saves 12.8 and 23.3 ms per
  batch, against 13.8 and 22.1 ms from the previous session's lines per shape (36 down matmuls, 4 of them Q4_1).
- So per batch `fit` is 8.5-57% faster than `split` at 2-149 rows and never more than 0.9% slower than any other
  setting, inside the spread of identical paths. One line per shape trails: down at 36 rows on path A takes 3,370 us,
  path C 3,131 (+7.6%; at 33 rows path A is 2.3% faster).

**`EXPS="mm fit"` results (K3, 2026-10-21; build `1d7e20d`, Qwen3-4B Q4_0, 3 runs per mode, the modes interleaved;
all 66 runs rc 0).** Valid by the rules: the same greedy text in all three; at `-ub 128`, `-ub 64` and `-ub 16` mean
KLD 0.000000, maximum 0.000055 and the same top token 100% for both `rows` and `fit` (the floor of identical logits);
the same perplexity, 9.7573, at every batch size; the counters cost at most 1% but at pp2-pp8 (up to 2.3%, about
those tests' spread) and tg128 under `rows` (-1.0%). `split` reproduces the session before within 0.6% from pp8 up
(pp2 +3.9%, pp4 -1.3%), `rows` within 0.4% from pp16 up (pp2-pp8 within 1.3%).

| t/s, mean of 3 (pp128: 6, both tests) | `split` | `rows` | `fit` | `fit` against `split` | against `rows` | predicted | `spacemit` (the session before) | IME path (the odd-length run) |
|---|---|---|---|---|---|---|---|---|
| pp2 | 18.94 | 18.32 | 20.46 | +8.0% | +11.7% | 20.5 (-0.2%) | 20.45 | |
| pp4 | 36.84 | 36.57 | 39.04 | +6.0% | +6.8% | 41.4 (-5.7%) | 41.35 | |
| pp8 | 58.25 | 59.19 | 68.48 | +17.6% | +15.7% | 68.0 (+0.7%) | 68.02 | |
| pp16 | 69.92 | 69.57 | 87.05 | +24.5% | +25.1% | 87.5 (-0.5%) | 87.48 | 86.65 |
| pp32 | 71.41 | 101.73 | 101.56 | +42.2% | -0.2% | 101.7 (-0.1%) | 86.81 | 85.88 |
| pp33 | 63.48 | 67.21 | 86.62 | +36.5% | +28.9% | | 82.12 | 81.15 |
| pp36 | 66.07 | 64.90 | 88.45 | +33.9% | +36.3% | 88.8 (-0.4%) | 86.68 | 83.93 |
| pp37 | 64.31 | 66.27 | 87.25 | +35.7% | +31.7% | | 83.72 | 80.74 |
| pp40 | 49.35 | 70.79 | 91.77 | +86.0% | +29.6% | 91.9 (-0.1%) | 87.69 | 86.28 |
| pp44 | 51.94 | 76.02 | 94.50 | +81.9% | +24.3% | | 88.26 | 85.23 |
| pp48 | 54.74 | 80.61 | 96.74 | +76.7% | +20.0% | 96.9 (-0.2%) | 88.86 | 88.09 |
| pp64 | 79.68 | 108.40 | 114.32 | +43.5% | +5.5% | 114.9 (-0.5%) | 98.31 | 90.74 |
| pp65 | 48.66 | 84.76 | 103.80 | +113.3% | +22.5% | 103.9 (-0.1%) | 94.98 | 87.57 |
| pp96 | 57.23 | 107.96 | 115.04 | +101.0% | +6.6% | 115.5 (-0.4%) | 94.11 | 87.08 |
| pp100 | 53.29 | 88.74 | 105.88 | +98.7% | +19.3% | 105.7 (+0.2%) | 93.66 | 85.41 |
| pp128 | 92.08 | 106.40 | 106.33 | +15.5% | -0.1% | 106.5 (-0.2%) | 84.36 | 79.49 |
| pp129 | 85.98 | 93.29 | 93.37 | +8.6% | +0.1% | 93.6 (-0.2%) | 78.86 | 74.04 |
| pp512 | 95.07 | 94.85 | 94.89 | -0.2% | +0.0% | unchanged | 95.11 | |
| tg128 | 11.25 | 11.27 | 11.25 | +0.0% | -0.2% | unchanged | 11.22 | |

- Against the predictions fixed before the run: every one holds but pp4. Outputs identical; the bench's two (above);
  the eleven lengths from 16 to 129 tokens within 0.5% of their predictions (each allowed 3%); pp2 and pp8 within 0.7%
  of `spacemit`'s; pp512 and tg128 within 0.2%. pp4 is 39.04 against 41.4 (-5.7%): its three runs 37.6, 41.7 and
  37.9 (`fit-ops` 41.4, 39.4, 39.1; `spacemit` in the session before 41.3, 41.7, 41.1). At 4 rows `fit` takes
  `spacemit`'s paths and the bench finds the same matmul time (78.7 against 78.8 ms per batch), so the miss is most
  likely the spread of a single 4-token batch in this session, which `split` (35.7-37.6) and `rows` (35.8-37.2) show
  too (inference; not isolated).
- By the rule fixed before the run (proposed 2026-10-21, run unchanged): no tested length is more than 1% slower than
  `split` or `rows` (the closest: -0.2% against `rows` at pp32 and tg128, -0.2% against `split` at pp512), and every
  length from 2 to 100 tokens is at least 5% faster than `split` (the narrowest pp4, +6.0%, then pp2, +8.0%; pp8 to
  pp100 +17.6% to +113%). So `fit` is to become the default, on the evidence of Qwen3-4B (the last bullet).
- Against the other rules: from 16 to 129 tokens `fit` runs at 86.6-115.0 t/s, 0.5-34% faster than the IME path (the
  odd-length run, another session); level with `spacemit` at pp16, where both take path B for every shape (-0.5%),
  and 2-26% faster from 32 tokens (the session before). Level with `rows` at pp128 and pp129 (both on path A for every
  shape) and at pp32 (path B's gains on gate and up cancel its losses on q, k, v and the attention output): -0.2 to
  +0.1%; 5.5-36% faster at the other prompt lengths.
- What is left, from the counters of the `fit-ops` runs: the dips just past each multiple of 32 (pp33-37 86.6-88.5
  against pp32's 101.6; pp65 103.8 against 114.3; pp100 105.9 against pp96's 115.0; pp129 93.4 against pp128's 106.3)
  are path A's last round. Its leftover 4-row groups go to the first tiles (tile 0 is the slowest in every such step),
  so the matmuls' imbalance is 44-146 ms per batch at those lengths (8-14% of the batch) against 0-11 ms at the
  multiples of 32: down at 33-48 rows (30-50 ms), down and the attention output at 100 (70 ms), every shape at 129 (146
  ms). That is the ceiling of a balanced last round (the leftover groups split by columns over all tiles, the candidate
  above), an estimate. Down at 36 rows besides: path A 9.9 ms per batch slower than path C (about 2% of pp36).
- Not covered: other models. On Qwen3-0.6B (rows of 1,024, 2,048 and 3,072 values) `fit` takes path B up to 323 rows
  for q, k, v, gate and up, 153 for the attention output, 97 for down (Q4_1: 91), where on Qwen3-4B path B ends at 119
  rows. On 4B, against path A at 32, 64 and 96 rows (these counters), path B was 7-22% slower for k and v (1,024
  columns), between 6% slower and 10% faster for q and the attention output (4,096 and 2,560 columns), 4-14% faster for
  gate and up (9,728 columns); every matrix of 0.6B has 1,024-3,072 columns. Weighting 4B's ratios (interpolated by
  column count) with 0.6B's bytes puts `fit`'s matmuls on 0.6B about 13%, 4% and 1% slower than `rows`' at 32, 64 and
  96 rows (an estimate; 0.6B's shorter rows may move it either way); none from 113 rows, where `split` also takes path
  A (on 4B path B ran there only at 113 rows). The test: the same run on 0.6B, plus prompts of 160-320 tokens, where
  only `fit` takes path B.
- Proposed next (waits for the user), before the default changes: that test on 0.6B (no code change), judged by the
  same rule over all its lengths, 160-320 tokens included. If it holds, `fit` becomes the default (the unset setting;
  `split`, `rows` and `spacemit` stay for comparisons) with the acceptance run on both models; if not, the rule is
  revised from 0.6B's counters first.

**`EXPS="mm fit"` on Qwen3-0.6B (K3, 2026-10-22; build `1d7e20d`, Qwen3-0.6B Q4_0, 3 runs per mode, the modes
interleaved, all 66 runs rc 0; prompts of 160-320 tokens in a llama-bench run of their own, 3 runs per setting,
interleaved).** Valid: the same greedy text in all three; at `-ub 128`, `-ub 64` and `-ub 16` mean KLD -0.000001,
maximum 0.000057 and the same top token 100% for both `rows` and `fit`; perplexity 14.4890 at every batch size; pp512
and tg128 within 0.5% in all three. The counters cost at most 1% but at tg128 (-2.0 to -2.7%) and pp2-pp16 (up to 3.4%
either way, those tests' spread). `split` matches M2e.3's acceptance run at tg128 (55.77 against 55.8) and is 1.7%
below it at pp128 (655.6 against 667).

| t/s, mean of 3 (pp128: 6, both tests) | `split` | `rows` | `fit` | `fit` against `split` | against `rows` | the rule |
|---|---|---|---|---|---|---|
| pp2 | 100.22 | 102.00 | 102.91 | +2.7% | +0.9% | under 5% against `split` |
| pp4 | 184.53 | 187.69 | 193.62 | +4.9% | +3.2% | under 5% against `split` |
| pp8 | 306.48 | 299.88 | 320.31 | +4.5% | +6.8% | under 5% against `split` |
| pp16 | 407.48 | 407.58 | 437.05 | +7.3% | +7.2% | |
| pp32 | 466.36 | 550.66 | 510.85 | +9.5% | -7.2% | more than 1% slower than `rows` |
| pp33 | 275.96 | 407.59 | 481.29 | +74.4% | +18.1% | |
| pp36 | 294.57 | 397.70 | 503.68 | +71.0% | +26.6% | |
| pp37 | 295.77 | 405.36 | 489.46 | +65.5% | +20.7% | |
| pp40 | 313.68 | 425.70 | 507.98 | +61.9% | +19.3% | |
| pp44 | 329.73 | 449.29 | 508.25 | +54.1% | +13.1% | |
| pp48 | 300.31 | 472.15 | 512.81 | +70.8% | +8.6% | |
| pp64 | 560.61 | 657.06 | 629.74 | +12.3% | -4.2% | more than 1% slower than `rows` |
| pp65 | 280.84 | 541.20 | 605.83 | +115.7% | +11.9% | |
| pp96 | 337.73 | 652.44 | 649.08 | +92.2% | -0.5% | |
| pp100 | 309.71 | 566.40 | 630.93 | +103.7% | +11.4% | |
| pp128 | 655.59 | 653.61 | 655.60 | +0.0% | +0.3% | |
| pp129 | 586.83 | 586.78 | 627.49 | +6.9% | +6.9% | |
| pp160, pp192 (own run) | 649.33, 643.44 | 648.76, 642.92 | 658.71, 653.03 | +1.4%, +1.5% | +1.5%, +1.6% | |
| pp256, pp320 (own run) | 603.70, 591.67 | 600.16, 589.84 | 615.16, 602.51 | +1.9%, +1.8% | +2.5%, +2.1% | |
| pp512 | 516.36 | 514.58 | 517.23 | +0.2% | +0.5% | |
| tg128 | 55.77 | 55.57 | 55.71 | -0.1% | +0.3% | |

- By the rule fixed before the run (the 4B run's, unchanged): not met on Qwen3-0.6B, so `fit` does not become the
  default. The first part fails at pp32 and pp64 (7.2% and 4.2% slower than `rows`), the second at pp2, pp4 and pp8
  (2.7-4.9% faster than `split`, under 5%). Everywhere else `fit` is the fastest or level: 7-116% faster than `split`
  and 7-27% than `rows` at the lengths that are not multiples of 32, level at pp96 and pp128, 1.4-2.5% faster than
  both at 160-320 tokens, where only `fit` takes path B (for q, k, v, gate and up).
- Against the estimate fixed before the run (`fit`'s matmuls about 13%, 4% and 1% slower than `rows`' at 32, 64 and
  96 rows): the counters give +15.8%, +5.1% and +1.2% (the matmuls but the head, per batch).
- Why the matmuls lose: path B's pair turns. `gemm_slabs` keeps upstream's schedule (ime.cpp:446-490, read
  2026-10-22): the even tile of a pair copies its first slab and the activations while the odd one waits; then in
  every round one multiplies while the other copies; after the last wait the even tile is done and the odd one still
  multiplies its last slab. Read so, the odd tile ends one slab's multiply after the even one, about 1/R of the step,
  R being the slabs per tile (n / 256). The counters show it on both models: an odd tile is the slowest in all 10,089
  path-B steps of this run and all 9,834 of the 4B run (in 32-42% of the steps on paths A and C), and from 32 rows the
  odd tiles' mean time is above the even tiles' by 18-23% at R = 4, 10-12% at R = 8 and 7-8% at R = 12 here, and by
  19-23% (R = 4), 8.5-9% (10), 5-6% (16) and 2.3-2.5% (38) on 4B: about 0.9/R. On 4B only k and v have R = 4 (5% of
  the matmuls' time on path A at 64 rows), on 0.6B the attention output and down too (47%). The schedule read in the
  code fits these numbers; no test has yet run path B without the turns.
- Without that tail (the even tiles' times taken as each path-B step's; an estimate that assumes a pair's copies would
  not slow each other without the turns), path B would be level with path A at 32 rows (the batch +0.9% against
  `rows` on 0.6B, -2.7% on 4B) and 0-16% faster per shape at 64-128 rows on 0.6B, 8-16% at 64-96 on 4B; per batch on
  0.6B -3.8%, -6.3% and -6.0% against `rows` at 64, 96 and 128 rows and 5-6% below `split` at 2-8 rows.
- The steps that read path B's outputs are slower too on 0.6B: q's per-head norm by 31-53% at 64-128 rows, k's by
  42-44% at 64-96, SwiGLU by 4-7%, the V cache write by 3-9%; in all, the steps other than the matmuls take 1.4-1.8 ms
  more per batch at 64-129 rows (0.6-1.5% of a batch). On 4B they change by -0.6 to +0.4 ms per batch (of 0.3-1.4 s;
  k's norm +12-17% at 64-96 rows). Cause not isolated (path B writes every row in 32-column pieces from all tiles,
  path A whole rows from one tile).
- At 2-8 rows path B is barely faster than path C per shape on 0.6B (-6 to +3% at 2 rows, against 6-15% on 4B), which
  is the second part's miss.
- Proposed next (waits for the user): path B without the turns at several rows, as a test setting (each tile copies
  the activations, then copies and multiplies its own slabs one after the other, as gemv's `staged`; one condition in
  `gemm_slabs`), run like this one on both models and judged by the same rule. Predictions (from the estimate above,
  so upper bounds): the odd tiles' gap under 2% in every path-B step, and the even tiles' times within 2% of today's
  (more would mean the copies contend, and the turns would stay for small batches); on 0.6B against `rows` pp32 level
  (-0.9%, at the rule's edge), pp64 up to +4% and pp96 and pp128 up to +6-7%, against `split` pp2-pp8 up to +6-7% (the
  5% then holds only if the estimate does); on 4B up to +2-4% over today's `fit` at 2-100 tokens. Upstream's path B
  takes the same turns, so its several-row batches would have the same tail (a prediction; the IME path is not
  instrumented). Narrowing `fit`'s rule instead (path B only where R >= 8) would leave pp32 on 0.6B about 2.4% slower
  than `rows` (q, gate and up on path B are 6-12% slower than on path A there) and would be fitted to two models.

**Path B without the turns as a fifth setting (prepared 2026-10-22; the user's decision on the proposal above).**
`FLAGOS_SPACEMIT_MM=fit,staged` (test-only until adopted) takes `fit`'s paths, but on path B every tile copies the
activations and its first slab, then multiplies each slab and copies its next, without the pair waits: the schedule a
tile without a partner already had (`gemm_slabs`: `partner` false for every tile, one condition). The same bytes as the
other four. `spacemit_mm_paths_take` also counts the path-B steps whose tile 0 ran without the turns, so the check sees
which setting dropped them.

Checks (`check_mm_paths`, five settings, the same 15 cases): `fit,staged` must take `fit`'s paths, path B without the
turns, and only it may drop them. On the Mac, 2026-10-22: 30/30 identical with the expected paths and turns, tiles in
turn and on threads; four mutations caught (the turns kept whatever the setting: 24 of 30 runs fail; without the turns
the odd tiles skip their first slab's copy: 23; their last round: 24; `fit,staged` without `fit`'s rule: 26);
`test-backend-ops -o MUL_MAT` 3/3 under `fit,staged`, in turn and on threads; ThreadSanitizer over the whole check
program: 0 reports; the RVV syntax check clean.

Measurement: `--bench-mm` with a fifth column; `EXPS="mm fit fit-staged"` on both models: modes `ref`, `ops`, `mm`,
`mm-ops`, `fit`, `fit-ops`, `fit-staged`, `fit-staged-ops`; pp128 with tg128 (`-ub 128`), pp512, the prompts of 2-129
tokens and of 160-320 (where on Qwen3-0.6B only `fit` and `fit,staged` take path B; on Qwen3-4B path A everywhere);
the logits at batches of 128, 64 and 16 rows. The summary adds `fit-staged-ops` against `fit-ops` and, for both,
every matmul's even and odd tiles (`op-stats.py pairs`: their mean times, the odd ones' excess, how often an odd tile
was the slowest).

Predictions (estimates fixed before the run; the llama-bench ones are upper bounds, from the even tiles' times in the
last runs' counters, which assume that the two tiles of a pair do not slow each other's copies without the turns):

| What | Prediction |
|---|---|
| outputs | identical: KLD at the floor and the same top token at `-ub 128`, `-ub 64` and `-ub 16`; the same text |
| the turns' tail (`op-stats.py pairs`) | in `fit-staged-ops`' path-B steps from 32 rows the odd tiles' excess within ±2% (`fit-ops`: 18-23% at R = 4 down to 2.3-2.5% at R = 38) and an odd tile the slowest in 30-70% of the steps (`fit-ops`: 100%) |
| the copies (`fit-staged-ops` against `fit-ops`) | from 32 rows every tile within 2% of `fit`'s even tiles in every path-B step; below 32 rows, where the copies are a larger part of a step, no prediction |
| bench per batch (4B's shapes) | `fit,staged` 1.5-4% below `fit` at 2-113 rows; per shape from 32 rows k and v 15-20% below `fit`, the attention output 7-9%, q 5-6%, gate and up 2-3%; from 128 rows the same as `fit` (path A) |
| llama-bench, Qwen3-0.6B, against `fit` | at most +3-4% at pp2-pp8, +6% at pp16, +7% at pp32-pp48, +9% at pp64-pp96, +6% at pp100-pp129, +4% at pp160-pp320 |
| the same against the rule | at most pp2-pp8 +6-7% over `split` and pp16 +13%; pp32 -1% against `rows` (at the rule's edge), pp64 +4%, pp96 +7%, pp128 +6% |
| llama-bench, Qwen3-4B, against `fit` | at most +1.5-2.5% at pp2-pp8, +4% at pp16, +2-3% at pp32-pp100; pp128, pp129 and pp160-pp320 the same (path A) |
| pp512, tg128 | unchanged within 0.5% on both models |

Rules (proposed 2026-10-22; the user fixes them before the run):
- Valid if the outputs are identical and the bench's controls stay within 3% (160-512 rows, path A in all five).
- The tail counts as removed if the odd tiles' excess is within ±2% in `fit-staged-ops`' path-B steps from 32 rows;
  otherwise the run says nothing about the turns.
- Adopt `fit,staged` as the default (the unset setting; `split`, `rows`, `spacemit` and `fit` stay for comparisons),
  after the acceptance run on both models, if llama-bench finds on both models every tested length no more than 1%
  slower than `split` or `rows` and the lengths where `split` runs path C (2-100 tokens) at least 5% faster than
  `split` (the rule `fit` was judged by). Where `fit,staged` is more than 1% slower than `fit`, the copies contend, and
  the turns would stay for those batches (designed from the counters after the run).

How to run (on the K3, inside tmux, after a bundle of the commit):

```bash
cd /root/llama.cpp-flagos && git -c safe.directory=$PWD pull --ff-only ~/sx-fx6.bundle feature/flagos-spacemit
docs/flagos-k3/scripts/fusion-experiments.sh prepare 2>&1 | tee ~/fx-prepare6.log
~/tcmtest/tcmrelease --apply && build-flagos/bin/flagos-check-spacemit --expect-device --bench-mm > ~/fx-bench-fs.log 2>&1
grep -E "^bench    mm per" ~/fx-bench-fs.log
EXPS="mm fit fit-staged" REPS=5 docs/flagos-k3/scripts/fusion-experiments.sh run ~/models/Qwen3-0.6B-Q4_0.gguf ~/fx/fs06 > ~/fx-fs06.log 2>&1
EXPS="mm fit fit-staged" REPS=3 docs/flagos-k3/scripts/fusion-experiments.sh run ~/models/Qwen3-4B-Q4_0.gguf ~/fx/fs4b > ~/fx-fs4b.log 2>&1
```

- `prepare`: rc 0 expected; its paths check "15/15" on the K3.
- Five repetitions on 0.6B: its pp2-pp8 varied by up to 6% between runs in the last session, around the rule's 5%.
  176 runs on 0.6B, 112 on 4B; progress from a second window: `grep -c 'rc=' ~/fx-fs06.log`.
- Then on the Mac: `scp -rq k3:/root/fx/fs06 ~/fx-fs06 && scp -rq k3:/root/fx/fs4b ~/fx-fs4b`, `scp -q
  k3:/root/fx-fs06.log k3:/root/fx-fs4b.log k3:/root/fx-bench-fs.log ~/`.

**Bench results (K3, 2026-10-22, build `696549b`, `--bench-mm`; per batch at Qwen3-4B's shapes, the matmuls but the
head, ms).** Valid by the bench's part of the rule: the controls stay within 3% (160-512 rows, path A in all five:
`fit,staged` +0.3, -0.1 and +0.6% against `split`), and at 128-149 rows, where `rows`, `fit` and `fit,staged` all take
path A, the three agree within 0.1%. `split`, `rows`, `spacemit` and `fit` reproduce the session of `1d7e20d` within
1.6%, but `rows` at 8 rows (-5.0%; that session's was itself 3.6% off `split`).

| Rows | `split` | `fit` | `fit,staged` | against `fit` | against `split` (`fit`'s) |
|---|---|---|---|---|---|
| 2 | 94.0 | 79.5 | 90.2 | +13.5% | -4.0% (-15.4%) |
| 4 | 92.6 | 78.3 | 86.9 | +11.0% | -6.2% (-15.4%) |
| 8 | 117.4 | 97.1 | 105.0 | +8.1% | -10.6% (-17.3%) |
| 16 | 202.1 | 152.8 | 151.2 | -1.0% | -25.2% (-24.4%) |
| 17, 24, 31 | 236.1, 297.8, 446.5 | 172.1, 233.6, 334.6 | 167.1, 227.5, 322.6 | -2.9, -2.6, -3.6% | |
| 32-48 (6 counts) | | | | -3.1 to -4.0% | |
| 64-113 (6 counts) | | | | -2.6 to -4.2% | |
| 128-512 (6 counts) | | | | -0.2 to +0.9% (path A) | |

- Against the predictions fixed before the run: from 16 to 113 rows `fit,staged` is 1.0-4.2% below `fit` (predicted
  1.5-4%; 16 rows -1.0% and 64 rows -4.2% just outside), and from 128 rows level with it (predicted). At 2-8 rows the
  prediction fails: `fit,staged` is 8-14% slower than `fit`, and only 4-11% below `split` (`fit` 15-17%). So on 4B's
  shapes the turns pay at the smallest batches and cost from 16 rows. Most likely the copies: with few rows a step is
  mostly copies, and without the turns both tiles of a pair copy at once (inference; it matches the 1-row path, where
  the turns are worth 5% of a token; not isolated, and a pair's shared port is not told apart from the DRAM bandwidth).
- For the llama-bench rule (an inference from these numbers; the runs will settle it): at pp2 and pp4 on 4B
  `fit,staged`'s matmuls lead `split`'s by only 4-6%, so the rule's second part (5% faster than `split` in t/s) is at
  risk there. A hybrid, the turns below a row count and none from it (16 on these shapes), would keep `fit`'s times at
  2-8 rows and `fit,staged`'s from 16; its threshold to be read from the bench's lines per shape and the runs'
  counters, after the runs.
- The paths check in `prepare` on the K3: 15/15 identical, paths and turns as expected (spine-runtime).
- Per shape (`fit,staged` against `fit` where `fit` takes path B): at 2 and 4 rows the turns pay for every shape (6-25%
  slower without them), at 8 rows for all but k and v and Q4_1's down (+0.9% and +0.8%, level), at 16 rows only for
  Q4_0's down (K 9728: +3.3%; +0.2% at 17 rows), from 17 rows for none. From 32 rows: q 4.7-6.3% faster (predicted
  5-6%), k and v 11.2-16.3% (predicted 15-20%; below it up to 64 rows), the attention output 5.1-7.8% (predicted 7-9%;
  below it up to 48 rows), gate and up 2.8-4.9% (predicted 2-3%; above it). Whether the shortfall on the narrow shapes
  is the copies contending and the excess on gate and up the waits' own cost (39 per call) is not isolated; the runs'
  counters (`fit-staged-ops` against `fit-ops`, `op-stats.py pairs`) separate them.
- The hybrid with the threshold at 16 rows, estimated from these lines per shape (weighted as the batch): within 0.7%
  of the faster of the two per shape at every row count measured (at 16 rows Q4_0's down keeps 0.7% of the batch),
  so 15.4-17.3% below `split` at 2-8 rows and 1.1-4.2% below `fit` at 16-113 rows. Qwen3-0.6B's shapes are not in the
  bench; its crossover will show in the runs' counters at 2-16 rows.

**`EXPS="mm fit fit-staged"` on Qwen3-0.6B (K3, 2026-10-22; build `696549b`, Qwen3-0.6B Q4_0, 5 runs per mode, the
modes interleaved, all 176 runs rc 0).** Valid: the same greedy text in all four; at `-ub 128`, `-ub 64` and `-ub 16`
mean KLD -0.000001, maximum 0.000057 and the same top token 100% for `rows`, `fit` and `fit,staged`; perplexity
14.4890 at every batch size; pp512 and tg128 within 0.3% in all four. The counters cost at most 1% but at tg128 (-2.1
to -2.5%) and pp2-pp16 (up to 2.5%). `split`, `rows` and `fit` reproduce the session before within 1.2% from pp8 up
(pp2 and pp4 up to 4% higher).

| t/s, mean of 5 (pp128: 10, both tests) | `split` | `rows` | `fit` | `fit,staged` | against `split` | against `rows` | against `fit` | predicted against `fit` (at most) |
|---|---|---|---|---|---|---|---|---|
| pp2 | 102.58 | 102.23 | 103.11 | 98.01 | -4.5% | -4.1% | -4.9% | +3-4% |
| pp4 | 192.25 | 192.54 | 196.21 | 191.64 | -0.3% | -0.5% | -2.3% | +3-4% |
| pp8 | 308.15 | 311.16 | 317.71 | 317.92 | +3.2% | +2.2% | +0.1% | +3-4% |
| pp16 | 406.52 | 406.02 | 433.74 | 456.02 | +12.2% | +12.3% | +5.1% | +6% |
| pp32 | 464.66 | 547.09 | 506.42 | 544.16 | +17.1% | -0.5% | +7.5% | +7% |
| pp33-pp48 (6 lengths) | | | | 515.2-546.3 | +65-87% | +17-35% | +6.8-7.8% | +7% |
| pp64 | 560.94 | 657.84 | 629.23 | 684.27 | +22.0% | +4.0% | +8.7% | +9% |
| pp65 | 281.45 | 537.43 | 601.78 | 657.33 | +133.6% | +22.3% | +9.2% | +9% |
| pp96 | 339.12 | 660.31 | 645.95 | 703.71 | +107.5% | +6.6% | +8.9% | +9% |
| pp100 | 309.58 | 565.85 | 627.62 | 665.48 | +115.0% | +17.6% | +6.0% | +6% |
| pp128 | 653.34 | 651.24 | 652.83 | 695.23 | +6.4% | +6.8% | +6.5% | +6% |
| pp129 | 588.72 | 587.49 | 626.81 | 667.42 | +13.4% | +13.6% | +6.5% | +6% |
| pp160-pp320 (4 lengths) | | | | 623.8-685.5 | +6.0-6.3% | +5.8-6.7% | +4.1-4.6% | +4% |
| pp512 | 514.11 | 513.96 | 514.38 | 514.22 | +0.0% | +0.1% | -0.0% | unchanged |
| tg128 | 55.46 | 55.58 | 55.45 | 55.56 | +0.2% | -0.0% | +0.2% | unchanged |

- Against the predictions fixed before the run: from 16 tokens every length within a point of its bound (pp16 +5.1%
  against at most +6%, pp32-pp48 +6.8-7.8% against +7%, pp64-pp96 +8.7-9.2% against +9%, pp100-pp129 +6.0-6.5%
  against +6%, pp160-pp320 +4.1-4.6% against +4%); some lie slightly above their bound because some tiles ran faster
  than `fit`'s even tiles (gate and up at 32 rows by 3.3-3.8%: the waits' own cost, inference). At pp2-pp8 the
  prediction fails as in the bench: -4.9%, -2.3% and +0.1% against `fit`. pp512 and tg128 unchanged (predicted).
- The mechanism (`op-stats.py pairs` and the counters per tile, path-B steps by `fit`'s rule): from 32 rows the odd
  tiles' excess fell from 7-24% under `fit` to 0.1-3.6%; within the rule's ±2% from 64 rows, not at 32 rows, where k,
  v, the attention output and down keep 2.1-3.6%. An odd tile is still the slowest in most of these steps (up to
  100%), by little; the even and odd tiles do their two first copies in opposite orders, one candidate (not isolated).
  From 32 rows every tile is within -3.8 to +3.0% of `fit`'s even tiles (the copies do not contend there); at 2-8 rows
  2.5-34% above them (they do; most for k, v, the attention output and down at 2-4 rows). Per shape the crossover
  lies between 8 and 16 rows, as on 4B's shapes in the bench: at 8 rows q, k, gate and up already run faster without
  the turns (0.2-3.0%), v, the attention output and down still slower (1.5-8.4%); at 16 rows all are 6-13% faster.
- By the rule fixed before the run: not met on Qwen3-0.6B. Its first part fails at pp2 (4.5% slower than `split`,
  4.1% than `rows`), its second at pp2-pp8 (-4.5%, -0.3% and +3.2% against `split`). From pp16 `fit,staged` is the
  fastest of the four or level with `rows` (pp32, -0.5%): 4-134% faster than `split` and `rows` at the other lengths,
  pp128 695 t/s (`split` 653; M2e.3's acceptance run 667).
- The hybrid (the turns below 16 rows, none from 16) runs `fit`'s schedule at pp2-pp8 and `fit,staged`'s from pp16, so
  these runs give its t/s on 0.6B (an estimate: the two schedules measured in one session, not the hybrid itself):
  0.5-3.1% over `split` and 0.9-2.1% over `rows` at pp2-pp8, `fit,staged`'s from pp16. The rule's first part would
  hold at every length tested, its second not at pp2-pp8: on 0.6B no setting reaches 5% over `split` there, as path B
  is barely faster than path C at 2-8 rows on its small matrices (the run before).

**`EXPS="mm fit fit-staged"` on Qwen3-4B (K3, 2026-10-22; build `696549b`, Qwen3-4B Q4_0, 3 runs per mode, the modes
interleaved, all 112 runs rc 0).** Valid: the same greedy text in all four; at `-ub 128`, `-ub 64` and `-ub 16` mean
KLD 0.000000, maximum 0.000055 and the same top token 100% for `rows`, `fit` and `fit,staged`; perplexity 9.7573 at
every batch size; pp512 and tg128 within 0.3% in all four. The counters cost at most 1% but at pp2-pp8 (up to 2.8%,
those tests' spread). `split`, `rows` and `fit` reproduce the session of `1d7e20d` within 1.1% from pp8 up (pp2 and pp4
within 5%).

| t/s, mean of 3 (pp128: 6, both tests) | `split` | `rows` | `fit` | `fit,staged` | against `split` | against `rows` | against `fit` | predicted against `fit` (at most) |
|---|---|---|---|---|---|---|---|---|
| pp2 | 18.53 | 18.86 | 20.84 | 19.18 | +3.5% | +1.7% | -8.0% | +1.5-2.5% |
| pp4 | 36.88 | 36.76 | 40.85 | 38.39 | +4.1% | +4.4% | -6.0% | +1.5-2.5% |
| pp8 | 58.86 | 59.18 | 68.25 | 64.34 | +9.3% | +8.7% | -5.7% | +1.5-2.5% |
| pp16 | 69.76 | 69.69 | 87.42 | 90.16 | +29.2% | +29.4% | +3.1% | +4% |
| pp32 | 71.36 | 101.51 | 101.39 | 105.32 | +47.6% | +3.8% | +3.9% | +2-3% |
| pp33-pp48 (6 lengths) | | | | 89.5-99.7 | +38-92% | +24-40% | +2.6-3.1% | +2-3% |
| pp64 | 79.63 | 108.51 | 114.11 | 118.39 | +48.7% | +9.1% | +3.8% | +2-3% |
| pp65 | 48.65 | 84.87 | 103.81 | 107.40 | +120.8% | +26.5% | +3.5% | +2-3% |
| pp96 | 57.24 | 107.92 | 115.04 | 118.17 | +106.4% | +9.5% | +2.7% | +2-3% |
| pp100 | 53.25 | 88.73 | 105.88 | 108.35 | +103.5% | +22.1% | +2.3% | +2-3% |
| pp128 | 92.16 | 106.40 | 106.34 | 106.34 | +15.4% | -0.1% | +0.0% | the same |
| pp129 | 85.94 | 93.61 | 93.40 | 93.41 | +8.7% | -0.2% | +0.0% | the same |
| pp160-pp320 (4 lengths) | | | | 101.6-106.0 | -0.3 to -0.1% | -0.1 to +0.0% | -0.1 to +0.1% | the same |
| pp512 | 95.01 | 94.88 | 94.91 | 95.18 | +0.2% | +0.3% | +0.3% | unchanged |
| tg128 | 11.26 | 11.24 | 11.24 | 11.26 | +0.0% | +0.2% | +0.2% | unchanged |

- Against the predictions fixed before the run: from 16 tokens within a point of the bounds (pp16 +3.1% against +4%,
  pp32-pp100 +2.3-3.9% against +2-3%; pp32, pp64 and pp65 slightly above), from 128 tokens the same as `fit` (path A
  everywhere), pp512 and tg128 unchanged. At pp2-pp8 the prediction fails as in the bench: 5.7-8.0% slower than `fit`.
- The mechanism, as on 0.6B: from 32 rows the odd tiles' excess fell from 2.3-22.6% under `fit` to 0.2-3.8%, within
  ±2% from 96 rows, not at 32-64 rows for k, v and the attention output (1.8-3.8%); every tile within -3.0 to +2.2%
  of `fit`'s even tiles from 32 rows, 2.5-31.5% above them at 2-8 rows. Q4_1's down keeps 4.3-7.0% at 2-16 rows with
  or without the turns (an odd tile the slowest in 92-100% of its steps): an odd-tile disadvantage the turns do not
  explain, cause not isolated. So the rule's ±2% criterion is not met at 32-64 rows on either model; the turns' tail
  accounts for most of the odd excess (it fell by 70-99% from 32 rows), and a residual of a few percent remains.
- By the rule fixed before the run: not met on Qwen3-4B either. Its first part holds at every length (the closest:
  -0.3% against `split` at pp192, -0.2% against `rows` at pp129), its second fails at pp2 and pp4 (+3.5% and +4.1%
  over `split`; `fit` had +12.5% and +10.8% in the same session), as the bench foretold.
- The hybrid (the turns below 16 rows), from these runs as on 0.6B (`fit`'s schedule at pp2-pp8, `fit,staged`'s from
  pp16; an estimate): at least 10.8% over `split` at every length from 2 to 100 tokens, never more than 0.3% slower
  than `split` or `rows`, 2.3-3.9% over `fit` from pp32 to pp100. So on 4B both parts of the rule would hold; on 0.6B
  the first part would, the second not at pp2-pp8 (+0.5-3.1% over `split`).

**The threshold as a sixth setting (prepared 2026-10-22; user decision).** `FLAGOS_SPACEMIT_MM=fit,mixed` (test-only
until adopted) takes `fit`'s paths, and on path B keeps the pair turns while the batch's quantized activations take
fewer bytes than one 32-column slab, none from there (`slabs_paired`): 16 rows for Q4_0 and Q4_1, 29 for Q8_0 and Q6_K
(run as Q8_0). The row length cancels, so the switch depends on the batch's rows and the weight's layout, not on the
model; one condition in the code, the same bytes as the other five. Why this rule: on both models the crossover lies
between 8 and 16 rows for rows of 1,024-9,728 values and 4-38 slabs per tile; a slab's copy and its multiply both
grow with the row length, so their ratio depends on the rows and on the layout's bytes and kernel; the byte rule gives
16 for both Q4 layouts (measured) and 29 for Q8_0 (a prediction). A threshold a few rows off costs little, as the two
schedules meet at the crossover; their large gaps lie far from it.

Checks (`check_mm_paths`: six settings, 21 cases, six of them new, on both sides of each threshold, among them Q4_1
at 16 rows, where the activations take exactly a slab's bytes). On the Mac, 2026-10-22: 42/42 identical with the
expected paths and turns, tiles in turn and on threads; five mutations caught (`<=` for `<`: 2 of 42 runs fail, both
at that case; the threshold inverted: 36; always the turns: 22; never: 14; `fit,mixed` without `fit`'s paths: 38);
`test-backend-ops -o MUL_MAT` 3/3 under `fit,mixed`, in turn and on threads; ThreadSanitizer over the whole check
program: 0 reports; the RVV syntax check clean.

Measurement:
1. `flagos-check-spacemit --bench-grid` (new; on no model): Q4_0, Q4_1 and Q8_0 weights of 1,024, 4,096 and 9,728
   values per row and 256, 512, 1,024, 3,072 and 9,728 columns (1, 2, 4, 12 and 38 slabs per tile) at 2-48 rows,
   under `fit`, `fit,staged` and `fit,mixed`. Each graph chains up to 64 copies of the weight (32 MB together, one
   launch), the three run in turn for three rounds and each keeps its fastest. Per shape: from which row count
   `fit,staged` stays faster than `fit` on path B, and `fit,mixed`'s worst and mean excess over the faster of the
   two; per layout a summary; the rows where none takes path B are the grid's controls.
2. `--bench-mm` with a sixth column (Qwen3-4B's shapes, as before).
3. `EXPS="mm fit-mixed"` on Qwen3-0.6B (5 runs per mode) and Qwen3-4B (3), both Q4_0: the rule; `fit` and
   `fit,staged` from the session before. The logits also at batches of 8 rows (the turns' side).
4. `EXPS="mm fit fit-staged fit-mixed"` on Qwen3-0.6B in Q8_0 (5 runs per mode): the Q8_0 threshold (29 rows) end to
   end, all three schedules in one session, as no Q8_0 run came before. Qwen's own Q8_0 file if it can be had,
   otherwise the Q4_0 model requantized to Q8_0 (`llama-quantize --allow-requantize`: Q4_0's values rounded to Q8_0,
   the same kernels and bytes per row as a true Q8_0, which is all this run measures).

Qwen3 only (user decision, 2026-10-22): no model of another family for now. Whether the switch picks the faster
schedule on shapes the rule was not derived from rests on the grid (all three schedules in one session, three layouts,
1-38 slabs per tile) and the Q8_0 run.

Predictions (estimates fixed before the run):

| What | Prediction |
|---|---|
| outputs | identical in every run: KLD at the floor and the same top token at `-ub 128`, 64, 16 and 8; the same text |
| the K3's checks | 21/21 identical, paths and turns as expected |
| grid, Q4_0 and Q4_1, 4 or more slabs per tile | `fit,staged` faster from 10-16 rows at rows of 1,024 and 4,096 values, from 16 or never at 9,728 (path B ends at 19 rows there); `fit,mixed` within 4% of the faster schedule at every row count, its worst at 16 rows of 9,728 values |
| grid, Q8_0 | the crossover later, at 16-32 rows (the byte rule's 29); `fit,mixed` within 4% at 4 or more slabs per tile |
| grid, 1-2 slabs per tile | not measured before: the crossover earlier (the turns' tail is half or all of a tile's multiplies), possibly from the fewest rows; `fit,mixed` may lose there up to the gap between the schedules at 8-15 rows |
| grid's controls | within 2% (the same code in all three) |
| `--bench-mm`, per batch | `fit,mixed` within 1% of `fit` at 2-8 rows and of `fit,staged` from 16 |
| llama-bench, Qwen3-4B | pp2-pp8 at `fit`'s values of the session before (20.8, 40.9, 68.3 t/s; those lengths drift 2-5% between sessions), from pp16 at `fit,staged`'s within 1%: at least 10% over `split` at 2-100 tokens; from pp128 as `split` and `rows` (path A) |
| llama-bench, Qwen3-0.6B | pp2-pp8 at `fit`'s (0.5-3.1% over `split`), from pp16 at `fit,staged`'s (12-134% over `split`, pp32 within 1% of `rows`) |
| llama-bench, Qwen3-0.6B Q8_0 | `fit` and `fit,staged` cross between 16 and 32 tokens (the byte rule's 29); `fit,mixed` within 1% of the faster of the two at every length, but possibly pp16 (the turns there; up to the gap if Q8_0 crosses below 16 rows) |
| pp512, tg128 | unchanged within 0.5% in every run |

Rules (proposed 2026-10-22; the user fixes them before the run):
- Valid if the outputs are identical in every run, the K3's checks pass, and the benches' controls stay within 3%.
- Adopt `fit,mixed` as the default (the unset setting; `split`, `rows`, `spacemit`, `fit` and `fit,staged` stay for
  comparisons), after the acceptance runs on both Qwen3 models, if in every run (the two Q4_0 models and the Q8_0 one)
  (1) no tested length is more than 1% slower than `split` or `rows`, and (2) every length from 16 to 100 tokens is at
  least 5% faster than `split` (part 2 from 16 tokens: confirmed by the user on 2026-10-22 as a trial, to go back to 2
  tokens if needed; from 2 tokens no setting passes on 0.6B); and if the grid finds `fit,mixed` within 4% of the
  faster schedule at every row count with 4 or more slabs per tile. In the Q8_0 run `fit,mixed` is also held to the
  faster of `fit` and `fit,staged`: more than 1% behind it at some length means the Q8_0 threshold is off, and it is
  refined from that run's counters (without blocking adoption if (1) and (2) hold). A miss at 1-2 slabs per tile in
  the grid is reported and the rule refined (the switch could also take the slabs per tile), without blocking
  adoption: no Qwen3 matrix has so few (the narrowest, 1,024 columns, have 4 slabs per tile).

How to run (on the K3, inside tmux, after a bundle of the commit):

```bash
cd /root/llama.cpp-flagos && git -c safe.directory=$PWD pull --ff-only ~/sx-fx7.bundle feature/flagos-spacemit
docs/flagos-k3/scripts/fusion-experiments.sh prepare 2>&1 | tee ~/fx-prepare7.log
~/tcmtest/tcmrelease --apply && build-flagos/bin/flagos-check-spacemit --expect-device --bench-mm --bench-grid > ~/fx-bench-mixed.log 2>&1
grep -E "^bench    (mm per|grid [^ ]+:)" ~/fx-bench-mixed.log
EXPS="mm fit-mixed" REPS=5 docs/flagos-k3/scripts/fusion-experiments.sh run ~/models/Qwen3-0.6B-Q4_0.gguf ~/fx/mx06 > ~/fx-mx06.log 2>&1
EXPS="mm fit-mixed" REPS=3 docs/flagos-k3/scripts/fusion-experiments.sh run ~/models/Qwen3-4B-Q4_0.gguf ~/fx/mx4b > ~/fx-mx4b.log 2>&1
# Qwen3-0.6B in Q8_0: Qwen's file copied to ~/models if it can be had, otherwise requantized here from the Q4_0
[ -f ~/models/Qwen3-0.6B-Q8_0.gguf ] || { cmake --build build-flagos --target llama-quantize > ~/fx-quantize.log 2>&1 && build-flagos/bin/llama-quantize --allow-requantize ~/models/Qwen3-0.6B-Q4_0.gguf ~/models/Qwen3-0.6B-Q8_0.gguf Q8_0 >> ~/fx-quantize.log 2>&1; }
EXPS="mm fit fit-staged fit-mixed" REPS=5 docs/flagos-k3/scripts/fusion-experiments.sh run ~/models/Qwen3-0.6B-Q8_0.gguf ~/fx/mx06q8 > ~/fx-mx06q8.log 2>&1
```

- `prepare`: rc 0 expected; its paths check "21/21 identical, paths and turns as expected".
- The Q8_0 file: unsloth's on ModelScope, the source of the Q4_0 models (build.md §7), 639,447,744 bytes; on the
  board `cd ~/models && wget -c -q --show-progress
  https://modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF/resolve/master/Qwen3-0.6B-Q8_0.gguf`, or with `curl -L -C - -o`
  on the Mac and `scp` to `k3:/root/models/`. Hugging Face did not connect from the Mac (2026-10-22).
- Runs: 135 on 0.6B, 87 on 4B, 225 on 0.6B in Q8_0; progress: `grep -c 'rc=' <log>`; a run is finished when its
  log's last line is `logs: <out>`.
- Then on the Mac: `scp -q k3:/root/fx-bench-mixed.log k3:/root/fx-prepare7.log ~/` and, per run,
  `scp -rq k3:/root/fx/<out> ~/fx-<out> && scp -q k3:/root/fx-<out>.log ~/`.

**The grid on the K3 (2026-10-22, build `cdfd29e`; `~/fx-bench-mixed.log`).** `prepare`: rc 0, its paths check 21/21
identical with the expected paths and turns; the Q8_0 file in place (639,447,744 bytes). One run of `--bench-mm
--bench-grid`:
- `--bench-mm` (Qwen3-4B's matmuls but the head): `fit,mixed` within 1.0% of `fit` at 2-8 rows (-1.0, +0.5 and
  +0.1%), within 0.4% of `fit,staged` from 16 to 256 rows but at 17 (-1.4%), +1.0% at 512 rows, where all six run
  path A. The five other settings within 1.8% of the last session's bench at every row count, but `spacemit` at 512
  rows (-2.7%). As predicted: the switch runs the schedule it should.
- The grid's controls (rows off path B, the same code in all three) within 1.0% for Q4_0 and 1.1% for Q4_1; for Q8_0
  3.9% at one row (9,728 x 3,072 at 8 rows, path C) and 3.0% at two, all at 9,728 values per row, Q8_0's only rows
  off path B. By the validity rule (3%) the Q8_0 grid is not valid, the Q4 grids are. On path B, `fit,mixed` and the
  setting that runs its schedule differ by 0.2% (median), 1.1% (95th percentile), 2.5% at most.
- The row count from which `fit,staged` stays faster than `fit` (the crossover), by layout, row length and columns
  (slabs per tile in brackets):

| | 256 (1) | 512 (2) | 1,024 (4) | 3,072 (12) | 9,728 (38) |
|---|---|---|---|---|---|
| Q4_0, rows of 1,024 values | 2 | 2 | 10 | 10 | 2 |
| Q4_0, 4,096 | 6 | 10 | 14 | 14 | 10 |
| Q4_0, 9,728 (path B to 16 of the grid's rows) | 6 | 10 | 14 | never | 14 |
| Q4_1, 1,024 | 2 | 2 | 6 | 10 | 2 |
| Q4_1, 4,096 | 2 | 6 | 10 | 10 | 6 |
| Q4_1, 9,728 (to 16 rows) | 2 | 6 | 10 | 10 | 16 |
| Q8_0, 1,024 | 2 | 10 | 14 | 14 | 10 |
| Q8_0, 4,096 | 6 | 14 | 24 | 28 | 20 |
| Q8_0, 9,728 (to 4 rows; the turns faster at 2 and 4) | never | never | never | never | never |

  Against the predictions: at 4 or more slabs per tile 10 of the 18 Q4 crossovers lie in the predicted range (10-16
  rows; 16 or never at 9,728 values), the other 8 earlier, down to 2 rows at 1,024 x 9,728 in both layouts; Q8_0's at
  4,096 values in the predicted 16-32 (20-28), at 1,024 values earlier (10-14). At 1-2 slabs per tile earlier than at
  4, as predicted. `fit,mixed` against the faster of the two at 4 or more slabs per tile: worst +6.7% (Q4_0, 1,024 x
  1,024 at 14 rows), +10.6% (Q4_1, 9,728 x 1,024 at 14 rows) and +10.2% (Q8_0, 1,024 x 1,024 at 28 rows), each at the
  last row count before its threshold (predicted: within 4%); at 1-2 slabs per tile +35.0%, +40.8% and +46.2%. On
  average at 4 or more slabs per tile (from the two schedules' times): `fit,mixed` +1.0% over the faster, `fit` +3.9%,
  `fit,staged` +3.7%.
- Path B at 6, 10 and 14 rows is slower than at 8, 12 and 16 in 270 of the grid's 330 such pairs (both schedules;
  median +5.3%; Q4_0 4,096 x 1,024 with the turns 156 us at 6 rows, 143 at 8). The kernels take 4 rows while 4 remain,
  then one at a time (`q4_0_gemm`, flagos-spacemit-ime-kernels.cpp:2917-2936, as upstream's `gemm_kernel_i8i4_hp`,
  ime2_kernels.cpp:5611), so 6 rows take three kernel calls per slab and 8 rows two; that a call costs about as much
  with 1 row as with 4 is an inference (not isolated; at 2 against 4 rows, where the copies dominate, the difference
  disappears: -2.5% in the median). Paths A and C and the IME path call the kernels alike, and the quantize step
  splits the rows the same way (full groups of 4 interleaved for the 4-row kernel, the rest row by row,
  flagos-spacemit-ime.cpp:408-439).

Reading, from `gemm_slabs` (the in-tree protocol, ime.cpp:446-490) and the grid; the costs were not timed apart. With
the turns a pair takes about R(c + p) + max(c, p) + (R + 1)w, with c a slab's copy, p its multiply and w a wait, the
odd tile ending one step after the even one; without them each tile takes R(c' + p), c' a copy slowed by the
partner's at the same time. The turns pay when R(c' - c) > max(c, p) + (R + 1)w: the slabs per tile and how much a
pair's copies slow each other decide, not whether the activations' bytes reach a slab's, so the byte rule asked the
wrong question. The grid fits that reading: at 1 slab per tile, where the tail falls on the only slab, the turns lead
by 7.9% at most and only at 2-4 rows, and are behind from 6 rows in every shape; at 4 slabs per tile, where neither
schedule nears the DRAM ceiling (15-24 GB/s with the turns at 2 rows, 15-18 without), they lead by up to 40%; at 38
slabs per tile both read the weights at 23-30 GB/s at 2-4 rows, near the 28 GB/s ceiling, where keeping copies apart
gains little, and at rows of 1,024 values (the shortest copies, against 39 waits) `fit,staged` leads from 2 rows for
Q4_0 and Q4_1; longer rows cross later (Q4_0 at 4-12 slabs per tile: 10 rows at 1,024 values, 14 at 4,096; Q8_0: 14
and 24-28), as a wait does not shrink with the copy; and the remainder rows lengthen p at 6, 10 and 14 rows, which
jags the crossover. The test that would settle the mechanism: path B's copies, multiplies and waits timed apart per
tile.

Verdict by the rules (part 2 from 16 tokens, the user's trial): the grid's clause fails on the valid Q4 grids (6.7%
and 10.6% against 4% allowed), so `fit,mixed` is not adopted, whatever the model runs would show; the switch itself
works (the bench). The three model sessions were not run: their prompts (2, 4, 8, 16, 32 tokens and up) hold no length
from 9 to 15 or from 17 to 31, where the thresholds act, so `fit,mixed` would have run `fit`'s code up to pp8 (on Q8_0
up to pp16) and `fit,staged`'s from the next length, which says nothing about the threshold beyond the grid: a gap in
this run's preparation. The earlier "crossover between 8 and 16 rows" was those lengths' resolution: the grid puts it
near 10 rows at rows of 1,024 values, Qwen3-0.6B's (an estimate for the model from the nearest shapes).

No rule of the byte form passes the clause: the best of "the turns from r slabs per tile while m * a < f * slab", f
linear in log2 of the row length, fitted to the grid's own rows, still misses by 6.5% at 4 or more slabs per tile
(then by 12.3% at 1-2), or at best by 10.2% over all shapes: the crossover moves with the slabs per tile, the row
length, the layout and the remainder rows together. Options (proposed 2026-10-22; the user's choice):
1. Measure per weight shape: at load or at a shape's first graph, both schedules timed for each distinct weight
   shape at 1-32 rows (in the grid `fit,staged` leads in every shape from 28 rows), the faster used per row count;
   general by construction, outputs unchanged (the same bytes), noise only near a crossover, where the two are close;
   costs load time (seconds for Qwen3-4B, an estimate) and more code than the others.
2. A table from the grid (per layout, by row length, slabs per tile and rows): small and deterministic; right at the
   grid's shapes, unverified between them (a grid of other shapes, e.g. Qwen3's rows of 2,560 values, would test it).
3. `fit,mixed` as it is, the grid's clause amended after the grid (recorded as such): +1.0% on average against the
   faster schedule, up to 6.7-10.6% just below its thresholds and 35-46% at 1-2 slabs per tile (no Qwen3 matrix has
   so few).
Independent of these: the last 2-3 rows of a batch through the 4-row kernel (padded to 4) on every path, predicted to
remove the excess at 6, 10 and 14 rows (median 5% there; an estimate). For Q4_0 the outputs of those rows change: its
4-row quantizer takes one scale per 32 values for all four rows (the maximum over the four, quantize_a_4row_i8_hp,
flagos-spacemit-ime-kernels.cpp:1492-1540; reference 507-560), where the 1-row quantizer takes each row's own, so
those rows would be quantized like every full group already is, a question for the logits check. For Q4_1 and Q8_0
the 4-row quantizer keeps a scale per row (quantize_a_nrow_i8_ref, 674-697), so outputs stay identical if the 4-row
and 1-row kernels give the same values per row (a check on the K3 decides: the IME kernels run only there). Not
small: the activations need room for whole groups (the workspace, the TCM checks of path B and `fit`, the copy into
TCM), and the padding rows' outputs a scratch buffer.

The user's direction (2026-10-22): a generally good rule rather than the best in every case, so option 1 is dropped.
Proposed for that: `fit,mixed` and no turns when a tile has one slab (the tail then falls on the only slab): on the
grid the 1-slab cases go from +11.9% on average (worst +45.7%) to +0.3% (worst +7.9%), nothing else changes (4 or
more slabs per tile: +1.0% on average, worst +10.7%; 2: +4.3%, worst +22.6%; from the two schedules' times). The
grid's clause would then be replaced, after the grid, by accepting these misses (to be recorded as such), and the
model runs would decide by parts 1 and 2 with prompt lengths near the thresholds added.

**`fit,staged` chosen (user, 2026-10-22), and its check on Q8_0.** The user set small batches aside (both parts of
the rule from 16 tokens, a decision made after the data) and chose `fit,staged` on the recorded runs: on Qwen3-0.6B
and Qwen3-4B (Q4_0) the fastest setting measured from 16 tokens or level with the fastest (0.6B pp32: -0.5% against
`rows`), 12-134% (0.6B) and 29-121% (4B) faster than `split` at 16-100 tokens, pp512 and tg128 unchanged, outputs
identical; its losses at 2-8 tokens (against `fit` up to -8.0%, against `split` -4.5% on 0.6B at pp2) are the small
batches set aside. `fit,mixed` and the one-slab condition are not pursued. No Q8_0 model has run end to end, so before
the default changes, one session on Qwen3-0.6B in Q8_0 (unsloth's file, 639,447,744 bytes) with `split` (the
reference), `fit` and `fit,staged`: `EXPS="fit fit-staged" REPS=5` (132 runs: the text 3, the logits at `-ub` 128, 64
and 16 9, llama-bench 120: pp128 and tg128, pp512, pp2-pp129 at 17 lengths and pp160-pp320, 5 times per mode, with
and without the counters).

Predictions (estimates fixed before the run, from the grid's Q8_0 shapes with rows of 1,024 values, those of 0.6B's
q, k, v, gate and up; its o and down have 2,048 and 3,072 values, not in the grid):

| What | Prediction |
|---|---|
| outputs | identical: the same text, KLD at the floor, the same top token at `-ub` 128, 64 and 16 |
| pp2-pp8 | `fit,staged` slower than `fit` (the grid: the turns 6-15% faster at 2-8 rows on these shapes) |
| pp16 | `fit,staged` within 3% of `fit` (the grid: level at 16 rows of 1,024 values, the turns 8-9% faster at 4,096) |
| pp32-pp129 | `fit,staged` 4-9% faster than `fit` (the grid: 7-11% at 32 rows of 1,024 values; Q4_0 0.6B: 6.5-9.2%) |
| pp160-pp320 | `fit,staged` 2-5% faster than `fit` (Q4_0 0.6B: 4.1-4.6%) |
| against `split` | faster from pp16, by an amount nothing measured predicts for Q8_0 (Q4_0 0.6B: 12-134% at 16-100 tokens) |
| pp512, tg128 | unchanged within 0.5% |

Criterion (fixed before the run): the choice holds on Q8_0 if from 16 tokens no tested length is more than 1% slower
than `split` and every length from 16 to 100 tokens is at least 5% faster than `split` (the rule's two parts from 16
tokens; `rows` is not in this run). Against `fit` the run reports, with no bar. If the criterion fails, the default
change waits and the Q8_0 case is examined first.

**The Q8_0 check on the K3 (2026-10-22, build `cdfd29e`; 5 runs per mode, the modes interleaved; `~/fx-q8fs.log`).**
All 132 runs returned 0, and the criterion is met: the choice holds on Q8_0.

| Tokens | `fit,staged` against `split` | against `fit` | Predicted against `fit` |
|---|---|---|---|
| pp2, pp4, pp8 | -3.7%, +1.7%, -1.7% | -9.7%, -5.8%, -5.6% | slower: met |
| pp16 | +5.4% | +1.8% | within 3%: met |
| pp32-pp48 (7 lengths) | +7.5% to +78.4% | +6.7% to +8.1% | 4-9%: met |
| pp64, pp65 | +11.4%, +126.5% | +9.8%, +10.2% | 4-9%: about a point above |
| pp96-pp129 (4 lengths) | +9.3% to +105.7% | +7.1% to +7.7% | 4-9%: met |
| pp160-pp256 (3 lengths) | +8.6% to +9.0% | +5.2% to +5.4% | 2-5%: 0.2-0.4 points above |
| pp320 | -0.2% | +0.0% | 2-5%: missed (path B no longer runs, below) |
| pp512, tg128 | -0.2%, +0.2% | +0.0%, -0.1% | unchanged within 0.5%: met |

- The criterion: from 16 tokens the slowest length against `split` is pp320 at -0.2% (bar -1%), and the smallest gain
  at 16-100 tokens is pp16's +5.4% (bar +5%; the standard deviations 0.3% of the means), so both parts hold. pp2 and
  pp8 (-3.7%, -1.7%; pp8's runs scatter by 3.0-3.3% in all modes) are among the small batches set aside.
- Outputs: the same text in all three modes; KLD at the floor (mean 0.000000, maximum 0.000056) for both candidates at
  `-ub` 128, 64 and 16; `split`'s perplexity the same at all three (13.6107). The same top token in 99.951% of the
  2,040 tokens compared, not 100% as predicted: one token, in both candidates at all three batch sizes. Checked
  2026-10-23 from the runs' own output: the candidates' perplexity (`Mean PPL(Q)`, in `kld-*.out`: llama-perplexity's
  plain `LOG` goes to stdout) is 13.610728 ± 1.024828 in all six runs and `split`'s 13.6107 ± 1.02483 at all three
  batch sizes, the same to every digit printed; and the provider's check on the K3 (the prepare step, all checks
  passed) compares every setting's output with `split`'s byte for byte, Q8_0 at 3, 28 and 64 rows among its cases
  (`check_mm_paths`). So the one token comes from the comparison: llama-perplexity keeps the reference's
  log-probabilities in 16 bits and takes the reference's top token from those (tools/perplexity/perplexity.cpp:79-104,
  222-239), so a near-tie can count as a different top token. Not shown directly (a one-ulp difference at one token
  would not move the perplexity's digits); `split` against its own base would. Also read there: the base clamps
  log-probabilities at 16 below the top logit (perplexity.cpp:86), so its own perplexity reads 1.0% lower (`Mean
  PPL(Q)/PPL(base)` 1.010142 with the same logits); no criterion here uses it.
- Path B's reach (read in the code, matching the counters): path B runs while the batch's activations and one slab fit
  in a tile's 393,216 bytes of TCM (`slabs_hold_activations`, flagos-spacemit-ime.cpp:283-285). With Q8_0's 38-byte
  activation blocks and 34-byte weight blocks per 32 values (flagos-spacemit-ime-kernels.h:36-67) that is up to 294
  rows for 0.6B's q, k, v, gate and up (rows of 1,024 values), 133 for o (2,048) and 79 for down (3,072). In `fit-ops`
  down's odd tiles lag at 65 rows and not at 96, o's at 129 and not at 160, the other five's at 256 and not at 320. So
  at pp320 both settings run the same paths (equal times, odd excess ±0.0% in both); the prediction, taken from Q4_0
  0.6B, missed because Q4_0's smaller blocks fit 323 rows of 1,024 values. The gain against `fit` falls as matrices
  leave path B: 6.7-10.2% while all seven take it (32-65 tokens), 7.1-7.7% without down (96-129), 5.2-5.4% without o
  and down (160-256), none at 320. Within 32-65 tokens it is largest at 64 and 65 (by the code the turns' tail, one
  slab's multiply, grows with the rows while the copies' collisions do not; inference).
- The turns' tail as on Q4_0: under `fit`, in path B from 32 rows, the odd tiles lag by 19.0-24.0% with 4 slabs per
  tile (k, v, o, down), 10.1-12.0% with 8 (q) and 6.9-8.1% with 12 (gate, up), about 0.9/R; under `fit,staged` by
  0.2-3.5%, less with more rows (0.8-3.5% at 32-65 rows, 0.2-1.3% at 160-256): the residual odd-tile disadvantage seen
  on Q4_0, cause not isolated.

Next, by the user's plan (2026-10-22): prompts of 9-15 and 17-31 tokens, a model of another family of similar size,
bigger models; then graph splitting, with fusion later. The default change and its acceptance runs after those checks
(proposed).

**Prompts of 8-32 tokens (the user's step 2), prepared 2026-10-23.** The runner's `PPODD` sets the prompt lengths of
its run of single batches in llama-bench's `-p` form, ranges included (default the 17 lengths above; checked on the
Mac: without it the runner's commands are the same as before for every experiment, and llama-bench takes `-p 8-32`
and rejects a malformed range). Three sessions, `EXPS="fit fit-staged" PPODD=8-32 PPL_TEXT=`, without the logits check
(the provider's check on the K3 compares the paths byte for byte at 2-128 rows, 17, 19, 21 and 28 among them):
Qwen3-0.6B in Q4_0 and in Q8_0 (`REPS=5`), Qwen3-4B in Q4_0 (`REPS=3`).

Predictions (estimates fixed before the run, from the runs at 8, 16 and 32 tokens, the grid and the code):

| `fit,staged` | 0.6B Q4_0 | 0.6B Q8_0 | 4B Q4_0 |
|---|---|---|---|
| measured against `fit` at pp8 / pp16 / pp32 | +0.1% / +5.1% / +7.5% | -5.6% / +1.8% / +6.7% | -5.7% / +3.1% / +3.9% |
| measured against `split` at pp16 / pp32 | +12.2% / +17.1% | +5.4% / +7.5% | +29.2% / +47.6% |
| against `fit` at 9-15 tokens | level or faster | slower up to about 12-15 | slower up to about 11-14 |
| against `fit` at 17-31 tokens | +5-8% | +2-7% | +3-4% |
| against `split` at 17-31 tokens | above +12% | above +5.4% | above +29% |

- `split` from 17 rows (read in the code): its path C cuts the rows into blocks of 16 for the matrices with at most 64
  output columns per batch row and reads their weights once per block (flagos-spacemit-ime.cpp:461-468, 504-524): on
  0.6B k, v, o and down, about 47% of a layer's weight bytes, on 4B k and v, about 5%. So on 0.6B `split` should lose
  speed from pp16 to pp17, and the gain over it at 17-31 tokens should exceed pp16's (estimate; no model of the cost).
- Every setting's rate should dip at lengths that are not a multiple of 4: the last rows go one at a time
  (flagos-spacemit-ime.cpp:133-142; the grid: 6, 10 and 14 rows a median 5.3% slower than 8, 12 and 16 on path B). So
  the comparisons are read length by length.

Criterion (fixed before the run): on each model, at 17-31 tokens no length more than 1% slower than `split` and every
length at least 5% faster (the rule's two parts, from 16 tokens). 9-15 tokens and the comparison with `fit` are
reported, with no bar. If the criterion fails, the default change waits and the failing lengths are examined first.

**Prompts of 8-32 tokens on the K3 (2026-10-23, build `2c7c761`, whose code is `cdfd29e`'s; the modes interleaved, 5
runs per mode on Qwen3-0.6B in Q4_0 and in Q8_0, 3 on Qwen3-4B; `~/fx-mid06.log`, `~/fx-mid06q8.log`,
`~/fx-mid4b.log`).** All runs returned 0 (123, 123 and 75), every mode generated the same text on each model, and the
criterion is met on all three models.

| `fit,staged` | 0.6B Q4_0 | 0.6B Q8_0 | 4B Q4_0 |
|---|---|---|---|
| against `split` at 17-31 tokens (the criterion: +5% or more) | +16.4% to +30.6% (lowest pp27) | +7.5% to +22.2% (pp27) | +23.5% to +36.7% (pp20) |
| predicted | above +12%: met | above +5.4%: met | above +29%: missed at pp20-pp28 (4B's down, below) |
| against `split` at 9-16 tokens (no bar) | +4.0% to +14.9% | -1.2% to +7.2% | +15.0% to +29.6% |
| against `fit` at 9-15 tokens | +1.9% to +7.0% | -3.1% to +3.5% | -0.6% to +3.6% |
| predicted | level or faster: met | slower up to about 12-15: met (the crossover below) | slower up to about 11-14: missed (crossover at 9-10 rows) |
| against `fit` at 17-31 tokens | +4.2% to +7.9% | +2.7% to +8.4% | +2.4% to +4.8% |
| predicted | +5-8%: 14 of 15 lengths (pp17 +4.2%) | +2-7%: 10 of 15 (5 up to 1.4 points above) | +3-4%: 11 of 15 (4 within 0.9 points) |
| pp8 / pp16 / pp32 against `fit`, here (earlier runs) | +0.5 / +5.4 / +8.2% (+0.1 / +5.1 / +7.5%) | -5.1 / -0.1 / +7.2% (-5.6 / +1.8 / +6.7%) | -6.4 / +3.5 / +3.5% (-5.7 / +3.1 / +3.9%) |
| pp16 / pp32 against `split`, here (earlier runs) | +12.1 / +17.8% (+12.2 / +17.1%) | +3.0 / +7.2% (+5.4 / +7.5%) | +29.5 / +47.1% (+29.2 / +47.6%) |

- The criterion: at 17-31 tokens the smallest gains over `split` are +16.4% (0.6B Q4_0, pp27), +7.5% (Q8_0, pp27) and
  +23.5% (4B, pp20). No length from 8 to 32 is more than 1.2% slower than `split` (Q8_0's pp8 and pp9, -1.2% each,
  standard deviations 1.6-2.4%, below 16 tokens).
- pp16 on Q8_0, the one point that did not repeat: `split` and `fit` within 1% of the Q8_0 check's (300.3 against
  301.7 t/s, 309.5 against 312.3), `fit,staged` 309.1 ± 5.8 against 318.0 ± 0.9, so +3.0% over `split` where that check
  had +5.4% (the counter modes +4.0% and +5.1%). At exactly 16 tokens on Q8_0 the gain sits at part 2's bar (3.0-5.4%
  in the four measurements); from 17 tokens it is at least 7.5%. Not this run's criterion (17-31 tokens); part 2 from
  16 tokens is the user's trial (2026-10-22).
- The crossover with `fit` (the counter modes' graph times): on 0.6B Q4_0 level at 8 rows (+0.2%), `fit,staged` 2.7-5.0%
  faster from 9; on Q8_0 slower at 8-10 and 12-13 rows (0.5-7.1%), level at 11, 1.4-2.7% faster from 14; on 4B slower
  at 8-9 (3.3-4.5%), faster from 10 (1.0-4.0%) but for 12 (+0.8%). llama-bench agrees within its noise (Q8_0's
  pp10-pp13 within ±1.7%).
- `split` from 17 rows, as predicted: on 0.6B its rate falls 16.9% (Q4_0) and 16.3% (Q8_0) from pp16 to pp17, on 4B
  7.4%, against 0.7-4.6% for both fit settings. The counters: on 0.6B Q4_0 k, v, o and down take 75-83% longer at 17
  rows than at 16, q, gate and up 13-15%; on 4B k and v 73-77%, the others 12-17%. By the code
  (flagos-spacemit-ime.cpp:461-468, 504-524), from 17 rows those matrices take blocks of 16 rows, one (block, column
  range) task per tile, so half the tiles multiply 16 rows over twice the columns they had at 16 rows; their time then
  stays near that level (k on 0.6B Q4_0: 2,732-2,852 µs per graph at 17-29 rows against 1,612 at 16; 3,350 at 31,
  where the 15-row block's remainder rows make it the slower half).
- 4B's down leaves path B at 20 rows (read in the code, seen in the counters): path B needs the batch's activations
  and one slab in a tile's 393,216 bytes of TCM (`slabs_hold_activations`, flagos-spacemit-ime.cpp:283-285). 4B's down
  has rows of 9,728 values: 11,020 bytes of Q4_0 activations per row next to a 175,104-byte slab, so up to 19 rows;
  its four Q4_1 layers (i8 activations, 11,552 bytes per row; slab 184,832) up to 18
  (flagos-spacemit-ime-kernels.h:54-68). Path A needs 4 rows for every tile (32 rows), so from there to 31 rows down
  runs on path C in both fit settings, as in `split`: under `fit,staged` its time is within 0.7% of `split`'s at 20
  rows (Q4_1: 0.3% at 19), against 36% less at 18; under `fit` its odd tiles' excess falls from 8.3% to 0.7% (Q4_1:
  8.8% to 1.9%). So the gain over `split` falls from +34.3% at pp19 to +23.5% at pp20 and returns at pp32 (+47.1%,
  down on path A). The prediction missed it by not computing path B's reach on 4B, as Q8_0's pp320 did. On 0.6B every
  matrix stays on path B through 32 rows (down up to 97 rows in Q4_0, 79 in Q8_0).
- The remainder rows (measured): in every setting on every model, a batch of 4q+4 tokens takes less time than one of
  4q+3: under `fit,staged` 5.0-11.7% less on 0.6B and 9.1-15.5% on 4B (where no path changes between the two
  lengths), under `split` 2.9-16.4%. The counters put the difference in the matmuls: each of a batch's last 1-3 rows
  costs 0.6-0.9 of a group of 4 (three of them 1.8-2.6 groups); gemm_rows runs them one at a time
  (flagos-spacemit-ime.cpp:133-142), and why one row costs that much inside the kernel is not isolated. The padding
  proposed above (the last 2-3 rows through the 4-row kernel) would bring a batch of 4q+3 tokens near the time of 4q+4
  and save a few percent at 4q+2 (estimates); it applies to every setting, `split` included, and changes Q4_0's
  outputs (a logits check decides).
- The pair turns at 8-31 rows (counters): under `fit` the odd tiles lag by 4.6-22.4% on 0.6B (about 0.9/R as before,
  growing slowly with the rows) and 1.6-20.4% on 4B; under `fit,staged` by 0.3-4.8% on 0.6B and up to 6.6% on 4B (its
  Q4_1 down at 8 rows): the residual odd-tile disadvantage seen from 32 rows, also at 8-31.
- pp128-pp320, pp512 and tg128 as in the earlier runs: against `fit` +3.5 to +6.0% (0.6B Q4_0), +5.3 to +7.8% to pp256
  and level at pp320 (Q8_0), level on 4B; pp512 and tg128 within 0.4% of `split` on all three.
- For bigger models (the user's step 4), read in the code: the longer a matrix's rows, the fewer batch rows path B
  holds, so more of a model's matmuls run on path C between that limit and 32 rows under `split`, `fit` and
  `fit,staged` (`spacemit` keeps path B there, reading the activations from the workspace: gemm_slabs,
  flagos-spacemit-ime.cpp:326-328). For Q4_0, a down with rows of 12,288 values would hold 12 rows (13,920 bytes of
  activations per row next to a 221,184-byte slab); a model's limits follow from its shapes before any run.

**Another family: Llama-3.2-1B-Instruct (the user's step 3), prepared 2026-10-23.** The user's choice among candidates
checked the same day against the provider's rules: llama.cpp's graph builder per architecture (only ops the provider
takes), the published configs (Q4_0 rows a multiple of 256 values, every weight's output rows a multiple of 32,
attention heads of at most 128 values: flagos-spacemit-weights.cpp:10-28, flagos-spacemit-attn.cpp:26) and the GGUF
files' tensor lists (read from their headers). The files: unsloth's on ModelScope, like the Qwen3 files, downloaded on
the board directly (its DNS resolves modelscope.cn): Q4_0 773,025,824 bytes (SHA-256 `66bfbb2d...1a70c2`), every
matmul Q4_0 but `ffn_down` of layers 0-1 (Q4_1, the imatrix rule), the tied embedding and head Q6_K, `rope_freqs`
F32; Q8_0 1,321,082,528 bytes (`3f87a880...c7ffd1`), all Q8_0. 16 layers, hidden 2,048, 32 query heads of 64 values
and 8 KV heads, feed-forward 8,192, vocabulary 128,256. On the Mac's simulated device (`build-host`, the same op checks
as the board) both files take 2 graph splits with flash attention on. Also fit by the code and the files (unsloth's
Q4_0 and Q8_0 on ModelScope): ERNIE-4.5-0.3B, OLMo-2-1B, Llama-3.2-3B, Ministral-3-3B, SmolLM3-3B. Not: Gemma 3
(GELU, tanh, SCALE; heads of 256 values on 1B), Phi-4-mini (a SCALE on Q in every layer, src/models/phi3.cpp:123),
Granite (SCALE), LFM2 (convolutions), Falcon3 (heads of 256).

What differs from Qwen3 (read in the code; shares by weight count):

| Matrix (K x N) | q 2048 x 2048 | k, v 2048 x 512 | o 2048 x 2048 | gate, up 2048 x 8192 | down 8192 x 2048 |
|---|---|---|---|---|---|
| share of a layer's weights | 6.9% | 3.4% (both) | 6.9% | 55.2% (both) | 27.6% |
| slabs per tile (N/256; Qwen3 4-38) | 8 | 2 | 8 | 32 | 8 |
| path B holds, rows (Q4_0 / Q8_0) | 153 / 133 | 153 / 133 | 153 / 133 | 153 / 133 | 26 / 11 (Q4_1: 24) |
| `split`'s 16-row blocks from (rows) | 32 | 17 | 32 | 128 | 32 |

- Under `fit` and `fit,staged`, down leaves path B at 27 rows in Q4_0 (25 for the Q4_1 layers) and at 12 in Q8_0, then
  runs path C, as under `split`, until path A at 32. Under `spacemit` it stays on path B with the activations outside
  TCM (gemm_slabs, above), a comparison at 12-31 rows that no run has made.
- `split` should slow less at 17 rows than on Qwen3-0.6B: only k and v take 16-row blocks there (3.4% of the weights,
  against 47%). From 33 rows its blocks leave the tiles uneven for q, o, down, k and v (at 33 rows 9 tasks for 8
  tiles: four tiles take two tasks of 16 rows by 768 columns, three none; flagos-spacemit-ime.cpp:504-524), as on
  Qwen3 at 33 tokens.
- Heads of 64 values and normal-mode RoPE with frequency factors run in a model on the provider for the first time
  (their op tests passed on the K3: ROPE 165 cases, attention 1,324; build.md §7). So the sessions start with the
  acceptance script against the CPU.

Sessions, in one chain: (1) `spacemit_check.py --milestone m2e --model` the Q4_0 file (2 graph splits, flash
attention kept under `-fa auto`, perplexity within 1% of the CPU, KLD against the CPU at 512-token batches and `-ub 1`;
llama-bench of the provider, the CPU and the IME path); (2) `EXPS="mm pb fit fit-staged" REPS=5` on the Q4_0 file with
`PPODD=2,4,8,16,24,26,27,28,32,33,36,37,40,44,48,64,65,96,100,128,129` (down's limit at 26-28 added); (3) the same on
the Q8_0 file with `PPODD=2,4,8,11,12,13,16,24,32,33,36,37,40,44,48,64,65,96,100,128,129` (its limit at 11-13).

Predictions (estimates fixed before the run, from the Qwen3 runs' per-matrix times, the shapes and the code):

- Placement and accuracy: 2 graph splits per token and flash attention on the provider, as on the Mac. Perplexity
  within 1% of the CPU likely, not certain (Qwen3-0.6B +0.89%, 4B +0.05%); mean KLD against the CPU at 512-token
  batches between 4B's 0.0034 and 0.6B's 0.0147 (no model of it).
- Outputs: every candidate the same text as `split` and KLD at the floor at `-ub` 128, 64 and 16.
- Counters: under `fit` the odd tiles lag by about 0.9/R (about 45% for k and v, 11% for q, o and down, 3% for gate and
  up), under `fit,staged` by 0-5%; down's lag under `fit` falls to about zero from 27 rows in Q4_0 (25 for Q4_1) and
  from 12 in Q8_0.
- `fit,staged` against `split`, Q4_0: +12 to +25% at 16-26 tokens (path B's gain over path C at 16 rows on Qwen3,
  1.15-1.25x per matrix on 0.6B and 1.22-1.53x on 4B, weighted by these shares), +8 to +18% at 27-31 (down as in
  `split`), +25 to +50% at 32 (down on path A; 4B +47%), at least that from 33 to 100 at the lengths that are not
  multiples of 16 (`split`'s blocks), +5 to +15% at 128-129 (path B against `split`'s path A for six matrices),
  within ±1% at 160-512 (path A everywhere), tg128 within ±0.5%. `split` from pp16 to pp17: -4 to -8% (Qwen3-0.6B
  -16.9%).
- Q8_0: +2 to +8% at 12-31 tokens (path B's gain at 16 rows on 0.6B in Q8_0: 1.04-1.10x per matrix; down as in
  `split`), so part 2 of the rule (+5%) is at risk there; from 32 tokens as Q4_0, with smaller gains per matrix.
- Against `fit`: slower at 2-8 tokens (as on Qwen3), +2 to +6% at 10-26 (the turns' tail, largest on k and v), +1 to
  +3% from 27 (down off path B in both).
- `spacemit` as `fit` where path B holds the activations; at 27-31 tokens (Q4_0) and 12-31 (Q8_0), where its down
  reads them from the workspace, no estimate of the sign; 2-8% slower than `fit,staged` at 128-129 (its down there).
  `rows` as `split` below 32 tokens; from 32 level with `fit,staged` at multiples of 32 and behind it elsewhere (as on
  Qwen3-0.6B).

Criterion (fixed before the run), per file: from 16 tokens no tested length more than 1% slower than `split` and every
tested length from 16 to 100 at least 5% faster (the rule's two parts from 16 tokens), with outputs identical; and, on
the Q4_0 file only, the acceptance script's checks pass: the paths new to a model (heads of 64 values, RoPE with
frequency factors) do not depend on the weight format, and Q8_0 changes only the matmuls, whose op tests pass on the
K3 (if its perplexity bound fails, the model's accuracy is examined first, against the IME path on the same file).
The comparisons with `fit`, `rows` and `spacemit` are reported, with no bar. If the criterion fails, the default change
waits and the failing lengths are examined first.

**Llama-3.2-1B in Q4_0 on the K3 (2026-10-23, build `2c7c761`, whose code is `cdfd29e`'s; `~/acc-l1b.log`,
`~/fx-l1b.log`).** The acceptance script passes 32 of its 34 checks; the two failures are its perplexity bound, so by
the criterion the model's accuracy is examined first and the default change waits. The benchmark meets its part of the
criterion with wide margins.

- Accuracy against the CPU (8 x 512 tokens): perplexity 14.149 against 13.4182 (+5.45%), with `-fa on` and under
  `-fa auto`; mean KLD 0.0548 ± 0.0022 at 512-token batches (99% 0.468, maximum 1.28, same top token 88.5%) and
  0.0513 ± 0.0020 at `-ub 1` (0.460, 1.23, 89.1%), against Qwen3-0.6B's 0.0147 and Qwen3-4B's 0.0034, so the prediction
  missed. The generated text leaves the CPU's after 47 characters. Passed: 2 graph splits per token, flash attention
  kept under `-fa auto`, all op tests. llama-bench (3 runs): the provider pp128 371.8 and tg128 33.5 t/s, the IME path
  349.3 and 32.5, the CPU 76.1 and 16.4.
- What the two KLD values say (inference, from the code): they are nearly equal, so the paths that run only in batches
  (Q4_0's 4-row quantizer, one scale per 32 values over four rows, flagos-spacemit-ime-kernels.cpp:1526-1534) explain
  at most their difference, and two different attention kernels (tiled at 512-token batches, one-chunk at `-ub 1`)
  would have to fail alike. The suspects: the lossy Q4_1 conversion of `ffn_down` in layers 0-1 (the check tool
  measures the conversion alone at NMSE 5.86e-3 on random weights, its INFO line in the same log; llama-quantize makes
  exactly these layers Q4_1 to "guard against craziness in the first few ffn_down layers", src/llama-quant.cpp:620),
  RoPE in normal mode with frequency factors, and the head's requantization. Queued after the Q8_0 session: the IME
  path on the same file and the Q8_0 file against the CPU, and `-fa off` (attention on the CPU).
- The benchmark: all 220 runs returned 0, the five settings generated the same text, and every candidate's logits equal
  `split`'s at `-ub` 128, 64 and 16 (mean KLD 0.000000, maximum 0.000042, same top token 100%). From 16 tokens the
  worst length against `split` is pp160 at -0.15% (bar -1%), and the smallest gain at 16-100 tokens is pp28's +19.5%
  (bar +5%). `fit,staged` is the fastest of the five at every tested length from 16 to 129 tokens; from 160 tokens and
  in tg128 all five lie within 1% of one another; below 16 tokens `spacemit` and `fit` lead it by 2-7% (the small
  batches set aside).

| `fit,staged` against | `split` | `fit` | `rows` | `spacemit` |
|---|---|---|---|---|
| pp2, pp4, pp8 | +3.8, +20.6, +11.8% | -5.2, -4.0, -2.4% | +6.8, +22.3, +12.1% | -7.3, -6.2, -3.1% |
| pp16, pp24, pp26 | +25.6 to +31.8% | +4.2 to +5.1% | +26.5 to +31.7% | +4.4 to +5.7% |
| pp27, pp28 | +20.0, +19.5% | +3.4, +3.2% | +19.5, +19.4% | +8.8, +8.3% |
| pp32 | +32.4% | +4.1% | +3.6% | +21.1% |
| pp33-pp48 (6 lengths) | +72.3 to +91.1% | +3.0 to +3.8% | +19.4 to +32.9% | +6.2 to +13.2% |
| pp64, pp65 | +43.7, +102.7% | +3.6, +3.9% | +9.0, +24.0% | +22.1, +15.3% |
| pp96, pp100 | +97.7, +98.4% | +4.1, +3.9% | +11.6, +24.8% | +21.3, +14.1% |
| pp128, pp129 | +11.8, +20.2% | +3.4, +3.7% | +12.0, +20.2% | +12.0, +19.5% |
| pp160-pp512 (5 lengths) | -0.2 to +0.1% | -0.2 to +0.3% | -0.3 to +0.2% | -0.8 to +0.1% |
| tg128 | +0.4% | +0.4% | +0.5% | +0.1% |

- Against the predictions: against `split` +25.6 to +31.8% at 16-26 tokens (predicted +12 to +25%) and +19.5 to
  +20.0% at 27-28 (+8 to +18%), both above: path B's gain per matrix at 16 rows is 1.24-1.55x, nearer Qwen3-4B's than
  0.6B's, as Llama's gate, up and down (83% of the weights) gain 1.38-1.55x. pp32 +32.4% (+25 to +50%: met), every
  length from 33 to 100 at least that (met), pp128 and pp129 +11.8% and +20.2% (+5 to +15%: pp129 above), 160-512 and
  tg128 within 0.4% (met). Against `fit`: slower at 2-8 tokens (met), +4.2 to +5.1% at 16-26 (+2 to +6%: met), +3.0 to
  +4.1% from 27 tokens (+1 to +3%: up to 1.1 points above; k and v keep their turns' tail on path B). `split` from pp16
  to pp17: not tested (pp17 not in the run). `spacemit` as `fit` where path B holds the activations (pp16-26 within
  0.6%: met); 10.7-16.3% behind `fit,staged` at pp128-129 (predicted 2-8%: missed). `rows` as `split` below 32 tokens
  (met), but behind `fit,staged` even at multiples of 32, by 3.6-12.0% (predicted level: missed).
- Path B with the activations outside TCM, the open comparison: at pp27 and pp28, where only `spacemit` keeps down on
  path B (the activations read from the workspace), it is 5.0% and 4.7% slower than `fit`, whose down takes path C
  there; at pp32, against `fit`'s path A, 14.0% slower. So `fit`'s rule (path C below 32 rows when the activations do
  not fit) is the faster choice on this shape.
- The counters, as predicted: under `fit` the odd tiles lag by 16-44% on k and v (2 slabs per tile; rising with the
  rows toward the predicted 45%), 4-12% on q, o and down (8 slabs; predicted 11%) and 1-3% on gate and up (32 slabs;
  predicted 3%), under `fit,staged` by 0.2-6.0% (predicted 0-5%). Down's lag under `fit` falls to 1.3% at 27 rows
  (Q4_1: 2.2% at 26); from there down takes `split`'s time under `fit,staged` (ratio 0.99-1.00) and from 32 rows is
  1.54x faster (path A). At 33 rows `split`'s k, v, q and o take 3.3-3.6x as long as under `fit,staged` (its 16-row
  blocks give four tiles two tasks and three none, flagos-spacemit-ime.cpp:504-524), its gate and up 1.56x (no blocks
  below 128 rows).

**Llama-3.2-1B in Q8_0, and the accuracy follow-up (K3, 2026-10-23, same build; `~/fx-l1bq8.log`, `~/kld/l1b-*.txt`).**
The Q8_0 benchmark meets its criterion, narrowly as predicted: all 220 runs returned 0, the five settings generated the
same text, every candidate's logits equal `split`'s (mean KLD 0.000000, maximum 0.000048, same top token 100%); from
16 tokens the worst length against `split` is pp256 at -0.28%, and the smallest gain at 16-100 tokens pp16's +5.70%
(bar +5%).

| `fit,staged` against | `split` | `fit` | `rows` | `spacemit` |
|---|---|---|---|---|
| pp2, pp4, pp8 | +7.2, +18.5, +6.0% | -10.8, -6.3, -10.0% | +4.3, +15.3, +5.4% | -10.0, -7.9, -11.0% |
| pp11, pp12, pp13 | +19.4, +1.2, +3.0% | -0.7, -1.0, +0.5% | +18.7, +3.7, +3.3% | -1.2, -6.7, -6.5% |
| pp16, pp24 | +5.7, +9.1% | +3.2, +2.4% | +6.1, +8.4% | -5.0, -5.8% |
| pp32 | +20.6% | +4.0% | +1.0% | +16.6% |
| pp33-pp48 (6 lengths) | +62.6 to +82.5% | +3.2 to +4.0% | +21.1 to +36.5% | +1.6 to +9.4% |
| pp64, pp65, pp96, pp100 | +27.4, +86.5, +79.4, +78.5% | +3.8 to +4.5% | +8.3, +26.4, +9.6, +25.3% | +8.9 to +17.4% |
| pp128, pp129 | +10.7, +20.6% | +4.0, +3.8% | +11.2, +21.1% | +11.1, +20.7% |
| pp160-pp512 (5 lengths), tg128 | -0.3 to +0.1% | -0.2 to +0.2% | -0.3 to +0.1% | -0.3 to +0.1% |

- Against the predictions: +1.2 to +9.1% against `split` at 12-24 tokens (predicted +2 to +8%: pp12 below, pp24 above),
  so part 2 held at pp16 by 0.7 points; from 32 tokens the Q4_0 pattern with smaller gains (met). Against `fit`: level
  at 11-13 tokens (predicted +2 to +6%: missed; the crossover is near 13-16 rows, as on Qwen3-0.6B in Q8_0), +2.4 to
  +3.2% at 16-24, +3.2 to +4.5% from 32 (predicted +1 to +3%: above). `rows` as `split` below 32 tokens (met).
- `spacemit` is the fastest setting at 4-24 tokens on this file (1-11% ahead of `fit,staged`). The counters place it
  in down: under `fit` and `fit,staged` down leaves path B at 12 rows (`fit`'s odd-tile lag +9.3% at 11 rows, -0.2% at
  12) and takes `split`'s time (24.1 against 23.9 ms per graph at 12 rows), while `spacemit` keeps it on path B with
  the activations outside TCM, 20-26% faster at 12-24 rows (25.1 against 31.6 ms at 16). On the Q4_0 file the same
  path is 18.5-21.9% slower than path C at 27-28 rows (its Q4_1 layers 9.5-11.6%), and at 32 rows path A beats both on
  both files. So path B with the activations outside TCM wins at 12-24 rows (Q8_0) and loses at 27-32 rows (Q4_0, Q4_1
  and Q8_0); the mechanism is not isolated, and with one shape per file the boundary between 24 and 27 rows is an
  inference. A rule using that path below some row count would gain about 5-9% at 12-24 tokens on this file (an
  estimate from `spacemit`'s lead); not pursued without the user.
- Accuracy against the CPU (8 x 512 tokens): the provider on the Q8_0 file matches the CPU (mean KLD 0.000756 ±
  0.000025, same top token 99.27%, perplexity 12.8092 against 12.8217, -0.05%), and on the Q4_0 file with `-fa off`
  (attention on the CPU) it misses as with flash attention (mean KLD 0.0559, same top token 88.3%, perplexity +5.44%).
  Both files run the same graph (RoPE in normal mode with frequency factors, heads of 64 values, the norms), so those
  ops and the Q8_0 matmuls are cleared, and the error lies in the Q4_0 file's weights on the provider: the Q4_1
  conversion of layers 0-1's down, the head's requantization from Q6_K, or the Q4_0 kernels (inference from the two
  runs).
- Invalid: the follow-up's first CPU run on the Q4_0 file produced NaN log-probabilities (its log has no `Final
  estimate`, which llama-perplexity prints only when the variance is positive, tools/perplexity/perplexity.cpp:650-657),
  so the provider and IME comparisons made against its base (both report the base perplexity as nan) are discarded.
  The same CPU command before (the acceptance script) and after (`l1b-cpu2.txt`) gave 13.4182: an intermittent fault in
  a CPU-only run, cause not isolated (compare question 8's intermittent CPU-backend crash).
- Next (prepared 2026-10-23): two test files made by llama-quantize from the Q4_0 file, every other tensor copied
  unchanged (src/llama-quant.cpp:705-708, 1161-1163): layers 0-1's down in Q8_0, and the head in Q8_0 (layers 0-1 kept
  in Q4_1, as the imatrix rule applies only with an imatrix, :619); each against the CPU on its own file, and the IME
  comparison rerun. Predicted: if the Q4_1 conversion is the main cause, the first file's KLD falls to Qwen3-0.6B's
  range (0.015 or less); the head alone moved Qwen3's KLD by 0.0001-0.0002 (M2c.2), so the second file should change
  little.

**The cause: the Q4_1 conversion (K3 and the Mac, 2026-10-23; `~/kld/t-*.txt`).** Each file against the CPU on itself,
8 x 512 tokens, in one session:

| File | CPU perplexity | Provider against the CPU: perplexity, mean KLD, same top token |
|---|---|---|
| the Q4_0 file | 13.4182 | +5.52%, 0.0548, 88.3% (the upstream IME path: +5.62%, 0.0554, 88.3%) |
| layers 0-1's down in Q8_0 (2 tensors converted) | 13.4226 | +0.24%, 0.0026, 97.0% |
| the head in Q8_0 (1 tensor converted) | 13.3983 | +5.76%, 0.0538, 88.3% |

- Giving the provider those two tensors in a layout it reads exactly (the Q8_0 file matched the CPU at KLD 0.00076)
  removes 95% of the KLD and brings the perplexity within 1%, while the CPU's own perplexity barely moves (+0.03%):
  the Q4_1 conversion of layers 0-1's down is the cause, both predictions met. The head's requantization changes the
  KLD by 0.001, within its error. The upstream IME path makes the same error, so it is the in-tree route's (copied at
  M2c.1), not the provider's.
- The mechanism, read in the code and the file: the conversion stores Q4_1's fp16 minimum m as an integer zero point,
  zp = -round(m/d) clamped to [0, 15], and the kernels compute d*(q - zp) (make_block_q4_1x32,
  flagos-spacemit-ime-kernels.cpp:163-170, from upstream), so every weight of a 32-value block moves by m + d*zp, up to
  d/2. Read from the file on the Mac (all 524,288 blocks of each tensor): -m/d lies in 0.6-14.6, so no block is
  clamped, and the rounding adds error energy of 0.677% and 0.672% of the weights' energy in layers 0 and 1 (the check
  tool's 5.86e-3 on random weights alike). The shift is common to a block's 32 weights, so it adds up in a dot product
  rather than averaging out (inference), and it falls on the layers llama-quantize makes Q4_1 because Q4_0 fails there.
- The bad CPU run had chunks 1-4 normal (13.65, 13.64, 15.57, 12.32, running estimates) and NaN from chunk 5; the disk
  had 91 GB free.
- The same test on Qwen3 (K3, the same day; test files with the Q4_1 layers in Q8_0, layers 0-2 of Qwen3-0.6B and 0-3
  of Qwen3-4B, each converted from q4_1 as listed by llama-quantize; each file against the CPU on itself): Qwen3-0.6B
  mean KLD 0.0149 -> 0.0044 (-71%), same top token 93.5% -> 96.4%; Qwen3-4B 0.00334 (M2e's 0.003354) -> 0.00260
  (-22%), 96.9% -> 97.3%; the CPU's own perplexity moves by +0.09% and -0.02%. So the conversion is also most of
  Qwen3-0.6B's error against the CPU, which M2e had put down to the IME approach as a whole (the upstream IME path
  had the same KLD). The printed `Mean PPL(Q)/PPL(base)` ratios (0.6B 1.018 -> 1.013, 4B 1.0099 -> 1.0092) carry the
  base's clamp (perplexity.cpp:86), about 1% on Qwen3 (4B's 1.0099 against M2e's +0.05% from the final estimates), so
  KLD is the measure here.
- For step 3: the criterion's accuracy part fails for a reason outside the settings and outside the paths new in this
  model; the benchmark results stand (every setting computes the same bytes).
- Options for Q4_1 (estimates; a decision for the user and the mentor, D4): (a) requantize Q4_1 to Q8_0 at load, as the
  head's Q6_K already is: accurate (this run: KLD 0.0026 with those layers in Q8_0), 1.79x the bytes for those layers,
  about 2% more weight bytes per token on this model and on Qwen3-4B (generation about 2% slower), 1% on Qwen3-0.6B;
  (b) keep the 19-byte layout and add the minimum's exact term outside the IME kernel, out += sum over blocks of (m +
  d*zp) * sum(a), from the activation sums the i8 format already stores: exact, about 1/32 more arithmetic for those
  layers and 2 more bytes per block, new kernel code; (c) keep it, documented as the IME approach's limit. Measuring the
  same test on Qwen3-0.6B (layers 0-2) and Qwen3-4B (0-3) would show how much of their KLD (0.0147, 0.0034) it is.

**The fix: Q4_1 requantized to Q8_0 at load (option (a), the user's choice, 2026-10-23; implemented and checked on the
Mac the same day, run on the K3 2026-10-24).**
- What changed (providers/spacemit only): a layout `q4_1_q8_0_32x32` for Q4_1 matmul weights, the default. At load
  each 32-row group is dequantized and requantized by ggml's own Q8_0 quantizer, the function llama-quantize used for
  the test file above, then stored in the q8_0 32x32 layout and multiplied by the Q8_0 kernels, as the Q6_K head
  already is (`repack_q4_1_q8_0`, `unpack_q4_1_q8_0` in flagos-spacemit-ime-kernels.cpp; flagos-spacemit-weights.cpp;
  `mm_format_of` in flagos-spacemit-ime.cpp). Reads return the nearest Q4_1 and writes must be whole, as for Q6_K. The
  old route stays for comparisons behind `FLAGOS_SPACEMIT_Q4_1=zero-point` (test-only, read once per process, as a
  weight must be read in the layout it was stored in). flagos-check-spacemit checks Q4_1 as it checks Q6_K (against
  the CPU's Q4_1, and tightly against ggml's Q8_0 of the same weights), and the zero-point layout's own checks under
  the switch.
- The cost, from the layouts (estimates of speed): 34 bytes per 32 weights instead of 19 for the Q4_1 layers, so the
  provider reads 15.7 MB more per generated token on Llama-3.2-1B (+1.9% of its weight bytes), 46.7 MB on Qwen3-4B
  (+1.9%) and 4.4 MB on Qwen3-0.6B (+1.1%); generation about as much slower, prompt processing less.
- Checked on the Mac (host build, simulated device): flagos-check-spacemit passes all checks on both routes; Q4_1
  against the CPU's Q4_1 max NMSE 1.8e-5 (16 cases, 1-128 rows; bound 5e-4), against ggml's Q8_0 of the same weights
  4.4e-8 (bound 5e-6), the requantization alone 1.45e-5 (the zero-point conversion: 5.86e-3), reads within 5.4e-6 of
  the GGUF weights (bound 5e-5), the layer with a Q4_1 down 5.4e-4 (bound 1e-3). Two mutations caught: stored scales
  0.3% too large (only by the tight bound: 8-9e-6) and a 32-row group stored in reverse (everywhere). test-backend-ops
  passes every claimed case (MUL_MAT 3, ADD 54, MUL 46, RMS_NORM 52, ROPE 165, SET_ROWS 87, GET_ROWS 9, SWIGLU 12).
  Llama-3.2-1B Q4_0 on the simulated device against the Mac's CPU (2 x 512 tokens of plan.md): perplexity +0.24%,
  mean KLD 0.0022, same top token 96.3%; the zero-point route +1.48%, 0.0565, 84.7%, close to the K3's 0.0548, so the
  reference kernels reproduce the IME's arithmetic here.
- Predicted on the K3 (estimates): Llama-3.2-1B Q4_0 against the CPU within 0.5% in perplexity, mean KLD 0.002-0.003
  (the test file gave 0.0026); Qwen3-0.6B and 4B as their test files with the Q4_1 layers in Q8_0 (the provider's
  weights for those layers are the same bytes): mean KLD about 0.0044 and 0.0026; tg128 1-2% slower on Llama-3.2-1B
  and Qwen3-4B and about 1% on
  Qwen3-0.6B (FLAGOS_SPACEMIT_Q4_1=zero-point in the same session as the comparison); the fit settings' comparisons
  unchanged in kind, as Q4_1 now has Q8_0's thresholds (path B to 11 rows for a down of 8,192 values, as on the Q8_0
  file).

**The fix on the K3 (2026-10-24, build `bd4b655`; `~/kld/fix-*.txt`, `~/kld/fix-bench.csv`).** flagos-check-spacemit passes all checks on the
board on both routes (the default, and `FLAGOS_SPACEMIT_Q4_1=zero-point`). Accuracy against the CPU, 8 x 512 tokens,
`-fa on`, both routes against the same CPU run:

| Q4_0 file | CPU perplexity | `zero-point` (the old default): perplexity, mean KLD, same top token | the fix: perplexity, mean KLD, same top token |
|---|---|---|---|
| Llama-3.2-1B | 13.4182 | +5.45%, 0.0548, 88.5% | +0.20%, 0.00302, 97.2% |
| Qwen3-0.6B | 14.3598 | +1.02%, 0.0148, 93.5% | +0.61%, 0.00472, 95.9% |
| Qwen3-4B | 9.7740 | +0.08%, 0.00335, 96.7% | -0.00%, 0.00249, 97.4% |

- The fix removes 94.5%, 68.2% and 25.8% of the mean KLD, and Llama-3.2-1B's perplexity is now within the acceptance
  script's 1% bound (by this run; the script itself not rerun). `zero-point` gives the old default's outputs:
  perplexity 14.149043, 14.506814 and 9.782019 (step 3's and M2e's values), Qwen3-4B's KLD 0.003354 (M2e's to the
  last digit).
- Qwen3-0.6B on the old route reads +1.02% against this CPU run, above the 1% bound it met at M2e (+0.89% against a CPU
  run of 14.3795; CPU runs here are not bit-reproducible, `build.md` §7); the fix leaves it +0.61%.
- Against the predictions: Llama-3.2-1B's perplexity within 0.5% (met), its KLD 0.00302 just above 0.002-0.003
  (missed by 0.8%); Qwen3-0.6B 0.00472 against about 0.0044 (+7%), Qwen3-4B 0.00249 against about 0.0026 (-4%). The
  predictions took the test files' values, which were measured against the CPU on the requantized weights; the
  reference here is the CPU on the Q4_1 weights. On Llama-3.2-1B the provider's output is the test file's (perplexity
  13.445556 ± 0.897488 in both runs: the same quantizer made both sets of weights) and the CPU run on the Q4_0 file
  is the same in both sessions (13.4182), so the change from 0.0026 to 0.0030 is the reference's alone. On Qwen3 the
  CPU runs also moved between the sessions (0.6B 14.3816 -> 14.3598, 4B 9.7717 -> 9.7740), so its differences are not
  attributed.
- How far the requantization itself moves the result (the Mac, CPU only, Llama-3.2-1B Q4_0, 8 x 512 tokens of
  plan.md; layers 0-1's down rewritten by llama-quantize in F32, which holds the Q4_1 values exactly and runs without
  quantizing the activations, or in Q8_0): against the F32 file, the CPU's Q4_1 path gives mean KLD 0.00109 (same top
  token 97.9%) and the requantized Q8_0 0.00138 (97.7%); the two differ from each other by 0.00139, and the Q4_0 file
  against itself gives 0.000000. So the requantized layers lie a little further from exact arithmetic than the CPU's
  own Q4_1 path, and a KLD against the CPU's Q4_1 path counts the difference between two inexact computations. Option
  (b) (exact Q4_1 weights, the IME's activations) would remove only the requantization's part; its share on the K3 is
  not measured (on Llama-3.2-1B, changing the reference from the requantized weights to the Q4_1 ones moved the KLD by
  0.0004).

Speed in the same session (3 interleaved rounds of llama-bench pp128 and tg128 per route, `-fa auto`, the default
matmul setting), the fix against `zero-point`:

| Q4_0 file | tg128 | pp128 |
|---|---|---|
| Llama-3.2-1B | 32.70 against 33.50, -2.4% (rounds -2.1 to -2.9%) | 370.0 against 374.4, -1.2% |
| Qwen3-0.6B | 54.98 against 55.57, -1.0% (rounds +0.1 to -1.7%) | 649.6 against 651.2, -0.3% |
| Qwen3-4B | 11.03 against 11.26, -2.0% (rounds -1.8 to -2.5%) | 91.6 against 92.1, -0.6% |

- Against the predictions (tg128 1-2% slower on Llama-3.2-1B and Qwen3-4B, about 1% on Qwen3-0.6B, prompt processing
  less): Qwen3-0.6B and prompt processing met, Qwen3-4B at the range's top, Llama-3.2-1B 0.4 points above it (its
  weight bytes per token grow 1.9%; the rest not isolated).
- Qwen3-4B's tg128 with the fix, 11.03, is under M0's target of 11.1, the IME path's speed at M0, which carries the
  same conversion error (the IME path not run in this session). Option (b), exact, would keep the 19-byte layout plus
  2 bytes per block: about +0.25% of the weight bytes instead of +1.9% (an estimate), for new kernel code. The user
  kept the fix as the default after these results (2026-10-24).

**Bigger models: Qwen3-8B and Qwen3-14B (the user's step 4), prepared 2026-10-24.** The same family as the targets, so
only the size changes (step 3 covered another family). Both files downloaded on the board from ModelScope and checked
there by SHA-256; their tensor lists read from their headers (the first 20 MB of each, on the Mac):

- Qwen3-8B Q4_0: `bartowski/Qwen_Qwen3-8B-GGUF`, `Qwen_Qwen3-8B-Q4_0.gguf`, 4,787,332,640 bytes (`c9bd6e55...2b893c`).
  Unsloth's 8B repository has no Q4_0 (it has Q4_1, Q4_K_M, IQ4_NL, Q8_0 and others, by its file list); bartowski's
  file has unsloth's recipe: an imatrix, `ffn_down` of layers 0-3 in Q4_1 (llama-quantize's rule for the first
  n_layer/8 layers with an imatrix, src/llama-quant.cpp:618-622), the head in Q6_K, every other matmul and the token
  embedding in Q4_0. Not chosen: QuantFactory's (no imatrix, so no Q4_1 layers) and Mungert's (the head in Q4_0).
- Qwen3-14B Q4_0: `unsloth/Qwen3-14B-GGUF`, `Qwen3-14B-Q4_0.gguf`, 8,543,001,984 bytes (`009f54ff...9ff78e`): `ffn_down`
  of layers 0-4 in Q4_1, the head in Q6_K, the rest Q4_0 (bartowski's 14B file has the same types).

Checked against the code (nothing run yet): every weight takes an IME layout (flagos-spacemit-weights.cpp:30-47); the
attention heads have 128 values, and 40 query heads over 8 KV heads are claimed, as 8 divides 40
(flagos-spacemit-attn.cpp:415-418). The one-chunk kernels group only rows that share a KV head
(flagos-spacemit-rvv-attn.cpp:1247-1276) and the split over the tiles takes 4 query heads at a time, then the rest one
by one (:1725-1726), so 14B's 5 query heads per KV head run as 4 + 1, which no model run has had (test-backend-ops
covers 1, 4 and 12 per KV head at heads of 128 values, tests/test-backend-ops.cpp:9596-9598); with 40 heads the
split's TCM layout still holds blocks of 64 positions (split_layout_of, flagos-spacemit-rvv-attn.cpp:1057-1095: about
311 KB). The token embedding is Q4_0 in both files (Q6_K on Qwen3-0.6B and 4B, whose head shares it) and stays on the
X100 cores, as the provider's GET_ROWS takes F32 tables only (flagos-spacemit-ops.cpp:68-71); the head is a tensor of
its own, requantized to Q8_0 as before. So 2 graph splits per token are expected.

| | Qwen3-4B (for comparison) | Qwen3-8B | Qwen3-14B |
|---|---|---|---|
| layers; hidden; feed-forward; query / KV heads | 36; 2,560; 9,728; 32 / 8 | 36; 4,096; 12,288; 32 / 8 | 40; 5,120; 17,408; 40 / 8 |
| Q4_1 `ffn_down`, requantized to Q8_0 | layers 0-3 | layers 0-3 | layers 0-4 |
| shares of a layer's weights: q, k + v, o, gate + up, down | 10.4, 5.2, 10.4, 49.4, 24.7% | 8.7, 4.3, 8.7, 52.2, 26.1% | 7.9, 3.2, 7.9, 54.0, 27.0% |
| slabs per tile (N/256), same order | 16, 4, 10, 38, 10 | 16, 4, 16, 48, 16 | 20, 4, 20, 68, 20 |
| path B holds, rows: q, k, v, gate, up / o / down / the Q8_0 downs | 119 / 68 / 19 / 5 | 68 / 68 / 12 / none | 51 / 51 / 4 / none |
| `split`'s 16-row blocks from, rows: q / o, down / gate, up (k, v: 17) | 64 / 40 / 150 | 64 / 64 / 190 | 79 / 79 / 268 |
| weight bytes per generated token | 2.51 GB | 4.67 GB | 8.48 GB |

- New on these shapes (read in the code): a 32-row Q8_0 slab of 12,288 or 17,408 values (417,792 or 591,872 bytes) is
  larger than a tile's TCM, so the requantized downs never take path B under any setting
  (flagos-spacemit-ime.cpp:497-498), and the 1-row path reads them straight from DRAM in 4-slab chunks instead of
  staging them (:211-221; staging was 9.8% faster per token on Qwen3-4B's shapes, build.md §7). Every Q4_0 slab still
  fits (221,184 and 313,344 bytes).
- `fit` and `fit,staged`: path B for q, k, v, o, gate and up up to 68 rows (8B) or 51 (14B), against 119 on 4B, then
  path A; down on path B up to 12 or 4 rows, then path C, then path A from 32 rows; the Q8_0 downs path C, then path A
  from 32. So from 69 or 52 tokens every matrix takes path A, whose last round (the leftover rows on the first tiles)
  costs about two-thirds of a full round when it holds 1 row and nearly a full one with 4 (Qwen3-4B's counters: 68.6
  ps per weight per full round, +46.7 with 1 row, +63.5 with 4).
- `split`: path C, with path A from 113 rows wherever its 16-row blocks give every tile whole rows (q, k, v, o, down);
  gate and up stay on path C in one block of all rows up to 189 rows on 8B and 267 on 14B, so the two settings differ
  there at 160 tokens (8B) and 160-256 (14B), where on 4B they did not. Its blocks leave the tiles uneven for q, o and
  down from 65 rows on 8B and from 79 on 14B (5 blocks for 8 tiles: five tiles take two tasks, three none; the busiest
  has 2.46x the mean work at 65 rows, 2.03x at 79; flagos-spacemit-ime.cpp:504-524), and for k and v from 17 rows;
  from 32 to 78 rows 14B's q, o and down run in one block of all rows, evenly spread.

Sessions, one file at a time, the 8B file first (the user's choice; commands in the chat of 2026-10-24), each a chain
of three: (1) the acceptance script (2 graph splits, flash attention kept under `-fa auto`, perplexity within 1% of the
CPU, KLD against the CPU at 512-token batches and at `-ub 1 -b 512`, which on 14B also runs its 4 + 1 heads in both
attention kernels; llama-bench of the provider, the CPU and the IME path); (2) `EXPS="fit fit-staged" REPS=3` with
`PPODD=2,4,8,12,13,16,24,32,33,36,37,40,44,48,51,52,64,65,68,69,78,79,96,100,128,129` on both files (8B's limits at 12-13
and 68-69, 14B's at 51-52 and `split`'s blocks at 78-79 added) and the runner's 160-320; `rows` and `spacemit` left out,
as proposed and agreed the same day (no bar in the criterion), `fit` kept for the turns at these slab counts; (3)
optional, the fix against `FLAGOS_SPACEMIT_Q4_1=zero-point`, 3 interleaved rounds of pp128 and tg128. Board time, an
estimate from step 3's 47 minutes for 220 runs, scaled by the weights: on 8B about 45 minutes for (1) and 1.5 h for
(2), on 14B about 75 minutes and 2.5-3 h.

Predictions (estimates fixed before the run). Speed from a model of a batch, matrix by matrix: Qwen3-4B's op counters
of the `fit,staged` run (2026-10-22, above) give each path's time per weight at each row count (path A 68.6 ps per
weight per full round plus the last round's; path B by rows and slabs per tile; path C by rows, and where `split`'s
16-row blocks are uneven by the busiest tile's work, at 2.71 ps per weight per row; k and v from 4B's own k and v),
applied to every matrix of a model with its path by the code (`spacemit_mul_mat_gemm`), plus 4B's other steps scaled by
width (1.35x on 8B, 1.8x on 14B; 4-10% of a batch). On 4B path C's time per weight grows a little with the row length
(down 0-20% above gate at 16-37 rows), which the model leaves out, so it errs toward `split`. Checked on runs it was not
fitted to: on Llama-3.2-1B Q4_0 (step 3) the gains against `split` fall within -11 to +4 points of the measured ones
(pp16-26 +16 to +31% against +25.6 to +31.8%, pp27-28 +23 to +24% against +19.5 to +20.0%, pp33-48 +66 to +82%
against +72 to +91%, pp65-100 +88 to +93% against +98 to +103%, pp128 +10% against +11.8%); on 4B itself within -9 to
+1 points.

| `fit,staged` against `split` (the model) | Qwen3-8B | Qwen3-14B |
|---|---|---|
| pp2, pp4, pp8 | +1, +8, +7% | +2, +9, +8% |
| pp12, pp13 | +18, +15% | +15, +17% |
| pp16, pp24 | +19, +28% | +20, +29% |
| pp32 | +48% | +51% |
| pp33-pp51 (7 lengths) | +38 to +52% | +38 to +54% |
| pp52 | +53% | +29% (path A from 52 rows) |
| pp64 | +48% | +48% |
| pp65, pp68, pp69 | +113, +106, +57% | +21, +15, +17% |
| pp78, pp79 | +63, +63% | +28, +64% |
| pp96, pp100 | +86, +62% | +88, +62% |
| pp128, pp129 | +18, +10% | +19, +11% |
| pp160 | +17% | +18% |
| pp192, pp256 | within 1% | +17, +15% |
| pp320, pp512 | within 1% | within 1% |
| `split`'s t/s at pp128 and pp512 | about 50 and 55 | about 27 and 31 |

- So both parts of the criterion hold on both files with at least 10 points to spare after the model's largest
  overestimate (+4 points). The narrowest is 14B at 65-69 tokens (+15 to +21%), where every matrix takes path A (three
  rounds, the last nearly empty) against `split`'s one block of all rows, evenly spread; on 8B it is pp16 (+19%),
  where down runs path C under both.
- tg128 the same in all settings within 0.5% (the 1-row path does not depend on them): 8B 5.6-6.2 t/s, 14B 3.1-3.5
  (the weights' bytes per token against 4B's 2.51 GB at 11.03 t/s, plus 4B's other per-token costs, scaled; the
  requantized downs read from DRAM add about 0.5%).
- Placement and accuracy: 2 graph splits per token, flash attention on the provider under `-fa auto`; perplexity within
  1% of the CPU likely (4B -0.00% with the fix; no model of how it changes with size); mean KLD against the CPU
  0.0015-0.0035 at 512-token batches (4B 0.00249), no higher at `-ub 1 -b 512`.
- Outputs: `fit` and `fit,staged` the same text as `split` and KLD at the floor at `-ub` 128, 64 and 16.
- Against `fit`: slower at 2-8 tokens (as on every model so far), about level at 12-13, 1.5-3.5% faster from 16 tokens
  to 68 (8B) or 51 (14B), within 1% beyond (path A under both).
- Counters: the paths as above; under `fit` the odd tiles lag by about 0.9/R (8B: 22% for k and v, 6% for q and o, 2%
  for gate and up; 14B: 22%, 4.5%, 1.3%), under `fit,staged` by 0-5%.
- The fix (optional (5)): tg128 2.0-3.0% slower than `zero-point` (the requantized downs add 88 MB per token on 8B,
  +1.9%, and 196 MB on 14B, +2.4%, and are read from DRAM), pp128 within 1.5%.

Criterion (fixed before the run), per file, as step 3's: from 16 tokens no tested length more than 1% slower than
`split`, every tested length from 16 to 100 at least 5% faster, outputs identical; and the acceptance script's checks
pass. If both files meet it, the user's plan of 2026-10-22 is complete and the default change to `fit,staged` with its
acceptance runs comes next (proposed); if not, the change waits and the failing lengths are examined first. The lengths
below 16 tokens and the comparisons with `fit` are reported, with no bar.

**Qwen3-8B on the K3: the acceptance script (2026-10-24, the board's build of `bd4b655`, as for the fix's runs;
`~/acc-q8b.log`, its logs copied to `~/acc-q8b-logs`).** All 34 checks pass:
- Placement: 2 graph splits per token (1 with the provider disabled), flash attention kept on the provider under `-fa
  auto` (C5a), 4,453.8 MiB of weights on the provider (4.67 GB, the table's bytes per token) and a compute buffer of
  304.8 MiB. test-backend-ops passes every claimed case (1,742, of which `FLASH_ATTN_EXT` 1,324), the reference kernels
  13/13; no TCM block left held.
- Accuracy against the CPU (8 x 512 tokens): perplexity 8.2002 against 8.2121 (-0.14%), with `-fa on` and with `-fa
  auto`; mean KLD 0.003116 ± 0.000117 at 512-token batches and 0.002137 ± 0.000079 at `-ub 1 -b 512`. The greedy text
  differs from the CPU's after 47 characters, as on Llama-3.2-1B (reported, not checked: the IME kernels quantize the
  activations differently, spacemit_check.py:282).
- llama-bench, pp128 and tg128 (`-ub 128 -fa 1`, 3 repetitions, the default matmul setting `split`): the provider 50.59
  and 6.05 t/s, the IME path 45.07 and 6.08, the CPU 11.24 and 3.24. So against the IME path prompt processing +12.2%
  (Qwen3-4B at M2e's final run, before the fix: +16.6%) and generation -0.5%, in one session.
- Against the predictions: all met. 2 graph splits and flash attention on the provider; perplexity within the 1% bound;
  the KLD inside 0.0015-0.0035 at 512-token batches (Qwen3-4B with the fix 0.00249, against another CPU run) and lower
  at `-ub 1 -b 512`; tg128 6.05 inside 5.6-6.2; `split`'s pp128 50.59 against about 50. Generation reads 4.67 GB of
  weights per token at 6.05 t/s, 28.3 GB/s, against 27.7 on Qwen3-4B (2.51 GB at 11.03 t/s, the fix's session):
  computed from the speeds; how a token's time divides between the weights and the rest is not measured here (the
  runner's op counters will show it).
- The script's fixed benchmarks run Qwen3-4B's shapes and read as in step 3's run (2026-10-23, `~/acc-l1b.log`): the
  Q4_0 matmul at 128 rows 9,643 against 9,668 µs, one-token attention at n_kv 4096 1,115 against 1,131 µs, so the board
  ran as then; the gemv per Qwen3-4B token 89.3 ms against 88.0, for 2,507 MB of weights against 2,460 (the fix's Q8_0
  downs).

**Qwen3-8B on the K3: the benchmark (2026-10-24, build `bd4b655`; `EXPS="fit fit-staged" REPS=3` with the lengths
above; `~/fx-q8b.log`, its logs in `~/fx/q8b` on the board).** Outputs identical: the same text in the three settings,
and against `split` mean KLD 0.000000, maximum 0.000052 (the floor; step 3's Q8_0 file 0.000048) and the same top token
100% at `-ub` 128, 64 and 16. The counters move a result by at most 0.9% (`fit,staged`'s pp24), apart from `fit`'s
pp2 (-2.7%). t/s, mean of 3 runs
(standard deviations at most 0.37), the model's predictions in brackets:

| tokens | `split` | `fit,staged` | against `split` (predicted) | against `fit` |
|---|---|---|---|---|
| 2, 4, 8 | 9.58, 16.53, 30.37 | 10.11, 21.28, 35.45 | +5.5, +28.7, +16.7% (+1, +8, +7) | -8.6, -5.0, -5.1% |
| 12, 13 | 34.36, 31.67 | 43.66, 36.93 | +27.1, +16.6% (+18, +15) | +1.2, +1.3% |
| 16, 24 | 35.76, 35.32 | 42.99, 46.80 | +20.2, +32.5% (+19, +28) | +1.6, +3.0% |
| 32 | 36.24 | 60.41 | +66.7% (+48) | +3.4% |
| 33, 36, 37, 40, 44, 48 | 32.67-35.43 | 50.44-57.10 | +49.8 to +61.2% (+38 to +50) | +2.5 to +2.9% |
| 51, 52 | 34.00, 36.51 | 48.27, 58.97 | +42.0, +61.5% (+52, +53) | +2.5, +2.9% |
| 64 | 43.67 | 65.39 | +49.7% (+48) | +3.0% |
| 65, 68, 69 | 26.34, 27.48, 27.71 | 59.24, 59.75, 45.67 | +124.9, +117.4, +64.8% (+113, +106, +57) | +2.8, +2.8, -0.1% |
| 78, 79 | 29.85, 29.90 | 44.39, 38.46 | +48.7, +28.6% (+63, +63) | 0.0, 0.0% |
| 96, 100 | 31.35, 29.35 | 60.40, 49.17 | +92.7, +67.5% (+86, +62) | 0.0, -0.0% |
| 128, 129, 160 | 50.65, 47.91, 51.36 | 59.19, 52.32, 59.93 | +16.9, +9.2, +16.7% (+18, +10, +17) | -0.1, -0.1, +0.2% |
| 192, 256, 320, 512 | 59.04, 58.62, 58.17, 55.88 | 58.89, 58.61, 58.32, 56.29 | -0.3, -0.0, +0.3, +0.7% (within 1) | -0.6 to +0.3% |
| tg128 | 6.05 | 6.04 | -0.1% (within 0.5) | -0.1% |

- The criterion is met on this file: from 16 tokens at worst -0.25% against `split` (pp192), every tested length from
  16 to 100 at least +20.2% (pp16, the narrowest predicted too), outputs identical, the acceptance script's checks
  passed (above).
- Against the predictions: from 16 to 100 tokens the gains lie between -34.7 and +19.1 points of the model's, above it
  at 16 of the 19 lengths; from 128 tokens within 1 point. Above: `split` is slower than modelled, as down's path C on
  rows of 12,288 values costs 5-45% more per weight than the model's (taken from gate's rows of 2,560 on 4B; the row
  length left out, as stated), and the 4 requantized Q8_0 downs, which the model costed per weight as Q4_0, cost
  1.65-1.71x that on path A (without leftover rows) and 1.6-4.4x on path C. Below, at pp51 (-10 points), pp78 (-14)
  and pp79 (-35): path A's leftover rows, next. Against `fit`, the odd tiles and tg128: as predicted (the odd tiles'
  excess under `fit` on path B: k and v +8 to +22%, q and o +2 to +6%, gate and up +0.7 to +2.0%, the predicted 22, 6
  and 2% reached from about 32 rows; under `fit,staged` -0.6 to +5.4%). The paths'
  boundaries show in the counters where the code puts them: down from path B to path C at 13 rows, the other matrices
  to path A at 69, the Q8_0 downs on path A from 32.
- Found, path A's leftover rows (read in the code, matching the counters): `gemm_rows` multiplies 4 rows per kernel
  call while 4 remain, then one row per call (flagos-spacemit-ime.cpp:134-143), and path A gives each tile 4
  consecutive rows per round (:483-490), so the batch's last m mod 4 rows all go to one tile, which runs them one call
  at a time over every slab of the weights. On this run's counters path A's last round costs, per weight (a full
  round 66-67): 45.7 when it holds one tile with 1 row (pp65's down), 61.7-63.4 with tiles of 4 rows (pp68, pp69), 96.4
  with a tile of 2 rows (pp78) and 133.6-138.0 with a tile of 3 (pp51's down, pp79): r leftover rows cost about r
  one-row passes (whether the extra passes read DRAM or cache is not isolated). The model took 52 and 58 for 2 and 3
  rows, interpolated: no earlier run had a length with 2 or 3 leftover rows on path A (the 11 runner logs before this
  one, from 32 rows all 4q or 4q+1; the shape benches' path-A cells likewise, and the grid's at 32, 40 and 48 rows).
  The 4q+2 and 4q+3 batches tested before (8-31 tokens, the grid's 6, 10 and 14 rows) ran on path B or C, where
  every tile runs the one-row calls on its own columns: they cost 5-16% there (step 2, "the remainder rows").
  `split` pays the same where it takes path A (from 113 rows). Path B
  also runs the leftover rows one call each, on every tile (pp51's path-B matrices 8-9% above the model), which
  dealing them out cannot change.
- Inherited from SpacemiT's code, and not fixed upstream (checked 2026-10-24): the provider's path A is a port of the
  in-tree IME path's (ggml-cpu/spacemit/ime.cpp:405-434), which ggml-spacemit carries unchanged; the loop is the same,
  whitespace aside, in ggml-org master (`df9beac`) and in both files on 9 of spacemit-com's 10 branches (the tenth,
  `add-spacemit-backend`, predates path A), and the dense kernels still take 4 rows or 1 per call (the 2-row kernels
  are MoE's, `moe_m2_*`; ime2_kernels.cpp is master's byte for byte). No pull request or issue in either repository
  addresses it (their lists, 2026-10-24); upstream's one fix in this scheduler, #16629 (2025), corrected an
  out-of-bounds task index. SpacemiT's open #51 adds an f16 Q4_0 path for prompt processing to ggml-spacemit whose
  kernel zero-pads the last of its 16-row tiles, so it would not run rows one call each (by its diff; not merged, not
  measured). Correct results in every case: the cost is time, where a call takes path A and m mod 4 is 2 or 3, in
  SpacemiT's rule and `split` from 113 rows (gate and up from 150, 190 or 268 rows on 4B, 8B, 14B), under `fit` and
  `fit,staged` from 32 rows wherever path B's TCM cannot hold the batch (down from 32 on all three; q, k, v, o, gate
  and up from 69 on 8B and 52 on 14B; on 4B o from 69, the rest from 120).
- The model with this correction (r one-row passes; 8B's measured ratios for down's path C and the Q8_0 downs, so
  in-sample for those): the gains at the tested lengths from 16 to 100 within -0.2 to +8.1 points of the measured,
  from 128 within ±0.7. For lengths not tested it predicts dips at 4q+2 and 4q+3 tokens wherever `fit,staged` takes
  path A and `split` path C (estimates): on 8B from 113 to 189 tokens, gate and up only: pp131 -2.0%, pp135 -0.8%,
  pp163 -0.2% (part 1 allows -1%); on 4B pp131 -1.8%; on 14B, every matrix on path A from 52 tokens against `split`'s
  balanced path C up to 78: pp55 -6.3%, pp59 -2.8%, pp67 -14.7%, pp71 -10.8%, pp75 -6.9%, and under +5% at pp63, pp66
  and pp70, none of them in the planned 14B lengths; pp131 -2.1%.
- Proposed (the user's decision): the 14B benchmark with those lengths added (54, 55, 58, 59, 62, 63, 66, 67, 70, 71,
  74, 75, 130, 131, 135); and path A dealing the leftover rows one per tile idle in its last round (each row keeps its
  bytes and its one-row kernel call, so the outputs stay identical by the code; when tile 7 holds them, the rest go to
  tiles 0 onwards after their group). Estimated by the corrected model: `fit,staged` +16% at pp51, +13% at pp78, +34% at
  pp79 on 8B; `split` +9 to +11% at pp131-163 and `fit,staged` +20 to +26% there; on 14B `fit,staged` at least +11%
  against `split` at every length listed above.
- Generation (tg128; E3c's counters and the op counters on the defaults): 165.9 ms per token, the matmuls 159.3 ms
  (4.67 GB at 29.3 GB/s), the head 22.5 ms of it, the 4 requantized downs 7.8 ms (1.96 ms each against 1.00 for a Q4_0
  down: 1.89x the bytes, read from DRAM), attention 1.6 ms; the AI cores idle 1.0% of a token.
- The fix against `FLAGOS_SPACEMIT_Q4_1=zero-point` (3 interleaved rounds, `~/kld/step4-fix-bench.csv`): tg128 6.038
  against 6.182, -2.34% (rounds -2.18 to -2.50%), pp128 50.67 against 50.85, -0.35%; predicted 2.0-3.0% and within
  1.5%: met. With `zero-point` the provider's tg128 would be above the IME path's 6.08 (acceptance run), with the fix
  0.5% under it.

**Path A's leftover rows dealt to the idle tiles (the user's decision, 2026-10-24; implemented and checked on the Mac
the same day, not yet run on the K3).**
- What changed (providers/spacemit only): in path A (`spacemit_mul_mat_gemm`, flagos-spacemit-ime.cpp) the groups of 4
  rows go to the tiles as before, and the batch's last m % 4 rows, which the quantize step stores one by one, go one
  per tile among those without a group in the last round (`path_a_tail_tile`): tiles g to 7, g the tile after the last
  group, which held them all; a second row each when fewer are idle than rows; a third row, when only tile 7 is idle
  (m % 32 = 31), to tile 0 after its group. By 8B's counters two one-row passes (1.44 rounds) take less time than a
  group and a row (0.93 + 0.68), hence two per idle tile before a busy one. Each row keeps its bytes and its one-row
  kernel call, whose result does not depend on how many rows the call was given (the 1-row kernels do not read
  `count_m`, flagos-spacemit-ime-kernels.cpp), so the output stays the same. ime.cpp's assignment stays for
  comparisons behind `FLAGOS_SPACEMIT_TAIL=one-tile` (test-only, read at every call). `split` takes path A from 113
  rows, so it changes there too.
- Why this way: the smallest change that removes the measured cost (r one-row passes on one tile) with the outputs
  unchanged. Not chosen: the last round balanced by columns over all tiles (also the dips past each multiple of 32,
  but a larger change, and on 8B's and 14B's downs the last round's rows do not fit in a tile's TCM at once); padding
  the leftover rows to a group of 4 (changes Q4_0's outputs: its 4-row quantizer shares a scale across the rows); a 2-
  or 3-row kernel (none for dense matmuls, new kernel code).
- Checked on the Mac (host build, simulated device): flagos-check-spacemit passes all checks; its path check, now 32
  cases (11 new with 2 or 3 leftover rows on path A, among them the tile-7 cases at 62 and 63 rows and the split's
  path A at 115) with a seventh setting, `rows` with `FLAGOS_SPACEMIT_TAIL=one-tile`, 64/64 identical with the tiles in
  turn and on threads, and every tile's count of leftover rows (test-only, `spacemit_mm_tail_take`) as the rule gives.
  Three mutations caught: the last leftover row skipped (32 failures, by the outputs), rows dealt one tile off (26, by
  the counts alone, the outputs being identical), m % 32 = 31's third row on tile 7 (4, at 63 rows). ThreadSanitizer
  clean. test-backend-ops unchanged (MUL_MAT 3, ADD 54, MUL 46, RMS_NORM 52, ROPE 165, SET_ROWS 87, GET_ROWS 9, SWIGLU
  12). Not timed on the Mac: its reference kernels do not cost what the IME's do per row.
- For the K3: `--bench-mm` has the seventh setting too and rows of 34, 35, 51, 63, 79, 115 and 131; the runner has
  `EXPS` `tail` (`split` with one-tile) and `fs-tail` (`fit,staged` with one-tile), logits checked at `-ub 79` and
  `-ub 131` (`KLD_SETS="ub79 ub131"` keeps it to those with `fit-staged` in the same session).
- Predictions (estimates fixed before the run; the costs of a last round from 8B's counters, in full rounds: a group
  0.93, a row alone 0.68, two rows on a tile 1.44, three 2.05): outputs identical (the checks on the board; KLD at the
  floor at `-ub 79` and `-ub 131`); lengths of 4q and 4q+1 tokens, of 32k+30 (the tile-7 case with 2 rows), tg128
  and pp512 unchanged within 1%. `--bench-mm`, gate, up and down on path A, one-tile's time over the dealt one's: 34
  rows 1.45x, 35 1.82x, 51 1.58x, 63 1.17x, 79 1.38x, 115 1.28x, 131 1.29x (the rounds, today against dealt: 2.44
  against 1.68, 3.05 against 1.68, 3.05 against 1.93, 3.05 against 2.61, 4.05 against 2.93, 5.05 against 3.93, 6.05
  against 4.68). The models, `fit,staged` against `fs-tail` (the corrected model above): 8B pp51 +16%, pp78 +13%, pp79
  +34%, pp131 +26%, pp135 +20%, pp163 +21%; 14B pp55 +52%, pp59 +16%, pp63 +16%, pp67 +47%, pp71 +35%, pp75 +35%, pp79
  +35%; `split` against `tail` from 113 tokens: 8B pp131 +11%, pp135 +9%, pp163 +9%, 14B pp131 +11%; and `fit,staged`
  then at least 11% faster than `split` at 14B's lengths listed in step 4's corrections (pp55-75, pp131; the
  least at pp131, +11.2%, and pp59, +12.3%). Corrected 2026-10-25, before the run: that model deals the rows by the
  first proposal, which differs from the rule as implemented where m % 32 is 27 or 30 (a group and a row on tile 0,
  110.2 ps per weight, against two rows on one idle tile, 93.4); with the implemented rule 14B's pp59 is +26.5% (not
  +16%) and +23.0% against `split` (not +12.3%), so the least are pp131 and pp63 (+16.1%). No other length or bench
  row listed is 27 or 30 mod 32.
- Proposed rule (fixed before the run): the dealing stays if the outputs are identical, no tested length is more than
  1% slower than with one-tile under the same matmul setting, and tg128 and pp512 are within 1%; otherwise it is
  examined first.

**Qwen3-14B on the K3: the acceptance script (2026-10-24, build `bd4b655`; `~/acc-q14b.log`, its logs copied to
`~/acc-q14b-logs`).** All 34 checks pass:
- Placement: 2 graph splits per token (1 with the provider disabled), flash attention kept on the provider under `-fa
  auto` (C5a), 8,089.9 MiB of weights on the provider (8.48 GB) and a compute buffer of 306.8 MiB. test-backend-ops
  passes every claimed case (1,742, of which `FLASH_ATTN_EXT` 1,324), the reference kernels 13/13; no TCM block left
  held.
- Accuracy against the CPU (8 x 512 tokens): perplexity 7.757 against 7.7512 (+0.07%), with `-fa on` and with `-fa
  auto`; mean KLD 0.004728 ± 0.000549 at 512-token batches and 0.004029 ± 0.000505 at `-ub 1 -b 512`, the first runs
  of a model with 5 query heads per KV head (the one-chunk kernels take them as 4 + 1). The greedy text is the CPU's.
- llama-bench, pp128 and tg128 (`-ub 128 -fa 1`, 3 repetitions, `split`): the provider 27.48 and 3.37 t/s, the IME
  path 24.75 and 3.45, the CPU 5.73 and 1.89; so against the IME path prompt processing +11.0% and generation -2.3%,
  in one session.
- Against the predictions: 2 graph splits and flash attention on the provider, perplexity within the 1% bound, tg128
  3.37 inside 3.1-3.5 and `split`'s pp128 27.48 against about 27: met. Missed: the mean KLD at 512-token batches,
  0.0047 against the predicted 0.0015-0.0035 (8B 0.0031, 4B with the fix 0.00249, each against its own CPU run); at
  `-ub 1 -b 512` lower (0.0040), as predicted. The script reports the KLD with no bound (a KLD check fails only when
  no value comes out, spacemit_check.py:375-379), so the check passed. Cause not isolated; the tests that would
  settle it: the IME path's KLD on this file against the same CPU logits (on Qwen3-0.6B the two matched, the IME
  approach's own), and test files with the 5 requantized downs or the head in Q8_0, as for Llama-3.2-1B.
- Generation reads 8.48 GB of weights per token at 3.37 t/s, 28.6 GB/s (8B 28.3, 4B 27.7; computed from the speeds).
  The script's fixed benchmarks (Qwen3-4B's shapes) read as in 8B's session: the Q4_0 matmul at 128 rows 9,631
  against 9,643 µs, one-token attention at n_kv 4096 1,111 against 1,115 µs, the gemv per Qwen3-4B token 88.6 against
  89.3 ms; the 16M-float `ADD` 19.86 against 16.11 GB/s (not examined).

**Qwen3-14B on the K3: the benchmark (2026-10-24, build `bd4b655`; `EXPS="fit fit-staged" REPS=3`; `~/fx-q14b.log`,
its logs in `~/fx/q14b` on the board).** The run took step 4's first list of lengths (the 26 in the board's
`info.txt`), not the one with the lengths added after 8B's run (54, 55, 58, 59, 62, 63, 66, 67, 70, 71, 74, 75, 130,
131, 135): those are not tested here; the dealing's 14B session has them under both assignments (`tail` and `fs-tail`
run today's). Outputs identical: the same text in the three settings, and against `split` mean KLD 0.000000, maximum
0.000056 (the floor) and the same top token 100% at `-ub` 128, 64 and 16. The counters move a result by at most 1.0%
from 12 tokens (`ops`' pp12), by up to 3.7% at 2-8 tokens (pp2 -2.0 to -3.7% in the three settings, pp4 +2.6 and
+2.8% under `ops` and `fit,staged`). t/s, mean of 3 runs (standard deviations at most 0.36), the model's predictions
in brackets:

| tokens | `split` | `fit,staged` | against `split` (predicted) | against `fit` |
|---|---|---|---|---|
| 2, 4, 8 | 4.81, 10.72, 15.63 | 5.85, 11.75, 18.10 | +21.8, +9.6, +15.8% (+2, +9, +8) | -4.9, -8.6, +0.7% |
| 12, 13 | 17.08, 15.03 | 20.25, 18.15 | +18.6, +20.8% (+15, +17) | +0.9, +1.3% |
| 16, 24 | 16.99, 16.94 | 21.27, 22.06 | +25.2, +30.2% (+20, +29) | +1.3, +1.7% |
| 32 | 17.15 | 33.53 | +95.5% (+51) | +2.9% |
| 33, 36, 37, 40, 44, 48 | 15.61-16.92 | 27.67-31.35 | +70.6 to +85.3% (+38 to +51) | +2.1 to +2.8% |
| 51, 52 | 15.82, 17.40 | 26.14, 27.10 | +65.2, +55.8% (+54, +29) | +2.2, -0.0% |
| 64 | 18.08 | 33.02 | +82.6% (+48) | 0.0% |
| 65, 68, 69 | 17.06, 17.61, 17.19 | 25.32, 24.46, 24.59 | +48.4, +38.9, +43.1% (+21, +15, +17) | -0.1, +0.1, 0.0% |
| 78, 79 | 16.97, 15.77 | 23.81, 20.56 | +40.3, +30.4% (+28, +64) | 0.0, 0.0% |
| 96, 100 | 16.68, 15.68 | 32.87, 26.51 | +97.1, +69.0% (+88, +62) | -0.0, +0.1% |
| 128, 129, 160 | 27.54, 25.76, 27.52 | 32.81, 28.48, 32.75 | +19.1, +10.6, +19.0% (+19, +11, +18) | 0.0, -0.0, +0.3% |
| 192, 256, 320, 512 | 27.53, 27.30, 32.27, 31.42 | 32.70, 32.28, 32.28, 31.44 | +18.8, +18.3, +0.0, +0.1% (+17, +15, within 1) | -0.1 to +0.0% |
| tg128 | 3.37 | 3.37 | +0.0% (within 0.5) | +0.1% |

- The criterion is met at this run's lengths: from 16 tokens never slower than `split` (the least pp320, +0.0%), every
  tested length from 16 to 100 at least +25.2% (pp16), outputs identical, the acceptance script's checks passed
  (above). With 8B's run both files meet it at the lengths tested; the lengths where the corrected model put dips were
  not tested on 14B (estimates in the last bullet).
- Against the predictions: from 16 to 100 tokens the gains lie between -33.5 (pp79) and +44.5 points (pp32) of the
  model's, above it at 18 of the 19 lengths; from 128 tokens within -0.1 to +3.1 points. Above: `split` is slower than
  modelled (by 10-23% at 52-78 tokens), as on 8B but more: by the op counters its down on path C on rows of 17,408
  values costs 1.5-2.0x per weight the model's at 8-78 rows (8B's rows of 12,288: 1.05-1.45x), and the 5 requantized
  downs, costed per weight as Q4_0, 3.1-4.2x there; the other matrices' path C within 0.74-1.39x (k, with `split`'s
  uneven blocks, the widest). `fit,staged` lies within -11 to +4% of the corrected model (its down on path C too at
  8-24 rows). Below, at pp79: path A's leftover rows (3, on tile 3). Against `fit`: slower at pp2 and pp4 (-4.9,
  -8.6%), not at pp8 (+0.7%, the first model where it is not), about level at 12-13, +1.3 to +2.9% from 16 to 51
  tokens (predicted 1.5-3.5%), within 0.3% from 52 (path A under both): as predicted but pp8 and pp16.
- The odd tiles on path B (predicted under `fit` 22% for k and v, 4.5% for q and o, 1.3% for gate and up; under
  `fit,staged` 0-5%): under `fit` k and v +8.5 to +21.7%, rising with the rows (21.7% at 51), q and o +1.8 to +4.6%,
  gate and up +0.5 to +1.4%; under `fit,staged` -0.2 to +5.5% (k at 16 rows the highest). The paths' boundaries show
  in the counters where the code puts them: down on path B at 2-4 rows, path C at 8-24, path A from 32; the
  requantized downs on path C up to 24 rows (never path B) and path A from 32; the other matrices on path B up to 51
  rows, path A from 52. Path A's last round as on 8B: at 65 rows tile 0's leftover row makes the even tiles the slower
  (-8%), at 78 and 79 tile 3's 2 and 3 rows the odd ones (+4.0 to +6.2%, +10.7 to +13.1%).
- Generation (tg128; E3c's counters and the op counters on the defaults): 296.0 ms per token in graph compute, the
  matmuls 288.9 ms (8.48 GB at 29.4 GB/s; 8B 29.3), the head 28.3 ms of it, the 5 requantized downs 17.1 ms (3.43 ms
  each against 1.79 for a Q4_0 down: 1.91x, the bytes 1.89x, read from DRAM), attention 2.5 ms; imbalance 6.9 ms, the
  AI cores idle 0.6% of a token. tg128 3.37 in all three settings, as predicted.
- The fix against `FLAGOS_SPACEMIT_Q4_1=zero-point` (3 interleaved rounds, `~/kld/step4-fix-bench.csv`): tg128 3.371
  against 3.463, -2.64% (rounds -2.49 to -2.80%), pp128 27.51 against 27.67, -0.59%; predicted 2.0-3.0% and within
  1.5%: met. With `zero-point` the provider's tg128 would be 0.4% above the IME path's 3.45 (acceptance run), with
  the fix 2.3% under it.
- The untested lengths re-estimated (estimates, made after this run): the corrected model's `split` and `fit,staged`,
  each scaled by its measured-over-model ratio at the nearest tested lengths (`split` 0.82-0.97 at 52-79 tokens,
  1.02-1.03 at 128-160; `fit,staged` 1.00-1.03). With today's assignment `fit,staged` against `split`: pp54 +34.0%,
  pp55 +7.8%, pp58 +43.1%, pp59 +15.0%, pp62 +52.2%, pp63 +22.1%, pp66 +20.8%, pp67 +2.1%, pp70 +27.3%, pp71 +8.3%,
  pp74 +33.7%, pp75 +13.8%, pp130 +3.5%, pp131 -2.0%, pp135 -0.6%, so part 2 would fail at pp67 and part 1 at pp131
  (8B's pp131 estimated -2.0% too). With the leftover rows dealt by the implemented rule, `split` dealt as well from
  113 tokens: at least +41.1% at 54-79 tokens (pp63) and +10.1 to +11.3% at 130-135. The dealing's 14B session
  measures both.

**Path A's leftover rows dealt, on the K3: Qwen3-8B (2026-10-24, build `682c676`; `EXPS="fit-staged tail fs-tail"
KLD_SETS="ub79 ub131" REPS=3`, step 4's 27 lengths for 8B and 160-320; `~/fx-q8b-tail.log`, `~/bench-mm-tail.out`, the
logs in `~/fx/q8b-tail` on the board).** `ref` (`split`) and `fit-staged` deal the rows, `tail` and `fs-tail` are the
same settings with `FLAGOS_SPACEMIT_TAIL=one-tile`. Outputs identical: the same text in the four settings; against
`ref` mean KLD 0.000000, maximum 0.000052 (the floor) and the same top token 100% at `-ub 79` and `-ub 131` in the
three others; on the board the path check 32/32 identical, each tile's leftover rows as the rule gives. The counters
move a result by at most 0.85% (`ops`' pp68). t/s, mean of 3 runs (standard deviations at most 0.44), dealt against
one-tile under the same setting, the predictions in brackets (`split` takes path A from 113 tokens):

| tokens (m % 32) | `fit,staged` dealt, one-tile | change | `split` dealt, one-tile | change |
|---|---|---|---|---|
| 34, 35 (2, 3) | 49.80, 44.44; 49.02, 40.11 | +12.1, +22.2% | (path A not taken) | |
| 51, 55 (19, 23) | 54.78, 48.22; 56.13, 49.75 | +13.6% (+16), +12.8% | | |
| 62 (30) | 55.26, 56.02 | **-1.4%** (within 1) | | |
| 63 (31) | 53.62, 51.82 | +3.5% | | |
| 78, 79 (14, 15) | 49.00, 44.33; 49.47, 38.47 | +10.5% (+13), +28.6% (+34) | | |
| 115, 119 (19, 23) | 54.55, 44.70; 55.64, 45.93 | +22.0, +21.1% | 48.19, 43.88; 48.47, 44.50 | +9.8, +8.9% |
| 123 (27) | 51.44, 46.89 | +9.7% | 46.80, 44.84 | +4.4% |
| 127 (31) | 50.63, 47.73 | +6.1% | 46.46, 45.27 | +2.6% |
| 130, 131 (2, 3) | 52.54, 46.96; 52.63, 42.74 | +11.9%, +23.1% (+26) | 47.77, 45.26; 47.47, 42.97 | +5.5%, +10.5% (+11) |
| 135, 163 (7, 3) | 51.19, 43.76; 53.82, 45.21 | +17.0% (+20), +19.0% (+21) | 46.95, 43.56; 48.20, 44.43 | +7.8% (+9), +8.5% (+9) |
| the rest (0 or 1 row left over, or `split` off path A) | | -0.26 to +0.39% (within 1) | | -0.40 to +0.64% |
| tg128, pp512 | 6.08, 6.08; 56.35, 56.21 | +0.0, +0.25% (within 1) | 6.08, 6.07; 56.30, 56.20 | +0.2, +0.2% |

- The rule is not met at one length: pp62 under `fit,staged`, -1.36% (-0.98% under the counters, whose totals give
  -0.99%). Everywhere else the dealing is within 0.4% of one-tile where it gives the same calls, and 2.6-28.6% faster
  where it moves rows; tg128 and pp512 within 0.3%. So by the rule the dealing is examined first; what follows is that
  examination.
- The cause (read in the code, measured by the counters). At m % 32 = 30 the rule puts both leftover rows on tile 7,
  the only idle tile, as ime.cpp does. But the dealt loop multiplies each row in a call of its own (`multiply(m4 + j,
  1)`, flagos-spacemit-ime.cpp:549 at `682c676`), so tile 7 streams its weights twice. With one-tile, tile 7 multiplies
  both rows in one call (:537 there): two one-row kernel calls per block of columns, the weights streamed once.
  That the second call finds the weights in cache is an inference. At 62 rows only the downs take path A (the other
  matrices take path B, unchanged within 0.1%): the Q4_0 downs take 266.5 ms per batch dealt against 255.5 with
  one-tile (+4.3%), the requantized ones 63.0 against 61.4 (+2.6%). The same two calls happen at m % 32 = 27 (tile 6,
  rows 0 and 2) and 31 (tile 7, rows 0 and 1), where the dealing still gains, since one-tile puts 3 rows on one tile
  there. The prediction assumed two rows on a tile cost 1.44 rounds however they are called.
- Path A's last round in full rounds, by the counters. These are `fit,staged`'s Q4_0 downs, at a round of 3,331 µs per
  down (the median at 32-512 rows); the requantized downs and, from 78 rows, the other matrices show the same
  pattern.
  - A row alone: 0.69. Groups alone: 0.92-0.93. A group and a row on another tile: 0.94-0.95.
  - Rows dealt one per idle tile: 0.69-0.74 with no group in the round, 1.02-1.11 with groups.
  - One tile's 2 rows in one call: 1.34 alone, 1.45-1.47 with groups. One tile's 3 rows in one call: 1.99-2.11.
  - Tile 7's 2 rows in two calls, with 7 groups: 1.58, against 1.47 in one call.
  - Dealt at m % 32 = 27: 1.55; at m % 32 = 31: 1.79 (one-tile 2.07 and 2.11).
- Against the predictions:
  - `--bench-mm` (Qwen3-4B's shapes), one-tile's time over the dealt one's for gate, up and down: 1.35-1.40 at 34
    rows (predicted 1.45), 1.67-1.69 at 35 (1.82), 1.45-1.48 at 51 (1.58), 1.11-1.15 at 63 (1.17), 1.33 at 79
    (1.38), 1.24-1.26 at 115 (1.28), 1.25 at 131 (1.29). The other matrices are alike; the bench's single timings
    differ by up to 6.8% where the calls are the same.
  - The model's lengths come in 0.5-5.4 points under their predictions (table).
  - Both are under because the dealt rows share the round with the groups: a last round with both costs 1.02-1.11
    rounds, against the model's 0.93 (a group's); with no group it costs 0.69-0.74 (the model's 0.68). That the rows'
    passes and the groups share DRAM bandwidth is an inference.
- Step 3's criterion at this session's lengths (`fit,staged` against `split`) is met under the dealing: from 32
  tokens at worst -0.22% (pp256), every length from 32 to 100 at least +49.1% (pp64), and 115-163 tokens +9.0 to
  +14.8%. It is met with one-tile too: at worst -0.54% at pp131 (the corrected model had put it at -2.0%), and -0.46%
  at pp192.
- Generation: tg128 6.07-6.08 in all four settings; 165.1-165.3 ms per token by E3c's counters.
- Proposed (the user's decision): a tile's leftover rows multiplied in one call, and a tile with two rows given
  consecutive ones. At m % 32 = 27, tile 6 would take rows 0-1 and tile 7 row 2 (today tile 6 has rows 0 and 2).
  - What stays the same: the outputs, by the code (each row the same one-row kernel call on the same bytes), and
    each tile's count of leftover rows. At m % 32 = 30 the calls become one-tile's.
  - The check would also count each tile's calls with leftover rows, so a return to one call per row fails it.
  - Predicted on 8B (estimates from the last-round costs above, with the m-row matmuls 84-89% of a batch by the
    counters): pp62 level with one-tile (the same calls); pp123 about +1.6% (1.55 -> about 1.47 rounds); pp63 +1 to
    +2% and pp127 +1.7 to +3.4% (1.79 -> 1.61-1.70, as tile 0's group and row become the slowest). Nothing else
    changes. Corrected the same day, before any run: at 63 and 127 rows tile 0, with its group and a row, is already
    the slowest. By the counters the odd tiles are never the slowest there, while at 62 rows tile 7 always is and at
    123 rows an even tile, tile 6 with its two calls. So the fix leaves that round as it is: pp63 and pp127 unchanged.
    The fix's predictions are in its block below.
- For 14B's session on this build, re-estimated from 8B's last-round costs before its results (every matrix takes
  path A at 59-63 rows on 14B): pp62 about -4% against one-tile, pp59 about +19% (the dealing block's corrected
  +26.5%), pp63 about +11% (+16%). The other predicted lengths may come in a few points under their predictions, as
  on 8B. These are for today's dealing; 14B's session runs on the fix instead (its block below).

**A tile's leftover rows in one call (the user's decision, 2026-10-25; implemented and checked on the Mac the same
day, run on the K3 the same day: the next block).**
- What changed (providers/spacemit only):
  - `path_a_tail_tile` takes the number of leftover rows. When fewer tiles are idle than rows, it gives the first
    idle tiles two consecutive rows.
  - Path A's loop multiplies each tile's run of rows in one call (flagos-spacemit-ime.cpp).
  - Each tile's count of leftover rows stays as it was. With 8 tiles three cases change:
    - m % 32 = 27: tile 6 takes rows 0-1 in one call and tile 7 row 2. Before, tile 6 had rows 0 and 2 in two
      calls and tile 7 row 1.
    - 30: tile 7 takes both rows in one call, as one-tile does.
    - 31: tile 7 takes rows 0-1 in one call; tile 0 keeps row 2 as before.
- Why the outputs stay the same, by the code:
  - The six 1-row kernels (IME and reference) never read the row count they are given (flagos-spacemit-ime-kernels.cpp;
    the only read is a 4-row kernel's assert, :915).
  - `gemm_rows` advances one row per call (flagos-spacemit-ime.cpp:175-183).
  - The leftover rows are quantized one by one into consecutive slots (:482-492), so a call of 2 rows gives the
    kernel the same bytes as two calls of 1.
  - One-tile has made such calls since the dealing came in, with identical outputs on the Mac and on the board.
- Why this way: it is the smallest change that gives m % 32 = 30 one-tile's calls and removes the second pass at 27.
  Not chosen:
  - one-tile at m % 32 = 30 alone: a special case that leaves 27's two calls;
  - gathering a tile's rows that are not consecutive into one call: a TCM copy per row and scattered output rows, a
    larger change for the same calls.
- Checked on the Mac (host build, simulated device):
  - flagos-check-spacemit passes all checks. The path check now also counts each tile's calls with leftover rows
    (`spacemit_mm_tail_take`): 64/64 identical, with rows and calls by tile as expected.
  - Two mutations caught: one call per row, as before the fix (10 failures, at 59, 62 and 63 rows, by the calls);
    and the previous rule in the new loop (4, at 59 rows, where it gives tile 6 rows 0 and 2).
  - ThreadSanitizer clean; test-backend-ops unchanged (MUL_MAT 3, ADD 54, MUL 46, RMS_NORM 52, ROPE 165, SET_ROWS 87,
    GET_ROWS 9, SWIGLU 12).
  - An exhaustive simulation, with the rule's text taken from the source and the previous rule and ime.cpp's loop
    alongside: tile counts 1-64 and every batch of up to 3 full rounds and 3 rows, 25,088 cases. Every row is
    multiplied once, the groups are unchanged, a tile's leftover rows come in one call after its groups, and each
    tile's count is as before. The calls are the previous ones wherever every tile has at most one leftover row, and
    one-tile's wherever they are all on one tile.
- For the K3: `--bench-mm` has rows of 59 and 62 too; the runner has the logits sets `ub59` and `ub63`.
- Predictions (estimates fixed before the run, from the last-round costs above). Path A's matmul time is taken as
  (k + L) rounds, k the full rounds and L the last. At m % 32 = 27 a tile's 2 rows in one call are taken as 1.47
  rounds, as at 30; at 31 tile 0 stays the slowest, at 1.79.
  - Outputs identical: the path check on the board; KLD at the floor at `-ub 59` and `-ub 63`.
  - Qwen3-8B against one-tile under the same setting, `fit,staged`:
    - pp62, pp94 and pp126 level within 1% (the same calls);
    - pp59 +7%, pp91 +15%, pp123 +12% (today's dealing +9.7%);
    - pp63 +3%, pp95 +7.5%, pp127 +6%, as today: the fix leaves m % 32 = 31's slowest tile as it is.
  - Qwen3-8B, `split`: pp123 +5% (today's +4.4%), pp126 level, pp127 +2.6% (as today), level below 113 tokens (no
    path A). tg128 and pp512 within 1%.
  - Qwen3-8B against today's dealing (the run of 2026-10-24): pp62 +1.3%, pp123 +1.6% (`split` +0.7%), pp63 and
    pp127 unchanged. Today's dealing would also have been 2.8% and 2.2% slower than one-tile at pp94 and pp126 (not
    tested).
  - `--bench-mm`, gate, up and down, one-tile's time over the dealt one's: 1.24 at 59 rows (today's dealing 1.20),
    1.00 at 62 (0.96), 1.11 at 63 (as today).
  - Qwen3-14B, `fit,staged` against one-tile (8B's costs in rounds carried over, the matmuls' shares from 14B's first
    run): pp62 level, the lengths of 4q and 4q+1 tokens level; pp51 +19%, pp54 +17%, pp55 +43%, pp58 +17%, pp59
    +22%, pp63 +10%, pp66 +22%, pp67 +43%, pp70 +12%, pp71 +29%, pp74 +12%, pp75 +30%, pp78 +10%, pp79 +29%, pp130
    +12%, pp131 +24%, pp135 +17%.
  - Qwen3-14B, `split`: pp130 +5%, pp131 +10%, pp135 +7%, its path-A share taken as 0.38 (8B's 0.39).
  - These replace the dealing block's 14B predictions, which assumed 0.93 rounds for a last round of groups and dealt
    rows (8B measured 1.02-1.11).
- Rule (the dealing's, unchanged): the dealing with the fix stays if the outputs are identical, no tested length is
  more than 1% slower than with one-tile under the same matmul setting, and tg128 and pp512 are within 1%; otherwise
  it is examined first. Tested on 8B at the lengths the fix changes (2026-10-24's run stands for the others, whose
  calls are the same) and on 14B at all its lengths.

**The fix on the K3: Qwen3-8B at the lengths it changes, Qwen3-14B's dealing session (2026-10-25, build `f0e0549`;
`EXPS="fit-staged tail fs-tail" REPS=3`; 8B with `KLD_SETS="ub59 ub63"` at the 9 lengths below 128 with m % 32 = 27,
30 or 31, 14B with `KLD_SETS="ub63 ub79 ub131"` at its 26 lengths; `~/fx-q8b-fix.log`, `~/bench-mm-fix.out`,
`~/fx-q14b-tail.log`, the logs in `~/fx/q8b-fix` and `~/fx/q14b-tail` on the board).** The settings as in the 8B
block above: `ref` (`split`) and `fit-staged` deal the rows, `tail` and `fs-tail` are the same settings with
`FLAGOS_SPACEMIT_TAIL=one-tile`.
- Outputs identical on both models: the same text in the four settings; against `ref` mean KLD 0.000000, maximum
  0.000059 on 8B (`-ub 59` and `-ub 63`) and 0.000055-0.000058 on 14B, the same in the three others at each batch
  size. The same top token 100%, but on 14B at `-ub 79` 99.951% (one token of 2,048) in all three. `tail` is among
  them, and at 79 rows it makes `ref`'s calls: under `split` no matrix takes path A below 113 rows
  (flagos-spacemit-ime.cpp:513-519, 529), and its matmuls are within 0.75% of `ref`'s by the counters. So 99.951% is
  the floor at that batch size; that a near tie in the stored logits causes it is an inference. On the board all
  checks pass, the path check 32/32 with each tile's leftover rows and calls as the rule gives.
- The counters move a result by at most 0.50% on 8B and 0.89% on 14B. Standard deviations at most 0.57 and 0.22 t/s.
- Qwen3-8B, t/s (mean of 3 runs), dealt against one-tile under the same setting, the predictions in brackets:

| tokens (m % 32) | `fit,staged` dealt, one-tile | change | `split` dealt, one-tile | change |
|---|---|---|---|---|
| 62, 94, 126 (30) | 55.94, 56.01; 51.93, 51.90; 53.39, 53.31 | -0.1, +0.1, +0.2% (level) | 126: 47.97, 47.93 | +0.1% (level) |
| 59, 91, 123 (27) | 53.18, 50.65; 48.97, 43.83; 51.06, 46.79 | +5.0% (+7), +11.7% (+15), +9.1% (+12) | 123: 46.56, 44.78 | +4.0% (+5) |
| 63, 95, 127 (31) | 53.95, 51.74; 49.23, 45.17; 51.00, 47.63 | +4.3% (+3), +9.0% (+7.5), +7.1% (+6) | 127: 46.67, 45.18 | +3.3% (+2.6) |
| the rest (no leftover rows, or `split` off path A) | | -0.03 to +0.31% | | -0.34 to +0.58% |
| tg128, pp512 | 6.05, 6.04; 56.06, 55.61 | +0.2, +0.8% (within 1) | 6.05, 6.04; 56.08, 55.93 | +0.2, +0.3% |

- The rule is met on 8B. No length is more than 0.34% slower than one-tile (`split`'s pp128, the same calls); pp62 is
  -0.12%, where the first dealing was -1.36%. tg128 and pp512 are within 0.8%.
- Against the first dealing (its run of 2026-10-24). Each dealing is taken against one-tile in its own session, since
  one-tile's calls are the same in both; this session ran 0.0-0.7% slower at the lengths whose calls are the same,
  while the matmuls' rounds and the even and odd tiles' mean busy times there agree within 0.2%.
  - m % 32 = 30: +1.2 points (pp62; predicted +1.3).
  - 31: +0.8 and +1.0 (pp63, pp127; `split`'s pp127 +0.7; predicted unchanged).
  - 27: -0.6 (pp123; `split` -0.4; predicted +1.6).
- Path A's last round by the counters, in rounds of 32 rows (3,331-3,332 µs per Q4_0 down in both sessions), the
  first dealing against the fix, `fit,staged`. The slowest tile, by the even and odd tiles' counters, is the same in
  both runs.
  - m % 32 = 30, tile 7 with its two rows (an odd tile the slowest every time): the Q4_0 downs 1.58 -> 1.48 (one-tile
    1.47-1.48 in both sessions), the requantized downs 1.75 -> 1.59-1.60.
  - 31, tile 0 with its group and row (an even tile the slowest in 92-100% of the steps): the Q4_0 downs 1.79 ->
    1.73, the requantized downs 1.90 -> 1.78-1.88, gate and up 1.78-1.79 -> 1.71-1.72, q and o 1.73-1.74 ->
    1.71-1.73, k and v 1.61 -> 1.65-1.68. Tile 0's work is the same in both. That it runs faster because tile 7 no
    longer streams the weights twice is an inference.
  - 27, tile 6 (an even tile the slowest every time): the Q4_0 downs 1.55 -> 1.59-1.60, the requantized downs
    1.73-1.76 -> 1.90-1.92, gate and up 1.54-1.55 -> 1.57, q and o 1.52-1.53 -> 1.52-1.54, k and v 1.43 -> 1.37-1.40.
    So for the larger matrices, two rows in one call next to tile 7's single row take longer than the two calls did.
    At 30, next to tile 6's group, one call takes 0.10 rounds less than two.
  - At 123 rows the other tiles slowed too: the even and the odd tiles' mean busy times on the Q4_0 downs rose 1.1%
    and 1.4%, although the odd tiles' work is the same in both. At 127 rows both fell 1.2-1.5%. The cause is not
    isolated; a same-session run of the two assignments at 59, 91 and 123 tokens would settle the difference.
- The predictions missed only by these two costs. With the measured last rounds (1.59-1.60 at 27, 1.73 at 31) the
  same formula gives pp59 +5.3%, pp91 +11.8%, pp123 +9.3%, pp63 +4.3%, pp95 +9.0% and pp127 +7.1%, within 0.3 points
  of the measurements. The predictions had taken 30's one call (1.47) for 27, and kept 31 at the first dealing's 1.79.
- `--bench-mm` (Qwen3-4B's shapes), one-tile's time over the dealt one's:
  - 62 rows 0.99-1.00 (predicted 1.00);
  - 59 rows 1.14-1.24 (predicted 1.24): k and v 1.24, the others 1.14-1.20, as the counters' 27;
  - 63 rows 1.11-1.22, against the first dealing's 1.11-1.18 (predicted unchanged). The rows whose calls are the same
    differ by up to 0.06 between the two runs.
- Generation on 8B: tg128 6.04-6.05 in the four settings (the first dealing's run 6.07-6.08); 165.8-166.3 ms per token
  by E3c's counters.
- Qwen3-14B, t/s (mean of 3 runs), dealt against one-tile, the predictions in brackets:

| tokens (m % 32) | `fit,staged` dealt, one-tile | change |
|---|---|---|
| 51 (19) | 29.96, 26.12 | +14.7% (+19) |
| 54, 55 (22, 23) | 27.17, 22.99; 26.90, 18.88 | +18.2% (+17), +42.5% (+43) |
| 58, 59 (26, 27) | 28.34, 24.21; 23.13, 19.95 | +17.1% (+17), +15.9% (+22) |
| 62, 63 (30, 31) | 25.26, 25.25; 23.61, 20.88 | +0.0% (level), +13.1% (+10) |
| 66, 67 (2, 3) | 25.59, 21.00; 25.68, 18.02 | +21.9% (+22), +42.5% (+43) |
| 70, 71 (6, 7) | 24.61, 21.96; 24.44, 18.86 | +12.1% (+12), +29.6% (+29) |
| 74, 75 (10, 11) | 25.55, 22.97; 25.17, 19.75 | +11.2% (+12), +27.4% (+30) |
| 78, 79 (14, 15) | 26.26, 23.77; 26.55, 20.55 | +10.5% (+10), +29.2% (+29) |
| 130, 131, 135 (2, 3, 7) | 28.60, 25.43; 28.65, 23.03; 27.72, 23.57 | +12.5% (+12), +24.4% (+24), +17.6% (+17) |
| `split` at 130, 131, 135 | 25.64, 24.37; 25.45, 23.10; 25.14, 23.37 | +5.2% (+5), +10.2% (+10), +7.6% (+7) |
| the rest (no leftover rows, or `split` off path A) | | `fit,staged` 0.00 to +0.43%, `split` -0.39 to +0.52% |
| tg128, pp512 | 3.35, 3.35; 31.24, 31.29 | 0.0, -0.2% (`split` +0.3, -0.7%) |

- The rule is met on 14B. No length is more than 0.39% slower than one-tile, except `split`'s pp512 at -0.73% (the
  same calls; its `-ops` runs +0.16%). tg128 and pp512 are within 1%.
- The predictions are within 3 points except at pp51, pp59 and pp63.
  - pp59 and pp63 missed by 8B's two costs: 14B's last round is 1.61-1.64 rounds at 59 rows for gate, up and the
    Q4_0 downs, and 1.72-1.73 at 63.
  - pp51 missed by the share. At 51 rows only the downs take path A: 30.0% of the dealt batch and 39.0% of the
    one-tile one, by the counters. The prediction took 0.39 from 14B's first run (one-tile), where the formula needs
    the dealt share; with 0.30 it gives +14.5%.
- Step 3's criterion is met on 14B under the dealing at this session's lengths. `fit,staged` is never slower than
  `split` (worst -0.03%, pp512); at 51-96 tokens it is at least +39.2% faster (pp68), at 129-135 +10.3 to +12.6%. With
  one-tile (`fs-tail` against `tail`) it would also have been met at the lengths feared before the run: pp67 +11.4%
  and pp131 -0.3%, estimated at +2.1% and -2.0%.
- Against 14B's first session (2026-10-24, before the dealing), at the lengths whose calls are the same: within 0.2%
  for `fit,staged` and 0.5% for `split`; tg128 3.35 against 3.37.
- Generation on 14B: tg128 3.34-3.35; 299.5-299.7 ms per token by E3c's counters.
- Step 4 is done. The dealing with the fix meets its rule on Qwen3-8B and 14B, and `fit,staged` with it meets step 3's
  criterion on both.
- Proposed (the user's decision):
  - Keep the fix as it is and go to the default change (`fit,staged`, with the dealing) and its acceptance runs. The
    rule is met on both models; what remains at m % 32 = 27 (about 0.5% at pp123) is a cross-session measurement at
    one class of lengths, and the matrices differ in its direction.
  - Or the previous rule in the fix's loop. That gives the first dealing's calls at 27 (two on tile 6, rows 0 and 2)
    and the fix's at 30 and 31 (a tile's consecutive rows in one call). Each case's calls have been measured, and
    these were the faster ones for the batch in each, so expected about +0.5% at m % 32 = 27 and nothing else
    (cross-session). The change is the previous one-line rule plus the check counting runs of consecutive rows, with a
    same-session run at 59, 91 and 123 tokens on 8B.

**The default change (the user's decision, 2026-10-25: the fix kept as it is; made and checked on the Mac the same
day, accepted on the K3 2026-10-26, below).** The unset `FLAGOS_SPACEMIT_MM` (and an empty or unknown value) now
selects `fit,staged`: path B without the pair turns where TCM holds the batch's activations next to a 32-column slab,
else path A from 32 rows, else path C, with path A's leftover rows dealt to the idle tiles, a tile's in one call
(`mm_setting_from_env`, flagos-spacemit-ime.cpp). `split`, the default until then, and `rows`, `spacemit`, `fit` and
`fit,mixed` stay as test-only settings for comparisons, the same bytes in all six. The runner gains `EXPS=split`, since
its `ref` (the defaults) now measures what `fit-staged` measured, and its earlier runs compared with `split`.
- Why: the rule of 2026-10-22 (from 16 tokens never more than 1% slower than `split` or `rows`, and every length from
  16 to 100 tokens at least 5% faster than `split`) was met on Qwen3-0.6B and Qwen3-4B in Q4_0, and step 3's
  criterion (the same against `split` alone) on Qwen3-0.6B in Q8_0, Llama-3.2-1B in Q4_0 and Q8_0, Qwen3-8B and
  Qwen3-14B. Prompts below 16 tokens were set aside by the user (2026-10-22): there `fit,staged` was up to 8.0% slower
  than `fit` on 4B (pp2-pp8) and 4.5% slower than `split` on 0.6B (pp2).
- Checked on the Mac: all checks pass. The path check runs `split` by name and the unset setting as an eighth, which
  must take `fit,staged`'s paths with no turns and deal the leftover rows: 64/64 cases (tiles in turn and on threads).
  Three mutations of the default are caught: the unset setting as `split` with `fit,staged` right by name (60 failing
  lines, in each only the default off its path; the other 4 runs are the two cases where both rules give path C), the
  unset setting and `fit,staged` both as `split` (60), both as `fit` (56).
  test-backend-ops `MUL_MAT` with the TCM stand-in passes its 3 claimed cases, serial and on threads.
- What moves on the K3, by the code. TCM holds the activations next to a slab up to 119 rows for Qwen3-4B's q, k, v,
  gate and up, 68 for the attention output, 19 for down and 5 for the Q4_1 downs (stored as Q8_0); on Qwen3-0.6B up to
  323, 153, 97 and 79. So at llama-bench's pp128 (`-ub 128`) 4B's gate and up leave path C for path A (the other
  matrices were on path A already), and 0.6B's matrices other than down take path B (path A before). Prompts of 2-31
  tokens take path B where it fits. Generation (the 1-row path), perplexity's 512-token batches and pp512 (path A
  under both settings) keep their paths.

Predictions for the acceptance runs (fixed before them; the script's llama-bench takes `-ub 128 -fa 1 -r 3`):

| | Qwen3-4B Q4_0, `--build --depth` | Qwen3-0.6B Q4_0 |
|---|---|---|
| checks | 37/37 | 34/34 |
| perplexity at 512-token batches, `-fa on` and `-fa auto` | the fix's of 2026-10-24 to the last digit (-0.00% against that day's CPU run of 9.7740) | the fix's to the last digit (+0.61% against 14.3598) |
| mean KLD at 512-token batches | the fix's 0.00249 within 0.0002 (the CPU reference is not bit-reproducible) | 0.00472 within 0.0002 |
| mean KLD at `-ub 1 -b 512` | under M2e.3's 0.002021 by the fix (not measured at `-ub 1` since): about 0.0013-0.0018 | under 0.012145: about 0.002-0.004 |
| greedy text | the provider's as before (its prompt's few rows now on path B: the same bytes) | the same |
| pp128 | about 105-108 t/s (step 3's session: `split` 92.16, `fit,staged` 106.34; the fix -0.6%) | about 690-715 (653.34 and 695.23 there; the fix -0.3%; the script read 2% above the runner at M2e.3) |
| tg128 | about 11.0 (the fix's 11.03); the change moves no 1-row matmul | about 55.0-55.8 |
| tg32 at depths 0 / 1024 / 4096 | about 11.1 / 9.7 / 7.7 (M2e.3's 11.30 / 9.93 / 7.83 less the fix's 1-2%), not moved by the change | not run |
| pp512 at depth 4096 | about 34.4 (M2e.3's 34.63 less the fix's 0.6%), path A under both settings | not run |

- The script's benchmark (Qwen3-4B's shapes in both runs) moves where the default takes another path: predicted from
  `--bench-mm` of 2026-10-25 (`~/bench-mm-fix.out`, `split` against `fit,staged` at the same shapes). Q4_0 2560x9728
  (gate, up) at 4, 16, 64 and 128 rows -17, -26, -44 and -28% (128 rows about 6,960 us against `split`'s
  9,640-9,720). Q4_0 9728x2560 (down) +6% at 4 rows (path B without the turns at few rows), -31% at 16, -26% at 64,
  unchanged at 128. The Q4_1 down +6% at 4 rows, -15% at 64, unchanged at 16 (path C under both) and 128. The Q6_K
  head at 4 and 16 rows on path B (path C before): between -17% and +6% at 4 rows (the range of 4B's other shapes
  there), faster at 16.
- The IME path's and the CPU's rows as in the earlier runs (code not changed): on 4B about 79 and 21 t/s at pp128.

Rule (proposed with the change; the user fixes it before the runs):
- The change stands if both runs pass every check and the provider's perplexity at 512-token batches is the fix's to
  the last digit on both models (the default's outputs are `split`'s bytes, and at 512 rows both take path A).
- Speeds are reported against the predictions, without a bar: generation, the depth runs and pp512 take the paths
  they took before, so their differences are the session's (tg128 moves by about 4-5% between sessions, `build.md`
  §7). A pp128 more than 5% under its prediction is examined before the change is called accepted.
- If a check fails, the default goes back to `split` (one line in `mm_setting_from_env`) until the failure is
  understood.

How to run (on the K3, inside tmux, after a bundle of the commit; about two hours for both, an estimate):

```bash
cd /root/llama.cpp-flagos && git -c safe.directory=$PWD pull --ff-only ~/sx-default.bundle feature/flagos-spacemit
python3 ggml/src/ggml-flagos/providers/spacemit/tools/spacemit_check.py --milestone m2e --build --depth --model ~/models/Qwen3-4B-Q4_0.gguf > ~/acc-default-q4b.log 2>&1; cp -r build-flagos/m2e-logs ~/acc-default-q4b-logs
python3 ggml/src/ggml-flagos/providers/spacemit/tools/spacemit_check.py --milestone m2e > ~/acc-default-q06b.log 2>&1; cp -r build-flagos/m2e-logs ~/acc-default-q06b-logs
grep -H "Final estimate" ~/kld/fix-*.txt
```

- The fix's perplexity values for the comparison are in `~/kld/fix-*.txt` (the last command lists them).
- Then on the Mac: `scp -q k3:/root/acc-default-q4b.log k3:/root/acc-default-q06b.log ~/`.

**The acceptance runs on the K3 (2026-10-25 to 26, build `65280e5`; `~/acc-default-q4b.log` and
`~/acc-default-q06b.log` on the Mac, the fix's values from `~/kld/fix-*.txt`).** Both runs pass every check, and the
provider's perplexity at 512-token batches is the fix's to the last digit on both models; no pp128 is under its
prediction. So by its rule the change stands: `fit,staged` with the dealing is the accepted default for the matmuls of
several rows, `split` and the other settings stay test-only for comparisons.

| | Qwen3-4B Q4_0, `--build --depth`: predicted | measured | Qwen3-0.6B Q4_0: predicted | measured |
|---|---|---|---|---|
| checks | 37/37 | 37/37 | 34/34 | 34/34 |
| perplexity at 512-token batches, `-fa on` and `-fa auto` | the fix's, 9.7736 | 9.7736 in both, against the CPU's 9.774 (-0.00%) | the fix's, 14.4471 | 14.4471 in both, against the CPU's 14.3706 (+0.53%) |
| mean KLD at 512-token batches | 0.00249 within 0.0002 | 0.002489 ± 0.000156, the fix's to the last digit | 0.00472 within 0.0002 | 0.004722 ± 0.000221 (the fix's 0.004720) |
| mean KLD at `-ub 1 -b 512` | about 0.0013-0.0018 | 0.001229 ± 0.000058 | about 0.002-0.004 | 0.002533 ± 0.000089 |
| greedy text | as before | leaves the CPU's after 143 characters, as at M2e.1 and M2e.3 | as before | identical to the CPU's (M2e.3: after 88 characters) |
| pp128 | about 105-108 | 106.70 | about 690-715 | 705.50 |
| tg128 | about 11.0 | 11.00 | about 55.0-55.8 | 55.16 |
| tg32 at depths 0 / 1024 / 4096 | about 11.1 / 9.7 / 7.7 | 11.08 / 9.64 / 7.72 | not run | not run |
| pp512 at depth 4096 | about 34.4 | 34.68 | not run | not run |
| the IME path's and the CPU's pp128 | about 79 and 21 | 79.26 and 20.98 | not predicted | 542.13 and 151.83 |

- Perplexity: the fix's files hold 9.773559 and 14.447111, which both runs print as 9.7736 and 14.4471 (512-token
  batches take path A under both settings). The CPU's run on 4B gave the fix's 9.7740 again, and with it the KLD to the
  last digit. On 0.6B it moved (14.3598 on 2026-10-24, now 14.3706; 14.3795, 14.3786 and 14.3816 in earlier sessions),
  so the ratio reads +0.53% against +0.61% and the KLD 0.004722 against 0.004720: the reference's change, not the
  provider's.
- `-ub 1 -b 512` gives one-token ubatches, so every matmul takes the 1-row path, which the change does not touch (it is
  taken before the setting is read, flagos-spacemit-ime.cpp:509-511 against :528). These are the first such runs on
  the two models since the Q4_1 fix: 4B 0.002021 -> 0.001229 (-39%), 0.6B 0.012145 -> 0.002533 (-79%), against CPU
  runs of other sessions; at 512-token batches the fix removed 26% and 68%. 4B lands under its predicted range, which
  scaled M2e.3's value by the 26%; why the fix removes more at one-token ubatches was not examined.
- Greedy text: on 0.6B the provider's text now matches the CPU's in full. Not the change's doing: the prompt's few rows
  take path B instead of path C, and the path check (next item) shows the same bytes. This is the first script run on
  0.6B since the Q4_1 fix, and the CPU's runs move between sessions, so it is not attributed (the test: the model run
  with and without `FLAGOS_SPACEMIT_Q4_1=zero-point`, in one session).
- The path check passed on the board in each session's three runs of flagos-check-spacemit (with `--full`, with the
  reference kernels, and before the benchmark; a difference or a wrong path fails the program, check_spacemit.cpp:1379
  and :3140): the default takes `fit,staged`'s paths without the turns and gives `split`'s bytes at 2-128 rows of Q4_0,
  Q4_1, Q8_0 and Q6_K, here with the board's TCM.
- Speed in the same session against the IME path: on 4B pp128 +34.6% (+16.7% in M2e.3's acceptance run), tg128
  +0.2%, tg32 at depths 0 / 1024 / 4096 -0.1 / +3.7 / +21.6%, pp512 at depth 4096 +38.1%; on 0.6B pp128 +30.1%
  (+21.1%), tg128 +1.5%. Against M2e.3's run (across sessions, the Q4_1 fix in between): pp128 +15.3% on 4B and +5.7%
  on 0.6B, while the IME path read 79.29 -> 79.26 and 550.78 -> 542.13; tg128 -2.2% and -1.1% (the fix's -2.0% and
  -1.0% in its own session). tg32 at depth 1024 is 2.9% under M2e.3's where the fix's 1-2% was predicted, within the
  4-5% that generation moves between sessions (`build.md` §7). M0's targets on 4B: pp128 82 met; tg128 11.00 under
  11.1, as since the fix, the IME path reading 10.98 in this session.

The script's benchmark at Qwen3-4B's shapes, this session's 4B and 0.6B runs against the last two sessions under
`split` (Qwen3-8B's and Qwen3-14B's acceptance runs of 2026-10-24 and 25; the same code at these row counts, which are
multiples of 4 and so leave no rows over), the predictions in brackets:

| matmul | 4 rows | 16 rows | 64 rows | 128 rows |
|---|---|---|---|---|
| Q4_0 2560x9728 (gate, up) | -18.9 / -13.8% (-17%) | -25.3 / -25.6% (-26%) | -44.2 / -44.1% (-44%) | -27.3 / -27.4% (-28%: 7,010 and 6,995 us against about 6,960) |
| Q4_0 9728x2560 (down) | -1.9 / +2.9% (+6%) | -35.4 / -34.8% (-31%) | -25.1 / -25.2% (-26%) | +1.5 / -0.9% (unchanged) |
| Q4_1 9728x2560 (stored as Q8_0) | +4.2 / +4.8% (+6%) | -1.0 / -0.0% (unchanged: path C under both) | -13.3 / -13.3% (-15%) | +0.1 / +1.5% (unchanged) |
| Q6_K 2560x151936 (stored as Q8_0) | -5.6 / -7.5% (-17 to +6%) | -20.4 / -18.5% (faster) | not run | not run |

- Every move lies within 4.4 points of its prediction (down at 16 rows the furthest, faster than predicted), but down
  at 4 rows in the 4B session (7.9 points). That shape varies between sessions under both settings: `fit,staged` took
  629, 660 and 680 us in three sessions (the last `--bench-mm` of 2026-10-25, the prediction's source) and `split`
  622-644 in four, so the predicted +6% was one session's difference.
- At 1 row (the 1-row path, not touched) the times lie within 1.8% of the `split` sessions', but down in the 0.6B
  session: 520 us against 460-466 in the other three sessions, on the same code path.

### 7.4 Still proposed

- Concurrent DRAM bandwidth: a stand-alone program streaming memory on the X100 cores and the AI cores at once, against
  each alone; decides whether §6's parallel use of the X100 cores could add bandwidth at all.
- The next candidates, ordered by the `ops` tables (by estimate, rank 2 is #10, RoPE's sin/cos).

## 8. Log

- 2026-10-21: written from the session that ran the structural experiments: the design's provider modes, Common's
  catalog, our 30-step layer, the measured budget, candidates and splitting options. Nothing built or measured for it.
- 2026-10-21: candidates ranked by estimate (§5; #10 and three work-split findings added from reading the kernels);
  the op counters built (§7.1) and rank 1, #6, prepared behind `FLAGOS_SPACEMIT_QUANTIZE=shared` (§7.2), with the
  runner `scripts/fusion-experiments.sh` and `scripts/op-stats.py`; checked on the Mac, not yet run on the K3.
- 2026-10-21: the baseline run prepared (§7.1, `EXPS=base`; user: run it first, as the foundation for the candidates);
  the runner's `ops` modes also write E3c's counters, and its summary prints E3c's budget of a generated token and the
  op counters' totals against E3c's.
- 2026-10-21: the baseline run on the K3 (§7.1): valid; every op's time per token and per batch; the ceilings from the
  fixed names; three larger levers that are not fusion (the pair turns' tail, pp128's path C, the split attention's
  imbalance at depth) and an unexplained fixed cost of a small step.
- 2026-10-21: path A for small prompt batches prepared (§7.3; user request): `FLAGOS_SPACEMIT_MM=rows`, its check,
  `flagos-check-spacemit --bench-mm` and `EXPS=mm`; checked on the Mac, not yet run on the K3.
- 2026-10-21: path A for small prompt batches run on the K3 (§7.3): valid, outputs identical; pp32 +41.8%, pp64
  +36.9%, pp96 +89.0%, pp128 +15.4%, pp512 and tg128 unchanged. The proposed rule is met at the row counts measured,
  which are path A's best case; a model fitted to the bench predicts `rows` slower than today at 33-39 rows (33-44 if a
  round of few tiles reads slowly) and a balanced split 18-56% faster at every row count from 33 to 149. Not adopted;
  proposed next: llama-bench at odd prompt lengths (no code change), then the balanced split.
- 2026-10-21: the split rule's history checked (§7.3; user question): the provider has never run SpacemiT's rule at
  2-149 rows. Path B was deferred at M2b and adopted for 1 row only (2026-10-11), and no test has compared it with path
  C at several rows. Read besides: their path B puts the batch's activations in TCM when they fit. Modelled: their rule
  10-55% faster than today at 16-149 rows (estimates), slower than `rows` at the multiples of 32 that favour path A.
  Proposed next, reordered: the odd-length llama-bench with the IME path added, then path B for several rows before the
  balanced split.
- 2026-10-21: the odd-length run on the K3 (§7.3): `rows` never more than 1.1% slower than `split` (pp36) and 3-89%
  faster at 12 of 14 lengths; the predicted loss at 33-39 tokens did not appear. A round of path A costs about the same
  with one tile as with eight (234-239 ms per batch for a full group, 174-179 for a 1-row group). The IME path (SpacemiT's
  rule) has no dips, beats `split` at 16-100 tokens and `rows` at the lengths that are not multiples of 32 below 100, and
  loses to `rows` at the multiples of 32. Proposed next: path B for several rows.
- 2026-10-21: path B for several rows prepared (§7.3; the user's decision after the review): `FLAGOS_SPACEMIT_MM=spacemit`
  (`gemm_slabs`), `check_mm_paths` over all three settings (13 cases, NaN-filled outputs), `--bench-mm` at 25 row
  counts, `EXPS=pb`; on the Mac 26/26 identical with the expected paths, two mutations caught, ThreadSanitizer clean; not
  yet run on the K3.
- 2026-10-21: path B's bench on the K3 (§7.3, build `17413e6`): valid; path B 13-52% faster than path C per batch at
  2-113 rows, but 10% slower at 128-149 rows, where gate and up's activations no longer fit in TCM; path A still fastest
  at the multiples of 32 and from 128 rows. The llama-bench run (`EXPS="mm pb"`) in progress.
- 2026-10-21: `EXPS="mm pb"` on the K3 (§7.3): valid, outputs identical; `spacemit` 11-95% faster than `split` from 2 to
  100 tokens and ahead of the IME path at every length, but 8% slower at 128 and 129 tokens, so not adopted by the
  proposed rule. Per shape, path B wins where its activations fit in TCM and loses where they do not; a rule per call
  on that (path B where they fit, else path A, else path C) is estimated at 0-3% from the fastest path for each shape.
- 2026-10-21: that rule prepared as `FLAGOS_SPACEMIT_MM=fit` (§7.3; user decision), with predictions and proposed
  rules: `check_mm_paths` over four settings and 15 cases, 30/30 on the Mac, three mutations of the rule caught;
  `--bench-mm` with a fourth column; `EXPS=fit`. Not yet run on the K3.
- 2026-10-22: `fit` on the K3 (§7.3; the bench and `EXPS="mm fit"` run 2026-10-21, build `1d7e20d`, Qwen3-4B): valid,
  outputs identical. Per batch `fit`'s matmuls are never more than 0.9% slower than the fastest other setting and 5-14%
  faster than all three where the shapes' fastest paths differ. llama-bench: 6-113% faster than `split` from 2 to 100
  tokens, +15.5% at pp128, +8.6% at pp129, pp512 and tg128 unchanged; never more than 0.2% slower than `split` or
  `rows`; every prediction within 0.7% but pp4's (-5.7%, a noisy test). The adoption rule is met on Qwen3-4B. What
  remains is path A's last round (8-14% of a batch just past each multiple of 32). Not covered: Qwen3-0.6B, where `fit`
  takes path B up to 323 rows and 4B's ratios put it up to about 13% slower than `rows` at 32 rows (an estimate);
  proposed next, before the default changes: the same run on 0.6B.
- 2026-10-22: `fit` on Qwen3-0.6B (§7.3; `EXPS="mm fit"` and prompts of 160-320 tokens, build `1d7e20d`): valid,
  outputs identical; the adoption rule is not met (pp32 and pp64 7.2% and 4.2% slower than `rows`; pp2-pp8 2.7-4.9%
  faster than `split`, under 5%), so `fit` is not adopted; elsewhere 7-116% faster than `split`. Cause measured on
  both models: path B's pair turns (upstream's schedule) leave each pair's odd tile one slab's multiply behind (an odd
  tile the slowest in every path-B step; the gap about 0.9/R, R the slabs per tile), and four of 0.6B's seven matrices
  have R = 4. Proposed next: path B without the turns at several rows, as a test setting.
- 2026-10-22: path B without the turns prepared as `FLAGOS_SPACEMIT_MM=fit,staged` (§7.3; user decision), with
  predictions and proposed rules: `check_mm_paths` over five settings, 30/30 on the Mac, four mutations caught,
  ThreadSanitizer clean; `--bench-mm` with a fifth column; `EXPS=fit-staged` (also prompts of 160-320 tokens) and
  `op-stats.py pairs` (every matmul's even and odd tiles). Not yet run on the K3.
- 2026-10-22: `fit,staged` on the K3 (§7.3, build `696549b`): the bench at 4B's shapes 1.0-4.2% below `fit` from 16
  rows but 8-14% above it at 2-8 rows (the copies contend without the turns); on Qwen3-0.6B (5 runs per mode, outputs
  identical) 5-9% faster than `fit` from pp16 (the predictions within a point), the odd tiles' excess 0.1-3.6% from
  32 rows (`fit` 7-24%), but 4.9% slower than `fit` at pp2, so the rule fails on 0.6B at pp2-pp8. Per shape the
  crossover lies between 8 and 16 rows on both models. The 4B run in progress.
- 2026-10-22: `fit,staged` on Qwen3-4B (§7.3; 3 runs per mode, outputs identical): 2.3-3.9% faster than `fit` from
  pp16 to pp100 and the same from pp128, but 5.7-8.0% slower at pp2-pp8, so the rule's second part fails at pp2 and
  pp4 (+3.5% and +4.1% over `split`). The odd tiles' excess falls by 70-99% from 32 rows on both models but stays
  above the ±2% criterion at 32-64 rows (a residual odd-tile disadvantage, cause not isolated). Not adopted. The
  hybrid (the turns below 16 rows), estimated from the two schedules measured in the same sessions: both parts of the
  rule on 4B; on 0.6B the first part, the second not at pp2-pp8 (no setting reaches 5% over `split` there).
- 2026-10-22: the threshold prepared as `FLAGOS_SPACEMIT_MM=fit,mixed` (§7.3; user decision): the turns while the
  batch's activations take fewer bytes than a slab (16 rows for Q4_0 and Q4_1, 29 for Q8_0, no model constant);
  `check_mm_paths` over six settings and 21 cases, 42/42 on the Mac, five mutations caught, ThreadSanitizer clean;
  `--bench-grid`, a bench on shapes rather than a model (three layouts, 1-38 slabs per tile, 2-48 rows);
  `EXPS=fit-mixed` on both Qwen3 models, and all three schedules on Qwen3-0.6B in Q8_0 (Qwen3 only, user decision).
  Not yet run on the K3.
- 2026-10-22: the grid and the bench on the K3 (§7.3, build `cdfd29e`): `fit,mixed` switches as built, but the
  crossover moves with the slabs per tile, the row length and the layout (2-16 rows for Q4_0 and Q4_1, 10-28 for
  Q8_0), so the threshold misses by up to 6.7% (Q4_0) and 10.6% (Q4_1) at 4 or more slabs per tile, 4% allowed: not
  adopted, and no rule of its form passes even fitted to the grid. The model sessions not run (no prompt length near
  the thresholds). Also found: 6-, 10- and 14-row batches slower than 8, 12 and 16 (the kernels' remainder rows).
  Part 2 of the rule from 16 tokens (user, as a trial). Next: how to switch, the user's choice.
- 2026-10-22: `fit,staged` chosen by the user on the recorded runs (small batches set aside: both parts of the rule
  from 16 tokens, a decision after the data); `fit,mixed` and the one-slab condition not pursued. Next: its check on
  Qwen3-0.6B in Q8_0 against `split` and `fit` (§7.3, predictions and criterion fixed before the run), then the
  default change and the acceptance runs on both models.
- 2026-10-23: the Q8_0 check read (§7.3; run on the K3 2026-10-22, build `cdfd29e`, 5 runs per mode): the criterion
  met (from 16 tokens at worst -0.2% against `split`, every length from 16 to 100 at least +5.4%), outputs identical by
  text, KLD and perplexity (13.610728 in all six runs, `split`'s 13.6107; the same top token 99.951%, one token, from
  the reference's 16-bit storage, not shown directly). Against
  `fit` +1.8% at pp16, +6.7-10.2% at 32-129 tokens, +5.2-5.4% at 160-256, level at 320 (path B's TCM limit for Q8_0,
  294 rows of 1,024 values), -5.6 to -9.7% at 2-8. The choice holds on Q8_0. Next: the user's plan (prompts of 9-15
  and 17-31 tokens, another family, bigger models), then graph splitting.
- 2026-10-23: step 2 prepared (§7.3; user request): the runner's `PPODD` (the prompt lengths of its single-batch run;
  default unchanged, checked on the Mac by dry runs of every experiment), three sessions at 8-32 tokens with
  predictions and criterion. Not yet run on the K3.
- 2026-10-23: step 2 run on the K3 and read (§7.3; build `2c7c761`, 5 runs per mode on Qwen3-0.6B in Q4_0 and Q8_0,
  3 on Qwen3-4B): the criterion met on all three (at 17-31 tokens at least +16.4%, +7.5% and +23.5% against `split`;
  at 8-32 never more than 1.2% slower); against `fit` +4.2 to +7.9%, +2.7 to +8.4% and +2.4 to +4.8% at 17-31 tokens,
  the crossover at about 9, 14 and 10 rows. pp16 on Q8_0 +3.0% against `split` (the Q8_0 check: +5.4%), at part 2's
  bar. Also found: `split`'s 16-row blocks from 17 rows (k, v, o, down 75-83% slower on 0.6B), 4B's down off path B
  from 20 rows (19 for Q4_1) until path A at 32, and the remainder rows (a batch of 4q+4 tokens 5-16% faster than one
  of 4q+3, in every setting). Next: the user's step 3 (a model of another family of similar size).
- 2026-10-23: step 3 prepared (§7.3; the user's choice among candidates checked the same day against the provider's
  rules): Llama-3.2-1B-Instruct in Q4_0 and Q8_0 (unsloth, ModelScope, downloaded on the board), 2 graph splits on
  the Mac's simulated device; the acceptance script against the CPU, then `EXPS="mm pb fit fit-staged"` on both files,
  with predictions and criterion (part 2 at risk on Q8_0 at 12-31 tokens, where down leaves path B). Not yet run on
  the K3.
- 2026-10-23: step 3's Q4_0 half run on the K3 and read (§7.3; build `2c7c761`): the acceptance script 32/34, its
  perplexity bound failed (+5.45% against the CPU; mean KLD 0.0548 at 512-token batches, 0.0513 at `-ub 1`, against
  Qwen3-0.6B's 0.0147), so the accuracy is examined first and the default change waits; suspects the Q4_1 conversion
  of layers 0-1's `ffn_down`, RoPE with frequency factors, the head's requantization. The benchmark's criterion met
  (from 16 tokens at worst -0.15% against `split`, at least +19.5% at 16-100); `fit,staged` the fastest setting at
  every length from 16 to 129 tokens; `spacemit`'s path B with the activations outside TCM 4.7-5.0% slower than path
  C at 27-28 rows. The Q8_0 session and the accuracy runs (the IME path, the Q8_0 file, `-fa off`) running.
- 2026-10-23: step 3's Q8_0 half and the accuracy follow-up read (§7.3): the Q8_0 benchmark's criterion met narrowly
  (pp16 +5.70% against `split`, bar +5%; from 16 tokens at worst -0.28%); `spacemit` fastest at 4-24 tokens there, its
  down on path B with the activations outside TCM 20-26% faster than path C at 12-24 rows (18.5-21.9% slower at 27-28
  rows on Q4_0). Accuracy: the Q8_0 file matches the CPU (mean KLD 0.00076), the Q4_0 file with `-fa off` misses as
  before (0.0559), so attention, RoPE and the other non-matmul ops are cleared and the error lies in the Q4_0 file's
  weight path; one CPU run gave NaN (intermittent; its two comparisons discarded). Next: test files with layers 0-1's
  down, or the head, in Q8_0.
- 2026-10-23: the cause found (§7.3): with layers 0-1's down in Q8_0 the provider matches the CPU (perplexity +0.24%,
  mean KLD 0.0026, from +5.52% and 0.0548); the head in Q8_0 changes nothing; the upstream IME path makes the same
  error (0.0554). The in-tree Q4_1 conversion rounds the block minimum to an integer zero point (no block clamped in
  this file; error energy 0.67% of the weights'). Options (Q4_1 to Q8_0 at load, an exact correction term, or keep)
  for the user and the mentor; the Qwen3 test proposed.
- 2026-10-23: the fix, option (a) (user): Q4_1 requantized to Q8_0 at load, as Q6_K (layout `q4_1_q8_0_32x32`, the
  default; `FLAGOS_SPACEMIT_Q4_1=zero-point` keeps the old route, test-only). On the Mac all checks pass on both routes,
  two mutations caught, test-backend-ops unchanged; Llama-3.2-1B on the simulated device against the CPU: perplexity
  +0.24%, KLD 0.0022 (zero-point +1.48%, 0.0565). About 1-2% more weight bytes per token. Not yet run on the K3.
- 2026-10-23: the Q4_1 test on Qwen3 (K3, §7.3): with their Q4_1 layers in Q8_0 the provider's mean KLD against the
  CPU falls 0.0149 -> 0.0044 on Qwen3-0.6B (-71%) and 0.00334 -> 0.00260 on Qwen3-4B (-22%): the conversion is also
  most of 0.6B's error, which M2e had put down to the IME approach as a whole.
- 2026-10-24: the fix on the K3 (§7.3): all checks pass on both routes; against one CPU run per model the mean KLD
  falls 0.0548 -> 0.00302 on Llama-3.2-1B (perplexity +5.45% -> +0.20%), 0.0148 -> 0.00472 on Qwen3-0.6B (+1.02% ->
  +0.61%) and 0.00335 -> 0.00249 on Qwen3-4B; `zero-point` reproduces the old default. The Qwen3 predictions missed by
  +7% and -4% (they took the test files' values, measured against the CPU on the requantized weights). On the Mac (CPU
  only, against those layers in F32) the requantized Q8_0 lies a little further from exact arithmetic than the CPU's
  own Q4_1 path (KLD 0.00138 against 0.00109). Speed in the same session (3 rounds): tg128 -2.4%, -1.0% and -2.0%,
  pp128 -1.2%, -0.3% and -0.6%; Qwen3-4B's tg128 11.03, under M0's 11.1. Kept as the default (user).
- 2026-10-24: step 4 prepared (§7.3; the user's choice of models): Qwen3-8B Q4_0 (bartowski's file, unsloth's recipe;
  unsloth has no 8B Q4_0) and Qwen3-14B Q4_0 (unsloth), downloaded and checked on the board, fit by the code (2 graph
  splits expected; 14B's 5 query heads per KV head new; the requantized downs too large for TCM, so never path B and
  read from DRAM in generation). Predictions from a per-matrix model calibrated on Qwen3-4B's counters (within -11 to
  +4 points of Llama-3.2-1B's measured gains): `fit,staged` at least 15% faster than `split` from 16 to 100 tokens on
  both files, the narrowest 14B at 65-69 tokens; criterion as step 3's.
- 2026-10-24: step 4, Qwen3-8B's acceptance script on the K3 (§7.3): 34/34 checks; perplexity -0.14% against the CPU,
  mean KLD 0.003116 / 0.002137 (512-token batches / `-ub 1 -b 512`); pp128 50.59 (`split`) and tg128 6.05 against the
  IME path's 45.07 and 6.08; every prediction for it met. The runner's session on this file next.
- 2026-10-24: step 4, Qwen3-8B's benchmark on the K3 (§7.3): the criterion met (from 16 tokens at worst -0.25% against
  `split`, at least +20.2% from 16 to 100, outputs identical); against `fit`, the odd tiles, tg128 and the fix's cost
  (-2.34%) as predicted. Found: path A runs the batch's last m mod 4 rows on one tile, one kernel call each
  (flagos-spacemit-ime.cpp:134-143, 483-490), so 2 or 3 leftover rows cost 2 or 3 one-row passes (pp78, pp79 14 and 35
  points under the model); the corrected model predicts `fit,staged` slower than `split` at untested lengths (8B
  pp131 -2.0%, 14B pp55-75 at 4q+2 and 4q+3 tokens, down to -14.7% at pp67). Proposed: those lengths in the 14B run;
  path A dealing the leftover rows one per idle tile (outputs identical by the code).
- 2026-10-24: the same loop found in SpacemiT's in-tree path and in ggml-spacemit on every branch that has path A, and
  in ggml-org master, with no fix posted upstream (§7.3). Path A's leftover rows dealt to the idle tiles (§7.3; the
  user's decision): `path_a_tail_tile`, ime.cpp's assignment behind `FLAGOS_SPACEMIT_TAIL=one-tile` (test-only); on
  the Mac all checks pass, the path check 64/64 with each tile's leftover rows counted, three mutations caught,
  ThreadSanitizer clean, test-backend-ops unchanged; `--bench-mm`'s seventh setting and the runner's `tail` and
  `fs-tail` for the K3; predictions and rule fixed. Not yet run on the K3.
- 2026-10-25: step 4, Qwen3-14B on the K3 (runs of 2026-10-24, build `bd4b655`; §7.3): the acceptance script 34/34
  (perplexity +0.07% against the CPU, mean KLD 0.004728 / 0.004029, the first above its predicted 0.0015-0.0035,
  cause not isolated; the script reports KLD without a bound), pp128 27.48 and tg128 3.37 against the IME path's
  24.75 and 3.45. The benchmark, at step 4's first 26 lengths (the ones added after 8B's run not in it), met the
  criterion: from 16 tokens never slower than `split`, at least +25.2% from 16 to 100, outputs identical; the gains up
  to 44.5 points above the model's, `split`'s down on path C on rows of 17,408 values 1.5-2.0x the model's cost per
  weight; the fix's cost -2.64%. The untested lengths re-estimated: with today's assignment pp67 +2.1% and pp131 -2.0%
  (the criterion would fail there), with the dealing at least +10.1%. The dealing's predictions corrected for pp59
  (the model used the first proposal where m % 32 is 27 or 30), before its run.
- 2026-10-25: the dealing on the K3, Qwen3-8B (run of 2026-10-24, build `682c676`; §7.3). Outputs identical: the same
  text, KLD at the floor at `-ub 79` and `-ub 131`, the path check 32/32 on the board. The dealing is 2.6-28.6%
  faster than one-tile where it moves rows (`fit,staged` pp79 +28.6%, `split` pp131 +10.5%), 0.5-5.4 points under
  the predictions; tg128 and pp512 within 0.3%. But pp62 under `fit,staged` is -1.36%, so the rule is not met: at
  m % 32 = 30 tile 7 multiplies the same two rows as with one-tile, in two calls instead of one (1.58 rounds against
  1.47 by the counters). Step 3's criterion is met at every length under both assignments. Proposed (the user's
  decision): a tile's leftover rows in one call, its two rows consecutive (predicted: pp62 level with one-tile,
  pp123 about +1.6%; pp63 and pp127 unchanged, as corrected before any run, since tile 0 is the slowest there).
- 2026-10-25: the fix, a tile's leftover rows in one call (§7.3; the user's decision): `path_a_tail_tile` gives the
  first idle tiles two consecutive rows when fewer are idle than rows, and path A multiplies a tile's run in one call.
  With 8 tiles only m % 32 = 27, 30 (one-tile's calls) and 31 change. On the Mac all checks pass, with each tile's
  calls counted; two mutations are caught; ThreadSanitizer is clean; test-backend-ops is unchanged; an exhaustive
  simulation of the rule (25,088 cases) is clean. Predictions for 8B at the lengths the fix changes and for 14B's
  session (now on the fix) fixed; the dealing's rule unchanged. Not yet run on the K3.
- 2026-10-25: the fix on the K3 (§7.3; build `f0e0549`): Qwen3-8B at the 9 lengths it changes and Qwen3-14B's dealing
  session at its 26. Outputs identical on both; the rule is met on both (no length more than 0.39% slower than
  one-tile but `split`'s pp512 on 14B, -0.73% with the same calls; pp62 -0.12% on 8B and +0.04% on 14B); step 3's
  criterion met on 14B at every length; step 4 done. Against the first dealing on 8B (cross-session, each against
  one-tile): m % 32 = 30 +1.2 points as predicted, 31 +0.8-1.0 (predicted unchanged: tile 0 stays the slowest but runs
  faster), 27 -0.6 (predicted +1.6: two rows in one call next to tile 7's single row take 1.59-1.60 rounds against
  the two calls' 1.55; cause not isolated). With the measured last rounds the formula reproduces every 8B length
  within 0.3 points; 14B within 3 points but pp51 (the one-tile share taken for the dealt one), pp59 and pp63 (8B's
  two costs). Proposed: keep the fix and go to the default change, or the previous rule in the fix's loop (about +0.5%
  at m % 32 = 27, to be checked in one session).
- 2026-10-25: the default change (§7.3; the user's decision, the fix kept as it is): `fit,staged` with the dealing is
  the unset `FLAGOS_SPACEMIT_MM`; `split` (the default until then), `rows`, `spacemit`, `fit` and `fit,mixed` remain,
  test-only, for comparisons; the runner gains `EXPS=split`. On the Mac all checks pass, the path check running `split`
  by name and the default as an eighth setting (64/64), three mutations of the default caught. Predictions and rule
  for the acceptance runs fixed (Qwen3-4B with `--build --depth`, Qwen3-0.6B): every check passing, perplexity the
  fix's to the last digit, pp128 about +15% on 4B and +6% on 0.6B, generation and the depth runs unchanged. Not yet
  run on the K3.
- 2026-10-26: the default change on the K3 (§7.3; build `65280e5`, runs of 2026-10-25 to 26): Qwen3-4B 37/37 with
  `--build --depth` and Qwen3-0.6B 34/34; the provider's perplexity the fix's to the last digit on both (9.7736 and
  14.4471); mean KLD 0.002489 and 0.004722 at 512-token batches, 0.001229 and 0.002533 at `-ub 1 -b 512` (on these
  models the first since the Q4_1 fix); pp128 106.70 and 705.50 (predicted 105-108 and 690-715; the IME path 79.26
  and 542.13 in the same sessions), tg128 11.00 and 55.16, the depth runs as predicted; the script's benchmark within
  4.4 points of its predictions but down at 4 rows (a spread between sessions). The change stands by its rule:
  accepted. Next: graph splitting, with fusion later (user).
