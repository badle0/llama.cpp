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
tested length from 16 to 100 at least 5% faster (the rule's two parts from 16 tokens), with outputs identical; and the
acceptance script's checks pass (if its perplexity bound fails, the model's accuracy is examined first, against the IME
path on the same file). The comparisons with `fit`, `rows` and `spacemit` are reported, with no bar. If the criterion
fails, the default change waits and the failing lengths are examined first.

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
