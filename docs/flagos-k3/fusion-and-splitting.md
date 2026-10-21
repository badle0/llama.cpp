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

**Not measured:** time per op kind (how much of the 85.2 ms the norms, ropes, quantize steps, cache writes and each
matmul role take). This is the main gap; the op counters of §7.1 measure it.

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

### 7.3 Still proposed

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
