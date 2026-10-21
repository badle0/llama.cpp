# Fusion and graph splitting: groundwork for the SpacemiT provider

Started 2026-10-21 (user: explore operator fusion and graph splitting while the mentor's answers are pending; pruning
left aside). Exploration only: nothing here is built. Measured numbers cite `structural-experiments.md` §6 (runs 4-5),
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
matmul role take). This is the main gap; §7 item 1.

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

## 7. Proposed measurements (not built; need the user's go-ahead)

1. **Time per op kind per token:** extend the counters (`FLAGOS_SPACEMIT_TEST_STATS`) to sum kernel time and imbalance
   per op kind and per matmul role (Q, K, V, O, gate, up, down, head), for generation and prompt processing. Ranks §5
   by measurement; it is also M4's first item ("profile a generation step"). Small test-only change, one K3 run.
2. **Concurrent DRAM bandwidth:** a stand-alone program streaming memory on the X100 cores and the AI cores at once,
   against each alone. Decides whether §6's parallel use of the X100 cores could add bandwidth at all.
3. Later, by the results: one internal candidate (6 or 7) behind a test switch, outputs byte-identical, measured with
   the counters.

## 8. Log

- 2026-10-21: written from the session that ran the structural experiments: the design's provider modes, Common's
  catalog, our 30-step layer, the measured budget, candidates and splitting options. Nothing built or measured for it.
