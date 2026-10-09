# Choosing the ggml device type for the K3: GPU, IGPU or ACCEL

Written 2026-10-07. **Decision (mentor, 2026-10-08): ACCEL.** This note keeps the reasoning and measurements behind it. It defines what "right" means, what each type predicts, and the experiments (X0-X6) that test the predictions. Scripts: `scripts/patch-spacemit.py`, `scripts/e1-device-type.sh`.

## 1. What the device type controls in llama.cpp

The type is a placement policy, not a hardware label. llama.cpp derives four things from it:

| Effect | GPU | IGPU | ACCEL | Code |
|---|---|---|---|---|
| Model device (owns layers) | yes, every one | yes, first one only, and only if no GPU exists | no (unless named with `--device`) | `src/llama.cpp:210-276` |
| Weights of offloaded layers | device buffer | device buffer | - | `src/llama-model.cpp` |
| Weights of CPU layers | - | - | device buffer first, if the device supports the weight's op | `src/llama-model.cpp:903-917` |
| KV cache | device buffer for offloaded layers | same | CPU buffer | `src/llama-kv-cache.cpp:211-217` |
| Backend order (priority) | model devices first | same | after model devices, before CPU; always created | `src/llama-context.cpp:320-350` |
| Default offload (`-ngl auto`) | sized from `get_memory` | same | not applicable | `common/common.h:465`, `common/fit.cpp:112-130` |

The scheduler then places ops (`ggml/src/ggml-backend.cpp`):
- An op whose output or weight already lives in a buffer runs on the highest-priority backend that supports both that buffer type and the op; if none does, it aborts (877-930).
- Weight ops with host-resident weights may move to a higher-priority backend through `offload_op` (956-967).
- Backends then spread to neighbouring ops they support (1113-1190) and take over ops whose input buffers they accept (1193).
- Splits run one after another; an input in a buffer the next backend cannot use is copied.

ggml's own definitions (`ggml/include/ggml-backend.h:134-145`): GPU "using dedicated memory"; IGPU "integrated GPU device using host memory"; ACCEL "intended to be used together with the CPU backend (e.g. BLAS or AMX)".

## 2. What "right behavior" means

K3 facts that matter: one coherent DRAM shared by the X100 cores (CPU backend) and the A100 cores (AI backend); ggml runs splits one after another, so the two never compute at the same time; the A100 path is much faster (E1 smoke run: prompt 19x, generation 3.3x over the X100 cores).

Time per step is then roughly: sum of op times on their backends, plus a fixed cost per split switch, plus bytes copied between backends. Every supported op is faster on the AI backend, and a copy between two buffers in the same DRAM is pure waste. So the best possible placement ("oracle") is: every op the AI backend supports runs there, in the fewest splits the unsupported ops allow, with no copies and every tensor stored once.

A device type is right if, under it, llama.cpp reaches the oracle by default and under its standard flags, without breaking correctness:

| | Criterion | Measured by |
|---|---|---|
| R1 | Correct results, no aborts, for every normal flag combination | text, perplexity vs the CPU reference; `test-backend-ops` |
| R2 | Coverage: all supported ops on the AI backend | buffer placement; split assignments |
| R3 | Few splits: close to the number of separate runs of unsupported ops | `graph splits` in the load log |
| R4 | No copies, single storage; memory budget not double-counted | buffer sizes, RSS, `--fit` decisions |
| R5 | Good default: no flags needed | `auto` mode (no `-ngl`) |
| R6 | Control and coexistence: `-ngl`, `--device`, `-ot`, `-nkvo` behave as documented; one backend instance per engine; sensible next to a discrete GPU | `half`, `accel-dev` modes; code analysis |

R1 is a hard requirement; R2-R4 decide speed; R5-R6 decide how llama.cpp users experience the device.

## 3. Predictions

The type matters most through state ownership: ops follow pre-allocated data (weights, KV cache). Whether ownership limits the AI backend depends on its buffer policy, i.e. which buffers it accepts as op inputs.

- **P1, strict buffers** (backend accepts only its own buffers, like stock ggml-spacemit): as ACCEL the KV cache stays in CPU buffers, so KV writes and attention run on the X100 cores, with extra splits and copies in every layer. GPU and IGPU own the KV cache and stay close to the oracle.
- **P2, host-accepting buffers** (our planned provider, `plan.md` D4): the AI backend can read the CPU-owned KV cache, so all three types reach the same placement on a K3-only board. The choice then rests on R5-R6, not speed.
- **P3**: on a board without a discrete GPU, IGPU behaves exactly like GPU.
- **P4**: `-ngl auto` offloads every layer for GPU and IGPU, because the device reports system RAM.
- **P5**: naming an ACCEL device with `--device` creates a second backend instance for it (code reading; untested).
- **P6** (code only, no discrete GPU on the board): next to a discrete GPU, GPU type competes for layers in proportion to reported memory (31 GiB of system RAM against the GPU's VRAM, likely a bad split); IGPU is dropped entirely; ACCEL keeps accelerating the CPU-side layers.

## 4. Experiments

| | Question | How | Tests |
|---|---|---|---|
| X0 | Are the backend's kernels correct in isolation? | `test-backend-ops -b SPACEMIT0 -o <op>` for the ops Qwen3 uses | R1 |
| X1 | Placement, splits and correctness per type | `e1-device-type.sh`, modes `accel gpu igpu cpu`, `PPL_TEXT=` set | P1, P3; R1-R4 |
| X2 | Speed per type, at KV depth 0 and 1024 | same run, `REPS=3` | P1 size |
| X3 | Does buffer policy, not type, explain X1/X2? | rerun with `ACCEPT_HOST=1 MODES="accel gpu igpu"` | P2 |
| X4 | Default behavior | mode `auto` (GPU type, no `-ngl`) | P4; R5 |
| X5 | Control | modes `half` (`-ngl` partial) and `accel-dev` (`--device`) | P5; R6 |
| X6 | Coexistence with a discrete GPU | code analysis only | P6; R6 |

Decision rule:
- If P1 and P2 hold, the type does not limit speed for our provider. Choose by R5-R6 and the deployment target: a K3 on its own favours GPU or IGPU (layer and KV control through `-ngl`); a K3 next to a discrete GPU favours ACCEL (or IGPU, if leaving the K3 unused there is acceptable).
- If P2 fails (ACCEL still slower with host-accepting buffers), ownership matters beyond buffer policy: choose GPU or IGPU.
- If P3 holds, IGPU matches ggml's definition ("host memory") better than GPU ("dedicated memory"). Prefer it unless being dropped next to a discrete GPU is a problem.
- A mode that fails R1 is excluded until the cause is known; first check whether the cause is the type or the backend (X0).

## 5. Results

**X0, 2026-10-07** (ggml-spacemit `4e782bc` + switches, spine-runtime 0.6.3, default ACCEL, strict buffers; `test-backend-ops -b SPACEMIT0 -o <op>`, which allocates every test tensor in the backend's own buffer, so IME kernels and repacking are exercised):

| Op | Passed | Op | Passed |
|---|---|---|---|
| MUL_MAT | 784/1072, FAIL | GET_ROWS | 8/8 |
| SOFT_MAX | 24/212, FAIL | SET_ROWS | 86/86 |
| SCALE | 1/4, FAIL | MUL | 67/67 |
| CPY | 60/73, FAIL | ADD | 75/75 |
| CONT | 18/26, FAIL | GLU | 0/0, most likely the filter: GLU cases are named by their GLU op (`-o SWIGLU`), so `-o GLU` matches none (2026-10-09; the exact command was not recorded). `test-backend-ops` did not cover SwiGLU; the GPU-type model runs did execute it |
| RMS_NORM, ROPE, FLASH_ATTN_EXT | no summary line; the run probably died (cause pending) | | |

Consequence: E1's ACCEL-vs-GPU differences mix device-type effects with backend bugs. ACCEL mode's garbage text is more likely a bug exposed by the ops that ACCEL placement puts on the backend than a property of ACCEL itself. Perplexity per mode (X1) and the buffer-policy swap (X3) are needed before drawing conclusions about the type. Failure details (which types and shapes) are pending.

**X1, X2, X4, X5 on Qwen3-0.6B Q4_0, 2026-10-07** (strict buffers; perplexity on 8 x 512 tokens of `docs/*.md`; llama-bench 3 runs, `-t 8 -ub 128 -fa 1`):

| Mode | Text for "The capital of France is" | Perplexity | pp128 | pp128 @1024 | tg128 | tg128 @1024 |
|---|---|---|---|---|---|---|
| accel | garbage | **151936.0** (= vocabulary size: all tokens equally likely) | 561.3 | 270.1 | 21.3 | 12.5 |
| gpu | "Paris, and the capital of the United States is Washington, D.C." | 14.5404 | 594.4 | 293.2 | 57.3 | 35.3 |
| igpu | same as gpu | 14.5404 | 597.3 | 293.6 | 57.4 | 35.2 |
| cpu (X100 only) | "Paris. The capital of France is also the capital of the country." | 14.3794 | 29.6 | 26.8 | 16.6 | 12.7 |
| auto (GPU type, no `-ngl`) | same as gpu | - | - | - | - | - |
| half (`-ngl 14`) | same as cpu | - | - | - | - | - |
| accel-dev (`--device SPACEMIT0`) | garbage | - | - | - | - | - |

Placement: `half` offloads 14/29 layers, KV cache 2400 MiB CPU + 2080 MiB SPACEMIT, 2 graph splits. `accel-dev` keeps the whole KV cache on the CPU (4480 MiB), SPACEMIT compute buffer 1612 MiB, **58 graph splits per 512-token batch and 280 per generated token**; the GPU-type modes report 2.

Reading:
- P3 confirmed: IGPU behaves exactly like GPU on this board.
- P1's mechanism confirmed: as ACCEL the backend and the CPU alternate about 10 times per layer for each generated token; generation is 2.7x slower than GPU type and, at depth 1024, no faster than the X100 cores alone.
- GPU/IGPU perplexity is 1.1% above the CPU's; `half`, which keeps the output layer on the CPU, reproduces the CPU text, so the output layer's lossy repack is the likely cause.
- ACCEL output is unusable (uniform logits in batched evaluation, garbage when generating). Localizing the failing op is pending; until then it cannot be attributed to the ACCEL type itself.

**X0 details, 2026-10-07** (`scripts/x0-summary.py` over saved logs; FAIL counts):

| Op | Failures | Detail | Hit by Qwen3 runs? |
|---|---|---|---|
| MUL_MAT | 290 | F16 weights 208 (NMSE up to 427); Q8_0 46 (no error value, likely sentinel/overrun); Q6_K 10 (NMSE up to 4e15); IQ2_S/IQ3_XXS 22 | F16 only without flash attention; Q6_K/Q8_0 possibly (output layer types) |
| SOFT_MAX | 189 | every failing case has a mask (NMSE 0.02-0.3) | only without flash attention |
| CPY | 14 | F32/F16/BF16, permuted and plain; "sentinel mismatch" = writes past the output | shape-dependent |
| CONT | 9 | F16/BF16 only | unlikely |
| SCALE | 4 | ignores `bias` | no (`bias=0`) |
| RMS_NORM | aborts | `rvv_kernels.cpp:1656: GGML_ASSERT(epsilon > 0.0f)` on eps = 0 cases | no (eps 1e-6) |
| ROPE | aborts | `rvv_kernels.cpp:3998: GGML_ASSERT(ctx.workspace_size >= ...)`: workspace not sized for ROPE (in a `-o ROPE` run; other ops' graphs can grow the workspace first) | no: Qwen3's F32 NEOX heads of 128 take the scalar path with a stack cache (`rvv_kernels.cpp:3992-3996`), which skips the assert (clarified 2026-10-09) |
| FLASH_ATTN_EXT | segfault + 4 | head size 40 with ALiBi crashes; ALiBi cases NMSE ~0.006 | no (head size 128, no ALiBi) |

Its `supports_op` claims many cases its kernels get wrong. None of these is yet tied to ACCEL mode's output (perplexity exactly the vocabulary size = identical logits, which looks like an unwritten result rather than an inaccurate kernel). The tensor diff (`scripts/tensor-diff.py`) decides that.

**Tensor dump, 2026-10-07** (`llama-eval-callback`, prompt "The capital of France is"): the CPU and GPU dumps have 986 tensors, the ACCEL dump 1126, i.e. 5 more per layer. That matches attention without flash attention (`kq`, softmax, `kqv`, permute, cont). (The first diff version compared sums only and flagged `Qcur-0` ROPE in both ACCEL and GPU; that was a false alarm from sign-mixed values and is fixed: `tensor-diff.py` now compares printed values by tensor name.)

Hypothesis under test (each step has code or measurement behind it; the chain as a whole is unconfirmed):
1. ggml-spacemit's `supports_op` checks `t->buffer` only (`ggml-spacemit.cpp:614-617`). Views of the KV cache have no buffer of their own before allocation, so it accepts attention ops on the CPU-resident KV cache.
2. As ACCEL, the scheduler then places attention on SPACEMIT and copies KV views in (280 splits per token, 1612 MiB compute buffer). llama.cpp's automatic flash attention compares the attention node's device with the layer's device and disables flash attention on a mismatch (`src/llama-context.cpp:506-560`): layer on CPU, attention on SPACEMIT0.
3. Attention without flash attention uses F16 `MUL_MAT` and masked `SOFT_MAX`, both broken in X0. Logits come out garbage.
4. As GPU, the KV cache and the layer are on SPACEMIT, flash attention stays on, the broken kernels are not used.

Decisive check: ACCEL with `-fa on` should be correct; GPU with `-fa off` should be garbage.

Device-type finding regardless of the outcome (R5): under ACCEL, a layer's device is the CPU, so `-fa auto` disables flash attention whenever an ACCEL backend runs attention. Under GPU or IGPU the layer's device is the accelerator and flash attention stays on by default. An ACCEL provider would need users to pass `-fa on`, or working non-flash attention kernels. `e1-device-type.sh` now passes `-fa on` (env `FA`) so all modes use the same attention path.

**Confirmed, 2026-10-08.** Load logs: ACCEL "layer 0 is assigned to device CPU but Flash Attention is assigned to device SPACEMIT0 (usually due to missing support) ... set to disabled"; GPU "Flash Attention enabled". Dumps: CPU and GPU have 28 `FLASH_ATTN_EXT` and 0 `SOFT_MAX`, ACCEL the reverse. Cross-over perplexity (8 x 512 tokens):

| Run | Perplexity |
|---|---|
| ACCEL, `-fa auto` (switched off) | 151936.0 |
| ACCEL, `-fa on` | 14.5404, identical to GPU |
| GPU, `-fa auto` (stays on) | 14.5404 |
| GPU, `-fa off` | 151936.0 |

Conclusions:
- The wrong output is a ggml-spacemit bug: its attention path without flash attention (F16 `MUL_MAT`, masked `SOFT_MAX`) is broken under either type, and its `supports_op` accepts ops on views of CPU-resident tensors. Not a property of ACCEL.
- The speed numbers stand: `llama-bench` always ran with `-fa 1`, so the 2.4-2.8x generation gap was measured on a correct computation. Whether it comes only from the backend refusing CPU buffers is X3.
- R5 finding confirmed: under ACCEL, `-fa auto` turns flash attention off whenever the accelerator runs attention.

**X3, 2026-10-08** (Qwen3-0.6B Q4_0; `ACCEPT_HOST=1`: ggml-spacemit may use non-weight tensors in CPU buffers; flash attention on everywhere):

| | ACCEL, strict buffers | ACCEL, reads CPU buffers | GPU / IGPU |
|---|---|---|---|
| Perplexity | 14.5404 (`-fa on`) | 14.5404 | 14.5404 |
| Graph splits, prefill / generation | 58 / 280 | 2 / **224** | 2 / 2 |
| SPACEMIT compute buffer | 1612 MiB | 298.75 MiB | 298.75 MiB |
| pp128 / pp128 @1024 (t/s) | 561.3 / 270.1 | 595.4 / 289.0 | 596.6 / 293.8 (GPU) |
| tg128 / tg128 @1024 (t/s) | 21.3 / 12.5 | **35.0 / 21.6** | 57.4 / 35.2 (GPU) |

- Prompt processing: P2 holds. Reading CPU buffers removes the KV copies (compute buffer back to 298.75 MiB) and gives the same 2 splits and the same speed as GPU, within 0-2%.
- Generation: at batch size 1 ACCEL still has 224 splits per token (8 per layer) and is 39% slower than GPU, though 64-73% faster than with strict buffers. Cause: X3b below.

**X3b, 2026-10-08: the 224 splits.** `GGML_SCHED_DEBUG=2` on one generation step (`scripts/sched-summary.py`): 112 CPU splits run only `RMS_NORM` named `norm` (4 per layer x 28), everything else runs on SPACEMIT0; per layer `Qcur`, `Kcur`, `ffn_inp` and `l_out` are copied to the CPU and back.

Cause: llama.cpp's norm-pinning rule (`src/llama-context.cpp:2499-2520` here, `:2457-2470` in SpacemiT's copy). For batches under 32 tokens or full offload, every tensor named `norm` (here also `l_last`) is forced onto the backend of `model.dev_layer(il)`. Under ACCEL that device is the CPU for every layer, so generation pins all norms to the CPU; forced assignments are never changed by the scheduler (`ggml/src/ggml-backend.cpp:1080`). Added with pipeline parallelism (PR #6017, 2024-03-13); marked `FIXME: fix in ggml_backend_sched`.

Test: a diagnostic switch in SpacemiT's copy only, `LLAMA_NO_NORM_PIN` (skips the rule when set). Same session, back to back, ACCEL, reads CPU buffers, `-fa on`, Qwen3-0.6B Q4_0, llama-bench 3 runs:

| | Norm pinning on | Norm pinning off | GPU type (X3) |
|---|---|---|---|
| Splits per generated token | 224 | 2 | 2 |
| pp128 (t/s) | 477.1 +/- 57.9 | 595.0 +/- 5.4 | 596.6 |
| pp128 @1024 | 262.0 +/- 11.7 | 290.2 +/- 1.6 | 293.8 |
| tg128 | 34.2 +/- 2.6 | 57.1 +/- 0.1 | 57.4 |
| tg128 @1024 | 21.4 +/- 0.9 | 34.6 +/- 0.1 | 35.2 |
| Perplexity | 14.5404 | 14.5404 | 14.5404 |

Conclusion for D3 (ACCEL): **ACCEL reaches GPU-type speed and accuracy (within 1-2%)** when (1) the backend reads non-weight tensors from CPU buffers (the provider's design, `plan.md` §2.3), (2) flash attention stays on, and (3) the norm-pinning rule does not pin to the CPU. (2) and (3) are both llama.cpp heuristics that assume the layer's device is where the layer's work runs, which is false for ACCEL. Fix candidates in `src/llama-context.cpp`, for the mentor to decide (fork patch or upstream issue): in the flash-attention check, accept an ACCEL backend that supports the op when the layer's device is the CPU; in norm pinning, skip the pin when the layer's device is the CPU. Without them, users need `-fa on` and accept about 39% slower generation. The pinning also adds run-to-run jitter (pp128 +/- 57.9 vs +/- 5.4).

**X3b on Qwen3-4B Q4_0, 2026-10-08** (36 layers; same settings: reads CPU buffers, `-fa on`, llama-bench 3 runs):

| | ACCEL, pinning on | ACCEL, `LLAMA_NO_NORM_PIN=1` | GPU type |
|---|---|---|---|
| graph splits (bs=512 / bs=1) | 2 / 288 | 2 / 2 | 2 / 2 |
| Text, perplexity | identical, 9.7480 | identical, 9.7480 | identical, 9.7480 |
| pp128 | 81.69 | 82.20 | 82.21 |
| pp128 @ d1024 | 59.61 | 59.88 | 59.59 |
| tg128 | 7.52 (-30.0%) | 10.75 | 10.76 |
| tg128 @ d1024 | 6.90 (-24.3%) | 9.12 | 9.19 |

- 288 = 36 layers x 4 norms x 2, as predicted. ACCEL without pinning equals GPU type within 0.8%; results are bit-identical in all modes.
- Prefill is unaffected (the rule only applies to batches under 32 tokens).
- Cost per extra split, from time per token: 0.6B about 53 us (80 us at depth 1024), 4B about 140 us (123 us at depth 1024). It is not a fixed per-switch cost; it grows with model size. Cause not measured (candidates: ggml-spacemit creating a spine-runtime stream per `graph_compute`, cache effects of moving 2560-wide activations between core types). My prediction of about 15% loss for 4B was wrong: the loss is 30%.
- Reference: SpacemiT's published upstream-IME numbers for this model (`docs/build-riscv64-spacemit.md`) are pp128 79.74, tg128 11.29; ggml-spacemit here is +3.1% / -4.7% against them (different build and run, not an A/B).

## 6. Where the concepts come from

There is no single ggml or llama.cpp document that states a theory of device types. The semantics live in header comments, a few functions, and the pull requests that introduced them. Sections 1-3 of this note are a synthesis of those sources plus the K3's hardware facts, tested by the measurements in section 5.

Core idea: the device type decides which tensors a device owns (where weights and the KV cache are allocated). The scheduler then runs each op on a backend that can use the op's buffers, and copies an input whenever the chosen backend cannot. A GPU-type device owns whole layers in memory it treats as separate; an ACCEL device owns only the weights of ops it claims, while everything else stays in CPU memory. On shared-memory hardware an ACCEL backend can read that CPU memory in place if it declares host buffers compatible, which is why the ACCEL memory model is "own only what you transform (repacked weights), read everything else where it is". FlagOS's `ai_accelerator`/`cpu_accelerator` is a separate label (`flagos-provider.cpp:54-69`) with no effect on memory or placement.

Code, in reading order:

| What | Where |
|---|---|
| Type definitions | `ggml/include/ggml-backend.h:134-145` |
| Which devices hold layers | `src/llama.cpp:210-276` |
| CPU-side buffer list, ACCEL first ("CPU: ACCEL -> GPU host -> CPU extra -> CPU") | `src/llama-model.cpp:903-917` |
| A weight goes to the first buffer type whose device supports the weight's op | `src/llama-model-loader.cpp:907` (`weight_buft_supported`), `1047-1052` (`select_weight_buft`) |
| KV cache placement | `src/llama-kv-cache.cpp:211-217` |
| Backend list and priority | `src/llama-context.cpp:330-350` |
| Automatic flash-attention rule | `src/llama-context.cpp:506-560` |
| Ops follow pre-allocated data; views judged by their source buffer | `ggml/src/ggml-backend.cpp:877-930` |
| Offload of host-weight ops; spread and upgrade passes; copies | `ggml/src/ggml-backend.cpp:956-967`, `1113-1193`, pass 5 |
| CPU accepts any host buffer | `ggml/src/ggml-cpu/ggml-cpu.cpp:480-481` |

Upstream changes (read the PR discussions on GitHub, `ggml-org/llama.cpp`):

| PR | Date | What it introduced |
|---|---|---|
| #6210 | 2024-06-13 | BLAS as a separate backend; `supports_buft` and the scheduler pass that moves ops to backends with compatible buffers. Origin of "an accelerator working next to the CPU on host memory" |
| #9707 | 2024-10-03 | device and backend registry interfaces |
| #10026 | 2024-10-30 | model loader refactor: the `ACCEL` type and the CPU-side buffer list |
| #10446 | 2024-12-07 | online repacking: weights repacked at load into a separate buffer type, the precedent for the A2 memory model |
| #15797 | 2025-09-11 | the `IGPU` type |

Design PDF V2.0: §5.3 (`memory_domain_id`: engines sharing one memory domain), §6.3 (a provider must define host-visible, UMA and device-local semantics), §12.1 (GGML owns placement and copy buffers; the provider owns workspace and private caches), §13.1 (vendors must document their memory model).

General background, not ggml-specific: data-locality-aware task scheduling on heterogeneous machines (for example StarPU, Augonnet et al., 2011); shared virtual memory on SoCs (the HSA specification); the Roofline model (Williams, Waterman, Patterson, 2009), which explains why token generation is memory-bandwidth-bound and avoided copies matter.
