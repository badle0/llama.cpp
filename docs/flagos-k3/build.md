# Build and test plan

All commands run from the repo root unless noted. Every build lives in its own directory, so configurations never interfere:

| Dir | Machine | Purpose |
|---|---|---|
| `build/` | K3 and Mac | FlagOS (provider-neutral now, SpacemiT provider later) + plain CPU |
| `build-ime/` | K3 | upstream SpacemiT IME baseline (ggml-cpu extra buffer type) |
| `~/llama.cpp-spacemit/build/` | K3 | SpacemiT's standalone `ggml-spacemit` backend (separate clone) |

Rules of thumb:
- Reconfigure (`cmake -S . -B …`) only when CMake options or CMake files change. Otherwise `cmake --build` is incremental.
- Build only the targets you need; the full llama.cpp tree takes a long time on the K3.
- Warnings are expected (RVV intrinsics, `_Float16` pedantic). Only `error:` / `FAILED:` matter.
- Long jobs on the K3 go inside `tmux` (`tmux new -s build`, detach `Ctrl-b d`, reattach `tmux attach -t build`).

## 0. One-time setup

K3:
```bash
apt update && apt install -y cmake ninja-build tmux pkg-config
```

Mac: Homebrew `cmake`; Xcode command-line tools. Access to the board: `ssh k3` (alias in `~/.ssh/config` on the Mac, TLS via `openssl s_client` ProxyCommand; not stored in this repo).

## 1. Get the code

```bash
git clone -b feature/flagos-spacemit https://github.com/<your-user>/<your-fork>.git ~/llama.cpp-flagos
cd ~/llama.cpp-flagos
git remote add kevin https://github.com/kevinzs2048/llama.cpp   # mentor's repo, read-only use
git fetch kevin feature/flagos-amd-890-opt
```

Verify the baseline:
```bash
git log --oneline -3                                    # your commits on top of 5794e12
git merge-base --is-ancestor 5794e12 HEAD && echo "on PDF baseline"
```

Pick up mentor updates later: `git fetch kevin && git rebase kevin/feature/flagos-amd-890-opt`.

## 2. Build A — FlagOS provider-neutral + plain CPU (`build/`)

K3:
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_FLAGOS=ON -DGGML_FLAGOS_DENGLIN=OFF -DGGML_FLAGOS_AMD=OFF
cmake --build build --target \
  ggml-flagos flagos-check-provider flagos-check-target flagos-check-graph-plan \
  llama-cli llama-bench test-backend-ops 2>&1 | tee build.log
grep -nE 'error:|FAILED' build.log | head
```

Mac (same, plus disable Metal and Accelerate BLAS so comparisons are CPU vs FlagOS only):
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_FLAGOS=ON -DGGML_FLAGOS_DENGLIN=OFF -DGGML_FLAGOS_AMD=OFF \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF
cmake --build build -j "$(sysctl -n hw.ncpu)" --target <same targets>
```

Configure output must contain `FlagOS: building provider-neutral core with no device provider` (and `riscv64 detected` on the K3). `GGML_FLAGOS_DENGLIN` defaults to ON and fails without the Denglin SDK, so always pass it OFF.

Verify:
```bash
for t in provider target graph-plan; do ./build/bin/flagos-check-$t; done   # 3 × "... checks passed"
./build/bin/llama-cli --version
```

## 3. Build B — upstream SpacemiT IME baseline (`build-ime/`, K3 only)

```bash
cmake -S . -B build-ime -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CPU_RISCV64_SPACEMIT=ON -DGGML_CPU_REPACK=OFF \
  -DGGML_RVV=ON -DGGML_RV_ZFH=ON -DGGML_RV_ZVFH=ON \
  -DGGML_RV_ZICBOP=ON -DGGML_RV_ZIHINTPAUSE=ON -DGGML_RV_ZBA=ON \
  2>&1 | grep -E 'RISCV64_SPACEMIT_IME_SPEC|riscv64 detected|Error'
cmake --build build-ime --target llama-bench llama-cli test-backend-ops 2>&1 | tee build-ime.log
```

- Expect `RISCV64_SPACEMIT_IME_SPEC: RISCV64_SPACEMIT_IME1;RISCV64_SPACEMIT_IME2`.
- `-DGGML_RV_ZBA=ON` is required (`ime.cpp` has `#error` without `__riscv_zba`; there is no CMake default for it).
- `-DGGML_CPU_REPACK=OFF` follows `docs/build-riscv64-spacemit.md` (keeps the generic repack buffer type from competing).

Run (TCM path currently aborts on this board, see `k3-hardware.md`):
```bash
SPACEMIT_DISABLE_TCM=1 ./build-ime/bin/test-backend-ops -o MUL_MAT -b CPU 2>&1 | tail -3
```
Note: `test-backend-ops` does not place weights in CPU extra buffer types, so it exercises SpacemiT's thread setup but most likely not the IME kernels. Confirm IME with a real model (§7).

## 4. Mac-only issue already fixed on this branch

`libggml-flagos` failed to link on macOS (`_ggml_backend_register` undefined): `flagos-registry.cpp` called a `libggml` function from a backend library. Commit `eaa25ff` makes `ggml_backend_flagos_reg_devices()` a no-op; registration still happens via `ggml-backend-reg.cpp`. Reproduce the macOS behaviour on Linux with `-DCMAKE_SHARED_LINKER_FLAGS="-Wl,--no-undefined"`.

## 5. Build C — SpacemiT's `ggml-spacemit` (separate clone, K3)

```bash
cd ~
wget -q https://github.com/spacemit-com/spine-runtime/releases/download/0.6.3/spine-runtime.riscv64.0.6.3.tar.gz
tar xzf spine-runtime.riscv64.0.6.3.tar.gz && ln -sfn ~/spine-runtime.riscv64.0.6.3 ~/spine-runtime   # include/ lib/

git clone --depth 1 -b agent/ggml-spacemit-backend https://github.com/spacemit-com/llama.cpp ~/llama.cpp-spacemit
cd ~/llama.cpp-spacemit
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_SPACEMIT=ON -DSPERT_DIR=$HOME/spine-runtime \
  -DGGML_CPU_REPACK=OFF -DLLAMA_OPENSSL=OFF -DGGML_RVV=ON -DGGML_RV_ZVFH=ON -DGGML_RV_ZFH=ON \
  -DGGML_RV_ZICBOP=ON -DGGML_RV_ZIHINTPAUSE=ON -DGGML_RV_ZBA=ON > cmake.log 2>&1
cmake --build build --target llama-bench llama-completion llama-perplexity test-backend-ops > build.log 2>&1
grep -E 'SpacemiT backend using|IME_SPEC' cmake.log; grep -cE 'error:|FAILED' build.log
```
Status: built on the K3 2026-10-07 at `4e782bc` (+ the E1 device-type switch below) with spine-runtime 0.6.3: `RISCV64_SPACEMIT_IME_SPEC: IME1;IME2`, 0 errors. The CMake flags are SpacemiT's current ones (their `mtmd-backend` branch, `docs/build-riscv64-spacemit.md`), minus the cross-compile toolchain file. It turns `GGML_CPU_RISCV64_SPACEMIT` off automatically. Registry name `SPACEMIT`, device `SPACEMIT0` (from source, not yet seen on the board). This llama.cpp version uses `llama-completion` for plain prompts; `llama-cli` is the chat tool.

For experiment E1 (`plan.md` M0.9), apply the device-type switch before building: `git -C ~/llama.cpp-spacemit apply ~/llama.cpp-flagos/docs/flagos-k3/scripts/spacemit-device-type.patch`. Without `GGML_SPACEMIT_DEVICE_TYPE=gpu` the build behaves exactly like stock (ACCEL).

## 6. Build D - FlagOS with the SpacemiT provider (`build-flagos/`, K3)

The SpacemiT provider (`plan.md` M1-M2d, M2c). Kept separate from `build/`, which stays the plain-CPU baseline.
```bash
python3 ggml/src/ggml-flagos/providers/spacemit/tools/spacemit_check.py --milestone m2c2 --build
```
The script configures `build-flagos/` with `-DGGML_FLAGOS=ON -DGGML_FLAGOS_DENGLIN=OFF -DGGML_FLAGOS_AMD=OFF -DGGML_FLAGOS_SPACEMIT=ON`, builds the provider, the FlagOS check tools (with `flagos-check-spacemit`), `test-backend-ops`, `llama-completion`, `llama-bench` and `llama-perplexity`, runs the milestone's checks and prints PASS/FAIL per check (INFO lines for measurements); logs go to `build-flagos/<milestone>-logs/`. From M2a the provider links spine-runtime from `~/spine-runtime` (override with `--spert-dir`). Without `--build` it only reruns the checks. Options: `--milestone m1|m2a|m2b|m2d|m2c|m2c2` (default m2c2; each milestone's expectations hold only for the code of that milestone, e.g. m2a expects `ADD` as the only claimed op), `--model` (default `~/models/Qwen3-0.6B-Q4_0.gguf`), `--ppl-text`, `--ime-build`, `--skip-support`, `--build-dir`, `--spert-dir`, `--tcm-dir` (default `~/tcmtest`, for the TCM checks), `--segv <segv.so>` (preloads the crash reporter `scripts/segv.c` into the model, perplexity and llama-bench runs).

Accuracy against the CPU's logits (M2c's exit; M2d's numbers in §7 were measured the same way, with 512-row batches only). The base logits come from a CPU-only run; reuse the file from the M2d measurement if it is still there (set `KB` to its path):
```bash
B=build-flagos/bin; M=~/models/Qwen3-4B-Q4_0.gguf; KB=~/kld/qwen3-4b-cpu.bin; mkdir -p ~/kld
[ -f "$KB" ] || FLAGOS_SPACEMIT_DISABLE=1 $B/llama-perplexity -m $M -f ~/ppl.txt -c 512 --chunks 8 -t 8 -fa on \
  --kl-divergence-base "$KB" > ~/kld/base.log 2>&1
~/tcmtest/tcmrelease --apply
$B/llama-perplexity -m $M -f ~/ppl.txt -c 512 --chunks 8 -t 8 -fa on --kl-divergence-base "$KB" --kl-divergence \
  > ~/kld/m2c.log 2>&1; grep -E 'Mean +KLD|99.0% +KLD|Maximum KLD|Same top p' ~/kld/m2c.log
# the same through the generation path: 1-token ubatches use the 1-row kernels and the GEMV path (about 10 minutes)
~/tcmtest/tcmrelease --apply
$B/llama-perplexity -m $M -f ~/ppl.txt -c 512 --chunks 8 -t 8 -fa on -ub 1 --kl-divergence-base "$KB" --kl-divergence \
  > ~/kld/m2c-ub1.log 2>&1; grep -E 'Mean +KLD|99.0% +KLD|Maximum KLD|Same top p' ~/kld/m2c-ub1.log
```

Mac: same CMake options plus `-DGGML_METAL=OFF -DGGML_BLAS=OFF`; the provider compiles but finds no device (`flagos-check-spacemit` reports "device checks skipped").

Mac host-test device (from M2d): in a separate build directory, `-DFLAGOS_SPACEMIT_HOST_TEST_DEVICE=ON` exposes a simulated device (8 tiles run one after another by the serial executor, reference kernels), so the provider's claims, tiling, executor and reference kernels run every claimed `test-backend-ops` case before the K3. The RVV and IME kernels compile only on the K3, so the K3 run is still needed. The option exists only off riscv64, and is for tests: llama.cpp would also use the simulated device in model runs.
```bash
cmake -S . -B build-host -DCMAKE_BUILD_TYPE=Release -DGGML_FLAGOS=ON -DGGML_FLAGOS_DENGLIN=OFF -DGGML_FLAGOS_AMD=OFF \
  -DGGML_FLAGOS_SPACEMIT=ON -DFLAGOS_SPACEMIT_HOST_TEST_DEVICE=ON -DGGML_METAL=OFF -DGGML_BLAS=OFF
cmake --build build-host -j "$(sysctl -n hw.ncpu)" --target flagos-check-spacemit test-backend-ops
./build-host/bin/flagos-check-spacemit --expect-device | tail -1        # "all checks passed"
for o in ADD MUL RMS_NORM ROPE SET_ROWS GET_ROWS SWIGLU RMS_NORM_MUL_ADD ADD_RMS_NORM RMS_NORM_MUL_ROPE ROPE_SET_ROWS; do
  printf '%s: ' $o; ./build-host/bin/test-backend-ops test -o $o -b FlagOS:SpacemiT:0 2>&1 | grep 'tests passed'
done
```
The riscv64-only code (RVV and IME kernels) can be parsed on the Mac without a RISC-V toolchain: `docs/flagos-k3/scripts/rvv-syntax-check.sh` runs Apple clang for riscv64 with RVV (a stand-in `riscv_vector.h` in `scripts/rvv-syntax/`), checking types, intrinsic signatures and template instantiations. Nothing is assembled or linked, so the K3 build remains the real test. Expected: 0 errors; 2 unused-variable warnings in the copied IME kernel (body unchanged, D6).

## 7. Models and benchmarks

```bash
mkdir -p ~/models && cd ~/models
MS=https://modelscope.cn/models/unsloth          # ModelScope: fast from the board; Qwen's own GGUF repos have no Q4_0
wget -c -q --show-progress $MS/Qwen3-0.6B-GGUF/resolve/master/Qwen3-0.6B-Q4_0.gguf   # 382156480 bytes
wget -c -q --show-progress $MS/Qwen3-4B-GGUF/resolve/master/Qwen3-4B-Q4_0.gguf       # 2375773472 bytes
```
List a repo's files: `wget -qO- "https://modelscope.cn/api/v1/models/unsloth/Qwen3-4B-GGUF/repo/files?Revision=master&Recursive=true"`. The 4B file matches SpacemiT's table (2.21 GiB); their 0.6B file was about 6 MiB smaller, so compare 0.6B numbers to their table loosely.
SpacemiT's doc benchmarks Qwen3-0.6B Q4_0 and qwen35 2B Q4_1 (`docs/build-riscv64-spacemit.md`); A100 IME supports Q2_K–Q6_K, Q4_0/1, Q5_0/1, Q8_0.

Baselines (M0.5, M0.6): `scripts/m0-baselines.sh <model>` runs every mode interleaved, then perplexity, and saves `summary.txt` (modes and switches described in its header).

**Qwen3-4B Q4_0, 2026-10-08** (5 interleaved runs, `-t 8 -p 128 -n 128 -ub 128 -fa 1 -mmp 0`; perplexity on 8 x 512 tokens of `~/ppl.txt`, `-fa on`):

| Mode | pp128 (t/s) | tg128 (t/s) | Perplexity |
|---|---|---|---|
| `cpu`: our fork, X100 cores only | 21.33 | 6.07 | 9.7740 |
| `ime`: our fork, upstream IME path + TCM | 78.95 | **11.10** | 9.7680 |
| `spacemit`: ggml-spacemit as shipped | 76.07 | 6.82 | 9.7679 |
| `spacemit-best`: ggml-spacemit reading CPU buffers, norm pinning off | **81.94** | 10.75 | 9.7480 |

- `ime` reproduces SpacemiT's published numbers for this model (79.74 / 11.29) within 1-2%, and has the fastest generation.
- Perplexity differs by at most 0.27% across modes, far inside the +/- 0.70 error bar: the upstream IME kernels are accurate on a real model (the first check that actually uses the IME weight layout; `test-backend-ops` does not).
- ggml-spacemit as shipped generates 39% slower than `ime` (ACCEL with its own buffers only plus norm pinning, `device-type.md` §5). With both fixed it is 3% slower in generation and 4% faster in prefill.
- Targets for the provider (M2e exit): tg128 >= 11.1 (`ime`) and pp128 >= 82 (`spacemit-best`) on this model.
**Provider M2b (Q4_0 matmuls on the AI cores), 2026-10-09** (`spacemit_check.py --milestone m2b`: llama-bench 3 runs, same flags; modes run back to back, not interleaved; perplexity as above):

| Model | Mode | pp128 (t/s) | tg128 (t/s) | Perplexity |
|---|---|---|---|---|
| Qwen3-4B Q4_0 | `provider` | 51.08 | 5.82 | 9.7617 (-0.13% vs `cpu`) |
| | `cpu` | 21.12 | 5.55 | 9.7740 (separate rerun; the runner's CPU run crashed once, see below) |
| | `ime` | 79.10 | 10.99 | - |
| Qwen3-0.6B Q4_0 | `provider` | 333.27 | 26.75 | within 1% of `cpu` |
| | `cpu` | 155.56 | 28.17 | |
| | `ime` | 547.19 | 54.39 | |

- The runner's CPU-only perplexity run (provider disabled) crashed once: SIGSEGV reading address 0 in `libggml-cpu.so.0.19.0` at offset 0x99316, during its first pass (`dmesg`, 2026-10-09 12:59). The crashing instruction is a vector load in `ggml_vec_dot_q4_1_q8_1` (RVV version, `ggml-cpu/arch/riscv/quants.c:277`; `addr2line` on the library built 10-08 20:31), so it was one of Qwen3-4B's 4 Q4_1 `ffn_down` matmuls, which run on the X100 cores in CPU-only and provider mode alike until M2c. The fork does not modify the CPU backend, and upstream has not changed this function since our base; the identical rerun passed, and 5 more runs passed (1 crash in 8 CPU-only runs of Qwen3-4B). The disassembly rules out a code bug: at 0x992f6 the loop sets `a2 = a5 + 16` (the second half of the activation block `y[ib].qs`), and no instruction before the crashing load at 0x99316 writes `a2`; meanwhile two loads through `a5` succeeded (0x992fe, 0x9930e), so `a2` cannot be 0 by the program's own data flow. Not a llama.cpp bug; two explanations remain, both in the platform (kernel or X100 core): (a) `a2` was changed from outside the program, for example while the thread was in the kernel; (b) the load was valid but hit a page that needed a fault, and the fault address was recorded wrongly as 0 (the `dmesg` address is the core's report of the fault, not necessarily `a2`). Tests without llama.cpp (`scripts/vfault.c`), 2026-10-09: about 17 million page faults per run on 16-byte vector loads of the mapped model file (8 and 16 threads, the latter with 300k preemptions) and on scalar loads left the register intact every time (0 corrupted), so a page fault on a vector load does not by itself cause (a) at any rate near the crash's (about 1 per 100k-400k faults). Mode `x` (29 million faults in the middle of a 16-byte vector load crossing into an unmapped page) also passed: every byte loaded, the register intact, no misreported fault address. Transparent huge pages are `madvise`-only on this board and ggml never requests them, so background page merging cannot have moved the activation buffer either. Neither mechanism reproduces; the cause is unknown after one occurrence in 8 runs. Treated as platform noise: rerun a run that dies with signal 11, load `segv.so` during acceptance runs so a recurrence leaves a register dump, and report it to SpacemiT together with the TCM issues if it recurs (plan question 8). Crash reporter for another occurrence: `scripts/segv.c` (`LD_PRELOAD`; prints all registers and a backtrace).
- Prefill gains 2.1-2.4x over the X100 cores; generation does not (4B +5%, 0.6B -5%). Graph splits: 355 per token on 4B (178 CPU + 177 provider, about 5 round trips per layer, `sched-summary.py`), 277 on 0.6B; 1 without the provider.
- Kernels (`flagos-check-spacemit --bench`, Qwen3-4B FFN shapes): 1 row 0.60 ms for 14 MB of weights, about 23 GB/s, the AI cores' memory bandwidth; 4 rows cost about the same as 1; at 128 rows path A reaches 960 GFLOP/s and path C 648 GFLOP/s. The CPU column of that benchmark uses ggml's plain Q4_0 path, not the repacked kernels llama.cpp uses (`REPACK = 1`), so it understates the X100 cores.
- Estimated 4B generation budget (172 ms per token): Q4_0 matmuls on the AI cores about 85 ms; Q6_K output head on the X100 cores about 25 ms; other X100 work about 15 ms; the remaining about 45 ms are the 355 handoffs (about 0.13 ms each, consistent with X3b's 40 ms for 286 extra splits). `ime` avoids all handoffs: its CPU worker threads are pinned to the AI cores (`ggml-cpu/spacemit/ime.cpp:1692`), so the whole graph runs there.
- IME confirmed active (separate check, `-lv 4`): `CPU_RISCV64_SPACEMIT model buffer size = 2349.12 MiB` (repacked weights) next to `CPU_Mapped model buffer size = 2246.67 MiB` (the mmap'd GGUF), `graph splits = 1`. The single split (no backend handoffs) is the structural reason the IME path generates fastest. The baseline run itself lacked these lines (library INFO messages need `-lv 4`; fixed in the script).

**Provider M2d (the rest of a layer except attention on the AI cores), 2026-10-09/10** (commits `f1cf809` C5b, `7a2b897`, `e5f69ff`; `spacemit_check.py --milestone m2d`: 26/26 on Qwen3-4B, 28/28 on Qwen3-0.6B; llama-bench 3 runs, same flags, back to back):

| Model | Mode | pp128 (t/s) | tg128 (t/s) | Perplexity |
|---|---|---|---|---|
| Qwen3-4B Q4_0 | `provider` | 53.00 | 7.16 | 9.7945 (+0.21% vs `cpu`) |
| | `cpu` | 21.34 | 5.95 | 9.7740 |
| | `ime` | 79.49 | 11.06 | - |
| Qwen3-0.6B Q4_0 | `provider` | 369.32 | 29.03 | 14.4361 (+0.61% vs `cpu`) |
| | `cpu` | 155.60 | 26.03 | 14.3492 |
| | `ime` | 546.31 | 54.34 | - |

- Correctness on the A100 cores: every `test-backend-ops` set passes with the ported RVV kernels and with the references (ADD 54, MUL 46, RMS_NORM 52, ROPE 165, SET_ROWS 87, GET_ROWS 9, SWIGLU 12; RMS_NORM_MUL_ADD 30, ADD_RMS_NORM 25, RMS_NORM_MUL_ROPE 144, ROPE_SET_ROWS 24). `flagos-check-spacemit` layer NMSE 4.15e-4 with the RVV and IME kernels and 3.84e-4 with the references (Mac 3.97e-4); ROPE with heads of 256 (the RVV rotation) NMSE 5.9e-16; Q4_0 matmuls 40 cases, max NMSE 2.8e-5. Robustness: 20 consecutive `flagos-check-spacemit` runs (8,000 deliberately failing launches) with no failure or hang.
- Placement (`GGML_SCHED_DEBUG=2`, needs `-lv 5`; `sched-summary.py`): 83 splits per token, 42 CPU + 41 provider = 1 (token embedding) + 36 x 2 (layer, attention) + 2 (last layer part, output head) + 4 x 2 (Q4_1 `ffn_down`); the same for 512-token and 1-token graphs, so C5b works. The CPU runs only `FLASH_ATTN_EXT` x36, the 4 Q4_1 and the Q6_K head `MUL_MAT`s and the embedding `GET_ROWS`. Copies per token: `Qcur` x36, `ffn_swiglu` x4, `result_norm` x1, about 0.75 MB; K and V go straight into the CPU-resident cache.
- Generation: 171.8 -> 139.7 ms per token; 272 fewer splits at about 0.118 ms each, as M2b's handoff model predicted (estimate was 7.5-8 t/s). Prefill only +4%: handoffs cost per ubatch, so 128-token batches gain little (the plan's 60-65 t/s estimate was wrong).
- Accuracy against the CPU's logits (`llama-perplexity --kl-divergence`, 8 x 512 tokens): mean KLD 0.00238 (IME 0.00339), 99% KLD 0.0215 (0.0266), max 0.127 (0.350), same top token 97.0% (96.9%). M2d is at least as close to the CPU as the upstream IME path; the CPU still runs the output head and the Q4_1 matmuls, which IME quantizes. Use IME's KLD as the ceiling for M2c. (The absolute `PPL(Q)/PPL(base)` values, 1.010 and 1.008, are both about 0.8% above the plain perplexity runs; likely the saved base clips log-probabilities 16 nats below the top logit, `tools/perplexity/perplexity.cpp:86`.)
- Flash attention off (`-fa off`): perplexity 9.8052 vs `cpu` 9.7652 (+0.41%), so the transposed V write (one-element rows on one tile) and the CPU's non-flash attention are correct; pp128 -20% (52.95 -> 42.59), tg128 -8.5% (7.09 -> 6.49). `-fa auto` keeps flash attention on under M2d (attention runs on the layer's device, the CPU).
- Context 2048 (2 chunks): perplexity 7.7498 vs `cpu` 7.7232 (+0.34%). A 1,000-token prompt in 8 ubatches (7 without outputs) generates correctly.
- Generation against context depth (`llama-bench -n 32 -d`): provider 7.36 / 5.61 / 1.72 t/s at depth 0 / 1024 / 4096, `cpu` 5.61 / 4.75 / 1.65, `ime` 11.17 / 9.35 / 6.35. Attention on the X100 cores costs +445 ms per token at depth 4096 (IME on the A100 cores: +68 ms): beyond about 1k tokens of context, attention dominates and M2e decides generation speed.
- What limits speed (requantized copies, speed only: `llama-quantize --allow-requantize`): moving the 4 Q4_1 `ffn_down` matmuls to the AI cores (`--tensor-type ffn_down=q4_0`) raises pp128 53.31 -> 83.01 (+56%) and tg128 6.95 -> 7.26; they are 3% of the FLOPs but took 36% of prefill, because ggml-cpu repacks only Q4_0 (no Q4_1 in `ggml-cpu/repack.cpp`). Also moving the output head (`--pure`, Q4_0 head) gives pp128 85.00, tg128 8.19: the Q6_K head on the X100 costs about 25 ms per token. With the Q4_1 matmuls moved, pp512 (`-ub 512`) is 87.19 against `ime` 87.56 (original model: 54.26): our matmul paths match IME for large batches, and path B is not needed.
- Microbenchmarks (`--bench`): launch 11.2 us, barrier 0.89 us per node (M2a 9.9 / 0.79); 16M-float ADD 22.7 GB/s with the ported RVV kernel (M2a's hand-written one: 21.3); matmul timings unchanged from M2b.
- Qwen3-0.6B: 65 splits per token (1 + 28 x 2 + 2 + 3 x 2: 28 layers, 3 with Q4_1 `ffn_down`), M2b 277. tg128 26.75 -> 29.03 (+8.5%), much less than on 4B: the output head (tied to the token embedding; about 128 MB per token if Q6_K like 4B's, type not checked) stays on the X100 cores and is roughly a third of the 34 ms per token (estimate), so M2c matters more here. The `cpu` row varies between days (tg128 28.17 at M2b, 26.03 now), so compare within a run. Perplexity +0.61%, within 1%; the smaller model is more sensitive to the IME's 8-bit activations.


**Provider M2c.1 (Q4_1 matmuls on the AI cores), 2026-10-10** (commit `6577170`; `spacemit_check.py --milestone m2c`: 28/28 on Qwen3-4B; llama-bench 3 runs, same flags, back to back):

| Model | Mode | pp128 (t/s) | tg128 (t/s) | Perplexity |
|---|---|---|---|---|
| Qwen3-4B Q4_0 | `provider` | 83.49 | 7.22 | 9.7647 (-0.10% vs `cpu`) |
| | `cpu` | 21.14 | 5.58 | 9.7740 |
| | `ime` | 79.29 | 11.05 | - |

- Correctness on the A100 cores (`flagos-check-spacemit --full`, IME kernels): Q4_1 against the CPU on the converted weights, 20 cases up to 512 rows and up to 9728 x 2560 (the 4B `ffn_down`), max NMSE 9.7e-7 (bound 5e-4; the Mac's reference 1.1e-6). The zero-point probe (zero point 15 against 32 equal activations) gives 3.0e-7 for 1 row and for 4, so the 1-row kernel's exact branch, which the provider enables, computes like the 4-row kernel (upstream's 16-bit branch would give about 4.6, shown by emulation on the Mac). Q4_0 unchanged: 40 cases, 2.8e-5. With the reference kernels the layer NMSE is 3.84e-4, as in M2d. Every `test-backend-ops` set still passes with both kernel sets; no `MUL_MAT` case fits the IME layouts. The conversion alone changes a matmul on random weights by NMSE 5.86e-3, a CPU computation that gives the same value on the Mac: both machines produce the same zero points.
- Placement: 75 splits per token, as predicted (M2d 83): the 4 layers with a Q4_1 `ffn_down` no longer go back to the X100 cores for it. Provider buffers: model 1955.75 MiB, compute 72 MiB.
- Prefill: pp128 53.00 -> 83.49 (+58%), above `ime` (79.29) and the prefill target of 82 (`spacemit-best`, M0.5 above), so the pp half of the M2e exit is met. The same as with the requantized model (83.01), although the Q4_1 kernel is 1.1-1.5x slower than Q4_0's at 4-128 rows: the 4 matmuls are 3% of the FLOPs, so their extra 14 ms per 128-token batch is under 1% of prefill.
- Generation: tg128 7.16 -> 7.22, within the variation between days (the `cpu` row went 5.95 -> 5.58); provider over `cpu` 1.20 -> 1.29; `llama-completion` 7.46 t/s against 5.97 with the provider disabled. The same-day comparison with the requantized model (+4.5%, M2d above) is the better estimate. What remains to `ime`'s 11.05 is the Q6_K output head on the X100 cores (about 25 ms per token, M2c.2) and attention (M2e).
- Accuracy against the CPU's logits (`llama-perplexity --kl-divergence`, 8 x 512 tokens, 2,040 scored; commands in §6):

| Run | Mean KLD | 99% KLD | Max KLD | Same top token |
|---|---|---|---|---|
| M2c.1, 512-token batches (prefill path) | 0.00316 ± 0.00022 | 0.0244 | 0.241 | 96.86 ± 0.39% |
| M2c.1, `-ub 1` (generation path) | 0.00203 ± 0.00020 | 0.0175 | 0.287 | 97.30 ± 0.36% |
| M2d, 512-token batches | 0.00238 | 0.0215 | 0.127 | 97.0% |
| `ime`, 512-token batches | 0.00339 | 0.0266 | 0.350 | 96.9% |

  The prefill path rose 0.0008 over M2d (about 3 standard errors): the cost of the zero-point conversion and the 8-bit activations in the 4 Q4_1 matmuls. It equals `ime` within one standard error. The generation path, measured here for the first time in any milestone, is the more accurate one: in prefill the 4-row Q4_0 quantizer (`quantize_a_4row_i8_hp`) gives 4 consecutive tokens one shared activation scale per 32 values; in generation each token has its own (the Q4_1 kernels keep per-row scales in both). This is also the first model-level check of M2b's 1-row Q4_0 kernel (fp16 accumulation).
- Microbenchmarks (`--bench`, the 4B `ffn_down` shape 9728 x 2560, in us; Q4_0 at the same shape; the CPU column is what llama.cpp runs for Q4_1, since ggml-cpu does not repack it):

| Rows | Q4_1 provider | Q4_0 provider | Q4_1 `cpu` |
|---|---|---|---|
| 1 | 579 | 593 | 1804 |
| 4 | 740 | 669 | 9016 |
| 16 | 2180 | 1554 | 28192 |
| 64 | 6955 | 4806 | 117211 |
| 128 | 10129 | 6676 | 228752 |

  1 row is memory-bound: 25.5 GB/s of Q4_1 weights against 23.6 for Q4_0, so the exact 1-row branch costs nothing. At 128 rows (path A for both) 629 against 955 GFLOP/s: the 32x32 kernel's work per 32-value block. Launch 11.0 us, per node 0.90 us, 16M-float `ADD` 22.4 GB/s (as M2d). Qwen3-0.6B not run (expected 59 splits).

Manual checks:
- Always report `pp` and `tg` separately; repeat and interleave runs and note mean and median.
- Confirm IME is active: a `CPU_RISCV64_SPACEMIT` model buffer in `./build-ime/bin/llama-perplexity ... -lv 4` output (buffer name from `ggml/src/ggml-cpu/spacemit/ime.cpp:1477`).
- Where threads run: `ps -L -o tid,psr,comm -p $(pgrep llama-bench)` (`psr` 8-15 = A100).

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ime.cpp: #error "riscv zba extension not enabled"` | add `-DGGML_RV_ZBA=ON` |
| `wait tcm buffer failed for cpu_id: 0` (abort) | TCM path; use `SPACEMIT_DISABLE_TCM=1`; see `k3-hardware.md` |
| `open(/dev/tcm_sync_mem) failed, errno=2` | device absent; barrier falls back to heap (harmless on its own) |
| `taskset -c 8 …` → invalid argument | A100 cores need `/proc/set_ai_thread` first |
| CMake fatal error about Denglin SDK | pass `-DGGML_FLAGOS_DENGLIN=OFF` |
| macOS `_ggml_backend_register` undefined | needs commit `eaa25ff` (§4) |
| `git` fails with `127.0.0.1:<port>` | Mac git proxy points at a stopped proxy; set it to the proxy app's port |
| `GGML_NATIVE is not compatible with GGML_BACKEND_DL` | don't enable `GGML_BACKEND_DL` for bring-up |
