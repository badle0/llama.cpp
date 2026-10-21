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
python3 ggml/src/ggml-flagos/providers/spacemit/tools/spacemit_check.py --milestone m2e --build
```
The script configures `build-flagos/` with `-DGGML_FLAGOS=ON -DGGML_FLAGOS_DENGLIN=OFF -DGGML_FLAGOS_AMD=OFF -DGGML_FLAGOS_SPACEMIT=ON`, builds the provider, the FlagOS check tools (with `flagos-check-spacemit`), `test-backend-ops`, `llama-completion`, `llama-bench` and `llama-perplexity`, runs the milestone's checks and prints PASS/FAIL per check (INFO lines for measurements); logs go to `build-flagos/<milestone>-logs/`. From M2a the provider links spine-runtime from `~/spine-runtime` (override with `--spert-dir`). Without `--build` it only reruns the checks. Options: `--milestone m1|m2a|m2b|m2d|m2c|m2c2|m2e` (default m2e; each milestone's expectations hold only for the code of that milestone, e.g. m2a expects `ADD` as the only claimed op; from M2e.1 m2c2 fails its support check, as `FLASH_ATTN_EXT` is claimed), `--depth` (also tg32 at depths 0, 1024 and 4096 and pp512 at depth 4096, for the provider, the CPU and the IME build; about an hour more), `--model` (default `~/models/Qwen3-0.6B-Q4_0.gguf`), `--ppl-text`, `--ime-build`, `--skip-support`, `--build-dir`, `--spert-dir`, `--tcm-dir` (default `~/tcmtest`, for the TCM checks), `--segv <segv.so>` (preloads the crash reporter `scripts/segv.c` into the model, perplexity and llama-bench runs). m2e adds: the model run's default `-fa auto` must keep flash attention on the provider (C5a) and a token must take 2 graph splits; perplexity with `-fa on` and with `-fa auto`; mean KLD against the CPU at the default batch and at `-ub 1` (the CPU run writes the base file, deleted afterwards; compare the values with M2c.2's by hand).

Accuracy against the CPU's logits (M2c's exit; M2d's numbers in §7 were measured the same way, with 512-row batches only). The base logits come from a CPU-only run; reuse the file from the M2d measurement if it is still there (set `KB` to its path):
```bash
B=build-flagos/bin; M=~/models/Qwen3-4B-Q4_0.gguf; KB=~/kld/qwen3-4b-cpu.bin; mkdir -p ~/kld
[ -f "$KB" ] || FLAGOS_SPACEMIT_DISABLE=1 $B/llama-perplexity -m $M -f ~/ppl.txt -c 512 --chunks 8 -t 8 -fa on \
  --kl-divergence-base "$KB" > ~/kld/base.log 2>&1
~/tcmtest/tcmrelease --apply
# llama-perplexity runs n_batch / n_ctx = 4 sequences at once, each in its own cache stream: a 512-token ubatch holds
# 128 tokens of each
$B/llama-perplexity -m $M -f ~/ppl.txt -c 512 --chunks 8 -t 8 -fa on --kl-divergence-base "$KB" --kl-divergence \
  > ~/kld/m2c.log 2>&1; grep -E 'Mean +KLD|99.0% +KLD|Maximum KLD|Same top p' ~/kld/m2c.log
# the same through the generation path (about 10 minutes): -b 512 runs one sequence, so every ubatch is one token (the
# 1-row matmuls; from n_kv 512 the split over the tiles). With the default batch, -ub 1 puts a token of each of 2
# sequences into every ubatch (2 streams), as in every -ub 1 run before 2026-10-14 (§7, "M2e.3 round 2")
~/tcmtest/tcmrelease --apply
$B/llama-perplexity -m $M -f ~/ppl.txt -c 512 -b 512 --chunks 8 -t 8 -fa on -ub 1 --kl-divergence-base "$KB" \
  --kl-divergence > ~/kld/m2c-ub1.log 2>&1; grep -E 'Mean +KLD|99.0% +KLD|Maximum KLD|Same top p' ~/kld/m2c-ub1.log
```

Mac: same CMake options plus `-DGGML_METAL=OFF -DGGML_BLAS=OFF`; the provider compiles but finds no device (`flagos-check-spacemit` reports "device checks skipped").

Mac host-test device (from M2d): in a separate build directory, `-DFLAGOS_SPACEMIT_HOST_TEST_DEVICE=ON` exposes a simulated device (8 tiles run one after another by the serial executor, reference kernels), so the provider's claims, tiling, executor and reference kernels run every claimed `test-backend-ops` case before the K3. The RVV and IME kernels compile only on the K3, so the K3 run is still needed. The option exists only off riscv64, and is for tests: llama.cpp would also use the simulated device in model runs. The serial stand-in has no TCM unless `FLAGOS_SPACEMIT_TEST_TCM=<bytes>` gives it one (heap memory); `flagos-check-spacemit` sets it to 384 KiB for its 1-row strategy check, so the 1-row path and path A run on the Mac too.
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
- ggml-spacemit as shipped generates 39% slower than `ime` (ACCEL with its own buffers only plus norm pinning, `device-type.md` §5). With both fixed it is 3% slower in generation and 4% faster in prompt processing.
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
  - **Second occurrence, 2026-10-13, with registers.** The M2e.1 runner's CPU-only `llama-bench -p 512 -n 0 -d 4096` (provider disabled, after about 10 minutes of full load on the X100 cores from the CPU depth runs) got SIGSEGV with fault address 0 at the same instruction: `libggml-cpu.so.0 + 0x99316`, `ggml_vec_dot_q4_1_q8_1 + 0x40`, called from `ggml_compute_forward_mul_mat` on an OpenMP worker (`segv.so` dump in `build-flagos/m2e-logs/bench-cpu-depth-pp.err`; `segv.so` exits with 139, so the kernel logs no segfault line).
  - The registers are exactly those of block 25 of one Q4_1 row's dot product, except `a2`: `a0` = 304 blocks (K 9728), `t1` = 25, `s4` = 0x17c0 = 6080 bytes per row (304 x 20), `s6` = the next weight row, `a3 = s6 - 6080 + 25 x 20 + 4` = `&x[25].qs`, `a5 = s0 + 4 + 25 x 36` = `&y[25].qs` (`s0` the activations, 36-byte Q8_1 blocks).
  - `a2` is 0 where the code set it to `a5 + 16` (0x2b2789aa88) a few instructions before. This rules out (b): the register itself is 0, not a wrongly reported fault address. (a) remains: `a2` was cleared from outside the program's instruction stream, by a kernel path or by the core.
  - Twice at the same instruction, with every other register intact, so not random corruption anywhere. A path that clears `a2` when the thread is interrupted between 0x992f6 and 0x99316 (two vector loads through `a5` run in between) would fit; unconfirmed. Both occurrences came in long CPU-only runs under full X100 load.
  - Still not in llama.cpp, nor in the provider, which runs nothing heavy on the X100 cores since M2e.1. To report to SpacemiT (question 8) with this dump, the disassembly of 0x992d6-0x99316 of the current library and `uname -a`.
- Prompt processing gains 2.1-2.4x over the X100 cores; generation does not (4B +5%, 0.6B -5%). Graph splits: 355 per token on 4B (178 CPU + 177 provider, about 5 round trips per layer, `sched-summary.py`), 277 on 0.6B; 1 without the provider.
- Kernels (`flagos-check-spacemit --bench`, Qwen3-4B FFN shapes): 1 row 0.60 ms for 14 MB of weights, about 23 GB/s, the AI cores' memory bandwidth; 4 rows cost about the same as 1; at 128 rows path A reaches 960 GFLOP/s and path C 648 GFLOP/s (two shapes, noted 2026-10-21: at 128 rows the down projection 9728 x 2560 takes path A and gate and up 2560 x 9728 path C; the same shape on both paths: `fusion-and-splitting.md` §7.3). The CPU column of that benchmark uses ggml's plain Q4_0 path, not the repacked kernels llama.cpp uses (`REPACK = 1`), so it understates the X100 cores.
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
- Generation: 171.8 -> 139.7 ms per token; 272 fewer splits at about 0.118 ms each, as M2b's handoff model predicted (estimate was 7.5-8 t/s). Prompt processing only +4%: handoffs cost per ubatch, so 128-token batches gain little (the plan's 60-65 t/s estimate was wrong).
- Accuracy against the CPU's logits (`llama-perplexity --kl-divergence`, 8 x 512 tokens): mean KLD 0.00238 (IME 0.00339), 99% KLD 0.0215 (0.0266), max 0.127 (0.350), same top token 97.0% (96.9%). M2d is at least as close to the CPU as the upstream IME path; the CPU still runs the output head and the Q4_1 matmuls, which IME quantizes. Use IME's KLD as the ceiling for M2c. (The absolute `PPL(Q)/PPL(base)` values, 1.010 and 1.008, are both about 0.8% above the plain perplexity runs; likely the saved base clips log-probabilities 16 nats below the top logit, `tools/perplexity/perplexity.cpp:86`.)
- Flash attention off (`-fa off`): perplexity 9.8052 vs `cpu` 9.7652 (+0.41%), so the transposed V write (one-element rows on one tile) and the CPU's non-flash attention are correct; pp128 -20% (52.95 -> 42.59), tg128 -8.5% (7.09 -> 6.49). `-fa auto` keeps flash attention on under M2d (attention runs on the layer's device, the CPU).
- Context 2048 (2 chunks): perplexity 7.7498 vs `cpu` 7.7232 (+0.34%). A 1,000-token prompt in 8 ubatches (7 without outputs) generates correctly.
- Generation against context depth (`llama-bench -n 32 -d`): provider 7.36 / 5.61 / 1.72 t/s at depth 0 / 1024 / 4096, `cpu` 5.61 / 4.75 / 1.65, `ime` 11.17 / 9.35 / 6.35. Attention on the X100 cores costs +445 ms per token at depth 4096 (IME on the A100 cores: +68 ms): beyond about 1k tokens of context, attention dominates and M2e decides generation speed.
- What limits speed (requantized copies, speed only: `llama-quantize --allow-requantize`): moving the 4 Q4_1 `ffn_down` matmuls to the AI cores (`--tensor-type ffn_down=q4_0`) raises pp128 53.31 -> 83.01 (+56%) and tg128 6.95 -> 7.26; they are 3% of the FLOPs but took 36% of prompt processing, because ggml-cpu repacks only Q4_0 (no Q4_1 in `ggml-cpu/repack.cpp`). Also moving the output head (`--pure`, Q4_0 head) gives pp128 85.00, tg128 8.19: the Q6_K head on the X100 costs about 25 ms per token. With the Q4_1 matmuls moved, pp512 (`-ub 512`) is 87.19 against `ime` 87.56 (original model: 54.26): our matmul paths match IME for large batches, and path B is not needed.
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
- Prompt processing: pp128 53.00 -> 83.49 (+58%), above `ime` (79.29) and the pp128 target of 82 (`spacemit-best`, M0.5 above), so the pp half of the M2e exit is met. The same as with the requantized model (83.01), although the Q4_1 kernel is 1.1-1.5x slower than Q4_0's at 4-128 rows: the 4 matmuls are 3% of the FLOPs, so their extra 14 ms per 128-token batch is under 1% of prompt processing.
- Generation: tg128 7.16 -> 7.22, within the variation between days (the `cpu` row went 5.95 -> 5.58); provider over `cpu` 1.20 -> 1.29; `llama-completion` 7.46 t/s against 5.97 with the provider disabled. The same-day comparison with the requantized model (+4.5%, M2d above) is the better estimate. What remains to `ime`'s 11.05 is the Q6_K output head on the X100 cores (about 25 ms per token, M2c.2) and attention (M2e).
- Accuracy against the CPU's logits (`llama-perplexity --kl-divergence`, 8 x 512 tokens, 2,040 scored; commands in §6):

| Run | Mean KLD | 99% KLD | Max KLD | Same top token |
|---|---|---|---|---|
| M2c.1, 512-token batches (batched paths) | 0.00316 ± 0.00022 | 0.0244 | 0.241 | 96.86 ± 0.39% |
| M2c.1, `-ub 1` (generation path) | 0.00203 ± 0.00020 | 0.0175 | 0.287 | 97.30 ± 0.36% |
| M2d, 512-token batches | 0.00238 | 0.0215 | 0.127 | 97.0% |
| `ime`, 512-token batches | 0.00339 | 0.0266 | 0.350 | 96.9% |

  The 512-token batches rose 0.0008 over M2d (about 3 standard errors): the cost of the zero-point conversion and the 8-bit activations in the 4 Q4_1 matmuls. It equals `ime` within one standard error. The generation path, measured here for the first time in any milestone, is the more accurate one: in batches the 4-row Q4_0 quantizer (`quantize_a_4row_i8_hp`) gives 4 consecutive tokens one shared activation scale per 32 values; in generation each token has its own (the Q4_1 kernels keep per-row scales in both). This is also the first model-level check of M2b's 1-row Q4_0 kernel (fp16 accumulation).
- Microbenchmarks (`--bench`, the 4B `ffn_down` shape 9728 x 2560, in us; Q4_0 at the same shape; the CPU column is what llama.cpp runs for Q4_1, since ggml-cpu does not repack it):

| Rows | Q4_1 provider | Q4_0 provider | Q4_1 `cpu` |
|---|---|---|---|
| 1 | 579 | 593 | 1804 |
| 4 | 740 | 669 | 9016 |
| 16 | 2180 | 1554 | 28192 |
| 64 | 6955 | 4806 | 117211 |
| 128 | 10129 | 6676 | 228752 |

  1 row is memory-bound: 25.5 GB/s of Q4_1 weights against 23.6 for Q4_0, so the exact 1-row branch costs nothing. At 128 rows (path A for both) 629 against 955 GFLOP/s: the 32x32 kernel's work per 32-value block. Launch 11.0 us, per node 0.90 us, 16M-float `ADD` 22.4 GB/s (as M2d). Qwen3-0.6B not run (expected 59 splits).


**Provider M2c.2 (the Q6_K output head and Q8_0 matmuls on the AI cores), 2026-10-10** (commit `72e0666`; `spacemit_check.py --milestone m2c2`: 29/29 on Qwen3-4B; llama-bench 3 runs, same flags, back to back):

| Model | Mode | pp128 (t/s) | tg128 (t/s) | Perplexity |
|---|---|---|---|---|
| Qwen3-4B Q4_0 | `provider` | 84.50 | 7.84 | 9.7581 (-0.16% vs `cpu`) |
| | `cpu` | 21.14 | 5.68 | 9.7740 |
| | `ime` | 79.30 | 11.03 | - |

- Correctness on the A100 cores (IME kernels, `--full`): Q8_0 against the CPU 14 cases, max NMSE 4.45e-8; Q6_K against the CPU's Q6_K 14 cases (both heads, 151936 rows, at 1, 7 and 128 rows), 5.93e-5 (bound 5e-4); Q6_K against the CPU's Q8_0 of the same values (the requantization alone) 5.07e-7 (bound 5e-6). All identical to the Mac's reference kernels to three digits: both 8-bit IME kernels are exact per 32-value block, and both machines store the same head bytes. `test-backend-ops -o MUL_MAT` passes its 3 claimed cases (Q8_0 2880 x 2880, Q6_K 2048 x 6144, Q6_K 151936 x 2048) with the IME and with the reference kernels: the first `test-backend-ops` cases for an IME matmul. Q4_0 and Q4_1 unchanged (2.81e-5, 9.68e-7, probe 3.02e-7).
- Placement: 74 splits per token, as predicted (M2c.1 75). Provider buffers: model 2349.87 MiB (+394.1 MiB, the head as Q8_0; `ime`'s repacked buffer is 2349.12 MiB), compute 301.75 MiB (from 72: the logits and the head's activations now live in the provider's compute buffer instead of the CPU's).
- The head (`--bench`, 2560 x 151936): 1 row 16.6 ms on the AI cores against 26.0 ms on the X100 cores, i.e. 413 MB at 24.9 GB/s, the same bandwidth as the Q4_0/Q4_1 1-row matmuls: the direct path does not lose for Q8_0 here (ggml-spacemit took path B for it). 4 rows 17.0 ms (memory-bound), 16 rows 27.7 ms (CPU 279 ms).
- Generation: tg128 7.22 -> 7.84 (+8.6%; 138.5 -> 127.6 ms per token; the `cpu` row is 1.8% higher than at M2c.1, so about +7% net of the day). The head saves 9.4 ms and one split about 0.1 ms. `llama-completion` 7.80 t/s against 5.85 with the provider disabled. Prompt processing unchanged (84.50, noise): a prompt batch computes the head for its last token only.
- Accuracy against the CPU's logits (same base file and commands as M2c.1):

| Run | Mean KLD | 99% KLD | Max KLD | Same top token |
|---|---|---|---|---|
| M2c.2, 512-token batches | 0.00326 ± 0.00021 | 0.0242 | 0.223 | 96.72 ± 0.40% |
| M2c.2, `-ub 1` | 0.00220 ± 0.00021 | 0.0161 | 0.287 | 97.16 ± 0.37% |
| M2c.1, 512-token batches | 0.00316 | 0.0244 | 0.241 | 96.86% |
| M2c.1, `-ub 1` | 0.00203 | 0.0175 | 0.287 | 97.30% |
| `ime`, 512-token batches | 0.00339 | 0.0266 | 0.350 | 96.9% |

  The head's requantization adds 0.00010 (512-token batches) and 0.00017 (`-ub 1`) to the mean KLD, under the exit limit of 0.0005 and about the 0.0002 estimated from the logit noise; the tails (99%, maximum) and the same-top-token rate do not move beyond noise. With the same conversions as `ime` (Q4_1 zero points, Q6_K -> Q8_0), the provider is slightly closer to the CPU than `ime` is.
- Remaining generation time (estimates): about 99 ms streaming 2.46 GB of weights at about 25 GB/s, about 9 ms for 74 handoffs, about 20 ms for attention on the X100 cores, the embedding lookup and the other ops; `ime` takes 90.7 ms, so its 1-row matmuls run at 27 GB/s or more. Qwen3-0.6B not run (expected 58 splits).


**1-row matmul strategies, 2026-10-11** (`plan.md` "1-row matmul bandwidth"; commit `a522799`, `flagos-check-spacemit --bench`, test-only `FLAGOS_SPACEMIT_GEMV`; us per 1-row matmul, 30 repetitions, 5 for the head; per-token total = the shapes weighted by how often a Qwen3-4B token runs them: 36 q, 72 k and v, 36 attention outputs, 72 gate and up, 32 + 4 down, 1 head; 2460 MB of weights):

| Strategy | q 2560x4096 | k, v 2560x1024 | attn out 4096x2560 | gate, up 2560x9728 | down 9728x2560 | down Q4_1 | head (Q8_0) 2560x151936 | per token |
|---|---|---|---|---|---|---|---|---|
| direct (128 columns per call, the old path) | 259 | 30 | 262 | 606 | 603 | 586 | 16624 | 102.8 ms |
| direct:32 | 256 | 28 | 253 | 595 | 607 | 571 | 16849 | 101.8 |
| direct:512 (5 of 8 tiles busy at 2560 columns) | 265 | 130 | 217 | 602 | 533 | 565 | 16660 | 106.0 |
| range | 265 | 30 | 258 | 613 | 609 | 574 | 16847 | 103.8 |
| **staged:32 (adopted)** | **232** | 36 | **234** | **518** | 585 | 610 | 14943 | **92.7** |
| staged:64 | 246 | 35 | 243 | 545 | 628 | 710 | 15896 | 98.1 |
| staged:128 | 268 | 36 | 271 | 587 | 639 | 707 | 15770 | 103.3 |
| staged:256 | 285 | 151 | 266 | 633 | 645 | 708 | 15140 | 114.8 |
| staged-range:64 | 252 | 36 | 243 | 567 | 704 | 744 | 14323 | 101.0 |
| prefetch:32 | 334 | 122 | 326 | 695 | 1180 | 1121 | 22091 | 146.9 |
| prefetch:64 | 364 | 94 | 422 | 785 | 1099 | 1107 | 31611 | 162.8 |
| prefetch-range:32 | 320 | 115 | 324 | 701 | 1122 | 1127 | 22537 | 144.9 |
| copy:64 (staging's copies alone) | 232 | 22 | 232 | 550 | 580 | 591 | 15719 | 94.5 |
| compute:64 (its multiplications alone) | 70 | 31 | 73 | 140 | 254 | 309 | 3538 | 30.4 |

- Every strategy gave the same bytes as direct (`flagos-check-spacemit`: 72/72, the copy-only control differs). Each run is its own launch, so the per-token totals include about 250 launches (about 4 ms) the model does not pay.
- k and v (1.5 MB each) stay in a cache across the repetitions (30 us = 49 GB/s); a token reads them once, from DRAM. With 1024 columns, `direct:512` and `staged:256` leave 6 and 4 tiles without work there.
- Reading in `plan.md` (first version, revised with the tile counts below): staging wins because its copies (`memcpy1d`) and the interleaving of copy and multiply phases keep fewer tiles streaming from DRAM at once, which then delivers more in total (`direct:512` with 5 busy tiles: 27 GB/s on the down projection, against 23 with 8).

**Provider with 1-row staging, 2026-10-11** (commit `507cb52`, `staged:32` the 1-row path; `spacemit_check.py --milestone m2c2`: 29/29 on Qwen3-4B; llama-bench 3 runs, same flags):

| Model | Mode | pp128 (t/s) | tg128 (t/s) | Perplexity |
|---|---|---|---|---|
| Qwen3-4B Q4_0 | `provider` | 84.40 | 8.31 | 9.7581 (-0.16% vs `cpu`) |
| | `cpu` | 21.11 | 5.54 | 9.7740 |
| | `ime` | 79.37 | 11.01 | - |

- Generation: tg128 7.84 -> 8.31 (+6.0%; 127.6 -> 120.3 ms per token); `llama-completion` 7.80 -> 8.64 t/s against 5.48 with the provider disabled. The benchmark predicted 10.4 ms less per token (8.54 t/s). Part of the difference: k and v come from a cache in the benchmark but from DRAM in the model, where 4 slabs per tile leave staging little to overlap (estimate 1-1.5 ms, not measured); the `cpu` row is 2.5% below the day before. Perplexity is M2c.2's to the last digit (the matmuls return the same bytes); prompt processing unchanged (the 4-row path); 74 splits.

**Tile counts and the IME reference, 2026-10-11** (same benchmark and units as the strategy table above; `@t`: only the first t tiles work; `ime`: `scripts/ime-gemv-bench.cpp`, upstream's path B, with its default memory `HPAGE` (transparent huge pages through `madvise`) and with `SPACEMIT_MEM_BACKEND=POSIX` (`posix_memalign`: 4 KiB pages on this board, like the provider's buffers)):

| Strategy | q | k, v | attn out | gate, up | down | down Q4_1 | head | per token |
|---|---|---|---|---|---|---|---|---|
| `staged` (32 columns, 8 tiles: the 1-row path) | 223 | 37 | 242 | 510 | 600 | 607 | 15208 | 92.9 ms |
| `staged@6` | 231 | 68 | 246 | 497 | 583 | 597 | 14003 | 92.9 |
| `staged@5` | 245 | 98 | 262 | 531 | 620 | 677 | 14925 | 101.0 |
| `staged@4` | 270 | 122 | 280 | 591 | 664 | 791 | 15735 | 111.3 |
| `staged:64` | 235 | 45 | 263 | 537 | 637 | 700 | 15089 | 98.0 |
| `direct` (128 columns) | 259 | 43 | 254 | 604 | 599 | 583 | 16753 | 103.3 |
| `direct:32` | 255 | 31 | 252 | 594 | 606 | 569 | 16793 | 101.7 |
| `direct:32@6` | 232 | 61 | 232 | 539 | 540 | 560 | 15510 | 94.9 |
| `direct:32@5` | 217 | 88 | 222 | 525 | 524 | 566 | 15944 | 94.9 |
| `direct:32@4` | 217 | 111 | 215 | 544 | 550 | 628 | 17606 | 100.4 |
| `direct:512` | 258 | 115 | 228 | 596 | 531 | 565 | 16669 | 104.6 |
| `copy` (copies alone) | 225 | 27 | 234 | 542 | 561 | 574 | 15603 | 93.3 |
| `copy@6` | 220 | 50 | 225 | 526 | 528 | 549 | 15619 | 92.2 |
| `copy@4` | 233 | 108 | 236 | 577 | 585 | 631 | 18159 | 105.6 |
| `copy@2` | 239 | 110 | 237 | 536 | 525 | 608 | 15028 | 97.9 |
| `compute` (multiplications alone) | 68 | 29 | 68 | 138 | 146 | 206 | 3540 | 26.0 |
| `ime`, huge pages | 206 | 45 | 208 | 462 | 461 | 513 | 14156 | 82.4 |
| `ime`, 4 KiB pages | 227 | 47 | 220 | 457 | 501 | 525 | 13950 | 84.4 |

- Strategy check 54/54 identical to direct. `ime` against ggml-cpu's plain path: NMSE 1.6e-6 (Q4_0), 5.8e-3 (Q4_1: its zero-point conversion, on random weights), 3.8e-5 (Q6_K requantized to Q8_0), so the IME path ran.
- `ime` tg128 (llama-bench, same session, back to back): 11.05 ± 0.02 with huge pages, 10.80 ± 0.01 with 4 KiB pages, i.e. 2.1 ms per token, as the benchmark's 2.0.
- Reading in `plan.md` ("1-row matmul bandwidth", results after adoption).

**1-row round 2: the pairs taking turns, huge pages, 2026-10-11** (commit `2aa32d0`; `flagos-check-spacemit --bench-gemv`, every weight read from DRAM: each run takes the next of enough copies to fill 32 MB; us per 1-row matmul, per-token totals as above; the two page settings in two runs a few minutes apart):

| Strategy | q | k, v | attn out | gate, up | down | down Q4_1 | head | per token |
|---|---|---|---|---|---|---|---|---|
| huge pages: `staged` (not valid, below) | 282 | 533 | 312 | 483 | 569 | 655 | 13789 | 129.1 ms |
| `staged@6` | 242 | 83 | 262 | 511 | 593 | 671 | 13608 | 96.2 |
| **`paired`** | 208 | 68 | 224 | 462 | 496 | 577 | 13125 | **85.0** |
| `paired:64` | 211 | 70 | 226 | 465 | 517 | 597 | 13698 | 86.9 |
| `paired@6` | 216 | 73 | 233 | 477 | 491 | 599 | 13175 | 87.0 |
| `direct` | 230 | 68 | 247 | 523 | 560 | 581 | 14834 | 94.8 |
| `copy` | 206 | 65 | 240 | 481 | 541 | 588 | 14975 | 90.0 |
| `compute` | 76 | 40 | 89 | 139 | 191 | 256 | 3518 | 29.5 |
| 4 KiB pages: `staged` | 236 | 89 | 247 | 604 | 554 | 609 | 14214 | 101.7 |
| `staged@6` | 244 | 81 | 252 | 604 | 578 | 627 | 15448 | 103.6 |
| `paired` | 222 | 73 | 231 | 549 | 527 | 555 | 14897 | 95.0 |
| `paired:64` | 230 | 78 | 239 | 522 | 542 | 547 | 14607 | 94.3 |
| `paired@6` | 230 | 76 | 238 | 510 | 569 | 583 | 13353 | 92.9 |
| `direct` | 271 | 83 | 272 | 617 | 618 | 599 | 16571 | 108.6 |
| `copy` | 237 | 72 | 239 | 559 | 545 | 605 | 15455 | 97.9 |
| `compute` | 76 | 42 | 88 | 139 | 194 | 259 | 3546 | 29.8 |
| `ime`, huge pages, weights rotated | 227 | 92 | 233 | 476 | 513 | 547 | 13218 | 89.2 |

- Both runs pass every check; the strategy check gives 78/78 identical to direct on the AI cores, `paired` with real pair waits (no timeouts). Huge pages: 64 of 64 MiB of a test buffer (`always [madvise] never`), 0 with `FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1`.
- The huge-page `staged` row is not valid: every copy of a weight had its own small output buffer, each on a whole huge page faulted in by its first write (about 0.75 ms each). The first strategy at each shape paid for those inside its timing: 18 for k and v (22 copies), 2 each for q and the attention output; the shapes with 1-3 copies were faulted in during the warm-up. The later rows and the 4 KiB run are not affected. Fixed afterwards: buffers under 2 MiB keep the default pages, and every copy runs once before the timing.
- Generation (llama-bench tg128, back to back; two rounds interleaved, plus the first single runs):

| tg128 (t/s) | huge pages | 4 KiB pages |
|---|---|---|
| `staged` | 8.77, 8.69, 8.75 | 8.66, 8.74, 8.64 |
| `paired` | 9.20, 9.17, 9.18 | 9.06, 9.09 |

  `paired` is 5.0% faster on huge pages (114.4 -> 108.9 ms per token) and 4.6% on 4 KiB pages; huge pages add 0.7% to `staged` (within the noise) and 1.2% to `paired`. The `staged` 4 KiB row matches round 1's configuration: 8.68 today against 8.31 then, so results move about 4% between sessions.
- Pages of the model: during generation (`-n 512`) Rss 2813 MiB, AnonHugePages 2374 MiB, so the 2350 MiB weight buffer lies on huge pages (THP defrag `madvise`). The benchmark's larger page effect (`paired` 95.0 -> 85.0 ms) therefore comes from comparing two runs minutes apart on the shared board; the interleaved tg128 runs measure it.
- The IME reference with its weights rotated: 89.2 ms per token (82.4 unrotated: the old benchmark also served mid-sized shapes partly from caches). `paired` on huge pages is 4.7% faster per token and faster or equal at every shape except the Q4_1 down projection (+5%).
- Reading in `plan.md` ("1-row matmul bandwidth", round 2 results).

**Round 2 adopted, confirmation run, 2026-10-11** (commit `b8187d2`: `paired` the default, buffers from 2 MiB on huge pages; `spacemit_check.py --milestone m2c2`: 29/29 on Qwen3-4B; llama-bench 3 runs, same flags as above):

| Model | Mode | pp128 (t/s) | tg128 (t/s) | Perplexity |
|---|---|---|---|---|
| Qwen3-4B Q4_0 | `provider` | 83.09 | 8.68 | 9.7581 (-0.16% vs `cpu`) |
| | `cpu` | 21.17 | 5.56 | 9.7740 |
| | `ime` | 79.34 | 11.02 | - |

| 1-row benchmark, huge pages | q | k, v | attn out | gate, up | down | down Q4_1 | head | per token |
|---|---|---|---|---|---|---|---|---|
| **`paired` (the default)** | 206 | 86 | 217 | 460 | 494 | 586 | 13942 | **86.6 ms** |
| `staged` | 222 | 82 | 246 | 482 | 568 | 643 | 14247 | 92.5 |
| `direct` | 230 | 83 | 239 | 529 | 562 | 604 | 14809 | 96.1 |

- Every check passes with `paired` as the default (the strategy check 78/78 on the AI cores); huge pages 64 of 64 MiB; 74 splits; perplexity M2c.2's to the last digit (the matmuls return the same bytes).
- Generation against round 1's run of the same runner (`staged`, 4 KiB pages): tg128 8.31 -> 8.68 (+4.5%), `llama-completion` 8.64 -> 8.91 t/s (+3.1%; 32 tokens, the noisiest of the three), with the `cpu` and `ime` rows as then (5.54 and 11.01), so the board ran as then. The runner's llama-bench reads about 5% below the stand-alone `-p 0 -n 128` command in both rounds (`staged` on 4 KiB pages 8.31 against 8.68; `paired` on huge pages 8.68 against 9.18); compare runner with runner and stand-alone with stand-alone.
- pp128 83.1, 1.5% below the last two runs while `cpu` and `ime` held within 0.3%: small, possibly noise. Prompt processing uses paths A and C (gate and up at 128 rows take path C; corrected 2026-10-21), untouched, and the 1-row path only for the head of each batch's last token.
- The benchmark's `staged` row is now valid on huge pages: `paired` is 6.4% below it (round 2 estimated about 92), and 2.9% below the in-tree path's 89.2 (round 2).

**Experiments with the switches, 2026-10-11** (stand-alone `llama-bench -t 8 -p 0 -n 128 -fa 1 -mmp 0 -r 3`; tg128 in t/s, ± the spread of the 3 runs; huge pages unless noted; in the order run):

| Experiment | Results (t/s) |
|---|---|
| 1-row strategy, two rounds interleaved | `paired` 9.00 ± 0.18, 9.18 ± 0.10; `staged` 8.78, 8.79; `direct` 8.65, 8.72 |
| `paired`: columns per slab, tiles | 32 (the default) 9.22; 64: 8.93; 128: 8.72; 6 tiles 9.06; 4 tiles 8.33 |
| pages, two rounds interleaved | huge 9.27, 9.27; 4 KiB 9.16, 9.15 |

- The pairs' turns: `paired` +3.5% over `staged` (110.0 against 113.8 ms per token; +4.4% in the second round, whose `paired` run was steadier; round 2: +5.0%). Staging over direct reads: only +1.2% on huge pages, against 3.7% in the benchmark on huge pages and 6.4% on 4 KiB pages (round 2), likely because direct reads gain more from huge pages. The benchmark overstates differences in generation about 1.5-3 fold.
- The defaults are the best of these variants: slabs of 64 and 128 columns lose 3% and 5%, 6 tiles 2%, 4 tiles 10%. Huge pages: +1.3% (107.9 against 109.2 ms per token; round 2: +1.2%).
- The same configuration (`paired`, huge pages) read 9.00 to 9.27 over the session, rising, so only interleaved comparisons count.

Generation against context depth (2026-10-12, `llama-bench -p 0 -n 32 -d 0,1024,4096 -r 1`, tg32 in t/s):

| Depth | `provider` | `cpu` | `ime` |
|---|---|---|---|
| 0 | 9.70 | 5.87 | 11.05 |
| 1024 | 6.74 | 4.56 | 9.34 |
| 4096 | 1.80 | 1.52 | 6.41 |

- From depth 0 to 4096 a token takes 452 ms more on the provider, 488 on `cpu` and 66 on `ime` (M2d: 445 and 68): attention on the X100 cores costs about 7 times what it costs on the A100 cores. The faster matmuls lifted the provider at depth 0 by 32% since M2d (7.36), at 4096 by 5% (1.72), where it barely beats `cpu`. Beyond about 1k tokens M2e decides generation speed.
- The runner's lower tg128. Its flags stand-alone (`llama-bench -p 128 -n 128 -ub 128 -r 3`), run right after the depth runs above: pp128 83.09 ± 1.13, tg128 8.66 ± 0.03, about what `direct` gives. Follow-ups on a rested board (2026-10-12; llama-bench creates a new context, so a new provider backend and spine-runtime stream, for every test):

| Run | tg128 (t/s) |
|---|---|
| two tg128 tests in one process (`-p 0 -n 128,128`) | 9.16, 9.20 |
| the runner's flags, 30 s of rest before each test (`--delay 30`) | 9.25 (pp128 83.60) |
| the runner's flags, no rest: `paired` (the default), `staged`, `direct` | 9.23, 8.76, 8.69 |

  So neither a later context nor the prompt test before slows generation, and the 1-row path has its TCM after a prompt test (the strategies rank as when run alone). The same command gave 8.66 and 9.23; what differed is what ran before: both slow measurements (8.66 here, the runner's 8.68) started right after minutes of full load on the X100 cores (the `cpu` depth runs with their 4096-token prompts; the runner's CPU-only perplexity, 8 x 512 tokens, right before its llama-bench). The provider still runs part of every token there (attention, the embedding lookup, the thread that drives the AI cores); `ime` runs all its threads on the A100 cores (`cpu_mask: ff00`) and was not slowed in the runner, where it follows the `cpu` llama-bench. Most likely heat on the X100 side, possibly another user's job; clocks not measured. On a rested board the provider's tg128 reads 9.16-9.27 (11 runs over two days; one disturbed run 9.00).
- Prompt processing with and without huge pages (`-p 128 -n 0 -ub 128`, two rounds interleaved): 83.07 and 83.59 against 84.34 and 83.72 on 4 KiB pages, so huge pages cost prompt processing about 0.8% if anything (within the spread of single runs), while generation gains 1.3%.

**Provider M2e.1 (attention on the AI cores), 2026-10-13** (`FLASH_ATTN_EXT` with the in-tree VLEN-1024 kernels, six faults fixed, and C5a; commits `64e6cbd` (C5a) and `715b9bc`; `spacemit_check.py --milestone m2e --depth`: 37/37 on Qwen3-4B; llama-bench 3 runs, same flags as above):

| Model | Mode | pp128 (t/s) | tg128 (t/s) | Perplexity |
|---|---|---|---|---|
| Qwen3-4B Q4_0 | `provider` | 87.70 | 11.27 | 9.782 (+0.08% vs `cpu`; the same with `-fa on` and `-fa auto`) |
| | `cpu` | 21.06 | 5.61 | 9.774 |
| | `ime` | 79.37 | 11.03 | - |

| Depth (`-p 0 -n 32 -d 0,1024,4096 -r 1`; pp512 at depth 4096) | tg32 d0 | d1024 | d4096 | pp512 d4096 |
|---|---|---|---|---|
| `provider` | 11.25 | 9.53 | 6.48 | 24.87 |
| `ime` | 11.09 | 9.34 | 6.34 | 25.13 |
| `cpu` | 5.93 | 4.94 | 1.56 | crashed (below) |
| `provider` before M2e.1 (2026-10-12) | 9.70 | 6.74 | 1.80 | - |

- Correctness on the AI cores: `test-backend-ops -o FLASH_ATTN_EXT` 1324/1324 with the RVV kernels (all claimed cases: ALiBi, softcap, head sizes 40-128, so fix (f) holds and ALiBi and softcap stay claimed) and with the references (13/13 op sets); 1742 cases claimed in all; `flagos-check-spacemit` passes with the RVV and with the reference kernels, its attention checks and the three fault probes included.
- Placement: `-fa auto` keeps flash attention on the provider ("Flash Attention enabled", C5a) and a token takes 2 graph splits (from 74), as predicted. Buffers unchanged (model 2349.87 MiB, compute 301.75 MiB).
- Accuracy against the CPU's logits, as M2c.2:

| Run | Mean KLD | M2c.2 |
|---|---|---|
| 512-token batches (tiled attention kernel) | 0.003354 ± 0.000231 | 0.00326 ± 0.00021 |
| `-ub 1` (one-chunk attention kernels) | 0.002050 ± 0.000155 | 0.00220 ± 0.00021 |

  Both within one standard error of M2c.2's, far inside the exit limit (+0.0005): the copied kernels change the model's outputs no more than the CPU's attention did. Greedy text differs from the CPU's after 143 characters, coherent.
- Speed:
  - Generation +30% (runner tg128 8.68 -> 11.27; `llama-completion` 11.34 against 5.89 with the provider disabled).
  - Prompt processing +5.5% (pp128 87.70).
  - The extra cost per token from depth 0 to 4096 is now 16.0 us per position (`ime` 16.5; before 110 on the X100 cores), so depth 4096 generates 3.6x faster than before.
  - Against `ime` in the same session: generation 1-2% ahead at every depth, pp128 10% ahead, pp512 at depth 4096 1% behind; parity within the noise.
  - The project's targets for Qwen3-4B (M0: tg128 >= 11.1, pp128 >= 82) are met.
  - The runner's tg128 (11.27) now matches `llama-completion`'s (11.34): the provider hardly uses the X100 cores any more, so their state after the heavy runs no longer shows (see `plan.md`, the open note under "1-row matmul bandwidth").
- Attention benchmark (`flagos-check-spacemit --bench`; per call, one layer, 36 per token):

| Shape | Provider | CPU (8 X100) |
|---|---|---|
| 1 token, n_kv 256 | 141 us (5.1 ms per token; 7.4 GB/s of K and V) | 439 us (15.8 ms) |
| 1 token, n_kv 1024 | 475 us (17.1 ms; 8.8 GB/s) | 933 us (33.6 ms) |
| 1 token, n_kv 4096 | 1722 us (62.0 ms; 9.7 GB/s) | 12037 us (433.3 ms) |
| 128 tokens, n_kv 128 | 2952 us (106 ms per batch) | 5555 us (200 ms) |
| 128 tokens, n_kv 4096 | 97963 us (3527 ms per batch) | 142071 us (5115 ms) |

  - With one token per call, attention (the one-chunk kernels) reads K and V at 9.7 GB/s at n_kv 4096, a third of what the 1-row matmuls stream in the same run (28.2 GB/s): the latency bound the M2e design estimated, and M2e.3's headroom (about 62 -> 23 ms per token at 4096, tg32 there about 6.5 -> 8.7, estimate).
  - The tiled kernel at depth is compute-bound and slow: 8.6 GFLOP per layer for 128 tokens at n_kv 4096 in 98 ms, about 88 GFLOP/s (the IME matmuls reach 600-890 at 128 rows; the X100 cores 60). It is about 70% of pp512 at depth 4096 (`ime` has the same kernel and the same speed). The tiled kernel rereads and re-transposes K for every 64-token query tile and for each query head of a KV head. (Found later, L2 below: a third of this time was the kernel's scalar check of every mask value; ggml-spacemit's kernel, adopted in L3, avoids it.)
- Other benchmark lines: 1-row matmuls per token `paired` 87.2 ms (28.2 GB/s), `staged` 93.7, `direct` 99.4 (as in round 2); launch 11.5 us, per node 0.94 us (M2a: 9.9 and 0.79; with one launch per token since M2e.1, this costs about 1 ms per token).
- The CPU-only pp512 at depth 4096 died with SIGSEGV in `ggml_vec_dot_q4_1_q8_1`: the crash of 2026-10-09 again, at the same instruction, this time with the registers (the crash entry under "Provider M2b" above, second occurrence).

**One-chunk attention kernels across three backends, 2026-10-13** (L1 of the ggml-spacemit comparison, `plan.md` M2e.3; `test-backend-ops perf -o FLASH_ATTN_EXT -p 'hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=(4096|8192|16384),nb=1,'`, i.e. Qwen3-4B's shape in generation: 8 KV heads x 4 query heads, head size 128, 1 token; us per run; `tcmrelease` before each run):

| n_kv | provider (huge pages) | provider (`FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1`) | ggml-spacemit `4e782bc` (huge pages, its default) | ggml-spacemit (`SPACEMIT_MEM_BACKEND=posix`) | IME path (`build-ime`, CPU backend, 4 KiB pages) |
|---|---|---|---|---|---|
| 4096 | 2053 | 935 | 2019 | 944 | 946 |
| 8192 | 3891 | 1856 | 3883 | 1873 | 1893 |
| 16384 | 7713 | 3775 | 7810 | 3768 | 3821 |

- The three backends run the same one-chunk kernel (ggml-spacemit's one-chunk code is identical to the in-tree one; ours is that copy with faults (a)-(c) fixed). On 4 KiB pages all three are equal within 2%: 0.23 us per position, 17.9 GB/s of K and V. So ggml-spacemit's one-chunk kernels offer nothing to adopt.
- On huge pages the same runs take 2.1-2.2x as long, at every size. `test-backend-ops` puts the tensors in the tested backend's own buffer (`tests/test-backend-ops.cpp:1553`): huge pages for the provider and for ggml-spacemit, 4 KiB pages for the CPU backend.
  - In these cases the 8 KV heads lie exactly 512 x n_kv bytes apart (2, 4 or 8 MiB; K is allocated as [128, 2 n_kv, 8]), so the 8 tiles stream addresses a multiple of 2 MiB apart.
  - On 2 MiB pages the physical addresses keep that alignment, so the streams collide in the same cache sets or memory banks. On 4 KiB pages the physical frames are scattered.
  - The model is not affected: its KV cache is in llama.cpp's CPU buffer on 4 KiB pages, with the heads 256 B apart (`k3-hardware.md` §2).
- Against our benchmark: the same kernel reads K and V at 17.9 GB/s when each tile streams its own contiguous region (here), but at 9.7 GB/s on the real cache's layout (`bench attn` above, also 4 KiB pages), where every tile reads its head's 256 B of each 2 KiB cell and so touches every page.
  - The harnesses differ: `test-backend-ops` reuses its tensors, the benchmark rotates copies. But at n_kv 16384 the tensors are 64 MiB and the time per position stays flat, so both measure DRAM.
  - So the cache layout's access pattern costs the one-chunk kernels about 45% of their bandwidth: what M2e.3's candidate (a) removes (`plan.md`).

**Tiled attention kernel: ggml-spacemit's in the provider, 2026-10-13** (L2 of the ggml-spacemit comparison, `plan.md` M2e.3; commit `68207bf`, test-only `FLAGOS_SPACEMIT_ATTN=trim`; `flagos-check-spacemit --expect-device --bench-attn`, the two kernels taking turns twice, the faster turn of each; Qwen3-4B's heads, us per call, one layer):

| Shape | in-tree kernel | ggml-spacemit's | change |
|---|---|---|---|
| 128 tokens, n_kv 128 | 2929 | 2160 | -26.3% |
| 512 tokens, n_kv 512 | 36569 | 25408 | -30.5% |
| 128 tokens after 3968 (n_kv 4096) | 98641 | 65992 | -33.1% |
| 512 tokens after 4096 (n_kv 4608) | 438415 | 294116 | -32.9% |

| llama-bench, Qwen3-4B Q4_0 (`-t 8 -fa 1 -mmp 0`; two rounds of `-r 3` taking turns, depth `-r 1`) | in-tree kernel | ggml-spacemit's | change |
|---|---|---|---|
| pp128 | 87.15, 87.19 | 89.14, 89.21 | +2.3% |
| pp512 | 85.27, 84.94 | 90.82, 90.95 | +6.8% |
| pp512 at depth 4096 | 24.86 | 33.22 | +33.6% (`ime` 25.13 in M2e.1's session) |

- Correct: outputs bit-identical to the in-tree kernel's in all three tiled cases (NMSE against the CPU 8.4e-7); `test-backend-ops -o FLASH_ATTN_EXT` passes every case with it.
- Why it is faster: mostly not the trimming.
  - The in-tree kernel reads all 4096 mask values of every 64 x 64 tile, one at a time, to see whether it can skip the tile. ggml-spacemit's walks each row back from its end and stops at the first visible value: one read per row of a visible tile.
  - A fit gives about 16 ns per value read (32 cycles at 2.0 GHz) plus 32 us per diagonal tile from the trimming. Fitted on two shapes, it predicts the other two within 2% and 17%, and the model's gains within 0.4 points (pp128 +1.9%, pp512 +7.2%, depth +33.7%).
  - So that check was a third of the in-tree kernel's time, at every depth. The tiled kernel at depth (M2e.1 above: 88 GFLOP/s) now runs at about 130 GFLOP/s.
  - Why a scalar loop over the mask copy (in TCM) costs 32 cycles a value is not verified: scalar loads from TCM, or the AI cores' scalar floating-point path. L4 (`plan.md`) tests the figure.
  - The in-tree IME path runs the in-tree kernel, so its prompt processing at depth carries the same cost (for the SpacemiT report).
- Decision (L3, by the rule fixed before measuring): adopted; `forward_tiled` is ggml-spacemit's kernel since then.

**Tiled attention kernel: the remaining scalar mask reads (L4), 2026-10-13** (commit `a832349`, test-only `FLAGOS_SPACEMIT_ATTN=vscan`, the same harnesses; us per call):

| Shape | scalar counts (L3's kernel) | vector counts | change |
|---|---|---|---|
| 128 tokens, n_kv 128 | 2152 | 1713 | -20.4% |
| 512 tokens, n_kv 512 | 25630 | 18677 | -27.1% |
| 128 tokens after 3968 (n_kv 4096) | 66206 | 63567 | -4.0% |
| 512 tokens after 4096 (n_kv 4608) | 296075 | 279597 | -5.6% |

| llama-bench, as above | scalar counts | vector counts | change |
|---|---|---|---|
| pp128 | 89.48, 89.14 | 92.33, 91.93 | +3.2% |
| pp512 | 92.49, 91.07 | 95.40, 95.77 | +4.2% |
| pp512 at depth 4096 | 33.17 | 34.49 | +4.0% |

- Outputs bit-identical (3/3 tiled cases); test-backend-ops passes every case with it. The scalar-count column repeats L2's ggml-spacemit column within 1%: the adoption kept its speed.
- Per tile (fit over the four shapes): the vector counts save about 4.5 us on a fully visible tile, 28 us on a diagonal one, 50 us on a fully masked one.
  - So each of a visible tile's 64 scalar reads (one per row, 256 B apart) cost about 70 ns; in sequence a read cost about 16 ns (L2; `k3-hardware.md` §2).
  - From these savings the model's gains were predicted within 0.5 points before llama-bench ran.
- Decision (L4, by its rule): adopted.
- L3 and L4 together, against M2e.1's in-tree kernel (two sessions): pp128 87.2 -> 92.1, pp512 85.1 -> 95.6, pp512 at depth 4096 24.9 -> 34.5 (`ime` 25.13).

**Acceptance run after L3, 2026-10-13** (commit `a832349`: ggml-spacemit's tiled kernel the default, `vscan` behind its switch; `spacemit_check.py --milestone m2e` on Qwen3-4B, without `--depth`: 36/36, M2e.1's 37 less the depth check):
- Accuracy: perplexity 9.782 with `-fa on` and with `-fa auto` (CPU 9.774, +0.08%); mean KLD 0.003354 ± 0.000231 (512-token batches) and 0.002050 ± 0.000155 (`-ub 1`). All four are M2e.1's to the last digit, and the greedy text leaves the CPU's at the same character (143): what bit-identical attention outputs predict. L4's outputs are bit-identical to these, so this holds for it too.
- Placement unchanged: `-fa auto` keeps flash attention on the provider (C5a), 2 graph splits per token, buffers 2349.87 / 301.75 MiB, no TCM block left held.
- Speed (llama-bench, 3 runs): provider pp128 89.41, tg128 11.28 (M2e.1: 87.70, 11.27); `ime` 79.39, 11.00; `cpu` 20.87, 5.59. `llama-completion` 11.41 t/s against 5.6 with the provider disabled.
- The benchmark measured the `vscan` switch a third time: -20.7% (128 tokens, depth 0), -26.9% (512, depth 0), -4.4% (128 after 3968), -5.9% (512 after 4096), as in L4's runs. 1-row matmuls per token: `paired` 85.9 ms (28.7 GB/s), `staged` 91.5, `direct` 96.7.

**M2e's final state (M2e.1 with L3 and L4), 2026-10-13** (commit `633284c`, code as `fc1240a`; `spacemit_check.py --milestone m2e --depth` on Qwen3-4B Q4_0: 37/37):

| Qwen3-4B Q4_0 | provider | `ime` | `cpu` | provider against `ime` |
|---|---|---|---|---|
| pp128 | 92.27 | 79.13 | 21.38 | +16.6% |
| tg128 | 11.30 | 11.03 | 5.92 | +2.4% |
| tg32 at depth 0 / 1024 / 4096 | 11.38 / 9.56 / 6.49 | 11.12 / 9.37 / 6.40 | 5.68 / 4.67 / 1.53 | +2.3 / +2.0 / +1.4% |
| pp512 at depth 4096 | 34.56 | 25.13 | 11.89 | +37.5% |

- Every prediction held: pp128 about 92, generation unchanged (M2e.1: tg128 11.27, tg32 11.25 / 9.53 / 6.48), pp512 at depth 4096 about 34.5 (M2e.1: 24.87, +39%).
- Accuracy: perplexity 9.782 with `-fa on` and `-fa auto` (+0.08% against the CPU), mean KLD 0.003354 / 0.002050, greedy text off the CPU's at character 143: M2e.1's to the last digit. Placement unchanged: C5a, 2 graph splits per token, the same buffers, no TCM block left held.
- Against the CPU: pp128 4.3x, tg128 1.9x, tg32 at depth 4096 4.2x, pp512 at depth 4096 2.9x.
- The CPU-only pp512 at depth 4096 completed this time (11.89 t/s), where M2e.1's run crashed (above): the crash is intermittent, two occurrences, then one clean run of the same test.
- At depth 4096 a token takes 154 ms: attention 65.6 ms (the benchmark's 1-token line at n_kv 4096) and the 1-row matmuls about 88 ms. Attention in generation is now 43% of a token there: M2e.3's target.
- Noise, not changes: the X100 side had one-off outliers (the `cpu` column of the 1-row Q4_0 2560 x 9728 matmul 3944 us, twice its usual; the CPU-only `llama-completion` 5.95 t/s against 5.6 in the run before). On the provider side the benchmark moved by 1-3% against the run before (1-row matmuls per token 87.9 against 85.9 ms, 1-token attention at n_kv 4096 1822 against 1786 us, one 1-row line 467 against 421 us) while tg128 held (11.30 against 11.28).

**Qwen3-0.6B at M2e's final state, 2026-10-13** (commit `633284c`; `spacemit_check.py --milestone m2e` with its default model, without `--build` or `--depth`: 34/34, the 4B run's checks less configure and build):

| Qwen3-0.6B Q4_0 | provider | `ime` | `cpu` | provider against `ime` |
|---|---|---|---|---|
| pp128 | 654.10 | 553.20 | 150.01 | +18.2% |
| tg128 | 55.60 | 54.25 | 26.83 | +2.5% |

- M2e's exit for 0.6B:
  - pp128 at least 594 (ggml-spacemit's best, M0): met, +10%.
  - tg128 at least the IME path's: met in the same session, +2.5%. The M0 figure (57.3) is not reached, but the IME path measured 5% under its own M0 figure in this session, the known variation between sessions.
- Against M2d (0.6B): pp128 369.32 -> 654.10 (+77%), tg128 29.03 -> 55.60 (+92%). `llama-completion` 56.34 t/s against 30.73 with the provider off.
- Generation is mostly weight streaming: 18.0 ms per token, of which the 395 MiB of weights at the benchmark's 28.7 GB/s take about 14.4 ms. Attention at this context is about 2 ms (estimate from the 4B benchmark).
- Accuracy passes, with the thinnest margin so far:
  - perplexity 14.5068 against the CPU's 14.3795, +0.89% (bound 1%; M2d +0.61%, 4B now +0.08%);
  - mean KLD 0.014712 ± 0.000669 (512-token batches) and 0.012409 ± 0.000565 (`-ub 1`), the first for 0.6B, 4-6x 4B's, with the same order (batches above `-ub 1`) as on 4B;
  - greedy text off the CPU's after 88 characters (4B: 143), coherent.

  The upstream IME path has the same error (measured the same day, `build-ime` with `-fa on` over the same 8 x 512 tokens against a fresh CPU base):
  - its mean KLD is 0.014046 ± 0.000592, against the provider's 0.014712 ± 0.000669: the same within one standard error;
  - its perplexity ratio (llama-perplexity's KLD mode) is 1.016377 ± 0.007407.

  So 0.6B's larger gap belongs to the IME approach (int8 activations, shared by both), not to what the provider adds (requantized head, Q4_1 conversion, attention on the AI cores).
- With 8 chunks the perplexity ratio on 0.6B carries ±0.74% at one standard error. The runner's 1% bound cannot tell +0.89% from +1.64%, and by the ratio the IME path itself would miss it. On 0.6B, KLD is the measure to judge accuracy by.
- The CPU run behind that base gave PPL 14.3706, where the runner's identical CPU run gave 14.3795: CPU runs are not bit-reproducible here (0.06%; cause not checked), far below the noise that matters above.
- Placement as on 4B: C5a, 2 graph splits per token; model 394.71 MiB, compute 298.75 MiB.

**M2e.3 round 1: the grouped one-chunk kernel's settings, 2026-10-14** (`plan.md` "M2e.3 design"; test-only `FLAGOS_SPACEMIT_ATTN`; `flagos-check-spacemit --expect-device --bench-attn`, every setting and the default taking turns twice, the faster turn of each; us per one-token call, Qwen3-4B's heads; in brackets against the default):

| n_kv | default | `stream` | `stream,burst` | `compute` | `cached` | `burst` | `cached,burst` |
|---|---|---|---|---|---|---|---|
| 256 | 139 | 117 (-15.8%) | 101 (-27.2%) | 71 (-48.7%) | 133 (-4.3%) | 125 (-9.8%) | 128 (-7.6%) |
| 1024 | 476 | 387 (-18.6%) | 338 (-29.0%) | 159 (-66.6%) | 450 (-5.5%) | 413 (-13.1%) | 424 (-10.9%) |
| 4096 | 1633 | 1307 (-19.9%) | 1511 (-7.4%) | 506 (-69.0%) | 1771 (+8.5%) | 1813 (+11.0%) | 1655 (+1.3%) |
| 8192 | 3404 | 2547 (-25.2%) | 3197 (-6.1%) | 984 (-71.1%) | 3654 (+7.3%) | 3815 (+12.1%) | 3298 (-3.1%) |

| tg32, llama-bench (Qwen3-4B Q4_0, `-fa 1 -mmp 0`, two rounds taking turns) | depth 1024 | depth 4096 |
|---|---|---|
| default | 9.60, 9.59 | 6.59, 6.55 |
| `cached` | 9.58, 9.63 (+0.1%) | 6.61, 6.62 (+0.7%) |
| `burst` | 9.57, 9.54 (-0.4%) | 6.49, 6.47 (-1.4%) |
| `cached,burst` | 9.62, 9.63 (+0.3%) | 6.66, 6.65 (+1.3%) |

- Correct: `cached`, `burst` and `cached,burst` give the default's outputs byte for byte on the four one-token cases; test-backend-ops passes every case with each.
- The kernel is bound by its staging. At n_kv 4096, staging alone (`stream`) takes 80% of the call and the arithmetic alone (`compute`) 31%; at 8192, 75% and 29%. The two add up to 104-111%, the excess being the per-position mask reads and loop that both keep, so the phases run one after the other, without overlap.
- `cached`: -4 to -6% per call at short context, nothing worth having at depth (+0.7% tg32 at depth 4096). Its +7-9% per call at n_kv 4096 and 8192 is not borne out by the model, and is not consistent with `cached,burst` either: noise at those two shapes.
  - In this kernel the 12 scalar reads of TCM per position cost little: unlike the old tiled kernel's mask scan, where each read decided the next, they are independent and interleaved with vector work.
- `burst`: with 8 rows' loads in flight the staging gets faster over a short cache (`stream,burst` against `stream`: -14% at 256 positions, -13% at 1024) and slower over a long one (+16% at 4096, +26% at 8192). In the model it costs 1.4% at depth 4096. Why is not known (more requests in flight at 2 KiB strides may load the address translation or the DRAM).
- The model's differences are smaller than the benchmark's at long context: for effects of a few percent there, the model is the better judge.
- Decision (by the rule fixed before measuring): nothing adopted. None reaches 5% less time at n_kv 4096, and the model moves by at most 1.4%. The settings and their switch were removed the same day; the benchmark keeps the one-token shape at n_kv 8192.

**M2e.3 round 2: one-token attention split over the tiles, adopted, 2026-10-14** (`plan.md` "M2e.3 design"; measured at commit `e06cf90`, where the split sat behind `FLAGOS_SPACEMIT_ATTN=split[,paired]` and the default was the one-chunk kernels; `flagos-check-spacemit --expect-device --bench-attn`, both settings and the default taking turns twice, the faster turn of each; us per one-token call, Qwen3-4B's heads; in brackets against the one-chunk kernels):

| n_kv | one-chunk kernels | `split` | `split,paired` |
|---|---|---|---|
| 256 (below the split's 512: the one-chunk kernels in all three) | 134 | 134 | 135 |
| 1024 | 479 (8.8 GB/s) | 374 (-22.0%) | 385 (-19.6%) |
| 4096 | 1743 (9.6 GB/s) | 1303 (-25.2%, 12.9 GB/s) | 1132 (-35.1%, 14.8 GB/s) |
| 8192 | 3108 (10.8 GB/s) | 2366 (-23.9%) | 2116 (-31.9%, 15.9 GB/s) |

| tg32, llama-bench (Qwen3-4B Q4_0, `-fa 1 -mmp 0`, two rounds taking turns) | depth 1024 | depth 4096 | ms per token at depth 4096 |
|---|---|---|---|
| one-chunk kernels | 9.57, 9.59 | 6.53, 6.50 | 153.5 |
| `split` | 9.66, 9.58 (+0.4%) | 7.52, 7.51 (+15.3%) | 133.1 |
| `split,paired` | 9.89, 9.86 (+3.1%) | 7.86, 7.79 (+20.1%) | 127.8 |

| Mean KLD against the CPU at `-ub 1 -b 512` (one sequence, one token per ubatch) | one-chunk kernels | `split,paired` |
|---|---|---|
| Qwen3-4B, 8 chunks | 0.002050 ± 0.000155 (PPL ratio 1.006083) | 0.002001 ± 0.000138 (1.006909) |
| Qwen3-0.6B, 32 chunks | 0.014032 ± 0.000381 (1.016157) | 0.013979 ± 0.000341 (1.016668) |

- Correct: the check's one-token cases (8 per setting) within the bounds, max NMSE 2.3e-5, and all 16 outputs differ from the one-chunk kernels' (the split ran); test-backend-ops passes every case with each setting.
- Speed: both settings pass the rule (at least 20% less at n_kv 4096, no shape slower), though below the predictions (-30 to -40% and -50 to -60%). At n_kv 4096 K and V are read at 12.9 and 14.8 GB/s, against 28 for the 1-row matmuls: either the new arithmetic (a K transpose per KV head, V summed in F32) costs more than round 1's `compute` measured, or 8 tiles copying at once do not reach 28 GB/s. Not separated.
- In the model the gains are larger than the benchmark suggested at depth 4096 (estimated +12% and +18%). At depth 1024 `split` gains almost nothing, where the benchmark gave -22% at n_kv 1024. Likely cause, not measured: there the cache holds 1280 positions in 20 blocks, 17 of them with visible positions, so dealt round-robin tile 0 takes 3 blocks and every other tile 2. The turns recover part of it (+3.1%).
- Accuracy: with the split the KLD is slightly lower on both models (V summed in F32) and the PPL ratio slightly higher, both well within one standard error.
- The first KLD runs of round 2, with llama-perplexity's default batch, gave the one-chunk kernels' numbers to the last digit with both settings: the split had not run.
  - llama-perplexity runs n_batch / n_ctx = 4 sequences at once, each in its own cache stream (`kv_unified` off). For several streams, llama.cpp builds ubatches with `split_equal` (`src/llama-batch.cpp:547`), which at `-ub 1` puts a token of each of 2 sequences into every ubatch. The split takes one stream only, so it did not run.
  - Checked on the Mac with a CPU build and `LLAMA_BATCH_DEBUG=1`: for 2048 tokens at `-ub 1`, 1024 ubatches of 2 tokens in 2 streams. With `-b 512` (one sequence) or `-kvu` (one stream) every ubatch is one token. At the default `-ub 512` a ubatch holds 128 tokens of each of the 4 sequences.
  - So every earlier `-ub 1` run (M2c.1, M2c.2, M2e.1, the runner's) gave the matmuls 2 rows (path C), not the 1-row path that generation uses. Attention was one token per stream there, the one-chunk kernels as recorded.
  - On Qwen3-4B one-token ubatches give the one-chunk kernels' KLD and PPL ratio of the 2-token runs to the last digit (0.002050, 1.006083): the 1-row matmuls produce the same logits, so the recorded `-ub 1` values stand for generation's path. On 0.6B the CPU base moved with the batch (PPL 16.9033 with 4 sequences, 16.9058 with one), so those two runs cannot be compared.
  - The runner's and the recipe's `-ub 1` runs take `-b 512` since. The rule's reference for 4B, M2e's 0.002050, is the same value measured with one-token ubatches in the same run.
- Decision (by the rule fixed before measuring, `plan.md`): adopted. `split,paired` is 4.1% ahead of `split` at depth 4096, more than the rule's 2% margin, so the split with the turns is the default for the calls it takes: one token and one stream from n_kv 512, llama.cpp's cache layout, no ALiBi or softcap. `FLAGOS_SPACEMIT_ATTN=split` (without the turns, the same outputs) and `one-chunk` (the kernels before) stay for comparisons, and the benchmark adds a one-token shape at n_kv 512, where the split starts.
- Not covered: several sequences generating at once (several streams, or several tokens per call) keep the one-chunk kernels, as ggml-cpu keeps its own split of the positions to one stream (`ggml-cpu/ops.cpp:9115`).

**Acceptance run after M2e.3 round 2, 2026-10-14** (code as the adoption commit: the split with the turns the default; `spacemit_check.py --milestone m2e --depth` on Qwen3-4B Q4_0: 35/35; with its default model, Qwen3-0.6B, without `--depth`: 34/34):

| Qwen3-4B Q4_0 | provider | `ime` | `cpu` | provider at M2e's final state |
|---|---|---|---|---|
| pp128 | 92.53 | 79.29 | 21.14 | 92.27 |
| tg128 | 11.25 | 11.02 | 5.71 | 11.30 |
| tg32 at depth 0 / 1024 / 4096 | 11.30 / 9.93 / 7.83 | 11.13 / 9.35 / 6.33 | 5.86 / 4.71 / 1.59 | 11.38 / 9.56 / 6.49 |
| pp512 at depth 4096 | 34.63 | 25.14 | 12.02 | 34.56 |

- Generation at depth, against M2e's final state: +3.9% at depth 1024 and +20.6% at 4096. Against the IME path in the same session: +1.5 / +6.2 / +23.7% at depths 0 / 1024 / 4096.
  - A token at depth 4096 takes 127.7 ms (154.1 before). Each position of context costs 9.6 us per token (16.2 before; the IME path 16.6).
  - Depth 0 and prompt processing are unchanged within the noise: the split runs only for one token per call, from n_kv 512.
- Accuracy (4B): perplexity 9.782 against the CPU's 9.7772 (+0.05%), with `-fa on` and `-fa auto`. Mean KLD 0.003364 ± 0.000231 (512-token batches) and 0.002021 ± 0.000142 (`-ub 1 -b 512`).
  - The provider's perplexity is M2e's to the last digit, since its 512-token batches do not take the split. The CPU's moved (9.774 before), and the two KLD values moved with it, by 1-2e-5.
  - Greedy text leaves the CPU's at character 143, as before.
- Placement unchanged: C5a keeps flash attention on the provider, 2 graph splits per token, no TCM block left held. `llama-completion` 11.26 t/s against 5.88 with the provider off.
- Benchmark: one-token attention 1109 us at n_kv 4096 (one-chunk kernels 1771, -37%) and 2080 at 8192 (3332, -38%). The 1-row matmuls take 86.8 ms per token (28.4 GB/s), as before.
- Qwen3-0.6B (34/34): pp128 667.19 and tg128 55.78 against the IME path's 550.78 and 54.27 (+21.1%, +2.8%; M2e's final state 654.10 and 55.60).
  - Perplexity 14.5068 against the CPU's 14.3786 (+0.89%): the provider's value is unchanged, the CPU's moved again (14.3795 before).
  - Mean KLD 0.014773 ± 0.000741 (512-token batches) and 0.012145 ± 0.000541 (`-ub 1 -b 512`, the first with one-token ubatches on 0.6B, so not comparable with M2e's 0.012409).
  - Greedy text leaves the CPU's after 88 characters, as before; `llama-completion` 56.87 t/s against 28.52.
- Two findings of the benchmark, the same in the check run before it:
  - At n_kv 512 every tile has one block, and the default is slower than the one-chunk kernels: 262 against 252 us (+3.9%; 260 against 253 in the run before). The split without the turns takes 210 (-20%). That breaks round 2's "no one-token shape slower"; it costs about 2% of a token, at depths 256-511 only. Left as it is (user decision 2026-10-14); the remedy would be no turns when no tile has more than one block (`plan.md`).
  - The split without the turns varies between runs where the turns do not. At n_kv 4096 it took 1303, 1093 and 1172 us in three runs; the turns 1132, 1108 and 1109; the one-chunk kernels 1743, 1753 and 1771. So at depth the turns are the robust choice.

Manual checks:
- Always report `pp` and `tg` separately; repeat and interleave runs and note mean and median.
- Confirm IME is active: a `CPU_RISCV64_SPACEMIT` model buffer in `./build-ime/bin/llama-perplexity ... -lv 4` output (buffer name from `ggml/src/ggml-cpu/spacemit/ime.cpp:1477`).
- Where threads run: `ps -L -o tid,psr,comm -p $(pgrep llama-bench)` (`psr` 8-15 = A100).
- Compare speeds only within one session, interleaved and repeated: between sessions or harnesses tg128 moves by about 4-5%.

Experiment switches of the provider (environment variables; test-only unless noted; the context of each is in `plan.md`):

| Variable | Applies to | Effect |
|---|---|---|
| `FLAGOS_SPACEMIT_DISABLE=1` | everything | no SpacemiT device: llama.cpp runs on the X100 cores (the runner's `cpu` mode) |
| `FLAGOS_SPACEMIT_GEMV=<s>[:<c>][@<t>]` | 1-row matmuls, read at every call | `<s>`: `paired` (default), `staged` (no turns), `direct` (from DRAM, c = 128), `copy` (copies only, wrong results); `<c>`: columns per slab, a multiple of 32 (shrinks to fit the TCM); `@<t>`: only the first t tiles work |
| `FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1` | buffers (Linux) | the default 4 KiB pages; otherwise buffers from 2 MiB lie on huge pages |
| `FLAGOS_SPACEMIT_ATTN=split` or `one-chunk` | one-token attention from n_kv 512 (one stream, llama.cpp's cache layout, no ALiBi or softcap), read whenever an op is looked up | the default since 2026-10-14 (M2e.3 round 2, `plan.md`): the KV positions split over the tiles, the two tiles of a pair taking turns between staging and arithmetic, and the partial results merged in a second step (`paired` names it); `split`: the same without the turns (identical outputs); `one-chunk`: the one-chunk kernels as before round 2 (V summed in F16, so the outputs differ in rounding) |
| `FLAGOS_SPACEMIT_TEST_REFERENCE=1` | kernels on the K3 | the portable reference kernels on the AI cores (slow); tells a kernel error from an orchestration error |
| `FLAGOS_SPACEMIT_STREAM=per-call` | executor on the K3 | a new spine-runtime stream per graph, releasing the AI cores in between (experiment switch, plan §2.4) |
| `FLAGOS_SPACEMIT_MM=rows`, `spacemit` or `fit` | matmuls of several rows, read at every call (`fusion-and-splitting.md` §7.3) | `split` (the default): ime.cpp's choice of path A, path C where ime.cpp takes path B; `rows`: path A whenever every tile gets 4 rows; `spacemit`: path B where `split` takes path C and TCM holds a 32-column slab (SpacemiT's rule); `fit`: path B where TCM holds the batch's activations next to a slab, else path A from 32 rows, else path C. The same bytes in all four |
| `FLAGOS_SPACEMIT_QUANTIZE=shared` | matmuls (`fusion-and-splitting.md` §7.2) | one quantization of the activations for Q, K and V and one for gate and up |
| `FLAGOS_SPACEMIT_TEST_OP_STATS=<file>` | every graph (`fusion-and-splitting.md` §7.1) | per-op counters, written to the file when the process ends |
| `FLAGOS_SPACEMIT_TEST_TCM=<bytes>` | host executor (Mac) | a TCM stand-in per tile, so the 1-row path and paths A and B run; `flagos-check-spacemit` sets 393216 for its 1-row and path checks |
| `FLAGOS_SPACEMIT_TEST_THREADS=1` | host executor (Mac) | the tiles of each launch on threads, so pair waits wait (with the TCM stand-in: everything as on the AI cores) |
| `FLAGOS_SPACEMIT_TEST_FAIL_NODE=<n>` | tests | tile 0 reports a failure at step n |

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
