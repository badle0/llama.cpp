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

For experiment E1 (`plan.md` M0.9), the device-type switch was applied before building (`docs/flagos-k3/scripts/spacemit-device-type.patch`, removed 2026-11-02 with the scripts of decided experiments; in git at `cad674fda`). Without `GGML_SPACEMIT_DEVICE_TYPE=gpu` the build behaves exactly like stock (ACCEL).

## 6. Build D - FlagOS with the SpacemiT provider (`build-flagos/`, K3)

The SpacemiT provider (`plan.md` M1-M2d, M2c). Kept separate from `build/`, which stays the plain-CPU baseline.
```bash
python3 ggml/src/ggml-flagos/providers/spacemit/tools/spacemit_check.py --build
```
The script configures `build-flagos/` with `-DGGML_FLAGOS=ON -DGGML_FLAGOS_DENGLIN=OFF -DGGML_FLAGOS_AMD=OFF -DGGML_FLAGOS_SPACEMIT=ON`, builds the provider, the FlagOS check tools (with `flagos-check-spacemit`), `test-backend-ops`, `llama-completion`, `llama-bench`, `llama-perplexity` and (since 2026-11-03) `llama-server`, runs the acceptance checks (the header of `spacemit_check.py` lists them) and prints PASS/FAIL per check (INFO lines for measurements); logs go to `build-flagos/logs-<model file stem>/` (since 2026-11-02; `<milestone>-logs-<model file stem>/` from 2026-10-27, `<milestone>-logs/` before), with every PASS/INFO line also in its `summary.txt`, whose last line reads `ACCEPTANCE: <passed>/<checks> checks passed`. From M2a the provider links spine-runtime from `~/spine-runtime` (override with `--spert-dir`). Without `--build` it only reruns the checks. Options (until 2026-11-02 also `--milestone m1` to `m2e` and `igpu`, each milestone's expectations; removed with the earlier milestones' code, `igpu`'s expectations are the runner's only ones, so the commands recorded below with `--milestone igpu` run today without it): `--depth` (also tg32 at depths 0, 1024 and 4096 and pp512 at depth 4096, for the provider, the CPU and the IME build; about an hour more), `--model` (default `~/models/Qwen3-0.6B-Q4_0.gguf`), `--ppl-text`, `--ime-build`, `--skip-support`, `--build-dir`, `--spert-dir`, `--tcm-dir` (default `~/tcmtest`, for the TCM checks), `--segv <segv.so>` (preloads the crash reporter `scripts/segv.c` into the model, perplexity and llama-bench runs), `--skip-server` (leaves out the llama-server checks), `--server-baseline <server.json>` (also checks that llama-server's replies equal a saved `logs-<model>/server.json`). m2e adds: the model run's default `-fa auto` must keep flash attention on the provider (C5a) and a token must take 2 graph splits; perplexity with `-fa on` and with `-fa auto`; mean KLD against the CPU at the default batch and at `-ub 1` (the CPU run writes the base file, deleted afterwards; compare the values with M2c.2's by hand). Since 2026-11-03 (`plan.md` "The server check, the baseline before S2") the runner also starts `llama-server` (`-c 4096 -t 8 -lv 4 --cache-reuse 16`, its defaults otherwise: 4 slots in one unified KV cache, idle slots saved to a host-RAM prompt cache) and checks the placement, a greedy reply equal to `llama-completion`'s, a follow-up turn and two interleaved conversations (reloaded from the prompt cache) against the same prompts from scratch (since the fix after the first K3 run, `plan.md`: the first token's KL divergence over the tokens both top-5 lists share at most 0.1, each top token in the other's top 5), and KV shifting; every reply goes to `logs-<model>/server.json`, the baseline S2 must reproduce, and the server's TTFT and time per token to INFO lines. Expected counts since then: 48 on Qwen3-0.6B with `--build` (46 without), 47 on Qwen3-4B with `--depth` (46 without), one more each with `--server-baseline`.

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
wget -c -q --show-progress $MS/Qwen3-0.6B-GGUF/resolve/master/Qwen3-0.6B-Q8_0.gguf   # 639447744 bytes (fusion-and-splitting.md §7.3)
wget -c -q --show-progress $MS/Qwen3-14B-GGUF/resolve/master/Qwen3-14B-Q4_0.gguf     # 8543001984 bytes (§7.3, step 4)
# unsloth's Qwen3-8B repo has no Q4_0; bartowski's has unsloth's recipe (Q4_1 ffn_down in layers 0-3, Q6_K head)
wget -c -q --show-progress https://modelscope.cn/models/bartowski/Qwen_Qwen3-8B-GGUF/resolve/master/Qwen_Qwen3-8B-Q4_0.gguf   # 4787332640 bytes
```
List a repo's files: `wget -qO- "https://modelscope.cn/api/v1/models/unsloth/Qwen3-4B-GGUF/repo/files?Revision=master&Recursive=true"`. The 4B file matches SpacemiT's table (2.21 GiB); their 0.6B file was about 6 MiB smaller, so compare 0.6B numbers to their table loosely.
SpacemiT's doc benchmarks Qwen3-0.6B Q4_0 and qwen35 2B Q4_1 (`docs/build-riscv64-spacemit.md`); A100 IME supports Q2_K–Q6_K, Q4_0/1, Q5_0/1, Q8_0.

Baselines (M0.5, M0.6): `scripts/m0-baselines.sh <model>` ran every mode interleaved, then perplexity, and saved `summary.txt` (modes and switches described in its header; removed 2026-11-02 with the scripts of decided experiments, in git at `cad674fda`).

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
  - **Third and fourth dumps, 2026-11-01** (the S1 step 3c acceptance run on Qwen3-4B with the crash reporter preloaded; `build-flagos/igpu-logs-Qwen3-4B-Q4_0/bench-cpu-depth.err` and `bench-cpu-depth-pp.err`). The CPU-only `llama-bench` runs of tg32 at depth 4096 and of pp512 at depth 4096 (provider disabled, 8 threads on the X100 cores) each got SIGSEGV with fault address 0 at the same instruction, `libggml-cpu.so.0 + 0x99316` (`ggml_vec_dot_q4_1_q8_1 + 0x40`), called from `ggml_compute_forward_mul_mat + 0x75c` on an OpenMP worker. The registers are again those of one Q4_1 row's dot product of K 9728 (`a0` = 304 blocks, `s4` = 6080 bytes per row), at block 41 and at block 211 (`t1` = 0x29 and 0xd3; `a3 = s6 - 6080 + 20 t1 + 4` and `a5 = s0 + 4 + 36 t1` checked in both), and in both `a2` is 0 where the code set it to `a5 + 16`. So four times at this instruction (2026-10-09, 10-13 and twice 11-01), three of them with registers, each with `a2` alone wrong, every one in a long CPU-only run under full X100 load (a perplexity run and three depth benchmarks); the signal-11 deaths of 2026-10-27 and 10-29, also CPU-only, left no dump.
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

**Acceptance run after the default change, 2026-10-26** (build `65280e5`: the matmuls of several rows take `fit,staged`, path B without the pair turns where TCM holds the activations next to a slab, else path A from 32 rows with its leftover rows dealt, else path C, `fusion-and-splitting.md` §7.3; Q4_1 requantized to Q8_0 at load since 2026-10-23; `spacemit_check.py --milestone m2e --build --depth` on Qwen3-4B Q4_0: 37/37; with its default model, Qwen3-0.6B, without `--build` or `--depth`: 34/34; runs of 2026-10-25 to 26):

| Qwen3-4B Q4_0 | provider | `ime` | `cpu` | provider after M2e.3 round 2 |
|---|---|---|---|---|
| pp128 | 106.70 | 79.26 | 20.98 | 92.53 |
| tg128 | 11.00 | 10.98 | 5.51 | 11.25 |
| tg32 at depth 0 / 1024 / 4096 | 11.08 / 9.64 / 7.72 | 11.09 / 9.30 / 6.35 | 5.48 / 4.46 / 1.57 | 11.30 / 9.93 / 7.83 |
| pp512 at depth 4096 | 34.68 | 25.11 | 11.72 | 34.63 |

- pp128 +34.6% against the IME path in the same session (+16.7% after M2e.3 round 2): at 128 rows gate and up leave path C for path A. Generation and the depth runs within 3% of M2e.3's (the 1-row matmuls and pp512 keep their paths; the Q4_1 fix took tg128 2.0% in its session). At depth 4096 generation +21.6% and pp512 +38.1% against the IME path.
- Accuracy (4B): perplexity 9.7736 against the CPU's 9.774 (-0.00%), with `-fa on` and `-fa auto`, the Q4_1 fix's to the last digit. Mean KLD 0.002489 ± 0.000156 (512-token batches) and 0.001229 ± 0.000058 (`-ub 1 -b 512`). Greedy text leaves the CPU's after 143 characters, as before.
- Qwen3-0.6B (34/34): pp128 705.50 and tg128 55.16 against the IME path's 542.13 and 54.34 (+30.1%, +1.5%; after M2e.3 round 2 667.19 and 55.78). Perplexity 14.4471 against the CPU's 14.3706 (+0.53%): the provider's is the fix's, the CPU's moved (14.3598 on 2026-10-24). Mean KLD 0.004722 ± 0.000221 and 0.002533 ± 0.000089. Greedy text identical to the CPU's (after 88 characters before; not attributed, `fusion-and-splitting.md` §7.3).
- Placement unchanged: C5a keeps flash attention on the provider, 2 graph splits per token, no TCM block left held. `llama-completion` 11.13 t/s against 5.54 with the provider off (0.6B: 55.19 against 27.39).

**The IGPU switch, 2026-10-27** (D3; `PROJECT.md` item 22, `plan.md` §2.3 and §2.8; checked on the Mac's simulated device, accepted on the K3 the same day): the provider reports IGPU by default, so llama.cpp offloads every layer and puts the KV cache in the provider's buffer; the cache ops that needs are in: the layout rule keeps `cache_k_*`, `cache_v_*`, `cache_r_*` and `cache_s_*` plain, `SET_ROWS` writes F32 rows into Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL and BF16 caches, `ROPE` runs in place on F16 (the context shift), `CPY` writes F32 rows into those types (a quantized cache's shift; its Hadamard matmuls and attention over a quantized cache stay on the CPU with copies, so `-fa on` is the supported form with `-ctk`/`-ctv`). Mac: `flagos-check-spacemit --expect-device` passes every check (its `cache` line: SET_ROWS, CPY, ROPE and a Q8_0 shift through the scheduler within NMSE 1e-6 of the CPU, the quantized bytes those of ggml's reference quantizers), as does the same with `FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=accel` (the device reports ACCEL) and with `FLAGOS_SPACEMIT_TEST_THREADS=1`; `test-backend-ops` SET_ROWS 180/180, ROPE 273/273, CPY 15/15, ROPE_SET_ROWS 24/24, FLASH_ATTN_EXT 1324/1324, MUL_MAT 3/3; ThreadSanitizer clean. On the K3 (inside tmux; the runner's default milestone is `igpu` now):

```bash
~/tcmtest/tcmrelease --apply
python3 ggml/src/ggml-flagos/providers/spacemit/tools/spacemit_check.py --milestone igpu --build --depth --model ~/models/Qwen3-4B-Q4_0.gguf
python3 ggml/src/ggml-flagos/providers/spacemit/tools/spacemit_check.py --milestone igpu
```

Pass when every check passes (the new ones: every layer and the KV cache on the provider, the accel switch reporting ACCEL, the Q8_0-cache and context-shift runs completing with and without the provider), the perplexity at 512-token batches is 2026-10-26's to the last digit, tg128 and pp128 are within 1% of 2026-10-26 (4B 11.00 / 106.70, 0.6B 55.16 / 705.50), pp512 at depth 4096 up to 4% faster allowed (E2). Recorded, not gated: the Q8_0-cache run's speed.

Results on the K3, 2026-10-27 (build `695ad6d`, the switch's commit; spine-runtime 0.6.3; the 4B runner twice, once with `--build`, since its summary scrolled out of tmux; the 0.6B runner once; the depth tests one run each):

| provider, IGPU | Qwen3-4B Q4_0 | 2026-10-26 (ACCEL) | Qwen3-0.6B Q4_0 | 2026-10-26 (ACCEL) |
|---|---|---|---|---|
| checks | 36/40, see below | 37/37 | 39/39 | 34/34 |
| pp128 / tg128 | 106.26 / 10.99 | 106.70 / 11.00 | 690.49 / 55.02 (runner); 703.67 / 55.37 (recheck) | 705.50 / 55.16 |
| `ime` in the same session | 79.30 / 10.97 | 79.26 / 10.98 | 539.04 / 53.92 | 542.13 / 54.34 |
| tg32 at depth 0 / 1024 / 4096 | 11.05 / 9.64 / 7.70 (first run 11.09 / 9.67 / 7.68) | 11.08 / 9.64 / 7.72 | | |
| pp512 at depth 4096 | 36.03 (first run 35.96; `ime` 25.10) | 34.68 | | |
| perplexity, `-fa on` and `-fa auto` | 9.7736 against the CPU's 9.774 | the same | 14.4471 against the CPU's 14.3706 | the same |
| mean KLD, 512-token batches / `-ub 1 -b 512` | 0.002489 / 0.001229 | the same | 0.004722 / 0.002533 | the same |
| Q8_0 KV cache, `llama-completion -n 32 -c 512 -fa on` (recorded) | 6.95 t/s against the CPU's 5.26 | | 28.92 against 27.74 | |

- Placement as the switch intends: 37/37 (0.6B 29/29) layers offloaded, the KV cache in the provider's buffer (5760 MiB on 4B at the default context), 2 graph splits per token, flash attention kept under `-fa auto`, no TCM block left held. The Q8_0-cache and context-shift runs complete with and without the provider (399 evaluations after the first token on both), the text identical to the CPU's on 0.6B and leaving it after 143 characters on 4B, as the F16 run does and did. The accel switch reports ACCEL with its warning.
- Accuracy: perplexity and both KLDs are 2026-10-26's to the last digit on both models (identical logits, as E2 measured for IGPU against ACCEL).
- Speed: within the 1% rule on 4B (pp128 -0.4%, tg128 -0.1%; the depth runs within 1% of 2026-10-26's but pp512 at depth 4096, +3.9%, the gain E2 had measured). On 0.6B the runner's pp128 came out 2.1% under 2026-10-26's, so the two types were rechecked back to back in one session (`llama-bench -p 128 -n 128 -ub 128 -fa 1 -r 5`): IGPU 703.67 / 55.37, ACCEL 702.73 / 54.82, IGPU again 695.60 / 55.07; the types agree within 0.2% on pp128 and IGPU's tg128 is 0.5-1.0% above ACCEL's, and the IGPU runs themselves span 1.2%, so the runner's 690 was session spread and the recheck's 703.67 is within 0.3% of 705.50. Met.
- The 4B runner's four failures are one event: its CPU-only perplexity run (`FLAGOS_SPACEMIT_DISABLE=1`, no provider device) died with signal 11 after "calculating perplexity over 8 chunks", so the two perplexity comparisons had no CPU value and the two KLD runs read an empty logits base. The same CPU run redone by hand completed (9.774) and the two KLD runs gave 2026-10-26's values. This is the intermittent CPU-backend crash of question 8 (2026-10-09, 2026-10-13; a NaN 2026-10-24), the fourth time; by the code the switch cannot reach a run with the provider disabled, not tested further.
- The Q8_0 cache runs attention on the CPU with copies of the cache views, as designed: 37% slower than the F16 cache on 4B (6.95 against 11.08 in the same runner), 48% on 0.6B (28.92 against 55.3); a measurement of the stores-only design, not a target.
- Found on the way: the runner's log directory was per milestone, so the 0.6B run overwrote the 4B run's logs but the depth ones, and its summary went to the terminal only; changed after the run: a log directory per milestone and model, the summary also in its `summary.txt`. The runner's `--bench` now runs the matmul benchmark with `split` and the default only (`--bench-mm` keeps all seven settings; the five others are decided and recorded in `fusion-and-splitting.md` §7.3), so the acceptance run is shorter; the runs above still had all seven.

**SpacemiT's Q4_0 scale fix, 2026-10-28** (`PROJECT.md` item 26, `plan.md` R11; checked on the Mac, the K3 run pending): the three Q4_0 activation quantizers (`quantize_a_nrow_i8_hp_ref`, `quantize_a_row_i8_hp`, `quantize_a_4row_i8_hp` in `flagos-spacemit-ime-kernels.cpp`) take SpacemiT's stable factor, from the fork's in-tree copy (`spacemit-com/llama.cpp` `mtmd-backend` `27acf9d`, PR #30, "fix qwen2.5-3b q4_0 hp scale overflow", 2026-08-05; not in upstream llama.cpp, nor in `ggml-spacemit`'s own copy at `855a53b`). A block of 256 activations has one F16 scale and, per sub-block of 32, an F16 ratio to it, and the IME kernels sum a block's sub-block products in F16 (`gemm_kernel_i8i4_hp_*`); the fix stores the ratios times the factor and the block scale divided by it, the same product with 1/factor times the headroom in those sums. SpacemiT's factor is 0.1; ours is 0.125 (marked "flagos"): a power of two scales every F16 intermediate exactly, so outputs stay bit-identical to before wherever the old sums stayed in range, while 0.1 rounds every stored scale anew. On the Mac's simulated device (reference kernels):

| check | before | SpacemiT's 0.1 | ours, 0.125 | bound |
|---|---|---|---|---|
| Q4_0 matmul, one large activation sub-block against seven tiny ones (weights of scale 0.5, every nibble 15; new probe), 1 / 4 / 5 rows | inf / inf / inf | 2.17e-05 each | 3.63e-07 each | 5e-4 |
| Q4_0 matmul, 24 random cases, max NMSE | 2.56e-05 | 2.61e-05 | 2.56e-05 | 5e-4 |
| Qwen3-like layer, max NMSE | 5.40e-04 | 5.44e-04 | 5.40e-04 | 1e-3 |
| every other number of the check program | | | unchanged | |

`test-backend-ops -o MUL_MAT` 3/3 as before (its claimed cases are Q8_0 and Q6_K; its Q4_0 shapes have output sizes the IME layout refuses, so the check program is the Q4_0 coverage).

On the K3 (inside tmux; the bundle from `695ad6d`, the board's build):

```bash
~/tcmtest/tcmrelease --apply
python3 ggml/src/ggml-flagos/providers/spacemit/tools/spacemit_check.py --milestone igpu --build --depth --model ~/models/Qwen3-4B-Q4_0.gguf
python3 ggml/src/ggml-flagos/providers/spacemit/tools/spacemit_check.py --milestone igpu
```

Pass rule, fixed before the run: every check passes, the probe included (on the K3 it runs the RVV quantizers and the IME kernels; question 8's CPU-only crash, if it recurs, is redone by hand as on 2026-10-27); identical logits, i.e. the perplexity at 512-token batches and both mean KLDs 2026-10-27's to the last digit (4B 9.7736, 0.002489 / 0.001229; 0.6B 14.4471, 0.004722 / 0.002533); tg128 and pp128 within 1% of 2026-10-27's (4B 10.99 / 106.26, 0.6B 55.37 / 703.67). Predicted: all three hold (by the code, only an intermediate falling below F16's normal range, eight times sooner than before, can change a digit; the real models' sums do not overflow, or 2026-10-27's runs would have shown inf). If the digits differ, the difference is examined before adoption; a KLD within 2% of 2026-10-27's would then be the fallback rule.

Results on the K3, 2026-10-28 (commit `ec0681a`; `spacemit_check.py --milestone igpu --build --depth` on Qwen3-4B Q4_0; the 0.6B runner's output not yet recorded): 42/42 checks, the probe included; placement as before (37/37 layers, the KV cache on the provider, 2 splits, flash attention kept, no TCM block held); speed within the 1% rule (pp128 106.81 against 106.26, tg128 11.01 against 10.99; tg32 at depths 0 / 1024 / 4096 11.14 / 9.59 / 7.72 against 11.05 / 9.64 / 7.70; pp512 at depth 4096 36.04 against 36.03; the IME path in the same session 79.02 / 10.99; the Q8_0 cache 7.21, recorded). **The logits are not identical, so the fix is not accepted yet:**

| measure | 2026-10-27 | 2026-10-28 |
|---|---|---|
| perplexity, provider / CPU | 9.7736 / 9.7740 | 9.7983 / 9.7793 (+0.25% / +0.05%) |
| mean KLD, 512-token batches | 0.002489 ± 0.000156 | 0.007697 ± 0.002394 (3.1x; the error bar 15x) |
| mean KLD, `-ub 1 -b 512` | 0.001229 ± 0.000058 | 0.002858 ± 0.000106 (2.3x; the error bar 2x) |

Read from the numbers, not yet tested: the 15x error bar at a 3x mean is a heavy tail, a few tokens far off rather than a uniform shift; it sits in the batched run (the 4-row quantizer, paths A, B and C), since both KLD runs share one CPU base and the one-token run moved less; the CPU reference itself moved by 0.05% with no CPU code changed, as Qwen3-0.6B's did between 2026-10-24 and 10-26 (question 8), so part of the change may be session variance. A mechanism the prediction underweighted: the factor makes the kernels' F16 products 8 times smaller, so products that were small normals become subnormals, with less precision (SpacemiT's 0.1 has the same exposure). What settles it: one session, one CPU base, the three quantizers' factor set by a test-only switch to 1.0 (the old code), 0.125 and 0.1, the provider's perplexity and both KLDs for each; if 1.0 restores the 0.0025 class, the factor is the cause (then: revert with the overflow documented, a milder factor, or a per-block factor applied only to blocks that would overflow, which leaves every other block bit-identical since the kernels only multiply stored values); if 1.0 and 0.125 agree, the session is the cause and the fix stands with these numbers as the baseline.

Qwen3-0.6B the same day (`--milestone igpu`, 39/39): placement as before (29/29 layers, the KV cache on the provider, 2 splits); pp128 692.74 and tg128 54.93 against 2026-10-27's runner 690.49 / 55.02 and recheck 703.67 / 55.37 (within that day's session spread of 2%); the IME path in the same session 538.39 / 54.20; greedy text identical to the CPU's in all three runs; the Q8_0 cache 26.17. Accuracy: perplexity 14.4218 against the CPU's 14.3439 (+0.54%; 2026-10-27: 14.4471 against 14.3706, +0.53%): the ratio unchanged, both absolute values down 0.19% together, the CPU's third value in three sessions (14.3598, 14.3706, 14.3439) with the same CPU code and the same `~/ppl.txt`; mean KLD 0.005610 ± 0.000513 at 512-token batches (0.004722 ± 0.000221: +19%, the error bar 2.3x) and 0.009795 ± 0.000366 at one token (0.002533 ± 0.000089: 3.9x, the error bar 4x). So the two models disagree on where the change sits: on 4B the batched KLD moved most, on 0.6B the one-token KLD, and on 0.6B the perplexity ratio did not move at all. That pattern fits a CPU reference that is not reproducible between runs on the board better than a single quantizer mechanism, and the KLD base is that same CPU run's logits. First experiment, before the factor comparison and needing no code: the CPU perplexity twice in one session on Qwen3-0.6B, the two KLD base files compared byte for byte and the two perplexities to the last digit; then the provider twice. If the CPU runs differ, the board's CPU path is nondeterministic (question 8) and every KLD acceptance needs its base checked; if they agree, the factor comparison follows in one session.

**The CPU path's nondeterminism under several threads, 2026-10-28 and 29 (investigated and closed by the user's decision; the step-by-step runs are in the board's logs, not here).** Found: two consecutive 8-thread CPU-only runs of `llama-perplexity` on Qwen3-0.6B Q4_0 differ (14.3505 against 14.3654), while single-thread runs give 14.3706 twice with byte-identical logits, the correct value; four of five 8-thread runs were wrong. Established by tests, every run with the provider disabled, "forced switching" meaning 8 threads pinned to 7 X100 cores: the fault follows context switching, not load (forced switching wrong in every run, 7 threads on 7 cores 3 of 16, single threads 0 of 14, 2 threads on 1 core 0 of 3, OpenMP and ggml's own thread pool alike); it needs vector code (a build with no vector instruction in our binaries agrees 4 of 4 under the forcing); and it needs ggml-cpu's Q4_1 matmul path: the Q8_0 file of the same model agrees 3 of 3, and of three files requantized from it with `llama-quantize --allow-requantize`, only the one with `ffn_down` in Q4_1 differs, while Q4_0 everywhere and a Q6_K head agree 3 of 3 each. The kernels are upstream's RISC-V `quantize_row_q8_1` and `ggml_vec_dot_q4_1_q8_1` (`ggml-cpu/arch/riscv/quants.c:73-120` and `:277-326`, unchanged in the fork), the site of the two identical crash dumps above. A hash of every tensor after every op (`k3-opdiff.cpp`, never committed, removed from the board 2026-11-01, the one copy on the Mac in `~/k3-preempt-test/`; three runs of 4,096 tokens on the Q4_1 file) shows the first differing op of every pair of runs to be an `ffn_out-N` matmul of the first ubatch with identical inputs, and exactly one element of its 524,288 wrong by a large amount (0.93 for -0.17, 0.43 for 0.045, 1.50 for -0.061), the neighbours intact, the wrong value nowhere else in the output; a block-by-block replication of the kernel from the dumped inputs, exact to 2e-7, matches none of 5,804 single-event hypotheses per strike (accumulator reset or restored at any block, a block skipped or doubled, a term lost, a stale reduction, a neighbouring column's, row's or block's data). So one call of the dot product per strike computes with data that is not among its inputs, a stale read of the work buffer after the barrier or a read through a wrong address: a memory-system fault, not arithmetic (an inference, not isolated). Not reproduced without llama.cpp in five harness versions of 250,000-370,000 switches each (`k3-preempt-test-v3.c`, never committed, built on the board in `/root`, the Mac's copy in `~/k3-preempt-test/`): vector registers and CSRs across switches (0 deviations in 25 billion steps), the two kernels copied verbatim and the library's own code by `dlsym`, ggml's exact cross-core pattern (block ranges of every activation row into one shared buffer, its chunking), and the real inputs of a diverging op; every version kept the same bytes in its buffer from pass to pass, so a stale read could not show, and the version alternating two data sets was not run. Implications: (1) every CPU reference on the board is a single-thread run, done twice and byte-compared, with the provider disabled (`PROJECT.md` §6); an 8-thread CPU run of a model with Q4_1 tensors is wrong at random, which looks exactly like a provider accuracy regression; (2) the absolute KLDs recorded before 2026-10-28 were taken against 8-thread bases and are suspect, same-session comparisons against one base keep their meaning, and the provider reproduces to every digit; (3) the provider and the IME path are untouched: they never run these kernels (Q4_1 is requantized to Q8_0 at load; the IME kernels run on the AI cores) and their tiles are never switched out; (4) for SpacemiT (question 8): the recipe (the Q4_1 test file with `k3-opdiff`, 8 threads on 7 cores, a strike within the first ubatch of every run), the kernels, the crash dumps, the board's kernel 6.18.3; the kernel source is upstream llama.cpp's, the fault the platform's.

**The one-token KLD of the factor against the verified base, 2026-10-28.** The value: 0.009888 ± 0.000374 (`-ub 1 -b 512`), against 0.002533 ± 0.000089 with the old quantizer on 2026-10-27's correct base, and 0.009795 against the bad base of the morning: so the base was not what moved this number; **the factor costs the 1-row path 3.9x in one-token KLD on Qwen3-0.6B and the batched paths +19%**, and the fix as taken is not acceptable. A mechanism, read from the reference kernel's arithmetic and not yet isolated: the kernels form each sub-block's scale product `d × ratio` in F16 and the factor makes it 8 times smaller, so products that were small normals (below about 4.9e-4) become F16 subnormals (below 6.1e-5), with up to ten times less relative precision; the 1-row kernel would feel it most if it keeps those products in F16 throughout. The overflow the fix guards against needs a large weight scale and a dominant sub-block (the probe: scale 0.5, nibble 15, one sub-block of 16 full-scale activations; by the reference's arithmetic a sub-block's product can reach `ratio × d × 60,960`, so the old code overflows from about d 0.13 at the worst alignment), which SpacemiT met on Qwen2.5-3B and Qwen3 does not meet. Next: a test-only switch for the factor (1, 0.5, 0.25, 0.125, 0.1, and a per-block form that scales only blocks whose largest sub-block ratio exceeds a threshold, which leaves ordinary blocks bit-identical since the kernels only multiply stored scales), the KLD curve on 0.6B against `kb3.bin` in one session, then the decision; the revert of the fix is the fallback, with the overflow recorded as a known limit of SpacemiT's kernels.

**The switch, 2026-10-28 (Mac):** `FLAGOS_SPACEMIT_TEST_Q4_0_SCALE` (the switch table below). Verified on the Mac's simulated device under every value: `1` gives the pre-fix numbers bit for bit (the 24 random Q4_0 cases' max NMSE 2.56e-05 and the layer's 5.40e-04, as before the fix) with the probe at inf reported as expected and not a failure; `0.5`, `0.125`, `adaptive:4` and `adaptive:2` pass every check with the probe at 3.63e-07; `bogus` warns and keeps 0.125; the new policy check finds the stored block scale and sub-block ratio equal to what each policy gives, bit for bit, and the adaptive form leaving a flat block at 1 while scaling the dominant one. The K3 run, on Qwen3-0.6B against the verified base `kb3.bin`, in one session, inside tmux:

```bash
cd /root/llama.cpp-flagos && M=~/models/Qwen3-0.6B-Q4_0.gguf && B=build-flagos/bin/llama-perplexity && ~/tcmtest/tcmrelease --apply && for f in 1 0.5 0.25 0.125 0.1 adaptive:4 adaptive:2; do FLAGOS_SPACEMIT_TEST_Q4_0_SCALE=$f $B -m $M -f ~/ppl.txt -c 512 --chunks 8 -t 8 -fa on --kl-divergence-base ~/kb3.bin --kl-divergence > ~/q4-$f-b.log 2>&1; FLAGOS_SPACEMIT_TEST_Q4_0_SCALE=$f $B -m $M -f ~/ppl.txt -c 512 -b 512 --chunks 8 -t 8 -fa on -ub 1 --kl-divergence-base ~/kb3.bin --kl-divergence > ~/q4-$f-1.log 2>&1; echo "$f: batches $(grep -ho 'Mean *KLD: *[0-9.]* *± *[0-9.]*' ~/q4-$f-b.log), one token $(grep -ho 'Mean *KLD: *[0-9.]* *± *[0-9.]*' ~/q4-$f-1.log)"; done
```

Predictions, fixed before the run: `1` returns to 2026-10-27's class, batched about 0.0047 and one-token about 0.0025, within the error bars (the base is a different correct run); `0.125` reproduces today's 0.005599 and 0.009888 exactly (the provider is deterministic); `0.5` and `0.25` lie between, worsening monotonically as the factor falls; `0.1` at or above `0.125`'s; `adaptive:4` within `1`'s error bars if few blocks have a sub-block over four times the average (unmeasured); `adaptive:2` between `adaptive:4` and `0.125`. Decision rule: the policy kept is the one closest to `1` among those that pass the probe; if `adaptive:4` is within `1`'s error bars on both measures it becomes the default and the fix stands in that form; if no scaled policy is within 2% of `1`, the quantizer reverts to `1` (the fix off), the probe stays as the record of the overflow, and the hazard goes to SpacemiT's report. Then Qwen3-4B's acceptance against a verified single-thread base (two `-t 1` CPU runs, byte-compared).

**The curve, run on the K3 the same day** (Qwen3-0.6B Q4_0, one session, every run against the verified base `kb3.bin`; the switch's commit):

| factor | KLD, 512-token batches | KLD, one token per ubatch |
|---|---|---|
| 1 (the quantizer before the fix) | 0.004722 ± 0.000221 | 0.002533 ± 0.000089 |
| 0.5 | 0.004676 ± 0.000228 | 0.002885 ± 0.000125 (+14%) |
| 0.25 | 0.004737 ± 0.000237 | 0.004194 ± 0.000203 (+66%) |
| 0.125 (the fix as taken) | 0.005599 ± 0.000520 (+19%) | 0.009888 ± 0.000374 (3.9x) |
| 0.1 (SpacemiT's) | 0.006646 ± 0.000425 (+41%) | 0.013960 ± 0.000523 (5.5x) |
| adaptive:4 | 0.005070 ± 0.000369 | 0.008075 ± 0.000299 (3.2x) |
| adaptive:2 | 0.005522 ± 0.000497 | 0.009925 ± 0.000406 (3.9x) |

Every prediction held: factor 1 gives 2026-10-27's two values to the last digit against a different base (so the provider is deterministic, that day's base was correct, and factor 1 is the pre-fix quantizer bit for bit); 0.125 reproduces the morning's two values exactly; the one-token KLD rises monotonically as the factor falls and the batched KLD holds to 0.25. Not predicted: `adaptive:4` sits near 0.125, so on this model most activation blocks have a sub-block above four times the average; the per-block form helps nothing, because the blocks with a dominant sub-block are exactly where the factor costs. **Decision by the rule fixed before the run: no scaled policy is within 2% of 1 on both measures (0.5, the mildest that passes the probe, costs the one-token KLD 14%), so the quantizer reverts to 1, the default of the switch since this commit; the fix is not adopted.** Checked on the Mac after the change: every number of the check program equals the pre-fix run's, the probe at inf reported as expected. For SpacemiT's report (question 8): their factor 0.1 costs Qwen3-0.6B 5.5x in one-token KLD on the IME path's own arithmetic; the overflow it guards needs a large weight scale with a dominant sub-block (met on their Qwen2.5-3B, by none of our models), and the proper fix is single precision for the scale products inside the kernels. For a model that does overflow, `FLAGOS_SPACEMIT_TEST_Q4_0_SCALE=0.5` gives 2x headroom at the measured cost. Qwen3-4B's acceptance of the pre-fix quantizer stands as 2026-10-27's (its logits unchanged by construction); the 4B acceptance runs that follow use a verified single-thread base.

**S1 step 1 on the K3, 2026-10-29** (commit `c4fd4a92e`; `PROJECT.md` item 29; the runner's milestone `igpu`, Qwen3-0.6B then Qwen3-4B with `--depth`, the CPU references single-threaded). The model side is as predicted on both models: every layer and the KV cache on the provider, 2 graph splits per token, the perplexity and both KLDs of 2026-10-27 to the last digit (4B 9.7736 against the CPU's 9.774, mean KLD 0.002489 / 0.001229 at 512-token batches / one token; 0.6B 14.4471, 0.004722 / 0.002533), and the speed within the 1% rule: 4B pp128 105.96 and tg128 11.00 (2026-10-27: 106.26 / 10.99), tg32 at depths 0 / 1024 / 4096 11.13 / 9.59 / 7.71 (11.05 / 9.64 / 7.70), pp512 at depth 4096 35.98 (36.03); 0.6B pp128 705.51 and tg128 54.83. The CPU-only depth benchmark on 4B lost its pp512-at-depth-4096 cell to a signal 11 (question 8's intermittent CPU crash, the fifth time; the provider disabled). Qwen3-0.6B 36/41 and Qwen3-4B 36/40, the failures one event on both: `flagos-check-spacemit` dies with signal 11 in every pass (the default, the reference pass, the `accel` pass and `--bench`), which also fails the runner's loader report, read from the first pass's lost stdout. Nothing of S1 is involved: the crash is in `check_mul_mat`, after the loader and op checks have run, in the test-only Q4_0 scale-policy check of 2026-10-28. Located on the board without gdb (not installed): `stdbuf -o0 -e0` to see the last lines, the crash reporter `scripts/segv.c` preloaded (pc 0x20, the fault address itself, on the main thread; `ra` in the check program just after `jal q4_0_quantize_row@plt` by `addr2line` and `objdump`; `t0 = t1 = 0x20`), `LD_BIND_NOW=1` crashing identically (so not a lazy binding), and the reference pass dying on a tile with pc 0x80. Cause, read in `objdump -d libggml-flagos.so.0`: GCC 15.2.0 (Bianbu 15.2.0-16ubuntu1bb2, `-O3 -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii`) inlines the RVV quantizer `quantize_a_row_i8_hp` into the wrapper `q4_0_quantize_row` and, because of the exported `q4_0_block_factor` call inside that quantizer's per-block loop (the `adaptive:<T>` form's per-block decision, commit `ac9580721`), spills vector registers into a vlenb-sized stack frame; the wrapper's `reference` branch then releases the frame and tail-jumps to `quantize_a_nrow_i8_hp_ref<1>` with `csrr t0,vlenb; add sp,sp,t0; mv t1,t0; ... jr t1`, so the jump goes to the address vlenb: 0x20 on the X100 cores, 0x80 on the A100 cores (the reference pass). The reference kernel itself (207 instructions, no `jr`, no vector code) is clean. A compiler fault in the epilogue; the trigger ours. Who runs the reference branch: the check program's policy check and `FLAGOS_SPACEMIT_TEST_REFERENCE=1`; the default path (the RVV quantizer, `reference == false`) returns before the faulty epilogue, which is why the model runs, the perplexities and the benchmarks of this session were right; the last K3 run of the check program before this one (2026-10-28, commit `ec0681a`, 42/42 with every pass) had the factor as a constant, with no call in the loop. Fix (user's decision, 2026-10-29): the adaptive form deleted (the curve above had rejected it), each quantizer reads the factor once at entry into a local, no call inside any quantizer loop; verified on the K3 first by the wrapper's disassembly, then by the check program's passes and the runner. Checked on the Mac: the host build with no warning, the check program under the default, 0.5, 0.125 and two rejected values (which warn and use 1), `test-backend-ops` MUL_MAT 3/3, the riscv64 syntax check 0 errors. On the K3 (2026-10-30, the patch on `c4fd4a92e`): the rebuild warns only on three pre-existing F32-to-`_Float16` narrowings in the copied reference kernel `gemm_kernel_i8i4_hp_mrow_ref` (lines 701, 710, 723, since commit `6577170`; the K3's GCC 15 warns, the Mac's compiler does not); the two wrappers' disassembly has no `vlenb` read into the frame and every `jr t1` follows an `auipc t1; ld t1` of the reference function's GOT entry with nothing writing `t1` in between (the only `vlenb` reads go to `a5` and `s0`, the quantizer's own 128-versus-32 branch), and one `q4_0_scale_factor` call per wrapper at entry; then `flagos-check-spacemit --expect-device --full` in the default, reference and `accel` passes and `--bench` all return 0 with "all checks passed", the loader line in every pass (`libggml-flagos.so.0: api 1, struct_size 176, 14 kernels, 5 names by get_proc_address, tcm 393216 per tile`), the policy check on both blocks and the probe at inf as expected. The runner on Qwen3-0.6B the same day (`--milestone igpu`, no rebuild): 0 FAIL, the four check-program passes and the loader report passing, the model, Q8_0-cache and context-shift runs complete with output identical to the CPU's, every layer and the KV cache on the provider, perplexity 14.4471 against the CPU's 14.3706 and KLD 0.004722 / 0.002533, yesterday's to the last digit (no kernel arithmetic changed); tg128 54.82 (yesterday 54.83), pp128 695.47 against 705.51 (-1.4%), settled as session spread by an A/B in one session on the board (the library without the patch against the library with it, three `llama-bench` runs each): 696.4 / 694.5 / 691.4 against 695.7 / 696.8 / 689.4, the same 694 on average, so the removed call costs nothing. Qwen3-4B's run is not repeated: no kernel arithmetic changed on its path, and its logits and speed stand as yesterday's. For SpacemiT's report (question 8): the compiler and flags, and the epilogue pattern.

**S1 step 2 on the K3, 2026-10-30** (the patch on `8d976d1cc`; `plan.md` "S1 step 2 spec"; `PROJECT.md` item 30; the runner's milestone `igpu`, Qwen3-0.6B with `--build` twice, then Qwen3-4B with `--depth`; the CPU references single-threaded). As predicted: the contract table changes no answer and no step list, so the logits are 2026-10-27's to the last digit on both models (4B perplexity 9.7736 against the CPU's 9.774, mean KLD 0.002489 / 0.001229 at 512-token batches / one token; 0.6B 14.4471 against 14.3706, 0.004722 / 0.002533), every layer and the KV cache on the provider, and the speed within the 1% rule: 4B pp128 106.55 and tg128 11.00 (step 1's session 105.96 / 11.00), tg32 at depths 0 / 1024 / 4096 11.03 / 9.70 / 7.71 (11.13 / 9.59 / 7.71), pp512 at depth 4096 36.06 (35.98); 0.6B pp128 693.95 and 691.26, tg128 54.61 and 54.66 in the two sessions (step 1's 695.47 / 54.82). The loader report reads `14 kernels, 6 names by get_proc_address` in every pass, and the check program's new `contracts 12 bound to 14 kernels` line passes in every pass. Counts: Qwen3-0.6B 41/41 and 41/41; Qwen3-4B 39/40, the failure `benchmark ran: rc=-11`: the `--bench` pass of `flagos-check-spacemit` died with signal 11 somewhere after its `cache` check had started printing (its stdout ends at a pipe buffer's edge, its stderr has no message, and `dmesg` has no segfault line for it, unlike the 2026-10-09 crash). Not reproduced: the same binary's `--bench` passed in both 0.6B sessions, three times in a row under the crash reporter `segv.so` afterwards (every bench line within its usual values), and the three plain passes of the same checks inside the 4B session itself passed, so 1 death in 9 `--bench` runs that day, unplaced. Nothing points at step 2 (the bound table is exercised by every pass that passed); what the bench pass shares with question 8's known fault is its exposure, not a finding: the check program's correctness references run the CPU backend with 8 threads on the X100 cores (`check_spacemit.cpp:1355`, `:1503`, `:2025`), Q4_1 matmuls included, which is the path (`ggml_vec_dot_q4_1_q8_1` under several threads) where the CPU-only perplexity runs died 5 times and computed wrong results at random (item 28). Accepted by its rule, as the 2026-10-27 run was with its CPU-only death; proposed to the user, not done: the check program's correctness references at one thread (§6's rule for CPU-versus-provider comparisons, which the check program is), the speed comparisons of `check_gemv_strategies` kept at 8, and the runner's `--segv` extended to the check program's passes so a recurrence leaves a register dump. The same day, in the same board session, E7 (`structural-experiments.md` run 6) measured spine-runtime's shared-buffer view across suspensions with `scripts/spert-view-bench.cpp`; its results do not touch this build.

**The conforming tile loop, the check program's hygiene and E7b on the K3, 2026-10-30** (session 1 of the step-by-step plan, `structural-experiments.md` run 7; the patch on `c3075078a`). The executor's tile loop takes its `shared_buffer()` view again after every barrier, as the header grants it only while the tile runs on that worker (`spert.hpp:474-478`, E7); the A/B against the old loop (`FLAGOS_SPACEMIT_TEST_VIEW=once`; Qwen3-4B, three interleaved `llama-bench` runs each) gave pp128 106.55 against 106.08 (+0.4%) and tg128 10.96 against 10.99 (-0.3%, lower in all three pairs), within the 1% rule, so the re-take is the default; outputs unchanged by the code. The check program's correctness references (`check_mul_mat`, `check_flash_attn`, `check_layer`, `diagnose_e3b`) run the CPU backend single-threaded (§6's rule; the speed benchmarks keep 8 threads), and the runner's `--segv` now preloads the crash reporter into the check program's passes too: `--full` all checks passed in 141 s, the reference pass in 221 s, `--bench` passed, all under `segv.so`. E7b (`flagos-check-spacemit --e3b=5` with `FLAGOS_SPACEMIT_TEST_TCM_SLICE`, test-only: each tile an `alloc_shared` slice instead of the whole buffer, the ordered launches' pair waits yielding the core): every ordered run the runtime executed matched one launch byte for byte, but slices of half, a third or the whole pool ended in spine-runtime's `Status::Deadlock` (several tiles of one launch on one worker, their slices exceeding the pool) with the stream dead for the process; at a pool eighth (49,152 bytes) 20 of 20 lines 5 of 5 same. Nothing changes on the default path (the yield is wired only for `per-step-join`, the slice only with the switch).

**S1 step 2b on the K3, 2026-10-31** (the default path pruned before the plan cache; `plan.md` "S1 step 2b"; the patch on `13db144aa`). The four kernel switches that were parsed from the environment on every call on every tile (`FLAGOS_SPACEMIT_GEMV` at every 1-row matmul step, `MM` and `TAIL` at every several-row step, `ATTN` at every attention node) are parsed once and parsed again only after `spacemit_switches_reset()`, which the check program calls from `set_env` and the plugin serves by name (the seventh `get_proc_address` name); the op counters' step names and the step counters' attach calls run only for a counted graph. A/B against the library built before the patch (preloaded; three interleaved `llama-bench` runs per model): Qwen3-4B tg128 11.02 against 10.96 (+0.55%, higher in all three pairs), pp128 106.44 against 106.49; Qwen3-0.6B tg128 54.77 against 54.72 (+0.09%), pp128 697.56 against 694.72; no loss anywhere, so adopted by the rule; the size of the gain is not isolated (the two models disagree on it: an estimate of 2,000 parses per token predicts more on 0.6B, which shows spread only). The check program's `--full` pass under `segv.so` passed with every switch-flipping line as before (the gemv, mm, attention and E3b checks change switches inside one process, so they test the reset); on the Mac eight settings, ThreadSanitizer, the riscv64 syntax check. Found on the way, and the plan's mutation met by accident: with the reset only in `env_override`, the gemv check's `copy` control stopped differing and every mm setting took path C, because those checks set switches with raw `set_env` calls; the reset moved into `set_env`.

**S1 step 3 on the K3, 2026-10-31** (the plan cache, A4's experiment; `plan.md` "S1 step 3 spec", its "Result on the K3"; `structural-experiments.md` run 11; the patch `sx-step3.patch` on `a45231bba`; `EXPS="plan" REPS=3 DEPTH=0` on Qwen3-4B with the perplexity text, then Qwen3-0.6B without it; the modes `ref`, `e3c`, `plan-s`, `plan-u` interleaved, three `llama-bench` runs each). The check program passed under `FLAGOS_SPACEMIT_PLAN=none`, `structure` and `uid`. Both candidates hit 0.99 of the generated tokens' graphs (the first graph of each run the miss; by uid under U, by structure under S) and give the reference's logits on Qwen3-4B (perplexity 9.7736 at 512-token batches and 9.7619 at one token, mean KLD 0.000000, maximum 5.4e-5 and 5.7e-5, same top p 100%, the base file's floor). Speed, tg128 against `ref` (4B `ref` 106.54 pp128, 11.01 tg128; 0.6B 698.62, 54.89): 4B `e3c` 10.95 (-0.5%), `plan-s` 10.82 (-1.7%), `plan-u` 10.94 (-0.6%), pp128 all within 0.6%; 0.6B `e3c` 54.10 (-1.4%), `plan-s` 51.68 (-5.8%), `plan-u` 54.06 (-1.5%), pp128 within 0.8%. The counters' host column per generated token, µs: 4B `e3c` 613, `plan-s` 1,588, `plan-u` 557; 0.6B 406, 1,230, 409; the key alone 1,015 / 50 µs on 4B and 801 / 39 on 0.6B (S / U). By the rule of 2026-10-29 no cache is kept: U saves 56 µs per 4B token (0.2 ms required), S costs 0.8-1.0 ms. Read from the columns (differences of means, not a timed split): the per-node contract check is about 490 µs of a 4B token and the expansion the cache replaces about 105 µs; the cache attacked the smaller part. Decision (user): S removed, U kept, and a trusted-uid form (`uid-trust`, the contract check skipped on a hit, llama.cpp's CUDA and Hexagon form and the deck's "redone only when the structure changes") to be built and measured in one more session per model (step 3b, `plan.md`).

**S1 step 3b on the K3, 2026-11-01** (the trusted uid hit; `plan.md` "S1 step 3b spec", its "Result on the K3"; `structural-experiments.md` run 12; the patch `sx-step3b.patch` on `02a1cb9ee`; `EXPS="plan" REPS=3 DEPTH=0` on Qwen3-4B with the perplexity text, then Qwen3-0.6B without it; the modes `ref`, `e3c`, `plan-u` (`FLAGOS_SPACEMIT_PLAN=uid`, the contract check on every graph), `plan-ut` (`uid-trust`, a hit skipping the check) interleaved, three `llama-bench` runs each). The check program passed under `none`, `uid` and `uid-trust`. Both settings hit 0.99 of the generated tokens' graphs and give the reference's logits on Qwen3-4B (perplexity 9.7736 at 512-token batches and 9.7619 at one token, mean KLD 0.000000, maximum 5.4e-5 and 5.7e-5, same top p 100%, the base file's floor); the text check ran under `ref` alone (`EXPS=plan` adds no mode to it; corrected 2026-11-01: this paragraph said "the generated text identical in every mode", which no run tested), so the modes' sameness is the KLD's, on 4B only. Speed, against `ref` (4B `ref` 106.48 ± 0.09 pp128, 11.03 ± 0.03 tg128; 0.6B 692.28 ± 6.28, 54.87 ± 0.09): 4B tg128 `e3c` 10.94 (-0.8%), `plan-u` 10.94 (-0.8%), `plan-ut` 11.00 (-0.2%), pp128 within 0.4%; 0.6B tg128 53.94 (-1.7%), 53.87 (-1.8%), 55.10 (+0.4%), pp128 `plan-ut` 701.00 (+1.3%, within one spread). The counters' host column per generated token, µs: 4B `e3c` 630, `plan-u` 575, `plan-ut` 46; 0.6B 446, 486, 39; the key alone 24 / 18 µs on 4B and 19 / 15 on 0.6B (`uid` / `uid-trust`); the token 91,398 / 91,375 / 90,884 µs on 4B and 18,540 / 18,563 / 18,150 on 0.6B. By the rule (0.2 ms of host work per 4B token saved against `e3c`, identical logits): `uid-trust` adopted, 584 µs saved on 4B and 407 on 0.6B. Against the counted modes it gains tg128 0.6% on 4B and 2.2% on 0.6B; against `ref` the counters it carries hide the gain (they cost `e3c` 0.8% and 1.7% here), so the clean A/B is step 3c's. Decision (user): `uid-trust` the default, the checked form `uid` removed with its probes and runner mode (`plan.md` "S1 step 3c spec").

**S1 step 3c on the K3, 2026-11-01** (the trusted uid hit the default, the checked form removed; `plan.md` "S1 step 3c spec", its "Result on the K3"; `structural-experiments.md` run 13; the patch `sx-step3c.patch` on `65f31604b`). The check program passed under the switch unset, `none` and `uid-trust`. Run 13 (`EXPS="plan" REPS=3 DEPTH=0`, the modes interleaved, three `llama-bench` runs each), without counters, the default against `none`: tg128 11.07 ± 0.01 against 11.01 ± 0.00 on Qwen3-4B and 56.16 ± 0.17 against 54.88 ± 0.23 on Qwen3-0.6B (`none` 0.6% and 2.3% below), pp128 within spread (106.36 against 106.45, 703.57 against 697.47), the logits identical on 4B (mean KLD 0.000000, maximum 5.4e-5 and 5.7e-5, same top p 100%); with the counters the host work per generated token 46 against 629 µs on 4B and 39 against 495 on 0.6B. Then the acceptance runner (`--segv ~/segv.so`, the default milestone `igpu`, Qwen3-0.6B, then Qwen3-4B with `--depth`; no rebuild; the CPU references single-threaded): **39/39 and 40/40.** The check program's four passes (the default with `--full`, the reference kernels, `accel`, `--bench`) all passed, each with the loader line (`14 kernels, 8 names by get_proc_address`), the contracts line and the plan line (15/15 under `none`, 14/14 under `uid-trust`, the outputs identical); every layer and the KV cache on the provider (29/29 and 37/37 layers, KV 4480 and 5760 MiB); 2 graph splits per token; flash attention kept under `-fa auto`; the Q8_0-cache and context-shift runs complete (the shift with 399 evaluations after the first token); the text identical to the CPU's on 0.6B and leaving it after 143 characters on 4B, as before; no TCM block held; no crash dump from any provider run. Accuracy, 2026-10-27's to the last digit with the error bars on both models; the tails against the CPU recorded here for the first time:

| | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| perplexity, provider `-fa on` and `-fa auto` / CPU | 14.4471 / 14.3706 | 9.7736 / 9.7740 |
| mean KLD, 512-token batches | 0.004722 ± 0.000221 | 0.002489 ± 0.000156 |
| mean KLD, `-ub 1 -b 512` | 0.002533 ± 0.000089 | 0.001229 ± 0.000058 |
| maximum KLD, 512-token batches / one token | 0.238789 / 0.083388 | 0.239437 / 0.061669 |
| 99.0% KLD, 512-token batches / one token | 0.036518 / 0.015029 | 0.017823 / 0.008408 |
| same top p, 512-token batches / one token | 95.931 ± 0.438 / 97.304 ± 0.359 % | 97.353 ± 0.356 / 98.039 ± 0.307 % |

Speed, the runner's `llama-bench` (3 repetitions; the modes back to back, not interleaved):

| t/s | Qwen3-0.6B | Qwen3-4B | 4B at step 2 (2026-10-30) |
|---|---|---|---|
| provider pp128 / tg128 | 704.21 ± 6.19 / 56.13 ± 0.25 | 106.40 ± 0.15 / 11.07 ± 0.01 | 106.55 / 11.00 |
| `ime` pp128 / tg128, same session | 540.34 / 54.27 | 79.14 / 10.98 | |
| CPU pp128 / tg128 | 155.54 / 27.81 | 21.11 / 5.57 | |
| provider tg32 at depth 0 / 1024 / 4096 | | 11.13 / 9.70 / 7.69 | 11.03 / 9.70 / 7.71 |
| `ime` tg32 at depth 0 / 1024 / 4096 | | 11.06 / 9.35 / 6.40 | |
| pp512 at depth 4096, provider / `ime` | | 36.08 / 25.13 | 36.06 |
| Q8_0 KV cache, `llama-completion -n 32` (recorded), provider / CPU | 28.42 / 31.37 | 7.15 / 5.55 | |

- **Accepted by step 3c's rule** (`plan.md`, item 7): the acceptance runs pass with the usual numbers and the default is nowhere below `none` by more than the spread, so `uid-trust` stays the default. The rule's parenthetical, "perplexity within 0.1% of the CPU", is 4B's figure; on 0.6B the usual is +0.53%, met to the last digit.
- Speed within the 1% rule against step 2's session, the last acceptance (the depth rows single runs), and the runner reproduces run 13's default within its spread (4B 106.40 / 11.07 against 106.36 / 11.07; 0.6B 704.21 / 56.13 against 703.57 / 56.16). On 4B the provider's tg128 is 0.8% above the upstream IME path's in this session (11.07 ± 0.01 against 10.98 ± 0.02, back to back; one session's reading); on 2026-10-27 and 10-28 the two were within 0.2%. The Q8_0 cache is recorded only; its CPU-only figure, 32 tokens of `llama-completion` on 8 threads, moved from 27.74 (2026-10-27) to 31.37, the provider's from 28.92 to 28.42.
- **Found in the runner, fixed after the run:** the check "perplexity -fa auto: flash attention on the provider" had not run since 2026-10-29: commit `3d272f045` gave the `-fa auto` run the flags `-t 8`, and the runner found that run by its empty argument list, so every count since was one short without a message (step 2's 41/41 with `--build` is the full set less this check). By hand in this run: the `-fa auto` perplexity runs logged "Flash Attention enabled" and 2 graph splits on both models. The fix finds the run by name; checked on the Mac against a stand-in `llama-perplexity`: the check reports again, fails when that run's attention leaves the provider, and the runner before the fix reproduces the missing check. From the next run the counts are 40 on Qwen3-0.6B (without `--build`) and 41 on Qwen3-4B with `--depth`.
- The CPU-only depth benchmark on 4B (provider disabled, 8 threads on the X100 cores) died twice, losing its tg32 and pp512 at depth 4096: question 8's CPU crash, the sixth and seventh time, both at the instruction of 2026-10-09 and 2026-10-13 and with the register pattern of 2026-10-13 (the dumps in the crash entry under "Provider M2b" above). Not gated (the CPU's depth rows are recorded only); the provider's runs untouched.

**S1 cleanup on the K3, 2026-11-02** (the code of decided experiments removed; `plan.md` "S1 cleanup before step 5", its "Result on the K3"; `PROJECT.md` item 37; the patch `sx-cleanup.patch` on `cad674fda`). The acceptance runner under the crash reporter, Qwen3-0.6B with `--build`, then Qwen3-4B with `--depth`: **42/42 and 41/41.** The counts rest on a base of 40 checks (step 3c's 39 plus the `-fa auto` attention check, which that run skipped and the same commit restored): `--build` adds the configure and build checks, `--depth` the depth bench. Accuracy, 2026-10-27's to the last digit on both models:

| Model | Perplexity | CPU | Mean KLD, 512-token batches | Mean KLD, one token per ubatch |
|---|---|---|---|---|
| Qwen3-4B Q4_0 | 9.7736 | 9.7740 (-0.00%) | 0.002489 ± 0.000156 | 0.001229 ± 0.000058 |
| Qwen3-0.6B Q4_0 | 14.4471 | 14.3706 (+0.53%) | 0.004722 ± 0.000221 | 0.002533 ± 0.000089 |

Speed within spread of run 13's (that run's figures in brackets): 4B pp128 105.65 and tg128 11.10 (106.40 / 11.07), with the upstream IME build at 79.18 / 10.99 in the same session; tg32 at depths 0 / 1024 / 4096 11.19 / 9.76 / 7.75 (11.13 / 9.70 / 7.69) and pp512 at depth 4096 35.88 (36.08); 0.6B pp128 698.34 and tg128 56.33 (704.21 / 56.13). Placement unchanged (37 of 37 and 29 of 29 layers with the KV cache on the provider, 2 graph splits per token, flash attention kept under `-fa auto`), the Q8_0-cache and context-shift runs complete (399 evaluations after the first token), the text leaving the CPU's after 143 characters on 4B as before, no TCM block held, no crash dump from a provider run. **Accepted:** the cleanup changed no output and no speed. One known event: the 4B CPU-only `llama-bench` run (provider disabled, 8 threads on the X100 cores) died with signal 11, question 8's crash; its depth runs completed (tg32 5.93 / 4.70 / 1.56, pp512 at depth 4096 11.73). An INFO line, not gated.

**S1 step 5 on the K3, 2026-11-02** (the diagnostics under the deck's names; `plan.md` "S1 step 5 spec", its "Result on the K3"; `PROJECT.md` item 38; commit `fdf1bd733`, pulled from a bundle). The acceptance runner under the crash reporter, Qwen3-0.6B with `--build`, then Qwen3-4B with `--depth`: **43/43 and 42/42**, the cleanup's counts plus the new check "ops disabled by name leave the provider" (`FLASH_ATTN_EXT` refused under `-fa on`). Both KLDs of 2026-10-27 to the last digit with their error bars on both models (4B 0.002489 ± 0.000156 and 0.001229 ± 0.000058; 0.6B 0.004722 ± 0.000221 and 0.002533 ± 0.000089; the perplexity lines passed inside the count, their figures not in the excerpt reported). Speed within spread of the cleanup run's (that run's figures in brackets): 4B pp128 106.01 and tg128 11.10 (105.65 / 11.10), the upstream IME build 79.17 / 11.00 in the same session, the CPU-only build 21.09 / 5.61 (this time without a crash); tg32 at depths 0 / 1024 / 4096 11.17 / 9.72 / 7.75 (11.19 / 9.76 / 7.75), pp512 at depth 4096 35.89 (35.88); 0.6B pp128 696.56 and tg128 56.39 (698.34 / 56.33), the IME build 544.58 / 54.10, the CPU 154.36 / 27.43. **Accepted:** the renames and the two new diagnostics changed no output and no speed, and the disabled-op path works on the board. The by-hand checks of the spec, 2026-11-03: one `EXPS=e3c REPS=1 DEPTH=0` run of the structural runner on Qwen3-0.6B wrote `bench-e3c-std-1.stats` and `bench-e3c-a-std-1.stats` under the new name, its E3c table in line with run 13's (one generated token 18,036 µs against 18,170, host 38 against 39, idle 967 against 1,000 or 5.36%, lookup 15.7 and key 13.8 µs, 0.99 hits by uid and 0.01 misses per token, 846 steps in one launch; under `SYNC_LAUNCH` 846 launches at 11.10 µs of overhead each, E3a's cost 12.72 µs per step; tg128 `ref` 55.96, `e3c` 55.44, `e3c-a` 34.73, one repetition, so no spread); placement 2 / 2 splits, flash attention enabled, 29/29 layers, the text "Paris". The trace's step count printed 0 on the K3: llama.cpp's tools drop the trace unless run with `-lv 4`: the provider logs at ggml's INFO level, which common's log callback maps to verbosity 4 (`common/log.cpp:443`) above the default threshold 3, and the spec's command lacked it. On the Mac's simulated device with `-lv 4` (Llama-3.2-1B Q4_0, `-n 1`): 844 step lines in two graphs of 422, the `launch <n> steps` lines summing to the same 844, all `success`; under `SYNC_LAUNCH` 844 per-step status lines, none failing; without `-lv 4` 0. The K3 count with `-lv 4` is still to run. `GGML_FLAGOS_OP_STATS=~/ops.jsonl llama-bench -p 0 -n 32` on Qwen3-4B (2026-11-03, tg32 11.09 ± 0.06; `op-stats.py show`): 160 one-row graphs of 1,086 steps at n_kv 256, 89.14 ms each (kernels 85.88, imbalance 2.71, barriers 0.55), by group within a few percent of the 2026-10-21 baseline (`fusion-and-splitting.md` §4, tg128, so attention not comparable): matmul multiply steps 85.03 ms (95.4%; 83.46), the quantize steps 1.09 (1.04), `ROPE` 0.57 (0.59), `RMS_NORM` 0.42 (0.43), `MUL` 0.40 (0.40), `ADD` 0.30 (0.32), `GLU` 0.29 (0.30), attention 0.86 (1.44 over tg128's longer context). The Q6_K head run as Q8_0 13.54 ms (15.2%); a Q4_1 `ffn_down` requantized to Q8_0 1.01 ms against 0.49 for a Q4_0 one; the Q4_0 matmuls read their 1,895.6 MiB at about 29.5 GB/s, the head about 30.5 (computed from the sizes, an estimate). `GGML_FLAGOS_DISABLE_OPS=FLASH_ATTN_EXT` under `-fa on` on Qwen3-4B (2026-11-03, `llama-bench -p 128 -n 128`, one run each, interleaving none): pp128 106.12 ± 0.04 to 95.01 ± 0.96 (-10.5%), tg128 11.10 ± 0.02 to 8.44 ± 0.03 (-24.0%), 28.4 ms more per generated token and 141 ms more per 128-token batch, with attention on the X100 cores and the graph split around it in every layer; on the provider attention takes about 1.4 ms of a tg128 token and 62 ms of a pp128 batch (the 2026-10-21 baseline), so the rest is the route through the CPU (the split's copies and synchronizations, the CPU's attention), not decomposed.

**One token per ubatch at 4096 context, 2026-11-03** (the first model-level check of the one-token path beyond n_kv 512: the runner's `-ub 1` KLD runs at `-c 512`, `tools/spacemit_check.py:420-422`; no code change, the build of `fdf1bd733`). Qwen3-0.6B Q4_0, `~/ppl.txt`, `-c 4096 --chunks 1 -fa on`, so the 2,047 scored tokens sit at positions 2048-4095 and every one-token graph attends over 2,048 to 4,095 cache cells. The CPU reference single-threaded with the provider disabled, twice, the two base files byte-identical (`cmp`), perplexity 11.3793 ± 0.85186 both times (about 8.6 minutes a run). The provider at `-t 8`, against that base:

| Qwen3-0.6B, `-c 4096`, 1 chunk | one token per ubatch (`-b 4096 -ub 1`) | batched (the defaults, 512-token ubatches) |
|---|---|---|
| perplexity (exact) / CPU's 11.3793 | 11.3677 (-0.10%) | 11.3655 (-0.12%) |
| mean KLD | 0.002388 ± 0.000089 | 0.004313 ± 0.000198 |
| maximum KLD | 0.091536 | 0.218180 |
| same top p | 97.802 ± 0.324 % | 97.313 ± 0.357 % |

The same figures at `-c 512`, 8 chunks (the step 3c table above): 0.002533 ± 0.000089 and 0.004722 ± 0.000221, maximum 0.083388 and 0.238789, same top p 97.304 and 95.931 %. Read: no loss of accuracy from n_kv 512 to 4096 on the one-token path or the batched one (different tokens of the same text, so a check against a gross long-context error, not a like-for-like comparison); one token per ubatch stays the closer of the two, as at 512. The TCM release tool listed all eight blocks free before each provider run. Not run: contexts beyond 4096, the Q8_0 KV cache (whose attention runs on the CPU: the provider's `FLASH_ATTN_EXT` takes F16 K and V only, `flagos-spacemit-attn.cpp:414-421`). Reading the tool's output: the KLD run's "Mean PPL(base)" (11.3216) is rebuilt from the base file, which stores each token's log-probabilities clamped at 16 below its top logit (`tools/perplexity/perplexity.cpp:86`, rebuilt at `:214`), while "Mean PPL(Q)" is exact (`:210`); so its ratio line (1.0041 and 1.0039) compares an exact perplexity with a clamped one, and the provider-to-CPU comparison is the exact pair in the table. That the clamp accounts for all of 11.3793 against 11.3216 is an inference; the CPU run as Q against the same base would settle it (it should print 11.3793 and 11.3216 with a KLD near 0).

Qwen3-4B Q4_0, the same commands (2026-11-03): the two single-threaded CPU base files byte-identical; the base runs' own perplexity (their `Final estimate` lines) not in the excerpt reported, so the exact comparison with the CPU is still to fill in.

| Qwen3-4B, `-c 4096`, 1 chunk | one token per ubatch (`-b 4096 -ub 1`) | batched (512-token ubatches) | at `-c 512` (step 3c), one token / batched |
|---|---|---|---|
| perplexity (exact) | 7.7230 | 7.7487 | |
| mean KLD | 0.001447 ± 0.000208 | 0.002280 ± 0.000199 | 0.001229 ± 0.000058 / 0.002489 ± 0.000156 |
| maximum KLD | 0.410694 | 0.353781 | 0.061669 / 0.239437 |
| same top p | 98.632 ± 0.257 % | 98.192 ± 0.295 % | 98.039 / 97.353 % |

Read: the means and the same-top-p shares are in line with `-c 512` (one token per ubatch +18%, about one standard error; batched -8%), but **the one-token path's tail is heavier at 4096 on 4B**: its maximum 0.41 against 0.06 at 512, and the per-token spread implied by its error bar (the standard error times the square root of 2,047 tokens) 0.0094 against 0.0026, 3.6 times; the batched path's spread grew 1.3 times (0.0090 against 0.0071), and on 0.6B neither grew (0.0040 both, one token per ubatch). Not explained. Two readings fit: a few hard tokens in this stretch of the text that the one-token path handles worse (the batched path reaches 0.35 on it), or a long-context effect in the one-token path of 4B's heads (32 query heads over 8 KV heads; one-token calls from n_kv 512 take the split attention, `flagos-spacemit-attn.cpp:165-166`, checked against the CPU at the NMSE bound 5e-4, `check_spacemit.cpp:1667`). The tests that separate them: the default run again with the whole output kept (whether the maximum reproduces, and the tail percentiles), the same with `FLAGOS_SPACEMIT_ATTN=one-chunk` (no split), and with `GGML_FLAGOS_DISABLE_OPS=FLASH_ATTN_EXT` (attention on the CPU); plus the split's measured NMSE in the runner's `check-spacemit.out`.

The split's NMSE, read the same day, and what it measures. The check program's attention lines (`check-spacemit.out` of the last acceptance runs on the K3, both models' logs alike; the same program on the Mac's simulated device, where the portable reference kernels run), each kernel against ggml-cpu's single-threaded `FLASH_ATTN_EXT`:

| kernel | V summed in | max NMSE against the CPU |
|---|---|---|
| RVV one-chunk (the K3) | F16 (`flagos-spacemit-rvv-attn.cpp:633-934`) | 2.78e-6 |
| portable one-chunk (the Mac) | F32 (`flagos-spacemit-attn.cpp:45`) | 8.78e-5 |
| RVV split, one token from n_kv 512 (the K3) | F32 (`flagos-spacemit-rvv-attn.cpp:1711-1714`) | 1.59e-4 |
| portable split (the Mac) | F32 | 1.61e-4 |

ggml-cpu's one-token path sums V in an F16 accumulator, rounded at every position (`ggml/src/ggml-cpu/ops.cpp:8566-8632`), single-threaded over all n_kv positions in one chunk (`:9116-9135`); its batched path (64 query rows and more, as in the KLD bases) sums in F32 (`ops.cpp`, the tiled kernel's `VKQ32`). The pattern, measured: the kernels that sum in F16 as the reference does match it to about 1e-6, those that sum in F32 differ from it by about 1e-4, on either machine. Read: the split's 1.6e-4 is most likely the reference's own rounding (an F16 sum over thousands of positions; a random walk of half-ulp errors over 4,096 positions gives a relative error near 1.6e-2, an NMSE near 2.5e-4, an estimate), not an imprecision of the split, which sums in F32 as the CPU's batched kernel does. So the split is not the likely cause of the 4B tail: in the KLD runs both sides summed attention in F32 (the provider's split against the CPU's batched base). Unverified until an exact reference exists: the check program compares against ggml-cpu, not against an F64 computation. Expected under this reading: `FLAGOS_SPACEMIT_ATTN=one-chunk` (F16 sums) leaves the KLD the same or worse, not better.

**S1 step 5 follow-up on the K3, 2026-11-03** (the three switches whose warnings llama.cpp's fit step hid, read at registration; `plan.md` "S1 step 5 follow-up"; `PROJECT.md` item 39; pulled from a bundle onto `6cf86f21b`). The acceptance runner with `--build` under the crash reporter on Qwen3-0.6B: **43/43**; both KLDs 2026-10-27's to the last digit (0.004722 ± 0.000221 and 0.002533 ± 0.000089); speed within spread of step 5's run (pp128 694.96 and tg128 56.37 against 696.56 / 56.39; the IME build 538.18 / 54.22 and the CPU 152.99 / 26.72 in the session); no TCM block held. `GGML_FLAGOS_DISABLE_OPS=FLASH_ATTN_EXT llama-completion -fa on` (fitting on) prints its warning once, where it printed nothing before. **Accepted.** Qwen3-4B not rerun: the change reads three switches earlier and touches no model path.

**One token per ubatch at n_kv 256, 2026-11-03** (the first model-level check of the one-chunk kernels, which every one-token step below 256 cached positions takes: `llama-perplexity` scores only each chunk's second half, `tools/perplexity/perplexity.cpp:1792`, and the cache rounds n_kv up to a multiple of 256, at least 256, `src/llama-kv-cache.cpp:1238-1243`, so the runner's `-ub 1` KLD at `-c 512` scores only n_kv 512, the split; the trace on the Mac's simulated device: at `-c 256` all 4,128 attention steps of a chunk one-chunk, at `-c 512` the first half's 4,128 one-chunk and the second half's 4,096 split). Qwen3-0.6B Q4_0, the build of the follow-up, `-c 256 -b 256 --chunks 16 -fa on`, 2,032 scored one-token steps; the CPU base single-threaded twice, perplexity 16.9595 ± 1.26323 both times, no error line, about 5.2 minutes a run (the `cmp` line not in the excerpt reported and the second file deleted by the block, so the byte identity rests on a third run against the kept file, still to run); the provider at `-t 8`, 29/29 layers, 2 graph splits:

| Qwen3-0.6B, one token per ubatch | n_kv of the scored steps, attention kernel | mean KLD | 99% KLD | maximum KLD | same top p |
|---|---|---|---|---|---|
| `-c 256`, 16 chunks | 256, one-chunk (V summed in F16) | 0.002722 ± 0.000163 | 0.014740 | 0.258274 | 96.900 % |
| `-c 512`, 8 chunks (the runner, step 3c's table) | 512, split | 0.002533 ± 0.000089 | 0.015029 | 0.083388 | 97.304 % |
| `-c 4096`, 1 chunk (above) | 2,304-4,096, split | 0.002388 ± 0.000089 | | 0.091536 | 97.802 % |

Batched at `-c 256` (256-token ubatches): mean KLD 0.004177 ± 0.000133, 99% 0.024664, maximum 0.103787, same top p 95.915 % (at `-c 512` 0.004722, 0.036518, 0.238789, 95.931 %). Read: the one-chunk path is as accurate as the split in bulk (the mean one standard error above 512's, the median 0.0017 and the 99% point the same); only its extreme top is heavier (99.9% 0.0744, a maximum of 0.26 against 0.08, the per-token spread implied by the error bar 0.0074 against 0.0040), in a different window of the text (the second halves of 256-token chunks, other tokens than at 512). The same kind of tail as Qwen3-4B's at 4096 on the split, so not tied to one attention kernel; whether a few hard tokens or something in the one-token path is not settled by these runs. Perplexity against the CPU's 16.9595: 16.9110 one token per ubatch (-0.29%) and 16.9772 batched (+0.10%), both within the paired uncertainty the KLD runs report (about ±0.35%); their "PPL(base)", rebuilt from the clamped base, 16.9097, 0.29% below the exact figure, as at 4096.

**Qwen3-4B at 4096 context, the separating runs, 2026-11-03** (the build of the step 5 follow-up; the base `kld4k-4b.bin` of the first 4B probe; `-c 4096 -b 4096 -ub 1 --chunks 1 -fa on`, 2,047 scored one-token steps; 37/37 layers each time):

| Qwen3-4B, one token per ubatch | attention | mean KLD | without the worst token | median | 99% | 99.9% | maximum | same top p | perplexity |
|---|---|---|---|---|---|---|---|---|---|
| run 1, the default again | split, on the provider (2 splits) | 0.001447 ± 0.000208 | 0.001247 | 0.000374 | 0.009281 | 0.033365 | 0.410694 | 98.632 % | 7.7230 |
| run 2, `FLAGOS_SPACEMIT_ATTN=one-chunk` | one-chunk, on the provider (V summed in F16; 2 splits) | 0.001581 ± 0.000138 | 0.001495 | 0.000450 | 0.011213 | 0.075197 | 0.176544 | 98.437 % | 7.6938 |
| run 3, `GGML_FLAGOS_DISABLE_OPS=FLASH_ATTN_EXT` | ggml-cpu's (74 splits, the warning shown) | 0.001452 ± 0.000183 | 0.001279 | 0.000387 | 0.010008 | 0.050168 | 0.355261 | 98.583 % | 7.7072 |
| at `-c 512` (step 3c's table) | split, on the provider | 0.001229 ± 0.000058 | | | 0.008408 | | 0.061669 | 98.039 % | |

Run 1 gives the first probe's figures to every printed digit (mean, maximum, same top p, perplexity): the result is deterministic, and the follow-up's build gives the step 5 build's output on the K3. **The heavier tail is one token:** without its worst token run 1's mean is 0.001247, `-c 512`'s within its error bar, and that token alone accounts for 0.0091 of the 0.0094 per-token spread (the standard error times the square root of 2,047); the other 2,046 tokens' spread, what the error bar leaves after it, is 0.0026, `-c 512`'s, so a second token can reach at most 0.12 (with ggml-cpu's attention the same: 0.0027 left, at most 0.13). **Attention is not its cause:** with ggml-cpu's attention (run 3) the worst token reaches 0.36, and the bulk is the same (0.001279 without it); with the one-chunk kernels 0.18, their bulk slightly higher (0.001495 without the worst token, median 0.00045 against 0.00037), as their F16 sums predicted. The default, the split, has the lowest median, 99% and 99.9% of the three. Read: one token of this window is sensitive enough that every variant's rounding moves its distribution far (0.18 to 0.41), and the batched path reaches 0.35 on the same window (the first probe); which token, and why it is so sensitive, the KLD summary cannot show (that needs a per-token view). No defect found in the one-token path at long context. The perplexities differ by at most 0.4%, within the paired uncertainty (about ±0.34%); the CPU's own figure for this base was not saved (its runs printed only their last line).

**Batch size and context on the K3: the bottleneck suite, 2026-11-03** (Qwen3-4B Q4_0, the build of the step 5 follow-up, `llama-bench -t 8 -fa on -lm none -r 3`; the suite's text tested on the Mac first). Step 1, one prompt graph of N rows each (`-p N -n 0`; one row is generation's shape):

| rows | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|
| t/s | 11.45 | 18.91 | 36.05 | 60.79 | 80.99 | 103.55 | **116.55** | 106.85 | 103.37 | 96.39 |
| time per graph against one row's 87.3 ms | 1.00 | 1.21 | 1.27 | 1.51 | 2.26 | 3.54 | 6.29 | 13.72 | 28.36 | 60.82 |

Read (the derived figures computed from these, estimates where so labelled): one row reads the 2.51 GB the provider stores per token at 28.7 GB/s, the AI cores' streaming ceiling, so generation at batch size 1 is bound by DRAM bandwidth; up to 8 rows the graph takes at most 1.5 times one row's time (5.3 times the throughput), from 16 rows the time grows nearly in proportion, so the knee lies between 8 and 16 rows (the roofline estimate gave about 8 for Q4_0); the plateau is 116.55 t/s at 64 rows, about 0.85 int8 TOPS (2 operations per weight, the head excluded since a prompt graph computes one row of logits), the ceiling of our kernels, not necessarily the chip's; past 64 rows the speed falls (96.39 at 512), where the prompt's own attention grows (12% of a pp512 batch on 2026-10-21, `fusion-and-splitting.md` §4) and the matmuls change path. Two rows cost 1.21 times one, nearly what four cost (1.27): probably the 4-row kernel computing padding rows (the IME kernels come as `_m1` and `_m4`), not measured. Against the 2026-10-21 sweep (an older build, `fusion-and-splitting.md` §7.3) today's default matches or beats its best setting from 32 rows (pp64 116.55 against 108.39 at best). Step 2, generation against context depth (`-p 0 -n 32 -d N`):

| tokens cached | 0 | 1,024 | 2,048 | 4,096 | 8,192 |
|---|---|---|---|---|---|
| tg32 t/s | 11.20 | 9.75 | 8.98 | 7.75 | 6.09 |
| ms per token | 89.3 | 102.6 | 111.4 | 129.0 | 164.2 |
| KV cache read per token (n_kv rounded up to 256; 144 KiB a position) | 36 MiB | 180 MiB | 324 MiB | 612 MiB | 1,188 MiB |

Depths 0, 1024 and 4096 repeat step 5's run within 0.3%. Read: the cache costs 8.6 µs per cached position per token from 2,048 to 8,192 (computed), so it is read at about 17 GB/s at the margin, about 60% of the weights' rate (11-16 GB/s including attention's fixed costs at the shallower depths; all the extra time counted as cache reading, an estimate that ignores attention's arithmetic); at about 10,400 cached positions the cache would take as long as all the matmuls (extrapolated). Step 3, parallel decoding (`llama-batched-bench -c 8192 -b 2048 -ub 512 -npp 128 -ntg 128 -npl 1,2,4,8,16`; a KV stream per sequence; 37/37 layers, 2 graph splits, so attention stayed on the provider):

| sequences decoded together | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| total generation t/s (S_TG) | 10.91 | 17.59 | 31.77 | 46.74 | 59.35 |
| each sequence's t/s | 10.91 | 8.79 | 7.94 | 5.84 | 3.71 |
| ms per step / the same rows in one prompt graph (step 1) | 91.7 / 87.3 | 113.7 / 105.8 | 125.9 / 111.0 | 171.2 / 131.6 | 269.6 / 197.6 |
| prompt phase, S_PP t/s | 105.22 | 104.81 | 102.93 | 103.27 | 71.18 |

Read: 2.9 times the total at 4 sequences with each at 73% of one alone, 4.3 times at 8 (54%), 5.4 times at 16 (34%). A step costs more than the same number of rows in one prompt graph, 15 ms more at 4 sequences and 72 at 16: each sequence reads its own cache (B × 256 positions × 144 KiB a step), at an implied 8-10 GB/s (all the gap counted as cache reading, an estimate), below one sequence's 17 GB/s at depth; with several streams one-token attention takes the one-step kernels (the split needs a single stream, `flagos-spacemit-attn.cpp:209`). Not measured: the per-op split of a parallel step (the op counters on `llama-batched-bench` would give it).

Step 4, the op counters at 1, 8 and 128 rows (prompt graphs at n_kv 256, `op-stats.py show`): per graph 87.5, 131.9 and 1,198 ms (step 1's ratios), the tiles' imbalance 2.7, 8.8 and 3.8 ms (3.1%, 6.6%, 0.3%). Computed from the weights' sizes: at one row every matmul reads its weights at 26-31 GB/s, on the bandwidth line (the head, Q6_K run as Q8_0, 13.2 ms, 15.1% of a token); at 128 rows the Q4_0 matmuls run at 0.91-0.99 int8 TOPS, so the kernels' ceiling is about 0.96 TOPS and the ridge 0.96 TOPS / 29 GB/s / 3.6 operations per byte per row = about 9.3 rows (measured knee 8-16); the requantized Q4_1 downs (Q8_0) reach 0.55 TOPS and cost 2.13 times their one-row time at 8 rows, against 1.28-1.63 for the Q4_0 matmuls. At 8 rows the imbalance is largest: the quantize step of `ffn_down`'s input (its slowest tile 2.1 ms, its fastest 0.06) and the matmuls' spread across tiles. Attention at n_kv 256: 0.67 ms at one row, 4.2 at 8, 62 at 128.

Manual checks:
- Always report `pp` and `tg` separately; repeat and interleave runs and note mean and median.
- Confirm IME is active: a `CPU_RISCV64_SPACEMIT` model buffer in `./build-ime/bin/llama-perplexity ... -lv 4` output (buffer name from `ggml/src/ggml-cpu/spacemit/ime.cpp:1477`).
- Where threads run: `ps -L -o tid,psr,comm -p $(pgrep llama-bench)` (`psr` 8-15 = A100).
- Compare speeds only within one session, interleaved and repeated: between sessions or harnesses tg128 moves by about 4-5%.

Experiment switches of the provider (environment variables; test-only unless noted; the context of each is in `plan.md`):

| Variable | Applies to | Effect |
|---|---|---|
| `FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=accel` or `gpu` | the device's ggml type at registration (`plan.md` §2.8) | the provider reports that type instead of IGPU (the default since 2026-10-27), with a warning, for comparisons: ACCEL was the type until 2026-10-26, GPU is E2's other value; since the revert of C5a and C5b (2026-10-28, `upstream-changes.md` U1, U2) `accel` runs as stock llama.cpp runs an ACCEL device: norms pinned to the CPU and flash attention off under `-fa auto`, so it no longer reproduces the ACCEL numbers of 2026-10-26; a type comparison only, with `-fa on` for speed; read at registration since 2026-11-03, so its warning shows in llama.cpp's tools (`plan.md` "S1 step 5 follow-up") |
| `FLAGOS_SPACEMIT_DISABLE=1` | everything | no SpacemiT device: llama.cpp runs on the X100 cores (the runner's `cpu` mode) |
| `FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1` | buffers (Linux) | the default 4 KiB pages; otherwise buffers from 2 MiB lie on huge pages |
| `FLAGOS_SPACEMIT_ATTN=split` or `one-chunk` | one-token attention from n_kv 512 (one stream, llama.cpp's cache layout, no ALiBi or softcap), parsed once per process (S1 step 2b), at registration since 2026-11-03 so an unknown value's warning shows in llama.cpp's tools (`plan.md` "S1 step 5 follow-up") | the default since 2026-10-14 (M2e.3 round 2, `plan.md`): the KV positions split over the tiles, the two tiles of a pair taking turns between staging and arithmetic, and the partial results merged in a second step (`paired` names it); `split`: the same without the turns (identical outputs); `one-chunk`: the one-chunk kernels as before round 2 (V summed in F16, so the outputs differ in rounding) |
| `GGML_FLAGOS_KERNEL_VARIANTS=0` | kernels on the K3 (the plugin's; read once) | the portable reference kernels on the AI cores (slow); tells a kernel error from an orchestration error; the deck's name since 2026-11-02 (S1 step 5; was `FLAGOS_SPACEMIT_TEST_REFERENCE=1`) |
| `GGML_FLAGOS_SYNC_LAUNCH=1` | `graph_compute` (`core/flagos-spacemit-diagnostics.cpp`); read at every graph | one launch per step, the host waiting for each (E3a; `structural-experiments.md` §5); any value but `0` or empty is on, as Denglin reads the same name; the runner's `e3`; was `FLAGOS_SPACEMIT_TEST_LAUNCH=per-step` until 2026-11-02 |
| `GGML_FLAGOS_STATS=<file>`, `-` or `1` | `graph_compute`, every graph (E3c; `structural-experiments.md` §5) | the per-graph counters, one JSON line per kind of graph appended to the file when the process ends (`-` or `1`: to stderr); the runner's `e3c`; was `FLAGOS_SPACEMIT_TEST_STATS` until 2026-11-02 |
| `GGML_FLAGOS_DISABLE_OPS=<op>[,<op>...]` | `spacemit_find_op`, so `supports_op` and `graph_compute` agree; parsed once | the named ggml ops (`MUL_MAT`, `FLASH_ATTN_EXT`, `GLU`, ... as `ggml_op_name` spells them; an unknown name warned once and ignored) are refused, so llama.cpp's scheduler runs them on the CPU and copies their operands out of our buffers (disabling `MUL_MAT` copies every weight back per graph: a diagnostic, not a mode); the runner's "ops disabled by name" check. A graph computed on the backend directly with such an op fails as any unclaimed op does, so under it the check program's `layer` and `plan` checks fail (seen on the Mac 2026-11-02); the scheduler is the diagnostic's place. Parsed at registration since 2026-11-03 (`plan.md` "S1 step 5 follow-up"), so its once-only warnings (the ops named, an unknown name) show in llama.cpp's tools too; before, the first refusal came inside llama.cpp's trial load for parameter fitting (on by default, `common/common.h:468`), which passes every line below ERROR on as DEBUG (`common/fit.cpp:49-53`), and the warnings were lost (seen on the Mac 2026-11-03). New 2026-11-02 (S1 step 5) |
| `GGML_FLAGOS_TRACE_LAUNCH=1` | `graph_compute`; read once | before each replay one `GGML_LOG_INFO` line per step (its number, contract name, node, op, shape, first operand), then `launch <n> steps` and the replay's status; with `GGML_FLAGOS_SYNC_LAUNCH=1` the core issues one replay per step and logs each one's status, so the first failing step is named. llama.cpp's tools drop the trace unless run with `-lv 4`: the provider logs at ggml's INFO level, which common's log callback maps to verbosity 4 (`common/log.cpp:443`) above the default threshold 3 (seen on the K3 2026-11-03); `llama-bench` drops every library log line, warnings and errors included, unless run with `-v` (`tools/llama-bench/llama-bench.cpp:2223`), while the counter files are written directly and need neither. New 2026-11-02 (S1 step 5) |
| `GGML_FLAGOS_PROVIDER_PATH=<path>` | the plugin loader (`core/flagos-spacemit-loader.cpp`) | the plugin library to `dlopen`, that path only, no fallback (test-only; S1 step 1's loader); was `FLAGOS_SPACEMIT_PLUGIN` until 2026-11-02 |
| `FLAGOS_SPACEMIT_STREAM=per-call` | executor on the K3 | a new spine-runtime stream per graph, releasing the AI cores in between (experiment switch, plan §2.4) |
| `FLAGOS_SPACEMIT_MM=split` | matmuls of several rows, parsed once per process (S1 step 2b; `fusion-and-splitting.md` §7.3) | `fit,staged` (the default since 2026-10-25): path B without the pair turns where TCM holds the batch's activations next to a 32-column slab, else path A where every tile gets 4 rows, else path C; `split`, for comparisons: ime.cpp's choice of path A and path C where ime.cpp takes path B (the default until 2026-10-25); the same bytes |
| `FLAGOS_SPACEMIT_QUANTIZE=shared` | matmuls (`fusion-and-splitting.md` §7.2) | one quantization of the activations for Q, K and V and one for gate and up; the plan cache off under it |
| `FLAGOS_SPACEMIT_PLAN=none` | `graph_compute`'s plan cache (`plan.md` "S1 design" §3, steps 3-3c; `structural-experiments.md` §5) | the full path on every graph: the contract check and the expansion, nothing cached; unset or `uid-trust` (the default since 2026-11-01): a split whose uid, node objects and configuration word are the last plan's takes its step list and workspace sizes from the cache without the check (the host work per generated token 630 to 46 µs on Qwen3-4B, run 12); the runner's `plan-n` |
| `GGML_FLAGOS_OP_STATS=<file>`, `-` or `1` | every graph (`fusion-and-splitting.md` §7.1) | per-op counters, written to the file (or stderr) when the process ends; `scripts/op-stats.py` reads them; was `FLAGOS_SPACEMIT_TEST_OP_STATS` until 2026-11-02 |
| `FLAGOS_SPACEMIT_TEST_TCM=<bytes>` | host executor (Mac) | a TCM stand-in per tile, so the 1-row path and paths A and B run; `flagos-check-spacemit` sets 393216 for its 1-row and path checks |
| `FLAGOS_SPACEMIT_TEST_THREADS=1` | host executor (Mac) | the tiles of each launch on threads, so pair waits wait (with the TCM stand-in: everything as on the AI cores) |
| `FLAGOS_SPACEMIT_TEST_FAIL_NODE=<n>` | tests | tile 0 reports a failure at step n |

Removed 2026-11-02 with the code they switched (`plan.md` "S1 cleanup before step 5"; their rows as of `cad674fda`): `FLAGOS_SPACEMIT_TEST_VIEW`, `TEST_TCM_SLICE`, `TEST_JOIN_FENCE` and `FLAGOS_SPACEMIT_TEST_LAUNCH=per-step-join` (E3b, E7, E7b), `TEST_Q4_0_SCALE` and the compile-time `FLAGOS_SPACEMIT_TEST_UPSTREAM_Q4_1_M1` (question 8's kernel reports), `FLAGOS_SPACEMIT_Q4_1=zero-point` (the in-tree Q4_1 layout, replaced by the requantization to Q8_0 on 2026-10-23), `FLAGOS_SPACEMIT_GEMV` (the 1-row strategy, `paired` since 2026-10-11), `FLAGOS_SPACEMIT_TAIL` and `FLAGOS_SPACEMIT_MM=rows`, `spacemit`, `fit` and `fit,mixed` (§7.3, `fit,staged` since 2026-10-25). The runs recorded above with them remain as recorded.

Renamed 2026-11-02 (S1 step 5, `plan.md` "S1 step 5 spec"; the deck's names, read by the core where it acts on them): ~~`FLAGOS_SPACEMIT_TEST_STATS`~~ `GGML_FLAGOS_STATS`, ~~`FLAGOS_SPACEMIT_TEST_OP_STATS`~~ `GGML_FLAGOS_OP_STATS`, ~~`FLAGOS_SPACEMIT_TEST_REFERENCE=1`~~ `GGML_FLAGOS_KERNEL_VARIANTS=0`, ~~`FLAGOS_SPACEMIT_TEST_LAUNCH=per-step`~~ `GGML_FLAGOS_SYNC_LAUNCH=1`, ~~`FLAGOS_SPACEMIT_PLUGIN`~~ `GGML_FLAGOS_PROVIDER_PATH`; new `GGML_FLAGOS_DISABLE_OPS` and `GGML_FLAGOS_TRACE_LAUNCH`. An old name that is still set is reported once ("`FLAGOS_SPACEMIT_X` is now `GGML_FLAGOS_Y`") and otherwise ignored, so a command from the history that sets one measures the default. The runs recorded above with the old names remain as recorded.

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
