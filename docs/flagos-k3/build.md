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

M1 skeleton (`plan.md` M1). Kept separate from `build/`, which stays the plain-CPU baseline.
```bash
python3 ggml/src/ggml-flagos/providers/spacemit/tools/spacemit_check.py --milestone m2a --build
```
The script configures `build-flagos/` with `-DGGML_FLAGOS=ON -DGGML_FLAGOS_DENGLIN=OFF -DGGML_FLAGOS_AMD=OFF -DGGML_FLAGOS_SPACEMIT=ON`, builds the provider, the FlagOS check tools, `test-backend-ops` and `llama-completion`, runs the milestone's checks and prints PASS/FAIL per check (INFO lines for measurements); logs go to `build-flagos/<milestone>-logs/`. From M2a the provider links spine-runtime from `~/spine-runtime` (override with `--spert-dir`). Without `--build` it only reruns the checks. Options: `--milestone m1|m2a`, `--model` (default `~/models/Qwen3-0.6B-Q4_0.gguf`), `--skip-support`, `--build-dir`, `--spert-dir`, `--tcm-dir` (default `~/tcmtest`, for the TCM checks).

Mac: same CMake options plus `-DGGML_METAL=OFF -DGGML_BLAS=OFF`; the provider compiles but finds no device (`flagos-check-spacemit` reports "device checks skipped").

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
| | `cpu` | 21.12 | 5.55 | 9.7740 (separate rerun; the runner's CPU run died silently once) |
| | `ime` | 79.10 | 10.99 | - |
| Qwen3-0.6B Q4_0 | `provider` | 333.27 | 26.75 | within 1% of `cpu` |
| | `cpu` | 155.56 | 28.17 | |
| | `ime` | 547.19 | 54.39 | |

- Prefill gains 2.1-2.4x over the X100 cores; generation does not (4B +5%, 0.6B -5%). Graph splits: 355 per token on 4B (178 CPU + 177 provider, about 5 round trips per layer, `sched-summary.py`), 277 on 0.6B; 1 without the provider.
- Kernels (`flagos-check-spacemit --bench`, Qwen3-4B FFN shapes): 1 row 0.60 ms for 14 MB of weights, about 23 GB/s, the AI cores' memory bandwidth; 4 rows cost about the same as 1; at 128 rows path A reaches 960 GFLOP/s and path C 648 GFLOP/s. The CPU column of that benchmark uses ggml's plain Q4_0 path, not the repacked kernels llama.cpp uses (`REPACK = 1`), so it understates the X100 cores.
- Estimated 4B generation budget (172 ms per token): Q4_0 matmuls on the AI cores about 85 ms; Q6_K output head on the X100 cores about 25 ms; other X100 work about 15 ms; the remaining about 45 ms are the 355 handoffs (about 0.13 ms each, consistent with X3b's 40 ms for 286 extra splits). `ime` avoids all handoffs: its CPU worker threads are pinned to the AI cores (`ggml-cpu/spacemit/ime.cpp:1692`), so the whole graph runs there.
- IME confirmed active (separate check, `-lv 4`): `CPU_RISCV64_SPACEMIT model buffer size = 2349.12 MiB` (repacked weights) next to `CPU_Mapped model buffer size = 2246.67 MiB` (the mmap'd GGUF), `graph splits = 1`. The single split (no backend handoffs) is the structural reason the IME path generates fastest. The baseline run itself lacked these lines (library INFO messages need `-lv 4`; fixed in the script).

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
