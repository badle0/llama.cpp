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
mkdir -p ~/spine-runtime && cd ~/spine-runtime
wget https://github.com/spacemit-com/spine-runtime/releases/download/0.6.0/spine-runtime.riscv64.0.6.0.tar.gz
tar xzf spine-runtime.riscv64.0.6.0.tar.gz --strip-components=1        # include/ lib/ must exist

git clone -b agent/ggml-spacemit-backend https://github.com/spacemit-com/llama.cpp ~/llama.cpp-spacemit
cd ~/llama.cpp-spacemit
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_SPACEMIT=ON -DSPERT_DIR=$HOME/spine-runtime
cmake --build build --target test-backend-ops llama-bench llama-cli 2>&1 | tee build.log
LD_LIBRARY_PATH=$HOME/spine-runtime/lib ./build/bin/test-backend-ops -o MUL_MAT 2>&1 | tail -5
```
Status: not yet run. It turns `GGML_CPU_RISCV64_SPACEMIT` off automatically. The backend name reported by its registry is unverified; run without `-b` to list backends.

## 6. Adding the SpacemiT provider (plan, not done)

Minimal wiring, mirroring AMD/Denglin:
- `ggml/src/ggml-flagos/providers/spacemit/{flagos-spacemit-api.h, flagos-spacemit.cpp, provider.cmake}`
- `ggml/CMakeLists.txt`: `option(GGML_FLAGOS_SPACEMIT ...)` next to the AMD/Denglin options (L201–204)
- `ggml/src/ggml-flagos/CMakeLists.txt`: `include(providers/spacemit/provider.cmake)` when ON
- `flagos-registry.cpp` `flagos_compiled_providers()` (L69–78): push the provider under `#ifdef GGML_FLAGOS_HAVE_SPACEMIT`
- Then configure `build/` with `-DGGML_FLAGOS_SPACEMIT=ON`. M1 done when the registry lists 1 FlagOS device and model output is unchanged.

Registry check (lists backends and device counts):
```bash
python3 - <<'EOF'
import ctypes, glob
g = ctypes.CDLL(glob.glob('build/bin/libggml.*')[0])
g.ggml_backend_reg_get.restype = ctypes.c_void_p; g.ggml_backend_reg_get.argtypes = [ctypes.c_size_t]
g.ggml_backend_reg_name.restype = ctypes.c_char_p; g.ggml_backend_reg_name.argtypes = [ctypes.c_void_p]
g.ggml_backend_reg_dev_count.restype = ctypes.c_size_t; g.ggml_backend_reg_dev_count.argtypes = [ctypes.c_void_p]
for i in range(g.ggml_backend_reg_count()):
    r = g.ggml_backend_reg_get(i); print(g.ggml_backend_reg_name(r).decode(), g.ggml_backend_reg_dev_count(r))
EOF
```

## 7. Models and benchmarks

```bash
pip install -U huggingface_hub --break-system-packages
HF_ENDPOINT=https://hf-mirror.com hf download <repo> --include '*Q4_0*' --local-dir ~/models
```
SpacemiT's doc benchmarks Qwen3-0.6B Q4_0 and qwen35 2B Q4_1 (`docs/build-riscv64-spacemit.md`); A100 IME supports Q2_K–Q6_K, Q4_0/1, Q5_0/1, Q8_0.

```bash
M=~/models/<file>.gguf
./build/bin/llama-bench -m $M -t 8 -p 128 -n 64                                   # plain RVV, X100
SPACEMIT_DISABLE_TCM=1 ./build-ime/bin/llama-bench -m $M -t 8 -p 128 -n 64        # upstream IME2, A100, no TCM
```
- Always report `pp` and `tg` separately; repeat and interleave runs (A B A B) and note mean and median.
- Confirm IME is active: `SPACEMIT_DISABLE_TCM=1 ./build-ime/bin/llama-cli -m $M -p Hi -n 16 -no-cnv 2>&1 | grep -iE 'model buffer|SPACEMIT'` should show a SPACEMIT buffer.
- Where threads run: `ps -L -o tid,psr,comm -p $(pgrep llama-bench)` (`psr` 8–15 = A100).

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
