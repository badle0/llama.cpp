# SpacemiT K3 board — facts and investigations

Everything here was measured on the board (2026-09-24) unless marked otherwise.

## 1. System

| Item | Value |
|---|---|
| OS / kernel | Bianbu 4.0 (Ubuntu-based), Linux 6.18.3-generic riscv64 |
| Toolchain | GCC/G++ 15.2.0 (Bianbu build; assembles IME1 and IME2), CMake 4.2.3, Ninja 1.13, git 2.53, Python 3.14 |
| Memory / disk | 31 GiB RAM, no swap; ~104 GB free on `/` |
| Access | via TLS tunnel; `ssh k3` alias on the Mac (hostname kept out of this repo) |
| Persistence | unconfirmed whether the disk survives platform resets; keep work pushed to the fork |

## 2. Cores

| CPUs | Core | marchid | arch id | Max clock | Default use |
|---|---|---|---|---|---|
| 0–7 | Spacemit X100 | `0x8000000058000002` | `0x5064` | 2.4 GHz | all processes (`Cpus_allowed_list: 0-7`, `nproc` = 8) |
| 8–15 | Spacemit A100 | `0x8000000041000002` | `0xA064` | 2.0 GHz | AI cores; gated |

- `lscpu -e` shows 16 CPUs; plain `taskset -c 8` fails with EINVAL; no `isolcpus` on the kernel cmdline.
- **Gate:** a thread writes `"0"` to `/proc/set_ai_thread` (world-writable), then `sched_setaffinity` to 8–15 works. Verified with a test program (`before: cpu 6` → `after: cpu 8`). Upstream: `bind_ai_thread()` in `ggml-cpu/spacemit/ime.cpp` (~L1670); spine-runtime does it internally (`libspert.so` contains the string).
- **ISA:** identical on X100 and A100 (full-string diff empty): `rv64imafdcvh` + zba/zbb/zbs, zfh/zfhmin, zvfh/zvfhmin, zicbop/zicboz, zihintpause, vector crypto, etc. No `zvfbfwma`. IME is a vendor extension and is not listed.
- **VLEN differs by core type: X100 256 bits, A100 1024 bits** (measured 2026-10-07). `docs/flagos-k3/scripts/vlen-probe.c`: a thread that opts in and pins to CPU 8 before any vector instruction reads 1024; a thread that reads 256 on an X100 core reads 1024 after moving to CPU 8. `scripts/spert-info.cpp`: spine-runtime reports `vlen=128` bytes, and all 8 tiles (CPUs 8-15) read 1024. The kernel message `hmp_set_ai_thread: ... vector has been enabled already!!!` appears when a thread used vector instructions before opting in; such threads still get VLEN 1024. The earlier "256 on CPU 8" entry was wrong (probably read before the thread moved). FlagTree's QEMU CI setting `vlen=1024` for arch `0xA064` matches.
- Consequence: a thread's `vlenb` changes when it migrates between core types. Read it on the thread that runs the kernel. ggml-cpu reads it once at start-up on the main thread (`ggml-cpu/ggml-cpu.c:742`) and picks RVV repack layouts by it at load time (`ggml-cpu/repack.cpp:4591-4712`), i.e. for 256, even when worker threads later run on A100 cores. SpacemiT's recommended builds set `-DGGML_CPU_REPACK=OFF` (reason not stated; this may be why).
- The default `-march` from the fork's CMake (`rv64gcv_zfh_zvfh_zicbop_zihintpause`) is fully supported on both core types.

## 3. SpacemiT software present

| Present | Absent |
|---|---|
| `/usr/lib/libspine_tcm.so` (0.2.0) + `/usr/include/spine_tcm.h`, CMake/pkg-config files, `/usr/share/spine_tcm` | spine-runtime (`libspert`, `spert.hpp`) — install from release 0.6.0 |
| `/usr/include/spine_llm_engine.h`, `spine_llm_argparser.h`, `spine_vision_engine.h` (SpacemiT's own engine API; C++) | `/dev/tcm_sync_mem`, `/dev/hugetlb_1g` |
| `/dev/tcm` (char 10,259), kernel: `tcm 0.tcm: direct mmap phys 0x0 size 0x300000 block_size 0x60000 block_num 8` | |

## 4. TCM investigation (resolved 2026-10-07)

TCM = tightly coupled memory: a per-core, software-managed SRAM scratchpad (like CUDA shared memory). Here: 8 blocks × 384 KiB, one per A100 core.

**Resolution.** TCM works on this board with the installed libspine_tcm. Two causes, both on our side of the library:
1. **Stale ownership:** all 8 blocks were recorded as acquired by threads of a dead earlier run in the persistent `/dev/shm/tcm_sync_standalone`; the library never detects dead owners. Fix: `~/tcmtest/tcmrelease --apply` (built from `tcmrelease.c` on the board; force-releases only blocks whose owner thread no longer exists; dry run without `--apply`). Needed again after any IME run that crashes or is interrupted mid-compute.
2. **Too many threads:** with TCM on, the IME path pins compute thread *i* to AI core 8+*i* and aborts for *i* ≥ 8 (`ime.cpp:1705`); test-backend-ops uses 16 threads. Use `OMP_THREAD_LIMIT=8` for test-backend-ops and `-t ≤8` for llama-bench/cli.

With both: `OMP_THREAD_LIMIT=8 ./build-ime/bin/test-backend-ops -o MUL_MAT -b CPU` → **1198/1198 passed, TCM enabled**, 0 blocks left acquired. Details and evidence below. Scope: `test-backend-ops` allocates every tensor, weights included, in the backend's default buffer (`tests/test-backend-ops.cpp:1384-1393`), i.e. the plain CPU buffer, never the IME extra buffer type. So this run validates thread pinning and TCM acquire/release on A100 cores, not the IME kernels.

Symptom (upstream IME path, `build-ime/`):
```
CPU_RISCV64_SPACEMIT: tcm is available, blk_size: 393216, blk_num: 8, is_fake_tcm: 0
CPU_RISCV64_SPACEMIT: num_cores: 16, num_perfer_cores: 8, perfer_core_arch_id: a064, ... use_ime2: 1, mem_backend: HPAGE, cpu_mask: ff00, aicpu_id_offset: 8
CPU_RISCV64_SPACEMIT: alloc_chunk: open(/dev/tcm_sync_mem) failed, errno=2
CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap
ggml/src/ggml-cpu/spacemit/ime.cpp:1728: wait tcm buffer failed for cpu_id: 0      <- abort
```
With `SPACEMIT_DISABLE_TCM=1`: 1198/1198 `MUL_MAT` tests pass.

Established (with evidence):
- `/dev/tcm_sync_mem` is expected by upstream `ggml/src/ggml-cpu/spacemit/spine_mem_pool.cpp` L30/L582, introduced by `ggml-org/llama.cpp` commit `81b0d882` (PR #22863, 2026-05-14, author alex-spacemit); unchanged in the mentor's fork and in upstream master. Not modified by any FlagOS commit.
- It is used **only** for shared memory holding thread barriers (`ime_env.cpp` L294–301), with a heap fallback. It is not the TCM acquisition path.
- The abort comes from `spine_mem_pool_tcm_mem_wait` → `spine_tcm_mem_try_wait(cpu_id, 1000*1000)` (`spine_mem_pool.cpp` L704) → board `libspine_tcm.so` `spine_tcm_runtime_mem_try_wait`.
- No contention: no process holds `/dev/tcm` (`fuser`, `ps` empty).
- ABI matches: the board library exports all 11 `spine_tcm_runtime_*` symbols (`@@SPINE_TCM_0`); the first 40 lines of the header diff are formatting only.

Offline findings (2026-09-25, from `spacemit-com/llama.cpp` @ `4e782bc` source; items marked *disasm* come from disassembling the spine-runtime 0.6.0 release binaries and are unverified on the board):
- Call order differs from SpacemiT's current code. Ours (`ime.cpp:1719-1731`): each worker, including the main thread as worker 0, opts in, pins, calls `mem_get` once and then `try_wait`/`release` on every compute. SpacemiT since commit `5a23f07a45` (2026-07-20): workers pin and call `mem_get`; the unpinned caller thread (a "control thread", `ggml-cpu.c:3260`) calls `try_wait` for all ids before kickoff and `release_all` after (`ime.cpp:1818-1850`, `ggml-cpu.c:3475-3479`). So get-then-try_wait on the same id is fine in itself; calling `try_wait` from an AI-opted-in thread pinned to that core is what differs.
- ggml-spacemit never calls libspine_tcm itself. *disasm:* `libspert.so` calls only `mem_get` (once per core at pool init, on the host thread before the workers start) and `mem_free` (at teardown), never `try_wait`. If `mem_get` returns NULL it silently falls back to `posix_memalign`, so ggml-spacemit's size check cannot detect missing TCM.
- The 0.6.0 release ships `spacemit-tcm_3.0.1_riscv64.deb`; the board has libspine_tcm 0.2.0. *disasm (3.0.1):* `try_wait` synchronises through a sync backend: `/dev/tcm_sync_mem`, then POSIX shm `/tcm_sync_standalone`, then `/tmp/tcm_sync_standalone.shm`. The 3.0.1 postinst creates `/dev/shm/tcm_sync_standalone` and force-releases all blocks. `SPINE_TCM_RUNTIME_LOG=true` makes the library log its driver version and sync mode. On the board (2026-10-07): the installed library contains the same backend strings (`/dev/tcm_sync_mem`, `shm_open /tcm_sync_standalone`, `/tmp/tcm_sync_standalone.shm`, and the tcm-block "unsupported" message), and `/dev/shm/tcm_sync_standalone` exists (root, 0600, 4096 B, dated Sep 24). So a missing sync backend is unlikely to be the cause.
- tcmtest, first run (2026-10-07, direct-link, board header, id 0, `try_wait` timeout arg 1000000): info, D, F **pass** (`mem_get`/`mem_free` work); A, B, C, E **hang** — killed by the 10 s guard alarm despite the "1 s" timeout. So call order is not the cause (SpacemiT's order E hangs too); `try_wait` itself blocks. Hypotheses: (a) the timeout unit is ms (3.0.1 log string `timeout:%d(ms)`), making llama.cpp's wait ~17 min; (b) stale "acquired" state in the persistent `/dev/shm/tcm_sync_standalone` left by an aborted run (3.0.1's postinst force-releases all blocks via `spacemit-tcm-smi -c`). Force-release changes shared board state — needs agreement first.
- **Root cause found (2026-10-07):** all 8 blocks report `is_acquired=1` with owner tids 3913899, 3914014–3914020 (one thread plus seven created together = an 8-thread ggml pool, presumably an earlier IME run that died without releasing). The ownership lives in the persistent `/dev/shm/tcm_sync_standalone` (records tagged `TCM2`, 0x38 B per block, owner tid at +0x2c, e.g. `0x3bb8ab` = 3913899); `fuser` shows no holder. The library does not detect dead owners. Timeout: argument 1000 → 12.2 ms, 2000 → 24.2 ms, i.e. ~12 µs per unit (measured, not from source), so llama.cpp's `1000*1000` gives up after ~12 s — hence the original abort. Recovery: force-release only blocks whose owner thread no longer exists (tool: `~/tcmtest/tcmrelease.c` on the board; dry run by default, `--apply` to release). Dry run confirmed all 8 owners dead (`ps -eLo` empty); after `--apply`, tcmtest A–F all **pass** and no block is left acquired afterwards. IME `test-backend-ops` with TCM enabled, first try: aborted again and left all 8 blocks acquired. Cause confirmed from the log (`ime.cpp:1706: thread_n 8…12 exceeds perfer_core_ids size 8`): test-backend-ops uses `hardware_concurrency()` = 16 threads (`tests/test-backend-ops.cpp:10638`), and with TCM on, threads 8–15 hit `GGML_ABORT("thread_n %d exceeds perfer_core_ids size")` (`ime.cpp:1705`) after threads 0–7 had acquired their blocks; nothing caps the thread count at the 8 AI cores. Workaround: `OMP_THREAD_LIMIT=8` for test-backend-ops (OpenMP build), `-t ≤8` for llama-bench/cli; run `tcmrelease --apply` before IME runs. Re-test with `OMP_THREAD_LIMIT=8`, TCM on: ~1150 log lines of `MUL_MAT` cases all OK (incl. q4_K, q6_K) before the run was cut off mid-case, most likely by an interrupt (no abort, no OOM in `dmesg`, 0 blocks left acquired). That case alone (`q4_K m=6144 n=1 k=2048`) passes both with TCM (1.31 s wall) and without (1.91 s). Full uninterrupted suite with TCM: **1198/1198 passed**, 0 blocks left acquired (2026-10-07). The kernel's `hmp_set_ai_thread: … vector has been enabled already!!!` messages appear on every opt-in and seem harmless (tests pass). Worth reporting to SpacemiT: no dead-owner recovery; the timeout unit does not match `timeout_us`.
- The vendored `spine_tcm.h` is byte-identical in SpacemiT's branch; the 3.0.1 header differs only in the loader handle's storage (API unchanged).

Remaining (non-blocking):
1. `diff -w /usr/include/spine_tcm.h ggml/src/ggml-cpu/spacemit/spine_tcm.h` from the repo root (first attempt ran from `~`).
2. ~~Does spine-runtime's `shared_buffer()` get real TCM?~~ Yes (2026-10-07, spine-runtime 0.6.3, `scripts/spert-info.cpp`): the library maps all 8 blocks (`MakeDriverBuffer (v2)`), tile *i* runs on CPU 8+*i* and its `shared_buffer()` (384 KiB) lies inside a `/dev/tcm` mapping; 0 blocks left acquired afterwards. Block affinity masks are per core pair: blocks 0-1 `0x300` (CPUs 8-9), 2-3 `0xc00`, 4-5 `0x3000`, 6-7 `0xc000`.
3. Report to SpacemiT: no dead-owner recovery in the sync file; `try_wait` timeout unit is ~12 µs, not the `timeout_us` the header names.
4. Upstream IME path: aborting for threads ≥ 8 instead of capping or running them without TCM is worth a fix or an upstream issue (not our file — discuss with the mentor first).

Standalone probe: [`tcmtest.c`](tcmtest.c) (build/run lines in its header; cases A–F).

What disabling TCM does (upstream IME path): results unchanged (kernels read from normal memory when `tcm_buffer == nullptr`, `ime.cpp` ~L405–440); likely slower; and thread pinning to specific A100 cores is skipped (it sits inside the `use_tcm` branch), so the scheduler places the opted-in threads. Treat such numbers as "IME2, no TCM, unpinned".

Relevant env vars (upstream IME path): `SPACEMIT_DISABLE_TCM`, `SPACEMIT_PERFER_CORE_ID`, `SPACEMIT_PERFER_CORE_ARCH`, `SPACEMIT_CORE_ARCH`, `SPACEMIT_MEM_BACKEND`. ggml-spacemit: `GGML_SPACEMIT_WORKERS`, `SPACEMIT_MEM_BACKEND` (`hugepage|posix|hugetlb`).
