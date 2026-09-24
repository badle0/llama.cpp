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
- **VLEN:** 256 bits on both X100 and A100 (measured with `__riscv_vlenb()` on CPU 6 and CPU 8). FlagTree's QEMU CI uses `vlen=1024` with `SPACEMIT_EP_QEMU_SET_CORE_ARCH=0xA064`; that does not match this hardware's RVV length.
- The default `-march` from the fork's CMake (`rv64gcv_zfh_zvfh_zicbop_zihintpause`) is fully supported. llama.cpp's generic RVV repack implements only VLEN 256, which matches.

## 3. SpacemiT software present

| Present | Absent |
|---|---|
| `/usr/lib/libspine_tcm.so` (0.2.0) + `/usr/include/spine_tcm.h`, CMake/pkg-config files, `/usr/share/spine_tcm` | spine-runtime (`libspert`, `spert.hpp`) — install from release 0.6.0 |
| `/usr/include/spine_llm_engine.h`, `spine_llm_argparser.h`, `spine_vision_engine.h` (SpacemiT's own engine API; C++) | `/dev/tcm_sync_mem`, `/dev/hugetlb_1g` |
| `/dev/tcm` (char 10,259), kernel: `tcm 0.tcm: direct mmap phys 0x0 size 0x300000 block_size 0x60000 block_num 8` | |

## 4. TCM investigation (open)

TCM = tightly coupled memory: a per-core, software-managed SRAM scratchpad (like CUDA shared memory). Here: 8 blocks × 384 KiB, one per A100 core.

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

Pending, in order:
1. `diff -w /usr/include/spine_tcm.h ggml/src/ggml-cpu/spacemit/spine_tcm.h` — empty ⇒ same API.
2. Standalone test outside llama.cpp (`/tmp/tcmtest.c`, direct-link mode with the vendored header): print version, layout, per-block `cpu_affinity_mask/owner_tid/is_acquired/query`, then as an AI thread on CPU 8 call `mem_get(0)` and `try_wait(0, 1s)`. Build: `gcc -O1 -I ggml/src/ggml-cpu/spacemit /tmp/tcmtest.c -lspine_tcm -o /tmp/tcmtest`.
   - fails here too ⇒ library/driver issue (report to SpacemiT with this evidence)
   - succeeds ⇒ issue is in llama.cpp's thread/ordering around TCM
3. Check whether spine-runtime's TCM path (`runtime->shared_buffer()`, used by `ggml-spacemit`) works on this board.

What disabling TCM does (upstream IME path): results unchanged (kernels read from normal memory when `tcm_buffer == nullptr`, `ime.cpp` ~L405–440); likely slower; and thread pinning to specific A100 cores is skipped (it sits inside the `use_tcm` branch), so the scheduler places the opted-in threads. Treat such numbers as "IME2, no TCM, unpinned".

Relevant env vars (upstream IME path): `SPACEMIT_DISABLE_TCM`, `SPACEMIT_PERFER_CORE_ID`, `SPACEMIT_PERFER_CORE_ARCH`, `SPACEMIT_CORE_ARCH`, `SPACEMIT_MEM_BACKEND`. ggml-spacemit: `GGML_SPACEMIT_WORKERS`, `SPACEMIT_MEM_BACKEND` (`hugepage|posix|hugetlb`).
