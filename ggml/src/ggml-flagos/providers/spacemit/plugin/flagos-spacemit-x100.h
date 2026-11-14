#pragma once

#include <cstdint>

// spine-runtime's X100 backend ("spacemit-k3-x100", tiles on the X100 cores) for load-time work, the weight repack
// under FLAGOS_SPACEMIT_REPACK_VIA=x100, its default here (plan.md "Repack on spert's X100 backend"). Without
// spine-runtime in the build both are trivial: no cores, and every index on the calling thread.

// the backend's cores (backend_info num_cores), looked up once per process; 0 when the backend is invalid or the build
// has no spine-runtime. The first call starts the backend (its workers, its /dev/tcm mapping)
uint32_t spacemit_x100_cores();

// runs fn(arg, i) for i in [0, n), n >= 2: a stream of n - 1 cores taken for this call, one launch of n - 1 tiles for
// 1..n-1 while the calling thread runs 0, then the sync; an index whose tile did not finish runs again on the calling
// thread (counted in *reran). The stream is released before the return. Returns false when nothing was launched (no
// backend, no stream, launch refused): then every index ran on the calling thread. fn must be idempotent per index
bool spacemit_x100_run(uint32_t n, void (*fn)(void * arg, uint32_t i), void * arg, uint32_t * reran);
