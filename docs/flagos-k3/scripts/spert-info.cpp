// M0.2 and M0.4 (plan.md): what spine-runtime reports, the VLEN its workers see, and whether
// shared_buffer() is real TCM (inside a /dev/tcm mapping) or the runtime's heap fallback.
// Build on the K3: g++ -O2 -std=c++17 -I$HOME/spine-runtime/include spert-info.cpp -L$HOME/spine-runtime/lib -lspert -o spert-info
// Run: LD_LIBRARY_PATH=$HOME/spine-runtime/lib SPINE_TCM_RUNTIME_LOG=true ./spert-info
#include <spert.hpp>

#include <sched.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>

struct tile_info {
    long         vlenb;
    int          cpu;
    const void * tcm;
    size_t       tcm_size;
};

static void probe(spert::Context * ctx, tile_info * out) {
    long v;
    __asm__ volatile("csrr %0, 0xc22" : "=r"(v));
    tile_info & r = out[ctx->program_id()];
    r.vlenb       = v;
    r.cpu         = sched_getcpu();
    auto buf      = ctx->shared_buffer();
    r.tcm         = buf.data;
    r.tcm_size    = buf.size;
}

// name of the /proc/self/maps entry that contains p, or "" if none
static void mapping_of(const void * p, char * out, size_t n) {
    out[0]   = 0;
    FILE * f = fopen("/proc/self/maps", "r");
    if (!f) {
        return;
    }
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        uintptr_t lo = 0, hi = 0;
        if (sscanf(line, "%" SCNxPTR "-%" SCNxPTR, &lo, &hi) == 2 && (uintptr_t) p >= lo && (uintptr_t) p < hi) {
            const char * name = strchr(line, '/');
            snprintf(out, n, "%s", name ? name : "[anonymous]\n");
            break;
        }
    }
    fclose(f);
}

int main() {
    spert::BackendInfo bi = spert::backend_info();
    printf("backend_info: vlen=%zu bytes (VLEN %zu), num_cores=%zu, shared_mem_size=%zu, core_arch_id=0x%" PRIx64 "\n",
           bi.vlen, bi.vlen * 8, bi.num_cores, bi.shared_mem_size, (uint64_t) bi.core_arch_id);

    static tile_info out[64] = {};
    const uint32_t   n       = bi.num_cores > 0 && bi.num_cores <= 64 ? (uint32_t) bi.num_cores : 1;
    spert::Stream    s(n);
    spert::Future    f  = s.launch(spert::Grid{ n }, probe, out);
    auto             st = f.sync();
    printf("launch: grid=%u status=%d\n", n, (int) st);

    for (uint32_t i = 0; i < n; i++) {
        char where[256];
        mapping_of(out[i].tcm, where, sizeof(where));
        printf("tile %u: cpu=%d VLEN=%ld shared_buffer=%p size=%zu in %s", i, out[i].cpu, out[i].vlenb * 8, out[i].tcm,
               out[i].tcm_size, where[0] ? where : "no mapping\n");
    }
    return 0;
}
