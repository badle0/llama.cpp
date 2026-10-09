// vfault.c: do vector loads survive page faults intact on this board?
// Background: one llama-perplexity run crashed in ggml_vec_dot_q4_1_q8_1 with a load through a2 reported at address 0,
// although a2 had been set from a valid pointer 3 instructions earlier and nothing in between writes it. Either a2 was
// changed from outside the program, or the fault address was reported wrongly (build.md §7).
// Modes (each round drops the page mappings, so the loads fault again):
//   v  16-byte vector load at the start of a page of <file>: the fault is on the first byte; checks a2 (default)
//   s  the same with a scalar byte load (control)
//   x  16-byte vector load that starts 8 bytes before the end of a mapped page and continues into an unmapped one,
//      so the fault happens in the middle of the instruction. Uses anonymous memory of <file>'s size (zero pages, no
//      huge pages). Checks a2 and that all 16 bytes were loaded. A wrongly reported fault address would crash it
//      (run it under segv.so to see the registers).
// Build (K3): gcc -O2 -march=rv64gcv -o vfault vfault.c -lpthread
// Usage: vfault <file> [threads=8] [rounds=100] [v|s|x]
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

static const uint8_t *   g_map;
static size_t            g_size, g_page;
static long              g_threads;
static int               g_rounds;
static char              g_mode;
static pthread_barrier_t g_barrier;
static unsigned long     g_probes, g_bad;

// a2 = canary; 16-byte vector load at p (page fault); return a2
static inline unsigned long probe_vector(const uint8_t * p, unsigned long canary) {
    unsigned long out;
    __asm__ volatile("mv a2, %[c]\n\t"
                     "vsetivli zero, 16, e8, m1, ta, ma\n\t"
                     "vle8.v v1, (%[p])\n\t"
                     "mv %[o], a2\n\t"
                     : [o] "=&r"(out) : [c] "r"(canary), [p] "r"(p) : "a2", "v1", "memory");
    return out;
}

// the same with a scalar byte load
static inline unsigned long probe_scalar(const uint8_t * p, unsigned long canary) {
    unsigned long out;
    __asm__ volatile("mv a2, %[c]\n\t"
                     "lbu t0, 0(%[p])\n\t"
                     "mv %[o], a2\n\t"
                     : [o] "=&r"(out) : [c] "r"(canary), [p] "r"(p) : "a2", "t0", "memory");
    return out;
}

// a2 = canary; v1 = all 0xff; 16-byte vector load at p that faults part way; store the loaded bytes; return a2
static inline unsigned long probe_cross(const uint8_t * p, unsigned long canary, uint8_t * bytes) {
    unsigned long out;
    __asm__ volatile("mv a2, %[c]\n\t"
                     "vsetivli zero, 16, e8, m1, ta, ma\n\t"
                     "vmv.v.i v1, -1\n\t"
                     "vle8.v v1, (%[p])\n\t"
                     "vse8.v v1, (%[b])\n\t"
                     "mv %[o], a2\n\t"
                     : [o] "=&r"(out) : [c] "r"(canary), [p] "r"(p), [b] "r"(bytes) : "a2", "v1", "memory");
    return out;
}

static void * worker(void * arg) {
    const long   id    = (long) arg;
    const size_t pages = g_size / g_page;
    for (int r = 0; r < g_rounds; r++) {
        pthread_barrier_wait(&g_barrier);  // mappings were dropped: first touches fault again
        unsigned long n = 0, bad = 0;
        if (g_mode == 'x') {
            for (size_t pg = 2 * (size_t) id; pg + 1 < pages; pg += 2 * (size_t) g_threads) {
                const unsigned long canary = 0xC0FFEE0000000000UL ^ (pg << 8) ^ (unsigned long) id;
                uint8_t             bytes[16];
                (void) *(const volatile uint8_t *) (g_map + pg * g_page);  // map page pg; page pg + 1 stays unmapped
                const unsigned long got     = probe_cross(g_map + (pg + 1) * g_page - 8, canary, bytes);
                int                 missing = 0;
                for (int k = 0; k < 16; k++) {
                    missing += bytes[k] != 0;  // zero pages: every loaded byte is 0
                }
                if ((got != canary || missing) && bad++ < 5) {
                    fprintf(stderr, "CORRUPTED: round %d thread %ld page %zu: a2 = %#lx (expected %#lx), %d of 16 bytes wrong\n",
                            r, id, pg, got, canary, missing);
                }
                n++;
            }
        } else {
            for (size_t pg = (size_t) id; pg < pages; pg += (size_t) g_threads) {
                const uint8_t *     p      = g_map + pg * g_page + (pg * 64) % (g_page - 64);
                const unsigned long canary = 0xC0FFEE0000000000UL ^ (pg << 8) ^ (unsigned long) id;
                const unsigned long got    = g_mode == 'v' ? probe_vector(p, canary) : probe_scalar(p, canary);
                if (got != canary && bad++ < 5) {
                    fprintf(stderr, "CORRUPTED: round %d thread %ld page %zu: a2 = %#lx, expected %#lx\n", r, id, pg, got, canary);
                }
                n++;
            }
        }
        __atomic_add_fetch(&g_probes, n, __ATOMIC_RELAXED);
        __atomic_add_fetch(&g_bad, bad, __ATOMIC_RELAXED);
        pthread_barrier_wait(&g_barrier);
        if (id == 0) {
            madvise((void *) g_map, g_size, MADV_DONTNEED);
        }
    }
    return NULL;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file> [threads=8] [rounds=100] [v|s|x]\n", argv[0]);
        return 2;
    }
    g_threads = argc > 2 ? atol(argv[2]) : 8;
    g_rounds  = argc > 3 ? atoi(argv[3]) : 100;
    g_mode    = argc > 4 ? argv[4][0] : 'v';
    if (g_threads < 1 || g_threads > 64 || (g_mode != 'v' && g_mode != 's' && g_mode != 'x')) {
        fprintf(stderr, "threads must be 1-64, mode v, s or x\n");
        return 2;
    }
    const int   fd = open(argv[1], O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) {
        perror(argv[1]);
        return 1;
    }
    g_page = (size_t) sysconf(_SC_PAGESIZE);
    g_size = (size_t) st.st_size / g_page * g_page;
    if (g_mode == 'x') {
        g_map = mmap(NULL, g_size, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (g_map != MAP_FAILED) {
            madvise((void *) g_map, g_size, MADV_NOHUGEPAGE);  // 4 KiB pages, so the second page faults on its own
        }
    } else {
        g_map = mmap(NULL, g_size, PROT_READ, MAP_PRIVATE, fd, 0);
    }
    if (g_map == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    pthread_barrier_init(&g_barrier, NULL, (unsigned) g_threads);
    struct rusage ru0, ru1;
    getrusage(RUSAGE_SELF, &ru0);
    pthread_t t[64];
    for (long i = 0; i < g_threads; i++) {
        pthread_create(&t[i], NULL, worker, (void *) i);
    }
    for (long i = 0; i < g_threads; i++) {
        pthread_join(t[i], NULL);
    }
    getrusage(RUSAGE_SELF, &ru1);
    const char * what = g_mode == 'v' ? "vector loads" : g_mode == 's' ? "scalar loads" : "vector loads faulting mid-instruction";
    printf("%s, %ld threads: %lu probes, %ld page faults, %ld preemptions; %lu corrupted\n", what, g_threads, g_probes,
           ru1.ru_minflt - ru0.ru_minflt, ru1.ru_nivcsw - ru0.ru_nivcsw, g_bad);
    return g_bad != 0;
}
