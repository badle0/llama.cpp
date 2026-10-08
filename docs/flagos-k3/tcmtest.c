// Standalone libspine_tcm probe for the K3 TCM abort (see k3-hardware.md section 4).
//
// Reproduces, outside llama.cpp, the sequence of the upstream IME path:
//   /proc/set_ai_thread opt-in -> pin to A100 core -> spine_tcm_mem_get(id) once
//   -> per graph compute: spine_tcm_mem_try_wait(id, 1 s) ... spine_tcm_mem_release(id)
// (ggml/src/ggml-cpu/spacemit/ime.cpp:1692-1742, spine_mem_pool.cpp:699-709)
//
// Build on the K3 from the repo root. Default = header-only loader mode (dlopen), same as llama.cpp:
//   gcc -O1 -Wall -pthread -I ggml/src/ggml-cpu/spacemit docs/flagos-k3/tcmtest.c -ldl -o /tmp/tcmtest
// Direct-link mode:
//   gcc -O1 -Wall -pthread -DSPINE_TCM_DIRECT_LINK -I ggml/src/ggml-cpu/spacemit docs/flagos-k3/tcmtest.c -lspine_tcm -o /tmp/tcmtest-direct
//
// Run:
//   /tmp/tcmtest all                      # info, A, D, F, B, E, C, info; each case in its own process
//   /tmp/tcmtest <info|A|B|C|D|E|F> [id] [timeout_us]
//   SPINE_TCM_RUNTIME_LOG=true /tmp/tcmtest all   # library logs its driver version and sync backend
//
// Cases (id defaults to 0 = A100 core 8):
//   A  opt-in + pin, try_wait -> release                  (handoff pair alone)
//   B  opt-in + pin, mem_get once, then 2x (try_wait -> release), then mem_free   (our llama.cpp order)
//   C  no opt-in, no pin, try_wait -> release             (does the library care where the caller runs?)
//   D  opt-in + pin, mem_get -> mem_free                  (the header's documented example)
//   E  worker thread (opt-in + pin) mem_get; unpinned control thread try_wait -> release; worker mem_free
//      (spacemit-com/llama.cpp order since 5a23f07a45: ime.cpp tcm_mem_wait_all/release_all)
//   F  no opt-in, no pin, mem_get -> mem_free             (spine-runtime/libspert order, per disassembly;
//      libspert silently falls back to heap if mem_get returns NULL)

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "spine_tcm.h"

#define AICPU_ID_OFFSET 8  // A100 cores are CPUs 8-15 (aicpu_id_offset in the IME log)
#define CASE_ALARM_S    10 // kill a case that hangs instead of timing out

static int    g_id         = 0;
static size_t g_timeout_us = 1000 * 1000; // same value as spine_mem_pool.cpp:704

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static long self_tid(void) {
    return (long) syscall(SYS_gettid);
}

static void dump_block(const char * when) {
    spine_tcm_block_info_t bi;
    memset(&bi, 0, sizeof(bi));
    int rc = spine_tcm_block_info(g_id, &bi);
    int q  = spine_tcm_mem_query(g_id);
    printf("  [%-18s] id=%d info_rc=%d va=%p size=%zu phys=0x%" PRIx64 " affinity=0x%" PRIx64
           " owner_tid=%d is_acquired=%d query=%d | self tid=%ld cpu=%d\n",
           when, g_id, rc, bi.va, bi.size, bi.phys_addr, bi.cpu_affinity_mask, bi.owner_tid, bi.is_acquired, q,
           self_tid(), sched_getcpu());
}

static int open_lib(void) {
    if (spine_tcm_open_handle(NULL) != 0) {
        printf("  spine_tcm_open_handle failed\n");
        return -1;
    }
    if (!spine_tcm_is_available()) {
        printf("  spine_tcm_is_available() == 0\n");
        return -1;
    }
    return 0;
}

// Same as bind_ai_thread() in ime.cpp: only when currently on an X100 core.
static void opt_in_ai_thread(void) {
    int cpu = sched_getcpu();
    if (cpu >= AICPU_ID_OFFSET) {
        printf("  already on cpu %d, skipping /proc/set_ai_thread\n", cpu);
        return;
    }
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd < 0) {
        printf("  open /proc/set_ai_thread failed: %s\n", strerror(errno));
        return;
    }
    ssize_t n = write(fd, "0", 1);
    printf("  /proc/set_ai_thread write -> %zd%s%s\n", n, n < 0 ? " " : "", n < 0 ? strerror(errno) : "");
    close(fd);
}

static void pin_to_ai_core(void) {
    int       cpu = g_id + AICPU_ID_OFFSET;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    int s = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    sched_yield();
    printf("  pin to cpu %d -> %s, now on cpu %d\n", cpu, s == 0 ? "ok" : strerror(s), sched_getcpu());
}

static void setup(int opt_in, int pin) {
    printf("  start: tid=%ld cpu=%d opt_in=%d pin=%d\n", self_tid(), sched_getcpu(), opt_in, pin);
    if (opt_in) {
        opt_in_ai_thread();
    }
    if (pin) {
        pin_to_ai_core();
    }
}

static void * timed_try_wait(void) {
    errno     = 0;
    double t0 = now_ms();
    void * p  = spine_tcm_mem_try_wait(g_id, g_timeout_us);
    int    e  = errno;
    printf("  try_wait(%d, %zu us) -> %p after %.1f ms (errno=%d %s)\n", g_id, g_timeout_us, p, now_ms() - t0, e,
           e ? strerror(e) : "");
    return p;
}

static int do_release(void) {
    errno  = 0;
    int rc = spine_tcm_mem_release(g_id);
    printf("  release(%d) -> %d (errno=%d)\n", g_id, rc, errno);
    return rc;
}

static int case_info(void) {
    printf("  version=%s available=%d\n", spine_tcm_version(), spine_tcm_is_available());
    spine_tcm_mem_info_t mi;
    memset(&mi, 0, sizeof(mi));
    int rc = spine_tcm_mem_info(&mi);
    printf("  layout rc=%d blk_size=%zu blk_num=%zu is_fake_tcm=%d\n", rc, mi.blk_size, mi.blk_num, mi.is_fake_tcm);
    int saved = g_id;
    for (size_t i = 0; i < mi.blk_num; i++) {
        g_id = (int) i;
        dump_block("info");
    }
    g_id = saved;
    return rc == 0 ? 0 : 1;
}

// try_wait -> release, with or without opt-in/pinning
static int case_handoff(int opt_in, int pin) {
    setup(opt_in, pin);
    dump_block("before try_wait");
    void * p = timed_try_wait();
    dump_block("after try_wait");
    if (p == NULL) {
        return 1;
    }
    int rc = do_release();
    dump_block("after release");
    return rc == 0 ? 0 : 1;
}

static int case_A(void) {
    return case_handoff(1, 1);
}

static int case_C(void) {
    return case_handoff(0, 0);
}

// llama.cpp order: mem_get once, then try_wait/release per graph compute
static int case_B(void) {
    setup(1, 1);
    dump_block("before mem_get");
    errno     = 0;
    void * p0 = spine_tcm_mem_get(g_id);
    printf("  mem_get(%d) -> %p (errno=%d)\n", g_id, p0, errno);
    dump_block("after mem_get");
    int fail = p0 == NULL;
    for (int iter = 0; iter < 2 && !fail; iter++) {
        printf("  -- graph compute %d\n", iter);
        void * p = timed_try_wait();
        dump_block("after try_wait");
        if (p == NULL) {
            fail = 1;
            break;
        }
        if (p != p0) {
            printf("  note: try_wait buffer %p != mem_get buffer %p\n", p, p0);
        }
        fail |= do_release() != 0;
        dump_block("after release");
    }
    if (p0 != NULL) {
        printf("  mem_free(%d) -> %d\n", g_id, spine_tcm_mem_free(g_id));
        dump_block("after mem_free");
    }
    return fail;
}

// mem_get -> mem_free, with or without opt-in/pinning
static int case_get_free(int opt_in, int pin) {
    setup(opt_in, pin);
    dump_block("before mem_get");
    errno    = 0;
    void * p = spine_tcm_mem_get(g_id);
    printf("  mem_get(%d) -> %p (errno=%d)\n", g_id, p, errno);
    dump_block("after mem_get");
    if (p == NULL) {
        return 1;
    }
    int rc = spine_tcm_mem_free(g_id);
    printf("  mem_free(%d) -> %d\n", g_id, rc);
    dump_block("after mem_free");
    return rc == 0 ? 0 : 1;
}

static int case_D(void) {
    return case_get_free(1, 1);
}

static int case_F(void) {
    return case_get_free(0, 0);
}

// Case E: a pinned AI worker owns the mem_get reference; the unpinned control thread does the handoff.
static pthread_mutex_t e_mu    = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  e_cv    = PTHREAD_COND_INITIALIZER;
static int             e_state = 0; // 0 = start, 1 = worker has mem_get result, 2 = control thread done
static void *          e_buf   = NULL;

static void e_set_state(int s) {
    pthread_mutex_lock(&e_mu);
    e_state = s;
    pthread_cond_broadcast(&e_cv);
    pthread_mutex_unlock(&e_mu);
}

static void e_wait_state(int s) {
    pthread_mutex_lock(&e_mu);
    while (e_state < s) {
        pthread_cond_wait(&e_cv, &e_mu);
    }
    pthread_mutex_unlock(&e_mu);
}

static void * case_E_worker(void * arg) {
    (void) arg;
    printf("  [worker]\n");
    setup(1, 1);
    errno = 0;
    e_buf = spine_tcm_mem_get(g_id);
    printf("  [worker] mem_get(%d) -> %p (errno=%d)\n", g_id, e_buf, errno);
    dump_block("worker mem_get");
    e_set_state(1);
    e_wait_state(2);
    if (e_buf != NULL) {
        printf("  [worker] mem_free(%d) -> %d\n", g_id, spine_tcm_mem_free(g_id));
        dump_block("worker mem_free");
    }
    return NULL;
}

static int case_E(void) {
    printf("  [control] tid=%ld cpu=%d (no opt-in, no pin)\n", self_tid(), sched_getcpu());
    pthread_t worker;
    if (pthread_create(&worker, NULL, case_E_worker, NULL) != 0) {
        printf("  pthread_create failed\n");
        return 2;
    }
    e_wait_state(1);
    int fail = e_buf == NULL;
    if (!fail) {
        dump_block("control before");
        void * p = timed_try_wait();
        dump_block("control try_wait");
        if (p == NULL) {
            fail = 1;
        } else {
            if (p != e_buf) {
                printf("  note: try_wait buffer %p != mem_get buffer %p\n", p, e_buf);
            }
            fail |= do_release() != 0;
            dump_block("control release");
        }
    }
    e_set_state(2);
    pthread_join(worker, NULL);
    return fail;
}

static int run_case(const char * name) {
    if (open_lib() != 0) {
        return 2;
    }
    if (!strcmp(name, "info")) {
        return case_info();
    }
    if (!strcmp(name, "A")) {
        return case_A();
    }
    if (!strcmp(name, "B")) {
        return case_B();
    }
    if (!strcmp(name, "C")) {
        return case_C();
    }
    if (!strcmp(name, "D")) {
        return case_D();
    }
    if (!strcmp(name, "E")) {
        return case_E();
    }
    if (!strcmp(name, "F")) {
        return case_F();
    }
    fprintf(stderr, "unknown case '%s'\n", name);
    return 2;
}

// Each case runs in a fresh process so TCM state cannot leak between cases
// (leaks across processes still show up in the next case's "before" dump).
static int run_in_child(const char * name) {
    printf("=== case %s (id=%d, timeout=%zu us)\n", name, g_id, g_timeout_us);
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        alarm(CASE_ALARM_S);
        int rc = run_case(name);
        fflush(stdout);
        _exit(rc);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    const char * verdict = "FAIL";
    if (WIFSIGNALED(status)) {
        printf("=== case %s: killed by signal %d%s\n\n", name, WTERMSIG(status),
               WTERMSIG(status) == SIGALRM ? " (hung)" : "");
        return 1;
    }
    if (WEXITSTATUS(status) == 0) {
        verdict = "PASS";
    } else if (WEXITSTATUS(status) == 2) {
        verdict = "ERROR";
    }
    printf("=== case %s: %s\n\n", name, verdict);
    return WEXITSTATUS(status);
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char * which = argc > 1 ? argv[1] : "all";
    if (argc > 2) {
        g_id = atoi(argv[2]);
    }
    if (argc > 3) {
        g_timeout_us = (size_t) strtoull(argv[3], NULL, 10);
    }
    if (strcmp(which, "all") != 0) {
        return run_in_child(which);
    }
    const char * order[] = { "info", "A", "D", "F", "B", "E", "C", "info" };
    int          fails   = 0;
    for (size_t i = 0; i < sizeof(order) / sizeof(order[0]); i++) {
        fails += run_in_child(order[i]) != 0;
    }
    return fails ? 1 : 0;
}
