// segv.c: crash reporter. On SIGSEGV/SIGBUS prints the fault address, the faulting thread's registers and a
// backtrace to stderr, then exits with code 139. riscv64 Linux only.
// Build (K3): gcc -O2 -shared -fPIC -o segv.so segv.c -ldl
// Use: LD_PRELOAD=/path/to/segv.so <program> ...   (child processes inherit it)
// Self-test: LD_PRELOAD=./segv.so sh -c 'kill -SEGV $$'   (a signal sent by kill shows the sender's pid as address)
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>

static const char * const k_names[32] = {
    "pc", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6" };

static void put(const char * s) { (void) !write(2, s, strlen(s)); }

static void on_fault(int sig, siginfo_t * si, void * context) {
    char line[512];
    // riscv64 Linux: the machine context starts with the 32 general registers, slot 0 holding the pc
    const unsigned long * r = (const unsigned long *) &((ucontext_t *) context)->uc_mcontext;
    Dl_info info;
    memset(&info, 0, sizeof info);
    dladdr((void *) r[0], &info);
    snprintf(line, sizeof line, "\n=== segv.so: signal %d, fault address %p, thread %ld\n=== pc %#lx = %s + %#lx (%s + %#lx)\n",
             sig, si->si_addr, (long) syscall(SYS_gettid), r[0],
             info.dli_fname ? info.dli_fname : "?", info.dli_fbase ? r[0] - (unsigned long) info.dli_fbase : 0UL,
             info.dli_sname ? info.dli_sname : "?", info.dli_saddr ? r[0] - (unsigned long) info.dli_saddr : 0UL);
    put(line);
    for (int i = 0; i < 32; i++) {
        snprintf(line, sizeof line, "%-4s %016lx%s", k_names[i], r[i], i % 4 == 3 ? "\n" : "   ");
        put(line);
    }
    void * frames[32];
    backtrace_symbols_fd(frames, backtrace(frames, 32), 2);
    _exit(139);
}

__attribute__((constructor)) static void segv_install(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_fault;
    sa.sa_flags     = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
}
