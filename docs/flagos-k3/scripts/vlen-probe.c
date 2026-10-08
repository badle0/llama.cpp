// M0.4 (plan.md R1): RVV VLEN seen by a thread on an A100 core.
// "first": opt in and pin before any vector instruction in the thread, then read vlenb.
// "late":  read vlenb on an X100 core first, then opt in, pin, and read again.
// Build on the K3: gcc -O1 -march=rv64gc -pthread vlen-probe.c -o vlen-probe
// Run: ./vlen-probe; dmesg | grep hmp_set_ai_thread | tail -4
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>

#define AI_CPU 8

struct result {
    long tid;
    long vlenb_before;
    long vlenb_after;
    int  rc;
    int  cpu;
};

// csr 0xc22 is vlenb; the first access turns on vector state for the thread
static long read_vlenb(void) {
    long v;
    __asm__ volatile("csrr %0, 0xc22" : "=r"(v));
    return v;
}

// only plain syscalls here: libc string functions may use vector instructions
static int opt_in_and_pin(void) {
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd < 0) {
        return -1;
    }
    long n = write(fd, "0", 1);
    close(fd);
    if (n != 1) {
        return -2;
    }
    unsigned long mask[16];
    for (int i = 0; i < 16; i++) {
        ((volatile unsigned long *) mask)[i] = 0;
    }
    mask[AI_CPU / 64] = 1ul << (AI_CPU % 64);
    return syscall(SYS_sched_setaffinity, 0, sizeof(mask), mask) == 0 ? 0 : -3;
}

static void * first(void * arg) {
    struct result * r = arg;
    r->tid          = syscall(SYS_gettid);
    r->vlenb_before = -1;
    r->rc           = opt_in_and_pin();
    r->vlenb_after  = read_vlenb();
    r->cpu          = sched_getcpu();
    return NULL;
}

static void * late(void * arg) {
    struct result * r = arg;
    r->tid          = syscall(SYS_gettid);
    r->vlenb_before = read_vlenb();
    r->rc           = opt_in_and_pin();
    r->vlenb_after  = read_vlenb();
    r->cpu          = sched_getcpu();
    return NULL;
}

int main(void) {
    struct result a = { 0 }, b = { 0 };
    pthread_t     t;
    pthread_create(&t, NULL, first, &a);
    pthread_join(t, NULL);
    pthread_create(&t, NULL, late, &b);
    pthread_join(t, NULL);
    printf("first: tid=%ld rc=%d cpu=%d VLEN=%ld\n", a.tid, a.rc, a.cpu, a.vlenb_after * 8);
    printf("late:  tid=%ld rc=%d cpu=%d VLEN before=%ld after=%ld\n", b.tid, b.rc, b.cpu, b.vlenb_before * 8, b.vlenb_after * 8);
    return 0;
}
