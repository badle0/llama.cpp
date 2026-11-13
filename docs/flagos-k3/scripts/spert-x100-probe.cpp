// spert-x100-probe: facts about spine-runtime's spacemit-k3-x100 backend for load-time work (the weight repack, T6).
// Built and run by spert-x100-probe.sh. One fact per line, "<step> <what>: key=value ...". A failed step prints the
// failure and the probe goes on.
// usage: spert-x100-probe [step ...]  steps: info grant place work overhead idle info-default coexist oversize maps all
//        (default all, which leaves out oversize; any order, they run in the order above; "a,b" also works)
// env:   PROBE_MIB (1024, MiB per work buffer), REPS (5), PROBE_GRANT_S (20: seconds to wait for one stream before
//        it is reported blocked; 0 creates the stream on the calling thread with no wait limit), PROBE_CORE_IDS (a
//        comma list for one core_ids request in grant), PROBE_OVERSIZE=1 (lets the oversize step ask for more cores
//        than the x100 backend reports; such a request may wait in the shared arbiter and stay there)

#include <spert.hpp>

#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#ifdef __linux__
#    include <sched.h>
#    include <sys/syscall.h>
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;

static const char * k_x100_name = "spacemit-k3-x100";

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    if (!v || !*v) {
        return def;
    }
    const int n = atoi(v);
    return n >= 0 ? n : def;
}

static int cur_cpu() {
#ifdef __linux__
    return sched_getcpu();
#else
    return -1;
#endif
}

static long cur_tid() {
#ifdef __linux__
    return (long) syscall(SYS_gettid);
#else
    return (long) (std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0x7fffffff);
#endif
}

static double cpu_seconds() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

static std::string proc_threads() {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) {
        return "n/a";
    }
    std::string n = "?";
    char        line[256];
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "Threads:", 8) == 0) {
            n = std::to_string(atoi(line + 8));
            break;
        }
    }
    fclose(f);
    return n;
}

static const char * proc_tcm() {
    FILE * f = fopen("/proc/self/maps", "r");
    if (!f) {
        return "n/a";
    }
    const char * r = "no";
    char         line[4096];
    while (fgets(line, sizeof line, f)) {
        if (strstr(line, "/dev/tcm")) {
            r = "yes";
            break;
        }
    }
    fclose(f);
    return r;
}

static void maps(const char * label) {
    printf("maps %s: tcm=%s threads=%s cpu_s=%.3f\n", label, proc_tcm(), proc_threads().c_str(), cpu_seconds());
}

template <typename T> static std::string join(const std::vector<T> & v) {
    std::string s;
    for (size_t i = 0; i < v.size(); i++) {
        s += (i ? "," : "") + std::to_string(v[i]);
    }
    return s.empty() ? "-" : s;
}

static std::string fmt_list(const std::vector<double> & v, const char * f) {
    std::string s;
    char        b[64];
    for (size_t i = 0; i < v.size(); i++) {
        snprintf(b, sizeof b, f, v[i]);
        s += (i ? "," : "") + std::string(b);
    }
    return s.empty() ? "-" : s;
}

static double median(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static std::string cfg_label(const spert::StreamConfig & c) {
    return "n_cores=" + std::to_string(c.n_cores) + " core_ids=" + join(c.core_ids);
}

// ---- backends and streams ----

static spert::BackendHandle g_x100;
static bool                 g_x100_looked = false;

static spert::BackendHandle x100_backend() {
    if (!g_x100_looked) {
        g_x100        = spert::backend(k_x100_name);
        g_x100_looked = true;
    }
    return g_x100;
}

static void print_info(const char * label, const char * name, spert::BackendHandle h) {
    const spert::BackendInfo i = spert::backend_info(h);
    printf("info %s: name=%s valid=%d shared_mem_size=%zu vlen=%zu num_cores=%zu core_arch_id=%lld (0x%llx)\n", label,
           name, h.valid() ? 1 : 0, i.shared_mem_size, i.vlen, i.num_cores, (long long) i.core_arch_id,
           (unsigned long long) i.core_arch_id);
}

struct made_stream {
    spert::Stream * s       = nullptr;
    double          ms      = 0.0;
    bool            blocked = false;
};

static int g_grant_s   = 20;
static int g_abandoned = 0;

// a stream constructor may wait for its grant; wait at most g_grant_s, then leave it to its thread and go on
static made_stream make_stream(const std::function<spert::Stream *()> & make) {
    made_stream r;
    if (g_grant_s == 0) {
        const auto t0 = clk::now();
        r.s           = make();
        r.ms          = ms_since(t0);
        return r;
    }
    struct shared_state {
        std::mutex              m;
        std::condition_variable cv;
        bool                    done      = false;
        bool                    abandoned = false;
        spert::Stream *         s         = nullptr;
        double                  ms        = 0.0;
    };
    auto sh = std::make_shared<shared_state>();
    std::thread t([sh, make] {
        const auto      t0 = clk::now();
        spert::Stream * s  = make();
        const double    ms = ms_since(t0);
        std::lock_guard<std::mutex> lk(sh->m);
        if (sh->abandoned) {
            return;  // leaked on purpose: its grant may be live
        }
        sh->s    = s;
        sh->ms   = ms;
        sh->done = true;
        sh->cv.notify_one();
    });
    std::unique_lock<std::mutex> lk(sh->m);
    if (sh->cv.wait_for(lk, std::chrono::seconds(g_grant_s), [&] { return sh->done; })) {
        r.s  = sh->s;
        r.ms = sh->ms;
        lk.unlock();
        t.join();
    } else {
        sh->abandoned = true;
        r.blocked     = true;
        r.ms          = g_grant_s * 1000.0;
        lk.unlock();
        t.detach();
        g_abandoned++;
    }
    return r;
}

static double destroy_stream(spert::Stream *& s) {
    const auto t0 = clk::now();
    delete s;
    s = nullptr;
    return ms_since(t0);
}

// the x100 stream kept across place, work, overhead, idle and coexist: n_cores 0 (all available)
static const spert::StreamConfig g_best;
static spert::Stream *     g_x = nullptr;

static spert::Stream * x100_stream(const char * step) {
    if (g_x) {
        return g_x;
    }
    spert::BackendHandle h = x100_backend();
    if (!h.valid()) {
        printf("%s stream: skip, x100 backend invalid\n", step);
        return nullptr;
    }
    const spert::StreamConfig cfg = g_best;
    made_stream               m   = make_stream([h, cfg] { return new spert::Stream(h, cfg); });
    if (m.blocked) {
        printf("%s stream: %s blocked over %d s\n", step, cfg_label(cfg).c_str(), g_grant_s);
        return nullptr;
    }
    printf("%s stream: %s valid=%d core_count=%u create_ms=%.2f\n", step, cfg_label(cfg).c_str(),
           m.s->valid() ? 1 : 0, m.s->core_count(), m.ms);
    if (!m.s->valid() || m.s->core_count() == 0) {
        destroy_stream(m.s);
        return nullptr;
    }
    g_x = m.s;
    return g_x;
}

// ---- kernels and the work ----

static void place_tile(spert::Context * ctx, int * cpus, long * tids) {
    const uint32_t i = ctx->program_id(0);
    cpus[i]          = cur_cpu();
    tids[i]          = cur_tid();
}

static void empty_tile(spert::Context *) {}

static void part(size_t size, uint32_t n, uint32_t i, size_t & b, size_t & e) {
    b = size * i / n;
    e = size * (i + 1) / n;
}

static void transform(const uint8_t * src, uint8_t * dst, size_t b, size_t e) {
    for (size_t i = b; i < e; i++) {
        const uint8_t s = src[i];
        dst[i]          = (uint8_t) (((s << 4) | (s >> 4)) ^ 0x5a);
    }
}

// tile k does range first + k of nparts
static void work_tile(spert::Context * ctx, const uint8_t * src, uint8_t * dst, size_t size, uint32_t nparts,
                      uint32_t first) {
    size_t b, e;
    part(size, nparts, first + ctx->program_id(0), b, e);
    transform(src, dst, b, e);
}

static uint64_t checksum(const uint8_t * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    size_t   i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, p + i, 8);
        h = (h ^ w) * 1099511628211ull;
    }
    for (; i < n; i++) {
        h = (h ^ p[i]) * 1099511628211ull;
    }
    return h;
}

static uint8_t * g_src      = nullptr;
static uint8_t * g_dst      = nullptr;
static size_t    g_size     = 0;
static uint64_t  g_ref      = 0;
static bool      g_have_ref = false;

static uint8_t * map_buf(size_t n, const char *& advice) {
    void * p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }
#ifdef MADV_HUGEPAGE
    advice = madvise(p, n, MADV_HUGEPAGE) == 0 ? "ok" : "failed";
#else
    advice = "unavailable";
#endif
    return static_cast<uint8_t *>(p);
}

static bool ensure_buffers(const char * step) {
    if (g_src) {
        return true;
    }
    const size_t mib  = (size_t) std::max(1, env_int("PROBE_MIB", 1024));
    const size_t n    = mib << 20;
    const char * a1   = "-";
    const char * a2   = "-";
    const auto   t0   = clk::now();
    uint8_t *    src  = map_buf(n, a1);
    uint8_t *    dst  = src ? map_buf(n, a2) : nullptr;
    if (!src || !dst) {
        printf("%s buffers: mmap of 2 x %zu MiB failed\n", step, mib);
        if (src) {
            munmap(src, n);
        }
        return false;
    }
    for (size_t i = 0; i < n; i++) {
        src[i] = (uint8_t) ((i * 2654435761ull) >> 13);
    }
    memset(dst, 0, n);
    g_src  = src;
    g_dst  = dst;
    g_size = n;
    printf("%s buffers: mib=%zu each hugepage_advice=%s,%s alloc_touch_ms=%.1f\n", step, mib, a1, a2, ms_since(t0));
    return true;
}

static void run_threads(uint32_t n) {
    std::vector<std::thread> w;
    w.reserve(n - 1);
    for (uint32_t i = 1; i < n; i++) {
        w.emplace_back([i, n] {
            size_t b, e;
            part(g_size, n, i, b, e);
            transform(g_src, g_dst, b, e);
        });
    }
    size_t b, e;
    part(g_size, n, 0, b, e);
    transform(g_src, g_dst, b, e);
    for (std::thread & t : w) {
        t.join();
    }
}

enum variant { V_ONE, V_T8, V_T4, V_X, V_XC, V_COUNT };

static const char * k_variant_name[V_COUNT] = { "one", "threads8", "threads4", "x100", "x100+caller" };

// one timed pass on a zeroed destination; returns seconds, or a negative value when the launch failed
static double run_variant(int v, spert::Stream * s, uint64_t & sum, const char * step) {
    memset(g_dst, 0, g_size);
    const auto t0 = clk::now();
    if (v == V_ONE) {
        transform(g_src, g_dst, 0, g_size);
    } else if (v == V_T8) {
        run_threads(8);
    } else if (v == V_T4) {
        run_threads(4);
    } else {
        const uint32_t cc     = s->core_count();
        const uint32_t nparts = v == V_X ? cc : cc + 1;
        const uint32_t first  = v == V_X ? 0 : 1;
        spert::Future  fut    = s->launch(spert::Grid(cc), work_tile, (const uint8_t *) g_src, g_dst, g_size, nparts,
                                          first);
        if (v == V_XC) {
            size_t b, e;
            part(g_size, nparts, 0, b, e);
            transform(g_src, g_dst, b, e);
        }
        if (!fut.valid()) {
            printf("%s %s: launch returned an invalid future\n", step, k_variant_name[v]);
            return -1.0;
        }
        const spert::Status st = fut.sync();
        if (st != spert::Status::Ok) {
            printf("%s %s: sync status %s\n", step, k_variant_name[v], spert::to_string(st));
            return -1.0;
        }
    }
    const double secs = ms_since(t0) / 1000.0;
    sum               = checksum(g_dst, g_size);
    return secs;
}

// ---- steps ----

static void step_info() {
    print_info("x100", k_x100_name, x100_backend());
}

static void step_info_default() {
    print_info("default", "(default_backend)", spert::default_backend());
}

// one stream, created and destroyed; returns its core_count, 0 when invalid or blocked
static uint32_t try_grant(const char * step, spert::BackendHandle h, const spert::StreamConfig & cfg) {
    made_stream m = make_stream([h, cfg] { return new spert::Stream(h, cfg); });
    if (m.blocked) {
        printf("%s %s: blocked over %d s (left waiting)\n", step, cfg_label(cfg).c_str(), g_grant_s);
        return 0;
    }
    const bool     valid = m.s->valid();
    const uint32_t cc    = m.s->core_count();
    const double   d_ms  = destroy_stream(m.s);
    printf("%s %s: valid=%d core_count=%u create_ms=%.2f destroy_ms=%.2f\n", step, cfg_label(cfg).c_str(),
           valid ? 1 : 0, cc, m.ms, d_ms);
    return valid ? cc : 0;
}

static std::vector<int> env_ids(const char * name) {
    std::vector<int> ids;
    const char *     v = getenv(name);
    while (v && *v) {
        char *     end = nullptr;
        const long n   = strtol(v, &end, 10);
        if (end == v) {
            break;
        }
        ids.push_back((int) n);
        v = *end == ',' ? end + 1 : end;
    }
    return ids;
}

static bool g_oversize_named = false;

static void step_grant() {
    spert::BackendHandle h = x100_backend();
    if (!h.valid()) {
        printf("grant: skip, x100 backend invalid\n");
        return;
    }
    const uint32_t n = (uint32_t) spert::backend_info(h).num_cores;
    printf("grant num_cores: %u\n", n);

    // only requests of at most n cores: one that waits for more may block every later request
    std::vector<spert::StreamConfig> cfgs = { { 0, {} } };
    if (n >= 1) {
        cfgs.push_back({ n, {} });
    }
    if (n >= 2) {
        cfgs.push_back({ n / 2, {} });
    }
    const std::vector<int> ids = env_ids("PROBE_CORE_IDS");
    if (ids.empty()) {
        printf("grant core_ids: skip, the API gives no id list; set PROBE_CORE_IDS (a comma list) for one request\n");
    } else if (ids.size() > n) {
        printf("grant core_ids: skip, PROBE_CORE_IDS has %zu ids, more than num_cores %u\n", ids.size(), n);
    } else {
        cfgs.push_back({ (uint32_t) ids.size(), ids });
    }

    uint32_t best = 0;
    for (const spert::StreamConfig & cfg : cfgs) {
        best = std::max(best, try_grant("grant", h, cfg));
    }
    printf("grant largest: core_count=%u (the stream of the later steps asks n_cores=0)\n", best);
    if (!g_oversize_named || env_int("PROBE_OVERSIZE", 0) != 1) {
        printf("grant oversize: skipped (more than num_cores: n_cores 8, 16, core_ids 0-7); enable with step oversize "
               "and PROBE_OVERSIZE=1\n");
    }
}

// requests for more cores than the backend reports; they may wait in the shared arbiter, so only on request
static void step_oversize() {
    if (env_int("PROBE_OVERSIZE", 0) != 1) {
        printf("oversize: disabled, set PROBE_OVERSIZE=1 to run it\n");
        return;
    }
    spert::BackendHandle h = x100_backend();
    if (!h.valid()) {
        printf("oversize: skip, x100 backend invalid\n");
        return;
    }
    if (g_x) {
        const double d_ms = destroy_stream(g_x);
        printf("oversize x100 destroy: destroy_ms=%.2f (the requests start with no x100 stream held)\n", d_ms);
    }
    const uint32_t n = (uint32_t) spert::backend_info(h).num_cores;
    printf("oversize num_cores: %u\n", n);
    const std::vector<spert::StreamConfig> all = {
        { 8, {} },
        { 16, {} },
        { 8, { 0, 1, 2, 3, 4, 5, 6, 7 } },
    };
    int done = 0;
    for (const spert::StreamConfig & cfg : all) {
        if (cfg.n_cores > n) {
            try_grant("oversize", h, cfg);
            done++;
        }
    }
    if (done == 0) {
        printf("oversize: none, num_cores %u is 16 or more\n", n);
    }
}

static void step_place() {
    spert::Stream * s = x100_stream("place");
    if (!s) {
        return;
    }
    const uint32_t cc     = s->core_count();
    const long     caller = cur_tid();
    for (int run = 1; run <= 2; run++) {
        std::vector<int>  cpus(cc, -2);
        std::vector<long> tids(cc, 0);
        spert::Future     fut = s->launch(spert::Grid(cc), place_tile, cpus.data(), tids.data());
        if (!fut.valid()) {
            printf("place run%d: launch returned an invalid future\n", run);
            continue;
        }
        const spert::Status st = fut.sync();
        std::set<long>      distinct(tids.begin(), tids.end());
        printf("place run%d: status=%s grid=%u cpus=%s distinct_threads=%zu caller_thread_used=%d caller_cpu=%d\n", run,
               spert::to_string(st), cc, join(cpus).c_str(), distinct.size(), distinct.count(caller) ? 1 : 0,
               cur_cpu());
    }
}

static void step_work() {
    if (!ensure_buffers("work")) {
        return;
    }
    const int       reps = std::max(1, env_int("REPS", 5));
    spert::Stream * s    = x100_stream("work");
    if (!s) {
        printf("work x100: skip x100 and x100+caller, no valid x100 stream\n");
    }
    const int           nv = s ? V_COUNT : V_X;
    std::vector<double> mibs[V_COUNT];
    std::vector<int>    fails(V_COUNT, 0);
    uint64_t            sums[V_COUNT] = {};
    bool                same[V_COUNT];
    std::fill(same, same + V_COUNT, true);
    const double mib = (double) g_size / (1 << 20);
    for (int r = 0; r < reps; r++) {
        for (int v = 0; v < nv; v++) {
            uint64_t     sum  = 0;
            const double secs = run_variant(v, s, sum, "work");
            if (secs < 0) {
                fails[v]++;
                continue;
            }
            mibs[v].push_back(mib / secs);
            if (!g_have_ref) {
                g_ref      = sum;
                g_have_ref = true;
            }
            if (sum != g_ref) {
                same[v] = false;
            }
            sums[v] = sum;
        }
    }
    bool all = true;
    for (int v = 0; v < nv; v++) {
        printf("work %s: median_mib_s=%.0f mib_s=%s fails=%d checksum=0x%016llx%s\n", k_variant_name[v],
               median(mibs[v]), fmt_list(mibs[v], "%.0f").c_str(), fails[v], (unsigned long long) sums[v],
               same[v] ? "" : " MISMATCH");
        all = all && same[v] && fails[v] == 0;
    }
    printf("work checksum: %s ref=0x%016llx grid=%u reps=%d\n", all ? "all equal" : "NOT all equal",
           (unsigned long long) g_ref, s ? s->core_count() : 0, reps);
}

static void step_overhead() {
    const int k_batches = 5;
    const int k_rounds  = 400;

    spert::Stream * s = x100_stream("overhead");
    if (s) {
        const uint32_t      cc    = s->core_count();
        int                 fails = 0;
        std::vector<double> us;
        for (int b = 0; b < k_batches; b++) {
            const auto t0 = clk::now();
            for (int r = 0; r < k_rounds; r++) {
                spert::Future       fut = s->launch(spert::Grid(cc), empty_tile);
                const spert::Status st  = fut.sync();
                if (!fut.valid() || st != spert::Status::Ok) {
                    if (fails++ == 0) {
                        printf("overhead x100_launch_sync: first failure valid=%d status=%s\n", fut.valid() ? 1 : 0,
                               spert::to_string(st));
                    }
                }
            }
            us.push_back(ms_since(t0) * 1000.0 / k_rounds);
        }
        printf("overhead x100_launch_sync: grid=%u median_us=%.2f batches_us=%s fails=%d\n", cc, median(us),
               fmt_list(us, "%.2f").c_str(), fails);
    } else {
        printf("overhead x100_launch_sync: skip, no valid x100 stream\n");
    }

    std::vector<double> us;
    for (int b = 0; b < k_batches; b++) {
        const auto t0 = clk::now();
        for (int r = 0; r < k_rounds; r++) {
            std::thread t[7];
            for (std::thread & x : t) {
                x = std::thread([] {});
            }
            for (std::thread & x : t) {
                x.join();
            }
        }
        us.push_back(ms_since(t0) * 1000.0 / k_rounds);
    }
    printf("overhead threads7_create_join: median_us=%.2f batches_us=%s\n", median(us), fmt_list(us, "%.2f").c_str());
}

static void step_idle() {
    spert::Stream * s = x100_stream("idle");
    if (s) {
        const double c0 = cpu_seconds();
        std::this_thread::sleep_for(std::chrono::seconds(2));
        printf("idle alive: cpu_s_in_2s=%.3f threads=%s\n", cpu_seconds() - c0, proc_threads().c_str());
        const double d_ms = destroy_stream(g_x);
        printf("idle destroy: destroy_ms=%.2f threads=%s\n", d_ms, proc_threads().c_str());
    } else {
        printf("idle alive: skip, no valid x100 stream\n");
    }
    const double c0 = cpu_seconds();
    std::this_thread::sleep_for(std::chrono::seconds(2));
    printf("idle no_stream: cpu_s_in_2s=%.3f threads=%s\n", cpu_seconds() - c0, proc_threads().c_str());
}

static void step_coexist() {
    spert::Stream * x = x100_stream("coexist");
    made_stream     a = make_stream([] { return new spert::Stream(8u); });
    if (a.blocked) {
        printf("coexist a100: Stream(8) blocked over %d s (left waiting)\n", g_grant_s);
    } else {
        printf("coexist a100: Stream(8) valid=%d core_count=%u create_ms=%.2f\n", a.s->valid() ? 1 : 0,
               a.s->core_count(), a.ms);
        if (a.s->valid() && a.s->core_count() > 0) {
            const uint32_t    cc = a.s->core_count();
            std::vector<int>  cpus(cc, -2);
            std::vector<long> tids(cc, 0);
            spert::Future     fut = a.s->launch(spert::Grid(cc), place_tile, cpus.data(), tids.data());
            if (fut.valid()) {
                const spert::Status st = fut.sync();
                std::set<long>      distinct(tids.begin(), tids.end());
                printf("coexist a100 place: status=%s grid=%u cpus=%s distinct_threads=%zu\n", spert::to_string(st), cc,
                       join(cpus).c_str(), distinct.size());
            } else {
                printf("coexist a100 place: launch returned an invalid future\n");
            }
        }
    }
    maps("coexist both alive");
    if (x && ensure_buffers("coexist")) {
        uint64_t sum = 0;
        if (!g_have_ref) {
            run_variant(V_ONE, nullptr, g_ref, "coexist");
            g_have_ref = true;
        }
        const double secs = run_variant(V_X, x, sum, "coexist");
        if (secs >= 0) {
            printf("coexist x100 work: grid=%u mib_s=%.0f checksum=0x%016llx %s\n", x->core_count(),
                   (double) g_size / (1 << 20) / secs, (unsigned long long) sum, sum == g_ref ? "equal" : "MISMATCH");
        }
    } else if (!x) {
        printf("coexist x100 work: skip, no valid x100 stream\n");
    }
    if (a.s) {
        const double d_ms = destroy_stream(a.s);
        printf("coexist a100 destroy: destroy_ms=%.2f\n", d_ms);
    }
    if (g_x) {
        const double d_ms = destroy_stream(g_x);
        printf("coexist x100 destroy: destroy_ms=%.2f\n", d_ms);
    }
}

struct step {
    const char * name;
    void (*fn)();
    bool in_all;
};

static const step k_steps[] = {
    { "info", step_info, true },
    { "grant", step_grant, true },
    { "place", step_place, true },
    { "work", step_work, true },
    { "overhead", step_overhead, true },
    { "idle", step_idle, true },
    { "info-default", step_info_default, true },
    { "coexist", step_coexist, true },
    { "oversize", step_oversize, false },
    { "maps", nullptr, false },
};

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    g_grant_s = env_int("PROBE_GRANT_S", 20);

    std::set<std::string> want;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        size_t      p = 0;
        while (p <= a.size()) {
            const size_t q = std::min(a.find(',', p), a.size());
            if (q > p) {
                want.insert(a.substr(p, q - p));
            }
            p = q + 1;
        }
    }
    const bool all = want.empty() || want.count("all");
    for (const std::string & w : want) {
        bool known = w == "all";
        for (const step & s : k_steps) {
            known = known || w == s.name;
        }
        if (!known) {
            fprintf(stderr,
                    "unknown step %s; steps: info grant place work overhead idle info-default coexist oversize maps all\n",
                    w.c_str());
            return 2;
        }
    }
    g_oversize_named = want.count("oversize") > 0;
    auto selected    = [&](const step & s) { return s.fn && ((all && s.in_all) || want.count(s.name) > 0); };

    std::string list;
    for (const step & s : k_steps) {
        if (selected(s)) {
            list += (list.empty() ? "" : " ") + std::string(s.name);
        }
    }
    const char * core_ids = getenv("PROBE_CORE_IDS");
    printf("start: pid=%d steps=%s probe_mib=%d reps=%d grant_wait_s=%d probe_core_ids=%s probe_oversize=%d\n",
           (int) getpid(), list.c_str(), env_int("PROBE_MIB", 1024), env_int("REPS", 5), g_grant_s,
           core_ids && *core_ids ? core_ids : "-", env_int("PROBE_OVERSIZE", 0));
    maps("start");

    for (const step & s : k_steps) {
        if (!selected(s)) {
            continue;
        }
        const auto t0 = clk::now();
        s.fn();
        printf("step %s: wall_ms=%.1f\n", s.name, ms_since(t0));
        maps((std::string("after ") + s.name).c_str());
    }

    if (g_x) {
        const double d_ms = destroy_stream(g_x);
        printf("end x100 destroy: destroy_ms=%.2f\n", d_ms);
    }
    if (g_src) {
        munmap(g_src, g_size);
        munmap(g_dst, g_size);
    }
    printf("done: abandoned_grants=%d\n", g_abandoned);
    fflush(stdout);
    if (g_abandoned) {
        _exit(3);  // a thread still waits inside spert; skip the runtime's exit-time teardown
    }
    return 0;
}
