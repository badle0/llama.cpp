#!/usr/bin/env bash
# The memory probe: how much memory a process holds while a model generates on the provider, against the CPU and the
# upstream IME build. The question (SpacemiT, 2026-11): does the provider hold the original weights next to its
# repacked copy? Each run is one llama-bench (tg only, one repetition, -v for llama's buffer lines), per MODEL:
#   prov-mmap    the provider, -t 8, -mmp 1 (the weights file mapped)
#   prov-nommap  the provider, -t 8, -mmp 0 (the weights read into memory)
#   cpu          FLAGOS_SPACEMIT_DISABLE=1, the same binary, -t 1, -mmp 1, NGEN_CPU tokens. Never more than one
#                thread: a CPU run with several threads on the X100 cores is wrong at random (PROJECT.md section 6)
#   ime          only when STEPS names it: the upstream SpacemiT build (build.md section 3), -t 8, -mmp 1. It uses
#                TCM unless SPACEMIT_DISABLE_TCM=1 is set; release dead blocks first (~/tcmtest/tcmrelease --apply)
# A python sampler starts each run and, every INTERVAL while it lives, reads /proc/<pid>/status (VmRSS, RssAnon,
# RssFile, RssShmem, VmHWM, VmSwap), /proc/<pid>/smaps_rollup (Pss_Anon, Pss_File, AnonHugePages) and /proc/meminfo
# (MemAvailable, Cached), all kB, into <run>.samples.csv; every 2 s it keeps /proc/<pid>/smaps (the last 16, as
# <run>.smaps.0-15, their times in <run>.smaps.idx). Without /proc (macOS) it reads only the RSS from ps ("mode: ps").
# The summary reads only files in <out>, so it runs again on a copied folder. "Steady" is the last sample at or
# before 90% of the run's lifetime (generation is the long last phase, teardown the very end).
# The script builds nothing and does not touch /dev/tcm; run the /dev/tcm check of PROJECT.md section 6 before it,
# and keep it apart from other IME runs with TCM. -mmp is deprecated for --load-mode but still read
# (tools/llama-bench/llama-bench.cpp:844).
#
# usage: mem-probe.sh run [out]   (default out: ~/mem-probe)
#        mem-probe.sh summary <out>
# env:   MODELS (default "$HOME/models/Qwen3-4B-Q4_0.gguf"; space-separated), BUILD (default <repo>/build-flagos),
#        BUILD_IME (default <repo>/build-ime), NGEN (384), NGEN_CPU (24), INTERVAL (seconds between samples, 0.25),
#        TIMEOUT (seconds per run, 3600; the run is killed after it), STEPS (default "prov-mmap prov-nommap cpu"; add
#        ime to run it), DRY=1 (print the commands only), PROC_ROOT (default /proc; for tests of the sampler only)
set -u

# an abort must end the process: without this ggml starts gdb for a backtrace and waits for it
export GGML_NO_BACKTRACE=1

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
MODELS=${MODELS:-"$HOME/models/Qwen3-4B-Q4_0.gguf"}
BUILD=${BUILD:-$REPO/build-flagos}
BUILD_IME=${BUILD_IME:-$REPO/build-ime}
NGEN=${NGEN:-384}
NGEN_CPU=${NGEN_CPU:-24}
INTERVAL=${INTERVAL:-0.25}
TIMEOUT=${TIMEOUT:-3600}
STEPS=${STEPS:-"prov-mmap prov-nommap cpu"}
DRY=${DRY:-}
PROC_ROOT=${PROC_ROOT:-/proc}

unset FLAGOS_SPACEMIT_DISABLE GGML_FLAGOS_DISABLE_OPS GGML_FLAGOS_KERNEL_VARIANTS FLAGOS_SPACEMIT_REPACK_THREADS

step_on() { # step_on <step>: whether STEPS names it
    case " $STEPS " in *" $1 "*) return 0 ;; esac
    return 1
}

sample() { # sample <prefix> <command...>: run it under the sampler
    python3 - "$1" "$TIMEOUT" "$INTERVAL" "$PROC_ROOT" "${@:2}" <<'PY'
import os, subprocess, sys, time

STATUS = ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmHWM", "VmSwap")
ROLLUP = ("Pss_Anon", "Pss_File", "AnonHugePages")
MEMINFO = ("MemAvailable", "Cached")
COLS = ("t",) + STATUS + ROLLUP + MEMINFO
SMAPS_EVERY = 2.0
SMAPS_KEEP = 16

def parse_kv(text, keys):
    # "Key:   123 kB" lines -> {key: kB}
    vals = {}
    for line in text.splitlines():
        k, sep, rest = line.partition(":")
        f = rest.split()
        if sep and k in keys and f and f[0].isdigit():
            vals[k] = int(f[0])
    return vals

def read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return None

def meminfo(root):
    return parse_kv(read(os.path.join(root, "meminfo")) or "", MEMINFO)

def ps_rss(pid):
    try:
        r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, universal_newlines=True)
    except OSError:
        return {}
    f = r.stdout.split()
    return {"VmRSS": int(f[0])} if f and f[0].isdigit() else {}

def main(argv):
    prefix, timeout, interval, root, cmd = argv[0], float(argv[1]), float(argv[2]), argv[3], argv[4:]
    sysf = open(prefix + ".sys", "w")

    def note(s):
        sysf.write(s + "\n")
        sysf.flush()

    def mem_note(label):
        m = meminfo(root)
        note("%s: MemAvailable %s Cached %s" % (label, m.get("MemAvailable", ""), m.get("Cached", "")))

    mem_note("before")
    note("command: " + " ".join(cmd))
    # take a leading "env A=B" here, so the pid is the program's even if env does not exec it
    env = dict(os.environ)
    if cmd and cmd[0] == "env":
        cmd = cmd[1:]
        while cmd and "=" in cmd[0]:
            k, _, v = cmd.pop(0).partition("=")
            env[k] = v
    with open(prefix + ".out", "w") as out, open(prefix + ".err", "w") as err:
        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env)
        except OSError as e:
            err.write("cannot start: %s\n" % e)
            open(prefix + ".rc", "w").write("127\n")
            return
    pid = proc.pid
    pdir = os.path.join(root, str(pid))
    proc_mode = os.path.exists(os.path.join(pdir, "status")) or os.path.exists(os.path.join(root, "meminfo"))
    mode = "proc" if proc_mode else "ps"
    note("pid: %d" % pid)
    note("mode: " + mode)

    samples = open(prefix + ".samples.csv", "w")
    samples.write("# mode: %s\n%s\n" % (mode, ",".join(COLS)))
    idx = open(prefix + ".smaps.idx", "w")
    n_snap, next_snap, next_tick, killed = 0, 0.0, t0, False
    while proc.poll() is None:
        el = time.monotonic() - t0
        if el > timeout:
            proc.kill()
            killed = True
            break
        vals, snap = {}, None
        if proc_mode:
            vals.update(parse_kv(read(os.path.join(pdir, "status")) or "", STATUS))
            vals.update(parse_kv(read(os.path.join(pdir, "smaps_rollup")) or "", ROLLUP))
            vals.update(meminfo(root))
            if el >= next_snap:
                snap = read(os.path.join(pdir, "smaps"))
                while next_snap <= el:
                    next_snap += SMAPS_EVERY
        else:
            vals.update(ps_rss(pid))
        # exited during the reads: the values may be partial, drop them
        if proc.poll() is not None:
            break
        samples.write(",".join(["%.3f" % el] + [str(vals.get(c, "")) for c in COLS[1:]]) + "\n")
        samples.flush()
        if snap:
            k = n_snap % SMAPS_KEEP
            with open("%s.smaps.%d" % (prefix, k), "w") as f:
                f.write(snap)
            idx.write("%d %.3f\n" % (k, el))
            idx.flush()
            n_snap += 1
        next_tick += interval
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_tick = time.monotonic()
    end = time.monotonic()
    rc = proc.wait()
    note("lifetime: %.3f" % (end - t0))
    if killed:
        note("timeout: killed after %g s" % timeout)
    open(prefix + ".rc", "w").write("%d\n" % rc)
    time.sleep(2)
    mem_note("after")

if __name__ == "__main__":
    main(sys.argv[1:])
PY
}

run() { # run <prefix> <command...>
    echo "$(date +%H:%M:%S) $*"
    [ -n "$DRY" ] && return 0
    sample "$@"
}

summary() {
    python3 - "$1" <<'PY'
import csv, glob, os, re, sys

out = sys.argv[1]
ALL_STEPS = ("prov-mmap", "prov-nommap", "cpu", "ime")
TAG = {"prov-mmap": "m", "prov-nommap": "n", "cpu": "c", "ime": "i"}
MIB = 1024.0

def text(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return None

def parse_samples(t):
    mode, rows, cols = "?", [], None
    for line in t.splitlines():
        if line.startswith("#"):
            m = re.match(r"#\s*mode:\s*(\S+)", line)
            if m:
                mode = m.group(1)
            continue
        f = line.split(",")
        if cols is None:
            cols = f
            continue
        row = {}
        for c, v in zip(cols, f):
            try:
                row[c] = float(v)
            except ValueError:
                row[c] = None
        rows.append(row)
    return mode, rows

def parse_sys(t):
    d = {}
    for line in (t or "").splitlines():
        k, sep, v = line.partition(": ")
        if sep:
            d[k] = v
    return d

def sys_mem(v):
    m = re.search(r"MemAvailable (\d+)", v or "")
    return int(m.group(1)) if m else None

HEADER = re.compile(r"^[0-9a-f]+-[0-9a-f]+\s+\S+\s+\S+\s+\S+\s+\S+\s*(.*)$")

def parse_smaps(t):
    # [(path, Size kB, Rss kB)] per mapping
    maps = []
    for line in t.splitlines():
        m = HEADER.match(line)
        if m:
            maps.append([m.group(1).strip(), 0, 0])
            continue
        k, sep, rest = line.partition(":")
        f = rest.split()
        if maps and sep and k in ("Size", "Rss") and f and f[0].isdigit():
            maps[-1][1 if k == "Size" else 2] = int(f[0])
    return [tuple(m) for m in maps]

def is_anon(path):
    return path == "" or path.startswith(("[heap", "[stack", "[anon"))

def smaps_figures(maps):
    gguf = [m for m in maps if m[0].replace(" (deleted)", "").endswith(".gguf")]
    return (sum(m[2] for m in gguf), sum(m[1] for m in gguf), sum(m[2] for m in maps if is_anon(m[0])))

def snapshots(prefix):
    # {index: time} of the files now on disk: the last idx line for an index wrote its content
    snaps = {}
    for line in (text(prefix + ".smaps.idx") or "").splitlines():
        f = line.split()
        if len(f) == 2:
            snaps[int(f[0])] = float(f[1])
    return snaps

def bench_ts(t):
    res = []
    try:
        for r in csv.DictReader((t or "").splitlines()):
            if r.get("avg_ts"):
                n_p, n_g = int(r.get("n_prompt") or 0), int(r.get("n_gen") or 0)
                res.append("%s %.2f t/s" % ("tg%d" % n_g if n_p == 0 else "pp%d" % n_p, float(r["avg_ts"])))
    except (ValueError, csv.Error):
        return []
    return res

def mib(v):
    return "-" if v is None else "%.1f" % (v / MIB)

info = text(os.path.join(out, "info.txt"))
models, steps = [], list(ALL_STEPS)
if info:
    print(info.rstrip())
    for line in info.splitlines():
        m = re.match(r"model: (.*) \((\d+) bytes\)$", line)
        if m:
            models.append((os.path.basename(m.group(1))[:-5] if m.group(1).endswith(".gguf")
                           else os.path.basename(m.group(1)), int(m.group(2))))
        m = re.match(r"steps: (.*?)(;|$)", line)
        if m:
            steps = [s for s in ALL_STEPS if s in m.group(1).split()]
names = {n for n, _ in models}
for p in glob.glob(os.path.join(out, "*.rc")) + glob.glob(os.path.join(out, "*.samples.csv")):
    base = os.path.basename(p).rsplit(".rc", 1)[0].rsplit(".samples.csv", 1)[0]
    for s in ALL_STEPS:
        if base.startswith(s + "-") and base[len(s) + 1:] not in names:
            names.add(base[len(s) + 1:])
            models.append((base[len(s) + 1:], None))
        if base.startswith(s + "-") and s not in steps:
            steps.append(s)
steps = [s for s in ALL_STEPS if s in steps]

BUF = re.compile(r"[A-Za-z0-9_:.]+\s+model buffer size\s*=\s*[0-9.]+ MiB")
res = {}
problems = []
for model, _ in models:
    for step in steps:
        prefix = os.path.join(out, "%s-%s" % (step, model))
        label = "%s %s" % (step, model)
        rc = (text(prefix + ".rc") or "").strip()
        st = text(prefix + ".samples.csv")
        if not rc and st is None:
            problems.append("absent: %s (no files)" % label)
            continue
        sysd = parse_sys(text(prefix + ".sys"))
        mode, rows = parse_samples(st or "")
        life = float(sysd["lifetime"]) if "lifetime" in sysd else (rows[-1]["t"] if rows else None)
        ts = bench_ts(text(prefix + ".out"))
        print("%s: rc %s, lifetime %s s, %d samples, mode %s%s" % (label, rc or "?", "%.1f" % life if life else "?",
              len(rows), mode, (", " + ", ".join(ts)) if ts else ""))
        if "timeout" in sysd:
            print("   TIMEOUT %s" % sysd["timeout"])
        if rc != "0":
            problems.append("BAD: %s rc %s" % (label, rc or "?"))
        steady = None
        if rows and life:
            before = [r for r in rows if r["t"] <= 0.9 * life]
            steady = before[-1] if before else None
        if steady is None:
            print("   no sample at or before 90%% of the lifetime")
        elif mode == "proc":
            print("   steady at %.1f s (MiB): RssAnon %s, RssFile %s, RssShmem %s, Pss_Anon %s, Pss_File %s, "
                  "AnonHugePages %s, VmSwap %s" % (steady["t"], mib(steady.get("RssAnon")), mib(steady.get("RssFile")),
                  mib(steady.get("RssShmem")), mib(steady.get("Pss_Anon")), mib(steady.get("Pss_File")),
                  mib(steady.get("AnonHugePages")), mib(steady.get("VmSwap"))))
            anon = [r["RssAnon"] for r in rows if r.get("RssAnon") is not None]
            hwm = [r["VmHWM"] for r in rows if r.get("VmHWM") is not None]
            print("   peak (MiB): RssAnon %s, last VmHWM %s" % (mib(max(anon) if anon else None),
                  mib(hwm[-1] if hwm else None)))
            during = [r["MemAvailable"] for r in rows if r.get("MemAvailable") is not None]
            print("   MemAvailable MiB (system-wide, shared board, indicative only): before %s, min during %s, after %s"
                  % (mib(sys_mem(sysd.get("before"))), mib(min(during) if during else None), mib(sys_mem(sysd.get("after")))))
            snaps = snapshots(prefix)
            at = [(t, k) for k, t in snaps.items() if t <= steady["t"]]
            if at:
                t, k = max(at)
                maps = parse_smaps(text("%s.smaps.%d" % (prefix, k)) or "")
                g_rss, g_size, a_rss = smaps_figures(maps)
                print("   smaps at %.1f s: .gguf mappings resident %s MiB of %s mapped; anonymous mappings Rss %s MiB"
                      % (t, mib(g_rss), mib(g_size), mib(a_rss)))
            else:
                print("   no smaps snapshot at or before the steady time (kept: %s)" %
                      (", ".join("%.1f s" % t for t in sorted(snaps.values())) or "none"))
        else:
            rss = [r["VmRSS"] for r in rows if r.get("VmRSS") is not None]
            print("   steady at %.1f s (MiB): RSS %s; peak RSS %s" % (steady["t"], mib(steady.get("VmRSS")),
                  mib(max(rss) if rss else None)))
        for b in BUF.findall(text(prefix + ".err") or ""):
            print("   buffer %s" % re.sub(r"\s+", " ", b))
        res[(step, model)] = (rc, mode, steady)

# comparison: only runs that ended with rc 0 and have a steady sample
ok = {k: v for k, v in res.items() if v[0] == "0" and v[2] is not None}
proc_only = ok and all(v[1] == "proc" for v in ok.values())

def figs(step, model):
    v = ok.get((step, model))
    if v is None:
        return None
    s = v[2]
    if not proc_only:
        return {"R": s.get("VmRSS")}
    a, f, sh = s.get("RssAnon"), s.get("RssFile"), s.get("RssShmem")
    tot = None if None in (a, f, sh) else a + f + sh
    return {"A": a, "F": f, "T": tot}

kinds = ("A", "F", "T") if proc_only else ("R",)
pairs = [(s, "cpu") for s in steps if s != "cpu" and "cpu" in steps]
if "prov-mmap" in steps and "prov-nommap" in steps:
    pairs.append(("prov-mmap", "prov-nommap"))
cols = ["%s.%s" % (TAG[s], k) for s in steps for k in kinds]
cols += ["%s-%s.%s" % (TAG[a], TAG[b], k) for a, b in pairs for k in kinds]
print()
if proc_only:
    print("comparison, steady MiB: A RssAnon, F RssFile, T A+F+RssShmem; x-y = run x minus run y")
else:
    print("comparison, steady MiB: R the RSS (ps mode, or mixed modes); x-y = run x minus run y")
print("runs: " + ", ".join("%s %s" % (TAG[s], s) for s in steps) + "; - = absent, failed or no steady sample")
print("%-18s %6s " % ("model", "file") + " ".join("%6s" % c for c in cols))
for model, size in models:
    cells = []
    for s in steps:
        f = figs(s, model)
        cells += ["%6s" % ("-" if f is None or f[k] is None else "%.0f" % (f[k] / MIB)) for k in kinds]
    for a, b in pairs:
        fa, fb = figs(a, model), figs(b, model)
        for k in kinds:
            if fa is None or fb is None or fa[k] is None or fb[k] is None:
                cells.append("%6s" % "-")
            else:
                cells.append("%+6.0f" % ((fa[k] - fb[k]) / MIB))
    print("%-18s %6s " % (model[:18], "-" if size is None else "%.0f" % (size / 1048576.0)) + " ".join(cells))
for p in problems:
    print(p)
print("logs: %s" % out)
PY
}

case "${1:-}" in
    run)
        OUT=${2:-$HOME/mem-probe}
        if [ -z "$DRY" ]; then
            for m in $MODELS; do
                [ -f "$m" ] || { echo "no model $m"; exit 1; }
            done
            [ -x "$BUILD/bin/llama-bench" ] || { echo "no $BUILD/bin/llama-bench: build it first"; exit 1; }
            mkdir -p "$OUT"
            mode=ps
            [ -r "$PROC_ROOT/meminfo" ] && mode=proc
            { echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
              echo "build: $BUILD at $(git -c safe.directory="$REPO" -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
              step_on ime && echo "build ime: $BUILD_IME; SPACEMIT_DISABLE_TCM=${SPACEMIT_DISABLE_TCM:-unset}"
              for m in $MODELS; do echo "model: $m ($(wc -c < "$m" | tr -d ' ') bytes)"; done
              echo "steps: $STEPS; n_gen: $NGEN; n_gen cpu: $NGEN_CPU; interval: $INTERVAL s"
              echo "kernel: $(uname -sr); THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo unreadable)"
              echo "sampler mode: $mode (proc root $PROC_ROOT)"; } > "$OUT/info.txt"
        fi
        BENCH="-fa 1 -p 0 -r 1 -o csv -v"
        for m in $MODELS; do
            name=${m##*/}
            name=${name%.gguf}
            if step_on prov-mmap; then
                run "$OUT/prov-mmap-$name" "$BUILD/bin/llama-bench" -m "$m" -t 8 -n "$NGEN" -mmp 1 $BENCH
            fi
            if step_on prov-nommap; then
                run "$OUT/prov-nommap-$name" "$BUILD/bin/llama-bench" -m "$m" -t 8 -n "$NGEN" -mmp 0 $BENCH
            fi
            if step_on cpu; then
                run "$OUT/cpu-$name" env FLAGOS_SPACEMIT_DISABLE=1 "$BUILD/bin/llama-bench" \
                    -m "$m" -t 1 -n "$NGEN_CPU" -mmp 1 $BENCH
            fi
            if step_on ime; then
                if [ -x "$BUILD_IME/bin/llama-bench" ]; then
                    run "$OUT/ime-$name" "$BUILD_IME/bin/llama-bench" -m "$m" -t 8 -n "$NGEN" -mmp 1 $BENCH
                else
                    echo "ime: no $BUILD_IME/bin/llama-bench, skipped"
                    [ -n "$DRY" ] || echo "ime: skipped, no $BUILD_IME/bin/llama-bench" >> "$OUT/info.txt"
                fi
            fi
        done
        [ -n "$DRY" ] || summary "$OUT"
        ;;
    summary)
        [ -n "${2:-}" ] || { echo "usage: $0 summary <out>"; exit 2; }
        summary "$2"
        ;;
    *)
        echo "usage: $0 run [out] | summary <out>"
        exit 2
        ;;
esac
