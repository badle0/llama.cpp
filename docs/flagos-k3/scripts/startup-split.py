#!/usr/bin/env python3
# Where llama-server's start-up goes, stage by stage (docs/flagos-k3/plan.md "The start-up split"; batch-one.md C1).
# No provider code: the stages are the log's own timestamps at -lv 4, placed on an outside clock by the moment the
# first log line appears; on Linux /proc gives each stage's CPU time and page faults. Run it with nothing else on the
# AI cores. It drops only the model file's clean pages from the page cache for the cold runs, no system setting.
#
# Usage, from anywhere: startup-split.py [--build-dir DIR] [--out DIR] [--runs N] [model.gguf ...]
# Per model, in order: the file read cold (the disk's rate); N warm starts; one with --load-mode none; one with the
# provider disabled (-t 1, by PROJECT.md section 6's rule); 2 cold starts. After each, one short request twice: the
# first shows what the first request adds. Each start's log stays in the --out directory.
import argparse
import bisect
import ctypes
import http.client
import json
import mmap
import os
import re
import socket
import statistics
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TIMEOUT_S = 600
PROMPT = "The capital of France is"

# each mark is the first line matching it after the previous mark's line
MARKS = [
    ("fit", r"common_init_: fitting params"),
    ("fitted", r"fitting params to free memory took"),
    ("tensors", r"load_tensors: loading model tensors"),
    ("sized", r"model buffer size"),
    ("loaded", r"logit bias|llama_context: constructing"),
    ("kv", r"llama_kv_cache: size ="),
    ("warm", r"warming up the model"),
    ("warmed", r"load_model: initializing"),
    ("ready", r"all slots are idle"),
]
# the table's rows: name, from mark, to mark ("start" the process's start, "log" its first log line, "up" /health ok)
STAGES = [
    ("process start to first log line", "start", "log"),
    ("to the parameter fit", "log", "fit"),
    ("parameter fit (a trial load)", "fit", "fitted"),
    ("metadata and vocabulary", "fitted", "tensors"),
    ("tensors, file mapped, buffers", "tensors", "sized"),
    ("weights copied and repacked", "sized", "loaded"),
    ("context and KV cache", "loaded", "kv"),
    ("graph reserve", "kv", "warm"),
    ("warm-up run", "warm", "warmed"),
    ("slots and template", "warmed", "ready"),
    ("ready to /health ok", "ready", "up"),
]


def proc_stat(pid):
    """(minor faults, major faults, user ticks, system ticks) of the process, all its threads, or None without /proc."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            s = f.read()
    except OSError:
        return None
    fields = s[s.rindex(")") + 2:].split()  # from field 3, the state (the name in field 2 may hold spaces)
    return int(fields[7]), int(fields[9]), int(fields[11]), int(fields[12])


def resident(path):
    """The fraction of the file in the page cache, by mincore, reading none of it; None if that fails."""
    size = os.path.getsize(path)
    pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    try:
        with open(path, "rb") as f:
            m = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_COPY)
        view = (ctypes.c_char * size).from_buffer(m)
        vec = (ctypes.c_ubyte * pages)()
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        rc = libc.mincore(ctypes.addressof(view), size, vec)
        del view
        m.close()
    except (OSError, ValueError, AttributeError, BufferError):
        return None
    return None if rc != 0 else sum(b & 1 for b in bytes(vec)) / pages


def evict(path):
    """Drops the file's clean pages from the page cache; False where the call does not exist (macOS)."""
    if not hasattr(os, "posix_fadvise"):
        return False
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return True


def read_file(path):
    """Seconds to read the whole file in order."""
    t0 = time.monotonic()
    with open(path, "rb", buffering=0) as f:
        while f.read(16 * 1024 * 1024):
            pass
    return time.monotonic() - t0


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def health_ok(port):
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
        c.request("GET", "/health")
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status == 200 and json.loads(body).get("status") == "ok"
    except (OSError, http.client.HTTPException, ValueError):
        return False


def complete(port):
    """One short request: (wall ms, prompt tokens, the server's prompt ms)."""
    body = json.dumps({"prompt": PROMPT, "n_predict": 1, "temperature": 0, "cache_prompt": False})
    t0 = time.monotonic()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=TIMEOUT_S)
    c.request("POST", "/completion", body, {"Content-Type": "application/json"})
    r = json.loads(c.getresponse().read())
    c.close()
    return 1000 * (time.monotonic() - t0), r["timings"]["prompt_n"], r["timings"]["prompt_ms"]


def log_marks(text):
    """{mark: seconds on the log's clock}, "log" the first line's; a mark not found is left out."""
    marks, i, lines = {}, 0, []
    for line in text.splitlines():
        m = re.match(r"(\d+)\.(\d\d)\.(\d{3})\.(\d{3}) ", line)
        if m:
            t = 60 * int(m[1]) + int(m[2]) + int(m[3]) / 1e3 + int(m[4]) / 1e6
            lines.append((t, line))
    if lines:
        marks["log"] = lines[0][0]
    for name, pattern in MARKS:
        for j in range(i, len(lines)):
            if re.search(pattern, lines[j][1]):
                marks[name] = lines[j][0]
                i = j + 1
                break
    return marks


def one_start(cmd, env, log_path):
    """One server start; {"stages": {name: seconds}, "cpu": {name: (user s, sys s, minor k, major)}, ...} or {"failed"}."""
    port = free_port()
    run = {}
    samples = []
    with open(log_path, "wb") as log:
        t0 = time.monotonic()
        srv = subprocess.Popen(cmd + ["--port", str(port)], stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT, env=env)
        try:
            first, up, next_health = None, None, 0.0
            while up is None:
                now = time.monotonic() - t0
                if srv.poll() is not None:
                    run["failed"] = f"exited with rc={srv.returncode} before /health ok"
                    break
                if now > TIMEOUT_S:
                    run["failed"] = f"no /health ok in {TIMEOUT_S} s"
                    break
                st = proc_stat(srv.pid)
                if st is not None:
                    samples.append((now,) + st)
                if first is None and log_path.stat().st_size > 0:
                    first = now
                if now >= next_health:
                    if health_ok(port):
                        up = time.monotonic() - t0
                    next_health = now + 0.1
                time.sleep(0.02)
            if up is not None:
                run["up"] = up
                run["requests"] = [complete(port), complete(port)]
        except (OSError, http.client.HTTPException, ValueError, KeyError) as e:
            run["failed"] = f"{type(e).__name__}: {e}"
        finally:
            if srv.poll() is None:
                srv.terminate()
                try:
                    srv.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    srv.kill()
                    srv.wait()
    text = log_path.read_text(errors="replace")
    run["sizes"] = re.findall(r"load_tensors:\s+(\S.*?) model buffer size =\s+([\d.]+) MiB", text)
    if "failed" in run:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        run["failed"] += "; last log line: " + (lines[-1][:120] if lines else "(none)")
        return run
    marks = log_marks(text)
    offset = first - marks["log"] if first is not None and "log" in marks else None
    points = {"start": 0.0, "up": up}
    if offset is not None:
        points.update({k: v + offset for k, v in marks.items()})
    run["stages"], run["cpu"] = {}, {}
    tick = os.sysconf("SC_CLK_TCK")
    times = [s[0] for s in samples]

    def at(t):
        return samples[min(bisect.bisect_left(times, t), len(samples) - 1)]

    for name, a, b in STAGES:
        if a in points and b in points:
            run["stages"][name] = points[b] - points[a]
            if samples:
                s, e = at(points[a]), at(points[b])
                run["cpu"][name] = ((e[3] - s[3]) / tick, (e[4] - s[4]) / tick, (e[1] - s[1]) / 1e3, e[2] - s[2])
    return run


def median(runs, get):
    values = [v for v in (get(r) for r in runs) if v is not None]
    return statistics.median(values) if values else None


def cell(v, fmt="{:.2f}"):
    if v is None:
        return "-"
    return fmt.format(0.0 if abs(v) < 0.005 else v)  # two clocks placed by polling: no "-0.00" from their small gap


def main():
    ap = argparse.ArgumentParser(description="llama-server's start-up, stage by stage")
    ap.add_argument("--build-dir", default=str(REPO / "build-flagos"))
    ap.add_argument("--out", default=os.path.expanduser("~/kvh/startup"))
    ap.add_argument("--runs", type=int, default=3, help="warm starts per model (default 3)")
    ap.add_argument("models", nargs="*", default=[os.path.expanduser(f"~/models/Qwen3-{m}-Q4_0.gguf") for m in ("4B", "0.6B")])
    args = ap.parse_args()
    server = Path(args.build_dir).resolve() / "bin" / "llama-server"
    if not server.is_file():
        raise SystemExit(f"no {server}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    release = Path(os.path.expanduser("~/tcmtest/tcmrelease"))
    n_failed = 0

    for model in args.models:
        stem = Path(model).stem
        base = [str(server), "-m", model, "--host", "127.0.0.1", "-c", "4096", "-lv", "4", "--cache-reuse", "16"]  # the runner's
        env_cpu = {**os.environ, "FLAGOS_SPACEMIT_DISABLE": "1"}
        conditions = [(f"warm{i + 1}", "warm", base + ["-t", "8"], os.environ, False) for i in range(args.runs)]
        conditions += [("none", "none", base + ["-t", "8", "--load-mode", "none"], os.environ, False)]
        conditions += [("cpu", "cpu", base + ["-t", "1"], env_cpu, False)]
        conditions += [(f"cold{i + 1}", "cold", base + ["-t", "8"], os.environ, True) for i in range(2)]

        size_mib = os.path.getsize(model) / 2**20
        print(f"== {stem}: file {size_mib:.0f} MiB", flush=True)
        if evict(model):
            print(f"file read cold: in the page cache {cell(resident(model), '{:.1%}')} after eviction", end="", flush=True)
            s = read_file(model)
            print(f", {s:.2f} s, {size_mib / s:.0f} MiB/s", flush=True)
        else:
            print("no posix_fadvise here: the file read and the cold starts are not cold", flush=True)
            read_file(model)

        by_kind = {}
        for name, kind, cmd, env, cold in conditions:
            if release.is_file():
                subprocess.run([str(release), "--apply"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            note = ""
            if cold and evict(model):
                note = f" (in the page cache at start {cell(resident(model), '{:.1%}')})"
            run = one_start(cmd, env, out / f"{stem}-{name}.log")
            by_kind.setdefault(kind, []).append(run)
            if "failed" in run:
                n_failed += 1
                print(f"{name}: FAILED: {run['failed']}", flush=True)
            else:
                a, b = run["requests"]
                print(f"{name}: /health ok at {run['up']:.2f} s; first request {a[0]:.0f} ms, second {b[0]:.0f} ms "
                      f"({a[1]} prompt tokens){note}", flush=True)

        good = {k: [r for r in v if "failed" not in r] for k, v in by_kind.items()}
        print(f"\nmedians, seconds ({len(good['warm'])} warm, {len(good['none'])} none, {len(good['cpu'])} cpu, "
              f"{len(good['cold'])} cold); user, sys and faults are the warm runs'")
        print(f"{'stage':34} {'warm':>7} {'user':>7} {'sys':>7} {'minflt k':>9} {'majflt':>7} {'none':>7} {'cpu':>7} {'cold':>7}")
        rows = [n for n, _, _ in STAGES]
        for n in rows:
            cols = [median(good[k], lambda r, n=n: r["stages"].get(n)) for k in ("warm", "none", "cpu", "cold")]
            cpu = [median(good["warm"], lambda r, n=n, i=i: r["cpu"][n][i] if n in r["cpu"] else None) for i in range(4)]
            print(f"{n:34} {cell(cols[0]):>7} {cell(cpu[0]):>7} {cell(cpu[1]):>7} {cell(cpu[2], '{:.0f}'):>9} "
                  f"{cell(cpu[3], '{:.0f}'):>7} {cell(cols[1]):>7} {cell(cols[2]):>7} {cell(cols[3]):>7}")
        ups = [median(good[k], lambda r: r["up"]) for k in ("warm", "none", "cpu", "cold")]
        print(f"{'/health ok':34} {cell(ups[0]):>7} {'':>7} {'':>7} {'':>9} {'':>7} {cell(ups[1]):>7} {cell(ups[2]):>7} {cell(ups[3]):>7}")
        for i, label in enumerate(("first request, ms", "second request, ms")):
            reqs = [median(good[k], lambda r, i=i: r["requests"][i][0]) for k in ("warm", "none", "cpu", "cold")]
            print(f"{label:34} {cell(reqs[0], '{:.0f}'):>7} {'':>7} {'':>7} {'':>9} {'':>7} {cell(reqs[1], '{:.0f}'):>7} "
                  f"{cell(reqs[2], '{:.0f}'):>7} {cell(reqs[3], '{:.0f}'):>7}")
        if good["warm"]:
            sizes = good["warm"][0]["sizes"]
            print("model buffers: " + ", ".join(f"{n} {s} MiB" for n, s in sizes))
            ups_w = [r["up"] for r in good["warm"]]
            print(f"warm /health ok, lowest and highest: {min(ups_w):.2f} and {max(ups_w):.2f} s")
        print(flush=True)
    if n_failed:
        raise SystemExit(f"{n_failed} start(s) failed")


if __name__ == "__main__":
    main()
