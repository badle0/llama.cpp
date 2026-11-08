#!/usr/bin/env python3
# TTFT baseline for the SpacemiT provider (plan.md, "TTFT harness spec"): the figures T1 and the later levers are held
# against. Run on the K3; standard library only; it imports the acceptance runner's helpers (spacemit_check.py).
# Placement first, from a short -lv 4 start: every layer and the KV cache on the provider and 2 graph splits, else the
# run stops (--allow-cpu: reported only). The IME build's llama-bench before and after the cases as the control. The
# cases, always in this order:
#   grid: time to the first token by cached depth and new tail (C3, C5); chat: 20 turns through /v1/chat/completions,
#   thinking off then on, each on a fresh server with the defaults (C6); cancel: the connection closed during a long
#   prefill, then at once a 1-token request (C10); tg512: 512 tokens with ignore_eos from depths 0, 500 and 4096 (C11);
#   restart: kill -9 during a prefill, then a new server (C14); second: a second server beside the first (C15);
#   alongside: the IME build's llama-bench looping beside the server, reported without a verdict (C15).
# Checks: placement, requests answering, cache_n and prompt_n as asked, restart's next server, second's B after A has
# gone. No limit on any speed. Rows go to ttft.jsonl, every line and the tables to summary.txt, in
# <build-dir>/logs-ttft-<model>/. --quick cuts every case down to minutes.
# Usage: spacemit_ttft.py [--build-dir DIR] [--model GGUF] [--cases LIST] [--repeats N] [--quick] [--ime-build DIR]
#                         [--tcm-dir DIR] [--segv SO] [--allow-cpu]
import argparse
import csv
import glob
import http.client
import io
import json
import math
import os
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import spacemit_check as sc

CASES = ["grid", "chat", "cancel", "tg512", "restart", "second", "alongside"]
N_CTX = 8704
MEASURE_FLAGS = ["-t", "8", "-np", "1", "-c", str(N_CTX), "--cache-ram", "0"]
CHAT_FLAGS = ["-t", "8", "-c", str(N_CTX)]
BASE_MIN = 16384
GRID_DEPTH0 = [1, 2, 3, 4, 5, 8, 9, 16, 32, 63, 64, 65, 128, 256, 500, 511, 512, 513, 575, 576, 1024, 2048, 4095]
GRID_DEPTH0_QUICK = [1, 4, 9, 64, 65]
GRID_DEPTHS = [256, 1024, 2048, 4096, 8192]
GRID_DEPTHS_QUICK = [256]
GRID_TAILS = [1, 2, 3, 4, 8, 12, 26, 38, 63, 64, 110]
GRID_TAILS_QUICK = [1, 4, 12, 63, 64]
CHAT_TURNS = [
    "I want to start a small vegetable garden on a sunny balcony. What should I grow first?",
    "Which of those grows fastest?",
    "How often should I water it?",
    "What size of pot does it need?",
    "Can I grow it next to tomatoes?",
    "What soil mix do you recommend?",
    "How do I know when it is ready to harvest?",
    "What pests should I watch for?",
    "How do I deal with aphids without chemicals?",
    "Should I use fertilizer, and how much?",
    "What can I plant after harvesting it?",
    "How much sun do tomatoes need each day?",
    "Give me a simple weekly care schedule for the balcony.",
    "What should I do during a heat wave?",
    "How can I save seeds for next year?",
    "Which herbs grow well in the same conditions?",
    "Can I grow anything over the winter?",
    "How do I compost kitchen scraps in a small flat?",
    "Summarize the plan in five bullet points.",
    "What is the most common mistake beginners make?",
]
rows = []  # every row written to ttft.jsonl, for the summary
live = []  # child processes still running, stopped on Ctrl-C
ime_loops = []  # alongside's IME loops, stopped on Ctrl-C
jsonl = None


def row(case, **fields):
    r = {"case": case, **fields}
    rows.append(r)
    if jsonl is not None:
        jsonl.write(json.dumps(r) + "\n")
        jsonl.flush()
    return r


def fmt(v, digits=1):
    return "-" if v is None else f"{v:.{digits}f}"


def pct(values, q):
    """Nearest rank: the value at index ceil(q * n) - 1 of the sorted values, or None if there are none."""
    v = sorted(x for x in values if x is not None)
    return v[max(0, math.ceil(q * len(v)) - 1)] if v else None


def read_text(path):
    try:
        return sc.ANSI.sub("", Path(path).read_text(errors="replace"))
    except OSError:
        return ""


def could_not_get(proc):
    return sum("could not get" in ln for ln in read_text(proc.log_path).splitlines())


# servers

def start_server(ctx, tag, flags):
    """llama-server with the flags, registered in live, its output in <logs>/server-<tag>.log. Returns (proc, url,
    seconds to /health ok or None, why not)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env.update(sc.preload)
    log_path = ctx.logs / f"server-{tag}.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen([str(ctx.server), "-m", ctx.model, "--host", "127.0.0.1", "--port", str(port), *flags],
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
    proc.log_path = log_path
    live.append(proc)
    secs, why = sc.server_wait(proc, url)
    if secs is None:
        why = f"{why}; last log line: {sc.last_line(read_text(log_path))[:100]}"
    return proc, url, secs, why


def stop_server(proc):
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if proc in live:
        live.remove(proc)


def stop_all():
    for loop in ime_loops:
        loop.stop_event.set()
    for proc in list(live):
        stop_server(proc)
    for loop in ime_loops:
        loop.join(5)
    for proc in list(live):
        stop_server(proc)


# HTTP, one connection per request

def connect(url, timeout):
    host, port = url.split("//", 1)[1].rsplit(":", 1)
    return http.client.HTTPConnection(host, int(port), timeout=timeout)


def error_text(data, text):
    if isinstance(data, dict) and "error" in data:
        e = data["error"]
        return str(e.get("message", e)) if isinstance(e, dict) else str(e)
    return " ".join(text.split())[:200] or "empty reply"


def post(url, path, body, timeout):
    """A non-streamed request: (status, parsed JSON or None, error text or None)."""
    conn = connect(url, timeout)
    try:
        conn.request("POST", path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = conn.getresponse()
        text = r.read().decode(errors="replace")
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if r.status != 200:
            return r.status, data, f"HTTP {r.status}: {error_text(data, text)}"
        return r.status, data, None if data is not None else "the reply is not JSON"
    except sc.HTTP_ERRORS as e:
        return None, None, f"{type(e).__name__}: {e}"
    finally:
        conn.close()


def stream(url, path, body, timeout):
    """A streamed request: (status, client ms to the first data line that is not an error, [(ms, event)], the last
    timings or {}, error text or None). The clock starts just before the request is sent."""
    conn = connect(url, timeout)
    status, first, events, timings, err = None, None, [], {}, None
    try:
        payload = json.dumps(body).encode()
        t0 = time.perf_counter()
        conn.request("POST", path, payload, {"Content-Type": "application/json"})
        r = conn.getresponse()
        status = r.status
        if status != 200:
            # an error before the slot starts (a bad request) comes as a plain JSON reply. The server sends the 200
            # status as the slot starts (server-context.cpp, "signal HTTP to send the headers"), so a failed decode
            # ("Compute error.") comes later, as an error event in the stream
            text = r.read().decode(errors="replace")
            try:
                data = json.loads(text)
            except ValueError:
                data = None
            err = f"HTTP {status}: {error_text(data, text)}"
        else:
            for raw in r:
                line = raw.decode(errors="replace").rstrip("\r\n")
                if not line or line.startswith(":"):
                    continue
                if line == "data: [DONE]":
                    break
                if not line.startswith("data: "):
                    continue
                ms = (time.perf_counter() - t0) * 1000.0
                ev = json.loads(line[len("data: "):])
                if isinstance(ev, dict) and "error" in ev:
                    err = f"error in the stream: {error_text(ev, line)}"
                    break
                events.append((ms, ev))
                if first is None:
                    first = ms
                if isinstance(ev, dict) and isinstance(ev.get("timings"), dict):
                    timings = ev["timings"]
            if err is None and first is None:
                err = "no data line in the reply"
    except sc.HTTP_ERRORS as e:
        err = f"{type(e).__name__}: {e}"
    finally:
        conn.close()
    return status, first, events, timings, err


def stream_text(events):
    return "".join(ev.get("content") or "" for _, ev in events if isinstance(ev, dict))


# prompts as tokens

class Tails:
    """Tails of new tokens after a cached depth: BASE[o:o + n], o from 1000 (even requests) or 2000 (odd ones), bumped
    until BASE[o] differs from BASE[depth] (the next fill diverges at depth) and from the first token of the previous
    request at this depth (the cache diverges at depth)."""

    def __init__(self, base):
        self.base, self.count, self.first = base, 0, {}

    def __call__(self, depth, n):
        o = 1000 if self.count % 2 == 0 else 2000
        while self.base[o] in (self.base[depth], self.first.get(depth)):
            o += 1
        self.count += 1
        self.first[depth] = self.base[o]
        return self.base[o:o + n]


def n_kv(depth, tail):
    return min(N_CTX, math.ceil((depth + tail) / 256) * 256)


def completion(prompt, n_predict, **extra):
    """A streamed /completion body for the grid, cancel and tg512."""
    return {"prompt": prompt, "n_predict": n_predict, "cache_prompt": True, "temperature": 0, "stream": True, **extra}


def measured(timings, want_cache_n, want_prompt_n):
    cache_n, prompt_n = timings.get("cache_n"), timings.get("prompt_n")
    return {"prompt_ms": timings.get("prompt_ms"), "cache_n": cache_n, "prompt_n": prompt_n,
            "want_cache_n": want_cache_n, "want_prompt_n": want_prompt_n,
            "as_asked": cache_n == want_cache_n and prompt_n == want_prompt_n}


def no_timings(err, timings):
    return err if err is not None or timings else "no timings in the reply"


def fill(ctx, url, case, d_old, d_new):
    """Extend the cached prompt from d_old to d_new tokens, n_predict 0, not streamed; returns its row."""
    status, data, err = post(url, "/completion", {"prompt": ctx.base[:d_new], "n_predict": 0, "cache_prompt": True,
                                                  "temperature": 0}, 1800)
    t = (data.get("timings") if isinstance(data, dict) else None) or {}
    return row(case, kind="fill", depth=d_new, from_depth=d_old, status=status, error=no_timings(err, t),
               **measured(t, d_old, d_new - d_old))


def describe(r):
    parts = [r.get("kind", "row")]
    for key in ("depth", "tail", "repeat", "run"):
        if r.get(key) is not None:
            parts.append(f"{key} {r[key]}")
    return " ".join(parts)


def report_asked(case, mine):
    want = [r for r in mine if "as_asked" in r]
    bad = [r for r in want if not r["as_asked"]]
    detail = "; ".join(f"{describe(r)}: cache_n {r['cache_n']} (asked {r['want_cache_n']}), prompt_n {r['prompt_n']} "
                       f"(asked {r['want_prompt_n']})" for r in bad[:3])
    sc.report(f"{case}: cache_n and prompt_n as asked ({len(want) - len(bad)} of {len(want)} rows)",
              bool(want) and not bad,
              detail + (f"; {len(bad) - 3} more" if len(bad) > 3 else "") if bad else
              ("every row as asked" if want else "no rows"))


# placement, the control, the IME build

def placement(ctx):
    """Start a measuring server at -lv 4, tokenize README.md into BASE on it, read placement from its log. Returns
    whether the run goes on."""
    proc, url, secs, why = start_server(ctx, "placement", MEASURE_FLAGS + ["-lv", "4"])
    tok_err = None
    try:
        if secs is not None:
            status, data, tok_err = post(url, "/tokenize", {"content": (sc.REPO / "README.md").read_text(),
                                                            "add_special": False}, 300)
            tokens = data.get("tokens") if isinstance(data, dict) else None
            if tok_err is None and not tokens:
                tok_err = "no tokens in the reply"
            if tok_err is None:
                base = []
                while len(base) < BASE_MIN:
                    base += tokens
                ctx.base = base
    finally:
        stop_server(proc)
    text = read_text(proc.log_path)
    off = re.findall(r"offloaded (\d+)/(\d+) layers", text)
    seq = text.rfind("llama_context: n_seq_max")
    last_ctx = text[seq:] if seq >= 0 else text
    kv = [(m.group(1), float(m.group(2))) for m in re.finditer(r"(\S+) KV buffer size\s*=\s*([0-9.]+) MiB", last_ctx)]
    found = re.findall(r"graph splits = (\d+)(?: \(with bs=\d+\), (\d+) \(with bs=1\))?", text)
    splits = [int(n) for n in found[-1] if n] if found else []
    n_ctx = re.findall(r"llama_context: n_ctx\s*=\s*(\d+)", text)
    ok = (bool(off) and off[-1][0] == off[-1][1] and any(n.startswith("FlagOS:SpacemiT") and v > 0 for n, v in kv) and
          bool(splits) and all(n == 2 for n in splits))
    detail = ((f"offloaded {off[-1][0]}/{off[-1][1]} layers" if off else "no offload line") +
              f"; KV buffers {', '.join(f'{n} {v:.2f} MiB' for n, v in kv) or 'none'}; graph splits "
              f"{'/'.join(map(str, splits)) or '?'}; n_ctx {n_ctx[-1] if n_ctx else '?'}")
    row("placement", up_s=secs, offloaded=list(off[-1]) if off else None, kv_buffers=kv, graph_splits=splits,
        n_ctx=int(n_ctx[-1]) if n_ctx else None, ok=ok, tokenize_error=tok_err,
        base_tokens=len(ctx.base) if ctx.base else None)
    name = "placement: every layer and the KV cache on the provider, 2 graph splits"
    if secs is None:
        sc.report(name, False, f"the placement server did not come up: {why}")
        return False
    if tok_err is not None:
        sc.report(name, False, f"{detail}; /tokenize failed: {tok_err}")
        return False
    if ok:
        sc.report(name, True, f"up after {secs:.1f} s; {detail}")
    elif ctx.allow_cpu:
        sc.info(name, f"up after {secs:.1f} s; {detail} (--allow-cpu)")
    else:
        sc.report(name, False, f"up after {secs:.1f} s; {detail}")
        return False
    sc.info("prompts", f"README.md is {len(tokens)} tokens, repeated to {len(ctx.base)}")
    return True


def ime_args(ctx):
    return ["-m", ctx.model, "-t", "8", "-p", "128", "-n", "128", "-ub", "128", "-fa", "1", "-mmp", "0", "-r", "3",
            "-o", "csv"]


def bench_speeds(out):
    """pp128 and tg128 from llama-bench's CSV, as the runner reads them."""
    speeds = {}
    for r in csv.DictReader(io.StringIO(out)):
        try:
            if r.get("n_prompt") == "128" and r.get("n_gen") == "0":
                speeds["pp128"] = float(r["avg_ts"])
            elif r.get("n_prompt") == "0" and r.get("n_gen") == "128":
                speeds["tg128"] = float(r["avg_ts"])
        except (KeyError, TypeError, ValueError):
            pass
    return speeds


def speeds_text(speeds):
    return ", ".join(f"{k} {v:.2f} t/s" for k, v in speeds.items()) or "no result"


def control(ctx, when):
    """The IME build's llama-bench, batch-one.md section 6's control."""
    if not ctx.ime_bench.is_file():
        sc.info(f"control ({when})", f"skipped: no {ctx.ime_bench}")
        return
    release = sc.tcm_tool(ctx.tcm_dir, "tcmrelease")
    if release:
        sc.run([release, "--apply"], ctx.logs / f"control-{when}-tcmrelease", timeout=120)
    t0 = time.time()
    rc, out, err = sc.run([str(ctx.ime_bench), *ime_args(ctx)], ctx.logs / f"control-{when}", env_extra=sc.preload,
                          timeout=1800)
    speeds = bench_speeds(out)
    sc.info(f"control ({when})", f"IME llama-bench {speeds_text(speeds)} (rc={rc}, {time.time() - t0:.0f} s)" +
            ("" if rc == 0 else f": {sc.last_line(err)[:100]}"))
    row("control", when=when, rc=rc, **speeds)


class ImeLoop(threading.Thread):
    """alongside's loop: until stop_event, tcmrelease --apply (if present), then the IME llama-bench with the
    control's arguments."""

    def __init__(self, ctx):
        super().__init__(daemon=True)
        self.ctx, self.stop_event, self.proc, self.first_start, self.iterations, self.error = (
            ctx, threading.Event(), None, None, [], None)

    def run(self):
        ctx, release = self.ctx, sc.tcm_tool(self.ctx.tcm_dir, "tcmrelease")
        env = dict(os.environ)
        env.update(sc.preload)
        try:
            i = 0
            while not self.stop_event.is_set():
                i += 1
                if release:
                    sc.run([release, "--apply"], ctx.logs / f"alongside-tcmrelease-{i}", timeout=120)
                if self.stop_event.is_set():
                    break
                out_path = ctx.logs / f"alongside-ime-{i}.out"
                with open(out_path, "w") as out, open(ctx.logs / f"alongside-ime-{i}.err", "w") as err:
                    proc = subprocess.Popen([str(ctx.ime_bench), *ime_args(ctx)], stdin=subprocess.DEVNULL, stdout=out,
                                            stderr=err, env=env)
                live.append(proc)
                self.proc = proc
                t0 = time.time()
                if self.first_start is None:
                    self.first_start = t0
                rc = proc.wait()
                if proc in live:
                    live.remove(proc)
                self.iterations.append({"iteration": i, "rc": rc, "seconds": round(time.time() - t0, 1),
                                        **bench_speeds(read_text(out_path))})
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def finish(self, timeout=600):
        """Stop after the current IME process ends on its own, killing it after timeout."""
        self.stop_event.set()
        self.join(timeout)
        if self.is_alive() and self.proc is not None:
            stop_server(self.proc)
            self.join(30)


def thermal():
    """{zone type or name: degrees C} from /sys/class/thermal, or None where there are none."""
    temps = {}
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            c = int(Path(zone, "temp").read_text().strip()) / 1000.0
        except (OSError, ValueError):
            continue
        try:
            name = Path(zone, "type").read_text().strip() or Path(zone).name
        except OSError:
            name = Path(zone).name
        temps[name if name not in temps else f"{name} ({Path(zone).name})"] = c
    return temps or None


# the cases

def case_grid(ctx):
    depths, tails_at = (GRID_DEPTHS_QUICK, GRID_TAILS_QUICK) if ctx.quick else (GRID_DEPTHS, GRID_TAILS)
    plan = [(0, GRID_DEPTH0_QUICK if ctx.quick else GRID_DEPTH0)] + [(d, tails_at) for d in depths]
    total = sum(len(t) * ctx.repeats for _, t in plan) + len(plan) - 1
    mine, failed = [], None
    proc, url, secs, why = start_server(ctx, "grid", MEASURE_FLAGS)
    try:
        if secs is None:
            failed = f"the server did not come up: {why}"
        else:
            tails, prev = Tails(ctx.base), 0
            for depth, tail_list in plan:
                if depth != prev:
                    r = fill(ctx, url, "grid", prev, depth)
                    mine.append(r)
                    if r["error"]:
                        failed = f"the fill to depth {depth}: {r['error']}"
                        break
                    prev = depth
                for rep in range(1, ctx.repeats + 1):
                    for n in tail_list:
                        body = completion(ctx.base[:depth] + tails(depth, n), 1)
                        status, first, _, t, err = stream(url, "/completion", body, 900)
                        r = row("grid", kind="tail", depth=depth, tail=n, repeat=rep, client_ms=first,
                                n_kv=n_kv(depth, n), status=status, error=no_timings(err, t), **measured(t, depth, n))
                        mine.append(r)
                        if r["error"]:
                            failed = f"depth {depth} tail {n} repeat {rep}: {r['error']}"
                            break
                    if failed:
                        break
                if failed:
                    break
                at = [r for r in mine if r.get("kind") == "tail" and r["depth"] == depth]
                sc.info(f"grid depth {depth}", "client ms p50 by tail: " + ", ".join(
                    f"{n} {fmt(pct([r['client_ms'] for r in at if r['tail'] == n], 0.5))}" for n in tail_list))
    finally:
        stop_server(proc)
    answered = sum(1 for r in mine if not r["error"])
    sc.report(f"grid: every request answered ({total})", failed is None,
              f"{answered} of {total} answered" + (f"; stopped at {failed}" if failed else ""))
    report_asked("grid", mine)


def case_chat(ctx):
    turns = CHAT_TURNS[:3] if ctx.quick else CHAT_TURNS
    max_tokens = 128 if ctx.quick else 512
    for mode, thinking in (("off", False), ("on", True)):
        failed, answered, proc = None, 0, None
        try:
            proc, url, secs, why = start_server(ctx, f"chat-{mode}", CHAT_FLAGS)
            if secs is None:
                failed = f"the server did not come up: {why}"
            else:
                messages = []
                for i, turn in enumerate(turns, 1):
                    messages.append({"role": "user", "content": turn})
                    body = {"messages": messages, "stream": True, "temperature": 0, "max_tokens": max_tokens,
                            "chat_template_kwargs": {"enable_thinking": thinking}}
                    status, first, events, t, err = stream(url, "/v1/chat/completions", body, 900)
                    reasoning_ms = content_ms = finish = None
                    content = []
                    for ms, ev in events:
                        for choice in (ev.get("choices") or []) if isinstance(ev, dict) else []:
                            delta = choice.get("delta") or {}
                            if delta.get("reasoning_content") and reasoning_ms is None:
                                reasoning_ms = ms
                            if delta.get("content"):
                                content.append(delta["content"])
                                if content_ms is None:
                                    content_ms = ms
                            if choice.get("finish_reason"):
                                finish = choice["finish_reason"]
                    text = "".join(content)
                    err = no_timings(err, t)
                    row("chat", mode=mode, turn=i, status=status, first_event_ms=first, first_reasoning_ms=reasoning_ms,
                        first_content_ms=content_ms, cache_n=t.get("cache_n"), prompt_n=t.get("prompt_n"),
                        prompt_ms=t.get("prompt_ms"), predicted_n=t.get("predicted_n"),
                        predicted_per_second=t.get("predicted_per_second"), finish_reason=finish,
                        hit_max_tokens=finish == "length", content_chars=len(text), error=err)
                    if err:
                        failed = f"turn {i}: {err}"
                        break
                    answered += 1
                    messages.append({"role": "assistant", "content": text})
        finally:
            stop_server(proc)
        sc.report(f"chat, thinking {mode}: every turn answered ({len(turns)})", failed is None,
                  f"{answered} of {len(turns)} answered" + (f"; stopped at {failed}" if failed else ""))


def cancel_request(url, body, wait_s):
    """Send body, wait until wait_s after the send or the first data line, whichever comes first, then close the
    socket. Headers come at once, so the raw socket is read for "data: ". Returns (answered first, ms at the close,
    status line or None, error or None)."""
    conn = connect(url, 30)
    finished, status, err, buf = False, None, None, b""
    t0 = time.perf_counter()
    try:
        conn.request("POST", "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
        while True:
            left = wait_s - (time.perf_counter() - t0)
            if left <= 0:
                break
            ready, _, _ = select.select([conn.sock], [], [], left)
            if not ready:
                break
            chunk = conn.sock.recv(65536)
            if not chunk:
                break
            buf += chunk
            if b"data: " in buf:
                finished = True
                break
    except sc.HTTP_ERRORS as e:
        err = f"{type(e).__name__}: {e}"
    finally:
        closed_ms = (time.perf_counter() - t0) * 1000.0
        conn.close()
    if buf.startswith(b"HTTP/"):
        status = buf.split(b"\r\n", 1)[0].decode(errors="replace")
    return finished, closed_ms, status, err


def case_cancel(ctx):
    depth = 256 if ctx.quick else 4096
    tail_n = 512 if ctx.quick else 2048
    repeats = 2 if ctx.quick else 3
    mine, failed, answered, waits = [], None, 0, []
    proc, url, secs, why = start_server(ctx, "cancel", MEASURE_FLAGS)
    try:
        if secs is None:
            failed = f"the server did not come up: {why}"
        else:
            r = fill(ctx, url, "cancel", 0, depth)
            mine.append(r)
            if r["error"]:
                failed = f"the fill to depth {depth}: {r['error']}"
            tails = Tails(ctx.base)
            for rep in range(1, repeats + 1):
                if failed:
                    break
                done, closed_ms, status, err = cancel_request(
                    url, completion(ctx.base[:depth] + tails(depth, tail_n), 1), 2.0)
                row("cancel", kind="cancelled", depth=depth, tail=tail_n, repeat=rep, finished_before_close=done,
                    closed_ms=closed_ms, status=status, error=err)
                status, first, _, t, err = stream(url, "/completion", completion(ctx.base[:depth] + tails(depth, 1), 1),
                                                  900)
                m = measured(t, depth, 1)
                wait = first - m["prompt_ms"] if None not in (first, m["prompt_ms"]) else None
                r = row("cancel", kind="followup", depth=depth, tail=1, repeat=rep, client_ms=first, wait_ms=wait,
                        status=status, error=no_timings(err, t), **m)
                mine.append(r)
                if r["error"]:
                    failed = f"follow-up {rep}: {r['error']}"
                    break
                answered += 1
                waits.append(wait)
    finally:
        stop_server(proc)
    sc.report(f"cancel: every follow-up answered ({repeats})", failed is None,
              f"{answered} of {repeats} answered" + (f"; stopped at {failed}" if failed else ""))
    report_asked("cancel", mine)
    sc.info("cancel", f"waits (client ms less prompt_ms) {', '.join(fmt(w) for w in waits) or 'none'} ms")


def case_tg512(ctx):
    depths = [0, 256] if ctx.quick else [0, 500, 4096]
    runs = 2 if ctx.quick else 10
    n_predict = 64 if ctx.quick else 512
    if thermal() is None:
        sc.info("tg512", "no /sys/class/thermal zones here: thermal_before and thermal_after are None")
    mine, failed, answered, total = [], None, 0, len(depths) * runs
    proc, url, secs, why = start_server(ctx, "tg512", MEASURE_FLAGS)
    try:
        if secs is None:
            failed = f"the server did not come up: {why}"
        else:
            tails, prev = Tails(ctx.base), 0
            for depth in depths:
                if depth != prev:
                    r = fill(ctx, url, "tg512", prev, depth)
                    mine.append(r)
                    if r["error"]:
                        failed = f"the fill to depth {depth}: {r['error']}"
                        break
                    prev = depth
                for run in range(1, runs + 1):
                    before = thermal()
                    body = completion(ctx.base[:depth] + tails(depth, 1), n_predict, ignore_eos=True)
                    status, first, events, t, err = stream(url, "/completion", body, 900)
                    after = thermal()
                    ts = [ms for ms, ev in events if isinstance(ev, dict) and (ev.get("content") or ev.get("tokens"))]
                    client = (len(ts) - 1) / ((ts[-1] - ts[0]) / 1000.0) if len(ts) > 1 and ts[-1] > ts[0] else None
                    per100 = [100.0 / ((ts[i + 100] - ts[i]) / 1000.0) for i in range(0, len(ts) - 100, 100)
                              if ts[i + 100] > ts[i]]
                    err = no_timings(err, t)
                    if err is None and t.get("predicted_n") != n_predict:
                        err = f"predicted_n {t.get('predicted_n')}, asked {n_predict}"
                    r = row("tg512", kind="run", depth=depth, run=run, n_predict=n_predict, client_ms=first,
                            predicted_n=t.get("predicted_n"), predicted_per_second=t.get("predicted_per_second"),
                            client_per_second=client, per_100=per100, thermal_before=before, thermal_after=after,
                            status=status, error=err, **measured(t, depth, 1))
                    mine.append(r)
                    if err:
                        failed = f"depth {depth} run {run}: {err}"
                        break
                    answered += 1
                if failed:
                    break
                runs_d = [r for r in mine if r.get("kind") == "run" and r["depth"] == depth]
                speeds = [r["predicted_per_second"] for r in runs_d]
                sc.info(f"tg512 depth {depth}",
                        f"server t/s p50 {fmt(pct(speeds, 0.5), 2)}, min {fmt(min(speeds), 2)}, max "
                        f"{fmt(max(speeds), 2)}; per 100 tokens, first run {per100_text(runs_d[0])}, last run "
                        f"{per100_text(runs_d[-1])}")
    finally:
        stop_server(proc)
    sc.report(f"tg512: every run answered with predicted_n == {n_predict} ({total})", failed is None,
              f"{answered} of {total} answered" + (f"; stopped at {failed}" if failed else ""))
    report_asked("tg512", mine)


def per100_text(r):
    return "[" + ", ".join(f"{v:.2f}" for v in r.get("per_100") or []) + "] t/s"


def tg64(ctx, url):
    """BASE[:32], 64 tokens with ignore_eos, not streamed, twice: (the faster predicted_per_second or None, error or
    None). The faster of two, so that one slow run (the Mac's runs moved by up to 10%) does not fail restart's check,
    while a server slow throughout still does."""
    speeds = []
    for _ in range(2):
        status, data, err = post(url, "/completion", {"prompt": ctx.base[:32], "cache_prompt": False, "n_predict": 64,
                                                      "ignore_eos": True, "temperature": 0}, 300)
        speed = ((data.get("timings") if isinstance(data, dict) else None) or {}).get("predicted_per_second")
        if err is not None or speed is None:
            return None, err or "no timings in the reply"
        speeds.append(speed)
    return max(speeds), None


def kill_during(ctx, proc, url, t_kill):
    """A streamed 2048-token prefill, the server killed with SIGKILL t_kill s after the send. Returns (what the client
    saw, ms at the kill)."""
    conn = connect(url, 10)
    payload = json.dumps({"prompt": ctx.base[:2048], "cache_prompt": False, "n_predict": 1, "stream": True,
                          "temperature": 0}).encode()
    saw = None
    t0 = time.perf_counter()
    try:
        conn.request("POST", "/completion", payload, {"Content-Type": "application/json"})
    except sc.HTTP_ERRORS as e:
        saw = f"the send failed: {type(e).__name__}: {e}"
    time.sleep(max(0.0, t_kill - (time.perf_counter() - t0)))
    killed_ms = (time.perf_counter() - t0) * 1000.0
    try:
        os.kill(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
    if proc in live:
        live.remove(proc)
    if saw is None:
        try:
            r = conn.getresponse()
            saw = f"HTTP {r.status}"
            saw += f", the reply complete ({len(r.read())} bytes)"
        except sc.HTTP_ERRORS as e:
            saw = (f"{saw}, then " if saw else "") + f"{type(e).__name__}: {e}"
    conn.close()
    return saw, killed_ms


def tcm_maps(pid):
    try:
        return sum("/dev/tcm" in ln for ln in Path(f"/proc/{pid}/maps").read_text(errors="replace").splitlines())
    except OSError:
        return None


def case_restart(ctx):
    kills = [1.0] if ctx.quick else [0.5, 1.0, 1.5]
    names = [f"restart {k} (kill at {t} s): the next server" for k, t in enumerate(kills, 1)]
    tcmtest, release = sc.tcm_tool(ctx.tcm_dir, "tcmtest"), sc.tcm_tool(ctx.tcm_dir, "tcmrelease")
    done, ended = 0, None
    s, url, secs, why = start_server(ctx, "restart-0", MEASURE_FLAGS)
    try:
        if secs is None:
            ended = f"the first server did not come up: {why}"
        else:
            before, err = tg64(ctx, url)
            row("restart", kind="before", tg64=before, error=err)
            for k, t_kill in enumerate(kills, 1):
                saw, killed_ms = kill_during(ctx, s, url, t_kill)
                marked = tcm_rc = dry_rc = None
                if tcmtest:
                    tcm_rc, out, _ = sc.run([tcmtest, "info"], ctx.logs / f"restart-{k}-tcm-info", timeout=120)
                    marked = sum("is_acquired=1" in ln for ln in out.splitlines())
                else:
                    sc.info(names[k - 1], f"marked blocks not read: no tcmtest in {ctx.tcm_dir}")
                if release:
                    dry_rc = sc.run([release], ctx.logs / f"restart-{k}-tcmrelease-dry", timeout=120)[0]
                else:
                    sc.info(names[k - 1], f"tcmrelease's dry run skipped: no tcmrelease in {ctx.tcm_dir}")
                s, url, secs, why = start_server(ctx, f"restart-{k}", MEASURE_FLAGS)
                done = k
                if secs is None:
                    row("restart", kind="round", round=k, kill_at=t_kill, killed_ms=killed_ms, client_saw=saw,
                        marked=marked, tcm_rc=tcm_rc, dry_rc=dry_rc, start_s=None, error=why, ok=False)
                    sc.report(names[k - 1], False, f"the new server did not come up: {why}; client saw: {saw}")
                    ended = f"round {k}'s new server did not come up"
                    break
                _, first, _, _, first_err = stream(url, "/completion", {"prompt": ctx.base[:32], "cache_prompt": False,
                                                                        "n_predict": 1, "stream": True,
                                                                        "temperature": 0}, 300)
                maps = tcm_maps(s.pid)
                if maps is None:
                    sc.info(names[k - 1], f"/dev/tcm lines not read: no /proc/{s.pid}/maps here")
                after, tg_err = tg64(ctx, url)
                cng = could_not_get(s)
                marked_ok = tcmtest is None or (tcm_rc == 0 and marked == 0)
                speed_ok = None not in (before, after) and after >= 0.95 * before
                why_not = [what for what, bad in (("'could not get' lines", cng), ("no /dev/tcm mapped", maps == 0),
                                                  ("the first request failed", first_err is not None),
                                                  ("blocks marked", not marked_ok),
                                                  ("tg64 below 95% of before", not speed_ok)) if bad]
                ok = not why_not
                change = f" ({100.0 * (after / before - 1.0):+.1f}%)" if None not in (before, after) and before else ""
                sc.report(names[k - 1], ok,
                          f"up after {secs:.1f} s; {cng} 'could not get' lines; /dev/tcm lines "
                          f"{'not read' if maps is None else maps}; marked blocks "
                          f"{'not read' if marked is None else marked}" +
                          ("" if tcm_rc in (None, 0) else f" (rc={tcm_rc})") +
                          f"; first token {fmt(first)} ms" + (f" ({first_err})" if first_err else "") +
                          f"; tg64 {fmt(after, 2)} t/s against {fmt(before, 2)} before{change}" +
                          (f" ({tg_err})" if tg_err else "") + f"; client saw: {saw}" +
                          (f"; failed on: {', '.join(why_not)}" if why_not else ""))
                row("restart", kind="round", round=k, kill_at=t_kill, killed_ms=killed_ms, client_saw=saw,
                    marked=marked, tcm_rc=tcm_rc, dry_rc=dry_rc, start_s=secs, first_ms=first, first_error=first_err,
                    tcm_maps=maps, before=before, after=after, tg64_error=tg_err, could_not_get=cng, ok=ok)
                before = after
    finally:
        stop_server(s)
    for name in names[done:]:
        sc.report(name, False, f"not run: {ended}")


def case_second(ctx):
    req = {"prompt": ctx.base[:64], "cache_prompt": False, "n_predict": 32, "stream": True, "temperature": 0}
    name = "second: B answers after A has gone, with A's text"
    a = b = None
    try:
        a, url_a, secs, why = start_server(ctx, "second-a", MEASURE_FLAGS)
        if secs is None:
            sc.report(name, False, f"A did not come up: {why}")
            return
        status, first, events, _, err = stream(url_a, "/completion", req, 300)
        text_a = stream_text(events)
        row("second", kind="a-reply", status=status, first_ms=first, text=text_a, error=err)
        if err:
            sc.report(name, False, f"A did not answer: {err}")
            return
        b, url_b, secs_b, why_b = start_server(ctx, "second-b", MEASURE_FLAGS)
        cng = could_not_get(b)
        row("second", kind="b-start", start_s=secs_b, could_not_get=cng, error=None if secs_b is not None else why_b)
        if secs_b is None:
            sc.report(name, False, f"B did not come up while A lived: {why_b}; {cng} 'could not get' lines")
            return
        t0 = time.perf_counter()
        status1, _, _, _, err1 = stream(url_b, "/completion", req, 300)
        seconds1 = time.perf_counter() - t0
        alive = b.poll() is None
        cng1 = could_not_get(b)
        row("second", kind="b-reply-beside-a", status=status1, seconds=seconds1, error=err1, b_alive=alive,
            could_not_get=cng1)
        stop_server(a)
        a = None
        status2, first2, events2, _, err2 = stream(url_b, "/completion", req, 300)
        text_b = stream_text(events2)
        same = text_b == text_a
        row("second", kind="b-reply-after-a", status=status2, first_ms=first2, text=text_b, equal_to_a=same, error=err2)
        i = next((i for i, (x, y) in enumerate(zip(text_b, text_a)) if x != y), min(len(text_b), len(text_a)))
        sc.report(name, status2 == 200 and err2 is None and same,
                  f"HTTP {status2}, first token {fmt(first2)} ms, " +
                  (f"{len(text_b)} characters equal to A's" if same else
                   f"text differs from A's at character {i}: B {text_b[i:i + 20]!r}, A {text_a[i:i + 20]!r}") +
                  (f"; {err2}" if err2 else "") + f"; B up after {secs_b:.1f} s beside A")
        if status1 == 200 and err1 is None and cng1 == 0:
            sc.info("second", "B was not refused while A lived (the cores were not refused here): "
                              f"HTTP {status1} after {seconds1:.1f} s, B alive after: {alive}")
        else:
            sc.info("second", f"B refused while A lived: HTTP {status1} after {seconds1:.1f} s ({err1 or 'no error'}), "
                              f"{cng1} 'could not get' lines, B alive after: {alive}")
    finally:
        stop_server(a)
        stop_server(b)


def case_alongside(ctx):
    if not ctx.ime_bench.is_file():
        sc.info("alongside", f"skipped: no {ctx.ime_bench}")
        return
    req = {"prompt": ctx.base[:64], "cache_prompt": False, "n_predict": 64, "stream": True, "temperature": 0}
    n_alone, n_beside = (1, 2) if ctx.quick else (3, 5)
    release = sc.tcm_tool(ctx.tcm_dir, "tcmrelease")
    ref, loop, ended, mine = None, None, None, []

    def ask(phase, i):
        status, first, events, t, err = stream(url, "/completion", req, 300)
        text = stream_text(events)
        r = row("alongside", kind="q", phase=phase, i=i, status=status, first_ms=first,
                predicted_per_second=t.get("predicted_per_second"), equal=ref is None or text == ref, error=err)
        mine.append(r)
        return text, err

    s, url, secs, why = start_server(ctx, "alongside", MEASURE_FLAGS)
    try:
        if secs is None:
            ended = f"the server did not come up: {why}"
            return
        for i in range(1, n_alone + 1):
            text, err = ask("alone", i)
            if err:
                ended = f"alone {i}: {err}"
                return
            if ref is None:
                ref = text
        loop = ImeLoop(ctx)
        ime_loops.append(loop)
        loop.start()
        while loop.is_alive() and (loop.first_start is None or time.time() - loop.first_start < 15):
            time.sleep(0.2)
        for i in range(1, n_beside + 1):
            _, err = ask("beside", i)
            if err:
                ended = f"beside {i}: {err}"
                return
        loop.finish(600)
        if release:
            sc.run([release, "--apply"], ctx.logs / "alongside-tcmrelease-after", timeout=120)
        _, err = ask("after", 1)
        if err:
            ended = f"after: {err}"
    except BaseException:
        if loop is not None:
            loop.stop_event.set()
            stop_server(loop.proc)
        raise
    finally:
        if loop is not None and loop.is_alive():
            loop.finish(600)
        stop_server(s)
        if loop is not None:
            for it in loop.iterations:
                row("alongside", kind="ime", **it)
            if loop.error:
                sc.info("alongside", f"the IME loop stopped by an error: {loop.error}")
        if ended:
            sc.info("alongside", f"ended: {ended}")
        if mine:
            for line in alongside_lines(rows):
                sc.info("alongside", line)


def alongside_lines(all_rows):
    qs = [r for r in all_rows if r["case"] == "alongside" and r.get("kind") == "q"]
    ime = [r for r in all_rows if r["case"] == "alongside" and r.get("kind") == "ime"]
    beside = [r for r in qs if r["phase"] == "beside"]
    lines = [f"{sum(1 for r in beside if r['equal'] and not r['error'])} of {len(beside)} beside equal the reference"]
    for phase in ("alone", "beside", "after"):
        rs = [r for r in qs if r["phase"] == phase]
        lines.append(f"{phase}: " + (", ".join(f"{fmt(r['predicted_per_second'], 2)} t/s (first token "
                                               f"{fmt(r['first_ms'])} ms{'' if r['equal'] else ', text differs'}"
                                               f"{', ' + r['error'] if r['error'] else ''})" for r in rs) or "none"))
    lines.append("IME iterations: " + ("; ".join(f"{r['iteration']}: rc={r['rc']}, {r['seconds']} s, "
                                                 f"{speeds_text({k: r[k] for k in ('pp128', 'tg128') if k in r})}"
                                                 for r in ime) or "none"))
    return lines


# the summary

def summary(ctx):
    sc.say("\nSummary")
    case_rows = {c: [r for r in rows if r["case"] == c] for c in CASES}
    grid = [r for r in case_rows["grid"] if r.get("kind") == "tail" and not r["error"]]
    if grid:
        depths, tails = sorted({r["depth"] for r in grid}), sorted({r["tail"] for r in grid})
        for title, key, q in (("client ms p50", "client_ms", 0.5), ("client ms p95", "client_ms", 0.95),
                              ("prompt_ms p50", "prompt_ms", 0.5)):
            sc.say(f"\ngrid, {title} (tails down, depths across)")
            sc.say(f"{'tail':>6}" + "".join(f"{d:>10}" for d in depths))
            for n in tails:
                sc.say(f"{n:>6}" + "".join(f"{fmt(grid_cell(grid, d, n, q, key)):>10}" for d in depths))
    for mode in ("off", "on"):
        rs = [r for r in case_rows["chat"] if r["mode"] == mode]
        if rs:
            sc.say(f"\nchat, thinking {mode}")
        for r in rs:
            sc.say(f"  turn {r['turn']:>2}: HTTP {r['status']}, first event {fmt(r['first_event_ms'])} ms, first "
                   f"reasoning {fmt(r['first_reasoning_ms'])} ms, first content {fmt(r['first_content_ms'])} ms, "
                   f"cache_n {r['cache_n']}, prompt_n {r['prompt_n']}, prompt_ms {fmt(r['prompt_ms'])}, predicted_n "
                   f"{r['predicted_n']} at {fmt(r['predicted_per_second'], 2)} t/s, finish {r['finish_reason']}"
                   f"{' (max_tokens)' if r['hit_max_tokens'] else ''}, {r['content_chars']} characters" +
                   (f"; {r['error']}" if r["error"] else ""))
    cancelled = {r["repeat"]: r for r in case_rows["cancel"] if r.get("kind") == "cancelled"}
    if cancelled:
        sc.say("\ncancel")
    for r in case_rows["cancel"]:
        if r.get("kind") == "followup":
            c = cancelled.get(r["repeat"], {})
            sc.say(f"  repeat {r['repeat']}: closed after {fmt(c.get('closed_ms'))} ms (answered first: "
                   f"{c.get('finished_before_close')}); follow-up client {fmt(r['client_ms'])} ms, prompt_ms "
                   f"{fmt(r['prompt_ms'])}, wait {fmt(r['wait_ms'])} ms, cache_n {r['cache_n']}, prompt_n "
                   f"{r['prompt_n']}" +
                   (f"; {r['error']}" if r["error"] else ""))
    tg = [r for r in case_rows["tg512"] if r.get("kind") == "run" and not r["error"]]
    if tg:
        sc.say(f"\ntg512 ({tg[0]['n_predict']} tokens), server t/s")
    for d in sorted({r["depth"] for r in tg}):
        rs = [r for r in tg if r["depth"] == d]
        speeds = [r["predicted_per_second"] for r in rs]
        sc.say(f"  depth {d:>5}: p50 {fmt(pct(speeds, 0.5), 2)}, min {fmt(min(speeds), 2)}, max {fmt(max(speeds), 2)} "
               f"over {len(rs)} runs; per 100 tokens, first run {per100_text(rs[0])}, last run {per100_text(rs[-1])}")
    rounds = [r for r in case_rows["restart"] if r.get("kind") == "round"]
    if rounds:
        sc.say("\nrestart")
    for r in rounds:
        sc.say(f"  round {r['round']} (kill at {r['kill_at']} s): {'PASS' if r['ok'] else 'FAIL'}; client saw "
               f"{r['client_saw']}; marked {r['marked']}; new server up after {fmt(r['start_s'])} s, first token "
               f"{fmt(r.get('first_ms'))} ms, /dev/tcm lines {r.get('tcm_maps')}, 'could not get' "
               f"{r.get('could_not_get')}, tg64 {fmt(r.get('after'), 2)} against {fmt(r.get('before'), 2)} t/s")
    if case_rows["second"]:
        sc.say("\nsecond")
    for r in case_rows["second"]:
        fields = {k: v for k, v in r.items() if k not in ("case", "kind", "text")}
        sc.say(f"  {r['kind']}: " +
               ", ".join(f"{k} {fmt(v) if isinstance(v, float) else v}" for k, v in fields.items()))
    if case_rows["alongside"]:
        sc.say("\nalongside")
        for line in alongside_lines(rows):
            sc.say(f"  {line}")

    sc.say("\nTargets (batch-one.md section 5, agreed 2026-11-06) against this run")
    parts = []
    for n in (26, 38, 110):
        p50, p95 = grid_cell(grid, 4096, n, 0.5), grid_cell(grid, 4096, n, 0.95)
        parts.append(f"tail {n} not measured in this run" if p50 is None else
                     f"tail {n} p50 {fmt(p50)} ms, p95 {fmt(p95)} ms")
    sc.info("target", "chat turn at depth 4096 at most 1 s: " + "; ".join(parts))
    p50, p95 = grid_cell(grid, 0, 500, 0.5), grid_cell(grid, 0, 500, 0.95)
    sc.info("target", "fresh 500-token prompt at most 4 s: " +
            ("not measured in this run" if p50 is None else f"depth 0 tail 500 p50 {fmt(p50)} ms, p95 {fmt(p95)} ms"))
    for d, target in ((0, "no worse than today"), (4096, "at least 9 t/s")):
        rs = [r for r in tg if r["depth"] == d]
        sc.info("target", f"tg512 from depth {d} {target}: " +
                ("not measured in this run" if not rs else
                 f"p50 {fmt(pct([r['predicted_per_second'] for r in rs], 0.5), 2)} t/s over {rs[0]['n_predict']} "
                 f"tokens" +
                 (" (this run is the baseline)" if d == 0 else "")))
    sc.info("target", "a saved system prompt (T3's target) is not measured here")


def grid_cell(grid, depth, tail, q, key="client_ms"):
    return pct([r[key] for r in grid if r["depth"] == depth and r["tail"] == tail], q)


# main

def run_case(ctx, name, fn):
    load = os.getloadavg()
    sc.info(name, f"start, load {load[0]:.2f} {load[1]:.2f} {load[2]:.2f}")
    row("load", of=name, at="start", loadavg=list(load))
    try:
        fn(ctx)
    except Exception as e:
        sc.report(name, False, f"stopped by an error: {e}")
        (ctx.logs / f"{name}-error.txt").write_text(traceback.format_exc())
    for proc in list(live):
        stop_server(proc)
    load = os.getloadavg()
    sc.info(name, f"end, load {load[0]:.2f} {load[1]:.2f} {load[2]:.2f}")
    row("load", of=name, at="end", loadavg=list(load))


def case_list(text):
    names = [s.strip() for s in text.split(",") if s.strip()]
    bad = [n for n in names if n not in CASES]
    if bad or not names:
        raise argparse.ArgumentTypeError(f"unknown case {', '.join(bad) or repr(text)} (choose from {','.join(CASES)})")
    return [c for c in CASES if c in names]


def git_commit(logs):
    git = ["git", "-c", f"safe.directory={sc.REPO}", "-C", str(sc.REPO)]
    rc, out, _ = sc.run([*git, "rev-parse", "--short", "HEAD"], logs / "git-rev-parse", timeout=60)
    if rc != 0 or not out.strip():
        return "unknown"
    rc_s, status, _ = sc.run([*git, "status", "--porcelain"], logs / "git-status", timeout=60)
    n = len(status.splitlines()) if rc_s == 0 else 0
    return out.strip() + (f" ({n} files uncommitted)" if n else "")


def on_sigterm(signum, frame):
    raise KeyboardInterrupt


def main():
    global jsonl
    ap = argparse.ArgumentParser(description="TTFT baseline for the FlagOS SpacemiT provider (run on the K3)")
    ap.add_argument("--build-dir", default=str(sc.REPO / "build-flagos"),
                    help="FlagOS build directory (default: %(default)s)")
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-0.6B-Q4_0.gguf"),
                    help="model (default: %(default)s)")
    ap.add_argument("--cases", type=case_list, default=list(CASES),
                    help=f"comma list from {','.join(CASES)} (default: all; always run in that order)")
    ap.add_argument("--repeats", type=int, default=None, help="the grid's repeats (default: 5, 2 with --quick)")
    ap.add_argument("--quick", action="store_true", help="every case cut down to minutes")
    ap.add_argument("--ime-build", default=str(sc.REPO / "build-ime"),
                    help="upstream IME build for the control and alongside (skipped if absent)")
    ap.add_argument("--tcm-dir", default=os.path.expanduser("~/tcmtest"), help="directory with tcmtest and tcmrelease")
    ap.add_argument("--segv", default="",
                    help="crash reporter (scripts/segv.c built as .so) to preload into the servers and llama-bench")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="report placement off the provider without stopping the run")
    args = ap.parse_args()
    if args.repeats is not None and args.repeats < 1:
        ap.error("--repeats must be at least 1")
    if args.segv:
        if not Path(args.segv).is_file():
            sys.exit(f"--segv: not found: {args.segv}")
        sc.preload["LD_PRELOAD"] = str(Path(args.segv).resolve())

    bdir = Path(args.build_dir).resolve()
    ctx = SimpleNamespace(model=args.model, server=bdir / "bin" / "llama-server", quick=args.quick,
                          repeats=args.repeats if args.repeats is not None else (2 if args.quick else 5),
                          ime_bench=Path(args.ime_build) / "bin" / "llama-bench", tcm_dir=args.tcm_dir,
                          allow_cpu=args.allow_cpu, base=None, logs=bdir / f"logs-ttft-{Path(args.model).stem}")
    problem = (f"model not found: {args.model}" if not Path(args.model).is_file() else
               f"not found: {ctx.server}" if not ctx.server.is_file() else "")
    if problem:
        sc.report("model and llama-server", False, problem)
        sc.say(f"\nTTFT: 0/{len(sc.results)} checks passed")
        sys.exit(1)

    ctx.logs.mkdir(parents=True, exist_ok=True)
    sc.summary = open(ctx.logs / "summary.txt", "w")
    jsonl = open(ctx.logs / "ttft.jsonl", "w")
    signal.signal(signal.SIGTERM, on_sigterm)
    commit = git_commit(ctx.logs)
    # without the node name: the board's name stays out of logs that may be pasted into the docs
    u = os.uname()
    uname = f"{u.sysname} {u.release} {u.machine}"
    start = time.strftime("%Y-%m-%d %H:%M:%S")
    sc.say(f"repo {sc.REPO}\nbuild {bdir}\nlogs {ctx.logs}\nmodel {args.model}\ncommit {commit}\nhost {uname}\n"
           f"start {start}" + (f"\ncrash reporter {sc.preload['LD_PRELOAD']}" if sc.preload else "") + "\n")
    row("meta", repo=str(sc.REPO), build_dir=str(bdir), logs=str(ctx.logs), model=args.model, commit=commit,
        uname=uname, start=start, crash_reporter=sc.preload.get("LD_PRELOAD"),
        args={**vars(args), "repeats": ctx.repeats}, measure_flags=MEASURE_FLAGS, chat_flags=CHAT_FLAGS)

    fns = {"grid": case_grid, "chat": case_chat, "cancel": case_cancel, "tg512": case_tg512, "restart": case_restart,
           "second": case_second, "alongside": case_alongside}
    interrupted = False
    try:
        if placement(ctx):
            control(ctx, "start")
            for name in args.cases:
                run_case(ctx, name, fns[name])
            control(ctx, "end")
    except KeyboardInterrupt:
        interrupted = True
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        sc.say("\nstopped by Ctrl-C or SIGTERM: the children stopped, the summary from the rows so far")
        stop_all()
    summary(ctx)
    n_fail = sc.results.count(False)
    sc.say(f"\nTTFT: {len(sc.results) - n_fail}/{len(sc.results)} checks passed" +
           ("" if n_fail == 0 else f", see {ctx.logs}"))
    jsonl.close()
    sc.summary.close()
    sys.exit(130 if interrupted else 1 if n_fail else 0)


if __name__ == "__main__":
    main()
