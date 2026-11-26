#!/usr/bin/env python3
# Per-op tables from the trace of SpacemiT's llama.cpp fork (ggml_trace.json, one begin and one end event per graph
# node on each worker thread), and the fork side by side with our op tables (the text that op-stats.py show prints).
#
# Usage: fork-trace.py show <trace>                     the fork's decodes, then its per-graph table by kind of node
#        fork-trace.py compare <ours-ops.txt> <trace> [--ours-rows N] [--n-kv K]
#                                                       our per-split section of the same rows against the fork's graph:
#                                                       table A by op (MUL_MAT by weight type), table B by dense weight
# The trace may be .gz (by suffix) or plain. A decode (one model graph) starts at each CPU (pid 2) begin event
# "GET_ROWS (model.input_embed)"; the device's (pid 3) events up to the next start belong to it. The first decode (the
# warm-up), the last (cut short in the file), incomplete or inconsistent ones and those with a node count other than
# the most common one are not used; the rest are averaged node by node. Per node: own = the slowest worker's end - begin,
# wall = the latest end - the earliest begin, gap = the next node's earliest begin - this node's latest end (clamped at 0).
import argparse
import bisect
import gzip
import json
import re
import sys
from collections import Counter, OrderedDict

START = "GET_ROWS (model.input_embed)"


def fail(msg):
    print(msg, file=sys.stderr)
    sys.exit(2)


def weight_of(args):
    # (short name, type) of the first source that is a weight, else None
    for s in ("src0", "src1"):
        name = args.get(s + "_name")
        if not isinstance(name, str):
            continue
        if name.startswith("blk.") or name == "output.weight" or name.endswith(".weight"):
            short = re.sub(r"^blk\.\d+\.", "", name)
            if short.endswith(".weight"):
                short = short[: -len(".weight")]
            return short, str(args.get(s + "_type", "?"))
    return None


def kind_of(args, kinds):
    # (label, op, short, type) of a node, one shared tuple per label
    op = str(args.get("op_name", "?"))
    w = weight_of(args)
    if op in ("MUL_MAT", "MUL_MAT_ID") and w:
        k = (f"{op} {w[0]} {w[1]}", op, w[0], w[1])
    elif op == "MUL_MAT":
        k = ("MUL_MAT (no weight)", op, None, None)
    else:
        k = (op, op, None, None)
    return kinds.setdefault(k[0], k)


def read_trace(path):
    # device events as (ts, ph, tid, name, kind) sorted by ts, decode starts as (ts, rows), and counts
    if path.endswith(".gz"):
        f = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        f = open(path, encoding="utf-8", errors="replace")
    dev, starts, kinds, names = [], [], {}, {}
    read = skipped = 0
    with f:
        for line in f:
            s = line.strip()
            if s in ("", "[", "]"):
                continue
            if s.endswith(","):
                s = s[:-1]
            try:
                e = json.loads(s)
            except ValueError:
                skipped += 1
                continue
            if not isinstance(e, dict):
                skipped += 1
                continue
            read += 1
            ph, ts, pid = e.get("ph"), e.get("ts"), e.get("pid")
            if ph not in ("B", "E") or not isinstance(ts, (int, float)):
                continue
            name = names.setdefault(str(e.get("name")), str(e.get("name")))
            args = e.get("args") if isinstance(e.get("args"), dict) else {}
            if pid == 3:
                dev.append((ts, ph, e.get("tid"), name, kind_of(args, kinds) if ph == "B" else None))
            elif pid == 2 and ph == "B" and name == START:
                shape = str(args.get("out_shape", ""))
                dims = re.findall(r"\d+", shape)
                starts.append((ts, int(dims[1]) if len(dims) > 1 else None))
    dev.sort(key=lambda x: x[0])
    starts.sort(key=lambda x: x[0])
    return dev, starts, read, skipped


def pair_worker(evs):
    # (begin, end, name, kind) per node of one worker, and the first problem or None
    pairs = []
    i = 0
    while i < len(evs):
        b = evs[i]
        if b[1] != "B":
            return pairs, f"E '{b[3]}' with no B after node {len(pairs)}"
        if i + 1 >= len(evs) or evs[i + 1][1] != "E":
            return pairs, f"node {len(pairs)} '{b[3]}' has no E"
        e = evs[i + 1]
        if e[3] != b[3]:
            return pairs, f"node {len(pairs)} B '{b[3]}' ended by E '{e[3]}'"
        pairs.append((b[0], e[0], b[3], b[4]))
        i += 2
    return pairs, None


def load(path):
    try:
        dev, starts, read, skipped = read_trace(path)
    except OSError as err:
        fail(f"cannot read {path}: {err}")
    workers = sorted({e[2] for e in dev}, key=str)
    t0 = [s[0] for s in starts]
    before = 0
    buckets = [[] for _ in starts]
    for e in dev:
        i = bisect.bisect_right(t0, e[0]) - 1
        if i < 0:
            before += 1
        else:
            buckets[i].append(e)
    decodes = []
    for i, (ts, rows) in enumerate(starts):
        evs = buckets[i]
        d = {"index": i, "start": ts, "rows": rows, "reasons": [], "incomplete": False, "inconsistent": False,
             "span": (max(e[0] for e in evs) - min(e[0] for e in evs)) if evs else 0.0,
             "period": (starts[i + 1][0] - ts) if i + 1 < len(starts) else None}
        per = {}
        if not evs:
            d["incomplete"] = True
            d["reasons"].append("incomplete: no device events")
        problems = OrderedDict()
        for w in workers:
            pairs, problem = pair_worker([e for e in evs if e[2] == w])
            per[w] = pairs
            if problem:
                problems.setdefault(problem, []).append(str(w))
        for problem, ws in problems.items():
            d["incomplete"] = True
            d["reasons"].append(f"incomplete: worker{'s' if len(ws) > 1 else ''} {','.join(ws)} {problem}")
        counts = [len(p) for p in per.values()]
        d["counts"] = counts
        d["nodes"] = counts[0] if counts and min(counts) == max(counts) else None
        if counts and min(counts) != max(counts):
            d["inconsistent"] = True
            d["reasons"].append("inconsistent: node counts " + ", ".join(f"w{w} {len(p)}" for w, p in per.items()))
        if workers:
            ref_w = workers[0]
            for k in range(min(counts)):
                ref = per[ref_w][k][2]
                other = next((w for w in workers if per[w][k][2] != ref), None)
                if other is not None:
                    d["inconsistent"] = True
                    d["reasons"].append(f"inconsistent: node {k} is '{ref}' on worker {ref_w} but '{per[other][k][2]}' "
                                        f"on worker {other}")
                    break
        d["per"] = per
        decodes.append(d)
    return {"path": path, "read": read, "skipped": skipped, "before": before, "workers": workers,
            "decodes": decodes}


def select(t):
    # mark the decodes used and average their nodes into t["graph"]
    ds = t["decodes"]
    for d in ds:
        if d["index"] == 0:
            d["reasons"].insert(0, "first (warm-up)")
        if d["index"] == len(ds) - 1:
            d["reasons"].insert(0, "last (cut short)")
    cand = [d for d in ds if not d["reasons"]]
    t["graph"] = None
    if not cand:
        return
    c = Counter(d["nodes"] for d in cand)
    top = max(c.values())
    mode = max(n for n, v in c.items() if v == top)
    kept = []
    for d in cand:
        if d["nodes"] != mode:
            d["reasons"].append(f"node count {d['nodes']}, not the most common {mode}")
            continue
        if kept:
            ref = kept[0]["per"][t["workers"][0]]
            mine = d["per"][t["workers"][0]]
            k = next((k for k in range(mode) if mine[k][3] is not ref[k][3]), None)
            if k is not None:
                d["reasons"].append(f"node {k} is {mine[k][3][0]}, decode {kept[0]['index']} has {ref[k][3][0]}")
                continue
        kept.append(d)
    own, wall, gap = [0.0] * mode, [0.0] * mode, [0.0] * mode
    clamped = 0
    for d in kept:
        nodes = list(zip(*d["per"].values()))  # nodes[k] = the k-th pair of every worker
        for k, ps in enumerate(nodes):
            own[k] += max(e - b for b, e, _, _ in ps)
            wall[k] += max(p[1] for p in ps) - min(p[0] for p in ps)
            if k + 1 < len(nodes):
                g = min(p[0] for p in nodes[k + 1]) - max(p[1] for p in ps)
                if g < 0:
                    clamped += 1
                    g = 0
                gap[k] += g
    n = len(kept)
    rows = Counter(d["rows"] for d in kept)
    t["graph"] = {"decodes": n, "nodes": mode, "rows": rows.most_common(1)[0][0], "all_rows": sorted(rows, key=str),
                  "span": sum(d["span"] for d in kept) / n / 1000, "clamped": clamped,
                  "kinds": [p[3] for p in kept[0]["per"][t["workers"][0]]],
                  "own": [x / n / 1000 for x in own], "wall": [x / n / 1000 for x in wall],
                  "gap": [x / n / 1000 for x in gap]}


def load_selected(path):
    t = load(path)
    select(t)
    return t


def print_decodes(t):
    print(f"file {t['path']}: {t['read']} events read, {t['skipped']} lines skipped, {len(t['decodes'])} decodes found")
    if t["before"]:
        print(f"   {t['before']} device events before the first decode ignored")
    for d in t["decodes"]:
        c = d["counts"]
        nodes = (str(c[0]) if min(c) == max(c) else f"{min(c)}-{max(c)}") if c else "0"
        period = f"{d['period'] / 1000:.3f}" if d["period"] is not None else "-"
        print(f"   decode {d['index']}: rows {d['rows']}, nodes {nodes}, span {d['span'] / 1000:.3f} ms, "
              f"period {period} ms, {'; '.join(d['reasons']) or 'used'}")


def check_usable(t):
    if len(t["decodes"]) < 3:
        fail(f"{t['path']}: {len(t['decodes'])} decode(s) found, need at least 3 (the first and last are not used)")
    if not t["graph"]:
        fail(f"{t['path']}: no usable decode")


def by_kind(g):
    # label -> {"kind", "n", "own", "wall", "gap"} per graph
    out = OrderedDict()
    for k, kind in enumerate(g["kinds"]):
        v = out.setdefault(kind[0], {"kind": kind, "n": 0, "own": 0.0, "wall": 0.0, "gap": 0.0})
        v["n"] += 1
        v["own"] += g["own"][k]
        v["wall"] += g["wall"][k]
        v["gap"] += g["gap"][k]
    return out


def show(path):
    t = load_selected(path)
    print_decodes(t)
    check_usable(t)
    g = t["graph"]
    own, gap, wall = sum(g["own"]), sum(g["gap"]), sum(g["wall"])
    print(f"per graph ({g['decodes']} decodes averaged): {g['rows']} rows, {g['nodes']} nodes; span {g['span']:.3f} ms = "
          f"own {own:.3f} + between nodes {gap:.3f} + rest {g['span'] - own - gap:.3f}; wall {wall:.3f} ms")
    if len(g["all_rows"]) > 1:
        print(f"   note: the used decodes have rows {g['all_rows']}")
    if g["clamped"]:
        print(f"   note: {g['clamped']} negative gaps clamped at 0")
    rows = by_kind(g)
    w = max([44] + [len(n) + 2 for n in rows])
    print(f"{'name':{w}}{'n':>6}{'own ms':>12}{'share':>8}{'wall ms':>12}{'gap ms':>11}")
    for name, v in sorted(rows.items(), key=lambda x: -x[1]["own"]):
        share = 100 * v["own"] / own if own else 0.0
        print(f"{name:{w}}{v['n']:6d}{v['own']:12.3f}{share:7.1f}%{v['wall']:12.3f}{v['gap']:11.3f}")
    print(f"{'(total)':{w}}{g['nodes']:6d}{own:12.3f}{100.0 if own else 0.0:7.1f}%{wall:12.3f}{gap:11.3f}")


SECTION = re.compile(r"^== per-split, (\d+) row\(s\), n_kv (\d+):(.*)$")
ROW = re.compile(r"^\s{3}(\S.*?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)%")


def read_ours(path):
    # per-split sections: {"rows", "n_kv", "graphs", "title", "steps", "wall" (us), "rows_": [(name, n, wall us)]}
    try:
        f = open(path, encoding="utf-8", errors="replace")
    except OSError as err:
        fail(f"cannot read {path}: {err}")
    secs, cur = [], None
    with f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("== "):
                m = SECTION.match(line)
                cur = None
                if m:
                    g = re.search(r"(\d+) graphs counted", m.group(3))
                    cur = {"rows": int(m.group(1)), "n_kv": int(m.group(2)), "graphs": int(g.group(1)) if g else 0,
                           "title": line[3:], "steps": None, "wall": None, "rows_": []}
                    secs.append(cur)
                continue
            if cur is None:
                continue
            if line.startswith("   per graph:"):
                m = re.search(r"([0-9.]+) steps; wall ([0-9.]+) us", line)
                if m:
                    cur["steps"], cur["wall"] = float(m.group(1)), float(m.group(2))
                continue
            if line.startswith("   name"):
                continue
            m = ROW.match(line)
            if m:
                try:
                    cur["rows_"].append((m.group(1), float(m.group(2)), float(m.group(3))))
                except ValueError:
                    pass
    return secs


def fmt_n(x):
    return f"{x:.0f}" if abs(x - round(x)) < 0.005 else f"{x:.2f}"


def ratio(a, b):
    return f"{b / a:.2f}" if a else "-"


def compare(ours_path, trace_path, ours_rows, n_kv):
    secs = read_ours(ours_path)
    t = load_selected(trace_path)
    if len(t["decodes"]) < 3 or not t["graph"]:
        print_decodes(t)
        check_usable(t)
    g = t["graph"]
    rows = ours_rows if ours_rows is not None else g["rows"]
    match = [s for s in secs if s["rows"] == rows and (n_kv is None or s["n_kv"] == n_kv)]
    if not match:
        found = "\n".join(f"   {s['title']}" for s in secs) or "   (none)"
        fail(f"{ours_path}: no per-split section with {rows} row(s)" + (f" and n_kv {n_kv}" if n_kv is not None else "")
             + f"; sections found:\n{found}")
    s = max(match, key=lambda x: x["graphs"])  # max keeps the first on a tie
    own, gap = sum(g["own"]), sum(g["gap"])
    ours_wall = s["wall"] / 1000 if s["wall"] is not None else sum(r[2] for r in s["rows_"]) / 1000
    steps = fmt_n(s["steps"]) if s["steps"] is not None else "?"
    print(f"ours: {ours_path}: {s['rows']} row(s), n_kv {s['n_kv']}, {s['graphs']} graphs counted; per graph {steps} "
          f"steps, wall {ours_wall:.1f} ms")
    print(f"fork: {trace_path}: {g['rows']} rows, {g['decodes']} decodes used, {g['nodes']} nodes; span {g['span']:.1f} ms,"
          f" own total {own:.1f} ms, between nodes {gap:.1f} ms")

    # table A: by op, MUL_MAT by weight type
    ta = OrderedDict()
    tb = OrderedDict()
    zero = lambda: {"on": 0.0, "oms": 0.0, "fn": 0, "fms": 0.0, "o": False, "f": False}
    for name, n, wall in s["rows_"]:
        op = re.split(r"[ .]", name, maxsplit=1)[0]
        tok = name.split()
        grp = op
        if op == "MUL_MAT":
            grp = f"MUL_MAT {tok[2]}" if len(tok) >= 3 else "MUL_MAT (no weight)"
            if len(tok) >= 3:
                v = tb.setdefault((tok[1], tok[2]), zero())
                v["on"] += n
                v["oms"] += wall / 1000
                v["o"] = True
        v = ta.setdefault(grp, zero())
        v["on"] += n
        v["oms"] += wall / 1000
        v["o"] = True
    for label, v in by_kind(g).items():
        _, op, short, wtype = v["kind"]
        grp = op
        if op == "MUL_MAT":
            grp = f"MUL_MAT {wtype}" if short is not None else "MUL_MAT (no weight)"
            if short is not None:
                x = tb.setdefault((short, wtype), zero())
                x["fn"] += v["n"]
                x["fms"] += v["own"]
                x["f"] = True
        x = ta.setdefault(grp, zero())
        x["fn"] += v["n"]
        x["fms"] += v["own"]
        x["f"] = True

    def cells(v):
        o = (f"{fmt_n(v['on']):>8}{v['oms']:>11.1f}") if v["o"] else f"{'-':>8}{'-':>11}"
        f = (f"{v['fn']:>8d}{v['fms']:>13.1f}") if v["f"] else f"{'-':>8}{'-':>13}"
        r = ratio(v["oms"], v["fms"]) if v["o"] and v["f"] else "-"
        return f"{o}{f}{r:>11}"

    print()
    print("A: by op (MUL_MAT by weight type)")
    w = max([28] + [len(k) + 2 for k in ta])
    print(f"{'group':{w}}{'ours n':>8}{'ours ms':>11}{'fork n':>8}{'fork own ms':>13}{'fork/ours':>11}")
    for k, v in sorted(ta.items(), key=lambda x: -max(x[1]["oms"], x[1]["fms"])):
        print(f"{k:{w}}{cells(v)}")
    tot = {"on": sum(r[1] for r in s["rows_"]), "oms": sum(r[2] for r in s["rows_"]) / 1000, "fn": g["nodes"],
           "fms": own, "o": True, "f": True}
    print(f"{'(total)':{w}}{cells(tot)}")
    print(f"ours wall {ours_wall:.1f} ms; fork own {own:.1f} ms + between nodes {gap:.1f} ms = {own + gap:.1f} ms")

    print()
    print("B: dense matmuls by weight")
    w = max([24] + [len(k[0]) + 2 for k in tb])
    print(f"{'weight':{w}}{'type':8}{'ours n':>8}{'ours ms':>11}{'fork n':>8}{'fork own ms':>13}{'fork/ours':>11}")
    for (short, wtype), v in sorted(tb.items(), key=lambda x: (not x[1]["o"], -x[1]["oms"], -x[1]["fms"])):
        print(f"{short:{w}}{wtype:8}{cells(v)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="per-op tables from the SpacemiT fork's ggml_trace.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("show")
    p.add_argument("trace")
    p = sub.add_parser("compare")
    p.add_argument("ours")
    p.add_argument("trace")
    p.add_argument("--ours-rows", type=int)
    p.add_argument("--n-kv", type=int)
    a = ap.parse_args()
    if a.cmd == "show":
        show(a.trace)
    else:
        compare(a.ours, a.trace, a.ours_rows, a.n_kv)
