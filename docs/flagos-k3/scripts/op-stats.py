#!/usr/bin/env python3
# Summary of the SpacemiT provider's op counters (FLAGOS_SPACEMIT_TEST_OP_STATS; docs/flagos-k3/fusion-and-splitting.md
# §7). Each line of a counters file is one name (or "(all)") of one kind of graph, per graph.
#
# Usage: op-stats.py show <file>...                  per kind of graph, every name's time per graph, largest first
#        op-stats.py diff <file>... -- <file>...     per kind of graph in both sets, the second set against the first,
#                                                    name by name, largest change first
# The files of one set (one mode's runs) are merged: per kind (launch, rows, n_kv), every name's values weighted by the
# graphs counted. Times in us per graph: wall = the mean tile's kernel and barrier time, kernel = the mean tile's kernel
# time, imbal. = each step's slowest tile less the mean, summed (busy - kernel), barrier = wall - busy (the barriers' own
# cost and the tiles' skew leaving them).
import json
import sys
from collections import OrderedDict


def load(paths):
    # (launch, rows, n_kv) -> {"graphs", "structures", "names": OrderedDict(label -> sums weighted by graphs)}
    kinds = {}
    for path in paths:
        for line in open(path):
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            g = r["graphs"]
            if g == 0:
                continue  # a kind seen once: only its first graph, which is not counted
            k = kinds.setdefault((r["launch"], r["rows"], r["n_kv"]),
                                 {"graphs": 0, "structures": set(), "tiles": r["tiles"], "names": OrderedDict()})
            if r["label"] == "(all)":
                k["graphs"] += g
                k["structures"].add((path, r["structure"]))
            v = k["names"].setdefault(r["label"], {"g": 0, "n": 0.0, "wall": 0.0, "kernel": 0.0, "busy": 0.0,
                                                    "launch": 0.0, "tile": [0.0] * r["tiles"],
                                                    "slowest": [0] * r["tiles"]})
            v["g"] += g
            for f, src in (("n", "n"), ("wall", "wall_us"), ("kernel", "kernel_us"), ("busy", "busy_us"),
                           ("launch", "launch_us")):
                v[f] += r[src] * g
            for t, x in enumerate(r["tile_us"][: len(v["tile"])]):
                v["tile"][t] += x * g
            for t, x in enumerate(r["slowest"][: len(v["slowest"])]):
                v["slowest"][t] += x
    return kinds


def per_graph(v, graphs):
    # a name's values per graph of the kind (a name missing from some structure counts 0 there)
    d = max(graphs, 1)
    out = {f: v[f] / d for f in ("n", "wall", "kernel", "busy", "launch")}
    out["imbal"] = out["busy"] - out["kernel"]
    out["barrier"] = out["wall"] - out["busy"]
    out["tile"] = [x / d for x in v["tile"]]
    total = sum(v["slowest"])
    t = max(range(len(v["slowest"])), key=lambda i: v["slowest"][i]) if total else 0
    out["slowest"] = f"t{t} {100 * v['slowest'][t] / total:.0f}%" if total else "-"
    return out


def title(key, k):
    launch, rows, n_kv = key
    return (f"== {launch}, {rows} row(s), n_kv {n_kv}: {k['graphs']} graphs counted"
            f" ({len({s for _, s in k['structures']})} structure(s)), {k['tiles']} tiles")


def show(paths):
    kinds = load(paths)
    for key in sorted(kinds):
        k = kinds[key]
        rows = {name: per_graph(v, k["graphs"]) for name, v in k["names"].items()}
        a = rows.pop("(all)", None)
        print(title(key, k))
        if a:
            print(f"   per graph: {a['n']:.0f} steps; wall {a['wall']:.1f} us = kernel {a['kernel']:.1f} + imbalance "
                  f"{a['imbal']:.1f} + barrier {a['barrier']:.1f}; the host's launch and wait {a['launch']:.1f} us")
        print(f"   {'name':46}{'n':>5}{'wall':>11}{'share':>7}{'kernel':>11}{'imbal.':>9}{'barrier':>9}"
              f"{'tiles min-max':>17}{'slowest':>10}")
        for name, v in sorted(rows.items(), key=lambda x: -x[1]["wall"]):
            share = 100 * v["wall"] / a["wall"] if a and a["wall"] else 0.0
            print(f"   {name:46}{v['n']:5.4g}{v['wall']:11.1f}{share:6.1f}%{v['kernel']:11.1f}{v['imbal']:9.1f}"
                  f"{v['barrier']:9.1f}{min(v['tile']):8.1f}-{max(v['tile']):<8.1f}{v['slowest']:>10}")
        print()


def diff(paths_a, paths_b):
    ka, kb = load(paths_a), load(paths_b)
    for key in sorted(set(ka) & set(kb)):
        a, b = ka[key], kb[key]
        va = {n: per_graph(v, a["graphs"]) for n, v in a["names"].items()}
        vb = {n: per_graph(v, b["graphs"]) for n, v in b["names"].items()}
        print(f"== {key[0]}, {key[1]} row(s), n_kv {key[2]}: graphs counted {a['graphs']} (first set), "
              f"{b['graphs']} (second set); the second set against the first")
        print(f"   {'name':46}{'n':>5}{'wall':>11}{'n':>6}{'wall':>11}{'change':>10}{'kernel':>10}{'imbal.':>9}"
              f"{'barrier':>9}")
        names = list(OrderedDict.fromkeys(list(va) + list(vb)))
        zero = {"n": 0.0, "wall": 0.0, "kernel": 0.0, "imbal": 0.0, "barrier": 0.0}
        rows = []
        for n in names:
            x, y = va.get(n, zero), vb.get(n, zero)
            rows.append((n, x, y, y["wall"] - x["wall"]))
        for n, x, y, d in sorted(rows, key=lambda r: (r[0] == "(all)", -abs(r[3]))):
            rel = f" ({100 * d / x['wall']:+.2f}%)" if n == "(all)" and x["wall"] else ""
            print(f"   {n:46}{x['n']:5.4g}{x['wall']:11.1f}{y['n']:6.4g}{y['wall']:11.1f}{d:+10.1f}"
                  f"{y['kernel'] - x['kernel']:+10.1f}{y['imbal'] - x['imbal']:+9.1f}{y['barrier'] - x['barrier']:+9.1f}{rel}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("show", "diff"):
        sys.exit("usage: op-stats.py show <file>... | op-stats.py diff <file>... -- <file>...")
    if sys.argv[1] == "show":
        show(sys.argv[2:])
    else:
        args = sys.argv[2:]
        if "--" not in args:
            sys.exit("diff: <file>... -- <file>...")
        i = args.index("--")
        diff(args[:i], args[i + 1:])
