#!/usr/bin/env python3
# Compare two llama-eval-callback dumps of the same model and prompt. Tensors are matched by name (and repeat count);
# the error is the largest difference over the printed values, relative to the largest printed value of dump_b.
# Prints the first tensor above the tolerance and which tensor names exist in only one dump.
# Usage: tensor-diff.py <dump_a> <dump_b> [rel_tol, default 0.10]
import math
import re
import sys

HEADER = re.compile(r"^\w+: +(.+?) = \((\w+)\) +(\w+)\(")
NUM = re.compile(r"-?(?:\d+\.\d+|nan|inf)")


def load(path):
    tensors, seen = [], {}
    for line in open(path, errors="replace"):
        m = HEADER.match(line)
        if m:
            name = m.group(1).strip()
            seen[name] = seen.get(name, 0) + 1
            tensors.append({"key": (name, seen[name]), "op": m.group(3), "vals": []})
        elif tensors and "sum =" not in line:
            tensors[-1]["vals"] += [float(v) for v in NUM.findall(line)]
    return tensors


def rel_err(a, b):
    if not a or len(a) != len(b):
        return None
    bad = [x for x in a + b if math.isnan(x) or math.isinf(x)]
    if bad:
        return math.inf
    scale = max(max(abs(x) for x in b), 1e-6)
    return max(abs(x - y) for x, y in zip(a, b)) / scale


a, b = load(sys.argv[1]), load(sys.argv[2])
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
bmap = {t["key"]: t for t in b}
amap = {t["key"]: t for t in a}
print(f"{len(a)} tensors in {sys.argv[1]}, {len(b)} in {sys.argv[2]}")
history = []
for t in a:
    o = bmap.get(t["key"])
    if o is None:
        continue
    err = rel_err(t["vals"], o["vals"])
    history.append((t, err))
    if err is not None and err > tol:
        print(f"first tensor off by more than {tol:.0%} of its magnitude: {t['key'][0]} ({t['op']}), error {err:.3g}")
        for h, e in history[-5:]:
            es = "shape differs" if e is None else f"{e:.3g}"
            print(f"   {h['key'][0]:32} {h['op']:14} {es}")
        break
else:
    print(f"no common tensor off by more than {tol:.0%}")
only_a = [t["key"][0] for t in a if t["key"] not in bmap]
only_b = [t["key"][0] for t in b if t["key"] not in amap]
print(f"only in {sys.argv[1]}: {len(only_a)} {sorted(set(n.rsplit('-', 1)[0] for n in only_a))[:8]}")
print(f"only in {sys.argv[2]}: {len(only_b)} {sorted(set(n.rsplit('-', 1)[0] for n in only_b))[:8]}")
