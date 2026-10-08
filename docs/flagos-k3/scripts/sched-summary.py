#!/usr/bin/env python3
# Summarise the last graph in a GGML_SCHED_DEBUG=2 log: splits per backend, the ops in each backend's splits,
# and the tensors copied into splits (split inputs).
# Usage: sched-summary.py <log>
import collections
import re
import sys

SPLIT = re.compile(r"## SPLIT #(\d+): (\S+) # (\d+) inputs")
NODE = re.compile(r"node #\s*(\d+) \(\s*([^)]*?)\):\s+(.*?) \(\s*[^)]*\) \[")
INPUT = re.compile(r"\[([^\s\[\]]+) \(")

graphs, cur = [], None
for line in open(sys.argv[1], errors="replace"):
    m = SPLIT.search(line)
    if m:
        if m.group(1) == "0":
            cur = []
            graphs.append(cur)
        if cur is not None:
            cur.append({"backend": m.group(2), "inputs": INPUT.findall(line[m.end():]), "nodes": []})
        continue
    m = NODE.search(line)
    if m and cur:
        cur[-1]["nodes"].append((m.group(2).strip(), re.sub(r"-\d+$", "", m.group(3).strip())))

if not graphs:
    sys.exit("no '## SPLIT' lines: set GGML_SCHED_DEBUG=2 and turn on verbose logging")
g = graphs[-1]
counts = collections.Counter(s["backend"] for s in g)
print(f"{len(graphs)} graphs in the log; last graph has {len(g)} splits: " + ", ".join(f"{b} {n}" for b, n in counts.items()))
by_backend = collections.defaultdict(collections.Counter)
for s in g:
    for op, name in s["nodes"]:
        by_backend[s["backend"]][f"{op} {name}"] += 1
for backend, ops in by_backend.items():
    print(f"-- {backend} splits run {sum(ops.values())} ops; most common:")
    for k, n in ops.most_common(10):
        print(f"   {n:4d} x {k}")
copied = collections.Counter((s["backend"], re.sub(r"-\d+", "", i)) for s in g for i in s["inputs"])
print("-- tensors copied into splits (backend, tensor):")
for (b, t), n in copied.most_common(8):
    print(f"   {n:4d} x into {b}: {t}")
