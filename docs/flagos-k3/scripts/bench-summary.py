#!/usr/bin/env python3
# Summarise llama-bench -o jsonl lines tagged with a "mode" field: mean, median, min-max t/s per test.
import json
import statistics
import sys

rows = {}
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    test = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
    rows.setdefault((test, r.get("n_depth", 0), r.get("mode", "?")), []).append(r["avg_ts"])

print(f"   {'test':6} {'depth':>6} {'mode':14} {'mean':>8} {'median':>8}  min-max   (n)")
for (test, depth, mode), v in sorted(rows.items()):
    print(f"   {test:6} {depth:>6} {mode:14} {statistics.mean(v):8.2f} {statistics.median(v):8.2f}  {min(v):.2f}-{max(v):.2f}  ({len(v)})")
