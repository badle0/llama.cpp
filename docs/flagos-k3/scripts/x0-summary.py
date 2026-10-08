#!/usr/bin/env python3
# Summarise test-backend-ops logs (one op per file): pass/fail counts, failures grouped by type and flag
# parameters with the largest error, and the abort message when a run died.
# Usage: x0-summary.py ~/x0-*.log
import re
import sys

CHATTER = re.compile(r"ggml-spacemit: \S+ graph-compute n_nodes \d+")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
DESC = re.compile(r"\b([A-Z][A-Z0-9_]*)\(([^()]*)\)")
ERR = re.compile(r"\b(NMSE|ERR) = ([0-9.eE+-]+|nan|inf)")
ABORT = re.compile(r"\S+\.(?:c|cpp|h|hpp):\d+: .*")
FLAG_KEYS = {"mask", "max_bias", "sinks", "inplace", "bias", "v", "o", "mode", "permute", "permute_src", "permute_dst"}


def key_of(params):
    keep = []
    for kv in re.split(r",(?![^\[]*\])", params):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        if re.search(r"[a-z]", v) or k in FLAG_KEYS:
            keep.append(f"{k}={v}")
    return ",".join(keep)


for path in sys.argv[1:]:
    cases, cur, abort_msg = [], None, None
    for raw in open(path, errors="replace"):
        line = CHATTER.sub("", ANSI.sub("", raw))
        m = ABORT.search(line)
        if m and abort_msg is None and "backtrace" not in line:
            abort_msg = m.group(0).strip()
        if "what():" in line and abort_msg is None:
            abort_msg = line.strip()
        for d in DESC.finditer(line):
            if "=" in d.group(2):
                cur = [d.group(1), d.group(2), None]
        e = ERR.search(line)
        if e and cur is not None:
            cur[2] = e.group(2)
        status = "FAIL" if "FAIL" in line else ("OK" if re.search(r"\bOK\b", line) else None)
        if "not supported" in line:
            cur = None
        elif status and cur is not None:
            cases.append((cur[0], cur[1], status, cur[2]))
            cur = None
    fails = [c for c in cases if c[2] == "FAIL"]
    name = cases[0][0] if cases else path
    print(f"== {name}: {len(cases) - len(fails)} OK, {len(fails)} FAIL ({path})")
    if cur is not None:
        print(f"   run ended inside: {cur[0]}({cur[1][:150]})")
    if abort_msg:
        print(f"   abort: {abort_msg[:200]}")
    groups = {}
    for op, params, _, err in fails:
        g = groups.setdefault(key_of(params), [0, 0.0, params])
        g[0] += 1
        try:
            g[1] = max(g[1], float(err))
        except (TypeError, ValueError):
            g[1] = float("nan") if err else g[1]
    for k, (n, err, example) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:5]:
        print(f"   {n:4d} x  max err {err:.3g}  {k[:110]}")
