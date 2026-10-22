#!/usr/bin/env bash
# Fusion measurements on the K3 (docs/flagos-k3/fusion-and-splitting.md §7), one candidate at a time. Every mode runs
# today's build (build A, as structural-experiments.sh) with test-only switches, against the reference (the defaults) in
# the same session, the modes interleaved in every repetition. Experiments:
#   base the baseline: only ref and ops, where every op's time per token or batch is measured on the defaults (§7.1)
#   q6   one quantization for Q, K and V and one for gate and up (§5, #6): FLAGOS_SPACEMIT_QUANTIZE=shared
#   mm   path A for every batch that gives each tile 4 rows (§7.3, not fusion): FLAGOS_SPACEMIT_MM=rows; adds prompts
#        of 32, 64 and 96 tokens (one batch each)
#   pb   path B where the split gives no path A, as ime.cpp (§7.3, not fusion): FLAGOS_SPACEMIT_MM=spacemit; adds
#        prompts of 2-129 tokens (one batch each; with mm these replace mm's three)
#   fit  path B where TCM holds the activations next to a slab, else path A from 32 rows, else path C (§7.3):
#        FLAGOS_SPACEMIT_MM=fit; the same prompts as pb
#   fit-staged
#        fit without path B's pair turns, every tile copying and multiplying its own slabs (§7.3):
#        FLAGOS_SPACEMIT_MM=fit,staged; the same prompts as pb, and prompts of 160-320 tokens (one batch each)
#
# Usage: fusion-experiments.sh prepare                  build A's tools and its provider checks, run once (they compare
#                                                       every candidate with the defaults, byte for byte)
#        fusion-experiments.sh run <model.gguf> [out]   the runs of $EXPS, then the summary (logs in out; default
#                                                       ~/fx/<model name>)
#        fusion-experiments.sh summary <out>            the summary again from the logs
# Env:   EXPS      experiments (default "q6"; ref and ops always run)
#        REPS      llama-bench repetitions per mode (default 5; 0 skips llama-bench)
#        DEPTH     1: also tg32 at depths 0, 1024 and 4096 and pp512 at depth 4096 (default 0)
#        PPL_TEXT  perplexity text for the logits check (default ~/ppl.txt; empty skips it); CHUNKS (default 8)
#        KLD_UB1   1: q6's logits check also with one token per ubatch, the 1-row matmuls (default 1; ~10 min a run);
#                  mm's checks its logits with ubatches of 128 and 64 rows, where the two paths differ, pb's with 64
#                  and 16 rows, fit's and fit-staged's with 128, 64 and 16
#        BUILD_A   the build (default <repo>/build-flagos); TCM_RELEASE (default ~/tcmtest/tcmrelease)
#        TIMEOUT   seconds per run (default 3600); DRY=1 prints the commands instead of running them
# Modes (all build A, llama.cpp's defaults otherwise; with the counters each run writes <log>.ops and E3c's <log>.stats,
# from the same clock reads, when it ends):
#   ref      the defaults
#   ops      FLAGOS_SPACEMIT_TEST_OP_STATS and FLAGOS_SPACEMIT_TEST_STATS: the op counters and E3c's, on the defaults
#   q6       FLAGOS_SPACEMIT_QUANTIZE=shared
#   q6-ops   q6 with the op counters
#   mm       FLAGOS_SPACEMIT_MM=rows
#   mm-ops   mm with the op counters
#   pb       FLAGOS_SPACEMIT_MM=spacemit
#   pb-ops   pb with the op counters
#   fit      FLAGOS_SPACEMIT_MM=fit
#   fit-ops  fit with the op counters
#   fit-staged      FLAGOS_SPACEMIT_MM=fit,staged
#   fit-staged-ops  fit-staged with the op counters
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../.." && pwd)}
BUILD_A=${BUILD_A:-$REPO/build-flagos}
TCM_RELEASE=${TCM_RELEASE:-$HOME/tcmtest/tcmrelease}
EXPS=${EXPS:-q6}
REPS=${REPS:-5}
DEPTH=${DEPTH:-0}
PPL_TEXT=${PPL_TEXT-$HOME/ppl.txt}
CHUNKS=${CHUNKS:-8}
KLD_UB1=${KLD_UB1:-1}
TIMEOUT=${TIMEOUT:-3600}
DRY=${DRY:-}
TOOLS="llama-completion llama-bench llama-perplexity"
PROMPT="The capital of France is"
# --foreground: llama-completion sets the terminal up at start and would otherwise be stopped (structural-experiments.sh)
TO=""
if command -v timeout > /dev/null; then TO="timeout --foreground $TIMEOUT"; fi
# every run sets its own
unset FLAGOS_SPACEMIT_TEST_OP_STATS FLAGOS_SPACEMIT_TEST_STATS FLAGOS_SPACEMIT_QUANTIZE FLAGOS_SPACEMIT_MM \
    FLAGOS_SPACEMIT_TEST_LAUNCH FLAGOS_SPACEMIT_DISABLE FLAGOS_SPACEMIT_ATTN FLAGOS_SPACEMIT_GEMV

mode_env() {
    case $1 in
        q6 | q6-ops) echo "FLAGOS_SPACEMIT_QUANTIZE=shared" ;;
        mm | mm-ops) echo "FLAGOS_SPACEMIT_MM=rows" ;;
        pb | pb-ops) echo "FLAGOS_SPACEMIT_MM=spacemit" ;;
        fit | fit-ops) echo "FLAGOS_SPACEMIT_MM=fit" ;;
        fit-staged | fit-staged-ops) echo "FLAGOS_SPACEMIT_MM=fit,staged" ;;
        *) echo "" ;;
    esac
}
has() { case " $EXPS " in *" $1 "*) return 0 ;; esac; return 1; }
# the board's checkout belongs to another uid, so every git call allows it for itself (structural-experiments.sh)
g() { git -c safe.directory="$REPO" "$@"; }
release() { if [ -x "$TCM_RELEASE" ]; then "$TCM_RELEASE" --apply > /dev/null 2>&1; fi; return 0; }

# run <mode> <log> <tool> [args...]: the tool with the mode's environment, after a TCM release; stdout and stderr to
# <log>.out and <log>.err, the exit code to <log>.rc
run() {
    local mode=$1 log=$2 tool=$3 env rc
    shift 3
    env=$(mode_env "$mode")
    case $mode in
        *ops) env="${env:+$env }FLAGOS_SPACEMIT_TEST_OP_STATS=$log.ops FLAGOS_SPACEMIT_TEST_STATS=$log.stats" ;;
    esac
    if [ -n "$DRY" ]; then
        echo "   [$mode] ${env:+$env }$BUILD_A/bin/$tool $*"
        return 0
    fi
    rm -f "$log.ops" "$log.stats"
    release
    # shellcheck disable=SC2086 # $TO and $env are lists of words
    $TO env $env "$BUILD_A/bin/$tool" "$@" < /dev/null > "$log.out" 2> "$log.err"
    rc=$?
    echo "$rc" > "$log.rc"
    return "$rc"
}

# bench <mode> <tag> [args...]: llama-bench, its rows tagged with the mode and appended to bench.jsonl
bench() {
    local mode=$1 tag=$2
    shift 2
    run "$mode" "$OUT/bench-$mode-$tag" llama-bench -m "$M" -t 8 -fa auto -mmp 0 -r 1 -o jsonl "$@"
    echo "   bench $tag $mode: rc=$?"
    if [ -z "$DRY" ]; then
        sed "s/^{/{\"mode\":\"$mode\",/" "$OUT/bench-$mode-$tag.out" >> "$OUT/bench.jsonl"
    fi
}

prepare() {
    echo "== build A: $BUILD_A"
    [ -z "$DRY" ] || return 0
    if [ ! -f "$BUILD_A/CMakeCache.txt" ]; then
        echo "   not configured: run ggml/src/ggml-flagos/providers/spacemit/tools/spacemit_check.py --build first"
        exit 1
    fi
    # shellcheck disable=SC2086
    cmake --build "$BUILD_A" --target $TOOLS flagos-check-spacemit > "$BUILD_A/fx-build.log" 2>&1
    echo "   build A: rc=$?; $(grep -cE 'error:|FAILED' "$BUILD_A/fx-build.log") error lines ($BUILD_A/fx-build.log)"
    local rc to=""
    if command -v timeout > /dev/null; then to="timeout --foreground 900"; fi
    release
    # shellcheck disable=SC2086
    $to "$BUILD_A/bin/flagos-check-spacemit" --expect-device < /dev/null > "$BUILD_A/fx-check.out" 2> "$BUILD_A/fx-check.err"
    rc=$?
    echo "   flagos-check-spacemit: rc=$rc: $(tail -1 "$BUILD_A/fx-check.out")"
    grep -E "^layer" "$BUILD_A/fx-check.out" | grep -oE "one quantization for[^;]*" | sed 's/^/      /'
    grep -E "FAIL" "$BUILD_A/fx-check.err" | head -5 | sed 's/^/      /'
}

summary() {
    echo "== $(head -1 "$OUT/info.txt" 2> /dev/null); $(sed -n 2p "$OUT/info.txt" 2> /dev/null)"
    echo "   $(sed -n 3p "$OUT/info.txt" 2> /dev/null)"
    echo
    echo "== generated text (greedy, 32 tokens; a candidate changes no output, so all lines the same)"
    for f in "$OUT"/text-*.out; do
        [ -f "$f" ] || continue
        printf '   %-8s rc %-3s %s\n' "$(basename "$f" .out | sed 's/^text-//')" "$(cat "${f%.out}.rc" 2> /dev/null)" \
            "$(tr '\n' ' ' < "$f" | cut -c1-100)"
    done
    if [ -s "$OUT/bench.jsonl" ]; then
        echo
        echo "== llama-bench t/s: mean +- sd over the repetitions (change against ref); the counters' cost: ops against"
        echo "   ref, and each candidate's -ops mode against the candidate"
        python3 -I - "$OUT/bench.jsonl" << 'EOF'
import json, statistics, sys
rows, modes, tests = {}, [], []
for line in open(sys.argv[1]):
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    test = (f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}") + (f"@{r['n_depth']}" if r.get("n_depth") else "")
    if r.get("n_depth") == 0 and r["n_gen"] == 32:
        test += "@0"
    m = r.get("mode", "?")
    modes += [m] if m not in modes else []
    tests += [test] if test not in tests else []
    rows.setdefault((m, test), []).append(r["avg_ts"])
mean = lambda m, t: statistics.mean(rows[(m, t)]) if rows.get((m, t)) else None
def cell(m, t):
    v, ref = rows.get((m, t)), mean("ref", t)
    if not v:
        return f"{'-':>22}"
    avg, sd = statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
    change = f" {100 * (avg / ref - 1):+.2f}%" if ref and m != "ref" else ""
    return f"{f'{avg:.2f}+-{sd:.2f}{change}':>22}"
if len(tests) <= 6:  # a row per mode
    print("   " + f"{'mode':10}" + "".join(f"{t:>22}" for t in tests))
    for m in modes:
        print("   " + f"{m:10}" + "".join(cell(m, t) for t in tests))
else:  # a row per test
    print("   " + f"{'test':10}" + "".join(f"{m:>22}" for m in modes))
    for t in tests:
        print("   " + f"{t:10}" + "".join(cell(m, t) for m in modes))
for m in modes:
    if m.endswith("ops"):
        base = "ref" if m == "ops" else m[: -len("-ops")]
        cost = [f"{t} {100 * (mean(m, t) / mean(base, t) - 1):+.2f}%" for t in tests if mean(m, t) and mean(base, t)]
        print(f"   counters' cost, {m} against {base}: " + ", ".join(cost))
EOF
    fi
    if compgen -G "$OUT/bench-*.stats" > /dev/null; then
        echo
        echo "== one generated token, us, mean over the repetitions, from E3c's counters (as structural-experiments.sh):"
        echo "   outside graph_compute; host work in it; launch, less the mean tile's kernels and barrier waits; the barriers'"
        echo "   own cost; imbalance; the mean tile's kernels; idle, the token less busy. Last, the op counters' totals from"
        echo "   the same runs against E3c's (kernels, busy, barriers' own cost; the op counters leave each kind's first graph out)"
        python3 -I - "$OUT" << 'EOF'
import glob, json, os, re, statistics, sys
kv_class = lambda n: 0 if n < 1024 else 1024 if n < 4096 else 4096  # E3c's KV classes (flagos-spacemit.cpp)
res = {}
for sf in sorted(glob.glob(os.path.join(sys.argv[1], "bench-*.stats"))):
    stem = sf[: -len(".stats")]
    m = re.match(r"bench-(.+)-(std|pp512|ppsmall|ppodd|tg|pp)-\d+$", os.path.basename(stem))
    if not m:
        continue
    kinds = [json.loads(line) for line in open(sf) if line.startswith("{")]
    ops = [json.loads(line) for line in open(stem + ".ops") if line.startswith("{")] if os.path.exists(stem + ".ops") else []
    try:
        rows = [json.loads(line) for line in open(stem + ".out") if line.startswith("{")]
    except OSError:
        rows = []
    for r in rows:
        if not r.get("n_gen") or not r.get("avg_ts"):
            continue
        d = r.get("n_depth") or 0
        kv = kv_class(d)
        k = next((x for x in kinds if x["rows"] == 1 and x["kv"] == kv and x["graphs"] > 0), None)
        if k is None:
            continue
        g, t = k["graphs"], max(k["tiles"], 1)
        token = 1e6 / r["avg_ts"]
        graph, launch = k["graph_us"] / g, k["launch_us"] / g
        kern, bar, busy = k["kernel_us"] / t / g, k["barrier_us"] / t / g, k["busy_us"] / g
        a = [x for x in ops if x["label"] == "(all)" and x["rows"] == 1 and x["graphs"] > 0 and kv_class(x["n_kv"]) == kv]
        og = sum(x["graphs"] for x in a)
        check = None
        if og:
            ok, ob, ow = (sum(x[f] * x["graphs"] for x in a) / og for f in ("kernel_us", "busy_us", "wall_us"))
            own, e3c_own = ow - ob, kern + bar - busy
            check = (100 * (ok / kern - 1), 100 * (ob / busy - 1), 100 * (own / e3c_own - 1) if e3c_own else 0.0)
        test = f"tg{r['n_gen']}@{d}" if r["n_gen"] == 32 else f"tg{r['n_gen']}" + (f"@{d}" if d else "")
        res.setdefault((m.group(1), test), []).append({
            "token": token, "outside": token - graph, "host": graph - launch, "launch": launch - kern - bar,
            "barriers": kern + bar - busy, "imbal.": busy - kern, "kernels": kern, "idle": token - busy,
            "steps": k["steps"] / g, "check": check})
cols = ["token", "outside", "host", "launch", "barriers", "imbal.", "kernels", "idle", "steps"]
print("   " + f"{'mode':8}{'test':11}" + "".join(f"{c:>10}" for c in cols) + f"{'idle %':>8}   op/E3c: kernels, busy, barriers")
order = {"tg128": 0, "tg32@0": 1, "tg32@1024": 2, "tg32@4096": 3}
for key in sorted(res, key=lambda k: (order.get(k[1], 9), k[1], k[0])):
    v = {c: statistics.mean(x[c] for x in res[key]) for c in cols}
    chk = [x["check"] for x in res[key] if x["check"]]
    cmp = "   " + ", ".join(f"{statistics.mean(c[i] for c in chk):+.2f}%" for i in range(3)) if chk else ""
    cells = "".join(f"{v[c]:10.1f}" if c == "steps" else f"{v[c]:10.0f}" for c in cols)
    print("   " + f"{key[0]:8}{key[1]:11}" + cells + f"{100 * v['idle'] / v['token']:8.2f}" + cmp)
EOF
    fi
    if compgen -G "$OUT/bench-ops-*.ops" > /dev/null; then
        echo
        echo "== op counters on the defaults (mode ops): per kind of graph, every name's us per graph"
        python3 -I "$HERE/op-stats.py" show "$OUT"/bench-ops-*.ops | sed 's/^/   /'
        for c in $EXPS; do
            if compgen -G "$OUT/bench-$c-ops-*.ops" > /dev/null; then
                echo "== op counters, $c-ops against ops: name by name, us per graph"
                python3 -I "$HERE/op-stats.py" diff "$OUT"/bench-ops-*.ops -- "$OUT"/bench-"$c"-ops-*.ops | sed 's/^/   /'
            fi
        done
        # the turns' own effect: the same paths with and without them, and the matmuls' even and odd tiles
        if compgen -G "$OUT/bench-fit-ops-*.ops" > /dev/null && compgen -G "$OUT/bench-fit-staged-ops-*.ops" > /dev/null; then
            echo "== op counters, fit-staged-ops against fit-ops: name by name, us per graph"
            python3 -I "$HERE/op-stats.py" diff "$OUT"/bench-fit-ops-*.ops -- "$OUT"/bench-fit-staged-ops-*.ops |
                sed 's/^/   /'
            for c in fit fit-staged; do
                echo "== op counters, $c-ops: the matmuls' even and odd tiles (path B's pair turns), us per graph"
                python3 -I "$HERE/op-stats.py" pairs "$OUT"/bench-"$c"-ops-*.ops | sed 's/^/   /'
            done
        fi
    fi
    if compgen -G "$OUT/kld-*.err" > /dev/null; then
        echo "== logits against ref's (a candidate changes no output: KLD 0, same top p 100%)"
        for f in "$OUT"/ppl-*.err "$OUT"/kld-*.err; do
            [ -f "$f" ] || continue
            printf '   %-14s rc %-3s %s\n' "$(basename "$f" .err)" "$(cat "${f%.err}.rc" 2> /dev/null)" "$(cat "${f%.err}.out" "$f" |
                grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+|Mean +KLD: *-?[0-9.]+|Maximum KLD: *[0-9.]+|Same top p: *[0-9.]+' |
                tr -s ' ' | paste -sd';' -)"
        done
    fi
    echo "logs: $OUT"
}

run_all() {
    local modes kld_modes kld_sets m i s base args
    M=${1:?usage: fusion-experiments.sh run <model.gguf> [out]}
    OUT=${2:-$HOME/fx/$(basename "$M" .gguf)}
    [ -n "$DRY" ] || [ -f "$M" ] || { echo "model not found: $M"; exit 1; }
    [ -n "$DRY" ] || [ -x "$BUILD_A/bin/llama-bench" ] || { echo "not built: $BUILD_A (fusion-experiments.sh prepare)"; exit 1; }
    [ -n "$DRY" ] || mkdir -p "$OUT"
    rm -f "$OUT/bench.jsonl"

    # the logits check's settings: q6's at 512-token batches and one token per ubatch, mm's at 128 and 64 rows, pb's at
    # 64 and 16 rows, fit's and fit-staged's at all three (each set once)
    modes="ref ops" kld_modes="" kld_sets=""
    if has q6; then
        modes+=" q6 q6-ops" kld_modes+=" q6" kld_sets+=" batch"
        if [ "$KLD_UB1" = 1 ]; then kld_sets+=" ub1"; fi
    fi
    if has mm; then modes+=" mm mm-ops" kld_modes+=" mm" kld_sets+=" ub128 ub64"; fi
    if has pb; then modes+=" pb pb-ops" kld_modes+=" pb" kld_sets+=" ub64 ub16"; fi
    if has fit; then modes+=" fit fit-ops" kld_modes+=" fit" kld_sets+=" ub128 ub64 ub16"; fi
    if has fit-staged; then
        modes+=" fit-staged fit-staged-ops" kld_modes+=" fit-staged" kld_sets+=" ub128 ub64 ub16"
    fi
    kld_sets=$(printf '%s\n' $kld_sets | awk '!seen[$0]++' | tr '\n' ' ')
    if [ -z "$DRY" ]; then
        {
            echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
            echo "model: $M; EXPS: $EXPS; REPS: $REPS; DEPTH: $DEPTH; PPL_TEXT: $PPL_TEXT; CHUNKS: $CHUNKS"
            echo "A: $BUILD_A at $(g -C "$REPO" rev-parse --short HEAD)"
        } > "$OUT/info.txt"
    fi

    echo "== generated text"
    for m in ref $kld_modes; do
        run "$m" "$OUT/text-$m" llama-completion -m "$M" -p "$PROMPT" -n 32 --temp 0 -no-cnv -t 8
        echo "   $m: rc=$?"
    done

    if [ "$REPS" -gt 0 ]; then
        echo "== llama-bench, $REPS repetitions, the modes interleaved"
        for i in $(seq "$REPS"); do
            for m in $modes; do
                bench "$m" "std-$i" -p 128 -n 128 -ub 128
                bench "$m" "pp512-$i" -p 512 -n 0
                if has pb || has fit || has fit-staged; then
                    bench "$m" "ppodd-$i" -p 2,4,8,16,32,33,36,37,40,44,48,64,65,96,100,128,129 -n 0
                elif has mm; then
                    bench "$m" "ppsmall-$i" -p 32,64,96 -n 0
                fi
                # where Qwen3-0.6B takes path B under fit and the split takes path A (on Qwen3-4B path A everywhere)
                if has fit-staged; then
                    bench "$m" "pplarge-$i" -p 160,192,256,320 -n 0
                fi
                if [ "$DEPTH" = 1 ]; then
                    bench "$m" "tg-$i" -p 0 -n 32 -d 0,1024,4096
                    bench "$m" "pp-$i" -p 512 -n 0 -d 4096
                fi
            done
        done
    fi

    if [ -n "$PPL_TEXT" ] && [ -n "$kld_modes" ] && { [ -n "$DRY" ] || [ -f "$PPL_TEXT" ]; }; then
        echo "== logits against the reference"
        for s in $kld_sets; do
            case $s in
                batch) args="" ;;
                # one sequence, so every ubatch is one token: the 1-row matmuls (build.md §7, M2e.3 round 2)
                ub1) args="-b 512 -ub 1" ;;
                ub128) args="-ub 128" ;;
                ub64) args="-ub 64" ;;
                ub16) args="-ub 16" ;;
            esac
            base=$OUT/kld-base-ref-$s.bin
            rm -f "$base"
            # shellcheck disable=SC2086 # $args is a list of words
            run ref "$OUT/ppl-ref-$s" llama-perplexity -m "$M" -f "$PPL_TEXT" -c 512 --chunks "$CHUNKS" -t 8 $args \
                --kl-divergence-base "$base"
            echo "   ppl ref $s (writes the base logits): rc=$?"
            for m in $kld_modes; do
                # shellcheck disable=SC2086
                run "$m" "$OUT/kld-$m-$s" llama-perplexity -m "$M" -f "$PPL_TEXT" -c 512 --chunks "$CHUNKS" -t 8 $args \
                    --kl-divergence-base "$base" --kl-divergence
                echo "   kld $m $s: rc=$?"
            done
            rm -f "$base"
        done
    fi

    if [ -z "$DRY" ]; then
        summary
    fi
}

case ${1:-} in
    prepare) prepare ;;
    run) shift; run_all "$@" ;;
    summary) OUT=${2:?usage: fusion-experiments.sh summary <out>}; summary ;;
    *) sed -n '2,44p' "$0"; exit 1 ;;
esac
