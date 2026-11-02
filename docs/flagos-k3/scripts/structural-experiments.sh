#!/usr/bin/env bash
# Structural experiments on the K3 (docs/flagos-k3/structural-experiments.md) on today's build: E3a, one launch per
# step; E3c, test-only counters of where a token's time goes; the plan cache against the full path (plan.md "S1 step 3c
# spec"). Every mode runs against the reference (today's build, defaults) in the same session, the modes interleaved in
# every repetition. E1, E2, E2-hp, E2-neg, E3b, E4 and R12-hw ran on builds and switches removed 2026-11-02 once their
# decisions were made; their scripts are in git history (cad674fda).
#
# Usage: structural-experiments.sh prepare                  build A's tools (today's build-flagos/) and its provider
#                                                           checks, 5 times with e3 (they compare E3a with one launch,
#                                                           byte for byte)
#        structural-experiments.sh run <model.gguf> [out]   the runs of $EXPS, then the summary (logs in out; default
#                                                           ~/sx/<model name>)
#        structural-experiments.sh summary <out>            the summary again from the logs
# Env:   EXPS      experiments (default "e3c"; also e3 and plan); the reference always runs
#        REPS      llama-bench repetitions per mode (default 3; 0 skips llama-bench)
#        DEPTH     1: also tg32 at depths 0, 1024 and 4096 and pp512 at depth 4096 (default 1)
#        PPL_TEXT  perplexity text (default ~/ppl.txt; empty skips perplexity and KLD); CHUNKS (default 8)
#        KLD_UB1   1: also the KLD with one token per ubatch (-ub 1 -b 512) for plan-n (default 1; ~10 min a run)
#        BUILD_A   today's build (default <repo>/build-flagos); TCM_RELEASE (default ~/tcmtest/tcmrelease)
#        TIMEOUT   seconds per run (default 3600); DRY=1 prints the commands instead of running them
# Modes (build A, today's; -fa is llama.cpp's default, auto):
#   ref           today's state (the plan cache on, its default since plan.md "S1 step 3c spec")
#   e3            FLAGOS_SPACEMIT_TEST_LAUNCH=per-step   E3a: one launch per step, the host waiting for each
#   e3c           FLAGOS_SPACEMIT_TEST_STATS=<log>.stats  E3c: the counters, written when each run ends
#   e3c-a         as e3c, with FLAGOS_SPACEMIT_TEST_LAUNCH=per-step: E3c on E3a, what a launch costs inside a run
#   plan-n        FLAGOS_SPACEMIT_PLAN=none: no plan cache, the contract check and the expansion on every graph;
#                 ref against plan-n is the default's speed gain
#   plan-nc       as e3c, with FLAGOS_SPACEMIT_PLAN=none: plan-n with the counters (e3c against plan-nc: the host
#                 columns)
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../.." && pwd)}
BUILD_A=${BUILD_A:-$REPO/build-flagos}
TCM_RELEASE=${TCM_RELEASE:-$HOME/tcmtest/tcmrelease}
EXPS=${EXPS:-e3c}
REPS=${REPS:-3}
DEPTH=${DEPTH:-1}
PPL_TEXT=${PPL_TEXT-$HOME/ppl.txt}
CHUNKS=${CHUNKS:-8}
KLD_UB1=${KLD_UB1:-1}
TIMEOUT=${TIMEOUT:-3600}
DRY=${DRY:-}
TOOLS="llama-completion llama-bench llama-perplexity"
PROMPT="The capital of France is"
# --foreground: without it timeout runs the command outside the terminal's foreground, and llama-completion, which sets
# the terminal up at start (console::init), is stopped by the kernel (SIGTTOU) and never ends
TO=""
if command -v timeout > /dev/null; then TO="timeout --foreground $TIMEOUT"; fi
# every run sets its own
unset FLAGOS_SPACEMIT_TEST_DEVICE_TYPE FLAGOS_SPACEMIT_TEST_LAUNCH FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES \
    FLAGOS_SPACEMIT_TEST_STATS FLAGOS_SPACEMIT_DISABLE FLAGOS_SPACEMIT_ATTN FLAGOS_SPACEMIT_PLAN FLAGOS_SPACEMIT_QUANTIZE

mode_env() {
    case $1 in
        e3 | e3c-a) echo "FLAGOS_SPACEMIT_TEST_LAUNCH=per-step" ;;
        plan-n | plan-nc) echo "FLAGOS_SPACEMIT_PLAN=none" ;;
        *) echo "" ;;
    esac
}

has() { case " $EXPS " in *" $1 "*) return 0 ;; esac; return 1; }
# the board's checkout belongs to another uid (copied from the Mac), so git refuses it ("dubious ownership"); as with
# `git -c safe.directory=$PWD pull` there, every call allows the checkouts for itself and changes no git settings
g() { git -c safe.directory="$REPO" "$@"; }
release() { if [ -x "$TCM_RELEASE" ]; then "$TCM_RELEASE" --apply > /dev/null 2>&1; fi; return 0; }

# run <mode> <log> <tool> [args...]: the tool of the mode's build with the mode's environment, after a TCM release;
# stdout and stderr to <log>.out and <log>.err, the exit code to <log>.rc
run() {
    local mode=$1 log=$2 tool=$3 bin env rc
    shift 3
    bin=$BUILD_A/bin/$tool
    env=$(mode_env "$mode")
    # E3c: every run writes its counters next to its log, when it ends
    case $mode in e3c | e3c-a | plan-nc) env="${env:+$env }FLAGOS_SPACEMIT_TEST_STATS=$log.stats" ;; esac
    if [ -n "$DRY" ]; then
        echo "   [$mode] ${env:+$env }$bin $*"
        return 0
    fi
    rm -f "$log.stats"
    release
    # no terminal input: nothing may wait for it or set it up (as spacemit_check.py runs its tools)
    # shellcheck disable=SC2086 # $TO and $env are lists of words
    $TO env $env "$bin" "$@" < /dev/null > "$log.out" 2> "$log.err"
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

prepare_a() {
    echo "== build A: $BUILD_A"
    [ -z "$DRY" ] || return 0
    if [ ! -f "$BUILD_A/CMakeCache.txt" ]; then
        echo "   not configured: run ggml/src/ggml-flagos/providers/spacemit/tools/spacemit_check.py --build first"
        exit 1
    fi
    # shellcheck disable=SC2086
    cmake --build "$BUILD_A" --target $TOOLS flagos-check-spacemit > "$BUILD_A/sx-build.log" 2>&1
    echo "   build A: rc=$?; $(grep -cE 'error:|FAILED' "$BUILD_A/sx-build.log") error lines ($BUILD_A/sx-build.log)"
    # the checks include the layer with the KV cache in the provider's buffer and E3a's comparisons of one launch per
    # step with one per layer, byte for byte; with e3 they run 5 times. A hang ends after 15 minutes
    local n_check=1 i rc to=""
    if has e3; then n_check=5; fi
    if command -v timeout > /dev/null; then to="timeout --foreground 900"; fi
    for i in $(seq "$n_check"); do
        release
        # shellcheck disable=SC2086
        $to "$BUILD_A/bin/flagos-check-spacemit" --expect-device < /dev/null > "$BUILD_A/sx-check.out" \
            2> "$BUILD_A/sx-check.err"
        rc=$?
        echo "   flagos-check-spacemit ($i/$n_check): rc=$rc: $(tail -1 "$BUILD_A/sx-check.out")"
        grep -E "^layer|^attn     split" "$BUILD_A/sx-check.out" |
            grep -oE "KV cache in our buffer[^;]*|one launch per step[^;]*" | sed 's/^/      /'
        grep -E "FAIL" "$BUILD_A/sx-check.err" | head -5 | sed 's/^/      /'
    done
}

prepare() {
    prepare_a
}

summary() {
    local m f rc splits fa kv off
    echo
    echo "== placement (llama-completion; graph splits for the batch / 1-token graphs; flash attention under -fa auto)"
    printf '   %-14s %-4s %-9s %-10s %-11s %s\n' mode rc splits "flash att." offloaded "KV cache"
    for f in "$OUT"/load-*.err; do
        [ -f "$f" ] || continue
        m=$(basename "$f" .err)
        m=${m#load-}
        rc=$(cat "${f%.err}.rc" 2> /dev/null)
        # one number when both graphs take the same count (src/llama-context.cpp, "graph splits")
        splits=$(grep -oE 'graph splits = [0-9]+( \(with bs=[0-9]+\), [0-9]+ \(with bs=1\))?' "$f" | tail -1 |
            sed -E 's/graph splits = ([0-9]+) \(with bs=[0-9]+\), ([0-9]+) \(with bs=1\)/\1 \/ \2/; s/graph splits = ([0-9]+)$/\1 \/ \1/')
        fa=$(grep -oE 'Flash Attention (enabled|not supported, set to disabled)' "$f" | tail -1 | awk '{print $NF}')
        off=$(grep -oE 'offloaded [0-9]+/[0-9]+ layers' "$f" | tail -1 | awk '{print $2}')
        kv=$(grep -oE '[A-Za-z0-9:_]+ KV buffer size = +[0-9.]+ MiB' "$f" | sed -E 's/ KV buffer size = +/ /' | sort -u |
            paste -sd, -)
        printf '   %-14s %-4s %-9s %-10s %-11s %s\n' "$m" "${rc:-?}" "${splits:--}" "${fa:-set}" "${off:--}" "${kv:--}"
    done
    echo
    echo "== generated text"
    for f in "$OUT"/load-*.out; do
        [ -f "$f" ] || continue
        m=$(basename "$f" .out)
        printf '   %-14s %s\n' "${m#load-}" "$(tr '\n' ' ' < "$f" | cut -c1-100)"
    done
    if [ -s "$OUT/bench.jsonl" ]; then
        echo
        echo "== llama-bench t/s: mean +- sd over the repetitions (change against ref)"
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
print("   " + f"{'mode':14}" + "".join(f"{t:>22}" for t in tests))
for m in modes:
    cells = []
    for t in tests:
        v = rows.get((m, t))
        ref = rows.get(("ref", t))
        if not v:
            cells.append(f"{'-':>22}")
            continue
        mean, sd = statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
        change = f" {100 * (mean / statistics.mean(ref) - 1):+.1f}%" if ref and m != "ref" else ""
        cells.append(f"{f'{mean:.2f}+-{sd:.2f}{change}':>22}")
    print("   " + f"{m:14}" + "".join(cells))
EOF
    fi
    if compgen -G "$OUT/bench-e3c*.stats" > /dev/null; then
        echo
        echo "== E3c: one generated token, us, mean over the repetitions. busy: each step's slowest tile, summed; idle (the"
        echo "   deck's measure, R12): the token less busy = outside (llama.cpp, the CPU split, sampling) + host (graph_compute"
        echo "   outside the launches) + launch (the launches less the mean tile's kernels and barrier waits) + barriers (their"
        echo "   own cost); imbal.: busy less the mean tile's kernels. The plan cache (FLAGOS_SPACEMIT_PLAN; on in every build-A"
        echo "   mode since step 3c (e3c by default), off in plan-nc; logs from before step 3c ran e3c without it): hits by uid"
        echo "   (p.uid) and misses (p.miss) per token; lookup: the time from the top of graph_compute to the step list in hand"
        echo "   (on a miss and under plan-nc the contract check and the expansion);"
        echo "   key: from after the check, on a miss the store (a hit skips the check, so its key is its lookup; a miss's probe"
        echo "   lies before the check, in lookup only); plan-nc counts no hits or misses, its key about 0"
        python3 -I - "$OUT" << 'EOF'
import glob, json, os, re, statistics, sys
res = {}
for sf in sorted(glob.glob(os.path.join(sys.argv[1], "bench-*.stats"))):
    stem = sf[: -len(".stats")]
    m = re.match(r"bench-(.+)-(std|tg|pp)-\d+$", os.path.basename(stem))
    if not m:
        continue
    kinds = [json.loads(line) for line in open(sf) if line.startswith("{")]
    try:
        rows = [json.loads(line) for line in open(stem + ".out") if line.startswith("{")]
    except OSError:
        rows = []
    for r in rows:
        if not r.get("n_gen") or not r.get("avg_ts"):
            continue
        d = r.get("n_depth") or 0
        kv = 0 if d < 1024 else 1024 if d < 4096 else 4096  # the counters' KV classes (flagos-spacemit.cpp)
        k = next((x for x in kinds if x["rows"] == 1 and x["kv"] == kv and x["graphs"] > 0), None)
        if k is None:
            continue
        g, t = k["graphs"], max(k["tiles"], 1)
        token = 1e6 / r["avg_ts"]
        graph, launch = k["graph_us"] / g, k["launch_us"] / g
        kern, bar, busy = k["kernel_us"] / t / g, k["barrier_us"] / t / g, k["busy_us"] / g
        test = f"tg{r['n_gen']}@{d}" if r["n_gen"] == 32 else f"tg{r['n_gen']}" + (f"@{d}" if d else "")
        res.setdefault((m.group(1), test), []).append({
            "token": token, "outside": token - graph, "host": graph - launch, "launch": launch - kern - bar,
            "barriers": kern + bar - busy, "imbal.": busy - kern, "kernels": kern, "idle": token - busy,
            "steps": k["steps"] / g, "launches": k["launches"] / g,
            "p.uid": k.get("plan_hits_uid", 0) / g, "p.miss": k.get("plan_misses", 0) / g,
            "lookup": k.get("plan_lookup_us", 0.0) / g, "key": k.get("plan_key_us", 0.0) / g})
cols = ["token", "outside", "host", "launch", "barriers", "imbal.", "kernels", "idle", "steps", "launches",
        "p.uid", "p.miss", "lookup", "key"]
fractions = ("steps", "launches", "p.uid", "p.miss", "lookup", "key")
print("   " + f"{'mode':8}{'test':11}" + "".join(f"{c:>10}" for c in cols) + f"{'idle %':>8}{'launch/each':>12}")
order = {"tg128": 0, "tg32@0": 1, "tg32@1024": 2, "tg32@4096": 3}
mean = {}
for key in sorted(res, key=lambda k: (order.get(k[1], 9), k[1], k[0])):
    v = mean[key] = {c: statistics.mean(x[c] for x in res[key]) for c in cols}
    each = v["launch"] / v["launches"] if v["launches"] else 0.0
    cells = "".join(f"{v[c]:10.2f}" if c in fractions[2:] else f"{v[c]:10.1f}" if c in fractions else f"{v[c]:10.0f}"
                    for c in cols)
    print("   " + f"{key[0]:8}{key[1]:11}" + cells + f"{100 * v['idle'] / v['token']:8.2f}{each:12.2f}")
for (mode, test), v in mean.items():
    one = mean.get(("e3c", test))
    if mode == "e3c-a" and one and one["steps"]:
        print(f"   {test}: E3a's cost per step (e3c-a less e3c, over e3c's {one['steps']:.0f} steps): "
              f"{(v['token'] - one['token']) / one['steps']:.2f} us")
EOF
    fi
    if compgen -G "$OUT/ppl-*.err" > /dev/null || compgen -G "$OUT/kld-*.err" > /dev/null; then
        echo
        echo "== perplexity, and KLD against the reference's logits (identical logits: KLD 0, same top p 100%)"
        for f in "$OUT"/ppl-*.err "$OUT"/kld-*.err; do
            [ -f "$f" ] || continue
            m=$(basename "$f" .err)
            printf '   %-18s rc %-3s %s\n' "$m" "$(cat "${f%.err}.rc" 2> /dev/null)" "$(cat "${f%.err}.out" "$f" |
                grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+|Mean PPL\(Q\) *: *[0-9.]+|Mean +KLD: *-?[0-9.]+|Maximum KLD: *[0-9.]+|Same top p: *[0-9.]+' |
                tr -s ' ' | paste -sd';' -)"
        done
    fi
    echo "logs: $OUT"
}

run_all() {
    local place bench_modes depth_modes kld_modes kld1_modes m i base
    M=${1:?usage: structural-experiments.sh run <model.gguf> [out]}
    OUT=${2:-$HOME/sx/$(basename "$M" .gguf)}
    [ -n "$DRY" ] || [ -f "$M" ] || { echo "model not found: $M"; exit 1; }
    [ -n "$DRY" ] || [ -x "$BUILD_A/bin/llama-bench" ] || { echo "not built: $BUILD_A (structural-experiments.sh prepare)"; exit 1; }
    [ -n "$DRY" ] || mkdir -p "$OUT"
    rm -f "$OUT/bench.jsonl"

    place="ref" bench_modes="ref" depth_modes="ref" kld_modes="" kld1_modes=""
    if has e3; then
        place+=" e3" bench_modes+=" e3" depth_modes+=" e3" kld_modes+=" e3"
    fi
    # the counters change no output (the checks compare them byte for byte), so no KLD
    if has e3c; then
        bench_modes+=" e3c e3c-a" depth_modes+=" e3c e3c-a"
    fi
    # the plan cache, the default since plan.md "S1 step 3c spec" (ref), against none: plan-n without the counters
    # (ref against plan-n: the default's speed gain), e3c against plan-nc with them (the host columns); plan-n's
    # logits must read identical to the reference's (step 3c §7)
    if has plan; then
        bench_modes+=" plan-n"
        case " $bench_modes " in *" e3c "*) ;; *) bench_modes+=" e3c" ;; esac
        bench_modes+=" plan-nc" depth_modes+=" plan-n" kld_modes+=" plan-n" kld1_modes+=" plan-n"
    fi
    if [ -z "$DRY" ]; then
        {
            echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
            echo "model: $M; EXPS: $EXPS; REPS: $REPS; DEPTH: $DEPTH; PPL_TEXT: $PPL_TEXT; CHUNKS: $CHUNKS"
            echo "A: $BUILD_A at $(g -C "$REPO" rev-parse --short HEAD)"
        } > "$OUT/info.txt"
    fi

    echo "== placement and text"
    for m in $place; do
        run "$m" "$OUT/load-$m" llama-completion -m "$M" -p "$PROMPT" -n 32 --temp 0 -no-cnv -t 8 -lv 4
        echo "   $m: rc=$?"
    done

    if [ "$REPS" -gt 0 ]; then
        echo "== llama-bench, $REPS repetitions, the modes interleaved"
        for i in $(seq "$REPS"); do
            for m in $bench_modes; do
                bench "$m" "std-$i" -p 128 -n 128 -ub 128
            done
            if [ "$DEPTH" = 1 ]; then
                for m in $depth_modes; do
                    bench "$m" "tg-$i" -p 0 -n 32 -d 0,1024,4096
                    bench "$m" "pp-$i" -p 512 -n 0 -d 4096
                done
            fi
        done
    fi

    if [ -n "$PPL_TEXT" ] && [ -n "$kld_modes$kld1_modes" ] && { [ -n "$DRY" ] || [ -f "$PPL_TEXT" ]; }; then
        echo "== perplexity and KLD against the reference"
        base=$OUT/kld-base-ref.bin
        rm -f "$base"
        run ref "$OUT/ppl-ref" llama-perplexity -m "$M" -f "$PPL_TEXT" -c 512 --chunks "$CHUNKS" -t 8 \
            --kl-divergence-base "$base"
        echo "   ppl ref (writes the base logits): rc=$?"
        for m in $kld_modes; do
            run "$m" "$OUT/kld-$m" llama-perplexity -m "$M" -f "$PPL_TEXT" -c 512 --chunks "$CHUNKS" -t 8 \
                --kl-divergence-base "$base" --kl-divergence
            echo "   kld $m: rc=$?"
        done
        if [ "$KLD_UB1" = 1 ] && [ -n "$kld1_modes" ]; then
            # one sequence, so every ubatch is one token, as in generation (build.md §6)
            run ref "$OUT/ppl-ref-ub1" llama-perplexity -m "$M" -f "$PPL_TEXT" -c 512 -b 512 -ub 1 --chunks "$CHUNKS" \
                -t 8 --kl-divergence-base "$base.ub1"
            echo "   ppl ref -ub 1 (writes the base logits): rc=$?"
            for m in $kld1_modes; do
                run "$m" "$OUT/kld-$m-ub1" llama-perplexity -m "$M" -f "$PPL_TEXT" -c 512 -b 512 -ub 1 \
                    --chunks "$CHUNKS" -t 8 --kl-divergence-base "$base.ub1" --kl-divergence
                echo "   kld $m -ub 1: rc=$?"
            done
        fi
        rm -f "$base" "$base.ub1"
    fi

    if [ -z "$DRY" ]; then
        summary
    fi
}

case ${1:-} in
    prepare) prepare ;;
    run) shift; run_all "$@" ;;
    summary) OUT=${2:?usage: structural-experiments.sh summary <out>}; summary ;;
    *) sed -n '2,29p' "$0"; exit 1 ;;
esac
