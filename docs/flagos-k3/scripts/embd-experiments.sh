#!/usr/bin/env bash
# The embedding lookup on the K3 (plan.md "Embedding lookup spec"): llama-bench with the lookup on the provider (the
# default) against FLAGOS_SPACEMIT_EMBD=cpu (the placement before, 2 graph splits per token), taking turns, the order
# reversed on even turns. Run in tmux after the /dev/tcm check (PROJECT.md section 6), after the runner's build; this
# script builds nothing and does not check /dev/tcm.
#
# usage: embd-experiments.sh run [out]      (default out: ~/embd)
#        embd-experiments.sh summary <out>
# env:   BUILD (default <repo>/build-flagos), MODELS (default the two Qwen3 files in ~/models), TURNS (3), REPS (3),
#        DEPTH_MODEL (default ~/models/Qwen3-4B-Q4_0.gguf; tg32 and pp512 at depth 4096, empty: skipped),
#        TIMEOUT (seconds per llama-bench, 3600), DRY=1 (print the commands only)
set -u

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
BUILD=${BUILD:-$REPO/build-flagos}
MODELS=${MODELS:-"$HOME/models/Qwen3-4B-Q4_0.gguf $HOME/models/Qwen3-0.6B-Q4_0.gguf"}
DEPTH_MODEL=${DEPTH_MODEL-$HOME/models/Qwen3-4B-Q4_0.gguf}
TURNS=${TURNS:-3}
REPS=${REPS:-3}
TIMEOUT=${TIMEOUT:-3600}
DRY=${DRY:-}

unset FLAGOS_SPACEMIT_EMBD FLAGOS_SPACEMIT_DISABLE GGML_FLAGOS_DISABLE_OPS GGML_FLAGOS_KERNEL_VARIANTS

run() { # run <out file> <command...>: stdout to the file, stderr beside it, the return code in .rc
    local out=$1
    shift
    echo "$(date +%H:%M:%S) $*"
    [ -n "$DRY" ] && return 0
    timeout "$TIMEOUT" "$@" > "$out" 2> "${out%.*}.err"
    echo $? > "${out%.*}.rc"
}

bench() { # bench <tag> <model> <turn> <llama-bench arguments...>
    local tag=$1 model=$2 turn=$3 name order s
    shift 3
    name=$(basename "$model" .gguf)
    order="provider cpu"
    [ $((turn % 2)) -eq 0 ] && order="cpu provider"
    for s in $order; do
        if [ "$s" = cpu ]; then
            run "$OUT/$tag-$name-$s-$turn.csv" env FLAGOS_SPACEMIT_EMBD=cpu "$BUILD/bin/llama-bench" -m "$model" "$@"
        else
            run "$OUT/$tag-$name-$s-$turn.csv" "$BUILD/bin/llama-bench" -m "$model" "$@"
        fi
    done
}

summary() {
    python3 - "$1" <<'PY'
import csv, glob, os, re, sys
out = sys.argv[1]
cells = {}  # (tag, model, test) -> {setting: [t/s per turn]}
bad = []
for path in sorted(glob.glob(os.path.join(out, "*.csv"))):
    m = re.match(r"(bench|depth)-(.+)-(provider|cpu)-(\d+)\.csv$", os.path.basename(path))
    if not m:
        continue
    tag, model, setting, turn = m.groups()
    rc = open(path[:-4] + ".rc").read().strip() if os.path.exists(path[:-4] + ".rc") else "?"
    rows = list(csv.DictReader(open(path)))
    if rc != "0" or not rows:
        bad.append(f"{os.path.basename(path)}: rc {rc}, {len(rows)} rows")
    for r in rows:
        test = (f"pp{r['n_prompt']}" if r["n_gen"] == "0" else f"tg{r['n_gen']}") + (
            f" d{r['n_depth']}" if r.get("n_depth", "0") != "0" else "")
        cells.setdefault((tag, model, test), {}).setdefault(setting, []).append(float(r["avg_ts"]))
print(f"llama-bench t/s, the lookup on the provider against FLAGOS_SPACEMIT_EMBD=cpu; best of the turns and each turn")
for (tag, model, test), by in sorted(cells.items()):
    p, c = by.get("provider", []), by.get("cpu", [])
    if not p or not c:
        print(f"   {model:18s} {test:12s} incomplete: provider {p}, cpu {c}")
        continue
    change = (max(p) / max(c) - 1) * 100
    print(f"   {model:18s} {test:12s} provider {max(p):8.2f}  cpu {max(c):8.2f}  {change:+5.1f}%   turns: provider "
          f"{' '.join(f'{v:.2f}' for v in p)}; cpu {' '.join(f'{v:.2f}' for v in c)}")
for b in bad:
    print(f"   BAD {b}")
print(f"logs: {out}")
PY
}

case "${1:-}" in
    run)
        OUT=${2:-$HOME/embd}
        [ -n "$DRY" ] || mkdir -p "$OUT"
        if [ -z "$DRY" ]; then
            [ -x "$BUILD/bin/llama-bench" ] || { echo "no $BUILD/bin/llama-bench: run the runner with --build first"; exit 1; }
            for m in $MODELS $DEPTH_MODEL; do
                [ -f "$m" ] || { echo "no model $m"; exit 1; }
            done
            { echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
              echo "build: $BUILD at $(git -c safe.directory="$REPO" -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
              echo "models: $MODELS; depth model: ${DEPTH_MODEL:-none}; turns: $TURNS; reps: $REPS"; } > "$OUT/info.txt"
        fi
        for turn in $(seq 1 "$TURNS"); do
            for model in $MODELS; do
                bench bench "$model" "$turn" -t 8 -fa 1 -mmp 0 -r "$REPS" -p 128,512 -n 128 -ub 512 -o csv
            done
            if [ -n "$DEPTH_MODEL" ]; then
                bench depth "$DEPTH_MODEL" "$turn" -t 8 -fa 1 -mmp 0 -r 1 -p 512 -n 32 -d 4096 -o csv
            fi
        done
        [ -n "$DRY" ] || summary "$OUT"
        ;;
    summary)
        [ -n "${2:-}" ] || { echo "usage: $0 summary <out>"; exit 2; }
        summary "$2"
        ;;
    *)
        echo "usage: $0 run [out] | summary <out>"
        exit 2
        ;;
esac
