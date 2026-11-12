#!/usr/bin/env bash
# The embedding lookup on the K3 (plan.md "Embedding lookup spec"): the settings of FLAGOS_SPACEMIT_EMBD compared, by
# default ggml (ggml's dequantizers on the provider) against rvv (step 2's RVV kernel for Q6_K). Steps, in order:
#   check  flagos-check-spacemit --expect-device (its output in check.out); a failure stops the script with exit 1
#   bench  flagos-check-spacemit --expect-device --bench-embd (bench-embd.out), its lines repeated in the summary
#   llama  llama-bench under each setting, taking turns, the order reversed on even turns; files
#          <tag>-<model>-<setting>-<turn>.csv, the summary one column per setting and each one's change against the first
# Run in tmux after the /dev/tcm check (PROJECT.md section 6), after the runner's build; this script builds nothing and
# does not check /dev/tcm.
#
# usage: embd-experiments.sh run [out]      (default out: ~/embd)
#        embd-experiments.sh summary <out>  (SETTINGS as for the run)
# env:   BUILD (default <repo>/build-flagos), MODELS (default the two Qwen3 files in ~/models), TURNS (3), REPS (3),
#        SETTINGS (default "ggml rvv": each run as FLAGOS_SPACEMIT_EMBD=<value>, non-empty; the first is the base of
#        the changes), STEPS (default "check bench llama"),
#        DEPTH_MODEL (default ~/models/Qwen3-4B-Q4_0.gguf; tg32 and pp512 at depth 4096, empty: skipped),
#        TIMEOUT (seconds per command, 3600), DRY=1 (print the commands only)
set -u

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
BUILD=${BUILD:-$REPO/build-flagos}
MODELS=${MODELS:-"$HOME/models/Qwen3-4B-Q4_0.gguf $HOME/models/Qwen3-0.6B-Q4_0.gguf"}
DEPTH_MODEL=${DEPTH_MODEL-$HOME/models/Qwen3-4B-Q4_0.gguf}
TURNS=${TURNS:-3}
REPS=${REPS:-3}
SETTINGS=${SETTINGS:-"ggml rvv"}
STEPS=${STEPS:-"check bench llama"}
TIMEOUT=${TIMEOUT:-3600}
DRY=${DRY:-}

unset FLAGOS_SPACEMIT_EMBD FLAGOS_SPACEMIT_DISABLE GGML_FLAGOS_DISABLE_OPS GGML_FLAGOS_KERNEL_VARIANTS

step_on() { # step_on <step>: whether STEPS names it
    case " $STEPS " in *" $1 "*) return 0 ;; esac
    return 1
}

run() { # run <out file> <command...>: stdout to the file, stderr beside it, the return code in .rc
    local out=$1
    shift
    echo "$(date +%H:%M:%S) $*"
    [ -n "$DRY" ] && return 0
    timeout "$TIMEOUT" "$@" > "$out" 2> "${out%.*}.err"
    echo $? > "${out%.*}.rc"
}

reversed() { # reversed <words...>
    local out="" w
    for w in "$@"; do out="$w $out"; done
    echo $out
}

bench() { # bench <tag> <model> <turn> <llama-bench arguments...>
    local tag=$1 model=$2 turn=$3 name order s
    shift 3
    name=$(basename "$model" .gguf)
    order=$SETTINGS
    [ $((turn % 2)) -eq 0 ] && order=$(reversed $SETTINGS)
    for s in $order; do
        run "$OUT/$tag-$name-$s-$turn.csv" env FLAGOS_SPACEMIT_EMBD="$s" "$BUILD/bin/llama-bench" -m "$model" "$@"
    done
}

summary() {
    SETTINGS="$SETTINGS" python3 - "$1" <<'PY'
import csv, glob, os, re, sys
out = sys.argv[1]
settings = os.environ["SETTINGS"].split()
alt = "|".join(re.escape(s) for s in sorted(settings, key=len, reverse=True))
cells = {}  # (tag, model, test) -> {setting: [t/s per turn]}
bad = []
for path in sorted(glob.glob(os.path.join(out, "*.csv"))):
    m = re.match(r"(bench|depth)-(.+)-(" + alt + r")-(\d+)\.csv$", os.path.basename(path))
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
for step, name in (("check", "check.out"), ("bench", "bench-embd.out")):
    path = os.path.join(out, name)
    if not os.path.exists(path):
        continue
    rc_path = path[:-4] + ".rc"
    rc = open(rc_path).read().strip() if os.path.exists(rc_path) else "?"
    lines = open(path, errors="replace").read().splitlines()
    if step == "check":
        print(f"check: rc {rc}, {lines[-1] if lines else 'no output'}")
        continue
    print(f"flagos-check-spacemit --bench-embd (rc {rc}):")
    for line in lines:
        if line.startswith("bench    embd"):
            print("   " + line[len("bench    "):])
    if rc != "0":
        bad.append(f"{name}: rc {rc}")
base = settings[0]
print(f"llama-bench t/s by FLAGOS_SPACEMIT_EMBD ({', '.join(settings)}; changes against {base}); best of the turns, "
      f"then each turn")
for (tag, model, test), by in sorted(cells.items()):
    missing = [s for s in settings if not by.get(s)]
    if missing:
        print(f"   {model:18s} {test:12s} incomplete: " + "; ".join(f"{s} {by.get(s, [])}" for s in settings))
        continue
    best = {s: max(by[s]) for s in settings}
    cols = "  ".join(f"{s} {best[s]:8.2f}" + ("" if s == base else f" {(best[s] / best[base] - 1) * 100:+5.1f}%")
                     for s in settings)
    turns = "; ".join(f"{s} {' '.join(f'{v:.2f}' for v in by[s])}" for s in settings)
    print(f"   {model:18s} {test:12s} {cols}   turns: {turns}")
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
            if step_on check || step_on bench; then
                [ -x "$BUILD/bin/flagos-check-spacemit" ] || { echo "no $BUILD/bin/flagos-check-spacemit: run the runner with --build first"; exit 1; }
            fi
            if step_on llama; then
                [ -x "$BUILD/bin/llama-bench" ] || { echo "no $BUILD/bin/llama-bench: run the runner with --build first"; exit 1; }
                for m in $MODELS $DEPTH_MODEL; do
                    [ -f "$m" ] || { echo "no model $m"; exit 1; }
                done
            fi
            { echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
              echo "build: $BUILD at $(git -c safe.directory="$REPO" -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
              echo "steps: $STEPS; settings: $SETTINGS"
              echo "models: $MODELS; depth model: ${DEPTH_MODEL:-none}; turns: $TURNS; reps: $REPS"; } > "$OUT/info.txt"
        fi
        if step_on check; then
            run "$OUT/check.out" "$BUILD/bin/flagos-check-spacemit" --expect-device
            if [ -z "$DRY" ] && [ "$(cat "$OUT/check.rc")" != 0 ]; then
                echo "flagos-check-spacemit failed (rc $(cat "$OUT/check.rc"); $OUT/check.out, check.err): stopping"
                exit 1
            fi
        fi
        if step_on bench; then
            run "$OUT/bench-embd.out" "$BUILD/bin/flagos-check-spacemit" --expect-device --bench-embd
        fi
        if step_on llama; then
            for turn in $(seq 1 "$TURNS"); do
                for model in $MODELS; do
                    bench bench "$model" "$turn" -t 8 -fa 1 -mmp 0 -r "$REPS" -p 128,512 -n 128 -ub 512 -o csv
                done
                if [ -n "$DEPTH_MODEL" ]; then
                    bench depth "$DEPTH_MODEL" "$turn" -t 8 -fa 1 -mmp 0 -r 1 -p 512 -n 32 -d 4096 -o csv
                fi
            done
        fi
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
