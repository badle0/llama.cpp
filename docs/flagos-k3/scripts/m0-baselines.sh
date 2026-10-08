#!/usr/bin/env bash
# M0.5 and M0.6 (plan.md): speed and perplexity of every way the K3 runs a model today, runs interleaved.
#   cpu            our fork, plain CPU backend on the X100 cores          ($REPO/build)
#   ime            our fork, upstream IME path + TCM on the A100 cores    ($REPO/build-ime)
#   spacemit       ggml-spacemit as shipped: ACCEL, own buffers only      ($SPACEMIT_BIN)
#   spacemit-best  ggml-spacemit reading CPU buffers, norm pinning off    (needs patch-spacemit.py and the LLAMA_NO_NORM_PIN patch)
# Usage: m0-baselines.sh <model.gguf> [out_dir]
# Env:   MODES (default: all four), REPS (5), DEPTHS (0; "0,1024" adds a KV-depth test, slow for cpu),
#        PPL_TEXT (default ~/ppl.txt; empty skips perplexity), CHUNKS (8), REPO (~/llama.cpp-flagos),
#        SPACEMIT_BIN (~/llama.cpp-spacemit/build/bin), SPERT_DIR (~/spine-runtime)
set -uo pipefail

M=${1:?usage: m0-baselines.sh <model.gguf> [out_dir]}
OUT=${2:-$HOME/m0/$(basename "$M" .gguf)}
REPO=${REPO:-$HOME/llama.cpp-flagos}
SPACEMIT_BIN=${SPACEMIT_BIN:-$HOME/llama.cpp-spacemit/build/bin}
SPERT_DIR=${SPERT_DIR:-$HOME/spine-runtime}
MODES=${MODES:-cpu ime spacemit spacemit-best}
REPS=${REPS:-5}
DEPTHS=${DEPTHS:-0}
PPL_TEXT=${PPL_TEXT-$HOME/ppl.txt}
CHUNKS=${CHUNKS:-8}
HERE=$(cd "$(dirname "$0")" && pwd)

step() { echo "[$(date +%T)] $*"; }
release() { if [ -x "$HOME/tcmtest/tcmrelease" ]; then "$HOME/tcmtest/tcmrelease" --apply > /dev/null; fi; }

# bin <mode> <tool>: path of a tool in the build that a mode uses
bin() {
    case $1 in
        cpu) echo "$REPO/build/bin/$2" ;;
        ime) echo "$REPO/build-ime/bin/$2" ;;
        spacemit | spacemit-best) echo "$SPACEMIT_BIN/$2" ;;
        *) echo "unknown-mode-$1" ;;
    esac
}

# run <mode> <tool> args...: release stale TCM blocks, give the mode exactly its environment, no keyboard input
run() {
    local mode=$1 tool=$2
    shift 2
    release
    local clean=(-u GGML_SPACEMIT_DEVICE_TYPE -u GGML_SPACEMIT_ACCEPT_HOST -u LLAMA_NO_NORM_PIN -u SPACEMIT_DISABLE_TCM)
    case $mode in
        spacemit) env "${clean[@]}" LD_LIBRARY_PATH="$SPERT_DIR/lib" "$(bin "$mode" "$tool")" "$@" < /dev/null ;;
        spacemit-best) env "${clean[@]}" LD_LIBRARY_PATH="$SPERT_DIR/lib" GGML_SPACEMIT_ACCEPT_HOST=1 LLAMA_NO_NORM_PIN=1 \
            "$(bin "$mode" "$tool")" "$@" < /dev/null ;;
        *) env "${clean[@]}" "$(bin "$mode" "$tool")" "$@" < /dev/null ;;
    esac
}

# preflight: every tool exists, and spacemit-best really has both test switches compiled in
fail=0
tools="llama-bench"
if [ -n "$PPL_TEXT" ]; then
    tools="$tools llama-perplexity"
    if [ ! -s "$PPL_TEXT" ]; then echo "missing perplexity text: $PPL_TEXT"; fail=1; fi
fi
for mode in $MODES; do
    for tool in $tools; do
        if [ ! -x "$(bin "$mode" "$tool")" ]; then echo "missing: $(bin "$mode" "$tool") (mode $mode)"; fail=1; fi
    done
    if [ "$mode" = spacemit-best ]; then
        for sw in GGML_SPACEMIT_ACCEPT_HOST LLAMA_NO_NORM_PIN; do
            if ! grep -qa "$sw" "$SPACEMIT_BIN"/lib*.so* 2> /dev/null; then echo "ggml-spacemit build lacks the $sw switch"; fail=1; fi
        done
    fi
done
if [ ! -s "$M" ]; then echo "missing model: $M"; fail=1; fi
if [ "$fail" -ne 0 ]; then exit 1; fi

mkdir -p "$OUT"
rm -f "$OUT/bench.jsonl"
step "model $(basename "$M"), modes: $MODES, $REPS reps, depths $DEPTHS, output $OUT"

for i in $(seq "$REPS"); do
    for mode in $MODES; do
        step "bench $mode, rep $i/$REPS"
        run "$mode" llama-bench -m "$M" -t 8 -p 128 -n 128 -d "$DEPTHS" -ub 128 -fa 1 -mmp 0 -r 1 -o jsonl 2> "$OUT/bench-$mode-$i.log" \
            | sed "s/^{/{\"mode\":\"$mode\",/" >> "$OUT/bench.jsonl"
        rc=${PIPESTATUS[0]}
        if [ "$rc" -ne 0 ]; then echo "   failed (rc=$rc), see $OUT/bench-$mode-$i.log"; fi
    done
done

if [ -n "$PPL_TEXT" ]; then
    for mode in $MODES; do
        step "perplexity $mode"
        run "$mode" llama-perplexity -m "$M" -f "$PPL_TEXT" -c 512 --chunks "$CHUNKS" -t 8 -fa on -lv 4 > "$OUT/ppl-$mode.log" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ]; then echo "   failed (rc=$rc), see $OUT/ppl-$mode.log"; fi
    done
fi

summary() {
    echo "== $(basename "$M"): llama-bench t/s ($REPS interleaved runs, -t 8 -ub 128 -fa 1)"
    python3 "$HERE/bench-summary.py" "$OUT/bench.jsonl"
    if [ -n "$PPL_TEXT" ]; then
        echo
        echo "== perplexity ($CHUNKS chunks of 512 tokens, $(basename "$PPL_TEXT"), -fa on)"
        for mode in $MODES; do
            printf '   %-14s %s\n' "$mode" "$(grep -oE 'PPL = [0-9.]+ \+/- [0-9.]+' "$OUT/ppl-$mode.log" || echo 'no result')"
        done
        echo
        echo "== where the weights went (perplexity load logs)"
        for mode in $MODES; do
            echo "-- $mode"
            grep -E "model buffer size|repack|graph splits" "$OUT/ppl-$mode.log" | sed -E 's/^[0-9.]+ [A-Z] //; s/^/   /' | head -6
        done
    fi
    echo "logs: $OUT"
}
echo
summary | tee "$OUT/summary.txt"
