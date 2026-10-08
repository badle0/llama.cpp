#!/usr/bin/env bash
# E1 (device-type.md X1-X5): how the ggml device type changes llama.cpp's behavior on the K3.
# Needs ggml-spacemit built with patch-spacemit.py (knobs GGML_SPACEMIT_DEVICE_TYPE and GGML_SPACEMIT_ACCEPT_HOST).
# Usage: e1-device-type.sh <model.gguf> [out_dir]
# Env:   MODES   modes with full measurements (default "accel gpu igpu cpu")
#        EXTRA   placement-only modes (default "auto half accel-dev"; set EXTRA= to skip)
#        REPS    llama-bench repetitions per mode (default 3; 0 skips the benchmark)
#        DEPTHS  KV depths for llama-bench (default 0,1024)
#        PPL_TEXT text file for llama-perplexity (empty skips it); CHUNKS (default 8)
#        HALF    layers for the "half" mode (default 14); FA flash attention for text and perplexity (default on)
#        ACCEPT_HOST=1  buffer-policy swap (X3): the backend may use non-weight tensors in host buffers
# Modes: accel = ACCEL; gpu / igpu = GPU / IGPU with all layers offloaded; cpu = GPU type with -ngl 0 (X100 CPU only);
#        auto = GPU type without -ngl; half = GPU type, -ngl $HALF; accel-dev = ACCEL plus --device SPACEMIT0.
set -uo pipefail

M=${1:?usage: e1-device-type.sh <model.gguf> [out_dir]}
TAG=""
if [ -n "${ACCEPT_HOST:-}" ]; then TAG="-host"; fi
OUT=${2:-$HOME/e1/$(basename "$M" .gguf)$TAG}
BIN=${BIN:-$HOME/llama.cpp-spacemit/build/bin}
SPERT_DIR=${SPERT_DIR:-$HOME/spine-runtime}
MODES=${MODES-accel gpu igpu cpu}
EXTRA=${EXTRA-auto half accel-dev}
REPS=${REPS:-3}
DEPTHS=${DEPTHS:-0,1024}
PPL_TEXT=${PPL_TEXT:-}
CHUNKS=${CHUNKS:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
export LD_LIBRARY_PATH=$SPERT_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
if [ -n "${ACCEPT_HOST:-}" ]; then export GGML_SPACEMIT_ACCEPT_HOST=1; else unset GGML_SPACEMIT_ACCEPT_HOST; fi
mkdir -p "$OUT"
rm -f "$OUT/bench.jsonl"

release() { if [ -x "$HOME/tcmtest/tcmrelease" ]; then "$HOME/tcmtest/tcmrelease" --apply > /dev/null; fi; }
dev_type() {
    case $1 in
        accel | accel-dev) echo accel ;;
        igpu) echo igpu ;;
        *) echo gpu ;;
    esac
}
flags() {
    case $1 in
        gpu | igpu) echo "-ngl 99" ;;
        auto) echo "" ;;
        half) echo "-ngl ${HALF:-14}" ;;
        accel-dev) echo "-ngl 0 --device SPACEMIT0" ;;
        *) echo "-ngl 0" ;;
    esac
}

echo "== devices as llama.cpp sees them"
for t in accel gpu igpu; do
    echo "-- $t"
    GGML_SPACEMIT_DEVICE_TYPE=$t "$BIN/llama-completion" --list-devices 2>&1 | grep -v '^$' | sed 's/^/   /'
done

for mode in $MODES $EXTRA; do
    release
    # shellcheck disable=SC2046
    GGML_SPACEMIT_DEVICE_TYPE=$(dev_type "$mode") "$BIN/llama-completion" -m "$M" -p "The capital of France is" \
        -n 32 -no-cnv --temp 0 -t 8 -lv 4 -fa ${FA:-on} $(flags "$mode") > "$OUT/text-$mode.txt" 2> "$OUT/load-$mode.log"
    echo "text $mode: rc=$?"
done

if [ -n "$PPL_TEXT" ]; then
    for mode in $MODES; do
        release
        # shellcheck disable=SC2046
        GGML_SPACEMIT_DEVICE_TYPE=$(dev_type "$mode") "$BIN/llama-perplexity" -m "$M" -f "$PPL_TEXT" -c 512 \
            --chunks "$CHUNKS" -t 8 -fa ${FA:-on} $(flags "$mode") > "$OUT/ppl-$mode.log" 2>&1
        echo "ppl $mode: rc=$?"
    done
fi

for i in $(seq "$REPS"); do
    for mode in $MODES; do
        release
        # shellcheck disable=SC2046
        GGML_SPACEMIT_DEVICE_TYPE=$(dev_type "$mode") "$BIN/llama-bench" -m "$M" -t 8 $(flags "$mode") \
            -p 128 -n 128 -d "$DEPTHS" -ub 128 -fa 1 -mmp 0 -r 1 -o jsonl 2> "$OUT/bench-$mode-$i.log" \
            | sed "s/^{/{\"mode\":\"$mode\",/" >> "$OUT/bench.jsonl"
        echo "bench $mode rep $i: rc=${PIPESTATUS[0]}"
    done
done

echo
echo "== placement and splits${TAG:+ (ACCEPT_HOST)}"
for mode in $MODES $EXTRA; do
    echo "-- $mode (type $(dev_type "$mode"), flags: $(flags "$mode"))"
    grep -E "offloaded|model buffer size|KV buffer size|compute buffer size|graph splits" "$OUT/load-$mode.log" \
        | sed -E 's/^[0-9.]+ [A-Z] //; s/^/   /'
done

echo
echo "== generated text"
for mode in $MODES $EXTRA; do
    printf '   %-10s %s\n' "$mode" "$(tr '\n' ' ' < "$OUT/text-$mode.txt" | cut -c1-90)"
done

if [ -n "$PPL_TEXT" ]; then
    echo
    echo "== perplexity ($CHUNKS chunks of 512 tokens, $(basename "$PPL_TEXT"))"
    for mode in $MODES; do
        printf '   %-10s %s\n' "$mode" "$(grep -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' "$OUT/ppl-$mode.log" || echo 'no result, see log')"
    done
fi

if [ "$REPS" -gt 0 ]; then
    echo
    echo "== llama-bench t/s"
    python3 "$HERE/bench-summary.py" "$OUT/bench.jsonl"
fi
echo "logs: $OUT"
