#!/usr/bin/env bash
# The output-layer probe: the two numbers needed before a kernel is written for the model's output layer (token_embd,
# Q6_K in the GGUF, read once per generated token; today it is requantized to a Q8_0 layout at load and its 1-row
# matmul reads 644 MiB in 22.4 ms on Qwen3.5-4B, 394 MiB in 13.6 ms on Qwen3-4B).
# It builds head-probe.cpp (beside this script) against the SDK in SPERT_DIR into <out>/head-probe and runs it once.
# The program prints one fact per line, "<part> <what>: key=value", "check <name>: PASS|FAIL" per check and a summary
# of lines that start with "head-probe:"; a failed step prints the failure and the probe goes on.
#   part a  a 1 GiB read-only sum (RVV at the core's own VLEN, a checksum every configuration must agree on): the
#           default A100 stream at 1, 2, 4 and 8 tiles; spine-runtime's spacemit-k3-x100 backend (4 tiles, 4 tiles and
#           the calling thread); 8 and 4 unpinned std::thread; and both backends at once, the split chosen from the
#           solo rates and at 0.75, A100 + x100 and A100 + 8 threads
#   part b  a 1-row Q6_K matmul straight on the file's layout (26.25 bytes per 32 weights against the Q8_0 copy's 34):
#           a scalar reference, ggml's float decode as a second reference, an RVV kernel for VLEN 1024 (two epilogues:
#           the block's lanes reduced in integers, or float lanes reduced once a row), compute-only on 8 and 1 tiles,
#           the whole table once on 8, 4 and 1 tiles for both head shapes, and the read-only sum of the Q8_0 head's
#           bytes for comparison
# Both parts take the AI cores, and spine-runtime maps /dev/tcm: this script refuses to start while another process
# has /dev/tcm mapped (PROJECT.md section 6; set HEAD_PROBE_FORCE=1 to run anyway), and changes no system setting.
# Memory: at most about 1.2 GiB at once (the 1 GiB buffer is freed before the tables, and one table lives at a time).
# Expected run time: a few minutes.
#
# usage: head-probe.sh run [out]   (default out: ~/head-probe; an existing directory is refused)
# env:   SPERT_DIR (default $HOME/spine-runtime: include/ and lib/, the 0.6.3 SDK the provider is built with),
#        CXX (g++), PARTS (default all; "a", "b" or "all"), PROBE_MIB (1024), REPS (5), PROBE_GRANT_S (20; 0 = no
#        wait limit), PROBE_ROWS1 (248320), PROBE_ROWS2 (151936), PROBE_COLS (2560), PROBE_COMPUTE_MS (50),
#        PROBE_SAMPLE (512), PROBE_SEED (1), SMALL=1 (small sizes, for a smoke test), TIMEOUT (600),
#        HEAD_PROBE_FORCE=1 (start although another process has /dev/tcm mapped), DRY=1 (print the commands only)
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
SPERT_DIR=${SPERT_DIR:-$HOME/spine-runtime}
CXX=${CXX:-g++}
PARTS=${PARTS:-all}
PROBE_MIB=${PROBE_MIB:-1024}
REPS=${REPS:-5}
PROBE_GRANT_S=${PROBE_GRANT_S:-20}
PROBE_ROWS1=${PROBE_ROWS1:-248320}
PROBE_ROWS2=${PROBE_ROWS2:-151936}
PROBE_COLS=${PROBE_COLS:-2560}
PROBE_COMPUTE_MS=${PROBE_COMPUTE_MS:-50}
PROBE_SAMPLE=${PROBE_SAMPLE:-512}
PROBE_SEED=${PROBE_SEED:-1}
SMALL=${SMALL:-}
TIMEOUT=${TIMEOUT:-600}
DRY=${DRY:-}
export PROBE_MIB REPS PROBE_GRANT_S PROBE_ROWS1 PROBE_ROWS2 PROBE_COLS PROBE_COMPUTE_MS PROBE_SAMPLE PROBE_SEED

case "${1:-}" in
    run)
        OUT=${2:-$HOME/head-probe}
        if [ -e "$OUT" ]; then
            echo "$OUT exists: give a new log directory"
            exit 1
        fi
        # the /dev/tcm check of PROJECT.md section 6: the provider and the IME path overwrite each other's blocks
        G=$(for p in /proc/[0-9]*; do grep -qs /dev/tcm "$p"/maps && echo "${p#/proc/} $(cat "$p"/comm 2>/dev/null)"; done)
        echo "processes with /dev/tcm mapped: ${G:-none}"
        if [ -n "$G" ] && [ -z "${HEAD_PROBE_FORCE:-}" ]; then
            echo "another process holds /dev/tcm; stop it or set HEAD_PROBE_FORCE=1"
            exit 1
        fi

        LIB=$SPERT_DIR/lib
        LINK=()
        if [ -e "$LIB/libspert.so" ]; then
            LINK=(-L"$LIB" -lspert)
        else
            for f in "$LIB"/libspert.so.*; do
                [ -e "$f" ] && { LINK=("$f"); break; }
            done
        fi
        if [ ${#LINK[@]} -eq 0 ]; then
            [ -n "$DRY" ] || { echo "no libspert.so or libspert.so.* in $LIB: set SPERT_DIR"; exit 1; }
            LINK=("$LIB/libspert.so.*")
        fi
        # -march and -mabi as the provider's RVV kernels are built with (providers/spacemit/plugin.cmake);
        # -ffp-contract=off so the probe's bit-for-bit checks are not changed by a chance FMA
        BUILD_CMD=("$CXX" -O3 -std=c++17 -ffp-contract=off -march=rv64gcv_zfh_zvfh_zba -mabi=lp64d -pthread
                   -DHEAD_PROBE_SPERT -I"$SPERT_DIR/include" "$HERE/head-probe.cpp" -o "$OUT/head-probe"
                   "${LINK[@]}" -Wl,-rpath,"$LIB")
        RUN_CMD=(timeout --foreground -k 10 "$TIMEOUT" "$OUT/head-probe" $PARTS)
        [ -n "$SMALL" ] && RUN_CMD+=(--small)

        echo "$(date +%H:%M:%S) ${BUILD_CMD[*]} > $OUT/build.out 2> $OUT/build.err"
        if [ -z "$DRY" ]; then
            [ -f "$SPERT_DIR/include/spert.hpp" ] || { echo "no $SPERT_DIR/include/spert.hpp: set SPERT_DIR"; exit 1; }
            mkdir -p "$OUT"
            if ! "${BUILD_CMD[@]}" > "$OUT/build.out" 2> "$OUT/build.err"; then
                echo "build failed; the first 30 lines of $OUT/build.err:"
                head -n 30 "$OUT/build.err"
                exit 1
            fi
            { echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
              echo "repo: $REPO at $(git -c safe.directory="$REPO" -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
              echo "kernel: $(uname -sr)"
              echo "spert: $SPERT_DIR, VERSION_NUMBER $(cat "$SPERT_DIR/VERSION_NUMBER" 2>/dev/null || echo absent)"
              echo "compiler: $($CXX --version 2>/dev/null | head -n 1)"
              echo "link: ${LINK[*]}"
              echo "parts: $PARTS; probe_mib: $PROBE_MIB; reps: $REPS; grant wait: $PROBE_GRANT_S s; timeout: $TIMEOUT s"
              echo "rows: $PROBE_ROWS1, $PROBE_ROWS2; cols: $PROBE_COLS; compute_ms: $PROBE_COMPUTE_MS; sample: $PROBE_SAMPLE; seed: $PROBE_SEED; small: ${SMALL:-unset}"
              echo "tcm users at start: ${G:-none}"
              env | grep '^SPERT_' | sort | while read -r v; do echo "env: $v"; done; } > "$OUT/info.txt"
        fi

        echo "$(date +%H:%M:%S) ${RUN_CMD[*]} < /dev/null > $OUT/probe.out 2> $OUT/probe.err"
        [ -n "$DRY" ] && exit 0
        "${RUN_CMD[@]}" < /dev/null > "$OUT/probe.out" 2> "$OUT/probe.err"
        rc=$?
        echo "$rc" > "$OUT/probe.rc"
        grep -E '^(head-probe:|check .*FAIL)' "$OUT/probe.out"
        echo "rc: $rc (1 = a check failed, 3 = a stream grant was left waiting, 124 = killed after $TIMEOUT s); probe.err: $(wc -l < "$OUT/probe.err" | tr -d ' ') lines; logs: $OUT"
        ;;
    *)
        echo "usage: $0 run [out]"
        exit 2
        ;;
esac
