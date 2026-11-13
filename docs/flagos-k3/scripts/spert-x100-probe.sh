#!/usr/bin/env bash
# The spine-runtime X100 probe: facts about spine-runtime's spacemit-k3-x100 backend (a spert backend on the X100
# cores) for load-time work such as the weight repack (T6, std::thread today: spacemit_pack in
# plugin/flagos-spacemit-weights.cpp). It builds spert-x100-probe.cpp (beside this script) against the SDK in SPERT_DIR
# into <out>/spert-x100-probe and runs it once. The program prints one fact per line, "<step> <what>: key=value", and
# goes on after a failed step. Steps, in this order:
#   info          the x100 backend: valid, shared_mem_size, vlen, num_cores, core_arch_id
#   grant         x100 streams one at a time, never more than the backend's num_cores N: n_cores 0, N, N/2, and
#                 one core_ids request only when PROBE_CORE_IDS is set (the API gives no id list); valid, core_count, ms
#   place         the CPU and thread of each tile on the largest grant, twice
#   work          a repack-like pass over PROBE_MIB MiB: one thread, 8 and 4 std::thread (the T6 pattern), x100 tiles,
#                 x100 tiles plus the calling thread; median MiB/s of REPS runs and a checksum that must agree
#   overhead      empty launch+sync on the x100 stream against create+join of 7 std::thread, us per round
#   idle          process CPU seconds in 2 s with the x100 stream idle, then 2 s after it is destroyed
#   info-default  the default backend (the A100 AI cores), as info
#   coexist       a default-backend Stream(8) beside the x100 stream, its tile CPUs, one x100 work pass meanwhile
#   oversize      not in the default list; only when STEPS names it and PROBE_OVERSIZE=1, and always last: x100
#                 requests above N (n_cores 8 if N < 8, n_cores 16 if N < 16, core_ids 0-7 with n_cores 8 if N < 8)
# "maps" lines at the start and after each step: /dev/tcm mapped or not, the thread count, process CPU seconds.
# A stream that waits more than PROBE_GRANT_S for its grant is reported blocked and left waiting; the program then
# ends with rc 3.
# oversize risk: a request above N may wait in the arbiter's FIFO, block every later request, and may leave a waiting
# entry in the shared arbiter after the run. Run it only on a board where nobody else uses spert's x100 backend.
#
# The script builds only the probe and changes no system setting. info-default and coexist take the AI cores and TCM:
# run the /dev/tcm check of PROJECT.md section 6 first and keep the run apart from other TCM users (the provider, IME
# runs). STEPS without those two is expected to touch only the x100 backend; that is not verified, and the maps lines
# (tcm=) show it.
#
# usage: spert-x100-probe.sh run [out]   (default out: ~/spert-x100)
# env:   SPERT_DIR (default $HOME/spine-runtime: include/ and lib/, the 0.6.3 SDK the provider is built with),
#        CXX (g++), STEPS (default all, which leaves out oversize; a space-separated subset of the steps above, e.g.
#        "all oversize"), PROBE_MIB (1024), REPS (5), PROBE_GRANT_S (20; 0 = no wait limit), PROBE_CORE_IDS (unset;
#        a comma list such as 4,5,6,7 for one core_ids request in grant), PROBE_OVERSIZE (unset; 1 lets the oversize
#        step run), TIMEOUT (seconds for the whole run, 600), DRY=1 (print the commands only)
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
SPERT_DIR=${SPERT_DIR:-$HOME/spine-runtime}
CXX=${CXX:-g++}
STEPS=${STEPS:-all}
PROBE_MIB=${PROBE_MIB:-1024}
REPS=${REPS:-5}
PROBE_GRANT_S=${PROBE_GRANT_S:-20}
PROBE_CORE_IDS=${PROBE_CORE_IDS:-}
PROBE_OVERSIZE=${PROBE_OVERSIZE:-}
TIMEOUT=${TIMEOUT:-600}
DRY=${DRY:-}
export PROBE_MIB REPS PROBE_GRANT_S PROBE_CORE_IDS PROBE_OVERSIZE

case "${1:-}" in
    run)
        OUT=${2:-$HOME/spert-x100}
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
        BUILD_CMD=("$CXX" -O2 -std=c++17 -pthread -I"$SPERT_DIR/include" "$HERE/spert-x100-probe.cpp"
                   -o "$OUT/spert-x100-probe" "${LINK[@]}" -Wl,-rpath,"$LIB")
        RUN_CMD=(timeout --foreground -k 10 "$TIMEOUT" "$OUT/spert-x100-probe" $STEPS)

        echo "$(date +%H:%M:%S) ${BUILD_CMD[*]} > $OUT/build.txt 2>&1"
        if [ -z "$DRY" ]; then
            [ -f "$SPERT_DIR/include/spert.hpp" ] || { echo "no $SPERT_DIR/include/spert.hpp: set SPERT_DIR"; exit 1; }
            mkdir -p "$OUT"
            if ! "${BUILD_CMD[@]}" > "$OUT/build.txt" 2>&1; then
                echo "build failed; the first 30 lines of $OUT/build.txt:"
                head -n 30 "$OUT/build.txt"
                exit 1
            fi
            { echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
              echo "repo: $REPO at $(git -c safe.directory="$REPO" -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
              echo "kernel: $(uname -sr)"
              echo "spert: $SPERT_DIR, VERSION_NUMBER $(cat "$SPERT_DIR/VERSION_NUMBER" 2>/dev/null || echo absent)"
              echo "link: ${LINK[*]}"
              echo "steps: $STEPS; probe_mib: $PROBE_MIB; reps: $REPS; grant wait: $PROBE_GRANT_S s; timeout: $TIMEOUT s"
              echo "probe_core_ids: ${PROBE_CORE_IDS:-unset}; probe_oversize: ${PROBE_OVERSIZE:-unset}"
              env | grep '^SPERT_' | sort | while read -r v; do echo "env: $v"; done; } > "$OUT/info.txt"
        fi

        echo "$(date +%H:%M:%S) ${RUN_CMD[*]} < /dev/null > $OUT/probe.out 2> $OUT/probe.err"
        [ -n "$DRY" ] && exit 0
        "${RUN_CMD[@]}" < /dev/null > "$OUT/probe.out" 2> "$OUT/probe.err"
        rc=$?
        echo "$rc" > "$OUT/probe.rc"
        cat "$OUT/probe.out"
        echo "rc: $rc (124 = killed after $TIMEOUT s); probe.err: $(wc -l < "$OUT/probe.err" | tr -d ' ') lines; logs: $OUT"
        ;;
    *)
        echo "usage: $0 run [out]"
        exit 2
        ;;
esac
