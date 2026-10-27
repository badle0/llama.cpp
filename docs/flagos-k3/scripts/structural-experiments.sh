#!/usr/bin/env bash
# Structural experiments on the K3 (docs/flagos-k3/structural-experiments.md): E2, the provider as an IGPU or GPU device
# on stock llama.cpp; E1, ACCEL on stock llama.cpp (without C5a and C5b); E3, one launch per step; E2-hp, the KV cache
# on huge pages; E2-neg, the two settings that failed with the KV cache in the provider's buffer (F8); E4, stock upstream
# llama.cpp loading the provider as a run-time module (no upstream change at all); E3c, test-only counters of where a
# token's time goes; R12-hw, upstream's branch of the 1-row Q4_1 kernel on the board (prepare only). Every mode runs
# against the reference (today's build, defaults) in the same session, the modes interleaved in every repetition.
#
# Usage: structural-experiments.sh prepare                  the builds $EXPS needs. B (e1, e2, e2hp, e2neg): a worktree
#                                                           of the repository's HEAD at $WT_B whose src/llama-context.cpp
#                                                           has C5a and C5b reverse-applied (upstream's file), built in
#                                                           $BUILD_B. S and M (e4): S, a worktree of upstream $UPSTREAM_BASE
#                                                           at $WT_S built with backends as run-time modules in $BUILD_S;
#                                                           M, this repository's provider built as such a module in
#                                                           $BUILD_M; then a smoke check (the stock programs list the
#                                                           device through GGML_BACKEND_PATH). Always: build A's tools
#                                                           (today's build-flagos/) and its provider checks, 5 times with
#                                                           e3 (they compare E3a with one launch, byte for byte; E3b only
#                                                           flagos-check-spacemit --e3b, since 2026-10-21).
#                                                           R (r12), after A: $BUILD_R12, configured as A with
#                                                           FLAGOS_SPACEMIT_TEST_UPSTREAM_Q4_1_M1 defined (upstream's
#                                                           branch of the 1-row Q4_1 kernel); its provider checks run
#                                                           once, and their Q4_1 line shows the zero-point probe
#        structural-experiments.sh run <model.gguf> [out]   the runs of $EXPS, then the summary (logs in out; default
#                                                           ~/sx/<model name>)
#        structural-experiments.sh summary <out>            the summary again from the logs
# Env:   EXPS      experiments (default "e2 e2hp e2neg e4"; also e1, e3, e3a (E3a alone, without E3b), e3c and r12);
#                  the reference always runs
#        REPS      llama-bench repetitions per mode (default 3; 0 skips llama-bench)
#        DEPTH     1: also tg32 at depths 0, 1024 and 4096 and pp512 at depth 4096 (default 1)
#        PPL_TEXT  perplexity text (default ~/ppl.txt; empty skips perplexity and KLD); CHUNKS (default 8)
#        KLD_UB1   1: also the KLD with one token per ubatch (-ub 1 -b 512) for E2's IGPU (default 1; ~10 min a run)
#        BUILD_A   today's build (default <repo>/build-flagos); WT_B (default <repo>-no-c5); BUILD_B (default
#                  $WT_B/build-flagos); WT_S (default <repo>-stock); BUILD_S (default $WT_S/build-dl); BUILD_M (default
#                  <repo>/build-flagos-dl); BUILD_R12 (default <repo>/build-r12); SPERT_DIR (default ~/spine-runtime);
#                  TCM_RELEASE (default ~/tcmtest/tcmrelease)
#        TIMEOUT   seconds per run (default 3600); DRY=1 prints the commands instead of running them
# Modes (build A: today's, with C5a and C5b; build B: upstream's llama-context.cpp; build S: stock upstream with backends
# as run-time modules, the provider loaded from build M; -fa is llama.cpp's default, auto, unless named; the provider
# reports IGPU unless FLAGOS_SPACEMIT_TEST_DEVICE_TYPE names another type, D3):
#   ref           A  IGPU                                    today's state
#   e1-auto       B  FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=accel  E1: ACCEL; -fa auto turns flash attention off
#   e1-on         B  as e1-auto, -fa on                      E1: ACCEL; the norms pinned to the CPU
#   e2-igpu       B  FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=igpu, -ngl 99
#   e2-gpu        B  FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=gpu, -ngl 99
#   e2-igpu-auto  B  as e2-igpu with llama.cpp's default -ngl (its automatic fit); placement only
#   e3            A  FLAGOS_SPACEMIT_TEST_LAUNCH=per-step        E3a: one launch per step, the host waiting for each
#   e3b           A  FLAGOS_SPACEMIT_TEST_LAUNCH=per-step-join   E3b: ordered in the cores, the host waiting once
#   e3c           A  FLAGOS_SPACEMIT_TEST_STATS=<log>.stats       E3c: the counters, written when each run ends
#   e3c-a         A  as e3c, with FLAGOS_SPACEMIT_TEST_LAUNCH=per-step: E3c on E3a, what a launch costs inside a run
#   ref-nohp      A  FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1    E2-hp, with e2-igpu-nohp (B, igpu, no huge pages)
#   neg-{q8,shift}-{igpu,ref}  E2-neg: a Q8_0 KV cache and a context shift, as e2-igpu (B) and as ref (A, IGPU)
#   e4-igpu       S  GGML_BACKEND_PATH=<M's libggml-flagos.so>, FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=igpu, -ngl 99
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../.." && pwd)}
BUILD_A=${BUILD_A:-$REPO/build-flagos}
WT_B=${WT_B:-$REPO-no-c5}
BUILD_B=${BUILD_B:-$WT_B/build-flagos}
WT_S=${WT_S:-$REPO-stock}
BUILD_S=${BUILD_S:-$WT_S/build-dl}
BUILD_M=${BUILD_M:-$REPO/build-flagos-dl}
BUILD_R12=${BUILD_R12:-$REPO/build-r12}
# where CMake writes a backend module (ggml/src/CMakeLists.txt, ggml_add_backend_library: the runtime output directory)
MODULE=$BUILD_M/bin/libggml-flagos.so
SPERT_DIR=${SPERT_DIR:-$HOME/spine-runtime}
TCM_RELEASE=${TCM_RELEASE:-$HOME/tcmtest/tcmrelease}
EXPS=${EXPS:-e2 e2hp e2neg e4}
REPS=${REPS:-3}
DEPTH=${DEPTH:-1}
PPL_TEXT=${PPL_TEXT-$HOME/ppl.txt}
CHUNKS=${CHUNKS:-8}
KLD_UB1=${KLD_UB1:-1}
TIMEOUT=${TIMEOUT:-3600}
DRY=${DRY:-}
# C5a and C5b (upstream-changes.md U1, U2) and the upstream commit the fork is based on
C5_COMMITS="64e6cbd f1cf809"
UPSTREAM_BASE=ba360ef
TOOLS="llama-completion llama-bench llama-perplexity"
PROMPT="The capital of France is"
# --foreground: without it timeout runs the command outside the terminal's foreground, and llama-completion, which sets
# the terminal up at start (console::init), is stopped by the kernel (SIGTTOU) and never ends
TO=""
if command -v timeout > /dev/null; then TO="timeout --foreground $TIMEOUT"; fi
# every run sets its own
unset FLAGOS_SPACEMIT_TEST_DEVICE_TYPE FLAGOS_SPACEMIT_TEST_LAUNCH FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES \
    FLAGOS_SPACEMIT_TEST_STATS FLAGOS_SPACEMIT_DISABLE FLAGOS_SPACEMIT_ATTN FLAGOS_SPACEMIT_GEMV

mode_build() {
    case $1 in
        ref | ref-nohp | e3 | e3b | e3c | e3c-a | neg-*-ref) echo "$BUILD_A" ;;
        e4-*) echo "$BUILD_S" ;;
        *) echo "$BUILD_B" ;;
    esac
}
mode_env() {
    case $1 in
        ref-nohp) echo "FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1" ;;
        e1-*) echo "FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=accel" ;;
        e2-igpu | e2-igpu-auto | neg-*-igpu) echo "FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=igpu" ;;
        e2-gpu) echo "FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=gpu" ;;
        e2-igpu-nohp) echo "FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=igpu FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1" ;;
        e3 | e3c-a) echo "FLAGOS_SPACEMIT_TEST_LAUNCH=per-step" ;;
        e3b) echo "FLAGOS_SPACEMIT_TEST_LAUNCH=per-step-join" ;;
        # the stock programs know no "flagos" module by name; GGML_BACKEND_PATH loads one from any path
        e4-igpu) echo "GGML_BACKEND_PATH=$MODULE FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=igpu" ;;
        *) echo "" ;;
    esac
}
mode_ngl() {
    case $1 in
        e2-igpu | e2-gpu | e2-igpu-nohp | neg-*-igpu | e4-igpu) echo "-ngl 99" ;;
        *) echo "" ;;
    esac
}
# llama-bench's -fa value, and the flag for llama-completion and llama-perplexity (none: their default, auto)
mode_fa_bench() { case $1 in e1-on) echo on ;; *) echo auto ;; esac; }
mode_fa() { case $1 in e1-on) echo "-fa on" ;; *) echo "" ;; esac; }

has() { case " $EXPS " in *" $1 "*) return 0 ;; esac; return 1; }
# the board's checkout belongs to another uid (copied from the Mac), so git refuses it ("dubious ownership"); as with
# `git -c safe.directory=$PWD pull` there, every call allows the checkouts for itself and changes no git settings
g() { git -c safe.directory="$REPO" -c safe.directory="$WT_B" -c safe.directory="$WT_S" "$@"; }
release() { if [ -x "$TCM_RELEASE" ]; then "$TCM_RELEASE" --apply > /dev/null 2>&1; fi; return 0; }

# run <mode> <log> <tool> [args...]: the tool of the mode's build with the mode's environment, after a TCM release;
# stdout and stderr to <log>.out and <log>.err, the exit code to <log>.rc
run() {
    local mode=$1 log=$2 tool=$3 bin env rc
    shift 3
    bin=$(mode_build "$mode")/bin/$tool
    env=$(mode_env "$mode")
    # E3c: every run writes its counters next to its log, when it ends
    case $mode in e3c | e3c-a) env="${env:+$env }FLAGOS_SPACEMIT_TEST_STATS=$log.stats" ;; esac
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
    # shellcheck disable=SC2046 # mode_ngl gives a list of words
    run "$mode" "$OUT/bench-$mode-$tag" llama-bench -m "$M" -t 8 -fa "$(mode_fa_bench "$mode")" -mmp 0 -r 1 -o jsonl \
        $(mode_ngl "$mode") "$@"
    echo "   bench $tag $mode: rc=$?"
    if [ -z "$DRY" ]; then
        sed "s/^{/{\"mode\":\"$mode\",/" "$OUT/bench-$mode-$tag.out" >> "$OUT/bench.jsonl"
    fi
}

# build B (E1, E2): a worktree of HEAD whose src/llama-context.cpp has C5a and C5b reverse-applied
prepare_b() {
    local head c
    head=$(g -C "$REPO" rev-parse HEAD) || exit 1
    echo "== build B: $WT_B at $head, src/llama-context.cpp without C5a and C5b"
    if [ -n "$DRY" ]; then
        echo "   git worktree add/checkout $WT_B $head; reverse-apply $C5_COMMITS; cmake -S $WT_B -B $BUILD_B ..."
        return 0
    fi
    if [ -d "$WT_B" ]; then
        g -C "$WT_B" checkout -q -f --detach "$head" || exit 1
    else
        g -C "$REPO" worktree add -q --detach "$WT_B" "$head" || exit 1
    fi
    for c in $C5_COMMITS; do
        g -C "$WT_B" diff "$c^" "$c" -- src/llama-context.cpp | g -C "$WT_B" apply -R || {
            echo "reverse-applying $c failed"
            exit 1
        }
    done
    if g -C "$WT_B" diff --quiet "$UPSTREAM_BASE" -- src; then
        echo "   src/ is upstream's ($UPSTREAM_BASE)"
    else
        echo "   WARNING: src/ differs from upstream $UPSTREAM_BASE beyond C5a and C5b:"
        g -C "$WT_B" diff --stat "$UPSTREAM_BASE" -- src
    fi
    mkdir -p "$BUILD_B"
    cmake -S "$WT_B" -B "$BUILD_B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_FLAGOS=ON -DGGML_FLAGOS_DENGLIN=OFF \
        -DGGML_FLAGOS_AMD=OFF -DGGML_FLAGOS_SPACEMIT=ON -DFLAGOS_SPACEMIT_SPERT_DIR="$SPERT_DIR" \
        > "$BUILD_B/sx-configure.log" 2>&1 || { echo "   configure failed: $BUILD_B/sx-configure.log"; exit 1; }
    # shellcheck disable=SC2086 # TOOLS is a list
    cmake --build "$BUILD_B" --target $TOOLS > "$BUILD_B/sx-build.log" 2>&1
    echo "   build B: rc=$?; $(grep -cE 'error:|FAILED' "$BUILD_B/sx-build.log") error lines ($BUILD_B/sx-build.log)"
}

# the soname a shared library carries, or one it needs (readelf on Linux; empty where it is missing)
elf_names() { if command -v readelf > /dev/null; then readelf -d "$2" 2> /dev/null | sed -nE "s/.*\($1\).*\[(.*)\]/\1/p"; fi; }

# builds S and M (E4): stock upstream with every backend a run-time module (GGML_BACKEND_DL), and this repository's
# provider built as such a module; the stock programs then load it from M through GGML_BACKEND_PATH
prepare_sm() {
    echo "== build S: $WT_S at upstream $UPSTREAM_BASE, backends as run-time modules"
    if [ -n "$DRY" ]; then
        echo "   git worktree add/checkout $WT_S $UPSTREAM_BASE; cmake -S $WT_S -B $BUILD_S -DGGML_BACKEND_DL=ON ..."
        echo "== build M: $BUILD_M; cmake -S $REPO -B $BUILD_M -DGGML_BACKEND_DL=ON -DGGML_FLAGOS=ON ...; target ggml-flagos"
        return 0
    fi
    if [ -d "$WT_S" ]; then
        g -C "$WT_S" checkout -q -f --detach "$UPSTREAM_BASE" || exit 1
    else
        g -C "$REPO" worktree add -q --detach "$WT_S" "$UPSTREAM_BASE" || exit 1
    fi
    mkdir -p "$BUILD_S"
    # upstream's defaults otherwise; its SpacemiT CPU path (GGML_CPU_RISCV64_SPACEMIT) stays off, as in A and B: it would
    # be a second TCM user in the process
    cmake -S "$WT_S" -B "$BUILD_S" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON \
        > "$BUILD_S/sx-configure.log" 2>&1 || { echo "   configure failed: $BUILD_S/sx-configure.log"; exit 1; }
    # the programs depend on ggml, which depends on every backend module (add_dependencies), the CPU's included
    # shellcheck disable=SC2086
    cmake --build "$BUILD_S" --target $TOOLS > "$BUILD_S/sx-build.log" 2>&1
    echo "   build S: rc=$?; $(grep -cE 'error:|FAILED' "$BUILD_S/sx-build.log") error lines ($BUILD_S/sx-build.log)"

    echo "== build M: $BUILD_M, the provider as a run-time module"
    mkdir -p "$BUILD_M"
    cmake -S "$REPO" -B "$BUILD_M" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON \
        -DGGML_FLAGOS=ON -DGGML_FLAGOS_DENGLIN=OFF -DGGML_FLAGOS_AMD=OFF -DGGML_FLAGOS_SPACEMIT=ON \
        -DFLAGOS_SPACEMIT_SPERT_DIR="$SPERT_DIR" > "$BUILD_M/sx-configure.log" 2>&1 ||
        { echo "   configure failed: $BUILD_M/sx-configure.log"; exit 1; }
    cmake --build "$BUILD_M" --target ggml-flagos > "$BUILD_M/sx-build.log" 2>&1
    echo "   build M: rc=$?; $(grep -cE 'error:|FAILED' "$BUILD_M/sx-build.log") error lines ($BUILD_M/sx-build.log)"
    if [ ! -f "$MODULE" ]; then
        echo "   no module at $MODULE"
        exit 1
    fi

    echo "== smoke check: the stock programs with the provider module"
    # one libggml-base in the process: the dynamic linker reuses the stock one only if the module needs it by the same
    # name (same ggml version on both sides)
    echo "   libggml-base: S $(elf_names SONAME "$BUILD_S/bin/libggml-base.so"), M $(elf_names SONAME \
        "$BUILD_M/bin/libggml-base.so"); the module needs: $(elf_names NEEDED "$MODULE" | tr '\n' ' ')"
    release
    GGML_BACKEND_PATH="$MODULE" FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=igpu "$BUILD_S/bin/llama-completion" --list-devices \
        < /dev/null > "$BUILD_S/sx-devices.log" 2>&1
    echo "   llama-completion --list-devices: rc=$? ($BUILD_S/sx-devices.log)"
    grep -E "SpacemiT|reporting|failed|error" "$BUILD_S/sx-devices.log" | sed 's/^/   /'
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
    # the checks include E2's layer with the KV cache in the provider's buffer and E3a's comparisons of one launch per
    # step with one per layer, byte for byte (E3b's left them 2026-10-21: flagos-check-spacemit --e3b); with e3 they run
    # 5 times. A hang ends after 15 minutes
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
        grep -E "FAIL|two tiles on one core|ordered launches" "$BUILD_A/sx-check.err" | head -5 | sed 's/^/      /'
    done
}

# build R (R12-hw): build A's configuration with upstream's branch of the 1-row Q4_1 kernel compiled in. Its provider
# checks include the zero-point probe (zero point 15 against 32 equal activations), which fails the run if the 1-row
# result is wrong; with upstream's branch it shows whether the 16-bit product wraps on the board
prepare_r12() {
    echo "== build R: $BUILD_R12, the 1-row Q4_1 kernel with upstream's branch (FLAGOS_SPACEMIT_TEST_UPSTREAM_Q4_1_M1)"
    if [ -n "$DRY" ]; then
        echo "   cmake -S $REPO -B $BUILD_R12 ... -DCMAKE_CXX_FLAGS=-DFLAGOS_SPACEMIT_TEST_UPSTREAM_Q4_1_M1;" \
            "flagos-check-spacemit --expect-device"
        return 0
    fi
    mkdir -p "$BUILD_R12"
    cmake -S "$REPO" -B "$BUILD_R12" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_FLAGOS=ON -DGGML_FLAGOS_DENGLIN=OFF \
        -DGGML_FLAGOS_AMD=OFF -DGGML_FLAGOS_SPACEMIT=ON -DFLAGOS_SPACEMIT_SPERT_DIR="$SPERT_DIR" \
        -DCMAKE_CXX_FLAGS=-DFLAGOS_SPACEMIT_TEST_UPSTREAM_Q4_1_M1 > "$BUILD_R12/sx-configure.log" 2>&1 ||
        { echo "   configure failed: $BUILD_R12/sx-configure.log"; exit 1; }
    cmake --build "$BUILD_R12" --target flagos-check-spacemit > "$BUILD_R12/sx-build.log" 2>&1
    echo "   build R: rc=$?; $(grep -cE 'error:|FAILED' "$BUILD_R12/sx-build.log") error lines ($BUILD_R12/sx-build.log)"
    local rc to=""
    if command -v timeout > /dev/null; then to="timeout --foreground 900"; fi
    release
    # shellcheck disable=SC2086
    $to "$BUILD_R12/bin/flagos-check-spacemit" --expect-device < /dev/null > "$BUILD_R12/sx-check.out" \
        2> "$BUILD_R12/sx-check.err"
    rc=$?
    echo "   flagos-check-spacemit, upstream's branch: rc=$rc: $(tail -1 "$BUILD_R12/sx-check.out")"
    grep -E "^matmul   Q4_1" "$BUILD_R12/sx-check.out" | grep -oE "zero point 15[^;]*" | sed 's/^/      /'
    grep -E "FAIL" "$BUILD_R12/sx-check.err" | head -8 | sed 's/^/      /'
}

prepare() {
    if has e1 || has e2 || has e2hp || has e2neg; then prepare_b; fi
    if has e4; then prepare_sm; fi
    prepare_a
    if has r12; then prepare_r12; fi
}

summary() {
    local m f rc splits fa kv off type
    echo
    echo "== placement (llama-completion; graph splits for the batch / 1-token graphs; flash attention under -fa auto)"
    printf '   %-14s %-4s %-9s %-10s %-11s %-6s %s\n' mode rc splits "flash att." offloaded type "KV cache"
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
        # the type the mode's switch reports; its one warning cannot tell: under llama.cpp's automatic fit (no -ngl) it
        # comes while the fit turns messages into debug ones. The offloaded layers and the KV cache show the effect
        case $(mode_env "$m") in
            *DEVICE_TYPE=igpu*) type=IGPU ;;
            *DEVICE_TYPE=accel*) type=ACCEL ;;
            *DEVICE_TYPE=gpu*) type=GPU ;;
            *) type=IGPU ;;
        esac
        kv=$(grep -oE '[A-Za-z0-9:_]+ KV buffer size = +[0-9.]+ MiB' "$f" | sed -E 's/ KV buffer size = +/ /' | sort -u |
            paste -sd, -)
        printf '   %-14s %-4s %-9s %-10s %-11s %-6s %s\n' "$m" "${rc:-?}" "${splits:--}" "${fa:-set}" "${off:--}" "$type" \
            "${kv:--}"
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
        echo "   own cost); imbal.: busy less the mean tile's kernels"
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
            "steps": k["steps"] / g, "launches": k["launches"] / g})
cols = ["token", "outside", "host", "launch", "barriers", "imbal.", "kernels", "idle", "steps", "launches"]
print("   " + f"{'mode':7}{'test':11}" + "".join(f"{c:>10}" for c in cols) + f"{'idle %':>8}{'launch/each':>12}")
order = {"tg128": 0, "tg32@0": 1, "tg32@1024": 2, "tg32@4096": 3}
mean = {}
for key in sorted(res, key=lambda k: (order.get(k[1], 9), k[1], k[0])):
    v = mean[key] = {c: statistics.mean(x[c] for x in res[key]) for c in cols}
    each = v["launch"] / v["launches"] if v["launches"] else 0.0
    cells = "".join(f"{v[c]:10.1f}" if c in ("steps", "launches") else f"{v[c]:10.0f}" for c in cols)
    print("   " + f"{key[0]:7}{key[1]:11}" + cells + f"{100 * v['idle'] / v['token']:8.2f}{each:12.2f}")
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
    if compgen -G "$OUT/neg-*.err" > /dev/null; then
        echo
        echo "== E2-neg (expected since the provider writes the cache: all four runs end with rc 0, none with \"cannot run"
        echo "   the operation\"; the shift runs have a 256-token context, so more tokens generated than that means the"
        echo "   context was shifted)"
        for f in "$OUT"/neg-*.err; do
            m=$(basename "$f" .err)
            # tokens generated: llama-completion's "eval time = ... / N runs"; its "context full" line is a debug one,
            # shown only from -lv 5
            gen=$(grep -oE 'eval time = +[0-9.]+ ms / +[0-9]+ runs' "$f" | tail -1 | grep -oE '[0-9]+ runs' | cut -d' ' -f1)
            printf '   %-16s rc %-4s generated %-5s %s\n' "$m" "$(cat "${f%.err}.rc" 2> /dev/null)" "${gen:--}" \
                "$(grep -m1 -oE 'pre-allocated tensor.*|GGML_ASSERT.*' "$f" | cut -c1-100)"
        done
    fi
    echo "logs: $OUT"
}

run_all() {
    local place bench_modes depth_modes kld_modes kld1_modes m i base
    M=${1:?usage: structural-experiments.sh run <model.gguf> [out]}
    OUT=${2:-$HOME/sx/$(basename "$M" .gguf)}
    [ -n "$DRY" ] || [ -f "$M" ] || { echo "model not found: $M"; exit 1; }
    local need=$BUILD_A
    if has e1 || has e2 || has e2hp || has e2neg; then need+=" $BUILD_B"; fi
    if has e4; then need+=" $BUILD_S"; fi
    for b in $need; do
        [ -n "$DRY" ] || [ -x "$b/bin/llama-bench" ] || { echo "not built: $b (structural-experiments.sh prepare)"; exit 1; }
    done
    if has e4 && [ -z "$DRY" ] && [ ! -f "$MODULE" ]; then
        echo "not built: $MODULE (EXPS=e4 structural-experiments.sh prepare)"
        exit 1
    fi
    [ -n "$DRY" ] || mkdir -p "$OUT"
    rm -f "$OUT/bench.jsonl"

    place="ref" bench_modes="ref" depth_modes="ref" kld_modes="" kld1_modes=""
    if has e2; then
        place+=" e2-igpu e2-gpu e2-igpu-auto" bench_modes+=" e2-igpu e2-gpu" depth_modes+=" e2-igpu e2-gpu"
        kld_modes+=" e2-igpu e2-gpu" kld1_modes+=" e2-igpu"
    fi
    if has e1; then
        place+=" e1-auto e1-on" bench_modes+=" e1-auto e1-on" depth_modes+=" e1-auto e1-on" kld_modes+=" e1-auto e1-on"
    fi
    if has e3; then
        place+=" e3 e3b" bench_modes+=" e3 e3b" depth_modes+=" e3 e3b" kld_modes+=" e3 e3b"
    elif has e3a; then
        place+=" e3" bench_modes+=" e3" depth_modes+=" e3" kld_modes+=" e3"
    fi
    # the counters change no output (the checks compare them byte for byte), so no KLD
    if has e3c; then
        bench_modes+=" e3c e3c-a" depth_modes+=" e3c e3c-a"
    fi
    if has e2hp; then
        case " $depth_modes " in *" e2-igpu "*) ;; *) depth_modes+=" e2-igpu" ;; esac
        depth_modes+=" ref-nohp e2-igpu-nohp"
    fi
    if has e4; then
        place+=" e4-igpu" bench_modes+=" e4-igpu" depth_modes+=" e4-igpu" kld_modes+=" e4-igpu" kld1_modes+=" e4-igpu"
    fi
    if [ -z "$DRY" ]; then
        {
            echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
            echo "model: $M; EXPS: $EXPS; REPS: $REPS; DEPTH: $DEPTH; PPL_TEXT: $PPL_TEXT; CHUNKS: $CHUNKS"
            echo "A: $BUILD_A at $(g -C "$REPO" rev-parse --short HEAD)"
            if has e1 || has e2 || has e2hp || has e2neg; then
                echo "B: $BUILD_B at $(g -C "$WT_B" rev-parse --short HEAD), changed: $(g -C "$WT_B" diff --stat | tail -1)"
            fi
            if has e4; then
                echo "S: $BUILD_S at $(g -C "$WT_S" rev-parse --short HEAD), changed: $(g -C "$WT_S" diff --stat | tail -1)"
                echo "M: $MODULE, built from $(g -C "$REPO" rev-parse --short HEAD)"
            fi
        } > "$OUT/info.txt"
    fi

    echo "== placement and text"
    for m in $place; do
        # shellcheck disable=SC2046
        run "$m" "$OUT/load-$m" llama-completion -m "$M" -p "$PROMPT" -n 32 --temp 0 -no-cnv -t 8 -lv 4 \
            $(mode_ngl "$m") $(mode_fa "$m")
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
            # shellcheck disable=SC2046
            run "$m" "$OUT/kld-$m" llama-perplexity -m "$M" -f "$PPL_TEXT" -c 512 --chunks "$CHUNKS" -t 8 \
                $(mode_ngl "$m") $(mode_fa "$m") --kl-divergence-base "$base" --kl-divergence
            echo "   kld $m: rc=$?"
        done
        if [ "$KLD_UB1" = 1 ] && [ -n "$kld1_modes" ]; then
            # one sequence, so every ubatch is one token, as in generation (build.md §6)
            run ref "$OUT/ppl-ref-ub1" llama-perplexity -m "$M" -f "$PPL_TEXT" -c 512 -b 512 -ub 1 --chunks "$CHUNKS" \
                -t 8 --kl-divergence-base "$base.ub1"
            echo "   ppl ref -ub 1 (writes the base logits): rc=$?"
            for m in $kld1_modes; do
                # shellcheck disable=SC2046
                run "$m" "$OUT/kld-$m-ub1" llama-perplexity -m "$M" -f "$PPL_TEXT" -c 512 -b 512 -ub 1 \
                    --chunks "$CHUNKS" -t 8 $(mode_ngl "$m") --kl-divergence-base "$base.ub1" --kl-divergence
                echo "   kld $m -ub 1: rc=$?"
            done
        fi
        rm -f "$base" "$base.ub1"
    fi

    if has e2neg; then
        echo "== E2-neg"
        for m in neg-q8-igpu neg-q8-ref; do
            # shellcheck disable=SC2046
            run "$m" "$OUT/$m" llama-completion -m "$M" -p "$PROMPT" -n 8 --temp 0 -no-cnv -t 8 -c 512 \
                -ctk q8_0 -ctv q8_0 -fa on $(mode_ngl "$m")
            echo "   $m: rc=$?"
        done
        for m in neg-shift-igpu neg-shift-ref; do
            # shellcheck disable=SC2046
            run "$m" "$OUT/$m" llama-completion -m "$M" -p "$PROMPT" -n 400 --ignore-eos --temp 0 -no-cnv -t 8 \
                -c 256 --context-shift -lv 4 $(mode_ngl "$m")
            echo "   $m: rc=$?"
        done
    fi

    if [ -z "$DRY" ]; then
        summary
    fi
}

case ${1:-} in
    prepare) prepare ;;
    run) shift; run_all "$@" ;;
    summary) OUT=${2:?usage: structural-experiments.sh summary <out>}; summary ;;
    *) sed -n '2,30p' "$0"; exit 1 ;;
esac
