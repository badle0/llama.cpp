#!/usr/bin/env bash
# Structural experiments on the K3 (docs/flagos-k3/structural-experiments.md): E2, the provider as an IGPU or GPU device
# on stock llama.cpp; E1, ACCEL on stock llama.cpp (without C5a and C5b); E3, one launch per step; E2-hp, the KV cache
# on huge pages; E2-neg, the two settings expected to fail with the KV cache in the provider's buffer. Every mode runs
# against the reference (today's build, defaults) in the same session, the modes interleaved in every repetition.
#
# Usage: structural-experiments.sh prepare                  build B: a worktree of the repository's HEAD at $WT_B whose
#                                                           src/llama-context.cpp has C5a and C5b reverse-applied
#                                                           (upstream's file), built in $BUILD_B; then build A's tools
#                                                           (today's build-flagos/) and its provider checks
#        structural-experiments.sh run <model.gguf> [out]   the runs of $EXPS, then the summary (logs in out; default
#                                                           ~/sx/<model name>)
#        structural-experiments.sh summary <out>            the summary again from the logs
# Env:   EXPS      experiments (default "e2 e1 e3 e2hp e2neg"); the reference always runs
#        REPS      llama-bench repetitions per mode (default 3; 0 skips llama-bench)
#        DEPTH     1: also tg32 at depths 0, 1024 and 4096 and pp512 at depth 4096 (default 1)
#        PPL_TEXT  perplexity text (default ~/ppl.txt; empty skips perplexity and KLD); CHUNKS (default 8)
#        KLD_UB1   1: also the KLD with one token per ubatch (-ub 1 -b 512) for E2's IGPU (default 1; ~10 min a run)
#        BUILD_A   today's build (default <repo>/build-flagos); WT_B (default <repo>-no-c5); BUILD_B (default
#                  $WT_B/build-flagos); SPERT_DIR (default ~/spine-runtime); TCM_RELEASE (default ~/tcmtest/tcmrelease)
#        TIMEOUT   seconds per run (default 3600); DRY=1 prints the commands instead of running them
# Modes (build A: today's, with C5a and C5b; build B: upstream's llama-context.cpp; -fa is llama.cpp's default, auto,
# unless named):
#   ref           A  ACCEL                                   today's state
#   e1-auto       B  ACCEL                                   E1: -fa auto turns flash attention off
#   e1-on         B  ACCEL, -fa on                           E1: the norms pinned to the CPU
#   e2-igpu       B  FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=igpu, -ngl 99
#   e2-gpu        B  FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=gpu, -ngl 99
#   e2-igpu-auto  B  as e2-igpu with llama.cpp's default -ngl (its automatic fit); placement only
#   e3            A  FLAGOS_SPACEMIT_TEST_LAUNCH=per-step
#   ref-nohp      A  FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1    E2-hp, with e2-igpu-nohp (B, igpu, no huge pages)
#   neg-{q8,shift}-{igpu,ref}  E2-neg: a Q8_0 KV cache and a context shift, as e2-igpu (B) and as ref (A)
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../.." && pwd)}
BUILD_A=${BUILD_A:-$REPO/build-flagos}
WT_B=${WT_B:-$REPO-no-c5}
BUILD_B=${BUILD_B:-$WT_B/build-flagos}
SPERT_DIR=${SPERT_DIR:-$HOME/spine-runtime}
TCM_RELEASE=${TCM_RELEASE:-$HOME/tcmtest/tcmrelease}
EXPS=${EXPS:-e2 e1 e3 e2hp e2neg}
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
    FLAGOS_SPACEMIT_DISABLE FLAGOS_SPACEMIT_ATTN FLAGOS_SPACEMIT_GEMV

mode_build() {
    case $1 in
        ref | ref-nohp | e3 | neg-*-ref) echo "$BUILD_A" ;;
        *) echo "$BUILD_B" ;;
    esac
}
mode_env() {
    case $1 in
        ref-nohp) echo "FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1" ;;
        e2-igpu | e2-igpu-auto | neg-*-igpu) echo "FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=igpu" ;;
        e2-gpu) echo "FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=gpu" ;;
        e2-igpu-nohp) echo "FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=igpu FLAGOS_SPACEMIT_TEST_NO_HUGE_PAGES=1" ;;
        e3) echo "FLAGOS_SPACEMIT_TEST_LAUNCH=per-step" ;;
        *) echo "" ;;
    esac
}
mode_ngl() {
    case $1 in
        e2-igpu | e2-gpu | e2-igpu-nohp | neg-*-igpu) echo "-ngl 99" ;;
        *) echo "" ;;
    esac
}
# llama-bench's -fa value, and the flag for llama-completion and llama-perplexity (none: their default, auto)
mode_fa_bench() { case $1 in e1-on) echo on ;; *) echo auto ;; esac; }
mode_fa() { case $1 in e1-on) echo "-fa on" ;; *) echo "" ;; esac; }

has() { case " $EXPS " in *" $1 "*) return 0 ;; esac; return 1; }
# the board's checkout belongs to another uid (copied from the Mac), so git refuses it ("dubious ownership"); as with
# `git -c safe.directory=$PWD pull` there, every call allows the two checkouts for itself and changes no git settings
g() { git -c safe.directory="$REPO" -c safe.directory="$WT_B" "$@"; }
release() { if [ -x "$TCM_RELEASE" ]; then "$TCM_RELEASE" --apply > /dev/null 2>&1; fi; return 0; }

# run <mode> <log> <tool> [args...]: the tool of the mode's build with the mode's environment, after a TCM release;
# stdout and stderr to <log>.out and <log>.err, the exit code to <log>.rc
run() {
    local mode=$1 log=$2 tool=$3 bin env rc
    shift 3
    bin=$(mode_build "$mode")/bin/$tool
    env=$(mode_env "$mode")
    if [ -n "$DRY" ]; then
        echo "   [$mode] ${env:+$env }$bin $*"
        return 0
    fi
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

prepare() {
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

    echo "== build A: $BUILD_A"
    if [ ! -f "$BUILD_A/CMakeCache.txt" ]; then
        echo "   not configured: run ggml/src/ggml-flagos/providers/spacemit/tools/spacemit_check.py --build first"
        exit 1
    fi
    # shellcheck disable=SC2086
    cmake --build "$BUILD_A" --target $TOOLS flagos-check-spacemit > "$BUILD_A/sx-build.log" 2>&1
    echo "   build A: rc=$?; $(grep -cE 'error:|FAILED' "$BUILD_A/sx-build.log") error lines ($BUILD_A/sx-build.log)"
    # the checks include E2's layer with the KV cache in the provider's buffer and E3's per-step comparisons
    release
    "$BUILD_A/bin/flagos-check-spacemit" --expect-device > "$BUILD_A/sx-check.out" 2> "$BUILD_A/sx-check.err"
    echo "   flagos-check-spacemit: rc=$?: $(tail -1 "$BUILD_A/sx-check.out")"
    grep -E "^layer|^attn     split" "$BUILD_A/sx-check.out" | grep -oE "KV cache in our buffer[^;]*|one launch per step[^;]*" |
        sed 's/^/   /'
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
            *DEVICE_TYPE=gpu*) type=GPU ;;
            *) type=ACCEL ;;
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
        echo "== E2-neg (expected: the igpu runs abort with \"cannot run the operation\", the ref runs end with rc 0; the"
        echo "   shift runs have a 256-token context, so more tokens generated than that means the context was shifted)"
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
    for b in "$BUILD_A" "$BUILD_B"; do
        [ -n "$DRY" ] || [ -x "$b/bin/llama-bench" ] || { echo "not built: $b (structural-experiments.sh prepare)"; exit 1; }
    done
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
        place+=" e3" bench_modes+=" e3" depth_modes+=" e3" kld_modes+=" e3"
    fi
    if has e2hp; then
        case " $depth_modes " in *" e2-igpu "*) ;; *) depth_modes+=" e2-igpu" ;; esac
        depth_modes+=" ref-nohp e2-igpu-nohp"
    fi
    if [ -z "$DRY" ]; then
        {
            echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
            echo "model: $M; EXPS: $EXPS; REPS: $REPS; DEPTH: $DEPTH; PPL_TEXT: $PPL_TEXT; CHUNKS: $CHUNKS"
            echo "A: $BUILD_A at $(g -C "$REPO" rev-parse --short HEAD)"
            echo "B: $BUILD_B at $(g -C "$WT_B" rev-parse --short HEAD), changed: $(g -C "$WT_B" diff --stat | tail -1)"
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

    if [ -n "$PPL_TEXT" ] && { [ -n "$DRY" ] || [ -f "$PPL_TEXT" ]; }; then
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
