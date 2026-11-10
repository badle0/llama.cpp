#!/usr/bin/env bash
# T1 step 1 on the K3 (plan.md "T1 spec: the tiled kernel below 64 tokens"): the build, the checks, then
# FLAGOS_SPACEMIT_ATTN_TILED=64 (today's kernels) against 9 (the tiled kernel from 9 tokens per call) in the check
# program's benchmark, llama-bench and the KLD against the CPU base. The decision rule is applied by hand to the
# summary. Run in tmux after the /dev/tcm check (PROJECT.md section 6); this script does not check /dev/tcm.
#
# Usage: t1-experiments.sh run [out]       the steps of $STEPS in order, logs in out (default ~/t1), then the summary;
#                                          a failed check stops the run before the later steps
#        t1-experiments.sh summary <out>   the summary again from the logs
# Steps (STEPS, default "build check bench llama kld"):
#   build   cmake --build $BUILD with the runner's targets (tools/spacemit_check.py, build())
#   check   flagos-check-spacemit --expect-device; test-backend-ops FLASH_ATTN_EXT unset and under
#           FLAGOS_SPACEMIT_ATTN_KVH=off FLAGOS_SPACEMIT_ATTN_TILED=1 (each: rc 0 and N/N tests passed, N > 0)
#   bench   flagos-check-spacemit --expect-device --bench-attn (its TILED lines: 64 against 9, two turns each)
#   llama   llama-bench per model, turn and setting, the settings in order on odd turns and reversed on even ones
#   kld     llama-perplexity --kl-divergence against $KLD_BASE per ubatch and setting; the base is never written
#           (both --kl-divergence-base and --kl-divergence on every run), its size and sha256 recorded before and after
# Env:   BUILD      the build (default <repo>/build-flagos; a relative path is taken from the repo)
#        MODELS     llama-bench models (default "~/models/Qwen3-4B-Q4_0.gguf ~/models/Qwen3-0.6B-Q4_0.gguf")
#        SETTINGS   FLAGOS_SPACEMIT_ATTN_TILED values (default "64 9"; the first is the base of the changes)
#        TURNS      llama-bench turns (default 2); REPS its -r (default 3)
#        PROMPTS    its -p (default 9,12,16,26,32,38,48,63,64,512); DEPTHS its -d (default 0,1024,4096)
#        UBS        KLD ubatch sizes (default "9 26")
#        KLD_MODEL  (default ~/models/Qwen3-4B-Q4_0.gguf); KLD_TEXT (default ~/kvh-wiki.test.raw)
#        KLD_BASE   the CPU base, read only (default ~/kvh/base1.kld); KLD_CTX -c and -b (default 4096)
#        STEPS      as above; TIMEOUT seconds per command (default 3600); DRY=1 prints the commands instead
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../.." && pwd)}
BUILD=${BUILD:-$REPO/build-flagos}
case $BUILD in /*) ;; *) BUILD=$REPO/$BUILD ;; esac
MODELS=${MODELS:-$HOME/models/Qwen3-4B-Q4_0.gguf $HOME/models/Qwen3-0.6B-Q4_0.gguf}
SETTINGS=${SETTINGS:-64 9}
TURNS=${TURNS:-2}
REPS=${REPS:-3}
PROMPTS=${PROMPTS:-9,12,16,26,32,38,48,63,64,512}
DEPTHS=${DEPTHS:-0,1024,4096}
UBS=${UBS:-9 26}
KLD_MODEL=${KLD_MODEL:-$HOME/models/Qwen3-4B-Q4_0.gguf}
KLD_TEXT=${KLD_TEXT:-$HOME/kvh-wiki.test.raw}
KLD_BASE=${KLD_BASE:-$HOME/kvh/base1.kld}
KLD_CTX=${KLD_CTX:-4096}
STEPS=${STEPS:-build check bench llama kld}
TIMEOUT=${TIMEOUT:-3600}
DRY=${DRY:-}
DEVICE=FlagOS:SpacemiT:0
# the runner's targets, so that the plugin is built as the runner builds it
TARGETS="ggml-flagos flagos-check-provider flagos-check-target flagos-check-graph-plan flagos-check-registry
    flagos-check-spacemit test-backend-ops llama-completion llama-bench llama-perplexity llama-server"
# --foreground: as structural-experiments.sh, so that a tool never runs outside the terminal's foreground
TO=""
if command -v timeout > /dev/null; then TO="timeout --foreground $TIMEOUT"; fi
# the switches of build.md section 7 and SpacemiT's TCM switch: every run sets its own, nothing else may change them
unset FLAGOS_SPACEMIT_ATTN FLAGOS_SPACEMIT_ATTN_KVH FLAGOS_SPACEMIT_ATTN_TILED FLAGOS_SPACEMIT_DISABLE \
    FLAGOS_SPACEMIT_MM FLAGOS_SPACEMIT_PLAN FLAGOS_SPACEMIT_QUANTIZE GGML_FLAGOS_KERNEL_VARIANTS \
    GGML_FLAGOS_SYNC_LAUNCH GGML_FLAGOS_STATS GGML_FLAGOS_DISABLE_OPS GGML_FLAGOS_TRACE_LAUNCH GGML_FLAGOS_OP_STATS \
    GGML_FLAGOS_PROVIDER_PATH FLAGOS_SPACEMIT_TEST_TCM FLAGOS_SPACEMIT_TEST_THREADS FLAGOS_SPACEMIT_TEST_FAIL_NODE \
    SPACEMIT_DISABLE_TCM

has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
g() { git -c safe.directory="$REPO" "$@"; }  # the board's checkout belongs to another uid (structural-experiments.sh)
rev() { local x out=""; for x in $1; do out="$x${out:+ $out}"; done; echo "$out"; }
sha() { if command -v sha256sum > /dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
base_record() { echo "$1: $(wc -c < "$KLD_BASE" | tr -d ' ') bytes, sha256 $(sha "$KLD_BASE")"; }

# run <stdout file> <command...>: one line with the time first (tail -f shows progress), no terminal input; stderr to
# the file's .err, the exit code to its .rc
run() {
    local out=$1 rc
    shift
    echo "$(date +%T) $(basename "${out%.*}"): $*"
    [ -z "$DRY" ] || return 0
    # shellcheck disable=SC2086 # $TO is a list of words
    $TO "$@" < /dev/null > "$out" 2> "${out%.*}.err"
    rc=$?
    echo "$rc" > "${out%.*}.rc"
    return "$rc"
}

# rc 0 and "N/N tests passed" with N > 0, as the runner's op_test; prints the reason otherwise
op_ok() {
    local log=$1 rc passed n total
    rc=$(cat "$log.rc" 2> /dev/null)
    passed=$(sed 's/\x1b\[[0-9;]*m//g' "$log.out" | grep -oE '[0-9]+/[0-9]+ tests passed' | tail -1)
    n=${passed%%/*} total=${passed#*/}
    total=${total%% *}
    if [ "$rc" = 0 ] && [ -n "$passed" ] && [ "$n" = "$total" ] && [ "$n" -gt 0 ]; then
        return 0
    fi
    echo "$(basename "$log"): rc=${rc:-?}, ${passed:-no summary}"
    return 1
}

step_build() {
    [ -n "$DRY" ] || [ -f "$BUILD/CMakeCache.txt" ] ||
        { echo "not configured: $BUILD (tools/spacemit_check.py --build configures it)"; exit 1; }
    # shellcheck disable=SC2086 # $TARGETS is a list of words
    run "$OUT/build.out" cmake --build "$BUILD" --target $TARGETS
    local rc=$?
    [ -n "$DRY" ] && return 0
    if [ "$rc" != 0 ]; then
        echo "build failed: rc=$rc; $(cat "$OUT/build.out" "$OUT/build.err" | grep -E 'error:|FAILED' | head -1)"
        exit 1
    fi
}

step_check() {
    local why=""
    run "$OUT/check.out" "$BUILD/bin/flagos-check-spacemit" --expect-device ||
        why+="flagos-check-spacemit rc=$(cat "$OUT/check.rc" 2> /dev/null): $(grep -m1 FAIL "$OUT/check.err"); "
    run "$OUT/tbo-unset.out" "$BUILD/bin/test-backend-ops" test -o FLASH_ATTN_EXT -b "$DEVICE"
    run "$OUT/tbo-tiled1.out" env FLAGOS_SPACEMIT_ATTN_KVH=off FLAGOS_SPACEMIT_ATTN_TILED=1 \
        "$BUILD/bin/test-backend-ops" test -o FLASH_ATTN_EXT -b "$DEVICE"
    [ -n "$DRY" ] && return 0
    why+=$(op_ok "$OUT/tbo-unset")
    why+=$(op_ok "$OUT/tbo-tiled1")
    if [ -n "$why" ]; then
        echo "check failed, no later step runs: $why"
        exit 1
    fi
    echo "$(date +%T) check passed"
}

step_bench() {
    run "$OUT/bench-attn.out" "$BUILD/bin/flagos-check-spacemit" --expect-device --bench-attn ||
        echo "bench: rc=$(cat "$OUT/bench-attn.rc" 2> /dev/null)"
}

step_llama() {
    local m name turn order s
    for m in $MODELS; do
        name=$(basename "$m" .gguf)
        if [ -z "$DRY" ] && [ ! -f "$m" ]; then
            echo "model not found, skipped: $m"
            continue
        fi
        for turn in $(seq "$TURNS"); do
            order=$SETTINGS
            if [ $((turn % 2)) = 0 ]; then order=$(rev "$SETTINGS"); fi
            for s in $order; do
                run "$OUT/llama-$name-$s-$turn.csv" env FLAGOS_SPACEMIT_ATTN_TILED="$s" "$BUILD/bin/llama-bench" \
                    -m "$m" -t 8 -fa 1 -mmp 0 -r "$REPS" -p "$PROMPTS" -n 32 -d "$DEPTHS" -o csv
            done
        done
    done
}

step_kld() {
    local ub s f
    if [ -z "$DRY" ]; then
        for f in "$KLD_BASE" "$KLD_MODEL" "$KLD_TEXT"; do
            [ -f "$f" ] || { echo "kld not started: $f not found"; return 1; }
        done
        base_record before > "$OUT/kld-base.txt"
    fi
    for ub in $UBS; do
        for s in $SETTINGS; do
            # --kl-divergence with the base: read only (without it llama-perplexity would write the base)
            run "$OUT/kld-ub$ub-$s.out" env FLAGOS_SPACEMIT_ATTN_TILED="$s" "$BUILD/bin/llama-perplexity" \
                -m "$KLD_MODEL" -f "$KLD_TEXT" -c "$KLD_CTX" -b "$KLD_CTX" -ub "$ub" --chunks 1 -fa on -t 8 \
                --kl-divergence-base "$KLD_BASE" --kl-divergence
        done
    done
    [ -n "$DRY" ] && return 0
    base_record after >> "$OUT/kld-base.txt"
    cat "$OUT/kld-base.txt"
}

summary() {
    [ -d "$OUT" ] || { echo "no logs: $OUT"; exit 1; }
    python3 -I - "$OUT" << 'EOF'
import csv, glob, os, re, sys

out = sys.argv[1]


def read(path):
    try:
        with open(path, errors="replace") as f:
            return re.sub(r"\x1b\[[0-9;]*m", "", f.read())
    except OSError:
        return ""


def rc(stem):
    return read(stem + ".rc").strip() or "?"


info = read(os.path.join(out, "info.txt"))
m = re.search(r"^settings: (.*)$", info, re.M)
settings = m.group(1).split() if m else []
print(info.strip())

print("\n== 1. check")
if os.path.exists(os.path.join(out, "build.rc")):
    text = read(os.path.join(out, "build.out")) + read(os.path.join(out, "build.err"))
    n = sum(("error:" in ln or "FAILED" in ln) for ln in text.splitlines())
    print(f"   build: rc {rc(os.path.join(out, 'build'))}, {n} error lines")
stem = os.path.join(out, "check")
if os.path.exists(stem + ".rc"):
    text = read(stem + ".out")
    last = text.strip().splitlines()[-1] if text.strip() else "no output"
    print(f"   flagos-check-spacemit --expect-device: rc {rc(stem)}: {last}")
    for ln in text.splitlines():
        if ln.startswith("attn "):
            print("   " + ln)
    for ln in read(stem + ".err").splitlines():
        if ln.startswith("FAIL"):
            print("   " + ln)
for tag, what in (("unset", "unset"), ("tiled1", "FLAGOS_SPACEMIT_ATTN_KVH=off FLAGOS_SPACEMIT_ATTN_TILED=1")):
    stem = os.path.join(out, "tbo-" + tag)
    if os.path.exists(stem + ".rc"):
        text = read(stem + ".out")
        passed = re.findall(r"\d+/\d+ tests passed", text)
        verdict = re.findall(r"Backend FlagOS:SpacemiT:0: (\w+)", text)
        print(f"   test-backend-ops FLASH_ATTN_EXT, {what}: rc {rc(stem)}, "
              f"{passed[-1] if passed else 'no summary'}, {verdict[-1] if verdict else '?'}")

print("\n== 2. bench: one call, us (best of two turns), FLAGOS_SPACEMIT_ATTN_TILED 64 against 9; the kernels by the "
      "counter")
pat = re.compile(r"bench +attn (\d+) tokens n_kv +(\d+), FLAGOS_SPACEMIT_ATTN_TILED: 64 +([\d.]+) us "
                 r"\( *([\d.]+) GB/s, ([\w-]+)\); 9 +([\d.]+) us \(([+-]?[\d.]+)%, +([\d.]+) GB/s, ([\w-]+)\)")
rows = [m.groups() for m in pat.finditer(read(os.path.join(out, "bench-attn.out")))]
if rows:
    print(f"   {'tokens':>6} {'n_kv':>5} {'us 64':>9} {'us 9':>9} {'change':>8}  {'kernel 64':<10} {'kernel 9':<10}")
    for tok, kv, t64, _, k64, t9, ch, _, k9 in sorted(rows, key=lambda r: (int(r[0]), int(r[1]))):
        print(f"   {tok:>6} {kv:>5} {float(t64):9.0f} {float(t9):9.0f} {float(ch):+7.1f}%  {k64:<10} {k9:<10}")
else:
    print(f"   no TILED lines (rc {rc(os.path.join(out, 'bench-attn'))})")

print("\n== 3. llama-bench t/s, best of the turns, per setting, and the change against the first setting")
runs = {}
for path in sorted(glob.glob(os.path.join(out, "llama-*.csv"))):
    model, s, turn = os.path.basename(path)[len("llama-"):-len(".csv")].rsplit("-", 2)
    for r in csv.DictReader(read(path).splitlines()):
        try:
            p, n, d, ts = int(r["n_prompt"]), int(r["n_gen"]), int(r["n_depth"]), float(r["avg_ts"])
        except (KeyError, ValueError):
            continue
        test = f"pp{p}" if n == 0 else f"tg{n}" if p == 0 else f"pp{p}+tg{n}"
        key = (model, test, d, s)
        runs[key] = max(runs.get(key, 0.0), ts)
for model in sorted({k[0] for k in runs}):
    sets = [s for s in settings if any(k[0] == model and k[3] == s for k in runs)] or \
           sorted({k[3] for k in runs if k[0] == model}, key=lambda x: -int(x) if x.isdigit() else 0)
    tests = sorted({k[1] for k in runs if k[0] == model},
                   key=lambda t: (t.startswith("tg"), int(re.sub(r"\D.*", "", t[2:]) or 0)))
    depths = sorted({k[2] for k in runs if k[0] == model})
    width = 9 * len(sets) + 9
    print(f"   {model}: cells " + " / ".join(f"t/s under {s}" for s in sets) + f", change under {' / '.join(sets[1:])}")
    print(f"   {'test':<8}" + "".join(f"{'depth ' + str(d):>{width}}" for d in depths))
    for t in tests:
        cells = ""
        for d in depths:
            v = [runs.get((model, t, d, s)) for s in sets]
            cell = "".join(f"{x:9.2f}" if x else f"{'-':>9}" for x in v)
            change = [f"{100 * (x / v[0] - 1):+.1f}%" if x and v[0] else "-" for x in v[1:]]
            cells += f"{cell + ' ' + '/'.join(change):>{width}}"
        print(f"   {t:<8}" + cells)
if not runs:
    print("   no llama-bench rows")

print("\n== 4. KLD against the base, per ubatch and setting")
print("   " + "\n   ".join(read(os.path.join(out, "kld-base.txt")).strip().splitlines() or ["no base record"]))
klds = []
for path in glob.glob(os.path.join(out, "kld-ub*.out")):
    m = re.match(r"kld-ub(\d+)-(.+)\.out$", os.path.basename(path))
    if m:
        klds.append((int(m.group(1)), m.group(2), path[:-len(".out")]))
order = {s: i for i, s in enumerate(settings)}
for ub, s, stem in sorted(klds, key=lambda k: (k[0], order.get(k[1], 99), k[1])):
    text = (read(stem + ".out") + read(stem + ".err")).replace(chr(0xb1), "+/-")
    def grab(pattern):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else "-"
    kld = grab(r"Mean +KLD: *(-?[\d.]+ \+/- +[\d.]+)")
    ratio = grab(r"Mean PPL\(Q\)/PPL\(base\) *: *([\d.]+ \+/- +[\d.]+)")
    top = grab(r"Same top p: *([\d.]+ \+/- +[\d.]+ %)")
    secs = grab(r"([\d.]+) seconds per pass")
    print(f"   -ub {ub:<3} {s:>3}: rc {rc(stem)}, mean KLD {re.sub(' +', ' ', kld)}, PPL(Q)/PPL(base) "
          f"{re.sub(' +', ' ', ratio)}, same top p {re.sub(' +', ' ', top)}, {secs} s per pass")
if not klds:
    print("   no KLD runs")
print(f"\nlogs: {out}")
EOF
}

run_all() {
    OUT=${1:-$HOME/t1}
    [ -n "$DRY" ] || mkdir -p "$OUT" || exit 1
    if [ -z "$DRY" ]; then
        {
            echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
            echo "build: $BUILD at $(g -C "$REPO" rev-parse --short HEAD 2> /dev/null)"
            echo "steps: $STEPS; models: $MODELS; turns: $TURNS; reps: $REPS; prompts: $PROMPTS; depths: $DEPTHS"
            echo "ubs: $UBS; kld: $KLD_MODEL on $KLD_TEXT, -c $KLD_CTX, base $KLD_BASE"
            echo "settings: $SETTINGS"
        } > "$OUT/info.txt"
    fi
    local step
    for step in build check bench llama kld; do
        if has "$step"; then
            echo "$(date +%T) == $step"
            "step_$step"
        fi
    done
    [ -n "$DRY" ] || summary
}

case ${1:-} in
    run) shift; run_all "$@" ;;
    summary) OUT=${2:?usage: t1-experiments.sh summary <out>}; summary ;;
    *) sed -n '2,26p' "$0"; exit 1 ;;
esac
