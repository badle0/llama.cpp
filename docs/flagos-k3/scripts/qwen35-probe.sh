#!/usr/bin/env bash
# The Qwen3.5 probe (PROJECT.md section 8, item 5 of the to-do order): what a Qwen3.5 file does on the provider.
# Given a model file, with the provider only, in order:
#   gen    llama-completion at -lv 4, once with -fa auto (llama's default) and once with -fa on (gen-auto.out,
#          gen-on.out): does it load and generate, how many graph splits, which buffers hold the weights
#   sched  llama-completion with GGML_SCHED_DEBUG=2 at -lv 5 (sched.out): the ops of the last graph, a one-token
#          decode, by backend
#   cpu    only with CPU=1: the same prompt with FLAGOS_SPACEMIT_DISABLE=1 on ONE thread (cpu.out), for the text.
#          Never more than one thread: a CPU run with several threads on the X100 cores is wrong at random for a model
#          with Q4_1 tensors (PROJECT.md section 6, item 28)
# The summary gives the graph split counts, the CPU/provider op histogram of the one-token decode and llama's model
# buffer lines. The script builds nothing and does not touch /dev/tcm; run the /dev/tcm check of PROJECT.md section 6
# before it, and keep it apart from IME runs with TCM.
#
# A hybrid model (Qwen3.5) with the recurrent state in the provider's buffer stops llama.cpp at context creation:
# "pre-allocated tensor (cache_r_l0 ...) in a buffer (FlagOS:SpacemiT) that cannot run the operation (SCALE)"
# (ggml/src/ggml-backend.cpp:930; the SCALE that clears a state slot, src/llama-graph.cpp:3362). NOKVOFF=1 adds
# --no-kv-offload, which puts the recurrent state and the KV cache in CPU buffers and lets the run finish. Run the
# script twice: without it for the default path, with it for the numbers (seen on the Mac's host-test device,
# 2026-11-12).
#
# usage: qwen35-probe.sh run <model.gguf> [out]   (default out: ~/q35-probe)
#        qwen35-probe.sh summary <out>
# env:   BUILD (default <repo>/build-flagos), THREADS (8), NGEN (16), NCTX (2048), PROMPT, TIMEOUT (seconds per
#        command, 3600), NOKVOFF=1 (adds --no-kv-offload to every run), CPU=1 (adds the one-thread CPU reference),
#        STEPS (default "gen sched"; cpu is added by CPU=1), DRY=1 (print the commands only)
set -u

# an abort (the default path's SCALE one) must end the process: without this ggml starts gdb for a backtrace and waits
# for it, and on the K3 that wait did not end (2026-11-13)
export GGML_NO_BACKTRACE=1

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
BUILD=${BUILD:-$REPO/build-flagos}
THREADS=${THREADS:-8}
NGEN=${NGEN:-16}
NCTX=${NCTX:-2048}
PROMPT=${PROMPT:-"The capital of France is"}
TIMEOUT=${TIMEOUT:-3600}
CPU=${CPU:-}
NOKVOFF=${NOKVOFF:-}
STEPS=${STEPS:-"gen sched"}
DRY=${DRY:-}
KVARG=""
[ -n "$NOKVOFF" ] && KVARG="--no-kv-offload"

unset FLAGOS_SPACEMIT_DISABLE GGML_FLAGOS_DISABLE_OPS GGML_FLAGOS_KERNEL_VARIANTS GGML_SCHED_DEBUG

step_on() { # step_on <step>: whether STEPS names it
    case " $STEPS " in *" $1 "*) return 0 ;; esac
    return 1
}

run() { # run <out file> <command...>: stdout to the file, stderr beside it, the return code in .rc
    local out=$1
    shift
    echo "$(date +%H:%M:%S) $*"
    [ -n "$DRY" ] && return 0
    timeout -k 10 "$TIMEOUT" "$@" > "$out" 2> "${out%.*}.err"
    echo $? > "${out%.*}.rc"
}

summary() {
    python3 - "$1" <<'PY'
import collections, os, re, sys

out = sys.argv[1]
info = os.path.join(out, "info.txt")
if os.path.exists(info):
    print(open(info, errors="replace").read().rstrip())

def read(name):
    text = ""
    for suffix in (".out", ".err"):
        path = os.path.join(out, name + suffix)
        if os.path.exists(path):
            text += open(path, errors="replace").read()
    rc_path = os.path.join(out, name + ".rc")
    rc = open(rc_path).read().strip() if os.path.exists(rc_path) else "?"
    return rc, text

ANSI = re.compile(r"\x1b\[[0-9;]*m")

def splits(text):
    found = re.findall(r"graph splits = (\d+)(?: \(with bs=(\d+)\), (\d+) \(with bs=1\))?", text)
    if not found:
        return "no graph splits line"
    batch, bs, one = found[-1]
    if one:
        return "%s with bs=%s, %s with bs=1" % (batch, bs, one)
    return "%s (one count only)" % batch

def buffers(text):
    return [re.sub(r"\s+", " ", m.group(0)).strip() for m in
            re.finditer(r"[A-Za-z0-9_:.]+\s+\w+ buffer size\s*=\s*[0-9.]+ MiB", ANSI.sub("", text))]

def text_of(name):
    path = os.path.join(out, name + ".out")
    if not os.path.exists(path):
        return ""
    return " ".join(ANSI.sub("", open(path, errors="replace").read()).split())

for name, label in (("gen-auto", "gen -fa auto"), ("gen-on", "gen -fa on"), ("cpu", "cpu -t 1")):
    rc, text = read(name)
    if not text and rc == "?":
        continue
    print("%s: rc %s, graph splits %s" % (label, rc, splits(text)))
    lines = [ln for ln in ANSI.sub("", text).splitlines()
             if any(w in ln.lower() for w in ("error", "assert", "cannot run", "abort", "not supported"))]
    for ln in lines[:4]:
        print("   PROBLEM %s" % ln.strip()[:160])
    for b in buffers(text):
        print("   buffer %s" % b)
    body = text_of(name)
    if body:
        print("   text %s" % body[:120])

texts = {n: text_of(n) for n in ("gen-auto", "gen-on", "cpu")}
if texts.get("gen-auto") and texts.get("cpu"):
    same = texts["gen-auto"] == texts["cpu"]
    i = next((i for i, (a, b) in enumerate(zip(texts["gen-auto"], texts["cpu"])) if a != b),
             min(len(texts["gen-auto"]), len(texts["cpu"])))
    print("provider text against the one-thread CPU: %s" %
          ("identical" if same else "differs after %d characters" % i))

rc, text = read("sched")
if text:
    print("sched: rc %s, graph splits %s" % (rc, splits(text)))
    # one block per graph: GGML_SCHED_DEBUG=2 prints "## SPLIT #0" first for every graph it computes
    blocks = ANSI.sub("", text).split("## SPLIT #0:")
    if len(blocks) < 2:
        print("   no scheduler assignments in the log (GGML_SCHED_DEBUG=2 needs -lv 5)")
    else:
        last = "## SPLIT #0:" + blocks[-1]
        per_backend = collections.Counter()
        ops = collections.defaultdict(collections.Counter)
        # the log prefixes every line with a timestamp, so nothing is anchored; a node's own backend is the first
        # [backend cause] field after its op
        for m in re.finditer(r"## SPLIT #(\d+): (\S+)", last):
            per_backend[m.group(2)] += 1
        for m in re.finditer(r"node #\s*\d+ \(\s*([A-Z_0-9]+)\s*\):.*?\[\s*(\S+?)[\s\]]", last):
            ops[m.group(2)][m.group(1)] += 1
        print("   graphs in the log %d; the last one: %d splits (%s)" %
              (len(blocks) - 1, sum(per_backend.values()),
               ", ".join("%s %d" % (k, v) for k, v in sorted(per_backend.items())) or "none"))
        for backend in sorted(ops):
            total = sum(ops[backend].values())
            print("   %-6s %4d nodes: %s" % (backend, total,
                  ", ".join("%s %d" % (op, n) for op, n in ops[backend].most_common())))
        print("   (op and backend names are the scheduler's, cut to 10 and 5 characters)")
print("logs: %s" % out)
PY
}

case "${1:-}" in
    run)
        MODEL=${2:-}
        OUT=${3:-$HOME/q35-probe}
        [ -n "$MODEL" ] || { echo "usage: $0 run <model.gguf> [out]"; exit 2; }
        [ -n "$CPU" ] && STEPS="$STEPS cpu"
        if [ -z "$DRY" ]; then
            [ -f "$MODEL" ] || { echo "no model $MODEL"; exit 1; }
            [ -x "$BUILD/bin/llama-completion" ] || { echo "no $BUILD/bin/llama-completion: build it first"; exit 1; }
            mkdir -p "$OUT"
            { echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
              echo "build: $BUILD at $(git -c safe.directory="$REPO" -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
              echo "model: $MODEL ($(wc -c < "$MODEL") bytes)"
              echo "steps: $STEPS; threads: $THREADS; n_gen: $NGEN; n_ctx: $NCTX"; } > "$OUT/info.txt"
        fi
        if step_on gen; then
            run "$OUT/gen-auto.out" "$BUILD/bin/llama-completion" \
                -m "$MODEL" -p "$PROMPT" -n "$NGEN" -c "$NCTX" -t "$THREADS" --temp 0 -s 1 -no-cnv -lv 4 $KVARG
            run "$OUT/gen-on.out" "$BUILD/bin/llama-completion" \
                -m "$MODEL" -p "$PROMPT" -n "$NGEN" -c "$NCTX" -t "$THREADS" --temp 0 -s 1 -no-cnv -lv 4 -fa on $KVARG
        fi
        if step_on sched; then
            run "$OUT/sched.out" env GGML_SCHED_DEBUG=2 "$BUILD/bin/llama-completion" \
                -m "$MODEL" -p "$PROMPT" -n 2 -c "$NCTX" -t "$THREADS" --temp 0 -s 1 -no-cnv -lv 5 $KVARG
        fi
        if step_on cpu; then
            run "$OUT/cpu.out" env FLAGOS_SPACEMIT_DISABLE=1 "$BUILD/bin/llama-completion" \
                -m "$MODEL" -p "$PROMPT" -n "$NGEN" -c "$NCTX" -t 1 --temp 0 -s 1 -no-cnv -lv 4 $KVARG
        fi
        [ -n "$DRY" ] || summary "$OUT"
        ;;
    summary)
        [ -n "${2:-}" ] || { echo "usage: $0 summary <out>"; exit 2; }
        summary "$2"
        ;;
    *)
        echo "usage: $0 run <model.gguf> [out] | summary <out>"
        exit 2
        ;;
esac
