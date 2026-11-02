#!/usr/bin/env python3
# Acceptance run for the SpacemiT provider (plan.md §5). Run on the K3; standard library only. It checks:
#   (--build) configure and build build-flagos/; FlagOS Common's checks; flagos-check-spacemit with the device present
#   (with --full: Qwen3-4B's shapes), with the reference kernels on the AI cores, with FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=
#   accel (which must report ACCEL) and with FLAGOS_SPACEMIT_DISABLE=1 (no device); the plugin loaded through the
#   loader; llama.cpp lists the device; the provider claims exactly ADD, MUL, RMS_NORM, ROPE, SET_ROWS, GET_ROWS,
#   SWIGLU, MUL_MAT (Q4_0, Q4_1, Q8_0 and Q6_K weights in their IME layouts), FLASH_ATTN_EXT (F16 K and V, head sizes
#   up to 128) and CPY (F32 into the quantized and BF16 cache types), and test-backend-ops passes for each and for the
#   multi-op graphs built from them, with the RVV and with the reference kernels; the benchmark of the check program;
#   a model run with the provider and with FLAGOS_SPACEMIT_DISABLE=1 (the IME kernels quantize activations differently
#   from the CPU, so the text is reported, not compared): the weights, every layer and the KV cache on the provider
#   (IGPU, D3), -fa auto keeping flash attention there (FAIL with llama.cpp's warning if it turns it off, or if the
#   graphs take as many splits as the model has layers, as when the CPU runs attention), and a token in 2 graph splits;
#   a Q8_0 KV cache and a context shift with the provider as with the CPU (structural-experiments.md E2-neg, F8);
#   perplexity with -fa on and with -fa auto, each within 1% of the CPU (the CPU single-threaded, PROJECT.md §6); mean
#   KLD against the CPU's logits with 512-token batches (the tiled kernel) and -ub 1 -b 512 (one token per ubatch: the
#   1-row matmuls, from n_kv 512 the split over the tiles), to compare by hand with build.md §7; llama-bench pp128 and
#   tg128 for the provider, the CPU and (if built) the upstream IME build; no TCM block left held.
#   --depth: after the other llama-bench runs, tg32 at depths 0, 1024 and 4096 and pp512 at depth 4096 for the same
#   modes; FAIL if the provider's runs miss one
# Usage: spacemit_check.py [--build] [--build-dir DIR] [--model GGUF] [--ppl-text FILE] [--ime-build DIR]
#                          [--skip-support] [--depth] [--segv SO]
import argparse
import csv
import io
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
DEVICE = "FlagOS:SpacemiT:0"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# the ops the provider claims (GLU cases are named by their GLU op), and the test-backend-ops graphs that chain them
OPS = ["ADD", "MUL", "RMS_NORM", "ROPE", "SET_ROWS", "GET_ROWS", "SWIGLU", "MUL_MAT", "FLASH_ATTN_EXT", "CPY"]
GRAPHS = ["RMS_NORM_MUL_ADD", "ADD_RMS_NORM", "RMS_NORM_MUL_ROPE", "ROPE_SET_ROWS"]
results = []
preload = {}  # LD_PRELOAD for the model runs (--segv)
summary = None  # <logs>/summary.txt: every line printed by report and info (the terminal's scrollback is short)


def say(line):
    print(line, flush=True)
    if summary is not None:
        summary.write(line + "\n")
        summary.flush()


def report(name, ok, detail):
    results.append(ok)
    say(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def info(name, detail):
    say(f"[INFO] {name}: {detail}")


def run(cmd, log, env_extra=None, env_drop=(), timeout=900):
    """Run cmd, write stdout and stderr to log files, return (returncode, stdout, stderr)."""
    env = {k: v for k, v in os.environ.items() if k not in env_drop}
    env.update(env_extra or {})
    try:
        p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, env=env, timeout=timeout,
                           errors="replace")
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = -1, e.stdout or "", e.stderr or ""
        out = out if isinstance(out, str) else out.decode(errors="replace")
        err = (err if isinstance(err, str) else err.decode(errors="replace")) + f"\ntimeout after {timeout} s\n"
    except FileNotFoundError:
        rc, out, err = -1, "", f"not found: {cmd[0]}\n"
    Path(f"{log}.out").write_text(out)
    Path(f"{log}.err").write_text(err)
    return rc, out, err


def last_line(text):
    lines = text.strip().splitlines()
    return lines[-1] if lines else "no output"


def read_log(log):
    """The stdout lines run() wrote for this log, or none if the file is not there."""
    try:
        return Path(f"{log}.out").read_text().splitlines()
    except OSError:
        return []


def build(bdir, logs, spert_dir):
    cfg = ["cmake", "-S", str(REPO), "-B", str(bdir), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_FLAGOS=ON",
           "-DGGML_FLAGOS_DENGLIN=OFF", "-DGGML_FLAGOS_AMD=OFF", "-DGGML_FLAGOS_SPACEMIT=ON"]
    if spert_dir:
        cfg.append(f"-DFLAGOS_SPACEMIT_SPERT_DIR={spert_dir}")
    rc, out, err = run(cfg, logs / "configure", timeout=600)
    enabled = next((ln for ln in out.splitlines() if "enabling SpacemiT K3 provider" in ln), "")
    report("configure", rc == 0 and bool(enabled), enabled.replace("-- ", "") if enabled else f"rc={rc}: {last_line(err)}")
    if rc != 0:
        return False
    targets = ["ggml-flagos", "flagos-check-provider", "flagos-check-target", "flagos-check-graph-plan",
               "flagos-check-registry", "flagos-check-spacemit", "test-backend-ops", "llama-completion", "llama-bench",
               "llama-perplexity"]
    rc, out, err = run(["cmake", "--build", str(bdir), "--target", *targets], logs / "build", timeout=3600)
    errors = [ln for ln in (out + err).splitlines() if "error:" in ln or "FAILED" in ln]
    report("build", rc == 0 and not errors, f"rc={rc}, {len(errors)} error lines" + (f": {errors[0][:120]}" if errors else ""))
    return rc == 0


def tcm_tool(tcm_dir, name):
    path = Path(tcm_dir) / name
    return str(path) if path.is_file() and os.access(path, os.X_OK) else None


def check_spacemit(bdir, logs, name, log, args, env_extra=None, timeout=300):
    # a hang here means tiles did not stop together after a failure
    rc, out, err = run([str(bdir / "bin" / "flagos-check-spacemit"), "--expect-device", *args], logs / log,
                       env_extra={**(env_extra or {}), **preload}, timeout=timeout)
    detail = "; ".join(ln for ln in out.splitlines()
                       if ln.startswith(("device", "flagos ", "loader ", "ops ", "matmul   Q4_0", "matmul   Q4_1",
                                         "matmul   Q8_0", "matmul   Q6_K", "gemv ", "rows ", "cache ", "attn ",
                                         "layer")))
    fails = [ln for ln in err.splitlines() if ln.startswith("FAIL") or "timeout" in ln]
    report(name, rc == 0, detail if rc == 0 else "; ".join(fails[:3]) or last_line(err))
    for ln in out.splitlines():
        if ln.startswith("info "):
            info(name, ln[len("info "):].strip())


def check_tools(bdir, logs):
    for tool in ["provider", "target", "graph-plan", "registry"]:
        rc, out, err = run([str(bdir / "bin" / f"flagos-check-{tool}")], logs / f"check-{tool}")
        report(f"flagos-check-{tool}", rc == 0, last_line(out) if rc == 0 else f"{last_line(err or out)} (rc={rc})")
    check_spacemit(bdir, logs, "flagos-check-spacemit (device present, Qwen3-4B shapes)", "check-spacemit", ["--full"],
                   timeout=900)
    # the same with the reference kernels on the AI cores: tells a kernel error from a tiling error
    check_spacemit(bdir, logs, "flagos-check-spacemit (reference kernels)", "check-spacemit-reference", [],
                   env_extra={"FLAGOS_SPACEMIT_TEST_REFERENCE": "1"}, timeout=900)
    # S1 step 1: the core reaches the device through a plugin table it loaded with dlopen; the check program's loader
    # line names the library it came from
    loader = next((ln for ln in read_log(logs / "check-spacemit") if ln.startswith("loader ")), "")
    path = loader[len("loader "):].split(": api ", 1)[0].strip()
    report("plugin loaded through the loader", bool(path) and Path(path).is_file(), loader or "no loader line")
    # the test-only switch back to ACCEL, the type until 2026-10-26: the checks pass and the device says so
    rc, out, err = run([str(bdir / "bin" / "flagos-check-spacemit"), "--expect-device"], logs / "check-spacemit-accel",
                       env_extra={"FLAGOS_SPACEMIT_TEST_DEVICE_TYPE": "accel", **preload}, timeout=900)
    device = next((ln for ln in out.splitlines() if ln.startswith("device")), "") or "no device line"
    fails = "; ".join(ln for ln in err.splitlines() if ln.startswith("FAIL") or "timeout" in ln)
    report("flagos-check-spacemit (FLAGOS_SPACEMIT_TEST_DEVICE_TYPE=accel)", rc == 0 and "type ACCEL" in device,
           device + ("" if rc == 0 else f"; rc={rc}: {fails[:300] or last_line(err)}"))
    rc, out, _ = run([str(bdir / "bin" / "flagos-check-spacemit"), "--expect-none"], logs / "check-spacemit-disabled",
                     env_extra={"FLAGOS_SPACEMIT_DISABLE": "1", **preload})
    report("flagos-check-spacemit (FLAGOS_SPACEMIT_DISABLE=1)", rc == 0, "device absent" if rc == 0 else f"rc={rc}")


def list_devices(bdir, logs):
    rc, out, err = run([str(bdir / "bin" / "llama-completion"), "--list-devices"], logs / "list-devices")
    line = next((ln.strip() for ln in (out + err).splitlines() if DEVICE in ln), "")
    report("llama.cpp device list", bool(line), line or f"{DEVICE} not listed (rc={rc})")


def support(bdir, logs):
    rc, out, _ = run([str(bdir / "bin" / "test-backend-ops"), "support", "-b", DEVICE], logs / "support", timeout=1800)
    lines = ANSI.sub("", out).splitlines()
    claimed = [ln.strip() for ln in lines if ln.rstrip().endswith(" SUPPORTED") and "NOT SUPPORTED" not in ln]
    n_no = sum("NOT SUPPORTED" in ln for ln in lines)
    ops = sorted({ln.split("(")[0] for ln in claimed})
    report("test-backend-ops support", rc == 0 and set(ops) == set(OPS),
           f"{len(claimed)} op cases claimed ({', '.join(ops) or 'none'}), {n_no} not claimed (rc={rc})")


def op_test(bdir, logs, op, env_extra=None, tag=""):
    """test-backend-ops for one op name; returns (ok, summary)."""
    rc, out, _ = run([str(bdir / "bin" / "test-backend-ops"), "test", "-o", op, "-b", DEVICE],
                     logs / f"test-{op.lower()}{tag}", env_extra=env_extra, timeout=1800)
    clean = ANSI.sub("", out)
    passed = re.search(r"(\d+)/(\d+) tests passed", clean)
    verdict = re.search(rf"Backend {re.escape(DEVICE)}: (\w+)", clean)
    ok = rc == 0 and passed is not None and passed.group(1) == passed.group(2) and int(passed.group(2)) > 0
    return ok, f"{passed.group(0) if passed else 'no summary'}, {verdict.group(1) if verdict else '?'} (rc={rc})"


def op_correctness(bdir, logs):
    for op in OPS + GRAPHS:
        report(f"test-backend-ops {op}", *op_test(bdir, logs, op))
    # the same with the reference kernels on the AI cores, in one line
    failed = [op for op in OPS + GRAPHS
              if not op_test(bdir, logs, op, {"FLAGOS_SPACEMIT_TEST_REFERENCE": "1"}, "-reference")[0]]
    report("test-backend-ops, reference kernels", not failed,
           f"{len(OPS + GRAPHS) - len(failed)}/{len(OPS + GRAPHS)} ops pass" + (f"; failed: {', '.join(failed)}" if failed else ""))


def bench(bdir, logs):
    rc, out, err = run([str(bdir / "bin" / "flagos-check-spacemit"), "--expect-device", "--bench"], logs / "bench",
                       env_extra=preload, timeout=900)
    lines = [ln.split("bench", 1)[1].strip() for ln in out.splitlines() if ln.startswith("bench")]
    report("benchmark ran", rc == 0 and len(lines) >= 3, f"rc={rc}" + ("" if rc == 0 else f": {last_line(err)}"))
    for ln in lines:
        info("benchmark", ln)


def tcm_hygiene(tcm_dir, logs):
    tool = tcm_tool(tcm_dir, "tcmtest")
    if tool is None:
        info("TCM blocks", f"skipped: no tcmtest in {tcm_dir}")
        return
    rc, out, _ = run([tool, "info"], logs / "tcm-info", timeout=120)
    held = sum("is_acquired=1" in ln for ln in out.splitlines())
    report("no TCM block left held", rc == 0 and held == 0, f"{held} of 8 blocks held" + ("" if rc == 0 else f" (rc={rc})"))


def tokens_per_second(log):
    # the generation line, not "prompt eval time"
    m = re.search(r"(?<!prompt )eval time =.*?([0-9.]+) tokens per second", log)
    return float(m.group(1)) if m else None


FA_ENABLED = "Flash Attention enabled"
FA_DISABLED = "Flash Attention not supported, set to disabled"


def flash_attn_auto(log):
    """Outcome of -fa auto (src/llama-context.cpp resolve_fused_ops) in the last context the log creates: ("enabled"
    or "disabled", its log lines), or (None, "") if the log has no such check (flash attention set explicitly, or the
    run failed before it)."""
    lines = [ln.strip() for ln in ANSI.sub("", log).splitlines()]
    # the memory fit pass (common/fit.cpp) creates a context of its own first, logged from -lv 5: the last outcome counts
    last = max((i for i, ln in enumerate(lines) if ln.endswith((FA_ENABLED, FA_DISABLED))), default=None)
    if last is None:
        return None, ""
    if lines[last].endswith(FA_ENABLED):
        return "enabled", lines[last]
    # the warning before it names the devices of the layer and of the op
    why = next((ln for ln in reversed(lines[:last]) if "Flash Attention is assigned to device" in ln), "")
    return "disabled", "; ".join(ln for ln in (why, lines[last]) if ln)


def splits_per_token(log):
    """Graph splits of the 1-token graph in the last context the log creates, or None."""
    found = re.findall(r"graph splits = (\d+)(?: \(with bs=\d+\), (\d+) \(with bs=1\))?", log)
    return int(found[-1][1] or found[-1][0]) if found else None


def attention_on_provider(log, rc):
    """(ok, detail): -fa auto kept flash attention in the last context the log creates, and its graphs (batch and 1
    token) take fewer splits than the model has layers. "enabled" alone holds as well when the provider declines the
    model's FLASH_ATTN_EXT and the CPU runs it (the node's device is then its layer's), which splits every layer in two
    (M2c.2: 74 per Qwen3-4B token)."""
    state, fa_lines = flash_attn_auto(log)
    found = re.findall(r"graph splits = (\d+)(?: \(with bs=\d+\), (\d+) \(with bs=1\))?", log)
    splits = [int(n) for n in found[-1] if n] if found else []
    m = re.search(r"print_info: n_layer\s+=\s+(\d+)", log)
    layers = int(m.group(1)) if m else None
    ok = state == "enabled" and bool(splits) and layers is not None and max(splits) < layers
    return ok, (f"{fa_lines or f'no flash-attention check in the log (rc={rc})'}; graph splits "
                f"{'/'.join(map(str, splits)) or '?'}, layers {layers or '?'}")


def model_run(bdir, logs, model):
    if not Path(model).is_file():
        report("model run", False, f"model not found: {model}")
        return
    cmd = [str(bdir / "bin" / "llama-completion"), "-m", model, "-p", "The capital of France is", "-n", "32",
           "--temp", "0", "-no-cnv", "-t", "8", "-lv", "4"]
    rc_on, text_on, log_on = run(cmd, logs / "model-enabled", env_extra=preload, env_drop=("FLAGOS_SPACEMIT_DISABLE",))
    rc_off, text_off, log_off = run(cmd, logs / "model-disabled", env_extra={"FLAGOS_SPACEMIT_DISABLE": "1", **preload})
    report("model runs", rc_on == 0 and rc_off == 0, f"rc enabled={rc_on}, disabled={rc_off}")
    # the IME kernels quantize activations differently from the CPU, so greedy text may drift; perplexity decides
    same = next((i for i, (a, b) in enumerate(zip(text_on, text_off)) if a != b), min(len(text_on), len(text_off)))
    info("model output", "identical to the CPU" if text_on == text_off else f"differs from the CPU after {same} characters")
    info("text", " ".join(text_on.split())[:90])

    placed = [f"{m.group(1)} {m.group(2)} MiB" for m in
              re.finditer(r"FlagOS:SpacemiT\S*\s+(model|KV|compute) buffer size\s*=\s*([0-9.]+)", log_on)
              if float(m.group(2)) > 0]
    splits = [re.findall(r"graph splits = ([^\n]*)", log) for log in (log_on, log_off)]
    speed = [tokens_per_second(log) for log in (log_on, log_off)]
    report("weights placed on the provider", any(p.startswith("model") for p in placed), ", ".join(placed) or "none in use")
    # IGPU (D3): llama.cpp offloads every layer and puts their KV cache in the provider's buffer
    off = re.findall(r"offloaded (\d+)/(\d+) layers", ANSI.sub("", log_on))
    kv = any(p.startswith("KV") for p in placed)
    report("every layer and the KV cache on the provider", bool(off) and off[-1][0] == off[-1][1] and kv,
           (f"offloaded {off[-1][0]}/{off[-1][1]} layers" if off else "no offload line") +
           f"; {', '.join(placed) or 'no provider buffers in use'}")
    # the run leaves -fa at its default, auto; llama.cpp keeps flash attention on the provider, whose device is the
    # layer's under IGPU
    report("-fa auto keeps flash attention on the provider", *attention_on_provider(log_on, rc_on))
    # only the token embedding stays on the CPU
    n_split = splits_per_token(log_on)
    report("a token takes 2 graph splits", n_split == 2,
           f"per token with the provider {n_split}; enabled {splits[0]}, disabled {splits[1]}")
    info("generation speed", f"enabled {speed[0]} t/s, disabled {speed[1]} t/s")


def generated_tokens(log):
    """Tokens llama-completion generated after the first: its "eval time = ... / N runs" (not "prompt eval time")."""
    found = re.findall(r"(?<!prompt )eval time =\s+[0-9.]+ ms /\s+(\d+) runs", log)
    return int(found[-1]) if found else None


def cache_runs(bdir, logs, model):
    """The two settings that aborted with the KV cache in the provider's buffer before the provider wrote it
    (structural-experiments.md E2-neg, F8), each with the provider and with FLAGOS_SPACEMIT_DISABLE=1: a Q8_0 KV cache
    (its SET_ROWS on the provider, the attention over it on the CPU) and a context shift (the K cache rotated in place,
    ROPE on F16). -c 256 with 400 tokens to generate makes llama-completion shift the context; it reports 399 evals."""
    if not Path(model).is_file():
        report("KV cache runs", False, f"model not found: {model}")
        return
    cmd = [str(bdir / "bin" / "llama-completion"), "-m", model, "-p", "The capital of France is", "--temp", "0",
           "-no-cnv", "-t", "8", "-lv", "4"]
    for tag, label, args in (
            ("q8", "Q8_0 KV cache", ["-n", "32", "-c", "512", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on"]),
            ("shift", "context shift", ["-n", "400", "--ignore-eos", "-c", "256", "--context-shift"])):
        rc_on, text_on, log_on = run([*cmd, *args], logs / f"cache-{tag}-enabled", env_extra=preload,
                                     env_drop=("FLAGOS_SPACEMIT_DISABLE",))
        rc_off, text_off, log_off = run([*cmd, *args], logs / f"cache-{tag}-disabled",
                                        env_extra={"FLAGOS_SPACEMIT_DISABLE": "1", **preload})
        if tag == "q8":
            report(f"{label} runs", rc_on == 0 and rc_off == 0, f"rc enabled={rc_on}, disabled={rc_off}")
            info(f"{label} generation speed",
                 f"enabled {tokens_per_second(log_on)} t/s, disabled {tokens_per_second(log_off)} t/s")
        else:
            n_on, n_off = generated_tokens(log_on), generated_tokens(log_off)
            report(f"{label} runs", rc_on == 0 and n_on == 399,
                   f"rc enabled={rc_on}, disabled={rc_off}; evals after the first token: enabled {n_on}, "
                   f"disabled {n_off}")
        same = next((i for i, (a, b) in enumerate(zip(text_on, text_off)) if a != b), min(len(text_on), len(text_off)))
        info(f"{label} output",
             "identical to the CPU" if text_on == text_off else f"differs from the CPU after {same} characters")


def perplexity(bdir, logs, model, text):
    if not Path(text).is_file():
        report("perplexity", False, f"text not found: {text}")
        return
    # as M0.6 (build.md §7): 8 x 512 tokens, flash attention on; also the provider with the default -fa auto, which
    # keeps flash attention on it under IGPU (the CPU alone always keeps it, so one CPU run serves both)
    cmd = [str(bdir / "bin" / "llama-perplexity"), "-m", model, "-f", text, "-c", "512", "--chunks", "8", "-lv", "4"]
    # the CPU run also saves its logits for the KLD runs (about 600 MB at Qwen3's vocabulary).
    # The CPU run is single-threaded since 2026-10-29 (PROJECT.md §6, item 28): with several threads on the X100 cores
    # the CPU path is wrong at random for models with Q4_1 tensors, and a wrong base reads as a provider regression.
    # It takes longer (about 5 minutes on Qwen3-0.6B, about half an hour on 4B); the provider's run keeps 8 threads.
    kld_base = logs / "kld-base.bin"
    runs = [("enabled", ["-t", "8", "-fa", "on"], {}, ("FLAGOS_SPACEMIT_DISABLE",)),
            ("disabled", ["-t", "1", "-fa", "on", "--kl-divergence-base", str(kld_base)], {"FLAGOS_SPACEMIT_DISABLE": "1"},
             ()),
            ("enabled-fa-auto", ["-t", "8"], {}, ("FLAGOS_SPACEMIT_DISABLE",))]
    kld_base.unlink(missing_ok=True)
    ppl, problems, fa_auto = {}, {}, None
    for name, fa, extra, drop in runs:
        rc, out, err = run([*cmd, *fa], logs / f"ppl-{name}", env_extra={**extra, **preload}, env_drop=drop,
                           timeout=3600)
        m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", out + err)
        ppl[name] = float(m.group(1)) if rc == 0 and m else None
        problems[name] = ""
        if ppl[name] is None:
            # a negative rc is the signal that ended the run (-9: killed, e.g. out of memory; -11: segfault)
            problems[name] = (f"{name} run gave no result: rc={rc}, last log line: {last_line(err or out)[:100]}" +
                              (f"; segv.so report in {logs / f'ppl-{name}.err'}" if "=== segv.so" in err else ""))
        # by name: since the thread flags of 2026-10-29 no run's argument list is empty, and "if not fa" stopped
        # finding this one (the check went missing from the counts without a message until 2026-11-01)
        if name == "enabled-fa-auto":
            fa_auto = attention_on_provider(err, rc)
    cpu = ppl["disabled"]
    for name, label in (("enabled", " (-fa on)"), ("enabled-fa-auto", " (-fa auto)")):
        both = None not in (ppl[name], cpu)
        ok = both and abs(ppl[name] / cpu - 1.0) <= 0.01
        detail = f"provider {ppl[name]}, CPU {cpu}" + (f" ({100.0 * (ppl[name] / cpu - 1.0):+.2f}%)" if both else "")
        report(f"perplexity within 1% of the CPU{label}", ok,
               "; ".join(s for s in (detail, problems[name], problems["disabled"]) if s))
    if fa_auto is not None:
        # the -fa auto run measures the provider's attention only if it ran there
        report("perplexity -fa auto: flash attention on the provider", *fa_auto)
    kl_divergence(bdir, logs, model, text, kld_base)


def kl_divergence(bdir, logs, model, text, base):
    """Mean KLD of the provider's logits against the CPU's (base, from the CPU's perplexity run), as build.md §6: with
    512-token batches (4 sequences of 128 tokens per ubatch: the tiled kernel) and with -ub 1 -b 512 (one sequence, so
    every ubatch is one token as in generation: the 1-row matmuls, the split over the tiles from n_kv 512; about 10
    minutes for Qwen3-4B). With the default batch, -ub 1 would put a token of each of 2 sequences into every ubatch."""
    if not base.is_file():
        report("KLD against the CPU", False, f"no CPU logits: {base} (the CPU perplexity run failed)")
        return
    cmd = [str(bdir / "bin" / "llama-perplexity"), "-m", model, "-f", text, "-c", "512", "--chunks", "8", "-t", "8",
           "-fa", "on", "--kl-divergence-base", str(base), "--kl-divergence"]
    for label, ub, tag in (("512-token batches", [], ""), ("-ub 1 -b 512", ["-ub", "1", "-b", "512"], "-ub1")):
        rc, out, err = run([*cmd, *ub], logs / f"kld{tag}", env_extra=preload, env_drop=("FLAGOS_SPACEMIT_DISABLE",),
                           timeout=3600)
        m = re.search(r"Mean\s+KLD:\s+([0-9.]+)\s+\S+\s+([0-9.]+)", out + err)
        report(f"KLD against the CPU ({label})", rc == 0 and m is not None,
               f"mean {m.group(1)} +/- {m.group(2)}" if rc == 0 and m else
               f"no result: rc={rc}, last log line: {last_line(err or out)[:100]}")
    base.unlink(missing_ok=True)


def llama_bench(bdir, logs, model, ime_build, tcm_dir, depth):
    # as M0.5 (build.md §7), 3 repetitions
    args = ["-m", model, "-t", "8", "-p", "128", "-n", "128", "-ub", "128", "-fa", "1", "-mmp", "0", "-r", "3", "-o", "csv"]
    # --depth: generation against context depth as measured 2026-10-12 (build.md §7), after the runs above
    depth_args = ["-m", model, "-t", "8", "-p", "0", "-n", "32", "-d", "0,1024,4096", "-fa", "1", "-mmp", "0", "-r", "1",
                  "-o", "csv"]
    depths = [0, 1024, 4096]
    # and prompt processing at depth (plan.md, M2e design), in a run of its own so the tg32 series stays as measured
    pp_depth_args = ["-m", model, "-t", "8", "-p", "512", "-n", "0", "-d", "4096", "-fa", "1", "-mmp", "0", "-r", "1",
                     "-o", "csv"]
    modes = [("provider", bdir, {}, ("FLAGOS_SPACEMIT_DISABLE",)), ("cpu", bdir, {"FLAGOS_SPACEMIT_DISABLE": "1"}, ())]
    if ime_build and (Path(ime_build) / "bin" / "llama-bench").is_file():
        modes.append(("ime", Path(ime_build), {}, ()))
    release = tcm_tool(tcm_dir, "tcmrelease")

    def bench_rows(name, bd, extra, drop, bench_args, tag, timeout):
        if name == "ime" and release:
            run([release, "--apply"], logs / f"tcm-release-ime{tag}", timeout=120)
        rc, out, _ = run([str(bd / "bin" / "llama-bench"), *bench_args], logs / f"bench-{name}{tag}",
                         env_extra={**extra, **preload}, env_drop=drop, timeout=timeout)
        # a run cut off by the timeout still gives the rows it finished
        return rc, list(csv.DictReader(io.StringIO(out)))

    for name, bd, extra, drop in modes:
        rc, rows = bench_rows(name, bd, extra, drop, args, "", 1800)
        speeds = {}
        for row in rows:
            if row.get("n_prompt") == "128" and row.get("n_gen") == "0":
                speeds["pp128"] = float(row["avg_ts"])
            elif row.get("n_prompt") == "0" and row.get("n_gen") == "128":
                speeds["tg128"] = float(row["avg_ts"])
        info(f"llama-bench {name}", ", ".join(f"{k} {v:.2f} t/s" for k, v in speeds.items()) or f"no result (rc={rc})")
    if not depth:
        return
    for name, bd, extra, drop in modes:
        # llama-bench processes each depth's prompt, untimed, before its test: most of the run's time. It prints each
        # row as its test ends and exits at a failed test, so a crash or the timeout keeps the earlier rows
        rc, rows = bench_rows(name, bd, extra, drop, depth_args, "-depth", 3600)
        tg = {int(row["n_depth"]): float(row["avg_ts"]) for row in rows
              if row.get("n_prompt") == "0" and row.get("n_gen") == "32" and row.get("n_depth", "").isdigit()}
        rc_pp, rows = bench_rows(name, bd, extra, drop, pp_depth_args, "-depth-pp", 3600)
        pp = next((float(row["avg_ts"]) for row in rows
                   if row.get("n_prompt") == "512" and row.get("n_gen") == "0" and row.get("n_depth") == "4096"), None)
        missing = [f"tg32 d{d}" for d in depths if d not in tg] + (["pp512 d4096"] if pp is None else [])
        complete = not missing and rc == 0 and rc_pp == 0
        detail = "; ".join(s for s in (
            ("tg32 " + ", ".join(f"d{d} {v:.2f}" for d, v in sorted(tg.items())) + " t/s") if tg else "",
            f"pp512 d4096 {pp:.2f} t/s" if pp is not None else "",
            "" if complete else f"missing {', '.join(missing) or 'none'} (rc={rc}, {rc_pp})") if s)
        if name == "provider":
            report(f"llama-bench depth {name}", complete, detail)
        else:
            info(f"llama-bench depth {name}", detail)


def main():
    ap = argparse.ArgumentParser(description="acceptance run for the FlagOS SpacemiT provider (run on the K3)")
    ap.add_argument("--build", action="store_true", help="configure and build before checking")
    ap.add_argument("--build-dir", default=str(REPO / "build-flagos"), help="FlagOS build directory (default: %(default)s)")
    ap.add_argument("--spert-dir", default="", help="spine-runtime release directory (default: ~/spine-runtime)")
    ap.add_argument("--tcm-dir", default=os.path.expanduser("~/tcmtest"), help="directory with tcmtest and tcmrelease")
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-0.6B-Q4_0.gguf"), help="model for the output check")
    ap.add_argument("--ppl-text", default=os.path.expanduser("~/ppl.txt"), help="perplexity text (default: %(default)s)")
    ap.add_argument("--ime-build", default=str(REPO / "build-ime"), help="upstream IME build for llama-bench (skipped if absent)")
    ap.add_argument("--skip-support", action="store_true", help="skip the test-backend-ops support listing")
    ap.add_argument("--depth", action="store_true",
                    help="llama-bench: also tg32 at depths 0, 1024 and 4096 and pp512 at depth 4096 for each mode "
                         "(long)")
    ap.add_argument("--segv", default="", help="crash reporter (scripts/segv.c built as .so) to preload into the model runs and the check program's passes")
    args = ap.parse_args()
    if args.segv:
        if not Path(args.segv).is_file():
            sys.exit(f"--segv: not found: {args.segv}")
        preload["LD_PRELOAD"] = str(Path(args.segv).resolve())

    bdir = Path(args.build_dir).resolve()
    # a log directory per model, so one model's run does not overwrite another's logs
    logs = bdir / f"logs-{Path(args.model).stem}"
    logs.mkdir(parents=True, exist_ok=True)
    global summary
    summary = open(logs / "summary.txt", "w")
    say(f"repo {REPO}\nbuild {bdir}\nlogs {logs}" +
        (f"\ncrash reporter {preload['LD_PRELOAD']}" if preload else "") + "\n")

    if args.build and not build(bdir, logs, args.spert_dir):
        sys.exit(1)
    release = tcm_tool(args.tcm_dir, "tcmrelease")
    if release:
        run([release, "--apply"], logs / "tcm-release", timeout=120)
    check_tools(bdir, logs)
    list_devices(bdir, logs)
    if not args.skip_support:
        support(bdir, logs)
    op_correctness(bdir, logs)
    bench(bdir, logs)
    model_run(bdir, logs, args.model)
    cache_runs(bdir, logs, args.model)
    perplexity(bdir, logs, args.model, args.ppl_text)
    llama_bench(bdir, logs, args.model, args.ime_build, args.tcm_dir, args.depth)
    tcm_hygiene(args.tcm_dir, logs)

    n_fail = results.count(False)
    say(f"\nACCEPTANCE: {len(results) - n_fail}/{len(results)} checks passed" + ("" if n_fail == 0 else f", see {logs}"))
    summary.close()
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
