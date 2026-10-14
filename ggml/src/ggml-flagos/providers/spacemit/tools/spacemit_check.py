#!/usr/bin/env python3
# Acceptance run for the SpacemiT provider milestones (plan.md M1, M2a, M2b, M2d, M2c, M2c.2, M2e.1). Run on the K3;
# standard library only.
#   all:  (--build) configure and build build-flagos/; FlagOS Common checks; flagos-check-spacemit with the
#         device present and with FLAGOS_SPACEMIT_DISABLE=1; llama.cpp lists the device; op claims; a model
#         gives identical output with the provider enabled and disabled
#   m1:   the provider claims no op and nothing is placed on it
#   m2a:  it claims exactly ADD; test-backend-ops ADD passes; launch / per-node / bandwidth benchmark for both
#         stream policies; no TCM block left held
#   m2b:  as m2a, plus Q4_0 matmuls at Qwen3-4B's shapes against the CPU (IME and reference kernels), matmul
#         benchmark, model weights placed on the provider, perplexity within 1% of the CPU, pp128/tg128 against
#         the CPU and (if built) the upstream IME build
#   m2d:  as m2b, with the layer ops: it claims exactly ADD, MUL, RMS_NORM, ROPE, SET_ROWS, GET_ROWS, SWIGLU and
#         MUL_MAT; test-backend-ops passes for each and for the multi-op graphs built from them, with the RVV and
#         with the reference kernels; flagos-check-spacemit adds a Qwen3-like layer against the CPU
#   m2c:  as m2d, with Q4_1 matmuls (the IME layout q4_1 32x32, lossy): flagos-check-spacemit adds Q4_1 against the
#         CPU on the converted weights, the conversion's own effect and the 1-row kernel's zero-point probe (both
#         informational), and the layer with a Q4_1 down projection; the benchmark adds Q4_1 at Qwen3-4B's ffn_down
#   m2c2: as m2c, with Q8_0 and Q6_K matmuls (the IME layout q8_0 32x32; Q6_K requantized to Q8_0 at load): it claims
#         MUL_MAT cases of test-backend-ops (Q8_0 and Q6_K weights with rows a multiple of 32), which pass with the IME
#         and with the reference kernels (which read Q6_K back, so they cannot see the requantization);
#         flagos-check-spacemit adds Q8_0 against the CPU and Q6_K against the CPU's Q6_K and, with a tight bound that
#         checks the requantization, against the CPU's Q8_0 of the same values, up to Qwen3's output heads (151936
#         rows); the benchmark adds Q6_K at Qwen3-4B's output head
#   m2e:  as m2c2, with attention (M2e.1): it claims FLASH_ATTN_EXT too (F16 K and V, head sizes up to 128), whose
#         test-backend-ops cases pass with the RVV and with the reference kernels; flagos-check-spacemit adds attention
#         against the CPU with the KV cache's layout and the probes for the three faults of the copied kernels; the
#         model run's default -fa auto keeps flash attention on the provider (C5a; FAIL with llama.cpp's warning if it
#         turns it off, or if the graphs take as many splits as the model has layers, as when the CPU runs attention)
#         and a token takes 2 graph splits; perplexity with -fa on and with -fa auto (attention on the provider), each
#         within 1% of the CPU; mean KLD against the CPU's logits with 512-token batches (tiled kernel) and -ub 1 -b
#         512 (one token per ubatch: from n_kv 512 the split over the tiles), to compare by hand with build.md §7
#         (Qwen3-4B at M2e.3 round 2: 0.003354 and 0.002001)
#   --depth (m2b and later): after the other llama-bench runs, tg32 at depths 0, 1024 and 4096 and pp512 at depth
#         4096 for the provider, the CPU and (if built) the upstream IME build; FAIL if the provider's runs miss one
# Usage: spacemit_check.py [--milestone m1|m2a|m2b|m2d|m2c|m2c2|m2e] [--build] [--build-dir DIR] [--model GGUF]
#                          [--ppl-text FILE] [--ime-build DIR] [--skip-support] [--depth] [--segv SO]
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
# the ops M2d claims (GLU cases are named by their GLU op), and the test-backend-ops graphs that chain them
M2D_OPS = ["ADD", "MUL", "RMS_NORM", "ROPE", "SET_ROWS", "GET_ROWS", "SWIGLU"]
M2D_GRAPHS = ["RMS_NORM_MUL_ADD", "ADD_RMS_NORM", "RMS_NORM_MUL_ROPE", "ROPE_SET_ROWS"]
results = []
preload = {}  # LD_PRELOAD for the model runs (--segv)


def report(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def info(name, detail):
    print(f"[INFO] {name}: {detail}", flush=True)


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
                       env_extra=env_extra, timeout=timeout)
    detail = "; ".join(ln for ln in out.splitlines()
                       if ln.startswith(("device", "flagos ", "ops ", "matmul   Q4_0", "matmul   Q4_1", "matmul   Q8_0",
                                         "matmul   Q6_K", "gemv ", "rows ", "attn ", "layer")))
    fails = [ln for ln in err.splitlines() if ln.startswith("FAIL") or "timeout" in ln]
    report(name, rc == 0, detail if rc == 0 else "; ".join(fails[:3]) or last_line(err))
    for ln in out.splitlines():
        if ln.startswith("info "):
            info(name, ln[len("info "):].strip())


def check_tools(bdir, logs, milestone):
    for tool in ["provider", "target", "graph-plan", "registry"]:
        rc, out, err = run([str(bdir / "bin" / f"flagos-check-{tool}")], logs / f"check-{tool}")
        report(f"flagos-check-{tool}", rc == 0, last_line(out) if rc == 0 else f"{last_line(err or out)} (rc={rc})")
    if milestone in ("m2b", "m2d", "m2c", "m2c2", "m2e"):
        check_spacemit(bdir, logs, "flagos-check-spacemit (device present, Qwen3-4B shapes)", "check-spacemit", ["--full"],
                       timeout=900)
        # the same with the reference kernels on the AI cores: tells a kernel error from a tiling error
        check_spacemit(bdir, logs, "flagos-check-spacemit (reference kernels)", "check-spacemit-reference", [],
                       env_extra={"FLAGOS_SPACEMIT_TEST_REFERENCE": "1"}, timeout=900)
    else:
        check_spacemit(bdir, logs, "flagos-check-spacemit (device present)", "check-spacemit", [])
    rc, out, _ = run([str(bdir / "bin" / "flagos-check-spacemit"), "--expect-none"], logs / "check-spacemit-disabled",
                     env_extra={"FLAGOS_SPACEMIT_DISABLE": "1"})
    report("flagos-check-spacemit (FLAGOS_SPACEMIT_DISABLE=1)", rc == 0, "device absent" if rc == 0 else f"rc={rc}")


def list_devices(bdir, logs):
    rc, out, err = run([str(bdir / "bin" / "llama-completion"), "--list-devices"], logs / "list-devices")
    line = next((ln.strip() for ln in (out + err).splitlines() if DEVICE in ln), "")
    report("llama.cpp device list", bool(line), line or f"{DEVICE} not listed (rc={rc})")


def support(bdir, logs, milestone):
    rc, out, _ = run([str(bdir / "bin" / "test-backend-ops"), "support", "-b", DEVICE], logs / "support", timeout=1800)
    lines = ANSI.sub("", out).splitlines()
    claimed = [ln.strip() for ln in lines if ln.rstrip().endswith(" SUPPORTED") and "NOT SUPPORTED" not in ln]
    n_no = sum("NOT SUPPORTED" in ln for ln in lines)
    ops = sorted({ln.split("(")[0] for ln in claimed})
    if milestone == "m1":
        ok = rc == 0 and not claimed and n_no > 0
    elif milestone == "m2a":
        ok = rc == 0 and ops == ["ADD"]
    elif milestone in ("m2d", "m2c"):
        # no MUL_MAT case of test-backend-ops fits the IME layouts (16-row Q4_0/Q4_1 weights), so MUL_MAT may show 0
        ok = rc == 0 and set(M2D_OPS) <= set(ops) <= set(M2D_OPS) | {"MUL_MAT"}
    elif milestone == "m2c2":
        # Q8_0 and Q6_K MUL_MAT cases with rows a multiple of 32 fit the q8_0 32x32 layout
        ok = rc == 0 and set(ops) == set(M2D_OPS) | {"MUL_MAT"}
    elif milestone == "m2e":
        # as m2c2, plus the FLASH_ATTN_EXT cases with F16 K and V and head sizes up to 128
        ok = rc == 0 and set(ops) == set(M2D_OPS) | {"MUL_MAT", "FLASH_ATTN_EXT"}
    else:
        # no MUL_MAT case of test-backend-ops fits the IME layout (plan.md §1), so MUL_MAT may show 0 claims
        ok = rc == 0 and "ADD" in ops and set(ops) <= {"ADD", "MUL_MAT"}
    report("test-backend-ops support", ok, f"{len(claimed)} op cases claimed ({', '.join(ops) or 'none'}), {n_no} not claimed (rc={rc})")


def op_test(bdir, logs, op, env_extra=None, tag=""):
    """test-backend-ops for one op name; returns (ok, summary)."""
    rc, out, _ = run([str(bdir / "bin" / "test-backend-ops"), "test", "-o", op, "-b", DEVICE],
                     logs / f"test-{op.lower()}{tag}", env_extra=env_extra, timeout=1800)
    clean = ANSI.sub("", out)
    passed = re.search(r"(\d+)/(\d+) tests passed", clean)
    verdict = re.search(rf"Backend {re.escape(DEVICE)}: (\w+)", clean)
    ok = rc == 0 and passed is not None and passed.group(1) == passed.group(2) and int(passed.group(2)) > 0
    return ok, f"{passed.group(0) if passed else 'no summary'}, {verdict.group(1) if verdict else '?'} (rc={rc})"


def op_correctness(bdir, logs, milestone):
    if milestone not in ("m2d", "m2c", "m2c2", "m2e"):
        report("test-backend-ops ADD", *op_test(bdir, logs, "ADD"))
        return
    # from M2c.2 test-backend-ops has MUL_MAT cases the provider claims (op_test needs at least one to run), from M2e.1
    # FLASH_ATTN_EXT cases
    ops = (M2D_OPS + (["MUL_MAT"] if milestone in ("m2c2", "m2e") else []) +
           (["FLASH_ATTN_EXT"] if milestone == "m2e" else []) + M2D_GRAPHS)
    for op in ops:
        report(f"test-backend-ops {op}", *op_test(bdir, logs, op))
    # the same with the reference kernels on the AI cores, in one line
    failed = [op for op in ops if not op_test(bdir, logs, op, {"FLAGOS_SPACEMIT_TEST_REFERENCE": "1"}, "-reference")[0]]
    report("test-backend-ops, reference kernels", not failed,
           f"{len(ops) - len(failed)}/{len(ops)} ops pass" + (f"; failed: {', '.join(failed)}" if failed else ""))


def bench(bdir, logs):
    rc, out, err = run([str(bdir / "bin" / "flagos-check-spacemit"), "--expect-device", "--bench"], logs / "bench", timeout=900)
    # the 1-row strategy table (one line per shape and strategy) stays in the log; its per-token totals are shown
    lines = [ln.split("bench", 1)[1].strip() for ln in out.splitlines()
             if ln.startswith("bench") and (not ln.startswith("bench    gemv ") or " per Qwen3-4B token " in ln)]
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


def model_run(bdir, logs, model, milestone):
    if not Path(model).is_file():
        report("model run", False, f"model not found: {model}")
        return
    cmd = [str(bdir / "bin" / "llama-completion"), "-m", model, "-p", "The capital of France is", "-n", "32",
           "--temp", "0", "-no-cnv", "-t", "8", "-lv", "4"]
    rc_on, text_on, log_on = run(cmd, logs / "model-enabled", env_extra=preload, env_drop=("FLAGOS_SPACEMIT_DISABLE",))
    rc_off, text_off, log_off = run(cmd, logs / "model-disabled", env_extra={"FLAGOS_SPACEMIT_DISABLE": "1", **preload})
    report("model runs", rc_on == 0 and rc_off == 0, f"rc enabled={rc_on}, disabled={rc_off}")
    if milestone in ("m2b", "m2d", "m2c", "m2c2", "m2e"):
        # the IME kernels quantize activations differently from the CPU, so greedy text may drift; perplexity decides
        same = next((i for i, (a, b) in enumerate(zip(text_on, text_off)) if a != b), min(len(text_on), len(text_off)))
        info("model output", "identical to the CPU" if text_on == text_off else f"differs from the CPU after {same} characters")
        info("text", " ".join(text_on.split())[:90])
    else:
        report("model output identical", text_on == text_off and len(text_on) > 0, " ".join(text_on.split())[:90])

    placed = [f"{m.group(1)} {m.group(2)} MiB" for m in
              re.finditer(r"FlagOS:SpacemiT\S*\s+(model|KV|compute) buffer size\s*=\s*([0-9.]+)", log_on)
              if float(m.group(2)) > 0]
    splits = [re.findall(r"graph splits = ([^\n]*)", log) for log in (log_on, log_off)]
    speed = [tokens_per_second(log) for log in (log_on, log_off)]
    if milestone == "m1":
        report("nothing placed on the provider", not placed, "no provider buffers in use" if not placed else ", ".join(placed))
        report("graph splits unchanged", splits[0] == splits[1] and bool(splits[0]), f"enabled {splits[0]}, disabled {splits[1]}")
    elif milestone in ("m2b", "m2d", "m2c", "m2c2", "m2e"):
        report("weights placed on the provider", any(p.startswith("model") for p in placed), ", ".join(placed) or "none in use")
        if milestone == "m2e":
            # the run leaves -fa at its default, auto; with C5a llama.cpp keeps flash attention on the provider
            report("-fa auto keeps flash attention on the provider (C5a)", *attention_on_provider(log_on, rc_on))
            # exit M2e.1 (plan.md): only the token embedding stays on the CPU
            n_split = splits_per_token(log_on)
            report("a token takes 2 graph splits", n_split == 2,
                   f"per token with the provider {n_split}; enabled {splits[0]}, disabled {splits[1]}")
        else:
            info("graph splits", f"enabled {splits[0]}, disabled {splits[1]}")
    else:
        info("provider buffers", ", ".join(placed) or "none in use")
        info("graph splits", f"enabled {splits[0]}, disabled {splits[1]}")
    info("generation speed", f"enabled {speed[0]} t/s, disabled {speed[1]} t/s")


def perplexity(bdir, logs, model, text, milestone):
    if not Path(text).is_file():
        report("perplexity", False, f"text not found: {text}")
        return
    # as M0.6 (build.md §7): 8 x 512 tokens, flash attention on; from M2e.1 also the provider with the default -fa auto,
    # which keeps flash attention on it with C5a (the CPU alone always keeps it, so one CPU run serves both)
    cmd = [str(bdir / "bin" / "llama-perplexity"), "-m", model, "-f", text, "-c", "512", "--chunks", "8", "-t", "8",
           "-lv", "4"]
    # from M2e.1 the CPU run also saves its logits for the KLD runs (about 600 MB at Qwen3's vocabulary)
    kld_base = logs / "kld-base.bin"
    runs = [("enabled", ["-fa", "on"], {}, ("FLAGOS_SPACEMIT_DISABLE",)),
            ("disabled", ["-fa", "on"] + (["--kl-divergence-base", str(kld_base)] if milestone == "m2e" else []),
             {"FLAGOS_SPACEMIT_DISABLE": "1"}, ())]
    if milestone == "m2e":
        kld_base.unlink(missing_ok=True)
        runs.append(("enabled-fa-auto", [], {}, ("FLAGOS_SPACEMIT_DISABLE",)))
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
        if not fa:
            fa_auto = attention_on_provider(err, rc)
    cpu = ppl["disabled"]
    checks = [("enabled", "")]
    if milestone == "m2e":
        checks = [("enabled", " (-fa on)"), ("enabled-fa-auto", " (-fa auto)")]
    for name, label in checks:
        both = None not in (ppl[name], cpu)
        ok = both and abs(ppl[name] / cpu - 1.0) <= 0.01
        detail = f"provider {ppl[name]}, CPU {cpu}" + (f" ({100.0 * (ppl[name] / cpu - 1.0):+.2f}%)" if both else "")
        report(f"perplexity within 1% of the CPU{label}", ok,
               "; ".join(s for s in (detail, problems[name], problems["disabled"]) if s))
    if fa_auto is not None:
        # the -fa auto run measures the provider's attention only if it ran there
        report("perplexity -fa auto: flash attention on the provider", *fa_auto)
    if milestone == "m2e":
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
    ap.add_argument("--milestone", choices=["m1", "m2a", "m2b", "m2d", "m2c", "m2c2", "m2e"], default="m2e",
                    help="expectations to check (default: %(default)s)")
    ap.add_argument("--build", action="store_true", help="configure and build before checking")
    ap.add_argument("--build-dir", default=str(REPO / "build-flagos"), help="FlagOS build directory (default: %(default)s)")
    ap.add_argument("--spert-dir", default="", help="spine-runtime release directory (default: ~/spine-runtime)")
    ap.add_argument("--tcm-dir", default=os.path.expanduser("~/tcmtest"), help="directory with tcmtest and tcmrelease")
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-0.6B-Q4_0.gguf"), help="model for the output check")
    ap.add_argument("--ppl-text", default=os.path.expanduser("~/ppl.txt"), help="perplexity text (m2b; default: %(default)s)")
    ap.add_argument("--ime-build", default=str(REPO / "build-ime"), help="upstream IME build for llama-bench (m2b; skipped if absent)")
    ap.add_argument("--skip-support", action="store_true", help="skip the test-backend-ops support listing")
    ap.add_argument("--depth", action="store_true",
                    help="llama-bench: also tg32 at depths 0, 1024 and 4096 and pp512 at depth 4096 for each mode "
                         "(m2b and later; long)")
    ap.add_argument("--segv", default="", help="crash reporter (scripts/segv.c built as .so) to preload into the model runs")
    args = ap.parse_args()
    if args.segv:
        if not Path(args.segv).is_file():
            sys.exit(f"--segv: not found: {args.segv}")
        preload["LD_PRELOAD"] = str(Path(args.segv).resolve())

    bdir = Path(args.build_dir).resolve()
    logs = bdir / f"{args.milestone}-logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"milestone {args.milestone}\nrepo {REPO}\nbuild {bdir}\nlogs {logs}" +
          (f"\ncrash reporter {preload['LD_PRELOAD']}" if preload else "") + "\n", flush=True)

    if args.build and not build(bdir, logs, args.spert_dir):
        sys.exit(1)
    release = tcm_tool(args.tcm_dir, "tcmrelease")
    if release:
        run([release, "--apply"], logs / "tcm-release", timeout=120)
    check_tools(bdir, logs, args.milestone)
    list_devices(bdir, logs)
    if not args.skip_support:
        support(bdir, logs, args.milestone)
    if args.milestone in ("m2a", "m2b", "m2d", "m2c", "m2c2", "m2e"):
        op_correctness(bdir, logs, args.milestone)
        bench(bdir, logs)
    model_run(bdir, logs, args.model, args.milestone)
    if args.milestone in ("m2b", "m2d", "m2c", "m2c2", "m2e"):
        perplexity(bdir, logs, args.model, args.ppl_text, args.milestone)
        llama_bench(bdir, logs, args.model, args.ime_build, args.tcm_dir, args.depth)
    if args.milestone in ("m2a", "m2b", "m2d", "m2c", "m2c2", "m2e"):
        tcm_hygiene(args.tcm_dir, logs)

    n_fail = results.count(False)
    print(f"\n{args.milestone.upper()}: {len(results) - n_fail}/{len(results)} checks passed" + ("" if n_fail == 0 else f", see {logs}"))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
