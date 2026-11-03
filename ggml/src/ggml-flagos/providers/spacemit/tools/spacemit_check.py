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
#   llama-server (--skip-server leaves it out): every layer and the KV cache on the provider with 4 slots in one unified
#   KV cache, a greedy reply equal to llama-completion's, a follow-up turn and two interleaved conversations (reloaded
#   from the server's prompt cache) each against the same prompt from scratch, KV shifting (--cache-reuse); every reply
#   saved in logs/server.json, and with --server-baseline equal to a saved one's;
#   perplexity with -fa on and with -fa auto, each within 1% of the CPU (the CPU single-threaded, PROJECT.md §6);
#   FLASH_ATTN_EXT disabled by name (GGML_FLAGOS_DISABLE_OPS) leaves the provider, every layer split around it, the
#   perplexity over 2 chunks within 1% of the enabled run's; mean KLD against the CPU's logits with 512-token batches
#   (the tiled kernel) and -ub 1 -b 512 (one token per ubatch: the 1-row matmuls, from n_kv 512 the split over the
#   tiles), to compare by hand with build.md §7; llama-bench pp128 and tg128 for the provider, the CPU and (if built)
#   the upstream IME build; no TCM block left held.
#   --depth: after the other llama-bench runs, tg32 at depths 0, 1024 and 4096 and pp512 at depth 4096 for the same
#   modes; FAIL if the provider's runs miss one
# Usage: spacemit_check.py [--build] [--build-dir DIR] [--model GGUF] [--ppl-text FILE] [--ime-build DIR]
#                          [--skip-support] [--depth] [--segv SO] [--skip-server] [--server-baseline JSON]
import argparse
import csv
import http.client
import io
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
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
               "llama-perplexity", "llama-server"]
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
                   env_extra={"GGML_FLAGOS_KERNEL_VARIANTS": "0"}, timeout=900)
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
              if not op_test(bdir, logs, op, {"GGML_FLAGOS_KERNEL_VARIANTS": "0"}, "-reference")[0]]
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


# the server check (plan.md, "The server check, the baseline before S2"): the prompts and the reply names
SERVER_PROMPT_A = "The capital of France is"
SERVER_B1 = "My favourite colour is green. I live in a small town by the sea.\nQuestion: Where do I live? Answer:"
SERVER_B2 = "\nQuestion: What is my favourite colour? Answer:"
SERVER_X1 = ("The train leaves at seven and the bus at nine. Tickets cost four euros.\nQuestion: When does the train "
             "leave? Answer:")
SERVER_X2 = "\nQuestion: How much do tickets cost? Answer:"
SERVER_Y1 = "Anna has two cats and a dog. The dog is called Max.\nQuestion: How many cats does Anna have? Answer:"
SERVER_Y2 = "\nQuestion: What is the dog called? Answer:"
SERVER_D = ["Meeting notes.\n",
            "Item one: the budget for next year is due on Friday, and the report must be sent to the board before then. ",
            "Item two: my favourite colour is green and my sister's favourite colour is blue; we both like the sea and "
            "long walks in the hills on Sundays.\n",
            "Question: What is my favourite colour? Answer:"]
SERVER_E = " ".join(["The quick brown fox jumps over the lazy dog."] * 50)
SERVER_REPLIES = ["A", "B1", "B2", "B2-scratch", "C-x1", "C-y1", "C-x2", "C-y2", "C-x2-scratch", "C-y2-scratch", "D1",
                  "D2", "E"]
# the first token's top-1 probability, reused turn against scratch: an estimate (0.0031 on the simulated device)
SERVER_TOLERANCE = 0.05
# the errors of a request or a /health poll (URLError and HTTPError are OSErrors, a bad reply a ValueError)
HTTP_ERRORS = (OSError, ValueError, KeyError, http.client.HTTPException)


def server_post(url, path, body, timeout=600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def server_wait(srv, url, timeout=600):
    """(seconds, "") once GET /health answers {"status": "ok"}, or (None, why) if the server exits or the time runs out.
    It answers 503 while the model loads."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if srv.poll() is not None:
            return None, f"llama-server exited with rc={srv.returncode} before it was ready"
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return time.time() - t0, ""
        except HTTP_ERRORS:
            pass
        time.sleep(1)
    return None, f"/health not ok after {timeout} s"


def server_requests(url, replies):
    """The check's requests in order, each reply's content, first-token top_logprobs and timings saved in replies under
    its name. No id_slot: the server chooses the slot, so the interleaved conversations are reloaded from its prompt
    cache. Returns (error, common prefix of D's two prompts in tokens); after an error the later requests are not sent."""
    current = [""]

    def ask(name, prompt, n_predict, cache_prompt, n_probs=0):
        current[0] = name
        body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": cache_prompt}
        if n_probs:
            body["n_probs"] = n_probs
        r = server_post(url, "/completion", body)
        probs = r.get("completion_probabilities") or []
        replies[name] = {"content": r["content"], "top_logprobs": probs[0].get("top_logprobs") if probs else None,
                         "timings": r.get("timings", {})}
        return r["content"]

    try:
        # A: a fresh server against llama-completion
        ask("A", SERVER_PROMPT_A, 32, False)
        # B: a follow-up turn reusing the slot's cache, then the same prompt from scratch
        p2 = SERVER_B1 + ask("B1", SERVER_B1, 12, True, 5) + SERVER_B2
        ask("B2", p2, 12, True, 5)
        ask("B2-scratch", p2, 12, False, 5)
        # C: two conversations interleaved; each new task saves the other slot to the prompt cache and clears it
        x2 = SERVER_X1 + ask("C-x1", SERVER_X1, 12, True, 5) + SERVER_X2
        y2 = SERVER_Y1 + ask("C-y1", SERVER_Y1, 12, True, 5) + SERVER_Y2
        ask("C-x2", x2, 12, True, 5)
        ask("C-y2", y2, 12, True, 5)
        ask("C-x2-scratch", x2, 12, False, 5)
        ask("C-y2-scratch", y2, 12, False, 5)
        # D: KV shifting, a cached A+B+C+Q then A+C+Q
        d1, d2 = "".join(SERVER_D), SERVER_D[0] + "".join(SERVER_D[2:])
        ask("D1", d1, 12, True, 5)
        ask("D2", d2, 12, True, 5)
        # E: about 500 prompt tokens, for the time to the first token and per token
        ask("E", SERVER_E, 32, False)
        # D's common prefix as the server counts its prompt (with BOS)
        current[0] = "tokenize"
        t1, t2 = (server_post(url, "/tokenize", {"content": p, "add_special": True})["tokens"] for p in (d1, d2))
        prefix = next((i for i, (a, b) in enumerate(zip(t1, t2)) if a != b), min(len(t1), len(t2)))
        return "", prefix
    except HTTP_ERRORS as e:
        return f"request {current[0]} failed: {e}", None


def first_token(reply):
    """(token, probability) of the entry of highest probability in the first token's top_logprobs, or None."""
    probs = (reply or {}).get("top_logprobs") or []
    if not probs:
        return None
    top = max(probs, key=lambda t: t["logprob"])
    return top["token"], math.exp(top["logprob"])


def server_reuse(replies, reused, scratch, why):
    """(ok, detail): the turn that reused the cache (cache_n > 0) against the same prompt from scratch (cache_n 0), the
    first token's top-1 the same token in both, its probabilities at most SERVER_TOLERANCE apart."""
    a, b = replies.get(reused), replies.get(scratch)
    if a is None or b is None:
        return False, f"no reply {reused if a is None else scratch}: {why or 'not sent'}"
    ca, cb = a["timings"].get("cache_n"), b["timings"].get("cache_n")
    ta, tb = first_token(a), first_token(b)
    detail = (f"{reused} cache_n {ca}, prompt_n {a['timings'].get('prompt_n')}; {scratch} cache_n {cb}, prompt_n "
              f"{b['timings'].get('prompt_n')}")
    if ta is None or tb is None:
        return False, f"{detail}; no top_logprobs in {reused if ta is None else scratch}"
    diff = abs(ta[1] - tb[1])
    ok = (ca or 0) > 0 and cb == 0 and ta[0] == tb[0] and diff <= SERVER_TOLERANCE
    return ok, f"{detail}; top-1 {ta[0]!r} {ta[1]:.4f} / {tb[0]!r} {tb[1]:.4f}, difference {diff:.4f}"


def server_checks(bdir, logs, model, baseline):
    """llama-server on the provider through the paths single-user clients use: its default 4 slots in one unified KV
    cache, a follow-up turn reusing the cache, conversations reloaded from its host-RAM prompt cache, KV shifting
    (--cache-reuse 16); every reply saved in logs/server.json, the baseline S2 must reproduce (with baseline, check 6
    compares against one). A missing binary or model, or a server that does not come up, fails every check."""
    names = ["llama-server: every layer and the KV cache on the provider, 4 slots in one unified KV cache",
             "llama-server's greedy reply is llama-completion's",
             "llama-server: a follow-up turn reusing its cache matches the turn from scratch",
             "llama-server: interleaved conversations reload from the prompt cache and match from scratch",
             "llama-server: KV shifting (--cache-reuse) runs on the provider"]
    if baseline:
        names.append("llama-server's replies equal the baseline's")
    server = bdir / "bin" / "llama-server"
    problem = (f"not found: {server}" if not server.is_file() else
               f"model not found: {model}" if not Path(model).is_file() else "")
    if problem:
        for name in names:
            report(name, False, problem)
        return
    # the reference first, so that it and the server never run at the same time
    rc_ref, ref_out, ref_err = run([str(bdir / "bin" / "llama-completion"), "-m", model, "-p", SERVER_PROMPT_A, "-n",
                                    "32", "--temp", "0", "-no-cnv", "-t", "8", "-c", "4096"], logs / "server-reference",
                                   env_extra=preload, env_drop=("FLAGOS_SPACEMIT_DISABLE",))
    # the environment of the model runs, as run() builds it
    env = {k: v for k, v in os.environ.items() if k != "FLAGOS_SPACEMIT_DISABLE"}
    env.update(preload)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    replies, up_s, failure, prefix = {}, None, "", None
    log_path = logs / "server.log"
    with open(log_path, "w") as log:
        srv = None
        try:
            srv = subprocess.Popen([str(server), "-m", model, "--host", "127.0.0.1", "--port", str(port), "-t", "8", "-c",
                                    "4096", "-lv", "4", "--cache-reuse", "16"], stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, env=env)
            up_s, failure = server_wait(srv, url)
            if up_s is not None:
                failure, prefix = server_requests(url, replies)
        except OSError as e:
            failure = f"llama-server did not start: {e}"
        finally:
            if srv is not None and srv.poll() is None:
                srv.terminate()
                try:
                    srv.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    srv.kill()
                    srv.wait()
    text = ANSI.sub("", log_path.read_text(errors="replace"))
    if up_s is None:
        why = f"{failure}; last log line: {last_line(text)[:100]}"
        for name in names:
            report(name, False, why)
        return
    (logs / "server.json").write_text(json.dumps(replies, indent=1) + "\n")

    # 1. placement and layout, from the last context the log creates
    off = re.findall(r"offloaded (\d+)/(\d+) layers", text)
    seq = re.findall(r"llama_context: n_seq_max\s*=\s*(\d+)", text)
    unified = re.findall(r"llama_context: kv_unified\s*=\s*(\w+)", text)
    splits = re.findall(r"graph splits = (\d+)(?: \(with bs=\d+\), (\d+) \(with bs=1\))?", text)
    splits = [int(n) for n in splits[-1] if n] if splits else []
    last_ctx = text[text.rfind("llama_context: n_seq_max"):] if seq else ""
    kv = [(m.group(1), float(m.group(2))) for m in re.finditer(r"(\S+) KV buffer size\s*=\s*([0-9.]+) MiB", last_ctx)]
    ok = (bool(off) and off[-1][0] == off[-1][1] and any(n.startswith("FlagOS:SpacemiT") and v > 0 for n, v in kv) and
          seq[-1:] == ["4"] and unified[-1:] == ["true"] and bool(splits) and all(n == 2 for n in splits))
    report(names[0], ok,
           f"up after {up_s:.1f} s; " + (f"offloaded {off[-1][0]}/{off[-1][1]} layers" if off else "no offload line") +
           f"; KV buffers {', '.join(f'{n} {v:.2f} MiB' for n, v in kv) or 'none'}; n_seq_max "
           f"{seq[-1] if seq else '?'}, kv_unified {unified[-1] if unified else '?'}, graph splits "
           f"{'/'.join(map(str, splits)) or '?'}")

    # 2. A against llama-completion, whose output repeats the prompt and ends with two newlines
    a = replies.get("A")
    if a is None:
        report(names[1], False, f"no reply A: {failure}")
    elif rc_ref != 0 or not ref_out.startswith(SERVER_PROMPT_A):
        report(names[1], False, f"llama-completion: rc={rc_ref}, output does not start with the prompt: "
                                f"{ref_out[:40]!r}; last log line: {last_line(ref_err)[:100]}")
    else:
        got, want = a["content"].rstrip("\n"), ref_out[len(SERVER_PROMPT_A):].rstrip("\n")
        i = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
        report(names[1], got == want, f"{len(got)} characters equal" if got == want else
               f"first difference at character {i} (lengths {len(got)}, {len(want)}): server {got[i:i + 20]!r}, "
               f"llama-completion {want[i:i + 20]!r}")

    # 3. the follow-up turn
    report(names[2], *server_reuse(replies, "B2", "B2-scratch", failure))

    # 4. the interleaved conversations, reloaded from the prompt cache
    found = len(re.findall(r"found better prompt", text))
    (ok_x, det_x), (ok_y, det_y) = (server_reuse(replies, f"C-{c}2", f"C-{c}2-scratch", failure) for c in "xy")
    report(names[3], ok_x and ok_y and found > 0, f"{det_x}; {det_y}; {found} 'found better prompt' lines")

    # 5. the shift: D2 reused a chunk beyond the common prefix
    d2 = replies.get("D2")
    chunks = re.findall(r"reusing chunk with size [^\n]*", text)
    if d2 is None or prefix is None:
        report(names[4], False, f"no reply D2 or no common prefix: {failure}; {len(chunks)} 'reusing chunk' lines")
    else:
        cache_n, prompt_n = d2["timings"].get("cache_n") or 0, d2["timings"].get("prompt_n") or 0
        ok = bool(d2["content"]) and cache_n >= prefix + 16 and bool(chunks)
        report(names[4], ok, f"D2: {len(d2['content'])} characters, cache_n {cache_n} of {cache_n + prompt_n} prompt "
                             f"tokens, common prefix {prefix} tokens; {len(chunks)} 'reusing chunk' lines" +
               (f", the last: {chunks[-1]}" if chunks else ""))

    # 6. the replies against a saved baseline, exactly
    if baseline:
        try:
            base = json.loads(Path(baseline).read_text())
        except (OSError, ValueError) as e:
            report(names[5], False, f"baseline not read: {e}")
            base = None
        if base is not None:
            order = SERVER_REPLIES + sorted((set(base) | set(replies)) - set(SERVER_REPLIES))
            differ = []
            for name in order:
                mine, theirs = replies.get(name), base.get(name)
                if mine is None or theirs is None:
                    differ.append(f"{name} missing in {'this run' if mine is None else 'the baseline'}")
                    continue
                fields = [k for k in ("content", "top_logprobs") if mine.get(k) != theirs.get(k)]
                if fields:
                    differ.append(f"{name} ({', '.join(fields)})")
            report(names[5], not differ, f"{len(order) - len(differ)}/{len(order)} replies equal to {baseline}" +
                   (f"; differ: {'; '.join(differ[:5])}" if differ else ""))

    info("llama-server load", f"{up_s:.1f} s to /health ok")
    if "E" in replies:
        t = replies["E"]["timings"]
        info("llama-server E", f"TTFT {t.get('prompt_ms', 0):.0f} ms for {t.get('prompt_n')} prompt tokens; "
                               f"{t.get('predicted_per_token_ms', 0):.1f} ms per output token "
                               f"({t.get('predicted_n')} tokens)")
    if "B2" in replies and "B2-scratch" in replies:
        r, s = replies["B2"]["timings"], replies["B2-scratch"]["timings"]
        info("llama-server B", f"follow-up prompt {r.get('prompt_ms', 0):.0f} ms for {r.get('prompt_n')} tokens with "
                               f"the cache, {s.get('prompt_ms', 0):.0f} ms for {s.get('prompt_n')} from scratch")
    if a is not None:
        info("llama-server text", " ".join(a["content"].split())[:90])
    if failure:
        info("llama-server requests", failure)


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
    ppl, problems, fa_auto, enabled_log = {}, {}, None, ""
    for name, fa, extra, drop in runs:
        rc, out, err = run([*cmd, *fa], logs / f"ppl-{name}", env_extra={**extra, **preload}, env_drop=drop,
                           timeout=3600)
        if name == "enabled":
            enabled_log = out + err
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
    # GGML_FLAGOS_DISABLE_OPS=FLASH_ATTN_EXT: the provider refuses the op and the CPU runs attention; the perplexity
    # of 2 chunks against the enabled run's over the same chunks, its running estimate after [2] (what a --chunks 2
    # run reports as its final estimate)
    rc, out, err = run([str(bdir / "bin" / "llama-perplexity"), "-m", model, "-f", text, "-c", "512", "--chunks", "2",
                        "-t", "8", "-fa", "on", "-lv", "4"], logs / "ppl-disabled-ops",
                       env_extra={"GGML_FLAGOS_DISABLE_OPS": "FLASH_ATTN_EXT", **preload},
                       env_drop=("FLAGOS_SPACEMIT_DISABLE",), timeout=3600)
    # attention on the CPU splits every layer in two (74 per Qwen3-4B token, attention_on_provider), so splits at or
    # above the layer count say the op left the provider. Not attention_on_provider itself: under -fa on llama.cpp
    # makes no auto check, and its flash-attention state is None
    found = re.findall(r"graph splits = (\d+)(?: \(with bs=\d+\), (\d+) \(with bs=1\))?", err)
    splits = [int(n) for n in found[-1] if n] if found else []
    m = re.search(r"print_info: n_layer\s+=\s+(\d+)", err)
    layers = int(m.group(1)) if m else None
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", out + err)
    value = float(m.group(1)) if rc == 0 and m else None
    m = re.search(r"\[2\]([0-9.]+)", enabled_log)
    ref = float(m.group(1)) if m else None
    both = None not in (value, ref)
    ok = bool(splits) and layers is not None and max(splits) >= layers and both and abs(value / ref - 1.0) <= 0.01
    missing = "; ".join(s for s in (
        "" if splits else "no graph splits line",
        "" if layers is not None else "no n_layer line",
        "" if value is not None else
        (f"disabled-ops run gave no result: rc={rc}, last log line: {last_line(err or out)[:100]}" +
         (f"; segv.so report in {logs / 'ppl-disabled-ops.err'}" if "=== segv.so" in err else "")),
        "" if ref is not None else "no [2] running estimate in the enabled run's log") if s)
    detail = (f"graph splits {'/'.join(map(str, splits)) or '?'}, layers {layers or '?'}; PPL "
              f"{'?' if value is None else f'{value:.4f}'} against the enabled run's first two chunks "
              f"{'?' if ref is None else f'{ref:.4f}'}" + (f" ({100.0 * (value / ref - 1.0):+.2f}%)" if both else ""))
    report("ops disabled by name leave the provider", ok, "; ".join(s for s in (detail, missing) if s))
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
    ap.add_argument("--skip-server", action="store_true", help="skip the llama-server checks")
    ap.add_argument("--server-baseline", default="", metavar="JSON",
                    help="a saved logs/server.json: also check that llama-server's replies equal it")
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
    if not args.skip_server:
        server_checks(bdir, logs, args.model, args.server_baseline or None)
    perplexity(bdir, logs, args.model, args.ppl_text)
    llama_bench(bdir, logs, args.model, args.ime_build, args.tcm_dir, args.depth)
    tcm_hygiene(args.tcm_dir, logs)

    n_fail = results.count(False)
    say(f"\nACCEPTANCE: {len(results) - n_fail}/{len(results)} checks passed" + ("" if n_fail == 0 else f", see {logs}"))
    summary.close()
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
