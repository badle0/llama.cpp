#!/usr/bin/env python3
# M1 acceptance run for the SpacemiT provider (plan.md, M1). Run on the K3; standard library only.
#   1. (--build) configure and build a FlagOS build with GGML_FLAGOS_SPACEMIT=ON
#   2. FlagOS Common checks and flagos-check-spacemit (device present; absent with FLAGOS_SPACEMIT_DISABLE=1)
#   3. llama.cpp lists the device
#   4. test-backend-ops: the provider claims no op
#   5. a model gives identical output with the provider enabled and disabled, and nothing is placed on it
# Usage: m1_check.py [--build] [--build-dir DIR] [--model GGUF] [--skip-support]
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
DEVICE = "FlagOS:SpacemiT:0"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


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


def build(bdir, logs):
    cfg = ["cmake", "-S", str(REPO), "-B", str(bdir), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_FLAGOS=ON",
           "-DGGML_FLAGOS_DENGLIN=OFF", "-DGGML_FLAGOS_AMD=OFF", "-DGGML_FLAGOS_SPACEMIT=ON"]
    rc, out, _ = run(cfg, logs / "configure", timeout=600)
    report("configure", rc == 0 and "enabling SpacemiT K3 provider" in out,
           f"rc={rc}" + ("" if "enabling SpacemiT K3 provider" in out else ", provider not enabled"))
    if rc != 0:
        return False
    targets = ["ggml-flagos", "flagos-check-provider", "flagos-check-target", "flagos-check-graph-plan",
               "flagos-check-registry", "flagos-check-spacemit", "test-backend-ops", "llama-completion"]
    rc, out, err = run(["cmake", "--build", str(bdir), "--target", *targets], logs / "build", timeout=3600)
    errors = [ln for ln in (out + err).splitlines() if "error:" in ln or "FAILED" in ln]
    report("build", rc == 0 and not errors, f"rc={rc}, {len(errors)} error lines" + (f": {errors[0][:120]}" if errors else ""))
    return rc == 0


def check_tools(bdir, logs):
    for tool in ["provider", "target", "graph-plan", "registry"]:
        rc, out, err = run([str(bdir / "bin" / f"flagos-check-{tool}")], logs / f"check-{tool}")
        last = (out if rc == 0 else err or out).strip().splitlines()
        report(f"flagos-check-{tool}", rc == 0, (last[-1] if last else "no output") + ("" if rc == 0 else f" (rc={rc})"))
    rc, out, err = run([str(bdir / "bin" / "flagos-check-spacemit"), "--expect-device"], logs / "check-spacemit")
    detail = "; ".join(ln for ln in out.splitlines() if ln.startswith(("device", "flagos ")))
    fails = [ln for ln in err.splitlines() if ln.startswith("FAIL")]
    report("flagos-check-spacemit (device present)", rc == 0, detail if rc == 0 else "; ".join(fails[:3]) or err[-200:])
    rc, out, _ = run([str(bdir / "bin" / "flagos-check-spacemit"), "--expect-none"], logs / "check-spacemit-disabled",
                     env_extra={"FLAGOS_SPACEMIT_DISABLE": "1"})
    report("flagos-check-spacemit (FLAGOS_SPACEMIT_DISABLE=1)", rc == 0, "device absent" if rc == 0 else f"rc={rc}")


def list_devices(bdir, logs):
    rc, out, err = run([str(bdir / "bin" / "llama-completion"), "--list-devices"], logs / "list-devices")
    line = next((ln.strip() for ln in (out + err).splitlines() if DEVICE in ln), "")
    report("llama.cpp device list", bool(line), line or f"{DEVICE} not listed (rc={rc})")


def support(bdir, logs):
    rc, out, _ = run([str(bdir / "bin" / "test-backend-ops"), "support", "-b", DEVICE], logs / "support", timeout=1800)
    clean = ANSI.sub("", out)
    n_yes = sum(1 for ln in clean.splitlines() if ln.rstrip().endswith(" SUPPORTED") and "NOT SUPPORTED" not in ln)
    n_no = clean.count("NOT SUPPORTED")
    report("test-backend-ops support", rc == 0 and n_yes == 0 and n_no > 0,
           f"{n_yes} op cases claimed, {n_no} not claimed (rc={rc})")


def model_run(bdir, logs, model):
    if not Path(model).is_file():
        report("model run", False, f"model not found: {model}")
        return
    cmd = [str(bdir / "bin" / "llama-completion"), "-m", model, "-p", "The capital of France is", "-n", "32",
           "--temp", "0", "-no-cnv", "-t", "8", "-lv", "4"]
    rc_on, text_on, log_on = run(cmd, logs / "model-enabled", env_drop=("FLAGOS_SPACEMIT_DISABLE",))
    rc_off, text_off, log_off = run(cmd, logs / "model-disabled", env_extra={"FLAGOS_SPACEMIT_DISABLE": "1"})
    report("model runs", rc_on == 0 and rc_off == 0, f"rc enabled={rc_on}, disabled={rc_off}")
    report("model output identical", text_on == text_off and len(text_on) > 0, " ".join(text_on.split())[:90])

    placed = [(m.group(1), float(m.group(2))) for m in
              re.finditer(r"FlagOS:SpacemiT\S*\s+(model|KV|compute) buffer size\s*=\s*([0-9.]+)", log_on)]
    on_provider = [f"{kind} {size} MiB" for kind, size in placed if size > 0]
    report("nothing placed on the provider", not on_provider,
           "no provider buffers in use" if not on_provider else ", ".join(on_provider))

    splits = [re.findall(r"graph splits = ([^\n]*)", log) for log in (log_on, log_off)]
    report("graph splits unchanged", splits[0] == splits[1] and bool(splits[0]),
           f"enabled {splits[0]}, disabled {splits[1]}")


def main():
    ap = argparse.ArgumentParser(description="M1 acceptance run for the FlagOS SpacemiT provider (run on the K3)")
    ap.add_argument("--build", action="store_true", help="configure and build before checking")
    ap.add_argument("--build-dir", default=str(REPO / "build-flagos"), help="FlagOS build directory (default: %(default)s)")
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-0.6B-Q4_0.gguf"), help="model for the output check")
    ap.add_argument("--skip-support", action="store_true", help="skip the test-backend-ops support listing")
    args = ap.parse_args()

    bdir = Path(args.build_dir).resolve()
    logs = bdir / "m1-logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"repo {REPO}\nbuild {bdir}\nlogs {logs}\n", flush=True)

    if args.build and not build(bdir, logs):
        sys.exit(1)
    check_tools(bdir, logs)
    list_devices(bdir, logs)
    if not args.skip_support:
        support(bdir, logs)
    model_run(bdir, logs, args.model)

    n_fail = results.count(False)
    print(f"\nM1: {len(results) - n_fail}/{len(results)} checks passed" + ("" if n_fail == 0 else f", see {logs}"))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
