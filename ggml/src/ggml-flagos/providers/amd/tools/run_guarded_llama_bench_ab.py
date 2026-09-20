#!/usr/bin/env python3
"""Run one guarded llama-bench backend pair and archive comparable JSONL."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PAIR_FIELDS = (
    "model_filename",
    "model_size",
    "model_n_params",
    "n_batch",
    "n_ubatch",
    "n_threads",
    "type_k",
    "type_v",
    "n_gpu_layers",
    "n_cpu_moe",
    "split_mode",
    "no_kv_offload",
    "flash_attn",
    "embeddings",
    "no_op_offload",
    "n_prompt",
    "n_gen",
    "n_depth",
)


def parse_environment(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, setting = value.partition("=")
        if not separator or not ENV_NAME.fullmatch(name) or "\0" in setting:
            raise ValueError(f"invalid environment assignment: {value}")
        result[name] = setting
    return result


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"benchmark produced no rows: {path}")
    return rows


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def row_key(row: dict) -> tuple:
    missing = [field for field in PAIR_FIELDS if field not in row]
    if missing:
        raise ValueError("benchmark row lacks fields: " + ", ".join(missing))
    return tuple(row[field] for field in PAIR_FIELDS)


def index_rows(rows: list[dict], label: str) -> dict[tuple, dict]:
    indexed = {}
    for row in rows:
        key = row_key(row)
        if key in indexed:
            raise ValueError(f"duplicate benchmark row in {label}: {key}")
        rate = row.get("avg_ts")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"invalid avg_ts in {label}: {rate}")
        indexed[key] = row
    return indexed


def validate_expected_rows(rows: list[dict], models: list[Path], prompt: int, generation: int, label: str) -> None:
    expected = {
        (str(model), expected_prompt, expected_generation)
        for model in models
        for expected_prompt, expected_generation in ((prompt, 0), (0, generation))
    }
    actual = set()
    for row in rows:
        model = row.get("model_filename")
        n_prompt = row.get("n_prompt")
        n_gen = row.get("n_gen")
        if (not isinstance(model, str) or not model or "\0" in model or
                isinstance(n_prompt, bool) or not isinstance(n_prompt, int) or
                isinstance(n_gen, bool) or not isinstance(n_gen, int)):
            raise ValueError(f"invalid model/test identity in {label}")
        actual.add((model, n_prompt, n_gen))
    if len(rows) != len(expected) or actual != expected:
        raise ValueError(
            f"{label} did not produce exactly PP{prompt} and TG{generation} for every requested model")


def compare_rows(candidate_rows: list[dict], baseline_rows: list[dict], candidate_label: str, baseline_label: str) -> list[dict]:
    candidate = index_rows(candidate_rows, candidate_label)
    baseline = index_rows(baseline_rows, baseline_label)
    if candidate.keys() != baseline.keys():
        missing_candidate = baseline.keys() - candidate.keys()
        missing_baseline = candidate.keys() - baseline.keys()
        details = []
        if missing_candidate:
            details.append(f"missing from {candidate_label}: {len(missing_candidate)} row(s)")
        if missing_baseline:
            details.append(f"missing from {baseline_label}: {len(missing_baseline)} row(s)")
        raise ValueError("benchmark rows do not pair: " + "; ".join(details))

    comparisons = []
    def sort_key(item: tuple) -> tuple:
        n_prompt = int(item[-3])
        n_gen = int(item[-2])
        test_order = 0 if n_prompt > 0 and n_gen == 0 else 1
        return str(item[0]), test_order, n_prompt, n_gen

    for key in sorted(candidate, key=sort_key):
        candidate_row = candidate[key]
        baseline_row = baseline[key]
        candidate_rate = float(candidate_row["avg_ts"])
        baseline_rate = float(baseline_row["avg_ts"])
        n_prompt = int(candidate_row["n_prompt"])
        n_gen = int(candidate_row["n_gen"])
        if n_prompt > 0 and n_gen == 0:
            test = f"PP{n_prompt}"
        elif n_gen > 0 and n_prompt == 0:
            test = f"TG{n_gen}"
        else:
            test = f"PP{n_prompt}+TG{n_gen}"
        comparisons.append({
            "model": candidate_row["model_filename"],
            "model_type": candidate_row.get("model_type", ""),
            "test": test,
            "candidate_ts": candidate_rate,
            "baseline_ts": baseline_rate,
            "relative_percent": 100.0 * (candidate_rate / baseline_rate - 1.0),
        })
    return comparisons


def markdown_summary(comparisons: list[dict], candidate_label: str, baseline_label: str) -> str:
    lines = [
        f"| Model | Test | {candidate_label} t/s | {baseline_label} t/s | Relative |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        model = Path(row["model"]).name.replace("|", "\\|")
        lines.append(
            f"| {model} | {row['test']} | {row['candidate_ts']:.2f} | "
            f"{row['baseline_ts']:.2f} | {row['relative_percent']:+.2f}% |"
        )
    return "\n".join(lines) + "\n"


def benchmark_command(args, binary: Path, device: str, override: str) -> list[str]:
    command = [
        str(binary),
        "-m", ",".join(str(model) for model in args.model),
        "-p", str(args.prompt),
        "-n", str(args.generation),
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "-r", str(args.repetitions),
        "-ngl", str(args.gpu_layers),
        "-t", str(args.threads),
        "-o", "jsonl",
    ]
    if device:
        command += ["-dev", device]
    if override:
        command += ["-ot", override]
    return command


def run_one(guard: Path, timeout: int, command: list[str], environment: dict[str, str], stdout_path: Path, stderr_path: Path) -> int:
    guard_command = [
        sys.executable,
        str(guard),
        "--timeout", str(timeout),
        "--max-processes-per-boot", "2",
        "--",
        *command,
    ]
    with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
        return subprocess.run(
            guard_command,
            env={**os.environ, **environment},
            stdout=stdout,
            stderr=stderr,
            check=False,
        ).returncode


def checked_file(path: Path, description: str, executable: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{description} is not a file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise ValueError(f"{description} is not executable: {resolved}")
    return resolved


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--guard", type=Path, default=here / "run_guarded_benchmark.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--candidate-label", default="FlagOS")
    parser.add_argument("--baseline-label", default="native ROCm")
    parser.add_argument("--candidate-env", action="append", default=[])
    parser.add_argument("--baseline-env", action="append", default=[])
    parser.add_argument("--candidate-device", default="")
    parser.add_argument("--baseline-device", default="ROCm0")
    parser.add_argument("--candidate-override", default="")
    parser.add_argument("--baseline-override", default="")
    parser.add_argument("--model", action="append", type=Path, required=True)
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--generation", type=int, default=128)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--first", choices=("candidate", "baseline"), default="candidate")
    args = parser.parse_args()

    try:
        guard = checked_file(args.guard, "guard")
        candidate_binary = checked_file(args.candidate_binary, "candidate binary", True)
        baseline_binary = checked_file(args.baseline_binary, "baseline binary", True)
        args.model = [checked_file(model, "model") for model in args.model]
        candidate_environment = parse_environment(args.candidate_env)
        baseline_environment = parse_environment(args.baseline_env)
        if any(value <= 0 for value in (
                args.prompt, args.generation, args.batch, args.ubatch,
                args.repetitions, args.threads, args.timeout)):
            raise ValueError("benchmark sizes, repetitions, threads, and timeout must be positive")
        if args.gpu_layers < 0:
            raise ValueError("gpu-layers must not be negative")
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.exists():
            raise ValueError(f"output directory already exists: {output_dir}")
        if not output_dir.parent.is_dir():
            raise ValueError(f"output parent directory does not exist: {output_dir.parent}")
    except ValueError as error:
        print(f"FlagOS benchmark A/B: {error}", file=sys.stderr)
        return 2

    status_result = subprocess.run(
        [sys.executable, str(guard), "--check-only", "--json", "--max-processes-per-boot", "2"],
        text=True,
        capture_output=True,
        check=False,
    )
    if status_result.returncode != 0:
        print(status_result.stderr.strip() or "guard preflight failed", file=sys.stderr)
        return status_result.returncode
    try:
        guard_status = json.loads(status_result.stdout)
        remaining = guard_status.get("remaining") if isinstance(guard_status, dict) else None
        if isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 2:
            raise ValueError("current boot does not have two guarded command slots")
    except (json.JSONDecodeError, ValueError) as error:
        print(f"FlagOS benchmark A/B: invalid guard status: {error}", file=sys.stderr)
        return 2

    os.mkdir(output_dir, 0o700)
    candidate_command = benchmark_command(
        args, candidate_binary, args.candidate_device, args.candidate_override)
    baseline_command = benchmark_command(
        args, baseline_binary, args.baseline_device, args.baseline_override)
    runs = {
        "candidate": (candidate_command, candidate_environment),
        "baseline": (baseline_command, baseline_environment),
    }
    order = [args.first, "baseline" if args.first == "candidate" else "candidate"]
    metadata = {
        "boot_id": guard_status.get("boot_id"),
        "started_unix": int(time.time()),
        "status": "ready",
        "order": order,
        "candidate_label": args.candidate_label,
        "baseline_label": args.baseline_label,
        "commands": {"candidate": candidate_command, "baseline": baseline_command},
        "environment_names": {
            "candidate": sorted(candidate_environment),
            "baseline": sorted(baseline_environment),
        },
        "run_started_unix": {},
        "run_finished_unix": {},
        "return_codes": {},
    }
    run_metadata_path = output_dir / "run.json"
    write_json(run_metadata_path, metadata)
    for name in order:
        command, environment = runs[name]
        metadata["status"] = "running"
        metadata["active_run"] = name
        metadata["run_started_unix"][name] = int(time.time())
        write_json(run_metadata_path, metadata)
        return_code = run_one(
            guard, args.timeout, command, environment,
            output_dir / f"{name}.jsonl", output_dir / f"{name}.log")
        metadata["return_codes"][name] = return_code
        metadata["run_finished_unix"][name] = int(time.time())
        metadata.pop("active_run", None)
        metadata["status"] = "partial" if return_code == 0 else "failed"
        write_json(run_metadata_path, metadata)
        if return_code != 0:
            print(
                f"FlagOS benchmark A/B: {name} failed with status {return_code}; "
                f"see {output_dir / f'{name}.log'}",
                file=sys.stderr,
            )
            return return_code

    try:
        candidate_rows = read_jsonl(output_dir / "candidate.jsonl")
        baseline_rows = read_jsonl(output_dir / "baseline.jsonl")
        validate_expected_rows(
            candidate_rows, args.model, args.prompt, args.generation, args.candidate_label)
        validate_expected_rows(
            baseline_rows, args.model, args.prompt, args.generation, args.baseline_label)
        comparisons = compare_rows(
            candidate_rows,
            baseline_rows,
            args.candidate_label,
            args.baseline_label,
        )
    except ValueError as error:
        metadata["status"] = "invalid-output"
        metadata["validation_error"] = str(error)
        metadata["finished_unix"] = int(time.time())
        write_json(run_metadata_path, metadata)
        print(f"FlagOS benchmark A/B: {error}", file=sys.stderr)
        return 2
    summary = {
        "boot_id": guard_status.get("boot_id"),
        "candidate_label": args.candidate_label,
        "baseline_label": args.baseline_label,
        "comparisons": comparisons,
    }
    write_json(output_dir / "summary.json", summary)
    table = markdown_summary(comparisons, args.candidate_label, args.baseline_label)
    (output_dir / "summary.md").write_text(table)
    metadata["status"] = "complete"
    metadata["finished_unix"] = int(time.time())
    write_json(run_metadata_path, metadata)
    print(table, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
