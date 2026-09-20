#!/usr/bin/env python3
"""Run a bounded AMD benchmark while guarding the current boot's GPU health.

The Radeon 890M development host can enter a non-recoverable unified-MES state
after several separate HIP model processes, even when each individual run
finishes successfully. This wrapper does not hide that driver failure. It
prevents concurrent runs, limits process churn per boot, bounds execution time,
and refuses to continue once the kernel log reports a GPU scheduler fault.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time


GPU_ERROR = re.compile(
    r"MES failed to respond|MES ring buffer is full|ring .* timeout|"
    r"GPU reset|GPU fault|amdgpu.*page.?fault|GPU page.?fault|"
    r"GCVM_L2_PROTECTION_FAULT_STATUS|VM_L2_PROTECTION_FAULT_STATUS",
    re.IGNORECASE,
)


class KernelLogUnavailable(RuntimeError):
    """The guard cannot prove that the current boot is healthy."""


class GuardInterrupted(RuntimeError):
    def __init__(self, signum: int):
        super().__init__(f"received signal {signum}")
        self.signum = signum


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def state_path() -> Path:
    runtime = Path(f"/run/user/{os.getuid()}")
    directory = runtime if runtime.is_dir() else Path("/tmp")
    suffix = "" if directory == runtime else f"-{os.getuid()}"
    return directory / f"flagos-amd-benchmark-state{suffix}.json"


def kernel_gpu_errors() -> list[str]:
    try:
        result = subprocess.run(
            ["journalctl", "-k", "-b", "-o", "cat", "--no-pager"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except OSError as error:
        raise KernelLogUnavailable(f"cannot execute journalctl: {error}") from error
    except subprocess.TimeoutExpired as error:
        raise KernelLogUnavailable("journalctl timed out") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise KernelLogUnavailable(f"journalctl failed: {detail}")
    return [line for line in result.stdout.splitlines() if GPU_ERROR.search(line)]


def read_state(file, newly_created: bool = False) -> dict:
    file.seek(0)
    serialized = file.read()
    if not serialized:
        if newly_created:
            return {}
        raise ValueError("existing state file is empty")
    try:
        value = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise ValueError("state file is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("state file root must be an object")
    return value


def write_state(file, state: dict) -> None:
    file.seek(0)
    file.truncate()
    json.dump(state, file, indent=2)
    file.write("\n")
    file.flush()
    os.fsync(file.fileno())


def state_counter(state: dict, name: str, default: int, positive: bool = False) -> int:
    value = state.get(name, default)
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid {name} in state file")
    return value


def process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # The guard created the group, so this should not happen. Treat an
        # inaccessible group as alive and fail closed rather than claiming it
        # was cleaned up.
        return True


def wait_process_group_gone(
        process: subprocess.Popen, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while True:
        # poll() reaps the group leader when it has exited. The group can still
        # contain grandchildren, so the leader alone is not the success test.
        process.poll()
        if not process_group_exists(process.pid):
            return process.poll() is not None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))


def stop_process_group(process: subprocess.Popen, grace_seconds: int = 5) -> bool:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.poll() is not None
    if wait_process_group_gone(process, grace_seconds):
        return True
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return process.poll() is not None
    if wait_process_group_gone(process, grace_seconds):
        return True
    print(
        "FlagOS AMD guard: process group remains stuck after SIGKILL; reboot is required",
        file=sys.stderr,
    )
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=180, help="wall timeout in seconds")
    parser.add_argument(
        "--max-processes-per-boot",
        type=int,
        default=None,
        help="refuse additional processes in one boot (default: 2; check-only uses the recorded limit)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="check current-boot health and process count without starting a command",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable status with --check-only",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if args.timeout <= 0 or (args.max_processes_per_boot is not None and args.max_processes_per_boot <= 0):
        parser.error("limits must be positive")
    if args.check_only and command:
        parser.error("--check-only does not accept a command")
    if args.json and not args.check_only:
        parser.error("--json requires --check-only")
    if not args.check_only and not command:
        parser.error("a command after -- is required")

    try:
        current_boot = boot_id()
    except OSError as error:
        print(f"FlagOS AMD guard: cannot read boot ID: {error}", file=sys.stderr)
        return 6
    if not current_boot:
        print("FlagOS AMD guard: cannot read boot ID: value is empty", file=sys.stderr)
        return 6

    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    newly_created = False
    try:
        try:
            state_fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
            newly_created = True
        except FileExistsError:
            state_fd = os.open(path, flags)
        state_info = os.fstat(state_fd)
        if (not stat.S_ISREG(state_info.st_mode) or state_info.st_uid != os.getuid() or
                state_info.st_nlink != 1):
            raise OSError("state path is not a private regular file")
        os.fchmod(state_fd, 0o600)
        state_info = os.fstat(state_fd)
        if state_info.st_mode & 0o077:
            raise OSError("state file permissions are not private")
    except OSError as error:
        print(f"FlagOS AMD guard: cannot open state file: {error}", file=sys.stderr)
        return 6
    with os.fdopen(state_fd, "r+", encoding="utf-8") as state_file:
        try:
            fcntl.flock(state_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("FlagOS AMD guard: another guarded benchmark is already running", file=sys.stderr)
            return 2

        try:
            state = read_state(state_file, newly_created)
        except (OSError, ValueError) as error:
            print(f"FlagOS AMD guard: refusing to run: {error}", file=sys.stderr)
            return 6
        recorded_boot = state.get("boot_id")
        if newly_created:
            state = {"boot_id": current_boot, "commands_started": 0, "unhealthy": False}
            # Publish a valid initial record before journal access or any
            # other operation that can fail.  Otherwise a transient first-run
            # failure leaves an empty O_EXCL-created file, and every later run
            # correctly-but-permanently rejects it as corrupt.
            try:
                write_state(state_file, state)
            except OSError as error:
                print(f"FlagOS AMD guard: cannot initialize state file: {error}", file=sys.stderr)
                return 6
        elif not isinstance(recorded_boot, str) or not recorded_boot:
            print("FlagOS AMD guard: refusing to run: invalid boot_id in state file", file=sys.stderr)
            return 6
        elif recorded_boot != current_boot:
            state = {"boot_id": current_boot, "commands_started": 0, "unhealthy": False}

        try:
            started = state_counter(state, "commands_started", 0) \
                if "commands_started" in state else state_counter(state, "processes_started", 0)
            succeeded = state_counter(state, "commands_succeeded", 0)
            failed = state_counter(state, "commands_failed", 0)
            launch_attempts = state_counter(state, "launch_attempts", started)
            limit = args.max_processes_per_boot
            limit_recorded = "last_process_limit" in state
            if limit is None:
                limit = state_counter(state, "last_process_limit", 2, positive=True)
        except ValueError as error:
            print(f"FlagOS AMD guard: refusing to run: {error}", file=sys.stderr)
            return 6

        if state.get("launch_in_progress") is not None and not isinstance(
                state.get("launch_in_progress"), bool):
            print(
                "FlagOS AMD guard: refusing to run: invalid launch_in_progress in state file",
                file=sys.stderr,
            )
            return 6
        if "unhealthy" in state and not isinstance(state.get("unhealthy"), bool):
            print("FlagOS AMD guard: refusing to run: invalid unhealthy in state file", file=sys.stderr)
            return 6
        if succeeded + failed > started or launch_attempts < started:
            print("FlagOS AMD guard: refusing to run: inconsistent counters in state file", file=sys.stderr)
            return 6
        if state.get("launch_in_progress"):
            state["unhealthy"] = True
            state["last_guard_error"] = "previous guarded command did not reach a clean terminal state"
            write_state(state_file, state)
            print(
                "FlagOS AMD guard: previous guarded command did not finish cleanly; "
                "current boot is blocked",
                file=sys.stderr,
            )
            return 3

        try:
            errors = kernel_gpu_errors()
        except KernelLogUnavailable as error:
            print(f"FlagOS AMD guard: refusing to run: {error}", file=sys.stderr)
            return 5
        if errors:
            state["unhealthy"] = True
            state["last_gpu_error"] = errors[-1]
            write_state(state_file, state)
            print(f"FlagOS AMD guard: current boot is unhealthy: {errors[-1]}", file=sys.stderr)
            return 3
        if state.get("unhealthy"):
            print("FlagOS AMD guard: current boot was previously marked unhealthy", file=sys.stderr)
            return 3

        if args.check_only:
            state["last_health_check_unix"] = int(time.time())
            write_state(state_file, state)
            if args.json:
                print(json.dumps({
                    "boot_id": current_boot,
                    "healthy": True,
                    "commands_started": started,
                    "commands_succeeded": succeeded,
                    "commands_failed": failed,
                    "limit": limit,
                    "remaining": max(0, limit - started),
                }, separators=(",", ":")))
                return 0
            if limit_recorded or args.max_processes_per_boot is not None:
                count_text = f"{started}/{limit} guarded commands started"
            else:
                count_text = f"{started} guarded commands started; no legacy limit was recorded (next-run default: {limit})"
            print(
                f"FlagOS AMD guard: boot {current_boot} is healthy; "
                f"{count_text}; "
                f"{succeeded} succeeded, {failed} failed"
            )
            return 0
        if started >= limit:
            print(
                "FlagOS AMD guard: per-boot model-process limit reached; "
                "use repetitions inside one process or reboot before another A/B pair",
                file=sys.stderr,
            )
            return 4

        # Install the handlers before publishing launch_in_progress or calling
        # Popen.  A terminal hangup in the old spawn-to-wait window could leave
        # the new process group running after the guard had exited.  During
        # Popen and the following state fsync the handler only records the first
        # signal: once Popen returns we have a reliable process-group ID and can
        # terminate it.  During wait it raises so the blocking call is escaped
        # immediately.
        previous_handlers = {}
        interrupted_signum = None
        raise_interrupt = False
        process = None

        def interrupt(signum, _frame):
            nonlocal interrupted_signum
            if interrupted_signum is None:
                interrupted_signum = signum
            if raise_interrupt:
                raise GuardInterrupted(interrupted_signum)

        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGQUIT):
            previous_handlers[signum] = signal.signal(signum, interrupt)
        terminated_cleanly = True
        try:
            state["launch_attempts"] = launch_attempts + 1
            state["launch_in_progress"] = True
            state["last_launch_attempt_unix"] = int(time.time())
            try:
                write_state(state_file, state)
            except OSError as error:
                print(f"FlagOS AMD guard: cannot record launch attempt: {error}", file=sys.stderr)
                return 6

            if interrupted_signum is not None:
                state["launch_in_progress"] = False
                state["last_guard_error"] = "guard interrupted before command start"
                state["last_return_code"] = 128 + interrupted_signum
                write_state(state_file, state)
                print(
                    f"FlagOS AMD guard: received signal {interrupted_signum} before command start",
                    file=sys.stderr,
                )
                return 128 + interrupted_signum

            try:
                process = subprocess.Popen(command, start_new_session=True)
            except OSError as error:
                state["launch_in_progress"] = False
                state["last_launch_error"] = str(error)
                write_state(state_file, state)
                print(f"FlagOS AMD guard: cannot start command: {error}", file=sys.stderr)
                return 127

            state.pop("processes_started", None)
            state["commands_started"] = started + 1
            state["last_process_limit"] = limit
            state["last_started_unix"] = int(time.time())
            state["last_command"] = command
            try:
                write_state(state_file, state)
            except OSError as error:
                terminated = stop_process_group(process)
                print(
                    "FlagOS AMD guard: cannot record the started command; "
                    + ("command was terminated" if terminated else "command could not be terminated; reboot is required")
                    + f": {error}",
                    file=sys.stderr,
                )
                return 6 if terminated else 3

            raise_interrupt = True
            if interrupted_signum is not None:
                raise GuardInterrupted(interrupted_signum)
            return_code = process.wait(timeout=args.timeout)
            raise_interrupt = False
            if not wait_process_group_gone(process, 1):
                print(
                    "FlagOS AMD guard: command leader exited but descendants remain; terminating",
                    file=sys.stderr,
                )
                terminated_cleanly = stop_process_group(process)
                return_code = 125
        except subprocess.TimeoutExpired:
            raise_interrupt = False
            print(
                f"FlagOS AMD guard: benchmark exceeded {args.timeout}s; terminating",
                file=sys.stderr,
            )
            terminated_cleanly = stop_process_group(process)
            return_code = 124
        except GuardInterrupted as error:
            raise_interrupt = False
            print(
                f"FlagOS AMD guard: received signal {error.signum}; terminating benchmark",
                file=sys.stderr,
            )
            terminated_cleanly = stop_process_group(process)
            return_code = 128 + error.signum
        finally:
            raise_interrupt = False
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

        if not terminated_cleanly:
            state["launch_in_progress"] = False
            state["unhealthy"] = True
            state["last_guard_error"] = "guarded process remained alive after SIGKILL"
            state["commands_failed"] = failed + 1
            state["last_return_code"] = return_code
            write_state(state_file, state)
            return 3

        try:
            errors = kernel_gpu_errors()
        except KernelLogUnavailable as error:
            state["launch_in_progress"] = False
            state["unhealthy"] = True
            state["last_log_check_error"] = str(error)
            state["last_return_code"] = return_code
            write_state(state_file, state)
            print(
                f"FlagOS AMD guard: post-run health check failed; "
                f"current boot is now blocked: {error}",
                file=sys.stderr,
            )
            return 5
        if errors:
            state["launch_in_progress"] = False
            state["unhealthy"] = True
            state["last_gpu_error"] = errors[-1]
            state["commands_failed"] = failed + 1
            state["last_return_code"] = return_code
            write_state(state_file, state)
            print(f"FlagOS AMD guard: GPU error detected after run: {errors[-1]}", file=sys.stderr)
            return 3
        counter = "commands_succeeded" if return_code == 0 else "commands_failed"
        state["launch_in_progress"] = False
        state[counter] = (succeeded if counter == "commands_succeeded" else failed) + 1
        state["last_return_code"] = return_code
        write_state(state_file, state)
        return return_code


if __name__ == "__main__":
    raise SystemExit(main())
