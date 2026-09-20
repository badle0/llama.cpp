#!/usr/bin/env python3

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "run_guarded_benchmark", HERE / "run_guarded_benchmark.py")
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


class GuardedBenchmarkTest(unittest.TestCase):
    def test_successful_leader_cannot_leave_descendant(self):
        parent_code = """
import subprocess
import sys

subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
"""
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "guard.json"
            argv = [
                "run_guarded_benchmark.py", "--timeout", "10",
                "--max-processes-per-boot", "2", "--",
                sys.executable, "-c", parent_code,
            ]
            with mock.patch.object(GUARD, "boot_id", return_value="test-boot"), \
                    mock.patch.object(GUARD, "state_path", return_value=state), \
                    mock.patch.object(GUARD, "kernel_gpu_errors", return_value=[]), \
                    mock.patch.object(sys, "argv", argv):
                self.assertEqual(GUARD.main(), 125)

            record = json.loads(state.read_text())
            self.assertFalse(record["launch_in_progress"])
            self.assertEqual(record["commands_started"], 1)
            self.assertEqual(record["commands_failed"], 1)
            self.assertEqual(record["last_return_code"], 125)

    def test_stop_process_group_kills_descendant_after_leader_exits(self):
        parent_code = """
import signal
import subprocess
import sys
import time

signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
subprocess.Popen([
    sys.executable, "-c",
    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
])
print("ready", flush=True)
time.sleep(30)
"""
        process = subprocess.Popen(
            [sys.executable, "-c", parent_code],
            start_new_session=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(process.stdout.readline().strip(), "ready")
        # Give the descendant time to exec with SIGTERM ignored. The group
        # leader exits on SIGTERM, so checking only process.wait() would return
        # a false success and leave the descendant behind.
        time.sleep(0.05)
        self.assertTrue(GUARD.stop_process_group(process, grace_seconds=1))
        self.assertFalse(GUARD.process_group_exists(process.pid))
        process.stdout.close()

    def test_signal_during_popen_terminates_new_process_group(self):
        real_popen = subprocess.Popen
        child = None

        def interrupting_popen(command, start_new_session):
            nonlocal child
            child = real_popen(command, start_new_session=start_new_session)
            # Reproduce the former race: the child exists, but Popen has not
            # returned to main and therefore its local process variable has not
            # yet been assigned.
            os.kill(os.getpid(), signal.SIGHUP)
            return child

        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "guard.json"
            argv = [
                "run_guarded_benchmark.py", "--timeout", "10",
                "--max-processes-per-boot", "2", "--", "/bin/sleep", "30",
            ]
            with mock.patch.object(GUARD, "boot_id", return_value="test-boot"), \
                    mock.patch.object(GUARD, "state_path", return_value=state), \
                    mock.patch.object(GUARD, "kernel_gpu_errors", return_value=[]), \
                    mock.patch.object(GUARD.subprocess, "Popen", side_effect=interrupting_popen), \
                    mock.patch.object(sys, "argv", argv):
                self.assertEqual(GUARD.main(), 128 + signal.SIGHUP)

            self.assertIsNotNone(child)
            self.assertIsNotNone(child.poll())
            record = json.loads(state.read_text())
            self.assertFalse(record["launch_in_progress"])
            self.assertEqual(record["commands_started"], 1)
            self.assertEqual(record["commands_failed"], 1)
            self.assertEqual(record["last_return_code"], 128 + signal.SIGHUP)


if __name__ == "__main__":
    unittest.main()
