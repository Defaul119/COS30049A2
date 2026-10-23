"""Tests for tools/supervise.py: a crash is restarted, a finish or a Ctrl+C exit is not, a crash loop stops.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SUPERVISE = Path(__file__).resolve().parents[1] / "tools" / "supervise.py"


def supervise(tmp: Path, job: str, *extra: str) -> tuple[int, str]:
    """Run the supervisor on a Python one-liner; returns (exit code, the supervise log)."""
    log = tmp / "supervise.log"
    result = subprocess.run([sys.executable, str(SUPERVISE), "--name", "test", "--delay", "0", "--log", str(log), *extra,
                             "--", sys.executable, "-c", job], capture_output=True, text=True, timeout=60)
    return result.returncode, log.read_text(encoding="utf-8")


class SuperviseTests(unittest.TestCase):
    def test_a_crash_is_restarted_until_the_job_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "ran-once"
            # Crashes on the first run, finishes on the second (like the crawl resuming after a network error).
            job = (f"import pathlib, sys; m = pathlib.Path({str(marker)!r})\n"
                   "if m.exists(): sys.exit(0)\nm.touch(); sys.exit(1)")
            code, log = supervise(Path(tmp), job, "--min-runtime", "0")
            self.assertEqual(code, 0)
            self.assertEqual(log.count("start:"), 2)
            self.assertIn("CRASH: exit code 1", log)
            self.assertIn("finished; not restarting", log)

    def test_a_deliberate_stop_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, log = supervise(Path(tmp), "import sys; sys.exit(130)")
            self.assertEqual(code, 130)
            self.assertEqual(log.count("start:"), 1)

    def test_a_crash_loop_at_start_up_gives_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, log = supervise(Path(tmp), "import sys; sys.exit(3)", "--max-quick", "3")
            self.assertEqual(code, 3)
            self.assertEqual(log.count("start:"), 3)
            self.assertIn("giving up", log)


if __name__ == "__main__":
    unittest.main()
