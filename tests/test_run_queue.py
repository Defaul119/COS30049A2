"""Tests for tools/run_queue.py: steps run in order, finished steps are skipped on a restart, a failure skips only
the steps that need it, and a zero exit without the step's files counts as a failure.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RUN_QUEUE = Path(__file__).resolve().parents[1] / "tools" / "run_queue.py"


def touch(path: Path) -> list[str]:
    """`run` arguments for a step that appends one line to `path` (so a test can count the runs)."""
    return ["-c", f"open({str(path)!r}, 'a').write('ran\\n')"]


def run_queue(tmp: Path, steps: list[dict]) -> tuple[int, dict, str]:
    """Run the queue on a plan; returns (exit code, the state file, the queue log)."""
    plan = tmp / "plan.json"
    plan.write_text(json.dumps({"steps": steps}), encoding="utf-8")
    result = subprocess.run([sys.executable, str(RUN_QUEUE), "--plan", str(plan), "--state", str(tmp / "state.json"),
                             "--log-dir", str(tmp / "logs"), "--supervise-log", str(tmp / "supervise.log"),
                             "--poll", "0.1", "--max-restarts", "0", "--restart-delay", "0"],
                            capture_output=True, text=True, timeout=120)
    state = json.loads((tmp / "state.json").read_text(encoding="utf-8"))
    return result.returncode, state, (tmp / "logs" / "queue.log").read_text(encoding="utf-8")


class RunQueueTests(unittest.TestCase):
    def test_steps_run_in_order_and_a_restart_skips_finished_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            steps = [{"name": "a", "run": touch(tmp / "a.txt"), "check": [str(tmp / "a.txt")]},
                     {"name": "b", "after": ["a"], "run": touch(tmp / "b.txt"), "wait_for": [str(tmp / "a.txt")]}]
            code, state, log = run_queue(tmp, steps)
            self.assertEqual(code, 0)
            self.assertEqual({s["status"] for s in state.values()}, {"done"})
            self.assertLess(log.index("a: done"), log.index("b: start"))
            run_queue(tmp, steps)
            self.assertEqual((tmp / "a.txt").read_text().count("ran"), 1)  # not run again
            self.assertEqual((tmp / "b.txt").read_text().count("ran"), 1)

    def test_a_failure_skips_only_the_steps_that_need_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            steps = [{"name": "broken", "run": ["-c", "import sys; sys.exit(3)"]},
                     {"name": "needs-broken", "after": ["broken"], "run": touch(tmp / "x.txt")},
                     {"name": "independent", "run": touch(tmp / "y.txt")}]
            code, state, log = run_queue(tmp, steps)
            self.assertEqual(code, 1)
            self.assertEqual(state["broken"]["status"], "failed")
            self.assertEqual(state["needs-broken"]["status"], "skipped")
            self.assertEqual(state["independent"]["status"], "done")
            self.assertFalse((tmp / "x.txt").exists())
            self.assertIn("not done: broken, needs-broken", log)

    def test_a_zero_exit_without_the_steps_files_is_a_failure(self):
        # lm.train returns 0 after a Ctrl+C has saved a checkpoint; the check files catch that.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            code, state, _ = run_queue(tmp, [{"name": "a", "run": ["-c", "pass"], "check": [str(tmp / "model.pt")]}])
            self.assertEqual(code, 1)
            self.assertEqual(state["a"]["status"], "failed")
            self.assertIn("model.pt", state["a"]["reason"])

    def test_the_real_plan_is_valid(self):
        plan = RUN_QUEUE.parents[1] / "configs" / "queue_after_pretraining.json"
        result = subprocess.run([sys.executable, str(RUN_QUEUE), "--plan", str(plan), "--dry-run",
                                 "--state", str(Path(tempfile.gettempdir()) / "no-such-queue-state.json")],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        steps = json.loads(plan.read_text(encoding="utf-8"))["steps"]
        self.assertEqual(len(result.stdout.splitlines()), len(steps))  # one line per step


if __name__ == "__main__":
    unittest.main()
