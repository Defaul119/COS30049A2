"""Run a plan of jobs one after another, unattended: the work queued behind pretraining (24 Oct 2026).

Why: when the 152M pretraining ends, about 13 hours of GPU jobs remain (TxGPT's 9 runs and its scoring, the 152M
chat and explanation stages and every evaluation), and the CPU steps that need DS2 wait on
the crawl. Started by hand, the GPU would sit idle overnight. A plan file (configs/queue_after_pretraining.json)
lists the steps in run order; each step has

    name      label for the logs and the state file
    run       the arguments after `python` (run with this interpreter: start the queue in cos30049-a2-gpu,
              which has every package)
    wait_for  files written by jobs outside the queue (DS2's manifest, the pretrained model.pt): wait for them
    after     earlier steps that must have finished; if one failed or was skipped, this step is skipped
    gpu       wait until the GPU is free (pretraining has released its memory) before starting
    check     files the step must leave behind: a zero exit without them counts as a failure

Each step runs under tools/supervise.py, so a crash is restarted (every job here resumes from its own saved
state), and a step that keeps crashing fails without stopping the steps that don't depend on it. Finished steps
are recorded in a state file (reports/logs/queue_state_<plan>.json) and skipped on the next start, so after a
Ctrl+C (each job saves first) or a power cut the same command continues; failed and skipped steps are retried.
Each step's output is shown and also appended to reports/logs/queue/<name>.log; tools/status.py shows the queue.

Usage (from the repository root, in its own console window, environment cos30049-a2-gpu):
    python tools/run_queue.py --plan configs/queue_after_pretraining.json
    python tools/run_queue.py --plan configs/queue_after_pretraining.json --dry-run   # show the plan and state
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SUPERVISE = ROOT / "tools" / "supervise.py"
LOGS = ROOT / "reports" / "logs"
INTERRUPTED = 130


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def note(message: str, log: Path) -> None:
    line = f"{now()} [queue] {message}"
    print(line, flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def gpu_memory_used_mb() -> float | None:
    """Memory in use on the first GPU (MB), or None if nvidia-smi is not available."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return float(out[0]) if out else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def wait(ready: Callable[[], bool], what: str, poll: float, log: Path) -> None:
    """Block until ready(), checking every `poll` seconds and noting the wait once an hour."""
    if ready():
        return
    note(f"waiting for {what}", log)
    last = time.monotonic()
    while not ready():
        time.sleep(poll)
        if time.monotonic() - last > 3600:
            note(f"still waiting for {what}", log)
            last = time.monotonic()
    note(f"ready: {what}", log)


def run_step(step: dict[str, Any], args: argparse.Namespace) -> int:
    """Run one step under the supervisor, copying its output to the console and to the step's log file."""
    command = [sys.executable, str(SUPERVISE), "--name", f"queue:{step['name']}", "--log", str(args.supervise_log),
               "--delay", str(args.restart_delay), "--max-restarts", str(args.max_restarts),
               "--", sys.executable, *step["run"]]
    # Unbuffered, so a job's progress lines reach the log as they happen, not when a pipe buffer fills.
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
    args.log_dir.mkdir(parents=True, exist_ok=True)
    with (args.log_dir / f"{step['name']}.log").open("a", encoding="utf-8") as out:
        job = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")

        def copy(text: str) -> None:
            sys.stdout.write(text)
            sys.stdout.flush()
            out.write(text)
            out.flush()

        try:
            for line in job.stdout:
                copy(line)
            return job.wait()
        except KeyboardInterrupt:
            # The same Ctrl+C reached the job and its supervisor: keep reading while the job saves, so a full
            # pipe never blocks it, then stop the queue.
            copy(job.communicate()[0] or "")
            raise


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, required=True, help="the plan file (JSON with a list of steps)")
    p.add_argument("--state", type=Path, help="state file (default: reports/logs/queue_state_<plan>.json)")
    p.add_argument("--log-dir", type=Path, default=LOGS / "queue", help="queue.log and one log per step")
    p.add_argument("--supervise-log", type=Path, default=LOGS / "supervise.log")
    p.add_argument("--poll", type=float, default=60, help="seconds between checks while waiting")
    p.add_argument("--gpu-free-mb", type=float, default=3000, help="the GPU counts as free below this memory in use")
    p.add_argument("--max-restarts", type=int, default=2, help="restarts of a crashed step before it fails")
    p.add_argument("--restart-delay", type=float, default=60, help="seconds before a crashed step is restarted")
    p.add_argument("--dry-run", action="store_true", help="show each step's state and stop")
    args = p.parse_args()

    steps = json.loads(args.plan.read_text(encoding="utf-8"))["steps"]
    names: list[str] = []
    for step in steps:
        if step["name"] in names:
            p.error(f"step name {step['name']!r} appears twice")
        unknown = [a for a in step.get("after", []) if a not in names]
        if unknown:
            p.error(f"step {step['name']!r} comes after {unknown}, which are not earlier steps")
        names.append(step["name"])
    state_path = args.state or LOGS / f"queue_state_{args.plan.stem}.json"
    state: dict[str, dict] = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    log = args.log_dir / "queue.log"

    def record(name: str, status: str, **extra: Any) -> None:
        state[name] = {"status": status, "time": now(), "position": f"{names.index(name) + 1}/{len(names)}", **extra}
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(state, indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())  # on disk before the rename, so a power cut cannot leave an empty state file
        tmp.replace(state_path)

    if args.dry_run:
        for i, step in enumerate(steps, 1):
            print(f"{i:>2}. {step['name']:<20} {state.get(step['name'], {}).get('status', 'pending'):<8} "
                  f"python {' '.join(step['run'])}")
        return 0

    # Only "done" survives a restart: a step that failed or was skipped last time gets another go.
    run_status = {name: "done" for name, s in state.items() if s.get("status") == "done"}
    note(f"start: {args.plan} ({len(steps)} steps, {len(run_status)} already done)", log)
    name = None
    try:
        for step in steps:
            name = step["name"]
            if run_status.get(name) == "done":
                continue
            blocked = [a for a in step.get("after", []) if run_status.get(a) != "done"]
            if blocked:
                run_status[name] = "skipped"
                record(name, "skipped", reason=f"needs {', '.join(blocked)}")
                note(f"{name}: skipped (needs {', '.join(blocked)})", log)
                continue
            for path in step.get("wait_for", []):
                wait(lambda: (ROOT / path).exists(), path, args.poll, log)
            if step.get("gpu"):
                def gpu_free() -> bool:
                    used = gpu_memory_used_mb()
                    return used is None or used < args.gpu_free_mb
                wait(gpu_free, f"a free GPU (under {args.gpu_free_mb:.0f} MB in use; now {gpu_memory_used_mb()} MB)",
                     args.poll, log)
            record(name, "running")
            note(f"{name}: start: python {' '.join(step['run'])}", log)
            started = time.monotonic()
            code = run_step(step, args)
            minutes = (time.monotonic() - started) / 60
            if code == INTERRUPTED:
                record(name, "stopped", minutes=round(minutes, 1))
                note(f"{name}: stopped (exit code {code}); run the same command to continue", log)
                return INTERRUPTED
            missing = [c for c in step.get("check", []) if not (ROOT / c).exists()]
            if code == 0 and not missing:
                run_status[name] = "done"
                record(name, "done", minutes=round(minutes, 1))
                note(f"{name}: done in {minutes:.0f} min", log)
            else:
                run_status[name] = "failed"
                reason = f"exit code {code}" if code else f"missing {', '.join(missing)}"
                record(name, "failed", reason=reason, minutes=round(minutes, 1))
                note(f"{name}: FAILED ({reason}) after {minutes:.0f} min; steps after it continue", log)
    except KeyboardInterrupt:
        if name and state.get(name, {}).get("status") == "running":
            record(name, "stopped")
        note("stopped by Ctrl+C; run the same command to continue (finished steps are skipped)", log)
        return INTERRUPTED

    failed = [n for n, s in run_status.items() if s != "done"]
    note(f"finished: {len(names) - len(failed)} of {len(names)} steps done"
         + (f"; not done: {', '.join(failed)}" if failed else ""), log)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
