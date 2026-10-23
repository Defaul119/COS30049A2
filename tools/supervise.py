"""Run a long job and restart it after a crash, so an unattended run does not stop for good.

Why: on 23 Oct 2026 the DS2 crawl died overnight on an error nobody had caught, and nobody was
there to restart it. Every long job in this project resumes from its own saved state when run
again (DS2 from its visit log, training from its checkpoint), so restarting after a crash is
always safe.

What counts as a crash: any exit code except 0 (the job finished) and 130 (a deliberate Ctrl+C,
which explorer.cli returns). A Ctrl+C in the job's console reaches this supervisor too; it then
waits for the job to stop cleanly and does not restart it. If the job crashes within
`--min-runtime` seconds `--max-quick` times in a row (a bug at start-up, not a flaky network), the
supervisor gives up instead of looping.

Every start, exit and restart is appended to reports/logs/supervise.log.

Usage (from the repository root, in the job's own console window):
    python tools/supervise.py --name corpus -- python -m explorer.cli corpus
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "reports" / "logs" / "supervise.log"
DONE, INTERRUPTED = 0, 130


def note(name: str, message: str, log: Path = LOG) -> None:
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC [{name}] {message}"
    print(line, flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="label for the log")
    p.add_argument("--delay", type=float, default=60, help="seconds to wait before a restart")
    p.add_argument("--min-runtime", type=float, default=120, help="a run shorter than this counts as a quick crash")
    p.add_argument("--max-quick", type=int, default=3, help="give up after this many quick crashes in a row")
    p.add_argument("--log", type=Path, default=LOG, help="where to append the start/exit/restart lines")
    p.add_argument("command", nargs=argparse.REMAINDER, help="the job, after --")
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        p.error("give the job's command after --")

    quick = 0
    while True:
        note(args.name, log=args.log, message=f"start: {' '.join(command)}")
        started = time.monotonic()
        job = subprocess.Popen(command)
        try:
            code = job.wait()
        except KeyboardInterrupt:
            # The same Ctrl+C reached the job; let it save and stop, then stop too.
            code = job.wait()
            note(args.name, log=args.log, message=f"Ctrl+C: the job stopped with exit code {code}; not restarting")
            return INTERRUPTED
        runtime = time.monotonic() - started
        if code in (DONE, INTERRUPTED):
            note(args.name, log=args.log, message=f"exit code {code} after {runtime:.0f} s: {'finished' if code == DONE else 'stopped'}; not restarting")
            return code
        quick = quick + 1 if runtime < args.min_runtime else 0
        if quick >= args.max_quick:
            note(args.name, log=args.log, message=f"exit code {code}: {quick} crashes within {args.min_runtime:.0f} s in a row; giving up")
            return code
        note(args.name, log=args.log, message=f"CRASH: exit code {code} after {runtime:.0f} s; restarting in {args.delay:.0f} s")
        try:
            time.sleep(args.delay)
        except KeyboardInterrupt:
            note(args.name, log=args.log, message="Ctrl+C while waiting to restart; not restarting")
            return INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
