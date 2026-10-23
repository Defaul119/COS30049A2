"""One-screen status of the long jobs: the DS2 crawl, pretraining, the crawl's supervisor and the GPU.

Reads only files the jobs already write (and nvidia-smi, if present), so it never disturbs them.

Usage (from the repository root, any environment with Python 3.10+):
    python tools/status.py              # once
    python tools/status.py --watch 60   # refresh every 60 seconds; Ctrl+C stops only this view
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISITS = ROOT / "data" / "interim" / "corpus" / "visits.jsonl"
CORPUS_LOG = ROOT / "reports" / "logs" / "corpus.log"
SUPERVISE_LOG = ROOT / "reports" / "logs" / "supervise.log"
TRAIN_LOG = ROOT / "runs" / "text-150m" / "console.log"
SETTINGS = ROOT / "config" / "settings.json"
# The line each crawl run logs when it starts: "Resuming with N visited, K kept; target T".
RESUMED = re.compile(r"^(\S+ \S+) INFO Resuming with \d+ visited, (\d+) kept")
MIN_RUN_MINUTES = 20


def tail(path: Path, n: int = 1) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 64 * 1024))
        return f.read().decode("utf-8", errors="replace").splitlines()[-n:]


def crawl() -> str:
    if not VISITS.exists():
        return "DS2 crawl: no visits yet"
    lines = VISITS.read_text(encoding="utf-8").splitlines()
    kept = sum('"reason":"kept"' in line for line in lines)
    target = json.loads(SETTINGS.read_text(encoding="utf-8"))["corpus"]["targetAddresses"]
    idle = (time.time() - VISITS.stat().st_mtime) / 60
    text = f"DS2 crawl: {kept:,} of {target:,} kept, {len(lines):,} visited; last visit {idle:.0f} min ago"
    # Rate over the current run only (since its "Resuming with ..." line), so time the crawl spent stopped
    # never lowers the forecast. A new run starts slowly (it first re-fetches the addresses that were mid-fetch),
    # so the rate is shown only after MIN_RUN_MINUTES.
    lines_log = CORPUS_LOG.read_text(encoding="utf-8", errors="replace").splitlines() if CORPUS_LOG.exists() else []
    starts = [(datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S"), int(m[2])) for m in map(RESUMED.match, lines_log) if m]
    if starts:
        t0, k0 = starts[-1]
        minutes = (datetime.now() - t0).total_seconds() / 60
        if minutes < MIN_RUN_MINUTES:
            text += f"\n           rate: measuring (this run started {minutes:.0f} min ago at {k0:,} kept)"
        elif kept > k0:
            rate = (kept - k0) / minutes
            hours = (target - kept) / rate / 60
            finish = datetime.fromtimestamp(time.time() + hours * 3600)
            text += (f"\n           ~{rate:.1f} kept/min since {t0:%H:%M} -> {target:,} in about {hours:.0f} h "
                     f"({finish:%a %H:%M})")
    if idle > 15:
        text += "\n           WARNING: no visit for over 15 minutes; check the crawl window and supervise.log"
    return text


def supervisor() -> str:
    last = tail(SUPERVISE_LOG)
    return f"Supervisor: {last[0]}" if last else "Supervisor: no log yet"


def training() -> str:
    last = [line for line in tail(TRAIN_LOG, 5) if line.startswith("[train] step")]
    if not last:
        return "Pretraining: no step lines yet"
    idle = (time.time() - TRAIN_LOG.stat().st_mtime) / 60
    text = "Pretraining: " + last[-1].replace("[train] ", "")
    if idle > 5:
        text += f"\n             WARNING: log not written for {idle:.0f} min; check the PRETRAINING window"
    return text


def gpu() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        return f"GPU: {out} (temp, clock, power, utilisation)" if out else "GPU: nvidia-smi gave no output"
    except (OSError, subprocess.TimeoutExpired):
        return "GPU: nvidia-smi not available"


def show() -> None:
    print(f"--- {datetime.now():%a %d %b %H:%M:%S} ---")
    for part in (crawl(), supervisor(), training(), gpu()):
        print(part)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--watch", type=float, metavar="SECONDS", help="refresh every SECONDS until Ctrl+C")
    args = p.parse_args()
    if not args.watch:
        show()
        return
    try:
        while True:
            show()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
