"""One-screen status of the long jobs: the DS2 crawl, pretraining, the crawl's supervisor, the queue and the GPU.

Reads only files the jobs already write (and nvidia-smi, if present), so it never disturbs them.

Usage (from the repository root, any environment with Python 3.10+):
    python tools/status.py              # once
    python tools/status.py --watch 60   # refresh every 60 seconds; Ctrl+C stops only this view
    python tools/status.py --watch 60 --kaggle tritran72/cos30049a2-qwen3b-explain   # plus the Kaggle run
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISITS = ROOT / "data" / "interim" / "corpus" / "visits.jsonl"
CORPUS_LOG = ROOT / "reports" / "logs" / "corpus.log"
SUPERVISE_LOG = ROOT / "reports" / "logs" / "supervise.log"
TRAIN_LOG = ROOT / "runs" / "text-150m" / "console.log"
QUEUE_LOGS = ROOT / "reports" / "logs" / "queue"
QUEUE_STATE = ROOT / "reports" / "logs" / "queue_state_queue_after_pretraining.json"
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
    if kept >= target:
        text += "\n           done: target reached"
    elif idle > 15:
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


def queue() -> str:
    """tools/run_queue.py: its last note, step counts, and the running step's latest output line."""
    last = tail(QUEUE_LOGS / "queue.log")
    if not last:
        return "Queue: not started (python tools/run_queue.py --plan configs/queue_after_pretraining.json)"
    text = f"Queue: {last[0]}"
    state = json.loads(QUEUE_STATE.read_text(encoding="utf-8")) if QUEUE_STATE.exists() else {}
    if state:
        counts = {s: sum(v["status"] == s for v in state.values()) for s in ("done", "failed", "skipped")}
        text += f"\n       {counts['done']} done, {counts['failed']} failed, {counts['skipped']} skipped"
        running = [(name, v) for name, v in state.items() if v["status"] == "running"]
        if running:
            name, v = running[-1]
            since = datetime.strptime(v["time"], "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc).astimezone()
            text += f"; running {name} (step {v['position']}, since {since:%a %H:%M})"
            line = tail(QUEUE_LOGS / f"{name}.log")
            if line:
                text += f"\n       {line[0][:160]}"
        failed = [name for name, v in state.items() if v["status"] == "failed"]
        if failed:
            text += f"\n       FAILED: {', '.join(failed)} (reports/logs/queue/<name>.log)"
    return text


def kaggle_cli() -> str | None:
    """The Kaggle CLI: on PATH, or in the GPU environment next to this interpreter's environments."""
    found = shutil.which("kaggle")
    if found:
        return found
    candidate = Path(sys.executable).parents[1] / "cos30049-a2-gpu" / "Scripts" / "kaggle.exe"
    return str(candidate) if candidate.exists() else None


def kaggle(kernel: str) -> str:
    """The latest run of a Kaggle notebook (kaggle/qwen3b_explain.ipynb): its status, from the Kaggle API."""
    cli = kaggle_cli()
    if not cli:
        return "Kaggle: CLI not found (pip install kaggle in cos30049-a2-gpu)"
    try:
        out = subprocess.run([cli, "kernels", "status", kernel], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "Kaggle: no answer from the API this time"
    status = out.rsplit("status", 1)[-1].strip(' "').replace("KernelWorkerStatus.", "") if out else "unknown"
    return f"Kaggle: {kernel} {status} (live log: kaggle kernels logs -f {kernel})"


def gpu() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        return f"GPU: {out} (temp, clock, power, utilisation)" if out else "GPU: nvidia-smi gave no output"
    except (OSError, subprocess.TimeoutExpired):
        return "GPU: nvidia-smi not available"


def show(kernel: str | None = None) -> None:
    print(f"--- {datetime.now():%a %d %b %H:%M:%S} ---")
    for part in (crawl(), supervisor(), training(), queue(), gpu()):
        print(part)
    if kernel:
        print(kaggle(kernel))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--watch", type=float, metavar="SECONDS", help="refresh every SECONDS until Ctrl+C")
    p.add_argument("--kaggle", metavar="OWNER/NOTEBOOK", help="also show the latest run of this Kaggle notebook")
    args = p.parse_args()
    if not args.watch:
        show(args.kaggle)
        return
    try:
        while True:
            show(args.kaggle)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
