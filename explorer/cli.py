"""Command-line entry point for the A2 pipeline: python -m explorer.cli <step> [options].

One sub-command per pipeline step (plan §12.2; Appendix D lists them in run order). Each step
prints what it read, what it wrote and how long it took, and appends the same lines to
reports/logs/<step>.log, so the report's processing numbers come from logs rather than memory.

Usage (from the repository root):
    python -m explorer.cli snapshot               # DS1: select, fetch (or read the cache), verify, manifest
    python -m explorer.cli snapshot --rule-only   # write the selection rule into the manifest, fetch nothing
    python -m explorer.cli snapshot --refetch     # clear the DS1 cache and start again from a new tip
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from .esplora import EsploraClient
from .settings import PROJECT_ROOT, SETTINGS

LOG_DIR = PROJECT_ROOT / "reports" / "logs"

log = logging.getLogger("explorer")


def setup_logging(step: str) -> None:
    """Log INFO lines to the console and append them to reports/logs/<step>.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / f"{step}.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def esplora_clients() -> tuple[EsploraClient, EsploraClient]:
    """The primary (Blockstream) and fallback (mempool.space) clients, paced as settings.json says."""
    src, col = SETTINGS["source"], SETTINGS["collection"]
    return tuple(
        EsploraClient(base, col["requestPauseSeconds"], col["timeoutSeconds"], col["maxRetries"])
        for base in (src["apiBase"], src["fallbackApiBase"])
    )


def cmd_snapshot(args: argparse.Namespace) -> int:
    from .snapshot import run_snapshot

    primary, fallback = esplora_clients()
    return run_snapshot(primary, fallback, SETTINGS["collection"], refetch=args.refetch, rule_only=args.rule_only)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m explorer.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True, metavar="<step>")

    snapshot = steps.add_parser("snapshot", help="DS1: select, fetch, verify and write the manifest")
    snapshot.add_argument("--refetch", action="store_true",
                          help="clear data/raw/esplora and fetch everything again from a new tip")
    snapshot.add_argument("--rule-only", action="store_true",
                          help="write the selection rule into the manifest and stop before any request")
    snapshot.set_defaults(func=cmd_snapshot)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.step)
    log.info("Step %s started: %s", args.step, " ".join(sys.argv))
    started = time.monotonic()
    try:
        code = args.func(args)
    except KeyboardInterrupt:
        # Every reply is cached as it arrives, so nothing fetched so far is lost.
        log.info("Step %s interrupted after %.0f s; run the same command to resume", args.step, time.monotonic() - started)
        return 130
    log.info("Step %s finished in %.0f s (exit %d)", args.step, time.monotonic() - started, code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
