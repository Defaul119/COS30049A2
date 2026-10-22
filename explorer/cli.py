"""Command-line entry point for the A2 pipeline: python -m explorer.cli <step> [options].

One sub-command per pipeline step (plan §12.2; Appendix D lists them in run order). Each step
prints what it read, what it wrote and how long it took, and appends the same lines to
reports/logs/<step>.log, so the report's processing numbers come from logs rather than memory.

Usage (from the repository root):
    python -m explorer.cli snapshot               # DS1: select, fetch (or read the cache), verify, manifest
    python -m explorer.cli snapshot --rule-only   # write the selection rule into the manifest, fetch nothing
    python -m explorer.cli snapshot --refetch     # clear the DS1 cache and start again from a new tip
    python -m explorer.cli corpus                 # DS2: crawl for TxGPT; run again to resume
    python -m explorer.cli external               # DS3 and DS4 (or --elliptic / --bitcoinheist alone)
    python -m explorer.cli transactions           # DS1 clean table + cleaning log
    python -m explorer.cli features [--as-written]  # F1-F8, z-scores, baseline flags
    python -m explorer.cli split                  # fit / evaluation periods
    python -m explorer.cli process-external       # DS4 labels, transforms, splits
    python -m explorer.cli classify --dataset heist [--splits temporal,random,lofo]
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


def cmd_corpus(args: argparse.Namespace) -> int:
    from .corpus import ClientPool, make_clients, run_corpus

    params = dict(SETTINGS["corpus"])
    if args.target:
        params["targetAddresses"] = args.target
    src = SETTINGS["source"]
    clients = ClientPool(make_clients([src["apiBase"], src["fallbackApiBase"]], params["workersPerSource"]))
    return run_corpus(clients, params)


def cmd_transactions(args: argparse.Namespace) -> int:
    from .transactions import run_transactions

    return run_transactions()


def cmd_features(args: argparse.Namespace) -> int:
    from .features import run_features

    return run_features(as_written=args.as_written)


def cmd_split(args: argparse.Namespace) -> int:
    from .split import run_split

    return run_split()


def cmd_process_external(args: argparse.Namespace) -> int:
    from .labelled import run_process_heist

    return run_process_heist()


def cmd_classify(args: argparse.Namespace) -> int:
    from .classify import run_classify_heist

    if args.dataset == "heist":
        return run_classify_heist(args.splits.split(","), force=args.force)
    raise SystemExit("Elliptic needs its download first (python -m explorer.cli external --elliptic)")


def cmd_external(args: argparse.Namespace) -> int:
    from .external import run_external

    keys = [k for k in ("elliptic", "bitcoinheist") if getattr(args, k)] or ["elliptic", "bitcoinheist"]
    return run_external(keys)


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

    corpus = steps.add_parser("corpus", help="DS2: crawl address histories for TxGPT (resumable)")
    corpus.add_argument("--target", type=int, help="addresses to keep (default: settings.json -> corpus)")
    corpus.set_defaults(func=cmd_corpus)

    steps.add_parser("transactions", help="DS1: clean transaction table and cleaning log").set_defaults(
        func=cmd_transactions)
    features = steps.add_parser("features", help="DS1: F1-F8, z-scores, scorable flag, baseline flag")
    features.add_argument("--as-written", action="store_true",
                          help="the A1 method exactly as written, for the refinement ablation")
    features.set_defaults(func=cmd_features)
    steps.add_parser("split", help="DS1: fit and evaluation periods per address").set_defaults(func=cmd_split)

    steps.add_parser("process-external", help="DS4: labels, transforms and split schemes").set_defaults(
        func=cmd_process_external)
    classify = steps.add_parser("classify", help="DS3/DS4: search, threshold, refit, test once")
    classify.add_argument("--dataset", choices=["heist", "elliptic"], required=True)
    classify.add_argument("--splits", default="temporal,random,lofo", help="BitcoinHeist schemes, comma-separated")
    classify.add_argument("--force", action="store_true", help="redo results that are already saved")
    classify.set_defaults(func=cmd_classify)

    external = steps.add_parser("external", help="DS3/DS4: download, unpack, record checksums and licences")
    external.add_argument("--elliptic", action="store_true", help="DS3, from Kaggle (needs an API token)")
    external.add_argument("--bitcoinheist", action="store_true", help="DS4, from UCI")
    external.set_defaults(func=cmd_external)
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
