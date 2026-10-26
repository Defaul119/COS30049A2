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
    python -m explorer.cli inject                 # synthetic plants and their recomputed features
    python -m explorer.cli baseline --sweep       # robust-z threshold sweep
    python -m explorer.cli outliers               # Isolation Forest and LOF
    python -m explorer.cli clustering             # K-Means and DBSCAN
    python -m explorer.cli profiles               # DS1 + DS2 address profiles and DS2's coverage of DS1
    python -m explorer.cli regression             # naive, Ridge, random forest, gradient boosting, quantiles
    python -m explorer.cli txgpt                  # after lm.tx_score: choose raw or relative surprisal
    python -m explorer.cli fusion                 # two complementary methods fused at the same flag budget
    python -m explorer.cli evaluate               # every DS1 check and the recommended method
    python -m explorer.cli significance           # bootstrap intervals: which recall differences are real
    python -m explorer.cli sensitivity            # recall against plant strength (spike size, burst size, dormancy)
    python -m explorer.cli ds2-check              # the frozen DS1 methods on DS2's ~2,900 comparable histories
    python -m explorer.cli walk-forward           # every method at several fit/evaluation cut-offs
    python -m explorer.cli ds10                   # DS10: refetch a teammate's 27-address snapshot, cross-check it
    python -m explorer.cli ds10-check             # every DS1 model (TxGPT too) frozen on DS10
    python -m explorer.cli predict address bc1q...  # or: predict what-if / explain / heist / elliptic
    python -m explorer.cli predict explain --address bc1q... --txid <txid> --assistant   # guarded Assistant text
    python -m explorer.cli all --offline          # every step above, in order, from the data on disk
    python -m explorer.cli scores                 # scores.csv: every method, every transaction
    python -m explorer.cli process-external       # DS4 labels, transforms, splits
    python -m explorer.cli classify --dataset heist [--splits temporal,random,lofo]
    python -m explorer.cli figures                # every report figure (PDF) and reports/tables/eda_tests.csv
    python -m explorer.cli build-ds7              # DS7 explanation pairs, split by source address
    python -m explorer.cli build-ds7 --out data/processed/ds7_v2 --settings ds7v2   # round 2's larger DS7
    python -m explorer.cli build-random-test      # the Assistants' randomized test set (needs DS7 v2's test split)
    python -m explorer.cli label-check            # unsupervised scores against DS3/DS4 labels
    python -m explorer.cli reviewer-sheet         # 40 shuffled rows for the blind reviewer check
    python -m explorer.cli reviewer-score --ratings ratings_mkn.csv ratings_cdb.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Callable

from .esplora import EsploraClient, FailoverClient
from .settings import PROJECT_ROOT, SETTINGS

LOG_DIR = PROJECT_ROOT / "reports" / "logs"

log = logging.getLogger("explorer")


def setup_logging(step: str) -> None:
    """Log INFO lines to the console and append them to reports/logs/<step>.log.

    `predict` logs to stderr, so its stdout is only the JSON result and can be piped.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    console = sys.stderr if step == "predict" else sys.stdout
    for handler in (logging.StreamHandler(console), logging.FileHandler(LOG_DIR / f"{step}.log", encoding="utf-8")):
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
    # Requests go to the primary and fail over to the fallback when rate-limited; the cross-check
    # always uses the fallback directly, on histories it did not serve.
    client = FailoverClient([primary, fallback], SETTINGS["collection"]["failoverHoldSeconds"])
    return run_snapshot(client, fallback, SETTINGS["collection"], refetch=args.refetch, rule_only=args.rule_only)


def cmd_corpus(args: argparse.Namespace) -> int:
    from .corpus import ClientPool, make_clients, run_corpus

    params = dict(SETTINGS["corpus"])
    if args.target:
        params["targetAddresses"] = args.target
    clients = ClientPool(make_clients([SETTINGS["source"][key] for key in params["sources"]], params["workersPerSource"]))
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


def cmd_inject(args: argparse.Namespace) -> int:
    from .inject import run_inject

    return run_inject()


def cmd_baseline(args: argparse.Namespace) -> int:
    from .baseline import run_baseline_sweep

    return run_baseline_sweep()


def cmd_outliers(args: argparse.Namespace) -> int:
    from .outliers import run_outliers

    return run_outliers()


def cmd_clustering(args: argparse.Namespace) -> int:
    from .clustering import run_clustering

    return run_clustering()


def cmd_profiles(args: argparse.Namespace) -> int:
    from .profiles import run_profiles

    return run_profiles()


def cmd_regression(args: argparse.Namespace) -> int:
    from .regression import run_regression

    return run_regression()


def cmd_fusion(args: argparse.Namespace) -> int:
    from .fusion import run_fusion

    return run_fusion()


def cmd_txgpt(args: argparse.Namespace) -> int:
    from .txgpt import run_txgpt

    return run_txgpt()


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import run_evaluate

    return run_evaluate()


def cmd_walkforward(args: argparse.Namespace) -> int:
    from .walkforward import run_walkforward

    return run_walkforward()


def cmd_ds2_check(args: argparse.Namespace) -> int:
    from .ds2_check import run_ds2_check

    return run_ds2_check()


def cmd_ds10(args: argparse.Namespace) -> int:
    from .ds10 import run_ds10

    primary, fallback = esplora_clients()
    return run_ds10(FailoverClient([primary, fallback], SETTINGS["collection"]["failoverHoldSeconds"]))


def cmd_ds10_check(args: argparse.Namespace) -> int:
    from .ds10 import run_ds10_check

    return run_ds10_check()


def cmd_sensitivity(args: argparse.Namespace) -> int:
    from .sensitivity import run_sensitivity

    return run_sensitivity()


def cmd_significance(args: argparse.Namespace) -> int:
    from .significance import run_significance

    return run_significance()


def cmd_predict(args: argparse.Namespace) -> int:
    import json

    import pandas as pd

    from .predict import InsufficientHistory, Predictor, Unsupported, check_outputs

    predictor = Predictor.load()
    try:
        if args.kind == "address":
            result = predictor.score_address(args.address)
        elif args.kind == "what-if":
            result = predictor.score_what_if(args.address, args.amount, args.direction, args.time)
            if args.assistant:
                guarded = predictor.explain(result, use_assistant=True)
                result["explanation"], result["explanationSource"] = guarded["explanation"], guarded["source"]
        elif args.kind == "explain":
            result = predictor.explain(predictor.transaction(args.address, args.txid), use_assistant=args.assistant)
        elif args.kind == "heist":
            result = predictor.classify_heist(args.length, args.weight, args.count, args.looped, args.neighbors, args.income)
        else:
            result = predictor.classify_elliptic(pd.read_csv(args.rows))
    except Unsupported as err:
        log.info("Not in the snapshot: %s", err)
        return 2
    except InsufficientHistory as err:
        print(json.dumps(err.body, indent=2))
        return 3
    check_outputs(result)
    print(json.dumps(result, indent=2))
    return 0


def cmd_scores(args: argparse.Namespace) -> int:
    from .scores import run_scores

    return run_scores()


def cmd_process_external(args: argparse.Namespace) -> int:
    from .labelled import run_process_elliptic, run_process_heist

    return run_process_heist() or run_process_elliptic()


def cmd_classify(args: argparse.Namespace) -> int:
    from .classify import run_classify_heist

    if args.dataset == "heist":
        return run_classify_heist(args.splits.split(","), force=args.force)
    from .classify import run_classify_elliptic

    return run_classify_elliptic(force=args.force)


def cmd_label_check(args: argparse.Namespace) -> int:
    from .label_check import run_label_check

    return run_label_check()


def cmd_figures(args: argparse.Namespace) -> int:
    from .plots import run_figures

    return run_figures()


def cmd_build_ds7(args: argparse.Namespace) -> int:
    from .ds7 import OUT_DIR, run_build_ds7

    return run_build_ds7(Path(args.out).resolve() if args.out else OUT_DIR, settings_key=args.settings)


def cmd_build_random_test(args: argparse.Namespace) -> int:
    from .random_test import DS7_V2_TEST, run_build_random_test

    return run_build_random_test(Path(args.ds7_v2_test).resolve() if args.ds7_v2_test else DS7_V2_TEST)


def cmd_reviewer_sheet(args: argparse.Namespace) -> int:
    from .reviewer import run_reviewer_sheet

    return run_reviewer_sheet()


def cmd_reviewer_score(args: argparse.Namespace) -> int:
    from .reviewer import run_reviewer_score

    return run_reviewer_score([Path(p) for p in args.ratings])


def offline_steps() -> list[tuple[str, Callable[[], int]]]:
    """Every step that rebuilds results from data already on disk, in run order (plan Appendix D).

    Steps whose inputs are absent are skipped with a message: TxGPT needs lm.tx_score's output, and
    DS3/DS4 need their downloads (python -m explorer.cli external).
    """
    from .baseline import run_baseline_sweep
    from .classify import run_classify_elliptic, run_classify_heist
    from .clustering import run_clustering
    from .corpus import OUT_DIR as CORPUS_DIR
    from .ds2_check import run_ds2_check
    from .ds10 import OUT_DIR as DS10_DIR, run_ds10_check
    from .evaluate import run_evaluate
    from .external import EXTERNAL_DIR
    from .features import run_features
    from .fusion import run_fusion
    from .inject import run_inject
    from .label_check import run_label_check
    from .labelled import run_process_elliptic, run_process_heist
    from .outliers import run_outliers
    from .profiles import run_profiles
    from .regression import run_regression
    from .scores import run_scores
    from .sensitivity import run_sensitivity
    from .significance import run_significance
    from .split import run_split
    from .transactions import run_transactions
    from .txgpt import SCORES_PATH as TXGPT_SCORES, plant_scores_exist, run_txgpt
    from .walkforward import run_walkforward

    steps: list[tuple[str, Callable[[], int]]] = [
        ("transactions", run_transactions),
        ("features", run_features),
        ("features --as-written", lambda: run_features(as_written=True)),
        ("split", run_split),
        ("inject", run_inject),
        ("baseline", run_baseline_sweep),
        ("outliers", run_outliers),
        ("clustering", run_clustering),
        ("regression", run_regression),
    ]
    if (CORPUS_DIR / "ds2.parquet").exists():
        steps.append(("profiles", run_profiles))
    else:
        log.info("Skipping profiles: no DS2 yet (python -m explorer.cli corpus)")
    if TXGPT_SCORES.exists() and plant_scores_exist():
        steps.append(("txgpt", run_txgpt))
    else:
        # A fresh copy has the saved method (models/txgpt_method.joblib, with every score it needs) but not
        # the planted copies' scores, which only lm.tx_score --plants makes; later steps use the saved method.
        log.info("Skipping txgpt: keeping models/txgpt_method.joblib (re-tuning needs python -m lm.tx_score --plants)")
    steps += [("fusion", run_fusion), ("scores", run_scores), ("evaluate", run_evaluate),
              ("significance", run_significance), ("sensitivity", run_sensitivity), ("walk-forward", run_walkforward)]
    if (CORPUS_DIR / "ds2.parquet").exists():
        steps.append(("ds2-check", run_ds2_check))
    if (DS10_DIR / "transactions.csv").exists():
        steps.append(("ds10-check", run_ds10_check))
    if (EXTERNAL_DIR / "bitcoinheist" / "BitcoinHeistData.csv").exists():
        steps += [("process-external (DS4)", run_process_heist),
                  ("classify --dataset heist", lambda: run_classify_heist(["temporal", "random", "lofo"]))]
    else:
        log.info("Skipping DS4: not downloaded (python -m explorer.cli external --bitcoinheist)")
    if (EXTERNAL_DIR / "elliptic" / "elliptic_txs_features.csv").exists():
        steps += [("process-external (DS3)", run_process_elliptic),
                  ("classify --dataset elliptic", run_classify_elliptic)]
    else:
        log.info("Skipping DS3: not downloaded (python -m explorer.cli external --elliptic)")
    steps.append(("label-check", run_label_check))  # after outliers (it reuses their DS1 configurations)
    return steps


def cmd_all(args: argparse.Namespace) -> int:
    for name, step in offline_steps():
        started = time.monotonic()
        log.info("== %s", name)
        code = step()
        log.info("== %s finished in %.0f s (exit %d)", name, time.monotonic() - started, code)
        if code:
            log.info("Stopping: %s failed, and later steps depend on it", name)
            return code
    return 0


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
    steps.add_parser("inject", help="DS1: tuning and reporting plants, and their copies' features").set_defaults(
        func=cmd_inject)
    baseline = steps.add_parser("baseline", help="DS1: robust-z baseline threshold sweep")
    baseline.add_argument("--sweep", action="store_true", help="accepted for the plan's command; the sweep always runs")
    baseline.set_defaults(func=cmd_baseline)
    steps.add_parser("outliers", help="DS1: Isolation Forest and LOF searches").set_defaults(func=cmd_outliers)
    steps.add_parser("clustering", help="DS1: K-Means and DBSCAN").set_defaults(func=cmd_clustering)
    steps.add_parser("profiles", help="DS1 + DS2: address profiles (K-Means) and DS2's coverage of DS1").set_defaults(
        func=cmd_profiles)
    steps.add_parser("regression", help="DS1: expected size and gap, residual and range flags").set_defaults(
        func=cmd_regression)
    steps.add_parser("txgpt", help="DS1: TxGPT raw vs relative surprisal, threshold").set_defaults(func=cmd_txgpt)
    steps.add_parser("fusion", help="DS1: two complementary methods fused at the explorerMethod's flag budget").set_defaults(
        func=cmd_fusion)
    steps.add_parser("evaluate", help="DS1: injection, stability, agreement, sanity, ablation, selection").set_defaults(
        func=cmd_evaluate)
    steps.add_parser("significance", help="DS1: address-level bootstrap intervals for recall and method differences"
                     ).set_defaults(func=cmd_significance)
    steps.add_parser("sensitivity", help="DS1: recall of every method against plant strength").set_defaults(
        func=cmd_sensitivity)
    steps.add_parser("ds2-check", help="DS2: the frozen DS1 methods on ~2,900 more address histories").set_defaults(
        func=cmd_ds2_check)
    steps.add_parser("walk-forward", help="DS1: every method at several fit/evaluation cut-offs").set_defaults(
        func=cmd_walkforward)
    steps.add_parser("ds10", help="DS10: refetch a teammate's snapshot, cross-check it with their CSV").set_defaults(
        func=cmd_ds10)
    steps.add_parser("ds10-check", help="DS10: every DS1 model, TxGPT included, on the teammate's snapshot").set_defaults(
        func=cmd_ds10_check)
    predict = steps.add_parser("predict", help="use the trained models: an address, a what-if, DS4 or DS3 rows")
    kinds = predict.add_subparsers(dest="kind", required=True)
    address = kinds.add_parser("address", help="timeline and every transaction's scores for a snapshot address")
    address.add_argument("address")
    what_if = kinds.add_parser("what-if", help="score a hypothetical next transaction for a snapshot address")
    what_if.add_argument("--address", required=True)
    what_if.add_argument("--amount", type=float, required=True, help="BTC")
    what_if.add_argument("--direction", choices=["in", "out"], required=True)
    what_if.add_argument("--time", required=True, help="ISO 8601 UTC, e.g. 2026-10-01T12:00Z")
    explain = kinds.add_parser("explain", help="one snapshot transaction's explanation: the template, or the guarded Assistant")
    explain.add_argument("--address", required=True)
    explain.add_argument("--txid", required=True)
    for kind in (what_if, explain):
        kind.add_argument("--assistant", action="store_true",
                          help="let the Assistant write it (settings -> assistant); the template if every answer fails a check")
    heist = kinds.add_parser("heist", help="BitcoinHeist: does an address-day resemble the labelled ransomware rows?")
    for name in ("length", "weight", "count", "looped", "neighbors", "income"):
        heist.add_argument(f"--{name}", type=float, required=True)
    elliptic = kinds.add_parser("elliptic", help="Elliptic: class probabilities for feature rows (CSV)")
    elliptic.add_argument("--rows", required=True)
    predict.set_defaults(func=cmd_predict)
    steps.add_parser("scores", help="DS1: every saved method's score and flag per transaction").set_defaults(
        func=cmd_scores)

    steps.add_parser("process-external", help="DS3/DS4: joins, labels, transforms and split schemes").set_defaults(
        func=cmd_process_external)
    classify = steps.add_parser("classify", help="DS3/DS4: search, threshold, refit, test once")
    classify.add_argument("--dataset", choices=["heist", "elliptic"], required=True)
    classify.add_argument("--splits", default="temporal,random,lofo", help="BitcoinHeist schemes, comma-separated")
    classify.add_argument("--force", action="store_true", help="redo results that are already saved")
    classify.set_defaults(func=cmd_classify)
    steps.add_parser("figures", help="every figure whose inputs exist, as PDFs, plus eda_tests.csv").set_defaults(
        func=cmd_figures)
    build_ds7 = steps.add_parser("build-ds7", help="DS7: explanation pairs for the Assistants (rebuild after DS2)")
    build_ds7.add_argument("--out", help="output folder (default data/processed/ds7, round 1's frozen set)")
    build_ds7.add_argument("--settings", default="ds7", help="settings.json section: ds7 (round 1) or ds7v2 (round 2)")
    build_ds7.set_defaults(func=cmd_build_ds7)
    random_test = steps.add_parser("build-random-test",
                                   help="the Assistants' randomized test set: real held-out and synthetic prompts")
    random_test.add_argument("--ds7-v2-test", help="DS7 v2's test.jsonl (default data/processed/ds7_v2/test.jsonl)")
    random_test.set_defaults(func=cmd_build_random_test)
    steps.add_parser("reviewer-sheet", help="DS1: the blind sheet for the reviewer check, and its key").set_defaults(
        func=cmd_reviewer_sheet)
    reviewer_score = steps.add_parser("reviewer-score", help="DS1: reviewers' ratings against the flags; Cohen's kappa")
    reviewer_score.add_argument("--ratings", nargs="+", required=True, help="one returned sheet per reviewer")
    reviewer_score.set_defaults(func=cmd_reviewer_score)
    steps.add_parser("label-check", help="DS3/DS4: unsupervised methods' scores against the labels").set_defaults(
        func=cmd_label_check)

    everything = steps.add_parser("all", help="rebuild every result from the data on disk, in run order")
    everything.add_argument("--offline", action="store_true", help="the plan's spelling; every step here is offline")
    everything.set_defaults(func=cmd_all)

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
