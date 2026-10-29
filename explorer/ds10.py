"""DS10: a teammate's independent DS1-style snapshot, rebuilt through our pipeline and used to test every model.

Serves plan §11 (evaluation) and the HD criterion "additional datasets". On 23 Oct 2026 a teammate collected
24 main and 3 short-history addresses from Blockstream Esplora by DS1's rule (50+ confirmed transactions over
90+ days; 4-9 for short histories), with their own seed and blocks, and sent one CSV row per transaction.
None of its addresses is in DS1, DS1's candidate pool or DS2 (`ds10` checks this before anything else), so no
model has seen them: not the DS1 methods, and not TxGPT, which DS2 trained.

    ds10        refetches the 27 histories with DS1's client and checks (snapshot.load_history,
                verify_history), cuts them at the teammate's last block so both describe the same
                transactions, compares every row with the teammate's CSV (block, received, sent), and
                writes data/processed/ds10/{histories.json.gz, transactions.csv, manifest.json}
    ds10-check  every DS1 model on it, frozen as in ds2_check: the saved methods (regression residual
                scales from each address's own fit period), TxGPT (scored here, with the variant and
                threshold chosen on DS1), recall on planted transactions (the reporting seeds and DS1's
                plantsPerPattern) with address-level bootstrap intervals, the flag rate and sanity rule
                on the real evaluation rows, and an explanation set for the Assistant's writers
                (data/processed/ds10/explain.jsonl; lm.eval_explain --tag ds10 scores it)

The refetch has two jobs: it checks the teammate's numbers against an independent pipeline, and it
supplies what the CSV lacks (input and output counts, fee and size), which F5-F7 and TxGPT's tokens need.
DS3/DS4's classifiers do not apply here: they read those datasets' own features (Elliptic's 166
anonymised columns, BitcoinHeist's daily graph measures), which an address history does not have.

Writes reports/tables/ds10_crosscheck.csv, eval_ds10.csv, eval_ds10_plant_hits.csv, split_ds10.csv and
ds10_activity.csv (how DS1's and DS10's addresses transact, beside the recall).
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .baseline import MODELS_DIR, TABLES_DIR
from .corpus import OUT_DIR as CORPUS_DIR, SALT_PATH, ds1_exclusions, salted_id
from .ds2_check import adapt, features_in_parallel
from .esplora import EsploraEndpoints
from .features import FeatureConfig
from .inject import copy_features, make_plants
from .methods import fit_rows, real_metrics
from .predict import PRECOMPUTED_ONLY
from .scores import saved_methods
from .settings import PROJECT_ROOT, SETTINGS
from .significance import MEAN_OF, bootstrap_recalls, interval, plant_hits
from .snapshot import MANIFEST_PATH, load_history, utc_now, verify_history, write_gz
from .split import assign_periods
from .transactions import build_transactions

log = logging.getLogger(__name__)

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "ds10"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ds10"
TEAMMATE_DIR = OUT_DIR / "teammate"
SATS_PER_BTC = 100_000_000
DAY = 86_400


def teammate_tables(folder: Path = TEAMMATE_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """The teammate's transactions (amounts converted to integer satoshis), address summary and manifest."""
    tx = pd.read_csv(folder / "transactions.csv")
    for col in ("received", "sent", "net"):
        # The CSV gives BTC with 8 decimals; rounding recovers the exact satoshi integer.
        tx[f"{col}_sats"] = (tx[f"{col}_btc"] * SATS_PER_BTC).round().astype("int64")
    summary = pd.read_csv(folder / "address_summary.csv")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return tx, summary, manifest


def roles_of(summary: pd.DataFrame) -> dict[str, str]:
    """DS1's roles: the teammate's low / medium / high tiers are main addresses, short_history is short."""
    return {r.address: "short" if r.group == "short_history" else "main" for r in summary.itertuples()}


def overlaps(addresses: list[str]) -> dict[str, int]:
    """How many addresses DS10 shares with DS1 (and its candidate pool) and with DS2.

    DS2 holds salted hashes, so its check needs config/salt.txt (on the build machine only); without it
    the count is -1 (not checked).
    """
    out = {"ds1_or_its_candidate_pool": len(set(addresses) & ds1_exclusions()), "ds2": -1}
    ds2_path = CORPUS_DIR / "ds2.parquet"
    if SALT_PATH.exists() and ds2_path.exists():
        ds2 = pd.read_parquet(ds2_path, columns=["addr_id"])["addr_id"]
        salt = SALT_PATH.read_text(encoding="utf-8").strip()
        hashed = {salted_id(salt, a, len(ds2.iloc[0])) for a in addresses}
        out["ds2"] = len(hashed & set(ds2.unique()))
    return out


def cut_history(txs: list[dict[str, Any]], height: int) -> list[dict[str, Any]]:
    """The transactions confirmed at or below `height`: the history as the teammate saw it."""
    return [tx for tx in txs if tx["status"].get("block_height") is not None and tx["status"]["block_height"] <= height]


def compare_rows(ours: pd.DataFrame, theirs: pd.DataFrame) -> pd.DataFrame:
    """Per address: transactions in both tables, in one only, and shared ones whose block or amounts differ."""
    key = ["address", "txid"]
    cols = ["block_height", "received_sats", "sent_sats"]
    merged = ours[key + cols].merge(theirs[key + cols], on=key, how="outer", suffixes=("_ours", "_theirs"),
                                    indicator=True)
    both = merged["_merge"] == "both"
    differs = both & np.any([merged[f"{c}_ours"] != merged[f"{c}_theirs"] for c in cols], axis=0)
    merged = merged.assign(shared=both, only_ours=merged["_merge"] == "left_only",
                           only_theirs=merged["_merge"] == "right_only", differs=differs)
    table = merged.groupby("address")[["shared", "only_ours", "only_theirs", "differs"]].sum().astype(int).reset_index()
    table["agrees"] = (table["only_ours"] == 0) & (table["only_theirs"] == 0) & (table["differs"] == 0)
    return table


def run_ds10(client: EsploraEndpoints, raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> int:
    p = SETTINGS["ds10"]
    theirs, summary, their_manifest = teammate_tables(out_dir / "teammate")
    roles = roles_of(summary)
    shared = overlaps(sorted(roles))
    log.info("DS10: %d addresses (%s); shared with DS1 or its pool: %d, with DS2: %s", len(roles),
             pd.Series(roles).value_counts().to_dict(), shared["ds1_or_its_candidate_pool"],
             shared["ds2"] if shared["ds2"] >= 0 else "not checked (no salt)")
    if shared["ds1_or_its_candidate_pool"] > 0 or shared["ds2"] > 0:
        log.error("DS10 shares addresses with data the models were built on, so it cannot test them")
        return 1

    # The teammate's last block: transactions confirmed since then are left out, so both tables describe
    # the same histories.
    cut = int(theirs["block_height"].max())
    params = {**SETTINGS["collection"], "maxTxPerAddress": p["maxTxPerAddress"]}
    histories, checks = {}, []
    for address in sorted(roles):
        history = load_history(client, address, raw_dir, params)
        checks += [{"address": address, **c} for c in verify_history(history)]
        histories[address] = {"role": roles[address], "chain_stats": history.stats["chain_stats"],
                              "txs": cut_history(history.txs, cut)}
    failed = [c for c in checks if not c["passed"]]
    for c in failed:
        log.error("check failed for %s: %s (%s)", c["address"], c["name"], c["detail"])

    tx, steps = build_transactions(histories)
    compare = compare_rows(tx, theirs)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    compare.to_csv(TABLES_DIR / "ds10_crosscheck.csv", index=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    histories_path = out_dir / "histories.json.gz"
    write_gz(histories_path, json.dumps(histories, sort_keys=True, separators=(",", ":")))
    tx.to_csv(out_dir / "transactions.csv", index=False)
    totals = compare[["shared", "only_ours", "only_theirs", "differs"]].sum().astype(int).to_dict()
    manifest = {
        "dataset": "DS10: a teammate's independent DS1-style Esplora snapshot, rebuilt through explorer/snapshot.py",
        "built_utc": utc_now(),
        "teammate": {"collected_utc": their_manifest.get("collected_utc"), "reviewed_by": their_manifest.get("reviewed_by"),
                     "selection_rule": their_manifest.get("selection_rule"), "parameters": their_manifest.get("parameters"),
                     "funnel": their_manifest.get("funnel"),
                     "files": [{"path": f"teammate/{f.name}", "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
                               for f in sorted((out_dir / "teammate").iterdir()) if f.is_file()]},
        "cut_block_height": cut,
        "addresses": pd.Series(roles).value_counts().to_dict(),
        "rows": len(tx),
        "shared_addresses": shared,
        "checks_failed": failed,
        "crosscheck_with_teammate_csv": {**totals, "addresses_agreeing": int(compare["agrees"].sum())},
        "cleaning": steps,
        "files": [{"path": histories_path.name, "sha256": hashlib.sha256(histories_path.read_bytes()).hexdigest()}],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("Cross-check with the teammate's CSV: %s; %d of %d addresses agree exactly", totals,
             compare["agrees"].sum(), len(compare))
    log.info("Wrote %s, transactions.csv (%d rows) and %s", histories_path, len(tx), TABLES_DIR / "ds10_crosscheck.csv")
    return 1 if failed else 0


# -- ds10-check ---------------------------------------------------------------------------------


def activity_profile(tx: pd.DataFrame, source: str) -> dict[str, Any]:
    """How a snapshot's main addresses transact, to explain recall that differs from DS1's.

    Per address: the median gap between distinct block times (hours), and the median and interquartile
    range of how many earlier transactions fall in the burst window; then the median over addresses.
    """
    window = SETTINGS["features"]["burstWindowDays"] * DAY
    rows = []
    for _, g in tx[tx["role"] == "main"].groupby("address"):
        t = g["block_time_utc"].to_numpy()
        earlier = np.array([((t < x) & (t >= x - window)).sum() for x in t])
        q1, q3 = np.percentile(earlier, [25, 75])
        rows.append({"median_gap_hours": float(np.median(np.diff(np.unique(t))) / 3600),
                     "median_window_count": float(np.median(earlier)), "window_count_iqr": float(q3 - q1)})
    return {"source": source, "main_addresses": len(rows), **pd.DataFrame(rows).median().round(2).to_dict()}


def txgpt_lookup(frames: list[tuple[pd.DataFrame, str]]) -> pd.DataFrame:
    """TxGPT's surprisal and relative score for every row of each (frame, group column), keyed as TxGPTMethod expects."""
    import torch

    from lm.tx_score import check_vocabulary, load_model, score_frame

    from .txgpt import with_relative

    check_vocabulary(MODELS_DIR / "txgpt_meta.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(MODELS_DIR / "txgpt.pt", device)
    parts = []
    with torch.no_grad():
        for frame, group in frames:
            scores = score_frame(model, frame, device, group)
            if group != "address":
                scores = scores.drop(columns=["address"])
            scored = with_relative(scores, frame[[group, "txid", "block_time_utc"]], group)
            parts.append(scored.rename(columns={group: "key"}).drop(columns=["block_time_utc"]))
    return pd.concat(parts, ignore_index=True)


def explanation_set(features: pd.DataFrame, tx: pd.DataFrame, methods: dict[str, Any], params: dict[str, Any]) -> list[dict]:
    """DS7-style prompts for DS10 transactions (flagged, unflagged, unscored), with the template as reference."""
    from .ds7 import pairs_from, score_frame as ds7_scores
    from .predict import Predictor
    from .reviewer import selected_method

    predictor = Predictor(features, tx, ds7_scores(features, methods), methods, selected_method(), "")
    rng = random.Random(params["seed"])
    records = list(pairs_from("DS10", predictor, predictor.scores, sorted(features["address"].unique()), params, rng))
    return [{"id": f"ds10-{i:05d}", **{k: r[k] for k in ("source", "kind", "phrasing", "template", "messages")}}
            for i, r in enumerate(records)]


def run_ds10_check(out_dir: Path = OUT_DIR, n_jobs: int = SETTINGS["classification"]["nJobs"]) -> int:
    p = SETTINGS["ds10"]
    tx = pd.read_csv(out_dir / "transactions.csv")
    cfg = FeatureConfig.refined()
    features = features_in_parallel(tx, cfg, n_jobs)
    period, split = assign_periods(features, SETTINGS["model"]["fitFraction"])
    features["period"] = period
    log.info("DS10: %d addresses, %d transactions; periods %s", tx["address"].nunique(), len(tx),
             period.value_counts().to_dict())

    fit = fit_rows(features)
    methods = {n: adapt(m, fit) for n, m in saved_methods().items() if n not in PRECOMPUTED_ONLY}
    periods = features[["address", "txid", "period"]]
    by_address = dict(tuple(tx.groupby("address", sort=False)))
    periods_by_address = dict(tuple(periods.groupby("address", sort=False)))
    copies = []
    for seed in SETTINGS["injection"]["reportingSeeds"]:
        plants = make_plants(tx, periods, seed, "ds10-")
        # Whole copies, not just the planted rows (ds2_check keeps only those): TxGPT scores a planted
        # transaction from the copy's own earlier rows.
        copies += Parallel(n_jobs=n_jobs)(delayed(copy_features)(by_address[pl.address], periods_by_address[pl.address],
                                                                 pl, cfg) for pl in plants)
        log.info("seed %d: %d plants on %d addresses", seed, len(plants), len({pl.address for pl in plants}))
    planted_full = pd.concat(copies, ignore_index=True)

    # TxGPT never saw DS10, so it is tested here too: scored now, with DS1's variant and threshold kept.
    from .scores import method_path
    from .txgpt import TxGPTMethod

    saved = joblib.load(method_path("txgpt_method"))  # its accepted stage's variant and threshold, else M0's
    lookup = txgpt_lookup([(tx, "address"), (planted_full, "plant_id")])
    txgpt = TxGPTMethod(saved["method"].variant, lookup)  # a stage winner's file has no "variant" key
    txgpt.threshold = saved["threshold"]
    methods["txgpt_method"] = txgpt
    log.info("TxGPT: %s score, DS1 threshold %.3f, %d rows scored", txgpt.variant, txgpt.threshold, len(lookup))

    planted = planted_full[planted_full["is_synthetic"] == 1]
    hits = plant_hits(methods, planted)
    addresses = sorted(hits["address"].unique())
    rng = np.random.default_rng(SETTINGS["significance"]["seed"])
    weights = rng.multinomial(len(addresses), np.full(len(addresses), 1 / len(addresses)),
                              size=SETTINGS["significance"]["bootstrapResamples"])
    boot = bootstrap_recalls(hits, weights, addresses)
    level = SETTINGS["significance"]["ciLevel"]
    ds1 = pd.read_csv(TABLES_DIR / "eval_injection.csv").set_index("method")
    ds2_path = TABLES_DIR / "eval_ds2.csv"
    ds2 = pd.read_csv(ds2_path).set_index("method") if ds2_path.exists() else pd.DataFrame()
    rows = []
    for name, measures in boot.items():
        row: dict[str, Any] = {"method": name}
        for measure, values in measures.items():
            row[f"ds10_{measure}"] = float(values[0])
            row[f"ds10_{measure}_ci_low"], row[f"ds10_{measure}_ci_high"] = interval(values[1:], level)
        real = real_metrics(methods[name], features)
        row.update({f"ds10_{k}": v for k, v in real.items()})
        row["ds10_share_addresses_over_limit"] = real["addresses_over_limit"] / features["address"].nunique()
        if name in ds1.index:
            row["ds1_mean_recall"] = float(ds1.loc[name, "mean_recall"])
            for m in MEAN_OF:
                row[f"ds1_{m}"] = float(ds1.loc[name, f"{m}_mean"])
        if name in ds2.index:
            row["ds2_mean_recall"] = float(ds2.loc[name, "ds2_mean_recall"])
        rows.append(row)
    table = pd.DataFrame(rows)
    for source in ("ds10", "ds1", "ds2"):
        if f"{source}_mean_recall" in table:
            table[f"{source}_rank"] = table[f"{source}_mean_recall"].rank(ascending=False, method="min")
    table = table.sort_values("ds10_mean_recall", ascending=False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "eval_ds10.csv", index=False)
    hits.to_csv(TABLES_DIR / "eval_ds10_plant_hits.csv", index=False)
    split.to_csv(TABLES_DIR / "split_ds10.csv", index=False)
    profile = pd.DataFrame([activity_profile(pd.read_csv(MANIFEST_PATH.parent / "transactions.csv"), "DS1"),
                            activity_profile(tx, "DS10")])
    profile.to_csv(TABLES_DIR / "ds10_activity.csv", index=False)
    log.info("Activity, DS1 vs DS10:\n%s", profile.to_string(index=False))
    for r in table.itertuples():
        log.info("%-20s DS10 mean recall %.3f [%.3f, %.3f] (DS1 %s, DS2 %s); flag rate %.3f, %.0f%% of addresses over the "
                 "sanity limit", r.method, r.ds10_mean_recall, r.ds10_mean_recall_ci_low, r.ds10_mean_recall_ci_high,
                 f"{r.ds1_mean_recall:.3f}" if not np.isnan(getattr(r, "ds1_mean_recall", np.nan)) else "-",
                 f"{r.ds2_mean_recall:.3f}" if not np.isnan(getattr(r, "ds2_mean_recall", np.nan)) else "-",
                 r.ds10_flag_rate, 100 * r.ds10_share_addresses_over_limit)

    log.info("Wrote %s, eval_ds10_plant_hits.csv and split_ds10.csv", TABLES_DIR / "eval_ds10.csv")
    explain_path = out_dir / "explain.jsonl"
    if explain_path.exists():
        # A frozen test set: both rounds of the Assistants are scored on these exact prompts and templates
        # (docs/KAGGLE_LM_PLAN.md §3), so a later template change must not rewrite them. Delete it to rebuild.
        log.info("Kept the frozen explanation test set %s", explain_path)
        return 0
    explain_methods = {n: m for n, m in methods.items() if n not in PRECOMPUTED_ONLY}
    records = explanation_set(features, tx, explain_methods, {**SETTINGS["ds7"], **p["explanations"]})
    with open(explain_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Wrote %s (%d explanation prompts: %s)", explain_path, len(records),
             pd.Series([r["kind"] for r in records]).value_counts().to_dict())
    return 0
