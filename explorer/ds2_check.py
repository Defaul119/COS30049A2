"""Do the DS1 results hold beyond 25 hand-picked addresses? The frozen DS1 methods on DS2's address histories.

Serves plan §11 (evaluation) and the HD criterion "additional datasets". DS2, our Esplora crawl of about
4,000 address histories (hashed ids; DS1's candidate addresses excluded), so far trains TxGPT and
describes address profiles. Here it is a second, far larger evaluation population for the DS1 methods,
none of which ever saw it:

    addresses  every DS2 history that meets DS1's own main-address rule (settings.json -> collection:
               mainMinTx transactions over mainMinSpanDays), so the two populations are comparable
    processing DS1's cleaning output columns (direction from received vs sent, as amounts.address_flow),
               the same features (features.build_features), time split (split.assign_periods) and
               plants (inject.make_plants, the reporting seeds, ds2Check.plantsPerPattern per pattern)
    methods    every saved DS1 method, FROZEN: models, thresholds and fusion multipliers exactly as tuned
               on DS1. The one per-address quantity, the regression residual scale, comes from each DS2
               address's own fit period by the same rule as on DS1, because "how far this address's amounts
               usually miss the prediction" is part of "unusual for this address", not something to carry
               over from other addresses.
    metrics    recall per pattern with address-level bootstrap intervals (significance.bootstrap_recalls),
               and the flag rate and sanity rule on DS2's real evaluation rows

TxGPT is left out: it was trained on DS2, and its scores need their own GPU pass. eval_ds2.csv puts each
method's DS2 recall next to its DS1 recall (eval_injection.csv), with both rankings, so the report can say
whether the method ranking found on 25 addresses carries over.

Reads data/processed/corpus/ds2.parquet; writes reports/tables/eval_ds2.csv and eval_ds2_plant_hits.csv.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .baseline import TABLES_DIR
from .corpus import OUT_DIR as CORPUS_DIR
from .features import FeatureConfig, build_features
from .fusion import FusionMethod
from .inject import Plant, copy_features, make_plants
from .methods import Method, fit_rows, real_metrics
from .predict import PRECOMPUTED_ONLY
from .regression import RegressionMethod, residual_scales, targets
from .scores import saved_methods
from .settings import SETTINGS
from .significance import MEAN_OF, bootstrap_recalls, interval, plant_hits
from .split import assign_periods

log = logging.getLogger(__name__)

DAY = 86_400


def ds2_transactions(ds2: pd.DataFrame) -> pd.DataFrame:
    """DS2's compact rows in DS1's transaction-table columns, for the addresses that meet DS1's main rule."""
    c = SETTINGS["collection"]
    g = ds2.groupby("addr_id")["block_time_utc"].agg(["size", "min", "max"])
    keep = g.index[(g["size"] >= c["mainMinTx"]) & ((g["max"] - g["min"]) / DAY >= c["mainMinSpanDays"])]
    tx = ds2[ds2["addr_id"].isin(keep)].rename(columns={"addr_id": "address", "txid_hash": "txid"})
    net = tx["received_sats"] - tx["sent_sats"]
    tx = tx.assign(role="main", net_sats=net,
                   direction=np.where(net > 0, "in", np.where(net < 0, "out", "none")))
    return tx.sort_values(["address", "block_time_utc", "txid"], kind="stable").reset_index(drop=True)


def features_in_parallel(tx: pd.DataFrame, cfg: FeatureConfig, n_jobs: int) -> pd.DataFrame:
    """build_features one address at a time across workers (it is pure Python per transaction)."""
    parts = Parallel(n_jobs=n_jobs)(delayed(build_features)(group, cfg) for _, group in tx.groupby("address", sort=True))
    return pd.concat(parts, ignore_index=True)


def copy_one(rows: pd.DataFrame, periods: pd.DataFrame, plant: Plant, cfg: FeatureConfig) -> pd.DataFrame:
    """copy_features on one address's own rows, so a worker never receives all of DS2."""
    copy = copy_features(rows, periods, plant, cfg)
    return copy[copy["is_synthetic"] == 1]


def adapt(method: Method, fit: pd.DataFrame) -> Method:
    """The frozen method, with regression residual scales set from each new address's own fit period."""
    if isinstance(method, RegressionMethod):
        floor = SETTINGS["regression"]["residualFloor"]
        scales = {}
        for t, model in method.models.items():
            y = targets(fit)[t]
            keep = y.notna()
            scales[t], _ = residual_scales(model, fit[keep], y[keep], floor)
        return RegressionMethod(method.models, scales, method.default_scale)
    if isinstance(method, FusionMethod):
        fused = FusionMethod({n: adapt(m, fit) for n, m in method.components.items()}, method.multipliers)
        fused.threshold = method.threshold  # 1 at M0; a stage C9 winner's is its flag budget's cut
        return fused
    return method


def run_ds2_check(n_jobs: int = SETTINGS["classification"]["nJobs"]) -> int:
    """Step `ds2-check`: the frozen DS1 methods on the DS2 histories that meet DS1's selection rule, with plants;
    writes eval_ds2.csv, eval_ds2_plant_hits.csv and split_ds2.csv."""
    d = SETTINGS["ds2Check"]
    tx = ds2_transactions(pd.read_parquet(CORPUS_DIR / "ds2.parquet"))
    cfg = FeatureConfig.refined()
    features = features_in_parallel(tx, cfg, n_jobs)
    period, split = assign_periods(features, SETTINGS["model"]["fitFraction"])
    features["period"] = period
    log.info("DS2: %d addresses meet DS1's main rule; %d transactions; periods %s", tx["address"].nunique(), len(tx),
             period.value_counts().to_dict())

    fit = fit_rows(features)
    methods = {n: adapt(m, fit) for n, m in saved_methods().items() if n not in PRECOMPUTED_ONLY}
    periods = features[["address", "txid", "period"]]
    by_address = dict(tuple(tx.groupby("address", sort=False)))
    periods_by_address = dict(tuple(periods.groupby("address", sort=False)))
    p: dict[str, Any] = {**SETTINGS["injection"], "plantsPerPattern": d["plantsPerPattern"]}
    copies = []
    for seed in SETTINGS["injection"]["reportingSeeds"]:
        plants = make_plants(tx, periods, seed, "ds2-", p)
        copies += Parallel(n_jobs=n_jobs)(delayed(copy_one)(by_address[pl.address], periods_by_address[pl.address], pl, cfg)
                                          for pl in plants)
        log.info("seed %d: %d plants on %d addresses", seed, len(plants), len({pl.address for pl in plants}))
    planted = pd.concat(copies, ignore_index=True)

    hits = plant_hits(methods, planted)
    addresses = sorted(hits["address"].unique())
    rng = np.random.default_rng(SETTINGS["significance"]["seed"])
    weights = rng.multinomial(len(addresses), np.full(len(addresses), 1 / len(addresses)),
                              size=SETTINGS["significance"]["bootstrapResamples"])
    boot = bootstrap_recalls(hits, weights, addresses)
    level = SETTINGS["significance"]["ciLevel"]
    ds1 = pd.read_csv(TABLES_DIR / "eval_injection.csv").set_index("method")
    rows = []
    for name, measures in boot.items():
        row: dict[str, Any] = {"method": name}
        for measure, values in measures.items():
            row[f"ds2_{measure}"] = float(values[0])
            row[f"ds2_{measure}_ci_low"], row[f"ds2_{measure}_ci_high"] = interval(values[1:], level)
        real = real_metrics(methods[name], features)
        row.update({f"ds2_{k}": v for k, v in real.items()})
        row["ds2_share_addresses_over_limit"] = real["addresses_over_limit"] / features["address"].nunique()
        if name in ds1.index:
            row["ds1_mean_recall"] = float(ds1.loc[name, "mean_recall"])
            for m in MEAN_OF:
                row[f"ds1_{m}"] = float(ds1.loc[name, f"{m}_mean"])
        rows.append(row)
    table = pd.DataFrame(rows)
    table["ds2_rank"] = table["ds2_mean_recall"].rank(ascending=False, method="min").astype(int)
    if "ds1_mean_recall" in table:
        table["ds1_rank"] = table["ds1_mean_recall"].rank(ascending=False, method="min")
    table = table.sort_values("ds2_mean_recall", ascending=False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "eval_ds2.csv", index=False)
    hits.to_csv(TABLES_DIR / "eval_ds2_plant_hits.csv", index=False)
    split.to_csv(TABLES_DIR / "split_ds2.csv", index=False)
    for r in table.itertuples():
        log.info("%-20s DS2 mean recall %.3f [%.3f, %.3f] (DS1 %s); DS2 flag rate %.3f, %.0f%% of addresses over the "
                 "sanity limit", r.method, r.ds2_mean_recall, r.ds2_mean_recall_ci_low, r.ds2_mean_recall_ci_high,
                 f"{r.ds1_mean_recall:.3f}" if "ds1_mean_recall" in table and not np.isnan(r.ds1_mean_recall) else "-",
                 r.ds2_flag_rate, 100 * r.ds2_share_addresses_over_limit)
    log.info("Wrote %s, eval_ds2_plant_hits.csv and split_ds2.csv", TABLES_DIR / "eval_ds2.csv")
    return 0
