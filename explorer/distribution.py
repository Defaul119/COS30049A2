"""Distributions of the scoring methods' planted-pattern recall over many plant seeds (DS1, DS2 and DS10).

Serves the statistics campaign (user, 27 Oct 2026: "a ridiculously reasonable amount of testing ... distribution
math ... for comparisons and mean accuracy"). The reported results plant the three patterns with the 5 reporting
seeds and bootstrap over addresses (explorer/significance.py). Here the same patterns are planted with many more
seeds (settings distribution -> seeds: a block disjoint from the tuning and reporting seeds, so the reported numbers
are unchanged), and every seed's recall is recorded for every frozen method. Each method's recall then has a
distribution: its mean, spread and 95% interval, how often it beats the baseline on the same plants (a paired
comparison), and the distribution of its rank.

The methods are frozen exactly as the DS2 and DS10 checks use them: DS1's saved models on DS1, and on DS2 and DS10
the same models with their thresholds refitted on that data's fit period (ds2_check.adapt). TxGPT is left out:
scoring planted copies needs its model on a GPU, and its 5-seed results stand.

Writes reports/tables/stats_methods_seeds.csv (one row per data set, seed and method) and
stats_methods_summary.csv (one row per data set and method).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .baseline import TABLES_DIR
from .ds2_check import adapt, copy_one, ds2_transactions, features_in_parallel
from .features import FeatureConfig
from .inject import make_plants
from .methods import fit_rows
from .predict import PRECOMPUTED_ONLY
from .scores import saved_methods
from .settings import SETTINGS
from .significance import MEAN_OF, MEASURES, plant_hits
from .split import assign_periods

log = logging.getLogger(__name__)


def seed_recalls(hits: pd.DataFrame) -> pd.DataFrame:
    """Each method's recall per measure, pooled over one seed's plants (as plant_metrics pools a full set)."""
    rows = []
    for (method, seed), part in hits.groupby(["method", "seed"]):
        row: dict[str, Any] = {"method": method, "seed": int(seed)}
        for measure, (pattern, column) in MEASURES.items():
            values = part.loc[part["pattern"] == pattern, column]
            row[measure] = float(values.mean()) if len(values) else np.nan
        row["mean_recall"] = float(np.mean([row[m] for m in MEAN_OF]))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(seeds: pd.DataFrame) -> pd.DataFrame:
    """Per data set and method: the distribution of mean recall over seeds, and paired comparisons with the baseline."""
    level = SETTINGS["significance"]["ciLevel"]
    tail = 100 * (1 - level) / 2
    seeds = seeds.copy()
    seeds["rank"] = seeds.groupby(["dataset", "seed"])["mean_recall"].rank(ascending=False, method="min")
    base = seeds[seeds["method"] == "baseline"].set_index(["dataset", "seed"])["mean_recall"]
    seeds["minus_baseline"] = seeds["mean_recall"] - seeds.set_index(["dataset", "seed"]).index.map(base).to_numpy()
    rows = []
    for (dataset, method), g in seeds.groupby(["dataset", "method"]):
        r = g["mean_recall"]
        rows.append({"dataset": dataset, "method": method, "seeds": len(g), "mean": r.mean(), "sd": r.std(ddof=1),
                     "ci_low": np.percentile(r, tail), "median": r.median(), "ci_high": np.percentile(r, 100 - tail),
                     "minus_baseline_mean": g["minus_baseline"].mean(),
                     "beats_baseline_share": float((g["minus_baseline"] > 0).mean()),
                     "mean_rank": g["rank"].mean(), "rank1_share": float((g["rank"] == 1).mean()),
                     **{f"{m}_mean": g[m].mean() for m in MEASURES}})
    return pd.DataFrame(rows).sort_values(["dataset", "mean"], ascending=[True, False])


def datasets() -> dict[str, tuple[pd.DataFrame, bool, int, dict[str, Any]]]:
    """name -> (transactions, refit thresholds?, seed count, plant settings)."""
    from .corpus import OUT_DIR as CORPUS_DIR
    from .ds10 import OUT_DIR as DS10_DIR
    from .snapshot import MANIFEST_PATH

    d = SETTINGS["distribution"]
    inj = SETTINGS["injection"]
    return {
        "DS1": (pd.read_csv(MANIFEST_PATH.parent / "transactions.csv"), False, d["seedCount"]["DS1"], inj),
        "DS10": (pd.read_csv(DS10_DIR / "transactions.csv"), True, d["seedCount"]["DS10"], inj),
        "DS2": (ds2_transactions(pd.read_parquet(CORPUS_DIR / "ds2.parquet")), True, d["seedCount"]["DS2"],
                {**inj, "plantsPerPattern": SETTINGS["ds2Check"]["plantsPerPattern"]}),
    }


def run_distribution(only: list[str] | None = None, n_jobs: int = SETTINGS["classification"]["nJobs"]) -> int:
    """Step `distribution`: every scoring method's recall over many plant seeds on DS1, DS10 and DS2 (settings
    distribution); writes stats_methods_seeds.csv and stats_methods_summary.csv."""
    d = SETTINGS["distribution"]
    cfg = FeatureConfig.refined()
    saved = {n: m for n, m in saved_methods().items() if n not in PRECOMPUTED_ONLY}
    out = []
    for name, (tx, refit, count, p) in datasets().items():
        if only and name not in only:
            continue
        features = features_in_parallel(tx, cfg, n_jobs)
        features["period"], _ = assign_periods(features, SETTINGS["model"]["fitFraction"])
        methods = {n: adapt(m, fit_rows(features)) for n, m in saved.items()} if refit else saved
        periods = features[["address", "txid", "period"]]
        by_address = dict(tuple(tx.groupby("address", sort=False)))
        periods_by_address = dict(tuple(periods.groupby("address", sort=False)))
        for seed in range(d["firstSeed"], d["firstSeed"] + count):
            plants = make_plants(tx, periods, seed, f"{name.lower()}-", p)
            copies = Parallel(n_jobs=n_jobs)(delayed(copy_one)(by_address[pl.address], periods_by_address[pl.address],
                                                               pl, cfg) for pl in plants)
            out.append(seed_recalls(plant_hits(methods, pd.concat(copies, ignore_index=True))).assign(dataset=name))
            log.info("%s seed %d: %d plants", name, seed, len(plants))
    seeds = pd.concat(out, ignore_index=True)
    path = TABLES_DIR / "stats_methods_seeds.csv"
    if only and path.exists():  # a partial run keeps the other data sets' rows
        seeds = pd.concat([pd.read_csv(path).query("dataset not in @only"), seeds], ignore_index=True)
    seeds.to_csv(path, index=False)
    summary = summarise(seeds)
    summary.to_csv(TABLES_DIR / "stats_methods_summary.csv", index=False)
    for r in summary.itertuples():
        log.info("%-5s %-20s mean recall %.3f (sd %.3f, %d seeds) [%.3f, %.3f]; beats the baseline on %.0f%% of seeds; "
                 "mean rank %.1f", r.dataset, r.method, r.mean, r.sd, r.seeds, r.ci_low, r.ci_high,
                 100 * r.beats_baseline_share, r.mean_rank)
    log.info("Wrote %s and stats_methods_summary.csv", path)
    return 0
