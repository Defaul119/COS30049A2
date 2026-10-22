"""The time split: fit on the earlier part of each address's history, evaluate on the rest.

Serves plan §6.3 (docs/plan/sections/s06-processing.tex) and DR-02. For each address, take the
scorable transactions in time order; the cut-off time t* is the time of the transaction at the
`fitFraction` point (settings.json -> model). Scorable transactions before t* form the fit period
and those at or after t* the evaluation period. Comparing times, not row positions, means
same-block transactions never straddle the cut. Unscorable transactions are always the earliest
ones (scorability only grows with history), so they serve only as history.

Every model is fitted on fit-period rows pooled across addresses, and every threshold is set from
fit-period scores and applied unchanged to the evaluation period.

Reads and rewrites the `period` column of data/processed/esplora/features.csv; writes
reports/tables/split_ds1.csv (per-address counts and cut-off times).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .settings import PROJECT_ROOT, SETTINGS
from .snapshot import MANIFEST_PATH, utc_iso

log = logging.getLogger(__name__)

FEATURES_PATH = MANIFEST_PATH.parent / "features.csv"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
HISTORY, FIT, EVAL = "history", "fit", "eval"


def cut_off(times: list[int], fit_fraction: float) -> int | None:
    """t*: the time of the scorable transaction at the fit_fraction point, or None if there are none.

    `times` are the scorable transactions' times in order. With n of them, t* is times[floor(f * n)],
    so roughly the first f * n go to fit; ties at t* all go to evaluation.
    """
    if not times:
        return None
    return times[min(int(fit_fraction * len(times)), len(times) - 1)]


def assign_periods(features: pd.DataFrame, fit_fraction: float) -> tuple[pd.Series, pd.DataFrame]:
    """The period of every row (history, fit or eval) and a per-address summary."""
    period = pd.Series(HISTORY, index=features.index, dtype="object")
    summary = []
    for address, group in features.groupby("address", sort=True):
        scorable = group[group["scorable"]].sort_values(["block_time_utc", "txid"])
        t_star = cut_off(scorable["block_time_utc"].tolist(), fit_fraction)
        if t_star is not None:
            period.loc[scorable.index[scorable["block_time_utc"] < t_star]] = FIT
            period.loc[scorable.index[scorable["block_time_utc"] >= t_star]] = EVAL
        counts = period.loc[group.index].value_counts()
        summary.append({
            "address": address, "role": group["role"].iloc[0], "transactions": len(group),
            "history": int(counts.get(HISTORY, 0)), "fit": int(counts.get(FIT, 0)), "eval": int(counts.get(EVAL, 0)),
            "cut_off_utc": utc_iso(int(t_star)) if t_star is not None else None,
        })
    return period, pd.DataFrame(summary)


def run_split(features_path: Path = FEATURES_PATH, tables_dir: Path = TABLES_DIR) -> int:
    """Add the period column to features.csv, and to features_as_written.csv if it exists.

    Both methods share the scorability rule, so they get identical periods; the split is computed
    from the refined file and copied, which keeps the ablation's fit and evaluation sets identical.
    """
    features = pd.read_csv(features_path)
    period, summary = assign_periods(features, SETTINGS["model"]["fitFraction"])
    features["period"] = period
    features.to_csv(features_path, index=False)
    as_written_path = features_path.with_name("features_as_written.csv")
    if as_written_path.exists():
        other = pd.read_csv(as_written_path)
        if not other[["address", "txid", "scorable"]].equals(features[["address", "txid", "scorable"]]):
            raise ValueError("features_as_written.csv does not match features.csv row for row; rebuild both")
        other["period"] = period
        other.to_csv(as_written_path, index=False)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "split_ds1.csv", index=False)
    log.info("Periods: %s", period.value_counts().to_dict())
    log.info("Wrote the period column of %s%s and %s", features_path,
             " and features_as_written.csv" if as_written_path.exists() else "", tables_dir / "split_ds1.csv")
    return 0
