"""TxGPT as a DS1 method: surprisal scores, the raw/relative choice and the reason phrase.

Serves plan §10.1. lm.tx_score writes each DS1 transaction's surprisal per field (nats), and the
same for every planted copy. Two candidate scores (plan §10.1, step 4):
    raw       s_i, the total surprisal
    relative  a robust z of s_i against the same history's strictly earlier surprisals (floored
              spread), i.e. surprising *for this address* rather than surprising in general
Stage C7 (docs/TRAINING_PLAN.md) adds event-level variants, event_<raw|relative>_<W>h: the score plus the
strictly earlier ones within W hours, so a burst of mildly surprising transactions adds up.
The tuning plants (seeds 40000-40004) choose between them by mean recall; the flag threshold is
the top 2% of fit-period scores, and the baseline's minimum-history rule applies (methods.flags).
The reason shown with a flag names the field with the largest surprisal.

Reads data/processed/esplora/txgpt_scores.csv and data/interim/injections/txgpt/*.parquet; writes
reports/tables/tuning_txgpt.csv and models/txgpt_method.joblib.
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from lm.tx_tokens import FIELDS

from .baseline import FEATURE_FILES, MODELS_DIR, TABLES_DIR
from .features import median
from .inject import INTERIM_DIR
from .methods import fit_rows, pattern_recalls, plant_metrics, real_metrics, top_fraction_threshold
from .settings import SETTINGS
from .snapshot import MANIFEST_PATH

log = logging.getLogger(__name__)

SCORES_PATH = MANIFEST_PATH.parent / "txgpt_scores.csv"
SECONDS_PER_HOUR = 3600.0


def relative_scores(times: np.ndarray, surprisal: np.ndarray, min_scale: float, min_values: int) -> np.ndarray:
    """Robust z of each surprisal against the strictly earlier ones in the same time-ordered history.

    Same-time (same-block) transactions are never each other's history, as in features.py; with
    fewer than `min_values` earlier values the score is NaN (not enough history to compare).
    """
    out = np.full(len(surprisal), np.nan)
    mad_scale = SETTINGS["baseline"]["madScale"]
    for i in range(len(surprisal)):
        earlier = surprisal[times < times[i]]
        if len(earlier) >= min_values:
            center = median(list(earlier))
            spread = max(mad_scale * median(list(np.abs(earlier - center))), min_scale)
            out[i] = (surprisal[i] - center) / spread
    return out


def event_scores(times: np.ndarray, values: np.ndarray, window_seconds: float) -> np.ndarray:
    """Stage C7's event-level score: each value's positive part plus those of the strictly earlier values within
    `window_seconds` in the same time-ordered history (a burst of mildly surprising transactions adds up). NaN
    where the transaction's own value is NaN; an earlier NaN adds nothing."""
    positive = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        if np.isfinite(values[i]):
            earlier = (times < times[i]) & (times >= times[i] - window_seconds)
            out[i] = positive[i] + positive[earlier].sum()
    return out


def event_columns() -> list[str]:
    """The event-level variants' column names (settings txgpt -> eventWindowsHours)."""
    return [f"event_{base}_{w}h" for base in ("raw", "relative") for w in SETTINGS["txgpt"]["eventWindowsHours"]]


def with_relative(scores: pd.DataFrame, times: pd.DataFrame, group: str) -> pd.DataFrame:
    """Add `relative` and the event-level columns to a surprisal table (keyed by group and txid), per history."""
    t = SETTINGS["txgpt"]
    merged = scores.merge(times, on=[group, "txid"], how="left").sort_values([group, "block_time_utc", "txid"])
    parts = []
    for _, g in merged.groupby(group, sort=False):
        g = g.copy()
        when = g["block_time_utc"].to_numpy(dtype="float64")  # Unix seconds
        g["relative"] = relative_scores(g["block_time_utc"].to_numpy(), g["surprisal"].to_numpy(),
                                        t["relativeMinScale"], SETTINGS["features"]["minComparisonValues"])
        for base, column in (("raw", "surprisal"), ("relative", "relative")):
            for w in t["eventWindowsHours"]:
                g[f"event_{base}_{w}h"] = event_scores(when, g[column].to_numpy(dtype="float64"), w * SECONDS_PER_HOUR)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


class TxGPTMethod:
    """Looks scores up by (address or plant, txid); higher is more unusual."""

    def __init__(self, variant: str, lookup: pd.DataFrame):
        self.variant = variant
        self.lookup = lookup  # columns: key, txid, surprisal, relative, surprisal_<field>
        self.threshold = np.inf

    def _rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        key = frame["plant_id"] if "plant_id" in frame else frame["address"]
        wanted = pd.DataFrame({"key": key.to_numpy(), "txid": frame["txid"].to_numpy()})
        return wanted.merge(self.lookup, on=["key", "txid"], how="left")

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        # A transaction without a score (not enough history for the relative z) is not unusual.
        column = {"raw": "surprisal", "relative": "relative"}.get(self.variant, self.variant)  # or event_*
        return self._rows(frame)[column].fillna(-np.inf).to_numpy(dtype="float64")

    def reasons(self, frame: pd.DataFrame) -> list[str]:
        """The plain-words reason for each row: the field with the largest surprisal."""
        rows = self._rows(frame)[[f"surprisal_{f}" for f in FIELDS]].to_numpy()
        phrases = SETTINGS["txgpt"]["reasons"]
        return [phrases[FIELDS[int(np.nanargmax(r))]] if np.isfinite(r).any() else "" for r in rows]

    def fit(self, rows: pd.DataFrame) -> TxGPTMethod:
        """Only the threshold is fitted: the top 2% of fit-period scores (the model is frozen)."""
        scores = self.score(rows)
        self.threshold = top_fraction_threshold(scores[np.isfinite(scores)])
        return self


def load_lookup(features: pd.DataFrame) -> pd.DataFrame:
    """DS1 and planted-copy scores in one table, with the relative variant computed per history."""
    times = features[["address", "txid", "block_time_utc"]]
    ds1 = with_relative(pd.read_csv(SCORES_PATH), times, "address").rename(columns={"address": "key"})
    parts = [ds1]
    for path in sorted((INTERIM_DIR / "txgpt").glob("*.parquet")):
        copies = pd.read_parquet(INTERIM_DIR / "refined" / path.name, columns=["plant_id", "txid", "block_time_utc"])
        scored = with_relative(pd.read_parquet(path).drop(columns=["address"]), copies, "plant_id")
        parts.append(scored.rename(columns={"plant_id": "key"}))
    return pd.concat(parts, ignore_index=True).drop(columns=["block_time_utc"], errors="ignore")


def plant_scores_exist() -> bool:
    """Whether lm.tx_score --plants has scored the planted copies (data/interim/ is not shipped)."""
    return any((INTERIM_DIR / "txgpt").glob("*.parquet"))


def run_txgpt() -> int:
    from .baseline import load_copies

    if not plant_scores_exist():
        # Without the planted copies' scores every plant would count as missed, and the step would
        # replace the saved method with one tuned on nothing.
        log.error("No planted-copy scores in %s: run python -m lm.tx_score --plants first (minutes on a GPU, "
                  "about 2 h on a CPU). models/txgpt_method.joblib is left as it is.", INTERIM_DIR / "txgpt")
        return 1
    features = pd.read_csv(MANIFEST_PATH.parent / FEATURE_FILES["refined"])
    copies = load_copies("refined", "tuning")
    lookup = load_lookup(features)
    rows, methods = [], {}
    for variant in SETTINGS["txgpt"]["variants"]:
        method = TxGPTMethod(variant, lookup).fit(fit_rows(features))
        methods[variant] = method
        row = {"variant": variant, "threshold": method.threshold, **plant_metrics(method, copies), **real_metrics(method, features)}
        row["mean_recall"] = float(np.mean(pattern_recalls(row)))
        rows.append(row)
        log.info("TxGPT %s: mean tuning recall %.3f (spike %.2f, burst %.2f, dormant %.2f), flag rate %.3f",
                 variant, row["mean_recall"], row["recall_spike"], row["recall_burst_tx"], row["recall_dormant"], row["flag_rate"])
    table = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "tuning_txgpt.csv", index=False)
    best = table.sort_values(["mean_recall", "pr_auc"], ascending=False).iloc[0]["variant"]
    joblib.dump({"method": methods[best], "variant": best, "threshold": methods[best].threshold},
                MODELS_DIR / "txgpt_method.joblib", compress=3)
    log.info("TxGPT: chose the %s score -> models/txgpt_method.joblib", best)
    return 0
