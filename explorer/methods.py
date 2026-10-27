"""What every DS1 method shares: model inputs, flags, and the label-free checks it is judged by.

Serves plan §9.2-9.6 and §11.1 (A1 §2.3.2). A method is anything with `score(frame)` (higher is
more unusual) and a `threshold`; a transaction is flagged when it is scorable and its score reaches
the threshold. Unscorable transactions are shown in the Explorer but never flagged.

Checks, all on the evaluation period:
    plant recall    share of planted transactions flagged, per pattern (burst both per transaction,
                    A1's measure, and per event, at least one of the ten flagged)
    precision       share of flagged rows in the planted copies that are planted: a lower bound,
                    since real rows can be genuinely unusual (DS1 has no labels; plan D-A2-3)
    PR-AUC          threshold-free, against the planted labels, so a method cannot look good just
                    by flagging more
    flag rate       share of real evaluation transactions flagged
    sanity rule     no address may have more than maxFlaggedShare of its evaluation rows flagged
    stability       mean pairwise Jaccard of the top-k evaluation flags across refits
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .features import FEATURE_IDS, PACE_ID
from .settings import SETTINGS
from .split import EVAL, FIT

DIRECTIONS = ("in", "out")  # one-hot; "none" (received == sent) is both zero
# Stage A0: F9 (burst on the address's own clock) joins the pooled models' inputs when features.pace.enabled.
MODEL_FEATURES = FEATURE_IDS + ((PACE_ID,) if SETTINGS["features"]["pace"]["enabled"] else ())
INPUT_COLUMNS = [f"zc_{f}" for f in MODEL_FEATURES] + [f"dir_{d}" for d in DIRECTIONS]


class Method(Protocol):
    threshold: float

    def score(self, frame: pd.DataFrame) -> np.ndarray: ...


def model_inputs(frame: pd.DataFrame) -> np.ndarray:
    """The pooled models' input (plan §6.2): own-history z-scores clipped to +/- zClip, plus a
    direction one-hot, D = 6 (7 with F9, stage A0). A z that cannot be computed (too few comparable earlier values) is
    0, "no evidence of being unusual"; such rows are often unscorable and never flagged anyway."""
    z = frame[[f"zc_{f}" for f in MODEL_FEATURES]].astype("float64").fillna(0.0).to_numpy()
    d = np.stack([(frame["direction"] == x).to_numpy(dtype="float64") for x in DIRECTIONS], axis=1)
    return np.hstack([z, d])


def fit_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Fit-period rows: scorable by construction, real only (planted rows never enter a fit set)."""
    rows = features[features["period"] == FIT]
    if "is_synthetic" in rows:
        rows = rows[rows["is_synthetic"] == 0]
    return rows


def flags(method: Method, frame: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
    scores = method.score(frame)
    cut = method.threshold if threshold is None else threshold
    return frame["scorable"].to_numpy(dtype=bool) & (scores >= cut)


def top_fraction_threshold(fit_scores: np.ndarray, fraction: float = SETTINGS["model"]["flagTopFraction"]) -> float:
    """The score that the top `fraction` of fit-period scores reach (A1: flag the top 2%)."""
    return float(np.quantile(fit_scores, 1 - fraction))


def plant_metrics(method: Method, copies: pd.DataFrame, threshold: float | None = None) -> dict[str, Any]:
    """Recall per pattern, precision lower bound and PR-AUC on the planted copies (evaluation rows)."""
    ev = copies[copies["period"] == EVAL].copy()
    ev["score"] = method.score(ev)
    ev["flag"] = flags(method, ev, threshold)
    planted = ev[ev["is_synthetic"] == 1]
    per_plant = planted.groupby(["pattern", "plant_id"])["flag"].agg(["mean", "max"])
    out: dict[str, Any] = {}
    for pattern in ("spike", "dormant"):
        out[f"recall_{pattern}"] = float(per_plant.loc[pattern, "max"].mean()) if pattern in per_plant.index else np.nan
    if "burst" in per_plant.index:
        out["recall_burst_tx"] = float(per_plant.loc["burst", "mean"].mean())
        out["recall_burst_event"] = float(per_plant.loc["burst", "max"].mean())
    flagged = ev[ev["flag"]]
    out["precision_lower_bound"] = float(flagged["is_synthetic"].mean()) if len(flagged) else np.nan
    scores = np.where(ev["scorable"].to_numpy(dtype=bool), ev["score"].to_numpy(dtype="float64"), -np.inf)
    out["pr_auc"] = float(average_precision_score(ev["is_synthetic"], np.nan_to_num(scores, neginf=-1e9, posinf=1e9)))
    out["plant_prevalence"] = float(ev["is_synthetic"].mean())
    return out


def pattern_recalls(metrics: dict[str, Any]) -> list[float]:
    """The three per-pattern recalls the A1 target refers to (burst per transaction, A1's measure)."""
    return [metrics["recall_spike"], metrics["recall_burst_tx"], metrics["recall_dormant"]]


def meets_recall_target(metrics: dict[str, Any], p: dict[str, Any] = SETTINGS["injection"]) -> bool:
    return sum(r >= p["recallTarget"] for r in pattern_recalls(metrics)) >= p["patternsMeetingTarget"]


def real_metrics(method: Method, features: pd.DataFrame, threshold: float | None = None) -> dict[str, Any]:
    """Flag rate and the sanity rule on the real evaluation period."""
    ev = features[features["period"] == EVAL].copy()
    ev["flag"] = flags(method, ev, threshold)
    share = ev.groupby("address")["flag"].mean()
    limit = SETTINGS["sanity"]["maxFlaggedShare"]
    return {"flag_rate": float(ev["flag"].mean()), "flagged": int(ev["flag"].sum()), "eval_rows": len(ev),
            "max_address_share": float(share.max()), "addresses_over_limit": int((share > limit).sum())}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def stability(top_sets: list[set]) -> float:
    """Mean pairwise Jaccard of the top-k flag sets from repeated fits (A1 target >= 0.7)."""
    pairs = list(combinations(top_sets, 2))
    return float(np.mean([jaccard(a, b) for a, b in pairs])) if pairs else 1.0


def top_k(method: Method, features: pd.DataFrame, k: int) -> set[str]:
    """The k highest-scoring scorable evaluation transactions, as address:txid keys."""
    ev = features[(features["period"] == EVAL) & features["scorable"]]
    order = np.argsort(-method.score(ev), kind="stable")[:k]
    return set((ev["address"] + ":" + ev["txid"]).iloc[order])
