"""Isolation Forest and Local Outlier Factor: the A1 outlier methods, kept as HD extras.

Serves plan §9.6 (docs/plan/sections/s08-models.tex). Both are fitted on fit-period rows pooled
across addresses, using each transaction's own-history z-scores (clipped) and direction
(methods.model_inputs), which makes "unusual for this address" comparable across addresses. Both
score new transactions without refitting (criterion C3): Isolation Forest natively, LOF in novelty
mode. Score = -score_samples, so higher is more unusual; the flag threshold is the top 2% of
fit-period scores (A1).

Tuning has no labels to use, so hyperparameters are chosen by mean recall on the TUNING plants
(seeds 40000-40004) among configurations whose stability (mean pairwise Jaccard of the top
evaluation flags across bootstrap refits) reaches the target; reported results use other seeds.

Writes reports/tables/tuning_outliers.csv and models/{isolation_forest,lof}.joblib.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import LocalOutlierFactor

from .baseline import FEATURE_FILES, MODELS_DIR, TABLES_DIR, load_copies
from .methods import fit_rows, model_inputs, pattern_recalls, plant_metrics, real_metrics, stability, top_fraction_threshold, top_k
from .settings import SETTINGS
from .snapshot import MANIFEST_PATH

log = logging.getLogger(__name__)


class IsolationForestMethod:
    def __init__(self, params: dict[str, Any], seed: int):
        self.params, self.seed = params, seed

    def fit(self, rows: pd.DataFrame) -> IsolationForestMethod:
        self.model = IsolationForest(**self.params, random_state=self.seed).fit(model_inputs(rows))
        self.threshold = top_fraction_threshold(self.score(rows))
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(model_inputs(frame))


class LOFMethod:
    def __init__(self, params: dict[str, Any], seed: int):
        self.params, self.seed = params, seed  # LOF is deterministic; the seed is kept for a uniform interface

    def fit(self, rows: pd.DataFrame) -> LOFMethod:
        self.model = LocalOutlierFactor(**self.params, novelty=True).fit(model_inputs(rows))
        # In novelty mode score_samples on the training rows counts each row as its own neighbour,
        # so the fit-period threshold uses the training LOF values instead.
        self.threshold = top_fraction_threshold(-self.model.negative_outlier_factor_)
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(model_inputs(frame))


METHODS = {"isolation_forest": (IsolationForestMethod, "isolationForest"), "lof": (LOFMethod, "lof")}


def bootstrap_stability(cls: type, params: dict[str, Any], features: pd.DataFrame) -> float:
    """Mean pairwise Jaccard of the top-k evaluation flags over refits with different seeds on
    bootstrap resamples of the fit period (plan §11.1: seeds x resamples)."""
    m = SETTINGS["model"]
    rows = fit_rows(features)
    sets = []
    for i in range(m["stabilitySeeds"]):
        for j in range(m["stabilityBootstraps"]):
            sample = rows.sample(frac=1.0, replace=True, random_state=m["seed"] + 100 * i + j)
            sets.append(top_k(cls(params, m["seed"] + i).fit(sample), features, m["stabilityTopK"]))
    return stability(sets)


def run_outliers() -> int:
    m = SETTINGS["model"]
    features = {name: pd.read_csv(MANIFEST_PATH.parent / f) for name, f in FEATURE_FILES.items()}
    copies = {name: load_copies(name, "tuning") for name in FEATURE_FILES}
    rows: list[dict[str, Any]] = []
    for name, (cls, key) in METHODS.items():
        for params in ParameterGrid(SETTINGS["outliers"][key]):
            method = cls(params, m["seed"]).fit(fit_rows(features["refined"]))
            row = {"method": name, "features": "refined", "params": json.dumps(params),
                   **plant_metrics(method, copies["refined"]), **real_metrics(method, features["refined"]),
                   "stability": bootstrap_stability(cls, params, features["refined"])}
            row["mean_recall"] = float(np.mean(pattern_recalls(row)))
            rows.append(row)
            log.info("%s %s: mean recall %.3f (spike %.2f, burst %.2f, dormant %.2f), stability %.2f, flag rate %.3f",
                     name, params, row["mean_recall"], row["recall_spike"], row["recall_burst_tx"],
                     row["recall_dormant"], row["stability"], row["flag_rate"])
    table = pd.DataFrame(rows)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, (cls, key) in METHODS.items():
        candidates = table[(table["method"] == name)]
        stable = candidates[candidates["stability"] >= m["stabilityTarget"]]
        pool, rule = (stable, "highest mean tuning recall among stable configurations") if len(stable) else (
            candidates, "no configuration reached the stability target; highest mean tuning recall")
        best = pool.sort_values(["mean_recall", "pr_auc"], ascending=False).iloc[0]
        params = json.loads(best["params"])
        final = {feat: cls(params, m["seed"]).fit(fit_rows(features[feat])) for feat in FEATURE_FILES}
        # The ablation: the chosen configuration on the A1-as-written features, not re-tuned.
        aw = final["as_written"]
        rows.append({"method": name, "features": "as_written", "params": best["params"],
                     **plant_metrics(aw, copies["as_written"]), **real_metrics(aw, features["as_written"]),
                     "stability": np.nan})
        rows[-1]["mean_recall"] = float(np.mean(pattern_recalls(rows[-1])))
        joblib.dump({"method": final["refined"], "params": params, "rule": rule, "inputs": "methods.model_inputs",
                     "threshold": final["refined"].threshold}, MODELS_DIR / f"{name}.joblib", compress=3)
        log.info("%s: chose %s (%s) -> models/%s.joblib", name, params, rule, name)
    pd.DataFrame(rows).to_csv(TABLES_DIR / "tuning_outliers.csv", index=False)
    log.info("Wrote %s", TABLES_DIR / "tuning_outliers.csv")
    return 0
