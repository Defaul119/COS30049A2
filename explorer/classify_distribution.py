"""Distributions of the classifiers' test metrics: training seeds and a bootstrap of the test set (DS3, DS4).

Serves the statistics campaign (user, 27 Oct 2026: distributions and mean accuracy for every model). The
reported classifiers (explorer/classify.py) are tuned once and scored once. Here each tuned configuration, with the
threshold its validation set chose, is refitted on the same training rows with several seeds (settings
classDistribution -> seeds; only the random forest and gradient boosting depend on a seed), and every fit is scored on
the same test rows. The first seed's test scores are then resampled with replacement (a bootstrap over test rows),
which gives each metric a confidence interval and each pair of models a paired difference on identical resamples.

Elliptic uses its temporal split (steps 1-34 / 35-49), all features; BitcoinHeist its temporal and random schemes;
`year` and `day` are never features (classify.HEIST_META). Elliptic's rows are not shared (CC BY-NC-ND), so this
runs where the raw data is (kaggle/class_distribution attaches the Kaggle copy).

Writes reports/tables/stats_classifiers_seeds.csv, stats_classifiers_summary.csv and stats_classifiers_paired.csv.
"""
from __future__ import annotations

import json
import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .baseline import TABLES_DIR
from .classify import HEIST_META, MODEL_KINDS, heist_parts, make_model, stratified_subsample, test_metrics
from .labelled import ELLIPTIC_DIR, FEATURE_SETS, FIT, HEIST_DIR, TEST, VAL
from .settings import SETTINGS

log = logging.getLogger(__name__)
SEEDED = ("randomForest", "histGradientBoosting")
METRICS = ("accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc")


def schemes() -> dict[str, tuple[list[str], pd.DataFrame, pd.DataFrame, int | None, pd.DataFrame]]:
    """name -> (features, train rows, test rows, KNN subsample, the reported results with params and threshold)."""
    out = {}
    rows = pd.read_parquet(ELLIPTIC_DIR / "rows.parquet")
    rows = rows[rows["y"].notna()].assign(y=lambda f: f["y"].astype(int))
    results = pd.read_csv(TABLES_DIR / "elliptic_results.csv")
    out["Elliptic (temporal, all features)"] = (FEATURE_SETS["AF"], rows[rows["period"].isin([FIT, VAL])],
                                                rows[rows["period"] == TEST], None, results[results["feature_set"] == "AF"])
    heist = pd.read_parquet(HEIST_DIR / "rows.parquet")
    features = [c for c in heist.columns if c not in HEIST_META]
    results = pd.read_csv(TABLES_DIR / "heist_results.csv")
    for scheme in ("temporal", "random"):
        _, _, train, test = heist_parts(heist, scheme)
        out[f"BitcoinHeist ({scheme})"] = (features, train, test, SETTINGS["bitcoinheist"]["knnSubsample"],
                                           results[results["scheme"] == scheme])
    return out


def bootstrap(y: np.ndarray, p: dict[str, np.ndarray], thresholds: dict[str, float], draws: int, seed: int
              ) -> dict[str, dict[str, np.ndarray]]:
    """Per model: F1, precision, recall and PR-AUC on each resample of the test rows (the same resamples for every
    model, so differences between models are paired)."""
    rng = np.random.default_rng(seed)
    out = {m: {k: np.empty(draws) for k in ("f1", "precision", "recall", "pr_auc")} for m in p}
    for i in range(draws):
        idx = rng.integers(0, len(y), len(y))
        yi = y[idx]
        for m, scores in p.items():
            pred = scores[idx] >= thresholds[m]
            tp = np.sum(pred & (yi == 1))
            fp, fn = np.sum(pred) - tp, np.sum(yi) - tp
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            out[m]["precision"][i], out[m]["recall"][i] = precision, recall
            out[m]["f1"][i] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            out[m]["pr_auc"][i] = average_precision_score(yi, scores[idx])
    return out


def run_classifier_distribution() -> int:
    d, cfg = SETTINGS["classDistribution"], SETTINGS["classification"]
    level = SETTINGS["significance"]["ciLevel"]
    tail = 100 * (1 - level) / 2
    seed_rows, summary, paired = [], [], []
    for name, (features, train, test, knn_rows, results) in schemes().items():
        y = test["y"].to_numpy()
        first: dict[str, np.ndarray] = {}
        thresholds: dict[str, float] = {}
        for kind in MODEL_KINDS:
            reported = results[results["model"] == kind].iloc[0]
            params, thresholds[kind] = json.loads(reported["params"]), float(reported["threshold"])
            for seed in range(d["firstSeed"], d["firstSeed"] + (d["seeds"] if kind in SEEDED else 1)):
                fit = stratified_subsample(train, knn_rows, cfg["seed"]) if kind == "knn" and knn_rows else train
                model = make_model(kind, params, {**cfg, "seed": seed}, len(fit)).fit(fit[features], fit["y"])
                p = model.predict_proba(test[features])[:, 1]
                first.setdefault(kind, p)
                m = test_metrics(y, p, thresholds[kind])
                seed_rows.append({"scheme": name, "model": kind, "seed": seed, **{k: m[k] for k in METRICS}})
                log.info("%s %s seed %d: F1 %.4f, PR-AUC %.4f", name, kind, seed, m["f1"], m["pr_auc"])
        boot = bootstrap(y, first, thresholds, d["bootstrapDraws"], d["firstSeed"])
        seeds = pd.DataFrame([r for r in seed_rows if r["scheme"] == name])
        for kind in MODEL_KINDS:
            g = seeds[seeds["model"] == kind]
            row: dict[str, Any] = {"scheme": name, "model": kind, "seeds": len(g), "test_rows": len(y),
                                   **{f"{k}_mean": g[k].mean() for k in METRICS},
                                   **{f"{k}_seed_sd": g[k].std(ddof=1) if len(g) > 1 else 0.0 for k in METRICS}}
            for k, values in boot[kind].items():
                row[f"{k}_ci_low"], row[f"{k}_ci_high"] = np.percentile(values, tail), np.percentile(values, 100 - tail)
            summary.append(row)
        for a, b in combinations(MODEL_KINDS, 2):
            diff = boot[a]["f1"] - boot[b]["f1"]
            paired.append({"scheme": name, "model_a": a, "model_b": b, "f1_diff_mean": diff.mean(),
                           "f1_diff_ci_low": np.percentile(diff, tail), "f1_diff_ci_high": np.percentile(diff, 100 - tail),
                           "a_better_share": float((diff > 0).mean())})
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(seed_rows).to_csv(TABLES_DIR / "stats_classifiers_seeds.csv", index=False)
    pd.DataFrame(summary).to_csv(TABLES_DIR / "stats_classifiers_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(TABLES_DIR / "stats_classifiers_paired.csv", index=False)
    for r in summary:
        log.info("%-36s %-22s F1 %.3f (seed sd %.3f) [%.3f, %.3f]", r["scheme"], r["model"], r["f1_mean"],
                 r["f1_seed_sd"], r["f1_ci_low"], r["f1_ci_high"])
    return 0
