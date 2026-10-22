"""Classification on the labelled datasets: search, threshold, refit and a single test evaluation.

Serves plan §9.4 (docs/plan/sections/s08-models.tex) and the brief's classification requirement.
Supervised learning is used only where labels exist (selection criterion C1): DS3 Elliptic and DS4
BitcoinHeist, never DS1 or DS2. "Illicit" and "ransomware" are those datasets' own class names.

The protocol, for each split scheme and each of the four models:
1. Try the model's grid (every configuration for logistic regression and KNN, `nIter` seeded random
   draws for random forest and histogram gradient boosting), fitting on the fit set and scoring the
   validation set, which keeps the real class ratio. The selection score is the best F1 for the
   positive class over all thresholds.
2. The decision threshold is the one that maximised validation F1 for the chosen configuration,
   never anything measured on the test set.
3. Refit the chosen configuration on fit + validation and evaluate once on the test set.
Scalers sit inside scikit-learn Pipelines, so they only ever see training rows.

Each (dataset, scheme, model) result is saved as soon as it is done (data/interim/classify/), so an
interrupted run continues where it stopped. Writes reports/tables/<dataset>_tuning.csv,
<dataset>_results.csv and (BitcoinHeist) heist_lofo.csv, and the best model per dataset to
models/<dataset>_best.joblib.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .labelled import HEIST_DIR, TEST, VAL
from .settings import PROJECT_ROOT, SETTINGS

log = logging.getLogger(__name__)

INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "classify"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_KINDS = ("logistic", "knn", "randomForest", "histGradientBoosting")
FULL_GRID = {"logistic", "knn"}  # small grids are searched exhaustively; the rest by random draws


# -- models ------------------------------------------------------------------------------


def make_model(kind: str, params: dict[str, Any], cfg: dict[str, Any], n_rows: int) -> Pipeline:
    """A Pipeline for one configuration. Distance- and gradient-based models get a scaler inside it."""
    seed, n_jobs = cfg["seed"], cfg["nJobs"]
    if kind == "logistic":
        model = LogisticRegression(C=params["C"], class_weight=params["class_weight"], max_iter=cfg["logistic"]["max_iter"])
        return Pipeline([("scale", StandardScaler()), ("model", model)])
    if kind == "knn":
        model = KNeighborsClassifier(n_neighbors=params["n_neighbors"], weights=params["weights"], p=params["p"],
                                     algorithm="kd_tree", n_jobs=n_jobs)
        return Pipeline([("scale", StandardScaler()), ("model", model)])
    if kind == "randomForest":
        model = RandomForestClassifier(**params, max_samples=min(cfg["rfMaxSamples"], n_rows), n_jobs=n_jobs,
                                       random_state=seed)
        return Pipeline([("model", model)])
    if kind == "histGradientBoosting":
        model = HistGradientBoostingClassifier(**params, early_stopping=True, random_state=seed)
        return Pipeline([("model", model)])
    raise ValueError(f"unknown model kind {kind}")


def configurations(kind: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    space = {k: v for k, v in cfg[kind].items() if k != "max_iter"}
    if kind in FULL_GRID:
        return list(ParameterGrid(space))
    return list(ParameterSampler(space, n_iter=cfg["nIter"], random_state=cfg["seed"]))


# -- metrics -----------------------------------------------------------------------------


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(threshold, F1) maximising F1 for the positive class over every distinct score threshold."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)[:-1]
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])


def test_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    """Every metric the plan reports for a labelled test set, at a threshold fixed beforehand."""
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n_test": int(len(y)), "positives": int(y.sum()),
        "prevalence": float(y.mean()),
    }


def stratified_subsample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """At most n rows, keeping the class ratio (KNN's memory and time grow with the training rows)."""
    if len(frame) <= n:
        return frame
    return frame.groupby("y", group_keys=False).sample(frac=n / len(frame), random_state=seed)


# -- one model on one scheme -----------------------------------------------------------------


def search_and_evaluate(
    kind: str, features: list[str], fit: pd.DataFrame, val: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
    cfg: dict[str, Any], knn_rows: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], Pipeline, np.ndarray]:
    """Search `kind`'s grid on fit/val, refit the best on train, and evaluate once on test.

    Returns (result, tuning rows, final model, test scores).
    """
    if kind == "knn" and knn_rows:
        fit = stratified_subsample(fit, knn_rows, cfg["seed"])
        train = stratified_subsample(train, knn_rows, cfg["seed"])
    tuning, best = [], None
    for params in configurations(kind, cfg):
        started = time.perf_counter()
        model = make_model(kind, params, cfg, len(fit)).fit(fit[features], fit["y"])
        p_val = model.predict_proba(val[features])[:, 1]
        threshold, f1 = best_f1_threshold(val["y"].to_numpy(), p_val)
        row = {"model": kind, "params": json.dumps(params), "val_f1": f1, "val_pr_auc": average_precision_score(val["y"], p_val),
               "threshold": threshold, "seconds": round(time.perf_counter() - started, 1)}
        tuning.append(row)
        log.info("  %s %s: validation F1 %.4f, PR-AUC %.4f (%.0f s)", kind, params, f1, row["val_pr_auc"], row["seconds"])
        if best is None or f1 > best[0]["val_f1"]:
            best = (row, params)
    row, params = best
    model = make_model(kind, params, cfg, len(train)).fit(train[features], train["y"])
    p_test = model.predict_proba(test[features])[:, 1]
    result = {"model": kind, "params": params, "threshold": row["threshold"], "val_f1": row["val_f1"],
              "val_pr_auc": row["val_pr_auc"], "fit_rows": len(fit), "train_rows": len(train),
              **test_metrics(test["y"].to_numpy(), p_test, row["threshold"])}
    return result, tuning, model, p_test


# -- BitcoinHeist ----------------------------------------------------------------------------


HEIST_META = ["y", "family", "year", "temporal", "random", "temporal_sampled", "temporal_train_sampled",
              "random_sampled", "random_train_sampled"]


def heist_parts(rows: pd.DataFrame, scheme: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(fit, val, train, test) for a scheme: fit and train downsampled, val and test at the real ratio."""
    return (rows[rows[f"{scheme}_sampled"]], rows[rows[scheme] == VAL], rows[rows[f"{scheme}_train_sampled"]],
            rows[rows[scheme] == TEST])


def run_classify_heist(schemes: list[str], force: bool = False, rows_path: Path = HEIST_DIR / "rows.parquet") -> int:
    cfg, heist = SETTINGS["classification"], SETTINGS["bitcoinheist"]
    rows = pd.read_parquet(rows_path)
    features = [c for c in rows.columns if c not in HEIST_META]
    decisions = json.loads((rows_path.parent / "processing.json").read_text(encoding="utf-8"))
    log.info("BitcoinHeist: %d rows, features %s", len(rows), features)
    out_dir = INTERIM_DIR / "heist"
    out_dir.mkdir(parents=True, exist_ok=True)

    for scheme in [s for s in schemes if s in ("temporal", "random")]:
        fit, val, train, test = heist_parts(rows, scheme)
        log.info("Scheme %s: fit %d, val %d, train %d, test %d rows", scheme, len(fit), len(val), len(train), len(test))
        for kind in MODEL_KINDS:
            done = out_dir / f"{scheme}_{kind}.json"
            if done.exists() and not force:
                log.info("%s/%s already done; skipping (use --force to redo)", scheme, kind)
                continue
            result, tuning, model, p_test = search_and_evaluate(kind, features, fit, val, train, test, cfg, heist["knnSubsample"])
            np.save(out_dir / f"{scheme}_{kind}_test_scores.npy", p_test.astype("float32"))
            joblib.dump(model, out_dir / f"{scheme}_{kind}.joblib", compress=3)
            pd.DataFrame(tuning).assign(scheme=scheme).to_csv(out_dir / f"{scheme}_{kind}_tuning.csv", index=False)
            done.write_text(json.dumps({"scheme": scheme, **result}, indent=2), encoding="utf-8")
            log.info("%s/%s: test F1 %.4f, precision %.4f, recall %.4f, PR-AUC %.4f", scheme, kind,
                     result["f1"], result["precision"], result["recall"], result["pr_auc"])

    if "lofo" in schemes:
        run_heist_lofo(rows, features, decisions["leave_out_families"], cfg, heist, out_dir, force)
    return write_heist_tables(out_dir, features)


def run_heist_lofo(rows: pd.DataFrame, features: list[str], families: list[str], cfg: dict[str, Any],
                   heist: dict[str, Any], out_dir: Path, force: bool) -> None:
    """Leave one family out: each model's best random-split configuration, trained without one family's
    positives, threshold from validation without them, then recall on every row of that family."""
    fit, val, train, _ = heist_parts(rows, "random")
    for kind in MODEL_KINDS:
        best = json.loads((out_dir / f"random_{kind}.json").read_text(encoding="utf-8"))
        for family in families:
            done = out_dir / f"lofo_{kind}_{family}.json"
            if done.exists() and not force:
                continue
            drop = lambda f: f[f["family"] != family]  # noqa: E731 - the same filter for every part
            f_fit, f_val, f_train = drop(fit), drop(val), drop(train)
            if kind == "knn":
                f_fit = stratified_subsample(f_fit, heist["knnSubsample"], cfg["seed"])
                f_train = stratified_subsample(f_train, heist["knnSubsample"], cfg["seed"])
            model = make_model(kind, best["params"], cfg, len(f_fit)).fit(f_fit[features], f_fit["y"])
            threshold, _ = best_f1_threshold(f_val["y"].to_numpy(), model.predict_proba(f_val[features])[:, 1])
            model = make_model(kind, best["params"], cfg, len(f_train)).fit(f_train[features], f_train["y"])
            held = rows[rows["family"] == family]
            recall = float((model.predict_proba(held[features])[:, 1] >= threshold).mean())
            done.write_text(json.dumps({"model": kind, "family": family, "threshold": threshold,
                                        "positives": len(held), "recall": recall}), encoding="utf-8")
            log.info("Leave out %s, %s: recall %.4f on %d positives", family, kind, recall, len(held))


def write_heist_tables(out_dir: Path, features: list[str], tables_dir: Path = TABLES_DIR) -> int:
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*_*.json"))
               if not p.name.startswith("lofo_")]
    tables_dir.mkdir(parents=True, exist_ok=True)
    if results:
        table = pd.DataFrame(results)
        table["params"] = table["params"].map(json.dumps)
        table.to_csv(tables_dir / "heist_results.csv", index=False)
        tuning = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob("*_tuning.csv"))], ignore_index=True)
        tuning.to_csv(tables_dir / "heist_tuning.csv", index=False)
        # The model the prediction CLI uses: the best validation F1 on the temporal scheme, the
        # deployment-like split (train on the past, predict the future).
        temporal = [r for r in results if r["scheme"] == "temporal"]
        if temporal:
            best = max(temporal, key=lambda r: r["val_f1"])
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model = joblib.load(out_dir / f"temporal_{best['model']}.joblib")
            joblib.dump({"model": model, "features": features, "threshold": best["threshold"], "result": best},
                        MODELS_DIR / "heist_best.joblib", compress=3)
            log.info("Best temporal model: %s (validation F1 %.4f) -> models/heist_best.joblib", best["model"], best["val_f1"])
    lofo = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("lofo_*.json"))]
    if lofo:
        pd.DataFrame(lofo).to_csv(tables_dir / "heist_lofo.csv", index=False)
    log.info("Wrote reports/tables/heist_results.csv (%d rows), heist_tuning.csv, heist_lofo.csv (%d rows)",
             len(results), len(lofo))
    return 0
