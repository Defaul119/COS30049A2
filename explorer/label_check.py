"""The label check: how far do the unsupervised methods' "unusual" scores coincide with labels?

Serves plan §9.4 ("The label check", docs/plan/sections/s08-models.tex) and §10.2, and the HD
"additional ML implementations" line. DS1 has no labels, so the methods the Explorer uses are judged
there only by label-free checks (D-A2-3). Here the same four methods are fitted, without labels,
on the training period of each labelled dataset and then scored against that dataset's own classes:

* DS3 Elliptic: fit on every node of steps 1..trainLastStep, unknown nodes included (no label is
  read), AF features; score the labelled test nodes (later steps).
* DS4 BitcoinHeist, temporal scheme: fit on the fit and validation years; score the test years.
  `year` and `day` are never inputs.

Every method sees the same seeded sample of `fitSample` training rows (LOF and DBSCAN hold their
training rows in memory), scaled by one StandardScaler fitted on that sample only. Isolation Forest and LOF
use the configuration chosen on DS1 (models/{isolation_forest,lof}.joblib), K-Means chooses k by
silhouette and DBSCAN uses the noise-band rule, both on the sample and both label-free
(clustering.tune_kmeans / tune_dbscan). Scores are higher = more unusual, as on DS1.

Measures, on the test rows: PR-AUC and ROC-AUC against the labels, and precision, recall and lift
among the top `flagTopFraction` (2%) of test rows by score, next to prevalence (the chance level).
The supervised random forest's saved test scores (explorer.classify) give the upper reference.
"Illicit" and "ransomware" are those datasets' class names; the result is evidence for the
Explorer's wording ("unusual" is not "labelled illicit"), not a judgement of any DS1 address.

Writes reports/tables/label_check.csv and label_check_tuning.csv (metrics only: Elliptic's
licence forbids sharing its rows or derivatives).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .classify import HEIST_META, INTERIM_DIR, MODELS_DIR, TABLES_DIR
from .clustering import tune_dbscan, tune_kmeans
from .labelled import ELLIPTIC_DIR, FEATURE_SETS, FIT, HEIST_DIR, TEST, VAL
from .settings import SETTINGS

log = logging.getLogger(__name__)

METHODS = ("isolation_forest", "lof", "kmeans", "dbscan")


def ds1_params(name: str, models_dir: Path = MODELS_DIR) -> dict[str, Any]:
    """The configuration the DS1 search chose for an outlier method (explorer.outliers)."""
    path = models_dir / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run python -m explorer.cli outliers first")
    return joblib.load(path)["params"]


def fit_scorers(x_fit: np.ndarray, seed: int, params: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit the four methods on (unscaled) training rows. Returns ({method: score function}, tuning rows).

    Each score function maps unscaled rows to scores, higher = more unusual.
    """
    # One scaler, fitted on the training sample only, feeds all four methods, so they see the same space.
    scaler = StandardScaler().fit(x_fit)
    x = scaler.transform(x_fit)
    tuning: list[dict[str, Any]] = []

    forest = IsolationForest(**params["isolation_forest"], random_state=seed).fit(x)
    lof = LocalOutlierFactor(**params["lof"], novelty=True).fit(x)

    km_table, k = tune_kmeans(x, seed)
    kmeans = KMeans(n_clusters=k, n_init=SETTINGS["clustering"]["kmeansNInit"], random_state=seed).fit(x)
    tuning += [{"method": "kmeans", **r} for r in km_table.to_dict("records")]

    db_table, db = tune_dbscan(x, seed)
    core = DBSCAN(eps=db["eps"], min_samples=db["min_samples"]).fit(x).core_sample_indices_
    tuning += [{"method": "dbscan", **r} for r in db_table.to_dict("records")]
    if len(core) == 0:
        raise ValueError("DBSCAN found no core samples on the label-check sample")
    nearest_core = NearestNeighbors(n_neighbors=1).fit(x[core])

    scorers = {
        "isolation_forest": lambda rows: -forest.score_samples(scaler.transform(rows)),
        "lof": lambda rows: -lof.score_samples(scaler.transform(rows)),
        "kmeans": lambda rows: kmeans.transform(scaler.transform(rows)).min(axis=1),
        "dbscan": lambda rows: nearest_core.kneighbors(scaler.transform(rows))[0][:, 0],
    }
    return scorers, tuning


def label_metrics(y: np.ndarray, scores: np.ndarray, top_fraction: float) -> dict[str, Any]:
    """PR-AUC, ROC-AUC, and precision / recall / lift among the top `top_fraction` of rows by score."""
    n_top = max(1, int(round(top_fraction * len(y))))
    top = np.argsort(-scores, kind="stable")[:n_top]
    prevalence = float(y.mean())
    precision = float(y[top].mean())
    return {
        "pr_auc": float(average_precision_score(y, scores)), "roc_auc": float(roc_auc_score(y, scores)),
        "precision_top": precision, "recall_top": float(y[top].sum() / y.sum()),
        "lift_top": precision / prevalence if prevalence else np.nan, "top_rows": n_top,
        "n_test": int(len(y)), "positives": int(y.sum()), "prevalence": prevalence,
    }


def check(dataset: str, x_train: pd.DataFrame, test: pd.DataFrame, features: list[str],
          reference: tuple[str, np.ndarray] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit on a sample of x_train, score `test` (which has a 0/1 `y`), and measure against the labels."""
    p, m = SETTINGS["labelCheck"], SETTINGS["model"]
    sample = x_train.sample(n=min(p["fitSample"], len(x_train)), random_state=p["seed"])
    params = {name: ds1_params(name) for name in ("isolation_forest", "lof")}
    log.info("%s: fitting on %d of %d training rows, %d features; %d test rows (%d positive)", dataset, len(sample),
             len(x_train), len(features), len(test), int(test["y"].sum()))
    scorers, tuning = fit_scorers(sample[features].to_numpy("float64"), p["seed"], params)
    y, x_test = test["y"].to_numpy("int64"), test[features].to_numpy("float64")
    rows = []
    for name in METHODS:
        row = {"dataset": dataset, "method": name, "supervised": False, "fit_rows": len(sample),
               **label_metrics(y, scorers[name](x_test), m["flagTopFraction"])}
        rows.append(row)
        log.info("%s %s: PR-AUC %.4f (prevalence %.4f), precision in top %.0f%% %.4f, recall %.4f", dataset, name,
                 row["pr_auc"], row["prevalence"], 100 * m["flagTopFraction"], row["precision_top"], row["recall_top"])
    if reference is not None:
        name, scores = reference
        rows.append({"dataset": dataset, "method": name, "supervised": True, "fit_rows": np.nan,
                     **label_metrics(y, scores, m["flagTopFraction"])})
    return rows, [{"dataset": dataset, **r} for r in tuning]


def saved_scores(path: Path, n: int) -> np.ndarray | None:
    """A supervised model's saved test scores, if explorer.classify wrote them for these test rows."""
    if not path.exists():
        log.info("No %s; the supervised reference row is left out", path)
        return None
    scores = np.load(path)
    return scores if len(scores) == n else None


def elliptic_parts(rows: pd.DataFrame, train_last: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training nodes of steps <= train_last, labels unread; labelled test nodes)."""
    train = rows[rows["time_step"] <= train_last]
    test = rows[(rows["time_step"] > train_last) & rows["y"].notna()].assign(y=lambda f: f["y"].astype(int))
    return train, test


def run_label_check(tables_dir: Path = TABLES_DIR) -> int:
    """Step `label-check`: the unsupervised methods' scores against DS3's and DS4's labels (PR-AUC against
    prevalence), next to a supervised reference; writes label_check.csv and label_check_tuning.csv."""
    results, tuning = [], []
    heist_path, elliptic_path = HEIST_DIR / "rows.parquet", ELLIPTIC_DIR / "rows.parquet"

    if elliptic_path.exists():
        rows = pd.read_parquet(elliptic_path)
        train, test = elliptic_parts(rows, SETTINGS["elliptic"]["trainLastStep"])
        ref = saved_scores(INTERIM_DIR / "elliptic" / "AF_randomForest_test_scores.npy", len(test))
        r, t = check("elliptic_AF", train, test, FEATURE_SETS["AF"], None if ref is None else ("random_forest (supervised)", ref))
        results += r
        tuning += t
    else:
        log.info("Skipping DS3: %s missing (python -m explorer.cli process-external)", elliptic_path)

    if heist_path.exists():
        rows = pd.read_parquet(heist_path)
        features = [c for c in rows.columns if c not in HEIST_META]
        train, test = rows[rows["temporal"].isin([FIT, VAL])], rows[rows["temporal"] == TEST]
        ref = saved_scores(INTERIM_DIR / "heist" / "temporal_randomForest_test_scores.npy", len(test))
        r, t = check("heist_temporal", train, test, features, None if ref is None else ("random_forest (supervised)", ref))
        results += r
        tuning += t
    else:
        log.info("Skipping DS4: %s missing (python -m explorer.cli process-external)", heist_path)

    if not results:
        log.info("Nothing to check: neither labelled dataset is processed")
        return 0
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(tables_dir / "label_check.csv", index=False)
    pd.DataFrame(tuning).to_csv(tables_dir / "label_check_tuning.csv", index=False)
    log.info("Wrote %s and label_check_tuning.csv", tables_dir / "label_check.csv")
    return 0
