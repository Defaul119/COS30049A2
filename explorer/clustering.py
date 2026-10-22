"""Clustering of DS1 transactions: K-Means and DBSCAN, with distance-based unusualness.

Serves plan §9.4 (docs/plan/sections/s08-models.tex) and the brief's clustering model type. Both are
fitted on fit-period rows pooled across addresses, on each transaction's own-history z-scores
(clipped) and direction (methods.model_inputs, D = 6), and both score new transactions without
refitting (criterion C3).

K-Means: k is chosen by the best silhouette over settings.json -> clustering.kmeansK, reported next
to inertia (the elbow), Davies-Bouldin and Calinski-Harabasz. Each centroid is described in words
from its mean z-scores, which is what makes a cluster useful to an analyst. Unusualness is the
distance to the nearest centroid; the flag threshold is the top 2% of fit-period distances (A1).

DBSCAN: MinPts starts at 2D = 12 (the Week 7 rule); eps0 is the knee of the sorted 12-nearest-
neighbour distances (the point farthest from the chord). The grid tries eps factors x MinPts and
keeps settings whose fit-period noise share is within dbscanNoiseShare, choosing the best
silhouette on non-noise points. scikit-learn's DBSCAN cannot score new points, so a new point is
noise (flagged) when no core sample lies within eps, and its ranking score is the distance to the
nearest core sample (a NearestNeighbors index on the core samples); the report states this rule.

Writes reports/tables/tuning_kmeans.csv, tuning_dbscan.csv, kmeans_clusters.csv and
models/{kmeans,dbscan}.joblib.
"""
from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from .baseline import FEATURE_FILES, MODELS_DIR, TABLES_DIR
from .features import FEATURE_IDS
from .methods import INPUT_COLUMNS, fit_rows, model_inputs, top_fraction_threshold
from .settings import SETTINGS
from .snapshot import MANIFEST_PATH

log = logging.getLogger(__name__)

PHRASES = {  # how a centroid's mean z reads in words, by feature and sign
    "F1": ("larger than this address's usual amounts", "smaller than this address's usual amounts"),
    "F2": ("in a busier week than usual", "in a quieter week than usual"),
    "F3": ("after a longer gap than usual", "a quick follow-up"),
    "F4": ("large relative to the usual size", "small relative to the usual size"),
}


class KMeansMethod:
    def __init__(self, k: int, seed: int):
        self.k, self.seed = k, seed

    def fit(self, rows: pd.DataFrame) -> KMeansMethod:
        c = SETTINGS["clustering"]
        self.model = KMeans(n_clusters=self.k, n_init=c["kmeansNInit"], random_state=self.seed).fit(model_inputs(rows))
        self.threshold = top_fraction_threshold(self.score(rows))
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Distance to the nearest centroid (in z units)."""
        return self.model.transform(model_inputs(frame)).min(axis=1)

    def cluster(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(model_inputs(frame))


class DBSCANMethod:
    def __init__(self, eps: float, min_samples: int):
        self.eps, self.min_samples = eps, min_samples

    def fit(self, rows: pd.DataFrame) -> DBSCANMethod:
        x = model_inputs(rows)
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(x)
        core = self.model.core_sample_indices_
        self.core_labels = self.model.labels_[core]
        self.index = NearestNeighbors(n_neighbors=1).fit(x[core])
        self.threshold = self.eps  # noise: no core sample within eps
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Distance to the nearest core sample; at or beyond eps the point is noise."""
        distance, _ = self.index.kneighbors(model_inputs(frame))
        return distance[:, 0]

    def cluster(self, frame: pd.DataFrame) -> np.ndarray:
        """The nearest core sample's cluster, or -1 for noise."""
        distance, nearest = self.index.kneighbors(model_inputs(frame))
        return np.where(distance[:, 0] < self.eps, self.core_labels[nearest[:, 0]], -1)


def silhouette(x: np.ndarray, labels: np.ndarray, seed: int) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    n = min(SETTINGS["clustering"]["silhouetteSample"], len(x))
    return float(silhouette_score(x, labels, sample_size=n, random_state=seed))


def knee_distance(x: np.ndarray, k: int) -> float:
    """eps0: the sorted k-NN distance at the point farthest from the chord joining the curve's ends."""
    distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)  # +1: each point is its own nearest
    curve = np.sort(distances[:, -1])
    i = np.arange(len(curve), dtype="float64")
    start, end = np.array([0.0, curve[0]]), np.array([i[-1], curve[-1]])
    chord = (end - start) / np.linalg.norm(end - start)
    points = np.stack([i, curve], axis=1) - start
    gap = np.abs(points[:, 0] * chord[1] - points[:, 1] * chord[0])
    return float(curve[int(np.argmax(gap))])


def describe_centroid(centroid: np.ndarray) -> str:
    """A plain-words name for a K-Means centroid from its mean z-scores and direction mix."""
    c = SETTINGS["clustering"]
    parts = []
    for fid, value in zip(FEATURE_IDS, centroid[:4]):
        if abs(value) >= c["describeAbsZ"]:
            parts.append(PHRASES[fid][0 if value > 0 else 1])
    d_in, d_out = centroid[4], centroid[5]
    direction = ("mostly incoming" if d_in >= c["describeDirectionShare"] else
                 "mostly outgoing" if d_out >= c["describeDirectionShare"] else "mixed directions")
    return (", ".join(parts) if parts else "typical for the address") + f"; {direction}"


def tune_kmeans(rows: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, int]:
    x = model_inputs(rows)
    table = []
    for k in SETTINGS["clustering"]["kmeansK"]:
        model = KMeans(n_clusters=k, n_init=SETTINGS["clustering"]["kmeansNInit"], random_state=seed).fit(x)
        table.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette(x, model.labels_, seed),
                      "davies_bouldin": davies_bouldin_score(x, model.labels_),
                      "calinski_harabasz": calinski_harabasz_score(x, model.labels_)})
        log.info("K-Means k=%d: silhouette %.3f, Davies-Bouldin %.3f, inertia %.0f", k, table[-1]["silhouette"],
                 table[-1]["davies_bouldin"], model.inertia_)
    table = pd.DataFrame(table)
    best_k = int(table.loc[table["silhouette"].idxmax(), "k"])
    agrees = best_k == int(table.loc[table["davies_bouldin"].idxmin(), "k"])
    table["chosen"] = table["k"] == best_k
    table["davies_bouldin_agrees"] = agrees
    return table, best_k


def tune_dbscan(rows: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    c = SETTINGS["clustering"]
    x = model_inputs(rows)
    min_pts0 = c["dbscanMinPtsPerDimension"] * x.shape[1]
    eps0 = knee_distance(x, min_pts0)
    lo, hi = c["dbscanNoiseShare"]
    table = []
    for factor in c["dbscanEpsFactors"]:
        for min_pts in c["dbscanMinPts"]:
            labels = DBSCAN(eps=factor * eps0, min_samples=min_pts).fit(x).labels_
            noise = float(np.mean(labels == -1))
            kept = labels != -1
            table.append({"eps_factor": factor, "eps": factor * eps0, "min_samples": min_pts, "noise_share": noise,
                          "clusters": len(set(labels[kept])), "silhouette_non_noise": silhouette(x[kept], labels[kept], seed),
                          "in_noise_band": lo <= noise <= hi})
    table = pd.DataFrame(table)
    band = table[table["in_noise_band"]]
    pool, rule = (band, f"best silhouette among settings with {lo:.0%}-{hi:.0%} noise") if len(band) else (
        table.assign(off=(table["noise_share"] - (lo + hi) / 2).abs()), "no setting in the noise band; closest noise share")
    best = (pool.sort_values("silhouette_non_noise", ascending=False) if len(band) else pool.sort_values("off")).iloc[0]
    table["chosen"] = (table["eps"] == best["eps"]) & (table["min_samples"] == best["min_samples"])
    log.info("DBSCAN: eps0 %.3f (knee, MinPts %d); chose eps %.3f, MinPts %d, noise %.3f (%s)", eps0, min_pts0,
             best["eps"], best["min_samples"], best["noise_share"], rule)
    return table, {"eps": float(best["eps"]), "min_samples": int(best["min_samples"]), "eps0": eps0, "rule": rule}


def run_clustering() -> int:
    seed = SETTINGS["model"]["seed"]
    features = pd.read_csv(MANIFEST_PATH.parent / FEATURE_FILES["refined"])
    rows = fit_rows(features)
    log.info("Fit period: %d rows, inputs %s", len(rows), INPUT_COLUMNS)

    km_table, k = tune_kmeans(rows, seed)
    km = KMeansMethod(k, seed).fit(rows)
    labels = km.cluster(rows)
    clusters = pd.DataFrame(km.model.cluster_centers_, columns=INPUT_COLUMNS)
    clusters.insert(0, "cluster", range(k))
    clusters["fit_rows"] = np.bincount(labels, minlength=k)
    clusters["description"] = [describe_centroid(c) for c in km.model.cluster_centers_]

    db_table, db_choice = tune_dbscan(rows, seed)
    db = DBSCANMethod(db_choice["eps"], db_choice["min_samples"]).fit(rows)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    km_table.to_csv(TABLES_DIR / "tuning_kmeans.csv", index=False)
    clusters.to_csv(TABLES_DIR / "kmeans_clusters.csv", index=False)
    db_table.to_csv(TABLES_DIR / "tuning_dbscan.csv", index=False)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"method": km, "k": k, "threshold": km.threshold, "inputs": "methods.model_inputs",
                 "descriptions": clusters["description"].tolist()}, MODELS_DIR / "kmeans.joblib", compress=3)
    joblib.dump({"method": db, **db_choice, "threshold": db.threshold, "inputs": "methods.model_inputs"},
                MODELS_DIR / "dbscan.joblib", compress=3)
    for row in clusters.itertuples():
        log.info("Cluster %d (%d fit rows): %s", row.cluster, row.fit_rows, row.description)
    log.info("Wrote tuning_kmeans.csv, kmeans_clusters.csv, tuning_dbscan.csv and models/{kmeans,dbscan}.joblib")
    return 0
