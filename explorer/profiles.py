"""Address profiles: one row per address, clustered with K-Means; and the check that DS2 covers DS1.

Serves plan §9.4 ("Address profiles (DS1 + DS2)", priority item S-7) and the analysis question of
plan §7, "What kinds of addresses are in DS2?". TxGPT is trained on DS2 and scores DS1, so the report
must show that DS2's addresses behave like DS1's; the coverage table says, per feature, how many DS1
addresses fall inside DS2's central range.

Each address gets 8 features from its whole history (descriptive only, nothing is evaluated on them,
so the time split does not apply):

    tx_per_day        transactions per day of history (span floored at profiles.minSpanDays)
    in_share          share of its transactions that receive more than they send
    median_log_size   median log10(1 + |net amount| in sats)
    gap_cv            coefficient of variation of the gaps between consecutive transactions
    burstiness        (sd - mean) / (sd + mean) of the daily transaction counts over the span, the form of
                      Goh and Barabasi's index (which they apply to gaps): -1 the same count every day,
                      towards 1 activity packed into a few days (on daily counts, unlike gaps, a Poisson
                      process gives 0 only at about one transaction a day)
    mean_outputs      mean outputs per transaction
    history_tx        transactions in the history
    median_fee_rate   median fee rate of its transactions (sat/vB)

The features in profiles.logFeatures are compared on log10(1 + x), all are standardised inside a
scikit-learn Pipeline, and k is the best silhouette over profiles.k. DS1's short-history addresses are
left out: DS2 only keeps addresses with at least corpus.minTx transactions, so they would not compare.

Writes data/processed/profiles.csv (every address's features and profile; DS2 by its hashed ID),
reports/tables/profiles_tuning.csv, profiles.csv, ds2_coverage.csv and models/profiles.joblib, which
labels a new address's history with a profile (A3's optional "activity profile").
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .baseline import MODELS_DIR, TABLES_DIR
from .clustering import silhouette
from .corpus import OUT_DIR as CORPUS_DIR
from .settings import PROJECT_ROOT, SETTINGS
from .snapshot import MANIFEST_PATH

log = logging.getLogger(__name__)

PROFILES_PATH = PROJECT_ROOT / "data" / "processed" / "profiles.csv"
SECONDS_PER_DAY = 86_400
FEATURES = ["tx_per_day", "in_share", "median_log_size", "gap_cv", "burstiness", "mean_outputs", "history_tx",
            "median_fee_rate"]
WORDS = {  # how a profile's standardised centroid reads, by feature and sign
    "tx_per_day": ("many transactions per day", "few transactions per day"),
    "in_share": ("mostly receives", "mostly sends"),
    "median_log_size": ("large amounts", "small amounts"),
    "gap_cv": ("irregular gaps", "regular gaps"),
    "burstiness": ("bursty", "steady"),
    "mean_outputs": ("many outputs per transaction", "few outputs per transaction"),
    "history_tx": ("long history", "short history"),
    "median_fee_rate": ("high fee rates", "low fee rates"),
}


def address_features(tx: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """The 8 profile features for every address in a transaction table (DS1's or DS2's columns)."""
    min_span = SETTINGS["profiles"]["minSpanDays"]
    rows = []
    for address, g in tx.sort_values([id_column, "block_time_utc"]).groupby(id_column, sort=False):
        t = g["block_time_utc"].to_numpy(dtype="int64")
        net = (g["received_sats"] - g["sent_sats"]).to_numpy()
        gaps = np.diff(t).astype("float64")
        days = (t - t[0]) // SECONDS_PER_DAY
        daily = np.bincount(days)  # every day of the span, idle days included
        sd, mean = daily.std(), daily.mean()
        vsize = g["vsize_vb"].to_numpy()
        fee_rate = g["fee_sats"].to_numpy()[vsize > 0] / vsize[vsize > 0]
        rows.append({
            id_column: address,
            "tx_per_day": len(g) / max((t[-1] - t[0]) / SECONDS_PER_DAY, min_span),
            "in_share": float(np.mean(net > 0)),
            "median_log_size": float(np.median(np.log10(1 + np.abs(net)))),
            "gap_cv": float(gaps.std() / gaps.mean()) if len(gaps) and gaps.mean() > 0 else 0.0,
            "burstiness": float((sd - mean) / (sd + mean)),
            "mean_outputs": float(g["n_outputs"].mean()),
            "history_tx": len(g),
            "median_fee_rate": float(np.median(fee_rate)) if len(fee_rate) else 0.0,
        })
    return pd.DataFrame(rows)


def log_columns(x: pd.DataFrame) -> pd.DataFrame:
    """log10(1 + x) on the features that span orders of magnitude (profiles.logFeatures)."""
    x = x.copy()
    for name in SETTINGS["profiles"]["logFeatures"]:
        x[name] = np.log10(1 + x[name])
    return x


def pipeline(k: int, seed: int) -> Pipeline:
    return make_pipeline(FunctionTransformer(log_columns), StandardScaler(),
                         KMeans(n_clusters=k, n_init=SETTINGS["clustering"]["kmeansNInit"], random_state=seed))


def describe(centroid: np.ndarray) -> str:
    """A plain-words name for a profile: its most distinctive features, from its standardised centroid."""
    p = SETTINGS["profiles"]
    order = np.argsort(-np.abs(centroid))[: p["describeMax"]]
    parts = [WORDS[FEATURES[i]][0 if centroid[i] > 0 else 1] for i in order if abs(centroid[i]) >= p["describeAbsZ"]]
    return ", ".join(parts) if parts else "typical on every feature"


def coverage(ds1: pd.DataFrame, ds2: pd.DataFrame) -> pd.DataFrame:
    """Per feature: DS2's central range, DS1's spread, and the share of DS1 addresses inside DS2's range."""
    lo, hi = SETTINGS["profiles"]["coverageQuantiles"]
    rows = []
    for name in FEATURES:
        a, b = ds1[name], ds2[name]
        q_lo, q_hi = b.quantile([lo, hi])
        rows.append({"feature": name, "ds2_low": q_lo, "ds2_median": b.median(), "ds2_high": q_hi,
                     "ds1_min": a.min(), "ds1_median": a.median(), "ds1_max": a.max(),
                     "ds1_inside_share": float(a.between(q_lo, q_hi).mean()),
                     "ds1_median_percentile_in_ds2": float((b < a.median()).mean())})
    return pd.DataFrame(rows)


def run_profiles() -> int:
    seed = SETTINGS["model"]["seed"]
    ds1_tx = pd.read_csv(MANIFEST_PATH.parent / "transactions.csv")
    ds1_tx = ds1_tx[ds1_tx["role"] == "main"]
    ds1 = address_features(ds1_tx, "address").rename(columns={"address": "id"}).assign(dataset="DS1")
    ds2 = address_features(pd.read_parquet(CORPUS_DIR / "ds2.parquet"), "addr_id").rename(columns={"addr_id": "id"})
    ds2 = ds2.assign(dataset="DS2")
    frame = pd.concat([ds1, ds2], ignore_index=True)
    x = frame[FEATURES]
    log.info("Profiles: %d DS1 and %d DS2 addresses", len(ds1), len(ds2))

    tuning = []
    for k in SETTINGS["profiles"]["k"]:
        model = pipeline(k, seed).fit(x)
        z = model[:-1].transform(x)
        labels = model[-1].labels_
        tuning.append({"k": k, "silhouette": silhouette(z, labels, seed), "davies_bouldin": davies_bouldin_score(z, labels),
                       "inertia": model[-1].inertia_})
        log.info("Profiles k=%d: silhouette %.3f, Davies-Bouldin %.3f", k, tuning[-1]["silhouette"], tuning[-1]["davies_bouldin"])
    tuning = pd.DataFrame(tuning)
    k = int(tuning.loc[tuning["silhouette"].idxmax(), "k"])
    tuning["chosen"] = tuning["k"] == k
    model = pipeline(k, seed).fit(x)
    frame["profile"] = model.predict(x)

    centroids = model[-1].cluster_centers_
    profiles = pd.DataFrame({"profile": range(k), "description": [describe(c) for c in centroids]})
    for name in ("DS1", "DS2"):
        profiles[f"{name.lower()}_addresses"] = np.bincount(frame.loc[frame["dataset"] == name, "profile"], minlength=k)
    medians = frame.groupby("profile")[FEATURES].median().add_prefix("median_")
    profiles = profiles.join(medians, on="profile")
    profiles = profiles.join(pd.DataFrame(centroids, columns=[f"z_{f}" for f in FEATURES]))

    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROFILES_PATH, index=False)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(TABLES_DIR / "profiles_tuning.csv", index=False)
    profiles.to_csv(TABLES_DIR / "profiles.csv", index=False)
    coverage(ds1, ds2).to_csv(TABLES_DIR / "ds2_coverage.csv", index=False)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "descriptions": profiles["description"].tolist()},
                MODELS_DIR / "profiles.joblib", compress=3)
    for row in profiles.itertuples():
        log.info("Profile %d (%d DS1, %d DS2): %s", row.profile, row.ds1_addresses, row.ds2_addresses, row.description)
    log.info("Wrote %s, profiles_tuning.csv, profiles.csv, ds2_coverage.csv and models/profiles.joblib", PROFILES_PATH.name)
    return 0
