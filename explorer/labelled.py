"""Processing of the labelled datasets: DS4 (BitcoinHeist) and DS3 (Elliptic).

DS4: labels, transforms and the three split schemes.

Serves plan §6.4 (docs/plan/sections/s06-processing.tex) and report criterion 1. Every decision the
plan leaves to "the analysis" is made here by a written rule and recorded with the numbers behind
it in data/processed/bitcoinheist/processing.json:

* label: `white` is 0 and any ransomware family is 1; the family name is kept for the
  leave-one-family-out test. "Ransomware" is the dataset's own class name, not a judgement of any
  address in DS1 or DS2.
* features: length, weight, count, looped, neighbors, income. `address` is an identifier, and
  `year` and `day` would let a model learn when each family was active instead of what the
  activity looked like (a leakage shortcut), so none of them is a feature; year only defines splits.
* transforms: income becomes log10(income); weight, count, looped and neighbors become log(1 + x)
  when their skewness exceeds the setting (the plan's "if the analysis confirms heavy tails").
* temporal split: train on years up to Y and test on later years, Y being the latest year that
  leaves at least `minTestPositives` positives in the test set. Validation is the last training
  year, at the real class ratio.
* random split: stratified, `randomTestFraction` held out for test and `randomValidationFraction`
  of the rest for validation.
* leave one family out: for each of the `leaveOutFamilies` largest families.
* training sampling: all positives plus negatives downsampled to `negativesPerPositive`:1 (seeded),
  in fit and training sets only. Validation and test keep the real class ratio, because precision
  depends on the base rate.

Reads data/external/bitcoinheist/BitcoinHeistData.csv; writes data/processed/bitcoinheist/rows.parquet
(git-ignored, rebuilt by this step), data/processed/bitcoinheist/processing.json and
reports/tables/heist_{years,families,skew,splits}.csv.

DS3 (process_elliptic): the features-classes join, in/out degree from the edge list, the temporal
split, and the LF/AF feature sets. Elliptic is CC BY-NC-ND 4.0, so its processed rows stay on this
machine (data/processed/elliptic/ is git-ignored apart from download.json); only counts are written
to reports/tables/elliptic_{steps,summary}.csv.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .external import EXTERNAL_DIR, PROCESSED_DIR
from .settings import PROJECT_ROOT, SETTINGS

log = logging.getLogger(__name__)

HEIST_CSV = EXTERNAL_DIR / "bitcoinheist" / "BitcoinHeistData.csv"
HEIST_DIR = PROCESSED_DIR / "bitcoinheist"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
NEGATIVE_LABEL = "white"

# Values of the per-scheme split columns.
FIT, VAL, TEST, UNUSED = "fit", "val", "test", "unused"


def transform_features(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The model inputs, with the transforms the skewness rule selects. Returns (X, decisions)."""
    skew = df[params["features"]].skew()
    x = pd.DataFrame(index=df.index)
    applied = {}
    for col in params["features"]:
        if col == "income":
            # Income is in satoshis and spans many orders of magnitude; it is always positive.
            x["log10_income"] = np.log10(df[col])
            applied[col] = "log10"
        elif col in params["log1pCandidates"] and skew[col] > params["log1pIfSkewAbove"]:
            x[f"log1p_{col}"] = np.log1p(df[col])
            applied[col] = "log1p"
        else:
            x[col] = df[col].astype("float64")
            applied[col] = "none"
    return x.astype("float32"), {"skewness": skew.round(2).to_dict(), "transform": applied}


def downsample(index: pd.Index, y: pd.Series, ratio: int, rng: np.random.Generator) -> pd.Index:
    """All positives in `index` plus at most `ratio` negatives per positive, drawn without replacement."""
    pos = index[y.loc[index].to_numpy() == 1]
    neg = index[y.loc[index].to_numpy() == 0]
    keep = rng.choice(neg.to_numpy(), size=min(len(neg), ratio * len(pos)), replace=False)
    return pos.append(pd.Index(keep)).sort_values()


def temporal_year(positives_per_year: pd.Series, min_test_positives: int) -> int:
    """Y: the latest year that leaves at least `min_test_positives` positives in later years."""
    years = sorted(positives_per_year.index)
    candidates = [y for y in years[:-1] if positives_per_year.loc[positives_per_year.index > y].sum() >= min_test_positives]
    if not candidates:
        raise ValueError("No year leaves enough positives for a temporal test set")
    return max(candidates)


def stratified_holdout(index: pd.Index, y: pd.Series, fraction: float, rng: np.random.Generator) -> pd.Index:
    """A random `fraction` of `index`, drawn separately within each class so the ratio is kept."""
    held = []
    for cls in (0, 1):
        members = index[y.loc[index].to_numpy() == cls].to_numpy()
        held.append(rng.choice(members, size=round(fraction * len(members)), replace=False))
    return pd.Index(np.concatenate(held)).sort_values()


def process_heist(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Label, transform and split BitcoinHeist. Returns the processed rows and the decisions record.

    Columns of the result: the transformed features; `y`; `family` (the label, or "white");
    `year`; `temporal` and `random` (fit, val, test or unused); `temporal_sampled` and
    `random_sampled` (row is in that scheme's downsampled fit set); `temporal_train_sampled` and
    `random_train_sampled` (in the downsampled fit + val set used to refit the final model).
    """
    rng = np.random.default_rng(params["seed"])
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    y = (df["label"] != NEGATIVE_LABEL).astype("int8")
    x, decisions = transform_features(df, params)
    out = x.assign(y=y, family=df["label"].astype("category"), year=df["year"].astype("int16"))

    per_year = out.groupby("year")["y"].sum()
    year_y = temporal_year(per_year, params["minTestPositives"])
    temporal = np.where(out["year"] > year_y, TEST, np.where(out["year"] == year_y, VAL, FIT))
    out["temporal"] = pd.Categorical(temporal, categories=[FIT, VAL, TEST, UNUSED])
    fit_idx = out.index[out["temporal"] == FIT]
    train_idx = out.index[out["temporal"].isin([FIT, VAL])]
    out["temporal_sampled"] = out.index.isin(downsample(fit_idx, out["y"], params["negativesPerPositive"], rng))
    out["temporal_train_sampled"] = out.index.isin(downsample(train_idx, out["y"], params["negativesPerPositive"], rng))

    test_idx = stratified_holdout(out.index, out["y"], params["randomTestFraction"], rng)
    rest = out.index.difference(test_idx)
    val_idx = stratified_holdout(rest, out["y"], params["randomValidationFraction"], rng)
    random_split = pd.Series(FIT, index=out.index)
    random_split.loc[test_idx] = TEST
    random_split.loc[val_idx] = VAL
    out["random"] = pd.Categorical(random_split, categories=[FIT, VAL, TEST, UNUSED])
    out["random_sampled"] = out.index.isin(
        downsample(out.index[out["random"] == FIT], out["y"], params["negativesPerPositive"], rng))
    out["random_train_sampled"] = out.index.isin(
        downsample(out.index[out["random"] != TEST], out["y"], params["negativesPerPositive"], rng))

    families = out.loc[out["y"] == 1, "family"].value_counts()
    decisions.update({
        "rows_read": before, "exact_duplicates_removed": before - len(df), "rows": len(out),
        "positives": int(out["y"].sum()), "positive_share": float(out["y"].mean()),
        "temporal_last_training_year": year_y,
        "temporal_rule": f"latest year leaving at least {params['minTestPositives']} positives in later years",
        "leave_out_families": families.index[: params["leaveOutFamilies"]].tolist(),
        "features_excluded": ["address (identifier)", "year (defines splits only)", "day (defines splits only)"],
    })
    return out, decisions


def split_sizes(out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme in ("temporal", "random"):
        for part in (FIT, VAL, TEST):
            sel = out[out[scheme] == part]
            rows.append({"scheme": scheme, "part": part, "rows": len(sel), "positives": int(sel["y"].sum()),
                         "positive_share": round(float(sel["y"].mean()), 5) if len(sel) else None})
        for name, col in (("fit, downsampled", f"{scheme}_sampled"), ("fit + val, downsampled", f"{scheme}_train_sampled")):
            sel = out[out[col]]
            rows.append({"scheme": scheme, "part": name, "rows": len(sel), "positives": int(sel["y"].sum()),
                         "positive_share": round(float(sel["y"].mean()), 5)})
    return pd.DataFrame(rows)


def run_process_heist(csv_path: Path = HEIST_CSV, out_dir: Path = HEIST_DIR, tables_dir: Path = TABLES_DIR) -> int:
    """Step `process-external` for DS4: clean, transform and split BitcoinHeist; writes rows.parquet and the
    processing decisions (processing.json)."""
    params = SETTINGS["bitcoinheist"]
    df = pd.read_csv(csv_path)
    log.info("Read %s: %d rows", csv_path, len(df))
    out, decisions = process_heist(df, params)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "rows.parquet", index=False)
    (out_dir / "processing.json").write_text(json.dumps(decisions, indent=2) + "\n", encoding="utf-8")

    tables_dir.mkdir(parents=True, exist_ok=True)
    years = out.groupby("year")["y"].agg(rows="size", positives="sum").reset_index()
    years.to_csv(tables_dir / "heist_years.csv", index=False)
    fam = out[out["y"] == 1].groupby("family", observed=True)["year"].agg(rows="size", first_year="min", last_year="max")
    fam.sort_values("rows", ascending=False).reset_index().to_csv(tables_dir / "heist_families.csv", index=False)
    pd.DataFrame({"feature": list(decisions["skewness"]), "skewness": list(decisions["skewness"].values()),
                  "transform": [decisions["transform"][f] for f in decisions["skewness"]]}).to_csv(
        tables_dir / "heist_skew.csv", index=False)
    split_sizes(out).to_csv(tables_dir / "heist_splits.csv", index=False)
    log.info("Temporal split: train to %d, test after; leave-out families %s",
             decisions["temporal_last_training_year"], decisions["leave_out_families"])
    log.info("Transforms: %s", decisions["transform"])
    log.info("Wrote %s, processing.json and reports/tables/heist_*.csv", out_dir / "rows.parquet")
    return 0


# -- DS3 Elliptic -------------------------------------------------------------------------

ELLIPTIC_RAW = EXTERNAL_DIR / "elliptic"
ELLIPTIC_DIR = PROCESSED_DIR / "elliptic"
LOCAL = [f"local_{i:02d}" for i in range(1, 94)]
AGG = [f"agg_{i:02d}" for i in range(1, 73)]
DEGREES = ["in_degree", "out_degree"]
FEATURE_SETS = {"LF": LOCAL + DEGREES, "AF": LOCAL + AGG + DEGREES}


def process_elliptic(features: pd.DataFrame, classes: pd.DataFrame, edges: pd.DataFrame,
                     params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join, label and split Elliptic (plan §6.3). Returns the rows and the decisions record.

    * classes: 1 (illicit) -> 1, 2 (licit) -> 0, unknown -> missing; unlabelled rows are kept for
      the label-free checks but never trained or tested on.
    * the edge list adds in- and out-degree per transaction: a genuine dataset join.
    * time_step is never a feature; it only defines the split: tuning fits on steps up to
      fitLastStep and validates up to trainLastStep; the final model trains on steps up to
      trainLastStep and is tested once on the later steps (Weber et al. 2019).
    """
    out = features.merge(classes, on="txId", how="left", validate="one_to_one")
    out["y"] = out["class"].map({"1": 1.0, "2": 0.0, 1: 1.0, 2: 0.0})
    in_deg = edges.groupby("txId2").size()
    out_deg = edges.groupby("txId1").size()
    out["in_degree"] = out["txId"].map(in_deg).fillna(0).astype("float32")
    out["out_degree"] = out["txId"].map(out_deg).fillna(0).astype("float32")
    fit_last, train_last = params["fitLastStep"], params["trainLastStep"]
    out["period"] = np.where(out["time_step"] <= fit_last, FIT,
                             np.where(out["time_step"] <= train_last, VAL, TEST))
    labelled = out["y"].notna()
    by_step = out.groupby("time_step")["class"].value_counts().unstack(fill_value=0)
    decisions = {
        "rows": len(out), "edges": len(edges), "labelled": int(labelled.sum()),
        "illicit": int((out["y"] == 1).sum()), "licit": int((out["y"] == 0).sum()),
        "unknown": int((~labelled).sum()), "classes_missing_after_join": int(out["class"].isna().sum()),
        "missing_feature_values": int(out[LOCAL + AGG].isna().sum().sum()),
        "time_steps": [int(out["time_step"].min()), int(out["time_step"].max())],
        "split": {"fit_steps": [1, fit_last], "val_steps": [fit_last + 1, train_last],
                  "test_steps": [train_last + 1, int(out["time_step"].max())]},
        "feature_sets": {k: len(v) for k, v in FEATURE_SETS.items()},
        "constant_columns_handling": "dropped inside each model Pipeline (VarianceThreshold), so only training rows decide",
        "licence_note": SETTINGS["external"]["elliptic"]["_licence"],
    }
    for part in (FIT, VAL, TEST):
        sel = out[(out["period"] == part) & labelled]
        decisions[f"{part}_labelled"] = len(sel)
        decisions[f"{part}_illicit_share"] = round(float(sel["y"].mean()), 4)
    return out, decisions | {"by_step": by_step.reset_index().to_dict("list")}


def run_process_elliptic(raw: Path = ELLIPTIC_RAW, out_dir: Path = ELLIPTIC_DIR, tables_dir: Path = TABLES_DIR) -> int:
    """Step `process-external` for DS3: join Elliptic's features, classes and edges into one row per transaction
    with the temporal split; writes rows.parquet and processing.json."""
    names = ["txId", "time_step"] + LOCAL + AGG
    features = pd.read_csv(raw / "elliptic_txs_features.csv", header=None, names=names,
                           dtype={c: "float32" for c in LOCAL + AGG})
    classes = pd.read_csv(raw / "elliptic_txs_classes.csv", dtype={"class": str})
    edges = pd.read_csv(raw / "elliptic_txs_edgelist.csv")
    log.info("Read Elliptic: %d transactions, %d classes, %d edges", len(features), len(classes), len(edges))
    out, decisions = process_elliptic(features, classes, edges, SETTINGS["elliptic"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "rows.parquet", index=False)  # local only: the licence forbids sharing derivatives
    by_step = pd.DataFrame(decisions.pop("by_step")).rename(columns={"1": "illicit", "2": "licit"})
    (out_dir / "processing.json").write_text(json.dumps(decisions, indent=2) + "\n", encoding="utf-8")
    tables_dir.mkdir(parents=True, exist_ok=True)
    by_step.to_csv(tables_dir / "elliptic_steps.csv", index=False)  # counts only, not data rows
    pd.DataFrame([{k: v for k, v in decisions.items() if not isinstance(v, (dict, list, str))}]).to_csv(
        tables_dir / "elliptic_summary.csv", index=False)
    log.info("Elliptic: %d labelled (%d illicit); fit %d, val %d, test %d labelled rows", decisions["labelled"],
             decisions["illicit"], decisions["fit_labelled"], decisions["val_labelled"], decisions["test_labelled"])
    return 0
