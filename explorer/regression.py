"""Regression on DS1: each transaction's expected size and gap, and how far it lands from them.

Serves plan §9.5 (docs/plan/sections/s08-models.tex) and the brief's regression model type. The
targets are y_size = log10(1 + |net| sats) (= F1) and y_gap = log10(1 + hours since the previous
transaction). The inputs, all from strictly earlier transactions or the transaction's own time and
direction (DR-02), are the window medians and spreads of size and gap, the burst count, the
30-day count, the previous transaction's size and direction, the address's age and in-share, and
the hour and weekday. The current amount is never an input for size, nor the current gap for gap.

Models: a naive model that predicts the window median (exactly the robust-z centre, so beating it
shows what learning adds), Ridge, random forest and histogram gradient boosting, plus quantile
gradient boosting for a 90% expected range and a 98% range. Each target is tuned on the last part
of every address's fit period (time order kept) by MAE, then refitted on the whole fit period.
Metrics on the evaluation period: MAE, RMSE and R^2 on the log scale, skill against naive
(1 - MAE / MAE_naive), and the 90% range's coverage and mean width.

Unusualness: the standardised residual r = (y - y_hat) / s, with s = 1.4826 x the MAD of that
address's fit-period residuals (floored); score = max(|r_size|, |r_gap|), flagged at flagAbsR, the
baseline's 3.5. A second variant flags anything outside the 98% quantile range.

Writes reports/tables/tuning_regression.csv, regression_results.csv and models/regression.joblib
(the residual method) and models/regression_interval.joblib (the range variant).
"""
from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baseline import FEATURE_FILES, MODELS_DIR, TABLES_DIR
from .methods import fit_rows
from .settings import SETTINGS
from .snapshot import MANIFEST_PATH
from .split import EVAL

log = logging.getLogger(__name__)

INPUTS = [
    "dir_in", "dir_out", "center_F1", "spread_F1", "center_F3", "spread_F3", "F2", "timing_window_count_in_period",
    "prev_log_size", "prev_dir_in", "prev_dir_out", "age_log_hours", "in_share_prior", "F8_sin", "F8_cos",
    "weekday_sin", "weekday_cos",
]
TARGETS = {"size": "center_F1", "gap": "center_F3"}  # target -> its naive prediction (the window median)


def design(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["dir_in"] = (x["direction"] == "in").astype(float)
    x["dir_out"] = (x["direction"] == "out").astype(float)
    return x[INPUTS].astype("float64")


def targets(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """y_size and y_gap; the gap is missing for an address's first transaction."""
    return {"size": frame["F1"].astype("float64"), "gap": np.log10(1 + frame["F3"].astype("float64"))}


def make_model(kind: str, params: dict[str, Any], seed: int) -> Pipeline:
    """A Pipeline: missing window statistics are imputed from training rows; Ridge also gets a scaler."""
    impute = ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    if kind == "ridge":
        return Pipeline([impute, ("scale", StandardScaler()), ("model", Ridge(**params))])
    if kind == "randomForest":
        return Pipeline([impute, ("model", RandomForestRegressor(**params, random_state=seed, n_jobs=SETTINGS["classification"]["nJobs"]))])
    if kind == "histGradientBoosting":
        return Pipeline([impute, ("model", HistGradientBoostingRegressor(**params, random_state=seed))])
    raise ValueError(kind)


def grids() -> dict[str, list[dict[str, Any]]]:
    r = SETTINGS["regression"]
    return {"ridge": [{"alpha": a} for a in r["ridgeAlpha"]],
            "randomForest": list(ParameterGrid(r["randomForest"])),
            "histGradientBoosting": list(ParameterGrid(r["histGradientBoosting"]))}


def time_ordered_validation(rows: pd.DataFrame, fraction: float) -> pd.Series:
    """True for the last `fraction` of each address's rows in time order (the tuning validation set)."""
    rank = rows.groupby("address")["block_time_utc"].rank(method="first", pct=True)
    return rank > 1 - fraction


def naive(frame: pd.DataFrame, target: str, fallback: float) -> np.ndarray:
    return frame[TARGETS[target]].astype("float64").fillna(fallback).to_numpy()


class RegressionMethod:
    """Score = max |standardised residual| over size and gap; flagged at flagAbsR."""

    def __init__(self, models: dict[str, Pipeline], scales: dict[str, dict[str, float]], default_scale: dict[str, float]):
        self.models, self.scales, self.default_scale = models, scales, default_scale
        self.threshold = SETTINGS["regression"]["flagAbsR"]

    def residuals(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        x, ys = design(frame), targets(frame)
        out = {}
        for t, model in self.models.items():
            s = frame["address"].map(self.scales[t]).fillna(self.default_scale[t]).to_numpy()
            out[t] = (ys[t].to_numpy() - model.predict(x)) / s
        return out

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        # A missing residual (no gap for an address's first transaction) counts as 0, not unusual.
        r = [np.nan_to_num(np.abs(v), nan=0.0) for v in self.residuals(frame).values()]
        return np.max(np.vstack(r), axis=0)


class RegressionIntervalMethod:
    """Score = how far y lies outside the 98% quantile range, in range widths; flagged above 0."""

    threshold = 0.0

    def __init__(self, quantiles: dict[str, tuple[Pipeline, Pipeline]]):
        self.quantiles = quantiles

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        x, ys = design(frame), targets(frame)
        worst = np.full(len(frame), -np.inf)
        for t, (lo_model, hi_model) in self.quantiles.items():
            lo, hi = lo_model.predict(x), hi_model.predict(x)
            width = np.maximum(hi - lo, 1e-9)
            y = ys[t].to_numpy()
            outside = np.fmax((lo - y) / width, (y - hi) / width)
            worst = np.fmax(worst, np.nan_to_num(outside, nan=-np.inf))
        return worst


def residual_scales(model: Pipeline, rows: pd.DataFrame, y: pd.Series, floor: float) -> tuple[dict[str, float], float]:
    """Per-address 1.4826 x MAD of fit-period residuals (floored), and the median over addresses."""
    resid = pd.Series(y.to_numpy() - model.predict(design(rows)), index=rows.index)
    mad = resid.groupby(rows["address"]).apply(lambda r: float(np.median(np.abs(r - np.median(r)))))
    scales = (SETTINGS["baseline"]["madScale"] * mad).clip(lower=floor)
    return scales.to_dict(), float(scales.median())


def metrics(y: np.ndarray, pred: np.ndarray, naive_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y, pred)
    return {"mae": mae, "rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": r2_score(y, pred),
            "skill_vs_naive": 1 - mae / mean_absolute_error(y, naive_pred)}


def run_regression() -> int:
    r, seed = SETTINGS["regression"], SETTINGS["model"]["seed"]
    features = pd.read_csv(MANIFEST_PATH.parent / FEATURE_FILES["refined"])
    fit_all = fit_rows(features)
    ev = features[(features["period"] == EVAL) & features["scorable"]]
    tuning_rows, results, final, quantiles, scales, default_scale = [], [], {}, {}, {}, {}

    for t in TARGETS:
        y_fit = targets(fit_all)[t]
        rows, y = fit_all[y_fit.notna()], y_fit[y_fit.notna()]
        is_val = time_ordered_validation(rows, r["validationFraction"])
        fallback = float(y[~is_val].median())
        tr, va = rows[~is_val], rows[is_val]
        naive_val = naive(va, t, fallback)
        best: dict[str, Any] = {}
        for kind, configs in grids().items():
            for params in configs:
                model = make_model(kind, params, seed).fit(design(tr), y[~is_val])
                mae = mean_absolute_error(y[is_val], model.predict(design(va)))
                tuning_rows.append({"target": t, "model": kind, "params": json.dumps(params), "val_mae": mae,
                                    "val_naive_mae": mean_absolute_error(y[is_val], naive_val)})
                if kind not in best or mae < best[kind][0]:
                    best[kind] = (mae, params)
            log.info("%s / %s: best validation MAE %.4f with %s (naive %.4f)", t, kind, best[kind][0], best[kind][1],
                     mean_absolute_error(y[is_val], naive_val))

        # Refit every model family's best configuration on the whole fit period; evaluate once.
        y_ev_all = targets(ev)[t]
        ev_t, y_ev = ev[y_ev_all.notna()], y_ev_all[y_ev_all.notna()].to_numpy()
        naive_ev = naive(ev_t, t, float(y.median()))
        results.append({"target": t, "model": "naive", "params": "{}", **metrics(y_ev, naive_ev, naive_ev)})
        fitted = {}
        for kind, (_, params) in best.items():
            fitted[kind] = make_model(kind, params, seed).fit(design(rows), y)
            results.append({"target": t, "model": kind, "params": json.dumps(params),
                            **metrics(y_ev, fitted[kind].predict(design(ev_t)), naive_ev)})
        chosen = min(best, key=lambda k: best[k][0])
        final[t] = fitted[chosen]
        scales[t], default_scale[t] = residual_scales(final[t], rows, y, r["residualFloor"])

        hgb = best["histGradientBoosting"][1]
        q = {}
        for qq in r["intervalQuantiles"] + r["flagQuantiles"]:
            q[qq] = make_model("histGradientBoosting", {**hgb, "loss": "quantile", "quantile": qq}, seed).fit(design(rows), y)
        lo, hi = (q[x].predict(design(ev_t)) for x in r["intervalQuantiles"])
        results.append({"target": t, "model": f"quantile {r['intervalQuantiles']}", "params": json.dumps(hgb),
                        "coverage": float(np.mean((y_ev >= lo) & (y_ev <= hi))), "mean_width": float(np.mean(hi - lo))})
        quantiles[t] = (q[r["flagQuantiles"][0]], q[r["flagQuantiles"][1]])
        log.info("%s: chose %s (validation MAE %.4f); 90%% range coverage %.3f", t, chosen, best[chosen][0],
                 results[-1]["coverage"])

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tuning_rows).to_csv(TABLES_DIR / "tuning_regression.csv", index=False)
    pd.DataFrame(results).to_csv(TABLES_DIR / "regression_results.csv", index=False)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    method = RegressionMethod(final, scales, default_scale)
    joblib.dump({"method": method, "threshold": method.threshold, "inputs": INPUTS,
                 "chosen": {t: type(m.named_steps["model"]).__name__ for t, m in final.items()}},
                MODELS_DIR / "regression.joblib", compress=3)
    interval = RegressionIntervalMethod(quantiles)
    joblib.dump({"method": interval, "threshold": interval.threshold, "inputs": INPUTS,
                 "quantiles": r["flagQuantiles"]}, MODELS_DIR / "regression_interval.joblib", compress=3)
    log.info("Wrote tuning_regression.csv, regression_results.csv, models/regression.joblib and regression_interval.joblib")
    return 0
