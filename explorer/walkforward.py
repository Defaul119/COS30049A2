"""Does the DS1 ranking depend on where the time split falls? Every method at several fit/evaluation cut-offs.

Serves plan §6.3 (the time split, DR-02) and §11 (evaluation). The reported results use one cut-off per
address, at the fitFraction (70%) point of its scorable history, and CR-03 found addresses whose behaviour
changed after it. Here the split moves (settings.json -> walkForward.fitFractions): for each cut-off every
fitted method is refitted on the new fit period with its SAVED configuration (no re-tuning: the question is
whether the chosen methods hold up, not which grid point wins each fold), thresholds are recomputed by each
method's own rule, the reporting seeds plant into the new evaluation period, and the same checks run.
Time safety is unchanged: features never look past a transaction, and every fit uses only rows before the
cut-off.

At the reported cut-off the plants are the reporting plants and the refits reproduce the saved models, so
that row repeats eval_injection.csv (except the conformal range, whose refit keeps its saved margin).

Writes reports/tables/eval_walkforward.csv.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .baseline import TABLES_DIR
from .evaluate import FIXED, refit
from .features import FeatureConfig
from .inject import copy_features, make_plants
from .methods import fit_rows, pattern_recalls, plant_metrics, real_metrics
from .predict import PRECOMPUTED_ONLY
from .scores import saved_methods
from .settings import SETTINGS
from .snapshot import MANIFEST_PATH
from .split import assign_periods

log = logging.getLogger(__name__)


def run_walkforward(n_jobs: int = SETTINGS["classification"]["nJobs"]) -> int:
    transactions = pd.read_csv(MANIFEST_PATH.parent / "transactions.csv")
    features = pd.read_csv(MANIFEST_PATH.parent / "features.csv")
    saved = {n: m for n, m in saved_methods().items() if n not in PRECOMPUTED_ONLY}
    cfg, seed = FeatureConfig.refined(), SETTINGS["model"]["seed"]
    rows = []
    for fraction in SETTINGS["walkForward"]["fitFractions"]:
        period, _ = assign_periods(features, fraction)
        fold = features.assign(period=period)
        fit = fit_rows(fold)
        methods = {n: m if n in FIXED else refit(n, m, fit, seed) for n, m in saved.items()}
        periods = fold[["address", "txid", "period"]]
        plants = [pl for s in SETTINGS["injection"]["reportingSeeds"]
                  for pl in make_plants(transactions, periods, s, "reporting")]
        copies = pd.concat(Parallel(n_jobs=n_jobs)(delayed(copy_features)(transactions, periods, pl, cfg) for pl in plants),
                           ignore_index=True)
        for name, method in methods.items():
            row = {"fit_fraction": fraction, "method": name, "fit_rows": len(fit), **plant_metrics(method, copies),
                   **real_metrics(method, fold)}
            row["mean_recall"] = float(np.mean(pattern_recalls(row)))
            rows.append(row)
        part = pd.DataFrame(rows[-len(methods):]).sort_values("mean_recall", ascending=False)
        log.info("fit fraction %.2f: %d fit rows, %d evaluation rows, %d plants; %s", fraction, len(fit),
                 int((period == "eval").sum()), len(plants),
                 ", ".join(f"{r.method} {r.mean_recall:.2f}" for r in part.itertuples()))
    table = pd.DataFrame(rows)
    table["rank"] = table.groupby("fit_fraction")["mean_recall"].rank(ascending=False, method="min").astype(int)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "eval_walkforward.csv", index=False)
    log.info("Wrote %s", TABLES_DIR / "eval_walkforward.csv")
    return 0
