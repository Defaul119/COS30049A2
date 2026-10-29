"""Every method's score and flag for every DS1 transaction (A1 Objective 2; milestone M1).

Serves plan §12.4: data/processed/esplora/scores.csv is what predict.py and the A3 back end read.
Each method contributes `score_<method>` and `flag_<method>` columns; unscorable transactions keep
their scores for display but are never flagged. Methods whose saved model does not exist yet are
skipped, so this step can run as each method is finished.

Reads data/processed/esplora/features.csv and models/; writes data/processed/esplora/scores.csv.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from .baseline import MODELS_DIR, Baseline
from .methods import Method, flags
from .settings import SETTINGS
from .snapshot import MANIFEST_PATH

log = logging.getLogger(__name__)

SCORES_PATH = MANIFEST_PATH.parent / "scores.csv"


def stage_models() -> dict[str, str]:
    """Each method's last accepted training stage (settings.json -> stageModels; docs/TRAINING_PLAN.md)."""
    return {k: v for k, v in SETTINGS["stageModels"].items() if not k.startswith("_")}


def saved_methods(models_dir: Path = MODELS_DIR) -> dict[str, Method]:
    """The final DS1 methods that exist so far, by name: a method with an accepted training stage is that stage's
    winner (models/stage/<stage>_<method>.joblib), every other its M0 model (models/<method>.joblib)."""
    methods: dict[str, Method] = {}
    if (models_dir / "baseline.json").exists():
        chosen = json.loads((models_dir / "baseline.json").read_text(encoding="utf-8"))["chosen"]["refined"]
        methods["baseline"] = Baseline(chosen["threshold"])
    accepted = stage_models()
    for name in ("isolation_forest", "lof", "kmeans", "dbscan", "regression", "regression_interval", "fusion", "txgpt_method"):
        if name in accepted:
            path = models_dir / "stage" / f"{accepted[name]}_{name}.joblib"
            if not path.exists():  # an accepted stage without its winner would silently fall back to M0
                raise FileNotFoundError(f"{path} is missing: stageModels names stage {accepted[name]} for {name}")
        else:
            path = models_dir / f"{name}.joblib"
        if path.exists():
            methods[name] = joblib.load(path)["method"]
    return methods


def run_scores(features_path: Path = MANIFEST_PATH.parent / "features.csv", out_path: Path = SCORES_PATH) -> int:
    features = pd.read_csv(features_path)
    out = features[["address", "role", "txid", "block_time_utc", "direction", "net_sats", "period", "scorable"]].copy()
    for name, method in saved_methods().items():
        out[f"score_{name}"] = method.score(features)
        out[f"flag_{name}"] = flags(method, features)
        ev = out["period"] == "eval"
        log.info("%s: %d flagged of %d (%d of %d in the evaluation period); threshold %.4g", name,
                 out[f"flag_{name}"].sum(), len(out), out.loc[ev, f"flag_{name}"].sum(), ev.sum(), method.threshold)
    out.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)
    return 0
