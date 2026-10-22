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
from .snapshot import MANIFEST_PATH

log = logging.getLogger(__name__)

SCORES_PATH = MANIFEST_PATH.parent / "scores.csv"


def saved_methods(models_dir: Path = MODELS_DIR) -> dict[str, Method]:
    """The final DS1 methods that exist so far, by name."""
    methods: dict[str, Method] = {}
    if (models_dir / "baseline.json").exists():
        chosen = json.loads((models_dir / "baseline.json").read_text(encoding="utf-8"))["chosen"]["refined"]
        methods["baseline"] = Baseline(chosen["threshold"])
    for name in ("isolation_forest", "lof", "kmeans", "dbscan", "regression"):
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
