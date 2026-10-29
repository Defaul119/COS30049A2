"""The A1 robust-z baseline and its threshold sweep.

Serves plan §9.3 (docs/plan/sections/s08-models.tex) and A1 §2.3. A transaction is flagged when it is
scorable and max_k |z_k| >= the threshold; its score is that maximum, so the baseline also joins the
threshold-free ranking checks. The z-scores come from features.py (the port of the prototype).

The sweep A1 promised tries every threshold in settings.json -> baselineSweep on the TUNING plants
(seeds 40000-40004) and the real evaluation period, and records recall per pattern, the flag rate
and the addresses breaking the sanity rule. The chosen threshold is the smallest that meets both
the sanity rule and the recall target; if none does, the smallest that meets the sanity rule. A
choice other than A1's 3.5 is change request CR-03, which A1 reserved for exactly this: it is
recorded in models/baseline.json and CHANGELOG.md, and settings.json changes only when the team
approves it.

Writes reports/tables/tuning_baseline.csv and models/baseline.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .inject import INTERIM_DIR
from .methods import meets_recall_target, plant_metrics, real_metrics
from .settings import PROJECT_ROOT, SETTINGS
from .snapshot import MANIFEST_PATH

log = logging.getLogger(__name__)

TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
MODELS_DIR = PROJECT_ROOT / "models"
FEATURE_FILES = {"refined": "features.csv", "as_written": "features_as_written.csv"}


class Baseline:
    """Score = max |z| over F1-F4 for a scorable transaction, 0 otherwise (features.score_baseline)."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["score_baseline"].to_numpy(dtype="float64")


def load_copies(method: str, set_name: str) -> pd.DataFrame:
    """Every planted copy for one feature method and plant set (tuning or reporting), all seeds."""
    seeds = SETTINGS["injection"]["tuningSeeds" if set_name == "tuning" else "reportingSeeds"]
    return pd.concat([pd.read_parquet(INTERIM_DIR / method / f"{set_name}_{s}.parquet") for s in seeds], ignore_index=True)


def choose_threshold(rows: list[dict[str, Any]]) -> tuple[float, str]:
    """The sweep's rule: smallest threshold meeting sanity and recall; else smallest meeting sanity;
    else the one with the fewest addresses over the limit."""
    both = [r for r in rows if r["meets_sanity"] and r["meets_recall_target"]]
    if both:
        return min(r["threshold"] for r in both), "smallest threshold meeting the sanity rule and the recall target"
    sane = [r for r in rows if r["meets_sanity"]]
    if sane:
        return min(r["threshold"] for r in sane), "no threshold met the recall target; smallest meeting the sanity rule"
    best = min(rows, key=lambda r: (r["addresses_over_limit"], r["threshold"]))
    return best["threshold"], "no threshold met the sanity rule; fewest addresses over the limit"


def run_baseline_sweep(processed_dir: Path = MANIFEST_PATH.parent) -> int:
    """Step `baseline`: every threshold of settings baselineSweep on the tuning plants, for both feature variants;
    writes reports/tables/tuning_baseline.csv and the chosen threshold to models/baseline.json."""
    rows = []
    for method, filename in FEATURE_FILES.items():
        features = pd.read_csv(processed_dir / filename)
        copies = load_copies(method, "tuning")
        for threshold in SETTINGS["baselineSweep"]:
            b = Baseline(threshold)
            row = {"features": method, "threshold": threshold, **plant_metrics(b, copies), **real_metrics(b, features)}
            row["meets_recall_target"] = meets_recall_target(row)
            row["meets_sanity"] = row["addresses_over_limit"] == 0
            rows.append(row)
            log.info("%s |z| >= %.1f: recall spike %.2f, burst %.2f/%.2f, dormant %.2f; flag rate %.3f; %d addresses over the limit",
                     method, threshold, row["recall_spike"], row["recall_burst_tx"], row["recall_burst_event"],
                     row["recall_dormant"], row["flag_rate"], row["addresses_over_limit"])
    table = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "tuning_baseline.csv", index=False)

    choice: dict[str, Any] = {}
    for method in FEATURE_FILES:
        threshold, rule = choose_threshold([r for r in rows if r["features"] == method])
        choice[method] = {"threshold": threshold, "rule": rule,
                          "differs_from_a1": threshold != SETTINGS["baseline"]["flagAbsZ"]}
        log.info("%s: chosen |z| >= %.1f (%s)%s", method, threshold, rule,
                 "; differs from A1's 3.5, so this is CR-03 for the team" if choice[method]["differs_from_a1"] else "")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "baseline.json").write_text(json.dumps({
        "method": "robust-z baseline (A1 §2.3)", "score": "max |z| over F1-F4 when scorable",
        "a1_threshold": SETTINGS["baseline"]["flagAbsZ"], "chosen": choice,
        "tuned_on": "tuning plants, seeds " + ", ".join(map(str, SETTINGS["injection"]["tuningSeeds"])),
    }, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote %s and %s", TABLES_DIR / "tuning_baseline.csv", MODELS_DIR / "baseline.json")
    return 0
