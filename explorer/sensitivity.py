"""How strong must a pattern be before each DS1 method sees it? Recall against plant strength.

Serves plan §11.1 (injection recall) and report criterion 2 (data analysis). The reported injection results
use one strength per pattern (settings.json -> injection: a 10x spike, 10 transactions in 24 hours, 60-120
days of dormancy), which says nothing about where a method stops seeing a pattern. Here each pattern is
planted again at several strengths (settings.json -> sensitivity), with the REPORTING seeds and every
other setting unchanged, and scored by the frozen final methods.

The swept parameter changes neither which addresses get plants nor where a spike lands (make_plants draws
those first, and the amount and dormancy only scale), so at the default strengths the plants are exactly
the reporting plants and their recall must equal eval_injection.csv; the log checks this. A burst's size
changes how many random draws it takes, so bursts of other sizes are fresh plants.

Writes reports/tables/eval_sensitivity.csv.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .baseline import TABLES_DIR
from .features import FeatureConfig
from .inject import copy_features, make_plants
from .methods import flags
from .predict import PRECOMPUTED_ONLY
from .scores import saved_methods
from .settings import SETTINGS
from .snapshot import MANIFEST_PATH
from .split import EVAL

log = logging.getLogger(__name__)


def strengths() -> list[tuple[str, str, float, dict[str, Any]]]:
    """(pattern, label, x for the plot, injection-setting overrides) for every strength in the sweep."""
    s = SETTINGS["sensitivity"]
    out = [("spike", f"{m:g}x", float(m), {"spikeMultiplier": m}) for m in s["spikeMultiplier"]]
    out += [("burst", f"{n} in {SETTINGS['injection']['burstWindowHours']} h", float(n), {"burstSize": n})
            for n in s["burstSize"]]
    out += [("dormant", f"{lo}-{hi} days", float(np.sqrt(lo * hi)), {"dormantDaysMin": lo, "dormantDaysMax": hi})
            for lo, hi in s["dormantDays"]]
    return out


def is_default(overrides: dict[str, Any]) -> bool:
    return all(SETTINGS["injection"][k] == v for k, v in overrides.items())


def run_sensitivity(n_jobs: int = SETTINGS["classification"]["nJobs"]) -> int:
    """Step `sensitivity`: each method's recall as each planted pattern is made weaker or stronger; writes
    eval_sensitivity.csv."""
    transactions = pd.read_csv(MANIFEST_PATH.parent / "transactions.csv")
    periods = pd.read_csv(MANIFEST_PATH.parent / "features.csv", usecols=["address", "txid", "period"])
    # TxGPT's surprisal needs its own GPU scoring pass for each new copy (lm.tx_score), so it sits this one out.
    methods = {n: m for n, m in saved_methods().items() if n not in PRECOMPUTED_ONLY}
    cfg = FeatureConfig.refined()
    rows = []
    for pattern, label, x, overrides in strengths():
        p = {**SETTINGS["injection"], **overrides}
        plants = [pl for seed in SETTINGS["injection"]["reportingSeeds"]
                  for pl in make_plants(transactions, periods, seed, "reporting", p) if pl.pattern == pattern]
        frames = Parallel(n_jobs=n_jobs)(delayed(copy_features)(transactions, periods, pl, cfg) for pl in plants)
        planted = pd.concat([f[(f["is_synthetic"] == 1) & (f["period"] == EVAL)] for f in frames], ignore_index=True)
        for name, method in methods.items():
            per_plant = planted.assign(hit=flags(method, planted)).groupby("plant_id")["hit"].agg(["max", "mean"])
            rows.append({"pattern": pattern, "strength": label, "x": x, "default": is_default(overrides),
                         "method": name, "plants": len(per_plant),
                         # A1's measures: a spike or dormancy is caught if flagged; a burst per transaction
                         # (the share of its transactions flagged) and per event (at least one flagged).
                         "recall": float(per_plant["mean" if pattern == "burst" else "max"].mean()),
                         "recall_event": float(per_plant["max"].mean())})
        log.info("%s %s: %d plants; %s", pattern, label, len(plants),
                 ", ".join(f"{r['method']} {r['recall']:.2f}" for r in rows[-len(methods):]))
    table = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "eval_sensitivity.csv", index=False)
    log.info("Wrote %s", TABLES_DIR / "eval_sensitivity.csv")
    if check_against_injection(table):
        log.info("Default-strength recall differs from eval_injection.csv: rerun evaluate, or look for a planting change")
        return 1
    return 0


def check_against_injection(table: pd.DataFrame) -> int:
    """At the default strengths the plants are the reporting plants: recall must agree. Returns the disagreements."""
    path = TABLES_DIR / "eval_injection.csv"
    if not path.exists():
        return 0
    injection = pd.read_csv(path).set_index("method")
    column = {"spike": "recall_spike_mean", "burst": "recall_burst_tx_mean", "dormant": "recall_dormant_mean"}
    different = 0
    for r in table[table["default"]].itertuples():
        if r.method in injection.index:
            reported = injection.loc[r.method, column[r.pattern]]
            different += abs(r.recall - reported) > 1e-9
            log.info("check %s %s at the default strength: %.3f here, %.3f in eval_injection.csv%s", r.method, r.pattern,
                     r.recall, reported, "" if abs(r.recall - reported) < 1e-9 else "  <- DIFFERENT")
    return different
