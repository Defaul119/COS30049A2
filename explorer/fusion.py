"""Score fusion: two DS1 methods that catch different patterns, combined into one flag at the same flag budget.

Serves plan §11.4 (the "flag if either flags" combination, an option for A3) and the brief's additional
ML implementations. The reporting plants (reports/tables/eval_injection.csv, eval_agreement.csv) show
methods that are complementary rather than redundant: the robust-z baseline catches dormancy that the
regression residual barely sees, the residual catches spikes and bursts the baseline misses, and their
flag sets overlap little. OR-ing two methods at their own thresholds would flag far more transactions,
so recall would rise for the wrong reason; the fusion is therefore tuned at an EQUAL FLAG BUDGET.

Method: each component's score is divided by m x that component's own flag threshold and the fused score
is the largest of these, flagged at 1. A transaction is flagged when any component's score reaches m times
that component's flag level, and the fused score still ranks every transaction (for PR-AUC and stability).
The multipliers are chosen on the TUNING plants (seeds 40000-40004, never the reported ones), by the rule
outliers.py uses for its grids: the highest mean tuning recall among configurations that keep the flag rate
on the real evaluation period within the budget (the flag rate of settings.json -> explorerMethod) AND reach
the stability target; ties go to the higher PR-AUC. Stability needs 25 refits, so it is computed for the
best `stabilityCandidates` configurations of each pair within the budget. A first run without the stability
condition chose regression + Isolation Forest, whose reporting stability (0.61) then excluded it from
selection: the regression component's flags move between refits. The saved fusion's reporting-plant
results come from `evaluate`, alongside every other method, so it is selected (or not) by the same rule.

The single methods in settings.json -> fusion.references are searched over the same multipliers (a single
method's multiplier just moves its threshold). Their rows, next to the pairs', trace each candidate's
recall against its flag rate, so the fusion can be judged at every flag budget, not only at the chosen one.

Writes reports/tables/tuning_fusion.csv and models/fusion.joblib.
"""
from __future__ import annotations

import logging
from itertools import product

import joblib
import numpy as np
import pandas as pd

from .baseline import FEATURE_FILES, MODELS_DIR, TABLES_DIR, load_copies
from .methods import Method, pattern_recalls, plant_metrics, real_metrics
from .scores import saved_methods
from .settings import SETTINGS
from .snapshot import MANIFEST_PATH
from .split import EVAL

log = logging.getLogger(__name__)


class FusionMethod:
    """Score = max over components of score / (multiplier x the component's threshold); flagged at 1."""

    threshold = 1.0

    def __init__(self, components: dict[str, Method], multipliers: dict[str, float]):
        self.components, self.multipliers = components, multipliers

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        parts = [m.score(frame) / (self.multipliers[n] * m.threshold) for n, m in self.components.items()]
        return np.max(np.vstack(parts), axis=0)


class Cached:
    """A component's scores computed once for one frame, so the multiplier search never re-scores."""

    def __init__(self, threshold: float, scores: pd.Series):
        self.threshold, self.scores = threshold, scores

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.scores.loc[frame.index].to_numpy(dtype="float64")


def cached(method: Method, frame: pd.DataFrame) -> Cached:
    rows = frame[frame["period"] == EVAL]
    return Cached(method.threshold, pd.Series(method.score(rows), index=rows.index))


def search(names: list[str], methods: dict[str, Method], copies: pd.DataFrame, features: pd.DataFrame,
           budget: float) -> pd.DataFrame:
    """Every multiplier combination for one candidate pair: tuning-plant metrics and real flag rate."""
    on_copies = {n: cached(methods[n], copies) for n in names}
    on_real = {n: cached(methods[n], features) for n in names}
    rows = []
    for combo in product(SETTINGS["fusion"]["multipliers"], repeat=len(names)):
        mult = dict(zip(names, combo))
        row = {"pair": " + ".join(names), "components": len(names), **{f"multiplier_{i + 1}": m for i, m in enumerate(combo)},
               **plant_metrics(FusionMethod(on_copies, mult), copies),
               **real_metrics(FusionMethod(on_real, mult), features)}
        row["mean_recall"] = float(np.mean(pattern_recalls(row)))
        row["within_budget"] = row["flag_rate"] <= budget
        rows.append(row)
    return pd.DataFrame(rows)


def best_within_budget(table: pd.DataFrame) -> pd.Series | None:
    """Highest mean tuning recall within the flag budget; ties go to the higher PR-AUC."""
    pool = table[table["within_budget"]]
    return pool.sort_values(["mean_recall", "pr_auc"], ascending=False).iloc[0] if len(pool) else None


def choose(table: pd.DataFrame, target: float) -> tuple[pd.Series | None, str]:
    """The rule across every pair: best within the budget among stable configurations, else best within the budget."""
    within = table[table["within_budget"] & (table["components"] > 1)]
    stable = within[within["stability"] >= target]
    if len(stable):
        return best_within_budget(stable), "highest mean tuning recall within the flag budget among stable configurations"
    return best_within_budget(within), "no configuration within the budget reached the stability target; highest recall"


def multipliers_of(row: pd.Series) -> dict[str, float]:
    names = row["pair"].split(" + ")
    return {n: float(row[f"multiplier_{i + 1}"]) for i, n in enumerate(names)}


def run_fusion() -> int:
    from .evaluate import method_stability  # evaluate imports this module, so import it only when running

    f, target = SETTINGS["fusion"], SETTINGS["model"]["stabilityTarget"]
    features = pd.read_csv(MANIFEST_PATH.parent / FEATURE_FILES["refined"])
    copies = load_copies("refined", "tuning")
    # The M0 components, as when the fusion was chosen: a rebuild must reproduce it whatever later stages promoted
    # (stage C9 searched fusions of the promoted methods and kept this one; CHANGELOG 29 Oct).
    methods = saved_methods(promoted=False)
    budget_method = SETTINGS["explorerMethod"]
    budget = real_metrics(methods[budget_method], features)["flag_rate"]
    log.info("Flag budget: %.4f, the flag rate of explorerMethod (%s) on the real evaluation period", budget, budget_method)

    tables = []
    for names in f["references"] + f["candidates"]:
        missing = [n for n in names if n not in methods]
        unusable = [n for n in names if n in methods and not methods[n].threshold > 0]
        if missing or unusable:
            # The fused score divides by each threshold, so a component flagged "above 0" cannot join.
            log.info("Skipping %s: %s", " + ".join(names), f"not saved yet: {missing}" if missing else
                     f"threshold not positive: {unusable}")
            continue
        table = search(names, methods, copies, features, budget).assign(stability=np.nan)
        best = best_within_budget(table)
        if len(names) == 1:
            if best is not None:
                log.info("Reference %s alone: best within the budget %.3f (multiplier %s, flag rate %.4f)", names[0],
                         best["mean_recall"], best["multiplier_1"], best["flag_rate"])
            tables.append(table)
            continue
        top = table[table["within_budget"]].sort_values(["mean_recall", "pr_auc"], ascending=False).head(f["stabilityCandidates"])
        for idx, row in top.iterrows():
            fused = FusionMethod({n: methods[n] for n in names}, multipliers_of(row))
            table.loc[idx, "stability"] = method_stability("fusion", fused, features)
            log.info("%s %s: tuning mean recall %.3f (spike %.2f, burst %.2f, dormant %.2f), flag rate %.4f, stability %.2f",
                     row["pair"], list(multipliers_of(row).values()), row["mean_recall"], row["recall_spike"],
                     row["recall_burst_tx"], row["recall_dormant"], row["flag_rate"], table.loc[idx, "stability"])
        tables.append(table)

    every = pd.concat(tables, ignore_index=True)
    best, rule = choose(every, target)
    if best is None:
        log.info("No fusion keeps the flag rate within %.4f; nothing saved", budget)
        return 1
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    every.to_csv(TABLES_DIR / "tuning_fusion.csv", index=False)
    multipliers = multipliers_of(best)
    names = list(multipliers)
    method = FusionMethod({n: methods[n] for n in names}, multipliers)
    rule += " (the explorerMethod's real evaluation flag rate is the budget; ties by PR-AUC; across settings.fusion.candidates)"
    joblib.dump({"method": method, "threshold": method.threshold, "components": names, "multipliers": multipliers,
                 "budget": budget, "budget_method": budget_method, "rule": rule, "stability": float(best["stability"]),
                 "tuning": {k: (float(best[k]) if isinstance(best[k], (int, float, np.number)) else best[k])
                            for k in ("mean_recall", "recall_spike", "recall_burst_tx", "recall_dormant",
                                      "pr_auc", "flag_rate")}},
                MODELS_DIR / "fusion.joblib", compress=3)
    log.info("Chose %s with multipliers %s (%s) -> models/fusion.joblib; wrote %s", " + ".join(names), multipliers, rule,
             TABLES_DIR / "tuning_fusion.csv")
    return 0
