"""Stage D6 (docs/TRAINING_PLAN.md section 4): an ensemble of the four Elliptic classifiers, gated against the random
forest, M0's best classifier.

Serves TRAINING_PLAN sections 3-4 and the HD criterion "additional ML implementations". The members are the four
reported all-feature classifiers with their tuned settings (elliptic_results.csv), fitted on the fit steps (1-26)
exactly as explorer/classify.py fits them. The validation steps (27-34) are split in time, so the ensemble is never
judged on the rows that chose it:

- steps 27-30 (settings ensemble.weightLastStep) choose the ensemble: soft voting (a weighted mean of the members'
  probabilities, every weight vector in ensemble.weights with at least two members) or stacking (a logistic
  regression on the members' log-odds), and its decision threshold (the best F1, classify.best_f1_threshold);
- steps 31-34 gate it against the random forest, whose threshold is chosen on steps 27-30 the same way: the F1
  difference with a paired bootstrap of the rows (stages.bootstrapResamples) and McNemar's exact test on the rows
  each gets right. No gate time step's F1 may fall by more than the random forest's own seed spread on that step
  (refits with the classDistribution seeds), the classifier version of the per-pattern rule.

Only an accepted ensemble reaches the test tier (steps 35-49, once): its members refitted on steps 1-34 and its
weights and threshold unchanged, against the reported random forest on identical bootstrap resamples. Elliptic's
rows are not shared (CC BY-NC-ND), so this runs where the raw data is (kaggle/ensemble attaches the Kaggle copy);
models stay in memory, and only metrics are written.

Writes reports/tables/ensemble_search.csv, ensemble_results.csv and rows of reports/tables/stages.csv.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .classify import MODEL_KINDS, TABLES_DIR, best_f1_threshold, make_model, test_metrics
from .classify_distribution import bootstrap
from .labelled import ELLIPTIC_DIR, FEATURE_SETS, FIT, TEST, VAL
from .settings import SETTINGS
from .stages import decide, interval, mcnemar_p, record

log = logging.getLogger(__name__)

REFERENCE = "randomForest"  # M0's best classifier on Elliptic (stats_classifiers_summary.csv)


@dataclass
class Ensemble:
    """Soft voting (`weights`, one per member in MODEL_KINDS order) or stacking (`stacker`); flagged at `threshold`."""

    combine: str
    threshold: float = 0.5
    weights: np.ndarray | None = None
    stacker: Pipeline | None = None

    def score(self, p: np.ndarray) -> np.ndarray:
        """p: rows x members, each member's probability of the positive class."""
        if self.combine == "mean":
            return p @ self.weights / self.weights.sum()
        return self.stacker.predict_proba(log_odds(p))[:, 1]


def log_odds(p: np.ndarray) -> np.ndarray:
    clip = SETTINGS["ensemble"]["probabilityClip"]
    q = np.clip(p, clip, 1 - clip)
    return np.log(q / (1 - q))


def member_settings() -> dict[str, dict[str, Any]]:
    """Each member's tuned configuration and validation threshold, as reported (elliptic_results.csv)."""
    results = pd.read_csv(TABLES_DIR / "elliptic_results.csv")
    results = results[results["feature_set"] == SETTINGS["ensemble"]["featureSet"]].set_index("model")
    return {k: {"params": json.loads(results.loc[k, "params"]), "threshold": float(results.loc[k, "threshold"])}
            for k in MODEL_KINDS}


def fit_members(train: pd.DataFrame, features: list[str], members: dict[str, dict[str, Any]],
                seed: int) -> dict[str, Pipeline]:
    cfg = {**SETTINGS["classification"], "seed": seed}
    return {k: make_model(k, m["params"], cfg, len(train)).fit(train[features], train["y"]) for k, m in members.items()}


def probabilities(models: dict[str, Pipeline], frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.column_stack([models[k].predict_proba(frame[features])[:, 1] for k in MODEL_KINDS])


def search(p: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, Ensemble]:
    """Every soft-voting weight vector with at least two members, and stacking at every C, each at its best-F1
    threshold on (p, y); the winner has the highest F1, then the fewest members."""
    e = SETTINGS["ensemble"]
    candidates, seen = [], set()  # (ensemble, what the search table records about it)
    for w in product(e["weights"], repeat=len(MODEL_KINDS)):
        w = np.asarray(w, dtype="float64")
        key = tuple(np.round(w / w.sum(), 9)) if w.sum() else None
        if np.count_nonzero(w) < 2 or key in seen:  # (1, 1, 0, 0) and (2, 2, 0, 0) are the same vote
            continue
        seen.add(key)
        candidates.append((Ensemble("mean", weights=w), {"members": int(np.count_nonzero(w)),
                                                        **{f"w_{k}": float(x) for k, x in zip(MODEL_KINDS, w)}}))
    for c in e["stackC"]:
        stacker = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=SETTINGS["classification"]["logistic"]["max_iter"]))
        candidates.append((Ensemble("stack", stacker=stacker.fit(log_odds(p), y)), {"members": len(MODEL_KINDS), "stack_c": c}))
    rows, models = [], []
    for i, (m, about) in enumerate(candidates):
        m.threshold, f1 = best_f1_threshold(y, m.score(p))
        rows.append({"candidate": i, "combine": m.combine, **about, "f1": f1, "threshold": m.threshold})
        models.append(m)
    table = pd.DataFrame(rows).sort_values(["f1", "members"], ascending=[False, True]).reset_index(drop=True)
    table["chosen"] = table.index == 0
    return table, models[int(table.loc[0, "candidate"])]


def step_f1(y: np.ndarray, flagged: np.ndarray, steps: np.ndarray) -> dict[str, float]:
    return {f"step {s}": float(f1_score(y[steps == s], flagged[steps == s], zero_division=0)) for s in np.unique(steps)}


def compare(y: np.ndarray, p_new: np.ndarray, p_ref: np.ndarray, thresholds: tuple[float, float]) -> dict[str, Any]:
    """F1 of both on all rows, the bootstrap interval of the difference (paired resamples) and McNemar's test."""
    s = SETTINGS["stages"]
    new, ref = p_new >= thresholds[0], p_ref >= thresholds[1]
    draws = bootstrap(y, {"new": p_new, "ref": p_ref}, {"new": thresholds[0], "ref": thresholds[1]},
                      s["bootstrapResamples"], s["seed"])
    f1_new, f1_ref = f1_score(y, new, zero_division=0), f1_score(y, ref, zero_division=0)
    low, high = interval(draws["new"]["f1"] - draws["ref"]["f1"], s["ciLevel"])
    return {"f1_new": f1_new, "f1_ref": f1_ref, "diff": f1_new - f1_ref, "diff_ci_low": low, "diff_ci_high": high,
            "f1_new_ci": interval(draws["new"]["f1"], s["ciLevel"]),
            "mcnemar_p": mcnemar_p(ref == y.astype(bool), new == y.astype(bool))}


def run_ensemble() -> int:
    """Step `ensemble` (stage D6): choose a soft-voting or stacking ensemble on Elliptic steps 27-30, gate it against
    the random forest on 31-34, and test it once only if accepted; writes ensemble_{search,results}.csv."""
    e = SETTINGS["ensemble"]
    features = FEATURE_SETS[e["featureSet"]]
    rows = pd.read_parquet(ELLIPTIC_DIR / "rows.parquet")
    rows = rows[rows["y"].notna()].assign(y=lambda f: f["y"].astype(int))
    fit, val, test = (rows[rows["period"] == part] for part in (FIT, VAL, TEST))
    choose, gate = val[val["time_step"] <= e["weightLastStep"]], val[val["time_step"] > e["weightLastStep"]]
    log.info("Elliptic: fit %d rows, choose (steps %d-%d) %d, gate (steps %d-%d) %d, test %d", len(fit),
             choose["time_step"].min(), choose["time_step"].max(), len(choose), gate["time_step"].min(),
             gate["time_step"].max(), len(gate), len(test))

    members = member_settings()
    seed = SETTINGS["classification"]["seed"]
    models = fit_members(fit, features, members, seed)
    p_choose, p_gate = probabilities(models, choose, features), probabilities(models, gate, features)
    y_choose, y_gate, steps = choose["y"].to_numpy(), gate["y"].to_numpy(), gate["time_step"].to_numpy()
    table, winner = search(p_choose, y_choose)
    table.to_csv(TABLES_DIR / "ensemble_search.csv", index=False)
    best = table.iloc[0]
    log.info("Chose %s (%d members) at threshold %.4f: F1 %.4f on steps up to %d", winner.combine, best["members"],
             winner.threshold, best["f1"], e["weightLastStep"])

    # The reference: the random forest member, its threshold chosen on the same steps; its seed spread per gate step.
    r = MODEL_KINDS.index(REFERENCE)
    ref_threshold, _ = best_f1_threshold(y_choose, p_choose[:, r])
    d = SETTINGS["classDistribution"]
    per_seed = []
    for s in range(d["firstSeed"], d["firstSeed"] + d["seeds"]):
        m = fit_members(fit, features, {REFERENCE: members[REFERENCE]}, s)[REFERENCE]
        t, _ = best_f1_threshold(y_choose, m.predict_proba(choose[features])[:, 1])
        per_seed.append(step_f1(y_gate, m.predict_proba(gate[features])[:, 1] >= t, steps))
    spread = pd.DataFrame(per_seed).std(ddof=1).to_dict()

    p_new, p_ref = winner.score(p_gate), p_gate[:, r]
    c = compare(y_gate, p_new, p_ref, (winner.threshold, ref_threshold))
    result = decide(c["diff_ci_low"], step_f1(y_gate, p_ref >= ref_threshold, steps),
                    step_f1(y_gate, p_new >= winner.threshold, steps), spread, loss_agrees=True)
    gate_steps = f"{gate['time_step'].min()}-{gate['time_step'].max()}"
    record({"family": "classifier", "model": "ensemble", "stage": "D6", "tier": "dev",
            "metric": f"F1, Elliptic steps {gate_steps}", "value": c["f1_new"], "ci_low": c["f1_new_ci"][0],
            "ci_high": c["f1_new_ci"][1], "previous_stage": f"M0 {REFERENCE}", "diff_previous": c["diff"],
            "diff_previous_ci_low": c["diff_ci_low"], "diff_previous_ci_high": c["diff_ci_high"],
            "mcnemar_p": c["mcnemar_p"], "accepted": result.accepted, "reasons": "; ".join(result.reasons),
            "seeds": f"members {seed}; spread {d['firstSeed']}-{d['firstSeed'] + d['seeds'] - 1}"})
    log.info("D6 dev: ensemble F1 %.4f vs %s %.4f, difference %+.4f [%+.4f, %+.4f], McNemar p %.3g -> %s %s",
             c["f1_new"], REFERENCE, c["f1_ref"], c["diff"], c["diff_ci_low"], c["diff_ci_high"], c["mcnemar_p"],
             "ACCEPTED" if result.accepted else "rejected", "; ".join(result.reasons))
    results = [{"tier": "dev", "model": name, "threshold": t, **test_metrics(y_gate, p, t)}
               for name, p, t in (("ensemble", p_new, winner.threshold), (REFERENCE, p_ref, ref_threshold))]

    if result.accepted:
        # Test tier, once: members refitted on steps 1-34 (classify.py's protocol), weights and threshold unchanged;
        # the reference is the reported random forest (refitted the same way, its threshold from steps 27-34).
        train = rows[rows["period"].isin([FIT, VAL])]
        p_test = probabilities(fit_members(train, features, members, seed), test, features)
        y_test, t_ref = test["y"].to_numpy(), members[REFERENCE]["threshold"]
        p_new, p_ref = winner.score(p_test), p_test[:, r]
        c = compare(y_test, p_new, p_ref, (winner.threshold, t_ref))
        record({"family": "classifier", "model": "ensemble", "stage": "D6", "tier": "test",
                "metric": "F1, Elliptic steps 35-49", "value": c["f1_new"], "ci_low": c["f1_new_ci"][0],
                "ci_high": c["f1_new_ci"][1], "previous_stage": f"M0 {REFERENCE}", "diff_previous": c["diff"],
                "diff_previous_ci_low": c["diff_ci_low"], "diff_previous_ci_high": c["diff_ci_high"],
                "mcnemar_p": c["mcnemar_p"], "accepted": c["diff_ci_low"] > 0,
                "reasons": "" if c["diff_ci_low"] > 0 else f"difference interval reaches {c['diff_ci_low']:+.4f}",
                "seeds": f"members {seed}"})
        log.info("D6 test: ensemble F1 %.4f vs %s %.4f, difference %+.4f [%+.4f, %+.4f], McNemar p %.3g", c["f1_new"],
                 REFERENCE, c["f1_ref"], c["diff"], c["diff_ci_low"], c["diff_ci_high"], c["mcnemar_p"])
        results += [{"tier": "test", "model": name, "threshold": t, **test_metrics(y_test, p, t)}
                    for name, p, t in (("ensemble", p_new, winner.threshold), (REFERENCE, p_ref, t_ref))]
    pd.DataFrame(results).to_csv(TABLES_DIR / "ensemble_results.csv", index=False)
    log.info("Wrote ensemble_search.csv, ensemble_results.csv and stages.csv rows")
    return 0
