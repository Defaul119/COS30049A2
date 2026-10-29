"""Evaluation of every DS1 method, and the choice of the method the Explorer uses.

Serves plan §11 (docs/plan/sections/s10-evaluation.tex), A1 §2.3.2 and decision D-A2-3: on DS1
there are no labels, so methods are judged only by checks that need none, on the evaluation
period, each at its own operating threshold:

    injection   recall per pattern on the REPORTING plants (seeds 30049-30053, never used for
                tuning), mean and SD over the 5 seeds; burst per transaction and per event; the
                precision lower bound; PR-AUC against the plants (threshold-free, so a method
                cannot look good just by flagging more)
    stability   5 model seeds x 5 bootstrap resamples of the fit period; mean pairwise Jaccard of
                the top 50 evaluation flags. The baseline and TxGPT do not fit anything on DS1,
                so their stability is 1 by construction, and the table says so.
    agreement   pairwise Jaccard of the methods' flag sets on the real evaluation period
    sanity      the largest share of any address's evaluation rows flagged (limit 20%), and per address
                which methods break it and how the address changed after the cut
    flag rate   the share of real evaluation transactions flagged
    ablation    the A1-as-written method against the refinements (CR-02, CR-04, CR-05)

Selection (plan §11.4, the A1 rule): among methods with stability >= the target, the highest
mean injection recall; within 5 percentage points, the easier one to explain, in the order of
settings.json -> evaluation.explainabilityOrder. The runner-up and the "flag if either flags"
combination are reported too. The choice is a recommendation: it becomes explorerMethod in
settings.json only when the team records it.

Writes reports/tables/eval_{injection,injection_by_seed,stability,agreement,sanity,sanity_addresses,ablation,
selection}.csv and models/selection.json.
"""
from __future__ import annotations

import json
import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from .baseline import FEATURE_FILES, MODELS_DIR, TABLES_DIR, Baseline, load_copies
from .clustering import DBSCANMethod, KMeansMethod
from .fusion import FusionMethod
from .methods import (
    Method, fit_rows, flags, jaccard, plant_metrics, real_metrics, stability, top_k,
)
from .outliers import IsolationForestMethod, LOFMethod
from .regression import RegressionIntervalMethod, RegressionMethod, design, residual_scales, targets
from .scores import saved_methods
from .settings import SETTINGS
from .snapshot import MANIFEST_PATH
from .split import EVAL, FIT
from .stage_search import IntervalCandidate, Pooled

log = logging.getLogger(__name__)

FIXED = {"baseline", "txgpt_method"}  # nothing is fitted on DS1, so refits cannot change them
PATTERN_KEYS = ["recall_spike", "recall_burst_tx", "recall_burst_event", "recall_dormant",
                "precision_lower_bound", "pr_auc"]


def refit(name: str, method: Method, rows: pd.DataFrame, seed: int) -> Method:
    """The same configuration fitted again on `rows` (a bootstrap resample) with `seed`."""
    if isinstance(method, IsolationForestMethod):
        return IsolationForestMethod(method.params, seed).fit(rows)
    if isinstance(method, LOFMethod):
        return LOFMethod(method.params, seed).fit(rows)
    if isinstance(method, KMeansMethod):
        return KMeansMethod(method.k, seed).fit(rows)
    if isinstance(method, DBSCANMethod):
        return DBSCANMethod(method.eps, method.min_samples).fit(rows)
    if isinstance(method, RegressionMethod):
        floor = SETTINGS["regression"]["residualFloor"]
        models, scales, default = {}, {}, {}
        for t, model in method.models.items():
            y = targets(rows)[t]
            keep = y.notna()
            models[t] = reseeded(model, seed).fit(design(rows[keep]), y[keep])
            scales[t], default[t] = residual_scales(models[t], rows[keep], y[keep], floor)
        return RegressionMethod(models, scales, default)
    if isinstance(method, RegressionIntervalMethod):
        quantiles = {}
        for t, pair in method.quantiles.items():
            y = targets(rows)[t]
            keep = y.notna()
            quantiles[t] = tuple(reseeded(m, seed).fit(design(rows[keep]), y[keep]) for m in pair)
        # The conformal margins are kept: a refit measures how much the ranges move, not a new calibration.
        return RegressionIntervalMethod(quantiles, getattr(method, "margins", {}))
    if isinstance(method, FusionMethod):
        # Each fitted component is refitted (fixed ones stay); the tuned multipliers are part of the configuration.
        parts = {n: m if n in FIXED else refit(n, m, rows, seed) for n, m in method.components.items()}
        fused = FusionMethod(parts, method.multipliers)
        fused.threshold = method.threshold  # 1 at M0; a stage C9 winner's is its flag budget's cut
        return fused
    if isinstance(method, (Pooled, IntervalCandidate)):
        # A training stage's winner (settings stageModels): the same model, inputs and calibration share fitted
        # again; its threshold, set at the method's flag budget by the stage search, is part of the configuration.
        fresh = (Pooled(method.kind, method.params, method.input_set, seed) if isinstance(method, Pooled) else
                 IntervalCandidate(method.extra, method.pair, method.share, method.hgb, seed)).fit(rows)
        fresh.threshold = method.threshold
        return fresh
    raise ValueError(f"no refit rule for {name}")


def reseeded(pipeline: Any, seed: int) -> Any:
    """An unfitted copy of a fitted Pipeline, with the model's seed replaced where it has one."""
    fresh = clone(pipeline)
    if "random_state" in fresh.named_steps["model"].get_params():
        fresh.set_params(model__random_state=seed)
    return fresh


def method_stability(name: str, method: Method, features: pd.DataFrame) -> float:
    m = SETTINGS["model"]
    if name in FIXED:
        return 1.0
    rows = fit_rows(features)
    sets = []
    for i in range(m["stabilitySeeds"]):
        for j in range(m["stabilityBootstraps"]):
            sample = rows.sample(frac=1.0, replace=True, random_state=m["seed"] + 100 * i + j)
            fitted = refit(name, method, sample, m["seed"] + i)
            sets.append(top_k(fitted, features, m["stabilityTopK"]))
    return stability(sets)


def injection(methods: dict[str, Method], copies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_seed = []
    for name, method in methods.items():
        for seed, part in copies.groupby("seed"):
            by_seed.append({"method": name, "seed": int(seed), **plant_metrics(method, part)})
    by_seed = pd.DataFrame(by_seed)
    summary = by_seed.groupby("method")[PATTERN_KEYS].agg(["mean", "std"])
    summary.columns = [f"{k}_{stat}" for k, stat in summary.columns]
    summary = summary.reset_index()
    summary["mean_recall"] = summary[["recall_spike_mean", "recall_burst_tx_mean", "recall_dormant_mean"]].mean(axis=1)
    p = SETTINGS["injection"]
    summary["patterns_meeting_target"] = (summary[["recall_spike_mean", "recall_burst_tx_mean", "recall_dormant_mean"]]
                                          >= p["recallTarget"]).sum(axis=1)
    return summary, by_seed


def agreement(methods: dict[str, Method], features: pd.DataFrame) -> pd.DataFrame:
    ev = features[features["period"] == EVAL]
    keys = (ev["address"] + ":" + ev["txid"]).to_numpy()
    sets = {name: set(keys[flags(m, ev)]) for name, m in methods.items()}
    rows = [{"a": a, "b": b, "jaccard": jaccard(sets[a], sets[b])} for a, b in combinations(sets, 2)]
    for name, s in sets.items():
        others = set().union(*(v for k, v in sets.items() if k != name))
        rows.append({"a": name, "b": "(only this method)", "jaccard": np.nan, "only_count": len(s - others),
                     "flagged": len(s)})
    return pd.DataFrame(rows)


def ablation(methods: dict[str, Method]) -> pd.DataFrame:
    """Each method on the A1-as-written features vs the refined ones, on the reporting plants.

    Fitted methods are refitted with the chosen configuration on the as-written fit period;
    regression, TxGPT and the fusion (which contains a regression) have no as-written counterpart
    (A1 did not define them), nor have the training stages' winners (chosen on the refined features)."""
    rows = []
    seed = SETTINGS["model"]["seed"]
    for variant in FEATURE_FILES:
        features = pd.read_csv(MANIFEST_PATH.parent / FEATURE_FILES[variant])
        copies = load_copies(variant, "reporting")
        for name, method in methods.items():
            if name.startswith("regression") or name in ("txgpt_method", "fusion") or isinstance(method, Pooled):
                continue  # no A1-as-written counterpart (a stage winner was chosen on the refined features)
            m = method if (name == "baseline" or variant == "refined") else refit(name, method, fit_rows(features), seed)
            if name == "baseline" and variant == "as_written":
                chosen = json.loads((MODELS_DIR / "baseline.json").read_text(encoding="utf-8"))["chosen"]["as_written"]
                m = Baseline(chosen["threshold"])
            rows.append({"method": name, "features": variant, **plant_metrics(m, copies), **real_metrics(m, features)})
    table = pd.DataFrame(rows)
    table["mean_recall"] = table[["recall_spike", "recall_burst_tx", "recall_dormant"]].mean(axis=1)
    return table


def sanity_by_address(methods: dict[str, Method], features: pd.DataFrame) -> pd.DataFrame:
    """Each method's flagged share per address (the sanity rule's own denominator: every evaluation row),
    next to how the address's evaluation period differs from its fit period.

    It shows whether a sanity failure belongs to a method's threshold or to an address whose behaviour
    changed after the 70/30 cut: in the second case every method flags it, and no threshold fixes it.
    """
    ev, fit = features[features["period"] == EVAL], features[features["period"] == FIT]
    limit = SETTINGS["sanity"]["maxFlaggedShare"]

    def profile(rows: pd.DataFrame) -> pd.DataFrame:
        g = rows.assign(abs_btc=rows["net_sats"].abs() / 1e8, is_in=rows["direction"] == "in").groupby("address")
        return g.agg(median_gap_hours=("F3", "median"), median_abs_btc=("abs_btc", "median"), in_share=("is_in", "mean"))

    change = profile(fit).join(profile(ev), lsuffix="_fit", rsuffix="_eval")
    rows = []
    for name, m in methods.items():
        flagged = pd.Series(flags(m, ev), index=ev.index).groupby(ev["address"]).agg(["sum", "size"])
        for address, r in flagged.iterrows():
            rows.append({"method": name, "address": address, "eval_rows": int(r["size"]), "flagged": int(r["sum"]),
                         "share": r["sum"] / r["size"], "over_limit": r["sum"] / r["size"] > limit})
    table = pd.DataFrame(rows)
    table["methods_over_limit"] = table.groupby("address")["over_limit"].transform("sum")
    return table.merge(change, left_on="address", right_index=True, how="left").sort_values(
        ["methods_over_limit", "address", "method"], ascending=[False, True, True])


def select(summary: pd.DataFrame, stab: pd.DataFrame, sanity: pd.DataFrame) -> dict[str, Any]:
    """The A1 selection rule with the plan's explainability tie-break."""
    e = SETTINGS["evaluation"]
    table = summary.merge(stab, on="method").merge(sanity, on="method")
    stable = table[table["stability"] >= SETTINGS["model"]["stabilityTarget"]]
    pool = stable if len(stable) else table
    best_recall = pool["mean_recall"].max()
    close = pool[pool["mean_recall"] >= best_recall - e["tieMarginPoints"] / 100]
    order = {name: i for i, name in enumerate(e["explainabilityOrder"])}
    chosen = min(close["method"], key=lambda n: order.get(n, len(order)))
    ranked = pool.sort_values("mean_recall", ascending=False)["method"].tolist()
    runner_up = next((n for n in ranked if n != chosen), None)
    return {"chosen": chosen, "runner_up": runner_up,
            "rule": "highest mean injection recall among stable methods; within "
                    f"{e['tieMarginPoints']} points, the easier to explain ({' > '.join(e['explainabilityOrder'])})",
            "no_method_stable": len(stable) == 0, "table": table}


def combination(methods: dict[str, Method], names: list[str], features: pd.DataFrame, copies: pd.DataFrame) -> dict[str, Any]:
    """'Flag if either flags' for two methods: recall and the sanity rule (an option for A3)."""
    class Either:
        threshold = 0.5

        def score(self, frame: pd.DataFrame) -> np.ndarray:
            return np.any([flags(methods[n], frame) for n in names], axis=0).astype(float)

    either = Either()
    return {"methods": " + ".join(names), **plant_metrics(either, copies), **real_metrics(either, features)}


def run_evaluate() -> int:
    features = pd.read_csv(MANIFEST_PATH.parent / FEATURE_FILES["refined"])
    copies = load_copies("refined", "reporting")
    methods = saved_methods()
    log.info("Evaluating %s on %d reporting plants", list(methods), copies["plant_id"].nunique())

    summary, by_seed = injection(methods, copies)
    stab = pd.DataFrame([{"method": n, "stability": method_stability(n, m, features),
                          "by_construction": n in FIXED} for n, m in methods.items()])
    sanity = pd.DataFrame([{"method": n, **real_metrics(m, features)} for n, m in methods.items()])
    choice = select(summary, stab, sanity)
    table = choice.pop("table")
    combo = combination(methods, [choice["chosen"], choice["runner_up"]], features, copies) if choice["runner_up"] else {}

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES_DIR / "eval_injection.csv", index=False)
    by_seed.to_csv(TABLES_DIR / "eval_injection_by_seed.csv", index=False)
    stab.to_csv(TABLES_DIR / "eval_stability.csv", index=False)
    sanity.to_csv(TABLES_DIR / "eval_sanity.csv", index=False)
    sanity_by_address(methods, features).to_csv(TABLES_DIR / "eval_sanity_addresses.csv", index=False)
    agreement(methods, features).to_csv(TABLES_DIR / "eval_agreement.csv", index=False)
    ablation(methods).to_csv(TABLES_DIR / "eval_ablation.csv", index=False)
    table.to_csv(TABLES_DIR / "eval_selection.csv", index=False)
    (MODELS_DIR / "selection.json").write_text(json.dumps({**choice, "combination": combo}, indent=2, default=float) + "\n",
                                               encoding="utf-8")
    for row in table.sort_values("mean_recall", ascending=False).itertuples():
        log.info("%-20s mean recall %.3f (spike %.2f, burst %.2f, dormant %.2f), stability %.2f, flag rate %.3f, max address share %.2f",
                 row.method, row.mean_recall, row.recall_spike_mean, row.recall_burst_tx_mean, row.recall_dormant_mean,
                 row.stability, row.flag_rate, row.max_address_share)
    log.info("Recommended method: %s (runner-up %s). Record it as explorerMethod in settings.json to adopt it.",
             choice["chosen"], choice["runner_up"])
    return 0
