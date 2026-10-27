"""Training stages C1-C6 (docs/TRAINING_PLAN.md section 4): each DS1 scoring method's grid, searched on the dev tier
at the method's own flag budget, and the winner put through the strict gate of explorer/stages.py.

Serves TRAINING_PLAN sections 3-4 and report criterion "model-selection justification". Three rules keep the search
honest:

1. Equal flag budget. A method that flags more catches more plants, so every candidate's threshold flags at most as
   many real evaluation transactions as the method did at M0 (its dev flag rate in stage_dev/M0_summary.csv, capped
   at sanity.maxFlaggedShare): the equal-budget rule of explorer/fusion.py. Mean planted recall then measures how
   well a candidate ranks transactions, not how many it flags.
2. Search and gate on different plants. Candidates are ranked on the tuning seeds in stageSearch.searchSeeds; the
   winner is gated against M0 on the other tuning seeds (stageSearch.gateSeeds), so the best of many candidates is
   not judged on the plants that chose it. The reporting seeds are never read.
3. Only candidates that could pass are ranked: the sanity rule (no address's flag share rising above both the limit
   and its M0 value) and the per-pattern rule (no pattern falling by more than its M0 seed spread, on the search
   seeds) are label-free or dev-only, so they filter the grid first. Isolation Forest and LOF also need the
   stability target, as at M0.

Each candidate is fitted on the real fit period and scored as arrays; the winner is rebuilt as a Method
(`score(frame)`, `threshold`) whose dev hits stages.py records, and its search-seed recalls must equal the arrays'.

Candidates (settings.json -> stageSearch): how the baseline combines |z| (C1); Isolation Forest, LOF, K-Means and
DBSCAN / HDBSCAN hyperparameters and input sets, F9 included as an option (C2-C5); regression's model family per
target, input set and residual floor, and the range variant's quantiles and calibration share (C6). Per-address
Isolation Forests are left out: they cannot score an address they were not fitted on (criterion C3).

Writes reports/tables/stage_search/<stage>_<method>.csv (every candidate), models/stage/<stage>_<method>.joblib
(the winner), reports/tables/stage_dev/<stage>_{hits,summary}.csv and rows of reports/tables/stages.csv.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

from .baseline import MODELS_DIR, load_copies
from .clustering import knee_distance
from .features import FEATURE_IDS, PACE_ID
from .methods import DIRECTIONS, fit_rows, real_metrics, stability
from .regression import INPUTS, TARGETS, conformal_margin, grids, make_model, targets, time_ordered_validation
from .settings import SETTINGS
from .snapshot import MANIFEST_PATH
from .split import EVAL
from .stages import DEV_DIR, PATTERN_MEASURES, TABLES_DIR, dev_hits, m0_spread, run_compare

log = logging.getLogger(__name__)

SEARCH_DIR = TABLES_DIR / "stage_search"
STAGE_MODELS_DIR = MODELS_DIR / "stage"
STAGES = {"C1": ("baseline",), "C2": ("isolation_forest",), "C3": ("lof",), "C4": ("kmeans",), "C5": ("dbscan",),
          "C6": ("regression", "regression_interval")}
STABLE = {"isolation_forest", "lof"}  # methods whose M0 choice also needed the stability target
REGRESSION_BASE = [c for c in INPUTS if c != PACE_ID]  # regression.INPUTS without stage A0's F9


@dataclass
class Data:
    fit: pd.DataFrame     # real fit-period rows (every candidate is fitted on these)
    real: pd.DataFrame    # real evaluation rows (the flag budget and the sanity rule)
    copies: pd.DataFrame  # evaluation rows of every tuning copy (the winner's dev hits)
    search: pd.DataFrame  # the copies of the search seeds only (ranking)


@dataclass
class Candidate:
    params: dict[str, Any]
    make: Callable[[], Any]  # the fitted Method
    scores: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None  # (real, search) scores without a Method
    refit: Callable[[int, pd.DataFrame], Any] | None = None           # (seed, rows) -> fitted Method: stability


def load_data() -> Data:
    features = pd.read_csv(MANIFEST_PATH.parent / "features.csv")
    copies = load_copies("refined", "tuning")
    copies = copies[copies["period"] == EVAL].reset_index(drop=True)
    search = copies[copies["seed"].isin(SETTINGS["stageSearch"]["searchSeeds"])].reset_index(drop=True)
    return Data(fit_rows(features), features[features["period"] == EVAL].reset_index(drop=True), copies, search)


def pooled_inputs(frame: pd.DataFrame, input_set: str) -> np.ndarray:
    """methods.model_inputs for one of stageSearch.inputSets: clipped z-scores (0 when missing), and the direction
    one-hot when the set has it."""
    spec = SETTINGS["stageSearch"]["inputSets"][input_set]
    z = frame[[f"zc_{f}" for f in spec["features"]]].astype("float64").fillna(0.0).to_numpy()
    if not spec["direction"]:
        return z
    return np.hstack([z, np.stack([(frame["direction"] == d).to_numpy(dtype="float64") for d in DIRECTIONS], axis=1)])


def budget_cut(scores: np.ndarray, scorable: np.ndarray, budget: float) -> float:
    """The lowest threshold that flags at most floor(budget x rows) scorable rows; rows tied at the cut are all left
    out rather than letting the budget be exceeded."""
    s = np.where(scorable, np.nan_to_num(scores, nan=-np.inf), -np.inf)
    k = int(np.floor(budget * len(s)))
    values = np.sort(s[np.isfinite(s)])[::-1]
    if k == 0 or not len(values):
        return float("inf")
    cut = values[min(k, len(values)) - 1]
    if np.sum(s >= cut) > k:
        higher = values[values > cut]
        cut = higher[-1] if len(higher) else np.inf
    return float(cut)


def recalls(copies: pd.DataFrame, flagged: np.ndarray) -> dict[str, float]:
    """Per pattern, the dev measure of stages.PATTERN_MEASURES over the plants in `copies`."""
    planted = (copies["is_synthetic"] == 1).to_numpy()
    per_plant = (pd.DataFrame({"pattern": copies["pattern"].to_numpy()[planted],
                               "plant": copies["plant_id"].to_numpy()[planted], "hit": flagged[planted]})
                 .groupby(["pattern", "plant"])["hit"].agg(["max", "mean"]))
    return {p: float(per_plant.loc[p, m].mean()) for p, m in PATTERN_MEASURES.items()}


def hit_recalls(hits: pd.DataFrame) -> dict[str, float]:
    """The same measures from stages.dev_hits rows (one row per plant)."""
    return {p: float(hits.loc[hits["pattern"] == p, m].mean()) for p, m in PATTERN_MEASURES.items()}


def m0_reference(method: str, before: str = "M0") -> dict[str, Any]:
    """The method at the last accepted stage: its flag budget, largest address share, search-seed recalls and seed
    spread (the gate's per-pattern rule)."""
    summary = pd.read_csv(DEV_DIR / f"{before}_summary.csv").set_index("method").loc[method]
    hits = pd.read_csv(DEV_DIR / f"{before}_hits.csv")
    hits = hits[(hits["method"] == method) & hits["seed"].isin(SETTINGS["stageSearch"]["searchSeeds"])]
    return {"budget": min(float(summary["flag_rate"]), SETTINGS["sanity"]["maxFlaggedShare"]),
            "max_address_share": float(summary["max_address_share"]), "recalls": hit_recalls(hits),
            "spread": m0_spread(method)}


def evaluate(data: Data, real_scores: np.ndarray, search_scores: np.ndarray, ref: dict[str, Any]) -> dict[str, Any]:
    """One candidate at the flag budget: threshold, flag rate and sanity on the real rows; recalls and PR-AUC on the
    search plants; and whether the gate's sanity and per-pattern rules could pass."""
    limit = SETTINGS["sanity"]["maxFlaggedShare"]
    scorable = data.real["scorable"].to_numpy(dtype=bool)
    cut = budget_cut(real_scores, scorable, ref["budget"])
    flagged = scorable & (np.nan_to_num(real_scores, nan=-np.inf) >= cut)
    share = pd.Series(flagged).groupby(data.real["address"].to_numpy()).mean()
    search_scorable = data.search["scorable"].to_numpy(dtype=bool)
    s = np.nan_to_num(search_scores, nan=-np.inf)
    rec = recalls(data.search, search_scorable & (s >= cut))
    ranked = np.clip(np.where(search_scorable, s, -np.inf), -1e9, 1e9)
    row = {"threshold": cut, "flag_rate": float(flagged.mean()), "flagged": int(flagged.sum()),
           "max_address_share": float(share.max()), **{f"recall_{p}": r for p, r in rec.items()},
           "mean_recall": float(np.mean(list(rec.values()))),
           "pr_auc": float(average_precision_score(data.search["is_synthetic"], ranked))}
    row["passes_sanity"] = row["flag_rate"] <= limit and row["max_address_share"] <= max(limit, ref["max_address_share"])
    row["passes_patterns"] = all(ref["recalls"][p] - rec[p] <= ref["spread"].get(p, 0.0) for p in PATTERN_MEASURES)
    return row


def candidate_stability(c: Candidate, data: Data) -> float:
    """outliers.bootstrap_stability for a candidate: mean pairwise Jaccard of the top-k real evaluation rows over
    refits with different seeds on bootstrap resamples of the fit period."""
    m = SETTINGS["model"]
    ev = data.real[data.real["scorable"].astype(bool)]
    keys = (ev["address"] + ":" + ev["txid"]).to_numpy()
    sets = []
    for i in range(m["stabilitySeeds"]):
        for j in range(m["stabilityBootstraps"]):
            sample = data.fit.sample(frac=1.0, replace=True, random_state=m["seed"] + 100 * i + j)
            scores = c.refit(m["seed"] + i, sample).score(ev)
            sets.append(set(keys[np.argsort(-scores, kind="stable")[: m["stabilityTopK"]]]))
    return stability(sets)


# --- the candidates' Methods ------------------------------------------------------------------------------------


class Combined:
    """C1: the baseline's |z| over F1-F4 combined by `rule`; "max" is score_baseline itself (A1, M0)."""

    def __init__(self, rule: str):
        self.rule, self.threshold = rule, float("inf")

    def fit(self, rows: pd.DataFrame) -> Combined:
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        if self.rule == "max":
            return frame["score_baseline"].to_numpy(dtype="float64")
        z = -np.sort(-np.abs(frame[[f"z_{f}" for f in FEATURE_IDS]].astype("float64").fillna(0.0).to_numpy()), axis=1)
        return {"top2": z[:, :2].sum(axis=1), "l2": np.sqrt((z ** 2).sum(axis=1)), "sum": z.sum(axis=1)}[self.rule]


class Pooled:
    """C2-C5: one pooled scikit-learn model on an input set (higher score = more unusual).

    isolation_forest and lof: -score_samples (LOF in novelty mode, optionally after a scaler); kmeans: distance to
    the nearest centroid, or that distance over the cluster's median fit-period distance ("scaled"); dbscan: eps is
    eps_factor x the knee of the MinPts = 2D neighbour distances (clustering.py), and the score is the distance to
    the nearest core sample; hdbscan: distance to the nearest clustered fit-period point.
    """

    def __init__(self, kind: str, params: dict[str, Any], input_set: str, seed: int):
        self.kind, self.params, self.input_set, self.seed = kind, dict(params), input_set, seed
        self.threshold = float("inf")  # set at the flag budget by the search

    def fit(self, rows: pd.DataFrame) -> Pooled:
        p, x, n_jobs = dict(self.params), pooled_inputs(rows, self.input_set), SETTINGS["classification"]["nJobs"]
        self.scaler = {"none": None, "standard": StandardScaler(), "robust": RobustScaler()}[p.pop("scaler", "none")]
        if self.scaler is not None:
            x = self.scaler.fit_transform(x)
        if self.kind == "isolation_forest":
            self.model = IsolationForest(**p, random_state=self.seed, n_jobs=n_jobs).fit(x)
        elif self.kind == "lof":
            self.model = LocalOutlierFactor(**p, novelty=True, n_jobs=n_jobs).fit(x)
        elif self.kind == "kmeans":
            self.model = KMeans(n_clusters=p["k"], init=p["init"], n_init=SETTINGS["clustering"]["kmeansNInit"],
                                random_state=self.seed).fit(x)
            distance = self.model.transform(x).min(axis=1)
            self.scale = np.array([max(float(np.median(distance[self.model.labels_ == c])), 1e-9)
                                   if np.any(self.model.labels_ == c) else 1.0 for c in range(p["k"])])
        elif self.kind == "dbscan":
            self.eps = p["eps_factor"] * knee_distance(x, SETTINGS["clustering"]["dbscanMinPtsPerDimension"] * x.shape[1])
            core = DBSCAN(eps=self.eps, min_samples=p["min_samples"]).fit(x).core_sample_indices_
            if not len(core):
                raise ValueError(f"no core samples at eps {self.eps:.3f}")
            self.index = NearestNeighbors(n_neighbors=1).fit(x[core])
        elif self.kind == "hdbscan":
            kept = HDBSCAN(min_cluster_size=p["min_cluster_size"]).fit(x).labels_ != -1
            if not kept.any():
                raise ValueError("every fit-period point is noise")
            self.index = NearestNeighbors(n_neighbors=1).fit(x[kept])
        else:
            raise ValueError(self.kind)
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        x = pooled_inputs(frame, self.input_set)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        if self.kind in ("isolation_forest", "lof"):
            return -self.model.score_samples(x)
        if self.kind == "kmeans":
            d = self.model.transform(x)
            nearest = d.argmin(axis=1)
            distance = d[np.arange(len(d)), nearest]
            return distance / self.scale[nearest] if self.params["score"] == "scaled" else distance
        return self.index.kneighbors(x)[0][:, 0]


def regression_design(frame: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    """regression.design with stage C6's input set: the M0 inputs plus `extra` (F9 or nothing)."""
    x = frame.assign(dir_in=(frame["direction"] == "in").astype(float), dir_out=(frame["direction"] == "out").astype(float))
    return x[REGRESSION_BASE + extra].astype("float64")


def residual_score(frame: pd.DataFrame, predictions: dict[str, np.ndarray], scales: dict[str, dict[str, float]],
                   default_scale: dict[str, float]) -> np.ndarray:
    """regression.RegressionMethod's score: max |y - y_hat| / s over the targets (a missing gap counts as 0)."""
    ys, r = targets(frame), []
    for t, pred in predictions.items():
        s = frame["address"].map(scales[t]).fillna(default_scale[t]).to_numpy()
        r.append(np.nan_to_num(np.abs((ys[t].to_numpy() - pred) / s), nan=0.0))
    return np.max(np.vstack(r), axis=0)


class RegressionCandidate:
    """C6: regression.RegressionMethod with a chosen family per target, input set and residual floor."""

    def __init__(self, models: dict[str, Any], scales: dict[str, dict[str, float]], default_scale: dict[str, float],
                 extra: list[str]):
        self.models, self.scales, self.default_scale, self.extra = models, scales, default_scale, extra
        self.threshold = float("inf")

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        x = regression_design(frame, self.extra)
        return residual_score(frame, {t: m.predict(x) for t, m in self.models.items()}, self.scales, self.default_scale)


class IntervalCandidate:
    """C6: regression.RegressionIntervalMethod with a quantile pair, input set and conformal calibration share: the
    quantile models are fitted on the first part of each address's fit period, the margin on the last `share`."""

    def __init__(self, extra: list[str], pair: list[float], share: float, hgb: dict[str, dict[str, Any]], seed: int):
        self.extra, self.pair, self.share, self.hgb, self.seed = extra, pair, share, hgb, seed
        self.quantiles: dict[str, tuple[Any, Any]] = {}
        self.margins: dict[str, float] = {}
        self.threshold = float("inf")

    def fit(self, rows: pd.DataFrame) -> IntervalCandidate:
        for t in TARGETS:
            y_all = targets(rows)[t]
            r, y = rows[y_all.notna()], y_all[y_all.notna()]
            cal = time_ordered_validation(r, self.share)
            x = regression_design(r, self.extra)
            lo, hi = (make_model("histGradientBoosting", {**self.hgb[t], "loss": "quantile", "quantile": q}, self.seed)
                      .fit(x[~cal], y[~cal]) for q in self.pair)
            self.quantiles[t] = (lo, hi)
            self.margins[t] = conformal_margin(lo.predict(x[cal]), hi.predict(x[cal]), y[cal].to_numpy(),
                                               self.pair[1] - self.pair[0])
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        x, ys = regression_design(frame, self.extra), targets(frame)
        worst = np.full(len(frame), -np.inf)
        for t, (lo_model, hi_model) in self.quantiles.items():
            lo, hi = lo_model.predict(x) - self.margins[t], hi_model.predict(x) + self.margins[t]
            width, y = np.maximum(hi - lo, 1e-9), ys[t].to_numpy()
            worst = np.fmax(worst, np.nan_to_num(np.fmax((lo - y) / width, (y - hi) / width), nan=-np.inf))
        return worst


# --- the grids ----------------------------------------------------------------------------------------------------


def pooled_candidate(kind: str, params: dict[str, Any], input_set: str, data: Data) -> Candidate:
    def refit(seed: int, rows: pd.DataFrame) -> Pooled:
        return Pooled(kind, params, input_set, seed).fit(rows)

    return Candidate({"input_set": input_set, "model": kind, **params},
                     make=lambda: refit(SETTINGS["model"]["seed"], data.fit), refit=refit)


def pooled_candidates(kinds: tuple[str, ...], data: Data) -> Iterator[Candidate]:
    s = SETTINGS["stageSearch"]
    keys = {"isolation_forest": "isolationForest", "lof": "lof", "kmeans": "kmeans", "dbscan": "dbscan", "hdbscan": "hdbscan"}
    for input_set in s["inputSets"]:
        for kind in kinds:
            for params in ParameterGrid(s[keys[kind]]):
                yield pooled_candidate(kind, params, input_set, data)


def regression_fits(data: Data) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Per (input set, target, family): the family's parameters chosen by validation MAE on the last part of each
    address's fit period (as regression.run_regression does at M0), refitted on the whole fit period, with its
    per-address residual MAD and its predictions for the real evaluation rows and the search copies."""
    r, s, seed = SETTINGS["regression"], SETTINGS["stageSearch"]["regression"], SETTINGS["model"]["seed"]
    fits = {}
    for name, extra in s["inputSets"].items():
        x_real, x_search = regression_design(data.real, extra), regression_design(data.search, extra)
        for t in TARGETS:
            y_all = targets(data.fit)[t]
            rows, y = data.fit[y_all.notna()], y_all[y_all.notna()]
            is_val = time_ordered_validation(rows, r["validationFraction"]).to_numpy()
            x = regression_design(rows, extra)
            for family in s["families"]:
                configs = grids()[family]
                maes = [mean_absolute_error(y[is_val], make_model(family, p, seed).fit(x[~is_val], y[~is_val])
                                            .predict(x[is_val])) for p in configs]
                params = configs[int(np.argmin(maes))]
                model = make_model(family, params, seed).fit(x, y)
                resid = pd.Series(y.to_numpy() - model.predict(x), index=rows.index)
                mad = resid.groupby(rows["address"]).apply(lambda v: float(np.median(np.abs(v - np.median(v)))))
                fits[(name, t, family)] = {"params": params, "val_mae": float(min(maes)), "model": model, "mad": mad,
                                           "real": model.predict(x_real), "search": model.predict(x_search)}
                log.info("regression %s / %s / %s: validation MAE %.4f with %s", name, t, family, min(maes), params)
    return fits


def regression_candidates(data: Data, fits: dict[tuple[str, str, str], dict[str, Any]]) -> Iterator[Candidate]:
    s = SETTINGS["stageSearch"]["regression"]
    for name, extra in s["inputSets"].items():
        for size_family in s["families"]:
            for gap_family in s["families"]:
                chosen = {"size": fits[(name, "size", size_family)], "gap": fits[(name, "gap", gap_family)]}
                for floor in s["residualFloor"]:
                    scales = {t: (SETTINGS["baseline"]["madScale"] * f["mad"]).clip(lower=floor).to_dict()
                              for t, f in chosen.items()}
                    default = {t: float(np.median(list(v.values()))) for t, v in scales.items()}

                    def scores(chosen=chosen, scales=scales, default=default):
                        return tuple(residual_score(frame, {t: f[part] for t, f in chosen.items()}, scales, default)
                                     for frame, part in ((data.real, "real"), (data.search, "search")))

                    def make(chosen=chosen, scales=scales, default=default, extra=extra):
                        return RegressionCandidate({t: f["model"] for t, f in chosen.items()}, scales, default, extra)

                    yield Candidate({"input_set": name, "size": size_family, "gap": gap_family, "residual_floor": floor,
                                     "size_params": chosen["size"]["params"], "gap_params": chosen["gap"]["params"]},
                                    make=make, scores=scores)


def interval_candidates(data: Data, fits: dict[tuple[str, str, str], dict[str, Any]]) -> Iterator[Candidate]:
    s, seed = SETTINGS["stageSearch"], SETTINGS["model"]["seed"]
    for name, extra in s["regression"]["inputSets"].items():
        hgb = {t: fits[(name, t, "histGradientBoosting")]["params"] for t in TARGETS}
        for pair in s["regressionInterval"]["quantiles"]:
            for share in s["regressionInterval"]["calibrationShare"]:
                yield Candidate({"input_set": name, "quantiles": pair, "calibration_share": share, "hgb": hgb},
                                make=lambda extra=extra, pair=pair, share=share, hgb=hgb:
                                IntervalCandidate(extra, pair, share, hgb, seed).fit(data.fit))


def candidates(method: str, data: Data, fits: dict | None) -> list[Candidate]:
    if method == "baseline":
        return [Candidate({"combine": rule}, make=lambda rule=rule: Combined(rule))
                for rule in SETTINGS["stageSearch"]["baseline"]["combine"]]
    if method == "regression":
        return list(regression_candidates(data, fits))
    if method == "regression_interval":
        return list(interval_candidates(data, fits))
    return list(pooled_candidates(("dbscan", "hdbscan") if method == "dbscan" else (method,), data))


# --- the search ---------------------------------------------------------------------------------------------------


def search(method: str, pool: list[Candidate], data: Data, ref: dict[str, Any]) -> tuple[pd.DataFrame, int, str]:
    """Score every candidate at the flag budget; return the table, the winner's index and the rule that chose it."""
    rows = []
    for i, c in enumerate(pool):
        try:
            if c.scores is not None:
                real_scores, search_scores = c.scores()
            else:
                fitted = c.make()
                real_scores, search_scores = fitted.score(data.real), fitted.score(data.search)
        except ValueError as e:
            log.info("%s %s skipped: %s", method, c.params, e)
            continue
        rows.append({"candidate": i, "params": json.dumps(c.params), **evaluate(data, real_scores, search_scores, ref)})
        r = rows[-1]
        log.info("%s %s: mean recall %.3f (spike %.3f, burst %.3f, dormant %.3f), flag rate %.3f, max share %.3f%s",
                 method, c.params, r["mean_recall"], r["recall_spike"], r["recall_burst"], r["recall_dormant"],
                 r["flag_rate"], r["max_address_share"], "" if r["passes_sanity"] and r["passes_patterns"] else " (filtered)")
    table = pd.DataFrame(rows)
    table["passes"] = table["passes_sanity"] & table["passes_patterns"]
    table["stability"] = np.nan
    ranked = table.sort_values(["passes", "mean_recall", "pr_auc"], ascending=False)
    passing = ranked[ranked["passes"]]
    if not len(passing):
        winner, rule = ranked.index[0], "no candidate passed the sanity and per-pattern rules; highest search recall"
    elif method not in STABLE:
        winner, rule = passing.index[0], "highest search recall among candidates passing the sanity and per-pattern rules"
    else:
        target, winner = SETTINGS["model"]["stabilityTarget"], None
        for idx in passing.index[: SETTINGS["stageSearch"]["stabilityChecks"]]:
            table.loc[idx, "stability"] = candidate_stability(pool[int(table.loc[idx, "candidate"])], data)
            log.info("%s candidate %d: stability %.3f", method, table.loc[idx, "candidate"], table.loc[idx, "stability"])
            if table.loc[idx, "stability"] >= target:
                winner, rule = idx, "highest search recall among passing candidates that reach the stability target"
                break
        if winner is None:
            winner, rule = passing.index[0], "no checked candidate reached the stability target; highest search recall"
    table["chosen"] = table.index == winner
    return table, int(table.loc[winner, "candidate"]), rule


def run_stage(stage: str, data: Data, before: str = "M0") -> int:
    """Search every method of one stage, save the winners and their dev hits, and gate them against `before` on the
    gate seeds."""
    s = SETTINGS["stageSearch"]
    fits = regression_fits(data) if "regression" in STAGES[stage] else None
    all_hits, summaries = [], []
    for method in STAGES[stage]:
        ref = m0_reference(method, before)
        pool = candidates(method, data, fits)
        log.info("%s %s: %d candidates at a flag budget of %.4f", stage, method, len(pool), ref["budget"])
        table, best, rule = search(method, pool, data, ref)
        SEARCH_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(SEARCH_DIR / f"{stage}_{method}.csv", index=False)
        row = table[table["chosen"]].iloc[0]
        winner = pool[best].make()
        winner.threshold = float(row["threshold"])
        hits = dev_hits({method: winner}, data.copies)
        check = hit_recalls(hits[hits["seed"].isin(s["searchSeeds"])])
        if any(abs(check[p] - row[f"recall_{p}"]) > 1e-9 for p in PATTERN_MEASURES):
            raise RuntimeError(f"{method}: the winner's recalls {check} differ from its search row")
        all_hits.append(hits)
        every = hit_recalls(hits)
        summaries.append({"stage": stage, "method": method, **{f"recall_{p}": r for p, r in every.items()},
                          "mean_recall": float(np.mean(list(every.values()))), **real_metrics(winner, data.real),
                          "budget": ref["budget"], "params": pool[best].params, "rule": rule})
        STAGE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"method": winner, "stage": stage, "params": pool[best].params, "rule": rule,
                     "budget": ref["budget"], "threshold": winner.threshold}, STAGE_MODELS_DIR / f"{stage}_{method}.joblib",
                    compress=3)
        log.info("%s %s: chose %s (%s); search recall %.3f vs %s %.3f", stage, method, pool[best].params, rule,
                 row["mean_recall"], before, float(np.mean(list(ref["recalls"].values()))))
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(all_hits, ignore_index=True).assign(stage=stage).to_csv(DEV_DIR / f"{stage}_hits.csv", index=False)
    pd.DataFrame(summaries).to_csv(DEV_DIR / f"{stage}_summary.csv", index=False)
    return run_compare(before, stage, stage, seeds=s["gateSeeds"])


def run_search(stages: list[str]) -> int:
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise KeyError(f"unknown stages {unknown}; known: {list(STAGES)}")
    data = load_data()
    log.info("Fit %d rows, real evaluation %d rows, tuning copies %d evaluation rows (%d on the search seeds)",
             len(data.fit), len(data.real), len(data.copies), len(data.search))
    for stage in stages:
        run_stage(stage, data)
    return 0
