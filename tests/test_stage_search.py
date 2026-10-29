"""Stages C1-C6 (explorer/stage_search.py): the flag budget, the recalls, the candidates' Methods and the choice."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from explorer import stage_search as ss
from explorer.features import FEATURE_IDS


def frame(n: int, seed: int = 0) -> pd.DataFrame:
    """A synthetic stand-in for features.csv rows: every column the candidates read."""
    rng = np.random.default_rng(seed)
    out = pd.DataFrame({f"zc_{f}": rng.normal(size=n) for f in FEATURE_IDS + ("F9",)})
    for f in FEATURE_IDS:
        out[f"z_{f}"] = out[f"zc_{f}"] * 2
    out["score_baseline"] = out[[f"z_{f}" for f in FEATURE_IDS]].abs().max(axis=1)
    out["direction"] = rng.choice(["in", "out"], size=n)
    out["address"] = rng.choice(["a", "b", "c"], size=n)
    out["block_time_utc"] = np.arange(n)
    out["scorable"] = True
    for c in ss.REGRESSION_BASE + ["F9"]:
        if c not in ("dir_in", "dir_out"):
            out[c] = rng.normal(size=n)
    out["F1"] = rng.normal(5, 1, size=n)
    out["F3"] = rng.exponential(24, size=n)
    return out


def test_budget_cut_never_exceeds_the_budget():
    scores = np.array([5.0, 4.0, 4.0, 3.0, 2.0, 1.0, np.nan, 9.0])
    scorable = np.array([True] * 7 + [False])
    cut = ss.budget_cut(scores, scorable, 0.25)  # 2 of 8 rows: 5 and one 4 would be 2, but the 4s tie
    assert cut == 5.0 and np.sum(scorable & (scores >= cut)) == 1
    cut = ss.budget_cut(scores, scorable, 0.5)  # 4 rows: 5, 4, 4, 3
    assert cut == 3.0
    assert ss.budget_cut(scores, scorable, 0.0) == float("inf")


def test_recalls_use_each_patterns_measure():
    copies = pd.DataFrame({"is_synthetic": [1, 1, 1, 1, 1, 0], "pattern": ["spike", "burst", "burst", "dormant", "dormant", "spike"],
                           "plant_id": ["s1", "b1", "b1", "d1", "d2", "s1"]})
    r = ss.recalls(copies, np.array([True, True, False, False, True, True]))
    assert r == {"spike": 1.0, "burst": 0.5, "dormant": 0.5}


def test_combined_max_is_the_baseline_score():
    f = frame(50)
    assert np.allclose(ss.Combined("max").score(f), f["score_baseline"])
    z = f[[f"z_{x}" for x in FEATURE_IDS]].abs().to_numpy()
    assert np.allclose(ss.Combined("sum").score(f), z.sum(axis=1))
    assert np.all(ss.Combined("top2").score(f) >= ss.Combined("max").score(f))


@pytest.mark.parametrize("kind, params", [
    ("isolation_forest", {"n_estimators": 20, "max_samples": 64, "max_features": 1.0}),
    ("lof", {"n_neighbors": 10, "metric": "manhattan", "scaler": "robust"}),
    ("kmeans", {"k": 3, "init": "random", "score": "scaled"}),
    ("dbscan", {"eps_factor": 1.0, "min_samples": 8}),
    ("hdbscan", {"min_cluster_size": 10}),
])
def test_pooled_candidates_score_new_rows(kind, params):
    fit, new = frame(300, 1), frame(40, 2)
    new.loc[0, [f"zc_{f}" for f in FEATURE_IDS + ("F9",)]] = 15.0  # far from everything seen in the fit rows
    m = ss.Pooled(kind, params, "F1-F4+F9+dir", seed=3).fit(fit)
    scores = m.score(new)
    assert scores.shape == (40,) and np.isfinite(scores).all()
    assert scores[0] == scores.max()


def test_regression_candidates_match_their_arrays():
    from explorer.regression import make_model, targets

    fit, new = frame(200, 4), frame(30, 5)
    models, scales, default = {}, {}, {}
    for t in ("size", "gap"):
        y = targets(fit)[t]
        models[t] = make_model("ridge", {"alpha": 1.0}, 0).fit(ss.regression_design(fit, ["F9"]), y)
        scales[t], default[t] = {"a": 0.5, "b": 1.0}, 0.7
    m = ss.RegressionCandidate(models, scales, default, ["F9"])
    x = ss.regression_design(new, ["F9"])
    assert np.allclose(m.score(new), ss.residual_score(new, {t: models[t].predict(x) for t in models}, scales, default))
    interval = ss.IntervalCandidate([], [0.05, 0.95], 0.2, {"size": {}, "gap": {}}, 0).fit(fit)
    assert set(interval.margins) == {"size", "gap"} and np.isfinite(interval.score(new)).any()


def test_search_ranks_passing_candidates_at_the_budget():
    real = frame(100, 6)
    search = frame(200, 7)
    search["is_synthetic"] = (np.arange(200) < 20).astype(int)
    search["pattern"] = np.where(np.arange(200) % 3 == 0, "spike", np.where(np.arange(200) % 3 == 1, "burst", "dormant"))
    search["plant_id"] = [f"p{i % 10}" for i in range(200)]
    search.loc[search["is_synthetic"] == 1, "score_baseline"] += 100  # every rule but "sum" finds them; all do
    data = ss.Data(fit=real, real=real, copies=search, search=search)
    ref = {"budget": 0.1, "max_address_share": 1.0, "recalls": {"spike": 0, "burst": 0, "dormant": 0}, "spread": {}}
    pool = [ss.Candidate({"combine": r}, make=lambda r=r: ss.Combined(r)) for r in ("sum", "max")]
    table, best, rule = ss.search("baseline", pool, data, ref)
    assert (table["flag_rate"] <= 0.1).all()
    assert best == 1 and table.loc[table["chosen"], "mean_recall"].iloc[0] == table["mean_recall"].max()
    assert "passing" in rule


def test_refit_keeps_a_stage_winners_configuration_and_threshold():
    # evaluate.refit (stability, walk-forward) must refit a promoted winner, not reject it (settings stageModels).
    from explorer.evaluate import refit

    rows = frame(300)
    winner = ss.Pooled("kmeans", {"k": 2, "init": "k-means++", "score": "distance"}, "F1-F4+F9+dir", 1).fit(rows)
    winner.threshold = 1.5
    again = refit("kmeans", winner, rows.sample(frac=1.0, replace=True, random_state=0), 7)
    assert isinstance(again, ss.Pooled) and again.threshold == 1.5 and again.seed == 7
    assert again.input_set == winner.input_set and again.params == winner.params
    assert np.isfinite(again.score(rows)).all()


def test_saved_methods_loads_the_accepted_stage(monkeypatch):
    import tempfile
    from pathlib import Path

    import joblib

    from explorer import scores

    with tempfile.TemporaryDirectory() as tmp:
        models = Path(tmp)
        (models / "stage").mkdir()
        joblib.dump({"method": "m0 model"}, models / "kmeans.joblib")
        joblib.dump({"method": "stage model"}, models / "stage" / "C4_kmeans.joblib")
        monkeypatch.setitem(scores.SETTINGS, "stageModels", {"_about": "x", "kmeans": "C4"})
        assert scores.saved_methods(models)["kmeans"] == "stage model"
        assert scores.saved_methods(models, promoted=False)["kmeans"] == "m0 model"  # fusion.py's own search
        monkeypatch.setitem(scores.SETTINGS, "stageModels", {})
        assert scores.saved_methods(models)["kmeans"] == "m0 model"
        monkeypatch.setitem(scores.SETTINGS, "stageModels", {"lof": "C3"})  # accepted, but its winner is missing
        with pytest.raises(FileNotFoundError):
            scores.saved_methods(models)


def test_fusion_candidates_score_as_their_method(monkeypatch):
    # C9: the cached arrays used for ranking must equal the winner's FusionMethod scores bit for bit.
    from explorer import scores as scores_module
    from explorer.fusion import FusionMethod

    rows = frame(400)
    data = ss.Data(fit=rows, real=rows.iloc[:200].reset_index(drop=True), copies=rows, search=rows.iloc[200:].reset_index(drop=True))
    a = ss.Pooled("kmeans", {"k": 2, "init": "k-means++", "score": "distance"}, "F1-F4+dir", 1).fit(rows)
    b = ss.Pooled("lof", {"n_neighbors": 20, "metric": "euclidean", "scaler": "none"}, "F1-F4", 1).fit(rows)
    a.threshold, b.threshold = 1.2, 1.7
    monkeypatch.setattr(scores_module, "saved_methods", lambda: {"kmeans": a, "lof": b})
    pool = list(ss.fusion_candidates(data))
    assert len(pool) == len(ss.SETTINGS["stageSearch"]["fusion"]["ratios"])
    c = pool[2]
    real, search = c.scores()
    method = c.make()
    assert isinstance(method, FusionMethod)
    assert np.array_equal(real, method.score(data.real), equal_nan=True)
    assert np.array_equal(search, method.score(data.search), equal_nan=True)
