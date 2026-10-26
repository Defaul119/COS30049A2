"""Tests for the DS1 extensions (24 Oct): conformal ranges, score fusion, bootstrap intervals, the strength
sweep, the DS2 check and walk-forward refits.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from explorer import ds2_check, evaluate, fusion, regression, sensitivity, significance
from explorer.baseline import Baseline
from explorer.methods import fit_rows, flags
from explorer.outliers import IsolationForestMethod
from explorer.settings import SETTINGS
from tests.test_methods import MethodChecksTests


def setUpModule():
    MethodChecksTests.setUpClass()  # synthetic histories with features, periods and planted copies


class Fixed:
    """A method whose scores are a column of the frame."""

    def __init__(self, column: str, threshold: float):
        self.column, self.threshold = column, threshold

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.column].to_numpy(dtype="float64")


class ConformalTests(unittest.TestCase):
    def test_a_too_narrow_range_is_widened_until_it_covers_the_level(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, 2000)
        lo, hi = np.full_like(y, -0.5), np.full_like(y, 0.5)  # covers about 38%
        margin = regression.conformal_margin(lo, hi, y, 0.9)
        self.assertGreater(margin, 0)
        self.assertGreaterEqual(np.mean((y >= lo - margin) & (y <= hi + margin)), 0.9)

    def test_a_too_wide_range_gets_a_negative_margin(self):
        y = np.random.default_rng(2).normal(0, 1, 2000)
        self.assertLess(regression.conformal_margin(np.full_like(y, -5.0), np.full_like(y, 5.0), y, 0.9), 0)

    def test_the_margin_widens_the_flagging_range(self):
        class Constant:
            def __init__(self, value: float):
                self.value = value

            def predict(self, x: pd.DataFrame) -> np.ndarray:
                return np.full(len(x), self.value)

        frame = MethodChecksTests.features.head(20)
        pair = {"size": (Constant(0.0), Constant(1.0))}
        raw = regression.RegressionIntervalMethod(pair).score(frame)
        wide = regression.RegressionIntervalMethod(pair, {"size": 10.0}).score(frame)
        self.assertTrue((wide <= raw).all())
        self.assertTrue((wide <= 0).all())  # nothing lies outside a range ten units wider on each side


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"a": [0.0, 6.0, 1.0, 4.0], "b": [0.0, 1.0, 9.0, 4.0], "scorable": True})

    def test_flagged_when_any_component_reaches_its_multiple_of_its_threshold(self):
        f = fusion.FusionMethod({"a": Fixed("a", 2.0), "b": Fixed("b", 3.0)}, {"a": 2.0, "b": 2.0})
        # a/(2*2) and b/(2*3): row 1 by a (1.5), row 2 by b (1.5), row 3 by neither (1.0 and 0.67)
        np.testing.assert_allclose(f.score(self.frame), [0.0, 1.5, 1.5, 1.0])
        self.assertEqual(flags(f, self.frame).tolist(), [False, True, True, True])

    def test_cached_scores_follow_the_frame_index(self):
        c = fusion.Cached(1.0, pd.Series([5.0, 7.0], index=[10, 20]))
        self.assertEqual(c.score(pd.DataFrame(index=[20, 10])).tolist(), [7.0, 5.0])

    def test_the_rule_keeps_to_the_budget_then_recall_then_pr_auc(self):
        table = pd.DataFrame([
            {"mean_recall": 0.9, "pr_auc": 0.1, "within_budget": False},
            {"mean_recall": 0.7, "pr_auc": 0.1, "within_budget": True},
            {"mean_recall": 0.7, "pr_auc": 0.3, "within_budget": True},
            {"mean_recall": 0.6, "pr_auc": 0.9, "within_budget": True}])
        best = fusion.best_within_budget(table)
        self.assertEqual((best["mean_recall"], best["pr_auc"]), (0.7, 0.3))
        self.assertIsNone(fusion.best_within_budget(table.assign(within_budget=False)))

    def test_a_stable_pair_beats_a_better_but_unstable_one(self):
        table = pd.DataFrame([
            {"pair": "a + b", "components": 2, "mean_recall": 0.8, "pr_auc": 0.1, "within_budget": True, "stability": 0.6},
            {"pair": "a + c", "components": 2, "mean_recall": 0.7, "pr_auc": 0.1, "within_budget": True, "stability": 0.9},
            {"pair": "a", "components": 1, "mean_recall": 0.9, "pr_auc": 0.1, "within_budget": True, "stability": np.nan}])
        best, rule = fusion.choose(table, 0.7)
        self.assertEqual(best["pair"], "a + c")  # a single method is never saved as the fusion
        self.assertIn("among stable", rule)
        best, rule = fusion.choose(table.assign(stability=0.5), 0.7)
        self.assertEqual(best["pair"], "a + b")
        self.assertIn("no configuration", rule)

    def test_refit_refits_fitted_components_and_keeps_fixed_ones(self):
        features = MethodChecksTests.features
        rows = fit_rows(features)
        f = fusion.FusionMethod({"baseline": Baseline(5.0), "isolation_forest": IsolationForestMethod({"n_estimators": 50}, 1)
                                 .fit(rows)}, {"baseline": 1.0, "isolation_forest": 1.0})
        again = evaluate.refit("fusion", f, rows.sample(frac=1.0, replace=True, random_state=4), 7)
        self.assertIsInstance(again, fusion.FusionMethod)
        self.assertIs(again.components["baseline"], f.components["baseline"])
        self.assertIsNot(again.components["isolation_forest"], f.components["isolation_forest"])
        self.assertEqual(len(again.score(features)), len(features))


class SignificanceTests(unittest.TestCase):
    def hits(self) -> pd.DataFrame:
        rows = []
        for method, rate in (("good", 0.9), ("poor", 0.3)):
            for k in range(40):
                address = f"a{k % 8}"
                for pattern in ("spike", "burst", "dormant"):
                    hit = float(k % 10 < 10 * rate)
                    rows.append({"method": method, "address": address, "pattern": pattern, "plant_id": f"{pattern}{k}",
                                 "seed": 1, "max": hit, "mean": hit})
        return pd.DataFrame(rows)

    def test_the_point_estimate_is_the_pooled_share_and_draws_vary_around_it(self):
        hits = self.hits()
        addresses = sorted(hits["address"].unique())
        weights = np.random.default_rng(3).multinomial(len(addresses), np.full(len(addresses), 1 / len(addresses)), size=500)
        boot = significance.bootstrap_recalls(hits, weights, addresses)
        self.assertAlmostEqual(boot["good"]["recall_spike"][0], 0.9)
        self.assertAlmostEqual(boot["poor"]["mean_recall"][0], 0.3)
        table = significance.significance_table(boot, "poor", 0.95, 5).set_index("method")
        self.assertGreater(table.loc["good", "diff_ci_low"], 0)  # a 60-point gap is not noise
        self.assertLess(table.loc["good", "p_value"], 0.05)
        self.assertTrue(np.isnan(table.loc["poor", "p_value"]))

    def test_two_sided_p(self):
        self.assertEqual(significance.two_sided_p(np.array([1.0, 2.0, 3.0])), 0.0)
        self.assertEqual(significance.two_sided_p(np.array([-1.0, 1.0])), 1.0)


class SensitivityTests(unittest.TestCase):
    def test_the_sweep_contains_each_default_strength_once(self):
        defaults = [s for s in sensitivity.strengths() if sensitivity.is_default(s[3])]
        self.assertEqual(sorted(s[0] for s in defaults), ["burst", "dormant", "spike"])
        self.assertTrue(all(x > 0 for _, _, x, _ in sensitivity.strengths()))


class DS2CheckTests(unittest.TestCase):
    def test_only_histories_meeting_the_main_rule_are_kept_and_directions_follow_net(self):
        c = SETTINGS["collection"]
        day = 86_400
        rows = []
        for addr, n, span_days in (("long", c["mainMinTx"], c["mainMinSpanDays"] + 1), ("few", c["mainMinTx"] - 1, 400),
                                   ("brief", c["mainMinTx"] + 5, c["mainMinSpanDays"] - 1)):
            for i in range(n):
                rows.append({"addr_id": addr, "txid_hash": f"{addr}{i}", "block_height": i,
                             "block_time_utc": int(i * span_days * day / (n - 1)), "received_sats": 5 if i % 2 else 0,
                             "sent_sats": 0 if i % 2 else 5, "n_inputs": 1, "n_outputs": 2, "fee_sats": 100, "vsize_vb": 140})
        tx = ds2_check.ds2_transactions(pd.DataFrame(rows))
        self.assertEqual(set(tx["address"]), {"long"})
        self.assertEqual(set(tx["direction"]), {"in", "out"})
        self.assertTrue((tx["net_sats"] == tx["received_sats"] - tx["sent_sats"]).all())

    def test_regression_scales_come_from_the_new_addresses_own_fit_period(self):
        rows = fit_rows(MethodChecksTests.features)
        y = regression.targets(rows)["size"]
        model = regression.make_model("ridge", {"alpha": 1.0}, 1).fit(regression.design(rows), y)
        method = regression.RegressionMethod({"size": model}, {"size": {"elsewhere": 1.0}}, {"size": 1.0})
        adapted = ds2_check.adapt(method, rows)
        self.assertEqual(set(adapted.scales["size"]), set(rows["address"]))
        self.assertIs(ds2_check.adapt(Baseline(5.0), rows).__class__, Baseline)


if __name__ == "__main__":
    unittest.main()


class DistributionTests(unittest.TestCase):
    """explorer/distribution.py (27 Oct): every method's recall over many plant seeds."""

    def test_two_seeds_on_ds1_give_one_row_per_seed_and_method_and_a_summary(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        import pandas as pd

        from explorer import distribution
        from explorer.settings import SETTINGS

        with tempfile.TemporaryDirectory() as tmp:
            small = {**SETTINGS["distribution"], "seedCount": {"DS1": 2, "DS2": 0, "DS10": 0}}
            with mock.patch.dict(SETTINGS, {"distribution": small}), mock.patch.object(distribution, "TABLES_DIR", Path(tmp)):
                distribution.run_distribution(only=["DS1"], n_jobs=2)
            seeds = pd.read_csv(Path(tmp) / "stats_methods_seeds.csv")
            summary = pd.read_csv(Path(tmp) / "stats_methods_summary.csv")
        self.assertEqual(sorted(seeds["seed"].unique()), [small["firstSeed"], small["firstSeed"] + 1])
        self.assertEqual(len(seeds), 2 * seeds["method"].nunique())
        self.assertTrue(seeds["mean_recall"].between(0, 1).all())
        base = summary.set_index("method").loc["baseline"]
        self.assertEqual((base["seeds"], base["minus_baseline_mean"], base["beats_baseline_share"]), (2, 0.0, 0.0))
