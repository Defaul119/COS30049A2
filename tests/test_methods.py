"""Tests for the shared DS1 method checks (methods.py), the baseline sweep rule and IF/LOF (plan §9, §11.1).

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from explorer import baseline, inject, methods, outliers
from explorer.features import FeatureConfig, build_features
from explorer.settings import SETTINGS
from explorer.split import EVAL, FIT, assign_periods
from tests.test_inject import history


class Oracle:
    """Scores planted rows 1 and real rows 0: a perfect method, to check the checks themselves."""

    threshold = 0.5

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["is_synthetic"].to_numpy(dtype="float64") if "is_synthetic" in frame else np.zeros(len(frame))


class MethodChecksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tx = pd.concat([history("big", 150, 1), history("mid", 90, 2), history("small", 60, 3)], ignore_index=True)
        features = build_features(tx, FeatureConfig.refined())
        period, _ = assign_periods(features, SETTINGS["model"]["fitFraction"])
        cls.features = features.assign(period=period)
        periods = cls.features[["address", "txid", "period"]]
        p = {**SETTINGS["injection"], "plantsPerPattern": 4}
        plants = inject.make_plants(tx, periods, 40000, "tuning", p)
        cls.copies = pd.concat([inject.copy_features(tx, periods, pl, FeatureConfig.refined()) for pl in plants],
                               ignore_index=True)

    def test_a_perfect_method_scores_perfectly(self):
        m = methods.plant_metrics(Oracle(), self.copies)
        for key in ("recall_spike", "recall_burst_tx", "recall_burst_event", "recall_dormant", "precision_lower_bound", "pr_auc"):
            self.assertEqual(m[key], 1.0, key)
        self.assertTrue(methods.meets_recall_target(m))

    def test_unscorable_rows_are_never_flagged(self):
        always = type("Always", (), {"threshold": 0.0, "score": lambda self, f: np.ones(len(f))})()
        flagged = methods.flags(always, self.features)
        self.assertFalse(flagged[~self.features["scorable"].to_numpy(dtype=bool)].any())

    def test_model_inputs_are_clipped_z_plus_direction(self):
        x = methods.model_inputs(self.features)
        self.assertEqual(x.shape, (len(self.features), 6))
        self.assertLessEqual(np.abs(x[:, :4]).max(), SETTINGS["features"]["zClip"])
        self.assertTrue(set(np.unique(x[:, 4:])) <= {0.0, 1.0})

    def test_outlier_methods_flag_about_the_top_two_percent_of_the_fit_period(self):
        rows = methods.fit_rows(self.features)
        self.assertTrue((rows["period"] == FIT).all())
        for cls, params in ((outliers.IsolationForestMethod, {"n_estimators": 100}), (outliers.LOFMethod, {"n_neighbors": 20})):
            m = cls(params, 1).fit(rows)
            share = float(np.mean(m.score(rows) >= m.threshold)) if cls is outliers.IsolationForestMethod else \
                float(np.mean(-m.model.negative_outlier_factor_ >= m.threshold))
            self.assertAlmostEqual(share, SETTINGS["model"]["flagTopFraction"], delta=0.02, msg=cls.__name__)
            self.assertEqual(len(m.score(self.copies)), len(self.copies))

    def test_stability_and_sanity_measures(self):
        self.assertEqual(methods.stability([{"a", "b"}, {"a", "b"}]), 1.0)
        self.assertEqual(methods.stability([{"a"}, {"b"}]), 0.0)
        real = methods.real_metrics(Oracle(), self.features)
        self.assertEqual((real["flagged"], real["addresses_over_limit"]), (0, 0))
        self.assertEqual(real["eval_rows"], int((self.features["period"] == EVAL).sum()))


class SweepRuleTests(unittest.TestCase):
    def test_smallest_threshold_meeting_both_then_sanity_alone(self):
        rows = [{"threshold": 2.5, "meets_sanity": False, "meets_recall_target": True, "addresses_over_limit": 3},
                {"threshold": 3.0, "meets_sanity": True, "meets_recall_target": True, "addresses_over_limit": 0},
                {"threshold": 3.5, "meets_sanity": True, "meets_recall_target": True, "addresses_over_limit": 0}]
        self.assertEqual(baseline.choose_threshold(rows)[0], 3.0)
        for r in rows:
            r["meets_recall_target"] = False
        self.assertEqual(baseline.choose_threshold(rows)[0], 3.0)
        for r in rows:
            r["meets_sanity"] = False
        # None meets the sanity rule: fewest addresses over the limit, then the smaller threshold.
        self.assertEqual(baseline.choose_threshold(rows)[0], 3.0)


if __name__ == "__main__":
    unittest.main()
