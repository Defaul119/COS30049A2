"""Tests for the label check: top-share metrics, and the four fitted scorers' direction (plan §9.4).

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from explorer import label_check

PARAMS = {"isolation_forest": {"n_estimators": 100, "max_samples": "auto", "max_features": 1.0},
          "lof": {"n_neighbors": 20}}


class LabelMetricsTests(unittest.TestCase):
    def test_perfect_ranking_puts_every_positive_in_the_top_share(self):
        y = np.array([0] * 98 + [1] * 2)
        m = label_check.label_metrics(y, np.arange(100, dtype=float), 0.02)
        self.assertEqual(m["top_rows"], 2)
        self.assertEqual(m["precision_top"], 1.0)
        self.assertEqual(m["recall_top"], 1.0)
        self.assertAlmostEqual(m["lift_top"], 50.0)
        self.assertAlmostEqual(m["pr_auc"], 1.0)

    def test_reversed_ranking_scores_below_chance(self):
        y = np.array([1] * 2 + [0] * 98)
        m = label_check.label_metrics(y, np.arange(100, dtype=float), 0.02)
        self.assertEqual(m["precision_top"], 0.0)
        self.assertLess(m["roc_auc"], 0.5)


class ScorerTests(unittest.TestCase):
    def test_every_method_scores_a_far_point_above_the_training_rows(self):
        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(0, 1, (400, 3)), rng.normal(6, 1, (400, 3))])
        scorers, tuning = label_check.fit_scorers(x, 0, PARAMS)
        self.assertEqual(set(scorers), set(label_check.METHODS))
        self.assertEqual({r["method"] for r in tuning}, {"kmeans", "dbscan"})
        far = np.array([[30.0, -30.0, 30.0]])
        for name, score in scorers.items():
            self.assertGreater(score(far)[0], np.quantile(score(x), 0.99), name)

    def test_elliptic_training_part_keeps_unknown_nodes_and_test_part_only_labelled(self):
        rows = pd.DataFrame({"time_step": [1, 2, 3, 3, 4], "y": [np.nan, 1.0, 0.0, np.nan, 1.0]})
        train, test = label_check.elliptic_parts(rows, 2)
        self.assertEqual(len(train), 2)
        self.assertTrue(train["y"].isna().any())
        self.assertEqual(test["y"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
