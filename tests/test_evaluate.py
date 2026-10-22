"""Tests for DS1 evaluation (plan §11): the selection rule and refitting every method type.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from explorer import evaluate, regression
from explorer.clustering import DBSCANMethod, KMeansMethod
from explorer.methods import fit_rows
from explorer.outliers import IsolationForestMethod, LOFMethod
from explorer.settings import SETTINGS
from tests.test_methods import MethodChecksTests


def frames(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t = pd.DataFrame(rows)
    return (t[["method", "mean_recall"]], t[["method", "stability"]],
            t[["method"]].assign(flag_rate=0.02, max_address_share=0.1))


class SelectionTests(unittest.TestCase):
    def test_stable_methods_only_then_explainability_within_the_margin(self):
        summary, stab, sanity = frames([
            {"method": "lof", "mean_recall": 0.90, "stability": 0.5},       # best recall but unstable
            {"method": "dbscan", "mean_recall": 0.80, "stability": 0.9},
            {"method": "baseline", "mean_recall": 0.77, "stability": 1.0},  # within 5 points: easier to explain
            {"method": "kmeans", "mean_recall": 0.60, "stability": 0.8}])
        choice = evaluate.select(summary, stab, sanity)
        self.assertEqual((choice["chosen"], choice["runner_up"]), ("baseline", "dbscan"))
        self.assertFalse(choice["no_method_stable"])

    def test_outside_the_margin_recall_wins(self):
        summary, stab, sanity = frames([
            {"method": "dbscan", "mean_recall": 0.85, "stability": 0.9},
            {"method": "baseline", "mean_recall": 0.70, "stability": 1.0}])
        self.assertEqual(evaluate.select(summary, stab, sanity)["chosen"], "dbscan")


class RefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        MethodChecksTests.setUpClass()
        cls.features = MethodChecksTests.features
        cls.rows = fit_rows(cls.features)

    def test_every_fitted_method_type_refits(self):
        y = regression.targets(self.rows)["size"]
        model = regression.make_model("ridge", {"alpha": 1.0}, 1).fit(regression.design(self.rows), y)
        scales, default = regression.residual_scales(model, self.rows, y, 0.05)
        methods = {
            "isolation_forest": IsolationForestMethod({"n_estimators": 50}, 1).fit(self.rows),
            "lof": LOFMethod({"n_neighbors": 10}, 1).fit(self.rows),
            "kmeans": KMeansMethod(3, 1).fit(self.rows),
            "dbscan": DBSCANMethod(1.0, 8).fit(self.rows),
            "regression": regression.RegressionMethod({"size": model}, {"size": scales}, {"size": default}),
        }
        sample = self.rows.sample(frac=1.0, replace=True, random_state=3)
        for name, method in methods.items():
            again = evaluate.refit(name, method, sample, 7)
            self.assertIs(type(again), type(method), name)
            self.assertEqual(len(again.score(self.features)), len(self.features), name)

    def test_fixed_methods_are_stable_by_construction(self):
        self.assertEqual(evaluate.method_stability("baseline", None, self.features), 1.0)

    def test_kmeans_stability_is_a_jaccard(self):
        value = evaluate.method_stability("kmeans", KMeansMethod(3, 1), self.features)
        self.assertTrue(0.0 <= value <= 1.0)


if __name__ == "__main__":
    unittest.main()
