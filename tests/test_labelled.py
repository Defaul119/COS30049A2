"""Tests for DS4 processing (labelled.py) and the classification protocol (classify.py), plan §6.4 and §9.4.

Small synthetic tables stand in for BitcoinHeist, so these run in seconds.
Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from explorer import classify, labelled
from explorer.settings import SETTINGS


def synthetic_heist(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    year = rng.integers(2011, 2019, n)
    label = np.where(rng.random(n) < 0.05, rng.choice(["fooLocker", "barCrypt", "bazWare"], n), "white")
    positive = label != "white"
    return pd.DataFrame({
        "address": [f"a{i}" for i in range(n)], "year": year, "day": rng.integers(1, 366, n),
        "length": rng.integers(0, 144, n), "weight": rng.lognormal(0, 2, n), "count": rng.integers(1, 3000, n) ** 2,
        "looped": rng.integers(0, 10, n) ** 3, "neighbors": rng.integers(1, 50, n) ** 3,
        "income": np.where(positive, rng.lognormal(19, 1, n), rng.lognormal(18, 2, n)), "label": label,
    })


class ProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = {**SETTINGS["bitcoinheist"], "minTestPositives": 30}
        cls.out, cls.decisions = labelled.process_heist(synthetic_heist(), cls.params)

    def test_identifiers_and_time_are_never_features(self):
        features = [c for c in self.out.columns if c not in classify.HEIST_META]
        self.assertEqual(len(features), len(self.params["features"]))
        for leak in ("address", "year", "day"):
            self.assertNotIn(leak, features)
        self.assertIn("log10_income", features)

    def test_temporal_year_is_the_latest_leaving_enough_positives(self):
        per_year = pd.Series({2015: 50, 2016: 40, 2017: 25, 2018: 2})
        self.assertEqual(labelled.temporal_year(per_year, 27), 2016)  # 2017 would leave only 2
        self.assertEqual(labelled.temporal_year(per_year, 60), 2015)

    def test_downsampling_touches_training_sets_only(self):
        out, ratio = self.out, self.params["negativesPerPositive"]
        for scheme in ("temporal", "random"):
            sampled = out[out[f"{scheme}_sampled"]]
            self.assertTrue((sampled[scheme] == labelled.FIT).all())
            self.assertLessEqual((sampled["y"] == 0).sum(), ratio * sampled["y"].sum())
            self.assertFalse(out.loc[out[f"{scheme}_train_sampled"], scheme].eq(labelled.TEST).any())
        test = out[out["random"] == labelled.TEST]
        self.assertAlmostEqual(test["y"].mean(), out["y"].mean(), delta=0.01)  # real ratio kept

    def test_temporal_test_years_follow_the_training_years(self):
        y = self.decisions["temporal_last_training_year"]
        self.assertTrue((self.out.loc[self.out["temporal"] == labelled.TEST, "year"] > y).all())
        self.assertTrue((self.out.loc[self.out["temporal"] == labelled.VAL, "year"] == y).all())


class ProtocolTests(unittest.TestCase):
    def test_threshold_comes_from_validation_and_the_test_is_scored_once(self):
        params = {**SETTINGS["bitcoinheist"], "minTestPositives": 30}
        rows, _ = labelled.process_heist(synthetic_heist(seed=1), params)
        features = [c for c in rows.columns if c not in classify.HEIST_META]
        fit, val, train, test = classify.heist_parts(rows, "random")
        cfg = {**SETTINGS["classification"], "nJobs": 1, "nIter": 2}
        result, tuning, model, p_test = classify.search_and_evaluate(
            "logistic", features, fit, val, train, test, cfg, knn_rows=None)
        self.assertEqual(len(tuning), len(classify.configurations("logistic", cfg)))
        best = max(tuning, key=lambda r: r["val_f1"])
        self.assertEqual(result["threshold"], best["threshold"])
        self.assertEqual(result["n_test"], len(test))
        self.assertEqual(result["tp"] + result["fn"], int(test["y"].sum()))

    def test_best_f1_threshold(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.35, 0.8])
        threshold, f1 = classify.best_f1_threshold(y, p)
        self.assertEqual((threshold, round(f1, 4)), (0.35, 0.8))


class EllipticTests(unittest.TestCase):
    def test_join_degrees_labels_and_split(self):
        names = ["txId", "time_step"] + labelled.LOCAL + labelled.AGG
        rng = np.random.default_rng(0)
        features = pd.DataFrame(rng.normal(size=(4, len(names))), columns=names)
        features["txId"] = [10, 11, 12, 13]
        features["time_step"] = [1, 27, 35, 49]
        classes = pd.DataFrame({"txId": [10, 11, 12, 13], "class": ["1", "2", "unknown", "2"]})
        edges = pd.DataFrame({"txId1": [10, 10, 11], "txId2": [11, 12, 12]})
        out, decisions = labelled.process_elliptic(features, classes, edges, SETTINGS["elliptic"])
        self.assertEqual(out["y"].tolist()[:2], [1.0, 0.0])
        self.assertTrue(np.isnan(out["y"].iloc[2]))
        self.assertEqual(out["out_degree"].tolist(), [2, 1, 0, 0])
        self.assertEqual(out["in_degree"].tolist(), [0, 1, 2, 0])
        self.assertEqual(out["period"].tolist(), [labelled.FIT, labelled.VAL, labelled.TEST, labelled.TEST])
        self.assertNotIn("time_step", labelled.FEATURE_SETS["AF"])
        self.assertEqual((decisions["labelled"], decisions["illicit"]), (3, 1))


if __name__ == "__main__":
    unittest.main()
