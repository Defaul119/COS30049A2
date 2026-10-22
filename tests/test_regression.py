"""Tests for the DS1 regression method (plan §9.5): time-safe inputs, validation split, residual scoring.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from explorer import regression
from explorer.features import FeatureConfig, build_features
from explorer.methods import fit_rows
from explorer.settings import SETTINGS
from explorer.split import assign_periods
from tests.test_inject import history


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tx = pd.concat([history("big", 150, 1), history("mid", 90, 2)], ignore_index=True)
        features = build_features(tx, FeatureConfig.refined())
        period, _ = assign_periods(features, SETTINGS["model"]["fitFraction"])
        cls.features = features.assign(period=period)

    def test_history_inputs_use_only_earlier_transactions(self):
        big = self.features[self.features["address"] == "big"].sort_values(["block_time_utc", "txid"])
        first, second = big.iloc[0], big.iloc[1]
        self.assertTrue(math.isnan(first["prev_log_size"]))
        self.assertEqual(first["age_log_hours"], 0.0)
        self.assertAlmostEqual(second["prev_log_size"], math.log10(1 + abs(first["net_sats"])))
        self.assertEqual(second["in_share_prior"], float(first["direction"] == "in"))
        self.assertEqual(list(regression.design(big).columns), regression.INPUTS)

    def test_validation_is_the_last_part_of_each_address(self):
        rows = fit_rows(self.features)
        is_val = regression.time_ordered_validation(rows, 0.2)
        for _, group in rows.assign(v=is_val).groupby("address"):
            ordered = group.sort_values("block_time_utc")["v"].to_numpy()
            self.assertTrue((np.diff(ordered.astype(int)) >= 0).all())  # never back to training after validation
            self.assertAlmostEqual(ordered.mean(), 0.2, delta=0.05)

    def test_a_tenfold_amount_gets_a_large_residual(self):
        rows = fit_rows(self.features)
        y = regression.targets(rows)["size"]
        model = regression.make_model("ridge", {"alpha": 1.0}, 1).fit(regression.design(rows), y)
        scales, default = regression.residual_scales(model, rows, y, SETTINGS["regression"]["residualFloor"])
        method = regression.RegressionMethod({"size": model}, {"size": scales}, {"size": default})
        ev = self.features[self.features["period"] == "eval"].copy()
        spiked = ev.iloc[[5]].copy()
        spiked["F1"] = spiked["F1"] + 1  # ten times the amount, on the log10 scale
        self.assertGreater(method.score(spiked)[0], method.score(ev.iloc[[5]])[0])
        self.assertGreaterEqual(method.score(spiked)[0], method.threshold)
        self.assertTrue(all(v >= SETTINGS["regression"]["residualFloor"] for v in scales.values()))


if __name__ == "__main__":
    unittest.main()
