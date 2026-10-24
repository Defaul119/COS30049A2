"""Tests for the address profiles (plan §9.4, S-7): the 8 features on hand-checkable histories,
the profile names and the DS2-covers-DS1 table.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from explorer.profiles import FEATURES, SECONDS_PER_DAY, address_features, coverage, describe, pipeline
from explorer.settings import SETTINGS


def history(name: str, times: list[int], received: list[int], sent: list[int]) -> pd.DataFrame:
    n = len(times)
    return pd.DataFrame({"addr_id": name, "block_time_utc": times, "received_sats": received, "sent_sats": sent,
                         "n_outputs": [2] * n, "fee_sats": [500] * n, "vsize_vb": [250] * n})


class FeatureTests(unittest.TestCase):
    def test_a_regular_daily_history(self):
        days = [d * SECONDS_PER_DAY for d in range(10)]
        f = address_features(history("regular", days, [1000] * 6 + [0] * 4, [0] * 6 + [99] * 4), "addr_id").iloc[0]
        self.assertAlmostEqual(f["tx_per_day"], 10 / 9)
        self.assertAlmostEqual(f["in_share"], 0.6)
        self.assertEqual(f["gap_cv"], 0.0)  # equal gaps
        self.assertEqual(f["burstiness"], -1.0)  # one transaction every day: perfectly regular
        self.assertEqual((f["history_tx"], f["mean_outputs"], f["median_fee_rate"]), (10, 2.0, 2.0))
        self.assertAlmostEqual(f["median_log_size"], np.median(np.log10(1 + np.array([1000] * 6 + [99] * 4))))

    def test_a_burst_then_silence_is_bursty_and_a_one_day_history_has_a_finite_rate(self):
        times = [0, 60, 120, 180, 40 * SECONDS_PER_DAY]
        bursty = address_features(history("bursty", times, [1] * 5, [0] * 5), "addr_id").iloc[0]
        self.assertGreater(bursty["burstiness"], 0.5)
        self.assertGreater(bursty["gap_cv"], 1.0)
        one_day = address_features(history("day", [0, 10, 20], [1] * 3, [0] * 3), "addr_id").iloc[0]
        self.assertEqual(one_day["tx_per_day"], 3 / SETTINGS["profiles"]["minSpanDays"])


class ProfileTests(unittest.TestCase):
    def test_a_profile_is_named_by_its_most_distinctive_features(self):
        centroid = np.zeros(len(FEATURES))
        centroid[FEATURES.index("burstiness")] = -2.0
        centroid[FEATURES.index("tx_per_day")] = 1.0
        self.assertEqual(describe(centroid), "steady, many transactions per day")
        self.assertEqual(describe(np.zeros(len(FEATURES))), "typical on every feature")

    def test_the_pipeline_labels_new_addresses_and_coverage_counts_ds1_inside_ds2(self):
        rng = np.random.default_rng(0)
        ds2 = pd.DataFrame(rng.uniform(0.1, 10, size=(200, len(FEATURES))), columns=FEATURES)
        model = pipeline(3, SETTINGS["model"]["seed"]).fit(ds2)
        self.assertTrue(set(model.predict(ds2.head(5))) <= {0, 1, 2})
        ds1 = pd.DataFrame([ds2.median()] * 4).reset_index(drop=True)  # inside DS2's range on every feature
        ds1.loc[0, "gap_cv"] = 1000.0  # far outside DS2's range
        cov = coverage(ds1, ds2).set_index("feature")
        self.assertEqual(cov.loc["gap_cv", "ds1_inside_share"], 0.75)


if __name__ == "__main__":
    unittest.main()
