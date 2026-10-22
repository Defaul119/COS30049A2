"""Tests for synthetic injection (plan §6.4): where and how each pattern is planted, and time safety.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from explorer import inject
from explorer.features import DAY, FeatureConfig, build_features
from explorer.settings import SETTINGS
from explorer.split import EVAL, assign_periods

T0 = 1_700_000_000


def history(address: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        direction = "in" if i % 3 else "out"
        amount = int(rng.lognormal(11, 0.3))
        received, sent = (amount, 0) if direction == "in" else (0, amount)
        rows.append({"address": address, "role": "main", "txid": f"{address}{i:04d}", "block_height": 800_000 + i,
                     "block_time_utc": T0 + i * DAY + int(rng.integers(0, 3600)), "received_sats": received,
                     "sent_sats": sent, "net_sats": received - sent, "direction": direction,
                     "n_inputs": 1 + i % 2, "n_outputs": 2, "fee_sats": 300, "vsize_vb": 141})
    return pd.DataFrame(rows)


class InjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tx = pd.concat([history("big", 120, 1), history("small", 40, 2)], ignore_index=True)
        features = build_features(cls.tx, FeatureConfig.refined())
        period, _ = assign_periods(features, SETTINGS["model"]["fitFraction"])
        cls.periods = features.assign(period=period)[["address", "txid", "period"]]
        cls.plants = inject.make_plants(cls.tx, cls.periods, 40000, "tuning")
        cls.p = SETTINGS["injection"]

    def eval_start(self, address: str) -> int:
        ev = self.periods[(self.periods["address"] == address) & (self.periods["period"] == EVAL)]
        return int(self.tx[self.tx["txid"].isin(ev["txid"])]["block_time_utc"].min())

    def test_counts_and_proportional_allocation(self):
        for pattern in inject.PATTERNS:
            chosen = [pl.address for pl in self.plants if pl.pattern == pattern]
            self.assertEqual(len(chosen), self.p["plantsPerPattern"])
        share_big = np.mean([pl.address == "big" for pl in self.plants])
        self.assertGreater(share_big, 0.6)  # 'big' has three times the evaluation rows

    def test_each_pattern_lands_where_the_plan_says(self):
        for pl in self.plants:
            times = [r["block_time_utc"] for r in pl.rows]
            last_real = int(self.tx[self.tx["address"] == pl.address]["block_time_utc"].max())
            if pl.pattern == "spike":
                self.assertEqual(len(pl.rows), 1)
                self.assertGreaterEqual(times[0], self.eval_start(pl.address))
                self.assertLess(times[0], last_real)
            elif pl.pattern == "burst":
                self.assertEqual(len(pl.rows), self.p["burstSize"])
                self.assertLessEqual(max(times) - min(times), self.p["burstWindowHours"] * 3600)
                self.assertGreaterEqual(min(times), self.eval_start(pl.address))
            else:
                gap_days = (times[0] - last_real) / DAY
                self.assertTrue(self.p["dormantDaysMin"] <= gap_days <= self.p["dormantDaysMax"])

    def test_spike_is_ten_times_the_usual_amount_in_the_majority_direction(self):
        pl = next(p for p in self.plants if p.pattern == "spike")
        row = pl.rows[0]
        h = self.tx[(self.tx["address"] == pl.address) & (self.tx["block_time_utc"] < row["block_time_utc"])]
        usual = h[h["direction"] == row["direction"]]["net_sats"].abs().median()
        self.assertAlmostEqual(abs(row["net_sats"]), self.p["spikeMultiplier"] * usual, delta=1)

    def test_planted_copy_is_time_safe_and_marks_synthetic_rows(self):
        pl = next(p for p in self.plants if p.pattern == "spike")
        frame = inject.copy_features(self.tx, self.periods, pl, FeatureConfig.refined())
        self.assertEqual(int(frame["is_synthetic"].sum()), 1)
        real = build_features(self.tx[self.tx["address"] == pl.address], FeatureConfig.refined()).set_index("txid")
        t_plant = pl.rows[0]["block_time_utc"]
        before = frame[(frame["block_time_utc"] < t_plant) & (frame["is_synthetic"] == 0)].set_index("txid")
        cols = ["z_F1", "z_F2", "z_F3", "z_F4", "score_baseline"]
        pd.testing.assert_frame_equal(before[cols], real.loc[before.index, cols])
        self.assertEqual(frame.loc[frame["is_synthetic"] == 1, "period"].tolist(), [EVAL])

    def test_same_seed_same_plants(self):
        again = inject.make_plants(self.tx, self.periods, 40000, "tuning")
        self.assertEqual([p.rows for p in again], [p.rows for p in self.plants])


if __name__ == "__main__":
    unittest.main()
