"""Tests for the blind reviewer check: row choice, the sheet hiding flags, and the scoring (plan §11.1).

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from explorer import reviewer


def scores_frame(n: int = 60, flagged: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    score = rng.random(n)
    flag = np.zeros(n, dtype=bool)
    flag[np.argsort(-score)[:flagged]] = True
    return pd.DataFrame({"address": "a", "txid": [f"t{i}" for i in range(n)], "period": "eval", "scorable": True,
                         "score_m": score, "flag_m": flag})


class PickRowsTests(unittest.TestCase):
    def test_takes_top_flagged_and_random_unflagged_shuffled(self):
        scores = scores_frame()
        chosen = reviewer.pick_rows(scores, "m", 5, 5, 1)
        self.assertEqual(len(chosen), 10)
        top5 = set(scores.sort_values("score_m", ascending=False)["txid"].head(5))
        self.assertEqual(set(chosen.loc[chosen["flagged"], "txid"]), top5)
        self.assertFalse(scores.set_index("txid").loc[chosen.loc[~chosen["flagged"], "txid"], "flag_m"].any())
        self.assertNotEqual(chosen["flagged"].tolist(), [True] * 5 + [False] * 5)
        self.assertEqual(chosen["row"].iloc[0], "R01")

    def test_same_seed_gives_the_same_sheet(self):
        scores = scores_frame()
        a, b = reviewer.pick_rows(scores, "m", 5, 5, 7), reviewer.pick_rows(scores, "m", 5, 5, 7)
        self.assertEqual(a["txid"].tolist(), b["txid"].tolist())

    def test_sheet_shows_no_flag_or_score(self):
        chosen = reviewer.pick_rows(scores_frame(), "m", 2, 2, 1)
        features = pd.DataFrame({"address": "a", "txid": chosen["txid"], "block_time_utc": 1_700_000_000,
                                 "direction": "in", "net_sats": 150_000_000, "usual_F1": 1e8, "F2": 1, "usual_F2": 2.0,
                                 "F3": 5.0, "usual_F3": 24.0, "n_inputs": 1, "n_outputs": 2, "fee_sats": 1000,
                                 "vsize_vb": 200})
        sheet = reviewer.sheet_columns(chosen, features)
        self.assertFalse({c for c in sheet.columns if "flag" in c or "score" in c})
        self.assertEqual(sheet["amount_btc"].iloc[0], 1.5)
        self.assertTrue((sheet[reviewer.RATING] == "").all())


class ScoreTests(unittest.TestCase):
    def key(self) -> pd.DataFrame:
        return pd.DataFrame({"row": ["R1", "R2", "R3", "R4"], "flagged": [True, True, False, False]})

    def test_shares_and_kappa_for_two_reviewers(self):
        a = pd.Series([1, 1, 0, 0], index=["R1", "R2", "R3", "R4"], name="a")
        b = pd.Series([1, 0, 0, 0], index=["R1", "R2", "R3", "R4"], name="b")
        table = reviewer.score_ratings(self.key(), [a, b])
        self.assertEqual(table.loc[0, "flagged_worth_a_look"], 1.0)
        self.assertEqual(table.loc[1, "flagged_worth_a_look"], 0.5)
        self.assertEqual(table.loc[2, "raw_agreement"], 0.75)
        self.assertAlmostEqual(table.loc[2, "cohen_kappa"], 0.5)

    def test_one_reviewer_leaves_kappa_empty(self):
        a = pd.Series([1, 1, 0, 0], index=["R1", "R2", "R3", "R4"], name="a")
        self.assertTrue(np.isnan(reviewer.score_ratings(self.key(), [a]).iloc[-1]["cohen_kappa"]))

    def test_blank_rating_is_refused_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.csv"
            pd.DataFrame({"row": ["R1", "R2"], reviewer.RATING: ["yes", ""]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                reviewer.read_ratings(path)


if __name__ == "__main__":
    unittest.main()
