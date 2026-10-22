"""Tests for TxGPT as a DS1 method (explorer/txgpt.py): the relative score and the reason phrase.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from explorer import txgpt
from explorer.settings import SETTINGS
from lm.tx_tokens import FIELDS


class TxGPTMethodTests(unittest.TestCase):
    def test_relative_score_uses_strictly_earlier_surprisals(self):
        times = np.array([1, 2, 3, 4, 5, 6, 6, 7])
        s = np.array([10, 10, 10, 10, 10, 10, 50, 10], dtype=float)
        rel = txgpt.relative_scores(times, s, min_scale=0.5, min_values=5)
        self.assertTrue(np.isnan(rel[:5]).all())  # fewer than five earlier values
        self.assertEqual(rel[5], 0.0)
        self.assertEqual(rel[6], (50 - 10) / 0.5)  # its same-block neighbour is not its history
        self.assertEqual(rel[7], 0.0)  # one large earlier value moves neither the median nor the MAD

    def test_lookup_scores_and_reasons(self):
        lookup = pd.DataFrame({"key": ["a", "a"], "txid": ["t1", "t2"], "surprisal": [3.0, 9.0],
                               "relative": [np.nan, 4.0], **{f"surprisal_{f}": [0.5, 0.5] for f in FIELDS}})
        lookup.loc[1, "surprisal_AMT"] = 7.0
        frame = pd.DataFrame({"address": ["a", "a", "a"], "txid": ["t1", "t2", "missing"]})
        raw = txgpt.TxGPTMethod("raw", lookup)
        self.assertEqual(raw.score(frame)[:2].tolist(), [3.0, 9.0])
        self.assertEqual(raw.score(frame)[2], -np.inf)
        self.assertEqual(txgpt.TxGPTMethod("relative", lookup).score(frame)[0], -np.inf)
        self.assertEqual(raw.reasons(frame)[1], SETTINGS["txgpt"]["reasons"]["AMT"])


if __name__ == "__main__":
    unittest.main()
