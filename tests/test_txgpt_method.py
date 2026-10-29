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


def test_event_scores_add_strictly_earlier_positive_parts_in_the_window():
    # Stage C7: own positive part + strictly earlier ones within the window; same-time rows are not history.
    import numpy as np

    from explorer.txgpt import event_scores

    times = np.array([0.0, 100.0, 100.0, 5000.0, 9000.0])
    values = np.array([1.0, -2.0, 3.0, np.nan, 0.5])
    out = event_scores(times, values, 3600.0)
    assert out[0] == 1.0
    assert out[1] == 0.0 + 1.0 and out[2] == 3.0 + 1.0  # the two same-time rows never count each other
    assert np.isnan(out[3])                            # no own score, no event score
    assert out[4] == 0.5                               # 5000 is more than an hour earlier


if __name__ == "__main__":
    unittest.main()
