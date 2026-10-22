"""Tests for the DS1 cleaning step (transactions.py) and the time split (split.py), plan §6.1 and §6.3.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import unittest

import pandas as pd

from explorer.split import EVAL, FIT, HISTORY, assign_periods, cut_off
from explorer.snapshot import trim_tx
from explorer.transactions import build_transactions

ME, OTHER = "bc1qme", "bc1qother"


def tx(txid: str, received: int = 0, sent: int = 0, t: int = 1_700_000_000, confirmed: bool = True) -> dict:
    vin = [{"prevout": {"scriptpubkey_address": ME if sent else OTHER, "value": sent or 9_000}}]
    vout = [{"scriptpubkey_address": ME, "value": received}] if received else []
    vout.append({"scriptpubkey_address": OTHER, "value": 100})
    full = {"txid": txid, "vin": vin, "vout": vout, "fee": 150, "weight": 561,
            "status": {"confirmed": confirmed, "block_height": 800_000, "block_time": t}}
    return trim_tx(full, ME)


class TransactionsTests(unittest.TestCase):
    def test_cleaning_steps_and_derived_columns(self):
        txs = [tx("b", received=5_000, t=2), tx("a", sent=3_000, t=1), tx("a", sent=3_000, t=1),
               tx("c", received=1, t=3, confirmed=False), tx("d", t=4)]
        frame, steps = build_transactions({ME: {"role": "main", "chain_stats": {}, "txs": txs}})
        self.assertEqual([s["removed"] for s in steps], [0, 1, 1, 1, 0])
        self.assertEqual(frame["txid"].tolist(), ["a", "b"])  # time order
        a = frame.iloc[0]
        self.assertEqual((a["net_sats"], a["direction"], a["vsize_vb"], a["n_inputs"], a["n_outputs"]),
                         (-3_000, "out", 141, 1, 1))
        self.assertEqual(frame["fee_sats"].dtype, "int64")

    def test_a_missing_required_value_stops_the_pipeline(self):
        broken = tx("a", received=10)
        broken["fee"] = None
        with self.assertRaises(ValueError):
            build_transactions({ME: {"role": "main", "chain_stats": {}, "txs": [broken]}})


class SplitTests(unittest.TestCase):
    def test_cut_off_is_the_time_at_the_fit_fraction(self):
        self.assertEqual(cut_off(list(range(10)), 0.7), 7)
        self.assertIsNone(cut_off([], 0.7))

    def test_periods_respect_scorability_and_never_split_a_block(self):
        times = [1, 2, 3, 4, 5, 6, 7, 7, 7, 8]  # three transactions share the block at t = 7
        frame = pd.DataFrame({
            "address": ME, "role": "main", "txid": [f"t{i}" for i in range(10)], "block_time_utc": times,
            "scorable": [False, False] + [True] * 8,
        })
        period, summary = assign_periods(frame, 0.7)
        self.assertEqual(period.tolist()[:2], [HISTORY, HISTORY])
        # Scorable times 3..8 (8 rows): t* = times[int(0.7 * 8)] = 7, so the whole t = 7 block is evaluation.
        self.assertEqual(period.tolist()[2:], [FIT] * 4 + [EVAL] * 4)
        self.assertEqual(summary.iloc[0][["history", "fit", "eval"]].tolist(), [2, 4, 4])

    def test_an_address_with_nothing_scorable_is_all_history(self):
        frame = pd.DataFrame({"address": OTHER, "role": "short", "txid": ["x", "y"], "block_time_utc": [1, 2],
                              "scorable": [False, False]})
        period, summary = assign_periods(frame, 0.7)
        self.assertEqual(set(period), {HISTORY})
        self.assertIsNone(summary.iloc[0]["cut_off_utc"])


if __name__ == "__main__":
    unittest.main()
