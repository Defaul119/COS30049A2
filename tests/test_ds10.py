"""Tests for DS10 (explorer/ds10.py): the teammate's snapshot, its cut and the row-by-row cross-check.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from explorer import ds10


def tx(txid: str, height: int | None) -> dict:
    return {"txid": txid, "status": {"confirmed": height is not None, "block_height": height}}


class CutTests(unittest.TestCase):
    def test_transactions_after_the_teammates_last_block_are_left_out(self):
        txs = [tx("c", 120), tx("b", 110), tx("a", 100)]  # newest first, as Esplora pages are
        self.assertEqual([t["txid"] for t in ds10.cut_history(txs, 110)], ["b", "a"])

    def test_an_unconfirmed_transaction_is_never_kept(self):
        self.assertEqual(ds10.cut_history([tx("m", None), tx("a", 100)], 200), [tx("a", 100)])


class TeammateTablesTests(unittest.TestCase):
    def test_btc_amounts_become_exact_satoshis_and_tiers_become_roles(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            pd.DataFrame({"address": ["A", "A"], "txid": ["t1", "t2"], "block_height": [1, 2],
                          "received_btc": [0.00012345, 0.0], "sent_btc": [0.0, 1.1],
                          "net_btc": [0.00012345, -1.1]}).to_csv(folder / "transactions.csv", index=False)
            pd.DataFrame({"address": ["A", "S"], "group": ["high", "short_history"]}).to_csv(
                folder / "address_summary.csv", index=False)
            (folder / "manifest.json").write_text("{}", encoding="utf-8")
            theirs, summary, _ = ds10.teammate_tables(folder)
        # 1.1 BTC is 110000000.00000001 in floating point: rounding must give the exact integer.
        self.assertEqual(theirs["received_sats"].tolist(), [12345, 0])
        self.assertEqual(theirs["sent_sats"].tolist(), [0, 110_000_000])
        self.assertEqual(ds10.roles_of(summary), {"A": "main", "S": "short"})


class CompareRowsTests(unittest.TestCase):
    def frame(self, rows):
        return pd.DataFrame(rows, columns=["address", "txid", "block_height", "received_sats", "sent_sats"])

    def test_identical_tables_agree(self):
        rows = [("A", "t1", 1, 10, 0), ("A", "t2", 2, 0, 5)]
        table = ds10.compare_rows(self.frame(rows), self.frame(rows))
        self.assertEqual(table.loc[0, ["shared", "only_ours", "only_theirs", "differs"]].tolist(), [2, 0, 0, 0])
        self.assertTrue(table.loc[0, "agrees"])

    def test_missing_rows_and_a_different_amount_are_each_counted(self):
        ours = self.frame([("A", "t1", 1, 10, 0), ("A", "t2", 2, 0, 5), ("B", "t9", 3, 7, 0)])
        theirs = self.frame([("A", "t1", 1, 11, 0), ("A", "t3", 4, 1, 0), ("B", "t9", 3, 7, 0)])
        table = ds10.compare_rows(ours, theirs).set_index("address")
        self.assertEqual(table.loc["A", ["shared", "only_ours", "only_theirs", "differs"]].tolist(), [1, 1, 1, 1])
        self.assertFalse(table.loc["A", "agrees"])
        self.assertTrue(table.loc["B", "agrees"])


class ActivityProfileTests(unittest.TestCase):
    def test_gaps_and_window_counts_are_medians_over_main_addresses(self):
        hour = 3600
        rows = [("A", "main", k * 2 * hour) for k in range(5)]  # every 2 hours: 0..4 earlier in the window
        rows += [("B", "main", k * 10 * hour) for k in range(5)]  # every 10 hours
        rows += [("S", "short", 0), ("S", "short", 1)]  # short histories are left out
        tx = pd.DataFrame(rows, columns=["address", "role", "block_time_utc"])
        p = ds10.activity_profile(tx, "test")
        self.assertEqual(p["main_addresses"], 2)
        self.assertEqual(p["median_gap_hours"], 6.0)  # median of 2 and 10
        self.assertEqual(p["median_window_count"], 2.0)  # 0,1,2,3,4 earlier in 7 days for both


if __name__ == "__main__":
    unittest.main()
