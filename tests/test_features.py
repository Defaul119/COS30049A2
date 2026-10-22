"""Tests for explorer/features.py: parity with the A1 prototype and time safety (DR-02).

Parity: tests/fixtures/scoring_parity.json holds seeded histories and the output of the prototype's
own scoreAddress (reference/a1_prototype/scoring.ts), built by tools/make_scoring_fixture.py. The
Python port must agree on every feature, window, usual value, z-score, score and flag.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import json
import math
import unittest
from dataclasses import replace
from pathlib import Path

from explorer.features import FEATURE_IDS, FeatureConfig, RawTx, score_address

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "scoring_parity.json"
TOL = 1e-9


def close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=TOL, abs_tol=TOL)


def raw(txs: list[dict]) -> list[RawTx]:
    return [RawTx(t["txid"], t["time"], t["blockHeight"], t["receivedSats"], t["sentSats"]) for t in txs]


class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.cfg = FeatureConfig.refined(cls.fixture["settings"])

    def test_every_field_matches_the_prototype(self):
        window_keys = {"count": "count", "countInWindowDays": "count_in_period", "extended": "extended",
                       "direction": "direction", "start": "start", "end": "end"}
        for address, txs in self.fixture["inputs"].items():
            ours = score_address(raw(txs), self.cfg)
            theirs = self.fixture["expected"][address]
            self.assertEqual([o["txid"] for o in ours], [t["txid"] for t in theirs], address)
            for o, t in zip(ours, theirs):
                where = f"{address} {t['txid'][:8]}"
                self.assertEqual(o["earlier_count"], t["earlierCount"], where)
                self.assertTrue(close(o["history_span_days"], t["historySpanDays"]), where)
                self.assertEqual(o["scored"], t["scored"], where)
                self.assertEqual(o["flagged"], t["flagged"], where)
                self.assertTrue(close(o["score"], t["score"]), where)
                for fid in FEATURE_IDS:
                    for part in ("features", "usual", "z"):
                        self.assertTrue(close(o[part][fid], t[part][fid]),
                                        f"{where} {part}.{fid}: {o[part][fid]} vs {t[part][fid]}")
                for ours_key, theirs_key in (("timing_window", "timingWindow"), ("size_window", "sizeWindow")):
                    if t[theirs_key] is None:
                        self.assertIsNone(o[ours_key], where)
                    else:
                        self.assertEqual({k: o[ours_key][v] for k, v in window_keys.items()}, t[theirs_key], where)

    def test_fixture_exercises_the_hard_cases(self):
        expected = self.fixture["expected"]
        self.assertGreater(sum(t["flagged"] for v in expected.values() for t in v), 0)
        self.assertTrue(any(t["sizeWindow"] and t["sizeWindow"]["extended"] for v in expected.values() for t in v))
        self.assertFalse(any(t["scored"] for t in expected["short"]))
        times = [t["time"] for t in self.fixture["inputs"]["ties_constant"]]
        self.assertLess(len(set(times)), len(times))  # same-block ties present


class TimeSafetyTests(unittest.TestCase):
    """DR-02: a transaction's scores depend only on strictly earlier transactions of the address."""

    @classmethod
    def setUpClass(cls):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.inputs = fixture["inputs"]
        cls.cfg = FeatureConfig.refined(fixture["settings"])

    def test_changing_a_later_transaction_never_changes_an_earlier_one(self):
        for address, txs in self.inputs.items():
            base = score_address(raw(txs), self.cfg)
            cut = base[len(base) // 2]["time"]
            changed = [RawTx(t.txid, t.time, t.block_height, t.received_sats * 1000 + 7, t.sent_sats)
                       if t.time > cut else t for t in raw(txs)]
            after = score_address(changed, self.cfg)
            for b, a in zip(base, after):
                if b["time"] <= cut:
                    self.assertEqual(a, b, f"{address} {b['txid'][:8]} changed when a later transaction did")

    def test_same_block_transactions_are_not_each_others_history(self):
        scored = score_address(raw(self.inputs["ties_constant"]), self.cfg)
        by_time: dict[int, list[dict]] = {}
        for s in scored:
            by_time.setdefault(s["time"], []).append(s)
        tied = [group for group in by_time.values() if len(group) > 1]
        self.assertTrue(tied)
        for group in tied:
            self.assertEqual(len({s["earlier_count"] for s in group}), 1)
            self.assertEqual(len({s["features"]["F2"] for s in group}), 1)

    def test_first_transaction_has_no_gap_and_is_never_flagged(self):
        for txs in self.inputs.values():
            first = score_address(raw(txs), self.cfg)[0]
            self.assertIsNone(first["features"]["F3"])
            self.assertFalse(first["flagged"])


class AsWrittenTests(unittest.TestCase):
    def test_as_written_has_no_extension_no_direction_split_and_no_floor(self):
        cfg = FeatureConfig.as_written()
        self.assertEqual((cfg.comparison_min_tx, cfg.same_direction, cfg.transform), (0, False, False))
        self.assertTrue(all(v == 0 for v in cfg.min_scale.values()))
        refined = FeatureConfig.refined()
        self.assertEqual(replace(cfg, comparison_min_tx=refined.comparison_min_tx, same_direction=refined.same_direction,
                                 transform=True, min_scale=refined.min_scale), refined)

    def test_a_constant_history_gives_an_infinite_z_without_the_floor(self):
        day = 86_400
        txs = [RawTx(f"{i:064x}", 1_700_000_000 + i * day, i, 10_000, 0) for i in range(20)]
        txs.append(RawTx(f"{99:064x}", 1_700_000_000 + 20 * day, 20, 10_001, 0))
        last_as_written = score_address(txs, FeatureConfig.as_written())[-1]
        last_refined = score_address(txs, FeatureConfig.refined())[-1]
        self.assertEqual(last_as_written["z"]["F1"], math.inf)
        self.assertTrue(last_as_written["flagged"])
        self.assertLess(abs(last_refined["z"]["F1"]), 1)  # the CR-05 floor keeps a 1-sat change ordinary
        self.assertFalse(last_refined["flagged"])


if __name__ == "__main__":
    unittest.main()
