"""Tests for prediction (plan §12.1) and the template explanations: A1 shapes, the limitation line,
the wording rule, the error cases and what-if time safety.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import json
import unittest

import pandas as pd

from explorer import explain
from explorer.baseline import Baseline
from explorer.features import FeatureConfig, build_features
from explorer.methods import flags
from explorer.predict import InsufficientHistory, Predictor, Unsupported, check_outputs
from explorer.settings import SETTINGS
from explorer.split import assign_periods
from tests.test_inject import history


def predictor() -> Predictor:
    tx = pd.concat([history("busy", 120, 1), history("tiny", 4, 2).assign(role="short")], ignore_index=True)
    features = build_features(tx, FeatureConfig.refined())
    period, _ = assign_periods(features, SETTINGS["model"]["fitFraction"])
    features = features.assign(period=period)
    methods = {"baseline": Baseline(2.0)}  # a low threshold, so the synthetic history has flags
    scores = features[["address", "txid"]].copy()
    for name, m in methods.items():
        scores[f"score_{name}"] = m.score(features)
        scores[f"flag_{name}"] = flags(m, features)
    return Predictor(features, tx, scores, methods, "baseline", "2026-10-22")


class PredictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = predictor()

    def test_address_result_has_the_a1_shape_and_wording(self):
        result = self.p.score_address("busy")
        json.dumps(result)  # serialisable as the API will send it
        check_outputs(result)
        self.assertEqual(sum(d["txCount"] for d in result["days"]), 120)
        tx = result["transactions"][-1]
        for key in ("txid", "time", "blockHeight", "direction", "netSats", "features", "comparison", "limitation"):
            self.assertIn(key, tx)
        self.assertEqual([f["id"] for f in tx["features"]], ["F1", "F2", "F3", "F4"])
        flagged = [t for t in result["transactions"] if t["flagged"]]
        self.assertTrue(flagged)
        for t in flagged:
            self.assertIn(SETTINGS["limitationLine"], t["explanation"])
        self.assertEqual(result["totals"]["flaggedCount"], len(flagged))

    def test_errors_match_a1s_404_and_422(self):
        with self.assertRaises(Unsupported):
            self.p.score_address("bc1qnotinthesnapshot")
        with self.assertRaises(InsufficientHistory) as caught:
            self.p.score_address("tiny")
        self.assertEqual(caught.exception.body["status"], "insufficient_history")

    def test_what_if_uses_only_history_before_its_time(self):
        rows = self.p.features[self.p.features["address"] == "busy"].sort_values("block_time_utc")
        mid = int(rows["block_time_utc"].iloc[60]) + 1
        when = pd.Timestamp(mid, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        typical = self.p.score_what_if("busy", 0.0006, "in", when)
        huge = self.p.score_what_if("busy", 6.0, "in", when)
        self.assertGreater(huge["scores"]["baseline"], typical["scores"]["baseline"])
        self.assertTrue(huge["flagged"])
        self.assertIsNone(huge["sourceUrl"])
        self.assertLessEqual(huge["comparison"][0]["txCount"], 61)  # nothing after the hypothetical's time


class ExplainTests(unittest.TestCase):
    def test_banned_words_are_rejected(self):
        for word in SETTINGS["bannedWords"]:
            with self.assertRaises(ValueError):
                explain.check_wording(f"This looks {word.upper()} to us")
        explain.check_wording("Unusual for this address; worth a look.")

    def test_fallback_reasons_when_no_feature_reaches_contributing(self):
        features = [{"id": f, "value": 1.0, "usual": 1.0, "z": z, "role": "within-usual"}
                    for f, z in zip(["F1", "F2", "F3", "F4"], [0.5, -1.5, 1.0, None])]
        reasons, fallback = explain.reasons_for(features)
        self.assertTrue(fallback)
        self.assertEqual([r["id"] for r in reasons], ["F2", "F3"])


if __name__ == "__main__":
    unittest.main()
