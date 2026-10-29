"""Tests for lm/adopt.py: the fixed rule that decides whether the round-2 writer replaces round 1.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lm import adopt


def summaries(tables: Path, new: dict, glossary_gain: float) -> None:
    """Summary tables for the frozen tests and the glossary: round 1 fixed, round 2 with `new` overrides."""
    old = {"guardrail_pass": 0.99, "number_faithfulness": 0.995, "direction_errors": 2}
    for tag in ("", "_ds10"):
        pd.DataFrame([{"model": "text150m", **old}, {"model": "text150m_v2", **{**old, **new}}]).to_csv(
            tables / f"assistant_eval{tag}.csv", index=False)
    pd.DataFrame([{"model": "text150m", "glossary_key_facts": 0.20},
                  {"model": "text150m_v2", "glossary_key_facts": 0.20 + glossary_gain}]).to_csv(
        tables / "assistant_eval_glossary.csv", index=False)


class AdoptionRuleTests(unittest.TestCase):
    def decide(self, new: dict, gain: float) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            tables = Path(tmp)
            summaries(tables, new, gain)
            with patch.object(adopt, "TABLES", tables), patch("sys.argv", ["adopt"]):
                adopt.main()
            return pd.read_csv(tables / "adoption.csv")["decision"].iloc[0]

    def test_adopts_only_when_every_criterion_holds(self) -> None:
        self.assertEqual(self.decide({"guardrail_pass": 0.995}, 0.10), "adopt")  # exactly 10 points is enough
        self.assertEqual(self.decide({"direction_errors": 3}, 0.30), "keep")     # one more wrong comparison word
        self.assertEqual(self.decide({"number_faithfulness": 0.99}, 0.30), "keep")
        self.assertEqual(self.decide({}, 0.09), "keep")                           # key facts up 9 points only


if __name__ == "__main__":
    unittest.main()
