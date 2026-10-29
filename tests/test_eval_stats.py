"""Tests for lm/eval_stats.py: the writers' rates, paired differences and McNemar counts (statistics campaign C).

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lm import eval_stats


def answers(path: Path, correct: list[bool], key_facts: list[float] | None = None) -> None:
    n = len(correct)
    frame = pd.DataFrame({"id": [f"p{i}" for i in range(n)], "kind": "flagged", "correct": correct})
    if key_facts is not None:
        frame = frame.assign(kind="glossary", key_facts=key_facts)
    frame.to_csv(path, index=False)


class EvalStatsTests(unittest.TestCase):
    def test_rates_paired_difference_and_mcnemar_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tables = Path(tmp)
            # DS7 v1 test (no suffix): round 1 right on 6 of 10, round 2 on 8; they disagree on 2 + 0 prompts
            answers(tables / "assistant_eval_text150m.csv", [True] * 6 + [False] * 4)
            answers(tables / "assistant_eval_text150m_v2.csv", [True] * 8 + [False] * 2)
            answers(tables / "assistant_eval_text150m_glossary.csv", [False] * 4, [0.0, 0.5, 0.5, 1.0])
            answers(tables / "assistant_eval_text150m_v2_glossary.csv", [False] * 4, [1.0, 0.5, 1.0, 1.0])
            with patch.object(eval_stats, "TABLES", tables):
                self.assertEqual(eval_stats.main(), 0)
            summary = pd.read_csv(tables / "stats_writers_summary.csv")
            paired = pd.read_csv(tables / "stats_writers_paired.csv")
        ds7 = summary[summary["test_set"] == "DS7 v1 test"].set_index("writer")["fully_correct"]
        self.assertAlmostEqual(ds7["text150m"], 0.6)
        self.assertAlmostEqual(ds7["text150m_v2"], 0.8)
        row = paired[paired["test_set"] == "DS7 v1 test"].iloc[0]
        self.assertAlmostEqual(row["diff"], -0.2)  # writer_a (round 1) minus writer_b (round 2), observed
        self.assertEqual((row["only_a_right"], row["only_b_right"]), (0, 2))
        self.assertLessEqual(row["ci_low"], row["diff"])
        self.assertGreaterEqual(row["ci_high"], row["diff"])
        gloss = paired[paired["test_set"] == "glossary (key facts)"].iloc[0]
        self.assertAlmostEqual(gloss["diff"], 0.5 - 0.875)  # the observed difference of the key-fact shares


if __name__ == "__main__":
    unittest.main()
