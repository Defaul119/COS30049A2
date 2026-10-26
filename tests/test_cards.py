"""Tests for tools/cards.py: the data card describes every column of every shipped file, and the generated blocks
are filled in place (plan §11, DATA_CARD.md and MODEL_CARD.md).

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cards", ROOT / "tools" / "cards.py")
cards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cards)


class DataCardTests(unittest.TestCase):
    def test_every_listed_file_exists(self) -> None:
        missing = [p for p in cards.listed_files() if not p.exists()]
        self.assertEqual(missing, [])

    def test_every_column_and_field_is_described(self) -> None:
        undescribed = []
        for path in cards.listed_files():
            rel = path.relative_to(ROOT).as_posix()
            frame = cards.table(path)
            if frame is not None:
                undescribed += [f"{rel}: {c}" for c in frame.columns if not cards.describe(rel, str(c))]
            elif path.suffix == ".jsonl":
                undescribed += [f"{rel}: {f}" for f in cards.jsonl_fields(path)[1] if not cards.describe("", f)]
        self.assertEqual(undescribed, [])

    def test_patterns_name_the_feature(self) -> None:
        self.assertIn("F3", cards.describe("any.csv", "z_F3"))
        self.assertIn("Isolation Forest", cards.describe("any.csv", "flag_isolation_forest"))
        self.assertIsNone(cards.describe("any.csv", "no_such_column"))


class FillTests(unittest.TestCase):
    def test_replaces_only_between_markers_and_is_idempotent(self) -> None:
        text = "intro\n<!-- generated:x -->\nold\n<!-- /generated:x -->\noutro\n"
        once = cards.fill(text, {"x": lambda: "new"})
        self.assertEqual(once, "intro\n<!-- generated:x -->\nnew\n<!-- /generated:x -->\noutro\n")
        self.assertEqual(cards.fill(once, {"x": lambda: "new"}), once)

    def test_missing_marker_is_an_error(self) -> None:
        with self.assertRaises(SystemExit):
            cards.fill("no markers here", {"x": lambda: "new"})


if __name__ == "__main__":
    unittest.main()
