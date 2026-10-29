"""Tests for tools/compare_tables.py: a rebuilt table matches only within the tolerance (README 11, milestone M4).

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_tables", ROOT / "tools" / "compare_tables.py")
compare_tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_tables)


def table(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


class CompareTablesTests(unittest.TestCase):
    def test_identical_and_within_tolerance(self) -> None:
        old = table(method=["a", "b"], recall=[0.5, np.nan], flagged=[239, 10])
        self.assertEqual(compare_tables.compare(old.copy(), old, 0.01)[0], "same")
        new = old.assign(recall=[0.505, np.nan])
        self.assertEqual(compare_tables.compare(new, old, 0.01)[0], "within")

    def test_beyond_tolerance_text_missing_and_shape_differ(self) -> None:
        old = table(method=["a", "b"], recall=[0.5, 0.2])
        self.assertTrue(compare_tables.compare(old.assign(recall=[0.52, 0.2]), old, 0.01)[0].startswith("differs"))
        self.assertTrue(compare_tables.compare(old.assign(method=["a", "c"]), old, 0.01)[0].startswith("differs"))
        self.assertTrue(compare_tables.compare(old.assign(recall=[0.5, np.nan]), old, 0.01)[0].startswith("differs"))
        self.assertTrue(compare_tables.compare(old.iloc[:1], old, 0.01)[0].startswith("differs"))

    def test_large_values_are_compared_relative_and_run_columns_ignored(self) -> None:
        old = table(threshold=[1000.0], seconds=[12.0], commit=["abc"])
        new = table(threshold=[1005.0], seconds=[99.0], commit=["def"])
        self.assertEqual(compare_tables.compare(new, old, 0.01)[0], "within")


if __name__ == "__main__":
    unittest.main()
