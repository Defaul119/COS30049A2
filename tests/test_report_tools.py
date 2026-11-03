"""Tests for the report tools: the word count follows the brief's rule, and numbers are looked up exactly.

Serves the brief's 3,000-word limit (tools/report_wordcount.py) and the rule that every reported number comes from
a pipeline file (tools/report_numbers.py): a lookup that matches no row or two rows must fail, never guess.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wordcount = load("report_wordcount")
numbers = load("report_numbers")


class WordCountTests(unittest.TestCase):
    def test_floats_comments_and_tables_are_not_counted(self) -> None:
        tex = ("One two three. % a comment with five words\n"
               "\\begin{figure}[ht]\\caption{Many words in a caption here}\\end{figure}\n"
               "\\input{generated/tab_ds1.tex}\n\\begin{table}[ht]words in a table\\end{table}")
        self.assertEqual(wordcount.words(tex), 3)

    def test_a_number_macro_is_one_word_and_commands_keep_their_text(self) -> None:
        self.assertEqual(wordcount.words("Recall was \\V{ds1.baseline.mean} on \\emph{every} seed."), 6)

    def test_labels_and_references_are_not_words(self) -> None:
        self.assertEqual(wordcount.words("See Table~\\ref{tab:ds1}.\\label{sec:x}"), 2)


class NumberLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame({"method": ["a", "b", "b"], "value": [1.0, 2.0, 3.0]})

    def test_one_matching_row_is_returned(self) -> None:
        self.assertEqual(numbers.row(self.df, method="a")["value"], 1.0)

    def test_no_row_or_several_rows_is_an_error(self) -> None:
        with self.assertRaises(LookupError):
            numbers.row(self.df, method="c")
        with self.assertRaises(LookupError):
            numbers.row(self.df, method="b")

    def test_formats(self) -> None:
        self.assertEqual(numbers.count(7966), "7{,}966")
        self.assertEqual(numbers.pct(0.1043), "10.4\\%")
        self.assertEqual(numbers.num(0.65507), "0.655")


if __name__ == "__main__":
    unittest.main()
