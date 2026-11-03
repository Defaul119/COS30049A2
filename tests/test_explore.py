"""Tests for the interactive terminal Explorer (explorer/explore.py): choosing an address by number or by pasting
it, the flagged list, one transaction's verdicts and explanation, the what-if, paging, bad input, and the wording
rule on everything it prints (A1 NFR-01).

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from explorer.explore import Console, Explorer
from explorer.settings import SETTINGS
from tests.test_predict import predictor


class ExploreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = predictor()
        txs = cls.p.score_address("busy")["transactions"]
        cls.flagged = next(i for i, t in enumerate(txs, 1) if t["flagged"])

    def run_script(self, inputs: list[str]) -> tuple[int, str]:
        """Run the Explorer on scripted answers; input that runs out ends it as Ctrl+Z / end of file would."""
        out: list[str] = []
        answers = iter(inputs)

        def read(prompt: str) -> str:
            out.append(prompt)
            try:
                return next(answers)
            except StopIteration:
                raise EOFError from None

        code = Explorer(self.p, Console(read, out.append)).run()
        return code, "\n".join(out)

    def test_lists_addresses_and_opens_one_by_number(self):
        code, out = self.run_script(["1", "q"])
        self.assertEqual(code, 0)
        self.assertIn("busy", out)
        self.assertIn("(short history)", out)  # the 4-transaction address is listed, marked, and last
        self.assertIn("Address busy", out)
        self.assertIn("Flagged transactions", out)
        self.assertTrue(out.rstrip().endswith("Goodbye."))

    def test_explaining_a_flagged_transaction_shows_every_verdict_and_the_limitation(self):
        _, out = self.run_script(["busy", str(self.flagged), "q"])
        self.assertIn(f"Transaction #{self.flagged}", out)
        self.assertIn("(decides the flag)", out)
        self.assertIn("written by the template", out)
        self.assertIn(SETTINGS["limitationLine"], out)

    def test_short_history_address_says_why_and_goes_back(self):
        _, out = self.run_script(["tiny", "", "q"])
        self.assertIn("does not have enough earlier history", out)
        self.assertEqual(out.count("Addresses (page 1 of 1)"), 2)

    def test_what_if_scores_a_large_amount(self):
        _, out = self.run_script(["busy", "w", "6", "in", "", "", "q"])
        self.assertIn("Hypothetical transaction", out)
        self.assertIn("FLAG", out)
        self.assertIn("n/a", out)  # TxGPT scores only in advance, never a what-if

    def test_bad_input_is_answered_and_never_echoed(self):
        # A banned word typed by the user must not crash the wording check: input is never echoed.
        code, out = self.run_script(["fraud", "999", "busy", "xyz", "w", "abc", "w", "1", "sideways", "q"])
        self.assertEqual(code, 0)
        self.assertIn("Not on the list", out)
        self.assertIn("Not understood", out)
        self.assertIn("positive number", out)
        self.assertIn("must be in or out", out)

    def test_assistant_falls_back_to_the_template_without_its_weights(self):
        with patch("explorer.predict.load_writers", return_value=([], {"text150m": "weights missing"})):
            _, out = self.run_script(["busy", f"a {self.flagged}", "q"])
        self.assertIn("text150m is not available on this machine: weights missing", out)
        self.assertIn("written by the template", out)

    def test_pages_of_transactions_and_addresses(self):
        per_page = SETTINGS["explore"]["transactionsPerPage"]
        _, out = self.run_script(["busy", "t", "n", "b", "q"])
        self.assertIn(f"All transactions (page 2 of {-(-120 // per_page)})", out)
        with patch.dict(SETTINGS["explore"], {"addressesPerPage": 1}):
            _, out = self.run_script(["n", "q"])
        self.assertIn("Addresses (page 2 of 2)", out)


if __name__ == "__main__":
    unittest.main()
