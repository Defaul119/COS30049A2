"""Tests for TxGPT scoring and its baselines (plan §10.1): contexts, time safety, NLL bookkeeping.

A tiny untrained model is enough: these check the bookkeeping, not what the model has learned.
Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from lm import tx_baselines, tx_score
from lm.model import GPT, GPTConfig, pad_vocab
from lm.tx_tokens import BOS, TOKENS_PER_TX, VOCAB_SIZE
from tests.test_tx_tokens import tx


def tiny_model() -> GPT:
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=pad_vocab(VOCAB_SIZE), seq_len=768, n_layer=1, n_head=2, d_model=32, mlp_hidden=64)
    return GPT(cfg).eval()


def history(n: int) -> list[dict]:
    return [tx(i * 5000, i, received=1000 * (i % 7 + 1), n_in=1 + i % 4) for i in range(n)]


class ScoreTests(unittest.TestCase):
    def test_contexts_match_training_windows(self):
        rows = history(200)
        early, late = tx_score.contexts(rows, [0, 150])
        self.assertEqual(early[0], BOS)
        self.assertEqual(len(early), 1 + TOKENS_PER_TX)
        self.assertNotEqual(late[0], BOS)  # the history starts outside the window
        self.assertEqual(len(late), (tx_score.CONTEXT_TX + 1) * TOKENS_PER_TX)
        self.assertLessEqual(len(late), 768)

    def test_later_transactions_never_change_an_earlier_score(self):
        model, device = tiny_model(), torch.device("cpu")
        rows = history(40)
        base = tx_score.surprisal(model, rows, device)
        changed = rows[:20] + [dict(r, received_sats=r["received_sats"] * 1000) for r in rows[20:]]
        after = tx_score.surprisal(model, changed, device)
        np.testing.assert_allclose(after[:20], base[:20], rtol=1e-5)
        self.assertEqual(base.shape, (40, TOKENS_PER_TX))
        self.assertTrue((base > 0).all())

    def test_batching_does_not_change_scores(self):
        model, device = tiny_model(), torch.device("cpu")
        rows = history(30)
        np.testing.assert_allclose(tx_score.surprisal(model, rows, device, batch=4),
                                   tx_score.surprisal(model, rows, device, batch=64), rtol=1e-5)


class BaselineTests(unittest.TestCase):
    def test_habit_learns_a_repeated_pattern_and_global_is_a_distribution(self):
        repeated = {"a": np.zeros((20, TOKENS_PER_TX), dtype=np.int64)}
        nll = tx_baselines.habit_nll(repeated)
        self.assertTrue((np.diff(nll[:, 0]) < 0).all())  # each repeat is less surprising
        g = tx_baselines.global_nll(repeated, repeated)
        self.assertEqual(g.shape, (20, TOKENS_PER_TX))

    def test_markov_captures_alternation_that_global_frequencies_cannot(self):
        alternating = np.zeros((40, TOKENS_PER_TX), dtype=np.int64)
        alternating[1::2, 1] = 1  # direction flips every transaction: IN, OUT, IN, OUT ...
        data = {"a": alternating}
        markov, glob = tx_baselines.markov_nll(data, data), tx_baselines.global_nll(data, data)
        self.assertLess(markov[1:, 1].mean(), glob[1:, 1].mean())
        self.assertEqual(markov[0, 1], glob[0, 1])  # no previous transaction: falls back to global


if __name__ == "__main__":
    unittest.main()
