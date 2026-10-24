"""Tests for TxGPT's tokenizer and token files (plan §6.7): buckets, lengths, windows and masks.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import unittest

from lm import prepare_tx
from lm.tx_tokens import (
    BOS, FIELD_OFFSETS, FIELDS, PAD, TOKENS_PER_TX, VOCAB_SIZE, amount_bucket, count_bucket, fee_bucket, field_of,
    gap_bucket, history_tokens,
)


def tx(t: int, height: int, received: int = 0, sent: int = 0, n_in: int = 1, n_out: int = 2, fee: int = 300,
       vsize: int = 150) -> dict:
    return {"block_time_utc": t, "block_height": height, "received_sats": received, "sent_sats": sent,
            "n_inputs": n_in, "n_outputs": n_out, "fee_sats": fee, "vsize_vb": vsize}


class BucketTests(unittest.TestCase):
    def test_vocabulary_matches_the_plan(self):
        self.assertEqual(VOCAB_SIZE, 94)
        self.assertEqual(TOKENS_PER_TX, 6)

    def test_bucket_edges(self):
        self.assertEqual(gap_bucket(None, False), 0)
        self.assertEqual(gap_bucket(0, True), 1)
        self.assertEqual(gap_bucket(600, False), 2)          # 10 minutes: up to 1/6 hour
        self.assertEqual(gap_bucket(3600, False), 3)         # exactly 1 hour: the "up to 1 h" bucket
        self.assertEqual(gap_bucket(10_000 * 3600, False), 17)  # above a year
        self.assertEqual([amount_bucket(n) for n in (0, 1, 2, 3, 1024, 10**20)], [0, 0, 1, 1, 10, 46])
        self.assertEqual([count_bucket(n) for n in (1, 2, 3, 4, 5, 9, 10, 19, 20, 49, 50, 500)],
                         [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        self.assertEqual([fee_bucket(f, 100) for f in (50, 100, 150, 12_800, 99_999)], [0, 1, 1, 8, 8])

    def test_history_is_bos_then_six_tokens_per_transaction_in_field_order(self):
        rows = [tx(0, 10, received=5000), tx(0, 10, sent=100), tx(7200, 11, received=1)]
        tokens = history_tokens(rows)
        self.assertEqual(len(tokens), 1 + 6 * len(rows))
        self.assertEqual(tokens[0], BOS)
        self.assertEqual([field_of(t) for t in tokens[1:7]], FIELDS)
        self.assertEqual(tokens[1], FIELD_OFFSETS["GAP"] + 0)   # first: no earlier transaction
        self.assertEqual(tokens[7], FIELD_OFFSETS["GAP"] + 1)   # same block as the previous one
        self.assertEqual(tokens[8], FIELD_OFFSETS["DIR"] + 1)   # out
        self.assertTrue(all(t < VOCAB_SIZE for t in tokens))

    def test_planted_row_with_the_previous_height_gets_its_real_gap(self):
        # explorer/inject.py gives a planted row the previous real row's height at a later time.
        tokens = history_tokens([tx(0, 10, received=5000), tx(90 * 86400, 10, received=5000)])
        self.assertEqual(tokens[7], FIELD_OFFSETS["GAP"] + gap_bucket(90 * 86400, False))


class WindowTests(unittest.TestCase):
    seq_len = prepare_tx.SEQ_TX * TOKENS_PER_TX

    def history(self, n: int) -> list[int]:
        return history_tokens([tx(i * 3600, i, received=1000 + i) for i in range(n)])

    def test_short_history_is_one_padded_window(self):
        [(w, m)] = prepare_tx.windows(self.history(10), self.seq_len)
        self.assertEqual(len(w), self.seq_len)
        self.assertEqual(w[0], BOS)
        self.assertEqual(sum(m), 60)  # 61 real tokens, the first never a target
        self.assertEqual(w[61:], [PAD] * (self.seq_len - 61))

    def test_long_history_windows_overlap_on_transaction_boundaries_and_cover_the_end(self):
        n = 300
        tokens = self.history(n)
        out = prepare_tx.windows(tokens, self.seq_len)
        self.assertTrue(all(len(w) == self.seq_len and len(m) == self.seq_len for w, m in out))
        self.assertEqual(out[0][0][0], BOS)
        self.assertTrue(all(BOS not in w for w, _ in out[1:]))  # BOS only at a true start
        self.assertTrue(all(field_of(w[0]) == "GAP" for w, _ in out[1:]))
        last_tx_tokens = tokens[-TOKENS_PER_TX:]
        last_window = [t for t in out[-1][0] if t != PAD]
        self.assertEqual(last_window[-TOKENS_PER_TX:], last_tx_tokens)

    def test_split_is_by_address_and_deterministic(self):
        ids = [f"a{i:03d}" for i in range(200)]
        split = prepare_tx.split_addresses(ids)
        self.assertEqual(sorted(sum(split.values(), [])), ids)
        self.assertEqual((len(split["train"]), len(split["val"]), len(split["test"])), (180, 10, 10))
        self.assertEqual(split, prepare_tx.split_addresses(list(reversed(ids))))


if __name__ == "__main__":
    unittest.main()
