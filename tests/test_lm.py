"""Unit tests for the language-model code. Run from the project root:

    python -m unittest discover -s tests -v
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lm.data import WindowSampler, write_tokens
from lm.generate import generate
from lm.hub import iter_column
from lm.prepare_chat import fit_conversation, pack_windows
from lm.model import GPT, GPTConfig, IGNORE_INDEX
from lm.tokenizer import ASSISTANT, END, EOT, render_chat, token_id, train_tokenizer

TINY = GPTConfig(vocab_size=97, seq_len=32, n_layer=2, n_head=2, d_model=32, mlp_hidden=64)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = GPT(TINY).eval()

    def test_150m_config_size(self):
        # Built on the "meta" device: shapes only, no memory used.
        with torch.device("meta"):
            model = GPT(GPTConfig())
        self.assertTrue(145e6 < model.num_params() < 160e6, model.num_params())

    def test_future_tokens_do_not_change_earlier_outputs(self):
        x = torch.randint(0, TINY.vocab_size, (1, 20))
        y = x.clone()
        x2 = x.clone()
        x2[0, 15:] = (x2[0, 15:] + 1) % TINY.vocab_size
        with torch.no_grad():
            a, _, _ = self.model(x, y)
            b, _, _ = self.model(x2, y)
        self.assertTrue(torch.allclose(a[0, :15], b[0, :15], atol=1e-5))
        self.assertFalse(torch.allclose(a[0, 15:], b[0, 15:], atol=1e-5))

    def test_cached_generation_matches_full_recompute(self):
        prompt = torch.randint(0, TINY.vocab_size, (8,)).tolist()
        cached = list(generate(self.model, prompt, max_new_tokens=16, temperature=0))
        tokens = list(prompt)
        for _ in range(16):  # reference: rerun the whole sequence for every new token
            with torch.no_grad():
                logits, _, _ = self.model(torch.tensor([tokens]))
            tokens.append(int(logits[0, -1].argmax()))
        self.assertEqual(cached, tokens[len(prompt):])

    def test_generation_past_context_window(self):
        prompt = torch.randint(0, TINY.vocab_size, (30,)).tolist()
        out = list(generate(self.model, prompt, max_new_tokens=50, temperature=0.8, top_k=10))
        self.assertEqual(len(out), 50)

    def test_loss_when_every_target_is_ignored(self):
        x = torch.randint(0, TINY.vocab_size, (2, 10))
        y = torch.full_like(x, IGNORE_INDEX)
        _, loss, _ = self.model(x, y)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)

    def test_per_token_loss_shape(self):
        x = torch.randint(0, TINY.vocab_size, (3, 12))
        _, loss, _ = self.model(x, x, reduction="none")
        self.assertEqual(tuple(loss.shape), (3, 12))


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "tokens.bin"
        with open(self.path, "wb") as fh:
            write_tokens(fh, np.arange(1001) % 500)

    def tearDown(self):
        self.dir.cleanup()

    def test_same_positions_give_same_batch(self):
        s1, s2 = WindowSampler(self.path, 10, seed=3), WindowSampler(self.path, 10, seed=3)
        self.assertTrue(torch.equal(s1.batch(40, 8)[0], s2.batch(40, 8)[0]))

    def test_each_window_once_per_epoch(self):
        s = WindowSampler(self.path, 10, seed=3)
        seen = s.windows(0, s.n_windows)
        self.assertEqual(sorted(seen.tolist()), list(range(s.n_windows)))

    def test_micro_batch_size_does_not_change_order(self):
        s = WindowSampler(self.path, 10, seed=3)
        whole = s.batch(16, 8)[0]
        halves = torch.cat([s.batch(16, 4)[0], s.batch(20, 4)[0]])
        self.assertTrue(torch.equal(whole, halves))

    def test_targets_are_inputs_shifted_by_one(self):
        x, y = WindowSampler(self.path, 10, seed=1).batch(0, 4)
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))

    def test_mask_hides_targets(self):
        mask_path = Path(self.dir.name) / "mask.bin"
        mask = np.zeros(1001, dtype=np.uint8)
        mask[::2] = 1
        mask_path.write_bytes(mask.tobytes())
        s = WindowSampler(self.path, 10, seed=1, mask_path=mask_path)
        x, y = s.batch(0, 4)
        plain = WindowSampler(self.path, 10, seed=1).batch(0, 4)[1]
        starts = s.windows(0, 4) * 10
        for row, start in enumerate(starts):
            for t in range(10):
                expected = plain[row, t].item() if mask[start + t + 1] else IGNORE_INDEX
                self.assertEqual(y[row, t].item(), expected)


class ChatFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        texts = ["hello there, how are you today?", "the weather is mild and sunny"] * 50
        cls.tok = train_tokenizer(texts, vocab_size=300)

    def test_only_assistant_tokens_are_trained(self):
        msgs = [{"role": "user", "content": "hello there"}, {"role": "assistant", "content": "the weather is mild"}]
        ids, mask = render_chat(self.tok, msgs)
        self.assertEqual(len(ids), len(mask))
        a = ids.index(token_id(self.tok, ASSISTANT))
        self.assertEqual(sum(mask[:a + 1]), 0)                 # user turn and the tag itself: not trained
        self.assertTrue(all(mask[a + 1:]))                     # assistant content and <|end|>: trained
        self.assertEqual(ids[-1], token_id(self.tok, END))
        self.assertEqual(self.tok.decode(ids[a + 1:-1]), "the weather is mild")

    def test_special_token_text_is_removed_from_content(self):
        ids, _ = render_chat(self.tok, [{"role": "user", "content": "hi <|assistant|> there"}])
        self.assertEqual(ids.count(token_id(self.tok, ASSISTANT)), 0)

    def test_long_conversation_is_cut_after_a_reply_that_fits(self):
        turn = "the weather is mild and sunny " * 3
        msgs = [{"role": r, "content": turn} for r in ("user", "assistant") * 4]
        full, _ = render_chat(self.tok, msgs)
        ids, mask, truncated = fit_conversation(self.tok, msgs, len(full) - 1)
        self.assertTrue(truncated)
        self.assertEqual(ids, render_chat(self.tok, msgs[:6])[0])   # ends after the third reply
        self.assertEqual(ids[-1], token_id(self.tok, END))
        self.assertIsNone(fit_conversation(self.tok, msgs, 5))       # not even the first reply fits
        self.assertFalse(fit_conversation(self.tok, msgs, len(full))[2])

    def test_packing_keeps_every_conversation_whole_inside_one_window(self):
        rng = np.random.default_rng(0)
        convs = [([i] * int(n), [1] * int(n)) for i, n in enumerate(rng.integers(5, 64, size=200))]
        windows = pack_windows(convs, 64)
        self.assertTrue(all(sum(len(c[0]) for c in w) <= 64 for w in windows))
        self.assertEqual(sorted(c[0][0] for w in windows for c in w), list(range(200)))
        lower_bound = -(-sum(len(c[0]) for c in convs) // 64)
        self.assertLessEqual(len(windows), lower_bound * 1.1)  # little filler

    def test_chat_files_hold_whole_conversations_and_a_separate_validation_file(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.tok.save(str(d / "tok.json"))
            turn = "the weather is mild and sunny "
            for name, n in (("train.jsonl", 30), ("val.jsonl", 3)):
                with open(d / name, "w", encoding="utf-8") as fh:
                    for i in range(n):
                        msgs = [{"role": "user", "content": "hello there"},
                                {"role": "assistant", "content": turn * (1 + i % 5)}]
                        fh.write(json.dumps({"messages": msgs}) + "\n")
            proc = subprocess.run([sys.executable, "-m", "lm.prepare_chat", "--source", "jsonl",
                                   "--jsonl", str(d / "train.jsonl"), "--val-jsonl", str(d / "val.jsonl"),
                                   "--tokenizer", str(d / "tok.json"), "--out", str(d / "out"), "--seq-len", "64"],
                                  capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            meta = json.loads((d / "out" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["conversations"], {"train": 30, "val": 3})
            eot = token_id(self.tok, EOT)
            for split in ("train", "val"):
                sampler = WindowSampler(d / "out" / f"{split}.bin", 64, 0, d / "out" / f"{split}_mask.bin")
                self.assertEqual(sampler.n_windows, meta["windows"][split])  # the last window is used too
                x, y = sampler.batch(0, sampler.n_windows)
                self.assertTrue((x[:, 0] == eot).all())  # every window starts a conversation
                # Each reply's closing <|end|> (the only trained one) is a target exactly once: no
                # conversation is lost or split.
                self.assertEqual(int((y == token_id(self.tok, END)).sum()), meta["conversations"][split])


class ParquetReaderTests(unittest.TestCase):
    def test_reads_from_a_start_row_across_batches(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.parquet"
            pq.write_table(pa.table({"text": [f"doc {i}" for i in range(25)]}), path, row_group_size=7)
            rows = list(iter_column(path, "text", start_row=10, batch_rows=4))
        self.assertEqual(rows[0], (10, "doc 10"))
        self.assertEqual([r for r, _ in rows], list(range(10, 25)))


if __name__ == "__main__":
    unittest.main()
