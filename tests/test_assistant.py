"""Tests for the Assistant's guardrail, the DS7 phrasings, and the 3B model's reply masking (plan §9-10).

Run from the project root:  python -m pytest -q
The masking test runs only where transformers is installed (the GPU environment) and the base
model's tokenizer is already in the local Hugging Face cache.
"""
from __future__ import annotations

import importlib.util
import unittest

from explorer import assistant, ds7
from explorer.settings import SETTINGS


def tx(flagged: bool = True, scored: bool = True) -> dict:
    """A TransactionDetail with one flag-level and one contributing feature."""
    features = [
        {"id": "F1", "value": 0.5, "usual": 0.05, "z": 4.1, "role": "flag"},
        {"id": "F2", "value": 2, "usual": 1.0, "z": 0.3, "role": "within-usual"},
        {"id": "F3", "value": 30.0, "usual": 10.0, "z": 2.2, "role": "contributing"},
        {"id": "F4", "value": 9.5, "usual": 1.0, "z": 1.1, "role": "within-usual"},
    ]
    return {"method": {"name": "Robust z baseline (A1)"}, "scored": scored, "flagged": flagged, "direction": "out",
            "features": features}


class GuardrailTests(unittest.TestCase):
    def test_every_phrasing_passes_and_repeats_the_flagged_values(self):
        for flagged in (True, False):
            t = tx(flagged)
            messages, facts = assistant.prompt_messages(t), assistant.summary(t)
            for k in range(SETTINGS["ds7"]["phrasings"]):
                reply = ds7.phrase(t, k)
                self.assertTrue(assistant.check(reply, messages, facts)["passed"], (flagged, k, reply))
                self.assertEqual(SETTINGS["limitationLine"] in reply, flagged)
                self.assertEqual(assistant.coverage(reply, facts), 1.0)

    def test_unscored_phrasings_pass(self):
        t = tx(scored=False)
        for k in range(SETTINGS["ds7"]["phrasings"]):
            self.assertTrue(assistant.check(ds7.phrase(t, k), assistant.prompt_messages(t), assistant.summary(t))["passed"])

    def test_invented_number_missing_line_and_banned_word_each_fail(self):
        t = tx()
        messages, facts = assistant.prompt_messages(t), assistant.summary(t)
        good = ds7.phrase(t, 0)
        self.assertFalse(assistant.check(good.replace("0.5 BTC", "0.7 BTC"), messages, facts)["numbers_ok"])
        self.assertFalse(assistant.check(good.replace(SETTINGS["limitationLine"], ""), messages, facts)["limitation_ok"])
        self.assertFalse(assistant.check(good + " This looks suspicious.", messages, facts)["wording_ok"])

    def test_explain_falls_back_to_the_template_when_the_answer_fails(self):
        t = tx()
        result = assistant.explain(t, lambda messages: "It moved 123 BTC.")
        self.assertEqual(result["source"], "template")
        self.assertEqual(result["text"], ds7.phrase(t, 0))
        self.assertEqual(assistant.explain(t, lambda m: ds7.phrase(t, 2))["source"], "assistant")

    def test_rouge_l(self):
        self.assertEqual(assistant.rouge_l("a b c", "a b c"), 1.0)
        self.assertAlmostEqual(assistant.rouge_l("a x c", "a b c"), 2 / 3)
        self.assertEqual(assistant.rouge_l("", "a"), 0.0)

    def test_split_is_decided_by_address(self):
        p = SETTINGS["ds7"]
        self.assertEqual({ds7.split_of("addr-1", p) for _ in range(3)}, {ds7.split_of("addr-1", p)})
        shares = [ds7.split_of(f"a{i}", p) for i in range(2000)]
        self.assertAlmostEqual(shares.count("train") / 2000, p["trainShare"], delta=0.03)


def qwen_tokenizer():
    from transformers import AutoTokenizer

    from lm.qlora import load_config

    cfg = load_config("configs/qwen3b_explain.json")
    return AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg["revision"], local_files_only=True)


def tokenizer_available() -> bool:
    if importlib.util.find_spec("transformers") is None:
        return False
    try:
        qwen_tokenizer()
        return True
    except OSError:
        return False


@unittest.skipUnless(tokenizer_available(), "needs transformers and the cached Qwen tokenizer (GPU environment)")
class QwenMaskTests(unittest.TestCase):
    def test_labels_cover_exactly_the_replies(self):
        from lm.qlora import IGNORE, encode

        tok = qwen_tokenizer()
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "Answer one."}, {"role": "user", "content": "Q2"},
                    {"role": "assistant", "content": "Answer two."}]
        ids, labels, truncated = encode(tok, messages, 1024)
        self.assertFalse(truncated)
        learnt = tok.decode([i for i, l in zip(ids, labels) if l != IGNORE])
        self.assertIn("Answer one.", learnt)
        self.assertIn("Answer two.", learnt)
        self.assertNotIn("Q1", learnt)
        self.assertNotIn("assistant", learnt)
        self.assertEqual(tok.decode(ids), tok.apply_chat_template(messages, tokenize=False))

    def test_long_conversation_is_cut_after_the_last_reply_that_fits(self):
        from lm.qlora import encode

        tok = qwen_tokenizer()
        messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "short"},
                    {"role": "user", "content": "word " * 600}, {"role": "assistant", "content": "late"}]
        ids, labels, truncated = encode(tok, messages, 200)
        self.assertTrue(truncated)
        self.assertLessEqual(len(ids), 200)
        self.assertIsNone(encode(tok, [{"role": "user", "content": "word " * 600},
                                       {"role": "assistant", "content": "x"}], 50))


if __name__ == "__main__":
    unittest.main()
