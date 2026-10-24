"""Tests for the Assistant's guardrail, the DS7 phrasings, the 3B model's reply masking and adapter saves (plan §9-10).

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

    def test_flagged_without_a_standout_feature_never_reads_as_not_flagged(self):
        # A pattern-level method (e.g. Isolation Forest) can flag a transaction whose features are each
        # within usual; the explanation then lists the largest differences and must not deny the flag.
        t = tx()
        for f in t["features"]:
            f["role"] = "within-usual"
        for k in range(SETTINGS["ds7"]["phrasings"]):
            reply = ds7.phrase(t, k)
            self.assertTrue(assistant.check(reply, assistant.prompt_messages(t), assistant.summary(t))["passed"])
            for denial in ("not flag", "enough to flag", "neither is unusual"):
                self.assertNotIn(denial, reply.lower(), (k, reply))
            self.assertIn(SETTINGS["limitationLine"], reply)

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

    def test_explain_tries_each_writer_then_the_template(self):
        t = tx()
        bad, good = (lambda m: "It moved 123 BTC."), (lambda m: ds7.phrase(t, 2))
        self.assertEqual(assistant.explain(t)["source"], "template")
        first = assistant.explain(t, [("text150m", good), ("qwen3b", bad)])
        self.assertEqual(first["source"], "text150m")
        self.assertEqual(len(first["attempts"]), 1)
        second = assistant.explain(t, [("text150m", bad), ("qwen3b", good)])
        self.assertEqual((second["source"], second["text"]), ("qwen3b", ds7.phrase(t, 2)))
        neither = assistant.explain(t, [("text150m", bad), ("qwen3b", bad)])
        self.assertEqual((neither["source"], neither["text"]), ("template", ds7.phrase(t, 0)))
        self.assertFalse(neither["attempts"][1]["checks"]["numbers_ok"])

    def test_rouge_l(self):
        self.assertEqual(assistant.rouge_l("a b c", "a b c"), 1.0)
        self.assertAlmostEqual(assistant.rouge_l("a x c", "a b c"), 2 / 3)
        self.assertEqual(assistant.rouge_l("", "a"), 0.0)

    def test_glossary_paraphrases_share_their_topic_split(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "glossary.jsonl"
            rows = [{"topic": f"t{t}", "question": f"Question {q} on topic {t}?", "answer": "An answer."}
                    for t in range(40) for q in range(3)]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            pairs = list(ds7.glossary_pairs(path, SETTINGS["ds7"]))
        by_topic: dict[str, set[str]] = {}
        for r, p in zip(rows, pairs):
            by_topic.setdefault(r["topic"], set()).add(p["split"])
        self.assertTrue(all(len(splits) == 1 for splits in by_topic.values()))

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


class SaveAdapterTests(unittest.TestCase):
    def test_new_adapter_replaces_the_old_one_and_no_scratch_folder_is_left(self):
        import tempfile
        from pathlib import Path

        from lm.qlora import save_adapter

        class Stub:
            """Writes what peft's save_pretrained writes, with a weights file that says which save it is."""
            def __init__(self, version: str):
                self.version = version

            def save_pretrained(self, folder: Path) -> None:
                folder.mkdir(parents=True)
                (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
                (folder / "adapter_model.safetensors").write_text(self.version, encoding="utf-8")

        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "adapter"
            save_adapter(Stub("one"), target)
            save_adapter(Stub("two"), target)
            self.assertEqual((target / "adapter_model.safetensors").read_text(encoding="utf-8"), "two")
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["adapter"])


class EvalGeneratorTests(unittest.TestCase):
    def test_text150m_generator_answers_a_ds7_prompt(self):
        import tempfile
        from pathlib import Path

        import torch

        from lm.writers import text150m_generator
        from lm.model import GPT, GPTConfig
        from lm.tokenizer import train_tokenizer

        with tempfile.TemporaryDirectory() as d:
            tok = train_tokenizer(["the amount was unusual for this address"] * 50, vocab_size=300)
            tok.save(str(Path(d) / "tokenizer.json"))
            cfg = GPTConfig(vocab_size=320, seq_len=1024, n_layer=1, n_head=2, d_model=32, mlp_hidden=64)
            torch.save({"model": GPT(cfg).state_dict(), "model_config": cfg.to_dict()}, Path(d) / "model_best.pt")
            answer = text150m_generator(str(Path(d) / "model_best.pt"), 5)(assistant.prompt_messages(tx()))
        self.assertIsInstance(answer, str)


if __name__ == "__main__":
    unittest.main()
