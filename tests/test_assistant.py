"""Tests for the Assistant's guardrail, the DS7 phrasings and the writers (plan §9-10).

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

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
        first = assistant.explain(t, [("text150m", good), ("text150m_v2", bad)])
        self.assertEqual(first["source"], "text150m")
        self.assertEqual(len(first["attempts"]), 1)
        second = assistant.explain(t, [("text150m", bad), ("text150m_v2", good)])
        self.assertEqual((second["source"], second["text"]), ("text150m_v2", ds7.phrase(t, 2)))
        neither = assistant.explain(t, [("text150m", bad), ("text150m_v2", bad)])
        self.assertEqual((neither["source"], neither["text"]), ("template", ds7.phrase(t, 0)))
        self.assertFalse(neither["attempts"][1]["checks"]["numbers_ok"])

    def test_comparison_phrase_against_its_z_sign_fails_only_the_direction_check(self):
        t = tx()
        messages, facts = assistant.prompt_messages(t), assistant.summary(t)
        good = ds7.phrase(t, 0)
        self.assertEqual(assistant.wrong_directions(good, facts), [])
        bad = assistant.check(good.replace("Larger amount", "Smaller amount"), messages, facts)
        self.assertEqual((bad["numbers_ok"], bad["direction_ok"], bad["passed"]), (True, False, False))
        # Within-usual features count too: F2's z is +0.3, so "quieter week" contradicts it wherever it appears.
        self.assertEqual(assistant.wrong_directions(good + "\nA quieter week than usual.", facts),
                         ["quieter week than usual"])

    def test_z_sign_leaves_a_rounded_zero_open(self):
        self.assertEqual([assistant.z_sign(z) for z in ("4.1", "-0.7", "-0.0", "0.0", None)], [1, -1, -1, None, None])

    def test_round1_answers_with_a_wrong_comparison_word_are_caught(self):
        # The 152M model's cases found by hand on 25 Oct (context.txt), one per test set.
        import json
        from pathlib import Path

        import pandas as pd

        from lm.eval_explain import facts_of

        root = Path(__file__).resolve().parents[1]
        cases = [("ds7/test.jsonl", "text150m", "test-00127", "larger amount than usual"),
                 ("ds10/explain.jsonl", "text150m_ds10", "ds10-00153", None)]
        for data, table, rid, phrase in cases:
            rows = (json.loads(line) for line in open(root / "data" / "processed" / data, encoding="utf-8"))
            row = next(r for r in rows if r["id"] == rid)
            answers = pd.read_csv(root / "reports" / "tables" / f"assistant_eval_{table}.csv").set_index("id")
            wrong = assistant.wrong_directions(answers.loc[rid, "answer"], facts_of(row["messages"][:-1]))
            self.assertTrue(wrong, rid)
            if phrase:
                self.assertEqual(wrong, [phrase])

    def test_key_facts_accept_any_listed_phrase(self):
        facts = ["median", ["MAD", "median absolute deviation"], "1.4826"]
        self.assertEqual(assistant.key_facts_found("The  Median and its median absolute deviation", facts), 2 / 3)
        self.assertIsNone(assistant.key_facts_found("anything", []))

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

    def test_a_glossary_row_can_carry_its_own_split_and_key_facts(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "glossary.jsonl"
            rows = [{"topic": "t", "question": "Q?", "answer": "A.", "split": "test", "key_facts": ["A"]},
                    {"topic": "t", "question": "Q2?", "answer": "A."}]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            held, taught = ds7.glossary_pairs(path, SETTINGS["ds7"])
        self.assertEqual((held["split"], held["key_facts"]), ("test", ["A"]))
        self.assertEqual(taught["split"], ds7.split_of("t", SETTINGS["ds7"]))

    def test_split_is_decided_by_address(self):
        p = SETTINGS["ds7"]
        self.assertEqual({ds7.split_of("addr-1", p) for _ in range(3)}, {ds7.split_of("addr-1", p)})
        shares = [ds7.split_of(f"a{i}", p) for i in range(2000)]
        self.assertAlmostEqual(shares.count("train") / 2000, p["trainShare"], delta=0.03)


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


class DS7V2BuildTests(unittest.TestCase):
    """Round 2's build (C7) on the real DS1 data and a few DS2 addresses, against the real frozen v1 test set."""

    def build(self, glossary_rows: list[dict]):
        import json
        import tempfile
        from pathlib import Path
        from unittest import mock

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, tmp, True)
        (tmp / "glossary.jsonl").write_text("\n".join(json.dumps(r) for r in glossary_rows) + "\n", encoding="utf-8")
        # DS1 gives about 1,000 pairs, so this target adds a handful of DS2 addresses.
        small = {**SETTINGS["ds7v2"], "targetPairs": 1100}
        with mock.patch.dict(SETTINGS, {"ds7small": small}):
            ds7.run_build_ds7(tmp, tables_dir=tmp, settings_key="ds7small")
        return tmp

    def test_glossary_test_file_and_no_frozen_prompt_in_training(self):
        import json

        taught = {"topic": "units", "question": "What units are amounts in?", "answer": "BTC.", "split": "train"}
        held = {**taught, "question": "Which unit do amounts use?", "split": "test", "key_facts": ["BTC"]}
        out = self.build([taught, held])
        rows = {s: [json.loads(line) for line in open(out / f"{s}.jsonl", encoding="utf-8")]
                for s in ("train", "val", "test", "glossary_test")}
        self.assertEqual([(r["key_facts"], r["template"]) for r in rows["glossary_test"]], [(["BTC"], "BTC.")])
        self.assertIn("What units are amounts in?", {r["messages"][1]["content"] for r in rows["train"]})
        frozen = ds7.frozen_prompts(ds7.PROJECT_ROOT / SETTINGS["ds7v2"]["frozenTest"])
        self.assertFalse(frozen & {r["messages"][1]["content"] for s in ("train", "val") for r in rows[s]})
        self.assertTrue(any(r["source"] == "DS2" for r in rows["train"]))
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([f["path"].rsplit("/", 1)[-1] for f in manifest["files"]],
                         ["train.jsonl", "val.jsonl", "test.jsonl", "glossary_test.jsonl"])

    def test_training_on_a_frozen_test_question_stops_the_build(self):
        # "Who writes the explanations?" is one of the frozen v1 test set's glossary prompts.
        with self.assertRaises(ValueError):
            self.build([{"topic": "assistant", "question": "Who writes the explanations?", "answer": "A.", "split": "train"}])


class RandomTestSetTests(unittest.TestCase):
    """explorer/random_test.py: synthetic transactions agree with themselves, and real prompts avoid the frozen test."""

    def test_synthetic_values_usuals_and_z_agree_and_stay_in_bounds(self):
        import math
        import random

        from explorer import random_test

        p, rng, flag_z = SETTINGS["randomTest"], random.Random(1), random_test.flag_abs_z()
        scale = {"F1": lambda v: math.log10(1 + v * 1e8), "F2": float, "F3": lambda v: math.log10(1 + v),
                 "F4": lambda v: math.log10(max(v, 0.001))}
        for _ in range(300):
            tx = random_test.synthetic_tx(rng, flag_z)
            for f in tx["features"]:
                self.assertEqual(f["role"], random_test.role(f["z"], flag_z))
                if f["z"] is not None and abs(f["z"]) > 0.01:
                    self.assertEqual(f["z"] > 0, scale[f["id"]](f["value"]) > scale[f["id"]](f["usual"]), f)
            values = {f["id"]: f["value"] for f in tx["features"]}
            if values:
                self.assertLessEqual(values["F1"], 10 ** p["maxSatsLog10"] / 1e8)
                self.assertLessEqual(values["F3"], p["maxGapHours"])
                self.assertLessEqual(values["F2"], p["maxCount"])
            self.assertEqual(tx["flagged"], any(f["role"] == "flag" for f in tx["features"]))
            random_test.record(tx, rng, "synthetic")  # raises if the reply fails the guardrail

    def test_real_prompts_come_from_the_pool_and_never_from_the_frozen_test(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest import mock

        from explorer import random_test

        frozen = ds7.frozen_prompts(ds7.PROJECT_ROOT / SETTINGS["ds7v2"]["frozenTest"])
        v1_test = [json.loads(line) for line in open(ds7.PROJECT_ROOT / SETTINGS["ds7v2"]["frozenTest"], encoding="utf-8")]
        v1_val = [json.loads(line) for line in open(ds7.OUT_DIR / "val.jsonl", encoding="utf-8")]
        pool = [{**r, "template": r["messages"][-1]["content"]} for r in v1_val[:40]] + v1_test[:20]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool), encoding="utf-8")
            small = {**SETTINGS["randomTest"], "realPrompts": 30, "syntheticPrompts": 10}
            with mock.patch.dict(SETTINGS, {"randomTest": small}):
                random_test.run_build_random_test(tmp / "pool.jsonl", tmp / "out")
            real = [json.loads(line) for line in open(tmp / "out" / "real.jsonl", encoding="utf-8")]
            synthetic = (tmp / "out" / "synthetic.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual((len(real), len(synthetic)), (30, 10))
        self.assertFalse({r["messages"][1]["content"] for r in real if r["kind"] != "unscored"} & frozen)


class VerdictAndReasonsTests(unittest.TestCase):
    """The verdict check (guardrail) and the reasons check (evaluation), added 26 Oct."""

    def test_every_phrasing_states_the_right_verdict_and_reasons(self):
        for flagged in (True, False):
            t = tx(flagged)
            facts, template = assistant.summary(t), ds7.phrase(t, 0)
            for k in range(SETTINGS["ds7"]["phrasings"]):
                reply = ds7.phrase(t, k)
                self.assertTrue(assistant.verdict_ok(reply, facts), (flagged, k, reply))
                self.assertTrue(assistant.reasons_ok(reply, template, facts), (flagged, k, reply))

    def test_wrong_verdict_fails_the_guardrail_and_a_missing_reason_fails_reasons(self):
        t = tx(flagged=False)
        messages, facts, good = assistant.prompt_messages(t), assistant.summary(t), ds7.phrase(t, 0)
        bad = good.replace("Not flagged by", "Flagged by")
        self.assertEqual((assistant.check(bad, messages, facts)["verdict_ok"], assistant.check(bad, messages, facts)["passed"]),
                         (False, False))
        dropped = "\n".join(line for line in good.splitlines() if "gap" not in line.lower())
        self.assertFalse(assistant.reasons_ok(dropped, good, facts))
        unscored = tx(scored=False)
        self.assertTrue(assistant.verdict_ok(ds7.phrase(unscored, 3), assistant.summary(unscored)))
        self.assertFalse(assistant.verdict_ok("Not flagged by Robust z baseline (A1).", assistant.summary(unscored)))

    def test_hard_prompts_vary_and_their_replies_pass(self):
        import random

        from explorer import random_test
        from lm.eval_explain import facts_of

        rng, flag_z = random.Random(3), random_test.flag_abs_z()
        rows = [random_test.hard_record(rng, flag_z) for _ in range(120)]  # each reply is checked as it is made
        variants = [r["variant"] for r in rows]
        self.assertGreater(len({v["method"] for v in variants}), 4)
        self.assertTrue(any(v["reworded"] is not None for v in variants) and any(v["shuffled"] for v in variants))
        for r in rows:  # the evaluation reads the facts under any instruction wording
            self.assertEqual(facts_of(r["messages"][:-1]).get("scored", False), r["kind"] != "unscored")


class StrictMeasureTests(unittest.TestCase):
    """Order, number binding and cases (added 26 Oct, after the round-1 rates looked too high)."""

    def test_order_and_binding_catch_swapped_reasons_and_numbers(self):
        t = tx()
        facts, good = assistant.summary(t), ds7.phrase(t, 0)
        self.assertTrue(assistant.order_ok(good, good, facts) and assistant.binding_ok(good, facts))
        lines = good.splitlines()
        reasons = [i for i, line in enumerate(lines) if line.startswith("- ")]
        swapped = lines[:]
        swapped[reasons[0]], swapped[reasons[1]] = lines[reasons[1]], lines[reasons[0]]
        self.assertFalse(assistant.order_ok("\n".join(swapped), good, facts))
        self.assertTrue(assistant.reasons_ok("\n".join(swapped), good, facts))  # same features, wrong order
        self.assertFalse(assistant.binding_ok(good.replace("0.5 BTC moved", "30 hours moved"), facts))

    def test_case_of_names_the_decision(self):
        self.assertEqual(assistant.case_of(assistant.summary(tx())), "flagged, 2 reasons")
        quiet = tx(flagged=False)
        for f in quiet["features"]:
            f["role"] = "within-usual"
        self.assertEqual(assistant.case_of(assistant.summary(quiet)), "not flagged, nothing stands out")
        self.assertEqual(assistant.case_of(assistant.summary(tx(scored=False))), "unscored")

    def test_balanced_set_has_at_most_n_per_case_and_no_other_test_prompt(self):
        import json
        import shutil
        import tempfile
        from pathlib import Path
        from unittest import mock

        from explorer import random_test
        from lm.eval_explain import facts_of

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ("real.jsonl", "manifest.json"):
                shutil.copy(random_test.OUT_DIR / name, tmp / name)
            small = {**SETTINGS["randomTest"], "balancedPerCase": 3}
            with mock.patch.dict(SETTINGS, {"randomTest": small}):
                random_test.run_build_balanced_test(tmp, n_jobs=2)
            rows = [json.loads(line) for line in open(tmp / "balanced.jsonl", encoding="utf-8")]
        cases = [assistant.case_of(facts_of(r["messages"][:-1])) for r in rows]
        self.assertTrue(all(cases.count(c) <= 3 for c in set(cases)) and len(set(cases)) >= 8)
        others = (ds7.frozen_prompts(ds7.PROJECT_ROOT / SETTINGS["ds7v2"]["frozenTest"])
                  | ds7.frozen_prompts(random_test.OUT_DIR / "real.jsonl"))
        self.assertFalse({r["messages"][1]["content"] for r in rows} & others)


class MoreTestSetsTests(unittest.TestCase):
    """build-more-tests (27 Oct): close calls, and every set is fresh."""

    def test_close_call_needs_two_listed_values_that_differ_by_less_than_the_margin(self):
        def facts(*zs, role="contributing"):
            return {"scored": True, "flagged": False,
                    "features": [{"feature": n, "z": z, "role": role} for n, z in zip(("Size", "Burst", "Gap"), zs)]}
        from explorer.random_test import close_call
        self.assertTrue(close_call(facts("2.4", "2.2", "-0.1"), 0.3))
        self.assertFalse(close_call(facts("3.4", "2.2", "-0.1"), 0.3))
        self.assertFalse(close_call(facts("2.2", "-2.2"), 0.3))  # a tie the prompt cannot order is not a close call
        self.assertTrue(close_call(facts("1.1", "-0.9", "0.2", role="within-usual"), 0.3))  # the pair when nothing stands out

    def test_new_sets_repeat_no_earlier_test_prompt(self):
        import json
        import shutil
        import tempfile
        from pathlib import Path
        from unittest import mock

        from explorer import random_test

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ("real.jsonl", "balanced.jsonl", "manifest.json"):
                shutil.copy(random_test.OUT_DIR / name, tmp / name)
            small = {**SETTINGS["randomTest"], "perSourcePerCase": 2, "closePrompts": 5, "extremePrompts": 5,
                     "maxShiftPrompts": 5}
            with mock.patch.dict(SETTINGS, {"randomTest": small}):
                random_test.run_build_more_tests(tmp, n_jobs=2)
            new = {name: [json.loads(line) for line in open(tmp / f"{name}.jsonl", encoding="utf-8")]
                   for name in ("crawl_ds1", "crawl_ds2", "crawl_ds10", "close_calls", "extreme", "max_shift")}
            earlier = (ds7.frozen_prompts(ds7.PROJECT_ROOT / SETTINGS["ds7v2"]["frozenTest"])
                       | ds7.frozen_prompts(tmp / "real.jsonl") | ds7.frozen_prompts(tmp / "balanced.jsonl"))
        real_new = {r["messages"][1]["content"] for n in ("crawl_ds1", "crawl_ds2", "crawl_ds10", "close_calls")
                    for r in new[n] if r["kind"] != "unscored"}
        self.assertFalse(real_new & earlier)
        self.assertTrue(all(r["variant"]["reworded"] is not None and r["variant"]["method"] != "baseline"
                            for r in new["max_shift"]))


if __name__ == "__main__":
    unittest.main()
