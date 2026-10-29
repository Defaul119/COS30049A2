"""152M stage B3's weak-case examples (explorer/weak_cases.py) and the writer gate (stages.run_compare_writers)."""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from explorer import assistant, stages, weak_cases
from explorer.random_test import REWORDED, flag_abs_z, hard_record

DATA = weak_cases.OUT_DIR


def rows(name: str) -> list[dict]:
    with open(DATA / name, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_training_instructions_are_never_test_instructions():
    assert not set(weak_cases.TRAIN_INSTRUCTIONS) & (set(REWORDED) | {assistant.INSTRUCTION})
    assert all("\n" not in i for i in weak_cases.TRAIN_INSTRUCTIONS)  # the facts start on the second line


def test_hard_record_defaults_draw_as_the_frozen_sets_were_built():
    # The new optional arguments must not change the draws of hard.jsonl and max_shift.jsonl.
    a = hard_record(random.Random(5), flag_abs_z())
    b = hard_record(random.Random(5), flag_abs_z(), None, REWORDED, None)
    assert a == b


@pytest.mark.skipif(not (DATA / "train.jsonl").exists(), reason="build-weak-cases not run")
def test_built_files_leave_out_every_frozen_prompt_and_each_other():
    frozen = weak_cases.frozen_test_prompts()
    train, dev = rows("train.jsonl"), rows("dev.jsonl")
    scored = lambda rs: {r["messages"][1]["content"] for r in rs if r["kind"] != "unscored"}
    assert not scored(train) & frozen
    assert not scored(dev) & frozen
    assert not scored(train) & scored(dev)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert len(train) == manifest["settings"]["trainPrompts"] and len(dev) == manifest["settings"]["devPrompts"]


@pytest.mark.skipif(not (DATA / "dev.jsonl").exists(), reason="build-weak-cases not run")
def test_every_reply_passes_the_guardrail_and_uses_its_own_wordings():
    allowed = set(weak_cases.TRAIN_INSTRUCTIONS) | {assistant.INSTRUCTION}
    for r in rows("dev.jsonl"):
        prompt = r["messages"][1]["content"]
        assert prompt.partition("\n")[0] in allowed or r["kind"] == "unscored"
        facts = json.loads(prompt.partition("\n")[2]) if r["kind"] != "unscored" else {"scored": False}
        assert assistant.check(r["messages"][-1]["content"], r["messages"][:-1], facts)["passed"]


def answers(path: Path, correct: list[bool], cases: list[str], passed: list[bool] | None = None) -> Path:
    n = len(correct)
    pd.DataFrame({"id": [f"w-{i}" for i in range(n)], "kind": "flagged", "prompt": [f"p{i}" for i in range(n)],
                  "answer": "x", "numbers_ok": True, "passed": passed or [True] * n, "correct": correct,
                  "case": cases}).to_csv(path, index=False)
    return path


def test_writer_gate_accepts_a_clear_gain_and_rejects_a_fallen_case(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        monkeypatch.setitem(stages.SETTINGS, "stages", {**stages.SETTINGS["stages"], "table": "stages.csv"})
        monkeypatch.setattr(stages, "TABLES_DIR", tmp)
        monkeypatch.setattr(stages, "git_commit", lambda: "test")
        cases = ["a"] * 100 + ["b"] * 100
        before = answers(tmp / "assistant_eval_r1_dev.csv", [False] * 50 + [True] * 50 + [True] * 50 + [False] * 50, cases)
        better = answers(tmp / "assistant_eval_r2_dev.csv", [True] * 100 + [True] * 50 + [False] * 50, cases)
        stages.run_compare_writers([before], [better], "B3", "text150m", "M0")
        row = pd.read_csv(tmp / "stages.csv").iloc[-1]
        assert bool(row["accepted"]) and row["diff_previous"] == pytest.approx(0.25)
        worse_b = answers(tmp / "assistant_eval_r3_dev.csv", [True] * 100 + [True] * 20 + [False] * 80, cases)
        stages.run_compare_writers([before], [worse_b], "B3", "text150m", "M0")
        row = pd.read_csv(tmp / "stages.csv").iloc[-1]
        assert not bool(row["accepted"]) and "case b fell" in row["reasons"]
