"""Weak-case explanation examples for the 152M explanation stage (docs/TRAINING_PLAN.md stage B3).

At M0 the round-1 writer is fully correct on 0-56% of the hard and maximal-shift test prompts per case
(assistant_eval_by_case_random_hard.csv, _max_shift.csv): it was taught one instruction, the baseline's name, facts
in DS7's order, and few prompts with three or four reasons. It invents method names ("Robusta") and loses the
verdict when a method other than the baseline flags independently of the features. This step builds training and
dev examples of exactly those kinds, so the explanation stage sees them:

    train.jsonl   added to DS7 v2's training conversations by the stage-data notebook (lm.prepare_chat --extra)
    dev.jsonl     the B3 gate's held-out slice (TRAINING_PLAN section 2, dev tier for the writer)

Three rules keep the test tier honest:
1. Own seeds (settings.json -> weakCases), never the randomized test sets' 50000-50003.
2. Own instruction wordings (TRAIN_INSTRUCTIONS): the test sets' reworded instructions (random_test.REWORDED) stay
   unseen, so a gain on max_shift.jsonl is a gain on new wording, not on a copied one.
3. Every prompt of a frozen test file (stages frozenFiles) is excluded, and dev never repeats a train prompt.

Each row is built as random_test.hard_record builds the test rows: a synthetic transaction (random_test.
synthetic_tx), explained for the baseline or any other method (whose flag need not follow the features), maybe
with a reworded instruction and shuffled facts, with a reply in one of DS7's phrasings that passes the guardrail.
A share of rows draws z from a mixture with more contributing and flag-level values, so prompts with three or four
reasons are common.

Usage:  python -m explorer.cli build-weak-cases
Writes data/processed/weak_cases/{train,dev}.jsonl and manifest.json (SHA-256, cases per file).
"""
from __future__ import annotations

import json
import logging
import random
from glob import glob
from pathlib import Path
from typing import Any

from . import assistant
from .ds7 import frozen_prompts
from .random_test import REWORDED, flag_abs_z, hard_record, write
from .settings import PROJECT_ROOT, SETTINGS
from .snapshot import utc_now

log = logging.getLogger(__name__)

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "weak_cases"

# Instructions for training only: none is in random_test.REWORDED or is assistant.INSTRUCTION. Each fits on one line,
# so the facts still start on the second, where lm.eval_explain reads them.
TRAIN_INSTRUCTIONS = [
    "Tell an analyst, in plain words, what this result says about the transaction. Use only these facts:",
    "Describe this transaction's result for an investigator, relying only on the facts below:",
    "Give a plain-language explanation of the scoring result below. Use nothing but these facts:",
    "An analyst asks what this result means. Answer from these facts only:",
    "Explain to an analyst whether anything about this transaction stands out, using only these facts:",
    "Put this result into plain words for a reviewer. Base the answer only on the following facts:",
    "Using nothing beyond the facts below, explain this transaction's result to an analyst:",
    "Briefly explain the result for this transaction to an analyst; only these facts may be used:",
]


def frozen_test_prompts() -> set[str]:
    """Every prompt of the frozen test and final files (stages frozenFiles), so none is trained or tuned on."""
    used: set[str] = set()
    for pattern in SETTINGS["stages"]["frozenFiles"]:
        for path in sorted(glob(str(PROJECT_ROOT / pattern))):
            if path.endswith(".jsonl"):
                used |= frozen_prompts(Path(path))
    return used


def draw(rng: random.Random, n: int, flag_z: float, excluded: set[str]) -> list[dict[str, Any]]:
    """`n` weak-case rows with prompts outside `excluded` (which grows with each prompt drawn)."""
    p = SETTINGS["weakCases"]
    shares = {k: p[k] for k in ("hardBaselineShare", "hardOtherFlagShare", "rewordShare", "shuffleShare")}
    many = {"zMixture": p["manyReasonMixture"]}
    rows = []
    while len(rows) < n:
        ranges = many if rng.random() < p["manyReasonShare"] else None
        row = hard_record(rng, flag_z, shares, TRAIN_INSTRUCTIONS, ranges)
        prompt = row["messages"][1]["content"]
        # Unscored prompts carry no facts and are all alike (ds7.frozen_prompts leaves them out too).
        if row["kind"] != "unscored" and prompt in excluded:
            continue
        excluded.add(prompt)
        rows.append({**row, "source": "weak"})
    return rows


def cases(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        c = assistant.case_of(json.loads(r["messages"][1]["content"].partition("\n")[2]))
        out[c] = out.get(c, 0) + 1
    return dict(sorted(out.items()))


def run_build_weak_cases(out_dir: Path = OUT_DIR) -> int:
    """Step `build-weak-cases`: write train.jsonl, dev.jsonl and manifest.json (cases per file, SHA-256)."""
    p = SETTINGS["weakCases"]
    if set(TRAIN_INSTRUCTIONS) & (set(REWORDED) | {assistant.INSTRUCTION}):
        raise ValueError("a training instruction is also a test-set or DS7 instruction")
    flag_z, excluded = flag_abs_z(), frozen_test_prompts()
    log.info("%d frozen test prompts excluded", len(excluded))
    train = draw(random.Random(p["seed"]), p["trainPrompts"], flag_z, excluded)
    dev = draw(random.Random(p["devSeed"]), p["devPrompts"], flag_z, excluded)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"train.jsonl": write(out_dir / "train.jsonl", train, "wtrain"),
             "dev.jsonl": write(out_dir / "dev.jsonl", dev, "wdev")}
    manifest = {"dataset": "Weak-case explanation examples for the 152M explanation stage (TRAINING_PLAN B3)",
                "built_utc": utc_now(), "settings": p, "flag_abs_z": flag_z, "excluded_prompts": len(excluded),
                "cases": {"train": cases(train), "dev": cases(dev)},
                "files": [{"path": f"data/processed/weak_cases/{name}", "sha256": d} for name, d in files.items()]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("train", train), ("dev", dev)):
        log.info("%s: %d prompts; %s", name, len(rows), manifest["cases"][name])
    return 0
