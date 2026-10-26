"""A randomized test set for both Explorer Assistants: real held-out prompts and synthetic transactions.

Serves the round-2 evaluation (docs/KAGGLE_LM_PLAN.md; the user asked on 26 Oct for a new randomized test of the
152M model and Qwen 3B). Built once with settings -> randomTest -> seed and then frozen (committed), so every
writer, both models in both rounds, answers exactly the same prompts:

    real.jsonl        prompts drawn at random from DS7 v2's test split. Both rounds split DS7 by the same seeded
                      hash of the address, so no writer of either round trained on these addresses; every prompt
                      of the frozen v1 test set is left out, so the two test sets do not overlap.
    synthetic.jsonl   transactions made up at random. Each feature's z is drawn (within usual, contributing or
                      flag-level, either sign, a few within rounding of 0), and its value follows from z, a usual
                      value and a spread on the feature's own scale (baseline -> zTransform), so value, usual and
                      z always agree; roles and the flag come from them as in explorer.predict. Some features have
                      no comparison and some transactions are unscored, as in real data. No writer has seen these
                      numbers, so number faithfulness and the direction check meet new combinations.

Each row carries the template (the ROUGE-L reference) and a reply in a random DS7 phrasing (rouge_own), and every
reply passes the guardrail, as in DS7. Score a writer with
    python -m lm.eval_explain --model <writer> ... --test data/processed/random_test/real.jsonl --tag random_real
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

from . import assistant
from .baseline import MODELS_DIR
from .ds7 import frozen_prompts, phrase
from .explain import METHOD_NAMES, template_explanation
from .settings import PROJECT_ROOT, SETTINGS
from .snapshot import utc_now

log = logging.getLogger(__name__)

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "random_test"
DS7_V2_TEST = PROJECT_ROOT / "data" / "processed" / "ds7_v2" / "test.jsonl"
SATS_PER_BTC = 100_000_000


def flag_abs_z() -> float:
    """The saved baseline's flag threshold, as explorer.predict uses it."""
    return float(json.loads((MODELS_DIR / "baseline.json").read_text(encoding="utf-8"))["chosen"]["refined"]["threshold"])


def role(z: float | None, flag_z: float) -> str:
    """explorer.predict.Predictor.feature_details' rule."""
    if z is None:
        return "no-comparison"
    return "flag" if abs(z) >= flag_z else "contributing" if abs(z) >= SETTINGS["baseline"]["contributingAbsZ"] else "within-usual"


def draw_z(rng: random.Random, p: dict[str, Any], flag_z: float) -> float:
    """A z from the mixture: flag-level, contributing, within rounding of 0, or within usual; either sign."""
    u = rng.random()
    sign = rng.choice((-1, 1))
    zc, clip = SETTINGS["baseline"]["contributingAbsZ"], SETTINGS["features"]["zClip"]
    if u < p["flagShare"]:
        return sign * rng.uniform(flag_z, clip)
    if u < p["flagShare"] + p["contributingShare"]:
        return sign * rng.uniform(zc, flag_z)
    if u < p["flagShare"] + p["contributingShare"] + p["nearZeroShare"]:
        return sign * rng.uniform(0, 0.05)
    return max(-zc + 1e-6, min(zc - 1e-6, rng.gauss(0, p["withinSd"])))


def clipped(z: float) -> float:
    clip = SETTINGS["features"]["zClip"]
    return max(-clip, min(clip, z))


def synthetic_tx(rng: random.Random, flag_z: float) -> dict[str, Any]:
    """One made-up TransactionDetail (the fields explorer.assistant and explorer.explain read)."""
    p = SETTINGS["randomTest"]
    method = {"name": METHOD_NAMES["baseline"]}
    direction = rng.choice(("in", "out"))
    if rng.random() < p["unscoredShare"]:
        return {"method": method, "scored": False, "flagged": False, "direction": direction, "features": []}
    spread = {fid: rng.uniform(*bounds) for fid, bounds in p["spread"].items()}
    mix = p["zMixture"]

    # F1 Size: z on log10(1 + satoshis); shown in BTC.
    # Each value is capped to what a real address could show, and z is computed from the capped value.
    usual_sats = 10 ** rng.uniform(*p["usualSatsLog10"])
    value_log = min(math.log10(1 + usual_sats) + draw_z(rng, mix, flag_z) * spread["F1"], p["maxSatsLog10"])
    value_sats = max(1.0, round(10 ** value_log - 1))
    z1 = clipped((math.log10(1 + value_sats) - math.log10(1 + usual_sats)) / spread["F1"])
    # F2 Burst: a count of earlier transactions in the window; z on the count itself.
    usual_count = round(rng.uniform(*p["usualCount"]), 1)
    count = min(p["maxCount"], max(0, round(usual_count + draw_z(rng, mix, flag_z) * spread["F2"])))
    z2 = clipped((count - usual_count) / spread["F2"])
    # F3 Gap: hours since the previous transaction; z on log10(1 + hours).
    usual_gap = 10 ** rng.uniform(*p["usualGapHoursLog10"])
    gap = 10 ** (math.log10(1 + usual_gap) + draw_z(rng, mix, flag_z) * spread["F3"]) - 1
    gap = min(p["maxGapHours"], max(0.01, gap))
    z3 = clipped((math.log10(1 + gap) - math.log10(1 + usual_gap)) / spread["F3"])
    # F4 Relative size: this amount over the usual amount (F1's usual), against the address's usual ratio.
    ratio = value_sats / max(usual_sats, 1)
    usual_ratio = 10 ** rng.uniform(*p["usualRatioLog10"])
    z4 = clipped((math.log10(max(ratio, 0.001)) - math.log10(max(usual_ratio, 0.001))) / spread["F4"])

    features = [{"id": "F1", "value": value_sats / SATS_PER_BTC, "usual": usual_sats / SATS_PER_BTC, "z": z1},
                {"id": "F2", "value": count, "usual": usual_count, "z": z2},
                {"id": "F3", "value": gap, "usual": usual_gap, "z": z3},
                {"id": "F4", "value": ratio, "usual": usual_ratio, "z": z4}]
    if rng.random() < p["noComparisonShare"]:
        # Too few earlier transactions in the same direction: the size features have no comparison.
        for f in (features[0], features[3]):
            f.update({"usual": None, "z": None})
    for f in features:
        f["role"] = role(f["z"], flag_z)
    return {"method": method, "scored": True, "flagged": any(f["role"] == "flag" for f in features),
            "direction": direction, "features": features}


def record(tx: dict[str, Any], rng: random.Random, source: str) -> dict[str, Any]:
    """A test row like DS7's: the prompt, a reply in a random phrasing (checked by the guardrail) and the template."""
    k = rng.randrange(SETTINGS["ds7"]["phrasings"])
    reply = phrase(tx, k)
    messages = assistant.prompt_messages(tx)
    checks = assistant.check(reply, messages, assistant.summary(tx))
    if not checks["passed"]:
        raise ValueError(f"a synthetic reply fails its own guardrail ({checks}): {reply!r}")
    kind = "unscored" if not tx["scored"] else "flagged" if tx["flagged"] else "not_flagged"
    return {"source": source, "kind": kind, "phrasing": k, "template": template_explanation(tx),
            "messages": messages + [{"role": "assistant", "content": reply}]}


def write(path: Path, rows: list[dict[str, Any]], prefix: str) -> str:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for n, r in enumerate(rows):
            fh.write(json.dumps({"id": f"{prefix}-{n:05d}", **r}, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_build_random_test(ds7_v2_test: Path = DS7_V2_TEST, out_dir: Path = OUT_DIR) -> int:
    p = SETTINGS["randomTest"]
    rng = random.Random(p["seed"])
    if not ds7_v2_test.exists():
        raise SystemExit(f"{ds7_v2_test} is missing: it is built on Kaggle (kaggle/qwen_data); copy it from that output")
    frozen = frozen_prompts(PROJECT_ROOT / SETTINGS["ds7v2"]["frozenTest"])
    with open(ds7_v2_test, encoding="utf-8") as fh:
        pool = [json.loads(line) for line in fh if line.strip()]
    pool = [r for r in pool if r["kind"] != "glossary" and r["messages"][1]["content"] not in frozen]
    real = [{k: r[k] for k in ("source", "kind", "phrasing", "template", "messages")}
            for r in rng.sample(pool, min(p["realPrompts"], len(pool)))]
    flag_z = flag_abs_z()
    synthetic = [record(synthetic_tx(rng, flag_z), rng, "synthetic") for _ in range(p["syntheticPrompts"])]

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"real.jsonl": write(out_dir / "real.jsonl", real, "rreal"),
             "synthetic.jsonl": write(out_dir / "synthetic.jsonl", synthetic, "rsyn")}
    kinds = {name: {k: sum(r["kind"] == k for r in rows) for k in ("flagged", "not_flagged", "unscored")}
             for name, rows in (("real", real), ("synthetic", synthetic))}
    manifest = {"dataset": "Randomized Assistant test set (real held-out DS7 v2 prompts and synthetic transactions)",
                "built_utc": utc_now(), "settings": p, "flag_abs_z": flag_z, "pool": len(pool),
                "ds7_v2_test_sha256": hashlib.sha256(ds7_v2_test.read_bytes()).hexdigest(), "kinds": kinds,
                "files": [{"path": f"data/processed/random_test/{name}", "sha256": digest} for name, digest in files.items()]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("Randomized test set: %d real prompts (from %d eligible) and %d synthetic: %s", len(real), len(pool),
             len(synthetic), kinds)
    return 0
