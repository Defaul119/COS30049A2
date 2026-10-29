"""A randomized test set for the Explorer Assistant: real held-out prompts and synthetic transactions.

Serves the round-2 evaluation (docs/KAGGLE_LM_PLAN.md; the user asked on 26 Oct for a new randomized test of the
language models). Built once with settings -> randomTest -> seed and then frozen (committed), so every
writer, the 152M model in both rounds, answers exactly the same prompts:

    real.jsonl        prompts drawn at random from DS7 v2's test split. Both rounds split DS7 by the same seeded
                      hash of the address, so no writer of either round trained on these addresses; every prompt
                      of the frozen v1 test set is left out, so the two test sets do not overlap.
    synthetic.jsonl   transactions made up at random. Each feature's z is drawn (within usual, contributing or
                      flag-level, either sign, a few within rounding of 0), and its value follows from z, a usual
                      value and a spread on the feature's own scale (baseline -> zTransform), so value, usual and
                      z always agree; roles and the flag come from them as in explorer.predict. Some features have
                      no comparison and some transactions are unscored, as in real data. No writer has seen these
                      numbers, so number faithfulness and the direction check meet new combinations.
    hard.jsonl        synthetic transactions with what neither model trained on (user, 26 Oct: "try another
                      randomized dataset" after round 1's 99% pass rates): any of the nine methods, a method
                      other than the baseline flagging independently of the features (flagged with nothing
                      standing out, or not flagged despite a flag-level feature), the instruction reworded and the
                      facts shuffled in some prompts. Its own seed, so real.jsonl and synthetic.jsonl stay as built.
    balanced.jsonl    real prompts, balanced by case (user, 26 Oct: "a bit too high... could be data imbalance"):
                      every transaction of DS10's unseen addresses and of DS2's test-split addresses explained,
                      duplicates and prompts of the other test sets removed, then an equal number drawn from each
                      case (assistant.case_of), so the easy cases cannot carry the rate. Built by its own step,
                      `python -m explorer.cli build-balanced-test`, as it computes features for thousands of
                      transactions.

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

import pandas as pd

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


def synthetic_tx(rng: random.Random, flag_z: float, ranges: dict[str, Any] | None = None) -> dict[str, Any]:
    """One made-up TransactionDetail (the fields explorer.assistant and explorer.explain read); `ranges` overrides
    settings randomTest's ranges (the extreme set)."""
    p = {**SETTINGS["randomTest"], **(ranges or {})}
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


def record(tx: dict[str, Any], rng: random.Random, source: str,
           messages: list[dict[str, str]] | None = None, facts: dict[str, Any] | None = None) -> dict[str, Any]:
    """A test row like DS7's: the prompt, a reply in a random phrasing (checked by the guardrail) and the template."""
    k = rng.randrange(SETTINGS["ds7"]["phrasings"])
    reply = phrase(tx, k)
    messages = messages or assistant.prompt_messages(tx)
    checks = assistant.check(reply, messages, facts or assistant.summary(tx))
    if not checks["passed"]:
        raise ValueError(f"a synthetic reply fails its own guardrail ({checks}): {reply!r}")
    kind = "unscored" if not tx["scored"] else "flagged" if tx["flagged"] else "not_flagged"
    return {"source": source, "kind": kind, "phrasing": k, "template": template_explanation(tx),
            "messages": messages + [{"role": "assistant", "content": reply}]}


# Instructions neither model was trained with (DS7 uses assistant.INSTRUCTION only); each fits on one line, so the
# facts still start on the second, where lm.eval_explain reads them.
REWORDED = [
    "Why was this transaction flagged or not? Answer in plain words for an analyst, using only these facts:",
    "Summarise what stands out about this transaction for an analyst. Use only the facts below:",
    "Using only the facts below, tell an analyst whether this transaction is unusual for its address and why:",
    "Explain the result below to a colleague who is new to the Explorer. Stick to these facts:",
    "Write a short explanation of this scoring result, based only on these facts:",
    "In plain English, what does this result mean? Only use these facts:",
]


def hard_record(rng: random.Random, flag_z: float, shares: dict[str, float] | None = None,
                instructions: list[str] = REWORDED, ranges: dict[str, Any] | None = None) -> dict[str, Any]:
    """A synthetic transaction explained for a random method, maybe with a reworded instruction and shuffled facts;
    `shares` overrides how often each happens (the maximal-shift set has every one). `instructions` and `ranges`
    (synthetic_tx's) let explorer.weak_cases draw training examples with wordings the test sets never use; the
    defaults draw exactly as the frozen sets were built."""
    p = {**SETTINGS["randomTest"], **(shares or {})}
    tx = synthetic_tx(rng, flag_z, ranges)
    method = "baseline" if rng.random() < p["hardBaselineShare"] else rng.choice(
        [m for m in METHOD_NAMES if m != "baseline"])
    tx["method"] = {"name": METHOD_NAMES[method]}
    if method != "baseline" and tx["scored"]:
        # Its flag does not follow the features' roles: a pattern method can flag with nothing standing out,
        # or pass over a flag-level feature (explorer.predict takes the flag from the method itself).
        tx["flagged"] = rng.random() < p["hardOtherFlagShare"]
    facts = assistant.summary(tx)
    shuffled = bool(facts.get("features")) and rng.random() < p["shuffleShare"]
    if shuffled:
        rng.shuffle(facts["features"])
    reworded = rng.randrange(len(instructions)) if rng.random() < p["rewordShare"] else None
    instruction = assistant.INSTRUCTION if reworded is None else instructions[reworded]
    messages = [{"role": "system", "content": assistant.SYSTEM_PROMPT},
                {"role": "user", "content": f"{instruction}\n{json.dumps(facts, ensure_ascii=False)}"}]
    row = record(tx, rng, "hard", messages, facts)
    row["variant"] = {"method": method, "reworded": reworded, "shuffled": shuffled}
    return row


def write(path: Path, rows: list[dict[str, Any]], prefix: str) -> str:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for n, r in enumerate(rows):
            fh.write(json.dumps({"id": f"{prefix}-{n:05d}", **r}, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def every_tx(features: pd.DataFrame, transactions: pd.DataFrame) -> list[dict[str, Any]]:
    """The TransactionDetail of every transaction, as explorer.predict describes it for the selected method.

    Only the selected method decides what a prompt says (its flag; the features' roles are the baseline's), so
    only it is scored.
    """
    from .ds7 import score_frame
    from .predict import Predictor
    from .reviewer import selected_method
    from .scores import saved_methods

    name = selected_method()
    methods = {name: saved_methods()[name]}
    predictor = Predictor(features, transactions, score_frame(features, methods), methods, name, "")
    scores = predictor.scores.set_index(["address", "txid"])
    rows = predictor.features.reset_index(drop=True)
    out = []
    for _, row in rows.iterrows():
        s = scores.loc[(row["address"], row["txid"])]
        out.append(predictor.detail(row, {name: s[f"score_{name}"]}, {name: bool(s[f"flag_{name}"])}))
    return out


def run_build_balanced_test(out_dir: Path = OUT_DIR, n_jobs: int = SETTINGS["classification"]["nJobs"]) -> int:
    """balanced.jsonl: real prompts from addresses no writer trained on, an equal number per case (settings
    randomTest -> _balanced), and its entry in manifest.json."""
    from .ds2_check import features_in_parallel
    from .ds7 import DS2_PATH, ds2_transactions, split_of
    from .ds10 import OUT_DIR as DS10_DIR
    from .features import FeatureConfig

    p = SETTINGS["randomTest"]
    cfg = FeatureConfig.refined()
    ids = sorted(pd.read_parquet(DS2_PATH, columns=["addr_id"])["addr_id"].unique())
    test_ids = [a for a in ids if split_of(a, SETTINGS["ds7"]) == "test"]
    sources = {"DS2 test split": ds2_transactions(DS2_PATH, test_ids),
               "DS10": pd.read_csv(DS10_DIR / "transactions.csv")}
    excluded = frozen_prompts(PROJECT_ROOT / SETTINGS["ds7v2"]["frozenTest"])
    for path in (DS10_DIR / "explain.jsonl", out_dir / "real.jsonl"):
        excluded |= frozen_prompts(path)
    pool: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    seen: set[str] = set()
    for source, tx in sources.items():
        details = every_tx(features_in_parallel(tx, cfg, n_jobs), tx)
        log.info("%s: %d addresses, %d transactions explained", source, tx["address"].nunique(), len(details))
        for detail in details:
            prompt = assistant.prompt_messages(detail)[1]["content"]
            if prompt in seen or prompt in excluded:
                continue
            seen.add(prompt)
            pool.setdefault(assistant.case_of(assistant.summary(detail)), []).append((source, detail))

    rng = random.Random(p["balancedSeed"])
    rows = []
    for case in sorted(pool):
        for source, detail in rng.sample(pool[case], min(p["balancedPerCase"], len(pool[case]))):
            rows.append(record(detail, rng, source))
    rng.shuffle(rows)
    digest = write(out_dir / "balanced.jsonl", rows, "rbal")
    counts = {case: {"unique_in_pool": len(pool[case]), "drawn": min(p["balancedPerCase"], len(pool[case]))}
              for case in sorted(pool)}
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [f for f in manifest["files"] if not f["path"].endswith("balanced.jsonl")]
    manifest["files"].append({"path": "data/processed/random_test/balanced.jsonl", "sha256": digest})
    manifest["balanced"] = {"built_utc": utc_now(), "sources": {s: int(t["address"].nunique()) for s, t in sources.items()},
                            "cases": counts}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("Balanced test set: %d prompts; per case (unique in pool, drawn): %s", len(rows),
             {c: (v["unique_in_pool"], v["drawn"]) for c, v in counts.items()})
    return 0


def crawled_details(n_jobs: int) -> dict[str, list[dict[str, Any]]]:
    """Every transaction of the crawled addresses no writer trained on, explained: DS1's and DS2's test-split
    addresses (DS7 splits by the same address hash in both rounds) and DS10."""
    from .ds2_check import features_in_parallel
    from .ds7 import DS2_PATH, ds2_transactions, split_of
    from .ds10 import OUT_DIR as DS10_DIR
    from .features import FeatureConfig
    from .snapshot import MANIFEST_PATH

    ds1_f = pd.read_csv(MANIFEST_PATH.parent / "features.csv")
    ds1_t = pd.read_csv(MANIFEST_PATH.parent / "transactions.csv")
    ds1_test = [a for a in ds1_f["address"].unique() if split_of(a, SETTINGS["ds7"]) == "test"]
    ids = sorted(pd.read_parquet(DS2_PATH, columns=["addr_id"])["addr_id"].unique())
    ds2_t = ds2_transactions(DS2_PATH, [a for a in ids if split_of(a, SETTINGS["ds7"]) == "test"])
    ds10_t = pd.read_csv(DS10_DIR / "transactions.csv")
    cfg = FeatureConfig.refined()
    out = {"DS1 test split": every_tx(ds1_f[ds1_f["address"].isin(ds1_test)], ds1_t[ds1_t["address"].isin(ds1_test)]),
           "DS2 test split": every_tx(features_in_parallel(ds2_t, cfg, n_jobs), ds2_t),
           "DS10": every_tx(features_in_parallel(ds10_t, cfg, n_jobs), ds10_t)}
    for source, details in out.items():
        log.info("%s: %d transactions explained", source, len(details))
    return out


def close_call(facts: dict[str, Any], margin: float) -> bool:
    """The prompt's listed |z| values (the reasons, or the three largest when nothing stands out) include two that
    differ, but by less than `margin`: the order or the pair is hardest to get right from the prompt."""
    if not facts.get("scored", True):
        return False
    scored = [f for f in facts["features"] if f["z"] is not None]
    reasons = [f for f in scored if f["role"] in assistant.MENTIONED_ROLES]
    values = sorted((abs(float(f["z"])) for f in (reasons or scored)), reverse=True)[: len(reasons) or 3]
    return any(0 < a - b < margin for a, b in zip(values, values[1:]))


def run_build_more_tests(out_dir: Path = OUT_DIR, n_jobs: int = SETTINGS["classification"]["nJobs"]) -> int:
    """crawl_ds1/ds2/ds10.jsonl, close_calls.jsonl, extreme.jsonl and max_shift.jsonl (settings randomTest -> _more)."""
    from .ds10 import OUT_DIR as DS10_DIR

    p = SETTINGS["randomTest"]
    rng, flag_z = random.Random(p["moreSeed"]), flag_abs_z()
    used = frozen_prompts(PROJECT_ROOT / SETTINGS["ds7v2"]["frozenTest"])
    for path in (DS10_DIR / "explain.jsonl", out_dir / "real.jsonl", out_dir / "balanced.jsonl"):
        used |= frozen_prompts(path)
    def prompt(detail: dict[str, Any]) -> str:
        return assistant.prompt_messages(detail)[1]["content"]

    pools: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for source, details in crawled_details(n_jobs).items():
        seen: set[str] = set()
        for d in details:
            text = prompt(d)
            if text in used or text in seen:
                continue
            seen.add(text)
            pools.setdefault(source, {}).setdefault(assistant.case_of(assistant.summary(d)), []).append(d)

    sets: dict[str, list[dict[str, Any]]] = {}
    for source, name in (("DS1 test split", "crawl_ds1"), ("DS2 test split", "crawl_ds2"), ("DS10", "crawl_ds10")):
        rows = []
        for case in sorted(pools[source]):
            for d in rng.sample(pools[source][case], min(p["perSourcePerCase"], len(pools[source][case]))):
                rows.append(record(d, rng, source))
                used.add(prompt(d))
        rng.shuffle(rows)
        sets[name] = rows
    close = [(s, d) for s, cases in pools.items() for ds in cases.values() for d in ds
             if prompt(d) not in used and close_call(assistant.summary(d), p["closeMargin"])]
    sets["close_calls"] = [record(d, rng, s) for s, d in rng.sample(close, min(p["closePrompts"], len(close)))]
    sets["extreme"] = [record(synthetic_tx(rng, flag_z, rng.choice(p["extremeRegimes"])), rng, "extreme")
                       for _ in range(p["extremePrompts"])]
    shift = {"hardBaselineShare": 0.0, "rewordShare": 1.0, "shuffleShare": 1.0}
    sets["max_shift"] = [{**hard_record(rng, flag_z, shift), "source": "max_shift"} for _ in range(p["maxShiftPrompts"])]

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {}
    for name, rows in sets.items():
        digest = write(out_dir / f"{name}.jsonl", rows, f"r{name.replace('_', '')}")
        manifest["files"] = [f for f in manifest["files"] if not f["path"].endswith(f"/{name}.jsonl")]
        manifest["files"].append({"path": f"data/processed/random_test/{name}.jsonl", "sha256": digest})
        cases: dict[str, int] = {}
        for r in rows:
            c = assistant.case_of(json.loads(r["messages"][1]["content"].partition("\n")[2]))
            cases[c] = cases.get(c, 0) + 1
        summary[name] = {"prompts": len(rows), "cases": dict(sorted(cases.items()))}
        log.info("%s: %d prompts; %s", name, len(rows), summary[name]["cases"])
    manifest["more"] = {"built_utc": utc_now(), "close_candidates": len(close), "sets": summary,
                        "pools": {s: {c: len(v) for c, v in sorted(cases.items())} for s, cases in pools.items()}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


def run_build_random_test(ds7_v2_test: Path = DS7_V2_TEST, out_dir: Path = OUT_DIR) -> int:
    """Step `build-random-test`: real.jsonl (held-out DS7 v2 prompts), synthetic.jsonl and hard.jsonl, each
    with its own seed, and their manifest."""
    p = SETTINGS["randomTest"]
    rng = random.Random(p["seed"])
    if not ds7_v2_test.exists():
        raise SystemExit(f"{ds7_v2_test} is missing: build it with python -m explorer.cli build-ds7 --settings ds7v2 "
                         "--out data/processed/ds7_v2 (or copy it from the kaggle/ds7_data notebook's output)")
    frozen = frozen_prompts(PROJECT_ROOT / SETTINGS["ds7v2"]["frozenTest"])
    with open(ds7_v2_test, encoding="utf-8") as fh:
        pool = [json.loads(line) for line in fh if line.strip()]
    pool = [r for r in pool if r["kind"] != "glossary" and r["messages"][1]["content"] not in frozen]
    real = [{k: r[k] for k in ("source", "kind", "phrasing", "template", "messages")}
            for r in rng.sample(pool, min(p["realPrompts"], len(pool)))]
    flag_z = flag_abs_z()
    synthetic = [record(synthetic_tx(rng, flag_z), rng, "synthetic") for _ in range(p["syntheticPrompts"])]
    # Its own seed, so adding it left real.jsonl and synthetic.jsonl exactly as first built.
    hard_rng = random.Random(p["hardSeed"])
    hard = [hard_record(hard_rng, flag_z) for _ in range(p["hardPrompts"])]

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"real.jsonl": write(out_dir / "real.jsonl", real, "rreal"),
             "synthetic.jsonl": write(out_dir / "synthetic.jsonl", synthetic, "rsyn"),
             "hard.jsonl": write(out_dir / "hard.jsonl", hard, "rhard")}
    kinds = {name: {k: sum(r["kind"] == k for r in rows) for k in ("flagged", "not_flagged", "unscored")}
             for name, rows in (("real", real), ("synthetic", synthetic), ("hard", hard))}
    manifest = {"dataset": "Randomized Assistant test set (real held-out DS7 v2 prompts and synthetic transactions)",
                "built_utc": utc_now(), "settings": p, "flag_abs_z": flag_z, "pool": len(pool),
                "ds7_v2_test_sha256": hashlib.sha256(ds7_v2_test.read_bytes()).hexdigest(), "kinds": kinds,
                "files": [{"path": f"data/processed/random_test/{name}", "sha256": digest} for name, digest in files.items()]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("Randomized test set: %d real prompts (from %d eligible), %d synthetic and %d hard: %s", len(real),
             len(pool), len(synthetic), len(hard), kinds)
    return 0
