"""Evaluate an Explorer Assistant on the DS7 test set, with the measures of plan §10.2.

Serves plan §9 (the Assistant's evaluation table) and the round-1-versus-round-2 comparison: every
writer answers the same test prompts with greedy decoding, and every answer is measured by the same
functions the guardrail uses (explorer.assistant), so the reported pass rate is the guardrail's
real behaviour:

    numbers_ok      every number in the answer appears in the prompt (number faithfulness)
    limitation_ok   a flagged item's answer has the limitation line exactly
    wording_ok      no banned word
    direction_ok    no comparison phrase contradicts its feature's z sign (wrong_direction lists them)
    verdict_ok      the answer states the right result: flagged, not flagged or not scored
    passed          all five (the guardrail would show the answer, not the template)
    reasons_ok      the answer names exactly the template's features, no fewer and no more
    order_ok        ... in the template's order (strongest first)
    binding_ok      every number on a reason's line belongs to that line's feature
    correct         passed, reasons_ok, order_ok and binding_ok: a fully correct explanation
    case            the kind of decision: unscored; flagged or not; how many reasons, or nothing standing
                    out (then the two largest differences, which the model must pick by |z| itself)
    coverage        share of flag-level and contributing features whose value the answer repeats
    rouge_l         ROUGE-L against the template answer
    rouge_own       ROUGE-L against the row's own reply (one of DS7's five phrasings), which does not
                    penalise a correct paraphrase the way rouge_l does
    copy            the answer is the template, word for word
    key_facts       glossary rows with a key_facts list: the share of key facts the answer states

`--model template` scores the template itself: the control row, which should pass everything.
`--rescore` measures the answers already saved in assistant_eval_<model>.csv again, without a model,
so answers written before a measure existed are judged by it too (round 2, docs/KAGGLE_LM_PLAN.md §5).

Usage:
    python -m lm.eval_explain --model template
    python -m lm.eval_explain --model text150m --checkpoint runs/text-150m-explain/model_best.pt
    python -m lm.eval_explain --model template --test data/processed/ds10/explain.jsonl --tag ds10   # another set
    python -m lm.eval_explain --model text150m --rescore          # saved answers, current measures
    python -m lm.eval_explain --model text150m_v2 --checkpoint ...  # a round-2 writer gets its own rows

Writes reports/tables/assistant_eval_<model>.csv (one row per test prompt, with the answer), adds or
replaces that model's row in reports/tables/assistant_eval.csv, and its rows in assistant_eval_by_case.csv
(fully correct per case). With --tag T the names end in _T, so a second test set (DS10, explorer/ds10.py)
never overwrites the DS7 tables.

The headline rate weights every prompt equally, so the mix of a test set moves it: unscored prompts are
one identical sentence and easy, and in the cases with reasons the prompt labels each feature's role. The
summary therefore also gives the mean over cases (macro), the rate where nothing stands out (the one case
in which the writer must rank features itself), and the rate over unique prompts (duplicates counted once).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import pandas as pd

from explorer import assistant

from .writers import text150m_generator

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
# Writers by kind; a round-2 model is named <kind>_v2 so its rows sit beside round 1's.
KINDS = ("template", "text150m")


def facts_of(messages: list[dict[str, str]]) -> dict:
    """The JSON facts after the user message's first line, whatever the instruction's wording (the hard
    randomized set rewords it); {} for glossary questions, which have none."""
    _, _, rest = messages[1]["content"].partition("\n")
    return json.loads(rest) if rest.lstrip().startswith("{") else {}


def prompt_key(row: dict) -> str:
    """A short fingerprint of the prompt, so identical prompts (e.g. every unscored one) are counted once."""
    return hashlib.sha256(row["messages"][1]["content"].encode()).hexdigest()[:16]


def measure(answer: str, row: dict) -> dict:
    """Every measure of one answer to one test row (its prompt, template, own reply and key facts)."""
    prompt, template = row["messages"][:-1], row["template"]
    facts = facts_of(prompt)
    out = assistant.check(answer, prompt, facts)
    out["wrong_direction"] = ", ".join(assistant.wrong_directions(answer, facts))
    out["coverage"] = assistant.coverage(answer, facts)
    out["rouge_l"] = assistant.rouge_l(answer, template)
    out["rouge_own"] = assistant.rouge_l(answer, row["messages"][-1]["content"])
    out["reasons_ok"] = assistant.reasons_ok(answer, template, facts)
    out["order_ok"] = assistant.order_ok(answer, template, facts)
    out["binding_ok"] = assistant.binding_ok(answer, facts)
    out["correct"] = out["passed"] and out["reasons_ok"] and out["order_ok"] and out["binding_ok"]
    out["case"] = assistant.case_of(facts) if row["kind"] != "glossary" else "glossary"
    out["copy"] = answer.strip() == template.strip()
    out["key_facts"] = assistant.key_facts_found(answer, row.get("key_facts", []))
    out["banned"] = ", ".join(assistant.banned_words(answer))
    return out


def summarise(frame: pd.DataFrame, model: str) -> dict:
    """One model's row of assistant_eval.csv.

    Explanation measures apply to explanation prompts only: a glossary answer may rightly contain numbers
    that are not in its question (e.g. the MAD's 1.4826), so glossary rows get wording, ROUGE-L and key facts.
    """
    expl, gloss = frame[frame["kind"] != "glossary"], frame[frame["kind"] == "glossary"]
    flagged = expl[expl["kind"] == "flagged"]
    facts = gloss["key_facts"].dropna() if len(gloss) else pd.Series(dtype=float)
    nan = float("nan")
    hard = expl[expl["case"].str.endswith("nothing stands out")]
    unique = expl.drop_duplicates("prompt")
    return {"model": model, "prompts": len(expl), "number_faithfulness": expl["numbers_ok"].mean(),
            "limitation_line": flagged["limitation_ok"].mean(),
            "banned_word_share": 1 - expl["wording_ok"].mean(), "guardrail_pass": expl["passed"].mean(),
            "feature_coverage": expl["coverage"].dropna().mean(), "rouge_l": expl["rouge_l"].mean(),
            "direction_errors": int((~expl["direction_ok"]).sum()), "rouge_own": expl["rouge_own"].mean(),
            "copy_rate_flagged": flagged["copy"].mean(), "verdict_accuracy": expl["verdict_ok"].mean(),
            "reasons_accuracy": expl["reasons_ok"].mean(), "order_accuracy": expl["order_ok"].mean(),
            "binding_accuracy": expl["binding_ok"].mean(), "fully_correct": expl["correct"].mean(),
            "fully_correct_macro": expl.groupby("case")["correct"].mean().mean(),
            "fully_correct_nothing_stands_out": hard["correct"].mean() if len(hard) else nan,
            "nothing_stands_out_prompts": len(hard), "unique_prompts": len(unique),
            "fully_correct_unique": unique["correct"].mean(),
            "glossary_prompts": len(gloss),
            "glossary_banned_word_share": 1 - gloss["wording_ok"].mean() if len(gloss) else nan,
            "glossary_rouge_l": gloss["rouge_l"].mean() if len(gloss) else nan,
            "glossary_key_facts": facts.mean() if len(facts) else nan,
            "glossary_correct": (facts == 1.0).mean() if len(facts) else nan}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help=f"writer name: one of {', '.join(KINDS)}, or <kind>_<round> (e.g. text150m_v2)")
    p.add_argument("--checkpoint", help="text150m: the explanation stage's model_best.pt")
    p.add_argument("--test", default=str(ROOT / "data" / "processed" / "ds7" / "test.jsonl"))
    p.add_argument("--n", type=int, default=500, help="test prompts, drawn with --seed (plan: about 500)")
    p.add_argument("--seed", type=int, default=30049)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--tag", default="", help="suffix for the output tables (e.g. ds10)")
    p.add_argument("--rescore", action="store_true",
                   help="measure the answers saved in assistant_eval_<model>[_<tag>].csv again; no model is loaded")
    args = p.parse_args()
    kind = args.model.split("_")[0]
    if kind not in KINDS:
        p.error(f"--model must start with one of {KINDS}")
    suffix = f"_{args.tag}" if args.tag else ""
    answers_path = TABLES / f"assistant_eval_{args.model}{suffix}.csv"
    path = TABLES / f"assistant_eval{suffix}.csv"
    previous = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["model"])
    previous = previous.set_index("model")

    rows = [json.loads(line) for line in open(args.test, encoding="utf-8") if line.strip()]
    rows = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    if args.rescore:
        saved = pd.read_csv(answers_path, keep_default_na=False).set_index("id")["answer"]
        if set(saved.index) != {r["id"] for r in rows}:
            raise SystemExit(f"{answers_path} holds other prompts than --test/--n/--seed draw; rescore with the same ones")
    elif kind == "text150m":
        one = text150m_generator(args.checkpoint, args.max_new_tokens)

    def answer(r: dict) -> str:
        if args.rescore:
            return saved[r["id"]]
        if kind == "template":
            return r["template"]
        return one(r["messages"][:-1])

    results, started = [], time.time()
    for r in rows:
        a = answer(r)
        results.append({"id": r["id"], "kind": r["kind"], "prompt": prompt_key(r), "answer": a, **measure(a, r)})
        if len(results) % 50 == 0:
            print(f"[eval] {len(results)}/{len(rows)} ({time.time() - started:.0f} s)", flush=True)

    frame = pd.DataFrame(results)
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(answers_path, index=False)
    summary = summarise(frame, args.model)
    if args.rescore and args.model in previous.index:
        # Generation time and the model's path belong to the run that wrote the answers.
        summary["seconds"], summary["source"] = previous.loc[args.model, ["seconds", "source"]]
    else:
        summary["seconds"] = round(time.time() - started, 1)
        summary["source"] = args.checkpoint or "explorer.explain.template_explanation"
    others = previous.drop(index=args.model, errors="ignore").reset_index()
    table = pd.concat([others, pd.DataFrame([summary])], ignore_index=True)
    table.to_csv(path, index=False)
    cases = (frame[frame["kind"] != "glossary"].groupby("case")["correct"].agg(prompts="size", fully_correct="mean")
             .reset_index().assign(model=args.model))
    case_path = TABLES / f"assistant_eval_by_case{suffix}.csv"
    if case_path.exists():
        cases = pd.concat([pd.read_csv(case_path).query("model != @args.model"), cases], ignore_index=True)
    cases[["model", "case", "prompts", "fully_correct"]].to_csv(case_path, index=False)
    print(f"[eval] {summary}")


if __name__ == "__main__":
    main()
