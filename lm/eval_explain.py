"""Evaluate an Explorer Assistant on the DS7 test set, with the measures of plan §10.2.

Serves plan §9 (the Assistant's evaluation table) and the 152M-versus-3B comparison: every model
answers the same test prompts with greedy decoding, and every answer is measured by the same
functions the guardrail uses (explorer.assistant), so the reported pass rate is the guardrail's
real behaviour:

    numbers_ok      every number in the answer appears in the prompt (number faithfulness)
    limitation_ok   a flagged item's answer has the limitation line exactly
    wording_ok      no banned word
    direction_ok    no comparison phrase contradicts its feature's z sign (wrong_direction lists them)
    verdict_ok      the answer states the right result: flagged, not flagged or not scored
    passed          all five (the guardrail would show the answer, not the template)
    reasons_ok      the answer names exactly the template's features, no fewer and no more
    correct         passed and reasons_ok: a fully correct explanation
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
    python -m lm.eval_explain --model qwen3b --config configs/qwen3b_explain.json --adapter runs/qwen3b-explain/adapter_best
    python -m lm.eval_explain --model qwen3b ... --batch-size 16      # Qwen answers 16 prompts per pass
    python -m lm.eval_explain --model template --test data/processed/ds10/explain.jsonl --tag ds10   # another set
    python -m lm.eval_explain --model qwen3b --rescore            # saved answers, current measures
    python -m lm.eval_explain --model qwen3b_v2 ... --adapter ...   # a round-2 writer gets its own rows

Writes reports/tables/assistant_eval_<model>.csv (one row per test prompt, with the answer) and
adds or replaces that model's row in reports/tables/assistant_eval.csv. With --tag T both names end
in _T, so a second test set (DS10, explorer/ds10.py) never overwrites the DS7 tables.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd

from explorer import assistant

from .writers import qwen_batch_generator, text150m_generator

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
# Writers by kind; a round-2 model is named <kind>_v2 so its rows sit beside round 1's.
KINDS = ("template", "text150m", "qwen3b")


def facts_of(messages: list[dict[str, str]]) -> dict:
    """The JSON facts after the user message's first line, whatever the instruction's wording (the hard
    randomized set rewords it); {} for glossary questions, which have none."""
    _, _, rest = messages[1]["content"].partition("\n")
    return json.loads(rest) if rest.lstrip().startswith("{") else {}


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
    out["correct"] = out["passed"] and out["reasons_ok"]
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
    return {"model": model, "prompts": len(expl), "number_faithfulness": expl["numbers_ok"].mean(),
            "limitation_line": flagged["limitation_ok"].mean(),
            "banned_word_share": 1 - expl["wording_ok"].mean(), "guardrail_pass": expl["passed"].mean(),
            "feature_coverage": expl["coverage"].dropna().mean(), "rouge_l": expl["rouge_l"].mean(),
            "direction_errors": int((~expl["direction_ok"]).sum()), "rouge_own": expl["rouge_own"].mean(),
            "copy_rate_flagged": flagged["copy"].mean(), "verdict_accuracy": expl["verdict_ok"].mean(),
            "reasons_accuracy": expl["reasons_ok"].mean(), "fully_correct": expl["correct"].mean(),
            "glossary_prompts": len(gloss),
            "glossary_banned_word_share": 1 - gloss["wording_ok"].mean() if len(gloss) else nan,
            "glossary_rouge_l": gloss["rouge_l"].mean() if len(gloss) else nan,
            "glossary_key_facts": facts.mean() if len(facts) else nan,
            "glossary_correct": (facts == 1.0).mean() if len(facts) else nan}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help=f"writer name: one of {', '.join(KINDS)}, or <kind>_<round> (e.g. qwen3b_v2)")
    p.add_argument("--checkpoint", help="text150m: the explanation stage's model_best.pt")
    p.add_argument("--config", help="qwen3b: the run's config (base model and revision)")
    p.add_argument("--adapter", help="qwen3b: the adapter folder")
    p.add_argument("--test", default=str(ROOT / "data" / "processed" / "ds7" / "test.jsonl"))
    p.add_argument("--n", type=int, default=500, help="test prompts, drawn with --seed (plan: about 500)")
    p.add_argument("--seed", type=int, default=30049)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1,
                   help="qwen3b: prompts answered together (greedy either way; round 1 used 1)")
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
    elif kind == "qwen3b":
        batch = qwen_batch_generator(args.config, args.adapter, args.max_new_tokens)

    def answer(chunk: list[dict]) -> list[str]:
        if args.rescore:
            return [saved[r["id"]] for r in chunk]
        if kind == "template":
            return [r["template"] for r in chunk]
        prompts = [r["messages"][:-1] for r in chunk]
        return [one(m) for m in prompts] if kind == "text150m" else batch(prompts)

    results, started, size = [], time.time(), max(1, args.batch_size)
    for start in range(0, len(rows), size):
        chunk = rows[start:start + size]
        answers = answer(chunk)
        results += [{"id": r["id"], "kind": r["kind"], "answer": a, **measure(a, r)} for r, a in zip(chunk, answers)]
        if len(results) // 50 > (len(results) - len(chunk)) // 50:
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
        summary["source"] = args.checkpoint or args.adapter or "explorer.explain.template_explanation"
    others = previous.drop(index=args.model, errors="ignore").reset_index()
    table = pd.concat([others, pd.DataFrame([summary])], ignore_index=True)
    table.to_csv(path, index=False)
    print(f"[eval] {summary}")


if __name__ == "__main__":
    main()
