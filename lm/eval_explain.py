"""Evaluate an Explorer Assistant on the DS7 test set, with the measures of plan §10.2.

Serves plan §9 (the Assistant's evaluation table) and the 152M-versus-3B comparison: every model
answers the same test prompts with greedy decoding, and every answer is measured by the same
functions the guardrail uses (explorer.assistant), so the reported pass rate is the guardrail's
real behaviour:

    numbers_ok      every number in the answer appears in the prompt (number faithfulness)
    limitation_ok   a flagged item's answer has the limitation line exactly
    wording_ok      no banned word
    passed          all three (the guardrail would show the answer, not the template)
    coverage        share of flag-level and contributing features whose value the answer repeats
    rouge_l         ROUGE-L against the template answer

`--model template` scores the template itself: the control row, which should pass everything.

Usage:
    python -m lm.eval_explain --model template
    python -m lm.eval_explain --model text150m --checkpoint runs/text-150m-explain/model_best.pt
    python -m lm.eval_explain --model qwen3b --config configs/qwen3b_explain.json --adapter runs/qwen3b-explain/adapter_best

Writes reports/tables/assistant_eval_<model>.csv (one row per test prompt, with the answer) and
adds or replaces that model's row in reports/tables/assistant_eval.csv.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from explorer import assistant

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
Generate = Callable[[list[dict[str, str]]], str]


def facts_of(messages: list[dict[str, str]]) -> dict:
    """The JSON facts in a DS7 prompt (the user message's second line); {} for glossary questions."""
    user = messages[1]["content"]
    if not user.startswith(assistant.INSTRUCTION):
        return {}
    return json.loads(user[len(assistant.INSTRUCTION):].strip())


def measure(answer: str, prompt: list[dict[str, str]], template: str) -> dict:
    facts = facts_of(prompt)
    out = assistant.check(answer, prompt, facts)
    out["coverage"] = assistant.coverage(answer, facts)
    out["rouge_l"] = assistant.rouge_l(answer, template)
    out["banned"] = ", ".join(assistant.banned_words(answer))
    return out


def text150m_generator(checkpoint: str, max_new_tokens: int) -> Generate:
    from .chat import reply
    from .generate import load_model
    from .tokenizer import load_tokenizer

    model = load_model(checkpoint)
    tok = load_tokenizer(str(Path(checkpoint).with_name("tokenizer.json")))
    return lambda messages: reply(model, tok, messages, max_new_tokens=max_new_tokens, temperature=0)


def qwen_generator(config: str, adapter: str, max_new_tokens: int) -> Generate:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from .qlora import load_config, load_tokenizer

    cfg = load_config(config)
    tok = load_tokenizer(cfg)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], revision=cfg["revision"], quantization_config=quant,
                                                torch_dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, adapter).eval()

    @torch.no_grad()
    def generate(messages: list[dict[str, str]]) -> str:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    return generate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["template", "text150m", "qwen3b"], required=True)
    p.add_argument("--checkpoint", help="text150m: the explanation stage's model_best.pt")
    p.add_argument("--config", help="qwen3b: the run's config (base model and revision)")
    p.add_argument("--adapter", help="qwen3b: the adapter folder")
    p.add_argument("--test", default=str(ROOT / "data" / "processed" / "ds7" / "test.jsonl"))
    p.add_argument("--n", type=int, default=500, help="test prompts, drawn with --seed (plan: about 500)")
    p.add_argument("--seed", type=int, default=30049)
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.test, encoding="utf-8") if line.strip()]
    rows = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    if args.model == "template":
        generate = None
    elif args.model == "text150m":
        generate = text150m_generator(args.checkpoint, args.max_new_tokens)
    else:
        generate = qwen_generator(args.config, args.adapter, args.max_new_tokens)

    results, started = [], time.time()
    for i, row in enumerate(rows):
        prompt = row["messages"][:-1]
        answer = row["template"] if generate is None else generate(prompt)
        results.append({"id": row["id"], "kind": row["kind"], "answer": answer, **measure(answer, prompt, row["template"])})
        if (i + 1) % 50 == 0:
            print(f"[eval] {i + 1}/{len(rows)} ({time.time() - started:.0f} s)", flush=True)

    frame = pd.DataFrame(results)
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / f"assistant_eval_{args.model}.csv", index=False)
    summary = {"model": args.model, "prompts": len(frame), "number_faithfulness": frame["numbers_ok"].mean(),
               "limitation_line": frame.loc[frame["kind"] == "flagged", "limitation_ok"].mean(),
               "banned_word_share": 1 - frame["wording_ok"].mean(), "guardrail_pass": frame["passed"].mean(),
               "feature_coverage": frame["coverage"].dropna().mean(), "rouge_l": frame["rouge_l"].mean(),
               "seconds": round(time.time() - started, 1),
               "source": args.checkpoint or args.adapter or "explorer.explain.template_explanation"}
    path = TABLES / "assistant_eval.csv"
    others = pd.read_csv(path).query("model != @args.model") if path.exists() else pd.DataFrame()
    table = pd.concat([others, pd.DataFrame([summary])], ignore_index=True)
    table.to_csv(path, index=False)
    print(f"[eval] {summary}")


if __name__ == "__main__":
    main()
