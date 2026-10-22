"""The Explorer Assistant's prompt, its guardrail, and the measures its explanations are judged by.

Serves plan §10.2-10.3 (docs/plan/sections/s09-lm.tex, "Guardrail for A3" and the Assistant 3B):
both Assistants (the 152M model trained from scratch and the QLoRA-fine-tuned 3B model) see the
same prompt, built here from a scored transaction, and every answer passes the same validator
before the Explorer shows it:

    numbers      every number in the answer appears in the prompt (after the prompt's own formatting)
    limitation   a flagged transaction's answer contains the fixed limitation line exactly
    wording      no banned word (settings.json -> bannedWords; A1 NFR-01)

The Explorer tries its writers in order (the team's design, 23 Oct): the 152M model, trained from
scratch, explains DS1 transactions first; if its answer fails a check, the fine-tuned Qwen 3B
writes one; if that fails too, the template explanation is shown. The checks are code, not a
model, so they are exact. The template is authoritative, and the Assistants only rephrase it.
Qwen 3B also answers general questions as the Explorer's chatbot; that is outside this module. The evaluation measures (number faithfulness,
limitation line, banned words, feature coverage, ROUGE-L against the template) use the same
functions, so the reported pass rate is the guardrail's real behaviour.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from .explain import BANNED, FEATURE_NAMES, describe_value, template_explanation
from .settings import SETTINGS

SYSTEM_PROMPT = (
    "You are the Explorer Assistant. You explain, in plain words, why a Bitcoin transaction is or is not "
    "unusual for its own address, using only the facts you are given. Never guess who controls an "
    "address or why it transacts. Unusual activity is not evidence of wrongdoing."
)
INSTRUCTION = "Explain this transaction's result in plain words for an analyst, using only these facts:"
NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Roles whose features the explanation must mention (the template lists them; plan "feature coverage").
MENTIONED_ROLES = ("flag", "contributing")

Generator = Callable[[list[dict[str, str]]], str]


def summary(tx: dict[str, Any]) -> dict[str, Any]:
    """The facts the Assistant may use, from a TransactionDetail (predict.Predictor.detail).

    Values are pre-formatted exactly as the template formats them, so a faithful answer can only
    repeat numbers that are already here.
    """
    method = tx["method"]["name"]
    if not tx.get("scored", True):
        return {"scored": False, "method": method}
    features = [{
        "feature": FEATURE_NAMES[f["id"]], "value": describe_value(f["id"], f["value"]),
        "usual": describe_value(f["id"], f["usual"], usual=True),
        "z": None if f["z"] is None else f"{f['z']:.1f}", "role": f["role"],
    } for f in tx["features"]]
    out = {"scored": True, "method": method, "flagged": bool(tx["flagged"]), "direction": tx["direction"],
           "burstWindowDays": SETTINGS["features"]["burstWindowDays"], "features": features}
    if tx.get("model_reason"):
        out["modelReason"] = tx["model_reason"]
    return out


def prompt_messages(tx: dict[str, Any]) -> list[dict[str, str]]:
    """System and user messages for one transaction; the reply is what the Assistant writes."""
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{INSTRUCTION}\n{json.dumps(summary(tx), ensure_ascii=False)}"}]


def numbers_in(text: str) -> set[str]:
    return set(NUMBER.findall(text))


def numbers_faithful(answer: str, prompt: str) -> bool:
    """Every number in the answer also appears in the prompt."""
    return numbers_in(answer) <= numbers_in(prompt)


def has_limitation(answer: str) -> bool:
    return SETTINGS["limitationLine"] in answer


def banned_words(answer: str) -> list[str]:
    return [m.group(0) for m in BANNED.finditer(answer)]


def coverage(answer: str, facts: dict[str, Any]) -> float | None:
    """Share of flag-level and contributing features whose value the answer repeats (None if there are none)."""
    wanted = [f for f in facts.get("features", []) if f["role"] in MENTIONED_ROLES and f["value"] != "-"]
    if not wanted:
        return None
    return sum(f["value"] in answer for f in wanted) / len(wanted)


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F-measure on words: the longest common subsequence, with no extra dependency."""
    a, b = candidate.split(), reference.split()
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(a), lcs / len(b)
    return 2 * precision * recall / (precision + recall)


def check(answer: str, messages: list[dict[str, str]], facts: dict[str, Any]) -> dict[str, Any]:
    """The guardrail's three checks on one answer, and whether it passes all of them."""
    prompt = "\n".join(m["content"] for m in messages)
    result = {"numbers_ok": numbers_faithful(answer, prompt),
              "limitation_ok": has_limitation(answer) or not facts.get("flagged", False),
              "wording_ok": not banned_words(answer), "nonempty": bool(answer.strip())}
    result["passed"] = all(result.values())
    return result


def explain(tx: dict[str, Any], writers: list[tuple[str, Generator]] | None = None) -> dict[str, Any]:
    """The explanation the Explorer shows: the first writer's answer that passes every check, else the template.

    `writers` is an ordered list of (name, generate), e.g. [("text150m", ...), ("qwen3b", ...)];
    each generate maps chat messages to the model's reply. None or [] (the default, plan risk 15's
    `use_assistant=False`) always returns the template. `attempts` records every answer and its checks.
    """
    messages, facts = prompt_messages(tx), summary(tx)
    attempts = []
    for name, generate in writers or []:
        answer = generate(messages).strip()
        checks = check(answer, messages, facts)
        attempts.append({"source": name, "answer": answer, "checks": checks})
        if checks["passed"]:
            return {"text": answer, "source": name, "attempts": attempts}
    return {"text": template_explanation(tx), "source": "template", "attempts": attempts}
