"""Plain-language reasons for a transaction's scores: the template the Explorer shows (FR-04, AC-03).

Serves plan §12.1 (predict.py's `explain`) and the A1 wording rule (NFR-01): describe what is
unusual, never what it might mean about a person. A port of the prototype's explain.ts: the
features at or above the contributing level, strongest first; if a method flags a transaction
where no single feature reaches that level, the two largest deviations, so a flagged transaction
always has an explanation. Every flagged item ends with the fixed limitation line, and
`check_wording` rejects any user-facing text containing a banned word.

This template is authoritative: the Explorer Assistant may only rephrase it (plan §10.2).
"""
from __future__ import annotations

import re
from typing import Any

from .settings import SETTINGS

SATS_PER_BTC = 100_000_000
HOURS_PER_DAY = 24
FEATURE_NAMES = {"F1": "Size", "F2": "Burst", "F3": "Gap", "F4": "Relative size"}
FEATURE_DESCRIPTIONS = {
    "F1": "How much the address received or sent in this transaction (BTC).",
    "F2": "How many earlier transactions the address made in the previous 7 days.",
    "F3": "Hours since the address's previous transaction.",
    "F4": "This amount divided by the address's usual amount in the same direction.",
}
METHOD_NAMES = {
    "baseline": "Robust z baseline (A1)",
    "isolation_forest": "Isolation Forest",
    "lof": "Local Outlier Factor",
    "kmeans": "K-Means distance",
    "dbscan": "DBSCAN noise",
    "regression": "Regression residual",
    "regression_interval": "Regression expected range",
    "fusion": "Score fusion (two methods, one flag budget)",
    "txgpt_method": "TxGPT surprisal",
}


def format_btc(sats: float) -> str:
    """Up to 8 decimals with trailing zeros removed: 0.05 BTC, 1.23456789 BTC."""
    text = f"{abs(sats) / SATS_PER_BTC:.8f}".rstrip("0").rstrip(".")
    return f"{text} BTC"


def format_hours(hours: float) -> str:
    if hours < 1:
        return "under an hour"
    if hours < 2 * HOURS_PER_DAY:
        return f"{hours:.0f} hours" if round(hours) != 1 else "1 hour"
    return f"{hours / HOURS_PER_DAY:.0f} days"


def format_ratio(ratio: float) -> str:
    return f"{ratio:.1f}x" if ratio >= 0.1 else f"{ratio:.3f}x"


def plural(n: float, word: str) -> str:
    return f"{n:g} {word}{'' if n == 1 else 's'}"


def describe_value(fid: str, value: float | None, usual: bool = False) -> str:
    if value is None:
        return "-"
    if fid == "F1":
        return format_btc(value * SATS_PER_BTC)
    if fid == "F2":
        return f"{value:.1f} transactions" if usual else plural(value, "transaction")
    if fid == "F3":
        return format_hours(value)
    return format_ratio(value)


def headline(fid: str, z: float) -> str:
    up = z > 0
    return {
        "F1": "Larger amount than usual" if up else "Smaller amount than usual",
        "F2": "Busier week than usual" if up else "Quieter week than usual",
        "F3": "Longer gap than usual" if up else "Shorter gap than usual",
        "F4": "Large compared with the usual amount" if up else "Small compared with the usual amount",
    }[fid]


def detail(feature: dict[str, Any]) -> str:
    fid = feature["id"]
    value = describe_value(fid, feature["value"])
    usual = describe_value(fid, feature["usual"], usual=True)
    return {
        "F1": f"{value} moved, against a usual {usual}.",
        "F2": f"{value} in the previous 7 days, against a usual {usual}.",
        "F3": f"{value} since the previous transaction, against a usual {usual}.",
        "F4": f"{value} the usual amount for this direction.",
    }[fid]


def reasons_for(features: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """(reasons, fallback): contributing features strongest first, else the two largest deviations."""
    scored = [f for f in features if f["z"] is not None]
    strong = [f for f in scored if f["role"] in ("flag", "contributing")]
    fallback = not strong
    chosen = sorted(scored if fallback else strong, key=lambda f: -abs(f["z"]))[: 2 if fallback else 4]
    return [{"id": f["id"], "headline": headline(f["id"], f["z"]), "detail": detail(f), "z": f["z"],
             "level": "largest" if fallback else f["role"]} for f in chosen], fallback


def template_explanation(tx: dict[str, Any]) -> str:
    """The Explorer's explanation of one scored transaction (a TransactionDetail dictionary)."""
    method = tx["method"]["name"]
    if not tx.get("scored", True):
        return "This transaction is not scored: the address does not yet have enough earlier history to compare it with."
    reasons, fallback = reasons_for(tx["features"])
    lines = [f"Flagged by {method}." if tx["flagged"] else f"Not flagged by {method}."]
    if tx.get("model_reason"):
        lines.append(f"The model's reason: {tx['model_reason']}.")
    if reasons:
        lines.append("Largest differences from this address's usual activity:" if fallback else
                     "What stands out compared with this address's own history:")
        lines += [f"- {r['headline']}: {r['detail']}" for r in reasons]
    if tx["flagged"]:
        lines.append(SETTINGS["limitationLine"])
    text = "\n".join(lines)
    check_wording(text)
    return text


BANNED = re.compile(r"\b(" + "|".join(map(re.escape, SETTINGS["bannedWords"])) + r")", re.IGNORECASE)


def check_wording(text: str) -> None:
    """Raise ValueError if user-facing text uses a banned word (A1 NFR-01)."""
    match = BANNED.search(text)
    if match:
        raise ValueError(f"banned word {match.group(0)!r} in user-facing text")
