"""The rule that decides whether the round-2 152M writer replaces round 1 (docs/KAGGLE_LM_PLAN.md section 3).

Fixed on 26 Oct 2026, before any round-2 result. On the frozen tests (DS7 v1 test, DS10) and the glossary test, the
round-2 writer replaces round 1 only if all of these hold (settings.json -> assistant -> adoption):

1. on each frozen test, its guardrail pass rate and number faithfulness are no lower than round 1's;
2. on each frozen test, it writes no more wrong comparison words (direction errors) than round 1;
3. on the glossary test, its open-question key-fact accuracy is at least `keyFactGain` higher than round 1's.

Reads lm.eval_explain's summary tables (reports/tables/assistant_eval{_<tag>}.csv), writes one row per criterion
and test to reports/tables/adoption.csv, and prints the decision. It never changes settings: the team records the
decision (settings assistant.text150mCheckpoint), as with the selection rule for the scoring methods.

Usage:  python -m lm.adopt                       # text150m (round 1) against text150m_v2 (round 2)
        python -m lm.adopt --old text150m --new text150m_v2
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
RULE = json.loads((ROOT / "config" / "settings.json").read_text(encoding="utf-8"))["assistant"]["adoption"]


def summary_row(tag: str, model: str) -> pd.Series:
    """One writer's row of assistant_eval{_tag}.csv; a missing table or row stops the decision."""
    path = TABLES / f"assistant_eval{'_' + tag if tag else ''}.csv"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run lm.eval_explain for both writers on that test first")
    table = pd.read_csv(path).set_index("model")
    if model not in table.index:
        raise SystemExit(f"{path} has no row for {model}")
    return table.loc[model]


def criteria(old: str, new: str) -> pd.DataFrame:
    """Every comparison the rule makes, with whether it passed."""
    rows = []
    for test, tag in RULE["frozenTests"].items():
        a, b = summary_row(tag, old), summary_row(tag, new)
        for measure in ("guardrail_pass", "number_faithfulness"):
            rows.append({"test": test, "measure": measure, "old": a[measure], "new": b[measure],
                         "rule": "no lower", "passed": bool(b[measure] >= a[measure])})
        rows.append({"test": test, "measure": "direction_errors", "old": a["direction_errors"],
                     "new": b["direction_errors"], "rule": "no more",
                     "passed": bool(b["direction_errors"] <= a["direction_errors"])})
    a, b = summary_row(RULE["glossaryTag"], old), summary_row(RULE["glossaryTag"], new)
    gain = b["glossary_key_facts"] - a["glossary_key_facts"]
    rows.append({"test": "glossary test", "measure": "glossary_key_facts", "old": a["glossary_key_facts"],
                 "new": b["glossary_key_facts"], "rule": f"at least {RULE['keyFactGain']:+.2f}",
                 # a small tolerance: 0.30 - 0.20 is 0.0999... in floating point
                 "passed": bool(not math.isnan(gain) and gain >= RULE["keyFactGain"] - 1e-9)})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old", default="text150m", help="the writer in use (round 1)")
    p.add_argument("--new", default="text150m_v2", help="the candidate (round 2)")
    args = p.parse_args()
    table = criteria(args.old, args.new).assign(old_writer=args.old, new_writer=args.new)
    adopt = bool(table["passed"].all())
    table["decision"] = "adopt" if adopt else "keep"
    table.to_csv(TABLES / "adoption.csv", index=False)
    print(table[["test", "measure", "old", "new", "rule", "passed"]].to_string(index=False))
    print(f"Decision: {'ADOPT ' + args.new if adopt else 'KEEP ' + args.old} (docs/KAGGLE_LM_PLAN.md section 3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
