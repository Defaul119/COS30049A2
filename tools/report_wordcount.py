"""Count the A2 report's body words, the way the brief limits them (at most 3,000).

Serves the brief's limit (docs/brief/assignment2_description.txt, section 4: "up to 3000 words, excluding cover
page, table of contents, bibliography, and appendix") and the plan's budget (docs/plan/sections/s12-report.tex:
2,850 words, so a different counting rule cannot push the report over). texcount is not installed with TinyTeX,
so this strips LaTeX itself.

Counted: the prose of the body sections (report/sections/01-08). Not counted: comments, figure and table floats
(including captions), generated tables, and LaTeX commands. A \\V{key} number counts as one word, as it would in
the PDF.

Usage (repository root): python tools/report_wordcount.py [--per-section]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ROOT / "report" / "sections"
BODY = ["01-introduction", "02-problem", "03-collection", "04-processing", "05-analysis", "06-selection",
        "07-implementation", "08-conclusion"]
# Words budgeted per section by the plan (s12-report.tex), for the per-section view.
BUDGET = {"01-introduction": 170, "02-problem": 210, "03-collection": 410, "04-processing": 400,
          "05-analysis": 370, "06-selection": 400, "07-implementation": 730, "08-conclusion": 160}


def strip(tex: str) -> str:
    """The words a reader sees in the body text, without floats, comments or markup."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)                                    # comments
    tex = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", " ", tex, flags=re.S)  # floats and captions
    tex = re.sub(r"\\input\{[^}]*\}", " ", tex)                             # generated tables
    tex = re.sub(r"\\V\{[^}]*\}", " N ", tex)                               # a number is one word
    tex = re.sub(r"\\(label|ref|cite|includegraphics|begin|end)(\[[^]]*\])?\{[^}]*\}", " ", tex)
    tex = re.sub(r"\$[^$]*\$", " x ", tex)                                  # inline maths: one word
    tex = re.sub(r"\\[a-zA-Z]+\*?", " ", tex)                               # remaining commands, keep arguments
    tex = re.sub(r"[{}\[\]~\\]", " ", tex)
    return tex


def words(tex: str) -> int:
    return len([w for w in strip(tex).split() if re.search(r"[A-Za-z0-9]", w)])


def count_body() -> dict[str, int]:
    return {name: words((SECTIONS / f"{name}.tex").read_text(encoding="utf-8")) for name in BODY}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-section", action="store_true", help="show each section against its budget")
    args = ap.parse_args()
    counts = count_body()
    if args.per_section:
        for name, n in counts.items():
            print(f"{name:20s} {n:5d}  (budget {BUDGET[name]})")
    total = sum(counts.values())
    print(f"Body words: {total} (limit 3,000; plan 2,850)")
    return 0 if total <= 3000 else 1


if __name__ == "__main__":
    raise SystemExit(main())
