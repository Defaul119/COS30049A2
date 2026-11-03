"""Build the A2 report PDF and run its checks: numbers, word count, citations, wording.

Serves the brief's report (docs/brief/assignment2_description.txt, section 4) and the plan's writing rules
(docs/plan/sections/s12-report.tex): every number from reports/tables (tools/report_numbers.py), at most 3,000 body
words (tools/report_wordcount.py), every citation in the reference list and every entry cited, and the project's
wording rule (CLAUDE.md: the banned words never describe the tool or our addresses).

Writes report/main.pdf. Exit code 1 if any check fails; the PDF is still built so the problem can be seen.

Usage (repository root): python tools/report_build.py [--skip-numbers]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
SECTIONS = REPORT / "sections"
sys.path.insert(0, str(ROOT / "tools"))
import report_wordcount  # noqa: E402  (a sibling script, not a package)

BANNED = json.loads((ROOT / "config" / "settings.json").read_text(encoding="utf-8"))["bannedWords"]
# "illicit" and "ransomware" may appear only as the labelled datasets' own class names (CLAUDE.md), so the check
# accepts them only next to words that make that explicit.
CLASS_NAME_CONTEXT = re.compile(r"(illicit (class|or licit|share|behaviour)|own illicit|illicit; ransomware)")


def body_text() -> str:
    return "\n".join((SECTIONS / f"{n}.tex").read_text(encoding="utf-8") for n in report_wordcount.BODY)


def check_citations() -> list[str]:
    """Problems: a cited (author, year) with no reference entry, or an entry never cited."""
    refs = (SECTIONS / "09-references.tex").read_text(encoding="utf-8")
    entries = re.findall(r"\\item (.+?)\((\d{4}|n\.d\.)\)", refs)
    keys = {(re.split(r"[ ,]", a.strip())[0] if not a.startswith("The ") else " ".join(a.split()[:2]), y)
            for a, y in entries}
    body = re.sub(r"\\ ", " ", re.sub(r"\s+", " ", body_text()))
    problems = []
    for surname, year in sorted(keys):
        if not re.search(re.escape(surname) + r"[^()]{0,60}?" + re.escape(year), body):
            problems.append(f"reference never cited: {surname} ({year})")
    for m in re.finditer(r"\(([^()]*?\b(?:19|20)\d{2}|[^()]*?n\.d\.)\)", body):
        for part in m.group(1).split(";"):
            part = part.strip()
            ym = re.search(r"((?:19|20)\d{2}|n\.d\.)$", part)
            if not ym or not re.match(r"[A-Z]", part):
                continue
            names = re.sub(r"(et al\.?|and|,|\\|\.)", " ", part[: ym.start()]).split()
            if not any((n, ym.group(1)) in keys or any(k[0].startswith(n) and k[1] == ym.group(1) for k in keys)
                       for n in names[:1] + [" ".join(names[:2])]):
                problems.append(f"citation without an entry: ({part})")
    return problems


def check_wording() -> list[str]:
    problems = []
    for name in report_wordcount.BODY:
        text = (SECTIONS / f"{name}.tex").read_text(encoding="utf-8")
        for word in BANNED:
            for m in re.finditer(rf"\b{word}\w*", text, flags=re.I):
                around = text[max(0, m.start() - 20): m.end() + 20].replace("\n", " ")
                if word == "illicit" and CLASS_NAME_CONTEXT.search(around):
                    continue
                problems.append(f"{name}: banned word '{m.group(0)}' in '...{around}...'")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-numbers", action="store_true", help="reuse report/generated (faster)")
    args = ap.parse_args()
    failed = False
    if not args.skip_numbers:
        if subprocess.run([sys.executable, str(ROOT / "tools" / "report_numbers.py")], cwd=ROOT).returncode:
            return 1
    total = sum(report_wordcount.count_body().values())
    (REPORT / "generated" / "wordcount.tex").write_text(f"{total:,} ".replace(",", "{,}"), encoding="utf-8")
    print(f"Body words: {total}")
    if total > 3000:
        print("FAIL: over the 3,000-word limit")
        failed = True
    for problem in check_citations() + check_wording():
        print("FAIL:", problem)
        failed = True
    for _ in range(3):  # contents, lists and references settle after three passes
        run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=REPORT,
                             capture_output=True, text=True)
        if run.returncode:
            print(run.stdout[-3000:])
            print("FAIL: pdflatex stopped; see report/main.log")
            return 1
    warnings = re.findall(r"LaTeX Warning: (Reference|Citation) [^\n]*", run.stdout)
    for w in warnings:
        print("FAIL:", w)
        failed = True
    print(f"Wrote {(REPORT / 'main.pdf').relative_to(ROOT)}" + (" (with failed checks)" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
