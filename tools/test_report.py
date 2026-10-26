"""Write reports/assistant_tests.pdf: every Explorer Assistant test set, how it was built, and every writer's results.

Serves the round-2 evaluation (docs/KAGGLE_LM_PLAN.md; the user asked on 27 Oct for "a pdf of their logs, including
the testings of each dataset made from the crawled data"). For each test set it gives the source data, how the set
was drawn, its make-up by case (explorer.assistant.case_of), the template control, every writer scored on it so far
(reports/tables/assistant_eval*_<tag>.csv) and the excerpt of the build log that made it. Numbers are read from the
tables and files at run time, so rerun this after new results arrive.

Usage:  python tools/test_report.py        (needs pdflatex: MiKTeX on the PC; writes reports/assistant_tests.pdf)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explorer import assistant  # noqa: E402
from lm.eval_explain import facts_of  # noqa: E402

TABLES, LOGS = ROOT / "reports" / "tables", ROOT / "reports" / "logs"
RANDOM = ROOT / "data" / "processed" / "random_test"
OUT = ROOT / "reports" / "assistant_tests.pdf"
WRITERS = {"template": "Template (control)", "text150m": "152M, round 1", "qwen3b": "Qwen 3B, round 1",
           "text150m_v2": "152M, round 2", "qwen3b_v2": "Qwen 3B, round 2"}
CASE_ORDER = ["unscored", "flagged, nothing stands out", "flagged, 1 reason", "flagged, 2 reasons", "flagged, 3 reasons",
              "flagged, 4 reasons", "not flagged, nothing stands out", "not flagged, 1 reason", "not flagged, 2 reasons",
              "not flagged, 3 reasons", "not flagged, 4 reasons"]

# (title, file, eval tag, source data, how it was drawn, build log or None)
SETS = [
    ("DS7 v1 test (frozen, round 1)", ROOT / "data/processed/ds7/test.jsonl", "",
     "DS1 and DS2 transactions of the test-split addresses (DS7 splits by a seeded hash of the address).",
     "Round 1's DS7 build: up to 40 pairs per DS1 address and 10 per DS2 address, half flagged at most.", None),
    ("DS10 (teammate's snapshot)", ROOT / "data/processed/ds10/explain.jsonl", "ds10",
     "DS10: Le Quang Anh Hoang's independent Esplora snapshot (27 addresses), unseen by every model.",
     "The DS7 rule applied to DS10 by `explorer.cli ds10-check`; frozen.", None),
    ("Random: real", RANDOM / "real.jsonl", "random_real",
     "DS7 v2 test split: DS1 and DS2 test-split addresses.",
     "500 drawn at random (seed 50000), none in the frozen v1 test.", "build-random-test"),
    ("Random: synthetic", RANDOM / "synthetic.jsonl", "random_synthetic",
     "Made up: values, usual values and z drawn from settings randomTest, capped to realistic limits.",
     "500 transactions (seed 50000); z from a mixture, values derived from z.", "build-random-test"),
    ("Random: hard", RANDOM / "hard.jsonl", "random_hard",
     "Made up, explained for any of the nine methods.",
     "500 (seed 50001): 289 non-baseline, about half reworded and half shuffled.", "build-random-test"),
    ("Random: balanced", RANDOM / "balanced.jsonl", "random_balanced",
     "Every transaction of DS10 and of DS2's 173 test-split addresses (46,513 explained).",
     "De-duplicated, other test sets removed, up to 100 per case (seed 50002).", "build-balanced-test"),
    ("Crawled: DS1 test split", RANDOM / "crawl_ds1.jsonl", "random_crawl_ds1",
     "DS1 (our Esplora snapshot): the test-split addresses' transactions.",
     "Up to 60 per case (seed 50003); no prompt of an earlier set.", "build-more-tests"),
    ("Crawled: DS2 test split", RANDOM / "crawl_ds2.jsonl", "random_crawl_ds2",
     "DS2 (our Esplora crawl of 4,012 addresses): the 173 test-split addresses.",
     "Up to 60 per case (seed 50003); no prompt of an earlier set.", "build-more-tests"),
    ("Crawled: DS10", RANDOM / "crawl_ds10.jsonl", "random_crawl_ds10",
     "DS10 (the teammate's crawl): all 27 addresses.",
     "Up to 60 per case (seed 50003); no prompt of an earlier set.", "build-more-tests"),
    ("Close calls", RANDOM / "close_calls.jsonl", "random_close_calls",
     "DS1, DS2 and DS10 test-split transactions.",
     "300 whose listed |z| differ by less than 0.3, so order and pair are hardest.", "build-more-tests"),
    ("Extreme values", RANDOM / "extreme.jsonl", "random_extreme",
     "Made up, one extreme regime per transaction.",
     "300: 1-100 sat or about 1,000 BTC amounts, minute or multi-year gaps, 80-200 transactions a week.", "build-more-tests"),
    ("Maximal shift", RANDOM / "max_shift.jsonl", "random_max_shift",
     "Made up, for methods other than the baseline.",
     "300: every prompt reworded, its facts shuffled, the flag independent of the features.", "build-more-tests"),
]


def tex(text: str) -> str:
    """Text safe for LaTeX."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"), ("|", r"$|$")):
        text = text.replace(a, b)
    return text


def pct(v) -> str:
    return "--" if v is None or pd.isna(v) else f"{100 * v:.1f}"


def composition(path: Path) -> tuple[int, dict[str, int]]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r["kind"] != "glossary"]
    cases: dict[str, int] = {}
    for r in rows:
        c = assistant.case_of(facts_of(r["messages"][:-1]))
        cases[c] = cases.get(c, 0) + 1
    return len(rows), cases


def results(tag: str) -> pd.DataFrame | None:
    path = TABLES / f"assistant_eval{'_' + tag if tag else ''}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["order"] = frame["model"].map(list(WRITERS).index)
    return frame.sort_values("order")


def log_excerpt(step: str | None, title: str) -> list[str]:
    """The lines of the last run of `step` that mention this set (or the run's own summary lines)."""
    if not step or not (LOGS / f"{step}.log").exists():
        return []
    lines = (LOGS / f"{step}.log").read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [i for i, line in enumerate(lines) if "started" in line and step in line]
    run = lines[starts[-1]:] if starts else lines
    return [line[:150] for line in run if "Warning" not in line and "concat" not in line][:14]


def section(title: str, path: Path, tag: str, source: str, drawn: str, step: str | None) -> str:
    n, cases = composition(path)
    parts = [rf"\subsection*{{{tex(title)}}}",
             rf"\textbf{{Source.}} {tex(source)}\\ \textbf{{How drawn.}} {tex(drawn)}\\",
             rf"\textbf{{File.}} \texttt{{{tex(str(path.relative_to(ROOT)).replace(chr(92), '/'))}}} ({n} prompts)",
             r"\par\smallskip\noindent\begin{tabular}{lr}\toprule Case & Prompts\\\midrule"]
    parts += [rf"{tex(c)} & {cases[c]}\\" for c in CASE_ORDER if c in cases]
    parts.append(r"\bottomrule\end{tabular}")
    frame = results(tag)
    if frame is not None:
        parts.append(r"\par\smallskip\noindent\begin{tabular}{lrrrrrrr}\toprule "
                     r"Writer & Guardrail & Verdict & Reasons & Order & Fully & Macro & Nothing out\\\midrule")
        for r in frame.itertuples():
            parts.append(rf"{tex(WRITERS.get(r.model, r.model))} & {pct(r.guardrail_pass)} & {pct(getattr(r, 'verdict_accuracy', None))} & "
                         rf"{pct(getattr(r, 'reasons_accuracy', None))} & {pct(getattr(r, 'order_accuracy', None))} & "
                         rf"{pct(getattr(r, 'fully_correct', None))} & {pct(getattr(r, 'fully_correct_macro', None))} & "
                         rf"{pct(getattr(r, 'fully_correct_nothing_stands_out', None))}\\")
        parts.append(r"\bottomrule\end{tabular}")
        missing = [WRITERS[w] for w in ("text150m", "qwen3b", "qwen3b_v2", "text150m_v2") if w not in set(frame["model"])]
        if missing:
            parts.append(rf"\par\noindent{{\small Not yet scored on this set: {tex(', '.join(missing))}.}}")
    excerpt = log_excerpt(step, title)
    if excerpt:
        parts.append(r"\par\smallskip\noindent{\small Build log (\texttt{reports/logs/" + tex(step) + r".log}, last run):}"
                     + "\n" + r"\begin{Verbatim}[fontsize=\scriptsize]" + "\n" + "\n".join(excerpt) + "\n" + r"\end{Verbatim}")
    return "\n".join(parts)


METHOD_LABELS = {"baseline": "Robust-z baseline", "fusion": "Score fusion", "regression": "Regression residual",
                 "regression_interval": "Regression range", "kmeans": "K-Means distance", "isolation_forest": "Isolation Forest",
                 "dbscan": "DBSCAN noise", "lof": "Local Outlier Factor"}
CLASSIFIER_LABELS = {"logistic": "Logistic regression", "knn": "k-nearest neighbours", "randomForest": "Random forest",
                     "histGradientBoosting": "Gradient boosting"}


def statistics() -> str:
    """The statistics campaign: scoring methods over many plant seeds, classifiers over seeds and a bootstrap."""
    parts = [r"\section*{Statistics campaign}",
             "Distributions rather than single runs (requested 27 Oct). The scoring methods' planted patterns were "
             "replanted with many seeds (\\texttt{explorer/distribution.py}); the classifiers were refitted with ten seeds "
             "and their test scores bootstrapped 1,000 times (\\texttt{explorer/classify\\_distribution.py}). "
             "The language models were scored on 12 test sets with intervals and paired tests (\texttt{lm/eval\_stats.py})."]
    methods = TABLES / "stats_methods_summary.csv"
    if methods.exists():
        s = pd.read_csv(methods)
        parts.append(r"\subsection*{Scoring methods: mean planted recall over plant seeds}")
        parts.append(r"\begin{longtable}{llrrrrr}\toprule Data & Method & Seeds & Mean (sd) & 95\% of seeds & "
                     r"Beats baseline & Mean rank\\\midrule\endhead")
        for r in s.itertuples():
            parts.append(rf"{r.dataset} & {tex(METHOD_LABELS.get(r.method, r.method))} & {r.seeds} & {r.mean:.3f} ({r.sd:.3f}) & "
                         rf"{r.ci_low:.3f}--{r.ci_high:.3f} & {100 * r.beats_baseline_share:.0f}\% & {r.mean_rank:.1f}\\")
        parts.append(r"\bottomrule\end{longtable}")
        if (ROOT / "reports/figures/stats_methods.pdf").exists():
            parts.append(r"\begin{center}\includegraphics[width=\textwidth]{../figures/stats_methods.pdf}\end{center}")
    classifiers = TABLES / "stats_classifiers_summary.csv"
    if classifiers.exists():
        c = pd.read_csv(classifiers)
        parts.append(r"\subsection*{Classifiers: test F1 over training seeds, with a bootstrap of the test rows}")
        parts.append(r"\begin{longtable}{llrrrr}\toprule Data and split & Model & Seeds & F1 mean (seed sd) & Bootstrap 95\% CI, first seed & PR-AUC\\"
                     r"\midrule\endhead")
        for r in c.itertuples():
            parts.append(rf"{tex(r.scheme)} & {CLASSIFIER_LABELS.get(r.model, r.model)} & {r.seeds} & "
                         rf"{r.f1_mean:.3f} ({r.f1_seed_sd:.3f}) & {r.f1_ci_low:.3f}--{r.f1_ci_high:.3f} & {r.pr_auc_mean:.3f}\\")
        parts.append(r"\bottomrule\end{longtable}")
        if (ROOT / "reports/figures/stats_classifiers.pdf").exists():
            parts.append(r"\begin{center}\includegraphics[width=\textwidth]{../figures/stats_classifiers.pdf}\end{center}")
    writers = TABLES / "stats_writers_summary.csv"
    if writers.exists():
        w = pd.read_csv(writers)
        pairs = pd.read_csv(TABLES / "stats_writers_paired.csv")
        parts.append(r"\subsection*{Explanation writers: fully correct with 95\% bootstrap intervals, and paired tests}")
        parts.append(r"\begin{tabular}{llrr}\toprule Test set & Writer & Rate & 95\% CI\\\midrule")
        for r in w[w["test_set"].str.startswith(("pooled", "glossary"))].itertuples():
            parts.append(rf"{tex(r.test_set)} & {tex(WRITERS.get(r.writer, r.writer))} & {pct(r.fully_correct)} & "
                         rf"{pct(r.ci_low)}--{pct(r.ci_high)}\\")
        parts.append(r"\bottomrule\end{tabular}")
        q = pairs[(pairs["writer_a"] == "qwen3b") & (pairs["writer_b"] == "qwen3b_v2")]
        if len(q):
            parts.append(r"\par\smallskip\noindent Qwen round 1 minus round 2 on identical prompts (negative: round 2 better); "
                         r"McNemar's exact test counts the prompts only one of them got right.\par\smallskip")
            parts.append(r"\begin{longtable}{lrrrrr}\toprule Test set & Difference & 95\% CI & Only r1 right & "
                         r"Only r2 right & McNemar $p$\\\midrule\endhead")
            for r in q.itertuples():
                counts = "--" if pd.isna(r.only_a_right) else f"{int(r.only_a_right)} & {int(r.only_b_right)} & " + (r"$<$0.0001" if r.mcnemar_p < 1e-4 else f"{r.mcnemar_p:.4f}")
                counts = counts if counts != "--" else "-- & -- & --"
                parts.append(rf"{tex(r.test_set)} & {100 * r.diff:+.1f} & {100 * r.ci_low:+.1f} to {100 * r.ci_high:+.1f} & {counts}\\")
            parts.append(r"\bottomrule\end{longtable}")
        if (ROOT / "reports/figures/stats_writers.pdf").exists():
            parts.append(r"\begin{center}\includegraphics[width=0.9\textwidth]{../figures/stats_writers.pdf}\end{center}")
    return "\n".join(parts)


def glossary_section() -> str:
    """The open-question test: held-out phrasings of taught topics, judged by their key facts."""
    path = ROOT / "data/processed/ds7_v2/glossary_test.jsonl"
    frame = results("glossary")
    if not path.exists() or frame is None:
        return ""
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    parts = [r"\subsection*{Glossary test (open questions)}",
             rf"\textbf{{Source.}} \texttt{{tools/make\_glossary.py}}: {len(rows)} held-out phrasings of the 65 taught topics, "
             r"each with the key facts a correct answer must state.\\ \textbf{How drawn.} One or two per topic, written "
             r"differently from the taught questions; round 1's five test questions are among them.\\",
             r"\par\smallskip\noindent\begin{tabular}{lrrr}\toprule Writer & Key facts stated & Fully right & ROUGE-L\\\midrule"]
    for r in frame.itertuples():
        parts.append(rf"{tex(WRITERS.get(r.model, r.model))} & {pct(r.glossary_key_facts)} & {pct(r.glossary_correct)} & "
                     rf"{r.glossary_rouge_l:.3f}\\")
    parts.append(r"\bottomrule\end{tabular}")
    return "\n".join(parts)


def main() -> int:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    writers = ["template", "text150m", "qwen3b", "qwen3b_v2", "text150m_v2"]
    overview = [r"\begin{longtable}{p{4.2cm}rrrrrr}\toprule Test set & Prompts & Template & 152M r1 & Qwen r1 & Qwen r2 & "
                r"152M r2\\\midrule\endhead"]
    for title, path, tag, *_ in SETS:
        n, _ = composition(path)
        frame = results(tag)
        table = frame.set_index("model") if frame is not None and "fully_correct" in frame else None
        cells = [pct(table.at[m, "fully_correct"]) if table is not None and m in table.index else "--" for m in writers]
        overview.append(rf"{tex(title)} & {n} & {' & '.join(cells)}\\")
    overview.append(r"\bottomrule\end{longtable}")
    body = "\n\n".join(section(*s) for s in SETS) + "\n\n" + glossary_section()
    doc = rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[margin=2cm]{{geometry}}
\usepackage{{booktabs,longtable,fancyvrb,xcolor,graphicx}}
\usepackage[hidelinks]{{hyperref}}
\setlength{{\parindent}}{{0pt}}
\title{{Test sets, statistics and results}}
\author{{COS30049 A2, group HCMC\_COS30049.1\_4 (generated by \texttt{{tools/test\_report.py}})}}
\date{{{built}, commit \texttt{{{commit}}}}}
\begin{{document}}
\maketitle
\section*{{What is measured}}
Every writer answers each prompt; code then checks the answer (\texttt{{explorer/assistant.py}}, \texttt{{lm/eval\_explain.py}}).
\emph{{Guardrail}}: numbers only from the facts, comparison words that match the data, the right verdict (flagged, not
flagged, not scored), the limitation line on every flag, no disallowed wording. \emph{{Fully correct}}: the guardrail
passes and the answer names the template's reasons, strongest first, each number on its own feature's line (features
whose $|z|$ shows the same in the prompt may come in either order). \emph{{Macro}}: the mean over cases, so common easy
cases cannot carry the rate. \emph{{Nothing out}}: fully correct where no feature stands out, the one case in which a
writer must rank features itself. All figures are percentages. The template is the reference, so it scores 100 by
construction; it checks that each set is consistent.

\section*{{Overview: fully correct, by test set}}
{chr(10).join(overview)}

{statistics()}

\section*{{The language models' test sets}}
{body}
\end{{document}}
"""
    build = ROOT / "reports" / "assistant_tests"
    build.mkdir(parents=True, exist_ok=True)
    (build / "assistant_tests.tex").write_text(doc, encoding="utf-8")
    pdflatex = shutil.which("pdflatex") or str(Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe")
    for _ in range(2):  # longtable needs two passes
        done = subprocess.run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "assistant_tests.tex"], cwd=build,
                              capture_output=True, text=True)
        if done.returncode:
            print(done.stdout[-3000:])
            return 1
    shutil.copy(build / "assistant_tests.pdf", OUT)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
