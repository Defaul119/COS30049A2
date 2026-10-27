# A2 report: plan to get started

Written 25 Oct 2026, when the training phase closed, from the brief (`docs/brief/`, unchanged in the
copy downloaded on 25 Oct), the plan's report chapter (`docs/plan/sections/s12-report.tex`) and the
results now in `reports/`. The plan's section-by-section budget still holds; this file adds what each
section can now quote and the order to write it in.

## What the brief asks for

- **Report**: PDF, at most 3,000 words (cover, contents, bibliography and appendix not counted), in this
  order: title; group identities (group name and ID, names and student IDs, **tutor: Arthur Hoang**);
  Introduction (motivation, **intended user**); Problem framing (challenge, limits of existing solutions,
  why ML); Data collection; Data processing; Model selection (criteria, alternatives, preliminary tests);
  Implementation: technical; Implementation: evaluation; Conclusion; Bibliography (Harvard);
  Appendix (cropped, explained screenshots; PDF exported directly, never a scan).
- **Zip**: processed datasets used by the final models (CSV/JSON), a README (conda set-up, processing,
  training, prediction), all source code and the final models.
- **Also**: meeting minutes (weekly, PDF) and the signed contribution form (PDF). One member submits
  everything, each file named after the group (`session-xx-group-n-AssignmentName.pdf` pattern).
- **Generative AI**: the brief bans it; the tutor allowed it on condition of a separate declaration.
  `docs/AI_USE_LOG.md` is its source.
- **Marks** (40): report 12 (collection and processing 5, analysis 2, model selection 2, writing 1,
  references 1, completeness 1); code 20 (core 10, datasets 4, README 2, comments 2, structure 2);
  HD bonus 8 (additional datasets 4, additional ML 4).

## What A1 set up (its report is in `_context/a1_submission/`, text in `_context/text/a1_report.txt`)

- **Project**: "Cryptocurrency transaction tracking and anomaly detection" (the Bitcoin Address Activity
  Explorer), framed as cybersecurity in financial services: banks, exchanges and payment providers that
  handle cryptocurrency must decide which transactions deserve a closer look. Of the brief's five topics this
  is closest to "cybersecurity threat detection in banking"; the Introduction should say so in one line.
- **Intended user**: a student researcher examining one supported Bitcoin address; analysts doing triage
  benefit too. The output is an unusual-activity indicator, never a judgement of fraud.
- **Promises the A2 report must close** (A1 objectives 1, 2 and 4, Table 2.4):
  - Snapshot of 20-30 addresses (50+ transactions over 90+ days) plus 2-3 short histories, with a manifest
    reviewed by a second member: met by DS1 (25 + 3); the second member's review is still due.
  - Baseline, Isolation Forest and LOF scoring every transaction by 26 Oct: met (`scores.csv`).
  - Label-free checks on the later part of the snapshot: planted recall >= 70% for 2 of 3 patterns, stability
    (top-50 Jaccard) >= 0.7, method agreement reported, a blind reviewer check with Cohen's kappa, the 20%
    sanity rule and the selection rule (highest recall among stable methods; within 5 points, the easier to
    explain): all computed (`eval_selection.csv`, `eval_stability.csv`, `eval_agreement.csv`,
    `eval_sanity.csv`) except the reviewer check, which needs two teammates who did not build the scoring.
  - Objective 3 (a walkthrough with two participants; high-severity issues fixed before A2): the team's status
    is unknown; report it if it happened, or say it moved to A3.

## Where each mark comes from now

| Rubric line | What we have | Gap |
|---|---|---|
| Collection and processing (5) | DS1 snapshot with manifest and checks; DS2 crawl (4,012 histories); DS3/DS4 labelled; DS5-DS7 text; DS10 refetched and cross-checked (5,785 of 5,785 rows); `cleaning_ds1.csv`, leakage controls, injection design | Write it; the most words go here |
| Analysis (2) | 13 EDA figures, `eda_tests.csv` | Pick 3-4 figures; each finding as observation, statistic, decision |
| Model selection (2) | criteria, grids (`tuning_*.csv`), stability gate, selection rule, `eval_significance.csv` (fusion +0.029, p = 0.063: baseline kept) | Condense the candidates table |
| Core code (10) | the pipeline, `predict.py`, guarded Assistant, tests (pytest; README gives the count) | README, comments pass |
| Datasets (4) | processed CSV/JSON under `data/processed/` | Decide what ships (only what final models use) |
| Additional datasets (4) | DS2, DS3, DS4, DS5, DS6, DS7, DS10 | Show each one's use in one table |
| Additional ML (4) | IF, LOF, K-Means, DBSCAN, regression + conformal, fusion, TxGPT and the 152M model (both trained from scratch) | Keep each to a sentence or a table row |
| Accuracy / precision / recall / F1 | DS3 and DS4 classifiers (`elliptic_results.csv`, `heist_results.csv`); planted recall on DS1, DS2, DS10 | State why DS1 has no accuracy or F1 (no labels) |

## Headline results to build the story on (quote the CSVs, never these lines)

- DS1: the robust-z baseline is the Explorer's method (mean planted recall 0.655; the fusion's 0.684 is
  not significantly better, p = 0.063). TxGPT is the most precise (PR-AUC 0.333 against 0.109) but
  flags only 1.5%.
- External tests of the frozen methods: DS2 (2,933 histories) keeps the ranking (Spearman 0.905); on
  DS10, data no model saw, TxGPT ranks first (0.598) and the baseline second (0.431); bursts are hard to
  find there because DS10's addresses are busier and less even (`ds10_activity.csv`).
- Labelled data: unsupervised scores do not match "illicit" labels (`label_check.csv`): "unusual" is
  not "labelled illicit", the report's central caution.
- Assistant: the 152M model (trained from scratch) passes the guardrail on 99.3% of DS7 test prompts and is
  fully correct on 87.0% (pooled over 12 test sets: 72.7%); round 2 results replace these if they pass the
  adoption rule. The report does not mention Qwen at all (user, 27 Oct, after the tutor's ruling).

## Before writing: freeze the results (M2)

1. Decide the guardrail direction check and the "Relative size" template wording; re-score the saved
   answers if adopted (no regeneration needed).
2. Check TxGPT S seed 3's low validation loss (0.661) in `txgpt_runs.csv`.
3. Re-run `python -m explorer.cli all --offline` once, run the tests, tag `results-v1`.

## Writing order and steps

1. **Skeleton**: LaTeX from the A1 preamble (`docs/a1/main.tex`, minus xltabular, pdflscape,
   csquotes and seqsplit); one file per brief section; `tools/wordcount.py` (body text only);
   a script that turns `reports/tables/*.csv` into LaTeX tables and one that checks every number in the
   text against the CSVs.
2. **Figures**: choose about 8 for the body (proposal: pipeline diagram, `eda_feature_distributions`,
   `eda_pca`, `eval_injection_recall`, `eval_significance`, `eval_ds10_check`, `eval_pr_curves`,
   `eval_label_check`); the rest go to the appendix.
3. **Write in this order** (budget words): Data collection 410 and Data processing 400 first (5 marks),
   then Data analysis 370, Model selection 400, Evaluation 560 and Technical 170, and last Introduction
   170, Problem framing 210 and Conclusion 160, so they match the results. Total 2,850.
4. **In parallel** (code marks): README (A2 outline: conda, processing, training, prediction),
   DATA_CARD.md (DS1-DS10), MODEL_CARD.md, LICENSES.md.
5. **Review**: Cam Dan Bui reads against the rubric and the completeness checklist (plan s12);
   references checked entry by entry; the number check and word count pass.
6. **Team items**: weekly A2 meeting minutes, the blind reviewer check, the AI declaration, and the
   contribution form (anyone under 33% has their mark scaled down).

## Open questions for the tutor

- Do tables and captions count towards the 3,000 words? Until answered, body text stays at or below 2,850.
