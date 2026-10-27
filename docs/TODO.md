# A2 to-do list (whole project)

Written 27 Oct 2026 00:45 GMT+7 by Claude at Cao Tri Tran's request ("make yourself a todo list of the entire
project up to this point"; full decision rights overnight). Due Sun 8 Nov 23:59; the plan submits Sat 7 Nov.
`context.txt` holds the live state; this file is the checklist. Owner in brackets.

## Decide now (27 Oct)

- [x] Qwen dropped on the tutor's ruling (only models trained from scratch): removed from main, kept on the
      `qwen-archive` branch; glossary topic fixed; tables, cards, PDF rebuilt; pytest 218 passed (27 Oct). [Claude]
- [ ] Ask the tutor: DS6 (AI-written conversations) for the 152M chat stage? Claude-drafted glossary as training
      data? Qwen in the report at all? [Cao, in class]
- [ ] 152M pretraining length: (1) stop at this week's quota (~1.3B more tokens, done ~Thu 29 Oct), (2) 50 h across
      both accounts (~Fri 30 Oct; Claude's recommendation now that the 152M model is the only writer), (3) as
      planned (Sun 1 Nov). [Cao]
- [ ] Kaggle, before the 152M stages: push ds7_data (new slug cos30049a2-ds7-data; DS7 v2 with the fixed glossary),
      then text150m_stage_data with it as input. [Claude pushes from the PC]
- [x] README test count (27 Oct). [Claude]

## Done

- [x] DS1 snapshot (25 + 3 addresses), DS2 crawl (4,012 addresses), DS3 Elliptic, DS4 BitcoinHeist, DS5 FineWeb-Edu
      (2.0B tokens), DS6 smol-smoltalk, DS7 v1 (10,160 pairs), DS10 (teammate snapshot, cross-checked).
- [x] Features F1-F4 (time-safe), nine scoring methods, planted-pattern evaluation, stability, selection rule
      (baseline), conformal ranges, fusion, bootstrap intervals, sensitivity, walk-forward, DS2 check, DS10 check.
- [x] Classifiers on DS3/DS4 (temporal, random, leave-one-family-out); label check (unusual is not labelled).
- [x] TxGPT (sizes S/M/L, S kept); 152M model pretrained, chat- and explanation-tuned; Qwen 3B QLoRA (round 1).
- [x] predict.py with the guarded Assistant cascade; README; figures and EDA notebook; results-v1 tag.
- [x] Mac fresh-clone replication (all but regression/fusion identical; cause found).
- [x] Round 2 code: direction and verdict checks, strict measures (reasons, order, binding, per case), DS7 v2,
      glossary v2 (fact-checked), two-GPU training, WSD schedule, corpus continuation, batched evaluation, Kaggle
      notebooks (tools/kaggle_notebooks.py).
- [x] Randomized test sets: real, synthetic, hard, balanced (data/processed/random_test/).
- [x] Showcase page for the team (claude.ai artifact HREXw8jomq4fTu838wyAYo).

## Overnight 26-27 Oct [Claude]

- [x] Start 152M continued pretraining on honglquanganh (session 1 running since 27 Oct 01:10, T4 x2).
- [x] 3-4 more randomized test sets from the crawled data (user's request): more balanced draws, close calls,
      out-of-range values, maximal-shift prompts. Template control on each.
- [x] When Qwen v2 ends (~05:00-07:00): bring home, re-score with the strict measures, push the random-test
      evaluation (all writers, every set).
- [x] PDF test report: every test set (how built, from which crawled data, composition, build log) and every
      writer's results; reports/assistant_tests.pdf.
- [ ] Keep context.txt, CHANGELOG, AI_USE_LOG and the tracker current; commit and push each step.

## Round 2, language models (dates GMT+7)

- [x] Qwen v2: trained, evaluated and adopted (27 Oct), then dropped the same day on the tutor's ruling. [Kaggle]
- [ ] 152M continued pretraining until Sun 1 Nov 23:59 (honglquanganh, ~50 GPU h); resume each session with
      RESUME_FROM. [Claude pushes; Kaggle]
- [ ] Stage data (needs the ds7_data output) then chat + explanation stages after pretraining. [Claude]
- [x] Glossary test for the round-1 writer (in the stages notebook).
- [ ] Tue 3 Nov: adoption rule (KAGGLE_LM_PLAN.md section 3); tables into reports/tables; models into models/;
      tag results-v1.1 if any number changed. [Claude]
- [ ] Optional: Modal account (user) for extra 152M tokens; more teammate Kaggle accounts. [Cao, team]

## Statistics campaign (user, 27 Oct)

- [x] A: every scoring method over 100/100/20 plant seeds; B: classifiers over 10 seeds + test bootstrap;
      C: every writer on 12 test sets with bootstrap CIs and McNemar tests (stats_*.csv, PDF section).
- [ ] Rerun `python -m lm.eval_stats` after the 152M v2 evaluation (Tue 3 Nov). [Claude]
- [ ] C+: the 152M explanation stage retrained with 5 seeds (seed spread of a language model). [Kaggle]

## Fixes and decisions still open

- [ ] features.csv written with 12 significant digits (Mac/PC identical) -> rerun, results-v1.1. [Cao decides]
- [x] README fixes from the Mac replication: Miniforge / ToS note, pytest.ini (testpaths = tests), timings,
      LF line endings for processed CSVs, logs churn, eda_tests.csv without DS4 (27 Oct). [Claude]
- [x] TxGPT S seed 3 validation loss check: a seed-dependent validation half; full-set re-score keeps S seed 3
      (27 Oct).
- [ ] Packaging: DS4 final model too big (263 MB) -> smaller final model; models/assistant
      as a GitHub release asset (user's OK needed); fp16 export of the 152M model.
- [x] DATA_CARD.md, MODEL_CARD.md (generated tables: `python tools/cards.py`), LICENSES.md (27 Oct; the Qwen
      licence, NOTICE and "Built with Qwen" went with Qwen). [Claude]
- [x] `tools/package.py --tag v1.0 [--with-models]`: zip + SHA256SUMS.txt from the tag (27 Oct; trial from HEAD:
      71.7 MB, 477 files, data hashes match the manifests). [Claude]
- [ ] Find Canvas's upload limit (plan §11): decides whether --with-models (152M model
      610 MB fp32, ~305 MB fp16) goes in the zip or the weights go to a release link. Confirm the zip leaves out CLAUDE.md,
      context.txt, docs/HANDOFF.md and docs/brief/. [team]
- [ ] A licence for our own code (none chosen; LICENSES.md says ask the group). [team decides]

## Report (12 marks; <= 3,000 words, aim 2,850; Harvard, numbered)

- [ ] Outline is docs/REPORT_PLAN.md; write in LaTeX with the A1 preamble. Data collection and processing is the
      biggest item (5 marks). Quote reports/tables only. [team; Claude drafts on request]
- [ ] Robustness paragraph: power cut recovery, Mac replication, frozen tests, strict scoring.
- [ ] AI declaration from docs/AI_USE_LOG.md (glossary fact-checked by Claude, not a person, unless a teammate
      reads it).

## Team actions

- [ ] Blind reviewer check (two people who did not build the scoring). [Minh Khang, Cam Dan]
- [ ] Human read-through of the 65 glossary answers. [any teammate]
- [ ] Phone-verified Kaggle accounts from the other two members. [Minh Khang, Cam Dan]
- [ ] Hoang Anh regenerates his Kaggle token after the project (it was pasted into the chat).
- [ ] M4 Wed 4 Nov: a teammate rebuilds from the README on a fresh clone; code freeze. [teammate]
- [ ] Weekly minutes; contribution form (anyone under 33% is scaled down). [Cam Dan, all]
- [ ] Submit Sat 7 Nov. [Cam Dan]
