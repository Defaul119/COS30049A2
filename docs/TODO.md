# A2 to-do list (whole project)

Written 27 Oct 2026 00:45 GMT+7 by Claude at Cao Tri Tran's request ("make yourself a todo list of the entire
project up to this point"; full decision rights overnight). Due Sun 8 Nov 23:59; the plan submits Sat 7 Nov.
`context.txt` holds the live state; this file is the checklist. Owner in brackets.

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

- [ ] Start 152M continued pretraining on honglquanganh when its tokens are ready (~01:00).
- [ ] 3-4 more randomized test sets from the crawled data (user's request): more balanced draws, close calls,
      out-of-range values, maximal-shift prompts. Template control on each.
- [ ] When Qwen v2 ends (~05:00-07:00): bring home, re-score with the strict measures, push the random-test
      evaluation (all writers, every set).
- [ ] PDF test report: every test set (how built, from which crawled data, composition, build log) and every
      writer's results; reports/assistant_tests.pdf.
- [ ] Keep context.txt, CHANGELOG, AI_USE_LOG and the tracker current; commit and push each step.

## Round 2, language models (dates GMT+7)

- [ ] Qwen v2: training and its own evaluation (running since 26 Oct 19:05, tritran72). [Kaggle]
- [ ] 152M continued pretraining until Sun 1 Nov 23:59 (honglquanganh, ~50 GPU h); resume each session with
      RESUME_FROM. [Claude pushes; Kaggle]
- [ ] honglquanganh: stage data (needs its qwen-data output) then chat + explanation stages Mon 2 Nov. [Claude]
- [ ] Glossary test for both round-1 writers (in the Qwen v2 and stages notebooks).
- [ ] Tue 3 Nov: adoption rule (KAGGLE_LM_PLAN.md section 3); tables into reports/tables; models into models/;
      tag results-v1.1 if any number changed. [Claude]
- [ ] Optional: Modal account (user) for extra 152M tokens; more teammate Kaggle accounts. [Cao, team]

## Fixes and decisions still open

- [ ] features.csv written with 12 significant digits (Mac/PC identical) -> rerun, results-v1.1. [Cao decides]
- [ ] README fixes from the Mac replication: Miniforge / ToS note, pytest.ini (testpaths = tests), timings,
      LF line endings for processed CSVs, logs churn, eda_tests.csv without DS4. [Claude]
- [ ] TxGPT S seed 3 validation loss check before freezing.
- [ ] Packaging: DS4 final model too big (263 MB) -> smaller final model; models/assistant + models/qwen3b_adapter
      as GitHub release assets (user's OK needed); fp16 export of the 152M model.
- [ ] DATA_CARD.md (DS1-DS10, random test sets), MODEL_CARD.md, LICENSES.md (+ Qwen licence text, NOTICE,
      "Built with Qwen"), SHA256SUMS.txt.

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
