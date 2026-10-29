# A2 to-do list

The checklist for the whole project (live state in `context.txt`). Owner in brackets. Rewritten 29 Oct 2026.
Due Sun 8 Nov 23:59 GMT+7; the team submits Sat 7 Nov. Results freeze Tue 3 Nov; code freeze Wed 4 Nov.

## Training (docs/TRAINING_PLAN.md; every stage through the gate, a row in stages.csv)

- [x] M0 milestone (tag `m0`); stage gate (`explorer/stages.py`); frozen test files recorded.
- [x] A0 F9 feature: rejected. C1 baseline, C2 Isolation Forest, C6 regression: rejected.
- [x] C3 LOF, C4 K-Means, C5 DBSCAN, C6 regression range variant: accepted (models/stage/).
- [x] D1-D5 not run (validation F1 already 0.95-0.99); D6 ensemble rejected; D7 reported as a data limit.
- [ ] C7 TxGPT: event-level burst scoring, time tokens relative to the address's median gap, context / lr /
      dropout, sizes x seeds, seed ensemble. [Claude; Kaggle]
- [ ] C8 TxGPT data: sizes x data fractions (scaling study). [Claude; Kaggle]
- [ ] C9 fusion of every accepted method (TxGPT included); C10 selection rule over the accepted stages. [Claude]
- [ ] Test tier, once per accepted candidate (reporting seeds 30049-30053, DS2 test addresses). [Claude; Kaggle]
- [ ] B1 152M pretraining: s1-s3 done (2.29B of 3.0B, val 3.0967); s4 on dansns to 3.0B. [Kaggle]
- [ ] B3 weak-case explanation examples (hard, max_shift, nothing stands out) from training-split addresses with
      new seeds; B3-B5 recipe tried on the round-1 model. [Claude; Kaggle]
- [ ] B2-B6 on the round-2 model: ds7_data -> text150m_stage_data -> text150m_stages; dev gate by case. [Kaggle]
- [ ] Adoption rule (KAGGLE_LM_PLAN.md section 3); round-2 tables and model into reports/ and models/. [Claude]
- [ ] Optional C+: the 152M explanation stage with 5 seeds (a language model's seed spread). [Kaggle]

## Freeze (Tue 3 Nov)

- [ ] Everything once on DS10; `python -m lm.eval_stats`; `tools/test_report.py`; `tools/cards.py`; figures;
      tag `results-v1.1`. [Claude]
- [ ] features.csv with float_format="%.12g" (Mac = PC) before the freeze rerun. [Cao decides]

## Packaging and code (Wed 4 Nov code freeze)

- [ ] DS4 final model 263 MB -> a smaller final model (e.g. histogram gradient boosting). [Claude]
- [ ] fp16 export of the 152M model (~305 MB); weights as a release asset only with the user's OK. [Claude, Cao]
- [ ] Canvas upload limit (decides `--with-models`); confirm the zip leaves out CLAUDE.md, context.txt,
      docs/HANDOFF.md, docs/brief/. [team]
- [ ] `python tools/package.py --tag v1.0`; README test count and timings checked. [Claude]
- [ ] A licence for our own code (LICENSES.md says ask the group). [team decides]
- [ ] M4: a teammate rebuilds everything from the README on a fresh clone; tag `v1.0`. [teammate]

## Report (not started on purpose: the user, 29 Oct)

- [ ] Outline in docs/REPORT_PLAN.md; LaTeX with the A1 preamble; quote reports/tables only; no Qwen. [team, Claude]
- [ ] Robustness paragraph: power cut, Mac replication, frozen tests, strict scoring, lost hand-off.
- [ ] AI declaration from docs/AI_USE_LOG.md (glossary fact-checked by Claude, not a person, unless a teammate
      reads it).

## Team actions

- [ ] Blind reviewer check (two people who did not build the scoring). [Minh Khang, Cam Dan]
- [ ] Human read-through of the 65 glossary topics. [any teammate]
- [ ] One DS1 transaction checked by hand on a block explorer. [anyone]
- [ ] Teammates regenerate their Kaggle tokens after the project; the user rotates tritran72's.
- [ ] Weekly minutes; contribution form (anyone under 33% is scaled down; remind the user). [Cam Dan, all]
- [ ] Submit Sat 7 Nov. [Cam Dan]

## Done (details in CHANGELOG.md)

- [x] Data: DS1, DS2 (4,012 addresses), DS3, DS4, DS5 (2.0B + 3.0B tokens), DS6, DS7 v1/v2, glossary (306 pairs),
      DS10 (cross-checked), randomized test sets (10).
- [x] Features F1-F4 (time-safe); nine scoring methods; plants, stability, selection (baseline); conformal ranges;
      fusion; significance, sensitivity, walk-forward, DS2 check, DS10 check; statistics A/B/C.
- [x] Classifiers on DS3/DS4; label check ("unusual" is not "labelled illicit").
- [x] TxGPT S/M/L (S kept); 152M round 1 (pretrain, chat, explanation); guarded Assistant in predict.py.
- [x] README (A2 guide), DATA_CARD, MODEL_CARD, LICENSES, tools/package.py; Mac replication; results-v1 tag.
- [x] Qwen dropped on the tutor's ruling (27 Oct; branch qwen-archive); tutor accepted DS6 and the glossary.
