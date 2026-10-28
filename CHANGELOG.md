# Change log

Changes to settings, data, decisions and the plan, newest first. Where the plan and reality
disagree, the measured reality is recorded here (CLAUDE.md, "How to work here").

## 28 October 2026

**152M session 2 finished; session 3 started.** honglquanganh v2 (11.47 h): step 5,616 (~1.47B of the 3.0B new
tokens), validation loss 3.1262 at step 5,600, now below round 1's 3.1304. The overnight handover had been stopped
by Claude Code for low memory before it did anything, so session 3 was pushed by hand at 06:35 on kainetyson
(checkpoint dataset kainetyson/cos30049a2-s3-text-150m-cont; data manifests identical).

**Stages C1-C6 prepared (`explorer/stage_search.py`, settings `stageSearch`, Kaggle notebook `stage_search`).** Each
DS1 scoring method's grid is searched at the method's M0 flag budget (its M0 dev flag rate of the real evaluation
rows, capped at 20%), ranked on tuning seeds 40000-40002, and only the winner is gated against M0 on seeds
40003-40004; `stage-compare` takes the seeds to use. Rules fixed before any result (TRAINING_PLAN section 4). F9
returns as a candidate input (stage A0 was rejected for raising DBSCAN's flag rate, which the budget now rules out).

**Stage D6 prepared (`explorer/ensemble.py`, settings `ensemble`, Kaggle notebook `ensemble`; the user, 28 Oct).**
Soft voting (weights 0/1/2/4 per member, at least two members) or stacking (logistic regression on the members'
log-odds) of the four Elliptic classifiers; chosen on validation steps 27-30, gated against the random forest on
steps 31-34, tested once on steps 35-49 only if accepted. D1-D5 skipped (validation F1 already 0.95-0.99) and D7
reported as a data limit (TRAINING_PLAN section 4).

## 27 October 2026 (afternoon)

**Stage A0 rejected by the gate (Kaggle CPU, tritran72 cos30049a2-stage-cpu v2, commit f0e7b70, ~1 h).** Both passes
(M0 as committed, then `features.pace.enabled` true) refit every DS1 method in one session; dev = the tuning plants
only. Mean planted recall M0 -> A0 (paired address bootstrap, 95%): DBSCAN 0.422 -> 0.587 (+0.165 [+0.127, +0.198];
spike 0.436 -> 0.732, burst 0.097 -> 0.261) but its flag rate nearly doubled (0.039 -> 0.070) and its largest
address share rose 0.377 -> 0.434, so rejected; regression 0.693 -> 0.606 (burst 0.723 -> 0.542); Isolation Forest
0.445 -> 0.404; K-Means 0.463 -> 0.433; LOF, fusion, TxGPT and the baseline unchanged within their intervals (the
baseline and TxGPT do not read F9). No method passed, so F9 stays off (settings unchanged) and M0 remains the last
accepted stage for every scoring method. Recorded: `reports/tables/stages.csv`, `stage_dev/{M0,A0}_*.csv`, logs in
`reports/logs/stages/A0/`. Noted: at M0, regression's dev flag rate (0.224) is already over the 20% sanity limit.
A possible follow-up inside stage C5 (DBSCAN's own stage): F9 as a candidate input with the threshold re-chosen at
M0's flag budget, so a gain cannot come from flagging more.

**Stage A0 prepared: F9, burst on the address's own clock** (docs/TRAINING_PLAN.md; option 1, the user's choice).
- *F9:* strictly earlier transactions inside the last `features.pace.windowGaps` (3) x the address's own median gap
  over its timing window (at least `minGapHours`, 1), with its robust z against the same window (spread floor
  `minScale`, 1). F2 counts a fixed 7 days, so ten transactions in a day barely move it on a busy address.
- *Scope:* an input to the pooled scikit-learn methods (`zc_F9` in `methods.INPUT_COLUMNS`) and to regression
  (`F9` beside F2) only. The baseline, the explanations, the A1 prototype parity and DS7 keep F1-F4.
- *Switch:* `features.pace.enabled` is false as committed, so every saved model and result is unchanged; the stage
  run sets it true. It is committed as true only if A0 passes the gate.
- *Gate tooling:* `stage-dev` (every saved method on the tuning plants, seeds 40000-40004, plus the sanity rule on
  real evaluation rows), `stage-compare` (address-level paired bootstrap of the mean pattern recall; per-pattern drop
  against M0's seed SD from `stats_methods_seeds.csv`) -> `reports/tables/stages.csv`. Kaggle CPU notebook
  `kaggle/stage_cpu` runs both passes (before and after the overrides) in one session (the user: CPU stages on
  Kaggle only). `tools/kaggle_notebooks.py <commit> [folder ...]` now rewrites only the named notebooks.

**152M continued pretraining, session 1 done** (`honglquanganh/cos30049a2-text150m-train` v1, T4 x2, commit
`028ec4c`, 11.46 h).
- *fp16 check:* round 1's weights give validation loss 3.1304 in fp16 (bf16 on the 3060: 3.1305; tolerance 0.01). Pass.
- *Bench:* micro-batch 8, 11,382 tok/s per GPU (22,765 on both). Measured in training: ~18,550 tok/s (the bench
  overstates by ~19%: evaluations and checkpoints are not in it).
- *Gate:* GPU_HOURS 50 at the bench rate allows ~3.7B tokens, so the gate took all **3.0B prepared tokens**
  (`total_tokens` 3,000,762,368; ~5.0B in all with round 1). This is the 3.0B option left open on 27 Oct; it
  needs no decision now. Left after session 1: ~2.25B tokens = ~34 h at the measured rate = 3 more sessions. The
  decay (last 15%) starts at 2.55B, in session 4.
- *Trained:* 2,881 steps x 262,144 = 0.755B tokens. Validation loss (same 160 windows): 3.1304 -> 3.1913 (step 800,
  after the warm-up back to the peak learning rate) -> 3.1650 (step 2800), falling again. A rise after re-warming is
  expected with a warm-up / stable / decay schedule; the decay brings the loss down. So TRAINING_PLAN B1's "falls
  after every session" applies from session 2 on (it must fall from 3.1650), and against round 1's 3.1305 only
  after the decay.
- *Session 2* pushed 13:15 (v2, same account, resumes from v1's output). The notebook's `RESUME_FROM` now accepts
  `"auto"`: it finds the input holding `text-150m-cont/ckpt_last.pt` wherever Kaggle mounts it (inputs moved to
  `/kaggle/input/datasets/<user>/...`), which also covers the cross-account dataset of session 3.

## 27 October 2026 (late morning)

**Qwen dropped: only models trained from scratch.** The tutor confirmed (the user, in class) the message below:
a ready-made model that generates answers (Qwen) is not accepted in the model training work; only what the group
builds and trains counts. This reverses the 22 Oct addition to CR-06 decision 4 and the 24 Oct relaxation; the
project is back to decision 4 as first written (TxGPT and the 152M model, both trained from scratch).
- *Kept:* branch `qwen-archive` (= `b1f5c1b`) holds every Qwen file, table and result, so the work is traceable.
- *Removed from main:* `lm/qlora.py`, `lm/chat_subset.py`, `configs/qwen3b_*`, the Qwen Kaggle notebooks and the
  random-set notebook that evaluated Qwen, every Qwen answer table and summary row, `NOTICE` and the Qwen licence,
  the GPU environment's transformers / peft / bitsandbytes / accelerate, the Qwen writer and its tests. The
  Assistant's `writerOrder` is now `["text150m"]`, with the template as the fallback.
- *Renamed:* `kaggle/qwen_data` -> `kaggle/ds7_data` (slug `cos30049a2-ds7-data`); it builds DS7 v2 only.
- *Glossary:* the topic that said Qwen writes first ("two-assistants") is now "assistant-model" (the 152M model,
  trained from scratch); `glossary.jsonl` regenerated, and the one affected frozen test question
  (`glossary_test-00053`) rebuilt the same way `build-ds7` writes it. DS7 v2 and the 152M stage data are rebuilt
  on Kaggle from this before the stages run, so the 152M model is never taught that Qwen exists.
- *Results:* `lm.eval_stats` rerun; the 152M model's numbers are unchanged (pooled fully correct 72.7%); the
  writer figure, the test-report PDF and the cards are rebuilt. Tests: 218 passed, none skipped (the three
  skipped tests and the adapter-save test were Qwen's).

**Qwen under review (09:00; superseded by the entry above).** The user forwarded a message (in Vietnamese; the sender is not yet recorded)
saying Qwen, as a ready-made AI that generates answers, won't be accepted in the model training work, and that only
the algorithms and models the group built and trained should be integrated. It contradicts the tutor's 24 Oct
announcement that fine-tuning a public model is acceptable. Nothing is removed until the user decides; the full
message, its translation and the list of changes if Qwen is dropped are in `context.txt` (Part 1).

**Working notes reorganized.** `context.txt` is now live state / reference / history, with no detail dropped;
`CLAUDE.md`'s status, decision 4, machine and repository sections describe 27 Oct instead of 20 Oct;
`docs/HANDOFF.md` is marked historical; `docs/TODO.md` gains a "Decide now" section. README: expected test count
219 passed, 3 skipped (the card and package tests came after the 215 count).

## 27 October 2026 (overnight; the user gave Claude full decision rights)

**Qwen 3B round 2 adopted** (by the rule fixed in advance, KAGGLE_LM_PLAN.md section 3).
- *Run:* `tritran72/cos30049a2-qwen3b-explain-v2` on two T4s, 619 tok/s (round 1: 326 on one), stopped early by
  `patience` at step 2,600 of 5,820, best validation loss 0.0234, 7.9 h.
- *Scoring:* its answers were re-scored here with the current measures (its notebook predates them).

| Rule (all must hold) | Round 1 | Round 2 |
|---|---|---|
| Guardrail pass, DS7 v1 test / DS10 | 98.9% / 98.6% | 99.8% / 99.3% |
| Number faithfulness | 100% / 100% | 100% / 100% |
| Wrong comparison words | 5 / 4 | 1 / 2 |
| Glossary test: key facts | 15.3% | 41.3% (+26 points) |

Also better: fully correct on DS7 v1 test / DS10 95.5% / 95.9% -> 96.2% / 96.6%, and glossary answers fully right
6.8% -> 31.1%. It no longer copies the template word for word (copy rate on flagged prompts 96% -> 0%), so ROUGE-L
against the template falls (0.783 -> 0.669) while every check rises.

`settings.json -> assistant` now points at `models/qwen3b_adapter_v2` with `configs/qwen3b_explain_v2.json`. Round 1
stays in `models/qwen3b_adapter` as the fallback. Adapter SHA-256: adapter_model.safetensors 9b1c5f40...0890,
adapter_config.json 91e3b7f7...0ddf (full hashes in context.txt). `data/processed/ds7_v2/glossary_test.jsonl` (74
held-out questions) is now committed as a frozen test set.

**Statistics campaign A: every scoring method's planted-pattern recall over many seeds** (`explorer/distribution.py`,
Kaggle CPU, 1.8 h; seeds 70000-; `stats_methods_{seeds,summary}.csv`).

| Data set | Seeds | Baseline mean recall (sd) | Baseline first | Fusion beats baseline |
|---|---|---|---|---|
| DS1 | 100 | 0.661 (0.035) | 4% | 95% |
| DS10 | 100 | 0.449 (0.031) | 99% | 1% |
| DS2 | 20 | 0.502 (0.010) | 100% | 0% |

On DS1, where fusion and regression were tuned, fusion beats the baseline on 95% of seeds (0.685, +0.024) and
regression on 59%. On addresses the methods never saw, the baseline ranks first on 99-100% of seeds, ahead of
fusion every time. The selection rule (within five points, the easier method to explain) picked the method that
transfers best; the tuned methods' DS1 advantage is partly fitted to DS1. TxGPT is not in this study (its scoring
needs a GPU).

**Statistics campaign B: classifiers over 10 training seeds and a 1,000-draw test bootstrap**
(`explorer/classify_distribution.py`, Kaggle CPU, 63 min; `stats_classifiers_{seeds,summary,paired}.csv`). The seed
means reproduce the reported numbers; for example, the Elliptic random forest has F1 0.815, seed SD 0.002, 95% CI
0.796-0.835. On Elliptic the forest beats every other model on all 1,000 paired resamples (over gradient boosting:
+0.042, CI 0.031-0.053). On BitcoinHeist the temporal split stays near chance for every model (best F1 0.070) and the
random split reaches 0.246. Seed spread is small everywhere (SD 0.013 at most), so the model ranking is not luck.

**TxGPT runs re-scored on the full validation set (the open "seed 3" check).** During training, lm.train scored
the first 320 of 643 validation windows of a stream shuffled with the run's seed, so each seed saw a different half
and seed 3 looked 0.02 better at every size (0.661 vs 0.682). `lm.tx_select` now scores every run on all validation
and test windows (CPU, about 9 min) and selects on those: mean loss S 0.6717, M 0.6731, L 0.6742 (seed SD now
0.001, was 0.012). The choice is unchanged (S, seed 3; `models/txgpt.pt` byte-identical), so no downstream result
moves; `txgpt_runs.csv`, `txgpt_sizes.csv`, the size figure and the model card show the full-set losses, with the
training-time ones kept as `best_val_loss`.

**Submission zip** (`python tools/package.py --tag v1.0`, plan Appendix D). Built from `git archive` of the tag, so
nothing uncommitted can leak in; leaves out CLAUDE.md, context.txt, docs/HANDOFF.md and the unit's brief (for the
team to confirm); writes SHA256SUMS.txt into the zip and beside it, then re-reads the zip against it.
`--with-models` adds the writers' weights. A trial from HEAD: 71.7 MB, 477 files, and the DS1, DS2, DS7 and DS10
files' SHA-256 equal their manifests'.

**Data card, model card and licences.** `DATA_CARD.md` (every dataset's source, licence, collection, size, missing
values, biases and users, then every shipped file with its rows, size and SHA-256 and every column with its type,
missing count and meaning) and `MODEL_CARD.md` (every final model's purpose, data, parameters, limits and intended
use). Their tables are written by `python tools/cards.py` from the files and `reports/tables/`, so no number is
typed by hand; `tests/test_cards.py` fails if a column has no description. `LICENSES.md`, `NOTICE` and
`licenses/QWEN_RESEARCH_LICENSE.txt` meet the Qwen licence's three conditions (licence text, attribution line,
"Built with Qwen"). No licence is chosen for our own code yet (team decision).

**README brought up to date (Mac replication findings 1-7, round 2).** Miniforge or the Miniconda terms-of-service
step; `pytest.ini` (testpaths = tests) so a full `_context/` cannot break collection (still 210 passed, 3 skipped);
test time about two minutes; ds10-check about 40 min on a laptop; which `git status` changes after a rebuild are
harmless (logs, eda_tests.csv without DS4, line endings: `git diff --ignore-cr-at-eol`, checked on a DS10 file);
the Qwen round 2 adapter and config, the writer order, the statistics steps, the random test sets and the test
report PDF. The features.csv rounding fix is still the user's decision.

**Statistics campaign C: the explanation writers with intervals and paired tests** (`python -m lm.eval_stats`, no
model runs; settings `writerStats`, seed 72000, 2,000 draws; `stats_writers_{summary,paired}.csv`, figure
`stats_writers.pdf`). Every writer answered the same prompts on 12 test sets (DS7 v1 test, DS10 and the ten
randomized sets), so each set is resampled once and every comparison is paired. Pooled over the 12 sets (4,948
prompts, each set weighted equally), fully correct is 72.7% (95% CI 71.6-73.8) for the 152M model, 88.9% (88.1-89.8)
for Qwen round 1 and 92.7% (92.0-93.4) for Qwen round 2. McNemar's exact test on the prompts only one Qwen round got
right finds round 2 better on synthetic (p 0.0005), hard (0.027), balanced (+11.1 points, p < 0.0001) and max_shift
(+19.7 points, p < 0.0001), and on glossary key facts (+25.9 points, CI 16.2-35.8). The other eight sets show no
significant difference, and round 2 is never significantly worse.

**Order and reasons checks made fair to rounding.** The prompt shows z to one decimal, and the template orders by
the unrounded value. Two features can therefore show the same |z| in an order the model cannot know, and when nothing
stands out the second and third largest can tie. `order_ok` and `reasons_ok` now accept any choice among features
that show the same |z|. Most of the "wrong pair" misses were such ties.

Round 1, re-scored:

| Model | DS7 test, fully correct | DS10, fully correct | Reasons right | Order right |
|---|---|---|---|---|
| 152M | 87.0% | 83.2% | 97.8% (DS7) | 87.6% |
| Qwen | 95.5% | 95.9% | 99.8% (DS7) | 95.7% |

The 152M model's main weakness is the order of its reasons.

**Six more randomized test sets** (`build-more-tests`, settings `randomTest` -> `_more`, seed 50003). None repeats
a prompt of an earlier set, and the template control passes all of them:
- `crawl_ds1`, `crawl_ds2`, `crawl_ds10`: one set per crawled source (174, 421 and 334 prompts), up to 60 per case.
- `close_calls`: 300 real prompts whose listed |z| differ by under 0.3.
- `extreme`: 300 synthetic, one extreme regime each (1-100 sat or ~1,000 BTC amounts; minute or multi-year gaps;
  80-200 transactions a week).
- `max_shift`: 300 prompts, every one reworded, shuffled and for a non-baseline method.

**DS7 v2 rebuilt on the teammate's Kaggle account is byte-identical** to the build on the user's account (every data
file's SHA-256 equal, from commits 44a63f8 and a2c0d43): the build is deterministic across machines and code versions.

## 26 October 2026

**Even stricter scoring, results by case, and a balanced real test set (user, ~22:00: "a bit too high... could be
data imbalance").** The checks found three reasons the rates were high:
- *Easy cases dominate.* Where features have reasons, the prompt labels their roles, so naming them is close to
  copying (about 100%). The one real decision is "not flagged, nothing stands out": pick the two largest |z|. It is
  37% of the DS7 test, and there both models score 80-82%.
- *Duplicates.* Every unscored prompt is the same sentence: 38 of the DS7 test's 445 prompts are repeats, 31 of DS10's.
- *Lenient checks.* Two are added to "fully correct": `order_ok` (reasons strongest first, as the template
  promises) and `binding_ok` (each number on its own feature's line). `lm.eval_explain` also reports
  `fully_correct_macro` (the mean over cases), `fully_correct_nothing_stands_out` and `fully_correct_unique`, and
  writes `assistant_eval_by_case*.csv`.

Round 1 re-scored, same answers:

| Test | Model | Fully correct | Mean over cases | Nothing stands out | Order right |
|---|---|---|---|---|---|
| DS7 v1 test | 152M | 83.8% | 71.2% | 80.2% | 84.5% |
| DS7 v1 test | Qwen | 91.2% | 90.0% | 80.8% | 91.5% |
| DS10 | 152M | 81.4% | 71.2% | 78.4% | 82.1% |
| DS10 | Qwen | 92.4% | 86.9% | 82.4% | 92.4% |

Binding was 99.8-100%: neither model puts a number under the wrong feature.

`build-balanced-test` (`random_test/balanced.jsonl`, seed 50002): every transaction of DS10's 27 addresses and
DS2's 173 test-split addresses explained (46,513), de-duplicated and stripped of the other test sets' prompts, then
up to 100 drawn per case: 883 prompts. The natural mix confirms the imbalance: 27,139 of the unique prompts are
"nothing stands out", 87 are "not flagged, 4 reasons", and all unscored prompts collapse to 1. The data is cleaned
and balanced in pandas, not KNIME (user's suggestion): a GUI workflow could not be rerun from the conda environment
or reviewed in git, and the brief names pandas.

**Stricter scoring and a hard randomized set (user, late evening: "that's high... try another randomized
dataset").** The guardrail's 99% only checks surface rules, so:
- *Verdict check, now in the guardrail* (`explorer.assistant.verdict_ok`): the answer must state the right result
  (flagged, not flagged or not scored). "Not flagged" for a flagged transaction kept every number and passed before.
- *Reasons check, an evaluation measure* (`reasons_ok`): the answer must name exactly the template's features.
  `fully_correct` = guardrail passed and reasons right.

Round 1 re-scored on the same answers (verdicts 100% everywhere):

| Test | Model | Reasons right | Fully correct |
|---|---|---|---|
| DS7 v1 test | 152M | 95.7% | 95.1% |
| DS7 v1 test | Qwen | 96.0% | 95.5% |
| DS10 | 152M | 94.8% | 94.2% |
| DS10 | Qwen | 96.2% | 95.9% |

The misses are real: where no feature stands out, the template lists the two largest differences, and both models
often pick the wrong pair (that choice needs z values compared).

`random_test/hard.jsonl` (seed 50001; real.jsonl and synthetic.jsonl unchanged) has 500 synthetic transactions
explained for any of the nine methods: 289 non-baseline, with the flag independent of the features (71 not flagged
despite a flag-level feature, 13 flagged with nothing standing out). 249 have a reworded instruction and 223 shuffled
facts, none of which either model trained on. The template control passes everything on it.

**A randomized test set for both Assistants (user's request, evening).** `python -m explorer.cli build-random-test`
(`explorer/random_test.py`, settings `randomTest`, seed 50000) writes `data/processed/random_test/`. It is
built once and frozen, so all four writers (152M and Qwen, rounds 1 and 2) answer the same prompts:
- `real.jsonl`: 500 prompts drawn from DS7 v2's test split (1,401 eligible). Both rounds split by the same address
  hash, so neither trained on these addresses, and no prompt of the frozen v1 test set is included.
- `synthetic.jsonl`: 500 made-up transactions. Each feature's z is drawn from a mixture (within usual,
  contributing, flag-level, near 0, either sign), and its value follows from z, a drawn usual value and spread on
  the feature's own scale, capped at 1,000 BTC, 5 years and 500 transactions a week. Roles and the flag follow
  explorer.predict. 5% have no size comparison and 9% are unscored.

The template control passes everything on both. The writers are scored on Kaggle: Qwen rounds 1-2 and the 152M
round 1 by `kaggle/assistant_random_eval`, the 152M round 2 by `kaggle/text150m_stages`.

**Glossary fact-checked by Claude instead of a team read-through (user's request, evening).** Every one of the 65
answers was checked against the code, `config/settings.json` and the result tables. Fifty-nine were accurate.
Six were corrected:
- the address selection: 20 evenly spaced blocks with addresses visited in shuffled order, not "randomly sampled
  recent blocks";
- why the threshold is 5: no tested threshold met the 20% limit; 3.5 left 7 addresses over it and 5 left 4, while
  still finding at least 70% of 2 of the 3 patterns (now read from `tuning_baseline.csv`);
- "largest differences": these also appear for most ordinary transactions (76% of unflagged DS1 transactions
  have no feature at |z| >= 2);
- the two assistants: they write only when the Assistant is switched on;
- reproducibility: one command rebuilds the features, scoring models and tables, not "every model";
- Size: the amount is what was received minus what was sent.

This is not the human read-through the plan asked for (§7 risk), and the AI declaration should say so. The 152M
round-1 weights, tokenizer and validation tokens were uploaded as a private Kaggle dataset on the teammate's account
(`honglquanganh/cos30049a2-text150m-base`, user's OK).

**Round 2's Qwen side: DS7 v2, the glossary test, two-GPU training and batched evaluation (C5, C7, C8, C9).**
- *The glossary test now holds out questions, not topics (differs from the plan, §4).* Round 1 split the glossary
  by topic, so its 5 test prompts were on 2 topics no model was ever taught ("assistant", "drift"). A correct answer
  was impossible, which is why both models scored 0 of 5. A model can only be asked to recall what it was taught, so
  `tools/make_glossary.py` now teaches every topic and holds out new phrasings of each: 306 pairs on 65 topics (210
  taught, 22 validation, 74 test), each topic with key facts that a correct answer must state. Round 1's 5 test
  questions are among the held-out ones, so the frozen v1 test set stays unseen. Several answers were out of date
  and are corrected: the writer order (Qwen first), nine methods (fusion added), the direction check, DS2 and DS10.
- `build-ds7 --out data/processed/ds7_v2 --settings ds7v2`: every DS2 address, up to 15 pairs each (target
  40,000), with the same split rule as round 1. The build stops if a frozen v1 test prompt would be trained on.
  It runs in a Kaggle CPU notebook (`kaggle/qwen_data`); only its manifest comes home.
- `lm.chat_subset --retention-start 2500`: 10,000 DS6 conversations with round 1's 200 retention conversations
  still held out (checked byte for byte, and round 1's three files rebuild identically).
- `lm/qlora.py` trains on every GPU of a session under torchrun (`lm/dist.py`). Round 1's Kaggle run used one of
  the two T4s. A 2-process CPU run reproduces the 1-process run exactly (same losses, identical adapter). New
  options: `repeat_sources` (glossary x4), `monitor_sources` (glossary validation loss on its own), `patience`
  (early stop). `configs/qwen3b_explain_v2.json`: a fresh adapter, at most 2 epochs.
- `lm.eval_explain --batch-size`: Qwen answers many prompts per pass (round 1: one prompt every 11 s on a T4).

**Guardrail direction check and the Relative size wording (round 2, C6; user's decision).** The guardrail
(`explorer.assistant.check`) now also rejects a comparison phrase that contradicts its feature's z sign, e.g.
"Quieter week than usual" for a busier week, which the number check cannot see. `lm.eval_explain` adds
`direction_ok`, `rouge_own` (ROUGE-L against the row's own reply), `copy` and glossary `key_facts`, and `--rescore`
measures saved answers again without a model. Round 1 re-scored (same answers): DS7 test, the 152M model has 2
direction errors and Qwen 5, so the guardrail pass falls from 99.8% / 100% to 99.3% / 98.9%; DS10, 2 and 4 (99.3% /
98.6%). The hand count of 25 Oct (1 and 4) missed three: two are the 152M model writing "Small compared with the
usual amount" for 0.4x against a usual 0.3x, which is what the A1 wording invited. So the template's F4 headline now
compares the ratio with its usual ratio ("Higher / Lower ratio to the usual amount than usual"), and its detail adds
"against a usual 0.3x" like every other feature. The Explorer's explanation text changes for F4 lines only. The DS7
v1 test and DS10 prompt files stay frozen (`ds10-check` no longer rewrites an existing `explain.jsonl`), and the
direction check also knows the A1 wording, so both rounds are judged on identical tests.

**The PC's GPU is no longer used (user).** Every GPU step of round 2 runs on Kaggle, or on other free cloud GPUs if
they add capacity. A teammate's Kaggle account (Le Quang Anh Hoang) arrived the same day, so the 152M model can start
before the 31 Oct quota reset. The token is kept outside the repository.

**Language models, round 2, on Kaggle (user).** The user is not happy with the language models' accuracy, so the
152M model gets 4-5B more pretraining tokens and Qwen 3B another 12-20 GPU hours, with every data, training and
evaluation step on Kaggle (`docs/KAGGLE_LM_PLAN.md`). This reverses the 24 Oct decision to give the 152M model no
more GPU time. The saved DS7 answers show where accuracy is lost: explanations keep every number in 99.8% (152M)
and 100% (Qwen) of answers, but both models answer none of the 5 test glossary prompts correctly, and those cover
only 2 topics. ROUGE-L against the template also understates correct paraphrases (152M: 0.683 against the
template, 0.719 against each row's own reply). So round 2 first adds what it needs to measure: a held-out glossary
test with key facts per topic, the direction check and a copy rate. It also fixes, before any result exists, the
rule under which a round-2 model replaces round 1. The round-1 models (results-v1) stay the fallback.

**A fresh clone rebuilt on a MacBook (Apple M5) reproduces every result except regression and fusion.**
The README was followed exactly. Tests and the quick check match, and `all --offline` ran in 65 min. The baseline,
Isolation Forest, LOF, K-Means, DBSCAN, regression_interval and TxGPT agree to 1e-4 on DS1, DS2 and DS10.
Regression's DS1 mean recall moves from 0.673 to 0.659, fusion's from 0.684 to 0.675 (p against the baseline from
0.063 to 0.325), and no conclusion changes. Cause: `features.csv` differs between the two CPUs at about 1e-13, and
the gap model (histogram gradient boosting) turns that into a different model. Fitted on the PC's features, the
Mac reproduces the committed model, and thread count makes no difference. Writing `features.csv` with 12
significant digits makes both machines identical; not applied yet (it changes the committed numbers: results-v1.1).
Set-up findings for the README (Anaconda's terms-of-service prompt, pytest collecting `_context/`, timings,
line endings) are listed in context.txt.

## 25 October 2026

**DS10 added: a teammate's independent DS1-style snapshot, used as a held-out test of every model.** On 23 Oct
Le Quang Anh Hoang collected 24 main and 3 short-history addresses (5,785 transactions) from Blockstream Esplora by
DS1's rule, with their own seed and sampled blocks (`data/processed/ds10/teammate/`, unchanged). `explorer/ds10.py`
(steps `ds10`, `ds10-check`; settings `ds10`) checks first that none of its addresses is in DS1, DS1's candidate
pool or DS2 (none is, so no model has seen it, TxGPT included), then refetches the 27 histories with DS1's own
client and checks. Cut at the teammate's last block (972,828), all 5,785 transactions match the teammate's CSV
exactly (same block, received and sent; `ds10_crosscheck.csv`), and every DS1 history check passes. Every model
was then run frozen, as tuned on DS1, with the reporting seeds' plants (750) and address-level bootstrap intervals
(`eval_ds10.csv`, `eval_ds10_check.pdf`):
- TxGPT, tested here for the first time on data it never saw, ranks first: mean recall 0.598 (95% CI 0.534 to
  0.657), against 0.547 on DS1, at a 3.8% flag rate; it catches 93% of dormancy plants and 72% of spikes.
- The baseline is second (0.431, CI 0.334 to 0.528; DS1 0.655, DS2 0.494) and again the only method with no
  address over the sanity limit; the fusion (0.398) and regression (0.345, 3 addresses over the limit) follow;
  the outlier and clustering methods fall to 0.14-0.16.
- Every method finds planted bursts far less often than on DS1 (baseline 0.066 per transaction against 0.381).
  DS10's addresses transact more often and less evenly (median gap 13.1 h against 23.1 h on DS1; the spread of
  their 7-day counts, as an interquartile range, 6 against 4; `ds10_activity.csv`), so ten transactions in a day
  stand out less. The method ranking still agrees with DS2's (Spearman 0.81 over 8 methods) and less with
  DS1's (0.67 over 9), which suggests that DS1-tuned thresholds travel less well than the methods themselves.
- The Elliptic and BitcoinHeist classifiers cannot run on DS10: they need those datasets' own features.
- The explanation writers on 291 DS10 prompts built as DS7's are (`data/processed/ds10/explain.jsonl`: 109 flagged,
  155 not flagged, 27 unscored): the 152M model passes the guardrail on all of them, keeps every number and
  mentions every flagged feature (ROUGE-L 0.678; `assistant_eval_ds10.csv`). Qwen 3B also passes on all 291,
  keeps every number and mentions 99.8% of flagged features (ROUGE-L 0.793; 7.2 s per prompt on the RTX 3060).
  An ad hoc check of comparison words against each feature's z sign, not yet in the guardrail, finds the same
  slips as on DS7: Qwen 4 of 291 answers, the 152M model 1, the template none.
`lm/eval_explain.py` gained `--tag`, so a second test set writes `assistant_eval_<model>_ds10.csv` and
`assistant_eval_ds10.csv` and leaves the DS7 tables alone.

**TxGPT trained and evaluated.** The size sweep (three seeds per size) ties: mean validation loss S 0.675, M 0.677,
L 0.677, each with a standard deviation of about 0.012 (`txgpt_sizes.csv`). With 9.06M training tokens a bigger
model does not help, so the smallest (S, 0.8M parameters) is kept. On the DS1 reporting plants, `txgpt_method`
reaches a mean recall of 0.547 (95% CI 0.442 to 0.638) while flagging only 1.5% of real transactions (34 rows),
against the baseline's 0.655 at 10.4%; it has the best PR-AUC (0.333; baseline 0.109) and the best precision lower
bound (0.42; baseline 0.13), and stability 1.0. The paired bootstrap gives -0.108 against the baseline (p = 0.18).
It is the most precise method but flags the fewest rows, so the baseline stays the explorer's method; the
significance tables and figures were re-run with TxGPT included. TxGPT is not in the sensitivity, walk-forward or
DS2 checks: its scores are precomputed for DS1, and DS2 trained it.

**The 152M Explorer Assistant finished its chat (DS6) and explanation (DS7) stages.** Final validation loss 1.441
after the chat stage (1,526 steps) and 0.071 after the explanation stage (382 steps). On the DS7 test set (445
scored prompts) its explanations keep every number in 99.8% of answers, always carry the limitation line, use no
banned word, mention 99.2% of the flagged features and pass the guardrail 99.8% of the time (one answer falls back
to the template); ROUGE-L against the template text is 0.683 (`assistant_eval.csv`). Its five glossary answers
score ROUGE-L 0.085: as planned, it explains flags and Qwen 3B answers open questions.

**Qwen 3B fine-tuned on Kaggle (T4, fp16 with loss scaling) and evaluated on the same DS7 test set.** All 1,446 steps
(4.64M training tokens, 326 tokens/s); best validation loss 0.0752. Loss on 200 held-out DS6 conversations fell from
0.812 at step 100 to 0.754 at the end, so mixing DS6 into the training data kept its general chat. On the 445 test
prompts it keeps every number, always carries the limitation line, uses no banned word, mentions every flagged
feature and passes the guardrail every time; ROUGE-L 0.783 (152M model: 0.683), glossary ROUGE-L 0.103. The run
used Kaggle's own PyTorch (2.11.0+cu128; the notebook pins transformers, peft, bitsandbytes and accelerate, and
`kaggle_run.json` records every version). The result confirms the provisional writer order: Qwen first, then the
152M model, which is also the writer in the CPU environment, where Qwen cannot run. Both writers now load from
`models/` (git-ignored, over GitHub's 100 MB limit; SHA-256 in context.txt).

## 24 October 2026

**DS1 machine learning extended: calibrated ranges, a fused method, uncertainty, strength, time and DS2 (user,
20:10: the marks weigh ML far above the language models).** Six new steps, each with a settings.json section, a
table written by code and a figure; tests in `tests/test_extensions.py`.
- *Conformal regression ranges* (`regression.conformal`): the HGB quantile ranges promised more than they
  delivered (90% gap range: 0.787 coverage; 98%: 0.894). Split-conformal calibration (CQR) on the last 20% of each
  address's fit period, still time-safe, gives 0.894 and 0.976 (size 98%: 0.973 -> 0.988), and the worst
  address's coverage rises from 0.63 to 0.87 (gap 98%). The calibrated `regression_interval` now flags 3.5% of
  real transactions and is the only method with no address over the 20% sanity limit; its planted recall is
  low (0.268), which is what an honest 98% range buys. One address's amounts change after the cut and no range
  covers it (0% at size 90%, raw or calibrated): reported per address in `regression_results.csv`.
- *Score fusion* (`explorer/fusion.py`, step `fusion`): two methods, one flag, tuned on the tuning plants at the
  explorerMethod's flag rate (equal flag budget, so recall cannot rise just by flagging more) and gated on
  stability like outliers.py's grids. The first, ungated run chose regression + Isolation Forest (tuning recall
  0.681 at an 8.6% flag rate), whose reporting stability, 0.61, then excluded it from selection; the gated rule
  chose baseline (x1.1) + regression (x3.0): stability 0.89, reporting mean recall 0.684 against the baseline's
  0.655 at the same 10.4% flag rate. Isolation Forest, K-Means and LOF add nothing to the baseline at their own
  thresholds (their flags are a subset of its flags). `tuning_fusion.csv` also traces each single method's
  recall-flag rate curve: above about 5% flagged the fused pairs dominate every single method.
- *Significance* (step `significance`): address-level cluster bootstrap (2,000 draws; plants on one address are
  not independent), paired across methods. Fusion vs baseline +0.029 (95% CI -0.001 to +0.055, p = 0.063);
  regression vs baseline +0.018 (-0.073 to +0.114, p = 0.70); every other method is lower with p < 0.001. The
  selection keeps the baseline: the fusion is within the 5-point tie margin and not significantly better, so
  the explainability tie-break decides, now with the evidence that the gap is within the noise. The absolute
  recalls are far less certain than the seed SD suggests (baseline 0.655, CI 0.546-0.753).
- *Sensitivity* (step `sensitivity`): every pattern planted again at several strengths (reporting seeds; the
  default strengths reproduce eval_injection.csv exactly, checked). The regression residual sees 2x spikes
  (0.76) where the baseline needs about 10x (0.74); the baseline never sees bursts well (0.30-0.42 at 3-20
  transactions) while the residual does (0.69-0.76); and the baseline sees dormancy from 14-30 days (0.79)
  while the residual needs 120+ days. This is the mechanism behind the fusion's gain.
- *Walk-forward* (step `walk-forward`): the split moved to 50, 60, 70 and 80% of each history, every method
  refitted with its saved configuration (no re-tuning), plants re-drawn in each evaluation period. The ranking
  holds at every cut-off (fusion, regression and baseline at 0.65-0.71 mean recall; K-Means, Isolation Forest
  and DBSCAN 0.39-0.45; LOF 0.32-0.33; the calibrated range 0.21-0.28), the baseline is the steadiest
  (0.650-0.658), and the regression residual breaks the sanity rule on 10-14 addresses at every cut-off, so its
  failure is structural rather than an accident of the 70% cut. At 70% the refits reproduce eval_injection.csv.
- *DS2 check* (step `ds2-check`): the frozen DS1 methods on the 2,933 DS2 histories that meet DS1's own
  main-address rule (827,291 transactions; 15,000 plants over the five reporting seeds). Every method recalls
  less than on DS1, so DS1's absolute recalls were optimistic for a broader population, but the ranking holds
  (Spearman 0.905 across the 8 methods) with one telling swap: the baseline, which fits nothing, ranks FIRST on
  DS2 (0.494, 95% CI 0.479-0.508; DS1 0.655), ahead of the fusion (0.471; DS1 0.684) and the regression residual
  (0.425; DS1 0.673), whose models were fitted on 25 addresses. The parameter-free method transfers best: the
  strongest evidence yet for the baseline as the Explorer's method. The sanity rule is milder on DS2 (10% of
  addresses over the limit for the baseline, 16% on DS1), and the conformal margin calibrated on DS1 does not
  carry over (DS2 flag rate 7.0% against 3.5%). `inject.make_plants` now splits the table by address once
  (identical plants, checked against the committed specs; a DS2 seed went from about 6 minutes to under 1).

**The Qwen 3B fine-tune moves to Kaggle; the GPU queue runs it last, as the fallback (user, 20:20).** The marks
weigh the machine-learning work far above the language models (the tutor called the latter an optional extra
for HD), so the home GPU now trains TxGPT and the 152M stages first while a Kaggle GPU fine-tunes Qwen in
parallel. `kaggle/qwen3b_explain.ipynb` clones the repository at a pinned commit, installs the pins of
`environment-gpu.yml`, rebuilds the DS6 chat subset and asserts that all six data files are byte-identical to
the local ones (checked on a fresh clone), then benchmarks, trains and evaluates; `kaggle_run.json` records
the GPU and library versions for the report. Kaggle's T4 and P100 have no native bf16, so `lm.qlora` computes
in fp16 with loss scaling on such GPUs (bf16 stays on the RTX 3060; `meta.json` records which), and
`--max-hours` stops cleanly inside Kaggle's 12-hour sessions. `lm.train` and `lm.writers` now choose bf16 by
compute capability too: `torch.cuda.is_bf16_supported()` answers yes on a T4 because it can emulate bf16,
several times slower. If the Kaggle run arrives first, copying its run folder into `runs/qwen3b-explain/`
turns the queue's Qwen steps into no-ops (a run already at its last step trains nothing and leaves its
`meta.json` alone).

**Power cut during pretraining; recovered with nothing lost but ten minutes of training.** At about 19:16 (GMT+7)
the PC lost power with the 152M pretraining at step 7,447 of 7,630 (the last console line is at 19:14:31; the PC
rebooted at 19:17). The resume design did its job: `ckpt_last.pt` (step 7,419, saved at 19:05:57, 30-minute
interval) and `model_best.pt` (step 7,400, validation loss 3.133) both loaded with every tensor finite and the
optimizer state intact, and the same command resumed at step 7,419 at 19:43. The cost was 28 optimizer steps
(7.3M tokens, about 10 minutes of GPU time) plus 27 minutes with the GPU idle. Every file written since 17:30 (85
files, among them DS2's 898,490 rows, the DS7 splits, the TxGPT and DS7 token files, the tables and the figures)
was parsed back without error, `git fsck` found no corrupt objects, and the GPU queue's state (4 of 25 steps done)
was intact, so the queue continued with the same command too. Two weaknesses were fixed the same evening, before
the Qwen run: (1) a rename can reach the disk before the file's data does, so after a power cut an "atomic"
checkpoint can still be empty; `lm.train.save_atomic` and the queue's state file now fsync before renaming;
(2) `lm.qlora` overwrote its adapter folders in place and kept the adapter weights apart from `state.pt`, so a cut
mid-save could have broken the resume of a 6-hour run. Adapters are now written to a scratch folder and moved over
the old files one at a time (`save_adapter`), and `state.pt` carries its own copy of the adapter weights, so a
resume always pairs the weights with the right step (checked with the CPU smoke run; new test in
`tests/test_assistant.py`). A UPS would remove the risk; restoring power does not restart the jobs by itself.

**Fine-tuning a public model is acceptable (CR-06 decision 4 relaxed).** The tutor, Arthur Hoang, announced by
Outlook that fine-tuning is acceptable, so training from scratch is no longer an HD requirement. The 152M model is
kept (pretraining is nearly done, and it is the only language model that runs in the markers' CPU environment), but
it gets no GPU time beyond the planned stages. The explanation writers will be ordered by their measured DS7
results instead of "from scratch first"; the remaining effort goes to integration (`predict.py`), reproducibility
and the report.

**`tools/run_queue.py` (new) runs everything after pretraining unattended.** `configs/queue_after_pretraining.json`
lists the 25 steps (DS2-dependent data builds, TxGPT's 9 runs and scoring, the DS1 evaluation, the 152M chat and
explanation stages, Qwen, every explanation evaluation). Each step waits for the files it needs from outside the
queue and for a free GPU, runs under `tools/supervise.py` (new `--max-restarts`, so one failing step cannot hold up
the rest), and must leave its output files behind; finished steps are skipped on a restart. It also re-runs the
template evaluation, since DS7's test split changes once DS2 is in. `tools/status.py` shows the queue.
At 18:05 the Qwen steps (bench, training, evaluation) were moved ahead of TxGPT and the 152M stages, against
CLAUDE.md's "runs after the 152M model and TxGPT": with fine-tuning allowed Qwen is the main writer and the chatbot,
and it is the one job never run on this GPU, so its bench and first validation now happen while someone is awake.
Nothing else changes; every step still runs tonight.

**The guarded Assistant is wired into `predict.py`** (plan §12.1's `explain(tx_result, use_assistant=False)`, which
had not been written). `Predictor.explain` tries the writers in `settings.json -> assistant.writerOrder` and shows the
first answer that passes every guardrail check, else the template; a rejected answer is never returned, only the
checks it failed. Writers this machine cannot run (no GPU or no `transformers` for Qwen in the markers' CPU
environment; a model file not downloaded) are skipped with the reason. The address and what-if results keep the
instant template text; the Assistant runs per transaction (`predict explain --address A --txid T --assistant`).
The writer code moved to `lm/writers.py`, so the DS7 evaluation and the Explorer decode identically. The writer
order is provisional (Qwen, then 152M) until the DS7 results.

**Address profiles and DS2's coverage of DS1 (plan §9.4 S-7, §7; `python -m explorer.cli profiles`).** Eight
address-level features for the 25 DS1 main addresses and the 4,012 DS2 addresses, K-Means over k = 3-8 by silhouette.
The silhouette is flat and low (0.18-0.19 for every k; best k = 8, the top of the plan's grid), so the profiles
divide a continuum rather than separate distinct types; the report says so and uses them descriptively. Profiles
are named by their most distinctive features (`profiles.describeAbsZ` 0.75, not clustering's 1.0, which left
three of eight unnamed). Coverage: every DS1 address lies inside DS2's 1-99% range on 6 of 8 features (92% on
burstiness, 96% on outputs per transaction); DS1's median fee rate sits at DS2's 8th percentile, the one clear
difference. New: `explorer/profiles.py`, `reports/tables/{profiles,profiles_tuning,ds2_coverage}.csv`,
`reports/figures/eda_{ds2_coverage,profiles}.pdf`, `models/profiles.joblib`, `data/processed/profiles.csv`.

**The DS2 crawl's final step crashed writing its manifest** (16:36): the first crawl run (22 Oct, an earlier client)
recorded its request count but `None` for its error counts, and the totals added them. `ds2.parquet` had been written,
the manifest had not. `explorer/corpus.py` now counts those as 0 and records how many runs had no error counts
(`runs_without_error_counts`, 1), with a test; the manifest was rebuilt without re-crawling (4,012 addresses, 898,490
rows, every check passed).

Pre-flight check of the next training stages (TxGPT, 152M chat and explanation, Qwen), before the GPU frees up.

- **TxGPT read every planted transaction as "same block".** `explorer/inject.py` gives a planted row the previous
  real row's block height at a later time, and the tokenizer decided "same block" by height alone: all spike and
  dormant plants and 445 of 500 burst plants (tuning seed 40000) got the SAME_BLOCK gap token instead of their real
  gap, so TxGPT could not have seen a dormancy. Now "same block" needs the same height and the same time. Real data
  is unchanged (in DS1 a shared height always means a shared time), so DS2's tokens, the baseline and every reported
  number stay the same.
- **TxGPT scoring now keeps same-block transactions out of each other's history (DR-02).** `lm.tx_score` cuts each
  context at strictly earlier block times and measures the scored transaction's gap from the last strictly earlier
  one; before, a second transaction in a block was scored with its sibling in view (3.6% of DS1 rows share a block).
  `lm.tx_baselines` keeps the plain sequence NLL, as its count-based baselines also see every earlier row.
- **`lm.tx_select` (new)**: the plan's size rule (lowest mean validation loss over 3 seeds), the runs and sizes tables
  (`reports/tables/txgpt_runs.csv`, `txgpt_sizes.csv`) and the fp16 export `models/txgpt.pt` + `models/txgpt_meta.json`,
  which `lm.tx_score` now uses by default. Nothing did these before.
- **Chat data packs whole conversations into windows.** `lm.prepare_chat` wrote conversations back to back and the
  trainer cuts fixed 1,024-token windows, so a conversation could straddle two windows and the second part's reply
  was learned without its question. For DS7 that would teach the 152M model to write numbers it cannot see, against
  the number-faithfulness guardrail. Conversations are now packed whole (best fit, longest first) and the remainder
  of each window is filled with untrained `<|endoftext|>` tokens; `data/chat` is rebuilt with it (24 Oct). `--val-jsonl` makes the
  explanation stage validate on DS7's own `val.jsonl` (the split Qwen uses), not every 100th training pair, and a
  trailing filler token stops the last window of each file being skipped.
- **`configs/text_150m_explain.json` (new)**: stage 3 as plan §9 states it (from the chat model; DS7, 3 passes;
  lr 1e-4, warm-up 20; DS7 validation), with early stopping after 4 evaluations without improvement.
- **152M fine-tunes use micro-batch 4, not 8.** The pretraining bench measured 11.7 GB of 12.9 at 8 (only 1.5%
  faster than 4 at 7.0 GB); with the desktop also on the GPU, Windows would spill to system RAM and run several
  times slower. Same batch per optimizer step (more accumulation); validation batches doubled to keep its size.
- **`lm.eval_explain --model text150m` would have crashed**: it kept `load_model`'s (model, tokenizer) pair as the
  model. Fixed, with a test.

## 23 October 2026

- **DS7 glossary drafted** (`tools/make_glossary.py` -> `data/processed/ds7/glossary.jsonl`): 160 question-answer
  pairs on 61 topics, drafted by Claude at Cao Tri Tran's request (the team delegated it; the plan aimed for about
  300 team-written pairs). Every number is read from settings.json and models/baseline.json when the script runs;
  answers are checked for banned words and the limitation line. Marked 'draft for review'. DS7 now splits glossary
  pairs by topic (paraphrases never cross splits), and lm.eval_explain scores glossary rows separately (wording,
  ROUGE-L), since a correct glossary answer may contain numbers its question does not.
- **torch.compile not tried** (decided 23 Oct): it could shorten pretraining, but the GPU queue already finishes
  three days before M2 and DS2 is not ready earlier, so the gain does not move any deadline.
- **Team decisions (23 Oct).** The team delegated every open A2 decision to Cao Tri Tran, who recorded:
  (1) CR-03: the baseline stays at |z| >= 5 and the sanity-rule failure is reported as a finding about
  behaviour change (`eval_sanity_addresses.csv`), not hidden by a cap; (2) the Assistant 3B change to
  Qwen2.5-3B-Instruct and the model roles are accepted, closing the CR-06 amendment; (3) `explorerMethod`
  is the robust-z baseline (`settings.json`), as the A1 selection rule recommends. Not delegable: the blind
  reviewer check (reviewers must not have built the scoring) and the signed contribution form.
- **CR-03 evidence: the sanity rule fails for every method, because a few addresses changed.**
  `reports/tables/eval_sanity_addresses.csv` (new, from `evaluate`) gives each method's flagged share per
  address and how the address's evaluation period differs from its fit period. One address is over the
  20% limit for all 7 methods and another for 6: one became about 4x busier after the 70/30 cut, one is an
  almost perfectly regular daily receiver whose rare sends stand out, and one turned from mostly receiving
  to mostly sending. Raising the baseline threshold cannot fix this. Options for the team: (a) keep
  |z| >= 5 and report these addresses as a finding about behaviour change (recommended: it is what the
  Explorer is for); (b) for A3, show a 'behaviour changed' notice once an address's flags cluster, instead
  of flagging each transaction; (c) cap flags per address (meets the rule by construction; not recommended).
  Re-running `evaluate` reproduced every existing table exactly.
- **DS2 target 4,000 addresses** (was 5,000; user's decision, 23 Oct 09:00). Above the plan's 3,000
  minimum; at the measured ~1.6 kept per minute it is reached about 18:00 on 24 Oct, before pretraining
  ends, so the GPU never waits for DS2. The crawl stops itself at 4,000 and writes ds2.parquet.
- **DS2 crawl crash and fix.** At about 07:50 mempool.space closed a connection without replying; the
  client retried only `URLError` and timeouts, so `http.client.RemoteDisconnected` escaped, and because
  `explorer.corpus` records only `EsploraError` as a failed visit, the whole crawl stopped (774 kept,
  nothing lost; about 45 minutes of crawling lost). `explorer/esplora.py` now retries `ConnectionError`
  and `http.client.HTTPException` like a timeout (tests reproduce the drop). `tools/supervise.py` now
  runs the crawl and restarts it after any crash (not after a finish or a Ctrl+C), so an unattended run
  cannot stop for good. Measured overnight rate: about 1.6 kept addresses per minute.
- **Proof-check pass (user's request; 23 Oct 02:20-02:35).** Fixed: (1) the 152M model's greedy
  decoding in `lm.eval_explain` used lm.chat's repetition penalty, which also counts the prompt's tokens
  and so pushes the model away from copying the prompt's numbers; it is now 1.0 for explanations.
  (2) Two DS7 phrasings said "neither is unusual enough to flag" when no single feature stands out, which
  contradicts a flag from a pattern-level method; the intros are now neutral (a regression test fails on
  the old wording). (3) `lm.qlora`'s ETA was wrong after a resume. Added: `lm.qlora --bench` (longest
  batches first, so an out-of-memory error shows in minutes) and a retention loss on 200 held-out DS6
  conversations (`chat_retention.jsonl`), evidence that DS7 fine-tuning did not degrade general chat.
  Three notebook notes corrected: the baseline is chosen because regression fails the stability test
  (0.61 < 0.7), not by a tie-break; regression is the exception to "bursts are hardest"; TxGPT's gap
  buckets are fixed in lm/tx_tokens.py, not set from the quantiles.
- **Training speed checked.** The software is near the practical limit (flash attention, fused AdamW,
  bf16, TF32; ~49% MFU). The GPU is thermally throttled (about 1,870 of 2,130 MHz at 85 C), so better
  cooling is the main lever. torch.compile is possible after all (triton-windows 3.3.1 matches torch
  2.7.1), contrary to CLAUDE.md's "no Triton on Windows"; not tried, since testing it pauses pretraining.
  The GPU queue after pretraining is about 13 h, so TxGPT waits for pretraining instead of pausing it.
- **Figures and the EDA notebook.** `python -m explorer.cli figures` writes 17 figures to
  `reports/figures/` and their statistics to `reports/tables/eda_tests.csv`; `notebooks/01_eda.ipynb`
  calls the same functions. Findings that differ from the plan's hypotheses (plan §7 says to report
  them): in and out amounts differ significantly for only 9 of 23 DS1 addresses, weaker support for
  CR-04 than expected (the ablation measures its effect); BitcoinHeist's features barely differ by
  label (all rank-biserial |r| <= 0.13), which explains its weak classification; on held-out families,
  logistic regression's high recall comes with a ~49% false-positive rate, so it is not a
  counter-example to "models learn families, not behaviour". scipy is now pinned (the figures use it).
- **Assistant 3B: Qwen2.5-3B-Instruct, one mixed fine-tune** (user's decision, later on 23 Oct;
  replaces the base-model plan below). Revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, same
  Qwen Research License. Reason: the Assistant is also to serve as a chatbot next to the
  transaction checks, and a chat-tuned start is far more reliable than any chat stage we can afford
  (making the base model a dependable chatbot would need roughly 30-60M tokens, 17-35 h on the
  3060). One stage (`configs/qwen3b_explain.json`): DS7 plus 2,500 DS6 conversations (to keep its
  general chat from degrading), 2 passes, about 10.5M tokens, estimated 5-6 h. `qwen3b_chat.json`
  is removed. The report's comparison becomes "trained from scratch vs fine-tuned from a
  chat-tuned pretrained model", and says so.
- **Assistant roles (user's design):** the 152M model explains DS1 transactions only; Qwen 3B is
  the chatbot and the second writer. `explorer.assistant.explain` tries 152M, then Qwen, then the
  template; every answer passes the same code checks, which stay deterministic.
- **Assistant 3B base model: Qwen2.5-3B instead of Llama 3.2 3B** (user's decision). Qwen is not
  gated, so no Hugging Face token is needed; the plan already named it as an option. Base model
  `Qwen/Qwen2.5-3B` at revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`. Licence: Qwen Research
  License (non-commercial, research or evaluation use; derivatives must ship the licence text, a
  NOTICE line "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba
  Cloud. All Rights Reserved.", and "Built with Qwen" in the documentation). **The team approved
  Llama specifically, so it must be told.** (Accepted later on 23 Oct; see the team decisions above.) The base (not Instruct) model is used so that its two
  stages (DS6 chat, then DS7) mirror the 152M model's.
- **Fine-tuning pipeline ready** (runs after pretraining and TxGPT): `explorer.cli build-ds7`
  (DS7 pairs, five phrasings, split by address; DS1 only until DS2 exists), `explorer/assistant.py`
  (prompt and guardrail), `lm.chat_subset` (DS6 subset as messages), `lm.qlora` (4-bit NF4, LoRA
  r=16 on all projections, loss on replies only, resumable), `lm.eval_explain` (the plan's measures
  for template, 152M and 3B). Smoke-tested on the CPU with a tiny random model of the same
  architecture. GPU environment gained transformers 4.52.4, peft 0.15.2, bitsandbytes 0.48.1 and
  accelerate 1.7.0, chosen so that no installed package changed under the running pretraining.
- **DS2 crawl: 8 workers instead of 4.** Replies take a median 4.3 s (mempool.space throttles large
  pages), so 4 workers reached only about 1 request/s; 8 stay under the 2 request starts/s the
  pause allows. The data collected is unchanged. **Measured: no faster.** Over 01:15-01:36 the
  crawl visited 4.9 candidates and kept 2.1 addresses per minute, the same as with 4 workers, so
  the limit is mempool.space's bandwidth per IP, not the worker count. 8 is kept (it does no harm).
  At this rate 3,000 kept is reached about 23:30 on 23 Oct and 5,000 about 15:30 on 24 Oct.

- **DS1 pipeline run end to end** (transactions to evaluate). The selection rule recommends the
  **robust-z baseline** (runner-up K-Means); `explorerMethod` stays unset until the team records it.
- **The baseline flags far more than A1 expected.** At A1's |z| >= 3.5 it flags about 15% of
  evaluation transactions (the plan expected about 2%), and no threshold in the sweep (2.5-5.0)
  meets the 20%-per-address sanity rule, so the sweep fell back to |z| >= 5.0, the fewest addresses
  over the limit (`reports/tables/tuning_baseline.csv`). This is CR-03 for the team: keep 5.0, or
  accept that the sanity rule fails for a few addresses with very regular histories.
- **Added the label check** (`explorer.cli label-check`): Isolation Forest, LOF, K-Means and DBSCAN
  fitted without labels on a 10,000-row training sample of DS3 and DS4, scored against their labels.
  One StandardScaler fitted on that sample feeds all four (DBSCAN's nearest-core scoring cannot sit
  in a Pipeline); it never sees test rows.
- **Added the reviewer sheet and score** (`reviewer-sheet`, `reviewer-score`). The key that says
  which rows were flagged stays in `data/interim/` (git-ignored), because the repository is public.

## 22 October 2026

- **Added a fine-tuned 3B Explorer Assistant (plan D-A2-4, §9).** The tutor, Arthur Hoang, agreed
  to an extra model: a pretrained base model of about 3B parameters, QLoRA-fine-tuned on DS6 and
  DS7 after the 152M model and TxGPT finish. It sits alongside the from-scratch models and
  doesn't replace them; the report compares the two Assistants. The tutor's agreement is
  confirmed and the team approved the change as an amendment to CR-06. Base model chosen:
  Llama 3.2 3B (Llama 3.2 Community License; "Built with Llama" attribution; adapters ship only).
- **DS1 snapshot complete** (every automated check passed); `histories.json.gz` and the manifest
  are committed. **DS3 Elliptic classification done**: metrics tables only (no rows or models).

- **Blockstream rate cap.** Since July 2025 Blockstream allows 700 unauthenticated requests per hour
  per IP (and 500,000 per month). DS1 now fails over to mempool.space for an hour after repeated
  429s and records which host served each address; the cross-check always uses a host that did not
  serve the history. DS2 crawls mempool.space only (`settings.json -> corpus.sources`).
- **gzip.** mempool.space throttles large replies (~235 KB/s); requests now ask for gzip, several
  times faster for consolidation-heavy pages. Replies are decompressed before caching (unchanged).
- **DS2 excludes DS1's whole candidate pool** (a superset of DS1), so it could start before DS1
  finished; the leakage guarantee is stronger than the plan's.
- **DS3 Elliptic licence: CC BY-NC-ND 4.0.** Processed Elliptic rows and Elliptic-trained models are
  git-ignored and never zipped; only the download record, code, metrics and figures ship (plan §6.3).
- **DS4 decisions by the written rules:** temporal split trains to 2016 (the last year leaving
  >= 1,000 test positives); leave-out families CryptoWall, CryptoLocker, Cerber, Locky, CryptXXX;
  log transforms for weight, count, looped, neighbors (skewness > 1) and income; no exact duplicates.
- **Random forest bootstrap capped** at 100,000 rows per tree (`classification.rfMaxSamples`) to
  bound memory on DS4; the plan's grid otherwise unchanged.
- **F7 is log10(1 + fee rate)** so zero-fee coinbase transactions get 0 instead of minus infinity.
- **lm.train gained `--seed` and `early_stop_evals`** (the hand-off noted `--seed` was missing).
- **Interrupted runs.** Two forced stops lost in-memory request statistics; their records in
  `runs.jsonl` are reconstructed lower bounds from the cache and are labelled as such.

- **Moved to the Windows PC.** Cloned from GitHub into `C:\Users\ADMIN\COS30049A2` (short path,
  outside OneDrive). The hand-off's step 1 (create a GitHub repository) was already done: the
  remote is the public `Defaul119/COS30049A2`. Installed Miniforge 26.7.2-1 (conda) for the user
  at `C:\Users\ADMIN\miniforge3`, since the PC had no conda; Anaconda Prompt is replaced by
  "Miniforge Prompt". One day behind the plan's schedule (day 1 was 21 Oct).
- **settings.json → collection:** added the DS1 selection parameters that plan §6.1 gives in
  words, so the code holds no magic numbers: `snapshotBlockCount` 20,
  `snapshotOldestBlockBack` 4000, `snapshotNewestBlockBack` 100, `snapshotTxPagesPerBlock` 4,
  `historyFetchAttempts` 2 (re-fetch once if a block lands mid-fetch, as `scripts/d1_check.py`
  does), `crossCheckAddresses` 2.
- **DS1 snapshot (`explorer/snapshot.py`, `python -m explorer.cli snapshot`).** Implementation
  details the plan leaves open: the candidate pool is sorted before the seeded shuffle, so the
  visit order does not depend on the order blocks were read; a fetched candidate is kept only if
  every automated check passes (otherwise skipped as `checks_failed`); histories fetched but not
  kept move to `data/raw/esplora/skipped/` (git-ignored), so `data/raw/esplora/<address>/` holds
  exactly the DS1 addresses; every reply, including each candidate's `/address` reply and the
  sampled blocks, is cached, so a re-run is offline and an interrupted run resumes.
- **settings.json → external** (new section): source page, download URL, DOI, citation, licence
  and expected files for DS3 (Elliptic) and DS4 (BitcoinHeist), read by
  `python -m explorer.cli external`. The Elliptic licence is left empty until it is read on the
  Kaggle page on the day of download (plan §6.3).
- **DS5 preparation is much faster than planned.** `lm.prepare_text` produced 265M training tokens
  in about 3 minutes on this PC's connection, so the 2B-token corpus takes about 25 minutes, not
  the 1.5–3 h the plan estimated. No decision changes. Final corpus: 2,000,417,414 training and
  10,001,319 validation tokens from 1,938,432 documents (`data/text/meta.json`).
- **Working pace.** The user asked for the work to finish as fast as possible at the highest
  quality rather than follow the plan's calendar, so each step now starts as soon as the one it
  depends on is verified. The milestones' content is unchanged.
- **DS1 raw pages are too large to ship.** Consolidation-heavy addresses have history pages of
  about 3 MB each gzipped, so the raw cache is about 0.4 GB, not the plan's ~30 MB (§12.4). The raw
  replies stay on the collecting PC (git-ignored, SHA-256 of each file in the manifest). The
  snapshot also writes `data/processed/esplora/histories.json.gz`, each transaction trimmed to the
  fields the pipeline uses; amounts recompute exactly from it (tested), and it is what the
  repository and the zip carry.
- **DS2 crawl uses both public hosts.** Blockstream and mempool.space each serve three workers,
  with a shared pacer per host so each still sees at most one request start every 0.5 s (the same
  ceiling as DS1). The plan used mempool.space only as a fallback; using both cuts the crawl from
  an estimated 7–14 h to a few hours.
- **Pretraining gate (plan §5.3).** The first GPU benchmark exposed a bug in `lm.train --bench`:
  after one size ran out of memory, the exception kept that step's tensors alive, so every smaller
  size also failed or spilled into system RAM (1,483 tokens/s). Fixed. Measured on the RTX 3060:
  micro-batch 8 at 12,152 tokens/s (11.7 GB, too close to the 12.9 GB card), **micro-batch 4 at
  11,967 tokens/s (7.0 GB)**, micro-batch 2 at 11,310 tokens/s. 2B tokens at micro-batch 4 take
  about 46 h, under the 72 h gate, so `total_tokens` stays at 2,000,000,000 and `micro_batch` is
  set to 4.
