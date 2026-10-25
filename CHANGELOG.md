# Change log

Changes to settings, data, decisions and the plan, newest first. Where the plan and reality
disagree, the measured reality is recorded here (CLAUDE.md, "How to work here").

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
