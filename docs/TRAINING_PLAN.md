# Training stages and strict testing plan (from-scratch models only)

Written 27 Oct 2026 at Cao Tri Tran's request, on the `from-scratch` branch, after the tutor ruled that only
models the team trains from scratch count. Goal: the highest accuracy each model can reach before the results
freeze (Tue 3 Nov), with a strict test after **every** training stage and a comparison of every stage against the
milestone below. Numbers here are copied from `reports/tables/` for planning only; the report quotes the tables.

## 1. Milestone M0 (git tag `m0`): where every model stands now

| Model | Metric | Dev / DS1 | DS10 (unseen) | Source |
|---|---|---|---|---|
| Baseline (robust z, in use) | mean planted recall, 100 seeds | 0.661 | 0.449 | `stats_methods_summary.csv` |
| Fusion (baseline + regression) | same | 0.685 | 0.416 | same |
| Regression | same | 0.673 | 0.333 | same |
| TxGPT S (from scratch, 0.8M) | mean planted recall | 0.547 | 0.598 | `eval_significance.csv`, `eval_ds10.csv` |
| Elliptic, random forest | F1 (temporal test), 10 seeds | 0.815 [0.796, 0.835] | | `stats_classifiers_summary.csv` |
| BitcoinHeist, random forest | F1 (temporal / random) | 0.070 / 0.246 | | same |
| 152M explanation writer (round 1) | fully correct, pooled over 12 sets | 72.7% [71.6, 73.8] | 83.2% | `stats_writers_summary.csv` |
| 152M pretraining | validation loss (160 fixed windows) | 3.1305 at 2.0B tokens | | `runs/text-150m/` |

Known weak spots at M0: burst transactions for every scoring method (TxGPT 0.17 on DS1, baseline 0.066 on DS10);
regression on dormant addresses; the 152M writer on the hard (43.2%) and max_shift (18.7%) sets; Elliptic after
step 43 (real-world drift); BitcoinHeist across time (near chance: a data limit, not chased).

## 2. Three tiers of data (never tune on the test)

| Tier | Scoring methods and TxGPT | 152M writer | Classifiers | Used |
|---|---|---|---|---|
| **Dev** | DS1 with tuning seeds 40000-40004; DS2's 201 validation addresses with tuning plants | DS7 v2 validation split + a new held-out slice of the weak-case examples | Elliptic validation steps (inside 1-34) | after every stage, as often as needed |
| **Test** | DS1 with reporting seeds 30049-30053; DS2's 200 test addresses | DS7 v1 test, the ten randomized sets, the glossary test | Elliptic steps 35-49; DS4 test splits | once per frozen candidate |
| **Final** | DS10 | DS10 explanations | | once, at the end |

Before every evaluation the test and final files are checked against their SHA-256 in the manifests; a mismatch
stops the run. Every result row records the git commit, the stage name and the seeds.

## 3. The gate after every stage

1. **Run the dev evaluation** with the same measures as M0 (no measure changes mid-campaign).
2. **Compare with the previous stage and with M0, paired:** the same prompts / plants, bootstrap 95% interval of
   the difference, McNemar's exact test for the writer.
3. **Accept the stage only if all hold** (fixed now, before any result):
   - the primary metric rises and the 95% interval of the difference is above 0;
   - no pattern / test case falls by more than its own seed spread (one standard deviation at M0);
   - the guardrail pass rate and number faithfulness do not fall (writer); the flag rate stays inside the 20%
     sanity limit (scoring methods);
   - the stage's training loss and dev metric agree (no gain on dev with a rising validation loss).
4. **Otherwise revert** to the last accepted stage and record the failure (it is a result too).
5. Append one row to `reports/tables/stages.csv` (written by code): stage, model, commit, dev metric, interval,
   difference to the previous stage and to M0, accepted or not.

Primary metrics: scoring methods and TxGPT, mean planted recall at the method's flag budget; 152M writer, fully
correct; classifiers, F1 (with accuracy, precision and recall reported beside it).

## 4. Training stages

### A. TxGPT and the scoring methods (CPU or Kaggle CPU; minutes per stage)

| Stage | What | Weak spot it targets |
|---|---|---|
| A1 | TxGPT added as a fusion candidate (baseline + TxGPT) | DS1-strong vs unseen-strong methods |
| A2 | Event-level burst scoring: surprisal summed over a short window | bursts (0.17) |
| A3 | Burst feature relative to the address's own pace (gap / its median gap) | bursts on busy addresses (DS10) |
| A4 | TxGPT time tokens relative to the address's median gap; retrain S, M, L x 3 seeds | bursts, generalisation |
| A5 | More DS2 histories (crawl in the background), then sizes x data fractions (scaling study) | data-limited TxGPT |
| A6 | Elliptic decision threshold set on validation steps | classifier F1 |

### B. The 152M model (Kaggle GPU)

| Stage | What | Gate data |
|---|---|---|
| B1 | Continued pretraining, one Kaggle session at a time, to ~7.5B tokens by Sun 1 Nov 23:59 | validation loss on the same 160 windows after **every session**; it must fall |
| B2 | Chat stage on DS6 (only if the tutor accepts DS6) | chat validation loss |
| B3 | Explanation stage on DS7 v2 + weak-case examples (hard, max_shift, nothing stands out) built from training-split addresses with new seeds | dev fully correct, by case |
| B4 | Short hard-case stage (weak cases mixed with ordinary ones) | dev, by case; no case may fall |
| B5 | Settings sweep (learning rate, epochs, extra-data share) and 3-5 seeds; pick on dev | dev mean and seed spread |
| B6 | Best-of-k at answer time: the first of k answers that passes the full code checks | dev, no training |

B3-B5 are first tried on the round-1 model (one GPU hour each, the user's account) while B1 runs, so the recipe is
known before the round-2 model exists; the winning recipe then runs once on the round-2 model (Mon 2 Nov).

## 5. When testing runs

| When | What runs | Tier |
|---|---|---|
| After every stage above | its gate (section 3) | Dev |
| After every 152M pretraining session | validation loss on the fixed windows | Dev |
| Fri 30 Oct (scoring methods, TxGPT, classifiers frozen) | every accepted candidate once | Test |
| Mon 2 Nov (152M round 2 trained) | the writer once, against round 1 by the adoption rule | Test |
| Tue 3 Nov (freeze) | everything once on DS10; statistics, figures, cards, PDF rebuilt; tag `results-v1.1` | Final |

## 6. The comparison the report shows

One table per model family from `stages.csv`: M0, then each accepted stage, with dev and test results, the
difference to M0 with its interval, and the stages that were rejected. That table is the evidence for the
model-selection section: every choice made by a rule fixed in advance, on data the final numbers never touched.
