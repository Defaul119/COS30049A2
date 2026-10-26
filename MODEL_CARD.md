# Model card

Every final model in this project: what it is for, what it was trained on, its parameters, how well it does,
its limits and its intended use, in the spirit of Mitchell et al. (2019). The metric tables are written by
`python tools/cards.py` from `reports/tables/`; the rest is by hand.

**Intended use, for every model here.** Help a person decide which of a Bitcoin address's transactions are worth
a look, by ranking each against that address's own past and saying why. Not intended to judge people, to decide
anything automatically, or to label an address. *Unusual activity requires investigation; it does not indicate
wrongdoing.*

**Evaluation without labels.** DS1, DS2 and DS10 have no ground truth, so the scoring methods are measured on
transactions we plant (spikes, bursts, dormant-then-active), with tuning seeds 40000-40004 and reporting seeds
30049-30053. Recall on plants is a real measure; precision is only a lower bound (a real flag that is not a
plant may still be worth a look). Accuracy, precision, recall and F1 are reported only where a dataset has
labels: DS3 and DS4, against the datasets' own labels.

## 1. Scoring methods (DS1, unsupervised)

All of them fit on each DS1 address's earlier 70% and are evaluated on the rest (DR-02). The pooled models take
six inputs per transaction: the four own-history z-scores (F1 size, F2 burst, F3 gap, F4 relative size) clipped to
±`features.zClip`, and the direction. Every threshold, grid and seed is in `config/settings.json`; every search is
in `reports/tables/tuning_*.csv`.

| Model | File | What it is | Chosen setting |
|---|---|---|---|
| Robust-z baseline (**used by the Explorer**) | `models/baseline.json` | A1's method: max \|z\| over F1-F4 against the address's own median and MAD | flag at \|z\| ≥ 5.0 (A1 had 3.5; no threshold met the sanity rule, 5.0 put the fewest addresses over the limit) |
| Isolation Forest | `models/isolation_forest.joblib` | random isolation trees on the pooled inputs | 500 trees, 1,024 samples, 75% of features |
| LOF (novelty mode) | `models/lof.joblib` | local density against the fit period | 50 neighbours |
| K-Means | `models/kmeans.joblib` | distance to the nearest centre | k = 2 |
| DBSCAN | `models/dbscan.joblib` | distance to the nearest core point | see section 2 |
| Regression residual | `models/regression.joblib` | predicts each transaction's size (random forest) and gap (histogram gradient boosting) from earlier history; scores the miss | section 3 |
| Conformal regression range | `models/regression_interval.joblib` | quantile models (1% and 99%) widened by conformal calibration; scores how far outside the range | section 3 |
| Fusion | `models/fusion.joblib` | baseline and regression scores, scaled and combined at the baseline's flag budget | multipliers 1.1 and 3.0 |
| TxGPT surprisal | `models/txgpt_method.joblib` | TxGPT's surprisal of each transaction given its history (section 5) | flag at 12.2 nats |

**Selection** (`models/selection.json`): the highest mean planted-pattern recall among stable methods, and within
five points, the easier one to explain. That picked the baseline, with fusion as runner-up.

<!-- generated:methods -->
| Method | DS1 mean recall (5 reporting seeds) | DS1, 100 seeds: mean (sd) | DS1 precision, lower bound | DS2 mean recall [95% CI] | DS10 mean recall [95% CI] |
|---|---|---|---|---|---|
| The robust-z baseline | 0.655 | 0.661 (0.035) | 0.125 | 0.494 [0.479, 0.508] | 0.431 [0.334, 0.528] |
| DBSCAN | 0.406 | 0.415 (0.034) | 0.140 | 0.265 [0.253, 0.277] | 0.138 [0.070, 0.232] |
| Fusion | 0.684 | 0.685 (0.034) | 0.141 | 0.471 [0.456, 0.486] | 0.398 [0.311, 0.489] |
| Isolation Forest | 0.429 | 0.452 (0.036) | 0.168 | 0.305 [0.293, 0.317] | 0.151 [0.085, 0.226] |
| K-Means distance | 0.446 | 0.463 (0.034) | 0.127 | 0.333 [0.320, 0.346] | 0.157 [0.091, 0.239] |
| LOF | 0.320 | 0.316 (0.023) | 0.103 | 0.166 [0.156, 0.176] | 0.149 [0.074, 0.226] |
| The regression residual | 0.673 | 0.673 (0.024) | 0.098 | 0.425 [0.412, 0.438] | 0.345 [0.248, 0.451] |
| The conformal regression range | 0.268 | 0.252 (0.026) | 0.231 | 0.193 [0.184, 0.202] | 0.158 [0.101, 0.230] |
| TxGPT surprisal | 0.547 | -- | 0.421 | -- | 0.598 [0.534, 0.657] |
<!-- /generated:methods -->

**Limitations.** Plants are our guess at what unusual looks like; recall on them says nothing about patterns we
did not plant. DS1 has 28 addresses, so a few addresses carry a large share of the flags
(`reports/tables/eval_sanity.csv`). Fusion and regression were tuned on DS1 and fall behind the baseline on
unseen addresses (DS2, DS10; `reports/tables/stats_methods_summary.csv`). TxGPT, trained on DS2, has the highest
recall on DS10 (`reports/tables/eval_ds10.csv`).

## 2. Clustering (DS1 transactions, DS1 + DS2 address profiles)

K-Means and DBSCAN group DS1 transactions on the pooled inputs; a transaction far from every cluster scores
higher. The address-profile K-Means (`models/profiles.joblib`) groups whole addresses (DS1 and DS2) by eight
activity summaries (transactions per day, in-share, median size, gap variation, burstiness, outputs, history
length, fee rate) and describes each group in words (Assignment 3's optional activity profile).

<!-- generated:clustering -->
| Model | Chosen setting | Quality |
|---|---|---|
| K-Means (transactions) | k = 2 | silhouette 0.657, Davies-Bouldin 0.834 |
| DBSCAN (transactions) | eps = 2.626, min_samples = 8 | 2 clusters, noise share 2.0%, silhouette (non-noise) 0.786 |
| K-Means (address profiles) | k = 8 | silhouette 0.191, Davies-Bouldin 1.422 |
<!-- /generated:clustering -->

**Limitations.** Clusters describe the data's shape, not behaviour anyone has confirmed. The address profiles'
silhouette is low, so the boundaries between profiles are soft.

## 3. Regression (DS1)

For each transaction, models predict its size (log10 of 1 + satoshis) and the gap since the previous one
(hours, logged) from earlier history only. Candidates are Ridge, random forest and histogram gradient boosting,
compared with a naive model (the address's usual value); MAE and R² are on the evaluation period. The ranges use
quantile gradient boosting with split-conformal calibration.

<!-- generated:regression -->
| Target | Model | MAE | R² | Skill vs naive | Range coverage (nominal) |
|---|---|---|---|---|---|
| size | Naive (the usual value) | 0.186 | 0.730 | 0.000 |  |
| size | Ridge | 0.181 | 0.783 | 0.027 |  |
| size | Random forest | 0.168 | 0.788 | 0.099 |  |
| size | Histogram gradient boosting | 0.181 | 0.771 | 0.025 |  |
| size | quantile [0.05, 0.95] |  |  |  | 89.7% (90.0%) |
| size | quantile [0.05, 0.95] conformal |  |  |  | 89.7% (90.0%) |
| size | quantile [0.01, 0.99] |  |  |  | 97.3% (98.0%) |
| size | quantile [0.01, 0.99] conformal |  |  |  | 98.8% (98.0%) |
| gap | Naive (the usual value) | 0.476 | 0.039 | 0.000 |  |
| gap | Ridge | 0.279 | 0.635 | 0.414 |  |
| gap | Random forest | 0.189 | 0.803 | 0.603 |  |
| gap | Histogram gradient boosting | 0.191 | 0.803 | 0.599 |  |
| gap | quantile [0.05, 0.95] |  |  |  | 78.7% (90.0%) |
| gap | quantile [0.05, 0.95] conformal |  |  |  | 89.4% (90.0%) |
| gap | quantile [0.01, 0.99] |  |  |  | 89.4% (98.0%) |
| gap | quantile [0.01, 0.99] conformal |  |  |  | 97.6% (98.0%) |
<!-- /generated:regression -->

**Limitations.** Size barely beats the naive usual value (the skill column), so the size residual adds little
beyond the baseline's F1; the gap models do much better. The gradient-boosting gap model is sensitive to the last
bits of its inputs: on an Apple Silicon Mac it trains a slightly different model (CHANGELOG, Mac replication).

## 4. Classifiers (DS3 Elliptic, DS4 BitcoinHeist)

Logistic regression, k-nearest neighbours, random forest and histogram gradient boosting, each tuned on a
validation split and scored on a held-out test split. Elliptic uses its temporal split (time steps 1-34 / 35-49)
with all features (AF) or local ones (LF); BitcoinHeist uses a temporal and a random split, and
leave-one-family-out (`reports/tables/heist_lofo.csv`). Metrics are against the datasets' own labels; "illicit"
and "ransomware" are those datasets' class names. Seed spread and bootstrap intervals are in
`reports/tables/stats_classifiers_summary.csv`.

<!-- generated:classifiers -->
| Data and split | Model | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|---|
| Elliptic, temporal, features AF | Histogram gradient boosting | 0.971 | 0.808 | 0.734 | 0.769 | 0.803 |
| Elliptic, temporal, features AF | k-nearest neighbours | 0.957 | 0.684 | 0.620 | 0.650 | 0.659 |
| Elliptic, temporal, features AF | Logistic regression | 0.949 | 0.619 | 0.578 | 0.598 | 0.412 |
| Elliptic, temporal, features AF | Random forest | 0.979 | 0.928 | 0.727 | 0.815 | 0.792 |
| Elliptic, temporal, features LF | Histogram gradient boosting | 0.965 | 0.738 | 0.721 | 0.730 | 0.784 |
| Elliptic, temporal, features LF | k-nearest neighbours | 0.969 | 0.792 | 0.703 | 0.745 | 0.741 |
| Elliptic, temporal, features LF | Logistic regression | 0.969 | 0.839 | 0.640 | 0.726 | 0.619 |
| Elliptic, temporal, features LF | Random forest | 0.970 | 0.801 | 0.721 | 0.759 | 0.781 |
| BitcoinHeist, random split | Histogram gradient boosting | 0.975 | 0.199 | 0.252 | 0.222 | 0.151 |
| BitcoinHeist, random split | k-nearest neighbours | 0.957 | 0.099 | 0.254 | 0.143 | 0.071 |
| BitcoinHeist, random split | Logistic regression | 0.510 | 0.020 | 0.691 | 0.039 | 0.017 |
| BitcoinHeist, random split | Random forest | 0.979 | 0.250 | 0.239 | 0.244 | 0.177 |
| BitcoinHeist, temporal split | Histogram gradient boosting | 0.705 | 0.012 | 0.722 | 0.024 | 0.042 |
| BitcoinHeist, temporal split | k-nearest neighbours | 0.789 | 0.011 | 0.452 | 0.021 | 0.025 |
| BitcoinHeist, temporal split | Logistic regression | 0.170 | 0.006 | 0.976 | 0.012 | 0.005 |
| BitcoinHeist, temporal split | Random forest | 0.942 | 0.038 | 0.436 | 0.070 | 0.052 |
<!-- /generated:classifiers -->

**Not shipped.** Elliptic's licence allows no derivatives, and the BitcoinHeist model is 263 MB; README section
5 retrains both. **Not applied to DS1.** These classifiers learn the datasets' labels; they never score our
addresses, and `reports/tables/label_check.csv` shows unusual is not the same as labelled.
**Limitations.** BitcoinHeist's temporal split stays near chance for every model: the families change over time.

## 5. TxGPT (DS2)

A small GPT trained from scratch on DS2's transaction histories, each transaction written as six tokens (gap,
direction, amount, inputs, outputs, fee rate; vocabulary 94). Size S: 4 layers, 4 heads, width 128, 768-token
context (128 transactions), trained on 3,611 DS2 addresses with validation and test addresses held out. Three
sizes with three seeds each; S was kept because the sizes tie. Files: `models/txgpt.pt`, `models/txgpt_meta.json`.

<!-- generated:txgpt -->
| Size | Parameters | Mean validation loss (sd), 3 seeds | Kept |
|---|---|---|---|
| S | 820,352 | 0.6717 (0.0010) | yes |
| M | 4,779,264 | 0.6731 (0.0008) |  |
| L | 14,211,456 | 0.6742 (0.0008) |  |
<!-- /generated:txgpt -->

**Limitations.** Surprisal is high for anything rare in DS2, including activity that is merely new to an address.

## 6. The Explorer Assistants (explanations)

The Explorer's explanation is always available from a template (`explorer/explain.py`). With `--assistant`, a
language model may write it instead, but only if its answer passes the code guardrail
(`explorer/assistant.py`): every number comes from the facts, the limitation line is present, no banned word,
the comparison words point the right way, and the verdict matches the flag. Otherwise the template's text is used.
The writer order is Qwen 3B, then the 152M model, then the template.

| Writer | What it is | Trained on | Files |
|---|---|---|---|
| Qwen 3B, round 2 (**used**) | Qwen2.5-3B-Instruct (revision `aa8e7253`), 4-bit QLoRA: rank 16, alpha 32, dropout 0.05, on every attention and MLP projection. **Built with Qwen.** | DS7 v2 with the glossary repeated four times, plus 2,500 DS6 conversations; two Kaggle T4s, stopped early at step 2,600 (best validation loss 0.0234) | `models/qwen3b_adapter_v2/` (adapter only), `configs/qwen3b_explain_v2.json` |
| Qwen 3B, round 1 | the same, one T4 | DS7 v1 plus 2,500 DS6 conversations | `models/qwen3b_adapter/` |
| 152M model, round 1 | trained from scratch: 18 layers, 12 heads, width 768, 1,024-token context, our 32,768-token tokenizer | pretrained on 2.0B DS5 tokens, chat-tuned on DS6, then taught DS7 v1 | `models/assistant/` |
| Template | fixed sentences from the facts | none | `explorer/explain.py` |

The language-model files are over GitHub's 100 MB limit and are not in the repository (README section 8).
"Fully correct" means the answer passes the guardrail and also names the right reasons, in the right order, each
tied to its own number. The pooled rate weights every test set equally; intervals are bootstrap 95%
(`python -m lm.eval_stats`).

<!-- generated:writers -->
| Writer | Guardrail pass, DS7 test / DS10 | Fully correct, DS7 test / DS10 | Fully correct, pooled (12 sets) [95% CI] | Glossary key facts [95% CI] |
|---|---|---|---|---|
| Template (control) | 100.0% / 100.0% | 100.0% / 100.0% | -- | -- |
| 152M, round 1 | 99.3% / 99.3% | 87.0% / 83.2% | 72.7% [71.6%, 73.8%] | -- |
| Qwen 3B, round 1 | 98.9% / 98.6% | 95.5% / 95.9% | 88.9% [88.1%, 89.8%] | 15.3% [9.5%, 22.1%] |
| Qwen 3B, round 2 (used) | 99.8% / 99.3% | 96.2% / 96.6% | 92.7% [92.0%, 93.4%] | 41.3% [31.3%, 50.9%] |
<!-- /generated:writers -->

**Limitations.** A language model can still give a wrong reason that the guardrail does not catch; the template
fallback limits the damage but does not remove it. Glossary answers (questions about the Explorer's terms) are
the weakest part: round 2 states well under half of the expected key facts. English only. Qwen's licence (Qwen
Research License) allows non-commercial use only; see `LICENSES.md`.
