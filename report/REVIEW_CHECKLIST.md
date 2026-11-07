# A2 report review: file by file

Each row is a file to check against `report/main.pdf`, with the page printed at the bottom of the report
(in a PDF viewer, add 2: printed page 1 is PDF page 3). `v2` has the same body but different appendix pages.

Results are frozen at `results-v1.1`: write problems down rather than editing `reports/tables`, `data` or `models`.

- **Hoang Anh** (87 rows): 3 Data Collection, 4 Data Processing, 7.1 Technical implementation, Files the report does not quote
- **Cao Tri** (144 rows): Cover and whole report, 6 Machine Learning Model Selection, 7.2 Implementation evaluation, Files the report does not quote
- **Dan** (65 rows): Cover and whole report, 1 Introduction, 2 Problem Framing, 5 Data Analysis, 8 Conclusion, Bibliography, Appendices A to C, Files the report does not quote

## Cover and whole report (pages Cover, all pages)

| Done | Who | File | Data | Page | What to check |
|---|---|---|---|---|---|
| [ ] | Dan | `report/sections/00-cover.tex` | - | Cover | Group name, all four names and student IDs, tutor Arthur Hoang. The word count on the cover reads 2,991. |
| [ ] | Dan | `tools/report_wordcount.py` | - | Cover | Run it. It must print 2,991 body words. Tables and captions are not counted, so ask the tutor if they should be. |
| [ ] | Dan | `tools/report_build.py` | - | All | Run it. The citation and wording checks must pass with exit code 0. |
| [ ] | Cao Tri | `tools/report_numbers.py` | All datasets | All | Every number in the report is read here. Spot-check that each key reads the file this list names. |
| [ ] | Cao Tri | `report/generated/numbers.tex` | All datasets | All | The generated values. Rerun tools/report_numbers.py and confirm git shows no change in this file. |
| [ ] | Cao Tri | `report/generated/tab_*.tex` | All datasets | All | The generated tables 1 to 7. Compare a few cells with the CSV files named in the sections below. |

## 1 Introduction (pages 1)

| Done | Who | File | Data | Page | What to check |
|---|---|---|---|---|---|
| [ ] | Dan | `report/sections/01-introduction.tex` | - | 1 | Motivation, topic wording and intended users read correctly and match Assignment 1. |
| [ ] | Dan | `config/settings.json` | - | 1 | limitationLine matches the sentence quoted in the Introduction, word for word. |

## 2 Problem Framing (pages 1)

| Done | Who | File | Data | Page | What to check |
|---|---|---|---|---|---|
| [ ] | Dan | `report/sections/02-problem.tex` | - | 1 | Challenge, limits of existing tools and why machine learning. RQ1 to RQ4 match the answers on page 10. |
| [ ] | Dan | `reports/tables/eda_tests.csv` | DS1 Esplora snapshot | 1 | Median amounts of the addresses differ by a factor of 9,949. |

## 3 Data Collection (pages 1-2)

| Done | Who | File | Data | Page | What to check |
|---|---|---|---|---|---|
| [ ] | Hoang Anh | `report/sections/03-collection.tex` | - | 1-2 | Read the whole section against the rows below. Table 1 on page 2 lists eight datasets. |
| [ ] | Hoang Anh | `explorer/snapshot.py` | DS1 Esplora snapshot | 1 | The rule as written: block heights, shuffled candidates, main and short rules, the count and sum checks. |
| [ ] | Hoang Anh | `explorer/esplora.py` | DS1, DS2, DS10 | 1-2 | Returns inputs and outputs per transaction. Paces requests, backs off, fails over to mempool.space, caches replies. |
| [ ] | Hoang Anh | `config/settings.json` | DS1 Esplora snapshot | 1 | collection: 50-1,000 confirmed transactions, 90 days, 3-9 for short histories, 25 + 3 addresses. |
| [ ] | Hoang Anh | `data/processed/esplora/manifest.json` | DS1 Esplora snapshot | 1-2 | 20 block heights, 5,995 candidates, 25 main and 3 short addresses, the rule recorded with its seed. |
| [ ] | Hoang Anh | `data/processed/esplora/histories.json.gz` | DS1 Esplora snapshot | 2 | Opens, holds 28 addresses, and its SHA-256 matches the manifest. |
| [ ] | Hoang Anh | `reports/tables/cleaning_ds1.csv` | DS1 Esplora snapshot | 2 | DS1 holds 7,966 transactions. |
| [ ] | Hoang Anh | `explorer/corpus.py` | DS2 Esplora crawl | 2 | Removes DS1's candidate pool, stores addresses only as salted hashes, resumes after a stop. |
| [ ] | Hoang Anh | `data/processed/corpus/manifest.json` | DS2 Esplora crawl | 2 | 200 blocks, 1,533 removed, 9,079 visited, 4,012 kept, 106 fetch errors, 539 too many, 4,422 too few. |
| [ ] | Hoang Anh | `data/processed/corpus/ds2.parquet` | DS2 Esplora crawl | 2 | 4,012 address IDs and 898,490 rows. Addresses are hashes, never real addresses. |
| [ ] | Hoang Anh | `explorer/ds10.py` | DS10 teammate snapshot | 2 | Refetches the snapshot and compares it with the teammate's CSV. |
| [ ] | Hoang Anh | `data/processed/ds10/teammate/manifest.json` | DS10 teammate snapshot | 2 | 6 random blocks, 8 addresses per tier, 3 short histories, as the report says. |
| [ ] | Hoang Anh | `data/processed/ds10/teammate/*` | DS10 teammate snapshot | 2 | The teammate's original files. transactions.csv has 5,785 rows. |
| [ ] | Hoang Anh | `data/processed/ds10/manifest.json` | DS10 teammate snapshot | 2 | 5,785 rows and differs = 0 in the cross-check. |
| [ ] | Hoang Anh | `data/processed/ds10/histories.json.gz` | DS10 teammate snapshot | 2 | Our refetched histories. Opens, and its SHA-256 matches the manifest. |
| [ ] | Hoang Anh | `data/processed/ds10/transactions.csv` | DS10 teammate snapshot | 2 | All 5,785 transactions, in DS1's column layout. |
| [ ] | Hoang Anh | `reports/tables/ds10_crosscheck.csv` | DS10 teammate snapshot | 2 | All 5,785 agree, and no address is shared with DS1 or DS2. |
| [ ] | Hoang Anh | `explorer/external.py` | DS3, DS4 | 2 | Downloads and checksums. Elliptic's data is not redistributed. |
| [ ] | Hoang Anh | `data/processed/elliptic/download.json` | DS3 Elliptic | 2 | Source and licence CC BY-NC-ND 4.0, as in Table 1. |
| [ ] | Hoang Anh | `reports/tables/elliptic_summary.csv` | DS3 Elliptic | 2 | 46,564 labelled of 203,769 transactions, over 49 time steps. |
| [ ] | Hoang Anh | `data/processed/bitcoinheist/download.json` | DS4 BitcoinHeist | 2 | UCI repository 526, CC BY 4.0, as in Table 1. |
| [ ] | Hoang Anh | `data/processed/bitcoinheist/processing.json` | DS4 BitcoinHeist | 2 | 2,916,697 address-days, 41,413 in ransomware families. |
| [ ] | Hoang Anh | `lm/hub.py` | DS5, DS6 | 2 | Dataset names and licences match Table 1 (ODC-By 1.0, Apache 2.0). |
| [ ] | Hoang Anh | `lm/prepare_text.py` | DS5 FineWeb-Edu | 2 | How FineWeb-Edu becomes training tokens. |
| [ ] | Hoang Anh | `lm/prepare_chat.py` | DS6 smol-smoltalk | 2 | How smol-smoltalk becomes the chat stage's data. |
| [ ] | Hoang Anh | `report/sources/manual.json` | DS5 FineWeb-Edu | 2 | 5.0 billion tokens in Table 1: 2.0 in round 1 plus 3.0 in round 2. |
| [ ] | Hoang Anh | `explorer/ds7.py` | DS7 explanation pairs | 2 | Question-answer pairs built from the pipeline's own facts. |
| [ ] | Hoang Anh | `data/processed/ds7/*` | DS7 explanation pairs | 2 | Round 1 pairs: train, val, test and glossary files open, and the manifest's hashes match. |
| [ ] | Hoang Anh | `data/processed/ds7_v2/*` | DS7 explanation pairs | 2 | Round 2 glossary files. Note DS7 v2's train and val come from a Kaggle run and are not in the repo. |
| [ ] | Hoang Anh | `reports/tables/ds7_summary.csv` | DS7 explanation pairs | 2 | Pair counts per split agree with the DS7 files. |
| [ ] | Hoang Anh | `DATA_CARD.md` | All datasets | 2 | Each dataset's source, licence and size agree with Table 1. |
| [ ] | Hoang Anh | `LICENSES.md` | All datasets | 2 | Licences in Table 1 match this file. |
| [ ] | Hoang Anh | `.gitignore` | DS2 Esplora crawl | 2 | config/salt.txt, Kaggle tokens and the large downloads stay out of git. |

## 4 Data Processing (pages 2-4)

| Done | Who | File | Data | Page | What to check |
|---|---|---|---|---|---|
| [ ] | Hoang Anh | `report/sections/04-processing.tex` | - | 2-4 | Read the whole section. Figure 1 on page 3 is drawn here: confirm it matches the real steps. |
| [ ] | Hoang Anh | `explorer/transactions.py` | DS1 Esplora snapshot | 2 | The five cleaning checks in the order listed. None removes a row. |
| [ ] | Hoang Anh | `data/processed/esplora/transactions.csv` | DS1 Esplora snapshot | 2 | 7,966 rows, no repeated (address, txid), sorted by address and time, no missing values. |
| [ ] | Hoang Anh | `explorer/amounts.py` | DS1 Esplora snapshot | 3 | Received, sent, net amount and direction. |
| [ ] | Hoang Anh | `explorer/features.py` | DS1 Esplora snapshot | 3 | F1-F4, robust z from the median and MAD, the floor on the spread, strictly earlier history, same-block rule. |
| [ ] | Hoang Anh | `config/settings.json` | DS1 Esplora snapshot | 3 | features: the values under Table 2 (30-day window reaching back to 20, 10 earlier transactions over 14 days, 7-day burst). |
| [ ] | Hoang Anh | `data/processed/esplora/features.csv` | DS1 Esplora snapshot | 3 | Features are empty when history is too short, shown as not scored, never filled in. |
| [ ] | Hoang Anh | `data/processed/esplora/features_as_written.csv` | DS1 Esplora snapshot | 4 | Assignment 1's original features, used only for the ablation in Finding 2. |
| [ ] | Hoang Anh | `explorer/split.py` | DS1 Esplora snapshot | 3 | Each address fits on its earlier 70% and is evaluated on the rest. |
| [ ] | Hoang Anh | `reports/tables/split_ds1.csv` | DS1 Esplora snapshot | 3 | The fit and evaluation periods per address. |
| [ ] | Hoang Anh | `explorer/inject.py` | DS1 Esplora snapshot | 3 | The three planted patterns on copies of each history's evaluation period. |
| [ ] | Hoang Anh | `config/settings.json` | DS1 Esplora snapshot | 3 | injection: spike 10 times, burst of 10 within 24 hours, dormant 60-120 days, 50 of each per seed. |
| [ ] | Hoang Anh | `data/processed/esplora/injections/*` | DS1 Esplora snapshot | 3 | Five reporting seeds and five tuning seeds, 50 of each pattern per seed. |
| [ ] | Hoang Anh | `explorer/labelled.py` | DS3, DS4 | 3 | Elliptic joined on the transaction ID, steps 1-34 and 35-49, AF and LF. BitcoinHeist: year, day and address excluded. |
| [ ] | Hoang Anh | `lm/tx_tokens.py` | DS2 Esplora crawl | 3-4 | Six tokens per transaction: gap, direction, amount, inputs, outputs, fee rate. |
| [ ] | Hoang Anh | `lm/prepare_tx.py` | DS2 Esplora crawl | 3 | How histories become TxGPT's training windows. |
| [ ] | Hoang Anh | `models/txgpt_meta.json` | DS2 Esplora crawl | 4 | Vocabulary of 94. |
| [ ] | Hoang Anh | `tests/test_features.py` | DS1 Esplora snapshot | 3 | Run it with pytest. It must pass, and it should cover the strictly-earlier rule. |
| [ ] | Hoang Anh | `tests/test_processing.py` | DS1 Esplora snapshot | 2 | Run it with pytest. It must pass. |
| [ ] | Hoang Anh | `tests/test_inject.py` | DS1 Esplora snapshot | 3 | Run it with pytest. It must pass. |

## 5 Data Analysis (pages 4-5)

| Done | Who | File | Data | Page | What to check |
|---|---|---|---|---|---|
| [ ] | Dan | `report/sections/05-analysis.tex` | - | 4-5 | Read the whole section. Each finding gives the observation, its statistic and the decision. |
| [ ] | Dan | `notebooks/01_eda.ipynb` | DS1, DS3, DS4 | 4 | Opens with its outputs, and its conclusions agree with the four findings. |
| [ ] | Dan | `explorer/plots.py` | DS1, DS3, DS4 | 4-5 | Draws the figures. Run python -m explorer.cli figures and confirm they build. |
| [ ] | Dan | `reports/tables/eda_tests.csv` | DS1 Esplora snapshot | 4 | Finding 1: skewness 38.9 raw and -0.44 on a log scale, excess kurtosis 7,813. |
| [ ] | Dan | `reports/tables/eda_tests.csv` | DS1 Esplora snapshot | 4 | Finding 2: scale factor 9,949, and 9 of the 23 addresses tested differ. |
| [ ] | Dan | `reports/tables/eda_tests.csv` | DS1 Esplora snapshot | 4 | Finding 3: 87.3% of variance, 60.5% within radius 2, correlations 0.33 and -0.57. |
| [ ] | Dan | `reports/tables/eda_tests.csv` | DS3, DS4 | 4-5 | Finding 4: 11.6%, 6.5%, 77% unlabelled, 1.4% positives, largest effect 0.13. |
| [ ] | Dan | `reports/tables/eval_ablation.csv` | DS1 Esplora snapshot | 4 | 31.3% of transactions flagged at recall 0.439, against 10.4% and 0.655. |
| [ ] | Dan | `reports/figures/eda_feature_distributions.pdf` | DS1 Esplora snapshot | 4 | Figure 2 shows these four features, and the caption is true. |
| [ ] | Dan | `reports/figures/eda_direction_amounts.pdf` | DS1 Esplora snapshot | 5 | Figure 3 shows received and sent amounts per address, and the caption is true. |

## 6 Machine Learning Model Selection (pages 5-6)

| Done | Who | File | Data | Page | What to check |
|---|---|---|---|---|---|
| [ ] | Cao Tri | `report/sections/06-selection.tex` | - | 5 | Read the whole section. Criteria and candidates match what was actually done. |
| [ ] | Cao Tri | `config/settings.json` | DS1 Esplora snapshot | 5 | model.stabilityTarget 0.7 and sanity.maxFlaggedShare 20%. |
| [ ] | Cao Tri | `explorer/baseline.py` | DS1 Esplora snapshot | 5 | The robust z baseline flags on any feature far from the address's median. |
| [ ] | Cao Tri | `reports/tables/tuning_baseline.csv` | DS1 Esplora snapshot | 5 | The threshold sweep behind the chosen baseline. |
| [ ] | Cao Tri | `models/baseline.json` | DS1 Esplora snapshot | 6 | The saved baseline threshold, listed in Table 3's first row. |
| [ ] | Cao Tri | `explorer/clustering.py` | DS1 Esplora snapshot | 5 | K-Means and DBSCAN score the distance to normal transactions. |
| [ ] | Cao Tri | `reports/tables/tuning_kmeans.csv` | DS1 Esplora snapshot | 5 | k chosen by silhouette and Davies-Bouldin. |
| [ ] | Cao Tri | `reports/tables/tuning_dbscan.csv` | DS1 Esplora snapshot | 5 | The radius taken from the k-distance elbow. |
| [ ] | Cao Tri | `models/kmeans.joblib` | DS1 Esplora snapshot | 6 | First-search K-Means. Loads. |
| [ ] | Cao Tri | `models/dbscan.joblib` | DS1 Esplora snapshot | 6 | First-search DBSCAN. Loads. |
| [ ] | Cao Tri | `models/stage/C4_kmeans.joblib` | DS1 Esplora snapshot | 5 | The K-Means stage winner the Explorer uses. |
| [ ] | Cao Tri | `models/stage/C5_dbscan.joblib` | DS1 Esplora snapshot | 5 | The DBSCAN stage winner the Explorer uses. |
| [ ] | Cao Tri | `explorer/regression.py` | DS1 Esplora snapshot | 5 | Predicts each amount and gap, scores the miss, and calibrates ranges by conformal prediction. |
| [ ] | Cao Tri | `reports/tables/tuning_regression.csv` | DS1 Esplora snapshot | 5 | Each regressor's search against the naive forecast. |
| [ ] | Cao Tri | `models/regression.joblib` | DS1 Esplora snapshot | 6 | The saved regression method. Loads. |
| [ ] | Cao Tri | `models/regression_interval.joblib` | DS1 Esplora snapshot | 6 | The saved conformal range method. Loads. |
| [ ] | Cao Tri | `models/stage/C6_regression_interval.joblib` | DS1 Esplora snapshot | None | A stage candidate rejected on the test tier. Confirm the Explorer does not use it. |
| [ ] | Cao Tri | `explorer/outliers.py` | DS1 Esplora snapshot | 5 | Isolation Forest and LOF in novelty mode. |
| [ ] | Cao Tri | `reports/tables/tuning_outliers.csv` | DS1 Esplora snapshot | 5 | The searches behind both outlier detectors. |
| [ ] | Cao Tri | `models/isolation_forest.joblib` | DS1 Esplora snapshot | 6 | The saved Isolation Forest. Loads. |
| [ ] | Cao Tri | `models/lof.joblib` | DS1 Esplora snapshot | 6 | First-search LOF. Loads. |
| [ ] | Cao Tri | `models/stage/C3_lof.joblib` | DS1 Esplora snapshot | 5 | The LOF stage winner the Explorer uses. |
| [ ] | Cao Tri | `explorer/fusion.py` | DS1 Esplora snapshot | 5 | Fusion combines regression and Isolation Forest at a fixed flag budget. |
| [ ] | Cao Tri | `reports/tables/tuning_fusion.csv` | DS1 Esplora snapshot | 5 | The search behind the fusion. |
| [ ] | Cao Tri | `models/fusion.joblib` | DS1 Esplora snapshot | 6 | Its components are regression and isolation_forest. |
| [ ] | Cao Tri | `explorer/txgpt.py` | DS1, DS2 | 5 | TxGPT scores each transaction by its surprisal. |
| [ ] | Cao Tri | `lm/tx_score.py` | DS1, DS2 | 5 | Computes the surprisal scores. |
| [ ] | Cao Tri | `reports/tables/tuning_txgpt.csv` | DS1 Esplora snapshot | 5 | The variant and threshold chosen for TxGPT. |
| [ ] | Cao Tri | `data/processed/esplora/txgpt_scores.csv` | DS1 Esplora snapshot | 5 | TxGPT's scores for DS1's transactions. |
| [ ] | Cao Tri | `models/txgpt.pt` | DS2 Esplora crawl | 6 | The trained TxGPT weights. Table 3 says it was trained from scratch on DS2. |
| [ ] | Cao Tri | `models/txgpt_method.joblib` | DS1 Esplora snapshot | 6 | The saved TxGPT method. Loads. |
| [ ] | Cao Tri | `explorer/classify.py` | DS3, DS4 | 5 | Compares logistic regression, KNN, random forest and histogram gradient boosting. |
| [ ] | Cao Tri | `reports/tables/elliptic_tuning.csv` | DS3 Elliptic | 6 | The Elliptic searches, on a validation split. |
| [ ] | Cao Tri | `reports/tables/heist_tuning.csv` | DS4 BitcoinHeist | 6 | The BitcoinHeist searches, on a validation split. |
| [ ] | Cao Tri | `explorer/profiles.py` | DS1, DS2 | 6 | K-Means address profiles, in Table 3. |
| [ ] | Cao Tri | `data/processed/profiles.csv` | DS1, DS2 | 6 | The address profiles. |
| [ ] | Cao Tri | `reports/tables/profiles.csv` | DS1, DS2 | 6 | The profiles' centres. |
| [ ] | Cao Tri | `reports/tables/profiles_tuning.csv` | DS1, DS2 | 6 | k chosen by silhouette, as Table 3 says. |
| [ ] | Cao Tri | `reports/tables/ds2_coverage.csv` | DS1, DS2 | 6 | How well DS2 covers DS1's addresses. |
| [ ] | Cao Tri | `models/profiles.joblib` | DS1, DS2 | 6 | The saved profile model. Loads. |
| [ ] | Cao Tri | `explorer/methods.py` | DS1 Esplora snapshot | 5 | The shared scoring and recall code behind every method. |
| [ ] | Cao Tri | `explorer/scores.py` | DS1 Esplora snapshot | 5 | Which saved model each method uses, including the stage winners. |
| [ ] | Cao Tri | `data/processed/esplora/scores.csv` | DS1 Esplora snapshot | 5 | Every method's score and flag per transaction. |
| [ ] | Cao Tri | `explorer/stages.py` | DS1 Esplora snapshot | 5 | The gate: higher recall with a bootstrap interval above zero, and no pattern falling by more than its seed spread. |
| [ ] | Cao Tri | `explorer/stage_search.py` | DS1 Esplora snapshot | 5 | The stage searches. |
| [ ] | Cao Tri | `explorer/ensemble.py` | DS3 Elliptic | 5 | The classifier ensemble stage, which failed. |
| [ ] | Cao Tri | `reports/tables/stages.csv` | All datasets | 5 | LOF, K-Means and DBSCAN passed. The other scoring and classifier stages failed. |
| [ ] | Cao Tri | `reports/tables/stage_dev/*` | DS1 Esplora snapshot | 5 | The dev-tier results behind the gate. |
| [ ] | Cao Tri | `reports/tables/stage_search/*` | DS1 Esplora snapshot | 5 | The per-stage search tables. |
| [ ] | Cao Tri | `reports/tables/stage_test/*` | DS1 Esplora snapshot | 5 | The test-tier results behind the gate. |
| [ ] | Cao Tri | `reports/tables/ensemble_results.csv` | DS3 Elliptic | 5 | The ensemble stage's result. |
| [ ] | Cao Tri | `reports/tables/ensemble_search.csv` | DS3 Elliptic | 5 | The ensemble stage's search. |
| [ ] | Cao Tri | `explorer/evaluate.py` | DS1 Esplora snapshot | 5 | Applies the selection rule from Assignment 1. |
| [ ] | Cao Tri | `models/selection.json` | DS1 Esplora snapshot | 5 | chosen is baseline, runner_up is fusion. |
| [ ] | Cao Tri | `explorer/significance.py` | DS1 Esplora snapshot | 5 | The paired bootstrap of the difference. |
| [ ] | Cao Tri | `reports/tables/eval_significance.csv` | DS1 Esplora snapshot | 5 | Fusion 0.682, baseline 0.655, p = 0.324. |
| [ ] | Cao Tri | `explorer/explain.py` | DS1 Esplora snapshot | 5 | The fixed template text. |
| [ ] | Cao Tri | `explorer/assistant.py` | DS7 explanation pairs | 5 | The guardrail: numbers, limitation line, banned words, comparison direction, verdict. |
| [ ] | Cao Tri | `report/generated/tab_candidates.tex` | All datasets | 6 | Table 3 lists every model actually trained, on the right data. |

## 7.1 Technical implementation (pages 6)

| Done | Who | File | Data | Page | What to check |
|---|---|---|---|---|---|
| [ ] | Hoang Anh | `report/sections/07-implementation.tex` | - | 6 | Read section 7.1. Libraries, structure and the three commands match the code. |
| [ ] | Hoang Anh | `environment.yml` | - | 6 | Versions are pinned. pandas, scikit-learn, matplotlib and PyTorch are there. |
| [ ] | Hoang Anh | `environment-gpu.yml` | - | 6 | The GPU environment used only to retrain the language models. |
| [ ] | Hoang Anh | `requirements.txt` | - | 6 | Agrees with the versions in environment.yml. |
| [ ] | Hoang Anh | `config/settings.json` | All datasets | 6 | Holds every threshold, window, grid and seed. |
| [ ] | Hoang Anh | `explorer/settings.py` | All datasets | 6 | Loads the settings file, and now also defines TABLES_DIR and MODELS_DIR. |
| [ ] | Hoang Anh | `explorer/cli.py` | - | 6 | One command-line tool runs every step. Run python -m explorer.cli --help. |
| [ ] | Hoang Anh | `explorer/predict.py` | DS1 Esplora snapshot | 6 | The Predictor class. predict address, predict what-if and predict explain all run. |
| [ ] | Hoang Anh | `explorer/explore.py` | DS1 Esplora snapshot | 6 | The interactive Explorer. Run python -m explorer.cli explore and open one transaction. |
| [ ] | Hoang Anh | `tests/test_*.py` | - | 6 | Run python -m pytest -q. It must report 273 passed. |
| [ ] | Hoang Anh | `tools/compare_tables.py` | - | 6 | Compares rebuilt tables with the committed ones, number by number. |
| [ ] | Hoang Anh | `lm/model.py` | DS5, DS6, DS7 | 6 | The decoder design used by TxGPT and the Assistant. |
| [ ] | Hoang Anh | `lm/train.py` | DS2, DS5, DS6, DS7 | 6 | Resumes from its own checkpoint after an interruption. |
| [ ] | Hoang Anh | `lm/tx_select.py` | DS2 Esplora crawl | 6 | Picks the smallest TxGPT size when the sizes tie. |
| [ ] | Hoang Anh | `reports/tables/txgpt_sizes.csv` | DS2 Esplora crawl | 6 | Three sizes tie, and the smallest has 820,352 parameters. |
| [ ] | Hoang Anh | `reports/tables/txgpt_runs.csv` | DS2 Esplora crawl | 6 | Three seeds for each size. |
| [ ] | Hoang Anh | `report/sources/manual.json` | DS5 FineWeb-Edu | 6 | 5.0 billion DS5 tokens, as the sentence says. |
| [ ] | Hoang Anh | `explorer/inject.py` | DS2 Esplora crawl | 6 | Engineering challenge: the table is split by address first. |
| [ ] | Hoang Anh | `explorer/features.py` | DS1 Esplora snapshot | 6 | Engineering challenge: features.csv is rounded to 12 significant digits. |
| [ ] | Hoang Anh | `configs/*` | DS2, DS5, DS6, DS7 | 6 | The training configs for TxGPT and the 152M model. |
| [ ] | Hoang Anh | `kaggle/*` | DS2, DS5, DS6, DS7 | 6 | The notebooks that ran on Kaggle's GPUs, pinned to commits. |
| [ ] | Hoang Anh | `models/assistant/model.pt` *(on Cao Tri's PC)* | DS5, DS6, DS7 | 6 | 610 MB, not on GitHub yet. SHA-256 matches settings.json (_text150mCheckpoint). |
| [ ] | Hoang Anh | `models/assistant/tokenizer.json` *(on Cao Tri's PC)* | DS5, DS6, DS7 | 6 | Not on GitHub yet. It goes with the weights above. |

## 7.2 Implementation evaluation (pages 6-9)

| Done | Who | File | Data | Page | What to check |
|---|---|---|---|---|---|
| [ ] | Cao Tri | `report/sections/07-implementation.tex` | - | 6-9 | Read section 7.2 against the rows below. |
| [ ] | Cao Tri | `reports/tables/eval_selection.csv` | DS1 Esplora snapshot | 6 | The baseline finds 0.776 of burst events and 0.381 of burst transactions. |
| [ ] | Cao Tri | `report/generated/tab_objectives.tex` | DS1 Esplora snapshot | 7 | Table 4 agrees with the rows below. Two rows are Open: settle them before submitting. |
| [ ] | Cao Tri | `reports/tables/eval_selection.csv` | DS1 Esplora snapshot | 7 | Table 4: spike 0.744, dormant 0.840, burst 0.381, stability 1.000. |
| [ ] | Cao Tri | `reports/tables/eval_sanity.csv` | DS1 Esplora snapshot | 6-7 | 4 DS1 addresses break the sanity rule. |
| [ ] | Cao Tri | `reports/tables/eval_sanity_addresses.csv` | DS1 Esplora snapshot | 6 | Which addresses those are. |
| [ ] | Cao Tri | `reports/tables/eval_stability.csv` | DS1 Esplora snapshot | 7 | The stability column in Tables 4 and 5. |
| [ ] | Cao Tri | `explorer/reviewer.py` | DS1 Esplora snapshot | 7 | The blind reviewer check. Table 4 says it is still Open. |
| [ ] | Cao Tri | `reports/reviewer/*` | DS1 Esplora snapshot | 7 | The sheet and instructions. Run reviewer-score if the ratings exist, then update Table 4. |
| [ ] | Cao Tri | `report/sources/manual.json` | - | 7 | The text of Table 4's two Open rows. Correct it if the team ran either check. |
| [ ] | Cao Tri | `reports/tables/eval_selection.csv` | DS1 Esplora snapshot | 7 | Table 5's stability and flag columns. TxGPT flags 1.5%, precision bound 0.421 against 0.125. |
| [ ] | Cao Tri | `reports/tables/eval_significance.csv` | DS1 Esplora snapshot | 7 | Table 5's recall columns and confidence intervals. |
| [ ] | Cao Tri | `reports/tables/eval_injection.csv` | DS1 Esplora snapshot | 8 | The bars in Figure 4. |
| [ ] | Cao Tri | `reports/tables/eval_injection_by_seed.csv` | DS1 Esplora snapshot | 8 | The spread across the five reporting seeds. |
| [ ] | Cao Tri | `reports/tables/eval_plant_hits.csv` | DS1 Esplora snapshot | 7 | Per-plant results behind the recalls. |
| [ ] | Cao Tri | `reports/figures/eval_injection_recall.pdf` | DS1 Esplora snapshot | 8 | Figure 4, with the 70% target line. |
| [ ] | Cao Tri | `reports/tables/regression_results.csv` | DS1 Esplora snapshot | 7 | The random forest's skill: 60% on gaps, 10% on amounts. Coverage 78.4% to 87.5% against 90%. |
| [ ] | Cao Tri | `reports/tables/regression_results.csv` | DS1 Esplora snapshot | 8 | Table 6's MAE, RMSE, R-squared and skill columns. |
| [ ] | Cao Tri | `reports/tables/tuning_kmeans.csv` | DS1 Esplora snapshot | 8 | Table 6's note: k = 2, silhouette 0.657, Davies-Bouldin 0.834. |
| [ ] | Cao Tri | `reports/tables/tuning_dbscan.csv` | DS1 Esplora snapshot | 8 | Table 6's note: 1.3 times the elbow, silhouette 0.786, 2.0% noise. |
| [ ] | Cao Tri | `reports/tables/kmeans_clusters.csv` | DS1 Esplora snapshot | 8 | The K-Means clusters behind that note. |
| [ ] | Cao Tri | `reports/tables/eval_agreement.csv` | DS1 Esplora snapshot | 7 | Overlap 0.24-0.37 with the other single methods, 0.06 with TxGPT. Table 4's 0.512 with fusion. |
| [ ] | Cao Tri | `explorer/walkforward.py` | DS1 Esplora snapshot | 7 | Refits every method at each cut-off. |
| [ ] | Cao Tri | `reports/tables/eval_walkforward.csv` | DS1 Esplora snapshot | 7 | 4 cut-offs. Baseline 0.650-0.658, LOF 0.331-0.410, as the mean of spike, burst and dormant. |
| [ ] | Cao Tri | `explorer/ds2_check.py` | DS2 Esplora crawl | 7 | Runs the frozen methods on DS2 histories. |
| [ ] | Cao Tri | `reports/tables/split_ds2.csv` | DS2 Esplora crawl | 7 | 2,933 main histories. |
| [ ] | Cao Tri | `reports/tables/eval_ds2.csv` | DS2 Esplora crawl | 9 | DS2's points in Figure 5. |
| [ ] | Cao Tri | `reports/tables/eval_ds2_plant_hits.csv` | DS2 Esplora crawl | 7 | Per-plant results for DS2. |
| [ ] | Cao Tri | `reports/tables/eval_ds10.csv` | DS10 teammate snapshot | 7-8 | Fusion falls to 0.367, the baseline holds 0.431, TxGPT is first at 0.598 [0.534, 0.657]. |
| [ ] | Cao Tri | `reports/tables/eval_ds10_plant_hits.csv` | DS10 teammate snapshot | 8 | Per-plant results for DS10. |
| [ ] | Cao Tri | `reports/tables/split_ds10.csv` | DS10 teammate snapshot | 8 | DS10's fit and evaluation periods. |
| [ ] | Cao Tri | `reports/tables/ds10_activity.csv` | DS1, DS10 | 7 | Median gap 13.1 hours on DS10 against 23.1 on DS1. |
| [ ] | Cao Tri | `explorer/distribution.py` | DS1, DS2, DS10 | 7 | Repeats the evaluation over many plant seeds. |
| [ ] | Cao Tri | `reports/tables/stats_methods_summary.csv` | DS1, DS2, DS10 | 7 | The baseline ranks first on DS10 in 96% of seeds. |
| [ ] | Cao Tri | `reports/tables/stats_methods_seeds.csv` | DS1, DS2, DS10 | 7 | The per-seed results behind that 96%. |
| [ ] | Cao Tri | `reports/figures/eval_ds10_check.pdf` | DS2, DS10 | 9 | Figure 5, and the caption is true. |
| [ ] | Cao Tri | `reports/tables/elliptic_results.csv` | DS3 Elliptic | 9 | Table 7's Elliptic rows. |
| [ ] | Cao Tri | `reports/tables/heist_results.csv` | DS4 BitcoinHeist | 8-9 | Table 7's BitcoinHeist rows. No temporal F1 above 0.07. |
| [ ] | Cao Tri | `explorer/classify_distribution.py` | DS3, DS4 | 8 | Refits classifiers over ten training seeds. |
| [ ] | Cao Tri | `reports/tables/stats_classifiers_summary.csv` | DS3, DS4 | 8 | Random forest F1 0.815, interval [0.796, 0.835]. |
| [ ] | Cao Tri | `reports/tables/stats_classifiers_seeds.csv` | DS3, DS4 | 8 | The per-seed results behind that interval. |
| [ ] | Cao Tri | `reports/tables/stats_classifiers_paired.csv` | DS3, DS4 | None | Paired comparisons, not quoted. Agree with the summary. |
| [ ] | Cao Tri | `reports/tables/elliptic_f1_by_step.csv` | DS3 Elliptic | 8 | The collapse from time step 43. |
| [ ] | Cao Tri | `reports/tables/heist_lofo.csv` | DS4 BitcoinHeist | 8 | Held-out families found at 0.3%-3.3% by the tree models. |
| [ ] | Cao Tri | `models/heist_best.joblib` *(on Cao Tri's PC)* | DS4 BitcoinHeist | 9 | 276 MB, not on GitHub yet. The chosen DS4 model, best temporal validation F1. |
| [ ] | Cao Tri | `explorer/label_check.py` | DS3, DS4 | 8 | Scores the unsupervised methods against the labelled classes. |
| [ ] | Cao Tri | `reports/tables/label_check.csv` | DS3, DS4 | 8 | Unsupervised PR-AUC below the class share. Random forest 12.2 times chance on Elliptic. |
| [ ] | Cao Tri | `reports/tables/label_check_tuning.csv` | DS3, DS4 | 8 | The tuning behind the label check. |
| [ ] | Cao Tri | `lm/eval_explain.py` | DS7, DS10 | 8 | How the Assistant's answers are scored. |
| [ ] | Cao Tri | `lm/eval_stats.py` | Random test sets | 8 | The fully correct rates and intervals. |
| [ ] | Cao Tri | `lm/adopt.py` | Random test sets | 8 | The rule that adopted round 2. |
| [ ] | Cao Tri | `reports/tables/adoption.csv` | Random test sets | 8 | Round 2 passes every criterion. |
| [ ] | Cao Tri | `reports/tables/assistant_eval.csv` | DS7 explanation pairs | 8 | text150m_v2 guardrail pass 100.0%. |
| [ ] | Cao Tri | `reports/tables/assistant_eval_ds10.csv` | DS10 teammate snapshot | 8 | text150m_v2 guardrail pass 100.0% on DS10. |
| [ ] | Cao Tri | `reports/tables/stats_writers_summary.csv` | Random test sets | 8 | 96.9%, 93.1%, pooled 94.8% [94.1%, 95.4%], glossary 48%. |
| [ ] | Cao Tri | `reports/tables/stats_writers_paired.csv` | Random test sets | None | Paired comparisons, not quoted. Agree with the summary. |
| [ ] | Cao Tri | `reports/tables/assistant_eval_*` | Random test sets | 8 | The per-set and per-case tables behind the pooled rate. |
| [ ] | Cao Tri | `explorer/random_test.py` | Random test sets | 8 | Builds the randomized test sets. |
| [ ] | Cao Tri | `data/processed/random_test/*` | Random test sets | 8 | The test sets. Open, and the manifest's hashes match. |
| [ ] | Cao Tri | `data/processed/frozen_tests.json` | Random test sets | 8 | Run check_frozen in explorer/stages.py. It must not raise. |
| [ ] | Cao Tri | `data/processed/ds10/explain.jsonl` | DS10 teammate snapshot | 8 | DS10's Assistant prompts. |
| [ ] | Cao Tri | `reports/assistant_tests.pdf` | Random test sets | 8 | Every Assistant test set and result, in one document. |
| [ ] | Cao Tri | `MODEL_CARD.md` | All datasets | 8 | Metrics agree with the tables above. |

## 8 Conclusion (pages 10)

| Done | Who | File | Data | Page | What to check |
|---|---|---|---|---|---|
| [ ] | Dan | `report/sections/08-conclusion.tex` | - | 10 | RQ1 to RQ4 answer the questions on page 1. Claims agree with sections 6 and 7. |
| [ ] | Dan | `reports/tables/eval_selection.csv` | DS1 Esplora snapshot | 10 | The baseline flags 10.4% of transactions, the figure in Impact and limitations. |
| [ ] | Dan | `config/settings.json` | - | 10 | Limitations are honest: DS1 has only 28 addresses. |

## Bibliography (pages 11-12)

| Done | Who | File | Data | Page | What to check |
|---|---|---|---|---|---|
| [ ] | Dan | `report/sections/09-references.tex` | - | 11-12 | Harvard format, alphabetical, 29 entries, every entry cited. Open each source and confirm it supports the claim. |
| [ ] | Dan | `docs/REFERENCES.md` | - | 11-12 | Why each source was chosen. By hand: Blockstream, Elliptic, Lai, mempool.space, Pineau. Check the years of Blazquez-Garcia and Boukerche. |

## Appendices A to C (pages 13-18)

| Done | Who | File | Data | Page | What to check |
|---|---|---|---|---|---|
| [ ] | Dan | `report/sections/10-appendix.tex` | - | 13-18 | Captions are true and match the figures. |
| [ ] | Dan | `report/generated/example.tex` | DS1 Esplora snapshot | 13 | Run README section 3's predict what-if. The output matches Appendix A. |
| [ ] | Dan | `reports/figures/eda_pca.pdf` | DS1 Esplora snapshot | 13 | Figure 6, with Finding 3. |
| [ ] | Dan | `reports/figures/eda_feature_correlation.pdf` | DS1 Esplora snapshot | 14 | Figure 7, with Finding 3. |
| [ ] | Dan | `reports/figures/eval_agreement.pdf` | DS1 Esplora snapshot | 14 | Figure 8, against eval_agreement.csv. |
| [ ] | Dan | `reports/figures/eval_walkforward.pdf` | DS1 Esplora snapshot | 15 | Figure 9, against eval_walkforward.csv. |
| [ ] | Dan | `reports/figures/eval_conformal.pdf` | DS1 Esplora snapshot | 15 | Figure 10, against regression_results.csv. |
| [ ] | Dan | `reports/figures/eval_elliptic_f1_by_step.pdf` | DS3 Elliptic | 15 | Figure 11: every model fails from step 43. |
| [ ] | Dan | `reports/figures/eval_label_check.pdf` | DS3, DS4 | 16 | Figure 12, against label_check.csv. |
| [ ] | Dan | `explorer/sensitivity.py` | DS1 Esplora snapshot | 16 | Recall against plant strength. |
| [ ] | Dan | `reports/tables/eval_sensitivity.csv` | DS1 Esplora snapshot | 16 | The data in Figure 13. |
| [ ] | Dan | `reports/figures/eval_sensitivity.pdf` | DS1 Esplora snapshot | 16 | Figure 13, and the caption is true. |
| [ ] | Dan | `reports/figures/stats_methods.pdf` | DS1, DS2, DS10 | 16 | Figure 14: 100 plant seeds on DS1 and DS10, 20 on DS2. |
| [ ] | Dan | `reports/figures/eval_stages.pdf` | All datasets | 17 | Figure 15, against stages.csv. |
| [ ] | Dan | `reports/figures/eval_txgpt_sizes.pdf` | DS2 Esplora crawl | 17 | Figure 16, against txgpt_sizes.csv. |
| [ ] | Dan | `reports/figures/stats_writers.pdf` | Random test sets | 18 | Figure 17, against stats_writers_summary.csv. |
| [ ] | Dan | `README.md` | All datasets | 18 | The commands in Appendix C run exactly as written. Follow sections 2 and 3 from the zip. |
| [ ] | Dan | `tools/cards.py` | All datasets | 18 | Refreshes DATA_CARD.md and MODEL_CARD.md. |

## Files the report does not quote

| Done | Who | File | Data | Page | What to check |
|---|---|---|---|---|---|
| [ ] | Cao Tri | `explorer/__init__.py` | - | None | Package marker. |
| [ ] | Cao Tri | `lm/__init__.py` | - | None | Package marker. |
| [ ] | Cao Tri | `lm/chat.py` | DS6 | None | Interactive chat with the 152M model. Not quoted. |
| [ ] | Cao Tri | `lm/generate.py` | - | None | Text generation behind the Assistant. Not quoted by name. |
| [ ] | Cao Tri | `lm/writers.py` | - | None | Loads the Assistant for the guardrail. Not quoted by name. |
| [ ] | Cao Tri | `lm/data.py` | - | None | Training data windows. Not quoted by name. |
| [ ] | Cao Tri | `lm/dist.py` | - | None | Two-GPU training. Not quoted. |
| [ ] | Cao Tri | `lm/tokenizer.py` | - | None | The 152M tokenizer. Not quoted by name. |
| [ ] | Cao Tri | `lm/smoke.py` | - | None | A short training check. Not quoted. |
| [ ] | Cao Tri | `lm/tx_baselines.py` | DS2 | None | TxGPT's comparison baselines. Results in txgpt_nll.csv, not quoted. |
| [ ] | Cao Tri | `reports/tables/txgpt_nll.csv` | DS2 | None | Not quoted. |
| [ ] | Cao Tri | `explorer/weak_cases.py` | DS7 explanation pairs | None | Stage B3's weak-case examples. Not quoted by name. |
| [ ] | Cao Tri | `data/processed/weak_cases/*` | DS7 explanation pairs | None | Stage B3's data. Not quoted by name. The zip keeps it because the final Assistant used it. |
| [ ] | Hoang Anh | `tools/kaggle_notebooks.py` | - | None | Generates the Kaggle notebooks. |
| [ ] | Hoang Anh | `tools/package.py` | - | None | Builds the zip. Decide what its EXCLUDE list should add. |
| [ ] | Hoang Anh | `tools/make_glossary.py` | DS7 explanation pairs | None | Writes the glossary. A person still has to read its 65 topics. |
| [ ] | Hoang Anh | `tools/make_scoring_fixture.py` | - | None | Test fixture for the A1 scoring rule. |
| [ ] | Hoang Anh | `tools/run_queue.py` | - | None | Job queue used during training. |
| [ ] | Hoang Anh | `tools/status.py` | - | None | Job status helper. |
| [ ] | Hoang Anh | `tools/supervise.py` | - | None | Job supervisor. |
| [ ] | Hoang Anh | `tools/test_report.py` | Random test sets | None | Builds assistant_tests.pdf. Its log_excerpt ignores its title argument, a known small bug. |
| [ ] | Hoang Anh | `scripts/d1_check.py` | - | None | Assignment 1's one-address check. Not used by the final models: decide whether to drop it from the zip. |
| [ ] | Hoang Anh | `data/d1/*` | - | None | Assignment 1's sample replies. Not used by the final models: decide whether to drop it from the zip. |
| [ ] | Hoang Anh | `reference/a1_prototype/*` | - | None | The A1 prototype's scoring code, which predict.py follows. Not quoted. |
| [ ] | Dan | `docs/a1/*` | - | None | Assignment 1's LaTeX. Decide whether to drop it from the zip. |
| [ ] | Dan | `docs/plan/*` | - | None | The delivery plan's LaTeX. Module docstrings cite it. Decide whether to drop it from the zip. |
| [ ] | Dan | `docs/A2_delivery_plan.pdf` | - | None | The delivery plan. Decide whether to drop it from the zip. |
| [ ] | Dan | `docs/AI_USE_LOG.md` | - | None | The source for the AI declaration. Confirm it is accurate. |
| [ ] | Dan | `docs/AUDIT.md` | - | None | The 29 Oct audit. |
| [ ] | Dan | `docs/KAGGLE_LM_PLAN.md` | - | None | The plan for round 2 and its adoption rule. |
| [ ] | Dan | `docs/REPORT_PLAN.md` | - | None | The plan for this report. |
| [ ] | Dan | `docs/TODO.md` | - | None | The team's open items. |
| [ ] | Dan | `docs/TRAINING_PLAN.md` | - | None | The training stages and their gate. |
| [ ] | Dan | `CHANGELOG.md` | - | None | Every decision and result, newest first. |
| [ ] | Cao Tri | `reports/tables/elliptic_importance.csv` | DS3, DS4 | None | Not quoted in the report. Written by the classification steps. |
| [ ] | Cao Tri | `reports/tables/elliptic_steps.csv` | DS3, DS4 | None | Not quoted in the report. Written by the classification steps. |
| [ ] | Cao Tri | `reports/tables/heist_families.csv` | DS3, DS4 | None | Not quoted in the report. Written by the classification steps. |
| [ ] | Cao Tri | `reports/tables/heist_skew.csv` | DS3, DS4 | None | Not quoted in the report. Written by the classification steps. |
| [ ] | Cao Tri | `reports/tables/heist_splits.csv` | DS3, DS4 | None | Not quoted in the report. Written by the classification steps. |
| [ ] | Cao Tri | `reports/tables/heist_years.csv` | DS3, DS4 | None | Not quoted in the report. Written by the classification steps. |
| [ ] | Dan | `reports/figures/eda_ds2_coverage.pdf` | All datasets | None | Not in the report. Confirm it still builds, and decide whether the zip needs it. |
| [ ] | Dan | `reports/figures/eda_elliptic_classes.pdf` | All datasets | None | Not in the report. Confirm it still builds, and decide whether the zip needs it. |
| [ ] | Dan | `reports/figures/eda_flag_rate.pdf` | All datasets | None | Not in the report. Confirm it still builds, and decide whether the zip needs it. |
| [ ] | Dan | `reports/figures/eda_gap_ecdf.pdf` | All datasets | None | Not in the report. Confirm it still builds, and decide whether the zip needs it. |
| [ ] | Dan | `reports/figures/eda_heist_features.pdf` | All datasets | None | Not in the report. Confirm it still builds, and decide whether the zip needs it. |
| [ ] | Dan | `reports/figures/eda_heist_years.pdf` | All datasets | None | Not in the report. Confirm it still builds, and decide whether the zip needs it. |
| [ ] | Dan | `reports/figures/eda_history_sizes.pdf` | All datasets | None | Not in the report. Confirm it still builds, and decide whether the zip needs it. |
| [ ] | Dan | `reports/figures/eda_periodicity.pdf` | All datasets | None | Not in the report. Confirm it still builds, and decide whether the zip needs it. |
| [ ] | Dan | `reports/figures/eda_profiles.pdf` | All datasets | None | Not in the report. Confirm it still builds, and decide whether the zip needs it. |
| [ ] | Dan | `reports/figures/eval_ds2_check.pdf` | All datasets | None | Not in the report. Confirm it still builds, and decide whether the zip needs it. |
| [ ] | Dan | `reports/figures/eval_fusion_frontier.pdf` | All datasets | None | Not in the report. Confirm it still builds, and decide whether the zip needs it. |
| [ ] | Dan | `reports/figures/eval_heist_lofo.pdf` | All datasets | None | Not in the report. Confirm it still builds, and decide whether the zip needs it. |
| [ ] | Dan | `reports/figures/eval_pr_curves.pdf` | All datasets | None | Not in the report. Confirm it still builds, and decide whether the zip needs it. |
| [ ] | Dan | `reports/figures/eval_significance.pdf` | All datasets | None | Not in the report. Confirm it still builds, and decide whether the zip needs it. |
| [ ] | Dan | `reports/figures/stats_classifiers.pdf` | All datasets | None | Not in the report. Confirm it still builds, and decide whether the zip needs it. |
