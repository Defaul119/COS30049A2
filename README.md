# Bitcoin Address Activity Explorer: Assignment 2 (data and machine learning)

COS30049 Computing Technology Innovation Project, group **HCMC_COS30049.1_4** (Swinburne, HCMC campus).

The Explorer ranks each transaction of a Bitcoin address by how **unusual** it is for that address's own
past, and says why in plain words. This repository holds Assignment 2: the datasets, every model (clustering,
regression, classification, outlier detection and two language models), their evaluation, and a prediction
interface that Assignment 3's web app will call. It is an investigation aid, never a fraud detector:
*unusual activity requires investigation; it does not indicate wrongdoing.*

```
config/settings.json   every threshold, window, seed and grid (no magic numbers in the code)
explorer/              the pipeline: one module per step, all run through explorer/cli.py
lm/                    the language models, both trained from scratch: TxGPT and the 152M text model; evaluation
configs/               language-model training settings
data/processed/        the processed datasets (DS1, DS2, DS7, DS10 and records for DS3/DS4)
models/                the trained models the Explorer uses
reports/               figures (PDF), tables (CSV, the source of every number in the report), logs, and
                       assistant_tests.pdf (every Assistant test set and result)
tests/                 unit tests (pytest)
kaggle/                the notebooks that trained and tested the language models on Kaggle
tools/                 helpers: job status, the Kaggle notebook generator, the test-report PDF, the rebuild check
docs/                  the delivery plan, the brief, the report plan, the AI use log, archive/LM_TRAINING.md
CHANGELOG.md           every decision and measured result, newest first
DATA_CARD.md           every dataset: source, licence, collection, biases, and every file and column
MODEL_CARD.md          every final model: purpose, data, parameters, metrics, limits, intended use
LICENSES.md            third-party licences
```

Every model here is trained by the team; the language models are trained from scratch (no pretrained model is
fine-tuned).

## 1. What you need

- Windows 10/11, macOS (Apple Silicon: M1 or later) or Linux, with git and conda:
  [Miniforge](https://github.com/conda-forge/miniforge) (what we use) or
  [Miniconda](https://docs.anaconda.com/miniconda/). A fresh Miniconda refuses `conda env create` until you accept
  Anaconda's terms of service: run the `conda tos accept ...` commands its error message prints, then retry.
  Miniforge needs no such step.
- About 5 GB of free disk and 8 GB of RAM. No GPU: an NVIDIA GPU is only needed to *retrain* the language
  models (section 7).
- Intel Macs: PyTorch stopped publishing Intel-Mac builds after version 2.2, so `torch==2.7.1` in
  `environment.yml` will not install there. Use an Apple Silicon Mac, Windows or Linux.

## 2. Set up (about 10 minutes, once)

```
git clone https://github.com/Defaul119/COS30049A2.git
cd COS30049A2
conda env create -f environment.yml
conda activate cos30049-a2
python -m pytest -q
```

Expected: `259 passed` in about three minutes (the first run takes longer while the new libraries load).
Run every later command from the repository folder with `cos30049-a2` active. On Windows, run
`set PYTHONUTF8=1` first in each new terminal (Command Prompt) so the logs print correctly; macOS and Linux
need nothing.

## 3. Quick check: predictions from the shipped models (2 minutes)

The repository ships the processed data and the trained models, so prediction works straight after set-up.

**An address's timeline and every transaction's scores** (DS1 snapshot addresses only):

```
python -m explorer.cli predict address 1Bbcsfxj2JRUD5crcdZVHBcViT61vrJ3dz
```

prints JSON with `snapshot`, `range`, `days` (214 days), `totals` and `transactions` (377). Its totals:

```
"totals": {"txCount": 377, "scoredCount": 367, "flaggedCount": 3, "flaggedDays": 3}
```

An address outside the snapshot exits with code 2 and suggests supported ones:
`Not in the snapshot: NOTANADDRESS is not in the snapshot; try one of [...]`.

**A hypothetical next transaction** (the "user input -> prediction" Assignment 3 needs):

```
python -m explorer.cli predict what-if --address 1Bbcsfxj2JRUD5crcdZVHBcViT61vrJ3dz --amount 0.05 --direction out --time 2026-10-20T12:00Z
```

```
"scored": true, "flagged": true, "score": 5.98, ...
"features": [{"id": "F1", "name": "Size", "value": 0.05, "usual": 0.000525, "z": 5.88, "role": "flag"}, ...]
```

**One transaction's explanation:**

```
python -m explorer.cli predict explain --address 1Bbcsfxj2JRUD5crcdZVHBcViT61vrJ3dz --txid b2afad6cfb794fdc2de8879ff9f9cfff68933664b7c45048cd747bc5d622736b
```

```
"explanation": "Flagged by Robust z baseline (A1).\nWhat stands out compared with this address's own history:\n- Longer gap than usual: 5 days since the previous transaction, against a usual 1 hour.\nUnusual activity requires investigation; it does not indicate wrongdoing.",
"source": "template"
```

Add `--assistant` to let a language model write it instead (section 8); without the language-model files the
answer stays the template's, and `skippedWriters` says why.

## 4. Rebuild every result from the shipped data (offline, about 1.5-2 hours on a laptop)

```
python -m explorer.cli all --offline
python -m explorer.cli figures
```

`all` runs every step below in order, offline, and stops at the first failure. Each step writes its log to
`reports/logs/<step>.log`. Afterwards, check the rebuild against our committed results:

```
python tools/compare_tables.py
```

It compares every table in `reports/tables/` with the committed copy, number by number, and prints only the tables
that differ (exit code 1 if any differs by more than 0.01 x max(1, |our value|); see section 11). Times were measured on our PC (Intel
i5-12600K, 16 threads); a laptop takes longer.

| Step | Reads | Writes | Time |
|---|---|---|---|
| `transactions` | `data/processed/esplora/histories.json.gz` (DS1) | `transactions.csv`, `reports/tables/cleaning_ds1.csv` | 1 s |
| `features` (and `--as-written`) | DS1 transactions | `features.csv`: F1-F8, robust z-scores, baseline flag | 2 s |
| `split` | features | fit / evaluation periods, `split_ds1.csv` | 1 s |
| `inject` | DS1 | planted transactions (tuning and reporting seeds) in `data/interim/` | 45 s |
| `baseline` | DS1 + plants | robust-z threshold sweep, `models/baseline.json` | 3 s |
| `outliers` | DS1 + plants | Isolation Forest and LOF searches and models | 3 min |
| `clustering` | DS1 + plants | K-Means and DBSCAN | 15 s |
| `regression` | DS1 + plants | Ridge, random forest, gradient boosting, conformal ranges | 1 min |
| `profiles` | DS1 + DS2 | address profiles, DS2's coverage of DS1 | 10 s |
| `fusion` | the first-round (M0) methods + plants | two methods fused at one flag budget | 18 min |
| `scores` | every saved model | `data/processed/esplora/scores.csv` | 10 s |
| `evaluate` | everything above | recall, stability, agreement, sanity, ablation, the selection | 5 min |
| `significance` | plant hits | address-level bootstrap intervals | 15 s |
| `sensitivity` | DS1 | recall against plant strength | 4 min |
| `walk-forward` | DS1 | every method at four fit/evaluation cut-offs | 5 min |
| `ds2-check` | `data/processed/corpus/ds2.parquet` | the frozen methods on 2,933 DS2 histories | 5 min |
| `ds10-check` | `data/processed/ds10/transactions.csv` | every model, TxGPT included, on DS10 | 6 min (about 40 min on a laptop) |
| `label-check` | DS3/DS4 if downloaded | unsupervised scores against the labels | skipped without them |
| `figures` | `reports/tables/` | every figure in `reports/figures/` | 15 s |

On a fresh copy three things are skipped, and the log says so:
- `txgpt` (re-tuning the TxGPT method) needs the planted copies' TxGPT scores, made by
  `python -m lm.tx_score --plants` (minutes on a GPU, about 2 hours on a CPU). Without them the saved
  `models/txgpt_method.joblib`, which carries every score it needs, is used as it is.
- DS3 (Elliptic) and DS4 (BitcoinHeist) processing and classification need their downloads (section 5).

`git status` after a rebuild also lists files that changed for harmless reasons:
- `reports/logs/*.log` change on every run (they record times).
- Without the DS4 download, `reports/tables/eda_tests.csv` loses its six DS4 rows.
- On macOS and Linux, rebuilt files in `data/processed/` can show as modified because the committed copies have
  Windows line endings; compare them with `git diff --ignore-cr-at-eol --stat data/processed`.

Any single step can be run alone: `python -m explorer.cli <step>` (for example `evaluate`).
`python -m explorer.cli --help` lists them all.

**Statistics and extra test sets (not part of `all`).** These steps repeat the evaluation many times to give
spreads and intervals instead of single numbers; each writes `reports/tables/stats_*.csv`, and `figures` draws
them.

| Command | What it does | Time |
|---|---|---|
| `python -m explorer.cli distribution` | every scoring method's planted-pattern recall over 100 plant seeds (DS1, DS10) and 20 (DS2) | about 2 h on 4 CPU cores |
| `python -m explorer.cli class-distribution` | the DS3/DS4 classifiers over 10 training seeds, with a 1,000-draw test bootstrap (needs section 5) | about 1 h on 4 CPU cores |
| `python -m explorer.cli build-random-test`, `build-balanced-test`, `build-more-tests` | the ten randomized Assistant test sets in `data/processed/random_test/` (committed). `build-random-test` first needs DS7 v2's test split: `python -m explorer.cli build-ds7 --settings ds7v2 --out data/processed/ds7_v2` | seconds each |
| `python -m lm.eval_stats` | the Assistants' fully correct rates on all 12 test sets, with bootstrap intervals and McNemar tests | 10 s |
| `python tools/test_report.py` | `reports/assistant_tests.pdf`: every test set, how it was drawn, and every writer's results (needs pdflatex) | 1 min |

## 5. Getting the data again (optional, needs internet)

New downloads give slightly different numbers, because the blockchain keeps growing.

| Command | What it does | Notes |
|---|---|---|
| `python -m explorer.cli snapshot --refetch` | DS1: choose 25 + 3 addresses by the rule in `settings.json`, fetch and verify them, write the manifest | Blockstream's public API (about 700 requests an hour per IP), falling back to mempool.space; 30-60 min |
| `python -m explorer.cli corpus` | DS2: crawl about 4,000 address histories (hashed IDs) | Hours; resumes where it stopped |
| `python -m explorer.cli ds10` | DS10: refetch the teammate's 27 addresses and cross-check them with their CSV | About 12 min |
| `python -m explorer.cli external --bitcoinheist` | DS4 from UCI (CC BY 4.0), then `process-external` and `classify --dataset heist` | Classification takes about 35 min |
| `python -m explorer.cli external --elliptic` | DS3 from Kaggle (CC BY-NC-ND 4.0), then `process-external` and `classify --dataset elliptic` | Needs a Kaggle account: save its API token as `~/.kaggle/kaggle.json`, or download the zip by hand when the step says how |

The classifiers' trained models stay out of the repository: Elliptic's licence allows no derivatives, and the
BitcoinHeist model is 263 MB. So `predict heist` and `predict elliptic` work only after these steps.

## 6. Training the models

- **scikit-learn models** (baseline, Isolation Forest, LOF, K-Means, DBSCAN, regression, fusion, the DS3/DS4
  classifiers) are trained by the steps in sections 4 and 5. Every grid, seed and threshold is in
  `config/settings.json`; every search is in `reports/tables/tuning_*.csv`.
- **Language models** need `environment-gpu.yml` and an NVIDIA GPU (section 7).
- **Training stages** (`docs/TRAINING_PLAN.md`): after the first round (milestone M0, git tag `m0`) every model was
  retrained one stage at a time, and each stage was kept only if it passed a gate fixed before any result: higher
  recall with a bootstrap interval above 0, no planted pattern falling by more than its seed spread, the flag rate
  within the sanity limit. Every decision is a row of `reports/tables/stages.csv`. The commands, each a CPU job:

  | Command | What it does |
  |---|---|
  | `python -m explorer.cli stage-search --stages C3` | one method's grid at its M0 flag budget, ranked on tuning seeds 40000-40002, gated on 40003-40004 (C1-C7, C9) |
  | `python -m explorer.cli ensemble` | stage D6: an ensemble of the Elliptic classifiers, gated against the random forest |
  | `python -m explorer.cli stage-test --snapshot C`, then `--before M0 --after C --stage C-test --methods ...` | the test tier: the kept stages scored once on the reporting plants and gated against M0 |
  | `python -m explorer.cli build-weak-cases`, `writer-gate --before ... --after ...` | stage B3 of the 152M writer: weak-case training examples, and the gate between two writers' dev answers |

  Kept stages replace the M0 models through `config/settings.json -> stageModels` (LOF C3, K-Means C4, DBSCAN C5);
  the winners are in `models/stage/`.

## 7. Retraining the language models (GPU, optional)

```
conda env create -f environment-gpu.yml
conda activate cos30049-a2-gpu
```

| Model | Commands | Time on our RTX 3060 |
|---|---|---|
| TxGPT (3 sizes x 3 seeds) | `python -m lm.prepare_tx`, then `python -m lm.train --config configs/txgpt_s.json --seed 1` (and `_m`, `_l`, seeds 1-3), `python -m lm.tx_select`, `python -m lm.tx_score --plants`, `python -m explorer.cli txgpt` | 6-30 min per run |
| 152M text model | `docs/archive/LM_TRAINING.md`: pretraining (2B tokens), chat stage, explanation stage (`configs/text_150m_explain.json`) | about 3 days in all |
| 152M round 2 | `kaggle/text150m_data/`, `kaggle/text150m_train/` (continued pretraining on two Kaggle T4s), `kaggle/ds7_data/`, `kaggle/text150m_stage_data/`, `kaggle/text150m_stages/` (chat and explanation stages, then every evaluation) | about 2 days of Kaggle sessions |
| Explanation evaluation | `python -m lm.eval_explain --model template`, `--model text150m --checkpoint models/assistant/model.pt` | 5 min (152M) |

`tools/run_queue.py --plan configs/queue_after_pretraining.json` runs the whole GPU sequence unattended and
resumes after an interruption. `python tools/kaggle_notebooks.py <commit>` regenerates the Kaggle notebooks,
pinned to that commit of this repository; `docs/KAGGLE_LM_PLAN.md` describes round 2 and the rule that decides
whether a retrained model replaces the one in use.

## 8. Prediction

From the command line (section 3 shows the outputs):

```
python -m explorer.cli predict address <address>
python -m explorer.cli predict what-if --address <address> --amount 0.05 --direction out --time 2026-10-20T12:00Z [--assistant]
python -m explorer.cli predict explain --address <address> --txid <txid> [--assistant]
python -m explorer.cli predict heist --length 18 --weight 0.008 --count 1 --looped 0 --neighbors 2 --income 1.2e8   # after section 5
python -m explorer.cli predict elliptic --rows <feature rows.csv>                                                   # after section 5
```

From Python (what Assignment 3's back end calls):

```python
from explorer.predict import Predictor
p = Predictor.load()
timeline = p.score_address("1Bbcsfxj2JRUD5crcdZVHBcViT61vrJ3dz")
guess = p.score_what_if("1Bbcsfxj2JRUD5crcdZVHBcViT61vrJ3dz", 0.05, "out", "2026-10-20T12:00Z")
text = p.explain(guess, use_assistant=False)
```

**The Assistant (`--assistant`).** Explanations are written by the first writer in
`settings.json -> assistant -> writerOrder` that passes the code guardrail (every number from the facts, the
limitation line, no banned word, the right comparison words and verdict), else by the template. The writer is
the 152M model trained from scratch, which runs in the CPU environment. Its weights (`models/assistant/`) are over
GitHub's 100 MB limit and are not in the repository; without them the template writes.

## 9. Data

| ID | Dataset | Source and licence | Used for |
|---|---|---|---|
| DS1 | Our snapshot: 25 + 3 addresses, 7,966 transactions | Blockstream Esplora API (public blockchain data) | Every unsupervised model and the evaluation |
| DS2 | Our crawl: 4,012 address histories, hashed IDs | Esplora API (mempool.space) | TxGPT training, profiles, the DS2 check |
| DS3 | Elliptic Data Set | Kaggle, CC BY-NC-ND 4.0 (not redistributed) | Classification |
| DS4 | BitcoinHeist | UCI repository 526, CC BY 4.0 | Classification |
| DS5 | FineWeb-Edu | Hugging Face, ODC-By 1.0 | 152M pretraining |
| DS6 | smol-smoltalk | Hugging Face, Apache 2.0 | Chat fine-tuning |
| DS7 | Our explanation pairs from DS1 and DS2 (v1; v2 adds DS2 coverage and a glossary; stage B3 adds 8,000 weak-case examples in `data/processed/weak_cases/`) | Built by `explorer/ds7.py` and `explorer/weak_cases.py` | Teaching and testing the Assistant |
| DS10 | A teammate's independent 27-address snapshot | Esplora API; refetched and cross-checked | Held-out test of every model |
| Random tests | Ten randomized Assistant test sets (real, crawled, made-up, balanced, close calls, extremes) | Built by `explorer/random_test.py` from DS1, DS2 and DS10 | Testing the Assistant on prompts it never saw |

Each processed folder carries a `manifest.json` (source, date, selection rule, checks, SHA-256). DS2 stores
addresses only as salted hashes; the salt never leaves the build machine. `DATA_CARD.md` describes every
dataset, file and column; `MODEL_CARD.md` every model and its metrics (both refreshed by `python tools/cards.py`);
`LICENSES.md` every licence.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `conda: command not found` | Open a new terminal after installing Miniconda (on macOS run `conda init zsh` once). |
| `conda env create` is slow or fails to solve | `conda install -n base conda-libmamba-solver` and retry, or create an empty Python 3.11 env and run `pip install` with the versions in `environment.yml`. |
| `No matching distribution found for torch==2.7.1` | An Intel Mac (section 1) or a Python other than 3.11. |
| Out of memory during `fusion`, `ds2-check` or `classify` | Lower `classification -> nJobs` in `config/settings.json` (for example to 2). |
| `UnicodeEncodeError` on Windows | `set PYTHONUTF8=1` in that terminal. |
| `CondaToSNonInteractiveError` (terms of service) | A fresh Miniconda: run the `conda tos accept` commands the message prints, or use Miniforge (section 1). |
| pytest reports collection errors under `_context/` | Run it from the repository folder so `pytest.ini` applies (it limits pytest to `tests/`). |
| macOS logs fill with `divide by zero encountered in matmul` | A harmless numpy warning on Apple Silicon; the results are unaffected. |
| `external --elliptic` asks for a token | Save a Kaggle API token as `~/.kaggle/kaggle.json`, or download the zip by hand as the message says. |
| `predict heist` or `predict elliptic` cannot find a model | Run section 5's commands for that dataset first. |
| A step says `Stopping: <step> failed` | Read `reports/logs/<step>.log`, fix the cause, and run that step alone, then `all` again. |

## 11. Reproducibility

- Versions are pinned in `environment.yml`; every random choice takes its seed from `config/settings.json`.
  Planted transactions use seeds 40000-40004 for tuning and 30049-30053 for the reported results.
- Most tables reproduce exactly. `features.csv` is written with 12 significant digits, because numpy's last-bit
  rounding differs between Apple Silicon, Intel and Linux and the regression's trees amplified that into
  different fits. Models that use several threads (gradient boosting, random forests) can still differ in the
  last decimal places; `python tools/compare_tables.py` reports any difference larger than 0.01 in a recall or a
  rate, which is worth reporting as a bug.
- `CHANGELOG.md` records every decision and measured result; `docs/AI_USE_LOG.md` records how generative AI was
  used, for the AI declaration the tutor asked for.

**Reporting a problem:** send the command, the last 30 lines of `reports/logs/<step>.log`, your operating
system and chip, and the output of `conda list --export > env.txt`.
