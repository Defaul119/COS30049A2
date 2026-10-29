# COS30049 Assignment 2: Bitcoin Address Activity Explorer

**Read `context.txt` first**: Part 1 is the live state (what is running, next steps, what waits on the team),
Part 2 is reference (Kaggle, stages, results), Part 3 is the history. `docs/TODO.md` is the checklist,
`CHANGELOG.md` has every change with its numbers, `docs/TRAINING_PLAN.md` the training stages and their gate.
This file holds the fixed facts and rules. The full plan is `docs/A2_delivery_plan.pdf` (LaTeX sources in
`docs/plan/sections/` are easier to search).

## The project

- Unit COS30049 Computing Technology Innovation Project (Swinburne, HCMC campus), group **HCMC_COS30049.1_4**.
- The Explorer shows a Bitcoin address's transaction history and ranks each transaction by how **unusual** it is
  *for that address's own past*, with a plain-language reason. An investigation aid, **never a fraud detector**.
- A1 (submitted 11 Oct): plan + React prototype. **A2 (this repository): data, ML models, code zip, 3,000-word
  report.** A3 (6 Dec): the full web app (React + FastAPI) on A2's `predict.py`.

## People and dates

- **The user:** Cao Tri Tran (106214726), Data/ML Lead; builds A2 on the PC and the Mac with Claude. The team
  delegated all open A2 decisions to him (23 Oct).
- **Team:** Cam Dan Bui (105967074), leader, report integrator; Le Quang Anh Hoang (106218605), PM, back end;
  Minh Khang Nguyen (106216803), front end. **Tutor:** Arthur Hoang (on the report cover).
- **Due Sun 8 Nov 2026, 23:59 GMT+7** (40% of the unit); **the team submits Sat 7 Nov**. Late: -10% per day.
  Hurdle: 40% in every assignment.
- **Key dates left:** results freeze **Tue 3 Nov** (tag `results-v1.1`); M4 **Wed 4 Nov** (a teammate rebuilds
  from the README; code freeze; tag `v1.0`); submit Sat 7 Nov.
- **Pace:** as fast as possible at the highest quality; ignore the plan's calendar, keep its content and rules.
- **Generative AI:** allowed by the tutor if a separate AI declaration is submitted; `docs/AI_USE_LOG.md` is its
  source; keep it current.
- **Contribution form:** anyone under 33% is scaled down (share/33 x group mark); A1 gave two members 20%. Remind
  the user before the A2 form is signed.

## The marks (40)

| Report (12, PDF, ≤3,000 words) | | Code and zip (20) | | HD (8) | |
|---|---|---|---|---|---|
| Data collection and processing | 5 | Core functionality | 10 | Additional datasets | 4 |
| Data analysis | 2 | Datasets | 4 | Additional ML implementations | 4 |
| Model-selection justification | 2 | README (conda set-up, training, prediction) | 2 | | |
| Writing and structure | 1 | Comments | 2 | | |
| Harvard referencing | 1 | Code structure | 2 | | |
| Completeness (all required sections) | 1 | | | | |

- The brief needs **at least two of classification, clustering, regression**, evaluated with accuracy,
  precision, recall, F1 and similar. Brief: `docs/brief/assignment2_description.pdf` (+ `.txt`).

## Decisions (plan §4, change request CR-06)

1. **All three model types.** Clustering (K-Means, DBSCAN) and regression (Ridge, random forest, histogram
   gradient boosting with quantile loss) on our data; classification (logistic regression, KNN, random forest,
   histogram gradient boosting) on the labelled datasets; A1's robust-z baseline, Isolation Forest and LOF (novelty
   mode) kept as HD extras.
2. **Datasets:**

   | ID | Dataset | Role |
   |---|---|---|
   | DS1 | Our Esplora snapshot: 25 main + 3 short-history addresses (≈5,000 tx) | Every unsupervised model; evaluation |
   | DS2 | Our Esplora crawl: 4,012 address histories, hashed IDs | Trains TxGPT; address profiles; transfer check |
   | DS3 | Elliptic (Weber et al. 2019), Kaggle | Classification (temporal split 1–34 / 35–49) |
   | DS4 | BitcoinHeist (UCI 526, CC BY 4.0) | Classification (temporal, random, leave-one-family-out) |
   | DS5 | FineWeb-Edu (ODC-By) | 152M pretraining |
   | DS6 | smol-smoltalk (Apache 2.0) | 152M chat stage (tutor accepted, 27 Oct) |
   | DS7 | Explanation pairs from our pipeline (+ glossary) | 152M explanation stage |
   | DS8 / DS9 | Elliptic++ actors / Ransomwhere (CC BY 4.0) | Optional (Could tier) |
   | DS10 | A teammate's independent DS1-style snapshot (27 addresses) | Held-out final test of every model, frozen |

3. **Label-based metrics only where labels exist:** full metrics on DS3/DS4; recall (precision as a lower bound)
   on transactions we plant in DS1; **never** precision, recall, F1 or accuracy for real DS1 transactions.
4. **Only models the team trains from scratch** (the tutor's ruling, 27 Oct). **Never fine-tune a pretrained
   model; never use a ready-made generator.** Two language models:
   - **TxGPT**: a small GPT over transaction tokens; per-transaction surprisal = unusualness score (HD-critical).
   - **Explorer Assistant**: the 152M text model in `lm/`, pretrained on DS5, chat-tuned on DS6, taught on DS7 to
     explain flags. Every answer passes the code guardrail (numbers, limitation line, wording, comparison
     direction, verdict) or the template text is shown.
   - Round 2 (26 Oct): more pretraining + the stages again on Kaggle (`docs/KAGGLE_LM_PLAN.md`); round 2 replaces
     round 1 only by the adoption rule there.
   - Qwen (a QLoRA fine-tune, 22-27 Oct) was dropped; its work is on branch `qwen-archive`. **The report never
     mentions Qwen** (tutor, 27 Oct).
5. **Training stages** (`docs/TRAINING_PLAN.md`, 27 Oct): every model retrained stage by stage on dev data, each
   stage through a strict gate fixed in advance and compared with milestone M0 (tag `m0`); the test tier once per
   frozen candidate, DS10 once at the end. Every stage writes a row to `reports/tables/stages.csv`.
6. **One repository, two environments:** `environment.yml` (CPU, for markers) and `environment-gpu.yml` (CUDA,
   only to retrain the language models).

## Rules that must never be broken

- **Wording.** Say "unusual", "flag", "worth a look". Never use fraud, criminal, illicit, guilty or suspicious for
  the tool or for DS1/DS2/DS10 addresses (`bannedWords` in settings). "Illicit" and "ransomware" appear only as
  the labelled datasets' own class names. Every flagged item carries the limitation line: *"Unusual activity
  requires investigation; it does not indicate wrongdoing."*
- **Time safety (A1 DR-02).** Features use only *strictly earlier* transactions of the same address; same-block
  transactions are never each other's history. Fit on the earlier 70% of each history, evaluate on the rest.
  Scalers live inside scikit-learn `Pipeline`s. Elliptic uses its temporal split. BitcoinHeist never uses `year`
  or `day`. DS1 addresses are excluded from DS2.
- **Seeds and tiers.** Tuning plants 40000–40004 (stages: search 40000–40002, gate 40003–40004); reported plants
  30049–30053. Never tune on test or final data; test files are checked against `data/processed/frozen_tests.json`.
- **No magic numbers.** Every threshold, window, grid and seed lives in `config/settings.json`.
- **Libraries.** pandas, scikit-learn, matplotlib; PyTorch only for the language models. No XGBoost or LightGBM.
- **Data in the zip** is only what a final model uses; large downloads ship as scripts + SHA-256. The DS2 salt
  (`config/salt.txt`) never leaves the machine.
- **Every reported number** is written to `reports/tables/*.csv` by code; the report quotes those files only.
- **Report:** ≤3,000 words (aim 2,850), Harvard numbered (as in A1), the tutor's name on the cover, the brief's
  section order.

## Machines and environments

- **No GPU work runs locally** (the user, 26 Oct): every GPU or long CPU run is a Kaggle notebook. Locally: code
  edits, pytest, git and the Kaggle CLI. Kaggle accounts, quotas and push procedure: `context.txt` Part 2.
- **PC:** Windows 11, i5-12600K, 16 GB RAM (RTX 3060 unused). Repo `C:\Users\ADMIN\COS30049A2`; Miniforge
  Prompt in the repo root, `set "PYTHONUTF8=1"` first. The Kaggle CLI and tokens live only here.
- **Mac:** Apple M5, 16 GB; clone `~/Desktop/COS30049/COS30049A2`;
  `source ~/miniconda3/etc/profile.d/conda.sh && conda activate cos30049-a2`.
- Envs: `cos30049-a2` (CPU) for everything markers see; `cos30049-a2-gpu` (CUDA) only for the language models and
  the Kaggle CLI. Use `pathlib` for every path. `torch.compile` stays off.
- Every trainer resumes from its own checkpoint: Ctrl+C or Kaggle's time limit saves; the same command resumes.

## Repository

```
explorer/        one module per pipeline step; one CLI: python -m explorer.cli <step>
                 data: snapshot, corpus, external, transactions, features, split, inject, ds10, ds7, random_test
                 models: baseline, clustering, regression, outliers, fusion, classify, ensemble
                 evaluation: evaluate, significance, sensitivity, walkforward, ds2_check, distribution, reviewer
                 stages: stages (the gate, test tier, writer gate), stage_search (C1-C7, C9), weak_cases (B3)
                 use: predict, explain, assistant, profiles, plots
lm/              152M text model (model, tokenizer, data, train, chat, generate), TxGPT (tx_tokens, tx_score,
                 tx_select), evaluation (eval_explain, eval_stats, writers), two-GPU training (dist)
configs/         training configs (text_150m_*, txgpt_s/m/l)
config/          settings.json: every threshold, window, grid and seed
kaggle/          Kaggle notebooks, pinned to a commit by tools/kaggle_notebooks.py
tools/           status, supervise, run_queue, cards, package, test_report, make_glossary, kaggle_notebooks
tests/           pytest suite (255 passed on 29 Oct)
data/ models/ reports/{figures,tables,logs}/   committed outputs (large files git-ignored)
notebooks/       01_eda.ipynb (only calls the package)
reference/       the A1 prototype's TypeScript (scoring and JSON shapes predict.py follows)
docs/            plan, brief, A1 LaTeX, TODO, TRAINING_PLAN, KAGGLE_LM_PLAN, REPORT_PLAN, AI_USE_LOG,
                 archive/ (LM_TRAINING: the round-1 LM guide; HANDOFF: the 20 Oct Mac -> PC hand-off)
README.md, DATA_CARD.md, MODEL_CARD.md, LICENSES.md, CHANGELOG.md, environment*.yml
_context/        course material and notes (git-ignored)
```

README.md is the current run guide; plan Appendix D lists the original command order.

## Where the plan covers each topic (`docs/plan/sections/`)

| File | Topic |
|---|---|
| s00-summary | the plan on one page |
| s01-deliverables | files to submit; rubric line → what earns it; A1 promises to keep |
| s03-decisions | D-A2-1 to D-A2-6, dataset table, metrics rule |
| s04-compute | hardware, FLOPs estimate, pretraining gate, per-job time budget |
| s05-data | DS1 selection rule and checks, DS2 crawl, DS3–DS7 |
| s06-processing | cleaning, F1–F8, robust z, time split, injection, DS3/DS4 processing, TxGPT tokens, leakage |
| s07-analysis | EDA questions → plots → statistics → decisions |
| s08-models | selection criteria, candidates, every model's grid and protocol |
| s09-lm | TxGPT and the Explorer Assistant |
| s10-evaluation | label-free checks, labelled metrics, method selection, objectives |
| s11-code | predict.py, code rules, README outline, zip contents |
| s12-report | report outline with word budget, writing rules, completeness checklist |
| s13-schedule | schedule and Gantt chart |
| s14-risks | risk register, priority list (Must/Should/Could/Won't), submission checklist |
| a-cr06 … d-commands | CR-06 and messages; datasets considered; references; command list |

## `_context/` (git-ignored, read as needed)

- `brief/`, `a1_submission/` (A1 report, minutes, contribution form), `text/` (**plain-text extractions** of the
  brief, A1 report and minutes, week 1–8 slides: search these first), `slides/week1–8/` (weeks 4–7 = the ML
  half; their notebooks show the tutor's code style), `a1_working/` (A1 LaTeX: reuse its preamble minus
  xltabular, pdflscape, csquotes, seqsplit; risk register; handover audit T1–T10; minutes), `a1_prototype/`,
  `notes/`. The PC's copy is partial; the full set is on the Mac.

## How to work here

- Work on branch `from-scratch`. Commit after each working step with a clear message; push after each finished
  batch. Never move a pushed tag.
- **Commit identity:** `Defaul119 <231157022+Defaul119@users.noreply.github.com>` (set in the repo's git config).
  **Never add `Co-Authored-By: Claude`** or any AI attribution to commits or PRs (the user's rule; AI use goes in
  the AI declaration).
- Remote https://github.com/Defaul119/COS30049A2 (public). Never commit `_context/`, `config/salt.txt`, Kaggle
  tokens or large data.
- `python -m pytest -q` must pass before every commit: check its **exit code**; never pipe it into `tail`.
- Code style as in `explorer/esplora.py` and `lm/train.py`: module docstrings saying what the module does and which
  requirement it serves (e.g. "DR-02"), docstrings with units, comments that explain *why*, type hints, no dead
  code.
- Long jobs never run as children of Claude's tool shell: their own window, or
  `python tools/supervise.py --name <job> -- <command>`. Check background jobs at the start of every session.
- When the plan and reality disagree, follow reality, note it in `CHANGELOG.md`, and tell the user. Never silently
  change a CR-06 decision.
- Keep replies short and answer the user's questions first.
