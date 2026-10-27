# COS30049 Assignment 2: Bitcoin Address Activity Explorer

**Current state: read `context.txt` first.** Part 1 is the live state (what is running, open decisions,
next steps), Part 2 is reference (environments, Kaggle, models, results), Part 3 is the dated history.
`docs/TODO.md` is the checklist and `CHANGELOG.md` has every change with its numbers.

This file is loaded automatically by Claude Code. It holds the project's fixed facts and rules; it was
first written on the Mac on 20 October 2026, when the work moved to the Windows PC (`docs/HANDOFF.md`
is that hand-off, now historical). Both machines now hold a clone. The full plan is
`docs/A2_delivery_plan.pdf`; its LaTeX sources in `docs/plan/sections/` are easier to search.

## The project in one paragraph

A unit project (COS30049 Computing Technology Innovation Project, Swinburne, HCMC campus) for
group **HCMC_COS30049.1_4**. The Explorer shows a Bitcoin address's transaction history and ranks
each transaction by how **unusual** it is *for that address's own past*, with a plain-language
reason. It is an investigation aid, **never a fraud detector**. Assignment 1 (submitted
11 Oct 2026) was the project plan and a React prototype. **Assignment 2 (this repository) is the
data, the machine-learning models, the code zip and a 3,000-word report.** Assignment 3
(6 Dec 2026) is the full web app (React + FastAPI) built on A2's `predict.py`.

## People, dates and status

- **The user** is Cao Tri Tran (student 106214726), the group's Data/ML Lead, and does most of
  the A2 build on the PC and the Mac with Claude's help. (Assumed from the Mac username; correct
  this line if it is wrong.)
- **Team** (A1 roles): Cam Dan Bui (105967074), leader and report integrator; Le Quang Anh Hoang
  (106218605), project manager and back end; Minh Khang Nguyen (106216803), front end.
- **Tutor's name:** Arthur Hoang (given by the user, 22 Oct 2026). It goes on the report cover.
- **Due:** Sunday 8 November 2026, 23:59 GMT+7 (Week 8, 40% of the unit). **The plan submits on
  Saturday 7 November.** Late penalty: 10% per day. Unit hurdle: at least 40% in every assignment.
- **Milestones:** M1 Sun 25 Oct (baseline, Isolation Forest and LOF score every DS1 transaction;
  A1 promised this by 26 Oct). M2 Thu 29 Oct (all models evaluated; results frozen). M3 Sat 31 Oct
  (code feature-complete). M4 Wed 4 Nov (a teammate rebuilds everything from the README; code
  freeze). M5 Sat 7 Nov (submitted).
- **Status (27 Oct):** every model is built and evaluated, `results-v1` is tagged (25 Oct), and the
  cards, licences and packaging script exist. Open: the language models' round 2 on Kaggle, the
  Qwen question (decision 4), packaging and the report. Details in `context.txt`. CR-06 (plan
  Appendix A) was amended 22 Oct, relaxed 24 Oct, and on 23 Oct the team delegated all open A2
  decisions to Cao Tri Tran.
- **The user's pace:** finish as fast as possible at the highest quality; ignore the plan's calendar
  dates, keep its content and rules.
- **Generative AI:** the brief bans it, but the tutor allowed it on condition that a separate AI
  declaration is submitted. Keep a running list of what Claude helped with, for that declaration.
- **Contribution form:** anyone under 33% has their mark scaled down (share/33 × group mark). The
  A1 form gave two members 20%. Remind the user before the A2 form is signed.

## The marks (40)

| Report (12, PDF, ≤3,000 words) | | Code and zip (20) | | HD (8) | |
|---|---|---|---|---|---|
| Data collection and processing | 5 | Core functionality | 10 | Additional datasets | 4 |
| Data analysis | 2 | Datasets | 4 | Additional ML implementations | 4 |
| Model-selection justification | 2 | README (conda set-up, training, prediction) | 2 | | |
| Writing and structure | 1 | Comments | 2 | | |
| Harvard referencing | 1 | Code structure | 2 | | |
| Completeness (all required sections) | 1 | | | | |

The brief requires **at least two of three model types (classification, clustering, regression)**
and evaluation with accuracy, precision, recall, F1 and similar metrics. The brief is in
`docs/brief/assignment2_description.pdf`, with a searchable text copy (`.txt`) beside it.

## Decisions already made (plan §4, change request CR-06)

1. **All three model types.** Clustering (K-Means, DBSCAN) and regression (Ridge, random forest,
   histogram gradient boosting with quantile loss) on our own data. Classification (logistic
   regression, KNN, random forest, histogram gradient boosting) on the labelled public datasets.
   A1's methods (robust-z baseline, Isolation Forest, LOF in novelty mode) are kept as HD extras.
2. **Datasets:**

   | ID | Dataset | Role |
   |---|---|---|
   | DS1 | Our Esplora snapshot: 25 main + 3 short-history addresses (≈5,000 tx) | Every unsupervised model; evaluation |
   | DS2 | Our Esplora crawl: ≈5,000 address histories, hashed IDs, compact rows | Trains TxGPT; address profiles |
   | DS3 | Elliptic (Weber et al. 2019), Kaggle | Classification (temporal split 1–34 / 35–49) |
   | DS4 | BitcoinHeist (UCI 526, CC BY 4.0) | Classification (temporal, random, leave-one-family-out) |
   | DS5 | FineWeb-Edu (ODC-By) | Text-model pretraining |
   | DS6 | smol-smoltalk (Apache 2.0) | Chat fine-tune |
   | DS7 | Explanation pairs built from our pipeline | Explanation fine-tune |
   | DS8 / DS9 | Elliptic++ actors / Ransomwhere (CC BY 4.0) | Optional (Could tier) |
   | DS10 | A teammate's independent DS1-style Esplora snapshot (27 addresses; added 25 Oct, `explorer/ds10.py`) | Held-out test of every model, frozen |

3. **Label-based metrics only where labels exist**: full metrics on DS3 and DS4 "against the
   dataset's labels"; recall (and precision as a lower bound) on transactions we plant in DS1;
   **never** precision, recall, F1 or accuracy for real DS1 transactions (A1 risk 13).
4. **Two language models trained from scratch** (the tutor's HD requirement; no fine-tuning of
   public models): **TxGPT**, a small GPT over transaction tokens whose per-transaction surprisal
   is an unusualness score (HD-critical); and the **Explorer Assistant**, the 152M text model in
   `lm/`, pretrained on DS5, chat-tuned on DS6, then taught on DS7 to explain flags behind a
   guardrail that falls back to the template text.
   **Added 22 Oct (tutor's OK confirmed, team approved, CR-06 amended):** a third, extra model,
   **Qwen2.5-3B-Instruct QLoRA-fine-tuned** once on DS7 mixed with 2,500 DS6 conversations (plan §9,
   `sec:assistant3b`; `lm/qlora.py`, `configs/qwen3b_explain.json`); it also serves as the chatbot.
   Changed from Llama 3.2 3B on 23 Oct at the user's request (not gated, no HF token); accepted by
   the team (23 Oct: the team delegated all open A2 decisions to Cao Tri Tran).
   **Relaxed 24 Oct:** the tutor announced (Outlook) that fine-tuning a public model is acceptable, so
   "from scratch" is no longer required; the 152M model stays (the only LM the CPU env can run) but gets no
   more GPU time, and the writer order follows the measured DS7 results (context.txt, DIRECTION).
   **Roles (user, 23 Oct):** the 152M model only explains DS1 transactions; Qwen 3B is the general
   chatbot AND the second writer: `explorer.assistant.explain` tries 152M, then Qwen, then the template,
   each answer passing the code guardrail (numbers, limitation line, wording). The 3B model is extra and doesn't replace the
   from-scratch models, runs after the 152M model and TxGPT (relaxed 24 Oct: the queue runs Qwen first), uses the same DS7 splits, measures
   (`lm/eval_explain.py`) and guardrail (`explorer/assistant.py`), and ships adapters only. Licence:
   Qwen Research License (non-commercial; ship its text and a NOTICE line; "Built with Qwen").
   **Round 2 (user, 26 Oct; supersedes "no more GPU time"):** both language models get more training on
   Kaggle (`docs/KAGGLE_LM_PLAN.md`); Qwen round 2 was adopted 27 Oct, the 152M model's continued
   pretraining is running.
   **Under review (27 Oct):** a message forwarded by the user (source not yet recorded) says Qwen, as a
   ready-made answer-generating AI, won't be accepted in the model training work; only what the group
   built and trained should be integrated. It contradicts the tutor's 24 Oct announcement. Nothing is
   removed until the user decides; `context.txt` (Part 1) lists what dropping Qwen would change.
5. **One repository, two environments**: `environment.yml` (CPU, for markers) and
   `environment-gpu.yml` (CUDA, only to retrain the language models).

## Rules that must never be broken

- **Wording.** Say "unusual", "flag", "worth a look". Never use these words for the tool or for
  DS1/DS2 addresses: fraud, criminal, illicit, guilty, suspicious (`bannedWords` in settings).
  "Illicit" and "ransomware" appear only as the labelled datasets' own class names. Every flagged
  item carries the limitation line: *"Unusual activity requires investigation; it does not
  indicate wrongdoing."*
- **Time safety (A1 DR-02).** A transaction's features use only *strictly earlier* transactions of
  the same address; same-block transactions are never each other's history. Fit on the earlier
  70% of each address's history, evaluate on the rest. Scalers live inside scikit-learn
  `Pipeline`s. Elliptic uses its temporal split. BitcoinHeist never uses `year` or `day` as
  features. DS1 addresses are excluded from DS2. Tuning plants use seeds 40000–40004 and reported
  plants use 30049–30053.
- **No magic numbers.** Every threshold, window and seed comes from `config/settings.json` (shared
  with the A1 prototype). Add new sections there rather than constants in code.
- **Libraries.** pandas, scikit-learn and matplotlib (named by the brief); PyTorch only for the
  language models. No XGBoost or LightGBM.
- **Data in the zip** is only what a final model uses; large downloads ship as scripts plus SHA-256.
  The DS2 salt (`config/salt.txt`) never leaves the machine.
- **Every reported number** is written to `reports/tables/*.csv` by code, and the report quotes
  those files, never hand-typed numbers.
- **Report:** ≤3,000 words (aim for 2,850), Harvard style (numbered, as in A1), the tutor's name
  on the cover, the brief's section order.

## Machine and environments

- **Since 26 Oct no GPU work runs locally** (the user's rule): every GPU step runs on Kaggle (two
  accounts; procedures in `context.txt`, Part 2, KAGGLE). Locally: code edits, pytest and git only.
- **PC:** Windows; Intel i5-12600K (16 threads); RTX 3060 12 GB (no longer used); 16 GB RAM.
  Repo `C:\Users\ADMIN\COS30049A2`; run from a Miniforge/Anaconda Prompt in the repository root,
  `set "PYTHONUTF8=1"` first. The Kaggle CLI and tokens live only here.
- **Mac:** Apple M5, 16 GB; clone `~/Desktop/COS30049/COS30049A2`; Miniconda is not on PATH
  (`source ~/miniconda3/etc/profile.d/conda.sh && conda activate cos30049-a2`).
- Environments: `cos30049-a2` (CPU) for everything markers will see; `cos30049-a2-gpu` (CUDA) only to
  retrain the language models. Use `pathlib` for every path. `torch.compile` stays off (decided 23 Oct).
- Round 1 of text pretraining (2.0B tokens) ran on the PC and finished 24 Oct. Every trainer resumes
  from its own checkpoint: Ctrl+C (or Kaggle's time limit) saves, the same command resumes.

## Repository

The plan's layout (§12.2), as built (27 Oct):

```
explorer/        one module per pipeline step (snapshot, corpus, external, transactions, features, split,
                 inject, baseline, clustering, regression, outliers, fusion, classify, evaluate,
                 significance, sensitivity, walkforward, ds2_check, ds10, distribution, reviewer,
                 predict, explain, assistant, ds7, random_test, profiles, plots, ...); one CLI:
                 python -m explorer.cli <step>
lm/              language models: the 152M text model (model, tokenizer, data, train, chat, generate),
                 TxGPT (tx_tokens, tx_score, tx_select), Qwen QLoRA (qlora), evaluation
                 (eval_explain, eval_stats, writers), two-GPU training (dist)
configs/         training configs (text_150m_*, txgpt_s/m/l, qwen3b_explain*, queue_after_pretraining)
config/          settings.json: every threshold, window and seed
kaggle/          the Kaggle notebooks (regenerated, pinned, by tools/kaggle_notebooks.py)
tools/           status, supervise, run_queue, cards, package, test_report, make_glossary, kaggle_notebooks
tests/           pytest suite (219 passed, 3 skipped on 27 Oct)
data/ models/ reports/{figures,tables,logs}/   committed outputs (large files git-ignored)
notebooks/       01_eda.ipynb (only calls the package)
reference/       the A1 prototype's TypeScript (scoring and JSON shapes predict.py follows)
docs/            plan (PDF + LaTeX), brief, A1 LaTeX, TODO, KAGGLE_LM_PLAN, REPORT_PLAN, AI_USE_LOG,
                 LM_TRAINING (the old LM guide), HANDOFF (historical)
README.md, DATA_CARD.md, MODEL_CARD.md, LICENSES.md, NOTICE, CHANGELOG.md, environment*.yml
_context/        course material and notes (git-ignored; see below)
```

Plan Appendix D lists every command in run order; README.md is the current guide.

## Where the plan covers each topic (`docs/plan/sections/`)

| File | Topic |
|---|---|
| s00-summary | the plan on one page |
| s01-deliverables | files to submit; rubric line → what earns it; A1 promises to keep |
| s03-decisions | D-A2-1 to D-A2-6, dataset table, metrics rule |
| s04-compute | hardware, FLOPs estimate, pretraining gate, per-job time budget |
| s05-data | DS1 selection rule and checks, DS2 crawl, DS3–DS7 |
| s06-processing | cleaning, F1–F8, robust z, time split, injection, DS3/DS4 processing, TxGPT tokens, leakage list, tests |
| s07-analysis | EDA questions → plots → statistics → decisions |
| s08-models | selection criteria, candidates table, every model's grid and protocol |
| s09-lm | TxGPT and the Explorer Assistant |
| s10-evaluation | label-free checks, labelled metrics, method selection, objectives table |
| s11-code | predict.py, code rules, README outline, zip contents |
| s12-report | report outline with word budget, writing rules, completeness checklist |
| s13-schedule | day-by-day schedule and Gantt chart |
| s14-risks | risk register, priority list (Must/Should/Could/Won't), submission checklist |
| a-cr06 … d-commands | CR-06 and messages; datasets considered; references; command list |

## `_context/` (git-ignored, read as needed)

- `brief/`: the A2 brief PDF. `a1_submission/`: the A1 report, minutes and contribution form as
  submitted.
- `text/`: **plain-text extractions** of the brief, the A1 report and minutes, and the week 1–8
  slides (including the Week 6 `.pptx`). Search these first.
- `slides/week1–8/`: original slides and workshop notebooks. Weeks 4–7 are the ML half; the
  notebooks show the code style the tutor expects.
- `a1_working/`: A1's LaTeX (`main.tex`, `a1_latex_sections/`; reuse its preamble for the A2
  report, minus xltabular, pdflscape, csquotes and seqsplit), the risk register CSV, the handover
  audit (open team actions T1–T10), and the minutes export.
- `a1_prototype/`: the A1 repository without node_modules (React prototype, D1 check report).
- `notes/`: Claude's saved notes from earlier Mac sessions.

## How to work here

- Commit after each working step, with clear messages. Tag `results-v1` at M2 and `v1.0` at M4.
- **Commit identity:** author `Defaul119 <231157022+Defaul119@users.noreply.github.com>` (already
  set in this repository's git config). **Never add a `Co-Authored-By: Claude` line** or any other
  AI attribution to commit messages or PR descriptions; the user asked for this. AI use is declared
  in the separate AI declaration instead.
- The remote is https://github.com/Defaul119/COS30049A2 (public, by the user's choice). Never
  commit `_context/`, `config/salt.txt`, `kaggle.json` or large data.
- `python -m pytest -q` must pass before every commit (it also runs the unittest-style tests). Check
  its exit code; never pipe it into `tail` before a commit (that hid a failure on 26 Oct).
- Push after each finished batch of work (the user asked, 23 Oct). Keep replies short and answer the
  user's questions first.
- Write code like `explorer/esplora.py` and `lm/train.py`: module docstrings that say what the
  module does and which requirement it serves (e.g. "DR-02"), docstrings with units, and comments
  that explain *why*. Type hints; no dead code.
- `python tools/status.py` shows every long job at once (crawl progress and ETA, pretraining, GPU).
- Long jobs that run unattended go under `python tools/supervise.py --name <job> -- <command>`, which
  restarts them after a crash (every job here resumes from its own saved state).
- Long jobs (pretraining, the DS2 crawl, Elliptic tuning) run in their own terminal so they
  survive the Claude session; check on them by reading their logs.
- When the plan and reality disagree (a measured speed, a licence, a result), follow reality,
  note it in `CHANGELOG.md`, and tell the user. Don't silently change a decision in CR-06.
