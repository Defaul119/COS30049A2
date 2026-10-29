# Project audit, 29 October 2026

The user asked (29 Oct): "re check everything, make sure it's all perfected ... recheck all the code, models,
testing and take note carefully, then wrap up the project." This file is the record of that check. It covers what
was checked, how, what was found, what was fixed and what is still open. CHANGELOG.md has the same changes in
date order.

## 1. Code

| Check | How | Found | Done |
|---|---|---|---|
| Static errors | pyflakes 3.2 over explorer/, lm/, tools/, scripts/, tests/ and every Kaggle notebook's code | 4 unused imports; an undefined name in a string annotation; the notebooks import `threading` in a shared setup cell that some don't use | Imports removed; annotation removed; the notebook import left (harmless). pyflakes is clean |
| Test-file structure | read every test file | 5 files called `unittest.main()` mid-file, so running them as scripts skipped the later classes (pytest was unaffected) | Guard moved to each file's end |
| Reproducibility trap | traced what `all --offline` rebuilds after the stage promotion | the `fusion` step searched over the promoted methods, so a fresh rebuild could choose a fusion other than M0's, which stage C9 kept | `fusion` searches over the M0 models (`saved_methods(promoted=False)`); test added |
| Wasted work | read today's TxGPT change | the event-level columns of the rejected stage C7 were computed by every pipeline run | Only the C7 search computes them |
| Cross-machine results | the Mac replication's open item | features.csv differs in the last bits between Apple Silicon, Intel and Linux; the regression's trees amplified it | features.csv is written with 12 significant digits (settings `features.csvSignificantDigits`) |
| Dependencies | searched every import | seaborn was installed by both environments but used nowhere | Removed from environment.yml, environment-gpu.yml, requirements.txt and LICENSES.md |
| Documentation | AST scan of public functions and classes | 90 long public functions or classes without a docstring, among them every pipeline step | Docstrings for the 49 a reader meets first (every `run_<step>` says what it reads and writes; the 152M model's classes) |
| CLI and README | compared the CLI with README sections 4, 6 and 11 | new commands undocumented; "git status" as the rebuild check | Stage commands documented; `tools/compare_tables.py` is the rebuild check |
| Writer statistics | read lm/eval_stats.py | the glossary row reported the bootstrap mean difference, every other row the observed one | Observed difference everywhere; test added |
| Zip contents | built a trial zip from HEAD | today's clean-up would have left the 152M retraining guide out | Only the hand-off note is left out; test added |
| Model-selection evidence | TRAINING_PLAN section 6 against the figures | no figure showed the stage decisions | New figure `eval_stages`: every gate row of stages.csv with its interval |
| Data card | compared DATA_CARD.md's file list with the shipped data | the weak-case files (stage B3) were not described | Listed with their columns (`tools/cards.py`) |

## 2. Models and results

| Check | How | Result |
|---|---|---|
| Saved models match the tables | loaded every saved method and re-scored the committed features | all 9 methods reproduce eval_sanity.csv's flag rates and scores.csv's scores exactly |
| Live configuration | settings and model files | stageModels = LOF C3, K-Means C4, DBSCAN C5; fusion = baseline x1.1 + regression x3.0 (M0); selection = baseline |
| Core functionality (README section 3) | ran every command | totals 377 / 367 / 3 / 3; unknown address exit 2; what-if 5.98 (F1 z 5.88); template explanation; DS4 prediction: all as documented |
| Training stages | stages.csv | A0, C1, C2, C6 regression, C7, C9, D6 rejected; C3-C5 kept on dev and on the test tier; C6's range variant kept on dev, rejected on test |
| MODEL_CARD.md | read against the live models | it described the first-round LOF, K-Means and DBSCAN; it now names the stage winners, their files and settings |
| Fresh-clone rebuild | Kaggle notebook `rebuild`: README sections 2-4 on a clean machine, then `tools/compare_tables.py` | see section 5 |

## 3. Tests

- Suite: 260 passed (was 252 this morning); the adoption rule is now code too (lm/adopt.py). New: the fusion search over M0 models, a stage winner's refit, the C9
  candidates' arithmetic, event-level scores, the comparison tool, the writer statistics, every Kaggle notebook
  compiling and pinned to a full commit, a dev row judged against its reference reply, the zip's contents.
- Modules no test touches: walk-forward (it reuses the tested refit rules), the interactive chat, the smoke test,
  the status tool, the glossary generator, the test-report PDF builder. These are scripts or interactive tools.
- The rebuild notebook also runs the suite on Kaggle's Linux image.

## 4. References

`docs/REFERENCES.md` is the report's list. Every entry is from 2020 or later (the user's rule: no more than 7 years
old). Each was checked against its source: Crossref for 30 DOIs, Zenodo for the software releases, arXiv for the
preprints. Corrections found while checking:
- the NeurIPS title of Hoffmann et al. (2022), and the journal title of Liang et al. (2024);
- the pages of Le and Zhang (2022), and the issue of Wu and Keogh (2023);
- the issue years of the two ACM Computing Surveys papers.

The 152M architecture was checked against LLaMA's section 2, and the Elliptic step-43 collapse against EvolveGCN's
text. Pareja et al. (2020) replaces Weber et al. (2019) in the figure note. One decision is left to the team: how to
cite the Elliptic dataset record (2019).

## 5. Results of the long checks (filled in as they finish)

- **Fresh-clone rebuild** (tritran72 `cos30049a2-rebuild`, commit 0aded70): pending.
- **Stage B3 gate** (weak-case examples for the 152M explanation stage): ACCEPTED. On both dev files, fully correct
  0.708 (A) -> 0.942 (B), +0.234 [+0.204, +0.266], every case higher; on DS7 v2's ordinary validation prompts
  A 0.961 and B 0.963 (round 1 0.859). The planned recipe without B3 (A) was rejected by the same gate. Round 2's
  explanation stage therefore adds the weak cases.
- **152M round 2**: pretraining session 4 running (ends about 01:30 Fri); the chat and explanation stages follow on
  the same account, then the adoption rule (docs/KAGGLE_LM_PLAN.md section 3).

## 6. Still open (not code)

- The Elliptic dataset citation (docs/REFERENCES.md, end).
- Canvas's upload limit, which decides whether the 152M weights (610 MB) go in the zip. The zip without them is
  76.8 MB, 567 files, and every file matches SHA256SUMS.txt.
- Team items: the blind reviewer check; a human read of the glossary; one DS1 transaction checked by hand; the M4
  rebuild by a teammate (with `tools/compare_tables.py`); minutes; the AI declaration; the contribution form.
