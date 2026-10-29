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
| Live configuration | settings and model files | stageModels = LOF C3, K-Means C4, DBSCAN C5; fusion = regression x2.0 + Isolation Forest x0.9 (the M0 procedure re-run in the fresh-clone rebuild; section 5); selection = baseline |
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

- **Fresh-clone rebuild** (tritran72 `cos30049a2-rebuild`, commit 0aded70, Kaggle Linux, Python 3.13, the pinned
  numpy/pandas/scikit-learn; 9.5 h): pytest 255 passed there. `tools/compare_tables.py`: 146 tables, 110 same,
  10 within tolerance, 26 different. Every difference was traced:
  - **Inputs are identical.** transactions.csv has the same values (its bytes differ only in line endings);
    features.csv differs only in rounding: the committed one had been written at full precision, before the
    12-significant-digit change, and the tables built from it were older still (the regression's tuning dated from
    24 Oct). Refitting one random-forest configuration on this PC gives the committed number (0.208802) on the
    24 Oct features and Kaggle's number (0.204786) on Kaggle's features: the platform is not the cause.
  - **DS1 tables (21): stale, now replaced.** The rebuild is the first set built in one run from the current code
    and the promoted stage models (the committed eval tables mixed 24, 27 and 29 Oct runs). The rebuilt models
    load and re-score scores.csv exactly on Windows. The regression's tuning now chooses different gradient-boosting
    settings for size (the random forest it uses for size is unchanged), so its scores moved (mean planted recall
    0.673 -> 0.626) and the fusion search chose a new pair: regression x2.0 + Isolation Forest x0.9 (was baseline
    x1.1 + regression x3.0), 0.682 against the baseline's 0.655, p = 0.324 (was 0.684, p = 0.063). **No conclusion
    changes:** the baseline stays the Explorer's method (fusion runner-up); on DS10 the baseline still leads the
    pooled methods; the DS2 ranking agrees with DS1 (Spearman 0.929, was 0.905).
  - **Elliptic and BitcoinHeist (5 tables): platform, kept as committed.** Only logistic regression and KNN differ
    (the tree models match to 1e-8). Refitting Elliptic's AF logistic model (C = 100) on this PC reproduces the
    committed numbers exactly; on Linux the solver stops at another point of a flat optimum (AF test F1 0.598 ->
    0.532, LF 0.726 -> 0.750). KNN's distance ties fall the other way (one BitcoinHeist threshold). The committed
    Windows results stay; README section 11 tells a rebuilder that these two models may differ by platform.
  - eda_tests.csv and the figures were regenerated from the new tables (`figures`); MODEL_CARD/DATA_CARD by
    tools/cards.py; the README quick check still gives 377 / 367 / 3 / 3 and what-if 5.98.
  - **Cross-platform check (this PC, Windows, Python 3.11):** features, split, regression, inject and fusion re-run
    from the adopted commit in a separate worktree. The regression chose the same models and settings (tables within
    0.001 of Kaggle's), and the fusion search chose the same pair and multipliers (regression x2.0 + Isolation
    Forest x0.9). A teammate's M4 rebuild should therefore match within compare_tables' tolerance.
  - Evidence: reports/logs/rebuild/ (compare.csv, compare.log, kaggle_run.json, pytest.log), reports/logs/all.log.
  - Open: the glossary (tools/make_glossary.py, DS7) still says fusion combines the baseline and regression; the
    152M writers learned that sentence. Decision for the team: correct the glossary topic for A3, or describe
    fusion without naming its pair.
- **Stage B3 gate** (weak-case examples for the 152M explanation stage): ACCEPTED. On both dev files, fully correct
  0.708 (A) -> 0.942 (B), +0.234 [+0.204, +0.266], every case higher; on DS7 v2's ordinary validation prompts
  A 0.961 and B 0.963 (round 1 0.859). The planned recipe without B3 (A) was rejected by the same gate. Round 2's
  explanation stage therefore adds the weak cases.
- **152M round 2: ADOPTED (30 Oct).** Pretraining ended at step 11,447 (validation loss 3.0014). Dev gate "B round 2":
  fully correct 0.616 -> 0.938, +0.322 [+0.287, +0.358], every case higher. The adoption rule passed every criterion
  (guardrail 1.000 on DS7 test and DS10, direction errors 2 -> 0, glossary key facts +0.258). Pooled fully correct
  over 12 sets 0.727 -> 0.948. Round 1's dev answers re-run on Kaggle were byte-identical to the committed ones (the
  evaluation is deterministic). Caution for the report: the largest gains are on the hard and max-shift sets, the
  case types B3's examples target; the examples used their own seeds and wordings and no test prompt, but the
  report should say the sets and the examples share case types.

## 6. Still open (not code)

- The Elliptic dataset citation (docs/REFERENCES.md, end).
- Canvas's upload limit, which decides whether the 152M weights (610 MB) go in the zip. The zip without them is
  76.8 MB, 567 files, and every file matches SHA256SUMS.txt.
- Team items: the blind reviewer check; a human read of the glossary; one DS1 transaction checked by hand; the M4
  rebuild by a teammate (with `tools/compare_tables.py`); minutes; the AI declaration; the contribution form.
