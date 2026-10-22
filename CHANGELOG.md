# Change log

Changes to settings, data, decisions and the plan, newest first. Where the plan and reality
disagree, the measured reality is recorded here (CLAUDE.md, "How to work here").

## 23 October 2026

- **Assistant 3B base model: Qwen2.5-3B instead of Llama 3.2 3B** (user's decision). Qwen is not
  gated, so no Hugging Face token is needed; the plan already named it as an option. Base model
  `Qwen/Qwen2.5-3B` at revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`. Licence: Qwen Research
  License (non-commercial, research or evaluation use; derivatives must ship the licence text, a
  NOTICE line "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba
  Cloud. All Rights Reserved.", and "Built with Qwen" in the documentation). **The team approved
  Llama specifically, so it must be told.** The base (not Instruct) model is used so that its two
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
  pause allows. The data collected is unchanged.

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
