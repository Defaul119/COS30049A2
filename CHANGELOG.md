# Change log

Changes to settings, data, decisions and the plan, newest first. Where the plan and reality
disagree, the measured reality is recorded here (CLAUDE.md, "How to work here").

## 22 October 2026

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
