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
  the 1.5–3 h the plan estimated. No decision changes.
