# Hand-off: Mac → Windows PC (20 October 2026)

Read `CLAUDE.md` first, then this file. This is what the first session on the PC should do.

## What happened before this hand-off

- **15 Oct.** The text language model pipeline (`lm/`) was built and tested on the Mac's CPU:
  15 unit tests and the 8-stage smoke test pass, and Ctrl+C saving and resume are verified. It
  has never run on a GPU.
- **20 Oct.** Every source was read: the week 1–8 slides, the A1 report, minutes and contribution
  form, the A2 brief, and the A1 prototype. The 64-page delivery plan was written
  (`docs/A2_delivery_plan.pdf`). This repository was assembled: the A1 `explorer/` code and
  `settings.json` copied in, and the environment split into CPU (`environment.yml`) and GPU
  (`environment-gpu.yml`).
- **Not started:** every A2 deliverable. DS1 is about a week behind A1's schedule, so it comes first.

## First PC session, step by step

### 0. Where to put it

Unzip to a short path that isn't synced by OneDrive, for example `C:\cos30049\asm2`. OneDrive
and long paths both break large data folders. Then, in an **Anaconda Prompt**:

```
cd C:\cos30049\asm2
set PYTHONUTF8=1
claude
```

### 1. Put it on GitHub (recommended)

The Mac had no GitHub login, so this repository has local commits but no remote. If the PC has
git and a GitHub login: create a **private** repository (for example `cos30049-a2`), add it as
`origin`, and push `main`. `_context/` is git-ignored, so course material stays local.
Teammates need access for their review tasks.

### 2. Environments and checks (about 20 minutes)

```
conda env create -f environment-gpu.yml
conda env create -f environment.yml
conda activate cos30049-a2-gpu
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest -q
python -m lm.smoke
```

Expected: `True NVIDIA GeForce RTX 3060`, all tests passing, `8/8 stages passed`. Also check free
disk space (needs at least 40 GB).

### 3. GPU lane: start the long job first (its own terminal)

Before starting: Windows Update paused for two weeks, sleep set to Never, High-performance power
plan. Then, in a **separate** Anaconda Prompt with `cos30049-a2-gpu` active:

```
python -m lm.prepare_text --delete-raw          # about 6.5 GB download, 1.5–3 h; resumable
python -m lm.train --config configs/text_150m_pretrain.json --bench
```

Apply the gate (plan §5.3) to the benchmark result:
- 2B tokens in 72 h or less: keep `total_tokens` at 2,000,000,000.
- Otherwise set `total_tokens` = 60 × 3600 × measured tokens/s, but never below 1B.
- Set `micro_batch` to the benchmark's recommendation.
- Record the measured speed and the decision in `CHANGELOG.md`.

Then start it and leave it running:

```
python -m lm.train --config configs/text_150m_pretrain.json
```

Watch it twice a day: validation loss should keep falling from about 10.4, and the GPU should
stay below about 83 °C. Logs are in `runs/text-150m/log.jsonl`. It gets paused (Ctrl+C) on
26 Oct, when TxGPT needs the GPU.

### 4. CPU/network lane: build DS1 (the critical path)

Build `explorer/snapshot.py` and `explorer/cli.py` (`python -m explorer.cli snapshot`) to the
spec in plan §6.1 (`docs/plan/sections/s05-data.tex`). In short:

1. **Write the selection rule into the manifest before fetching anything** (A1 risk 19).
   Parameters come from `settings.json` → `collection` (seed 30049; main addresses need 50–1,000
   transactions and a span of at least 90 days, target 25; short-history addresses need 3–9,
   target 3).
2. Pick 20 block heights spread evenly between tip − 4,000 and tip − 100. Take output addresses
   from up to 4 pages of each block's transactions (`/block/<hash>/txs/<start>`, 25 per page;
   get hashes with `/block-height/<h>`, a small method to add to `EsploraClient`). Dedupe, record
   script types, shuffle with the seed.
3. Visit candidates in order with `/address/<a>` (`chain_stats.tx_count`). Log every candidate
   with the reason it was kept or skipped.
4. Fetch each kept history with `EsploraClient.address_history` (cursor paging, 0.5 s pace,
   back-off). Save every raw page unchanged as `data/raw/esplora/<address>/page-NNN.json.gz`,
   and read from that cache on re-runs unless `--refetch` is given. Re-check `chain_stats` after
   fetching, as `scripts/d1_check.py` does, in case a block lands mid-fetch.
5. Verify automatically: the count equals `tx_count`; no duplicate txids; every transaction is
   confirmed and has a block time; summed received equals `funded_txo_sum` and summed sent equals
   `spent_txo_sum` for every address; and for 2 addresses, the first page matches mempool.space.
6. Write `data/processed/esplora/manifest.json` with every DR-01 field (plan §6.1): source,
   timestamps, pace, request statistics, tip height, rule and seed, kept and skipped candidates
   with reasons, fields, missing-value counts, raw page paths with SHA-256, the terms note, and
   rebuild steps.

Expect about 28 addresses and roughly 5,000 transactions, in 0.5–1.5 hours of fetching.

The same day, if time allows: `python -m explorer.cli external --elliptic --bitcoinheist`.
BitcoinHeist is a direct UCI download. Elliptic needs a Kaggle API token in
`%USERPROFILE%\.kaggle\kaggle.json`, which the user creates. Record SHA-256 and the licence shown
on each page.

### 5. Then follow the schedule

Plan §14 (`s13-schedule.tex`) gives every day to 8 Nov, and §15.3 says what to cut if a day
slips. The next days:

- **22 Oct:** `transactions.py`; `features.py` as a line-by-line port of
  `reference/a1_prototype/scoring.ts` with a parity test; `split.py`; the DR-02 tests. Start the
  DS2 crawl overnight.
- **23 Oct:** EDA notebook for DS1; `inject.py` and its tests.
- **24 Oct:** DS3 and DS4 processing and analysis; `lm/prepare_tx.py` for DS2 tokens.
- **25 Oct:** baseline with threshold sweep, Isolation Forest and LOF. That is **M1**.

## Waiting on the user or the team

- [ ] Send the CR-06 team message and the tutor message (plan Appendix A). Record the answers in
      `CHANGELOG.md` and the minutes.
- [ ] The tutor's full name, for the report cover.
- [ ] A Kaggle API token, for Elliptic.
- [ ] Teammate tasks: Cam Dan Bui reviews the DS1 manifest and checks one transaction by hand
      (22 Oct); Le Quang Anh Hoang reviews `features.py`, `split.py` and `inject.py` (24–25 Oct);
      reviewer ratings from Minh Khang Nguyen and Cam Dan Bui (29 Oct – 2 Nov); clean install
      (4 Nov); proofread (5 Nov).
- [ ] Minutes for Weeks 1–8 and the contribution form (the team does these separately).

## Small corrections to the plan, found while packing

- `lm.train` has no `--seed` option yet. TxGPT's 3 seeds × 3 sizes (plan §10.1) need it added,
  or one config per seed.
- `lm/hub.py` downloads datasets only. The plan's mention of downloading released model weights
  needs its own small script.
- Training logs are `runs/<name>/log.jsonl` (already written by `train.py`), not `metrics.csv`.
- `README.md` is still the language-model guide. Plan §12.3 gives the outline of the A2 README
  that replaces it; keep the GPU steps as a section.
