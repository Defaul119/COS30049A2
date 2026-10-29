# Data card

Every dataset this project uses: where it came from, its licence, when and how it was collected, its size,
missing values, known biases and which models use it. The tables at the end (**Files and columns**) are written
by `python tools/cards.py` from the files themselves; every other number here is copied from a manifest named
beside it. The Explorer is an investigation aid, never a fraud detector: *unusual activity requires
investigation; it does not indicate wrongdoing.*

| ID | Dataset | Collected | Size | Licence | Used by |
|---|---|---|---|---|---|
| DS1 | Our Esplora snapshot: 25 main + 3 short-history addresses | 22 Oct 2026 | 7,966 transactions | public blockchain data | every unsupervised method, the evaluation, prediction |
| DS2 | Our Esplora crawl: 4,012 address histories, hashed IDs | 21-24 Oct 2026 | 898,490 rows | public blockchain data | TxGPT training, address profiles, the DS2 check, DS7 |
| DS3 | Elliptic Data Set (Weber et al., 2019) | 22 Oct 2026 | 203,769 transactions | CC BY-NC-ND 4.0 (not redistributed) | classification |
| DS4 | BitcoinHeist (Akcora et al., 2020; UCI 526) | 22 Oct 2026 | 2,916,697 rows | CC BY 4.0 | classification |
| DS5 | FineWeb-Edu, 10B-token sample | 22 Oct 2026 (round 2: 26 Oct) | 2.0B tokens (round 2: +3.0B) | ODC-By 1.0 | 152M model pretraining |
| DS6 | smol-smoltalk | 24 Oct 2026 | 450,584 conversations | Apache 2.0 | chat stage (152M) |
| DS7 | Our explanation pairs from DS1 and DS2 | 24 Oct 2026 (v2: 26 Oct) | 10,160 pairs (v2: 40,232 + 74) | ours | teaching and testing the Assistants |
| DS10 | A teammate's independent DS1-style snapshot | 23 Oct 2026 (refetched 25 Oct) | 27 addresses, 5,785 transactions | public blockchain data | held-out test of every model |
| Random tests | Ten randomized Assistant test sets | 26 Oct 2026 | 4,212 prompts | ours | testing the Assistants |

DS8 (Elliptic++) and DS9 (Ransomwhere) were optional in the plan and were not used.

## DS1: Esplora snapshot

- **Source.** Blockstream's Esplora API (`https://blockstream.info/api`), falling back to the Esplora-compatible
  mempool.space when rate-limited; each address records which host served it. Manifest:
  `data/processed/esplora/manifest.json`.
- **Terms.** The API documentation states no terms of use; Blockstream caps unauthenticated use at 700 requests an
  hour. We sent at most two GET requests a second per host. Addresses are public data; nothing here says who
  controls an address or why it transacts.
- **Collection** (rule recorded before fetching, seed 30049). Twenty block heights spread evenly over about four
  weeks (tip - 4,000 to tip - 100, tip 972,615); the output addresses of up to four pages of transactions per
  block (5,995 candidates); candidates shuffled and visited in order. Main addresses need 50-1,000 confirmed
  transactions over at least 90 days (25 kept); short-history addresses need 3-9 (3 kept). An address is kept only
  if every automated check passes (the count matches the API's own count, received and sent sum to the API's
  totals, no duplicate or unconfirmed transactions, every transaction involves the address).
- **Size.** 28 addresses, 7,966 transactions; a main address has a median of 274, spanning 136-2,124 days.
- **Cleaning** (`reports/tables/cleaning_ds1.csv`): parse, keep confirmed, remove duplicate (address, txid) pairs,
  drop rows that neither pay nor spend, sort by time. No step removed a row.
- **Missing values.** None in the fields used. Feature columns are empty by design where history is too short
  (the first transaction has no gap; a z-score needs enough earlier values); the column tables below count them.
- **Biases.** Sampling from recent blocks favours addresses active now; the 50-1,000 range leaves out both busy
  services and casual users; 28 addresses is a small sample, so results are reported per address and with
  intervals. An address is not a person: one owner may use many addresses.
- **Time safety (DR-02).** A transaction's features use only strictly earlier transactions of the same address;
  models fit on each address's earlier 70% and are evaluated on the rest (`period`).
- **Raw replies** (about 0.4 GB, every API reply unchanged) stay on the collecting machine as evidence;
  `python -m explorer.cli snapshot --refetch` collects a new snapshot.

## DS2: Esplora crawl

- **Source and terms** as DS1. Manifest: `data/processed/corpus/manifest.json`.
- **Collection.** 200 block heights over the last 12 months (21 Oct 2025 - 21 Oct 2026); output addresses of up
  to four pages per block (49,260 candidates); every address in DS1's candidate pool removed (1,533), so DS1 never
  leaks into TxGPT's training data; shuffled with seed 30049 and visited until 4,000 were kept. Of 9,079 visited:
  4,012 kept, 4,422 too few transactions, 539 too many (busy services), 106 fetch errors, 0 failed checks.
- **Size.** 4,012 addresses, 898,490 rows (20-997 transactions each, median 138), from 2015 to 2026.
- **Privacy.** Addresses and transaction ids appear only as salted SHA-256 prefixes. The salt
  (`config/salt.txt`) never leaves the build machine, so the ids cannot be linked back.
- **Missing values.** None.
- **Biases.** As DS1, plus the 1,000-transaction cap (539 busy addresses skipped).

## DS3: Elliptic Data Set

- **Citation.** Weber, M. et al. (2019) 'Anti-money laundering in Bitcoin: experimenting with graph convolutional
  networks for financial forensics', KDD '19 Workshop on Anomaly Detection in Finance.
- **Source and licence.** Kaggle (`ellipticco/elliptic-data-set`), CC BY-NC-ND 4.0, checked 22 Oct 2026. No
  derivatives are allowed, so neither the data nor the models trained on it are in the repository;
  `python -m explorer.cli external --elliptic` downloads it and `data/processed/elliptic/download.json` holds the
  checksums.
- **Content** (`reports/tables/elliptic_summary.csv`). 203,769 transactions and 234,355 payment-flow edges over 49
  time steps; 46,564 labelled (4,545 illicit, 42,019 licit), 157,205 unknown. Features: AF (all) and LF (local
  only). "Illicit" and "licit" are the dataset's own class names.
- **Split.** Temporal: time steps 1-34 for training and validation, 35-49 for testing.
- **Missing values.** None after the join.
- **Biases.** The illicit share changes over time (fit 9.8%, validation 19.9%, test 6.5%), and most transactions
  are unlabelled.

## DS4: BitcoinHeist

- **Citation.** Akcora, C.G., Li, Y., Gel, Y.R. and Kantarcioglu, M. (2020) 'BitcoinHeist: topological data
  analysis for ransomware prediction on the Bitcoin blockchain', IJCAI-PRICAI 2020. DOI 10.24432/C5BG8V.
- **Source and licence.** UCI Machine Learning Repository, dataset 526, CC BY 4.0, checked 20 Oct 2026.
  `python -m explorer.cli external --bitcoinheist` downloads it; `data/processed/bitcoinheist/download.json` holds
  the checksums.
- **Content.** 2,916,697 address-days; 41,413 (1.4%) labelled as one of the ransomware families. Features: length,
  weight, count, looped, neighbors, income (heavy-tailed: logged, see `processing.json`). `year` and `day` define
  the splits and are never features; `address` is an identifier only.
- **Splits.** Temporal (training up to 2016), random, and leave-one-family-out (five families).
- **Missing values.** None; no exact duplicates.
- **Biases.** Positives are rare (1.4%), and on the temporal split every model stays near chance
  (`reports/tables/heist_results.csv`).

## DS5: FineWeb-Edu

- **Source and licence.** Hugging Face `HuggingFaceFW/fineweb-edu`, `sample/10BT/`, ODC-By 1.0.
- **Use.** Round 1: files 000-002, 1,938,432 documents, 2,000,417,414 training and 10,001,319 validation tokens
  (`data/text/meta.json`). Round 2 (Kaggle): 3,000,941,125 new tokens from file 002 onwards
  (`lm/prepare_text.py --continue-from`), the same validation tokens.
- **Not shipped** (gigabytes); `python -m lm.prepare_text` rebuilds the token files from the Hub.
- **Biases.** English educational web text only.

## DS6: smol-smoltalk

- **Source and licence.** Hugging Face `HuggingFaceTB/smol-smoltalk`, Apache 2.0.
- **Use.** 446,099 training and 4,485 validation conversations (294,891,521 tokens) for the 152M model's chat
  stage (`data/chat/meta.json`).
- **Not shipped**; `python -m lm.prepare_chat` rebuilds it.

## DS7: explanation pairs

- **Built by** `python -m explorer.cli build-ds7` from DS1 and DS2 pipeline outputs: a user question about one
  transaction, its facts as JSON, and the template's answer as the reply. Every reply passes the guardrail
  (`explorer.assistant.check`). Pairs are split by a seeded hash of the source address, so an address never spans
  two splits.
- **v1** (round 1, frozen): 10,160 pairs, including 160 glossary pairs; its test split is the frozen comparison set.
- **v2** (round 2): 36,210 training, 2,385 validation and 1,637 test pairs, plus a glossary of 306 answers on 65
  topics fact-checked against the code and tables (by Claude, at the user's request) and split by question, and a
  74-question glossary test. The training and validation files are rebuilt with
  `build-ds7 --settings ds7v2 --out data/processed/ds7_v2`; the glossary and its test are committed.
- **Balance.** Real prompts are mostly "nothing stands out"; the randomized sets below add balanced ones.

## DS10: a teammate's snapshot

- **Collected** by Le Quang Anh Hoang on 23 Oct 2026 with a separate script and a rule written before collecting:
  random recent blocks, three activity tiers with equal quotas, 50-600 transactions over at least 90 days, plus
  three short-history addresses (`data/processed/ds10/teammate/manifest.json`).
- **Refetched and cross-checked** by `python -m explorer.cli ds10`: all 5,785 transactions agree with the
  teammate's CSV, none differs, and no address is shared with DS1, DS1's candidate pool or DS2.
- **Use.** A held-out test: no model was tuned on DS10.
- **Biases.** The teammate's manifest notes the pool favours active addresses (often services).

## Randomized Assistant test sets

Built by `explorer/random_test.py` (seeds 50000-50003; settings `randomTest`; manifest
`data/processed/random_test/manifest.json`), so the Assistants are also tested on prompts unlike their training
mix:

| Set | Drawn from |
|---|---|
| real | DS7 v2's test split, none of them in the frozen v1 test |
| synthetic | made-up transactions with values capped to realistic limits |
| hard | made-up, out of range: other methods named, reworded questions, shuffled facts |
| balanced | real prompts, up to 100 per case (flagged or not, with 1-4 reasons; nothing stands out; unscored) |
| crawl_ds1, crawl_ds2, crawl_ds10 | real prompts from each crawled dataset's test addresses |
| close_calls | real prompts whose largest z lies near the flag line (almost all just below it) |
| extreme | made-up, extreme values within each regime |
| max_shift | made-up, every variation at once |

## Other files

- `data/d1/` holds A1's one-address data check (raw API replies, `scripts/d1_check.py`).
- Large or restricted files are not in the repository: DS3 and DS4 raw files (download commands and SHA-256 in
  their `download.json`), DS5 and DS6 token files, DS7 v2's training files, and DS1's raw replies.

## Files and columns

Written by `python tools/cards.py`. Types are as pandas reads them; "Missing" counts empty cells.

<!-- generated:files -->
| File | Dataset | Rows | Size | SHA-256 (first 16) | Holds |
|---|---|---|---|---|---|
| `data/processed/esplora/manifest.json` | DS1 |  | 178 kB | `2f564d94d3e787e4` | source, selection rule, blocks, candidate pool, per-address checks, terms of use, rebuild steps |
| `data/processed/esplora/histories.json.gz` | DS1 | 28 addresses | 795 kB | `2bf7f2c16d8bc42f` | every kept address's history as the API returned it (address -> role, chain statistics, transactions) |
| `data/processed/esplora/transactions.csv` | DS1 | 7,966 | 1.3 MB | `bf06fd90ddbf53d7` | one row per address and transaction: the cleaned table every model starts from |
| `data/processed/esplora/features.csv` | DS1 | 7,966 | 7.0 MB | `e899e5347caf601e` | the transactions with features F1-F8, own-history z-scores, the comparison windows, the baseline flag and the fit/evaluation period |
| `data/processed/esplora/features_as_written.csv` | DS1 | 7,966 | 6.8 MB | `9ac74703e0760eda` | the same, computed exactly as A1 section 2.3 wrote the method (for the refinement ablation) |
| `data/processed/esplora/scores.csv` | DS1 | 7,966 | 2.8 MB | `6123cad9abf5fe59` | every saved method's score and flag per transaction |
| `data/processed/esplora/txgpt_scores.csv` | DS1 | 7,966 | 1.5 MB | `055bd9a53df73a82` | TxGPT's surprisal per transaction and per token field |
| `data/processed/esplora/injections/reporting_30049.json` | DS1 |  | 258 kB | `8e9af9fa1bfb4249` | planted transactions, one seed per file (150 planted spike, burst and dormant patterns each); tuning seeds 40000-40004, reporting seeds 30049-30053 |
| `data/processed/esplora/injections/reporting_30050.json` | DS1 |  | 259 kB | `536137bb7b1ce4f2` | as above |
| `data/processed/esplora/injections/reporting_30051.json` | DS1 |  | 259 kB | `09b7da39b1418ce5` | as above |
| `data/processed/esplora/injections/reporting_30052.json` | DS1 |  | 258 kB | `690c477dd5879423` | as above |
| `data/processed/esplora/injections/reporting_30053.json` | DS1 |  | 259 kB | `f3fcbcb412abd222` | as above |
| `data/processed/esplora/injections/tuning_40000.json` | DS1 |  | 256 kB | `89c18be19e2b12d3` | as above |
| `data/processed/esplora/injections/tuning_40001.json` | DS1 |  | 255 kB | `3f9d53f401c10aa6` | as above |
| `data/processed/esplora/injections/tuning_40002.json` | DS1 |  | 256 kB | `d168c076d9883a1f` | as above |
| `data/processed/esplora/injections/tuning_40003.json` | DS1 |  | 255 kB | `8d97416cb4c5a2e9` | as above |
| `data/processed/esplora/injections/tuning_40004.json` | DS1 |  | 255 kB | `2f254cf4e23a35c1` | as above |
| `data/processed/corpus/manifest.json` | DS2 |  | 5 kB | `c664758d521a1ca4` | crawl rule, hosts, candidate pool, visit outcomes, checks |
| `data/processed/corpus/ds2.parquet` | DS2 | 898,490 | 31.9 MB | `26ba894acb16a95b` | one row per address and transaction, identifiers salted and hashed |
| `data/processed/profiles.csv` | DS1, DS2 | 4,037 | 607 kB | `9f0abb1df6713e05` | one row per address: its activity summary and K-Means profile |
| `data/processed/ds10/teammate/README.md` | DS10 |  | 2 kB | `c7107224d79a4b83` | the teammate's own description of the snapshot |
| `data/processed/ds10/teammate/manifest.json` | DS10 |  | 3 kB | `2c2c9a81bcf3ae89` | the teammate's selection rule, parameters, funnel and terms, written before collecting |
| `data/processed/ds10/teammate/address_summary.csv` | DS10 | 27 | 3 kB | `1eec1d65b3fd4a14` | the teammate's 27 addresses |
| `data/processed/ds10/teammate/transactions.csv` | DS10 | 5,785 | 1.1 MB | `c9d636e789c6604a` | the teammate's transaction table, as delivered |
| `data/processed/ds10/manifest.json` | DS10 |  | 4 kB | `8a70f8d52e846d32` | our refetch: checks and the cross-check against the teammate's CSV |
| `data/processed/ds10/histories.json.gz` | DS10 | 27 addresses | 571 kB | `dfc1c4f91687b1d3` | our refetch of the 27 histories through explorer/snapshot.py |
| `data/processed/ds10/transactions.csv` | DS10 | 5,785 | 961 kB | `2178de9cde9dd582` | the refetched transactions in DS1's format |
| `data/processed/ds10/explain.jsonl` | DS10 | 291 | 499 kB | `483138d415c2d3be` | explanation prompts built from DS10 by the DS7 rule (frozen test) |
| `data/processed/ds7/manifest.json` | DS7 |  | 1 kB | `82e2f893a33fa9bd` | round 1 build: settings, pair counts, split rule, checksums |
| `data/processed/ds7/train.jsonl` | DS7 | 9,149 | 12.8 MB | `0590c2b4b338c228` | round 1 training pairs |
| `data/processed/ds7/val.jsonl` | DS7 | 561 | 786 kB | `7818a89e3ffb8c16` | round 1 validation pairs |
| `data/processed/ds7/test.jsonl` | DS7 | 450 | 762 kB | `c2f1be1d519ad6ab` | round 1 test pairs (frozen; every writer is compared on it) |
| `data/processed/ds7/glossary.jsonl` | DS7 | 160 | 59 kB | `b2558774a1d1855c` | round 1 glossary: questions and answers on the Explorer's terms |
| `data/processed/ds7_v2/glossary.jsonl` | DS7 | 306 | 149 kB | `246abce080c20045` | round 2 glossary (306 answers on 65 topics, fact-checked), split by question |
| `data/processed/ds7_v2/glossary_test.jsonl` | DS7 | 74 | 84 kB | `d77a471e27d1fbb5` | round 2 glossary test prompts with the key facts an answer must state (frozen) |
| `data/processed/random_test/manifest.json` | Random tests |  | 9 kB | `651b16c9a4da81a5` | seeds, draws and composition of every randomized set |
| `data/processed/random_test/balanced.jsonl` | Random tests | 883 | 1.7 MB | `6fc75de02e15cc2d` | one randomized Assistant test set per file |
| `data/processed/random_test/close_calls.jsonl` | Random tests | 300 | 547 kB | `e4d5033bddd11f8d` | as above |
| `data/processed/random_test/crawl_ds1.jsonl` | Random tests | 174 | 305 kB | `9bae7b91267f75b5` | as above |
| `data/processed/random_test/crawl_ds10.jsonl` | Random tests | 334 | 612 kB | `be770aa91aad9c29` | as above |
| `data/processed/random_test/crawl_ds2.jsonl` | Random tests | 421 | 784 kB | `399455158235021b` | as above |
| `data/processed/random_test/extreme.jsonl` | Random tests | 300 | 530 kB | `00b58f9f819d8112` | as above |
| `data/processed/random_test/hard.jsonl` | Random tests | 500 | 910 kB | `75ae3ec52fe3f602` | as above |
| `data/processed/random_test/max_shift.jsonl` | Random tests | 300 | 550 kB | `a9cd63afe8908dfb` | as above |
| `data/processed/random_test/real.jsonl` | Random tests | 500 | 866 kB | `2a2a0ca9912df2fe` | as above |
| `data/processed/random_test/synthetic.jsonl` | Random tests | 500 | 894 kB | `b0410fbbcb38d695` | as above |
| `data/processed/bitcoinheist/download.json` | DS4 |  | 1 kB | `61e6cd3578c6b8ff` | source, citation, licence, archive and file checksums |
| `data/processed/bitcoinheist/processing.json` | DS4 |  | 1 kB | `37c31a5dc01a963b` | skewness, transforms, split rules and exclusions |
| `data/processed/elliptic/download.json` | DS3 |  | 2 kB | `434318b1cb3477fa` | source, citation, licence, archive and file checksums |

**`data/processed/esplora/transactions.csv`** (7,966 rows)

| Column | Type | Missing | Meaning |
|---|---|---|---|
| `address` | text | 0 | Bitcoin address (public identifier) |
| `role` | text | 0 | main (full history) or short (3 short-history addresses) |
| `txid` | text | 0 | transaction id (hex) |
| `block_height` | integer | 0 | block the transaction was confirmed in |
| `block_time_utc` | integer | 0 | block time, Unix seconds (UTC) |
| `received_sats` | integer | 0 | satoshis paid to the address (sum of its outputs) |
| `sent_sats` | integer | 0 | satoshis spent from the address (sum of its inputs) |
| `net_sats` | integer | 0 | received_sats - sent_sats (satoshis; positive = in) |
| `direction` | text | 0 | in, out or none (net amount zero) |
| `n_inputs` | integer | 0 | inputs of the transaction |
| `n_outputs` | integer | 0 | outputs of the transaction |
| `fee_sats` | integer | 0 | transaction fee, satoshis |
| `vsize_vb` | integer | 0 | virtual size, vbytes |

**`data/processed/esplora/features.csv`** (7,966 rows)

| Column | Type | Missing | Meaning |
|---|---|---|---|
| `address` | text | 0 | Bitcoin address (public identifier) |
| `role` | text | 0 | main (full history) or short (3 short-history addresses) |
| `txid` | text | 0 | transaction id (hex) |
| `block_height` | integer | 0 | block the transaction was confirmed in |
| `block_time_utc` | integer | 0 | block time, Unix seconds (UTC) |
| `received_sats` | integer | 0 | satoshis paid to the address (sum of its outputs) |
| `sent_sats` | integer | 0 | satoshis spent from the address (sum of its inputs) |
| `net_sats` | integer | 0 | received_sats - sent_sats (satoshis; positive = in) |
| `direction` | text | 0 | in, out or none (net amount zero) |
| `n_inputs` | integer | 0 | inputs of the transaction |
| `n_outputs` | integer | 0 | outputs of the transaction |
| `fee_sats` | integer | 0 | transaction fee, satoshis |
| `vsize_vb` | integer | 0 | virtual size, vbytes |
| `earlier_count` | integer | 0 | transactions at this address strictly earlier than this one (DR-02) |
| `history_span_days` | number | 0 | days from the address's first transaction to this one |
| `scorable` | true/false | 0 | enough earlier history to score (settings features.minHistoryTx and minHistorySpanDays) |
| `F1` | number | 0 | Size: log10(1 + |net_sats|) |
| `usual_F1` | number | 60 | usual value of F1 in its comparison window, readable units (F1 satoshis, F2 count, F3 hours, F4 ratio) |
| `center_F1` | number | 60 | median of F1's window on the comparison scale (F3 and F4 logged) |
| `spread_F1` | number | 60 | robust spread of F1's window (scaled MAD with a floor), comparison scale |
| `z_F1` | number | 260 | robust z-score of F1: (value - center) / spread; empty without enough history |
| `zc_F1` | number | 260 | z_F1 clipped to +/- features.zClip, for the pooled models |
| `F2` | integer | 0 | Burst: earlier transactions in the last features.burstWindowDays days (count) |
| `usual_F2` | number | 30 | usual value of F2 in its comparison window, readable units (F1 satoshis, F2 count, F3 hours, F4 ratio) |
| `center_F2` | number | 30 | median of F2's window on the comparison scale (F3 and F4 logged) |
| `spread_F2` | number | 30 | robust spread of F2's window (scaled MAD with a floor), comparison scale |
| `z_F2` | number | 138 | robust z-score of F2: (value - center) / spread; empty without enough history |
| `zc_F2` | number | 138 | z_F2 clipped to +/- features.zClip, for the pooled models |
| `F3` | number | 30 | Gap: hours since the previous transaction; empty for the first |
| `usual_F3` | number | 60 | usual value of F3 in its comparison window, readable units (F1 satoshis, F2 count, F3 hours, F4 ratio) |
| `center_F3` | number | 60 | median of F3's window on the comparison scale (F3 and F4 logged) |
| `spread_F3` | number | 60 | robust spread of F3's window (scaled MAD with a floor), comparison scale |
| `z_F3` | number | 165 | robust z-score of F3: (value - center) / spread; empty without enough history |
| `zc_F3` | number | 165 | z_F3 clipped to +/- features.zClip, for the pooled models |
| `F4` | number | 60 | Relative size: |net_sats| / the usual |net_sats| in the same direction (ratio); empty without a window |
| `usual_F4` | number | 118 | usual value of F4 in its comparison window, readable units (F1 satoshis, F2 count, F3 hours, F4 ratio) |
| `center_F4` | number | 118 | median of F4's window on the comparison scale (F3 and F4 logged) |
| `spread_F4` | number | 118 | robust spread of F4's window (scaled MAD with a floor), comparison scale |
| `z_F4` | number | 311 | robust z-score of F4: (value - center) / spread; empty without enough history |
| `zc_F4` | number | 311 | z_F4 clipped to +/- features.zClip, for the pooled models |
| `timing_window_count` | number | 30 | earlier transactions in the timing window (all directions; used by F2 and F3) |
| `timing_window_count_in_period` | number | 30 | of those, how many fall in the last features.comparisonWindowDays days |
| `timing_window_extended` | text | 30 | true if the window reached further back to collect features.comparisonMinTx transactions |
| `timing_window_start` | number | 30 | oldest transaction in the timing window, Unix seconds |
| `timing_window_end` | number | 30 | newest transaction in the timing window, Unix seconds |
| `size_window_count` | number | 60 | earlier same-direction transactions in the size window (used by F1 and F4) |
| `size_window_count_in_period` | number | 60 | of those, how many fall in the last features.comparisonWindowDays days |
| `size_window_extended` | text | 60 | true if the window reached further back to collect features.comparisonMinTx transactions |
| `size_window_start` | number | 60 | oldest transaction in the size window, Unix seconds |
| `size_window_end` | number | 60 | newest transaction in the size window, Unix seconds |
| `size_window_direction` | text | 60 | direction the size window was restricted to (in or out) |
| `score_baseline` | number | 0 | the robust-z baseline score (higher = more unusual) |
| `flag_baseline` | true/false | 0 | true if the robust-z baseline flags the transaction |
| `F5` | number | 0 | log2(n_inputs) (ablation and TxGPT only, never explained) |
| `F6` | number | 0 | log2(n_outputs) (ablation and TxGPT only) |
| `F7` | number | 0 | log10(1 + fee rate in sat/vB) (ablation and TxGPT only) |
| `F8_sin` | number | 0 | hour of day as sin(2 pi hour / 24) (ablation and TxGPT only) |
| `F8_cos` | number | 0 | hour of day as cos(2 pi hour / 24) |
| `prev_log_size` | number | 30 | log10(1 + |net_sats|) of the previous transaction; empty for the first |
| `prev_dir_in` | number | 0 | 1 if the previous transaction was incoming, else 0 |
| `prev_dir_out` | number | 0 | 1 if the previous transaction was outgoing, else 0 |
| `age_log_hours` | number | 0 | log10(1 + hours since the address's first transaction) |
| `in_share_prior` | number | 30 | share of earlier transactions that were incoming; empty for the first |
| `weekday_sin` | number | 0 | weekday as sin(2 pi day / 7), Monday = 0 |
| `weekday_cos` | number | 0 | weekday as cos(2 pi day / 7) |
| `F9` | number | 58 | Pace burst (stage A0): strictly earlier transactions within features.pace.windowGaps x the address's own median gap; an input of the pooled methods only when a stage chose it (the K-Means winner) |
| `z_F9` | number | 192 | robust z-score of F9: (value - center) / spread; empty without enough history |
| `zc_F9` | number | 192 | z_F9 clipped to +/- features.zClip, for the pooled models |
| `period` | text | 0 | fit (earlier 70% of the scorable history), eval (the rest) or history (not scorable) |

**`data/processed/esplora/features_as_written.csv`** (7,966 rows)

| Column | Type | Missing | Meaning |
|---|---|---|---|
| `address` | text | 0 | Bitcoin address (public identifier) |
| `role` | text | 0 | main (full history) or short (3 short-history addresses) |
| `txid` | text | 0 | transaction id (hex) |
| `block_height` | integer | 0 | block the transaction was confirmed in |
| `block_time_utc` | integer | 0 | block time, Unix seconds (UTC) |
| `received_sats` | integer | 0 | satoshis paid to the address (sum of its outputs) |
| `sent_sats` | integer | 0 | satoshis spent from the address (sum of its inputs) |
| `net_sats` | integer | 0 | received_sats - sent_sats (satoshis; positive = in) |
| `direction` | text | 0 | in, out or none (net amount zero) |
| `n_inputs` | integer | 0 | inputs of the transaction |
| `n_outputs` | integer | 0 | outputs of the transaction |
| `fee_sats` | integer | 0 | transaction fee, satoshis |
| `vsize_vb` | integer | 0 | virtual size, vbytes |
| `earlier_count` | integer | 0 | transactions at this address strictly earlier than this one (DR-02) |
| `history_span_days` | number | 0 | days from the address's first transaction to this one |
| `scorable` | true/false | 0 | enough earlier history to score (settings features.minHistoryTx and minHistorySpanDays) |
| `F1` | number | 0 | Size: log10(1 + |net_sats|) |
| `usual_F1` | number | 102 | usual value of F1 in its comparison window, readable units (F1 satoshis, F2 count, F3 hours, F4 ratio) |
| `center_F1` | number | 102 | median of F1's window on the comparison scale (F3 and F4 logged) |
| `spread_F1` | number | 102 | robust spread of F1's window (scaled MAD with a floor), comparison scale |
| `z_F1` | number | 518 | robust z-score of F1: (value - center) / spread; empty without enough history |
| `zc_F1` | number | 518 | z_F1 clipped to +/- features.zClip, for the pooled models |
| `F2` | integer | 0 | Burst: earlier transactions in the last features.burstWindowDays days (count) |
| `usual_F2` | number | 102 | usual value of F2 in its comparison window, readable units (F1 satoshis, F2 count, F3 hours, F4 ratio) |
| `center_F2` | number | 102 | median of F2's window on the comparison scale (F3 and F4 logged) |
| `spread_F2` | number | 102 | robust spread of F2's window (scaled MAD with a floor), comparison scale |
| `z_F2` | number | 518 | robust z-score of F2: (value - center) / spread; empty without enough history |
| `zc_F2` | number | 518 | z_F2 clipped to +/- features.zClip, for the pooled models |
| `F3` | number | 30 | Gap: hours since the previous transaction; empty for the first |
| `usual_F3` | number | 130 | usual value of F3 in its comparison window, readable units (F1 satoshis, F2 count, F3 hours, F4 ratio) |
| `center_F3` | number | 130 | median of F3's window on the comparison scale (F3 and F4 logged) |
| `spread_F3` | number | 130 | robust spread of F3's window (scaled MAD with a floor), comparison scale |
| `z_F3` | number | 542 | robust z-score of F3: (value - center) / spread; empty without enough history |
| `zc_F3` | number | 542 | z_F3 clipped to +/- features.zClip, for the pooled models |
| `F4` | number | 102 | Relative size: |net_sats| / the usual |net_sats| in the same direction (ratio); empty without a window |
| `usual_F4` | number | 194 | usual value of F4 in its comparison window, readable units (F1 satoshis, F2 count, F3 hours, F4 ratio) |
| `center_F4` | number | 194 | median of F4's window on the comparison scale (F3 and F4 logged) |
| `spread_F4` | number | 194 | robust spread of F4's window (scaled MAD with a floor), comparison scale |
| `z_F4` | number | 575 | robust z-score of F4: (value - center) / spread; empty without enough history |
| `zc_F4` | number | 575 | z_F4 clipped to +/- features.zClip, for the pooled models |
| `timing_window_count` | number | 102 | earlier transactions in the timing window (all directions; used by F2 and F3) |
| `timing_window_count_in_period` | number | 102 | of those, how many fall in the last features.comparisonWindowDays days |
| `timing_window_extended` | text | 102 | true if the window reached further back to collect features.comparisonMinTx transactions |
| `timing_window_start` | number | 102 | oldest transaction in the timing window, Unix seconds |
| `timing_window_end` | number | 102 | newest transaction in the timing window, Unix seconds |
| `size_window_count` | number | 102 | earlier same-direction transactions in the size window (used by F1 and F4) |
| `size_window_count_in_period` | number | 102 | of those, how many fall in the last features.comparisonWindowDays days |
| `size_window_extended` | text | 102 | true if the window reached further back to collect features.comparisonMinTx transactions |
| `size_window_start` | number | 102 | oldest transaction in the size window, Unix seconds |
| `size_window_end` | number | 102 | newest transaction in the size window, Unix seconds |
| `size_window_direction` | text | 102 | direction the size window was restricted to (in or out) |
| `score_baseline` | number | 0 | the robust-z baseline score (higher = more unusual) |
| `flag_baseline` | true/false | 0 | true if the robust-z baseline flags the transaction |
| `F5` | number | 0 | log2(n_inputs) (ablation and TxGPT only, never explained) |
| `F6` | number | 0 | log2(n_outputs) (ablation and TxGPT only) |
| `F7` | number | 0 | log10(1 + fee rate in sat/vB) (ablation and TxGPT only) |
| `F8_sin` | number | 0 | hour of day as sin(2 pi hour / 24) (ablation and TxGPT only) |
| `F8_cos` | number | 0 | hour of day as cos(2 pi hour / 24) |
| `prev_log_size` | number | 30 | log10(1 + |net_sats|) of the previous transaction; empty for the first |
| `prev_dir_in` | number | 0 | 1 if the previous transaction was incoming, else 0 |
| `prev_dir_out` | number | 0 | 1 if the previous transaction was outgoing, else 0 |
| `age_log_hours` | number | 0 | log10(1 + hours since the address's first transaction) |
| `in_share_prior` | number | 30 | share of earlier transactions that were incoming; empty for the first |
| `weekday_sin` | number | 0 | weekday as sin(2 pi day / 7), Monday = 0 |
| `weekday_cos` | number | 0 | weekday as cos(2 pi day / 7) |
| `F9` | number | 196 | Pace burst (stage A0): strictly earlier transactions within features.pace.windowGaps x the address's own median gap; an input of the pooled methods only when a stage chose it (the K-Means winner) |
| `z_F9` | number | 626 | robust z-score of F9: (value - center) / spread; empty without enough history |
| `zc_F9` | number | 626 | z_F9 clipped to +/- features.zClip, for the pooled models |
| `period` | text | 0 | fit (earlier 70% of the scorable history), eval (the rest) or history (not scorable) |

**`data/processed/esplora/scores.csv`** (7,966 rows)

| Column | Type | Missing | Meaning |
|---|---|---|---|
| `address` | text | 0 | Bitcoin address (public identifier) |
| `role` | text | 0 | main (full history) or short (3 short-history addresses) |
| `txid` | text | 0 | transaction id (hex) |
| `block_time_utc` | integer | 0 | block time, Unix seconds (UTC) |
| `direction` | text | 0 | in, out or none (net amount zero) |
| `net_sats` | integer | 0 | received_sats - sent_sats (satoshis; positive = in) |
| `period` | text | 0 | fit (earlier 70% of the scorable history), eval (the rest) or history (not scorable) |
| `scorable` | true/false | 0 | enough earlier history to score (settings features.minHistoryTx and minHistorySpanDays) |
| `score_baseline` | number | 0 | the robust-z baseline score (higher = more unusual) |
| `flag_baseline` | true/false | 0 | true if the robust-z baseline flags the transaction |
| `score_isolation_forest` | number | 0 | Isolation Forest score (higher = more unusual) |
| `flag_isolation_forest` | true/false | 0 | true if Isolation Forest flags the transaction |
| `score_lof` | number | 0 | LOF score (higher = more unusual) |
| `flag_lof` | true/false | 0 | true if LOF flags the transaction |
| `score_kmeans` | number | 0 | K-Means distance score (higher = more unusual) |
| `flag_kmeans` | true/false | 0 | true if K-Means distance flags the transaction |
| `score_dbscan` | number | 0 | DBSCAN score (higher = more unusual) |
| `flag_dbscan` | true/false | 0 | true if DBSCAN flags the transaction |
| `score_regression` | number | 0 | the regression residual score (higher = more unusual) |
| `flag_regression` | true/false | 0 | true if the regression residual flags the transaction |
| `score_regression_interval` | number | 0 | the conformal regression range score (higher = more unusual) |
| `flag_regression_interval` | true/false | 0 | true if the conformal regression range flags the transaction |
| `score_fusion` | number | 0 | fusion score (higher = more unusual) |
| `flag_fusion` | true/false | 0 | true if fusion flags the transaction |
| `score_txgpt_method` | number | 0 | TxGPT surprisal score (higher = more unusual) |
| `flag_txgpt_method` | true/false | 0 | true if TxGPT surprisal flags the transaction |

**`data/processed/esplora/txgpt_scores.csv`** (7,966 rows)

| Column | Type | Missing | Meaning |
|---|---|---|---|
| `txid` | text | 0 | transaction id (hex) |
| `address` | text | 0 | Bitcoin address (public identifier) |
| `surprisal_GAP` | number | 0 | TxGPT surprisal of the GAP token, nats |
| `surprisal_DIR` | number | 0 | TxGPT surprisal of the DIR token, nats |
| `surprisal_AMT` | number | 0 | TxGPT surprisal of the AMT token, nats |
| `surprisal_NIN` | number | 0 | TxGPT surprisal of the NIN token, nats |
| `surprisal_NOUT` | number | 0 | TxGPT surprisal of the NOUT token, nats |
| `surprisal_FEE` | number | 0 | TxGPT surprisal of the FEE token, nats |
| `surprisal` | number | 0 | TxGPT surprisal of the whole transaction: sum of the six fields, nats |

**`data/processed/corpus/ds2.parquet`** (898,490 rows)

| Column | Type | Missing | Meaning |
|---|---|---|---|
| `addr_id` | text | 0 | first hex characters of SHA-256(salt + address); the salt is never shipped |
| `txid_hash` | text | 0 | first hex characters of SHA-256(salt + txid); only to find duplicates |
| `block_height` | integer | 0 | block the transaction was confirmed in |
| `block_time_utc` | integer | 0 | block time, Unix seconds (UTC) |
| `received_sats` | integer | 0 | satoshis paid to the address (sum of its outputs) |
| `sent_sats` | integer | 0 | satoshis spent from the address (sum of its inputs) |
| `n_inputs` | integer | 0 | inputs of the transaction |
| `n_outputs` | integer | 0 | outputs of the transaction |
| `fee_sats` | integer | 0 | transaction fee, satoshis |
| `vsize_vb` | integer | 0 | virtual size, vbytes |

**`data/processed/profiles.csv`** (4,037 rows)

| Column | Type | Missing | Meaning |
|---|---|---|---|
| `id` | text | 0 | address (DS1) or its salted hash (DS2) |
| `tx_per_day` | number | 0 | transactions per day of history |
| `in_share` | number | 0 | share of its transactions that are incoming |
| `median_log_size` | number | 0 | median log10(1 + |net amount| in satoshis) |
| `gap_cv` | number | 0 | coefficient of variation of the gaps between consecutive transactions |
| `burstiness` | number | 0 | (sd - mean) / (sd + mean) of the daily transaction counts, from -1 (regular) to 1 (bursty) |
| `mean_outputs` | number | 0 | mean outputs per transaction |
| `history_tx` | integer | 0 | transactions in the history |
| `median_fee_rate` | number | 0 | median fee rate, sat/vB |
| `dataset` | text | 0 | DS1 or DS2 |
| `profile` | integer | 0 | K-Means activity profile (cluster number) |

**`data/processed/ds10/teammate/address_summary.csv`** (27 rows)

| Column | Type | Missing | Meaning |
|---|---|---|---|
| `address` | text | 0 | Bitcoin address (public identifier) |
| `group` | text | 0 | the teammate's activity tier (low, medium or high) |
| `address_type` | text | 0 | script type of the address |
| `confirmed_tx_count` | integer | 0 | confirmed transactions at collection time |
| `first_tx_utc` | text | 0 | first transaction, UTC date and time |
| `last_tx_utc` | text | 0 | last transaction, UTC date and time |
| `span_days` | number | 0 | days from first to last transaction |

**`data/processed/ds10/teammate/transactions.csv`** (5,785 rows)

| Column | Type | Missing | Meaning |
|---|---|---|---|
| `address` | text | 0 | Bitcoin address (public identifier) |
| `txid` | text | 0 | transaction id (hex) |
| `block_height` | integer | 0 | block the transaction was confirmed in |
| `confirmed_utc` | text | 0 | block time, UTC date and time |
| `block_time` | integer | 0 | block time, Unix seconds |
| `received_btc` | number | 0 | BTC paid to the address |
| `sent_btc` | number | 0 | BTC spent from the address |
| `net_btc` | number | 0 | received_btc - sent_btc (positive = in) |
| `direction` | text | 0 | in or out |

**`data/processed/ds10/transactions.csv`** (5,785 rows)

| Column | Type | Missing | Meaning |
|---|---|---|---|
| `address` | text | 0 | Bitcoin address (public identifier) |
| `role` | text | 0 | main (full history) or short (3 short-history addresses) |
| `txid` | text | 0 | transaction id (hex) |
| `block_height` | integer | 0 | block the transaction was confirmed in |
| `block_time_utc` | integer | 0 | block time, Unix seconds (UTC) |
| `received_sats` | integer | 0 | satoshis paid to the address (sum of its outputs) |
| `sent_sats` | integer | 0 | satoshis spent from the address (sum of its inputs) |
| `net_sats` | integer | 0 | received_sats - sent_sats (satoshis; positive = in) |
| `direction` | text | 0 | in, out or none (net amount zero) |
| `n_inputs` | integer | 0 | inputs of the transaction |
| `n_outputs` | integer | 0 | outputs of the transaction |
| `fee_sats` | integer | 0 | transaction fee, satoshis |
| `vsize_vb` | integer | 0 | virtual size, vbytes |

**JSON Lines files** (one JSON object per line; every field below, and the folders that use it)

| Field | In | Meaning |
|---|---|---|
| `id` | `data/processed/ds10/`, `data/processed/ds7/`, `data/processed/ds7_v2/`, `data/processed/random_test/` | address (DS1) or its salted hash (DS2) |
| `source` | `data/processed/ds10/`, `data/processed/ds7/`, `data/processed/ds7_v2/`, `data/processed/random_test/` | where the prompt came from: DS1, DS2, DS10 (their test splits in the random sets), glossary, or the generator (synthetic, hard, extreme, max_shift) |
| `kind` | `data/processed/ds10/`, `data/processed/ds7/`, `data/processed/ds7_v2/`, `data/processed/random_test/` | flagged, not_flagged, unscored or glossary |
| `phrasing` | `data/processed/ds10/`, `data/processed/ds7/`, `data/processed/ds7_v2/`, `data/processed/random_test/` | which of the five user-question phrasings was used (0-4; empty for glossary prompts) |
| `template` | `data/processed/ds10/`, `data/processed/ds7/`, `data/processed/ds7_v2/`, `data/processed/random_test/` | the template's answer to the same prompt (what the measures compare with) |
| `messages` | `data/processed/ds10/`, `data/processed/ds7/`, `data/processed/ds7_v2/`, `data/processed/random_test/` | the conversation: system prompt, user question with the facts as JSON, and the reference reply |
| `topic` | `data/processed/ds7/`, `data/processed/ds7_v2/` | glossary topic |
| `question` | `data/processed/ds7/`, `data/processed/ds7_v2/` | the question as a user would ask it |
| `answer` | `data/processed/ds7/`, `data/processed/ds7_v2/` | the reference answer |
| `status` | `data/processed/ds7/`, `data/processed/ds7_v2/` | review status of the answer |
| `split` | `data/processed/ds7_v2/` | train, val or test (glossary v2 splits by question, not by topic) |
| `key_facts` | `data/processed/ds7_v2/` | facts a correct glossary answer must state |
| `variant` | `data/processed/random_test/` | how a made-up prompt was varied: the method named, a reworded question (index) and shuffled facts |
<!-- /generated:files -->
