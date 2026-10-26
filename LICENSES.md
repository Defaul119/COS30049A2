# Licences

What each part of this project is licensed under, and what that requires of us. Checked on the dates given in
each dataset's manifest (`DATA_CARD.md`).

## Our own work

The code, documentation, processed data (DS1, DS2, DS7, DS10's refetch, the randomized test sets) and the
models trained from scratch (the scoring methods, TxGPT, the 152M text model) were written by group
HCMC_COS30049.1_4 for COS30049 at Swinburne University of Technology. No open-source licence has been chosen for
them; ask the group before reusing them. DS10's original files (`data/processed/ds10/teammate/`) were collected by
Le Quang Anh Hoang, a group member.

## Data

| Dataset | Licence | What it requires | In this repository |
|---|---|---|---|
| DS1, DS2, DS10 (Bitcoin blockchain via the Esplora API) | Public data; the API states no terms of use (Blockstream caps unauthenticated use at 700 requests an hour) | Polite use: we sent at most two GET requests a second per host | Processed files; DS2's identifiers salted and hashed |
| DS3 Elliptic Data Set (Weber et al., 2019) | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (creator Claudio Bellei) | Attribution; non-commercial; **no derivatives** | Download script and checksums only; no data, no trained model |
| DS4 BitcoinHeist (Akcora et al., 2020; UCI 526) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Attribution | Download script, checksums and processing record |
| DS5 FineWeb-Edu (Hugging Face) | [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/); also subject to Common Crawl's terms of use | Attribution | Not shipped; rebuilt by `lm/prepare_text.py` |
| DS6 smol-smoltalk (Hugging Face) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Attribution, licence notice | Not shipped; rebuilt by `lm/prepare_chat.py` |
| DS7 and the randomized test sets | Ours (built from DS1, DS2 and DS10) | | Shipped |

The labelled datasets' class names ("illicit", "ransomware") are theirs; we never apply them to our addresses.

## Models

| Model | Base | Licence | What it requires |
|---|---|---|---|
| Qwen 3B adapters (`models/qwen3b_adapter*/`) | Qwen2.5-3B-Instruct, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` | [Qwen Research License](licenses/QWEN_RESEARCH_LICENSE.txt) (19 Sep 2024) | Non-commercial use only; give recipients the licence text (`licenses/QWEN_RESEARCH_LICENSE.txt`); keep the attribution line in a Notice file (`NOTICE`); show "Built with Qwen" in the documentation (README, MODEL_CARD); we ship adapters only, never modified Qwen files |
| 152M text model (`models/assistant/`) | none (trained from scratch) | ours | Trained on DS5 (ODC-By, attributed above) and DS6 (Apache 2.0) |
| TxGPT, scoring methods, clustering, regression | none | ours | Trained on DS1 and DS2 |
| DS3 and DS4 classifiers | none | follow their data | Elliptic's allow no derivatives to be shared, so neither is shipped |

**Built with Qwen.**

## Software

Installed by `environment.yml` (CPU) and `environment-gpu.yml` (GPU); none of it is copied into the repository.

| Package | Licence |
|---|---|
| Python 3.11 | PSF License |
| numpy, pandas, scikit-learn, scipy, seaborn, joblib, jupyterlab | BSD 3-Clause |
| matplotlib | Matplotlib License (PSF-based) |
| PyTorch | BSD-style (3-Clause) |
| pyarrow, tokenizers, huggingface_hub, transformers, peft, accelerate | Apache 2.0 |
| bitsandbytes, pytest | MIT |
