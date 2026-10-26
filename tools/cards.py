"""Writes the generated parts of DATA_CARD.md and MODEL_CARD.md from the shipped data and the result tables.

Serves plan §11 (docs/plan/sections/s11-code.tex): the data card lists "every file and column with type and unit,
missing values", and the model card gives each final model's metrics. CLAUDE.md asks that every reported number
come from code, so each card's narrative is written by hand and this script rewrites only the text between a
`<!-- generated:NAME -->` marker and its `<!-- /generated:NAME -->`. Row counts, checksums and metrics are read from
the files themselves, so they cannot drift from them. tests/test_cards.py checks that every column of every listed
file has a description here.

Usage:  python tools/cards.py           rewrite the generated blocks
        python tools/cards.py --check   exit 1 if a block is out of date, without writing
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
DATA_CARD = ROOT / "DATA_CARD.md"
MODEL_CARD = ROOT / "MODEL_CARD.md"

# Every shipped data file (globs allowed), in the card's order: (dataset, what it holds).
FILES: dict[str, tuple[str, str]] = {
    "data/processed/esplora/manifest.json": ("DS1", "source, selection rule, blocks, candidate pool, per-address checks, "
                                             "terms of use, rebuild steps"),
    "data/processed/esplora/histories.json.gz": ("DS1", "every kept address's history as the API returned it "
                                                 "(address -> role, chain statistics, transactions)"),
    "data/processed/esplora/transactions.csv": ("DS1", "one row per address and transaction: the cleaned table every "
                                                "model starts from"),
    "data/processed/esplora/features.csv": ("DS1", "the transactions with features F1-F8, own-history z-scores, the "
                                            "comparison windows, the baseline flag and the fit/evaluation period"),
    "data/processed/esplora/features_as_written.csv": ("DS1", "the same, computed exactly as A1 section 2.3 wrote the "
                                                       "method (for the refinement ablation)"),
    "data/processed/esplora/scores.csv": ("DS1", "every saved method's score and flag per transaction"),
    "data/processed/esplora/txgpt_scores.csv": ("DS1", "TxGPT's surprisal per transaction and per token field"),
    "data/processed/esplora/injections/*.json": ("DS1", "planted transactions, one seed per file (150 planted spike, "
                                                 "burst and dormant patterns each); tuning seeds 40000-40004, "
                                                 "reporting seeds 30049-30053"),
    "data/processed/corpus/manifest.json": ("DS2", "crawl rule, hosts, candidate pool, visit outcomes, checks"),
    "data/processed/corpus/ds2.parquet": ("DS2", "one row per address and transaction, identifiers salted and hashed"),
    "data/processed/profiles.csv": ("DS1, DS2", "one row per address: its activity summary and K-Means profile"),
    "data/processed/ds10/teammate/README.md": ("DS10", "the teammate's own description of the snapshot"),
    "data/processed/ds10/teammate/manifest.json": ("DS10", "the teammate's selection rule, parameters, funnel and "
                                                   "terms, written before collecting"),
    "data/processed/ds10/teammate/address_summary.csv": ("DS10", "the teammate's 27 addresses"),
    "data/processed/ds10/teammate/transactions.csv": ("DS10", "the teammate's transaction table, as delivered"),
    "data/processed/ds10/manifest.json": ("DS10", "our refetch: checks and the cross-check against the teammate's CSV"),
    "data/processed/ds10/histories.json.gz": ("DS10", "our refetch of the 27 histories through explorer/snapshot.py"),
    "data/processed/ds10/transactions.csv": ("DS10", "the refetched transactions in DS1's format"),
    "data/processed/ds10/explain.jsonl": ("DS10", "explanation prompts built from DS10 by the DS7 rule (frozen test)"),
    "data/processed/ds7/manifest.json": ("DS7", "round 1 build: settings, pair counts, split rule, checksums"),
    "data/processed/ds7/train.jsonl": ("DS7", "round 1 training pairs"),
    "data/processed/ds7/val.jsonl": ("DS7", "round 1 validation pairs"),
    "data/processed/ds7/test.jsonl": ("DS7", "round 1 test pairs (frozen; every writer is compared on it)"),
    "data/processed/ds7/glossary.jsonl": ("DS7", "round 1 glossary: questions and answers on the Explorer's terms"),
    "data/processed/ds7_v2/glossary.jsonl": ("DS7", "round 2 glossary (306 answers on 65 topics, fact-checked), "
                                             "split by question"),
    "data/processed/ds7_v2/glossary_test.jsonl": ("DS7", "round 2 glossary test prompts with the key facts an answer "
                                                  "must state (frozen)"),
    "data/processed/random_test/manifest.json": ("Random tests", "seeds, draws and composition of every randomized set"),
    "data/processed/random_test/*.jsonl": ("Random tests", "one randomized Assistant test set per file"),
    "data/processed/bitcoinheist/download.json": ("DS4", "source, citation, licence, archive and file checksums"),
    "data/processed/bitcoinheist/processing.json": ("DS4", "skewness, transforms, split rules and exclusions"),
    "data/processed/elliptic/download.json": ("DS3", "source, citation, licence, archive and file checksums"),
}

MODELS = {"histGradientBoosting": "Histogram gradient boosting", "knn": "k-nearest neighbours",
          "logistic": "Logistic regression", "randomForest": "Random forest", "ridge": "Ridge",
          "naive": "Naive (the usual value)"}
METHODS = {"baseline": "the robust-z baseline", "isolation_forest": "Isolation Forest", "lof": "LOF",
           "kmeans": "K-Means distance", "dbscan": "DBSCAN", "regression": "the regression residual",
           "regression_interval": "the conformal regression range", "fusion": "fusion",
           "txgpt_method": "TxGPT surprisal"}

# Column and field descriptions, with units. A file's own entry wins over the shared one.
COLUMNS: dict[str, str] = {
    "address": "Bitcoin address (public identifier)",
    "role": "main (full history) or short (3 short-history addresses)",
    "txid": "transaction id (hex)",
    "block_height": "block the transaction was confirmed in",
    "block_time_utc": "block time, Unix seconds (UTC)",
    "received_sats": "satoshis paid to the address (sum of its outputs)",
    "sent_sats": "satoshis spent from the address (sum of its inputs)",
    "net_sats": "received_sats - sent_sats (satoshis; positive = in)",
    "direction": "in, out or none (net amount zero)",
    "n_inputs": "inputs of the transaction",
    "n_outputs": "outputs of the transaction",
    "fee_sats": "transaction fee, satoshis",
    "vsize_vb": "virtual size, vbytes",
    "earlier_count": "transactions at this address strictly earlier than this one (DR-02)",
    "history_span_days": "days from the address's first transaction to this one",
    "scorable": "enough earlier history to score (settings features.minHistoryTx and minHistorySpanDays)",
    "F1": "Size: log10(1 + |net_sats|)",
    "F2": "Burst: earlier transactions in the last features.burstWindowDays days (count)",
    "F3": "Gap: hours since the previous transaction; empty for the first",
    "F4": "Relative size: |net_sats| / the usual |net_sats| in the same direction (ratio); empty without a window",
    "timing_window_count": "earlier transactions in the timing window (all directions; used by F2 and F3)",
    "timing_window_count_in_period": "of those, how many fall in the last features.comparisonWindowDays days",
    "timing_window_extended": "true if the window reached further back to collect features.comparisonMinTx transactions",
    "timing_window_start": "oldest transaction in the timing window, Unix seconds",
    "timing_window_end": "newest transaction in the timing window, Unix seconds",
    "size_window_count": "earlier same-direction transactions in the size window (used by F1 and F4)",
    "size_window_count_in_period": "of those, how many fall in the last features.comparisonWindowDays days",
    "size_window_extended": "true if the window reached further back to collect features.comparisonMinTx transactions",
    "size_window_start": "oldest transaction in the size window, Unix seconds",
    "size_window_end": "newest transaction in the size window, Unix seconds",
    "size_window_direction": "direction the size window was restricted to (in or out)",
    "F5": "log2(n_inputs) (ablation and TxGPT only, never explained)",
    "F6": "log2(n_outputs) (ablation and TxGPT only)",
    "F7": "log10(1 + fee rate in sat/vB) (ablation and TxGPT only)",
    "F8_sin": "hour of day as sin(2 pi hour / 24) (ablation and TxGPT only)",
    "F8_cos": "hour of day as cos(2 pi hour / 24)",
    "prev_log_size": "log10(1 + |net_sats|) of the previous transaction; empty for the first",
    "prev_dir_in": "1 if the previous transaction was incoming, else 0",
    "prev_dir_out": "1 if the previous transaction was outgoing, else 0",
    "age_log_hours": "log10(1 + hours since the address's first transaction)",
    "in_share_prior": "share of earlier transactions that were incoming; empty for the first",
    "weekday_sin": "weekday as sin(2 pi day / 7), Monday = 0",
    "weekday_cos": "weekday as cos(2 pi day / 7)",
    "period": "fit (earlier 70% of the scorable history), eval (the rest) or history (not scorable)",
    "surprisal": "TxGPT surprisal of the whole transaction: sum of the six fields, nats",
    # profiles.csv
    "id": "address (DS1) or its salted hash (DS2)",
    "tx_per_day": "transactions per day of history",
    "in_share": "share of its transactions that are incoming",
    "median_log_size": "median log10(1 + |net amount| in satoshis)",
    "gap_cv": "coefficient of variation of the gaps between consecutive transactions",
    "burstiness": "(sd - mean) / (sd + mean) of the daily transaction counts, from -1 (regular) to 1 (bursty)",
    "mean_outputs": "mean outputs per transaction",
    "history_tx": "transactions in the history",
    "median_fee_rate": "median fee rate, sat/vB",
    "dataset": "DS1 or DS2",
    "profile": "K-Means activity profile (cluster number)",
    # ds2.parquet
    "addr_id": "first hex characters of SHA-256(salt + address); the salt is never shipped",
    "txid_hash": "first hex characters of SHA-256(salt + txid); only to find duplicates",
    # DS10 teammate files
    "group": "the teammate's activity tier (low, medium or high)",
    "address_type": "script type of the address",
    "confirmed_tx_count": "confirmed transactions at collection time",
    "first_tx_utc": "first transaction, UTC date and time",
    "last_tx_utc": "last transaction, UTC date and time",
    "span_days": "days from first to last transaction",
    "confirmed_utc": "block time, UTC date and time",
    "block_time": "block time, Unix seconds",
    "received_btc": "BTC paid to the address",
    "sent_btc": "BTC spent from the address",
    "net_btc": "received_btc - sent_btc (positive = in)",
    # JSON Lines (explanation pairs and test sets)
    "source": "where the prompt came from: DS1, DS2, DS10 (their test splits in the random sets), glossary, or the "
              "generator (synthetic, hard, extreme, max_shift)",
    "kind": "flagged, not_flagged, unscored or glossary",
    "phrasing": "which of the five user-question phrasings was used (0-4; empty for glossary prompts)",
    "messages": "the conversation: system prompt, user question with the facts as JSON, and the reference reply",
    "template": "the template's answer to the same prompt (what the measures compare with)",
    "variant": "how a made-up prompt was varied: the method named, a reworded question (index) and shuffled facts",
    "key_facts": "facts a correct glossary answer must state",
    "topic": "glossary topic",
    "question": "the question as a user would ask it",
    "answer": "the reference answer",
    "status": "review status of the answer",
    "split": "train, val or test (glossary v2 splits by question, not by topic)",
}
FILE_COLUMNS: dict[str, dict[str, str]] = {
    "data/processed/ds10/teammate/transactions.csv": {"direction": "in or out"},
}
PATTERNS: list[tuple[str, Callable[[re.Match[str]], str]]] = [
    (r"usual_(F\d)", lambda m: f"usual value of {m[1]} in its comparison window, readable units (F1 satoshis, "
                               f"F2 count, F3 hours, F4 ratio)"),
    (r"center_(F\d)", lambda m: f"median of {m[1]}'s window on the comparison scale (F3 and F4 logged)"),
    (r"spread_(F\d)", lambda m: f"robust spread of {m[1]}'s window (scaled MAD with a floor), comparison scale"),
    (r"z_(F\d)", lambda m: f"robust z-score of {m[1]}: (value - center) / spread; empty without enough history"),
    (r"zc_(F\d)", lambda m: f"z_{m[1]} clipped to +/- features.zClip, for the pooled models"),
    (r"score_(\w+)", lambda m: f"{METHODS.get(m[1], m[1])} score (higher = more unusual)"),
    (r"flag_(\w+)", lambda m: f"true if {METHODS.get(m[1], m[1])} flags the transaction"),
    (r"surprisal_([A-Z]+)", lambda m: f"TxGPT surprisal of the {m[1]} token, nats"),
]


def describe(path: str, column: str) -> str | None:
    """The description of one column of one file, or None if it has none (tests/test_cards.py)."""
    if column in FILE_COLUMNS.get(path, {}):
        return FILE_COLUMNS[path][column]
    if column in COLUMNS:
        return COLUMNS[column]
    for pattern, text in PATTERNS:
        if m := re.fullmatch(pattern, column):
            return text(m)
    return None


def listed_files() -> list[Path]:
    """Every file FILES names, globs expanded, in the card's order."""
    out: list[Path] = []
    for pattern in FILES:
        out.extend(sorted(ROOT.glob(pattern)) if "*" in pattern else [ROOT / pattern])
    return out


def entry(path: Path) -> tuple[str, str]:
    rel = path.relative_to(ROOT).as_posix()
    for pattern, value in FILES.items():
        if Path(rel).match(pattern) if "*" in pattern else rel == pattern:
            return value
    raise KeyError(rel)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kind_of(dtype: object) -> str:
    text = str(dtype)
    return ("integer" if text.startswith("int") else "number" if text.startswith("float") else
            "true/false" if text == "bool" else "text")


def table(path: Path) -> pd.DataFrame | None:
    """The file as a table, or None for a file that is not one (JSON, README)."""
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return None


def jsonl_fields(path: Path) -> tuple[int, dict[str, int]]:
    """(records, field -> how many records carry it non-empty)."""
    records, present = 0, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            records += 1
            for k, v in row.items():
                present[k] = present.get(k, 0) + (v is not None)
    return records, present


def size_text(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} kB"


def data_files_block() -> str:
    """The summary table of every shipped data file, then each table's columns."""
    lines = ["| File | Dataset | Rows | Size | SHA-256 (first 16) | Holds |", "|---|---|---|---|---|---|"]
    details: list[str] = []
    previous = ""
    for path in listed_files():
        rel = path.relative_to(ROOT).as_posix()
        dataset, holds = entry(path)
        # Files from one glob share a description; say it once.
        holds, previous = ("as above" if holds == previous else holds), holds
        if not path.exists():
            lines.append(f"| `{rel}` | {dataset} | missing | | | {holds} |")
            continue
        rows = ""
        frame = table(path)
        if frame is not None:
            rows = f"{len(frame):,}"
            details.append(f"\n**`{rel}`** ({len(frame):,} rows)\n\n| Column | Type | Missing | Meaning |\n|---|---|---|---|")
            for col, dtype in frame.dtypes.items():
                missing = int(frame[col].isna().sum())
                details.append(f"| `{col}` | {kind_of(dtype)} | {missing:,} | {describe(rel, str(col)) or ''} |")
        elif path.suffix == ".jsonl":
            records, present = jsonl_fields(path)
            rows = f"{records:,}"
        elif rel.endswith(".json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                rows = f"{len(json.load(fh)):,} addresses"
        lines.append(f"| `{rel}` | {dataset} | {rows} | {size_text(path.stat().st_size)} | "
                     f"`{sha256(path)[:16]}` | {holds} |")
    fields: dict[str, set[str]] = {}
    for path in listed_files():
        if path.suffix == ".jsonl" and path.exists():
            for field in jsonl_fields(path)[1]:
                fields.setdefault(field, set()).add(path.parent.relative_to(ROOT).as_posix())
    details.append("\n**JSON Lines files** (one JSON object per line; every field below, and the folders that use "
                   "it)\n\n| Field | In | Meaning |\n|---|---|---|")
    for field, folders in fields.items():
        details.append(f"| `{field}` | {', '.join(f'`{f}/`' for f in sorted(folders))} | {describe('', field) or ''} |")
    return "\n".join(lines + details)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def methods_block() -> str:
    """Planted-pattern recall of every unsupervised method on DS1, DS2 and DS10."""
    sel = pd.read_csv(TABLES / "eval_selection.csv").set_index("method")
    ds2 = pd.read_csv(TABLES / "eval_ds2.csv").set_index("method")
    ds10 = pd.read_csv(TABLES / "eval_ds10.csv").set_index("method")
    dist = pd.read_csv(TABLES / "stats_methods_summary.csv")
    dist = dist[dist["dataset"] == "DS1"].set_index("method")
    lines = ["| Method | DS1 mean recall (5 reporting seeds) | DS1, 100 seeds: mean (sd) | DS1 precision, lower bound | "
             "DS2 mean recall [95% CI] | DS10 mean recall [95% CI] |", "|---|---|---|---|---|---|"]
    for method, r in sel.iterrows():
        d = f"{dist.loc[method, 'mean']:.3f} ({dist.loc[method, 'sd']:.3f})" if method in dist.index else "--"
        cells = []
        for frame, key in ((ds2, "ds2"), (ds10, "ds10")):
            cells.append(f"{frame.loc[method, f'{key}_mean_recall']:.3f} [{frame.loc[method, f'{key}_mean_recall_ci_low']:.3f}, "
                         f"{frame.loc[method, f'{key}_mean_recall_ci_high']:.3f}]" if method in frame.index else "--")
        name = METHODS.get(method, method)
        lines.append(f"| {name[0].upper() + name[1:]} | {r['mean_recall']:.3f} | {d} | "
                     f"{r['precision_lower_bound_mean']:.3f} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)


def classifiers_block() -> str:
    """Test-set metrics of every classifier on DS3 (Elliptic) and DS4 (BitcoinHeist), against the datasets' labels."""
    lines = ["| Data and split | Model | Accuracy | Precision | Recall | F1 | PR-AUC |", "|---|---|---|---|---|---|---|"]
    for name, key, label in (("elliptic_results.csv", "feature_set", "Elliptic, temporal, features {}"),
                             ("heist_results.csv", "scheme", "BitcoinHeist, {} split")):
        for r in pd.read_csv(TABLES / name).itertuples():
            lines.append(f"| {label.format(getattr(r, key))} | {MODELS.get(r.model, r.model)} | {r.accuracy:.3f} | "
                         f"{r.precision:.3f} | "
                         f"{r.recall:.3f} | {r.f1:.3f} | {r.pr_auc:.3f} |")
    return "\n".join(lines)


def regression_block() -> str:
    """Every regression model's error on the evaluation period, and the conformal ranges' coverage."""
    frame = pd.read_csv(TABLES / "regression_results.csv")
    lines = ["| Target | Model | MAE | R² | Skill vs naive | Range coverage (nominal) |", "|---|---|---|---|---|---|"]
    for r in frame.itertuples():
        cover = f"{pct(r.coverage)} ({pct(r.nominal)})" if pd.notna(r.coverage) else ""
        mae = f"{r.mae:.3f}" if pd.notna(r.mae) else ""
        r2 = f"{r.r2:.3f}" if pd.notna(r.r2) else ""
        skill = f"{r.skill_vs_naive:.3f}" if pd.notna(r.skill_vs_naive) else ""
        lines.append(f"| {r.target} | {MODELS.get(r.model, r.model)} | {mae} | {r2} | {skill} | {cover} |")
    return "\n".join(lines)


def clustering_block() -> str:
    """The chosen K-Means and DBSCAN settings with their internal quality measures."""
    km = pd.read_csv(TABLES / "tuning_kmeans.csv")
    km = km[km["chosen"]].iloc[0]
    db = pd.read_csv(TABLES / "tuning_dbscan.csv")
    db = db[db["chosen"]].iloc[0]
    pr = pd.read_csv(TABLES / "profiles_tuning.csv")
    pr = pr[pr["chosen"]].iloc[0]
    return "\n".join([
        "| Model | Chosen setting | Quality |", "|---|---|---|",
        f"| K-Means (transactions) | k = {int(km['k'])} | silhouette {km['silhouette']:.3f}, "
        f"Davies-Bouldin {km['davies_bouldin']:.3f} |",
        f"| DBSCAN (transactions) | eps = {db['eps']:.3f}, min_samples = {int(db['min_samples'])} | "
        f"{int(db['clusters'])} clusters, noise share {pct(db['noise_share'])}, silhouette (non-noise) "
        f"{db['silhouette_non_noise']:.3f} |",
        f"| K-Means (address profiles) | k = {int(pr['k'])} | silhouette {pr['silhouette']:.3f}, "
        f"Davies-Bouldin {pr['davies_bouldin']:.3f} |"])


def txgpt_block() -> str:
    """TxGPT's three sizes: validation loss over three seeds each."""
    lines = ["| Size | Parameters | Mean validation loss (sd), 3 seeds | Kept |", "|---|---|---|---|"]
    for r in pd.read_csv(TABLES / "txgpt_sizes.csv").itertuples():
        lines.append(f"| {r.size.upper()} | {r.parameters:,} | {r.mean_val_loss:.4f} ({r.std_val_loss:.4f}) | "
                     f"{'yes' if r.chosen else ''} |")
    return "\n".join(lines)


WRITER_NAMES = {"template": "Template (control)", "text150m": "152M, round 1", "qwen3b": "Qwen 3B, round 1",
                "qwen3b_v2": "Qwen 3B, round 2 (used)", "text150m_v2": "152M, round 2"}


def writers_block() -> str:
    """Each Explanation writer: guardrail and fully correct on the frozen tests, pooled rate, glossary key facts."""
    v1 = pd.read_csv(TABLES / "assistant_eval.csv").set_index("model")
    ds10 = pd.read_csv(TABLES / "assistant_eval_ds10.csv").set_index("model")
    stats = pd.read_csv(TABLES / "stats_writers_summary.csv")
    pooled = stats[stats["test_set"].str.startswith("pooled")].set_index("writer")
    glossary = stats[stats["test_set"].str.startswith("glossary")].set_index("writer")
    sets = pooled["test_set"].iloc[0] if len(pooled) else "pooled"
    lines = [f"| Writer | Guardrail pass, DS7 test / DS10 | Fully correct, DS7 test / DS10 | Fully correct, {sets} "
             f"[95% CI] | Glossary key facts [95% CI] |", "|---|---|---|---|---|"]
    for model in v1.index:
        pool = (f"{pct(pooled.loc[model, 'fully_correct'])} [{pct(pooled.loc[model, 'ci_low'])}, "
                f"{pct(pooled.loc[model, 'ci_high'])}]" if model in pooled.index else "--")
        gloss = (f"{pct(glossary.loc[model, 'fully_correct'])} [{pct(glossary.loc[model, 'ci_low'])}, "
                 f"{pct(glossary.loc[model, 'ci_high'])}]" if model in glossary.index else "--")
        lines.append(f"| {WRITER_NAMES.get(model, model)} | {pct(v1.loc[model, 'guardrail_pass'])} / "
                     f"{pct(ds10.loc[model, 'guardrail_pass'])} | {pct(v1.loc[model, 'fully_correct'])} / "
                     f"{pct(ds10.loc[model, 'fully_correct'])} | {pool} | {gloss} |")
    return "\n".join(lines)


BLOCKS: dict[Path, dict[str, Callable[[], str]]] = {
    DATA_CARD: {"files": data_files_block},
    MODEL_CARD: {"methods": methods_block, "clustering": clustering_block, "regression": regression_block,
                 "classifiers": classifiers_block, "txgpt": txgpt_block, "writers": writers_block},
}


def fill(text: str, blocks: dict[str, Callable[[], str]]) -> str:
    """The card with each generated block replaced; a marker that is missing is an error."""
    for name, make in blocks.items():
        start, end = f"<!-- generated:{name} -->", f"<!-- /generated:{name} -->"
        if start not in text or end not in text:
            raise SystemExit(f"missing markers for block {name!r}")
        head, rest = text.split(start, 1)
        text = f"{head}{start}\n{make()}\n{end}{rest.split(end, 1)[1]}"
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a card is out of date; write nothing")
    args = parser.parse_args()
    stale = []
    for path, blocks in BLOCKS.items():
        old = path.read_text(encoding="utf-8")
        new = fill(old, blocks)
        if new != old:
            stale.append(path.name)
            if not args.check:
                path.write_text(new, encoding="utf-8", newline="\n")
    if not stale:
        print("Cards are current.")
    else:
        print(("Out of date: " if args.check else "Rewrote: ") + ", ".join(stale))
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
