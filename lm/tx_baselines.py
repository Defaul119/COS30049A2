"""Is TxGPT learning anything? Its test NLL against three baselines that need no neural network.

Serves plan §10.1. On DS2's held-out TEST addresses (never seen in training), every transaction's
negative log-likelihood per field (nats) under:

    global   each field's token frequency over the training addresses
    habit    each field's token frequency in THIS address's history so far (Laplace-smoothed),
             close to what the robust-z baseline assumes
    markov   each field's token given the same field in the address's previous transaction,
             estimated on the training addresses (Laplace-smoothed); the first transaction uses global
    txgpt    the model, with up to 127 earlier transactions as context (lm.tx_score)

If TxGPT beats the habit and Markov baselines it has learned structure beyond counting; if it does
not, that is reported plainly and TxGPT is not chosen as the Explorer's method.

Usage:
    python -m lm.tx_baselines --model runs/txgpt-m-s1/model_best.pt [--model ...]

Writes reports/tables/txgpt_nll.csv (mean NLL per field and per transaction, by method).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .tx_score import check_vocabulary, load_model, surprisal
from .tx_tokens import FIELD_OFFSETS, FIELD_SIZES, FIELDS, tx_tokens


def field_tokens(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per address, an (n_tx, 6) array of within-field token indices (0 .. field size - 1)."""
    out = {}
    frame = frame.sort_values(["addr_id", "block_time_utc", "txid_hash"], kind="stable")
    for addr, g in frame.groupby("addr_id", sort=True):
        rows = g.to_dict("records")
        toks = [tx_tokens(r, rows[i - 1] if i else None) for i, r in enumerate(rows)]
        out[addr] = np.array([[t - FIELD_OFFSETS[f] for t, f in zip(tx, FIELDS)] for tx in toks], dtype=np.int64)
    return out


def global_nll(train: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> np.ndarray:
    nll = []
    for k, f in enumerate(FIELDS):
        counts = np.bincount(np.concatenate([a[:, k] for a in train.values()]), minlength=FIELD_SIZES[f]) + 1.0
        logp = np.log(counts / counts.sum())
        nll.append(np.concatenate([-logp[a[:, k]] for a in test.values()]))
    return np.stack(nll, axis=1)


def habit_nll(test: dict[str, np.ndarray]) -> np.ndarray:
    """Laplace-smoothed frequency of each field's token in the address's own earlier transactions."""
    rows = []
    for a in test.values():
        counts = {f: np.ones(FIELD_SIZES[f]) for f in FIELDS}
        for tx in a:
            rows.append([-np.log(counts[f][t] / counts[f].sum()) for f, t in zip(FIELDS, tx)])
            for f, t in zip(FIELDS, tx):
                counts[f][t] += 1
    return np.array(rows)


def markov_nll(train: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> np.ndarray:
    nll = []
    glob = global_nll(train, test)
    for k, f in enumerate(FIELDS):
        size = FIELD_SIZES[f]
        trans = np.ones((size, size))
        for a in train.values():
            np.add.at(trans, (a[:-1, k], a[1:, k]), 1)
        logp = np.log(trans / trans.sum(axis=1, keepdims=True))
        column, i = [], 0
        for a in test.values():
            column.append(glob[i, k])  # no previous transaction: fall back to global
            column += [-logp[p, c] for p, c in zip(a[:-1, k], a[1:, k])]
            i += len(a)
        nll.append(np.array(column))
    return np.stack(nll, axis=1)


def txgpt_nll(model_path: Path, frame: pd.DataFrame, test_ids: list[str], device: torch.device) -> np.ndarray:
    model, _ = load_model(model_path, device)
    out = []
    frame = frame.sort_values(["addr_id", "block_time_utc", "txid_hash"], kind="stable")
    for addr in sorted(test_ids):
        # The plain sequence NLL: the habit and Markov baselines also count every earlier row,
        # same-block ones included, so all four methods condition on the same history.
        rows = frame[frame["addr_id"] == addr].to_dict("records")
        out.append(surprisal(model, rows, device, strictly_earlier=False))
    return np.concatenate(out)


def summarise(name: str, nll: np.ndarray) -> dict[str, float]:
    row = {"method": name, "transactions": len(nll), "nll_per_tx": float(nll.sum(axis=1).mean()),
           "nll_per_token": float(nll.mean())}
    row.update({f"nll_{f}": float(nll[:, k].mean()) for k, f in enumerate(FIELDS)})
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", default=[], help="TxGPT weights; repeat to compare several")
    p.add_argument("--source", default="data/processed/corpus/ds2.parquet")
    p.add_argument("--meta", default="data/txgpt/meta.json")
    p.add_argument("--out", default="reports/tables/txgpt_nll.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    check_vocabulary(Path(args.meta))
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.source)
    held_out = set(meta["test_addr_ids"]) | set(meta["val_addr_ids"])
    tokens = field_tokens(frame)
    train = {a: t for a, t in tokens.items() if a not in held_out}
    test = {a: tokens[a] for a in sorted(meta["test_addr_ids"])}
    rows = [summarise("global", global_nll(train, test)), summarise("habit", habit_nll(test)),
            summarise("markov", markov_nll(train, test))]
    for path in args.model:
        rows.append(summarise(f"txgpt {path}", txgpt_nll(Path(path), frame, meta["test_addr_ids"], torch.device(args.device))))
    table = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
