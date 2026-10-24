"""Score transactions with a trained TxGPT: surprisal per field, from earlier transactions only.

Serves plan §10.1. A transaction's surprisal is s_i = -sum over its 6 fields of log p(token | every
token before it), kept per field (nats). Each transaction is scored with as much earlier context as
the window holds (up to 127 earlier transactions, with BOS when the history starts inside the
window), exactly the contexts the model was trained on. The model is causal, so a transaction is
scored only from what came before it, and the context is cut at strictly earlier block times, so
same-block transactions are never each other's history (DR-02).

The vocabulary and bucket edges are checked against the training data's meta.json, so training and
scoring always tokenise the same way.

Usage:
    python -m lm.tx_score            # every DS1 transaction, with the model lm.tx_select exported
    python -m lm.tx_score --plants   # also the planted copies
    python -m lm.tx_score --model runs/txgpt-m-s1/model_best.pt --meta data/txgpt/meta.json   # one sweep run

Writes data/processed/esplora/txgpt_scores.csv (and, with --plants, every row of every planted copy
to data/interim/injections/txgpt/<set>_<seed>.parquet).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .model import GPT, GPTConfig, IGNORE_INDEX
from .tx_tokens import BOS, FIELDS, PAD, TOKENS_PER_TX, describe, tx_tokens

CONTEXT_TX = 127  # earlier transactions a 768-token window holds next to the one being scored


def load_model(path: Path, device: torch.device) -> tuple[GPT, dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def check_vocabulary(meta_path: Path) -> None:
    """Stop if the data the model was trained on used different buckets from this code's."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ours = describe()
    differs = [k for k in ours if json.dumps(meta.get(k), sort_keys=True) != json.dumps(ours[k], sort_keys=True)]
    if differs:
        raise SystemExit(f"Token definitions differ from {meta_path}: {differs}")


def contexts(rows: list[dict[str, Any]], indices: list[int], strictly_earlier: bool = True) -> list[list[int]]:
    """For each transaction index, the token window ending with that transaction's 6 tokens.

    With strictly_earlier (DR-02, used for every DS1 and planted score), the context holds only
    transactions at strictly earlier block times: a same-block sibling is never history, and the
    scored transaction's gap is measured from the last strictly earlier one, exactly as for the
    first transaction of a block in training. Without it, the context is the plain sequence (every
    row before i), the standard language-model NLL that lm.tx_baselines compares with its baselines.
    """
    per_tx = [tx_tokens(r, rows[i - 1] if i else None) for i, r in enumerate(rows)]
    times = [float(r["block_time_utc"]) for r in rows]
    out = []
    for i in indices:
        end = i
        if strictly_earlier:
            while end > 0 and times[end - 1] >= times[i]:
                end -= 1
        start = max(0, end - CONTEXT_TX)
        window = [BOS] if start == 0 else []
        for j in range(start, end):
            window += per_tx[j]
        window += tx_tokens(rows[i], rows[end - 1] if end else None)
        out.append(window)
    return out


@torch.no_grad()
def surprisal(model: GPT, rows: list[dict[str, Any]], device: torch.device, indices: list[int] | None = None,
              batch: int = 32, strictly_earlier: bool = True) -> np.ndarray:
    """(len(indices), 6) surprisal in nats for the chosen transactions of one time-ordered history."""
    indices = list(range(len(rows))) if indices is None else indices
    windows = contexts(rows, indices, strictly_earlier)
    out = np.zeros((len(windows), TOKENS_PER_TX), dtype=np.float32)
    for b in range(0, len(windows), batch):
        chunk = windows[b:b + batch]
        width = max(len(w) for w in chunk) - 1
        x = torch.full((len(chunk), width), PAD, dtype=torch.long)
        y = torch.full((len(chunk), width), IGNORE_INDEX, dtype=torch.long)
        for k, w in enumerate(chunk):
            x[k, :len(w) - 1] = torch.tensor(w[:-1])
            y[k, :len(w) - 1] = torch.tensor(w[1:])
        _, loss, _ = model(x.to(device), y.to(device), reduction="none")
        loss = loss.float().cpu().numpy()
        for k, w in enumerate(chunk):
            end = len(w) - 1  # the scored transaction's 6 targets are the last 6 positions
            out[b + k] = loss[k, end - TOKENS_PER_TX:end]
    return out


def score_frame(model: GPT, frame: pd.DataFrame, device: torch.device, group: str, only: pd.Series | None = None) -> pd.DataFrame:
    """Surprisal columns for the rows of `frame` (all, or those where `only` is true), scoring each
    `group` (an address, or a planted copy) as one history in (time, txid) order."""
    parts = []
    for _, g in frame.groupby(group, sort=False):
        g = g.sort_values(["block_time_utc", "txid"], kind="stable")
        rows = g.to_dict("records")
        wanted = np.flatnonzero(only.loc[g.index].to_numpy()) if only is not None else np.arange(len(g))
        if len(wanted) == 0:
            continue
        s = surprisal(model, rows, device, wanted.tolist())
        part = g.iloc[wanted][["txid"] + ([group] if group != "address" else []) + ["address"]].copy()
        for f, col in zip(FIELDS, s.T):
            part[f"surprisal_{f}"] = col
        part["surprisal"] = s.sum(axis=1)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/txgpt.pt")
    p.add_argument("--meta", default="models/txgpt_meta.json")
    p.add_argument("--transactions", default="data/processed/esplora/transactions.csv")
    p.add_argument("--out", default="data/processed/esplora/txgpt_scores.csv")
    p.add_argument("--plants", action="store_true", help="also score every planted copy")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    check_vocabulary(Path(args.meta))
    device = torch.device(args.device)
    model, ckpt = load_model(Path(args.model), device)
    print(f"[tx_score] {args.model}: step {ckpt.get('step')}, validation loss {ckpt.get('val_loss')}, on {device}")

    tx = pd.read_csv(args.transactions)
    scores = score_frame(model, tx, device, "address")
    scores.to_csv(args.out, index=False)
    print(f"[tx_score] {len(scores):,} DS1 transactions -> {args.out}")

    if args.plants:
        interim = Path("data/interim/injections")
        out_dir = interim / "txgpt"
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted((interim / "refined").glob("*.parquet")):
            copies = pd.read_parquet(path)
            # Every row, not just the evaluation period: the relative score compares a transaction
            # with the same copy's earlier surprisals.
            part = score_frame(model, copies, device, "plant_id")
            part.to_parquet(out_dir / path.name, index=False)
            print(f"[tx_score] {path.name}: {len(part):,} rows of planted copies")


if __name__ == "__main__":
    main()
