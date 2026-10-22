"""Build TxGPT's token files from DS2 (data/processed/corpus/ds2.parquet).

Serves plan §6.7 and §10.1. Each address's history becomes BOS + 6 tokens per transaction
(lm/tx_tokens.py). Addresses, not transactions, are split 90/5/5 into train, validation and test,
so no test address is ever seen in training; DS1 is excluded from DS2 upstream.

Windows: a history's first window is BOS plus up to 127 transactions; longer histories continue in
windows of 128 transactions that start every 64 transactions (a 384-token stride, aligned to
transaction boundaries), so the model learns both "beginning" and "middle" contexts, and BOS
appears only at a true history start. Every window is padded to the full 768 tokens and written
back to back, which is exactly the layout lm.data.WindowSampler cuts. The mask file switches off
padding and each window's first token, so no target ever crosses from one window into the next.

Writes data/txgpt/{train,val,test}.bin, {train,val,test}_mask.bin and meta.json (vocabulary,
bucket edges, split sizes, the test addresses' ids and the source file's SHA-256).

Usage:
    python -m lm.prepare_tx
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TOKEN_DTYPE
from .tx_tokens import BOS, PAD, TOKENS_PER_TX, describe, history_tokens

SEQ_TX = 128          # transactions per window: 768 tokens
STRIDE_TX = 64        # later windows start every 64 transactions (384 tokens)
SPLIT = {"train": 0.90, "val": 0.05, "test": 0.05}
SEED = 30049


def windows(tokens: list[int], seq_len: int) -> list[tuple[list[int], list[int]]]:
    """(window, mask) pairs for one history (tokens starting with BOS), each exactly seq_len long."""
    body = tokens[1:]
    n_tx = len(body) // TOKENS_PER_TX

    def padded(w: list[int]) -> tuple[list[int], list[int]]:
        # Mask 0 on the first token: as a target it would be predicted from the previous window.
        return w + [PAD] * (seq_len - len(w)), [0] + [1] * (len(w) - 1) + [0] * (seq_len - len(w))

    first_tx = SEQ_TX - 1  # the first window holds BOS + 127 transactions (763 tokens), padded to 768
    out = [padded([BOS] + body[: first_tx * TOKENS_PER_TX])]
    start = STRIDE_TX
    while n_tx > first_tx:
        out.append(padded(body[start * TOKENS_PER_TX: (start + SEQ_TX) * TOKENS_PER_TX]))
        if start + SEQ_TX >= n_tx:
            break
        start += STRIDE_TX
    return out


def split_addresses(addr_ids: list[str], seed: int = SEED) -> dict[str, list[str]]:
    order = np.random.default_rng(seed).permutation(sorted(addr_ids)).tolist()
    n_train = round(SPLIT["train"] * len(order))
    n_val = round(SPLIT["val"] * len(order))
    return {"train": order[:n_train], "val": order[n_train:n_train + n_val], "test": order[n_train + n_val:]}


def address_histories(frame: pd.DataFrame) -> dict[str, list[int]]:
    """Token history per address id, oldest first (ties in a block ordered by txid hash)."""
    frame = frame.sort_values(["addr_id", "block_time_utc", "txid_hash"], kind="stable")
    return {addr: history_tokens(g.to_dict("records")) for addr, g in frame.groupby("addr_id", sort=True)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="data/processed/corpus/ds2.parquet")
    p.add_argument("--out", default="data/txgpt")
    args = p.parse_args()
    source, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seq_len = SEQ_TX * TOKENS_PER_TX

    histories = address_histories(pd.read_parquet(source))
    split = split_addresses(list(histories))
    counts = {}
    for part, ids in split.items():
        tokens, masks = [], []
        for addr in ids:
            for w, m in windows(histories[addr], seq_len):
                tokens += w
                masks += m
        np.asarray(tokens, dtype=TOKEN_DTYPE).tofile(out / f"{part}.bin")
        np.asarray(masks, dtype=np.uint8).tofile(out / f"{part}_mask.bin")
        counts[part] = {"addresses": len(ids), "windows": len(tokens) // seq_len, "tokens": len(tokens),
                        "trained_tokens": int(sum(masks)),
                        "transactions": sum((len(histories[a]) - 1) // TOKENS_PER_TX for a in ids)}
        print(f"[tx] {part}: {counts[part]}")

    meta = {
        "kind": "tx", **describe(), "seq_len": seq_len, "window_transactions": SEQ_TX, "stride_transactions": STRIDE_TX,
        "train_tokens": counts["train"]["tokens"], "split": counts, "split_seed": SEED,
        "test_addr_ids": split["test"], "val_addr_ids": split["val"],
        "source": {"path": source.as_posix(), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[tx] wrote {out}/meta.json: vocabulary {meta['vocab_size']}, {meta['train_tokens']:,} training tokens")


if __name__ == "__main__":
    main()
