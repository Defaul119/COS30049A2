"""DS1: from the snapshot's histories to a clean table, one row per (address, transaction).

Serves plan §6.1 (docs/plan/sections/s06-processing.tex) and report criterion 1 (processing). Each
cleaning step counts the rows it removes, and the counts are written to
reports/tables/cleaning_ds1.csv, which the report's processing table quotes.

Units are always in the column names (_sats, _utc, _vb). Amounts stay integer satoshis; BTC appears
only in plots and in the interface. block_time_utc is Unix seconds (UTC).

Reads data/processed/esplora/histories.json.gz (written by `snapshot`); writes
data/processed/esplora/transactions.csv and reports/tables/cleaning_ds1.csv.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .amounts import address_flow
from .settings import PROJECT_ROOT
from .snapshot import HISTORIES_FILE, MANIFEST_PATH, read_gz

log = logging.getLogger(__name__)

HISTORIES_PATH = MANIFEST_PATH.parent / HISTORIES_FILE
OUT_PATH = MANIFEST_PATH.parent / "transactions.csv"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
WEIGHT_UNITS_PER_VBYTE = 4

COLUMNS = [
    "address", "role", "txid", "block_height", "block_time_utc", "received_sats", "sent_sats", "net_sats",
    "direction", "n_inputs", "n_outputs", "fee_sats", "vsize_vb",
]
REQUIRED = ["txid", "block_height", "block_time_utc", "fee_sats", "vsize_vb"]


def build_transactions(histories: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """The clean table and the cleaning log (one entry per step, with rows before, removed and after).

    Raises ValueError if a required value is missing after cleaning: that would mean the API changed,
    and no number should be invented in its place (plan §6.1).
    """
    steps: list[dict[str, Any]] = []

    def record(step: str, before: int, after: int, note: str) -> None:
        steps.append({"step": step, "rows_before": before, "removed": before - after, "rows_after": after,
                      "note": note})
        log.info("%s: %d -> %d rows (%d removed)", step, before, after, before - after)

    raw = [(address, h["role"], tx) for address, h in sorted(histories.items()) for tx in h["txs"]]
    record("parse every saved page", len(raw), len(raw), f"{len(histories)} addresses")

    confirmed = [r for r in raw if r[2]["status"].get("confirmed")]
    record("keep confirmed transactions", len(raw), len(confirmed),
           "the chain endpoint returns only confirmed ones; checked anyway")

    seen: set[tuple[str, str]] = set()
    unique = []
    for address, role, tx in confirmed:
        if (address, tx["txid"]) not in seen:
            seen.add((address, tx["txid"]))
            unique.append((address, role, tx))
    record("remove duplicate (address, txid) pairs", len(confirmed), len(unique),
           "can occur at page boundaries if a block arrives mid-fetch")

    rows = []
    for address, role, tx in unique:
        flow = address_flow(tx, address)
        status = tx["status"]
        rows.append({
            "address": address, "role": role, "txid": tx["txid"], "block_height": status.get("block_height"),
            "block_time_utc": status.get("block_time"), "received_sats": flow.received, "sent_sats": flow.sent,
            "net_sats": flow.net, "direction": flow.direction, "n_inputs": tx["n_vin"], "n_outputs": tx["n_vout"],
            "fee_sats": tx.get("fee"),
            "vsize_vb": math.ceil(tx["weight"] / WEIGHT_UNITS_PER_VBYTE) if tx.get("weight") is not None else None,
        })
    involved = [r for r in rows if r["received_sats"] > 0 or r["sent_sats"] > 0]
    record("drop transactions that neither pay nor spend from the address", len(rows), len(involved),
           "expected 0; a non-zero count is a data problem")
    if len(involved) != len(rows):
        log.warning("%d transactions do not involve their address: check the amount rule", len(rows) - len(involved))

    frame = pd.DataFrame(involved, columns=COLUMNS)
    missing = frame[REQUIRED].isna().sum()
    if missing.any():
        raise ValueError(f"Missing values after cleaning, which means the API changed: {missing[missing > 0].to_dict()}")
    # Same-block transactions share a time; txid only fixes their order, and they are never part of
    # each other's history (features.py).
    frame = frame.sort_values(["address", "block_time_utc", "txid"], kind="stable").reset_index(drop=True)
    for col in ["block_height", "block_time_utc", "received_sats", "sent_sats", "net_sats", "n_inputs",
                "n_outputs", "fee_sats", "vsize_vb"]:
        frame[col] = frame[col].astype("int64")
    record("sort by address, block time and txid", len(frame), len(frame), "time order for the features")
    return frame, steps


def run_transactions(histories_path: Path = HISTORIES_PATH, out_path: Path = OUT_PATH,
                     tables_dir: Path = TABLES_DIR) -> int:
    histories = json.loads(read_gz(histories_path))
    log.info("Read %s: %d addresses", histories_path, len(histories))
    frame, steps = build_transactions(histories)
    frame.to_csv(out_path, index=False)
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(steps).to_csv(tables_dir / "cleaning_ds1.csv", index=False)
    by_dir = frame["direction"].value_counts().to_dict()
    log.info("Wrote %s: %d rows, %d addresses, directions %s", out_path, len(frame), frame["address"].nunique(), by_dir)
    log.info("Wrote %s", tables_dir / "cleaning_ds1.csv")
    return 0
