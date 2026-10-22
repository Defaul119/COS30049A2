"""The transaction tokenizer for TxGPT: each transaction becomes 6 tokens.

Serves plan §6.7 (docs/plan/sections/s06-processing.tex). Fields, always in this order (gap first,
so the model predicts a transaction's content knowing when it happened):

    GAP   18  FIRST (no earlier transaction), SAME_BLOCK, then hours since the previous transaction
              with upper bounds GAP_HOURS and one bucket above the last
    DIR    3  IN, OUT, ZERO (net 0)
    AMT   47  floor(log2 |net sats|) clipped to 0-46: one bucket per doubling (net 0 -> 0)
    NIN    7  input count: 1, 2, 3-4, 5-9, 10-19, 20-49, 50 or more
    NOUT   7  output count, the same buckets
    FEE    9  fee rate in sat/vB: below 1, then doubling bounds 2 ... 128, and above

plus PAD, BOS (start of a history) and UNK: 94 ids, padded to 128 by lm.model.pad_vocab. Each field
has its own id range, so a token always says which field it is. The bucket edges are written into
the data's meta.json, and scoring reads them from there, so training and scoring use identical
buckets.

A history is BOS followed by 6 tokens per transaction, oldest first. Transactions in the same
block are ordered by txid (hashed in DS2) and their gap token is SAME_BLOCK.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from typing import Any, Iterable

PAD, BOS, UNK = 0, 1, 2
SPECIALS = ["PAD", "BOS", "UNK"]

GAP_HOURS = [1 / 6, 1, 3, 6, 12, 24, 48, 96, 168, 336, 720, 1440, 2160, 4320, 8760]
COUNT_BOUNDS = [1, 2, 4, 9, 19, 49]  # inclusive upper bounds: 1, 2, 3-4, 5-9, 10-19, 20-49, then 50+
FEE_BOUNDS = [1, 2, 4, 8, 16, 32, 64, 128]  # exclusive upper bounds in sat/vB, then 128+
AMT_MAX = 46

FIELDS = ["GAP", "DIR", "AMT", "NIN", "NOUT", "FEE"]
FIELD_SIZES = {
    "GAP": 2 + len(GAP_HOURS) + 1,
    "DIR": 3,
    "AMT": AMT_MAX + 1,
    "NIN": len(COUNT_BOUNDS) + 1,
    "NOUT": len(COUNT_BOUNDS) + 1,
    "FEE": len(FEE_BOUNDS) + 1,
}
FIELD_OFFSETS: dict[str, int] = {}
_next = len(SPECIALS)
for _field in FIELDS:
    FIELD_OFFSETS[_field] = _next
    _next += FIELD_SIZES[_field]
VOCAB_SIZE = _next  # 94
TOKENS_PER_TX = len(FIELDS)
HOUR = 3600


def gap_bucket(gap_seconds: float | None, same_block: bool) -> int:
    """0 FIRST, 1 SAME_BLOCK, then 2 + the index of the first upper bound the gap does not exceed."""
    if gap_seconds is None:
        return 0
    if same_block:
        return 1
    return 2 + bisect_left(GAP_HOURS, gap_seconds / HOUR)


def amount_bucket(abs_net_sats: int) -> int:
    return 0 if abs_net_sats <= 0 else min(int(math.floor(math.log2(abs_net_sats))), AMT_MAX)


def count_bucket(n: int) -> int:
    return bisect_left(COUNT_BOUNDS, n)


def fee_bucket(fee_sats: float, vsize_vb: float) -> int:
    rate = fee_sats / vsize_vb if vsize_vb else 0.0
    return next((i for i, bound in enumerate(FEE_BOUNDS) if rate < bound), len(FEE_BOUNDS))


def tx_tokens(row: dict[str, Any], prev: dict[str, Any] | None) -> list[int]:
    """The 6 token ids of one transaction, given the previous one in the history (or None).

    `row` needs block_height, block_time_utc, received_sats, sent_sats, n_inputs, n_outputs,
    fee_sats and vsize_vb.
    """
    net = int(row["received_sats"]) - int(row["sent_sats"])
    gap = None if prev is None else float(row["block_time_utc"]) - float(prev["block_time_utc"])
    same_block = prev is not None and int(row["block_height"]) == int(prev["block_height"])
    buckets = {
        "GAP": gap_bucket(gap, same_block),
        "DIR": 0 if net > 0 else 1 if net < 0 else 2,
        "AMT": amount_bucket(abs(net)),
        "NIN": count_bucket(int(row["n_inputs"])),
        "NOUT": count_bucket(int(row["n_outputs"])),
        "FEE": fee_bucket(float(row["fee_sats"]), float(row["vsize_vb"])),
    }
    return [FIELD_OFFSETS[f] + buckets[f] for f in FIELDS]


def history_tokens(rows: Iterable[dict[str, Any]]) -> list[int]:
    """BOS followed by 6 tokens per transaction; `rows` must already be in time order."""
    out, prev = [BOS], None
    for row in rows:
        out += tx_tokens(row, prev)
        prev = row
    return out


def field_of(token: int) -> str:
    """Which field (or special) a token id belongs to."""
    if token < len(SPECIALS):
        return SPECIALS[token]
    for f in FIELDS:
        if FIELD_OFFSETS[f] <= token < FIELD_OFFSETS[f] + FIELD_SIZES[f]:
            return f
    return "UNK"


def describe() -> dict[str, Any]:
    """The vocabulary and bucket edges, for meta.json."""
    return {"vocab_size": VOCAB_SIZE, "specials": SPECIALS, "fields": FIELDS, "field_sizes": FIELD_SIZES,
            "field_offsets": FIELD_OFFSETS, "gap_hours_upper_bounds": GAP_HOURS,
            "count_upper_bounds_inclusive": COUNT_BOUNDS, "fee_rate_upper_bounds_exclusive": FEE_BOUNDS,
            "amount_max_bucket": AMT_MAX}
