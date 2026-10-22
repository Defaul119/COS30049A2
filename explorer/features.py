"""Features F1-F8, own-history robust z-scores and the baseline flag, for every DS1 transaction.

Serves plan §6.2 (docs/plan/sections/s06-processing.tex) and A1 §2.3. `score_address` is a
line-by-line port of the A1 prototype's scoreAddress (reference/a1_prototype/scoring.ts) and reads
the same settings.json, so the report, the prototype and the pipeline cannot drift apart;
tests/test_features.py checks the two agree on the prototype's own output.

Time safety (DR-02): everything computed for a transaction uses only transactions confirmed
STRICTLY EARLIER at the same address. Transactions in the same block share a time and are never
part of each other's history.

Two comparison windows, both "the previous 30 days, extended back to at least the last 20
transactions" (CR-02):
    timing window - all earlier transactions; used for F2 Burst and F3 Gap
    size window   - earlier transactions in the same direction (CR-04); used for F1 Size and F4
                    Relative size. Without it, an address that collects small payments and sweeps
                    them out every few weeks has every sweep flagged, because each sweep would be
                    compared with small incoming amounts.

`FeatureConfig.as_written()` is A1 §2.3 exactly as written (no window extension, all directions,
no transform, no floor on the spread), for the refinement ablation.

Reads data/processed/esplora/transactions.csv; writes data/processed/esplora/features.csv
(or features_as_written.csv).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .settings import SETTINGS
from .snapshot import MANIFEST_PATH

log = logging.getLogger(__name__)

TRANSACTIONS_PATH = MANIFEST_PATH.parent / "transactions.csv"

FEATURE_IDS = ("F1", "F2", "F3", "F4")
SIZE_FEATURES = ("F1", "F4")
HOUR = 3600
DAY = 86_400


@dataclass(frozen=True)
class FeatureConfig:
    """Every setting the features and the baseline use (settings.json -> features, baseline)."""

    burst_window_days: float
    comparison_window_days: float
    comparison_min_tx: int
    min_comparison_values: int
    min_history_tx: int
    min_history_span_days: float
    same_direction: bool
    transform: bool
    mad_scale: float
    min_scale: dict[str, float]
    flag_abs_z: float
    z_clip: float

    @classmethod
    def refined(cls, settings: dict[str, Any] = SETTINGS) -> FeatureConfig:
        """The method with the CR-02, CR-04 and CR-05 refinements: what the prototype runs."""
        f, b = settings["features"], settings["baseline"]
        return cls(
            burst_window_days=f["burstWindowDays"], comparison_window_days=f["comparisonWindowDays"],
            comparison_min_tx=f["comparisonMinTx"], min_comparison_values=f["minComparisonValues"],
            min_history_tx=f["minHistoryTx"], min_history_span_days=f["minHistorySpanDays"],
            same_direction=f["sizeComparison"] == "same-direction", transform=True,
            mad_scale=b["madScale"], min_scale=dict(b["minScale"]), flag_abs_z=b["flagAbsZ"], z_clip=f["zClip"],
        )

    @classmethod
    def as_written(cls, settings: dict[str, Any] = SETTINGS) -> FeatureConfig:
        """A1 §2.3 as written (settings.json -> featuresAsWritten), for the ablation."""
        w = settings["featuresAsWritten"]
        base = cls.refined(settings)
        return cls(**{
            **base.__dict__, "comparison_min_tx": w["comparisonMinTx"],
            "same_direction": w["sizeComparison"] == "same-direction", "transform": w["transform"],
            "min_scale": dict(w["minScale"]),
        })


@dataclass(frozen=True)
class RawTx:
    """One address-related transaction: the input to scoring, as in scoring.ts."""

    txid: str
    time: int  # unix seconds, UTC
    block_height: int
    received_sats: int
    sent_sats: int


def median(values: list[float]) -> float:
    s = sorted(values)
    mid = len(s) >> 1
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def robust_z(value: float, sample: list[float], min_scale: float, mad_scale: float) -> tuple[float, float]:
    """Modified z-score of `value` against `sample` (Iglewicz and Hoaglin 1993), and the sample median.

    The spread has a floor (`min_scale`) so a history of identical values cannot divide by zero and
    turn every new value into an extreme flag (CR-05). With no floor (the as-written method) a zero
    spread gives z = 0 for an identical value and an infinite z otherwise, which is what that method
    implies. `sample` must hold only values from strictly earlier transactions (DR-02).
    """
    center = median(sample)
    mad = median([abs(v - center) for v in sample])
    scale = max(mad_scale * mad, min_scale)
    diff = value - center
    if scale == 0:
        return (0.0 if diff == 0 else math.copysign(math.inf, diff)), center
    return diff / scale, center


def transform(fid: str, v: float | None, cfg: FeatureConfig) -> float | None:
    """The scale on which the baseline compares each feature (heavy-tailed ones are logged)."""
    if v is None or not cfg.transform:
        return v
    if fid == "F3":
        return math.log10(1 + v)
    if fid == "F4":
        return math.log10(max(v, 0.001))
    return v


def readable(fid: str, t: float, cfg: FeatureConfig) -> float:
    """Back from the comparison scale to readable units, for the "usual" column."""
    if not cfg.transform:
        return 10 ** t - 1 if fid == "F1" else t
    if fid in ("F1", "F3"):
        return 10 ** t - 1  # satoshis, hours
    if fid == "F4":
        return 10 ** t  # ratio
    return t


def direction(t: RawTx) -> str:
    return "in" if t.received_sats > t.sent_sats else "out" if t.received_sats < t.sent_sats else "none"


def collect_window(
    txs: list[RawTx], earlier_end: int, now: int, keep: Callable[[int], bool], cfg: FeatureConfig
) -> tuple[list[int], int]:
    """Indices of earlier transactions (newest first) that pass `keep`: every one in the last
    comparison_window_days, then older ones until at least comparison_min_tx are collected.
    Returns (indices, how many fall inside the period itself)."""
    idx: list[int] = []
    in_period = 0
    for j in range(earlier_end - 1, -1, -1):
        if not keep(j):
            continue
        recent = txs[j].time >= now - cfg.comparison_window_days * DAY
        if not recent and len(idx) >= cfg.comparison_min_tx:
            break
        if recent:
            in_period += 1
        idx.append(j)
    return idx, in_period


def describe(txs: list[RawTx], window: tuple[list[int], int], dir_: str) -> dict[str, Any] | None:
    idx, in_period = window
    if not idx:
        return None
    return {"count": len(idx), "count_in_period": in_period, "extended": len(idx) > in_period,
            "direction": dir_, "start": txs[idx[-1]].time, "end": txs[idx[0]].time}


def score_address(input_txs: list[RawTx], cfg: FeatureConfig) -> list[dict[str, Any]]:
    """Port of scoring.ts scoreAddress: features, windows, z-scores, score and flag per transaction.

    Returns one dict per transaction in time order (ties broken by txid). Feature units: F1
    log10(1 + |net| sats), F2 count, F3 hours, F4 ratio; z is None where it cannot be computed.
    """
    txs = sorted(input_txs, key=lambda t: (t.time, t.txid))
    abs_net = [abs(t.received_sats - t.sent_sats) for t in txs]
    dirs = [direction(t) for t in txs]

    # Earlier transactions of i are indices [0, earlier_end[i]): same-time transactions are excluded.
    earlier_end: list[int] = []
    for i, t in enumerate(txs):
        earlier_end.append(earlier_end[i - 1] if i > 0 and txs[i - 1].time == t.time else i)

    # Pass 1: raw features and the two comparison windows.
    timing = [collect_window(txs, earlier_end[i], t.time, lambda j: True, cfg) for i, t in enumerate(txs)]
    size = [
        collect_window(txs, earlier_end[i], t.time, lambda j, i=i: dirs[j] == dirs[i], cfg)
        if cfg.same_direction else timing[i]
        for i, t in enumerate(txs)
    ]
    features: list[dict[str, float | None]] = []
    for i, t in enumerate(txs):
        e = earlier_end[i]
        burst = 0
        j = e - 1
        while j >= 0 and txs[j].time >= t.time - cfg.burst_window_days * DAY:
            burst += 1
            j -= 1
        usual_abs = median([abs_net[j] for j in size[i][0]]) if size[i][0] else None
        features.append({
            "F1": math.log10(1 + abs_net[i]),
            "F2": burst,
            "F3": (t.time - txs[e - 1].time) / HOUR if e > 0 else None,
            "F4": None if usual_abs is None else abs_net[i] / max(usual_abs, 1),
        })

    # Pass 2: robust z of each feature against the same feature over its comparison window.
    out = []
    for i, t in enumerate(txs):
        e = earlier_end[i]
        history_span_days = (t.time - txs[0].time) / DAY if e > 0 else 0
        scored = e >= cfg.min_history_tx and history_span_days >= cfg.min_history_span_days
        z: dict[str, float | None] = {}
        usual: dict[str, float | None] = {}
        for fid in FEATURE_IDS:
            window = size[i] if fid in SIZE_FEATURES else timing[i]
            own = transform(fid, features[i][fid], cfg)
            sample = [v for v in (transform(fid, features[j][fid], cfg) for j in window[0]) if v is not None]
            usual[fid] = readable(fid, median(sample), cfg) if sample else None
            z[fid] = (None if own is None or len(sample) < cfg.min_comparison_values
                      else robust_z(own, sample, cfg.min_scale[fid], cfg.mad_scale)[0])
        zs = [v for v in z.values() if v is not None]
        score = max(abs(v) for v in zs) if scored and zs else 0
        out.append({
            "txid": t.txid, "time": t.time, "block_height": t.block_height,
            "received_sats": t.received_sats, "sent_sats": t.sent_sats,
            "net_sats": t.received_sats - t.sent_sats, "earlier_count": e,
            "history_span_days": history_span_days, "scored": scored,
            "features": features[i], "usual": usual, "z": z,
            "timing_window": describe(txs, timing[i], "all"),
            "size_window": describe(txs, size[i], dirs[i] if cfg.same_direction else "all"),
            "flagged": scored and score >= cfg.flag_abs_z, "score": score,
        })
    return out


def extended_features(row: dict[str, Any]) -> dict[str, float]:
    """F5-F8 (plan §6.2): used only in the ablation and as TxGPT tokens, never in explanations.

    F7 is log10(1 + fee rate in sat/vB) rather than log10(fee rate), so a zero-fee coinbase
    transaction (a mining payout address) gets 0 instead of minus infinity.
    """
    hour = (row["block_time_utc"] % DAY) / HOUR
    return {
        "F5": math.log2(row["n_inputs"]),
        "F6": math.log2(row["n_outputs"]),
        "F7": math.log10(1 + row["fee_sats"] / row["vsize_vb"]),
        "F8_sin": math.sin(2 * math.pi * hour / 24),
        "F8_cos": math.cos(2 * math.pi * hour / 24),
    }


def window_columns(prefix: str, w: dict[str, Any] | None) -> dict[str, Any]:
    keys = ("count", "count_in_period", "extended", "start", "end")
    return {f"{prefix}_{k}": (w[k] if w else None) for k in keys}


def build_features(transactions: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Score every address in the clean transaction table and return one row per transaction.

    Columns: the transaction's own fields; earlier_count; history_span_days; scorable; F1-F4 and
    their usual values and z-scores (z_*), clipped copies (zc_*, +/- cfg.z_clip) for pooled models;
    window descriptions; score_baseline and flag_baseline; and F5-F8.
    """
    rows = []
    for address, group in transactions.groupby("address", sort=True):
        extra = group.set_index("txid")
        scored = score_address(
            [RawTx(r.txid, int(r.block_time_utc), int(r.block_height), int(r.received_sats), int(r.sent_sats))
             for r in group.itertuples(index=False)],
            cfg,
        )
        for s in scored:
            src = extra.loc[s["txid"]]
            row = {
                "address": address, "role": src["role"], "txid": s["txid"], "block_height": s["block_height"],
                "block_time_utc": s["time"], "received_sats": s["received_sats"], "sent_sats": s["sent_sats"],
                "net_sats": s["net_sats"], "direction": src["direction"], "n_inputs": src["n_inputs"],
                "n_outputs": src["n_outputs"], "fee_sats": src["fee_sats"], "vsize_vb": src["vsize_vb"],
                "earlier_count": s["earlier_count"], "history_span_days": s["history_span_days"],
                "scorable": s["scored"],
            }
            for fid in FEATURE_IDS:
                z = s["z"][fid]
                row[fid] = s["features"][fid]
                row[f"usual_{fid}"] = s["usual"][fid]
                row[f"z_{fid}"] = z
                row[f"zc_{fid}"] = None if z is None else max(-cfg.z_clip, min(cfg.z_clip, z))
            row.update(window_columns("timing_window", s["timing_window"]))
            row.update(window_columns("size_window", s["size_window"]))
            row["size_window_direction"] = s["size_window"]["direction"] if s["size_window"] else None
            row["score_baseline"] = s["score"]
            row["flag_baseline"] = s["flagged"]
            row.update(extended_features(row))
            rows.append(row)
    return pd.DataFrame(rows)


def run_features(as_written: bool = False, transactions_path: Path = TRANSACTIONS_PATH) -> int:
    """Score every DS1 transaction and write features.csv (or features_as_written.csv)."""
    cfg = FeatureConfig.as_written() if as_written else FeatureConfig.refined()
    transactions = pd.read_csv(transactions_path)
    frame = build_features(transactions, cfg)
    out_path = transactions_path.parent / ("features_as_written.csv" if as_written else "features.csv")
    frame.to_csv(out_path, index=False)
    scorable = frame["scorable"]
    log.info("%s method: %d transactions, %d scorable, %d flagged by the baseline (%.1f%% of scorable)",
             "As-written" if as_written else "Refined", len(frame), scorable.sum(), frame["flag_baseline"].sum(),
             100 * frame["flag_baseline"].sum() / max(scorable.sum(), 1))
    log.info("Wrote %s", out_path)
    return 0
