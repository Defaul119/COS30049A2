"""Synthetic injection: known unusual transactions planted in copies of real DS1 histories.

Serves plan §6.4 (docs/plan/sections/s06-processing.tex) and A1 §2.3.2. DS1 has no labels, so the
only positives with known answers are ones we plant. Each plant goes into a COPY of one address's
history, inside its evaluation period, so plants never mask one another; features are then
recomputed time-safely for the whole copy (a plant changes the features of everything after it),
and frozen models score the copy.

Patterns (A1 §2.3.2):
    spike   one transaction at a random time between two real ones, in the address's majority
            direction over the previous 30 days, for spikeMultiplier x that direction's median |net|
            so far; input/output counts, fee and size are the address's medians so far
    burst   burstSize transactions inside one random burstWindowHours window, each a copy of a
            randomly drawn earlier real transaction, so only the timing is unusual
    dormant one transaction appended dormantDaysMin-dormantDaysMax days (uniform) after the
            address's last real one, with a typical amount, so only the gap is unusual

plantsPerPattern plants per pattern go to addresses in proportion to their evaluation-period size.
Tuning uses seeds 40000-40004 and reported results use 30049-30053, so no method is tuned on the
plants it is scored on. Planted rows carry is_synthetic = 1 and never enter a fit set.

Writes data/processed/esplora/injections/<set>_<seed>.json (plant specifications, committed) and
data/interim/injections/<method>/<set>_<seed>.parquet (features of every planted copy; rebuilt).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .features import DAY, FeatureConfig, build_features
from .settings import PROJECT_ROOT, SETTINGS
from .snapshot import MANIFEST_PATH
from .split import EVAL

log = logging.getLogger(__name__)

PATTERNS = ("spike", "burst", "dormant")
HOUR = 3600
SPECS_DIR = MANIFEST_PATH.parent / "injections"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "injections"
TX_COLUMNS = ["address", "role", "txid", "block_height", "block_time_utc", "received_sats", "sent_sats", "net_sats",
              "direction", "n_inputs", "n_outputs", "fee_sats", "vsize_vb"]


@dataclass
class Plant:
    """One planted pattern: the synthetic transactions added to one address's history."""

    plant_id: str
    seed: int
    pattern: str
    address: str
    rows: list[dict[str, Any]]


def majority_direction(earlier: pd.DataFrame, t: int, window_days: float) -> str:
    """The address's majority direction (in or out) over the window before t, or over all earlier
    transactions if the window holds none; a tie goes to the most recent transaction's direction."""
    moving = earlier[earlier["direction"] != "none"]
    recent = moving[moving["block_time_utc"] >= t - window_days * DAY]
    pool = recent if len(recent) else moving
    n_in, n_out = int((pool["direction"] == "in").sum()), int((pool["direction"] == "out").sum())
    if n_in == n_out:
        return pool["direction"].iloc[-1] if len(pool) else "in"
    return "in" if n_in > n_out else "out"


def typical_amount(earlier: pd.DataFrame, direction: str) -> float:
    """Median |net| of earlier transactions in `direction` (all earlier ones if none go that way)."""
    same = earlier[earlier["direction"] == direction]
    return float((same if len(same) else earlier)["net_sats"].abs().median())


def synthetic_row(earlier: pd.DataFrame, plant_id: str, k: int, t: int, direction: str, amount: int,
                  shape: pd.Series | None = None) -> dict[str, Any]:
    """A planted transaction. `shape` (a real earlier row) supplies counts, fee and size; otherwise
    the address's medians so far do."""
    if shape is None:
        shape = earlier[["n_inputs", "n_outputs", "fee_sats", "vsize_vb"]].median().round()
    amount = max(int(amount), 1)
    received, sent = (amount, 0) if direction == "in" else (0, amount)
    return {
        "address": earlier["address"].iloc[0], "role": earlier["role"].iloc[0], "txid": f"{plant_id}-{k:02d}",
        "block_height": int(earlier["block_height"].iloc[-1]), "block_time_utc": int(t),
        "received_sats": received, "sent_sats": sent, "net_sats": received - sent, "direction": direction,
        "n_inputs": int(shape["n_inputs"]), "n_outputs": int(shape["n_outputs"]),
        "fee_sats": int(shape["fee_sats"]), "vsize_vb": int(shape["vsize_vb"]),
    }


def plant_spike(h: pd.DataFrame, ev: pd.DataFrame, plant_id: str, rng: np.random.Generator, p: dict[str, Any]) -> list[dict]:
    times = np.unique(ev["block_time_utc"].to_numpy())
    gaps = [(a, b) for a, b in zip(times[:-1], times[1:]) if b - a >= 2]
    a, b = gaps[rng.integers(len(gaps))]
    t = int(rng.integers(a + 1, b))
    earlier = h[h["block_time_utc"] < t]
    direction = majority_direction(earlier, t, SETTINGS["features"]["comparisonWindowDays"])
    amount = p["spikeMultiplier"] * typical_amount(earlier, direction)
    return [synthetic_row(earlier, plant_id, 0, t, direction, round(amount))]


def plant_burst(h: pd.DataFrame, ev: pd.DataFrame, plant_id: str, rng: np.random.Generator, p: dict[str, Any]) -> list[dict]:
    window = p["burstWindowHours"] * HOUR
    lo, last = int(ev["block_time_utc"].min()), int(h["block_time_utc"].max())
    start = int(rng.integers(lo, max(lo, last - window) + 1))
    times = np.sort(rng.choice(np.arange(start, start + window), size=p["burstSize"], replace=False))
    earlier = h[h["block_time_utc"] < start]
    drawn = earlier.iloc[rng.integers(0, len(earlier), size=p["burstSize"])]
    return [synthetic_row(earlier, plant_id, k, int(t), row["direction"] if row["direction"] != "none" else "in",
                          abs(int(row["net_sats"])), shape=row)
            for k, (t, (_, row)) in enumerate(zip(times, drawn.iterrows()))]


def plant_dormant(h: pd.DataFrame, ev: pd.DataFrame, plant_id: str, rng: np.random.Generator, p: dict[str, Any]) -> list[dict]:
    t = int(h["block_time_utc"].max() + rng.uniform(p["dormantDaysMin"], p["dormantDaysMax"]) * DAY)
    direction = majority_direction(h, t, SETTINGS["features"]["comparisonWindowDays"])
    return [synthetic_row(h, plant_id, 0, t, direction, round(typical_amount(h, direction)))]


PLANTERS = {"spike": plant_spike, "burst": plant_burst, "dormant": plant_dormant}


def make_plants(transactions: pd.DataFrame, periods: pd.DataFrame, seed: int, set_name: str,
                p: dict[str, Any] = SETTINGS["injection"]) -> list[Plant]:
    """Plants for one seed: plantsPerPattern per pattern, addresses drawn in proportion to their
    evaluation-period size. An address qualifies if its evaluation period spans at least two block
    times, so a spike has a gap between real transactions to land in."""
    rng = np.random.default_rng(seed)
    tx = transactions.merge(periods[["address", "txid", "period"]], on=["address", "txid"], how="left")
    ev_all = tx[tx["period"] == EVAL]
    eval_counts = ev_all.groupby("address").size()
    qualifies = ev_all.groupby("address")["block_time_utc"].nunique() >= 2
    eval_counts = eval_counts[qualifies.reindex(eval_counts.index, fill_value=False)].sort_index()
    plants = []
    for pattern in PATTERNS:
        chosen = rng.choice(eval_counts.index.to_numpy(), size=p["plantsPerPattern"],
                            p=(eval_counts / eval_counts.sum()).to_numpy())
        for k, address in enumerate(chosen):
            h = tx[tx["address"] == address].sort_values(["block_time_utc", "txid"])
            ev = h[h["period"] == EVAL]
            plant_id = f"{set_name}{seed}-{pattern}-{k:02d}"
            plants.append(Plant(plant_id, seed, pattern, address, PLANTERS[pattern](h[TX_COLUMNS], ev, plant_id, rng, p)))
    return plants


def copy_features(transactions: pd.DataFrame, periods: pd.DataFrame, plant: Plant, cfg: FeatureConfig) -> pd.DataFrame:
    """Features of the planted copy of one address, recomputed time-safely from scratch.

    Real rows keep their period; planted rows are evaluation rows marked is_synthetic = 1.
    """
    real = transactions[transactions["address"] == plant.address]
    copy = pd.concat([real, pd.DataFrame(plant.rows)], ignore_index=True)
    frame = build_features(copy, cfg)
    synthetic = {r["txid"] for r in plant.rows}
    frame["is_synthetic"] = frame["txid"].isin(synthetic).astype("int8")
    frame = frame.merge(periods[["txid", "period"]][periods["address"] == plant.address], on="txid", how="left")
    frame.loc[frame["is_synthetic"] == 1, "period"] = EVAL
    return frame.assign(plant_id=plant.plant_id, pattern=plant.pattern, seed=plant.seed)


def run_inject(n_jobs: int = SETTINGS["classification"]["nJobs"]) -> int:
    """Plant every tuning and reporting seed, and precompute the copies' features for both methods."""
    transactions = pd.read_csv(MANIFEST_PATH.parent / "transactions.csv")
    periods = pd.read_csv(MANIFEST_PATH.parent / "features.csv", usecols=["address", "txid", "period"])
    p = SETTINGS["injection"]
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    configs = {"refined": FeatureConfig.refined(), "as_written": FeatureConfig.as_written()}
    for set_name, seeds in (("tuning", p["tuningSeeds"]), ("reporting", p["reportingSeeds"])):
        for seed in seeds:
            plants = make_plants(transactions, periods, seed, set_name)
            (SPECS_DIR / f"{set_name}_{seed}.json").write_text(
                json.dumps([asdict(pl) for pl in plants], indent=1) + "\n", encoding="utf-8")
            for method, cfg in configs.items():
                frames = Parallel(n_jobs=n_jobs)(delayed(copy_features)(transactions, periods, pl, cfg) for pl in plants)
                out = INTERIM_DIR / method / f"{set_name}_{seed}.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
            log.info("%s seed %d: %d plants (%s) -> %s", set_name, seed, len(plants),
                     ", ".join(f"{pat} {sum(pl.pattern == pat for pl in plants)}" for pat in PATTERNS),
                     SPECS_DIR / f"{set_name}_{seed}.json")
    return 0
