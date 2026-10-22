"""DS1: choose the snapshot addresses, fetch their full histories, verify them, write the manifest.

Serves plan §6.1 (docs/plan/sections/s05-data.tex): A1 Objective 1, DR-01 (provenance manifest),
risk 18 (amounts derived correctly) and risk 19 (choosing addresses because they already look odd).

The selection rule is written into the manifest before anything is fetched (risk 19). Candidates
are the output addresses of transactions in blocks spread over the four weeks before the tip; they
are shuffled with the seed and visited in that order, and every visited candidate is logged with
the reason it was kept or skipped.

Every API response is saved unchanged (gzipped) under data/raw/esplora/, and a re-run reads that
cache instead of the network. An interrupted run therefore resumes where it stopped, and a finished
one rebuilds offline with identical results. `refetch` clears the cache and starts from a new tip.

Writes:
    data/raw/esplora/selection/          tip height, the sampled blocks, every candidate's /address
                                         reply (candidates.jsonl) and one line per run (runs.jsonl)
    data/raw/esplora/<address>/          page-NNN.json.gz (history, newest first), address.json.gz
                                         (the chain_stats the pages match), mempool-page-001.json.gz
                                         (fallback cross-check, first addresses only)
    data/raw/esplora/skipped/<address>/  histories fetched but not kept
    data/processed/esplora/manifest.json the DR-01 manifest
    data/processed/esplora/histories.json.gz  the DS1 dataset: each kept address's transactions,
                                         trimmed to the fields the pipeline uses (trim_tx)

The raw folder is git-ignored: busy addresses' pages make it about 0.4 GB. The manifest records
the SHA-256 of every raw file, and the trimmed histories are what the repository and zip carry.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import random
import shutil
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any

from .amounts import AddressFlow, address_flow
from .esplora import PAGE_SIZE, EsploraClient, EsploraError
from .settings import PROJECT_ROOT, SETTINGS

log = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "esplora"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "esplora" / "manifest.json"
HISTORIES_FILE = "histories.json.gz"  # beside the manifest: the DS1 dataset the pipeline reads

SELECTION_DIR = "selection"
SKIPPED_DIR = "skipped"
TIP_FILE = "tip-height.txt.gz"
CANDIDATES_FILE = "candidates.jsonl"
RUNS_FILE = "runs.jsonl"
STATS_FILE = "address.json.gz"  # written last, so its presence marks a complete history
FALLBACK_PAGE_FILE = "mempool-page-001.json.gz"

SECONDS_PER_DAY = 86_400
PROGRESS_EVERY = 100  # candidates between progress lines in the log

# The parameters of settings.json -> collection that define the selection rule.
RULE_KEYS = [
    "selectionSeed", "snapshotBlockCount", "snapshotOldestBlockBack", "snapshotNewestBlockBack",
    "snapshotTxPagesPerBlock", "mainMinTx", "mainMaxTx", "mainMinSpanDays", "targetMainAddresses",
    "shortMinTx", "shortMaxTx", "targetShortAddresses", "maxTxPerAddress",
]

SKIP_REASONS = {
    "too_few": "fewer confirmed transactions than the short-history range",
    "between_ranges": "more than the short-history range but fewer than the main range",
    "too_many": "more confirmed transactions than the main range (busy services; a stated bias)",
    "span_too_short": "main range, but the full history spans fewer days than the minimum",
    "target_reached": "in range, but that group's target was already met",
    "checks_failed": "fetched, but an automated check failed",
    "fetch_error": "the API still returned an error after every retry",
}

FIELDS_USED = [
    "txid", "status.confirmed", "status.block_height", "status.block_time",
    "vin[].prevout.scriptpubkey_address", "vin[].prevout.value",
    "vout[].scriptpubkey_address", "vout[].scriptpubkey_type", "vout[].value", "fee", "weight",
    "/address: chain_stats.tx_count, funded_txo_sum, spent_txo_sum (checks only)",
]

TERMS_NOTE = (
    "The Esplora API documentation (" + SETTINGS["source"]["apiDocs"] + ") states no rate limits or "
    "terms of use. Blockstream's general Terms of Use (v1.0, 16 Jan 2018, https://blockstream.com/terms/) "
    "cover its website and services but do not mention the Explorer or its API. This project sends at "
    "most two requests per second, only GET requests, and calls the API offline to build the snapshot. "
    "Addresses are public blockchain data; nothing here says who controls an address or why it transacts."
)


# -- cache helpers -------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_iso(unix_seconds: int) -> str:
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).isoformat(timespec="seconds")


def write_gz(path: Path, text: str) -> None:
    """Gzip `text` to `path`, atomically.

    mtime=0 keeps the bytes, and so the SHA-256 in the manifest, identical when the same reply is
    written again. Writing to a temporary file first means an interrupted run never leaves a
    truncated file that a re-run would trust as cached.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
    tmp.replace(path)


def read_gz(path: Path) -> str:
    return gzip.decompress(path.read_bytes()).decode("utf-8")


def cached_get(client: EsploraClient, api_path: str, cache_file: Path) -> str:
    """The cached reply for `api_path` if there is one; otherwise fetch it and cache it unchanged."""
    if cache_file.exists():
        return read_gz(cache_file)
    text = client.get_text(api_path)
    write_gz(cache_file, text)
    return text


class CandidateStats:
    """Every candidate's /address reply, one JSON line each, so a re-run never asks twice."""

    def __init__(self, path: Path):
        self.path = path
        self.replies: dict[str, str] = {}
        if not path.exists():
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        good: list[str] = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut short by an interrupted run; that candidate is asked again
            self.replies[row["address"]] = row["reply"]
            good.append(line)
        if len(good) != len(lines):  # rewrite without the broken line, so the next append starts clean
            path.write_text("".join(f"{line}\n" for line in good), encoding="utf-8")

    def get(self, client: EsploraClient, address: str) -> dict[str, Any]:
        if address not in self.replies:
            reply = client.get_text(f"/address/{address}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"address": address, "fetched_utc": utc_now(), "reply": reply}) + "\n")
            self.replies[address] = reply
        return json.loads(self.replies[address])


# -- the selection rule ------------------------------------------------------------


def selection_rule(params: dict[str, Any]) -> dict[str, Any]:
    """The selection rule in words, with every parameter, as recorded before any fetch."""
    p = params
    return {
        "steps": [
            f"Read the tip height once. Take {p['snapshotBlockCount']} block heights spread evenly from "
            f"tip - {p['snapshotOldestBlockBack']} to tip - {p['snapshotNewestBlockBack']}: about four weeks "
            "of blocks, far enough from the tip to avoid reorganisations.",
            f"From each block, take the output addresses of up to {p['snapshotTxPagesPerBlock']} pages of "
            f"{PAGE_SIZE} transactions. Remove duplicates and record each address's script type.",
            f"Sort the pool, shuffle it with seed {p['selectionSeed']}, and visit candidates in that order "
            "with one /address request each, reading chain_stats.tx_count.",
            f"Main addresses: keep a candidate with {p['mainMinTx']}-{p['mainMaxTx']} confirmed transactions "
            f"whose full history spans at least {p['mainMinSpanDays']} days (checked after fetching). "
            f"Stop at {p['targetMainAddresses']}.",
            f"Short-history addresses: keep the first {p['targetShortAddresses']} candidates with "
            f"{p['shortMinTx']}-{p['shortMaxTx']} confirmed transactions.",
            "A fetched candidate is kept only if every automated check passes. Every visited candidate is "
            "logged with the reason it was kept or skipped. Visiting stops when both targets are met.",
        ],
        "parameters": {key: p[key] for key in RULE_KEYS},
        "seed": p["selectionSeed"],
    }


def block_heights(tip_height: int, count: int, oldest_back: int, newest_back: int) -> list[int]:
    """`count` block heights spread evenly from tip - oldest_back to tip - newest_back, oldest first."""
    oldest, newest = tip_height - oldest_back, tip_height - newest_back
    step = (newest - oldest) / (count - 1)
    return [oldest + round(i * step) for i in range(count)]


def add_output_addresses(txs: list[dict[str, Any]], pool: dict[str, str]) -> int:
    """Add each output address not already in `pool`, mapped to its script type. Returns how many were new."""
    added = 0
    for tx in txs:
        for out in tx.get("vout", []):
            address = out.get("scriptpubkey_address")
            if address and address not in pool:
                pool[address] = out.get("scriptpubkey_type", "unknown")
                added += 1
    return added


def count_class(tx_count: int, params: dict[str, Any]) -> str:
    """Which range a confirmed-transaction count falls in: main, short, or the skip reason."""
    if params["mainMinTx"] <= tx_count <= params["mainMaxTx"]:
        return "main"
    if params["shortMinTx"] <= tx_count <= params["shortMaxTx"]:
        return "short"
    if tx_count < params["shortMinTx"]:
        return "too_few"
    if tx_count < params["mainMinTx"]:
        return "between_ranges"
    return "too_many"


def read_block(
    client: EsploraClient, height: int, pages_per_block: int, blocks_dir: Path
) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    """One sampled block's summary and its first transaction pages, cached under blocks_dir/<height>/."""
    folder = blocks_dir / str(height)
    block_hash = cached_get(client, f"/block-height/{height}", folder / "hash.txt.gz").strip()
    header = json.loads(cached_get(client, f"/block/{block_hash}", folder / "block.json.gz"))
    # A block with fewer transactions than the pages asked for would make the later start indexes
    # invalid, so the page count comes from the block's own tx_count.
    n_pages = min(pages_per_block, math.ceil(header["tx_count"] / PAGE_SIZE))
    pages = [
        json.loads(cached_get(client, f"/block/{block_hash}/txs/{start}", folder / f"txs-{start:04d}.json.gz"))
        for start in range(0, n_pages * PAGE_SIZE, PAGE_SIZE)
    ]
    summary = {"height": height, "hash": block_hash, "time_utc": utc_iso(header["timestamp"]),
               "tx_count": header["tx_count"], "pages_read": n_pages}
    return summary, pages


def pool_from_blocks(
    blocks: list[tuple[dict[str, Any], list[list[dict[str, Any]]]]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The candidate pool (address -> script type) from read blocks, in block order, with per-block counts."""
    pool: dict[str, str] = {}
    summaries = []
    for summary, pages in blocks:
        added = sum(add_output_addresses(page, pool) for page in pages)
        summaries.append({**summary, "new_candidates": added})
        log.info("Block %d: %d pages read, %d new candidates (pool %d)",
                 summary["height"], summary["pages_read"], added, len(pool))
    return summaries, pool


def sample_blocks(
    client: EsploraClient, tip_height: int, params: dict[str, Any], selection_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Read the DS1 sampled blocks and build the candidate pool. Returns block summaries and the pool."""
    heights = block_heights(tip_height, params["snapshotBlockCount"], params["snapshotOldestBlockBack"],
                            params["snapshotNewestBlockBack"])
    return pool_from_blocks(
        [read_block(client, h, params["snapshotTxPagesPerBlock"], selection_dir / "blocks") for h in heights]
    )


# -- histories -------------------------------------------------------------------


STATUS_FIELDS = ("confirmed", "block_height", "block_hash", "block_time")
SCRIPT_FIELDS = ("scriptpubkey_address", "scriptpubkey_type", "value")


def trim_tx(tx: dict[str, Any], address: str) -> dict[str, Any]:
    """The parts of a transaction the pipeline uses, for one address.

    Keeps the status, fee, weight, size, input and output counts, and only the inputs and outputs
    that belong to `address` (with their positions). `amounts.address_flow` gives the same received
    and sent on this as on the full transaction. Raw pages of consolidation-heavy addresses reach
    about 3 MB each gzipped; the trimmed form keeps DS1 small enough to commit and ship.
    """
    status = tx.get("status", {})
    return {
        "txid": tx["txid"],
        "status": {k: status.get(k) for k in STATUS_FIELDS},
        "fee": tx.get("fee"), "weight": tx.get("weight"), "size": tx.get("size"),
        "n_vin": len(tx.get("vin", [])), "n_vout": len(tx.get("vout", [])),
        "vin": [{"n": i, "prevout": {k: v["prevout"].get(k) for k in SCRIPT_FIELDS}}
                for i, v in enumerate(tx.get("vin", []))
                if v.get("prevout") and v["prevout"].get("scriptpubkey_address") == address],
        "vout": [{"n": i, **{k: o.get(k) for k in SCRIPT_FIELDS}}
                 for i, o in enumerate(tx.get("vout", [])) if o.get("scriptpubkey_address") == address],
    }


MISSING_KEYS = (
    "txid", "status.block_height", "status.block_time", "fee", "weight",
    "vin[].prevout on a non-coinbase input", "vout[].value",
    "vout[].scriptpubkey_address (OP_RETURN and bare scripts have none; expected, not an error)",
)


def tx_missing(tx: dict[str, Any]) -> list[int]:
    """Missing-value counts for one raw transaction, in MISSING_KEYS order (DR-01)."""
    status = tx.get("status", {})
    vin, vout = tx.get("vin", []), tx.get("vout", [])
    return [
        int("txid" not in tx), int(status.get("block_height") is None), int(status.get("block_time") is None),
        int(tx.get("fee") is None), int(tx.get("weight") is None),
        sum(1 for v in vin if not v.get("is_coinbase") and not v.get("prevout")),
        sum(1 for o in vout if o.get("value") is None),
        sum(1 for o in vout if not o.get("scriptpubkey_address")),
    ]


@dataclass
class History:
    """One address's confirmed history, newest first, with each transaction trimmed (trim_tx)."""

    address: str
    folder: Path
    stats: dict[str, Any]  # the /address reply the pages were checked against
    txs: list[dict[str, Any]]
    page_sizes: list[int]
    missing: list[int]  # tx_missing summed over the raw transactions

    @classmethod
    def from_pages(cls, address: str, folder: Path, stats: dict[str, Any],
                   pages: list[list[dict[str, Any]]]) -> History:
        raw = [tx for page in pages for tx in page]
        missing = [sum(col) for col in zip(*(tx_missing(tx) for tx in raw))] or [0] * len(MISSING_KEYS)
        return cls(address, folder, stats, [trim_tx(tx, address) for tx in raw], [len(p) for p in pages], missing)

    @cached_property
    def flows(self) -> list[AddressFlow]:
        return [address_flow(tx, self.address) for tx in self.txs]

    @property
    def span_days(self) -> float:
        """Days from the first to the last confirmed transaction (0 if any block time is missing)."""
        times = [f.block_time for f in self.flows if f.block_time]
        return (max(times) - min(times)) / SECONDS_PER_DAY if len(times) == len(self.flows) and times else 0.0

    @property
    def script_type(self) -> str:
        for tx in self.txs:
            for out in tx["vout"]:
                return out.get("scriptpubkey_type") or "unknown"
            for vin in tx["vin"]:
                return vin["prevout"].get("scriptpubkey_type") or "unknown"
        return "unknown"


def find_cached_history(address: str, raw_dir: Path) -> Path | None:
    """The folder holding a complete saved history for `address`, kept or skipped, if any."""
    for folder in (raw_dir / address, raw_dir / SKIPPED_DIR / address):
        if (folder / STATS_FILE).exists():
            return folder
    return None


def fetch_history(client: EsploraClient, address: str, folder: Path, params: dict[str, Any]) -> None:
    """Fetch the confirmed history newest first, following the last-seen-txid cursor, saving each page.

    chain_stats is read before and after the pages. If a block lands mid-fetch the two differ and
    the history is fetched again (up to `historyFetchAttempts` times), so the saved pages match the
    saved stats. The stats file is written last, marking the history complete.
    """
    for attempt in range(1, params["historyFetchAttempts"] + 1):
        shutil.rmtree(folder, ignore_errors=True)
        before = client.get_text(f"/address/{address}")
        count, cursor, page_no = 0, None, 1
        while count < params["maxTxPerAddress"]:
            suffix = f"/{cursor}" if cursor else ""
            text = client.get_text(f"/address/{address}/txs/chain{suffix}")
            write_gz(folder / f"page-{page_no:03d}.json.gz", text)
            page = json.loads(text)
            count += len(page)
            if len(page) < PAGE_SIZE:
                break
            cursor = page[-1]["txid"]
            page_no += 1
        after = client.get_text(f"/address/{address}")
        if json.loads(before)["chain_stats"] == json.loads(after)["chain_stats"]:
            break
        log.info("%s changed during the fetch (attempt %d); fetching again", address, attempt)
    write_gz(folder / STATS_FILE, after)


def read_history(address: str, folder: Path) -> History:
    pages = [json.loads(read_gz(p)) for p in sorted(folder.glob("page-*.json.gz"))]
    return History.from_pages(address, folder, json.loads(read_gz(folder / STATS_FILE)), pages)


def load_history(client: EsploraClient, address: str, raw_dir: Path, params: dict[str, Any]) -> History:
    """The saved history if complete, otherwise fetched now into data/raw/esplora/<address>/."""
    folder = find_cached_history(address, raw_dir)
    if folder is None:
        folder = raw_dir / address
        fetch_history(client, address, folder, params)
    return read_history(address, folder)


def place_history(history: History, raw_dir: Path, kept: bool) -> None:
    """Move the history to the kept folder or to skipped/, so data/raw/esplora/<address> is exactly DS1."""
    target = raw_dir / history.address if kept else raw_dir / SKIPPED_DIR / history.address
    if history.folder != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(history.folder), str(target))
        history.folder = target


# -- checks ----------------------------------------------------------------------


def verify_history(history: History) -> list[dict[str, Any]]:
    """The automated DS1 checks for one address (plan §6.1, risk 18). Each is {name, passed, detail}."""
    chain = history.stats["chain_stats"]
    flows = history.flows
    txids = [f.txid for f in flows]
    received = sum(f.received for f in flows)
    sent = sum(f.sent for f in flows)
    sizes = history.page_sizes

    def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
        return {"name": name, "passed": bool(passed), "detail": detail}

    return [
        check("count equals chain_stats.tx_count", len(flows) == chain["tx_count"],
              f"{len(flows)} fetched, {chain['tx_count']} reported"),
        check("no duplicate txids", len(set(txids)) == len(txids), f"{len(set(txids))} unique of {len(txids)}"),
        check("every transaction confirmed with a block time", all(f.confirmed and f.block_time for f in flows),
              "status.confirmed and status.block_time"),
        check("summed received equals funded_txo_sum", received == chain["funded_txo_sum"],
              f"{received} vs {chain['funded_txo_sum']} sats"),
        check("summed sent equals spent_txo_sum", sent == chain["spent_txo_sum"],
              f"{sent} vs {chain['spent_txo_sum']} sats"),
        check(f"every page but the last holds {PAGE_SIZE}",
              all(s == PAGE_SIZE for s in sizes[:-1]) and bool(sizes) and sizes[-1] <= PAGE_SIZE, f"sizes {sizes}"),
        check("every transaction involves the address", all(f.received > 0 or f.sent > 0 for f in flows),
              "received or sent > 0"),
    ]


def cross_check(fallback: EsploraClient, history: History) -> dict[str, Any]:
    """Compare the fallback source's newest page with ours: same block time and amounts per shared txid.

    Transactions confirmed after our fetch appear only on the fallback page and are not compared.
    """
    result: dict[str, Any] = {"address": history.address, "source": fallback.base_url}
    try:
        text = cached_get(fallback, f"/address/{history.address}/txs/chain", history.folder / FALLBACK_PAGE_FILE)
    except EsploraError as err:
        return {**result, "passed": False, "shared": 0, "mismatched": [], "detail": str(err)}
    ours = {f.txid: f for f in history.flows}
    theirs = [address_flow(tx, history.address) for tx in json.loads(text)]
    shared = [f for f in theirs if f.txid in ours]
    mismatched = [
        f.txid for f in shared
        if (f.block_time, f.received, f.sent) != (ours[f.txid].block_time, ours[f.txid].received, ours[f.txid].sent)
    ]
    return {
        **result, "passed": bool(shared) and not mismatched, "shared": len(shared), "mismatched": mismatched,
        "detail": f"{len(shared)} transactions on both first pages; {len(mismatched)} differ in time or amount",
    }


def missing_values(histories: list[History]) -> dict[str, int]:
    """Missing-value counts over every kept raw transaction, for the DR-01 manifest."""
    return {key: sum(h.missing[i] for h in histories) for i, key in enumerate(MISSING_KEYS)}


def write_histories(path: Path, histories: list[History], roles: dict[str, str]) -> dict[str, Any]:
    """Write the shipped DS1 file: every kept address's stats and trimmed transactions, newest first.

    Returns its path, size and SHA-256 for the manifest. The raw pages stay in data/raw/esplora/
    (git-ignored, recorded by SHA-256); the pipeline from `transactions` on reads only this file.
    """
    body = {h.address: {"role": roles[h.address], "chain_stats": h.stats["chain_stats"], "txs": h.txs}
            for h in histories}
    write_gz(path, json.dumps(body, sort_keys=True, separators=(",", ":")))
    return {"path": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "contents": "address -> {role, chain_stats, txs}; each tx trimmed to its status, fee, weight, size, "
                        "input and output counts, and the inputs and outputs belonging to the address"}


def provenance(raw_dir: Path) -> list[dict[str, Any]]:
    """Path, size and SHA-256 of every saved reply except skipped histories."""
    files = []
    for path in sorted(raw_dir.rglob("*")):
        rel = path.relative_to(raw_dir)
        if path.is_file() and rel.parts[0] != SKIPPED_DIR and not path.name.endswith(".tmp"):
            files.append({
                "path": rel.as_posix(), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    return files


# -- the run -----------------------------------------------------------------------


def address_summary(history: History, role: str, visit_rank: int, raw_dir: Path) -> dict[str, Any]:
    times = [f.block_time for f in history.flows]
    return {
        "address": history.address, "role": role, "visit_rank": visit_rank, "script_type": history.script_type,
        "tx_count": len(history.flows),
        "in": sum(f.direction == "in" for f in history.flows),
        "out": sum(f.direction == "out" for f in history.flows),
        "first_tx_utc": utc_iso(min(times)), "last_tx_utc": utc_iso(max(times)),
        "span_days": round(history.span_days, 1),
        "received_sats": sum(f.received for f in history.flows), "sent_sats": sum(f.sent for f in history.flows),
        "pages": len(history.page_sizes), "folder": history.folder.relative_to(raw_dir).as_posix(),
    }


def request_totals(runs: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for source in ("primary", "fallback"):
        keys = ("requests", "retries", "http_429", "http_5xx", "network_errors")
        totals[source] = {k: sum(run[source][k] for run in runs) for k in keys}
        medians = [run[source]["median_latency_ms"] for run in runs if run[source]["median_latency_ms"] is not None]
        totals[source]["median_of_run_median_latency_ms"] = round(statistics.median(medians)) if medians else None
    return totals


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def base_manifest(params: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    """The manifest's fixed part. The rule's recorded time, the manual check and the review are
    carried over from an earlier manifest with the same rule, so a re-run does not erase them."""
    rule = selection_rule(params)
    same_rule = previous.get("selection_rule", {}).get("parameters") == rule["parameters"]
    rule["recorded_utc"] = previous["selection_rule"]["recorded_utc"] if same_rule else utc_now()
    source = SETTINGS["source"]
    return {
        "dataset": "DS1: Esplora snapshot of 25 main and 3 short-history Bitcoin addresses",
        "status": "selection rule recorded; nothing fetched yet",
        "limitation": SETTINGS["limitationLine"],
        "source": {k: source[k] for k in ("name", "apiBase", "fallbackName", "fallbackApiBase", "apiDocs")},
        "selection_rule": rule,
        "manual_check": previous.get("manual_check") if same_rule and previous.get("manual_check") else {
            "instructions": "A second team member opens one kept transaction on the block explorer, finds the "
                            "address in its inputs and outputs, and confirms the received and sent values "
                            "derived by explorer/amounts.py.",
            "checked_by": None, "date": None, "txid": None, "result": None,
        },
        "review": previous.get("review") if same_rule and previous.get("review") else {
            "instructions": "A second team member reproduces the retrieval from the rebuild steps and reviews "
                            "this manifest.",
            "reviewed_by": None, "date": None, "notes": None,
        },
        "terms": TERMS_NOTE,
        "rebuild": [
            "conda activate cos30049-a2",
            "python -m explorer.cli snapshot            # re-verifies from data/raw/esplora; no network",
            "python -m explorer.cli snapshot --refetch  # new tip: different addresses and numbers",
        ],
    }


def run_snapshot(
    primary: EsploraClient,
    fallback: EsploraClient,
    params: dict[str, Any],
    raw_dir: Path = RAW_DIR,
    manifest_path: Path = MANIFEST_PATH,
    refetch: bool = False,
    rule_only: bool = False,
) -> int:
    """Select, fetch, verify and write the manifest. Returns 0 when both targets are met and every
    check passes, 1 otherwise (the manifest is written either way, with the reason in `status`)."""
    if refetch and raw_dir.exists():
        log.info("--refetch: clearing %s", raw_dir)
        shutil.rmtree(raw_dir)
    previous = {} if refetch or not manifest_path.exists() else json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = base_manifest(params, previous)
    write_manifest(manifest_path, manifest)  # the rule is on disk before the first request (risk 19)
    log.info("Selection rule written to %s (recorded %s)", manifest_path, manifest["selection_rule"]["recorded_utc"])
    if rule_only:
        return 0

    selection_dir = raw_dir / SELECTION_DIR
    run: dict[str, Any] = {"started_utc": utc_now(), "completed": False}
    try:
        tip_height = int(cached_get(primary, "/blocks/tip/height", selection_dir / TIP_FILE))
        log.info("Tip height %d", tip_height)
        blocks, pool = sample_blocks(primary, tip_height, params, selection_dir)

        order = sorted(pool)
        random.Random(params["selectionSeed"]).shuffle(order)
        stats_cache = CandidateStats(selection_dir / CANDIDATES_FILE)
        targets = {"main": params["targetMainAddresses"], "short": params["targetShortAddresses"]}
        kept: dict[str, list[tuple[History, int]]] = {"main": [], "short": []}
        skipped: list[dict[str, Any]] = []
        visited = 0
        for rank, address in enumerate(order, 1):
            if all(len(kept[g]) >= targets[g] for g in targets):
                break
            visited = rank
            if rank % PROGRESS_EVERY == 0:
                log.info("Visited %d of %d candidates; kept %d main, %d short",
                         rank, len(order), len(kept["main"]), len(kept["short"]))

            def skip(reason: str, detail: str = "", tx_count: int | None = None) -> None:
                skipped.append({"rank": rank, "address": address, "tx_count": tx_count, "reason": reason,
                                "detail": detail})

            try:
                tx_count = stats_cache.get(primary, address)["chain_stats"]["tx_count"]
            except EsploraError as err:
                skip("fetch_error", str(err))
                continue
            group = count_class(tx_count, params)
            if group not in kept:
                skip(group, tx_count=tx_count)
                continue
            if len(kept[group]) >= targets[group]:
                skip("target_reached", f"{group} target of {targets[group]} already met", tx_count)
                continue
            try:
                history = load_history(primary, address, raw_dir, params)
            except EsploraError as err:
                skip("fetch_error", str(err), tx_count)
                continue
            failed = [c["name"] for c in verify_history(history) if not c["passed"]]
            if failed:
                skip("checks_failed", "; ".join(failed), tx_count)
            elif group == "main" and history.span_days < params["mainMinSpanDays"]:
                skip("span_too_short", f"{history.span_days:.1f} days", tx_count)
            else:
                kept[group].append((history, rank))
            is_kept = kept[group][-1][0] is history if kept[group] else False
            place_history(history, raw_dir, is_kept)
            log.info("Candidate %d %s: %d tx, %s -> %s", rank, address, tx_count, group,
                     "kept" if is_kept else skipped[-1]["reason"])

        crosses = [cross_check(fallback, h) for h, _ in kept["main"][: params["crossCheckAddresses"]]]
        histories = [h for g in ("main", "short") for h, _ in kept[g]]
        per_address = [{"address": h.address, "checks": verify_history(h)} for h in histories]
        all_checks = all(c["passed"] for a in per_address for c in a["checks"]) and all(c["passed"] for c in crosses)
        targets_met = all(len(kept[g]) >= targets[g] for g in targets)
        run["completed"] = True
    finally:
        run["finished_utc"] = utc_now()
        run["primary"] = primary.stats.summary()
        run["fallback"] = fallback.stats.summary()
        runs_path = selection_dir / RUNS_FILE
        # A run served entirely from the cache made no requests, so it adds nothing to the record;
        # skipping it keeps an offline rebuild from changing the raw folder.
        if run["primary"]["requests"] or run["fallback"]["requests"]:
            runs_path.parent.mkdir(parents=True, exist_ok=True)
            with runs_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(run) + "\n")

    runs = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines()] if runs_path.exists() else []
    tx_counts = [len(h.flows) for h in histories]
    reason_counts = {r: sum(s["reason"] == r for s in skipped) for r in SKIP_REASONS}
    if not targets_met:
        status = f"incomplete: the pool ran out with {len(kept['main'])} main and {len(kept['short'])} short addresses"
    elif not all_checks:
        status = "complete, but a check failed (see checks)"
    else:
        status = "complete; every automated check passed"
    manifest.update({
        "status": status,
        "built_utc": utc_now(),
        "retrieval": {
            "pace_seconds": params["requestPauseSeconds"], "max_retries": params["maxRetries"],
            "timeout_seconds": params["timeoutSeconds"],
            "back_off": "exponential (1, 2, 4 ... s, at most 60 s) on HTTP 429 and 5xx, honouring Retry-After",
            "runs": runs, "totals": request_totals(runs),
            "note": "A run that reads only the cache makes no requests. Interrupted runs are listed too.",
        },
        "tip": {"height": tip_height, "file": f"{SELECTION_DIR}/{TIP_FILE}"},
        "blocks": blocks,
        "candidate_pool": {
            "size": len(pool),
            "by_script_type": {t: sum(v == t for v in pool.values()) for t in sorted(set(pool.values()))},
            "visited": visited, "not_visited": len(order) - visited,
        },
        "totals": {
            "main_addresses": len(kept["main"]), "short_addresses": len(kept["short"]),
            "transactions": sum(tx_counts),
            "median_tx_per_main_address": statistics.median(len(h.flows) for h, _ in kept["main"]) if kept["main"] else None,
            "span_days_main": [min((round(h.span_days, 1) for h, _ in kept["main"]), default=None),
                               max((round(h.span_days, 1) for h, _ in kept["main"]), default=None)],
        },
        "kept": [address_summary(h, g, r, raw_dir) for g in ("main", "short") for h, r in kept[g]],
        "skipped": {"reasons": SKIP_REASONS, "counts": reason_counts, "candidates": skipped},
        "checks": {"all_passed": all_checks, "per_address": per_address, "fallback_cross_check": crosses},
        "fields_used": FIELDS_USED,
        "missing_values": missing_values(histories),
        "dataset_file": write_histories(manifest_path.parent / HISTORIES_FILE, histories,
                                        {h.address: g for g in ("main", "short") for h, _ in kept[g]}),
        "provenance": {
            "root": "data/raw/esplora",
            "note": "Every API reply, unchanged and gzipped. Kept on the collecting machine as evidence (about "
                    "0.4 GB, too large for the repository or the zip); the SHA-256 of each file is listed here.",
            "files": provenance(raw_dir),
        },
    })
    write_manifest(manifest_path, manifest)

    log.info("Kept %d main and %d short addresses, %d transactions; visited %d of %d candidates",
             len(kept["main"]), len(kept["short"]), sum(tx_counts), visited, len(order))
    log.info("Skipped: %s", ", ".join(f"{r} {n}" for r, n in reason_counts.items() if n))
    log.info("Checks: %s; fallback cross-check: %s", "all passed" if all_checks else "FAILED",
             ", ".join(c["detail"] for c in crosses) or "not run")
    log.info("Requests this run: %s", run["primary"])
    log.info("Wrote %s", manifest_path)
    return 0 if targets_met and all_checks else 1
