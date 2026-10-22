"""DS2: crawl about 5,000 address histories for TxGPT, kept as compact rows with hashed ids.

Serves plan §6.2 (docs/plan/sections/s05-data.tex): TxGPT needs far more sequences than DS1's 28
addresses. Also A1 risk 16 (privacy: addresses and txids are stored only as salted hashes, and the
salt in config/salt.txt never leaves this machine) and the leakage rule that no DS1 address
enters DS2, so TxGPT never trains on an address it later scores. DS2 excludes DS1's whole
candidate pool, a superset of DS1, so the crawl need not wait for the DS1 fetch to finish.

Discovery uses DS1's block sampling over 200 blocks spread across the last 12 months, for seasonal
variety. Candidates are shuffled with the seed and visited by worker threads on two
Esplora-compatible hosts (Blockstream and mempool.space). Each host gets at most one request start
per `requestPauseSeconds`, however many workers share it, so the crawl is as polite as DS1's
single client while finishing several times faster.

The crawl resumes after any interruption: every visited candidate is logged, and every kept
history is saved as compact rows before it is logged. Run the same command again to continue.

Writes:
    data/raw/corpus/blocks/<height>/      the sampled blocks' pages (git-ignored; repeatable discovery)
    data/interim/corpus/visits.jsonl      one line per visited candidate (raw addresses; git-ignored)
    data/interim/corpus/rows.jsonl        each kept address's compact rows (git-ignored)
    data/processed/corpus/ds2.parquet     the DS2 table, one row per (address, transaction)
    data/processed/corpus/manifest.json   rule, counts, checks and request statistics (no addresses)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import queue
import random
import secrets
import statistics
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .amounts import address_flow
from .esplora import EsploraClient, EsploraError
from .settings import PROJECT_ROOT, SETTINGS
from .snapshot import (
    MANIFEST_PATH as DS1_MANIFEST_PATH, RAW_DIR as DS1_RAW_DIR, SELECTION_DIR, History, add_output_addresses,
    block_heights, cached_get, pool_from_blocks, read_block, read_gz, utc_iso, utc_now, verify_history,
)

log = logging.getLogger(__name__)

SALT_PATH = PROJECT_ROOT / "config" / "salt.txt"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "corpus"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "corpus"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "corpus"

COLUMNS = [
    "addr_id", "txid_hash", "block_height", "block_time_utc", "received_sats", "sent_sats",
    "n_inputs", "n_outputs", "fee_sats", "vsize_vb",
]
COLUMN_NOTES = {
    "addr_id": "first hex characters of SHA-256(salt + address); the salt is not shipped",
    "txid_hash": "first hex characters of SHA-256(salt + txid); used only to remove duplicates",
    "block_height": "confirmation height",
    "block_time_utc": "block time, Unix seconds (UTC)",
    "received_sats": "outputs paying the address, satoshis (explorer/amounts.py)",
    "sent_sats": "inputs spent from the address, satoshis (explorer/amounts.py)",
    "n_inputs": "inputs in the transaction",
    "n_outputs": "outputs in the transaction",
    "fee_sats": "transaction fee, satoshis",
    "vsize_vb": "virtual size, vbytes = ceil(weight / 4)",
}
WEIGHT_UNITS_PER_VBYTE = 4

SKIP_REASONS = {
    "too_few": "fewer confirmed transactions than the minimum",
    "too_many": "more confirmed transactions than the maximum (busy services; a stated bias)",
    "checks_failed": "fetched, but an automated check failed",
    "fetch_error": "the API still returned an error after every retry",
}


# -- privacy -------------------------------------------------------------------------


def load_or_create_salt(path: Path = SALT_PATH) -> str:
    """The project salt for DS2's hashes, created once (256 random bits) if it does not exist yet."""
    if not path.exists():
        path.write_text(secrets.token_hex(32) + "\n", encoding="utf-8")
        log.info("Created a new salt at %s (git-ignored; never share or zip it)", path)
    return path.read_text(encoding="utf-8").strip()


def salted_id(salt: str, value: str, hex_chars: int) -> str:
    """A short, salted SHA-256 id: stable for this project, not reversible without the salt."""
    return hashlib.sha256((salt + value).encode("utf-8")).hexdigest()[:hex_chars]


def ds1_exclusions(ds1_raw_dir: Path = DS1_RAW_DIR, manifest_path: Path = DS1_MANIFEST_PATH) -> set[str]:
    """Every address in DS1's candidate pool, plus the DS1 addresses themselves: what DS2 excludes.

    The pool (the outputs of DS1's sampled blocks) is fixed before DS1 keeps anything and contains
    every address DS1 can keep, so excluding all of it keeps DS1 out of DS2 even while the DS1
    fetch is still running. Raises if DS1's blocks have not all been read yet.
    """
    col = SETTINGS["collection"]
    blocks = sorted((ds1_raw_dir / SELECTION_DIR / "blocks").glob("*/block.json.gz"))
    if len(blocks) < col["snapshotBlockCount"]:
        raise SystemExit("DS1's sampled blocks must be read before the DS2 crawl (run: python -m explorer.cli snapshot)")
    pool: dict[str, str] = {}
    for page in sorted((ds1_raw_dir / SELECTION_DIR / "blocks").glob("*/txs-*.json.gz")):
        add_output_addresses(json.loads(read_gz(page)), pool)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return set(pool) | {row["address"] for row in manifest.get("kept", [])}


# -- concurrency ---------------------------------------------------------------------


class SharedPace:
    """Spaces request starts at least `interval` seconds apart across every thread using one host."""

    def __init__(self, interval: float):
        self.interval = interval
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.perf_counter()  # monotonic too, but finer than time.monotonic's ~16 ms ticks on Windows
            start = max(now, self._next_start)
            self._next_start = start + self.interval
        if start > now:
            time.sleep(start - now)


class ClientPool:
    """Lends each worker thread one client at a time, so no client is used by two threads at once."""

    def __init__(self, clients: list[EsploraClient]):
        self.clients = clients
        self._free: queue.Queue[EsploraClient] = queue.Queue()
        for client in clients:
            self._free.put(client)

    @contextmanager
    def client(self) -> Iterator[EsploraClient]:
        client = self._free.get()
        try:
            yield client
        finally:
            self._free.put(client)

    def stats_by_host(self) -> dict[str, dict[str, Any]]:
        """Request statistics summed over the clients of each host."""
        out: dict[str, dict[str, Any]] = {}
        for client in self.clients:
            s = client.stats.summary()
            host = out.setdefault(client.base_url, {"requests": 0, "retries": 0, "http_429": 0, "http_5xx": 0,
                                                    "network_errors": 0, "latencies": []})
            for key in ("requests", "retries", "http_429", "http_5xx", "network_errors"):
                host[key] += s[key]
            host["latencies"] += client.stats.latencies_ms
        for host in out.values():
            lat = host.pop("latencies")
            host["median_latency_ms"] = round(statistics.median(lat)) if lat else None
        return out


def make_clients(sources: list[str], workers_per_source: int) -> list[EsploraClient]:
    """`workers_per_source` clients per host, sharing that host's pace (settings.json -> collection)."""
    col = SETTINGS["collection"]
    clients = []
    for base in sources:
        pace = SharedPace(col["requestPauseSeconds"])
        clients += [EsploraClient(base, 0.0, col["timeoutSeconds"], col["maxRetries"], pacer=pace.wait)
                    for _ in range(workers_per_source)]
    return clients


# -- one candidate ---------------------------------------------------------------------


def compact_rows(history: History, salt: str, hex_chars: int) -> list[list[Any]]:
    """One compact row per transaction (in COLUMNS order), newest first as fetched."""
    addr_id = salted_id(salt, history.address, hex_chars)
    rows = []
    for tx, flow in zip(history.txs, history.flows):
        rows.append([
            addr_id, salted_id(salt, tx["txid"], hex_chars), flow.block_height, flow.block_time,
            flow.received, flow.sent, tx["n_vin"], tx["n_vout"], tx.get("fee"),
            math.ceil(tx["weight"] / WEIGHT_UNITS_PER_VBYTE) if tx.get("weight") is not None else None,
        ])
    return rows


def crawl_candidate(
    clients: ClientPool, address: str, params: dict[str, Any], attempts: int, salt: str
) -> dict[str, Any]:
    """Visit one candidate: read its count, and if in range fetch, check and compact its history.

    chain_stats is read before and after the history; if a block lands in between, the history is
    fetched again (up to `attempts` times) so the rows match the counts they are checked against.
    Never raises for API errors; they become a `fetch_error` visit.
    """
    with clients.client() as client:
        visit: dict[str, Any] = {"address": address, "source": client.base_url, "tx_count": None}
        try:
            stats = client.get_json(f"/address/{address}")
            n = stats["chain_stats"]["tx_count"]
            visit["tx_count"] = n
            if n < params["minTx"]:
                return {**visit, "reason": "too_few"}
            if n > params["maxTx"]:
                return {**visit, "reason": "too_many"}
            for _ in range(attempts):
                _, pages = client.address_history(address, cap=params["maxTx"])
                after = client.get_json(f"/address/{address}")
                if after["chain_stats"] == stats["chain_stats"]:
                    break
                stats = after
        except (EsploraError, ValueError, KeyError) as err:  # ValueError covers a reply that is not JSON
            return {**visit, "reason": "fetch_error", "detail": f"{type(err).__name__}: {err}"}
    history = History.from_pages(address, Path(), after, pages)
    failed = [c["name"] for c in verify_history(history) if not c["passed"]]
    if failed:
        return {**visit, "tx_count": len(history.flows), "reason": "checks_failed", "detail": "; ".join(failed)}
    return {**visit, "tx_count": len(history.flows), "reason": "kept",
            "rows": compact_rows(history, salt, params["idHexChars"])}


# -- the crawl -------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every complete line of a JSONL log; a line cut short by an interruption is dropped."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def discover(clients: ClientPool, params: dict[str, Any], raw_dir: Path) -> tuple[int, list[dict[str, Any]], dict[str, str]]:
    """Tip height, sampled-block summaries and the candidate pool. Blocks are read in parallel and
    merged in height order, so the pool does not depend on which worker finished first."""
    with clients.client() as client:
        tip = int(cached_get(client, "/blocks/tip/height", raw_dir / "tip-height.txt.gz"))
    heights = block_heights(tip, params["blockCount"], params["oldestBlockBack"], params["newestBlockBack"])

    def read(height: int) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
        with clients.client() as client:
            return read_block(client, height, params["txPagesPerBlock"], raw_dir / "blocks")

    with ThreadPoolExecutor(len(clients.clients)) as ex:
        blocks = list(ex.map(read, heights))
    summaries, pool = pool_from_blocks(blocks)
    return tip, summaries, pool


def run_corpus(
    clients: ClientPool,
    params: dict[str, Any],
    raw_dir: Path = RAW_DIR,
    interim_dir: Path = INTERIM_DIR,
    out_dir: Path = OUT_DIR,
    ds1: set[str] | None = None,
    salt: str | None = None,
) -> int:
    """Discover, crawl until the target is reached (or the pool runs out), then write DS2 and its
    manifest. Returns 0 if at least `minimumAddresses` were kept and every check passed, else 1."""
    ds1 = ds1_exclusions() if ds1 is None else ds1
    salt = load_or_create_salt() if salt is None else salt
    attempts = SETTINGS["collection"]["historyFetchAttempts"]
    started = utc_now()
    visits_path, rows_path = interim_dir / "visits.jsonl", interim_dir / "rows.jsonl"
    visits = {v["address"]: v for v in read_jsonl(visits_path)}
    kept = sum(v["reason"] == "kept" for v in visits.values())
    in_flight: set[Future] = set()
    max_in_flight = 2 * len(clients.clients)  # keeps every worker busy without queueing far ahead
    ex = ThreadPoolExecutor(len(clients.clients))
    try:
        # Inside the try, so a discovery that fails part-way still leaves its request record.
        tip, blocks, pool = discover(clients, params, raw_dir)
        excluded = sorted(a for a in pool if a in ds1)
        order = sorted(a for a in pool if a not in ds1)
        random.Random(params["seed"]).shuffle(order)
        log.info("Pool %d candidates after removing %d addresses from DS1's candidate pool", len(order), len(excluded))
        pending = iter([a for a in order if a not in visits])
        log.info("Resuming with %d visited, %d kept; target %d", len(visits), kept, params["targetAddresses"])
        exhausted = False
        while True:
            while not exhausted and kept < params["targetAddresses"] and len(in_flight) < max_in_flight:
                address = next(pending, None)
                if address is None:
                    exhausted = True
                else:
                    in_flight.add(ex.submit(crawl_candidate, clients, address, params, attempts, salt))
            if not in_flight:
                break
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                visit = future.result()
                rows = visit.pop("rows", None)
                if rows is not None:
                    # Rows first, then the visit line: a crash between the two only re-crawls this address.
                    append_jsonl(rows_path, {"addr_id": rows[0][0], "source": visit["source"], "rows": rows})
                    kept += 1
                append_jsonl(visits_path, visit)
                visits[visit["address"]] = visit
                if len(visits) % params["progressEvery"] == 0:
                    log.info("Visited %d candidates, kept %d (%s)", len(visits), kept,
                             ", ".join(f"{h}: {s['requests']} requests, {s['http_429']} x 429"
                                       for h, s in clients.stats_by_host().items()))
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        run = {"started_utc": started, "finished_utc": utc_now(), "hosts": clients.stats_by_host()}
        append_jsonl(interim_dir / "runs.jsonl", run)

    return write_outputs(visits, rows_path, interim_dir, out_dir, params, salt, ds1, tip, blocks, pool, excluded, order)


def write_outputs(
    visits: dict[str, dict[str, Any]], rows_path: Path, interim_dir: Path, out_dir: Path, params: dict[str, Any],
    salt: str, ds1: set[str], tip: int, blocks: list[dict[str, Any]], pool: dict[str, str], excluded: list[str],
    order: list[str],
) -> int:
    """Assemble ds2.parquet from the saved rows, run the DS2 checks, and write the manifest."""
    import pandas as pd  # the crawl itself needs only the standard library

    kept_visits = {v["address"]: v for v in visits.values() if v["reason"] == "kept"}
    kept_ids = {salted_id(salt, a, params["idHexChars"]): v for a, v in kept_visits.items()}
    by_id = {r["addr_id"]: r for r in read_jsonl(rows_path) if r["addr_id"] in kept_ids}
    frame = pd.DataFrame([row for r in by_id.values() for row in r["rows"]], columns=COLUMNS)
    frame = frame.sort_values(["addr_id", "block_time_utc", "txid_hash"], kind="stable").reset_index(drop=True)
    missing = int(frame.isna().sum().sum())
    if not missing:  # integers stay integers in the parquet; a gap would otherwise force floats
        frame[COLUMNS[2:]] = frame[COLUMNS[2:]].astype("int64")

    counts = frame.groupby("addr_id").size()
    ds1_ids = {salted_id(salt, a, params["idHexChars"]) for a in ds1}
    checks = [
        {"name": "every kept address has rows", "passed": set(counts.index) == set(kept_ids),
         "detail": f"{len(counts)} addresses with rows, {len(kept_ids)} kept"},
        {"name": "row count equals tx_count for every address",
         "passed": all(counts.get(i, 0) == v["tx_count"] for i, v in kept_ids.items()),
         "detail": "rows per addr_id against the visit's tx_count"},
        {"name": "no duplicate txid_hash within an address",
         "passed": not frame.duplicated(["addr_id", "txid_hash"]).any(), "detail": "(addr_id, txid_hash) pairs"},
        {"name": "no DS1 address in DS2", "passed": not (ds1_ids & set(counts.index)),
         "detail": f"{len(excluded)} pool addresses were in DS1's candidate pool and removed before the crawl"},
        {"name": "no missing values", "passed": not missing, "detail": f"{missing} missing cells"},
    ]
    all_passed = all(c["passed"] for c in checks)

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "ds2.parquet"
    frame.to_parquet(parquet_path, index=False)
    reasons = {r: sum(v["reason"] == r for v in visits.values()) for r in ["kept", *SKIP_REASONS]}
    runs = read_jsonl(interim_dir / "runs.jsonl")
    hosts: dict[str, dict[str, int]] = {}
    for run in runs:
        for host, s in run["hosts"].items():
            h = hosts.setdefault(host, {"requests": 0, "retries": 0, "http_429": 0, "http_5xx": 0, "network_errors": 0})
            for key in h:
                h[key] += s[key]
    enough = len(counts) >= params["minimumAddresses"]
    manifest = {
        "dataset": "DS2: Esplora crawl of address histories for TxGPT, compact rows with hashed ids",
        "status": ("complete; every check passed" if all_passed and enough
                   else f"needs attention: {len(counts)} addresses kept, checks {'passed' if all_passed else 'FAILED'}"),
        "built_utc": utc_now(),
        "rule": {
            "steps": [
                f"Read the tip height. Take {params['blockCount']} block heights spread evenly from tip - "
                f"{params['oldestBlockBack']} to tip - {params['newestBlockBack']} (about the last 12 months).",
                f"From each block, take the output addresses of up to {params['txPagesPerBlock']} pages of "
                "transactions; remove duplicates; remove every address in DS1's candidate pool (a superset of the DS1 "
                "addresses, so DS1 can never leak into TxGPT's training data).",
                f"Sort the pool, shuffle it with seed {params['seed']}, and visit candidates in that order.",
                f"Keep a candidate with {params['minTx']}-{params['maxTx']} confirmed transactions whose history "
                f"passes every check. Stop once {params['targetAddresses']} are kept (the minimum is "
                f"{params['minimumAddresses']}).",
                "Worker threads finish in any order, so the last few kept addresses can differ between runs; "
                "the crawl may overshoot the target by up to the number of candidates in flight.",
            ],
            "parameters": {k: params[k] for k in ("seed", "blockCount", "oldestBlockBack", "newestBlockBack",
                                                  "txPagesPerBlock", "minTx", "maxTx", "targetAddresses",
                                                  "minimumAddresses")},
        },
        "privacy": "Addresses and txids appear only as salted SHA-256 prefixes. The salt (config/salt.txt) is "
                   "not in the repository or the zip, so the ids cannot be linked back to addresses.",
        "sources": {"hosts": list(hosts), "requests": hosts, "runs": len(runs),
                    "pace": f"at most one request start every {SETTINGS['collection']['requestPauseSeconds']} s "
                            f"per host, {params['workersPerSource']} workers per host"},
        "tip": tip,
        "blocks": {"count": len(blocks), "heights": [blocks[0]["height"], blocks[-1]["height"]],
                   "time_utc": [blocks[0]["time_utc"], blocks[-1]["time_utc"]]},
        "candidate_pool": {"size": len(pool), "ds1_removed": len(excluded), "visited": len(visits),
                           "not_visited": len(order) - len(visits),
                           "by_script_type": {t: sum(v == t for v in pool.values()) for t in sorted(set(pool.values()))}},
        "visits": {"reasons": {"kept": "kept", **SKIP_REASONS}, "counts": reasons},
        "totals": {
            "addresses": int(len(counts)), "rows": int(len(frame)),
            "tx_per_address": {"min": int(counts.min()), "median": float(counts.median()), "max": int(counts.max())}
            if len(counts) else None,
            "time_utc": [utc_iso(int(frame["block_time_utc"].min())), utc_iso(int(frame["block_time_utc"].max()))]
            if len(frame) else None,
        },
        "columns": COLUMN_NOTES,
        "checks": {"all_passed": all_passed, "list": checks},
        "files": [{"path": "data/processed/corpus/ds2.parquet", "bytes": parquet_path.stat().st_size,
                   "sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest()}],
        "rebuild": ["python -m explorer.cli corpus   # resumes; a fresh crawl needs data/interim/corpus/ removed "
                    "and gives different addresses"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("DS2: %d addresses, %d rows; visits %s", len(counts), len(frame), reasons)
    for check in checks:
        log.info("[%s] %s: %s", "PASS" if check["passed"] else "FAIL", check["name"], check["detail"])
    log.info("Wrote %s and %s", parquet_path, out_dir / "manifest.json")
    return 0 if all_passed and enough else 1
