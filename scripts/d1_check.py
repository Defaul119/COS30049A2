"""D1 data-source check (WBS 1.1.2, decision gate D1).

Answers the question every requirement depends on: can the Esplora API give,
for one real Bitcoin address, each confirmed transaction's time, the
address-related amount and its direction, through a reproducible method?

Usage (from the project root):
    python scripts/d1_check.py                      # pick an active address from a recent block
    python scripts/d1_check.py --address bc1q...    # check a specific address

Writes:
    data/d1/<address>/...           raw API responses, unchanged
    data/d1/address-samples.json    real addresses by type (used by the front-end validator tests)
    reports/d1-check.md             the filled D1 worksheet
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from explorer.amounts import AddressFlow, address_flow, format_btc  # noqa: E402
from explorer.esplora import PAGE_SIZE, EsploraClient, EsploraError  # noqa: E402
from explorer.settings import PROJECT_ROOT, SETTINGS  # noqa: E402

SRC = SETTINGS["source"]
COL = SETTINGS["collection"]

D1_MIN_TX = COL["mainMinTx"]  # same lower bound as the snapshot rule
D1_MAX_TX = 300  # keep the D1 fetch small: at most 12 history pages


def utc(ts: int | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def collect_candidates(client: EsploraClient, block_hash: str, samples: dict) -> list[str]:
    """Addresses seen in the first 50 transactions of a block (outputs and spent inputs)."""
    found: dict[str, str] = {}
    for start in (0, PAGE_SIZE):
        for tx in client.block_txs(block_hash, start):
            spks = [o for o in tx.get("vout", [])]
            spks += [v["prevout"] for v in tx.get("vin", []) if v.get("prevout")]
            for spk in spks:
                addr, kind = spk.get("scriptpubkey_address"), spk.get("scriptpubkey_type")
                if addr and addr not in found:
                    found[addr] = kind
                    bucket = samples.setdefault(kind, [])
                    if len(bucket) < 8:
                        bucket.append(addr)
    return list(found)


def pick_address(client: EsploraClient, seed: int, max_candidates: int, log: list[str], samples: dict):
    rng = random.Random(seed)
    block_hash = client.tip_hash()
    for _ in range(3):  # tip block, then up to two parents
        block = client.block(block_hash)
        candidates = collect_candidates(client, block_hash, samples)
        rng.shuffle(candidates)
        log.append(
            f"Block {block['height']} ({block_hash}): {len(candidates)} candidate addresses, "
            f"visited in seeded random order (seed {seed})."
        )
        checked = 0
        for addr in candidates[:max_candidates]:
            stats = client.address(addr)
            checked += 1
            n = stats["chain_stats"]["tx_count"]
            if D1_MIN_TX <= n <= D1_MAX_TX:
                log.append(f"Checked {checked} candidates; picked the first with {D1_MIN_TX}-{D1_MAX_TX} confirmed transactions ({n}).")
                return addr, block
        log.append(f"Checked {checked} candidates in block {block['height']}; none qualified, moving to the parent block.")
        block_hash = block["previousblockhash"]
    raise SystemExit("No qualifying address found in three blocks; rerun with --max-candidates higher.")


def fetch_full_history(client: EsploraClient, address: str):
    for _ in range(2):  # retry once if a new block lands mid-fetch
        before = client.address(address)
        txs, pages = client.address_history(address, cap=COL["maxTxPerAddress"])
        after = client.address(address)
        if before["chain_stats"] == after["chain_stats"]:
            return before, txs, pages
    return after, txs, pages


def run_checks(address: str, stats: dict, txs: list[dict], pages: list[list], flows: list[AddressFlow]):
    chain = stats["chain_stats"]
    heights = [f.block_height for f in flows]
    checks = [
        ("History is complete (count matches the API's chain_stats.tx_count)",
         len(txs) == chain["tx_count"], f"{len(txs)} fetched vs {chain['tx_count']} reported"),
        ("No duplicate transaction IDs",
         len({t['txid'] for t in txs}) == len(txs), f"{len({t['txid'] for t in txs})} unique of {len(txs)}"),
        ("Every transaction is confirmed and has a block time",
         all(f.confirmed and f.block_time for f in flows), "status.confirmed and status.block_time present"),
        ("History is ordered newest first",
         all(a >= b for a, b in zip(heights, heights[1:])), "block heights never increase down the list"),
        (f"Paging: every page except the last holds {PAGE_SIZE}",
         all(len(p) == PAGE_SIZE for p in pages[:-1]) and len(pages[-1]) <= PAGE_SIZE,
         f"{len(pages)} pages, sizes {[len(p) for p in pages]}"),
        ("Every transaction involves the address (received or sent > 0)",
         all(f.received > 0 or f.sent > 0 for f in flows), "derived from vout and vin.prevout"),
        ("Sum of received equals the API's funded_txo_sum",
         sum(f.received for f in flows) == chain["funded_txo_sum"],
         f"{format_btc(sum(f.received for f in flows))} vs {format_btc(chain['funded_txo_sum'])}"),
        ("Sum of sent equals the API's spent_txo_sum",
         sum(f.sent for f in flows) == chain["spent_txo_sum"],
         f"{format_btc(sum(f.sent for f in flows))} vs {format_btc(chain['spent_txo_sum'])}"),
    ]
    return checks


def cross_check(fallback: EsploraClient, address: str, flows: list[AddressFlow], out_dir: Path):
    try:
        page = fallback.address_txs_chain(address)
    except EsploraError as err:
        return None, f"Fallback request failed: {err}"
    write_json(out_dir / "mempool-page-001.json", page)
    ours = {f.txid: f for f in flows}
    theirs = {t["txid"]: address_flow(t, address) for t in page}
    shared = [txid for txid in theirs if txid in ours]
    mismatched = [
        txid for txid in shared
        if (ours[txid].block_time, ours[txid].received, ours[txid].sent)
        != (theirs[txid].block_time, theirs[txid].received, theirs[txid].sent)
    ]
    ok = bool(shared) and not mismatched
    return ok, f"{len(shared)} transactions on both sources' first page; {len(mismatched)} differ in time or amount"


def pick_samples(flows: list[AddressFlow]) -> list[AddressFlow]:
    """Five transactions spread across the history, with both directions if possible."""
    n = len(flows)
    idx = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    chosen = [flows[i] for i in idx]
    directions = {f.direction for f in chosen}
    for want in ("in", "out"):
        if want not in directions:
            extra = next((f for f in flows if f.direction == want), None)
            if extra:
                chosen[len(chosen) // 2] = extra
    return sorted(chosen, key=lambda f: f.block_time or 0)


def build_report(ctx: dict) -> str:
    flows: list[AddressFlow] = ctx["flows"]
    samples = ctx["samples"]
    checks = ctx["checks"]
    all_pass = all(ok for _, ok, _ in checks) and ctx["cross_ok"] is not False
    first, last = min(f.block_time for f in flows), max(f.block_time for f in flows)
    span_days = (last - first) / 86400
    tx_url = SRC["explorerTxUrl"]
    ins = sum(1 for f in flows if f.direction == "in")
    outs = sum(1 for f in flows if f.direction == "out")

    lines = [
        "# D1 data-source check",
        "",
        f"**Run:** {ctx['run_at']} · **Command:** `{ctx['command']}`",
        f"**Primary source:** {SRC['name']} ({SRC['apiBase']}) · **Fallback:** {SRC['fallbackName']} ({SRC['fallbackApiBase']})",
        "",
        f"## Verdict: {'PASS' if all_pass else 'NEEDS ATTENTION'}",
        "",
        "| Requirement (gate D1) | Result |",
        "|---|---|",
        "| Transaction ID per transaction | Yes: `txid` |",
        "| Confirmed time | Yes: `status.block_time` (unix seconds, shown in UTC) |",
        "| Address-related amount | Yes: derived from `vout[].value` and `vin[].prevout.value` |",
        "| Direction | Yes: sign of net = received − sent |",
        f"| Reproducible retrieval | Yes: `/address/<address>/txs/chain[/<last_seen_txid>]`, {PAGE_SIZE} per page, newest first |",
        "| Source link per transaction | Yes: " + tx_url.replace("{txid}", "&lt;txid&gt;") + " |",
        "",
        "## Address inspected",
        "",
        f"- Address: `{ctx['address']}` ({ctx['address_type']})",
        f"- How it was chosen: {' '.join(ctx['selection_log'])}",
        f"- Confirmed transactions: {len(flows)} ({ins} in, {outs} out)",
        f"- First and last confirmed: {utc(first)} → {utc(last)} ({span_days:.0f} days)",
        f"- Total received {format_btc(sum(f.received for f in flows))}; total sent {format_btc(sum(f.sent for f in flows))}",
        "- The address was picked at random from a recent block. Nothing here says who controls it or why it transacts.",
        "",
        "## Five transactions with derived amount and direction",
        "",
        "| # | Transaction ID | Confirmed (UTC) | Block | Received | Sent | Net | Direction |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, f in enumerate(samples, 1):
        link = tx_url.replace("{txid}", f.txid)
        lines.append(
            f"| {i} | [`{f.txid[:12]}…{f.txid[-6:]}`]({link}) | {utc(f.block_time)} | {f.block_height} | "
            f"{format_btc(f.received)} | {format_btc(f.sent)} | {format_btc(f.net, signed=True)} | {f.direction} |"
        )
    lines += [
        "",
        "Full transaction IDs:",
        "",
        *[f"{i}. `{f.txid}`" for i, f in enumerate(samples, 1)],
        "",
        "**Hand check (a team member, not the script):** open transaction 1 on the block explorer, find the",
        "address in the inputs and outputs, and confirm the received and sent values above. Record who checked",
        "and when: ____________________",
        "",
        "## Automated checks",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
        *[f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |" for name, ok, detail in checks],
        f"| Fallback source returns the same times and amounts | "
        f"{'PASS' if ctx['cross_ok'] else ('FAIL' if ctx['cross_ok'] is False else 'NOT RUN')} | {ctx['cross_detail']} |",
        "",
        "The two sum checks are the strongest evidence: adding up the derived amounts over the whole",
        "history reproduces the totals the API computes independently, so the received/sent rule is right",
        "for this address (Risk 18).",
        "",
        "## Paging, limits and errors",
        "",
        f"- History endpoint returns {PAGE_SIZE} confirmed transactions per page, newest first. The next page is",
        "  requested with the last transaction ID seen, e.g. `/address/<address>/txs/chain/<last_txid>`.",
        f"- History pages fetched: {len(ctx['pages'])}. A page shorter than {PAGE_SIZE} marks the end.",
        f"- Requests to {SRC['name']}: {ctx['primary_stats']['requests']} "
        f"(HTTP 429: {ctx['primary_stats']['http_429']}, 5xx: {ctx['primary_stats']['http_5xx']}, "
        f"retries: {ctx['primary_stats']['retries']}, median latency {ctx['primary_stats']['median_latency_ms']} ms).",
        f"- Requests to the fallback: {ctx['fallback_stats']['requests']} (HTTP 429: {ctx['fallback_stats']['http_429']}).",
        f"- Pace: one request every {COL['requestPauseSeconds']} s, exponential back-off on 429/5xx, honouring `Retry-After`.",
        "- Esplora caps how much history it will serve for very busy addresses; the snapshot rule already",
        f"  excludes addresses above {COL['mainMaxTx']} transactions, and the client caps at {COL['maxTxPerAddress']}.",
        "",
        "## Fields used",
        "",
        "`txid`, `status.confirmed`, `status.block_height`, `status.block_time`, `vout[].scriptpubkey_address`,",
        "`vout[].value`, `vin[].prevout.scriptpubkey_address`, `vin[].prevout.value`; and from `/address/<address>`:",
        "`chain_stats.tx_count`, `funded_txo_sum`, `spent_txo_sum` (used only for the checks above).",
        "",
        "## Access and use terms",
        "",
        f"- API documentation: {SRC['apiDocs']}. It states no rate limits or terms.",
        "- Blockstream's general Terms of Use (v1.0, 16 Jan 2018, https://blockstream.com/terms/) cover its",
        "  website and services but do not mention the Explorer or its API.",
        "- Third-party comparisons report the free public API allows roughly 50 requests per second. This project",
        "  sends at most 2 per second and calls the API only offline, once, to build the snapshot.",
        "- **Team action:** record in the manifest that no explicit API terms were found, and confirm with the",
        "  tutor that using public blockchain data this way is acceptable.",
        "",
        "## Raw evidence",
        "",
        f"Unchanged API responses are in `data/d1/{ctx['address']}/`. Rerun with "
        f"`python scripts/d1_check.py --address {ctx['address']}` to reproduce (new blocks may add transactions).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--address", help="check this address instead of picking one")
    parser.add_argument("--seed", type=int, default=COL["selectionSeed"])
    parser.add_argument("--max-candidates", type=int, default=150)
    args = parser.parse_args()

    primary = EsploraClient(SRC["apiBase"], COL["requestPauseSeconds"], COL["timeoutSeconds"], COL["maxRetries"])
    fallback = EsploraClient(SRC["fallbackApiBase"], COL["requestPauseSeconds"], COL["timeoutSeconds"], COL["maxRetries"])

    print(f"Tip height: {primary.tip_height()} ({SRC['name']}), {fallback.tip_height()} (fallback)")
    samples: dict[str, list[str]] = {}
    selection_log: list[str] = []
    if args.address:
        address = args.address
        selection_log.append("Given on the command line.")
    else:
        address, _ = pick_address(primary, args.seed, args.max_candidates, selection_log, samples)
    print(f"Address: {address}")

    stats, txs, pages = fetch_full_history(primary, address)
    print(f"Fetched {len(txs)} transactions in {len(pages)} pages")
    out_dir = PROJECT_ROOT / "data" / "d1" / address
    write_json(out_dir / "address-stats.json", stats)
    for i, page in enumerate(pages, 1):
        write_json(out_dir / f"blockstream-page-{i:03d}.json", page)

    flows = [address_flow(tx, address) for tx in txs]
    address_type = next(
        (o["scriptpubkey_type"] for tx in txs for o in tx["vout"] if o.get("scriptpubkey_address") == address),
        next((v["prevout"]["scriptpubkey_type"] for tx in txs for v in tx["vin"]
              if v.get("prevout") and v["prevout"].get("scriptpubkey_address") == address), "unknown"),
    )
    checks = run_checks(address, stats, txs, pages, flows)
    cross_ok, cross_detail = cross_check(fallback, address, flows, out_dir)

    if samples:
        write_json(PROJECT_ROOT / "data" / "d1" / "address-samples.json",
                   {"note": "Real mainnet addresses seen in a recent block, grouped by script type. Used only as validator test vectors.",
                    "collected": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "byType": samples})

    report = build_report({
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "command": "python scripts/d1_check.py" + (f" --address {args.address}" if args.address else ""),
        "address": address,
        "address_type": address_type,
        "selection_log": selection_log,
        "flows": flows,
        "samples": pick_samples(flows),
        "pages": pages,
        "checks": checks,
        "cross_ok": cross_ok,
        "cross_detail": cross_detail,
        "primary_stats": primary.stats.summary(),
        "fallback_stats": fallback.stats.summary(),
    })
    report_path = PROJECT_ROOT / "reports" / "d1-check.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"[{'PASS' if cross_ok else 'FAIL'}] Fallback agreement: {cross_detail}")
    print(f"Worksheet written to {report_path.relative_to(PROJECT_ROOT)}")
    return 0 if all(ok for _, ok, _ in checks) and cross_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
