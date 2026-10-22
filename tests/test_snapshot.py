"""Tests for the DS1 snapshot (plan §6.1): the selection rule, the checks and the cache.

The end-to-end tests run against a small fake Esplora API, so they need no network.
Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from explorer import snapshot
from explorer.esplora import EsploraError, RequestStats
from explorer.settings import SETTINGS

DAY = 86_400
T0 = 1_780_000_000  # unix seconds; any fixed time works
OTHER = "bc1qsomeoneelse"


class FakeClient:
    """Answers get_text from a dict of API path -> reply; anything else is a 404."""

    def __init__(self, replies: dict[str, Any], base_url: str = "https://fake.example/api"):
        self.replies = replies
        self.base_url = base_url
        self.stats = RequestStats()

    def get_text(self, path: str) -> str:
        self.stats.requests += 1
        if path not in self.replies:
            raise EsploraError(404, self.base_url + path, "not found")
        reply = self.replies[path]
        return reply if isinstance(reply, str) else json.dumps(reply)


def tx(txid: str, address: str, received: int, sent: int, day: int) -> dict[str, Any]:
    vin = [{"prevout": {"scriptpubkey_address": address if sent else OTHER, "value": sent or 5_000}}]
    vout = [{"scriptpubkey_address": address, "scriptpubkey_type": "v0_p2wpkh", "value": received}] if received else []
    vout.append({"scriptpubkey_address": OTHER, "scriptpubkey_type": "v0_p2wpkh", "value": 1_000})
    status = {"confirmed": True, "block_height": 50 + day, "block_time": T0 + day * DAY}
    return {"txid": txid, "vin": vin, "vout": vout, "fee": 200, "weight": 560, "status": status}


def history(address: str, days: list[int]) -> list[dict[str, Any]]:
    """Alternating receive/send transactions on the given days, newest first as Esplora returns them."""
    txs = [tx(f"{address}-{i}", address, 10_000 if i % 2 == 0 else 0, 4_000 if i % 2 else 0, d)
           for i, d in enumerate(days)]
    return sorted(txs, key=lambda t: -t["status"]["block_time"])


def address_reply(address: str, txs: list[dict[str, Any]], tx_count: int | None = None) -> dict[str, Any]:
    funded = sum(o["value"] for t in txs for o in t["vout"] if o.get("scriptpubkey_address") == address)
    spent = sum(v["prevout"]["value"] for t in txs for v in t["vin"] if v["prevout"]["scriptpubkey_address"] == address)
    count = len(txs) if tx_count is None else tx_count
    return {"address": address, "chain_stats": {"tx_count": count, "funded_txo_sum": funded, "spent_txo_sum": spent}}


def params(**overrides: Any) -> dict[str, Any]:
    p = dict(SETTINGS["collection"])
    p.update(snapshotBlockCount=2, snapshotOldestBlockBack=10, snapshotNewestBlockBack=2,
             mainMinTx=3, mainMaxTx=5, mainMinSpanDays=10, targetMainAddresses=1,
             shortMinTx=1, shortMaxTx=1, targetShortAddresses=1, maxTxPerAddress=5, crossCheckAddresses=1)
    p.update(overrides)
    return p


def fake_world() -> tuple[dict[str, Any], dict[str, Any]]:
    """A tip at 100, two sampled blocks (90 and 98), and one candidate of each kind."""
    histories = {
        "main_ok": history("main_ok", [0, 5, 12, 30]),       # 4 tx over 30 days: kept
        "main_brief": history("main_brief", [0, 1, 2, 3]),   # 4 tx over 3 days: span too short
        "short_ok": history("short_ok", [7]),                # 1 tx: short-history
        "middle": history("middle", [1, 2]),                 # 2 tx: between the ranges
    }
    stats = {a: address_reply(a, h) for a, h in histories.items()}
    stats["busy"] = {"address": "busy", "chain_stats": {"tx_count": 900, "funded_txo_sum": 0, "spent_txo_sum": 0}}
    stats[OTHER] = {"address": OTHER, "chain_stats": {"tx_count": 0, "funded_txo_sum": 0, "spent_txo_sum": 0}}

    def block_tx(addresses: list[str]) -> dict[str, Any]:
        return {"vout": [{"scriptpubkey_address": a, "scriptpubkey_type": "v0_p2wpkh", "value": 1} for a in addresses]}

    primary: dict[str, Any] = {
        "/blocks/tip/height": "100",
        "/block-height/90": "hash90", "/block-height/98": "hash98",
        "/block/hash90": {"timestamp": T0, "tx_count": 2},
        "/block/hash98": {"timestamp": T0 + DAY, "tx_count": 2},
        "/block/hash90/txs/0": [block_tx(["main_ok", "busy"]), block_tx(["short_ok", OTHER])],
        "/block/hash98/txs/0": [block_tx(["main_brief", "middle"]), block_tx(["main_ok"])],
    }
    for address, reply in stats.items():
        primary[f"/address/{address}"] = reply
    for address, txs in histories.items():
        primary[f"/address/{address}/txs/chain"] = txs
    fallback = {"/address/main_ok/txs/chain": histories["main_ok"]}
    return primary, fallback


class PureFunctionTests(unittest.TestCase):
    def test_block_heights_spread_from_oldest_to_newest(self):
        heights = snapshot.block_heights(900_000, SETTINGS["collection"])
        col = SETTINGS["collection"]
        self.assertEqual(len(heights), col["snapshotBlockCount"])
        self.assertEqual(heights[0], 900_000 - col["snapshotOldestBlockBack"])
        self.assertEqual(heights[-1], 900_000 - col["snapshotNewestBlockBack"])
        gaps = {b - a for a, b in zip(heights, heights[1:])}
        self.assertLessEqual(max(gaps) - min(gaps), 1)  # even up to rounding

    def test_count_class_covers_every_range(self):
        p = SETTINGS["collection"]
        cases = {0: "too_few", p["shortMinTx"]: "short", p["shortMaxTx"]: "short",
                 p["shortMaxTx"] + 1: "between_ranges", p["mainMinTx"]: "main", p["mainMaxTx"]: "main",
                 p["mainMaxTx"] + 1: "too_many"}
        for count, expected in cases.items():
            self.assertEqual(snapshot.count_class(count, p), expected, count)

    def test_output_addresses_are_deduplicated_with_first_script_type(self):
        pool: dict[str, str] = {}
        txs = [{"vout": [{"scriptpubkey_address": "a", "scriptpubkey_type": "p2tr"},
                         {"scriptpubkey_type": "op_return"}]},
               {"vout": [{"scriptpubkey_address": "a", "scriptpubkey_type": "other"},
                         {"scriptpubkey_address": "b", "scriptpubkey_type": "p2pkh"}]}]
        self.assertEqual(snapshot.add_output_addresses(txs, pool), 2)
        self.assertEqual(pool, {"a": "p2tr", "b": "p2pkh"})

    def test_gzip_bytes_are_identical_on_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json.gz"
            snapshot.write_gz(path, '{"a": 1}')
            first = path.read_bytes()
            snapshot.write_gz(path, '{"a": 1}')
            self.assertEqual(path.read_bytes(), first)
            self.assertEqual(snapshot.read_gz(path), '{"a": 1}')

    def test_checks_catch_a_sum_mismatch(self):
        txs = history("addr", [0, 20, 40])
        stats = address_reply("addr", txs)
        stats["chain_stats"]["funded_txo_sum"] += 1
        h = snapshot.History("addr", Path("."), stats, [txs])
        failed = [c["name"] for c in snapshot.verify_history(h) if not c["passed"]]
        self.assertEqual(failed, ["summed received equals funded_txo_sum"])


class SnapshotRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self.tmp.name) / "raw"
        self.manifest_path = Path(self.tmp.name) / "manifest.json"

    def tearDown(self):
        self.tmp.cleanup()

    def run_once(self, primary: FakeClient, fallback: FakeClient, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        code = snapshot.run_snapshot(primary, fallback, params(), self.raw, self.manifest_path, **kwargs)
        return code, json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def test_rule_only_writes_the_rule_and_makes_no_request(self):
        primary, fallback = FakeClient({}), FakeClient({})
        code, manifest = self.run_once(primary, fallback, rule_only=True)
        self.assertEqual(code, 0)
        self.assertEqual(primary.stats.requests + fallback.stats.requests, 0)
        self.assertEqual(manifest["selection_rule"]["seed"], params()["selectionSeed"])
        self.assertIn("nothing fetched", manifest["status"])

    def test_selects_by_the_rule_and_logs_every_skip(self):
        replies, fallback_replies = fake_world()
        code, manifest = self.run_once(FakeClient(replies), FakeClient(fallback_replies))
        self.assertEqual(code, 0, manifest["status"])
        kept = {k["address"]: k["role"] for k in manifest["kept"]}
        self.assertEqual(kept, {"main_ok": "main", "short_ok": "short"})
        allowed = {"busy": {"too_many"}, "middle": {"between_ranges"}, OTHER: {"too_few"},
                   "main_brief": {"span_too_short", "target_reached"}}
        for row in manifest["skipped"]["candidates"]:
            self.assertIn(row["reason"], allowed[row["address"]], row)
        self.assertTrue(manifest["checks"]["all_passed"])
        self.assertEqual(manifest["checks"]["fallback_cross_check"][0]["shared"], 4)
        # Only DS1 addresses sit at the top level of the raw folder; a span-too-short history is moved aside.
        self.assertTrue((self.raw / "main_ok" / snapshot.STATS_FILE).exists())
        self.assertFalse((self.raw / "main_brief").exists())

    def test_second_run_rebuilds_from_the_cache_without_requests(self):
        replies, fallback_replies = fake_world()
        _, first = self.run_once(FakeClient(replies), FakeClient(fallback_replies))
        offline, offline_fallback = FakeClient({}), FakeClient({})
        code, second = self.run_once(offline, offline_fallback)
        self.assertEqual(code, 0)
        self.assertEqual(offline.stats.requests + offline_fallback.stats.requests, 0)
        self.assertEqual(second["kept"], first["kept"])
        self.assertEqual(second["selection_rule"]["recorded_utc"], first["selection_rule"]["recorded_utc"])
        self.assertEqual(len(second["retrieval"]["runs"]), 1)  # the offline run adds no run record

    def test_a_history_that_changes_mid_fetch_is_fetched_again(self):
        replies, fallback_replies = fake_world()
        primary = FakeClient(replies)
        calls = {"n": 0}
        stale = address_reply("main_ok", history("main_ok", [0, 5, 12, 30]), tx_count=3)
        original = primary.get_text

        def get_text(path: str) -> str:
            # The very first history-fetch stats call sees one transaction fewer, as if a block landed.
            if path == "/address/main_ok":
                calls["n"] += 1
                if calls["n"] == 2:
                    primary.stats.requests += 1
                    return json.dumps(stale)
            return original(path)

        primary.get_text = get_text  # type: ignore[method-assign]
        code, manifest = self.run_once(primary, FakeClient(fallback_replies))
        self.assertEqual(code, 0, manifest["status"])
        self.assertIn("main_ok", [k["address"] for k in manifest["kept"]])


if __name__ == "__main__":
    unittest.main()
