"""Tests for the DS2 crawl (plan §6.2): hashing, pacing, DS1 exclusion and resuming.

Two fake Esplora hosts stand in for Blockstream and mempool.space, so no network is needed.
Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

import pandas as pd

from explorer import corpus
from explorer.esplora import EsploraClient, EsploraError
from explorer.settings import SETTINGS
from explorer.snapshot import write_gz
from tests.test_snapshot import DAY, T0, address_reply, history

SALT = "test-salt"


class FakeHost(EsploraClient):
    """A real client whose transport answers from a dict, so paging and parsing run unchanged."""

    def __init__(self, base_url: str, replies: dict[str, Any]):
        super().__init__(base_url, pause_seconds=0.0)
        self.replies = replies

    def get_text(self, path: str) -> str:
        self.stats.requests += 1
        if path not in self.replies:
            raise EsploraError(404, self.base_url + path, "not found")
        reply = self.replies[path]
        return reply if isinstance(reply, str) else json.dumps(reply)


def params(**overrides: Any) -> dict[str, Any]:
    p = {"seed": 1, "blockCount": 2, "oldestBlockBack": 10, "newestBlockBack": 2, "txPagesPerBlock": 4,
         "minTx": 3, "maxTx": 5, "targetAddresses": 2, "minimumAddresses": 1, "workersPerSource": 1,
         "idHexChars": 16, "progressEvery": 1}
    p.update(overrides)
    return p


def world() -> dict[str, Any]:
    histories = {"a_ok": history("a_ok", [0, 5, 9, 30]), "b_ok": history("b_ok", [1, 2, 40]),
                 "ds1addr": history("ds1addr", [0, 1, 2]), "few": history("few", [3])}
    replies: dict[str, Any] = {
        "/blocks/tip/height": "100", "/block-height/90": "h90", "/block-height/98": "h98",
        "/block/h90": {"timestamp": T0, "tx_count": 2}, "/block/h98": {"timestamp": T0 + DAY, "tx_count": 1},
        "/block/h90/txs/0": [{"vout": [{"scriptpubkey_address": a, "scriptpubkey_type": "p2tr"} for a in ("a_ok", "few")]},
                             {"vout": [{"scriptpubkey_address": "ds1addr", "scriptpubkey_type": "p2tr"}]}],
        "/block/h98/txs/0": [{"vout": [{"scriptpubkey_address": a, "scriptpubkey_type": "p2tr"} for a in ("b_ok", "busy")]}],
        "/address/busy": {"chain_stats": {"tx_count": 900, "funded_txo_sum": 0, "spent_txo_sum": 0}},
    }
    for address, txs in histories.items():
        replies[f"/address/{address}"] = address_reply(address, txs)
        replies[f"/address/{address}/txs/chain"] = txs
    return replies


class HelperTests(unittest.TestCase):
    def test_salted_ids_are_stable_short_and_salt_dependent(self):
        a = corpus.salted_id(SALT, "bc1qexample", 16)
        self.assertEqual(a, corpus.salted_id(SALT, "bc1qexample", 16))
        self.assertEqual(len(a), 16)
        self.assertNotEqual(a, corpus.salted_id("other-salt", "bc1qexample", 16))

    def test_shared_pace_spaces_request_starts(self):
        pace = corpus.SharedPace(0.1)
        started = time.perf_counter()
        for _ in range(4):
            pace.wait()
        # Three gaps after the first start; the tolerance covers the OS timer's ~16 ms resolution.
        self.assertGreaterEqual(time.perf_counter() - started, 0.3 - 0.02)

    def test_ds1_exclusions_cover_the_whole_ds1_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "esplora"
            blocks = raw / "selection" / "blocks"
            manifest = Path(tmp) / "manifest.json"
            n_blocks = SETTINGS["collection"]["snapshotBlockCount"]
            for h in range(n_blocks - 1):
                write_gz(blocks / str(h) / "block.json.gz", "{}")
            with self.assertRaises(SystemExit):  # one block still unread: the pool is not fixed yet
                corpus.ds1_exclusions(raw, manifest)
            write_gz(blocks / str(n_blocks) / "block.json.gz", "{}")
            write_gz(blocks / "0" / "txs-0000.json.gz",
                     json.dumps([{"vout": [{"scriptpubkey_address": "pool_a"}, {"scriptpubkey_address": "pool_b"}]}]))
            manifest.write_text(json.dumps({"kept": [{"address": "kept_x"}]}), encoding="utf-8")
            self.assertEqual(corpus.ds1_exclusions(raw, manifest), {"pool_a", "pool_b", "kept_x"})

    def test_salt_is_created_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "salt.txt"
            first = corpus.load_or_create_salt(path)
            self.assertEqual(corpus.load_or_create_salt(path), first)
            self.assertEqual(len(first), 64)


class CrawlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.dirs = {"raw_dir": root / "raw", "interim_dir": root / "interim", "out_dir": root / "out"}

    def tearDown(self):
        self.tmp.cleanup()

    def crawl(self, replies: dict[str, Any]) -> tuple[int, corpus.ClientPool]:
        pool = corpus.ClientPool([FakeHost("https://one.example/api", replies), FakeHost("https://two.example/api", replies)])
        code = corpus.run_corpus(pool, params(), ds1={"ds1addr"}, salt=SALT, **self.dirs)
        return code, pool

    def test_crawl_keeps_in_range_addresses_hashed_and_excludes_ds1(self):
        code, _ = self.crawl(world())
        self.assertEqual(code, 0)
        frame = pd.read_parquet(self.dirs["out_dir"] / "ds2.parquet")
        self.assertEqual(list(frame.columns), corpus.COLUMNS)
        self.assertEqual(frame.groupby("addr_id").size().sort_values().tolist(), [3, 4])
        self.assertNotIn(corpus.salted_id(SALT, "ds1addr", 16), set(frame["addr_id"]))
        manifest_text = (self.dirs["out_dir"] / "manifest.json").read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        self.assertTrue(manifest["checks"]["all_passed"])
        self.assertEqual(manifest["candidate_pool"]["ds1_removed"], 1)
        for raw in ("a_ok", "b_ok", "ds1addr"):  # no raw address may reach the shipped files
            self.assertNotIn(raw, manifest_text)
        a = frame[frame["addr_id"] == corpus.salted_id(SALT, "a_ok", 16)]
        self.assertEqual(a["received_sats"].sum() - a["sent_sats"].sum(), 2 * 10_000 - 2 * 4_000)

    def test_rerun_resumes_without_crawling_again(self):
        self.crawl(world())
        before = pd.read_parquet(self.dirs["out_dir"] / "ds2.parquet")
        code, pool = self.crawl({})  # every reply would now be a 404; only the caches can answer
        self.assertEqual(code, 0)
        self.assertEqual(sum(c.stats.requests for c in pool.clients), 0)
        pd.testing.assert_frame_equal(pd.read_parquet(self.dirs["out_dir"] / "ds2.parquet"), before)


if __name__ == "__main__":
    unittest.main()
