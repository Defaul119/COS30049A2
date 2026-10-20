"""Unit tests for the address amount and direction rule (report §2.3, Risk 18).

Run from the project root:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from explorer.amounts import address_flow, format_btc

ROOT = Path(__file__).resolve().parent.parent
ME = "bc1qexampleaddressxxxxxxxxxxxxxxxxxxxxxx"
OTHER = "bc1qsomeoneelsexxxxxxxxxxxxxxxxxxxxxxxxx"


def out(address: str | None, value: int) -> dict:
    return {"scriptpubkey_address": address, "value": value}


def vin(address: str | None, value: int) -> dict:
    return {"prevout": {"scriptpubkey_address": address, "value": value}}


def tx(vins: list, vouts: list, confirmed: bool = True) -> dict:
    return {
        "txid": "ab" * 32,
        "vin": vins,
        "vout": vouts,
        "status": {"confirmed": confirmed, "block_height": 900000, "block_time": 1_790_000_000},
    }


class AddressFlowTests(unittest.TestCase):
    def test_simple_receive(self):
        f = address_flow(tx([vin(OTHER, 50_000)], [out(ME, 40_000), out(OTHER, 9_000)]), ME)
        self.assertEqual((f.received, f.sent, f.net, f.direction), (40_000, 0, 40_000, "in"))

    def test_send_with_change_back_to_same_address(self):
        f = address_flow(tx([vin(ME, 100_000)], [out(OTHER, 70_000), out(ME, 29_000)]), ME)
        self.assertEqual((f.received, f.sent, f.net, f.direction), (29_000, 100_000, -71_000, "out"))

    def test_several_inputs_and_outputs_for_the_address(self):
        t = tx([vin(ME, 10_000), vin(ME, 15_000), vin(OTHER, 5_000)], [out(ME, 1_000), out(ME, 2_000), out(OTHER, 26_000)])
        f = address_flow(t, ME)
        self.assertEqual((f.received, f.sent, f.net), (3_000, 25_000, -22_000))

    def test_coinbase_input_has_no_prevout(self):
        coinbase = {"is_coinbase": True, "prevout": None}
        f = address_flow(tx([coinbase], [out(ME, 312_500_000)]), ME)
        self.assertEqual((f.received, f.sent, f.direction), (312_500_000, 0, "in"))

    def test_outputs_without_an_address_are_ignored(self):
        f = address_flow(tx([vin(ME, 10_000)], [out(None, 0), out(OTHER, 9_000)]), ME)
        self.assertEqual((f.received, f.sent, f.net), (0, 10_000, -10_000))

    def test_equal_in_and_out_has_no_direction(self):
        f = address_flow(tx([vin(ME, 5_000)], [out(ME, 5_000)]), ME)
        self.assertEqual((f.net, f.direction), (0, "none"))

    def test_status_fields_are_kept(self):
        f = address_flow(tx([vin(OTHER, 1)], [out(ME, 1)], confirmed=False), ME)
        self.assertFalse(f.confirmed)
        self.assertEqual(f.block_height, 900000)

    def test_format_btc(self):
        self.assertEqual(format_btc(1), "0.00000001 BTC")
        self.assertEqual(format_btc(-150_000_000), "-1.50000000 BTC")
        self.assertEqual(format_btc(42, signed=True), "+0.00000042 BTC")


class RealD1DataTests(unittest.TestCase):
    """The rule reproduces the API's own totals over a real address's whole history."""

    def setUp(self):
        d1 = ROOT / "data" / "d1"
        dirs = [p for p in d1.iterdir() if p.is_dir()] if d1.exists() else []
        if not dirs:
            self.skipTest("run scripts/d1_check.py first")
        self.dir = dirs[0]
        self.address = self.dir.name

    def test_sums_match_chain_stats(self):
        stats = json.loads((self.dir / "address-stats.json").read_text(encoding="utf-8"))["chain_stats"]
        txs = [t for page in sorted(self.dir.glob("blockstream-page-*.json")) for t in json.loads(page.read_text(encoding="utf-8"))]
        flows = [address_flow(t, self.address) for t in txs]
        self.assertEqual(len(flows), stats["tx_count"])
        self.assertEqual(sum(f.received for f in flows), stats["funded_txo_sum"])
        self.assertEqual(sum(f.sent for f in flows), stats["spent_txo_sum"])
        self.assertTrue(all(f.received > 0 or f.sent > 0 for f in flows))


if __name__ == "__main__":
    unittest.main()
