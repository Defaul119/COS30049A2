"""Tests for the Esplora client's transport against a real local HTTP server: gzip and retries.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import gzip
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from explorer.esplora import EsploraClient, EsploraError

PAGE = [{"txid": "ab" * 32, "vin": [], "vout": []}] * 3


class Handler(BaseHTTPRequestHandler):
    limited_calls = 0
    dropped_calls = 0

    def do_GET(self):  # noqa: N802 - the http.server API name
        body = json.dumps(PAGE).encode()
        if self.path == "/drop-always" or (self.path == "/drop-once" and Handler.dropped_calls < 1):
            # Close without any reply: the client sees http.client.RemoteDisconnected, as from
            # mempool.space on 23 Oct, which used to escape the retries and stop the DS2 crawl.
            if self.path == "/drop-once":
                Handler.dropped_calls += 1
            self.close_connection = True
            return
        if self.path == "/limited" and Handler.limited_calls < 1:
            Handler.limited_calls += 1
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(b'{"error":"Too Many Requests"}')
            return
        gzipped = "gzip" in self.headers.get("Accept-Encoding", "") and self.path != "/plain"
        if gzipped:
            body = gzip.compress(body)
        self.send_response(200)
        if gzipped:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep test output quiet
        pass


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_gzip_and_plain_replies_read_the_same(self):
        client = EsploraClient(self.base, pause_seconds=0.0)
        self.assertEqual(client.get_json("/gz"), PAGE)
        self.assertEqual(client.get_json("/plain"), PAGE)

    def test_a_rate_limited_request_is_retried(self):
        client = EsploraClient(self.base, pause_seconds=0.0, max_retries=2)
        self.assertEqual(client.get_json("/limited"), PAGE)
        self.assertEqual((client.stats.rate_limited, client.stats.retries), (1, 1))

    def test_a_dropped_connection_is_retried(self):
        client = EsploraClient(self.base, pause_seconds=0.0, max_retries=2)
        self.assertEqual(client.get_json("/drop-once"), PAGE)
        self.assertEqual((client.stats.network_errors, client.stats.retries), (1, 1))

    def test_a_connection_that_keeps_dropping_becomes_an_esplora_error(self):
        # An EsploraError is what explorer.corpus records as a fetch_error visit; anything else stops the crawl.
        client = EsploraClient(self.base, pause_seconds=0.0, max_retries=1)
        with self.assertRaises(EsploraError):
            client.get_text("/drop-always")

    def test_errors_carry_the_status(self):
        client = EsploraClient("http://127.0.0.1:9", pause_seconds=0.0, max_retries=0, timeout_seconds=2)
        with self.assertRaises(EsploraError) as caught:
            client.get_text("/nothing-listens-here")
        self.assertIsNone(caught.exception.status)


if __name__ == "__main__":
    unittest.main()
