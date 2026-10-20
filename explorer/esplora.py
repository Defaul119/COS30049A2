"""A small, polite client for the Esplora HTTP API.

Works with Blockstream (https://blockstream.info/api) and the Esplora-compatible
mempool.space API. It only sends GET requests, waits between requests, and
backs off on HTTP 429 and 5xx responses (Risk 15). Standard library only, so
the D1 check runs on a bare Python install.

API reference: https://github.com/Blockstream/esplora/blob/master/API.md
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

USER_AGENT = "BitcoinAddressActivityExplorer/0.1 (COS30049 student research project)"
PAGE_SIZE = 25  # Esplora returns 25 confirmed transactions per history page


class EsploraError(RuntimeError):
    def __init__(self, status: int | None, url: str, detail: str):
        super().__init__(f"{status or 'network error'} for {url}: {detail[:200]}")
        self.status = status
        self.url = url
        self.detail = detail


@dataclass
class RequestStats:
    requests: int = 0
    retries: int = 0
    rate_limited: int = 0
    server_errors: int = 0
    network_errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        lat = sorted(self.latencies_ms)
        return {
            "requests": self.requests,
            "retries": self.retries,
            "http_429": self.rate_limited,
            "http_5xx": self.server_errors,
            "network_errors": self.network_errors,
            "median_latency_ms": round(lat[len(lat) // 2]) if lat else None,
            "max_latency_ms": round(lat[-1]) if lat else None,
        }


class EsploraClient:
    def __init__(
        self,
        base_url: str,
        pause_seconds: float = 0.5,
        timeout_seconds: float = 30,
        max_retries: int = 5,
    ):
        self.base_url = base_url.rstrip("/")
        self.pause_seconds = pause_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.stats = RequestStats()
        self._last_request_at = 0.0

    # -- transport -----------------------------------------------------------

    def _wait_turn(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.pause_seconds:
            time.sleep(self.pause_seconds - elapsed)

    def get_text(self, path: str) -> str:
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                    body = resp.read().decode("utf-8")
                self._record(started)
                return body
            except urllib.error.HTTPError as err:
                self._record(started)
                detail = err.read().decode("utf-8", errors="replace")
                retryable = err.code == 429 or err.code >= 500
                if err.code == 429:
                    self.stats.rate_limited += 1
                elif err.code >= 500:
                    self.stats.server_errors += 1
                if retryable and attempt < self.max_retries:
                    self._back_off(attempt, err.headers.get("Retry-After"))
                    continue
                raise EsploraError(err.code, url, detail) from err
            except (urllib.error.URLError, TimeoutError) as err:
                self._record(started)
                self.stats.network_errors += 1
                if attempt < self.max_retries:
                    self._back_off(attempt, None)
                    continue
                raise EsploraError(None, url, str(err)) from err
        raise AssertionError("unreachable")

    def get_json(self, path: str) -> Any:
        return json.loads(self.get_text(path))

    def _record(self, started: float) -> None:
        now = time.monotonic()
        self._last_request_at = now
        self.stats.requests += 1
        self.stats.latencies_ms.append((now - started) * 1000)

    def _back_off(self, attempt: int, retry_after: str | None) -> None:
        self.stats.retries += 1
        delay = 2.0**attempt
        if retry_after and retry_after.isdigit():
            delay = max(delay, float(retry_after))
        time.sleep(min(delay, 60.0))

    # -- endpoints -----------------------------------------------------------

    def tip_height(self) -> int:
        return int(self.get_text("/blocks/tip/height"))

    def tip_hash(self) -> str:
        return self.get_text("/blocks/tip/hash").strip()

    def block(self, block_hash: str) -> dict[str, Any]:
        return self.get_json(f"/block/{block_hash}")

    def block_txs(self, block_hash: str, start_index: int = 0) -> list[dict[str, Any]]:
        return self.get_json(f"/block/{block_hash}/txs/{start_index}")

    def address(self, address: str) -> dict[str, Any]:
        return self.get_json(f"/address/{address}")

    def address_txs_chain(
        self, address: str, last_seen_txid: str | None = None
    ) -> list[dict[str, Any]]:
        suffix = f"/{last_seen_txid}" if last_seen_txid else ""
        return self.get_json(f"/address/{address}/txs/chain{suffix}")

    def address_history(
        self, address: str, cap: int = 1000
    ) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
        """Fetch confirmed history newest-first, following the last-seen-txid cursor.

        Returns (transactions, raw_pages). Stops at `cap` transactions or when a
        page comes back shorter than PAGE_SIZE.
        """
        txs: list[dict[str, Any]] = []
        pages: list[list[dict[str, Any]]] = []
        last_seen: str | None = None
        while len(txs) < cap:
            page = self.address_txs_chain(address, last_seen)
            pages.append(page)
            txs.extend(page)
            if len(page) < PAGE_SIZE:
                break
            last_seen = page[-1]["txid"]
        return txs[:cap], pages
