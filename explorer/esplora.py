"""A small, polite client for the Esplora HTTP API.

Works with Blockstream (https://blockstream.info/api) and the Esplora-compatible
mempool.space API. It only sends GET requests, waits between requests, and
backs off on HTTP 429 and 5xx responses (Risk 15). Standard library only, so
the D1 check runs on a bare Python install.

Since July 2025 Blockstream caps unauthenticated use at 700 requests per hour
per IP (and 500,000 per month), so a long collection is rate-limited even at a
polite pace. FailoverClient sends requests to the other host for a while when
the active one answers HTTP 429.

API reference: https://github.com/Blockstream/esplora/blob/master/API.md
"""
from __future__ import annotations

import gzip
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

USER_AGENT = "BitcoinAddressActivityExplorer/0.1 (COS30049 student research project)"
PAGE_SIZE = 25  # Esplora returns 25 confirmed transactions per history page

log = logging.getLogger(__name__)


def decode_body(raw: bytes, encoding: str | None, errors: str = "strict") -> str:
    """The reply as text, decompressed first if the host sent it gzipped."""
    if encoding and encoding.lower() == "gzip":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors=errors)


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




class EsploraEndpoints:
    """The API calls, built on `get_text`/`get_json`, which each client class supplies."""

    def get_text(self, path: str) -> str:
        raise NotImplementedError

    def get_json(self, path: str) -> Any:
        return json.loads(self.get_text(path))

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


class EsploraClient(EsploraEndpoints):
    def __init__(
        self,
        base_url: str,
        pause_seconds: float = 0.5,
        timeout_seconds: float = 30,
        max_retries: int = 5,
        pacer: Callable[[], None] | None = None,
    ):
        """`pacer`, if given, replaces the per-client pause: several clients (one per worker thread) on
        the same host share one pacer, so the host still sees requests spaced as settings.json says."""
        self.base_url = base_url.rstrip("/")
        self.pause_seconds = pause_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.pacer = pacer
        self.stats = RequestStats()
        self._last_request_at = 0.0

    # -- transport -----------------------------------------------------------

    def _wait_turn(self) -> None:
        if self.pacer is not None:
            self.pacer()
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.pause_seconds:
            time.sleep(self.pause_seconds - elapsed)

    def get_text(self, path: str) -> str:
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            # gzip: mempool.space throttles large replies to ~235 KB/s, and history pages of
            # consolidation-heavy addresses reach 7 MB of JSON, which compresses about 4x.
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/json", "Accept-Encoding": "gzip"}
            )
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                    body = decode_body(resp.read(), resp.headers.get("Content-Encoding"))
                self._record(started)
                return body
            except urllib.error.HTTPError as err:
                self._record(started)
                detail = decode_body(err.read(), err.headers.get("Content-Encoding"), errors="replace")
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

    def host_stats(self) -> dict[str, dict[str, Any]]:
        """Request statistics keyed by host, the shape FailoverClient also returns."""
        return {self.base_url: self.stats.summary()}

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


class FailoverClient(EsploraEndpoints):
    """Sends each request to the active host and switches host when the active one rate-limits.

    Every host serves the same Esplora API over the same chain, so a reply from either is
    interchangeable. After a switch the other host stays active for `hold_seconds` (the plan's
    "switch to mempool.space for an hour"), then the first host is tried again. `last_base_url`
    names the host that served the latest reply, so callers can record where data came from.
    """

    def __init__(self, clients: list[EsploraClient], hold_seconds: float):
        self.clients = clients
        self.hold_seconds = hold_seconds
        self.switches = 0
        self.last_base_url: str | None = None
        self._active = 0
        self._hold_until = 0.0

    @property
    def base_url(self) -> str:
        return self.clients[self._active].base_url

    def get_text(self, path: str) -> str:
        if self._active and time.monotonic() >= self._hold_until:
            self._active = 0
        error: EsploraError | None = None
        for _ in self.clients:
            client = self.clients[self._active]
            try:
                text = client.get_text(path)
            except EsploraError as err:
                if err.status != 429:
                    raise
                error = err
                self._active = (self._active + 1) % len(self.clients)
                self._hold_until = time.monotonic() + self.hold_seconds
                self.switches += 1
                log.warning("%s is rate-limiting; using %s for the next %.0f min", client.base_url,
                            self.clients[self._active].base_url, self.hold_seconds / 60)
                continue
            self.last_base_url = client.base_url
            return text
        raise error  # every host is rate-limiting

    def host_stats(self) -> dict[str, dict[str, Any]]:
        return {c.base_url: c.stats.summary() for c in self.clients}
