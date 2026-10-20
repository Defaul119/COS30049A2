"""Address-related amount and direction for one transaction (report §2.3).

received = sum of outputs paying the address
sent     = sum of inputs spent from the address
net      = received - sent; positive is "in", negative is "out"

All values are integer satoshis. Esplora includes each input's `prevout`, so no
extra request is needed to know which inputs belonged to the address.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SATS_PER_BTC = 100_000_000


@dataclass(frozen=True)
class AddressFlow:
    txid: str
    confirmed: bool
    block_height: int | None
    block_time: int | None  # unix seconds, UTC
    received: int
    sent: int

    @property
    def net(self) -> int:
        return self.received - self.sent

    @property
    def direction(self) -> str:
        if self.net > 0:
            return "in"
        if self.net < 0:
            return "out"
        return "none"


def address_flow(tx: dict[str, Any], address: str) -> AddressFlow:
    received = sum(
        out.get("value", 0)
        for out in tx.get("vout", [])
        if out.get("scriptpubkey_address") == address
    )
    sent = sum(
        vin["prevout"].get("value", 0)
        for vin in tx.get("vin", [])
        if vin.get("prevout") and vin["prevout"].get("scriptpubkey_address") == address
    )
    status = tx.get("status", {})
    return AddressFlow(
        txid=tx["txid"],
        confirmed=bool(status.get("confirmed")),
        block_height=status.get("block_height"),
        block_time=status.get("block_time"),
        received=received,
        sent=sent,
    )


def format_btc(sats: int, signed: bool = False) -> str:
    sign = "-" if sats < 0 else ("+" if signed and sats > 0 else "")
    whole, frac = divmod(abs(sats), SATS_PER_BTC)
    return f"{sign}{whole}.{frac:08d} BTC"
