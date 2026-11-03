"""The Explorer in a terminal: pick a snapshot address, see its flagged transactions, ask why, try a what-if.

Serves plan §12.1 (the brief's "use the model for prediction") for a person at a terminal, for testing and the
M4 check: it calls the same Predictor as `predict` and A3's back end, and shows the answers as text instead of
JSON. For every transaction it also lists each saved model's verdict and says what kind of model it is, so a
tester can see which model decides the Explorer's flag (settings explorerMethod) and how the others compare.

    python -m explorer.cli explore

Only the DS1 snapshot's addresses can be scored: the models were fitted on them (A3 adds live look-ups).
Every line printed passes the wording rule (A1 NFR-01), and every flagged explanation carries the limitation
line. Rows per page are in settings.json -> explore.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from .explain import FEATURE_NAMES, METHOD_NAMES, check_wording, describe_value, format_btc, plural, reasons_for
from .predict import PRECOMPUTED_ONLY, InsufficientHistory, Predictor, day
from .settings import SETTINGS

# One line per saved model for the verdict table: what kind of model is doing the work.
METHOD_KINDS = {
    "baseline": "statistics: |z| against the address's own median and MAD (A1's rule; nothing trained)",
    "isolation_forest": "outlier detection: random isolation trees",
    "lof": "outlier detection: local density against the nearest neighbours",
    "kmeans": "clustering: distance to the nearest K-Means centre",
    "dbscan": "clustering: distance to DBSCAN's nearest dense core",
    "regression": "regression: the miss between the predicted and actual size and gap",
    "regression_interval": "regression: distance outside the conformal expected range",
    "fusion": "combination: regression and Isolation Forest scores together",
    "txgpt_method": "language model: TxGPT's surprisal (trained from scratch on DS2)",
}
WRITER_NAMES = {"text150m": "the 152M Explorer Assistant (a language model trained from scratch)"}
# "5", "a 5" or "a5": open transaction 5, the second form with the Assistant's explanation.
OPEN_PATTERN = re.compile(r"^(a\s*)?(\d+)$", re.IGNORECASE)


class Quit(Exception):
    """The user typed q, or the input ran out: leave from wherever the loop is."""


@dataclass
class Console:
    """Where the Explorer reads and writes; tests pass scripted input and collect the output."""
    read: Callable[[str], str] = input
    write: Callable[[str], None] = print

    def ask(self, prompt: str) -> str:
        try:
            answer = self.read(prompt).strip()
        except EOFError:  # input piped from a file ran out
            raise Quit from None
        if answer.lower() in ("q", "quit", "exit"):
            raise Quit
        return answer

    def say(self, text: str = "") -> None:
        # User input is never echoed, so only the Explorer's own words reach this check.
        check_wording(text)
        self.write(text)


def short_time(iso_time: str) -> str:
    """2024-07-11T03:32:50Z -> 2024-07-11 03:32 (UTC, as every time here)."""
    return iso_time[:16].replace("T", " ")


def main_reason(tx: dict[str, Any]) -> str:
    """The strongest reason the template gives, with its z, for one row of a table."""
    if not tx["scored"]:
        return "not scored: too little earlier history"
    reasons, _ = reasons_for(tx["features"])
    return f"{reasons[0]['headline']} (z {reasons[0]['z']:.1f})" if reasons else "-"


def pages(count: int, per_page: int) -> int:
    return max(1, -(-count // per_page))


class Explorer:
    """The interactive loop: address list -> one address -> one transaction or a what-if."""

    def __init__(self, predictor: Predictor, console: Console | None = None):
        self.p = predictor
        self.io = console or Console()
        self.addresses_per_page = SETTINGS["explore"]["addressesPerPage"]
        self.tx_per_page = SETTINGS["explore"]["transactionsPerPage"]

    def run(self) -> int:
        snap = self.p.snapshot()
        self.io.say("Bitcoin Address Activity Explorer")
        self.io.say(f"Snapshot of {snap['addressCount']} addresses from {snap['source']}, taken {snap['snapshotDate']}.")
        self.io.say(f"Flags are decided by {snap['method']['name']}: any feature with |z| at least {self.p.flag_abs_z():g}.")
        self.io.say(SETTINGS["limitationLine"])
        try:
            while True:
                self.show_address(self.choose_address())
        except Quit:
            self.io.say("Goodbye.")
        return 0

    # -- the address list ------------------------------------------------------------

    def address_table(self) -> pd.DataFrame:
        """One row per snapshot address: transactions, how many are scored and flagged, first and last day.

        Main addresses first, busiest first; the short-history ones (too little history to score) last.
        """
        table = self.p.features.groupby("address").agg(
            transactions=("txid", "size"), scored=("scorable", "sum"),
            first=("block_time_utc", "min"), last=("block_time_utc", "max"))
        flagged = self.p.scores.groupby("address")[f"flag_{self.p.selected}"].sum()
        table["flagged"] = flagged.reindex(table.index).fillna(0).astype(int)
        roles = self.p.transactions.groupby("address")["role"].first()
        table["short"] = roles.reindex(table.index).eq("short")
        return table.sort_values(["short", "transactions"], ascending=[True, False]).reset_index()

    def choose_address(self) -> str:
        table = self.address_table()
        width = int(table["address"].str.len().max())
        total, page = pages(len(table), self.addresses_per_page), 0
        while True:
            start = page * self.addresses_per_page
            self.io.say(f"\nAddresses (page {page + 1} of {total})")
            self.io.say(f"{'#':>3}  {'Address':<{width}}  {'Tx':>5}  {'Scored':>6}  {'Flagged':>7}  {'First':<10}  Last")
            for i, row in table.iloc[start:start + self.addresses_per_page].iterrows():
                note = "  (short history)" if row["short"] else ""
                self.io.say(f"{i + 1:>3}  {row['address']:<{width}}  {row['transactions']:>5}  {int(row['scored']):>6}  "
                            f"{row['flagged']:>7}  {day(row['first'])}  {day(row['last'])}{note}")
            answer = self.io.ask("Pick a number or paste an address (n next page, p previous page, q quit): ")
            if answer.lower() == "n":
                page = min(page + 1, total - 1)
            elif answer.lower() == "p":
                page = max(page - 1, 0)
            elif answer.isdigit() and 1 <= int(answer) <= len(table):
                return str(table.at[int(answer) - 1, "address"])
            elif answer in set(table["address"]):
                return answer
            elif answer:
                self.io.say(f"Not on the list: type a number from 1 to {len(table)}, or paste one of the addresses "
                            "shown. Only the snapshot's addresses can be scored.")

    # -- one address -------------------------------------------------------------------

    def show_address(self, address: str) -> None:
        try:
            result = self.p.score_address(address)
        except InsufficientHistory as err:
            history, required = err.body["history"], err.body["history"]["required"]
            self.io.say(f"\n{address}")
            self.io.say(f"{err.body['message']} It has {plural(history['earlierTx'], 'transaction')} over "
                        f"{plural(round(history['spanDays']), 'day')}; scoring needs at least "
                        f"{plural(required['minTx'], 'transaction')} over {plural(required['minSpanDays'], 'day')}.")
            self.io.ask("Press Enter to go back to the list (q quits): ")
            return
        txs = result["transactions"]
        flagged = [(i, t) for i, t in enumerate(txs, 1) if t["flagged"]]
        self.summary(result, flagged)
        while True:
            answer = self.io.ask("\n[#] why   [a #] the Assistant explains   [f] flagged list   [t] all transactions   "
                                 "[w] what-if   [b] back   [q] quit: ")
            if not answer:
                continue
            if answer.lower() == "b":
                return
            if answer.lower() == "f":
                self.summary(result, flagged)
            elif answer.lower() == "t":
                self.browse(txs)
            elif answer.lower() == "w":
                self.what_if(address)
            elif not self.open_transaction(answer, txs):
                self.io.say(f"Not understood: type a transaction number from 1 to {len(txs)}, or one of the letters.")

    def summary(self, result: dict[str, Any], flagged: list[tuple[int, dict[str, Any]]]) -> None:
        totals, span = result["totals"], result["range"]
        self.io.say(f"\nAddress {result['address']}")
        self.io.say(f"{totals['txCount']} transactions from {span['start'][:10]} to {span['end'][:10]}; "
                    f"{totals['scoredCount']} scored, {totals['flaggedCount']} flagged on {totals['flaggedDays']} days.")
        self.tx_table(f"Flagged transactions ({len(flagged)})", flagged)

    def tx_table(self, title: str, rows: list[tuple[int, dict[str, Any]]]) -> None:
        self.io.say(f"\n{title}")
        if not rows:
            self.io.say("  (none)")
            return
        self.io.say(f"{'#':>5}  {'Time (UTC)':<16}  {'Dir':<3}  {'Amount':>18}  {'Score':>6}  {'Flag':<4}  Main reason")
        for number, tx in rows:
            score = f"{tx['score']:.2f}" if tx["scored"] and tx["score"] is not None else "-"
            self.io.say(f"{number:>5}  {short_time(tx['time'])}  {tx['direction']:<3}  {format_btc(tx['netSats']):>18}  "
                        f"{score:>6}  {'FLAG' if tx['flagged'] else '':<4}  {main_reason(tx)}")

    def browse(self, txs: list[dict[str, Any]]) -> None:
        """Every transaction of the address, page by page; any number opens that transaction."""
        numbered = list(enumerate(txs, 1))
        total, page = pages(len(numbered), self.tx_per_page), 0
        while True:
            start = page * self.tx_per_page
            self.tx_table(f"All transactions (page {page + 1} of {total})", numbered[start:start + self.tx_per_page])
            answer = self.io.ask("[#] why   [a #] the Assistant explains   [n] next page   [p] previous page   "
                                 "[b] back   [q] quit: ")
            if not answer:
                continue
            if answer.lower() == "b":
                return
            if answer.lower() == "n":
                page = min(page + 1, total - 1)
            elif answer.lower() == "p":
                page = max(page - 1, 0)
            elif not self.open_transaction(answer, txs):
                self.io.say(f"Not understood: type a transaction number from 1 to {len(txs)}, or one of the letters.")

    def open_transaction(self, answer: str, txs: list[dict[str, Any]]) -> bool:
        """Handle "5" or "a 5"; False if the answer is neither or out of range."""
        match = OPEN_PATTERN.match(answer)
        if not match or not 1 <= int(match.group(2)) <= len(txs):
            return False
        number = int(match.group(2))
        self.show_transaction(txs[number - 1], f"Transaction #{number}", use_assistant=bool(match.group(1)))
        return True

    # -- one transaction ---------------------------------------------------------------

    def show_transaction(self, tx: dict[str, Any], label: str, use_assistant: bool) -> None:
        self.io.say(f"\n{label}: {short_time(tx['time'])} UTC, {tx['direction']}, {format_btc(tx['netSats'])}")
        if tx.get("sourceUrl"):
            self.io.say(f"  {tx['sourceUrl']}")
        if not tx["scored"]:
            self.io.say(f"  {tx['explanation']}")
            return
        self.features_block(tx)
        self.verdicts_block(tx)
        self.explanation_block(tx, use_assistant)

    def features_block(self, tx: dict[str, Any]) -> None:
        """F1-F4 against the address's own history: the inputs to every model and the template's reasons."""
        self.io.say(f"\nCompared with this address's own history (the flag line is |z| {self.p.flag_abs_z():g}):")
        self.io.say(f"  {'Feature':<14}  {'This transaction':<22}  {'Usual':<22}  {'z':>6}")
        for f in tx["features"]:
            z = "-" if f["z"] is None else f"{f['z']:.1f}"
            mark = {"flag": "  <- over the flag line", "contributing": "  <- stands out"}.get(f["role"], "")
            self.io.say(f"  {FEATURE_NAMES[f['id']]:<14}  {describe_value(f['id'], f['value']):<22}  "
                        f"{describe_value(f['id'], f['usual'], usual=True):<22}  {z:>6}{mark}")

    def verdicts_block(self, tx: dict[str, Any]) -> None:
        """Every saved model's score and flag for this transaction, and what kind of model each one is."""
        width = max(len(name) for name in METHOD_NAMES.values())
        votes = sum(bool(v) for v in tx["flags"].values())
        self.io.say(f"\nEvery saved model's verdict ({votes} of {len(tx['flags'])} that could score it flag it; "
                    f"the Explorer shows {METHOD_NAMES[self.p.selected]}'s):")
        for name, title in METHOD_NAMES.items():
            if name in tx["scores"]:
                flag = "FLAG" if tx["flags"].get(name) else "-"
                score = "-" if tx["scores"][name] is None else f"{tx['scores'][name]:.2f}"
            elif name in PRECOMPUTED_ONLY:  # TxGPT's scores are made in advance (lm.tx_score), not for a what-if
                flag, score = "n/a", "-"
            else:
                continue
            decides = " (decides the flag)" if name == self.p.selected else ""
            self.io.say(f"  {flag:<4}  {title:<{width}}  {score:>7}  {METHOD_KINDS.get(name, '')}{decides}")

    def explanation_block(self, tx: dict[str, Any], use_assistant: bool) -> None:
        if use_assistant:
            self.io.say("\nAsking the Assistant (loading it the first time takes up to a minute; each answer "
                        "takes about 10 s on a CPU)...")
        result = self.p.explain(tx, use_assistant=use_assistant)
        who = WRITER_NAMES.get(result["source"], result["source"])
        if result["source"] == "template":
            who = "the template (fixed sentences filled in from the numbers above; no model)"
        else:
            who += "; its answer passed every guardrail check"
        self.io.say(f"\nExplanation, written by {who}:")
        for line in result["explanation"].splitlines():
            self.io.say(f"  {line}")
        for attempt in result["attempts"]:
            if attempt["failedChecks"]:
                self.io.say(f"  (The {attempt['source']} answer was rejected by the guardrail: failed "
                            f"{', '.join(attempt['failedChecks'])}; the template is shown instead.)")
        for name, why in result["skippedWriters"].items():
            self.io.say(f"  ({name} is not available on this machine: {why})")

    # -- what-if -----------------------------------------------------------------------

    def what_if(self, address: str) -> None:
        """A hypothetical next transaction, scored against the address's real history before its time."""
        default_time = f"{self.p.snapshot_date}T12:00Z"
        self.io.say("\nWhat-if: score a made-up transaction against this address's real, earlier history.")
        try:
            amount = float(self.io.ask("Amount in BTC (for example 0.05): "))
        except ValueError:
            amount = 0.0
        if not amount > 0:
            self.io.say("The amount must be a positive number of BTC, such as 0.05.")
            return
        direction = self.io.ask("Direction, in or out [out]: ").lower() or "out"
        if direction not in ("in", "out"):
            self.io.say("The direction must be in or out.")
            return
        when = self.io.ask(f"Time in UTC [{default_time}]: ") or default_time
        try:
            datetime.fromisoformat(when.replace("Z", "+00:00"))
        except ValueError:
            self.io.say(f"The time must look like {default_time}.")
            return
        try:
            tx = self.p.score_what_if(address, amount, direction, when)
        except InsufficientHistory as err:
            self.io.say(err.body["message"])
            return
        self.show_transaction(tx, "Hypothetical transaction", use_assistant=False)
        if self.io.ask("\nType a for the Assistant's explanation, or press Enter to go back: ").lower() == "a":
            self.explanation_block(tx, use_assistant=True)


def run_explore() -> int:
    """The `explore` step: load the shipped models once, then the interactive loop."""
    # joblib asks Windows for the physical core count through a tool Windows 11 no longer ships, and prints a
    # traceback in the middle of the screen. Naming a count (about half the logical cores: the physical count on
    # most machines) skips the question; it sets only how many worker threads run, never a result.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 1) // 2)))
    return Explorer(Predictor.load()).run()
