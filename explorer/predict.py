"""Prediction: everything the Explorer (and A3's back end) asks of the trained models.

Serves plan §12.1 and the brief's "use your model for prediction". Artefacts are loaded once by
`Predictor.load()`; every method returns plain dictionaries in the JSON shapes of the A1 prototype
(reference/a1_prototype/types.ts), so A3's FastAPI layer only has to serialise them.

    score_address(address)        timeline, totals and every transaction's detail: scores from
                                  every method, the selected method's flag, the features and the
                                  comparison basis, the template explanation and the limitation line
    score_what_if(address, ...)   a hypothetical next transaction, scored against the real history
                                  by every method that can score a new transaction
    classify_heist(...)           the probability that an address-day resembles the labelled
                                  ransomware rows of BitcoinHeist
    classify_elliptic(rows)       the probability of Elliptic's own "illicit" class for feature rows
    txgpt_surprisal(address)      surprisal per transaction and per field
    explain(tx, use_assistant)    one transaction's explanation: the template, or the first Assistant
                                  answer that passes the guardrail (plan §10.3; settings -> assistant)

score_address and score_what_if always carry the template explanation, which is instant; the Assistant
runs only when one transaction's explanation is asked for, as the Explorer's detail panel does.

Typed addresses outside the snapshot raise Unsupported, and addresses with too little history raise
InsufficientHistory: A1's planned 404 and 422 responses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .assistant import Generator
from .assistant import explain as guarded_explanation
from .baseline import MODELS_DIR, TABLES_DIR
from .classify import INTERIM_DIR as CLASSIFY_DIR
from .explain import FEATURE_DESCRIPTIONS, FEATURE_NAMES, METHOD_NAMES, SATS_PER_BTC, check_wording, template_explanation
from .features import FEATURE_IDS, FeatureConfig, build_features
from .labelled import FEATURE_SETS, HEIST_DIR
from .methods import Method, flags
from .scores import SCORES_PATH, saved_methods
from .settings import PROJECT_ROOT, SETTINGS
from .snapshot import MANIFEST_PATH

# Methods that need a precomputed score for every transaction and so cannot score a new one yet.
PRECOMPUTED_ONLY = {"txgpt_method"}


class Unsupported(LookupError):
    """The address is not in the snapshot (A1's 404)."""


class InsufficientHistory(ValueError):
    """Too little earlier history to score (A1's 422); `body` is the InsufficientHistoryBody."""

    def __init__(self, message: str, body: dict[str, Any]):
        super().__init__(message)
        self.body = body


def iso(unix_seconds: int) -> str:
    return datetime.fromtimestamp(int(unix_seconds), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def day(unix_seconds: int) -> str:
    return datetime.fromtimestamp(int(unix_seconds), tz=timezone.utc).strftime("%Y-%m-%d")


def finite(value: Any) -> float | None:
    """A JSON-safe number: NaN and missing become None."""
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else value


def load_writers() -> tuple[list[tuple[str, Generator]], dict[str, str]]:
    """The Assistant's writers this machine can run, in settings' writerOrder, and why any others were skipped.

    A writer whose weights are missing (they are not in git: README) is skipped, and the template alone writes.
    """
    from lm import writers  # torch is only needed once the Assistant is asked for

    cfg = SETTINGS["assistant"]
    files = {"text150m": [cfg["text150mCheckpoint"]]}
    loaded, skipped = [], {}
    for name in cfg["writerOrder"]:
        reason = writers.available(files[name], PROJECT_ROOT)
        if reason:
            skipped[name] = reason
        else:
            loaded.append((name, writers.text150m_generator(PROJECT_ROOT / cfg["text150mCheckpoint"], cfg["maxNewTokens"])))
    return loaded, skipped


@dataclass
class Predictor:
    features: pd.DataFrame
    transactions: pd.DataFrame
    scores: pd.DataFrame
    methods: dict[str, Method]
    selected: str
    snapshot_date: str
    # The Assistant's writers, loaded on the first explain(..., use_assistant=True); None until then.
    writers: list[tuple[str, Generator]] | None = None
    skipped_writers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls) -> Predictor:
        processed = MANIFEST_PATH.parent
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        selected = SETTINGS.get("explorerMethod")
        if not selected and (MODELS_DIR / "selection.json").exists():
            selected = json.loads((MODELS_DIR / "selection.json").read_text(encoding="utf-8"))["chosen"]
        return cls(pd.read_csv(processed / "features.csv"), pd.read_csv(processed / "transactions.csv"),
                   pd.read_csv(SCORES_PATH), saved_methods(), selected or "baseline",
                   manifest["selection_rule"]["recorded_utc"][:10])

    # -- shared pieces -----------------------------------------------------------------

    def method_info(self, name: str) -> dict[str, Any]:
        return {"id": name, "name": METHOD_NAMES.get(name, name), "flagAbsZ": self.flag_abs_z(),
                "contributingAbsZ": SETTINGS["baseline"]["contributingAbsZ"],
                "note": "Flags mean unusual for this address's own history, nothing more."}

    def flag_abs_z(self) -> float:
        method = self.methods.get("baseline")
        return float(method.threshold) if method else float(SETTINGS["baseline"]["flagAbsZ"])

    def snapshot(self) -> dict[str, Any]:
        src = SETTINGS["source"]
        return {"snapshotDate": self.snapshot_date, "source": src["name"], "sourceUrl": src["apiBase"],
                "illustrative": False, "dayTimeZone": "UTC", "addressCount": int(self.features["address"].nunique()),
                "method": self.method_info(self.selected)}

    def requirement(self) -> dict[str, Any]:
        return {"minTx": SETTINGS["features"]["minHistoryTx"], "minSpanDays": SETTINGS["features"]["minHistorySpanDays"]}

    def history_of(self, address: str) -> pd.DataFrame:
        rows = self.features[self.features["address"] == address]
        if rows.empty:
            examples = sorted(self.features["address"].unique())[:3]
            raise Unsupported(f"{address} is not in the snapshot; try one of {examples}")
        return rows.sort_values(["block_time_utc", "txid"])

    def feature_details(self, row: pd.Series) -> list[dict[str, Any]]:
        out = []
        for fid in FEATURE_IDS:
            z = finite(row[f"z_{fid}"])
            value, usual = finite(row[fid]), finite(row[f"usual_{fid}"])
            if fid == "F1":  # stored as log10(1 + sats); shown in BTC
                value = None if value is None else (10 ** value - 1) / SATS_PER_BTC
                usual = None if usual is None else usual / SATS_PER_BTC
            role = ("no-comparison" if z is None else "flag" if abs(z) >= self.flag_abs_z() else
                    "contributing" if abs(z) >= SETTINGS["baseline"]["contributingAbsZ"] else "within-usual")
            out.append({"id": fid, "name": FEATURE_NAMES[fid], "description": FEATURE_DESCRIPTIONS[fid],
                        "value": value, "usual": usual, "z": z, "role": role})
        return out

    def comparison(self, row: pd.Series) -> list[dict[str, Any]]:
        basis = []
        for prefix, fids, label in (("size_window", ["F1", "F4"], "Earlier transactions in the same direction"),
                                    ("timing_window", ["F2", "F3"], "All earlier transactions")):
            count = finite(row[f"{prefix}_count"])
            if count is None:
                continue
            extended = bool(row[f"{prefix}_extended"])
            basis.append({
                "features": fids, "label": label,
                "description": (f"{int(count)} earlier transactions: the previous {SETTINGS['features']['comparisonWindowDays']} days"
                                + (f", extended back to reach {SETTINGS['features']['comparisonMinTx']}" if extended else "")),
                "txCount": int(count), "start": iso(row[f"{prefix}_start"]), "end": iso(row[f"{prefix}_end"]),
            })
        return basis

    def core(self, row: pd.Series) -> dict[str, Any]:
        return {"txid": row["txid"], "address": row["address"], "time": iso(row["block_time_utc"]),
                "blockHeight": int(row["block_height"]), "direction": row["direction"],
                "receivedSats": int(row["received_sats"]), "sentSats": int(row["sent_sats"]),
                "netSats": int(row["net_sats"]),
                "sourceUrl": SETTINGS["source"]["explorerTxUrl"].replace("{txid}", row["txid"])
                if not str(row["txid"]).startswith("what-if") else None}

    def detail(self, row: pd.Series, method_scores: dict[str, float], method_flags: dict[str, bool]) -> dict[str, Any]:
        out = {**self.core(row), "scored": bool(row["scorable"]), "flagged": bool(method_flags.get(self.selected, False)),
               "score": finite(method_scores.get(self.selected)), "method": self.method_info(self.selected),
               "features": self.feature_details(row), "comparison": self.comparison(row),
               "scores": {k: finite(v) for k, v in method_scores.items()}, "flags": method_flags,
               "limitation": SETTINGS["limitationLine"]}
        out["explanation"] = template_explanation(out)
        return out

    # -- the prediction API ------------------------------------------------------------

    def score_address(self, address: str) -> dict[str, Any]:
        rows = self.history_of(address)
        scored = rows[rows["scorable"]]
        if scored.empty:
            raise InsufficientHistory(f"{address} has too little history to score", {
                "status": "insufficient_history", "message": "This address does not have enough earlier history to compare with.",
                "history": {"earlierTx": len(rows), "spanDays": float(rows["history_span_days"].max()),
                            "required": self.requirement()}, "firstScoredDate": None})
        s = self.scores[self.scores["address"] == address].set_index("txid")
        names = [c[len("score_"):] for c in s.columns if c.startswith("score_")]
        details = []
        for _, row in rows.iterrows():
            scores = {n: s.at[row["txid"], f"score_{n}"] for n in names}
            flagged = {n: bool(s.at[row["txid"], f"flag_{n}"]) for n in names}
            details.append(self.detail(row, scores, flagged))
        days: dict[str, dict[str, Any]] = {}
        for d in details:
            entry = days.setdefault(d["time"][:10], {"date": d["time"][:10], "txCount": 0, "flaggedCount": 0})
            entry["txCount"] += 1
            entry["flaggedCount"] += int(d["flagged"])
        span = float(rows["history_span_days"].max())
        return {
            "address": address, "snapshot": self.snapshot(),
            "range": {"start": details[0]["time"], "end": details[-1]["time"]},
            "days": list(days.values()),
            "totals": {"txCount": len(details), "scoredCount": int(rows["scorable"].sum()),
                       "flaggedCount": sum(d["flagged"] for d in details),
                       "flaggedDays": sum(1 for d in days.values() if d["flaggedCount"])},
            "history": {"status": "sufficient", "txCount": len(details), "spanDays": span, "required": self.requirement(),
                        "firstScoredDate": day(scored["block_time_utc"].min())},
            "transactions": details,
        }

    def score_what_if(self, address: str, amount_btc: float, direction: str, time_utc: str) -> dict[str, Any]:
        """Score a hypothetical transaction against the address's real, strictly earlier history."""
        if direction not in ("in", "out"):
            raise ValueError("direction must be 'in' or 'out'")
        self.history_of(address)
        real = self.transactions[self.transactions["address"] == address]
        t = int(datetime.fromisoformat(time_utc.replace("Z", "+00:00")).timestamp())
        earlier = real[real["block_time_utc"] < t]
        if earlier.empty:
            raise InsufficientHistory("the hypothetical transaction is before the address's first one", {
                "status": "insufficient_history", "message": "There is no earlier history at that time.",
                "history": {"earlierTx": 0, "spanDays": 0.0, "required": self.requirement()}, "firstScoredDate": None})
        sats = int(round(amount_btc * SATS_PER_BTC))
        received, sent = (sats, 0) if direction == "in" else (0, sats)
        shape = earlier[["n_inputs", "n_outputs", "fee_sats", "vsize_vb"]].median().round().astype(int)
        hypothetical = {"address": address, "role": real["role"].iloc[0], "txid": "what-if", "block_height": int(earlier["block_height"].max()),
                        "block_time_utc": t, "received_sats": received, "sent_sats": sent, "net_sats": received - sent,
                        "direction": direction, **shape.to_dict()}
        frame = build_features(pd.concat([earlier, pd.DataFrame([hypothetical])], ignore_index=True), FeatureConfig.refined())
        row = frame[frame["txid"] == "what-if"].assign(period="eval")
        usable = {n: m for n, m in self.methods.items() if n not in PRECOMPUTED_ONLY}
        scores = {n: float(m.score(row)[0]) for n, m in usable.items()}
        flagged = {n: bool(flags(m, row)[0]) for n, m in usable.items()}
        result = self.detail(row.iloc[0], scores, flagged)
        result["note"] = "A hypothetical transaction, compared with the address's real history before that time."
        return result

    def classify_heist(self, length: float, weight: float, count: float, looped: float, neighbors: float,
                       income: float) -> dict[str, Any]:
        saved = joblib.load(MODELS_DIR / "heist_best.joblib")
        decisions = json.loads((HEIST_DIR / "processing.json").read_text(encoding="utf-8"))
        raw = {"length": length, "weight": weight, "count": count, "looped": looped, "neighbors": neighbors, "income": income}
        x = {}
        for name, value in raw.items():
            how = decisions["transform"][name]
            x[{"log10": f"log10_{name}", "log1p": f"log1p_{name}"}.get(how, name)] = (
                np.log10(value) if how == "log10" else np.log1p(value) if how == "log1p" else value)
        frame = pd.DataFrame([x])[saved["features"]]
        p = float(saved["model"].predict_proba(frame)[0, 1])
        text = (f"Probability {p:.3f} that this address-day resembles BitcoinHeist's labelled ransomware rows "
                f"(threshold {saved['threshold']:.3f}; model {saved['result']['model']}, trained on years up to "
                f"{decisions['temporal_last_training_year']}). This compares graph statistics with a labelled "
                "dataset; it says nothing about who controls an address.")
        return {"probability": p, "resembles_labelled_positive_rows": p >= saved["threshold"], "explanation": text}

    def classify_elliptic(self, rows: pd.DataFrame) -> list[dict[str, Any]]:
        """Elliptic's best model (kept on this machine only: the dataset is CC BY-NC-ND)."""
        results = pd.read_csv(TABLES_DIR / "elliptic_results.csv")
        best = results.sort_values("val_f1", ascending=False).iloc[0]
        model = joblib.load(CLASSIFY_DIR / "elliptic" / f"{best['feature_set']}_{best['model']}.joblib")
        p = model.predict_proba(rows[FEATURE_SETS[best["feature_set"]]])[:, 1]
        return [{"txId": int(t) if "txId" in rows else None, "probability": float(q),
                 "label": "illicit (Elliptic's class)" if q >= best["threshold"] else "licit (Elliptic's class)"}
                for t, q in zip(rows.get("txId", [None] * len(p)), p)]

    def transaction(self, address: str, txid: str) -> dict[str, Any]:
        """One transaction's detail, as score_address gives it."""
        for tx in self.score_address(address)["transactions"]:
            if tx["txid"] == txid:
                return tx
        raise Unsupported(f"{txid} is not a transaction of {address} in the snapshot")

    def explain(self, tx: dict[str, Any], use_assistant: bool = False) -> dict[str, Any]:
        """The explanation for one transaction detail (from score_address, transaction or score_what_if).

        Without use_assistant (the default, plan risk 26) it is the template. With it, each writer this
        machine can run tries in turn and the first answer that passes every guardrail check is shown,
        else the template. A rejected answer is never returned, only which checks it failed.
        """
        if use_assistant and self.writers is None:
            self.writers, self.skipped_writers = load_writers()
        result = guarded_explanation(tx, self.writers if use_assistant else [])
        return {"txid": tx["txid"], "flagged": tx["flagged"], "explanation": result["text"], "source": result["source"],
                "attempts": [{"source": a["source"],
                              "failedChecks": [k for k, ok in a["checks"].items() if k != "passed" and not ok]}
                             for a in result["attempts"]],
                "skippedWriters": self.skipped_writers if use_assistant else {},
                "limitation": SETTINGS["limitationLine"]}

    def txgpt_surprisal(self, address: str) -> list[dict[str, Any]]:
        self.history_of(address)
        path = MANIFEST_PATH.parent / "txgpt_scores.csv"
        frame = pd.read_csv(path)
        return frame[frame["address"] == address].to_dict("records")


def check_outputs(obj: Any) -> None:
    """Every user-facing string in a prediction passes the wording rule (A1 NFR-01)."""
    if isinstance(obj, str):
        check_wording(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k not in ("label",):  # Elliptic's own class names are quoted as the dataset's labels
                check_outputs(v)
    elif isinstance(obj, list):
        for v in obj:
            check_outputs(v)

