"""DS7: the explanation set both Explorer Assistants are fine-tuned and evaluated on.

Serves plan §6.7 (docs/plan/sections/s05-data.tex, "DS7: the explanation set") and §10.2-10.3.
Each pair is one scored transaction from DS1 or DS2:

* the prompt (explorer.assistant.prompt_messages): the facts of the transaction as the selected
  method scored it, pre-formatted as the template formats them;
* the reply: the template explanation in one of five phrasings, chosen with the seed. Phrasing 0 is
  the template itself; the others say the same things, with the same numbers, in other words, so
  the model learns to explain rather than to memorise one sentence (plan: "five phrasings").

Every reply passes the guardrail's own checks before it is written (a reply that would fail is a
bug, and the build stops), so the data never teaches the model what the validator rejects.

Sampling: at most `maxPairsPerAddress` transactions per address (`ds1PairsPerAddress` for DS1), up to half of them flagged by the
selected method, plus at most `unscoredPerAddress` unscored one; DS1 addresses first, then DS2
addresses in seeded order until `targetPairs`. DS2 histories go through the same feature code as
DS1 (features.build_features), so a DS2 transaction is scored exactly as DS1 would be.

Split by address (plan: "test pairs never come from a training address"): a salted hash of the
address decides train, validation or test, so an address's pairs all land in one split. No
address or txid is written: DS2 ids are already salted hashes, and DS1 addresses are replaced by
the same short hash.

Writes data/processed/ds7/{train,val,test}.jsonl (one {"id", "source", "kind", "phrasing",
"messages"} per line; test rows also carry "template", the ROUGE-L reference),
data/processed/ds7/manifest.json and reports/tables/ds7_summary.csv. Team-written glossary pairs,
if data/processed/ds7/glossary.jsonl exists, are split by question and added.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from . import assistant
from .baseline import MODELS_DIR, TABLES_DIR
from .explain import reasons_for, template_explanation
from .features import FeatureConfig, build_features
from .methods import flags
from .predict import PRECOMPUTED_ONLY, Predictor
from .reviewer import selected_method
from .scores import SCORES_PATH, saved_methods
from .settings import PROJECT_ROOT, SETTINGS
from .snapshot import MANIFEST_PATH, utc_now

log = logging.getLogger(__name__)

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "ds7"
DS2_PATH = PROJECT_ROOT / "data" / "processed" / "corpus" / "ds2.parquet"
SPLITS = ("train", "val", "test")

# -- the five phrasings ------------------------------------------------------------------
# Each takes the template's facts (the opener's method name, the reasons from explain.reasons_for)
# and says them differently. Numbers come only from the reasons' own text, so every phrasing has
# exactly the template's numbers.

OPENERS = [
    ("Flagged by {m}.", "Not flagged by {m}."),
    ("{m} flags this transaction.", "{m} does not flag this transaction."),
    ("Result: flagged by {m}.", "Result: not flagged by {m}."),
    ("This transaction was flagged by {m}.", "This transaction was not flagged by {m}."),
    ("{m} marks this transaction as unusual for its address.", "{m} does not mark this transaction as unusual."),
]
INTROS = [
    ("What stands out compared with this address's own history:", "Largest differences from this address's usual activity:"),
    ("Compared with this address's own past:", "The biggest differences from what this address usually does:"),
    ("Why:", "Closest to standing out:"),
    ("Here is what differs from the address's usual pattern:", "No single feature is far from usual; these differ most:"),
    ("The reasons, strongest first:", "The two largest differences from usual:"),
]
# The second intro of each pair is used when no feature reaches the contributing level. A method that
# scores the whole pattern (e.g. Isolation Forest) can still flag such a transaction, so these intros
# never say whether it was flagged; the opener does.
UNSCORED = [
    "This transaction is not scored: the address does not yet have enough earlier history to compare it with.",
    "Not scored yet. The address has too little earlier history to say what is usual for it.",
    "There is no score for this transaction, because the address's earlier history is too short to compare against.",
    "This one cannot be scored: there is not yet enough earlier activity from the address to compare it with.",
    "Unscored: an unusualness score needs more earlier transactions from this address than it has so far.",
]


def phrase(tx: dict[str, Any], k: int) -> str:
    """The explanation of `tx` in phrasing k (0 is the template itself)."""
    if k == 0:
        return template_explanation(tx)
    if not tx.get("scored", True):
        return UNSCORED[k]
    method = tx["method"]["name"]
    reasons, fallback = reasons_for(tx["features"])
    lines = [OPENERS[k][0 if tx["flagged"] else 1].format(m=method)]
    if tx.get("model_reason"):
        lines.append(f"The model's reason: {tx['model_reason']}." if k % 2 else f"According to the model, {tx['model_reason']}.")
    if reasons:
        lines.append(INTROS[k][1 if fallback else 0])
        for r in reasons:
            lines.append(f"- {r['headline']}. {r['detail']}" if k in (1, 3) else f"- {r['headline']}: {r['detail']}")
    if tx["flagged"]:
        lines.append(SETTINGS["limitationLine"])
    return "\n".join(lines)


# -- sources -----------------------------------------------------------------------------


def short_hash(value: str, salt: str, chars: int = 12) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:chars]


def split_of(address: str, params: dict[str, Any]) -> str:
    """train, val or test, decided by a seeded hash of the address so all its pairs share a split."""
    u = int(short_hash(address, str(params["seed"]), 8), 16) / 16 ** 8
    return "train" if u < params["trainShare"] else "val" if u < params["trainShare"] + params["valShare"] else "test"


def ds2_transactions(path: Path, addresses: list[str] | None = None) -> pd.DataFrame:
    """DS2's compact rows in DS1's transaction-table layout (explorer.transactions.COLUMNS)."""
    rows = pd.read_parquet(path)
    if addresses is not None:
        rows = rows[rows["addr_id"].isin(addresses)]
    net = rows["received_sats"] - rows["sent_sats"]
    out = pd.DataFrame({
        "address": rows["addr_id"], "role": "ds2", "txid": rows["txid_hash"], "block_height": rows["block_height"],
        "block_time_utc": rows["block_time_utc"], "received_sats": rows["received_sats"],
        "sent_sats": rows["sent_sats"], "net_sats": net,
        "direction": net.gt(0).map({True: "in", False: "out"}).where(net != 0, "none"),
        "n_inputs": rows["n_inputs"], "n_outputs": rows["n_outputs"], "fee_sats": rows["fee_sats"],
        "vsize_vb": rows["vsize_vb"],
    })
    return out.sort_values(["address", "block_time_utc", "txid"], kind="stable").reset_index(drop=True)


def score_frame(features: pd.DataFrame, methods: dict[str, Any]) -> pd.DataFrame:
    """scores.csv's layout for any feature table: score_<m> and flag_<m> for each method that can score it."""
    out = features[["address", "txid"]].copy()
    for name, method in methods.items():
        out[f"score_{name}"] = method.score(features)
        out[f"flag_{name}"] = flags(method, features)
    return out


def pick(rows: pd.DataFrame, flag_col: str, params: dict[str, Any], rng: random.Random) -> list[int]:
    """Row positions to explain for one address: flagged, then unflagged scorable, then unscored."""
    cap = params["maxPairsPerAddress"]
    scored = rows[rows["scorable"]]
    flagged = list(scored.index[scored[flag_col]])
    calm = list(scored.index[~scored[flag_col]])
    unscored = list(rows.index[~rows["scorable"]])
    for group in (flagged, calm, unscored):
        rng.shuffle(group)
    chosen = flagged[: cap // 2]
    chosen += calm[: cap - len(chosen) - min(params["unscoredPerAddress"], len(unscored))]
    chosen += unscored[: params["unscoredPerAddress"]]
    return chosen[:cap]


def pairs_from(source: str, predictor: Predictor, scores: pd.DataFrame, addresses: list[str],
               params: dict[str, Any], rng: random.Random) -> Iterator[dict[str, Any]]:
    """One record per chosen transaction of each address, in phrasing chosen with the seed."""
    method = predictor.selected
    names = [c[len("score_"):] for c in scores.columns if c.startswith("score_")]
    scores = scores.set_index(["address", "txid"])
    features = predictor.features.reset_index(drop=True)
    keys = pd.MultiIndex.from_frame(features[["address", "txid"]])
    features = features.assign(selected_flag=scores[f"flag_{method}"].reindex(keys).astype(bool).to_numpy())
    for address in addresses:
        for i in pick(features[features["address"] == address], "selected_flag", params, rng):
            row = features.loc[i]
            s = scores.loc[(address, row["txid"])]
            tx = predictor.detail(row, {n: s[f"score_{n}"] for n in names}, {n: bool(s[f"flag_{n}"]) for n in names})
            k = rng.randrange(params["phrasings"])
            reply = phrase(tx, k)
            messages = assistant.prompt_messages(tx)
            checks = assistant.check(reply, messages, assistant.summary(tx))
            if not checks["passed"]:
                raise ValueError(f"DS7 reply fails its own guardrail ({checks}): {reply!r}")
            kind = "unscored" if not tx["scored"] else "flagged" if tx["flagged"] else "not_flagged"
            yield {"source": source, "group": short_hash(address, str(params["seed"])), "split": split_of(address, params),
                   "kind": kind, "phrasing": k, "template": template_explanation(tx),
                   "messages": messages + [{"role": "assistant", "content": reply}]}


def glossary_pairs(path: Path, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Team-written questions and answers ({"question", "answer"} per line), split by question."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                q = json.loads(line)
                if assistant.banned_words(q["answer"]):
                    raise ValueError(f"glossary answer uses a banned word: {q['question']!r}")
                yield {"source": "glossary", "group": short_hash(q["question"], str(params["seed"])),
                       "split": split_of(q["question"], params), "kind": "glossary", "phrasing": None,
                       "template": q["answer"],
                       "messages": [{"role": "system", "content": assistant.SYSTEM_PROMPT},
                                    {"role": "user", "content": q["question"]},
                                    {"role": "assistant", "content": q["answer"]}]}


def run_build_ds7(out_dir: Path = OUT_DIR, ds2_path: Path = DS2_PATH, tables_dir: Path = TABLES_DIR) -> int:
    params = SETTINGS["ds7"]
    rng = random.Random(params["seed"])
    method = selected_method()
    methods = {n: m for n, m in saved_methods().items() if n not in PRECOMPUTED_ONLY}
    processed = MANIFEST_PATH.parent
    ds1 = Predictor(pd.read_csv(processed / "features.csv"), pd.read_csv(processed / "transactions.csv"),
                    pd.read_csv(SCORES_PATH), saved_methods(), method, "")
    ds1_params = {**params, "maxPairsPerAddress": params["ds1PairsPerAddress"]}
    records = list(pairs_from("DS1", ds1, ds1.scores, sorted(ds1.features["address"].unique()), ds1_params, rng))
    log.info("DS1: %d pairs from %d addresses (method %s)", len(records), ds1.features["address"].nunique(), method)

    if ds2_path.exists():
        ids = sorted(pd.read_parquet(ds2_path, columns=["addr_id"])["addr_id"].unique())
        rng.shuffle(ids)
        # Enough addresses to reach the target at the per-address cap, with a margin for short histories.
        wanted = ids[: max(0, params["targetPairs"] - len(records)) // params["maxPairsPerAddress"] * 2]
        transactions = ds2_transactions(ds2_path, wanted)
        features = build_features(transactions, FeatureConfig.refined())
        ds2 = Predictor(features, transactions, score_frame(features, methods), methods, method, "")
        for record in pairs_from("DS2", ds2, ds2.scores, wanted, params, rng):
            if len(records) >= params["targetPairs"]:
                break
            records.append(record)
        log.info("DS2: pairs up to %d in total, from %d candidate addresses", len(records), len(wanted))
    else:
        log.info("No %s yet: DS7 holds DS1 pairs only; rebuild when the DS2 crawl has finished", ds2_path)

    glossary = out_dir / "glossary.jsonl"
    if glossary.exists():
        records += list(glossary_pairs(glossary, params))

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = []
    for split in SPLITS:
        part = [r for r in records if r["split"] == split]
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for n, r in enumerate(part):
                row = {"id": f"{split}-{n:05d}", "source": r["source"], "kind": r["kind"], "phrasing": r["phrasing"],
                       "messages": r["messages"]}
                if split == "test":
                    row["template"] = r["template"]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        frame = pd.DataFrame(part, columns=["source", "kind", "group"])
        for (source, kind), g in frame.groupby(["source", "kind"]):
            counts.append({"split": split, "source": source, "kind": kind, "pairs": len(g), "groups": g["group"].nunique()})
    summary = pd.DataFrame(counts)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "ds7_summary.csv", index=False)
    manifest = {
        "dataset": "DS7: Explorer explanation pairs (derived from DS1 and DS2 pipeline outputs)",
        "built_utc": utc_now(), "method": method, "settings": params, "pairs": len(records),
        "includes_ds2": ds2_path.exists(), "glossary": glossary.exists(),
        "rule": "split by a seeded hash of the source address; every reply passes explorer.assistant.check",
        "files": [{"path": f"data/processed/ds7/{s}.jsonl", "pairs": int(summary.loc[summary["split"] == s, "pairs"].sum()),
                   "sha256": hashlib.sha256((out_dir / f"{s}.jsonl").read_bytes()).hexdigest()} for s in SPLITS],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote %d pairs to %s: %s", len(records), out_dir,
             summary.groupby("split")["pairs"].sum().to_dict())
    return 0
