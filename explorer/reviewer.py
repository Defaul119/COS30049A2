"""The reviewer check: two teammates rate flagged and unflagged DS1 transactions, blind (A1 §2.3.2).

Serves plan §11.1 (docs/plan/sections/s10-evaluation.tex) and A1 Objective 4. DS1 has no labels,
so this is the one check that asks a person; its result is reported as judgement, never as
ground truth (D-A2-3).

reviewer-sheet: from the selected method's evaluation-period scores, the `flaggedRows` highest-
scoring flagged transactions and `unflaggedRows` scorable unflagged ones drawn at random (seeded),
shuffled together. The sheet shows each transaction's facts next to the address's usual values,
with empty `worth_a_look` and `note` columns, and no flag or score. Which rows were flagged is
written to a separate key under data/interim/reviewer/ (git-ignored, so a reviewer cannot look it
up in the public repository; the same seed rebuilds it).

reviewer-score: each reviewer's share of flagged and of unflagged rows rated worth a look, and,
with two reviewers, their raw agreement and Cohen's kappa. With one reviewer the kappa is left
empty, and the report says so; a missing rating is never filled in.

Writes reports/reviewer/sheet.csv, data/interim/reviewer/key.csv and reports/tables/reviewer.csv.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .baseline import MODELS_DIR, TABLES_DIR
from .explain import SATS_PER_BTC
from .scores import SCORES_PATH
from .settings import PROJECT_ROOT, SETTINGS
from .snapshot import MANIFEST_PATH
from .split import EVAL

log = logging.getLogger(__name__)

SHEET_PATH = PROJECT_ROOT / "reports" / "reviewer" / "sheet.csv"
KEY_PATH = PROJECT_ROOT / "data" / "interim" / "reviewer" / "key.csv"
RATING = "worth_a_look"
YES, NO = {"y", "yes", "1", "true"}, {"n", "no", "0", "false"}


def selected_method(models_dir: Path = MODELS_DIR) -> str:
    """The method the Explorer uses: settings.json -> explorerMethod, else evaluate's recommendation."""
    if SETTINGS["explorerMethod"]:
        return SETTINGS["explorerMethod"]
    path = models_dir / "selection.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run python -m explorer.cli evaluate first")
    return json.loads(path.read_text(encoding="utf-8"))["chosen"]


def pick_rows(scores: pd.DataFrame, method: str, n_flagged: int, n_unflagged: int, seed: int) -> pd.DataFrame:
    """The top-scoring flagged and a random sample of unflagged scorable evaluation rows, shuffled.

    Returns the chosen rows with `flagged` and `score` columns and a `row` label (R01, R02, ...).
    """
    ev = scores[(scores["period"] == EVAL) & scores["scorable"]]
    flag, score = ev[f"flag_{method}"].astype(bool), ev[f"score_{method}"]
    top = ev[flag].assign(score=score[flag]).sort_values("score", ascending=False, kind="stable").head(n_flagged)
    rest = ev[~flag]
    calm = rest.sample(n=min(n_unflagged, len(rest)), random_state=seed).assign(score=score[~flag])
    if len(top) < n_flagged:
        log.info("Only %d flagged evaluation rows for %s (asked for %d)", len(top), method, n_flagged)
    chosen = pd.concat([top.assign(flagged=True), calm.assign(flagged=False)])
    chosen = chosen.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    width = len(str(len(chosen)))
    return chosen.assign(row=[f"R{i + 1:0{width}d}" for i in range(len(chosen))])


def sheet_columns(chosen: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """What a reviewer sees: the transaction's facts and the address's usual values, in readable units."""
    f = chosen[["row", "address", "txid"]].merge(features, on=["address", "txid"], how="left", validate="one_to_one")
    link = SETTINGS["source"]["explorerTxUrl"]
    return pd.DataFrame({
        "row": f["row"],
        "address": f["address"],
        "time_utc": pd.to_datetime(f["block_time_utc"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M"),
        "direction": f["direction"].map({"in": "received", "out": "sent", "none": "net zero"}),
        "amount_btc": (f["net_sats"].abs() / SATS_PER_BTC).round(8),
        "usual_amount_btc": (f["usual_F1"] / SATS_PER_BTC).round(8),
        "tx_in_previous_7_days": f["F2"],
        "usual_tx_in_7_days": f["usual_F2"].round(1),
        "hours_since_previous": f["F3"].round(1),
        "usual_hours_between": f["usual_F3"].round(1),
        "inputs": f["n_inputs"],
        "outputs": f["n_outputs"],
        "fee_rate_sat_per_vb": (f["fee_sats"] / f["vsize_vb"]).round(1),
        "explorer_link": [link.format(txid=t) for t in f["txid"]],
        RATING: "",
        "note": "",
    })


def run_reviewer_sheet(scores_path: Path = SCORES_PATH, features_path: Path = MANIFEST_PATH.parent / "features.csv",
                       sheet_path: Path = SHEET_PATH, key_path: Path = KEY_PATH) -> int:
    """Step `reviewer-sheet`: a shuffled blind sheet of flagged and unflagged DS1 rows for the reviewer check,
    and its key (kept out of the repository)."""
    p = SETTINGS["reviewer"]
    method = selected_method()
    scores, features = pd.read_csv(scores_path), pd.read_csv(features_path)
    chosen = pick_rows(scores, method, p["flaggedRows"], p["unflaggedRows"], p["seed"])
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_columns(chosen, features).to_csv(sheet_path, index=False)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    chosen[["row", "address", "txid", "flagged", "score"]].assign(method=method).to_csv(key_path, index=False)
    log.info("Wrote %s (%d rows: %d flagged by %s, %d not) and the key %s", sheet_path, len(chosen),
             int(chosen["flagged"].sum()), method, int((~chosen["flagged"]).sum()), key_path)
    log.info("Each reviewer fills %r with yes or no for every row, and returns the file", RATING)
    return 0


def read_ratings(path: Path) -> pd.Series:
    """One reviewer's yes/no ratings as 1/0, by row. Raises if any row is blank or unreadable."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    answers = frame[RATING].str.strip().str.lower()
    bad = frame.loc[~answers.isin(YES | NO), "row"].tolist()
    if bad:
        raise ValueError(f"{path}: rows {bad} need yes or no in {RATING!r}")
    return pd.Series(answers.isin(YES).astype(int).to_numpy(), index=frame["row"], name=path.stem)


def score_ratings(key: pd.DataFrame, ratings: list[pd.Series]) -> pd.DataFrame:
    """Per reviewer: share of flagged and unflagged rows rated worth a look; then agreement and kappa."""
    flagged = key.set_index("row")["flagged"].astype(bool)
    rows = []
    for r in ratings:
        r = r.reindex(flagged.index)
        if r.isna().any():
            raise ValueError(f"{r.name}: rows {r.index[r.isna()].tolist()} are missing")
        rows.append({"reviewer": r.name, "rows": len(r), "flagged_worth_a_look": float(r[flagged].mean()),
                     "unflagged_worth_a_look": float(r[~flagged].mean()),
                     "agreement_with_flags": float((r.astype(bool) == flagged).mean())})
    summary = {"reviewer": "agreement between reviewers", "rows": len(flagged)}
    if len(ratings) == 2:
        a, b = (r.reindex(flagged.index).astype(int) for r in ratings)
        summary |= {"raw_agreement": float((a == b).mean()), "cohen_kappa": float(cohen_kappa_score(a, b))}
    else:
        summary |= {"raw_agreement": np.nan, "cohen_kappa": np.nan,
                    "note": f"{len(ratings)} reviewer(s): kappa needs exactly two"}
    return pd.DataFrame(rows + [summary])


def run_reviewer_score(rating_paths: list[Path], key_path: Path = KEY_PATH, tables_dir: Path = TABLES_DIR) -> int:
    key = pd.read_csv(key_path)
    table = score_ratings(key, [read_ratings(p) for p in rating_paths])
    tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_dir / "reviewer.csv", index=False)
    for row in table.to_dict("records")[:-1]:
        log.info("%s: %.0f%% of flagged and %.0f%% of unflagged rows rated worth a look", row["reviewer"],
                 100 * row["flagged_worth_a_look"], 100 * row["unflagged_worth_a_look"])
    log.info("Cohen's kappa: %s; wrote %s", table.iloc[-1]["cohen_kappa"], tables_dir / "reviewer.csv")
    return 0
