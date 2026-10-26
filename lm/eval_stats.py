"""Confidence intervals and paired tests for the Explorer Assistants' results on every test set.

Serves the statistics campaign (user, 27 Oct 2026: distributions and mean accuracy for every model, for comparison;
part C). Every writer answered the same prompts of each test set (lm.eval_explain), and each answer was judged
fully correct or not, so:

    per set and writer   the fully correct rate with a bootstrap 95% interval over the set's prompts
    pooled               the mean of the set rates (each set weighted equally), its interval from resampling
                         within every set at once
    paired               two writers on identical prompts: the difference in rate with a paired bootstrap interval,
                         and McNemar's exact test on the prompts where exactly one of them was right
    glossary             the share of key facts stated, with its interval, and the paired difference

No model runs here: it reads reports/tables/assistant_eval_<writer>[_<tag>].csv. Writes
reports/tables/stats_writers_summary.csv and stats_writers_paired.csv.

Usage:  python -m lm.eval_stats
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
SETTINGS = json.loads((ROOT / "config" / "settings.json").read_text(encoding="utf-8"))
WRITERS = ("text150m", "qwen3b", "qwen3b_v2", "text150m_v2")
SETS = {"DS7 v1 test": "", "DS10": "_ds10", **{f"random {s}": f"_random_{s}" for s in (
    "real", "synthetic", "hard", "balanced", "crawl_ds1", "crawl_ds2", "crawl_ds10", "close_calls", "extreme", "max_shift")}}


def answers(writer: str, suffix: str) -> pd.Series | None:
    """Fully correct (bool) per explanation prompt id, or None if the writer was not scored on the set."""
    path = TABLES / f"assistant_eval_{writer}{suffix}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame = frame[frame["kind"] != "glossary"]
    return frame.set_index("id")["correct"].astype(str).eq("True")


def interval(values: np.ndarray, level: float) -> tuple[float, float]:
    tail = 100 * (1 - level) / 2
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))


def main() -> int:
    p = SETTINGS["writerStats"]
    level = SETTINGS["significance"]["ciLevel"]
    rng = np.random.default_rng(p["seed"])
    data = {name: {w: a for w in WRITERS if (a := answers(w, suffix)) is not None} for name, suffix in SETS.items()}
    data = {name: by_writer for name, by_writer in data.items() if by_writer}
    summary, paired = [], []
    rates: dict[str, dict[str, tuple[float, np.ndarray]]] = {}  # writer -> set -> (rate, bootstrap rates)
    sizes: dict[str, int] = {}
    for name, by_writer in data.items():
        ids = sorted(set.intersection(*(set(a.index) for a in by_writer.values())))
        sizes[name] = len(ids)
        # One set of resampled prompts per test set, shared by every writer, so every comparison is paired.
        draws = rng.integers(0, len(ids), (p["bootstrapDraws"], len(ids)))
        boots = {}
        for w, a in by_writer.items():
            x = a.loc[ids].to_numpy(dtype=float)
            boots[w] = x[draws].mean(axis=1)
            low, high = interval(boots[w], level)
            summary.append({"test_set": name, "writer": w, "prompts": len(ids), "fully_correct": x.mean(),
                            "ci_low": low, "ci_high": high})
            rates.setdefault(w, {})[name] = (x.mean(), boots[w])
        for a, b in combinations(by_writer, 2):
            xa, xb = by_writer[a].loc[ids].to_numpy(), by_writer[b].loc[ids].to_numpy()
            only_a, only_b = int((xa & ~xb).sum()), int((~xa & xb).sum())
            diff = boots[a] - boots[b]
            low, high = interval(diff, level)
            paired.append({"test_set": name, "writer_a": a, "writer_b": b, "prompts": len(ids),
                           "diff": xa.mean() - xb.mean(), "ci_low": low, "ci_high": high,
                           "only_a_right": only_a, "only_b_right": only_b,
                           "mcnemar_p": binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0})
    # Pooled: the mean over the sets every writer was scored on, each set weighted equally; the sets' bootstrap
    # draws are independent, so the pooled draws are the means of the per-set draws.
    common = [n for n in data if all(n in by_set for by_set in rates.values())]
    for w, by_set in rates.items():
        if common:
            low, high = interval(np.mean([by_set[n][1] for n in common], axis=0), level)
            summary.append({"test_set": f"pooled ({len(common)} sets)", "writer": w, "prompts": sum(sizes[n] for n in common),
                            "fully_correct": float(np.mean([by_set[n][0] for n in common])), "ci_low": low, "ci_high": high})
    glossary = {w: pd.read_csv(f) for w in WRITERS if (f := TABLES / f"assistant_eval_{w}_glossary.csv").exists()}
    if glossary:
        ids = sorted(set.intersection(*(set(g["id"]) for g in glossary.values())))
        gdraws = rng.integers(0, len(ids), (p["bootstrapDraws"], len(ids)))
        gboot = {}
        for w, g in glossary.items():
            x = g.set_index("id").loc[ids, "key_facts"].to_numpy(dtype=float)
            gboot[w] = x[gdraws].mean(axis=1)
            low, high = interval(gboot[w], level)
            summary.append({"test_set": "glossary (key facts)", "writer": w, "prompts": len(ids), "fully_correct": x.mean(),
                            "ci_low": low, "ci_high": high})
        for a, b in combinations(glossary, 2):
            low, high = interval(gboot[a] - gboot[b], level)
            paired.append({"test_set": "glossary (key facts)", "writer_a": a, "writer_b": b, "prompts": len(ids),
                           "diff": float(gboot[a].mean() - gboot[b].mean()), "ci_low": low, "ci_high": high})
    pd.DataFrame(summary).to_csv(TABLES / "stats_writers_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(TABLES / "stats_writers_paired.csv", index=False)
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
