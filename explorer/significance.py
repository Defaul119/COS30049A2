"""How sure are the DS1 recall numbers? Address-level bootstrap intervals and paired differences.

Serves plan §11.4 (the selection rule) and report criterion "model-selection justification": the rule
picks the highest mean injection recall, and within tieMarginPoints the easier method to explain. Whether
a gap between two methods is a real difference or noise decides whether that tie-break is fair, and the
seed-to-seed SD in eval_injection.csv measures only one source of noise.

Unit of resampling: the ADDRESS. Plants in one address share its history (and its behaviour change after
the cut, CR-03), so they are not independent; resampling plants would give intervals that are too narrow.
Each of `resamples` draws picks the 25 addresses with replacement, keeps every reporting plant on them
(all five reporting seeds), and recomputes every method's recall per pattern and their mean on that same
draw, so differences between methods are paired. Intervals are percentile intervals at `ciLevel`; the
two-sided p-value of a difference is twice the smaller share of draws on either side of zero.

Writes reports/tables/eval_plant_hits.csv (every reporting plant, every method: flagged or not) and
reports/tables/eval_significance.csv (per method: recalls with intervals, and the paired difference from
settings.json -> explorerMethod).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .baseline import TABLES_DIR, load_copies
from .methods import Method, flags
from .scores import saved_methods
from .settings import SETTINGS
from .split import EVAL

log = logging.getLogger(__name__)

# Per pattern: which per-plant summary the A1 measures use (methods.plant_metrics): "max" = at least one of the
# plant's transactions flagged; "mean" = the share of them flagged (burst per transaction, A1's measure).
MEASURES = {"recall_spike": ("spike", "max"), "recall_burst_tx": ("burst", "mean"),
            "recall_burst_event": ("burst", "max"), "recall_dormant": ("dormant", "max")}
MEAN_OF = ("recall_spike", "recall_burst_tx", "recall_dormant")  # the selection rule's mean recall


def plant_hits(methods: dict[str, Method], copies: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, plant): whether any, and what share, of the plant's transactions were flagged."""
    planted = copies[(copies["period"] == EVAL) & (copies["is_synthetic"] == 1)]
    out = []
    for name, method in methods.items():
        hit = pd.Series(flags(method, planted), index=planted.index)
        g = (planted.assign(hit=hit).groupby(["seed", "pattern", "plant_id", "address"])["hit"]
             .agg(["max", "mean"]).reset_index())
        out.append(g.assign(method=name))
    return pd.concat(out, ignore_index=True)


def bootstrap_recalls(hits: pd.DataFrame, weights: np.ndarray, addresses: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Per method and measure: the point estimate (index 0) followed by one value per bootstrap draw.

    `weights` is draws x addresses: how many times each address was drawn. A measure is the pooled share
    over the drawn plants (sum of hits / number of plants), exactly as plant_metrics pools the full set.
    """
    where = {a: i for i, a in enumerate(addresses)}
    ones = np.vstack([np.ones(len(addresses)), weights])  # row 0: every address once, the point estimate
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, part in hits.groupby("method"):
        out[name] = {}
        for measure, (pattern, column) in MEASURES.items():
            rows = part[part["pattern"] == pattern]
            idx = rows["address"].map(where).to_numpy()
            hit_sum, count = np.zeros(len(addresses)), np.zeros(len(addresses))
            np.add.at(hit_sum, idx, rows[column].to_numpy(dtype="float64"))
            np.add.at(count, idx, 1.0)
            n = ones @ count
            out[name][measure] = np.where(n > 0, (ones @ hit_sum) / np.where(n > 0, n, 1.0), np.nan)
        out[name]["mean_recall"] = np.mean([out[name][m] for m in MEAN_OF], axis=0)
    return out


def interval(values: np.ndarray, level: float) -> tuple[float, float]:
    tail = 100 * (1 - level) / 2
    return float(np.nanpercentile(values, tail)), float(np.nanpercentile(values, 100 - tail))


def two_sided_p(diffs: np.ndarray) -> float:
    """Twice the smaller share of bootstrap differences on either side of zero (capped at 1)."""
    d = diffs[~np.isnan(diffs)]
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))) if len(d) else float("nan")


def significance_table(boot: dict[str, dict[str, np.ndarray]], reference: str, level: float,
                       margin_points: float) -> pd.DataFrame:
    """Each method's measures with bootstrap intervals, and its difference to the reference method (inside the
    tie margin or not)."""
    rows = []
    ref = boot[reference]["mean_recall"]
    for name, measures in boot.items():
        row: dict = {"method": name, "reference": reference}
        for measure, values in measures.items():
            row[measure] = float(values[0])
            row[f"{measure}_ci_low"], row[f"{measure}_ci_high"] = interval(values[1:], level)
        diff = measures["mean_recall"] - ref
        row["diff_mean_recall"] = float(diff[0])
        row["diff_ci_low"], row["diff_ci_high"] = interval(diff[1:], level)
        row["p_value"] = two_sided_p(diff[1:]) if name != reference else np.nan
        # Does the interval of the difference reach beyond the selection rule's tie margin on either side?
        row["ci_wider_than_tie_margin"] = (row["diff_ci_high"] - row["diff_ci_low"]) / 2 > margin_points / 100
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_recall", ascending=False)


def run_significance() -> int:
    """Step `significance`: address-level bootstrap intervals for every method's recall and for its difference to
    the explorerMethod; writes eval_significance.csv and eval_plant_hits.csv."""
    s = SETTINGS["significance"]
    copies = load_copies("refined", "reporting")
    methods = saved_methods()
    reference = SETTINGS["explorerMethod"]
    hits = plant_hits(methods, copies)
    addresses = sorted(hits["address"].unique())
    rng = np.random.default_rng(s["seed"])
    weights = rng.multinomial(len(addresses), np.full(len(addresses), 1 / len(addresses)), size=s["bootstrapResamples"])
    table = significance_table(bootstrap_recalls(hits, weights, addresses), reference, s["ciLevel"],
                               SETTINGS["evaluation"]["tieMarginPoints"])

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    hits.to_csv(TABLES_DIR / "eval_plant_hits.csv", index=False)
    table.to_csv(TABLES_DIR / "eval_significance.csv", index=False)
    log.info("%d plants on %d addresses, %d address-level bootstrap draws, %.0f%% intervals; reference %s",
             hits["plant_id"].nunique(), len(addresses), s["bootstrapResamples"], 100 * s["ciLevel"], reference)
    for r in table.itertuples():
        log.info("%-20s mean recall %.3f [%.3f, %.3f]; minus %s %+.3f [%+.3f, %+.3f], p = %s", r.method, r.mean_recall,
                 r.mean_recall_ci_low, r.mean_recall_ci_high, reference, r.diff_mean_recall, r.diff_ci_low, r.diff_ci_high,
                 "-" if np.isnan(r.p_value) else f"{r.p_value:.3f}")
    log.info("Wrote %s and %s", TABLES_DIR / "eval_significance.csv", TABLES_DIR / "eval_plant_hits.csv")
    return 0
