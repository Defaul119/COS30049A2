"""The strict gate after every training stage, and the stage log reports/tables/stages.csv.

Serves docs/TRAINING_PLAN.md sections 2, 3 and 6 (and report criterion "model-selection justification"): every
model is retrained one stage at a time, and a stage is kept only if it passes rules fixed before any result:

1. the primary metric rises against the last accepted stage, and the bootstrap interval of the paired
   difference lies above 0;
2. no case (plant pattern, writer test case) falls by more than its own seed spread at M0;
3. the writer's guardrail pass rate and number faithfulness do not fall; a scoring method's flag rate stays
   inside the sanity limit (settings.json -> sanity.maxFlaggedShare), and its largest per-address share does not
   rise above both that limit and its value before the stage (at M0 some addresses are over the limit already,
   for every method: their behaviour changed after the 70/30 cut, eval_sanity_addresses.csv);
4. training loss and dev metric agree (no dev gain with a rising validation loss).

Differences are paired: the previous and the new candidate are scored on the same units (plants, prompts, rows),
and each bootstrap draw resamples GROUPS of units (an address's plants share its history, so they are not
independent; see explorer/significance.py). For 0/1 outcomes (a writer's answer fully correct or not) McNemar's
exact test is reported beside the interval.

Before any test-tier or final-tier evaluation, `check_frozen()` compares every frozen file with the SHA-256
recorded by `freeze()` at milestone M0; a mismatch stops the run, so no test set changes during the campaign.

Usage:  python -m explorer.cli stages-freeze      # once, at M0: record the frozen files' SHA-256
        python -m explorer.cli stage-dev --stage A0   # dev tier of every saved DS1 scoring method
        python -m explorer.cli stage-compare --before M0 --after A0 --stage A0   # the gate -> stages.csv
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .settings import PROJECT_ROOT, SETTINGS

log = logging.getLogger(__name__)

TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
COLUMNS = ["time_utc", "commit", "family", "model", "stage", "tier", "metric", "value", "ci_low", "ci_high",
           "previous_stage", "diff_previous", "diff_previous_ci_low", "diff_previous_ci_high", "diff_m0",
           "diff_m0_ci_low", "diff_m0_ci_high", "mcnemar_p", "accepted", "reasons", "seeds"]
# Shares read back from a CSV can differ from the same share computed again in the last bit (0.37735849056603776 is
# written as 0.3773584905660377), so "rose" means rose by more than this.
SHARE_TOLERANCE = 1e-9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen_paths() -> list[Path]:
    """Every test-tier and final-tier file (settings.json -> stages.frozenFiles; glob patterns allowed)."""
    paths: list[Path] = []
    for pattern in SETTINGS["stages"]["frozenFiles"]:
        hits = sorted(PROJECT_ROOT.glob(pattern))
        if not hits:
            raise FileNotFoundError(f"frozen file pattern {pattern!r} matches nothing")
        paths.extend(hits)
    return paths


def freeze(manifest: Path | None = None) -> dict[str, str]:
    """Record the SHA-256 of every frozen file (run once, at M0). Refuses to overwrite an existing manifest."""
    manifest = manifest or PROJECT_ROOT / SETTINGS["stages"]["frozenManifest"]
    if manifest.exists():
        raise FileExistsError(f"{manifest} exists: the frozen files are recorded once, at M0")
    digests = {p.relative_to(PROJECT_ROOT).as_posix(): sha256(p) for p in frozen_paths()}
    manifest.write_text(json.dumps(digests, indent=2) + "\n", encoding="utf-8")
    return digests


def check_frozen(manifest: Path | None = None) -> None:
    """Raise if any frozen file is missing or differs from its SHA-256 recorded at M0."""
    manifest = manifest or PROJECT_ROOT / SETTINGS["stages"]["frozenManifest"]
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    bad = [rel for rel, digest in recorded.items()
           if not (PROJECT_ROOT / rel).exists() or sha256(PROJECT_ROOT / rel) != digest]
    if bad:
        raise RuntimeError(f"frozen test files changed since M0: {bad}")


def group_draws(groups: np.ndarray, draws: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Cluster-bootstrap weights: (draws x groups) counts of how often each group is drawn, and each unit's group
    index. Row 0 is every group once (the point estimate)."""
    names, index = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=draws)
    return np.vstack([np.ones(len(names)), counts]), index


def weighted_means(values: np.ndarray, weights: np.ndarray, index: np.ndarray) -> np.ndarray:
    """The mean of `values` under each row of group weights (units inherit their group's weight)."""
    sums, counts = np.zeros(weights.shape[1]), np.zeros(weights.shape[1])
    np.add.at(sums, index, values)
    np.add.at(counts, index, 1.0)
    return (weights @ sums) / (weights @ counts)


def interval(values: np.ndarray, level: float) -> tuple[float, float]:
    tail = 100 * (1 - level) / 2
    return float(np.nanpercentile(values, tail)), float(np.nanpercentile(values, 100 - tail))


def paired_difference(before: np.ndarray, after: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    """mean(after) - mean(before) over the same units, with a group-level bootstrap interval
    (settings.json -> stages: bootstrapResamples, ciLevel, seed)."""
    before, after = np.asarray(before, dtype="float64"), np.asarray(after, dtype="float64")
    if before.shape != after.shape or before.shape != np.shape(groups):
        raise ValueError("before, after and groups must be aligned unit by unit")
    s = SETTINGS["stages"]
    weights, index = group_draws(np.asarray(groups), s["bootstrapResamples"], s["seed"])
    diff = weighted_means(after - before, weights, index)
    return (float(diff[0]), *interval(diff[1:], s["ciLevel"]))


def mcnemar_p(before: np.ndarray, after: np.ndarray) -> float:
    """McNemar's exact test on paired 0/1 outcomes (only the discordant pairs count)."""
    before, after = np.asarray(before, dtype=bool), np.asarray(after, dtype=bool)
    only_before, only_after = int(np.sum(before & ~after)), int(np.sum(after & ~before))
    n = only_before + only_after
    return float(binomtest(only_after, n, 0.5).pvalue) if n else 1.0


@dataclass
class GateResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)


def decide(diff_ci_low: float, case_before: dict[str, float], case_after: dict[str, float],
           case_spread: dict[str, float], loss_agrees: bool, guardrail_before: float | None = None,
           guardrail_after: float | None = None, flagged_share: float | None = None,
           address_share_before: float | None = None, address_share_after: float | None = None) -> GateResult:
    """Apply the four rules of TRAINING_PLAN section 3; every failed rule is listed in `reasons`."""
    reasons = []
    if not diff_ci_low > 0:
        reasons.append(f"difference interval reaches {diff_ci_low:+.4f} (not above 0)")
    for case, before in case_before.items():
        drop = before - case_after[case]
        if drop > case_spread.get(case, 0.0):
            reasons.append(f"case {case} fell {drop:.4f} (> its M0 seed spread {case_spread.get(case, 0.0):.4f})")
    if guardrail_before is not None and guardrail_after is not None and guardrail_after < guardrail_before:
        reasons.append(f"guardrail pass rate fell {guardrail_before:.4f} -> {guardrail_after:.4f}")
    limit = SETTINGS["sanity"]["maxFlaggedShare"]
    if flagged_share is not None and flagged_share > limit:
        reasons.append(f"flag rate {flagged_share:.3f} over the sanity limit {limit}")
    if (address_share_before is not None and address_share_after is not None
            and address_share_after > max(limit, address_share_before) + SHARE_TOLERANCE):
        reasons.append(f"largest address share rose {address_share_before:.3f} -> {address_share_after:.3f}")
    if not loss_agrees:
        reasons.append("dev metric rose while the validation loss rose")
    return GateResult(not reasons, reasons)


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                          text=True).stdout.strip()


def record(row: dict, table: Path | None = None) -> pd.DataFrame:
    """Append one stage row to reports/tables/stages.csv (the commit and time are filled in here)."""
    table = table or TABLES_DIR / SETTINGS["stages"]["table"]
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise KeyError(f"not stages.csv columns: {sorted(unknown)}")
    full = {c: row.get(c) for c in COLUMNS}
    full["time_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    full["commit"] = full["commit"] or git_commit()
    old = pd.read_csv(table) if table.exists() else pd.DataFrame(columns=COLUMNS)
    out = pd.concat([old, pd.DataFrame([full])], ignore_index=True) if len(old) else pd.DataFrame([full])
    table.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(table, index=False)
    return out


def last_accepted(family: str, model: str, table: Path | None = None) -> pd.Series | None:
    """The latest accepted row for one model (M0 rows are recorded as accepted), or None."""
    table = table or TABLES_DIR / SETTINGS["stages"]["table"]
    if not table.exists():
        return None
    t = pd.read_csv(table)
    t = t[(t["family"] == family) & (t["model"] == model) & (t["accepted"].astype(str) == "True")]
    return t.iloc[-1] if len(t) else None


DEV_DIR = TABLES_DIR / "stage_dev"
PATTERN_MEASURES = {"spike": "max", "burst": "mean", "dormant": "max"}  # = significance.MEAN_OF's three recalls


def dev_hits(methods: dict, copies: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, tuning plant): flagged at all (max) and the share flagged (mean); a method that
    cannot score these copies (TxGPT keeps scores for the reporting plants only) is left out and logged."""
    from .significance import plant_hits

    parts = []
    for name, method in methods.items():
        try:
            parts.append(plant_hits({name: method}, copies))
        except (KeyError, ValueError) as e:
            log.info("%s skipped on the dev plants: %s", name, e)
    return pd.concat(parts, ignore_index=True)


def run_dev(stage: str) -> int:
    """Dev tier for the DS1 scoring methods: every saved method on the TUNING plants (seeds 40000-40004, never
    the reporting seeds), plus the sanity rule on the real evaluation rows. Writes stage_dev/<stage>_hits.csv
    and <stage>_summary.csv; `run_compare` turns two stages' hits into stages.csv rows."""
    from .baseline import load_copies
    from .methods import real_metrics
    from .scores import saved_methods
    from .snapshot import MANIFEST_PATH

    methods = saved_methods()
    hits = dev_hits(methods, load_copies("refined", "tuning"))
    features = pd.read_csv(MANIFEST_PATH.parent / "features.csv")
    rows = []
    for name, part in hits.groupby("method"):
        recalls = {p: float(part.loc[part["pattern"] == p, m].mean()) for p, m in PATTERN_MEASURES.items()}
        rows.append({"stage": stage, "method": name, **{f"recall_{p}": r for p, r in recalls.items()},
                     "mean_recall": float(np.mean(list(recalls.values()))), **real_metrics(methods[name], features)})
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    hits.assign(stage=stage).to_csv(DEV_DIR / f"{stage}_hits.csv", index=False)
    summary = pd.DataFrame(rows).sort_values("mean_recall", ascending=False)
    summary.to_csv(DEV_DIR / f"{stage}_summary.csv", index=False)
    for r in summary.itertuples():
        log.info("%-20s dev mean recall %.3f (spike %.3f, burst %.3f, dormant %.3f); max address share %.3f",
                 r.method, r.mean_recall, r.recall_spike, r.recall_burst, r.recall_dormant, r.max_address_share)
    return 0


def pattern_mean_draws(hits: pd.DataFrame, weights: np.ndarray, addresses: list[str]) -> np.ndarray:
    """Mean of the three pattern recalls under each row of address weights (row 0 = the point estimate)."""
    where = {a: i for i, a in enumerate(addresses)}
    per_pattern = []
    for pattern, measure in PATTERN_MEASURES.items():
        rows = hits[hits["pattern"] == pattern]
        idx = rows["address"].map(where).to_numpy()
        sums, counts = np.zeros(len(addresses)), np.zeros(len(addresses))
        np.add.at(sums, idx, rows[measure].to_numpy(dtype="float64"))
        np.add.at(counts, idx, 1.0)
        per_pattern.append((weights @ sums) / (weights @ counts))
    return np.mean(per_pattern, axis=0)


def m0_spread(method: str) -> dict[str, float]:
    """Each pattern recall's seed spread (one SD over the 100 distribution seeds on DS1) at M0."""
    seeds = pd.read_csv(TABLES_DIR / "stats_methods_seeds.csv")
    seeds = seeds[(seeds["dataset"] == "DS1") & (seeds["method"] == method)]
    names = {"spike": "recall_spike", "burst": "recall_burst_tx", "dormant": "recall_dormant"}
    return {p: float(seeds[c].std()) for p, c in names.items()} if len(seeds) else {}


def run_compare(before: str, after: str, stage: str, loss_agrees: bool = True, seeds: list[int] | None = None) -> int:
    """Gate every DS1 scoring method of stage `after` against stage `before` (both from run_dev or a stage search),
    paired on the same tuning plants with an address-level bootstrap; one stages.csv row per method. `seeds` keeps
    only those tuning seeds' plants (a stage search gates on seeds it did not rank its candidates on)."""
    s = SETTINGS["stages"]
    hb, ha = (pd.read_csv(DEV_DIR / f"{x}_hits.csv") for x in (before, after))
    if seeds is not None:
        hb, ha = (h[h["seed"].isin(seeds)] for h in (hb, ha))
    sb, sa = (pd.read_csv(DEV_DIR / f"{x}_summary.csv").set_index("method") for x in (before, after))
    key = ["seed", "pattern", "plant_id", "address"]
    for method in sorted(set(ha["method"]) & set(hb["method"])):
        a, b = (h[h["method"] == method].sort_values(key).reset_index(drop=True) for h in (ha, hb))
        if not a[key].equals(b[key]):
            raise RuntimeError(f"{method}: the two stages were scored on different plants")
        addresses = sorted(a["address"].unique())
        weights, _ = group_draws(np.array(addresses), s["bootstrapResamples"], s["seed"])
        diff = pattern_mean_draws(a, weights, addresses) - pattern_mean_draws(b, weights, addresses)
        low, high = interval(diff[1:], s["ciLevel"])
        recalls_b, recalls_a = ({p: float(h.loc[h["pattern"] == p, m].mean()) for p, m in PATTERN_MEASURES.items()}
                                for h in (b, a))
        result = decide(low, recalls_b, recalls_a, m0_spread(method), loss_agrees,
                        flagged_share=float(sa.loc[method, "flag_rate"]),
                        address_share_before=float(sb.loc[method, "max_address_share"]),
                        address_share_after=float(sa.loc[method, "max_address_share"]))
        used = sorted(a["seed"].unique())
        record({"family": "scoring", "model": method, "stage": stage, "tier": "dev", "metric": "mean planted recall",
                "value": float(np.mean(list(recalls_a.values()))), "previous_stage": before,
                "diff_previous": float(diff[0]), "diff_previous_ci_low": low, "diff_previous_ci_high": high,
                "accepted": result.accepted, "reasons": "; ".join(result.reasons),
                "seeds": f"tuning {used[0]}-{used[-1]}" if seeds is None else "gate " + " ".join(map(str, used))})
        log.info("%-20s %s: %+.3f [%+.3f, %+.3f] -> %s %s", method, stage, diff[0], low, high,
                 "ACCEPTED" if result.accepted else "rejected", "; ".join(result.reasons))
    return 0


def run_freeze() -> int:
    digests = freeze()
    log.info("Recorded the SHA-256 of %d frozen files in %s", len(digests), SETTINGS["stages"]["frozenManifest"])
    return 0
