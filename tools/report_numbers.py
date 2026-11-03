"""Write every number and table the A2 report quotes, from the files the pipeline wrote.

Serves the rule "every reported number is written to reports/tables/*.csv by code; the report quotes those
files only" (CLAUDE.md) and the brief's report (docs/brief/assignment2_description.txt, section 4). The report's
prose never types a result: it writes \\V{key}, and this module defines each key from a CSV in reports/tables/,
a dataset manifest, config/settings.json, or report/sources/manual.json (the few values no pipeline file
records, each with its source). Re-running the evaluation and then this script updates the report, so the text
cannot drift from the results.

Writes report/generated/numbers.tex (the \\V keys) and report/generated/tab_*.tex (the tables). Fails if a key
the report uses is undefined, or if a value cannot be found.

Usage (repository root): python tools/report_numbers.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
REPORT = ROOT / "report"
OUT = REPORT / "generated"
SETTINGS = json.loads((ROOT / "config" / "settings.json").read_text(encoding="utf-8"))

# The names the report uses for each scoring method (as in the figures' legends).
METHOD_NAMES = {
    "baseline": "Robust z baseline (A1)",
    "fusion": "Score fusion",
    "regression": "Regression residual",
    "regression_interval": "Regression range (conformal)",
    "kmeans": "K-Means distance",
    "dbscan": "DBSCAN distance",
    "isolation_forest": "Isolation Forest",
    "lof": "Local Outlier Factor",
    "txgpt_method": "TxGPT surprisal",
}
METHOD_FAMILY = {
    "baseline": "Statistics", "fusion": "Combination", "regression": "Regression",
    "regression_interval": "Regression", "kmeans": "Clustering", "dbscan": "Clustering",
    "isolation_forest": "Outlier detection", "lof": "Outlier detection", "txgpt_method": "Sequence model",
}
MODEL_NAMES = {"logistic": "Logistic regression", "knn": "k-nearest neighbours", "randomForest": "Random forest",
               "histGradientBoosting": "Hist.\\ gradient boosting"}


# ---------------------------------------------------------------- reading and formatting

def table(name: str) -> pd.DataFrame:
    """One CSV from reports/tables (name without .csv)."""
    return pd.read_csv(TABLES / f"{name}.csv")


def manifest(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def row(df: pd.DataFrame, **where: Any) -> pd.Series:
    """The single row matching every column=value pair; anything else is an error, never a guess."""
    mask = pd.Series(True, index=df.index)
    for col, val in where.items():
        mask &= df[col] == val
    hits = df[mask]
    if len(hits) != 1:
        raise LookupError(f"expected one row for {where}, found {len(hits)}")
    return hits.iloc[0]


def num(x: float, digits: int = 3) -> str:
    """A decimal with a fixed number of places (proportions and losses)."""
    return f"{float(x):.{digits}f}"


def count(x: float) -> str:
    """A whole number with thousands separators, as LaTeX text."""
    return f"{int(round(float(x))):,}".replace(",", "{,}")


def pct(x: float, digits: int = 1) -> str:
    """A proportion (0-1) as a percentage."""
    return f"{100 * float(x):.{digits}f}\\%"


def ci(lo: float, hi: float, fmt: Callable[[float], str] = num) -> str:
    return f"[{fmt(lo)}, {fmt(hi)}]"


def eda(statistic: str) -> str:
    """A value of eda_tests.csv by its statistic's exact name."""
    d = table("eda_tests")
    return str(row(d, statistic=statistic)["value"])


# ---------------------------------------------------------------- the numbers

def numbers() -> dict[str, str]:
    """Every \\V key: its LaTeX text."""
    v: dict[str, str] = {}
    s = SETTINGS

    # DS1, DS2, DS10 collection (manifests; written by explorer/snapshot.py, corpus.py, ds10.py)
    m1 = manifest("data/processed/esplora/manifest.json")
    p = m1["selection_rule"]["parameters"]
    v["ds1.blocks"] = count(p["snapshotBlockCount"])
    v["ds1.candidates"] = count(m1["candidate_pool"]["size"])
    v["ds1.main_min_tx"], v["ds1.main_max_tx"] = count(p["mainMinTx"]), count(p["mainMaxTx"])
    v["ds1.main_min_days"] = count(p["mainMinSpanDays"])
    v["ds1.short_min_tx"], v["ds1.short_max_tx"] = count(p["shortMinTx"]), count(p["shortMaxTx"])
    v["ds1.main"], v["ds1.short"] = count(p["targetMainAddresses"]), count(p["targetShortAddresses"])
    cleaning = table("cleaning_ds1")
    v["ds1.tx"] = count(cleaning["rows_after"].iloc[-1])

    m2 = manifest("data/processed/corpus/manifest.json")
    v["ds2.blocks"] = count(m2["blocks"]["count"])
    v["ds2.ds1_removed"] = count(m2["candidate_pool"]["ds1_removed"])
    v["ds2.visited"] = count(m2["candidate_pool"]["visited"])
    v["ds2.kept"] = count(m2["visits"]["counts"]["kept"])
    v["ds2.rows"] = count(m2["totals"]["rows"])
    v["ds2.fetch_errors"] = count(m2["visits"]["counts"]["fetch_error"])
    v["ds2.too_many"] = count(m2["visits"]["counts"]["too_many"])
    v["ds2.too_few"] = count(m2["visits"]["counts"]["too_few"])

    m10 = manifest("data/processed/ds10/manifest.json")
    v["ds10.rows"] = count(m10["rows"])
    if m10["crosscheck_with_teammate_csv"]["differs"] != 0:
        raise ValueError("DS10 no longer agrees with the teammate's file; the report's sentence is wrong")

    # DS3, DS4
    es = table("elliptic_summary").iloc[0]
    v["ell.rows"], v["ell.labelled"] = count(es["rows"]), count(es["labelled"])
    hp = manifest("data/processed/bitcoinheist/processing.json")
    v["heist.rows"], v["heist.positives"] = count(hp["rows"]), count(hp["positives"])

    # Injection design and TxGPT tokens (settings, model metadata)
    inj = s["injection"]
    v["inj.spike"], v["inj.burst"] = count(inj["spikeMultiplier"]), count(inj["burstSize"])
    v["inj.burst_hours"] = count(inj["burstWindowHours"])
    v["inj.dormant_min"], v["inj.dormant_max"] = count(inj["dormantDaysMin"]), count(inj["dormantDaysMax"])
    v["inj.per_pattern"] = count(inj["plantsPerPattern"])
    v["txgpt.vocab"] = count(manifest("models/txgpt_meta.json")["vocab_size"])

    # Data analysis (eda_tests.csv)
    v["eda.scale_ratio"] = count(float(eda("ratio of largest to smallest address median amount")))
    v["eda.f1_skew_raw"] = num(float(eda("F1 skewness (raw)")), 1)
    v["eda.f1_skew_log"] = num(float(eda("F1 skewness (log10)")), 2)
    v["eda.f4_kurt_raw"] = count(float(eda("F4 kurtosis (raw, excess)")))
    v["eda.dir_tested"] = count(float(eda("addresses tested (>=5 tx each way)")))
    v["eda.dir_differ"] = count(float(eda("addresses whose in and out amounts differ (Mann-Whitney U, p<0.05)")))
    v["eda.ell_train_share"] = pct(float(eda("illicit share of labelled, training steps")))
    v["eda.ell_test_share"] = pct(float(eda("illicit share of labelled, test steps")))
    v["eda.ell_unknown"] = pct(float(eda("unknown share of all rows")), 0)
    v["eda.heist_share"] = pct(float(eda("overall ransomware share")))
    d = table("eda_tests")
    effects = d[d["statistic"].str.contains("rank-biserial", regex=False)]["value"].astype(float)
    if effects.empty:
        raise LookupError("eda_tests.csv has no BitcoinHeist rows: download DS4 (README section 5) first")
    v["eda.heist_max_effect"] = num(effects.abs().max(), 2)

    # Ablation (A1's features as written against the refined ones)
    ab = table("eval_ablation")
    for kind, key in (("as_written", "written"), ("refined", "refined")):
        r = row(ab, method="baseline", features=kind)
        v[f"abl.{key}_flag_rate"] = pct(r["flag_rate"])
        v[f"abl.{key}_recall"] = num(r["mean_recall"])

    # Selection
    v["sel.stability_target"] = num(s["model"]["stabilityTarget"], 1)
    v["sel.sanity"] = pct(s["sanity"]["maxFlaggedShare"], 0)
    sig = table("eval_significance")
    sel = table("eval_selection")
    for m in ("baseline", "fusion", "txgpt_method"):
        key = "txgpt" if m == "txgpt_method" else m
        v[f"ds1.{key}.mean"] = num(row(sig, method=m)["mean_recall"])
        v[f"ds1.{key}.flag_rate"] = pct(row(sel, method=m)["flag_rate"])
        v[f"ds1.{key}.precision"] = num(row(sel, method=m)["precision_lower_bound_mean"])
    v["sig.fusion_p"] = num(row(sig, method="fusion")["p_value"])

    # Technical
    v["tech.tests"] = count(collected_tests())
    sizes = table("txgpt_sizes")
    v["txgpt.s_params"] = count(row(sizes, chosen=True)["parameters"])

    # External checks
    # the DS2 check scores every main DS2 history (split_ds2.csv has one row per history)
    split2 = table("split_ds2")
    v["ds2.check_addresses"] = count((split2["role"] == "main").sum())
    d10 = table("eval_ds10")
    for m in ("baseline", "fusion", "txgpt_method"):
        key = "txgpt" if m == "txgpt_method" else m
        r = row(d10, method=m)
        v[f"ds10.{key}.mean"] = num(r["ds10_mean_recall"])
        v[f"ds10.{key}.ci"] = ci(r["ds10_mean_recall_ci_low"], r["ds10_mean_recall_ci_high"])
    st = table("stats_methods_summary")
    v["stats.ds10_baseline_rank1"] = pct(row(st, dataset="DS10", method="baseline")["rank1_share"], 0)

    # Classifiers
    cs = table("stats_classifiers_summary")
    rf = row(cs, scheme="Elliptic (temporal, all features)", model="randomForest")
    v["cls.ell_rf_f1"] = num(rf["f1_mean"])
    v["cls.ell_rf_f1_ci"] = ci(rf["f1_ci_low"], rf["f1_ci_high"])
    heist = table("heist_results")
    v["cls.heist_temporal_max_f1"] = num(heist[heist["scheme"] == "temporal"]["f1"].max(), 2)
    lofo = table("heist_lofo")
    trees = lofo[lofo["model"].isin(["randomForest", "histGradientBoosting"])]["recall"]
    v["cls.lofo_range"] = f"{pct(trees.min())}--{pct(trees.max())}"
    lc = table("label_check")
    r = row(lc, dataset="elliptic_AF", method="random_forest (supervised)")
    v["label.ell_rf_lift"] = num(r["pr_auc"] / r["prevalence"], 1)

    # Language models
    # The summary tables lm.adopt decides on (the per-prompt files also hold glossary prompts, which they exclude);
    # text150m_v2 is the round-2 writer in use.
    v["lm.guard_ds7"] = pct(row(table("assistant_eval"), model="text150m_v2")["guardrail_pass"])
    v["lm.guard_ds10"] = pct(row(table("assistant_eval_ds10"), model="text150m_v2")["guardrail_pass"])
    ws = table("stats_writers_summary")
    v["lm.full_ds7"] = pct(row(ws, test_set="DS7 v1 test", writer="text150m_v2")["fully_correct"])
    v["lm.full_ds10"] = pct(row(ws, test_set="DS10", writer="text150m_v2")["fully_correct"])
    pooled = row(ws, test_set="pooled (12 sets)", writer="text150m_v2")
    v["lm.pooled"] = pct(pooled["fully_correct"])
    v["lm.pooled_ci"] = f"95\\% interval {ci(pooled['ci_low'], pooled['ci_high'], pct)}"
    v["lm.glossary"] = pct(row(ws, test_set="glossary (key facts)", writer="text150m_v2")["fully_correct"], 0)

    # The few values no pipeline file records (report/sources/manual.json names each source)
    manual = json.loads((REPORT / "sources" / "manual.json").read_text(encoding="utf-8"))
    for key, entry in manual.items():
        if not key.startswith("_"):
            v[key] = entry["value"]
    return v


def collected_tests() -> int:
    """How many tests pytest collects (the README quotes the same count)."""
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:warnings"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    found = re.search(r"(\d+) tests? collected", out)
    if not found:
        raise RuntimeError("pytest --collect-only printed no test count")
    return int(found.group(1))


# ---------------------------------------------------------------- the tables

def tabular(caption: str, label: str, spec: str, header: list[str], rows: list[list[str]],
            note: str = "", size: str = "\\small", width: str = "\\textwidth", place: str = "htbp") -> str:
    """A booktabs table in a float, caption on top (the report's table style); place "H" keeps it in the text."""
    lines = [f"\\begin{{table}}[{place}]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabularx}}{{{width}}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabularx}"]
    if note:
        lines.append(f"\\par\\smallskip\\parbox{{{width}}}{{\\footnotesize {note}}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def tab_datasets(v: dict[str, str]) -> str:
    rows = [
        ["DS1", "Our Esplora snapshot", "Blockstream Esplora API (public data)",
         f"{v['ds1.main']}\\,+\\,{v['ds1.short']} addresses, {v['ds1.tx']} tx", "Every scoring method; evaluation"],
        ["DS2", "Our Esplora crawl (hashed IDs)", "Esplora API via mempool.space",
         f"{v['ds2.kept']} addresses, {v['ds2.rows']} tx", "Trains TxGPT; check on unseen addresses"],
        ["DS3", "Elliptic data set", "Kaggle (CC BY-NC-ND 4.0)", f"{v['ell.rows']} tx", "Classification"],
        ["DS4", "BitcoinHeist", "UCI repository 526 (CC BY 4.0)", f"{v['heist.rows']} address-days",
         "Classification"],
        ["DS5", "FineWeb-Edu", "Hugging Face (ODC-By 1.0)", f"{v['lm.pretrain_tokens']} billion tokens used",
         "Assistant pretraining"],
        ["DS6", "smol-smoltalk", "Hugging Face (Apache 2.0)", "chat conversations", "Assistant chat stage"],
        ["DS7", "Our explanation pairs", "Built from DS1 and DS2 facts", "question--answer pairs",
         "Teaching and testing the Assistant"],
        ["DS10", "A teammate's snapshot", "Esplora API, refetched and cross-checked",
         f"{count(manifest('data/processed/ds10/manifest.json')['crosscheck_with_teammate_csv']['addresses_agreeing'])}"
         f" addresses, {v['ds10.rows']} tx", "Held-out final test of every model"],
    ]
    return tabular("The datasets and the role of each (tx: transactions).", "tab:datasets",
                   "@{}l L{3.0cm} L{3.7cm} L{3.0cm} Y@{}", ["ID", "Dataset", "Source (licence)", "Size", "Role"],
                   rows, size="\\footnotesize")


def tab_features(v: dict[str, str]) -> str:
    f = SETTINGS["features"]
    window = (f"earlier transactions of the same address in the previous {f['comparisonWindowDays']} days, "
              f"extended back to reach {f['comparisonMinTx']}")
    rows = [
        ["F1", "Size", "Amount the address received or sent (BTC)", "log10; same direction only"],
        ["F2", "Burst", f"Earlier transactions in the previous {f['burstWindowDays']} days", "raw count"],
        ["F3", "Gap", "Hours since the address's previous transaction", "log10"],
        ["F4", "Relative size", "Amount divided by the usual amount in the same direction", "log10"],
    ]
    note = (f"Each feature is compared with {window}; a transaction is scored once the address has at least "
            f"{f['minHistoryTx']} earlier transactions over {f['minHistorySpanDays']} days.")
    return tabular("The four explained features (A1, refined). Each becomes a robust z-score against the same "
                   "address's earlier history.", "tab:features", "@{}l l Y l@{}",
                   ["ID", "Name", "Definition", "Compared on"], rows, note=note)


def tab_candidates(v: dict[str, str]) -> str:
    rows = [
        ["Statistics", "Robust z-score baseline (A1)", "DS1", "Largest $|z|$ over F1--F4", "Flag threshold"],
        ["Clustering", "K-Means; DBSCAN", "DS1", "Distance to normal clusters", "$k$; eps, min.\\ samples"],
        ["Clustering", "K-Means address profiles", "DS1, DS2", "Activity profile in words", "$k$ (silhouette)"],
        ["Regression", "Ridge; random forest; hist.\\ gradient boosting", "DS1",
         "Miss on predicted amount and gap", "Per-model grids"],
        ["Regression", "Quantile boosting + conformal", "DS1", "Distance outside a calibrated range",
         "Quantiles, coverage"],
        ["Outlier detection", "Isolation Forest; LOF (novelty)", "DS1", "Isolation depth; local density",
         "Trees, samples; neighbours"],
        ["Combination", "Score fusion", "DS1", "Weighted pair of scores", "Weights at a flag budget"],
        ["Classification", "Logistic regression; KNN; random forest; hist.\\ gradient boosting", "DS3, DS4",
         "Class probability", "Validation split"],
        ["Sequence model", "TxGPT (trained from scratch)", "DS2", "Surprisal of each transaction",
         "3 sizes $\\times$ 3 seeds"],
        ["Language model", "Explorer Assistant, 152M (trained from scratch)", "DS5--DS7",
         "Explanation (guardrailed)", "Adoption rule"],
    ]
    return tabular("Every model trained, by family. All were tuned on data that never produced a reported "
                   "result.", "tab:candidates", "@{}L{2.2cm} L{4.0cm} l L{3.4cm} Y@{}",
                   ["Family", "Models", "Data", "Output", "Tuned"], rows, size="\\footnotesize")


def tab_objectives(v: dict[str, str]) -> str:
    sel = table("eval_selection")
    b = row(sel, method="baseline")
    inj = SETTINGS["injection"]
    target = inj["recallTarget"]
    met = sum(b[f"recall_{p}_mean"] >= target for p in ("spike", "burst_tx", "dormant"))
    d10 = row(table("eval_ds10"), method="baseline")
    agree = row(table("eval_agreement"), a="baseline", b="fusion")["jaccard"]
    rows = [
        ["Snapshot of 20--30 main addresses (50+ tx, 90+ days) and 2--3 short ones",
         f"{v['ds1.main']} main and {v['ds1.short']} short addresses", "Met"],
        [f"Planted recall $\\geq$ {pct(target, 0)} for {inj['patternsMeetingTarget']} of 3 patterns",
         f"Spike {num(b['recall_spike_mean'])}, dormant {num(b['recall_dormant_mean'])}, "
         f"burst {num(b['recall_burst_tx_mean'])}",
         "Met" if met >= inj["patternsMeetingTarget"] else "Not met"],
        [f"Stability: top-50 Jaccard $\\geq$ {v['sel.stability_target']}", num(b["stability"]),
         "Met" if b["stability"] >= SETTINGS["model"]["stabilityTarget"] else "Not met"],
        [f"Sanity: no address over {v['sel.sanity']} flagged",
         f"DS1: {int(b['addresses_over_limit'])} addresses over; DS10: {int(d10['ds10_addresses_over_limit'])}",
         "Partly met"],
        ["Method agreement reported", f"Baseline--fusion Jaccard {num(agree)}", "Met"],
        ["Blind reviewer check (Cohen's kappa)", v["obj.reviewer"], "Open"],
        ["Accuracy, precision, recall, F1 where labels exist", "DS3 and DS4 (Table~\\ref{tab:classifiers})",
         "Met"],
    ]
    return tabular("Assignment~1's objectives for the selected method (the robust z baseline) and the results.",
                   "tab:objectives", "@{}Y L{5.6cm} l@{}", ["Objective (set in Assignment~1)", "Result", "Status"],
                   rows, size="\\footnotesize")


def tab_ds1(v: dict[str, str]) -> str:
    sig, sel = table("eval_significance"), table("eval_selection")
    order = sel.sort_values("mean_recall", ascending=False)["method"]
    rows = []
    for m in order:
        r, q = row(sig, method=m), row(sel, method=m)
        name = METHOD_NAMES[m] + (" $\\star$" if m == "baseline" else "")
        rows.append([name, METHOD_FAMILY[m], num(r["recall_spike"]), num(r["recall_burst_tx"]),
                     num(r["recall_dormant"]), f"{num(r['mean_recall'])} {ci(r['mean_recall_ci_low'], r['mean_recall_ci_high'])}",
                     num(q["stability"]), pct(q["flag_rate"])])
    note = ("Recall on planted transactions over five reporting seeds; intervals are 95\\% address-level "
            "bootstrap. Mean recall averages spike, burst (per transaction and per event) and dormant. "
            "Stability is the top-50 Jaccard over refits. Flag rate is the share of real evaluation-period "
            "transactions flagged. $\\star$ the selected method.")
    return tabular("Scoring methods on DS1 (sorted by mean planted recall).", "tab:ds1",
                   "@{}L{3.6cm} l r r r l r R@{}",
                   ["Method", "Family", "Spike", "Burst", "Dormant", "Mean [95\\% CI]", "Stab.", "Flags"],
                   rows, note=note, size="\\footnotesize")


def tab_classifiers(v: dict[str, str]) -> str:
    ell, heist = table("elliptic_results"), table("heist_results")
    rows = []
    for fs, label in (("AF", "Elliptic, all features"), ("LF", "Elliptic, local features")):
        for _, r in ell[ell["feature_set"] == fs].iterrows():
            rows.append([label, MODEL_NAMES[r["model"]], num(r["accuracy"]), num(r["precision"]),
                         num(r["recall"]), num(r["f1"]), num(r["pr_auc"])])
    for scheme, label in (("temporal", "BitcoinHeist, temporal"), ("random", "BitcoinHeist, random")):
        for _, r in heist[heist["scheme"] == scheme].iterrows():
            rows.append([label, MODEL_NAMES[r["model"]], num(r["accuracy"]), num(r["precision"]),
                         num(r["recall"]), num(r["f1"]), num(r["pr_auc"])])
    note = ("Test splits: Elliptic time steps 35--49; BitcoinHeist years after the training years (temporal) or "
            "a random 20\\%. Metrics are for the datasets' own positive classes (illicit; ransomware), at the "
            "threshold chosen on validation data.")
    return tabular("Classifiers on the labelled datasets.", "tab:classifiers", "@{}L{3.6cm} Y r r r r r@{}",
                   ["Data and split", "Model", "Accuracy", "Precision", "Recall", "F1", "PR-AUC"], rows,
                   note=note, size="\\footnotesize")


# ---------------------------------------------------------------- the example prediction (appendix)

# README section 3's what-if example: a DS1 address and a hypothetical outgoing transaction.
EXAMPLE = {"address": "1Bbcsfxj2JRUD5crcdZVHBcViT61vrJ3dz", "amount": 0.05, "direction": "out",
           "time": "2026-10-20T12:00Z"}


def example_tex() -> str:
    """The Predictor's what-if answer (the fields the user sees) and its explanation, as verbatim LaTeX."""
    sys.path.insert(0, str(ROOT))
    from explorer.predict import Predictor  # imported here: loading the models takes a few seconds

    p = Predictor.load()
    guess = p.score_what_if(EXAMPLE["address"], EXAMPLE["amount"], EXAMPLE["direction"], EXAMPLE["time"])
    text = p.explain(guess, use_assistant=False)
    lines = [f"input:    {EXAMPLE['amount']} BTC {EXAMPLE['direction']} at {EXAMPLE['time']}, address {EXAMPLE['address']}",
             f"scored: {str(guess['scored']).lower()}   flagged: {str(guess['flagged']).lower()}   "
             f"score (largest |z|): {guess['score']:.2f}",
             f"{'feature':20s} {'value':>12s} {'usual':>12s} {'z':>7s}  role"]
    for f in guess["features"]:
        lines.append(f"{f['id'] + ' ' + f['name']:20s} {f['value']:12.6g} {f['usual']:12.6g} {f['z']:7.2f}  {f['role']}")
    explanation = text["explanation"] if isinstance(text, dict) else str(text)
    wrapped = "\n".join(textwrap.fill(line, 100, subsequent_indent="  ") for line in explanation.splitlines())
    return ("{\\footnotesize\n\\begin{verbatim}\n" + "\n".join(lines) + "\n\\end{verbatim}\n}\n"
            "Explanation shown to the user (template writer):\n{\\footnotesize\n\\begin{verbatim}\n"
            + wrapped + "\n\\end{verbatim}\n}\n")


# ---------------------------------------------------------------- main

def used_keys() -> set[str]:
    """Every \\V{key} in the report's sections."""
    keys: set[str] = set()
    for tex in (REPORT / "sections").glob("*.tex"):
        keys |= set(re.findall(r"\\V\{([^}]+)\}", tex.read_text(encoding="utf-8")))
    return keys


def main() -> int:
    v = numbers()
    missing = sorted(used_keys() - set(v))
    if missing:
        print("Undefined keys used by the report: " + ", ".join(missing), file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["% Written by tools/report_numbers.py from reports/tables and the manifests. Do not edit."]
    lines += [f"\\expandafter\\def\\csname v@{k}\\endcsname{{{v[k]}}}" for k in sorted(v)]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, build in (("datasets", tab_datasets), ("features", tab_features), ("candidates", tab_candidates),
                        ("objectives", tab_objectives), ("ds1", tab_ds1), ("classifiers", tab_classifiers)):
        (OUT / f"tab_{name}.tex").write_text(build(v), encoding="utf-8")
    (OUT / "example.tex").write_text(example_tex(), encoding="utf-8")
    print(f"Wrote {len(v)} numbers, 6 tables and the example prediction to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
