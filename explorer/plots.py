"""Every figure in the report and the EDA notebook, and the statistics printed beside them.

Serves plan §7 (docs/plan/sections/s07-analysis.tex) and report criterion 2: each analysis is a
question, a plot, a statistic and the decision it informs. Each figure function returns
(figure, statistics); `python -m explorer.cli figures` saves every figure whose inputs exist as a
vector PDF in reports/figures/ and every statistic in reports/tables/eda_tests.csv, and
notebooks/01_eda.ipynb calls the same functions, so the notebook and the report never disagree.

Plot rules (plan §7, A1 NFR-01): a title; axis labels with units; log axes, stated on the axis,
wherever a variable spans orders of magnitude; the snapshot date in a footnote on DS1 plots; one
colour per method across the whole report, from a colour-blind-checked palette, and never colour
alone: every series is also named on an axis, in a legend or by a direct label.

Elliptic is CC BY-NC-ND 4.0, so its figures show counts, metrics and curves, never data rows.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")  # files only; the notebook switches to its inline backend itself
import matplotlib.pyplot as plt  # noqa: E402 - the backend must be chosen first
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy import stats as st  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from .explain import METHOD_NAMES, SATS_PER_BTC  # noqa: E402
from .features import FEATURE_IDS  # noqa: E402
from .settings import PROJECT_ROOT, SETTINGS  # noqa: E402
from .snapshot import MANIFEST_PATH  # noqa: E402

log = logging.getLogger(__name__)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
CLASSIFY_DIR = PROJECT_ROOT / "data" / "interim" / "classify"

# One fixed colour per method (validated categorical palette, light mode, in this order) and per
# role. Text stays in ink colours; colour only marks identity.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_COLOURS = dict(zip(SETTINGS["evaluation"]["explainabilityOrder"], PALETTE))
MODEL_COLOURS = {"logistic": PALETTE[0], "knn": PALETTE[1], "randomForest": PALETTE[2], "histGradientBoosting": PALETTE[6]}
MODEL_NAMES = {"logistic": "Logistic regression", "knn": "KNN", "randomForest": "Random forest",
               "histGradientBoosting": "Hist. gradient boosting"}
DIRECTION_COLOURS = {"in": PALETTE[0], "out": PALETTE[1]}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SEQUENTIAL = "Blues"
DIVERGING = "RdBu_r"  # blue-red poles with a light neutral middle

Stats = list[dict[str, Any]]
FigureFn = Callable[..., tuple[Figure, Stats]]


def style() -> None:
    """Recessive axes and grid, readable sizes, embedded fonts for the PDFs."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.bbox": "tight", "pdf.fonttype": 42, "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelcolor": INK, "axes.edgecolor": "#c3c2b7",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    })


def stat(figure: str, name: str, value: Any, note: str = "") -> dict[str, Any]:
    return {"figure": figure, "statistic": name, "value": value, "note": note}


def snapshot_note(fig: Figure) -> None:
    """The DS1 footnote: which snapshot this is, and the limitation."""
    built = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["built_utc"][:10]
    fig.text(0.01, -0.02, f"DS1: Esplora snapshot of 28 addresses, built {built} (UTC). Unusual is not wrongdoing.",
             fontsize=7, color=MUTED, ha="left", va="top")


# -- loaders ---------------------------------------------------------------------------------


def load_ds1() -> tuple[pd.DataFrame, pd.DataFrame]:
    """DS1 features (refined method) and the clean transaction table."""
    folder = MANIFEST_PATH.parent
    return pd.read_csv(folder / "features.csv"), pd.read_csv(folder / "transactions.csv")


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / f"{name}.csv")


# -- DS1 analyses ----------------------------------------------------------------------------


def fig_history_sizes(features: pd.DataFrame) -> tuple[Figure, Stats]:
    """How big and how long is each history? (decides whether 70/30 leaves enough evaluation rows)"""
    per = features.groupby(["address", "role"]).agg(tx=("txid", "size"), span=("history_span_days", "max")).reset_index()
    per = per.sort_values("tx", ascending=False)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [3, 1.3]})
    colours = [PALETTE[0] if r == "main" else PALETTE[1] for r in per["role"]]
    a.bar(range(len(per)), per["tx"], color=colours, width=0.8)
    a.set(title="Transactions per DS1 address", xlabel="Address (sorted; labels omitted)", ylabel="Transactions (count)")
    a.set_xticks([])
    a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PALETTE[0]), plt.Rectangle((0, 0), 1, 1, color=PALETTE[1])],
             labels=["main (50-1,000 tx)", "short history (3-9 tx)"], loc="upper right")
    main = per[per["role"] == "main"]
    b.boxplot(main["span"], widths=0.5, medianprops={"color": INK})
    b.set(title="History span (main)", ylabel="Days from first to last transaction", xticks=[])
    snapshot_note(fig)
    q1, q3 = main["tx"].quantile([0.25, 0.75])
    s = [stat("history_sizes", "median transactions per main address", float(main["tx"].median())),
         stat("history_sizes", "IQR transactions per main address", f"{q1:.0f}-{q3:.0f}"),
         stat("history_sizes", "median history span (days, main)", round(float(main["span"].median()), 1))]
    return fig, s


def fig_feature_distributions(features: pd.DataFrame) -> tuple[Figure, Stats]:
    """How are F1-F4 distributed? (heavy tails justify log scales, robust z and the spread floor)"""
    f = features[features["role"] == "main"]
    amount_btc = (10 ** f["F1"] - 1) / SATS_PER_BTC
    panels = [("F1", amount_btc[amount_btc > 0], "Amount moved (BTC, log scale)", True),
              ("F2", f["F2"].dropna(), "Earlier transactions in the previous 7 days (count)", False),
              ("F3", f["F3"].dropna()[f["F3"].dropna() > 0], "Hours since the previous transaction (log scale)", True),
              ("F4", f["F4"].dropna()[f["F4"].dropna() > 0], "Amount / usual amount, same direction (log scale)", True)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6))
    s: Stats = []
    for ax, (fid, values, label, logx) in zip(axes.ravel(), panels):
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 40) if logx else 40
        ax.hist(values, bins=bins, color=PALETTE[0], edgecolor="white", linewidth=0.4)
        if logx:
            ax.set_xscale("log")
        ax.set(title=f"{fid}", xlabel=label, ylabel="Transactions")
        raw_skew = float(st.skew(values))
        s.append(stat("feature_distributions", f"{fid} skewness (raw)", round(raw_skew, 2)))
        s.append(stat("feature_distributions", f"{fid} kurtosis (raw, excess)", round(float(st.kurtosis(values)), 2)))
        if logx:
            s.append(stat("feature_distributions", f"{fid} skewness (log10)", round(float(st.skew(np.log10(values))), 2),
                          "the scale the baseline compares on"))
    fig.suptitle("DS1 features are heavy-tailed: amounts, gaps and ratios span orders of magnitude", fontsize=10, y=1.0)
    fig.tight_layout()
    snapshot_note(fig)
    return fig, s


def fig_direction_amounts(features: pd.DataFrame) -> tuple[Figure, Stats]:
    """Do incoming and outgoing amounts differ, address by address? (evidence for CR-04)"""
    f = features[(features["role"] == "main") & features["direction"].isin(["in", "out"])].copy()
    f["btc"] = f["net_sats"].abs() / SATS_PER_BTC
    order = f.groupby("address")["btc"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 3.6))
    s: Stats = []
    differ = tested = 0
    for i, address in enumerate(order):
        g = f[f["address"] == address]
        for offset, d in ((-0.18, "in"), (0.18, "out")):
            v = g.loc[g["direction"] == d, "btc"]
            if len(v):
                ax.boxplot(v, positions=[i + offset], widths=0.3, showfliers=False, patch_artist=True,
                           boxprops={"facecolor": DIRECTION_COLOURS[d], "edgecolor": DIRECTION_COLOURS[d]},
                           medianprops={"color": INK}, whiskerprops={"color": DIRECTION_COLOURS[d]},
                           capprops={"color": DIRECTION_COLOURS[d]})
        a, b = g.loc[g["direction"] == "in", "btc"], g.loc[g["direction"] == "out", "btc"]
        if len(a) >= 5 and len(b) >= 5:
            tested += 1
            differ += st.mannwhitneyu(a, b).pvalue < 0.05
    ax.set_yscale("log")
    ax.set(title="Each address has its own scale, and its in and out amounts differ",
           xlabel="DS1 main address (sorted by median amount; labels omitted)", ylabel="|Net amount| (BTC, log scale)")
    ax.set_xticks([])
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=DIRECTION_COLOURS[d]) for d in ("in", "out")],
              labels=["received (in)", "sent (out)"], loc="upper left")
    snapshot_note(fig)
    medians = f.groupby("address")["btc"].median()
    s += [stat("direction_amounts", "addresses tested (>=5 tx each way)", tested),
          stat("direction_amounts", "addresses whose in and out amounts differ (Mann-Whitney U, p<0.05)", int(differ)),
          stat("direction_amounts", "ratio of largest to smallest address median amount", round(float(medians.max() / medians.min()), 0),
               "why one global threshold cannot fit every address")]
    return fig, s


def fig_feature_correlation(features: pd.DataFrame) -> tuple[Figure, Stats]:
    """Are the features redundant? (F1 and F4 are expected to correlate)"""
    cols = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8_sin", "F8_cos"]
    rho = features[features["scorable"]][cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(rho, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{rho.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(rho.iloc[i, j]) > 0.6 else INK)
    fig.colorbar(im, ax=ax, label="Spearman rho", shrink=0.8)
    ax.set_title("Feature correlations (scorable DS1 transactions)")
    snapshot_note(fig)
    pairs = rho.where(np.triu(np.ones(rho.shape, dtype=bool), 1)).stack()
    top = pairs.abs().sort_values(ascending=False).head(3)
    s = [stat("feature_correlation", f"rho {a}-{b}", round(float(pairs[(a, b)]), 2)) for a, b in top.index]
    s.append(stat("feature_correlation", "rho F1-F4", round(float(rho.loc["F1", "F4"]), 2), "absolute vs habitual size"))
    return fig, s


def fig_periodicity(transactions: pd.DataFrame) -> tuple[Figure, Stats]:
    """Is activity periodic in the day or week? (whether F8, the hour, is worth testing)"""
    t = pd.to_datetime(transactions["block_time_utc"], unit="s", utc=True)
    hours, days = t.dt.hour.value_counts().reindex(range(24), fill_value=0), t.dt.weekday.value_counts().reindex(range(7), fill_value=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw={"width_ratios": [2.4, 1]})
    a.bar(hours.index, hours.values, color=PALETTE[0], width=0.8)
    a.set(title="Transactions by hour of day", xlabel="Hour (UTC)", ylabel="Transactions", xticks=range(0, 24, 3))
    b.bar(days.index, days.values, color=PALETTE[0], width=0.8)
    b.set(title="By weekday", xlabel="Day (UTC)", ylabel="Transactions")
    b.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    snapshot_note(fig)
    s = [stat("periodicity", "chi-square vs uniform hours: p-value", float(st.chisquare(hours.values).pvalue)),
         stat("periodicity", "chi-square vs uniform weekdays: p-value", float(st.chisquare(days.values).pvalue)),
         stat("periodicity", "busiest hour (UTC)", int(hours.idxmax()))]
    return fig, s


def fig_gap_ecdf(features: pd.DataFrame) -> tuple[Figure, Stats]:
    """How are gaps distributed? (dormancy is rare; TxGPT's gap-bucket edges)"""
    gaps = features["F3"].dropna()
    gaps = gaps[gaps > 0].sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(gaps.values, np.arange(1, len(gaps) + 1) / len(gaps), color=PALETTE[0], linewidth=2)
    ax.set_xscale("log")
    dormant = SETTINGS["injection"]["dormantDaysMin"] * 24
    ax.axvline(dormant, color=MUTED, linestyle="--", linewidth=1)
    ax.text(dormant * 1.1, 0.1, f"{SETTINGS['injection']['dormantDaysMin']} days\n(planted dormancy)", fontsize=7, color=MUTED)
    ax.set(title="Gaps between an address's transactions", xlabel="Hours since the previous transaction (log scale)",
           ylabel="Share of transactions at or below")
    snapshot_note(fig)
    q = gaps.quantile([0.5, 0.9, 0.99])
    s = [stat("gap_ecdf", f"gap quantile {int(p * 100)}% (hours)", round(float(v), 1)) for p, v in q.items()]
    s.append(stat("gap_ecdf", f"share of gaps over {SETTINGS['injection']['dormantDaysMin']} days", round(float((gaps > dormant).mean()), 4)))
    return fig, s


def fig_pca(features: pd.DataFrame) -> tuple[Figure, Stats]:
    """Does the z-feature space have dense groups and sparse points? (supports DBSCAN beside K-Means)"""
    f = features[features["scorable"] & features["direction"].isin(["in", "out"])]
    z = f[[f"zc_{fid}" for fid in FEATURE_IDS]].fillna(0.0).to_numpy()
    pca = PCA(n_components=2, random_state=SETTINGS["model"]["seed"]).fit(z)
    xy = pca.transform(z)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for d in ("in", "out"):
        m = (f["direction"] == d).to_numpy()
        ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.5, color=DIRECTION_COLOURS[d], edgecolors="none",
                   label="received (in)" if d == "in" else "sent (out)")
    ax.set(title="Own-history z-scores: a dense core and sparse outliers",
           xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
    ax.legend(markerscale=2)
    snapshot_note(fig)
    radius = np.linalg.norm(z, axis=1)
    s = [stat("pca", "explained variance PC1+PC2", round(float(pca.explained_variance_ratio_.sum()), 3)),
         stat("pca", "share of transactions within |z| radius 2", round(float((radius <= 2).mean()), 3), "the dense core")]
    return fig, s


def fig_flag_rate(features: pd.DataFrame) -> tuple[Figure, Stats]:
    """How many transactions break A1's |z| >= 3.5 on the raw data? (early warning for the 20% sanity rule)"""
    f = features[(features["role"] == "main") & features["scorable"]]
    flag_z = SETTINGS["baseline"]["flagAbsZ"]
    share = (f[[f"z_{fid}" for fid in FEATURE_IDS]].abs().max(axis=1) >= flag_z).groupby(f["address"]).mean().sort_values()
    limit = SETTINGS["sanity"]["maxFlaggedShare"]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(range(len(share)), share.values, color=[PALETTE[7] if v > limit else PALETTE[0] for v in share.values], width=0.8)
    ax.axhline(limit, color=INK, linestyle="--", linewidth=1)
    ax.text(0, limit + 0.01, f"sanity limit {limit:.0%}", fontsize=7, color=INK)
    ax.set(title=f"Share of each address's scorable transactions with any |z| >= {flag_z} (A1's threshold)",
           xlabel="DS1 main address (sorted; labels omitted; above the limit in red)", ylabel="Share flagged")
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    snapshot_note(fig)
    s = [stat("flag_rate", f"overall share with |z| >= {flag_z}", round(float((f[[f'z_{x}' for x in FEATURE_IDS]].abs().max(axis=1) >= flag_z).mean()), 3)),
         stat("flag_rate", "addresses above the sanity limit", int((share > limit).sum()), "led to the threshold sweep (CR-03)")]
    return fig, s


# -- DS3 and DS4 analyses ----------------------------------------------------------------------


def fig_elliptic_classes(steps: pd.DataFrame) -> tuple[Figure, Stats]:
    """How imbalanced is Elliptic, and how does that change over time? (temporal split, PR metrics)"""
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    bottom = np.zeros(len(steps))
    for col, colour, label in (("licit", PALETTE[0], "licit"), ("illicit", PALETTE[7], "illicit"), ("unknown", "#c3c2b7", "unknown")):
        a.bar(steps["time_step"], steps[col], bottom=bottom, color=colour, width=0.85, label=label)
        bottom += steps[col].to_numpy()
    split = SETTINGS["elliptic"]["trainLastStep"] + 0.5
    share = steps["illicit"] / (steps["illicit"] + steps["licit"])
    for ax in (a, b):
        ax.axvline(split, color=INK, linestyle="--", linewidth=1)
    a.text(split + 0.3, bottom.max() * 0.9, "test steps", fontsize=7)
    a.set(title="Elliptic (DS3): transactions per time step, by the dataset's own class", ylabel="Transactions")
    a.legend(ncols=3, loc="upper left")
    b.plot(steps["time_step"], share, color=PALETTE[7], marker="o", markersize=3, linewidth=1.5)
    b.set(xlabel="Time step (about two weeks each)", ylabel="Illicit share\nof labelled")
    b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    train, test = steps["time_step"] <= split, steps["time_step"] > split
    lab = steps["illicit"] + steps["licit"]
    s = [stat("elliptic_classes", "illicit share of labelled, training steps", round(float(steps.loc[train, "illicit"].sum() / lab[train].sum()), 4)),
         stat("elliptic_classes", "illicit share of labelled, test steps", round(float(steps.loc[test, "illicit"].sum() / lab[test].sum()), 4)),
         stat("elliptic_classes", "unknown share of all rows", round(float(steps["unknown"].sum() / bottom.sum()), 3))]
    return fig, s


def fig_heist_years(years: pd.DataFrame, families: pd.DataFrame) -> tuple[Figure, Stats]:
    """How imbalanced is BitcoinHeist, and in which years? (fixes the temporal split and the leave-out families)"""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    share = years["positives"] / years["rows"]
    a.bar(years["year"], share, color=PALETTE[7], width=0.7)
    a.set(title="BitcoinHeist (DS4): ransomware share per year", xlabel="Year", ylabel="Share labelled ransomware")
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    top = families.head(8).iloc[::-1]
    b.barh(top["family"], top["rows"], color=PALETTE[0])
    b.set(title="Largest families (the dataset's labels)", xlabel="Rows (address-days)")
    fig.tight_layout()
    s = [stat("heist_years", "overall ransomware share", round(float(years["positives"].sum() / years["rows"].sum()), 4)),
         stat("heist_years", "highest yearly share", f"{share.max():.2%} ({int(years.loc[share.idxmax(), 'year'])})"),
         stat("heist_years", "families", len(families))]
    return fig, s


def fig_heist_features(rows: pd.DataFrame, sample: int = 50000) -> tuple[Figure, Stats]:
    """Do ransomware and white rows differ? Mann-Whitney U with rank-biserial effect size."""
    cols = [c for c in rows.columns if c.startswith(("log10_", "log1p_")) or c in ("length",)]
    seed = SETTINGS["model"]["seed"]
    pos = rows[rows["y"] == 1].sample(min(sample, int((rows["y"] == 1).sum())), random_state=seed)
    neg = rows[rows["y"] == 0].sample(min(sample, int((rows["y"] == 0).sum())), random_state=seed)
    fig, axes = plt.subplots(1, len(cols), figsize=(9, 3.2))
    s: Stats = []
    for ax, col in zip(np.atleast_1d(axes), cols):
        ax.boxplot([neg[col], pos[col]], widths=0.6, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "#cde2fb", "edgecolor": PALETTE[0]}, medianprops={"color": INK})
        ax.set_xticks([1, 2], ["white", "ransom."])
        ax.set_title(col, fontsize=8)
        u = st.mannwhitneyu(pos[col], neg[col])
        r = 1 - 2 * u.statistic / (len(pos) * len(neg))  # rank-biserial: -1..1, 0 = no difference
        s.append(stat("heist_features", f"{col}: rank-biserial effect (ransomware vs white)", round(float(-r), 3),
                      f"Mann-Whitney p={u.pvalue:.1e}; |r|<0.1 is a negligible difference"))
    fig.suptitle("BitcoinHeist features by label (transformed scale; balanced samples)", fontsize=10)
    fig.tight_layout()
    return fig, s


# -- evaluation figures ------------------------------------------------------------------------


def fig_injection_recall(injection: pd.DataFrame) -> tuple[Figure, Stats]:
    """Planted-pattern recall per DS1 method (reporting seeds), against A1's 70% target."""
    order = [m for m in SETTINGS["evaluation"]["explainabilityOrder"] if m in set(injection["method"])]
    inj = injection.set_index("method").loc[order]
    patterns = [("spike", "recall_spike"), ("burst (per tx)", "recall_burst_tx"), ("dormant", "recall_dormant")]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    width = 0.8 / len(order)
    for i, m in enumerate(order):
        xs = np.arange(len(patterns)) + (i - (len(order) - 1) / 2) * width
        ax.bar(xs, [inj.loc[m, f"{c}_mean"] for _, c in patterns], width=width * 0.92, color=METHOD_COLOURS[m],
               yerr=[inj.loc[m, f"{c}_std"] for _, c in patterns], error_kw={"elinewidth": 0.8, "ecolor": MUTED},
               label=METHOD_NAMES.get(m, m))
    target = SETTINGS["injection"]["recallTarget"]
    ax.axhline(target, color=INK, linestyle="--", linewidth=1, label=f"A1 target ({target:.0%})")
    ax.set_xticks(range(len(patterns)), [p for p, _ in patterns])
    ax.set(title="Recall on planted transactions (mean and SD over 5 reporting seeds)", ylabel="Recall", ylim=(0, 1.05))
    ax.legend(ncols=4, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.12))
    s = [stat("injection_recall", f"{m} mean recall", round(float(inj.loc[m, "mean_recall"]), 3)) for m in order]
    return fig, s


def fig_agreement(agreement: pd.DataFrame) -> tuple[Figure, Stats]:
    """Which methods flag the same DS1 transactions? (pairwise Jaccard of flag sets)"""
    pairs = agreement.dropna(subset=["b"])
    methods = [m for m in SETTINGS["evaluation"]["explainabilityOrder"] if m in set(pairs["a"]) | set(pairs["b"])]
    grid = pd.DataFrame(np.eye(len(methods)), index=methods, columns=methods)
    for r in pairs.itertuples():
        grid.loc[r.a, r.b] = grid.loc[r.b, r.a] = r.jaccard
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(grid, cmap=SEQUENTIAL, vmin=0, vmax=1)
    labels = [METHOD_NAMES.get(m, m) for m in methods]
    ax.set_xticks(range(len(methods)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(methods)), labels, fontsize=7)
    ax.grid(False)
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f"{grid.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if grid.iloc[i, j] > 0.6 else INK)
    fig.colorbar(im, ax=ax, label="Jaccard similarity of flag sets", shrink=0.8)
    ax.set_title("Method agreement on the DS1 evaluation period")
    off = pairs["jaccard"]
    return fig, [stat("agreement", "median pairwise Jaccard", round(float(off.median()), 3)),
                 stat("agreement", "most similar pair", f"{pairs.loc[off.idxmax(), 'a']}-{pairs.loc[off.idxmax(), 'b']} ({off.max():.2f})")]


def fig_label_check(check: pd.DataFrame) -> tuple[Figure, Stats]:
    """How far does "unusual" coincide with the labelled classes? PR-AUC as a multiple of chance.

    PR-AUC / prevalence is 1 for a random ranking; below 1, the labelled positives score as *less*
    unusual than the rest. Dots on a log axis with the reference at 1, so above and below chance
    read symmetrically.
    """
    names = {"isolation_forest": "Isolation Forest", "lof": "LOF", "kmeans": "K-Means distance", "dbscan": "DBSCAN distance"}
    titles = {"elliptic_AF": "Elliptic (DS3)", "heist_temporal": "BitcoinHeist (DS4)"}
    rows = check.assign(lift=check["pr_auc"] / check["prevalence"])
    methods = list(dict.fromkeys(rows["method"]))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    offsets = {ds: off for ds, off in zip(dict.fromkeys(rows["dataset"]), (-0.15, 0.15))}
    markers = {ds: m for ds, m in zip(offsets, ("o", "s"))}
    s: Stats = []
    for r in rows.itertuples():
        y = methods.index(r.method) + offsets[r.dataset]
        colour = METHOD_COLOURS.get(r.method, MUTED)
        ax.plot([1, r.lift], [y, y], color=colour, linewidth=1.5)
        ax.plot(r.lift, y, marker=markers[r.dataset], color=colour, markersize=7, linestyle="none")
        ax.text(r.lift * (1.12 if r.lift >= 1 else 0.89), y, f"{r.lift:.2f}x", va="center", fontsize=7,
                ha="left" if r.lift >= 1 else "right", color=INK)
        s.append(stat("label_check", f"{r.dataset} {r.method}: PR-AUC / prevalence", round(float(r.lift), 3),
                      f"PR-AUC {r.pr_auc:.4f}, ROC-AUC {r.roc_auc:.3f}"))
    ax.axvline(1, color=INK, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 40)
    ax.set_yticks(range(len(methods)), [names.get(m, "Random forest (supervised)") for m in methods])
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set(title="Unsupervised \"unusual\" scores against the datasets' own labels",
           xlabel="PR-AUC / prevalence (log scale): 1 = chance, below 1 = worse than random")
    ax.legend(handles=[plt.Line2D([], [], marker=markers[d], color=MUTED, linestyle="none") for d in offsets],
              labels=[titles.get(d, d) for d in offsets], loc="upper right", fontsize=7)
    return fig, s


def fig_pr_curves(curves: dict[str, list[tuple[str, np.ndarray, np.ndarray]]]) -> tuple[Figure, Stats]:
    """Precision-recall curves on each labelled test set (better than ROC under heavy imbalance)."""
    fig, axes = plt.subplots(1, len(curves), figsize=(9, 3.6))
    for ax, (title, items) in zip(np.atleast_1d(axes), curves.items()):
        prevalence = None
        for model, y, p in items:
            precision, recall, _ = precision_recall_curve(y, p)
            ax.plot(recall, precision, color=MODEL_COLOURS[model], linewidth=1.8, label=MODEL_NAMES[model])
            prevalence = float(np.mean(y))
        ax.axhline(prevalence, color=MUTED, linestyle=":", linewidth=1)
        ax.text(0.98, prevalence + 0.02, "chance", fontsize=7, ha="right", color=MUTED)
        ax.set(title=title, xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig, []


def fig_elliptic_f1_by_step(by_step: pd.DataFrame, results: pd.DataFrame) -> tuple[Figure, Stats]:
    """F1 per Elliptic test step at each model's fixed threshold (AF features): shows the drift."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    af = by_step[by_step["feature_set"] == "AF"]
    for model, g in af.groupby("model"):
        ax.plot(g["time_step"], g["f1"], color=MODEL_COLOURS[model], marker="o", markersize=3, linewidth=1.5,
                label=MODEL_NAMES[model])
    ax.set(title="Elliptic test F1 per time step (AF features, thresholds fixed on validation)", xlabel="Time step",
           ylabel="F1 (illicit class)", ylim=(0, 1.02))
    ax.legend(ncols=2, fontsize=7, loc="upper right")
    best = results[results["feature_set"] == "AF"].sort_values("val_f1", ascending=False).iloc[0]
    g = af[af["model"] == best["model"]]
    late = g[g["time_step"] >= 43]["f1"].mean()
    return fig, [stat("elliptic_f1_by_step", f"best model ({best['model']}) mean F1, steps 35-42",
                      round(float(g[g["time_step"] < 43]["f1"].mean()), 3)),
                 stat("elliptic_f1_by_step", f"best model ({best['model']}) mean F1, steps 43-49", round(float(late), 3),
                      "Weber et al. (2019) report the same collapse after a dark-market shutdown")]


def fig_heist_lofo(lofo: pd.DataFrame, results: pd.DataFrame) -> tuple[Figure, Stats]:
    """Recall on each held-out BitcoinHeist family: do models learn behaviour or families?

    Recall alone flatters a model whose threshold flags most rows, so each legend entry carries the
    model's false-positive rate on the random split's test set (the configuration it reuses).
    """
    families = list(dict.fromkeys(lofo["family"]))
    models = list(dict.fromkeys(lofo["model"]))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    rnd = results[results["scheme"] == "random"].set_index("model")
    fpr = rnd["fp"] / (rnd["fp"] + rnd["tn"])
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        g = lofo[lofo["model"] == m].set_index("family").loc[families]
        ax.bar(np.arange(len(families)) + (i - (len(models) - 1) / 2) * width, g["recall"], width=width * 0.92,
               color=MODEL_COLOURS[m], label=f"{MODEL_NAMES[m]} (flags {fpr[m]:.0%} of white rows)")
    ax.set_xticks(range(len(families)), families, fontsize=7)
    ax.set(title="Recall on a ransomware family the model never saw (leave one family out)", ylabel="Recall", ylim=(0, 1))
    ax.legend(ncols=2, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.12))
    s = [stat("heist_lofo", f"{m}: mean held-out-family recall", round(float(lofo.loc[lofo["model"] == m, "recall"].mean()), 3),
              f"false-positive rate on the random split's test set {fpr[m]:.3f}") for m in models]
    return fig, s


# -- building every figure ---------------------------------------------------------------------


def elliptic_curves() -> list[tuple[str, np.ndarray, np.ndarray]] | None:
    """(model, y, scores) on Elliptic's test steps, AF features, from explorer.classify's saved scores."""
    rows_path = PROJECT_ROOT / "data" / "processed" / "elliptic" / "rows.parquet"
    if not rows_path.exists():
        return None
    rows = pd.read_parquet(rows_path, columns=["y", "period"])
    y = rows.loc[(rows["period"] == "test") & rows["y"].notna(), "y"].astype(int).to_numpy()
    out = []
    for model in MODEL_NAMES:
        path = CLASSIFY_DIR / "elliptic" / f"AF_{model}_test_scores.npy"
        if path.exists():
            out.append((model, y, np.load(path)))
    return out or None


def heist_curves() -> list[tuple[str, np.ndarray, np.ndarray]] | None:
    rows_path = PROJECT_ROOT / "data" / "processed" / "bitcoinheist" / "rows.parquet"
    if not rows_path.exists():
        return None
    rows = pd.read_parquet(rows_path, columns=["y", "temporal"])
    y = rows.loc[rows["temporal"] == "test", "y"].to_numpy()
    out = []
    for model in MODEL_NAMES:
        path = CLASSIFY_DIR / "heist" / f"temporal_{model}_test_scores.npy"
        if path.exists():
            out.append((model, y, np.load(path)))
    return out or None


def available() -> dict[str, Callable[[], tuple[Figure, Stats]]]:
    """Every figure whose inputs exist, by file name (without .pdf), in report order."""
    figs: dict[str, Callable[[], tuple[Figure, Stats]]] = {}
    if (MANIFEST_PATH.parent / "features.csv").exists():
        features, transactions = load_ds1()
        figs.update({
            "eda_history_sizes": lambda: fig_history_sizes(features),
            "eda_feature_distributions": lambda: fig_feature_distributions(features),
            "eda_direction_amounts": lambda: fig_direction_amounts(features),
            "eda_feature_correlation": lambda: fig_feature_correlation(features),
            "eda_periodicity": lambda: fig_periodicity(transactions),
            "eda_gap_ecdf": lambda: fig_gap_ecdf(features),
            "eda_pca": lambda: fig_pca(features),
            "eda_flag_rate": lambda: fig_flag_rate(features),
        })
    if (TABLES_DIR / "elliptic_steps.csv").exists():
        figs["eda_elliptic_classes"] = lambda: fig_elliptic_classes(table("elliptic_steps"))
    if (TABLES_DIR / "heist_years.csv").exists():
        figs["eda_heist_years"] = lambda: fig_heist_years(table("heist_years"), table("heist_families"))
    heist_rows = PROJECT_ROOT / "data" / "processed" / "bitcoinheist" / "rows.parquet"
    if heist_rows.exists():
        figs["eda_heist_features"] = lambda: fig_heist_features(pd.read_parquet(heist_rows))
    if (TABLES_DIR / "eval_injection.csv").exists():
        figs["eval_injection_recall"] = lambda: fig_injection_recall(table("eval_injection"))
        figs["eval_agreement"] = lambda: fig_agreement(table("eval_agreement"))
    if (TABLES_DIR / "label_check.csv").exists():
        figs["eval_label_check"] = lambda: fig_label_check(table("label_check"))
    curves = {k: v for k, v in (("Elliptic (DS3), AF features, test steps", elliptic_curves()),
                                ("BitcoinHeist (DS4), temporal split, test years", heist_curves())) if v}
    if curves:
        figs["eval_pr_curves"] = lambda: fig_pr_curves(curves)
    if (TABLES_DIR / "elliptic_f1_by_step.csv").exists():
        figs["eval_elliptic_f1_by_step"] = lambda: fig_elliptic_f1_by_step(table("elliptic_f1_by_step"), table("elliptic_results"))
    if (TABLES_DIR / "heist_lofo.csv").exists():
        figs["eval_heist_lofo"] = lambda: fig_heist_lofo(table("heist_lofo"), table("heist_results"))
    return figs


def run_figures(figures_dir: Path = FIGURES_DIR, tables_dir: Path = TABLES_DIR) -> int:
    style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    all_stats: Stats = []
    for name, build in available().items():
        fig, stats = build()
        fig.savefig(figures_dir / f"{name}.pdf")
        plt.close(fig)
        all_stats += stats
        log.info("Wrote %s.pdf (%d statistics)", name, len(stats))
    pd.DataFrame(all_stats).to_csv(tables_dir / "eda_tests.csv", index=False)
    log.info("Wrote %d figures to %s and %d statistics to eda_tests.csv", len(list(figures_dir.glob('*.pdf'))),
             figures_dir, len(all_stats))
    return 0
