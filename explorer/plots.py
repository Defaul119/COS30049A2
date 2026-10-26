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
FUSION_COLOUR = "#3d4b5c"  # dark slate: a combination of two methods, set apart from the eight single-method hues
METHOD_COLOURS = {"fusion": FUSION_COLOUR,
                  **dict(zip([m for m in SETTINGS["evaluation"]["explainabilityOrder"] if m != "fusion"], PALETTE))}
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


def fig_ds2_coverage(profiles: pd.DataFrame, coverage: pd.DataFrame) -> tuple[Figure, Stats]:
    """What kinds of addresses are in DS2, and does DS2 cover DS1's range? (TxGPT trains on DS2, scores DS1)"""
    axes_of = {"tx_per_day": ("Transactions per day (log scale)", True), "in_share": ("Share of transactions received", False),
               "median_log_size": ("Median |net| (log10 sats)", False), "gap_cv": ("Gap coefficient of variation (log scale)", True)}
    names = list(axes_of)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    ds1, ds2 = profiles[profiles["dataset"] == "DS1"], profiles[profiles["dataset"] == "DS2"]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.6))
    for ax, (x, y) in zip(axes.ravel(), pairs):
        ax.scatter(ds2[x], ds2[y], s=5, color="#c3c2b7", alpha=0.6, edgecolors="none", label=f"DS2 ({len(ds2):,} addresses)")
        ax.scatter(ds1[x], ds1[y], s=30, marker="D", color=PALETTE[1], edgecolors="white", linewidths=0.8,
                   label=f"DS1 ({len(ds1)} main addresses)")
        ax.set(xlabel=axes_of[x][0], ylabel=axes_of[y][0])
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        if axes_of[x][1]:
            ax.set_xscale("log")
        if axes_of[y][1]:
            ax.set_yscale("log")
    axes[0, 0].legend(loc="lower left", fontsize=7, markerscale=1.2)
    fig.suptitle("DS2's addresses span the behaviour of DS1's, so TxGPT trains on the kind of history it scores",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    snapshot_note(fig)
    cov = coverage.set_index("feature")
    s = [stat("ds2_coverage", f"{f}: share of DS1 addresses inside DS2's central range", round(float(cov.loc[f, "ds1_inside_share"]), 3),
              f"DS1 median at DS2's {cov.loc[f, 'ds1_median_percentile_in_ds2']:.0%} percentile")
         for f in cov.index]
    return fig, s


def fig_profiles(summary: pd.DataFrame, tuning: pd.DataFrame) -> tuple[Figure, Stats]:
    """Address profiles (K-Means on 8 address-level features): each profile's standardised centroid."""
    z_cols = [c for c in summary.columns if c.startswith("z_")]
    labels = [f"{r.profile}: {r.description} ({r.ds1_addresses} DS1, {r.ds2_addresses:,} DS2)" for r in summary.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(summary) + 1.6))
    limit = float(np.ceil(np.abs(summary[z_cols].to_numpy()).max()))
    im = ax.imshow(summary[z_cols], cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(z_cols)), [c[2:].replace("_", " ") for c in z_cols], rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.grid(False)
    for i in range(len(summary)):
        for j, c in enumerate(z_cols):
            v = summary[c].iloc[i]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.6 * limit else INK)
    fig.colorbar(im, ax=ax, label="Centroid (SDs from the mean address)", shrink=0.8)
    chosen = tuning[tuning["chosen"]].iloc[0]
    ax.set_title(f"Address profiles of DS1 and DS2: K-Means, k = {int(chosen['k'])} by silhouette ({chosen['silhouette']:.2f})")
    snapshot_note(fig)
    s = [stat("profiles", "chosen k (best silhouette)", int(chosen["k"])),
         stat("profiles", "silhouette range over k", f"{tuning['silhouette'].min():.3f}-{tuning['silhouette'].max():.3f}",
              "below 0.25: the profiles divide a continuum rather than separate distinct types"),
         stat("profiles", "profiles holding a DS1 address", int((summary["ds1_addresses"] > 0).sum()))]
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


def fig_txgpt_sizes(runs: pd.DataFrame, sizes: pd.DataFrame) -> tuple[Figure, Stats]:
    """TxGPT's size sweep (plan §9.1): best validation loss against parameters, 3 seeds per size."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    # Seeds drawn just left of their size (log axis), so the error bar does not hide them.
    ax.scatter(runs["parameters"] * 0.88, runs["best_val_loss"], s=24, color=PALETTE[0], alpha=0.7, edgecolors="none",
               label="one run (seed)", zorder=3)
    ax.errorbar(sizes["parameters"], sizes["mean_val_loss"], yerr=sizes["std_val_loss"].fillna(0), color=INK,
                linewidth=1.5, marker="o", markersize=5, capsize=3, label="mean and SD over seeds", zorder=4)
    chosen = sizes[sizes["chosen"]].iloc[0]
    ax.scatter([chosen["parameters"]], [chosen["mean_val_loss"]], s=140, facecolors="none", edgecolors=PALETTE[1],
               linewidths=2, label="chosen: lowest mean validation loss", zorder=5)
    for r in sizes.itertuples():
        ax.annotate(f"{r.size.upper()}", (r.parameters, r.mean_val_loss), textcoords="offset points", xytext=(8, 6),
                    fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set(title="TxGPT: validation loss by model size", xlabel="Parameters (log scale)",
           ylabel="Best validation loss (nats per token)")
    ax.legend(fontsize=7, loc="best")
    s = [stat("txgpt_sizes", f"size {r.size}: mean best validation loss", round(float(r.mean_val_loss), 4),
              f"{int(r.parameters):,} parameters; SD {r.std_val_loss:.4f} over {int(r.runs)} seeds") for r in sizes.itertuples()]
    s.append(stat("txgpt_sizes", "chosen size", chosen["size"], "lowest mean validation loss (plan §9.1)"))
    return fig, s


# -- DS1 extensions (24 Oct): calibration, fusion, uncertainty, strength, time, DS2 ------------------


def method_order(names: Any) -> list[str]:
    """Methods in the report's fixed order (explainabilityOrder), keeping only those present."""
    present = set(names)
    return [m for m in SETTINGS["evaluation"]["explainabilityOrder"] if m in present]


def fig_conformal(results: pd.DataFrame) -> tuple[Figure, Stats]:
    """Do the regression ranges cover what they claim? Raw quantile models against conformal calibration."""
    q = results[results["model"].str.startswith("quantile")].copy()
    q["kind"] = np.where(q["model"].str.endswith("conformal"), "conformal", "raw")
    groups = list(dict.fromkeys(zip(q["target"], q["nominal"])))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (target, nominal) in enumerate(groups):
        ax.hlines(nominal, i - 0.32, i + 0.32, color=INK, linewidth=1.2, label="stated coverage" if i == 0 else None)
        for dx, kind, colour, face in ((-0.12, "raw", PALETTE[1], "none"), (0.12, "conformal", PALETTE[0], PALETTE[0])):
            r = q[(q["target"] == target) & (q["nominal"] == nominal) & (q["kind"] == kind)].iloc[0]
            ax.scatter(i + dx, r["coverage"], s=46, marker="o", facecolors=face, edgecolors=colour, linewidths=1.6,
                       zorder=3, label=("raw quantile models" if kind == "raw" else "conformal (CQR)") if i == 0 else None)
    ax.set_xticks(range(len(groups)), [f"{t}\n{n:.0%} range" for t, n in groups])
    ax.set(title="Regression ranges: coverage on the DS1 evaluation period", ylabel="Share of rows inside the range",
           ylim=(min(0.7, float(q["coverage"].min()) - 0.03), 1.01))
    ax.legend(fontsize=7, loc="lower right")
    s = []
    for r in q.itertuples():
        s.append(stat("conformal", f"{r.target} {r.nominal:.0%} range, {r.kind}: coverage", round(float(r.coverage), 3),
                      f"lowest address {r.coverage_address_min:.2f}; mean width {r.mean_width:.3f} log10 units"))
    return fig, s


def pareto(points: pd.DataFrame) -> pd.DataFrame:
    """The upper-left envelope: configurations no other one beats on both recall (higher) and flag rate (lower)."""
    ordered = points.sort_values(["flag_rate", "mean_recall"], ascending=[True, False])
    keep, best = [], -np.inf
    for idx, r in ordered.iterrows():
        if r["mean_recall"] > best:
            keep.append(idx)
            best = r["mean_recall"]
    return ordered.loc[keep]


def fig_fusion_frontier(tuning: pd.DataFrame, chosen: dict[str, Any]) -> tuple[Figure, Stats]:
    """Recall against flag rate (tuning plants): single methods, the motivating pair and the chosen pair."""
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    singles = tuning[tuning["components"] == 1]
    for name, part in singles.groupby("pair"):
        f = pareto(part)
        ax.plot(f["flag_rate"], f["mean_recall"], color=METHOD_COLOURS.get(name, MUTED), marker="o", markersize=3,
                linewidth=1.4, label=f"{METHOD_NAMES.get(name, name)} alone")
    chosen_pair = " + ".join(chosen["components"])
    for pair, style_ in ((chosen_pair, "-"), ("baseline + regression", "--")):
        part = tuning[tuning["pair"] == pair]
        if len(part) and (style_ == "-" or pair != chosen_pair):
            f = pareto(part)
            ax.plot(f["flag_rate"], f["mean_recall"], color=FUSION_COLOUR, linestyle=style_, marker="s", markersize=3,
                    linewidth=1.6, label=f"fusion: {pair}")
    ax.axvline(chosen["budget"], color=MUTED, linestyle=":", linewidth=1)
    ax.annotate(f"flag budget ({chosen['budget_method']})", (chosen["budget"], 0.02), textcoords="offset points",
                xytext=(4, 0), fontsize=7, color=MUTED, rotation=90, va="bottom")
    ax.scatter([chosen["tuning"]["flag_rate"]], [chosen["tuning"]["mean_recall"]], s=130, facecolors="none",
               edgecolors=PALETTE[1], linewidths=2, zorder=5, label="saved fusion")
    ax.set(title="Recall at each flag rate: single methods and fused pairs (tuning plants)",
           xlabel="Share of real evaluation transactions flagged", ylabel="Mean planted recall", ylim=(0, 1))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(fontsize=7, loc="lower right")
    s = [stat("fusion", "saved pair", chosen_pair, f"multipliers {chosen['multipliers']}"),
         stat("fusion", "tuning mean recall of the saved fusion", round(float(chosen["tuning"]["mean_recall"]), 3),
              f"flag rate {chosen['tuning']['flag_rate']:.4f}, budget {chosen['budget']:.4f}")]
    for name, part in singles.groupby("pair"):
        within = part[part["flag_rate"] <= chosen["budget"]]
        if len(within):
            s.append(stat("fusion", f"{name} alone: best tuning mean recall within the budget",
                          round(float(within["mean_recall"].max()), 3)))
    return fig, s


def fig_significance(sig: pd.DataFrame) -> tuple[Figure, Stats]:
    """Mean planted recall with address-level bootstrap intervals, and each method's difference from the reference."""
    order = sig.sort_values("mean_recall")["method"].tolist()
    t = sig.set_index("method").loc[order]
    reference = t["reference"].iloc[0]
    margin = SETTINGS["evaluation"]["tieMarginPoints"] / 100
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    y = np.arange(len(order))
    colours = [METHOD_COLOURS.get(m, MUTED) for m in order]
    a.errorbar(t["mean_recall"], y, xerr=[t["mean_recall"] - t["mean_recall_ci_low"], t["mean_recall_ci_high"] - t["mean_recall"]],
               fmt="none", ecolor=MUTED, elinewidth=1, capsize=2)
    a.scatter(t["mean_recall"], y, c=colours, s=36, zorder=3)
    a.set_yticks(y, [METHOD_NAMES.get(m, m) for m in order], fontsize=7)
    a.set(title="Mean planted recall, 95% interval", xlabel="Mean recall (reporting plants)", xlim=(0, 1))
    a.grid(axis="x", color=GRID)
    b.axvspan(-margin, margin, color=GRID, alpha=0.7, label=f"selection tie margin (±{margin:.0%})")
    b.axvline(0, color=INK, linewidth=0.8)
    b.errorbar(t["diff_mean_recall"], y, xerr=[t["diff_mean_recall"] - t["diff_ci_low"], t["diff_ci_high"] - t["diff_mean_recall"]],
               fmt="none", ecolor=MUTED, elinewidth=1, capsize=2)
    b.scatter(t["diff_mean_recall"], y, c=colours, s=36, zorder=3)
    b.set(title=f"Difference from {METHOD_NAMES.get(reference, reference)}", xlabel="Difference in mean recall")
    b.grid(axis="x", color=GRID)
    b.legend(fontsize=7, loc="upper left")
    s = [stat("significance", f"{m}: mean recall", round(float(t.loc[m, "mean_recall"]), 3),
              f"95% CI {t.loc[m, 'mean_recall_ci_low']:.3f}-{t.loc[m, 'mean_recall_ci_high']:.3f}; minus {reference} "
              f"{t.loc[m, 'diff_mean_recall']:+.3f} ({t.loc[m, 'diff_ci_low']:+.3f} to {t.loc[m, 'diff_ci_high']:+.3f}), "
              f"p {t.loc[m, 'p_value']:.3f}" if m != reference else "the reference")
         for m in reversed(order)]
    return fig, s


def fig_sensitivity(sens: pd.DataFrame) -> tuple[Figure, Stats]:
    """How strong must each pattern be before each method sees it? Recall against plant strength."""
    panels = [("spike", "Spike: multiple of the usual amount", True), ("burst", "Burst: transactions in 24 hours", False),
              ("dormant", "Dormancy: days silent (range mid)", True)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    target = SETTINGS["injection"]["recallTarget"]
    for ax, (pattern, xlabel, log_x) in zip(axes, panels):
        part = sens[sens["pattern"] == pattern]
        for m in method_order(part["method"]):
            line = part[part["method"] == m].sort_values("x")
            ax.plot(line["x"], line["recall"], marker="o", markersize=3, linewidth=1.4, color=METHOD_COLOURS.get(m, MUTED),
                    label=METHOD_NAMES.get(m, m))
        default = part[part["default"]]["x"].iloc[0]
        ax.axvline(default, color=MUTED, linestyle=":", linewidth=1)
        ax.axhline(target, color=INK, linestyle="--", linewidth=0.8)
        if log_x:
            ax.set_xscale("log")
            ticks = sorted(part["x"].unique())
            ax.set_xticks(ticks, [f"{v:.0f}" for v in ticks])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set(title=pattern.capitalize(), xlabel=xlabel + (" (log)" if log_x else ""), ylim=(0, 1.02))
    axes[0].set_ylabel("Recall (reporting seeds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, fontsize=7, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Recall against pattern strength (dotted: the reported strength; dashed: A1's 70% target)", fontsize=10)
    s = []
    for pattern, _, _ in panels:
        part = sens[sens["pattern"] == pattern]
        for m in method_order(part["method"]):
            line = part[part["method"] == m].sort_values("x")
            reached = line[line["recall"] >= target]
            s.append(stat("sensitivity", f"{m}: weakest {pattern} reaching the 70% target",
                          reached["strength"].iloc[0] if len(reached) else "none in the sweep",
                          f"recall at the strongest: {line['recall'].iloc[-1]:.2f}"))
    return fig, s


def fig_walkforward(wf: pd.DataFrame) -> tuple[Figure, Stats]:
    """Does the ranking depend on where the time split falls? Recall and flag rate at each cut-off."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    for m in method_order(wf["method"]):
        line = wf[wf["method"] == m].sort_values("fit_fraction")
        kw = {"color": METHOD_COLOURS.get(m, MUTED), "marker": "o", "markersize": 3, "linewidth": 1.4}
        a.plot(line["fit_fraction"], line["mean_recall"], label=METHOD_NAMES.get(m, m), **kw)
        b.plot(line["fit_fraction"], line["flag_rate"], **kw)
    reported = SETTINGS["model"]["fitFraction"]
    for ax in (a, b):
        ax.axvline(reported, color=MUTED, linestyle=":", linewidth=1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("Share of each history used for fitting (dotted: reported)")
    a.set(title="Mean planted recall", ylabel="Mean recall", ylim=(0, 1))
    b.set(title="Real transactions flagged", ylabel="Flag rate")
    b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    spread = wf.groupby("method")["mean_recall"].agg(lambda v: v.max() - v.min())
    best = wf.loc[wf.groupby("fit_fraction")["mean_recall"].idxmax()]
    s = [stat("walkforward", f"{m}: range of mean recall across cut-offs", round(float(spread[m]), 3))
         for m in method_order(wf["method"])]
    s += [stat("walkforward", f"best method at fit fraction {r.fit_fraction:.0%}", r.method, f"mean recall {r.mean_recall:.3f}")
          for r in best.itertuples()]
    return fig, s


def fig_ds2_check(ds2: pd.DataFrame, ds1_sanity: pd.DataFrame, ds1_addresses: int) -> tuple[Figure, Stats]:
    """Do DS1's results carry over to ~2,900 DS2 histories? Recall and the sanity rule, DS1 against DS2."""
    order = ds2.sort_values("ds2_mean_recall")["method"].tolist()
    t = ds2.set_index("method").loc[order]
    sanity = ds1_sanity.set_index("method")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    y = np.arange(len(order))
    for ax, ds1_col, ds2_col, title, xlabel in (
            (a, "ds1_mean_recall", "ds2_mean_recall", "Mean planted recall", "Mean recall"),
            (b, None, "ds2_share_addresses_over_limit", "Addresses breaking the sanity rule", "Share of addresses")):
        ds1_vals = (t[ds1_col] if ds1_col else
                    pd.Series([sanity.loc[m, "addresses_over_limit"] / ds1_addresses if m in sanity.index else np.nan
                               for m in order], index=order))
        ax.hlines(y, np.fmin(ds1_vals, t[ds2_col]), np.fmax(ds1_vals, t[ds2_col]), color=GRID, linewidth=2.5)
        ax.scatter(ds1_vals, y, s=36, facecolors="none", edgecolors=MUTED, linewidths=1.4, zorder=3, label="DS1 (25 addresses)")
        ax.scatter(t[ds2_col], y, s=36, c=[METHOD_COLOURS.get(m, MUTED) for m in order], zorder=4,
                   label=f"DS2 ({int(t['ds2_eval_rows'].iloc[0]):,} evaluation rows)")
        if ds2_col == "ds2_mean_recall":
            ax.errorbar(t[ds2_col], y, xerr=[t[ds2_col] - t["ds2_mean_recall_ci_low"], t["ds2_mean_recall_ci_high"] - t[ds2_col]],
                        fmt="none", ecolor=MUTED, elinewidth=1, capsize=2, zorder=2)
        ax.set(title=title, xlabel=xlabel, xlim=(0, 1))
        ax.grid(axis="x", color=GRID)
    a.set_yticks(y, [METHOD_NAMES.get(m, m) for m in order], fontsize=7)
    b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a.legend(fontsize=7, loc="lower right")
    fig.suptitle("The frozen DS1 methods on DS2 histories that meet DS1's address rule", fontsize=10)
    s = [stat("ds2_check", f"{m}: DS2 mean recall", round(float(t.loc[m, "ds2_mean_recall"]), 3),
              f"95% CI {t.loc[m, 'ds2_mean_recall_ci_low']:.3f}-{t.loc[m, 'ds2_mean_recall_ci_high']:.3f}; "
              f"DS1 {t.loc[m, 'ds1_mean_recall']:.3f}; DS2 flag rate {t.loc[m, 'ds2_flag_rate']:.3f}")
         for m in reversed(order)]
    ranks = t[["ds1_mean_recall", "ds2_mean_recall"]].dropna()
    if len(ranks) > 2:
        rho = st.spearmanr(ranks["ds1_mean_recall"], ranks["ds2_mean_recall"]).statistic
        s.append(stat("ds2_check", "Spearman correlation of method rankings, DS1 vs DS2", round(float(rho), 3),
                      f"{len(ranks)} methods"))
    return fig, s


def fig_ds10_check(ds10: pd.DataFrame, ds1_sanity: pd.DataFrame, ds1_addresses: int) -> tuple[Figure, Stats]:
    """Every model, frozen, on the teammate's independent snapshot (DS10): recall beside DS1 and DS2, and sanity."""
    order = ds10.sort_values("ds10_mean_recall")["method"].tolist()
    t = ds10.set_index("method").loc[order]
    sanity = ds1_sanity.set_index("method")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    y = np.arange(len(order))
    a.scatter(t["ds1_mean_recall"], y, s=36, facecolors="none", edgecolors=MUTED, linewidths=1.4, zorder=3,
              label="DS1 (25 addresses)")
    if "ds2_mean_recall" in t:
        a.scatter(t["ds2_mean_recall"], y, s=14, c=MUTED, marker="s", zorder=3, label="DS2 (2,933 addresses)")
    a.errorbar(t["ds10_mean_recall"], y, xerr=[t["ds10_mean_recall"] - t["ds10_mean_recall_ci_low"],
                                              t["ds10_mean_recall_ci_high"] - t["ds10_mean_recall"]],
               fmt="none", ecolor=MUTED, elinewidth=1, capsize=2, zorder=2)
    a.scatter(t["ds10_mean_recall"], y, s=40, c=[METHOD_COLOURS.get(m, MUTED) for m in order], zorder=4,
              label=f"DS10 ({int(t['ds10_eval_rows'].iloc[0]):,} evaluation rows)")
    a.set(title="Mean planted recall", xlabel="Mean recall", xlim=(0, 1))
    ds1_share = pd.Series([sanity.loc[m, "addresses_over_limit"] / ds1_addresses if m in sanity.index else np.nan
                           for m in order], index=order)
    b.hlines(y, np.fmin(ds1_share, t["ds10_share_addresses_over_limit"]),
             np.fmax(ds1_share, t["ds10_share_addresses_over_limit"]), color=GRID, linewidth=2.5)
    b.scatter(ds1_share, y, s=36, facecolors="none", edgecolors=MUTED, linewidths=1.4, zorder=3)
    b.scatter(t["ds10_share_addresses_over_limit"], y, s=40, c=[METHOD_COLOURS.get(m, MUTED) for m in order], zorder=4)
    b.set(title="Addresses breaking the sanity rule", xlabel="Share of addresses", xlim=(0, 1))
    b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for ax in (a, b):
        ax.grid(axis="x", color=GRID)
    a.set_yticks(y, [METHOD_NAMES.get(m, m) for m in order], fontsize=7)
    a.legend(fontsize=7, loc="lower right")
    fig.suptitle("Every model, frozen as tuned on DS1, on a teammate's independent snapshot (DS10)", fontsize=10)
    s = [stat("ds10_check", f"{m}: DS10 mean recall", round(float(t.loc[m, "ds10_mean_recall"]), 3),
              f"95% CI {t.loc[m, 'ds10_mean_recall_ci_low']:.3f}-{t.loc[m, 'ds10_mean_recall_ci_high']:.3f}; "
              f"DS1 {t.loc[m, 'ds1_mean_recall']:.3f}; DS10 flag rate {t.loc[m, 'ds10_flag_rate']:.3f}")
         for m in reversed(order)]
    for other in ("ds1", "ds2"):
        col = f"{other}_mean_recall"
        if col in t:
            ranks = t[[col, "ds10_mean_recall"]].dropna()
            if len(ranks) > 2:
                rho = st.spearmanr(ranks[col], ranks["ds10_mean_recall"]).statistic
                s.append(stat("ds10_check", f"Spearman correlation of method rankings, {other.upper()} vs DS10",
                              round(float(rho), 3), f"{len(ranks)} methods"))
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


def fig_stats_methods(seeds: pd.DataFrame, summary: pd.DataFrame) -> tuple[Figure, Stats]:
    """Each method's mean planted recall over many plant seeds (explorer/distribution.py), per data set."""
    sets = [d for d in ("DS1", "DS2", "DS10") if d in set(seeds["dataset"])]
    order = summary[summary["dataset"] == sets[0]].sort_values("mean")["method"].tolist()
    fig, axes = plt.subplots(1, len(sets), figsize=(3.3 * len(sets), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    s = []
    for ax, dataset in zip(axes, sets):
        part = seeds[seeds["dataset"] == dataset]
        data = [part.loc[part["method"] == m, "mean_recall"].to_numpy() for m in order]
        box = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                         medianprops={"color": INK, "linewidth": 1})
        for patch, m in zip(box["boxes"], order):
            patch.set(facecolor=METHOD_COLOURS.get(m, MUTED), alpha=0.75, edgecolor=MUTED)
        base = part.loc[part["method"] == "baseline", "mean_recall"].mean()
        ax.axvline(base, color=INK, linewidth=0.8, linestyle="--")
        n = part["seed"].nunique()
        ax.set(title=f"{dataset}: {n} plant seeds", xlabel="Mean planted recall", xlim=(0, 1))
        ax.grid(axis="x", color=GRID)
        t = summary[summary["dataset"] == dataset].set_index("method")
        s += [stat("stats_methods", f"{dataset} {m}: mean recall", round(float(t.at[m, "mean"]), 3),
                   f"sd {t.at[m, 'sd']:.3f}, 95% of seeds {t.at[m, 'ci_low']:.3f}-{t.at[m, 'ci_high']:.3f}; beats the "
                   f"baseline on {t.at[m, 'beats_baseline_share']:.0%} of seeds; mean rank {t.at[m, 'mean_rank']:.1f}")
              for m in reversed(order) if m in t.index]
    axes[0].set_yticks(np.arange(1, len(order) + 1), [METHOD_NAMES.get(m, m) for m in order], fontsize=7)
    fig.suptitle("Planted-pattern recall over many seeds (dashed: the baseline's mean)", fontsize=9)
    return fig, s


def fig_stats_classifiers(summary: pd.DataFrame) -> tuple[Figure, Stats]:
    """Classifier test F1: mean over training seeds, with a 95% bootstrap interval over test rows."""
    schemes = summary["scheme"].unique().tolist()
    fig, axes = plt.subplots(1, len(schemes), figsize=(3.3 * len(schemes), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    names = {"logistic": "Logistic regression", "knn": "k-nearest neighbours", "randomForest": "Random forest",
             "histGradientBoosting": "Gradient boosting"}
    s = []
    for ax, scheme in zip(axes, schemes):
        t = summary[summary["scheme"] == scheme].reset_index(drop=True)
        y = np.arange(len(t))
        ax.errorbar(t["f1_mean"], y, xerr=[(t["f1_mean"] - t["f1_ci_low"]).clip(lower=0),
                                           (t["f1_ci_high"] - t["f1_mean"]).clip(lower=0)],
                    fmt="o", color=INK, ecolor=MUTED, elinewidth=1, capsize=2, markersize=4)
        ax.set_yticks(y, [names.get(m, m) for m in t["model"]], fontsize=7)
        ax.set(title=scheme, xlabel="Test F1", xlim=(0, 1))
        ax.grid(axis="x", color=GRID)
        s += [stat("stats_classifiers", f"{scheme} {r.model}: F1", round(float(r.f1_mean), 3),
                   f"seed sd {r.f1_seed_sd:.3f} ({r.seeds} seeds); bootstrap 95% CI {r.f1_ci_low:.3f}-{r.f1_ci_high:.3f}")
              for r in t.itertuples()]
    fig.suptitle("Classifiers: mean over training seeds, 95% bootstrap interval over test rows", fontsize=9, y=1.02)
    return fig, s


WRITER_LABELS = {"text150m": "152M, round 1", "qwen3b": "Qwen 3B, round 1", "qwen3b_v2": "Qwen 3B, round 2",
                 "text150m_v2": "152M, round 2"}


def fig_stats_writers(summary: pd.DataFrame) -> tuple[Figure, Stats]:
    """Fully correct rate of each Assistant writer on every test set, with 95% bootstrap intervals (lm/eval_stats.py)."""
    sets = [s for s in summary["test_set"].unique() if not s.startswith("glossary")]
    writers = [w for w in WRITER_LABELS if w in set(summary["writer"])]
    colours = dict(zip(WRITER_LABELS, PALETTE))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(sets) + 1.2))
    y = np.arange(len(sets))[::-1]
    offsets = np.linspace(-0.22, 0.22, len(writers)) if len(writers) > 1 else [0.0]
    s = []
    for off, w in zip(offsets, writers):
        t = summary[summary["writer"] == w].set_index("test_set").reindex(sets)
        ax.errorbar(t["fully_correct"], y + off, xerr=[(t["fully_correct"] - t["ci_low"]).clip(lower=0),
                                                        (t["ci_high"] - t["fully_correct"]).clip(lower=0)],
                    fmt="o", color=colours[w], ecolor=colours[w], elinewidth=1, capsize=2, markersize=4,
                    label=WRITER_LABELS[w])
        s += [stat("stats_writers", f"{n} {w}: fully correct", round(float(t.at[n, "fully_correct"]), 3),
                   f"95% CI {t.at[n, 'ci_low']:.3f}-{t.at[n, 'ci_high']:.3f}; {int(t.at[n, 'prompts'])} prompts")
              for n in sets if not np.isnan(t.at[n, "fully_correct"])]
    ax.set_yticks(y, sets, fontsize=7)
    ax.set(xlabel="Fully correct (95% bootstrap interval over the set's prompts)", xlim=(0, 1))
    ax.grid(axis="x", color=GRID)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Explanation writers on every test set", fontsize=9)
    return fig, s


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
    profiles_path = PROJECT_ROOT / "data" / "processed" / "profiles.csv"
    if profiles_path.exists() and (TABLES_DIR / "ds2_coverage.csv").exists():
        figs["eda_ds2_coverage"] = lambda: fig_ds2_coverage(pd.read_csv(profiles_path), table("ds2_coverage"))
        figs["eda_profiles"] = lambda: fig_profiles(table("profiles"), table("profiles_tuning"))
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
    if (TABLES_DIR / "txgpt_runs.csv").exists():
        figs["eval_txgpt_sizes"] = lambda: fig_txgpt_sizes(table("txgpt_runs"), table("txgpt_sizes"))
    if (TABLES_DIR / "heist_lofo.csv").exists():
        figs["eval_heist_lofo"] = lambda: fig_heist_lofo(table("heist_lofo"), table("heist_results"))
    if (TABLES_DIR / "regression_results.csv").exists() and "nominal" in table("regression_results"):
        figs["eval_conformal"] = lambda: fig_conformal(table("regression_results"))
    fusion_model = PROJECT_ROOT / "models" / "fusion.joblib"
    if (TABLES_DIR / "tuning_fusion.csv").exists() and fusion_model.exists():
        import joblib

        figs["eval_fusion_frontier"] = lambda: fig_fusion_frontier(
            table("tuning_fusion"), {k: v for k, v in joblib.load(fusion_model).items() if k != "method"})
    if (TABLES_DIR / "eval_significance.csv").exists():
        figs["eval_significance"] = lambda: fig_significance(table("eval_significance"))
    if (TABLES_DIR / "eval_sensitivity.csv").exists():
        figs["eval_sensitivity"] = lambda: fig_sensitivity(table("eval_sensitivity"))
    if (TABLES_DIR / "eval_walkforward.csv").exists():
        figs["eval_walkforward"] = lambda: fig_walkforward(table("eval_walkforward"))
    if (TABLES_DIR / "eval_ds2.csv").exists() and (TABLES_DIR / "eval_sanity.csv").exists():
        evaluated = int((table("split_ds1")["eval"] > 0).sum())
        figs["eval_ds2_check"] = lambda: fig_ds2_check(table("eval_ds2"), table("eval_sanity"), evaluated)
    if (TABLES_DIR / "eval_ds10.csv").exists() and (TABLES_DIR / "eval_sanity.csv").exists():
        evaluated = int((table("split_ds1")["eval"] > 0).sum())
        figs["eval_ds10_check"] = lambda: fig_ds10_check(table("eval_ds10"), table("eval_sanity"), evaluated)
    if (TABLES_DIR / "stats_methods_seeds.csv").exists():
        figs["stats_methods"] = lambda: fig_stats_methods(table("stats_methods_seeds"), table("stats_methods_summary"))
    if (TABLES_DIR / "stats_classifiers_summary.csv").exists():
        figs["stats_classifiers"] = lambda: fig_stats_classifiers(table("stats_classifiers_summary"))
    if (TABLES_DIR / "stats_writers_summary.csv").exists():
        figs["stats_writers"] = lambda: fig_stats_writers(table("stats_writers_summary"))
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
