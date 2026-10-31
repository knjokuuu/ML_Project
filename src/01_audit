"""Data audit: shape, variable types, missing values, duplicates, target
distribution, outliers. Prints a report and saves tables and figures to results/.

Run from the repo root:  python src/01_audit.py [path/to/games.csv]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # write PNG files, no window needed
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_prep import (LEAKAGE, NOT_USED, SNAPSHOT_RISK, VARIABLE_TYPES, build_features,
                       column_role, load_raw, make_labelled_subset, top_values)

path = sys.argv[1] if len(sys.argv) > 1 else "data/games.csv"
RESULTS = Path("results")
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 100)

# Chart colours: two categorical slots (blue, orange) and neutral text/grid tones.
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e3e2de"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})

# ================================================================ 1. dataset shape
raw = load_raw(path)
print("=== 1. DATASET SHAPE ===")
print(f"raw data: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
print("duplicate rows:", raw.duplicated().sum(), "| duplicate AppIDs:", raw["AppID"].duplicated().sum())
print("Windows missing (should be 0 if the header fix worked):", raw["Windows"].isna().sum())

sub = make_labelled_subset(raw)
print(f"\nmodelling subset (>= 50 reviews): {sub.shape[0]:,} rows "
      f"({sub.shape[0] / raw.shape[0]:.1%} of the raw data)")
print("duplicate AppIDs in subset:", sub["AppID"].duplicated().sum())

# ================================================================ 2. variable types
print("\n=== 2. VARIABLE TYPES (every raw column) ===")
variables = pd.DataFrame({
    "column": raw.columns,
    "pandas_dtype": [str(raw[c].dtype) for c in raw.columns],
    "variable_type": [VARIABLE_TYPES[c] for c in raw.columns],
    "role": [column_role(c) for c in raw.columns],
    "missing_pct_subset": [round(sub[c].isna().mean() * 100, 1) for c in raw.columns],
    "n_unique_subset": [sub[c].nunique() for c in raw.columns],
})
print(variables.to_string(index=False))
variables.to_csv(RESULTS / "audit_variables.csv", index=False)
print("\nvariable types, grouped:")
print(variables["variable_type"].str.replace(r" \(.*\)", "", regex=True).value_counts().to_string())
print("\nroles:")
print(variables["role"].value_counts().to_string())

# ================================================================ 3. missing values
print("\n=== 3. MISSING VALUES (modelling subset, share of rows) ===")
print(sub.isna().mean().sort_values(ascending=False).head(15).round(3).to_string())
print("\nplaceholder zeros that isna() will not catch:")
for c in ["Required age", "Achievements", "Metacritic score", "User score", "Price"]:
    print(f"  {c}: share == 0 -> {(sub[c] == 0).mean():.3f}")
print("  Estimated owners '0 - 0':", round((sub["Estimated owners"] == "0 - 0").mean(), 3))
print("  Empty Supported languages:", round(sub["Supported languages"].fillna("[]").isin(["[]", ""]).mean(), 3))

# ================================================================ 4. target distribution
counts = sub["well_received"].value_counts().sort_index()
share = counts / counts.sum()
print("\n=== 4. TARGET DISTRIBUTION ===")
print(pd.DataFrame({"games": counts, "share": share.round(3)}).rename(
    index={0: "0 not well received", 1: "1 well received"}).to_string())
print("majority-class baseline accuracy:", round(share.max(), 3))

fig, ax = plt.subplots(figsize=(6.4, 4.4))
bars = ax.bar(["Not well received\n(< 80% positive)", "Well received\n(>= 80% positive)"],
              counts.values, color=[ORANGE, BLUE], width=0.55)
for bar, n, p in zip(bars, counts.values, share.values):
    ax.text(bar.get_x() + bar.get_width() / 2, n + counts.max() * 0.015, f"{n:,}  ({p:.1%})",
            ha="center", va="bottom", color=INK, fontsize=11)
ax.set_ylim(0, counts.max() * 1.15)
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
ax.set_ylabel("Number of games")
ax.set_title(f"{share[1]:.1%} of the {len(sub):,} modelled games are well received", pad=12)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(FIGURES / "01_class_distribution.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ================================================================ 5. outliers
print("\n=== 5. OUTLIERS / DISTRIBUTIONS (modelling subset) ===")
print(sub[["Price", "Required age", "Achievements", "Peak CCU", "total"]].describe(
    percentiles=[.5, .9, .99]).round(2).to_string())
print("Price > 100:", (sub["Price"] > 100).sum(), "| Achievements > 1000:", (sub["Achievements"] > 1000).sum())

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=False)
for ax, col, label in zip(axes, ["Price", "Achievements", "total"],
                          ["Price (USD)", "Achievements", "Reviews (positive + negative)"]):
    ax.hist(np.log10(1 + sub[col]), bins=40, color=BLUE, edgecolor="#fcfcfb", linewidth=0.5)
    med = sub[col].median()
    ax.axvline(np.log10(1 + med), color=ORANGE, linewidth=2)
    ax.set_xlabel(f"log10(1 + {label})")
    ax.set_title(f"{label.split(' (')[0]}: median {med:,.4g}, max {sub[col].max():,.0f}",
                 fontsize=10.5)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Number of games")
fig.suptitle("Price, achievements and review counts are heavily right-skewed "
             "(orange line = median)", x=0.01, ha="left", fontweight="bold", fontsize=12.5)
fig.tight_layout()
fig.savefig(FIGURES / "03_numeric_distributions.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ================================================================ 6. release-time features
X, y = build_features(sub)
num = X.drop(columns=["genres_list", "categories_list"])
print("\n=== 6. RELEASE-TIME FEATURES ===")
print(num.describe().round(2).to_string())
print("release_year range:", X["release_year"].min(), "-", X["release_year"].max(),
      "| unparsed dates:", X["release_year"].isna().sum())
print("top genres:", top_values(X["genres_list"], 15))
print("top categories:", top_values(X["categories_list"], 15))
print("\ncorrelation of numeric features with target:")
print(num.corrwith(y).round(3).sort_values().to_string())

by_year = y.groupby(X["release_year"]).agg(["mean", "size"])
print("\ntarget rate by release year:")
print(by_year.tail(15).round(3).to_string())

yrs = by_year[by_year["size"] >= 100]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.plot(yrs.index, yrs["mean"] * 100, color=BLUE, linewidth=2, marker="o", markersize=7,
        markeredgecolor="#fcfcfb", markeredgewidth=1.5)
ax.axhline(share[1] * 100, color=MUTED, linestyle="--", linewidth=1.2)
ax.text(yrs.index.min(), share[1] * 100 + 1.2, f"overall {share[1]:.1%}", color=MUTED, fontsize=10)
first, last = yrs.iloc[0], yrs.iloc[-1]
ax.annotate(f"{first['mean']:.0%}", (yrs.index[0], first["mean"] * 100), textcoords="offset points",
            xytext=(0, -16), ha="center", color=INK)
ax.annotate(f"{last['mean']:.0%}", (yrs.index[-1], last["mean"] * 100), textcoords="offset points",
            xytext=(0, 10), ha="center", color=INK)
lo, hi = yrs["mean"].min() * 100, yrs["mean"].max() * 100
ax.set_ylim(max(0, 5 * np.floor((lo - 8) / 5)), min(100, 5 * np.ceil((hi + 8) / 5)))
ax.set_xlabel("Release year (years with at least 100 games)")
ax.set_ylabel("% of games well received")
ax.set_title(f"Well-received share by release year: {first['mean']:.0%} in {yrs.index[0]} to "
             f"{last['mean']:.0%} in {yrs.index[-1]}", pad=12)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(FIGURES / "02_target_rate_by_year.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ================================================================ 7. decisions
print("\n=== 7. COLUMN DECISIONS ===")
print("Excluded as leakage:", LEAKAGE)
print("Excluded as ID/text/media:", NOT_USED)
print("Excluded as scrape-date snapshots:", SNAPSHOT_RISK)
print(f"\nSaved tables and figures to {RESULTS}/ (audit_variables.csv, figures/*.png)")