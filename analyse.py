"""
Dengue trend analysis — Sri Lanka vs South/Southeast Asian neighbours.
Outputs PNGs into analysis_output/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import pearsonr

# ── Setup ──────────────────────────────────────────────────────────────────
OUT = "analysis_output"
os.makedirs(OUT, exist_ok=True)

REGIONAL = [
    "SRI LANKA", "INDIA", "BANGLADESH", "MYANMAR", "THAILAND",
    "MALAYSIA", "INDONESIA", "PHILIPPINES", "VIET NAM",
    "CAMBODIA", "LAO PEOPLE'S DEMOCRATIC REPUBLIC", "NEPAL",
]

sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 13, "axes.labelsize": 11})

# ── Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv("data_sources/National_extract_V1_3.csv")
df.columns = df.columns.str.strip()
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["dengue_total"] = pd.to_numeric(df["dengue_total"], errors="coerce")

# Keep national-level rows only (no sub-national breakdowns)
nat = df[df["S_res"] == "Admin0"].copy()

# ── Annual aggregates ──────────────────────────────────────────────────────
annual = (
    nat.groupby(["adm_0_name", "Year"])["dengue_total"]
    .sum()
    .reset_index()
    .rename(columns={"adm_0_name": "Country", "dengue_total": "Cases"})
)

region = annual[annual["Country"].isin(REGIONAL)].copy()
sl = region[region["Country"] == "SRI LANKA"].copy()

# Limit to years with reasonable data coverage
YEAR_START, YEAR_END = 2000, 2025
region = region[(region["Year"] >= YEAR_START) & (region["Year"] <= YEAR_END)]
sl = sl[(sl["Year"] >= YEAR_START) & (sl["Year"] <= YEAR_END)]

print(f"Sri Lanka rows: {len(sl)}, years: {sl['Year'].min()}–{sl['Year'].max()}")
print(sl[["Year","Cases"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Sri Lanka annual trend with 5-yr rolling mean
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(sl["Year"], sl["Cases"] / 1000, color="#e05c5c", alpha=0.75, label="Annual cases")
roll = sl.set_index("Year")["Cases"].rolling(5, min_periods=3).mean()
ax.plot(roll.index, roll.values / 1000, color="#8b0000", lw=2.5, label="5-yr rolling mean")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
ax.set(title="Sri Lanka — Annual Dengue Cases (2000–2025)",
       xlabel="Year", ylabel="Cases (thousands)")
ax.legend()
plt.tight_layout()
fig.savefig(f"{OUT}/01_sl_annual_trend.png")
plt.close()
print("Saved 01_sl_annual_trend.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Year-on-year growth rate for Sri Lanka
# ══════════════════════════════════════════════════════════════════════════════
sl_sorted = sl.sort_values("Year").set_index("Year")
sl_sorted["Growth%"] = sl_sorted["Cases"].pct_change() * 100

fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#e05c5c" if g >= 0 else "#4c8cdb" for g in sl_sorted["Growth%"].fillna(0)]
ax.bar(sl_sorted.index, sl_sorted["Growth%"].fillna(0), color=colors)
ax.axhline(0, color="black", lw=0.8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
ax.set(title="Sri Lanka — Year-on-Year Dengue Growth Rate",
       xlabel="Year", ylabel="Growth vs prior year")
plt.tight_layout()
fig.savefig(f"{OUT}/02_sl_yoy_growth.png")
plt.close()
print("Saved 02_sl_yoy_growth.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Regional comparison: line chart (indexed to 2010=100)
# ══════════════════════════════════════════════════════════════════════════════
pivot = region.pivot_table(index="Year", columns="Country", values="Cases", aggfunc="sum")
# Only keep countries with data in at least 10 years
pivot = pivot.loc[:, pivot.notna().sum() >= 10]

# Index to 2010 baseline
base_year = 2010
base = pivot.loc[base_year] if base_year in pivot.index else pivot.iloc[0]
indexed = pivot.div(base).mul(100)

fig, ax = plt.subplots(figsize=(14, 7))
for country in indexed.columns:
    lw = 3.0 if country == "SRI LANKA" else 1.2
    alpha = 1.0 if country == "SRI LANKA" else 0.65
    ax.plot(indexed.index, indexed[country], lw=lw, alpha=alpha, label=country.title())

ax.axhline(100, color="grey", lw=0.8, ls="--", label=f"{base_year}=100 baseline")
ax.set(title=f"Regional Dengue Trend — Indexed to {base_year}=100",
       xlabel="Year", ylabel=f"Cases relative to {base_year}")
ax.legend(fontsize=8, ncol=2, loc="upper left")
plt.tight_layout()
fig.savefig(f"{OUT}/03_regional_indexed.png")
plt.close()
print("Saved 03_regional_indexed.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Heatmap: each country × year (log-normalised)
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np

heat = pivot.copy().T  # countries as rows, years as cols
heat_log = np.log1p(heat)

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(
    heat_log, ax=ax, cmap="YlOrRd",
    linewidths=0.3, linecolor="white",
    cbar_kws={"label": "log(cases+1)"},
    yticklabels=[c.title() for c in heat_log.index],
)
ax.set(title="Dengue Burden by Country & Year (log scale)", xlabel="Year")
plt.tight_layout()
fig.savefig(f"{OUT}/04_heatmap.png")
plt.close()
print("Saved 04_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Pearson correlation of each country's trend vs Sri Lanka
# ══════════════════════════════════════════════════════════════════════════════
sl_series = pivot["SRI LANKA"].dropna() if "SRI LANKA" in pivot.columns else None
correlations = {}
if sl_series is not None:
    for country in pivot.columns:
        if country == "SRI LANKA":
            continue
        shared = pivot[[country, "SRI LANKA"]].dropna()
        if len(shared) >= 8:
            r, p = pearsonr(shared[country], shared["SRI LANKA"])
            correlations[country.title()] = {"r": r, "p": p}

corr_df = pd.DataFrame(correlations).T.sort_values("r", ascending=True)
print("\nCorrelations with Sri Lanka:")
print(corr_df.to_string())

colors = ["#e05c5c" if r >= 0 else "#4c8cdb" for r in corr_df["r"]]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(corr_df.index, corr_df["r"], color=colors, edgecolor="white")
# Mark significance
for i, (_, row) in enumerate(corr_df.iterrows()):
    if row["p"] < 0.05:
        ax.text(row["r"] + (0.01 if row["r"] >= 0 else -0.01), i, "*",
                va="center", ha="left" if row["r"] >= 0 else "right", fontsize=12)
ax.axvline(0, color="black", lw=0.8)
ax.set(title="Pearson Correlation with Sri Lanka Dengue Trend\n(* = p<0.05)",
       xlabel="Pearson r", xlim=(-1.1, 1.1))
plt.tight_layout()
fig.savefig(f"{OUT}/05_correlation_vs_sl.png")
plt.close()
print("Saved 05_correlation_vs_sl.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Sri Lanka vs top 3 correlated neighbours — raw cases
# ══════════════════════════════════════════════════════════════════════════════
top3 = corr_df.tail(3).index.tolist()  # highest r values
fig, axes = plt.subplots(len(top3), 1, figsize=(12, 4 * len(top3)), sharex=True)
for ax, country in zip(axes, top3):
    c_data = region[region["Country"] == country.upper()].set_index("Year")["Cases"]
    ax2 = ax.twinx()
    ax.bar(sl["Year"], sl["Cases"] / 1000, color="#e05c5c", alpha=0.5, label="Sri Lanka (L)")
    ax2.plot(c_data.index, c_data / 1000, color="#2a5caa", lw=2, label=f"{country} (R)")
    ax.set_ylabel("Sri Lanka cases (k)", color="#e05c5c")
    ax2.set_ylabel(f"{country} cases (k)", color="#2a5caa")
    r_val = corr_df.loc[country, "r"]
    ax.set_title(f"Sri Lanka vs {country}  (r = {r_val:.2f})")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=9, loc="upper left")

axes[-1].set_xlabel("Year")
plt.suptitle("Sri Lanka vs Most Correlated Neighbours", y=1.01, fontsize=13)
plt.tight_layout()
fig.savefig(f"{OUT}/06_sl_vs_neighbours.png", bbox_inches="tight")
plt.close()
print("Saved 06_sl_vs_neighbours.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Epidemic years: bar chart of Sri Lanka's top 10 worst years
# ══════════════════════════════════════════════════════════════════════════════
top10 = sl.nlargest(10, "Cases").sort_values("Year")
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(top10["Year"].astype(str), top10["Cases"] / 1000,
              color="#c0392b", edgecolor="white")
for bar, val in zip(bars, top10["Cases"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val/1000:.1f}k", ha="center", va="bottom", fontsize=9)
ax.set(title="Sri Lanka — Top 10 Worst Dengue Years",
       xlabel="Year", ylabel="Cases (thousands)")
plt.tight_layout()
fig.savefig(f"{OUT}/07_sl_worst_years.png")
plt.close()
print("Saved 07_sl_worst_years.png")

# ══════════════════════════════════════════════════════════════════════════════
# Print summary stats
# ══════════════════════════════════════════════════════════════════════════════
print("\n══════════ Sri Lanka Summary ══════════")
print(f"Total cases (2000–2025):  {sl['Cases'].sum():,.0f}")
print(f"Peak year:                {sl.loc[sl['Cases'].idxmax(), 'Year']} ({sl['Cases'].max():,.0f} cases)")
print(f"Average per year:         {sl['Cases'].mean():,.0f}")
print(f"Worst 5 years: {sl.nlargest(5,'Cases')['Year'].tolist()}")

print("\nAll output saved to:", os.path.abspath(OUT))
