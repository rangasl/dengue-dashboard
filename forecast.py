"""
Dengue Forecasting — Sri Lanka 2025-2030
Methods: ARIMA · Prophet · Regional-feature Ridge regression · Biennial cycle
Outputs PNGs into analysis_output/
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet

OUT = "analysis_output"
HORIZON = 6   # forecast years: 2025-2030

plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 13,
                     "axes.labelsize": 11, "font.size": 10})

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv("data_sources/National_extract_V1_3.csv")
df.columns = df.columns.str.strip()
df["Year"]         = pd.to_numeric(df["Year"], errors="coerce")
df["dengue_total"] = pd.to_numeric(df["dengue_total"], errors="coerce")
nat = df[df["S_res"] == "Admin0"]

annual = (
    nat.groupby(["adm_0_name", "Year"])["dengue_total"]
    .sum().reset_index()
    .rename(columns={"adm_0_name": "Country", "dengue_total": "Cases"})
)

REGIONAL = ["SRI LANKA","INDIA","BANGLADESH","MYANMAR","THAILAND",
            "MALAYSIA","INDONESIA","PHILIPPINES","VIET NAM",
            "CAMBODIA","LAO PEOPLE'S DEMOCRATIC REPUBLIC","NEPAL"]

region = annual[(annual["Country"].isin(REGIONAL)) &
                (annual["Year"] >= 2000) & (annual["Year"] <= 2024)]

sl = (region[region["Country"] == "SRI LANKA"]
      .sort_values("Year").set_index("Year")["Cases"])

print(f"SL series: {len(sl)} annual observations ({sl.index.min()}–{sl.index.max()})")

future_years = np.arange(sl.index.max() + 1, sl.index.max() + 1 + HORIZON)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LAG ANALYSIS — does any country lead Sri Lanka?
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== LAG CORRELATIONS (lead/lag ±3 years vs Sri Lanka) ===")
pivot = region.pivot_table(index="Year", columns="Country", values="Cases")
pivot.index = pivot.index.astype(int)
sl_s = pivot["SRI LANKA"].dropna()

lag_results = {}
for country in [c for c in REGIONAL if c != "SRI LANKA"]:
    if country not in pivot.columns:
        continue
    other = pivot[country].dropna()
    row = {}
    for lag in range(-3, 4):            # negative = country leads SL
        shifted = other.shift(lag)
        common = pd.concat([sl_s, shifted], axis=1).dropna()
        if len(common) >= 6:
            r, p = pearsonr(common.iloc[:,0], common.iloc[:,1])
            row[lag] = (r, p)
    best_lag = max(row, key=lambda l: abs(row[l][0]))
    lag_results[country] = {"best_lag": best_lag,
                            "r": row[best_lag][0],
                            "p": row[best_lag][1],
                            "all": row}
    sig = "*" if row[best_lag][1] < 0.05 else ""
    direction = "LEADS SL" if best_lag < 0 else ("LAGS SL" if best_lag > 0 else "same year")
    print(f"  {country:<45} best lag={best_lag:+d}  r={row[best_lag][0]:.3f}{sig}  ({direction})")

# ══════════════════════════════════════════════════════════════════════════════
# 2. BIENNIAL CYCLE MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== BIENNIAL CYCLE ===")
# Use 2011-2024 (post-structural shift, excluding 2017 anomaly)
sl_cycle = sl[(sl.index >= 2011) & (sl.index != 2017)]
even_avg  = sl_cycle[sl_cycle.index % 2 == 0].mean()
odd_avg   = sl_cycle[sl_cycle.index % 2 == 1].mean()
overall   = sl_cycle.mean()
amplitude = (even_avg - odd_avg) / 2
print(f"Even-year avg: {even_avg:,.0f}  Odd-year avg: {odd_avg:,.0f}")
print(f"Cycle amplitude: ±{amplitude:,.0f} around {overall:,.0f}")

last_year = sl.index.max()
biennial_fc = {}
for yr in future_years:
    base = overall
    cycle_signal = amplitude if yr % 2 == 0 else -amplitude
    biennial_fc[yr] = base + cycle_signal

print("Biennial forecasts:", {k: f"{v:,.0f}" for k,v in biennial_fc.items()})

# ══════════════════════════════════════════════════════════════════════════════
# 3. ARIMA MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ARIMA ===")
# Log-transform to stabilise variance
sl_log = np.log(sl)

# ADF test
adf_stat, adf_p, *_ = adfuller(sl_log)
print(f"ADF p={adf_p:.3f} → {'stationary' if adf_p < 0.05 else 'non-stationary (will difference)'}")

# ARIMA(1,1,1) — robust choice for ~25 annual observations
try:
    model = SARIMAX(sl_log, order=(1, 1, 1), trend="c",
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    print(f"ARIMA AIC: {fit.aic:.1f}")

    fc = fit.get_forecast(steps=HORIZON)
    arima_mean = np.exp(fc.predicted_mean)
    ci         = fc.conf_int(alpha=0.30)   # 70% CI (not too wide for annual data)
    arima_lo   = np.exp(ci.iloc[:,0])
    arima_hi   = np.exp(ci.iloc[:,1])
    arima_mean.index = future_years
    arima_lo.index   = future_years
    arima_hi.index   = future_years
    print("ARIMA forecasts:", {k: f"{v:,.0f}" for k,v in arima_mean.items()})
    arima_ok = True
except Exception as e:
    print(f"ARIMA failed: {e}")
    arima_ok = False

# ══════════════════════════════════════════════════════════════════════════════
# 4. PROPHET MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== PROPHET ===")
prophet_df = pd.DataFrame({
    "ds": pd.to_datetime([f"{y}-01-01" for y in sl.index]),
    "y":  np.log(sl.values),          # log scale
})

m = Prophet(
    changepoint_prior_scale=0.3,   # allow trend shifts (2009 shift)
    seasonality_mode="additive",
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    interval_width=0.70,
)
m.fit(prophet_df)

future_df = m.make_future_dataframe(periods=HORIZON + 2, freq="YE")
forecast  = m.predict(future_df)

last_hist = prophet_df["ds"].dt.year.max()
fc_future = forecast[forecast["ds"].dt.year > last_hist].copy()
fc_future = fc_future.drop_duplicates(subset="ds").sort_values("ds").head(HORIZON)
fc_years  = fc_future["ds"].dt.year.values
prophet_fc = pd.Series(np.exp(fc_future["yhat"].values),        index=fc_years)
prophet_lo = pd.Series(np.exp(fc_future["yhat_lower"].values),  index=fc_years)
prophet_hi = pd.Series(np.exp(fc_future["yhat_upper"].values),  index=fc_years)
future_years = fc_years   # align all forecasts to same years
print("Prophet forecasts:", {k: f"{v:,.0f}" for k,v in prophet_fc.items()})

# ══════════════════════════════════════════════════════════════════════════════
# 5. REGIONAL FEATURE MODEL (Ridge regression with lagged India & Vietnam)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== REGIONAL FEATURE MODEL ===")

# Build feature matrix: use India lag=-1 (India same year) and Vietnam same year
# From lag analysis, India has strongest correlation
def build_features(pivot, sl_series, y0=2005, y1=2024):
    rows = []
    for yr in range(y0, y1+1):
        if yr not in sl_series.index:
            continue
        feat = {"Year": yr, "SL": sl_series[yr]}
        for country, lag in [("INDIA", 0), ("VIET NAM", 0),
                              ("MALAYSIA", 0), ("MYANMAR", 0)]:
            src_yr = yr + lag
            if country in pivot.columns and src_yr in pivot.index:
                feat[f"{country}_lag{lag}"] = pivot.loc[src_yr, country]
            else:
                feat[f"{country}_lag{lag}"] = np.nan
        # Biennial signal
        feat["biennial"] = 1 if yr % 2 == 0 else -1
        # Linear trend
        feat["trend"] = yr - 2009
        rows.append(feat)
    return pd.DataFrame(rows).dropna()

feat_df = build_features(pivot, sl_s)
X = feat_df.drop(columns=["Year", "SL"]).values
y = np.log(feat_df["SL"].values)

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

ridge = Ridge(alpha=10.0)

# LOO cross-validation MAPE
loo = LeaveOneOut()
preds, actuals = [], []
for train_idx, test_idx in loo.split(X_sc):
    ridge.fit(X_sc[train_idx], y[train_idx])
    preds.append(np.exp(ridge.predict(X_sc[test_idx])[0]))
    actuals.append(np.exp(y[test_idx[0]]))
mape = mean_absolute_percentage_error(actuals, preds) * 100
print(f"Ridge LOO MAPE: {mape:.1f}%")

# Refit on all data
ridge.fit(X_sc, y)

# Project: extend India/Vietnam using their own ARIMA-ish trend
def last_n_avg(series, n=3):
    return series.dropna().tail(n).mean()

india_recent = last_n_avg(pivot["INDIA"] if "INDIA" in pivot.columns else pd.Series())
vn_recent    = last_n_avg(pivot["VIET NAM"] if "VIET NAM" in pivot.columns else pd.Series())
my_recent    = last_n_avg(pivot["MALAYSIA"] if "MALAYSIA" in pivot.columns else pd.Series())
mm_recent    = last_n_avg(pivot["MYANMAR"] if "MYANMAR" in pivot.columns else pd.Series())

ridge_fc = {}
for i, yr in enumerate(future_years):
    feat_row = np.array([[
        india_recent * (1.02**i),
        vn_recent   * (1.01**i),
        my_recent   * (1.01**i),
        mm_recent   * (1.01**i),
        1 if yr % 2 == 0 else -1,
        yr - 2009,
    ]])
    feat_row_sc = scaler.transform(feat_row)
    ridge_fc[yr] = float(np.exp(ridge.predict(feat_row_sc)[0]))

print("Ridge forecasts:", {k: f"{v:,.0f}" for k,v in ridge_fc.items()})

# ══════════════════════════════════════════════════════════════════════════════
# 6. ENSEMBLE (simple average of 3 methods)
# ══════════════════════════════════════════════════════════════════════════════
ensemble = {}
for yr in future_years:
    vals = [arima_mean[yr] if arima_ok else None,
            prophet_fc[yr],
            ridge_fc[yr],
            biennial_fc[yr]]
    ensemble[yr] = np.mean([v for v in vals if v is not None])

print("\nEnsemble forecasts:", {k: f"{v:,.0f}" for k,v in ensemble.items()})

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — All forecasts on one chart
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))

# Historical
ax.plot(sl.index, sl.values / 1000, "o-", color="#c0392b", lw=2,
        ms=5, label="Historical (Sri Lanka)", zorder=5)

# ARIMA
if arima_ok:
    ax.plot(future_years, arima_mean.values / 1000, "s--", color="#2980b9",
            lw=1.8, ms=6, label="ARIMA (1,1,1)")
    ax.fill_between(future_years,
                    arima_lo.values / 1000,
                    arima_hi.values / 1000,
                    color="#2980b9", alpha=0.12)

# Prophet
ax.plot(future_years, prophet_fc.values / 1000, "^--", color="#8e44ad",
        lw=1.8, ms=6, label="Prophet")
ax.fill_between(future_years,
                prophet_lo.values / 1000,
                prophet_hi.values / 1000,
                color="#8e44ad", alpha=0.10)

# Ridge regional
ax.plot(future_years, [ridge_fc[y]/1000 for y in future_years], "D--",
        color="#27ae60", lw=1.8, ms=6, label="Regional features (Ridge)")

# Biennial
ax.plot(future_years, [biennial_fc[y]/1000 for y in future_years], "x--",
        color="#f39c12", lw=1.5, ms=8, label="Biennial cycle")

# Ensemble
ax.plot(future_years, [ensemble[y]/1000 for y in future_years], "o-",
        color="#2c3e50", lw=3, ms=8, label="Ensemble (mean)", zorder=6)

ax.axvline(x=sl.index.max() + 0.5, color="grey", ls=":", lw=1.2)
ax.text(sl.index.max() + 0.6, ax.get_ylim()[1] * 0.95,
        "← observed  forecast →", color="grey", fontsize=9)

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}k"))
ax.set(title="Sri Lanka Dengue Forecast 2025–2030 — Multi-Method Ensemble",
       xlabel="Year", ylabel="Cases (thousands)")
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.4)
plt.tight_layout()
fig.savefig(f"{OUT}/08_forecast_all_methods.png")
plt.close()
print("\nSaved 08_forecast_all_methods.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Lag correlation heatmap (which country leads/lags SL by how much)
# ══════════════════════════════════════════════════════════════════════════════
lag_matrix = {}
for country, info in lag_results.items():
    lag_matrix[country.title().replace("'S","'s")] = {
        l: info["all"][l][0] for l in info["all"]
    }

lag_df = pd.DataFrame(lag_matrix).T.sort_index()
lag_df.columns = [f"lag {l:+d}" for l in lag_df.columns]

fig, ax = plt.subplots(figsize=(12, 7))
import matplotlib.colors as mcolors
cmap = plt.cm.RdBu_r
norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = ax.imshow(lag_df.values, cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks(range(len(lag_df.columns)))
ax.set_xticklabels(lag_df.columns, fontsize=9)
ax.set_yticks(range(len(lag_df.index)))
ax.set_yticklabels(lag_df.index, fontsize=9)
for i in range(len(lag_df.index)):
    for j in range(len(lag_df.columns)):
        val = lag_df.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(val) > 0.5 else "black")
plt.colorbar(im, ax=ax, label="Pearson r with Sri Lanka")
ax.set_title(
    "Lead/Lag Correlation vs Sri Lanka\n"
    "(negative lag = country's data leads Sri Lanka by that many years)"
)
plt.tight_layout()
fig.savefig(f"{OUT}/09_lag_heatmap.png")
plt.close()
print("Saved 09_lag_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Forecast summary table
# ══════════════════════════════════════════════════════════════════════════════
print("\n══════════ FORECAST SUMMARY (cases) ══════════")
print(f"{'Year':<6} {'ARIMA':>10} {'Prophet':>10} {'Ridge':>10} {'Biennial':>10} {'ENSEMBLE':>10}")
print("-"*60)
for yr in future_years:
    a = f"{arima_mean[yr]:>10,.0f}" if arima_ok else "       N/A"
    p = f"{prophet_fc[yr]:>10,.0f}"
    r = f"{ridge_fc[yr]:>10,.0f}"
    b = f"{biennial_fc[yr]:>10,.0f}"
    e = f"{ensemble[yr]:>10,.0f}"
    print(f"{yr:<6} {a} {p} {r} {b} {e}")

print("\n══════════ KEY SIGNALS ══════════")
for country, info in sorted(lag_results.items(), key=lambda x: -abs(x[1]["r"])):
    if info["p"] < 0.05:
        lag = info["best_lag"]
        desc = f"India's cases in same year predict SL by r={info['r']:.2f}" if lag == 0 else \
               f"{'Leads' if lag < 0 else 'Lags'} SL by {abs(lag)} yr, r={info['r']:.2f}"
        print(f"  {country:<45} {desc}")

print("\nAll forecasts saved to analysis_output/")
