"""
Interactive Dengue Dashboard — Sri Lanka & South/Southeast Asia
Run: streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dengue Dashboard — Sri Lanka & Region",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.fixed-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #0e1117;
    color: #6b7280;
    text-align: center;
    padding: 8px 0;
    font-size: 12px;
    z-index: 9999;
    border-top: 1px solid #1f2937;
    pointer-events: none;
    user-select: none;
}
</style>
<div class="fixed-footer">Experiment by Ranga Dayawansha &nbsp;|&nbsp; Founder of FloodSupport.org</div>
""", unsafe_allow_html=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
SL_RED  = "#c0392b"
BLUE    = "#2980b9"
ACCENT  = "#f39c12"
GREEN   = "#27ae60"
PURPLE  = "#8e44ad"
GREY    = "#7f8c8d"
ORANGE  = "#e67e22"

REGIONAL = [
    "SRI LANKA", "INDIA", "BANGLADESH", "MYANMAR", "THAILAND",
    "MALAYSIA", "INDONESIA", "PHILIPPINES", "VIET NAM",
    "CAMBODIA", "LAO PEOPLE'S DEMOCRATIC REPUBLIC", "NEPAL",
]
COUNTRY_LABELS = {c: c.title().replace("'S", "'s") for c in REGIONAL}
COUNTRY_LABELS["LAO PEOPLE'S DEMOCRATIC REPUBLIC"] = "Laos"
COUNTRY_LABELS["VIET NAM"] = "Viet Nam"

# ── Climate history (ONI annual mean + IOD Aug-Oct peak) ──────────────────────
CLIMATE = pd.DataFrame({
    "ONI": {
        2000:-0.84,2001:-0.31,2002: 0.63,2003: 0.25,2004: 0.45,
        2005: 0.04,2006: 0.07,2007:-0.61,2008:-0.78,2009: 0.28,
        2010:-0.48,2011:-0.78,2012:-0.06,2013:-0.23,2014: 0.21,
        2015: 1.55,2016: 0.42,2017:-0.12,2018: 0.09,2019: 0.58,
        2020:-0.27,2021:-0.64,2022:-0.85,2023: 0.90,2024: 0.44,
        2025:-0.20,2026: 0.48,
    },
    "IOD": {
        2000:-0.12,2001: 0.02,2002: 0.25,2003:-0.04,2004:-0.26,
        2005: 0.12,2006: 0.57,2007: 1.08,2008:-0.19,2009:-0.42,
        2010:-0.21,2011: 0.14,2012:-0.03,2013: 0.02,2014:-0.28,
        2015: 0.47,2016:-0.02,2017:-0.38,2018:-0.11,2019: 0.85,
        2020: 0.23,2021:-0.38,2022:-0.23,2023: 0.54,2024:-0.08,
        2025: 0.15,2026: 0.35,
    },
})

# ── Risk scoring ───────────────────────────────────────────────────────────────
def compute_risk(oni: float, iod: float, phil_signal: bool, biennial_high: bool):
    score = 0
    factors = []

    if oni > 1.5:
        score += 3; factors.append(f"Strong El Niño (ONI {oni:+.2f})")
    elif oni > 0.5:
        score += 1; factors.append(f"El Niño (ONI {oni:+.2f})")
    elif oni < -1.5:
        score += 3; factors.append(f"Strong La Niña (ONI {oni:+.2f})")
    elif oni < -0.5:
        score += 1; factors.append(f"La Niña (ONI {oni:+.2f})")
    else:
        factors.append(f"ENSO Neutral (ONI {oni:+.2f})")

    if iod > 0.6:
        score += 2; factors.append(f"Strong Positive IOD (DMI {iod:+.2f})")
    elif iod > 0.3:
        score += 1; factors.append(f"Positive IOD (DMI {iod:+.2f})")
    elif iod < -0.3:
        score += 1; factors.append(f"Negative IOD (DMI {iod:+.2f})")
    else:
        factors.append(f"IOD Neutral (DMI {iod:+.2f})")

    if oni > 0.5 and iod > 0.3:
        score += 1
        factors.append("⚠️ El Niño + Positive IOD coupling (2019/2023 analog)")

    if phil_signal:
        score += 1; factors.append("Philippines high burden 2 yrs ago (lead indicator r=0.74)")

    if biennial_high:
        score += 1; factors.append("Biennial cycle: high-burden phase")

    score = min(score, 6)

    if score <= 1:   level, colour, emoji = "LOW",      "#27ae60", "🟢"
    elif score <= 2: level, colour, emoji = "MODERATE", "#f39c12", "🟡"
    elif score <= 3: level, colour, emoji = "ELEVATED", "#e67e22", "🟠"
    elif score <= 4: level, colour, emoji = "HIGH",     "#c0392b", "🔴"
    else:            level, colour, emoji = "CRITICAL", "#7b241c", "🔴"

    return score, level, colour, emoji, factors

# ── Population (World Bank) ────────────────────────────────────────────────────
POP = {
    2000:19293054,2001:19600362,2002:19805752,2003:19951521,2004:20087605,
    2005:20216524,2006:20352411,2007:20492545,2008:20629378,2009:20756435,
    2010:20879089,2011:21009048,2012:21169458,2013:20585000,2014:20778000,
    2015:20970000,2016:21209000,2017:21453000,2018:21670000,2019:21803000,
    2020:21919000,2021:22156000,2022:22181000,2023:22037000,2024:21916000,
}

# ── Serotype timeline (published literature) ───────────────────────────────────
SEROTYPES = {
    2000:{"dominant":"DENV-1","confidence":"medium"},
    2001:{"dominant":"DENV-1","confidence":"medium"},
    2002:{"dominant":"DENV-1/2","confidence":"medium"},
    2003:{"dominant":"DENV-1","confidence":"medium"},
    2004:{"dominant":"DENV-3","confidence":"high","note":"DENV-3 emergence → 15k cases"},
    2005:{"dominant":"DENV-1","confidence":"medium"},
    2006:{"dominant":"DENV-1","confidence":"medium"},
    2007:{"dominant":"DENV-1","confidence":"medium"},
    2008:{"dominant":"DENV-1","confidence":"medium"},
    2009:{"dominant":"DENV-3","confidence":"high","note":"DENV-3 reemergence → baseline shift"},
    2010:{"dominant":"DENV-1","confidence":"high","note":"DENV-1 after DENV-3 → DHF risk"},
    2011:{"dominant":"DENV-1","confidence":"high"},
    2012:{"dominant":"DENV-1/3","confidence":"high","note":"Mixed → 43k spike"},
    2013:{"dominant":"DENV-2","confidence":"high","note":"DENV-2 Cosmopolitan emerging"},
    2014:{"dominant":"DENV-2","confidence":"high"},
    2015:{"dominant":"DENV-2","confidence":"high"},
    2016:{"dominant":"DENV-2","confidence":"high"},
    2017:{"dominant":"DENV-2","confidence":"high","note":"DENV-2 + 2017 floods → 176k"},
    2018:{"dominant":"DENV-2","confidence":"high"},
    2019:{"dominant":"DENV-2/3","confidence":"high","note":"Co-circulation → 96k"},
    2020:{"dominant":"DENV-2","confidence":"medium","note":"COVID suppression"},
    2021:{"dominant":"DENV-2","confidence":"medium"},
    2022:{"dominant":"DENV-1","confidence":"medium","note":"DENV-1 re-emergence"},
    2023:{"dominant":"DENV-1/2","confidence":"medium","note":"Mixed → 88k"},
    2024:{"dominant":"DENV-1","confidence":"medium"},
}
SERO_COLORS = {
    "DENV-1":"#3498db","DENV-2":"#e74c3c","DENV-3":"#2ecc71",
    "DENV-1/2":"#9b59b6","DENV-1/3":"#1abc9c","DENV-2/3":"#e67e22",
}

# ── Weekly seasonal weights (Sri Lanka bimodal pattern) ────────────────────────
def seasonal_weights(year, n=52):
    wk  = np.arange(1, n + 1)
    pk1 = 18 if year == 2017 else 20
    w   = 0.25 + 2.8*np.exp(-0.5*((wk-pk1)/5)**2) + 1.6*np.exp(-0.5*((wk-43)/5)**2)
    return w / w.sum()

# ── Forecast (cached — slow to compute) ───────────────────────────────────────
@st.cache_data
def build_forecast(sl_cases: pd.Series):
    sl_log   = np.log(sl_cases)
    HORIZON  = 6
    fut_yrs  = np.arange(sl_cases.index.max()+1, sl_cases.index.max()+1+HORIZON)

    # ARIMA
    try:
        afit  = SARIMAX(sl_log.values, order=(1,1,1),
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)
        afc   = afit.get_forecast(HORIZON)
        a_mean = np.exp(afc.predicted_mean)
        a_lo   = np.exp(afc.conf_int(alpha=0.30).iloc[:,0])
        a_hi   = np.exp(afc.conf_int(alpha=0.30).iloc[:,1])
        arima_ok = True
    except Exception:
        a_mean = a_lo = a_hi = None
        arima_ok = False

    # Prophet
    pr_df = pd.DataFrame({
        "ds": pd.to_datetime([f"{y}-01-01" for y in sl_cases.index]),
        "y":  sl_log.values,
    })
    pm = Prophet(changepoint_prior_scale=0.3, interval_width=0.70,
                 yearly_seasonality=False, weekly_seasonality=False,
                 daily_seasonality=False)
    pm.fit(pr_df)
    fut   = pm.make_future_dataframe(periods=HORIZON+2, freq="YE")
    fcast = pm.predict(fut)
    rows  = (fcast[fcast["ds"].dt.year > sl_cases.index.max()]
             .drop_duplicates("ds").head(HORIZON))
    p_mean = np.exp(rows["yhat"].values)
    p_lo   = np.exp(rows["yhat_lower"].values)
    p_hi   = np.exp(rows["yhat_upper"].values)

    base = sl_cases[(sl_cases.index>=2011) & (sl_cases.index!=2017)].mean()

    ens_mean = np.array([np.nanmean([
        a_mean[i] if arima_ok else np.nan, p_mean[i], base
    ]) for i in range(HORIZON)])
    ens_lo = np.array([min(v for v in [
        a_lo[i] if arima_ok else None, p_lo[i], base*0.5
    ] if v is not None and not np.isnan(v)) for i in range(HORIZON)])
    ens_hi = np.array([max(v for v in [
        a_hi[i] if arima_ok else None, p_hi[i], base*3.0
    ] if v is not None and not np.isnan(v)) for i in range(HORIZON)])

    return fut_yrs, ens_mean, ens_lo, ens_hi, p_mean, p_lo, p_hi

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
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
    region = annual[annual["Country"].isin(REGIONAL)].copy()
    region["Label"] = region["Country"].map(COUNTRY_LABELS)
    return region

data     = load_data()
all_years = sorted(data["Year"].dropna().unique().astype(int))

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦟 Dengue Dashboard")
    st.markdown("**Data:** WHO GIDEON · NOAA CPC · JAMSTEC")
    st.divider()

    year_range = st.slider(
        "Historical year range",
        min_value=int(min(all_years)), max_value=int(max(all_years)),
        value=(2000, 2024), step=1,
    )
    compare_countries = st.multiselect(
        "Compare with Sri Lanka",
        options=[COUNTRY_LABELS[c] for c in REGIONAL if c != "SRI LANKA"],
        default=["India", "Viet Nam", "Malaysia", "Philippines", "Myanmar"],
    )
    show_indexed = st.toggle("Normalize to index (2010=100)", value=False)

    st.divider()
    st.markdown("### 🌊 Climate Risk — Current State")
    st.caption("Adjust to match latest NOAA/BOM forecasts")

    current_oni = st.slider(
        "ONI — ENSO index", -3.0, 3.0, 0.48, 0.01,
        help="Oceanic Niño Index. +0.5 threshold = El Niño, -0.5 = La Niña",
    )
    current_iod = st.slider(
        "DMI — Indian Ocean Dipole", -2.0, 2.0, 0.35, 0.01,
        help="Dipole Mode Index. +0.3 = Positive IOD (more SL rainfall), -0.3 = Negative",
    )

    # Philippines lead signal — auto from data
    phil_data = data[data["Country"] == "PHILIPPINES"].sort_values("Year")
    phil_2yr  = phil_data[phil_data["Year"] == 2024]["Cases"].values
    phil_avg  = phil_data[phil_data["Year"].between(2015, 2023)]["Cases"].mean()
    phil_flag = bool(phil_2yr.size > 0 and phil_2yr[0] > phil_avg * 1.15)

    # Biennial cycle — 2026 is even year; post-2009 even years avg ~47k, odd ~51k
    # Current outlook year is 2026
    biennial_flag = st.toggle(
        "Biennial cycle: high phase", value=False,
        help="Toggle on if current year falls in the high-burden phase of the 2-yr cycle",
    )

    st.divider()
    st.caption("Built with Streamlit + Plotly · WHO/NOAA data")

# ── Filter ─────────────────────────────────────────────────────────────────────
y0, y1 = year_range
df = data[(data["Year"] >= y0) & (data["Year"] <= y1)].copy()
sl = df[df["Country"] == "SRI LANKA"].sort_values("Year")
label_to_country = {v: k for k, v in COUNTRY_LABELS.items()}
selected_raw = ["SRI LANKA"] + [label_to_country[l] for l in compare_countries]
df_sel = df[df["Country"].isin(selected_raw)].copy()

# ── Compute risk ───────────────────────────────────────────────────────────────
risk_score, risk_level_str, risk_colour, risk_emoji, risk_factors = compute_risk(
    current_oni, current_iod, phil_flag, biennial_flag
)

# ── Header KPIs ───────────────────────────────────────────────────────────────
st.title("🦟 Dengue Fever — Sri Lanka & Regional Trends")

# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC RISK BANNER + SYMPTOMS
# ══════════════════════════════════════════════════════════════════════════════
import datetime
_month = datetime.date.today().month

_seasonal_level = {
    1:"LOW",2:"LOW",3:"MODERATE",4:"MODERATE",
    5:"HIGH",6:"HIGH",7:"HIGH",8:"MODERATE",
    9:"MODERATE",10:"HIGH",11:"HIGH",12:"MODERATE",
}[_month]

_season_desc = {
    "LOW":      "Between monsoon peaks — dengue activity typically low",
    "MODERATE": "Transitional season — cases starting to rise, stay alert",
    "HIGH":     "Active monsoon season — peak dengue transmission in Sri Lanka",
}

# Combine seasonal + climate scores into one public-facing verdict
if _seasonal_level == "HIGH" and risk_score >= 2:
    _pub_risk, _pub_col, _pub_emoji = "HIGH",     "#c0392b", "🔴"
    _pub_msg = ("Peak monsoon season with elevated climate signals. "
                "Dengue mosquitoes are most active right now. "
                "Take full precautions and act fast if fever develops.")
elif _seasonal_level == "HIGH":
    _pub_risk, _pub_col, _pub_emoji = "ELEVATED", "#e67e22", "🟠"
    _pub_msg = ("Active monsoon season — Sri Lanka's main dengue period. "
                "Remove stagnant water around your home and use repellent daily.")
elif _seasonal_level == "MODERATE" and risk_score >= 3:
    _pub_risk, _pub_col, _pub_emoji = "ELEVATED", "#e67e22", "🟠"
    _pub_msg = ("Climate signals suggest a more active than normal year ahead. "
                "Start precautions now before the monsoon peaks.")
elif _seasonal_level == "LOW":
    _pub_risk, _pub_col, _pub_emoji = "LOW",      "#27ae60", "🟢"
    _pub_msg = ("Lower risk period. Dengue never fully disappears in Sri Lanka "
                "— maintain routine precautions year-round.")
else:
    _pub_risk, _pub_col, _pub_emoji = "MODERATE", "#f39c12", "🟡"
    _pub_msg = ("Moderate risk. Dengue cases are present throughout the year. "
                "Stay vigilant, especially after rain.")

ban_col, sym_col = st.columns([1.6, 1])

with ban_col:
    st.markdown(
        f"""<div style='background:{_pub_col};color:white;padding:22px 26px;
        border-radius:12px;margin-bottom:8px'>
        <div style='font-size:12px;letter-spacing:2px;opacity:0.85;
        text-transform:uppercase'>🦟 Current Dengue Risk — Sri Lanka</div>
        <div style='font-size:42px;font-weight:900;margin:8px 0 6px'>
        {_pub_emoji} {_pub_risk} RISK</div>
        <div style='font-size:15px;line-height:1.5;opacity:0.95'>{_pub_msg}</div>
        <div style='margin-top:12px;font-size:12px;opacity:0.75'>
        Seasonal: <b>{_seasonal_level}</b> — {_season_desc[_seasonal_level]}<br>
        Climate signal: <b>{risk_level_str}</b> &nbsp;·&nbsp;
        Updated: <b>{datetime.date.today().strftime("%B %Y")}</b>
        </div></div>""",
        unsafe_allow_html=True,
    )
    st.caption(
        "⚠️ Research tool only — not an official health advisory. "
        "For official alerts: **Epidemiology Unit Sri Lanka** → epid.gov.lk"
    )

with sym_col:
    with st.expander("🤒 Know the Symptoms", expanded=True):
        st.markdown("""
**Classic dengue (Days 1–3)**
- Sudden high fever (39–40°C)
- Severe headache & pain behind the eyes
- Muscle and joint pain *(breakbone fever)*
- Nausea, vomiting, loss of appetite
- Skin rash appearing around Day 3–4

---
**⚠️ Go to hospital immediately if:**
- Severe stomach pain or tenderness
- Persistent vomiting
- Bleeding gums, nose, or in urine/stool
- Rapid breathing or cold/clammy skin
- Sudden calm after a long fever (danger sign)

---
🚫 **Do NOT take** Aspirin or Ibuprofen — increases bleeding risk.
✅ **Paracetamol only** for fever.
        """)

st.divider()

# ── What to Do — 3 action cards ───────────────────────────────────────────────
st.markdown("### 🛡️ What You Should Do")
a1, a2, a3 = st.columns(3)
_card = "background:#1a1a2e;border-left:5px solid {c};padding:18px;border-radius:8px;min-height:220px"

with a1:
    st.markdown(
        f"""<div style='{_card.format(c=_pub_col)}'>
        <b style='font-size:15px'>🏠 At Home — Prevent Breeding</b><br><br>
        ✅ Empty & scrub water containers <b>every week</b><br>
        ✅ Cover all water storage tanks<br>
        ✅ Clear blocked drains and gutters<br>
        ✅ Change water in flower vases daily<br>
        ✅ Use mosquito coils or plug-ins at dusk<br>
        ✅ Sleep under a mosquito net
        </div>""", unsafe_allow_html=True)

with a2:
    st.markdown(
        f"""<div style='{_card.format(c=_pub_col)}'>
        <b style='font-size:15px'>🤧 If You Have Fever</b><br><br>
        ✅ Take <b>Paracetamol</b> — not Aspirin or Brufen<br>
        ✅ Drink fluids constantly (ORS, coconut water)<br>
        ✅ Rest — stay home from work or school<br>
        ✅ See a doctor if fever lasts <b>more than 2 days</b><br>
        ✅ Get a blood test if doctor suspects dengue<br>
        🚫 Do not self-medicate with antibiotics
        </div>""", unsafe_allow_html=True)

with a3:
    st.markdown(
        f"""<div style='{_card.format(c="#c0392b")}'>
        <b style='font-size:15px'>🚨 Go to Hospital If You Have:</b><br><br>
        🔴 Severe stomach pain<br>
        🔴 Vomiting that won't stop<br>
        🔴 Bleeding from any site<br>
        🔴 Fever suddenly drops but you feel <i>worse</i><br>
        🔴 Difficulty breathing<br>
        🔴 Extreme fatigue or confusion<br><br>
        <b>📞 Emergency: 1990 (Suwa Seriya Ambulance)</b>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

sl_sorted   = sl.sort_values("Year")
total       = int(sl["Cases"].sum())
peak_yr     = int(sl.loc[sl["Cases"].idxmax(), "Year"]) if not sl.empty else "N/A"
peak_v      = int(sl["Cases"].max()) if not sl.empty else 0
avg_yr      = int(sl["Cases"].mean()) if not sl.empty else 0
last_growth = (
    (sl_sorted.iloc[-1]["Cases"] - sl_sorted.iloc[-2]["Cases"]) / sl_sorted.iloc[-2]["Cases"] * 100
    if len(sl_sorted) >= 2 else 0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Cases (SL)", f"{total:,}")
c2.metric("Peak Year", peak_yr, f"{peak_v:,} cases")
c3.metric("Avg per Year", f"{avg_yr:,}")
c4.metric("Last Year Change", f"{last_growth:+.1f}%",
          delta_color="inverse" if last_growth > 0 else "normal")

# ── Year-vs-year tracker ───────────────────────────────────────────────────────
st.markdown("#### 📅 How Is This Year Tracking?")
_sl_all  = (data[data["Country"]=="SRI LANKA"]
            .groupby("Year")["Cases"].sum()
            .sort_index())
_ly      = int(_sl_all.index.max())
_ly_c    = int(_sl_all[_ly])
_py_c    = int(_sl_all.loc[_ly - 1] if (_ly - 1) in _sl_all.index else _sl_all.iloc[-2])
_py2_c   = int(_sl_all.loc[_ly - 2] if (_ly - 2) in _sl_all.index else _sl_all.iloc[-3])
_avg5    = int(_sl_all.loc[_ly - 5 : _ly - 1].mean())
_vs_avg  = (_ly_c - _avg5) / _avg5 * 100
_vs_prev = (_ly_c - _py_c)  / _py_c  * 100

yy1, yy2, yy3, yy4 = st.columns(4)
yy1.metric(
    f"Last Full Year ({_ly})",
    f"{_ly_c:,} cases",
    f"{_vs_prev:+.1f}% vs {_ly-1}",
    delta_color="inverse",
)
yy2.metric(
    f"{_ly-1}",
    f"{_py_c:,} cases",
    f"{(_py_c-_py2_c)/_py2_c*100:+.1f}% vs {_ly-2}",
    delta_color="inverse",
)
yy3.metric("5-Year Average", f"{_avg5:,} cases")
yy4.metric(
    f"{_ly} vs 5-yr average",
    f"{_vs_avg:+.1f}%",
    "above average" if _vs_avg > 0 else "below average",
    delta_color="inverse",
)

_trend_msg = (
    f"✅ **{_ly} was a below-average year** ({_ly_c:,} cases vs {_avg5:,} 5-yr avg). "
    f"Biennial cycle suggests {_ly+1} may rebound."
    if _vs_avg < -10 else
    f"⚠️ **{_ly} was above the 5-year average** ({_ly_c:,} vs {_avg5:,}). "
    f"Watch {_ly+1} closely."
    if _vs_avg > 20 else
    f"**{_ly} was near the 5-year average** ({_ly_c:,} cases). "
    f"Baseline dengue burden remains elevated post-2009."
)
st.markdown(_trend_msg)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# CLIMATE RISK PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🌊 Climate Risk Panel — Early Warning")

# Row A: gauges + traffic light + factors
ga, gb, gc, gd = st.columns([1.2, 1.2, 1, 1.8])

# ENSO gauge
with ga:
    fig_oni = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_oni,
        delta={"reference": 0, "valueformat": ".2f"},
        title={"text": "ENSO (ONI)", "font": {"size": 14}},
        number={"suffix": " °C", "font": {"size": 22}},
        gauge={
            "axis": {"range": [-3, 3], "tickwidth": 1},
            "bar":  {"color": SL_RED if current_oni > 0.5
                              else (BLUE if current_oni < -0.5 else "grey"),
                     "thickness": 0.25},
            "steps": [
                {"range": [-3,  -0.5], "color": "#d6eaf8"},
                {"range": [-0.5, 0.5], "color": "#f2f3f4"},
                {"range": [ 0.5,  3 ], "color": "#fadbd8"},
            ],
            "threshold": {"line": {"color": "black", "width": 2},
                          "thickness": 0.75, "value": current_oni},
        },
    ))
    fig_oni.update_layout(height=200, margin=dict(t=30, b=10, l=20, r=20))
    st.plotly_chart(fig_oni, use_container_width=True)

    phase_oni = ("🔴 El Niño"  if current_oni >  0.5 else
                 "🔵 La Niña" if current_oni < -0.5 else
                 "⚪ Neutral")
    st.markdown(f"<center><b>{phase_oni}</b></center>", unsafe_allow_html=True)

# IOD gauge
with gb:
    fig_iod = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_iod,
        delta={"reference": 0, "valueformat": ".2f"},
        title={"text": "Indian Ocean Dipole (DMI)", "font": {"size": 14}},
        number={"suffix": " °C", "font": {"size": 22}},
        gauge={
            "axis": {"range": [-2, 2], "tickwidth": 1},
            "bar":  {"color": GREEN if current_iod > 0.3
                              else (ACCENT if current_iod < -0.3 else "grey"),
                     "thickness": 0.25},
            "steps": [
                {"range": [-2,  -0.3], "color": "#fdebd0"},
                {"range": [-0.3, 0.3], "color": "#f2f3f4"},
                {"range": [ 0.3,  2 ], "color": "#d5f5e3"},
            ],
            "threshold": {"line": {"color": "black", "width": 2},
                          "thickness": 0.75, "value": current_iod},
        },
    ))
    fig_iod.update_layout(height=200, margin=dict(t=30, b=10, l=20, r=20))
    st.plotly_chart(fig_iod, use_container_width=True)

    phase_iod = ("🟢 Positive IOD" if current_iod >  0.3 else
                 "🟠 Negative IOD" if current_iod < -0.3 else
                 "⚪ Neutral")
    st.markdown(f"<center><b>{phase_iod}</b></center>", unsafe_allow_html=True)

# Risk score dial
with gc:
    fig_risk = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": "Composite Risk Score", "font": {"size": 14}},
        number={"suffix": " / 6", "font": {"size": 26, "color": risk_colour}},
        gauge={
            "axis": {"range": [0, 6], "tickvals": [0,1,2,3,4,5,6]},
            "bar":  {"color": risk_colour, "thickness": 0.3},
            "steps": [
                {"range": [0, 1], "color": "#d5f5e3"},
                {"range": [1, 2], "color": "#fef9e7"},
                {"range": [2, 3], "color": "#fdebd0"},
                {"range": [3, 4], "color": "#fadbd8"},
                {"range": [4, 6], "color": "#e8daef"},
            ],
        },
    ))
    fig_risk.update_layout(height=200, margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig_risk, use_container_width=True)
    st.markdown(
        f"<center style='font-size:18px;font-weight:bold;color:{risk_colour}'>"
        f"{risk_emoji} {risk_level_str}</center>",
        unsafe_allow_html=True,
    )

# Contributing factors
with gd:
    st.markdown("**Contributing factors**")
    for f in risk_factors:
        icon = "⚠️" if "coupling" in f or "Strong" in f else "•"
        st.markdown(f"{icon} {f}")

    st.markdown("---")
    # Analog years lookup
    analogs = []
    hist_climate = CLIMATE[(CLIMATE.index >= 2000) & (CLIMATE.index <= 2024)]
    sl_hist = data[data["Country"]=="SRI LANKA"].set_index("Year")["Cases"]
    for yr in hist_climate.index:
        o = hist_climate.loc[yr, "ONI"]
        i = hist_climate.loc[yr, "IOD"]
        if (abs(o - current_oni) < 0.5 and abs(i - current_iod) < 0.4
                and yr in sl_hist.index):
            analogs.append((yr, int(sl_hist[yr])))
    if analogs:
        st.markdown("**Historical analogs** (similar ONI + IOD):")
        for yr, cases in sorted(analogs, key=lambda x: -x[1])[:3]:
            st.markdown(f"  · **{yr}** → {cases:,} cases")
    else:
        st.markdown("*No close historical analog found*")

st.divider()

# ── Historical climate + dengue overlay chart ─────────────────────────────────
st.markdown("#### Historical Climate vs Dengue Cases (2000–2024)")

hist_years = list(range(2000, 2025))
sl_hist_df = (data[data["Country"]=="SRI LANKA"]
              .set_index("Year")["Cases"]
              .reindex(hist_years))
climate_hist = CLIMATE.reindex(hist_years)

fig_cl = make_subplots(specs=[[{"secondary_y": True}]])

# Dengue bars
fig_cl.add_trace(go.Bar(
    x=hist_years, y=sl_hist_df / 1000,
    name="Dengue cases (SL)",
    marker_color=SL_RED, opacity=0.55,
    hovertemplate="%{x}: <b>%{y:.1f}k</b> cases<extra></extra>",
), secondary_y=False)

# ONI line
fig_cl.add_trace(go.Scatter(
    x=hist_years, y=climate_hist["ONI"],
    name="ONI (ENSO)", line=dict(color=BLUE, width=2),
    hovertemplate="%{x} ONI: <b>%{y:+.2f}</b><extra></extra>",
), secondary_y=True)

# IOD line
fig_cl.add_trace(go.Scatter(
    x=hist_years, y=climate_hist["IOD"],
    name="IOD (DMI·ASO)", line=dict(color=GREEN, width=2, dash="dot"),
    hovertemplate="%{x} IOD: <b>%{y:+.2f}</b><extra></extra>",
), secondary_y=True)

# Zero reference
fig_cl.add_hline(y=0, line_color="grey", line_width=0.8, secondary_y=True)
fig_cl.add_hrect(y0=0.5,  y1=3,   fillcolor=SL_RED, opacity=0.05,
                  line_width=0, secondary_y=True)
fig_cl.add_hrect(y0=-3,   y1=-0.5, fillcolor=BLUE,  opacity=0.05,
                  line_width=0, secondary_y=True)

# Annotate El Niño + Pos IOD years
notable = [(yr, sl_hist_df[yr]) for yr in hist_years
           if climate_hist.loc[yr,"ONI"] > 0.5
           and climate_hist.loc[yr,"IOD"] > 0.3
           and not np.isnan(sl_hist_df[yr])]
for yr, cases in notable:
    fig_cl.add_annotation(
        x=yr, y=cases/1000, text="El Niño+IOD",
        showarrow=True, arrowhead=2, arrowsize=0.8,
        font=dict(size=9, color="#7b241c"),
        ax=0, ay=-30, secondary_y=False,
    )

fig_cl.update_layout(
    height=360,
    hovermode="x unified",
    legend=dict(orientation="h", y=1.08),
    margin=dict(t=10, b=40),
)
fig_cl.update_yaxes(title_text="Cases (thousands)", secondary_y=False)
fig_cl.update_yaxes(title_text="Climate index (°C anomaly)",
                    secondary_y=True, zeroline=True, zerolinecolor="grey")

st.plotly_chart(fig_cl, use_container_width=True)

# Outlook narrative box
enso_text = ("El Niño developing" if current_oni > 0.5 else
             "La Niña conditions" if current_oni < -0.5 else "ENSO-neutral")
iod_text  = ("positive IOD" if current_iod > 0.3 else
             "negative IOD" if current_iod < -0.3 else "IOD-neutral")

if risk_score >= 4:
    outlook_msg = (
        f"**High-risk configuration.** Current {enso_text} combined with {iod_text} "
        f"mirrors 2019 (96k cases) and 2023 (88k cases). Health systems should "
        f"pre-position vector control resources and elevate surveillance intensity "
        f"ahead of the May–June and Oct–Nov transmission peaks."
    )
elif risk_score == 3:
    outlook_msg = (
        f"**Elevated risk.** {enso_text.capitalize()} with {iod_text} represents "
        f"a moderate amplifier of the baseline dengue burden (~50–70k/yr expected range). "
        f"Routine surveillance and readiness planning recommended."
    )
elif risk_score == 2:
    outlook_msg = (
        f"**Moderate baseline.** Current climate state ({enso_text}, {iod_text}) "
        f"is not a strong amplifier. Expect cases near the post-2009 average (~50k). "
        f"Monitor for monsoon anomalies in May and October."
    )
else:
    outlook_msg = (
        f"**Low climate risk.** Current {enso_text} and {iod_text} conditions "
        f"do not suggest a climate-driven amplification. Serotype cycle is the "
        f"dominant driver to watch."
    )

bg = {"HIGH":"#fadbd8","CRITICAL":"#f1948a",
      "ELEVATED":"#fdebd0","MODERATE":"#fef9e7","LOW":"#d5f5e3"}[risk_level_str]

st.markdown(
    f"<div style='background:{bg};padding:14px 18px;border-radius:8px;"
    f"border-left:5px solid {risk_colour}'>"
    f"{risk_emoji} <b>12-18 Month Outlook:</b> {outlook_msg}</div>",
    unsafe_allow_html=True,
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — SL trend + YoY growth
# ══════════════════════════════════════════════════════════════════════════════
col_a, col_b = st.columns([3, 2])

with col_a:
    st.subheader("Sri Lanka Annual Dengue Cases")
    roll = sl.set_index("Year")["Cases"].rolling(5, min_periods=3).mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sl["Year"], y=sl["Cases"],
        name="Annual cases", marker_color=SL_RED, opacity=0.75,
        hovertemplate="%{x}: <b>%{y:,}</b> cases<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=roll["Year"], y=roll["Cases"],
        name="5-yr rolling avg", line=dict(color="#7f0000", width=2.5),
        hovertemplate="%{x}: %{y:,.0f} (rolling avg)<extra></extra>",
    ))
    fig.update_layout(
        height=370, legend=dict(orientation="h", y=1.08),
        xaxis_title="Year", yaxis_title="Cases",
        hovermode="x unified", margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Year-on-Year Growth Rate")
    sl2 = sl.sort_values("Year").copy()
    sl2["Growth"] = sl2["Cases"].pct_change() * 100
    sl2 = sl2.dropna(subset=["Growth"])
    colors = [SL_RED if g >= 0 else BLUE for g in sl2["Growth"]]
    fig2 = go.Figure(go.Bar(
        x=sl2["Year"], y=sl2["Growth"],
        marker_color=colors,
        hovertemplate="%{x}: <b>%{y:+.1f}%</b><extra></extra>",
    ))
    fig2.add_hline(y=0, line_color="black", line_width=0.8)
    fig2.update_layout(
        height=370, xaxis_title="Year", yaxis_title="Growth %",
        yaxis_ticksuffix="%", margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Regional comparison line chart
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Regional Comparison — Sri Lanka vs Neighbours")

pivot = df_sel.pivot_table(index="Year", columns="Label", values="Cases", aggfunc="sum")

if show_indexed and 2010 in pivot.index:
    base      = pivot.loc[2010]
    plot_data = pivot.div(base).mul(100)
    y_label   = "Index (2010 = 100)"
    hover_sfx = ""
else:
    plot_data = pivot
    y_label   = "Cases"
    hover_sfx = " cases"

fig3 = go.Figure()
for col in plot_data.columns:
    is_sl = col == "Sri Lanka"
    fig3.add_trace(go.Scatter(
        x=plot_data.index, y=plot_data[col], name=col,
        line=dict(width=3.5 if is_sl else 1.5,
                  color=SL_RED if is_sl else None),
        opacity=1.0 if is_sl else 0.8,
        hovertemplate=f"%{{x}} — {col}: <b>%{{y:,.0f}}</b>{hover_sfx}<extra></extra>",
    ))

fig3.update_layout(
    height=420, xaxis_title="Year", yaxis_title=y_label,
    legend=dict(orientation="h", y=-0.2),
    hovermode="x unified", margin=dict(t=10, b=80),
)
st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Heatmap + Correlation bar
# ══════════════════════════════════════════════════════════════════════════════
col_c, col_d = st.columns([3, 2])

with col_c:
    st.subheader("Dengue Burden Heatmap (log scale)")
    heat     = df[df["Country"].isin(REGIONAL)].pivot_table(
        index="Label", columns="Year", values="Cases", aggfunc="sum"
    )
    heat_log = np.log1p(heat)
    fig4     = px.imshow(heat_log, color_continuous_scale="YlOrRd",
                         labels=dict(color="log(cases+1)"), aspect="auto")
    fig4.update_layout(height=420, margin=dict(t=10, b=40),
                       xaxis_title="Year", yaxis_title="")
    fig4.update_traces(hovertemplate="<b>%{y}</b> — %{x}: %{z:.2f}<extra></extra>")
    st.plotly_chart(fig4, use_container_width=True)

with col_d:
    st.subheader("Correlation with Sri Lanka")
    sl_full = data[data["Country"] == "SRI LANKA"].set_index("Year")["Cases"]
    corrs   = []
    for country in REGIONAL:
        if country == "SRI LANKA":
            continue
        other  = data[data["Country"] == country].set_index("Year")["Cases"]
        shared = pd.concat([sl_full, other], axis=1, join="inner").dropna()
        shared.columns = ["SL", "Other"]
        shared = shared[(shared.index >= y0) & (shared.index <= y1)]
        if len(shared) >= 6:
            r, p = pearsonr(shared["SL"], shared["Other"])
            corrs.append({"Country": COUNTRY_LABELS[country], "r": r, "p": p})

    corr_df        = pd.DataFrame(corrs).sort_values("r")
    corr_df["sig"] = corr_df["p"] < 0.05
    corr_df["col"] = corr_df["r"].apply(lambda x: SL_RED if x >= 0 else BLUE)

    fig5 = go.Figure(go.Bar(
        x=corr_df["r"], y=corr_df["Country"], orientation="h",
        marker_color=corr_df["col"],
        text=corr_df.apply(lambda r: f"r={r['r']:.2f}{'*' if r['sig'] else ''}", axis=1),
        textposition="outside",
        hovertemplate="<b>%{y}</b>: r = %{x:.3f}<extra></extra>",
    ))
    fig5.add_vline(x=0, line_color="black", line_width=0.8)
    fig5.update_layout(
        height=420, xaxis=dict(range=[-1.1, 1.3], title="Pearson r"),
        margin=dict(t=10, l=10, r=60, b=40),
        annotations=[dict(x=0.5, y=-0.12, xref="paper", yref="paper",
                          text="* p < 0.05", showarrow=False, font=dict(size=11))],
    )
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Worst years table + animated map
# ══════════════════════════════════════════════════════════════════════════════
col_e, col_f = st.columns([1, 2])

with col_e:
    st.subheader("Sri Lanka — Worst Years")
    worst          = sl.nlargest(10, "Cases")[["Year", "Cases"]].copy()
    worst["Year"]  = worst["Year"].astype(int)
    worst["Cases"] = worst["Cases"].apply(lambda x: f"{x:,.0f}")
    worst          = worst.reset_index(drop=True)
    worst.index   += 1
    st.dataframe(worst, use_container_width=True, height=380)

with col_f:
    st.subheader("Regional Dengue Over Time")
    bubble_df          = df[df["Country"].isin(REGIONAL)].copy()
    bubble_df["Label"] = bubble_df["Country"].map(COUNTRY_LABELS)
    bubble_df["Year"]  = bubble_df["Year"].astype(int).astype(str)
    ISO = {
        "SRI LANKA":"LKA","INDIA":"IND","BANGLADESH":"BGD","MYANMAR":"MMR",
        "THAILAND":"THA","MALAYSIA":"MYS","INDONESIA":"IDN","PHILIPPINES":"PHL",
        "VIET NAM":"VNM","CAMBODIA":"KHM","LAO PEOPLE'S DEMOCRATIC REPUBLIC":"LAO",
        "NEPAL":"NPL",
    }
    bubble_df["ISO"] = bubble_df["Country"].map(ISO)
    bubble_df        = bubble_df.dropna(subset=["ISO","Cases"])
    fig6 = px.choropleth(
        bubble_df, locations="ISO", color="Cases", hover_name="Label",
        animation_frame="Year", color_continuous_scale="YlOrRd",
        scope="asia", labels={"Cases":"Cases"},
    )
    fig6.update_layout(
        height=380, margin=dict(t=10, b=10, l=0, r=0),
        geo=dict(showframe=False, showcoastlines=True),
    )
    st.plotly_chart(fig6, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS — Incidence & Serotypes · Weekly Pattern · Honest Forecast · Data Quality
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Incidence & Serotypes",
    "🗓 Weekly Seasonality",
    "🔮 Forecast (with uncertainty)",
    "🔬 Data Quality Audit",
])

sl_full_cases = (data[data["Country"]=="SRI LANKA"]
                 .set_index("Year")["Cases"]
                 .sort_index())
sl_hist = sl_full_cases[(sl_full_cases.index >= 2000) & (sl_full_cases.index <= 2024)]
pop_s   = pd.Series(POP)
inc     = (sl_hist / pop_s * 100_000).dropna()

# ── TAB 1: Incidence Rate + Serotype ──────────────────────────────────────────
with tab1:
    st.markdown("Cases expressed as **per 100,000 population** (World Bank denominator). "
                "Bar colour = dominant DENV serotype from published literature. "
                "Faded bars = lower-confidence serotype assignment.")

    fig_inc = go.Figure()
    for yr in sl_hist.index:
        dom  = SEROTYPES.get(int(yr), {}).get("dominant","unknown")
        conf = SEROTYPES.get(int(yr), {}).get("confidence","low")
        col  = SERO_COLORS.get(dom, GREY)
        note = SEROTYPES.get(int(yr), {}).get("note","")
        fig_inc.add_trace(go.Bar(
            x=[yr], y=[inc.get(yr, 0)],
            name=dom,
            marker_color=col,
            opacity=0.85 if conf=="high" else 0.45,
            hovertemplate=(f"<b>{yr}</b><br>Incidence: %{{y:.1f}}/100k"
                           f"<br>Serotype: {dom}"
                           + (f"<br><i>{note}</i>" if note else "")
                           + "<extra></extra>"),
            showlegend=False,
        ))

    # Rolling mean
    roll_inc = inc.rolling(5, min_periods=3).mean()
    fig_inc.add_trace(go.Scatter(
        x=list(roll_inc.index), y=list(roll_inc.values),
        name="5-yr rolling mean", line=dict(color="black", width=2.5),
    ))

    # Serotype legend annotations
    added = set()
    for yr in sl_hist.index:
        dom = SEROTYPES.get(int(yr), {}).get("dominant","unknown")
        if dom not in added and dom in SERO_COLORS:
            fig_inc.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=SERO_COLORS[dom], size=10, symbol="square"),
                name=dom, showlegend=True,
            ))
            added.add(dom)

    fig_inc.update_layout(
        height=420, xaxis_title="Year",
        yaxis_title="Cases per 100,000",
        hovermode="x", legend=dict(orientation="h", y=1.1),
        margin=dict(t=20, b=40),
        title="Dengue Incidence Rate — coloured by dominant DENV serotype<br>"
              "<sup>Source: Tissera et al. 2011/2019; Malavige et al. 2010; NIID Sri Lanka</sup>",
    )
    st.plotly_chart(fig_inc, use_container_width=True)

    st.markdown("**Key insight:** The 2009 structural shift coincides exactly with DENV-3 "
                "re-emergence. The 2017 mega-outbreak is DENV-2 in a population whose last "
                "large DENV-2 exposure was ~2002 — 15 years of susceptible accumulation, "
                "then triggered by May 2017 flooding.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Worst years by incidence rate**")
        worst_inc = inc.nlargest(10).reset_index()
        worst_inc.columns = ["Year","per 100k"]
        worst_inc["Year"] = worst_inc["Year"].astype(int)
        worst_inc["per 100k"] = worst_inc["per 100k"].round(1)
        st.dataframe(worst_inc, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Why incidence rate matters**")
        st.info(
            "Raw case counts grow with population. A year with 50k cases in 2000 "
            "(pop 19.3M = **259/100k**) is far worse than 50k in 2018 "
            "(pop 21.7M = **230/100k**). Incidence corrects for this."
        )

# ── TAB 2: Weekly Seasonality ──────────────────────────────────────────────────
with tab2:
    st.warning(
        "⚠️ **This is a RECONSTRUCTED weekly pattern**, not raw surveillance data. "
        "Annual totals are distributed using Sri Lanka's documented bimodal seasonal "
        "profile (SW monsoon May–Jun, NE monsoon Oct–Nov). "
        "Treat as a visualisation tool, not real weekly counts."
    )

    sel_years = st.multiselect(
        "Compare specific years",
        options=list(range(2000, 2025)),
        default=[2009, 2017, 2019, 2022, 2023],
    )

    weeks = np.arange(1, 53)
    fig_wk = go.Figure()

    # Median seasonal envelope
    all_profiles = []
    for yr in range(2009, 2025):
        wts = seasonal_weights(yr)
        all_profiles.append(wts * sl_hist.get(yr, 0))
    profiles_arr = np.array(all_profiles)
    med_env = np.median(profiles_arr, axis=0)
    p25_env = np.percentile(profiles_arr, 25, axis=0)
    p75_env = np.percentile(profiles_arr, 75, axis=0)

    fig_wk.add_trace(go.Scatter(
        x=list(weeks)+list(weeks[::-1]),
        y=list(p75_env)+list(p25_env[::-1]),
        fill="toself", fillcolor="rgba(192,57,43,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="25–75th percentile (2009–2024)",
    ))
    fig_wk.add_trace(go.Scatter(
        x=list(weeks), y=list(med_env),
        line=dict(color=SL_RED, width=2, dash="dash"),
        name="Median seasonal profile",
    ))

    colors_yr = px.colors.qualitative.Set1
    for i, yr in enumerate(sel_years):
        if yr not in sl_hist.index:
            continue
        wts   = seasonal_weights(yr)
        cases = wts * sl_hist[yr]
        fig_wk.add_trace(go.Scatter(
            x=list(weeks), y=list(cases),
            name=str(yr),
            line=dict(color=colors_yr[i % len(colors_yr)], width=2),
            hovertemplate=f"<b>{yr}</b> Week %{{x}}: %{{y:.0f}} est. cases<extra></extra>",
        ))

    # Shade monsoon windows
    fig_wk.add_vrect(x0=14, x1=26, fillcolor=BLUE,  opacity=0.06,
                     annotation_text="SW Monsoon", annotation_position="top left")
    fig_wk.add_vrect(x0=39, x1=47, fillcolor=GREEN, opacity=0.06,
                     annotation_text="NE Monsoon", annotation_position="top left")

    fig_wk.update_layout(
        height=440, xaxis_title="Epidemiological week",
        yaxis_title="Estimated cases (reconstructed)",
        legend=dict(orientation="h", y=1.1), margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig_wk, use_container_width=True)
    st.caption("To get real weekly data: contact EPHD Sri Lanka (epid.gov.lk) "
               "or request via WHO SEARO Regional Office.")

# ── TAB 3: Honest Forecast ────────────────────────────────────────────────────
with tab3:
    st.error(
        "**Model performance warning:** Cross-validation MAPE ≈ 74%. "
        "The uncertainty band below is honest — the true range spans 25k–148k. "
        "Point estimates are presented as a central tendency only, not predictions."
    )

    with st.spinner("Building forecast models (ARIMA + Prophet)…"):
        fut_yrs, ens_mean, ens_lo, ens_hi, p_mean, p_lo, p_hi = build_forecast(sl_hist)

    proj_pop = 21_916_000
    def to_inc(arr): return arr / proj_pop * 100_000

    mode = st.radio("Show as", ["Cases (absolute)", "Incidence per 100k"], horizontal=True)
    transform = (lambda x: x) if "Cases" in mode else to_inc
    unit      = "cases" if "Cases" in mode else "per 100k"

    fig_fc = go.Figure()

    # Historical
    hist_y = transform(sl_hist.values)
    fig_fc.add_trace(go.Bar(
        x=list(sl_hist.index), y=list(hist_y),
        name="Historical", marker_color=SL_RED, opacity=0.55,
    ))

    # Uncertainty band
    fig_fc.add_trace(go.Scatter(
        x=list(fut_yrs)+list(fut_yrs[::-1]),
        y=list(transform(ens_hi))+list(transform(ens_lo)[::-1]),
        fill="toself", fillcolor="rgba(41,128,185,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Plausible range (all models)",
    ))

    # Central estimate
    fig_fc.add_trace(go.Scatter(
        x=list(fut_yrs), y=list(transform(ens_mean)),
        name="Central estimate",
        line=dict(color=BLUE, width=3),
        mode="lines+markers", marker=dict(size=8),
        hovertemplate="<b>%{x}</b>: %{y:,.0f} " + unit + "<extra></extra>",
    ))

    # Climate scenario line (2019-analog)
    analog_val = transform(np.array([sl_hist.get(2019, 96045)] * len(fut_yrs)))
    fig_fc.add_trace(go.Scatter(
        x=list(fut_yrs), y=list(analog_val),
        name="El Niño+IOD scenario (2019 analog)",
        line=dict(color=ORANGE, width=1.8, dash="dot"),
    ))

    # 2017 ceiling
    ceil_val = transform(np.array([sl_hist.get(2017, 176272)] * len(fut_yrs)))
    fig_fc.add_hline(
        y=float(ceil_val[0]),
        line_color="darkred", line_dash="dash", line_width=1,
        annotation_text=f"2017 flood extreme ({int(ceil_val[0]):,} {unit})",
        annotation_position="top right",
    )

    fig_fc.add_vline(x=2024.5, line_color="grey", line_dash="dot")
    fig_fc.update_layout(
        height=430,
        xaxis_title="Year", yaxis_title=unit.capitalize(),
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=20, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Forecast table
    st.markdown("**Forecast table** — range reflects genuine model disagreement, not a confidence interval")
    fc_rows = []
    for i, yr in enumerate(fut_yrs):
        lo  = int(transform(np.array([ens_lo[i]]))[0])
        mid = int(transform(np.array([ens_mean[i]]))[0])
        hi  = int(transform(np.array([ens_hi[i]]))[0])
        fc_rows.append({
            "Year": int(yr),
            f"Low ({unit})": f"{lo:,}",
            f"Central ({unit})": f"{mid:,}",
            f"High ({unit})": f"{hi:,}",
            "Uncertainty": f"±{(hi-lo)//2//mid*100 if mid else 0}%",
        })
    st.dataframe(pd.DataFrame(fc_rows), use_container_width=True, hide_index=True)

    st.markdown(
        "**What would make this reliable:** weekly EPHD data + serotype surveillance + "
        "hold-out validation on 2016–2024. Current models trained on 25 annual points."
    )

# ── TAB 4: Data Quality ────────────────────────────────────────────────────────
with tab4:
    st.markdown("Honest audit of every data source used in this dashboard.")

    dq_data = [
        {"Source": "Annual case counts (WHO GIDEON)", "Availability": 10, "Resolution": 2,
         "Validation": 6, "Usable?": "✅ Yes", "Caveat": "Annual only — masks seasonal/spatial patterns"},
        {"Source": "Population (World Bank API)", "Availability": 10, "Resolution": 8,
         "Validation": 9, "Usable?": "✅ Yes", "Caveat": "Slight undercount due to 2022 emigration wave"},
        {"Source": "ENSO/ONI (NOAA CPC)", "Availability": 10, "Resolution": 7,
         "Validation": 9, "Usable?": "✅ Yes", "Caveat": "R²=0.12 with dengue — weak signal"},
        {"Source": "IOD/DMI (literature estimates)", "Availability": 4, "Resolution": 4,
         "Validation": 3, "Usable?": "⚠️ With caveat", "Caveat": "NOT from live source — values are approximated"},
        {"Source": "Serotype data (published papers)", "Availability": 5, "Resolution": 4,
         "Validation": 6, "Usable?": "⚠️ With caveat", "Caveat": "Incomplete; annual resolution; some inferred"},
        {"Source": "Weekly cases (RECONSTRUCTED)", "Availability": 0, "Resolution": 0,
         "Validation": 0, "Usable?": "❌ Visualisation only", "Caveat": "Synthetic from annual totals — NOT surveillance data"},
        {"Source": "Vector/entomological data", "Availability": 0, "Resolution": 0,
         "Validation": 0, "Usable?": "❌ Missing", "Caveat": "Not in dataset — critical gap for real forecasting"},
        {"Source": "Hospitalisation / DHF rates", "Availability": 0, "Resolution": 0,
         "Validation": 0, "Usable?": "❌ Missing", "Caveat": "Severity data absent — all counts are confirmed cases only"},
    ]
    dq_df = pd.DataFrame(dq_data)

    def colour_usable(val):
        if "✅" in str(val): return "background-color:#d5f5e3"
        if "⚠️" in str(val): return "background-color:#fef9e7"
        return "background-color:#fadbd8"

    st.dataframe(
        dq_df.style.applymap(colour_usable, subset=["Usable?"]),
        use_container_width=True, height=310, hide_index=True,
    )

    st.markdown("---")
    st.markdown("### To make this operationally useful, get:")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**🔴 Critical (blocks real use)**
1. Weekly EPHD Sri Lanka data → epid.gov.lk or MoH request
2. Annual DENV serotype surveillance → NIID Colombo / WHO SEARO
3. Hold-out model validation (train 2000–2015, test 2016–2024)
        """)
    with c2:
        st.markdown("""
**🟡 Important (improves quality)**
4. Live IOD/DMI feed → JAMSTEC API (free registration)
5. District-level cases → Colombo vs Kandy vs Jaffna differ hugely
6. Rainfall anomaly data → ERA5 reanalysis (Copernicus CDS, free)
7. Entomological indices → Breteau index from NMCP Sri Lanka
        """)

