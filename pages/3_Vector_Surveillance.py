"""
Vector Surveillance & Control — Sri Lanka Dengue Dashboard
Entomological indices: House Index, Container Index, Breteau Index, Ovitrap Index
WHO risk thresholds applied per district / MOH area
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta

st.set_page_config(
    page_title="Vector Surveillance — Sri Lanka",
    page_icon="🦟",
    layout="wide",
)

st.markdown("""
<style>
.fixed-footer{position:fixed;bottom:0;left:0;width:100%;background:#111;
color:#aaa;text-align:center;padding:6px;font-size:12px;z-index:9999;
pointer-events:none;user-select:none}
.risk-badge{display:inline-block;padding:4px 12px;border-radius:20px;
font-weight:700;font-size:13px}
</style>
<div class="fixed-footer">Experiment by Ranga Dayawansha | Founder of FloodSupport.org</div>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
SL_RED   = "#c0392b"
ORANGE   = "#e67e22"
YELLOW   = "#f1c40f"
GREEN    = "#27ae60"
BLUE     = "#2980b9"
PURPLE   = "#8e44ad"

SURVEY_CSV = Path("data_sources/vector_surveys.csv")
SURVEY_CSV.parent.mkdir(exist_ok=True)

DISTRICTS = [
    "Colombo","Gampaha","Kalutara","Kandy","Matale","Nuwara Eliya",
    "Galle","Matara","Hambantota","Jaffna","Kilinochchi","Mannar",
    "Vavuniya","Mullaitivu","Batticaloa","Ampara","Trincomalee",
    "Kurunegala","Puttalam","Anuradhapura","Polonnaruwa","Badulla",
    "Monaragala","Ratnapura","Kegalle",
]

CONTAINER_TYPES = [
    "Water storage drums/tanks",
    "Flower pots & saucers",
    "Used tyres",
    "Construction site water",
    "Coconut shells",
    "Roof gutters & drains",
    "Overhead tanks (uncovered)",
    "Bottles & cans",
    "Other",
]

PROVINCE_MAP = {
    "Colombo":"Western","Gampaha":"Western","Kalutara":"Western",
    "Kandy":"Central","Matale":"Central","Nuwara Eliya":"Central",
    "Galle":"Southern","Matara":"Southern","Hambantota":"Southern",
    "Jaffna":"Northern","Kilinochchi":"Northern","Mannar":"Northern",
    "Vavuniya":"Northern","Mullaitivu":"Northern",
    "Batticaloa":"Eastern","Ampara":"Eastern","Trincomalee":"Eastern",
    "Kurunegala":"North Western","Puttalam":"North Western",
    "Anuradhapura":"North Central","Polonnaruwa":"North Central",
    "Badulla":"Uva","Monaragala":"Uva",
    "Ratnapura":"Sabaragamuwa","Kegalle":"Sabaragamuwa",
}

# WHO risk thresholds for Breteau Index
def bi_risk(bi):
    if bi is None or np.isnan(bi): return "Unknown", BLUE
    if bi < 5:   return "Low",       GREEN
    if bi < 20:  return "Moderate",  YELLOW
    if bi < 50:  return "High",      ORANGE
    return              "Critical",  SL_RED

def hi_risk(hi):
    if hi is None or np.isnan(hi): return "Unknown", BLUE
    if hi < 1:  return "Low",       GREEN
    if hi < 5:  return "Moderate",  YELLOW
    if hi < 10: return "High",      ORANGE
    return              "Critical",  SL_RED

# ── Generate demo data if CSV doesn't exist ────────────────────────────────────
def generate_demo_data():
    rng = np.random.default_rng(42)
    rows = []
    # Base BI by province (reflects urbanisation + case burden)
    base_bi = {
        "Western":30,"Southern":22,"Central":18,"Sabaragamuwa":16,
        "North Western":12,"Eastern":10,"North Central":9,
        "Uva":8,"Northern":6,
    }
    container_weights = [0.28,0.18,0.14,0.12,0.09,0.08,0.06,0.03,0.02]

    for district in DISTRICTS:
        prov  = PROVINCE_MAP[district]
        base  = base_bi[prov]
        # 10 weeks of data (weeks 15–24, 2026)
        for wk in range(15, 25):
            bi     = max(1, rng.normal(base, base * 0.25))
            houses = int(rng.integers(80, 250))
            hi     = max(0.5, rng.normal(bi / 3, 2))
            ci     = max(0.3, hi * rng.uniform(0.4, 0.7))
            pos_containers = int(round(bi * houses / 100))
            total_containers = int(pos_containers / max(ci/100, 0.01))

            # container type split
            ct_counts = rng.multinomial(pos_containers, container_weights)
            ct_dict   = {CONTAINER_TYPES[i]: int(ct_counts[i]) for i in range(len(CONTAINER_TYPES))}

            ovitraps  = int(rng.integers(10, 30))
            oi        = round(float(rng.uniform(max(0, hi-5), hi+20)), 1)

            survey_date = date(2026, 1, 1) + timedelta(weeks=wk - 1)
            row = {
                "date":      survey_date.isoformat(),
                "week":      wk,
                "year":      2026,
                "district":  district,
                "province":  prov,
                "moh_area":  f"{district} Central",
                "houses_inspected":   houses,
                "houses_positive":    int(round(houses * hi / 100)),
                "containers_inspected": total_containers,
                "containers_positive":  pos_containers,
                "ovitraps_deployed":  ovitraps,
                "ovitraps_positive":  int(round(ovitraps * oi / 100)),
                "HI": round(hi, 1),
                "CI": round(ci, 1),
                "BI": round(bi, 1),
                "OI": round(oi, 1),
                "source": "demo",
            }
            row.update(ct_dict)
            rows.append(row)
    return pd.DataFrame(rows)

# ── Load / init data ───────────────────────────────────────────────────────────
if SURVEY_CSV.exists():
    df = pd.read_csv(SURVEY_CSV, parse_dates=["date"])
else:
    df = generate_demo_data()
    df.to_csv(SURVEY_CSV, index=False)

# Ensure container type columns exist
for ct in CONTAINER_TYPES:
    if ct not in df.columns:
        df[ct] = 0

# ── Sidebar: data entry form ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Log New Survey")
    with st.form("survey_form", clear_on_submit=True):
        s_date     = st.date_input("Survey date", value=date.today())
        s_district = st.selectbox("District", DISTRICTS)
        s_moh      = st.text_input("MOH Area", placeholder="e.g. Maharagama")
        st.markdown("**Larval survey**")
        s_houses   = st.number_input("Houses inspected",     min_value=1, value=100)
        s_h_pos    = st.number_input("Houses positive",      min_value=0, value=10)
        s_cont     = st.number_input("Containers inspected", min_value=1, value=180)
        s_c_pos    = st.number_input("Containers positive",  min_value=0, value=20)
        st.markdown("**Container types (positive)**")
        ct_vals = {}
        for ct in CONTAINER_TYPES:
            ct_vals[ct] = st.number_input(ct, min_value=0, value=0, key=f"ct_{ct}")
        st.markdown("**Ovitrap survey**")
        s_ov_dep  = st.number_input("Ovitraps deployed", min_value=0, value=20)
        s_ov_pos  = st.number_input("Ovitraps positive", min_value=0, value=5)
        submitted = st.form_submit_button("✅ Save Survey", type="primary")

    if submitted:
        hi = round(s_h_pos / s_houses * 100, 1) if s_houses > 0 else 0
        ci = round(s_c_pos / s_cont    * 100, 1) if s_cont   > 0 else 0
        bi = round(s_c_pos / s_houses  * 100, 1) if s_houses > 0 else 0
        oi = round(s_ov_pos / s_ov_dep * 100, 1) if s_ov_dep > 0 else 0
        iso_week = s_date.isocalendar().week
        new_row = {
            "date": s_date.isoformat(), "week": iso_week, "year": s_date.year,
            "district": s_district, "province": PROVINCE_MAP.get(s_district,""),
            "moh_area": s_moh or f"{s_district} Central",
            "houses_inspected": s_houses, "houses_positive": s_h_pos,
            "containers_inspected": s_cont, "containers_positive": s_c_pos,
            "ovitraps_deployed": s_ov_dep, "ovitraps_positive": s_ov_pos,
            "HI": hi, "CI": ci, "BI": bi, "OI": oi, "source": "field",
        }
        new_row.update(ct_vals)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(SURVEY_CSV, index=False)
        st.success(f"Saved! BI={bi}, HI={hi}%, CI={ci}%")

    st.divider()
    st.markdown("**WHO Breteau Index thresholds**")
    st.markdown("""
🟢 **< 5** — Low risk
🟡 **5–20** — Moderate
🟠 **20–50** — High
🔴 **≥ 50** — Critical / epidemic potential
    """)

# ── Main page ──────────────────────────────────────────────────────────────────
st.title("🦟 Vector Surveillance & Control")
st.caption("Entomological indices · WHO risk thresholds · Container breeding analysis")

# Filter controls
col_f1, col_f2, col_f3 = st.columns(3)
years    = sorted(df["year"].dropna().unique().astype(int).tolist(), reverse=True)
sel_year = col_f1.selectbox("Year", years)
weeks    = sorted(df[df["year"]==sel_year]["week"].dropna().unique().astype(int).tolist(), reverse=True)
sel_week = col_f2.selectbox("Reference week", weeks)
view_src = col_f3.selectbox("Data source", ["All","Field data only","Demo data only"])

src_filter = {"All": None, "Field data only": "field", "Demo data only": "demo"}[view_src]
dfw = df[(df["year"]==sel_year) & (df["week"]==sel_week)].copy()
if src_filter:
    dfw = dfw[dfw["source"]==src_filter]

# Latest per district for the selected week
latest = (dfw.sort_values("date")
            .drop_duplicates(subset="district", keep="last")
            .copy())
latest["risk_label"], latest["risk_color"] = zip(*latest["BI"].apply(bi_risk))

st.divider()

# ── Hero metrics ───────────────────────────────────────────────────────────────
total_surveys  = len(dfw)
avg_bi         = latest["BI"].mean()
avg_hi         = latest["HI"].mean()
high_risk_ct   = (latest["BI"] >= 20).sum()
critical_ct    = (latest["BI"] >= 50).sum()
field_entries  = (df["source"]=="field").sum()

h1,h2,h3,h4,h5 = st.columns(5)
h1.metric("Districts surveyed",  f"{len(latest)}/25")
h2.metric("Avg Breteau Index",   f"{avg_bi:.1f}", "WHO risk >20")
h3.metric("Avg House Index",     f"{avg_hi:.1f}%","WHO risk >5%")
h4.metric("High-risk districts", str(high_risk_ct), f"{critical_ct} critical")
h5.metric("Field surveys logged",str(field_entries))

st.divider()

# ── District Risk Table ────────────────────────────────────────────────────────
st.subheader("📍 District Risk Overview — Breteau Index")

left, right = st.columns([1.8, 1])

with left:
    display = latest[["district","province","BI","HI","CI","OI","risk_label"]].copy()
    display = display.sort_values("BI", ascending=False).reset_index(drop=True)
    display.columns = ["District","Province","Breteau Index","House Index %",
                       "Container Index %","Ovitrap Index %","Risk Level"]

    def color_risk(val):
        colors = {"Low":GREEN,"Moderate":YELLOW,"High":ORANGE,"Critical":SL_RED}
        c = colors.get(val, "#555")
        return f"background-color:{c};color:{'#111' if val=='Moderate' else 'white'};font-weight:700"

    styled = display.style.applymap(color_risk, subset=["Risk Level"])
    st.dataframe(styled, hide_index=True, use_container_width=True, height=520)

with right:
    # BI horizontal bar with threshold lines
    plot_d = display.sort_values("Breteau Index", ascending=True)
    bar_colors = [
        SL_RED if v >= 50 else ORANGE if v >= 20 else YELLOW if v >= 5 else GREEN
        for v in plot_d["Breteau Index"]
    ]
    fig_bi = go.Figure()
    fig_bi.add_bar(
        y=plot_d["District"], x=plot_d["Breteau Index"],
        orientation="h", marker_color=bar_colors,
        text=[f"{v:.0f}" for v in plot_d["Breteau Index"]],
        textposition="outside",
    )
    for thresh, label, color in [(5,"Low/Mod",YELLOW),(20,"Mod/High",ORANGE),(50,"High/Critical",SL_RED)]:
        fig_bi.add_vline(x=thresh, line_dash="dash", line_color=color,
                         annotation_text=label, annotation_font_size=9)
    fig_bi.update_layout(
        height=540, xaxis_title="Breteau Index",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", margin=dict(l=5,r=50,t=10,b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_bi, use_container_width=True)

st.divider()

# ── Index comparison: HI, CI, BI, OI ──────────────────────────────────────────
st.subheader("📊 Index Comparison by District")
index_choice = st.radio(
    "Index", ["Breteau Index (BI)","House Index % (HI)","Container Index % (CI)","Ovitrap Index % (OI)"],
    horizontal=True
)
idx_col = {"Breteau Index (BI)":"BI","House Index % (HI)":"HI",
           "Container Index % (CI)":"CI","Ovitrap Index % (OI)":"OI"}[index_choice]

top_d = latest.sort_values(idx_col, ascending=False)
bar_c2 = [
    SL_RED if v >= 50 else ORANGE if v >= 20 else YELLOW if v >= 5 else GREEN
    for v in top_d[idx_col]
] if idx_col == "BI" else BLUE

fig_idx = go.Figure(go.Bar(
    x=top_d["district"], y=top_d[idx_col],
    marker_color=bar_c2,
    text=[f"{v:.1f}" for v in top_d[idx_col]],
    textposition="outside",
))
fig_idx.update_layout(
    height=380, yaxis_title=index_choice,
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font_color="#ccc", margin=dict(l=10,r=10,t=20,b=10),
)
st.plotly_chart(fig_idx, use_container_width=True)

st.divider()

# ── Weekly trend ───────────────────────────────────────────────────────────────
st.subheader("📈 Breteau Index Trend — Weekly")

sel_districts = st.multiselect(
    "Districts to track",
    DISTRICTS,
    default=["Colombo","Gampaha","Kandy","Galle","Ratnapura"],
)

df_trend = df[
    (df["year"]==sel_year) &
    (df["district"].isin(sel_districts))
].copy()
if src_filter:
    df_trend = df_trend[df_trend["source"]==src_filter]
df_trend = (df_trend.groupby(["week","district"])["BI"]
            .mean().reset_index())

fig_trend = go.Figure()
palette = [SL_RED, BLUE, "#27ae60", ORANGE, PURPLE, "#f1c40f", "#1abc9c"]
for i, dist in enumerate(sel_districts):
    d = df_trend[df_trend["district"]==dist]
    if d.empty: continue
    fig_trend.add_scatter(
        x=d["week"], y=d["BI"], mode="lines+markers",
        name=dist, line=dict(color=palette[i % len(palette)], width=2),
        marker=dict(size=6),
    )
for thresh, label, color in [(5,"Low/Mod",YELLOW),(20,"Mod/High",ORANGE),(50,"Critical",SL_RED)]:
    fig_trend.add_hline(y=thresh, line_dash="dot", line_color=color,
                        annotation_text=label, annotation_position="right")

fig_trend.update_layout(
    height=380, xaxis_title="Epidemiological week", yaxis_title="Breteau Index",
    legend=dict(orientation="h", y=1.05),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font_color="#ccc", margin=dict(l=10,r=80,t=30,b=10),
)
st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ── Container type analysis ────────────────────────────────────────────────────
st.subheader("🪣 Breeding Container Analysis")

ct_left, ct_right = st.columns(2)

with ct_left:
    ct_totals = {}
    for ct in CONTAINER_TYPES:
        if ct in dfw.columns:
            ct_totals[ct] = dfw[ct].sum()

    if any(v > 0 for v in ct_totals.values()):
        ct_df = pd.Series(ct_totals).sort_values(ascending=False)
        fig_ct = go.Figure(go.Pie(
            labels=ct_df.index,
            values=ct_df.values,
            hole=0.45,
            marker_colors=[SL_RED, ORANGE, BLUE, PURPLE, GREEN, YELLOW, "#1abc9c", "#e91e8c", GREY
                           if len(ct_df) > 8 else GREEN],
            textinfo="label+percent",
        ))
        fig_ct.update_layout(
            height=380, title=f"Container type distribution — Week {sel_week}",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc", showlegend=False,
            margin=dict(l=10,r=10,t=40,b=10),
        )
        st.plotly_chart(fig_ct, use_container_width=True)
    else:
        st.info("No container type data recorded for this week yet.")

with ct_right:
    st.markdown("**Why containers matter**")
    st.markdown("""
Container type data tells you **where interventions will have the biggest impact**:

| Container | Action |
|-----------|--------|
| 🛢 Water drums/tanks | Cover or treat with Abate |
| 🌸 Flower pots | Empty saucers weekly |
| 🚗 Used tyres | Puncture or remove |
| 🏗 Construction sites | Drain & cover |
| 🥥 Coconut shells | Clear from surroundings |
| 🏠 Roof gutters | Flush & clear monthly |
| 🪣 Overhead tanks | Screen all openings |

Targeting the **top 3 container types** typically eliminates 70–80% of breeding sites.
    """)

    if any(v > 0 for v in ct_totals.values()):
        top3 = sorted(ct_totals.items(), key=lambda x: x[1], reverse=True)[:3]
        st.markdown(f"**Top 3 this week:**")
        for rank, (name, count) in enumerate(top3, 1):
            st.markdown(f"{rank}. **{name}** — {int(count)} positive containers")

st.divider()

# ── Province heatmap ───────────────────────────────────────────────────────────
st.subheader("🗺 Province-level Vector Risk Heatmap")

prov_avg = (latest.groupby("province")
            .agg(avg_BI=("BI","mean"), avg_HI=("HI","mean"),
                 avg_CI=("CI","mean"), n_districts=("district","count"))
            .round(1).reset_index()
            .sort_values("avg_BI", ascending=False))
prov_avg["Risk"] = prov_avg["avg_BI"].apply(lambda b: bi_risk(b)[0])

fig_prov = px.treemap(
    prov_avg,
    path=["province"],
    values="avg_BI",
    color="avg_BI",
    color_continuous_scale=["#27ae60","#f1c40f","#e67e22","#c0392b"],
    color_continuous_midpoint=20,
    custom_data=["avg_HI","avg_CI","Risk","n_districts"],
)
fig_prov.update_traces(
    hovertemplate=(
        "<b>%{label}</b><br>"
        "Avg BI: %{value:.1f}<br>"
        "Avg HI: %{customdata[0]:.1f}%<br>"
        "Avg CI: %{customdata[1]:.1f}%<br>"
        "Risk: %{customdata[2]}<br>"
        "Districts: %{customdata[3]}"
        "<extra></extra>"
    )
)
fig_prov.update_layout(
    height=380,
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font_color="#111", margin=dict(l=5,r=5,t=10,b=5),
    coloraxis_colorbar=dict(title="Avg BI"),
)
st.plotly_chart(fig_prov, use_container_width=True)

st.divider()

# ── Survey log ─────────────────────────────────────────────────────────────────
with st.expander("📄 Full Survey Log", expanded=False):
    show_cols = ["date","week","district","moh_area","houses_inspected",
                 "houses_positive","containers_positive","HI","CI","BI","OI","source"]
    log_df = df[show_cols].sort_values(["date","district"], ascending=False)
    log_df["source"] = log_df["source"].map({"field":"✅ Field","demo":"🔵 Demo"}).fillna("🔵 Demo")
    st.dataframe(log_df.rename(columns={
        "date":"Date","week":"Wk","district":"District","moh_area":"MOH Area",
        "houses_inspected":"Houses","houses_positive":"H+",
        "containers_positive":"C+","source":"Source",
    }), hide_index=True, use_container_width=True, height=400)

    csv_dl = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download all survey data (CSV)",
                       csv_dl, "vector_surveys.csv", "text/csv")

st.caption(
    "Indices: HI = House Index, CI = Container Index, BI = Breteau Index, OI = Ovitrap Index. "
    "WHO epidemic threshold: BI ≥ 50, HI ≥ 5%. "
    "Demo data generated for illustration — replace with real NDCU entomological survey data."
)

GREY = "#7f8c8d"
