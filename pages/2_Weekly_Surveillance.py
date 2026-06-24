"""
Weekly Dengue Surveillance — Sri Lanka
Data: wer_district_weekly.csv (real WER data, NDCU)
Add new weeks by appending rows to data_sources/wer_district_weekly.csv
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Weekly Dengue Surveillance — Sri Lanka",
    page_icon="🦟",
    layout="wide",
)

st.markdown("""
<style>
.fixed-footer{position:fixed;bottom:0;left:0;width:100%;background:#111;
color:#aaa;text-align:center;padding:6px;font-size:12px;z-index:9999;
pointer-events:none;user-select:none}
</style>
<div class="fixed-footer">Experiment by Ranga Dayawansha | Founder of FloodSupport.org</div>
""", unsafe_allow_html=True)

SL_RED = "#c0392b"; BLUE = "#2980b9"; ORANGE = "#e67e22"
GREEN  = "#27ae60"; PURPLE = "#8e44ad"

DATA_CSV = Path("data_sources/wer_district_weekly.csv")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(DATA_CSV, parse_dates=["week_start","week_end"])
    df["week"]  = df["week"].astype(int)
    df["year"]  = df["year"].astype(int)
    return df

df = load_data()

available_weeks = (df.sort_values(["year","week"])
                     [["year","week","week_start","week_end"]]
                     .drop_duplicates()
                     .reset_index(drop=True))

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📅 Select Report Week")
    week_options = {
        f"Week {r.week} · {r.year}  ({r.week_start.strftime('%d %b')}–{r.week_end.strftime('%d %b %Y')})": (r.year, r.week)
        for r in available_weeks.itertuples()
    }
    sel_label  = st.selectbox("Report", list(reversed(list(week_options.keys()))))
    sel_year, sel_week = week_options[sel_label]

    st.divider()
    st.markdown("**ℹ️ Adding new weeks**")
    st.markdown(
        "Paste new WER district data into  \n"
        "`data_sources/wer_district_weekly.csv`  \n"
        "following the same column format."
    )

# ── Filter to selected week ────────────────────────────────────────────────────
dfw = df[(df["year"]==sel_year) & (df["week"]==sel_week)].copy()
dfw = dfw.sort_values("weekly_cases", ascending=False)

# Previous available week for comparison
prev_rows = available_weeks[
    (available_weeks["year"] < sel_year) |
    ((available_weeks["year"] == sel_year) & (available_weeks["week"] < sel_week))
]
has_prev  = not prev_rows.empty
if has_prev:
    prev_r    = prev_rows.iloc[-1]
    dfp       = df[(df["year"]==prev_r.year) & (df["week"]==prev_r.week)].copy()
    prev_label= f"Week {prev_r.week} · {prev_r.year}"
else:
    dfp = pd.DataFrame()
    prev_label = "previous week"

week_row = available_weeks[(available_weeks["year"]==sel_year)&(available_weeks["week"]==sel_week)].iloc[0]
date_range = f"{week_row.week_start.strftime('%d %b')}–{week_row.week_end.strftime('%d %b %Y')}"

# ── Page header ────────────────────────────────────────────────────────────────
st.title("🦟 Weekly Dengue Surveillance — Sri Lanka")
st.markdown(
    f"**Week {sel_week} ({date_range})** &nbsp;·&nbsp; "
    f"Source: [National Dengue Control Unit (NDCU)](https://www.epid.gov.lk/) — NaDSys"
)

# ── Hero metrics ───────────────────────────────────────────────────────────────
total_wk  = int(dfw["weekly_cases"].sum())
total_cum = int(dfw["cumulative_cases"].sum())
top_dist  = dfw.iloc[0]["district"] if not dfw.empty else "—"
top_cases = int(dfw.iloc[0]["weekly_cases"]) if not dfw.empty else 0

prev_total = int(dfp["weekly_cases"].sum()) if has_prev else None
wow_delta  = f"{(total_wk - prev_total)/prev_total*100:+.1f}% vs {prev_label}" if prev_total else "—"
wow_color  = "inverse" if prev_total and total_wk > prev_total else "normal"

# Compare cumulative with same-week last year if available
same_wk_ly = df[(df["year"]==sel_year-1)&(df["week"]==sel_week)]
cum_ly = int(same_wk_ly["cumulative_cases"].sum()) if not same_wk_ly.empty else None
cum_delta = f"{(total_cum-cum_ly)/cum_ly*100:+.1f}% vs {sel_year-1}" if cum_ly else "—"

# Province breakdown
prov_total = dfw.groupby("province")["weekly_cases"].sum()
top_prov   = prov_total.idxmax() if not prov_total.empty else "—"
top_prov_pct = f"{prov_total.max()/total_wk*100:.0f}% of cases" if total_wk else "—"

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric(f"Week {sel_week} Total",   f"{total_wk:,}",   wow_delta,  delta_color=wow_color)
m2.metric(f"Cumulative {sel_year}",   f"{total_cum:,}",  cum_delta)
m3.metric("Highest district",          top_dist,          f"{top_cases:,} cases")
m4.metric("Dominant province",          top_prov,          top_prov_pct)
m5.metric("Districts reporting",        f"{len(dfw)}",    f"of 26 RDHS areas")

st.divider()

# ── District bar chart ─────────────────────────────────────────────────────────
st.subheader(f"📍 Cases by District — Week {sel_week}, {sel_year}")

view = st.radio("Show", ["Weekly cases","Cumulative cases","Week-on-week change"],
                horizontal=True)

dfw_plot = dfw.sort_values("weekly_cases", ascending=True)

if view == "Weekly cases":
    fig = go.Figure()
    if has_prev and not dfp.empty:
        merged = dfw_plot.merge(dfp[["district","weekly_cases"]], on="district",
                                suffixes=("","_prev"), how="left")
        fig.add_bar(y=merged["district"], x=merged["weekly_cases_prev"],
                    name=prev_label, orientation="h",
                    marker_color=BLUE, opacity=0.65)
        fig.add_bar(y=merged["district"], x=merged["weekly_cases"],
                    name=f"Week {sel_week}", orientation="h",
                    marker_color=SL_RED, opacity=0.9)
        fig.update_layout(barmode="overlay")
    else:
        fig.add_bar(y=dfw_plot["district"], x=dfw_plot["weekly_cases"],
                    orientation="h", marker_color=SL_RED,
                    text=dfw_plot["weekly_cases"], textposition="outside")
    title_str = f"Suspected dengue cases — Week {sel_week}"

elif view == "Cumulative cases":
    dfw_plot = dfw.sort_values("cumulative_cases", ascending=True)
    fig = go.Figure()
    if cum_ly is not None:
        ly_data = same_wk_ly.sort_values("cumulative_cases", ascending=True)
        fig.add_bar(y=ly_data["district"], x=ly_data["cumulative_cases"],
                    name=f"{sel_year-1} (same week)", orientation="h",
                    marker_color=BLUE, opacity=0.65)
    fig.add_bar(y=dfw_plot["district"], x=dfw_plot["cumulative_cases"],
                name=f"{sel_year} cumulative", orientation="h",
                marker_color=SL_RED, opacity=0.9)
    fig.update_layout(barmode="overlay")
    title_str = f"Cumulative cases up to Week {sel_week}"

else:  # WoW change
    if has_prev and not dfp.empty:
        merged = dfw.merge(dfp[["district","weekly_cases"]], on="district",
                           suffixes=("","_prev"), how="left").fillna(0)
        merged["change"] = merged["weekly_cases"] - merged["weekly_cases_prev"]
        merged = merged.sort_values("change", ascending=True)
        colors = [SL_RED if x > 0 else GREEN for x in merged["change"]]
        fig = go.Figure(go.Bar(
            y=merged["district"], x=merged["change"], orientation="h",
            marker_color=colors,
            text=[f"{v:+d}" for v in merged["change"]], textposition="outside",
        ))
        title_str = f"Change vs {prev_label}"
    else:
        st.info("Need at least two weeks of data to show week-on-week change.")
        fig = go.Figure()
        title_str = ""

fig.update_layout(
    height=580, xaxis_title=title_str,
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font_color="#ccc", legend=dict(orientation="h", y=1.02),
    margin=dict(l=10, r=60, t=20, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# Province summary table
st.markdown(f"**Province Summary — Week {sel_week}, {sel_year}**")
prov_df = (dfw.groupby("province")
              .agg(weekly_cases=("weekly_cases","sum"),
                   cumulative_cases=("cumulative_cases","sum"),
                   districts=("district","count"))
              .sort_values("weekly_cases", ascending=False)
              .reset_index())
prov_df["% of week"] = (prov_df["weekly_cases"] / prov_df["weekly_cases"].sum() * 100).round(1)
prov_df.columns = ["Province","Weekly Cases","Cumulative 2026","Districts","% of Week"]
st.dataframe(prov_df, hide_index=True, use_container_width=True)

st.divider()

# ── Multi-week trend (if we have enough weeks) ─────────────────────────────────
weeks_available = df[df["year"]==sel_year]["week"].nunique()
if weeks_available >= 2:
    st.subheader(f"📈 Weekly Trend — {sel_year}")

    all_districts = sorted(df["district"].unique().tolist())
    default_sel   = ["Colombo","Gampaha","Kalutara","Galle","Matara","Ratnapura"]
    sel_dists     = st.multiselect("Districts", all_districts, default=default_sel)

    trend_df = (df[(df["year"]==sel_year) & (df["district"].isin(sel_dists))]
                  .sort_values("week"))

    fig_t = go.Figure()
    palette = [SL_RED, BLUE, GREEN, ORANGE, PURPLE, "#f1c40f","#1abc9c","#e91e8c"]
    for i, dist in enumerate(sel_dists):
        d = trend_df[trend_df["district"]==dist]
        if d.empty: continue
        fig_t.add_scatter(
            x=d["week"], y=d["weekly_cases"],
            mode="lines+markers+text",
            name=dist,
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=7),
            text=[str(v) for v in d["weekly_cases"]],
            textposition="top center",
            textfont=dict(size=10),
        )

    # Mark which weeks have real data
    for wk in df[df["year"]==sel_year]["week"].unique():
        fig_t.add_vline(x=wk, line_dash="dot", line_color="rgba(255,255,255,0.1)")

    fig_t.update_layout(
        height=400, xaxis_title="Epidemiological week",
        yaxis_title="Suspected cases",
        legend=dict(orientation="h", y=1.05),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", margin=dict(l=10,r=10,t=30,b=10),
    )
    st.plotly_chart(fig_t, use_container_width=True)
    st.caption(
        f"Weeks with data: {sorted(df[df['year']==sel_year]['week'].unique().tolist())}. "
        "Add more WER weeks to the CSV to fill the trend line."
    )
    st.divider()

# ── District data table ────────────────────────────────────────────────────────
st.subheader("📋 Full District Table")

display = dfw[["district","province","weekly_cases","cumulative_cases"]].copy()
if has_prev and not dfp.empty:
    display = display.merge(dfp[["district","weekly_cases"]].rename(
        columns={"weekly_cases":f"wk_{prev_r.week}"}), on="district", how="left")
    display[f"wk_{prev_r.week}"] = display[f"wk_{prev_r.week}"].fillna(0).astype(int)

display = display.sort_values("weekly_cases", ascending=False).reset_index(drop=True)
display.index += 1

rename = {"district":"District","province":"Province",
          "weekly_cases":f"Week {sel_week} Cases",
          "cumulative_cases":f"Cumulative {sel_year}"}
if has_prev and not dfp.empty:
    rename[f"wk_{prev_r.week}"] = f"Week {prev_r.week} Cases"

st.dataframe(display.rename(columns=rename), use_container_width=True)

# Download
csv_dl = dfw.to_csv(index=False).encode()
st.download_button(
    f"⬇️ Download Week {sel_week} data (CSV)",
    csv_dl, f"wer_week{sel_week}_{sel_year}.csv", "text/csv"
)

st.divider()
st.caption(
    f"Source: National Dengue Control Unit (NDCU), Ministry of Health Sri Lanka. "
    f"Week {sel_week} ({date_range}). NaDSys / Epidemiology Unit. "
    f"Data from official Weekly Epidemiological Reports — epid.gov.lk"
)
