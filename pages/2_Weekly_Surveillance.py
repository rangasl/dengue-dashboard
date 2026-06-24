"""
Weekly Dengue Surveillance — Sri Lanka
Source: National Dengue Control Unit (NDCU) Weekly Dengue Update
Data: Week 24, 2026 (08–14 June 2026)
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Weekly Dengue Surveillance — Sri Lanka",
    page_icon="🦟",
    layout="wide",
)

# Fixed footer
st.markdown("""
<style>
.fixed-footer{position:fixed;bottom:0;left:0;width:100%;background:#111;
color:#aaa;text-align:center;padding:6px;font-size:12px;z-index:9999;
pointer-events:none;user-select:none}
</style>
<div class="fixed-footer">Experiment by Ranga Dayawansha | Founder of FloodSupport.org</div>
""", unsafe_allow_html=True)

SL_RED  = "#c0392b"
BLUE    = "#2980b9"
ORANGE  = "#e67e22"
GREEN   = "#27ae60"
PURPLE  = "#8e44ad"
GREY    = "#7f8c8d"

# ── Data — NDCU Weekly Dengue Update, Week 24 2026 ────────────────────────────
REPORT_WEEK = 24
REPORT_YEAR = 2026
REPORT_DATES = "08–14 June 2026"

# Table 1: District cases
DISTRICT_DATA = [
    # district,           w23_25, w23_26, w24_25, w24_26, cum25,  cum26, province
    ("Colombo",              304,   292,   733,  1208,  6895, 10847, "Western"),
    ("Gampaha",              223,   212,   648,  1039,  3953,  7442, "Western"),
    ("Kalutara",              59,    64,   321,   424,  1065,  3164, "Western"),
    ("Kandy",                107,   151,   285,   376,  1769,  2239, "Central"),
    ("Matale",                17,    11,    45,    29,   621,   485, "Central"),
    ("Nuwara Eliya",           4,    13,    19,    23,    97,   178, "Central"),
    ("Galle",                 45,    43,   128,   275,  1132,  2597, "Southern"),
    ("Hambantota",            20,    18,    95,   102,   436,  1200, "Southern"),
    ("Matara",                57,    43,   220,   468,   856,  3061, "Southern"),
    ("Jaffna",                15,    91,    19,    24,   616,   605, "Northern"),
    ("Kilinochchi",            2,     3,     4,     0,    46,    47, "Northern"),
    ("Mannar",                 7,     0,     2,     4,    96,    47, "Northern"),
    ("Vavuniya",               3,     2,     3,     5,    27,    75, "Northern"),
    ("Mullaitivu",             0,     1,     4,     3,    10,    33, "Northern"),
    ("Batticaloa",            44,    45,    51,    49,  1468,  1057, "Eastern"),
    ("Ampara",                 4,     4,    21,    29,    82,   272, "Eastern"),
    ("Trincomalee",           32,    17,    33,    26,   767,   470, "Eastern"),
    ("Kalmunai",               6,     5,    17,    19,   278,   641, "Eastern"),
    ("Kurunegala",            63,    64,    73,   135,   700,  1095, "North Western"),
    ("Puttalam",               8,     8,    59,    83,   385,   708, "North Western"),
    ("Anuradhapura",           3,     7,    19,    24,   267,   298, "North Central"),
    ("Polonnaruwa",            5,    15,    27,    39,   187,   315, "North Central"),
    ("Badulla",               28,    20,    43,    56,   382,   505, "Uva"),
    ("Monaragala",            39,    22,    34,    31,   471,   493, "Uva"),
    ("Ratnapura",            177,   136,   258,   247,  3134,  2902, "Sabaragamuwa"),
    ("Kegalle",               42,    47,    94,   106,   747,  1145, "Sabaragamuwa"),
]
COLS = ["district","w23_25","w23_26","w24_25","w24_26","cum_25","cum_26","province"]
df_dist = pd.DataFrame(DISTRICT_DATA, columns=COLS)

# Table 2: High-risk MOH areas Week 24
MOH_DATA = [
    # MOH area, province, district, w23, w24
    ("Battaramulla",     "Western",       "Colombo",    30,  54),
    ("Boralesgamuwa",    "Western",       "Colombo",    31,  55),
    ("Dehiwala",         "Western",       "Colombo",    25,  55),
    ("Egodauyana",       "Western",       "Colombo",    23,  34),
    ("Gothatuwa",        "Western",       "Colombo",    47,  86),
    ("Hanwella",         "Western",       "Colombo",    40,  33),
    ("Homagama",         "Western",       "Colombo",    41,  99),
    ("Kaduwela",         "Western",       "Colombo",    80, 109),
    ("Kahathuduwa",      "Western",       "Colombo",    19,  35),
    ("Kesbewa",          "Western",       "Colombo",    28,  19),
    ("Kolonnawa",        "Western",       "Colombo",     5,  12),
    ("Maharagama",       "Western",       "Colombo",   121, 189),
    ("Moratuwa",         "Western",       "Colombo",    32,  48),
    ("Nugegoda",         "Western",       "Colombo",    31,  66),
    ("Padukka",          "Western",       "Colombo",    36,  28),
    ("Piliyandala",      "Western",       "Colombo",    20,  66),
    ("Pitakotte",        "Western",       "Colombo",    12,  20),
    ("Rathmalana",       "Western",       "Colombo",    24,  50),
    ("D1-CMC",           "Western",       "Colombo",     9,  20),
    ("D2A-CMC",          "Western",       "Colombo",     7,  15),
    ("D2B-CMC",          "Western",       "Colombo",    13,  11),
    ("D3-CMC",           "Western",       "Colombo",    22,  45),
    ("D4-CMC",           "Western",       "Colombo",    23,  35),
    ("D5-CMC",           "Western",       "Colombo",     9,  24),
    ("Aththanagalla",    "Western",       "Gampaha",    23,  33),
    ("Biyagama",         "Western",       "Gampaha",   132, 233),
    ("Divulapitiya",     "Western",       "Gampaha",    11,  20),
    ("Gampaha",          "Western",       "Gampaha",    27,  34),
    ("Ja-Ela",           "Western",       "Gampaha",    52,  80),
    ("Katana",           "Western",       "Gampaha",    25,  49),
    ("Kelaniya",         "Western",       "Gampaha",    84, 119),
    ("Mahara",           "Western",       "Gampaha",    62, 108),
    ("Meerigama",        "Western",       "Gampaha",    16,  31),
    ("Minuwangoda",      "Western",       "Gampaha",    47,  44),
    ("Negombo",          "Western",       "Gampaha",    19,  56),
    ("Pugoda (Dompe)",   "Western",       "Gampaha",    26,  39),
    ("Ragama",           "Western",       "Gampaha",    22,  23),
    ("Seeduwa",          "Western",       "Gampaha",    56,  96),
    ("Wattala",          "Western",       "Gampaha",    46,  75),
    ("Bandaragama",      "Western",       "Kalutara",   54,  75),
    ("Bulathsinhala",    "Western",       "Kalutara",   13,  12),
    ("Horana",           "Western",       "Kalutara",   34,  48),
    ("Ingiriya",         "Western",       "Kalutara",   30,  12),
    ("Madurawala",       "Western",       "Kalutara",   16,  25),
    ("Mathugama",        "Western",       "Kalutara",    6,  15),
    ("Millaniya",        "Western",       "Kalutara",   14,  11),
    ("Panadura",         "Western",       "Kalutara",   45, 113),
    ("Wadduwa",          "Western",       "Kalutara",   24,  32),
    ("Kalutara NIHS",    "Western",       "Kalutara",   29,  28),
    ("Payagala",         "Western",       "Kalutara",   24,  21),
    ("Akuressa",         "Southern",      "Matara",     12,  39),
    ("Athuraliya",       "Southern",      "Matara",     13,  19),
    ("Deniyaya",         "Southern",      "Matara",      9,  13),
    ("Devinuwara",       "Southern",      "Matara",     16,  25),
    ("Dickwella",        "Southern",      "Matara",      8,  14),
    ("Kamburupitiya",    "Southern",      "Matara",      2,  23),
    ("Malimbada",        "Southern",      "Matara",      6,  15),
    ("Matara MC",        "Southern",      "Matara",     85, 157),
    ("Matara PS",        "Southern",      "Matara",     14,  11),
    ("Morawaka",         "Southern",      "Matara",      7,  22),
    ("Mulatiyana",       "Southern",      "Matara",      6,  13),
    ("Pasgoda",          "Southern",      "Matara",      2,  12),
    ("Weligama",         "Southern",      "Matara",     26,  69),
    ("Welipitiya",       "Southern",      "Matara",      4,  14),
    ("Baddegama",        "Southern",      "Galle",       6,  17),
    ("Bope Poddala",     "Southern",      "Galle",      41,  72),
    ("Elpitiya",         "Southern",      "Galle",       8,  12),
    ("Habaraduwa",       "Southern",      "Galle",       4,  12),
    ("Hikkaduwa",        "Southern",      "Galle",       3,  11),
    ("Imaduwa",          "Southern",      "Galle",       4,  10),
    ("Karandeniya",      "Southern",      "Galle",       8,  13),
    ("MC-Galle",         "Southern",      "Galle",      10,  40),
    ("Thawalama",        "Southern",      "Galle",       5,  17),
    ("Ambalantota",      "Southern",      "Hambantota",  9,  10),
    ("Beliatta",         "Southern",      "Hambantota",  8,  12),
    ("Katuwana",         "Southern",      "Hambantota", 13,  15),
    ("Tangalle",         "Southern",      "Hambantota", 22,  24),
    ("Walasmulla",       "Southern",      "Hambantota",  9,  16),
    ("Gampola",          "Central",       "Kandy",      17,  39),
    ("Gangawata Korale", "Central",       "Kandy",      23,  39),
    ("Kandy MC",         "Central",       "Kandy",      52,  71),
    ("Kurunduwaththa",   "Central",       "Kandy",       5,  10),
    ("Menikhinna",       "Central",       "Kandy",      13,  14),
    ("Pasbage",          "Central",       "Kandy",      21,  15),
    ("Thalathuoya",      "Central",       "Kandy",       6,  11),
    ("Udunuwara",        "Central",       "Kandy",      19,  26),
    ("Waththegama",      "Central",       "Kandy",      11,  23),
    ("Werellagama",      "Central",       "Kandy",      16,  21),
    ("Yatinuwara",       "Central",       "Kandy",      42,  45),
    ("Ayagama",          "Sabaragamuwa",  "Ratnapura",  11,  15),
    ("Eheliyagoda",      "Sabaragamuwa",  "Ratnapura",  11,  14),
    ("Embilipitiya",     "Sabaragamuwa",  "Ratnapura",  28,  18),
    ("Kahawatta",        "Sabaragamuwa",  "Ratnapura",  12,  10),
    ("Kalawana",         "Sabaragamuwa",  "Ratnapura",   8,  20),
    ("Kuruvita",         "Sabaragamuwa",  "Ratnapura",  25,  20),
    ("Nivithigala",      "Sabaragamuwa",  "Ratnapura",  21,  26),
    ("Pelmadulla",       "Sabaragamuwa",  "Ratnapura",  32,  25),
    ("Rathnapura MC",    "Sabaragamuwa",  "Ratnapura",  21,  25),
    ("Ratnapura PS",     "Sabaragamuwa",  "Ratnapura",  30,  15),
    ("Dehiovita",        "Sabaragamuwa",  "Kegalle",    29,  18),
    ("Mawanella",        "Sabaragamuwa",  "Kegalle",    14,  14),
    ("Ruwanwella",       "Sabaragamuwa",  "Kegalle",     8,  14),
    ("Warakapola",       "Sabaragamuwa",  "Kegalle",    14,  10),
    ("Yatiyanthota",     "Sabaragamuwa",  "Kegalle",     9,  14),
    ("Batticaloa MOH",   "Eastern",       "Batticaloa", 24,  14),
    ("Thamankaduwa",     "North Central", "Polonnaruwa", 7,  12),
    ("Kuliyapitiya",     "North Western", "Kurunegala",  2,  14),
    ("Mallawapitiya",    "North Western", "Kurunegala", 10,  13),
    ("Chilaw",           "North Western", "Puttalam",    9,  31),
    ("Madampe",          "North Western", "Puttalam",    5,  10),
    ("Wennappuwa",       "North Western", "Puttalam",   18,  13),
    ("Badulla MOH",      "Uva",           "Badulla",    11,  26),
]
df_moh = pd.DataFrame(MOH_DATA, columns=["moh_area","province","district","w23","w24"])

# Table 3: Sentinel hospitals
HOSPITAL_DATA = [
    ("TH-Matara",            133, 166),
    ("NIID",                 133, 155),
    ("TH-Ratnapura",          51, 122),
    ("DGH-Negombo",           50, 106),
    ("TH-Colombo South",      70, 104),
    ("NHSL",                  60,  93),
    ("TH-Colombo North",      68,  84),
    ("NH-Galle",              48,  65),
    ("NH-Kandy",              40,  61),
    ("TH-Peradeniya",         42,  54),
    ("BH-Panadura",           27,  48),
    ("LRH",                   34,  43),
    ("TH-Kalutara",           22,  43),
    ("TH-Kurunegala",         30,  40),
    ("DGH-Horana",            38,  40),
    ("DGH-Gampaha",           28,  37),
    ("BH-Tangalle",           22,  32),
    ("DGH-Avissawella",       27,  32),
    ("BH-Kamburupitiya",      18,  24),
    ("PGH-Badulla",           15,  22),
    ("TH-Kuliyapitiya",        7,  20),
    ("DGH-Chilaw",             6,  18),
    ("BH-Gampola",             8,  15),
    ("BH-Balangoda",          10,  15),
    ("BH-Kahawatta",          16,  14),
    ("DGH-Nawalapitiya",      10,  13),
    ("BH-Minuwangoda",        10,  13),
    ("DGH-Polonnaruwa",        7,  13),
    ("DGH-Embilipitiya",       7,  12),
    ("BH-Wathupitiwala",      12,  12),
    ("DGH-Hambantota",         8,  11),
    ("BH-Warakapola",          7,  10),
    ("DGH-Ampara",             5,   9),
    ("BH-Balapitiya",          7,   9),
    ("TH-Batticaloa",         10,   8),
    ("DGH-Kegalle",            7,   7),
    ("DGH-Matale",             6,   5),
    ("DGH-Moneragala",         3,   5),
    ("BH-Mawanella",           7,   5),
    ("BH-Mirigama",            3,   4),
    ("BH-Dambulla",            2,   4),
]
df_hosp = pd.DataFrame(HOSPITAL_DATA, columns=["hospital","w23_midnight","w24_midnight"])

# Table 4: Deaths
DEATH_BY_DISTRICT = [
    ("Colombo", 7), ("Gampaha", 5), ("NIHS Kalutara", 1),
    ("Hambantota", 2), ("Galle", 2), ("Kandy", 1),
    ("Ratnapura", 2), ("Kalutara", 1), ("Trincomalee", 1),
    ("Matara", 1), ("Puttalam", 1),
]
DEATH_BY_AGE = [
    ("0-14",  2, 3),
    ("15-24", 0, 1),
    ("25-34", 1, 1),
    ("35-44", 0, 4),
    ("45-54", 1, 2),
    ("55-64", 1, 1),
    (">64",   1, 6),
]

# ── Page layout ────────────────────────────────────────────────────────────────
st.title("🦟 Weekly Dengue Surveillance — Sri Lanka")
st.markdown(
    f"**Week {REPORT_WEEK} ({REPORT_DATES})** · "
    f"Source: [National Dengue Control Unit (NDCU)](https://www.epid.gov.lk/) · "
    f"NaDSys / Epidemiology Unit"
)

# ── Hero metrics ───────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Week 24 Cases",     "4,824",   "+48.2% vs Week 23")
m2.metric("Week 23 Cases",     "3,255",   "+4.4 incr vs Wk 22")
m3.metric("Cumulative 2026",  "41,921",   "+58.3% vs 2025 same pt")
m4.metric("Cumulative 2025",  "26,487",   "up to Week 24")
m5.metric("Deaths 2026",         "24",    "CFR 0.06%")

st.divider()

# ── Section 1: District breakdown ─────────────────────────────────────────────
st.subheader("📍 District-wise Cases — Week 24")

view = st.radio("Compare", ["Week 24 (2025 vs 2026)", "Cumulative up to Week 24"], horizontal=True)

df_plot = df_dist[df_dist["district"] != "Total"].copy()

if view == "Week 24 (2025 vs 2026)":
    df_plot = df_plot.sort_values("w24_26", ascending=True)
    fig = go.Figure()
    fig.add_bar(y=df_plot["district"], x=df_plot["w24_25"],
                name="Week 24 · 2025", orientation="h",
                marker_color=BLUE, opacity=0.7)
    fig.add_bar(y=df_plot["district"], x=df_plot["w24_26"],
                name="Week 24 · 2026", orientation="h",
                marker_color=SL_RED, opacity=0.9)
    fig.update_layout(barmode="overlay", height=600,
                      xaxis_title="Suspected dengue cases",
                      legend=dict(orientation="h", y=1.02))
else:
    df_plot = df_plot.sort_values("cum_26", ascending=True)
    fig = go.Figure()
    fig.add_bar(y=df_plot["district"], x=df_plot["cum_25"],
                name="2025 cumulative", orientation="h",
                marker_color=BLUE, opacity=0.7)
    fig.add_bar(y=df_plot["district"], x=df_plot["cum_26"],
                name="2026 cumulative", orientation="h",
                marker_color=SL_RED, opacity=0.9)
    fig.update_layout(barmode="overlay", height=600,
                      xaxis_title="Cumulative cases up to Week 24",
                      legend=dict(orientation="h", y=1.02))

fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#ccc",
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# Province summary
st.markdown("**Province Summary — Week 24, 2026**")
prov = (df_dist[df_dist["district"] != "Total"]
        .groupby("province")[["w24_26","cum_26"]]
        .sum()
        .sort_values("w24_26", ascending=False)
        .reset_index())
prov["pct_w24"] = (prov["w24_26"] / prov["w24_26"].sum() * 100).round(1)
prov.columns = ["Province", "Week 24 Cases", "Cumulative 2026", "% of Week 24"]
st.dataframe(prov, hide_index=True, use_container_width=True)

st.divider()

# ── Section 2: Week-on-week change ────────────────────────────────────────────
st.subheader("📈 Week-on-Week Change by District (2026)")

df_wow = df_dist[df_dist["district"] != "Total"].copy()
df_wow["change"] = df_wow["w24_26"] - df_wow["w23_26"]
df_wow["pct_change"] = ((df_wow["w24_26"] - df_wow["w23_26"])
                        / df_wow["w23_26"].replace(0, 1) * 100).round(1)
df_wow = df_wow.sort_values("change", ascending=True)

colors = [SL_RED if x > 0 else GREEN for x in df_wow["change"]]
fig2 = go.Figure(go.Bar(
    y=df_wow["district"],
    x=df_wow["change"],
    orientation="h",
    marker_color=colors,
    text=[f"{v:+d}" for v in df_wow["change"]],
    textposition="outside",
))
fig2.update_layout(
    height=550,
    xaxis_title="Change in cases (Week 24 minus Week 23)",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#ccc",
    margin=dict(l=10, r=60, t=10, b=10),
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Section 3: High-risk MOH areas ────────────────────────────────────────────
st.subheader(f"🔴 High-Risk MOH Areas — Week {REPORT_WEEK} (112 areas flagged)")
st.caption("32 newly flagged areas this week. 9 areas de-listed from Week 23.")

provinces = ["All"] + sorted(df_moh["province"].unique().tolist())
sel_prov = st.selectbox("Filter by Province", provinces)

df_moh_show = df_moh if sel_prov == "All" else df_moh[df_moh["province"] == sel_prov]
df_moh_show = df_moh_show.sort_values("w24", ascending=False).reset_index(drop=True)

# Colour-code rows where w24 > w23 (rising)
def highlight_rising(row):
    color = "background-color:#3d0000;color:#ffaaaa" if row["w24"] > row["w23"] else ""
    return [color] * len(row)

st.dataframe(
    df_moh_show.rename(columns={
        "moh_area":"MOH Area","province":"Province","district":"District",
        "w23":"Week 23","w24":"Week 24"
    }),
    hide_index=True,
    use_container_width=True,
    height=400,
)

# Top 10 hotspots bar
st.markdown("**Top 10 hotspot MOH areas — Week 24**")
top10 = df_moh.sort_values("w24", ascending=True).tail(10)
fig3 = go.Figure(go.Bar(
    y=top10["moh_area"],
    x=top10["w24"],
    orientation="h",
    marker_color=SL_RED,
    text=top10["w24"],
    textposition="outside",
))
fig3.update_layout(
    height=350,
    xaxis_title="Cases in Week 24",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#ccc",
    margin=dict(l=10, r=40, t=10, b=10),
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Section 4: Hospital sentinel data ─────────────────────────────────────────
st.subheader("🏥 Sentinel Hospital Midnight Totals")
st.caption(
    f"Average midnight total Week 24: **1,590 patients** across 74 sentinel hospitals. "
    "40 hospitals reported an increase vs Week 23."
)

h1, h2 = st.columns([2, 1])
with h1:
    df_hosp_plot = df_hosp.sort_values("w24_midnight", ascending=True)
    fig4 = go.Figure()
    fig4.add_bar(y=df_hosp_plot["hospital"], x=df_hosp_plot["w23_midnight"],
                 name="Week 23", orientation="h", marker_color=BLUE, opacity=0.7)
    fig4.add_bar(y=df_hosp_plot["hospital"], x=df_hosp_plot["w24_midnight"],
                 name="Week 24", orientation="h", marker_color=SL_RED, opacity=0.9)
    fig4.update_layout(
        barmode="overlay", height=700,
        xaxis_title="Average midnight dengue patient count",
        legend=dict(orientation="h", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig4, use_container_width=True)

with h2:
    st.markdown("**Hospital capacity change**")
    df_hosp["change"] = df_hosp["w24_midnight"] - df_hosp["w23_midnight"]
    df_hosp["trend"] = df_hosp["change"].apply(
        lambda x: "🔴 Rising" if x > 0 else ("🟢 Falling" if x < 0 else "➡️ Same")
    )
    st.dataframe(
        df_hosp[["hospital","w23_midnight","w24_midnight","trend"]]
        .sort_values("w24_midnight", ascending=False)
        .rename(columns={
            "hospital":"Hospital",
            "w23_midnight":"Wk 23",
            "w24_midnight":"Wk 24",
            "trend":"Trend",
        }),
        hide_index=True,
        height=700,
    )

st.divider()

# ── Section 5: Deaths ──────────────────────────────────────────────────────────
st.subheader("💔 Dengue Deaths — 2026")

d1, d2 = st.columns(2)

with d1:
    st.markdown("**Deaths by District**")
    df_death_dist = pd.DataFrame(DEATH_BY_DISTRICT, columns=["District","Deaths"])
    df_death_dist = df_death_dist.sort_values("Deaths", ascending=True)
    fig5 = go.Figure(go.Bar(
        y=df_death_dist["District"],
        x=df_death_dist["Deaths"],
        orientation="h",
        marker_color=PURPLE,
        text=df_death_dist["Deaths"],
        textposition="outside",
    ))
    fig5.update_layout(
        height=350,
        xaxis_title="Deaths in 2026",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        margin=dict(l=10, r=40, t=10, b=10),
    )
    st.plotly_chart(fig5, use_container_width=True)

with d2:
    st.markdown("**Deaths by Age & Sex**")
    df_age = pd.DataFrame(DEATH_BY_AGE, columns=["Age Group","Male","Female"])
    fig6 = go.Figure()
    fig6.add_bar(x=df_age["Age Group"], y=df_age["Male"],
                 name="Male", marker_color=BLUE)
    fig6.add_bar(x=df_age["Age Group"], y=df_age["Female"],
                 name="Female", marker_color="#e91e8c")
    fig6.update_layout(
        barmode="group", height=350,
        yaxis_title="Deaths",
        legend=dict(orientation="h", y=1.05),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("""
| | 2025 | 2026 |
|---|---|---|
| Reported deaths | 29 | 24 |
| Case fatality rate | 0.05% | 0.06% |

⚠️ Deaths skewed female (18F vs 6M in 2026) and towards age >64.
    """)

st.divider()
st.caption(
    "Source: National Dengue Control Unit (NDCU), Ministry of Health Sri Lanka — "
    "Weekly Dengue Update Week 24, 2026 (08–14 June 2026). "
    "NaDSys (National Dengue Surveillance System). "
    "Data reflects reports as of bulletin date; week totals may vary due to duplicate deletion."
)
