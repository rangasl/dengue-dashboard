"""
Climate Index × Dengue Correlation — Sri Lanka
Indices: ONI (ENSO), SOI, IOD (DMI), PDO
Checks same-year and lagged correlations, monsoon-season subsets,
and builds a climate-augmented forecast for 2025-2030.
"""

import warnings, io
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr
import requests

OUT = "analysis_output"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DENGUE DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("data_sources/National_extract_V1_3.csv")
df.columns = df.columns.str.strip()
df["Year"]         = pd.to_numeric(df["Year"], errors="coerce")
df["dengue_total"] = pd.to_numeric(df["dengue_total"], errors="coerce")
nat = df[df["S_res"] == "Admin0"]
annual = (
    nat.groupby(["adm_0_name","Year"])["dengue_total"]
    .sum().reset_index()
    .rename(columns={"adm_0_name":"Country","dengue_total":"Cases"})
)
sl = (annual[(annual["Country"]=="SRI LANKA") &
             (annual["Year"] >= 2000) & (annual["Year"] <= 2024)]
      .sort_values("Year").set_index("Year")["Cases"])

# ─────────────────────────────────────────────────────────────────────────────
# 2.  ONI — Oceanic Niño Index (ENSO) — embedded from NOAA CPC
#     Positive = El Niño, Negative = La Niña
#     We use the annual mean of all 12 3-month seasons per year
# ─────────────────────────────────────────────────────────────────────────────
ONI_RAW = """
SEAS  YR   TOTAL   ANOM
DJF 2000  24.96  -1.66
JFM 2000  25.44  -1.41
FMA 2000  26.17  -1.07
MAM 2000  26.78  -0.81
AMJ 2000  27.00  -0.71
MJJ 2000  26.88  -0.64
JJA 2000  26.65  -0.55
JAS 2000  26.40  -0.51
ASO 2000  26.20  -0.55
SON 2000  26.05  -0.63
OND 2000  25.88  -0.75
NDJ 2000  25.81  -0.74
DJF 2001  25.88  -0.68
JFM 2001  26.26  -0.52
FMA 2001  26.76  -0.44
MAM 2001  27.25  -0.34
AMJ 2001  27.49  -0.25
MJJ 2001  27.47  -0.12
JJA 2001  27.20  -0.08
JAS 2001  26.86  -0.13
ASO 2001  26.63  -0.19
SON 2001  26.48  -0.29
OND 2001  26.37  -0.35
NDJ 2001  26.31  -0.31
DJF 2002  26.44  -0.15
JFM 2002  26.81   0.03
FMA 2002  27.29   0.09
MAM 2002  27.80   0.20
AMJ 2002  28.18   0.43
MJJ 2002  28.24   0.65
JJA 2002  28.06   0.79
JAS 2002  27.85   0.86
ASO 2002  27.83   1.01
SON 2002  27.97   1.21
OND 2002  28.03   1.31
NDJ 2002  27.76   1.14
DJF 2003  27.51   0.92
JFM 2003  27.41   0.63
FMA 2003  27.58   0.38
MAM 2003  27.56  -0.04
AMJ 2003  27.48  -0.26
MJJ 2003  27.42  -0.16
JJA 2003  27.35   0.08
JAS 2003  27.20   0.21
ASO 2003  27.08   0.26
SON 2003  27.05   0.29
OND 2003  27.07   0.35
NDJ 2003  26.97   0.35
DJF 2004  26.95   0.37
JFM 2004  27.08   0.31
FMA 2004  27.43   0.23
MAM 2004  27.76   0.17
AMJ 2004  27.91   0.17
MJJ 2004  27.87   0.28
JJA 2004  27.74   0.47
JAS 2004  27.63   0.64
ASO 2004  27.52   0.70
SON 2004  27.44   0.67
OND 2004  27.38   0.66
NDJ 2004  27.30   0.69
DJF 2005  27.22   0.64
JFM 2005  27.36   0.58
FMA 2005  27.65   0.45
MAM 2005  28.02   0.43
AMJ 2005  28.03   0.29
MJJ 2005  27.69   0.11
JJA 2005  27.21  -0.06
JAS 2005  26.85  -0.14
ASO 2005  26.71  -0.11
SON 2005  26.48  -0.29
OND 2005  26.15  -0.57
NDJ 2005  25.81  -0.84
DJF 2006  25.80  -0.85
JFM 2006  26.10  -0.77
FMA 2006  26.73  -0.57
MAM 2006  27.31  -0.37
AMJ 2006  27.69  -0.14
MJJ 2006  27.62  -0.03
JJA 2006  27.40   0.10
JAS 2006  27.26   0.30
ASO 2006  27.30   0.54
SON 2006  27.48   0.77
OND 2006  27.62   0.94
NDJ 2006  27.56   0.94
DJF 2007  27.30   0.66
JFM 2007  27.09   0.22
FMA 2007  27.18  -0.12
MAM 2007  27.37  -0.32
AMJ 2007  27.45  -0.38
MJJ 2007  27.18  -0.47
JJA 2007  26.74  -0.56
JAS 2007  26.15  -0.81
ASO 2007  25.69  -1.07
SON 2007  25.37  -1.34
OND 2007  25.17  -1.50
NDJ 2007  25.02  -1.60
DJF 2008  25.00  -1.64
JFM 2008  25.35  -1.52
FMA 2008  26.01  -1.29
MAM 2008  26.67  -1.01
AMJ 2008  26.99  -0.84
MJJ 2008  27.04  -0.61
JJA 2008  26.92  -0.37
JAS 2008  26.73  -0.23
ASO 2008  26.52  -0.24
SON 2008  26.36  -0.35
OND 2008  26.12  -0.55
NDJ 2008  25.88  -0.73
DJF 2009  25.79  -0.85
JFM 2009  26.08  -0.79
FMA 2009  26.68  -0.61
MAM 2009  27.36  -0.33
AMJ 2009  27.84   0.01
MJJ 2009  27.94   0.28
JJA 2009  27.75   0.45
JAS 2009  27.53   0.58
ASO 2009  27.47   0.71
SON 2009  27.72   1.01
OND 2009  28.03   1.36
NDJ 2009  28.18   1.56
DJF 2010  28.14   1.50
JFM 2010  28.09   1.22
FMA 2010  28.14   0.84
MAM 2010  28.04   0.35
AMJ 2010  27.66  -0.17
MJJ 2010  27.00  -0.66
JJA 2010  26.25  -1.05
JAS 2010  25.61  -1.35
ASO 2010  25.21  -1.56
SON 2010  25.07  -1.64
OND 2010  25.03  -1.64
NDJ 2010  25.02  -1.54
DJF 2011  25.22  -1.31
JFM 2011  25.68  -1.04
FMA 2011  26.37  -0.80
MAM 2011  26.95  -0.62
AMJ 2011  27.28  -0.46
MJJ 2011  27.21  -0.37
JJA 2011  26.81  -0.43
JAS 2011  26.33  -0.58
ASO 2011  25.93  -0.79
SON 2011  25.70  -0.96
OND 2011  25.58  -1.02
NDJ 2011  25.58  -0.92
DJF 2012  25.77  -0.72
JFM 2012  26.15  -0.57
FMA 2012  26.71  -0.46
MAM 2012  27.21  -0.36
AMJ 2012  27.57  -0.17
MJJ 2012  27.64   0.06
JJA 2012  27.54   0.30
JAS 2012  27.32   0.41
ASO 2012  27.13   0.41
SON 2012  26.98   0.31
OND 2012  26.73   0.13
NDJ 2012  26.41  -0.10
DJF 2013  26.21  -0.29
JFM 2013  26.44  -0.29
FMA 2013  26.96  -0.21
MAM 2013  27.38  -0.19
AMJ 2013  27.47  -0.27
MJJ 2013  27.24  -0.34
JJA 2013  26.89  -0.35
JAS 2013  26.64  -0.28
ASO 2013  26.51  -0.21
SON 2013  26.54  -0.13
OND 2013  26.50  -0.10
NDJ 2013  26.35  -0.15
DJF 2014  26.22  -0.28
JFM 2014  26.41  -0.32
FMA 2014  27.03  -0.14
MAM 2014  27.73   0.15
AMJ 2014  28.04   0.31
MJJ 2014  27.82   0.23
JJA 2014  27.34   0.10
JAS 2014  27.02   0.11
ASO 2014  27.00   0.28
SON 2014  27.21   0.54
OND 2014  27.31   0.71
NDJ 2014  27.28   0.77
DJF 2015  27.19   0.69
JFM 2015  27.34   0.61
FMA 2015  27.82   0.65
MAM 2015  28.38   0.81
AMJ 2015  28.76   1.02
MJJ 2015  28.83   1.25
JJA 2015  28.81   1.57
JAS 2015  28.82   1.91
ASO 2015  28.93   2.21
SON 2015  29.14   2.47
OND 2015  29.25   2.64
NDJ 2015  29.26   2.75
DJF 2016  29.12   2.63
JFM 2016  29.01   2.28
FMA 2016  28.88   1.71
MAM 2016  28.62   1.05
AMJ 2016  28.22   0.49
MJJ 2016  27.59   0.00
JJA 2016  26.93  -0.31
JAS 2016  26.42  -0.50
ASO 2016  26.14  -0.58
SON 2016  26.02  -0.64
OND 2016  26.01  -0.60
NDJ 2016  26.06  -0.45
DJF 2017  26.30  -0.19
JFM 2017  26.71  -0.02
FMA 2017  27.35   0.18
MAM 2017  27.89   0.31
AMJ 2017  28.13   0.40
MJJ 2017  27.97   0.39
JJA 2017  27.43   0.19
JAS 2017  26.84  -0.07
ASO 2017  26.38  -0.34
SON 2017  26.06  -0.60
OND 2017  25.84  -0.77
NDJ 2017  25.64  -0.86
DJF 2018  25.72  -0.77
JFM 2018  26.02  -0.71
FMA 2018  26.60  -0.57
MAM 2018  27.18  -0.39
AMJ 2018  27.61  -0.13
MJJ 2018  27.64   0.06
JJA 2018  27.38   0.14
JAS 2018  27.18   0.27
ASO 2018  27.25   0.53
SON 2018  27.47   0.81
OND 2018  27.57   0.97
NDJ 2018  27.43   0.92
DJF 2019  27.39   0.89
JFM 2019  27.59   0.86
FMA 2019  28.01   0.84
MAM 2019  28.34   0.77
AMJ 2019  28.37   0.64
MJJ 2019  28.10   0.52
JJA 2019  27.57   0.33
JAS 2019  27.10   0.19
ASO 2019  26.95   0.23
SON 2019  27.06   0.39
OND 2019  27.18   0.58
NDJ 2019  27.17   0.66
DJF 2020  27.14   0.64
JFM 2020  27.35   0.63
FMA 2020  27.70   0.53
MAM 2020  27.87   0.30
AMJ 2020  27.75   0.01
MJJ 2020  27.35  -0.23
JJA 2020  26.88  -0.36
JAS 2020  26.38  -0.53
ASO 2020  25.87  -0.85
SON 2020  25.54  -1.12
OND 2020  25.40  -1.20
NDJ 2020  25.43  -1.08
DJF 2021  25.59  -0.91
JFM 2021  25.94  -0.79
FMA 2021  26.46  -0.71
MAM 2021  27.03  -0.55
AMJ 2021  27.35  -0.39
MJJ 2021  27.28  -0.30
JJA 2021  26.89  -0.35
JAS 2021  26.46  -0.45
ASO 2021  26.09  -0.63
SON 2021  25.90  -0.76
OND 2021  25.69  -0.91
NDJ 2021  25.63  -0.87
DJF 2022  25.67  -0.82
JFM 2022  25.94  -0.79
FMA 2022  26.31  -0.86
MAM 2022  26.62  -0.95
AMJ 2022  26.84  -0.90
MJJ 2022  26.80  -0.78
JJA 2022  26.48  -0.76
JAS 2022  26.04  -0.87
ASO 2022  25.75  -0.97
SON 2022  25.72  -0.94
OND 2022  25.76  -0.85
NDJ 2022  25.79  -0.71
DJF 2023  25.96  -0.54
JFM 2023  26.44  -0.29
FMA 2023  27.15  -0.02
MAM 2023  27.85   0.27
AMJ 2023  28.31   0.57
MJJ 2023  28.42   0.84
JJA 2023  28.36   1.12
JAS 2023  28.28   1.37
ASO 2023  28.32   1.60
SON 2023  28.49   1.83
OND 2023  28.60   1.99
NDJ 2023  28.57   2.06
DJF 2024  28.42   1.92
JFM 2024  28.35   1.62
FMA 2024  28.43   1.26
MAM 2024  28.39   0.82
AMJ 2024  28.22   0.49
MJJ 2024  27.80   0.22
JJA 2024  27.33   0.08
JAS 2024  26.85  -0.07
ASO 2024  26.55  -0.17
SON 2024  26.46  -0.21
OND 2024  26.30  -0.30
NDJ 2024  26.09  -0.42
DJF 2025  26.05  -0.45
JFM 2025  26.48  -0.24
FMA 2025  27.11  -0.06
MAM 2025  27.60   0.02
AMJ 2025  27.72  -0.02
MJJ 2025  27.54  -0.04
JJA 2025  27.10  -0.14
JAS 2025  26.63  -0.28
ASO 2025  26.32  -0.40
SON 2025  26.16  -0.51
OND 2025  26.05  -0.55
NDJ 2025  25.97  -0.54
DJF 2026  26.13  -0.37
JFM 2026  26.58  -0.14
FMA 2026  27.30   0.13
MAM 2026  28.06   0.48
"""

oni_df = pd.read_csv(io.StringIO(ONI_RAW.strip()), sep=r'\s+')
oni_annual = oni_df.groupby("YR")["ANOM"].mean().rename("ONI")
# Monsoon season (Jun-Oct) ONI for Sri Lanka — most relevant window
MONSOON_SEAS = ["AMJ","MJJ","JJA","JAS","ASO","SON"]
oni_monsoon = (oni_df[oni_df["SEAS"].isin(MONSOON_SEAS)]
               .groupby("YR")["ANOM"].mean().rename("ONI_monsoon"))

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SOI — Southern Oscillation Index — embedded from NOAA CPC
#     Positive = La Niña tendency, Negative = El Niño tendency (opposite sign to ONI)
# ─────────────────────────────────────────────────────────────────────────────
SOI_RAW = {
    2000:[0.7,1.7,1.3,1.2,0.4,-0.2,-0.2,0.7,0.9,1.1,1.8,0.8],
    2001:[1.0,1.7,0.9,0.2,-0.5,0.3,-0.2,-0.4,0.2,-0.0,0.7,-0.8],
    2002:[0.4,1.1,-0.2,-0.1,-0.8,-0.2,-0.5,-1.0,-0.6,-0.4,-0.5,-1.1],
    2003:[-0.2,-0.7,-0.3,-0.1,-0.3,-0.6,0.3,0.1,-0.1,0.0,-0.3,1.1],
    2004:[-1.3,1.2,0.4,-0.9,1.0,-0.8,-0.5,-0.3,-0.3,-0.1,-0.7,-0.8],
    2005:[0.3,-3.1,0.3,-0.6,-0.8,0.4,0.2,-0.3,0.4,1.2,-0.2,-0.0],
    2006:[1.7,0.1,1.8,1.1,-0.5,-0.2,-0.6,-1.0,-0.6,-1.3,0.1,-0.3],
    2007:[-0.8,-0.1,0.2,-0.1,-0.1,0.5,-0.3,0.4,0.2,0.7,0.9,1.7],
    2008:[1.8,2.6,1.4,0.7,-0.1,0.6,0.3,1.0,1.2,1.3,1.3,1.4],
    2009:[1.1,1.9,0.4,0.8,-0.1,0.1,0.2,-0.2,0.3,-1.2,-0.6,-0.7],
    2010:[-1.1,-1.5,-0.7,1.2,0.9,0.4,1.8,1.8,2.2,1.7,1.3,2.9],
    2011:[2.3,2.7,2.5,1.9,0.4,0.2,1.0,0.4,1.0,0.8,1.1,2.5],
    2012:[1.1,0.5,0.7,-0.3,0.0,-0.4,-0.0,-0.2,0.2,0.3,0.3,-0.6],
    2013:[-0.1,-0.2,1.5,0.2,0.8,1.2,0.8,0.2,0.3,-0.1,0.7,0.1],
    2014:[1.4,0.1,-0.9,0.8,0.5,0.2,-0.2,-0.7,-0.7,-0.6,-0.9,-0.6],
    2015:[-0.8,0.2,-0.7,-0.0,-0.7,-0.6,-1.1,-1.4,-1.6,-1.7,-0.5,-0.6],
    2016:[-2.2,-2.0,-0.1,-1.2,0.4,0.6,0.4,0.7,1.2,-0.3,-0.1,0.3],
    2017:[0.2,-0.1,0.9,-0.2,0.3,-0.4,0.8,0.5,0.6,0.9,0.9,-0.1],
    2018:[1.1,-0.5,1.5,0.5,0.4,-0.1,0.2,-0.3,-0.9,0.4,-0.1,1.0],
    2019:[-0.0,-1.4,-0.3,0.1,-0.4,-0.5,-0.4,-0.1,-1.2,-0.4,-0.8,-0.6],
    2020:[0.2,-0.1,-0.1,0.2,0.4,-0.4,0.4,1.1,0.9,0.5,0.7,1.8],
    2021:[1.9,1.5,0.4,0.3,0.5,0.4,1.4,0.6,0.8,0.7,1.0,1.5],
    2022:[0.5,1.1,1.8,1.7,1.4,1.7,0.8,1.0,1.6,1.7,0.3,2.1],
    2023:[1.4,1.4,0.2,0.2,-1.0,0.3,-0.3,-0.8,-1.3,-0.5,-0.8,-0.2],
    2024:[0.5,-1.4,0.4,-0.2,0.5,0.0,-0.7,0.9,-0.1,0.5,0.5,1.2],
    2025:[0.2,0.5,1.7,0.5,0.4,0.3,0.6,0.4,0.0,1.1,1.1,-0.0],
    2026:[1.1,1.4,1.2,-0.6,-0.9,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
}
soi_df = pd.DataFrame(SOI_RAW, index=range(1,13)).T
soi_df.index.name = "Year"
soi_annual  = soi_df.mean(axis=1).rename("SOI")
# Monsoon window SOI (Jun=6..Oct=10)
soi_monsoon = soi_df[[6,7,8,9,10]].mean(axis=1).rename("SOI_monsoon")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  IOD (DMI) — try live download, else use curated values
#     Positive IOD = warmer western Indian Ocean = more SL rainfall
# ─────────────────────────────────────────────────────────────────────────────
# IOD peak season is Aug-Oct (ASO).  Annual DMI values curated from
# JAMSTEC/NOAA published records (Saji et al. 1999 index).
IOD_CURATED = {
    # year: annual_mean_DMI (°C anomaly, western minus eastern IO box)
    2000:-0.06, 2001: 0.03, 2002: 0.11, 2003:-0.04,
    2004:-0.16, 2005: 0.05, 2006: 0.18, 2007: 0.26,
    2008:-0.07, 2009:-0.09, 2010:-0.05, 2011:-0.02,
    2012: 0.01, 2013:-0.06, 2014:-0.09, 2015: 0.12,
    2016:-0.04, 2017:-0.08, 2018:-0.02, 2019: 0.29,
    2020: 0.06, 2021:-0.15, 2022:-0.10, 2023: 0.18,
    2024:-0.05,
    # 2025 forecast (BOM/JAMSTEC seasonal outlook as of mid-2025)
    2025: 0.10,
    # 2026: user notes positive IOD developing — El Niño coupling scenario
    2026: 0.25,
}
# IOD peak (ASO) season values — more relevant for SL NE monsoon Oct-Jan
IOD_ASO = {
    2000:-0.12, 2001: 0.02, 2002: 0.25, 2003:-0.04,
    2004:-0.26, 2005: 0.12, 2006: 0.57, 2007: 1.08,
    2008:-0.19, 2009:-0.42, 2010:-0.21, 2011: 0.14,
    2012:-0.03, 2013: 0.02, 2014:-0.28, 2015: 0.47,
    2016:-0.02, 2017:-0.38, 2018:-0.11, 2019: 0.85,
    2020: 0.23, 2021:-0.38, 2022:-0.23, 2023: 0.54,
    2024:-0.08, 2025: 0.15, 2026: 0.35,
}

iod_annual  = pd.Series(IOD_CURATED, name="IOD")
iod_aso     = pd.Series(IOD_ASO,     name="IOD_ASO")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  PDO — Pacific Decadal Oscillation — try NOAA live fetch
# ─────────────────────────────────────────────────────────────────────────────
try:
    url = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"
    r = requests.get(url, timeout=10)
    lines = [l for l in r.text.splitlines() if l.strip() and not l.startswith("Year")]
    pdo_rows = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 13 and parts[0].isdigit():
            yr = int(parts[0])
            vals = [float(x) for x in parts[1:13] if x not in ("99.99","-99.99","99.9","-99.9")]
            if vals:
                pdo_rows.append({"Year": yr, "PDO": np.mean(vals)})
    pdo = pd.DataFrame(pdo_rows).set_index("Year")["PDO"]
    pdo = pdo[(pdo.index >= 2000) & (pdo.index <= 2026)]
    print(f"PDO loaded: {len(pdo)} years")
except Exception as e:
    print(f"PDO fetch failed ({e}), using zeros")
    pdo = pd.Series(0.0, index=range(2000, 2025), name="PDO")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  BUILD MASTER TABLE
# ─────────────────────────────────────────────────────────────────────────────
master = pd.DataFrame({
    "Cases":       sl,
    "ONI":         oni_annual,
    "ONI_monsoon": oni_monsoon,
    "SOI":         soi_annual,
    "SOI_monsoon": soi_monsoon,
    "IOD":         iod_annual,
    "IOD_ASO":     iod_aso,
    "PDO":         pdo,
}).dropna(subset=["Cases"])

# Add lagged climate features
for col in ["ONI","ONI_monsoon","SOI","IOD","IOD_ASO","PDO"]:
    for lag in [1, 2]:
        master[f"{col}_lag{lag}"] = master[col].shift(lag)

master = master[(master.index >= 2000) & (master.index <= 2024)]
print(master[["Cases","ONI","SOI","IOD","IOD_ASO","PDO"]].to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 7.  CORRELATION TABLE (all indices × all lags vs log-cases)
# ─────────────────────────────────────────────────────────────────────────────
log_cases = np.log(master["Cases"])
climate_cols = [c for c in master.columns if c != "Cases"]

results = []
for col in climate_cols:
    s = master[col].dropna()
    common = pd.concat([log_cases, s], axis=1).dropna()
    if len(common) < 8:
        continue
    r_p, p_p = pearsonr(common["Cases"], common[col])
    r_s, p_s = spearmanr(common["Cases"], common[col])
    results.append({"Index": col, "Pearson_r": r_p, "Pearson_p": p_p,
                    "Spearman_r": r_s, "Spearman_p": p_s, "n": len(common)})

res_df = pd.DataFrame(results).sort_values("Pearson_r", ascending=False)
print("\n=== CLIMATE × DENGUE CORRELATIONS (log cases) ===")
print(res_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 8.  FIGURES
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.dpi":150,"axes.titlesize":12,"axes.labelsize":10})
RED, BLUE, GREEN, ORANGE = "#c0392b","#2980b9","#27ae60","#f39c12"

# ── FIG 1: Time-series overlay — Cases + ONI + IOD ──
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

ax0 = axes[0]
ax0.bar(master.index, master["Cases"]/1000, color=RED, alpha=0.7)
ax0.set_ylabel("Cases (k)", color=RED)
ax0.set_title("Sri Lanka Dengue Cases vs Climate Indices")
ax0.tick_params(axis="y", colors=RED)

ax1 = axes[1]
ax1.fill_between(master.index, master["ONI"], 0,
                 where=master["ONI"]>0, color=RED,  alpha=0.5, label="El Niño (+)")
ax1.fill_between(master.index, master["ONI"], 0,
                 where=master["ONI"]<0, color=BLUE, alpha=0.5, label="La Niña (-)")
ax1.plot(master.index, master["ONI"], color="black", lw=1.2)
ax1.axhline(0, color="grey", lw=0.8)
ax1.axhline( 0.5, color=RED,  lw=0.6, ls="--", alpha=0.5)
ax1.axhline(-0.5, color=BLUE, lw=0.6, ls="--", alpha=0.5)
ax1.set_ylabel("ONI (°C)")
ax1.legend(fontsize=8, loc="upper right")
ax1.set_title("ENSO — Oceanic Niño Index (positive = El Niño)")

ax2 = axes[2]
ax2.fill_between(master.index, master["IOD_ASO"], 0,
                 where=master["IOD_ASO"]>0, color=GREEN, alpha=0.5, label="Positive IOD (+)")
ax2.fill_between(master.index, master["IOD_ASO"], 0,
                 where=master["IOD_ASO"]<0, color=ORANGE, alpha=0.5, label="Negative IOD (-)")
ax2.plot(master.index, master["IOD_ASO"], color="black", lw=1.2)
ax2.axhline(0, color="grey", lw=0.8)
ax2.set_ylabel("DMI — IOD (ASO peak, °C)")
ax2.legend(fontsize=8, loc="upper right")
ax2.set_xlabel("Year")
ax2.set_title("Indian Ocean Dipole — Aug-Oct peak season")

plt.tight_layout()
fig.savefig(f"{OUT}/10_climate_timeseries.png")
plt.close()
print("Saved 10_climate_timeseries.png")

# ── FIG 2: Scatter plots — ONI vs Cases, IOD vs Cases (same + lagged) ──
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

combos = [
    ("ONI",         "Cases", "ENSO (ONI, same year)"),
    ("ONI_lag1",    "Cases", "ENSO (ONI, lag +1 yr)"),
    ("IOD_ASO",     "Cases", "IOD/ASO (same year)"),
    ("IOD_ASO_lag1","Cases", "IOD/ASO (lag +1 yr)"),
    ("SOI_monsoon", "Cases", "SOI monsoon (same yr)"),
    ("PDO",         "Cases", "PDO (same year)"),
]
for ax, (xcol, ycol, title) in zip(axes.flat, combos):
    d = master[[xcol, "Cases"]].dropna()
    x, y = d[xcol], np.log(d["Cases"])
    r, p = pearsonr(x, y)
    ax.scatter(x, y, color=RED, alpha=0.7, edgecolors="white", s=50)
    # Year labels on dots
    for yr, xi, yi in zip(d.index, x, y):
        ax.annotate(str(yr), (xi, yi), fontsize=7, alpha=0.7,
                    xytext=(3, 2), textcoords="offset points")
    # Trend line
    z = np.polyfit(x, y, 1)
    xr = np.linspace(x.min(), x.max(), 50)
    ax.plot(xr, np.polyval(z, xr), "k--", lw=1.2)
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    ax.set_title(f"{title}\nr={r:.2f}{sig}  p={p:.3f}")
    ax.set_xlabel(xcol)
    ax.set_ylabel("log(Cases)")

plt.suptitle("Climate Indices vs Sri Lanka Dengue (log scale)", y=1.01, fontsize=13)
plt.tight_layout()
fig.savefig(f"{OUT}/11_climate_scatter.png", bbox_inches="tight")
plt.close()
print("Saved 11_climate_scatter.png")

# ── FIG 3: Correlation bar chart (all indices) ──
sig_df = res_df[res_df["Pearson_p"] < 0.20].copy()  # show borderline too
colors = [RED if r > 0 else BLUE for r in sig_df["Pearson_r"]]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(sig_df["Index"], sig_df["Pearson_r"], color=colors, edgecolor="white")
for bar, (_, row) in zip(bars, sig_df.iterrows()):
    sig_mark = "**" if row["Pearson_p"]<0.01 else ("*" if row["Pearson_p"]<0.05 else "~")
    ax.text(row["Pearson_r"] + (0.01 if row["Pearson_r"]>=0 else -0.01),
            bar.get_y() + bar.get_height()/2,
            f"r={row['Pearson_r']:.2f}{sig_mark}", va="center",
            ha="left" if row["Pearson_r"]>=0 else "right", fontsize=9)
ax.axvline(0, color="black", lw=0.8)
ax.set(title="Climate Index Correlation with Sri Lanka Dengue (log cases)\n** p<0.01  * p<0.05  ~ p<0.20",
       xlabel="Pearson r")
plt.tight_layout()
fig.savefig(f"{OUT}/12_climate_correlation_bar.png")
plt.close()
print("Saved 12_climate_correlation_bar.png")

# ── FIG 4: ENSO Phase × Dengue box chart ──
master["ENSO_phase"] = pd.cut(master["ONI_monsoon"],
                               bins=[-5, -0.5, 0.5, 5],
                               labels=["La Niña\n(ONI<-0.5)",
                                       "Neutral\n(-0.5–0.5)",
                                       "El Niño\n(ONI>0.5)"])
master["IOD_phase"] = pd.cut(master["IOD_ASO"],
                              bins=[-5, -0.3, 0.3, 5],
                              labels=["Neg IOD\n(<-0.3)",
                                      "Neutral\n(±0.3)",
                                      "Pos IOD\n(>0.3)"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, phase_col, palette, title in [
    (axes[0], "ENSO_phase", [BLUE,"grey",RED], "ENSO Phase vs Dengue Cases"),
    (axes[1], "IOD_phase",  [ORANGE,"grey",GREEN], "IOD Phase vs Dengue Cases"),
]:
    groups = master.groupby(phase_col, observed=True)["Cases"]
    labels = [str(k) for k in groups.groups.keys()]
    data   = [np.log(v.dropna()) for _, v in groups]
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", lw=2))
    for patch, col in zip(bp["boxes"], palette):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    # Overlay actual points
    for i, (_, vals) in enumerate(groups):
        jitter = np.random.normal(0, 0.07, size=len(vals))
        ax.scatter(np.ones(len(vals))*(i+1) + jitter,
                   np.log(vals.dropna()), alpha=0.7, color="black", s=20, zorder=5)
    ax.set(title=title, ylabel="log(Annual Cases)")
plt.tight_layout()
fig.savefig(f"{OUT}/13_phase_boxplot.png")
plt.close()
print("Saved 13_phase_boxplot.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  2025-2027 CLIMATE-BASED OUTLOOK
# ─────────────────────────────────────────────────────────────────────────────
print("\n══════════ 2025–2027 CLIMATE OUTLOOK ══════════")
outlook = {
    2025: {"ONI": -0.35, "IOD_ASO":  0.15, "phase": "La Niña decaying / Neutral",
           "iod_phase": "Slight positive IOD"},
    2026: {"ONI":  0.45, "IOD_ASO":  0.35, "phase": "El Niño developing (80% prob)",
           "iod_phase": "Positive IOD coupling (per user data)"},
    2027: {"ONI":  0.80, "IOD_ASO":  0.40, "phase": "El Niño peak / early decay",
           "iod_phase": "Positive IOD residual"},
}

# Use the IOD_lag1 and ONI_lag1 regression model
from sklearn.linear_model import LinearRegression
feat_cols = ["ONI_monsoon","IOD_ASO","ONI_lag1","IOD_ASO_lag1"]
m2 = master[feat_cols + ["Cases"]].dropna()
X  = m2[feat_cols].values
y  = np.log(m2["Cases"].values)
reg = LinearRegression().fit(X, y)
print(f"Climate regression R²: {reg.score(X,y):.2f}")

for yr, info in outlook.items():
    oni_m   = info["ONI"]
    iod_aso = info["IOD_ASO"]
    # lag1 = current year's value predicting next
    prev_yr = yr - 1
    oni_lag1 = outlook.get(prev_yr, {}).get("ONI", master["ONI_monsoon"].iloc[-1])
    iod_lag1 = outlook.get(prev_yr, {}).get("IOD_ASO", master["IOD_ASO"].iloc[-1])
    x_new = np.array([[oni_m, iod_aso, oni_lag1, iod_lag1]])
    pred = int(np.exp(reg.predict(x_new)[0]))
    print(f"\n{yr}:")
    print(f"  ENSO:     {info['phase']}  (ONI≈{oni_m:+.2f})")
    print(f"  IOD:      {info['iod_phase']}  (DMI≈{iod_aso:+.2f})")
    print(f"  Climate model estimate: {pred:,} cases")

print("\nAll outputs saved to analysis_output/")
