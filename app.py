# app_top20_tiles.py — Top 20 Tiles View (compact + taller)
# Requirements: streamlit, pandas, numpy.
# Combines both roles into a single scoring system and renders dark tiles.

import os
import math
from pathlib import Path
import io
import re

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Advanced Striker Scouting – Top 20 Tiles", layout="wide")
st.title("🔎 Advanced Striker Scouting – Top 20 Tiles")
st.caption("Overall = league-weighted combined-role score. Potential = Overall + age bonus.")

# ----------------- STYLE -----------------
st.markdown(
    """
    <style>
    :root { --bg: #0f1115; --card: #161a22; --muted: #a8b3cf; --soft: #202633; }
    .block-container { padding-top: 0.8rem; }
    body { background-color: var(--bg); }
    .wrap { display:flex; justify-content:center; }
    .player-card { width:min(820px, 96%); display:grid; grid-template-columns: 110px 1fr 56px; gap:12px; align-items:center; background:var(--card); border:1px solid #252b3a; border-radius:18px; padding:16px; }
    .avatar { width:110px; height:110px; border-radius:12px; object-fit:cover; background:#0b0d12; border:1px solid #2a3145; }
    .name { font-weight:800; font-size:22px; color:#e8ecff; margin-bottom:6px; }
    .sub { color:var(--muted); font-size:15px; }
    .pill { padding:2px 10px; border-radius:9px; font-weight:800; font-size:18px; color:#0b0d12; display:inline-block; min-width:42px; text-align:center; }
    .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin:4px 0; }
    .rank { color:#94a0c6; font-weight:800; font-size:18px; text-align:right; }
    .chip { background:var(--soft); color:#cbd5f5; border:1px solid #2d3550; padding:3px 10px; border-radius:9px; font-size:13px; }
    .leftcol { display:flex; flex-direction:column; align-items:center; gap:8px; }
    .divider { height:12px; }
    .teamline { color:#e6ebff; font-size:18px; font-weight:700; margin-top:2px; }
</style>
    """,
    unsafe_allow_html=True,
)

# ----------------- CONFIG -----------------
INCLUDED_LEAGUES = [
    'England 1.', 'England 2.', 'England 3.', 'England 4.', 'England 5.',
    'England 6.', 'England 7.', 'England 8.', 'England 9.', 'England 10.',
    'Albania 1.', 'Algeria 1.', 'Andorra 1.', 'Argentina 1.', 'Armenia 1.',
    'Australia 1.', 'Austria 1.', 'Austria 2.', 'Azerbaijan 1.', 'Belgium 1.',
    'Belgium 2.', 'Bolivia 1.', 'Bosnia 1.', 'Brazil 1.', 'Brazil 2.', 'Brazil 3.',
    'Bulgaria 1.', 'Canada 1.', 'Chile 1.', 'Colombia 1.', 'Costa Rica 1.',
    'Croatia 1.', 'Cyprus 1.', 'Czech 1.', 'Czech 2.', 'Denmark 1.', 'Denmark 2.',
    'Ecuador 1.', 'Egypt 1.', 'Estonia 1.', 'Finland 1.', 'France 1.', 'France 2.',
    'France 3.', 'Georgia 1.', 'Germany 1.', 'Germany 2.', 'Germany 3.', 'Germany 4.',
    'Greece 1.', 'Hungary 1.', 'Iceland 1.', 'Israel 1.', 'Israel 2.', 'Italy 1.',
    'Italy 2.', 'Italy 3.', 'Japan 1.', 'Japan 2.', 'Kazakhstan 1.', 'Korea 1.',
    'Latvia 1.', 'Lithuania 1.', 'Malta 1.', 'Mexico 1.', 'Moldova 1.', 'Morocco 1.',
    'Netherlands 1.', 'Netherlands 2.', 'North Macedonia 1.', 'Northern Ireland 1.',
    'Norway 1.', 'Norway 2.', 'Paraguay 1.', 'Peru 1.', 'Poland 1.', 'Poland 2.',
    'Portugal 1.', 'Portugal 2.', 'Portugal 3.', 'Qatar 1.', 'Ireland 1.', 'Romania 1.',
    'Russia 1.', 'Saudi 1.', 'Scotland 1.', 'Scotland 2.', 'Scotland 3.', 'Serbia 1.',
    'Serbia 2.', 'Slovakia 1.', 'Slovakia 2.', 'Slovenia 1.', 'Slovenia 2.', 'South Africa 1.',
    'Spain 1.', 'Spain 2.', 'Spain 3.', 'Sweden 1.', 'Sweden 2.', 'Switzerland 1.',
    'Switzerland 2.', 'Tunisia 1.', 'Turkey 1.', 'Turkey 2.', 'Ukraine 1.', 'UAE 1.',
    'USA 1.', 'USA 2.', 'Uruguay 1.', 'Uzbekistan 1.', 'Venezuela 1.', 'Wales 1.'
]

FEATURES = [
    'Defensive duels per 90', 'Defensive duels won, %',
    'Aerial duels per 90', 'Aerial duels won, %',
    'PAdj Interceptions', 'Non-penalty goals per 90', 'xG per 90',
    'Shots per 90', 'Shots on target, %', 'Goal conversion, %',
    'Crosses per 90', 'Accurate crosses, %', 'Dribbles per 90',
    'Successful dribbles, %', 'Head goals per 90', 'Key passes per 90',
    'Touches in box per 90', 'Progressive runs per 90', 'Accelerations per 90',
    'Passes per 90', 'Accurate passes, %', 'xA per 90',
    'Passes to penalty area per 90', 'Accurate passes to penalty area, %',
    'Deep completions per 90', 'Smart passes per 90', ]

ROLES = {
    'Goal Threat CF': {
        'metrics': {'Non-penalty goals per 90': 3,'Shots per 90': 1.5,'xG per 90': 3,
                    'Touches in box per 90': 1,'Shots on target, %': 0.5}
    },
    'Link-Up CF': {
        'metrics': {'Passes per 90': 2, 'Passes to penalty area per 90': 1.5,
                    'Deep completions per 90': 1, 'Smart passes per 90': 1.5,
                    'Accurate passes, %': 1.5, 'Key passes per 90': 1,
                    'Dribbles per 90': 2, 'Successful dribbles, %': 1,
                    'Progressive runs per 90': 2, 'xA per 90': 3}
    },
}

# Combine both roles into ONE metric (sum weights; overlapping metrics add up)
COMBINED_METRICS = {}
for r in ROLES.values():
    for k, w in r['metrics'].items():
        COMBINED_METRICS[k] = COMBINED_METRICS.get(k, 0) + w

LEAGUE_STRENGTHS = {
    'England 1.':100.00,'Italy 1.':97.14,'Spain 1.':94.29,'Germany 1.':94.29,'France 1.':91.43,
    'Brazil 1.':82.86,'England 2.':71.43,'Portugal 1.':71.43,'Argentina 1.':71.43,
    'Belgium 1.':68.57,'Mexico 1.':68.57,'Turkey 1.':65.71,'Germany 2.':65.71,'Spain 2.':65.71,
    'France 2.':65.71,'USA 1.':65.71,'Russia 1.':65.71,'Colombia 1.':62.86,'Netherlands 1.':62.86,
    'Austria 1.':62.86,'Switzerland 1.':62.86,'Denmark 1.':62.86,'Croatia 1.':62.86,
    'Japan 1.':62.86,'Korea 1.':62.86,'Italy 2.':62.86,'Czech 1.':57.14,'Norway 1.':57.14,
    'Poland 1.':57.14,'Romania 1.':57.14,'Israel 1.':57.14,'Algeria 1.':57.14,'Paraguay 1.':57.14,
    'Saudi 1.':57.14,'Uruguay 1.':57.14,'Morocco 1.':57.00,'Brazil 2.':56.00,'Ukraine 1.':55.00,
    'Ecuador 1.':54.29,'Spain 3.':54.29,'Scotland 1.':58.00,'Chile 1.':51.43,'Cyprus 1.':51.43,
    'Portugal 2.':51.43,'Slovakia 1.':51.43,'Australia 1.':51.43,'Hungary 1.':51.43,'Egypt 1.':51.43,
    'England 3.':51.43,'France 3.':48.00,'Japan 2.':48.00,'Bulgaria 1.':48.57,'Slovenia 1.':48.57,
    'Venezuela 1.':48.00,'Germany 3.':45.71,'Albania 1.':44.00,'Serbia 1.':42.86,'Belgium 2.':42.86,
    'Bosnia 1.':42.86,'Kosovo 1.':42.86,'Nigeria 1.':42.86,'Azerbaijan 1.':50.00,'Bolivia 1.':50.00,
    'Costa Rica 1.':50.00,'South Africa 1.':50.00,'UAE 1.':50.00,'Georgia 1.':40.00,'Finland 1.':40.00,
    'Italy 3.':40.00,'Peru 1.':40.00,'Tunisia 1.':40.00,'USA 2.':40.00,'Armenia 1.':40.00,
    'North Macedonia 1.':40.00,'Qatar 1.':40.00,'Uzbekistan 1.':42.00,'Norway 2.':42.00,
    'Kazakhstan 1.':42.00,'Poland 2.':38.00,'Denmark 2.':37.00,'Czech 2.':37.14,'Israel 2.':37.14,
    'Netherlands 2.':37.14,'Switzerland 2.':37.14,'Iceland 1.':34.29,'Ireland 1.':34.29,'Sweden 2.':34.29,
    'Germany 4.':34.29,'Malta 1.':30.00,'Turkey 2.':31.43,'Canada 1.':28.57,'England 4.':28.57,
    'Scotland 2.':28.57,'Moldova 1.':28.57,'Austria 2.':25.71,'Lithuania 1.':25.71,'Brazil 3.':25.00,
    'England 7.':25.00,'Slovenia 2.':22.00,'Latvia 1.':22.86,'Serbia 2.':20.00,'Slovakia 2.':20.00,
    'England 9.':20.00,'England 8.':15.00,'Montenegro 1.':14.29,'Wales 1.':12.00,'Portugal 3.':11.43,
    'Northern Ireland 1.':11.43,'England 10.':10.00,'Scotland 3.':10.00,'England 6.':10.00
}

REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Contract expires","Goals"}

CF_PREFIXES = ('CF',)

def position_filter(pos):
    return str(pos).strip().upper().startswith(CF_PREFIXES)

# ----------------- DATA LOADING -----------------
@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))


def load_df(csv_name: str = "WORLDJUNE25.csv") -> pd.DataFrame:
    candidates = [
        Path.cwd() / csv_name,
        Path(__file__).resolve().parent.parent / csv_name,
        Path(__file__).resolve().parent / csv_name,
    ]
    for p in candidates:
        if p.exists():
            return _read_csv_from_path(str(p))
    st.warning(f"Could not find **{csv_name}**. Upload below.")
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if up is None:
        st.stop()
    return _read_csv_from_bytes(up.getvalue())


df = load_df()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("Filters")
    leagues_avail = sorted(set(INCLUDED_LEAGUES) | set(df.get("League", pd.Series([])).dropna().unique()))
    leagues_sel = st.multiselect("Leagues", leagues_avail, default=INCLUDED_LEAGUES)

    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    min_minutes, max_minutes = st.slider("Minutes played", 0, 5000, (500, 5000))
    age_min_data = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_data = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    min_age, max_age = st.slider("Age", age_min_data, age_max_data, (16, 40))

    pos_text = st.text_input("Position startswith", "CF")

    df["Contract expires"] = pd.to_datetime(df["Contract expires"], errors="coerce")
    apply_contract = st.checkbox("Filter by contract expiry", value=False)
    cutoff_year = st.slider("Max contract year (inclusive)", 2025, 2030, 2026)

    df["Market value"] = pd.to_numeric(df["Market value"], errors="coerce")
    mv_col = "Market value"
    mv_max_raw = int(np.nanmax(df[mv_col])) if df[mv_col].notna().any() else 50_000_000
    mv_cap = int(math.ceil(mv_max_raw / 5_000_000) * 5_000_000)
    st.markdown("**Market value (€)**")
    use_m = st.checkbox("Adjust in millions", True)
    if use_m:
        max_m = int(mv_cap // 1_000_000)
        mv_min_m, mv_max_m = st.slider("Range (M€)", 0, max_m, (0, max_m))
        min_value = mv_min_m * 1_000_000
        max_value = mv_max_m * 1_000_000
    else:
        min_value, max_value = st.slider("Range (€)", 0, mv_cap, (0, mv_cap), step=100_000)

    min_strength, max_strength = st.slider("League quality (strength)", 0, 101, (0, 101))
    beta = st.slider("League weighting beta (for OVERALL)", 0.0, 1.0, 0.40, 0.05)

    top_n = st.number_input("How many tiles", 5, 100, 20, 5)

# ----------------- VALIDATION -----------------
missing = [c for c in REQUIRED_BASE if c not in df.columns]
if missing:
    st.error(f"Dataset missing required base columns: {missing}")
    st.stop()
missing_feats = [c for c in FEATURES if c not in df.columns]
if missing_feats:
    st.error(f"Dataset missing required feature columns: {missing_feats}")
    st.stop()

# ----------------- FILTER POOL -----------------
df_f = df[df["League"].isin(leagues_sel)].copy()
df_f = df_f[df_f["Position"].astype(str).str.upper().str.startswith(pos_text.upper())]
df_f = df_f[df_f["Minutes played"].between(min_minutes, max_minutes)]
df_f = df_f[df_f["Age"].between(min_age, max_age)]
if apply_contract:
    df_f = df_f[df_f["Contract expires"].dt.year <= cutoff_year]
for c in FEATURES:
    df_f[c] = pd.to_numeric(df_f[c], errors="coerce")

df_f = df_f.dropna(subset=FEATURES)

# League strength map
df_f["League Strength"] = df_f["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
df_f = df_f[(df_f["League Strength"] >= float(min_strength)) & (df_f["League Strength"] <= float(max_strength))]
df_f = df_f[(df_f["Market value"] >= min_value) & (df_f["Market value"] <= max_value)]
if df_f.empty:
    st.warning("No players after filters. Loosen filters.")
    st.stop()

# ----------------- Percentiles per league -----------------
for feat in FEATURES:
    df_f[f"{feat} Percentile"] = df_f.groupby("League")[feat].transform(lambda x: x.rank(pct=True) * 100.0)

# ----------------- Combined role score -----------------
def combined_role_score(df_in: pd.DataFrame, metrics: dict) -> pd.Series:
    total_w = sum(metrics.values()) if metrics else 1.0
    wsum = np.zeros(len(df_in))
    for m, w in metrics.items():
        col = f"{m} Percentile"
        if col in df_in.columns:
            wsum += df_in[col].values * w
    return wsum / total_w

df_f["Combined Score"] = combined_role_score(df_f, COMBINED_METRICS)

# ----------------- Overall & Potential -----------------

def age_bonus(age: float) -> int:
    table = {27:0, 26:1, 25:2, 24:2, 23:3, 22:3, 21:4, 20:5, 19:6, 18:7, 17:8, 16:9}
    a = int(age) if not pd.isna(age) else 27
    if a >= 28: return 0
    if a < 16: return 9
    return table.get(a, 0)

# overall = (1-beta) * CombinedScore + beta * league_strength
df_f["Overall Rating"] = (1 - beta) * df_f["Combined Score"] + beta * (df_f["League Strength"].fillna(50))
df_f["Potential"] = df_f.apply(lambda r: r["Overall Rating"] + age_bonus(r["Age"]), axis=1)

# Contract year (just year)
df_f["Contract Year"] = pd.to_datetime(df_f["Contract expires"], errors="coerce").dt.year.fillna(0).astype(int)

# ----------------- Top N -----------------
ranked = df_f.sort_values("Overall Rating", ascending=False).head(int(top_n)).copy().reset_index(drop=True)

# ----------------- Image helper -----------------
FALLBACK_URL = "https://i.redd.it/43axcjdu59nd1.jpeg"

def guess_fotmob_url(team: str, player: str) -> str:
    def slug(x):
        return re.sub(r"[^a-z0-9]+", "-", str(x).lower()).strip("-")
    surname = str(player).split()[-1]
    parts = [slug(surname), slug(team)]
    return f"https://images.fotmob.com/image_resources/playerimages/{'-'.join(parts)}.png"

# SoFIFA-style rating colors (red→orange→yellow→light green→dark green)
# We interpolate across 5 anchors at score 0, 50, 65, 75, 85, 100
PALETTE = [
    (0,   (208,  2, 27)),   # deep red
    (50,  (245,166, 35)),   # orange
    (65,  (248,231, 28)),   # yellow
    (75,  (126,211, 33)),   # light green
    (85,  (65, 117,  5)),   # dark green
    (100, (40,  90,  4)),   # deeper green for 100
]

def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i]-a[i]) * t)) for i in range(3))

def rating_color(v: float) -> str:
    # clamp for color only
    v = max(0.0, min(100.0, float(v)))
    # find segment
    for i in range(len(PALETTE)-1):
        x0, c0 = PALETTE[i]
        x1, c1 = PALETTE[i+1]
        if v <= x1:
            t = 0 if x1 == x0 else (v - x0) / (x1 - x0)
            r,g,b = _lerp(c0, c1, t)
            return f"rgb({r},{g},{b})"
    r,g,b = PALETTE[-1][1]
    return f"rgb({r},{g},{b})"

# ----------------- RENDER -----------------
for idx, row in ranked.iterrows():
    rank = idx + 1
    player = str(row.get("Player", ""))
    surname = player.split()[-1] if player else ""
    team = str(row.get("Team", ""))
    pos = str(row.get("Position", ""))
    age = int(row.get("Age", 0)) if not pd.isna(row.get("Age", np.nan)) else 0
    overall = float(row["Overall Rating"]) if not pd.isna(row["Overall Rating"]) else 0
    potential = float(row["Potential"]) if not pd.isna(row["Potential"]) else overall
    contract_year = int(row.get("Contract Year", 0))

    overall_i = int(round(overall))
    potential_i = int(round(potential))

    img_url_try = guess_fotmob_url(team, surname)

    ov_style = f"background:{rating_color(overall_i)};"
    po_style = f"background:{rating_color(potential_i)};"

    st.markdown(f"""
    <div class='wrap'>
      <div class='player-card'>
        <div class='leftcol'>
          <img class='avatar' src='{img_url_try}' onerror=\"this.onerror=null;this.src='{FALLBACK_URL}';\" />
          <div class='row'>
            <span class='chip'>{age} y.o.</span>
            <span class='chip'>{contract_year if contract_year>0 else '—'}</span>
          </div>
        </div>
        <div>
          <div class='name'>{player}</div>
          <div class='row'>
            <span class='pill' style='{ov_style}'>{overall_i}</span>
            <span class='sub'>Overall rating</span>
          </div>
          <div class='row'>
            <span class='pill' style='{po_style}'>{potential_i}</span>
            <span class='sub'>Potential</span>
          </div>
          <div class='row'>
            <span class='chip'>{pos}</span>
          </div>
          <div class='teamline'>{team}</div>
        </div>
        <div class='rank'>#{rank}</div>
      </div>
    </div>
    <div class='divider'></div>
    """, unsafe_allow_html=True)











