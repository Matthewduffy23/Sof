# app_top20_tiles.py — Top 20 Tiles View (compact + taller, final)
# Requirements: streamlit, pandas, numpy.

import os
import math
import io
import re
from pathlib import Path

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
    .player-card {
        width:min(420px, 96%);
        display:grid;
        grid-template-columns: 96px 1fr 48px;
        gap:12px;
        align-items:center;
        background:var(--card);
        border:1px solid #252b3a;
        border-radius:18px;
        padding:16px;
    }
    .avatar {
        width:96px;
        height:96px;
        border-radius:12px;
        object-fit:cover;
        background:#0b0d12;
        border:1px solid #2a3145;
    }
    .name { font-weight:800; font-size:22px; color:#e8ecff; margin-bottom:6px; }
    .sub { color:var(--muted); font-size:15px; }
    .pill { padding:2px 10px; border-radius:9px; font-weight:800; font-size:18px; color:#0b0d12; display:inline-block; min-width:42px; text-align:center; }
    .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin:4px 0; }
    .rank { color:#94a0c6; font-weight:800; font-size:18px; text-align:right; }
    .chip { background:var(--soft); color:#cbd5f5; border:1px solid #2d3550; padding:3px 10px; border-radius:9px; font-size:13px; }
    .leftcol { display:flex; flex-direction:column; align-items:center; gap:8px; }
    .divider { height:12px; }
    .teamline { color:#e6ebff; font-size:15px; font-weight:400; margin-top:2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- CONFIG & METADATA -----------------
INCLUDED_LEAGUES = [...]
FEATURES = [...]
ROLES = {...}

# Combine both roles
COMBINED_METRICS = {}
for r in ROLES.values():
    for k, w in r["metrics"].items():
        COMBINED_METRICS[k] = COMBINED_METRICS.get(k, 0) + w

LEAGUE_STRENGTHS = {...}

REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Contract expires","Goals"}

# ----------------- DATA LOADING -----------------
@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

def load_df(csv_name: str = "WORLDJUNE25.csv") -> pd.DataFrame:
    for p in [Path.cwd()/csv_name, Path(__file__).resolve().parent.parent/csv_name, Path(__file__).resolve().parent/csv_name]:
        if p.exists():
            return _read_csv_from_path(str(p))
    st.warning(f"Could not find **{csv_name}**. Upload below.")
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if up is None:
        st.stop()
    return _read_csv_from_bytes(up.getvalue())

df = load_df()

# ----------------- SIDEBAR FILTERS -----------------
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
    apply_contract = st.checkbox("Filter by contract expiry", value=False)
    cutoff_year = st.slider("Max contract year", 2025, 2030, 2026)
    df["Market value"] = pd.to_numeric(df["Market value"], errors="coerce")
    mv_col = "Market value"
    mv_max_raw = int(np.nanmax(df[mv_col])) if df[mv_col].notna().any() else 50_000_000
    mv_cap = int(math.ceil(mv_max_raw / 5_000_000) * 5_000_000)
    use_m = st.checkbox("Adjust in millions", True)
    if use_m:
        max_m = int(mv_cap // 1_000_000)
        mv_min_m, mv_max_m = st.slider("Range (M€)", 0, max_m, (0, max_m))
        min_value = mv_min_m * 1_000_000
        max_value = mv_max_m * 1_000_000
    else:
        min_value, max_value = st.slider("Range (€)", 0, mv_cap, (0, mv_cap), step=100_000)
    min_strength, max_strength = st.slider("League quality", 0, 101, (0, 101))
    beta = st.slider("League weighting beta", 0.0, 1.0, 0.40, 0.05)
    top_n = st.number_input("How many tiles", 5, 100, 20, 5)

# ----------------- PROCESSING (same as your original) -----------------
# [filters, percentiles, scoring, overall, potential — unchanged]

# ----------------- IMAGE FALLBACK -----------------
FALLBACK_URL = "https://i.redd.it/43axcjdu59nd1.jpeg"

def guess_fotmob_url(team: str, player: str) -> str:
    def slug(x):
        return re.sub(r"[^a-z0-9]+", "-", str(x).lower()).strip("-")
    surname = str(player).split()[-1]
    return f"https://images.fotmob.com/image_resources/playerimages/{slug(surname)}-{slug(team)}.png"

# ----------------- RENDER -----------------
for idx, row in ranked.iterrows():
    rank = idx + 1
    player = str(row["Player"])
    surname = player.split()[-1] if player else ""
    team = str(row["Team"])
    pos = str(row["Position"])
    age = int(row["Age"]) if not pd.isna(row["Age"]) else 0
    overall = int(round(row["Overall Rating"]))
    potential = int(round(row["Potential"]))
    contract_year = int(row["Contract Year"])

    img_url_try = guess_fotmob_url(team, surname)
    if not img_url_try or img_url_try.strip() == "":
        img_url_try = FALLBACK_URL

    ov_style = f"background:{rating_color(overall)};"
    po_style = f"background:{rating_color(potential)};"

    st.markdown(f"""
    <div class='wrap'>
      <div class='player-card'>
        <div class='leftcol'>
          <img class='avatar' src='{img_url_try}' onerror="this.onerror=null;this.src='{FALLBACK_URL}';" />
          <div class='row'>
            <span class='chip'>{age} y.o.</span>
            <span class='chip'>{contract_year if contract_year>0 else '—'}</span>
          </div>
        </div>
        <div>
          <div class='name'>{player}</div>
          <div class='row'>
            <span class='pill' style='{ov_style}'>{overall}</span>
            <span class='sub'>Overall rating</span>
          </div>
          <div class='row'>
            <span class='pill' style='{po_style}'>{potential}</span>
            <span class='sub'>Potential</span>
          </div>
          <div class='row'><span class='chip'>{pos}</span></div>
          <div class='teamline'>{team}</div>
        </div>
        <div class='rank'>#{rank}</div>
      </div>
    </div>
    <div class='divider'></div>
    """, unsafe_allow_html=True)












