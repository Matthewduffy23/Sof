# app_top20_tiles.py — Top 20 Tiles with multi-source headshots
# Requirements: streamlit, pandas, numpy, requests, beautifulsoup4  (optional: lxml)

import os
import io
import re
import math
import base64
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
import numpy as np

# ----------------- PAGE -----------------
st.set_page_config(page_title="Advanced Striker Scouting — Top 20 Tiles", layout="wide")
st.title("🔎 Advanced Striker Scouting — Top 20 Tiles")
st.caption(
    "Overall = league-weighted combined-role score. Potential = Overall + age bonus. "
    "Headshots: SofaScore → SoFIFA → FIFACM → FotMob → Wikipedia → fallback (served as data URIs)."
)

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
        width:96px; height:96px; border-radius:12px;
        background-color:#0b0d12; border:1px solid #2a3145;
        background-repeat:no-repeat;
        background-position:center center;
        background-size:cover;
      }
      .source { color:#7f8cb5; font-size:11px; margin-top:3px; }
      .leftcol { display:flex; flex-direction:column; align-items:center; gap:4px; }
      .name { font-weight:800; font-size:22px; color:#e8ecff; margin-bottom:6px; }
      .sub { color:#a8b3cf; font-size:15px; }
      .pill { padding:2px 10px; border-radius:9px; font-weight:800; font-size:18px; color:#0b0d12; display:inline-block; min-width:42px; text-align:center; }
      .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin:4px 0; }
      .chip { background:var(--soft); color:#cbd5f5; border:1px solid #2d3550; padding:3px 10px; border-radius:9px; font-size:13px; }
      .teamline { color:#e6ebff; font-size:15px; font-weight:400; margin-top:2px; }
      .rank { color:#94a0c6; font-weight:800; font-size:18px; text-align:right; }
      .divider { height:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- CONFIG -----------------
FALLBACK_URL = "https://i.redd.it/43axcjdu59nd1.jpeg"
# Browser-like UA helps with some CDNs
UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/128.0.0.0 Safari/537.36")
}

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
    'Deep completions per 90', 'Smart passes per 90',
]

ROLES = {
    'Goal Threat CF': {
        'metrics': {
            'Non-penalty goals per 90': 3, 'Shots per 90': 1.5, 'xG per 90': 3,
            'Touches in box per 90': 1, 'Shots on target, %': 0.5
        }
    },
    'Link-Up CF': {
        'metrics': {
            'Passes per 90': 2, 'Passes to penalty area per 90': 1.5,
            'Deep completions per 90': 1, 'Smart passes per 90': 1.5,
            'Accurate passes, %': 1.5, 'Key passes per 90': 1,
            'Dribbles per 90': 2, 'Successful dribbles, %': 1,
            'Progressive runs per 90': 2, 'xA per 90': 3
        }
    },
}

# Combine both roles into one weight map (overlaps add)
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

# ----------------- DATA LOADING -----------------
@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

def load_df(csv_name: str = "WORLDJUNE25.csv") -> pd.DataFrame:
    candidates = [Path.cwd() / csv_name]
    try:
        candidates += [Path(__file__).resolve().parent.parent / csv_name,
                       Path(__file__).resolve().parent / csv_name]
    except NameError:
        pass
    for p in candidates:
        if p.exists():
            return _read_csv_from_path(str(p))
    st.warning(f"Could not find **{csv_name}**. Upload below.")
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if up is None:
        st.stop()
    return _read_csv_from_bytes(up.getvalue())

df = load_df()

# ----------------- CONNECTIVITY (used in sidebar) -----------------
@st.cache_data(show_spinner=False, ttl=5*60)
def check_connectivity() -> dict:
    targets = {
        "fallback": "https://i.redd.it/43axcjdu59nd1.jpeg",
        "sofifa":   "https://sofifa.com/",
        "fifacm":   "https://www.fifacm.com/players",
        "fotmob":   "https://www.fotmob.com/api/search?q=ronaldo",
        "sofascore":"https://api.sofascore.com/api/v1/search/all?q=messi",
        "wikipedia":"https://en.wikipedia.org/w/api.php",
    }
    ok = {}
    for k, u in targets.items():
        try:
            r = requests.get(u, headers=UA, timeout=6)
            ok[k] = r.ok
        except Exception:
            ok[k] = False
    return ok

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

    st.divider()
    st.subheader("Images")
    use_wikipedia = st.checkbox("Allow Wikipedia as last resort", value=True)
    show_img_source = st.checkbox("Show image source tag", value=True)

    # Optional: upload mapping CSV (Player,ImageURL)
    mapping_file = st.file_uploader("Optional headshot mapping CSV (Player,ImageURL)", type=["csv"])
    user_map = None
    if mapping_file is not None:
        try:
            mdf = pd.read_csv(mapping_file)
            if {"Player","ImageURL"}.issubset(set(mdf.columns)):
                user_map = dict(zip(mdf["Player"].astype(str), mdf["ImageURL"].astype(str)))
                st.caption(f"Loaded {len(user_map)} image mappings.")
            else:
                st.warning("CSV must have columns: Player, ImageURL")
        except Exception as e:
            st.warning(f"Could not read mapping CSV: {e}")

    net = check_connectivity()
    if not any(net.values()):
        st.error("No outbound internet detected — only fallback images will show.")
    else:
        bad = [k for k,v in net.items() if not v]
        if bad:
            st.warning("Some sources unreachable: " + ", ".join(bad))

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
df_f = df_f[df_f["Position"].astype(str).str.upper().str.startswith("CF")]
df_f = df_f[df_f["Minutes played"].between(min_minutes, max_minutes)]
df_f = df_f[df_f["Age"].between(min_age, max_age)]
if apply_contract:
    df_f = df_f[df_f["Contract expires"].dt.year <= cutoff_year]

for c in FEATURES:
    df_f[c] = pd.to_numeric(df_f[c], errors="coerce")
df_f = df_f.dropna(subset=FEATURES)

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
    # 27 +0 / 26 +1 / 25 +3 / 24 +3 / 23 +4 / 22 +5 / 21 +6 / 20 +6 / 19 +7 / 18 +8 / 17 +9 / 16 +10
    table = {27:0, 26:1, 25:3, 24:3, 23:4, 22:5, 21:6, 20:6, 19:7, 18:8, 17:9, 16:10}
    a = int(age) if not pd.isna(age) else 27
    if a >= 28: return 0
    if a < 16: return 9
    return table.get(a, 0)

df_f["Overall Rating"] = (1 - beta) * df_f["Combined Score"] + beta * (df_f["League Strength"].fillna(50))
df_f["Potential"] = df_f.apply(lambda r: r["Overall Rating"] + age_bonus(r["Age"]), axis=1)

df_f["Contract Year"] = pd.to_datetime(df_f["Contract expires"], errors="coerce").dt.year.fillna(0).astype(int)

# ----------------- Top N -----------------
ranked = df_f.sort_values("Overall Rating", ascending=False).head(int(top_n)).copy().reset_index(drop=True)

# ----------------- IMAGE RESOLUTION HELPERS -----------------
def _norm(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower()

TEAM_STRIP = re.compile(r"\b(fc|cf|sc|afc|ud|ac|bc|cd|sd|deportivo|club|city|united|town|athletic|sporting|sv|fk|sk|ik|if)\b", re.I)

def _simplify_team(t: str) -> str:
    t = _norm(t)
    t = TEAM_STRIP.sub("", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def _fetch_image_as_data_uri(url: str, timeout=8) -> str | None:
    try:
        r = requests.get(url, headers=UA, timeout=timeout, stream=True)
        if not r.ok:
            return None
        ctype = r.headers.get("Content-Type", "").lower()
        if not ctype.startswith("image/"):
            if url.lower().endswith(".png"): ctype = "image/png"
            elif url.lower().endswith(".jpg") or url.lower().endswith(".jpeg"): ctype = "image/jpeg"
            else: return None
        b64 = base64.b64encode(r.content).decode("ascii")
        return f"data:{ctype};base64,{b64}"
    except Exception:
        return None

# ---------- SofaScore ----------
@st.cache_data(show_spinner=False, ttl=60*60*24)
def sofascore_search_players(q: str) -> list:
    endpoints = [
        ("https://api.sofascore.com/api/v1/search/all", {"q": q}),
        ("https://api.sofascore.app/api/v1/search/all", {"q": q}),
    ]
    for url, params in endpoints:
        try:
            r = requests.get(url, params=params, headers=UA, timeout=7)
            if not r.ok:
                continue
            js = r.json()
            players = js.get("players") or js.get("results", {}).get("player") or js.get("players", {}).get("items") or []
            out = []
            for p in players:
                pid = p.get("id") or (p.get("player") or {}).get("id")
                name = p.get("name") or (p.get("player") or {}).get("name") or p.get("fullName")
                team = (p.get("team") or {}).get("name") or p.get("teamName") or ""
                if pid and name:
                    out.append({"id": pid, "name": name, "team": team})
            if out:
                return out
        except Exception:
            continue
    return []

def _score_name_team(player_q: str, team_q: str | None, name: str, team: str) -> int:
    score = 0
    if player_q and (player_q in name or name in player_q): score += 4
    pq_last = player_q.split()[-1] if player_q else ""
    if pq_last and pq_last in name: score += 2
    if team_q and team_q in team:  score += 3
    return score

@st.cache_data(show_spinner=False, ttl=60*60*24)
def sofascore_headshot(player_name: str, team_name: str | None = None) -> tuple[str | None, str]:
    if not player_name:
        return None, ""
    pn_raw = player_name.replace(".", " ").strip()
    tn_raw = (team_name or "").strip()
    surname = pn_raw.split()[-1] if pn_raw else ""
    queries = [f"{surname} {tn_raw}".strip(), f"{pn_raw} {tn_raw}".strip(), pn_raw]

    player_q = _norm(pn_raw)
    team_q = _simplify_team(tn_raw) if tn_raw else None

    best_id, best_score = None, -1
    for q in queries:
        plist = sofascore_search_players(q)
        for p in plist:
            name = _norm(p.get("name", ""))
            team = _simplify_team(p.get("team", ""))
            pid  = p.get("id")
            s = _score_name_team(player_q, team_q, name, team)
            if s > best_score:
                best_score, best_id = s, pid
        if best_id is not None and best_score >= 5:
            break
    if best_id is None:
        return None, ""

    # Try a few image URL variants
    img_candidates = [
        f"https://api.sofascore.com/api/v1/player/{best_id}/image",
        f"https://api.sofascore.app/api/v1/player/{best_id}/image",
        f"https://www.sofascore.com/api/v1/player/{best_id}/image",
        f"https://api.sofascore.com/api/v1/player/{best_id}/image?width=120&height=120",
    ]
    for u in img_candidates:
        data_uri = _fetch_image_as_data_uri(u)
        if data_uri:
            return u, "SofaScore"
    return None, ""

# ---------- SoFIFA ----------
@st.cache_data(show_spinner=False, ttl=60*60*24)
def sofifa_headshot(player_name: str, team_name: str | None = None) -> tuple[str | None, str]:
    try:
        q = f"{player_name} {team_name or ''}".strip()
        r = requests.get("https://sofifa.com/search", params={"keyword": q}, headers=UA, timeout=7)
        if not r.ok:
            return None, ""
        soup = BeautifulSoup(r.text, "html.parser")  # falls back if lxml not installed
        a = soup.select_one("a[href^='/player/']")
        if not a:
            return None, ""
        href = a.get("href", "")
        m = re.search(r"/player/(\d+)", href)
        if not m:
            return None, ""
        pid = int(m.group(1))
        a3 = f"{pid // 1000:03d}"
        b3 = f"{pid % 1000:03d}"
        for ver in ("25", "24", "23", "22", "21"):
            return f"https://cdn.sofifa.net/players/{a3}/{b3}/{ver}_120.png", "SoFIFA"
    except Exception:
        return None, ""
    return None, ""

# ---------- FIFACM ----------
@st.cache_data(show_spinner=False, ttl=60*60*24)
def fifacm_headshot(player_name: str, team_name: str | None = None) -> tuple[str | None, str]:
    try:
        q = f"{player_name} {team_name or ''}".strip()
        r = requests.get("https://www.fifacm.com/players", params={"name": q}, headers=UA, timeout=7)
        if not r.ok:
            return None, ""
        soup = BeautifulSoup(r.text, "html.parser")
        a = soup.select_one("a[href^='/player/']")
        if not a:
            return None, ""
        href = a.get("href", "")
        m = re.search(r"/player/(\d+)", href)
        if not m:
            return None, ""
        eaid = m.group(1)
        return f"https://cdn.fifacm.com/players/{eaid}.png", "FIFACM"
    except Exception:
        return None, ""

# ---------- FotMob ----------
@st.cache_data(show_spinner=False, ttl=60*60*24)
def fotmob_search(query: str) -> dict | None:
    try:
        r = requests.get("https://www.fotmob.com/api/search", params={"q": query}, headers=UA, timeout=6)
        if r.ok:
            return r.json()
    except Exception:
        return None
    return None

def _score_match(player_q: str, team_q: str | None, name: str, team: str) -> int:
    score = 0
    if player_q and (player_q in name or name in player_q):
        score += 4
    pq_last = player_q.split()[-1] if player_q else ""
    if pq_last and pq_last in name:
        score += 2
    if team_q and team_q in team:
        score += 3
    return score

@st.cache_data(show_spinner=False, ttl=60*60*24)
def fotmob_headshot(player_name: str, team_name: str | None = None) -> tuple[str | None, str]:
    try:
        pn_raw = player_name.replace(".", " ").strip()
        tn_raw = (team_name or "").strip()
        surname = pn_raw.split()[-1] if pn_raw else ""
        queries = [f"{surname} {tn_raw}".strip(), f"{pn_raw} {tn_raw}".strip(), pn_raw]
        player_q = _norm(pn_raw)
        team_q = _norm(tn_raw) if tn_raw else None
        best, best_score = None, -1
        for q in queries:
            data = fotmob_search(q)
            if not data:
                continue
            for p in data.get("players", []):
                name = _norm(p.get("name", ""))
                team = _norm(p.get("teamName", ""))
                pid = p.get("id")
                s = _score_match(player_q, team_q, name, team)
                if s > best_score:
                    best_score, best = s, pid
            if best is not None and best_score >= 5:
                break
        if best:
            return f"https://images.fotmob.com/image_resources/playerimages/{best}.png", "FotMob"
    except Exception:
        return None, ""
    return None, ""

# ---------- Wikipedia ----------
@st.cache_data(show_spinner=False, ttl=60*60*24)
def wiki_headshot(player_name: str, team_name: str | None = None) -> tuple[str | None, str]:
    try:
        q = f"{player_name} footballer"
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "prop": "pageimages",
                "piprop": "thumbnail",
                "pithumbsize": 256,
                "titles": q,
            },
            timeout=7,
            headers=UA,
        )
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            thumb = (page.get("thumbnail") or {}).get("source")
            if thumb:
                return thumb, "Wikipedia"
    except Exception:
        return None, ""
    return None, ""

def resolve_player_image_data_uri(player_name: str, team_name: str | None,
                                  *, use_wikipedia: bool,
                                  mapping: dict | None = None) -> tuple[str, str]:
    """
    Order:
      mapping → SofaScore → SoFIFA → FIFACM → FotMob → (Wikipedia if enabled) → fallback
    Returns (data_uri_or_url, source_tag).
    """
    # mapping override
    if mapping:
        url = mapping.get(player_name) or mapping.get(player_name.strip())
        if url:
            data_uri = _fetch_image_as_data_uri(url)
            if data_uri: return data_uri, "Mapping"
            return url, "Mapping(URL)"

    # try sources
    for resolver in (sofascore_headshot, sofifa_headshot, fifacm_headshot, fotmob_headshot):
        url, src = resolver(player_name, team_name)
        if url:
            data_uri = _fetch_image_as_data_uri(url)
            if data_uri: return data_uri, src
            return url, src

    if use_wikipedia:
        url, src = wiki_headshot(player_name, team_name)
        if url:
            data_uri = _fetch_image_as_data_uri(url)
            if data_uri: return data_uri, src
            return url, src

    return FALLBACK_URL, "Fallback"

# ----------------- SoFIFA-style colors -----------------
PALETTE = [
    (0,   (208,  2, 27)),   # red
    (50,  (245,166, 35)),   # orange
    (65,  (248,231, 28)),   # yellow
    (75,  (126,211, 33)),   # light green
    (85,  (65, 117,  5)),   # dark green
    (100, (40,  90,  4)),   # deeper green
]
def _lerp(a, b, t): return tuple(int(round(a[i] + (b[i]-a[i]) * t)) for i in range(3))
def rating_color(v: float) -> str:
    v = max(0.0, min(100.0, float(v)))
    for i in range(len(PALETTE) - 1):
        x0, c0 = PALETTE[i]; x1, c1 = PALETTE[i+1]
        if v <= x1:
            t = 0 if x1 == x0 else (v - x0) / (x1 - x0)
            r,g,b = _lerp(c0, c1, t); return f"rgb({r},{g},{b})"
    r,g,b = PALETTE[-1][1]; return f"rgb({r},{g},{b})"

# ----------------- RENDER -----------------
for idx, row in ranked.iterrows():
    rank = idx + 1
    player = str(row.get("Player", "")) or ""
    team = str(row.get("Team", "")) or ""
    pos = str(row.get("Position", "")) or ""
    age = int(row.get("Age", 0)) if not pd.isna(row.get("Age", np.nan)) else 0
    overall_i = int(round(float(row["Overall Rating"])))
    potential_i = int(round(float(row["Potential"])))
    contract_year = int(row.get("Contract Year", 0))

    primary, src_tag = resolve_player_image_data_uri(
        player, team, use_wikipedia=use_wikipedia, mapping=user_map
    )
    avatar_style = f"background-image: url('{primary}');"

    ov_style = f"background:{rating_color(overall_i)};"
    po_style = f"background:{rating_color(potential_i)};"

    src_html = f"<div class='source'>📸 {src_tag}</div>" if show_img_source else ""

    st.markdown(f"""
    <div class='wrap'>
      <div class='player-card'>
        <div class='leftcol'>
          <div class='avatar' style="{avatar_style}"></div>
          {src_html}
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

















