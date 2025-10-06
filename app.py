# app.py — Advanced Striker Scouting + Tiles (single-card) + robust FotMob headshots
# Requires: streamlit, pandas, numpy, matplotlib, requests

import os, io, math, re, unicodedata, json
from pathlib import Path

import requests
import streamlit as st
import pandas as pd
import numpy as np

# ----------------- PAGE -----------------
st.set_page_config(page_title="Advanced Striker Scouting System", layout="wide")
st.title("🔎 Advanced Striker Scouting System")
st.caption("Tables + dark Tiles with FotMob headshots (surname + team). Potential = Role Score.")

# ----------------- CONFIG (same as before; trimmed here for brevity) -----------------
INCLUDED_LEAGUES = [
    'England 1.','England 2.','England 3.','England 4.','England 5.',
    'England 6.','England 7.','England 8.','England 9.','England 10.',
    'Albania 1.','Algeria 1.','Andorra 1.','Argentina 1.','Armenia 1.',
    'Australia 1.','Austria 1.','Austria 2.','Azerbaijan 1.','Belgium 1.',
    'Belgium 2.','Bolivia 1.','Bosnia 1.','Brazil 1.','Brazil 2.','Brazil 3.',
    'Bulgaria 1.','Canada 1.','Chile 1.','Colombia 1.','Costa Rica 1.',
    'Croatia 1.','Cyprus 1.','Czech 1.','Czech 2.','Denmark 1.','Denmark 2.',
    'Ecuador 1.','Egypt 1.','Estonia 1.','Finland 1.','France 1.','France 2.',
    'France 3.','Georgia 1.','Germany 1.','Germany 2.','Germany 3.','Germany 4.',
    'Greece 1.','Hungary 1.','Iceland 1.','Israel 1.','Israel 2.','Italy 1.',
    'Italy 2.','Italy 3.','Japan 1.','Japan 2.','Kazakhstan 1.','Korea 1.',
    'Latvia 1.','Lithuania 1.','Malta 1.','Mexico 1.','Moldova 1.','Morocco 1.',
    'Netherlands 1.','Netherlands 2.','North Macedonia 1.','Northern Ireland 1.',
    'Norway 1.','Norway 2.','Paraguay 1.','Peru 1.','Poland 1.','Poland 2.',
    'Portugal 1.','Portugal 2.','Portugal 3.','Qatar 1.','Ireland 1.','Romania 1.',
    'Russia 1.','Saudi 1.','Scotland 1.','Scotland 2.','Scotland 3.','Serbia 1.',
    'Serbia 2.','Slovakia 1.','Slovakia 2.','Slovenia 1.','Slovenia 2.','South Africa 1.',
    'Spain 1.','Spain 2.','Spain 3.','Sweden 1.','Sweden 2.','Switzerland 1.',
    'Switzerland 2.','Tunisia 1.','Turkey 1.','Turkey 2.','Ukraine 1.','UAE 1.',
    'USA 1.','USA 2.','Uruguay 1.','Uzbekistan 1.','Venezuela 1.','Wales 1.'
]

PRESET_LEAGUES = {
    "Top 5 Europe": {'England 1.','France 1.','Germany 1.','Italy 1.','Spain 1.'},
    "Top 20 Europe": {
        'England 1.','Italy 1.','Spain 1.','Germany 1.','France 1.',
        'England 2.','Portugal 1.','Belgium 1.','Turkey 1.','Germany 2.','Spain 2.','France 2.',
        'Netherlands 1.','Austria 1.','Switzerland 1.','Denmark 1.','Croatia 1.','Italy 2.','Czech 1.','Norway 1.'
    },
    "EFL (England 2–4)": {'England 2.','England 3.','England 4.'}
}

FEATURES = [
    'Defensive duels per 90','Defensive duels won, %',
    'Aerial duels per 90','Aerial duels won, %',
    'PAdj Interceptions','Non-penalty goals per 90','xG per 90',
    'Shots per 90','Shots on target, %','Goal conversion, %',
    'Crosses per 90','Accurate crosses, %','Dribbles per 90',
    'Successful dribbles, %','Head goals per 90','Key passes per 90',
    'Touches in box per 90','Progressive runs per 90','Accelerations per 90',
    'Passes per 90','Accurate passes, %','xA per 90',
    'Passes to penalty area per 90','Accurate passes to penalty area, %',
    'Deep completions per 90','Smart passes per 90',
]

ROLES = {
    'Target Man CF': {
        'desc': "Aerial outlet, duel dominance, occupy CBs, threaten crosses & second balls.",
        'metrics': {'Aerial duels per 90': 3, 'Aerial duels won, %': 4}
    },
    'Goal Threat CF': {
        'desc': "High shot & xG volume, box presence, consistent SoT and finishing.",
        'metrics': {'Non-penalty goals per 90': 3,'Shots per 90': 1.5,'xG per 90': 3,
                    'Touches in box per 90': 1,'Shots on target, %': 0.5}
    },
    'Link-Up CF': {
        'desc': "Combine & create; link play; progress & deliver to the penalty area.",
        'metrics': {'Passes per 90': 2,'Passes to penalty area per 90': 1.5,'Deep completions per 90': 1,
                    'Smart passes per 90': 1.5,'Accurate passes, %': 1.5,'Key passes per 90': 1,
                    'Dribbles per 90': 2,'Successful dribbles, %': 1,'Progressive runs per 90': 2,'xA per 90': 3}
    },
    'All in': {
        'desc': "Blend of creation + scoring; balanced all-round attacking profile.",
        'metrics': {'xA per 90': 2,'Dribbles per 90': 2,'xG per 90': 3,'Non-penalty goals per 90': 3}
    }
}

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

# ----------------- DATA LOADER -----------------
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
    st.warning(f"Could not find **{csv_name}**. Please upload the CSV below.")
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if up is None:
        st.stop()
    return _read_csv_from_bytes(up.getvalue())

df = load_df()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("Filters")
    c1, c2, c3 = st.columns([1,1,1])
    use_top5  = c1.checkbox("Top-5 EU", value=False)
    use_top20 = c2.checkbox("Top-20 EU", value=False)
    use_efl   = c3.checkbox("EFL", value=False)

    seed = set()
    if use_top5:  seed |= PRESET_LEAGUES["Top 5 Europe"]
    if use_top20: seed |= PRESET_LEAGUES["Top 20 Europe"]
    if use_efl:   seed |= PRESET_LEAGUES["EFL (England 2–4)"]

    leagues_avail = sorted(set(INCLUDED_LEAGUES) | set(df.get("League", pd.Series([])).dropna().unique()))
    default_leagues = sorted(seed) if seed else INCLUDED_LEAGUES
    leagues_sel = st.multiselect("Leagues (add or prune the presets)", leagues_avail, default=default_leagues)

    # numeric coercions
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    st.slider("Minutes played", 0, 5000, (500, 5000), key="minmax_minutes")
    st.slider("Age", int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14,
              int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45, (16, 40), key="minmax_age")

    st.subheader("Position filter")
    st.caption("Players whose Position text starts with this (case-insensitive).")
    pos_text = st.text_input("Position startswith", "CF")

    apply_contract = st.checkbox("Filter by contract expiry", value=False)
    cutoff_year = st.slider("Max contract year (inclusive)", 2025, 2030, 2026)

    st.slider("League quality (strength)", 0, 101, (0, 101), key="minmax_strength")
    use_league_weighting = st.checkbox("Use league weighting in role score", value=False)
    beta = st.slider("League weighting beta", 0.0, 1.0, 0.40, 0.05,
                     help="0 = ignore league strength; 1 = only league strength")

    # Market value
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
    value_band_max = st.number_input("Value band (tab 4 max €)", min_value=0,
                                     value=min_value if min_value>0 else 5_000_000, step=250_000)

    st.subheader("Minimum performance thresholds")
    enable_min_perf = st.checkbox("Require minimum percentile on selected metrics", value=False)
    sel_metrics = st.multiselect("Metrics to threshold", FEATURES[:],
                                 default=['Non-penalty goals per 90','xG per 90'] if enable_min_perf else [])
    min_pct = st.slider("Minimum percentile (0–100)", 0, 100, 60)

    top_n = st.number_input("Top N per table", 5, 200, 50, 5)
    round_to = st.selectbox("Round output percentiles to", [0, 1], index=0)

# handy getters from sliders
min_minutes, max_minutes = st.session_state["minmax_minutes"]
min_age, max_age = st.session_state["minmax_age"]
min_strength, max_strength = st.session_state["minmax_strength"]

# ----------------- VALIDATION -----------------
missing = [c for c in REQUIRED_BASE if c not in df.columns]
if missing:
    st.error(f"Dataset missing required base columns: {missing}")
    st.stop()
missing_feats = [c for c in FEATURES if c not in df.columns]
if missing_feats:
    st.error(f"Dataset missing required feature columns: {missing_feats}")
    st.stop()

# -------- Position filter helper --------
def position_filter(pos):
    return str(pos).strip().upper().startswith(str(pos_text).upper())

# ----------------- FILTER POOL -----------------
df_f = df[df["League"].isin(leagues_sel)].copy()
df_f = df_f[df_f["Position"].astype(str).apply(position_filter)]
df_f = df_f[df_f["Minutes played"].between(min_minutes, max_minutes)]
df_f = df_f[df_f["Age"].between(min_age, max_age)]
df_f = df_f.dropna(subset=FEATURES)

df_f["Contract expires"] = pd.to_datetime(df_f["Contract expires"], errors="coerce")
if apply_contract:
    df_f = df_f[df_f["Contract expires"].dt.year <= cutoff_year]

df_f["League Strength"] = df_f["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
df_f = df_f[(df_f["League Strength"] >= float(min_strength)) & (df_f["League Strength"] <= float(max_strength))]
df_f = df_f[(df_f["Market value"] >= min_value) & (df_f["Market value"] <= max_value)]
if df_f.empty:
    st.warning("No players after filters. Loosen filters.")
    st.stop()

# ----------------- PERCENTILES (per league) -----------------
for c in FEATURES:
    df_f[c] = pd.to_numeric(df_f[c], errors="coerce")
df_f = df_f.dropna(subset=FEATURES)
for feat in FEATURES:
    df_f[f"{feat} Percentile"] = df_f.groupby("League")[feat].transform(lambda x: x.rank(pct=True) * 100.0)

# ----------------- ROLE SCORING -----------------
def compute_weighted_role_score(df_in: pd.DataFrame, metrics: dict, beta: float, league_weighting: bool) -> pd.Series:
    total_w = sum(metrics.values()) if metrics else 1.0
    wsum = np.zeros(len(df_in))
    for m, w in metrics.items():
        col = f"{m} Percentile"
        if col in df_in.columns:
            wsum += df_in[col].values * w
    player_score = wsum / total_w  # 0..100
    if league_weighting:
        league_scaled = (df_in["League Strength"].fillna(50) / 100.0) * 100.0
        return (1 - beta) * player_score + beta * league_scaled
    return player_score

for role_name, role_def in ROLES.items():
    df_f[f"{role_name} Score"] = compute_weighted_role_score(
        df_f, role_def["metrics"], beta=beta, league_weighting=use_league_weighting
    )

# ----------------- THRESHOLDS -----------------
if enable_min_perf and sel_metrics:
    keep_mask = np.ones(len(df_f), dtype=bool)
    for m in sel_metrics:
        pct_col = f"{m} Percentile"
        if pct_col in df_f.columns:
            keep_mask &= (df_f[pct_col] >= min_pct)
    df_f = df_f[keep_mask]
    if df_f.empty:
        st.warning("No players meet the minimum performance thresholds. Loosen thresholds.")
        st.stop()

# ----------------- TABLE HELPERS -----------------
def fmt_cols(df_in: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df_in.copy()
    out[score_col] = out[score_col].round(round_to).astype(int if round_to == 0 else float)
    cols = ["Player","Team","League","Position","Age","Contract expires","League Strength", score_col]
    return out[cols]

def top_table(df_in: pd.DataFrame, role: str, head_n: int) -> pd.DataFrame:
    col = f"{role} Score"
    ranked = df_in.dropna(subset=[col]).sort_values(col, ascending=False)
    ranked = fmt_cols(ranked, col).head(head_n).reset_index(drop=True)
    ranked.index = np.arange(1, len(ranked)+1)
    return ranked

def filtered_view(df_in: pd.DataFrame, *, age_max=None, contract_year=None, value_max=None):
    t = df_in.copy()
    if age_max is not None: t = t[t["Age"] <= age_max]
    if contract_year is not None: t = t[t["Contract expires"].dt.year <= contract_year]
    if value_max is not None: t = t[t["Market value"] <= value_max]
    return t

# ----------------- STRONG FOTMOB LOOKUP (surname + team) -----------------
FALLBACK_IMG = "https://i.redd.it/43axcjdu59nd1.jpeg"  # your requested fallback

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
def _canon(s: str) -> str:
    s = _strip_accents(s).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()
def _surname(full: str) -> str:
    s = str(full).strip()
    if "." in s:  # e.g., "C. Morris"
        s = s.split(".")[-1].strip()
    parts = s.split()
    return parts[-1] if parts else s

def _similar(a: str, b: str) -> float:
    ta, tb = set(_canon(a).split()), set(_canon(b).split())
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

HEADERS = {"User-Agent":"Mozilla/5.0 (compatible; scouting-app/1.0)"}

@st.cache_data(show_spinner=False, ttl=60*60*12)
def _http_json(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False, ttl=60*60*24)
def _url_exists(url: str) -> bool:
    try:
        h = requests.head(url, headers=HEADERS, timeout=6)
        return h.status_code == 200
    except Exception:
        return False

@st.cache_data(show_spinner=False, ttl=60*60*12)
def fotmob_headshot_by_surname_and_team(player_display: str, team_name: str) -> str | None:
    """
    1) Search team -> teamId -> pull roster -> match surname
    2) Else search players -> soft-match by team
    """
    try:
        target_team = _canon(team_name)
        target_surname = _canon(_surname(player_display))

        # -- Step 1: team search
        js = _http_json(f"https://www.fotmob.com/api/search?q={requests.utils.quote(team_name)}")
        if js and js.get("teams"):
            teams = js["teams"]
            teams_scored = sorted((( _similar(t.get("name",""), target_team), t ) for t in teams),
                                  key=lambda x: x[0], reverse=True)
            if teams_scored and teams_scored[0][0] >= 0.6:
                team_id = teams_scored[0][1].get("id")
                if team_id:
                    team_js = _http_json(f"https://www.fotmob.com/api/teams?id={team_id}")
                    if team_js:
                        # squads live under various keys; flatten
                        players = []
                        for key in ("players","squad","mainSquad","squadPlayers","u21","u23","u19","firstTeam"):
                            if key in team_js and isinstance(team_js[key], list):
                                players.extend(team_js[key])
                        # some structures: {'name':..,'id':..}, others nested under dict['players']
                        flat = []
                        for p in players:
                            if isinstance(p, dict) and "name" in p and "id" in p:
                                flat.append(p)
                            elif isinstance(p, dict) and "players" in p and isinstance(p["players"], list):
                                for q in p["players"]:
                                    if isinstance(q, dict) and "name" in q and "id" in q:
                                        flat.append(q)
                        # match by surname
                        for p in flat:
                            if _canon(_surname(p.get("name",""))) == target_surname:
                                pid = p.get("id")
                                url = f"https://images.fotmob.com/image_resources/playerimages/{pid}.png"
                                if _url_exists(url):
                                    return url
        # -- Step 2: player search (surname) then match by team
        js2 = _http_json(f"https://www.fotmob.com/api/search?q={requests.utils.quote(_surname(player_display))}")
        if js2 and js2.get("players"):
            cand = js2["players"]
            # score by team match first, then by exact surname
            scored = []
            for c in cand:
                team = c.get("team",{}).get("name","")
                s1 = _similar(team, target_team)
                s2 = 1.0 if _canon(_surname(c.get("name",""))) == target_surname else 0.0
                scored.append((s1 + 0.25*s2, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored:
                pid = scored[0][1].get("id")
                if pid:
                    url = f"https://images.fotmob.com/image_resources/playerimages/{pid}.png"
                    if _url_exists(url):
                        return url
    except Exception:
        pass
    return None

# ----------------- TABS -----------------
tabs = st.tabs(["Overall Top N","U23 Top N","Expiring Contracts","Value Band (≤ max €)","Tiles (Top 10 per role)"])
for role, role_def in ROLES.items():
    with tabs[0]:
        st.subheader(f"{role} — Overall Top {int(top_n)}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(df_f, role, int(top_n)), use_container_width=True)
        st.divider()
    with tabs[1]:
        u23_cutoff = st.number_input(f"{role} — U23 cutoff", min_value=16, max_value=30, value=23, step=1, key=f"u23_{role}")
        st.subheader(f"{role} — U{u23_cutoff} Top {int(top_n)}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, age_max=u23_cutoff), role, int(top_n)), use_container_width=True)
        st.divider()
    with tabs[2]:
        exp_year = st.number_input(f"{role} — Expiring by year", min_value=2024, max_value=2030, value=cutoff_year, step=1, key=f"exp_{role}")
        st.subheader(f"{role} — Contracts expiring ≤ {exp_year}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, contract_year=exp_year), role, int(top_n)), use_container_width=True)
        st.divider()
    with tabs[3]:
        v_max = st.number_input(f"{role} — Max value (€)", min_value=0, value=value_band_max, step=100_000, key=f"val_{role}")
        st.subheader(f"{role} — Value band ≤ €{v_max:,.0f}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, value_max=v_max), role, int(top_n)), use_container_width=True)
        st.divider()

# ----------------- GLOBAL CSS FOR DARK TILES -----------------
TILE_CSS = """
<style>
.card {
  background: #1f232b; border-radius: 18px; padding: 18px 20px; color: #eef1f5;
  display: flex; align-items: center; gap: 16px; position: relative;
  box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}
.card + .card { margin-top: 14px; }
.card .rank {
  position: absolute; right: 16px; top: 10px; color: #9aa3b2; font-size: 0.9rem;
}
.photo { width: 72px; height: 72px; border-radius: 12px; background: #2a2f39; overflow: hidden; flex-shrink: 0; }
.meta  { display:flex; flex-direction: column; gap: 6px; }
.name  { font-weight: 700; font-size: 1.05rem; line-height: 1.1; }
.team  { color: #cfd6e1; font-weight: 600; }
.sub   { color:#9aa3b2; font-size: 0.92rem; }
.badges{ display:flex; gap:10px; align-items:center; margin-top: 6px;}
.badgeNum { background:#1b6a36; color:#fff; padding:6px 10px; border-radius:10px; font-weight:700; }
.badgeLbl { background:#3b4151; color:#fff; padding:6px 10px; border-radius:10px; }
.photo img{ width:100%; height:100%; object-fit:cover; display:block; }
</style>
"""
st.markdown(TILE_CSS, unsafe_allow_html=True)

# ----------------- TILES VIEW -----------------
with tabs[4]:
    st.subheader("Player Tiles — Top 10 per role")
    st.caption("Headshots via FotMob by surname + team. Potential = Role Score.")

    for role, role_def in ROLES.items():
        score_col = f"{role} Score"
        top10 = df_f.dropna(subset=[score_col]).sort_values(score_col, ascending=False).head(10)
        if top10.empty:
            continue

        st.markdown(f"### {role}")
        for idx, (_, row) in enumerate(top10.iterrows(), start=1):
            rating = int(round(row[score_col]))
            potential = rating
            age = int(row["Age"]) if pd.notna(row["Age"]) else "—"
            league = row.get("League","—")
            pos = row.get("Position","—")
            team = row.get("Team","—")
            player = row["Player"]

            img = fotmob_headshot_by_surname_and_team(player, team) or FALLBACK_IMG

            card_html = f"""
            <div class="card">
              <div class="rank">#{idx}</div>
              <div class="photo"><img src="{img}" onerror="this.src='{FALLBACK_IMG}'"/></div>
              <div class="meta">
                <div class="name">{player}</div>
                <div class="team">{team}</div>
                <div class="sub">{league} · {pos} · {age}y</div>
                <div class="badges">
                  <div class="badgeNum">{rating}</div><div class="badgeLbl">Overall</div>
                  <div class="badgeNum">{potential}</div><div class="badgeLbl">Potential</div>
                </div>
              </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
        st.divider()



