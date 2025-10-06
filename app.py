# app.py — Striker Scouting + Compact Tiles (FotMob headshots, team badge, flags)
# Requirements: streamlit, pandas, numpy, requests, matplotlib (sklearn optional)

import io, os, re, math, json, unicodedata
from pathlib import Path
from typing import Optional, Tuple

import requests
import streamlit as st
import pandas as pd
import numpy as np

# ============= PAGE ============
st.set_page_config(page_title="Advanced Striker Scouting", layout="wide")
st.title("🔎 Advanced Striker Scouting System")
st.caption("Use the sidebar to shape your pool. Tables + compact Tiles with FotMob headshots.")

# ---------- CSS (cards/tiles, spacing, badges) ----------
st.markdown("""
<style>
/* narrow the main column a bit so cards don't look super wide */
section.main > div.block-container {
  max-width: 1100px; padding-top: 10px;
}
.tile { background:#1f232b; border-radius:18px; padding:16px 18px; margin:14px 0;
        box-shadow:0 6px 18px rgba(0,0,0,.18); border:1px solid rgba(255,255,255,.06); }
.tile-inner { display:flex; gap:16px; align-items:flex-start; }
.head-col { width:92px; flex:0 0 92px; display:flex; flex-direction:column; align-items:center; }
.headshot { width:92px; height:92px; border-radius:12px; background:#0f1116; object-fit:cover; display:block; }
.flag-age { margin-top:8px; color:#cbd5e1; font-size:.9rem; display:flex; gap:6px; align-items:center; }
.flag-age img { width:18px; height:14px; border-radius:2px; object-fit:cover; border:1px solid rgba(255,255,255,.15); }
.contract { margin-top:4px; color:#94a3b8; font-size:.9rem }

.body-col { flex:1 1 auto; min-width:0; }
.row-top { display:flex; justify-content:space-between; align-items:center; }
.player-name { font-weight:700; font-size:1.08rem; color:#e5e7eb; }
.rank { color:#94a3b8; font-size:.95rem; }

.teamline { margin-top:4px; display:flex; gap:8px; align-items:center; }
.teamline img { width:18px; height:18px; border-radius:50%; }
.teamline span { color:#a5b4fc; font-weight:700; }

.meta { margin-top:6px; color:#cbd5e1; font-size:.95rem; }

.badges { display:flex; gap:10px; margin-top:8px; }
.badge-num { background:#1f6e3a; color:#fff; padding:6px 8px; border-radius:8px; font-weight:700;
             min-width:34px; text-align:center; }
.badge-tag { background:#3b4151; color:#fff; padding:6px 8px; border-radius:8px; }

.poses { margin-top:8px; display:flex; gap:6px; flex-wrap:wrap; }
.pos-chip { background:#111827; color:#d1d5db; border:1px solid rgba(255,255,255,.06);
            padding:4px 6px; border-radius:8px; font-size:.85rem; }

hr.div { border:0; border-top:1px solid rgba(255,255,255,.06); margin:20px 0 10px; }
</style>
""", unsafe_allow_html=True)

# ============= CONFIG =============
INCLUDED_LEAGUES = [
    'England 1.', 'England 2.', 'England 3.', 'England 4.', 'England 5.',
    'England 6.', 'England 7.', 'England 8.', 'England 9.', 'England 10.',
    'Argentina 1.','Austria 1.','Belgium 1.','Brazil 1.','Chile 1.','Colombia 1.',
    'Croatia 1.','Czech 1.','Denmark 1.','France 1.','France 2.','Germany 1.',
    'Germany 2.','Italy 1.','Italy 2.','Netherlands 1.','Norway 1.','Poland 1.',
    'Portugal 1.','Scotland 1.','Spain 1.','Spain 2.','Sweden 1.','Switzerland 1.',
    'Turkey 1.','USA 1.'
]
PRESET_LEAGUES = {
    "Top 5 Europe": {'England 1.', 'France 1.', 'Germany 1.', 'Italy 1.', 'Spain 1.'},
    "Top 20 Europe": {
        'England 1.','Italy 1.','Spain 1.','Germany 1.','France 1.',
        'England 2.','Portugal 1.','Belgium 1.','Turkey 1.','Germany 2.','Spain 2.','France 2.',
        'Netherlands 1.','Austria 1.','Switzerland 1.','Denmark 1.','Croatia 1.','Italy 2.','Czech 1.','Norway 1.'
    },
    "EFL (England 2–4)": {'England 2.','England 3.','England 4.'}
}

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
    'Target Man CF': {
        'desc': "Aerial outlet, duel dominance, occupy CBs, crosses & second balls.",
        'metrics': {'Aerial duels per 90': 3, 'Aerial duels won, %': 4}
    },
    'Goal Threat CF': {
        'desc': "High shot & xG volume, box presence, SoT + finishing.",
        'metrics': {'Non-penalty goals per 90': 3,'Shots per 90': 1.5,'xG per 90': 3,
                    'Touches in box per 90': 1,'Shots on target, %': 0.5}
    },
    'Link-Up CF': {
        'desc': "Combine & create; progress; deliver to penalty area.",
        'metrics': {'Passes per 90': 2,'Passes to penalty area per 90': 1.5,'Deep completions per 90': 1,
                    'Smart passes per 90': 1.5,'Accurate passes, %': 1.5,'Key passes per 90': 1,
                    'Dribbles per 90': 2,'Successful dribbles, %': 1,'Progressive runs per 90': 2,'xA per 90': 3}
    },
    'All in': {
        'desc': "Blend of creation + scoring; balanced all-round attack.",
        'metrics': {'xA per 90': 2,'Dribbles per 90': 2,'xG per 90': 3,'Non-penalty goals per 90': 3}
    }
}

LEAGUE_STRENGTHS = {
    'England 1.':100.00,'Italy 1.':97.14,'Spain 1.':94.29,'Germany 1.':94.29,'France 1.':91.43,
    'Brazil 1.':82.86,'England 2.':71.43,'Portugal 1.':71.43,'Argentina 1.':71.43,
    'Belgium 1.':68.57,'Turkey 1.':65.71,'Germany 2.':65.71,'Spain 2.':65.71,'France 2.':65.71,
    'USA 1.':65.71,'Netherlands 1.':62.86,'Austria 1.':62.86,'Switzerland 1.':62.86,'Denmark 1.':62.86,
    'Croatia 1.':62.86,'Italy 2.':62.86,'Czech 1.':57.14,'Norway 1.':57.14,'Poland 1.':57.14,'Scotland 1.':58.00
}

REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Contract expires","Goals"}
FALLBACK_IMG = "https://i.redd.it/43axcjdu59nd1.jpeg"  # your fallback

# ============= DATA LOADER ============
@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

def load_df(csv_name: str = "WORLDJUNE25.csv") -> pd.DataFrame:
    candidates = [
        Path.cwd() / csv_name,
        Path(__file__).resolve().parent / csv_name,
        Path(__file__).resolve().parent.parent / csv_name,
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

# ============= SIDEBAR FILTERS ============
with st.sidebar:
    st.header("Filters")
    c1, c2, c3 = st.columns(3)
    use_top5  = c1.checkbox("Top-5 EU", value=False)
    use_top20 = c2.checkbox("Top-20 EU", value=False)
    use_efl   = c3.checkbox("EFL", value=False)

    seed = set()
    if use_top5:  seed |= PRESET_LEAGUES["Top 5 Europe"]
    if use_top20: seed |= PRESET_LEAGUES["Top 20 Europe"]
    if use_efl:   seed |= PRESET_LEAGUES["EFL (England 2–4)"]

    # leagues available from config + data
    leagues_in_data = []
    if "League" in df.columns:
        leagues_in_data = pd.Series(df["League"], dtype="object").dropna().unique().tolist()
    leagues_avail = sorted(set(INCLUDED_LEAGUES) | set(leagues_in_data))
    default_leagues = sorted(seed) if seed else INCLUDED_LEAGUES

    leagues_sel = st.multiselect("Leagues (add or prune the presets)", options=leagues_avail, default=default_leagues)

    # numeric coercions
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    min_minutes, max_minutes = st.slider("Minutes played", 0, 5000, (500, 5000))

    age_min_data = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_data = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    min_age, max_age = st.slider("Age", age_min_data, age_max_data, (16, 40))

    st.subheader("Position filter")
    st.caption("Players whose Position text starts with this (case-insensitive).")
    pos_text = st.text_input("Position startswith", "CF")

    apply_contract = st.checkbox("Filter by contract expiry", value=False)
    cutoff_year = st.slider("Max contract year (inclusive)", 2025, 2030, 2026)

    min_strength, max_strength = st.slider("League quality (strength)", 0, 101, (0, 101))
    use_league_weighting = st.checkbox("Use league weighting in role score", value=False)
    beta = st.slider("League weighting beta", 0.0, 1.0, 0.40, 0.05,
                     help="0 = ignore league strength; 1 = only league strength")

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
                                     value=min_value if min_value > 0 else 5_000_000, step=250_000)

    st.subheader("Minimum performance thresholds")
    enable_min_perf = st.checkbox("Require minimum percentile on selected metrics", value=False)
    sel_metrics = st.multiselect(
        "Metrics to threshold", FEATURES[:],
        default=(['Non-penalty goals per 90','xG per 90'] if enable_min_perf else [])
    )
    min_pct = st.slider("Minimum percentile (0–100)", 0, 100, 60)

    top_n = st.number_input("Top N per table", 5, 200, 50, 5)
    round_to = st.selectbox("Round output percentiles to", [0, 1], index=0)

# ============= VALIDATION ============
missing = [c for c in REQUIRED_BASE if c not in df.columns]
if missing:
    st.error(f"Dataset missing required base columns: {missing}")
    st.stop()
missing_feats = [c for c in FEATURES if c not in df.columns]
if missing_feats:
    st.error(f"Dataset missing required feature columns: {missing_feats}")
    st.stop()

# ============= FILTER POOL ============
def position_filter(pos):
    return str(pos).strip().upper().startswith(str(pos_text).upper())

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

# ============= PERCENTILES (per league) ============
for c in FEATURES:
    df_f[c] = pd.to_numeric(df_f[c], errors="coerce")
df_f = df_f.dropna(subset=FEATURES)
for feat in FEATURES:
    df_f[f"{feat} Percentile"] = df_f.groupby("League")[feat].transform(lambda x: x.rank(pct=True) * 100.0)

# ============= ROLE SCORES ============
def compute_weighted_role_score(df_in: pd.DataFrame, metrics: dict, beta: float, league_weighting: bool) -> pd.Series:
    total_w = max(1.0, sum(metrics.values()))
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

# ============= THRESHOLDS ============
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

# ============= HELPERS (tables) ============
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
    if contract_year is not None and "Contract expires" in t:
        ty = t["Contract expires"].dt.year
        t = t[ty <= contract_year]
    if value_max is not None: t = t[t["Market value"] <= value_max]
    return t

# ============= FOTMOB LOOKUPS (surname + team) =============
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
def _canon(s: str) -> str:
    s = _strip_accents(s).lower()
    return re.sub(r"[^a-z0-9]+", " ", str(s)).strip()

TEAM_ALIASES = {
    "man utd":"manchester united","man city":"manchester city","psg":"paris saint germain","inter":"internazionale",
    "bayern":"bayern munich","newcastle utd":"newcastle united","sheff wed":"sheffield wednesday",
    "sheff utd":"sheffield united"
}

@st.cache_data(show_spinner=False, ttl=60*60*24)
def fotmob_search(q: str) -> dict:
    try:
        r = requests.get("https://www.fotmob.com/api/search", params={"q": q}, timeout=6)
        if r.status_code != 200: return {}
        return r.json()
    except Exception:
        return {}

@st.cache_data(show_spinner=False, ttl=60*60*24)
def fotmob_player_image_by_surname_team(full_name: str, team_name: str) -> Optional[str]:
    """Search FotMob players; pick one whose surname AND team best match."""
    try:
        surname = _canon(full_name).split()[-1]  # last token
        js = fotmob_search(surname)
        players = js.get("players") or []
        if not players: return None
        team_target = _canon(TEAM_ALIASES.get(_canon(team_name), team_name))
        # score: +1 if surname token in name, plus Jaccard with team
        best = (0.0, None)
        for p in players:
            name = p.get("name","")
            team = p.get("team",{}).get("name","")
            score = 0.0
            if surname and surname in _canon(name).split():
                score += 0.7
            if team_target:
                a = set(_canon(team_target).split())
                b = set(_canon(team).split())
                if a and b: score += len(a & b) / len(a | b)
            if score > best[0]:
                best = (score, p)
        pick = best[1]
        if not pick: return None
        pid = pick.get("id")
        if not pid: return None
        url = f"https://images.fotmob.com/image_resources/playerimages/{pid}.png"
        try:
            hr = requests.head(url, timeout=6)
            if hr.status_code == 200: return url
        except Exception:
            pass
        return None
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=60*60*24)
def fotmob_team_badge(team_name: str) -> Optional[str]:
    try:
        q = TEAM_ALIASES.get(_canon(team_name), team_name)
        js = fotmob_search(q)
        teams = js.get("teams") or []
        if not teams: return None
        # pick team with best canonical name match
        target = _canon(team_name)
        best = (0.0, None)
        for t in teams:
            nm = _canon(t.get("name",""))
            if not nm: continue
            a, b = set(target.split()), set(nm.split())
            score = len(a & b) / len(a | b) if a and b else 0.0
            if score > best[0]:
                best = (score, t)
        pick = best[1]
        if not pick: return None
        tid = pick.get("id")
        if not tid: return None
        return f"https://images.fotmob.com/image_resources/logo/teamlogo/{tid}.png"
    except Exception:
        return None

# ============= FLAG HELPER (Birth country -> flagcdn) ============
COUNTRY_TO_ISO2 = {
    # quick coverage for common values; extend as needed
    "england":"gb", "scotland":"gb-sct", "wales":"gb-wls", "northern ireland":"gb-nir",
    "united kingdom":"gb", "great britain":"gb", "ireland":"ie",
    "france":"fr","germany":"de","italy":"it","spain":"es","portugal":"pt","netherlands":"nl",
    "belgium":"be","austria":"at","switzerland":"ch","denmark":"dk","norway":"no","sweden":"se",
    "poland":"pl","czech republic":"cz","czechia":"cz","croatia":"hr","serbia":"rs","turkey":"tr",
    "usa":"us","united states":"us","argentina":"ar","brazil":"br","colombia":"co","chile":"cl",
    "uruguay":"uy","mexico":"mx","japan":"jp","south korea":"kr","korea republic":"kr",
}

def flag_url_from_country(country: Optional[str]) -> Optional[str]:
    if not country or not isinstance(country, str): return None
    key = _canon(country)
    iso = COUNTRY_TO_ISO2.get(key)
    if not iso: return None
    # Use flagcdn SVG/PNG; PNG is safe in markdown <img>
    if iso.startswith("gb-"):
        return f"https://flagcdn.com/{iso[3:]}.png"  # regional will fall back poorly; simplest is omit
    return f"https://flagcdn.com/w20/{iso}.png"

# ============= TABS (tables + tiles) ============
tabs = st.tabs(["Overall Top N", "U23 Top N", "Expiring Contracts", "Value Band (≤ max €)", "Tiles (Top 10 per role)"])

for role, role_def in ROLES.items():
    with tabs[0]:
        st.subheader(f"{role} — Overall Top {int(top_n)}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(df_f, role, int(top_n)), use_container_width=True)
        st.markdown('<hr class="div">', unsafe_allow_html=True)

    with tabs[1]:
        u23_cutoff = st.number_input(f"{role} — U23 cutoff", min_value=16, max_value=30, value=23, step=1, key=f"u23_{role}")
        st.subheader(f"{role} — U{u23_cutoff} Top {int(top_n)}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, age_max=u23_cutoff), role, int(top_n)), use_container_width=True)
        st.markdown('<hr class="div">', unsafe_allow_html=True)

    with tabs[2]:
        exp_year = st.number_input(f"{role} — Expiring by year", min_value=2024, max_value=2030, value=cutoff_year, step=1, key=f"exp_{role}")
        st.subheader(f"{role} — Contracts expiring ≤ {exp_year}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, contract_year=exp_year), role, int(top_n)), use_container_width=True)
        st.markdown('<hr class="div">', unsafe_allow_html=True)

    with tabs[3]:
        v_max = st.number_input(f"{role} — Max value (€)", min_value=0, value=value_band_max, step=100_000, key=f"val_{role}")
        st.subheader(f"{role} — Value band ≤ €{v_max:,.0f}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, value_max=v_max), role, int(top_n)), use_container_width=True)
        st.markdown('<hr class="div">', unsafe_allow_html=True)

# ============= TILES VIEW ============
with tabs[4]:
    st.subheader("Player Tiles — Top 10 per role")
    st.caption("Headshots via FotMob by **surname + team**. Potential = Role Score.")

    for role, role_def in ROLES.items():
        score_col = f"{role} Score"
        top10 = df_f.dropna(subset=[score_col]).sort_values(score_col, ascending=False).head(10)
        if top10.empty: 
            continue

        st.markdown(f"### {role}")

        for i, (_, r) in enumerate(top10.iterrows(), start=1):
            name = str(r["Player"])
            team = str(r["Team"])
            league = str(r["League"])
            pos = str(r["Position"])
            age = int(r["Age"]) if pd.notna(r["Age"]) else "—"
            contract_year = int(r["Contract expires"].year) if pd.notna(r["Contract expires"]) else "—"
            rating = int(round(r[score_col]))
            potential = rating  # per requirement
            birth_ctry = r.get("Birth country") if "Birth country" in r else None

            # lookups
            img_url = fotmob_player_image_by_surname_team(name, team) or FALLBACK_IMG
            badge_url = fotmob_team_badge(team)
            flag_url = flag_url_from_country(birth_ctry)

            # positions chips (first 3 tokens)
            pos_tokens = [t.strip() for t in pos.split(",") if t.strip()]
            pos_html = "".join(f"<span class='pos-chip'>{p}</span>" for p in pos_tokens[:3])

            # HTML card
            html = f"""
            <div class="tile">
              <div class="tile-inner">
                <div class="head-col">
                  <img class="headshot" src="{img_url}" alt="player">
                  <div class="flag-age">
                    {f'<img src="{flag_url}" alt="flag">' if flag_url else ''}
                    <span>{age}y.</span>
                  </div>
                  <div class="contract">Contract {contract_year}</div>
                </div>
                <div class="body-col">
                  <div class="row-top">
                    <div class="player-name">{name}</div>
                    <div class="rank">#{i}</div>
                  </div>
                  <div class="teamline">
                    {f'<img src="{badge_url}" alt="badge">' if badge_url else ''}
                    <span>{team}</span>
                  </div>
                  <div class="meta">{league} · {pos}</div>
                  <div class="badges">
                    <div class="badge-num">{rating}</div>
                    <div class="badge-tag">Overall</div>
                    <div class="badge-num">{potential}</div>
                    <div class="badge-tag">Potential</div>
                  </div>
                  <div class="poses">{pos_html}</div>
                </div>
              </div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
        st.markdown('<hr class="div">', unsafe_allow_html=True)








