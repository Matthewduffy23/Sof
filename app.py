# -*- coding: utf-8 -*-
# CF-only tiles: fixed avatar, FotMob team badges, dataset birth-country -> flag,
# Goal Threat & Link-Up scores shown, sorted by "All in" (hidden score)

import io, re, math, base64, unicodedata
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
    "CF-only. Left pill = Goal Threat CF, right pill = Link-Up CF. "
    "Ranked by All-in (not displayed). Team badges via FotMob. Flags from dataset Birth country."
)

# ----------------- STYLE -----------------
st.markdown(
    """
    <style>
      :root { --bg:#0f1115; --card:#161a22; --muted:#a8b3cf; --soft:#202633; }
      .block-container { padding-top: 0.8rem; }
      body { background-color: var(--bg); }
      .wrap { display:flex; justify-content:center; }
      .player-card {
        width:min(540px, 96%);
        display:grid; grid-template-columns: 110px 1fr 48px; gap:14px; align-items:center;
        background:var(--card); border:1px solid #252b3a; border-radius:18px; padding:16px;
      }
      .avatar {
        width:110px; height:110px; border-radius:14px;
        background:#0b0d12 url('https://i.redd.it/43axcjdu59nd1.jpeg') center/cover no-repeat;
        border:1px solid #2a3145;
      }
      .icon { width:18px; height:18px; border-radius:3px; vertical-align:middle; margin-right:6px; }
      .leftcol { display:flex; flex-direction:column; align-items:center; gap:6px; }
      .name { font-weight:800; font-size:22px; color:#e8ecff; margin-bottom:6px; }
      .sub { color:#a8b3cf; font-size:15px; }
      .pill { padding:2px 10px; border-radius:9px; font-weight:800; font-size:18px; color:#0b0d12; display:inline-block; min-width:42px; text-align:center; }
      .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:4px 0; }
      .chip { background:var(--soft); color:#cbd5f5; border:1px solid #2d3550; padding:3px 10px; border-radius:9px; font-size:13px; }
      /* position colors */
      .chip.cf      { background:#0b2777; color:#e6f0ff; border-color:#081b55; }  /* dark blue */
      .chip.blue    { background:#1d4ed8; color:#e6f0ff; border-color:#1e3a8a; }  /* blue */
      .chip.lgreen  { background:#2e7d32; color:#eafff0; border-color:#1f5a23; }  /* light green */
      .chip.green   { background:#0f5132; color:#d1fae5; border-color:#0a3d25; }  /* green */
      .chip.dgreen  { background:#084c38; color:#c7fff2; border-color:#063729; }  /* dark green */
      .chip.yellow  { background:#806600; color:#fff7cc; border-color:#5c4800; }  /* yellow */
      .chip.orange  { background:#7a3a00; color:#ffe9d6; border-color:#5a2b00; }  /* orange */
      .chip.dorange { background:#632000; color:#ffd9c7; border-color:#4a1800; }  /* dark orange */

      .teamline { color:#e6ebff; font-size:15px; font-weight:500; margin-top:2px; display:flex; align-items:center; gap:8px; }
      .rank { color:#94a0c6; font-weight:800; font-size:18px; text-align:right; }
      .divider { height:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- CONFIG -----------------
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/128.0.0.0 Safari/537.36")}

REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Contract expires","Goals","Birth country"}

def _norm(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower()

TEAM_STRIP = re.compile(r"\b(fc|cf|sc|afc|ud|ac|bc|cd|sd|deportivo|club|city|united|town|athletic|sporting|sv|fk|sk|ik|if)\b", re.I)
def _simplify_team(t: str) -> str:
    t = _norm(t); t = TEAM_STRIP.sub("", t); t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# ----------------- DATA LOADING -----------------
@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

def load_df(csv_name: str = "WORLDJUNE25.csv") -> pd.DataFrame:
    p = Path.cwd()/csv_name
    if p.exists():
        return _read_csv_from_path(str(p))
    st.warning(f"Could not find **{csv_name}**. Upload below.")
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if up is None: st.stop()
    return _read_csv_from_bytes(up.getvalue())

df = load_df()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("Filters")
    leagues_avail = sorted(df.get("League", pd.Series([], dtype=str)).dropna().unique())
    leagues_sel = st.multiselect("Leagues", leagues_avail, default=leagues_avail)

    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    min_minutes, max_minutes = st.slider("Minutes played", 0, 5000, (500, 5000))
    age_min_data = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_data = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    min_age, max_age = st.slider("Age", age_min_data, age_max_data, (16, 40))

    df["Contract expires"] = pd.to_datetime(df["Contract expires"], errors="coerce")
    apply_contract = st.checkbox("Filter by contract expiry", value=False)
    cutoff_year = st.slider("Max contract year (inclusive)", 2025, 2030, 2026)

    top_n = st.number_input("How many tiles", 5, 100, 20, 5)

# ----------------- VALIDATION -----------------
missing = [c for c in REQUIRED_BASE if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

# ----------------- FILTER POOL (CF-only) -----------------
df_f = df[df["League"].isin(leagues_sel)].copy()
df_f = df_f[df_f["Minutes played"].between(min_minutes, max_minutes)]
df_f = df_f[df_f["Age"].between(min_age, max_age)]
if apply_contract:
    df_f = df_f[df_f["Contract expires"].dt.year <= cutoff_year]

# CF-only
df_f = df_f[df_f["Position"].astype(str).str.upper().str.strip().str.startswith("CF")]
if df_f.empty:
    st.warning("No CF players after filters.")
    st.stop()

# ----------------- FEATURES & SCORES -----------------
# Your requested role definitions
ROLES = {
    'Goal Threat CF': {
        'desc': "High shot & xG volume, box presence, consistent SoT and finishing.",
        'metrics': {
            'Non-penalty goals per 90': 3,
            'Shots per 90': 1.5,
            'xG per 90': 3,
            'Touches in box per 90': 1,
            'Shots on target, %': 0.5
        }
    },
    'Link-Up CF': {
        'desc': "Combine & create; link play; progress & deliver to the penalty area.",
        'metrics': {
            'Passes per 90': 2,
            'Passes to penalty area per 90': 1.5,
            'Deep completions per 90': 1,
            'Smart passes per 90': 1.5,
            'Accurate passes, %': 1.5,
            'Key passes per 90': 1,
            'Dribbles per 90': 2,
            'Successful dribbles, %': 1,
            'Progressive runs per 90': 2,
            'xA per 90': 3
        }
    },
    'All in': {
        'desc': "Blend of creation + scoring; balanced all-round attacking profile.",
        'metrics': {
            'xA per 90': 2,
            'Dribbles per 90': 2,
            'xG per 90': 3,
            'Non-penalty goals per 90': 3
        }
    }
}

# Build feature list and percentiles per league
FEATURES = sorted({m for role in ROLES.values() for m in role['metrics'].keys()})
for c in FEATURES:
    df_f[c] = pd.to_numeric(df_f[c], errors="coerce")
df_f = df_f.dropna(subset=FEATURES)

for feat in FEATURES:
    df_f[f"{feat} %"] = df_f.groupby("League")[feat].transform(lambda x: x.rank(pct=True) * 100.0)

def role_score(df_in: pd.DataFrame, metrics: dict) -> pd.Series:
    total_w = float(sum(metrics.values())) or 1.0
    s = np.zeros(len(df_in))
    for m, w in metrics.items():
        col = f"{m} %"
        if col in df_in.columns:
            s += df_in[col].values * w
    return s / total_w

df_f["Goal Threat"] = role_score(df_f, ROLES['Goal Threat CF']['metrics'])
df_f["Link-Up"]     = role_score(df_f, ROLES['Link-Up CF']['metrics'])
df_f["AllIn"]       = role_score(df_f, ROLES['All in']['metrics'])

df_f["Contract Year"] = pd.to_datetime(df_f["Contract expires"], errors="coerce").dt.year.fillna(0).astype(int)

# ----------------- FOTMOB TEAM BADGE -----------------
def _score_team(q_simpl: str, name_simpl: str) -> float:
    ta, tb = set(q_simpl.split()), set(name_simpl.split())
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

@st.cache_data(show_spinner=False, ttl=60*60*24)
def fotmob_team_badge(team_name: str) -> str | None:
    try:
        r = requests.get("https://www.fotmob.com/api/search", params={"q": team_name}, headers=UA, timeout=6)
        if not r.ok: return None
        data = r.json()
        best_id, best_s = None, -1.0
        qsim = _simplify_team(team_name)
        for t in data.get("teams", []):
            name = t.get("name","")
            s = _score_team(qsim, _simplify_team(name))
            if s > best_s:
                best_s, best_id = s, t.get("id")
        if best_id:
            return f"https://images.fotmob.com/image_resources/logo/teamlogo/{best_id}.png"
    except Exception:
        return None
    return None

def _fetch_image_as_data_uri(url: str, timeout=8) -> str | None:
    if not url: return None
    try:
        r = requests.get(url, headers=UA, timeout=timeout, stream=True)
        if not r.ok: return None
        ctype = r.headers.get("Content-Type","").lower()
        if not ctype.startswith("image/"):
            if url.endswith(".png"): ctype="image/png"
            elif url.endswith(".jpg") or url.endswith(".jpeg"): ctype="image/jpeg"
            elif url.endswith(".webp"): ctype="image/webp"
            else: return None
        b64 = base64.b64encode(r.content).decode("ascii")
        return f"data:{ctype};base64,{b64}"
    except Exception:
        return None

# ----------------- BIRTH COUNTRY -> FLAG -----------------
@st.cache_data(show_spinner=False, ttl=60*60*24)
def country_to_iso2(name: str) -> str | None:
    """Use RestCountries to convert country name to ISO2 (cca2)."""
    if not name: return None
    try:
        r = requests.get(f"https://restcountries.com/v3.1/name/{name}", params={"fullText":"false","fields":"cca2"}, timeout=8, headers=UA)
        if not r.ok: return None
        js = r.json()
        if isinstance(js, list) and js:
            return (js[0].get("cca2") or "").lower() or None
    except Exception:
        return None
    return None

def flag_from_birth_country(country_name: str) -> str | None:
    code = country_to_iso2(country_name)
    if not code: return None
    return f"https://flagcdn.com/w40/{code}.png"

# ----------------- COLORS FOR PILLS -----------------
PALETTE = [(0,(208,2,27)),(50,(245,166,35)),(65,(248,231,28)),(75,(126,211,33)),(85,(65,117,5)),(100,(40,90,4))]
def _lerp(a, b, t): return tuple(int(round(a[i] + (b[i]-a[i]) * t)) for i in range(3))
def rating_color(v: float) -> str:
    v = max(0.0, min(100.0, float(v)))
    for i in range(len(PALETTE)-1):
        x0,c0=PALETTE[i]; x1,c1=PALETTE[i+1]
        if v <= x1:
            t = 0 if x1==x0 else (v-x0)/(x1-x0); r,g,b = _lerp(c0,c1,t); return f"rgb({r},{g},{b})"
    r,g,b = PALETTE[-1][1]; return f"rgb({r},{g},{b})"

POS_CLASS = {
    "CF":"cf",
    "LWF":"blue","LW":"blue","LAMF":"blue","RW":"blue","RWF":"blue","RAMF":"blue",
    "AMF":"lgreen",
    "LCMF":"green","RCMF":"green",
    "RDMF":"dgreen","LDMF":"dgreen",
    "LWB":"yellow","RWB":"yellow",
    "LB":"orange","RB":"orange",
    "RCB":"dorange","CB":"dorange","LCB":"dorange",
}
def render_pos_chips(pos_text: str) -> str:
    parts = [p.strip().upper() for p in re.split(r"[,/]", pos_text or "") if p.strip()]
    html = []
    for p in parts:
        cls = POS_CLASS.get(p, "")
        html.append(f"<span class='chip {cls}'>{p}</span>")
    return " ".join(html) if html else "<span class='chip'>—</span>"

# ----------------- RANK & RENDER -----------------
ranked = df_f.sort_values("AllIn", ascending=False).head(int(top_n)).copy().reset_index(drop=True)

for idx, row in ranked.iterrows():
    rank = idx + 1
    player = str(row.get("Player","")) or ""
    team   = str(row.get("Team","")) or ""
    pos    = str(row.get("Position","")) or ""
    age    = int(row.get("Age",0)) if not pd.isna(row.get("Age",np.nan)) else 0
    gth    = int(round(float(row["Goal Threat"])))
    link   = int(round(float(row["Link-Up"])))
    cy     = int(row.get("Contract Year",0))
    birthc = str(row.get("Birth country","")) or ""

    # badge & flag
    badge_url = fotmob_team_badge(team)
    badge_img = f"<img class='icon' src='{_fetch_image_as_data_uri(badge_url) or badge_url}'/>" if badge_url else ""
    flag_url  = flag_from_birth_country(birthc)
    flag_img  = f"<img class='icon' src='{_fetch_image_as_data_uri(flag_url) or flag_url}'/>" if flag_url else ""

    # styles
    gth_style = f"background:{rating_color(gth)};"
    link_style = f"background:{rating_color(link)};"
    pos_html = render_pos_chips(pos)

    st.markdown(
        f"""
        <div class='wrap'>
          <div class='player-card'>
            <div class='leftcol'>
              <div class='avatar'></div>
              <div class='row'>
                {flag_img}<span class='chip'>{age}y.o.</span><span class='chip'>{(cy if cy>0 else "—")}</span>
              </div>
            </div>
            <div>
              <div class='name'>{player}</div>
              <div class='row'>
                <span class='pill' style='{gth_style}'>{gth}</span>
                <span class='sub'>Goal Threat</span>
              </div>
              <div class='row'>
                <span class='pill' style='{link_style}'>{link}</span>
                <span class='sub'>Link-Up CF</span>
              </div>
              <div class='row'>{pos_html}</div>
              <div class='teamline'>{badge_img}<span>{team}</span></div>
            </div>
            <div class='rank'>#{rank}</div>
          </div>
        </div>
        <div class='divider'></div>
        """,
        unsafe_allow_html=True,
    )





















