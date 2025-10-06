# app_top20_tiles.py — Top 20 Tiles (CF-only) with club badge + country flag
# Requirements: streamlit, pandas, numpy, requests, beautifulsoup4

import io, re, math, unicodedata
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
    "Goal Threat = shooting/xG profile. Link-Up CF = combining/creating profile. "
    "Ranking uses the All-in blend. Club badges via Wikipedia; flags from 'Birth country'."
)

# ----------------- STYLE -----------------
st.markdown("""
<style>
:root { --bg: #0f1115; --card: #161a22; --muted:#a8b3cf; --soft:#202633; }
.block-container{padding-top:0.8rem;} body{background:var(--bg);}
.wrap{display:flex; justify-content:center;}
.player-card{
  width:min(420px,96%); display:grid; grid-template-columns:96px 1fr 48px;
  gap:12px; align-items:center; background:var(--card); border:1px solid #252b3a;
  border-radius:18px; padding:16px;
}
.avatar{ width:96px; height:96px; border-radius:12px; background:#0b0d12 url('https://i.redd.it/43axcjdu59nd1.jpeg') center/cover no-repeat; border:1px solid #2a3145; }
.leftcol{ display:flex; flex-direction:column; align-items:center; gap:8px; }
.name{ font-weight:800; font-size:22px; color:#e8ecff; margin-bottom:6px; }
.sub{ color:var(--muted); font-size:15px; }
.row{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:4px 0; }
.pill{ padding:2px 10px; border-radius:9px; font-weight:800; font-size:18px; color:#0b0d12; min-width:42px; text-align:center; }
.rank{ color:#94a0c6; font-weight:800; font-size:18px; text-align:right; }
.chip{ background:var(--soft); color:#cbd5f5; border:1px solid #2d3550; padding:3px 10px; border-radius:9px; font-size:13px; }
.teamline{ color:#e6ebff; font-size:15px; font-weight:400; margin-top:2px; display:flex; align-items:center; gap:8px; }
.badge{ width:20px; height:20px; border-radius:4px; border:1px solid #2a3145; background:#0b0d12; object-fit:contain; }
.flag{ width:18px; height:12px; border-radius:2px; border:1px solid #2a3145; object-fit:cover; }
.divider{height:12px;}

/* position color system */
.pos-CF{ background:#0f2b6a; color:#cfe3ff; border:1px solid #1e3a8a; }           /* dark blue */
.pos-wide-blue{ background:#143d8a; color:#d7e6ff; border:1px solid #2956a3; }     /* blue */
.pos-amf-light{ background:#1f4d2a; color:#d8ffd8; border:1px solid #2d6a3a; }     /* light green */
.pos-cmf-green{ background:#174c36; color:#d8ffe6; border:1px solid #226b4c; }     /* green */
.pos-dmf-dark{ background:#0f3f2a; color:#c9ffe5; border:1px solid #1c5a41; }      /* dark green */
.pos-wingback{ background:#5a4a00; color:#fff3bd; border:1px solid #7a6400; }      /* yellow */
.pos-fullback{ background:#6a3a00; color:#ffe0c2; border:1px solid #8a4a05; }      /* orange */
.pos-cb{ background:#5a2900; color:#ffd8bf; border:1px solid #7a3a10; }            /* dark orange */
</style>
""", unsafe_allow_html=True)

# ----------------- CONFIG -----------------
UA = {"User-Agent": "tiles/1.0 (+streamlit)"}
REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Contract expires","Goals","Birth country"}

# Role definitions (as requested)
ROLES = {
    "Goal Threat CF": {
        "desc": "High shot & xG volume, box presence, consistent SoT and finishing.",
        "metrics": {
            "Non-penalty goals per 90": 3, "Shots per 90": 1.5, "xG per 90": 3,
            "Touches in box per 90": 1, "Shots on target, %": 0.5
        },
    },
    "Link-Up CF": {
        "desc": "Combine & create; link play; progress & deliver to the penalty area.",
        "metrics": {
            "Passes per 90": 2, "Passes to penalty area per 90": 1.5,
            "Deep completions per 90": 1, "Smart passes per 90": 1.5,
            "Accurate passes, %": 1.5, "Key passes per 90": 1,
            "Dribbles per 90": 2, "Successful dribbles, %": 1,
            "Progressive runs per 90": 2, "xA per 90": 3,
        },
    },
    "All in": {
        "desc": "Blend of creation + scoring; balanced all-round attacking profile.",
        "metrics": { "xA per 90": 2, "Dribbles per 90": 2, "xG per 90": 3, "Non-penalty goals per 90": 3 },
    },
}

# League strengths (unchanged)
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

# ----------------- HELPERS -----------------
def _norm(s:str)->str:
    s=(s or "").strip()
    s=unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    return s

def _slug(s:str)->str:
    return re.sub(r"[^a-z0-9]+","-", _norm(s).lower()).strip("-")

@st.cache_data(show_spinner=False, ttl=60*60*24)
def wiki_club_badge(team:str)->str|None:
    """Best-effort: use Wikipedia pageimages thumbnail as crest."""
    if not team: return None
    candidates=[team, f"{team} F.C.", f"{team} C.F.", f"{team} football club"]
    for title in candidates:
        try:
            r=requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action":"query","format":"json","prop":"pageimages",
                        "piprop":"thumbnail","pithumbsize":48,"titles":title},
                headers=UA, timeout=7)
            if not r.ok: continue
            pages=r.json().get("query",{}).get("pages",{})
            for _,p in pages.items():
                thumb=(p.get("thumbnail") or {}).get("source")
                if thumb: return thumb
        except Exception:
            continue
    return None

def flag_url(country:str)->str|None:
    if not country: return None
    # Country name works on this simple API
    return f"https://countryflagsapi.com/png/{_slug(country).replace('-', '%20')}"

# SoFIFA-like color ramp for numeric pills
PALETTE=[(0,(208,2,27)),(50,(245,166,35)),(65,(248,231,28)),(75,(126,211,33)),(85,(65,117,5)),(100,(40,90,4))]
def _lerp(a,b,t): return tuple(int(round(a[i]+(b[i]-a[i])*t)) for i in range(3))
def rating_color(v:float)->str:
    v=max(0.0, min(100.0, float(v)))
    for i in range(len(PALETTE)-1):
        x0,c0=PALETTE[i]; x1,c1=PALETTE[i+1]
        if v<=x1:
            t=0 if x1==x0 else (v-x0)/(x1-x0); r,g,b=_lerp(c0,c1,t); return f"rgb({r},{g},{b})"
    r,g,b=PALETTE[-1][1]; return f"rgb({r},{g},{b})"

# Position chip coloring
POS_CLASS_MAP={
    "CF":"pos-CF",
    # Blue wide/AMF wing roles
    "LWF":"pos-wide-blue","LW":"pos-wide-blue","LAMF":"pos-wide-blue",
    "RW":"pos-wide-blue","RWF":"pos-wide-blue","RAMF":"pos-wide-blue",
    # AMF + CMF greens
    "AMF":"pos-amf-light","LCMF":"pos-cmf-green","RCMF":"pos-cmf-green",
    # DMF dark green
    "RDMF":"pos-dmf-dark","LDMF":"pos-dmf-dark","DMF":"pos-dmf-dark",
    # Wingbacks yellow
    "LWB":"pos-wingback","RWB":"pos-wingback",
    # Fullbacks orange
    "LB":"pos-fullback","RB":"pos-fullback",
    # Centre-backs dark orange
    "RCB":"pos-cb","CB":"pos-cb","LCB":"pos-cb",
}

def pos_chips_html(position:str)->str:
    tags=[p.strip().upper() for p in (position or "").split(",")]
    chips=[]
    for p in tags:
        klass=POS_CLASS_MAP.get(p)
        if klass:
            chips.append(f"<span class='chip {klass}'>{p}</span>")
    # Always show CF chip first if present
    chips.sort(key=lambda x: (0 if "CF" in x else 1))
    return " ".join(chips)

# ----------------- DATA LOADING -----------------
@st.cache_data(show_spinner=False)
def _read_csv_from_path(p:str)->pd.DataFrame: return pd.read_csv(p)
@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(b:bytes)->pd.DataFrame: return pd.read_csv(io.BytesIO(b))

def load_df(csv_name="WORLDJUNE25.csv")->pd.DataFrame:
    candidates=[Path.cwd()/csv_name]
    try:
        here=Path(__file__).resolve()
        candidates+=[here.parent.parent/csv_name, here.parent/csv_name]
    except NameError:
        pass
    for p in candidates:
        if p.exists(): return _read_csv_from_path(str(p))
    st.warning(f"Could not find **{csv_name}**. Upload below.")
    up=st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if up is None: st.stop()
    return _read_csv_from_bytes(up.getvalue())

df=load_df()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("Filters")
    leagues_avail=sorted(df.get("League", pd.Series([])).dropna().unique().tolist())
    leagues_sel=st.multiselect("Leagues", leagues_avail, default=leagues_avail)

    df["Minutes played"]=pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"]=pd.to_numeric(df["Age"], errors="coerce")

    min_minutes, max_minutes=st.slider("Minutes played", 0, 5000, (500, 5000))
    age_min=int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max=int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    amin, amax = st.slider("Age", age_min, age_max, (16, 40))

    df["Contract expires"]=pd.to_datetime(df["Contract expires"], errors="coerce")
    use_contract=st.checkbox("Filter by contract expiry", value=False)
    cutoff=st.slider("Max contract year (inclusive)", 2025, 2030, 2026)

    df["Market value"]=pd.to_numeric(df["Market value"], errors="coerce")
    mv_max=int(np.nanmax(df["Market value"])) if df["Market value"].notna().any() else 50_000_000
    mv_cap=int(math.ceil(mv_max/5_000_000)*5_000_000)
    st.markdown("**Market value (€)**")
    in_m=st.checkbox("Adjust in millions", True)
    if in_m:
        up_lim=int(mv_cap/1_000_000)
        lo, hi = st.slider("Range (M€)", 0, up_lim, (0, up_lim))
        mv_lo, mv_hi = lo*1_000_000, hi*1_000_000
    else:
        mv_lo, mv_hi = st.slider("Range (€)", 0, mv_cap, (0, mv_cap), step=100_000)

    min_strength, max_strength=st.slider("League quality (strength)", 0, 101, (0,101))
    beta=st.slider("League weighting beta (for All-in)", 0.0, 1.0, 0.40, 0.05)
    top_n=st.number_input("How many tiles", 5, 100, 20, 5)

# ----------------- VALIDATION -----------------
missing=[c for c in REQUIRED_BASE if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

# ----------------- FILTER POOL -----------------
df_f=df[df["League"].isin(leagues_sel)].copy()
df_f=df_f[df_f["Position"].astype(str).str.upper().str.startswith("CF")]   # CF only gate
df_f=df_f[df_f["Minutes played"].between(min_minutes, max_minutes)]
df_f=df_f[df_f["Age"].between(amin, amax)]
if use_contract:
    df_f=df_f[df_f["Contract expires"].dt.year <= cutoff]
df_f=df_f[(df_f["Market value"] >= mv_lo) & (df_f["Market value"] <= mv_hi)]

# Feature list = union of all role metrics
FEATURES=sorted({m for role in ROLES.values() for m in role["metrics"].keys()})
for c in FEATURES:
    df_f[c]=pd.to_numeric(df_f[c], errors="coerce")
df_f=df_f.dropna(subset=FEATURES)
if df_f.empty:
    st.warning("No players after filters. Loosen filters.")
    st.stop()

# Strength & percentiles
df_f["League Strength"]=df_f["League"].map(LEAGUE_STRENGTHS).fillna(50.0)
for feat in FEATURES:
    df_f[f"{feat} %"]=df_f.groupby("League")[feat].transform(lambda x: x.rank(pct=True)*100.0)

def role_score(df_in:pd.DataFrame, role_key:str)->pd.Series:
    metrics=ROLES[role_key]["metrics"]
    tot=sum(metrics.values()) or 1.0
    s=np.zeros(len(df_in))
    for m,w in metrics.items():
        col=f"{m} %"
        if col in df_in.columns: s += df_in[col].values * w
    return s / tot

# Visible scores
df_f["Goal Threat"]=role_score(df_f, "Goal Threat CF")
df_f["Link-Up CF"]=role_score(df_f, "Link-Up CF")

# Ranking score (All-in + league strength beta)
df_f["All in"]=role_score(df_f, "All in")
df_f["All in (weighted)"] = (1-beta)*df_f["All in"] + beta*df_f["League Strength"]

# Contract year for tile
df_f["Contract Year"]=pd.to_datetime(df_f["Contract expires"], errors="coerce").dt.year.fillna(0).astype(int)

# Top N
ranked=df_f.sort_values("All in (weighted)", ascending=False).head(int(top_n)).reset_index(drop=True)

# ----------------- RENDER -----------------
for idx,row in ranked.iterrows():
    rank=idx+1
    player=str(row["Player"])
    team=str(row["Team"])
    pos=str(row["Position"])
    age=int(row["Age"]) if not pd.isna(row["Age"]) else 0
    gt=int(round(float(row["Goal Threat"])))
    lu=int(round(float(row["Link-Up CF"])))
    contract=int(row["Contract Year"])
    country=str(row.get("Birth country",""))

    # badge + flag
    badge=wiki_club_badge(team) or ""
    flag=flag_url(country) or ""

    gt_style=f"background:{rating_color(gt)};"
    lu_style=f"background:{rating_color(lu)};"

    st.markdown(f"""
    <div class='wrap'>
      <div class='player-card'>
        <div class='leftcol'>
          <div class='avatar'></div>
          <div class='row'>
            {'<img class="flag" src="'+flag+'" />' if flag else ''}
            <span class='chip'>{age}y.o.</span>
            <span class='chip'>{contract if contract>0 else '—'}</span>
          </div>
        </div>
        <div>
          <div class='name'>{player}</div>
          <div class='row'>
            <span class='pill' style='{gt_style}'>{gt}</span>
            <span class='sub'>Goal Threat</span>
          </div>
          <div class='row'>
            <span class='pill' style='{lu_style}'>{lu}</span>
            <span class='sub'>Link-Up CF</span>
          </div>
          <div class='row'>
            {pos_chips_html(pos) or "<span class='chip pos-CF'>CF</span>"}
          </div>
          <div class='teamline'>
            {('<img class="badge" src="'+badge+'" />') if badge else ""}
            {team}
          </div>
        </div>
        <div class='rank'>#{rank}</div>
      </div>
    </div>
    <div class='divider'></div>
    """, unsafe_allow_html=True)






















