# app_top20_tiles.py — CF tiles with Wikipedia badges & flags, presets, β=0.40 on displayed scores

import io, re, math, unicodedata
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
import numpy as np
from textwrap import dedent

# ----------------- PAGE -----------------
st.set_page_config(page_title="Advanced Striker Scouting — Top 20 Tiles", layout="wide")
st.title("🔎 Advanced Striker Scouting — Top 20 Tiles")
st.caption(
    "Goal Threat & Link-Up CF are league-weighted (β=0.40). Ranking uses All-in (blend) with your sidebar β. "
    "Club badges & country flags are fetched from Wikipedia."
)

# ----------------- STYLE -----------------
st.markdown(dedent("""
<style>
:root { --bg:#0f1115; --card:#161a22; --muted:#a8b3cf; --soft:#202633; }
.block-container{padding-top:0.8rem;} body{background:var(--bg);}
.wrap{display:flex;justify-content:center;}
.player-card{
  width:min(420px,96%);display:grid;grid-template-columns:96px 1fr 48px;gap:12px;align-items:center;
  background:var(--card);border:1px solid #252b3a;border-radius:18px;padding:16px;
}
.avatar{width:96px;height:96px;border-radius:12px;background:#0b0d12 url('https://i.redd.it/43axcjdu59nd1.jpeg') center/cover no-repeat;border:1px solid #2a3145;}
.leftcol{display:flex;flex-direction:column;align-items:center;gap:8px;}
.name{font-weight:800;font-size:22px;color:#e8ecff;margin-bottom:6px;}
.sub{color:var(--muted);font-size:15px;}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:4px 0;}
.pill{padding:2px 10px;border-radius:9px;font-weight:800;font-size:18px;color:#0b0d12;min-width:42px;text-align:center;}
.rank{color:#94a0c6;font-weight:800;font-size:18px;text-align:right;}
.chip{background:var(--soft);color:#cbd5f5;border:1px solid #2d3550;padding:3px 10px;border-radius:9px;font-size:13px;}
.teamline{color:#e6ebff;font-size:15px;font-weight:400;margin-top:2px;display:flex;align-items:center;gap:8px;}
.badge{width:20px;height:20px;border-radius:4px;border:1px solid #2a3145;background:#0b0d12;object-fit:contain;}
.flag{width:18px;height:12px;border-radius:2px;border:1px solid #2a3145;object-fit:cover;}
.divider{height:12px;}
/* position colors */
.pos-CF{background:#0f2b6a;color:#cfe3ff;border:1px solid #1e3a8a;}
.pos-wide-blue{background:#143d8a;color:#d7e6ff;border:1px solid #2956a3;}
.pos-amf-light{background:#1f4d2a;color:#d8ffd8;border:1px solid #2d6a3a;}
.pos-cmf-green{background:#174c36;color:#d8ffe6;border:1px solid #226b4c;}
.pos-dmf-dark{background:#0f3f2a;color:#c9ffe5;border:1px solid #1c5a41;}
.pos-wingback{background:#5a4a00;color:#fff3bd;border:1px solid #7a6400;}
.pos-fullback{background:#6a3a00;color:#ffe0c2;border:1px solid #8a4a05;}
.pos-cb{background:#5a2900;color:#ffd8bf;border:1px solid #7a3a10;}
</style>
"""), unsafe_allow_html=True)

# ----------------- CONFIG -----------------
UA = {"User-Agent": "tiles/1.0 (+streamlit)"}
REQUIRED = {"Player","Team","League","Age","Position","Minutes played","Market value","Contract expires","Goals","Birth country"}
BETA_DISPLAY = 0.40   # always used for the two visible scores

# Roles
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
            "Progressive runs per 90": 2, "xA per 90": 3
        },
    },
    "All in": {
        "desc": "Blend of creation + scoring; balanced all-round attacking profile.",
        "metrics": {"xA per 90": 2, "Dribbles per 90": 2, "xG per 90": 3, "Non-penalty goals per 90": 3},
    },
}

# League strengths (same list you had)
LEAGUE_STRENGTHS = {  # trimmed for brevity in this reply — paste your full dict here
    'England 1.':100.00,'Italy 1.':97.14,'Spain 1.':94.29,'Germany 1.':94.29,'France 1.':91.43,
    'England 2.':71.43,'Portugal 1.':71.43,'Argentina 1.':71.43,'Belgium 1.':68.57,'Mexico 1.':68.57,
    # ... keep the rest of your mapping ...
    'England 10.':10.00,'Scotland 3.':10.00,'England 6.':10.00
}

# League presets
PRESETS = {
    "All leagues": None,
    "Top 5 (ENG/ESP/ITA/GER/FRA)": {'England 1.','Spain 1.','Italy 1.','Germany 1.','France 1.'},
    "England pyramid (1–4)": {'England 1.','England 2.','England 3.','England 4.'},
    "Big-8 + 2nd tiers": {'England 1.','Spain 1.','Italy 1.','Germany 1.','France 1.',
                          'Netherlands 1.','Portugal 1.','Belgium 1.','Germany 2.','France 2.','Spain 2.'},
}

# ----------------- small utils -----------------
def _norm(s:str)->str:
    s=(s or "").strip()
    s=unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    return s

def _slug(s:str)->str:
    return re.sub(r"[^a-z0-9]+","-", _norm(s).lower()).strip("-")

@st.cache_data(show_spinner=False, ttl=60*60*24)
def wiki_thumb(title:str, size:int)->str|None:
    try:
        r=requests.get("https://en.wikipedia.org/w/api.php", params={
            "action":"query","format":"json","prop":"pageimages","piprop":"thumbnail",
            "pithumbsize": size, "titles": title
        }, headers=UA, timeout=7)
        if not r.ok: return None
        pages=r.json().get("query",{}).get("pages",{})
        for _,p in pages.items():
            th=(p.get("thumbnail") or {}).get("source")
            if th: return th
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False, ttl=60*60*24)
def wiki_club_badge(team:str)->str|None:
    """Try multiple club title variants on Wikipedia for crest."""
    if not team: return None
    variants=[team, f"{team} F.C.", f"{team} FC", f"{team} football club", f"{team} A.F.C.", f"{team} C.F."]
    for v in variants:
        hit=wiki_thumb(v, 48)
        if hit: return hit
    return None

@st.cache_data(show_spinner=False, ttl=60*60*24)
def wiki_country_flag(country:str)->str|None:
    """Use Wikipedia 'Flag of X' page thumbnail; fall back to flagcdn if needed."""
    if not country: return None
    for title in (f"Flag of {country}", f"Flag of the {country}", f"{country} flag"):
        hit=wiki_thumb(title, 18)
        if hit: return hit
    # soft fallback if a weird country name breaks Wikipedia
    return f"https://flagcdn.com/w20/{_slug(country)[:2]}.png"  # crude, but avoids blank

# color ramp (SoFIFA-like)
PALETTE=[(0,(208,2,27)),(50,(245,166,35)),(65,(248,231,28)),(75,(126,211,33)),(85,(65,117,5)),(100,(40,90,4))]
def _lerp(a,b,t): return tuple(int(round(a[i]+(b[i]-a[i])*t)) for i in range(3))
def rating_color(v:float)->str:
    v=max(0.0, min(100.0, float(v)))
    for i in range(len(PALETTE)-1):
        x0,c0=PALETTE[i]; x1,c1=PALETTE[i+1]
        if v<=x1:
            t=0 if x1==x0 else (v-x0)/(x1-x0)
            r,g,b=_lerp(c0,c1,t); return f"rgb({r},{g},{b})"
    r,g,b=PALETTE[-1][1]; return f"rgb({r},{g},{b})"

POS_CLASS = {
    "CF":"pos-CF",
    "LWF":"pos-wide-blue","LW":"pos-wide-blue","LAMF":"pos-wide-blue","RW":"pos-wide-blue","RWF":"pos-wide-blue","RAMF":"pos-wide-blue",
    "AMF":"pos-amf-light","LCMF":"pos-cmf-green","RCMF":"pos-cmf-green",
    "RDMF":"pos-dmf-dark","LDMF":"pos-dmf-dark","DMF":"pos-dmf-dark",
    "LWB":"pos-wingback","RWB":"pos-wingback",
    "LB":"pos-fullback","RB":"pos-fullback",
    "RCB":"pos-cb","CB":"pos-cb","LCB":"pos-cb",
}
def pos_chips_html(position:str)->str:
    tags=[p.strip().upper() for p in (position or "").split(",")]
    chips=[]
    for p in tags:
        if p in POS_CLASS:
            chips.append(f"<span class='chip {POS_CLASS[p]}'>{p}</span>")
    chips.sort(key=lambda x: (0 if "CF" in x else 1))
    return " ".join(chips) or "<span class='chip pos-CF'>CF</span>"

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
missing=[c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("Filters")

    leagues_all=sorted(df["League"].dropna().unique().tolist())
    preset = st.selectbox("League preset", list(PRESETS.keys()), index=0)
    if PRESETS[preset] is None:
        leagues_default=leagues_all
    else:
        leagues_default=sorted(leagues for leagues in PRESETS[preset] if leagues in leagues_all)
    leagues_sel = st.multiselect("Leagues", leagues_all, default=leagues_default)

    df["Minutes played"]=pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"]=pd.to_numeric(df["Age"], errors="coerce")
    min_minutes, max_minutes = st.slider("Minutes played", 0, 5000, (500, 5000))
    a_min = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    a_max = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    amin, amax = st.slider("Age", a_min, a_max, (16, 40))

    df["Contract expires"]=pd.to_datetime(df["Contract expires"], errors="coerce")
    use_contract=st.checkbox("Filter by contract expiry", False)
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

    min_strength, max_strength = st.slider("League quality (strength)", 0, 101, (0, 101))
    beta_rank = st.slider("League weighting beta (for All-in ranking)", 0.0, 1.0, 0.40, 0.05)
    top_n = st.number_input("How many tiles", 5, 100, 20, 5)

# ----------------- FILTER POOL -----------------
df_f = df[df["League"].isin(leagues_sel)].copy()
df_f = df_f[df_f["Position"].astype(str).str.upper().str.startswith("CF")]   # CF gate
df_f = df_f[df_f["Minutes played"].between(min_minutes, max_minutes)]
df_f = df_f[df_f["Age"].between(amin, amax)]
if use_contract:
    df_f = df_f[df_f["Contract expires"].dt.year <= cutoff]
df_f = df_f[(df_f["Market value"] >= mv_lo) & (df_f["Market value"] <= mv_hi)]

# feature set = union of all role metrics
FEATURES = sorted({m for r in ROLES.values() for m in r["metrics"].keys()})
for col in FEATURES:
    df_f[col] = pd.to_numeric(df_f[col], errors="coerce")
df_f = df_f.dropna(subset=FEATURES)

if df_f.empty:
    st.warning("No players after filters. Loosen filters.")
    st.stop()

# league strength + bounds
df_f["League Strength"] = df_f["League"].map(LEAGUE_STRENGTHS).fillna(50.0)
df_f = df_f[(df_f["League Strength"] >= float(min_strength)) & (df_f["League Strength"] <= float(max_strength))]

# percentiles by league
for feat in FEATURES:
    df_f[f"{feat} %"] = df_f.groupby("League")[feat].transform(lambda x: x.rank(pct=True)*100.0)

def role_score(df_in:pd.DataFrame, key:str)->pd.Series:
    metrics=ROLES[key]["metrics"]; tot=sum(metrics.values()) or 1.0
    s=np.zeros(len(df_in))
    for m,w in metrics.items():
        col=f"{m} %"
        if col in df_in.columns:
            s+=df_in[col].values*w
    return s/tot

# visible scores (β=0.40 always)
df_f["GT_raw"] = role_score(df_f, "Goal Threat CF")
df_f["LU_raw"] = role_score(df_f, "Link-Up CF")
df_f["Goal Threat"] = (1-BETA_DISPLAY)*df_f["GT_raw"] + BETA_DISPLAY*df_f["League Strength"]
df_f["Link-Up CF"] = (1-BETA_DISPLAY)*df_f["LU_raw"] + BETA_DISPLAY*df_f["League Strength"]

# ranking score (uses sidebar β)
df_f["All in raw"] = role_score(df_f, "All in")
df_f["All in (weighted)"] = (1-beta_rank)*df_f["All in raw"] + beta_rank*df_f["League Strength"]

# contract year
df_f["Contract Year"] = pd.to_datetime(df_f["Contract expires"], errors="coerce").dt.year.fillna(0).astype(int)

# sort & take top N
ranked = df_f.sort_values("All in (weighted)", ascending=False).head(int(top_n)).reset_index(drop=True)

# ----------------- RENDER -----------------
for i,row in ranked.iterrows():
    rank=i+1
    player=str(row["Player"])
    team=str(row["Team"])
    pos=str(row["Position"])
    age=int(row["Age"]) if not pd.isna(row["Age"]) else 0
    gt=int(round(float(row["Goal Threat"])))
    lu=int(round(float(row["Link-Up CF"])))
    contract=int(row["Contract Year"])
    country=str(row.get("Birth country",""))

    badge = wiki_club_badge(team) or ""
    flag  = wiki_country_flag(country) or ""

    gt_style=f"background:{rating_color(gt)};"
    lu_style=f"background:{rating_color(lu)};"

    html = f"""
<div class="wrap">
  <div class="player-card">
    <div class="leftcol">
      <div class="avatar"></div>
      <div class="row">
        {('<img class="flag" src="'+flag+'" />') if flag else ''}
        <span class="chip">{age}y.o.</span>
        <span class="chip">{contract if contract>0 else '—'}</span>
      </div>
    </div>
    <div>
      <div class="name">{player}</div>
      <div class="row">
        <span class="pill" style="{gt_style}">{gt}</span>
        <span class="sub">Goal Threat</span>
      </div>
      <div class="row">
        <span class="pill" style="{lu_style}">{lu}</span>
        <span class="sub">Link-Up CF</span>
      </div>
      <div class="row">
        {pos_chips_html(pos)}
      </div>
      <div class="teamline">
        {('<img class="badge" src="'+badge+'" />') if badge else ""}
        {team}
      </div>
    </div>
    <div class="rank">#{rank}</div>
  </div>
</div>
<div class="divider"></div>
"""
    st.markdown(dedent(html), unsafe_allow_html=True)























