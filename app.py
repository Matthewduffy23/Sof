# app_top20_tiles.py — Top 20 Tiles (CF) + FIFA-style profile rows (no right pill)
# Requirements: streamlit, pandas, numpy

import io, re, math, unicodedata
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# ----------------- PAGE -----------------
st.set_page_config(page_title="Advanced Striker Scouting — Top 20 Tiles", layout="wide")
st.title("🔎 Advanced Striker Scouting — Top 20 Tiles")
st.caption(
    "Ranked by ‘All in’ (league-weighted). Pills show Goal Threat, Link-Up CF, Target Man CF. "
    "Pool = Position starts with CF. Flag = Birth country. No team badges."
)

# ----------------- STYLE -----------------
st.markdown("""
<style>
  :root { --bg:#0f1115; --card:#161a22; --muted:#a8b3cf; --soft:#202633; }
  .block-container { padding-top:.8rem; }
  body{ background:var(--bg); font-family: system-ui,-apple-system,'Segoe UI','Segoe UI Emoji',Roboto,Helvetica,Arial,sans-serif;}
  .wrap{ display:flex; justify-content:center; }
  .player-card{
    width:min(420px,96%); display:grid; grid-template-columns:96px 1fr 48px;
    gap:12px; align-items:start; background:var(--card); border:1px solid #252b3a;
    border-radius:18px; padding:16px;
  }
  .avatar{ width:96px; height:96px; border-radius:12px; background:#0b0d12 url('https://i.redd.it/43axcjdu59nd1.jpeg') center/cover no-repeat; border:1px solid #2a3145; }
  .leftcol{ display:flex; flex-direction:column; align-items:center; gap:8px; }
  .name{ font-weight:800; font-size:24px; color:#e8ecff; margin-bottom:6px; }
  .sub{ color:#a8b3cf; font-size:15px; }
  .pill{ padding:1.5px 5px; border-radius:9px; font-weight:800; font-size:14px; color:#0b0d12; display:inline-block; min-width:42px; text-align:center; }
  .row{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:4px 0; }
  .chip{ background:var(--soft); color:#cbd5f5; border:1px solid #2d3550; padding:3px 10px; border-radius:10px; font-size:13px; line-height:18px; }
  .flagchip{ display:inline-flex; align-items:center; gap:6px; background:var(--soft); color:#cbd5f5; border:1px solid #2d3550; padding:2px 8px; border-radius:10px; font-size:13px; line-height:18px; height:22px;}
  .flagchip img{ width:18px; height:14px; border-radius:2px; display:block; }
  .pos{ color:#eaf0ff; font-weight:700; padding:3px 8px; border-radius:10px; font-size:12px; border:1px solid rgba(255,255,255,.08); }
  .teamline{ color:#e6ebff; font-size:15px; font-weight:400; margin-top:8px; }
  .rank{ color:#94a0c6; font-weight:800; font-size:18px; text-align:right; }
  .divider{ height:12px; }

  /* ====== Profile (dropdown) — FIFA-style rows ====== */
  .metrics-card { background:#141821; border:1px solid #252b3a; border-radius:14px; padding:14px; }
  .section-h { color:#e8ecff; font-weight:800; font-size:18px; margin:8px 0 12px; }
  .metrics { width:100%; display:flex; flex-direction:column; gap:12px; }
  .metric-fifa { display:grid; grid-template-columns: 56px 1fr 240px; gap:12px; align-items:center; }
  .metric-fifa .lab { color:#d4dcff; font-size:18px; letter-spacing:.2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

  /* dashed lane & fill */
  .dashbar { position:relative; height:10px; border-radius:8px; background:
      repeating-linear-gradient(to right, #2e3447 0 14px, transparent 14px 22px);
      border:1px solid #2b344d; overflow:hidden; }
  .dashfill { position:absolute; left:0; top:0; height:100%;
      background: repeating-linear-gradient(to right, #6dde8f 0 14px, transparent 14px 22px); }
  .caret { position:absolute; top:-7px; width:0; height:0; border-left:6px solid transparent;
           border-right:6px solid transparent; border-bottom:10px solid #7BFF7B; transform:translateX(-50%); }

  /* Left pill in the profile rows */
  .leftpill { padding:2px 8px; border-radius:8px; font-weight:900; font-size:18px; color:#0b0d12; text-align:center; min-width:48px; }

  .streamlit-expanderHeader { font-weight:700; color:#e8ecff; }
  .stTabs [data-baseweb="tab"] { color:#a8b3cf; }
  .stTabs [aria-selected="true"] { color:#e8ecff !important; }
</style>
""", unsafe_allow_html=True)

# ----------------- CONFIG -----------------
INCLUDED_LEAGUES = [
    'England 1.','England 2.','England 3.','England 4.','England 5.','England 6.','England 7.','England 8.','England 9.','England 10.',
    'Albania 1.','Algeria 1.','Andorra 1.','Argentina 1.','Armenia 1.','Australia 1.','Austria 1.','Austria 2.','Azerbaijan 1.',
    'Belgium 1.','Belgium 2.','Bolivia 1.','Bosnia 1.','Brazil 1.','Brazil 2.','Brazil 3.','Bulgaria 1.','Canada 1.','Chile 1.',
    'Colombia 1.','Costa Rica 1.','Croatia 1.','Cyprus 1.','Czech 1.','Czech 2.','Denmark 1.','Denmark 2.','Ecuador 1.','Egypt 1.',
    'Estonia 1.','Finland 1.','France 1.','France 2.','France 3.','Georgia 1.','Germany 1.','Germany 2.','Germany 3.','Germany 4.',
    'Greece 1.','Hungary 1.','Iceland 1.','Israel 1.','Israel 2.','Italy 1.','Italy 2.','Italy 3.','Japan 1.','Japan 2.','Kazakhstan 1.',
    'Korea 1.','Latvia 1.','Lithuania 1.','Malta 1.','Mexico 1.','Moldova 1.','Morocco 1.','Netherlands 1.','Netherlands 2.',
    'North Macedonia 1.','Northern Ireland 1.','Norway 1.','Norway 2.','Paraguay 1.','Peru 1.','Poland 1.','Poland 2.','Portugal 1.',
    'Portugal 2.','Portugal 3.','Qatar 1.','Ireland 1.','Romania 1.','Russia 1.','Saudi 1.','Scotland 1.','Scotland 2.','Scotland 3.',
    'Serbia 1.','Serbia 2.','Slovakia 1.','Slovakia 2.','Slovenia 1.','Slovenia 2.','South Africa 1.','Spain 1.','Spain 2.','Spain 3.',
    'Sweden 1.','Sweden 2.','Switzerland 1.','Switzerland 2.','Tunisia 1.','Turkey 1.','Turkey 2.','Ukraine 1.','UAE 1.','USA 1.',
    'USA 2.','Uruguay 1.','Uzbekistan 1.','Venezuela 1.','Wales 1.'
]

# Metrics we may use (superset). If a column isn't in your CSV, it's skipped safely.
FEATURES = [
    'Defensive duels per 90','Defensive duels won, %','Aerial duels per 90','Aerial duels won, %','PAdj Interceptions',
    'Non-penalty goals per 90','xG per 90','Shots per 90','Shots on target, %','Goal conversion, %','Crosses per 90',
    'Accurate crosses, %','Dribbles per 90','Successful dribbles, %','Head goals per 90','Key passes per 90',
    'Touches in box per 90','Progressive runs per 90','Accelerations per 90','Passes per 90','Accurate passes, %',
    'xA per 90','Passes to penalty area per 90','Accurate passes to penalty area, %','Deep completions per 90','Smart passes per 90',
    'Offensive duels per 90','Offensive duels won, %'
]

ROLES = {
    'Goal Threat CF': {'metrics':{'Non-penalty goals per 90':3,'Shots per 90':1.5,'xG per 90':3,'Touches in box per 90':1,'Shots on target, %':0.5}},
    'Link-Up CF':     {'metrics':{'Passes per 90':2,'Passes to penalty area per 90':1.5,'Deep completions per 90':1,'Smart passes per 90':1.5,
                                  'Accurate passes, %':1.5,'Key passes per 90':1,'Dribbles per 90':2,'Successful dribbles, %':1,'Progressive runs per 90':2,'xA per 90':3}},
    'Target Man CF':  {'metrics':{'Aerial duels won, %':4,'Aerial duels per 90':3}},
    'All in':         {'metrics':{'xA per 90':2,'Dribbles per 90':2,'xG per 90':3,'Non-penalty goals per 90':3}},
}
LEAGUE_STRENGTHS = {
    'England 1.':100.00,'Italy 1.':97.14,'Spain 1.':94.29,'Germany 1.':94.29,'France 1.':91.43,'Brazil 1.':82.86,
    'England 2.':71.43,'Portugal 1.':71.43,'Argentina 1.':71.43,'Belgium 1.':68.57,'Mexico 1.':68.57,'Turkey 1.':65.71,
    'Germany 2.':65.71,'Spain 2.':65.71,'France 2.':65.71,'USA 1.':65.71,'Russia 1.':65.71,'Colombia 1.':62.86,
    'Netherlands 1.':62.86,'Austria 1.':62.86,'Switzerland 1.':62.86,'Denmark 1.':62.86,'Croatia 1.':62.86,'Japan 1.':62.86,
    'Korea 1.':62.86,'Italy 2.':62.86,'Czech 1.':57.14,'Norway 1.':57.14,'Poland 1.':57.14,'Romania 1.':57.14,'Israel 1.':57.14,
    'Algeria 1.':57.14,'Paraguay 1.':57.14,'Saudi 1.':57.14,'Uruguay 1.':57.14,'Morocco 1.':57.0,'Brazil 2.':56.0,'Ukraine 1.':55.0,
    'Ecuador 1.':54.29,'Spain 3.':54.29,'Scotland 1.':58.0,'Chile 1.':51.43,'Cyprus 1.':51.43,'Portugal 2.':51.43,'Slovakia 1.':51.43,
    'Australia 1.':51.43,'Hungary 1.':51.43,'Egypt 1.':51.43,'England 3.':51.43,'France 3.':48.0,'Japan 2.':48.0,'Bulgaria 1.':48.57,
    'Slovenia 1.':48.57,'Venezuela 1.':48.0,'Germany 3.':45.71,'Albania 1.':44.0,'Serbia 1.':42.86,'Belgium 2.':42.86,'Bosnia 1.':42.86,
    'Kosovo 1.':42.86,'Nigeria 1.':42.86,'Azerbaijan 1.':50.0,'Bolivia 1.':50.0,'Costa Rica 1.':50.0,'South Africa 1.':50.0,'UAE 1.':50.0,
    'Georgia 1.':40.0,'Finland 1.':40.0,'Italy 3.':40.0,'Peru 1.':40.0,'Tunisia 1.':40.0,'USA 2.':40.0,'Armenia 1.':40.0,
    'North Macedonia 1.':40.0,'Qatar 1.':40.0,'Uzbekistan 1.':42.0,'Norway 2.':42.0,'Kazakhstan 1.':42.0,'Poland 2.':38.0,
    'Denmark 2.':37.0,'Czech 2.':37.14,'Israel 2.':37.14,'Netherlands 2.':37.14,'Switzerland 2.':37.14,'Iceland 1.':34.29,
    'Ireland 1.':34.29,'Sweden 2.':34.29,'Germany 4.':34.29,'Malta 1.':30.0,'Turkey 2.':31.43,'Canada 1.':28.57,'England 4.':28.57,
    'Scotland 2.':28.57,'Moldova 1.':28.57,'Austria 2.':25.71,'Lithuania 1.':25.71,'Brazil 3.':25.0,'England 7.':25.0,'Slovenia 2.':22.0,
    'Latvia 1.':22.86,'Serbia 2.':20.0,'Slovakia 2.':20.0,'England 9.':20.0,'England 8.':15.0,'Montenegro 1.':14.29,'Wales 1.':12.0,
    'Portugal 3.':11.43,'Northern Ireland 1.':11.43,'England 10.':10.0,'Scotland 3.':10.0,'England 6.':10.0
}
REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Contract expires","Goals","Birth country"}

# ----------------- DATA LOADING -----------------
@st.cache_data(show_spinner=False)
def _read_csv_from_path(p:str)->pd.DataFrame: return pd.read_csv(p)
@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(b:bytes)->pd.DataFrame: return pd.read_csv(io.BytesIO(b))
def load_df(name="WORLDJUNE25.csv")->pd.DataFrame:
    candidates=[Path.cwd()/name]
    if "__file__" in globals(): candidates+=[Path(__file__).resolve().parent/name]
    for p in candidates:
        if p.exists(): return _read_csv_from_path(str(p))
    st.warning(f"Could not find **{name}**. Upload below.")
    up=st.file_uploader("Upload WORLDJUNE25.csv",type=["csv"])
    if up is None: st.stop()
    return _read_csv_from_bytes(up.getvalue())
df=load_df()

# ----------------- SIDEBAR -----------------
PRESETS={
    "All leagues": INCLUDED_LEAGUES,
    "Big 5": ['England 1.','Spain 1.','Italy 1.','Germany 1.','France 1.'],
    "Top 15-ish": ['England 1.','Spain 1.','Italy 1.','Germany 1.','France 1.','Portugal 1.','Netherlands 1.','Belgium 1.','Turkey 1.','Scotland 1.','Austria 1.','Denmark 1.','Switzerland 1.','Brazil 1.','Argentina 1.'],
    "UK & Ireland": ['England 1.','England 2.','England 3.','England 4.','Scotland 1.','Scotland 2.','Wales 1.','Ireland 1.','Northern Ireland 1.'],
}
with st.sidebar:
    st.header("Filters")
    preset=st.selectbox("League preset",list(PRESETS.keys()),index=0)
    leagues_avail=sorted(set(INCLUDED_LEAGUES)|set(df.get("League",pd.Series([])).dropna().unique()))
    leagues_sel=st.multiselect("Leagues",leagues_avail,default=PRESETS[preset])
    beta=st.slider("League weighting beta (applies to all scores)",0.0,1.0,0.40,0.05)
    df["Minutes played"]=pd.to_numeric(df["Minutes played"],errors="coerce")
    df["Age"]=pd.to_numeric(df["Age"],errors="coerce")
    min_minutes,max_minutes=st.slider("Minutes played",0,5000,(500,5000))
    a_min=int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    a_max=int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    min_age,max_age=st.slider("Age",a_min,a_max,(16,40))
    df["Contract expires"]=pd.to_datetime(df["Contract expires"],errors="coerce")
    apply_contract=st.checkbox("Filter by contract expiry",False)
    cutoff_year=st.slider("Max contract year (inclusive)",2025,2030,2026)
    df["Market value"]=pd.to_numeric(df["Market value"],errors="coerce")
    mv_max=int(np.nanmax(df["Market value"])) if df["Market value"].notna().any() else 50_000_000
    mv_cap=int(math.ceil(mv_max/5_000_000)*5_000_000)
    st.markdown("**Market value (€)**")
    use_m=st.checkbox("Adjust in millions",True)
    if use_m:
        max_m=int(mv_cap//1_000_000)
        mv_min_m,mv_max_m=st.slider("Range (M€)",0,max_m,(0,max_m))
        min_value=mv_min_m*1_000_000; max_value=mv_max_m*1_000_000
    else:
        min_value,max_value=st.slider("Range (€)",0,mv_cap,(0,mv_cap),step=100_000)
    min_strength,max_strength=st.slider("League quality (strength)",0,101,(0,101))
    top_n=st.number_input("How many tiles",5,100,20,5)

# ----------------- VALIDATION -----------------
missing=[c for c in REQUIRED_BASE if c not in df.columns]
if missing: st.error(f"Dataset missing required base columns: {missing}"); st.stop()

# ----------------- FILTER POOL -----------------
df_f=df[df["League"].isin(leagues_sel)].copy()
df_f=df_f[df_f["Position"].astype(str).str.upper().str.startswith("CF")]
df_f=df_f[df_f["Minutes played"].between(min_minutes,max_minutes)]
df_f=df_f[df_f["Age"].between(min_age,max_age)]
if apply_contract: df_f=df_f[df_f["Contract expires"].dt.year<=cutoff_year]
for c in FEATURES:
    if c in df_f.columns:
        df_f[c]=pd.to_numeric(df_f[c],errors="coerce")
# keep rows that have at least one metric present
metric_cols = [c for c in FEATURES if c in df_f.columns]
df_f = df_f.dropna(subset=metric_cols, how="all")
df_f["League Strength"]=df_f["League"].map(LEAGUE_STRENGTHS).fillna(50.0)
df_f=df_f[(df_f["League Strength"]>=float(min_strength))&(df_f["League Strength"]<=float(max_strength))]
df_f=df_f[(df_f["Market value"]>=min_value)&(df_f["Market value"]<=max_value)]
if df_f.empty: st.warning("No players after filters. Loosen filters."); st.stop()

# ----------------- Percentiles -----------------
for feat in metric_cols:
    df_f[f"{feat} Percentile"]=df_f.groupby("League")[feat].transform(lambda x: x.rank(pct=True)*100.0)

def role_score(df_in:pd.DataFrame,metrics:dict)->pd.Series:
    total_w=sum(metrics.values()) if metrics else 1.0
    wsum=np.zeros(len(df_in))
    for m,w in metrics.items():
        col=f"{m} Percentile"
        if col in df_in.columns:
            wsum+=df_in[col].fillna(0).values*w
    return wsum/total_w

# raw scores
df_f["Score_GT_raw"]=role_score(df_f,ROLES["Goal Threat CF"]["metrics"])
df_f["Score_LU_raw"]=role_score(df_f,ROLES["Link-Up CF"]["metrics"])
df_f["Score_TM_raw"]=role_score(df_f,ROLES["Target Man CF"]["metrics"])
df_f["Score_ALL_raw"]=role_score(df_f,ROLES["All in"]["metrics"])
# league-weighted
ls=df_f["League Strength"].astype(float)
beta=float(beta)
df_f["Score_GT"]=(1-beta)*df_f["Score_GT_raw"]+beta*ls
df_f["Score_LU"]=(1-beta)*df_f["Score_LU_raw"]+beta*ls
df_f["Score_TM"]=(1-beta)*df_f["Score_TM_raw"]+beta*ls
df_f["Score_ALL"]=(1-beta)*df_f["Score_ALL_raw"]+beta*ls

ranked=df_f.sort_values("Score_ALL",ascending=False).head(int(top_n)).copy().reset_index(drop=True)

# ----------------- Colors -----------------
PALETTE=[(0,(208,2,27)),(50,(245,166,35)),(65,(248,231,28)),(75,(126,211,33)),(85,(65,117,5)),(100,(40,90,4))]
def _lerp(a,b,t): return tuple(int(round(a[i]+(b[i]-a[i])*t)) for i in range(3))
def rating_color(v:float)->str:
    v=max(0.0,min(100.0,float(v)))
    for i in range(len(PALETTE)-1):
        x0,c0=PALETTE[i]; x1,c1=PALETTE[i+1]
        if v<=x1:
            t=0 if x1==x0 else (v-x0)/(x1-x0)
            r,g,b=_lerp(c0,c1,t); return f"rgb({r},{g},{b})"
    r,g,b=PALETTE[-1][1]; return f"rgb({r},{g},{b})"

POS_COLORS={
    "CF":"#183153","LWF":"#1f3f8c","LW":"#1f3f8c","LAMF":"#1f3f8c","RW":"#1f3f8c","RWF":"#1f3f8c","RAMF":"#1f3f8c",
    "AMF":"#87d37c","LCMF":"#2ecc71","RCMF":"#2ecc71","RDMF":"#0e7a3b","LDMF":"#0e7a3b",
    "LWB":"#e7d000","RWB":"#e7d000","LB":"#ff8a00","RB":"#ff8a00","RCB":"#c45a00","CB":"#c45a00","LCB":"#c45a00",
}
def chip_color(p:str)->str: return POS_COLORS.get(p.strip().upper(),"#2d3550")

# ----------------- Flags (Twemoji) -----------------
COUNTRY_TO_CC = {
    "united kingdom":"gb","great britain":"gb","northern ireland":"gb",
    "england":"eng","scotland":"sct","wales":"wls","ireland":"ie","republic of ireland":"ie",
    "spain":"es","france":"fr","germany":"de","italy":"it","portugal":"pt","netherlands":"nl","belgium":"be",
    "austria":"at","switzerland":"ch","denmark":"dk","sweden":"se","norway":"no","finland":"fi","iceland":"is",
    "poland":"pl","czech republic":"cz","czechia":"cz","slovakia":"sk","slovenia":"si","croatia":"hr","serbia":"rs",
    "bosnia and herzegovina":"ba","montenegro":"me","kosovo":"xk","albania":"al","greece":"gr","hungary":"hu",
    "romania":"ro","bulgaria":"bg","russia":"ru","ukraine":"ua","georgia":"ge","kazakhstan":"kz","azerbaijan":"az",
    "armenia":"am","turkey":"tr","qatar":"qa","saudi arabia":"sa","uae":"ae","israel":"il","morocco":"ma",
    "algeria":"dz","tunisia":"tn","egypt":"eg","nigeria":"ng","ghana":"gh","senegal":"sn","ivory coast":"ci",
    "cote d'ivoire":"ci","south africa":"za","brazil":"br","argentina":"ar","uruguay":"uy","chile":"cl",
    "colombia":"co","peru":"pe","ecuador":"ec","paraguay":"py","bolivia":"bo","mexico":"mx","canada":"ca",
    "united states":"us","usa":"us","japan":"jp","korea":"kr","south korea":"kr","china":"cn","australia":"au",
    "new zealand":"nz","latvia":"lv","lithuania":"lt","estonia":"ee","moldova":"md","north macedonia":"mk",
    "malta":"mt","cyprus":"cy","luxembourg":"lu","andorra":"ad","monaco":"mc","san marino":"sm",
}
TWEMOJI_SPECIAL = {
    "eng": "1f3f4-e0067-e0062-e0065-e006e-e0067-e007f",
    "sct": "1f3f4-e0067-e0062-e0073-e0063-e006f-e0074-e007f",
    "wls": "1f3f4-e0067-e0062-e0077-e0061-e006c-e007f",
}
def country_norm(s: str) -> str:
    if not s: return ""
    return unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii").strip().lower()
def cc_to_twemoji_code(cc: str) -> str | None:
    if not cc or len(cc) != 2: return None
    a, b = cc.upper()
    cp1 = 0x1F1E6 + (ord(a) - ord('A'))
    cp2 = 0x1F1E6 + (ord(b) - ord('A'))
    return f"{cp1:04x}-{cp2:04x}"
def flag_chip_html(country_name: str, age: int, contract_year: int) -> str:
    n = country_norm(country_name)
    cc = COUNTRY_TO_CC.get(n, "")
    if cc in TWEMOJI_SPECIAL:
        code = TWEMOJI_SPECIAL[cc]
        src = f"https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/{code}.svg"
        flag_html = f"<span class='flagchip'><img src='{src}' alt='{country_name}'></span>"
    else:
        code = cc_to_twemoji_code(cc) if len(cc) == 2 else None
        if code:
            src = f"https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/{code}.svg"
            flag_html = f"<span class='flagchip'><img src='{src}' alt='{country_name}'></span>"
        else:
            flag_html = "<span class='chip'>—</span>"
    age_chip = f"<span class='chip'>{age}y.o.</span>"
    yr = f"{contract_year}" if contract_year > 0 else "—"
    contract_chip = f"<span class='chip'>{yr}</span>"
    return f"<div class='row'>{flag_html}{age_chip}{contract_chip}</div>"

# ======= Profile helpers =======
def pct_of_row(row: pd.Series, metric: str) -> float:
    """Return 0..100 percentile for a metric; safe even if missing."""
    col = f"{metric} Percentile"
    v = row.get(col, np.nan)
    return float(v) if pd.notna(v) else 0.0

def make_row(label: str, pct: float) -> str:
    pct = max(0.0, min(100.0, float(pct)))
    color = rating_color(pct)
    return f"""
      <div class="metric-fifa">
        <div class="leftpill" style="background:{color}">{int(round(pct))}</div>
        <div class="lab">{label}</div>
        <div class="dashbar">
          <div class="dashfill" style="width:{pct:.0f}%;"></div>
          <div class="caret" style="left:{pct:.0f}%;"></div>
        </div>
      </div>
    """

def render_section(items, row) -> str:
    rows = []
    for lab, met in items:
        rows.append(make_row(lab, pct_of_row(row, met)))
    return f"<div class='metrics'>{''.join(rows)}</div>"

# === sections (exact titles & metrics you requested) ===
ATTACKING_SPEC = [
    ("Crosses","Crosses per 90"),
    ("Crossing Accuracy %","Accurate crosses, %"),
    ("Goals: Non-Penalty","Non-penalty goals per 90"),
    ("xG","xG per 90"),
    ("Conversion Rate %","Goal conversion, %"),
    ("Header Goals","Head goals per 90"),
    ("Expected Assists","xA per 90"),
    ("Offensive Duels","Offensive duels per 90"),
    ("Offensive Duel Success %","Offensive duels won, %"),
    ("Progressive Runs","Progressive runs per 90"),
    ("Shots","Shots per 90"),
    ("Shooting Accuracy %","Shots on target, %"),
    ("Touches in Opposition Box","Touches in box per 90"),
]
DEFENSIVE_SPEC = [
    ("Aerial Duels","Aerial duels per 90"),
    ("Aerial Duel Success %","Aerial duels won, %"),
    ("Defensive Duels","Defensive duels per 90"),
    ("Defensive Duel Success %","Defensive duels won, %"),
    ("PAdj. Interceptions","PAdj Interceptions"),
]
POSSESSION_SPEC = [
    ("Deep Completions","Deep completions per 90"),
    ("Dribbles","Dribbles per 90"),
    ("Dribbling Success %","Successful dribbles, %"),
    ("Key Passes","Key passes per 90"),
    ("Passes","Passes per 90"),
    ("Passing Accuracy %","Accurate passes, %"),
    ("Passes to Penalty Area","Passes to penalty area per 90"),
    ("Passes to Penalty Area %","Accurate passes to penalty area, %"),
    ("Smart Passes","Smart passes per 90"),
]

# ----------------- RENDER -----------------
ranked=df_f.sort_values("Score_ALL",ascending=False).head(int(top_n)).copy().reset_index(drop=True)

for idx,row in ranked.iterrows():
    rank = idx+1
    player = str(row.get("Player","")) or ""
    team   = str(row.get("Team","")) or ""
    pos_full = str(row.get("Position","")) or ""
    age = int(row.get("Age",0)) if not pd.isna(row.get("Age",np.nan)) else 0
    cy = pd.to_datetime(row.get("Contract expires"), errors="coerce")
    contract_year = int(cy.year) if pd.notna(cy) else 0
    birth_country = str(row.get("Birth country","") or "")

    gt_i = int(round(float(row["Score_GT"])))
    lu_i = int(round(float(row["Score_LU"])))
    tm_i = int(round(float(row["Score_TM"])))
    ov_style = f"background:{rating_color(gt_i)};"
    lu_style = f"background:{rating_color(lu_i)};"
    tm_style = f"background:{rating_color(tm_i)};"

    # position chips (CF first)
    codes = [c for c in re.split(r"[,/; ]+", pos_full.strip().upper()) if c]
    if "CF" in codes: codes = ["CF"] + [c for c in codes if c!="CF"]
    chips_html = "".join(f"<span class='pos' style='background:{chip_color(c)}'>{c}</span> " for c in dict.fromkeys(codes))

    # Tile
    st.markdown(f"""
    <div class='wrap'>
      <div class='player-card'>
        <div class='leftcol'>
          <div class='avatar'></div>
          {flag_chip_html(birth_country, age, contract_year)}
        </div>
        <div>
          <div class='name'>{player}</div>
          <div class='row' style='align-items:center;'>
            <span class='pill' style='{ov_style}'>{gt_i}</span>
            <span class='sub'>Goal Threat</span>
          </div>
          <div class='row' style='align-items:center;'>
            <span class='pill' style='{lu_style}'>{lu_i}</span>
            <span class='sub'>Link-Up CF</span>
          </div>
          <div class='row' style='align-items:center;'>
            <span class='pill' style='{tm_style}'>{tm_i}</span>
            <span class='sub'>Target Man CF</span>
          </div>
          <div class='row'>{chips_html}</div>
          <div class='teamline'>{team}</div>
        </div>
        <div class='rank'>#{rank}</div>
      </div>
    </div>
    <div class='divider'></div>
    """, unsafe_allow_html=True)

    # ---- Per-player profile (dropdown) ----
    with st.expander(f"📊 Profile: {player}", expanded=False):
        tabs = st.tabs(["Attacking", "Defensive", "Possession"])
        with tabs[0]:
            html = f"<div class='metrics-card'><div class='section-h'>ATTACKING</div>{render_section(ATTACKING_SPEC, row)}</div>"
            st.markdown(html, unsafe_allow_html=True)
        with tabs[1]:
            html = f"<div class='metrics-card'><div class='section-h'>DEFENSIVE</div>{render_section(DEFENSIVE_SPEC, row)}</div>"
            st.markdown(html, unsafe_allow_html=True)
        with tabs[2]:
            html = f"<div class='metrics-card'><div class='section-h'>POSSESSION</div>{render_section(POSSESSION_SPEC, row)}</div>"
            st.markdown(html, unsafe_allow_html=True)



































