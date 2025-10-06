# app_top20_tiles.py — Top 20 CF Tiles (league-weighted role pills, flags, team badges)
# Requirements: streamlit, pandas, numpy, requests, beautifulsoup4

import io, re, math, unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
import numpy as np

# ----------------- PAGE & THEME -----------------
st.set_page_config(page_title="Advanced Striker Scouting — Top 20 Tiles", layout="wide")
st.title("🔎 Advanced Striker Scouting — Top 20 Tiles")
st.caption(
    "Rank = All in (balanced blend). Shown pills: Goal Threat & Link-Up CF (both league-weighted with β=0.40). "
    "Flags from dataset ‘Birth country’. Team badges via FotMob. Avatar is a default placeholder."
)

# ----------------- STYLE (half width, centered) -----------------
st.markdown("""
<style>
:root { --bg:#0f1115; --card:#161a22; --muted:#a8b3cf; --soft:#202633; }
.block-container { padding-top:.8rem; }
body { background:var(--bg); }
.wrap { display:flex; justify-content:center; }
.player-card{
  width:min(420px,96%); display:grid; grid-template-columns:96px 1fr 48px;
  gap:12px; align-items:center; background:var(--card);
  border:1px solid #252b3a; border-radius:18px; padding:16px;
}
.avatar{
  width:96px; height:96px; border-radius:12px; background:#0b0d12 url('https://i.redd.it/43axcjdu59nd1.jpeg') center/cover no-repeat;
  border:1px solid #2a3145;
}
.leftcol{ display:flex; flex-direction:column; gap:6px; align-items:center; }
.name{ font-weight:800; font-size:22px; color:#e8ecff; margin-bottom:4px; }
.sub{ color:var(--muted); font-size:15px; }
.row{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:4px 0; }
.pill{
  padding:2px 10px; border-radius:9px; font-weight:800; font-size:18px; color:#0b0d12;
  min-width:42px; text-align:center; display:inline-block;
}
.rank{ color:#94a0c6; font-weight:800; font-size:18px; text-align:right; }
.chip{ background:var(--soft); color:#cbd5f5; border:1px solid #2d3550; padding:3px 10px; border-radius:9px; font-size:13px; }
.teamline{ color:#e6ebff; font-size:15px; font-weight:500; margin-top:2px; display:flex; gap:8px; align-items:center; }
.teamlogo{ width:18px; height:18px; border-radius:3px; background:#0b0d12 center/cover no-repeat; border:1px solid #2a3145; display:inline-block; }
.flag{ font-size:18px; }
.divider{ height:12px; }

/* position tag colours */
.tag{ padding:2px 8px; border-radius:10px; font-size:12px; font-weight:700; color:#fff; border:1px solid #2d3550;}
.tag-blue-dark{ background:#1f3a8a; }        /* CF */
.tag-blue{ background:#2563eb; }             /* LW,RW,LWF,RWF,LAMF,RAMF */
.tag-green-light{ background:#22c55e; }      /* AMF */
.tag-green{ background:#16a34a; }            /* LCMF,RCMF */
.tag-green-dark{ background:#065f46; }       /* LDMF,RDMF */
.tag-yellow{ background:#ca8a04; color:#0b0d12;} /* LWB,RWB */
.tag-orange{ background:#ea580c; }           /* LB,RB */
.tag-orange-dark{ background:#9a3412; }      /* CB,LCB,RCB */
</style>
""", unsafe_allow_html=True)

# ----------------- CONSTANTS -----------------
UA = {"User-Agent": "tiles-app/1.0 (+streamlit)"}
BETA = 0.40  # fixed league-weighting beta

# Full league catalog (same style as your earlier app)
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

# League strengths (your original long table)
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

# Metrics to compute percentiles for
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
    'Deep completions per 90','Smart passes per 90'
]

# Role weights
ROLES = {
    'Goal Threat CF': {'metrics': {
        'Non-penalty goals per 90': 3,'Shots per 90': 1.5,'xG per 90': 3,
        'Touches in box per 90': 1,'Shots on target, %': 0.5
    }},
    'Link-Up CF': {'metrics': {
        'Passes per 90': 2,'Passes to penalty area per 90': 1.5,
        'Deep completions per 90': 1,'Smart passes per 90': 1.5,
        'Accurate passes, %': 1.5,'Key passes per 90': 1,
        'Dribbles per 90': 2,'Successful dribbles, %': 1,
        'Progressive runs per 90': 2,'xA per 90': 3
    }},
    'All in': {'metrics': {'xA per 90': 2,'Dribbles per 90': 2,'xG per 90': 3,'Non-penalty goals per 90': 3}}
}

REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Contract expires","Birth country"}

# ----------------- DATA LOADING -----------------
@st.cache_data(show_spinner=False)  # path
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)  # bytes
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

def load_df(csv_name: str = "WORLDJUNE25.csv") -> pd.DataFrame:
    candidates = [Path.cwd() / csv_name]
    for p in candidates:
        if p.exists(): return _read_csv_from_path(str(p))
    st.warning(f"Could not find **{csv_name}**. Upload below.")
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if up is None: st.stop()
    return _read_csv_from_bytes(up.getvalue())

df = load_df()

# ----------------- HELPERS: normalize, flags, team badge -----------------
def _norm(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    return s.lower()

# Expanded map (add more anytime)
COUNTRY_TO_ISO2 = {
    # UK variants
    "england":"GB","scotland":"GB","wales":"GB","northern ireland":"GB",
    "united kingdom":"GB","uk":"GB","great britain":"GB",
    # Europe
    "albania":"AL","andorra":"AD","austria":"AT","belarus":"BY","belgium":"BE","bosnia and herzegovina":"BA",
    "bulgaria":"BG","croatia":"HR","cyprus":"CY","czech republic":"CZ","czechia":"CZ","denmark":"DK",
    "estonia":"EE","finland":"FI","france":"FR","georgia":"GE","germany":"DE","greece":"GR","hungary":"HU",
    "iceland":"IS","ireland":"IE","israel":"IL","italy":"IT","kazakhstan":"KZ","kosovo":"XK","latvia":"LV",
    "lithuania":"LT","luxembourg":"LU","malta":"MT","moldova":"MD","montenegro":"ME","netherlands":"NL",
    "north macedonia":"MK","norway":"NO","poland":"PL","portugal":"PT","romania":"RO","russia":"RU",
    "san marino":"SM","serbia":"RS","slovakia":"SK","slovenia":"SI","spain":"ES","sweden":"SE",
    "switzerland":"CH","turkey":"TR","ukraine":"UA","monaco":"MC","liechtenstein":"LI",
    # Africa
    "algeria":"DZ","angola":"AO","benin":"BJ","botswana":"BW","burkina faso":"BF","burundi":"BI","cameroon":"CM",
    "cape verde":"CV","central african republic":"CF","chad":"TD","comoros":"KM","congo":"CG","congo dr":"CD",
    "democratic republic of the congo":"CD","dr congo":"CD","ivory coast":"CI","côte d’ivoire":"CI","cote d'ivoire":"CI",
    "djibouti":"DJ","egypt":"EG","equatorial guinea":"GQ","eritrea":"ER","ethiopia":"ET","gabon":"GA","gambia":"GM",
    "ghana":"GH","guinea":"GN","guinea-bissau":"GW","kenya":"KE","lesotho":"LS","liberia":"LR","libya":"LY",
    "madagascar":"MG","malawi":"MW","mali":"ML","mauritania":"MR","mauritius":"MU","morocco":"MA","mozambique":"MZ",
    "namibia":"NA","niger":"NE","nigeria":"NG","rwanda":"RW","sao tome and principe":"ST","senegal":"SN","seychelles":"SC",
    "sierra leone":"SL","somalia":"SO","south africa":"ZA","south sudan":"SS","sudan":"SD","tanzania":"TZ","togo":"TG",
    "tunisia":"TN","uganda":"UG","zambia":"ZM","zimbabwe":"ZW",
    # Asia + Oceania
    "afghanistan":"AF","armenia":"AM","azerbaijan":"AZ","bahrain":"BH","bangladesh":"BD","bhutan":"BT","brunei":"BN",
    "cambodia":"KH","china":"CN","east timor":"TL","hong kong":"HK","india":"IN","indonesia":"ID","iran":"IR",
    "iraq":"IQ","japan":"JP","jordan":"JO","kuwait":"KW","kyrgyzstan":"KG","laos":"LA","lebanon":"LB","macau":"MO",
    "malaysia":"MY","maldives":"MV","mongolia":"MN","myanmar":"MM","nepal":"NP","north korea":"KP","oman":"OM",
    "pakistan":"PK","palestine":"PS","philippines":"PH","qatar":"QA","saudi arabia":"SA","singapore":"SG",
    "south korea":"KR","korea":"KR","sri lanka":"LK","syria":"SY","taiwan":"TW","tajikistan":"TJ","thailand":"TH",
    "turkmenistan":"TM","united arab emirates":"AE","uae":"AE","uzbekistan":"UZ","vietnam":"VN",
    "australia":"AU","new zealand":"NZ",
    # Americas
    "antigua and barbuda":"AG","argentina":"AR","bahamas":"BS","barbados":"BB","belize":"BZ","bolivia":"BO",
    "brazil":"BR","canada":"CA","chile":"CL","colombia":"CO","costa rica":"CR","cuba":"CU","dominica":"DM",
    "dominican republic":"DO","ecuador":"EC","el salvador":"SV","grenada":"GD","guatemala":"GT","guyana":"GY",
    "haiti":"HT","honduras":"HN","jamaica":"JM","mexico":"MX","nicaragua":"NI","panama":"PA","paraguay":"PY",
    "peru":"PE","saint kitts and nevis":"KN","saint lucia":"LC","saint vincent and the grenadines":"VC",
    "suriname":"SR","trinidad and tobago":"TT","uruguay":"UY","venezuela":"VE","united states":"US","usa":"US"
}

def iso2_to_flag(iso2: str) -> str:
    if not iso2 or len(iso2) != 2 or not iso2.isalpha(): return "🌍"
    return chr(ord(iso2[0].upper())+127397) + chr(ord(iso2[1].upper())+127397)

def country_to_flag(country_name: str) -> str:
    n = _norm(country_name)
    code = COUNTRY_TO_ISO2.get(n)
    if not code:
        # crude fallback: take first 2 ASCII letters if present
        letters = re.sub(r"[^a-zA-Z]", "", country_name or "")
        code = (letters[:2].upper() if len(letters) >= 2 else "  ")
        if not code.isalpha(): return "🌍"
    return iso2_to_flag(code)

@st.cache_data(show_spinner=False, ttl=60*60*24)
def fotmob_team_badge(team_name: str) -> str | None:
    """Search FotMob for team → return badge URL, else None (leave blank)."""
    if not team_name: return None
    try:
        r = requests.get("https://www.fotmob.com/api/search", params={"q": team_name}, headers=UA, timeout=6)
        if not r.ok: return None
        js = r.json()
        best_id, best_score = None, -1
        for t in js.get("teams", []):
            name = t.get("name","")
            tid = t.get("id")
            score = 0
            if _norm(team_name) in _norm(name) or _norm(name) in _norm(team_name): score += 3
            if tid: score += 1
            if score > best_score:
                best_score, best_id = score, tid
        if best_id: return f"https://images.fotmob.com/image_resources/logo/teamlogo/{best_id}.png"
    except Exception:
        return None
    return None

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("Filters")

    # League presets
    preset = st.selectbox("League preset",
        ["All leagues", "Top 5 (ENG/ESP/GER/ITA/FRA)", "England 1+2", "England 1 only"], index=0)

    leagues_avail = sorted(set(INCLUDED_LEAGUES) | set(df.get("League", pd.Series([])).dropna().unique()))
    if preset == "Top 5 (ENG/ESP/GER/ITA/FRA)":
        default_leagues = ["England 1.","Spain 1.","Germany 1.","Italy 1.","France 1."]
    elif preset == "England 1+2":
        default_leagues = ["England 1.","England 2."]
    elif preset == "England 1 only":
        default_leagues = ["England 1."]
    else:
        default_leagues = leagues_avail

    leagues_sel = st.multiselect("Leagues", leagues_avail, default=default_leagues)

    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    _mn, _mx = int(np.nanmin(df["Minutes played"])), int(np.nanmax(df["Minutes played"]))
    min_minutes, max_minutes = st.slider("Minutes played", 0, max(1000,_mx), (500, max(500,_mx)))

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    age_min_data = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_data = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    min_age, max_age = st.slider("Age", age_min_data, age_max_data, (16, 40))

    df["Contract expires"] = pd.to_datetime(df["Contract expires"], errors="coerce")
    apply_contract = st.checkbox("Filter by contract expiry", value=False)
    cutoff_year = st.slider("Max contract year (inclusive)", 2025, 2030, 2026)

    # Market value optional filter
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

    st.markdown("**How many tiles**")
    top_n = st.number_input("", 5, 100, 20, 5)

# ----------------- VALIDATION -----------------
missing = [c for c in REQUIRED_BASE if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()
missing_feats = [c for c in FEATURES if c not in df.columns]
if missing_feats:
    st.error(f"Dataset missing feature columns: {missing_feats}")
    st.stop()

# ----------------- FILTER POOL -----------------
df_f = df[df["League"].isin(leagues_sel)].copy()
df_f = df_f[df_f["Position"].astype(str).str.upper().str.startswith("CF")]  # CF-only
df_f = df_f[df_f["Minutes played"].between(min_minutes, max_minutes)]
df_f = df_f[df_f["Age"].between(min_age, max_age)]
if apply_contract:
    df_f = df_f[df_f["Contract expires"].dt.year <= cutoff_year]
df_f = df_f[(df_f["Market value"] >= min_value) & (df_f["Market value"] <= max_value)]

for c in FEATURES:
    df_f[c] = pd.to_numeric(df_f[c], errors="coerce")
df_f = df_f.dropna(subset=FEATURES)
if df_f.empty:
    st.warning("No players after filters. Loosen filters.")
    st.stop()

# League strengths
df_f["League Strength"] = df_f["League"].map(LEAGUE_STRENGTHS).fillna(50.0)

# ----------------- Percentiles per league -----------------
for feat in FEATURES:
    df_f[f"{feat} Percentile"] = df_f.groupby("League")[feat].transform(lambda x: x.rank(pct=True)*100.0)

def weighted_score(df_in: pd.DataFrame, metrics: dict) -> pd.Series:
    total_w = sum(metrics.values()) or 1.0
    s = np.zeros(len(df_in))
    for k, w in metrics.items():
        col = f"{k} Percentile"
        if col in df_in.columns:
            s += df_in[col].values * w
    return s / total_w

# League-weighted role scores with fixed β
for role_name, role in ROLES.items():
    raw = weighted_score(df_f, role["metrics"])
    df_f[role_name] = (1 - BETA)*raw + BETA*df_f["League Strength"]

# Ranking key
df_f["__sort"] = df_f["All in"]
ranked = df_f.sort_values("__sort", ascending=False).head(int(top_n)).copy().reset_index(drop=True)
df_f.drop(columns=["__sort"], inplace=True)

# ----------------- COLOR SCALE FOR PILLS -----------------
PALETTE = [
    (0,(208,2,27)), (50,(245,166,35)), (65,(248,231,28)),
    (75,(126,211,33)), (85,(65,117,5)), (100,(40,90,4))
]
def _lerp(a,b,t): return tuple(int(round(a[i]+(b[i]-a[i])*t)) for i in range(3))
def rating_color(v: float) -> str:
    v = max(0,min(100,float(v)))
    for i in range(len(PALETTE)-1):
        x0,c0 = PALETTE[i]; x1,c1 = PALETTE[i+1]
        if v <= x1:
            t = 0 if x1==x0 else (v-x0)/(x1-x0)
            r,g,b = _lerp(c0,c1,t); return f"rgb({r},{g},{b})"
    r,g,b = PALETTE[-1][1]; return f"rgb({r},{g},{b})"

# ----------------- POSITION TAGS -----------------
TAG_COLORS = {
    "CF":"tag-blue-dark",
    "LWF":"tag-blue","LW":"tag-blue","RWF":"tag-blue","RW":"tag-blue","LAMF":"tag-blue","RAMF":"tag-blue",
    "AMF":"tag-green-light","LCMF":"tag-green","RCMF":"tag-green",
    "LDMF":"tag-green-dark","RDMF":"tag-green-dark",
    "LWB":"tag-yellow","RWB":"tag-yellow",
    "LB":"tag-orange","RB":"tag-orange",
    "CB":"tag-orange-dark","LCB":"tag-orange-dark","RCB":"tag-orange-dark"
}
def render_tags(pos_string: str) -> str:
    tags = []
    seen_cf = False
    for raw in re.split(r"[,/]|\\s+", str(pos_string)):
        t = raw.strip().upper()
        if not t: continue
        if t == "CF": seen_cf = True
        cls = TAG_COLORS.get(t)
        if cls: tags.append(f"<span class='tag {cls}'>{t}</span>")
    if not seen_cf:
        tags.insert(0, "<span class='tag tag-blue-dark'>CF</span>")
    return " ".join(tags[:6])

# ----------------- RENDER -----------------
for idx, row in ranked.iterrows():
    rank = idx + 1
    player = str(row.get("Player","")) or ""
    team = str(row.get("Team","")) or ""
    pos = str(row.get("Position","")) or ""
    age = int(row.get("Age",0)) if not pd.isna(row.get("Age",np.nan)) else 0
    contract_year = pd.to_datetime(row.get("Contract expires"), errors="coerce")
    contract_y = int(contract_year.year) if pd.notna(contract_year) else 0

    gt_i = int(round(float(row["Goal Threat CF"])))
    lu_i = int(round(float(row["Link-Up CF"])))

    flag = country_to_flag(str(row.get("Birth country","")))
    badge_url = fotmob_team_badge(team)
    badge_html = f"<span class='teamlogo' style=\"background-image:url('{badge_url}')\"></span>" if badge_url else ""

    ov_style = f"background:{rating_color(gt_i)};"
    po_style = f"background:{rating_color(lu_i)};"

    st.markdown(f"""
    <div class='wrap'>
      <div class='player-card'>
        <div class='leftcol'>
          <div class='avatar'></div>
          <div class='row'>
            <span class='flag'>{flag}</span>
            <span class='chip'>{age}y.o.</span>
            <span class='chip'>{contract_y if contract_y>0 else '—'}</span>
          </div>
        </div>

        <div>
          <div class='name'>{player}</div>
          <div class='row'>
            <span class='pill' style="{ov_style}">{gt_i}</span>
            <span class='sub'>Goal Threat</span>
          </div>
          <div class='row'>
            <span class='pill' style="{po_style}">{lu_i}</span>
            <span class='sub'>Link-Up CF</span>
          </div>
          <div class='row'>
            {render_tags(pos)}
          </div>
          <div class='teamline'>
            {badge_html}<span>{team}</span>
          </div>
        </div>

        <div class='rank'>#{rank}</div>
      </div>
    </div>
    <div class='divider'></div>
    """, unsafe_allow_html=True)
























