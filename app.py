# -*- coding: utf-8 -*-
# Top 20 Tiles – CF-only, fixed avatar, FotMob team badges, Wikipedia birth-country + flag

import io, re, math, base64, unicodedata, json
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
    "CF-only. Team badges from FotMob. Birth country + flag from Wikipedia/Wikidata. "
    "All avatars use the same fallback image (no player headshots)."
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
        width:min(540px, 96%);
        display:grid;
        grid-template-columns: 110px 1fr 48px;
        gap:14px;
        align-items:center;
        background:var(--card);
        border:1px solid #252b3a;
        border-radius:18px;
        padding:16px;
      }
      .avatar {
        width:110px; height:110px; border-radius:14px;
        background-color:#0b0d12; border:1px solid #2a3145;
        background-repeat:no-repeat; background-position:center center; background-size:cover;
      }
      .icon { width:18px; height:18px; border-radius:3px; vertical-align:middle; margin-right:6px; }
      .leftcol { display:flex; flex-direction:column; align-items:center; gap:6px; }
      .name { font-weight:800; font-size:22px; color:#e8ecff; margin-bottom:6px; }
      .sub { color:#a8b3cf; font-size:15px; }
      .pill { padding:2px 10px; border-radius:9px; font-weight:800; font-size:18px; color:#0b0d12; display:inline-block; min-width:42px; text-align:center; }
      .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:4px 0; }
      .chip { background:var(--soft); color:#cbd5f5; border:1px solid #2d3550; padding:3px 10px; border-radius:9px; font-size:13px; }
      /* position colors */
      .chip.cf { background:#0b2777; color:#e6f0ff; border-color:#081b55; }          /* dark blue */
      .chip.blue { background:#1d4ed8; color:#e6f0ff; border-color:#1e3a8a; }        /* blue */
      .chip.lgreen { background:#2e7d32; color:#eafff0; border-color:#1f5a23; }      /* light green */
      .chip.green { background:#0f5132; color:#d1fae5; border-color:#0a3d25; }       /* green */
      .chip.dgreen { background:#084c38; color:#c7fff2; border-color:#063729; }      /* dark green */
      .chip.yellow { background:#806600; color:#fff7cc; border-color:#5c4800; }      /* yellow */
      .chip.orange { background:#7a3a00; color:#ffe9d6; border-color:#5a2b00; }      /* orange */
      .chip.dorange { background:#632000; color:#ffd9c7; border-color:#4a1800; }     /* dark orange */

      .teamline { color:#e6ebff; font-size:15px; font-weight:500; margin-top:2px; display:flex; align-items:center; gap:8px; }
      .rank { color:#94a0c6; font-weight:800; font-size:18px; text-align:right; }
      .divider { height:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- CONFIG -----------------
FALLBACK_URL = "https://i.redd.it/43axcjdu59nd1.jpeg"   # used for every avatar
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/128.0.0.0 Safari/537.36")}

# Helper normalization
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

def _first_word(s: str) -> str:
    return (_norm(s).split() + [""])[0]

def _fetch_image_as_data_uri(url: str, timeout=8) -> str | None:
    try:
        r = requests.get(url, headers=UA, timeout=timeout, stream=True)
        if not r.ok:
            return None
        ctype = r.headers.get("Content-Type", "").lower()
        if not ctype.startswith("image/"):
            if url.lower().endswith(".png"): ctype = "image/png"
            elif url.lower().endswith(".jpg") or url.lower().endswith(".jpeg"): ctype = "image/jpeg"
            elif url.lower().endswith(".webp"): ctype = "image/webp"
            else: return None
        b64 = base64.b64encode(r.content).decode("ascii")
        return f"data:{ctype};base64,{b64}"
    except Exception:
        return None

# ----------------- DATA LOADING -----------------
REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Contract expires","Goals"}

@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

def load_df(csv_name: str = "WORLDJUNE25.csv") -> pd.DataFrame:
    for p in [Path.cwd()/csv_name]:
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

    df["Market value"] = pd.to_numeric(df["Market value"], errors="coerce")
    mv_max_raw = int(np.nanmax(df["Market value"])) if df["Market value"].notna().any() else 50_000_000
    mv_cap = int(math.ceil(mv_max_raw / 5_000_000) * 5_000_000)
    st.markdown("**Market value (€)**")
    use_m = st.checkbox("Adjust in millions", True)
    if use_m:
        max_m = int(mv_cap // 1_000_000)
        mv_min_m, mv_max_m = st.slider("Range (M€)", 0, max_m, (0, max_m))
        min_value = mv_min_m * 1_000_000; max_value = mv_max_m * 1_000_000
    else:
        min_value, max_value = st.slider("Range (€)", 0, mv_cap, (0, mv_cap), step=100_000)

    beta = st.slider("League weighting beta (for OVERALL)", 0.0, 1.0, 0.40, 0.05)
    top_n = st.number_input("How many tiles", 5, 100, 20, 5)

    st.divider()
    st.subheader("Badges & Flags")
    show_img_source = st.checkbox("Show data sources", value=False)

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

# CF-only filter (starts with "CF")
df_f = df_f[df_f["Position"].astype(str).str.upper().str.strip().str.startswith("CF")]

if df_f.empty:
    st.warning("No CF players after filters.")
    st.stop()

# FEATURES (only what you need for scoring – keep yours if you want)
FEATURES = [
    'Non-penalty goals per 90','xG per 90','Shots per 90','Touches in box per 90',
    'Shots on target, %','Passes per 90','Passes to penalty area per 90',
    'Deep completions per 90','Smart passes per 90','Accurate passes, %',
    'Dribbles per 90','Successful dribbles, %','Progressive runs per 90','xA per 90',
]
for c in FEATURES:
    if c in df_f.columns:
        df_f[c] = pd.to_numeric(df_f[c], errors="coerce")
df_f = df_f.dropna(subset=[c for c in FEATURES if c in df_f.columns])

# Percentiles per league
for feat in FEATURES:
    if feat in df_f.columns:
        df_f[f"{feat} Percentile"] = df_f.groupby("League")[feat].transform(lambda x: x.rank(pct=True) * 100.0)

# Combined score
COMBINED_METRICS = {
    'Non-penalty goals per 90':3,'Shots per 90':1.5,'xG per 90':3,'Touches in box per 90':1,'Shots on target, %':0.5,
    'Passes per 90':2,'Passes to penalty area per 90':1.5,'Deep completions per 90':1,'Smart passes per 90':1.5,
    'Accurate passes, %':1.5,'Dribbles per 90':2,'Successful dribbles, %':1,'Progressive runs per 90':2,'xA per 90':3
}
def combined_role_score(df_in: pd.DataFrame, metrics: dict) -> pd.Series:
    total_w = sum(metrics.values()) if metrics else 1.0
    wsum = np.zeros(len(df_in))
    for m, w in metrics.items():
        col = f"{m} Percentile"
        if col in df_in.columns:
            wsum += df_in[col].values * w
    return wsum / total_w
df_f["Combined Score"] = combined_role_score(df_f, COMBINED_METRICS)

# Overall & Potential (simple)
def age_bonus(a: float) -> int:
    table = {27:0,26:1,25:3,24:3,23:4,22:5,21:6,20:6,19:7,18:8,17:9,16:10}
    a = int(a) if not pd.isna(a) else 27
    return 0 if a>=28 else (9 if a<16 else table.get(a,0))
df_f["Overall Rating"] = (1 - beta) * df_f["Combined Score"] + beta * 50
df_f["Potential"] = df_f.apply(lambda r: r["Overall Rating"] + age_bonus(r["Age"]), axis=1)

df_f["Contract Year"] = pd.to_datetime(df_f["Contract expires"], errors="coerce").dt.year.fillna(0).astype(int)

# ----------------- BADGES (FotMob) -----------------
def _score_team(q_simpl: str, name_simpl: str) -> float:
    # token jaccard
    ta, tb = set(q_simpl.split()), set(name_simpl.split())
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

@st.cache_data(show_spinner=False, ttl=60*60*24)
def fotmob_team_badge(team_name: str) -> tuple[str | None, str]:
    try:
        q = team_name.strip()
        r = requests.get("https://www.fotmob.com/api/search", params={"q": q}, headers=UA, timeout=6)
        if not r.ok:
            return None, ""
        data = r.json()
        best_id, best_s = None, -1.0
        qsim = _simplify_team(q)
        for t in data.get("teams", []):
            name = t.get("name","")
            s = _score_team(qsim, _simplify_team(name))
            if s > best_s:
                best_s, best_id = s, t.get("id")
        if best_id:
            # plain logo (png); fotmob also has *_small variants
            url = f"https://images.fotmob.com/image_resources/logo/teamlogo/{best_id}.png"
            if _fetch_image_as_data_uri(url):  # just verifying
                return url, "FotMob"
            return url, "FotMob"
    except Exception:
        return None, ""
    return None, ""

# ----------------- BIRTH COUNTRY + FLAG (Wikipedia/Wikidata) -----------------
@st.cache_data(show_spinner=False, ttl=60*60*24)
def wiki_birth_country_and_flag(player_name: str) -> tuple[str | None, str | None, str]:
    """
    Returns (country_name, flag_url_or_data_uri, source_tag).
    Strategy:
      - Wikipedia search → page → pageprops.wikibase_item (QID)
      - Wikidata: prefer P27 (country of citizenship).
        Else P19 (place of birth) → that item's P17 (country).
      - Get country's label and P297 (alpha-2) → FlagCDN https://flagcdn.com/w40/{alpha2}.png
    """
    try:
        # search wikipedia for page
        r = requests.get("https://en.wikipedia.org/w/api.php",
                         params={"action":"query","format":"json","list":"search",
                                 "srsearch": f"{player_name} footballer", "srlimit": 1},
                         headers=UA, timeout=8)
        hits = (r.json().get("query",{}) or {}).get("search",[])
        if not hits: return None, None, ""
        pageid = hits[0].get("pageid")
        r2 = requests.get("https://en.wikipedia.org/w/api.php",
                          params={"action":"query","format":"json","prop":"pageprops",
                                  "pageids": pageid},
                          headers=UA, timeout=8)
        pages = (r2.json().get("query",{}) or {}).get("pages",{})
        props = list(pages.values())[0].get("pageprops",{})
        qid = props.get("wikibase_item")
        if not qid: return None, None, ""

        # pull entity
        r3 = requests.get(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json", headers=UA, timeout=10)
        ent = (r3.json().get("entities",{}) or {}).get(qid,{})
        claims = ent.get("claims",{})

        def _claim_q(claims_dict, pid):
            arr = claims_dict.get(pid, [])
            for c in arr:
                try:
                    return c["mainsnak"]["datavalue"]["value"]["id"]
                except Exception:
                    continue
            return None

        country_q = _claim_q(claims, "P27")  # country of citizenship
        if not country_q:
            # try place of birth -> country
            pob_q = _claim_q(claims, "P19")
            if pob_q:
                r4 = requests.get(f"https://www.wikidata.org/wiki/Special:EntityData/{pob_q}.json", headers=UA, timeout=10)
                pob_ent = (r4.json().get("entities",{}) or {}).get(pob_q,{})
                country_q = _claim_q(pob_ent.get("claims",{}), "P17")

        if not country_q:
            return None, None, ""

        # country entity
        r5 = requests.get(f"https://www.wikidata.org/wiki/Special:EntityData/{country_q}.json", headers=UA, timeout=10)
        c_ent = (r5.json().get("entities",{}) or {}).get(country_q,{})
        cname = (c_ent.get("labels",{}).get("en") or {}).get("value")
        cclaims = c_ent.get("claims",{})
        alpha2 = None
        for c in cclaims.get("P297", []):  # ISO 3166-1 alpha-2
            try:
                alpha2 = c["mainsnak"]["datavalue"]["value"].lower()
                break
            except Exception:
                continue
        flag_url = f"https://flagcdn.com/w40/{alpha2}.png" if alpha2 else None
        flag_data = _fetch_image_as_data_uri(flag_url) if flag_url else None
        return cname, (flag_data or flag_url), "Wikipedia/Wikidata"
    except Exception:
        return None, None, ""

# ----------------- COLORS -----------------
PALETTE = [(0,(208,2,27)),(50,(245,166,35)),(65,(248,231,28)),(75,(126,211,33)),(85,(65,117,5)),(100,(40,90,4))]
def _lerp(a, b, t): return tuple(int(round(a[i] + (b[i]-a[i]) * t)) for i in range(3))
def rating_color(v: float) -> str:
    v = max(0.0, min(100.0, float(v)))
    for i in range(len(PALETTE)-1):
        x0,c0=PALETTE[i]; x1,c1=PALETTE[i+1]
        if v <= x1:
            t = 0 if x1==x0 else (v-x0)/(x1-x0); r,g,b = _lerp(c0,c1,t); return f"rgb({r},{g},{b})"
    r,g,b = PALETTE[-1][1]; return f"rgb({r},{g},{b})"

# position chip mapping per your rule
POS_CLASS = {
    # CF & winger/AM family
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
ranked = df_f.sort_values("Overall Rating", ascending=False).head(int(top_n)).copy().reset_index(drop=True)

for idx, row in ranked.iterrows():
    rank = idx + 1
    player = str(row.get("Player","")) or ""
    team   = str(row.get("Team","")) or ""
    pos    = str(row.get("Position","")) or ""

    age = int(row.get("Age",0)) if not pd.isna(row.get("Age",np.nan)) else 0
    overall_i   = int(round(float(row["Overall Rating"])))
    potential_i = int(round(float(row["Potential"])))
    contract_year = int(row.get("Contract Year",0))

    # Fixed avatar
    avatar_style = f"background-image: url('{FALLBACK_URL}');"

    # FotMob badge
    badge_url, badge_src = fotmob_team_badge(team)
    badge_img = f"<img class='icon' src='{_fetch_image_as_data_uri(badge_url) or badge_url}'/>" if badge_url else ""

    # Birth country + flag from Wikipedia/Wikidata
    bcountry, flag_url, bsrc = wiki_birth_country_and_flag(player)
    flag_img = f"<img class='icon' src='{_fetch_image_as_data_uri(flag_url) or flag_url}'/>" if flag_url else ""
    bcountry_html = f"<span class='chip'>{bcountry}</span>" if bcountry else ""

    ov_style = f"background:{rating_color(overall_i)};"
    po_style = f"background:{rating_color(potential_i)};"

    src_bits = []
    if show_img_source:
        if badge_src: src_bits.append(f"Badge: {badge_src}")
        if bsrc:      src_bits.append(f"Birth: {bsrc}")
    src_html = f"<div class='sub'>{' | '.join(src_bits)}</div>" if src_bits else ""

    pos_html = render_pos_chips(pos)

    st.markdown(f"""
    <div class='wrap'>
      <div class='player-card'>
        <div class='leftcol'>
          <div class='avatar' style="{avatar_style}"></div>
          <div class='row'>
            {flag_img}{bcountry_html}<span class='chip'>{age}y.o.</span>
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
          <div class='row'>{pos_html}</div>
          <div class='teamline'>{badge_img}<span>{team}</span></div>
          <div class='row'><span class='chip'>{contract_year if contract_year>0 else "—"}</span></div>
          {src_html}
        </div>
        <div class='rank'>#{rank}</div>
      </div>
    </div>
    <div class='divider'></div>
    """, unsafe_allow_html=True)




















