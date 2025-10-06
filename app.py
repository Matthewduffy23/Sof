# app_top20_tiles.py — Top 20 Tiles with multi-source headshots + club badge + flag + colored positions
# Requirements: streamlit, pandas, numpy, requests, beautifulsoup4  (optional: lxml)

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
    "Overall = league-weighted combined-role score. Potential = Overall + age bonus. "
    "Headshots: Mapping → SofaScore → SoFIFA → FIFACM → FotMob → Transfermarkt → Flashscore → Wikipedia → fallback. "
    "Badges/flags via SofaScore (or Wikipedia/flagcdn fallback)."
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
        background-repeat:no-repeat;
        background-position:center center;
        background-size:cover;
      }
      .icon { width:18px; height:18px; border-radius:3px; vertical-align:middle; margin-right:6px; }
      .source { color:#7f8cb5; font-size:11px; margin-top:3px; }
      .leftcol { display:flex; flex-direction:column; align-items:center; gap:4px; }
      .name { font-weight:800; font-size:22px; color:#e8ecff; margin-bottom:6px; }
      .sub { color:#a8b3cf; font-size:15px; }
      .pill { padding:2px 10px; border-radius:9px; font-weight:800; font-size:18px; color:#0b0d12; display:inline-block; min-width:42px; text-align:center; }
      .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:4px 0; }
      .chip { background:var(--soft); color:#cbd5f5; border:1px solid #2d3550; padding:3px 10px; border-radius:9px; font-size:13px; }
      .chip.blue { background:#1d4ed8; color:#e6f0ff; border:1px solid #1e3a8a; }
      .chip.green { background:#0f5132; color:#d1fae5; border:1px solid #0a3d25; }
      .teamline { color:#e6ebff; font-size:15px; font-weight:500; margin-top:2px; display:flex; align-items:center; gap:8px; }
      .rank { color:#94a0c6; font-weight:800; font-size:18px; text-align:right; }
      .divider { height:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- CONFIG -----------------
FALLBACK_URL = "https://i.redd.it/43axcjdu59nd1.jpeg"
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/128.0.0.0 Safari/537.36")}

# ---- helper: normalization ----
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

# ----------------- IMAGE SEARCH (players) -----------------
def _tokens_for_sim(s: str) -> set:
    s = _simplify_team(s)  # also OK for names
    return set(t for t in s.split() if t)

def _sim(a: str, b: str) -> float:
    ta, tb = _tokens_for_sim(a), _tokens_for_sim(b)
    if not ta or not tb: return 0.0
    inter = len(ta & tb); union = len(ta | tb)
    return inter / max(1, union)

def _score_name_team(player_q: str, team_q: str | None, name: str, team: str) -> float:
    s = 6.0 * _sim(player_q, name)
    last = player_q.split()[-1] if player_q else ""
    if last and last in name: s += 2.0
    if team_q: s += 4.0 * _sim(team_q, team)
    return s

# SofaScore SEARCH
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
                out.append({"id": pid, "name": name, "team": team})
            if out:
                return out
        except Exception:
            continue
    return []

# SofaScore player info (for badge/flag)
@st.cache_data(show_spinner=False, ttl=60*60*24)
def sofascore_player_info(pid: int) -> dict | None:
    for base in ("https://api.sofascore.com", "https://api.sofascore.app"):
        try:
            r = requests.get(f"{base}/api/v1/player/{pid}", headers=UA, timeout=7)
            if not r.ok: 
                continue
            return r.json().get("player") or {}
        except Exception:
            continue
    return None

def _best_sofascore_match(player_name: str, team_name: str) -> dict | None:
    pn_raw = player_name.replace(".", " ").strip()
    tn_raw = team_name.strip()
    surname = pn_raw.split()[-1] if pn_raw else ""
    team_first = _first_word(tn_raw)
    queries = [
        f"{team_first} {surname}".strip(),
        f"{tn_raw} {pn_raw}".strip(),
        f"{surname} {tn_raw}".strip(),
        pn_raw, surname,
    ]
    player_q = _norm(pn_raw)
    team_q = _norm(tn_raw)

    best, best_score = None, -1.0
    for q in queries:
        plist = sofascore_search_players(q)
        for p in plist:
            name = _norm(p.get("name", ""))
            team = _norm(p.get("team", ""))
            pid  = p.get("id")
            s = _score_name_team(player_q, team_q, name, team)
            if s > best_score and pid:
                best_score, best = s, p
        if best is not None and best_score >= 3.0:
            break
    return best

# HEADSHOT resolvers (return url, "Source")
@st.cache_data(show_spinner=False, ttl=60*60*24)
def sofascore_headshot(player_name: str, team_name: str) -> tuple[str | None, str]:
    m = _best_sofascore_match(player_name, team_name)
    if not m: return None, ""
    pid = m["id"]
    for base in ("https://api.sofascore.com", "https://api.sofascore.app", "https://www.sofascore.com"):
        url = f"{base}/api/v1/player/{pid}/image"
        if _fetch_image_as_data_uri(url):  # just check reachable
            return url, "SofaScore"
    return None, ""

@st.cache_data(show_spinner=False, ttl=60*60*24)
def sofifa_headshot(player_name: str, team_name: str) -> tuple[str | None, str]:
    try:
        q = f"{_first_word(team_name)} {player_name}".strip()
        r = requests.get("https://sofifa.com/search", params={"keyword": q}, headers=UA, timeout=7)
        if not r.ok: return None, ""
        soup = BeautifulSoup(r.text, "html.parser")
        a = soup.select_one("a[href^='/player/']")
        if not a: return None, ""
        href = a.get("href", "")
        m = re.search(r"/player/(\d+)", href)
        if not m: return None, ""
        pid = int(m.group(1))
        a3 = f"{pid // 1000:03d}"; b3 = f"{pid % 1000:03d}"
        return f"https://cdn.sofifa.net/players/{a3}/{b3}/25_120.png", "SoFIFA"
    except Exception:
        return None, ""

@st.cache_data(show_spinner=False, ttl=60*60*24)
def fifacm_headshot(player_name: str, team_name: str) -> tuple[str | None, str]:
    try:
        q = f"{_first_word(team_name)} {player_name}".strip()
        r = requests.get("https://www.fifacm.com/players", params={"name": q}, headers=UA, timeout=7)
        if not r.ok: return None, ""
        soup = BeautifulSoup(r.text, "html.parser")
        a = soup.select_one("a[href^='/player/']")
        if not a: return None, ""
        m = re.search(r"/player/(\d+)", a.get("href",""))
        if not m: return None, ""
        eaid = m.group(1)
        return f"https://cdn.fifacm.com/players/{eaid}.png", "FIFACM"
    except Exception:
        return None, ""

@st.cache_data(show_spinner=False, ttl=60*60*24)
def fotmob_headshot(player_name: str, team_name: str) -> tuple[str | None, str]:
    try:
        pn = player_name.replace(".", " ").strip()
        tn = team_name.strip()
        surname = pn.split()[-1] if pn else ""
        team_first = _first_word(tn)
        queries = [f"{team_first} {surname}".strip(), f"{tn} {pn}".strip(), f"{surname} {tn}".strip(), pn, surname]
        pq = _norm(pn); tq = _norm(tn) if tn else None
        best, best_score = None, -1.0
        for q in queries:
            r = requests.get("https://www.fotmob.com/api/search", params={"q": q}, headers=UA, timeout=6)
            if not r.ok: continue
            for p in (r.json() or {}).get("players", []):
                name = _norm(p.get("name", "")); team = _norm(p.get("teamName", ""))
                pid = p.get("id")
                s = _score_name_team(pq, tq, name, team)
                if s > best_score:
                    best_score, best = s, pid
            if best is not None and best_score >= 3.0:
                break
        if best:
            return f"https://images.fotmob.com/image_resources/playerimages/{best}.png", "FotMob"
    except Exception:
        return None, ""
    return None, ""

@st.cache_data(show_spinner=False, ttl=60*60*24)
def transfermarkt_headshot(player_name: str, team_name: str) -> tuple[str | None, str]:
    try:
        q = f"{_first_word(team_name)} {player_name}".strip()
        r = requests.get("https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche", params={"query": q}, headers=UA, timeout=8)
        if not r.ok: return None, ""
        soup = BeautifulSoup(r.text, "html.parser")
        a = soup.select_one("a[href*='/profil/spieler/']")
        if not a: return None, ""
        href = a.get("href","")
        if not href.startswith("http"): href = "https://www.transfermarkt.com" + href
        r2 = requests.get(href, headers=UA, timeout=8)
        if not r2.ok: return None, ""
        soup2 = BeautifulSoup(r2.text, "html.parser")
        og = soup2.select_one("meta[property='og:image']")
        if og and og.get("content"):
            return og.get("content"), "Transfermarkt"
    except Exception:
        return None, ""
    return None, ""

@st.cache_data(show_spinner=False, ttl=60*60*24)
def flashscore_headshot(player_name: str, team_name: str) -> tuple[str | None, str]:
    try:
        q = f"{_first_word(team_name)} {player_name}".strip()
        r = requests.get("https://www.flashscore.com/search/", params={"q": q}, headers=UA, timeout=8)
        if not r.ok: return None, ""
        soup = BeautifulSoup(r.text, "html.parser")
        a = soup.select_one("a[href*='/player/']")
        if not a: return None, ""
        href = a.get("href","")
        if not href.startswith("http"): href = "https://www.flashscore.com" + href
        r2 = requests.get(href, headers=UA, timeout=8)
        if not r2.ok: return None, ""
        soup2 = BeautifulSoup(r2.text, "html.parser")
        og = soup2.select_one("meta[property='og:image']")
        if og and og.get("content"):
            return og.get("content"), "Flashscore"
    except Exception:
        return None, ""
    return None, ""

@st.cache_data(show_spinner=False, ttl=60*60*24)
def wiki_headshot(player_name: str) -> tuple[str | None, str]:
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",
                         params={"action":"query","format":"json","prop":"pageimages",
                                 "piprop":"thumbnail","pithumbsize":256,
                                 "titles": f"{player_name} footballer"},
                         headers=UA, timeout=7)
        pages = (r.json().get("query",{}) or {}).get("pages",{})
        for _,page in pages.items():
            thumb = (page.get("thumbnail") or {}).get("source")
            if thumb: return thumb, "Wikipedia"
    except Exception:
        return None, ""
    return None, ""

def resolve_player_image(player_name: str, team_name: str, mapping: dict | None, allow_wiki: bool) -> tuple[str, str]:
    # mapping override first
    if mapping:
        url = mapping.get(player_name) or mapping.get(player_name.strip())
        if url:
            data = _fetch_image_as_data_uri(url)
            return (data or url), "Mapping"

    for resolver in (sofascore_headshot, sofifa_headshot, fifacm_headshot, fotmob_headshot,
                     transfermarkt_headshot, flashscore_headshot):
        url, src = resolver(player_name, team_name)
        if url:
            data = _fetch_image_as_data_uri(url)
            return (data or url), src
    if allow_wiki:
        url, src = wiki_headshot(player_name)
        if url:
            data = _fetch_image_as_data_uri(url)
            return (data or url), src
    return FALLBACK_URL, "Fallback"

# ----------------- BADGE + FLAG -----------------
@st.cache_data(show_spinner=False, ttl=60*60*24)
def resolve_badge_and_flag(player_name: str, team_name: str, nationality_val: str | None = None) -> tuple[str | None, str | None]:
    """
    Returns (badge_data_uri_or_url, flag_data_uri_or_url).
    Strategy:
      - Try SofaScore (player info → team.id & country.alpha2).
      - Fallback badge: Wikipedia team page image.
      - Fallback flag: use nationality_val (if provided) with flagcdn best-effort.
    """
    # 1) SofaScore enrich
    m = _best_sofascore_match(player_name, team_name)
    if m and m.get("id"):
        info = sofascore_player_info(m["id"])
        if info:
            badge, flag = None, None
            team_id = (info.get("team") or {}).get("id")
            if team_id:
                burl = f"https://api.sofascore.com/api/v1/team/{team_id}/image"
                badge = _fetch_image_as_data_uri(burl) or burl
            country = (info.get("country") or {})
            cc = (country.get("alpha2") or "").lower()
            if cc:
                furl = f"https://flagcdn.com/w40/{cc}.png"
                flag = _fetch_image_as_data_uri(furl) or furl
            if badge or flag:
                return badge, flag

    # 2) Wikipedia team page image for badge
    badge = None
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",
                         params={"action":"query","format":"json","prop":"pageimages",
                                 "piprop":"thumbnail","pithumbsize":64,"titles": team_name},
                         headers=UA, timeout=7)
        pages = (r.json().get("query",{}) or {}).get("pages",{})
        for _,page in pages.items():
            thumb = (page.get("thumbnail") or {}).get("source")
            if thumb:
                badge = _fetch_image_as_data_uri(thumb) or thumb
                break
    except Exception:
        pass

    # 3) Flag from provided nationality (if any)
    flag = None
    if nationality_val:
        cc_guess = None
        v = _norm(nationality_val)
        # quick map for common names → ISO (extend as you like)
        special = {
            "england":"gb-eng", "scotland":"gb-sct", "wales":"gb-wls", "northern ireland":"gb-nir"
        }
        if v in special:
            cc_guess = special[v]
            furl = f"https://flagcdn.com/w40/{cc_guess}.png"
            flag = _fetch_image_as_data_uri(furl) or furl
        else:
            # try country name directly (FlagCDN expects ISO, but many country names are redirected by cloudfront)
            # best-effort: no strict guarantee
            for iso in (v[:2], v.replace(" ", "-")[:10]):
                if not iso or len(iso) < 2: continue
                furl = f"https://flagcdn.com/w40/{iso}.png"
                got = _fetch_image_as_data_uri(furl)
                if got:
                    flag = got
                    break

    return badge, flag

# ----------------- DATA LOADING & FILTERS (same as before; trimmed for brevity) -----------------
REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Contract expires","Goals"}
FEATURES = [
    'Defensive duels per 90','Defensive duels won, %','Aerial duels per 90','Aerial duels won, %',
    'PAdj Interceptions','Non-penalty goals per 90','xG per 90','Shots per 90','Shots on target, %',
    'Goal conversion, %','Crosses per 90','Accurate crosses, %','Dribbles per 90','Successful dribbles, %',
    'Head goals per 90','Key passes per 90','Touches in box per 90','Progressive runs per 90','Accelerations per 90',
    'Passes per 90','Accurate passes, %','xA per 90','Passes to penalty area per 90','Accurate passes to penalty area, %',
    'Deep completions per 90','Smart passes per 90',
]

@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame: return pd.read_csv(path_str)
@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame: return pd.read_csv(io.BytesIO(data))

def load_df(csv_name: str = "WORLDJUNE25.csv") -> pd.DataFrame:
    for p in [Path.cwd()/csv_name]:
        if p.exists(): return _read_csv_from_path(str(p))
    st.warning(f"Could not find **{csv_name}**. Upload below.")
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if up is None: st.stop()
    return _read_csv_from_bytes(up.getvalue())

df = load_df()

with st.sidebar:
    st.header("Filters")
    leagues = sorted(df["League"].dropna().unique().tolist())
    leagues_sel = st.multiselect("Leagues", leagues, default=leagues)
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
    st.subheader("Images")
    use_wikipedia = st.checkbox("Allow Wikipedia (last resort)", value=True)
    show_img_source = st.checkbox("Show image source tag", value=True)
    mapping_file = st.file_uploader("Optional headshot mapping CSV (Player,ImageURL)", type=["csv"])
    user_map = None
    if mapping_file is not None:
        try:
            mdf = pd.read_csv(mapping_file)
            if {"Player","ImageURL"}.issubset(mdf.columns):
                user_map = dict(zip(mdf["Player"].astype(str), mdf["ImageURL"].astype(str)))
                st.caption(f"Loaded {len(user_map)} image mappings.")
            else:
                st.warning("CSV must have columns: Player, ImageURL")
        except Exception as e:
            st.warning(f"Could not read mapping CSV: {e}")

# ---- validation
missing = [c for c in REQUIRED_BASE if c not in df.columns]
if missing: st.error(f"Dataset missing required columns: {missing}"); st.stop()
for c in FEATURES:
    if c not in df.columns:
        st.error(f"Dataset missing feature column: {c}"); st.stop()

# ---- filter pool
df_f = df[df["League"].isin(leagues_sel)].copy()
df_f = df_f[df_f["Minutes played"].between(min_minutes, max_minutes)]
df_f = df_f[df_f["Age"].between(min_age, max_age)]
if apply_contract: df_f = df_f[df_f["Contract expires"].dt.year <= cutoff_year]
for c in FEATURES: df_f[c] = pd.to_numeric(df_f[c], errors="coerce")
df_f = df_f.dropna(subset=FEATURES)
if df_f.empty: st.warning("No players after filters."); st.stop()

# ---- percentiles per league
for feat in FEATURES:
    df_f[f"{feat} Percentile"] = df_f.groupby("League")[feat].transform(lambda x: x.rank(pct=True) * 100.0)

# ---- scoring
COMBINED_METRICS = {
    'Non-penalty goals per 90':3,'Shots per 90':1.5,'xG per 90':3,'Touches in box per 90':1,'Shots on target, %':0.5,
    'Passes per 90':2,'Passes to penalty area per 90':1.5,'Deep completions per 90':1,'Smart passes per 90':1.5,
    'Accurate passes, %':1.5,'Key passes per 90':1,'Dribbles per 90':2,'Successful dribbles, %':1,
    'Progressive runs per 90':2,'xA per 90':3
}
def combined_role_score(df_in: pd.DataFrame, metrics: dict) -> pd.Series:
    total_w = sum(metrics.values()) if metrics else 1.0
    wsum = np.zeros(len(df_in))
    for m, w in metrics.items():
        col = f"{m} Percentile"
        if col in df_in.columns: wsum += df_in[col].values * w
    return wsum / total_w

df_f["Combined Score"] = combined_role_score(df_f, COMBINED_METRICS)
def age_bonus(a: float) -> int:
    table = {27:0,26:1,25:3,24:3,23:4,22:5,21:6,20:6,19:7,18:8,17:9,16:10}
    a = int(a) if not pd.isna(a) else 27
    return 0 if a>=28 else (9 if a<16 else table.get(a,0))
df_f["Overall Rating"] = 0.6 * df_f["Combined Score"] + 0.4 * 50  # simple baseline; tune if you had strengths
df_f["Potential"] = df_f.apply(lambda r: r["Overall Rating"] + age_bonus(r["Age"]), axis=1)
df_f["Contract Year"] = pd.to_datetime(df_f["Contract expires"], errors="coerce").dt.year.fillna(0).astype(int)

ranked = df_f.sort_values("Overall Rating", ascending=False).head(int(top_n)).copy().reset_index(drop=True)

# ---- colors
PALETTE = [(0,(208,2,27)),(50,(245,166,35)),(65,(248,231,28)),(75,(126,211,33)),(85,(65,117,5)),(100,(40,90,4))]
def _lerp(a, b, t): return tuple(int(round(a[i] + (b[i]-a[i]) * t)) for i in range(3))
def rating_color(v: float) -> str:
    v = max(0.0, min(100.0, float(v)))
    for i in range(len(PALETTE)-1):
        x0,c0=PALETTE[i]; x1,c1=PALETTE[i+1]
        if v <= x1:
            t = 0 if x1==x0 else (v-x0)/(x1-x0); r,g,b = _lerp(c0,c1,t); return f"rgb({r},{g},{b})"
    r,g,b = PALETTE[-1][1]; return f"rgb({r},{g},{b})"

# ---- position coloring
BLUE = {"CF","LW","RW"}
GREEN = {"LWF","LAMF","RWF","RAMF","AMF"}
def render_pos_chips(pos_text: str) -> str:
    parts = [p.strip().upper() for p in re.split(r"[,/]", pos_text or "") if p.strip()]
    html = []
    for p in parts:
        cls = "blue" if p in BLUE else ("green" if p in GREEN else "")
        html.append(f"<span class='chip {cls}'>{p}</span>")
    return " ".join(html) if html else "<span class='chip'>—</span>"

# ----------------- RENDER -----------------
for idx, row in ranked.iterrows():
    rank = idx + 1
    player = str(row.get("Player","")) or ""
    team = str(row.get("Team","")) or ""
    pos = str(row.get("Position","")) or ""
    nationality = str(row.get("Nationality","")) if "Nationality" in row.index else None

    age = int(row.get("Age",0)) if not pd.isna(row.get("Age",np.nan)) else 0
    overall_i = int(round(float(row["Overall Rating"])))
    potential_i = int(round(float(row["Potential"])))
    contract_year = int(row.get("Contract Year",0))

    # Player headshot (multi-source)
    head_uri, src_tag = resolve_player_image(player, team, mapping=user_map, allow_wiki=use_wikipedia)
    avatar_style = f"background-image: url('{head_uri}');"

    # Badge + Flag
    badge_uri, flag_uri = resolve_badge_and_flag(player, team, nationality)
    badge_img = f"<img class='icon' src='{badge_uri}'/>" if badge_uri else ""
    flag_img  = f"<img class='icon' src='{flag_uri}'/>"  if flag_uri  else ""

    ov_style = f"background:{rating_color(overall_i)};"
    po_style = f"background:{rating_color(potential_i)};"
    src_html = f"<div class='source'>📸 {src_tag}</div>" if show_img_source else ""
    pos_html = render_pos_chips(pos)

    st.markdown(f"""
    <div class='wrap'>
      <div class='player-card'>
        <div class='leftcol'>
          <div class='avatar' style="{avatar_style}"></div>
          {src_html}
          <div class='row'>
            {flag_img}<span class='chip'>{age}y.o.</span>
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
        </div>
        <div class='rank'>#{rank}</div>
      </div>
    </div>
    <div class='divider'></div>
    """, unsafe_allow_html=True)



















