# app_top20_tiles.py — Top 20 Tiles with multi-source headshots
# Requirements: streamlit, pandas, numpy, requests, beautifulsoup4, lxml

import os
import io
import re
import math
import base64
import unicodedata
from pathlib import Path
from functools import lru_cache

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
    "Headshots resolved: SofaScore → Sofifa → FIFACM → FotMob → Wikipedia → fallback."
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
        background-repeat:no-repeat, no-repeat;
        background-position:center center, center center;
        background-size:cover, cover;
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
UA = {"User-Agent": "tiles-app/1.0 (+streamlit)"}

# --- (keep INCLUDED_LEAGUES, FEATURES, ROLES, COMBINED_METRICS, LEAGUE_STRENGTHS, DATA LOADING, SIDEBAR, FILTERING) ---

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
def sofascore_headshot(player_name: str, team_name: str | None = None) -> tuple[str | None, str]:
    """Search SofaScore public API and return (image_url, 'SofaScore') if found."""
    try:
        q = f"{player_name} {team_name or ''}".strip()
        r = requests.get("https://api.sofascore.com/api/v1/search/all", params={"q": q}, headers=UA, timeout=7)
        if not r.ok:
            return None, ""
        data = r.json()
        for p in data.get("players", {}).get("items", []):
            pid = p.get("id")
            if pid:
                url = f"https://api.sofascore.app/api/v1/player/{pid}/image"
                return url, "SofaScore"
    except Exception:
        return None, ""
    return None, ""

# ---------- SoFIFA ----------
@st.cache_data(show_spinner=False, ttl=60*60*24)
def sofifa_headshot(player_name: str, team_name: str | None = None) -> tuple[str | None, str]:
    try:
        q = f"{player_name} {team_name or ''}".strip()
        r = requests.get("https://sofifa.com/search", params={"keyword": q}, headers=UA, timeout=7)
        if not r.ok:
            return None, ""
        soup = BeautifulSoup(r.text, "html.parser")
        a = soup.select_one("a[href^='/player/']")
        if not a:
            return None, ""
        m = re.search(r"/player/(\d+)", a.get("href", ""))
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
        m = re.search(r"/player/(\d+)", a.get("href", ""))
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
        queries = [f"{pn_raw} {tn_raw}".strip(), pn_raw]
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

# ---------- Master Resolver ----------
def resolve_player_image_url(player_name: str, team_name: str | None) -> tuple[str, str]:
    for resolver in (sofascore_headshot, sofifa_headshot, fifacm_headshot, fotmob_headshot, wiki_headshot):
        url, src = resolver(player_name, team_name)
        if url:
            data_uri = _fetch_image_as_data_uri(url)
            if data_uri:
                return data_uri, src
            else:
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

def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i]-a[i]) * t)) for i in range(3))

def rating_color(v: float) -> str:
    v = max(0.0, min(100.0, float(v)))
    for i in range(len(PALETTE) - 1):
        x0, c0 = PALETTE[i]; x1, c1 = PALETTE[i+1]
        if v <= x1:
            t = 0 if x1 == x0 else (v - x0) / (x1 - x0)
            r,g,b = _lerp(c0, c1, t); return f"rgb({r},{g},{b})"
    r,g,b = PALETTE[-1][1]; return f"rgb({r},{g},{b})"

# ----------------- RENDER -----------------
# If you didn't define this in the sidebar, default to True (show labels as requested)
show_img_source = globals().get("show_img_source", True)

for idx, row in ranked.iterrows():
    rank = idx + 1
    player = str(row.get("Player", "")) or ""
    team = str(row.get("Team", "")) or ""
    pos = str(row.get("Position", "")) or ""
    age = int(row.get("Age", 0)) if not pd.isna(row.get("Age", np.nan)) else 0
    overall_i = int(round(float(row["Overall Rating"])))
    potential_i = int(round(float(row["Potential"])))
    contract_year = int(row.get("Contract Year", 0))

    # Resolve image (data: URI if fetched, else fallback URL) + src tag
    primary, src_tag = resolve_player_image_url(player, team)
    avatar_style = f"background-image: url('{primary}');"

    ov_style = f"background:{rating_color(overall_i)};"
    po_style = f"background:{rating_color(potential_i)};"

    # Source label under the picture
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
















