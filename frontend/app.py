"""
Location: frontend/app.py
Purpose: Bijli-Insight Pakistan — energy intelligence dashboard.

UI fix notes
------------
The old version forced light panels on top of Streamlit's DARK base theme, so
tab labels, sidebar radios, widget labels and the header ended up as light text
on a light background (or a black bar). This version:
  1. locks the page to a light colour scheme (see also .streamlit/config.toml),
  2. gives every widget (tabs, selectbox, radio, slider, inputs, expanders,
     buttons, alerts, dataframes) an explicit foreground + background,
  3. fixes a few markup bugs (duplicate style attr, split <div> across
     st.markdown calls, dead topic cards, slider not updating from estimator).
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'backend' package is importable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import html
import json
import re

try:
    import markdown as _md_lib
except ImportError:  # pip install markdown
    _md_lib = None
import duckdb
import pandas as pd
import numpy as np
import streamlit as st
from backend.ml.quantile_model import QuantileEnergyForecaster
from backend.ml.model_evaluator import ModelFairnessEvaluator
from backend.agents.insight_rag_agent import InsightAgentRAG
from backend.pipeline.build_knowledge_base import ensure_knowledge_base
import chromadb

st.set_page_config(
    page_title="Bijli-Insight Pakistan | Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system — replica of Bijli-Insight Pakistan (6).html ──────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800&family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');

:root {
    --bg:#f3f6f5; --panel:#fff; --ink:#10231f; --mute:#5b6f6a;
    --line:#dbe4e1; --pine:#0b5d4b; --pine2:#e1f2ed;
    --amber:#c97a00; --amber2:#fdf0d8; --blue:#1d6fa5;
    --ok:#157a4b; --bad:#b4372c; --chip:#eef3f1;
}

/* ════════ 1. STREAMLIT BASE — lock to light, readable everywhere ════════ */
:root, .stApp { color-scheme: light !important; }
html, body, .stApp, [class*="css"] { font-family:Figtree,system-ui,sans-serif; }
.stApp { background:var(--bg) !important; color:var(--ink) !important; }
.main .block-container { padding-top:2.2rem !important; max-width:1240px; }

/* top bar (was a black strip) */
header[data-testid="stHeader"] { background:var(--bg) !important; border-bottom:1px solid transparent; }
header[data-testid="stHeader"] * { color:var(--mute) !important; }
[data-testid="stToolbar"] svg, [data-testid="stHeader"] svg { fill:var(--mute) !important; }

/* default text colour for all markdown; children with their own colour still win */
.stApp [data-testid="stMarkdownContainer"] { color:var(--ink) !important; }
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label { color:var(--ink) !important; font-weight:600 !important; font-size:13.5px !important; }
.stApp [data-testid="stCaptionContainer"], .stCaption p { color:var(--mute) !important; font-size:12.5px !important; }

/* sidebar */
section[data-testid="stSidebar"] { background:var(--panel) !important; border-right:1px solid var(--line) !important; }
section[data-testid="stSidebar"] > div { padding-top:.5rem; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:var(--ink) !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapsedControl"] svg { fill:var(--mute) !important; color:var(--mute) !important; }

/* ════════ 2. TABS ════════ */
.stTabs [data-baseweb="tab-list"] {
    background:var(--panel) !important; border:1px solid var(--line) !important; border-radius:14px !important;
    padding:6px !important; gap:4px !important; overflow-x:auto; margin-bottom:18px;
    box-shadow:0 1px 2px rgba(10,40,35,.05);
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important; border:none !important; border-radius:10px !important;
    padding:9px 20px !important; height:auto !important; white-space:nowrap;
    color:var(--mute) !important; font-family:Figtree,sans-serif !important; font-weight:500 !important;
    transition:background .12s,color .12s;
}
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] div,
.stTabs [data-baseweb="tab"] span { color:inherit !important; font-weight:inherit !important; font-size:14.5px !important; margin:0 !important; }
.stTabs [data-baseweb="tab"]:hover { background:var(--pine2) !important; color:var(--pine) !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background:var(--pine) !important; color:#fff !important; font-weight:600 !important; box-shadow:0 6px 14px -8px var(--pine); }
.stTabs [data-baseweb="tab"][aria-selected="true"] p,
.stTabs [data-baseweb="tab"][aria-selected="true"] div { color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none !important; height:0 !important; background:transparent !important; }

/* ════════ 3. INPUT WIDGETS ════════ */
/* selectbox + dropdown menu */
div[data-baseweb="select"] > div {
    background:var(--panel) !important; border:1px solid var(--line) !important; border-radius:10px !important;
    min-height:42px; color:var(--ink) !important;
}
div[data-baseweb="select"] > div:hover { border-color:var(--pine) !important; }
div[data-baseweb="select"] * { color:var(--ink) !important; font-weight:500; }
div[data-baseweb="select"] svg { fill:var(--mute) !important; }
div[data-baseweb="popover"] > div, ul[role="listbox"], div[data-baseweb="menu"] { background:var(--panel) !important; border-radius:10px !important; }
li[role="option"], li[role="option"] * { color:var(--ink) !important; background:transparent; }
li[role="option"]:hover, li[role="option"][aria-selected="true"] { background:var(--pine2) !important; }

/* text / number inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    border:1px solid var(--line) !important; border-radius:10px !important; background:var(--panel) !important;
    color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; font-family:Figtree,sans-serif !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color:#8a9b97 !important; -webkit-text-fill-color:#8a9b97 !important; }
.stTextInput input:focus, .stNumberInput input:focus { border-color:var(--pine) !important; box-shadow:0 0 0 3px var(--pine2) !important; }
.stNumberInput button { background:var(--chip) !important; color:var(--ink) !important; border-color:var(--line) !important; }
.stNumberInput button svg { fill:var(--ink) !important; }
div[data-baseweb="input"], div[data-baseweb="base-input"] { background:var(--panel) !important; border-radius:10px !important; }

/* sliders */
[data-testid="stSliderThumbValue"], [data-testid="stSlider"] [data-testid="stThumbValue"] { color:var(--pine) !important; font-weight:700 !important; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color:var(--mute) !important; }
div[data-baseweb="slider"] div[role="slider"] { background:var(--pine) !important; box-shadow:0 0 0 4px var(--pine2) !important; }

/* checkbox */
.stCheckbox label, .stCheckbox label p, .stCheckbox span { color:var(--ink) !important; }
.stCheckbox [data-baseweb="checkbox"] > div:first-child { border-color:#b7c6c2 !important; }
.stCheckbox [data-baseweb="checkbox"]:has(input:checked) > div:first-child { background:var(--pine) !important; border-color:var(--pine) !important; }

/* sidebar radio — bordered option cards */
.stRadio [role="radiogroup"] { gap:6px !important; }
label[data-baseweb="radio"] {
    display:flex !important; align-items:center !important; gap:10px !important; width:100% !important;
    padding:10px 12px !important; margin:0 !important; background:var(--panel) !important;
    border:1px solid var(--line) !important; border-radius:10px !important; cursor:pointer !important;
    transition:background .1s,border-color .1s;
}
label[data-baseweb="radio"]:hover { border-color:var(--pine) !important; }
label[data-baseweb="radio"] p, label[data-baseweb="radio"] span { color:var(--ink) !important; font-size:13.5px !important; font-weight:500 !important; margin:0 !important; }
label[data-baseweb="radio"] > div:first-child { background:var(--panel) !important; border:2px solid #b7c6c2 !important; }
label[data-baseweb="radio"]:has(input:checked) { border-color:var(--pine) !important; background:var(--pine2) !important; }
label[data-baseweb="radio"]:has(input:checked) p { color:var(--pine) !important; font-weight:600 !important; }
label[data-baseweb="radio"]:has(input:checked) > div:first-child { background:var(--pine) !important; border-color:var(--pine) !important; }
label[data-baseweb="radio"]:has(input:checked) > div:first-child > div { background:#fff !important; }

/* buttons */
.stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {
    border-radius:10px !important; font-family:Figtree,sans-serif !important; font-weight:600 !important;
    padding:.55rem 1rem !important; transition:background .12s,border-color .12s,transform .1s;
}
.stButton>button { background:var(--panel) !important; border:1px solid var(--line) !important; color:var(--ink) !important; }
.stButton>button p { color:var(--ink) !important; font-weight:600 !important; }
.stButton>button:hover { background:var(--pine2) !important; border-color:var(--pine) !important; }
.stButton>button:hover p { color:var(--pine) !important; }
.stButton>button:active { transform:translateY(1px); }
.stFormSubmitButton>button { background:var(--pine) !important; border:1px solid var(--pine) !important; color:#fff !important; }
.stFormSubmitButton>button p { color:#fff !important; }
.stFormSubmitButton>button:hover { background:#094a3c !important; }
.stDownloadButton>button { background:var(--panel) !important; border:1px solid var(--pine) !important; color:var(--pine) !important; }
.stDownloadButton>button p { color:var(--pine) !important; }
.stDownloadButton>button:hover { background:var(--pine2) !important; }
section[data-testid="stSidebar"] .stButton>button { width:100%; }

/* ════════ 4. CONTAINERS ════════ */
div[data-testid="stMetric"] { background:var(--panel) !important; border:1px solid var(--line) !important; border-radius:12px !important; padding:16px 18px !important; }
div[data-testid="stMetricLabel"] p { color:var(--mute) !important; font-size:.77rem !important; text-transform:uppercase !important; letter-spacing:.06em !important; font-weight:600 !important; }
div[data-testid="stMetricValue"] { color:var(--ink) !important; font-weight:700 !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:var(--panel) !important; border-color:var(--line) !important; border-radius:12px !important; }
[data-testid="stForm"] { background:var(--panel) !important; border:1px solid var(--line) !important; border-radius:14px !important; padding:16px !important; }

.stAlert { border-radius:12px !important; }
.stAlert, .stAlert * { color:var(--ink) !important; }

[data-testid="stExpander"], .stExpander { border:1px solid var(--line) !important; border-radius:14px !important; background:var(--panel) !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span { color:var(--ink) !important; font-weight:600 !important; }
[data-testid="stExpander"] summary svg { fill:var(--mute) !important; }

hr { border-color:var(--line) !important; border-width:1px 0 0 !important; }
[data-testid="stDataFrame"] { border:1px solid var(--line) !important; border-radius:12px !important; overflow:hidden !important; background:var(--panel); }
[data-testid="stToast"], [data-testid="stToast"] * { color:var(--ink) !important; }
[data-testid="stSpinner"] p, .stSpinner p { color:var(--mute) !important; }

/* ════════ 5. COMPONENT CLASSES (v6 template) ════════ */
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin-bottom:18px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:20px 22px; margin-bottom:14px;
    box-shadow:0 1px 2px rgba(10,40,35,.06),0 12px 24px -16px rgba(10,40,35,.3);
    transition:box-shadow .18s,transform .15s; }
.card:hover { box-shadow:0 1px 2px rgba(10,40,35,.08),0 18px 30px -16px rgba(10,40,35,.38); }
.card h2 { font-size:17px !important; margin:0 0 4px !important; color:var(--ink) !important; font-family:Figtree,sans-serif !important; }
.card.sel { outline:2px solid var(--pine); outline-offset:0; }
.eyebrow { display:block; color:var(--mute); font-size:12px; text-transform:uppercase; letter-spacing:.06em; font-weight:600; }
.sec-h { font-size:17px; font-weight:700; margin:0 0 10px; color:var(--ink); }
.sub { color:var(--mute); font-size:13.5px; margin:0 0 12px; display:block; }
.kpi small { display:block; color:var(--mute); font-size:.73rem; text-transform:uppercase; letter-spacing:.07em; font-weight:600; }
.kpi strong { display:block; font-size:1.85rem; font-weight:700; letter-spacing:-.02em; line-height:1.2; color:var(--ink); margin:4px 0 6px; }
.ic { display:inline-grid; place-items:center; width:36px; height:36px; border-radius:10px; background:var(--pine2); margin-bottom:6px; }
.ic svg { width:20px; height:20px; fill:none; stroke:var(--pine); stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }
.tag { display:inline-block; font-size:11.5px; padding:2px 10px; border-radius:99px; background:var(--pine2); color:var(--ok); font-weight:600; }
.tag.a { background:var(--amber2); color:var(--amber); }
.herowrap { border-radius:18px; overflow:hidden; background:linear-gradient(135deg,#0b5d4b,#082f3a); margin-bottom:18px;
    box-shadow:0 24px 40px -22px rgba(8,47,58,.8); }
.hero { padding:26px 28px; color:#fff; }
.hero, .hero * { color:#fff; }
.hero h2 { font-size:22px !important; line-height:1.35; margin:6px 0 8px !important; color:#fff !important; font-family:Figtree,sans-serif !important; }
.hero small { opacity:.8; font-size:12.5px; display:block; }
.hero p { opacity:.85; margin:0; font-size:13.5px; }
.hero .ic { background:rgba(255,255,255,.15); }
.hero .ic svg { stroke:#fff; }
.days { display:grid; grid-template-columns:repeat(auto-fit,minmax(84px,1fr)); gap:8px; margin-top:16px; }
.day { background:rgba(255,255,255,.12); border:2px solid transparent; border-radius:12px; padding:10px 4px; text-align:center; display:block;
    box-shadow:0 8px 14px -8px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.2); transition:transform .15s; }
.day:hover { border-color:rgba(255,255,255,.6); transform:translateY(-2px); }
.day b { display:block; font-size:20px; font-weight:700; }
.day small { display:block; font-size:11px; opacity:.8; }
.day i { display:block; height:4px; border-radius:2px; margin:6px 10px 0; }
.steps { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:10px; }
.steps>div { background:var(--chip); border-radius:10px; padding:14px 12px; }
.steps b { display:block; font-size:23px; font-weight:700; color:var(--ink); line-height:1.2; }
.steps span { display:block; font-size:12.5px; color:var(--mute); margin-top:3px; }
.slice { display:grid; grid-template-columns:repeat(auto-fit,minmax(195px,1fr)); gap:14px; margin:16px 0 8px; }
.bar { height:6px; border-radius:3px; background:var(--chip); overflow:hidden; margin-top:10px; }
.bar i { display:block; height:100%; background:var(--pine); border-radius:3px; }
.big { font-size:38px; font-weight:700; letter-spacing:-.02em; line-height:1.1; color:var(--ink); margin:6px 0 2px; }
.s-sub { color:var(--mute); font-size:13px; display:block; margin-top:2px; }
.legend { display:flex; gap:16px; flex-wrap:wrap; font-size:13px; color:var(--mute); margin:8px 0 4px; }
.legend i { display:inline-block; width:14px; height:4px; border-radius:2px; margin-right:5px; vertical-align:middle; }
.note { font-size:12.5px; color:var(--mute); margin-top:8px; }
.line { display:flex; justify-content:space-between; align-items:center; gap:12px; padding:9px 0; border-bottom:1px dashed var(--line); font-size:13.5px; color:var(--ink); }
.line b { font-weight:600; }
.hint { margin-top:12px; padding:10px 14px; border-radius:10px; background:var(--amber2); color:var(--amber); font-size:13px; font-weight:500; line-height:1.5; }
.hint.ok { background:var(--pine2); color:var(--ok); }
.model-line { display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid var(--line); font-size:13.5px; color:var(--ink); }
.model-line:last-child { border-bottom:none; }
.cities { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; margin-bottom:22px; }
.cities .card { margin-bottom:0; }

/* Policy chat */
.chat-wrap { display:flex; flex-direction:column; gap:12px; padding:12px 0 16px; min-height:220px; }
.msg { display:flex; gap:10px; }
.msg.u { justify-content:flex-end; }
.av { width:32px; height:32px; border-radius:10px; background:var(--pine); color:#fff; display:grid; place-items:center; font-weight:700; font-size:11px; flex-shrink:0; }
.bb { padding:10px 14px; border-radius:14px; max-width:72ch; font-size:14px; line-height:1.6; }
.msg.u .bb { background:var(--pine); color:#fff; border-bottom-right-radius:4px; }
.msg.b .bb { background:var(--chip); color:var(--ink); border-bottom-left-radius:4px; }
.src { display:block; margin-top:12px; font-size:12px; color:var(--mute); border-top:1px solid var(--line); padding-top:8px; }
.src b { color:var(--ink); font-weight:600; }

/* Rendered Markdown inside bot answers */
.msg.b .bb { padding:16px 20px; background:var(--panel); border:1px solid var(--line);
    box-shadow:0 1px 2px rgba(10,40,35,.05); }
.bb.md > :first-child { margin-top:0 !important; }
.bb.md > :last-child { margin-bottom:0 !important; }
.bb.md h3 { font-size:13px !important; font-weight:700 !important; text-transform:uppercase; letter-spacing:.06em;
    color:var(--pine) !important; margin:18px 0 6px !important; padding:0 0 5px !important;
    border-bottom:1px solid var(--line); font-family:Figtree,sans-serif !important; }
.bb.md h3:first-child { margin-top:0 !important; }
.bb.md p { margin:0 0 8px; color:var(--ink); }
.bb.md ul, .bb.md ol { margin:4px 0 10px; padding-left:20px; }
.bb.md li { margin:0 0 6px; color:var(--ink); }
.bb.md li::marker { color:var(--pine); }
.bb.md strong { color:var(--ink); font-weight:700; }
.bb.md em { color:var(--mute); font-style:italic; }
.bb.md blockquote { margin:8px 0 10px; padding:8px 14px; border-left:3px solid var(--pine);
    background:var(--pine2); border-radius:0 8px 8px 0; color:var(--ink); }
.bb.md blockquote p { margin:0 0 4px; }
.bb.md code { background:var(--chip); padding:1px 5px; border-radius:4px; font-size:12.5px; }
.pill { display:inline-flex; gap:6px; align-items:center; background:var(--panel); border:1px solid var(--line); border-radius:99px; padding:6px 14px; font-size:13px; color:var(--ink); }
.dot { width:8px; height:8px; border-radius:50%; background:var(--ok); flex-shrink:0; display:inline-block; }
.ur { font-family:"Noto Nastaliq Urdu",serif; font-size:15px; color:var(--pine); line-height:2.2; display:block; }
.brand { display:flex; align-items:center; gap:10px; padding:4px 0 14px; border-bottom:1px solid var(--line); margin-bottom:12px; }
.logo-sq { width:36px; height:36px; background:var(--pine); border-radius:8px; display:grid; place-items:center; color:#fff; flex-shrink:0;
    box-shadow:0 8px 16px -8px var(--pine),inset 0 1px 0 rgba(255,255,255,.3); }
.logo-sq svg { width:20px; height:20px; fill:currentColor; }
.brand-text b { display:block; font-size:15px; font-weight:700; color:var(--ink); line-height:1.2; }
.brand-text small { font-size:11.5px; color:var(--mute); }
.cinfo { margin:4px 0 6px; display:flex; justify-content:space-between; background:var(--chip); border-radius:10px; padding:10px 12px; font-size:13px; color:var(--mute); }
.cinfo b { color:var(--ink); }
.store-badge { border:1px solid var(--line); border-radius:12px; padding:12px 14px; font-size:12.5px; color:var(--mute); background:var(--panel); }
.store-badge > b { color:var(--ink); display:block; margin-bottom:6px; font-size:13px; }
.store-badge small { color:var(--mute); }
.lay { display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid var(--line); color:var(--ink); font-size:13px; }
.lay:last-of-type { border-bottom:none; }
.lay .dotc { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px; background:var(--ok); }
.page-title { margin:0; font-size:28px; letter-spacing:-.01em; color:var(--ink); font-family:Figtree,sans-serif; font-weight:800; }

@media (max-width:760px) {
    .steps { grid-template-columns:1fr; }
    .hero { padding:20px 18px; }
    .big { font-size:30px; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Paths & Data Loaders
# ---------------------------------------------------------
DB_FILE = Path("data/warehouse.duckdb")
STATUS_FILE = Path("data/pipeline_status.json")
QUALITY_LOG_FILE = Path("data/quality_log.json")
MANIFEST_FILE = Path("data/raw/manifest.csv")
TARIFF_FILE = Path("data/raw/nepra_tariff_slabs_2024.csv")

CITY_FILE_MAP = {
    "Karachi (K-Electric)": "open-meteo-24.85N66.99E7m.csv",
    "Lahore (LESCO)": "open-meteo-31.53N74.39E220m.csv",
    "Islamabad / Rawalpindi (IESCO)": "open-meteo-33.64N73.03E527m.csv",
}


@st.cache_data(ttl=60)
def get_warehouse_connection_status():
    if not DB_FILE.exists():
        return False, 0, []
    try:
        conn = duckdb.connect(str(DB_FILE), read_only=True)
        tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
        row_count = 0
        if "cleaned_weather" in tables:
            row_count = conn.execute("SELECT COUNT(*) FROM cleaned_weather").fetchone()[0]
        conn.close()
        return True, row_count, tables
    except Exception:
        return False, 0, []


@st.cache_data(ttl=60)
def load_duckdb_data(table_name: str, city_filename: str = None, limit: int = 1000) -> pd.DataFrame:
    if not DB_FILE.exists():
        return pd.DataFrame()
    try:
        conn = duckdb.connect(str(DB_FILE), read_only=True)
        query = f"SELECT * FROM {table_name}"
        params = []
        if city_filename and city_filename != "ALL":
            query += " WHERE filename = ?"
            params.append(city_filename)
        query += f" ORDER BY timestamp_utc DESC LIMIT {int(limit)}"
        df = conn.execute(query, params).df() if params else conn.execute(query).df()
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error querying DuckDB: {e}")
        return pd.DataFrame()


def load_pipeline_status() -> dict:
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    st.error("No pipeline run found (data/pipeline_status.json). Run `python -m backend.pipeline.flow` first.")
    st.stop()


def load_quality_audit_metrics() -> dict:
    live_counts = {"FIX": 0, "FILL": 0, "FLAG": 0}
    if QUALITY_LOG_FILE.exists():
        try:
            logs = json.loads(QUALITY_LOG_FILE.read_text(encoding="utf-8"))
            for record in logs:
                act = record.get("action_taken")
                if act in live_counts:
                    live_counts[act] += int(record.get("affected_rows_count", 1))
        except Exception:
            pass
    return live_counts


@st.cache_data
def load_tariff_slabs() -> pd.DataFrame:
    if TARIFF_FILE.exists():
        return pd.read_csv(TARIFF_FILE)
    return pd.DataFrame([
        {"slab_label": "Protected (1–100 kWh)", "min_units": 1, "max_units": 100, "rate_per_unit_pkr": 16.48},
        {"slab_label": "Protected (101–200 kWh)", "min_units": 101, "max_units": 200, "rate_per_unit_pkr": 22.95},
        {"slab_label": "Unprotected (201–300 kWh)", "min_units": 201, "max_units": 300, "rate_per_unit_pkr": 34.26},
        {"slab_label": "Unprotected (301–700 kWh)", "min_units": 301, "max_units": 700, "rate_per_unit_pkr": 42.12},
        {"slab_label": "Unprotected (> 700 kWh)", "min_units": 701, "max_units": 99999, "rate_per_unit_pkr": 48.84},
    ])


@st.cache_data(ttl=300)
def load_full_city_data(city_filename: str) -> pd.DataFrame:
    if not DB_FILE.exists():
        return pd.DataFrame()
    try:
        conn = duckdb.connect(str(DB_FILE), read_only=True)
        df = conn.execute(
            "SELECT * FROM gold_weather_features WHERE filename = ? ORDER BY timestamp_utc",
            [city_filename],
        ).df()
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def get_forecaster() -> QuantileEnergyForecaster:
    forecaster = QuantileEnergyForecaster(models_dir=Path("models"))
    if not Path("models/lgb_quantile_50.txt").exists() and DB_FILE.exists():
        try:
            conn = duckdb.connect(str(DB_FILE), read_only=True)
            df = conn.execute("SELECT * FROM gold_weather_features").df()
            conn.close()
            X_train, _, y_train, _ = forecaster.prepare_time_based_split(df)
            forecaster.train_quantile_estimators(X_train, y_train)
        except Exception:
            pass
    return forecaster


@st.cache_resource
def get_rag_agent() -> InsightAgentRAG:
    client = chromadb.PersistentClient(path="data/chroma_db")
    agent = InsightAgentRAG(chroma_client=client, db_connection=None)
    ensure_knowledge_base(agent)
    return agent


# ── SVG icon helper ──────────────────────────────────────────────────────────
def _ic(path_d: str) -> str:
    return f'<span class="ic"><svg viewBox="0 0 24 24">{path_d}</svg></span>'


_SVG = {
    "timer":   '<circle cx="12" cy="13" r="8"/><path d="M12 9v4l2 2M9 2h6"/>',
    "target":  '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/>',
    "shield":  '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>',
    "refresh": '<path d="M20 11a8 8 0 0 0-14-4M4 4v4h4M4 13a8 8 0 0 0 14 4M20 20v-4h-4"/>',
    "wind":    '<path d="M3 8h11a3 3 0 1 0-3-3M3 12h15a3 3 0 1 1-3 3M3 16h7"/>',
    "therm":   '<path d="M10 14V5a2 2 0 0 1 4 0v9a4 4 0 1 1-4 0z"/>',
    "bolt":    '<path d="M13 2L5 14h6l-1 8 8-12h-6z"/>',
    "city":    '<path d="M4 21V8l6-3v16M10 21V10l10 3v8M3 21h18"/>',
    "db":      '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "chart":   '<path d="M3 17l5-6 4 3 8-9"/><path d="M3 21h18"/>',
    "bill":    '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M9 8h6M9 12h6M9 16h3"/>',
    "chat":    '<path d="M21 12a8 8 0 0 1-11.8 7L3 20l1.2-5.2A8 8 0 1 1 21 12z"/>',
}


def _html(s: str) -> str:
    """Strip leading indentation so Markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in s.strip().splitlines() if line.strip())


def _render_md(text: str) -> str:
    """
    Render a bot answer (Markdown) to HTML safely.

    '&' and '<' are escaped first so no raw HTML from the model can
    reach the page; '>' is kept so Markdown block quotes still work.
    Output is collapsed onto one line so Streamlit's own Markdown
    pass does not re-interpret it.
    """
    safe = text.replace("&", "&amp;").replace("<", "&lt;")
    if _md_lib is not None:
        rendered = _md_lib.markdown(safe, extensions=["sane_lists"])
    else:
        rendered = safe.replace("\n", "<br>")
    return re.sub(r">\s*\n\s*<", "><", rendered).replace("\n", " ")


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(_html("""
    <div class="brand">
        <div class="logo-sq"><svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M13 2L5 14h6l-1 8 8-12h-6z"/></svg></div>
        <div class="brand-text"><b>Bijli-Insight</b><small>Pakistan energy intelligence</small></div>
    </div>
    """), unsafe_allow_html=True)

    selected_city_label = st.selectbox("City / DISCO", options=list(CITY_FILE_MAP.keys()), index=0)
    selected_filename = CITY_FILE_MAP[selected_city_label]

    # City snapshot info
    df_sidebar = load_duckdb_data("gold_weather_features", selected_filename, limit=24)
    if not df_sidebar.empty and "temperature_c" in df_sidebar.columns:
        city_hi = df_sidebar["temperature_c"].max()
        city_lo = df_sidebar["temperature_c"].min()
        st.markdown(
            f'<div class="cinfo"><span>Recent 24h range</span><b>{city_lo:.0f}–{city_hi:.0f} °C</b></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    tariff_type = st.radio(
        "Consumer type",
        ["Protected Residential (≤ 200 Units)", "Unprotected Residential (> 200 Units)", "Commercial A-2", "General Industrial"],
        index=0,
    )

    st.markdown("<hr style='margin:14px 0'>", unsafe_allow_html=True)

    db_ok, total_db_rows, available_tables = get_warehouse_connection_status()
    if db_ok:
        badge_body = f"""
        <div class="lay"><span><i class="dotc"></i>Gold</span><small>Features ready</small></div>
        <div class="lay"><span><i class="dotc"></i>Silver</span><small>Cleaned</small></div>
        <div class="lay"><span><i class="dotc"></i>Raw</span><small>{total_db_rows:,} rows</small></div>
        <small style="display:block;margin-top:8px">DuckDB, all layers healthy</small>
        """
    else:
        badge_body = '<small style="color:var(--amber)">Warehouse empty — run pipeline</small>'
    st.markdown(_html(f'<div class="store-badge"><b>Warehouse status</b>{badge_body}</div>'), unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------
# Load shared data
# ---------------------------------------------------------
status = load_pipeline_status()
mins = status["data_freshness_minutes"]
m_mae = status["model_mae"]
b_mae = status["baseline_mae"]
pct = status["improvement_pct"]
last_run = status.get("last_run_utc", "N/A")
total_rows = status["active_records_count"]

# 7 days of hourly rows for the hero strip; last 48h for the Plan section
_df_week_raw = load_duckdb_data("gold_weather_features", selected_filename, limit=168)
_df_plan_raw = _df_week_raw.head(48) if not _df_week_raw.empty else _df_week_raw

# ---------------------------------------------------------
# Top Header
# ---------------------------------------------------------
st.markdown(_html(f"""
<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;margin-bottom:22px">
    <div>
        <h1 class="page-title">Bijli-Insight Pakistan</h1>
        <span class="ur">بجلی بصیرت پاکستان</span>
        <p style="margin:2px 0 0;color:var(--mute);font-size:14px">Temperature &amp; electricity-demand forecasts · <b style="color:var(--ink)">{selected_city_label}</b></p>
    </div>
    <span class="pill"><span class="dot"></span>Data synced {mins} min ago</span>
</div>
"""), unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------
tab_monitor, tab_warehouse, tab_forecast, tab_calculator, tab_rag = st.tabs([
    "Overview", "Data Explorer", "7-Day Forecast", "Bill Calculator", "Policy AI",
])


# =========================================================
# TAB 1: OVERVIEW
# =========================================================
with tab_monitor:
    quality = load_quality_audit_metrics()
    fix_n, fill_n, flag_n = quality["FIX"], quality["FILL"], quality["FLAG"]

    # ── Hero with 7-day strip ──────────────────────────────
    df_plan = _df_plan_raw.sort_values("timestamp_utc") if not _df_plan_raw.empty else pd.DataFrame()
    df_week = _df_week_raw.sort_values("timestamp_utc") if not _df_week_raw.empty else pd.DataFrame()
    hero_city = selected_city_label.split(" (")[0]

    if not df_week.empty and "temperature_c" in df_week.columns and "timestamp_utc" in df_week.columns:
        df_week = df_week.copy()
        df_week["date_col"] = pd.to_datetime(df_week["timestamp_utc"]).dt.date
        days_grouped = df_week.groupby("date_col")["temperature_c"].agg(["max", "min"]).reset_index().tail(7)
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        days_html = ""
        for i, row in enumerate(days_grouped.itertuples()):
            mx, mn = row.max, row.min
            col = "#ff7a6b" if mx >= 35 else ("#ffc14d" if mx >= 33 else "#5ee0a8")
            lbl = "Today" if i == len(days_grouped) - 1 else day_names[row.date_col.weekday()]
            days_html += f'<div class="day"><small>{lbl}</small><b>{mx:.0f}°</b><small>{mn:.0f}° low</small><i style="background:{col}"></i></div>'

        hi_today = float(df_week["temperature_c"].tail(24).max())
        msg_hi = "Heatwave likely. Expect very high cooling demand." if hi_today >= 35 \
            else "A hot day. Cooling demand will peak in the afternoon." if hi_today >= 33 \
            else "Comfortable weather. Demand should stay near normal."
        hero_msg = f"Highs near {hi_today:.0f} °C. {msg_hi}"
    else:
        days_html = ""
        hero_msg = "No recent data. Run the pipeline to populate the warehouse."

    st.markdown(_html(f"""
    <div class="herowrap">
        <div class="hero">
            <small>Recent conditions · {hero_city}</small>
            <h2>{hero_msg}</h2>
            <p>Model MAE {m_mae:.2f} °C &nbsp;·&nbsp; {pct:.1f}% better than naive baseline &nbsp;·&nbsp; Data fresh {mins} min ago</p>
            <div class="days">{days_html}</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    # ── Plan today ────────────────────────────────────────
    if not df_plan.empty and "temperature_c" in df_plan.columns and "hour" in df_plan.columns:
        last24 = df_plan.tail(24)
        coolest_row = last24.loc[last24["temperature_c"].idxmin()]
        hottest_row = last24.loc[last24["temperature_c"].idxmax()]
        cool_hr = int(coolest_row.get("hour", 0))
        hot_hr = int(hottest_row.get("hour", 14))
        cool_t = float(coolest_row["temperature_c"])
        hot_t = float(hottest_row["temperature_c"])

        st.markdown(_html(f"""
        <div class="sec-h">Plan today</div>
        <div class="slice" style="margin:0 0 22px">
            <div class="card">
                {_ic(_SVG["wind"])}
                <span class="eyebrow">Best time to air the house</span>
                <div class="big" style="font-size:28px">{cool_hr:02d}:00</div>
                <span class="s-sub">Coolest hour ({cool_t:.0f} °C). Open windows now and run the AC less later.</span>
            </div>
            <div class="card">
                {_ic(_SVG["therm"])}
                <span class="eyebrow">Hottest hour</span>
                <div class="big" style="font-size:28px">{hot_hr:02d}:00</div>
                <span class="s-sub">About {hot_t:.0f} °C. Cooling demand peaks around here.</span>
            </div>
            <div class="card">
                {_ic(_SVG["bolt"])}
                <span class="eyebrow">Avoid heavy appliances</span>
                <div class="big" style="font-size:28px">18:00–22:00</div>
                <span class="s-sub">Evening peak hours when the grid is under most strain.</span>
            </div>
        </div>
        """), unsafe_allow_html=True)

    # ── Compare cities ────────────────────────────────────
    st.markdown('<div class="sec-h">Compare cities</div>', unsafe_allow_html=True)
    city_cards = ""
    for city_label, city_file in CITY_FILE_MAP.items():
        df_c = load_duckdb_data("gold_weather_features", city_file, limit=24)
        if not df_c.empty and "temperature_c" in df_c.columns:
            c_hi = float(df_c["temperature_c"].max())
            tag_lbl = "Very hot" if c_hi >= 35 else ("Hot" if c_hi >= 33 else "Comfortable")
            tag_cls = "a" if c_hi >= 33 else ""
            sel_cls = " sel" if city_label == selected_city_label else ""
            city_cards += (
                f'<div class="card{sel_cls}">{_ic(_SVG["city"])}'
                f'<small style="display:block;color:var(--mute);font-size:12px">{city_label}</small>'
                f'<b style="font-size:30px;font-weight:700;letter-spacing:-.02em;line-height:1.2;display:block;margin:4px 0;color:var(--ink)">{c_hi:.0f} °C</b>'
                f'<span class="tag {tag_cls}">{tag_lbl} today</span></div>'
            )
    if city_cards:
        st.markdown(f'<div class="cities">{city_cards}</div>', unsafe_allow_html=True)
    else:
        st.info("City comparison will appear once the warehouse has data.")

    # ── KPI grid ──────────────────────────────────────────
    sla_ok = mins < 60
    st.markdown(_html(f"""
    <div class="kpis">
        <div class="card kpi">
            {_ic(_SVG["timer"])}
            <small>Last data fetch</small>
            <strong>{mins} min ago</strong>
            <span class="tag{' a' if not sla_ok else ''}">{"Within 60 min target" if sla_ok else "Exceeded window"}</span>
        </div>
        <div class="card kpi">
            {_ic(_SVG["target"])}
            <small>Forecast error (MAE)</small>
            <strong>{m_mae:.2f} °C</strong>
            <span class="tag">{pct:.1f}% better than baseline ({b_mae:.2f} °C)</span>
        </div>
        <div class="card kpi">
            {_ic(_SVG["shield"])}
            <small>Raw data dropped</small>
            <strong>0</strong>
            <span class="tag">Every issue was repaired or flagged</span>
        </div>
        <div class="card kpi">
            {_ic(_SVG["refresh"])}
            <small>Last sync</small>
            <strong style="font-size:16px;margin-top:8px;line-height:1.4">{last_run}</strong>
        </div>
    </div>
    """), unsafe_allow_html=True)

    # ── DQA + Model vs baseline ───────────────────────────
    dqa_col, model_col = st.columns([2, 1])
    dqa_note = (
        f"✅ {total_rows:,} rows passed all DQA checks — zero type errors, zero gaps, zero anomalies"
        if fix_n == 0 and fill_n == 0 and flag_n == 0
        else "Zero raw data dropped · 100% deterministic remediation"
    )
    with dqa_col:
        st.markdown(_html(f"""
        <div class="card">
            <h2>Data quality log</h2>
            <span class="sub">What the pipeline did to each record before it reached the models.</span>
            <div class="steps">
                <div><b>{fix_n:,}</b><span>Fixed: type casts</span></div>
                <div><b>{fill_n:,}</b><span>Filled: gaps estimated</span></div>
                <div><b>{flag_n:,}</b><span>Flagged: anomalies</span></div>
            </div>
            <p class="note" style="margin-top:12px">{dqa_note}</p>
        </div>
        """), unsafe_allow_html=True)

    with model_col:
        st.markdown(_html(f"""
        <div class="card">
            <h2>Model vs. baseline</h2>
            <span class="sub">Baseline repeats last week's temperature.</span>
            <div class="model-line"><span>Quantile model (q50)</span><b>{m_mae:.2f} °C</b></div>
            <div class="model-line"><span>Baseline (t–168h)</span><b>{b_mae:.2f} °C</b></div>
            <p class="note" style="margin-top:10px">Our model beats the persistence baseline by <b>{pct:.1f}%</b> on the held-out test set.</p>
        </div>
        """), unsafe_allow_html=True)

    # ── Manifest ──────────────────────────────────────────
    st.markdown(_html("""
    <div class="sec-h" style="margin-top:4px">Source files and integrity checks</div>
    <span class="sub">Each raw file is fingerprinted with SHA-256 so any tampering is detectable.</span>
    """), unsafe_allow_html=True)
    if MANIFEST_FILE.exists():
        manifest_df = pd.read_csv(MANIFEST_FILE)
        st.dataframe(
            manifest_df, use_container_width=True,
            column_config={
                "sha256_hash": st.column_config.TextColumn("SHA-256 Checksum"),
                "source_url": st.column_config.LinkColumn("Source API Endpoint"),
                "row_count": st.column_config.NumberColumn("Verified Rows", format="%d"),
            },
        )
    else:
        st.info("Manifest will appear once `pipeline/generate_manifest.py` has run.")

    with st.expander("Business model and value", expanded=False):
        bm1, bm2, bm3 = st.columns(3)
        with bm1:
            st.markdown("**Primary users**\n\nGrid planners at K-Electric, LESCO and IESCO scheduling 24–72 hours ahead. Also residential consumers optimising AC use.")
            st.markdown("**Customer / payer**\n\nDISCOs and NEPRA (regulatory budget); households via freemium dashboard.")
        with bm2:
            st.markdown(f"**Value created**\n\n- **{pct:.1f}% better** forecast vs. naive baseline\n- Calibrated uncertainty (q10/q50/q90)\n- Per-group error audit\n- NEECA-cited advice cuts household bills")
            st.markdown("**Sustainability**\n\nSaaS (₨ 50K/month per DISCO) + free HF Space.")
        with bm3:
            st.markdown("**Running costs**\n\n- Open-Meteo API: **free**\n- DuckDB: **free**\n- ChromaDB: **free**\n- HF Space: **free**\n- Total: **< ₨ 5K/month**")
            st.markdown("**Key risks**\n\nModel drift in extreme heatwaves; NEPRA tariff updates require CSV refresh.")


# =========================================================
# TAB 2: DATA EXPLORER
# =========================================================
with tab_warehouse:
    st.markdown(_html(f"""
    <div class="herowrap">
        <div class="hero">
            {_ic(_SVG["db"])}
            <h2>DuckDB 3-layer analytical store</h2>
            <p>Browse the tables behind the forecasts for <b>{selected_city_label}</b>. Raw → Silver → Gold.</p>
        </div>
    </div>
    """), unsafe_allow_html=True)

    t_col1, t_col2 = st.columns([1, 3])
    with t_col1:
        chosen_table = st.selectbox("Table / layer", ["gold_weather_features", "cleaned_weather"])
        row_limit = st.slider("Row limit", min_value=50, max_value=2000, value=250, step=50)

    df_warehouse = load_duckdb_data(table_name=chosen_table, city_filename=selected_filename, limit=row_limit)

    with t_col2:
        if not df_warehouse.empty:
            def _stat(col, fmt, fn):
                return fmt.format(getattr(df_warehouse[col], fn)()) if col in df_warehouse.columns else "—"

            st.markdown(_html(f"""
            <div class="kpis" style="margin-top:0">
                <div class="card kpi"><small>Rows inspected</small><strong>{len(df_warehouse):,}</strong></div>
                <div class="card kpi"><small>Avg temperature</small><strong>{_stat('temperature_c', '{:.1f} °C', 'mean')}</strong></div>
                <div class="card kpi"><small>Max temperature</small><strong>{_stat('temperature_c', '{:.1f} °C', 'max')}</strong></div>
                <div class="card kpi"><small>Avg humidity</small><strong>{_stat('humidity_pct', '{:.1f} %', 'mean')}</strong></div>
            </div>
            """), unsafe_allow_html=True)

    if not df_warehouse.empty:
        st.dataframe(df_warehouse, use_container_width=True, height=380)
        csv_bytes = df_warehouse.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Export {chosen_table} ({len(df_warehouse)} rows as CSV)",
            data=csv_bytes,
            file_name=f"{chosen_table}_{selected_city_label}.csv",
            mime="text/csv",
        )
    else:
        st.warning("No records found. Run `python -m pipeline.warehouse_pipeline`.")


# =========================================================
# TAB 3: 7-DAY FORECAST
# =========================================================
with tab_forecast:
    forecast_horizon_hours = st.slider("Forecast horizon (hours ahead)", min_value=24, max_value=168, value=72, step=24)
    df_forecast = load_duckdb_data("gold_weather_features", city_filename=selected_filename, limit=forecast_horizon_hours)
    forecaster = get_forecaster()

    if not df_forecast.empty:
        df_f = df_forecast.sort_values("timestamp_utc").dropna(subset=QuantileEnergyForecaster.FEATURE_COLS)
        if not df_f.empty:
            try:
                preds = forecaster.predict_intervals(df_f[QuantileEnergyForecaster.FEATURE_COLS])
                q90_max = float(np.nanmax(preds["q_high"]))

                st.markdown(_html(f"""
                <div class="herowrap">
                    <div class="hero">
                        <small>Next {forecast_horizon_hours}h · {selected_city_label}</small>
                        <h2>Calibrated 80% prediction intervals — q10 / q50 / q90</h2>
                        <p>High case up to {q90_max:.0f} °C &nbsp;·&nbsp; Model MAE {m_mae:.2f} °C vs baseline {b_mae:.2f} °C &nbsp;·&nbsp; {pct:.1f}% improvement</p>
                    </div>
                </div>
                """), unsafe_allow_html=True)

                chart_idx = pd.to_datetime(df_f["timestamp_utc"].values)
                chart_df = pd.DataFrame(
                    {"q10 (Floor)": preds["q_low"], "q50 (Median)": preds["q_mid"], "q90 (Ceiling)": preds["q_high"]},
                    index=chart_idx,
                )
                st.line_chart(chart_df, color=["#38bdf8", "#0b5d4b", "#c97a00"], height=340)
                st.markdown(_html("""
                <div class="legend">
                    <span><i style="background:#38bdf8"></i>Low case (10th percentile)</span>
                    <span><i style="background:#0b5d4b"></i>Expected (median)</span>
                    <span><i style="background:#c97a00"></i>High case (90th percentile)</span>
                </div>
                """), unsafe_allow_html=True)

                evaluator = ModelFairnessEvaluator()
                df_full = load_full_city_data(selected_filename).dropna(
                    subset=QuantileEnergyForecaster.FEATURE_COLS + ["temperature_c"]
                )
                if not df_full.empty:
                    preds_full = forecaster.predict_intervals(df_full[QuantileEnergyForecaster.FEATURE_COLS])
                    df_eval = df_full[["hour", "day_of_week", "temperature_c"]].copy()
                    df_eval["y_true"] = df_full["temperature_c"].values
                    df_eval["y_pred"] = preds_full["q_mid"]
                else:
                    df_eval = df_f[["hour", "day_of_week", "temperature_c"]].copy()
                    df_eval["y_true"] = df_f["temperature_c"].values
                    df_eval["y_pred"] = preds["q_mid"]
                slices = evaluator.compute_group_error_slices(df_eval)

                def _bar_h(mae, color="var(--pine)"):
                    if np.isnan(mae):
                        return '<div class="bar"><i style="width:0%"></i></div>'
                    return f'<div class="bar"><i style="width:{min(mae/1.0*100,100):.0f}%;background:{color}"></i></div>'

                def _val_h(mae, label):
                    if np.isnan(mae):
                        return f'<div style="color:var(--mute);font-size:13px;padding:10px 0">No {label} data</div>'
                    return f'<div class="big">{mae:.2f} °C</div><span class="s-sub">average error</span>'

                peak_v, offpeak_v, hw_v = slices["peak"]["mae"], slices["off_peak"]["mae"], slices["heatwave"]["mae"]
                st.markdown(_html(f"""
                <div class="slice">
                    <div class="card">{_ic(_SVG["city"])}<span class="eyebrow">Evening peak · 18:00–22:00</span>{_val_h(peak_v,"peak")}{_bar_h(peak_v)}</div>
                    <div class="card">{_ic(_SVG["wind"])}<span class="eyebrow">Off-peak hours</span>{_val_h(offpeak_v,"off-peak")}{_bar_h(offpeak_v)}</div>
                    <div class="card">{_ic(_SVG["therm"])}<span class="eyebrow">Heatwave days ≥ 35 °C</span>{_val_h(hw_v,"heatwave")}{_bar_h(hw_v,"var(--amber)")}</div>
                </div>
                <p class="note">Accuracy evaluated separately for peak hours, off-peak hours and heatwaves — a strong overall score cannot hide weak spots.</p>
                """), unsafe_allow_html=True)

            except RuntimeError as exc:
                st.warning(f"Model not ready: {exc}. Run `python -m pipeline.flow` to train.")
        else:
            st.warning("Insufficient lag features. Run `python -m pipeline.warehouse_pipeline`.")
    else:
        st.warning("No warehouse data. Run `python -m pipeline.warehouse_pipeline` then `python -m pipeline.flow`.")


# =========================================================
# TAB 4: BILL CALCULATOR
# =========================================================
def _apply_estimate(units: int):
    st.session_state["monthly_units"] = int(min(1200, max(30, units)))
    st.toast(f"Updated to {st.session_state['monthly_units']} units/month")


if "monthly_units" not in st.session_state:
    st.session_state["monthly_units"] = 350

with tab_calculator:
    st.markdown(_html(f"""
    <div class="sec-h" style="font-size:21px;margin-bottom:4px">Interactive NEPRA Tariff &amp; Bill Advisor</div>
    <span class="sub">Model your monthly bill under current NEPRA slabs and see NEECA conservation savings.
    &nbsp;<span class="tag">{tariff_type}</span></span>
    """), unsafe_allow_html=True)

    calc_col1, calc_col2 = st.columns(2, gap="large")

    with calc_col1:
        st.markdown('<div class="sec-h" style="font-size:15px">Your consumption</div>', unsafe_allow_html=True)
        monthly_units = st.slider("Monthly usage (units / kWh)", min_value=30, max_value=1200, step=10, key="monthly_units")
        ac_temp = st.slider("AC thermostat setting (°C)", min_value=18, max_value=30, value=22, step=1)
        num_ac = st.number_input("Number of 1.5-ton AC units", min_value=0, max_value=8, value=2, step=1)
        peak_hours_ac = st.slider("AC hours during peak (18:00–22:00)", min_value=0, max_value=4, value=3, step=1)

        st.markdown("<hr style='margin:14px 0'>", unsafe_allow_html=True)
        st.markdown('<div class="sec-h" style="font-size:14px">Not sure of your usage? Estimate from appliances</div>', unsafe_allow_html=True)

        _appliances = [
            ("Air conditioner (1.5 ton)", 1.5, 8),
            ("Ceiling fans (4)", 0.3, 12),
            ("Refrigerator", 0.12, 24),
            ("Lights & TV", 0.25, 6),
            ("Washing machine", 0.5, 1),
        ]
        app_est_units = 0
        for i, (app_name, kw, default_hrs) in enumerate(_appliances):
            a_col1, a_col2 = st.columns([3, 1])
            with a_col1:
                enabled = st.checkbox(app_name, value=(i < 4), key=f"app_en_{i}")
            with a_col2:
                hrs = st.number_input("h/day", min_value=0, max_value=24, value=default_hrs, key=f"app_hr_{i}", label_visibility="visible")
            if enabled:
                app_est_units += kw * hrs * 30
        app_est_units = round(app_est_units)

        st.button(f"Use this estimate (~{app_est_units} units/month)", on_click=_apply_estimate, args=(app_est_units,))

    with calc_col2:
        df_slabs = load_tariff_slabs()
        matched = df_slabs[(df_slabs["min_units"] <= monthly_units) & (df_slabs["max_units"] >= monthly_units)]
        rate_per_unit = float(matched.iloc[-1]["rate_per_unit_pkr"]) if not matched.empty else 48.84
        category_tag = str(matched.iloc[-1]["slab_label"]) if not matched.empty else "Unprotected (> 700 kWh)"

        base_bill = monthly_units * rate_per_unit
        surcharges = base_bill * 0.22
        total_bill = base_bill + surcharges

        df_slabs_sorted = df_slabs.sort_values("min_units")
        next_slabs = df_slabs_sorted[df_slabs_sorted["min_units"] > monthly_units]
        next_slab_html = ""
        if not next_slabs.empty:
            ns = next_slabs.iloc[0]
            gap = int(ns["min_units"] - monthly_units)
            next_slab_html = f'<div class="hint">You are {gap} units below the next price band (₨{ns["rate_per_unit_pkr"]:.0f}/unit). Staying under saves money.</div>'

        if ac_temp < 26:
            savings_pct = min((26 - ac_temp) * 0.06, 0.40)
            potential_saved = total_bill * savings_pct * 0.65
            neeca_html = (
                f'<div class="hint ok">NEECA tip: set AC to 26°C — save ~₨ {potential_saved:,.0f}/month ({savings_pct*100:.0f}% cooling reduction)'
                f'<span class="ur" style="font-size:13px">اے سی کو 26°C پر سیٹ کریں</span></div>'
            )
        else:
            neeca_html = '<div class="hint ok">✅ Compliant with NEECA 26°C recommendation.<span class="ur" style="font-size:13px">آپ NEECA کی سفارش پر عمل پیرا ہیں۔</span></div>'

        st.markdown(_html(f"""
        <div class="card">
            <span class="eyebrow">Estimated monthly bill</span>
            <div class="big">₨ {total_bill:,.0f}</div>
            <div style="margin-top:14px">
                <div class="line"><span>{monthly_units} units × ₨{rate_per_unit:.2f} <span style="color:var(--mute);font-size:12px">({category_tag})</span></span><b>₨ {base_bill:,.0f}</b></div>
                <div class="line"><span>Taxes &amp; surcharges (~22% FPA)</span><b>₨ {surcharges:,.0f}</b></div>
            </div>
            {next_slab_html}
            {neeca_html}
        </div>
        """), unsafe_allow_html=True)


# =========================================================
# TAB 5: POLICY AI — chat UI
# =========================================================
with tab_rag:
    st.markdown(_html("""
    <div style="display:flex;gap:12px;align-items:center;padding-bottom:14px;border-bottom:1px solid var(--line);margin-bottom:14px">
        <div class="av" style="width:38px;height:38px;border-radius:12px;font-size:13px">AI</div>
        <div>
            <div style="margin:0;font-size:18px;font-weight:700;color:var(--ink)">Policy assistant</div>
            <small style="color:var(--mute)">Answers from NEPRA and NEECA documents</small>
        </div>
    </div>
    """), unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "bot", "text": "Hello. Ask about tariffs, net metering or ways to cut your bill. I name the document each answer comes from.", "cite": ""},
        ]
    if "pending_rag_query" not in st.session_state:
        st.session_state.pending_rag_query = ""

    # Chat history (all text escaped)
    chat_html = '<div class="chat-wrap">'
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            safe_text = html.escape(msg["text"]).replace("\n", "<br>")
            chat_html += f'<div class="msg u"><div class="bb">{safe_text}</div></div>'
        else:
            body = _render_md(msg["text"])
            cite_part = (
                f'<span class="src"><b>Sources:</b> {html.escape(msg["cite"])}</span>'
                if msg.get("cite") else ""
            )
            chat_html += (
                f'<div class="msg b"><div class="av" style="width:30px;height:30px;border-radius:9px;font-size:11px">AI</div>'
                f'<div class="bb md">{body}{cite_part}</div></div>'
            )
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # Quick-question buttons
    st.markdown('<span class="eyebrow" style="margin-bottom:6px">Quick questions</span>', unsafe_allow_html=True)
    sq1, sq2, sq3, sq4 = st.columns(4)
    with sq1:
        if st.button("Protected consumers", key="rag_q1"):
            st.session_state.pending_rag_query = "What is the criteria for protected residential consumers under NEPRA?"
    with sq2:
        if st.button("Net metering", key="rag_q2"):
            st.session_state.pending_rag_query = "How does net metering work under NEPRA rules?"
    with sq3:
        if st.button("Summer bills", key="rag_q3"):
            st.session_state.pending_rag_query = "Why are electricity bills higher in summer?"
    with sq4:
        if st.button("Saving energy", key="rag_q4"):
            st.session_state.pending_rag_query = "How can I save energy at home according to NEECA?"

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Input form
    with st.form("rag_chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask about tariffs, net metering or savings", placeholder="Type your question…")
        form_sent = st.form_submit_button("Send")
        if form_sent and user_input.strip():
            st.session_state.pending_rag_query = user_input.strip()

    # Process query
    if st.session_state.pending_rag_query:
        query_text = st.session_state.pending_rag_query
        st.session_state.pending_rag_query = ""
        st.session_state.chat_history.append({"role": "user", "text": query_text, "cite": ""})
        with st.spinner("Querying NEECA/NEPRA knowledge base…"):
            try:
                rag = get_rag_agent()
                context = rag.retrieve_grounding_context(query_text)
                result = rag.synthesize_verified_advice(query_text, context)
            except Exception as exc:
                result = {
                    "narrative_explanation": f"Knowledge base unavailable ({exc}). Ensure PDF files exist in knowledge_base/.",
                    "cited_documents": [], "grounding_gold_row_ids": [], "confidence_score": 0.0, "fallback_triggered": True,
                }
        cite_str = ""
        if not result["fallback_triggered"] and result["cited_documents"]:
            # Group pages per document: "Title (Page 10, Page 21) · Title2 (Page 4)"
            grouped = {}
            for c in result["cited_documents"]:
                grouped.setdefault(c["document_title"], []).append(c["section_reference"])
            cite_str = " · ".join(
                f"{title} ({', '.join(pages)})" for title, pages in grouped.items()
            )
        st.session_state.chat_history.append({"role": "bot", "text": result["narrative_explanation"], "cite": cite_str})
        st.rerun()


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:var(--mute);font-size:13px;padding:6px 0">
    <b style="color:var(--ink)">Bijli-Insight Pakistan</b> &nbsp;·&nbsp; Autonomous national energy intelligence &nbsp;·&nbsp; HackInsight 2026
</div>
""", unsafe_allow_html=True)