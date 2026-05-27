from pathlib import Path
import json
import sys
from datetime import datetime
import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR if (APP_DIR / "ml_artifacts").exists() else APP_DIR.parent
ARTIFACT_DIR = PROJECT_ROOT / "ml_artifacts"
DATASET_PATH = PROJECT_ROOT / "dataset_kemiskinan_jawa_barat.xlsx"
sys.path.insert(0, str(APP_DIR))
from prediction_service import predict_condition

st.set_page_config(page_title="Jabar Social Insight", layout="wide")

st.markdown("""
<style>
    :root {
        --background: #f8fafc; --foreground: #0f172a; --primary: #1e3a8a;
        --primary-hover: #1d4ed8; --secondary: #0f766e; --muted: #64748b;
        --border: #e2e8f0; --card: #ffffff; --danger: #b91c1c;
        --warning: #b45309; --success: #047857;
    }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
    .stApp { background: var(--background); }
    .block-container { padding: 2rem 1rem !important; max-width: 1280px !important; }

    *, *::before, *::after { box-sizing:border-box; }

    :root { font-size:16px; }
    .app-title { font-size:clamp(1.25rem,4vw,1.875rem); font-weight:700; letter-spacing:-0.025em; color:var(--foreground); margin:0; }
    .app-subtitle { font-size:clamp(0.75rem,2vw,0.875rem); color:var(--muted); margin-top:0.25rem; }
    .eyebrow { font-size:clamp(0.65rem,1.8vw,0.75rem); font-weight:600; text-transform:uppercase; letter-spacing:0.14em; color:var(--secondary); margin-bottom:0.25rem; }
    .section-title { font-size:clamp(1.15rem,3.5vw,1.5rem); font-weight:700; letter-spacing:-0.025em; color:var(--foreground); }
    .card-title { font-size:clamp(1rem,2.8vw,1.125rem); font-weight:600; color:var(--foreground); }
    .text-muted { font-size:clamp(0.8rem,2.2vw,0.875rem); color:var(--muted); line-height:1.5; }

    .card { background:var(--card); border:1px solid var(--border); border-radius:16px; box-shadow:0 1px 2px rgba(15,23,42,0.06); padding:clamp(1rem,3vw,1.5rem); margin-bottom:1rem; }
    .card-soft { box-shadow:0 1px 2px rgba(15,23,42,0.06),0 18px 48px rgba(15,23,42,0.06); border-color:#bfdbfe; }

    .metric-box { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:clamp(0.75rem,2vw,1rem) clamp(0.75rem,2.5vw,1.25rem); text-align:center; }
    .metric-label { font-size:clamp(0.65rem,1.8vw,0.75rem); font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); margin-bottom:0.25rem; }
    .metric-value { font-family:ui-monospace,SFMono-Regular,monospace; font-size:clamp(1.15rem,3.5vw,1.5rem); font-weight:700; letter-spacing:-0.025em; color:var(--primary); }

    .badge { display:inline-block; border-radius:9999px; padding:0.25rem 0.75rem; font-size:clamp(0.65rem,1.8vw,0.75rem); font-weight:600; border:1px solid; }
    .badge-low { background:#ecfdf5; color:#047857; border-color:#a7f3d0; }
    .badge-medium { background:#fffbeb; color:#b45309; border-color:#fde68a; }
    .badge-high { background:#fef2f2; color:#b91c1c; border-color:#fecaca; }
    .badge-blue { background:#eff6ff; color:#1e3a8a; border-color:#bfdbfe; }
    .badge-teal { background:#f0fdfa; color:#0f766e; border-color:#99f6e4; }

    div.stButton > button { border-radius:12px !important; font-weight:600 !important; font-size:clamp(0.8rem,2.2vw,0.875rem) !important; transition:all 0.2s ease-out !important; height:auto !important; border:1px solid var(--border) !important; min-height:44px !important; }
    div.stButton > button[kind="primary"] { background:var(--primary) !important; color:white !important; border-color:var(--primary) !important; }
    div.stButton > button[kind="primary"]:hover { background:var(--primary-hover) !important; }
    div.stButton > button[kind="secondary"] { background:var(--secondary) !important; color:white !important; border-color:var(--secondary) !important; }
    div.stButton > button[kind="secondary"]:hover { background:#0d9488 !important; }

    div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea { border-radius:12px !important; border:1px solid var(--border) !important; min-height:44px !important; font-size:clamp(0.8rem,2.2vw,0.875rem) !important; background:white !important; }
    div[data-testid="stNumberInput"] input:focus, div[data-testid="stTextInput"] input:focus { border-color:#93c5fd !important; box-shadow:0 0 0 4px rgba(30,58,138,0.1) !important; }
    div[data-testid="stNumberInput"] label, div[data-testid="stTextInput"] label { font-size:clamp(0.8rem,2.2vw,0.875rem) !important; font-weight:500 !important; color:#334155 !important; }
    div[data-testid="stNumberInput"] { margin-bottom:0.75rem !important; }

    section[data-testid="stSidebar"] { background:white !important; border-right:1px solid var(--border) !important; }
    section[data-testid="stSidebar"] .block-container { padding:2rem 1rem !important; }

    div[data-testid="stFileUploader"] { border:2px dashed var(--border) !important; border-radius:16px !important; padding:clamp(1rem,4vw,2rem) !important; background:white !important; text-align:center !important; }
    .custom-divider { height:1px; background:var(--border); margin:clamp(1rem,3vw,2rem) 0; border:none; }

    div[data-testid="stAlert"] { border-radius:12px !important; border:1px solid !important; background:#eff6ff !important; border-color:#bfdbfe !important; }
    div[data-testid="stAlert"]:has(> div > [kind="success"]) { background:#ecfdf5 !important; border-color:#a7f3d0 !important; }
    div[data-testid="stAlert"]:has(> div > [kind="error"]) { background:#fef2f2 !important; border-color:#fecaca !important; }

    div[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:12px !important; overflow:hidden !important; }
    div[data-testid="stDataFrame"] th { background:#f8fafc !important; font-size:clamp(0.65rem,1.8vw,0.75rem) !important; font-weight:600 !important; color:var(--muted) !important; text-transform:uppercase !important; letter-spacing:0.05em !important; }
    div[data-testid="stDataFrame"] td { font-size:clamp(0.8rem,2.2vw,0.875rem) !important; }

    div[role="radiogroup"] label { border-radius:8px !important; padding:0.5rem 0.75rem !important; font-size:clamp(0.8rem,2.2vw,0.875rem) !important; }
    h1,h2,h3,h4,h5,h6 { margin:0 !important; }
    .stAlert p { margin:0; }

    .sec { padding:clamp(2rem,5vw,4rem) 0; }
    .sec-white { background:white; }
    .sec-border-b { border-bottom:1px solid var(--border); }
    .sec-border-t { border-top:1px solid var(--border); }
    .sec-border-y { border-top:1px solid var(--border); border-bottom:1px solid var(--border); }

    .hero { padding:clamp(2rem,5vw,4rem) 0; background:white; border-bottom:1px solid var(--border); }
    .hero-grid { display:grid; grid-template-columns:1.05fr 0.95fr; gap:2.5rem; align-items:center; }
    .hero-h1 { font-size:clamp(1.5rem,5vw,3rem); font-weight:700; letter-spacing:-0.025em; line-height:1.15; color:var(--foreground); margin:0; }
    .hero-p { font-size:clamp(0.875rem,2.8vw,1.125rem); color:var(--muted); line-height:1.6; margin-top:0.75rem; }
    .hero-stat-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.75rem; margin-top:1.5rem; }
    .hero-stat { background:#f8fafc; border:1px solid var(--border); border-radius:12px; padding:clamp(0.5rem,1.5vw,0.75rem) clamp(0.75rem,2vw,1rem); }
    .hero-stat-lbl { font-size:clamp(0.6rem,1.6vw,0.7rem); font-weight:600; text-transform:uppercase; letter-spacing:0.12em; color:var(--muted); }
    .hero-stat-val { font-family:ui-monospace,monospace; font-size:clamp(0.95rem,2.8vw,1.125rem); font-weight:700; color:var(--foreground); margin-top:0.15rem; }

    .icon-box { width:44px; height:44px; display:flex; align-items:center; justify-content:center; border-radius:12px; font-size:1.25rem; }
    .icon-box-blue { background:#eff6ff; color:var(--primary); }
    .icon-box-teal { background:#f0fdfa; color:var(--secondary); }

    .section-header { margin-bottom:clamp(1rem,3vw,2rem); max-width:48rem; }

    .card-nx { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:clamp(1rem,3vw,1.25rem); box-shadow:0 1px 2px rgba(15,23,42,0.06); }
    .card-nx-soft { box-shadow:0 1px 2px rgba(15,23,42,0.06),0 18px 48px rgba(15,23,42,0.06); }
    .card-title-nx { font-size:clamp(1rem,2.8vw,1.125rem); font-weight:600; color:var(--foreground); line-height:1.3; }
    .card-desc { font-size:clamp(0.8rem,2.2vw,0.875rem); color:var(--muted); line-height:1.6; }
    .card-note { font-size:clamp(0.6rem,1.6vw,0.7rem); font-weight:600; text-transform:uppercase; letter-spacing:0.12em; color:var(--secondary); margin-top:1rem; }

    .workflow-card { position:relative; height:100%; }
    .workflow-arrow { position:absolute; right:-1.5rem; top:50%; transform:translateY(-50%); color:#cbd5e1; font-size:1.5rem; z-index:2; }

    .metric-icon-card { display:flex; align-items:flex-start; gap:1rem; }
    .metric-icon-box { width:clamp(36px,5vw,40px); height:clamp(36px,5vw,40px); border-radius:12px; background:#eff6ff; color:var(--primary); display:flex; align-items:center; justify-content:center; flex-shrink:0; font-size:clamp(1rem,3vw,1.25rem); }
    .metric-nx-label { font-size:clamp(0.7rem,2vw,0.8rem); font-weight:500; color:var(--muted); }
    .metric-nx-value { font-family:ui-monospace,monospace; font-size:clamp(1.15rem,3.5vw,1.5rem); font-weight:700; letter-spacing:-0.025em; color:var(--foreground); line-height:1.2; }
    .metric-nx-desc { font-size:clamp(0.75rem,2vw,0.85rem); color:var(--muted); }

    .empty-state { display:flex; flex-direction:column; align-items:center; justify-content:center; padding:clamp(1.5rem,5vw,3rem); border:2px dashed var(--border); border-radius:16px; background:white; text-align:center; }
    .empty-state-icon { font-size:clamp(1.5rem,5vw,2.5rem); margin-bottom:0.75rem; }
    .empty-state-title { font-size:clamp(1rem,2.8vw,1.125rem); font-weight:600; color:var(--foreground); }
    .empty-state-desc { font-size:clamp(0.8rem,2.2vw,0.875rem); color:var(--muted); margin-top:0.25rem; }

    .login-card { max-width:400px; margin:3rem auto; padding:2rem; background:white; border:1px solid var(--border); border-radius:20px; box-shadow:0 4px 24px rgba(15,23,42,0.08); }
    .login-title { font-size:1.25rem; font-weight:700; text-align:center; color:var(--foreground); margin-bottom:1.5rem; }

    .page-header { margin-bottom:clamp(1rem,3vw,2rem); }
    .page-header h1 { margin-top:0.75rem !important; }

    .threshold-card { padding:1.25rem; border-radius:12px; border:1px solid; text-align:center; }

    .detail-row { display:flex; justify-content:space-between; padding:0.5rem 0; border-bottom:1px solid #f1f5f9; }
    .detail-label { font-size:clamp(0.8rem,2.2vw,0.875rem); color:#64748b; }
    .detail-value { font-size:clamp(0.8rem,2.2vw,0.875rem); font-weight:500; color:#0f172a; }

    .stTabs [data-baseweb="tab-list"] { gap:0.5rem; flex-wrap:wrap !important; }
    .stTabs [data-baseweb="tab"] { border-radius:8px !important; padding:0.5rem 1rem !important; font-size:clamp(0.8rem,2.2vw,0.875rem) !important; }

    .sidebar-header { display:flex; align-items:center; gap:0.625rem; margin-bottom:0; }
    .sidebar-logo { width:36px; height:36px; background:#1e3a8a; border-radius:10px; display:flex; align-items:center; justify-content:center; color:white; font-weight:700; font-size:1rem; flex-shrink:0; }
    .sidebar-title { font-weight:700; font-size:1rem; color:#0f172a; }
    .sidebar-sub { font-size:0.7rem; color:#64748b; margin-top:-2px; }

    .delta-positive { color:#047857; }
    .delta-negative { color:#b91c1c; }
    .delta-neutral { color:#64748b; }

    .chip { display:inline-block; border-radius:9999px; padding:0.35rem 0.85rem; font-size:0.8rem; font-weight:500; border:1px solid; margin:0.15rem; }
    .chip-blue { background:#eff6ff; color:#1e3a8a; border-color:#bfdbfe; }

    .navbar { position:sticky; top:0; z-index:999; background:rgba(255,255,255,0.95); backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px); border-bottom:1px solid var(--border); margin:-2rem -1rem 0; padding:0 clamp(0.75rem,2vw,1rem); }
    .navbar-inner { display:flex; align-items:center; justify-content:space-between; height:clamp(60px,8vh,72px); max-width:1280px; margin:0 auto; gap:0.5rem; }
    .nav-left { display:flex; align-items:center; gap:0.5rem; flex-shrink:0; }
    .nav-logo-box { width:clamp(32px,4vw,36px); height:clamp(32px,4vw,36px); background:var(--primary); border-radius:10px; display:flex; align-items:center; justify-content:center; color:white; font-weight:700; font-size:clamp(0.8rem,2.2vw,0.9rem); flex-shrink:0; }
    .nav-title-text { font-weight:700; font-size:clamp(0.85rem,2.4vw,1rem); color:var(--foreground); line-height:1.2; }
    .nav-sub-text { font-size:clamp(0.6rem,1.6vw,0.7rem); color:var(--muted); }
    .nav-center { display:flex; align-items:center; gap:0.25rem; }
    .nav-center a { text-decoration:none; border-radius:8px; padding:0.5rem 0.75rem; font-size:clamp(0.8rem,2.2vw,0.875rem); font-weight:500; color:#475569; transition:all 0.15s ease-out; white-space:nowrap; }
    .nav-center a:hover { background:#f1f5f9; color:var(--foreground); }
    .nav-center a.active { background:#eff6ff; color:var(--primary); font-weight:600; }
    .nav-right { display:flex; align-items:center; gap:0.5rem; flex-shrink:0; }
    .nav-btn { display:inline-flex; align-items:center; gap:0.375rem; padding:0.5rem 1rem; border-radius:10px; font-size:clamp(0.8rem,2.2vw,0.875rem); font-weight:600; text-decoration:none; transition:all 0.15s ease-out; white-space:nowrap; min-height:40px; }
    .nav-btn-primary { background:var(--primary); color:white; border:1px solid var(--primary); }
    .nav-btn-primary:hover { background:var(--primary-hover); }
    .nav-btn-outline { background:white; color:var(--foreground); border:1px solid var(--border); }
    .nav-btn-outline:hover { background:#f8fafc; border-color:#cbd5e1; }

    .nav-toggle { display:none; }
    .nav-toggle-label { display:none; cursor:pointer; padding:0.5rem; border-radius:8px; user-select:none; }
    .nav-toggle-label:hover { background:#f1f5f9; }
    .nav-toggle-label span { display:block; width:22px; height:2px; background:#475569; margin:5px 0; border-radius:2px; transition:all 0.2s ease-out; }

    .responsive-grid { display:grid; gap:1rem; }
    .responsive-grid-2 { grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); }
    .responsive-grid-3 { grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); }
    .responsive-grid-4 { grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); }

    @media (max-width:768px) {
        .nav-center { display:none; flex-direction:column; position:absolute; top:100%; left:0; right:0; background:white; border-bottom:1px solid var(--border); padding:0.5rem; box-shadow:0 8px 24px rgba(15,23,42,0.1); gap:0.125rem; }
        .nav-center a { padding:0.75rem 1rem; font-size:1rem; width:100%; }
        .nav-toggle:checked ~ .nav-center { display:flex; }
        .nav-toggle-label { display:block; }
        .navbar-inner { height:clamp(56px,7vh,64px); }
        .nav-sub-text { display:none; }
        section[data-testid="stSidebar"] { width:100% !important; }
        .block-container { padding:1rem 0.75rem !important; }
        .navbar { margin:-1rem -0.75rem 0; }
        section[data-testid="stSidebar"] { display:none !important; }
        div[data-testid="column"] { min-width:100% !important; flex:0 0 100% !important; gap:1rem !important; padding-bottom:1rem !important; }
        .hero-stat-grid { grid-template-columns:1fr 1fr; }
        .login-card { margin:1.5rem auto; padding:1.25rem; }
        .page-header { margin-bottom:0.75rem; }
        .detail-row { flex-direction:column; gap:0.15rem; padding:0.75rem 0; }
    }

    @media (max-width:480px) {
        .hero-stat-grid { grid-template-columns:1fr; }
        .hero-h1 { font-size:clamp(1.25rem,6vw,1.5rem); }
        .nav-title-text { font-size:0.8rem; }
        .nav-logo-box { width:28px; height:28px; font-size:0.75rem; }
        .nav-btn { padding:0.375rem 0.625rem; font-size:0.75rem; min-height:36px; }
        .badge { font-size:0.6rem; padding:0.2rem 0.5rem; }
    }

    @media (max-width:1024px) {
        .hero-grid { grid-template-columns:1fr; gap:1.5rem; }
        .workflow-arrow { display:none; }
        div[data-testid="column"] { min-width:100% !important; flex:0 0 100% !important; }
    }

    @media (min-width:1025px) and (max-width:1280px) {
        .hero-h1 { font-size:clamp(1.75rem,3.5vw,2.25rem); }
    }

    @media (prefers-reduced-motion:reduce) { * { transition-duration:0.01ms !important; animation-duration:0.01ms !important; } }
</style>
""", unsafe_allow_html=True)

# ─── AUTH & STATE ─────────────────────────────────────
ADMIN_USER, ADMIN_PASS = "admin", "admin123"
for k, v in {"authenticated": False, "page": "Beranda", "last_result": None,
             "last_input": {}, "scenario_result": None, "scenario_input": {} }.items():
    if k not in st.session_state: st.session_state[k] = v

# ─── QUERY PARAM NAV ──────────────────────────────────
if "nav" in st.query_params:
    target = st.query_params["nav"]
    if target == "Logout":
        st.session_state.authenticated = False
        st.session_state.last_result = None
        st.session_state.scenario_result = None
        target = "Beranda"
    if target in ["Beranda","Dashboard","Scenario","Laporan","Info Model","Login","Admin"]:
        st.session_state.page = target
    st.query_params.clear()
    st.rerun()

# ─── DATA ──────────────────────────────────────────────
@st.cache_data
def load_data():
    with open(ARTIFACT_DIR / "frontend_dashboard_data.json", encoding="utf-8") as f:
        return json.load(f)

data = load_data()
meta = data["model_metadata"]
trend = pd.DataFrame(data["trend_data"])
fi = pd.DataFrame(data["feature_importance"])

# ─── HELPERS ───────────────────────────────────────────
def badge(level):
    c = {"Low Priority": "badge-low", "Medium Priority": "badge-medium", "High Priority": "badge-high"}.get(level, "badge-blue")
    return f'<span class="badge {c}">{level}</span>'

def mcard(lbl, val, clr="#1e3a8a"):
    return f'<div class="metric-box"><div class="metric-label">{lbl}</div><div class="metric-value" style="color:{clr}">{val}</div></div>'

def pcard(lbl, val, clr):
    return f'<div class="metric-box"><div class="metric-label" style="color:{clr}80">{lbl}</div><div class="metric-value" style="color:{clr}">{val}</div></div>'

def _svg(path, view="0 0 24 24"):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="{view}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0">{path}</svg>'

I_CAL      = _svg('<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>')
I_HOME     = _svg('<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>')
I_CHART    = _svg('<path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/>')
I_CLIP     = _svg('<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>')
I_DOC      = _svg('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>')
I_INFO     = _svg('<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>')
I_LOCK     = _svg('<rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>')
I_UNLOCK   = _svg('<rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M10 14v4"/><path d="M17 11V7a5 5 0 0 0-8.5-3.5"/><circle cx="12" cy="16" r="1"/>')
I_EXIT     = _svg('<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>')
I_TREND    = _svg('<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>')
I_SHIELD   = _svg('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/>')
I_PENCIL   = _svg('<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>')
I_COG      = _svg('<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>')
I_RULER    = _svg('<path d="M4 4l16 16"/><path d="M20 4L4 20"/><line x1="8" y1="4" x2="12" y2="8"/><line x1="4" y1="8" x2="8" y2="12"/><line x1="12" y1="12" x2="16" y2="16"/><line x1="16" y1="8" x2="20" y2="12"/>')
I_CK       = _svg('<polyline points="20 6 9 17 4 12"/>')
I_CK_CIRC  = _svg('<circle cx="12" cy="12" r="10"/><polyline points="9 12 11 14 15 10"/>')
I_TARGET   = _svg('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>')
I_BRAIN    = _svg('<path d="M9.5 2A3.5 3.5 0 0 1 13 5.5V7a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-.5A3.5 3.5 0 0 1 9.5 2z"/><path d="M14.5 2A3.5 3.5 0 0 0 11 5.5V7a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-.5A3.5 3.5 0 0 0 14.5 2z"/><path d="M12 13a3.5 3.5 0 0 0 3.5-3.5V9a2 2 0 0 1 2-2h1a3.5 3.5 0 0 1 0 7h-1a2 2 0 0 1-2 2v0a3.5 3.5 0 0 1-3.5 3.5z"/><path d="M12 13a3.5 3.5 0 0 1-3.5 3.5V16a2 2 0 0 0-2-2h-1a3.5 3.5 0 0 1 0-7h1a2 2 0 0 1 2 2v.5A3.5 3.5 0 0 0 12 13z"/>')
I_ALERT    = _svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>')
I_MINUS    = _svg('<line x1="5" y1="12" x2="19" y2="12"/>')
I_PLUS     = _svg('<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>')
I_UPLOAD   = _svg('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>')
I_TRASH    = _svg('<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/>')
I_EYE      = _svg('<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>')
I_DOWN     = _svg('<polyline points="6 9 12 15 18 9"/>')
I_COPY     = _svg('<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>')

def run_pred(tahun, gini, tpt, inflasi, ipm):
    r = predict_condition(tahun, gini, tpt, inflasi, ipm)
    st.session_state.last_result = r
    st.session_state.last_input = {"tahun": tahun, "gini": gini, "tpt": tpt, "inflasi": inflasi, "ipm": ipm}

def run_scenario(b, s):
    r1 = predict_condition(b["tahun"], b["gini"], b["tpt"], b["inflasi"], b["ipm"])
    r2 = predict_condition(s["tahun"], s["gini"], s["tpt"], s["inflasi"], s["ipm"])
    st.session_state.scenario_result = {"baseline": r1, "scenario": r2}
    st.session_state.scenario_input = {"baseline": b, "scenario": s}

def style_fig(fig):
    fig.update_layout(font_family="Inter,sans-serif", plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#e2e8f0", title_font=dict(size=12, color="#64748b"), tickfont=dict(size=11, color="#64748b"), zeroline=False),
        yaxis=dict(gridcolor="#e2e8f0", title_font=dict(size=12, color="#64748b"), tickfont=dict(size=11, color="#64748b"), zeroline=False),
        hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10))
    return fig

def page_eyebrow_title(text, title, desc=None):
    return f'''<div class="page-header"><p class="eyebrow">{text}</p>
    <h1 class="app-title">{title}</h1>{"<p class='app-subtitle'>"+desc+"</p>" if desc else ""}</div>'''

# ─── NAVBAR ─────────────────────────────────────────────
def render_navbar():
    page = st.session_state.page
    is_auth = st.session_state.authenticated
    links = ["Beranda","Dashboard","Scenario","Laporan","Info Model"]
    nav_items = "".join(
        f'<a href="?nav={l}" class="nav-link{" active" if page==l else ""}">{l}</a>'
        for l in links
    )
    right = ""
    if is_auth:
        right += f'<a href="?nav=Admin" class="nav-btn nav-btn-primary">{I_LOCK} Admin</a>'
        right += f'<a href="?nav=Logout" class="nav-btn nav-btn-outline">{I_EXIT} Logout</a>'
    else:
        right += f'<a href="?nav=Login" class="nav-btn nav-btn-outline">{I_UNLOCK} Login</a>'
    right += f'<a href="?nav=Dashboard" class="nav-btn nav-btn-primary" style="display:none" id="nav-mulai">Mulai Prediksi →</a>'

    st.markdown(f'''<div class="navbar" id="top-nav"><div class="navbar-inner">
        <div class="nav-left">
            <div class="nav-logo-box">JSI</div>
            <div><div class="nav-title-text">Jabar Social Insight</div><div class="nav-sub-text">Decision Support System</div></div>
        </div>
        <input type="checkbox" class="nav-toggle" id="nav-toggle" autocomplete="off">
        <label for="nav-toggle" class="nav-toggle-label" aria-label="Toggle navigation menu"><span></span><span></span><span></span></label>
        <div class="nav-center">{nav_items}</div>
        <div class="nav-right">{right}</div>
    </div></div>''', unsafe_allow_html=True)

render_navbar()

# ─── SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    st.markdown(f'''<div class="sidebar-header"><div class="sidebar-logo">JSI</div>
        <div><div class="sidebar-title">Jabar Social Insight</div><div class="sidebar-sub">Decision Support System</div></div></div>''',
        unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#e2e8f0;margin:1.25rem 0' />", unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:var(--muted);margin-bottom:0.5rem">Navigasi</p>', unsafe_allow_html=True)

    icons = {"Beranda":I_HOME,"Dashboard":I_CHART,"Scenario":I_CLIP,"Laporan":I_DOC,"Info Model":I_INFO}
    for lbl in ["Beranda","Dashboard","Scenario","Laporan","Info Model"]:
        active = "background:#eff6ff;color:var(--primary);font-weight:600" if st.session_state.page==lbl else "color:#475569"
        st.markdown(f'<a href="?nav={lbl}" style="display:flex;align-items:center;gap:0.5rem;padding:0.5rem 0.75rem;border-radius:8px;text-decoration:none;font-size:0.875rem;{active};transition:all 0.15s ease-out">{icons[lbl]} {lbl}</a>', unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#e2e8f0;margin:1rem 0' />", unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:var(--muted);margin-bottom:0.5rem">Akun</p>', unsafe_allow_html=True)

    if st.session_state.authenticated:
        st.markdown(f'<a href="?nav=Admin" style="display:flex;align-items:center;gap:0.5rem;padding:0.5rem 0.75rem;border-radius:8px;text-decoration:none;font-size:0.875rem;color:#475569;transition:all 0.15s ease-out">{I_LOCK} Admin</a>', unsafe_allow_html=True)
        st.markdown(f'<a href="?nav=Logout" style="display:flex;align-items:center;gap:0.5rem;padding:0.5rem 0.75rem;border-radius:8px;text-decoration:none;font-size:0.875rem;color:#b91c1c;transition:all 0.15s ease-out">{I_EXIT} Logout</a>', unsafe_allow_html=True)
    else:
        st.markdown(f'<a href="?nav=Login" style="display:flex;align-items:center;gap:0.5rem;padding:0.5rem 0.75rem;border-radius:8px;text-decoration:none;font-size:0.875rem;color:#475569;transition:all 0.15s ease-out">{I_UNLOCK} Login Admin</a>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  PAGE: BERANDA  (HeroSection + FeatureSection + WorkflowSection + ModelSummarySection + CtaSection)
# ═══════════════════════════════════════════════════════
if st.session_state.page == "Beranda":

    # ── HERO SECTION ────────────────────────────────────
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    hc1, hc2 = st.columns([1.05, 0.95], gap="large")
    with hc1:
        st.markdown('<p class="eyebrow">Machine Learning Decision Support</p>', unsafe_allow_html=True)
        st.markdown('<h1 class="hero-h1">Prediksi Kondisi Sosial Jawa Barat Berbasis Machine Learning</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-p">Platform ini membantu memprediksi estimasi tingkat kemiskinan, menentukan prioritas intervensi sosial, dan menghasilkan rekomendasi kebijakan berdasarkan indikator ekonomi dan sosial.</p>', unsafe_allow_html=True)
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("Mulai Prediksi →", type="primary", use_container_width=True):
                st.session_state.page = "Dashboard"; st.rerun()
        with b2:
            if st.button("Info Model →", use_container_width=True):
                st.session_state.page = "Info Model"; st.rerun()
        st.markdown('<div class="hero-stat-grid">', unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(f'<div class="hero-stat"><div class="hero-stat-lbl">Akurasi Model</div><div class="hero-stat-val">R² {meta["metrics"]["r2_score"]:.3f}</div></div>', unsafe_allow_html=True)
        with sc2:
            st.markdown(f'<div class="hero-stat"><div class="hero-stat-lbl">Akurasi Klasifikasi</div><div class="hero-stat-val">{meta["metrics"]["classification_accuracy"]:.3f}</div></div>', unsafe_allow_html=True)
        with sc3:
            st.markdown(f'<div class="hero-stat"><div class="hero-stat-lbl">Mode</div><div class="hero-stat-val" style="font-size:0.95rem">Static Artifact</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with hc2:
        st.markdown(f'''<div class="card-nx card-nx-soft" style="margin-bottom:1rem">
            <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem">
                <div class="icon-box icon-box-blue">{I_CHART}</div>
                <div><div class="card-title-nx">Preview Dashboard Prediksi</div>
                <div class="card-desc">Estimasi kemiskinan berdasarkan input indikator</div></div>
            </div>
            <div style="display:flex;align-items:center;justify-content:space-between;padding:0.75rem;background:#f8fafc;border-radius:12px;border:1px solid var(--border)">
                <div><div class="card-desc">Estimasi kemiskinan</div>
                <div style="font-family:ui-monospace,monospace;font-size:1.5rem;font-weight:700;color:var(--primary)">9.06%</div></div>
                <span class="badge badge-medium">Medium Priority</span>
            </div>
            <div class="card-desc" style="margin-top:0.75rem">Input 5 indikator → estimasi angka kemiskinan + prioritas intervensi + rekomendasi kebijakan.</div>
        </div>''', unsafe_allow_html=True)
        st.markdown('<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">', unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f'<div class="card-nx"><div class="card-desc">Model Regresi</div><div style="font-family:ui-monospace,monospace;font-size:0.9rem;font-weight:600;color:var(--foreground);margin-top:0.15rem">{meta["best_models"]["regression"]}</div></div>', unsafe_allow_html=True)
        with mc2:
            st.markdown(f'<div class="card-nx"><div class="card-desc">Model Klasifikasi</div><div style="font-family:ui-monospace,monospace;font-size:0.9rem;font-weight:600;color:var(--foreground);margin-top:0.15rem">{meta["best_models"]["classification"]}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── FEATURE SECTION ─────────────────────────────────
    st.markdown('<div class="sec">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><p class="eyebrow">Fitur MVP</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Fitur utama untuk analisis sosial berbasis data</h2>', unsafe_allow_html=True)
    st.markdown('<p class="card-desc" style="margin-top:0.75rem">Setiap fitur dirancang untuk membantu pengguna memahami hasil model, bukan hanya melihat angka prediksi.</p></div>', unsafe_allow_html=True)

    fts = [
        (I_TREND, "Prediksi Kemiskinan", "Menghasilkan estimasi angka kemiskinan dari lima indikator sosial ekonomi utama.", "Regresi berbasis artifact model."),
        (I_SHIELD, "Prioritas Intervensi", "Memetakan prediksi menjadi Low, Medium, atau High Priority memakai threshold model.", "Klasifikasi prioritas sosial."),
        (I_CLIP, "Rekomendasi Kebijakan", "Memberikan alasan, aksi kebijakan, dan timeline implementasi yang dapat ditindaklanjuti.", "Rule-based recommendation."),
        (I_CHART, "Scenario Comparison", "Membandingkan kondisi awal dan skenario kebijaan untuk melihat dampak perubahan indikator.", "Delta dan narasi otomatis."),
    ]
    fcols = st.columns(4)
    for i, (icon, title, desc, note) in enumerate(fts):
        with fcols[i]:
            st.markdown(f'''<div class="card-nx" style="height:100%">
                <div class="icon-box icon-box-blue">{icon}</div>
                <div class="card-title-nx" style="margin-top:1.25rem">{title}</div>
                <div class="card-desc" style="margin-top:0.75rem">{desc}</div>
                <div class="card-note">{note}</div></div>''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── WORKFLOW SECTION ────────────────────────────────
    st.markdown('<div class="sec sec-white sec-border-y">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><p class="eyebrow">Alur Sistem</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Dari indikator ke rekomendasi kebijakan</h2>', unsafe_allow_html=True)
    st.markdown('<p class="card-desc" style="margin-top:0.75rem">Workflow dibuat modular agar mode static dapat diganti ke ONNX atau backend Python pada tahap production.</p></div>', unsafe_allow_html=True)

    steps = [(I_PENCIL, "Input Indikator", "Tahun, Gini Ratio, TPT, inflasi, dan IPM menjadi parameter analisis."),
             (I_COG, "Model Prediksi", "Sistem menghitung estimasi kemiskinan melalui model Random Forest."),
             (I_RULER, "Prioritas", "Prediksi dipetakan ke level intervensi berdasarkan threshold artifact."),
             (I_CK_CIRC, "Rekomendasi", "Sistem menyusun alasan, aksi kebijakan, dan timeline implementasi.")]
    wcols = st.columns(4, gap="small")
    for i, (icon, title, desc) in enumerate(steps):
        with wcols[i]:
            st.markdown(f'''<div class="card-nx workflow-card" style="height:100%">
                <div class="icon-box icon-box-teal">{icon}</div>
                <div class="card-title-nx" style="margin-top:1.25rem">{title}</div>
                <div class="card-desc" style="margin-top:0.75rem">{desc}</div></div>''', unsafe_allow_html=True)
        if i < len(steps) - 1:
            with wcols[i]:
                st.markdown('<div class="workflow-arrow">→</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── MODEL SUMMARY SECTION ──────────────────────────
    st.markdown('<div class="sec">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><p class="eyebrow">Model Summary</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Performa model dan indikator yang dianalisis</h2>', unsafe_allow_html=True)
    st.markdown('<p class="card-desc" style="margin-top:0.75rem">Metrik model ditampilkan agar pengguna memahami dasar evaluasi dan batasan interpretasi.</p></div>', unsafe_allow_html=True)

    mcols = st.columns(4, gap="medium")
    met_items = [
        (I_TREND, f"{meta['metrics']['r2_score']:.4f}", "R2 Score", "Koefisien determinasi model regresi"),
        (_svg('<path d="M23 6l-9.5 9.5-5-5L1 18"/><polyline points="17 6 23 6 23 12"/>'), f"{meta['metrics']['mae']:.4f}", "MAE", "Mean Absolute Error"),
        (I_TARGET, f"{meta['metrics']['rmse']:.4f}", "RMSE", "Root Mean Squared Error"),
        (I_BRAIN, f"{meta['metrics']['classification_accuracy']:.4f}", "Akurasi Klasifikasi", "Ketepatan klasifikasi prioritas"),
    ]
    for j, (ico, val, lbl, dsc) in enumerate(met_items):
        with mcols[j]:
            st.markdown(f'''<div class="card-nx"><div class="metric-icon-card">
                <div class="metric-icon-box">{ico}</div>
                <div><div class="metric-nx-label">{lbl}</div>
                <div class="metric-nx-value">{val}</div>
                <div class="metric-nx-desc" style="margin-top:0.15rem">{dsc}</div></div></div></div>''', unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="card-nx"><div class="card-title-nx">Tren Historis Kemiskinan</div><div class="card-desc" style="margin-bottom:1rem;margin-top:0.25rem">Rata-rata tingkat kemiskinan Jawa Barat (2018-2024)</div>', unsafe_allow_html=True)
        fig = px.line(trend, x="tahun", y="rata_rata_kemiskinan", markers=True, labels={"tahun": "Tahun", "rata_rata_kemiskinan": "Kemiskinan (%)"})
        fig.update_traces(line=dict(color="#1e3a8a", width=2.5), marker=dict(color="#1e3a8a", size=7))
        style_fig(fig); st.plotly_chart(fig, use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card-nx"><div class="card-title-nx">Feature Importance</div><div class="card-desc" style="margin-bottom:1rem;margin-top:0.25rem">Kontribusi setiap indikator terhadap prediksi</div>', unsafe_allow_html=True)
        fi_s = fi.sort_values("importance", ascending=True)
        fig2 = px.bar(fi_s, x="importance", y="feature", orientation="h", labels={"importance": "Importance", "feature": ""}, text_auto=".1%", color_discrete_sequence=["#1d4ed8"])
        fig2.update_traces(marker=dict(line=dict(width=0)), textposition="outside", cliponaxis=False)
        style_fig(fig2); fig2.update_layout(xaxis=dict(range=[0, fi_s["importance"].max()*1.35]))
        st.plotly_chart(fig2, use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── CTA SECTION ─────────────────────────────────────
    st.markdown('<div class="sec sec-white sec-border-t">', unsafe_allow_html=True)
    ctac1, ctac2 = st.columns([1.2, 1])
    with ctac1:
        st.markdown('<p class="eyebrow">Siap Digunakan</p>', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Jalankan prediksi dan susun rekomendasi kebijakan dalam satu alur.</h2>', unsafe_allow_html=True)
        st.markdown('<p class="card-desc" style="margin-top:0.75rem">Gunakan dashboard untuk demonstrasi proyek Machine Learning, presentasi akademik, atau analisis awal dampak indikator sosial ekonomi.</p>', unsafe_allow_html=True)
    with ctac2:
        st.markdown('<div style="display:flex;gap:0.75rem;justify-content:flex-end;flex-wrap:wrap">', unsafe_allow_html=True)
        if st.button("Buka Dashboard →", type="primary", use_container_width=True):
            st.session_state.page = "Dashboard"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  PAGE: DASHBOARD  (PredictionWorkspace)
# ═══════════════════════════════════════════════════════
elif st.session_state.page == "Dashboard":
    st.markdown(page_eyebrow_title("Dashboard Prediksi", "Prediksi Kondisi Sosial Jawa Barat",
        "Input indikator ekonomi dan sosial untuk melihat estimasi angka kemiskinan, level prioritas intervensi, diagnosis indikator, dan rekomendasi kebijakan."), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(mcard("R2 Score", f"{meta['metrics']['r2_score']:.4f}"), unsafe_allow_html=True)
    with c2: st.markdown(mcard("MAE", f"{meta['metrics']['mae']:.4f}"), unsafe_allow_html=True)
    with c3: st.markdown(mcard("RMSE", f"{meta['metrics']['rmse']:.4f}"), unsafe_allow_html=True)
    with c4: st.markdown(mcard("Akurasi Klasifikasi", f"{meta['metrics']['classification_accuracy']:.4f}"), unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider' />", unsafe_allow_html=True)
    col_left, col_right = st.columns([0.4, 0.6], gap="large")

    with col_left:
        st.markdown(f'''<div class="card"><div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem">
            {I_PENCIL}<div><h3 class="card-title">Input Indikator</h3>
            <p class="text-muted" style="font-size:0.8rem">Masukkan indikator sosial ekonomi untuk menghitung estimasi kemiskinan.</p></div></div>''', unsafe_allow_html=True)

        tahun = st.number_input("Tahun Prediksi", min_value=2000, max_value=2100, value=2029, step=1, help="Tahun yang akan diprediksi")
        gini = st.number_input("Gini Ratio", min_value=0.0, max_value=1000.0, value=400.0, format="%.1f", help="Ketimpangan pendapatan (0-1000). Semakin tinggi, semakin timpang.")
        tpt = st.number_input("Tingkat Pengangguran Terbuka (%)", min_value=0.0, max_value=100.0, value=5.0, format="%.2f", help="Persentase angkatan kerja yang menganggur.")
        inflasi = st.number_input("Rata-rata Inflasi Tahunan (%)", min_value=-10.0, max_value=50.0, value=0.15, format="%.2f", help="Laju inflasi tahunan.")
        ipm = st.number_input("Indeks Pembangunan Manusia", min_value=0.0, max_value=100.0, value=73.5, format="%.2f", help="Indeks pembangunan manusia (0-100).")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        if st.button("Jalankan Prediksi", type="primary", use_container_width=True):
            run_pred(tahun, gini, tpt, inflasi, ipm)
            st.rerun()
        if st.session_state.last_result:
            if st.button("Gunakan Contoh", use_container_width=True):
                run_pred(2029, 400.0, 5.0, 0.15, 73.5); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        if st.session_state.last_result:
            r = st.session_state.last_result; li = st.session_state.last_input
            pct, level = r["prediksi_kemiskinan"], r["priority_level"]
            color = {"High Priority": "#b91c1c", "Medium Priority": "#b45309", "Low Priority": "#047857"}.get(level, "#1e3a8a")

            st.markdown(f'''<div class="card card-soft"><div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem">
                {I_CK_CIRC}<div><h3 class="card-title">Ringkasan Prediksi</h3>
                <p class="text-muted" style="font-size:0.8rem">Hasil simulasi untuk tahun {li.get("tahun")} dengan artifact model static.</p></div></div>
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.75rem;margin-bottom:1.5rem">
                    <div><p style="font-size:0.875rem;font-weight:500;color:#64748b">Estimasi angka kemiskinan</p>
                    <p style="font-family:monospace;font-size:3rem;font-weight:700;color:{color}">{pct:.2f}%</p></div>
                    {badge(level)}
                </div>
                <p style="font-weight:600;color:#0f172a">{r.get("status","")}</p>
                <p class="text-muted" style="margin-top:0.25rem">Sistem memetakan hasil regresi ke level prioritas intervensi memakai threshold dari artifact model.</p>
            ''', unsafe_allow_html=True)

            st.markdown(f'''<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.75rem;margin:1.25rem 0">
                {mcard("R2 Score", f"{r.get('metadata',{}).get('r2_score', meta['metrics']['r2_score']):.4f}")}
                {mcard("MAE", f"{r.get('metadata',{}).get('mae', meta['metrics']['mae']):.4f}")}
                {mcard("RMSE", f"{meta['metrics']['rmse']:.4f}")}
            </div>''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f'''<div class="card"><div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem">
                <div class="icon-box icon-box-blue" style="width:32px;height:32px;font-size:0.9rem">{I_CLIP}</div><div><h3 class="card-title">Rekomendasi Kebijakan</h3>
                <p class="text-muted" style="font-size:0.8rem">Disusun berdasarkan level prioritas dan rule rekomendasi pada artifact.</p></div></div>
                <div style="padding:1rem;background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;margin-bottom:1rem">
                    <p style="font-weight:600;font-size:0.875rem;color:#1e3a8a">Rekomendasi utama</p>
                    <p style="font-size:0.875rem;color:#1e3a8a;margin-top:0.25rem">{r["rekomendasi_utama"]}</p></div>''', unsafe_allow_html=True)

            st.markdown(f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem">{I_CK}<p style="font-weight:600;font-size:0.875rem">Alasan sistem</p></div>', unsafe_allow_html=True)
            for alasan in r.get("alasan", []):
                st.markdown(f'<div style="display:flex;gap:0.5rem;align-items:start;margin-bottom:0.25rem"><span style="color:#0f766e;flex-shrink:0">{I_CK}</span><span style="font-size:0.85rem;color:#475569">{alasan}</span></div>', unsafe_allow_html=True)

            st.markdown(f'''<div style="margin:1rem 0 0.5rem;display:flex;align-items:center;gap:0.5rem">{I_CLIP} <span style="font-weight:600;font-size:0.875rem;color:#0f172a">Aksi kebijakan</span></div>''', unsafe_allow_html=True)
            cols = st.columns(2)
            for i, a in enumerate(r["aksi_kebijakan"]):
                with cols[i % 2]:
                    st.markdown(f'<div style="padding:0.5rem 0.75rem;background:#f8fafc;border-radius:10px;border:1px solid #e2e8f0;font-size:0.85rem;margin-bottom:0.5rem">{a}</div>', unsafe_allow_html=True)

            st.markdown(f'''<div style="margin:1rem 0 0.5rem;display:flex;align-items:center;gap:0.5rem">{I_CAL} <span style="font-weight:600;font-size:0.875rem;color:#0f172a">Timeline Implementasi</span></div>''', unsafe_allow_html=True)
            tl = [{"Periode": p, "Aksi": "; ".join(a)} for p, a in r["timeline"].items()]
            st.dataframe(pd.DataFrame(tl), hide_index=True, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f'''<div class="card" style="padding:0.75rem 1rem"><div class="detail-row">
                <span class="detail-label">Input simulasi</span>
                <span class="detail-value">Tahun={li.get("tahun")}, Gini={li.get("gini")}, TPT={li.get("tpt")}%, Inflasi={li.get("inflasi")}%, IPM={li.get("ipm")}</span>
            </div></div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''<div class="empty-state" style="min-height:400px"><div class="empty-state-icon">{I_CHART}</div>
                <div class="empty-state-title">Belum ada hasil prediksi</div>
                <div class="empty-state-desc">Input indikator di samping, lalu klik Jalankan Prediksi.</div></div>''', unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider' />", unsafe_allow_html=True)
    subtabs = st.tabs(["Tren Historis", "Feature Importance", "Distribusi Prioritas"])
    with subtabs[0]:
        st.markdown('<div class="card"><h3 class="card-title">Tren Historis Kemiskinan</h3>', unsafe_allow_html=True)
        fig = px.line(trend, x="tahun", y="rata_rata_kemiskinan", markers=True, labels={"tahun": "Tahun", "rata_rata_kemiskinan": "Kemiskinan (%)"})
        fig.update_traces(line=dict(color="#1e3a8a", width=2.5), marker=dict(color="#1e3a8a", size=7))
        fig.add_hrect(y0=0, y1=meta["priority_thresholds"]["low_threshold"], fillcolor="#047857", opacity=0.06, annotation_text=" Low Priority", annotation_position="left")
        fig.add_hrect(y0=meta["priority_thresholds"]["low_threshold"], y1=meta["priority_thresholds"]["high_threshold"], fillcolor="#b45309", opacity=0.06, annotation_text=" Medium", annotation_position="left")
        fig.add_hrect(y0=meta["priority_thresholds"]["high_threshold"], y1=trend["rata_rata_kemiskinan"].max()*1.2, fillcolor="#b91c1c", opacity=0.06, annotation_text=" High", annotation_position="left")
        if st.session_state.last_input and st.session_state.last_input.get("tahun"):
            py = st.session_state.last_input["tahun"]
            pr = st.session_state.last_result
            pv = pr["prediksi_kemiskinan"] if pr else None
            fig.add_vline(x=py, line_dash="dash", line_color="#b91c1c", line_width=2)
            fig.add_annotation(x=py, y=trend["rata_rata_kemiskinan"].max()*1.05,
                text=f"← Prediksi {py}", showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
                ax=60, ay=0, arrowcolor="#b91c1c", font=dict(color="#b91c1c", size=12))
        style_fig(fig); st.plotly_chart(fig, use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
    with subtabs[1]:
        st.markdown('<div class="card"><h3 class="card-title">Feature Importance</h3>', unsafe_allow_html=True)
        fi_s = fi.sort_values("importance", ascending=True)
        fig2 = px.bar(fi_s, x="importance", y="feature", orientation="h", labels={"importance": "Importance", "feature": ""}, text_auto=".2%", color_discrete_sequence=["#1d4ed8"])
        fig2.update_traces(marker=dict(line=dict(width=0)), textposition="outside", cliponaxis=False)
        style_fig(fig2); fig2.update_layout(xaxis=dict(range=[0, fi_s["importance"].max()*1.35]))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:0.75rem;margin-top:1rem">', unsafe_allow_html=True)
        for _, row in fi_s.sort_values("importance", ascending=False).iterrows():
            st.markdown(f'<div style="padding:0.75rem;background:#f8fafc;border-radius:10px"><div style="font-size:0.75rem;color:#64748b">{row["feature"].replace("_"," ").title()}</div><div style="font-size:1.125rem;font-weight:700;color:#1e3a8a">{row["importance"]*100:.2f}%</div></div>', unsafe_allow_html=True)
        st.markdown('</div>'); st.markdown('</div>', unsafe_allow_html=True)
    with subtabs[2]:
        st.markdown('<div class="card"><h3 class="card-title">Distribusi Threshold Prioritas</h3>', unsafe_allow_html=True)
        fig3 = go.Figure()
        for lbl, lo, hi, clr in [("Low Priority", 0, meta["priority_thresholds"]["low_threshold"], "#047857"),
            ("Medium Priority", meta["priority_thresholds"]["low_threshold"], meta["priority_thresholds"]["high_threshold"], "#b45309"),
            ("High Priority", meta["priority_thresholds"]["high_threshold"], 100, "#b91c1c")]:
            fig3.add_trace(go.Bar(x=[lbl], y=[hi-lo], marker_color=clr, text=f"{lo:.1f}-{hi:.1f}%", textposition="auto"))
        fig3.update_layout(showlegend=False, yaxis_title="Rentang Kemiskinan (%)")
        style_fig(fig3); st.plotly_chart(fig3, use_container_width=True)
        st.markdown(f'''<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:1rem">
            <div class="threshold-card" style="border-color:#a7f3d0;background:#ecfdf5"><span class="badge badge-low">Low</span><div style="margin-top:0.5rem;font-weight:700;color:#047857">≤ {meta['priority_thresholds']['low_threshold']}%</div></div>
            <div class="threshold-card" style="border-color:#fde68a;background:#fffbeb"><span class="badge badge-medium">Medium</span><div style="margin-top:0.5rem;font-weight:700;color:#b45309">{meta['priority_thresholds']['low_threshold']}% – {meta['priority_thresholds']['high_threshold']}%</div></div>
            <div class="threshold-card" style="border-color:#fecaca;background:#fef2f2"><span class="badge badge-high">High</span><div style="margin-top:0.5rem;font-weight:700;color:#b91c1c">≥ {meta['priority_thresholds']['high_threshold']}%</div></div>
        </div>''', unsafe_allow_html=True); st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  PAGE: SCENARIO  (ScenarioWorkspace)
# ═══════════════════════════════════════════════════════
elif st.session_state.page == "Scenario":
    st.markdown(page_eyebrow_title("Scenario Comparison", "Simulasi Dampak Kebijakan",
        "Bandingkan baseline dan skenario perubahan indikator untuk melihat delta kemiskinan, perubahan prioritas, dan narasi dampak kebijakan."), unsafe_allow_html=True)

    col_b, col_s = st.columns(2, gap="large")

    with col_b:
        st.markdown(f'<div class="card" style="border-left:4px solid #1e3a8a"><div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem">{I_CHART} <h3 class="card-title">Baseline</h3></div>', unsafe_allow_html=True)
        b_tahun = st.number_input("Tahun", min_value=2000, max_value=2100, value=2029, step=1, key="b_tahun")
        b_gini = st.number_input("Gini Ratio", min_value=0.0, max_value=1000.0, value=400.0, format="%.1f", key="b_gini")
        b_tpt = st.number_input("TPT (%)", min_value=0.0, max_value=100.0, value=5.0, format="%.2f", key="b_tpt")
        b_inflasi = st.number_input("Inflasi (%)", min_value=-10.0, max_value=50.0, value=0.15, format="%.2f", key="b_inflasi")
        b_ipm = st.number_input("IPM", min_value=0.0, max_value=100.0, value=73.5, format="%.2f", key="b_ipm")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_s:
        st.markdown(f'<div class="card" style="border-left:4px solid #b45309"><div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem">{I_CHART} <h3 class="card-title" style="color:#b45309">Skenario</h3></div>', unsafe_allow_html=True)
        s_tahun = st.number_input("Tahun", min_value=2000, max_value=2100, value=2029, step=1, key="s_tahun")
        s_gini = st.number_input("Gini Ratio", min_value=0.0, max_value=1000.0, value=350.0, format="%.1f", key="s_gini")
        s_tpt = st.number_input("TPT (%)", min_value=0.0, max_value=100.0, value=4.0, format="%.2f", key="s_tpt")
        s_inflasi = st.number_input("Inflasi (%)", min_value=-10.0, max_value=50.0, value=0.10, format="%.2f", key="s_inflasi")
        s_ipm = st.number_input("IPM", min_value=0.0, max_value=100.0, value=75.0, format="%.2f", key="s_ipm")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Bandingkan Skenario", type="primary", use_container_width=True):
        b_in = {"tahun": b_tahun, "gini": b_gini, "tpt": b_tpt, "inflasi": b_inflasi, "ipm": b_ipm}
        s_in = {"tahun": s_tahun, "gini": s_gini, "tpt": s_tpt, "inflasi": s_inflasi, "ipm": s_ipm}
        run_scenario(b_in, s_in)
        st.rerun()

    if st.session_state.scenario_result:
        sr = st.session_state.scenario_result
        b, s = sr["baseline"], sr["scenario"]
        bp, sp = b["prediksi_kemiskinan"], s["prediksi_kemiskinan"]
        bl, sl = b["priority_level"], s["priority_level"]
        delta_pct = sp - bp
        arrow = "↑" if delta_pct > 0 else "↓"
        delta_cls = "delta-positive" if delta_pct < 0 else "delta-negative"
        delta_label = "Membaik" if delta_pct < 0 else "Memburuk" if delta_pct > 0 else "Stabil"
        delta_icon = I_CK_CIRC if delta_pct < 0 else I_ALERT if delta_pct > 0 else I_MINUS

        st.markdown("<hr class='custom-divider' />", unsafe_allow_html=True)
        st.markdown(f'''<div class="card card-soft"><div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem">
            {delta_icon}<div><h3 class="card-title">Hasil Perbandingan</h3>
            <p class="text-muted" style="font-size:0.8rem">Delta dan narasi dampak kebijakan.</p></div></div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-bottom:1.5rem">
                {pcard("Estimasi Baseline", f"{bp:.2f}%", "#1e3a8a")}
                {pcard("Estimasi Skenario", f"{sp:.2f}%", "#b45309")}
                {pcard(f"Delta ({arrow})", f"{delta_pct:+.2f}%", "#b91c1c" if delta_pct>0 else "#047857")}
            </div>
            <div style="display:flex;gap:1rem;margin-bottom:1rem">
                <div style="padding:0.5rem 1rem;background:#f8fafc;border-radius:8px"><span style="font-size:0.8rem;color:#64748b">Baseline:</span> {badge(bl)}</div>
                <div style="padding:0.5rem 1rem;background:#f8fafc;border-radius:8px"><span style="font-size:0.8rem;color:#64748b">Skenario:</span> {badge(sl)}</div>
                <div style="padding:0.5rem 1rem;background:#f8fafc;border-radius:8px"><span style="font-size:0.8rem;color:#64748b">Status:</span> <span style="font-weight:600;color:{"#047857" if delta_pct<0 else "#b91c1c"}">{delta_label}</span></div>
            </div>
        ''', unsafe_allow_html=True)

        if delta_pct < 0:
            st.success(f"Skenario menunjukkan perbaikan: prediksi kemiskinan turun {abs(delta_pct):.2f}%.")
        elif delta_pct > 0:
            st.error(f"Skenario menunjukkan peningkatan: prediksi kemiskinan naik {delta_pct:.2f}%.")
        else:
            st.info("Tidak ada perubahan signifikan antara baseline dan skenario.")

        st.markdown(f'''<div style="margin-top:1rem;padding:1rem;background:#f8fafc;border-radius:12px">
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem">{I_CLIP}<p style="font-weight:600;font-size:0.875rem">Perubahan Indikator</p></div>
        ''', unsafe_allow_html=True)
        si = st.session_state.scenario_input
        changes = []
        for k, lbl in [("gini","Gini Ratio"), ("tpt","TPT"), ("inflasi","Inflasi"), ("ipm","IPM")]:
            bv, sv = si["baseline"][k], si["scenario"][k]
            if bv != sv:
                delta_v = sv - bv
                changes.append({"Indikator": lbl, "Baseline": bv, "Skenario": sv, "Delta": f"{delta_v:+.2f}"})
        if changes:
            st.dataframe(pd.DataFrame(changes), hide_index=True, width='stretch')
        else:
            st.markdown('<p class="text-muted">Tidak ada perubahan indikator (selain tahun).</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'''<div class="card"><p style="font-weight:600;font-size:0.875rem">Rekomendasi Baseline</p>
                <p class="text-muted" style="margin-top:0.25rem">{b["rekomendasi_utama"]}</p>
                <div style="margin-top:0.5rem"><span class="chip chip-blue">Prioritas: {bl}</span></div></div>''', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''<div class="card"><p style="font-weight:600;font-size:0.875rem">Rekomendasi Skenario</p>
                <p class="text-muted" style="margin-top:0.25rem">{s["rekomendasi_utama"]}</p>
                <div style="margin-top:0.5rem"><span class="chip chip-blue">Prioritas: {sl}</span></div></div>''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'''<div class="empty-state" style="min-height:200px;margin-top:1rem"><div class="empty-state-icon">{I_CLIP}</div>
            <div class="empty-state-title">Belum ada perbandingan</div>
            <div class="empty-state-desc">Isi kedua skenario di atas lalu klik Bandingkan Skenario.</div></div>''', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  PAGE: LAPORAN  (ReportWorkspace)
# ═══════════════════════════════════════════════════════
elif st.session_state.page == "Laporan":
    st.markdown(page_eyebrow_title("Report", "Laporan Hasil Prediksi",
        "Ringkasan prediksi terbaru dalam format naratif yang siap disalin atau diunduh sebagai Markdown."), unsafe_allow_html=True)

    if st.session_state.last_result:
        r = st.session_state.last_result; li = st.session_state.last_input
        level = r["priority_level"]
        color = {"High Priority": "#b91c1c", "Medium Priority": "#b45309", "Low Priority": "#047857"}.get(level, "#1e3a8a")

        report_parts = [
            f"# Laporan Hasil Prediksi Kemiskinan\n",
            f"**Sistem:** Jabar Social Insight — Decision Support System",
            f"**Tanggal:** {datetime.now().strftime('%d %B %Y %H:%M')}",
            f"**Model:** {meta['best_models']['regression']} (R² = {meta['metrics']['r2_score']:.4f})\n",
            "---\n## 1. Parameter Input\n",
            "| Indikator | Nilai |",
            "|-----------|-------|",
            f"| Tahun Prediksi | {li.get('tahun', '-')} |",
            f"| Gini Ratio | {li.get('gini', '-')} |",
            f"| Tingkat Pengangguran Terbuka | {li.get('tpt', '-')}% |",
            f"| Rata-rata Inflasi Tahunan | {li.get('inflasi', '-')}% |",
            f"| Indeks Pembangunan Manusia | {li.get('ipm', '-')} |\n",
            "---\n## 2. Hasil Prediksi\n",
            f"| Metrik | Nilai |",
            f"|--------|-------|",
            f"| Estimasi Kemiskinan | **{r['prediksi_kemiskinan']:.2f}%** |",
            f"| Prioritas Intervensi | **{level}** |",
            f"| Status | {r.get('status', '-')} |\n",
            "---\n## 3. Rekomendasi Utama\n",
            f">{r['rekomendasi_utama']}\n",
            "---\n## 4. Alasan Sistem\n",
        ]
        for a in r.get("alasan", []):
            report_parts.append(f"- {a}")
        report_parts.append("")
        report_parts.append("---\n## 5. Aksi Kebijakan\n")
        for a in r["aksi_kebijakan"]:
            report_parts.append(f"- {a}")
        report_parts.append("")
        report_parts.append("---\n## 6. Timeline Implementasi\n")
        for period, actions in r["timeline"].items():
            report_parts.append(f"- **{period}:** {'; '.join(actions)}")
        report_parts.append(f"\n---\n*Dihasilkan oleh Jabar Social Insight — {datetime.now().strftime('%d %B %Y %H:%M')}*")
        report_md = "\n".join(report_parts)

        c1, c2 = st.columns([0.7, 0.3])
        with c1:
            st.markdown(f'''<div class="card"><span class="badge badge-blue">{I_DOC} Laporan Prediksi</span>
                <div style="margin-top:1rem;padding:1rem;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0">
                <div style="display:flex;gap:2rem;flex-wrap:wrap">
                <div><span style="font-size:0.8rem;color:#64748b">Estimasi Kemiskinan</span><br><span style="font-size:1.5rem;font-weight:700;color:{color}">{r['prediksi_kemiskinan']:.2f}%</span></div>
                <div><span style="font-size:0.8rem;color:#64748b">Prioritas</span><br>{badge(level)}</div>
                <div><span style="font-size:0.8rem;color:#64748b">Status</span><br><span style="font-weight:600">{r.get("status","-")}</span></div>
                </div></div>
                <div style="margin-top:1rem;padding:1rem;background:#eff6ff;border-radius:12px;border:1px solid #bfdbfe">
                {I_INFO} <strong style="font-size:0.9rem">Rekomendasi:</strong> {r["rekomendasi_utama"]}</div>
            </div>''', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.download_button("Download Markdown", report_md, file_name="laporan_prediksi.md", mime="text/markdown", use_container_width=True)
            if st.button("Salin ke Clipboard", use_container_width=True):
                st.text_area("Salin teks berikut:", report_md, height=200)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><h3 class="card-title">Pratinjau Laporan</h3>', unsafe_allow_html=True)
        st.markdown(f'<div style="padding:1.5rem;background:white;border:1px solid #e2e8f0;border-radius:12px;font-size:0.9rem;line-height:1.8;font-family:monospace">{report_md}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'''<div class="empty-state" style="min-height:300px"><div class="empty-state-icon">{I_DOC}</div>
            <div class="empty-state-title">Belum ada laporan</div>
            <div class="empty-state-desc">Lakukan prediksi di halaman Dashboard terlebih dahulu.</div></div>''', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  PAGE: INFO MODEL  (ModelInfoPage)
# ═══════════════════════════════════════════════════════
elif st.session_state.page == "Info Model":
    st.markdown(page_eyebrow_title("Model Info", "Transparansi Model Machine Learning",
        "Informasi model, fitur input, metrik evaluasi, threshold prioritas, dan catatan batasan agar hasil prediksi dapat dibaca secara proporsional."), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(mcard("R2 Score", f"{meta['metrics']['r2_score']:.4f}"), unsafe_allow_html=True)
    with c2: st.markdown(mcard("MAE", f"{meta['metrics']['mae']:.4f}"), unsafe_allow_html=True)
    with c3: st.markdown(mcard("Akurasi Klasifikasi", f"{meta['metrics']['classification_accuracy']:.4f}"), unsafe_allow_html=True)
    with c4: st.markdown(mcard("Data Terlatih", f"{meta['dataset_summary']['modeling_rows']}"), unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider' />", unsafe_allow_html=True)

    st.markdown('<p class="eyebrow">Visualisasi</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Feature Importance dan Distribusi Prioritas</h2>', unsafe_allow_html=True)
    st.markdown('<p class="text-muted" style="margin-bottom:1.5rem">Visualisasi ringkas untuk memahami kontribusi indikator dan sebaran level prioritas.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        st.markdown('<div class="card"><h3 class="card-title">Feature Importance</h3>', unsafe_allow_html=True)
        fi_s = fi.sort_values("importance", ascending=True)
        fig = px.bar(fi_s, x="importance", y="feature", orientation="h", labels={"importance": "Importance", "feature": ""}, text_auto=".2%", color_discrete_sequence=["#1d4ed8"])
        fig.update_traces(marker=dict(line=dict(width=0)), textposition="outside", cliponaxis=False)
        style_fig(fig); fig.update_layout(xaxis=dict(range=[0, fi_s["importance"].max()*1.35]))
        st.plotly_chart(fig, use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h3 class="card-title">Distribusi Prioritas</h3>', unsafe_allow_html=True)
        fig3 = go.Figure()
        for lbl, lo, hi, clr in [("Low Priority", 0, meta["priority_thresholds"]["low_threshold"], "#047857"),
            ("Medium Priority", meta["priority_thresholds"]["low_threshold"], meta["priority_thresholds"]["high_threshold"], "#b45309"),
            ("High Priority", meta["priority_thresholds"]["high_threshold"], 100, "#b91c1c")]:
            fig3.add_trace(go.Bar(x=[lbl], y=[hi-lo], marker_color=clr, text=f"{lo:.1f}-{hi:.1f}%", textposition="auto"))
        fig3.update_layout(showlegend=False, yaxis_title="Rentang Kemiskinan (%)")
        style_fig(fig3); st.plotly_chart(fig3, use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider' />", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3 class="card-title">Fitur Input</h3><p class="text-muted" style="margin-bottom:1rem">Kolom fitur yang digunakan model untuk membuat prediksi.</p>', unsafe_allow_html=True)
        st.markdown('<table style="width:100%;border-collapse:collapse;font-size:0.875rem"><thead><tr style="background:#f8fafc;border-bottom:1px solid #e2e8f0">'
                    '<th style="padding:0.75rem 1rem;text-align:left;font-weight:600;color:#64748b">No</th>'
                    '<th style="padding:0.75rem 1rem;text-align:left;font-weight:600;color:#64748b">Feature Column</th></tr></thead><tbody>', unsafe_allow_html=True)
        for i, feat in enumerate(meta["feature_columns"]):
            st.markdown(f'<tr style="border-bottom:1px solid #f1f5f9"><td style="padding:0.75rem 1rem;color:#64748b">{i+1}</td><td style="padding:0.75rem 1rem;font-family:monospace">{feat}</td></tr>', unsafe_allow_html=True)
        st.markdown('</tbody></table>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><h3 class="card-title">Threshold Prioritas</h3><p class="text-muted" style="margin-bottom:1rem">Aturan klasifikasi level intervensi berdasarkan prediksi kemiskinan.</p>', unsafe_allow_html=True)
        st.markdown(f'''<div class="threshold-card" style="border-color:#a7f3d0;background:#ecfdf5;margin-bottom:0.75rem"><p style="font-weight:600;font-size:0.875rem;color:#047857">Low Priority</p>
            <p style="font-family:mono;font-size:1.25rem;font-weight:700;color:#047857">≤ {meta['priority_thresholds']['low_threshold']}%</p></div>
            <div class="threshold-card" style="border-color:#fde68a;background:#fffbeb;margin-bottom:0.75rem"><p style="font-weight:600;font-size:0.875rem;color:#b45309">Medium Priority</p>
            <p style="font-family:mono;font-size:1.25rem;font-weight:700;color:#b45309">> {meta['priority_thresholds']['low_threshold']}% s/d {meta['priority_thresholds']['high_threshold']}%</p></div>
            <div class="threshold-card" style="border-color:#fecaca;background:#fef2f2"><p style="font-weight:600;font-size:0.875rem;color:#b91c1c">High Priority</p>
            <p style="font-family:mono;font-size:1.25rem;font-weight:700;color:#b91c1c">> {meta['priority_thresholds']['high_threshold']}%</p></div>''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider' />", unsafe_allow_html=True)
    st.markdown('<div class="card"><h3 class="card-title">Artifact dan Catatan Batasan</h3><p class="text-muted" style="margin-bottom:1rem">Ringkasan sumber model dan batasan pemakaian untuk presentasi akademik.</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div style="padding:1rem;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0"><p style="font-weight:600;font-size:0.875rem">Model regresi</p><p style="font-family:monospace;font-size:0.875rem;margin-top:0.5rem">{meta["best_models"]["regression"]}</p></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div style="padding:1rem;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0"><p style="font-weight:600;font-size:0.875rem">Model klasifikasi</p><p style="font-family:monospace;font-size:0.875rem;margin-top:0.5rem">{meta["best_models"]["classification"]}</p></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div style="padding:1rem;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0"><p style="font-weight:600;font-size:0.875rem">Dibuat pada</p><p style="font-size:0.875rem;margin-top:0.5rem">{meta.get("created_at","-")}</p></div>', unsafe_allow_html=True)
    st.markdown(f'''<div style="margin-top:1rem;padding:1rem 1.25rem;background:white;border:1px solid #e2e8f0;border-radius:12px">
        <p style="font-weight:600;font-size:0.875rem">Catatan batasan model</p>
        <p style="font-size:0.875rem;color:#64748b;margin-top:0.5rem;line-height:1.7">
        Aplikasi MVP memakai mode static demo berbasis artifact JSON dan formula simulasi. File PKL disimpan sebagai arsip model Python, sedangkan inference production di Vercel sebaiknya memakai ONNX atau backend Python terpisah. Metrik yang sangat tinggi tetap perlu dijelaskan sebagai hasil evaluasi dataset tertentu dan tidak boleh dibaca sebagai jaminan akurasi untuk semua kondisi.</p></div>''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider' />", unsafe_allow_html=True)
    st.markdown('<div class="card"><h3 class="card-title">Dataset & Tren Historis</h3>', unsafe_allow_html=True)
    st.markdown(f'''<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:0.75rem;margin-bottom:1.5rem">
        {mcard("Data Mentah", f"{meta['dataset_summary']['raw_rows']} baris")}
        {mcard("Data Modeling", f"{meta['dataset_summary']['modeling_rows']} baris")}
        {mcard("Tahun Terakhir", meta['dataset_summary']['latest_year'])}
        {mcard("Tipe Proyek", meta.get('model_type','-'))}
    </div>''', unsafe_allow_html=True)
    st.dataframe(trend, hide_index=True, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  PAGE: LOGIN  (AdminLoginForm)
# ═══════════════════════════════════════════════════════
elif st.session_state.page == "Login":
    st.markdown(f'''<div class="login-card"><div style="text-align:center;margin-bottom:0.5rem">{I_LOCK}</div>
        <div class="login-title">Login Admin</div>
        <p style="text-align:center;font-size:0.875rem;color:#64748b;margin-bottom:1.5rem">Akses terbatas untuk upload CSV dan pengecekan dataset terbaru.</p>''', unsafe_allow_html=True)
    uname = st.text_input("Username", key="login_user")
    pwd = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login", type="primary", use_container_width=True):
        if uname == ADMIN_USER and pwd == ADMIN_PASS:
            st.session_state.authenticated = True; st.session_state.page = "Admin"; st.rerun()
        else:
            st.error("Username atau password salah")
    st.markdown(f'<p style="text-align:center;font-size:0.75rem;color:#94a3b8;margin-top:0.75rem">Default: admin / admin123</p></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  PAGE: ADMIN  (AdminWorkspace)
# ═══════════════════════════════════════════════════════
elif st.session_state.page == "Admin":
    if not st.session_state.authenticated:
        st.error("Silakan login terlebih dahulu"); st.stop()

    st.markdown(page_eyebrow_title("Admin Dataset", "Kelola CSV Kemiskinan Jawa Barat",
        "Upload dataset CSV baru, validasi struktur kolom, dan lihat riwayat dataset yang sudah masuk ke aplikasi."), unsafe_allow_html=True)

    upload_dir = PROJECT_ROOT / "uploaded_data"
    upload_dir.mkdir(parents=True, exist_ok=True)

    tabs = st.tabs(["Upload CSV", "File Tersimpan", "Statistik"])
    with tabs[0]:
        st.markdown(f'''<div class="card"><div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem">
            {I_UPLOAD}<div><h3 class="card-title">Upload Dataset CSV</h3>
            <p class="text-muted" style="font-size:0.8rem">Upload CSV yang berisi indikator sosial-ekonomi untuk analisis lebih lanjut.</p></div></div>''', unsafe_allow_html=True)
        up = st.file_uploader("Pilih file CSV", type="csv")
        if up:
            try:
                df = pd.read_csv(up)
                st.success(f"**{up.name}** — {len(df)} baris, {len(df.columns)} kolom")
                st.dataframe(df.head(20), hide_index=True, width='stretch')
                df.to_csv(upload_dir / up.name, index=False)
                st.info(f"Tersimpan di: `uploaded_data/{up.name}`")

                buf = io.StringIO(); df.info(buf=buf)
                with st.expander("Lihat detail kolom"):
                    st.code(buf.getvalue())
            except Exception as e:
                st.error(f"Gagal membaca CSV: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="card"><h3 class="card-title">File Tersimpan</h3>', unsafe_allow_html=True)
        files = sorted(upload_dir.glob("*.csv"))
        if files:
            for f in files:
                df_f = pd.read_csv(f)
                cs = st.columns([0.35, 0.15, 0.15, 0.15, 0.2])
                cs[0].markdown(f'{I_DOC} **{f.name}**', unsafe_allow_html=True)
                cs[1].markdown(f'<span style="font-size:0.8rem;color:#64748b">{len(df_f)} baris</span>', unsafe_allow_html=True)
                cs[2].markdown(f'<span style="font-size:0.8rem;color:#64748b">{len(df_f.columns)} kolom</span>', unsafe_allow_html=True)
                if cs[3].button("Lihat", key=f"lihat_{f.name}", use_container_width=True):
                    st.dataframe(df_f, hide_index=True, width='stretch')
                if cs[4].button("Hapus", key=f"hapus_{f.name}", use_container_width=True):
                    f.unlink(); st.rerun()
                st.markdown("<hr style='border-color:#f1f5f9;margin:0' />", unsafe_allow_html=True)
        else:
            st.markdown(f'''<div class="empty-state"><div class="empty-state-icon">{I_DOC}</div>
                <div class="empty-state-title">Belum ada file</div>
                <div class="empty-state-desc">Upload file CSV melalui tab Upload CSV.</div></div>''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:
        files = sorted(upload_dir.glob("*.csv"))
        tr = sum(len(pd.read_csv(f)) for f in files)
        st.markdown('<div class="card"><h3 class="card-title">Statistik Dataset</h3>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(mcard("Total File", str(len(files))), unsafe_allow_html=True)
        with c2: st.markdown(mcard("Total Baris", f"{tr:,}"), unsafe_allow_html=True)
        with c3: st.markdown(mcard("Dataset Asli", f"{meta['dataset_summary']['modeling_rows']} baris"), unsafe_allow_html=True)
        with c4: st.markdown(mcard("Akurasi Model", f"{meta['metrics']['classification_accuracy']:.2%}"), unsafe_allow_html=True)
        st.markdown(f'''<div style="margin-top:1rem;padding:1rem;background:#f8fafc;border-radius:12px">
            <p style="font-weight:600;font-size:0.875rem">Informasi Admin</p>
            <p style="font-size:0.85rem;color:#64748b">Login sebagai: <strong>admin</strong></p>
            <p style="font-size:0.85rem;color:#64748b">Kolom yang diharapkan: <code>tahun, gini_ratio, tingkat_penganggur_terbuka, rata_rata_inflasi_tahunan, indeks_pembangunan_manusia</code></p></div>''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
