"""Shared loading, theming and UI helpers for the Streamlit app."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlintel.cases.store import CaseStore  # noqa: E402
from amlintel.config import get_profile  # noqa: E402

PROFILE = "demo"
DATA = ROOT / "data" / PROFILE

# ---------------------------------------------------------------------------
# Palette (validated reference palette: fixed categorical order, status reserved)
# ---------------------------------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

MODEL_COLORS = {"hybrid": SERIES[0], "lgbm": SERIES[1], "anomaly": SERIES[2], "rules": SERIES[3]}
MODEL_SHORT = {"hybrid": "Deployed priority", "lgbm": "LightGBM", "anomaly": "Isolation Forest",
               "rules": "Rules"}
BAND_COLORS = {"Critical": STATUS["critical"], "High": STATUS["serious"], "Elevated": STATUS["warning"]}
BAND_ICON = {"Critical": "▲▲", "High": "▲", "Elevated": "●"}
TYP_LABEL = {"structuring": "Structuring", "fan_out": "Fan-out", "fan_in": "Fan-in",
             "pass_through": "Rapid pass-through", "cycle": "Circular flow",
             "scatter_gather": "Scatter-gather", "none": "None"}

pio.templates["aml"] = go.layout.Template(layout=dict(
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK2, size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE,
    colorway=SERIES, hovermode="closest",
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK2)),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK)),
))
pio.templates.default = "aml"

CSS = f"""
<style>
:root {{ --ink:{INK}; --ink2:{INK2}; --muted:{MUTED}; --grid:{GRID}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
h1 {{ font-size: 1.75rem !important; letter-spacing: -0.01em; }}
h2, h3 {{ letter-spacing: -0.005em; }}
.kpi {{ background:#fff; border:1px solid rgba(11,11,11,.10); border-radius:10px; padding:14px 16px; height:100%; }}
.kpi .lbl {{ color:var(--ink2); font-size:.80rem; margin-bottom:4px; }}
.kpi .val {{ color:var(--ink); font-size:1.65rem; font-weight:650; line-height:1.15; }}
.kpi .sub {{ color:var(--muted); font-size:.78rem; margin-top:4px; }}
.pill {{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:.75rem; font-weight:600;
         border:1px solid rgba(11,11,11,.12); background:#fff; color:var(--ink); margin-right:4px; }}
.note {{ color:var(--ink2); font-size:.82rem; }}
.evidence {{ background:#fff; border-left:3px solid #2a78d6; padding:8px 12px; margin:6px 0;
             border-radius:4px; font-size:.88rem; color:var(--ink); }}
.synthetic {{ background:#f0efec; border-radius:8px; padding:6px 12px; font-size:.78rem; color:var(--ink2);
              display:inline-block; margin-bottom:6px; }}
</style>
"""


def setup(title: str, subtitle: str = "") -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def synthetic_badge() -> None:
    st.markdown('<span class="synthetic">Synthetic demonstration data · no real customers · '
                'figures are modelled, not observed</span>', unsafe_allow_html=True)


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="lbl">{label}</div><div class="val">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def money(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e6:
        return f"{x / 1e6:,.1f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:,.1f}k"
    return f"{x:,.0f}"


def pct(x: float, d: int = 0) -> str:
    return f"{100 * float(x):.{d}f}%"


def band_pill(band: str) -> str:
    c = BAND_COLORS.get(band, MUTED)
    return (f'<span class="pill" style="border-color:{c}"><span style="color:{c}">{BAND_ICON.get(band, "")}'
            f'</span> {band}</span>')


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _ensure_built() -> None:
    if not (DATA / "manifest.json").exists():
        from amlintel.build import build
        with st.spinner("First run: generating the synthetic dataset and training models (about a minute)…"):
            build(get_profile(PROFILE))


@st.cache_data(show_spinner=False)
def load(name: str) -> pd.DataFrame:
    _ensure_built()
    p = DATA / name
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    _ensure_built()
    return json.loads((DATA / name).read_text())


@st.cache_data(show_spinner=False)
def transactions() -> pd.DataFrame:
    return load("transactions.parquet")


@st.cache_data(show_spinner=False)
def window_tx() -> pd.DataFrame:
    t = transactions()
    return t[t.month_idx >= 0]


@st.cache_data(show_spinner=False)
def scored() -> pd.DataFrame:
    return load("scored.parquet")


@st.cache_data(show_spinner=False)
def test_months() -> list[str]:
    return load_json("metrics.json")["test_months"]


def store() -> CaseStore:
    """Writable copy of the seeded case database (edits stay local to the deployment)."""
    _ensure_built()
    live = DATA / "cases_live.db"
    if not live.exists():
        shutil.copy(DATA / "cases.db", live)
    return CaseStore(live)


@st.cache_resource(show_spinner=False)
def lgbm_model():
    import lightgbm as lgb
    _ensure_built()
    return lgb.Booster(model_file=str(DATA / "lgbm_model.txt"))


def current_user() -> str:
    return st.session_state.get("user", "Analyst (demo)")
