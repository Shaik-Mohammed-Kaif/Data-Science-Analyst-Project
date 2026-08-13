# ==============================================================
# AI-POWERED CUSTOMER CHURN & RETENTION ANALYTICS DASHBOARD
# ADVANCED POWER BI-STYLE EDITION — v3
# Streamlit + Pandas + NumPy + Plotly + Glassmorphism UI
# ==============================================================
# WHAT CHANGED IN THIS REVISION (nothing existing removed):
#   1) FIX — State map was slow / appeared blank: the India GeoJSON is now
#      loaded through a single @st.cache_data-wrapped function. Streamlit
#      reruns the ENTIRE script on every interaction (every filter, every
#      chart click) — without caching, that meant re-reading/re-fetching
#      the GeoJSON on every single rerun, which is the real cause of the
#      slowness/blank map. It is now read/fetched once per session and
#      instantly reused after that. A local file is tried first (fastest),
#      then a cached network fetch, then a guaranteed-to-render fallback
#      ranked bar chart — the map area can never end up blank.
#   2) NEW — Power BI-style cross-filtering: clicking a bar/donut slice on
#      the charts listed below sets that category as an active filter that
#      combines (AND logic) with the sidebar filters and instantly
#      recalculates every KPI, chart, and table:
#        • Customer Status donut         -> Customer_Status
#        • Churn Rate by Contract        -> Contract
#        • Churn Rate by Internet Type   -> Internet_Type
#        • Churn Rate by Payment Method  -> Payment_Method
#        • Churn Rate by Age Group       -> Age_Group
#        • Churn Category Distribution   -> Churn_Category
#        • State Churn Ranking chart     -> State
#      An "Active Filters" chip row appears under the header with
#      "Clear Chart Selections" (keeps sidebar filters) and
#      "Clear All Filters" (resets everything) buttons.
# ==============================================================

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================
# 1. CONFIG
# ==============================================================
DATA_PATH = "Churn_Customer_Data.csv"
INDIA_GEOJSON = "india_state.geojson"
INDIA_GEOJSON_URL = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"

st.set_page_config(
    page_title="AI Customer Churn & Retention Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEMES = {
    "Dark": {
        "bg": "#07111F", "bg2": "#0D1B2A", "card": "#102033", "card2": "#0F1E30", "border": "#203653",
        "primary": "#38BDF8", "secondary": "#8B5CF6", "success": "#22C55E", "warning": "#F59E0B",
        "danger": "#EF4444", "pink": "#EC4899", "cyan": "#06B6D4", "purple": "#A855F7", "blue": "#3B82F6",
        "text": "#FFFFFF", "muted": "#91A4BD",
        "app_bg": "radial-gradient(circle at 15% 0%, #0B1A2D 0%, #07111F 45%, #050B14 100%)",
        "sidebar_bg": "linear-gradient(180deg, #0D1B2A, #081422)",
        "glass_from": "rgba(16,32,51,0.90)", "glass_to": "rgba(13,27,42,0.75)",
        "glass_hover_from": "rgba(20,40,62,0.85)", "glass_hover_to": "rgba(16,32,51,0.70)",
        "hover_shadow": "rgba(56,189,248,0.35)",
        "btn_text": "#041018",
    },
    "Light / White": {
        "bg": "#F4F6FA", "bg2": "#FFFFFF", "card": "#FFFFFF", "card2": "#F8FAFC", "border": "#E2E8F0",
        "primary": "#0284C7", "secondary": "#7C3AED", "success": "#16A34A", "warning": "#D97706",
        "danger": "#DC2626", "pink": "#DB2777", "cyan": "#0891B2", "purple": "#9333EA", "blue": "#2563EB",
        "text": "#0F172A", "muted": "#64748B",
        "app_bg": "radial-gradient(circle at 15% 0%, #FFFFFF 0%, #F1F5F9 45%, #E9EEF5 100%)",
        "sidebar_bg": "linear-gradient(180deg, #FFFFFF, #F1F5F9)",
        "glass_from": "rgba(255,255,255,0.88)", "glass_to": "rgba(241,245,249,0.75)",
        "glass_hover_from": "rgba(255,255,255,0.97)", "glass_hover_to": "rgba(241,245,249,0.88)",
        "hover_shadow": "rgba(2,132,199,0.25)",
        "btn_text": "#FFFFFF",
    },
    "Cream / Vanilla": {
        "bg": "#FBF6EC", "bg2": "#FFFDF7", "card": "#FFFDF7", "card2": "#FAF3E4", "border": "#E8DCC5",
        "primary": "#B45309", "secondary": "#9333EA", "success": "#15803D", "warning": "#C2410C",
        "danger": "#B91C1C", "pink": "#BE185D", "cyan": "#0E7490", "purple": "#7E22CE", "blue": "#1D4ED8",
        "text": "#3B2F2F", "muted": "#8A7F6B",
        "app_bg": "radial-gradient(circle at 15% 0%, #FFFDF7 0%, #FBF6EC 45%, #F5EAD4 100%)",
        "sidebar_bg": "linear-gradient(180deg, #FFFDF7, #F5EAD4)",
        "glass_from": "rgba(255,253,247,0.90)", "glass_to": "rgba(250,243,228,0.78)",
        "glass_hover_from": "rgba(255,253,247,0.98)", "glass_hover_to": "rgba(250,243,228,0.90)",
        "hover_shadow": "rgba(180,83,9,0.25)",
        "btn_text": "#FFFDF7",
    },
}

if "theme_choice" not in st.session_state:
    st.session_state["theme_choice"] = "Dark"

# NEW: cross-filter (click-to-filter) session state — separate from the
# sidebar's own filter state, so the two can combine with AND logic.
if "chart_filters" not in st.session_state:
    st.session_state["chart_filters"] = {}   # {"Customer_Status": "Churned", ...}

# NOTE: the theme switcher widget itself is rendered further down, centered
# directly under the dashboard title (see "HEADER + THEME SWITCHER" section).
COLORS = THEMES[st.session_state["theme_choice"]]

MULTI_COLOR_SEQUENCE = [
    COLORS["primary"], COLORS["pink"], COLORS["warning"], COLORS["success"],
    COLORS["purple"], COLORS["cyan"], COLORS["danger"], COLORS["blue"],
    "#F472B6", "#34D399", "#FCD34D", "#818CF8",
]

STATUS_COLOR_MAP = {"Stayed": COLORS["success"], "Churned": COLORS["danger"], "Joined": COLORS["primary"]}
CHART_CONFIG = {"displayModeBar": True, "responsive": True, "displaylogo": False}

# Dataframe columns that support click-to-filter, keyed by a short label used
# in the "Active Filters" chip row (spec section 12 — safe chart->column map)
CLICKABLE_COLUMNS = {
    "Customer_Status": "Customer Status", "Contract": "Contract", "Internet_Type": "Internet Type",
    "Payment_Method": "Payment Method", "Age_Group": "Age Group", "Churn_Category": "Churn Category",
    "State": "State",
}

# ==============================================================
# 2. DEEP CSS — GLASSMORPHISM / GRADIENTS / HOVER ANIMATIONS
# ==============================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}

    .stApp {{
        background: {COLORS['app_bg']};
    }}
    .stApp, .stApp p, .stApp span, .stApp label {{
        color: {COLORS['text']};
    }}
    section[data-testid="stSidebar"] {{
        background: {COLORS['sidebar_bg']};
        border-right: 1px solid {COLORS['border']};
    }}
    #MainMenu, footer, header {{visibility: hidden;}}

    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(14px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes shimmer {{
        0%   {{ background-position: -300px 0; }}
        100% {{ background-position: 300px 0; }}
    }}

    .dash-header {{
        text-align: center;
        padding: 1.7rem 1rem 1.1rem 1rem;
        animation: fadeInUp 0.6s ease;
    }}
    .dash-title {{
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['secondary']}, {COLORS['pink']}, {COLORS['primary']});
        background-size: 300% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 6s linear infinite;
        margin: 0;
        line-height: 1.25;
    }}
    .dash-subtitle {{
        color: {COLORS['muted']};
        font-size: 0.97rem;
        margin-top: 0.45rem;
        letter-spacing: 0.4px;
    }}

    .count-pill {{
        display: inline-block;
        margin: 0.4rem auto 0.2rem auto;
        padding: 0.35rem 1.1rem;
        border-radius: 999px;
        background: linear-gradient(120deg, rgba(56,189,248,0.14), rgba(139,92,246,0.14));
        border: 1px solid {COLORS['border']};
        color: {COLORS['text']};
        font-size: 0.85rem;
        font-weight: 600;
        text-align: center;
    }}
    .count-wrap {{ text-align: center; margin-bottom: 0.6rem; }}

    /* NEW — active cross-filter chip row */
    .active-filters-wrap {{
        text-align: center;
        margin: 0.3rem auto 0.9rem auto;
        color: {COLORS['muted']};
        font-size: 0.85rem;
    }}
    .filter-chip {{
        display: inline-block;
        margin: 0.15rem 0.3rem;
        padding: 0.28rem 0.85rem;
        border-radius: 999px;
        background: linear-gradient(120deg, rgba(56,189,248,0.18), rgba(139,92,246,0.18));
        border: 1px solid {COLORS['primary']};
        color: {COLORS['text']};
        font-size: 0.78rem;
        font-weight: 600;
        animation: fadeInUp 0.35s ease;
    }}

    .section-header {{
        font-size: 1.28rem;
        font-weight: 700;
        color: {COLORS['text']};
        margin: 1.8rem 0 0.7rem 0;
        padding: 0.4rem 0 0.6rem 0.8rem;
        border-left: 4px solid {COLORS['primary']};
        border-bottom: 1px solid {COLORS['border']};
        animation: fadeInUp 0.5s ease;
        background: linear-gradient(90deg, rgba(56,189,248,0.06), transparent);
        border-radius: 0 8px 8px 0;
    }}

    .kpi-card {{
        position: relative;
        background: linear-gradient(160deg, {COLORS['glass_from']}, {COLORS['glass_to']});
        backdrop-filter: blur(10px);
        border: 1px solid {COLORS['border']};
        border-radius: 16px;
        padding: 1.15rem 0.8rem;
        text-align: center;
        height: 100%;
        overflow: hidden;
        transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
        animation: fadeInUp 0.5s ease;
    }}
    .kpi-card::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: var(--accent, {COLORS['primary']});
        opacity: 0.9;
    }}
    .kpi-card:hover {{
        transform: translateY(-6px) scale(1.015);
        border-color: var(--accent, {COLORS['primary']});
        box-shadow: 0 14px 30px -10px var(--accent, {COLORS['primary']});
    }}
    .kpi-icon-circle {{
        width: 42px; height: 42px;
        margin: 0 auto 0.4rem auto;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.25rem;
        background: color-mix(in srgb, var(--accent, {COLORS['primary']}) 18%, transparent);
        border: 1px solid var(--accent, {COLORS['primary']});
    }}
    .kpi-label {{
        color: {COLORS['muted']};
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 0.3rem;
    }}
    .kpi-value {{
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }}
    .kpi-desc {{
        color: {COLORS['muted']};
        font-size: 0.7rem;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: linear-gradient(160deg, {COLORS['glass_from']}, {COLORS['glass_to']}) !important;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease, background 0.2s ease;
        border-radius: 14px !important;
        animation: fadeInUp 0.55s ease;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 28px -14px {COLORS['hover_shadow']};
        border-color: {COLORS['primary']} !important;
        background: linear-gradient(160deg, {COLORS['glass_hover_from']}, {COLORS['glass_hover_to']}) !important;
    }}
    .js-plotly-plot .plotly .modebar {{
        background: transparent !important;
    }}

    .info-banner {{
        background-color: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-left: 4px solid {COLORS['warning']};
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: {COLORS['muted']};
        font-size: 0.85rem;
        margin-bottom: 0.9rem;
    }}

    .click-hint {{
        text-align: center; color: {COLORS['muted']}; font-size: 0.72rem;
        margin-top: -0.4rem; margin-bottom: 0.4rem; font-style: italic;
    }}

    .summary-box {{
        background: linear-gradient(160deg, {COLORS['glass_from']}, {COLORS['glass_to']});
        backdrop-filter: blur(8px);
        border: 1px solid {COLORS['border']};
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        color: {COLORS['text']};
        font-size: 0.94rem;
        line-height: 1.8;
        margin-bottom: 1rem;
        animation: fadeInUp 0.5s ease;
    }}
    .summary-box b {{ color: {COLORS['primary']}; }}

    .action-card {{
        background: linear-gradient(100deg, {COLORS['glass_from']}, {COLORS['glass_to']});
        border: 1px solid {COLORS['border']};
        border-left: 4px solid {COLORS['success']};
        border-radius: 12px;
        padding: 0.75rem 1.1rem;
        margin-bottom: 0.65rem;
        color: {COLORS['text']};
        font-size: 0.89rem;
        transition: transform 0.18s ease, border-color 0.18s ease;
    }}
    .action-card:hover {{
        transform: translateX(4px);
        border-left-color: {COLORS['primary']};
    }}

    .footer-box {{
        text-align: center;
        color: {COLORS['muted']};
        padding: 1.7rem 0 0.9rem 0;
        font-size: 0.85rem;
        border-top: 1px solid {COLORS['border']};
        margin-top: 2.2rem;
    }}
    .footer-box .foot-title {{
        color: {COLORS['text']};
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.25rem;
    }}

    div[data-testid="stMetricValue"] {{ color: {COLORS['text']}; }}

    .stButton > button {{
        background: linear-gradient(120deg, {COLORS['primary']}, {COLORS['blue']});
        color: {COLORS['btn_text']};
        font-weight: 700;
        border: none;
        border-radius: 9px;
        width: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 18px -6px {COLORS['primary']};
        color: {COLORS['btn_text']};
    }}
    .stDownloadButton > button {{
        background: linear-gradient(120deg, {COLORS['success']}, {COLORS['cyan']});
        color: {COLORS['btn_text']};
        font-weight: 700;
        border: none;
        border-radius: 9px;
        transition: transform 0.15s ease;
    }}
    .stDownloadButton > button:hover {{ transform: translateY(-2px); }}

    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div,
    .stTextInput input {{
        background-color: {COLORS['card']} !important;
        border-color: {COLORS['border']} !important;
        color: {COLORS['text']} !important;
    }}
    div[data-testid="stExpander"] {{
        background-color: {COLORS['card']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 12px;
    }}
    .stTabs [data-baseweb="tab-list"] {{ background-color: transparent; gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{ color: {COLORS['muted']}; }}
    .stTabs [aria-selected="true"] {{ color: {COLORS['primary']} !important; }}
    div[data-testid="stDataFrame"] {{ border: 1px solid {COLORS['border']}; border-radius: 10px; }}

    .sidebar-caption {{
        color: {COLORS['muted']};
        font-size: 0.78rem;
        margin-bottom: 0.6rem;
    }}
    .filter-status-badge {{
        display:inline-block; padding:0.15rem 0.55rem; border-radius:999px;
        background: rgba(56,189,248,0.14); border:1px solid {COLORS['border']};
        color:{COLORS['primary']}; font-size:0.72rem; font-weight:700; margin-top:0.3rem;
    }}

    .theme-switcher-wrap {{
        display: flex;
        justify-content: center;
        margin: 0.2rem 0 1.4rem 0;
    }}
    div[data-testid="stSegmentedControl"] {{
        display: flex;
        justify-content: center;
    }}
    div[data-testid="stSegmentedControl"] label {{
        background: {COLORS['card']} !important;
        border: 1px solid {COLORS['border']} !important;
        color: {COLORS['text']} !important;
        border-radius: 999px !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    div[data-testid="stSegmentedControl"] label:hover {{
        transform: translateY(-2px);
    }}
</style>
""", unsafe_allow_html=True)

# ==============================================================
# 3. HELPER FUNCTIONS
# ==============================================================

def safe_numeric(series):
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def safe_divide(numerator, denominator, default=0.0):
    try:
        if denominator in (0, None) or pd.isna(denominator):
            return default
        result = numerator / denominator
        if not np.isfinite(result):
            return default
        return result
    except Exception:
        return default


def format_currency(value):
    if value is None or pd.isna(value):
        return "₹0"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_00_00_000:
        return f"{sign}₹{value/1_00_00_000:,.2f} Cr"
    if value >= 1_00_000:
        return f"{sign}₹{value/1_00_000:,.2f} L"
    return f"{sign}₹{value:,.0f}"


def format_number(value):
    if value is None or pd.isna(value):
        return "0"
    return f"{value:,.0f}"


def calculate_churn_rate(data):
    if data is None or data.empty or "Customer_ID" not in data.columns:
        return 0.0
    total = data["Customer_ID"].nunique()
    if total == 0:
        return 0.0
    churned = data.loc[data["Churn_Flag"] == 1, "Customer_ID"].nunique()
    return safe_divide(churned * 100, total, 0.0)


def empty_chart(title="No sufficient data available for this analysis.", height=360):
    fig = go.Figure()
    fig.add_annotation(
        text=title, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color=COLORS["muted"]),
    )
    fig.update_layout(
        plot_bgcolor=COLORS["bg2"], paper_bgcolor=COLORS["bg2"],
        height=height, margin=dict(l=30, r=30, t=40, b=30),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def apply_chart_theme(fig, title=None, height=390, show_legend=True, hovermode="closest"):
    """Deep, centralized Plotly layout standardizer.
    Fully transparent plot + paper background (blends into the glass chart-card),
    no gridlines, soft hover styling, smooth transitions."""
    layout_kwargs = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], size=12, family="Inter, Segoe UI, sans-serif"),
        height=height,
        margin=dict(l=45, r=30, t=58 if title else 30, b=45),
        showlegend=show_legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["muted"], size=11), orientation="h",
                     yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor=COLORS["card"], font_color=COLORS["text"], bordercolor=COLORS["primary"]),
        hovermode=hovermode,
        bargap=0.28,
        transition=dict(duration=350, easing="cubic-in-out"),
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(size=15, color=COLORS["text"]), x=0.02, xanchor="left")
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linecolor=COLORS["border"],
                      color=COLORS["muted"], tickfont=dict(size=11))
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, linecolor=COLORS["border"],
                      color=COLORS["muted"], tickfont=dict(size=11))
    return fig


def clean_for_plot(data, cols):
    existing = [c for c in cols if c in data.columns]
    if not existing:
        return data.iloc[0:0].copy()
    return data.dropna(subset=existing).copy()


def group_churn_rate(data, dim):
    if data is None or data.empty or dim not in data.columns:
        return pd.DataFrame(columns=[dim, "Customers", "Churned", "Churn_Rate"])
    g = data.groupby(dim, dropna=True).agg(
        Customers=("Customer_ID", "nunique"), Churned=("Churn_Flag", "sum")
    ).reset_index()
    g["Churn_Rate"] = g.apply(lambda r: safe_divide(r["Churned"] * 100, r["Customers"], 0.0), axis=1)
    return g.sort_values("Churn_Rate", ascending=False)


def safe_bar(data, x, y, orientation="v", color=None, title=None, height=390,
             color_discrete_map=None, text_auto=False, color_seq=None, barmode=None,
             unique_bar_colors=True):
    if data is None or data.empty:
        return empty_chart(f"No sufficient data available — {title}" if title else None, height)
    try:
        fig = px.bar(
            data, x=x, y=y, orientation=orientation, color=color,
            color_discrete_map=color_discrete_map,
            color_discrete_sequence=color_seq or MULTI_COLOR_SEQUENCE,
            text_auto=text_auto, barmode=barmode,
        )
        fig.update_traces(marker_line_width=0)
        if color is None and unique_bar_colors:
            cat_axis = y if orientation == "h" else x
            n_bars = data[cat_axis].nunique() if cat_axis in data.columns else len(data)
            palette = color_seq or MULTI_COLOR_SEQUENCE
            bar_colors = [palette[i % len(palette)] for i in range(n_bars)]
            fig.update_traces(marker=dict(color=bar_colors))
        return apply_chart_theme(fig, title, height, show_legend=bool(color))
    except Exception:
        return empty_chart(f"Unable to render chart — {title}" if title else None, height)


def chart_card(fig):
    """Render a plotly figure inside a bordered, hover-animated card container
    (NON-interactive / no click-filter — used for charts that aren't part of
    the cross-filter set, exactly as before)."""
    with st.container(border=True):
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)


# ==============================================================
# 3b. NEW — CROSS-FILTER (CLICK-TO-FILTER) HELPERS
# ==============================================================

def _register_click(column_name, clicked_value):
    """Set/replace the click-filter for one column and force a rerun so the
    whole dashboard recalculates against the new combined filter set."""
    if clicked_value is None:
        return
    clicked_value = str(clicked_value)
    if st.session_state["chart_filters"].get(column_name) != clicked_value:
        st.session_state["chart_filters"][column_name] = clicked_value
        st.rerun()


def clickable_bar_card(data, x, y, column_name, title, orientation="v", height=390,
                        color_seq=None, texttemplate=None, unique_key=None):
    """Bar chart that behaves like a Power BI visual: clicking a bar applies
    that category as a cross-filter. Falls back to a normal (non-clickable)
    chart_card if the target column isn't in CLICKABLE_COLUMNS or the chart
    has no data — never crashes, never loses the existing chart."""
    if data is None or data.empty:
        chart_card(empty_chart(f"No sufficient data available — {title}" if title else None, height))
        return
    fig = safe_bar(data, x=x, y=y, orientation=orientation, title=title, height=height, color_seq=color_seq)
    if texttemplate:
        fig.update_traces(texttemplate=texttemplate, textposition="outside")

    key = unique_key or f"click_{column_name}_{title}"
    with st.container(border=True):
        event = st.plotly_chart(
            fig, width="stretch", config=CHART_CONFIG,
            on_select="rerun", selection_mode="points", key=key,
        )
    st.markdown('<div class="click-hint">Click a bar to filter the whole dashboard</div>', unsafe_allow_html=True)

    if event and event.get("selection", {}).get("points"):
        pt = event["selection"]["points"][0]
        clicked_val = pt.get("x") if orientation == "v" else pt.get("y")
        _register_click(column_name, clicked_val)


def clickable_donut_card(data, names, values, column_name, title, color_discrete_map=None, height=380,
                          unique_key=None):
    """Donut/pie chart with the same click-to-filter behavior."""
    if data is None or data.empty:
        chart_card(empty_chart(title, height))
        return
    fig = px.pie(data, names=names, values=values, hole=0.55, color=names,
                 color_discrete_map=color_discrete_map)
    fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color=COLORS["bg2"], width=2)))
    apply_chart_theme(fig, title, height, show_legend=True)

    key = unique_key or f"click_{column_name}_donut"
    with st.container(border=True):
        event = st.plotly_chart(
            fig, width="stretch", config=CHART_CONFIG,
            on_select="rerun", selection_mode="points", key=key,
        )
    st.markdown('<div class="click-hint">Click a slice to filter the whole dashboard</div>', unsafe_allow_html=True)

    if event and event.get("selection", {}).get("points"):
        pt = event["selection"]["points"][0]
        clicked_val = pt.get("label")
        _register_click(column_name, clicked_val)


# ==============================================================
# 3c. NEW — CACHED INDIA GEOJSON LOADER (fixes speed / blank map)
# ==============================================================

@st.cache_data(show_spinner=False, ttl=3600)
def load_india_geojson():
    """Loaded ONCE per hour (cached), never re-read/re-fetched on every
    Streamlit rerun. Tries a local bundled file first (instant, no network),
    then a network fetch with a short timeout. Returns (geojson, featurekey)
    or (None, None) if neither source is available — callers must handle
    the None case with the existing ranked-bar-chart fallback."""
    for local_path in [INDIA_GEOJSON, "assets/india_states.geojson", "assets/india_state.geojson"]:
        if os.path.exists(local_path):
            try:
                with open(local_path, "r") as f:
                    return json.load(f), "properties.st_nm"
            except Exception:
                continue
    try:
        import urllib.request
        with urllib.request.urlopen(INDIA_GEOJSON_URL, timeout=5) as resp:
            return json.loads(resp.read().decode()), "properties.NAME_1"
    except Exception:
        return None, None


# ==============================================================
# 4. DATA LOADING & VALIDATION
# ==============================================================

REQUIRED_NUMERIC_COLS = [
    "Age", "Number_of_Referrals", "Tenure_in_Months", "Monthly_Charge",
    "Total_Charges", "Total_Refunds", "Total_Extra_Data_Charges",
    "Total_Long_Distance_Charges", "Total_Revenue", "Revenue_Per_Month",
    "Total_Extra_Charges", "Net_Revenue", "Charge_Revenue_Ratio",
    "Revenue_to_Charge_Ratio", "Service_Count", "Streaming_Service_Count",
    "Protection_Service_Count", "Average_Charge_Per_Month",
    "Charge_Difference", "Refund_Rate", "Extra_Charge_Ratio",
]

CATEGORICAL_STRIP_COLS = [
    "Gender", "Married", "State", "Value_Deal", "Phone_Service", "Multiple_Lines",
    "Internet_Service", "Internet_Type", "Online_Security", "Online_Backup",
    "Device_Protection_Plan", "Premium_Support", "Streaming_TV", "Streaming_Movies",
    "Streaming_Music", "Unlimited_Data", "Contract", "Paperless_Billing",
    "Payment_Method", "Customer_Status", "Churn_Category", "Churn_Reason",
    "Age_Group", "Tenure_Group", "Customer_Lifecycle", "Referral_Group",
    "Revenue_Segment",
]


@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "Customer_ID" in df.columns:
        df = df.drop_duplicates(subset="Customer_ID", keep="first")

    for col in CATEGORICAL_STRIP_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    for col in REQUIRED_NUMERIC_COLS:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    if "Customer_Status" in df.columns:
        df["Churn_Flag"] = df["Customer_Status"].astype(str).str.strip().str.lower().eq("churned").astype(int)
    elif "churn_flag" in df.columns:
        df["Churn_Flag"] = safe_numeric(df["churn_flag"]).fillna(0).astype(int)
    else:
        df["Churn_Flag"] = 0

    for col in CATEGORICAL_STRIP_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    if "Monthly_Charge" in df.columns:
        df["Monthly_Charge"] = df["Monthly_Charge"].fillna(0).clip(lower=0)

    if "Total_Revenue" in df.columns:
        df["Plot_Size"] = safe_numeric(df["Total_Revenue"]).fillna(0).clip(lower=0)
    else:
        df["Plot_Size"] = 1.0

    return df


try:
    raw_df = load_data(DATA_PATH)
    load_error = None
except Exception as e:
    raw_df = pd.DataFrame()
    load_error = str(e)

if raw_df.empty:
    st.error(f"⚠️ Could not load dataset from '{DATA_PATH}'. Error: {load_error}")
    st.stop()

# ==============================================================
# 5. HEADER + THEME SWITCHER
# ==============================================================
st.markdown(f"""
<div class="dash-header">
    <div class="dash-title">🤖 AI-Powered Customer Churn & Retention Analytics</div>
    <div class="dash-subtitle">Customer Health • Revenue Impact • Churn Drivers • Geographic Risk • Retention Opportunities</div>
</div>
""", unsafe_allow_html=True)

_theme_labels = {"Dark": "🌙 Dark", "Light / White": "☀️ Light", "Cream / Vanilla": "🍦 Cream"}
_sw_l, _sw_mid, _sw_r = st.columns([1, 1.1, 1])
with _sw_mid:
    st.segmented_control(
        "Theme",
        options=list(THEMES.keys()),
        format_func=lambda x: _theme_labels.get(x, x),
        key="theme_choice",
        label_visibility="collapsed",
    )

# ==============================================================
# 6. SIDEBAR — VERTICAL FILTER PANEL (multiselect + Apply/Clear/Reset)
# ==============================================================

FILTER_DEFS = [
    ("State", "wid_state", "flt_state"),
    ("Contract", "wid_contract", "flt_contract"),
    ("Internet_Type", "wid_internet", "flt_internet"),
    ("Customer_Status", "wid_status", "flt_status"),
    ("Gender", "wid_gender", "flt_gender"),
    ("Age_Group", "wid_age_group", "flt_age_group"),
    ("Tenure_Group", "wid_tenure_group", "flt_tenure_group"),
    ("Revenue_Segment", "wid_rev_segment", "flt_rev_segment"),
    ("Customer_Lifecycle", "wid_lifecycle", "flt_lifecycle"),
    ("Payment_Method", "wid_payment", "flt_payment"),
    ("Churn_Category", "wid_churn_cat", "flt_churn_cat"),
]

ALL_OPTIONS = {}
for col, _, _ in FILTER_DEFS:
    if col in raw_df.columns:
        ALL_OPTIONS[col] = sorted(raw_df[col].dropna().unique().tolist())

for col, wkey, fkey in FILTER_DEFS:
    if col not in ALL_OPTIONS:
        continue
    if wkey not in st.session_state:
        st.session_state[wkey] = ALL_OPTIONS[col].copy()
    if fkey not in st.session_state:
        st.session_state[fkey] = ALL_OPTIONS[col].copy()

if "search_text" not in st.session_state:
    st.session_state["search_text"] = ""
if "state_view" not in st.session_state:
    st.session_state["state_view"] = "Top 10 High-Risk"

# IMPORTANT: a widget's session_state key can only be written BEFORE that widget
# is instantiated in a given script run. Clear/Reset therefore set a pending flag
# and rerun; this block (which runs before the multiselect widgets below are
# created) applies the actual reset.
if st.session_state.get("_pending_filter_reset"):
    reset_mode = st.session_state.pop("_pending_filter_reset")
    for col, wkey, fkey in FILTER_DEFS:
        if col in ALL_OPTIONS:
            st.session_state[wkey] = ALL_OPTIONS[col].copy()
            st.session_state[fkey] = ALL_OPTIONS[col].copy()
    if reset_mode == "reset":
        st.session_state["search_text"] = ""
        st.session_state["state_view"] = "Top 10 High-Risk"
        st.session_state["chart_filters"] = {}

st.sidebar.markdown("### 🎛️ Dashboard Filters")
st.sidebar.markdown('<div class="sidebar-caption">Use the slicers to dynamically explore customer churn. Selections apply after clicking <b>Apply Filters</b>. You can also click directly on charts below to cross-filter.</div>', unsafe_allow_html=True)

for col, wkey, fkey in FILTER_DEFS:
    if col not in ALL_OPTIONS:
        continue
    st.sidebar.multiselect(col.replace("_", " "), options=ALL_OPTIONS[col], key=wkey)

st.sidebar.markdown("---")
b1, b2, b3 = st.sidebar.columns(3)
apply_clicked = b1.button("✅ Apply", width="stretch")
clear_clicked = b2.button("🧹 Clear", width="stretch")
reset_clicked = b3.button("🔄 Reset", width="stretch")

if apply_clicked:
    for col, wkey, fkey in FILTER_DEFS:
        if col in ALL_OPTIONS:
            st.session_state[fkey] = st.session_state[wkey]
    st.rerun()

if clear_clicked:
    st.session_state["_pending_filter_reset"] = "clear"
    st.rerun()

if reset_clicked:
    st.session_state["_pending_filter_reset"] = "reset"
    st.rerun()

st.sidebar.markdown("---")
active_sidebar_filters = sum(
    1 for col, wkey, fkey in FILTER_DEFS
    if col in ALL_OPTIONS and len(st.session_state[fkey]) < len(ALL_OPTIONS[col])
)
active_chart_filters = len(st.session_state["chart_filters"])
st.sidebar.markdown(
    f'<span class="filter-status-badge">🔎 {active_sidebar_filters} sidebar filter(s) '
    f'+ {active_chart_filters} chart filter(s)</span>',
    unsafe_allow_html=True,
)
if active_chart_filters and st.sidebar.button("🔄 Clear Chart Selections", width="stretch", key="sidebar_clear_chart"):
    st.session_state["chart_filters"] = {}
    st.rerun()

# --- sidebar filters ---
filtered_df = raw_df.copy()
for col, wkey, fkey in FILTER_DEFS:
    if col not in raw_df.columns:
        continue
    selected = st.session_state.get(fkey, ALL_OPTIONS.get(col, []))
    filtered_df = filtered_df[filtered_df[col].isin(selected)]

# --- NEW: chart click filters (AND-combined with sidebar filters) ---
df = filtered_df.copy()
for col, val in st.session_state["chart_filters"].items():
    if col in df.columns:
        df = df[df[col].astype(str) == str(val)]

# ==============================================================
# 7. CUSTOMER COUNT + ACTIVE FILTERS DISPLAY
# ==============================================================
total_raw = raw_df["Customer_ID"].nunique()
total_filtered = df["Customer_ID"].nunique()
st.markdown(f"""
<div class="count-wrap">
    <span class="count-pill">Showing {total_filtered:,} of {total_raw:,} customers</span>
</div>
""", unsafe_allow_html=True)

if st.session_state["chart_filters"]:
    chips = "".join(
        f'<span class="filter-chip">● {CLICKABLE_COLUMNS.get(col, col)}: {val}</span>'
        for col, val in st.session_state["chart_filters"].items()
    )
    st.markdown(f'<div class="active-filters-wrap"><b>Active Chart Filters:</b><br>{chips}</div>', unsafe_allow_html=True)
    afc1, afc2, _sp = st.columns([1, 1, 3])
    if afc1.button("🔄 Clear Chart Selections", width="stretch", key="top_clear_chart"):
        st.session_state["chart_filters"] = {}
        st.rerun()
    if afc2.button("🧹 Clear All Filters", width="stretch", key="top_clear_all"):
        st.session_state["chart_filters"] = {}
        st.session_state["_pending_filter_reset"] = "reset"
        st.rerun()

if df.empty:
    st.warning("⚠️ No customers match the selected filters. Use **Clear Chart Selections** above or **Clear/Reset** in the sidebar to restore the full dataset.")
    st.stop()

# ==============================================================
# 8. KPI CALCULATIONS
# ==============================================================
total_customers = df["Customer_ID"].nunique()
churned_customers = df.loc[df["Churn_Flag"] == 1, "Customer_ID"].nunique()
churn_rate = calculate_churn_rate(df)
retention_rate = 100 - churn_rate
total_revenue = df["Total_Revenue"].sum() if "Total_Revenue" in df.columns else 0.0
net_revenue = df["Net_Revenue"].sum() if "Net_Revenue" in df.columns else 0.0
revenue_per_customer = safe_divide(total_revenue, total_customers, 0.0)
avg_tenure = df["Tenure_in_Months"].mean() if "Tenure_in_Months" in df.columns else 0.0
if pd.isna(avg_tenure):
    avg_tenure = 0.0

# ==============================================================
# 9. KPI CARDS
# ==============================================================
st.markdown('<div class="section-header">📌 Executive Business KPIs</div>', unsafe_allow_html=True)


def kpi_card(icon, label, value, desc, color):
    return f"""
    <div class="kpi-card" style="--accent:{color};">
        <div class="kpi-icon-circle">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-desc">{desc}</div>
    </div>
    """


row1 = st.columns(4)
kpis_row1 = [
    ("👥", "Total Customers", format_number(total_customers), "Current customer base", COLORS["primary"]),
    ("🚨", "Churned Customers", format_number(churned_customers), "Customers lost", COLORS["danger"]),
    ("📉", "Churn Rate", f"{churn_rate:.2f}%", "Share of base churned", COLORS["warning"]),
    ("🛡️", "Retention Rate", f"{retention_rate:.2f}%", "Customers retained", COLORS["success"]),
]
for col, kpi in zip(row1, kpis_row1):
    with col:
        st.markdown(kpi_card(*kpi), unsafe_allow_html=True)

row2 = st.columns(4)
kpis_row2 = [
    ("💰", "Total Revenue", format_currency(total_revenue), "Gross revenue generated", COLORS["cyan"]),
    ("💎", "Net Revenue", format_currency(net_revenue), "Revenue after refunds", COLORS["purple"]),
    ("💳", "Revenue / Customer", format_currency(revenue_per_customer), "Average customer value", COLORS["blue"]),
    ("⏳", "Average Tenure", f"{avg_tenure:.1f} Months", "Average customer lifetime", COLORS["pink"]),
]
for col, kpi in zip(row2, kpis_row2):
    with col:
        st.markdown(kpi_card(*kpi), unsafe_allow_html=True)

# ==============================================================
# 10. SECTION — CUSTOMER HEALTH
# ==============================================================
st.markdown('<div class="section-header">👥 Customer Health</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    if "Customer_Status" in df.columns:
        status_counts = df.groupby("Customer_Status")["Customer_ID"].nunique().reset_index()
        status_counts.columns = ["Customer_Status", "Count"]
        # CLICKABLE — Customer Status donut (spec Example 1)
        clickable_donut_card(status_counts, names="Customer_Status", values="Count",
                              column_name="Customer_Status", title="Customer Status Distribution",
                              color_discrete_map=STATUS_COLOR_MAP, height=380)
    else:
        chart_card(empty_chart("Customer Status Distribution"))

with c2:
    if "Contract" in df.columns and "Customer_Status" in df.columns:
        cs_contract = df.groupby(["Contract", "Customer_Status"])["Customer_ID"].nunique().reset_index()
        cs_contract.columns = ["Contract", "Customer_Status", "Customers"]
        fig = safe_bar(cs_contract, x="Contract", y="Customers", color="Customer_Status",
                        title="Customer Status by Contract", color_discrete_map=STATUS_COLOR_MAP, barmode="stack")
        chart_card(fig)
    else:
        chart_card(empty_chart("Customer Status by Contract"))

c3, c4 = st.columns(2)
with c3:
    if "Age_Group" in df.columns and "Customer_Status" in df.columns:
        cs_age = df.groupby(["Age_Group", "Customer_Status"])["Customer_ID"].nunique().reset_index()
        cs_age.columns = ["Age_Group", "Customer_Status", "Customers"]
        fig = safe_bar(cs_age, x="Age_Group", y="Customers", color="Customer_Status",
                        title="Customer Status by Age Group", color_discrete_map=STATUS_COLOR_MAP, barmode="group")
        chart_card(fig)
    else:
        chart_card(empty_chart("Customer Status by Age Group"))

with c4:
    if "Customer_Lifecycle" in df.columns:
        lc_counts = df.groupby("Customer_Lifecycle")["Customer_ID"].nunique().reset_index()
        lc_counts.columns = ["Customer_Lifecycle", "Customers"]
        fig = safe_bar(lc_counts, x="Customer_Lifecycle", y="Customers", title="Customer Lifecycle Distribution",
                        color=None, color_seq=MULTI_COLOR_SEQUENCE)
        chart_card(fig)
    else:
        chart_card(empty_chart("Customer Lifecycle Distribution"))

# ==============================================================
# 11. SECTION — REVENUE & CHURN ANALYSIS
# ==============================================================
st.markdown('<div class="section-header">💰 Revenue & Churn Analysis</div>', unsafe_allow_html=True)

c5, c6 = st.columns(2)
with c5:
    if "Contract" in df.columns and "Total_Revenue" in df.columns:
        rev_contract = df.groupby("Contract")["Total_Revenue"].sum().reset_index().sort_values("Total_Revenue", ascending=False)
        fig = safe_bar(rev_contract, x="Contract", y="Total_Revenue", title="Revenue by Contract",
                        color_seq=[COLORS["cyan"]])
        chart_card(fig)
    else:
        chart_card(empty_chart("Revenue by Contract"))

with c6:
    cr_contract = group_churn_rate(df, "Contract")
    # CLICKABLE — Contract (spec Example 2)
    clickable_bar_card(cr_contract, x="Contract", y="Churn_Rate", column_name="Contract",
                        title="Churn Rate by Contract", color_seq=[COLORS["danger"]],
                        texttemplate="%{y:.1f}%")

c7, c8 = st.columns(2)
with c7:
    if "Customer_Lifecycle" in df.columns and "Total_Revenue" in df.columns:
        rev_lc = df.groupby("Customer_Lifecycle")["Total_Revenue"].sum().reset_index()
        fig = safe_bar(rev_lc, x="Customer_Lifecycle", y="Total_Revenue", title="Revenue by Customer Lifecycle",
                        color_seq=MULTI_COLOR_SEQUENCE)
        chart_card(fig)
    else:
        chart_card(empty_chart("Revenue by Customer Lifecycle"))

with c8:
    if "Revenue_Segment" in df.columns and "Customer_Status" in df.columns and "Total_Revenue" in df.columns:
        seg_perf = df.groupby(["Revenue_Segment", "Customer_Status"])["Total_Revenue"].sum().reset_index()
        fig = safe_bar(seg_perf, x="Revenue_Segment", y="Total_Revenue", color="Customer_Status",
                        title="Revenue Segment Performance", color_discrete_map=STATUS_COLOR_MAP, barmode="group")
        chart_card(fig)
    else:
        chart_card(empty_chart("Revenue Segment Performance"))

# ==============================================================
# 12. SECTION — SERVICE RISK ANALYSIS
# ==============================================================
st.markdown('<div class="section-header">🌐 Service Risk Analysis</div>', unsafe_allow_html=True)

c9, c10 = st.columns(2)
with c9:
    cr_itype = group_churn_rate(df, "Internet_Type")
    # CLICKABLE — Internet Type
    clickable_bar_card(cr_itype, x="Internet_Type", y="Churn_Rate", column_name="Internet_Type",
                        title="Churn Rate by Internet Type", color_seq=MULTI_COLOR_SEQUENCE,
                        texttemplate="%{y:.1f}%")

with c10:
    if "Contract" in df.columns and "Internet_Type" in df.columns:
        pivot_src = df.groupby(["Contract", "Internet_Type"]).agg(
            Customers=("Customer_ID", "nunique"), Churned=("Churn_Flag", "sum")
        ).reset_index()
        pivot_src["Churn_Rate"] = pivot_src.apply(lambda r: safe_divide(r["Churned"] * 100, r["Customers"], 0.0), axis=1)
        pivot = pivot_src.pivot(index="Contract", columns="Internet_Type", values="Churn_Rate")
        if not pivot.empty:
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                colorscale=[[0, COLORS["success"]], [0.5, COLORS["warning"]], [1, COLORS["danger"]]],
                text=np.round(pivot.values, 1), texttemplate="%{text}%",
                colorbar=dict(title="Churn %"),
            ))
            chart_card(apply_chart_theme(fig, "Contract × Internet Type Churn Rate", 380, show_legend=False))
        else:
            chart_card(empty_chart("Contract × Internet Type Churn Rate"))
    else:
        chart_card(empty_chart("Contract × Internet Type Churn Rate"))

if "Service_Count" in df.columns and "Customer_Status" in df.columns:
    svc_status = df.groupby(["Service_Count", "Customer_Status"])["Customer_ID"].nunique().reset_index()
    svc_status.columns = ["Service_Count", "Customer_Status", "Customers"]
    svc_status = svc_status.sort_values("Service_Count")
    fig = safe_bar(svc_status, x="Service_Count", y="Customers", color="Customer_Status",
                    title="Service Count vs Customer Status", color_discrete_map=STATUS_COLOR_MAP, barmode="group")
    chart_card(fig)
else:
    chart_card(empty_chart("Service Count vs Customer Status"))

# ==============================================================
# 13. SECTION — CUSTOMER VALUE & TENURE
# ==============================================================
st.markdown('<div class="section-header">⏳ Customer Value & Tenure</div>', unsafe_allow_html=True)

st.markdown("##### Customer Value vs Tenure")
scatter_cols = ["Tenure_in_Months", "Total_Revenue", "Customer_Status", "Plot_Size", "Customer_ID", "Monthly_Charge"]
scatter_df = clean_for_plot(df, [c for c in scatter_cols if c in df.columns])
if not scatter_df.empty and "Plot_Size" in scatter_df.columns:
    # SAFE marker size — never negative (fixes the known ValueError class)
    scatter_df["Plot_Size"] = safe_numeric(scatter_df["Plot_Size"]).fillna(0).clip(lower=0)
    scatter_df = scatter_df[np.isfinite(scatter_df["Plot_Size"])]
    scatter_df["Plot_Size_Display"] = scatter_df["Plot_Size"].clip(lower=1)
    if not scatter_df.empty:
        fig = px.scatter(
            scatter_df, x="Tenure_in_Months", y="Total_Revenue",
            color="Customer_Status", size="Plot_Size_Display",
            color_discrete_map=STATUS_COLOR_MAP, opacity=0.75,
            hover_data={"Customer_ID": True, "Monthly_Charge": ":.2f", "Plot_Size_Display": False},
        )
        chart_card(apply_chart_theme(fig, None, 450))
    else:
        chart_card(empty_chart("Customer Value vs Tenure", 450))
else:
    chart_card(empty_chart("Customer Value vs Tenure", 450))

c11, c12 = st.columns(2)
with c11:
    if "Tenure_Group" in df.columns and "Total_Revenue" in df.columns:
        rev_tenure = df.groupby("Tenure_Group")["Total_Revenue"].sum().reset_index().sort_values("Tenure_Group")
        fig = safe_bar(rev_tenure, x="Tenure_Group", y="Total_Revenue", title="Average Revenue by Tenure Group",
                        color_seq=[COLORS["cyan"]])
        chart_card(fig)
    else:
        chart_card(empty_chart("Average Revenue by Tenure Group"))

with c12:
    cr_tenure = group_churn_rate(df, "Tenure_Group").sort_values("Tenure_Group") if "Tenure_Group" in df.columns else pd.DataFrame()
    fig = safe_bar(cr_tenure, x="Tenure_Group", y="Churn_Rate", title="Churn Rate by Tenure Group",
                    color_seq=[COLORS["danger"]])
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    chart_card(fig)

# ==============================================================
# 14. SECTION — GEOGRAPHIC RISK ANALYSIS
# ==============================================================
st.markdown('<div class="section-header">🗺️ Geographic Risk Analysis</div>', unsafe_allow_html=True)

state_summary = pd.DataFrame()
if "State" in df.columns:
    state_summary = df.groupby("State").agg(
        Customers=("Customer_ID", "nunique"),
        Churned=("Churn_Flag", "sum"),
        Total_Revenue=("Total_Revenue", "sum"),
        Net_Revenue=("Net_Revenue", "sum"),
    ).reset_index()
    state_summary["Churn_Rate"] = state_summary.apply(lambda r: safe_divide(r["Churned"] * 100, r["Customers"], 0.0), axis=1)
    state_summary["Retention_Rate"] = 100 - state_summary["Churn_Rate"]
    state_summary["Revenue_Per_Customer"] = state_summary.apply(lambda r: safe_divide(r["Total_Revenue"], r["Customers"], 0.0), axis=1)

# FIX: geojson is now loaded through the cached function above — this line
# is now instant on every rerun after the first load (previously this was
# the main source of the "state map slow / not visible" problem if the
# geojson was being read/fetched inline, uncached, on every interaction).
geojson_data, geojson_key = load_india_geojson()

geo_col, rank_col = st.columns([1.3, 1])
with geo_col:
    if geojson_data is not None and not state_summary.empty:
        try:
            fig = px.choropleth(
                state_summary, geojson=geojson_data, locations="State", featureidkey=geojson_key,
                color="Churn_Rate",
                color_continuous_scale=[[0, COLORS["success"]], [0.5, COLORS["warning"]], [1, COLORS["danger"]]],
                hover_data={"State": True, "Customers": True, "Churned": True, "Churn_Rate": ":.2f",
                            "Total_Revenue": ":.0f", "Retention_Rate": ":.2f"},
            )
            fig.update_geos(fitbounds="locations", visible=False)
            chart_card(apply_chart_theme(fig, "India State-Level Churn Rate Map", 500, show_legend=False))
        except Exception:
            geojson_data = None  # fall through to the guaranteed bar-chart fallback below

    if geojson_data is None:
        st.markdown(
            '<div class="info-banner">ℹ️ India state GeoJSON is not available; displaying state-level geographic analysis as a ranked chart instead.</div>',
            unsafe_allow_html=True,
        )
        if not state_summary.empty:
            cr_state_all = state_summary.sort_values("Churn_Rate", ascending=True)
            fig = px.bar(cr_state_all, x="Churn_Rate", y="State", orientation="h",
                         color="Churn_Rate", color_continuous_scale=[[0, COLORS["success"]], [1, COLORS["danger"]]])
            chart_card(apply_chart_theme(fig, "Churn Rate by State (All)", 500, show_legend=False))
        else:
            chart_card(empty_chart("Churn Rate by State", 500))

with rank_col:
    view_choice = st.selectbox("State Risk Ranking View", ["Top 10 High-Risk", "Bottom 10 Lowest Risk", "All States"], key="state_view")
    if not state_summary.empty:
        ranked = state_summary.sort_values("Churn_Rate", ascending=False)
        if view_choice == "Top 10 High-Risk":
            plot_data = ranked.head(10).sort_values("Churn_Rate")
        elif view_choice == "Bottom 10 Lowest Risk":
            plot_data = ranked.tail(10).sort_values("Churn_Rate")
        else:
            plot_data = ranked.sort_values("Churn_Rate")
        # CLICKABLE — State (spec Example 3)
        clickable_bar_card(plot_data, x="Churn_Rate", y="State", column_name="State",
                            title=f"State Churn Ranking — {view_choice}", orientation="h",
                            height=500, color_seq=[COLORS["danger"]], texttemplate="%{x:.1f}%",
                            unique_key=f"click_state_{view_choice}")
    else:
        chart_card(empty_chart("State Churn Ranking", 500))

if not state_summary.empty:
    rev_state = state_summary.sort_values("Total_Revenue", ascending=True)
    fig = px.bar(rev_state, x="Total_Revenue", y="State", orientation="h",
                 color="Total_Revenue", color_continuous_scale=[[0, COLORS["blue"]], [1, COLORS["cyan"]]])
    chart_card(apply_chart_theme(fig, "State Revenue Ranking", 520, show_legend=False))
else:
    chart_card(empty_chart("State Revenue Ranking", 520))

# ==============================================================
# 15. SECTION — FINANCIAL-STYLE REVENUE RANGE ANALYSIS (CANDLE CHART)
# ==============================================================
st.markdown('<div class="section-header">📊 Customer Revenue Range — Financial-Style Analysis</div>', unsafe_allow_html=True)
st.caption("No transaction-date field exists in this dataset, so this chart shows the **statistical revenue range per Revenue Segment** (Open = 25th percentile, High = maximum, Low = minimum, Close = median) in a financial candle-style format. This is NOT a stock-market chart.")

if "Revenue_Segment" in df.columns and "Total_Revenue" in df.columns and not df.empty:
    seg_stats = df.groupby("Revenue_Segment")["Total_Revenue"].agg(
        Open=lambda x: x.quantile(0.25),
        High="max",
        Low="min",
        Close="median",
    ).reset_index()
    seg_order = ["Low Value", "Medium Value", "High Value", "Very High Value"]
    seg_stats["sort_key"] = seg_stats["Revenue_Segment"].apply(lambda x: seg_order.index(x) if x in seg_order else 99)
    seg_stats = seg_stats.sort_values("sort_key")
    if not seg_stats.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=seg_stats["Revenue_Segment"],
            open=seg_stats["Open"], high=seg_stats["High"],
            low=seg_stats["Low"], close=seg_stats["Close"],
            increasing_line_color=COLORS["success"], increasing_fillcolor=COLORS["success"],
            decreasing_line_color=COLORS["danger"], decreasing_fillcolor=COLORS["danger"],
            name="Revenue Range",
        )])
        fig.update_layout(xaxis_rangeslider_visible=False)
        chart_card(apply_chart_theme(fig, "Customer Revenue Range by Segment (Financial-Style)", 440, show_legend=False))
    else:
        chart_card(empty_chart("Customer Revenue Range Analysis", 440))
else:
    chart_card(empty_chart("Customer Revenue Range Analysis", 440))

# ==============================================================
# 16. SECTION — PAYMENT & BILLING RISK
# ==============================================================
st.markdown('<div class="section-header">💳 Payment & Billing Risk</div>', unsafe_allow_html=True)

c13, c14 = st.columns(2)
with c13:
    cr_pay = group_churn_rate(df, "Payment_Method")
    # CLICKABLE — Payment Method
    clickable_bar_card(cr_pay, x="Payment_Method", y="Churn_Rate", column_name="Payment_Method",
                        title="Churn Rate by Payment Method", color_seq=MULTI_COLOR_SEQUENCE,
                        texttemplate="%{y:.1f}%")

with c14:
    if "Payment_Method" in df.columns and "Total_Revenue" in df.columns:
        rev_pay = df.groupby("Payment_Method")["Total_Revenue"].sum().reset_index()
        fig = safe_bar(rev_pay, x="Payment_Method", y="Total_Revenue", title="Revenue by Payment Method",
                        color_seq=[COLORS["cyan"]])
        chart_card(fig)
    else:
        chart_card(empty_chart("Revenue by Payment Method"))

c15, c16 = st.columns(2)
with c15:
    if "Is_Automatic_Payment" in df.columns:
        tmp = df.copy()
        tmp["Auto_Pay_Label"] = tmp["Is_Automatic_Payment"].map({1: "Automatic Payment", 0: "Manual Payment"}).fillna("Unknown")
        cr_auto = group_churn_rate(tmp, "Auto_Pay_Label")
        fig = safe_bar(cr_auto, x="Auto_Pay_Label", y="Churn_Rate", title="Automatic Payment vs Churn",
                        color_seq=[COLORS["purple"], COLORS["primary"]])
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        chart_card(fig)
    else:
        chart_card(empty_chart("Automatic Payment vs Churn"))

with c16:
    if "Is_Paperless" in df.columns:
        tmp = df.copy()
        tmp["Paperless_Label"] = tmp["Is_Paperless"].map({1: "Paperless", 0: "Paper Billing"}).fillna("Unknown")
        cr_paper = group_churn_rate(tmp, "Paperless_Label")
        fig = safe_bar(cr_paper, x="Paperless_Label", y="Churn_Rate", title="Paperless Billing vs Churn",
                        color_seq=[COLORS["warning"], COLORS["success"]])
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        chart_card(fig)
    else:
        chart_card(empty_chart("Paperless Billing vs Churn"))

# ==============================================================
# 17. SECTION — CUSTOMER SEGMENTATION
# ==============================================================
st.markdown('<div class="section-header">🎯 Customer Segmentation</div>', unsafe_allow_html=True)

c17, c18 = st.columns(2)
with c17:
    if "Revenue_Segment" in df.columns and "Customer_Status" in df.columns:
        seg_status = df.groupby(["Revenue_Segment", "Customer_Status"])["Customer_ID"].nunique().reset_index()
        seg_status.columns = ["Revenue_Segment", "Customer_Status", "Customers"]
        fig = safe_bar(seg_status, x="Revenue_Segment", y="Customers", color="Customer_Status",
                        title="Customer Status by Revenue Segment", color_discrete_map=STATUS_COLOR_MAP, barmode="group")
        chart_card(fig)
    else:
        chart_card(empty_chart("Customer Status by Revenue Segment"))

with c18:
    cr_age = group_churn_rate(df, "Age_Group") if "Age_Group" in df.columns else pd.DataFrame()
    # CLICKABLE — Age Group
    clickable_bar_card(cr_age, x="Age_Group", y="Churn_Rate", column_name="Age_Group",
                        title="Churn Rate by Age Group", color_seq=MULTI_COLOR_SEQUENCE,
                        texttemplate="%{y:.1f}%")

c19, c20 = st.columns(2)
with c19:
    if "Customer_Lifecycle" in df.columns and "Customer_Status" in df.columns:
        lc_status = df.groupby(["Customer_Lifecycle", "Customer_Status"])["Customer_ID"].nunique().reset_index()
        lc_status.columns = ["Customer_Lifecycle", "Customer_Status", "Customers"]
        fig = safe_bar(lc_status, x="Customer_Lifecycle", y="Customers", color="Customer_Status",
                        title="Customer Status by Lifecycle", color_discrete_map=STATUS_COLOR_MAP, barmode="group")
        chart_card(fig)
    else:
        chart_card(empty_chart("Customer Status by Lifecycle"))

with c20:
    if "Referral_Group" in df.columns and "Total_Revenue" in df.columns:
        ref_perf = df.groupby("Referral_Group").agg(
            Customers=("Customer_ID", "nunique"), Churned=("Churn_Flag", "sum"), Revenue=("Total_Revenue", "sum")
        ).reset_index()
        ref_perf["Churn_Rate"] = ref_perf.apply(lambda r: safe_divide(r["Churned"] * 100, r["Customers"], 0.0), axis=1)
        fig = safe_bar(ref_perf, x="Referral_Group", y="Churn_Rate", title="Referral Group Performance (Churn %)",
                        color_seq=[COLORS["pink"], COLORS["cyan"], COLORS["primary"], COLORS["purple"]])
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        chart_card(fig)
    else:
        chart_card(empty_chart("Referral Group Performance"))

# ==============================================================
# 18. SECTION — CHURN DRIVERS
# ==============================================================
st.markdown('<div class="section-header">🚨 Churn Drivers</div>', unsafe_allow_html=True)

churned_df = df[df["Churn_Flag"] == 1].copy()

c21, c22 = st.columns(2)
with c21:
    if not churned_df.empty and "Churn_Category" in churned_df.columns:
        cat_counts = churned_df.groupby("Churn_Category")["Customer_ID"].nunique().reset_index()
        cat_counts.columns = ["Churn_Category", "Customers"]
        cat_counts = cat_counts.sort_values("Customers", ascending=False)
        # CLICKABLE — Churn Category
        clickable_bar_card(cat_counts, x="Churn_Category", y="Customers", column_name="Churn_Category",
                            title="Churn Category Distribution", color_seq=MULTI_COLOR_SEQUENCE)
    else:
        chart_card(empty_chart("Churn Category Distribution"))

with c22:
    if not churned_df.empty and "Churn_Reason" in churned_df.columns:
        reason_counts = churned_df.groupby("Churn_Reason")["Customer_ID"].nunique().reset_index()
        reason_counts.columns = ["Churn_Reason", "Customers"]
        reason_counts = reason_counts.sort_values("Customers", ascending=False).head(10).sort_values("Customers")
        fig = px.bar(reason_counts, x="Customers", y="Churn_Reason", orientation="h",
                     color="Customers", color_continuous_scale=[[0, COLORS["warning"]], [1, COLORS["danger"]]])
        chart_card(apply_chart_theme(fig, "Top 10 Churn Reasons", 420, show_legend=False))
    else:
        chart_card(empty_chart("Top 10 Churn Reasons", 420))

c23, c24 = st.columns(2)
with c23:
    if not churned_df.empty and "Churn_Reason" in churned_df.columns and "Total_Revenue" in churned_df.columns:
        reason_rev = churned_df.groupby("Churn_Reason")["Total_Revenue"].sum().reset_index()
        reason_rev = reason_rev.sort_values("Total_Revenue", ascending=False).head(10).sort_values("Total_Revenue")
        fig = px.bar(reason_rev, x="Total_Revenue", y="Churn_Reason", orientation="h",
                     color="Total_Revenue", color_continuous_scale=[[0, COLORS["pink"]], [1, COLORS["danger"]]])
        chart_card(apply_chart_theme(fig, "Churn Reason — Revenue Exposure", 420, show_legend=False))
    else:
        chart_card(empty_chart("Churn Reason — Revenue Exposure", 420))

with c24:
    if not churned_df.empty and "Churn_Category" in churned_df.columns and "Contract" in churned_df.columns:
        cat_contract = churned_df.groupby(["Contract", "Churn_Category"])["Customer_ID"].nunique().reset_index()
        cat_contract.columns = ["Contract", "Churn_Category", "Customers"]
        fig = safe_bar(cat_contract, x="Contract", y="Customers", color="Churn_Category",
                        title="Churn Category by Contract", color_seq=MULTI_COLOR_SEQUENCE, barmode="stack")
        chart_card(fig)
    else:
        chart_card(empty_chart("Churn Category by Contract"))

# ==============================================================
# 19. SUMMARY TABLES
# ==============================================================
st.markdown('<div class="section-header">🧾 Summary Tables</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🗺️ State Performance", "📄 Contract Performance", "💎 Revenue Segment", "🚨 Churn Drivers"])

with tab1:
    if not state_summary.empty:
        tbl = state_summary[["State", "Customers", "Churned", "Churn_Rate", "Retention_Rate", "Total_Revenue", "Net_Revenue", "Revenue_Per_Customer"]].copy()
        tbl = tbl.sort_values("Churn_Rate", ascending=False)
        st.dataframe(
            tbl, hide_index=True, width="stretch", height=360,
            column_config={
                "Churn_Rate": st.column_config.ProgressColumn("Churn Rate %", min_value=0, max_value=100, format="%.1f%%"),
                "Retention_Rate": st.column_config.ProgressColumn("Retention %", min_value=0, max_value=100, format="%.1f%%"),
                "Total_Revenue": st.column_config.NumberColumn("Total Revenue", format="₹%,.0f"),
                "Net_Revenue": st.column_config.NumberColumn("Net Revenue", format="₹%,.0f"),
                "Revenue_Per_Customer": st.column_config.NumberColumn("Revenue / Customer", format="₹%,.0f"),
            },
        )
    else:
        st.info("No sufficient data available for this analysis.")

with tab2:
    if "Contract" in df.columns:
        ct = df.groupby("Contract").agg(
            Customers=("Customer_ID", "nunique"), Churned=("Churn_Flag", "sum"), Revenue=("Total_Revenue", "sum")
        ).reset_index()
        ct["Churn_Rate"] = ct.apply(lambda r: safe_divide(r["Churned"] * 100, r["Customers"], 0.0), axis=1)
        ct["Revenue_Per_Customer"] = ct.apply(lambda r: safe_divide(r["Revenue"], r["Customers"], 0.0), axis=1)
        ct = ct.sort_values("Churn_Rate", ascending=False)
        st.dataframe(
            ct, hide_index=True, width="stretch", height=200,
            column_config={
                "Churn_Rate": st.column_config.ProgressColumn("Churn Rate %", min_value=0, max_value=100, format="%.1f%%"),
                "Revenue": st.column_config.NumberColumn("Revenue", format="₹%,.0f"),
                "Revenue_Per_Customer": st.column_config.NumberColumn("Revenue / Customer", format="₹%,.0f"),
            },
        )
    else:
        st.info("No sufficient data available for this analysis.")

with tab3:
    if "Revenue_Segment" in df.columns:
        rst = df.groupby("Revenue_Segment").agg(
            Customers=("Customer_ID", "nunique"), Churned=("Churn_Flag", "sum"), Total_Revenue=("Total_Revenue", "sum")
        ).reset_index()
        rst["Churn_Rate"] = rst.apply(lambda r: safe_divide(r["Churned"] * 100, r["Customers"], 0.0), axis=1)
        rst = rst.sort_values("Total_Revenue", ascending=False)
        st.dataframe(
            rst, hide_index=True, width="stretch", height=200,
            column_config={
                "Churn_Rate": st.column_config.ProgressColumn("Churn Rate %", min_value=0, max_value=100, format="%.1f%%"),
                "Total_Revenue": st.column_config.NumberColumn("Total Revenue", format="₹%,.0f"),
            },
        )
    else:
        st.info("No sufficient data available for this analysis.")

with tab4:
    if not churned_df.empty and "Churn_Category" in churned_df.columns:
        cds = churned_df.groupby("Churn_Category").agg(
            Churned_Customers=("Customer_ID", "nunique"),
            Revenue_Exposure=("Total_Revenue", "sum"),
            Average_Tenure=("Tenure_in_Months", "mean"),
        ).reset_index().sort_values("Revenue_Exposure", ascending=False)
        st.dataframe(
            cds, hide_index=True, width="stretch", height=220,
            column_config={
                "Revenue_Exposure": st.column_config.NumberColumn("Revenue Exposure", format="₹%,.0f"),
                "Average_Tenure": st.column_config.NumberColumn("Avg Tenure (Months)", format="%.1f"),
            },
        )
    else:
        st.info("No churned customers in the current filtered view.")

# ==============================================================
# 20. CUSTOMER-LEVEL ANALYSIS
# ==============================================================
st.markdown('<div class="section-header">📋 Customer-Level Analysis</div>', unsafe_allow_html=True)

with st.expander("🔍 Open Customer-Level Analysis", expanded=False):
    st.text_input("Search by Customer ID or State", key="search_text", placeholder="e.g. 1234-ABCDE or Maharashtra")

    table_cols = [
        "Customer_ID", "Gender", "Age", "State", "Tenure_in_Months", "Contract",
        "Internet_Type", "Monthly_Charge", "Total_Revenue", "Net_Revenue",
        "Customer_Status", "Churn_Category", "Churn_Reason", "Revenue_Segment", "Customer_Lifecycle",
    ]
    table_cols = [c for c in table_cols if c in df.columns]
    table_df = df[table_cols]

    search_val = st.session_state.get("search_text", "").strip().lower()
    if search_val:
        mask = pd.Series(False, index=table_df.index)
        if "Customer_ID" in table_df.columns:
            mask |= table_df["Customer_ID"].astype(str).str.lower().str.contains(search_val, na=False)
        if "State" in table_df.columns:
            mask |= table_df["State"].astype(str).str.lower().str.contains(search_val, na=False)
        table_df = table_df[mask]

    st.caption(f"Showing {len(table_df):,} matching record(s)")
    st.dataframe(table_df, width="stretch", height=380)

# ==============================================================
# 21. EXPORT
# ==============================================================
st.markdown('<div class="section-header">⬇️ Export Filtered Data</div>', unsafe_allow_html=True)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered CSV",
    data=csv_bytes,
    file_name="filtered_churn_data.csv",
    mime="text/csv",
)

# ==============================================================
# 22. EXECUTIVE BUSINESS SUMMARY
# ==============================================================
st.markdown('<div class="section-header">🧠 Executive Business Summary</div>', unsafe_allow_html=True)


def top_category(data, dim, agg="count", value_col=None):
    if dim not in data.columns or data.empty:
        return "N/A"
    try:
        if agg == "count":
            g = data.groupby(dim)["Customer_ID"].nunique()
        else:
            g = data.groupby(dim)[value_col].sum()
        return str(g.idxmax()) if not g.empty else "N/A"
    except Exception:
        return "N/A"


highest_churn_contract = "N/A"
if "Contract" in df.columns:
    g = group_churn_rate(df, "Contract")
    if not g.empty:
        highest_churn_contract = g.iloc[0]["Contract"]

highest_churn_internet = "N/A"
if "Internet_Type" in df.columns:
    g = group_churn_rate(df, "Internet_Type")
    if not g.empty:
        highest_churn_internet = g.iloc[0]["Internet_Type"]

highest_churn_state = "N/A"
highest_revenue_state = "N/A"
if not state_summary.empty:
    highest_churn_state = state_summary.sort_values("Churn_Rate", ascending=False).iloc[0]["State"]
    highest_revenue_state = state_summary.sort_values("Total_Revenue", ascending=False).iloc[0]["State"]

most_common_churn_category = top_category(churned_df, "Churn_Category")
most_common_churn_reason = top_category(churned_df, "Churn_Reason")

highest_revenue_segment = "N/A"
if "Revenue_Segment" in df.columns and "Total_Revenue" in df.columns:
    g = df.groupby("Revenue_Segment")["Total_Revenue"].sum()
    if not g.empty:
        highest_revenue_segment = g.idxmax()

churned_revenue_exposure = churned_df["Total_Revenue"].sum() if (not churned_df.empty and "Total_Revenue" in churned_df.columns) else 0.0

st.markdown(f"""
<div class="summary-box">
    The current filtered view covers <b>{format_number(total_customers)}</b> customers, of whom
    <b>{format_number(churned_customers)}</b> have churned — a churn rate of <b>{churn_rate:.2f}%</b>
    and a retention rate of <b>{retention_rate:.2f}%</b>. Together these customers generate
    <b>{format_currency(total_revenue)}</b> in total revenue, averaging <b>{format_currency(revenue_per_customer)}</b>
    per customer, with an average tenure of <b>{avg_tenure:.1f} months</b>.<br><br>
    <b>{highest_churn_contract}</b> contracts show the highest churn among contract types, while
    <b>{highest_churn_internet}</b> is the highest-risk internet type. Geographically,
    <b>{highest_churn_state}</b> records the highest churn rate, whereas <b>{highest_revenue_state}</b>
    generates the most revenue. The most common churn category is <b>{most_common_churn_category}</b>,
    most frequently driven by <b>{most_common_churn_reason}</b>. The <b>{highest_revenue_segment}</b>
    segment contributes the highest revenue overall, while churned customers represent
    <b>{format_currency(churned_revenue_exposure)}</b> in total revenue exposure.
</div>
""", unsafe_allow_html=True)

# ==============================================================
# 23. BUSINESS RECOMMENDATIONS
# ==============================================================
st.markdown('<div class="section-header">🎯 Recommended Business Actions</div>', unsafe_allow_html=True)

recommendations = []

if churn_rate > 0:
    recommendations.append(
        f"Churn currently stands at <b>{churn_rate:.2f}%</b> — prioritize retention campaigns for high-value, high-risk customers to protect revenue."
    )
if highest_churn_contract != "N/A":
    recommendations.append(
        f"<b>{highest_churn_contract}</b> contracts show elevated churn — investigate contract pricing, onboarding, and engagement for this segment."
    )
if highest_churn_internet != "N/A":
    recommendations.append(
        f"<b>{highest_churn_internet}</b> internet type has the highest churn risk — review service quality and support experience."
    )
if highest_churn_state != "N/A":
    recommendations.append(
        f"<b>{highest_churn_state}</b> has the highest churn rate among states — investigate regional service quality and allocate additional retention resources there."
    )
if not churned_df.empty and "Tenure_in_Months" in churned_df.columns:
    short_tenure_share = safe_divide((churned_df["Tenure_in_Months"] <= 6).sum() * 100, len(churned_df), 0.0)
    if short_tenure_share > 0:
        recommendations.append(
            f"About <b>{short_tenure_share:.1f}%</b> of churned customers had a tenure of 6 months or less — strengthen onboarding and early lifecycle engagement."
        )
if most_common_churn_reason != "N/A":
    recommendations.append(
        f"The leading churn reason is <b>{most_common_churn_reason}</b> — since it represents the largest financial opportunity, prioritize interventions that directly address it."
    )
if "Is_Automatic_Payment" in df.columns:
    tmp = df.copy()
    tmp["Auto_Pay_Label"] = tmp["Is_Automatic_Payment"].map({1: "Automatic Payment", 0: "Manual Payment"})
    cr_auto = group_churn_rate(tmp.dropna(subset=["Auto_Pay_Label"]), "Auto_Pay_Label")
    if not cr_auto.empty and "Manual Payment" in cr_auto["Auto_Pay_Label"].values:
        manual_rate = cr_auto.loc[cr_auto["Auto_Pay_Label"] == "Manual Payment", "Churn_Rate"].values[0]
        if manual_rate > churn_rate:
            recommendations.append(
                "Customers on manual payment methods churn at a higher rate — encourage migration to automatic payments through incentives."
            )
if churned_revenue_exposure > 0:
    recommendations.append(
        f"Churned customers represent <b>{format_currency(churned_revenue_exposure)}</b> in lost revenue exposure — focus win-back offers on the highest-value churned accounts."
    )

if not recommendations:
    recommendations.append("No specific risk signals detected in the current filtered data. Continue monitoring churn trends regularly.")

for rec in recommendations:
    st.markdown(f'<div class="action-card">✅ {rec}</div>', unsafe_allow_html=True)

# ==============================================================
# 24. FOOTER
# ==============================================================

st.markdown("""
<h3 style="text-align: center;">
    🤖 AI-Powered Customer Churn & Retention Analytics
</h3>
""", unsafe_allow_html=True)

st.markdown("""
<p style="text-align: center;">
    Built with Python • Pandas • Plotly • Streamlit
</p>
""", unsafe_allow_html=True)

st.markdown("""
<p style="text-align: center;">
    👨‍💻 <strong>S Mohammed Kaif</strong><br>
    Data Scientist • Data Analyst • Machine Learning Engineer
</p>
""", unsafe_allow_html=True)

st.markdown("""
<p style="text-align: center;">
    🐙 <a href="https://github.com/Shaik-Mohammed-Kaif" target="_blank">
    GitHub
    </a>
    &nbsp; • &nbsp;
    💼 <a href="https://www.linkedin.com/in/s-mohammed-kaif-2a500a341" target="_blank">
    LinkedIn
    </a>
</p>
""", unsafe_allow_html=True)

st.markdown("""
<p style="text-align: center; opacity: 0.6; font-size: 12px;">
    © 2026 S Mohammed Kaif • AI-Powered Customer Churn & Retention Analytics
</p>
""", unsafe_allow_html=True)