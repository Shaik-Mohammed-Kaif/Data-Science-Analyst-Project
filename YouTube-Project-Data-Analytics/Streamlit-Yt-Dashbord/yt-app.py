"""
=====================================================================================
YOUTUBE INTELLIGENCE — Analytics, Prediction & Content Performance AI
=====================================================================================
A single-file, end-to-end Streamlit application that combines descriptive
analytics / BI with a machine-learning layer (regression + classification)
trained on the actual YouTube trending-videos dataset bundled with this app.

Run:
    streamlit run app.py

Author: S Mohammed Kaif
GitHub:   https://github.com/Shaik-Mohammed-Kaif
LinkedIn: https://www.linkedin.com/in/s-mohammed-kaif-2a500a341

Layout notes:
  - No sidebar. Navigation is a single horizontal dropdown placed directly
    under the (centered) app title, alongside a compact theme selector.
  - A slim horizontal status strip (dataset / model / quick stats) replaces
    what used to live in the sidebar.
  - Author name/title and the GitHub/LinkedIn buttons appear in exactly ONE
    place: the footer at the bottom of every page. Nowhere else.

Design rules honoured throughout this file (see also the "Model Information"
and "About" sections inside the app):
  - No fabricated KPIs, metrics, or predictions — everything shown is computed
    live from the dataset / trained models, or explicitly labelled
    "Not available for the current dataset."
  - Target-leakage prevention: view-derived columns (likes, comments,
    engagement rate, etc.) are excluded from the ML feature set used to
    predict views, since they are only known *after* a video has already
    accumulated views.
  - Classification thresholds (LOW / MEDIUM / HIGH) are computed from the
    TRAINING split only, then applied to train/test/new predictions alike.
  - random_state = 42 everywhere for reproducibility.
  - Multi-model selection: the user explicitly picks WHICH trained model
    (regression or classification) is used to generate each prediction —
    the app does not silently force the "best" model on the user.
=====================================================================================
"""

from __future__ import annotations

import ast
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ============================================================
# APP CONFIG
# ============================================================

RANDOM_STATE = 42
APP_VERSION = "2.0.0"

# ------------------------------------------------------------
# Application Identity
# ------------------------------------------------------------

APP_NAME = "YouTube Intelligence"
APP_SUBTITLE = "Analytics • Prediction • Classification • Business Intelligence"

DATA_PATH = Path(__file__).parent / "trending_videos.csv"

# ------------------------------------------------------------
# Author / Portfolio Branding
# ------------------------------------------------------------

AUTHOR_NAME = "S Mohammed Kaif"
AUTHOR_TITLE = "Data Science • Analytics • Machine Learning"

GITHUB_URL = "https://github.com/Shaik-Mohammed-Kaif"
LINKEDIN_URL = "https://www.linkedin.com/in/s-mohammed-kaif-2a500a341"

# ------------------------------------------------------------
# ML Configuration
# ------------------------------------------------------------

TARGET_REGRESSION = "view_count"
TARGET_CLASSIFICATION = "performance_class"

# Reproducibility
RANDOM_STATE = 42

# ------------------------------------------------------------
# Streamlit Page Configuration
# MUST BE THE FIRST STREAMLIT COMMAND
# ------------------------------------------------------------

st.set_page_config(
    page_title=f"{APP_NAME} | {AUTHOR_NAME}",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================================
# THEME SYSTEM
# =====================================================================================

THEMES = {

    # =========================================================================
    # 01 — PRIMARY YOUTUBE INTELLIGENCE THEME
    # =========================================================================
    "YouTube Red (Bubbles)": {
        "bg": "#0F0F0F",

        "bg_grad": (
            "radial-gradient(circle at 12% 18%, rgba(255,0,51,0.20), transparent 42%), "
            "radial-gradient(circle at 88% 12%, rgba(255,65,95,0.14), transparent 38%), "
            "radial-gradient(circle at 50% 88%, rgba(255,0,51,0.13), transparent 45%), "
            "linear-gradient(135deg, #0F0F0F 0%, #121212 50%, #0B0B0B 100%)"
        ),

        "card": "rgba(24,24,24,0.92)",
        "card_border": "rgba(255,0,51,0.25)",
        "card_border_hover": "rgba(255,0,51,0.55)",

        "text": "#F5F5F5",
        "muted": "#A7A7A7",
        "text_soft": "#D7D7D7",

        "primary": "#FF0033",
        "primary_hover": "#FF1744",
        "primary_soft": "rgba(255,0,51,0.14)",

        "accent": "#FF4D6D",
        "accent_soft": "rgba(255,77,109,0.14)",

        "success": "#3BB273",
        "success_soft": "rgba(59,178,115,0.15)",

        "warn": "#F5A623",
        "warn_soft": "rgba(245,166,35,0.15)",

        "danger": "#FF4057",
        "danger_soft": "rgba(255,64,87,0.15)",

        "sidebar_bg": "#111111",

        "input_bg": "#181818",
        "input_border": "#303030",

        "plot_template": "plotly_dark",

        "bubbles": True,
        "glass": True,
        "glow": True,
    },


    # =========================================================================
    # 02 — PROFESSIONAL DARK
    # =========================================================================
    "Dark": {
        "bg": "#0E1117",
        "bg_grad": (
            "radial-gradient(circle at 20% 10%, rgba(91,141,239,0.08), transparent 40%), "
            "radial-gradient(circle at 85% 80%, rgba(124,77,255,0.07), transparent 40%), "
            "#0E1117"
        ),

        "card": "rgba(26,29,38,0.94)",
        "card_border": "#2B2F3A",
        "card_border_hover": "#454B5A",

        "text": "#E8E8E8",
        "muted": "#9AA0AC",
        "text_soft": "#C9CCD3",

        "primary": "#5B8DEF",
        "primary_hover": "#719DFF",
        "primary_soft": "rgba(91,141,239,0.15)",

        "accent": "#7C4DFF",
        "accent_soft": "rgba(124,77,255,0.14)",

        "success": "#3BB273",
        "success_soft": "rgba(59,178,115,0.15)",

        "warn": "#F5A623",
        "warn_soft": "rgba(245,166,35,0.15)",

        "danger": "#FF5C70",
        "danger_soft": "rgba(255,92,112,0.15)",

        "sidebar_bg": "#12141C",

        "input_bg": "#171A22",
        "input_border": "#303542",

        "plot_template": "plotly_dark",

        "bubbles": False,
        "glass": True,
        "glow": True,
    },


    # =========================================================================
    # 03 — CREAM / VANILLA PROFESSIONAL
    # =========================================================================
    "Cream / Vanilla": {
        "bg": "#FBF6EC",

        "bg_grad": (
            "radial-gradient(circle at 15% 15%, rgba(201,138,44,0.08), transparent 40%), "
            "radial-gradient(circle at 85% 85%, rgba(178,58,46,0.06), transparent 42%), "
            "#FBF6EC"
        ),

        "card": "#FFFFFF",
        "card_border": "#E9DFC8",
        "card_border_hover": "#D4C39F",

        "text": "#2E2A22",
        "muted": "#7A7160",
        "text_soft": "#514A3D",

        "primary": "#B23A2E",
        "primary_hover": "#C34A3D",
        "primary_soft": "rgba(178,58,46,0.10)",

        "accent": "#C98A2C",
        "accent_soft": "rgba(201,138,44,0.12)",

        "success": "#3A8452",
        "success_soft": "rgba(58,132,82,0.12)",

        "warn": "#B8791A",
        "warn_soft": "rgba(184,121,26,0.12)",

        "danger": "#B23A2E",
        "danger_soft": "rgba(178,58,46,0.10)",

        "sidebar_bg": "#F3ECDB",

        "input_bg": "#FFFDF8",
        "input_border": "#DED2B9",

        "plot_template": "plotly_white",

        "bubbles": False,
        "glass": False,
        "glow": False,
    },


    # =========================================================================
    # 04 — CLEAN LIGHT
    # =========================================================================
    "Light": {
        "bg": "#FAFAFA",

        "bg_grad": (
            "radial-gradient(circle at 20% 10%, rgba(255,0,51,0.045), transparent 40%), "
            "#FAFAFA"
        ),

        "card": "#FFFFFF",
        "card_border": "#ECECEC",
        "card_border_hover": "#D6D6D6",

        "text": "#0F0F0F",
        "muted": "#606060",
        "text_soft": "#333333",

        "primary": "#FF0033",
        "primary_hover": "#E6002E",
        "primary_soft": "rgba(255,0,51,0.08)",

        "accent": "#2E86AB",
        "accent_soft": "rgba(46,134,171,0.10)",

        "success": "#3BB273",
        "success_soft": "rgba(59,178,115,0.10)",

        "warn": "#D97706",
        "warn_soft": "rgba(217,119,6,0.10)",

        "danger": "#D92D45",
        "danger_soft": "rgba(217,45,69,0.10)",

        "sidebar_bg": "#FFFFFF",

        "input_bg": "#FFFFFF",
        "input_border": "#D9D9D9",

        "plot_template": "plotly_white",

        "bubbles": False,
        "glass": False,
        "glow": False,
    },
}


def inject_theme(theme_name: str) -> dict:
    t = THEMES.get(theme_name, THEMES["YouTube Red (Bubbles)"])

    bubble_specs = [
        (90, 4, 0.0, 16, 0.16), (150, 14, 2.5, 21, 0.10), (55, 24, 0.8, 13, 0.20),
        (120, 33, 4.5, 19, 0.12), (70, 43, 1.6, 15, 0.18), (160, 52, 6.0, 24, 0.08),
        (50, 61, 0.3, 12, 0.22), (105, 70, 3.2, 18, 0.14), (80, 79, 1.1, 14, 0.19),
        (135, 88, 5.0, 22, 0.10), (45, 95, 2.0, 11, 0.24), (95, 9, 3.7, 17, 0.15),
    ]
    bubbles_html = ""
    if t["bubbles"]:
        bubbles_html = "".join(
            f'<div class="yti-bubble" style="'
            f'width:{w}px;height:{w}px;left:{l}%;animation-delay:{d}s;animation-duration:{dur}s;'
            f'opacity:{op};"></div>'
            for w, l, d, dur, op in bubble_specs
        )

    st.markdown(
        f"""
        <style>
        @keyframes floatUp {{
            0%   {{ transform: translateY(115vh) scale(0.75) translateX(0px); opacity: 0; }}
            8%   {{ opacity: 1; }}
            50%  {{ transform: translateY(45vh) scale(1.05) translateX(18px); }}
            92%  {{ opacity: 1; }}
            100% {{ transform: translateY(-22vh) scale(1.2) translateX(-10px); opacity: 0; }}
        }}
        .yti-bubble-layer {{ position: fixed; inset: 0; overflow: hidden; z-index: 0; pointer-events: none; }}
        .yti-bubble {{
            position: absolute; bottom: -10vh; border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, rgba(255,110,130,0.95), rgba(255,0,51,0.25) 55%, rgba(255,0,51,0.02) 80%);
            box-shadow: 0 0 22px rgba(255,0,51,0.35);
            filter: blur(0.3px);
            animation-name: floatUp; animation-timing-function: ease-in-out; animation-iteration-count: infinite;
        }}
        html, body {{ background-color: {t['bg']}; }}
        .stApp {{ background: {t['bg_grad']}; color: {t['text']}; }}
        [data-testid="stAppViewContainer"] {{ position: relative; z-index: 1; }}
        [data-testid="stHeader"] {{ background: transparent; }}

        /* Sidebar fully hidden — navigation lives in the horizontal top bar */
        [data-testid="stSidebar"] {{ display: none; }}
        [data-testid="collapsedControl"] {{ display: none; }}

        .block-container {{ padding-top: 1.6rem; max-width: 1200px; }}

        h1, h2, h3, h4, p, span, label, .stMarkdown {{ color: {t['text']}; }}

        .yti-card {{
            position: relative; z-index: 1;
            background: {t['card']};
            border: 1px solid {t['card_border']};
            border-left: 5px solid {t['primary']};
            border-radius: 14px;
            padding: 16px 20px;
            box-shadow: 0 2px 14px rgba(0,0,0,0.18);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            margin-bottom: 10px;
        }}
        .yti-card:hover {{ transform: translateY(-2px); box-shadow: 0 6px 22px {t['primary_soft']}; }}
        .yti-label {{ font-size: 0.78rem; color: {t['muted']}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }}
        .yti-value {{ font-size: 1.65rem; font-weight: 800; color: {t['text']}; }}
        .yti-sub {{ font-size: 0.78rem; color: {t['muted']}; margin-top: 2px; }}

        .yti-badge {{
            display: inline-block; padding: 4px 14px; border-radius: 999px;
            font-weight: 700; font-size: 0.85rem; letter-spacing: 0.03em;
        }}
        .yti-badge-high {{ background: rgba(59,178,115,0.18); color: {t['success']}; border: 1px solid {t['success']}; }}
        .yti-badge-medium {{ background: rgba(245,166,35,0.18); color: {t['warn']}; border: 1px solid {t['warn']}; }}
        .yti-badge-low {{ background: rgba(255,0,51,0.14); color: {t['primary']}; border: 1px solid {t['primary']}; }}

        .yti-result-card {{
            position: relative; z-index: 1;
            background: linear-gradient(135deg, {t['primary_soft']}, transparent);
            border: 1px solid {t['card_border']};
            border-radius: 18px; padding: 28px 30px; text-align: center;
            box-shadow: 0 4px 26px {t['primary_soft']};
        }}

        /* ---- Centered hero title ---- */
        .yti-hero-wrap {{ text-align: center; margin-bottom: 6px; }}
        .yti-hero-title {{
            font-size: 2.5rem; font-weight: 900; color: {t['text']};
            letter-spacing: -0.02em; margin-bottom: 0;
        }}
        .yti-hero-title span {{ color: {t['primary']}; }}
        .yti-hero-sub {{ font-size: 1rem; color: {t['muted']}; margin-top: 2px; margin-bottom: 14px; }}

        .yti-nav-card {{
            background: {t['card']}; border: 1px solid {t['card_border']};
            border-radius: 12px; padding: 14px 16px; margin-bottom: 8px;
            border-top: 3px solid {t['primary']};
        }}

        div.stButton > button {{
            background: {t['primary']}; color: white; border: none;
            border-radius: 10px; padding: 0.55em 1.4em; font-weight: 700;
            transition: transform 0.1s ease, box-shadow 0.15s ease;
        }}
        div.stButton > button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 14px {t['primary_soft']}; }}
        hr {{ border-color: {t['card_border']}; }}
        footer {{ visibility: hidden; }}

        /* ---- Horizontal control bar (theme + page nav) ---- */
        .yti-controlbar {{
            position: relative; z-index: 1;
            background: {t['card']};
            border: 1px solid {t['card_border']};
            border-radius: 14px;
            padding: 10px 16px;
            margin-bottom: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.12);
        }}

        /* ---- Horizontal status strip ---- */
        .yti-status-strip {{
            position: relative; z-index: 1;
            display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;
            margin-bottom: 18px;
        }}
        .yti-status-pill {{
            display: inline-flex; align-items: center; gap: 6px;
            background: {t['card']};
            border: 1px solid {t['card_border']};
            border-radius: 999px;
            padding: 6px 16px;
            font-size: 0.82rem;
            color: {t['text']};
        }}
        .yti-status-pill b {{ color: {t['primary']}; }}

        /* ---- Footer (the ONLY place author/social buttons appear) ---- */
        .yti-app-footer {{
            position: relative; z-index: 1;
            margin-top: 40px;
            padding: 22px 24px;
            border-top: 1px solid {t['card_border']};
            text-align: center;
        }}
        .yti-app-footer .yti-footer-name {{ font-weight: 800; font-size: 1rem; color: {t['text']}; }}
        .yti-app-footer .yti-footer-role {{ font-size: 0.82rem; color: {t['muted']}; margin-top: 2px; margin-bottom: 10px; }}
        .yti-social-btn {{
            display: inline-flex; align-items: center; gap: 6px;
            background: {t['primary_soft']};
            border: 1px solid {t['card_border']};
            color: {t['text']} !important;
            border-radius: 999px;
            padding: 6px 16px;
            font-size: 0.82rem; font-weight: 600;
            text-decoration: none !important;
            margin: 0 6px;
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }}
        .yti-social-btn:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px {t['primary_soft']}; }}
        </style>
        {f'<div class="yti-bubble-layer">{bubbles_html}</div>' if bubbles_html else ''}
        """,
        unsafe_allow_html=True,
    )
    return t


def social_buttons_html() -> str:
    return (
        f'<a class="yti-social-btn" href="{GITHUB_URL}" target="_blank">GitHub</a>'
        f'<a class="yti-social-btn" href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>'
    )


def render_app_header():
    """Centered title only — no author/social buttons here (footer-only)."""
    st.markdown(
        """
        <div class="yti-hero-wrap">
            <div class="yti-hero-title">🔴 YouTube <span>Intelligence</span></div>
            <div class="yti-hero-sub">Analytics • Prediction • Classification • Business Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_app_footer():
    """The ONLY place author name/title and GitHub/LinkedIn buttons appear."""
    st.markdown(
        f"""
        <div class="yti-app-footer">
            <div class="yti-footer-name">{AUTHOR_NAME}</div>
            <div class="yti-footer-role">{AUTHOR_TITLE} &nbsp;|&nbsp; YouTube Intelligence v{APP_VERSION}</div>
            <div>{social_buttons_html()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f'<div class="yti-card"><div class="yti-label">{label}</div>'
        f'<div class="yti-value">{value}</div>'
        f'<div class="yti-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def format_number(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    sign = "-" if n < 0 else ""
    n = abs(n)
    for unit, div in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= div:
            return f"{sign}{n / div:.2f}{unit}"
    return f"{sign}{n:.0f}"


def class_badge(label: str) -> str:
    cls = {"HIGH": "yti-badge-high", "MEDIUM": "yti-badge-medium", "LOW": "yti-badge-low"}.get(label, "yti-badge-medium")
    return f'<span class="yti-badge {cls}">{label}</span>'


# =====================================================================================
# DATA LOADING & DATA QUALITY
# Bundled CSV only — no upload control
# =====================================================================================

def _safe_parse_tags(raw):
    """Safely convert a string representation of tags into a Python list."""
    if not isinstance(raw, str) or not raw.strip():
        return []

    try:
        parsed = ast.literal_eval(raw)

        if isinstance(parsed, (list, tuple)):
            return [
                str(tag).strip()
                for tag in parsed
                if str(tag).strip()
            ]

    except (ValueError, SyntaxError, TypeError):
        pass

    return []


def _safe_numeric_conversion(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Convert available columns to numeric without failing on missing columns."""

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


@st.cache_data(
    show_spinner="Loading and preparing YouTube dataset..."
)
def load_dataset(_cache_key: str = "") -> pd.DataFrame:
    """
    Load the bundled YouTube dataset and perform safe preprocessing.

    Important:
    - No CSV upload is used.
    - Duplicate rows are removed.
    - Numeric columns are safely converted.
    - Dates are parsed.
    - Derived presentation fields are created.
    - Original ML target is preserved.
    """

    # -------------------------------------------------------------------------
    # File validation
    # -------------------------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.name}. "
            f"Place '{DATA_PATH.name}' in the same folder as this Streamlit app."
        )

    # -------------------------------------------------------------------------
    # Read CSV
    # -------------------------------------------------------------------------

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read {DATA_PATH.name}: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            "The bundled YouTube dataset is empty."
        )

    raw_rows, raw_cols = df.shape

    # -------------------------------------------------------------------------
    # Duplicate handling
    # -------------------------------------------------------------------------

    duplicate_count = int(
        df.duplicated().sum()
    )

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Date handling
    # -------------------------------------------------------------------------

    if "publish_date" in df.columns:

        df["publish_date"] = pd.to_datetime(
            df["publish_date"],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # Numeric columns
    # -------------------------------------------------------------------------

    numeric_columns = [
        "view_count",
        "like_count",
        "comment_count",
        "favorite_count",

        "duration_seconds",

        "engagement_score",
        "like_rate",
        "comment_rate",
        "engagement_rate",

        "tag_count",

        "title_length",
        "title_word_count",

        "description_length",
        "description_word_count",

        "publish_hour",
        "publish_month",
        "publish_day",
        "publish_week",
        "publish_quarter",

        "channel_video_count",

        "category_id",

        "is_hd",
        "has_caption",
        "is_weekend",
    ]

    df = _safe_numeric_conversion(
        df,
        numeric_columns,
    )

    # -------------------------------------------------------------------------
    # Text columns
    # -------------------------------------------------------------------------

    text_columns = [
        "title",
        "description",
        "channel_title",
        "category_name",
        "duration_category",
        "tags",
    ]

    for column in text_columns:

        if column in df.columns:
            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
            )

    # -------------------------------------------------------------------------
    # Target validation
    # -------------------------------------------------------------------------

    missing_target_rows = 0

    if "view_count" in df.columns:

        before_target_filter = len(df)

        df = df[
            df["view_count"].notna()
        ].copy()

        missing_target_rows = (
            before_target_filter - len(df)
        )

    # -------------------------------------------------------------------------
    # Tag processing
    # -------------------------------------------------------------------------

    if "tags" in df.columns:

        df["tags_list"] = (
            df["tags"]
            .apply(_safe_parse_tags)
        )

    # -------------------------------------------------------------------------
    # Video quality labels
    # -------------------------------------------------------------------------

    if "is_hd" in df.columns:

        df["hd_label"] = np.where(
            df["is_hd"].fillna(0) == 1,
            "HD",
            "SD",
        )

    # -------------------------------------------------------------------------
    # Caption labels
    # -------------------------------------------------------------------------

    if "has_caption" in df.columns:

        df["caption_label"] = np.where(
            df["has_caption"].fillna(0) == 1,
            "Captioned",
            "No Captions",
        )

    # -------------------------------------------------------------------------
    # Weekday / weekend labels
    # -------------------------------------------------------------------------

    if "is_weekend" in df.columns:

        df["weekend_label"] = np.where(
            df["is_weekend"].fillna(0) == 1,
            "Weekend",
            "Weekday",
        )

    # -------------------------------------------------------------------------
    # Date-derived fields
    # -------------------------------------------------------------------------

    if "publish_date" in df.columns:

        valid_dates = df["publish_date"].notna()

        if valid_dates.any():

            df.loc[
                valid_dates,
                "publish_year"
            ] = (
                df.loc[
                    valid_dates,
                    "publish_date"
                ].dt.year
            )

            df.loc[
                valid_dates,
                "publish_month_number"
            ] = (
                df.loc[
                    valid_dates,
                    "publish_date"
                ].dt.month
            )

            df.loc[
                valid_dates,
                "publish_day_number"
            ] = (
                df.loc[
                    valid_dates,
                    "publish_date"
                ].dt.day
            )

            df.loc[
                valid_dates,
                "publish_day_name"
            ] = (
                df.loc[
                    valid_dates,
                    "publish_date"
                ].dt.day_name()
            )

            df.loc[
                valid_dates,
                "publish_month_name"
            ] = (
                df.loc[
                    valid_dates,
                    "publish_date"
                ].dt.month_name()
            )

            df.loc[
                valid_dates,
                "publish_quarter"
            ] = (
                df.loc[
                    valid_dates,
                    "publish_date"
                ].dt.quarter
            )

    # -------------------------------------------------------------------------
    # YouTube URL + thumbnail
    # -------------------------------------------------------------------------

    if "video_id" in df.columns:

        video_ids = (
            df["video_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df["video_url"] = (
            "https://www.youtube.com/watch?v="
            + video_ids
        )

        df["thumbnail_url"] = (
            "https://img.youtube.com/vi/"
            + video_ids
            + "/mqdefault.jpg"
        )

    # -------------------------------------------------------------------------
    # Engagement rate
    #
    # Only calculate if the dataset does not already provide it.
    # -------------------------------------------------------------------------

    required_engagement_columns = {
        "like_count",
        "comment_count",
        "view_count",
    }

    if (
        "engagement_rate" not in df.columns
        and required_engagement_columns.issubset(df.columns)
    ):

        denominator = df["view_count"].replace(
            0,
            np.nan,
        )

        df["engagement_rate"] = (
            (
                df["like_count"].fillna(0)
                + df["comment_count"].fillna(0)
            )
            / denominator
        )

    # -------------------------------------------------------------------------
    # Engagement score fallback
    # -------------------------------------------------------------------------

    if (
        "engagement_score" not in df.columns
        and "engagement_rate" in df.columns
    ):

        df["engagement_score"] = (
            df["engagement_rate"]
        )

    # -------------------------------------------------------------------------
    # Ensure clean index
    # -------------------------------------------------------------------------

    df = (
        df
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Dataset metadata
    # -------------------------------------------------------------------------

    df.attrs["raw_rows"] = raw_rows
    df.attrs["raw_cols"] = raw_cols

    df.attrs["clean_rows"] = len(df)
    df.attrs["clean_cols"] = df.shape[1]

    df.attrs["dup_removed"] = duplicate_count

    df.attrs[
        "dropped_missing_target"
    ] = missing_target_rows

    df.attrs[
        "missing_cells"
    ] = int(
        df.isna().sum().sum()
    )

    return df


# =====================================================================================
# DATA QUALITY REPORT
# =====================================================================================

def quality_report(df: pd.DataFrame) -> dict:
    """
    Generate a compact data-quality summary for the application.
    """

    total_cells = max(
        int(df.size),
        1,
    )

    missing_cells = int(
        df.isna().sum().sum()
    )

    missing_percentage = (
        missing_cells
        / total_cells
        * 100
    )

    return {
        "rows": int(len(df)),

        "columns": int(
            df.shape[1]
        ),

        "original_rows": int(
            df.attrs.get(
                "raw_rows",
                len(df),
            )
        ),

        "original_columns": int(
            df.attrs.get(
                "raw_cols",
                df.shape[1],
            )
        ),

        "duplicates_removed": int(
            df.attrs.get(
                "dup_removed",
                0,
            )
        ),

        "rows_dropped_missing_target": int(
            df.attrs.get(
                "dropped_missing_target",
                0,
            )
        ),

        "missing_cells": missing_cells,

        "missing_pct": round(
            missing_percentage,
            2,
        ),

        "numeric_columns": int(
            df.select_dtypes(
                include=[np.number]
            ).shape[1]
        ),

        "categorical_columns": int(
            df.select_dtypes(
                include=[
                    "object",
                    "category",
                    "bool",
                ]
            ).shape[1]
        ),

        "date_columns": int(
            df.select_dtypes(
                include=[
                    "datetime64[ns]"
                ]
            ).shape[1]
        ),
    }


# =====================================================================================
# DATASET VALIDATION
# =====================================================================================

def validate_dataset(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate the minimum columns required by YouTube Intelligence.

    Returns:
        (is_valid, list_of_issues)
    """

    required_for_core_app = [
        "view_count",
    ]

    recommended_columns = [
        "video_id",
        "title",
        "channel_title",
        "category_name",
        "duration_seconds",
        "publish_date",
    ]

    issues = []

    # -------------------------------------------------------------------------
    # Required fields
    # -------------------------------------------------------------------------

    for column in required_for_core_app:

        if column not in df.columns:

            issues.append(
                f"Required column missing: {column}"
            )

    # -------------------------------------------------------------------------
    # Recommended fields
    # -------------------------------------------------------------------------

    missing_recommended = [
        column
        for column in recommended_columns
        if column not in df.columns
    ]

    if missing_recommended:

        issues.append(
            "Recommended columns missing: "
            + ", ".join(
                missing_recommended
            )
        )

    # -------------------------------------------------------------------------
    # Empty dataset
    # -------------------------------------------------------------------------

    if df.empty:

        issues.append(
            "Dataset contains no usable rows."
        )

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    is_valid = not any(
        issue.startswith(
            "Required column missing"
        )
        or issue.startswith(
            "Dataset contains no usable rows"
        )
        for issue in issues
    )

    return is_valid, issues


# =====================================================================================
# FEATURE ENGINEERING — LEAKAGE SAFE
# =====================================================================================

TARGET_COLUMN = "view_count"

# -------------------------------------------------------------------------------------
# NUMERIC FEATURES
# -------------------------------------------------------------------------------------

CANDIDATE_NUMERIC = [
    "duration_seconds",
    "publish_hour",
    "publish_month",
    "publish_day",
    "publish_week",
    "publish_quarter",
    "title_length",
    "title_word_count",
    "description_length",
    "description_word_count",
    "tag_count",
]


# -------------------------------------------------------------------------------------
# CATEGORICAL FEATURES
# -------------------------------------------------------------------------------------

CANDIDATE_CATEGORICAL = [
    "category_name",
    "duration_category",
    "publish_day_name",
    "publish_session",
    "month_part",
    "caption_label",
]


# -------------------------------------------------------------------------------------
# BOOLEAN FEATURES
# -------------------------------------------------------------------------------------

CANDIDATE_BOOL = [
    "is_weekend",
]


# =====================================================================================
# TARGET-LEAKAGE PROTECTION
# =====================================================================================

# These variables are intentionally excluded from prediction features because
# they are known only after publication or are directly related to the target.

LEAKAGE_EXCLUDED = [

    # Direct engagement outcomes
    "like_count",
    "comment_count",
    "favorite_count",

    # Derived engagement outcomes
    "engagement_score",
    "like_rate",
    "comment_rate",
    "engagement_rate",

    # Target-derived variables
    "view_bucket",
    "popular_category",

    # Potential post-publication / popularity proxy
    "channel_video_count",

    # Actual target
    TARGET_COLUMN,
]


# =====================================================================================
# CLASSIFICATION TARGET CONFIGURATION
# =====================================================================================

CLASS_LABELS = [
    "LOW",
    "MEDIUM",
    "HIGH",
]

LOW_QUANTILE = 0.33
HIGH_QUANTILE = 0.66


# =====================================================================================
# FEATURE AVAILABILITY
# =====================================================================================

def available_features(df: pd.DataFrame) -> dict:
    """
    Detect usable ML features from the current dataset.

    A feature is considered usable when:
        1. It exists in the dataframe.
        2. It contains more than one unique non-null value.
        3. It is not explicitly excluded for leakage.

    Returns:
        {
            "numeric": [...],
            "categorical": [...],
            "boolean": [...]
        }
    """

    def usable(column: str) -> bool:

        if column not in df.columns:
            return False

        if column in LEAKAGE_EXCLUDED:
            return False

        return (
            df[column]
            .nunique(dropna=True)
            > 1
        )

    return {

        "numeric": [
            column
            for column in CANDIDATE_NUMERIC
            if usable(column)
        ],

        "categorical": [
            column
            for column in CANDIDATE_CATEGORICAL
            if usable(column)
        ],

        "boolean": [
            column
            for column in CANDIDATE_BOOL
            if usable(column)
        ],
    }


# =====================================================================================
# DERIVED FEATURE CREATION
# =====================================================================================

def add_engineered_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create prediction-time features that can also be generated
    for a future/planned video.

    This function intentionally does NOT use:
        views
        likes
        comments
        engagement metrics
    """

    work = df.copy()

    # -------------------------------------------------------------------------
    # Date-derived features
    # -------------------------------------------------------------------------

    if "publish_date" in work.columns:

        dates = pd.to_datetime(
            work["publish_date"],
            errors="coerce",
        )

        if "publish_month" not in work.columns:
            work["publish_month"] = (
                dates.dt.month
            )

        if "publish_day" not in work.columns:
            work["publish_day"] = (
                dates.dt.day
            )

        if "publish_quarter" not in work.columns:
            work["publish_quarter"] = (
                dates.dt.quarter
            )

        if "publish_day_name" not in work.columns:
            work["publish_day_name"] = (
                dates.dt.day_name()
            )

        if "is_weekend" not in work.columns:
            work["is_weekend"] = (
                dates.dt.dayofweek >= 5
            )

    # -------------------------------------------------------------------------
    # Duration category
    # -------------------------------------------------------------------------

    if (
        "duration_seconds" in work.columns
        and "duration_category" not in work.columns
    ):

        work["duration_category"] = (
            work["duration_seconds"]
            .apply(derive_duration_category)
        )

    # -------------------------------------------------------------------------
    # Publish session
    # -------------------------------------------------------------------------

    if (
        "publish_hour" in work.columns
        and "publish_session" not in work.columns
    ):

        work["publish_session"] = (
            pd.to_numeric(
                work["publish_hour"],
                errors="coerce",
            )
            .fillna(12)
            .astype(int)
            .apply(derive_publish_session)
        )

    # -------------------------------------------------------------------------
    # Month part
    # -------------------------------------------------------------------------

    if (
        "publish_day" in work.columns
        and "month_part" not in work.columns
    ):

        work["month_part"] = (
            pd.to_numeric(
                work["publish_day"],
                errors="coerce",
            )
            .fillna(15)
            .astype(int)
            .apply(derive_month_part)
        )

    # -------------------------------------------------------------------------
    # Title features
    # -------------------------------------------------------------------------

    if "title" in work.columns:

        title = (
            work["title"]
            .fillna("")
            .astype(str)
        )

        if "title_length" not in work.columns:
            work["title_length"] = (
                title.str.len()
            )

        if "title_word_count" not in work.columns:
            work["title_word_count"] = (
                title.str.split()
                .str.len()
            )

    # -------------------------------------------------------------------------
    # Description features
    # -------------------------------------------------------------------------

    if "description" in work.columns:

        description = (
            work["description"]
            .fillna("")
            .astype(str)
        )

        if "description_length" not in work.columns:
            work["description_length"] = (
                description.str.len()
            )

        if "description_word_count" not in work.columns:
            work["description_word_count"] = (
                description.str.split()
                .str.len()
            )

    # -------------------------------------------------------------------------
    # Tag count
    # -------------------------------------------------------------------------

    if (
        "tags_list" in work.columns
        and "tag_count" not in work.columns
    ):

        work["tag_count"] = (
            work["tags_list"]
            .apply(
                lambda x:
                len(x)
                if isinstance(x, list)
                else 0
            )
        )

    return work


# =====================================================================================
# FEATURE FRAME BUILDER
# =====================================================================================

def build_feature_frame(
    df: pd.DataFrame,
):
    """
    Build the final ML feature matrix.

    Returns:
        X     -> cleaned feature dataframe
        feats -> feature metadata
    """

    work = add_engineered_features(
        df
    )

    feats = available_features(
        work
    )

    feature_columns = (
        feats["numeric"]
        + feats["categorical"]
        + feats["boolean"]
    )

    if not feature_columns:
        raise ValueError(
            "No usable prediction features were found."
        )

    X = work[
        feature_columns
    ].copy()

    # -------------------------------------------------------------------------
    # Boolean features
    # -------------------------------------------------------------------------

    for column in feats["boolean"]:

        X[column] = (
            X[column]
            .fillna(False)
            .astype(bool)
            .astype(int)
        )

    # -------------------------------------------------------------------------
    # Numeric features
    # -------------------------------------------------------------------------

    for column in (
        feats["numeric"]
        + feats["boolean"]
    ):

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

        median_value = X[column].median()

        if pd.isna(median_value):
            median_value = 0

        X[column] = (
            X[column]
            .fillna(median_value)
        )

    # -------------------------------------------------------------------------
    # Categorical features
    # -------------------------------------------------------------------------

    for column in feats["categorical"]:

        X[column] = (
            X[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        X[column] = X[column].replace(
            "",
            "Unknown",
        )

    return X, feats


# =====================================================================================
# CLASSIFICATION TARGET
# =====================================================================================

def make_classification_target(
    train_views: pd.Series,
    target_views: pd.Series,
    low_q: float = LOW_QUANTILE,
    high_q: float = HIGH_QUANTILE,
):
    """
    Create LOW / MEDIUM / HIGH performance classes.

    IMPORTANT:
    Thresholds are calculated ONLY from the training target.

    This prevents test-set information from influencing
    classification boundaries.
    """

    train_views = pd.to_numeric(
        train_views,
        errors="coerce",
    ).dropna()

    target_views = pd.to_numeric(
        target_views,
        errors="coerce",
    )

    if train_views.empty:
        raise ValueError(
            "Training view data is empty."
        )

    low_threshold = float(
        train_views.quantile(
            low_q
        )
    )

    high_threshold = float(
        train_views.quantile(
            high_q
        )
    )

    def label(value):

        if pd.isna(value):
            return np.nan

        if value < low_threshold:
            return "LOW"

        if value < high_threshold:
            return "MEDIUM"

        return "HIGH"

    labels = target_views.apply(
        label
    )

    thresholds = {

        "low_pct": low_q,

        "high_pct": high_q,

        "low_value": low_threshold,

        "high_value": high_threshold,
    }

    return labels, thresholds


# =====================================================================================
# PREDICTION-TIME HELPERS
# =====================================================================================

def derive_duration_category(
    seconds: float,
) -> str:
    """
    Convert duration into a simple business-friendly category.
    """

    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        seconds = 0

    if seconds < 60:
        return "Short"

    if seconds < 600:
        return "Medium"

    return "Long"


def derive_publish_session(
    hour: int,
) -> str:
    """
    Convert publishing hour into a business-friendly session.
    """

    try:
        hour = int(hour)
    except (TypeError, ValueError):
        hour = 12

    hour = hour % 24

    if 0 <= hour <= 6:
        return "Night"

    if 7 <= hour <= 12:
        return "Morning"

    if 13 <= hour <= 18:
        return "Afternoon"

    return "Evening"


def derive_month_part(
    day_of_month: int,
) -> str:
    """
    Divide month into beginning / middle / end.

    This gives the model a simple calendar-position feature.
    """

    try:
        day_of_month = int(
            day_of_month
        )
    except (TypeError, ValueError):
        day_of_month = 15

    if day_of_month <= 10:
        return "Beginning"

    if day_of_month <= 20:
        return "Middle"

    return "End"


# =====================================================================================
# FEATURE SUMMARY
# =====================================================================================

def feature_summary(
    df: pd.DataFrame,
) -> dict:
    """
    Return a human-readable summary of the features
    currently available to the ML pipeline.
    """

    feats = available_features(
        add_engineered_features(df)
    )

    return {

        "numeric_features": feats[
            "numeric"
        ],

        "categorical_features": feats[
            "categorical"
        ],

        "boolean_features": feats[
            "boolean"
        ],

        "total_features": (
            len(feats["numeric"])
            + len(feats["categorical"])
            + len(feats["boolean"])
        ),

        "target": TARGET_COLUMN,

        "leakage_protected": True,

        "excluded_features": LEAKAGE_EXCLUDED,
    }


# =====================================================================================
# ML TRAINING
# Leakage-Safe • Multi-Model • Regression + Classification
# =====================================================================================

# =====================================================================================
# PREPROCESSOR
# =====================================================================================

def _preprocessor(numeric_cols, categorical_cols):
    """
    Build a fresh, leakage-safe preprocessing pipeline.

    Numeric:
        median imputation + standardisation

    Categorical:
        most-frequent imputation + one-hot encoding

    Dense one-hot output is intentional: GradientBoostingClassifier/
    GradientBoostingRegressor do not reliably accept sparse matrices.
    The compatibility fallback supports older scikit-learn versions.
    """
    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        except TypeError:
            # scikit-learn < 1.2
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False,
            )

        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("No usable numeric or categorical features were found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

# =====================================================================================
# REGRESSION MODEL CANDIDATES
# =====================================================================================

def _regression_candidates():
    """Return robust regression candidates for the current dataset."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # XGBoost is optional; the app must still work without it.
    try:
        from xgboost import XGBRegressor

        models["XGBoost Regressor"] = XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
    except Exception:
        pass

    return models

# =====================================================================================
# CLASSIFICATION MODEL CANDIDATES
# =====================================================================================

def _classification_candidates():
    """
    Return classification algorithms used for
    LOW / MEDIUM / HIGH performance prediction.
    """

    return {

        "Logistic Regression":
            LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),

        "Decision Tree Classifier":
            DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
            ),

        "Random Forest Classifier":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

        "Gradient Boosting Classifier":
            GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
    }


# =====================================================================================
# REGRESSION TRAINING
# =====================================================================================

@st.cache_resource(
    show_spinner="Training YouTube regression models..."
)
@st.cache_resource(show_spinner="Training YouTube regression models...")
def train_regression(df: pd.DataFrame, _cache_key: str):
    """
    Train multiple regression models to estimate view_count.

    The target is log1p(view_count), then predictions are transformed
    back with expm1 before evaluation/display. Every model is isolated
    so one failed algorithm cannot disable the entire regression pipeline.
    """
    try:
        X, feats = build_feature_frame(df)

        if TARGET_COLUMN not in df.columns:
            return {"error": f"Required target column '{TARGET_COLUMN}' is missing."}

        y = pd.to_numeric(df.loc[X.index, TARGET_COLUMN], errors="coerce")
        valid_mask = y.notna() & np.isfinite(y)

        X = X.loc[valid_mask].copy()
        y = y.loc[valid_mask].astype(float).clip(lower=0)

        if len(X) < 20:
            return {
                "error": (
                    f"Not enough usable rows for regression: {len(X)}. "
                    "At least 20 rows are required."
                ),
                "features": feats,
            }

        if y.nunique() < 2:
            return {
                "error": "Regression target view_count must contain at least two distinct values.",
                "features": feats,
            }

        X_train, X_test, y_train_raw, y_test_raw = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )

        y_train_log = np.log1p(y_train_raw)

        rows = []
        fitted_models = {}
        model_errors = {}

        for name, model in _regression_candidates().items():
            try:
                pre = _preprocessor(
                    feats["numeric"] + feats["boolean"],
                    feats["categorical"],
                )

                pipe = Pipeline(
                    steps=[
                        ("preprocess", pre),
                        ("model", model),
                    ]
                )

                pipe.fit(X_train, y_train_log)

                pred_log = np.asarray(pipe.predict(X_test), dtype=float)
                predictions = np.maximum(np.expm1(pred_log), 0.0)

                mae = float(mean_absolute_error(y_test_raw, predictions))
                rmse = float(np.sqrt(mean_squared_error(y_test_raw, predictions)))
                r2 = float(r2_score(y_test_raw, predictions))

                non_zero = y_test_raw.to_numpy() > 0
                if non_zero.any():
                    actual = y_test_raw.to_numpy()[non_zero]
                    pred_nz = predictions[non_zero]
                    mape = float(np.mean(np.abs((actual - pred_nz) / actual)) * 100)
                else:
                    mape = np.nan

                fitted_models[name] = pipe
                rows.append(
                    {
                        "Model": name,
                        "MAE": mae,
                        "RMSE": rmse,
                        "R2": r2,
                        "MAPE": mape,
                        "Status": "READY",
                    }
                )

            except Exception as exc:
                model_errors[name] = f"{type(exc).__name__}: {exc}"
                rows.append(
                    {
                        "Model": name,
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "R2": np.nan,
                        "MAPE": np.nan,
                        "Status": "FAILED",
                        "Error": model_errors[name],
                    }
                )

        board = pd.DataFrame(rows)
        valid_board = board[board["RMSE"].notna()].copy()

        if valid_board.empty:
            return {
                "error": "All regression models failed during training.",
                "detailed_error": "\n".join(
                    f"{name}: {err}" for name, err in model_errors.items()
                ),
                "model_errors": model_errors,
                "features": feats,
                "leaderboard": board,
            }

        valid_board = (
            valid_board
            .sort_values(["RMSE", "MAE", "R2"], ascending=[True, True, False])
            .reset_index(drop=True)
        )
        valid_board.insert(0, "Rank", range(1, len(valid_board) + 1))

        best_name = str(valid_board.iloc[0]["Model"])
        best_model = fitted_models[best_name]
        best_predictions = np.maximum(
            np.expm1(best_model.predict(X_test)),
            0.0,
        )

        return {
            "leaderboard": valid_board,
            "all_model_results": board,
            "fitted_models": fitted_models,
            "best_model_name": best_name,
            "best_model": best_model,
            "features": feats,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train_raw,
            "y_test": y_test_raw,
            "y_train_log": y_train_log,
            "test_predictions": best_predictions,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "target_transform": "log1p(view_count)",
            "prediction_inverse_transform": "expm1(prediction)",
            "selection_rule": (
                "Lowest RMSE on the held-out test set; ties are resolved "
                "using lower MAE and higher R²."
            ),
            "feature_columns": (
                feats["numeric"] + feats["boolean"] + feats["categorical"]
            ),
            "model_errors": model_errors,
        }

    except Exception as exc:
        return {
            "error": f"Regression training setup failed: {type(exc).__name__}: {exc}",
            "detailed_error": repr(exc),
        }

# =====================================================================================
# CLASSIFICATION TRAINING
# =====================================================================================

@st.cache_resource(
    show_spinner="Training YouTube classification models..."
)
@st.cache_resource(show_spinner="Training YouTube classification models...")
def train_classification(df: pd.DataFrame, _cache_key: str):
    """
    Train LOW / MEDIUM / HIGH classifiers.

    Thresholds are learned from the training split only and then reused
    unchanged for the test split and future predictions.
    """
    try:
        X, feats = build_feature_frame(df)

        if TARGET_COLUMN not in df.columns:
            return {"error": f"Required target column '{TARGET_COLUMN}' is missing."}

        y_views = pd.to_numeric(
            df.loc[X.index, TARGET_COLUMN],
            errors="coerce",
        )
        valid_mask = y_views.notna() & np.isfinite(y_views)

        X = X.loc[valid_mask].copy()
        y_views = y_views.loc[valid_mask].astype(float).clip(lower=0)

        if len(X) < 20:
            return {
                "error": (
                    f"Not enough usable rows for classification: {len(X)}. "
                    "At least 20 rows are required."
                ),
                "features": feats,
            }

        X_train, X_test, views_train, views_test = train_test_split(
            X,
            y_views,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )

        _, thresholds = make_classification_target(
            views_train,
            views_train,
        )

        low_threshold = thresholds["low_value"]
        high_threshold = thresholds["high_value"]

        def apply_thresholds(series):
            return series.apply(
                lambda value: (
                    "LOW"
                    if value < low_threshold
                    else "MEDIUM"
                    if value < high_threshold
                    else "HIGH"
                )
            )

        y_train = apply_thresholds(views_train)
        y_test = apply_thresholds(views_test)

        class_counts = y_train.value_counts()
        imbalance_warning = None
        if not class_counts.empty and class_counts.max() > 0:
            ratio = float(class_counts.min() / class_counts.max())
            if ratio < 0.40:
                imbalance_warning = (
                    "Training classes are imbalanced "
                    f"(smallest={class_counts.min()}, largest={class_counts.max()})."
                )

        rows = []
        fitted_models = {}
        model_errors = {}

        for name, model in _classification_candidates().items():
            try:
                pre = _preprocessor(
                    feats["numeric"] + feats["boolean"],
                    feats["categorical"],
                )

                pipe = Pipeline(
                    steps=[
                        ("preprocess", pre),
                        ("model", model),
                    ]
                )

                pipe.fit(X_train, y_train)
                predictions = pipe.predict(X_test)

                accuracy = float(accuracy_score(y_test, predictions))
                precision = float(
                    precision_score(
                        y_test, predictions,
                        average="weighted",
                        zero_division=0,
                    )
                )
                recall = float(
                    recall_score(
                        y_test, predictions,
                        average="weighted",
                        zero_division=0,
                    )
                )
                f1 = float(
                    f1_score(
                        y_test, predictions,
                        average="weighted",
                        zero_division=0,
                    )
                )

                fitted_models[name] = pipe
                rows.append(
                    {
                        "Model": name,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1": f1,
                        "Status": "READY",
                    }
                )

            except Exception as exc:
                model_errors[name] = f"{type(exc).__name__}: {exc}"
                rows.append(
                    {
                        "Model": name,
                        "Accuracy": np.nan,
                        "Precision": np.nan,
                        "Recall": np.nan,
                        "F1": np.nan,
                        "Status": "FAILED",
                        "Error": model_errors[name],
                    }
                )

        board = pd.DataFrame(rows)
        valid_board = board[board["F1"].notna()].copy()

        if valid_board.empty:
            return {
                "error": "All classification models failed during training.",
                "detailed_error": "\n".join(
                    f"{name}: {err}" for name, err in model_errors.items()
                ),
                "model_errors": model_errors,
                "features": feats,
                "thresholds": thresholds,
                "leaderboard": board,
            }

        valid_board = (
            valid_board
            .sort_values(
                ["F1", "Accuracy", "Recall"],
                ascending=[False, False, False],
            )
            .reset_index(drop=True)
        )
        valid_board.insert(0, "Rank", range(1, len(valid_board) + 1))

        best_name = str(valid_board.iloc[0]["Model"])
        best_model = fitted_models[best_name]
        best_predictions = best_model.predict(X_test)

        class_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        available_labels = sorted(
            y_train.dropna().unique(),
            key=lambda x: class_order.get(x, 99),
        )

        return {
            "leaderboard": valid_board,
            "all_model_results": board,
            "fitted_models": fitted_models,
            "best_model_name": best_name,
            "best_model": best_model,
            "features": feats,
            "thresholds": thresholds,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "views_train": views_train,
            "views_test": views_test,
            "test_predictions": best_predictions,
            "class_labels": available_labels,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "imbalance_warning": imbalance_warning,
            "selection_rule": (
                "Highest weighted F1 score on the held-out test set; "
                "ties are resolved using higher accuracy and recall."
            ),
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "feature_columns": (
                feats["numeric"] + feats["boolean"] + feats["categorical"]
            ),
            "model_errors": model_errors,
        }

    except Exception as exc:
        return {
            "error": f"Classification training setup failed: {type(exc).__name__}: {exc}",
            "detailed_error": repr(exc),
        }

# =====================================================================================
# FEATURE NAMES
# =====================================================================================

def get_feature_names_out(
    pipe,
) -> list:
    """
    Retrieve transformed feature names from a trained pipeline.
    """

    try:

        return list(
            pipe
            .named_steps[
                "preprocess"
            ]
            .get_feature_names_out()
        )

    except Exception:

        return []


# =====================================================================================
# FEATURE IMPORTANCE
# =====================================================================================

def get_feature_importance(
    pipe,
):
    """
    Extract feature importance from tree-based models
    or absolute coefficient magnitude from linear models.
    """

    if pipe is None:
        return None

    try:

        model = pipe.named_steps.get(
            "model"
        )

    except Exception:

        return None

    names = get_feature_names_out(
        pipe
    )

    if not names:
        return None

    # -------------------------------------------------------------------------
    # Tree models
    # -------------------------------------------------------------------------

    if hasattr(
        model,
        "feature_importances_",
    ):

        importance = (
            model
            .feature_importances_
        )

        result = pd.DataFrame(
            {
                "Feature":
                    names,

                "Importance":
                    importance,
            }
        )

        return (
            result
            .sort_values(
                "Importance",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    # -------------------------------------------------------------------------
    # Linear / Logistic models
    # -------------------------------------------------------------------------

    if hasattr(
        model,
        "coef_",
    ):

        coefficients = (
            model.coef_
        )

        if coefficients.ndim > 1:

            coefficients = (
                np.abs(
                    coefficients
                )
                .mean(
                    axis=0
                )
            )

        else:

            coefficients = np.abs(
                coefficients
            )

        result = pd.DataFrame(
            {
                "Feature":
                    names,

                "Importance":
                    coefficients,
            }
        )

        return (
            result
            .sort_values(
                "Importance",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    return None


# =====================================================================================
# MODEL PREDICTION HELPERS
# =====================================================================================

def predict_views(
    model,
    input_df: pd.DataFrame,
) -> float:
    """
    Predict views using a selected trained regression model.

    The regression model was trained on log1p(view_count),
    therefore prediction is converted back using expm1().
    """

    if model is None:
        raise ValueError(
            "Regression model is not available."
        )

    prediction_log = model.predict(
        input_df
    )

    prediction = np.expm1(
        prediction_log
    )

    prediction = max(
        float(prediction[0]),
        0.0,
    )

    return prediction


def predict_performance_class(
    model,
    input_df: pd.DataFrame,
) -> dict:
    """
    Predict LOW / MEDIUM / HIGH performance.

    Returns both class and probabilities when
    probability estimates are available.
    """

    if model is None:
        raise ValueError(
            "Classification model is not available."
        )

    prediction = model.predict(
        input_df
    )[0]

    result = {
        "class": str(
            prediction
        ),
        "probabilities": None,
    }

    if hasattr(
        model,
        "predict_proba",
    ):

        try:

            probabilities = (
                model.predict_proba(
                    input_df
                )[0]
            )

            classes = (
                model
                .named_steps[
                    "model"
                ]
                .classes_
            )

            result[
                "probabilities"
            ] = {
                str(label):
                    float(prob)
                for label, prob
                in zip(
                    classes,
                    probabilities,
                )
            }

        except Exception:
            pass

    return result


# =====================================================================================
# BUSINESS RECOMMENDATION ENGINE
# =====================================================================================

def recommendation_for_class(
    perf_class: str,
) -> str:
    """
    Generate a business recommendation based on
    predicted YouTube performance class.
    """

    perf_class = str(
        perf_class
    ).strip().upper()

    recommendations = {

        "HIGH":
            (
                "This content profile historically resembles "
                "high-performing videos. Consider producing related "
                "content and reinforcing the same category, timing, "
                "content format and packaging strategy."
            ),

        "MEDIUM":
            (
                "This content profile is positioned in the mid-tier "
                "performance range. Strengthen title clarity, "
                "thumbnail appeal, audience targeting and publishing "
                "timing to improve the probability of reaching the "
                "HIGH performance bracket."
            ),

        "LOW":
            (
                "This content profile resembles lower-performing "
                "videos historically. Review the topic, category, "
                "title, thumbnail strategy, audience targeting and "
                "publishing approach before allocating significant "
                "resources."
            ),
    }

    return recommendations.get(
        perf_class,
        "No business recommendation is available for this performance class.",
    )


# =====================================================================================
# CLASS-BASED ACTION PLAN
# =====================================================================================

def action_plan_for_class(
    perf_class: str,
) -> dict:
    """
    Return a structured business action plan for the
    predicted performance tier.
    """

    perf_class = str(
        perf_class
    ).strip().upper()

    plans = {

        "HIGH": {
            "status": "Strong Opportunity",
            "priority": "Scale",
            "risk": "Lower relative performance risk",
            "actions": [
                "Consider producing related content.",
                "Maintain the successful content category.",
                "Reuse effective publishing-time patterns.",
                "Test similar title and topic structures.",
                "Monitor engagement after publication.",
            ],
        },

        "MEDIUM": {
            "status": "Growth Opportunity",
            "priority": "Optimize",
            "risk": "Moderate performance uncertainty",
            "actions": [
                "Improve title clarity and discoverability.",
                "Experiment with thumbnail packaging.",
                "Review publishing-time patterns.",
                "Strengthen audience targeting.",
                "Test related topics before scaling.",
            ],
        },

        "LOW": {
            "status": "Needs Optimization",
            "priority": "Review",
            "risk": "Higher performance uncertainty",
            "actions": [
                "Reconsider the topic or content angle.",
                "Improve title and thumbnail strategy.",
                "Review category-level historical performance.",
                "Investigate better publishing windows.",
                "Validate the concept before significant investment.",
            ],
        },
    }

    return plans.get(
        perf_class,
        {
            "status": "Unknown",
            "priority": "Review",
            "risk": "Unable to determine",
            "actions": [
                "Review the prediction inputs and model output."
            ],
        },
    )


# =====================================================================================
# ENGAGEMENT VS REACH INSIGHT
# =====================================================================================

def engagement_reach_recommendation(
    view_percentile: float,
    engagement_percentile: float,
):
    """
    Identify whether a video appears to have a
    reach problem or an engagement problem.

    Expected inputs:
        0.0 -> 1.0 percentile values.
    """

    try:

        view_percentile = float(
            view_percentile
        )

        engagement_percentile = float(
            engagement_percentile
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    # Keep percentiles within valid range
    view_percentile = np.clip(
        view_percentile,
        0,
        1,
    )

    engagement_percentile = np.clip(
        engagement_percentile,
        0,
        1,
    )

    # -------------------------------------------------------------------------
    # High reach / low engagement
    # -------------------------------------------------------------------------

    if (
        view_percentile >= 0.66
        and engagement_percentile <= 0.33
    ):

        return (
            "Reach is strong, but audience interaction is "
            "comparatively low. Consider stronger calls-to-action, "
            "community engagement and content formats that encourage "
            "likes, comments and discussion."
        )

    # -------------------------------------------------------------------------
    # Low reach / high engagement
    # -------------------------------------------------------------------------

    if (
        view_percentile <= 0.33
        and engagement_percentile >= 0.66
    ):

        return (
            "Audience engagement is strong relative to the observed "
            "view count. Discovery and reach may be limiting factors. "
            "Consider improving SEO, thumbnail packaging, titles and "
            "distribution strategy."
        )

    # -------------------------------------------------------------------------
    # High reach / high engagement
    # -------------------------------------------------------------------------

    if (
        view_percentile >= 0.66
        and engagement_percentile >= 0.66
    ):

        return (
            "Both reach and audience interaction are strong. "
            "This represents a potentially valuable content pattern "
            "to study and replicate across related videos."
        )

    # -------------------------------------------------------------------------
    # Low reach / low engagement
    # -------------------------------------------------------------------------

    if (
        view_percentile <= 0.33
        and engagement_percentile <= 0.33
    ):

        return (
            "Both reach and engagement are comparatively low. "
            "Review the topic, audience fit, title, thumbnail, "
            "publishing strategy and overall content positioning."
        )

    return None


# =====================================================================================
# PREDICTION CONFIDENCE / INTERPRETATION
# =====================================================================================

def prediction_confidence_label(
    probabilities: dict | None,
) -> str:
    """
    Convert classification probabilities into a
    simple business-friendly confidence label.
    """

    if not probabilities:
        return "Not Available"

    try:

        max_probability = max(
            probabilities.values()
        )

    except (
        TypeError,
        ValueError,
    ):

        return "Not Available"

    if max_probability >= 0.75:
        return "High Confidence"

    if max_probability >= 0.55:
        return "Moderate Confidence"

    return "Low Confidence"


# =====================================================================================
# BUSINESS DECISION SUMMARY
# =====================================================================================

def build_business_summary(
    perf_class: str,
    predicted_views: float | None = None,
    probabilities: dict | None = None,
) -> dict:
    """
    Build a complete business-oriented prediction summary.

    This keeps ML output separate from business interpretation.
    """

    perf_class = str(
        perf_class
    ).strip().upper()

    plan = action_plan_for_class(
        perf_class
    )

    confidence = prediction_confidence_label(
        probabilities
    )

    summary = {
        "performance_class": perf_class,
        "status": plan["status"],
        "priority": plan["priority"],
        "risk": plan["risk"],
        "confidence": confidence,
        "recommendation":
            recommendation_for_class(
                perf_class
            ),
        "actions":
            plan["actions"],
    }

    if predicted_views is not None:

        try:

            summary["predicted_views"] = max(
                float(predicted_views),
                0,
            )

        except (
            TypeError,
            ValueError,
        ):

            summary["predicted_views"] = None

    if probabilities:

        summary[
            "probabilities"
        ] = probabilities

    return summary


# =====================================================================================
# DATASET-LEVEL BUSINESS INSIGHT
# =====================================================================================

def dataset_business_insights(
    df: pd.DataFrame,
) -> list:
    """
    Generate descriptive business insights from the
    historical dataset.

    These are observational insights, not causal claims.
    """

    insights = []

    # -------------------------------------------------------------------------
    # Category insight
    # -------------------------------------------------------------------------

    if {
        "category_name",
        "view_count",
    }.issubset(df.columns):

        category_views = (
            df.groupby(
                "category_name"
            )["view_count"]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        if not category_views.empty:

            top_category = (
                category_views.index[0]
            )

            top_value = (
                category_views.iloc[0]
            )

            insights.append(
                f"{top_category} has the highest "
                f"average views among the categories "
                f"in the current dataset "
                f"({format_number(top_value)} average views)."
            )

    # -------------------------------------------------------------------------
    # Publishing-hour insight
    # -------------------------------------------------------------------------

    if {
        "publish_hour",
        "view_count",
    }.issubset(df.columns):

        hourly_views = (
            df.groupby(
                "publish_hour"
            )["view_count"]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        if not hourly_views.empty:

            best_hour = int(
                hourly_views.index[0]
            )

            insights.append(
                f"Hour {best_hour:02d}:00 shows the "
                "highest average views in the current "
                "historical dataset."
            )

    # -------------------------------------------------------------------------
    # Duration insight
    # -------------------------------------------------------------------------

    if {
        "duration_category",
        "view_count",
    }.issubset(df.columns):

        duration_views = (
            df.groupby(
                "duration_category"
            )["view_count"]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        if not duration_views.empty:

            best_duration = (
                duration_views.index[0]
            )

            insights.append(
                f"{best_duration} videos have the highest "
                "average views among the available duration groups."
            )

    return insights


# =====================================================================================
# TOP LAYOUT — title (centered) → control bar (theme + page nav, horizontal) →
# status strip (dataset / model / quick stats, horizontal) → page content → footer.
# No sidebar anywhere in this app.
# =====================================================================================
PAGES = [
    "🏠 Home",
    "📊 Executive Dashboard",
    "🎬 Content Analytics",
    "💬 Engagement Analytics",
    "📈 Trend Analysis",
    "🤖 ML Overview",
    "🔮 View Prediction",
    "🏷️ Performance Classification",
    "⚖️ Model Comparison",
    "📉 Model Evaluation",
    "🌟 Feature Importance",
    "🕘 Prediction History",
    "💡 Business Recommendations",
    "🔍 Dataset Explorer",
    "📚 Model Information",
    "ℹ️ About",
]

if "theme_name" not in st.session_state:
    st.session_state.theme_name = "YouTube Red (Bubbles)"
if "current_page" not in st.session_state:
    st.session_state.current_page = PAGES[0]
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

render_app_header()

ctrl_left, ctrl_right = st.columns([3.2, 1])
with ctrl_left:
    st.markdown('<div class="yti-controlbar">', unsafe_allow_html=True)
    page = st.selectbox(
        "Navigate", PAGES,
        index=PAGES.index(st.session_state.current_page),
        label_visibility="collapsed",
        key="current_page",
    )
    st.markdown('</div>', unsafe_allow_html=True)
with ctrl_right:
    st.markdown('<div class="yti-controlbar">', unsafe_allow_html=True)
    theme_name = st.selectbox(
        "Theme", list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme_name),
        label_visibility="collapsed",
        key="theme_name",
    )
    st.markdown('</div>', unsafe_allow_html=True)

THEME = inject_theme(theme_name)

# ---- Load data (bundled CSV only) + train models (cached) --------------------------
if DATA_PATH.exists():
    try:
        _dataset_cache_key = (
            f"{DATA_PATH.stat().st_mtime_ns}-"
            f"{DATA_PATH.stat().st_size}"
        )
    except OSError:
        _dataset_cache_key = "unknown"

    df_raw = load_dataset(_dataset_cache_key)
    dataset_status = "trending_videos.csv (bundled)"
else:
    df_raw = None
    dataset_status = "No dataset found"

reg_result, clf_result = None, None
if df_raw is not None and not df_raw.empty:
    try:
        _ml_cache_key = (
            f"{DATA_PATH.stat().st_mtime_ns}-"
            f"{DATA_PATH.stat().st_size}-"
            f"{len(df_raw)}-{df_raw.shape[1]}-"
            f"{'|'.join(map(str, df_raw.columns))}"
        )
    except OSError:
        _ml_cache_key = f"{len(df_raw)}-{df_raw.shape[1]}"

    reg_result = train_regression(df_raw, _ml_cache_key)
    clf_result = train_classification(df_raw, _ml_cache_key)

reg_ok = bool(reg_result and "error" not in reg_result)
clf_ok = bool(clf_result and "error" not in clf_result)

# ---- Horizontal status strip (replaces the old sidebar content) --------------------
if df_raw is not None and not df_raw.empty:
    status_pills = [
        f'<div class="yti-status-pill">✅ <b>{len(df_raw):,} rows</b> — {dataset_status}</div>',
        f'<div class="yti-status-pill">🔮 Regression: <b>{"Ready" if reg_ok else "Unavailable"}</b></div>',
        f'<div class="yti-status-pill">🏷️ Classification: <b>{"Ready" if clf_ok else "Unavailable"}</b></div>',
        f'<div class="yti-status-pill">🎬 Videos: <b>{len(df_raw):,}</b></div>',
        f'<div class="yti-status-pill">🗂️ Categories: <b>{df_raw["category_name"].nunique() if "category_name" in df_raw else "—"}</b></div>',
        f'<div class="yti-status-pill">👁️ Total Views: <b>{format_number(df_raw["view_count"].sum())}</b></div>',
    ]
    st.markdown(f'<div class="yti-status-strip">{"".join(status_pills)}</div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="yti-status-strip"><div class="yti-status-pill">⚠️ <b>No usable dataset loaded</b></div></div>',
        unsafe_allow_html=True,
    )

if df_raw is None or df_raw.empty:
    st.error(
        "No dataset is available. Make sure `trending_videos.csv` is placed "
        "next to `app.py` in the same folder, then restart the app."
    )
    render_app_footer()
    st.stop()


# =====================================================================================
# PAGE: HOME
# =====================================================================================
def page_home():
    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Videos Analyzed", f"{len(df_raw):,}")
    with c2: metric_card("Categories", f"{df_raw['category_name'].nunique()}" if "category_name" in df_raw else "—")
    with c3: metric_card("Channels", f"{df_raw['channel_title'].nunique()}" if "channel_title" in df_raw else "—")
    with c4: metric_card("Total Views", format_number(df_raw["view_count"].sum()))

    st.write("")
    st.subheader("What this application does")
    st.markdown(
        """
        This system turns historical YouTube performance data into two things a
        content team actually needs:

        1. **Descriptive analytics** — how has content performed historically,
           broken down by category, channel, timing and engagement.
        2. **Predictive intelligence** — given a *new or planned* video's
           attributes (category, duration, upload timing, title/description
           characteristics), a trained ML model estimates its expected view
           count and classifies its likely performance tier (LOW / MEDIUM / HIGH).
           You choose *which* trained model generates the prediction.
        """
    )

    st.write("")
    st.subheader("Explore")
    cards = [
        ("📊 Executive Dashboard", "High-level KPIs across the current dataset."),
        ("🎬 Content Analytics", "Top/bottom videos, category performance, filters."),
        ("🔮 View Prediction", "Predict expected views — pick your model."),
        ("🏷️ Performance Classification", "Classify expected performance as LOW / MEDIUM / HIGH."),
        ("⚖️ Model Comparison", "Compare all trained regression & classification models."),
        ("📚 Model Information", "Plain-language explanation of every metric used."),
    ]
    cols = st.columns(3)
    for i, (title, desc) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(
                f"<div class='yti-nav-card'><b>{title}</b><br>"
                f"<span style='font-size:0.85rem;color:{THEME['muted']}'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

    st.write("")
    st.subheader("Technology stack")
    st.markdown(
        "`Python` `Pandas` `NumPy` `scikit-learn` `Plotly` `Streamlit`  \n"
        "Regression: Linear Regression · Random Forest · Gradient Boosting  \n"
        "Classification: Logistic Regression · Decision Tree · Random Forest · Gradient Boosting"
    )


# =====================================================================================
# PAGE: EXECUTIVE DASHBOARD
# =====================================================================================

def page_dashboard():
    """
    Executive-level YouTube performance dashboard.

    Design goals:
    - Streamlit-native UI
    - Premium / minimal layout
    - Transparent Plotly charts
    - Theme-aware visuals
    - Robust against missing optional columns
    - Business-focused insights
    """

    # =========================================================================
    # DATA
    # =========================================================================

    df = df_raw.copy()

    # -------------------------------------------------------------------------
    # Helper functions
    # -------------------------------------------------------------------------

    def has(*columns):
        return all(col in df.columns for col in columns)

    def safe_mean(column):
        if column not in df.columns:
            return None
        value = pd.to_numeric(df[column], errors="coerce").mean()
        return value if pd.notna(value) else None

    def safe_sum(column):
        if column not in df.columns:
            return None
        value = pd.to_numeric(df[column], errors="coerce").sum()
        return value if pd.notna(value) else None

    def safe_format(value):
        return format_number(value) if value is not None else "—"

    # =========================================================================
    # PAGE HEADER
    # =========================================================================

    st.title("📊 Executive Dashboard")

    st.caption(
        "A live executive view of YouTube content performance, audience reach, "
        "engagement and publishing patterns."
    )

    st.write("")

    # =========================================================================
    # PRIMARY KPI ROW
    # =========================================================================

    st.subheader("Performance Overview")

    k1, k2, k3, k4 = st.columns(4)

    total_views = safe_sum("view_count")
    total_likes = safe_sum("like_count")
    total_comments = safe_sum("comment_count")

    with k1:
        st.metric(
            "🎬 Total Videos",
            f"{len(df):,}",
        )

    with k2:
        st.metric(
            "👁️ Total Views",
            safe_format(total_views),
        )

    with k3:
        st.metric(
            "👍 Total Likes",
            safe_format(total_likes),
        )

    with k4:
        st.metric(
            "💬 Total Comments",
            safe_format(total_comments),
        )

    st.write("")

    # =========================================================================
    # SECONDARY KPI ROW
    # =========================================================================

    k5, k6, k7, k8 = st.columns(4)

    avg_views = safe_mean("view_count")
    avg_engagement = safe_mean("engagement_rate")

    channel_count = (
        df["channel_title"].nunique()
        if "channel_title" in df.columns
        else None
    )

    if "category_name" in df.columns:
        top_category = (
            df["category_name"]
            .dropna()
            .value_counts()
            .idxmax()
            if not df["category_name"].dropna().empty
            else "—"
        )
    else:
        top_category = "—"

    with k5:
        st.metric(
            "📈 Avg Views / Video",
            safe_format(avg_views),
        )

    with k6:
        st.metric(
            "❤️ Avg Engagement Rate",
            f"{avg_engagement * 100:.2f}%"
            if avg_engagement is not None
            else "—",
        )

    with k7:
        st.metric(
            "📺 Total Channels",
            f"{channel_count:,}"
            if channel_count is not None
            else "—",
        )

    with k8:
        st.metric(
            "🏆 Top Category",
            str(top_category),
        )

    st.divider()

    # =========================================================================
    # PLOTLY THEME HELPER
    # =========================================================================

    def polish_chart(fig, height=380):
        """
        Apply transparent / premium styling to Plotly figures.
        """

        fig.update_layout(
            template=THEME["plot_template"],

            height=height,

            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",

            font=dict(
                color=THEME["text"],
            ),

            title=dict(
                font=dict(
                    size=17,
                    color=THEME["text"],
                ),
                x=0.02,
                xanchor="left",
            ),

            margin=dict(
                l=20,
                r=20,
                t=55,
                b=25,
            ),

            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=dict(
                    color=THEME["muted"],
                ),
            ),

            hoverlabel=dict(
                bgcolor=THEME["card"],
                bordercolor=THEME["card_border"],
                font=dict(
                    color=THEME["text"],
                ),
            ),

            xaxis=dict(
                gridcolor="rgba(255,255,255,0.055)",
                zerolinecolor="rgba(255,255,255,0.08)",
                tickfont=dict(
                    color=THEME["muted"],
                ),
                title_font=dict(
                    color=THEME["muted"],
                ),
            ),

            yaxis=dict(
                gridcolor="rgba(255,255,255,0.055)",
                zerolinecolor="rgba(255,255,255,0.08)",
                tickfont=dict(
                    color=THEME["muted"],
                ),
                title_font=dict(
                    color=THEME["muted"],
                ),
            ),
        )

        return fig

    # =========================================================================
    # CATEGORY INTELLIGENCE
    # =========================================================================

    if has("category_name", "view_count"):

        st.subheader("📊 Category Intelligence")

        st.caption(
            "Which content categories contribute most to total observed views?"
        )

        cat = (
            df.groupby("category_name", as_index=False)["view_count"]
            .sum()
            .sort_values("view_count", ascending=False)
        )

        c1, c2 = st.columns([1.35, 1])

        # ---------------------------------------------------------------------
        # Total Views by Category
        # ---------------------------------------------------------------------

        with c1:

            fig = px.bar(
                cat,
                x="view_count",
                y="category_name",
                orientation="h",
                title="Total Views by Category",
                labels={
                    "view_count": "Total Views",
                    "category_name": "Category",
                },
                text="view_count",
            )

            fig.update_traces(
                marker_color=THEME["primary"],
                marker_line_width=0,
                texttemplate="%{x:,.0f}",
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Views: %{x:,.0f}"
                    "<extra></extra>"
                ),
            )

            fig.update_layout(
                yaxis=dict(
                    categoryorder="total ascending",
                ),
            )

            fig = polish_chart(fig, 410)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

        # ---------------------------------------------------------------------
        # Video Distribution
        # ---------------------------------------------------------------------

        with c2:

            category_counts = (
                df["category_name"]
                .value_counts()
                .reset_index()
            )

            category_counts.columns = [
                "category_name",
                "count",
            ]

            fig = px.pie(
                category_counts,
                names="category_name",
                values="count",
                hole=0.62,
                title="Video Distribution by Category",
            )

            fig.update_traces(
                textposition="outside",
                textinfo="percent",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Videos: %{value:,}<br>"
                    "Share: %{percent}"
                    "<extra></extra>"
                ),
                marker=dict(
                    line=dict(
                        width=1,
                        color=THEME["bg"],
                    )
                ),
            )

            fig = polish_chart(fig, 410)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

    # =========================================================================
    # VIEWS DISTRIBUTION
    # =========================================================================
    
    if "view_count" in df.columns:
    
        st.subheader("👁️ View Performance Distribution")
    
        st.caption(
            "Distribution of historical video views. "
            "The logarithmic scale makes highly skewed YouTube view data easier to interpret."
        )
    
        view_series = pd.to_numeric(
            df["view_count"],
            errors="coerce"
        ).dropna()
    
        # Remove invalid / zero values because log-scale cannot display zero
        view_series = view_series[view_series > 0]
    
        if len(view_series) >= 2:
    
            # -------------------------------------------------------------
            # Use log10 transformation explicitly.
            # This is more reliable than applying type="log" directly
            # to a histogram.
            # -------------------------------------------------------------
    
            log_views = np.log10(view_series)
    
            hist_df = pd.DataFrame({
                "log_views": log_views
            })
    
            fig = px.histogram(
                hist_df,
                x="log_views",
                nbins=20,
                title="Distribution of Video Views",
                labels={
                    "log_views": "Views (Log₁₀ Scale)",
                    "count": "Number of Videos",
                },
            )
    
            fig.update_traces(
                marker_color=THEME["primary"],
                marker_line_width=0,
                opacity=0.82,
                hovertemplate=(
                    "Log₁₀ Views: %{x:.2f}<br>"
                    "Videos: %{y:,}"
                    "<extra></extra>"
                ),
            )
    
            # -------------------------------------------------------------
            # Transparent premium background
            # -------------------------------------------------------------
    
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
    
                height=410,
    
                margin=dict(
                    l=20,
                    r=20,
                    t=65,
                    b=35,
                ),
    
                font=dict(
                    color=THEME["text"],
                ),
    
                title=dict(
                    font=dict(
                        size=17,
                        color=THEME["text"],
                    ),
                    x=0.02,
                    xanchor="left",
                ),
    
                xaxis=dict(
                    title="Views (Log₁₀ Scale)",
                    gridcolor="rgba(255,255,255,0.06)",
                    zeroline=False,
                    tickfont=dict(
                        color=THEME["muted"],
                    ),
                    title_font=dict(
                        color=THEME["muted"],
                    ),
                ),
    
                yaxis=dict(
                    title="Number of Videos",
                    gridcolor="rgba(255,255,255,0.06)",
                    zeroline=False,
                    tickfont=dict(
                        color=THEME["muted"],
                    ),
                    title_font=dict(
                        color=THEME["muted"],
                    ),
                ),
    
                hoverlabel=dict(
                    bgcolor=THEME["card"],
                    bordercolor=THEME["card_border"],
                    font=dict(
                        color=THEME["text"],
                    ),
                ),
            )
    
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )
    
        else:
    
            st.info(
                "Not enough valid positive view-count values are available "
                "to generate the distribution."
            )

    # =========================================================================
    # PUBLISHING INTELLIGENCE
    # =========================================================================

    if has("publish_hour", "view_count"):

        st.subheader("🕐 Publishing Intelligence")

        st.caption(
            "Observed relationship between publishing hour and historical views."
        )

        hourly = (
            df.groupby("publish_hour", as_index=False)["view_count"]
            .agg(
                average_views="mean",
                median_views="median",
                videos="count",
            )
            .sort_values("publish_hour")
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=hourly["publish_hour"],
                y=hourly["average_views"],
                mode="lines+markers",
                name="Average Views",
                line=dict(
                    color=THEME["primary"],
                    width=3,
                ),
                marker=dict(
                    size=7,
                ),
                hovertemplate=(
                    "<b>%{x}:00</b><br>"
                    "Average Views: %{y:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=hourly["publish_hour"],
                y=hourly["median_views"],
                mode="lines+markers",
                name="Median Views",
                line=dict(
                    color=THEME["accent"],
                    width=2,
                    dash="dot",
                ),
                marker=dict(
                    size=5,
                ),
                hovertemplate=(
                    "<b>%{x}:00</b><br>"
                    "Median Views: %{y:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_xaxes(
            dtick=1,
            title="Publishing Hour",
        )

        fig.update_yaxes(
            type="log",
            title="Views — Log Scale",
        )

        fig = polish_chart(
            fig,
            400,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
            },
        )

    # =========================================================================
    # CONTENT CHARACTERISTICS
    # =========================================================================

    if has("duration_seconds", "view_count"):

        st.subheader("🎬 Content Characteristics")

        st.caption(
            "Explore how video duration relates to observed view performance."
        )

        duration_df = df[
            ["duration_seconds", "view_count"]
        ].copy()

        duration_df["duration_seconds"] = pd.to_numeric(
            duration_df["duration_seconds"],
            errors="coerce",
        )

        duration_df["view_count"] = pd.to_numeric(
            duration_df["view_count"],
            errors="coerce",
        )

        duration_df = duration_df.dropna()

        duration_df = duration_df[
            (duration_df["duration_seconds"] >= 0)
            & (duration_df["view_count"] >= 0)
        ]

        if not duration_df.empty:

            fig = px.scatter(
                duration_df,
                x="duration_seconds",
                y="view_count",
                title="Video Duration vs Views",
                labels={
                    "duration_seconds": "Duration (seconds)",
                    "view_count": "Views",
                },
                opacity=0.70,
            )

            fig.update_traces(
                marker=dict(
                    size=8,
                    color=THEME["primary"],
                    line=dict(
                        width=0,
                    ),
                ),
                hovertemplate=(
                    "Duration: %{x:.0f} sec<br>"
                    "Views: %{y:,.0f}"
                    "<extra></extra>"
                ),
            )

            fig.update_yaxes(
                type="log",
                title="Views — Log Scale",
            )

            fig = polish_chart(
                fig,
                410,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

    # =========================================================================
    # ENGAGEMENT INTELLIGENCE
    # =========================================================================

    if has("engagement_rate", "view_count"):

        st.subheader("❤️ Reach vs Engagement")

        st.caption(
            "Identify videos with strong reach, strong engagement, or gaps between them."
        )

        engagement_df = df[
            ["view_count", "engagement_rate"]
        ].copy()

        engagement_df["view_count"] = pd.to_numeric(
            engagement_df["view_count"],
            errors="coerce",
        )

        engagement_df["engagement_rate"] = pd.to_numeric(
            engagement_df["engagement_rate"],
            errors="coerce",
        )

        engagement_df = engagement_df.dropna()

        engagement_df = engagement_df[
            (engagement_df["view_count"] >= 0)
            & (engagement_df["engagement_rate"] >= 0)
        ]

        if not engagement_df.empty:

            fig = px.scatter(
                engagement_df,
                x="view_count",
                y="engagement_rate",
                title="Views vs Engagement Rate",
                labels={
                    "view_count": "Views",
                    "engagement_rate": "Engagement Rate",
                },
                opacity=0.70,
            )

            fig.update_traces(
                marker=dict(
                    size=8,
                    color=THEME["accent"],
                    line=dict(
                        width=0,
                    ),
                ),
                hovertemplate=(
                    "Views: %{x:,.0f}<br>"
                    "Engagement: %{y:.2%}"
                    "<extra></extra>"
                ),
            )

            fig.update_xaxes(
                type="log",
                title="Views — Log Scale",
            )

            fig.update_yaxes(
                tickformat=".1%",
                title="Engagement Rate",
            )

            fig = polish_chart(
                fig,
                410,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

    # =========================================================================
    # EXECUTIVE INSIGHTS
    # =========================================================================

    st.divider()

    st.subheader("💡 Executive Insights")

    insight_cols = st.columns(3)

    # -------------------------------------------------------------------------
    # Category insight
    # -------------------------------------------------------------------------

    with insight_cols[0]:

        if has("category_name", "view_count"):

            category_perf = (
                df.groupby("category_name")["view_count"]
                .median()
                .sort_values(ascending=False)
            )

            if not category_perf.empty:

                best_cat = category_perf.index[0]
                best_cat_views = category_perf.iloc[0]

                st.info(
                    f"🏆 **Strongest category by median views:** "
                    f"**{best_cat}** with approximately "
                    f"**{format_number(best_cat_views)}** median views."
                )

            else:
                st.info("No category insight available.")

        else:
            st.info("Category data is unavailable.")

    # -------------------------------------------------------------------------
    # Publishing insight
    # -------------------------------------------------------------------------

    with insight_cols[1]:

        if has("publish_hour", "view_count"):

            hour_perf = (
                df.groupby("publish_hour")["view_count"]
                .median()
                .sort_values(ascending=False)
            )

            if not hour_perf.empty:

                best_hour = int(hour_perf.index[0])

                st.info(
                    f"🕐 **Best observed publishing hour:** "
                    f"**{best_hour:02d}:00**, based on median historical views."
                )

            else:
                st.info("No publishing-time insight available.")

        else:
            st.info("Publishing-hour data is unavailable.")

    # -------------------------------------------------------------------------
    # Engagement insight
    # -------------------------------------------------------------------------

    with insight_cols[2]:

        if has("engagement_rate", "view_count"):

            median_views = df["view_count"].median()
            median_engagement = df["engagement_rate"].median()

            st.info(
                f"❤️ **Typical performance:** "
                f"Median views are approximately "
                f"**{format_number(median_views)}**, while median "
                f"engagement is **{median_engagement * 100:.2f}%**."
            )

        else:
            st.info("Engagement data is unavailable.")

    # =========================================================================
    # FOOTNOTE
    # =========================================================================

    st.write("")

    st.caption(
        "ℹ️ Dashboard metrics are calculated from the currently loaded dataset. "
        "Observed relationships are descriptive and should not be interpreted "
        "as causal effects."
    )


# =====================================================================================
# PAGE: CONTENT ANALYTICS
# =====================================================================================
def page_content():
    st.title("🎬 Content Analytics")
    st.caption(
        "Explore video performance across categories, channels, publishing dates "
        "and engagement patterns."
    )

    df = df_raw.copy()

    # =========================================================================
    # FILTER PANEL
    # =========================================================================

    with st.expander("🎛️ Filters", expanded=True):

        c1, c2, c3 = st.columns(3)

        # Category filter
        if "category_name" in df.columns:
            categories = sorted(
                df["category_name"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        else:
            categories = []

        selected_categories = c1.multiselect(
            "🗂️ Category",
            options=categories,
            default=categories,
            key="content_category_filter",
        )

        # Channel filter
        if "channel_title" in df.columns:
            channels = sorted(
                df["channel_title"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        else:
            channels = []

        selected_channels = c2.multiselect(
            "📺 Channel",
            options=channels,
            default=[],
            key="content_channel_filter",
        )

        # Date filter
        date_range = None

        if "publish_date" in df.columns:

            valid_dates = df["publish_date"].dropna()

            if not valid_dates.empty:

                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()

                date_range = c3.date_input(
                    "📅 Publish date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="content_date_filter",
                )

    # =========================================================================
    # APPLY FILTERS
    # =========================================================================

    if selected_categories and "category_name" in df.columns:
        df = df[
            df["category_name"].astype(str).isin(selected_categories)
        ]

    if selected_channels and "channel_title" in df.columns:
        df = df[
            df["channel_title"].astype(str).isin(selected_channels)
        ]

    if (
        date_range
        and isinstance(date_range, tuple)
        and len(date_range) == 2
        and "publish_date" in df.columns
    ):

        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)

        df = df[
            (df["publish_date"] >= start_date)
            & (df["publish_date"] < end_date)
        ]

    # =========================================================================
    # EMPTY DATA PROTECTION
    # =========================================================================

    if df.empty:

        st.warning(
            "⚠️ No videos match the selected filters. "
            "Try changing the category, channel or date range."
        )

        return

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================

    if "view_count" in df.columns:

        df["view_count"] = pd.to_numeric(
            df["view_count"],
            errors="coerce",
        )

    df = df.dropna(
        subset=["view_count"]
    )

    if df.empty:

        st.warning("⚠️ No valid view-count records are available.")

        return

    # =========================================================================
    # KPI STRIP
    # =========================================================================

    st.markdown("### ⚡ Current Selection")

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        metric_card(
            "Videos",
            f"{len(df):,}",
            f"of {len(df_raw):,} total"
        )

    with k2:
        metric_card(
            "Total Views",
            format_number(df["view_count"].sum()),
            "observed views"
        )

    with k3:

        if "channel_title" in df.columns:
            channel_count = df["channel_title"].nunique()
        else:
            channel_count = 0

        metric_card(
            "Channels",
            f"{channel_count:,}",
            "unique channels"
        )

    with k4:

        if "category_name" in df.columns:
            category_count = df["category_name"].nunique()
        else:
            category_count = 0

        metric_card(
            "Categories",
            f"{category_count:,}",
            "content categories"
        )

    st.caption(
        f"Showing {len(df):,} of {len(df_raw):,} videos."
    )

    # =========================================================================
    # CHART HELPER
    # =========================================================================

    def polish_chart(fig, height=420):

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=height,
            margin=dict(
                l=20,
                r=20,
                t=65,
                b=35,
            ),
            font=dict(
                color=THEME["text"]
            ),
            title=dict(
                font=dict(
                    size=17,
                    color=THEME["text"]
                ),
                x=0.02,
                xanchor="left",
            ),
            xaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False,
                title_font=dict(
                    color=THEME["muted"]
                ),
                tickfont=dict(
                    color=THEME["muted"]
                ),
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False,
                title_font=dict(
                    color=THEME["muted"]
                ),
                tickfont=dict(
                    color=THEME["muted"]
                ),
            ),
            hoverlabel=dict(
                bgcolor=THEME["card"],
                bordercolor=THEME["card_border"],
                font=dict(
                    color=THEME["text"]
                ),
            ),
        )

        return fig

    # =========================================================================
    # TOP / BOTTOM VIDEOS
    # =========================================================================

    c1, c2 = st.columns(2)

    # -------------------------------------------------------------------------
    # TOP 10
    # -------------------------------------------------------------------------

    with c1:

        st.markdown("### 🏆 Top 10 Videos")

        if "title" in df.columns:

            top = (
                df.nlargest(10, "view_count")
                .copy()
            )

            top["display_title"] = (
                top["title"]
                .fillna("Untitled")
                .astype(str)
                .apply(
                    lambda x:
                    x if len(x) <= 55
                    else x[:52] + "..."
                )
            )

            fig = px.bar(
                top,
                x="view_count",
                y="display_title",
                orientation="h",
                title="Highest-Viewed Videos",
                labels={
                    "view_count": "Views",
                    "display_title": "Video",
                },
            )

            fig.update_traces(
                marker_color=THEME["primary"],
                marker_line_width=0,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Views: %{x:,.0f}"
                    "<extra></extra>"
                ),
            )

            fig.update_layout(
                yaxis={
                    "categoryorder": "total ascending"
                }
            )

            fig = polish_chart(fig)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

    # -------------------------------------------------------------------------
    # BOTTOM 10
    # -------------------------------------------------------------------------

    with c2:

        st.markdown("### 📉 Bottom 10 Videos")

        if "title" in df.columns:

            bottom = (
                df.nsmallest(10, "view_count")
                .copy()
            )

            bottom["display_title"] = (
                bottom["title"]
                .fillna("Untitled")
                .astype(str)
                .apply(
                    lambda x:
                    x if len(x) <= 55
                    else x[:52] + "..."
                )
            )

            fig = px.bar(
                bottom,
                x="view_count",
                y="display_title",
                orientation="h",
                title="Lowest-Viewed Videos",
                labels={
                    "view_count": "Views",
                    "display_title": "Video",
                },
            )

            fig.update_traces(
                marker_color=THEME["accent"],
                marker_line_width=0,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Views: %{x:,.0f}"
                    "<extra></extra>"
                ),
            )

            fig.update_layout(
                yaxis={
                    "categoryorder": "total descending"
                }
            )

            fig = polish_chart(fig)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

    # =========================================================================
    # VIEWS DISTRIBUTION + CATEGORY
    # =========================================================================

    c3, c4 = st.columns(2)

    # -------------------------------------------------------------------------
    # VIEW DISTRIBUTION
    # -------------------------------------------------------------------------

    with c3:

        st.markdown("### 👁️ Views Distribution")

        views = (
            pd.to_numeric(
                df["view_count"],
                errors="coerce"
            )
            .dropna()
        )

        views = views[views > 0]

        if len(views) >= 2:

            log_views = np.log10(views)

            distribution_df = pd.DataFrame(
                {
                    "log_views": log_views
                }
            )

            fig = px.histogram(
                distribution_df,
                x="log_views",
                nbins=20,
                title="Video View Distribution",
                labels={
                    "log_views": "Views — Log₁₀ Scale",
                    "count": "Videos",
                },
            )

            fig.update_traces(
                marker_color=THEME["primary"],
                marker_line_width=0,
                opacity=0.85,
                hovertemplate=(
                    "Log₁₀ Views: %{x:.2f}<br>"
                    "Videos: %{y:,}"
                    "<extra></extra>"
                ),
            )

            fig = polish_chart(fig)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

        else:

            st.info(
                "Not enough valid positive view values "
                "to display the distribution."
            )

    # -------------------------------------------------------------------------
    # CATEGORY PERFORMANCE
    # -------------------------------------------------------------------------

    with c4:

        st.markdown("### 🗂️ Average Views by Category")

        if "category_name" in df.columns:

            category_avg = (
                df.groupby(
                    "category_name",
                    as_index=False
                )["view_count"]
                .mean()
                .sort_values(
                    "view_count",
                    ascending=False
                )
            )

            fig = px.bar(
                category_avg,
                x="category_name",
                y="view_count",
                title="Average Views by Category",
                labels={
                    "category_name": "Category",
                    "view_count": "Average Views",
                },
            )

            fig.update_traces(
                marker_color=THEME["primary"],
                marker_line_width=0,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Average Views: %{y:,.0f}"
                    "<extra></extra>"
                ),
            )

            fig = polish_chart(fig)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

    # =========================================================================
    # ENGAGEMENT INTELLIGENCE
    # =========================================================================

    if {
        "view_count",
        "like_count",
        "comment_count",
    }.issubset(df.columns):

        st.markdown("---")

        st.markdown("### 💬 Engagement Intelligence")

        engagement_df = df[
            [
                "view_count",
                "like_count",
                "comment_count",
            ]
        ].copy()

        engagement_df = engagement_df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        engagement_df = engagement_df.dropna()

        engagement_df = engagement_df[
            engagement_df["view_count"] > 0
        ]

        if not engagement_df.empty:

            engagement_df["like_rate_pct"] = (
                engagement_df["like_count"]
                / engagement_df["view_count"]
                * 100
            )

            engagement_df["comment_rate_pct"] = (
                engagement_df["comment_count"]
                / engagement_df["view_count"]
                * 100
            )

            c5, c6 = st.columns(2)

            # -----------------------------------------------------------------
            # LIKE RATE
            # -----------------------------------------------------------------

            with c5:

                fig = px.histogram(
                    engagement_df,
                    x="like_rate_pct",
                    nbins=25,
                    title="Like Rate Distribution",
                    labels={
                        "like_rate_pct": "Like Rate (%)",
                        "count": "Videos",
                    },
                )

                fig.update_traces(
                    marker_color=THEME["accent"],
                    marker_line_width=0,
                    opacity=0.82,
                )

                fig = polish_chart(
                    fig,
                    height=360
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                    },
                )

            # -----------------------------------------------------------------
            # VIEWS VS LIKES
            # -----------------------------------------------------------------

            with c6:

                scatter_df = engagement_df[
                    [
                        "view_count",
                        "like_count"
                    ]
                ].copy()

                scatter_df = scatter_df[
                    (scatter_df["view_count"] > 0)
                    & (scatter_df["like_count"] > 0)
                ]

                if not scatter_df.empty:

                    fig = px.scatter(
                        scatter_df,
                        x="view_count",
                        y="like_count",
                        title="Views vs Likes",
                        labels={
                            "view_count": "Views",
                            "like_count": "Likes",
                        },
                        log_x=True,
                        log_y=True,
                    )

                    fig.update_traces(
                        marker=dict(
                            size=8,
                            opacity=0.65,
                        ),
                        hovertemplate=(
                            "Views: %{x:,.0f}<br>"
                            "Likes: %{y:,.0f}"
                            "<extra></extra>"
                        ),
                    )

                    fig = polish_chart(
                        fig,
                        height=360
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={
                            "displayModeBar": False,
                            "responsive": True,
                        },
                    )

    # =========================================================================
    # VIDEO PERFORMANCE TABLE
    # =========================================================================

    st.markdown("---")

    st.markdown("### 📋 Video Performance Table")

    show_cols = [
        col
        for col in [
            "title",
            "channel_title",
            "category_name",
            "view_count",
            "like_count",
            "comment_count",
            "engagement_rate",
            "publish_date",
        ]
        if col in df.columns
    ]

    display_df = (
        df[show_cols]
        .sort_values(
            "view_count",
            ascending=False
        )
        .reset_index(drop=True)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "💡 Use the filters above to isolate specific content segments "
        "before comparing performance."
    )


# =====================================================================================
# PAGE: ENGAGEMENT ANALYTICS
# =====================================================================================
def page_engagement():
    st.title("💬 Engagement Analytics")
    st.caption(
        "Understand how viewers interact with videos through likes, comments, "
        "engagement rate and category-level engagement patterns."
    )

    df = df_raw.copy()

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================

    required_cols = ["view_count"]

    for col in [
        "view_count",
        "like_count",
        "comment_count",
        "engagement_rate",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # Create engagement rate if it does not already exist
    if (
        "engagement_rate" not in df.columns
        and {
            "view_count",
            "like_count",
            "comment_count",
        }.issubset(df.columns)
    ):
        df["engagement_rate"] = np.where(
            df["view_count"] > 0,
            (
                df["like_count"].fillna(0)
                + df["comment_count"].fillna(0)
            )
            / df["view_count"],
            np.nan,
        )

    # =========================================================================
    # KPI SECTION
    # =========================================================================

    st.markdown("### ⚡ Engagement Snapshot")

    k1, k2, k3, k4 = st.columns(4)

    # Total likes
    with k1:
        if "like_count" in df.columns:
            total_likes = df["like_count"].sum()
            metric_card(
                "Total Likes",
                format_number(total_likes),
                "audience reactions",
            )
        else:
            metric_card("Total Likes", "—", "")

    # Total comments
    with k2:
        if "comment_count" in df.columns:
            total_comments = df["comment_count"].sum()
            metric_card(
                "Total Comments",
                format_number(total_comments),
                "audience conversations",
            )
        else:
            metric_card("Total Comments", "—", "")

    # Average engagement
    with k3:
        if "engagement_rate" in df.columns:

            avg_engagement = (
                df["engagement_rate"]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .mean()
            )

            if pd.notna(avg_engagement):
                metric_card(
                    "Avg Engagement",
                    f"{avg_engagement * 100:.2f}%",
                    "likes + comments / views",
                )
            else:
                metric_card(
                    "Avg Engagement",
                    "—",
                    "no valid rates",
                )

        else:
            metric_card(
                "Avg Engagement",
                "—",
                "",
            )

    # Median engagement
    with k4:
        if "engagement_rate" in df.columns:

            median_engagement = (
                df["engagement_rate"]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .median()
            )

            if pd.notna(median_engagement):
                metric_card(
                    "Median Engagement",
                    f"{median_engagement * 100:.2f}%",
                    "central engagement level",
                )
            else:
                metric_card(
                    "Median Engagement",
                    "—",
                    "",
                )
        else:
            metric_card(
                "Median Engagement",
                "—",
                "",
            )

    # =========================================================================
    # CHART POLISHER
    # =========================================================================

    def polish_chart(fig, height=400):

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=height,

            margin=dict(
                l=20,
                r=20,
                t=65,
                b=40,
            ),

            font=dict(
                color=THEME["text"]
            ),

            title=dict(
                font=dict(
                    size=17,
                    color=THEME["text"]
                ),
                x=0.02,
                xanchor="left",
            ),

            xaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False,
                tickfont=dict(
                    color=THEME["muted"]
                ),
                title_font=dict(
                    color=THEME["muted"]
                ),
            ),

            yaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False,
                tickfont=dict(
                    color=THEME["muted"]
                ),
                title_font=dict(
                    color=THEME["muted"]
                ),
            ),

            hoverlabel=dict(
                bgcolor=THEME["card"],
                bordercolor=THEME["card_border"],
                font=dict(
                    color=THEME["text"]
                ),
            ),

            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(
                    color=THEME["text"]
                ),
            ),
        )

        return fig

    # =========================================================================
    # ENGAGEMENT RATE DISTRIBUTION
    # =========================================================================

    if "engagement_rate" in df.columns:

        engagement = (
            df["engagement_rate"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        engagement = engagement[
            engagement >= 0
        ]

        st.markdown("### 📈 Engagement Distribution")

        if len(engagement) >= 2:

            distribution_df = pd.DataFrame(
                {
                    "engagement_pct":
                    engagement * 100
                }
            )

            fig = px.histogram(
                distribution_df,
                x="engagement_pct",
                nbins=30,
                title="Engagement Rate Distribution",
                labels={
                    "engagement_pct":
                    "Engagement Rate (%)",
                    "count":
                    "Number of Videos",
                },
            )

            fig.update_traces(
                marker_color=THEME["primary"],
                marker_line_width=0,
                opacity=0.85,
                hovertemplate=(
                    "Engagement: %{x:.2f}%<br>"
                    "Videos: %{y:,}"
                    "<extra></extra>"
                ),
            )

            fig = polish_chart(
                fig,
                height=390
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

        else:
            st.info(
                "Not enough valid engagement-rate values "
                "to display the distribution."
            )

    # =========================================================================
    # VIEWS VS LIKES
    # =========================================================================

    if {
        "view_count",
        "like_count",
    }.issubset(df.columns):

        st.markdown("### ❤️ Reach vs Audience Interaction")

        likes_df = df[
            [
                "view_count",
                "like_count"
            ]
            + (
                ["category_name"]
                if "category_name" in df.columns
                else []
            )
            + (
                ["title"]
                if "title" in df.columns
                else []
            )
        ].copy()

        likes_df = likes_df.dropna(
            subset=[
                "view_count",
                "like_count",
            ]
        )

        likes_df = likes_df[
            (likes_df["view_count"] > 0)
            & (likes_df["like_count"] > 0)
        ]

        if not likes_df.empty:

            fig = px.scatter(
                likes_df,
                x="view_count",
                y="like_count",
                color=(
                    "category_name"
                    if "category_name" in likes_df.columns
                    else None
                ),
                hover_name=(
                    "title"
                    if "title" in likes_df.columns
                    else None
                ),
                log_x=True,
                log_y=True,
                title="Views vs Likes",
                labels={
                    "view_count": "Views",
                    "like_count": "Likes",
                    "category_name": "Category",
                },
            )

            fig.update_traces(
                marker=dict(
                    size=9,
                    opacity=0.70,
                ),
                hovertemplate=(
                    "Views: %{x:,.0f}<br>"
                    "Likes: %{y:,.0f}"
                    "<extra></extra>"
                ),
            )

            fig = polish_chart(
                fig,
                height=440
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

    # =========================================================================
    # VIEWS VS COMMENTS
    # =========================================================================

    if {
        "view_count",
        "comment_count",
    }.issubset(df.columns):

        c1, c2 = st.columns(2)

        with c1:

            comments_df = df[
                [
                    "view_count",
                    "comment_count"
                ]
                + (
                    ["category_name"]
                    if "category_name" in df.columns
                    else []
                )
                + (
                    ["title"]
                    if "title" in df.columns
                    else []
                )
            ].copy()

            comments_df = comments_df.dropna(
                subset=[
                    "view_count",
                    "comment_count",
                ]
            )

            comments_df = comments_df[
                (comments_df["view_count"] > 0)
                & (comments_df["comment_count"] > 0)
            ]

            if not comments_df.empty:

                fig = px.scatter(
                    comments_df,
                    x="view_count",
                    y="comment_count",
                    color=(
                        "category_name"
                        if "category_name"
                        in comments_df.columns
                        else None
                    ),
                    hover_name=(
                        "title"
                        if "title" in comments_df.columns
                        else None
                    ),
                    log_x=True,
                    log_y=True,
                    title="Views vs Comments",
                    labels={
                        "view_count": "Views",
                        "comment_count": "Comments",
                        "category_name": "Category",
                    },
                )

                fig.update_traces(
                    marker=dict(
                        size=9,
                        opacity=0.70,
                    ),
                )

                fig = polish_chart(
                    fig,
                    height=420
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                    },
                )

        # =====================================================================
        # CATEGORY ENGAGEMENT
        # =====================================================================

        with c2:

            if {
                "category_name",
                "engagement_rate",
            }.issubset(df.columns):

                category_engagement = (
                    df.groupby(
                        "category_name",
                        as_index=False
                    )[
                        "engagement_rate"
                    ]
                    .mean()
                    .dropna()
                    .sort_values(
                        "engagement_rate",
                        ascending=False
                    )
                )

                category_engagement[
                    "engagement_pct"
                ] = (
                    category_engagement[
                        "engagement_rate"
                    ] * 100
                )

                fig = px.bar(
                    category_engagement,
                    x="category_name",
                    y="engagement_pct",
                    title="Average Engagement by Category",
                    labels={
                        "category_name": "Category",
                        "engagement_pct":
                        "Average Engagement (%)",
                    },
                )

                fig.update_traces(
                    marker_color=THEME["primary"],
                    marker_line_width=0,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Engagement: %{y:.2f}%"
                        "<extra></extra>"
                    ),
                )

                fig = polish_chart(
                    fig,
                    height=420
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                    },
                )

    # =========================================================================
    # ENGAGEMENT OVER TIME
    # =========================================================================

    if {
        "publish_date",
        "engagement_rate",
    }.issubset(df.columns):

        time_df = df[
            [
                "publish_date",
                "engagement_rate"
            ]
        ].dropna()

        if not time_df.empty:

            time_df["engagement_pct"] = (
                time_df["engagement_rate"] * 100
            )

            time_df["publish_date"] = pd.to_datetime(
                time_df["publish_date"],
                errors="coerce"
            )

            time_df = time_df.dropna(
                subset=["publish_date"]
            )

            # Daily data can be noisy, so use monthly averages
            time_df["month"] = (
                time_df["publish_date"]
                .dt.to_period("M")
                .dt.to_timestamp()
            )

            monthly_engagement = (
                time_df.groupby(
                    "month",
                    as_index=False
                )["engagement_pct"]
                .mean()
            )

            if not monthly_engagement.empty:

                st.markdown("---")
                st.markdown("### 📅 Engagement Over Time")

                fig = px.line(
                    monthly_engagement,
                    x="month",
                    y="engagement_pct",
                    markers=True,
                    title="Average Monthly Engagement Rate",
                    labels={
                        "month": "Month",
                        "engagement_pct":
                        "Average Engagement (%)",
                    },
                )

                fig.update_traces(
                    line=dict(
                        width=3
                    ),
                    marker=dict(
                        size=7
                    ),
                    hovertemplate=(
                        "%{x|%b %Y}<br>"
                        "Engagement: %{y:.2f}%"
                        "<extra></extra>"
                    ),
                )

                fig = polish_chart(
                    fig,
                    height=400
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                    },
                )

    # =========================================================================
    # TOP ENGAGEMENT VIDEOS
    # =========================================================================

    if {
        "engagement_rate",
        "view_count",
    }.issubset(df.columns):

        top_engagement = df.copy()

        top_engagement[
            "engagement_rate"
        ] = pd.to_numeric(
            top_engagement["engagement_rate"],
            errors="coerce"
        )

        top_engagement = (
            top_engagement
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna(
                subset=[
                    "engagement_rate"
                ]
            )
        )

        if not top_engagement.empty:

            st.markdown("---")
            st.markdown("### 🏆 Highest Engagement Videos")

            display_cols = [
                col
                for col in [
                    "title",
                    "channel_title",
                    "category_name",
                    "view_count",
                    "like_count",
                    "comment_count",
                    "engagement_rate",
                ]
                if col in top_engagement.columns
            ]

            top_engagement = (
                top_engagement
                .sort_values(
                    "engagement_rate",
                    ascending=False
                )
                .head(10)
                [display_cols]
                .copy()
            )

            if "engagement_rate" in top_engagement.columns:

                top_engagement[
                    "engagement_rate"
                ] = (
                    top_engagement[
                        "engagement_rate"
                    ] * 100
                ).round(2)

            if "view_count" in top_engagement.columns:

                top_engagement[
                    "view_count"
                ] = top_engagement[
                    "view_count"
                ].round(0).astype("int64")

            st.dataframe(
                top_engagement,
                use_container_width=True,
                hide_index=True,
            )

    # =========================================================================
    # BUSINESS INSIGHT
    # =========================================================================

    if "engagement_rate" in df.columns:

        valid_eng = (
            df["engagement_rate"]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if not valid_eng.empty:

            avg_eng = valid_eng.mean()
            median_eng = valid_eng.median()

            st.markdown("---")
            st.markdown("### 💡 Engagement Intelligence")

            if avg_eng > median_eng:

                insight = (
                    "The average engagement rate is above the median, "
                    "suggesting that a smaller group of highly interactive "
                    "videos is pulling the overall average upward."
                )

            elif avg_eng < median_eng:

                insight = (
                    "The median engagement rate is above the average, "
                    "indicating that lower-engagement videos may be "
                    "pulling the overall average downward."
                )

            else:

                insight = (
                    "Average and median engagement are closely aligned, "
                    "suggesting a relatively balanced engagement distribution."
                )

            st.info(
                f"📌 **Business Insight:** {insight}"
            )

    st.caption(
        "💡 Engagement rate measures likes and comments relative to views. "
        "Scatter plots use logarithmic axes because YouTube performance "
        "typically spans a wide range of values."
    )


# =====================================================================================
# PAGE: TREND ANALYSIS
# =====================================================================================
def page_trends():
    st.title("📈 Trend Analysis")
    st.caption(
        "Explore how YouTube views, publishing activity and content performance "
        "change over time, across hours, weekdays and content categories."
    )

    df = df_raw.copy()

    # =========================================================================
    # VALIDATION
    # =========================================================================

    if "publish_date" not in df.columns:
        st.info(
            "Trend analysis is unavailable because the dataset does not contain "
            "a usable `publish_date` column."
        )
        return

    df["publish_date"] = pd.to_datetime(
        df["publish_date"],
        errors="coerce"
    )

    d = df.dropna(subset=["publish_date"]).copy()

    if d.empty:
        st.info(
            "Trend analysis is unavailable because no valid publication dates "
            "were found in the dataset."
        )
        return

    if "view_count" in d.columns:
        d["view_count"] = pd.to_numeric(
            d["view_count"],
            errors="coerce"
        )

    # =========================================================================
    # CHART POLISHER
    # =========================================================================

    def polish_trend_chart(fig, height=400):

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=height,

            margin=dict(
                l=20,
                r=20,
                t=65,
                b=40,
            ),

            font=dict(
                color=THEME["text"]
            ),

            title=dict(
                font=dict(
                    size=17,
                    color=THEME["text"]
                ),
                x=0.02,
                xanchor="left",
            ),

            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(
                    color=THEME["muted"]
                ),
                title_font=dict(
                    color=THEME["muted"]
                ),
            ),

            yaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False,
                tickfont=dict(
                    color=THEME["muted"]
                ),
                title_font=dict(
                    color=THEME["muted"]
                ),
            ),

            hoverlabel=dict(
                bgcolor=THEME["card"],
                bordercolor=THEME["card_border"],
                font=dict(
                    color=THEME["text"]
                ),
            ),

            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(
                    color=THEME["text"]
                ),
                title_font=dict(
                    color=THEME["muted"]
                ),
            ),
        )

        return fig

    # =========================================================================
    # TREND SNAPSHOT
    # =========================================================================

    st.markdown("### ⚡ Trend Snapshot")

    first_date = d["publish_date"].min()
    last_date = d["publish_date"].max()

    active_days = d["publish_date"].dt.date.nunique()

    total_views = (
        d["view_count"].sum()
        if "view_count" in d.columns
        else np.nan
    )

    avg_views = (
        d["view_count"].mean()
        if "view_count" in d.columns
        else np.nan
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        metric_card(
            "Date Range",
            f"{first_date:%b %Y}",
            f"to {last_date:%b %Y}",
        )

    with k2:
        metric_card(
            "Active Dates",
            f"{active_days:,}",
            "publishing dates",
        )

    with k3:
        metric_card(
            "Total Views",
            format_number(total_views),
            "within dated records",
        )

    with k4:
        metric_card(
            "Avg Views / Video",
            format_number(avg_views),
            "historical average",
        )

    st.write("")

    # =========================================================================
    # PREPARE DAILY DATA
    # =========================================================================

    d["date"] = d["publish_date"].dt.floor("D")

    if "view_count" in d.columns:

        daily = (
            d.groupby(
                "date",
                as_index=False
            )
            .agg(
                total_views=("view_count", "sum"),
                average_views=("view_count", "mean"),
                uploads=("view_count", "size"),
            )
            .sort_values("date")
        )

    else:

        daily = (
            d.groupby(
                "date",
                as_index=False
            )
            .size()
            .rename(
                columns={
                    "size": "uploads"
                }
            )
        )

    # =========================================================================
    # PERFORMANCE OVER TIME
    # =========================================================================

    st.markdown("### 📊 Performance Over Time")
    st.caption(
        "Compare historical view activity with publishing volume to understand "
        "how content performance changes across the available timeline."
    )

    c1, c2 = st.columns(2)

    # -------------------------------------------------------------------------
    # TOTAL VIEWS OVER TIME
    # -------------------------------------------------------------------------

    with c1:

        if "total_views" in daily.columns:

            fig = px.line(
                daily,
                x="date",
                y="total_views",
                markers=True,
                title="Total Views Over Time",
                labels={
                    "date": "Publish Date",
                    "total_views": "Total Views",
                },
            )

            fig.update_traces(
                line=dict(
                    color=THEME["primary"],
                    width=3,
                ),
                marker=dict(
                    color=THEME["primary"],
                    size=6,
                ),
                fill="tozeroy",
                fillcolor=THEME["primary_soft"],
                hovertemplate=(
                    "%{x|%d %b %Y}<br>"
                    "Views: %{y:,.0f}"
                    "<extra></extra>"
                ),
            )

            fig = polish_trend_chart(
                fig,
                height=410
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

    # -------------------------------------------------------------------------
    # UPLOADS OVER TIME
    # -------------------------------------------------------------------------

    with c2:

        fig = px.bar(
            daily,
            x="date",
            y="uploads",
            title="Publishing Activity Over Time",
            labels={
                "date": "Publish Date",
                "uploads": "Videos Published",
            },
        )

        fig.update_traces(
            marker_color=THEME["accent"],
            marker_line_width=0,
            opacity=0.85,
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "Uploads: %{y:,}"
                "<extra></extra>"
            ),
        )

        fig = polish_trend_chart(
            fig,
            height=410
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
            },
        )

    # =========================================================================
    # ROLLING VIEW TREND
    # =========================================================================

    if (
        "total_views" in daily.columns
        and len(daily) >= 3
    ):

        st.markdown("---")
        st.markdown("### 📈 Smoothed Performance Trend")

        window = min(
            7,
            max(
                2,
                len(daily) // 5
            )
        )

        daily["rolling_views"] = (
            daily["total_views"]
            .rolling(
                window=window,
                min_periods=1
            )
            .mean()
        )

        fig = go.Figure()

        # Raw trend
        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["total_views"],
                mode="lines",
                name="Observed Views",
                line=dict(
                    color=THEME["muted"],
                    width=1.5,
                ),
                opacity=0.45,
                hovertemplate=(
                    "%{x|%d %b %Y}<br>"
                    "Observed: %{y:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

        # Rolling trend
        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["rolling_views"],
                mode="lines",
                name=f"{window}-Period Rolling Average",
                line=dict(
                    color=THEME["primary"],
                    width=4,
                ),
                hovertemplate=(
                    "%{x|%d %b %Y}<br>"
                    "Rolling Avg: %{y:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title="Views Trend with Rolling Average",
            xaxis_title="Publish Date",
            yaxis_title="Views",
        )

        fig = polish_trend_chart(
            fig,
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
            },
        )

        st.caption(
            "The rolling average smooths short-term fluctuations and makes "
            "the broader direction of historical performance easier to inspect."
        )

    # =========================================================================
    # PUBLISHING TIME INTELLIGENCE
    # =========================================================================

    st.markdown("---")
    st.markdown("### 🕒 Publishing Time Intelligence")

    time_c1, time_c2 = st.columns(2)

    # -------------------------------------------------------------------------
    # PUBLISH HOUR
    # -------------------------------------------------------------------------

    with time_c1:

        if {
            "publish_hour",
            "view_count"
        }.issubset(d.columns):

            hour_df = d[
                [
                    "publish_hour",
                    "view_count"
                ]
            ].copy()

            hour_df["publish_hour"] = pd.to_numeric(
                hour_df["publish_hour"],
                errors="coerce"
            )

            hour_df = hour_df.dropna()

            hour_df = hour_df[
                hour_df["publish_hour"].between(
                    0,
                    23
                )
            ]

            if not hour_df.empty:

                hourly = (
                    hour_df.groupby(
                        "publish_hour",
                        as_index=False
                    )
                    .agg(
                        avg_views=(
                            "view_count",
                            "mean"
                        ),
                        videos=(
                            "view_count",
                            "size"
                        ),
                    )
                    .sort_values(
                        "publish_hour"
                    )
                )

                fig = px.line(
                    hourly,
                    x="publish_hour",
                    y="avg_views",
                    markers=True,
                    title="Average Views by Publish Hour",
                    labels={
                        "publish_hour": "Publish Hour",
                        "avg_views": "Average Views",
                    },
                )

                fig.update_traces(
                    line=dict(
                        color=THEME["primary"],
                        width=3,
                    ),
                    marker=dict(
                        color=THEME["primary"],
                        size=8,
                    ),
                    hovertemplate=(
                        "Hour: %{x}:00<br>"
                        "Avg Views: %{y:,.0f}"
                        "<extra></extra>"
                    ),
                )

                fig.update_xaxes(
                    tickmode="linear",
                    tick0=0,
                    dtick=2,
                    range=[0, 23],
                )

                fig = polish_trend_chart(
                    fig,
                    height=410
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                    },
                )

    # -------------------------------------------------------------------------
    # WEEKDAY PERFORMANCE
    # -------------------------------------------------------------------------

    with time_c2:

        if "view_count" in d.columns:

            weekday_df = d.copy()

            weekday_df["weekday"] = (
                weekday_df["publish_date"]
                .dt.day_name()
            )

            weekday_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]

            weekday_summary = (
                weekday_df.groupby(
                    "weekday",
                    as_index=False
                )
                .agg(
                    avg_views=(
                        "view_count",
                        "mean"
                    ),
                    uploads=(
                        "view_count",
                        "size"
                    ),
                )
            )

            weekday_summary["weekday"] = pd.Categorical(
                weekday_summary["weekday"],
                categories=weekday_order,
                ordered=True,
            )

            weekday_summary = (
                weekday_summary
                .sort_values("weekday")
            )

            fig = px.bar(
                weekday_summary,
                x="weekday",
                y="avg_views",
                title="Average Views by Publish Day",
                labels={
                    "weekday": "Day",
                    "avg_views": "Average Views",
                },
            )

            fig.update_traces(
                marker_color=THEME["primary"],
                marker_line_width=0,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Avg Views: %{y:,.0f}"
                    "<extra></extra>"
                ),
            )

            fig = polish_trend_chart(
                fig,
                height=410
            )

            fig.update_xaxes(
                tickangle=-30
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

    # =========================================================================
    # MONTHLY INTELLIGENCE
    # =========================================================================

    if "view_count" in d.columns:

        st.markdown("---")
        st.markdown("### 🗓️ Monthly Performance")

        monthly_df = d.copy()

        monthly_df["month"] = (
            monthly_df["publish_date"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        monthly = (
            monthly_df.groupby(
                "month",
                as_index=False
            )
            .agg(
                total_views=(
                    "view_count",
                    "sum"
                ),
                avg_views=(
                    "view_count",
                    "mean"
                ),
                uploads=(
                    "view_count",
                    "size"
                ),
            )
            .sort_values("month")
        )

        if not monthly.empty:

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=monthly["month"],
                    y=monthly["total_views"],
                    name="Total Views",
                    marker=dict(
                        color=THEME["primary"]
                    ),
                    opacity=0.75,
                    hovertemplate=(
                        "%{x|%b %Y}<br>"
                        "Views: %{y:,.0f}"
                        "<extra></extra>"
                    ),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=monthly["month"],
                    y=monthly["avg_views"],
                    name="Average Views / Video",
                    mode="lines+markers",
                    yaxis="y2",
                    line=dict(
                        color=THEME["accent"],
                        width=3,
                    ),
                    marker=dict(
                        size=7
                    ),
                    hovertemplate=(
                        "%{x|%b %Y}<br>"
                        "Avg Views: %{y:,.0f}"
                        "<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                title="Monthly Views and Average Video Performance",

                xaxis_title="Month",

                yaxis=dict(
                    title="Total Views"
                ),

                yaxis2=dict(
                    title="Average Views / Video",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),

                barmode="group",
            )

            fig = polish_trend_chart(
                fig,
                height=440
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

    # =========================================================================
    # CATEGORY TREND ANALYSIS
    # =========================================================================

    if {
        "category_name",
        "view_count"
    }.issubset(d.columns):

        st.markdown("---")
        st.markdown("### 🗂️ Category Trends Over Time")

        cat_df = d[
            [
                "publish_date",
                "category_name",
                "view_count"
            ]
        ].dropna().copy()

        cat_df["month"] = (
            cat_df["publish_date"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        cat_trend = (
            cat_df.groupby(
                [
                    "month",
                    "category_name"
                ],
                as_index=False
            )["view_count"]
            .sum()
        )

        if not cat_trend.empty:

            fig = px.line(
                cat_trend,
                x="month",
                y="view_count",
                color="category_name",
                markers=True,
                title="Monthly Views by Content Category",
                labels={
                    "month": "Month",
                    "view_count": "Total Views",
                    "category_name": "Category",
                },
            )

            fig.update_traces(
                line=dict(
                    width=2.5
                ),
                marker=dict(
                    size=6
                ),
                hovertemplate=(
                    "%{x|%b %Y}<br>"
                    "Views: %{y:,.0f}"
                    "<extra></extra>"
                ),
            )

            fig = polish_trend_chart(
                fig,
                height=470
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

    # =========================================================================
    # TREND BUSINESS INTELLIGENCE
    # =========================================================================

    if "view_count" in d.columns:

        st.markdown("---")
        st.markdown("### 💡 Trend Intelligence")

        insight_cols = st.columns(3)

        # ---------------------------------------------------------------------
        # BEST PUBLISH HOUR
        # ---------------------------------------------------------------------

        with insight_cols[0]:

            if "publish_hour" in d.columns:

                hour_insight = d[
                    [
                        "publish_hour",
                        "view_count"
                    ]
                ].copy()

                hour_insight["publish_hour"] = pd.to_numeric(
                    hour_insight["publish_hour"],
                    errors="coerce"
                )

                hour_insight = hour_insight.dropna()

                hour_insight = hour_insight[
                    hour_insight["publish_hour"].between(
                        0,
                        23
                    )
                ]

                if not hour_insight.empty:

                    hour_stats = (
                        hour_insight.groupby(
                            "publish_hour"
                        )["view_count"]
                        .agg(
                            ["mean", "count"]
                        )
                    )

                    # Require at least 2 videos where possible
                    reliable_hours = hour_stats[
                        hour_stats["count"] >= 2
                    ]

                    if reliable_hours.empty:
                        reliable_hours = hour_stats

                    best_hour = int(
                        reliable_hours[
                            "mean"
                        ].idxmax()
                    )

                    metric_card(
                        "Strongest Publish Hour",
                        f"{best_hour:02d}:00",
                        "highest historical avg views",
                    )

        # ---------------------------------------------------------------------
        # BEST WEEKDAY
        # ---------------------------------------------------------------------

        with insight_cols[1]:

            day_stats = (
                d.assign(
                    weekday=d[
                        "publish_date"
                    ].dt.day_name()
                )
                .groupby(
                    "weekday"
                )["view_count"]
                .agg(
                    ["mean", "count"]
                )
            )

            reliable_days = day_stats[
                day_stats["count"] >= 2
            ]

            if reliable_days.empty:
                reliable_days = day_stats

            if not reliable_days.empty:

                best_day = (
                    reliable_days[
                        "mean"
                    ].idxmax()
                )

                metric_card(
                    "Strongest Publish Day",
                    best_day,
                    "highest historical avg views",
                )

        # ---------------------------------------------------------------------
        # STRONGEST CATEGORY
        # ---------------------------------------------------------------------

        with insight_cols[2]:

            if "category_name" in d.columns:

                category_stats = (
                    d.groupby(
                        "category_name"
                    )["view_count"]
                    .agg(
                        ["mean", "count"]
                    )
                )

                reliable_categories = category_stats[
                    category_stats["count"] >= 2
                ]

                if reliable_categories.empty:
                    reliable_categories = category_stats

                if not reliable_categories.empty:

                    best_category = (
                        reliable_categories[
                            "mean"
                        ].idxmax()
                    )

                    metric_card(
                        "Strongest Category",
                        str(best_category),
                        "highest historical avg views",
                    )

    # =========================================================================
    # INTERPRETATION
    # =========================================================================

    st.markdown("---")
    st.markdown("### 🎯 How to Interpret These Trends")

    st.info(
        "📌 **Business Insight:** Use these charts to identify recurring historical "
        "patterns in publishing activity, timing and category performance. A high-performing "
        "hour, weekday or category represents an association in this dataset — it should "
        "not automatically be interpreted as proof that publishing at that time or choosing "
        "that category causes higher views."
    )

    st.caption(
        "💡 Trend Analysis is descriptive. Historical patterns can support content "
        "planning, but they should be combined with audience context, content quality, "
        "seasonality and model-based analysis before making decisions."
    )

# =====================================================================================
# PAGE: ML OVERVIEW
# =====================================================================================
def page_ml_overview():

    st.title("🤖 ML Overview")
    st.caption(
        "Explore the machine learning architecture, leakage-safe feature design, "
        "model readiness, evaluation strategy and business interpretation."
    )

    # =========================================================================
    # 1. MACHINE LEARNING OBJECTIVES
    # =========================================================================
    st.markdown("### 🎯 Machine Learning Objectives")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 🔮 Regression — View Prediction")
        st.write(
            "Estimate the expected **view_count** of a planned or newly "
            "published video using information available before or at "
            "publication time."
        )

        st.caption(
            "Output: estimated number of views."
        )

    with c2:
        st.markdown("#### 🏷️ Classification — Performance Tier")
        st.write(
            "Translate expected video performance into three interpretable "
            "business-oriented tiers: **LOW**, **MEDIUM** and **HIGH**."
        )

        st.caption(
            "Output: performance tier and, when supported, class probabilities."
        )

    # =========================================================================
    # 2. LEAKAGE-SAFE DESIGN
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🛡️ Leakage-Safe Feature Design")

    st.info(
        "The prediction pipeline deliberately excludes post-publication "
        "performance variables. This prevents the model from using information "
        "that would only become available after a video has already generated views."
    )

    allowed_features = [
        "duration_seconds",
        "publish_hour",
        "publish_month",
        "publish_day",
        "publish_week",
        "title_length",
        "title_word_count",
        "description_length",
        "description_word_count",
        "tag_count",
        "category_name",
        "duration_category",
        "publish_day_name",
        "publish_session",
        "month_part",
        "caption_label",
        "is_weekend",
    ]

    excluded_features = [
        "view_count",
        "like_count",
        "comment_count",
        "favorite_count",
        "engagement_score",
        "like_rate",
        "comment_rate",
        "engagement_rate",
        "view_bucket",
        "popular_category",
        "channel_video_count",
        "other post-publication performance variables",
    ]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### ✅ Features Allowed")

        for feature in allowed_features:
            st.markdown(f"- `{feature}`")

    with c2:
        st.markdown("#### 🚫 Features Excluded")

        for feature in excluded_features:
            st.markdown(f"- `{feature}`")

    # =========================================================================
    # 3. ACTUAL FEATURE METADATA
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🧩 Features Actually Used by the Models")

    # Prefer regression metadata
    if reg_ok and reg_result:
        features = reg_result.get("features", {})

    # Otherwise use classification metadata
    elif clf_ok and clf_result:
        features = clf_result.get("features", {})

    # Fallback to dataset-derived features
    else:
        try:
            _, features = build_feature_frame(df_raw)
        except Exception:
            features = {
                "numeric": [],
                "categorical": [],
                "boolean": [],
            }

    numeric_features = features.get("numeric", [])
    categorical_features = features.get("categorical", [])
    boolean_features = features.get("boolean", [])

    total_features = (
        len(numeric_features)
        + len(categorical_features)
        + len(boolean_features)
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Total Features",
            f"{total_features}",
            "leakage-safe inputs",
        )

    with c2:
        metric_card(
            "Numeric",
            f"{len(numeric_features)}",
            "numeric inputs",
        )

    with c3:
        metric_card(
            "Categorical",
            f"{len(categorical_features)}",
            "encoded inputs",
        )

    with c4:
        metric_card(
            "Boolean",
            f"{len(boolean_features)}",
            "binary inputs",
        )

    # =========================================================================
    # 4. FEATURE GROUP DETAILS
    # =========================================================================
    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("#### 🔢 Numeric Features")

        if numeric_features:
            for feature in numeric_features:
                st.markdown(f"- `{feature}`")
        else:
            st.caption("No numeric features available.")

    with c2:

        st.markdown("#### 🗂️ Categorical Features")

        if categorical_features:
            for feature in categorical_features:
                st.markdown(f"- `{feature}`")
        else:
            st.caption("No categorical features available.")

    with c3:

        st.markdown("#### ☑️ Boolean Features")

        if boolean_features:
            for feature in boolean_features:
                st.markdown(f"- `{feature}`")
        else:
            st.caption("No boolean features available.")

    # =========================================================================
    # 5. PIPELINE STATUS
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🧪 ML Pipeline Status")

    c1, c2 = st.columns(2)

    with c1:

        if reg_ok and reg_result:

            best_reg = reg_result.get(
                "best_model_name",
                "Available",
            )

            fitted_count = len(
                reg_result.get(
                    "fitted_models",
                    {},
                )
            )

            st.success(
                f"🔮 **Regression Pipeline — READY**\n\n"
                f"Best model: **{best_reg}**\n\n"
                f"Trained models: **{fitted_count}**"
            )

        else:

            st.error(
                "🔮 **Regression Pipeline — UNAVAILABLE**"
            )

            if reg_result:

                error = reg_result.get(
                    "error",
                    "No detailed error available.",
                )

                st.caption(
                    f"Reason: {error}"
                )

                model_errors = reg_result.get(
                    "model_errors",
                    {},
                )

                if model_errors:

                    with st.expander(
                        "🔍 Regression training diagnostics"
                    ):

                        for model_name, error_text in model_errors.items():

                            st.markdown(
                                f"**{model_name}**"
                            )

                            st.code(
                                str(error_text)
                            )

    with c2:

        if clf_ok and clf_result:

            best_clf = clf_result.get(
                "best_model_name",
                "Available",
            )

            fitted_count = len(
                clf_result.get(
                    "fitted_models",
                    {},
                )
            )

            st.success(
                f"🏷️ **Classification Pipeline — READY**\n\n"
                f"Best model: **{best_clf}**\n\n"
                f"Trained models: **{fitted_count}**"
            )

        else:

            st.error(
                "🏷️ **Classification Pipeline — UNAVAILABLE**"
            )

            if clf_result:

                error = clf_result.get(
                    "error",
                    "No detailed error available.",
                )

                st.caption(
                    f"Reason: {error}"
                )

                model_errors = clf_result.get(
                    "model_errors",
                    {},
                )

                if model_errors:

                    with st.expander(
                        "🔍 Classification training diagnostics"
                    ):

                        for model_name, error_text in model_errors.items():

                            st.markdown(
                                f"**{model_name}**"
                            )

                            st.code(
                                str(error_text)
                            )

    # =========================================================================
    # 6. REGRESSION INTELLIGENCE
    # =========================================================================
    st.markdown("---")
    st.markdown("## 🔮 Regression Intelligence")

    if reg_ok and reg_result:

        leaderboard = reg_result.get(
            "leaderboard",
            pd.DataFrame(),
        )

        if (
            isinstance(leaderboard, pd.DataFrame)
            and not leaderboard.empty
        ):

            best = leaderboard.iloc[0]

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                metric_card(
                    "Best Model",
                    str(best["Model"]),
                    "lowest test RMSE",
                )

            with c2:
                metric_card(
                    "R²",
                    f"{best['R2']:.3f}",
                    "explained variance",
                )

            with c3:
                metric_card(
                    "MAE",
                    format_number(best["MAE"]),
                    "mean absolute error",
                )

            with c4:
                metric_card(
                    "RMSE",
                    format_number(best["RMSE"]),
                    "root mean squared error",
                )

            st.markdown("#### 📋 Regression Model Comparison")

            display_cols = [
                c
                for c in [
                    "Rank",
                    "Model",
                    "R2",
                    "MAE",
                    "RMSE",
                    "Status",
                ]
                if c in leaderboard.columns
            ]

            display_df = leaderboard[
                display_cols
            ].copy()

            if "R2" in display_df.columns:
                display_df["R2"] = display_df["R2"].round(3)

            if "MAE" in display_df.columns:
                display_df["MAE"] = display_df["MAE"].round(2)

            if "RMSE" in display_df.columns:
                display_df["RMSE"] = display_df["RMSE"].round(2)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Selection rule: lowest RMSE on the held-out test set, "
                "with higher R² used as the tie-breaker."
            )

            # -------------------------------------------------------------
            # Regression model errors
            # -------------------------------------------------------------
            model_errors = reg_result.get(
                "model_errors",
                {},
            )

            if model_errors:

                with st.expander(
                    "⚠️ Models that could not be trained"
                ):

                    for model_name, error_text in model_errors.items():

                        st.markdown(
                            f"**{model_name}**"
                        )

                        st.code(
                            str(error_text)
                        )

        else:

            st.warning(
                "Regression is ready, but no evaluation leaderboard "
                "is available."
            )

    else:

        st.warning(
            "Regression models are currently unavailable for this dataset."
        )

    # =========================================================================
    # 7. CLASSIFICATION INTELLIGENCE
    # =========================================================================
    st.markdown("---")
    st.markdown("## 🏷️ Classification Intelligence")

    if clf_ok and clf_result:

        leaderboard = clf_result.get(
            "leaderboard",
            pd.DataFrame(),
        )

        if (
            isinstance(leaderboard, pd.DataFrame)
            and not leaderboard.empty
        ):

            best = leaderboard.iloc[0]

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                metric_card(
                    "Best Model",
                    str(best["Model"]),
                    "highest weighted F1",
                )

            with c2:
                metric_card(
                    "Accuracy",
                    f"{best['Accuracy'] * 100:.1f}%",
                    "test-set accuracy",
                )

            with c3:
                metric_card(
                    "Precision",
                    f"{best['Precision'] * 100:.1f}%",
                    "weighted precision",
                )

            with c4:
                metric_card(
                    "Weighted F1",
                    f"{best['F1'] * 100:.1f}%",
                    "primary selection metric",
                )

            # -------------------------------------------------------------
            # Thresholds
            # -------------------------------------------------------------
            thresholds = clf_result.get(
                "thresholds"
            )

            if thresholds:

                st.markdown(
                    "#### 🎯 Performance Tier Thresholds"
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    metric_card(
                        "LOW",
                        f"< {format_number(thresholds['low_value'])}",
                        "below lower threshold",
                    )

                with c2:
                    metric_card(
                        "MEDIUM",
                        (
                            f"{format_number(thresholds['low_value'])} "
                            f"— "
                            f"{format_number(thresholds['high_value'])}"
                        ),
                        "middle performance range",
                    )

                with c3:
                    metric_card(
                        "HIGH",
                        f"≥ {format_number(thresholds['high_value'])}",
                        "above upper threshold",
                    )

                st.caption(
                    "Thresholds are calculated from the training split "
                    "using the configured quantiles."
                )

            # -------------------------------------------------------------
            # Classification leaderboard
            # -------------------------------------------------------------
            st.markdown(
                "#### 📋 Classification Model Comparison"
            )

            display_cols = [
                c
                for c in [
                    "Rank",
                    "Model",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "Status",
                ]
                if c in leaderboard.columns
            ]

            display_df = leaderboard[
                display_cols
            ].copy()

            for metric in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
            ]:

                if metric in display_df.columns:

                    display_df[metric] = (
                        display_df[metric] * 100
                    ).round(1)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Selection rule: highest weighted F1 on the held-out "
                "test set, with accuracy used as the tie-breaker."
            )

            # -------------------------------------------------------------
            # Imbalance warning
            # -------------------------------------------------------------
            imbalance_warning = clf_result.get(
                "imbalance_warning"
            )

            if imbalance_warning:

                st.warning(
                    f"⚠️ {imbalance_warning}"
                )

            # -------------------------------------------------------------
            # Failed models
            # -------------------------------------------------------------
            model_errors = clf_result.get(
                "model_errors",
                {},
            )

            if model_errors:

                with st.expander(
                    "⚠️ Models that could not be trained"
                ):

                    for model_name, error_text in model_errors.items():

                        st.markdown(
                            f"**{model_name}**"
                        )

                        st.code(
                            str(error_text)
                        )

        else:

            st.warning(
                "Classification is ready, but no evaluation leaderboard "
                "is available."
            )

    else:

        st.warning(
            "Classification models are currently unavailable for this dataset."
        )

    # =========================================================================
    # 8. TRAIN / TEST SUMMARY
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📚 Training & Testing Summary")

    train_samples = None
    test_samples = None

    if reg_ok and reg_result:

        train_samples = reg_result.get(
            "n_train"
        )

        test_samples = reg_result.get(
            "n_test"
        )

    elif clf_ok and clf_result:

        train_samples = clf_result.get(
            "n_train"
        )

        test_samples = clf_result.get(
            "n_test"
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Dataset Rows",
            f"{len(df_raw):,}",
            "after dataset cleaning",
        )

    with c2:
        metric_card(
            "Training Samples",
            f"{train_samples:,}"
            if train_samples is not None
            else "—",
            "80% holdout design"
            if train_samples is not None
            else "training unavailable",
        )

    with c3:
        metric_card(
            "Testing Samples",
            f"{test_samples:,}"
            if test_samples is not None
            else "—",
            "held-out evaluation"
            if test_samples is not None
            else "evaluation unavailable",
        )

    # =========================================================================
    # 9. END-TO-END WORKFLOW
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🔄 End-to-End ML Workflow")

    workflow = [
        (
            "01",
            "Historical Data",
            "Load validated YouTube video records."
        ),
        (
            "02",
            "Data Cleaning",
            "Remove duplicates, normalize types and handle missing values."
        ),
        (
            "03",
            "Feature Engineering",
            "Create leakage-safe pre-publication predictors."
        ),
        (
            "04",
            "Model Training",
            "Train multiple regression and classification algorithms."
        ),
        (
            "05",
            "Evaluation",
            "Compare models using held-out test-set metrics."
        ),
        (
            "06",
            "Model Selection",
            "Identify the strongest model using the defined metric."
        ),
        (
            "07",
            "Business Decision",
            "Translate model output into actionable content intelligence."
        ),
    ]

    row1 = st.columns(4)

    for column, item in zip(
        row1,
        workflow[:4]
    ):

        number, title, description = item

        with column:

            st.markdown(
                f"**STEP {number}**"
            )

            st.markdown(
                f"#### {title}"
            )

            st.caption(
                description
            )

    st.write("")

    row2 = st.columns(3)

    for column, item in zip(
        row2,
        workflow[4:]
    ):

        number, title, description = item

        with column:

            st.markdown(
                f"**STEP {number}**"
            )

            st.markdown(
                f"#### {title}"
            )

            st.caption(
                description
            )

    # =========================================================================
    # 10. MODEL SELECTION LOGIC
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🎛️ Model Selection Logic")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("#### 🔮 Regression")

        st.metric(
            "Primary Metric",
            "RMSE ↓"
        )

        st.metric(
            "Tie-Breaker",
            "R² ↑"
        )

        st.caption(
            "Lower RMSE means smaller prediction errors. "
            "R² provides additional context about explained variance."
        )

    with c2:

        st.markdown("#### 🏷️ Classification")

        st.metric(
            "Primary Metric",
            "Weighted F1 ↑"
        )

        st.metric(
            "Tie-Breaker",
            "Accuracy ↑"
        )

        st.caption(
            "Weighted F1 balances precision and recall while "
            "accounting for class frequency."
        )

    # =========================================================================
    # 11. MODEL GOVERNANCE
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🛡️ Model Governance")

    with st.expander(
        "Read limitations before interpreting predictions"
    ):

        st.markdown(
            f"""
            **Dataset size**

            The current dataset contains **{len(df_raw):,} videos**.
            This is a relatively small dataset for robust machine learning.
            Model results should therefore be treated as directional
            decision-support signals rather than production-grade guarantees.

            **Target leakage**

            Final views, likes, comments and engagement-derived variables
            are deliberately excluded from the prediction feature set.

            **Temporal limitations**

            YouTube performance changes over time because of audience
            behaviour, trends, recommendation systems, competition and
            platform dynamics.

            **Generalisation**

            A random train/test split evaluates performance on held-out
            observations, but it does not fully reproduce future-time
            forecasting conditions.

            **Model comparison**

            Multiple algorithms are evaluated instead of assuming that
            one algorithm is automatically superior.

            **Prediction interpretation**

            A model prediction is an estimate based on historical patterns.
            It is not a guarantee of future views or engagement.
            """
        )

    # =========================================================================
    # 12. BUSINESS INTERPRETATION
    # =========================================================================
    st.markdown("---")
    st.markdown("### 💼 Business Interpretation")

    st.write(
        "YouTube Intelligence connects historical analytics with machine "
        "learning to support content-planning decisions."
    )

    b1, b2, b3 = st.columns(3)

    with b1:
        st.markdown("#### 📊 Understand")
        st.caption(
            "Discover which categories, formats, publishing patterns "
            "and content characteristics are associated with performance."
        )

    with b2:
        st.markdown("#### 🔮 Estimate")
        st.caption(
            "Use leakage-safe features to estimate expected views "
            "before publication."
        )

    with b3:
        st.markdown("#### 🎯 Decide")
        st.caption(
            "Use performance tiers and model comparisons as "
            "decision-support signals."
        )

    st.success(
        "🧠 **Key principle:** The application prioritizes "
        "leakage prevention, transparent evaluation and honest model "
        "availability over unsupported or fabricated predictions."
    )


# =====================================================================================
# PAGE: VIEW PREDICTION
# Prediction Studio
#
# Streamlit-native UI
# No raw HTML
# No unsafe_allow_html
# Leakage-safe feature construction
# Explicit regression model selection
# Prediction + Historical Context
# Time-Series / Category Comparison Visualization
# =====================================================================================

import plotly.graph_objects as go


def page_prediction():

    # =========================================================================
    # PAGE HEADER
    # =========================================================================

    st.title("🔮 Prediction Studio")

    st.caption(
        "Estimate expected YouTube views using leakage-safe features available "
        "before or at publication time."
    )

    # =========================================================================
    # MODEL AVAILABILITY
    # =========================================================================

    if not reg_ok:

        st.error(
            "🔮 **Regression prediction is currently unavailable.**"
        )

        error_message = "Regression model unavailable."

        if isinstance(reg_result, dict):

            error_message = reg_result.get(
                "error",
                "All regression models failed during training."
            )

        st.warning(
            f"**Training status:** {error_message}"
        )

        model_errors = {}

        if isinstance(reg_result, dict):

            model_errors = (
                reg_result.get("model_errors")
                or reg_result.get("errors")
                or {}
            )

        if model_errors:

            st.markdown(
                "### 🔍 Regression Training Diagnostics"
            )

            for model_name, model_error in model_errors.items():

                with st.expander(
                    f"❌ {model_name}"
                ):

                    st.code(
                        str(model_error),
                        language="text"
                    )

        detailed_error = None

        if isinstance(reg_result, dict):

            detailed_error = (
                reg_result.get("detailed_error")
                or reg_result.get("traceback")
            )

        if detailed_error:

            with st.expander(
                "🛠️ Detailed Training Error"
            ):

                st.code(
                    str(detailed_error),
                    language="text"
                )

        st.markdown(
            "### 📊 Prediction Readiness"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Dataset Rows",
                f"{len(df_raw):,}"
            )

        with c2:

            target_status = (
                "Available"
                if "view_count" in df_raw.columns
                else "Missing"
            )

            st.metric(
                "View Target",
                target_status
            )

        with c3:

            model_count = 0

            if isinstance(reg_result, dict):

                model_count = len(
                    reg_result.get(
                        "fitted_models",
                        {}
                    )
                )

            st.metric(
                "Trained Models",
                model_count
            )

        with c4:

            st.metric(
                "Prediction",
                "NOT READY"
            )

        st.info(
            "The application intentionally does not generate a prediction "
            "until at least one regression model has successfully trained."
        )

        return

    # =========================================================================
    # VALIDATE REGRESSION RESULT
    # =========================================================================

    if not isinstance(reg_result, dict):

        st.error(
            "Regression result metadata is unavailable."
        )

        return

    feats = reg_result.get(
        "features",
        {}
    )

    df = df_raw.copy()

    model_dict = reg_result.get(
        "fitted_models",
        {}
    )

    if (
        not isinstance(model_dict, dict)
        or not model_dict
    ):

        st.error(
            "Regression pipeline is marked READY, but no fitted models "
            "are available."
        )

        return

    model_names = list(
        model_dict.keys()
    )

    # =========================================================================
    # BEST MODEL
    # =========================================================================

    best_model_name = reg_result.get(
        "best_model_name",
        model_names[0]
    )

    if best_model_name not in model_names:

        best_model_name = model_names[0]

    # =========================================================================
    # MODEL SELECTION
    # =========================================================================

    st.markdown(
        "### 🧠 Prediction Model"
    )

    selected_model_name = st.selectbox(
        "Select regression model",
        model_names,
        index=model_names.index(
            best_model_name
        ),
        key="prediction_regression_model",
        format_func=lambda name:
            f"⭐ {name} (Best)"
            if name == best_model_name
            else name,
        help=(
            "Choose which trained regression model should generate "
            "the prediction."
        ),
    )

    selected_pipe = model_dict[
        selected_model_name
    ]

    # =========================================================================
    # LEADERBOARD
    # =========================================================================

    leaderboard = reg_result.get(
        "leaderboard",
        pd.DataFrame()
    )

    selected_row = None

    if (
        isinstance(
            leaderboard,
            pd.DataFrame
        )
        and not leaderboard.empty
        and "Model" in leaderboard.columns
    ):

        matches = leaderboard[
            leaderboard["Model"]
            == selected_model_name
        ]

        if not matches.empty:

            selected_row = matches.iloc[0]

    # =========================================================================
    # MODEL METRICS
    # =========================================================================

    if selected_row is not None:

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Model",
                selected_model_name
            )

        with c2:

            r2_value = selected_row.get(
                "R2",
                np.nan
            )

            st.metric(
                "R²",
                (
                    f"{float(r2_value):.3f}"
                    if pd.notna(r2_value)
                    else "—"
                )
            )

        with c3:

            mae_value = selected_row.get(
                "MAE",
                np.nan
            )

            st.metric(
                "MAE",
                (
                    format_number(
                        float(mae_value)
                    )
                    if pd.notna(mae_value)
                    else "—"
                )
            )

        with c4:

            rmse_value = selected_row.get(
                "RMSE",
                np.nan
            )

            st.metric(
                "RMSE",
                (
                    format_number(
                        float(rmse_value)
                    )
                    if pd.notna(rmse_value)
                    else "—"
                )
            )

    # =========================================================================
    # MODEL STATUS
    # =========================================================================

    if selected_model_name == best_model_name:

        st.success(
            f"⭐ **{selected_model_name}** is the current best regression "
            "model according to the evaluation strategy."
        )

    else:

        st.info(
            f"Using **{selected_model_name}** for this prediction. "
            f"Current best model: **{best_model_name}**."
        )

    # =========================================================================
    # VIDEO INFORMATION
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "### 🎬 Video Information"
    )

    st.caption(
        "Enter only information that would realistically be known "
        "before or at publication time."
    )

    numeric_features = feats.get(
        "numeric",
        []
    )

    categorical_features = feats.get(
        "categorical",
        []
    )

    boolean_features = feats.get(
        "boolean",
        []
    )

    # =========================================================================
    # FORM
    # =========================================================================

    with st.form(
        "prediction_form",
        clear_on_submit=False
    ):

        c1, c2, c3 = st.columns(3)

        with c1:

            if "category_name" in categorical_features:

                categories = sorted(
                    df["category_name"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if categories:

                    category = st.selectbox(
                        "🗂️ Content Category",
                        categories,
                        key="prediction_category"
                    )

                else:

                    category = None

                    st.warning(
                        "No content categories are available."
                    )

            else:

                category = None

        with c2:

            duration_min = st.number_input(
                "⏱️ Duration — Minutes",
                min_value=0.0,
                max_value=240.0,
                value=8.0,
                step=0.5,
                key="prediction_duration_min"
            )

        with c3:

            duration_sec_extra = st.number_input(
                "⏱️ Additional Seconds",
                min_value=0,
                max_value=59,
                value=0,
                step=1,
                key="prediction_duration_sec"
            )

        c4, c5, c6 = st.columns(3)

        with c4:

            upload_date = st.date_input(
                "📅 Planned Upload Date",
                value=datetime.today().date(),
                key="prediction_upload_date"
            )

        with c5:

            upload_hour = st.slider(
                "🕒 Upload Hour",
                min_value=0,
                max_value=23,
                value=15,
                key="prediction_upload_hour"
            )

        with c6:

            if "caption_label" in categorical_features:

                caption_choice = st.selectbox(
                    "💬 Captions",
                    [
                        "Captioned",
                        "No Captions"
                    ],
                    key="prediction_caption"
                )

            else:

                caption_choice = None

        c7, c8 = st.columns(2)

        with c7:

            title_text = st.text_input(
                "📝 Video Title",
                value="How I Grew My Channel in 30 Days",
                key="prediction_title"
            )

        with c8:

            tag_input = st.text_input(
                "🏷️ Tags",
                value="youtube, growth, tips",
                key="prediction_tags",
                help="Enter tags separated by commas."
            )

        description_text = st.text_area(
            "📄 Video Description",
            value=(
                "In this video I break down exactly what worked, "
                "the strategies I used and the lessons learned."
            ),
            height=120,
            key="prediction_description"
        )

        submitted = st.form_submit_button(
            "🔴 PREDICT EXPECTED VIEWS",
            use_container_width=True
        )

    if not submitted:

        return

    # =========================================================================
    # VALIDATION
    # =========================================================================

    title_text = str(
        title_text
    ).strip()

    description_text = str(
        description_text
    ).strip()

    tag_input = str(
        tag_input
    ).strip()

    if not title_text:

        st.warning(
            "⚠️ Please enter a video title."
        )

        return

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================

    duration_seconds = (
        float(duration_min) * 60.0
        + float(duration_sec_extra)
    )

    tag_list = [
        tag.strip()
        for tag in tag_input.split(",")
        if tag.strip()
    ]

    day_name = upload_date.strftime(
        "%A"
    )

    is_weekend = (
        day_name
        in [
            "Saturday",
            "Sunday"
        ]
    )

    row = {}

    # =========================================================================
    # NUMERIC
    # =========================================================================

    if "duration_seconds" in numeric_features:

        row["duration_seconds"] = duration_seconds

    if "publish_hour" in numeric_features:

        row["publish_hour"] = int(
            upload_hour
        )

    if "publish_month" in numeric_features:

        row["publish_month"] = int(
            upload_date.month
        )

    if "publish_day" in numeric_features:

        row["publish_day"] = int(
            upload_date.day
        )

    if "publish_week" in numeric_features:

        row["publish_week"] = int(
            upload_date.isocalendar().week
        )

    if "publish_quarter" in numeric_features:

        row["publish_quarter"] = int(
            ((upload_date.month - 1) // 3) + 1
        )

    if "title_length" in numeric_features:

        row["title_length"] = len(
            title_text
        )

    if "title_word_count" in numeric_features:

        row["title_word_count"] = len(
            title_text.split()
        )

    if "description_length" in numeric_features:

        row["description_length"] = len(
            description_text
        )

    if "description_word_count" in numeric_features:

        row["description_word_count"] = len(
            description_text.split()
        )

    if "tag_count" in numeric_features:

        row["tag_count"] = len(
            tag_list
        )

    # =========================================================================
    # BOOLEAN
    # =========================================================================

    if "is_weekend" in boolean_features:

        row["is_weekend"] = int(
            is_weekend
        )

    # =========================================================================
    # CATEGORICAL
    # =========================================================================

    if "category_name" in categorical_features:

        row["category_name"] = category

    if "duration_category" in categorical_features:

        row["duration_category"] = (
            derive_duration_category(
                duration_seconds
            )
        )

    if "publish_day_name" in categorical_features:

        row["publish_day_name"] = day_name

    if "publish_session" in categorical_features:

        row["publish_session"] = (
            derive_publish_session(
                upload_hour
            )
        )

    if "month_part" in categorical_features:

        row["month_part"] = (
            derive_month_part(
                upload_date.day
            )
        )

    if "caption_label" in categorical_features:

        row["caption_label"] = caption_choice

    # =========================================================================
    # INPUT DATAFRAME
    # =========================================================================

    input_row = pd.DataFrame(
        [row]
    )

    expected_features = (
        numeric_features
        + categorical_features
        + boolean_features
    )

    missing_input_features = [
        feature
        for feature in expected_features
        if feature not in input_row.columns
    ]

    if missing_input_features:

        st.error(
            "❌ Prediction input is missing required features."
        )

        with st.expander(
            "🔍 Missing Feature Diagnostics"
        ):

            st.write(
                missing_input_features
            )

        return

    input_row = input_row[
        expected_features
    ]

    # =========================================================================
    # REGRESSION PREDICTION
    # =========================================================================

    try:

        raw_prediction = selected_pipe.predict(
            input_row
        )[0]

        target_transform = str(
            reg_result.get(
                "target_transform",
                ""
            )
        ).lower()

        if "log1p" in target_transform:

            predicted_views = float(
                np.expm1(
                    raw_prediction
                )
            )

        else:

            predicted_views = float(
                raw_prediction
            )

        if not np.isfinite(
            predicted_views
        ):

            raise ValueError(
                "Model returned a non-finite prediction."
            )

        predicted_views = max(
            0.0,
            predicted_views
        )

    except Exception as exc:

        st.error(
            "❌ Prediction failed."
        )

        with st.expander(
            "🔍 Technical Prediction Error"
        ):

            st.code(
                str(exc),
                language="text"
            )

        return

    # =========================================================================
    # CLASSIFICATION
    # =========================================================================

    predicted_class = None
    class_probs = None
    classification_model_name = None

    if (
        clf_ok
        and isinstance(
            clf_result,
            dict
        )
    ):

        clf_models = clf_result.get(
            "fitted_models",
            {}
        )

        clf_best_name = clf_result.get(
            "best_model_name"
        )

        if (
            isinstance(
                clf_models,
                dict
            )
            and clf_models
            and clf_best_name in clf_models
        ):

            clf_pipe = clf_models[
                clf_best_name
            ]

            try:

                predicted_class = str(
                    clf_pipe.predict(
                        input_row
                    )[0]
                )

                classification_model_name = (
                    clf_best_name
                )

                if hasattr(
                    clf_pipe,
                    "predict_proba"
                ):

                    probabilities = (
                        clf_pipe.predict_proba(
                            input_row
                        )[0]
                    )

                    classes = getattr(
                        clf_pipe,
                        "classes_",
                        []
                    )

                    class_probs = {
                        str(label): float(probability)
                        for label, probability
                        in zip(
                            classes,
                            probabilities
                        )
                    }

            except Exception:

                predicted_class = None
                class_probs = None

    # =========================================================================
    # RESULT
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "## 🎯 Prediction Result"
    )

    with st.container(
        border=True
    ):

        st.markdown(
            "#### EXPECTED VIDEO VIEWS"
        )

        st.metric(
            "Predicted Views",
            format_number(
                predicted_views
            )
        )

        st.caption(
            f"Approximately **{predicted_views:,.0f} views**"
        )

        if predicted_class:

            if predicted_class == "HIGH":

                st.success(
                    "🟢 HIGH"
                )

            elif predicted_class == "MEDIUM":

                st.warning(
                    "🟡 MEDIUM"
                )

            elif predicted_class == "LOW":

                st.error(
                    "🔴 LOW"
                )

            else:

                st.info(
                    f"🏷️ {predicted_class}"
                )

        st.caption(
            f"Model: **{selected_model_name}**"
        )

    st.info(
        f"📌 **{selected_model_name}** estimates approximately "
        f"**{format_number(predicted_views)} views** from the "
        "pre-publication attributes you supplied."
    )

    st.caption(
        "This is a statistical estimate based on historical training data. "
        "It is not a guarantee of future YouTube performance."
    )

    # =========================================================================
    # HISTORICAL CONTEXT
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "### 📊 Historical Context"
    )

    historical_views = pd.Series(
        dtype="float64"
    )

    if "view_count" in df.columns:

        historical_views = pd.to_numeric(
            df["view_count"],
            errors="coerce"
        ).dropna()

        historical_views = historical_views[
            historical_views >= 0
        ]

    if not historical_views.empty:

        median_views = float(
            historical_views.median()
        )

        p25 = float(
            historical_views.quantile(
                0.25
            )
        )

        p75 = float(
            historical_views.quantile(
                0.75
            )
        )

        percentile = float(
            (
                historical_views
                <= predicted_views
            ).mean()
            * 100
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Historical Median",
                format_number(
                    median_views
                )
            )

        with c2:

            st.metric(
                "Prediction Percentile",
                f"{percentile:.1f}%"
            )

        with c3:

            st.metric(
                "Historical IQR",
                (
                    f"{format_number(p25)} – "
                    f"{format_number(p75)}"
                )
            )

        if predicted_views >= p75:

            st.success(
                "📈 Prediction is above the historical 75th percentile."
            )

        elif predicted_views >= median_views:

            st.info(
                "📊 Prediction is above or around the historical median."
            )

        elif predicted_views >= p25:

            st.info(
                "📊 Prediction falls inside the historical middle range."
            )

        else:

            st.warning(
                "📉 Prediction is below the historical 25th percentile."
            )

    else:

        st.info(
            "Historical view statistics are unavailable."
        )

    # =========================================================================
    # NEW VISUALIZATION
    # CATEGORY COMPARISON
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "### 📊 Prediction vs Category Performance"
    )

    st.caption(
        "Compare the predicted views with the historical average views "
        "of each content category."
    )

    if (
        "category_name" in df.columns
        and "view_count" in df.columns
    ):

        category_plot_df = df[
            [
                "category_name",
                "view_count"
            ]
        ].copy()

        category_plot_df["view_count"] = pd.to_numeric(
            category_plot_df["view_count"],
            errors="coerce"
        )

        category_plot_df = category_plot_df.dropna(
            subset=[
                "category_name",
                "view_count"
            ]
        )

        if not category_plot_df.empty:

            category_summary = (
                category_plot_df
                .groupby(
                    "category_name",
                    as_index=False
                )["view_count"]
                .mean()
                .sort_values(
                    "view_count",
                    ascending=False
                )
            )

            category_summary["view_count"] = (
                category_summary["view_count"]
                .clip(lower=0)
            )

            # Keep all categories but limit visual clutter

            category_summary = category_summary.head(
                10
            )

            # -------------------------------------------------------------
            # PLOT
            # -------------------------------------------------------------

            fig_category = go.Figure()

            fig_category.add_trace(
                go.Bar(
                    x=category_summary["category_name"],
                    y=category_summary["view_count"],
                    name="Historical Category Average",
                    opacity=0.82,
                    marker=dict(
                        line=dict(
                            width=1
                        )
                    ),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Average Views: %{y:,.0f}"
                        "<extra></extra>"
                    )
                )
            )

            # Prediction reference line

            fig_category.add_hline(
                y=predicted_views,
                line_width=3,
                line_dash="dash",
                annotation_text=(
                    f"Prediction: "
                    f"{format_number(predicted_views)}"
                ),
                annotation_position="top right"
            )

            # -------------------------------------------------------------
            # TRANSPARENT BACKGROUND
            # -------------------------------------------------------------

            fig_category.update_layout(
                height=460,
                autosize=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(
                    l=70,
                    r=70,
                    t=70,
                    b=100
                ),
                showlegend=False,
                hovermode="x unified"
            )

            fig_category.update_xaxes(
                title_text="Content Category",
                showgrid=False,
                showline=True,
                zeroline=False,
                automargin=True,
                tickangle=-30
            )

            fig_category.update_yaxes(
                title_text="Average Views",
                showgrid=True,
                showline=True,
                zeroline=False,
                automargin=True,
                tickformat="~s"
            )

            st.plotly_chart(
                fig_category,
                use_container_width=True,
                config={
                    "responsive": True,
                    "displaylogo": False,
                    "displayModeBar": True
                },
                key="prediction_category_comparison"
            )

            st.caption(
                "Bars represent historical average views by category. "
                "The dashed horizontal line represents the current model prediction."
            )

        else:

            st.info(
                "Category comparison cannot be generated from the available data."
            )

    else:

        st.info(
            "Category comparison requires both `category_name` "
            "and `view_count` columns."
        )

    # =========================================================================
    # MODEL SNAPSHOT
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "### 🧠 Model Snapshot"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Model Used",
            selected_model_name
        )

    with c2:

        if selected_row is not None:

            r2_value = selected_row.get(
                "R2",
                np.nan
            )

            st.metric(
                "R²",
                (
                    f"{float(r2_value):.3f}"
                    if pd.notna(r2_value)
                    else "—"
                )
            )

        else:

            st.metric(
                "R²",
                "—"
            )

    with c3:

        if selected_row is not None:

            mae_value = selected_row.get(
                "MAE",
                np.nan
            )

            st.metric(
                "MAE",
                (
                    format_number(
                        float(mae_value)
                    )
                    if pd.notna(mae_value)
                    else "—"
                )
            )

        else:

            st.metric(
                "MAE",
                "—"
            )

    with c4:

        if selected_row is not None:

            rmse_value = selected_row.get(
                "RMSE",
                np.nan
            )

            st.metric(
                "RMSE",
                (
                    format_number(
                        float(rmse_value)
                    )
                    if pd.notna(rmse_value)
                    else "—"
                )
            )

        else:

            st.metric(
                "RMSE",
                "—"
            )

    # =========================================================================
    # PERFORMANCE TIER
    # =========================================================================

    if predicted_class:

        st.markdown("---")

        st.markdown(
            "### 🏷️ Performance Tier"
        )

        c1, c2 = st.columns(2)

        with c1:

            with st.container(
                border=True
            ):

                st.markdown(
                    "#### PREDICTED PERFORMANCE"
                )

                if predicted_class == "HIGH":

                    st.success(
                        "🟢 HIGH"
                    )

                elif predicted_class == "MEDIUM":

                    st.warning(
                        "🟡 MEDIUM"
                    )

                elif predicted_class == "LOW":

                    st.error(
                        "🔴 LOW"
                    )

                else:

                    st.info(
                        predicted_class
                    )

                if classification_model_name:

                    st.caption(
                        f"Classification model: "
                        f"**{classification_model_name}**"
                    )

        with c2:

            thresholds = {}

            if isinstance(
                clf_result,
                dict
            ):

                thresholds = (
                    clf_result.get(
                        "thresholds",
                        {}
                    )
                    or {}
                )

            with st.container(
                border=True
            ):

                st.markdown(
                    "#### PERFORMANCE THRESHOLDS"
                )

                if thresholds:

                    low_value = thresholds.get(
                        "low_value"
                    )

                    high_value = thresholds.get(
                        "high_value"
                    )

                    if (
                        low_value is not None
                        and high_value is not None
                    ):

                        st.write(
                            f"**LOW:** < "
                            f"{format_number(low_value)}"
                        )

                        st.write(
                            f"**MEDIUM:** "
                            f"{format_number(low_value)} "
                            f"to "
                            f"{format_number(high_value)}"
                        )

                        st.write(
                            f"**HIGH:** ≥ "
                            f"{format_number(high_value)}"
                        )

                    else:

                        st.caption(
                            "Threshold metadata is unavailable."
                        )

                else:

                    st.caption(
                        "Threshold metadata is unavailable."
                    )

    # =========================================================================
    # PERFORMANCE PROBABILITY
    # =========================================================================

    if class_probs:

        st.markdown("---")

        st.markdown(
            "### 📈 Performance Tier Probability"
        )

        st.caption(
            f"Classification model **{classification_model_name}** "
            "estimates the probability of each performance tier."
        )

        ordered_classes = [
            "LOW",
            "MEDIUM",
            "HIGH"
        ]

        probability_rows = []

        for class_name in ordered_classes:

            if class_name in class_probs:

                probability_rows.append(
                    {
                        "Class": class_name,
                        "Probability": float(
                            class_probs[
                                class_name
                            ]
                        )
                    }
                )

        if probability_rows:

            probs_df = pd.DataFrame(
                probability_rows
            )

            fig_probability = go.Figure()

            for _, record in probs_df.iterrows():

                class_name = record[
                    "Class"
                ]

                probability = record[
                    "Probability"
                ]

                if class_name == "LOW":

                    bar_color = THEME.get(
                        "primary",
                        "#FF0033"
                    )

                elif class_name == "MEDIUM":

                    bar_color = THEME.get(
                        "warn",
                        "#FFB000"
                    )

                else:

                    bar_color = THEME.get(
                        "success",
                        "#00C853"
                    )

                fig_probability.add_trace(
                    go.Bar(
                        x=[
                            probability
                        ],
                        y=[
                            class_name
                        ],
                        orientation="h",
                        name=class_name,
                        marker=dict(
                            color=bar_color,
                            opacity=0.88,
                            line=dict(
                                width=1
                            )
                        ),
                        text=[
                            f"{probability * 100:.1f}%"
                        ],
                        textposition="outside",
                        cliponaxis=False,
                        hovertemplate=(
                            f"<b>{class_name}</b><br>"
                            f"Probability: "
                            f"{probability:.1%}"
                            "<extra></extra>"
                        )
                    )
                )

            fig_probability.update_layout(
                height=360,
                autosize=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(
                    l=90,
                    r=90,
                    t=60,
                    b=70
                ),
                showlegend=False,
                bargap=0.35
            )

            fig_probability.update_xaxes(
                range=[0, 1.08],
                tickformat=".0%",
                title_text="Probability",
                showgrid=True,
                showline=True,
                zeroline=False,
                automargin=True
            )

            fig_probability.update_yaxes(
                title_text="Performance Tier",
                showgrid=False,
                showline=False,
                automargin=True,
                categoryorder="array",
                categoryarray=[
                    "LOW",
                    "MEDIUM",
                    "HIGH"
                ]
            )

            st.plotly_chart(
                fig_probability,
                use_container_width=True,
                config={
                    "responsive": True,
                    "displaylogo": False,
                    "displayModeBar": True
                },
                key="prediction_class_probability"
            )

    # =========================================================================
    # BUSINESS RECOMMENDATION
    # =========================================================================

    if predicted_class:

        st.markdown("---")

        st.markdown(
            "### 💼 Business Recommendation"
        )

        try:

            recommendation = (
                recommendation_for_class(
                    predicted_class
                )
            )

            st.success(
                recommendation
            )

        except Exception:

            st.info(
                f"The prediction indicates a "
                f"**{predicted_class}** performance tier. "
                "Use the result together with historical analytics "
                "when planning content."
            )

        st.caption(
            "This recommendation is based on historical model patterns "
            "and should be treated as decision support."
        )

    # =========================================================================
    # HOW TO READ
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "### 💡 How to Read This Prediction"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### 🔮 Expected Views"
            )

            st.caption(
                "The regression model estimates the expected number "
                "of views for the supplied video profile."
            )

    with c2:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### 📊 Category Comparison"
            )

            st.caption(
                "The comparison chart shows how the prediction "
                "relates to historical category-level performance."
            )

    with c3:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### 🏷️ Performance Tier"
            )

            st.caption(
                "The classifier converts the video profile into "
                "LOW, MEDIUM or HIGH."
            )

    # =========================================================================
    # INPUT SUMMARY
    # =========================================================================

    with st.expander(
        "🔍 View Prediction Input Summary"
    ):

        summary_df = pd.DataFrame(
            {
                "Feature": list(
                    input_row.columns
                ),
                "Value": [
                    input_row.iloc[0][
                        column
                    ]
                    for column in input_row.columns
                ]
            }
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

    # =========================================================================
    # SAVE HISTORY
    # =========================================================================

    if "prediction_history" not in st.session_state:

        st.session_state.prediction_history = []

    prediction_record = {
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "title": title_text,
        "category": (
            category
            if category
            else "—"
        ),
        "duration_seconds": round(
            duration_seconds,
            2
        ),
        "upload_date": str(
            upload_date
        ),
        "upload_hour": int(
            upload_hour
        ),
        "predicted_views": round(
            predicted_views
        ),
        "predicted_class": (
            predicted_class
            if predicted_class
            else "—"
        ),
        "model_used": selected_model_name
    }

    st.session_state.prediction_history.append(
        prediction_record
    )

    st.success(
        "✅ Prediction saved to Prediction History."
    )


# =====================================================================================
# PAGE: PERFORMANCE CLASSIFICATION
#
# YouTube Intelligence
# Streamlit-native UI
# NO raw HTML
# NO unsafe_allow_html
# Leakage-safe feature construction
# Explicit multi-model selection
# Native classification result
# Probability visualization
# Business recommendation
# Input summary
# =====================================================================================

import plotly.graph_objects as go


def page_classification():

    # =========================================================================
    # PAGE HEADER
    # =========================================================================

    st.title("🏷️ Performance Classification")

    st.caption(
        "Classify planned YouTube content into LOW, MEDIUM or HIGH "
        "performance using leakage-safe pre-publication features."
    )

    # =========================================================================
    # MODEL AVAILABILITY
    # =========================================================================

    if not clf_ok:

        st.error(
            "🏷️ **Classification prediction is currently unavailable.**"
        )

        error_message = (
            "Classification model unavailable."
        )

        if isinstance(
            clf_result,
            dict
        ):

            error_message = clf_result.get(
                "error",
                "All classification models failed during training."
            )

        st.warning(
            f"**Training status:** {error_message}"
        )

        # ---------------------------------------------------------------------
        # MODEL ERRORS
        # ---------------------------------------------------------------------

        model_errors = {}

        if isinstance(
            clf_result,
            dict
        ):

            model_errors = (
                clf_result.get(
                    "model_errors"
                )
                or clf_result.get(
                    "errors"
                )
                or {}
            )

        if model_errors:

            st.markdown(
                "### 🔍 Classification Training Diagnostics"
            )

            for model_name, model_error in model_errors.items():

                with st.expander(
                    f"❌ {model_name}"
                ):

                    st.code(
                        str(model_error),
                        language="text"
                    )

        # ---------------------------------------------------------------------
        # DETAILED ERROR
        # ---------------------------------------------------------------------

        detailed_error = None

        if isinstance(
            clf_result,
            dict
        ):

            detailed_error = (
                clf_result.get(
                    "detailed_error"
                )
                or clf_result.get(
                    "traceback"
                )
            )

        if detailed_error:

            with st.expander(
                "🛠️ Detailed Training Error"
            ):

                st.code(
                    str(detailed_error),
                    language="text"
                )

        # ---------------------------------------------------------------------
        # READINESS
        # ---------------------------------------------------------------------

        st.markdown(
            "### 📊 Classification Readiness"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Dataset Rows",
                f"{len(df_raw):,}"
            )

        with c2:

            target_status = (
                "Available"
                if "view_count" in df_raw.columns
                else "Missing"
            )

            st.metric(
                "View Target",
                target_status
            )

        with c3:

            model_count = 0

            if isinstance(
                clf_result,
                dict
            ):

                model_count = len(
                    clf_result.get(
                        "fitted_models",
                        {}
                    )
                )

            st.metric(
                "Trained Models",
                model_count
            )

        with c4:

            st.metric(
                "Classification",
                "NOT READY"
            )

        st.info(
            "The application intentionally does not generate a classification "
            "until at least one classification model has successfully trained."
        )

        return

    # =========================================================================
    # VALIDATE RESULT METADATA
    # =========================================================================

    if not isinstance(
        clf_result,
        dict
    ):

        st.error(
            "Classification result metadata is unavailable."
        )

        return

    # =========================================================================
    # DATA + FEATURES
    # =========================================================================

    df = df_raw.copy()

    feats = clf_result.get(
        "features",
        {}
    )

    numeric_features = feats.get(
        "numeric",
        []
    )

    categorical_features = feats.get(
        "categorical",
        []
    )

    boolean_features = feats.get(
        "boolean",
        []
    )

    # =========================================================================
    # PERFORMANCE TIER DEFINITION
    # =========================================================================

    thresholds = clf_result.get(
        "thresholds",
        {}
    ) or {}

    low_value = thresholds.get(
        "low_value"
    )

    high_value = thresholds.get(
        "high_value"
    )

    st.markdown(
        "### 🎯 Performance Tier Definition"
    )

    if (
        low_value is not None
        and high_value is not None
    ):

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "LOW",
                f"< {format_number(low_value)}",
                help="Lower historical performance range."
            )

        with c2:

            st.metric(
                "MEDIUM",
                (
                    f"{format_number(low_value)} – "
                    f"{format_number(high_value)}"
                ),
                help="Middle historical performance range."
            )

        with c3:

            st.metric(
                "HIGH",
                f"≥ {format_number(high_value)}",
                help="Higher historical performance range."
            )

        st.caption(
            "Thresholds are derived from the training data and are not "
            "calculated from the video currently being classified."
        )

    else:

        st.info(
            "Performance threshold metadata is unavailable."
        )

    # =========================================================================
    # CLASS IMBALANCE WARNING
    # =========================================================================

    if clf_result.get(
        "imbalance_warning"
    ):

        st.warning(
            f"⚠️ {clf_result['imbalance_warning']}"
        )

    # =========================================================================
    # MODEL SELECTION
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "### 🧠 Classification Model"
    )

    model_dict = clf_result.get(
        "fitted_models",
        {}
    )

    if (
        not isinstance(
            model_dict,
            dict
        )
        or not model_dict
    ):

        st.error(
            "Classification is marked READY, but no fitted models "
            "are available."
        )

        return

    model_names = list(
        model_dict.keys()
    )

    best_model_name = clf_result.get(
        "best_model_name",
        model_names[0]
    )

    if best_model_name not in model_names:

        best_model_name = model_names[0]

    selected_model_name = st.selectbox(
        "Select classification model",
        model_names,
        index=model_names.index(
            best_model_name
        ),
        key="classification_model_select",
        format_func=lambda name:
            f"⭐ {name} (Best)"
            if name == best_model_name
            else name,
        help=(
            "Choose which trained classification model should "
            "generate the performance tier."
        )
    )

    selected_pipe = model_dict[
        selected_model_name
    ]

    # =========================================================================
    # MODEL LEADERBOARD
    # =========================================================================

    leaderboard = clf_result.get(
        "leaderboard",
        pd.DataFrame()
    )

    selected_row = None

    if (
        isinstance(
            leaderboard,
            pd.DataFrame
        )
        and not leaderboard.empty
        and "Model" in leaderboard.columns
    ):

        matches = leaderboard[
            leaderboard["Model"]
            == selected_model_name
        ]

        if not matches.empty:

            selected_row = matches.iloc[0]

    # =========================================================================
    # MODEL METRICS
    # =========================================================================

    if selected_row is not None:

        st.markdown(
            "### 📊 Model Performance"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            accuracy_value = selected_row.get(
                "Accuracy",
                np.nan
            )

            st.metric(
                "Accuracy",
                (
                    f"{float(accuracy_value) * 100:.1f}%"
                    if pd.notna(accuracy_value)
                    else "—"
                )
            )

        with c2:

            precision_value = selected_row.get(
                "Precision",
                np.nan
            )

            st.metric(
                "Precision",
                (
                    f"{float(precision_value) * 100:.1f}%"
                    if pd.notna(precision_value)
                    else "—"
                )
            )

        with c3:

            recall_value = selected_row.get(
                "Recall",
                np.nan
            )

            st.metric(
                "Recall",
                (
                    f"{float(recall_value) * 100:.1f}%"
                    if pd.notna(recall_value)
                    else "—"
                )
            )

        with c4:

            f1_value = selected_row.get(
                "F1",
                np.nan
            )

            st.metric(
                "Weighted F1",
                (
                    f"{float(f1_value) * 100:.1f}%"
                    if pd.notna(f1_value)
                    else "—"
                )
            )

    # =========================================================================
    # MODEL STATUS
    # =========================================================================

    if selected_model_name == best_model_name:

        st.success(
            f"⭐ **{selected_model_name}** is the current best classification "
            "model according to the configured evaluation strategy."
        )

    else:

        st.info(
            f"Using **{selected_model_name}** for this classification. "
            f"Current best model: **{best_model_name}**."
        )

    # =========================================================================
    # VIDEO INFORMATION
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "### 🎬 Video Information"
    )

    st.caption(
        "Enter only information that would realistically be known "
        "before or at publication time."
    )

    # =========================================================================
    # FORM
    # =========================================================================

    with st.form(
        "classification_form",
        clear_on_submit=False
    ):

        # ---------------------------------------------------------------------
        # ROW 1
        # ---------------------------------------------------------------------

        c1, c2, c3 = st.columns(3)

        with c1:

            if "category_name" in categorical_features:

                categories = sorted(
                    df["category_name"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if categories:

                    category = st.selectbox(
                        "🗂️ Content Category",
                        categories,
                        key="classification_category"
                    )

                else:

                    category = None

                    st.warning(
                        "No content categories are available."
                    )

            else:

                category = None

        with c2:

            duration_min = st.number_input(
                "⏱️ Duration — Minutes",
                min_value=0.0,
                max_value=240.0,
                value=8.0,
                step=0.5,
                key="classification_duration"
            )

        with c3:

            duration_sec_extra = st.number_input(
                "⏱️ Additional Seconds",
                min_value=0,
                max_value=59,
                value=0,
                step=1,
                key="classification_duration_seconds"
            )

        # ---------------------------------------------------------------------
        # ROW 2
        # ---------------------------------------------------------------------

        c4, c5, c6 = st.columns(3)

        with c4:

            upload_date = st.date_input(
                "📅 Planned Upload Date",
                value=datetime.today().date(),
                key="classification_date"
            )

        with c5:

            upload_hour = st.slider(
                "🕒 Upload Hour",
                min_value=0,
                max_value=23,
                value=15,
                key="classification_hour"
            )

        with c6:

            if "caption_label" in categorical_features:

                caption_choice = st.selectbox(
                    "💬 Captions",
                    [
                        "Captioned",
                        "No Captions"
                    ],
                    key="classification_caption"
                )

            else:

                caption_choice = None

        # ---------------------------------------------------------------------
        # TITLE + TAGS
        # ---------------------------------------------------------------------

        c7, c8 = st.columns(2)

        with c7:

            title_text = st.text_input(
                "📝 Video Title",
                value=(
                    "5 Editing Tricks Every Creator Needs"
                ),
                key="classification_title"
            )

        with c8:

            tag_input = st.text_input(
                "🏷️ Tags",
                value="youtube, tips",
                key="classification_tags",
                help="Enter tags separated by commas."
            )

        # ---------------------------------------------------------------------
        # DESCRIPTION
        # ---------------------------------------------------------------------

        description_text = st.text_area(
            "📄 Video Description",
            value=(
                "Learn practical editing techniques that can "
                "improve your videos."
            ),
            height=120,
            key="classification_description"
        )

        # ---------------------------------------------------------------------
        # SUBMIT
        # ---------------------------------------------------------------------

        submitted = st.form_submit_button(
            "🔴 CLASSIFY PERFORMANCE",
            use_container_width=True
        )

    # =========================================================================
    # STOP IF NOT SUBMITTED
    # =========================================================================

    if not submitted:

        return

    # =========================================================================
    # VALIDATION
    # =========================================================================

    title_text = str(
        title_text
    ).strip()

    description_text = str(
        description_text
    ).strip()

    tag_input = str(
        tag_input
    ).strip()

    if not title_text:

        st.warning(
            "⚠️ Please enter a video title."
        )

        return

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================

    duration_seconds = (
        float(duration_min) * 60.0
        + float(duration_sec_extra)
    )

    tag_list = [
        tag.strip()
        for tag in tag_input.split(",")
        if tag.strip()
    ]

    day_name = upload_date.strftime(
        "%A"
    )

    is_weekend = (
        day_name
        in [
            "Saturday",
            "Sunday"
        ]
    )

    row = {}

    # =========================================================================
    # NUMERIC FEATURES
    # =========================================================================

    if "duration_seconds" in numeric_features:

        row["duration_seconds"] = (
            duration_seconds
        )

    if "publish_hour" in numeric_features:

        row["publish_hour"] = int(
            upload_hour
        )

    if "publish_month" in numeric_features:

        row["publish_month"] = int(
            upload_date.month
        )

    if "publish_day" in numeric_features:

        row["publish_day"] = int(
            upload_date.day
        )

    if "publish_week" in numeric_features:

        row["publish_week"] = int(
            upload_date.isocalendar().week
        )

    if "publish_quarter" in numeric_features:

        row["publish_quarter"] = int(
            ((upload_date.month - 1) // 3) + 1
        )

    if "title_length" in numeric_features:

        row["title_length"] = len(
            title_text
        )

    if "title_word_count" in numeric_features:

        row["title_word_count"] = len(
            title_text.split()
        )

    if "description_length" in numeric_features:

        row["description_length"] = len(
            description_text
        )

    if "description_word_count" in numeric_features:

        row["description_word_count"] = len(
            description_text.split()
        )

    if "tag_count" in numeric_features:

        row["tag_count"] = len(
            tag_list
        )

    # =========================================================================
    # BOOLEAN FEATURES
    # =========================================================================

    if "is_weekend" in boolean_features:

        row["is_weekend"] = int(
            is_weekend
        )

    # =========================================================================
    # CATEGORICAL FEATURES
    # =========================================================================

    if "category_name" in categorical_features:

        row["category_name"] = category

    if "duration_category" in categorical_features:

        row["duration_category"] = (
            derive_duration_category(
                duration_seconds
            )
        )

    if "publish_day_name" in categorical_features:

        row["publish_day_name"] = day_name

    if "publish_session" in categorical_features:

        row["publish_session"] = (
            derive_publish_session(
                upload_hour
            )
        )

    if "month_part" in categorical_features:

        row["month_part"] = (
            derive_month_part(
                upload_date.day
            )
        )

    if "caption_label" in categorical_features:

        row["caption_label"] = caption_choice

    # =========================================================================
    # INPUT DATAFRAME
    # =========================================================================

    input_row = pd.DataFrame(
        [row]
    )

    # =========================================================================
    # EXACT FEATURE ORDER
    # =========================================================================

    expected_features = (
        numeric_features
        + categorical_features
        + boolean_features
    )

    missing_input_features = [
        feature
        for feature in expected_features
        if feature not in input_row.columns
    ]

    if missing_input_features:

        st.error(
            "❌ Classification input is missing required features."
        )

        with st.expander(
            "🔍 Missing Feature Diagnostics"
        ):

            st.write(
                missing_input_features
            )

        return

    input_row = input_row[
        expected_features
    ]

    # =========================================================================
    # CLASSIFICATION PREDICTION
    # =========================================================================

    try:

        predicted_class = str(
            selected_pipe.predict(
                input_row
            )[0]
        )

    except Exception as exc:

        st.error(
            "❌ Classification failed."
        )

        with st.expander(
            "🔍 Technical Classification Error"
        ):

            st.code(
                str(exc),
                language="text"
            )

        return

    # =========================================================================
    # PROBABILITIES
    # =========================================================================

    probs = None

    if hasattr(
        selected_pipe,
        "predict_proba"
    ):

        try:

            probability_values = (
                selected_pipe.predict_proba(
                    input_row
                )[0]
            )

            classes = getattr(
                selected_pipe,
                "classes_",
                []
            )

            probs = {
                str(label): float(probability)
                for label, probability
                in zip(
                    classes,
                    probability_values
                )
            }

        except Exception:

            probs = None

    # =========================================================================
    # RESULT
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "## 🎯 Classification Result"
    )

    # -------------------------------------------------------------------------
    # NATIVE RESULT CARD
    # -------------------------------------------------------------------------

    with st.container(
        border=True
    ):

        st.markdown(
            "#### PREDICTED PERFORMANCE"
        )

        if predicted_class == "HIGH":

            st.success(
                "🟢 HIGH"
            )

        elif predicted_class == "MEDIUM":

            st.warning(
                "🟡 MEDIUM"
            )

        elif predicted_class == "LOW":

            st.error(
                "🔴 LOW"
            )

        else:

            st.info(
                f"🏷️ {predicted_class}"
            )

        st.caption(
            f"Classification model: **{selected_model_name}**"
        )

    # =========================================================================
    # INTERPRETATION
    # =========================================================================

    if predicted_class == "HIGH":

        st.success(
            "📈 The supplied video profile is classified as "
            "**HIGH performance** based on historical training patterns."
        )

    elif predicted_class == "MEDIUM":

        st.info(
            "📊 The supplied video profile is classified as "
            "**MEDIUM performance** based on historical training patterns."
        )

    elif predicted_class == "LOW":

        st.warning(
            "📉 The supplied video profile is classified as "
            "**LOW performance** based on historical training patterns."
        )

    # =========================================================================
    # PROBABILITY VISUALIZATION
    # =========================================================================

    if probs:

        st.markdown("---")

        st.markdown(
            "### 📈 Performance Tier Probability"
        )

        st.caption(
            f"**{selected_model_name}** estimates the probability "
            "of each performance tier."
        )

        ordered_classes = [
            "LOW",
            "MEDIUM",
            "HIGH"
        ]

        probability_rows = []

        for class_name in ordered_classes:

            if class_name in probs:

                probability_rows.append(
                    {
                        "Class": class_name,
                        "Probability": float(
                            probs[class_name]
                        )
                    }
                )

        if probability_rows:

            probs_df = pd.DataFrame(
                probability_rows
            )

            # -----------------------------------------------------------------
            # PLOTLY FIGURE
            # -----------------------------------------------------------------

            fig = go.Figure()

            for _, record in probs_df.iterrows():

                class_name = record[
                    "Class"
                ]

                probability = record[
                    "Probability"
                ]

                if class_name == "LOW":

                    bar_color = THEME.get(
                        "primary",
                        "#FF0033"
                    )

                elif class_name == "MEDIUM":

                    bar_color = THEME.get(
                        "warn",
                        "#FFB000"
                    )

                else:

                    bar_color = THEME.get(
                        "success",
                        "#00C853"
                    )

                fig.add_trace(
                    go.Bar(
                        x=[
                            probability
                        ],
                        y=[
                            class_name
                        ],
                        orientation="h",
                        marker=dict(
                            color=bar_color,
                            opacity=0.88,
                            line=dict(
                                width=1
                            )
                        ),
                        text=[
                            f"{probability * 100:.1f}%"
                        ],
                        textposition="outside",
                        cliponaxis=False,
                        hovertemplate=(
                            "<b>"
                            + class_name
                            + "</b><br>"
                            "Probability: "
                            + f"{probability:.1%}"
                            + "<extra></extra>"
                        )
                    )
                )

            # -----------------------------------------------------------------
            # TRANSPARENT BACKGROUND
            # -----------------------------------------------------------------

            fig.update_layout(
                height=360,
                autosize=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(
                    l=90,
                    r=90,
                    t=50,
                    b=70
                ),
                showlegend=False,
                bargap=0.35
            )

            # -----------------------------------------------------------------
            # X AXIS
            # -----------------------------------------------------------------

            fig.update_xaxes(
                range=[
                    0,
                    1.08
                ],
                tickformat=".0%",
                title_text="Probability",
                showgrid=True,
                showline=True,
                zeroline=False,
                automargin=True
            )

            # -----------------------------------------------------------------
            # Y AXIS
            # -----------------------------------------------------------------

            fig.update_yaxes(
                title_text="Performance Tier",
                showgrid=False,
                showline=True,
                zeroline=False,
                automargin=True,
                categoryorder="array",
                categoryarray=[
                    "LOW",
                    "MEDIUM",
                    "HIGH"
                ]
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "responsive": True,
                    "displaylogo": False,
                    "displayModeBar": True
                },
                key="classification_probability_chart"
            )

    else:

        st.info(
            "📊 Probability visualization is not available for "
            "the selected classification model."
        )

    # =========================================================================
    # MODEL SNAPSHOT
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "### 🧠 Model Snapshot"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Model Used",
            selected_model_name
        )

    with c2:

        if selected_row is not None:

            accuracy_value = selected_row.get(
                "Accuracy",
                np.nan
            )

            st.metric(
                "Accuracy",
                (
                    f"{float(accuracy_value) * 100:.1f}%"
                    if pd.notna(accuracy_value)
                    else "—"
                )
            )

        else:

            st.metric(
                "Accuracy",
                "—"
            )

    with c3:

        if selected_row is not None:

            precision_value = selected_row.get(
                "Precision",
                np.nan
            )

            st.metric(
                "Precision",
                (
                    f"{float(precision_value) * 100:.1f}%"
                    if pd.notna(precision_value)
                    else "—"
                )
            )

        else:

            st.metric(
                "Precision",
                "—"
            )

    with c4:

        if selected_row is not None:

            f1_value = selected_row.get(
                "F1",
                np.nan
            )

            st.metric(
                "Weighted F1",
                (
                    f"{float(f1_value) * 100:.1f}%"
                    if pd.notna(f1_value)
                    else "—"
                )
            )

        else:

            st.metric(
                "Weighted F1",
                "—"
            )

    # =========================================================================
    # PERFORMANCE THRESHOLDS
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "### 🎯 Classification Thresholds"
    )

    if (
        low_value is not None
        and high_value is not None
    ):

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "LOW",
                f"< {format_number(low_value)}"
            )

        with c2:

            st.metric(
                "MEDIUM",
                (
                    f"{format_number(low_value)} – "
                    f"{format_number(high_value)}"
                )
            )

        with c3:

            st.metric(
                "HIGH",
                f"≥ {format_number(high_value)}"
            )

    else:

        st.info(
            "Threshold values are not available."
        )

    # =========================================================================
    # BUSINESS RECOMMENDATION
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "### 💼 Business Recommendation"
    )

    try:

        recommendation = (
            recommendation_for_class(
                predicted_class
            )
        )

        st.success(
            recommendation
        )

    except Exception:

        if predicted_class == "HIGH":

            st.success(
                "This content profile historically resembles "
                "high-performing videos. Consider producing related "
                "content and reinforcing the same category, timing, "
                "content format and packaging strategy."
            )

        elif predicted_class == "MEDIUM":

            st.info(
                "This content profile shows moderate historical "
                "performance potential. Consider testing the topic, "
                "title, timing and packaging before scaling."
            )

        else:

            st.warning(
                "This content profile historically resembles lower-"
                "performing videos. Consider improving topic selection, "
                "packaging, timing and audience relevance."
            )

    st.caption(
        "This recommendation is based on historical model patterns "
        "and should be treated as decision support, not a guarantee."
    )

    # =========================================================================
    # HOW TO READ
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "### 💡 How to Read This Classification"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### 🟢 HIGH"
            )

            st.caption(
                "The model identifies the supplied content profile "
                "as resembling higher-performing historical videos."
            )

    with c2:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### 🟡 MEDIUM"
            )

            st.caption(
                "The model identifies the supplied content profile "
                "as resembling the middle performance range."
            )

    with c3:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### 🔴 LOW"
            )

            st.caption(
                "The model identifies the supplied content profile "
                "as resembling lower-performing historical videos."
            )

    # =========================================================================
    # INPUT SUMMARY
    # =========================================================================

    with st.expander(
        "🔍 View Classification Input Summary"
    ):

        summary_df = pd.DataFrame(
            {
                "Feature": list(
                    input_row.columns
                ),
                "Value": [
                    input_row.iloc[0][
                        column
                    ]
                    for column in input_row.columns
                ]
            }
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

    # =========================================================================
    # SAVE HISTORY
    # =========================================================================

    if "classification_history" not in st.session_state:

        st.session_state.classification_history = []

    classification_record = {
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "title": title_text,
        "category": (
            category
            if category
            else "—"
        ),
        "duration_seconds": round(
            duration_seconds,
            2
        ),
        "upload_date": str(
            upload_date
        ),
        "upload_hour": int(
            upload_hour
        ),
        "predicted_class": predicted_class,
        "model_used": selected_model_name
    }

    st.session_state.classification_history.append(
        classification_record
    )

    st.success(
        "✅ Classification saved to Classification History."
    )


# =====================================================================================
# PAGE: MODEL COMPARISON
# Enhanced visualization
# Streamlit-native
# Transparent multi-color Plotly charts
# Safe Rank handling
# =====================================================================================

def page_model_comparison():

    st.title("⚖️ Model Comparison")

    st.caption(
        "Compare trained regression and classification models using "
        "held-out test-set performance."
    )

    # =========================================================================
    # OVERVIEW
    # =========================================================================

    regression_count = (
        len(reg_result.get("fitted_models", {}))
        if reg_ok and isinstance(reg_result, dict)
        else 0
    )

    classification_count = (
        len(clf_result.get("fitted_models", {}))
        if clf_ok and isinstance(clf_result, dict)
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Regression Models",
            regression_count
        )

    with c2:
        st.metric(
            "Classification Models",
            classification_count
        )

    with c3:
        st.metric(
            "Dataset Rows",
            f"{len(df_raw):,}"
        )

    with c4:
        st.metric(
            "Total Models",
            regression_count + classification_count
        )

    # =========================================================================
    # REGRESSION MODELS
    # =========================================================================

    st.markdown("---")
    st.markdown("## 📉 Regression Models")

    st.caption(
        "Lower MAE and RMSE are better. Higher R² is better."
    )

    if reg_ok and isinstance(reg_result, dict):

        board_reg = reg_result.get(
            "leaderboard",
            pd.DataFrame()
        )

        if (
            isinstance(board_reg, pd.DataFrame)
            and not board_reg.empty
        ):

            board_reg = board_reg.copy()

            best_regression_model = reg_result.get(
                "best_model_name"
            )

            # =============================================================
            # SELECTION RULE
            # =============================================================

            selection_rule = reg_result.get(
                "selection_rule",
                "Lowest RMSE on the held-out test set."
            )

            st.info(
                f"🎯 **Selection rule:** {selection_rule}"
            )

            if best_regression_model:

                st.success(
                    f"⭐ **{best_regression_model}** is the current "
                    "best regression model."
                )

            # =============================================================
            # FIND BEST ROW
            # =============================================================

            best_row = None

            if (
                "Model" in board_reg.columns
                and best_regression_model is not None
            ):

                best_matches = board_reg[
                    board_reg["Model"]
                    == best_regression_model
                ]

                if not best_matches.empty:

                    best_row = best_matches.iloc[0]

            # =============================================================
            # BEST MODEL METRICS
            # =============================================================

            if best_row is not None:

                c1, c2, c3 = st.columns(3)

                with c1:

                    r2_value = best_row.get(
                        "R2",
                        np.nan
                    )

                    st.metric(
                        "Best R²",
                        (
                            f"{float(r2_value):.3f}"
                            if pd.notna(r2_value)
                            else "—"
                        )
                    )

                with c2:

                    mae_value = best_row.get(
                        "MAE",
                        np.nan
                    )

                    st.metric(
                        "Best MAE",
                        (
                            format_number(
                                float(mae_value)
                            )
                            if pd.notna(mae_value)
                            else "—"
                        )
                    )

                with c3:

                    rmse_value = best_row.get(
                        "RMSE",
                        np.nan
                    )

                    st.metric(
                        "Best RMSE",
                        (
                            format_number(
                                float(rmse_value)
                            )
                            if pd.notna(rmse_value)
                            else "—"
                        )
                    )

            # =============================================================
            # REGRESSION LEADERBOARD
            # =============================================================

            st.markdown(
                "### 📋 Regression Leaderboard"
            )

            display_reg = board_reg.copy()

            # -------------------------------------------------------------
            # SAFE RANK
            # -------------------------------------------------------------

            if "Rank" not in display_reg.columns:

                display_reg.insert(
                    0,
                    "Rank",
                    range(
                        1,
                        len(display_reg) + 1
                    )
                )

            # -------------------------------------------------------------
            # SAFE STATUS
            # -------------------------------------------------------------

            display_reg["Status"] = (
                display_reg["Model"]
                .apply(
                    lambda model:
                    "⭐ Best"
                    if model == best_regression_model
                    else "Available"
                )
            )

            # -------------------------------------------------------------
            # COLUMN ORDER
            # -------------------------------------------------------------

            preferred_columns = [
                "Rank",
                "Model",
                "Status",
                "MAE",
                "RMSE",
                "R2"
            ]

            ordered_columns = [
                column
                for column in preferred_columns
                if column in display_reg.columns
            ]

            remaining_columns = [
                column
                for column in display_reg.columns
                if column not in ordered_columns
            ]

            display_reg = display_reg[
                ordered_columns
                + remaining_columns
            ]

            # -------------------------------------------------------------
            # TABLE FORMATTING
            # -------------------------------------------------------------

            format_dict = {}

            if "MAE" in display_reg.columns:

                format_dict["MAE"] = "{:,.0f}"

            if "RMSE" in display_reg.columns:

                format_dict["RMSE"] = "{:,.0f}"

            if "R2" in display_reg.columns:

                format_dict["R2"] = "{:.3f}"

            try:

                st.dataframe(
                    display_reg.style.format(
                        format_dict
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception:

                st.dataframe(
                    display_reg,
                    use_container_width=True,
                    hide_index=True,
                )

            # =============================================================
            # REGRESSION RMSE
            # =============================================================

            st.markdown(
                "### 📊 RMSE Comparison"
            )

            if (
                "Model" in board_reg.columns
                and "RMSE" in board_reg.columns
            ):

                plot_reg_rmse = board_reg[
                    [
                        "Model",
                        "RMSE"
                    ]
                ].copy()

                plot_reg_rmse = plot_reg_rmse.dropna(
                    subset=["RMSE"]
                )

                plot_reg_rmse = plot_reg_rmse.sort_values(
                    "RMSE",
                    ascending=True
                )

                # Transparent multi-color bars
                fig_rmse = px.bar(
                    plot_reg_rmse,
                    x="Model",
                    y="RMSE",
                    text="RMSE",
                    color="Model",
                    color_discrete_sequence=[
                        "#FF0033",
                        "#3BB273",
                        "#4C78A8",
                        "#F2C14E",
                        "#9B59B6",
                        "#00A8E8",
                        "#F58518",
                        "#54A24B",
                    ],
                    template=THEME["plot_template"],
                    title="Test-Set RMSE by Regression Model",
                )

                fig_rmse.update_traces(
                    texttemplate="%{y:,.0f}",
                    textposition="outside",
                    cliponaxis=False,
                    opacity=0.68,
                    marker_line_width=1,
                )

                fig_rmse.update_layout(
                    height=430,
                    showlegend=False,
                    xaxis_title="Regression Model",
                    yaxis_title="RMSE",
                    hovermode="x unified",
                    margin=dict(
                        l=20,
                        r=20,
                        t=70,
                        b=90,
                    ),
                )

                fig_rmse.update_xaxes(
                    tickangle=-20
                )

                st.plotly_chart(
                    fig_rmse,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                    },
                    key="model_comparison_regression_rmse",
                )

                st.caption(
                    "Lower RMSE means the model makes smaller prediction "
                    "errors on the held-out test set."
                )

            # =============================================================
            # REGRESSION MAE
            # =============================================================

            st.markdown(
                "### 📏 MAE Comparison"
            )

            if (
                "Model" in board_reg.columns
                and "MAE" in board_reg.columns
            ):

                plot_reg_mae = board_reg[
                    [
                        "Model",
                        "MAE"
                    ]
                ].copy()

                plot_reg_mae = plot_reg_mae.dropna(
                    subset=["MAE"]
                )

                plot_reg_mae = plot_reg_mae.sort_values(
                    "MAE",
                    ascending=True
                )

                fig_mae = px.bar(
                    plot_reg_mae,
                    x="Model",
                    y="MAE",
                    text="MAE",
                    color="Model",
                    color_discrete_sequence=[
                        "#00A8E8",
                        "#3BB273",
                        "#FF0033",
                        "#F2C14E",
                        "#9B59B6",
                        "#F58518",
                        "#54A24B",
                        "#4C78A8",
                    ],
                    template=THEME["plot_template"],
                    title="Test-Set MAE by Regression Model",
                )

                fig_mae.update_traces(
                    texttemplate="%{y:,.0f}",
                    textposition="outside",
                    cliponaxis=False,
                    opacity=0.68,
                    marker_line_width=1,
                )

                fig_mae.update_layout(
                    height=430,
                    showlegend=False,
                    xaxis_title="Regression Model",
                    yaxis_title="MAE",
                    hovermode="x unified",
                    margin=dict(
                        l=20,
                        r=20,
                        t=70,
                        b=90,
                    ),
                )

                fig_mae.update_xaxes(
                    tickangle=-20
                )

                st.plotly_chart(
                    fig_mae,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                    },
                    key="model_comparison_regression_mae",
                )

                st.caption(
                    "Lower MAE means a lower average absolute prediction error."
                )

            # =============================================================
            # REGRESSION R²
            # =============================================================

            st.markdown(
                "### 📈 R² Comparison"
            )

            if (
                "Model" in board_reg.columns
                and "R2" in board_reg.columns
            ):

                plot_reg_r2 = board_reg[
                    [
                        "Model",
                        "R2"
                    ]
                ].copy()

                plot_reg_r2 = plot_reg_r2.dropna(
                    subset=["R2"]
                )

                plot_reg_r2 = plot_reg_r2.sort_values(
                    "R2",
                    ascending=False
                )

                fig_r2 = px.bar(
                    plot_reg_r2,
                    x="Model",
                    y="R2",
                    text="R2",
                    color="Model",
                    color_discrete_sequence=[
                        "#3BB273",
                        "#4C78A8",
                        "#FF0033",
                        "#F2C14E",
                        "#9B59B6",
                        "#00A8E8",
                        "#F58518",
                        "#54A24B",
                    ],
                    template=THEME["plot_template"],
                    title="Test-Set R² by Regression Model",
                )

                fig_r2.update_traces(
                    texttemplate="%{y:.3f}",
                    textposition="outside",
                    cliponaxis=False,
                    opacity=0.68,
                    marker_line_width=1,
                )

                fig_r2.update_layout(
                    height=430,
                    showlegend=False,
                    xaxis_title="Regression Model",
                    yaxis_title="R²",
                    hovermode="x unified",
                    margin=dict(
                        l=20,
                        r=20,
                        t=70,
                        b=90,
                    ),
                )

                fig_r2.update_xaxes(
                    tickangle=-20
                )

                st.plotly_chart(
                    fig_r2,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                    },
                    key="model_comparison_regression_r2",
                )

                st.caption(
                    "Higher R² indicates that the model explains more "
                    "variation in the held-out test data."
                )

        else:

            st.warning(
                "Regression leaderboard is unavailable."
            )

    else:

        error_message = (
            reg_result.get(
                "error",
                "Regression models unavailable."
            )
            if isinstance(reg_result, dict)
            else "Regression models unavailable."
        )

        st.warning(
            error_message
        )

    # =========================================================================
    # CLASSIFICATION MODELS
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "## 🏷️ Classification Models"
    )

    st.caption(
        "Higher Accuracy, Precision, Recall and Weighted F1 are better."
    )

    if clf_ok and isinstance(clf_result, dict):

        board_clf = clf_result.get(
            "leaderboard",
            pd.DataFrame()
        )

        if (
            isinstance(board_clf, pd.DataFrame)
            and not board_clf.empty
        ):

            board_clf = board_clf.copy()

            best_classification_model = clf_result.get(
                "best_model_name"
            )

            # =============================================================
            # SELECTION RULE
            # =============================================================

            selection_rule = clf_result.get(
                "selection_rule",
                "Highest weighted F1 on the held-out test set."
            )

            st.info(
                f"🎯 **Selection rule:** {selection_rule}"
            )

            if best_classification_model:

                st.success(
                    f"⭐ **{best_classification_model}** is the current "
                    "best classification model."
                )

            # =============================================================
            # BEST MODEL
            # =============================================================

            best_row = None

            if (
                "Model" in board_clf.columns
                and best_classification_model is not None
            ):

                best_matches = board_clf[
                    board_clf["Model"]
                    == best_classification_model
                ]

                if not best_matches.empty:

                    best_row = best_matches.iloc[0]

            if best_row is not None:

                c1, c2, c3, c4 = st.columns(4)

                with c1:

                    value = best_row.get(
                        "Accuracy",
                        np.nan
                    )

                    st.metric(
                        "Best Accuracy",
                        (
                            f"{float(value) * 100:.1f}%"
                            if pd.notna(value)
                            else "—"
                        )
                    )

                with c2:

                    value = best_row.get(
                        "Precision",
                        np.nan
                    )

                    st.metric(
                        "Best Precision",
                        (
                            f"{float(value) * 100:.1f}%"
                            if pd.notna(value)
                            else "—"
                        )
                    )

                with c3:

                    value = best_row.get(
                        "Recall",
                        np.nan
                    )

                    st.metric(
                        "Best Recall",
                        (
                            f"{float(value) * 100:.1f}%"
                            if pd.notna(value)
                            else "—"
                        )
                    )

                with c4:

                    value = best_row.get(
                        "F1",
                        np.nan
                    )

                    st.metric(
                        "Best Weighted F1",
                        (
                            f"{float(value) * 100:.1f}%"
                            if pd.notna(value)
                            else "—"
                        )
                    )

            # =============================================================
            # CLASSIFICATION LEADERBOARD
            # =============================================================

            st.markdown(
                "### 📋 Classification Leaderboard"
            )

            display_clf = board_clf.copy()

            # SAFE RANK
            if "Rank" not in display_clf.columns:

                display_clf.insert(
                    0,
                    "Rank",
                    range(
                        1,
                        len(display_clf) + 1
                    )
                )

            # STATUS
            display_clf["Status"] = (
                display_clf["Model"]
                .apply(
                    lambda model:
                    "⭐ Best"
                    if model == best_classification_model
                    else "Available"
                )
            )

            preferred_columns = [
                "Rank",
                "Model",
                "Status",
                "Accuracy",
                "Precision",
                "Recall",
                "F1"
            ]

            ordered_columns = [
                column
                for column in preferred_columns
                if column in display_clf.columns
            ]

            remaining_columns = [
                column
                for column in display_clf.columns
                if column not in ordered_columns
            ]

            display_clf = display_clf[
                ordered_columns
                + remaining_columns
            ]

            # TABLE FORMAT
            format_dict = {}

            for column in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1"
            ]:

                if column in display_clf.columns:

                    format_dict[column] = "{:.1%}"

            try:

                st.dataframe(
                    display_clf.style.format(
                        format_dict
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception:

                st.dataframe(
                    display_clf,
                    use_container_width=True,
                    hide_index=True,
                )

            # =============================================================
            # CLASSIFICATION GROUPED CHART
            # =============================================================

            st.markdown(
                "### 📊 Classification Model Performance"
            )

            metric_columns = [
                column
                for column in [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1"
                ]
                if column in board_clf.columns
            ]

            if (
                "Model" in board_clf.columns
                and metric_columns
            ):

                plot_clf = board_clf.melt(
                    id_vars=["Model"],
                    value_vars=metric_columns,
                    var_name="Metric",
                    value_name="Score"
                )

                plot_clf = plot_clf.dropna(
                    subset=["Score"]
                )

                fig_clf = px.bar(
                    plot_clf,
                    x="Model",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    text="Score",
                    color_discrete_sequence=[
                        "#FF0033",
                        "#3BB273",
                        "#4C78A8",
                        "#F2C14E",
                    ],
                    template=THEME["plot_template"],
                    title="Classification Models — Test-Set Metrics",
                )

                fig_clf.update_traces(
                    texttemplate="%{y:.1%}",
                    textposition="outside",
                    cliponaxis=False,
                    opacity=0.68,
                    marker_line_width=1,
                )

                fig_clf.update_layout(
                    height=470,
                    yaxis=dict(
                        range=[
                            0,
                            1
                        ],
                        tickformat=".0%",
                        title="Score",
                    ),
                    xaxis_title="Classification Model",
                    margin=dict(
                        l=20,
                        r=20,
                        t=70,
                        b=100,
                    ),
                    legend_title="Metric",
                    hovermode="x unified",
                )

                fig_clf.update_xaxes(
                    tickangle=-20
                )

                st.plotly_chart(
                    fig_clf,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                    },
                    key="model_comparison_classification_metrics",
                )

                st.caption(
                    "Higher scores indicate stronger classification "
                    "performance on the held-out test set."
                )

            # =============================================================
            # F1 RANKING
            # =============================================================

            st.markdown(
                "### 🏆 Weighted F1 Ranking"
            )

            if (
                "Model" in board_clf.columns
                and "F1" in board_clf.columns
            ):

                plot_f1 = board_clf[
                    [
                        "Model",
                        "F1"
                    ]
                ].copy()

                plot_f1 = plot_f1.dropna(
                    subset=["F1"]
                )

                plot_f1 = plot_f1.sort_values(
                    "F1",
                    ascending=True
                )

                fig_f1 = px.bar(
                    plot_f1,
                    x="F1",
                    y="Model",
                    orientation="h",
                    text="F1",
                    color="Model",
                    color_discrete_sequence=[
                        "#9B59B6",
                        "#00A8E8",
                        "#F58518",
                        "#3BB273",
                        "#FF0033",
                        "#4C78A8",
                    ],
                    template=THEME["plot_template"],
                    title="Weighted F1 — Higher is Better",
                )

                fig_f1.update_traces(
                    texttemplate="%{x:.1%}",
                    textposition="outside",
                    cliponaxis=False,
                    opacity=0.68,
                    marker_line_width=1,
                )

                fig_f1.update_layout(
                    height=max(
                        320,
                        90 * len(plot_f1)
                    ),
                    xaxis=dict(
                        range=[
                            0,
                            1
                        ],
                        tickformat=".0%",
                        title="Weighted F1",
                    ),
                    yaxis_title="Classification Model",
                    margin=dict(
                        l=30,
                        r=70,
                        t=70,
                        b=40,
                    ),
                    showlegend=False,
                )

                st.plotly_chart(
                    fig_f1,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                    },
                    key="model_comparison_classification_f1",
                )

                st.caption(
                    "Weighted F1 balances precision and recall while "
                    "accounting for class frequency."
                )

        else:

            st.warning(
                "Classification leaderboard is unavailable."
            )

    else:

        error_message = (
            clf_result.get(
                "error",
                "Classification models unavailable."
            )
            if isinstance(clf_result, dict)
            else "Classification models unavailable."
        )

        st.warning(
            error_message
        )


# =====================================================================================
# PAGE: MODEL EVALUATION
# =====================================================================================
def page_model_evaluation():

    st.title("📉 Model Evaluation")

    # =========================================================================
    # REGRESSION
    # =========================================================================

    st.subheader("Regression")

    if reg_ok:

        model_names = list(
            reg_result["fitted_models"].keys()
        )

        best_reg_model = reg_result["best_model_name"]

        eval_model_name = st.selectbox(
            "Select regression model to evaluate",
            model_names,
            index=model_names.index(
                best_reg_model
            ),
            format_func=lambda n:
                f"⭐ {n} (Best)"
                if n == best_reg_model
                else n,
            key="reg_eval_select",
        )

        eval_pipe = reg_result[
            "fitted_models"
        ][eval_model_name]

        # ---------------------------------------------------------------------
        # TEST DATA
        # ---------------------------------------------------------------------

        try:

            X_test = reg_result["X_test"]
            y_test = reg_result["y_test"]

            if isinstance(
                y_test,
                pd.Series
            ):

                y_test = y_test.values

            else:

                y_test = np.asarray(
                    y_test
                ).reshape(-1)

            preds = eval_pipe.predict(
                X_test
            )

            preds = np.asarray(
                preds,
                dtype=float
            ).reshape(-1)

            # -------------------------------------------------------------
            # HANDLE LOG1P TARGET
            # -------------------------------------------------------------

            target_transform = str(
                reg_result.get(
                    "target_transform",
                    ""
                )
            ).lower()

            if "log1p" in target_transform:

                preds = np.expm1(
                    np.clip(
                        preds,
                        -50,
                        50
                    )
                )

                if reg_result.get(
                    "y_test_is_transformed",
                    False
                ):

                    y_test = np.expm1(
                        np.clip(
                            y_test,
                            -50,
                            50
                        )
                    )

            # -------------------------------------------------------------
            # CLEAN VALUES
            # -------------------------------------------------------------

            min_len = min(
                len(y_test),
                len(preds)
            )

            y_test = y_test[
                :min_len
            ]

            preds = preds[
                :min_len
            ]

            valid = (
                np.isfinite(y_test)
                &
                np.isfinite(preds)
            )

            y_test = y_test[
                valid
            ]

            preds = preds[
                valid
            ]

            if len(y_test) == 0:

                raise ValueError(
                    "No valid test-set predictions are available."
                )

            # -------------------------------------------------------------
            # RESIDUALS
            # -------------------------------------------------------------

            residuals = (
                y_test - preds
            )

            # -------------------------------------------------------------
            # METRICS
            # -------------------------------------------------------------

            mae = float(
                np.mean(
                    np.abs(
                        residuals
                    )
                )
            )

            rmse = float(
                np.sqrt(
                    np.mean(
                        residuals ** 2
                    )
                )
            )

            ss_res = np.sum(
                residuals ** 2
            )

            ss_tot = np.sum(
                (
                    y_test
                    - y_test.mean()
                ) ** 2
            )

            r2 = (
                1 - ss_res / ss_tot
                if ss_tot > 0
                else np.nan
            )

            # =============================================================
            # KPIs
            # =============================================================

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "R²",
                    (
                        f"{r2:.3f}"
                        if np.isfinite(r2)
                        else "—"
                    )
                )

            with c2:

                st.metric(
                    "MAE",
                    format_number(
                        mae
                    )
                )

            with c3:

                st.metric(
                    "RMSE",
                    format_number(
                        rmse
                    )
                )

            with c4:

                st.metric(
                    "Test Samples",
                    f"{len(y_test):,}"
                )

            # =============================================================
            # ACTUAL VS PREDICTED
            # =============================================================

            c1, c2 = st.columns(2)

            with c1:

                fig = px.scatter(
                    x=y_test,
                    y=preds,
                    template=THEME[
                        "plot_template"
                    ],
                    labels={
                        "x": "Actual Views",
                        "y": "Predicted Views"
                    },
                    title="Actual vs Predicted"
                )

                max_value = max(
                    float(y_test.max()),
                    float(preds.max())
                )

                fig.add_trace(
                    go.Scatter(
                        x=[
                            0,
                            max_value
                        ],
                        y=[
                            0,
                            max_value
                        ],
                        mode="lines",
                        line=dict(
                            dash="dash"
                        ),
                        name="Perfect fit"
                    )
                )

                fig.update_traces(
                    opacity=0.65
                )

                fig.update_layout(
                    height=380,
                    margin=dict(
                        l=20,
                        r=20,
                        t=55,
                        b=30
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="reg_actual_predicted"
                )

            # =============================================================
            # RESIDUAL PLOT
            # =============================================================

            with c2:

                fig = px.scatter(
                    x=preds,
                    y=residuals,
                    template=THEME[
                        "plot_template"
                    ],
                    labels={
                        "x": "Predicted Views",
                        "y": "Residual"
                    },
                    title="Residual Plot"
                )

                fig.add_hline(
                    y=0,
                    line_dash="dash"
                )

                fig.update_traces(
                    opacity=0.65
                )

                fig.update_layout(
                    height=380,
                    margin=dict(
                        l=20,
                        r=20,
                        t=55,
                        b=30
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="reg_residual_plot"
                )

            # =============================================================
            # PREDICTION ERROR DISTRIBUTION
            # =============================================================

            fig = px.histogram(
                x=residuals,
                nbins=25,
                template=THEME[
                    "plot_template"
                ],
                labels={
                    "x":
                        "Error (Actual − Predicted)",
                    "y":
                        "Number of Videos"
                },
                title="Prediction Error Distribution"
            )

            fig.add_vline(
                x=0,
                line_dash="dash"
            )

            fig.update_traces(
                opacity=0.65
            )

            fig.update_layout(
                height=350,
                margin=dict(
                    l=20,
                    r=20,
                    t=55,
                    b=30
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="reg_error_distribution"
            )

            st.caption(
                f"Evaluation model: **{eval_model_name}**"
            )

        except Exception as exc:

            st.error(
                "❌ Unable to generate regression evaluation."
            )

            with st.expander(
                "🔍 Technical evaluation error"
            ):

                st.code(
                    str(exc),
                    language="text"
                )

    else:

        st.warning(
            "Regression model unavailable."
        )

    # =========================================================================
    # CLASSIFICATION
    # =========================================================================

    st.markdown("---")

    st.subheader("Classification")

    if clf_ok:

        model_names = list(
            clf_result[
                "fitted_models"
            ].keys()
        )

        best_clf_model = clf_result[
            "best_model_name"
        ]

        eval_clf_name = st.selectbox(
            "Select classification model to evaluate",
            model_names,
            index=model_names.index(
                best_clf_model
            ),
            format_func=lambda n:
                f"⭐ {n} (Best)"
                if n == best_clf_model
                else n,
            key="clf_eval_select",
        )

        eval_clf_pipe = clf_result[
            "fitted_models"
        ][eval_clf_name]

        labels = clf_result[
            "class_labels"
        ]

        # ---------------------------------------------------------------------
        # PREDICTIONS
        # ---------------------------------------------------------------------

        try:

            clf_preds = eval_clf_pipe.predict(
                clf_result["X_test"]
            )

            y_true = np.asarray(
                clf_result["y_test"]
            )

            clf_preds = np.asarray(
                clf_preds
            )

            # -------------------------------------------------------------
            # METRICS
            # -------------------------------------------------------------

            report = classification_report(
                y_true,
                clf_preds,
                labels=labels,
                output_dict=True,
                zero_division=0
            )

            accuracy = float(
                report.get(
                    "accuracy",
                    0
                )
            )

            weighted_precision = float(
                report.get(
                    "weighted avg",
                    {}
                ).get(
                    "precision",
                    0
                )
            )

            weighted_recall = float(
                report.get(
                    "weighted avg",
                    {}
                ).get(
                    "recall",
                    0
                )
            )

            weighted_f1 = float(
                report.get(
                    "weighted avg",
                    {}
                ).get(
                    "f1-score",
                    0
                )
            )

            # =============================================================
            # KPIs
            # =============================================================

            st.markdown(
                "### 📊 Evaluation Metrics"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Accuracy",
                    f"{accuracy:.1%}"
                )

            with c2:

                st.metric(
                    "Precision",
                    f"{weighted_precision:.1%}"
                )

            with c3:

                st.metric(
                    "Recall",
                    f"{weighted_recall:.1%}"
                )

            with c4:

                st.metric(
                    "Weighted F1",
                    f"{weighted_f1:.1%}"
                )

            # =============================================================
            # CONFUSION MATRIX
            # =============================================================

            st.markdown(
                "### 🧩 Confusion Matrix"
            )

            cm = confusion_matrix(
                y_true,
                clf_preds,
                labels=labels
            )

            fig_cm = px.imshow(
                cm,
                x=labels,
                y=labels,
                text_auto=True,
                template=THEME[
                    "plot_template"
                ],
                color_continuous_scale=[
                    "#F5F5F5",
                    "#FF0033"
                ],
                title="Confusion Matrix",
                labels={
                    "x": "Predicted",
                    "y": "Actual",
                    "color": "Count"
                }
            )

            fig_cm.update_layout(
                height=400,
                margin=dict(
                    l=20,
                    r=20,
                    t=55,
                    b=30
                )
            )

            st.plotly_chart(
                fig_cm,
                use_container_width=True,
                key="classification_confusion_matrix"
            )

            st.caption(
                "Diagonal cells represent correctly classified videos. "
                "Off-diagonal cells represent classification errors."
            )

            # =============================================================
            # CLASS-WISE PERFORMANCE
            # =============================================================

            st.markdown(
                "### 📈 Class-wise Performance"
            )

            class_rows = []

            for label in labels:

                class_info = report.get(
                    label,
                    {}
                )

                class_rows.append(
                    {
                        "Class": label,
                        "Precision": class_info.get(
                            "precision",
                            0
                        ),
                        "Recall": class_info.get(
                            "recall",
                            0
                        ),
                        "F1": class_info.get(
                            "f1-score",
                            0
                        )
                    }
                )

            class_df = pd.DataFrame(
                class_rows
            )

            plot_df = class_df.melt(
                id_vars="Class",
                value_vars=[
                    "Precision",
                    "Recall",
                    "F1"
                ],
                var_name="Metric",
                value_name="Score"
            )

            fig_class = px.bar(
                plot_df,
                x="Class",
                y="Score",
                color="Metric",
                barmode="group",
                text="Score",
                template=THEME[
                    "plot_template"
                ],
                title="Class-wise Performance"
            )

            fig_class.update_traces(
                texttemplate="%{y:.1%}",
                textposition="outside",
                opacity=0.70,
                cliponaxis=False
            )

            fig_class.update_layout(
                height=380,
                yaxis=dict(
                    range=[
                        0,
                        1
                    ],
                    tickformat=".0%",
                    title="Score"
                ),
                xaxis_title="Class",
                margin=dict(
                    l=20,
                    r=20,
                    t=55,
                    b=30
                )
            )

            st.plotly_chart(
                fig_class,
                use_container_width=True,
                key="classification_classwise"
            )

            # =============================================================
            # CLASSIFICATION REPORT
            # =============================================================

            st.markdown(
                "#### Classification Report"
            )

            report_df = (
                pd.DataFrame(
                    report
                )
                .transpose()
                .round(3)
            )

            st.dataframe(
                report_df,
                use_container_width=True
            )

            st.caption(
                f"Evaluation model: **{eval_clf_name}** · "
                f"Test samples: **{len(y_true)}**"
            )

        except Exception as exc:

            st.error(
                "❌ Unable to generate classification evaluation."
            )

            with st.expander(
                "🔍 Technical evaluation error"
            ):

                st.code(
                    str(exc),
                    language="text"
                )

    else:

        st.warning(
            "Classification model unavailable."
        )


# =====================================================================================
# PAGE: FEATURE IMPORTANCE
# =====================================================================================
def page_feature_importance():

    st.title("🌟 Feature Importance")

    st.caption(
        "Feature importance indicates model contribution, not causal impact."
    )

    # =========================================================================
    # REGRESSION
    # =========================================================================

    st.markdown("---")
    st.subheader("Regression")

    if reg_ok and isinstance(
        reg_result,
        dict
    ):

        model_names = list(
            reg_result.get(
                "fitted_models",
                {}
            ).keys()
        )

        if model_names:

            best_model = reg_result.get(
                "best_model_name",
                model_names[0]
            )

            if best_model not in model_names:
                best_model = model_names[0]

            fi_reg_name = st.selectbox(
                "Select regression model",
                model_names,
                index=model_names.index(
                    best_model
                ),
                format_func=lambda n:
                    f"⭐ {n} (Best)"
                    if n == best_model
                    else n,
                key="fi_reg_select",
            )

            try:

                imp_reg = get_feature_importance(
                    reg_result[
                        "fitted_models"
                    ][fi_reg_name]
                )

                if (
                    imp_reg is not None
                    and not imp_reg.empty
                ):

                    imp_reg = imp_reg.copy()

                    # ---------------------------------------------------------
                    # CLEAN IMPORTANCE DATA
                    # ---------------------------------------------------------

                    if (
                        "Feature" not in imp_reg.columns
                        or "Importance" not in imp_reg.columns
                    ):

                        st.info(
                            "Feature importance data is not available "
                            "for this model."
                        )

                    else:

                        imp_reg["Importance"] = pd.to_numeric(
                            imp_reg["Importance"],
                            errors="coerce"
                        )

                        imp_reg = imp_reg.dropna(
                            subset=[
                                "Importance"
                            ]
                        )

                        imp_reg = imp_reg[
                            imp_reg["Importance"] >= 0
                        ]

                        imp_reg = imp_reg.sort_values(
                            "Importance",
                            ascending=False
                        )

                        top_reg = imp_reg.head(
                            15
                        ).copy()

                        # -----------------------------------------------------
                        # KPI
                        # -----------------------------------------------------

                        c1, c2, c3 = st.columns(3)

                        with c1:

                            st.metric(
                                "Features",
                                f"{len(imp_reg):,}"
                            )

                        with c2:

                            st.metric(
                                "Top Feature",
                                str(
                                    top_reg.iloc[0]["Feature"]
                                )
                                if not top_reg.empty
                                else "—"
                            )

                        with c3:

                            st.metric(
                                "Top Importance",
                                (
                                    f"{float(top_reg.iloc[0]['Importance']):.3f}"
                                    if not top_reg.empty
                                    else "—"
                                )
                            )

                        # -----------------------------------------------------
                        # PLOT
                        # -----------------------------------------------------

                        fig_reg = px.bar(
                            top_reg.sort_values(
                                "Importance",
                                ascending=True
                            ),
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            text="Importance",
                            template=THEME[
                                "plot_template"
                            ],
                            title=(
                                "Top 15 Regression Features"
                            ),
                        )

                        fig_reg.update_traces(
                            texttemplate="%{x:.3f}",
                            textposition="outside",
                            cliponaxis=False,
                            opacity=0.70,
                        )

                        fig_reg.update_layout(
                            height=500,
                            showlegend=False,
                            xaxis_title="Importance",
                            yaxis_title="Feature",
                            margin=dict(
                                l=20,
                                r=70,
                                t=60,
                                b=30,
                            ),
                        )

                        st.plotly_chart(
                            fig_reg,
                            use_container_width=True,
                            config={
                                "displayModeBar": False
                            },
                            key="feature_importance_regression",
                        )

                        st.caption(
                            f"Showing the 15 most important features "
                            f"for **{fi_reg_name}**."
                        )

                else:

                    st.info(
                        "Not available for this model type."
                    )

            except Exception as exc:

                st.warning(
                    "Unable to calculate regression feature importance."
                )

                with st.expander(
                    "🔍 Technical details"
                ):

                    st.code(
                        str(exc),
                        language="text"
                    )

        else:

            st.info(
                "No regression models are available."
            )

    else:

        st.warning(
            "Regression feature importance unavailable."
        )

    # =========================================================================
    # CLASSIFICATION
    # =========================================================================

    st.markdown("---")
    st.subheader("Classification")

    if clf_ok and isinstance(
        clf_result,
        dict
    ):

        model_names = list(
            clf_result.get(
                "fitted_models",
                {}
            ).keys()
        )

        if model_names:

            best_model = clf_result.get(
                "best_model_name",
                model_names[0]
            )

            if best_model not in model_names:
                best_model = model_names[0]

            fi_clf_name = st.selectbox(
                "Select classification model",
                model_names,
                index=model_names.index(
                    best_model
                ),
                format_func=lambda n:
                    f"⭐ {n} (Best)"
                    if n == best_model
                    else n,
                key="fi_clf_select",
            )

            try:

                imp_clf = get_feature_importance(
                    clf_result[
                        "fitted_models"
                    ][fi_clf_name]
                )

                if (
                    imp_clf is not None
                    and not imp_clf.empty
                ):

                    imp_clf = imp_clf.copy()

                    # ---------------------------------------------------------
                    # CLEAN IMPORTANCE DATA
                    # ---------------------------------------------------------

                    if (
                        "Feature" not in imp_clf.columns
                        or "Importance" not in imp_clf.columns
                    ):

                        st.info(
                            "Feature importance data is not available "
                            "for this model."
                        )

                    else:

                        imp_clf["Importance"] = pd.to_numeric(
                            imp_clf["Importance"],
                            errors="coerce"
                        )

                        imp_clf = imp_clf.dropna(
                            subset=[
                                "Importance"
                            ]
                        )

                        imp_clf = imp_clf[
                            imp_clf["Importance"] >= 0
                        ]

                        imp_clf = imp_clf.sort_values(
                            "Importance",
                            ascending=False
                        )

                        top_clf = imp_clf.head(
                            15
                        ).copy()

                        # -----------------------------------------------------
                        # KPI
                        # -----------------------------------------------------

                        c1, c2, c3 = st.columns(3)

                        with c1:

                            st.metric(
                                "Features",
                                f"{len(imp_clf):,}"
                            )

                        with c2:

                            st.metric(
                                "Top Feature",
                                str(
                                    top_clf.iloc[0]["Feature"]
                                )
                                if not top_clf.empty
                                else "—"
                            )

                        with c3:

                            st.metric(
                                "Top Importance",
                                (
                                    f"{float(top_clf.iloc[0]['Importance']):.3f}"
                                    if not top_clf.empty
                                    else "—"
                                )
                            )

                        # -----------------------------------------------------
                        # PLOT
                        # -----------------------------------------------------

                        fig_clf = px.bar(
                            top_clf.sort_values(
                                "Importance",
                                ascending=True
                            ),
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            text="Importance",
                            template=THEME[
                                "plot_template"
                            ],
                            title=(
                                "Top 15 Classification Features"
                            ),
                        )

                        fig_clf.update_traces(
                            texttemplate="%{x:.3f}",
                            textposition="outside",
                            cliponaxis=False,
                            opacity=0.70,
                        )

                        fig_clf.update_layout(
                            height=500,
                            showlegend=False,
                            xaxis_title="Importance",
                            yaxis_title="Feature",
                            margin=dict(
                                l=20,
                                r=70,
                                t=60,
                                b=30,
                            ),
                        )

                        st.plotly_chart(
                            fig_clf,
                            use_container_width=True,
                            config={
                                "displayModeBar": False
                            },
                            key="feature_importance_classification",
                        )

                        st.caption(
                            f"Showing the 15 most important features "
                            f"for **{fi_clf_name}**."
                        )

                else:

                    st.info(
                        "Not available for this model type."
                    )

            except Exception as exc:

                st.warning(
                    "Unable to calculate classification feature importance."
                )

                with st.expander(
                    "🔍 Technical details"
                ):

                    st.code(
                        str(exc),
                        language="text"
                    )

        else:

            st.info(
                "No classification models are available."
            )

    else:

        st.warning(
            "Classification feature importance unavailable."
        )


# =====================================================================================
# PAGE: PREDICTION HISTORY
# =====================================================================================
def page_history():

    st.title("🕘 Prediction History")

    st.caption(
        "Review predictions generated during the current session."
    )

    # =========================================================================
    # HISTORY CHECK
    # =========================================================================

    hist = st.session_state.get(
        "prediction_history",
        []
    )

    if not hist:

        st.info(
            "No predictions made yet this session. "
            "Try the View Prediction or Performance Classification pages."
        )

        return

    # =========================================================================
    # DATAFRAME
    # =========================================================================

    hist_df = pd.DataFrame(
        hist
    ).copy()

    # =========================================================================
    # OVERVIEW KPIs
    # =========================================================================

    total_predictions = len(
        hist_df
    )

    # Predicted views
    if "predicted_views" in hist_df.columns:

        numeric_views = pd.to_numeric(
            hist_df["predicted_views"],
            errors="coerce"
        )

        valid_views = numeric_views.dropna()

    else:

        valid_views = pd.Series(
            dtype=float
        )

    # Prediction classes
    if "predicted_class" in hist_df.columns:

        valid_classes = (
            hist_df["predicted_class"]
            .dropna()
            .astype(str)
        )

        valid_classes = valid_classes[
            ~valid_classes.isin(
                ["—", "None", "nan"]
            )
        ]

    else:

        valid_classes = pd.Series(
            dtype=str
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Predictions",
            f"{total_predictions:,}"
        )

    with c2:

        if not valid_views.empty:

            st.metric(
                "Average Predicted Views",
                format_number(
                    valid_views.mean()
                )
            )

        else:

            st.metric(
                "Average Predicted Views",
                "—"
            )

    with c3:

        if not valid_views.empty:

            st.metric(
                "Highest Prediction",
                format_number(
                    valid_views.max()
                )
            )

        else:

            st.metric(
                "Highest Prediction",
                "—"
            )

    with c4:

        if not valid_classes.empty:

            st.metric(
                "Latest Tier",
                valid_classes.iloc[-1]
            )

        else:

            st.metric(
                "Latest Tier",
                "—"
            )

    # =========================================================================
    # PREDICTION VIEW
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "### 📋 Prediction Records"
    )

    # Display newest prediction first
    display_df = hist_df.iloc[
        ::-1
    ].reset_index(
        drop=True
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # =========================================================================
    # PREDICTED VIEWS COMPARISON
    # =========================================================================

    if (
        "predicted_views" in hist_df.columns
        and not valid_views.empty
    ):

        st.markdown("---")

        st.markdown(
            "### 📈 Predicted Views"
        )

        chart_df = hist_df.copy()

        chart_df["Predicted Views"] = pd.to_numeric(
            chart_df["predicted_views"],
            errors="coerce"
        )

        chart_df = chart_df.dropna(
            subset=[
                "Predicted Views"
            ]
        )

        chart_df["Prediction"] = range(
            1,
            len(chart_df) + 1
        )

        fig = px.bar(
            chart_df,
            x="Prediction",
            y="Predicted Views",
            template=THEME[
                "plot_template"
            ],
            text="Predicted Views",
            title="Predicted Views by Prediction",
            hover_data=[
                column
                for column in [
                    "title",
                    "category",
                    "predicted_class",
                    "model_used"
                ]
                if column in chart_df.columns
            ],
        )

        fig.update_traces(
            texttemplate="%{y:,.0f}",
            textposition="outside",
            cliponaxis=False,
            opacity=0.70,
        )

        fig.update_layout(
            height=380,
            showlegend=False,
            xaxis_title="Prediction",
            yaxis_title="Predicted Views",
            margin=dict(
                l=20,
                r=50,
                t=60,
                b=30,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            },
            key="prediction_history_views",
        )

    # =========================================================================
    # PERFORMANCE TIER DISTRIBUTION
    # =========================================================================

    if not valid_classes.empty:

        st.markdown("---")

        st.markdown(
            "### 🏷️ Performance Tier Distribution"
        )

        class_order = [
            "LOW",
            "MEDIUM",
            "HIGH"
        ]

        class_counts = (
            valid_classes
            .value_counts()
            .reindex(
                class_order,
                fill_value=0
            )
            .reset_index()
        )

        class_counts.columns = [
            "Class",
            "Predictions"
        ]

        fig_class = px.bar(
            class_counts,
            x="Class",
            y="Predictions",
            text="Predictions",
            template=THEME[
                "plot_template"
            ],
            title="Predicted Performance Tiers",
        )

        fig_class.update_traces(
            textposition="outside",
            cliponaxis=False,
            opacity=0.70,
        )

        fig_class.update_layout(
            height=350,
            showlegend=False,
            xaxis_title="Performance Tier",
            yaxis_title="Number of Predictions",
            margin=dict(
                l=20,
                r=30,
                t=60,
                b=30,
            ),
        )

        st.plotly_chart(
            fig_class,
            use_container_width=True,
            config={
                "displayModeBar": False
            },
            key="prediction_history_classes",
        )

    # =========================================================================
    # DOWNLOAD
    # =========================================================================

    st.markdown("---")

    csv_bytes = hist_df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        "⬇️ Download prediction history as CSV",
        data=csv_bytes,
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # =========================================================================
    # CLEAR HISTORY
    # =========================================================================

    if st.button(
        "🗑️ Clear history",
        use_container_width=True
    ):

        st.session_state.prediction_history = []

        st.rerun()


# =====================================================================================
# PAGE: BUSINESS RECOMMENDATIONS
# =====================================================================================
def page_recommendations():

    st.title("💡 Business Recommendations")

    st.caption(
        "Rule-based recommendations derived from actual computed patterns "
        "in the loaded dataset — not guaranteed outcomes."
    )

    # =========================================================================
    # DATA
    # =========================================================================

    df = df_raw.copy()

    # Required numeric columns
    if "view_count" in df.columns:

        df["view_count"] = pd.to_numeric(
            df["view_count"],
            errors="coerce"
        )

    # =========================================================================
    # CATEGORY PERFORMANCE
    # =========================================================================

    if (
        "category_name" in df.columns
        and "view_count" in df.columns
    ):

        st.markdown("---")
        st.markdown("### 🗂️ Category Performance")

        cat_perf = (
            df.dropna(
                subset=[
                    "category_name",
                    "view_count"
                ]
            )
            .groupby("category_name")[
                "view_count"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        if not cat_perf.empty:

            top_category = cat_perf.index[0]
            top_category_views = cat_perf.iloc[0]

            weakest_category = cat_perf.index[-1]
            weakest_category_views = cat_perf.iloc[-1]

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Top-performing Category",
                    str(top_category)
                )

            with c2:

                st.metric(
                    "Top Avg Views",
                    format_number(
                        top_category_views
                    )
                )

            with c3:

                st.metric(
                    "Weakest Category",
                    str(weakest_category)
                )

            # -----------------------------------------------------------------
            # CATEGORY CHART
            # -----------------------------------------------------------------

            category_plot = (
                cat_perf
                .reset_index()
            )

            category_plot.columns = [
                "Category",
                "Average Views"
            ]

            fig_category = px.bar(
                category_plot,
                x="Category",
                y="Average Views",
                text="Average Views",
                template=THEME[
                    "plot_template"
                ],
                title="Average Views by Category"
            )

            fig_category.update_traces(
                texttemplate="%{y:,.0f}",
                textposition="outside",
                cliponaxis=False,
                opacity=0.70
            )

            fig_category.update_layout(
                height=400,
                showlegend=False,
                xaxis_title="Category",
                yaxis_title="Average Views",
                margin=dict(
                    l=20,
                    r=40,
                    t=60,
                    b=30
                )
            )

            st.plotly_chart(
                fig_category,
                use_container_width=True,
                config={
                    "displayModeBar": False
                },
                key="business_category_performance"
            )

            st.success(
                f"**Recommendation:** Focus more testing and content planning "
                f"around **{top_category}**, which has the highest average "
                f"views in the current dataset."
            )

            st.warning(
                f"**Attention:** **{weakest_category}** has the lowest average "
                f"views. Review its topic selection, packaging and audience fit "
                f"before increasing content volume in this category."
            )

    # =========================================================================
    # UPLOAD TIME
    # =========================================================================

    if (
        "publish_hour" in df.columns
        and "view_count" in df.columns
    ):

        st.markdown("---")
        st.markdown("### 🕒 Upload Timing")

        hour_perf = (
            df.dropna(
                subset=[
                    "publish_hour",
                    "view_count"
                ]
            )
            .groupby("publish_hour")[
                "view_count"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        if not hour_perf.empty:

            best_hour = int(
                hour_perf.index[0]
            )

            best_hour_views = float(
                hour_perf.iloc[0]
            )

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Best-performing Upload Hour",
                    f"{best_hour}:00"
                )

            with c2:

                st.metric(
                    "Average Views",
                    format_number(
                        best_hour_views
                    )
                )

            # -----------------------------------------------------------------
            # HOURLY CHART
            # -----------------------------------------------------------------

            hour_plot = (
                hour_perf
                .sort_index()
                .reset_index()
            )

            hour_plot.columns = [
                "Upload Hour",
                "Average Views"
            ]

            fig_hour = px.line(
                hour_plot,
                x="Upload Hour",
                y="Average Views",
                markers=True,
                template=THEME[
                    "plot_template"
                ],
                title="Average Views by Upload Hour"
            )

            fig_hour.update_traces(
                opacity=0.75
            )

            fig_hour.update_layout(
                height=380,
                xaxis_title="Upload Hour",
                yaxis_title="Average Views",
                margin=dict(
                    l=20,
                    r=30,
                    t=60,
                    b=30
                )
            )

            st.plotly_chart(
                fig_hour,
                use_container_width=True,
                config={
                    "displayModeBar": False
                },
                key="business_upload_hour"
            )

            st.info(
                f"**Timing recommendation:** In this dataset, "
                f"**{best_hour}:00** has the highest average views. "
                f"Use this as a testing signal rather than a guaranteed "
                f"best publishing time."
            )

    # =========================================================================
    # REACH VS ENGAGEMENT
    # =========================================================================

    if (
        "engagement_rate" in df.columns
        and "view_count" in df.columns
    ):

        st.markdown("---")
        st.markdown(
            "### 📊 Reach vs Engagement"
        )

        analysis_df = df[
            [
                "view_count",
                "engagement_rate"
            ]
        ].copy()

        analysis_df = analysis_df.dropna()

        if not analysis_df.empty:

            view_pct = (
                analysis_df[
                    "view_count"
                ]
                .rank(
                    pct=True
                )
            )

            eng_pct = (
                analysis_df[
                    "engagement_rate"
                ]
                .rank(
                    pct=True
                )
            )

            high_reach_low_eng = int(
                (
                    (view_pct >= 0.66)
                    &
                    (eng_pct <= 0.33)
                ).sum()
            )

            high_eng_low_reach = int(
                (
                    (view_pct <= 0.33)
                    &
                    (eng_pct >= 0.66)
                ).sum()
            )

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "High Reach / Low Engagement",
                    high_reach_low_eng
                )

                st.caption(
                    "Strong visibility but comparatively weak audience interaction."
                )

            with c2:

                st.metric(
                    "High Engagement / Low Reach",
                    high_eng_low_reach
                )

                st.caption(
                    "Strong audience response but comparatively weak discovery."
                )

            # -----------------------------------------------------------------
            # SCATTER PLOT
            # -----------------------------------------------------------------

            fig_engagement = px.scatter(
                analysis_df,
                x="view_count",
                y="engagement_rate",
                template=THEME[
                    "plot_template"
                ],
                title="Views vs Engagement Rate",
                labels={
                    "view_count": "Views",
                    "engagement_rate": "Engagement Rate"
                },
            )

            fig_engagement.update_traces(
                opacity=0.60
            )

            fig_engagement.update_xaxes(
                type="log"
            )

            fig_engagement.update_layout(
                height=400,
                margin=dict(
                    l=20,
                    r=30,
                    t=60,
                    b=30
                )
            )

            st.plotly_chart(
                fig_engagement,
                use_container_width=True,
                config={
                    "displayModeBar": False
                },
                key="business_reach_engagement"
            )

            if high_reach_low_eng > high_eng_low_reach:

                st.warning(
                    "The dataset contains more high-reach/low-engagement "
                    "videos than high-engagement/low-reach videos. "
                    "Improving audience interaction may be a useful focus."
                )

            elif high_eng_low_reach > high_reach_low_eng:

                st.info(
                    "The dataset contains more high-engagement/low-reach "
                    "videos. Discovery, distribution and packaging may "
                    "deserve additional attention."
                )

            else:

                st.info(
                    "High-reach/low-engagement and high-engagement/low-reach "
                    "patterns are balanced in the current dataset."
                )

    # =========================================================================
    # PERFORMANCE-TIER GUIDANCE
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "### 🎯 Performance-tier guidance"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.success(
            "**HIGH**\n\n"
            + recommendation_for_class(
                "HIGH"
            )
        )

    with c2:

        st.warning(
            "**MEDIUM**\n\n"
            + recommendation_for_class(
                "MEDIUM"
            )
        )

    with c3:

        st.error(
            "**LOW**\n\n"
            + recommendation_for_class(
                "LOW"
            )
        )

    st.caption(
        "These recommendations summarize historical patterns in the dataset "
        "and should be treated as decision support rather than guaranteed outcomes."
    )


# =====================================================================================
# PAGE: DATASET EXPLORER
# Clean + Interactive + Presentation Ready
# Streamlit-native UI
# =====================================================================================
def page_dataset_explorer():

    st.title("🔍 Dataset Explorer")

    st.caption(
        "Explore the loaded YouTube dataset, inspect data quality, "
        "search videos and export the filtered dataset."
    )

    # =========================================================================
    # DATA
    # =========================================================================

    df = df_raw.copy()

    if df.empty:

        st.warning(
            "The dataset is empty."
        )

        return

    # =========================================================================
    # DATA QUALITY
    # =========================================================================

    qr = quality_report(df)

    st.markdown("### 📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Rows",
            f"{qr['rows']:,}"
        )

    with c2:

        st.metric(
            "Columns",
            f"{qr['columns']:,}"
        )

    with c3:

        st.metric(
            "Missing %",
            f"{qr['missing_pct']}%"
        )

    with c4:

        st.metric(
            "Duplicates Removed",
            f"{qr['duplicates_removed']:,}"
        )

    # =========================================================================
    # SEARCH + FILTER
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "### 🔎 Search & Display"
    )

    c1, c2 = st.columns([1, 1])

    with c1:

        search = st.text_input(
            "Search title contains…",
            placeholder="e.g. Python, Music, Tutorial",
            key="dataset_search"
        )

    with c2:

        all_cols = list(
            df.columns
        )

        default_cols = [
            c
            for c in [
                "title",
                "channel_title",
                "category_name",
                "view_count",
                "like_count",
                "comment_count",
                "publish_date"
            ]
            if c in all_cols
        ]

        chosen = st.multiselect(
            "Columns to display",
            all_cols,
            default=default_cols,
            key="dataset_columns"
        )

    # =========================================================================
    # FILTER DATA
    # =========================================================================

    view = df.copy()

    if (
        search
        and "title" in view.columns
    ):

        view = view[
            view["title"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    # =========================================================================
    # FILTERED DATASET KPIs
    # =========================================================================

    st.markdown(
        "### 📋 Dataset Preview"
    )

    fc1, fc2, fc3 = st.columns(3)

    with fc1:

        st.metric(
            "Matching Videos",
            f"{len(view):,}"
        )

    with fc2:

        if "view_count" in view.columns:

            total_views = pd.to_numeric(
                view["view_count"],
                errors="coerce"
            ).sum()

            st.metric(
                "Views in Current View",
                format_number(
                    total_views
                )
            )

        else:

            st.metric(
                "Views in Current View",
                "—"
            )

    with fc3:

        st.metric(
            "Displayed Columns",
            f"{len(chosen):,}"
        )

    # =========================================================================
    # DATA TABLE
    # =========================================================================

    display_view = (
        view[chosen]
        if chosen
        else view
    )

    st.dataframe(
        display_view,
        use_container_width=True,
        hide_index=True,
        height=430
    )

    st.caption(
        f"Showing **{len(view):,}** matching rows "
        f"out of **{len(df):,}** total rows."
    )

    # =========================================================================
    # COLUMN DATA TYPES
    # =========================================================================

    st.markdown("---")

    with st.expander(
        "🧩 Column Data Types & Missing Values",
        expanded=False
    ):

        dtypes_df = pd.DataFrame(
            {
                "Column": df.dtypes.index,
                "Type": df.dtypes.values.astype(str),
                "Missing": df.isnull().sum().values,
            }
        )

        dtypes_df["Missing %"] = (
            dtypes_df["Missing"]
            / len(df)
            * 100
        ).round(2)

        st.dataframe(
            dtypes_df,
            use_container_width=True,
            hide_index=True
        )

    # =========================================================================
    # NUMERIC SUMMARY
    # =========================================================================

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if not numeric_df.empty:

        st.markdown("---")

        with st.expander(
            "📐 Numeric Summary",
            expanded=False
        ):

            summary_df = (
                numeric_df
                .describe()
                .T
                .reset_index()
                .rename(
                    columns={
                        "index": "Column"
                    }
                )
            )

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )

    # =========================================================================
    # EXPORT
    # =========================================================================

    st.markdown("---")

    st.markdown(
        "### ⬇️ Export"
    )

    csv_bytes = (
        view
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        "⬇️ Download this view as CSV",
        data=csv_bytes,
        file_name="dataset_export.csv",
        mime="text/csv",
        use_container_width=True,
        key="dataset_export_csv"
    )


# =====================================================================================
# PAGE: MODEL INFORMATION / FAQ
# Clean + Professional + Presentation Ready
# Streamlit-native UI
# =====================================================================================

def page_model_info():

    st.title("📚 Model Information")

    st.caption(
        "Frequently asked questions about the regression, classification, "
        "evaluation metrics and machine-learning methodology used in YouTube Intelligence."
    )

    # =========================================================================
    # MACHINE LEARNING BASICS
    # =========================================================================

    st.markdown("### 🤖 Machine Learning Basics")

    with st.expander("❓ What is regression?", expanded=True):

        st.write(
            "Regression predicts a continuous numerical value. "
            "In this project, regression is used to estimate the expected "
            "view count of a YouTube video."
        )

    with st.expander("❓ What is classification?"):

        st.write(
            "Classification predicts a category or class. "
            "Here, the classifier assigns planned videos to one of three "
            "performance tiers: **LOW**, **MEDIUM**, or **HIGH**."
        )

    with st.expander("❓ Why is regression used for views?"):

        st.write(
            "View count is a continuous numerical target. "
            "Regression models are designed to estimate continuous quantities, "
            "making them appropriate for predicting expected YouTube views."
        )

    with st.expander("❓ Why is classification used for performance tiers?"):

        st.write(
            "Exact view counts are useful for forecasting, but business decisions "
            "are often easier to communicate using performance tiers. "
            "Classification converts the prediction problem into three practical "
            "categories: **LOW**, **MEDIUM**, and **HIGH**."
        )

    # =========================================================================
    # DATA & LEAKAGE
    # =========================================================================

    st.markdown("---")
    st.markdown("### 🔐 Data & Leakage Prevention")

    with st.expander("❓ What is target leakage?"):

        st.write(
            "Target leakage occurs when a model uses information that would not "
            "realistically be available at prediction time. This can make model "
            "performance appear better than it really is."
        )

    with st.expander("❓ How is target leakage prevented in this project?"):

        st.write(
            "The prediction pipeline uses features that can be known before or "
            "at publication time. Post-publication metrics such as **likes, "
            "comments, and engagement rate** are excluded from the prediction "
            "feature set because they depend on how the video performs after "
            "publication."
        )

    with st.expander("❓ What does leakage-safe prediction mean?"):

        st.write(
            "Leakage-safe prediction means the model is only given information "
            "that would realistically be available when making the prediction. "
            "This makes the prediction workflow more realistic for pre-publication "
            "content planning."
        )

    # =========================================================================
    # MODEL EVALUATION
    # =========================================================================

    st.markdown("---")
    st.markdown("### 📊 Model Evaluation")

    with st.expander("❓ What is a train/test split?"):

        st.write(
            "The dataset is divided into two parts. The **training set** is used "
            "to fit the models, while the **held-out test set** is used to evaluate "
            "their performance on data that was not used during training."
        )

    with st.expander("❓ What is cross-validation?"):

        st.write(
            "Cross-validation evaluates a model across multiple train/validation "
            "folds and combines the results. It can provide a more robust estimate "
            "of model performance."
        )

        st.info(
            "Cross-validation is not used by default in this project because the "
            "available dataset is relatively small. It is a potential next step "
            "for improving evaluation robustness."
        )

    # =========================================================================
    # REGRESSION METRICS
    # =========================================================================

    st.markdown("---")
    st.markdown("### 📉 Regression Metrics")

    with st.expander("❓ What is MAE?"):

        st.write(
            "**Mean Absolute Error (MAE)** is the average absolute difference "
            "between predicted and actual views."
        )

        st.caption(
            "Lower MAE is better. The value is expressed in the same units as views."
        )

    with st.expander("❓ What is RMSE?"):

        st.write(
            "**Root Mean Squared Error (RMSE)** measures prediction error while "
            "giving greater weight to larger errors."
        )

        st.caption(
            "Lower RMSE is better."
        )

    with st.expander("❓ What is R²?"):

        st.write(
            "**R² (R-squared)** indicates how much variation in the target is "
            "explained by the model."
        )

        st.caption(
            "A value closer to 1 indicates stronger explanatory performance. "
            "R² can also be negative when a model performs worse than a simple "
            "baseline that predicts the average target."
        )

    # =========================================================================
    # CLASSIFICATION METRICS
    # =========================================================================

    st.markdown("---")
    st.markdown("### 🏷️ Classification Metrics")

    with st.expander("❓ What is accuracy?"):

        st.write(
            "**Accuracy** is the proportion of predictions that correctly match "
            "the actual class."
        )

        st.caption(
            "Higher accuracy is better."
        )

    with st.expander("❓ What is precision?"):

        st.write(
            "**Precision** measures how often predictions for a particular class "
            "are correct."
        )

        st.caption(
            "For example, among videos predicted as HIGH, precision indicates "
            "how many were actually HIGH."
        )

    with st.expander("❓ What is recall?"):

        st.write(
            "**Recall** measures how many of the videos that truly belong to a "
            "class were correctly identified by the model."
        )

        st.caption(
            "Higher recall means the model misses fewer examples of that class."
        )

    with st.expander("❓ What is F1 score?"):

        st.write(
            "**F1 score** combines precision and recall into a single metric "
            "using their harmonic mean."
        )

        st.caption(
            "It is useful when both precision and recall matter."
        )

    # =========================================================================
    # PERFORMANCE TIERS
    # =========================================================================

    st.markdown("---")
    st.markdown("### 🎯 Performance Tiers")

    with st.expander("❓ What do LOW, MEDIUM and HIGH mean?"):

        st.write(
            "The classification model assigns each video profile to one of "
            "three performance tiers based on thresholds derived from the "
            "training data."
        )

        st.markdown(
            """
            - 🔴 **LOW** — lower expected performance range
            - 🟡 **MEDIUM** — middle expected performance range
            - 🟢 **HIGH** — higher expected performance range
            """
        )

    with st.expander("❓ Where do the performance thresholds come from?"):

        st.write(
            "The classification thresholds are calculated from the training "
            "split rather than from the video being classified."
        )

        st.caption(
            "This prevents the current prediction from influencing its own "
            "performance-tier definition."
        )

    # =========================================================================
    # MODEL SELECTION
    # =========================================================================

    st.markdown("---")
    st.markdown("### 🧠 Model Selection")

    with st.expander("❓ How is the best regression model selected?"):

        st.write(
            "Regression models are compared using held-out test-set performance. "
            "The configured evaluation strategy prioritizes the model with the "
            "lowest RMSE, with additional metrics used to resolve ties."
        )

    with st.expander("❓ How is the best classification model selected?"):

        st.write(
            "Classification models are compared using held-out test-set metrics. "
            "The configured selection strategy prioritizes weighted F1."
        )

    with st.expander("❓ Can I choose a different model?"):

        st.write(
            "Yes. The application allows the user to explicitly select among "
            "the successfully trained regression or classification models."
        )

        st.caption(
            "The model marked ⭐ Best is recommended by the evaluation strategy, "
            "but it is not silently forced."
        )

    # =========================================================================
    # REPRODUCIBILITY
    # =========================================================================

    st.markdown("---")
    st.markdown("### 🔁 Reproducibility")

    with st.expander(
        "❓ What settings are used for reproducibility?",
        expanded=True
    ):

        st.markdown(
            """
            - `random_state = 42` is used wherever supported.
            - Regression models include Linear Regression, Random Forest and
              Gradient Boosting, with Extra Trees and XGBoost included when available.
            - Classification models include Logistic Regression, Decision Tree,
              Random Forest and Gradient Boosting.
            - Classification thresholds are calculated from the training split.
            - The user can explicitly select the trained model used for prediction.
            """
        )

    # =========================================================================
    # IMPORTANT LIMITATION
    # =========================================================================

    st.markdown("---")

    st.warning(
        "⚠️ **Important:** Model predictions are statistical estimates based on "
        "historical data. They should support decision-making, not be treated as "
        "guaranteed future YouTube performance."
    )


# =====================================================================================
# PAGE: ABOUT / PROJECT OVERVIEW
# Complete Project Story — From Raw Data to Business Intelligence
# Streamlit-native UI
# No raw HTML / no unsafe_allow_html
# Presentation Ready
# =====================================================================================

def page_about():

    st.title("ℹ️ About YouTube Intelligence")

    st.caption(
        f"Analytics • Prediction • Classification • Business Intelligence "
        f"| Version {APP_VERSION}"
    )

    # =========================================================================
    # PROJECT INTRODUCTION
    # =========================================================================

    st.markdown("## 🎬 Project Overview")

    st.write(
        "YouTube Intelligence is an end-to-end data science and machine-learning "
        "application built to understand historical YouTube video performance "
        "and convert that information into actionable business insights."
    )

    st.write(
        "The project starts with raw historical YouTube trending-video data and "
        "progressively transforms it into clean data, descriptive analytics, "
        "engineered features, machine-learning models, predictions and "
        "business recommendations."
    )

    st.info(
        "🎯 **Core objective:** Understand what patterns are associated with "
        "YouTube video performance and use those patterns to support "
        "pre-publication content decisions."
    )

    # =========================================================================
    # PROJECT JOURNEY
    # =========================================================================

    st.markdown("---")
    st.markdown("## 🔄 Project Journey")

    st.caption(
        "The complete workflow followed in this project."
    )

    journey = [
        (
            "1️⃣",
            "Raw Data",
            "Load historical YouTube trending-video data."
        ),
        (
            "2️⃣",
            "Data Understanding",
            "Inspect columns, data types, missing values, duplicates and target variables."
        ),
        (
            "3️⃣",
            "Data Cleaning",
            "Prepare the dataset for reliable analysis and modelling."
        ),
        (
            "4️⃣",
            "Exploratory Analysis",
            "Study views, categories, engagement and publishing patterns."
        ),
        (
            "5️⃣",
            "Business Analytics",
            "Convert statistical patterns into meaningful business insights."
        ),
        (
            "6️⃣",
            "Feature Engineering",
            "Create useful model features from information available before publication."
        ),
        (
            "7️⃣",
            "Leakage Prevention",
            "Remove post-publication information that would make prediction unrealistic."
        ),
        (
            "8️⃣",
            "Regression",
            "Predict the expected number of YouTube views."
        ),
        (
            "9️⃣",
            "Classification",
            "Assign videos to LOW, MEDIUM or HIGH performance tiers."
        ),
        (
            "🔟",
            "Model Evaluation",
            "Compare models using held-out test-set performance."
        ),
        (
            "1️⃣1️⃣",
            "Prediction Studio",
            "Generate predictions for a new planned video."
        ),
        (
            "1️⃣2️⃣",
            "Business Decision",
            "Translate model output into practical content recommendations."
        ),
    ]

    for icon, title, description in journey:

        with st.container(border=True):

            c1, c2 = st.columns([1, 5])

            with c1:

                st.markdown(
                    f"### {icon}"
                )

            with c2:

                st.markdown(
                    f"**{title}**"
                )

                st.caption(
                    description
                )

    # =========================================================================
    # BUSINESS PROBLEM
    # =========================================================================

    st.markdown("---")
    st.markdown("## 🎯 Business Problem")

    st.write(
        "A YouTube creator or content team needs more than a table of historical "
        "videos. They need to understand which types of content perform well, "
        "how publishing patterns relate to views, how engagement behaves and "
        "whether a planned video resembles historically successful content."
    )

    st.write(
        "Therefore, the project addresses two complementary questions:"
    )

    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.markdown(
                "### 📊 What happened?"
            )

            st.write(
                "Descriptive analytics explains historical performance using "
                "KPIs, distributions, categories, engagement and time-based analysis."
            )

    with c2:

        with st.container(border=True):

            st.markdown(
                "### 🔮 What might happen?"
            )

            st.write(
                "Machine-learning models estimate expected views and classify "
                "planned content into performance tiers."
            )

    # =========================================================================
    # DATASET
    # =========================================================================

    st.markdown("---")
    st.markdown("## 🗂️ Dataset")

    if "df_raw" in globals() and isinstance(df_raw, pd.DataFrame):

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Rows",
                f"{len(df_raw):,}"
            )

        with c2:

            st.metric(
                "Columns",
                f"{len(df_raw.columns):,}"
            )

        with c3:

            if "category_name" in df_raw.columns:

                category_count = (
                    df_raw["category_name"]
                    .nunique()
                )

                st.metric(
                    "Categories",
                    f"{category_count:,}"
                )

            else:

                st.metric(
                    "Categories",
                    "—"
                )

        with c4:

            if "view_count" in df_raw.columns:

                total_views = pd.to_numeric(
                    df_raw["view_count"],
                    errors="coerce"
                ).sum()

                st.metric(
                    "Total Views",
                    format_number(total_views)
                )

            else:

                st.metric(
                    "Total Views",
                    "—"
                )

    st.write(
        "The dataset contains historical YouTube video records. "
        "The application uses the loaded dataset as the source for the "
        "dashboard's descriptive statistics, visualizations and machine-learning workflow."
    )

    st.caption(
        "All displayed dataset KPIs are calculated from the currently loaded data."
    )

    # =========================================================================
    # DATA UNDERSTANDING & CLEANING
    # =========================================================================

    st.markdown("---")
    st.markdown("## 🧹 Data Understanding & Preparation")

    st.write(
        "Before modelling, the data needs to be understood and prepared. "
        "The project inspects the structure of the dataset, identifies relevant "
        "columns, checks missing values and handles duplicate records where applicable."
    )

    preparation_steps = [
        "Inspect dataset dimensions and column names.",
        "Understand numerical, categorical and date-related variables.",
        "Check missing values and data completeness.",
        "Identify duplicate records.",
        "Convert relevant fields into appropriate data types.",
        "Prepare the dataset for analysis and machine learning.",
    ]

    for step in preparation_steps:

        st.markdown(
            f"✓ {step}"
        )

    # =========================================================================
    # EXPLORATORY DATA ANALYSIS
    # =========================================================================

    st.markdown("---")
    st.markdown("## 📈 Exploratory Data Analysis")

    st.write(
        "Exploratory analysis is used to understand how YouTube performance "
        "behaves before applying machine learning."
    )

    eda_topics = [
        (
            "👁️ Views",
            "Understand the distribution and scale of video view counts."
        ),
        (
            "🗂️ Categories",
            "Compare performance across different content categories."
        ),
        (
            "👍 Engagement",
            "Study likes, comments and engagement-related patterns where available."
        ),
        (
            "🕒 Publishing Time",
            "Investigate relationships between publishing time and video performance."
        ),
        (
            "📅 Publishing Patterns",
            "Explore day, month and other temporal patterns."
        ),
        (
            "🎬 Content Characteristics",
            "Study characteristics such as duration and text-based metadata."
        ),
    ]

    for title, description in eda_topics:

        with st.container(border=True):

            st.markdown(
                f"**{title}**"
            )

            st.caption(
                description
            )

    # =========================================================================
    # BUSINESS INTELLIGENCE
    # =========================================================================

    st.markdown("---")
    st.markdown("## 💼 Business Intelligence")

    st.write(
        "The analytical layer converts raw statistical observations into "
        "business-oriented insights."
    )

    st.markdown(
        """
        **Examples of questions answered by the application:**

        - Which content categories have stronger average performance?
        - Which categories appear weaker?
        - How does publishing time relate to average views?
        - Which videos combine high reach with lower engagement?
        - Which videos show strong engagement but relatively low reach?
        - What historical patterns can help guide future content planning?
        """
    )

    st.caption(
        "These relationships describe historical patterns and should not automatically "
        "be interpreted as causal effects."
    )

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================

    st.markdown("---")
    st.markdown("## 🧩 Feature Engineering")

    st.write(
        "Machine-learning models cannot directly work with every raw field in the "
        "most useful form. The project therefore transforms available information "
        "into model-ready features."
    )

    feature_groups = {
        "⏱️ Video Features": [
            "duration_seconds",
            "duration_category",
        ],
        "📝 Text Features": [
            "title_length",
            "title_word_count",
            "description_length",
            "description_word_count",
            "tag_count",
        ],
        "📅 Publishing Features": [
            "publish_hour",
            "publish_day_name",
            "publish_day",
            "publish_week",
            "publish_month",
            "publish_session",
            "month_part",
            "is_weekend",
        ],
        "🗂️ Categorical Features": [
            "category_name",
            "caption_label",
        ],
    }

    for group_name, features in feature_groups.items():

        with st.expander(
            group_name
        ):

            st.write(
                ", ".join(features)
            )

    # =========================================================================
    # TARGET LEAKAGE
    # =========================================================================

    st.markdown("---")
    st.markdown("## 🔐 Leakage-Safe Machine Learning")

    st.warning(
        "⚠️ **Target leakage is one of the most important considerations in this project.**"
    )

    st.write(
        "A prediction model should only use information that would realistically "
        "be available when the prediction is made."
    )

    st.write(
        "For example, likes, comments and engagement metrics can be strongly "
        "related to views, but they are generally observed after a video has "
        "already been published and accumulated audience activity."
    )

    st.success(
        "✅ The prediction workflow therefore focuses on pre-publication or "
        "publication-time information such as category, duration, title, "
        "description, tags, captions and publishing characteristics."
    )

    # =========================================================================
    # REGRESSION
    # =========================================================================

    st.markdown("---")
    st.markdown("## 🔮 Regression — View Prediction")

    st.write(
        "Regression is used because the target variable, video views, is a "
        "continuous numerical quantity."
    )

    st.markdown(
        """
        **Goal:** Estimate the expected number of views for a planned video.

        **Model candidates include:**

        - Linear Regression
        - Random Forest Regressor
        - Gradient Boosting Regressor
        - Extra Trees when available
        - XGBoost when available
        """
    )

    st.write(
        "The models are trained using the leakage-safe feature set and evaluated "
        "on a held-out test set."
    )

    # =========================================================================
    # CLASSIFICATION
    # =========================================================================

    st.markdown("---")
    st.markdown("## 🏷️ Classification — Performance Tier")

    st.write(
        "Classification provides a simpler business interpretation by converting "
        "performance into three tiers."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.error(
            "🔴 **LOW**"
        )

        st.caption(
            "Lower performance range."
        )

    with c2:

        st.warning(
            "🟡 **MEDIUM**"
        )

        st.caption(
            "Middle performance range."
        )

    with c3:

        st.success(
            "🟢 **HIGH**"
        )

        st.caption(
            "Higher performance range."
        )

    st.write(
        "The classification thresholds are derived from the training data so "
        "that the current prediction does not define its own target class."
    )

    # =========================================================================
    # MODEL EVALUATION
    # =========================================================================

    st.markdown("---")
    st.markdown("## 📉 Model Evaluation")

    st.write(
        "Models are not judged only by whether they successfully train. "
        "Their performance is evaluated on held-out test data."
    )

    evaluation_metrics = [
        (
            "Regression",
            "MAE, RMSE and R²"
        ),
        (
            "Classification",
            "Accuracy, Precision, Recall and Weighted F1"
        ),
    ]

    for model_type, metrics in evaluation_metrics:

        with st.container(border=True):

            st.markdown(
                f"**{model_type}**"
            )

            st.caption(
                f"Evaluation metrics: {metrics}"
            )

    st.info(
        "⭐ The application marks the model selected by the configured evaluation "
        "strategy as the current Best model, while still allowing the user to "
        "inspect and select other trained models."
    )

    # =========================================================================
    # PREDICTION STUDIO
    # =========================================================================

    st.markdown("---")
    st.markdown("## 🎯 Prediction Studio")

    st.write(
        "Prediction Studio allows a user to enter a planned video's information "
        "and generate an estimate without requiring post-publication metrics."
    )

    st.markdown(
        """
        **Typical inputs include:**

        - Content category
        - Video duration
        - Planned upload date
        - Upload hour
        - Caption availability
        - Video title
        - Tags
        - Description
        """
    )

    st.write(
        "The selected regression model generates an expected view count, while "
        "the classification model can assign the corresponding performance tier."
    )

    # =========================================================================
    # BUSINESS RECOMMENDATIONS
    # =========================================================================

    st.markdown("---")
    st.markdown("## 💡 Business Recommendations")

    st.write(
        "The final layer translates analytical and machine-learning results "
        "into practical decision support."
    )

    st.markdown(
        """
        Recommendations can help teams think about:

        - Which categories may deserve more attention.
        - Which publishing patterns are associated with stronger historical performance.
        - Whether content has strong reach but comparatively weak engagement.
        - Whether strong engagement is accompanied by limited reach.
        - How HIGH, MEDIUM and LOW performance profiles can influence content planning.
        """
    )

    st.caption(
        "Recommendations describe historical patterns and are not guaranteed outcomes."
    )

    # =========================================================================
    # PREDICTION HISTORY
    # =========================================================================

    st.markdown("---")
    st.markdown("## 🕘 Prediction History")

    st.write(
        "Prediction History records predictions made during the current "
        "application session so that users can review and export previous results."
    )

    st.markdown(
        """
        Each saved prediction can include:

        - Timestamp
        - Video title
        - Category
        - Duration
        - Upload date and hour
        - Predicted views
        - Predicted performance tier
        - Model used
        """
    )

    # =========================================================================
    # MODEL INFORMATION
    # =========================================================================

    st.markdown("---")
    st.markdown("## 📚 Model Transparency")

    st.write(
        "The application provides dedicated pages for understanding the models, "
        "comparing trained models, evaluating test-set performance and inspecting "
        "feature importance."
    )

    transparency_points = [
        "Model selection is visible to the user.",
        "Evaluation metrics are shown using held-out test data.",
        "Feature importance is presented as model contribution, not causal impact.",
        "Prediction inputs are designed around leakage-safe information.",
        "Unavailable models or failed predictions are reported rather than fabricated.",
    ]

    for point in transparency_points:

        st.markdown(
            f"✓ {point}"
        )

    # =========================================================================
    # TECHNOLOGY STACK
    # =========================================================================

    st.markdown("---")
    st.markdown("## 🛠️ Technology Stack")

    stack = [
        ("🐍 Python", "Core programming language"),
        ("🐼 Pandas", "Data manipulation and analysis"),
        ("🔢 NumPy", "Numerical computation"),
        ("🤖 scikit-learn", "Machine learning and evaluation"),
        ("📊 Plotly", "Interactive data visualization"),
        ("🖥️ Streamlit", "Interactive application interface"),
    ]

    for technology, purpose in stack:

        with st.container(border=True):

            st.markdown(
                f"**{technology}**"
            )

            st.caption(
                purpose
            )

    # =========================================================================
    # REPRODUCIBILITY
    # =========================================================================

    st.markdown("---")
    st.markdown("## 🔁 Reproducibility")

    st.write(
        "The project is designed so that the analytical and modelling workflow "
        "can be reproduced from the loaded dataset and configured pipeline."
    )

    st.markdown(
        """
        - `random_state = 42` is used wherever supported.
        - Models are evaluated using held-out test data.
        - Regression and classification use defined feature sets.
        - Classification thresholds are derived from training data.
        - The selected model is explicitly displayed.
        - Metrics are calculated from model outputs rather than hard-coded values.
        """
    )

    # =========================================================================
    # HONESTY / DATA SCIENCE PRINCIPLE
    # =========================================================================

    st.markdown("---")
    st.markdown("## 🧠 Data Science & Transparency Principle")

    st.info(
        "Every KPI, chart, metric, model result, prediction and recommendation "
        "shown by the application should come from the loaded dataset or trained "
        "models. If the required data or model is unavailable, the application "
        "reports that limitation instead of inventing a result."
    )

    st.caption(
        "Historical correlation should not automatically be interpreted as causation."
    )

    # =========================================================================
    # LIMITATIONS
    # =========================================================================

    st.markdown("---")
    st.markdown("## ⚠️ Current Limitations")

    limitations = [
        "The available dataset is relatively small.",
        "Historical YouTube trends may not represent future platform behaviour.",
        "Model performance depends on the quality and representativeness of the dataset.",
        "A prediction is an estimate, not a guarantee of future views.",
        "Observed relationships do not establish causal effects.",
        "External factors such as trends, competition, audience changes and recommendation-system changes are not fully represented.",
    ]

    for limitation in limitations:

        st.markdown(
            f"• {limitation}"
        )

    # =========================================================================
    # FUTURE SCOPE
    # =========================================================================

    st.markdown("---")
    st.markdown("## 🚀 Future Scope")

    future_scope = [
        "📡 YouTube Data API integration",
        "⚡ Real-time data ingestion",
        "🔄 Scheduled model retraining",
        "📝 Advanced NLP for titles and descriptions",
        "🖼️ Thumbnail image analysis",
        "📅 Time-series forecasting",
        "🧪 A/B testing support",
        "☁️ Cloud deployment",
        "📈 Model monitoring and drift detection",
        "🎯 More advanced content-performance modelling",
    ]

    for item in future_scope:

        st.markdown(
            f"• {item}"
        )

    # =========================================================================
    # FINAL PROJECT SUMMARY
    # =========================================================================

    st.markdown("---")
    st.markdown("## 🏁 Final Project Summary")

    st.success(
        "Raw Data → Data Understanding → Cleaning → EDA → Business Analytics "
        "→ Feature Engineering → Leakage Prevention → Regression "
        "→ Classification → Model Evaluation → Prediction "
        "→ Business Recommendation"
    )

    st.write(
        "The main purpose of YouTube Intelligence is not simply to predict a "
        "number. It demonstrates how a complete data-science workflow can "
        "transform raw historical data into understandable analytics, "
        "machine-learning predictions and practical business decision support."
    )

    st.caption(
        f"YouTube Intelligence v{APP_VERSION} • "
        "Data Science • Analytics • Machine Learning"
    )


# =====================================================================================
# ROUTER
# =====================================================================================
ROUTES = {
    "🏠 Home": page_home,
    "📊 Executive Dashboard": page_dashboard,
    "🎬 Content Analytics": page_content,
    "💬 Engagement Analytics": page_engagement,
    "📈 Trend Analysis": page_trends,
    "🤖 ML Overview": page_ml_overview,
    "🔮 View Prediction": page_prediction,
    "🏷️ Performance Classification": page_classification,
    "⚖️ Model Comparison": page_model_comparison,
    "📉 Model Evaluation": page_model_evaluation,
    "🌟 Feature Importance": page_feature_importance,
    "🕘 Prediction History": page_history,
    "💡 Business Recommendations": page_recommendations,
    "🔍 Dataset Explorer": page_dataset_explorer,
    "📚 Model Information": page_model_info,
    "ℹ️ About": page_about,
}

ROUTES[page]()
render_app_footer()