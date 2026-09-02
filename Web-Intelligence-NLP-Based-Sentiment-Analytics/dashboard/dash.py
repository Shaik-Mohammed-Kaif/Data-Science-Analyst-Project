# ================================================================
# WEB INTELLIGENCE • NLP ANALYTICS
# TEXT INTELLIGENCE DASHBOARD
#
# Power BI inspired Streamlit dashboard
# Reference version updated:
#   • Same cream canvas
#   • Equal-width KPI cards
#   • Global slicers
#   • 8 attractive NLP visuals (4 × 2)
#   • Correct project column mapping
#   • Real VADER score chart
#   • Real text-length relationship chart
#   • Topic slicer uses lda_dominant_topic
#   • Model inference uses joblib (matching Notebook 04)
#   • No fake dates
#   • Native Streamlit analytical layout
# ================================================================

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="Text Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ================================================================
# MULTI-THEME CONTROL
# ================================================================

if "dashboard_theme" not in st.session_state:
    st.session_state["dashboard_theme"] = "Vanilla"

THEMES = {
    "Vanilla": {
        "BG": "#F7F1E7",
        "SURFACE": "#FFFCF7",
        "SURFACE_ALT": "#FBF6ED",
        "BORDER": "#E5D8C7",
        "INK": "#3A2417",
        "TEXT": "#4D3A2B",
        "MUTED": "#806F61",
        "ACCENT": "#A85A18",
        "ACCENT_DARK": "#71370D",
        "POSITIVE": "#4CAF50",
        "NEGATIVE": "#E53935",
        "NEUTRAL": "#F5A900",
        "PURPLE": "#7651A9",
        "BLUE": "#4B83C4",
        "TEAL": "#4C9188",
    },
    "NLP Sage": {
        "BG": "#EEF3EC",
        "SURFACE": "#FAFCF8",
        "SURFACE_ALT": "#F3F7F1",
        "BORDER": "#D3DED0",
        "INK": "#24372A",
        "TEXT": "#405247",
        "MUTED": "#708075",
        "ACCENT": "#587C5B",
        "ACCENT_DARK": "#35563A",
        "POSITIVE": "#4F9B62",
        "NEGATIVE": "#D65C52",
        "NEUTRAL": "#D6A12A",
        "PURPLE": "#78649B",
        "BLUE": "#587EA6",
        "TEAL": "#4F8F84",
    },
    "Midnight NLP": {
        "BG": "#15191D",
        "SURFACE": "#20262C",
        "SURFACE_ALT": "#1B2126",
        "BORDER": "#35404A",
        "INK": "#F4EEE5",
        "TEXT": "#D8D2C9",
        "MUTED": "#9EA6AD",
        "ACCENT": "#D58A45",
        "ACCENT_DARK": "#F0A15D",
        "POSITIVE": "#68B87A",
        "NEGATIVE": "#EF6C68",
        "NEUTRAL": "#E3B64B",
        "PURPLE": "#9B7BC5",
        "BLUE": "#6F9FD0",
        "TEAL": "#62A79D",
    },
}

ACTIVE_THEME = THEMES[
    st.session_state["dashboard_theme"]
]

BG = ACTIVE_THEME["BG"]
SURFACE = ACTIVE_THEME["SURFACE"]
SURFACE_ALT = ACTIVE_THEME["SURFACE_ALT"]
BORDER = ACTIVE_THEME["BORDER"]

INK = ACTIVE_THEME["INK"]
TEXT = ACTIVE_THEME["TEXT"]
MUTED = ACTIVE_THEME["MUTED"]

ACCENT = ACTIVE_THEME["ACCENT"]
ACCENT_DARK = ACTIVE_THEME["ACCENT_DARK"]

POSITIVE = ACTIVE_THEME["POSITIVE"]
NEGATIVE = ACTIVE_THEME["NEGATIVE"]
NEUTRAL = ACTIVE_THEME["NEUTRAL"]

PURPLE = ACTIVE_THEME["PURPLE"]
BLUE = ACTIVE_THEME["BLUE"]
TEAL = ACTIVE_THEME["TEAL"]


# ================================================================
# VANILLA THEME
# ================================================================

# ================================================================
# LIGHT STREAMLIT POLISH
# ================================================================

st.markdown(
    f"""
    <style>
        .stApp {{
            background: {BG};
            color: {TEXT};
        }}

        .main .block-container {{
            max-width: 1540px;
            padding-top: 1.0rem;
            padding-bottom: 1.8rem;
            padding-left: 1.15rem;
            padding-right: 1.15rem;
        }}

        h1, h2, h3, h4 {{
            color: {INK} !important;
            letter-spacing: -0.025em;
        }}

        h1 {{
            font-size: 2.45rem !important;
            line-height: 1.0 !important;
            margin-bottom: 0.25rem !important;
        }}

        h2 {{
            font-size: 1.05rem !important;
        }}

        h3 {{
            font-size: 0.92rem !important;
        }}

        p, label {{
            color: {TEXT};
        }}

        [data-testid="stMetric"] {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 9px;
            padding: 13px 14px;
            min-height: 103px;
            box-shadow: 0 1px 3px rgba(75, 45, 20, 0.04);
        }}

        [data-testid="stMetricLabel"] {{
            color: {MUTED} !important;
            font-size: 0.76rem !important;
        }}

        [data-testid="stMetricValue"] {{
            color: {INK} !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
        }}

        div[data-baseweb="select"] > div {{
            background: {SURFACE};
            border-color: {BORDER};
            border-radius: 7px;
        }}

        .stTextInput input,
        textarea {{
            background: {SURFACE} !important;
            color: {TEXT} !important;
            border-color: {BORDER} !important;
            border-radius: 7px !important;
        }}

        .stButton > button,
        .stDownloadButton > button {{
            background: {SURFACE};
            color: {INK};
            border: 1px solid {BORDER};
            border-radius: 7px;
            min-height: 38px;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            border-color: {ACCENT};
            color: {ACCENT};
        }}

        hr {{
            border-color: {BORDER};
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid {BORDER};
            border-radius: 7px;
        }}

        /* ========================================================
           SUBTLE FLOWING NLP BUBBLE BACKGROUND
           Decorative only — does not cover or intercept controls.
           ======================================================== */

        .stApp::before,
        .stApp::after {{
            content: "";
            position: fixed;
            z-index: 0;
            pointer-events: none;
            border-radius: 50%;
            filter: blur(1px);
        }}

        .stApp::before {{
            width: 330px;
            height: 330px;
            left: -120px;
            top: 12%;
            background: radial-gradient(
                circle,
                rgba(168, 90, 24, 0.13) 0%,
                rgba(168, 90, 24, 0.07) 38%,
                rgba(168, 90, 24, 0) 72%
            );
            animation: nlpBubbleFlowA 18s ease-in-out infinite alternate;
        }}

        .stApp::after {{
            width: 390px;
            height: 390px;
            right: -150px;
            bottom: 8%;
            background: radial-gradient(
                circle,
                rgba(118, 81, 169, 0.10) 0%,
                rgba(118, 81, 169, 0.045) 40%,
                rgba(118, 81, 169, 0) 72%
            );
            animation: nlpBubbleFlowB 22s ease-in-out infinite alternate;
        }}

        @keyframes nlpBubbleFlowA {{
            0% {{
                transform: translate3d(0, 0, 0) scale(0.92);
            }}
            50% {{
                transform: translate3d(90px, 70px, 0) scale(1.08);
            }}
            100% {{
                transform: translate3d(35px, 150px, 0) scale(0.98);
            }}
        }}

        @keyframes nlpBubbleFlowB {{
            0% {{
                transform: translate3d(0, 0, 0) scale(0.94);
            }}
            50% {{
                transform: translate3d(-95px, -65px, 0) scale(1.10);
            }}
            100% {{
                transform: translate3d(-35px, -145px, 0) scale(0.98);
            }}
        }}

        /* Keep dashboard content above the decorative background. */
        .main .block-container {{
            position: relative;
            z-index: 2;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)



# ================================================================
# IMAGE-REFERENCE DASHBOARD POLISH
# ================================================================

st.markdown(
    """
    <style>
        .main .block-container {
            max-width: 1540px;
        }


        /* Hide Streamlit automatic heading anchor icons */
        h1 a,
        h2 a,
        h3 a,
        h4 a {
            display: none !important;
            visibility: hidden !important;
        }


        /* ------------------------------------------------
           Executive KPI information-graphic cards
           ------------------------------------------------ */

        .kpi-card {
            width: 100%;
            min-height: 116px;
            box-sizing: border-box;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 13px;
            padding: 16px 14px;
            border: 1px solid rgba(229, 216, 199, 0.95);
            border-radius: 14px;
            background: rgba(255, 252, 247, 0.92);
            box-shadow: 0 4px 14px rgba(58, 36, 23, 0.07);
        }

        .kpi-icon-badge {
            width: 46px;
            height: 46px;
            min-width: 46px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background: rgba(168, 90, 24, 0.11);
            font-size: 1.45rem;
            line-height: 1;
            box-shadow: inset 0 0 0 1px rgba(168, 90, 24, 0.05);
        }

        .kpi-content {
            min-width: 0;
            text-align: center;
        }

        .kpi-label {
            color: #4D3A2B;
            font-size: 0.80rem;
            line-height: 1.25;
            margin-bottom: 7px;
            white-space: nowrap;
        }

        .kpi-value-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            white-space: nowrap;
        }

        .kpi-value {
            color: #3A2417;
            font-size: 1.48rem;
            line-height: 1;
            font-weight: 750;
        }

        .kpi-delta {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 23px;
            padding: 3px 7px;
            border-radius: 999px;
            font-size: 0.72rem;
            line-height: 1;
            font-weight: 650;
            white-space: nowrap;
        }

        .kpi-delta.positive {
            color: #2E7D32;
            background: #E8F5E9;
        }

        .kpi-delta.neutral {
            color: #6B625A;
            background: #F0EEEB;
        }

        .kpi-delta.negative {
            color: #C62828;
            background: #FFEBEE;
        }

        .kpi-delta.score,
        .kpi-delta.records,
        .kpi-delta.authors {
            color: #4D3A2B;
            background: #F3EDE4;
        }

        .reset-button-space {
            height: 8px;
        }

        /* Slicer row */
        div[data-baseweb="select"] > div {
            min-height: 42px;
        }

        /* KPI equal width */
        [data-testid="stMetric"] {
            width: 100%;
            box-sizing: border-box;
        }

        /* Keep the delta beside the metric value where Streamlit
           exposes it as a sibling. */
        [data-testid="stMetricDelta"] {
            white-space: nowrap;
            margin-left: 7px !important;
        }

        /* Chart card rhythm */
        [data-testid="stColumn"] {
            min-width: 0;
        }

        /* Native Streamlit radio theme control */
        div[role="radiogroup"] {
            gap: 5px;
        }

        div[role="radiogroup"] label {
            border: 1px solid rgba(128,111,97,0.25);
            border-radius: 8px;
            padding: 5px 9px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# PROJECT PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "final"
    / "nlp_final_dataset.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_sentiment_model.pkl"
)

VECTORIZER_PATH = (
    BASE_DIR
    / "models"
    / "tfidf_vectorizer.pkl"
)

KEYWORDS_PATH = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "final_top_keywords.csv"
)

BIGRAM_PATH = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "top_bigrams.csv"
)

TRIGRAM_PATH = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "top_trigrams.csv"
)

TOPIC_SUMMARY_PATH = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "final_topic_summary.csv"
)


# ================================================================
# HELPERS
# ================================================================

def find_column(frame, candidates):
    """Find a column without depending on exact capitalization."""

    mapping = {
        str(column).lower().strip(): column
        for column in frame.columns
    }

    for candidate in candidates:

        key = candidate.lower().strip()

        if key in mapping:
            return mapping[key]

    return None


def normalize_sentiment(value):
    """Normalize sentiment labels."""

    if pd.isna(value):
        return "Unknown"

    text = str(value).strip().lower()

    if "positive" in text:
        return "Positive"

    if "negative" in text:
        return "Negative"

    if "neutral" in text:
        return "Neutral"

    return str(value).title()


def topic_label(value):
    """Convert numeric/string topic IDs into readable labels."""

    if pd.isna(value):
        return "Unknown"

    text = str(value).strip()

    match = re.search(r"(\d+)", text)

    if match:
        return f"Topic {match.group(1)}"

    return text


def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def style_chart(fig, height=330):
    """Consistent chart styling."""

    fig.update_layout(
        height=height,
        margin=dict(
            l=14,
            r=14,
            t=48,
            b=24,
        ),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(
            family="Arial",
            color=TEXT,
            size=10,
        ),
        title_font=dict(
            family="Arial",
            color=INK,
            size=14,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(
                color=TEXT,
                size=9,
            ),
        ),
        hoverlabel=dict(
            bgcolor=SURFACE,
            font_color=INK,
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=BORDER,
        color=MUTED,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=BORDER,
        zeroline=False,
        linecolor=BORDER,
        color=MUTED,
    )

    return fig


def show_chart(fig):
    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


def empty_plot(message):
    st.info(message)


# ================================================================
# LOAD DATA
# ================================================================

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


if not DATA_PATH.exists():

    st.error(
        f"Dashboard dataset was not found:\n\n{DATA_PATH}"
    )

    st.stop()


df = load_dataset(DATA_PATH)


if df.empty:

    st.error(
        "The dashboard dataset contains no records."
    )

    st.stop()


# ================================================================
# REAL PROJECT COLUMN MAPPING
# ================================================================
#
# Notebook 06 explicitly saves:
#
# quote_id
# original_text
# processed_text
# author
# tags
# source_url
# compound_score
# sentiment
# predicted_sentiment
# prediction_score
# lda_dominant_topic
# lda_topic_score
# nmf_dominant_topic
# nmf_topic_score
#
# The previous dashboard was looking for:
#   vader_compound
#   dominant_topic
# which caused the 2nd plot / topic display to fail.
#
# This version maps the real columns first.
# ================================================================

TEXT_COL = find_column(
    df,
    [
        "original_text",
        "quote",
        "text",
        "processed_text",
    ],
)

PROCESSED_TEXT_COL = find_column(
    df,
    [
        "processed_text",
        "clean_text",
    ],
)

AUTHOR_COL = find_column(
    df,
    [
        "author",
        "author_name",
    ],
)

SENTIMENT_COL = find_column(
    df,
    [
        "sentiment",
        "sentiment_label",
    ],
)

COMPOUND_COL = find_column(
    df,
    [
        "compound_score",
        "vader_compound",
        "compound",
        "sentiment_score",
    ],
)

PREDICTED_SENTIMENT_COL = find_column(
    df,
    [
        "predicted_sentiment",
        "model_sentiment",
    ],
)

PREDICTION_SCORE_COL = find_column(
    df,
    [
        "prediction_score",
        "model_score",
    ],
)

LDA_TOPIC_COL = find_column(
    df,
    [
        "lda_dominant_topic",
        "dominant_topic",
        "topic",
        "topic_id",
    ],
)

LDA_TOPIC_SCORE_COL = find_column(
    df,
    [
        "lda_topic_score",
        "topic_score",
    ],
)

NMF_TOPIC_COL = find_column(
    df,
    [
        "nmf_dominant_topic",
    ],
)


# ================================================================
# STANDARD ANALYTICAL COLUMNS
# ================================================================

if AUTHOR_COL:

    df["_author"] = (
        df[AUTHOR_COL]
        .fillna("Unknown")
        .astype(str)
    )

else:

    df["_author"] = "Unknown"


if SENTIMENT_COL:

    df["_sentiment"] = (
        df[SENTIMENT_COL]
        .apply(normalize_sentiment)
    )

else:

    df["_sentiment"] = "Unknown"


# ---- REAL VADER SCORE ----

if COMPOUND_COL:

    df["_compound_score"] = numeric(
        df[COMPOUND_COL]
    )

else:

    df["_compound_score"] = np.nan


# ---- REAL LDA TOPIC ----

if LDA_TOPIC_COL:

    df["_lda_topic_raw"] = df[
        LDA_TOPIC_COL
    ]

    df["_topic"] = (
        df[LDA_TOPIC_COL]
        .apply(topic_label)
    )

else:

    df["_lda_topic_raw"] = np.nan
    df["_topic"] = "Unknown"


# ---- TOPIC SCORE ----

if LDA_TOPIC_SCORE_COL:

    df["_topic_score"] = numeric(
        df[LDA_TOPIC_SCORE_COL]
    )

else:

    df["_topic_score"] = np.nan


# ---- WORD COUNT ----

df["_word_count"] = np.nan

if PROCESSED_TEXT_COL:

    df["_word_count"] = (
        df[PROCESSED_TEXT_COL]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

elif TEXT_COL:

    df["_word_count"] = (
        df[TEXT_COL]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )


# ---- CHARACTER COUNT ----

if TEXT_COL:

    df["_char_count"] = (
        df[TEXT_COL]
        .fillna("")
        .astype(str)
        .str.len()
    )

else:

    df["_char_count"] = np.nan


# ================================================================
# HEADER
# ================================================================

head_left, head_main, head_theme = st.columns(
    [0.8, 6.4, 2.8],
    gap="medium",
)

with head_left:

    st.markdown(
        "## 🧠",
    )


with head_main:

    st.caption(
        "WEB INTELLIGENCE  •  NLP ANALYTICS"
    )

    st.title(
        "Text Intelligence Dashboard"
    )

    st.write(
        "Web intelligence • NLP analytics • Insights from text data"
    )


with head_theme:

    st.caption(
        "THEME"
    )

    theme_choice = st.radio(
        "Theme",
        options=[
            "Vanilla",
            "NLP Sage",
            "Midnight NLP",
        ],
        index=[
            "Vanilla",
            "NLP Sage",
            "Midnight NLP",
        ].index(
            st.session_state["dashboard_theme"]
        ),
        horizontal=True,
        label_visibility="collapsed",
        key="theme_selector",
    )

    if theme_choice != st.session_state["dashboard_theme"]:

        st.session_state["dashboard_theme"] = theme_choice
        st.rerun()


# ================================================================
# DECORATIVE NLP BUBBLE FIELD
# ================================================================

st.markdown(
    """
    <style>
        .nlp-bubble-field {
            position: relative;
            height: 38px;
            margin: -2px 0 2px 0;
            overflow: hidden;
            pointer-events: none;
        }

        .nlp-bubble-field span {
            position: absolute;
            bottom: -14px;
            display: block;
            border-radius: 50%;
            border: 1px solid rgba(168, 90, 24, 0.16);
            background: rgba(255, 252, 247, 0.30);
            box-shadow: 0 3px 12px rgba(86, 48, 17, 0.04);
            animation: nlpFloat 11s linear infinite;
        }

        .nlp-bubble-field span:nth-child(1) {
            left: 5%;
            width: 8px;
            height: 8px;
            animation-delay: -2s;
        }

        .nlp-bubble-field span:nth-child(2) {
            left: 17%;
            width: 14px;
            height: 14px;
            animation-delay: -7s;
        }

        .nlp-bubble-field span:nth-child(3) {
            left: 31%;
            width: 6px;
            height: 6px;
            animation-delay: -4s;
        }

        .nlp-bubble-field span:nth-child(4) {
            left: 46%;
            width: 11px;
            height: 11px;
            animation-delay: -9s;
        }

        .nlp-bubble-field span:nth-child(5) {
            left: 62%;
            width: 7px;
            height: 7px;
            animation-delay: -5s;
        }

        .nlp-bubble-field span:nth-child(6) {
            left: 76%;
            width: 15px;
            height: 15px;
            animation-delay: -8s;
        }

        .nlp-bubble-field span:nth-child(7) {
            left: 90%;
            width: 6px;
            height: 6px;
            animation-delay: -1s;
        }

        @keyframes nlpFloat {
            0% {
                transform: translate3d(0, 22px, 0) scale(0.72);
                opacity: 0;
            }
            18% {
                opacity: 0.45;
            }
            50% {
                transform: translate3d(20px, -4px, 0) scale(1);
                opacity: 0.28;
            }
            82% {
                opacity: 0.18;
            }
            100% {
                transform: translate3d(-18px, -52px, 0) scale(1.15);
                opacity: 0;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            .nlp-bubble-field span {
                animation: none;
                opacity: 0.20;
                bottom: 12px;
            }
        }
    </style>

    <div class="nlp-bubble-field" aria-hidden="true">
        <span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# GLOBAL SLICERS
# ================================================================

st.divider()

slicer_title_col, slicer_reset_col = st.columns(
    [5, 1],
    gap="medium",
)

with slicer_title_col:

    st.subheader(
        "Global Slicers"
    )

    st.caption(
        "Only fields available in the CSV are shown."
    )

with slicer_reset_col:

    st.markdown(
        "<div class='reset-button-space'></div>",
        unsafe_allow_html=True,
    )

    if st.button(
        "↺ Reset Filters",
        key="reset_all_dashboard_filters",
        width="stretch",
    ):

        st.session_state["sentiment_slicer"] = "All"
        st.session_state["author_slicer"] = "All"
        st.session_state["tags_slicer"] = "All"
        st.session_state["lda_topic_slicer"] = "All"
        st.session_state["nmf_topic_slicer"] = "All"

        st.rerun()


# Build slicer values ONLY from real CSV columns.
def real_slicer_values(column_name):

    if not column_name or column_name not in df.columns:
        return []

    values = (
        df[column_name]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values.ne("")
        & values.ne("nan")
        & values.ne("None")
    ]

    return sorted(
        values.unique().tolist(),
        key=str.lower,
    )


sentiment_values = real_slicer_values(
    SENTIMENT_COL
)

author_values = real_slicer_values(
    AUTHOR_COL
)

tags_values = real_slicer_values(
    find_column(
        df,
        [
            "tags",
            "tag",
        ],
    )
)

lda_topic_values = real_slicer_values(
    LDA_TOPIC_COL
)

nmf_topic_values = real_slicer_values(
    NMF_TOPIC_COL
)


# Four/five slicers are shown depending on which real CSV columns
# are available. No placeholder or invented field is displayed.
slicer_defs = []

if SENTIMENT_COL:
    slicer_defs.append(
        (
            "Sentiment",
            sentiment_values,
            "sentiment_slicer",
        )
    )

if AUTHOR_COL:
    slicer_defs.append(
        (
            "Author",
            author_values,
            "author_slicer",
        )
    )

TAGS_COL = find_column(
    df,
    [
        "tags",
        "tag",
    ],
)

if TAGS_COL:
    slicer_defs.append(
        (
            "Tags",
            tags_values,
            "tags_slicer",
        )
    )

if LDA_TOPIC_COL:
    slicer_defs.append(
        (
            "LDA Topic",
            lda_topic_values,
            "lda_topic_slicer",
        )
    )

if NMF_TOPIC_COL:
    slicer_defs.append(
        (
            "NMF Topic",
            nmf_topic_values,
            "nmf_topic_slicer",
        )
    )


if slicer_defs:

    slicer_columns = st.columns(
        len(slicer_defs),
        gap="medium",
    )

    for col, (label, values, key) in zip(
        slicer_columns,
        slicer_defs,
    ):

        options = [
            "All",
            *values,
        ]

        current_value = st.session_state.get(
            key,
            "All",
        )

        # If the CSV changed and an old selection no longer
        # exists, automatically return that slicer to All.
        if current_value not in options:
            current_value = "All"
            st.session_state[key] = "All"

        with col:

            st.selectbox(
                label,
                options,
                index=options.index(
                    current_value
                ),
                key=key,
            )

else:

    st.info(
        "No supported categorical columns are available "
        "in the CSV for slicers."
    )


# ================================================================
# GLOBAL FILTER ENGINE
# ================================================================

filtered_df = df.copy()


if (
    SENTIMENT_COL
    and st.session_state.get(
        "sentiment_slicer",
        "All",
    ) != "All"
):

    filtered_df = filtered_df[
        filtered_df[SENTIMENT_COL]
        .fillna("")
        .astype(str)
        .str.strip()
        == st.session_state[
            "sentiment_slicer"
        ]
    ]


if (
    AUTHOR_COL
    and st.session_state.get(
        "author_slicer",
        "All",
    ) != "All"
):

    filtered_df = filtered_df[
        filtered_df[AUTHOR_COL]
        .fillna("")
        .astype(str)
        .str.strip()
        == st.session_state[
            "author_slicer"
        ]
    ]


if (
    TAGS_COL
    and st.session_state.get(
        "tags_slicer",
        "All",
    ) != "All"
):

    filtered_df = filtered_df[
        filtered_df[TAGS_COL]
        .fillna("")
        .astype(str)
        .str.strip()
        == st.session_state[
            "tags_slicer"
        ]
    ]


if (
    LDA_TOPIC_COL
    and st.session_state.get(
        "lda_topic_slicer",
        "All",
    ) != "All"
):

    filtered_df = filtered_df[
        filtered_df[LDA_TOPIC_COL]
        .fillna("")
        .astype(str)
        .str.strip()
        == st.session_state[
            "lda_topic_slicer"
        ]
    ]


if (
    NMF_TOPIC_COL
    and st.session_state.get(
        "nmf_topic_slicer",
        "All",
    ) != "All"
):

    filtered_df = filtered_df[
        filtered_df[NMF_TOPIC_COL]
        .fillna("")
        .astype(str)
        .str.strip()
        == st.session_state[
            "nmf_topic_slicer"
        ]
    ]


# ================================================================
# EXECUTIVE KPIs
# ================================================================

st.divider()

st.subheader(
    "Executive KPIs"
)

total_records = len(filtered_df)

positive_count = int(
    (
        filtered_df["_sentiment"]
        == "Positive"
    ).sum()
)

neutral_count = int(
    (
        filtered_df["_sentiment"]
        == "Neutral"
    ).sum()
)

negative_count = int(
    (
        filtered_df["_sentiment"]
        == "Negative"
    ).sum()
)

positive_pct = (
    positive_count / total_records * 100
    if total_records
    else 0
)

neutral_pct = (
    neutral_count / total_records * 100
    if total_records
    else 0
)

negative_pct = (
    negative_count / total_records * 100
    if total_records
    else 0
)

avg_compound = filtered_df[
    "_compound_score"
].mean()

unique_author_count = filtered_df[
    "_author"
].nunique()


# Equal-width KPI cards.
# The icon is INSIDE the card and sits beside the KPI content.
kpi_items = [
    (
        "📋",
        "Records",
        f"{total_records:,}",
        "",
        "records",
    ),
    (
        "😊",
        "Positive",
        f"{positive_count:,}",
        f"↑ {positive_pct:.1f}%",
        "positive",
    ),
    (
        "😐",
        "Neutral",
        f"{neutral_count:,}",
        f"↑ {neutral_pct:.1f}%",
        "neutral",
    ),
    (
        "☹️",
        "Negative",
        f"{negative_count:,}",
        f"↑ {negative_pct:.1f}%",
        "negative",
    ),
    (
        "🎯",
        "Avg Sentiment Score",
        (
            "—"
            if pd.isna(avg_compound)
            else f"{avg_compound:.3f}"
        ),
        "",
        "score",
    ),
    (
        "👥",
        "Unique Authors",
        f"{unique_author_count:,}",
        "",
        "authors",
    ),
]


kpi_cols = st.columns(
    [1, 1, 1, 1, 1, 1],
    gap="medium",
)

for col, item in zip(
    kpi_cols,
    kpi_items,
):

    icon, label, value, delta, kind = item

    with col:

        icon_html = (
            '<div class="kpi-icon-badge">'
            + icon
            + "</div>"
        )

        delta_html = ""

        if delta:

            delta_html = (
                '<span class="kpi-delta '
                + kind
                + '">'
                + delta
                + "</span>"
            )

        card_html = (
            '<div class="kpi-card">'
            + icon_html
            + '<div class="kpi-content">'
            + '<div class="kpi-label">'
            + label
            + "</div>"
            + '<div class="kpi-value-row">'
            + '<span class="kpi-value">'
            + value
            + "</span>"
            + delta_html
            + "</div>"
            + "</div>"
            + "</div>"
        )

        st.markdown(
            card_html,
            unsafe_allow_html=True,
        )


# ================================================================
# 9 VISUALS — 3 ROWS × 3 COLUMNS
# ================================================================

st.divider()

st.subheader(
    "NLP Intelligence"
)

st.caption(
    "Interactive analytics • all nine visuals respond to the global slicers."
)

plot_df = filtered_df.copy()

# Always derive these display fields from the project's canonical
# analytical columns. Never reference _processed_text/_original_text.
plot_df["_score"] = pd.to_numeric(
    plot_df["_compound_score"],
    errors="coerce",
)

plot_df["_words"] = pd.to_numeric(
    plot_df["_word_count"],
    errors="coerce",
)

plot_df["_characters"] = pd.to_numeric(
    plot_df["_char_count"],
    errors="coerce",
)

plot_df["_record_no"] = range(
    1,
    len(plot_df) + 1,
)

# ================================================================
# ROW 1 — 1 / 2 / 3
# ================================================================

v1, v2, v3 = st.columns(
    [1, 1, 1],
    gap="medium",
)


# ------------------------------------------------
# 1. SENTIMENT DISTRIBUTION
# ------------------------------------------------

with v1:

    sentiment_plot = (
        plot_df["_sentiment"]
        .value_counts()
        .reindex(
            [
                "Positive",
                "Neutral",
                "Negative",
            ],
            fill_value=0,
        )
        .reset_index()
    )

    sentiment_plot.columns = [
        "Sentiment",
        "Records",
    ]

    fig = px.pie(
        sentiment_plot,
        names="Sentiment",
        values="Records",
        hole=0.58,
        title="1. Sentiment Distribution",
        color="Sentiment",
        color_discrete_map={
            "Positive": POSITIVE,
            "Neutral": NEUTRAL,
            "Negative": NEGATIVE,
        },
    )

    fig.update_traces(
        texttemplate="%{percent:.1%}",
        textposition="inside",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Records: %{value:,}<br>"
            "Share: %{percent:.1%}"
            "<extra></extra>"
        ),
    )

    show_chart(
        style_chart(
            fig,
            330,
        )
    )


# ------------------------------------------------
# 2. SENTIMENT FLOW BY RECORD
# ------------------------------------------------

with v2:

    flow_df = plot_df[
        [
            "_record_no",
            "_score",
            "_sentiment",
        ]
    ].dropna(
        subset=["_score"]
    )

    if not flow_df.empty:

        fig = px.scatter(
            flow_df,
            x="_record_no",
            y="_score",
            color="_sentiment",
            title="2. Sentiment Flow by Record",
            color_discrete_map={
                "Positive": POSITIVE,
                "Neutral": NEUTRAL,
                "Negative": NEGATIVE,
            },
            hover_data={
                "_record_no": True,
                "_score": ":.3f",
            },
        )

        fig.update_traces(
            marker=dict(
                size=7,
                opacity=0.70,
            )
        )

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_width=1,
            line_color=MUTED,
        )

        fig.update_yaxes(
            title="VADER Compound Score",
            range=[-1, 1],
        )

        fig.update_xaxes(
            title="Record",
        )

        show_chart(
            style_chart(
                fig,
                330,
            )
        )

    else:

        empty_plot(
            "Sentiment score data unavailable."
        )


# ------------------------------------------------
# 3. TOP 10 KEYWORDS
# ------------------------------------------------

with v3:

    keyword_df = pd.DataFrame()

    if KEYWORDS_PATH.exists():

        try:
            keyword_df = pd.read_csv(
                KEYWORDS_PATH
            )
        except Exception:
            keyword_df = pd.DataFrame()

    if not keyword_df.empty:

        keyword_name_col = find_column(
            keyword_df,
            [
                "keyword",
                "term",
                "word",
                "feature",
            ],
        )

        keyword_score_col = find_column(
            keyword_df,
            [
                "score",
                "tfidf",
                "mean_tfidf",
                "importance",
            ],
        )

        keyword_count_col = find_column(
            keyword_df,
            [
                "frequency",
                "count",
                "document_frequency",
            ],
        )

        if keyword_name_col and keyword_score_col:

            plot_keywords = (
                keyword_df[
                    [
                        keyword_name_col,
                        keyword_score_col,
                    ]
                ]
                .dropna()
                .sort_values(
                    keyword_score_col,
                    ascending=False,
                )
                .head(10)
                .sort_values(
                    keyword_score_col,
                    ascending=True,
                )
            )

            plot_keywords.columns = [
                "Keyword",
                "Score",
            ]

            fig = px.bar(
                plot_keywords,
                x="Score",
                y="Keyword",
                orientation="h",
                title="3. Top 10 Keywords",
                text="Score",
            )

            fig.update_traces(
                marker_color=PURPLE,
                marker_line_width=0,
                texttemplate="%{text:.3f}",
                textposition="outside",
                cliponaxis=False,
            )

            show_chart(
                style_chart(
                    fig,
                    330,
                )
            )

        elif keyword_name_col and keyword_count_col:

            plot_keywords = (
                keyword_df[
                    [
                        keyword_name_col,
                        keyword_count_col,
                    ]
                ]
                .dropna()
                .sort_values(
                    keyword_count_col,
                    ascending=False,
                )
                .head(10)
                .sort_values(
                    keyword_count_col,
                    ascending=True,
                )
            )

            plot_keywords.columns = [
                "Keyword",
                "Frequency",
            ]

            fig = px.bar(
                plot_keywords,
                x="Frequency",
                y="Keyword",
                orientation="h",
                title="3. Top 10 Keywords",
                text="Frequency",
            )

            fig.update_traces(
                marker_color=PURPLE,
                marker_line_width=0,
                textposition="outside",
                cliponaxis=False,
            )

            show_chart(
                style_chart(
                    fig,
                    330,
                )
            )

        else:

            empty_plot(
                "Keyword columns were not recognized."
            )

    else:

        empty_plot(
            "Keyword analysis output not found."
        )


# ================================================================
# ROW 2 — 4 / 5 / 6
# ================================================================

v4, v5, v6 = st.columns(
    [1, 1, 1],
    gap="medium",
)


# ------------------------------------------------
# 4. SENTIMENT BY TOP AUTHOR
# ------------------------------------------------

with v4:

    author_sentiment = (
        plot_df
        .groupby(
            [
                "_author",
                "_sentiment",
            ]
        )
        .size()
        .reset_index(
            name="Records"
        )
    )

    top_authors = (
        plot_df["_author"]
        .value_counts()
        .head(5)
        .index
    )

    author_sentiment = author_sentiment[
        author_sentiment["_author"].isin(
            top_authors
        )
    ]

    if not author_sentiment.empty:

        fig = px.bar(
            author_sentiment,
            x="Records",
            y="_author",
            color="_sentiment",
            orientation="h",
            barmode="stack",
            title="4. Sentiment by Top Authors",
            color_discrete_map={
                "Positive": POSITIVE,
                "Neutral": NEUTRAL,
                "Negative": NEGATIVE,
            },
        )

        show_chart(
            style_chart(
                fig,
                330,
            )
        )

    else:

        empty_plot(
            "Author sentiment data unavailable."
        )


# ------------------------------------------------
# 5. SENTIMENT SCORE DISTRIBUTION
# ------------------------------------------------

with v5:

    score_data = plot_df[
        "_score"
    ].dropna()

    if not score_data.empty:

        fig = px.histogram(
            score_data,
            x="_score",
            nbins=22,
            title="5. Sentiment Score Distribution",
        )

        fig.update_traces(
            marker_color=BLUE,
            marker_line_width=0,
        )

        fig.add_vline(
            x=0,
            line_dash="dash",
            line_width=1,
            line_color=MUTED,
        )

        fig.update_xaxes(
            title="Compound Sentiment Score",
            range=[-1, 1],
        )

        fig.update_yaxes(
            title="Records",
        )

        show_chart(
            style_chart(
                fig,
                330,
            )
        )

    else:

        empty_plot(
            "Sentiment score data unavailable."
        )


# ------------------------------------------------
# 6. TOP 10 LDA TOPICS
# ------------------------------------------------

with v6:

    topic_plot = (
        plot_df["_topic"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .head(10)
        .reset_index()
    )

    topic_plot.columns = [
        "Topic",
        "Records",
    ]

    if not topic_plot.empty:

        fig = px.pie(
            topic_plot,
            names="Topic",
            values="Records",
            hole=0.54,
            title="6. Top 10 Topics",
        )

        fig.update_traces(
            texttemplate="%{percent:.1%}",
            textposition="inside",
        )

        show_chart(
            style_chart(
                fig,
                330,
            )
        )

    else:

        empty_plot(
            "LDA topic data unavailable."
        )


# ================================================================
# ROW 3 — 7 / 8 / 9
# ================================================================

v7, v8, v9 = st.columns(
    [1, 1, 1],
    gap="medium",
)


# ------------------------------------------------
# 7. AVERAGE SENTIMENT SCORE BY TOPIC
# ------------------------------------------------

with v7:

    topic_score = (
        plot_df
        .groupby(
            "_topic",
            dropna=False,
        )["_score"]
        .mean()
        .reset_index()
    )

    topic_score.columns = [
        "Topic",
        "Average Score",
    ]

    topic_score["Topic"] = (
        topic_score["Topic"]
        .fillna("Unknown")
        .astype(str)
    )

    topic_score = (
        topic_score
        .dropna(
            subset=[
                "Average Score",
            ]
        )
        .sort_values(
            "Average Score",
            ascending=True,
        )
        .tail(10)
    )

    if not topic_score.empty:

        fig = px.bar(
            topic_score,
            x="Average Score",
            y="Topic",
            orientation="h",
            title="7. Average Sentiment Score by Topic",
            text="Average Score",
        )

        fig.update_traces(
            marker_color=TEAL,
            marker_line_width=0,
            texttemplate="%{text:.3f}",
            textposition="outside",
            cliponaxis=False,
        )

        fig.add_vline(
            x=0,
            line_dash="dash",
            line_width=1,
            line_color=MUTED,
        )

        show_chart(
            style_chart(
                fig,
                330,
            )
        )

    else:

        empty_plot(
            "Topic sentiment data unavailable."
        )


# ------------------------------------------------
# 8. TEXT LENGTH VS SENTIMENT SCORE
# ------------------------------------------------

with v8:

    relationship_df = plot_df[
        [
            "_characters",
            "_score",
            "_sentiment",
            "_author",
        ]
    ].dropna(
        subset=[
            "_characters",
            "_score",
        ]
    )

    if not relationship_df.empty:

        relationship_plot = relationship_df

        if len(relationship_plot) > 1500:

            relationship_plot = (
                relationship_plot.sample(
                    1500,
                    random_state=42,
                )
            )

        fig = px.scatter(
            relationship_plot,
            x="_characters",
            y="_score",
            color="_sentiment",
            title="8. Text Length vs Sentiment Score",
            color_discrete_map={
                "Positive": POSITIVE,
                "Neutral": NEUTRAL,
                "Negative": NEGATIVE,
            },
            hover_data={
                "_author": True,
                "_characters": True,
                "_score": ":.3f",
            },
        )

        fig.update_traces(
            marker=dict(
                size=7,
                opacity=0.62,
            )
        )

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_width=1,
            line_color=MUTED,
        )

        fig.update_xaxes(
            title="Text Length (Characters)",
        )

        fig.update_yaxes(
            title="Compound Sentiment Score",
            range=[-1, 1],
        )

        show_chart(
            style_chart(
                fig,
                330,
            )
        )

    else:

        empty_plot(
            "Text-length relationship unavailable."
        )


# ------------------------------------------------
# 9. SENTIMENT PERCENTAGE TABLE — HEATMAP STYLE
# ------------------------------------------------
#
# This is intentionally the final visual: a compact percentage
# matrix showing the sentiment composition by top author.
# Each cell contains the actual percentage, making the table
# useful for comparison rather than decorative only.
# ------------------------------------------------

with v9:

    table_base = (
        plot_df
        .groupby(
            [
                "_author",
                "_sentiment",
            ]
        )
        .size()
        .reset_index(
            name="Records"
        )
    )

    table_top_authors = (
        plot_df["_author"]
        .value_counts()
        .head(8)
        .index
    )

    table_base = table_base[
        table_base["_author"].isin(
            table_top_authors
        )
    ]

    if not table_base.empty:

        table_pivot = (
            table_base
            .pivot_table(
                index="_author",
                columns="_sentiment",
                values="Records",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(
                columns=[
                    "Positive",
                    "Neutral",
                    "Negative",
                ],
                fill_value=0,
            )
        )

        table_pivot["Total"] = table_pivot.sum(
            axis=1
        )

        percentage_table = (
            table_pivot[
                [
                    "Positive",
                    "Neutral",
                    "Negative",
                ]
            ]
            .div(
                table_pivot["Total"],
                axis=0,
            )
            .mul(100)
        )

        percentage_table = (
            percentage_table
            .sort_values(
                "Positive",
                ascending=False,
            )
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=percentage_table.values,
                x=[
                    "Positive",
                    "Neutral",
                    "Negative",
                ],
                y=percentage_table.index.astype(str),
                text=np.array(
                    [
                        [
                            f"{value:.1f}%"
                            for value in row
                        ]
                        for row in percentage_table.values
                    ]
                ),
                texttemplate="%{text}",
                hovertemplate=(
                    "Author: %{y}<br>"
                    "Sentiment: %{x}<br>"
                    "Share: %{z:.1f}%"
                    "<extra></extra>"
                ),
                colorscale=[
                    [0.0, SURFACE_ALT],
                    [0.5, BORDER],
                    [1.0, ACCENT],
                ],
                zmin=0,
                zmax=100,
            )
        )

        fig.update_layout(
            title="9. Sentiment Share by Top Author (%)",
            height=330,
            margin=dict(
                l=12,
                r=12,
                t=48,
                b=24,
            ),
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            font=dict(
                family="Arial",
                color=TEXT,
                size=9,
            ),
        )

        show_chart(
            style_chart(
                fig,
                330,
            )
        )

    else:

        empty_plot(
            "Sentiment percentage table unavailable."
        )




# ================================================================
# ONE-LINE BIO
# ================================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        padding:8px 0 2px 0;
        font-size:0.86rem;
        color:#806F61;
    ">
        Turning Web Text into Actionable NLP Insights •
        Sentiment Analytics • Topic Intelligence • Data Storytelling
    </div>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# FOOTER
# ================================================================

st.divider()

footer_left, footer_right = st.columns(
    [5, 1],
    gap="large",
)

with footer_left:

    st.markdown(
        "### 👨‍💻 S Mohammed Kaif"
    )

    st.markdown(
        "**Data Science • Data Analytics • Machine Learning • AI • Python**"
    )

    github_col, linkedin_col = st.columns(
        [1, 1],
        gap="small",
    )

    with github_col:

        st.link_button(
            "🐙 GitHub",
            "https://github.com/Shaik-Mohammed-Kaif",
            width="content",
        )

    with linkedin_col:

        st.link_button(
            "💼 LinkedIn",
            "https://www.linkedin.com/in/s-mohammed-kaif-2a500a341/",
            width="content",
        )

    st.markdown(
        "> **Learn → Practice → Analyze → Build → Improve → Grow**"
    )


with footer_right:

    st.caption(
        "© 2026 S Mohammed Kaif"
    )
