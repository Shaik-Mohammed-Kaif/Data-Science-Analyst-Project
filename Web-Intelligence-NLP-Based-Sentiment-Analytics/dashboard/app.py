# ================================================================
# WEB INTELLIGENCE • NLP ANALYTICS
# TEXT INTELLIGENCE DASHBOARD
#
# Power BI inspired Streamlit dashboard
# Reference version updated:   
#   • Same cream canvas + multi-theme controls  
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
#   • Clickable theme buttons + prediction performance visuals
# ================================================================

from pathlib import Path
import re
import string

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# NLTK — same preprocessing family used in Notebook 03
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError:
    nltk = None
    word_tokenize = None
    stopwords = None
    WordNetLemmatizer = None


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
# MULTI THEME SYSTEM
# ================================================================

THEMES = {
    "Vanilla": {
        "bg": "#F7F1E7",
        "surface": "#FFFCF7",
        "surface_alt": "#FBF6ED",
        "border": "#E5D8C7",
        "ink": "#3A2417",
        "text": "#4D3A2B",
        "muted": "#806F61",
        "accent": "#A85A18",
        "accent_dark": "#71370D",
        "positive": "#4CAF50",
        "negative": "#E53935",
        "neutral": "#F5A900",
        "purple": "#7651A9",
        "blue": "#4B83C4",
        "teal": "#4C9188",
        "heatmap": [
            [0.00, "#F8E6D9"],
            [0.50, "#F2C46D"],
            [1.00, "#7DBB83"],
        ],
    },
    "NLP Sage": {
        "bg": "#EEF4F0",
        "surface": "#FAFCFA",
        "surface_alt": "#E8F0EB",
        "border": "#D0DED5",
        "ink": "#20332A",
        "text": "#385047",
        "muted": "#6B7D74",
        "accent": "#2E7D68",
        "accent_dark": "#1F5B4C",
        "positive": "#3F9B72",
        "negative": "#C45D59",
        "neutral": "#D59A2A",
        "purple": "#7357A6",
        "blue": "#4F83B5",
        "teal": "#398F88",
        "heatmap": [
            [0.00, "#F4D8D3"],
            [0.50, "#E8D58A"],
            [1.00, "#91C8A6"],
        ],
    },
    "Midnight NLP": {
        "bg": "#111719",
        "surface": "#182123",
        "surface_alt": "#202C2E",
        "border": "#344345",
        "ink": "#F1F4F2",
        "text": "#D8E1DE",
        "muted": "#A3B2AE",
        "accent": "#75B7AB",
        "accent_dark": "#4D8F85",
        "positive": "#72B894",
        "negative": "#D47A7A",
        "neutral": "#D2B56C",
        "purple": "#9C80CF",
        "blue": "#70A7D5",
        "teal": "#67B4AC",
        "heatmap": [
            [0.00, "#5A3434"],
            [0.50, "#826F38"],
            [1.00, "#35634B"],
        ],
    },
}

if "dashboard_theme" not in st.session_state:
    st.session_state["dashboard_theme"] = "Vanilla"

ACTIVE_THEME = st.session_state["dashboard_theme"]
THEME = THEMES[ACTIVE_THEME]

BG = THEME["bg"]
SURFACE = THEME["surface"]
SURFACE_ALT = THEME["surface_alt"]
BORDER = THEME["border"]

INK = THEME["ink"]
TEXT = THEME["text"]
MUTED = THEME["muted"]

ACCENT = THEME["accent"]
ACCENT_DARK = THEME["accent_dark"]

POSITIVE = THEME["positive"]
NEGATIVE = THEME["negative"]
NEUTRAL = THEME["neutral"]

PURPLE = THEME["purple"]
BLUE = THEME["blue"]
TEAL = THEME["teal"]

HEATMAP_COLORS = THEME["heatmap"]


# ================================================================
# FLOWING BACKGROUND — SUBTLE NLP INFO GRAPHIC
# ================================================================
#
# Decorative background only:
# • does not change the theme palette
# • does not change the footer
# • does not cover/intercept controls
# • keeps the dashboard content visually dominant
# ================================================================

st.markdown(
    """
    <style>
        /* Soft ambient background bubbles */
        .stApp {
            position: relative;
            overflow-x: hidden;
        }

        .stApp::before,
        .stApp::after {
            content: "";
            position: fixed;
            width: 420px;
            height: 420px;
            border-radius: 50%;
            pointer-events: none;
            z-index: 0;
            opacity: 0.16;
            filter: blur(2px);
            background:
                radial-gradient(
                    circle at 35% 35%,
                    rgba(168, 90, 24, 0.12) 0%,
                    rgba(168, 90, 24, 0.055) 34%,
                    rgba(168, 90, 24, 0.00) 72%
                );
        }

        .stApp::before {
            left: -145px;
            top: 8vh;
            animation: kaifBubbleOrbitA 24s ease-in-out infinite alternate;
        }

        .stApp::after {
            right: -160px;
            top: 48vh;
            animation: kaifBubbleOrbitB 30s ease-in-out infinite alternate;
        }

        /* Small floating information bubbles */
        .kaif-nlp-bubbles {
            position: fixed;
            inset: 0;
            overflow: hidden;
            pointer-events: none;
            z-index: 0;
        }

        .kaif-nlp-bubbles span {
            position: absolute;
            bottom: -90px;
            display: block;
            border-radius: 50%;
            border: 1px solid rgba(128, 111, 97, 0.13);
            background:
                radial-gradient(
                    circle at 32% 28%,
                    rgba(255, 255, 255, 0.24),
                    rgba(128, 111, 97, 0.025) 46%,
                    rgba(128, 111, 97, 0.00) 72%
                );
            box-shadow:
                0 0 28px rgba(128, 111, 97, 0.055);
            opacity: 0;
            animation: kaifNlpFlow 25s linear infinite;
        }

        .kaif-nlp-bubbles span:nth-child(1) {
            width: 58px;
            height: 58px;
            left: 5%;
            animation-duration: 27s;
            animation-delay: -4s;
        }

        .kaif-nlp-bubbles span:nth-child(2) {
            width: 34px;
            height: 34px;
            left: 17%;
            animation-duration: 21s;
            animation-delay: -13s;
        }

        .kaif-nlp-bubbles span:nth-child(3) {
            width: 82px;
            height: 82px;
            left: 31%;
            animation-duration: 31s;
            animation-delay: -18s;
        }

        .kaif-nlp-bubbles span:nth-child(4) {
            width: 42px;
            height: 42px;
            left: 48%;
            animation-duration: 24s;
            animation-delay: -7s;
        }

        .kaif-nlp-bubbles span:nth-child(5) {
            width: 68px;
            height: 68px;
            left: 64%;
            animation-duration: 29s;
            animation-delay: -21s;
        }

        .kaif-nlp-bubbles span:nth-child(6) {
            width: 30px;
            height: 30px;
            left: 78%;
            animation-duration: 20s;
            animation-delay: -10s;
        }

        .kaif-nlp-bubbles span:nth-child(7) {
            width: 94px;
            height: 94px;
            left: 89%;
            animation-duration: 34s;
            animation-delay: -26s;
        }

        @keyframes kaifNlpFlow {
            0% {
                transform:
                    translate3d(0, 0, 0)
                    scale(0.72);
                opacity: 0;
            }

            10% {
                opacity: 0.48;
            }

            35% {
                transform:
                    translate3d(34px, -28vh, 0)
                    scale(0.94);
            }

            62% {
                transform:
                    translate3d(-26px, -62vh, 0)
                    scale(1.05);
                opacity: 0.30;
            }

            82% {
                transform:
                    translate3d(42px, -82vh, 0)
                    scale(0.88);
                opacity: 0.18;
            }

            100% {
                transform:
                    translate3d(-18px, -112vh, 0)
                    scale(0.70);
                opacity: 0;
            }
        }

        @keyframes kaifBubbleOrbitA {
            0% {
                transform: translate3d(0, 0, 0) scale(0.92);
            }
            50% {
                transform: translate3d(85px, 55px, 0) scale(1.08);
            }
            100% {
                transform: translate3d(25px, 115px, 0) scale(0.98);
            }
        }

        @keyframes kaifBubbleOrbitB {
            0% {
                transform: translate3d(0, 0, 0) scale(1.04);
            }
            50% {
                transform: translate3d(-90px, -55px, 0) scale(0.90);
            }
            100% {
                transform: translate3d(-35px, -125px, 0) scale(1.08);
            }
        }

        /* Keep every Streamlit control/content above the animation */
        .main,
        .main .block-container,
        header,
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stSidebar"],
        [data-testid="stAppViewContainer"] {
            position: relative;
            z-index: 2;
        }

        /* Respect accessibility settings */
        @media (prefers-reduced-motion: reduce) {
            .stApp::before,
            .stApp::after,
            .kaif-nlp-bubbles span {
                animation: none !important;
            }
        }
    </style>

    <div class="kaif-nlp-bubbles" aria-hidden="true">
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
    </div>
    """,
    unsafe_allow_html=True,
)


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

        /* Executive KPI alignment */
        [data-testid="stMetric"] {{
            width: 100%;
            box-sizing: border-box;
        }}

        [data-testid="stMetricLabel"] {{
            min-height: 20px;
        }}

        /* Executive KPI: value + delta on the SAME horizontal row */
        [data-testid="stMetric"] {{
            width: 100%;
            box-sizing: border-box;
        }}

        [data-testid="stMetric"] > div {{
            display: grid !important;
            grid-template-columns: minmax(0, auto) max-content;
            grid-template-rows: auto auto;
            column-gap: 8px;
            row-gap: 2px;
            align-items: center;
            min-width: 0;
        }}

        [data-testid="stMetricLabel"] {{
            grid-column: 1 / -1;
            grid-row: 1;
            min-height: 20px;
        }}

        [data-testid="stMetricValue"] {{
            grid-column: 1;
            grid-row: 2;
            display: inline-flex !important;
            align-items: baseline;
            width: auto !important;
            min-width: 0;
            white-space: nowrap;
        }}

        [data-testid="stMetricDelta"] {{
            grid-column: 2;
            grid-row: 2;
            display: inline-flex !important;
            align-items: center;
            justify-self: start;
            width: auto !important;
            margin: 0 !important;
            padding: 0 !important;
            white-space: nowrap;
            position: static !important;
            font-size: 0.78rem !important;
        }}

        [data-testid="stMetricDelta"] > div {{
            display: inline-flex !important;
            align-items: center;
            margin: 0 !important;
            padding: 0 !important;
        }}

        [data-testid="stMetricValue"] + [data-testid="stMetricDelta"] {{
            margin-top: 0 !important;
        }}

        /* For KPI cards without a delta, keep the value naturally placed. */
        [data-testid="stMetric"]:not(:has([data-testid="stMetricDelta"])) 
        [data-testid="stMetricValue"] {{
            grid-column: 1 / -1;
        }}


        div[data-baseweb="select"] > div {{{{
            background: {{SURFACE}};
            border-color: {{BORDER}};
            border-radius: 7px;
        }}}}

        .stTextInput input,
        textarea {{{{
            background: {{SURFACE}} !important;
            color: {{TEXT}} !important;
            border-color: {{BORDER}} !important;
            border-radius: 7px !important;
        }}}}

        .stButton > button,
        .stDownloadButton > button {{{{
            background: {{SURFACE}};
            color: {{INK}};
            border: 1px solid {{BORDER}};
            border-radius: 7px;
            min-height: 38px;
        }}}}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{{{
            border-color: {{ACCENT}};
            color: {{ACCENT}};
        }}}}

        hr {{{{
            border-color: {{BORDER}};
        }}}}

        [data-testid="stDataFrame"] {{{{
            border: 1px solid {{BORDER}};
            border-radius: 7px;
        }}}}
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

MODEL_RESULTS_PATH = (
    BASE_DIR
    / "data"
    / "final"
    / "sentiment_analysis_results.csv"
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
# ANALYZER PREPROCESSING — MATCH NOTEBOOK 03
# ================================================================
#
# The sentiment model was trained on:
#   original_text
#       ↓
#   lowercase / URL / HTML / punctuation cleanup
#       ↓
#   tokenization
#       ↓
#   English stopword removal
#       ↓
#   lemmatization
#       ↓
#   processed_text
#       ↓
#   TF-IDF
#
# The previous analyzer sent raw text directly to the TF-IDF
# vectorizer. That creates a train/inference preprocessing mismatch
# and can make many new texts collapse to the same prediction.
# This helper now reproduces the training-side preprocessing.
# ================================================================

@st.cache_resource
def load_nlp_resources():
    """Load the same NLTK resources used by Notebook 03."""

    if nltk is None:
        return None, None, None

    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }

    for package, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass

    try:
        stopword_set = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        return stopword_set, lemmatizer, True
    except Exception:
        return None, None, False


def preprocess_for_model(text):
    """
    Reproduce the Notebook 03 preprocessing used before TF-IDF.

    Falls back to deterministic regex preprocessing if NLTK resources
    are unavailable, so the dashboard still remains usable.
    """

    if text is None or pd.isna(text):
        return ""

    text = str(text).lower()

    # Same normalization logic as Notebook 03.
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    text = re.sub(
        r"<.*?>",
        " ",
        text,
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation,
        )
    )

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:
        return ""

    stopword_set, lemmatizer, nltk_ready = load_nlp_resources()

    if nltk_ready:
        try:
            tokens = word_tokenize(text)

            filtered_tokens = [
                token
                for token in tokens
                if token not in stopword_set
            ]

            lemmatized_tokens = [
                lemmatizer.lemmatize(token)
                for token in filtered_tokens
            ]

            return " ".join(
                lemmatized_tokens
            )
        except Exception:
            pass

    # Deterministic fallback close to the training representation.
    return " ".join(
        token
        for token in text.split()
        if token
    )



@st.cache_data(show_spinner=False)
def compute_keyword_analysis(frame, text_column, top_n=15):
    """
    Compute TF-IDF keywords directly from the CURRENT filtered corpus.

    This is intentionally calculated at dashboard runtime rather than
    depending on a pre-generated CSV. Therefore every slicer selection
    changes the keyword results automatically.
    """
    if frame.empty or not text_column or text_column not in frame.columns:
        return pd.DataFrame(columns=["Keyword", "TF-IDF Score"])

    texts = (
        frame[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    texts = texts[texts.ne("")]

    if texts.empty:
        return pd.DataFrame(columns=["Keyword", "TF-IDF Score"])

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
            max_features=2000,
        )
        matrix = vectorizer.fit_transform(texts)

        scores = np.asarray(
            matrix.mean(axis=0)
        ).ravel()

        result = pd.DataFrame(
            {
                "Keyword": vectorizer.get_feature_names_out(),
                "TF-IDF Score": scores,
            }
        )

        return (
            result
            .sort_values(
                "TF-IDF Score",
                ascending=False,
            )
            .head(top_n)
            .reset_index(drop=True)
        )

    except ValueError:
        return pd.DataFrame(columns=["Keyword", "TF-IDF Score"])


@st.cache_data(show_spinner=False)
def compute_ngram_analysis(frame, text_column, ngram_range, top_n=15):
    """
    Compute bigrams/trigrams directly from the CURRENT filtered corpus.

    Counts are recalculated after slicers are applied so phrase analytics
    stay synchronized with the rest of the dashboard.
    """
    if frame.empty or not text_column or text_column not in frame.columns:
        return pd.DataFrame(columns=["Phrase", "Frequency"])

    texts = (
        frame[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    texts = texts[texts.ne("")]

    if texts.empty:
        return pd.DataFrame(columns=["Phrase", "Frequency"])

    try:
        vectorizer = CountVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=ngram_range,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        )

        matrix = vectorizer.fit_transform(texts)

        frequencies = np.asarray(
            matrix.sum(axis=0)
        ).ravel()

        result = pd.DataFrame(
            {
                "Phrase": vectorizer.get_feature_names_out(),
                "Frequency": frequencies.astype(int),
            }
        )

        return (
            result
            .sort_values(
                "Frequency",
                ascending=False,
            )
            .head(top_n)
            .reset_index(drop=True)
        )

    except ValueError:
        return pd.DataFrame(columns=["Phrase", "Frequency"])


def sentiment_color_label(label):
    """Return a Streamlit status treatment for a sentiment label."""

    normalized = normalize_sentiment(label)

    if normalized == "Positive":
        return "positive"

    if normalized == "Negative":
        return "negative"

    return "neutral"


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

head_main, head_controls = st.columns(
    [6.5, 2.5],
    gap="large",
)

with head_main:

    st.caption(
        "WEB INTELLIGENCE  •  NLP ANALYTICS"
    )

    st.title(
        "Text Intelligence Dashboard"
    )

    st.write(
        "Executive NLP analytics for sentiment, language patterns, "
        "authors, keywords, phrases and discovered topics."
    )


with head_controls:

    st.write("")

    ready_button = st.button(
        "●  ANALYTICS READY",
        key="analytics_ready_button",
        width="stretch",
        disabled=True,
    )

    st.caption(
        f"Active theme  •  {ACTIVE_THEME}"
    )

    theme_a, theme_b, theme_c = st.columns(
        3,
        gap="small",
    )

    with theme_a:

        if st.button(
            "Vanilla",
            key="theme_vanilla",
            width="stretch",
        ):

            st.session_state[
                "dashboard_theme"
            ] = "Vanilla"

            st.rerun()


    with theme_b:

        if st.button(
            "Sage",
            key="theme_sage",
            width="stretch",
        ):

            st.session_state[
                "dashboard_theme"
            ] = "NLP Sage"

            st.rerun()


    with theme_c:

        if st.button(
            "Midnight",
            key="theme_midnight",
            width="stretch",
        ):

            st.session_state[
                "dashboard_theme"
            ] = "Midnight NLP"

            st.rerun()


# GLOBAL SLICERS
# ================================================================

st.divider()

filter_title, filter_reset = st.columns(
    [8, 1]
)

with filter_title:

    st.subheader(
        "Global Slicers"
    )

    st.caption(
        "Click a dropdown slicer to instantly refresh every KPI, visual, insight and table below."
    )


with filter_reset:

    st.write("")

    st.write("")

    reset_clicked = st.button(
        "Reset",
        key="reset_filters",
        width="stretch",
    )


# ================================================================
# SLICER OPTIONS
# ================================================================

author_values = sorted(
    df["_author"]
    .dropna()
    .unique()
    .tolist()
)

sentiment_values = [
    value
    for value in [
        "Positive",
        "Neutral",
        "Negative",
    ]
    if value in df["_sentiment"].unique()
]

topic_values = sorted(
    df["_topic"]
    .dropna()
    .unique()
    .tolist(),
    key=str,
)


if "author_filter" not in st.session_state:

    st.session_state[
        "author_filter"
    ] = ["All"]


if "sentiment_filter" not in st.session_state:

    st.session_state[
        "sentiment_filter"
    ] = ["All"]


if "topic_filter" not in st.session_state:

    st.session_state[
        "topic_filter"
    ] = ["All"]


if reset_clicked:

    st.session_state[
        "author_filter"
    ] = ["All"]

    st.session_state[
        "sentiment_filter"
    ] = ["All"]

    st.session_state[
        "topic_filter"
    ] = ["All"]

    st.rerun()


# ================================================================
# HORIZONTAL POWER BI STYLE SLICERS
# ================================================================

slicer_author, slicer_sentiment, slicer_topic = st.columns(
    [1, 1, 1],
    gap="medium",
)


with slicer_author:

    selected_author = st.selectbox(
        "Author",
        options=["All"] + author_values,
        key="author_filter",
    )

    selected_authors = (
        []
        if selected_author == "All"
        else [selected_author]
    )


with slicer_sentiment:

    selected_sentiment = st.selectbox(
        "Sentiment",
        options=["All"] + sentiment_values,
        key="sentiment_filter",
    )

    selected_sentiments = (
        []
        if selected_sentiment == "All"
        else [selected_sentiment]
    )


with slicer_topic:

    selected_topic = st.selectbox(
        "Topic",
        options=["All"] + topic_values,
        key="topic_filter",
    )

    selected_topics = (
        []
        if selected_topic == "All"
        else [selected_topic]
    )


# ================================================================
# GLOBAL FILTER ENGINE
# ================================================================

filtered_df = df.copy()


if (
    selected_authors
    and "All" not in selected_authors
):

    filtered_df = filtered_df[
        filtered_df["_author"].isin(
            selected_authors
        )
    ]


if (
    selected_sentiments
    and "All" not in selected_sentiments
):

    filtered_df = filtered_df[
        filtered_df["_sentiment"].isin(
            selected_sentiments
        )
    ]


if (
    selected_topics
    and "All" not in selected_topics
):

    filtered_df = filtered_df[
        filtered_df["_topic"].isin(
            selected_topics
        )
    ]


if filtered_df.empty:

    st.warning(
        "No records match the current slicers. "
        "Press Reset or choose a different combination."
    )

    st.stop()


# ================================================================
# FILTER CONTEXT
# ================================================================

context1, context2, context3, context4 = st.columns(
    4,
    gap="medium",
)

with context1:

    st.caption(
        f"FILTERED CORPUS  •  {len(filtered_df):,}"
    )

with context2:

    st.caption(
        f"TOTAL CORPUS  •  {len(df):,}"
    )

with context3:

    st.caption(
        f"COVERAGE  •  {len(filtered_df) / len(df) * 100:.1f}%"
    )

with context4:

    st.caption(
        "CLICK A DROPDOWN  •  VISUALS AUTO-REFRESH"
    )

    active_filters = sum(
        [
            bool(
                selected_authors
                and "All" not in selected_authors
            ),
            bool(
                selected_sentiments
                and "All" not in selected_sentiments
            ),
            bool(
                selected_topics
                and "All" not in selected_topics
            ),
        ]
    )

    st.caption(
        f"ACTIVE SLICERS  •  {active_filters}"
    )


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
    positive_count
    / total_records
    * 100
)

neutral_pct = (
    neutral_count
    / total_records
    * 100
)

negative_pct = (
    negative_count
    / total_records
    * 100
)

avg_compound = (
    filtered_df["_compound_score"]
    .mean()
)

unique_author_count = (
    filtered_df["_author"]
    .nunique()
)


# Equal-width executive KPI grid
# Each KPI gets the exact same column width.
# Sentiment share is shown as the compact delta on the right side
# of the KPI value area, keeping the Power BI-style hierarchy clean.

k1, k2, k3, k4, k5, k6 = st.columns(
    [1, 1, 1, 1, 1, 1],
    gap="medium",
)


with k1:

    st.metric(
        label="Records",
        value=f"{total_records:,}",
        border=False,
    )


with k2:

    st.metric(
        label="Positive",
        value=f"{positive_count:,}",
        delta=f"{positive_pct:.1f}%",
        delta_color="normal",
        border=False,
    )


with k3:

    st.metric(
        label="Neutral",
        value=f"{neutral_count:,}",
        delta=f"{neutral_pct:.1f}%",
        delta_color="off",
        border=False,
    )


with k4:

    st.metric(
        label="Negative",
        value=f"{negative_count:,}",
        delta=f"{negative_pct:.1f}%",
        delta_color="inverse",
        border=False,
    )


with k5:

    st.metric(
        label="Avg Sentiment",
        value=(
            "—"
            if pd.isna(avg_compound)
            else f"{avg_compound:.3f}"
        ),
        border=False,
    )


with k6:

    st.metric(
        label="Unique Authors",
        value=f"{unique_author_count:,}",
        border=False,
    )


# ================================================================
# FILTER-AWARE TEXT INTELLIGENCE TABLES
# ================================================================
#
# These are calculated from filtered_df, not only from saved CSV files.
# This fixes Streamlit deployment cases where outputs/tables files are
# absent from GitHub and also makes keywords/phrases respond to slicers.
# ================================================================

ANALYSIS_TEXT_COL = (
    PROCESSED_TEXT_COL
    if PROCESSED_TEXT_COL
    else TEXT_COL
)

keyword_df = compute_keyword_analysis(
    filtered_df,
    ANALYSIS_TEXT_COL,
    top_n=15,
)

bigram_df = compute_ngram_analysis(
    filtered_df,
    ANALYSIS_TEXT_COL,
    ngram_range=(2, 2),
    top_n=15,
)

trigram_df = compute_ngram_analysis(
    filtered_df,
    ANALYSIS_TEXT_COL,
    ngram_range=(3, 3),
    top_n=15,
)


# ================================================================
# NLP INTELLIGENCE — 8 VISUALS
# ================================================================

st.divider()

st.subheader(
    "NLP Intelligence — 8 Visuals"
)


# ================================================================
# ROW 1 — VISUALS 1–4
# ================================================================

v1, v2, v3, v4 = st.columns(
    4,
    gap="medium",
)


# ------------------------------------------------
# VISUAL 1 — SENTIMENT DISTRIBUTION
# ------------------------------------------------

with v1:

    sentiment_plot = (
        filtered_df["_sentiment"]
        .value_counts()
        .reindex(
            [
                "Positive",
                "Neutral",
                "Negative",
            ]
        )
        .fillna(0)
        .reset_index()
    )

    sentiment_plot.columns = [
        "Sentiment",
        "Count",
    ]

    fig = px.pie(
        sentiment_plot,
        names="Sentiment",
        values="Count",
        hole=0.61,
        title="1. Sentiment Distribution",
        color="Sentiment",
        color_discrete_map={
            "Positive": POSITIVE,
            "Neutral": NEUTRAL,
            "Negative": NEGATIVE,
        },
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Records: %{value}<br>"
            "Share: %{percent}"
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
# VISUAL 2 — REAL VADER SCORE DISTRIBUTION
# ------------------------------------------------
#
# FIX:
# Previous code searched for "vader_compound".
# The actual project column is "compound_score".
# This chart therefore now uses _compound_score.
# ------------------------------------------------

with v2:

    score_data = (
        filtered_df["_compound_score"]
        .dropna()
    )

    if not score_data.empty:

        fig = px.histogram(
            x=score_data,
            nbins=22,
            title="2. Sentiment Score Distribution",
            labels={
                "x": "VADER Compound Score",
                "y": "Records",
            },
        )

        fig.update_traces(
            marker_color=ACCENT,
            marker_line_color=SURFACE,
            marker_line_width=0.5,
            hovertemplate=(
                "Score: %{x:.3f}<br>"
                "Records: %{y}"
                "<extra></extra>"
            ),
        )

        fig.add_vline(
            x=0,
            line_width=1.5,
            line_dash="dash",
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
            "VADER compound_score is not available."
        )


# ------------------------------------------------
# VISUAL 3 — TOP AUTHORS
# ------------------------------------------------

with v3:

    author_plot = (
        filtered_df["_author"]
        .value_counts()
        .head(8)
        .sort_values()
        .reset_index()
    )

    author_plot.columns = [
        "Author",
        "Records",
    ]

    fig = px.bar(
        author_plot,
        x="Records",
        y="Author",
        orientation="h",
        title="3. Top Authors by Records",
        text="Records",
    )

    fig.update_traces(
        marker_color=ACCENT,
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


# ------------------------------------------------
# VISUAL 4 — TOP TF-IDF KEYWORDS
# ------------------------------------------------
#
# DEPLOYMENT FIX:
# Keywords are computed from filtered_df at runtime.
# The dashboard no longer depends on final_top_keywords.csv.
# This means:
#   • deployment works even when outputs/tables is not committed
#   • Author/Sentiment/Topic slicers refresh this visual
#   • local and Streamlit Cloud behavior stays consistent
# ------------------------------------------------

with v4:

    if not keyword_df.empty:

        plot_keywords = (
            keyword_df
            .head(8)
            .sort_values(
                "TF-IDF Score"
            )
        )

        fig = px.bar(
            plot_keywords,
            x="TF-IDF Score",
            y="Keyword",
            orientation="h",
            title="4. Top Keywords (TF-IDF)",
            text="TF-IDF Score",
        )

        fig.update_traces(
            marker_color=ACCENT,
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

    else:

        empty_plot(
            "No keyword terms are available for the current filters."
        )


# ================================================================
# ROW 2 — VISUALS 5–8
# ================================================================

v5, v6, v7, v8 = st.columns(
    4,
    gap="medium",
)


# ------------------------------------------------
# VISUAL 5 — TOP BIGRAMS
# ------------------------------------------------
#
# DEPLOYMENT FIX:
# Bigrams are computed from filtered_df at runtime instead of relying
# on top_bigrams.csv. This keeps the visual slicer-aware.
# ------------------------------------------------

with v5:

    if not bigram_df.empty:

        plot_bigrams = (
            bigram_df
            .head(7)
            .sort_values(
                "Frequency"
            )
        )

        fig = px.bar(
            plot_bigrams,
            x="Frequency",
            y="Phrase",
            orientation="h",
            title="5. Top Bigrams",
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
            "No bigrams are available for the current filters."
        )


# ------------------------------------------------
# VISUAL 6 — REAL LDA TOPIC DISTRIBUTION
# ------------------------------------------------
#
# FIX:
# Previous code searched for "dominant_topic".
# Notebook 06 saves "lda_dominant_topic".
# ------------------------------------------------

with v6:

    topic_plot = (
        filtered_df["_topic"]
        .value_counts()
        .sort_values()
        .tail(8)
        .reset_index()
    )

    topic_plot.columns = [
        "Topic",
        "Records",
    ]

    if not topic_plot.empty:

        fig = px.bar(
            topic_plot,
            x="Records",
            y="Topic",
            orientation="h",
            title="6. Topic Model Distribution",
            text="Records",
        )

        fig.update_traces(
            marker_color=BLUE,
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
            "LDA topic data is unavailable."
        )


# ------------------------------------------------
# VISUAL 7 — TOPIC × SENTIMENT
# ------------------------------------------------

with v7:

    topic_sentiment = (
        filtered_df
        .groupby(
            [
                "_topic",
                "_sentiment",
            ]
        )
        .size()
        .reset_index(
            name="Count"
        )
    )

    if not topic_sentiment.empty:

        pivot = (
            topic_sentiment
            .pivot(
                index="_topic",
                columns="_sentiment",
                values="Count",
            )
            .fillna(0)
        )

        for sentiment in [
            "Negative",
            "Neutral",
            "Positive",
        ]:

            if sentiment not in pivot.columns:

                pivot[sentiment] = 0


        pivot = pivot[
            [
                "Negative",
                "Neutral",
                "Positive",
            ]
        ]


        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                text=pivot.values,
                texttemplate="%{text}",
                hovertemplate=(
                    "Topic: %{y}<br>"
                    "Sentiment: %{x}<br>"
                    "Records: %{z}"
                    "<extra></extra>"
                ),
                colorscale=[
                    [0.00, "#F8E6D9"],
                    [0.50, "#F2C46D"],
                    [1.00, "#7DBB83"],
                ],
            )
        )

        fig.update_layout(
            title="7. Topic × Sentiment Heatmap",
            height=330,
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
                size=9,
            ),
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True,
            },
        )

    else:

        empty_plot(
            "Topic × sentiment data unavailable."
        )


# ------------------------------------------------
# VISUAL 8 — TEXT LENGTH × SENTIMENT
# ------------------------------------------------
#
# FIX:
# This uses real:
#   processed_text → word count
#   compound_score → sentiment score
#
# Therefore it does not depend on a nonexistent
# "vader_compound" or "dominant_topic" field.
# ------------------------------------------------

with v8:

    relationship_df = (
        filtered_df[
            [
                "_word_count",
                "_compound_score",
                "_sentiment",
            ]
        ]
        .dropna()
    )

    if len(relationship_df) >= 2:

        # Keep the chart readable on very large datasets.
        # No rows are removed when the current project is small.
        if len(relationship_df) > 1500:

            relationship_plot = (
                relationship_df
                .sample(
                    1500,
                    random_state=42,
                )
            )

        else:

            relationship_plot = (
                relationship_df
            )


        fig = px.scatter(
            relationship_plot,
            x="_word_count",
            y="_compound_score",
            color="_sentiment",
            title="8. Text Length × Sentiment",
            labels={
                "_word_count": "Words",
                "_compound_score": "VADER Compound",
            },
            color_discrete_map={
                "Positive": POSITIVE,
                "Neutral": NEUTRAL,
                "Negative": NEGATIVE,
            },
            opacity=0.70,
        )

        fig.add_hline(
            y=0,
            line_width=1,
            line_dash="dash",
            line_color=MUTED,
        )

        fig.update_traces(
            marker=dict(
                size=7,
            ),
            hovertemplate=(
                "Words: %{x}<br>"
                "Compound: %{y:.3f}"
                "<extra></extra>"
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
            "Not enough text-length and sentiment data "
            "for the relationship view."
        )


# ================================================================
# KEYWORD & PHRASE INTELLIGENCE
# ================================================================

st.divider()

st.subheader(
    "Keyword & Phrase Intelligence"
)

st.caption(
    "Runtime NLP analysis of the current filtered corpus • "
    "updates automatically with every slicer selection."
)

kp1, kp2 = st.columns(
    2,
    gap="medium",
)


with kp1:

    st.markdown(
        "#### Top Keywords"
    )

    if not keyword_df.empty:

        keyword_display = keyword_df.copy()

        keyword_display["TF-IDF Score"] = (
            keyword_display["TF-IDF Score"]
            .round(4)
        )

        st.dataframe(
            keyword_display,
            width="stretch",
            hide_index=True,
            height=280,
        )

    else:

        empty_plot(
            "No keyword terms are available for the current filters."
        )


with kp2:

    st.markdown(
        "#### Important Phrases"
    )

    phrase_tables = []

    if not bigram_df.empty:

        bigram_display = bigram_df.head(10).copy()

        bigram_display["Phrase Type"] = (
            "Bigram"
        )

        phrase_tables.append(
            bigram_display[
                [
                    "Phrase",
                    "Frequency",
                    "Phrase Type",
                ]
            ]
        )

    if not trigram_df.empty:

        trigram_display = trigram_df.head(10).copy()

        trigram_display["Phrase Type"] = (
            "Trigram"
        )

        phrase_tables.append(
            trigram_display[
                [
                    "Phrase",
                    "Frequency",
                    "Phrase Type",
                ]
            ]
        )

    if phrase_tables:

        phrases_display = pd.concat(
            phrase_tables,
            ignore_index=True,
        )

        st.dataframe(
            phrases_display,
            width="stretch",
            hide_index=True,
            height=280,
        )

    else:

        empty_plot(
            "No phrases are available for the current filters."
        )


# ================================================================
# DECISION INSIGHTS
# ================================================================

st.divider()

st.subheader(
    "Decision Insights"
)

dominant_sentiment = (
    filtered_df["_sentiment"]
    .value_counts()
    .idxmax()
)

leading_author = (
    filtered_df["_author"]
    .value_counts()
    .idxmax()
)

leading_topic = (
    filtered_df["_topic"]
    .value_counts()
    .idxmax()
)

average_words = (
    filtered_df["_word_count"]
    .mean()
)


d1, d2, d3, d4 = st.columns(
    4,
    gap="medium",
)


with d1:

    st.metric(
        "Dominant Sentiment",
        dominant_sentiment,
    )

    st.caption(
        "Largest sentiment category"
    )


with d2:

    st.metric(
        "Leading Author",
        leading_author,
    )

    st.caption(
        "Most represented author"
    )


with d3:

    st.metric(
        "Leading Topic",
        leading_topic,
    )

    st.caption(
        "Largest LDA topic cluster"
    )


with d4:

    st.metric(
        "Average Words",
        (
            "—"
            if pd.isna(average_words)
            else f"{average_words:.1f}"
        ),
    )

    st.caption(
        "Average document length"
    )


st.write(
    f"Current context: **{len(filtered_df):,} records** • "
    f"dominant sentiment: **{dominant_sentiment.lower()}** • "
    f"leading author: **{leading_author}** • "
    f"leading topic: **{leading_topic}**."
)


# ================================================================
# NLP TEXT ANALYZER
# ================================================================

st.divider()

st.subheader(
    "NLP Text Analyzer"
)

st.caption(
    "Enter new text and run it through the same NLP preprocessing "
    "and TF-IDF pipeline used during model training."
)


analyzer_text = st.text_area(
    "Analyze new text",
    placeholder=(
        "Example: I absolutely love this amazing experience."
    ),
    height=105,
)


if st.button(
    "Analyze Text",
    type="primary",
    key="run_text_analysis",
):

    if not analyzer_text.strip():

        st.warning(
            "Please enter text first."
        )

    elif (
        not MODEL_PATH.exists()
        or not VECTORIZER_PATH.exists()
    ):

        st.warning(
            "Model/vectorizer artifact not found."
        )

    else:

        try:

            # ----------------------------------------------------
            # STEP 1 — SAME PREPROCESSING AS NOTEBOOK 03
            # ----------------------------------------------------

            model_text = preprocess_for_model(
                analyzer_text
            )

            if not model_text.strip():

                st.warning(
                    "The text becomes empty after NLP preprocessing. "
                    "Please enter a little more meaningful text."
                )

                st.stop()


            # ----------------------------------------------------
            # STEP 2 — LOAD SAVED MODEL + TF-IDF
            # ----------------------------------------------------

            model = joblib.load(
                MODEL_PATH
            )

            vectorizer = joblib.load(
                VECTORIZER_PATH
            )


            # ----------------------------------------------------
            # STEP 3 — TRANSFORM PREPROCESSED TEXT
            # ----------------------------------------------------

            transformed = vectorizer.transform(
                [model_text]
            )


            # Check whether the input contains learned vocabulary.
            non_zero_features = int(
                transformed.nnz
            )


            # ----------------------------------------------------
            # STEP 4 — MODEL PREDICTION
            # ----------------------------------------------------

            prediction_raw = model.predict(
                transformed
            )[0]

            prediction = normalize_sentiment(
                prediction_raw
            )


            # ----------------------------------------------------
            # STEP 5 — MODEL SIGNAL / CONFIDENCE
            # ----------------------------------------------------

            model_signal = np.nan

            if hasattr(
                model,
                "predict_proba",
            ):

                probabilities = model.predict_proba(
                    transformed
                )

                class_probabilities = {
                    normalize_sentiment(label): float(prob)
                    for label, prob in zip(
                        model.classes_,
                        probabilities[0],
                    )
                }

                model_signal = float(
                    probabilities[0].max()
                )

            elif hasattr(
                model,
                "decision_function",
            ):

                decision_value = model.decision_function(
                    transformed
                )

                if np.ndim(decision_value) == 1:

                    model_signal = float(
                        np.abs(
                            decision_value[0]
                        )
                    )

                else:

                    model_signal = float(
                        np.max(
                            decision_value[0]
                        )
                    )

                class_probabilities = {}


            # ----------------------------------------------------
            # STEP 6 — VADER CROSS-CHECK
            # ----------------------------------------------------
            #
            # VADER is used here as an additional reference signal.
            # The trained ML model remains the primary prediction.
            # ----------------------------------------------------

            vader_compound = np.nan

            try:

                from nltk.sentiment import SentimentIntensityAnalyzer

                if nltk is not None:

                    try:
                        nltk.data.find(
                            "sentiment/vader_lexicon"
                        )
                    except LookupError:
                        nltk.download(
                            "vader_lexicon",
                            quiet=True,
                        )

                    sia_analyzer = (
                        SentimentIntensityAnalyzer()
                    )

                    vader_compound = float(
                        sia_analyzer
                        .polarity_scores(
                            analyzer_text
                        )["compound"]
                    )

            except Exception:
                pass


            if not pd.isna(vader_compound):

                if vader_compound >= 0.05:
                    vader_prediction = "Positive"
                elif vader_compound <= -0.05:
                    vader_prediction = "Negative"
                else:
                    vader_prediction = "Neutral"

            else:

                vader_prediction = "Unavailable"


            # ----------------------------------------------------
            # STEP 7 — RESULT DISPLAY
            # ----------------------------------------------------

            st.markdown(
                "#### Prediction Result"
            )

            result_left, result_right = st.columns(
                [1, 1],
                gap="medium",
            )


            with result_left:

                if prediction == "Positive":

                    st.success(
                        f"ML Prediction: {prediction}"
                    )

                elif prediction == "Negative":

                    st.error(
                        f"ML Prediction: {prediction}"
                    )

                else:

                    st.info(
                        f"ML Prediction: {prediction}"
                    )


            with result_right:

                if vader_prediction == "Positive":

                    st.success(
                        f"VADER Reference: {vader_prediction}"
                    )

                elif vader_prediction == "Negative":

                    st.error(
                        f"VADER Reference: {vader_prediction}"
                    )

                elif vader_prediction == "Neutral":

                    st.info(
                        f"VADER Reference: {vader_prediction}"
                    )

                else:

                    st.caption(
                        "VADER Reference: unavailable"
                    )


            analyzer_metric1, analyzer_metric2, analyzer_metric3 = st.columns(
                3,
                gap="medium",
            )


            with analyzer_metric1:

                if not pd.isna(model_signal):

                    if hasattr(
                        model,
                        "predict_proba",
                    ):

                        st.metric(
                            "Model Confidence",
                            f"{model_signal * 100:.1f}%",
                        )

                    else:

                        st.metric(
                            "Decision Signal",
                            f"{model_signal:.3f}",
                        )

                else:

                    st.metric(
                        "Model Signal",
                        "Available",
                    )


            with analyzer_metric2:

                st.metric(
                    "Vocabulary Matches",
                    f"{non_zero_features:,}",
                )


            with analyzer_metric3:

                st.metric(
                    "Text Length",
                    f"{len(analyzer_text.split()):,} words",
                )


            # ----------------------------------------------------
            # MODEL CLASS PROBABILITY / SIGNAL PLOT
            # ----------------------------------------------------

            if class_probabilities:

                probability_plot = pd.DataFrame(
                    {
                        "Sentiment": list(
                            class_probabilities.keys()
                        ),
                        "Probability": list(
                            class_probabilities.values()
                        ),
                    }
                )

                probability_plot = (
                    probability_plot
                    .sort_values(
                        "Probability",
                        ascending=True,
                    )
                )

                fig = px.bar(
                    probability_plot,
                    x="Probability",
                    y="Sentiment",
                    orientation="h",
                    title="Model Prediction Profile",
                    text="Probability",
                    range_x=[0, 1],
                )

                fig.update_traces(
                    marker_color=ACCENT,
                    marker_line_width=0,
                    texttemplate="%{text:.1%}",
                    textposition="outside",
                    cliponaxis=False,
                )

                show_chart(
                    style_chart(
                        fig,
                        270,
                    )
                )

            else:

                signal_plot = pd.DataFrame(
                    {
                        "Signal": [
                            "Model Decision Signal"
                        ],
                        "Value": [
                            model_signal
                        ],
                    }
                )

                fig = px.bar(
                    signal_plot,
                    x="Signal",
                    y="Value",
                    title="Model Decision Signal",
                    text="Value",
                )

                fig.update_traces(
                    marker_color=ACCENT,
                    marker_line_width=0,
                    texttemplate="%{text:.3f}",
                    textposition="outside",
                    cliponaxis=False,
                )

                show_chart(
                    style_chart(
                        fig,
                        270,
                    )
                )


            # ----------------------------------------------------
            # TRANSPARENCY NOTE
            # ----------------------------------------------------

            if non_zero_features == 0:

                st.warning(
                    "This text contains no terms from the model's learned "
                    "TF-IDF vocabulary. The classifier therefore has very "
                    "little lexical evidence and may fall back toward its "
                    "learned class bias."
                )

            if (
                vader_prediction != "Unavailable"
                and vader_prediction != prediction
            ):

                st.caption(
                    f"Model/VADER difference: ML predicts "
                    f"**{prediction}**, while VADER gives "
                    f"**{vader_prediction}** "
                    f"(compound {vader_compound:.3f})."
                )

            else:

                st.caption(
                    "The ML prediction is the primary dashboard result; "
                    "VADER is shown only as a reference signal."
                )


        except Exception as exc:

            st.error(
                f"Prediction failed: {exc}"
            )


# ================================================================
# MODEL PERFORMANCE — PREDICTION QUALITY
# ================================================================
#
# This section evaluates the saved best model on the same held-out
# 20% stratified split used in Notebook 04 (random_state=42).
#
# These are NOT human-ground-truth metrics:
# the target labels are VADER-generated weak labels.
# ================================================================

st.divider()

st.subheader(
    "Prediction Performance"
)

st.caption(
    "Held-out evaluation of the saved sentiment model against "
    "VADER-generated weak labels • 20% stratified test split."
)

performance_ready = (
    MODEL_PATH.exists()
    and VECTORIZER_PATH.exists()
    and TEXT_COL is not None
    and SENTIMENT_COL is not None
)

if performance_ready:

    try:

        performance_text = (
            df[TEXT_COL]
            .fillna("")
            .astype(str)
        )

        performance_target = (
            df[SENTIMENT_COL]
            .apply(normalize_sentiment)
        )

        valid_mask = (
            performance_text.str.strip().ne("")
            & performance_target.ne("Unknown")
        )

        X_perf = performance_text[valid_mask]
        y_perf = performance_target[valid_mask]

        if (
            len(X_perf) >= 10
            and y_perf.nunique() >= 2
        ):

            X_train_perf, X_test_perf, y_train_perf, y_test_perf = (
                train_test_split(
                    X_perf,
                    y_perf,
                    test_size=0.20,
                    random_state=42,
                    stratify=y_perf,
                )
            )

            perf_model = joblib.load(
                MODEL_PATH
            )

            perf_vectorizer = joblib.load(
                VECTORIZER_PATH
            )

            X_test_matrix = (
                perf_vectorizer.transform(
                    X_test_perf
                )
            )

            y_pred_perf = perf_model.predict(
                X_test_matrix
            )

            accuracy = accuracy_score(
                y_test_perf,
                y_pred_perf,
            )

            precision = precision_score(
                y_test_perf,
                y_pred_perf,
                average="weighted",
                zero_division=0,
            )

            recall = recall_score(
                y_test_perf,
                y_pred_perf,
                average="weighted",
                zero_division=0,
            )

            f1 = f1_score(
                y_test_perf,
                y_pred_perf,
                average="weighted",
                zero_division=0,
            )

            p1, p2, p3, p4 = st.columns(
                4,
                gap="medium",
            )

            with p1:

                st.metric(
                    "Accuracy",
                    f"{accuracy * 100:.1f}%",
                )

            with p2:

                st.metric(
                    "Weighted Precision",
                    f"{precision * 100:.1f}%",
                )

            with p3:

                st.metric(
                    "Weighted Recall",
                    f"{recall * 100:.1f}%",
                )

            with p4:

                st.metric(
                    "Weighted F1",
                    f"{f1 * 100:.1f}%",
                )


            pc1, pc2 = st.columns(
                2,
                gap="medium",
            )


            # ----------------------------------------------------
            # MODEL QUALITY VISUAL 1 — METRIC COMPARISON
            # ----------------------------------------------------

            with pc1:

                metric_plot = pd.DataFrame(
                    {
                        "Metric": [
                            "Accuracy",
                            "Weighted Precision",
                            "Weighted Recall",
                            "Weighted F1",
                        ],
                        "Score": [
                            accuracy,
                            precision,
                            recall,
                            f1,
                        ],
                    }
                ).sort_values(
                    "Score",
                    ascending=True,
                )

                fig = px.bar(
                    metric_plot,
                    x="Score",
                    y="Metric",
                    orientation="h",
                    title="Model Quality Profile",
                    text="Score",
                    range_x=[0, 1],
                )

                fig.update_traces(
                    marker_color=ACCENT,
                    marker_line_width=0,
                    texttemplate="%{text:.1%}",
                    textposition="outside",
                    cliponaxis=False,
                )

                show_chart(
                    style_chart(
                        fig,
                        315,
                    )
                )


            # ----------------------------------------------------
            # MODEL QUALITY VISUAL 2 — CONFUSION MATRIX
            # ----------------------------------------------------

            with pc2:

                labels = [
                    "Positive",
                    "Neutral",
                    "Negative",
                ]

                cm = confusion_matrix(
                    y_test_perf,
                    y_pred_perf,
                    labels=labels,
                )

                cm_row_sum = cm.sum(
                    axis=1,
                    keepdims=True,
                )

                cm_pct = np.divide(
                    cm,
                    cm_row_sum,
                    out=np.zeros_like(
                        cm,
                        dtype=float,
                    ),
                    where=cm_row_sum != 0,
                )

                cm_text = np.array(
                    [
                        [
                            f"{cm[r, c]}<br>{cm_pct[r, c]:.0%}"
                            for c in range(cm.shape[1])
                        ]
                        for r in range(cm.shape[0])
                    ]
                )

                fig = go.Figure(
                    data=go.Heatmap(
                        z=cm_pct,
                        x=[
                            "Positive",
                            "Neutral",
                            "Negative",
                        ],
                        y=[
                            "Positive",
                            "Neutral",
                            "Negative",
                        ],
                        text=cm_text,
                        texttemplate="%{text}",
                        hovertemplate=(
                            "Actual: %{y}<br>"
                            "Predicted: %{x}<br>"
                            "Records: %{customdata}<br>"
                            "Row Share: %{z:.1%}"
                            "<extra></extra>"
                        ),
                        customdata=cm,
                        colorscale=HEATMAP_COLORS,
                        zmin=0,
                        zmax=1,
                    )
                )

                fig.update_layout(
                    title="Confusion Matrix — Count + Row %",
                    height=315,
                    margin=dict(
                        l=15,
                        r=15,
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

                st.plotly_chart(
                    fig,
                    width="stretch",
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                    },
                )


            st.caption(
                f"Evaluation records: {len(y_test_perf):,} • "
                f"Best saved model: {type(perf_model).__name__} • "
                "Metrics reflect agreement with VADER weak labels."
            )

        else:

            st.info(
                "Not enough labeled records are available for "
                "a stratified held-out model evaluation."
            )

    except Exception as exc:

        st.warning(
            f"Model performance evaluation could not be loaded: {exc}"
        )

else:

    st.info(
        "Saved model, vectorizer or labeled text is unavailable."
    )


# ================================================================
# FILTERED DATA EXPLORER
# ================================================================

st.divider()

st.subheader(
    "Filtered Data Explorer"
)

explorer_df = (
    filtered_df
    .drop(
        columns=[
            "_author",
            "_sentiment",
            "_compound_score",
            "_lda_topic_raw",
            "_topic",
            "_topic_score",
            "_word_count",
            "_char_count",
        ],
        errors="ignore",
    )
    .copy()
)


search_value = st.text_input(
    "Search current filter context",
    placeholder=(
        "Search text, author, topic or any visible field..."
    ),
)


if search_value.strip():

    search_term = (
        search_value
        .strip()
        .lower()
    )

    search_mask = pd.Series(
        False,
        index=explorer_df.index,
    )

    for column in explorer_df.columns:

        try:

            search_mask |= (
                explorer_df[column]
                .astype(str)
                .str.lower()
                .str.contains(
                    search_term,
                    na=False,
                )
            )

        except Exception:
            pass

    explorer_df = explorer_df[
        search_mask
    ]


download_bytes = (
    explorer_df
    .to_csv(index=False)
    .encode("utf-8")
)


explorer_info, explorer_download = st.columns(
    [4, 1]
)

with explorer_info:

    st.caption(
        f"{len(explorer_df):,} records in current context"
    )

with explorer_download:

    st.download_button(
        "Download CSV",
        data=download_bytes,
        file_name="filtered_nlp_dataset.csv",
        mime="text/csv",
        width="stretch",
    )


st.dataframe(
    explorer_df.head(100),
    width="stretch",
    hide_index=True,
    height=350,
)


# ================================================================
# ANALYTICS METHODOLOGY
# ================================================================

st.divider()

st.subheader(
    "Analytics Methodology"
)

m1, m2, m3, m4 = st.columns(
    4,
    gap="medium",
)


with m1:

    st.markdown(
        "### Web Scraping"
    )

    st.write(
        "Public web content is collected and "
        "structured into analytical data."
    )


with m2:

    st.markdown(
        "### NLP Processing"
    )

    st.write(
        "Cleaning, tokenization and normalization "
        "prepare text for analysis."
    )


with m3:

    st.markdown(
        "### Sentiment ML"
    )

    st.write(
        "VADER weak labels support supervised "
        "sentiment classification."
    )


with m4:

    st.markdown(
        "### Topic Modeling"
    )

    st.write(
        "TF-IDF, n-grams and topic modeling expose "
        "latent themes."
    )


# ================================================================
# LIMITATIONS
# ================================================================

with st.expander(
    "Analytical Limitations"
):

    st.write(
        """
        **Sentiment labels:** The current dataset uses VADER-generated
        weak/pseudo labels rather than human-annotated ground truth.

        **Topic modeling:** LDA topics are unsupervised mathematical
        clusters and require contextual interpretation.

        **Model score:** LinearSVC decision scores are model signals,
        not calibrated probabilities.

        **Dataset scope:** Quotes to Scrape is a practice scraping
        corpus and should not be treated as representative of a
        broader real-world population.
        """
    )


# ================================================================
# ABOUT THE CREATOR / BIO
# ================================================================

st.divider()

bio_left, bio_right = st.columns(
    [1.2, 4.8],
    gap="large",
)

with bio_left:

    st.subheader(
        "About the Developer"
    )

    st.metric(
        "Project",
        "NLP Intelligence",
    )

    st.metric(
        "Interface",
        "Power BI Style",
    )


with bio_right:

    st.markdown(
        "### 👨‍💻 S Mohammed Kaif"
    )

    st.markdown(
        "**Data Science • Data Analytics • Machine Learning • AI • Python**"
    )

    st.write(
        "Data Science and Analytics enthusiast focused on building "
        "practical data products with Python, NLP, machine learning "
        "and interactive analytics."
    )

    st.write(
        "This project demonstrates an end-to-end workflow covering "
        "web scraping, data cleaning, NLP preprocessing, sentiment "
        "analysis, keyword intelligence, phrase analysis, topic "
        "modeling and interactive dashboard development."
    )

    st.markdown(
        "🔗 **GitHub:** "
        "[S Mohammed Kaif](https://github.com/Shaik-Mohammed-Kaif)"
    )

    st.markdown(
        "🔗 **LinkedIn:** "
        "[S Mohammed Kaif](https://www.linkedin.com/in/s-mohammed-kaif-2a500a341/)"
    )

    st.markdown(
        "> **Learn → Practice → Analyze → Build → Improve → Grow**"
    )


# FOOTER
# ================================================================

st.divider()

footer_left, footer_right = st.columns(
    [5, 1],
)

with footer_left:

    st.markdown(
        "### 👨‍💻 S Mohammed Kaif"
    )

    st.markdown(
        "**Data Science • Data Analytics • Machine Learning • AI • Python**"
    )

    st.markdown(
        "[GitHub](https://github.com/Shaik-Mohammed-Kaif)"
        "  •  "
        "[LinkedIn](https://www.linkedin.com/in/s-mohammed-kaif-2a500a341/)"
    )

    st.markdown(
        "> **Learn → Practice → Analyze → Build → Improve → Grow**"
    )


with footer_right:

    st.caption(
        "© 2026 S Mohammed Kaif"
    )


