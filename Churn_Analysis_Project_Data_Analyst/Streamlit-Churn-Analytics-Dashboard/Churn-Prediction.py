"""
============================================================================
CUSTOMER CHURN PREDICTION & ANALYTICS
AI-Powered Customer Risk Intelligence & Retention Analysis
============================================================================

A premium, production-quality Streamlit web application for customer
churn prediction and business analytics.

This application is intentionally schema-aware: it inspects the actual
CSV at runtime, detects the target column, numerical columns, categorical
columns, ID columns, and any pre-existing prediction/probability columns,
and builds its data-exploration, model-training and prediction pipeline
around whatever is really in the dataset. Nothing about the dataset
schema is hardcoded or assumed.

No pre-trained model file is required. The model is trained live, inside
the app, using only Customer_Churn_Predictions.csv.

Run with:
    streamlit run Stream_app.py

Author: S Mohammed Kaif
Version: 1.0
============================================================================
"""

# ----------------------------------------------------------------------
# 1. IMPORTS
# ----------------------------------------------------------------------
import os
import io
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 2. STREAMLIT PAGE CONFIGURATION
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction & Analytics",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# 3. GLOBAL CONSTANTS
# ----------------------------------------------------------------------

APP_TITLE = "CUSTOMER CHURN PREDICTION & ANALYTICS"
APP_SUBTITLE = "AI-Powered Customer Risk Intelligence & Retention Analysis"
APP_LABEL = "DATA SCIENCE  \u2022  MACHINE LEARNING  \u2022  BUSINESS INTELLIGENCE"
APP_VERSION = "1.0"
DEVELOPER_NAME = "S Mohammed Kaif"
COPYRIGHT_YEAR = "2026"

CSV_FILENAME = "Customer_Churn_Predictions.csv"

# Candidate names the app will look for when trying to auto-detect the
# churn target column. This list is only a set of *hints* - the app
# still validates whatever it finds before trusting it.
TARGET_NAME_HINTS = [
    "churn_flag", "Churn_Flag", "ChurnFlag",
    "Churn", "churn",
    "Customer_Status", "customer_status", "CustomerStatus",
    "Exited", "exited",
    "Churn_Status", "churn_status",
    "Is_Churned", "is_churned", "IsChurn",
]

# Columns that must NEVER be used as model features because they are
# either identifiers or they leak the outcome we are trying to predict.
LEAKAGE_NAME_HINTS = [
    "prediction", "predicted", "prediction_probability", "probability",
    "churn_probability", "churn_prediction", "risk_score", "risk_level",
    "model_prediction", "model_output", "churn_status_predicted",
    "churn_category", "churn_reason", "predicted_churn",
    "predicted_probability", "pred_proba", "pred_label",
]

ID_NAME_HINTS = [
    "customer_id", "customerid", "id", "subscription_id",
    "transaction_id", "cust_id", "custid", "uid", "user_id",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20

RISK_THRESHOLDS_DEFAULT = {
    "low": 0.30,
    "medium": 0.60,
    "high": 0.80,
}

# ----------------------------------------------------------------------
# 4. THEME DEFINITIONS
# ----------------------------------------------------------------------
# Each theme is a dictionary of CSS custom-property values. render_theme()
# below turns this into an injected <style> block so the whole app -
# cards, charts, buttons, text - shifts together.

THEMES = {
    "\U0001F31E Normal White": {
        "key": "white",
        "bg": "#FFFFFF",
        "bg_secondary": "#F7F8FA",
        "card": "#FFFFFF",
        "border": "#E5E7EB",
        "text_primary": "#111827",
        "text_secondary": "#6B7280",
        "accent": "#4F46E5",
        "accent_soft": "#EEF2FF",
        "accent_text": "#4338CA",
        "success": "#16A34A",
        "warning": "#D97706",
        "danger": "#DC2626",
        "shadow": "0 1px 2px rgba(16,24,40,0.06), 0 1px 3px rgba(16,24,40,0.08)",
        "shadow_hover": "0 8px 20px rgba(16,24,40,0.10)",
    },
    "\U0001F366 Vanilla Cream": {
        "key": "cream",
        "bg": "#FFF9ED",
        "bg_secondary": "#FFF4D6",
        "card": "#FFFCF5",
        "border": "#E8DCC3",
        "text_primary": "#2D2A26",
        "text_secondary": "#746D61",
        "accent": "#B4762C",
        "accent_soft": "#F6E9D0",
        "accent_text": "#8C5A1F",
        "success": "#3F7D3B",
        "warning": "#B4762C",
        "danger": "#B23B32",
        "shadow": "0 1px 2px rgba(90,72,32,0.08), 0 1px 3px rgba(90,72,32,0.10)",
        "shadow_hover": "0 8px 20px rgba(90,72,32,0.14)",
    },
    "\U0001F319 Dark": {
        "key": "dark",
        "bg": "#0E1117",
        "bg_secondary": "#151922",
        "card": "#181D27",
        "border": "#2A3140",
        "text_primary": "#F5F7FA",
        "text_secondary": "#A7AFBD",
        "accent": "#6366F1",
        "accent_soft": "#1E2233",
        "accent_text": "#A5B4FC",
        "success": "#34D399",
        "warning": "#FBBF24",
        "danger": "#F87171",
        "shadow": "0 1px 2px rgba(0,0,0,0.4), 0 4px 10px rgba(0,0,0,0.35)",
        "shadow_hover": "0 10px 24px rgba(0,0,0,0.5)",
    },
}

DEFAULT_THEME_LABEL = "\U0001F31E Normal White"


# ----------------------------------------------------------------------
# 5. THEME / CSS ENGINE
# ----------------------------------------------------------------------

def init_session_state():
    """Initialise every piece of session state the app relies on.

    Keeping this in one place means every other function can safely
    assume the keys already exist, instead of scattering
    `if "x" not in st.session_state` checks everywhere.
    """
    defaults = {
        "theme_label": DEFAULT_THEME_LABEL,
        "active_page": "Overview",
        "risk_thresholds": dict(RISK_THRESHOLDS_DEFAULT),
        "model_choice": "Logistic Regression",
        "last_prediction": None,
        "target_column_override": None,
        "training_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_theme():
    """Inject a <style> block built from the currently selected theme.

    This function is the single source of truth for the app's visual
    appearance. Every other render_* function relies on the CSS classes
    defined here (app-card, kpi-card, section-header, pill, etc.)
    instead of re-declaring styles inline.
    """
    theme = THEMES[st.session_state["theme_label"]]

    css = f"""
    <style>
        :root {{
            --bg: {theme['bg']};
            --bg-secondary: {theme['bg_secondary']};
            --card: {theme['card']};
            --border: {theme['border']};
            --text-primary: {theme['text_primary']};
            --text-secondary: {theme['text_secondary']};
            --accent: {theme['accent']};
            --accent-soft: {theme['accent_soft']};
            --accent-text: {theme['accent_text']};
            --success: {theme['success']};
            --warning: {theme['warning']};
            --danger: {theme['danger']};
            --shadow: {theme['shadow']};
            --shadow-hover: {theme['shadow_hover']};
        }}

        html, body, [class*="css"]  {{
            font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{
            background-color: var(--bg);
            color: var(--text-primary);
        }}

        section[data-testid="stSidebar"] {{
            background-color: var(--bg-secondary);
            border-right: 1px solid var(--border);
        }}

        section[data-testid="stSidebar"] * {{
            color: var(--text-primary) !important;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
        }}

        p, span, label, li {{
            color: var(--text-primary);
        }}

        /* ---------------- Hero ---------------- */
        .hero-wrap {{
            background: linear-gradient(135deg, var(--card) 0%, var(--bg-secondary) 100%);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 40px 44px;
            margin-bottom: 28px;
            box-shadow: var(--shadow);
        }}
        .hero-eyebrow {{
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.14em;
            color: var(--accent-text);
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .hero-title {{
            font-size: 40px;
            font-weight: 800;
            line-height: 1.12;
            color: var(--text-primary);
            margin: 0 0 10px 0;
            letter-spacing: -0.02em;
        }}
        .hero-subtitle {{
            font-size: 16px;
            color: var(--text-secondary);
            max-width: 720px;
            margin-bottom: 18px;
            line-height: 1.55;
        }}
        .status-row {{
            display: flex;
            gap: 18px;
            flex-wrap: wrap;
        }}
        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 999px;
            background: var(--accent-soft);
            border: 1px solid var(--border);
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
        }}
        .dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }}
        .dot-green {{ background-color: var(--success); }}
        .dot-red {{ background-color: var(--danger); }}
        .dot-amber {{ background-color: var(--warning); }}

        /* ---------------- Cards ---------------- */
        .app-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 22px 24px;
            box-shadow: var(--shadow);
            margin-bottom: 18px;
            transition: box-shadow 0.18s ease, transform 0.18s ease;
        }}
        .app-card:hover {{
            box-shadow: var(--shadow-hover);
            transform: translateY(-2px);
        }}

        .kpi-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: var(--shadow);
            transition: box-shadow 0.18s ease, transform 0.18s ease, filter 0.18s ease;
            height: 100%;
        }}
        .kpi-card:hover {{
            box-shadow: var(--shadow-hover);
            transform: translateY(-3px);
            filter: brightness(1.02);
        }}
        .kpi-label {{
            font-size: 12.5px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-size: 26px;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.01em;
        }}
        .kpi-sub {{
            font-size: 12.5px;
            color: var(--text-secondary);
            margin-top: 4px;
        }}

        /* ---------------- Section headers ---------------- */
        .section-header {{
            font-size: 22px;
            font-weight: 800;
            color: var(--text-primary);
            margin: 6px 0 4px 0;
            letter-spacing: -0.01em;
        }}
        .section-caption {{
            font-size: 13.5px;
            color: var(--text-secondary);
            margin-bottom: 16px;
        }}

        /* ---------------- Pills / badges ---------------- */
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.02em;
        }}
        .badge-low {{ background: rgba(22,163,74,0.14); color: var(--success); }}
        .badge-medium {{ background: rgba(217,119,6,0.14); color: var(--warning); }}
        .badge-high {{ background: rgba(220,38,38,0.14); color: var(--danger); }}
        .badge-veryhigh {{ background: var(--danger); color: #FFFFFF; }}
        .badge-neutral {{ background: var(--accent-soft); color: var(--accent-text); }}

        /* ---------------- Buttons ---------------- */
        div[data-testid="stButton"] > button, div[data-testid="stFormSubmitButton"] > button {{
            background: var(--accent);
            color: #FFFFFF;
            border: none;
            border-radius: 12px;
            padding: 0.6em 1.4em;
            font-weight: 700;
            font-size: 15px;
            box-shadow: var(--shadow);
            transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
        }}
        div[data-testid="stButton"] > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
            filter: brightness(1.07);
        }}
        div[data-testid="stButton"] > button:active, div[data-testid="stFormSubmitButton"] > button:active {{
            transform: translateY(0px);
            filter: brightness(0.97);
        }}
        div[data-testid="stButton"] > button:focus-visible {{
            outline: 2px solid var(--accent);
            outline-offset: 2px;
        }}
        div[data-testid="stDownloadButton"] > button {{
            background: var(--card);
            color: var(--accent-text);
            border: 1px solid var(--accent);
            border-radius: 12px;
            font-weight: 700;
            transition: transform 0.15s ease, filter 0.15s ease;
        }}
        div[data-testid="stDownloadButton"] > button:hover {{
            transform: translateY(-2px);
            filter: brightness(1.05);
        }}

        /* ---------------- Result card ---------------- */
        .result-card {{
            border-radius: 18px;
            padding: 26px 28px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
        }}
        .result-title {{
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-secondary);
            margin-bottom: 10px;
        }}
        .result-headline {{
            font-size: 24px;
            font-weight: 800;
            margin-bottom: 18px;
        }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-top: 1px solid var(--border);
            font-size: 14.5px;
        }}
        .metric-row:first-of-type {{ border-top: none; }}
        .metric-row .label {{ color: var(--text-secondary); font-weight: 600; }}
        .metric-row .value {{ color: var(--text-primary); font-weight: 700; }}

        /* ---------------- Footer ---------------- */
        .app-footer {{
            border-top: 1px solid var(--border);
            margin-top: 40px;
            padding-top: 22px;
            padding-bottom: 8px;
            text-align: center;
        }}
        .app-footer .footer-title {{
            font-weight: 800;
            font-size: 15px;
            color: var(--text-primary);
            letter-spacing: 0.02em;
        }}
        .app-footer .footer-flow {{
            font-size: 13px;
            color: var(--accent-text);
            font-weight: 600;
            margin: 8px 0;
        }}
        .app-footer .footer-stack {{
            font-size: 12.5px;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }}
        .app-footer .footer-copy {{
            font-size: 12px;
            color: var(--text-secondary);
        }}

        /* ---------------- Data quality indicator ---------------- */
        .quality-excellent {{ color: var(--success); font-weight: 800; }}
        .quality-good {{ color: var(--warning); font-weight: 800; }}
        .quality-attention {{ color: var(--danger); font-weight: 800; }}

        hr {{ border-color: var(--border) !important; }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }}

        div[data-baseweb="tab-list"] {{
            gap: 4px;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def status_dot(color_class: str) -> str:
    return f'<span class="dot {color_class}"></span>'


# ----------------------------------------------------------------------
# 6. DATA LOADING
# ----------------------------------------------------------------------

def find_csv_path(filename: str = CSV_FILENAME):
    """Locate the dataset CSV using a robust, portable search strategy.

    The app never hardcodes an absolute path (no D:\\... paths). Instead
    it searches a handful of sensible relative locations so the project
    keeps working after being cloned onto a different machine, as long
    as the folder structure described in the README is preserved:

        Churn_Analysis_Project_Data_Analyst/
        |-- Churn-Dataset-With-Geo-Loc-Analysis/
        |   `-- Customer_Churn_Predictions.csv
        `-- Streamlit_And-PowerBI-Dashbord-Churn-Analysis/
            `-- Stream_app.py
    """
    here = os.path.dirname(os.path.abspath(__file__))

    candidate_paths = [
        os.path.join(here, filename),
        os.path.join(here, "..", "Churn-Dataset-With-Geo-Loc-Analysis", filename),
        os.path.join(here, "..", filename),
        os.path.join(here, "data", filename),
        os.path.join(here, "Churn-Dataset-With-Geo-Loc-Analysis", filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "Churn-Dataset-With-Geo-Loc-Analysis", filename),
        filename,
    ]

    for path in candidate_paths:
        norm = os.path.normpath(path)
        if os.path.isfile(norm):
            return norm

    return None


@st.cache_data(show_spinner=False)
def load_data(csv_path: str):
    """Load the churn CSV into a DataFrame.

    Returns (dataframe, error_message). Exactly one of the two will be
    None. Errors are captured here instead of raised, so the caller can
    show a friendly Streamlit message instead of a traceback.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return None, (
            f"{CSV_FILENAME} was not found. Please place the dataset in the "
            f"project dataset directory."
        )
    except pd.errors.EmptyDataError:
        return None, "The dataset file is empty and cannot be loaded."
    except pd.errors.ParserError:
        return None, "The dataset file could not be parsed. Please check that it is a valid CSV."
    except Exception:
        return None, "The dataset could not be loaded due to an unexpected error."

    if df.shape[0] == 0 or df.shape[1] == 0:
        return None, "The dataset appears to be empty."

    # Strip incidental whitespace from column names - a very common
    # source of "column not found" bugs in real-world CSV exports.
    df.columns = [str(c).strip() for c in df.columns]

    return df, None


# ----------------------------------------------------------------------
# 7. SCHEMA DETECTION
# ----------------------------------------------------------------------

def _normalize(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def detect_id_columns(df: pd.DataFrame):
    """Detect columns that look like identifiers.

    A column is treated as an ID column if its normalized name matches a
    known ID hint, OR if it is (near-)unique per row and non-numeric,
    which is a strong practical signal of an identifier rather than a
    predictive feature.
    """
    id_cols = []
    n_rows = len(df)
    for col in df.columns:
        norm = _normalize(col)
        is_hint_match = any(hint in norm for hint in ID_NAME_HINTS)
        is_near_unique = n_rows > 0 and df[col].nunique(dropna=True) >= 0.98 * n_rows
        looks_id_like = is_hint_match or (is_near_unique and df[col].dtype == object)
        if looks_id_like:
            id_cols.append(col)
    return id_cols


def detect_leakage_columns(df: pd.DataFrame, target_col: str):
    """Detect columns that would leak the target and must be excluded
    from model features (existing prediction/probability columns, and
    any column that is essentially a restatement of the outcome).
    """
    leakage_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        norm = _normalize(col)
        if any(hint in norm for hint in LEAKAGE_NAME_HINTS):
            leakage_cols.append(col)
    return leakage_cols


def detect_statistical_leakage(df: pd.DataFrame, target_col: str, candidate_cols: list):
    """Catch leakage columns that name-based hints would miss - e.g. a
    'Customer_Status' column with values like Stayed/Churned/Joined is
    not literally named "prediction", but it perfectly determines the
    churn outcome and must never be used as a feature.

    A candidate column is flagged only when EVERY one of its values maps
    to a single target class (perfect purity), the column has a small,
    genuinely categorical number of distinct values, and each value is
    backed by enough rows that the purity is not just a coincidence of
    a tiny/unique category. This keeps the check from ever penalising
    ordinary numeric or high-cardinality behavioural features.
    """
    try:
        work, y = normalize_target_series(df, target_col)
    except Exception:
        return []

    n_rows = len(work)
    leak = []
    for col in candidate_cols:
        series = work[col]
        n_unique = series.nunique(dropna=True)
        if n_unique < 2 or n_unique > 30:
            continue
        group_sizes = series.value_counts()
        if group_sizes.min() < 5:
            continue
        purity = y.groupby(series.astype(str)).nunique()
        if purity.max() == 1:
            leak.append(col)
    return leak


def _column_looks_binary_target(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    uniques = set(non_null.unique().tolist())
    binary_sets = [
        {0, 1}, {"0", "1"}, {True, False},
        {"Yes", "No"}, {"yes", "no"},
        {"Churn", "No Churn"}, {"Churned", "Stayed"},
        {"Y", "N"},
    ]
    return uniques in binary_sets or (len(uniques) == 2)


def detect_target_column(df: pd.DataFrame, override: str = None):
    """Detect the churn target column.

    Strategy:
      1. If the user has manually overridden the target via the UI, use
         that (still validated below).
      2. Look for a column whose name matches a known churn-target hint
         AND that has a small number of classes (a real target, not an
         ID or a free-text field).
      3. As a last resort, look at any column with exactly two classes
         that is not flagged as an ID or leakage column.

    Returns (target_column_or_None, reason_string).
    """
    if override and override in df.columns:
        ok, reason = validate_target_column(df, override)
        return (override, "Manually selected by user") if ok else (None, reason)

    id_cols = set(detect_id_columns(df))

    # Pass 1: name-hint based candidates, preferring ones with the
    # fewest unique classes (cleanest binary targets first).
    hint_candidates = []
    for col in df.columns:
        norm = _normalize(col)
        for hint in TARGET_NAME_HINTS:
            if _normalize(hint) == norm:
                hint_candidates.append(col)
                break

    hint_candidates = [c for c in hint_candidates if c not in id_cols]
    hint_candidates.sort(key=lambda c: df[c].nunique(dropna=True))

    for col in hint_candidates:
        ok, reason = validate_target_column(df, col)
        if ok:
            return col, f"Auto-detected from column name '{col}'"

    # Pass 2: any binary-looking column not already excluded.
    fallback_candidates = [
        c for c in df.columns
        if c not in id_cols and _column_looks_binary_target(df[c])
    ]
    for col in fallback_candidates:
        norm = _normalize(col)
        if any(hint in norm for hint in LEAKAGE_NAME_HINTS):
            continue
        ok, reason = validate_target_column(df, col)
        if ok:
            return col, f"Auto-detected as a binary outcome column '{col}'"

    return None, "No valid churn target column could be identified."


def validate_target_column(df: pd.DataFrame, col: str):
    """Validate that a candidate column is safe and usable as a
    supervised-learning target. Returns (is_valid, message).
    """
    if col not in df.columns:
        return False, f"Column '{col}' does not exist in the dataset."

    series = df[col]

    if series.isna().all():
        return False, f"Column '{col}' is entirely missing and cannot be used as a target."

    non_null = series.dropna()
    n_classes = non_null.nunique()

    if n_classes < 2:
        return False, f"Column '{col}' has fewer than two classes and cannot be used for classification."

    if len(non_null) == len(df) and n_classes == len(df):
        return False, f"Column '{col}' looks like a unique identifier, not a target."

    if n_classes > 10:
        return False, f"Column '{col}' has too many distinct values ({n_classes}) to be a churn target."

    if len(non_null) < 20:
        return False, "Not enough rows with a valid target value to train a model."

    return True, "Valid target column."


def identify_feature_columns(df: pd.DataFrame, target_col: str):
    """Split the dataframe's columns into the buckets the rest of the
    app needs: numerical features, categorical features, ID columns
    (excluded), leakage columns (excluded), and the final feature list
    actually safe to feed into the model.
    """
    id_cols = detect_id_columns(df)
    leakage_cols = detect_leakage_columns(df, target_col)

    excluded = set(id_cols) | set(leakage_cols) | {target_col}
    feature_cols = [c for c in df.columns if c not in excluded]

    numeric_cols, categorical_cols = [], []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            # A numeric column with very few unique values that also
            # reads like a flag (0/1, Yes/No encoded as int) is still a
            # legitimate numeric/categorical-flag feature - we keep it
            # in numeric_cols since OneHot/Scale both handle it safely
            # via the ColumnTransformer branch it lands in.
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # Drop constant columns - they carry no signal and only slow down
    # / destabilize training.
    constant_cols = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]
    numeric_cols = [c for c in numeric_cols if c not in constant_cols]
    categorical_cols = [c for c in categorical_cols if c not in constant_cols]
    feature_cols = [c for c in feature_cols if c not in constant_cols]

    return {
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "id_cols": id_cols,
        "leakage_cols": leakage_cols,
        "constant_cols": constant_cols,
    }


def compute_data_quality(df: pd.DataFrame):
    """Compute the raw data-quality statistics shown in the Data Quality
    section, plus a headline Excellent / Good / Needs Attention rating
    derived from those actual numbers (never a random score).
    """
    n_rows, n_cols = df.shape
    missing_total = int(df.isna().sum().sum())
    missing_pct = (missing_total / (n_rows * n_cols) * 100) if n_rows and n_cols else 0.0
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = (duplicate_rows / n_rows * 100) if n_rows else 0.0
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    if missing_pct < 1 and duplicate_pct < 1 and len(constant_cols) == 0:
        rating = "Excellent"
    elif missing_pct < 8 and duplicate_pct < 5:
        rating = "Good"
    else:
        rating = "Needs Attention"

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_total": missing_total,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": duplicate_pct,
        "constant_cols": constant_cols,
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
        "rating": rating,
    }


# ----------------------------------------------------------------------
# 8. MODEL PREPARATION / TRAINING / EVALUATION
# ----------------------------------------------------------------------

def prepare_data(df: pd.DataFrame, target_col: str, schema: dict):
    """Build the (X, y) pair and normalize the target into a clean 0/1
    label the classifier can train on, regardless of whether the raw
    target column was numeric (0/1) or text (Yes/No, Churned/Stayed...).
    """
    feature_cols = schema["feature_cols"]

    work = df.dropna(subset=[target_col]).copy()
    X = work[feature_cols].copy()
    raw_y = work[target_col]

    positive_tokens = {
        "1", "yes", "y", "true", "churn", "churned", "exited", "high risk",
    }

    if pd.api.types.is_numeric_dtype(raw_y):
        uniques = sorted(raw_y.dropna().unique().tolist())
        if set(uniques) == {0, 1}:
            y = raw_y.astype(int)
        else:
            # Numeric but not already 0/1 (e.g. two arbitrary numeric
            # codes) - map the larger value to 1 (positive class).
            hi = max(uniques)
            y = (raw_y == hi).astype(int)
    else:
        y = raw_y.astype(str).str.strip().str.lower().isin(positive_tokens).astype(int)

    return X, y


def build_preprocessor(numeric_cols, categorical_cols):
    """Build a ColumnTransformer + imputation pipeline. Using a single
    sklearn Pipeline (rather than hand-rolled encoding/scaling) keeps
    training and inference perfectly in sync and avoids the classic
    "encoder fit on training data, mismatched at prediction time" bug.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


@st.cache_resource(show_spinner=False)
def train_model(_df_signature: str, df: pd.DataFrame, target_col: str,
                 numeric_cols: tuple, categorical_cols: tuple,
                 feature_cols: tuple, model_choice: str):
    """Train the churn classifier end to end and return everything the
    rest of the app needs to render results.

    `_df_signature` is a cheap fingerprint (shape + column names) used
    purely so Streamlit's cache_resource keys correctly on dataset
    identity without hashing the whole dataframe on every rerun.
    """
    schema = {
        "feature_cols": list(feature_cols),
        "numeric_cols": list(numeric_cols),
        "categorical_cols": list(categorical_cols),
    }

    X, y = prepare_data(df, target_col, schema)

    if y.nunique() < 2:
        return {"success": False, "error": "The target column only contains a single class after cleaning."}

    if len(X) < 20:
        return {"success": False, "error": "Not enough rows available to train a reliable model."}

    can_stratify = y.value_counts().min() >= 2
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y if can_stratify else None,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    preprocessor = build_preprocessor(schema["numeric_cols"], schema["categorical_cols"])

    if model_choice == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        )
    else:
        classifier = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    try:
        pipeline.fit(X_train, y_train)
    except Exception as exc:
        return {"success": False, "error": f"Model training failed: {type(exc).__name__}"}

    y_pred = pipeline.predict(X_test)
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    metrics = evaluate_model(y_test, y_pred, y_proba)
    feature_importance_df = compute_feature_importance(pipeline, schema, model_choice)

    return {
        "success": True,
        "pipeline": pipeline,
        "schema": schema,
        "target_col": target_col,
        "model_choice": model_choice,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
        "feature_importance": feature_importance_df,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


def evaluate_model(y_test, y_pred, y_proba):
    """Compute the standard classification metrics safely - if ROC-AUC
    cannot be computed (e.g. only one class present in a fold) it is
    simply omitted rather than raising.
    """
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
    }
    if y_proba is not None and len(set(y_test)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        except Exception:
            metrics["roc_auc"] = None
    return metrics


def compute_feature_importance(pipeline, schema, model_choice):
    """Recover human-readable feature names from the fitted
    ColumnTransformer and pair them with importance / coefficient
    values so the UI can show "Top Churn Drivers" without exposing
    raw encoded column names like `categorical__Contract_Two Year`.
    """
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        feature_names = preprocessor.get_feature_names_out()
        clean_names = [
            fn.split("__", 1)[1] if "__" in fn else fn for fn in feature_names
        ]

        if model_choice == "Random Forest" and hasattr(classifier, "feature_importances_"):
            values = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            values = classifier.coef_[0]
        else:
            return None

        imp_df = pd.DataFrame({"feature": clean_names, "importance": values})
        imp_df["abs_importance"] = imp_df["importance"].abs()
        imp_df = imp_df.sort_values("abs_importance", ascending=False).head(15)
        return imp_df
    except Exception:
        return None


def get_risk_level(probability: float, thresholds: dict = None):
    """Translate a churn probability into a transparent, configurable
    risk band. Returns (label, badge_css_class).
    """
    thresholds = thresholds or RISK_THRESHOLDS_DEFAULT
    if probability < thresholds["low"]:
        return "LOW RISK", "badge-low"
    elif probability < thresholds["medium"]:
        return "MEDIUM RISK", "badge-medium"
    elif probability < thresholds["high"]:
        return "HIGH RISK", "badge-high"
    else:
        return "VERY HIGH RISK", "badge-veryhigh"


def get_recommendation(risk_label: str):
    recs = {
        "LOW RISK": [
            "Maintain the current relationship with standard engagement.",
            "Consider cross-sell or loyalty-programme opportunities.",
        ],
        "MEDIUM RISK": [
            "Include the customer in a proactive engagement campaign.",
            "Run a satisfaction follow-up to catch issues early.",
        ],
        "HIGH RISK": [
            "Launch a targeted retention campaign for this customer.",
            "Review their current plan, pricing, and service quality.",
        ],
        "VERY HIGH RISK": [
            "Prioritise immediate retention outreach.",
            "Offer a personalised retention incentive.",
            "Escalate to a customer-support specialist.",
        ],
    }
    return recs.get(risk_label, ["Review the customer's account manually."])


def make_prediction(pipeline, input_dict: dict, feature_cols: list):
    """Run a single-customer prediction through the fitted pipeline.
    Returns (prediction_label, probability, error_message).
    """
    try:
        input_df = pd.DataFrame([{col: input_dict.get(col) for col in feature_cols}])
        prediction = pipeline.predict(input_df)[0]
        try:
            probability = float(pipeline.predict_proba(input_df)[0, 1])
        except Exception:
            probability = float(prediction)
        label = "Churn" if int(prediction) == 1 else "No Churn"
        return label, probability, None
    except Exception:
        return None, None, "Prediction could not be completed with the values provided."


# ----------------------------------------------------------------------
# 9. HERO SECTION
# ----------------------------------------------------------------------

def render_hero(csv_ok: bool, model_ok: bool):
    csv_dot = "dot-green" if csv_ok else "dot-red"
    csv_text = "CSV Connected" if csv_ok else "CSV Not Found"
    model_dot = "dot-green" if model_ok else "dot-amber"
    model_text = "ML Engine Ready" if model_ok else "ML Engine Not Trained"

    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-eyebrow">{APP_LABEL}</div>
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-subtitle">
                {APP_SUBTITLE}. Analyze customer behaviour, predict churn probability,
                identify high-risk customers, and support retention decisions with a
                transparent, dataset-driven model.
            </div>
            <div class="status-row">
                <span class="status-pill">{status_dot(csv_dot)}{csv_text}</span>
                <span class="status-pill">{status_dot(model_dot)}{model_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# 10. SIDEBAR
# ----------------------------------------------------------------------

def render_sidebar(df, csv_path, schema, target_col, target_reason, training_result):
    with st.sidebar:
        st.markdown("### \U0001F4C1 PROJECT")
        st.caption("Customer Churn Intelligence")

        st.markdown("---")
        st.markdown("### \U0001F3A8 THEME")
        theme_label = st.selectbox(
            "Select appearance",
            options=list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state["theme_label"]),
            label_visibility="collapsed",
        )
        if theme_label != st.session_state["theme_label"]:
            st.session_state["theme_label"] = theme_label
            st.rerun()

        st.markdown("---")
        st.markdown("### \U0001F5C4 DATASET")
        if df is not None:
            st.caption(f"{CSV_FILENAME}")
            st.caption(f"{df.shape[0]:,} rows  \u2022  {df.shape[1]} columns")
        else:
            st.caption("Not loaded")

        st.markdown("---")
        st.markdown("### \U0001F9E0 MODEL")
        if target_col:
            st.caption(f"Target column: **{target_col}**")
        else:
            st.caption("Target column: Not detected")

        model_choice = st.selectbox(
            "Algorithm",
            options=["Logistic Regression", "Random Forest"],
            index=["Logistic Regression", "Random Forest"].index(st.session_state["model_choice"]),
        )
        st.session_state["model_choice"] = model_choice

        if training_result and training_result.get("success"):
            st.success("Training status: Ready", icon="\u2705")
            acc = training_result["metrics"]["accuracy"]
            st.caption(f"Hold-out accuracy: {acc*100:.1f}%")
        elif training_result and not training_result.get("success"):
            st.warning(f"Training status: {training_result.get('error', 'Failed')}", icon="\u26A0\uFE0F")
        else:
            st.info("Training status: Pending", icon="\u23F3")

        st.markdown("---")
        st.markdown("### \U0001F39B CONTROLS")
        st.caption("Risk classification thresholds")
        low = st.slider("Low / Medium boundary", 0.05, 0.60,
                         st.session_state["risk_thresholds"]["low"], 0.01)
        medium = st.slider("Medium / High boundary", low + 0.01, 0.90,
                            max(st.session_state["risk_thresholds"]["medium"], low + 0.01), 0.01)
        high = st.slider("High / Very High boundary", medium + 0.01, 0.99,
                          max(st.session_state["risk_thresholds"]["high"], medium + 0.01), 0.01)
        st.session_state["risk_thresholds"] = {"low": low, "medium": medium, "high": high}

        if schema is not None:
            with st.expander("Target column override"):
                st.caption("Only change this if auto-detection picked the wrong column.")
                options = ["Auto-detect"] + list(df.columns)
                current = st.session_state.get("target_column_override") or "Auto-detect"
                choice = st.selectbox("Target column", options=options,
                                       index=options.index(current) if current in options else 0)
                new_override = None if choice == "Auto-detect" else choice
                if new_override != st.session_state["target_column_override"]:
                    st.session_state["target_column_override"] = new_override
                    st.rerun()

        st.markdown("---")
        st.markdown("### \u2139\uFE0F APP INFO")
        st.caption(f"Version: {APP_VERSION}")
        st.caption("Built with Python + Streamlit + Scikit-Learn")


# ----------------------------------------------------------------------
# 11. TOP NAVIGATION
# ----------------------------------------------------------------------

NAV_SECTIONS = ["Overview", "Data Explorer", "Prediction", "Model Analysis",
                 "Business Insights", "About"]

NAV_ICONS = {
    "Overview": "\U0001F4CA",
    "Data Explorer": "\U0001F50D",
    "Prediction": "\U0001F52E",
    "Model Analysis": "\U0001F9E0",
    "Business Insights": "\U0001F4BC",
    "About": "\u2139\uFE0F",
}


def render_navigation():
    labels = [f"{NAV_ICONS[s]}  {s}" for s in NAV_SECTIONS]
    current_index = NAV_SECTIONS.index(st.session_state["active_page"])
    selected_label = st.radio(
        "Navigate",
        options=labels,
        index=current_index,
        horizontal=True,
        label_visibility="collapsed",
    )
    selected_page = NAV_SECTIONS[labels.index(selected_label)]
    st.session_state["active_page"] = selected_page
    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
    return selected_page


# ----------------------------------------------------------------------
# 12. KPI HELPERS
# ----------------------------------------------------------------------

def find_column(df, *name_candidates):
    """Return the first column in df whose normalized name matches one
    of the given candidates, or None. Used so KPI/chart code can adapt
    to slightly different naming conventions without hardcoding a
    single exact spelling.
    """
    normalized_map = {_normalize(c): c for c in df.columns}
    for candidate in name_candidates:
        norm = _normalize(candidate)
        if norm in normalized_map:
            return normalized_map[norm]
    return None


def kpi_card(label, value, sub=None):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(df, target_col):
    """Render the KPI row. Every KPI is computed from columns that
    actually exist; anything that cannot be computed is skipped rather
    than faked.
    """
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Calculated directly from the loaded dataset.</div>', unsafe_allow_html=True)

    tenure_col = find_column(df, "tenure", "tenure_in_months", "Tenure_in_Months")
    charges_col = find_column(df, "monthlycharges", "monthly_charge", "Monthly_Charge")
    revenue_col = find_column(df, "totalcharges", "total_revenue", "Total_Revenue", "total_charges")

    cards = []

    cards.append(("Total Customers", f"{len(df):,}", None))

    if target_col and target_col in df.columns:
        _, y = prepare_data(df, target_col, {"feature_cols": []})
        churned = int(y.sum())
        churn_rate = churned / len(y) * 100 if len(y) else 0
        cards.append(("Churned Customers", f"{churned:,}", f"{churn_rate:.1f}% of total"))
        cards.append(("Churn Rate", f"{churn_rate:.1f}%", None))
    else:
        cards.append(("Churned Customers", "Not available in dataset", None))
        cards.append(("Churn Rate", "Not available in dataset", None))

    if tenure_col:
        cards.append(("Average Tenure", f"{df[tenure_col].mean():.1f} months", None))
    else:
        cards.append(("Average Tenure", "Not available in dataset", None))

    if charges_col:
        cards.append(("Avg. Monthly Charges", f"${df[charges_col].mean():,.2f}", None))
    else:
        cards.append(("Avg. Monthly Charges", "Not available in dataset", None))

    if revenue_col:
        cards.append(("Total Revenue", f"${df[revenue_col].sum():,.0f}", None))
        cards.append(("Avg. Revenue / Customer", f"${df[revenue_col].mean():,.2f}", None))

    cols = st.columns(4)
    for i, (label, value, sub) in enumerate(cards):
        with cols[i % 4]:
            kpi_card(label, value, sub)
        if (i + 1) % 4 == 0 and (i + 1) < len(cards):
            cols = st.columns(4)


def render_data_quality(df, quality):
    st.markdown('<div class="section-header">Data Quality</div>', unsafe_allow_html=True)
    rating = quality["rating"]
    rating_class = {
        "Excellent": "quality-excellent",
        "Good": "quality-good",
        "Needs Attention": "quality-attention",
    }[rating]
    st.markdown(
        f'<div class="section-caption">Overall rating: <span class="{rating_class}">{rating}</span></div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(6)
    stats = [
        ("Dataset Rows", f"{quality['n_rows']:,}"),
        ("Dataset Columns", f"{quality['n_cols']}"),
        ("Missing Values", f"{quality['missing_total']:,}"),
        ("Duplicate Rows", f"{quality['duplicate_rows']:,}"),
        ("Numerical Features", f"{quality['numeric_count']}"),
        ("Categorical Features", f"{quality['categorical_count']}"),
    ]
    for col, (label, value) in zip(cols, stats):
        with col:
            kpi_card(label, value)


def render_overview_page(df, target_col, quality):
    render_kpis(df, target_col)
    st.markdown("<br>", unsafe_allow_html=True)
    render_data_quality(df, quality)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Snapshot Charts</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">A quick visual pulse of the dataset.</div>', unsafe_allow_html=True)

    theme = THEMES[st.session_state["theme_label"]]
    col1, col2 = st.columns(2)

    with col1:
        if target_col and target_col in df.columns:
            _, y = prepare_data(df, target_col, {"feature_cols": []})
            counts = y.value_counts().rename({0: "No Churn", 1: "Churn"})
            fig = px.pie(
                names=counts.index, values=counts.values,
                hole=0.55, title="Churn Distribution",
                color=counts.index,
                color_discrete_map={"No Churn": theme["accent"], "Churn": theme["danger"]},
            )
            fig = style_plotly(fig, theme)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Churn distribution unavailable: target column not detected.")

    with col2:
        contract_col = find_column(df, "contract", "Contract")
        if contract_col:
            counts = df[contract_col].value_counts()
            fig = px.bar(
                x=counts.index, y=counts.values,
                title="Customers by Contract Type",
                labels={"x": "Contract", "y": "Customers"},
            )
            fig = style_plotly(fig, theme)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Contract-type chart unavailable: no contract column detected.")


# ----------------------------------------------------------------------
# 13. PLOTLY THEMING HELPER
# ----------------------------------------------------------------------

def style_plotly(fig, theme):
    """Apply consistent, restrained styling to every Plotly chart in the
    app so charts always match the active theme instead of clashing
    with a default white/blue Plotly template.
    """
    fig.update_layout(
        paper_bgcolor=theme["card"],
        plot_bgcolor=theme["card"],
        font=dict(color=theme["text_primary"], family="Inter, Segoe UI, sans-serif"),
        title_font=dict(size=15, color=theme["text_primary"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=48, l=10, r=10, b=10),
        colorway=[theme["accent"], theme["success"], theme["warning"],
                  theme["danger"], theme["accent_text"]],
    )
    fig.update_xaxes(showgrid=True, gridcolor=theme["border"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=theme["border"], zeroline=False)
    return fig


# ----------------------------------------------------------------------
# 14. DATA EXPLORER PAGE
# ----------------------------------------------------------------------

def render_data_explorer_page(df, schema):
    st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Search, filter, and inspect the raw dataset.</div>', unsafe_allow_html=True)

    tab_raw, tab_filtered, tab_stats, tab_missing = st.tabs(
        ["\U0001F4C4 Raw Data", "\U0001F50D Filtered Data", "\U0001F4C8 Statistics", "\u2753 Missing Values"]
    )

    with tab_raw:
        st.caption(f"Showing the first 100 rows of {len(df):,} total rows.")
        st.dataframe(df.head(100), height=420, use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("\U0001F4E5 Download Full Dataset (CSV)", csv_bytes,
                            file_name="churn_dataset_export.csv", mime="text/csv")

    with tab_filtered:
        st.caption("Build a filtered view using the controls below.")
        col_a, col_b = st.columns(2)

        with col_a:
            columns_to_show = st.multiselect(
                "Columns to display", options=list(df.columns),
                default=list(df.columns)[: min(10, len(df.columns))],
            )

        with col_b:
            categorical_cols = schema["categorical_cols"]
            filter_col = st.selectbox(
                "Filter by categorical column",
                options=["None"] + categorical_cols,
            )

        filtered = df.copy()

        if filter_col != "None":
            values = sorted(df[filter_col].dropna().unique().tolist())
            selected_values = st.multiselect(f"Values for '{filter_col}'", options=values, default=values)
            if selected_values:
                filtered = filtered[filtered[filter_col].isin(selected_values)]

        numeric_cols = schema["numeric_cols"]
        if numeric_cols:
            numeric_filter_col = st.selectbox("Filter by numeric range (optional)",
                                               options=["None"] + numeric_cols)
            if numeric_filter_col != "None":
                col_min = float(df[numeric_filter_col].min())
                col_max = float(df[numeric_filter_col].max())
                if col_min < col_max:
                    lo, hi = st.slider(f"{numeric_filter_col} range", col_min, col_max,
                                        (col_min, col_max))
                    filtered = filtered[
                        (filtered[numeric_filter_col] >= lo) & (filtered[numeric_filter_col] <= hi)
                    ]

        display_cols = columns_to_show if columns_to_show else list(df.columns)
        st.caption(f"{len(filtered):,} rows match the current filters.")
        st.dataframe(filtered[display_cols], height=420, use_container_width=True)

        filtered_csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("\U0001F4E5 Download Filtered Data (CSV)", filtered_csv,
                            file_name="churn_filtered_export.csv", mime="text/csv")

    with tab_stats:
        st.caption("Descriptive statistics for numerical columns.")
        if schema["numeric_cols"]:
            st.dataframe(df[schema["numeric_cols"]].describe().T, use_container_width=True)
        else:
            st.info("No numerical columns detected in this dataset.")

        st.caption("Category counts for categorical columns.")
        cat_choice = st.selectbox("Choose a categorical column",
                                   options=schema["categorical_cols"] or ["None"])
        if cat_choice and cat_choice != "None":
            counts = df[cat_choice].value_counts().reset_index()
            counts.columns = [cat_choice, "Count"]
            st.dataframe(counts, use_container_width=True, height=300)

    with tab_missing:
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            st.success("No missing values detected in this dataset.", icon="\u2705")
        else:
            missing_df = missing.reset_index()
            missing_df.columns = ["Column", "Missing Count"]
            missing_df["Missing %"] = (missing_df["Missing Count"] / len(df) * 100).round(2)
            st.dataframe(missing_df, use_container_width=True, height=320)


# ----------------------------------------------------------------------
# 15. PREDICTION PAGE
# ----------------------------------------------------------------------

def _widget_for_column(df, col, key_prefix):
    """Choose the right Streamlit input widget for a given feature
    column, based purely on its actual dtype and cardinality in the
    dataset - never a hardcoded field list.
    """
    series = df[col].dropna()
    key = f"{key_prefix}_{col}"

    if pd.api.types.is_bool_dtype(df[col]):
        return st.selectbox(col.replace("_", " "), options=[True, False], key=key)

    if pd.api.types.is_numeric_dtype(df[col]):
        col_min = float(series.min()) if not series.empty else 0.0
        col_max = float(series.max()) if not series.empty else 100.0
        col_mean = float(series.mean()) if not series.empty else 0.0
        is_integer_like = pd.api.types.is_integer_dtype(df[col]) or (series % 1 == 0).all()
        step = 1.0 if is_integer_like else round(max((col_max - col_min) / 100, 0.01), 2)
        return st.number_input(
            col.replace("_", " "),
            min_value=col_min, max_value=col_max if col_max > col_min else col_min + 1,
            value=col_mean, step=step, key=key,
        )

    # Object / categorical column
    uniques = sorted([str(v) for v in series.unique().tolist()])
    if set(u.lower() for u in uniques) <= {"yes", "no"}:
        return st.selectbox(col.replace("_", " "), options=["Yes", "No"], key=key)

    if len(uniques) == 0:
        return st.text_input(col.replace("_", " "), key=key)

    if len(uniques) <= 30:
        return st.selectbox(col.replace("_", " "), options=uniques, key=key)

    # High-cardinality text column - fall back to a controlled text
    # input pre-filled with the most common value, rather than a raw
    # free-text box with no context.
    default_val = df[col].mode().iloc[0] if not df[col].mode().empty else ""
    return st.text_input(col.replace("_", " "), value=str(default_val), key=key)


def render_prediction_page(df, schema, training_result):
    st.markdown('<div class="section-header">Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Enter customer attributes below and run the model to '
        'estimate churn risk.</div>',
        unsafe_allow_html=True,
    )

    if not training_result or not training_result.get("success"):
        st.warning(
            "Prediction is unavailable because the model has not been trained successfully. "
            "Check the Model Analysis page for details.",
            icon="\u26A0\uFE0F",
        )
        return

    feature_cols = training_result["schema"]["feature_cols"]
    pipeline = training_result["pipeline"]

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    with st.form("prediction_form"):
        st.markdown("##### Customer Attributes")
        input_values = {}

        n_cols_per_row = 3
        cols = st.columns(n_cols_per_row)
        for i, col in enumerate(feature_cols):
            with cols[i % n_cols_per_row]:
                input_values[col] = _widget_for_column(df, col, "predict")

        submitted = st.form_submit_button("\U0001F52E  PREDICT CUSTOMER CHURN")
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        label, probability, error = make_prediction(pipeline, input_values, feature_cols)

        if error:
            st.error(error)
            return

        risk_label, badge_class = get_risk_level(probability, st.session_state["risk_thresholds"])
        recommendations = get_recommendation(risk_label)

        st.session_state["last_prediction"] = {
            "inputs": input_values,
            "label": label,
            "probability": probability,
            "risk_label": risk_label,
            "recommendations": recommendations,
        }

    result = st.session_state.get("last_prediction")
    if result:
        render_prediction_result(result)


def render_prediction_result(result):
    probability = result["probability"]
    risk_label = result["risk_label"]
    badge_class = {
        "LOW RISK": "badge-low",
        "MEDIUM RISK": "badge-medium",
        "HIGH RISK": "badge-high",
        "VERY HIGH RISK": "badge-veryhigh",
    }[risk_label]

    icon = "\u2705" if risk_label == "LOW RISK" else (
        "\u26A0\uFE0F" if risk_label in ("MEDIUM RISK", "HIGH RISK") else "\U0001F6A8"
    )

    col_result, col_rec = st.columns([1.1, 1])

    with col_result:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">Customer Risk Result</div>
                <div class="result-headline">{icon} {risk_label}</div>
                <div class="metric-row">
                    <span class="label">Churn Probability</span>
                    <span class="value">{probability*100:.1f}%</span>
                </div>
                <div class="metric-row">
                    <span class="label">Prediction</span>
                    <span class="value">{"Likely to Churn" if result["label"] == "Churn" else "Likely to Stay"}</span>
                </div>
                <div class="metric-row">
                    <span class="label">Risk Level</span>
                    <span class="value"><span class="badge {badge_class}">{risk_label}</span></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(probability, 0.0), 1.0))

    with col_rec:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("##### Recommended Action")
        st.caption("Business recommendations, not guaranteed outcomes.")
        for rec in result["recommendations"]:
            st.markdown(f"- {rec}")
        st.markdown("</div>", unsafe_allow_html=True)

    export_df = pd.DataFrame([{
        **result["inputs"],
        "Prediction": result["label"],
        "Churn_Probability": round(probability, 4),
        "Risk_Level": risk_label,
    }])
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("\U0001F4E5 Download Prediction Result", csv_bytes,
                        file_name="churn_prediction_result.csv", mime="text/csv")


# ----------------------------------------------------------------------
# 16. MODEL ANALYSIS PAGE
# ----------------------------------------------------------------------

def render_model_analysis_page(training_result, target_col, target_reason):
    st.markdown('<div class="section-header">Model Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Transparent evaluation of the trained churn model.</div>',
                unsafe_allow_html=True)

    if not training_result:
        st.info("Model has not been trained yet.")
        return

    if not training_result.get("success"):
        st.warning(
            "Prediction training cannot be started because a valid churn target column "
            "could not be identified or the model failed to train. "
            f"Details: {training_result.get('error', 'Unknown error')}",
            icon="\u26A0\uFE0F",
        )
        return

    st.markdown(
        f'<div class="app-card">Target column detected: <b>{target_col}</b> &nbsp;&mdash;&nbsp; {target_reason}<br>'
        f'Algorithm: <b>{training_result["model_choice"]}</b> &nbsp;&mdash;&nbsp; '
        f'Trained on {training_result["train_rows"]:,} rows, evaluated on {training_result["test_rows"]:,} rows.</div>',
        unsafe_allow_html=True,
    )

    metrics = training_result["metrics"]
    theme = THEMES[st.session_state["theme_label"]]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Evaluation Metrics</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    kpi_pairs = [
        ("Accuracy", metrics["accuracy"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1 Score", metrics["f1"]),
        ("ROC-AUC", metrics["roc_auc"]),
    ]
    for col, (label, value) in zip(cols, kpi_pairs):
        with col:
            display_value = f"{value*100:.1f}%" if value is not None else "Not available"
            kpi_card(label, display_value)

    st.markdown("<br>", unsafe_allow_html=True)
    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown('<div class="section-header" style="font-size:18px;">Confusion Matrix</div>',
                     unsafe_allow_html=True)
        cm = metrics["confusion_matrix"]
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted: No Churn", "Predicted: Churn"],
            y=["Actual: No Churn", "Actual: Churn"],
            colorscale=[[0, theme["accent_soft"]], [1, theme["accent"]]],
            text=cm, texttemplate="%{text}", showscale=False,
        ))
        fig.update_layout(title="Confusion Matrix")
        fig = style_plotly(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        st.caption(
            f"True Negative: {tn}  \u2022  False Positive: {fp}  \u2022  "
            f"False Negative: {fn}  \u2022  True Positive: {tp}"
        )
        st.caption(
            "False Negatives are customers the model missed who actually churned - the "
            "costliest error for retention. False Positives are loyal customers flagged "
            "as at-risk, which mainly costs a wasted retention offer."
        )

    with col_roc:
        st.markdown('<div class="section-header" style="font-size:18px;">ROC Curve</div>',
                     unsafe_allow_html=True)
        if metrics.get("roc_curve"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=metrics["roc_curve"]["fpr"], y=metrics["roc_curve"]["tpr"],
                mode="lines", name="Model", line=dict(color=theme["accent"], width=3),
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
                line=dict(color=theme["text_secondary"], width=1, dash="dash"),
            ))
            fig.update_layout(title=f"ROC Curve (AUC = {metrics['roc_auc']:.3f})" if metrics["roc_auc"] else "ROC Curve",
                               xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            fig = style_plotly(fig, theme)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ROC curve is not available for this model configuration.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Top Churn Drivers</div>', unsafe_allow_html=True)
    st.caption("Model-associated features, ordered by influence on the prediction. This reflects "
               "correlation captured by the model, not proven causation.")

    imp_df = training_result.get("feature_importance")
    if imp_df is not None and not imp_df.empty:
        imp_df = imp_df.sort_values("abs_importance", ascending=True)
        fig = px.bar(
            imp_df, x="importance", y="feature", orientation="h",
            title="Top Churn Drivers",
            color="importance",
            color_continuous_scale=[theme["danger"], theme["border"], theme["success"]],
        )
        fig.update_layout(coloraxis_showscale=False, height=420)
        fig = style_plotly(fig, theme)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance is not available for this model configuration.")

    with st.expander("Full Classification Report"):
        report_df = pd.DataFrame(metrics["report"]).T
        st.dataframe(report_df, use_container_width=True)


# ----------------------------------------------------------------------

# 17. BUSINESS INSIGHTS PAGE

# ----------------------------------------------------------------------



def render_business_insights_page(df, target_col, schema):

    st.markdown('<div class="section-header">Business Insights</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-caption">Findings generated directly from the actual data - '

                'nothing here is fabricated.</div>', unsafe_allow_html=True)



    if not target_col or target_col not in df.columns:

        st.info("Business insights require a detected churn target column, which is not available "

                "for this dataset.")

        return



    theme = THEMES[st.session_state["theme_label"]]

    _, y = prepare_data(df, target_col, {"feature_cols": []})

    work = df.copy()

    work["_churn_flag_internal"] = y.values



    insight_cols = st.columns(2)

    idx = 0



    def next_col():

        nonlocal idx

        c = insight_cols[idx % 2]

        idx += 1

        return c



    # --- Contract pattern ---

    contract_col = find_column(df, "contract", "Contract")

    if contract_col:

        with next_col():

            st.markdown('<div class="app-card">', unsafe_allow_html=True)

            st.markdown("##### Contract Type & Churn")

            rate = work.groupby(contract_col)["_churn_flag_internal"].mean().sort_values(ascending=False) * 100

            fig = px.bar(x=rate.index, y=rate.values, labels={"x": "Contract", "y": "Churn Rate (%)"},

                         title="Churn Rate by Contract Type")

            fig = style_plotly(fig, theme)

            st.plotly_chart(fig, use_container_width=True)

            top_contract = rate.index[0]

            st.caption(f"Customers on **{top_contract}** contracts churn at the highest rate "

                       f"({rate.iloc[0]:.1f}%) in this dataset.")

            st.markdown("</div>", unsafe_allow_html=True)



    # --- Tenure pattern ---

    tenure_col = find_column(df, "tenure", "tenure_in_months", "Tenure_in_Months")

    tenure_group_col = find_column(df, "tenure_group", "Tenure_Group")

    if tenure_group_col or tenure_col:

        with next_col():

            st.markdown('<div class="app-card">', unsafe_allow_html=True)

            st.markdown("##### Tenure & Churn")

            group_col = tenure_group_col if tenure_group_col else tenure_col

            if tenure_group_col:

                rate = work.groupby(group_col)["_churn_flag_internal"].mean().sort_values(ascending=False) * 100

                fig = px.bar(x=rate.index, y=rate.values, labels={"x": "Tenure Group", "y": "Churn Rate (%)"},

                             title="Churn Rate by Tenure Group")

            else:

                fig = px.box(work, x="_churn_flag_internal", y=tenure_col,

                             labels={"_churn_flag_internal": "Churned (1) vs Stayed (0)", "y": tenure_col},

                             title="Tenure Distribution by Churn Outcome")

            fig = style_plotly(fig, theme)

            st.plotly_chart(fig, use_container_width=True)

            avg_tenure_churned = work.loc[work["_churn_flag_internal"] == 1, tenure_col].mean() if tenure_col else None

            avg_tenure_stayed = work.loc[work["_churn_flag_internal"] == 0, tenure_col].mean() if tenure_col else None

            if avg_tenure_churned is not None and avg_tenure_stayed is not None:

                st.caption(f"Churned customers average **{avg_tenure_churned:.1f}** months of tenure, versus "

                           f"**{avg_tenure_stayed:.1f}** months for retained customers.")

            st.markdown("</div>", unsafe_allow_html=True)



    # --- Revenue exposure ---

    revenue_col = find_column(df, "total_revenue", "Total_Revenue", "totalcharges", "total_charges")

    if revenue_col:

        with next_col():

            st.markdown('<div class="app-card">', unsafe_allow_html=True)

            st.markdown("##### Revenue Exposure")

            exposed_revenue = work.loc[work["_churn_flag_internal"] == 1, revenue_col].sum()

            total_revenue = work[revenue_col].sum()

            exposure_pct = (exposed_revenue / total_revenue * 100) if total_revenue else 0

            fig = px.pie(

                names=["At-Risk Revenue (Churned)", "Retained Revenue"],

                values=[exposed_revenue, total_revenue - exposed_revenue],

                hole=0.55, title="Revenue Exposure to Churn",

                color_discrete_sequence=[theme["danger"], theme["accent"]],

            )

            fig = style_plotly(fig, theme)

            st.plotly_chart(fig, use_container_width=True)

            st.caption(f"Churned customers represent **${exposed_revenue:,.0f}** "

                       f"({exposure_pct:.1f}%) of total tracked revenue.")

            st.markdown("</div>", unsafe_allow_html=True)



    # --- Payment method pattern ---

    payment_col = find_column(df, "paymentmethod", "Payment_Method")

    if payment_col:

        with next_col():

            st.markdown('<div class="app-card">', unsafe_allow_html=True)

            st.markdown("##### Payment Method & Churn")

            rate = work.groupby(payment_col)["_churn_flag_internal"].mean().sort_values(ascending=False) * 100

            fig = px.bar(x=rate.index, y=rate.values, labels={"x": "Payment Method", "y": "Churn Rate (%)"},

                         title="Churn Rate by Payment Method")

            fig = style_plotly(fig, theme)

            st.plotly_chart(fig, use_container_width=True)

            st.caption(f"**{rate.index[0]}** shows the highest churn rate among payment methods "

                       f"({rate.iloc[0]:.1f}%).")

            st.markdown("</div>", unsafe_allow_html=True)



    # --- Service-level pattern ---

    internet_col = find_column(df, "internetservice", "Internet_Service", "internet_type", "Internet_Type")

    if internet_col:

        with next_col():

            st.markdown('<div class="app-card">', unsafe_allow_html=True)

            st.markdown("##### Service Type & Churn")

            rate = work.groupby(internet_col)["_churn_flag_internal"].mean().sort_values(ascending=False) * 100

            fig = px.bar(x=rate.index, y=rate.values, labels={"x": "Internet Service", "y": "Churn Rate (%)"},

                         title="Churn Rate by Internet Service Type")

            fig = style_plotly(fig, theme)

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)



    # --- Monthly charges pattern ---

    charges_col = find_column(df, "monthlycharges", "Monthly_Charge")

    if charges_col:

        with next_col():

            st.markdown('<div class="app-card">', unsafe_allow_html=True)

            st.markdown("##### Monthly Charges & Churn")

            fig = px.box(

                work, x="_churn_flag_internal", y=charges_col,

                labels={"_churn_flag_internal": "Churned (1) vs Stayed (0)"},

                title="Monthly Charges Distribution by Churn Outcome",

            )

            fig = style_plotly(fig, theme)

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)



    st.markdown("<br>", unsafe_allow_html=True)

    render_retention_recommendations()





def render_retention_recommendations():

    st.markdown('<div class="section-header">Retention Recommendations</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-caption">General business recommendations by risk tier - '

                'not guaranteed outcomes.</div>', unsafe_allow_html=True)



    tiers = [

        ("VERY HIGH RISK", "badge-veryhigh", get_recommendation("VERY HIGH RISK")),

        ("HIGH RISK", "badge-high", get_recommendation("HIGH RISK")),

        ("MEDIUM RISK", "badge-medium", get_recommendation("MEDIUM RISK")),

        ("LOW RISK", "badge-low", get_recommendation("LOW RISK")),

    ]



    cols = st.columns(4)

    for col, (label, badge_class, recs) in zip(cols, tiers):

        with col:

            st.markdown('<div class="app-card">', unsafe_allow_html=True)

            st.markdown(f'<span class="badge {badge_class}">{label}</span>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            for r in recs:

                st.markdown(f"- {r}")

            st.markdown("</div>", unsafe_allow_html=True)





# ----------------------------------------------------------------------

# 18. ABOUT PAGE

# ----------------------------------------------------------------------



def render_about_page(schema, target_col, target_reason, quality):

    """

    Render the About page for Customer Churn AI.



    This page explains:

    - Project purpose

    - Business problem

    - End-to-end ML pipeline

    - Dataset structure

    - Feature audit

    - Preprocessing

    - Machine learning

    - Risk framework

    - Explainability

    - Business value

    - Technology stack

    - Privacy

    - Limitations

    - Developer information

    """



    # ==============================================================

    # PAGE HEADER

    # ==============================================================



    st.title("About Customer Churn AI")



    st.caption(

        "A transparent, dataset-driven customer risk intelligence "

        "platform for churn prediction and retention analysis."

    )



    st.divider()



    # ==============================================================

    # PROJECT HERO

    # ==============================================================



    st.header("🤖 Customer Churn AI")



    st.subheader(

        "Customer Risk Prediction & Retention Intelligence"

    )



    st.markdown(

        """

        **Data → Analysis → Machine Learning → Prediction

        → Explainability → Retention**

        """

    )



    st.info(

        """

        **Customer Churn AI** is an interactive machine learning

        application designed to analyze customer behaviour, identify

        churn patterns, estimate individual churn probability, and

        support data-driven retention decisions.



        The application works directly with the loaded customer churn

        dataset and dynamically inspects the available schema instead

        of depending on a fixed set of hard-coded customer columns.

        """

    )



    # ==============================================================

    # PROJECT CAPABILITIES

    # ==============================================================



    st.header("📌 What This Application Does")



    capability_col1, capability_col2, capability_col3 = st.columns(3)



    with capability_col1:

        st.markdown("### 📊 Analyze")

        st.write(

            "Explore customer behaviour, churn patterns, "

            "segments and dataset quality."

        )



    with capability_col2:

        st.markdown("### 🤖 Predict")

        st.write(

            "Estimate customer churn probability using "

            "machine learning classification models."

        )



    with capability_col3:

        st.markdown("### 💼 Act")

        st.write(

            "Translate customer risk into practical "

            "retention-oriented recommendations."

        )



    # ==============================================================

    # BUSINESS PROBLEM

    # ==============================================================



    st.header("🎯 Business Problem")



    st.markdown(

        """

        Customer churn can affect recurring revenue, customer lifetime

        value, acquisition efficiency and long-term business growth.



        Traditional reporting answers questions such as:



        **"Which customers have already churned?"**



        Predictive analytics extends this to:



        **"Which customers currently show a higher predicted probability

        of churn?"**



        The purpose of this application is to help analysts and business

        teams identify risk patterns and prioritize retention attention

        using data-driven evidence.

        """

    )



    # ==============================================================

    # END-TO-END PIPELINE

    # ==============================================================



    st.header("🔄 End-to-End Data & ML Pipeline")



    pipeline = [

        (

            "01",

            "DATA",

            "Load the customer churn CSV."

        ),

        (

            "02",

            "QUALITY",

            "Validate rows, columns, missing values and duplicates."

        ),

        (

            "03",

            "SCHEMA",

            "Detect target, IDs, numerical and categorical fields."

        ),

        (

            "04",

            "LEAKAGE",

            "Identify fields that should not enter the model."

        ),

        (

            "05",

            "PREPROCESS",

            "Impute, scale and encode model features."

        ),

        (

            "06",

            "TRAIN",

            "Train the available classification models."

        ),

        (

            "07",

            "EVALUATE",

            "Measure performance on hold-out data."

        ),

        (

            "08",

            "PREDICT",

            "Score customer information entered by the user."

        ),

        (

            "09",

            "EXPLAIN",

            "Identify important model-associated factors."

        ),

        (

            "10",

            "ACT",

            "Translate risk into retention recommendations."

        ),

    ]



    for number, title, description in pipeline:

        with st.container(border=True):

            col1, col2 = st.columns([1, 8])



            with col1:

                st.markdown(f"### `{number}`")



            with col2:

                st.markdown(f"**{title}**")

                st.caption(description)



    # ==============================================================

    # DATASET OVERVIEW

    # ==============================================================



    st.header("🗄 Dataset Overview")



    feature_cols = schema.get("feature_cols", [])

    numeric_cols = schema.get("numeric_cols", [])

    categorical_cols = schema.get("categorical_cols", [])



    dataset_col1, dataset_col2, dataset_col3 = st.columns(3)



    with dataset_col1:

        st.metric(

            "Model Features",

            len(feature_cols),

        )



    with dataset_col2:

        st.metric(

            "Numerical Features",

            len(numeric_cols),

        )



    with dataset_col3:

        st.metric(

            "Categorical Features",

            len(categorical_cols),

        )



    st.caption(

        "Source dataset: Customer_Churn_Predictions.csv"

    )



    # ==============================================================

    # TARGET DETECTION

    # ==============================================================



    st.header("🎯 Target Detection")



    if target_col:



        st.success(

            f"Churn target detected: `{target_col}`"

        )



        if target_reason:

            st.caption(target_reason)



    else:



        st.warning(

            "A valid churn target was not detected."

        )



    # ==============================================================

    # FEATURE AUDIT

    # ==============================================================



    st.header("🔍 Model Feature Audit")



    st.write(

        """

        Before training, the application evaluates the dataset schema

        to reduce the risk of identifiers, prediction artifacts and

        leakage-prone fields entering the model.

        """

    )



    id_cols = schema.get("id_cols", [])

    leakage_cols = schema.get("leakage_cols", [])

    constant_cols = schema.get("constant_cols", [])



    audit_col1, audit_col2, audit_col3 = st.columns(3)



    with audit_col1:



        st.subheader("ID Columns")



        if id_cols:

            for column in id_cols:

                st.code(column)

        else:

            st.success("None detected")



    with audit_col2:



        st.subheader("Leakage Columns")



        if leakage_cols:

            for column in leakage_cols:

                st.code(column)

        else:

            st.success("None detected")



    with audit_col3:



        st.subheader("Constant Columns")



        if constant_cols:

            for column in constant_cols:

                st.code(column)

        else:

            st.success("None detected")



    # ==============================================================

    # PREPROCESSING

    # ==============================================================



    st.header("⚙️ Data Preprocessing")



    preprocessing_col1, preprocessing_col2 = st.columns(2)



    with preprocessing_col1:



        st.subheader("Numerical Features")



        st.markdown(

            """

            - Missing-value imputation

            - Numerical scaling

            - Model-ready transformation

            """

        )



    with preprocessing_col2:



        st.subheader("Categorical Features")



        st.markdown(

            """

            - Missing-value imputation

            - One-hot encoding

            - Unknown-category handling

            """

        )



    st.info(

        """

        The preprocessing pipeline is fitted during training and reused

        during prediction. This keeps transformations consistent between

        training data and user-entered customer information.

        """

    )



    # ==============================================================

    # MACHINE LEARNING

    # ==============================================================



    st.header("🤖 Machine Learning")



    st.markdown(

        """

        The application treats churn prediction as a supervised

        classification problem.



        Depending on the installed libraries and current application

        configuration, the platform can support multiple model families.

        """

    )



    ml_col1, ml_col2, ml_col3, ml_col4 = st.columns(4)



    with ml_col1:

        st.metric("Problem", "Classification")



    with ml_col2:

        st.metric("Preprocessing", "Pipeline")



    with ml_col3:

        st.metric("Evaluation", "Hold-out")



    with ml_col4:

        st.metric("Output", "Probability")



    st.markdown(

        """

        **Supported model families**



        - Logistic Regression

        - Random Forest

        - Gradient Boosting

        - XGBoost

        - LightGBM

        - CatBoost



        Model performance should be assessed using multiple metrics rather

        than accuracy alone.

        """

    )



    # ==============================================================

    # RISK FRAMEWORK

    # ==============================================================



    st.header("🚦 Customer Risk Framework")



    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)



    with risk_col1:

        st.metric(

            "LOW",

            "Lower Risk"

        )

        st.caption(

            "Lower predicted churn probability."

        )



    with risk_col2:

        st.metric(

            "MEDIUM",

            "Moderate Risk"

        )

        st.caption(

            "Moderate predicted churn probability."

        )



    with risk_col3:

        st.metric(

            "HIGH",

            "Elevated Risk"

        )

        st.caption(

            "Elevated predicted churn probability."

        )



    with risk_col4:

        st.metric(

            "VERY HIGH",

            "Priority Risk"

        )

        st.caption(

            "Highest predicted churn probability."

        )



    st.caption(

        "Risk bands are determined by the probability thresholds "

        "configured in the application."

    )



    # ==============================================================

    # DATA QUALITY

    # ==============================================================



    st.header("🧪 Data Quality")



    # --------------------------------------------------------------

    # Handle quality as dictionary

    # --------------------------------------------------------------



    if isinstance(quality, dict):



        quality_col1, quality_col2, quality_col3, quality_col4 = (

            st.columns(4)

        )



        with quality_col1:

            st.metric(

                "Rows",

                f"{quality.get('n_rows', 0):,}"

            )



        with quality_col2:

            st.metric(

                "Columns",

                f"{quality.get('n_cols', 0):,}"

            )



        with quality_col3:

            st.metric(

                "Missing Values",

                f"{quality.get('missing_total', 0):,}"

            )



        with quality_col4:

            st.metric(

                "Duplicate Rows",

                f"{quality.get('duplicate_rows', 0):,}"

            )



        quality_col5, quality_col6, quality_col7 = st.columns(3)



        with quality_col5:

            st.metric(

                "Numerical",

                f"{quality.get('numeric_count', 0):,}"

            )



        with quality_col6:

            st.metric(

                "Categorical",

                f"{quality.get('categorical_count', 0):,}"

            )



        with quality_col7:

            rating = quality.get(

                "rating",

                "Not Available"

            )



            st.metric(

                "Quality Rating",

                str(rating)

            )



    else:



        st.info(

            "Detailed dataset quality information is currently "

            "unavailable."

        )



    # ==============================================================

    # EXPLAINABLE AI

    # ==============================================================



    st.header("🔍 Explainable AI")



    st.markdown(

        """

        The application is designed to make model predictions easier

        to interpret.



        Where supported, model feature importance and SHAP-based

        explanations can be used to identify factors associated with

        individual predictions.



        These explanations describe **model behaviour**.



        They should not automatically be interpreted as proof that a

        particular feature caused a customer to churn.

        """

    )



    # ==============================================================

    # BUSINESS VALUE

    # ==============================================================



    st.header("💼 Business Value")



    value_col1, value_col2, value_col3, value_col4 = st.columns(4)



    with value_col1:

        st.subheader("Identify")

        st.caption(

            "Find customers and segments with elevated predicted risk."

        )



    with value_col2:

        st.subheader("Prioritize")

        st.caption(

            "Focus retention attention where risk is higher."

        )



    with value_col3:

        st.subheader("Understand")

        st.caption(

            "Analyze patterns associated with observed churn."

        )



    with value_col4:

        st.subheader("Act")

        st.caption(

            "Support targeted retention decisions."

        )



    # ==============================================================

    # TECHNOLOGY STACK

    # ==============================================================



    st.header("🛠 Technology Stack")



    tech_col1, tech_col2, tech_col3 = st.columns(3)



    with tech_col1:



        st.subheader("Programming & Data")



        st.write(

            "Python • Pandas • NumPy"

        )



    with tech_col2:



        st.subheader("Machine Learning")



        st.write(

            "Scikit-Learn • XGBoost • LightGBM • CatBoost"

        )



    with tech_col3:



        st.subheader("Application")



        st.write(

            "Streamlit • Plotly • SHAP"

        )



    # ==============================================================

    # PRIVACY

    # ==============================================================



    st.header("🔐 Privacy & Data Processing")



    st.info(

        """

        The application is designed to process the provided customer

        dataset within the Streamlit application environment.



        Customer data is not intentionally sent to an external API by

        this application.



        Avoid uploading sensitive production customer information unless

        the deployment environment and organizational privacy requirements

        have been reviewed and approved.

        """

    )



    # ==============================================================

    # LIMITATIONS

    # ==============================================================



    st.header("⚠️ Important Limitations")



    limitations = [

        "Model predictions are probabilistic, not guarantees.",

        "Feature importance does not automatically establish causality.",

        "Historical patterns may not represent future customer behaviour.",

        "Model performance depends on data quality and target definition.",

        "Retention recommendations should be reviewed by business teams.",

        "Production models require monitoring and periodic validation.",

    ]



    for limitation in limitations:

        st.markdown(f"- {limitation}")



    # ==============================================================

    # DEVELOPER

    # ==============================================================



    st.header("👨‍💻 Developer")



    developer_col1, developer_col2 = st.columns([2, 1])



    with developer_col1:



        st.subheader(DEVELOPER_NAME)



        st.caption(

            "Data Science • Machine Learning • Analytics"

        )



        st.write(

            """

            This project demonstrates an end-to-end approach to

            customer churn analytics, predictive modelling,

            explainability and business-focused decision support.

            """

        )



    with developer_col2:



        st.link_button(

            "GitHub Project",

            "https://github.com/"

            "Shaik-Mohammed-Kaif/"

            "Data-Science-Analyst-Project",

            use_container_width=True,

        )



        st.link_button(

            "LinkedIn Profile",

            "https://www.linkedin.com/"

            "in/s-mohammed-kaif-2a500a341/",

            use_container_width=True,

        )



    # ==============================================================

    # FINAL PROJECT STATEMENT

    # ==============================================================



    st.divider()



    st.markdown(

        """

        <div style="text-align:center;">



        ### 🤖 CUSTOMER CHURN AI



        **Turning customer data into actionable risk intelligence.**



        *DATA SCIENCE • MACHINE LEARNING • BUSINESS INTELLIGENCE*



        </div>

        """,

        unsafe_allow_html=True,

    )





# ----------------------------------------------------------------------

# 19. PROFESSIONAL FOOTER

# ----------------------------------------------------------------------



def render_footer():

    """

    Render the Customer Churn AI professional footer.



    Uses only native Streamlit components.

    No raw HTML is used, preventing HTML code from appearing

    as visible text in the application.

    """



    # ==============================================================

    # FOOTER DIVIDER

    # ==============================================================



    st.divider()



    # ==============================================================

    # PROJECT BRAND

    # ==============================================================



    st.markdown(

        """

        ### 🤖 CUSTOMER CHURN AI

        """,

    )



    st.caption(

        "Customer Risk Prediction & Retention Intelligence"

    )



    st.markdown(

        """

        **Data** → **Intelligence** → **Prediction** → **Retention**

        """

    )



    # ==============================================================

    # BUILT WITH

    # ==============================================================



    st.markdown("#### Built with")



    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)



    with tech_col1:

        st.markdown("🐍 **Python**")

        st.markdown("🐼 **Pandas**")



    with tech_col2:

        st.markdown("🧠 **Scikit-Learn**")

        st.markdown("📊 **Plotly**")



    with tech_col3:

        st.markdown("🚀 **Streamlit**")

        st.markdown("⚡ **XGBoost**")



    with tech_col4:

        st.markdown("🌿 **LightGBM**")

        st.markdown("🐱 **CatBoost**")



    # ==============================================================

    # DEVELOPER

    # ==============================================================



    st.markdown("#### Developed by")



    developer_left, developer_center, developer_right = st.columns(

        [1, 2, 1]

    )



    with developer_center:

        st.markdown(

            f"### {DEVELOPER_NAME}"

        )



    # ==============================================================

    # CONNECT & EXPLORE

    # ==============================================================



    st.divider()



    connect_left, connect_center, connect_right = st.columns(

        [1, 2, 1]

    )



    with connect_center:

        st.markdown(

            "### 🔗 Connect & Explore"

        )



    # ==============================================================

    # SOCIAL LINKS

    # ==============================================================



    github_url = (

        "https://github.com/Shaik-Mohammed-Kaif/"

        "Data-Science-Analyst-Project"

    )



    linkedin_url = (

        "https://www.linkedin.com/in/"

        "s-mohammed-kaif-2a500a341/"

    )



    github_left, github_col, linkedin_col, linkedin_right = st.columns(

        [1, 2, 2, 1]

    )



    with github_col:



        st.link_button(

            "🐙  GitHub",

            github_url,

            use_container_width=True,

        )



    with linkedin_col:



        st.link_button(

            "in  LinkedIn",

            linkedin_url,

            use_container_width=True,

        )



    # ==============================================================

    # COPYRIGHT

    # ==============================================================



    st.divider()



    copyright_left, copyright_center, copyright_right = st.columns(

        [1, 3, 1]

    )



    with copyright_center:



        st.caption(

            f"© {COPYRIGHT_YEAR} {DEVELOPER_NAME} "

            "• Customer Churn AI"

        )



# ----------------------------------------------------------------------
# 20. MAIN APPLICATION DRIVER
# ----------------------------------------------------------------------

def main():
    init_session_state()
    render_theme()

    csv_path = find_csv_path(CSV_FILENAME)

    if csv_path is None:
        render_hero(csv_ok=False, model_ok=False)
        st.error(
            f"**{CSV_FILENAME} was not found.** Please place the dataset in the project "
            f"dataset directory:\n\n"
            f"`Churn_Analysis_Project_Data_Analyst/Churn-Dataset-With-Geo-Loc-Analysis/{CSV_FILENAME}`",
            icon="\U0001F6AB",
        )
        render_sidebar(None, None, None, None, None, None)
        render_footer()
        return

    df, load_error = load_data(csv_path)

    if load_error or df is None:
        render_hero(csv_ok=False, model_ok=False)
        st.error(load_error or "The dataset could not be loaded.", icon="\U0001F6AB")
        render_sidebar(None, None, None, None, None, None)
        render_footer()
        return

    # ---- Schema / target detection --------------------------------
    target_override = st.session_state.get("target_column_override")
    target_col, target_reason = detect_target_column(df, override=target_override)

    schema = None
    training_result = None

    if target_col:
        schema = identify_feature_columns(df, target_col)

        df_signature = f"{df.shape}-{'-'.join(df.columns)}-{target_col}-{st.session_state['model_choice']}"
        training_result = train_model(
            df_signature, df, target_col,
            tuple(schema["numeric_cols"]), tuple(schema["categorical_cols"]),
            tuple(schema["feature_cols"]), st.session_state["model_choice"],
        )
    else:
        schema = identify_feature_columns(df, df.columns[-1]) if len(df.columns) else {
            "feature_cols": [], "numeric_cols": [], "categorical_cols": [],
            "id_cols": [], "leakage_cols": [], "constant_cols": [],
        }

    model_ready = bool(training_result and training_result.get("success"))

    # ---- Sidebar -----------------------------------------------------
    render_sidebar(df, csv_path, schema, target_col, target_reason, training_result)

    # ---- Hero ----------------------------------------------------------
    render_hero(csv_ok=True, model_ok=model_ready)

    if not target_col:
        st.warning(
            "Prediction training cannot be started because a valid churn target column "
            "could not be identified in the uploaded dataset. You can manually choose a "
            "target column from the sidebar under **Target column override**.",
            icon="\u26A0\uFE0F",
        )

    # ---- Navigation ------------------------------------------------
    page = render_navigation()
    st.markdown("<hr style='margin-top:0;'>", unsafe_allow_html=True)

    quality = compute_data_quality(df)

    # ---- Page routing -----------------------------------------------
    if page == "Overview":
        render_overview_page(df, target_col, quality)
    elif page == "Data Explorer":
        render_data_explorer_page(df, schema)
    elif page == "Prediction":
        render_prediction_page(df, schema, training_result)
    elif page == "Model Analysis":
        render_model_analysis_page(training_result, target_col, target_reason)
    elif page == "Business Insights":
        render_business_insights_page(df, target_col, schema)
    elif page == "About":
        render_about_page(schema, target_col, target_reason, quality)

    render_footer()


if __name__ == "__main__":
    main()