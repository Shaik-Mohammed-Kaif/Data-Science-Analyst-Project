# ================================================================
# SUPERSTORE AI INTELLIGENCE PLATFORM
# End-to-end Streamlit ML Classification + Business Intelligence
# Target: Return_Flag
# Dataset: SuperStore_Feature_Engineered.csv
# ================================================================

import os
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance

# Optional models
try:
    from sklearn.svm import SVC
    SVC_AVAILABLE = True
except Exception:
    SVC_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# ================================================================
# CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="SuperStore AI Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DATA_PATH = "SuperStore_Feature_Engineered.csv"
TARGET_COLUMN = "Return_Flag"

LINKEDIN_URL = "https://www.linkedin.com/in/s-mohammed-kaif-2a500a341"
GITHUB_URL = "https://github.com/Shaik-Mohammed-Kaif"

PRIMARY_METRIC = "F1"
RANDOM_STATE = 42


# ================================================================
# THEME STATE
# ================================================================

THEMES = {
    "🤍 White Cream": {
        "bg": "#F7F2E8",
        "surface": "#FFFDF7",
        "surface2": "#F1E9DA",
        "text": "#171717",
        "muted": "#5F5A50",
        "border": "#D8CCB8",
        "accent": "#7A4E2D",
        "accent2": "#B27A45",
        "plot_bg": "#FFFDF7",
        "grid": "#DED5C7",
    },

    "🌑 Midnight": {
        "bg": "#0B1020",
        "surface": "#121A2B",
        "surface2": "#182238",
        "text": "#F5F7FB",
        "muted": "#B8C1D1",
        "border": "#2D3954",
        "accent": "#8AB4FF",
        "accent2": "#B48CFF",
        "plot_bg": "#121A2B",
        "grid": "#2B3850",
    },

    "🌊 Ocean": {
        "bg": "#071B26",
        "surface": "#0D2B3A",
        "surface2": "#123B4E",
        "text": "#F2FBFF",
        "muted": "#B7D3DE",
        "border": "#275466",
        "accent": "#5DD6E7",
        "accent2": "#72A7FF",
        "plot_bg": "#0D2B3A",
        "grid": "#245164",
    },

    "🌿 Emerald": {
        "bg": "#081A13",
        "surface": "#0E291E",
        "surface2": "#143A2A",
        "text": "#F1FFF8",
        "muted": "#B9D8C8",
        "border": "#2A5540",
        "accent": "#6FE7A5",
        "accent2": "#9BD66F",
        "plot_bg": "#0E291E",
        "grid": "#2A5540",
    },

    "🍷 Burgundy": {
        "bg": "#1A0A10",
        "surface": "#2A111A",
        "surface2": "#3A1723",
        "text": "#FFF6F8",
        "muted": "#D9BCC5",
        "border": "#5D2C3C",
        "accent": "#F19AB2",
        "accent2": "#E4B37A",
        "plot_bg": "#2A111A",
        "grid": "#593040",
    },
}


if "theme_name" not in st.session_state:
    st.session_state.theme_name = "🤍 White Cream"

if "nav_open" not in st.session_state:
    st.session_state.nav_open = False


T = THEMES[
    st.session_state.theme_name
]


# ================================================================
# PERSISTENT MODEL CONFIGURATION
# ================================================================

MODEL_DIR = Path("saved_models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------
# THIS IS THE ONLY MODEL FILE USED BY THE APP
# ------------------------------------------------

MODEL_BUNDLE_PATH = (
    MODEL_DIR /
    "superstore_return_model.joblib"
)


# ================================================================
# LOAD SAVED MODEL
# ================================================================

def load_saved_model_bundle():
    """
    Load the already-trained model from disk.

    The application NEVER trains a model here.

    Returns:
        dict : valid trained model bundle
        None : if the model does not exist or is invalid
    """

    if not MODEL_BUNDLE_PATH.exists():
        return None

    try:

        bundle = joblib.load(
            MODEL_BUNDLE_PATH
        )

        # --------------------------------------------
        # Validate bundle
        # --------------------------------------------

        if not isinstance(
            bundle,
            dict
        ):
            return None

        required_keys = {
            "best_model",
            "best_name",
            "feature_cols",
            "results"
        }

        if not required_keys.issubset(
            bundle.keys()
        ):
            return None

        return bundle

    except Exception:
        return None


# ================================================================
# SAVE TRAINED MODEL
# ================================================================

def save_model_bundle(bundle):
    """
    Save newly trained model bundle to the single
    persistent model location.
    """

    try:

        joblib.dump(
            bundle,
            MODEL_BUNDLE_PATH
        )

        return True, None

    except Exception as exc:

        return False, str(exc)


# ================================================================
# SESSION STATE INITIALIZATION
# ================================================================

if "model_results" not in st.session_state:
    st.session_state.model_results = None

if "trained_bundle" not in st.session_state:
    st.session_state.trained_bundle = None


# ================================================================
# AUTOMATIC MODEL INITIALIZATION
# ================================================================

def initialize_model():

    # --------------------------------------------
    # Already available in current session
    # --------------------------------------------

    existing = st.session_state.get(
        "model_results"
    )

    if existing is not None:
        return existing

    # --------------------------------------------
    # Load existing .joblib from disk
    # --------------------------------------------

    saved_bundle = load_saved_model_bundle()

    if saved_bundle is not None:

        st.session_state.model_results = (
            saved_bundle
        )

        st.session_state.trained_bundle = (
            saved_bundle
        )

        return saved_bundle

    return None


# ================================================================
# ACTIVE MODEL ACCESS
# ================================================================

def get_active_model_bundle():
    """
    Get the existing trained model.

    Priority:

    1. Current Streamlit session
    2. saved_models/superstore_return_model.joblib

    NO TRAINING IS PERFORMED.
    """

    # --------------------------------------------
    # Fast path
    # --------------------------------------------

    bundle = st.session_state.get(
        "model_results"
    )

    if bundle is not None:
        return bundle

    # --------------------------------------------
    # Disk fallback
    # --------------------------------------------

    bundle = load_saved_model_bundle()

    if bundle is not None:

        st.session_state.model_results = (
            bundle
        )

        st.session_state.trained_bundle = (
            bundle
        )

        return bundle

    return None


# ================================================================
# LOAD MODEL ON APPLICATION START
# ================================================================

initialize_model()


# ================================================================
# GLOBAL CSS
# ================================================================

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
    --bg: {T["bg"]};
    --surface: {T["surface"]};
    --surface2: {T["surface2"]};
    --text: {T["text"]};
    --muted: {T["muted"]};
    --border: {T["border"]};
    --accent: {T["accent"]};
    --accent2: {T["accent2"]};
}}

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background:
        radial-gradient(circle at 10% 10%, rgba(128,128,128,.08) 0 90px, transparent 92px),
        radial-gradient(circle at 90% 20%, rgba(128,128,128,.07) 0 120px, transparent 122px),
        radial-gradient(circle at 30% 90%, rgba(128,128,128,.06) 0 100px, transparent 102px),
        var(--bg);
    color: var(--text);
}}

.stApp::before,
.stApp::after {{
    content: "";
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    opacity: .12;
}}

.stApp::before {{
    width: 180px;
    height: 180px;
    right: 5%;
    top: 18%;
    background: var(--accent);
    filter: blur(4px);
    animation: floatOne 12s ease-in-out infinite;
}}

.stApp::after {{
    width: 130px;
    height: 130px;
    left: 4%;
    bottom: 10%;
    background: var(--accent2);
    filter: blur(5px);
    animation: floatTwo 15s ease-in-out infinite;
}}

@keyframes floatOne {{
    0%,100% {{ transform: translate(0,0); }}
    50% {{ transform: translate(-25px,35px); }}
}}

@keyframes floatTwo {{
    0%,100% {{ transform: translate(0,0); }}
    50% {{ transform: translate(30px,-25px); }}
}}

.block-container {{
    max-width: 1500px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}}

header[data-testid="stHeader"] {{
    background: transparent;
}}

section[data-testid="stSidebar"] {{
    background: var(--surface);
    border-right: 1px solid var(--border);
}}

section[data-testid="stSidebar"] * {{
    color: var(--text) !important;
}}

button {{
    border-radius: 10px !important;
}}

.stButton > button {{
    font-weight: 700;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
}}

.stButton > button:hover {{
    border-color: var(--accent);
    color: var(--accent);
}}

div[data-testid="stMetric"] {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 14px;
}}

div[data-testid="stMetric"] label {{
    color: var(--muted) !important;
    font-weight: 700;
}}

div[data-testid="stMetricValue"] {{
    color: var(--text) !important;
    font-weight: 800;
}}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea {{
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}}

label, .stMarkdown, p, li, h1, h2, h3, h4 {{
    color: var(--text);
}}

.section-label {{
    margin: 24px 0 12px 0;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.7px;
    color: var(--accent);
    text-transform: uppercase;
}}

.hero {{
    background: linear-gradient(135deg, var(--surface), var(--surface2));
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 30px;
    margin-bottom: 22px;
    box-shadow: 0 15px 45px rgba(0,0,0,.08);
}}

.hero-title {{
    font-size: clamp(30px, 4vw, 52px);
    line-height: 1.05;
    font-weight: 800;
    color: var(--text);
}}

.hero-subtitle {{
    margin-top: 12px;
    max-width: 950px;
    color: var(--muted);
    font-size: 16px;
    line-height: 1.7;
}}

.card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 22px;
    margin: 10px 0;
    box-shadow: 0 10px 28px rgba(0,0,0,.06);
}}

.kpi-card {{
    min-height: 150px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,.06);
}}

.kpi-label {{
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.1px;
    color: var(--muted);
}}

.kpi-value {{
    margin-top: 12px;
    font-size: 27px;
    font-weight: 800;
    color: var(--text);
    overflow-wrap: anywhere;
}}

.kpi-description {{
    margin-top: 8px;
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
}}

.nav-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 10px;
    margin-bottom: 12px;
}}

.footer {{
    margin-top: 45px;
    padding: 25px;
    text-align: center;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    color: var(--muted);
}}

.small-note {{
    color: var(--muted);
    font-size: 12px;
}}

[data-testid="stDataFrame"] {{
    border: 1px solid var(--border);
    border-radius: 14px;
}}

div[data-testid="stExpander"] {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
}}

.stTabs [data-baseweb="tab"] {{
    color: var(--muted);
    font-weight: 700;
}}

.stTabs [aria-selected="true"] {{
    color: var(--accent) !important;
}}
</style>
""",
    unsafe_allow_html=True
)


# ================================================================
# HELPERS
# ================================================================

def safe_unique(series):
    return sorted(series.dropna().astype(str).unique().tolist())


def money(x):
    return f"${x:,.2f}"


def pct(x):
    return f"{x:.2f}%"


def layout(fig, height=450):
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor=T["surface"],
        plot_bgcolor=T["plot_bg"],
        font=dict(color=T["text"], family="Inter"),
        margin=dict(l=40, r=25, t=55, b=45),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=T["text"])
        ),
        xaxis=dict(
            gridcolor=T["grid"],
            zerolinecolor=T["grid"]
        ),
        yaxis=dict(
            gridcolor=T["grid"],
            zerolinecolor=T["grid"]
        )
    )
    return fig


def card_kpi(label, value, description, icon):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div class="kpi-label">{label}</div>
                <div style="font-size:22px;">{icon}</div>
            </div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def clean_target(series):
    if pd.api.types.is_numeric_dtype(series):
        vals = sorted(series.dropna().unique().tolist())
        if len(vals) == 2:
            return series, vals

    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1, "y": 1, "true": 1, "1": 1,
        "returned": 1, "return": 1,
        "no": 0, "n": 0, "false": 0, "0": 0,
        "not returned": 0, "not_returned": 0,
    }

    mapped = s.map(mapping)

    if mapped.notna().all() and mapped.nunique() == 2:
        return mapped.astype(int), [0, 1]

    codes, uniques = pd.factorize(series.astype(str))
    if len(uniques) == 2:
        return pd.Series(codes, index=series.index), list(range(2))

    return None, []


def feature_columns(df):
    excluded_exact = {
        TARGET_COLUMN,
        "Return_Flag",
        "Returns",
    }

    excluded_patterns = [
        "_Total_",
        "Customer_Total_",
        "Product_Total_",
        "Region_Total_",
        "State_Total_",
        "City_Total_",
        "Segment_Total_",
        "Category_Total_",
        "SubCategory_Total_",
    ]

    cols = []
    for c in df.columns:
        if c in excluded_exact:
            continue

        # IDs/names are generally identifiers rather than stable predictors.
        low = c.lower()
        if low.endswith("_id") or low in {
            "row_id", "order_id", "customer_name", "product_name"
        }:
            continue

        # Date is converted to date-part features separately.
        if c == "Order_Date" or c == "Ship_Date":
            continue

        if any(p.lower() in low for p in [x.lower() for x in excluded_patterns]):
            continue

        cols.append(c)

    return cols


def prepare_features(df, cols):
    X = df[cols].copy()

    for c in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            X[c] = X[c].astype(str)

        if c in ["Order_Date", "Ship_Date"]:
            X[c] = pd.to_datetime(X[c], errors="coerce")
            X[c + "_year"] = X[c].dt.year
            X[c + "_month"] = X[c].dt.month
            X[c + "_day"] = X[c].dt.day
            X[c + "_dow"] = X[c].dt.dayofweek
            X.drop(columns=[c], inplace=True)

    # Convert object columns to string for stable OneHotEncoder handling.
    for c in X.select_dtypes(include=["object", "category"]).columns:
        X[c] = X[c].fillna("Missing").astype(str)

    return X


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [
        c for c in X.columns if c not in numeric_cols
    ]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


def build_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=350,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.07,
            random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=9,
            weights="distance"
        )
    }

    if SVC_AVAILABLE:
        models["SVM"] = SVC(
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    return models


def train_models(df, target, feature_cols, test_size):
    X = prepare_features(df, feature_cols)
    y, target_values = clean_target(df[target])

    if y is None:
        raise ValueError(
            f"'{target}' must contain exactly two classes."
        )

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify
    )

    preprocessor = build_preprocessor(X)
    models = build_models()

    results = []
    trained = {}

    for name, estimator in models.items():
        try:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator)
            ])

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X_test)[:, 1]
            else:
                proba = None

            accuracy = accuracy_score(y_test, pred)
            precision = precision_score(
                y_test, pred, average="binary", zero_division=0
            )
            recall = recall_score(
                y_test, pred, average="binary", zero_division=0
            )
            f1 = f1_score(
                y_test, pred, average="binary", zero_division=0
            )

            if proba is not None and len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, proba)
            else:
                auc = np.nan

            results.append({
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC-AUC": auc
            })
            trained[name] = pipe

        except Exception as e:
            results.append({
                "Model": name,
                "Accuracy": np.nan,
                "Precision": np.nan,
                "Recall": np.nan,
                "F1": np.nan,
                "ROC-AUC": np.nan,
                "Error": str(e)
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        ["F1", "ROC-AUC", "Accuracy"],
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    best_name = results_df.iloc[0]["Model"]
    best_model = trained[best_name]

    return {
        "results": results_df,
        "trained": trained,
        "best_name": best_name,
        "best_model": best_model,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "target_values": target_values,
        "target_series": y,
    }


def get_feature_importance(bundle):
    model = bundle["best_model"]
    X_test = bundle["X_test"]
    y_test = bundle["y_test"]

    try:
        r = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="f1",
            n_jobs=-1
        )

        imp = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": r.importances_mean
        }).sort_values(
            "Importance", ascending=False
        )

        return imp.head(20)

    except Exception:
        return pd.DataFrame(
            columns=["Feature", "Importance"]
        )


def plot_confusion(bundle):
    model = bundle["best_model"]
    pred = model.predict(bundle["X_test"])
    cm = confusion_matrix(bundle["y_test"], pred)

    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        labels=dict(
            x="Predicted",
            y="Actual",
            color="Count"
        )
    )
    fig.update_layout(title="Confusion Matrix")
    return layout(fig, 420)


# ================================================================
# LOAD DATA
# ================================================================

@st.cache_data(show_spinner=False)
def load_data(path):
    if not os.path.exists(path):
        return None, f"CSV file not found: {path}"

    try:
        data = pd.read_csv(path)
    except Exception as e:
        return None, f"Could not read CSV: {e}"

    if "Order_Date" in data.columns:
        data["Order_Date"] = pd.to_datetime(
            data["Order_Date"], errors="coerce"
        )

    if "Ship_Date" in data.columns:
        data["Ship_Date"] = pd.to_datetime(
            data["Ship_Date"], errors="coerce"
        )

    return data, None


df, load_error = load_data(DATA_PATH)

if load_error:
    st.error(load_error)
    st.info(
        "Place 'SuperStore_Feature_Engineered.csv' in the same folder "
        "as this Python file, or change DATA_PATH at the top."
    )
    st.stop()

if TARGET_COLUMN not in df.columns:
    st.error(f"Target column '{TARGET_COLUMN}' was not found.")
    st.write("Available columns:")
    st.code(", ".join(df.columns))
    st.stop()


# ================================================================
# TOP HEADER
# ================================================================

header_left, header_right = st.columns(
    [6, 2],
    vertical_alignment="center"
)

with header_left:

    st.markdown(
        """
        <style>

        /* ========================================================
           MAIN AI TITLE
           ======================================================== */

        .top-ai-title {
            font-size: 38px !important;
            font-weight: 850 !important;
            line-height: 1.1 !important;
            letter-spacing: -0.8px !important;

            margin: 0 !important;
            padding: 0 !important;

            animation: aiTitleEnter 0.55s ease-out;
        }


        /* ========================================================
           PLATFORM NAME
           ======================================================== */

        .top-platform-name {
            font-size: 20px !important;
            font-weight: 850 !important;
            line-height: 1.3 !important;

            margin-top: 9px !important;
            margin-bottom: 3px !important;

            letter-spacing: -0.1px;
        }


        /* ========================================================
           PLATFORM DESCRIPTION
           ======================================================== */

        .top-ai-subtitle {
            font-size: 16px !important;
            font-weight: 450 !important;
            line-height: 1.55 !important;

            opacity: 0.74;

            margin: 0 !important;
            padding: 0 !important;

            letter-spacing: 0.05px;

            animation: aiSubtitleEnter 0.8s ease-out;
        }


        /* ========================================================
           SUBTLE CURSOR
           ======================================================== */

        .top-ai-subtitle::after {

            content: "▌";

            display: inline-block;

            margin-left: 5px;

            font-size: 14px;

            opacity: 0.65;

            animation: cursorBlink 1s steps(1) infinite;
        }


        /* ========================================================
           TITLE ENTRY ANIMATION
           ======================================================== */

        @keyframes aiTitleEnter {

            0% {
                opacity: 0;
                transform: translateY(8px);
            }

            100% {
                opacity: 1;
                transform: translateY(0);
            }

        }


        /* ========================================================
           SUBTITLE ENTRY ANIMATION
           ======================================================== */

        @keyframes aiSubtitleEnter {

            0% {
                opacity: 0;
                transform: translateY(5px);
            }

            100% {
                opacity: 0.74;
                transform: translateY(0);
            }

        }


        /* ========================================================
           CURSOR BLINK
           ======================================================== */

        @keyframes cursorBlink {

            0%,
            45% {
                opacity: 0.65;
            }

            46%,
            100% {
                opacity: 0;
            }

        }


        /* ========================================================
           NAVIGATION BUTTON
           ======================================================== */

        div[data-testid="column"]:last-child
        div.stButton > button {

            min-height: 46px;

            border-radius: 12px;

            font-size: 14px;

            font-weight: 700;

            padding: 8px 16px;

            transition:
                transform 0.18s ease,
                box-shadow 0.18s ease;
        }


        div[data-testid="column"]:last-child
        div.stButton > button:hover {

            transform: translateY(-2px);

            box-shadow:
                0 6px 16px rgba(0, 0, 0, 0.09);
        }


        /* ========================================================
           RESPONSIVE
           ======================================================== */

        @media (max-width: 900px) {

            .top-ai-title {
                font-size: 30px !important;
            }

            .top-platform-name {
                font-size: 18px !important;
            }

            .top-ai-subtitle {
                font-size: 14px !important;
            }

        }

        </style>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # MAIN TITLE
    # ============================================================

    st.markdown(
        """
        <div class="top-ai-title">
            🤖 AI Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # PLATFORM NAME
    # ============================================================

    st.markdown(
        """
        <div class="top-platform-name">
            SuperStore AI Intelligence Platform
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # DESCRIPTION
    # ============================================================

    st.markdown(
        """
        <div class="top-ai-subtitle">
            Predictive analytics, machine learning, business intelligence,
            model evaluation and data-driven decision support.
        </div>
        """,
        unsafe_allow_html=True
    )


# ================================================================
# RIGHT — NAVIGATION
# ================================================================

with header_right:

    if st.button(
        "☰  Navigation",
        use_container_width=True,
        key="top_navigation_button"
    ):

        st.session_state.nav_open = (
            not st.session_state.nav_open
        )

        st.rerun()


# ================================================================
# FRONT HORIZONTAL FILTERS
# ================================================================

st.markdown(
    '<div class="section-label">GLOBAL BUSINESS FILTERS</div>',
    unsafe_allow_html=True
)

filter_cols = st.columns(5)

filtered = df.copy()

filter_specs = [
    ("Region", "🌎 Region"),
    ("Category", "📦 Category"),
    ("Segment", "👥 Segment"),
    ("Ship_Mode", "🚚 Ship Mode"),
    ("Order_Year", "📅 Year"),
]

for col, (field, label) in zip(filter_cols, filter_specs):
    with col:
        if field in filtered.columns:
            options = safe_unique(filtered[field])
            selected = st.multiselect(
                label,
                options,
                default=[],
                key=f"filter_{field}"
            )
            if selected:
                if field == "Order_Year":
                    vals = [int(x) for x in selected]
                    filtered = filtered[
                        filtered[field].isin(vals)
                    ]
                else:
                    filtered = filtered[
                        filtered[field].astype(str).isin(
                            [str(x) for x in selected]
                        )
                    ]

st.caption(
    f"Showing {len(filtered):,} of {len(df):,} records after filters."
)


# ================================================================
# NAVIGATION
# ================================================================

pages = [
    "Home",
    "Business Dashboard",
    "EDA & Data Quality",
    "ML Model Lab",
    "Prediction Studio",
    "Model Evaluation",
    "Feature Intelligence",
    "Business Insights",
    "Data Explorer",
    "About"
]

if st.session_state.nav_open:
    st.markdown(
        '<div class="section-label">APPLICATION NAVIGATION</div>',
        unsafe_allow_html=True
    )
    page = st.selectbox(
        "Select application module",
        pages,
        index=0
    )
else:
    # Compact navigation always visible without a permanent sidebar.
    page = st.selectbox(
        "Application module",
        pages,
        index=0,
        label_visibility="collapsed"
    )


# ================================================================
# THEME CONTROLS
# ================================================================

with st.expander("🎨 Appearance & Theme"):
    theme = st.selectbox(
        "Theme",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme_name)
    )
    if theme != st.session_state.theme_name:
        st.session_state.theme_name = theme
        st.rerun()


# ================================================================
# COMMON BUSINESS KPIs
# ================================================================

sales = filtered["Sales"].sum() if "Sales" in filtered.columns else 0
profit = filtered["Profit"].sum() if "Profit" in filtered.columns else 0
quantity = filtered["Quantity"].sum() if "Quantity" in filtered.columns else 0
orders = (
    filtered["Order_ID"].nunique()
    if "Order_ID" in filtered.columns else len(filtered)
)
customers = (
    filtered["Customer_ID"].nunique()
    if "Customer_ID" in filtered.columns else 0
)

margin = (profit / sales * 100) if sales else 0
aov = (sales / orders) if orders else 0
ppo = (profit / orders) if orders else 0


# ================================================================
# HOME
# ================================================================

if page == "Home":
    st.markdown(
        '<div class="section-label">APPLICATION OVERVIEW</div>',
        unsafe_allow_html=True
    )

    # ============================================================
    # APPLICATION OVERVIEW
    # ============================================================

    k1, k2, k3, k4 = st.columns(4, gap="large")

    with k1:
        card_kpi(
            "RECORDS",
            f"{len(filtered):,}",
            "Filtered observations",
            "📊"
        )

    with k2:
        card_kpi(
            "FEATURES",
            f"{len(feature_columns(filtered)):,}",
            "Candidate predictor columns",
            "🧩"
        )

    with k3:
        card_kpi(
            "CLASSES",
            f"{df[TARGET_COLUMN].nunique(dropna=True):,}",
            "Target classes",
            "🎯"
        )

    with k4:
        card_kpi(
            "TARGET",
            TARGET_COLUMN,
            "Prediction variable",
            "🤖"
        )

    # VERTICAL GAP
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ============================================================
    # BUSINESS CORE KPIs — ROW 1
    # ============================================================

    st.markdown(
        '<div class="section-label">BUSINESS CORE KPIs</div>',
        unsafe_allow_html=True
    )

    a, b, c, d = st.columns(4, gap="large")

    with a:
        card_kpi(
            "TOTAL SALES",
            money(sales),
            "Revenue generated",
            "💵"
        )

    with b:
        card_kpi(
            "TOTAL PROFIT",
            money(profit),
            "Business profitability",
            "💰"
        )

    with c:
        card_kpi(
            "TOTAL ORDERS",
            f"{orders:,}",
            "Unique transactions",
            "🧾"
        )

    with d:
        card_kpi(
            "CUSTOMERS",
            f"{customers:,}",
            "Unique customers",
            "👥"
        )

    # VERTICAL GAP BETWEEN KPI ROWS
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ============================================================
    # BUSINESS CORE KPIs — ROW 2
    # ============================================================

    e, f, g, h = st.columns(4, gap="large")

    with e:
        card_kpi(
            "QUANTITY SOLD",
            f"{quantity:,.0f}",
            "Units sold",
            "📦"
        )

    with f:
        card_kpi(
            "PROFIT MARGIN",
            pct(margin),
            "Overall profitability",
            "📈"
        )

    with g:
        card_kpi(
            "AVERAGE ORDER VALUE",
            money(aov),
            "Revenue per order",
            "🛒"
        )

    with h:
        card_kpi(
            "PROFIT PER ORDER",
            money(ppo),
            "Profit per order",
            "💎"
        )

    # VERTICAL GAP BEFORE WORKFLOW
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    # ================================================================
    # END-TO-END ML WORKFLOW
    # ================================================================
    
    st.markdown(
        '<div class="section-label">END-TO-END ML WORKFLOW</div>',
        unsafe_allow_html=True
    )
    
    workflow = [
        ("01", "📂", "Data Loading",
         "Direct CSV connection and dataset validation."),
    
        ("02", "🔎", "Data Understanding",
         "Rows, columns, data types and target inspection."),
    
        ("03", "📊", "EDA",
         "Distributions, relationships and business patterns."),
    
        ("04", "⚙️", "Preprocessing",
         "Missing values and categorical encoding."),
    
        ("05", "🤖", "Model Training",
         "Multiple classification algorithms are trained."),
    
        ("06", "🏆", "Model Comparison",
         "Accuracy, Precision, Recall, F1 and ROC-AUC."),
    
        ("07", "🎯", "Prediction",
         "Generate Return_Flag predictions and probabilities."),
    
        ("08", "📋", "Evaluation",
         "Confusion matrix and detailed classification metrics."),
    
        ("09", "💡", "Explainability",
         "Identify the most influential predictive features."),
    
        ("10", "💼", "Decision Support",
         "Convert predictive results into business actions.")
    ]
    
    
    # ================================================================
    # WORKFLOW ROW 1
    # ================================================================
    
    row1 = st.columns(5, gap="large")
    
    for i in range(5):
    
        num, icon, title, description = workflow[i]
    
        with row1[i]:
    
            st.markdown(
                f"### {icon} {num}"
            )
    
            st.markdown(
                f"**{title}**"
            )
    
            st.caption(
                description
            )
    
        # Arrow between workflow steps
        if i < 4:
            pass
    
    
    # ================================================================
    # WORKFLOW CONNECTOR
    # ================================================================
    
    arrow_cols = st.columns(9, gap="small")
    
    for i in range(9):
    
        with arrow_cols[i]:
    
            if i % 2 == 0:
                st.markdown(
                    "<div style='text-align:center; font-size:24px;'>→</div>",
                    unsafe_allow_html=True
                )
    
    
    # ================================================================
    # WORKFLOW ROW 2
    # ================================================================
    
    row2 = st.columns(5, gap="large")
    
    for i in range(5, 10):
    
        num, icon, title, description = workflow[i]
    
        position = i - 5
    
        with row2[position]:
    
            st.markdown(
                f"### {icon} {num}"
            )
    
            st.markdown(
                f"**{title}**"
            )
    
            st.caption(
                description
            )
    
    
    # ================================================================
    # WORKFLOW → BUSINESS DECISION
    # ================================================================
    
    st.markdown("")
    
    st.info(
        "📊 Data → 🔎 Understanding → 📈 EDA → ⚙️ Preprocessing → "
        "🤖 Training → 🏆 Comparison → 🎯 Prediction → 📋 Evaluation → "
        "💡 Explainability → 💼 Decision Support"
    )
    
    
    # ================================================================
    # PREDICTION OBJECTIVE
    # ================================================================
    
    st.markdown(
        '<div class="section-label">PREDICTION OBJECTIVE</div>',
        unsafe_allow_html=True
    )
    
    with st.container(border=True):
    
        st.subheader("🎯 Return Prediction Intelligence")
    
        st.write(
            "This application predicts Return_Flag using the available "
            "feature-engineered SuperStore columns."
        )
    
        st.write(
            "The machine-learning workflow evaluates multiple "
            "classification algorithms rather than relying on a single model."
        )
    
        st.write(
            "F1 Score is used as the primary model-selection metric, "
            "while Accuracy, Precision, Recall and ROC-AUC are also "
            "reported for a more complete evaluation."
        )


# ================================================================
# BUSINESS DASHBOARD
# ================================================================

elif page == "Business Dashboard":

    st.title("📊 Business Intelligence Dashboard")

    st.caption(
        "Executive view of sales, profitability, customers, products "
        "and operational performance."
    )

    # ============================================================
    # EXECUTIVE KPI ROW
    # ============================================================

    k1, k2, k3, k4 = st.columns(4, gap="large")

    with k1:
        card_kpi(
            "SALES",
            money(sales),
            "Filtered revenue",
            "💵"
        )

    with k2:
        card_kpi(
            "PROFIT",
            money(profit),
            "Filtered profit",
            "💰"
        )

    with k3:
        card_kpi(
            "MARGIN",
            pct(margin),
            "Profit / Sales",
            "📈"
        )

    with k4:
        card_kpi(
            "AOV",
            money(aov),
            "Average order value",
            "🛒"
        )

    st.markdown("")


    # ============================================================
    # ADDITIONAL BUSINESS KPIs
    # ============================================================

    orders_dashboard = (
        filtered["Order_ID"].nunique()
        if "Order_ID" in filtered.columns
        else len(filtered)
    )

    customers_dashboard = (
        filtered["Customer_ID"].nunique()
        if "Customer_ID" in filtered.columns
        else 0
    )

    quantity_dashboard = (
        filtered["Quantity"].sum()
        if "Quantity" in filtered.columns
        else 0
    )

    returns_dashboard = (
        filtered["Return_Flag"].sum()
        if "Return_Flag" in filtered.columns
        and pd.api.types.is_numeric_dtype(filtered["Return_Flag"])
        else 0
    )

    r1, r2, r3, r4 = st.columns(4, gap="large")

    with r1:
        card_kpi(
            "ORDERS",
            f"{orders_dashboard:,}",
            "Unique transactions",
            "🧾"
        )

    with r2:
        card_kpi(
            "CUSTOMERS",
            f"{customers_dashboard:,}",
            "Unique customers",
            "👥"
        )

    with r3:
        card_kpi(
            "QUANTITY",
            f"{quantity_dashboard:,.0f}",
            "Units sold",
            "📦"
        )

    with r4:
        card_kpi(
            "PROFIT / ORDER",
            money(profit / orders_dashboard)
            if orders_dashboard else money(0),
            "Average profit per order",
            "💎"
        )


    # ============================================================
    # BUSINESS PERFORMANCE SNAPSHOT
    # ============================================================

    st.markdown(
        '<div class="section-label">BUSINESS PERFORMANCE SNAPSHOT</div>',
        unsafe_allow_html=True
    )

    snapshot_left, snapshot_right = st.columns(2, gap="large")

    with snapshot_left:

        st.info(
            f"💼 **Revenue Performance**\n\n"
            f"The current filtered dataset generated "
            f"**{money(sales)}** in sales across "
            f"**{orders_dashboard:,}** unique orders."
        )

        st.success(
            f"📈 **Profitability**\n\n"
            f"Total profit is **{money(profit)}**, "
            f"resulting in an overall margin of **{pct(margin)}**."
        )

    with snapshot_right:

        st.info(
            f"👥 **Customer Base**\n\n"
            f"The filtered dataset represents "
            f"**{customers_dashboard:,}** customers with an "
            f"average order value of **{money(aov)}**."
        )

        st.warning(
            f"📦 **Order Volume**\n\n"
            f"Customers generated approximately "
            f"**{quantity_dashboard:,.0f}** units across the "
            f"filtered transaction data."
        )


    # ============================================================
    # CATEGORY ANALYSIS
    # ============================================================

    st.markdown(
        '<div class="section-label">CATEGORY PERFORMANCE</div>',
        unsafe_allow_html=True
    )

    if "Category" in filtered.columns:

        cat = (
            filtered
            .groupby("Category", as_index=False)
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
        )

        if not cat.empty:

            cat["Margin"] = (
                cat["Profit"] / cat["Sales"] * 100
            ).replace([float("inf"), -float("inf")], 0).fillna(0)

            c1, c2 = st.columns(2, gap="large")

            with c1:

                fig = px.bar(
                    cat,
                    x="Category",
                    y=["Sales", "Profit"],
                    barmode="group",
                    title="Category Sales vs Profit",
                    labels={
                        "value": "Amount",
                        "variable": "Metric"
                    },
                    text_auto=".2s",
                    color_discrete_sequence=[
                        T["accent"],
                        T["accent2"]
                    ]
                )

                fig.update_traces(
                    textposition="outside"
                )

                st.plotly_chart(
                    layout(fig, 450),
                    use_container_width=True
                )

            with c2:

                fig = px.pie(
                    cat,
                    names="Category",
                    values="Sales",
                    hole=0.55,
                    title="Sales Mix by Category",
                    color_discrete_sequence=[
                        T["accent"],
                        T["accent2"],
                        T["accent"]
                    ]
                )

                fig.update_traces(
                    textposition="inside",
                    textinfo="percent+label"
                )

                st.plotly_chart(
                    layout(fig, 450),
                    use_container_width=True
                )

            # Category profitability

            fig = px.bar(
                cat.sort_values("Profit", ascending=False),
                x="Category",
                y="Profit",
                title="Category Profitability",
                text_auto=".2s",
                color="Profit",
                color_continuous_scale=[
                    T["accent2"],
                    T["accent"]
                ]
            )

            st.plotly_chart(
                layout(fig, 420),
                use_container_width=True
            )


    # ============================================================
    # REGIONAL ANALYSIS
    # ============================================================

    st.markdown(
        '<div class="section-label">REGIONAL PERFORMANCE</div>',
        unsafe_allow_html=True
    )

    if "Region" in filtered.columns:

        reg = (
            filtered
            .groupby("Region", as_index=False)
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
        )

        if not reg.empty:

            reg["Margin"] = (
                reg["Profit"] / reg["Sales"] * 100
            ).replace([float("inf"), -float("inf")], 0).fillna(0)

            a, b = st.columns(2, gap="large")

            with a:

                fig = px.bar(
                    reg.sort_values("Sales", ascending=False),
                    x="Region",
                    y=["Sales", "Profit"],
                    barmode="group",
                    title="Regional Sales vs Profit",
                    text_auto=".2s",
                    color_discrete_sequence=[
                        T["accent"],
                        T["accent2"]
                    ]
                )

                fig.update_traces(
                    textposition="outside"
                )

                st.plotly_chart(
                    layout(fig, 450),
                    use_container_width=True
                )

            with b:

                fig = px.bar(
                    reg.sort_values("Margin", ascending=False),
                    x="Region",
                    y="Margin",
                    title="Regional Profit Margin",
                    text_auto=".2f",
                    color="Margin",
                    color_continuous_scale=[
                        T["accent2"],
                        T["accent"]
                    ]
                )

                st.plotly_chart(
                    layout(fig, 450),
                    use_container_width=True
                )


    # ============================================================
    # CUSTOMER SEGMENT ANALYSIS
    # ============================================================

    st.markdown(
        '<div class="section-label">CUSTOMER SEGMENT INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    if "Segment" in filtered.columns:

        seg = (
            filtered
            .groupby("Segment", as_index=False)
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
        )

        if not seg.empty:

            seg["Margin"] = (
                seg["Profit"] / seg["Sales"] * 100
            ).replace([float("inf"), -float("inf")], 0).fillna(0)

            s1, s2 = st.columns(2, gap="large")

            with s1:

                fig = px.bar(
                    seg.sort_values("Sales", ascending=False),
                    x="Segment",
                    y="Sales",
                    title="Sales by Customer Segment",
                    text_auto=".2s",
                    color="Sales",
                    color_continuous_scale=[
                        T["accent2"],
                        T["accent"]
                    ]
                )

                st.plotly_chart(
                    layout(fig, 420),
                    use_container_width=True
                )

            with s2:

                fig = px.bar(
                    seg.sort_values("Profit", ascending=False),
                    x="Segment",
                    y="Profit",
                    title="Profit by Customer Segment",
                    text_auto=".2s",
                    color="Profit",
                    color_continuous_scale=[
                        T["accent2"],
                        T["accent"]
                    ]
                )

                st.plotly_chart(
                    layout(fig, 420),
                    use_container_width=True
                )


    # ============================================================
    # SHIPPING MODE ANALYSIS
    # ============================================================

    st.markdown(
        '<div class="section-label">OPERATIONAL PERFORMANCE</div>',
        unsafe_allow_html=True
    )

    if "Ship_Mode" in filtered.columns:

        ship = (
            filtered
            .groupby("Ship_Mode", as_index=False)
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
        )

        if not ship.empty:

            o1, o2 = st.columns(2, gap="large")

            with o1:

                fig = px.pie(
                    ship,
                    names="Ship_Mode",
                    values="Sales",
                    hole=0.55,
                    title="Sales by Ship Mode",
                    color_discrete_sequence=[
                        T["accent"],
                        T["accent2"],
                        T["accent"]
                    ]
                )

                st.plotly_chart(
                    layout(fig, 420),
                    use_container_width=True
                )

            with o2:

                fig = px.bar(
                    ship.sort_values("Profit", ascending=False),
                    x="Ship_Mode",
                    y="Profit",
                    title="Profit by Ship Mode",
                    text_auto=".2s",
                    color="Profit",
                    color_continuous_scale=[
                        T["accent2"],
                        T["accent"]
                    ]
                )

                st.plotly_chart(
                    layout(fig, 420),
                    use_container_width=True
                )


    # ============================================================
    # MONTHLY SALES TREND
    # ============================================================

    st.markdown(
        '<div class="section-label">SALES TREND ANALYSIS</div>',
        unsafe_allow_html=True
    )

    if "Order_Date" in filtered.columns:

        trend_df = filtered.copy()

        trend_df["Order_Date"] = pd.to_datetime(
            trend_df["Order_Date"],
            errors="coerce"
        )

        trend_df = trend_df.dropna(
            subset=["Order_Date"]
        )

        if not trend_df.empty:

            monthly = (
                trend_df
                .groupby(
                    trend_df["Order_Date"].dt.to_period("M")
                )
                .agg(
                    Sales=("Sales", "sum"),
                    Profit=("Profit", "sum")
                )
                .reset_index()
            )

            monthly["Order_Date"] = (
                monthly["Order_Date"]
                .dt
                .to_timestamp()
            )

            fig = px.line(
                monthly,
                x="Order_Date",
                y=["Sales", "Profit"],
                markers=True,
                title="Monthly Sales and Profit Trend",
                color_discrete_sequence=[
                    T["accent"],
                    T["accent2"]
                ]
            )

            st.plotly_chart(
                layout(fig, 480),
                use_container_width=True
            )


    # ============================================================
    # TOP PRODUCTS
    # ============================================================

    st.markdown(
        '<div class="section-label">PRODUCT PERFORMANCE</div>',
        unsafe_allow_html=True
    )

    if "Product_Name" in filtered.columns:

        product_df = (
            filtered
            .groupby("Product_Name", as_index=False)
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum"),
                Quantity=("Quantity", "sum")
            )
            .sort_values(
                "Sales",
                ascending=False
            )
            .head(10)
        )

        if not product_df.empty:

            fig = px.bar(
                product_df.sort_values(
                    "Sales",
                    ascending=True
                ),
                x="Sales",
                y="Product_Name",
                orientation="h",
                title="Top 10 Products by Sales",
                text_auto=".2s",
                color="Sales",
                color_continuous_scale=[
                    T["accent2"],
                    T["accent"]
                ]
            )

            st.plotly_chart(
                layout(fig, 520),
                use_container_width=True
            )


    # ============================================================
    # EXECUTIVE BUSINESS INTERPRETATION
    # ============================================================

    st.markdown(
        '<div class="section-label">EXECUTIVE INTERPRETATION</div>',
        unsafe_allow_html=True
    )

    if (
        "Category" in filtered.columns
        and "Region" in filtered.columns
    ):

        category_sales = (
            filtered.groupby("Category")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        region_sales = (
            filtered.groupby("Region")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        top_category = (
            category_sales.index[0]
            if not category_sales.empty
            else "N/A"
        )

        top_region = (
            region_sales.index[0]
            if not region_sales.empty
            else "N/A"
        )

        st.success(
            f"🏆 **Sales Leadership:** "
            f"{top_category} is currently the leading category, "
            f"while {top_region} is the strongest region by sales."
        )

    if margin > 0:

        st.info(
            f"📈 **Profitability:** "
            f"The current filtered business generates "
            f"{money(profit)} profit from {money(sales)} sales, "
            f"representing a {pct(margin)} margin."
        )

    elif margin < 0:

        st.error(
            f"⚠️ **Profitability Risk:** "
            f"The current filtered dataset is operating at a "
            f"negative profit margin of {pct(margin)}."
        )

    else:

        st.warning(
            "⚠️ **Profitability:** "
            "The current filtered dataset has no positive profit contribution."
        )

# ================================================================
# EDA & DATA QUALITY
# ================================================================

elif page == "EDA & Data Quality":

    st.title("🔎 EDA & Data Quality Intelligence")

    st.markdown(
        '<div class="section-label">DATASET HEALTH OVERVIEW</div>',
        unsafe_allow_html=True
    )

    # ============================================================
    # DATA QUALITY CALCULATIONS
    # ============================================================

    total_rows = len(filtered)
    total_columns = len(filtered.columns)

    total_cells = filtered.shape[0] * filtered.shape[1]

    missing_cells = int(
        filtered.isna().sum().sum()
    )

    missing_rate = (
        missing_cells / total_cells * 100
        if total_cells > 0
        else 0
    )

    duplicate_rows = int(
        filtered.duplicated().sum()
    )

    duplicate_rate = (
        duplicate_rows / total_rows * 100
        if total_rows > 0
        else 0
    )

    numeric_columns = filtered.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = filtered.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    datetime_columns = filtered.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    # ============================================================
    # QUALITY STATUS
    # ============================================================

    if missing_rate == 0 and duplicate_rows == 0:
        quality_status = "Excellent"
        quality_icon = "🟢"

    elif missing_rate <= 2 and duplicate_rate <= 1:
        quality_status = "Good"
        quality_icon = "🟢"

    elif missing_rate <= 10 and duplicate_rate <= 5:
        quality_status = "Moderate"
        quality_icon = "🟡"

    else:
        quality_status = "Needs Review"
        quality_icon = "🔴"

    # ============================================================
    # DATA QUALITY KPIs
    # ============================================================

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        card_kpi(
            "ROWS",
            f"{total_rows:,}",
            "Current observations",
            "📊"
        )

    with q2:
        card_kpi(
            "COLUMNS",
            f"{total_columns:,}",
            "Available columns",
            "🧩"
        )

    with q3:
        card_kpi(
            "MISSING CELLS",
            f"{missing_cells:,}",
            f"{missing_rate:.2f}% of all cells",
            "⚠️"
        )

    with q4:
        card_kpi(
            "DUPLICATE ROWS",
            f"{duplicate_rows:,}",
            f"{duplicate_rate:.2f}% of records",
            "🧹"
        )

    # ============================================================
    # DATASET HEALTH SUMMARY
    # ============================================================

    st.markdown(
        '<div class="section-label">DATASET HEALTH SUMMARY</div>',
        unsafe_allow_html=True
    )

    h1, h2, h3 = st.columns(3)

    with h1:
        st.markdown(
            f"""
            <div class="card">
                <h3>{quality_icon} Data Quality Status</h3>
                <p>
                    Current dataset quality is classified as
                    <b>{quality_status}</b>.
                </p>
                <p class="small-note">
                    Based on missing values and duplicate records.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with h2:
        st.markdown(
            f"""
            <div class="card">
                <h3>🔢 Column Composition</h3>
                <p>
                    <b>{len(numeric_columns)}</b> numeric columns,
                    <b>{len(categorical_columns)}</b> categorical columns
                    and <b>{len(datetime_columns)}</b> datetime columns
                    detected.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with h3:
        st.markdown(
            f"""
            <div class="card">
                <h3>🧹 Duplicate Profile</h3>
                <p>
                    <b>{duplicate_rows:,}</b> duplicate records were found,
                    representing <b>{duplicate_rate:.2f}%</b> of the
                    filtered dataset.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ============================================================
    # COLUMN-LEVEL DATA QUALITY
    # ============================================================

    st.markdown(
        '<div class="section-label">COLUMN-LEVEL DATA QUALITY</div>',
        unsafe_allow_html=True
    )

    quality = pd.DataFrame({
        "Column": filtered.columns,

        "Data Type": [
            str(filtered[col].dtype)
            for col in filtered.columns
        ],

        "Missing": [
            int(filtered[col].isna().sum())
            for col in filtered.columns
        ],

        "Missing %": [
            filtered[col].isna().mean() * 100
            for col in filtered.columns
        ],

        "Unique": [
            filtered[col].nunique(dropna=True)
            for col in filtered.columns
        ],

        "Unique %": [
            (
                filtered[col].nunique(dropna=True)
                / len(filtered) * 100
            )
            if len(filtered) > 0
            else 0
            for col in filtered.columns
        ]
    })

    quality = quality.sort_values(
        "Missing %",
        ascending=False
    )

    st.dataframe(
        quality.round(2),
        use_container_width=True,
        hide_index=True
    )

    # ============================================================
    # MISSING VALUE ANALYSIS
    # ============================================================

    st.markdown(
        '<div class="section-label">MISSING VALUE ANALYSIS</div>',
        unsafe_allow_html=True
    )

    missing_profile = pd.DataFrame({
        "Column": filtered.columns,

        "Missing": [
            int(filtered[col].isna().sum())
            for col in filtered.columns
        ],

        "Missing %": [
            filtered[col].isna().mean() * 100
            for col in filtered.columns
        ]
    })

    missing_profile = missing_profile[
        missing_profile["Missing"] > 0
    ].sort_values(
        "Missing",
        ascending=False
    )

    if missing_profile.empty:

        st.success(
            "✅ No missing values were detected in the current filtered dataset."
        )

    else:

        m1, m2 = st.columns(
            2,
            gap="large"
        )

        with m1:

            fig = px.bar(
                missing_profile.head(20),
                x="Missing",
                y="Column",
                orientation="h",
                text="Missing",
                title="Top Columns by Missing Values"
            )

            fig.update_traces(
                textposition="outside",
                marker_color=T["accent"]
            )

            fig.update_layout(
                height=450,
                paper_bgcolor=T["surface"],
                plot_bgcolor=T["plot_bg"],
                font=dict(
                    color=T["text"],
                    size=13
                ),
                title=dict(
                    font=dict(
                        color=T["text"],
                        size=18
                    )
                ),
                xaxis=dict(
                    title="Missing Values",
                    color=T["text"],
                    gridcolor=T["grid"],
                    zerolinecolor=T["grid"]
                ),
                yaxis=dict(
                    title="Column",
                    color=T["text"],
                    gridcolor=T["grid"]
                ),
                margin=dict(
                    l=20,
                    r=35,
                    t=65,
                    b=35
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="missing_values_chart"
            )

        with m2:

            fig = px.bar(
                missing_profile.head(20),
                x="Missing %",
                y="Column",
                orientation="h",
                text="Missing %",
                title="Missing Percentage by Column"
            )

            fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
                marker_color=T["accent2"]
            )

            fig.update_layout(
                height=450,
                paper_bgcolor=T["surface"],
                plot_bgcolor=T["plot_bg"],
                font=dict(
                    color=T["text"],
                    size=13
                ),
                title=dict(
                    font=dict(
                        color=T["text"],
                        size=18
                    )
                ),
                xaxis=dict(
                    title="Missing Percentage",
                    color=T["text"],
                    gridcolor=T["grid"],
                    zerolinecolor=T["grid"]
                ),
                yaxis=dict(
                    title="Column",
                    color=T["text"],
                    gridcolor=T["grid"]
                ),
                margin=dict(
                    l=20,
                    r=35,
                    t=65,
                    b=35
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="missing_percentage_chart"
            )

    # ============================================================
    # NUMERIC EXPLORATION
    # ============================================================

    st.markdown(
        '<div class="section-label">NUMERIC EXPLORATION</div>',
        unsafe_allow_html=True
    )

    numeric = filtered.select_dtypes(
        include=np.number
    )

    if not numeric.empty:

        n1, n2 = st.columns(
            2,
            gap="large"
        )

        with n1:

            num_col = st.selectbox(
                "Select numeric column",
                numeric.columns.tolist(),
                key="eda_numeric_column"
            )

            fig = px.histogram(
                filtered,
                x=num_col,
                nbins=40,
                marginal="box",
                title=f"Distribution: {num_col}"
            )

            fig.update_traces(
                marker_color=T["accent"]
            )

            fig.update_layout(
                height=450,
                paper_bgcolor=T["surface"],
                plot_bgcolor=T["plot_bg"],
                font=dict(
                    color=T["text"],
                    size=13
                ),
                title=dict(
                    font=dict(
                        color=T["text"],
                        size=18
                    )
                ),
                xaxis=dict(
                    title=num_col,
                    color=T["text"],
                    gridcolor=T["grid"],
                    zerolinecolor=T["grid"]
                ),
                yaxis=dict(
                    title="Count",
                    color=T["text"],
                    gridcolor=T["grid"],
                    zerolinecolor=T["grid"]
                ),
                margin=dict(
                    l=20,
                    r=25,
                    t=65,
                    b=35
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="numeric_distribution_chart"
            )

        with n2:

            corr = numeric.corr(
                numeric_only=True
            )

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title="Numeric Correlation Matrix",
                color_continuous_scale=[
                    T["surface2"],
                    T["accent"],
                    T["accent2"]
                ]
            )

            fig.update_layout(
                height=450,
                paper_bgcolor=T["surface"],
                plot_bgcolor=T["plot_bg"],
                font=dict(
                    color=T["text"],
                    size=12
                ),
                title=dict(
                    font=dict(
                        color=T["text"],
                        size=18
                    )
                ),
                margin=dict(
                    l=20,
                    r=20,
                    t=65,
                    b=35
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="correlation_matrix_chart"
            )

    else:

        st.info(
            "No numeric columns are available for numeric EDA."
        )

    # ============================================================
    # DESCRIPTIVE STATISTICS
    # ============================================================

    st.markdown(
        '<div class="section-label">DESCRIPTIVE STATISTICS</div>',
        unsafe_allow_html=True
    )

    if not numeric.empty:

        descriptive = numeric.describe().T

        descriptive["Missing"] = [
            int(filtered[col].isna().sum())
            for col in descriptive.index
        ]

        descriptive["Unique"] = [
            int(filtered[col].nunique(dropna=True))
            for col in descriptive.index
        ]

        descriptive = descriptive.reset_index()

        descriptive = descriptive.rename(
            columns={
                "index": "Column"
            }
        )

        st.dataframe(
            descriptive.round(3),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Descriptive statistics are unavailable because no numeric columns were detected."
        )

    # ============================================================
    # CATEGORICAL DATA ANALYSIS
    # ============================================================

    st.markdown(
        '<div class="section-label">CATEGORICAL DATA EXPLORATION</div>',
        unsafe_allow_html=True
    )

    if categorical_columns:

        c1, c2 = st.columns(
            2,
            gap="large"
        )

        with c1:

            cat_col = st.selectbox(
                "Select categorical column",
                categorical_columns,
                key="eda_category_column"
            )

        with c2:

            top_n = st.slider(
                "Number of categories",
                min_value=5,
                max_value=20,
                value=10,
                key="eda_top_categories"
            )

        category_counts = (
            filtered[cat_col]
            .astype(str)
            .value_counts()
            .head(top_n)
            .reset_index()
        )

        category_counts.columns = [
            cat_col,
            "Count"
        ]

        fig = px.bar(
            category_counts,
            x=cat_col,
            y="Count",
            text="Count",
            title=f"Top {top_n} Values: {cat_col}"
        )

        fig.update_traces(
            textposition="outside",
            marker_color=T["accent"]
        )

        fig.update_layout(
            height=450,
            paper_bgcolor=T["surface"],
            plot_bgcolor=T["plot_bg"],
            font=dict(
                color=T["text"],
                size=13
            ),
            title=dict(
                font=dict(
                    color=T["text"],
                    size=18
                )
            ),
            xaxis=dict(
                title=cat_col,
                color=T["text"],
                gridcolor=T["grid"]
            ),
            yaxis=dict(
                title="Count",
                color=T["text"],
                gridcolor=T["grid"]
            ),
            margin=dict(
                l=20,
                r=25,
                t=65,
                b=50
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="categorical_distribution_chart"
        )

    else:

        st.info(
            "No categorical columns were detected."
        )

    # ============================================================
    # OUTLIER ANALYSIS
    # ============================================================

    st.markdown(
        '<div class="section-label">OUTLIER ANALYSIS</div>',
        unsafe_allow_html=True
    )

    if not numeric.empty:

        outlier_rows = []

        for col in numeric.columns:

            series = filtered[col].dropna()

            if len(series) < 4:
                continue

            q1_value = series.quantile(0.25)
            q3_value = series.quantile(0.75)

            iqr = q3_value - q1_value

            lower_bound = q1_value - 1.5 * iqr
            upper_bound = q3_value + 1.5 * iqr

            outliers = series[
                (series < lower_bound) |
                (series > upper_bound)
            ]

            outlier_count = len(outliers)

            outlier_percentage = (
                outlier_count / len(series) * 100
                if len(series) > 0
                else 0
            )

            outlier_rows.append({
                "Column": col,
                "Outliers": outlier_count,
                "Outlier %": outlier_percentage
            })

        outlier_df = pd.DataFrame(
            outlier_rows
        ).sort_values(
            "Outliers",
            ascending=False
        )

        if not outlier_df.empty:

            st.dataframe(
                outlier_df.round(2),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No suitable numeric columns were available for outlier analysis."
            )

    else:

        st.info(
            "No numeric columns are available for outlier detection."
        )

    # ============================================================
    # DATA QUALITY CONCLUSION
    # ============================================================

    st.markdown(
        '<div class="section-label">DATA QUALITY CONCLUSION</div>',
        unsafe_allow_html=True
    )

    if quality_status == "Excellent":

        st.success(
            "✅ The current filtered dataset has excellent data quality. "
            "No meaningful missing-value or duplicate-row issues were detected."
        )

    elif quality_status == "Good":

        st.success(
            "🟢 The dataset is generally healthy. "
            "Only limited missing values or duplicate records require attention."
        )

    elif quality_status == "Moderate":

        st.warning(
            "🟡 The dataset has moderate quality issues. "
            "Review missing values, duplicates and potential outliers before "
            "using the data for sensitive modelling decisions."
        )

    else:

        st.error(
            "🔴 The dataset requires data-quality review before modelling. "
            "Significant missing values or duplicate records are present."
        )

# ================================================================
# ML MODEL LAB
# ================================================================

elif page == "ML Model Lab":

    st.title("🧠 Machine Learning Model Lab")

    st.markdown(
        '<div class="section-label">MODEL INTELLIGENCE CENTER</div>',
        unsafe_allow_html=True
    )

    # ============================================================
    # CURRENT MODEL STATUS
    # ============================================================

    existing_bundle = st.session_state.get(
        "model_results"
    )

    if existing_bundle:

        best_name = existing_bundle.get(
            "best_name",
            "Unknown"
        )

        results_existing = existing_bundle.get(
            "results"
        )

        best_f1 = None
        best_accuracy = None

        if (
            results_existing is not None
            and not results_existing.empty
        ):

            best_rows = results_existing[
                results_existing["Model"] == best_name
            ]

            if not best_rows.empty:

                row = best_rows.iloc[0]

                best_f1 = row.get(
                    "F1",
                    np.nan
                )

                best_accuracy = row.get(
                    "Accuracy",
                    np.nan
                )

        st.success(
            f"✅ Trained model available: **{best_name}**"
        )

        status_cols = st.columns(4)

        with status_cols[0]:
            card_kpi(
                "MODEL STATUS",
                "READY",
                "Persistent model loaded",
                "✅"
            )

        with status_cols[1]:
            card_kpi(
                "BEST MODEL",
                best_name,
                "Selected using F1",
                "🏆"
            )

        with status_cols[2]:
            card_kpi(
                "F1 SCORE",
                f"{best_f1:.3f}"
                if pd.notna(best_f1)
                else "N/A",
                "Primary metric",
                "📈"
            )

        with status_cols[3]:
            card_kpi(
                "ACCURACY",
                f"{best_accuracy:.3f}"
                if pd.notna(best_accuracy)
                else "N/A",
                "Test-set accuracy",
                "🎯"
            )

    else:

        st.info(
            "ℹ️ No persistent trained model was found. "
            "Train the models once below."
        )

    # ============================================================
    # FEATURE CONFIGURATION
    # ============================================================

    st.markdown(
        '<div class="section-label">FEATURE CONFIGURATION</div>',
        unsafe_allow_html=True
    )

    candidate_features = feature_columns(
        filtered
    )

    selected_features = st.multiselect(
        "Predictor columns",
        candidate_features,
        default=candidate_features,
        key="ml_selected_features"
    )

    test_size = st.slider(
        "Test set size",
        min_value=0.15,
        max_value=0.40,
        value=0.20,
        step=0.05,
        key="ml_test_size"
    )

    st.info(
        "🏆 F1 is the primary model-selection metric. "
        "Accuracy, Precision, Recall and ROC-AUC are also evaluated. "
        "The highest Accuracy model is not automatically selected."
    )

    # ============================================================
    # TRAIN / RETRAIN
    # ============================================================

    train_button = st.button(
        "🚀 Train / Retrain Classification Models",
        type="primary",
        use_container_width=True
    )

    if train_button:

        if not selected_features:

            st.error(
                "❌ Select at least one predictor column."
            )

        else:

            with st.spinner(
                "Training classification models..."
            ):

                try:

                    bundle = train_models(
                        filtered,
                        TARGET_COLUMN,
                        selected_features,
                        test_size
                    )

                    # ------------------------------------------------
                    # STORE IN SESSION
                    # ------------------------------------------------

                    st.session_state.model_results = bundle
                    st.session_state.trained_bundle = bundle

                    # ------------------------------------------------
                    # SAVE TO DISK
                    # ------------------------------------------------

                    saved, save_error = save_model_bundle(
                        bundle
                    )

                    if saved:

                        st.success(
                            f"✅ Training completed successfully. "
                            f"Best model by F1: **{bundle['best_name']}**"
                        )

                        st.info(
                            "💾 Model bundle saved permanently. "
                            "You do not need to retrain every time "
                            "you open or refresh the Streamlit app."
                        )

                    else:

                        st.warning(
                            "⚠️ Training completed, but the model "
                            f"could not be saved: {save_error}"
                        )

                except Exception as exc:

                    st.error(
                        f"❌ Training failed: {exc}"
                    )

    # ============================================================
    # ACTIVE TRAINED MODEL
    # ============================================================

    bundle = st.session_state.get(
        "model_results"
    )

    if bundle is None:

        st.warning(
            "⚠️ No trained model is available yet."
        )

        st.caption(
            "Click 'Train / Retrain Classification Models' "
            "once to create the persistent model."
        )

    else:

        # ============================================================
        # MODEL COMPARISON
        # ============================================================

        st.markdown(
            '<div class="section-label">MODEL COMPARISON</div>',
            unsafe_allow_html=True
        )

        results = bundle[
            "results"
        ].copy()

        display_cols = [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ]

        available_cols = [
            col
            for col in display_cols
            if col in results.columns
        ]

        comparison = results[
            available_cols
        ].copy()

        numeric_cols = comparison.select_dtypes(
            include=np.number
        ).columns

        if len(numeric_cols) > 0:

            comparison[numeric_cols] = (
                comparison[numeric_cols].round(4)
            )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        # ============================================================
        # PERFORMANCE VISUALIZATION
        # ============================================================

        st.markdown(
            '<div class="section-label">MODEL PERFORMANCE</div>',
            unsafe_allow_html=True
        )

        metric_cols = [
            col
            for col in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC"
            ]
            if col in results.columns
        ]

        if metric_cols:

            fig = px.bar(
                results,
                x="Model",
                y=metric_cols,
                barmode="group",
                title="Classification Model Performance",
                text_auto=".3f"
            )

            fig.update_layout(
                height=520,
                xaxis_title="Classification Model",
                yaxis_title="Score",
                yaxis=dict(
                    range=[0, 1]
                ),
                margin=dict(
                    l=40,
                    r=40,
                    t=70,
                    b=40
                )
            )

            fig.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                layout(fig, 520),
                use_container_width=True
            )

        # ============================================================
        # TRAINING SUMMARY
        # ============================================================

        st.markdown(
            '<div class="section-label">TRAINING SUMMARY</div>',
            unsafe_allow_html=True
        )

        s1, s2, s3 = st.columns(3)

        with s1:

            st.metric(
                "Target",
                TARGET_COLUMN
            )

        with s2:

            st.metric(
                "Features Used",
                len(
                    bundle.get(
                        "feature_cols",
                        []
                    )
                )
            )

        with s3:

            st.metric(
                "Test Size",
                f"{test_size:.0%}"
            )

        st.success(
            f"🏆 **{bundle['best_name']}** is currently "
            f"selected as the best model using **F1 Score**."
        )

        # ============================================================
        # MODEL FILE STATUS
        # ============================================================

        st.markdown(
            '<div class="section-label">MODEL STORAGE</div>',
            unsafe_allow_html=True
        )

        storage_cols = st.columns(2)

        with storage_cols[0]:

            card_kpi(
                "MODEL FILE",
                "SAVED"
                if MODEL_BUNDLE_PATH.exists()
                else "NOT SAVED",
                str(MODEL_BUNDLE_PATH),
                "💾"
            )

        with storage_cols[1]:

            card_kpi(
                "PREDICTION READY",
                "YES",
                "Prediction Studio can use this model",
                "🎯"
            )

# ================================================================
# PREDICTION STUDIO
# ================================================================

elif page == "Prediction Studio":

    st.title("🎯 Prediction Studio")

    st.markdown(
        '<div class="section-label">MODEL STATUS</div>',
        unsafe_allow_html=True
    )

    # ============================================================
    # GET MODEL FROM SESSION
    # ============================================================

    bundle = st.session_state.get(
        "model_results"
    )

    # ============================================================
    # FALLBACK: LOAD FROM DISK
    # ============================================================

    if bundle is None:

        bundle = load_saved_model_bundle()

        if bundle is not None:

            st.session_state.model_results = bundle
            st.session_state.trained_bundle = bundle

    # ============================================================
    # NO MODEL
    # ============================================================

    if bundle is None:

        st.warning(
            "⚠️ No trained model is currently available."
        )

        st.info(
            "Go to **ML Model Lab → Train / Retrain Classification Models** "
            "once. After training, the model will be saved and reused "
            "automatically."
        )

        st.stop()

    # ============================================================
    # MODEL INFORMATION
    # ============================================================

    model = bundle[
        "best_model"
    ]

    features = bundle[
        "feature_cols"
    ]

    best_model_name = bundle.get(
        "best_name",
        "Selected Model"
    )

    results = bundle.get(
        "results"
    )

    # ============================================================
    # BEST MODEL METRICS
    # ============================================================

    best_f1 = np.nan
    best_accuracy = np.nan
    best_precision = np.nan
    best_recall = np.nan
    best_auc = np.nan

    if (
        results is not None
        and not results.empty
    ):

        best_rows = results[
            results["Model"] == best_model_name
        ]

        if not best_rows.empty:

            row = best_rows.iloc[0]

            best_accuracy = row.get(
                "Accuracy",
                np.nan
            )

            best_precision = row.get(
                "Precision",
                np.nan
            )

            best_recall = row.get(
                "Recall",
                np.nan
            )

            best_f1 = row.get(
                "F1",
                np.nan
            )

            best_auc = row.get(
                "ROC-AUC",
                np.nan
            )

    # ============================================================
    # MODEL KPI CARDS
    # ============================================================

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        card_kpi(
            "ACTIVE MODEL",
            best_model_name,
            "F1-selected model",
            "🏆"
        )

    with m2:

        card_kpi(
            "F1 SCORE",
            f"{best_f1:.3f}"
            if pd.notna(best_f1)
            else "N/A",
            "Primary selection metric",
            "📈"
        )

    with m3:

        card_kpi(
            "ROC-AUC",
            f"{best_auc:.3f}"
            if pd.notna(best_auc)
            else "N/A",
            "Class discrimination",
            "📊"
        )

    with m4:

        card_kpi(
            "FEATURES",
            f"{len(features):,}",
            "Model input columns",
            "🧩"
        )

    # ============================================================
    # NEW OBSERVATION
    # ============================================================

    st.markdown(
        '<div class="section-label">NEW OBSERVATION</div>',
        unsafe_allow_html=True
    )

    input_data = {}

    input_cols = st.columns(3)

    for i, feature in enumerate(
        features
    ):

        current_col = input_cols[
            i % 3
        ]

        with current_col:

            if feature not in filtered.columns:

                st.warning(
                    f"{feature} is not available."
                )

                continue

            series = filtered[
                feature
            ]

            # ----------------------------------------------------
            # NUMERIC INPUT
            # ----------------------------------------------------

            if pd.api.types.is_numeric_dtype(
                series
            ):

                valid_values = series.dropna()

                if not valid_values.empty:

                    default_value = float(
                        valid_values.median()
                    )

                else:

                    default_value = 0.0

                input_data[
                    feature
                ] = st.number_input(
                    feature,
                    value=default_value,
                    key=f"prediction_input_{feature}"
                )

            # ----------------------------------------------------
            # CATEGORICAL INPUT
            # ----------------------------------------------------

            else:

                options = safe_unique(
                    series
                )

                options = [
                    x
                    for x in options
                    if pd.notna(x)
                ]

                if options:

                    input_data[
                        feature
                    ] = st.selectbox(
                        feature,
                        options,
                        key=f"prediction_input_{feature}"
                    )

                else:

                    input_data[
                        feature
                    ] = "Missing"

    # ============================================================
    # PREDICTION BUTTON
    # ============================================================

    st.markdown("")

    predict_button = st.button(
        "🤖 Predict Return Risk",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        try:

            # ----------------------------------------------------
            # CREATE OBSERVATION
            # ----------------------------------------------------

            new_df = pd.DataFrame(
                [input_data]
            )

            # ----------------------------------------------------
            # SAME FEATURE PREPARATION AS TRAINING
            # ----------------------------------------------------

            new_df = prepare_features(
                new_df,
                features
            )

            # ----------------------------------------------------
            # PREDICTION
            # ----------------------------------------------------

            prediction = model.predict(
                new_df
            )[0]

            # ----------------------------------------------------
            # PROBABILITY
            # ----------------------------------------------------

            probability = None

            if hasattr(
                model,
                "predict_proba"
            ):

                try:

                    probabilities = (
                        model
                        .predict_proba(new_df)[0]
                    )

                    classes_model = (
                        model.classes_
                        if hasattr(
                            model,
                            "classes_"
                        )
                        else None
                    )

                    if (
                        classes_model is not None
                        and 1 in classes_model
                    ):

                        positive_index = list(
                            classes_model
                        ).index(1)

                        probability = float(
                            probabilities[
                                positive_index
                            ]
                        )

                    else:

                        probability = float(
                            probabilities[-1]
                        )

                except Exception:

                    probability = None

            # ====================================================
            # RESULT
            # ====================================================

            st.markdown(
                '<div class="section-label">PREDICTION RESULT</div>',
                unsafe_allow_html=True
            )

            probability_percent = (
                probability * 100
                if probability is not None
                else None
            )

            if probability is not None:

                if probability >= 0.70:

                    risk_label = "HIGH"
                    business_action = "Review"

                elif probability >= 0.40:

                    risk_label = "MEDIUM"
                    business_action = "Monitor"

                else:

                    risk_label = "LOW"
                    business_action = "Normal"

            else:

                risk_label = "N/A"
                business_action = "Review"

            r1, r2, r3, r4 = st.columns(4)

            with r1:

                card_kpi(
                    "PREDICTED CLASS",
                    str(prediction),
                    "Return_Flag prediction",
                    "🎯"
                )

            with r2:

                card_kpi(
                    "RETURN PROBABILITY",
                    f"{probability_percent:.2f}%"
                    if probability_percent is not None
                    else "N/A",
                    "Estimated probability",
                    "📊"
                )

            with r3:

                card_kpi(
                    "RISK LEVEL",
                    risk_label,
                    "Prediction risk band",
                    "⚠️"
                )

            with r4:

                card_kpi(
                    "BUSINESS ACTION",
                    business_action,
                    "Decision-support guidance",
                    "💼"
                )

            # ====================================================
            # VISUAL PROBABILITY
            # ====================================================

            st.markdown(
                '<div class="section-label">PREDICTION VISUALIZATION</div>',
                unsafe_allow_html=True
            )

            if probability is not None:

                visual_left, visual_right = st.columns(2)

                with visual_left:

                    probability_df = pd.DataFrame({
                        "Outcome": [
                            "Return Risk",
                            "Lower Return Risk"
                        ],
                        "Probability": [
                            probability,
                            1 - probability
                        ]
                    })

                    fig = px.bar(
                        probability_df,
                        x="Outcome",
                        y="Probability",
                        text="Probability",
                        title="Prediction Probability"
                    )

                    fig.update_yaxes(
                        range=[0, 1],
                        tickformat=".0%"
                    )

                    fig.update_traces(
                        texttemplate="%{text:.1%}",
                        textposition="outside"
                    )

                    st.plotly_chart(
                        layout(fig, 430),
                        use_container_width=True
                    )

                with visual_right:

                    gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=probability_percent,
                            number={
                                "suffix": "%"
                            },
                            title={
                                "text": "Return Risk Probability"
                            },
                            gauge={
                                "axis": {
                                    "range": [
                                        0,
                                        100
                                    ]
                                }
                            }
                        )
                    )

                    gauge.update_layout(
                        height=430,
                        margin=dict(
                            l=30,
                            r=30,
                            t=70,
                            b=30
                        )
                    )

                    st.plotly_chart(
                        gauge,
                        use_container_width=True
                    )

            # ====================================================
            # BUSINESS INTERPRETATION
            # ====================================================

            st.markdown(
                '<div class="section-label">BUSINESS INTERPRETATION</div>',
                unsafe_allow_html=True
            )

            if probability is not None:

                if probability >= 0.70:

                    st.error(
                        "🔴 **High predicted return risk.** "
                        "This observation should receive additional "
                        "business review before operational action."
                    )

                elif probability >= 0.40:

                    st.warning(
                        "🟡 **Moderate predicted return risk.** "
                        "Monitor the transaction and consider relevant "
                        "customer, product and operational factors."
                    )

                else:

                    st.success(
                        "🟢 **Lower predicted return risk.** "
                        "The model estimates a relatively lower probability "
                        "of the positive return class."
                    )

                st.caption(
                    "The probability is a model estimate, not a guarantee. "
                    "Use the prediction as decision support rather than an "
                    "automatic business rule."
                )

        except Exception as exc:

            st.error(
                f"❌ Prediction failed: {exc}"
            )

# ================================================================
# MODEL EVALUATION
# ================================================================

elif page == "Model Evaluation":

    st.title("📐 Model Evaluation")

    # ============================================================
    # MODEL AVAILABILITY
    # ============================================================

    bundle = st.session_state.model_results

    if not bundle:
        st.warning(
            "⚠️ No trained model is currently available. "
            "Train the classification models from ML Model Lab first."
        )
        st.stop()

    # ============================================================
    # SELECTED MODEL
    # ============================================================

    best_name = bundle["best_name"]
    results = bundle["results"].copy()

    best_rows = results[
        results["Model"] == best_name
    ]

    if best_rows.empty:
        st.error(
            "Selected model information could not be found "
            "in the model comparison results."
        )
        st.stop()

    best_row = best_rows.iloc[0]

    st.markdown(
        '<div class="section-label">MODEL INTELLIGENCE CENTER</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="card">
            <h2>🏆 {best_name}</h2>
            <p>
                Currently selected as the best-performing classification model
                using <b>F1 Score</b> as the primary selection metric.
            </p>
            <p class="small-note">
                Accuracy, Precision, Recall and ROC-AUC are also evaluated
                to provide a balanced view of model performance.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # MODEL PERFORMANCE KPIs
    # ============================================================

    st.markdown(
        '<div class="section-label">MODEL PERFORMANCE SCORECARD</div>',
        unsafe_allow_html=True
    )

    accuracy = best_row["Accuracy"]
    precision = best_row["Precision"]
    recall = best_row["Recall"]
    f1_score = best_row["F1"]
    roc_auc = best_row["ROC-AUC"]

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        card_kpi(
            "ACCURACY",
            pct(accuracy * 100),
            "Overall prediction correctness",
            "🎯"
        )

    with m2:
        card_kpi(
            "PRECISION",
            pct(precision * 100),
            "Positive prediction quality",
            "🔎"
        )

    with m3:
        card_kpi(
            "RECALL",
            pct(recall * 100),
            "Positive-class detection",
            "📡"
        )

    with m4:
        card_kpi(
            "F1 SCORE",
            pct(f1_score * 100),
            "Primary selection metric",
            "🏆"
        )

    with m5:
        card_kpi(
            "ROC-AUC",
            pct(roc_auc * 100)
            if pd.notna(roc_auc)
            else "N/A",
            "Class discrimination",
            "📈"
        )

    # ============================================================
    # PERFORMANCE INTERPRETATION
    # ============================================================

    st.markdown(
        '<div class="section-label">PERFORMANCE INTERPRETATION</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown(
            f"""
            <div class="card">
                <h3>🎯 Accuracy</h3>
                <p>
                    The model correctly classifies approximately
                    <b>{accuracy * 100:.2f}%</b> of test observations.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with p2:
        st.markdown(
            f"""
            <div class="card">
                <h3>📡 Recall</h3>
                <p>
                    The model identifies approximately
                    <b>{recall * 100:.2f}%</b> of the positive-class
                    observations.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with p3:
        st.markdown(
            f"""
            <div class="card">
                <h3>🏆 F1 Selection</h3>
                <p>
                    The selected model achieved an F1 score of
                    <b>{f1_score * 100:.2f}%</b>.
                    F1 balances Precision and Recall.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ============================================================
    # CONFUSION MATRIX
    # ============================================================

    st.markdown(
        '<div class="section-label">CONFUSION MATRIX</div>',
        unsafe_allow_html=True
    )

    confusion_fig = plot_confusion(bundle)

    confusion_fig = layout(
        confusion_fig,
        500
    )

    st.plotly_chart(
        confusion_fig,
        use_container_width=True
    )

    # ============================================================
    # CLASSIFICATION REPORT
    # ============================================================

    st.markdown(
        '<div class="section-label">CLASSIFICATION REPORT</div>',
        unsafe_allow_html=True
    )

    pred = bundle["best_model"].predict(
        bundle["X_test"]
    )

    report = classification_report(
        bundle["y_test"],
        pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    # Keep numerical metrics clean.
    numeric_report_cols = [
        col
        for col in [
            "precision",
            "recall",
            "f1-score",
            "support"
        ]
        if col in report_df.columns
    ]

    if numeric_report_cols:
        report_df[numeric_report_cols] = (
            report_df[numeric_report_cols].round(4)
        )

    st.dataframe(
        report_df,
        use_container_width=True,
        hide_index=False
    )

    # ============================================================
    # MODEL COMPARISON
    # ============================================================

    st.markdown(
        '<div class="section-label">ALL MODEL PERFORMANCE</div>',
        unsafe_allow_html=True
    )

    display_cols = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]

    available_cols = [
        col
        for col in display_cols
        if col in results.columns
    ]

    comparison_df = results[available_cols].copy()

    metric_cols = [
        col
        for col in available_cols
        if col != "Model"
    ]

    comparison_df[metric_cols] = (
        comparison_df[metric_cols].round(4)
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    # ============================================================
    # METRIC COMPARISON CHART
    # ============================================================

    st.markdown(
        '<div class="section-label">MODEL METRIC COMPARISON</div>',
        unsafe_allow_html=True
    )

    chart_metrics = [
        col
        for col in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ]
        if col in results.columns
    ]

    if chart_metrics:

        comparison_long = results.melt(
            id_vars="Model",
            value_vars=chart_metrics,
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            comparison_long,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            text="Score",
            title="Classification Model Performance Comparison",
            labels={
                "Score": "Score",
                "Model": "Model"
            }
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            cliponaxis=False
        )

        fig.update_yaxes(
            range=[
                0,
                min(
                    1.05,
                    max(
                        1,
                        float(
                            comparison_long["Score"].max()
                        ) + 0.10
                    )
                )
            ]
        )

        st.plotly_chart(
            layout(fig, 520),
            use_container_width=True
        )

    # ============================================================
    # EVALUATION CONCLUSION
    # ============================================================

    st.markdown(
        '<div class="section-label">EVALUATION CONCLUSION</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="card">
            <h3>🏆 Selected Model: {best_name}</h3>
            <p>
                <b>{best_name}</b> is currently selected because it achieved
                the strongest <b>F1 Score</b> among the evaluated models.
            </p>
            <p class="small-note">
                The evaluation should not be interpreted from Accuracy alone.
                Precision, Recall, F1 and ROC-AUC are reviewed together to
                understand classification quality and positive-class detection.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ================================================================
# FEATURE INTELLIGENCE
# ================================================================

elif page == "Feature Intelligence":

    st.title("🔬 Feature Intelligence")

    # ============================================================
    # LOAD EXISTING TRAINED MODEL
    # ============================================================

    bundle = get_active_model_bundle()

    if bundle is None:
        st.warning(
            "⚠️ No trained model is currently available."
        )

        st.info(
            "The existing trained model could not be loaded. "
            "Train the model once from ML Model Lab."
        )

        st.stop()

    best_model = bundle["best_model"]
    best_name = bundle["best_name"]

    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================
    # Use already-saved permutation importance whenever available.
    # This prevents expensive recalculation on every page rerun.
    # ============================================================

    if (
        "feature_importance" in bundle
        and isinstance(
            bundle["feature_importance"],
            pd.DataFrame
        )
        and not bundle["feature_importance"].empty
    ):

        imp = bundle["feature_importance"].copy()

    else:

        # --------------------------------------------------------
        # Calculate only when the existing joblib does not
        # already contain permutation importance.
        # --------------------------------------------------------

        try:

            imp = get_feature_importance(bundle)

        except Exception as e:

            st.error(
                f"Feature importance could not be calculated: {e}"
            )

            st.stop()

        if imp is None or imp.empty:

            st.error(
                "Feature importance could not be calculated "
                "from the existing trained model."
            )

            st.stop()

        # --------------------------------------------------------
        # Save calculated importance inside current bundle.
        # This avoids recalculation during future page loads.
        # --------------------------------------------------------

        bundle["feature_importance"] = imp.copy()

        try:
            joblib.dump(
                bundle,
                MODEL_PATH
            )

            st.session_state.model_results = bundle
            st.session_state.trained_bundle = bundle

        except Exception:
            # Model itself remains usable even if saving the
            # additional feature-importance cache fails.
            pass

    # ============================================================
    # CLEAN FEATURE IMPORTANCE DATA
    # ============================================================

    imp = imp.copy()

    imp["Importance"] = pd.to_numeric(
        imp["Importance"],
        errors="coerce"
    )

    imp = imp.dropna(
        subset=["Importance"]
    )

    imp = imp.sort_values(
        "Importance",
        ascending=False
    ).reset_index(
        drop=True
    )

    if imp.empty:

        st.error(
            "No valid feature importance values are available."
        )

        st.stop()

    positive_imp = imp[
        imp["Importance"] > 0
    ]

    top_feature = str(
        imp.iloc[0]["Feature"]
    )

    top_importance = float(
        imp.iloc[0]["Importance"]
    )

    zero_features = int(
        (imp["Importance"] <= 0).sum()
    )

    # ============================================================
    # MODEL F1
    # ============================================================

    model_f1 = None

    if "results" in bundle:

        results = bundle["results"]

        best_rows = results[
            results["Model"] == best_name
        ]

        if not best_rows.empty:

            model_f1 = best_rows.iloc[0]["F1"]

    # ============================================================
    # PREDICTIVE FEATURE ANALYSIS
    # ============================================================

    st.markdown(
        '<div class="section-label">PREDICTIVE FEATURE ANALYSIS</div>',
        unsafe_allow_html=True
    )

    # ============================================================
    # FEATURE KPI CARDS
    # ============================================================

    f1, f2, f3, f4 = st.columns(4)

    with f1:

        card_kpi(
            "BEST MODEL",
            best_name,
            "Existing trained model",
            "🏆"
        )

    with f2:

        card_kpi(
            "FEATURES ANALYZED",
            f"{len(imp):,}",
            "Permutation-tested features",
            "🧩"
        )

    with f3:

        card_kpi(
            "TOP FEATURE",
            str(top_feature),
            "Highest importance",
            "⭐"
        )

    with f4:

        card_kpi(
            "MODEL F1",
            pct(float(model_f1) * 100)
            if model_f1 is not None
            else "N/A",
            "Held-out test performance",
            "🎯"
        )

    # ============================================================
    # TOP FEATURE INSIGHT
    # ============================================================

    st.markdown(
        '<div class="section-label">TOP FEATURE INSIGHT</div>',
        unsafe_allow_html=True
    )

    i1, i2, i3 = st.columns(3)

    with i1:

        card_kpi(
            "TOP FEATURE",
            str(top_feature),
            "Most influential predictor",
            "🥇"
        )

    with i2:

        card_kpi(
            "IMPORTANCE",
            f"{top_importance:.5f}",
            "Permutation importance",
            "📊"
        )

    with i3:

        card_kpi(
            "ZERO IMPORTANCE",
            f"{zero_features:,}",
            "Features with no measured impact",
            "⚪"
        )

    # ============================================================
    # TOP PREDICTIVE FEATURES
    # ============================================================

    st.markdown(
        '<div class="section-label">TOP PREDICTIVE FEATURES</div>',
        unsafe_allow_html=True
    )

    top_n = min(
        15,
        len(imp)
    )

    top_imp = (
        imp.head(top_n)
        .sort_values(
            "Importance",
            ascending=True
        )
    )

    fig = px.bar(
        top_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        title=f"Top {top_n} Predictive Features — {best_name}"
    )

    fig.update_traces(
        texttemplate="%{x:.4f}",
        textposition="outside"
    )

    fig.update_layout(
        height=520,
        margin=dict(
            l=20,
            r=80,
            t=70,
            b=30
        )
    )

    st.plotly_chart(
        layout(fig, 520),
        use_container_width=True
    )

    # ============================================================
    # FEATURE IMPORTANCE STRUCTURE
    # ============================================================

    st.markdown(
        '<div class="section-label">FEATURE IMPORTANCE STRUCTURE</div>',
        unsafe_allow_html=True
    )

    s1, s2 = st.columns(2)

    with s1:

        structure = pd.DataFrame({
            "Importance Type": [
                "Positive Importance",
                "Zero / Negative Importance"
            ],
            "Features": [
                int(len(positive_imp)),
                int(zero_features)
            ]
        })

        fig = px.pie(
            structure,
            names="Importance Type",
            values="Features",
            hole=0.55,
            title="Importance Distribution"
        )

        fig.update_layout(
            height=430,
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            )
        )

        st.plotly_chart(
            layout(fig, 430),
            use_container_width=True
        )

    with s2:

        top_structure = imp.head(
            min(10, len(imp))
        ).copy()

        top_structure = top_structure.sort_values(
            "Importance",
            ascending=True
        )

        fig = px.bar(
            top_structure,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            title="Top 10 Importance Magnitudes"
        )

        fig.update_traces(
            texttemplate="%{x:.4f}",
            textposition="outside"
        )

        fig.update_layout(
            height=430,
            margin=dict(
                l=20,
                r=80,
                t=70,
                b=30
            )
        )

        st.plotly_chart(
            layout(fig, 430),
            use_container_width=True
        )

    # ============================================================
    # COMPLETE FEATURE IMPORTANCE TABLE
    # ============================================================

    st.markdown(
        '<div class="section-label">FEATURE IMPORTANCE RANKING</div>',
        unsafe_allow_html=True
    )

    ranking = imp.copy()

    ranking.insert(
        0,
        "Rank",
        range(
            1,
            len(ranking) + 1
        )
    )

    ranking["Importance"] = ranking[
        "Importance"
    ].round(5)

    st.dataframe(
        ranking,
        use_container_width=True,
        hide_index=True
    )

    # ============================================================
    # STATE-LEVEL BUSINESS INTELLIGENCE
    # ============================================================

    st.markdown(
        '<div class="section-label">STATE-LEVEL BUSINESS INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    if "State" in filtered.columns:

        state_options = sorted(
            safe_unique(
                filtered["State"]
            ),
            key=lambda x: str(x)
        )

        if state_options:

            selected_state = st.selectbox(
                "Select state for detailed analysis",
                state_options,
                key="feature_state_selector"
            )

            state_df = filtered[
                filtered["State"].astype(str)
                == str(selected_state)
            ].copy()

            state_sales = (
                state_df["Sales"].sum()
                if "Sales" in state_df.columns
                else 0
            )

            state_profit = (
                state_df["Profit"].sum()
                if "Profit" in state_df.columns
                else 0
            )

            state_orders = (
                state_df["Order_ID"].nunique()
                if "Order_ID" in state_df.columns
                else len(state_df)
            )

            state_margin = (
                state_profit / state_sales * 100
                if state_sales
                else 0
            )

            g1, g2, g3, g4 = st.columns(4)

            with g1:

                card_kpi(
                    "STATE SALES",
                    money(state_sales),
                    str(selected_state),
                    "💵"
                )

            with g2:

                card_kpi(
                    "STATE PROFIT",
                    money(state_profit),
                    str(selected_state),
                    "💰"
                )

            with g3:

                card_kpi(
                    "STATE MARGIN",
                    pct(state_margin),
                    "Profit / Sales",
                    "📈"
                )

            with g4:

                card_kpi(
                    "STATE ORDERS",
                    f"{state_orders:,}",
                    "Unique orders",
                    "🧾"
                )

            # ====================================================
            # STATE PERFORMANCE TABLE
            # ====================================================

            state_group = (
                filtered
                .groupby(
                    "State",
                    as_index=False
                )
                .agg(
                    Sales=("Sales", "sum"),
                    Profit=("Profit", "sum")
                )
            )

            if "Order_ID" in filtered.columns:

                order_group = (
                    filtered
                    .groupby("State")["Order_ID"]
                    .nunique()
                    .reset_index(
                        name="Orders"
                    )
                )

                state_group = state_group.merge(
                    order_group,
                    on="State",
                    how="left"
                )

            else:

                state_group["Orders"] = (
                    filtered
                    .groupby("State")
                    .size()
                    .values
                )

            state_group["Margin %"] = np.where(
                state_group["Sales"] != 0,
                (
                    state_group["Profit"]
                    / state_group["Sales"]
                    * 100
                ),
                0
            )

            state_group = state_group.sort_values(
                "Sales",
                ascending=False
            )

            st.markdown(
                '<div class="section-label">STATE PERFORMANCE TABLE</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                state_group.round(2),
                use_container_width=True,
                hide_index=True
            )

            # ====================================================
            # GEOGRAPHIC FEATURE INTELLIGENCE
            # ====================================================

            st.markdown(
                '<div class="section-label">GEOGRAPHIC FEATURE INTELLIGENCE</div>',
                unsafe_allow_html=True
            )

            gmap1, gmap2 = st.columns(2)

            with gmap1:

                state_sales_map = px.choropleth(
                    state_group,
                    locations="State",
                    locationmode="USA-states",
                    color="Sales",
                    scope="usa",
                    hover_name="State",
                    hover_data=[
                        "Sales",
                        "Profit",
                        "Orders",
                        "Margin %"
                    ],
                    title="State Sales Intelligence"
                )

                state_sales_map.update_layout(
                    height=520,
                    margin=dict(
                        l=10,
                        r=10,
                        t=70,
                        b=10
                    )
                )

                st.plotly_chart(
                    layout(
                        state_sales_map,
                        520
                    ),
                    use_container_width=True
                )

            with gmap2:

                state_profit_map = px.choropleth(
                    state_group,
                    locations="State",
                    locationmode="USA-states",
                    color="Profit",
                    scope="usa",
                    hover_name="State",
                    hover_data=[
                        "Sales",
                        "Profit",
                        "Orders",
                        "Margin %"
                    ],
                    title="State Profit Intelligence"
                )

                state_profit_map.update_layout(
                    height=520,
                    margin=dict(
                        l=10,
                        r=10,
                        t=70,
                        b=10
                    )
                )

                st.plotly_chart(
                    layout(
                        state_profit_map,
                        520
                    ),
                    use_container_width=True
                )

            # ====================================================
            # STATE SALES VS PROFIT
            # ====================================================

            state_chart = state_group.head(
                15
            ).copy()

            fig = px.bar(
                state_chart,
                x="State",
                y=[
                    "Sales",
                    "Profit"
                ],
                barmode="group",
                title="Top States — Sales vs Profit"
            )

            fig.update_layout(
                height=500,
                margin=dict(
                    l=20,
                    r=20,
                    t=70,
                    b=80
                )
            )

            st.plotly_chart(
                layout(fig, 500),
                use_container_width=True
            )

        else:

            st.info(
                "No state values are available in the current filtered dataset."
            )

    else:

        st.info(
            "State-level geographic intelligence is unavailable because "
            "the dataset does not contain a State column."
        )

    # ================================================================
    # MODEL EXPLAINABILITY
    # ================================================================
    
    st.markdown(
        '<div class="section-label">MODEL EXPLAINABILITY</div>',
        unsafe_allow_html=True
    )
    
    st.subheader("🧠 Permutation Importance")
    
    st.info(
        f"The existing trained model is **{best_name}**. "
        "No new model is trained on this page."
    )
    
    st.markdown(
        "Feature importance is calculated using **permutation importance** "
        "on the held-out test data. A higher value indicates that disturbing "
        "the feature tends to reduce the model's predictive **F1 performance** "
        "more strongly."
    )
    
    e1, e2 = st.columns(2)
    
    with e1:
    
        st.markdown("### 🥇 Top Predictive Feature")
    
        card_kpi(
            "TOP FEATURE",
            str(top_feature),
            "Most influential predictor",
            "⭐"
        )
    
    with e2:
    
        st.markdown("### 📊 Permutation Importance")
    
        card_kpi(
            "IMPORTANCE",
            f"{top_importance:.5f}",
            "Measured predictive importance",
            "📈"
        )
    
    st.markdown(
        "### ⚠️ Interpretation"
    )
    
    st.warning(
        "Feature importance represents predictive evidence, "
        "not proof of causation. A highly important feature does "
        "not necessarily mean that the feature causes the predicted outcome."
    )

# ================================================================
# BUSINESS INSIGHTS
# ================================================================

elif page == "Business Insights":

    st.title("💡 Business Intelligence Insights")

    # ============================================================
    # KPI CARD ALIGNMENT + CHART COLORS
    # ============================================================

    st.markdown(
        """
        <style>

        /* ========================================================
           KPI ROW
           ======================================================== */

        [data-testid="stHorizontalBlock"] {
            align-items: stretch !important;
        }


        /* ========================================================
           KPI CARD
           ======================================================== */

        .kpi-card {
            width: 100% !important;
            min-width: 0 !important;

            height: 145px !important;
            min-height: 145px !important;
            max-height: 145px !important;

            box-sizing: border-box !important;

            display: flex !important;
            flex-direction: column !important;

            align-items: center !important;
            justify-content: center !important;

            text-align: center !important;

            padding: 14px 10px !important;

            overflow: hidden !important;

            border-radius: 14px !important;

            margin: 0 !important;
        }


        /* ========================================================
           KPI ICON
           ======================================================== */

        .kpi-card .kpi-icon {
            width: 100% !important;

            display: flex !important;
            align-items: center !important;
            justify-content: center !important;

            text-align: center !important;

            margin: 0 0 6px 0 !important;

            line-height: 1 !important;
        }


        /* ========================================================
           KPI TITLE
           ======================================================== */

        .kpi-card .kpi-title {
            width: 100% !important;

            text-align: center !important;

            white-space: nowrap !important;

            overflow: hidden !important;

            text-overflow: ellipsis !important;

            line-height: 1.15 !important;

            margin: 0 0 5px 0 !important;
        }


        /* ========================================================
           KPI VALUE
           ======================================================== */

        .kpi-card .kpi-value {
            width: 100% !important;

            min-width: 0 !important;

            text-align: center !important;

            white-space: nowrap !important;

            overflow: hidden !important;

            text-overflow: ellipsis !important;

            line-height: 1.15 !important;

            margin: 0 !important;

            font-size: clamp(
                15px,
                1.25vw,
                22px
            ) !important;
        }


        /* ========================================================
           LONG KPI VALUE
           ======================================================== */

        .kpi-card .kpi-value.long-text {
            font-size: 14px !important;

            max-width: 100% !important;

            white-space: nowrap !important;

            overflow: hidden !important;

            text-overflow: ellipsis !important;
        }


        /* ========================================================
           KPI SUBTITLE
           ======================================================== */

        .kpi-card .kpi-subtitle {
            width: 100% !important;

            min-width: 0 !important;

            text-align: center !important;

            white-space: nowrap !important;

            overflow: hidden !important;

            text-overflow: ellipsis !important;

            line-height: 1.2 !important;

            margin: 5px 0 0 0 !important;

            font-size: 11px !important;
        }


        /* ========================================================
           NORMAL BUSINESS CARDS
           ======================================================== */

        .card {
            width: 100% !important;

            box-sizing: border-box !important;

            overflow: hidden !important;

            overflow-wrap: anywhere !important;

            word-break: normal !important;
        }


        /* ========================================================
           TABLET
           ======================================================== */

        @media (max-width: 1100px) {

            .kpi-card {
                height: 140px !important;
                min-height: 140px !important;
                max-height: 140px !important;

                padding: 12px 8px !important;
            }

            .kpi-card .kpi-value {
                font-size: 16px !important;
            }

        }


        /* ========================================================
           MOBILE
           ======================================================== */

        @media (max-width: 768px) {

            .kpi-card {
                height: 135px !important;
                min-height: 135px !important;
                max-height: 135px !important;
            }

            .kpi-card .kpi-value {
                font-size: 15px !important;
            }

            .kpi-card .kpi-title {
                font-size: 11px !important;
            }

            .kpi-card .kpi-subtitle {
                font-size: 10px !important;
            }

        }

        </style>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # BASIC DATA VALIDATION
    # ============================================================

    if filtered is None or filtered.empty:

        st.warning(
            "⚠️ No records are available for the current filters."
        )

        st.stop()

    df = filtered.copy()

    # ============================================================
    # SAFE NUMERIC HELPERS
    # ============================================================

    def safe_sum(column):

        if column in df.columns:

            return pd.to_numeric(
                df[column],
                errors="coerce"
            ).fillna(0).sum()

        return 0

    def safe_mean(column):

        if column in df.columns:

            return pd.to_numeric(
                df[column],
                errors="coerce"
            ).dropna().mean()

        return 0

    # ============================================================
    # CORE BUSINESS METRICS
    # ============================================================

    total_sales = safe_sum("Sales")

    total_profit = safe_sum("Profit")

    total_orders = (
        df["Order_ID"].nunique()
        if "Order_ID" in df.columns
        else len(df)
    )

    total_customers = (
        df["Customer_ID"].nunique()
        if "Customer_ID" in df.columns
        else (
            df["Customer_Name"].nunique()
            if "Customer_Name" in df.columns
            else 0
        )
    )

    total_products = (
        df["Product_Name"].nunique()
        if "Product_Name" in df.columns
        else 0
    )

    margin_value = (
        total_profit / total_sales * 100
        if total_sales != 0
        else 0
    )

    aov_value = (
        total_sales / total_orders
        if total_orders != 0
        else 0
    )

    profit_per_order = (
        total_profit / total_orders
        if total_orders != 0
        else 0
    )

    # ============================================================
    # BUSINESS INSIGHT SECTION
    # ============================================================

    st.markdown(
        '<div class="section-label">EXECUTIVE BUSINESS SNAPSHOT</div>',
        unsafe_allow_html=True
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:

        card_kpi(
            "TOTAL SALES",
            money(total_sales),
            "Current filtered revenue",
            "💵"
        )

    with k2:

        card_kpi(
            "TOTAL PROFIT",
            money(total_profit),
            "Current filtered profit",
            "💰"
        )

    with k3:

        card_kpi(
            "PROFIT MARGIN",
            pct(margin_value),
            "Profit / Sales",
            "📈"
        )

    with k4:

        card_kpi(
            "ORDERS",
            f"{total_orders:,}",
            "Unique orders",
            "🧾"
        )

    with k5:

        card_kpi(
            "CUSTOMERS",
            f"{total_customers:,}",
            "Unique customers",
            "👥"
        )

    with k6:

        card_kpi(
            "AOV",
            money(aov_value),
            "Average order value",
            "🛒"
        )

    # ============================================================
    # CATEGORY INTELLIGENCE
    # ============================================================

    if "Category" in df.columns:

        cat = (
            df
            .groupby(
                "Category",
                as_index=False
            )
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
        )

        cat["Margin %"] = np.where(
            cat["Sales"] != 0,
            cat["Profit"] / cat["Sales"] * 100,
            0
        )

        cat = cat.sort_values(
            "Sales",
            ascending=False
        )

    else:

        cat = pd.DataFrame(
            columns=[
                "Category",
                "Sales",
                "Profit",
                "Margin %"
            ]
        )

    # ============================================================
    # REGION INTELLIGENCE
    # ============================================================

    if "Region" in df.columns:

        reg = (
            df
            .groupby(
                "Region",
                as_index=False
            )
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
        )

        reg["Margin %"] = np.where(
            reg["Sales"] != 0,
            reg["Profit"] / reg["Sales"] * 100,
            0
        )

        reg = reg.sort_values(
            "Sales",
            ascending=False
        )

    else:

        reg = pd.DataFrame(
            columns=[
                "Region",
                "Sales",
                "Profit",
                "Margin %"
            ]
        )

    # ============================================================
    # SEGMENT INTELLIGENCE
    # ============================================================

    if "Segment" in df.columns:

        seg = (
            df
            .groupby(
                "Segment",
                as_index=False
            )
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
        )

        seg["Margin %"] = np.where(
            seg["Sales"] != 0,
            seg["Profit"] / seg["Sales"] * 100,
            0
        )

        seg = seg.sort_values(
            "Sales",
            ascending=False
        )

    else:

        seg = pd.DataFrame(
            columns=[
                "Segment",
                "Sales",
                "Profit",
                "Margin %"
            ]
        )

    # ============================================================
    # PRODUCT INTELLIGENCE
    # ============================================================

    if "Product_Name" in df.columns:

        prod = (
            df
            .groupby(
                "Product_Name",
                as_index=False
            )
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
        )

        prod["Margin %"] = np.where(
            prod["Sales"] != 0,
            prod["Profit"] / prod["Sales"] * 100,
            0
        )

        prod = prod.sort_values(
            "Sales",
            ascending=False
        )

    else:

        prod = pd.DataFrame(
            columns=[
                "Product_Name",
                "Sales",
                "Profit",
                "Margin %"
            ]
        )

    # ============================================================
    # TOP PERFORMERS
    # ============================================================

    bestcat = (
        cat.iloc[0]["Category"]
        if not cat.empty
        else "N/A"
    )

    bestreg = (
        reg.iloc[0]["Region"]
        if not reg.empty
        else "N/A"
    )

    bestseg = (
        seg.iloc[0]["Segment"]
        if not seg.empty
        else "N/A"
    )

    bestprod = (
        prod.iloc[0]["Product_Name"]
        if not prod.empty
        else "N/A"
    )

    bestcat_sales = (
        float(cat.iloc[0]["Sales"])
        if not cat.empty
        else 0
    )

    bestreg_sales = (
        float(reg.iloc[0]["Sales"])
        if not reg.empty
        else 0
    )

    bestseg_sales = (
        float(seg.iloc[0]["Sales"])
        if not seg.empty
        else 0
    )

    bestprod_sales = (
        float(prod.iloc[0]["Sales"])
        if not prod.empty
        else 0
    )

    # ============================================================
    # PROFIT LEADERS
    # ============================================================

    most_profitable_category = (
        cat.loc[
            cat["Profit"].idxmax(),
            "Category"
        ]
        if not cat.empty
        else "N/A"
    )

    highest_category_profit = (
        float(cat["Profit"].max())
        if not cat.empty
        else 0
    )

    most_profitable_region = (
        reg.loc[
            reg["Profit"].idxmax(),
            "Region"
        ]
        if not reg.empty
        else "N/A"
    )

    highest_region_profit = (
        float(reg["Profit"].max())
        if not reg.empty
        else 0
    )

    # ============================================================
    # SALES LEADERSHIP
    # ============================================================

    st.markdown(
        '<div class="section-label">SALES LEADERSHIP</div>',
        unsafe_allow_html=True
    )

    l1, l2, l3, l4 = st.columns(4)

    with l1:

        card_kpi(
            "TOP CATEGORY",
            str(bestcat),
            f"Sales {money(bestcat_sales)}",
            "🏆"
        )

    with l2:

        card_kpi(
            "TOP REGION",
            str(bestreg),
            f"Sales {money(bestreg_sales)}",
            "🌎"
        )

    with l3:

        card_kpi(
            "TOP SEGMENT",
            str(bestseg),
            f"Sales {money(bestseg_sales)}",
            "👥"
        )

    with l4:

        product_display = str(bestprod)

        if len(product_display) > 22:

            product_display = (
                product_display[:22] + "..."
            )

        card_kpi(
            "TOP PRODUCT",
            product_display,
            f"Sales {money(bestprod_sales)}",
            "📦"
        )

    # ============================================================
    # EXECUTIVE INTERPRETATION
    # ============================================================

    st.markdown(
        '<div class="section-label">EXECUTIVE INTERPRETATION</div>',
        unsafe_allow_html=True
    )

    st.info(
        f"🏆 {bestcat} leads category sales, "
        f"{bestreg} leads regional sales, and "
        f"{bestseg} leads segment sales. "
        f"The highest-sales product is {bestprod}."
    )

    # ============================================================
    # PROFITABILITY INTELLIGENCE
    # ============================================================

    st.markdown(
        '<div class="section-label">PROFITABILITY INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3, p4 = st.columns(4)

    profitable_categories = (
        int((cat["Profit"] > 0).sum())
        if not cat.empty
        else 0
    )

    loss_categories = (
        int((cat["Profit"] < 0).sum())
        if not cat.empty
        else 0
    )

    with p1:

        card_kpi(
            "TOP PROFIT CATEGORY",
            str(most_profitable_category),
            money(highest_category_profit),
            "💰"
        )

    with p2:

        card_kpi(
            "TOP PROFIT REGION",
            str(most_profitable_region),
            money(highest_region_profit),
            "🌎"
        )

    with p3:

        card_kpi(
            "PROFITABLE CATEGORIES",
            f"{profitable_categories}",
            "Positive profit",
            "🟢"
        )

    with p4:

        card_kpi(
            "LOSS CATEGORIES",
            f"{loss_categories}",
            "Negative profit",
            "🔴"
        )

    # ============================================================
    # CATEGORY PERFORMANCE
    # ============================================================

    st.markdown(
        '<div class="section-label">CATEGORY PERFORMANCE</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        if not cat.empty:

            fig = px.bar(
                cat,
                x="Category",
                y=[
                    "Sales",
                    "Profit"
                ],
                barmode="group",
                title="Category Sales vs Profit",
                text_auto=".2s",
                color_discrete_sequence=[
                    "#2563EB",
                    "#10B981"
                ]
            )

            fig.update_layout(
                height=470
            )

            st.plotly_chart(
                layout(fig, 470),
                use_container_width=True
            )

    with c2:

        if not cat.empty:

            fig = px.bar(
                cat.sort_values(
                    "Margin %",
                    ascending=True
                ),
                x="Margin %",
                y="Category",
                orientation="h",
                text="Margin %",
                title="Category Profit Margin",
                color_discrete_sequence=[
                    "#8B5CF6"
                ]
            )

            fig.update_traces(
                texttemplate="%{x:.1f}%",
                textposition="outside"
            )

            fig.update_layout(
                height=470
            )

            st.plotly_chart(
                layout(fig, 470),
                use_container_width=True
            )

    # ============================================================
    # REGIONAL INTELLIGENCE
    # ============================================================

    st.markdown(
        '<div class="section-label">REGIONAL INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    r1, r2 = st.columns(2)

    with r1:

        if not reg.empty:

            fig = px.bar(
                reg,
                x="Region",
                y=[
                    "Sales",
                    "Profit"
                ],
                barmode="group",
                title="Regional Sales vs Profit",
                text_auto=".2s",
                color_discrete_sequence=[
                    "#2563EB",
                    "#10B981"
                ]
            )

            fig.update_layout(
                height=470
            )

            st.plotly_chart(
                layout(fig, 470),
                use_container_width=True
            )

    with r2:

        if not reg.empty:

            fig = px.scatter(
                reg,
                x="Sales",
                y="Profit",
                size="Sales",
                color="Margin %",
                hover_name="Region",
                title="Regional Profitability Matrix",
                color_continuous_scale=[
                    "#EF4444",
                    "#F59E0B",
                    "#10B981"
                ]
            )

            fig.update_layout(
                height=470
            )

            st.plotly_chart(
                layout(fig, 470),
                use_container_width=True
            )

    # ============================================================
    # SEGMENT INTELLIGENCE
    # ============================================================

    st.markdown(
        '<div class="section-label">CUSTOMER SEGMENT INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    s1, s2 = st.columns(2)

    with s1:

        if not seg.empty:

            fig = px.bar(
                seg,
                x="Segment",
                y="Sales",
                color="Profit",
                title="Segment Sales Contribution",
                text_auto=".2s",
                color_continuous_scale=[
                    "#EF4444",
                    "#F59E0B",
                    "#10B981"
                ]
            )

            fig.update_layout(
                height=450
            )

            st.plotly_chart(
                layout(fig, 450),
                use_container_width=True
            )

    with s2:

        if not seg.empty:

            fig = px.pie(
                seg,
                names="Segment",
                values="Sales",
                hole=0.55,
                title="Sales Mix by Segment",
                color_discrete_sequence=[
                    "#2563EB",
                    "#10B981",
                    "#8B5CF6",
                    "#F59E0B"
                ]
            )

            fig.update_layout(
                height=450
            )

            st.plotly_chart(
                layout(fig, 450),
                use_container_width=True
            )

    # ============================================================
    # PRODUCT INTELLIGENCE
    # ============================================================

    st.markdown(
        '<div class="section-label">PRODUCT INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    if not prod.empty:

        top_products = prod.head(15).copy()

        fig = px.bar(
            top_products.sort_values(
                "Sales",
                ascending=True
            ),
            x="Sales",
            y="Product_Name",
            orientation="h",
            color="Profit",
            title="Top 15 Products by Sales",
            text="Sales",
            color_continuous_scale=[
                "#EF4444",
                "#F59E0B",
                "#10B981"
            ]
        )

        fig.update_traces(
            texttemplate="%{x:.2s}",
            textposition="outside"
        )

        fig.update_layout(
            height=650,
            margin=dict(
                l=20,
                r=100,
                t=70,
                b=30
            )
        )

        st.plotly_chart(
            layout(fig, 650),
            use_container_width=True
        )

        st.dataframe(
            top_products.round(2),
            use_container_width=True,
            hide_index=True
        )

    # ============================================================
    # LOW PROFIT / LOSS ANALYSIS
    # ============================================================

    st.markdown(
        '<div class="section-label">PROFITABILITY RISK ANALYSIS</div>',
        unsafe_allow_html=True
    )

    if not prod.empty:

        risky_products = (
            prod
            .sort_values(
                "Profit",
                ascending=True
            )
            .head(10)
        )

        fig = px.bar(
            risky_products,
            x="Profit",
            y="Product_Name",
            orientation="h",
            title="Lowest-Profit Products",
            color="Profit",
            color_continuous_scale=[
                "#B91C1C",
                "#EF4444",
                "#F59E0B"
            ]
        )

        fig.update_layout(
            height=500
        )

        st.plotly_chart(
            layout(fig, 500),
            use_container_width=True
        )

    # ============================================================
    # TIME INTELLIGENCE
    # ============================================================

    if "Order_Year" in df.columns:

        st.markdown(
            '<div class="section-label">TIME INTELLIGENCE</div>',
            unsafe_allow_html=True
        )

        yearly = (
            df
            .groupby(
                "Order_Year",
                as_index=False
            )
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
        )

        yearly["Margin %"] = np.where(
            yearly["Sales"] != 0,
            yearly["Profit"]
            / yearly["Sales"]
            * 100,
            0
        )

        t1, t2 = st.columns(2)

        with t1:

            fig = px.line(
                yearly,
                x="Order_Year",
                y=[
                    "Sales",
                    "Profit"
                ],
                markers=True,
                title="Yearly Sales & Profit Trend",
                color_discrete_sequence=[
                    "#2563EB",
                    "#10B981"
                ]
            )

            fig.update_layout(
                height=450
            )

            st.plotly_chart(
                layout(fig, 450),
                use_container_width=True
            )

        with t2:

            fig = px.bar(
                yearly,
                x="Order_Year",
                y="Margin %",
                text="Margin %",
                title="Yearly Profit Margin",
                color_discrete_sequence=[
                    "#8B5CF6"
                ]
            )

            fig.update_traces(
                texttemplate="%{y:.1f}%",
                textposition="outside"
            )

            fig.update_layout(
                height=450
            )

            st.plotly_chart(
                layout(fig, 450),
                use_container_width=True
            )

    # ============================================================
    # BUSINESS DECISION MATRIX
    # ============================================================

    st.markdown(
        '<div class="section-label">BUSINESS DECISION MATRIX</div>',
        unsafe_allow_html=True
    )

    if not cat.empty:

        decision = cat.copy()

        decision["Business Signal"] = np.select(
            [
                (
                    (decision["Sales"] >= decision["Sales"].median())
                    &
                    (decision["Profit"] >= decision["Profit"].median())
                ),
                (
                    (decision["Sales"] >= decision["Sales"].median())
                    &
                    (decision["Profit"] < decision["Profit"].median())
                ),
                (
                    (decision["Sales"] < decision["Sales"].median())
                    &
                    (decision["Profit"] >= decision["Profit"].median())
                )
            ],
            [
                "⭐ Scale",
                "⚠️ Optimize Margin",
                "💎 Protect Profitability"
            ],
            default="🔎 Review"
        )

        st.dataframe(
            decision.round(2),
            use_container_width=True,
            hide_index=True
        )

    # ============================================================
    # AUTOMATIC EXECUTIVE RECOMMENDATIONS
    # ============================================================

    st.markdown(
        '<div class="section-label">EXECUTIVE RECOMMENDATIONS</div>',
        unsafe_allow_html=True
    )

    recommendations = []

    recommendations.append(
        f"Focus growth analysis on {bestcat}, "
        f"which currently leads category sales."
    )

    recommendations.append(
        f"Prioritize {bestreg} for regional performance "
        f"and expansion analysis."
    )

    recommendations.append(
        f"Monitor {bestseg} because it represents "
        f"the strongest current segment by sales."
    )

    if not cat.empty:

        weakest_category = cat.loc[
            cat["Profit"].idxmin(),
            "Category"
        ]

        weakest_profit = cat["Profit"].min()

        recommendations.append(
            f"Review {weakest_category} because it has the "
            f"lowest category profit at {money(weakest_profit)}."
        )

    if not prod.empty:

        lowest_product = prod.loc[
            prod["Profit"].idxmin(),
            "Product_Name"
        ]

        recommendations.append(
            f"Investigate the profitability of "
            f"{lowest_product}, especially pricing, discount "
            f"and fulfillment economics."
        )

    recommendations.append(
        f"Current overall margin is {pct(margin_value)} "
        f"with an average order value of {money(aov_value)}."
    )

    for index, recommendation in enumerate(
        recommendations,
        start=1
    ):

        st.markdown(
            f"""
            <div class="card">
                <b>Recommendation {index}</b>
                <p>{recommendation}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ============================================================
    # MANAGEMENT SUMMARY
    # ============================================================

    st.markdown(
        '<div class="section-label">MANAGEMENT SUMMARY</div>',
        unsafe_allow_html=True
    )

    st.success(
        f"📊 Current filtered business performance: "
        f"{money(total_sales)} sales, "
        f"{money(total_profit)} profit, "
        f"{pct(margin_value)} margin across "
        f"{total_orders:,} orders. "
        f"The strongest sales drivers are "
        f"{bestcat}, {bestreg}, and {bestseg}."
    )

# ================================================================
# DATA EXPLORER
# ================================================================

elif page == "Data Explorer":

    st.title("🗂️ Data Explorer")

    # ============================================================
    # BASIC VALIDATION
    # ============================================================

    if filtered is None or filtered.empty:

        st.warning(
            "⚠️ No records are available for the current filters."
        )

        st.stop()

    df_explorer = filtered.copy()

    # ============================================================
    # DATA TYPES
    # ============================================================

    numeric_columns = (
        df_explorer
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    categorical_columns = (
        df_explorer
        .select_dtypes(exclude=np.number)
        .columns
        .tolist()
    )

    total_rows = len(df_explorer)
    total_columns = df_explorer.shape[1]
    total_numeric = len(numeric_columns)
    total_categorical = len(categorical_columns)

    # ============================================================
    # DATASET OVERVIEW
    # ============================================================

    st.markdown(
        '<div class="section-label">DATASET OVERVIEW</div>',
        unsafe_allow_html=True
    )

    a, b, c, d = st.columns(4)

    with a:

        card_kpi(
            "ROWS",
            f"{total_rows:,}",
            "Filtered rows",
            "📊"
        )

    with b:

        card_kpi(
            "COLUMNS",
            f"{total_columns:,}",
            "Dataset columns",
            "🧩"
        )

    with c:

        card_kpi(
            "NUMERIC",
            f"{total_numeric:,}",
            "Numeric columns",
            "🔢"
        )

    with d:

        card_kpi(
            "CATEGORICAL",
            f"{total_categorical:,}",
            "Non-numeric columns",
            "🔤"
        )

    st.write(
        f"Showing **{total_rows:,}** filtered records "
        f"across **{total_columns:,}** columns."
    )

    # ============================================================
    # DEEP SUMMARY STATISTICS
    # ============================================================

    st.markdown(
        '<div class="section-label">DEEP SUMMARY STATISTICS</div>',
        unsafe_allow_html=True
    )

    if numeric_columns:

        summary_stats = (
            df_explorer[numeric_columns]
            .describe()
            .T
            .reset_index()
        )

        summary_stats = summary_stats.rename(
            columns={
                "index": "Column",
                "count": "Count",
                "mean": "Mean",
                "std": "Std Dev",
                "min": "Minimum",
                "25%": "Q1",
                "50%": "Median",
                "75%": "Q3",
                "max": "Maximum"
            }
        )

        # Additional statistics

        summary_stats["Missing"] = [
            df_explorer[col].isna().sum()
            for col in summary_stats["Column"]
        ]

        summary_stats["Missing %"] = [
            (
                df_explorer[col].isna().sum()
                / len(df_explorer)
                * 100
            )
            if len(df_explorer) > 0
            else 0
            for col in summary_stats["Column"]
        ]

        summary_stats["Unique"] = [
            df_explorer[col].nunique()
            for col in summary_stats["Column"]
        ]

        summary_stats = summary_stats[
            [
                "Column",
                "Count",
                "Missing",
                "Missing %",
                "Unique",
                "Mean",
                "Std Dev",
                "Minimum",
                "Q1",
                "Median",
                "Q3",
                "Maximum"
            ]
        ]

        st.dataframe(
            summary_stats.round(2),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "ℹ️ No numeric columns are available "
            "for statistical analysis."
        )

    # ============================================================
    # DATA QUALITY SUMMARY
    # ============================================================

    st.markdown(
        '<div class="section-label">DATA QUALITY SUMMARY</div>',
        unsafe_allow_html=True
    )

    missing_total = int(
        df_explorer.isna().sum().sum()
    )

    duplicate_rows = int(
        df_explorer.duplicated().sum()
    )

    complete_rows = int(
        df_explorer.dropna().shape[0]
    )

    unique_rows = int(
        len(df_explorer.drop_duplicates())
    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:

        card_kpi(
            "MISSING VALUES",
            f"{missing_total:,}",
            "Total missing cells",
            "⚠️"
        )

    with q2:

        card_kpi(
            "DUPLICATES",
            f"{duplicate_rows:,}",
            "Duplicate records",
            "♻️"
        )

    with q3:

        card_kpi(
            "COMPLETE ROWS",
            f"{complete_rows:,}",
            "Rows without missing values",
            "✅"
        )

    with q4:

        card_kpi(
            "UNIQUE ROWS",
            f"{unique_rows:,}",
            "Distinct records",
            "🔎"
        )

    # ============================================================
    # CATEGORICAL SUMMARY
    # ============================================================

    st.markdown(
        '<div class="section-label">CATEGORICAL SUMMARY</div>',
        unsafe_allow_html=True
    )

    if categorical_columns:

        categorical_summary = []

        for column in categorical_columns:

            categorical_summary.append(
                {
                    "Column": column,
                    "Unique Values":
                        df_explorer[column].nunique(),
                    "Missing":
                        df_explorer[column].isna().sum(),
                    "Most Common":
                        (
                            df_explorer[column]
                            .mode()
                            .iloc[0]
                            if not df_explorer[column]
                            .mode()
                            .empty
                            else "N/A"
                        )
                }
            )

        categorical_summary = pd.DataFrame(
            categorical_summary
        )

        st.dataframe(
            categorical_summary,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "ℹ️ No categorical columns available."
        )

    # ============================================================
    # BUSINESS GROUPBY DEEP ANALYSIS
    # ============================================================

    st.markdown(
        '<div class="section-label">BUSINESS GROUPBY DEEP ANALYSIS</div>',
        unsafe_allow_html=True
    )

    business_columns = [
        col
        for col in [
            "Category",
            "Sub_Category",
            "Region",
            "State",
            "City",
            "Segment",
            "Ship_Mode",
            "Product_Name",
            "Customer_Name",
            "Order_Year"
        ]
        if col in df_explorer.columns
    ]

    numeric_business_columns = [
        col
        for col in [
            "Sales",
            "Profit",
            "Quantity",
            "Discount"
        ]
        if col in df_explorer.columns
    ]

    if business_columns:

        g1, g2 = st.columns(2)

        with g1:

            group_column = st.selectbox(
                "Select Business Dimension",
                business_columns,
                key="data_explorer_group_column"
            )

        with g2:

            if numeric_business_columns:

                measure_column = st.selectbox(
                    "Select Business Measure",
                    numeric_business_columns,
                    key="data_explorer_measure_column"
                )

            else:

                measure_column = None

        if measure_column:

            grouped_business = (
                df_explorer
                .groupby(
                    group_column,
                    as_index=False
                )
                .agg(
                    Total=(
                        measure_column,
                        "sum"
                    ),
                    Average=(
                        measure_column,
                        "mean"
                    ),
                    Minimum=(
                        measure_column,
                        "min"
                    ),
                    Maximum=(
                        measure_column,
                        "max"
                    ),
                    Records=(
                        measure_column,
                        "count"
                    )
                )
            )

            grouped_business["Median"] = (
                df_explorer
                .groupby(group_column)[measure_column]
                .median()
                .values
            )

            grouped_business["Contribution %"] = (
                grouped_business["Total"]
                / grouped_business["Total"].sum()
                * 100
                if grouped_business["Total"].sum() != 0
                else 0
            )

            grouped_business = (
                grouped_business
                .sort_values(
                    "Total",
                    ascending=False
                )
                .reset_index(drop=True)
            )

            # ----------------------------------------------------
            # GROUPBY KPI
            # ----------------------------------------------------

            top_business = (
                grouped_business.iloc[0][group_column]
                if not grouped_business.empty
                else "N/A"
            )

            top_business_value = (
                float(
                    grouped_business.iloc[0]["Total"]
                )
                if not grouped_business.empty
                else 0
            )

            lowest_business = (
                grouped_business.iloc[-1][group_column]
                if not grouped_business.empty
                else "N/A"
            )

            lowest_business_value = (
                float(
                    grouped_business.iloc[-1]["Total"]
                )
                if not grouped_business.empty
                else 0
            )

            unique_business = (
                grouped_business[group_column]
                .nunique()
            )

            gb1, gb2, gb3, gb4 = st.columns(4)

            with gb1:

                card_kpi(
                    "TOP BUSINESS DRIVER",
                    str(top_business),
                    f"{measure_column}: "
                    f"{money(top_business_value)}",
                    "🏆"
                )

            with gb2:

                card_kpi(
                    "LOWEST DRIVER",
                    str(lowest_business),
                    f"{measure_column}: "
                    f"{money(lowest_business_value)}",
                    "📉"
                )

            with gb3:

                card_kpi(
                    "BUSINESS GROUPS",
                    f"{unique_business:,}",
                    "Unique groups",
                    "🧩"
                )

            with gb4:

                card_kpi(
                    "TOTAL MEASURE",
                    money(
                        grouped_business["Total"].sum()
                    ),
                    f"Total {measure_column}",
                    "💰"
                )

            # ----------------------------------------------------
            # GROUPED BUSINESS TABLE
            # ----------------------------------------------------

            st.markdown(
                '<div class="section-label">'
                'GROUPED BUSINESS STATISTICS'
                '</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                grouped_business.round(2),
                use_container_width=True,
                hide_index=True
            )

            # ----------------------------------------------------
            # GROUPED BUSINESS CHART
            # ----------------------------------------------------

            if len(grouped_business) > 0:

                chart_data = (
                    grouped_business
                    .head(15)
                    .sort_values(
                        "Total",
                        ascending=True
                    )
                )

                fig = px.bar(
                    chart_data,
                    x="Total",
                    y=group_column,
                    orientation="h",
                    color="Total",
                    text="Total",
                    title=(
                        f"{measure_column} by "
                        f"{group_column}"
                    )
                )

                fig.update_traces(
                    texttemplate="%{x:.2s}",
                    textposition="outside"
                )

                fig.update_layout(
                    height=550
                )

                st.plotly_chart(
                    layout(fig, 550),
                    use_container_width=True
                )

    else:

        st.info(
            "ℹ️ No standard business dimensions "
            "were detected in the filtered dataset."
        )

    # ============================================================
    # FULL FILTERED DATA
    # ============================================================

    st.markdown(
        '<div class="section-label">FULL FILTERED DATASET</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"Showing **{len(df_explorer):,}** filtered records."
    )

    st.dataframe(
        df_explorer,
        use_container_width=True,
        height=600,
        hide_index=True
    )

    # ============================================================
    # FULL CSV EXPORT
    # ============================================================

    st.markdown(
        '<div class="section-label">DATA EXPORT</div>',
        unsafe_allow_html=True
    )

    csv_bytes = (
        df_explorer
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Full Filtered CSV",
        data=csv_bytes,
        file_name="SuperStore_Full_Filtered_Export.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ================================================================
    # FULL DATASET DOWNLOAD
    # ================================================================
    
    st.markdown(
    """
    <div class="section-label" style="
        text-align: center;
        width: 100%;
    ">
        OR
    </div>
    """,
    unsafe_allow_html=True
    )
    
    # Original complete dataset
    # IMPORTANT:
    # Replace `df` below with the variable that contains your
    # ORIGINAL / FULL CSV dataset if your app uses another name.
    full_csv_bytes = df.to_csv(
        index=False
    ).encode("utf-8")
    
    st.download_button(
        "⬇️ Download Full CSV",
        data=full_csv_bytes,
        file_name="SuperStore_Full_Dataset.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ============================================================
    # BEAUTIFUL PDF REPORT
    # ============================================================

    try:

        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            PageBreak
        )
        from reportlab.lib.units import mm
        from io import BytesIO

        def create_data_explorer_pdf():

            buffer = BytesIO()

            document = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=15 * mm,
                leftMargin=15 * mm,
                topMargin=15 * mm,
                bottomMargin=15 * mm
            )

            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                "ReportTitle",
                parent=styles["Title"],
                alignment=TA_CENTER,
                fontSize=22,
                leading=28,
                spaceAfter=10
            )

            subtitle_style = ParagraphStyle(
                "Subtitle",
                parent=styles["Normal"],
                alignment=TA_CENTER,
                fontSize=10,
                leading=14,
                spaceAfter=18
            )

            heading_style = ParagraphStyle(
                "SectionHeading",
                parent=styles["Heading2"],
                fontSize=14,
                leading=18,
                spaceBefore=12,
                spaceAfter=8
            )

            body_style = ParagraphStyle(
                "Body",
                parent=styles["Normal"],
                fontSize=9,
                leading=13
            )

            story = []

            # ----------------------------------------------------
            # TITLE
            # ----------------------------------------------------

            story.append(
                Paragraph(
                    "SuperStore AI Intelligence",
                    title_style
                )
            )

            story.append(
                Paragraph(
                    "Data Explorer & Business Analytics Report",
                    subtitle_style
                )
            )

            story.append(
                Paragraph(
                    f"Filtered Records: {total_rows:,} "
                    f"| Columns: {total_columns:,}",
                    body_style
                )
            )

            story.append(
                Spacer(1, 10)
            )

            # ----------------------------------------------------
            # OVERVIEW
            # ----------------------------------------------------

            story.append(
                Paragraph(
                    "Dataset Overview",
                    heading_style
                )
            )

            overview_data = [
                [
                    "Metric",
                    "Value"
                ],
                [
                    "Rows",
                    f"{total_rows:,}"
                ],
                [
                    "Columns",
                    f"{total_columns:,}"
                ],
                [
                    "Numeric Columns",
                    f"{total_numeric:,}"
                ],
                [
                    "Categorical Columns",
                    f"{total_categorical:,}"
                ],
                [
                    "Missing Values",
                    f"{missing_total:,}"
                ],
                [
                    "Duplicate Rows",
                    f"{duplicate_rows:,}"
                ]
            ]

            overview_table = Table(
                overview_data,
                colWidths=[
                    80 * mm,
                    80 * mm
                ]
            )

            overview_table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor("#1F4E78")
                        ),
                        (
                            "TEXTCOLOR",
                            (0, 0),
                            (-1, 0),
                            colors.white
                        ),
                        (
                            "FONTNAME",
                            (0, 0),
                            (-1, 0),
                            "Helvetica-Bold"
                        ),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.grey
                        ),
                        (
                            "ALIGN",
                            (1, 1),
                            (1, -1),
                            "RIGHT"
                        ),
                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "MIDDLE"
                        ),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [
                                colors.whitesmoke,
                                colors.lightgrey
                            ]
                        )
                    ]
                )
            )

            story.append(
                overview_table
            )

            # ----------------------------------------------------
            # NUMERIC SUMMARY
            # ----------------------------------------------------

            if numeric_columns:

                story.append(
                    Paragraph(
                        "Deep Summary Statistics",
                        heading_style
                    )
                )

                pdf_summary = (
                    summary_stats
                    .copy()
                )

                pdf_summary = pdf_summary[
                    [
                        "Column",
                        "Count",
                        "Missing",
                        "Mean",
                        "Median",
                        "Minimum",
                        "Maximum"
                    ]
                ]

                pdf_summary = pdf_summary.round(2)

                summary_table_data = [
                    list(pdf_summary.columns)
                ]

                for row in pdf_summary.itertuples(
                    index=False
                ):

                    summary_table_data.append(
                        [
                            str(value)
                            for value in row
                        ]
                    )

                summary_table = Table(
                    summary_table_data,
                    repeatRows=1,
                    colWidths=[
                        43 * mm,
                        20 * mm,
                        20 * mm,
                        23 * mm,
                        23 * mm,
                        25 * mm,
                        25 * mm
                    ]
                )

                summary_table.setStyle(
                    TableStyle(
                        [
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, 0),
                                colors.HexColor(
                                    "#2F75B5"
                                )
                            ),
                            (
                                "TEXTCOLOR",
                                (0, 0),
                                (-1, 0),
                                colors.white
                            ),
                            (
                                "FONTNAME",
                                (0, 0),
                                (-1, 0),
                                "Helvetica-Bold"
                            ),
                            (
                                "FONTSIZE",
                                (0, 0),
                                (-1, -1),
                                7
                            ),
                            (
                                "GRID",
                                (0, 0),
                                (-1, -1),
                                0.3,
                                colors.grey
                            ),
                            (
                                "ALIGN",
                                (1, 1),
                                (-1, -1),
                                "RIGHT"
                            ),
                            (
                                "VALIGN",
                                (0, 0),
                                (-1, -1),
                                "MIDDLE"
                            )
                        ]
                    )
                )

                story.append(
                    summary_table
                )

            # ----------------------------------------------------
            # BUSINESS GROUPBY
            # ----------------------------------------------------

            if (
                business_columns
                and numeric_business_columns
            ):

                story.append(
                    Paragraph(
                        "Business GroupBy Analysis",
                        heading_style
                    )
                )

                if "grouped_business" in locals():

                    pdf_grouped = (
                        grouped_business
                        .copy()
                        .head(30)
                    )

                    pdf_grouped = pdf_grouped.round(2)

                    group_table_data = [
                        list(
                            pdf_grouped.columns
                        )
                    ]

                    for row in pdf_grouped.itertuples(
                        index=False
                    ):

                        group_table_data.append(
                            [
                                str(value)
                                for value in row
                            ]
                        )

                    group_table = Table(
                        group_table_data,
                        repeatRows=1
                    )

                    group_table.setStyle(
                        TableStyle(
                            [
                                (
                                    "BACKGROUND",
                                    (0, 0),
                                    (-1, 0),
                                    colors.HexColor(
                                        "#548235"
                                    )
                                ),
                                (
                                    "TEXTCOLOR",
                                    (0, 0),
                                    (-1, 0),
                                    colors.white
                                ),
                                (
                                    "FONTNAME",
                                    (0, 0),
                                    (-1, 0),
                                    "Helvetica-Bold"
                                ),
                                (
                                    "FONTSIZE",
                                    (0, 0),
                                    (-1, -1),
                                    7
                                ),
                                (
                                    "GRID",
                                    (0, 0),
                                    (-1, -1),
                                    0.3,
                                    colors.grey
                                ),
                                (
                                    "VALIGN",
                                    (0, 0),
                                    (-1, -1),
                                    "MIDDLE"
                                )
                            ]
                        )
                    )

                    story.append(
                        group_table
                    )

            # ----------------------------------------------------
            # MANAGEMENT SUMMARY
            # ----------------------------------------------------

            story.append(
                Paragraph(
                    "Management Summary",
                    heading_style
                )
            )

            story.append(
                Paragraph(
                    f"The current filtered dataset contains "
                    f"<b>{total_rows:,}</b> records across "
                    f"<b>{total_columns:,}</b> columns. "
                    f"It contains <b>{total_numeric:,}</b> "
                    f"numeric and <b>{total_categorical:,}</b> "
                    f"categorical columns.",
                    body_style
                )
            )

            story.append(
                Spacer(1, 8)
            )

            story.append(
                Paragraph(
                    f"Data quality analysis identified "
                    f"<b>{missing_total:,}</b> missing cells "
                    f"and <b>{duplicate_rows:,}</b> duplicate "
                    f"records.",
                    body_style
                )
            )

            document.build(story)

            buffer.seek(0)

            return buffer.getvalue()

        # ========================================================
        # PDF BUTTON
        # ========================================================

        st.markdown(
            '<div class="section-label">REPORT GENERATION</div>',
            unsafe_allow_html=True
        )

        pdf_bytes = create_data_explorer_pdf()

        st.download_button(
            "📄 Generate & Download Beautiful PDF Report",
            data=pdf_bytes,
            file_name="SuperStore_Data_Explorer_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    except ImportError:

        st.warning(
            "⚠️ PDF generation requires ReportLab. "
            "Install it using: pip install reportlab"
        )

# ================================================================
# ABOUT
# ================================================================

elif page == "About":

    st.title("ℹ️ About the Platform")

    # ============================================================
    # PLATFORM HERO
    # ============================================================

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                SuperStore AI Intelligence Platform
            </div>
            <div class="hero-subtitle">
                An end-to-end data science application combining
                business analytics, exploratory analysis,
                classification modelling, evaluation,
                explainability and prediction.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # BUSINESS PROBLEM
    # ============================================================

    st.markdown(
        '<div class="section-label">BUSINESS PROBLEM</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.subheader("🎯 What problem does this solve?")

        st.write(
            """
            Historical transaction data can explain what happened,
            but organizations also need predictive signals for
            future decisions.
            """
        )

        st.write(
            """
            This application converts SuperStore transaction data
            into a reusable machine-learning workflow for predicting
            **Return_Flag**.
            """
        )

        st.write(
            """
            The platform also keeps the original business analytics
            layer covering sales, profit, customers, orders,
            category, region, segment, product and profitability
            intelligence.
            """
        )

    # ============================================================
    # PLATFORM CAPABILITIES
    # ============================================================

    st.markdown(
        '<div class="section-label">PLATFORM CAPABILITIES</div>',
        unsafe_allow_html=True
    )

    cap1, cap2, cap3, cap4 = st.columns(4)

    with cap1:

        with st.container(border=True):

            st.markdown(
                "<div style='text-align:center; font-size:32px;'>📊</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<h3 style='text-align:center;'>Business Analytics</h3>",
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div style="text-align:center;">
                    Sales, profit, orders, customers and
                    profitability intelligence.
                </div>
                """,
                unsafe_allow_html=True
            )

    with cap2:

        with st.container(border=True):

            st.markdown(
                "<div style='text-align:center; font-size:32px;'>🔎</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<h3 style='text-align:center;'>Data Explorer</h3>",
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div style="text-align:center;">
                    Dataset structure, quality, statistics
                    and business group analysis.
                </div>
                """,
                unsafe_allow_html=True
            )

    with cap3:

        with st.container(border=True):

            st.markdown(
                "<div style='text-align:center; font-size:32px;'>🤖</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<h3 style='text-align:center;'>Machine Learning</h3>",
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div style="text-align:center;">
                    Classification modelling for
                    Return_Flag prediction.
                </div>
                """,
                unsafe_allow_html=True
            )

    with cap4:

        with st.container(border=True):

            st.markdown(
                "<div style='text-align:center; font-size:32px;'>🧠</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<h3 style='text-align:center;'>Explainability</h3>",
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div style="text-align:center;">
                    Model evaluation and predictive
                    feature interpretation.
                </div>
                """,
                unsafe_allow_html=True
            )

    # ============================================================
    # TECHNOLOGY STACK
    # ============================================================

    st.markdown(
        '<div class="section-label">TECHNOLOGY STACK</div>',
        unsafe_allow_html=True
    )

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:
        card_kpi(
            "PYTHON",
            "✓",
            "Core application",
            "🐍"
        )

    with tech2:
        card_kpi(
            "STREAMLIT",
            "✓",
            "Interactive web application",
            "🌐"
        )

    with tech3:
        card_kpi(
            "PANDAS / NUMPY",
            "✓",
            "Data engineering",
            "📊"
        )

    with tech4:
        card_kpi(
            "SCIKIT-LEARN",
            "✓",
            "Machine learning",
            "🤖"
        )

    # ============================================================
    # MODEL GOVERNANCE
    # ============================================================

    st.markdown(
        '<div class="section-label">MODEL GOVERNANCE</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.subheader("🎯 Why F1 is primary")

        st.write(
            """
            Accuracy alone can be misleading when the target
            classes are imbalanced.
            """
        )

        st.write(
            """
            The platform therefore compares **Accuracy, Precision,
            Recall, F1 and ROC-AUC** and selects the best model
            primarily by **F1**.
            """
        )

        st.write(
            """
            This is a model-selection rule, not a claim that the
            model causes returns.
            """
        )

        st.write(
            """
            Feature importance is also interpreted as
            **predictive association rather than causation**.
            """
        )

    # ============================================================
    # END-TO-END ANALYTICS WORKFLOW
    # ============================================================

    st.markdown(
        '<div class="section-label">END-TO-END ANALYTICS WORKFLOW</div>',
        unsafe_allow_html=True
    )

    workflow = [
        ("STEP 01", "Data", "Transaction data", "📥"),
        ("STEP 02", "Explore", "EDA & quality", "🔎"),
        ("STEP 03", "Engineer", "Feature preparation", "⚙️"),
        ("STEP 04", "Model", "Classification", "🤖"),
        ("STEP 05", "Evaluate", "Model metrics", "📈"),
        ("STEP 06", "Predict", "Return prediction", "🎯")
    ]

    w1, w2, w3 = st.columns(3)

    for index, (step, title, description, icon) in enumerate(workflow):

        if index < 2:
            current_col = w1
        elif index < 4:
            current_col = w2
        else:
            current_col = w3

        with current_col:

            with st.container(border=True):

                st.markdown(
                    f"### {icon} {title}"
                )

                st.caption(step)

                st.write(
                    description
                )

        if index in [1, 3]:

            st.write("")

    # ============================================================
    # DATA SCIENCE PRINCIPLES
    # ============================================================

    st.markdown(
        '<div class="section-label">DATA SCIENCE PRINCIPLES</div>',
        unsafe_allow_html=True
    )

    principle1, principle2, principle3 = st.columns(3)

    with principle1:

        with st.container(border=True):

            st.markdown(
                "<div style='text-align:center; font-size:30px;'>📌</div>",
                unsafe_allow_html=True
            )

            st.subheader(
                "Descriptive Intelligence"
            )

            st.write(
                "Understand historical business performance "
                "before modelling."
            )

    with principle2:

        with st.container(border=True):

            st.markdown(
                "<div style='text-align:center; font-size:30px;'>⚖️</div>",
                unsafe_allow_html=True
            )

            st.subheader(
                "Responsible Evaluation"
            )

            st.write(
                "Compare multiple classification metrics "
                "instead of relying only on accuracy."
            )

    with principle3:

        with st.container(border=True):

            st.markdown(
                "<div style='text-align:center; font-size:30px;'>🔍</div>",
                unsafe_allow_html=True
            )

            st.subheader(
                "Explainable Decisions"
            )

            st.write(
                "Interpret model outputs as predictive signals "
                "rather than causal conclusions."
            )

    # ============================================================
    # PROJECT AUTHOR
    # ============================================================

    st.markdown(
        '<div class="section-label">PROJECT AUTHOR</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.markdown(
            "<h2 style='text-align:center;'>S Mohammed Kaif</h2>",
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="
                text-align:center;
                font-size:15px;
                margin-bottom:15px;
            ">
                Data Science • Machine Learning • AI • Data Analytics
            </div>
            """,
            unsafe_allow_html=True
        )

        author1, author2, author3 = st.columns([1, 1, 1])

        with author1:
            st.write("")

        with author2:

            st.link_button(
                "🔗 LinkedIn",
                LINKEDIN_URL,
                use_container_width=True
            )

            st.link_button(
                "💻 GitHub",
                GITHUB_URL,
                use_container_width=True
            )

        with author3:
            st.write("")

        st.markdown(
            """
            <div style="
                text-align:center;
                font-size:12px;
                opacity:0.7;
                margin-top:15px;
            ">
                SuperStore AI Intelligence Platform • 2026
            </div>
            """,
            unsafe_allow_html=True
        )

# ================================================================
# FOOTER
# ================================================================

if page != "About":

    st.divider()

    st.header("🤖 SuperStore AI Intelligence Platform")

    st.caption(
        "End-to-end data science, machine learning and business "
        "intelligence platform for SuperStore analytics, "
        "Return_Flag prediction and decision support."
    )

    # ============================================================
    # ANALYTICS / DATA SCIENCE / ML
    # ============================================================

    f1, f2, f3 = st.columns(3)

    with f1:

        st.subheader("📊 Analytics")

        st.write(
            "• Sales & Profit\n\n"
            "• Orders & Customers\n\n"
            "• Products & Categories\n\n"
            "• Regions & Segments\n\n"
            "• Profitability Intelligence"
        )

    with f2:

        st.subheader("🔎 Data Science")

        st.write(
            "• Data Exploration\n\n"
            "• Data Quality Analysis\n\n"
            "• Summary Statistics\n\n"
            "• Business GroupBy Analysis\n\n"
            "• Feature Analysis"
        )

    with f3:

        st.subheader("🤖 Machine Learning")

        st.write(
            "• Classification\n\n"
            "• Return_Flag Prediction\n\n"
            "• Model Comparison\n\n"
            "• F1-based Model Selection\n\n"
            "• Explainability"
        )

    st.divider()

    # ============================================================
    # TECHNOLOGY STACK
    # ============================================================

    st.subheader("🛠️ Technology Stack")

    tech1, tech2, tech3, tech4, tech5 = st.columns(5)

    with tech1:
        st.metric(
            "🐍 Python",
            "Core"
        )

    with tech2:
        st.metric(
            "🌐 Streamlit",
            "Application"
        )

    with tech3:
        st.metric(
            "📊 Pandas / NumPy",
            "Data"
        )

    with tech4:
        st.metric(
            "🤖 Scikit-learn",
            "ML"
        )

    with tech5:
        st.metric(
            "📈 Plotly",
            "Visuals"
        )

    st.divider()

    # ============================================================
    # RESPONSIBLE AI
    # ============================================================

    st.subheader("⚖️ Responsible Analytics & AI")

    st.info(
        "Business analytics describe historical patterns, while "
        "machine-learning outputs represent predictive signals.\n\n"
        "Model metrics and feature importance should therefore be "
        "interpreted as analytical evidence rather than causal conclusions.\n\n"
        "F1 is used as the primary model-selection metric while "
        "Accuracy, Precision, Recall and ROC-AUC provide additional "
        "evaluation context."
    )

    st.divider()

    # ============================================================
    # END-TO-END WORKFLOW
    # ============================================================

    st.subheader("🔄 End-to-End Workflow")

    workflow = [
        ("01", "📂 Data Loading"),
        ("02", "🔎 Data Understanding"),
        ("03", "📊 EDA"),
        ("04", "⚙️ Preprocessing"),
        ("05", "🤖 Model Training"),
        ("06", "🏆 Model Comparison"),
        ("07", "🎯 Prediction"),
        ("08", "📋 Evaluation"),
        ("09", "💡 Explainability"),
        ("10", "💼 Decision Support"),
    ]

    w1, w2, w3, w4, w5 = st.columns(5)

    workflow_columns = [
        w1, w2, w3, w4, w5
    ]

    for index, col in enumerate(
        workflow_columns
    ):

        with col:

            step1 = workflow[index * 2]
            step2 = workflow[index * 2 + 1]

            st.write(
                f"**{step1[0]} — {step1[1]}**"
            )

            st.caption(
                f"**{step2[0]} — {step2[1]}**"
            )

    st.divider()

    # ============================================================
    # PROJECT AUTHOR
    # ============================================================

    st.subheader("👨‍💻 Project Author")

    st.write(
        "**S Mohammed Kaif**"
    )

    st.caption(
        "Data Science • Machine Learning • AI • Data Analytics"
    )

    st.write(
        "SuperStore AI Intelligence Platform • 2026"
    )

    st.caption(
        "Built with Python • Streamlit • Pandas • NumPy • "
        "Scikit-learn • Plotly"
    )

    st.caption(
        "© 2026 S Mohammed Kaif"
    )

    # ============================================================
    # SOCIAL LINKS
    # ============================================================

    link1, link2 = st.columns(2)

    with link1:

        st.link_button(
            "🔗 LinkedIn",
            LINKEDIN_URL,
            use_container_width=True
        )

    with link2:

        st.link_button(
            "💻 GitHub",
            GITHUB_URL,
            use_container_width=True
        )
