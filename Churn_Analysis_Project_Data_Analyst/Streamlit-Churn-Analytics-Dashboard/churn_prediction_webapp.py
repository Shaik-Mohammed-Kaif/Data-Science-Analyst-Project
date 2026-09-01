"""
CUSTOMER CHURN AI
Customer Risk Prediction & Retention Intelligence
---------------------------------------------------
A single-file, portfolio-grade Streamlit application that trains
machine-learning models directly from a churn CSV and provides
prediction, explainability, analytics and retention intelligence.

Run:
    streamlit run Churn-Prediction.py
"""

# =========================================================
# IMPORTS
# =========================================================
import os
import time
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from scipy import stats

warnings.filterwarnings("ignore")

# ---- Optional ML libraries (never crash the app if missing OR broken) ----
# NOTE: on some Windows/conda setups, these libraries fail with binary
# incompatibility errors (ValueError / OSError / RuntimeError), not just
# ImportError (e.g. "numpy.dtype size changed" from a mismatched numpy/C
# extension build). We must catch broadly so the app never dies at import.
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# =========================================================
# CONSTANTS
# =========================================================
CSV_PATH = "Customer_Churn_Predictions.csv"
GITHUB_URL = "https://github.com/Shaik-Mohammed-Kaif"
RANDOM_STATE = 42

TARGET_HINTS = [
    "churn", "churn_flag", "customer_status", "exited", "is_churned",
    "churn_status", "churn_status_predicted", "churn_prediction",
]

ID_HINTS = ["id", "customer_id", "cust_id", "custid"]

LEAKAGE_NAME_HINTS = [
    "prediction", "predicted", "probability", "risk_score", "model_prediction",
    "predicted_probability", "predicted_churn", "churn_probability",
    "churn_category", "churn_reason", "churn_status_predicted",
    "churn_prediction", "churn_status",
]

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# THEME / CSS
# =========================================================
THEME_PALETTES = {
    "Light": dict(
        bg="#f5f7fb", bg2="#ffffff", card="#ffffff", border="#e4e8f0",
        text="#1a1f2b", subtext="#5b6478", accent="#3355ff", accent2="#7a3ff2",
        good="#0ea472", warn="#b57b00", bad="#d6423a",
    ),
    "Dark": dict(
        bg="#0b0f19", bg2="#111827", card="#161d2e", border="#232c3f",
        text="#e7ebf3", subtext="#93a0b8", accent="#6d8cff", accent2="#8b6dff",
        good="#33d69f", warn="#f6c344", bad="#ff6b6b",
    ),
    "Cream Vanilla": dict(
        bg="#faf3e6", bg2="#fffaf0", card="#fffdf7", border="#ecdcc0",
        text="#4a3728", subtext="#8a715a", accent="#c17f3e", accent2="#a8562f",
        good="#5f8d4e", warn="#b5852f", bad="#b5482f",
    ),
    "Bubbles": dict(
        bg="#eaf6fb", bg2="#ffffff", card="#ffffff", border="#cdeaf5",
        text="#0f2a3d", subtext="#4d7488", accent="#1fb6d0", accent2="#8e5cf0",
        good="#0fa679", warn="#d9a02b", bad="#e0554a",
    ),
}


def inject_css(theme: str) -> None:
    """Inject theme-aware CSS across the whole application."""
    palette = THEME_PALETTES.get(theme, THEME_PALETTES["Light"])
    bg = palette["bg"]; bg2 = palette["bg2"]; card = palette["card"]; border = palette["border"]
    text = palette["text"]; subtext = palette["subtext"]; accent = palette["accent"]
    accent2 = palette["accent2"]; good = palette["good"]; warn = palette["warn"]; bad = palette["bad"]

    bubbles_css = ""
    if theme == "Bubbles":
        circles = []
        sizes = [90, 140, 60, 110, 75, 160, 50, 100]
        lefts = [4, 18, 32, 46, 60, 72, 84, 94]
        durations = [18, 24, 15, 21, 27, 19, 16, 23]
        delays = [0, 3, 6, 1, 8, 4, 10, 2]
        for i in range(8):
            circles.append(
                f".cc-bubble:nth-child({i+1}) {{ width:{sizes[i]}px; height:{sizes[i]}px; "
                f"left:{lefts[i]}%; animation-duration:{durations[i]}s; animation-delay:{delays[i]}s; }}"
            )
        bubble_rules = "\n".join(circles)
        bubbles_css = f"""
        .cc-bubble-field {{ position: fixed; inset: 0; overflow: hidden; z-index: 0; pointer-events: none; }}
        .cc-bubble {{
            position: absolute; bottom: -160px; border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, rgba(31,182,208,0.28), rgba(142,92,240,0.12));
            animation-name: cc-float-up; animation-timing-function: ease-in; animation-iteration-count: infinite;
        }}
        @keyframes cc-float-up {{
            0% {{ transform: translateY(0) translateX(0); opacity: 0; }}
            10% {{ opacity: 0.9; }}
            100% {{ transform: translateY(-115vh) translateX(30px); opacity: 0; }}
        }}
        {bubble_rules}
        section[data-testid="stAppViewContainer"], section[data-testid="stSidebar"] {{ position: relative; z-index: 1; }}
        """

    st.markdown(f"""
    <style>
        :root {{
            --bg: {bg}; --bg2: {bg2}; --card: {card}; --border: {border};
            --text: {text}; --subtext: {subtext}; --accent: {accent};
            --accent2: {accent2}; --good: {good}; --warn: {warn}; --bad: {bad};
        }}
        {bubbles_css}
        .stApp {{ background: var(--bg); color: var(--text); }}
        section[data-testid="stSidebar"] {{
            background: var(--bg2); border-right: 1px solid var(--border);
        }}
        h1, h2, h3, h4, h5, h6, p, span, label, div {{ color: var(--text); }}
        .cc-header {{ text-align:center; padding: 0.6rem 0 1.2rem 0; }}
        .cc-title {{
            font-size: 2.1rem; font-weight: 800; letter-spacing: 1px;
            background: linear-gradient(90deg, var(--accent), var(--accent2));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .cc-subtitle {{ font-size: 1.02rem; color: var(--subtext); margin-top: 2px; }}
        .cc-tag {{
            font-size: 0.72rem; letter-spacing: 2px; color: var(--subtext);
            text-transform: uppercase; margin-top: 6px;
        }}
        .cc-card {{
            background: var(--card); border: 1px solid var(--border);
            border-radius: 14px; padding: 1.1rem 1.3rem; margin-bottom: 0.9rem;
        }}
        .cc-metric-label {{ font-size: 0.78rem; color: var(--subtext); text-transform: uppercase; letter-spacing: 1px;}}
        .cc-metric-value {{ font-size: 1.65rem; font-weight: 750; color: var(--text); }}
        .cc-pill {{
            display:inline-block; padding: 3px 12px; border-radius: 999px;
            font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px;
        }}
        .cc-pill-low {{ background: rgba(14,164,114,0.15); color: var(--good); }}
        .cc-pill-medium {{ background: rgba(181,123,0,0.15); color: var(--warn); }}
        .cc-pill-high {{ background: rgba(214,66,58,0.15); color: var(--bad); }}
        .cc-pill-veryhigh {{ background: var(--bad); color: white; }}
        .cc-section-title {{
            font-size: 1.05rem; font-weight: 700; margin: 0.4rem 0 0.6rem 0;
            border-left: 4px solid var(--accent); padding-left: 10px;
        }}
        .cc-footer {{
            text-align:center; padding: 1.6rem 0 0.6rem 0; color: var(--subtext);
            border-top: 1px solid var(--border); margin-top: 2rem; font-size: 0.85rem;
        }}
        .stButton>button {{
            border-radius: 10px; border: 1px solid var(--border);
            background: linear-gradient(90deg, var(--accent), var(--accent2));
            color: white; font-weight: 600;
        }}
        div[data-testid="stMetric"] {{
            background: var(--card); border: 1px solid var(--border);
            border-radius: 12px; padding: 0.8rem;
        }}
        .cc-status-dot {{ color: var(--good); }}
        thead tr th {{ background: var(--bg2) !important; }}
    </style>
    """, unsafe_allow_html=True)





def plot_template() -> str:
    return "plotly_dark" if st.session_state.theme == "Dark" else "plotly_white"


def render_bubble_background() -> None:
    if st.session_state.theme != "Bubbles":
        return
    bubbles_html = "".join(['<div class="cc-bubble"></div>' for _ in range(8)])
    st.markdown(f'<div class="cc-bubble-field">{bubbles_html}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template=plot_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig

# =========================================================
# SESSION STATE
# =========================================================
def init_session_state() -> None:
    defaults = {
        "selected_page": "Business Overview",
        "theme": "Light",
        "selected_model_name": None,
        "trained_models": {},
        "best_model_name": None,
        "model_metrics": None,
        "last_prediction": None,
        "prediction_inputs": {},
        "risk_low": 30,
        "risk_medium": 60,
        "risk_high": 80,
        "model_comparison": None,
        "training_done": False,
        "target_col": None,
        "positive_class_label": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return None


# =========================================================
# SCHEMA DETECTION
# =========================================================
@st.cache_data(show_spinner=False)
def detect_schema(df: pd.DataFrame) -> dict:
    """Inspect the dataframe and classify every column."""
    n_rows, n_cols = df.shape
    numeric_cols, categorical_cols, boolean_cols, date_cols, id_cols, constant_cols = [], [], [], [], [], []

    for col in df.columns:
        series = df[col]
        lower = col.lower()

        if series.nunique(dropna=True) <= 1:
            constant_cols.append(col)

        if any(h in lower for h in ID_HINTS) or (series.nunique() == n_rows and series.dtype == object):
            id_cols.append(col)
            continue

        if "date" in lower or "_dt" in lower or lower.endswith("_at"):
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() > 0.8:
                date_cols.append(col)
                continue

        if series.dropna().isin([0, 1]).all() and series.nunique() <= 2 and pd.api.types.is_numeric_dtype(series):
            boolean_cols.append(col)
            continue

        if pd.api.types.is_bool_dtype(series):
            boolean_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            uniq = series.dropna().astype(str).str.lower().unique()
            if set(uniq).issubset({"yes", "no"}) or set(uniq).issubset({"true", "false"}) or set(uniq).issubset({"y", "n"}):
                boolean_cols.append(col)
            else:
                categorical_cols.append(col)

    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "boolean_cols": boolean_cols,
        "date_cols": date_cols,
        "id_cols": id_cols,
        "constant_cols": constant_cols,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
    }


# =========================================================
# TARGET DETECTION
# =========================================================
def detect_target_candidates(df: pd.DataFrame) -> list:
    candidates = []
    for col in df.columns:
        lower = col.lower()
        if any(h == lower or h in lower for h in TARGET_HINTS):
            nunique = df[col].nunique(dropna=True)
            if 2 <= nunique <= 4 and df[col].notna().sum() > 10:
                candidates.append(col)
    return candidates


def normalize_target(series: pd.Series, positive_label: Optional[str] = None):
    """Map an arbitrary 2-class target to 0/1 where 1 = churn."""
    s = series.copy()
    original = s.copy()

    if pd.api.types.is_numeric_dtype(s) and set(s.dropna().unique()).issubset({0, 1}):
        y = s.astype(int)
        pos_label = 1
        return y, original, pos_label

    str_s = s.astype(str).str.strip().str.lower()
    positive_words = {"yes", "y", "true", "churn", "churned", "exited", "1"}
    negative_words = {"no", "n", "false", "no churn", "stayed", "retained", "0"}

    uniques = set(str_s.dropna().unique())

    if positive_label is not None:
        y = (str_s == str(positive_label).strip().lower()).astype(int)
        return y, original, positive_label

    if uniques.issubset(positive_words | negative_words) and len(uniques) == 2:
        y = str_s.apply(lambda v: 1 if v in positive_words else 0)
        pos_label = [v for v in original.astype(str).unique()
                     if v.strip().lower() in positive_words]
        pos_label = pos_label[0] if pos_label else None
        return y, original, pos_label

    return None, original, None


# =========================================================
# LEAKAGE DETECTION
# =========================================================
def detect_leakage_columns(df: pd.DataFrame, target_col: str, id_cols: list, y: pd.Series) -> list:
    leakage = set(id_cols)
    for col in df.columns:
        if col == target_col:
            continue
        lower = col.lower()
        if any(h in lower for h in LEAKAGE_NAME_HINTS):
            leakage.add(col)
            continue
        # perfect / near-perfect relationship with target -> leakage
        try:
            if df[col].nunique(dropna=True) <= 6:
                cross = pd.crosstab(df[col], y)
                row_max = cross.max(axis=1)
                row_sum = cross.sum(axis=1)
                purity = (row_max / row_sum).mean()
                if purity > 0.98 and df[col].nunique() > 1:
                    leakage.add(col)
        except Exception:
            pass
    return sorted(leakage)

# =========================================================
# DATE FEATURE ENGINEERING
# =========================================================
def engineer_date_features(df: pd.DataFrame, date_cols: list) -> pd.DataFrame:
    out = df.copy()
    for col in date_cols:
        parsed = pd.to_datetime(out[col], errors="coerce")
        out[f"{col}_year"] = parsed.dt.year
        out[f"{col}_month"] = parsed.dt.month
        out[f"{col}_day"] = parsed.dt.day
        out[f"{col}_day_of_week"] = parsed.dt.dayofweek
        reference = parsed.max()
        out[f"{col}_days_since"] = (reference - parsed).dt.days
        out = out.drop(columns=[col])
    return out


# =========================================================
# FEATURE / TARGET PREPARATION
# =========================================================
def prepare_features(df: pd.DataFrame, schema: dict, target_col: str, leakage_cols: list):
    exclude = set(leakage_cols) | {target_col}
    numeric_features = [c for c in schema["numeric_cols"] if c not in exclude and c not in schema["constant_cols"]]
    categorical_features = [c for c in schema["categorical_cols"] if c not in exclude and c not in schema["constant_cols"]]
    boolean_features = [c for c in schema["boolean_cols"] if c not in exclude and c not in schema["constant_cols"]]
    date_features = [c for c in schema["date_cols"] if c not in exclude]

    working = df.copy()
    if date_features:
        working = engineer_date_features(working, date_features)
        for c in date_features:
            derived = [f"{c}_year", f"{c}_month", f"{c}_day", f"{c}_day_of_week", f"{c}_days_since"]
            numeric_features.extend([d for d in derived if d in working.columns])

    # Normalize boolean-like text columns (Yes/No) into 0/1 numeric
    for c in boolean_features:
        col = working[c]
        if not pd.api.types.is_numeric_dtype(col):
            str_col = col.astype(str).str.strip().str.lower()
            working[c] = str_col.map({"yes": 1, "y": 1, "true": 1, "no": 0, "n": 0, "false": 0}).fillna(0)
        numeric_features.append(c)

    feature_cols = list(dict.fromkeys(numeric_features + categorical_features))
    X = working[feature_cols].copy()
    return X, feature_cols, numeric_features, categorical_features, working


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))
    return ColumnTransformer(transformers=transformers)


# =========================================================
# MODEL REGISTRY
# =========================================================
def get_available_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
    }
    if XGBClassifier is not None:
        try:
            models["XGBoost"] = XGBClassifier(
                n_estimators=300, random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0,
            )
        except Exception:
            pass
    if LGBMClassifier is not None:
        try:
            models["LightGBM"] = LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE, verbosity=-1)
        except Exception:
            pass
    if CatBoostClassifier is not None:
        try:
            models["CatBoost"] = CatBoostClassifier(
                iterations=300, random_state=RANDOM_STATE, verbose=False
            )
        except Exception:
            pass
    return models


def unavailable_models() -> list:
    available = get_available_models()
    optional = ["XGBoost", "LightGBM", "CatBoost"]
    return [name for name in optional if name not in available]

# =========================================================
# TRAINING & EVALUATION
# =========================================================
@st.cache_resource(show_spinner=False)
def train_models(
    _X: pd.DataFrame,
    _y: pd.Series,
    numeric_features: tuple,
    categorical_features: tuple,
    model_names: tuple,
    data_signature: str,
):
    """Train the requested models and return fitted pipelines + metrics."""
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)

    can_stratify = _y.value_counts().min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.20, random_state=RANDOM_STATE,
        stratify=_y if can_stratify else None,
    )

    all_models = get_available_models()
    results = {}
    fitted_pipelines = {}

    for name in model_names:
        if name not in all_models:
            continue
        model = all_models[name]
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        start = time.time()
        try:
            pipe.fit(X_train, y_train)
        except Exception as e:
            results[name] = {"status": f"Failed: {e}"}
            continue
        train_time = time.time() - start

        y_pred = pipe.predict(X_test)
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
            "PR-AUC": average_precision_score(y_test, y_proba) if y_proba is not None else np.nan,
            "Training Time": train_time,
            "Status": "Trained",
        }
        results[name] = metrics
        fitted_pipelines[name] = {
            "pipeline": pipe,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "X_test": X_test,
        }

    return fitted_pipelines, results, (X_train, X_test, y_train, y_test)


def select_best_model(metrics: dict) -> Optional[str]:
    """Select the best model using F1 primary, PR-AUC/Recall/ROC-AUC secondary."""
    valid = {k: v for k, v in metrics.items() if v.get("Status") == "Trained"}
    if not valid:
        return None
    ranked = sorted(
        valid.items(),
        key=lambda kv: (
            kv[1].get("F1", 0),
            kv[1].get("PR-AUC", 0) if not pd.isna(kv[1].get("PR-AUC", np.nan)) else 0,
            kv[1].get("Recall", 0),
            kv[1].get("ROC-AUC", 0) if not pd.isna(kv[1].get("ROC-AUC", np.nan)) else 0,
        ),
        reverse=True,
    )
    return ranked[0][0]


def risk_level(prob_pct: float) -> str:
    if prob_pct < st.session_state.risk_low:
        return "LOW RISK"
    elif prob_pct < st.session_state.risk_medium:
        return "MEDIUM RISK"
    elif prob_pct < st.session_state.risk_high:
        return "HIGH RISK"
    return "VERY HIGH RISK"


def risk_pill_class(level: str) -> str:
    return {
        "LOW RISK": "cc-pill-low",
        "MEDIUM RISK": "cc-pill-medium",
        "HIGH RISK": "cc-pill-high",
        "VERY HIGH RISK": "cc-pill-veryhigh",
    }.get(level, "cc-pill-low")

# =========================================================
# SIDEBAR
# =========================================================
def render_sidebar(schema: dict, model_names_available: list) -> None:
    with st.sidebar:
        st.markdown(
            "<div style='padding:0.4rem 0 0.8rem 0;'>"
            "<div style='font-weight:800; font-size:1.15rem; letter-spacing:1px;'>CUSTOMER CHURN AI</div>"
            "<div style='color:var(--subtext); font-size:0.8rem;'>Risk &amp; Retention Intelligence</div>"
            "</div>", unsafe_allow_html=True
        )
        st.divider()

        pages = [
            "Business Overview", "Dashboard", "Analytics", "Customers", "Predict Churn",
            "Model Performance", "Explainability", "Retention Intelligence",
            "Data Explorer", "About Us",
        ]
        icons = {
            "Business Overview": "🏢", "Dashboard": "📊", "Analytics": "📈", "Customers": "👥",
            "Predict Churn": "🎯", "Model Performance": "🤖", "Explainability": "🔍",
            "Retention Intelligence": "💼", "Data Explorer": "📁", "About Us": "ℹ️",
        }
        labels = [f"{icons[p]}  {p}" for p in pages]
        current_idx = pages.index(st.session_state.selected_page) if st.session_state.selected_page in pages else 0
        choice = st.radio("Navigation", labels, index=current_idx, label_visibility="collapsed")
        st.session_state.selected_page = pages[labels.index(choice)]

        st.divider()
        st.markdown("**Theme**")
        theme_options = ["Light", "Dark", "Cream Vanilla", "Bubbles"]
        current_theme = st.session_state.theme if st.session_state.theme in theme_options else "Light"
        st.session_state.theme = st.radio("Theme", theme_options,
                                           index=theme_options.index(current_theme),
                                           label_visibility="collapsed")

        st.divider()
        st.markdown("**ML Model**")
        if model_names_available:
            sel = st.selectbox("Selected Model", model_names_available,
                                index=model_names_available.index(st.session_state.selected_model_name)
                                if st.session_state.selected_model_name in model_names_available else 0)
            st.session_state.selected_model_name = sel

        train_mode = st.radio("Training Mode", ["Train Selected Model", "Train All Models"], label_visibility="collapsed")
        train_clicked = st.button("🚀 Train", use_container_width=True)

        st.divider()
        st.markdown("**Risk Thresholds (%)**")
        st.session_state.risk_low = st.slider("Low boundary", 0, 50, st.session_state.risk_low)
        st.session_state.risk_medium = st.slider("Medium boundary", st.session_state.risk_low + 1, 90, max(st.session_state.risk_medium, st.session_state.risk_low + 1))
        st.session_state.risk_high = st.slider("High boundary", st.session_state.risk_medium + 1, 99, max(st.session_state.risk_high, st.session_state.risk_medium + 1))

        st.divider()
        st.markdown("**System Status**")
        st.markdown("🟢 Dataset Connected" if schema else "🔴 Dataset Missing")
        st.markdown("🟢 Model Ready" if st.session_state.training_done else "🟡 Model Not Trained")

        st.session_state["_train_mode"] = train_mode
        st.session_state["_train_clicked"] = train_clicked


def render_header() -> None:
    st.markdown(
        """
        <div class="cc-header">
            <div class="cc-title">CUSTOMER CHURN AI</div>
            <div class="cc-subtitle">Customer Risk Prediction &amp; Retention Intelligence</div>
            <div class="cc-tag">DATA SCIENCE • MACHINE LEARNING • BUSINESS INTELLIGENCE</div>
        </div>
        """, unsafe_allow_html=True
    )

# =========================================================
# HELPER: FIND COLUMN BY LOOSE NAME
# =========================================================
def find_col(df: pd.DataFrame, *keywords) -> Optional[str]:
    for col in df.columns:
        lower = col.lower()
        if all(kw in lower for kw in keywords):
            return col
    return None


def metric_card(label: str, value: str) -> str:
    return f"""<div class="cc-card">
        <div class="cc-metric-label">{label}</div>
        <div class="cc-metric-value">{value}</div>
    </div>"""


# =========================================================
# PAGE: DASHBOARD
# =========================================================
def page_dashboard(df: pd.DataFrame, schema: dict, y_full: pd.Series) -> None:
    st.markdown('<div class="cc-section-title">Executive Overview</div>', unsafe_allow_html=True)

    filtered = df.copy()
    filter_mask = pd.Series(True, index=df.index)

    with st.expander("🔎 Dashboard Filters", expanded=False):
        cols = st.columns(3)
        cat_candidates = [c for c in schema["categorical_cols"] if df[c].nunique() <= 25][:6]
        for i, c in enumerate(cat_candidates):
            with cols[i % 3]:
                options = sorted(df[c].dropna().astype(str).unique().tolist())
                selected = st.multiselect(c, options, default=[])
                if selected:
                    filter_mask &= df[c].astype(str).isin(selected)

        num_candidates = [c for c in schema["numeric_cols"]][:3]
        for i, c in enumerate(num_candidates):
            with cols[i % 3]:
                lo, hi = float(df[c].min()), float(df[c].max())
                if lo < hi:
                    rng = st.slider(c, lo, hi, (lo, hi))
                    filter_mask &= df[c].between(rng[0], rng[1])

    filtered = df[filter_mask]
    y_filtered = y_full[filter_mask] if y_full is not None else None

    total_customers = len(filtered)
    churned = int(y_filtered.sum()) if y_filtered is not None else None
    churn_rate = f"{(churned / total_customers * 100):.1f}%" if churned is not None and total_customers else "Not Available"

    tenure_col = find_col(df, "tenure")
    monthly_col = find_col(df, "monthly", "charge")
    total_charge_col = find_col(df, "total", "charge")

    avg_tenure = f"{filtered[tenure_col].mean():.1f}" if tenure_col else "Not Available"
    avg_monthly = f"{filtered[monthly_col].mean():.2f}" if monthly_col else "Not Available"
    avg_total = f"{filtered[total_charge_col].mean():.2f}" if total_charge_col else "Not Available"

    high_risk = "Not Available"
    model_acc = "Not Available"
    if st.session_state.training_done and st.session_state.model_metrics:
        best = st.session_state.best_model_name
        if best and best in st.session_state.model_metrics:
            model_acc = f"{st.session_state.model_metrics[best]['Accuracy']*100:.1f}%"

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Total Customers", f"{total_customers:,}"), unsafe_allow_html=True)
    c2.markdown(metric_card("Churned Customers", f"{churned:,}" if churned is not None else "Not Available"), unsafe_allow_html=True)
    c3.markdown(metric_card("Churn Rate", churn_rate), unsafe_allow_html=True)
    c4.markdown(metric_card("Model Accuracy", model_acc), unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    c5.markdown(metric_card("Average Tenure", avg_tenure), unsafe_allow_html=True)
    c6.markdown(metric_card("Average Monthly Charges", avg_monthly), unsafe_allow_html=True)
    c7.markdown(metric_card("Average Total Charges", avg_total), unsafe_allow_html=True)
    c8.markdown(metric_card("High-Risk Customers", high_risk), unsafe_allow_html=True)

    st.markdown('<div class="cc-section-title">Churn Overview</div>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    if y_filtered is not None and len(filtered):
        with ch1:
            counts = y_filtered.value_counts().rename({0: "Stayed", 1: "Churned"})
            fig = px.pie(values=counts.values, names=counts.index, hole=0.55,
                         color=counts.index, color_discrete_map={"Stayed": "#33d69f", "Churned": "#ff6b6b"},
                         title="Churn Distribution")
            st.plotly_chart(style_fig(fig), use_container_width=True)

        cat_for_rate = cat_candidates[0] if cat_candidates else (schema["categorical_cols"][0] if schema["categorical_cols"] else None)
        if cat_for_rate:
            with ch2:
                tmp = filtered[[cat_for_rate]].copy()
                tmp["churn"] = y_filtered.values
                rate = tmp.groupby(cat_for_rate)["churn"].mean().sort_values(ascending=False) * 100
                fig2 = px.bar(x=rate.index.astype(str), y=rate.values,
                              labels={"x": cat_for_rate, "y": "Churn Rate (%)"},
                              title=f"Churn Rate by {cat_for_rate}")
                st.plotly_chart(style_fig(fig2), use_container_width=True)
    else:
        st.info("Churn target not available for this view.")

    ch3, ch4 = st.columns(2)
    contract_col = find_col(df, "contract")
    if contract_col and y_filtered is not None:
        with ch3:
            tmp = filtered[[contract_col]].copy()
            tmp["churn"] = y_filtered.values
            rate = tmp.groupby(contract_col)["churn"].mean().sort_values(ascending=False) * 100
            fig3 = px.bar(x=rate.index.astype(str), y=rate.values,
                          labels={"x": contract_col, "y": "Churn Rate (%)"}, title="Churn Rate by Contract")
            st.plotly_chart(style_fig(fig3), use_container_width=True)

    if tenure_col:
        with ch4:
            fig4 = px.histogram(filtered, x=tenure_col, color=y_filtered.map({0: "Stayed", 1: "Churned"}) if y_filtered is not None else None,
                                 nbins=30, title=f"{tenure_col} Distribution",
                                 color_discrete_map={"Stayed": "#33d69f", "Churned": "#ff6b6b"})
            st.plotly_chart(style_fig(fig4), use_container_width=True)

    num_for_corr = schema["numeric_cols"][:15]
    if len(num_for_corr) >= 2:
        st.markdown('<div class="cc-section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
        corr = filtered[num_for_corr].corr()
        fig5 = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(style_fig(fig5), use_container_width=True)

# =========================================================
# PAGE: ANALYTICS
# =========================================================
def page_analytics(df: pd.DataFrame, schema: dict, y_full: pd.Series) -> None:
    st.markdown('<div class="cc-section-title">Customer Overview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card("Rows", f"{schema['n_rows']:,}"), unsafe_allow_html=True)
    c2.markdown(metric_card("Numerical Features", f"{len(schema['numeric_cols'])}"), unsafe_allow_html=True)
    c3.markdown(metric_card("Categorical Features", f"{len(schema['categorical_cols'])}"), unsafe_allow_html=True)

    tabs = st.tabs(["Churn Behaviour", "Numerical Analysis", "Categorical Analysis", "Correlation", "Segment Analysis", "Business Insights"])

    with tabs[0]:
        if y_full is not None:
            rate = y_full.mean() * 100
            st.markdown(metric_card("Overall Churn Rate", f"{rate:.1f}%"), unsafe_allow_html=True)
            fig = px.pie(values=y_full.value_counts().values,
                         names=y_full.value_counts().index.map({0: "Stayed", 1: "Churned"}),
                         hole=0.5, title="Churn Behaviour")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.info("Churn target not detected.")

    with tabs[1]:
        num_col = st.selectbox("Select numerical feature", schema["numeric_cols"]) if schema["numeric_cols"] else None
        if num_col:
            fig = px.histogram(df, x=num_col, marginal="box", title=f"Distribution of {num_col}")
            st.plotly_chart(style_fig(fig), use_container_width=True)
            if y_full is not None:
                tmp = df[[num_col]].copy()
                tmp["Churn"] = y_full.map({0: "Stayed", 1: "Churned"}).values
                fig2 = px.box(tmp, x="Churn", y=num_col, color="Churn",
                              color_discrete_map={"Stayed": "#33d69f", "Churned": "#ff6b6b"},
                              title=f"{num_col} by Churn Status")
                st.plotly_chart(style_fig(fig2), use_container_width=True)
        else:
            st.info("No numerical columns available.")

    with tabs[2]:
        cat_col = st.selectbox("Select categorical feature", schema["categorical_cols"]) if schema["categorical_cols"] else None
        if cat_col:
            counts = df[cat_col].value_counts().head(20)
            fig = px.bar(x=counts.index.astype(str), y=counts.values, title=f"Distribution of {cat_col}")
            st.plotly_chart(style_fig(fig), use_container_width=True)
            if y_full is not None:
                tmp = df[[cat_col]].copy()
                tmp["churn"] = y_full.values
                rate = tmp.groupby(cat_col)["churn"].mean().sort_values(ascending=False) * 100
                fig2 = px.bar(x=rate.index.astype(str), y=rate.values, title=f"Churn Rate by {cat_col}",
                              labels={"x": cat_col, "y": "Churn Rate (%)"})
                st.plotly_chart(style_fig(fig2), use_container_width=True)
        else:
            st.info("No categorical columns available.")

    with tabs[3]:
        if len(schema["numeric_cols"]) >= 2:
            method = st.radio("Correlation Method", ["pearson", "spearman"], horizontal=True)
            corr = df[schema["numeric_cols"][:20]].corr(method=method)
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.info("Correlation analysis requires at least two numerical columns.")

    with tabs[4]:
        seg_col = next((c for c in schema["categorical_cols"] if df[c].nunique() <= 15), None)
        if seg_col and y_full is not None:
            tmp = df[[seg_col]].copy()
            tmp["churn"] = y_full.values
            seg = tmp.groupby(seg_col)["churn"].agg(["mean", "count"])
            seg["mean"] = (seg["mean"] * 100).round(1)
            seg.columns = ["Churn Rate (%)", "Customer Count"]
            st.dataframe(seg.sort_values("Churn Rate (%)", ascending=False), use_container_width=True)
        else:
            st.info("No suitable segment column detected.")

    with tabs[5]:
        render_business_insights(df, schema, y_full)


def render_business_insights(df: pd.DataFrame, schema: dict, y_full: pd.Series) -> None:
    if y_full is None:
        st.info("Business insights require a detected churn target.")
        return
    insights = []
    for c in schema["categorical_cols"]:
        if df[c].nunique() < 2 or df[c].nunique() > 15:
            continue
        tmp = df[[c]].copy()
        tmp["churn"] = y_full.values
        rate = tmp.groupby(c)["churn"].mean()
        if len(rate) < 2:
            continue
        top_cat, top_val = rate.idxmax(), rate.max()
        low_cat, low_val = rate.idxmin(), rate.min()
        if top_val - low_val > 0.05:
            insights.append(
                f"Customers with **{c} = {top_cat}** show a higher observed churn rate "
                f"({top_val*100:.1f}%) compared to **{c} = {low_cat}** ({low_val*100:.1f}%)."
            )
    if insights:
        for i in insights[:8]:
            st.markdown(f"- {i}")
        st.caption("These are observed associations in the data, not proof of causation.")
    else:
        st.info("No strong categorical patterns detected in this dataset.")


# =========================================================
# PAGE: CUSTOMERS
# =========================================================
def page_customers(df: pd.DataFrame, schema: dict, y_full: pd.Series) -> None:
    st.markdown('<div class="cc-section-title">Customer Explorer</div>', unsafe_allow_html=True)

    id_col = schema["id_cols"][0] if schema["id_cols"] else None
    search = st.text_input("Search (ID or any text field)", "")

    view = df.copy()
    if y_full is not None:
        view["Churn Status"] = y_full.map({0: "Stayed", 1: "Churned"}).values

    if search:
        mask = pd.Series(False, index=view.index)
        for c in view.columns:
            try:
                mask |= view[c].astype(str).str.contains(search, case=False, na=False)
            except Exception:
                continue
        view = view[mask]

    sort_col = st.selectbox("Sort by", view.columns.tolist(), index=0)
    ascending = st.checkbox("Ascending", value=False)
    view = view.sort_values(sort_col, ascending=ascending)

    page_size = 25
    total_pages = max(1, (len(view) - 1) // page_size + 1)
    page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start = (page_num - 1) * page_size
    st.dataframe(view.iloc[start:start + page_size], use_container_width=True)
    st.caption(f"Showing page {page_num} of {total_pages} — {len(view):,} matching customers")

    if id_col:
        st.markdown('<div class="cc-section-title">Customer Profile</div>', unsafe_allow_html=True)
        chosen_id = st.selectbox("Select a Customer ID", view[id_col].astype(str).unique().tolist()[:500])
        if chosen_id:
            record = df[df[id_col].astype(str) == chosen_id]
            if not record.empty:
                st.dataframe(record.T.rename(columns={record.index[0]: "Value"}), use_container_width=True)

# =========================================================
# PAGE: PREDICT CHURN
# =========================================================
FIELD_GROUPS = {
    "CUSTOMER PROFILE": ["gender", "age", "married", "senior", "dependents", "state", "partner"],
    "SERVICE INFORMATION": ["phone", "internet", "security", "backup", "protection", "support", "streaming", "multiple_lines", "unlimited"],
    "SUBSCRIPTION INFORMATION": ["contract", "tenure", "value_deal", "referral"],
    "BILLING INFORMATION": ["charge", "payment", "billing", "refund", "revenue"],
    "CUSTOMER BEHAVIOUR": ["service_count", "has_", "is_", "count"],
    "ADDITIONAL INFORMATION": [],
}


def assign_group(col: str) -> str:
    lower = col.lower()
    for group, keywords in FIELD_GROUPS.items():
        if group == "ADDITIONAL INFORMATION":
            continue
        if any(kw in lower for kw in keywords):
            return group
    return "ADDITIONAL INFORMATION"


def page_predict(df: pd.DataFrame, feature_cols: list, numeric_features: list, categorical_features: list) -> None:
    st.markdown('<div class="cc-section-title">🎯 Predict Customer Churn</div>', unsafe_allow_html=True)

    if not st.session_state.training_done or not st.session_state.trained_models:
        st.warning("Train a model first from the sidebar (Train Selected Model / Train All Models).")
        return

    model_name = st.session_state.selected_model_name or st.session_state.best_model_name
    if model_name not in st.session_state.trained_models:
        st.warning("Selected model is not trained yet. Please train it from the sidebar.")
        return

    pipeline_info = st.session_state.trained_models[model_name]
    pipeline = pipeline_info["pipeline"]

    grouped = {}
    for c in feature_cols:
        grouped.setdefault(assign_group(c), []).append(c)

    inputs = {}
    with st.form("prediction_form"):
        for group_name, cols in grouped.items():
            if not cols:
                continue
            st.markdown(f"**{group_name}**")
            ui_cols = st.columns(3)
            for i, c in enumerate(cols):
                target_col = ui_cols[i % 3]
                with target_col:
                    if c in numeric_features:
                        series = df[c].dropna()
                        default = float(series.median()) if len(series) else 0.0
                        inputs[c] = st.number_input(
                            c, value=default,
                            min_value=float(series.min()) if len(series) else None,
                            max_value=float(series.max()) if len(series) else None,
                        )
                    elif c in categorical_features:
                        options = sorted(df[c].dropna().astype(str).unique().tolist())
                        inputs[c] = st.selectbox(c, options) if options else st.text_input(c, "")
                    else:
                        inputs[c] = st.text_input(c, "")
            st.markdown("")

        submitted = st.form_submit_button("🔮 PREDICT CUSTOMER CHURN", use_container_width=True)

    if submitted:
        st.session_state.prediction_inputs = inputs
        input_df = pd.DataFrame([inputs])[feature_cols]
        try:
            proba = pipeline.predict_proba(input_df)[0][1]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        prob_pct = proba * 100
        level = risk_level(prob_pct)
        st.session_state.last_prediction = {
            "probability": prob_pct, "risk_level": level, "model": model_name, "inputs": inputs,
        }

    if st.session_state.last_prediction:
        render_prediction_result(df, feature_cols, numeric_features)


def render_prediction_result(df: pd.DataFrame, feature_cols: list, numeric_features: list) -> None:
    pred = st.session_state.last_prediction
    prob_pct = pred["probability"]
    level = pred["risk_level"]

    st.markdown('<div class="cc-section-title">Customer Risk Assessment</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="cc-card" style="text-align:center;">
            <div class="cc-metric-label">Churn Probability</div>
            <div class="cc-metric-value">{prob_pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="cc-card" style="text-align:center;">
            <div class="cc-metric-label">Risk Level</div>
            <span class="cc-pill {risk_pill_class(level)}">{level}</span>
        </div>""", unsafe_allow_html=True)
    with c3:
        verdict = "LIKELY TO CHURN" if prob_pct >= 50 else "LIKELY TO STAY"
        st.markdown(f"""<div class="cc-card" style="text-align:center;">
            <div class="cc-metric-label">Model: {pred['model']}</div>
            <div class="cc-metric-value" style="font-size:1.1rem;">{verdict}</div>
        </div>""", unsafe_allow_html=True)

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        title={"text": "Churn Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#3355ff"},
            "steps": [
                {"range": [0, st.session_state.risk_low], "color": "rgba(14,164,114,0.25)"},
                {"range": [st.session_state.risk_low, st.session_state.risk_medium], "color": "rgba(181,123,0,0.25)"},
                {"range": [st.session_state.risk_medium, st.session_state.risk_high], "color": "rgba(214,66,58,0.25)"},
                {"range": [st.session_state.risk_high, 100], "color": "rgba(214,66,58,0.5)"},
            ],
        },
    ))
    st.plotly_chart(style_fig(fig), use_container_width=True)

    # Risk band
    st.markdown(f"""
    <div style="display:flex; border-radius:10px; overflow:hidden; margin-bottom:1rem;">
        <div style="flex:{st.session_state.risk_low}; background:rgba(14,164,114,0.35); text-align:center; padding:6px; font-size:0.75rem;">LOW</div>
        <div style="flex:{st.session_state.risk_medium-st.session_state.risk_low}; background:rgba(181,123,0,0.35); text-align:center; padding:6px; font-size:0.75rem;">MEDIUM</div>
        <div style="flex:{st.session_state.risk_high-st.session_state.risk_medium}; background:rgba(214,66,58,0.35); text-align:center; padding:6px; font-size:0.75rem;">HIGH</div>
        <div style="flex:{100-st.session_state.risk_high}; background:rgba(214,66,58,0.6); text-align:center; padding:6px; font-size:0.75rem;">VERY HIGH</div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset benchmark
    st.markdown('<div class="cc-section-title">Dataset Benchmark</div>', unsafe_allow_html=True)
    y_full = st.session_state.get("_y_full")
    if y_full is not None:
        dataset_rate = y_full.mean() * 100
        diff = prob_pct - dataset_rate
        b1, b2, b3 = st.columns(3)
        b1.markdown(metric_card("Customer Probability", f"{prob_pct:.1f}%"), unsafe_allow_html=True)
        b2.markdown(metric_card("Dataset Churn Rate", f"{dataset_rate:.1f}%"), unsafe_allow_html=True)
        b3.markdown(metric_card("Difference", f"{diff:+.1f} pp"), unsafe_allow_html=True)

    # Customer vs dataset average
    st.markdown('<div class="cc-section-title">Customer vs Dataset Average</div>', unsafe_allow_html=True)
    inputs = pred["inputs"]
    comp_rows = []
    for c in numeric_features[:8]:
        if c in inputs:
            comp_rows.append({"Feature": c, "Customer": inputs[c], "Dataset Average": round(df[c].mean(), 2)})
    if comp_rows:
        comp_df = pd.DataFrame(comp_rows)
        fig_cmp = px.bar(comp_df.melt(id_vars="Feature", var_name="Type", value_name="Value"),
                          x="Feature", y="Value", color="Type", barmode="group",
                          title="Customer vs Dataset Average")
        st.plotly_chart(style_fig(fig_cmp), use_container_width=True)

    render_prediction_explanation(df, feature_cols, pred)
    render_retention_block(level)


def render_prediction_explanation(df: pd.DataFrame, feature_cols: list, pred: dict) -> None:
    st.markdown('<div class="cc-section-title">🔍 Why Did the Model Predict This?</div>', unsafe_allow_html=True)
    model_name = pred["model"]
    pipeline_info = st.session_state.trained_models.get(model_name)
    if not pipeline_info:
        st.info("Explanation unavailable.")
        return
    pipeline = pipeline_info["pipeline"]
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = None

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is not None and feature_names is not None:
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=False).head(10)
        fig = px.bar(imp_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                     title="Top Contributors to This Prediction")
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.caption("These features contributed to the model prediction. This does not imply the customer will churn because of these features alone.")
    else:
        st.info("Feature-level explanation is not available for this model type.")


def render_retention_block(level: str) -> None:
    st.markdown('<div class="cc-section-title">💼 Retention Recommendation</div>', unsafe_allow_html=True)
    recs = {
        "LOW RISK": ("LOW", ["Maintain customer relationship.", "Consider loyalty engagement.", "Explore cross-sell opportunities."]),
        "MEDIUM RISK": ("MEDIUM", ["Proactive engagement.", "Satisfaction follow-up.", "Service review."]),
        "HIGH RISK": ("HIGH", ["Targeted retention campaign.", "Review plan/pricing.", "Review service quality."]),
        "VERY HIGH RISK": ("CRITICAL", ["Immediate customer outreach.", "Personalized retention offer.", "Priority support."]),
    }
    priority, actions = recs.get(level, ("LOW", []))
    st.markdown(f"<span class='cc-pill {risk_pill_class(level)}'>Priority: {priority}</span>", unsafe_allow_html=True)
    for a in actions:
        st.markdown(f"- {a}")
    st.caption("These are recommendations based on the model's risk output, not guarantees of outcome.")

# =========================================================
# PAGE: MODEL PERFORMANCE
# =========================================================
def page_model_performance(schema: dict) -> None:
    st.markdown('<div class="cc-section-title">🤖 Model Performance</div>', unsafe_allow_html=True)

    if not st.session_state.training_done or not st.session_state.model_metrics:
        st.warning("No models trained yet. Use the sidebar to train a model.")
        return

    metrics = st.session_state.model_metrics
    best = st.session_state.best_model_name

    rows = []
    for name, m in metrics.items():
        if m.get("Status") != "Trained":
            rows.append({"Model": name, "Status": m.get("Status", "Unavailable")})
            continue
        rows.append({
            "Model": name,
            "Accuracy": round(m["Accuracy"], 4),
            "Precision": round(m["Precision"], 4),
            "Recall": round(m["Recall"], 4),
            "F1": round(m["F1"], 4),
            "ROC-AUC": round(m["ROC-AUC"], 4) if not pd.isna(m["ROC-AUC"]) else None,
            "PR-AUC": round(m["PR-AUC"], 4) if not pd.isna(m["PR-AUC"]) else None,
            "Training Time (s)": round(m["Training Time"], 3),
            "Status": "★ Selected" if name == best else "Trained",
        })
    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True)
    st.caption("Best model according to the configured evaluation priority: F1 → PR-AUC → Recall → ROC-AUC.")

    unavailable = unavailable_models()
    if unavailable:
        st.info("Not installed (skipped automatically): " + ", ".join(unavailable))

    model_choice = st.selectbox("Inspect a trained model", list(st.session_state.trained_models.keys()),
                                 index=list(st.session_state.trained_models.keys()).index(best) if best in st.session_state.trained_models else 0)
    info = st.session_state.trained_models[model_choice]
    y_test, y_pred, y_proba = info["y_test"], info["y_pred"], info["y_proba"]

    c1, c2 = st.columns(2)
    with c1:
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, x=["No Churn", "Churn"], y=["No Churn", "Churn"],
                         labels=dict(x="Predicted", y="Actual"), title="Confusion Matrix", color_continuous_scale="Blues")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with c2:
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={auc:.3f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="Random"))
            fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.info("ROC curve unavailable (model does not output probabilities).")

    c3, c4 = st.columns(2)
    with c3:
        if y_proba is not None:
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rec, y=prec, name=f"PR (AUC={ap:.3f})"))
            fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.info("PR curve unavailable.")

    with c4:
        model_obj = info["pipeline"].named_steps["model"]
        preprocessor = info["pipeline"].named_steps["preprocessor"]
        try:
            names = preprocessor.get_feature_names_out()
        except Exception:
            names = None
        if names is not None:
            if hasattr(model_obj, "feature_importances_"):
                imp = model_obj.feature_importances_
            elif hasattr(model_obj, "coef_"):
                imp = np.abs(model_obj.coef_[0])
            else:
                imp = None
            if imp is not None:
                imp_df = pd.DataFrame({"Feature": names, "Importance": imp}).sort_values("Importance", ascending=False).head(15)
                fig = px.bar(imp_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Feature Importance")
                st.plotly_chart(style_fig(fig), use_container_width=True)
            else:
                st.info("Feature importance not available for this model.")

    with st.expander("Classification Report"):
        st.text(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    with st.expander("Cross-Validation (optional)"):
        run_cv = st.checkbox("Run Stratified 5-Fold Cross-Validation", value=False)
        if run_cv:
            X_full = st.session_state.get("_X_full")
            y_full = st.session_state.get("_y_full")
            if X_full is not None:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                scoring = ["f1", "roc_auc", "average_precision"]
                cv_res = cross_validate(info["pipeline"], X_full, y_full, cv=skf, scoring=scoring)
                st.write({
                    "F1 (mean ± std)": f"{cv_res['test_f1'].mean():.3f} ± {cv_res['test_f1'].std():.3f}",
                    "ROC-AUC (mean ± std)": f"{cv_res['test_roc_auc'].mean():.3f} ± {cv_res['test_roc_auc'].std():.3f}",
                    "PR-AUC (mean ± std)": f"{cv_res['test_average_precision'].mean():.3f} ± {cv_res['test_average_precision'].std():.3f}",
                })


# =========================================================
# PAGE: EXPLAINABILITY (GLOBAL)
# =========================================================
def page_explainability(df: pd.DataFrame, feature_cols: list) -> None:
    st.markdown('<div class="cc-section-title">🔍 Explainability</div>', unsafe_allow_html=True)
    if not st.session_state.training_done:
        st.warning("Train a model first from the sidebar.")
        return

    model_name = st.session_state.selected_model_name or st.session_state.best_model_name
    info = st.session_state.trained_models.get(model_name)
    if not info:
        st.warning("Selected model not trained.")
        return
    pipeline = info["pipeline"]
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    tabs = st.tabs(["Global Feature Importance", "SHAP Summary", "Top Churn Drivers", "Model Behaviour"])

    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = None

    with tabs[0]:
        if names is not None and hasattr(model, "feature_importances_"):
            imp_df = pd.DataFrame({"Feature": names, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False).head(20)
            fig = px.bar(imp_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Global Feature Importance")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        elif names is not None and hasattr(model, "coef_"):
            imp_df = pd.DataFrame({"Feature": names, "Importance": np.abs(model.coef_[0])}).sort_values("Importance", ascending=False).head(20)
            fig = px.bar(imp_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Global Feature Importance (|coefficient|)")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

    with tabs[1]:
        if SHAP_AVAILABLE:
            try:
                X_test_sample = info["X_test"].sample(min(200, len(info["X_test"])), random_state=RANDOM_STATE)
                X_trans = preprocessor.transform(X_test_sample)
                if hasattr(X_trans, "toarray"):
                    X_trans = X_trans.toarray()
                if hasattr(model, "feature_importances_"):
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.Explainer(model, X_trans)
                shap_values = explainer(X_trans)
                st.success("SHAP summary computed on a sample of the test set.")
                st.caption("SHAP values indicate each feature's contribution to individual predictions.")
            except Exception:
                st.info("SHAP could not be computed for this model. Showing model-native feature importance instead (see Global Feature Importance tab).")
        else:
            st.info("SHAP is not installed. Showing model-native feature importance instead (see Global Feature Importance tab).")

    with tabs[2]:
        st.markdown("Top Positive Churn Contributors and Top Negative Churn Contributors are derived from the model's global feature importance / coefficients.")
        if names is not None and (hasattr(model, "coef_")):
            coefs = model.coef_[0]
            pos = pd.DataFrame({"Feature": names, "Coefficient": coefs}).sort_values("Coefficient", ascending=False).head(10)
            neg = pd.DataFrame({"Feature": names, "Coefficient": coefs}).sort_values("Coefficient", ascending=True).head(10)
            cc1, cc2 = st.columns(2)
            with cc1:
                fig = px.bar(pos.sort_values("Coefficient"), x="Coefficient", y="Feature", orientation="h", title="Top Positive Churn Contributors")
                st.plotly_chart(style_fig(fig), use_container_width=True)
            with cc2:
                fig = px.bar(neg.sort_values("Coefficient"), x="Coefficient", y="Feature", orientation="h", title="Top Negative Churn Contributors")
                st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.info("Directional contributors require a linear model (Logistic Regression). Use Global Feature Importance for tree-based models.")

    with tabs[3]:
        st.markdown(f"**Selected Model:** {model_name}")
        st.markdown(f"**Model Type:** `{type(model).__name__}`")
        st.caption("These features contributed to the model prediction; this is not a causal claim.")

# =========================================================
# PAGE: RETENTION INTELLIGENCE
# =========================================================
def page_retention(df: pd.DataFrame, y_full: pd.Series) -> None:
    st.markdown('<div class="cc-section-title">💼 Retention Intelligence</div>', unsafe_allow_html=True)

    if not st.session_state.training_done:
        st.warning("Train a model first to unlock retention intelligence at scale.")
        return

    st.markdown("#### Retention Priority Framework")
    framework = pd.DataFrame([
        {"Risk Level": "LOW", "Priority": "LOW", "Recommended Action": "Maintain relationship, loyalty engagement, cross-sell."},
        {"Risk Level": "MEDIUM", "Priority": "MEDIUM", "Recommended Action": "Proactive engagement, satisfaction follow-up, service review."},
        {"Risk Level": "HIGH", "Priority": "HIGH", "Recommended Action": "Targeted retention campaign, pricing review, service quality review."},
        {"Risk Level": "VERY HIGH", "Priority": "CRITICAL", "Recommended Action": "Immediate outreach, personalized offer, priority support."},
    ])
    st.dataframe(framework, use_container_width=True)

    if st.session_state.last_prediction:
        level = st.session_state.last_prediction["risk_level"]
        st.markdown("#### Current Prediction Recommendation")
        render_retention_block(level)
    else:
        st.info("Run a prediction on the 'Predict Churn' page to see a customer-specific recommendation here.")

    st.caption("Recommendations are guidance derived from the model's risk output, not guarantees of business outcome.")


# =========================================================
# PAGE: DATA EXPLORER
# =========================================================
def page_data_explorer(df: pd.DataFrame, schema: dict, leakage_cols: list) -> None:
    st.markdown('<div class="cc-section-title">📁 Data Explorer</div>', unsafe_allow_html=True)

    st.markdown("#### Data Quality")
    dq_cols = st.columns(4)
    dq_cols[0].markdown(metric_card("Rows", f"{schema['n_rows']:,}"), unsafe_allow_html=True)
    dq_cols[1].markdown(metric_card("Columns", f"{schema['n_cols']:,}"), unsafe_allow_html=True)
    dq_cols[2].markdown(metric_card("Missing Values", f"{schema['missing_values']:,}"), unsafe_allow_html=True)
    dq_cols[3].markdown(metric_card("Duplicate Rows", f"{schema['duplicate_rows']:,}"), unsafe_allow_html=True)

    dq_cols2 = st.columns(4)
    dq_cols2[0].markdown(metric_card("Numerical Features", f"{len(schema['numeric_cols'])}"), unsafe_allow_html=True)
    dq_cols2[1].markdown(metric_card("Categorical Features", f"{len(schema['categorical_cols'])}"), unsafe_allow_html=True)
    dq_cols2[2].markdown(metric_card("Boolean Features", f"{len(schema['boolean_cols'])}"), unsafe_allow_html=True)
    dq_cols2[3].markdown(metric_card("Constant Columns", f"{len(schema['constant_cols'])}"), unsafe_allow_html=True)

    quality_score = 100
    quality_score -= min(30, schema["missing_values"] / max(1, schema["n_rows"] * schema["n_cols"]) * 100)
    quality_score -= min(20, schema["duplicate_rows"] / max(1, schema["n_rows"]) * 100)
    st.markdown(metric_card("Data Quality Rating", f"{quality_score:.1f} / 100"), unsafe_allow_html=True)

    if leakage_cols:
        st.markdown("#### Potential Leakage Columns (excluded from modeling)")
        st.warning(", ".join(leakage_cols))

    st.markdown("#### Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("#### Column Explorer")
    selected_cols = st.multiselect("Select columns", df.columns.tolist(), default=df.columns.tolist()[:8])
    if selected_cols:
        st.dataframe(df[selected_cols], use_container_width=True)

    st.markdown("#### Descriptive Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.markdown("#### Statistical Analysis")
    render_statistical_tests(df, schema)

    st.markdown("#### Downloads")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download Full CSV", df.to_csv(index=False).encode(), "full_dataset.csv", use_container_width=True)
    if selected_cols:
        d2.download_button("Download Filtered CSV", df[selected_cols].to_csv(index=False).encode(), "filtered_dataset.csv", use_container_width=True)
    d3.download_button("Download Statistics", df.describe(include="all").T.to_csv().encode(), "statistics.csv", use_container_width=True)


def render_statistical_tests(df: pd.DataFrame, schema: dict) -> None:
    y_full = st.session_state.get("_y_full")
    if y_full is None:
        st.info("Statistical tests require a detected churn target.")
        return

    results = []
    for c in schema["numeric_cols"][:10]:
        try:
            group0 = df.loc[y_full == 0, c].dropna()
            group1 = df.loc[y_full == 1, c].dropna()
            if len(group0) > 5 and len(group1) > 5:
                stat, p = stats.ttest_ind(group0, group1, equal_var=False)
                interp = "Statistically significant difference" if p < 0.05 else "No significant difference"
                results.append({"Test": "T-Test", "Variable": c, "Statistic": round(stat, 3), "p-value": round(p, 4), "Interpretation": interp})
        except Exception:
            continue

    for c in schema["categorical_cols"][:6]:
        try:
            if df[c].nunique() < 2 or df[c].nunique() > 12:
                continue
            table = pd.crosstab(df[c], y_full)
            chi2, p, _, _ = stats.chi2_contingency(table)
            interp = "Statistically significant association" if p < 0.05 else "No significant association"
            results.append({"Test": "Chi-Square", "Variable": c, "Statistic": round(chi2, 3), "p-value": round(p, 4), "Interpretation": interp})
        except Exception:
            continue

    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        st.caption("Statistical significance does not imply causation.")
    else:
        st.info("Not enough suitable data for statistical tests.")


# =========================================================
# PAGE: BUSINESS OVERVIEW
# =========================================================
def go_to_page(page_name: str) -> None:
    st.session_state.selected_page = page_name


def page_business_overview(df: pd.DataFrame, schema: dict, y_full: pd.Series) -> None:
    st.markdown('<div class="cc-section-title">🏢 Business Overview</div>', unsafe_allow_html=True)
    st.markdown(
        "A single control center summarizing what this application does for the business, "
        "and quick links into every capability that has been built."
    )

    total_customers = len(df)
    churn_rate = f"{y_full.mean()*100:.1f}%" if y_full is not None else "Not Available"
    models_trained = len(st.session_state.trained_models)
    best_model = st.session_state.best_model_name or "Not trained yet"

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(metric_card("Customers in Dataset", f"{total_customers:,}"), unsafe_allow_html=True)
    k2.markdown(metric_card("Overall Churn Rate", churn_rate), unsafe_allow_html=True)
    k3.markdown(metric_card("Models Trained", f"{models_trained}"), unsafe_allow_html=True)
    k4.markdown(metric_card("Best Model", best_model), unsafe_allow_html=True)

    st.markdown('<div class="cc-section-title">What This Application Delivers</div>', unsafe_allow_html=True)

    capabilities = [
        ("📊", "Executive Dashboard", "Live KPIs, churn distribution, and filterable business charts.", "Dashboard"),
        ("📈", "Deep Analytics", "Numerical, categorical, correlation and segment-level churn analysis.", "Analytics"),
        ("👥", "Customer Explorer", "Search, sort, and inspect individual customer profiles.", "Customers"),
        ("🎯", "Churn Prediction", "Score any customer in real time with a dynamic, model-backed form.", "Predict Churn"),
        ("🤖", "Model Performance", "Compare every trained model on Accuracy, F1, ROC-AUC, PR-AUC and more.", "Model Performance"),
        ("🔍", "Explainability", "See which features drive predictions — SHAP or native importance.", "Explainability"),
        ("💼", "Retention Intelligence", "Risk-tiered, actionable retention playbooks for the business.", "Retention Intelligence"),
        ("📁", "Data Explorer", "Data quality, statistics, leakage audit and CSV downloads.", "Data Explorer"),
    ]

    for row_start in range(0, len(capabilities), 4):
        row = capabilities[row_start:row_start + 4]
        cols = st.columns(4)
        for col, (icon, title, desc, target_page) in zip(cols, row):
            with col:
                st.markdown(f"""<div class="cc-card" style="min-height:150px;">
                    <div style="font-size:1.6rem;">{icon}</div>
                    <div style="font-weight:700; margin:4px 0;">{title}</div>
                    <div style="font-size:0.82rem; color:var(--subtext);">{desc}</div>
                </div>""", unsafe_allow_html=True)
                if st.button(f"Open {title}", key=f"biz_nav_{target_page}", use_container_width=True):
                    go_to_page(target_page)
                    st.rerun()

    st.markdown('<div class="cc-section-title">Business Value Summary</div>', unsafe_allow_html=True)
    st.markdown("""
- **Early risk detection** — flags high and very-high risk customers before they leave.
- **Prioritized action** — every prediction comes with a concrete, risk-tiered retention recommendation.
- **Transparent decisions** — every score is explainable, not a black box.
- **No manual data prep** — the app detects schema, target, and leakage automatically from the raw CSV.
""")
    st.caption("This page summarizes existing capabilities in this application; figures above reflect the current dataset and training session.")


# =========================================================
# PAGE: ABOUT US
# =========================================================
def page_about() -> None:
    st.markdown('<div class="cc-section-title">ℹ️ About Us — Project Presentation</div>', unsafe_allow_html=True)

    st.markdown("""
**Project Objective**
Predict customer churn risk from real business data and translate model output into
concrete, prioritized retention actions for the business.

**Machine Learning Workflow**
CSV → Schema Detection → Target Detection → Leakage Detection → Feature Preparation →
Train/Test Split → Preprocessing (Imputation, Scaling, Encoding) → Multi-Model Training →
Evaluation → Model Comparison → Best Model Selection → Prediction → Explainability →
Retention Recommendation.

**Dataset**
The application loads and models the real `Customer_Churn_Predictions.csv` dataset —
no synthetic data, no hard-coded metrics.

**Models**
Logistic Regression, Random Forest, Gradient Boosting, Extra Trees, and — when installed —
XGBoost, LightGBM, and CatBoost.

**Evaluation**
Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, ROC & PR Curves.
Model selection prioritizes F1, PR-AUC and Recall over raw Accuracy, which matters for
imbalanced churn data.

**Explainability**
SHAP (when installed) or model-native feature importance / coefficients.

**Business Use Case**
Help retention and customer-success teams identify at-risk customers early and prioritize
outreach using a transparent, data-driven risk score.
""")

    st.markdown("#### Technology Stack")
    stack = {
        "Python": "Core language", "Pandas / NumPy": "Data processing", "Scikit-Learn": "ML modeling",
        "Plotly": "Interactive visualization", "Streamlit": "Web application framework",
        "XGBoost (optional)": "Installed" if XGBClassifier else "Not installed",
        "LightGBM (optional)": "Installed" if LGBMClassifier else "Not installed",
        "CatBoost (optional)": "Installed" if CatBoostClassifier else "Not installed",
        "SHAP (optional)": "Installed" if SHAP_AVAILABLE else "Not installed",
    }
    st.table(pd.DataFrame(stack.items(), columns=["Component", "Status"]))

    st.markdown('<div class="cc-section-title">Project Presentation Highlights</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""<div class="cc-card">
            <div style="font-weight:700; margin-bottom:6px;">🎯 Problem</div>
            <div style="font-size:0.88rem; color:var(--subtext);">
            Businesses lose recurring revenue when customers churn silently.
            Most teams react only after the customer has already left.</div>
        </div>""", unsafe_allow_html=True)
    with p2:
        st.markdown("""<div class="cc-card">
            <div style="font-weight:700; margin-bottom:6px;">💡 Solution</div>
            <div style="font-size:0.88rem; color:var(--subtext);">
            An end-to-end ML system that scores every customer's churn risk,
            explains why, and recommends the right retention action — automatically,
            straight from raw data.</div>
        </div>""", unsafe_allow_html=True)
    with p3:
        st.markdown("""<div class="cc-card">
            <div style="font-weight:700; margin-bottom:6px;">📈 Impact</div>
            <div style="font-size:0.88rem; color:var(--subtext);">
            Retention teams can prioritize outreach by risk tier instead of guessing,
            turning a reactive process into a proactive one.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("#### Project Journey")
    st.markdown("""
1. **Data Understanding** — inspected the raw churn dataset schema, types, and quality.
2. **Target & Leakage Audit** — identified the true churn label and removed columns that would leak the answer.
3. **Feature Engineering** — separated numerical, categorical, boolean, and date-derived features.
4. **Multi-Model Training** — trained and fairly compared several classification algorithms.
5. **Explainability Layer** — added transparent, feature-level reasoning behind every score.
6. **Business Layer** — translated model output into dashboards, risk tiers, and retention actions.
""")

    st.markdown("#### Presented By")
    st.markdown("""<div class="cc-card">
        <div style="font-weight:700;">S Mohammed Kaif</div>
        <div style="color:var(--subtext); font-size:0.85rem;">
        Data Science / Machine Learning — Customer Churn AI Project</div>
    </div>""", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
def render_footer() -> None:
    github_line = f'<a href="{GITHUB_URL}" target="_blank">GitHub Profile</a>' if GITHUB_URL != "YOUR_GITHUB_URL" else "GitHub Profile"
    st.markdown(f"""
    <div class="cc-footer">
        <strong>CUSTOMER CHURN AI</strong><br/>
        Customer Risk Prediction &amp; Retention Intelligence<br/>
        Python • Pandas • Scikit-Learn • Plotly • Streamlit<br/><br/>
        Developed by: <strong>S Mohammed Kaif</strong><br/>
        {github_line}
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# MAIN
# =========================================================
def main() -> None:
    init_session_state()
    inject_css(st.session_state.theme)
    render_bubble_background()

    df = load_data(CSV_PATH)
    if df is None:
        render_header()
        st.error(f"Dataset not found or invalid: `{CSV_PATH}`. Place the CSV next to this script and reload.")
        render_footer()
        return

    schema = detect_schema(df)

    render_header()

    # ---- Target detection ----
    candidates = detect_target_candidates(df)
    target_col = st.session_state.target_col

    if target_col is None:
        if candidates:
            target_col = candidates[0]
        else:
            target_col = None

    if target_col is None or target_col not in df.columns:
        st.warning("Automatic churn target detection did not find a confident match.")
        target_col = st.selectbox("Select Churn Target Column", df.columns.tolist())

    st.session_state.target_col = target_col

    y, original_target, pos_label = normalize_target(df[target_col], st.session_state.positive_class_label)

    if y is None:
        uniques = df[target_col].dropna().unique().tolist()
        st.warning("Could not automatically determine which class represents churn.")
        pos_label = st.selectbox("Select the positive (Churn) class", uniques)
        st.session_state.positive_class_label = pos_label
        y, original_target, pos_label = normalize_target(df[target_col], pos_label)

    if y is None or y.nunique() < 2:
        st.error("The selected target column does not contain at least two classes suitable for classification.")
        render_footer()
        return

    st.session_state.positive_class_label = pos_label

    # ---- Leakage detection ----
    leakage_cols = detect_leakage_columns(df, target_col, schema["id_cols"], y)

    # ---- Feature preparation ----
    X, feature_cols, numeric_features, categorical_features, working_df = prepare_features(df, schema, target_col, leakage_cols)
    st.session_state["_X_full"] = X
    st.session_state["_y_full"] = y

    # ---- Sidebar ----
    all_model_names = list(get_available_models().keys())
    render_sidebar(schema, all_model_names)

    # ---- Training trigger ----
    if st.session_state.get("_train_clicked"):
        train_mode = st.session_state.get("_train_mode")
        models_to_train = all_model_names if train_mode == "Train All Models" else [st.session_state.selected_model_name or all_model_names[0]]
        data_signature = f"{df.shape}-{target_col}-{len(feature_cols)}"
        with st.spinner("Training model(s) on the dataset..."):
            fitted, results, split = train_models(
                X, y, tuple(numeric_features), tuple(categorical_features),
                tuple(models_to_train), data_signature,
            )
        st.session_state.trained_models.update(fitted)
        if st.session_state.model_metrics is None:
            st.session_state.model_metrics = {}
        st.session_state.model_metrics.update(results)
        st.session_state.model_comparison = results
        best = select_best_model(st.session_state.model_metrics)
        if best:
            st.session_state.best_model_name = best
            if st.session_state.selected_model_name is None:
                st.session_state.selected_model_name = best
        st.session_state.training_done = len(st.session_state.trained_models) > 0
        st.success(f"Training complete. {len(fitted)} model(s) trained.")

    # ---- Page routing ----
    page = st.session_state.selected_page
    if page == "Business Overview":
        page_business_overview(df, schema, y)
    elif page == "Dashboard":
        page_dashboard(df, schema, y)
    elif page == "Analytics":
        page_analytics(df, schema, y)
    elif page == "Customers":
        page_customers(df, schema, y)
    elif page == "Predict Churn":
        page_predict(working_df, feature_cols, numeric_features, categorical_features)
    elif page == "Model Performance":
        page_model_performance(schema)
    elif page == "Explainability":
        page_explainability(df, feature_cols)
    elif page == "Retention Intelligence":
        page_retention(df, y)
    elif page == "Data Explorer":
        page_data_explorer(df, schema, leakage_cols)
    elif page == "About Us":
        page_about()

    render_footer()


if __name__ == "__main__":
    main()