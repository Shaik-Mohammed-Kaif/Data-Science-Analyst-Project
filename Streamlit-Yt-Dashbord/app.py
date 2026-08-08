"""
====================================================================================
 YOUTUBE TRENDING VIDEOS ANALYTICS — STREAMLIT BUSINESS INTELLIGENCE DASHBOARD
------------------------------------------------------------------------------------
 Stack   : Python, Streamlit, Pandas, NumPy, Plotly, SQLite
 Author  : S Mohammed Kaif — Jr. Data Analyst
 Run     : streamlit run app.py
====================================================================================
"""

import os
import re
import ast
import sqlite3
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

CSV_PATH = "trending_videos_data.csv"
CSV_PATH_FALLBACK = "trending_videos.csv"
DB_PATH = "trending_videos.db"
TABLE_NAME = "trending_videos"
CHART_H = 380

# ==================================================================================
# PAGE CONFIG
# ==================================================================================
st.set_page_config(
    page_title="YouTube Trending Videos Analytics",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================================================
# CUSTOM CSS — Black / YouTube Red / White premium BI theme
# ==================================================================================

def inject_css():
    st.markdown("""
    <style>

    .stApp {
        background: #0A0A0C;
    }

    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    section[data-testid="stSidebar"] {
        background: #0D0D0F;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    h1, h2, h3, h4, h5, h6, p, span, label, div, li {
        color: #EDEDEF;
    }

    /* ---------------- Header ---------------- */

    .yt-header {
        background: linear-gradient(
            135deg,
            #14090A 0%,
            #0A0A0C 60%
        );

        border: 1px solid rgba(230,33,23,0.35);
        border-radius: 18px;
        padding: 22px 28px;
        margin-bottom: 18px;

        box-shadow:
            0 0 40px rgba(230,33,23,0.12);

        position: relative;
        overflow: hidden;
    }

    .yt-title {
        font-size: 34px;
        font-weight: 900;
        line-height: 1.15;
    }

    .yt-title .brand {
        color: #E62117;
    }

    .yt-subtitle {
        color: #A0A0A5;
        font-size: 14px;
        margin-top: 2px;
    }

    .yt-updated {
        text-align: right;
        font-size: 12px;
        color: #A0A0A5;

        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: 8px 14px;

        background: rgba(255,255,255,0.03);
    }

    .yt-updated b {
        color: #FFFFFF;
        font-size: 15px;
    }

    /* ---------------- KPI cards ---------------- */

    .kpi-card {
        background: linear-gradient(
            160deg,
            #17171A 0%,
            #0F0F11 100%
        );

        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 14px;

        padding: 16px;
        min-height: 108px;

        transition:
            box-shadow .2s ease,
            transform .2s ease;
    }

    .kpi-card:hover {
        box-shadow:
            0 0 22px rgba(230,33,23,0.28);

        transform: translateY(-2px);
    }

    .kpi-icon {
        width: 34px;
        height: 34px;

        border-radius: 9px;
        background: #E62117;

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 17px;
        margin-bottom: 8px;
    }

    .kpi-label {
        font-size: 12px;
        color: #A0A0A5;

        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .3px;
    }

    .kpi-value {
        font-size: 24px;
        font-weight: 800;
        color: #FFFFFF;

        margin-top: 2px;
    }

    .kpi-sub {
        font-size: 11px;
        color: #6F6F73;

        margin-top: 2px;
    }

    /* ---------------- Panels ---------------- */

    .panel {
        background: #101012;

        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;

        padding: 16px 18px 10px 18px;
        margin-bottom: 16px;
    }

    .panel-title {
        font-size: 15px;
        font-weight: 700;
        color: #FFFFFF;

        margin-bottom: 6px;
    }

    /* ---------------- Weekend / Weekday ---------------- */

    .cmp-card {
        border-radius: 14px;
        padding: 18px;

        text-align: center;

        border: 1px solid rgba(255,255,255,0.10);
    }

    .cmp-weekday {
        background: linear-gradient(
            160deg,
            #111827,
            #0B1120
        );
    }

    .cmp-weekend {
        background: linear-gradient(
            160deg,
            #2A0E0C,
            #170706
        );
    }

    .cmp-label {
        font-size: 12px;
        letter-spacing: .5px;

        color: #C7C7CC;
        font-weight: 700;
    }

    .cmp-value {
        font-size: 24px;
        font-weight: 800;

        color: #FFFFFF;
        margin: 4px 0;
    }

    .cmp-metric {
        font-size: 11px;
        color: #9A9AA0;
    }

    /* ---------------- Insight cards ---------------- */

    .insight-card {
        background: #101012;

        border-left: 3px solid #E62117;
        border-radius: 10px;

        padding: 12px 14px;
        margin-bottom: 10px;
    }

    .insight-title {
        font-size: 11px;
        color: #A0A0A5;

        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: .4px;
    }

    .insight-value {
        font-size: 16px;
        color: #FFFFFF;

        font-weight: 800;
        margin-top: 2px;
    }

    /* ---------------- Dataset column cards ---------------- */

    .col-group {
        background: #101012;

        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;

        padding: 12px 14px;
        height: 100%;
    }

    .col-group-title {
        color: #E62117;

        font-weight: 800;
        font-size: 12.5px;

        margin-bottom: 6px;

        border-bottom:
            1px solid rgba(230,33,23,0.3);

        padding-bottom: 4px;
    }

    .col-item {
        font-size: 11.5px;
        color: #B5B5BA;

        padding: 1.5px 0;
    }

    /* ---------------- Footer ---------------- */

    .yt-footer {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;

        border-top:
            1px solid rgba(255,255,255,0.08);

        margin-top: 24px;
        padding: 16px 6px;

        font-size: 12px;
        color: #9A9AA0;
    }

    .yt-footer b {
        color: #FFFFFF;
    }

    /* ---------------- Buttons ---------------- */

    div.stButton > button {
        border-radius: 8px;

        border:
            1px solid rgba(230,33,23,0.4);

        background: #E62117;
        color: white;

        font-weight: 700;
    }

    div.stButton > button:hover {
        background: #B5170F;
    }

    /* ---------------- Tabs ---------------- */

    .stTabs [data-baseweb="tab"] {
        font-weight: 700;
        font-size: 13px;
    }

    /* ---------------- Dataframe ---------------- */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------------- Selectbox ---------------- */

    div[data-baseweb="select"] > div {
        background: #151517;
        border-color: rgba(255,255,255,0.10);
    }

    </style>
    """, unsafe_allow_html=True)


# ==================================================================================
# DATA LOADING LAYER
# ==================================================================================

@st.cache_data(show_spinner=False)
def load_data():
    """Load from SQLite if available, else fall back to CSV."""

    if os.path.exists(DB_PATH):
        conn = None

        try:
            conn = sqlite3.connect(DB_PATH)

            df = pd.read_sql_query(
                f"SELECT * FROM {TABLE_NAME}",
                conn
            )

            return df, "SQLite (trending_videos.db)"

        except Exception:
            pass

        finally:
            if conn is not None:
                conn.close()

    for path in [CSV_PATH, CSV_PATH_FALLBACK]:

        if os.path.exists(path):

            try:
                return (
                    pd.read_csv(path),
                    f"CSV ({path})"
                )

            except Exception:
                continue

    return None, None


# ==================================================================================
# SAFE URL COUNT
# ==================================================================================

def _safe_url_count(text):

    if pd.isna(text):
        return 0

    if not isinstance(text, str):
        return 0

    return len(
        re.findall(
            r"https?://[^\s<>\"]+",
            text
        )
    )


# ==================================================================================
# SAFE HASHTAG COUNT
# ==================================================================================

def _safe_hashtag_count(text):

    if pd.isna(text):
        return 0

    if not isinstance(text, str):
        return 0

    return len(
        re.findall(
            r"#\w+",
            text
        )
    )


# ==================================================================================
# SAFE TAG CLEANING
# ==================================================================================

def _safe_tags_clean(raw):

    if raw is None:
        return []

    if isinstance(raw, list):
        return [
            str(t).strip()
            for t in raw
            if str(t).strip()
        ]

    if not isinstance(raw, str):
        return []

    if not raw.strip():
        return []

    try:

        parsed = ast.literal_eval(raw)

        if isinstance(parsed, list):

            return [
                str(t).strip()
                for t in parsed
                if str(t).strip()
            ]

    except (ValueError, SyntaxError):
        pass

    return [
        t.strip()
        for t in raw.split(",")
        if t.strip()
    ]


# ==================================================================================
# DATA CLEANING + FEATURE ENGINEERING
# ==================================================================================

@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    """Type-safe cleaning + derived feature engineering."""

    if df is None:
        return pd.DataFrame()

    df = df.copy()

    if df.empty:
        return df.reset_index(drop=True)

    # --------------------------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------------------------

    numeric_cols = [
        "view_count",
        "like_count",
        "comment_count",
        "favorite_count",
        "engagement_score",
        "engagement_rate",
        "like_rate",
        "comment_rate",
        "duration_seconds",
        "title_length",
        "title_word_count",
        "description_length",
        "description_word_count",
        "tag_count",
        "publish_year",
        "publish_month",
        "publish_day",
        "publish_week",
        "publish_hour",
        "publish_minute",
        "publish_second",
        "category_id",
    ]

    for c in numeric_cols:

        if c in df.columns:

            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

    # --------------------------------------------------------------------------
    # Main performance metrics
    # --------------------------------------------------------------------------

    for c in [
        "view_count",
        "like_count",
        "comment_count",
        "favorite_count"
    ]:

        if c in df.columns:

            df[c] = (
                df[c]
                .fillna(0)
                .clip(lower=0)
            )

    # --------------------------------------------------------------------------
    # Engagement rate
    # --------------------------------------------------------------------------

    if (
        "engagement_rate" not in df.columns
        and {
            "like_count",
            "comment_count",
            "view_count"
        }.issubset(df.columns)
    ):

        df["engagement_rate"] = (
            (
                df["like_count"]
                + df["comment_count"]
            )
            /
            df["view_count"].replace(0, np.nan)
        ).fillna(0)

    elif "engagement_rate" in df.columns:

        df["engagement_rate"] = (
            df["engagement_rate"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

    # --------------------------------------------------------------------------
    # Like rate
    # --------------------------------------------------------------------------

    if (
        "like_rate" not in df.columns
        and {"like_count", "view_count"}.issubset(df.columns)
    ):

        df["like_rate"] = (
            df["like_count"]
            /
            df["view_count"].replace(0, np.nan)
        ).fillna(0)

    # --------------------------------------------------------------------------
    # Comment rate
    # --------------------------------------------------------------------------

    if (
        "comment_rate" not in df.columns
        and {"comment_count", "view_count"}.issubset(df.columns)
    ):

        df["comment_rate"] = (
            df["comment_count"]
            /
            df["view_count"].replace(0, np.nan)
        ).fillna(0)

    # --------------------------------------------------------------------------
    # Date columns
    # --------------------------------------------------------------------------

    for c in [
        "published_at",
        "published_at_date"
    ]:

        if c in df.columns:

            df[c] = pd.to_datetime(
                df[c],
                errors="coerce"
            )

    # --------------------------------------------------------------------------
    # Boolean conversion helper
    # --------------------------------------------------------------------------

    def _convert_bool(series):

        if pd.api.types.is_bool_dtype(series):

            return series.fillna(False).astype(bool)

        mapped = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
                "y": True,
                "n": False
            })
        )

        return mapped.fillna(False).astype(bool)

    # --------------------------------------------------------------------------
    # Weekend
    # --------------------------------------------------------------------------

    if "is_weekend" in df.columns:

        df["is_weekend"] = _convert_bool(
            df["is_weekend"]
        )

    # --------------------------------------------------------------------------
    # Caption
    # --------------------------------------------------------------------------

    if "caption" in df.columns:

        if not pd.api.types.is_bool_dtype(df["caption"]):

            df["caption"] = _convert_bool(
                df["caption"]
            )

        else:

            df["caption"] = (
                df["caption"]
                .fillna(False)
                .astype(bool)
            )

    # --------------------------------------------------------------------------
    # Has caption
    # --------------------------------------------------------------------------

    if "has_caption" in df.columns:

        df["has_caption"] = (
            pd.to_numeric(
                df["has_caption"],
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
            .clip(0, 1)
        )

    elif "caption" in df.columns:

        df["has_caption"] = (
            df["caption"]
            .astype(int)
        )

    # --------------------------------------------------------------------------
    # HD
    # --------------------------------------------------------------------------

    if "is_hd" in df.columns:

        df["is_hd"] = (
            pd.to_numeric(
                df["is_hd"],
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
            .clip(0, 1)
        )

    elif "definition" in df.columns:

        df["is_hd"] = (
            df["definition"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("hd")
            .astype(int)
        )

    # --------------------------------------------------------------------------
    # Text / categorical columns
    # --------------------------------------------------------------------------

    for c in [
        "category_name",
        "channel_title",
        "publish_session",
        "duration_category",
        "publish_day_name",
        "publish_month_name",
        "definition",
    ]:

        if c in df.columns:

            df[c] = (
                df[c]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )

    # --------------------------------------------------------------------------
    # Remove duplicate records
    # --------------------------------------------------------------------------

    df = df.drop_duplicates(
        keep="first"
    )

    # --------------------------------------------------------------------------
    # URL features
    # --------------------------------------------------------------------------

    if "urls_links" in df.columns:

        missing_urls = (
            df["urls_links"]
            .isna()
            .all()
        )

        if missing_urls and "description" in df.columns:

            df["url_count"] = (
                df["description"]
                .apply(_safe_url_count)
            )

    elif "description" in df.columns:

        df["url_count"] = (
            df["description"]
            .apply(_safe_url_count)
        )

    else:

        df["url_count"] = 0

    # --------------------------------------------------------------------------
    # Hashtag features
    # --------------------------------------------------------------------------

    if "hashtags" not in df.columns:

        if "description" in df.columns:

            df["hashtag_count"] = (
                df["description"]
                .apply(_safe_hashtag_count)
            )

        else:

            df["hashtag_count"] = 0

    elif "hashtag_count" not in df.columns:

        df["hashtag_count"] = (
            df["hashtags"]
            .apply(_safe_hashtag_count)
        )

    # --------------------------------------------------------------------------
    # Tags
    # --------------------------------------------------------------------------

    if "tags" in df.columns:

        df["tags_clean"] = (
            df["tags"]
            .apply(_safe_tags_clean)
        )

        if "tag_count" not in df.columns:

            df["tag_count"] = (
                df["tags_clean"]
                .apply(len)
            )

    elif "tag_count" not in df.columns:

        df["tag_count"] = 0

    # --------------------------------------------------------------------------
    # Final numeric cleanup
    # --------------------------------------------------------------------------

    for c in [
        "url_count",
        "hashtag_count",
        "tag_count"
    ]:

        if c in df.columns:

            df[c] = (
                pd.to_numeric(
                    df[c],
                    errors="coerce"
                )
                .fillna(0)
                .astype(int)
                .clip(lower=0)
            )

    # --------------------------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df.reset_index(drop=True)


# ==================================================================================
# NUMBER FORMATTER
# ==================================================================================

def format_number(n) -> str:

    try:

        if pd.isna(n):
            return "0"

        n = float(n)

    except (TypeError, ValueError):

        return str(n)

    if abs(n) >= 1_000_000_000:

        return f"{n / 1_000_000_000:.2f}B"

    if abs(n) >= 1_000_000:

        return f"{n / 1_000_000:.2f}M"

    if abs(n) >= 1_000:

        return f"{n / 1_000:.1f}K"

    return f"{n:.0f}"


# ==================================================================================
# PERCENTAGE FORMATTER
# ==================================================================================

def format_percentage(n) -> str:

    try:

        if pd.isna(n):
            return "0.00%"

        return f"{float(n) * 100:.2f}%"

    except (TypeError, ValueError):

        return "N/A"

# ==================================================================================
# FILTERS
# ==================================================================================
def apply_filters(df, category, channel, session, quality, caption, duration_cat, weekend, date_range):
    out = df.copy()
    if category:
        out = out[out["category_name"].isin(category)]
    if channel:
        out = out[out["channel_title"].isin(channel)]
    if session:
        out = out[out["publish_session"].isin(session)]
    if quality:
        out = out[out["definition"].isin(quality)]
    if caption != "All" and "has_caption" in out.columns:
        out = out[out["has_caption"] == (1 if caption == "With Caption" else 0)]
    if duration_cat:
        out = out[out["duration_category"].isin(duration_cat)]
    if weekend != "All" and "is_weekend" in out.columns:
        out = out[out["is_weekend"] == (weekend == "Weekend Only")]
    if date_range and "published_at_date" in out.columns and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        out = out[(out["published_at_date"] >= start) & (out["published_at_date"] <= end)]
    return out


# ==================================================================================
# KPI CALCULATION
# ==================================================================================
def calculate_kpis(df):
    if df.empty:
        return {k: 0 for k in ["total_videos", "total_views", "total_likes", "total_comments",
                                "avg_engagement_rate", "avg_view_count"]}
    return {
        "total_videos": len(df),
        "total_views": df["view_count"].sum(),
        "total_likes": df["like_count"].sum() if "like_count" in df else 0,
        "total_comments": df["comment_count"].sum() if "comment_count" in df else 0,
        "avg_engagement_rate": df["engagement_rate"].mean() if "engagement_rate" in df else 0,
        "avg_view_count": df["view_count"].mean(),
    }


# ==================================================================================
# PLOTLY THEME
# ==================================================================================
def style_fig(fig, title=None, height=CHART_H, show_legend=False):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Roboto, sans-serif", size=12, color="#E5E7EB"),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        height=height,
        showlegend=show_legend,
        title=dict(text=title, font=dict(size=14, color="#FFFFFF")) if title else None,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


RED_SCALE = ["#3A0906", "#7A130E", "#B5170F", "#E62117", "#FF6B5D"]
RED_SEQ = ["#E62117", "#FF6B5D", "#B5170F", "#FF9E92", "#7A130E", "#FFC5BD"]


# ==================================================================================
# CHART BUILDERS
# ==================================================================================
def create_views_over_time(df):
    if "published_at_date" not in df.columns or df["published_at_date"].isna().all():
        return None
    d = df.dropna(subset=["published_at_date"]).copy()
    d = d.groupby(d["published_at_date"].dt.date)["view_count"].sum().reset_index()
    d.columns = ["date", "views"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d["views"], mode="lines+markers",
                              line=dict(color="#E62117", width=3),
                              marker=dict(color="#E62117", size=6),
                              fill="tozeroy", fillcolor="rgba(230,33,23,0.15)",
                              hovertemplate="<b>%{x}</b><br>Views: %{y:,.0f}<extra></extra>"))
    return style_fig(fig)


def create_top_videos_chart(df, n=10, metric="view_count", pct=False):
    if "title" not in df.columns:
        return None
    d = df.nlargest(n, metric)[["title", metric]].copy()
    d["short_title"] = d["title"].apply(lambda t: (t[:35] + "…") if len(str(t)) > 35 else t)
    d = d.sort_values(metric)
    text_fmt = [format_percentage(v) for v in d[metric]] if pct else [format_number(v) for v in d[metric]]
    fig = go.Figure(go.Bar(
        x=d[metric], y=d["short_title"], orientation="h",
        marker_color="#E62117", text=text_fmt, textposition="outside",
        customdata=d["title"],
        hovertemplate="<b>%{customdata}</b><br>Value: %{x:,.4f}<extra></extra>",
    ))
    return style_fig(fig)


def create_top_channels_chart(df, n=10):
    d = df.groupby("channel_title")["view_count"].sum().sort_values(ascending=False).head(n).reset_index()
    d = d.sort_values("view_count")
    fig = go.Figure(go.Bar(
        x=d["view_count"], y=d["channel_title"], orientation="h",
        marker_color="#E62117", text=[format_number(v) for v in d["view_count"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Total Views: %{x:,.0f}<extra></extra>",
    ))
    return style_fig(fig)


def create_category_views_chart(df):
    d = df.groupby("category_name")["view_count"].sum().sort_values(ascending=False).reset_index()
    fig = go.Figure(go.Pie(
        labels=d["category_name"], values=d["view_count"], hole=0.55,
        marker=dict(colors=RED_SEQ), textinfo="percent",
        hovertemplate="<b>%{label}</b><br>Views: %{value:,.0f}<br>%{percent}<extra></extra>",
    ))
    return style_fig(fig, show_legend=True)


def create_category_engagement_chart(df):
    d = df.groupby("category_name")["engagement_rate"].mean().sort_values(ascending=False).reset_index()
    fig = go.Figure(go.Bar(
        x=d["category_name"], y=d["engagement_rate"]*100,
        marker_color="#E62117", text=[f"{v*100:.1f}%" for v in d["engagement_rate"]], textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg Engagement: %{y:.2f}%<extra></extra>",
    ))
    fig.update_yaxes(title="Engagement Rate (%)")
    return style_fig(fig)


def create_publish_hour_chart(df):
    d = df.groupby("publish_hour").agg(avg_views=("view_count", "mean"), n=("view_count", "size")).reindex(range(24)).reset_index()
    d.columns = ["hour", "avg_views", "n"]
    fig = go.Figure(go.Scatter(
        x=d["hour"], y=d["avg_views"], mode="lines+markers",
        line=dict(color="#E62117", width=3), marker=dict(size=6, color="#E62117"),
        customdata=d["n"],
        hovertemplate="<b>Hour %{x}:00</b><br>Avg Views: %{y:,.0f}<br>Videos: %{customdata}<extra></extra>",
    ))
    fig.update_xaxes(dtick=2, title="Hour of Day")
    return style_fig(fig)


def create_quality_chart(df):
    d = df.groupby("definition")["view_count"].mean().sort_values(ascending=False).reset_index()
    fig = go.Figure(go.Bar(
        x=d["view_count"], y=d["definition"].str.upper(), orientation="h",
        marker_color=["#E62117", "#F59E0B", "#22C55E", "#3B82F6"][:len(d)],
        text=[format_number(v) for v in d["view_count"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Avg Views: %{x:,.0f}<extra></extra>",
    ))
    return style_fig(fig)


def create_engagement_chart(df, n=10):
    return create_top_videos_chart(df, n=n, metric="engagement_rate", pct=True)


# ==================================================================================
# BUSINESS INSIGHTS
# ==================================================================================
def generate_business_insights(df):
    ins = {}
    if df.empty:
        return ins
    try:
        ins["top_category"] = df.groupby("category_name")["view_count"].sum().idxmax()
    except Exception:
        pass
    try:
        ins["highest_engagement_category"] = df.groupby("category_name")["engagement_rate"].mean().idxmax()
    except Exception:
        pass
    try:
        hr = df.groupby("publish_hour")["view_count"].mean().idxmax()
        ins["best_hour"] = f"{int(hr):02d}:00"
    except Exception:
        pass
    try:
        ins["best_session"] = df.groupby("publish_session")["engagement_rate"].mean().idxmax()
    except Exception:
        pass
    try:
        ins["top_channel"] = df.groupby("channel_title")["view_count"].sum().idxmax()
    except Exception:
        pass
    try:
        wk = df.groupby("is_weekend")["view_count"].mean()
        ins["weekend_winner"] = "Weekend" if wk.get(True, 0) > wk.get(False, 0) else "Weekday"
    except Exception:
        pass
    try:
        cap = df.groupby("has_caption")["engagement_rate"].mean()
        ins["caption_winner"] = "With Caption" if cap.get(1, 0) > cap.get(0, 0) else "Without Caption"
    except Exception:
        pass
    try:
        top_vid = df.loc[df["view_count"].idxmax()]
        ins["top_video"] = str(top_vid.get("title", "N/A"))[:45]
    except Exception:
        pass
    try:
        ins["best_duration"] = df.groupby("duration_category")["view_count"].mean().idxmax()
    except Exception:
        pass
    return ins


# ==================================================================================
# LOAD & CLEAN
# ==================================================================================
raw_df, data_source = load_data()

if raw_df is None:
    st.error("❌ Could not find `trending_videos.db` or a trending videos CSV in the app folder. "
              "Place `trending_videos.db` (table `trending_videos`) or `trending_videos_data.csv` "
              "next to `app.py` and restart.")
    st.stop()

df = clean_data(raw_df)
inject_css()

if df.empty:
    st.error("❌ Dataset loaded but contains zero usable rows.")
    st.stop()

REQUIRED_COLS = ["view_count", "like_count", "comment_count", "category_name",
                  "channel_title", "engagement_rate"]
missing_req = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_req:
    st.warning(f"⚠️ Missing expected columns: {', '.join(missing_req)}. Some sections may be limited.")

# ==================================================================================
# HEADER
# ==================================================================================
now = datetime.now()
header_html = f"""
<div class="yt-header">
  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
    <div style="display:flex; align-items:center; gap:14px;">
      <div style="width:52px;height:52px;background:#E62117;border-radius:12px;
                  display:flex;align-items:center;justify-content:center;font-size:22px;">▶</div>
      <div>
        <div class="yt-title"><span class="brand">YouTube</span> Trending Videos Analytics</div>
        <div class="yt-subtitle">Streamlit Business Intelligence Dashboard</div>
      </div>
    </div>
    <div class="yt-updated">
      Last Updated<br><b>{now.strftime('%d %b %Y')}</b><br>{now.strftime('%I:%M %p')}
    </div>
  </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ==================================================================================
# SIDEBAR FILTERS
# ==================================================================================
with st.sidebar:
    st.markdown("### 🔻 Filters")
    st.caption(f"Source: {data_source}  |  Rows: {len(df)}")
    st.markdown("---")

    date_range = None
    if "published_at_date" in df.columns and df["published_at_date"].notna().any():
        min_d, max_d = df["published_at_date"].min(), df["published_at_date"].max()
        date_range = st.date_input("📅 Date Range", value=(min_d.date(), max_d.date()),
                                     min_value=min_d.date(), max_value=max_d.date())

    category = st.multiselect("📁 Category", sorted(df["category_name"].unique()) if "category_name" in df else [])
    channel = st.multiselect("📡 Channel", sorted(df["channel_title"].unique()) if "channel_title" in df else [])
    session = st.multiselect("🕐 Publish Session", sorted(df["publish_session"].unique()) if "publish_session" in df else [])
    quality = st.multiselect("🎞️ Video Quality", sorted(df["definition"].unique()) if "definition" in df else [])
    caption = st.selectbox("💬 Caption Available", ["All", "With Caption", "Without Caption"])
    duration_cat = st.multiselect("⏱️ Duration Category", sorted(df["duration_category"].unique()) if "duration_category" in df else [])
    weekend = st.selectbox("📆 Weekend / Weekday", ["All", "Weekend Only", "Weekday Only"])

    st.markdown("---")
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.rerun()

fdf = apply_filters(df, category, channel, session, quality, caption, duration_cat, weekend, date_range)
FILTER_EMPTY = fdf.empty

if FILTER_EMPTY:
    st.info("No data available for the selected filters.")
    st.stop()

# ==================================================================================
# KPI CARDS
# ==================================================================================
kpis = calculate_kpis(fdf)

total_all_views = df["view_count"].sum() if "view_count" in df else 1

pct_of_total_views = (
    kpis["total_views"] / total_all_views * 100
    if total_all_views
    else 0
)

kpi_defs = [
    ("🎬", "Total Videos", format_number(kpis["total_videos"]),
     f"{len(fdf) / len(df) * 100:.0f}% of Total" if len(df) else "0% of Total"),

    ("👁️", "Total Views", format_number(kpis["total_views"]),
     f"{pct_of_total_views:.0f}% of Total"),

    ("👍", "Total Likes", format_number(kpis["total_likes"]),
     "Sum of Likes"),

    ("💬", "Total Comments", format_number(kpis["total_comments"]),
     "Sum of Comments"),

    ("🔥", "Avg Engagement Rate",
     format_percentage(kpis["avg_engagement_rate"]),
     "Filtered Avg"),

    ("📈", "Avg View Count",
     format_number(kpis["avg_view_count"]),
     "Per Video"),
]

cols = st.columns(6)

for col, (icon, label, value, sub) in zip(cols, kpi_defs):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("")
# ==================================================================================
# ROW 1 — Views Over Time | Top 10 Videos | Top 10 Channels
# ==================================================================================
r1c1, r1c2, r1c3 = st.columns([1.3, 1, 1])
with r1c1:
    st.markdown('<div class="panel"><div class="panel-title">1. Views Over Time</div>', unsafe_allow_html=True)
    fig = create_views_over_time(fdf)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No publish-date data available.")
    st.markdown('</div>', unsafe_allow_html=True)

with r1c2:
    st.markdown('<div class="panel"><div class="panel-title">2. Top 10 Videos by Views</div>', unsafe_allow_html=True)
    st.plotly_chart(create_top_videos_chart(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with r1c3:
    st.markdown('<div class="panel"><div class="panel-title">3. Top 10 Channels by Total Views</div>', unsafe_allow_html=True)
    st.plotly_chart(create_top_channels_chart(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================================
# ROW 2 — Views by Category | Engagement by Category | Views by Publish Hour
# ==================================================================================
r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    st.markdown('<div class="panel"><div class="panel-title">4. Views by Category</div>', unsafe_allow_html=True)
    st.plotly_chart(create_category_views_chart(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with r2c2:
    st.markdown('<div class="panel"><div class="panel-title">5. Engagement Rate by Category</div>', unsafe_allow_html=True)
    st.plotly_chart(create_category_engagement_chart(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with r2c3:
    st.markdown('<div class="panel"><div class="panel-title">6. Views by Publish Hour</div>', unsafe_allow_html=True)
    st.plotly_chart(create_publish_hour_chart(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================================
# ROW 3 — Weekend vs Weekday | Video Quality | Caption vs Engagement | Top Engagement
# ==================================================================================
r3c1, r3c2, r3c3, r3c4 = st.columns(4)

with r3c1:
    st.markdown('<div class="panel"><div class="panel-title">7. Weekend vs Weekday Performance</div>', unsafe_allow_html=True)
    wk_stats = fdf.groupby("is_weekend").agg(avg_views=("view_count", "mean"),
                                              avg_eng=("engagement_rate", "mean"))
    weekday_v = wk_stats.loc[False, "avg_views"] if False in wk_stats.index else 0
    weekday_e = wk_stats.loc[False, "avg_eng"] if False in wk_stats.index else 0
    weekend_v = wk_stats.loc[True, "avg_views"] if True in wk_stats.index else 0
    weekend_e = wk_stats.loc[True, "avg_eng"] if True in wk_stats.index else 0
    cc1, cc2 = st.columns(2)
    cc1.markdown(f"""<div class="cmp-card cmp-weekday">
        <div class="cmp-label">WEEKDAY</div>
        <div class="cmp-metric">AVG VIEWS</div><div class="cmp-value">{format_number(weekday_v)}</div>
        <div class="cmp-metric">AVG ENGAGEMENT</div><div class="cmp-value" style="font-size:16px;">{format_percentage(weekday_e)}</div>
        </div>""", unsafe_allow_html=True)
    cc2.markdown(f"""<div class="cmp-card cmp-weekend">
        <div class="cmp-label">WEEKEND</div>
        <div class="cmp-metric">AVG VIEWS</div><div class="cmp-value">{format_number(weekend_v)}</div>
        <div class="cmp-metric">AVG ENGAGEMENT</div><div class="cmp-value" style="font-size:16px;">{format_percentage(weekend_e)}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with r3c2:
    st.markdown('<div class="panel"><div class="panel-title">8. Video Quality vs Avg Views</div>', unsafe_allow_html=True)
    st.plotly_chart(create_quality_chart(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with r3c3:
    st.markdown('<div class="panel"><div class="panel-title">9. Caption Availability vs Engagement</div>', unsafe_allow_html=True)
    cap_stats = fdf.groupby("has_caption")["engagement_rate"].mean()
    with_cap = cap_stats.get(1, 0)
    without_cap = cap_stats.get(0, 0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=with_cap*100,
        number={"suffix": "%", "font": {"color": "#FFFFFF", "size": 26}},
        gauge={"axis": {"range": [0, max(10, with_cap*100*1.5)], "tickcolor": "#666"},
               "bar": {"color": "#E62117"},
               "bgcolor": "rgba(0,0,0,0)",
               "borderwidth": 0}
    ))
    st.plotly_chart(style_fig(fig, height=180), use_container_width=True)
    cw, cwo = st.columns(2)
    cw.metric("With Caption", format_percentage(with_cap))
    cwo.metric("Without Caption", format_percentage(without_cap))
    st.markdown('</div>', unsafe_allow_html=True)

with r3c4:
    st.markdown('<div class="panel"><div class="panel-title">10. Top 10 Videos by Engagement Rate</div>', unsafe_allow_html=True)
    st.plotly_chart(create_engagement_chart(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================================
# CONTENT & NLP ANALYTICS
# ==================================================================================
st.markdown("## 🧠 Content & NLP Analytics")
nlp_tabs = st.tabs(["Title & Description", "Tags / Hashtags / URLs", "Duration & Session"])

with nlp_tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(fdf, x="title_length", y="view_count", color="category_name",
                          hover_name="title" if "title" in fdf.columns else None,
                          color_discrete_sequence=RED_SEQ, log_y=True)
        st.plotly_chart(style_fig(fig, "Title Length vs Views"), use_container_width=True)
    with c2:
        fig = px.box(fdf, x="title_word_count", y="engagement_rate",
                     color_discrete_sequence=["#E62117"])
        st.plotly_chart(style_fig(fig, "Title Word Count vs Engagement"), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(fdf, x="description_length", y="view_count", color="category_name",
                          color_discrete_sequence=RED_SEQ, log_y=True)
        st.plotly_chart(style_fig(fig, "Description Length vs Views"), use_container_width=True)
    with c4:
        fig = px.scatter(fdf, x="description_word_count", y="engagement_rate",
                          color_discrete_sequence=["#E62117"], trendline=None)
        st.plotly_chart(style_fig(fig, "Description Word Count vs Engagement"), use_container_width=True)

with nlp_tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(fdf, x="hashtag_count", y="view_count", color_discrete_sequence=["#E62117"], log_y=True)
        st.plotly_chart(style_fig(fig, "Hashtag Count vs Views"), use_container_width=True)
    with c2:
        fig = px.scatter(fdf, x="tag_count", y="engagement_rate", color_discrete_sequence=["#E62117"])
        st.plotly_chart(style_fig(fig, "Tag Count vs Engagement"), use_container_width=True)
    fig = px.histogram(fdf, x="url_count", color_discrete_sequence=["#E62117"], nbins=10)
    st.plotly_chart(style_fig(fig, "URL Count Distribution"), use_container_width=True)

with nlp_tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        d = fdf.groupby("duration_category")["engagement_rate"].mean().reset_index()
        fig = px.bar(d, x="duration_category", y="engagement_rate", color_discrete_sequence=["#E62117"])
        st.plotly_chart(style_fig(fig, "Duration Category vs Engagement"), use_container_width=True)
    with c2:
        d = fdf.groupby("publish_session")["engagement_rate"].mean().reset_index()
        fig = px.bar(d, x="publish_session", y="engagement_rate", color_discrete_sequence=["#E62117"])
        st.plotly_chart(style_fig(fig, "Publish Session vs Engagement"), use_container_width=True)
    total_len = fdf["title_length"].fillna(0) + fdf["description_length"].fillna(0)
    fig = px.scatter(x=total_len, y=fdf["view_count"], color_discrete_sequence=["#E62117"], log_y=True,
                      labels={"x": "Total Content Length (title+description)", "y": "view_count"})
    st.plotly_chart(style_fig(fig, "Content Length vs Performance"), use_container_width=True)

# ==================================================================================
# BUSINESS INSIGHTS
# ==================================================================================
st.markdown("## 💡 Key Business Insights")
insights = generate_business_insights(fdf)
insight_defs = [
    ("🏆 Top Category (Views)", insights.get("top_category", "N/A")),
    ("🔥 Highest Engagement Category", insights.get("highest_engagement_category", "N/A")),
    ("⏰ Best Publishing Hour", insights.get("best_hour", "N/A")),
    ("🕐 Best Publishing Session", insights.get("best_session", "N/A")),
    ("📡 Top-Performing Channel", insights.get("top_channel", "N/A")),
    ("📆 Weekend vs Weekday Winner", insights.get("weekend_winner", "N/A")),
    ("💬 Caption Winner", insights.get("caption_winner", "N/A")),
    ("⏱️ Best Duration Category", insights.get("best_duration", "N/A")),
]
ic = st.columns(4)
for i, (title, value) in enumerate(insight_defs):
    with ic[i % 4]:
        st.markdown(f"""<div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-value">{value}</div></div>""", unsafe_allow_html=True)
if "top_video" in insights:
    st.markdown(f"""<div class="insight-card">
        <div class="insight-title">🥇 Highest-Performing Video</div>
        <div class="insight-value">{insights['top_video']}</div></div>""", unsafe_allow_html=True)

# ==================================================================================
# DATA EXPLORER
# ==================================================================================
with st.expander("🔎 Data Explorer"):
    st.write(f"**Filtered Rows:** {fdf.shape[0]}  |  **Filtered Columns:** {fdf.shape[1]}")
    st.dataframe(fdf, use_container_width=True, height=320)
    csv_bytes = fdf.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data (CSV)", data=csv_bytes,
                        file_name="filtered_trending_videos.csv", mime="text/csv")

# ==================================================================================
# DATASET COLUMNS INFO PANEL
# ==================================================================================
st.markdown("## 📚 Dataset Columns Used")
COLUMN_GROUPS = {
    "Video Information": ["video_id", "title", "description", "channel_id", "channel_title",
                            "category_id", "category_name"],
    "Publishing Information": ["published_at", "published_at_date", "publish_year", "publish_month",
                                 "publish_month_name", "publish_day", "publish_day_name", "publish_week",
                                 "publish_hour", "publish_session", "is_weekend"],
    "Performance Metrics": ["view_count", "like_count", "comment_count", "favorite_count",
                              "engagement_score", "engagement_rate", "like_rate", "comment_rate"],
    "Video Attributes": ["duration", "duration_seconds", "duration_category", "definition",
                          "is_hd", "caption", "has_caption"],
    "URL / Hashtag / Tag Features": ["url_count", "hashtag_count", "tags", "tags_clean", "tag_count"],
    "NLP / Text Features": ["title_length", "title_word_count", "description_length", "description_word_count"],
}
gcols = st.columns(3)
for i, (group, cols_list) in enumerate(COLUMN_GROUPS.items()):
    with gcols[i % 3]:
        items_html = "".join(f'<div class="col-item">• {c}</div>' for c in cols_list)
        st.markdown(f"""<div class="col-group">
            <div class="col-group-title">{group}</div>{items_html}</div>""", unsafe_allow_html=True)
        st.write("")

# ==================================================================================
# FOOTER
# ==================================================================================
st.markdown(f"""
<div class="yt-footer">
    <div>📊 Tool<br><b>Streamlit</b></div>
    <div>🗄️ Database<br><b>{ "SQLite" if "SQLite" in (data_source or "") else "CSV"}</b></div>
    <div>📁 Dataset<br><b>YouTube Trending Videos</b></div>
    <div>📌 Total Records<br><b>{len(df)}</b></div>
    <div>🚀 Project<br><b>YouTube Trending Videos Analytics</b></div>
</div>
""", unsafe_allow_html=True)