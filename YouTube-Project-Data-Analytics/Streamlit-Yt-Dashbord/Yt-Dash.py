"""
===============================================================================
YOUTUBE ANALYTICS DASHBOARD
===============================================================================
Professional historical YouTube analytics dashboard.

IMPORTANT
---------
This is an ANALYTICS dashboard, not a prediction/classification application.

It analyses historical YouTube data and creates:
    • Executive KPI cards
    • Monthly views trend
    • Engagement distribution
    • Top videos
    • Publishing day/month heatmap
    • Publishing-hour analysis
    • Category performance
    • Channel performance
    • Duration analysis
    • Likes/views relationship
    • Comments/views relationship
    • Engagement analysis
    • Video quality analysis
    • Caption analysis
    • Title/tag/hashtag analysis
    • Historical business insights
    • Native Streamlit data explorer
    • CSV download

DATA SOURCE
-----------
The application automatically searches beside this file for:

    1. trending_videos.db
       Preferred table: trending_videos

    2. trending_videos.csv
    3. trending_videos_data.csv

No CSV uploader is included.

RUN
---
    streamlit run app.py

DEPENDENCIES
------------
    pip install streamlit pandas numpy plotly

AUTHOR
------
S Mohammed Kaif
Data Science • Data Analytics • Machine Learning • Business Intelligence

===============================================================================
"""

from __future__ import annotations

import ast
import html
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# 01. PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="YouTube Analytics Dashboard",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# 02. APPLICATION CONSTANTS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_FILE = BASE_DIR / "trending_videos.db"

CSV_FILES = [
    BASE_DIR / "trending_videos.csv",
    BASE_DIR / "trending_videos_data.csv",
]

APP_TITLE = "YouTube Analytics Dashboard"

APP_SUBTITLE = (
    "Global Overview • Historical Video Performance & Audience Insights"
)

AUTHOR_NAME = "S Mohammed Kaif"

AUTHOR_ROLE = (
    "Data Science • Data Analytics • Machine Learning • Business Intelligence"
)

GITHUB_URL = "https://github.com/Shaik-Mohammed-Kaif"

LINKEDIN_URL = (
    "https://www.linkedin.com/in/s-mohammed-kaif-2a500a341"
)

RED = "#C9151E"
RED_DARK = "#A90F17"
RED_LIGHT = "#FFF0F1"

BLACK = "#171717"
TEXT = "#252525"
MUTED = "#707070"

WHITE = "#FFFFFF"
PAGE = "#F7F7F7"
CARD = "#FFFFFF"

BORDER = "#DCDCDC"
GRID = "#EAEAEA"

SOFT_RED = "#F6D4D6"
MEDIUM_RED = "#E96A70"

PLOTLY_FONT = "Arial, Helvetica, sans-serif"


# =============================================================================
# 03. SESSION STATE
# =============================================================================

if "active_page" not in st.session_state:
    st.session_state.active_page = "Overview"

if "selected_video" not in st.session_state:
    st.session_state.selected_video = "All"

if "selected_period" not in st.session_state:
    st.session_state.selected_period = "All"

if "selected_category" not in st.session_state:
    st.session_state.selected_category = "All"

if "selected_channel" not in st.session_state:
    st.session_state.selected_channel = "All"


# =============================================================================
# 04. GLOBAL THEME
# =============================================================================

st.markdown(
    f"""
<style>
/* ============================================================================
   GLOBAL PAGE
   ============================================================================ */

.stApp {{
    background: {PAGE};
    color: {TEXT};
}}

.block-container {{
    max-width: 1600px;
    padding-top: 0.65rem;
    padding-bottom: 2rem;
}}

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

header {{
    background: transparent !important;
}}

/* ============================================================================
   REFERENCE-INSPIRED TOP BAR
   ============================================================================ */

.reference-header {{
    background:
        linear-gradient(
            90deg,
            {RED_DARK} 0%,
            {RED} 55%,
            #DD252C 100%
        );
    color: white;
    border-radius: 4px;
    min-height: 82px;
    padding: 15px 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,.12);
}}

.reference-logo {{
    width: 43px;
    height: 31px;
    border-radius: 7px;
    background: white;
    color: {RED};
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    font-size: 18px;
    vertical-align: middle;
}}

.reference-title {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 27px;
    font-weight: 850;
    line-height: 1.05;
    margin-left: 10px;
    vertical-align: middle;
}}

.reference-subtitle {{
    font-size: 10px;
    opacity: .92;
    margin-left: 55px;
    margin-top: 5px;
}}

.source-text {{
    text-align: right;
    font-size: 9px;
    line-height: 1.55;
}}

.source-text strong {{
    color: white;
}}

/* ============================================================================
   SECTION TITLE
   ============================================================================ */

.section-title {{
    color: {RED_DARK};
    font-size: 17px;
    font-weight: 850;
    margin-top: 9px;
    margin-bottom: 7px;
}}

/* ============================================================================
   KPI CARD
   ============================================================================ */

.kpi-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-top: 5px solid {RED};
    border-radius: 3px;
    min-height: 103px;
    padding: 11px;
    box-shadow: 0 1px 5px rgba(0,0,0,.07);
    transition:
        transform .18s ease,
        box-shadow .18s ease;
}}

.kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 7px 18px rgba(201,21,30,.14);
}}

.kpi-icon {{
    width: 35px;
    height: 35px;
    border-radius: 3px;
    background: {RED};
    color: white;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
    float: left;
    margin-right: 8px;
}}

.kpi-label {{
    color: {MUTED};
    font-size: 9px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .03em;
}}

.kpi-value {{
    color: {BLACK};
    font-size: 22px;
    line-height: 1.1;
    font-weight: 850;
    margin-top: 5px;
}}

.kpi-note {{
    clear: both;
    color: {MUTED};
    font-size: 8.5px;
    padding-top: 6px;
}}

/* ============================================================================
   NATIVE STREAMLIT WIDGET POLISH
   ============================================================================ */

div[data-testid="stButton"] button {{
    border-radius: 3px;
    font-weight: 750;
    min-height: 34px;
}}

div[data-testid="stButton"] button[kind="primary"] {{
    background: {RED};
    border-color: {RED};
    color: white;
}}

div[data-testid="stButton"] button[kind="primary"]:hover {{
    background: {RED_DARK};
    border-color: {RED_DARK};
    color: white;
}}

div[data-testid="stLinkButton"] a {{
    border-radius: 3px;
    font-weight: 750;
}}

div[data-testid="stLinkButton"] a:hover {{
    border-color: {RED};
    color: {RED};
}}

div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label {{
    color: {RED_DARK};
    font-size: 10px;
    font-weight: 800;
}}

div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
}}

div[data-testid="stExpander"] {{
    border-color: {BORDER};
    background: white;
}}

/* ============================================================================
   FOOTER
   ============================================================================ */

.footer-rule {{
    height: 4px;
    background: {RED};
    border-radius: 2px;
    margin-top: 16px;
}}

.footer-caption {{
    color: {MUTED};
    font-size: 9px;
    text-align: center;
    margin-top: 7px;
}}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# 05. GENERAL HELPERS
# =============================================================================

def safe_numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Return a numeric Series.

    Missing columns produce an empty Series aligned to the supplied frame.
    """
    if column not in frame.columns:
        return pd.Series(index=frame.index, dtype="float64")

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def existing_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    Return the first available column from candidates.
    """
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    return None


def human_number(value) -> str:
    """
    Format large numbers into compact business-friendly notation.
    """
    if value is None:
        return "—"

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"

    if np.isnan(number):
        return "—"

    sign = "-" if number < 0 else ""
    number = abs(number)

    if number >= 1_000_000_000:
        return f"{sign}{number / 1_000_000_000:.2f}B"

    if number >= 1_000_000:
        return f"{sign}{number / 1_000_000:.2f}M"

    if number >= 1_000:
        return f"{sign}{number / 1_000:.1f}K"

    return f"{sign}{number:,.0f}"


def percentage(value) -> str:
    """
    Display either decimal or percentage-style rates correctly.
    """
    if value is None:
        return "—"

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"

    if np.isnan(number):
        return "—"

    if abs(number) <= 1:
        number *= 100

    return f"{number:.2f}%"


def short_text(
    value,
    length: int = 70,
) -> str:
    """
    Make long titles safe for visual tables.
    """
    if value is None:
        return "—"

    text = " ".join(str(value).split())

    if len(text) <= length:
        return text

    return text[: length - 1] + "…"


def escape_text(value) -> str:
    """
    Escape text used inside the few CSS-only decorative elements.
    """
    return html.escape(str(value))


def parse_tags(value) -> list[str]:
    """
    Safely parse Python-list-looking tag fields.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if not isinstance(value, str):
        return []

    text = value.strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

    except (ValueError, SyntaxError):
        pass

    # Fallback for simple delimiter-based tag strings.
    for separator in ["|", ","]:
        if separator in text:
            return [
                item.strip()
                for item in text.split(separator)
                if item.strip()
            ]

    return [text]


# =============================================================================
# 06. DATA SOURCE HELPERS
# =============================================================================

def sqlite_table_names(
    connection: sqlite3.Connection,
) -> list[str]:
    """
    Return user tables from SQLite.
    """
    query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
    """

    tables = pd.read_sql_query(
        query,
        connection,
    )

    if tables.empty:
        return []

    return tables["name"].astype(str).tolist()


def read_sqlite_database(
    path: Path,
) -> tuple[pd.DataFrame | None, str]:
    """
    Read the preferred SQLite table, then gracefully fall back to another
    non-system table.
    """
    try:
        with sqlite3.connect(path) as connection:
            tables = sqlite_table_names(connection)

            if "trending_videos" in tables:
                data = pd.read_sql_query(
                    'SELECT * FROM "trending_videos"',
                    connection,
                )

                if not data.empty:
                    return data, "trending_videos.db • trending_videos"

            for table in tables:
                if table.startswith("sqlite_"):
                    continue

                data = pd.read_sql_query(
                    f'SELECT * FROM "{table}"',
                    connection,
                )

                if not data.empty:
                    return data, f"trending_videos.db • {table}"

    except Exception:
        return None, "SQLite read failed"

    return None, "SQLite contains no usable data"


def read_csv_file(
    path: Path,
) -> tuple[pd.DataFrame | None, str]:
    """
    Read a CSV file.
    """
    try:
        data = pd.read_csv(path)

        if not data.empty:
            return data, path.name

    except Exception:
        return None, f"{path.name} read failed"

    return None, f"{path.name} is empty"


@st.cache_data(
    show_spinner="Loading YouTube analytics dataset..."
)
def load_dataset() -> tuple[pd.DataFrame | None, str]:
    """
    Automatically load the project's database first and then CSV fallback.
    """
    if DATABASE_FILE.exists():
        data, source = read_sqlite_database(DATABASE_FILE)

        if data is not None and not data.empty:
            return data, source

    for csv_file in CSV_FILES:
        if csv_file.exists():
            data, source = read_csv_file(csv_file)

            if data is not None and not data.empty:
                return data, source

    return None, "No compatible dataset found"


# =============================================================================
# 07. DATA PREPARATION
# =============================================================================

def prepare_dataset(
    original: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare a clean analytics frame without changing the original source file.
    """
    frame = original.copy()

    frame = frame.drop_duplicates().reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Numeric fields.
    # -------------------------------------------------------------------------

    numeric_columns = [
        "view_count",
        "like_count",
        "comment_count",
        "favorite_count",
        "share_count",
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
        "hashtag_count",
        "url_count",
        "publish_hour",
        "publish_year",
        "publish_month",
        "publish_day",
        "publish_week",
        "channel_video_count",
        "is_hd",
        "has_caption",
        "is_weekend",
    ]

    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    # -------------------------------------------------------------------------
    # Date.
    # -------------------------------------------------------------------------

    date_column = existing_column(
        frame,
        [
            "published_at",
            "published_at_date",
            "publish_date",
            "published_date",
        ],
    )

    if date_column:
        frame[date_column] = pd.to_datetime(
            frame[date_column],
            errors="coerce",
        )

        frame["_analysis_date"] = frame[date_column]

        if "publish_year" not in frame.columns:
            frame["publish_year"] = (
                frame["_analysis_date"].dt.year
            )

        if "publish_month" not in frame.columns:
            frame["publish_month"] = (
                frame["_analysis_date"].dt.month
            )

        if "publish_month_name" not in frame.columns:
            frame["publish_month_name"] = (
                frame["_analysis_date"].dt.month_name()
            )

        if "publish_day_name" not in frame.columns:
            frame["publish_day_name"] = (
                frame["_analysis_date"].dt.day_name()
            )

        if "publish_hour" not in frame.columns:
            frame["publish_hour"] = (
                frame["_analysis_date"].dt.hour
            )

        if "publish_week" not in frame.columns:
            frame["publish_week"] = (
                frame["_analysis_date"]
                .dt.isocalendar()
                .week
                .astype("float")
            )

        if "is_weekend" not in frame.columns:
            frame["is_weekend"] = (
                frame["_analysis_date"]
                .dt.dayofweek
                .ge(5)
            )

    # -------------------------------------------------------------------------
    # Engagement rate.
    # -------------------------------------------------------------------------

    if "view_count" in frame.columns:
        views = safe_numeric(
            frame,
            "view_count",
        ).replace(0, np.nan)

        if "like_rate" not in frame.columns:
            if "like_count" in frame.columns:
                frame["like_rate"] = (
                    safe_numeric(frame, "like_count")
                    / views
                )

        if "comment_rate" not in frame.columns:
            if "comment_count" in frame.columns:
                frame["comment_rate"] = (
                    safe_numeric(frame, "comment_count")
                    / views
                )

        if "engagement_rate" not in frame.columns:
            engagement = pd.Series(
                0.0,
                index=frame.index,
            )

            for column in [
                "like_count",
                "comment_count",
                "favorite_count",
                "share_count",
            ]:
                if column in frame.columns:
                    engagement = (
                        engagement
                        + safe_numeric(
                            frame,
                            column,
                        ).fillna(0)
                    )

            frame["engagement_rate"] = (
                engagement / views
            )

    # -------------------------------------------------------------------------
    # Weekend label.
    # -------------------------------------------------------------------------

    if "is_weekend" in frame.columns:
        if "weekend_label" not in frame.columns:
            weekend = safe_numeric(
                frame,
                "is_weekend",
            ).eq(1)

            if frame["is_weekend"].dtype == bool:
                weekend = frame["is_weekend"]

            frame["weekend_label"] = np.where(
                weekend,
                "Weekend",
                "Weekday",
            )

    # -------------------------------------------------------------------------
    # Quality labels.
    # -------------------------------------------------------------------------

    if "is_hd" in frame.columns:
        frame["hd_label"] = np.where(
            safe_numeric(frame, "is_hd").eq(1),
            "HD",
            "SD",
        )

    if "has_caption" in frame.columns:
        frame["caption_label"] = np.where(
            safe_numeric(frame, "has_caption").eq(1),
            "Captioned",
            "No Captions",
        )

    # -------------------------------------------------------------------------
    # Tags.
    # -------------------------------------------------------------------------

    if "tags" in frame.columns:
        frame["tags_list"] = frame["tags"].apply(
            parse_tags
        )

    # -------------------------------------------------------------------------
    # Basic text-derived features if absent.
    # -------------------------------------------------------------------------

    if "title" in frame.columns:
        title_text = frame["title"].fillna("").astype(str)

        if "title_length" not in frame.columns:
            frame["title_length"] = (
                title_text.str.len()
            )

        if "title_word_count" not in frame.columns:
            frame["title_word_count"] = (
                title_text.str.split().str.len()
            )

    return frame


# =============================================================================
# 08. PLOTLY THEME HELPERS
# =============================================================================

def style_figure(
    figure: go.Figure,
    height: int = 350,
) -> go.Figure:
    """
    Apply a consistent professional BI-style Plotly theme.
    """
    figure.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(
            family=PLOTLY_FONT,
            size=10,
            color=TEXT,
        ),
        margin=dict(
            l=45,
            r=20,
            t=44,
            b=42,
        ),
        title=dict(
            font=dict(
                family=PLOTLY_FONT,
                size=12,
                color=RED_DARK,
            ),
            x=0.01,
        ),
        legend=dict(
            font=dict(size=9),
            bgcolor="rgba(255,255,255,.82)",
        ),
        hoverlabel=dict(
            bgcolor=BLACK,
            font_color=WHITE,
        ),
    )

    figure.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        linecolor=BORDER,
    )

    figure.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        linecolor=BORDER,
    )

    return figure


def show_figure(
    figure: go.Figure,
    key: str,
) -> None:
    """
    Render Plotly through the native Streamlit chart component.
    """
    st.plotly_chart(
        figure,
        width="stretch",
        key=key,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": False,
        },
    )



# =============================================================================
# 09. PAGE ROUTING
# =============================================================================

def render_overview():
    """
    Main historical YouTube analytics overview.

    This is descriptive/diagnostic analytics only.
    """
    data, source_name = load_dataset()

    if data is None:
        st.error(
            "Dataset not found. Place `trending_videos.db`, "
            "`trending_videos.csv`, or `trending_videos_data.csv` "
            "beside this Python file."
        )
        st.info(
            "CSV upload has intentionally been removed. "
            "The application reads the project data source automatically."
        )
        st.stop()

    df = prepare_dataset(data)

    if df.empty:
        st.error("The loaded dataset has no usable rows.")
        st.stop()

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------

    header_left, header_right = st.columns(
        [3.5, 1.5],
        vertical_alignment="center",
    )

    with header_left:
        st.markdown(
            f"""
<div class="reference-header">
    <span class="reference-logo">▶</span>
    <span class="reference-title">{escape_text(APP_TITLE)}</span>
    <div class="reference-subtitle">
        {escape_text(APP_SUBTITLE)}
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with header_right:
        st.markdown(
            f"""
<div class="reference-header">
    <div class="source-text">
        <div>Data Source</div>
        <strong>{escape_text(source_name)}</strong>
        <div>
            Updated {datetime.now().strftime("%d %b %Y • %I:%M %p")}
        </div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Overview controls
    # -------------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Global Overview</div>',
        unsafe_allow_html=True,
    )

    control1, control2, control3, control4 = st.columns(
        [1.4, 1.05, 1.1, 1.25],
        gap="small",
    )

    with control1:
        video_options = ["All"]

        if "title" in df.columns:
            video_values = (
                df["title"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            video_options += video_values[:500]

        current_video = st.session_state.get(
            "selected_video",
            "All",
        )

        if current_video not in video_options:
            current_video = "All"

        st.session_state.selected_video = st.selectbox(
            "Select a Video",
            video_options,
            index=video_options.index(current_video),
            key="overview_video_filter",
        )

    with control2:
        period_options = ["All"]

        if "publish_year" in df.columns:
            year_values = (
                safe_numeric(
                    df,
                    "publish_year",
                )
                .dropna()
                .astype(int)
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            period_options += [
                str(year)
                for year in year_values
            ]

        current_period = st.session_state.get(
            "selected_period",
            "All",
        )

        if current_period not in period_options:
            current_period = "All"

        st.session_state.selected_period = st.selectbox(
            "Select a Period",
            period_options,
            index=period_options.index(current_period),
            key="overview_period_filter",
        )

    with control3:
        category_options = ["All"]

        if "category_name" in df.columns:
            category_options += sorted(
                df["category_name"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        current_category = st.session_state.get(
            "selected_category",
            "All",
        )

        if current_category not in category_options:
            current_category = "All"

        st.session_state.selected_category = st.selectbox(
            "Category View",
            category_options,
            index=category_options.index(current_category),
            key="overview_category_filter",
        )

    with control4:
        channel_options = ["All"]

        if "channel_title" in df.columns:
            channel_options += sorted(
                df["channel_title"]
                .dropna()
                .astype(str)
                .value_counts()
                .head(250)
                .index
                .tolist()
            )

        current_channel = st.session_state.get(
            "selected_channel",
            "All",
        )

        if current_channel not in channel_options:
            current_channel = "All"

        st.session_state.selected_channel = st.selectbox(
            "Channel View",
            channel_options,
            index=channel_options.index(current_channel),
            key="overview_channel_filter",
        )

    # -------------------------------------------------------------------------
    # Apply filters
    # -------------------------------------------------------------------------

    filtered = df.copy()

    if (
        st.session_state.selected_video != "All"
        and "title" in filtered.columns
    ):
        filtered = filtered[
            filtered["title"].astype(str)
            == st.session_state.selected_video
        ]

    if (
        st.session_state.selected_period != "All"
        and "publish_year" in filtered.columns
    ):
        filtered = filtered[
            safe_numeric(
                filtered,
                "publish_year",
            ).eq(
                int(st.session_state.selected_period)
            )
        ]

    if (
        st.session_state.selected_category != "All"
        and "category_name" in filtered.columns
    ):
        filtered = filtered[
            filtered["category_name"].astype(str)
            == st.session_state.selected_category
        ]

    if (
        st.session_state.selected_channel != "All"
        and "channel_title" in filtered.columns
    ):
        filtered = filtered[
            filtered["channel_title"].astype(str)
            == st.session_state.selected_channel
        ]

    if filtered.empty:
        st.warning(
            "No records match the current selections. "
            "Choose broader values from the controls above."
        )
        return

   # -------------------------------------------------------------------------
    # KPI calculations
    # -------------------------------------------------------------------------
    
    total_videos = len(filtered)
    
    total_views = (
        safe_numeric(
            filtered,
            "view_count",
        ).sum()
        if "view_count" in filtered.columns
        else None
    )
    
    total_likes = (
        safe_numeric(
            filtered,
            "like_count",
        ).sum()
        if "like_count" in filtered.columns
        else None
    )
    
    total_comments = (
        safe_numeric(
            filtered,
            "comment_count",
        ).sum()
        if "comment_count" in filtered.columns
        else None
    )
    
    average_engagement = (
        safe_numeric(
            filtered,
            "engagement_rate",
        ).mean()
        if "engagement_rate" in filtered.columns
        else None
    )
    
    average_views = (
        safe_numeric(
            filtered,
            "view_count",
        ).mean()
        if "view_count" in filtered.columns
        else None
    )
    
    
    # -------------------------------------------------------------------------
    # KPI display values
    # -------------------------------------------------------------------------
    
    total_videos_value = human_number(
        total_videos
    )
    
    total_views_value = (
        human_number(total_views)
        if total_views is not None
        else "N/A"
    )
    
    total_likes_value = (
        human_number(total_likes)
        if total_likes is not None
        else "N/A"
    )
    
    total_comments_value = (
        human_number(total_comments)
        if total_comments is not None
        else "N/A"
    )
    
    average_engagement_value = (
        percentage(average_engagement)
        if average_engagement is not None
        else "N/A"
    )
    
    average_views_value = (
        human_number(average_views)
        if average_views is not None
        else "N/A"
    )
    
    
    # -------------------------------------------------------------------------
    # KPI data
    # -------------------------------------------------------------------------
    
    kpi_data = [
        (
            "👥",
            "Total Videos",
            total_videos_value,
            "records",
        ),
        (
            "▶",
            "Total Views",
            total_views_value,
            "historical views",
        ),
        (
            "👍",
            "Total Likes",
            total_likes_value,
            "historical likes",
        ),
        (
            "💬",
            "Total Comments",
            total_comments_value,
            "historical comments",
        ),
        (
            "▥",
            "Avg Engagement Rate",
            average_engagement_value,
            "dataset average",
        ),
        (
            "👁",
            "Avg Views / Video",
            average_views_value,
            "average per video",
        ),
    ]
    
    
    # -------------------------------------------------------------------------
    # KPI rendering
    # -------------------------------------------------------------------------
    
    kpi_columns = st.columns(
        6,
        gap="small",
    )
    
    for kpi_column, values in zip(
        kpi_columns,
        kpi_data,
    ):
    
        icon, label, value, note = values
    
        with kpi_column:
    
            st.markdown(
                f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{escape_text(label)}</div>
        <div class="kpi-value">{escape_text(value)}</div>
        <div class="kpi-note">{escape_text(note)}</div>
    </div>
    """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # Performance story
    # -------------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Performance Story</div>',
        unsafe_allow_html=True,
    )

    story1, story2, story3 = st.columns(
        [1, 1.15, 1.55],
        gap="small",
    )

    with story1:
        with st.container(border=True):
            st.subheader("Engagement Distribution")

            engagement_rows = []

            for field, label in [
                ("like_count", "Likes"),
                ("comment_count", "Comments"),
                ("favorite_count", "Favorites"),
                ("share_count", "Shares"),
            ]:
                if field in filtered.columns:
                    value = safe_numeric(
                        filtered,
                        field,
                    ).sum()

                    if value > 0:
                        engagement_rows.append(
                            {
                                "Metric": label,
                                "Value": value,
                            }
                        )

            if engagement_rows:
                engagement_frame = pd.DataFrame(
                    engagement_rows
                )

                figure = px.pie(
                    engagement_frame,
                    names="Metric",
                    values="Value",
                    hole=.56,
                )

                figure.update_traces(
                    marker=dict(
                        colors=[
                            RED,
                            "#E04C53",
                            "#EC858A",
                            "#F2B7BA",
                        ][: len(engagement_frame)]
                    ),
                    textinfo="percent",
                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Value: %{value:,.0f}<br>"
                        "Share: %{percent}"
                        "<extra></extra>"
                    ),
                )

                figure.update_layout(
                    height=320,
                    margin=dict(
                        l=4,
                        r=4,
                        t=12,
                        b=4,
                    ),
                    legend=dict(
                        font=dict(size=8)
                    ),
                )

                show_figure(
                    figure,
                    "overview_engagement_distribution",
                )
            else:
                st.info(
                    "No engagement fields are available."
                )

    with story2:
        with st.container(border=True):
            st.subheader("Monthly Views Over Time")

            date_column = existing_column(
                filtered,
                [
                    "published_at",
                    "published_at_date",
                    "publish_date",
                    "published_date",
                ],
            )

            if (
                date_column
                and "view_count" in filtered.columns
            ):
                trend = filtered[
                    [
                        date_column,
                        "view_count",
                    ]
                ].copy()

                trend[date_column] = pd.to_datetime(
                    trend[date_column],
                    errors="coerce",
                )

                trend["view_count"] = safe_numeric(
                    trend,
                    "view_count",
                )

                trend = trend.dropna()

                if not trend.empty:
                    monthly = (
                        trend
                        .set_index(date_column)
                        .resample("MS")["view_count"]
                        .sum()
                        .reset_index()
                    )

                    figure = go.Figure()

                    figure.add_trace(
                        go.Scatter(
                            x=monthly[date_column],
                            y=monthly["view_count"],
                            mode="lines+markers",
                            line=dict(
                                color=RED_DARK,
                                width=3,
                            ),
                            marker=dict(
                                color=RED,
                                size=6,
                            ),
                            fill="tozeroy",
                            fillcolor="rgba(201,21,30,.13)",
                            hovertemplate=(
                                "<b>%{x|%b %Y}</b><br>"
                                "Views: %{y:,.0f}"
                                "<extra></extra>"
                            ),
                        )
                    )

                    figure.update_layout(
                        title="Historical Views Trend",
                        xaxis_title="Month",
                        yaxis_title="Views",
                    )

                    show_figure(
                        style_figure(
                            figure,
                            320,
                        ),
                        "overview_monthly_views",
                    )
                else:
                    st.info(
                        "No valid publishing dates."
                    )
            else:
                st.info(
                    "A publishing date and view_count "
                    "are required."
                )

    with story3:
        with st.container(border=True):
            st.subheader(
                "Top Performing Videos by Views"
            )

            st.caption(
                "Detailed breakdown of historical video performance"
            )

            if {
                "title",
                "view_count",
            }.issubset(filtered.columns):
                top_videos = filtered.copy()

                top_videos["view_count"] = safe_numeric(
                    top_videos,
                    "view_count",
                )

                top_videos = (
                    top_videos
                    .dropna(
                        subset=["view_count"]
                    )
                    .sort_values(
                        "view_count",
                        ascending=False,
                    )
                    .head(10)
                )

                if not top_videos.empty:
                    display = pd.DataFrame()

                    display["Video"] = (
                        top_videos["title"]
                        .map(
                            lambda value: short_text(
                                value,
                                72,
                            )
                        )
                    )

                    if "channel_title" in top_videos.columns:
                        display["Channel"] = (
                            top_videos["channel_title"]
                            .map(
                                lambda value: short_text(
                                    value,
                                    28,
                                )
                            )
                        )

                    display["Views"] = (
                        top_videos["view_count"]
                        .map(human_number)
                    )

                    st.dataframe(
                        display,
                        width="stretch",
                        hide_index=True,
                        height=365,
                    )
                else:
                    st.info(
                        "No view records are available."
                    )
            else:
                st.info(
                    "title and view_count are required."
                )

    # -------------------------------------------------------------------------
    # Publishing performance
    # -------------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">When should you publish a video?</div>',
        unsafe_allow_html=True,
    )

    publish_left, publish_right = st.columns(
        [1.7, 1],
        gap="small",
    )

    with publish_left:
        with st.container(border=True):
            st.subheader(
                "Publishing Performance by Day & Month"
            )

            st.caption(
                "Monthly comparison of total views across each day of the week"
            )

            date_column = existing_column(
                filtered,
                [
                    "published_at",
                    "published_at_date",
                    "publish_date",
                    "published_date",
                ],
            )

            if (
                date_column
                and "view_count" in filtered.columns
            ):
                heat_data = filtered[
                    [
                        date_column,
                        "view_count",
                    ]
                ].copy()

                heat_data[date_column] = pd.to_datetime(
                    heat_data[date_column],
                    errors="coerce",
                )

                heat_data["view_count"] = safe_numeric(
                    heat_data,
                    "view_count",
                )

                heat_data["Day"] = (
                    heat_data[date_column]
                    .dt.day_name()
                )

                heat_data["Month"] = (
                    heat_data[date_column]
                    .dt.month_name()
                )

                heat_data = heat_data.dropna()

                if not heat_data.empty:
                    day_order = [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday",
                    ]

                    month_order = [
                        "January",
                        "February",
                        "March",
                        "April",
                        "May",
                        "June",
                        "July",
                        "August",
                        "September",
                        "October",
                        "November",
                        "December",
                    ]

                    grouped_heat = (
                        heat_data
                        .groupby(
                            [
                                "Day",
                                "Month",
                            ],
                            as_index=False,
                        )["view_count"]
                        .sum()
                    )

                    pivot = (
                        grouped_heat
                        .pivot(
                            index="Day",
                            columns="Month",
                            values="view_count",
                        )
                        .reindex(
                            index=day_order,
                            columns=month_order,
                        )
                    )

                    available_months = [
                        month
                        for month in month_order
                        if (
                            month in pivot.columns
                            and pivot[month]
                            .notna()
                            .any()
                        )
                    ]

                    if available_months:
                        pivot = pivot[
                            available_months
                        ]

                        figure = go.Figure(
                            data=go.Heatmap(
                                z=pivot.values,
                                x=pivot.columns,
                                y=pivot.index,
                                colorscale=[
                                    [0, "#FFF6F6"],
                                    [.30, "#F8D5D7"],
                                    [.65, "#EA858A"],
                                    [1, RED_DARK],
                                ],
                                hovertemplate=(
                                    "<b>%{y}</b><br>"
                                    "%{x}<br>"
                                    "Views: %{z:,.0f}"
                                    "<extra></extra>"
                                ),
                                colorbar=dict(
                                    title="Views",
                                    thickness=11,
                                ),
                            )
                        )

                        figure.update_layout(
                            height=365,
                            margin=dict(
                                l=55,
                                r=10,
                                t=15,
                                b=58,
                            ),
                            paper_bgcolor=WHITE,
                            plot_bgcolor=WHITE,
                            font=dict(
                                family=PLOTLY_FONT,
                                size=9,
                                color=TEXT,
                            ),
                            xaxis=dict(
                                title="Month",
                                tickangle=-35,
                            ),
                            yaxis=dict(
                                title="Day",
                            ),
                        )

                        show_figure(
                            figure,
                            "overview_publish_heatmap",
                        )
                    else:
                        st.info(
                            "No month-level publishing data."
                        )
                else:
                    st.info(
                        "No valid publishing records."
                    )
            else:
                st.info(
                    "Publishing date and view_count are required."
                )

    with publish_right:
        with st.container(border=True):
            st.subheader(
                "Historical Performance Insights"
            )

            if {
                "category_name",
                "view_count",
            }.issubset(filtered.columns):
                category_perf = (
                    filtered
                    .assign(
                        _views=safe_numeric(
                            filtered,
                            "view_count",
                        )
                    )
                    .groupby(
                        "category_name"
                    )["_views"]
                    .sum()
                    .dropna()
                )

                if not category_perf.empty:
                    st.metric(
                        "Top Category",
                        short_text(
                            category_perf.idxmax(),
                            30,
                        ),
                    )

            if {
                "channel_title",
                "view_count",
            }.issubset(filtered.columns):
                channel_perf = (
                    filtered
                    .assign(
                        _views=safe_numeric(
                            filtered,
                            "view_count",
                        )
                    )
                    .groupby(
                        "channel_title"
                    )["_views"]
                    .sum()
                    .dropna()
                )

                if not channel_perf.empty:
                    st.metric(
                        "Top Channel",
                        short_text(
                            channel_perf.idxmax(),
                            30,
                        ),
                    )

            if (
                "view_count" in filtered.columns
                and "publish_day_name" in filtered.columns
            ):
                day_perf = (
                    filtered
                    .assign(
                        _views=safe_numeric(
                            filtered,
                            "view_count",
                        )
                    )
                    .groupby(
                        "publish_day_name"
                    )["_views"]
                    .mean()
                    .dropna()
                )

                if not day_perf.empty:
                    st.metric(
                        "Best Avg-View Day",
                        str(day_perf.idxmax()),
                    )

            if (
                "view_count" in filtered.columns
                and "publish_hour" in filtered.columns
            ):
                hour_frame = filtered.copy()

                hour_frame["publish_hour"] = safe_numeric(
                    hour_frame,
                    "publish_hour",
                )

                hour_frame["_views"] = safe_numeric(
                    hour_frame,
                    "view_count",
                )

                hour_perf = (
                    hour_frame
                    .groupby(
                        "publish_hour"
                    )["_views"]
                    .mean()
                    .dropna()
                )

                if not hour_perf.empty:
                    st.metric(
                        "Best Avg-View Hour",
                        f"{int(hour_perf.idxmax()):02d}:00",
                    )

    # -------------------------------------------------------------------------
    # Audience engagement
    # -------------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Audience Engagement Analysis</div>',
        unsafe_allow_html=True,
    )

    engagement1, engagement2, engagement3 = st.columns(
        3,
        gap="small",
    )

    with engagement1:
        with st.container(border=True):
            st.subheader("Likes vs Views")

            if {
                "view_count",
                "like_count",
            }.issubset(filtered.columns):
                scatter_data = filtered[
                    [
                        "view_count",
                        "like_count",
                    ]
                ].copy()

                scatter_data["view_count"] = safe_numeric(
                    scatter_data,
                    "view_count",
                )

                scatter_data["like_count"] = safe_numeric(
                    scatter_data,
                    "like_count",
                )

                scatter_data = scatter_data.dropna()

                scatter_data = scatter_data[
                    (
                        scatter_data["view_count"]
                        > 0
                    )
                    & (
                        scatter_data["like_count"]
                        >= 0
                    )
                ]

                if not scatter_data.empty:
                    figure = px.scatter(
                        scatter_data,
                        x="view_count",
                        y="like_count",
                        opacity=.58,
                    )

                    figure.update_traces(
                        marker=dict(
                            color=RED,
                            size=7,
                        ),
                        hovertemplate=(
                            "Views: %{x:,.0f}<br>"
                            "Likes: %{y:,.0f}"
                            "<extra></extra>"
                        ),
                    )

                    figure.update_xaxes(
                        type="log",
                        title="Views",
                    )

                    figure.update_yaxes(
                        type="log",
                        title="Likes",
                    )

                    show_figure(
                        style_figure(
                            figure,
                            315,
                        ),
                        "overview_likes_views",
                    )
                else:
                    st.info(
                        "No usable like/view records."
                    )
            else:
                st.info(
                    "like_count and view_count are required."
                )

    with engagement2:
        with st.container(border=True):
            st.subheader(
                "Engagement Rate by Category"
            )

            if {
                "category_name",
                "engagement_rate",
            }.issubset(filtered.columns):
                category_engagement = filtered[
                    [
                        "category_name",
                        "engagement_rate",
                    ]
                ].copy()

                category_engagement[
                    "engagement_rate"
                ] = safe_numeric(
                    category_engagement,
                    "engagement_rate",
                )

                category_engagement = (
                    category_engagement
                    .dropna()
                )

                if not category_engagement.empty:
                    category_engagement["Rate"] = (
                        category_engagement[
                            "engagement_rate"
                        ]
                    )

                    if (
                        category_engagement["Rate"]
                        .abs()
                        .max()
                        <= 1
                    ):
                        category_engagement[
                            "Rate"
                        ] *= 100

                    grouped = (
                        category_engagement
                        .groupby(
                            "category_name"
                        )["Rate"]
                        .mean()
                        .sort_values(
                            ascending=False
                        )
                        .head(10)
                        .reset_index()
                    )

                    grouped = grouped.sort_values(
                        "Rate"
                    )

                    figure = px.bar(
                        grouped,
                        x="Rate",
                        y="category_name",
                        orientation="h",
                    )

                    figure.update_traces(
                        marker_color=RED,
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Avg Engagement: %{x:.2f}%"
                            "<extra></extra>"
                        ),
                    )

                    figure.update_layout(
                        xaxis_title="Engagement Rate (%)",
                        yaxis_title="",
                    )

                    show_figure(
                        style_figure(
                            figure,
                            315,
                        ),
                        "overview_engagement_category",
                    )
                else:
                    st.info(
                        "No engagement data."
                    )
            else:
                st.info(
                    "category_name and engagement_rate are required."
                )

    with engagement3:
        with st.container(border=True):
            st.subheader(
                "Views by Duration Category"
            )

            if {
                "duration_category",
                "view_count",
            }.issubset(filtered.columns):
                duration_data = filtered[
                    [
                        "duration_category",
                        "view_count",
                    ]
                ].copy()

                duration_data["view_count"] = safe_numeric(
                    duration_data,
                    "view_count",
                )

                duration_data = (
                    duration_data
                    .dropna()
                )

                if not duration_data.empty:
                    duration_group = (
                        duration_data
                        .groupby(
                            "duration_category"
                        )["view_count"]
                        .mean()
                        .sort_values(
                            ascending=False
                        )
                        .reset_index()
                    )

                    figure = px.bar(
                        duration_group,
                        x="duration_category",
                        y="view_count",
                    )

                    figure.update_traces(
                        marker_color=RED,
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "Average Views: %{y:,.0f}"
                            "<extra></extra>"
                        ),
                    )

                    figure.update_layout(
                        xaxis_title="Duration",
                        yaxis_title="Average Views",
                    )

                    show_figure(
                        style_figure(
                            figure,
                            315,
                        ),
                        "overview_duration",
                    )
                else:
                    st.info(
                        "No duration records."
                    )
            else:
                st.info(
                    "duration_category and view_count are required."
                )

    # -------------------------------------------------------------------------
    # Channel performance
    # -------------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Channel Performance</div>',
        unsafe_allow_html=True,
    )

    if {
        "channel_title",
        "view_count",
    }.issubset(filtered.columns):
        channel_table = filtered.copy()

        channel_table["_views"] = safe_numeric(
            channel_table,
            "view_count",
        )

        channel_table = (
            channel_table
            .groupby(
                "channel_title",
                as_index=False,
            )
            .agg(
                Videos=("_views", "count"),
                Total_Views=("_views", "sum"),
                Average_Views=("_views", "mean"),
            )
            .sort_values(
                "Total_Views",
                ascending=False,
            )
            .head(15)
        )

        channel_table["Total Views"] = (
            channel_table["Total_Views"]
            .map(human_number)
        )

        channel_table["Average Views"] = (
            channel_table["Average_Views"]
            .map(human_number)
        )

        channel_table["Channel"] = (
            channel_table["channel_title"]
            .map(
                lambda value: short_text(
                    value,
                    60,
                )
            )
        )

        st.dataframe(
            channel_table[
                [
                    "Channel",
                    "Videos",
                    "Total Views",
                    "Average Views",
                ]
            ],
            width="stretch",
            hide_index=True,
            height=325,
        )
    else:
        st.info(
            "channel_title and view_count are required."
        )

    # -------------------------------------------------------------------------
    # Advanced content analysis
    # -------------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Advanced Content Analysis</div>',
        unsafe_allow_html=True,
    )

    advanced1, advanced2 = st.columns(
        2,
        gap="small",
    )

    with advanced1:
        with st.container(border=True):
            st.subheader(
                "Title Length vs Views"
            )

            if {
                "title_length",
                "view_count",
            }.issubset(filtered.columns):
                title_data = filtered[
                    [
                        "title_length",
                        "view_count",
                    ]
                ].copy()

                title_data["title_length"] = safe_numeric(
                    title_data,
                    "title_length",
                )

                title_data["view_count"] = safe_numeric(
                    title_data,
                    "view_count",
                )

                title_data = title_data.dropna()

                if not title_data.empty:
                    figure = px.scatter(
                        title_data,
                        x="title_length",
                        y="view_count",
                        opacity=.58,
                    )

                    figure.update_traces(
                        marker=dict(
                            color=RED,
                            size=7,
                        )
                    )

                    figure.update_yaxes(
                        type="log",
                        title="Views",
                    )

                    figure.update_xaxes(
                        title="Title Length",
                    )

                    show_figure(
                        style_figure(
                            figure,
                            335,
                        ),
                        "overview_title_length",
                    )
                else:
                    st.info(
                        "No title-length records."
                    )
            else:
                st.info(
                    "title_length and view_count are required."
                )

    with advanced2:
        with st.container(border=True):
            st.subheader(
                "Title Word Count vs Engagement"
            )

            if {
                "title_word_count",
                "engagement_rate",
            }.issubset(filtered.columns):
                word_data = filtered[
                    [
                        "title_word_count",
                        "engagement_rate",
                    ]
                ].copy()

                word_data["title_word_count"] = safe_numeric(
                    word_data,
                    "title_word_count",
                )

                word_data["engagement_rate"] = safe_numeric(
                    word_data,
                    "engagement_rate",
                )

                word_data = word_data.dropna()

                if not word_data.empty:
                    word_data["Rate"] = (
                        word_data["engagement_rate"]
                    )

                    if (
                        word_data["Rate"]
                        .abs()
                        .max()
                        <= 1
                    ):
                        word_data["Rate"] *= 100

                    figure = px.scatter(
                        word_data,
                        x="title_word_count",
                        y="Rate",
                        opacity=.58,
                    )

                    figure.update_traces(
                        marker=dict(
                            color=RED,
                            size=7,
                        )
                    )

                    figure.update_layout(
                        xaxis_title="Title Word Count",
                        yaxis_title="Engagement Rate (%)",
                    )

                    show_figure(
                        style_figure(
                            figure,
                            335,
                        ),
                        "overview_title_words",
                    )
                else:
                    st.info(
                        "No title-word engagement data."
                    )
            else:
                st.info(
                    "title_word_count and engagement_rate are required."
                )

    # -------------------------------------------------------------------------
    # Tags
    # -------------------------------------------------------------------------

    tag1, tag2 = st.columns(
        2,
        gap="small",
    )

    with tag1:
        with st.container(border=True):
            st.subheader("Most Common Tags")

            if "tags_list" in filtered.columns:
                all_tags = []

                for tag_list in filtered["tags_list"]:
                    if isinstance(tag_list, list):
                        all_tags.extend(tag_list)

                if all_tags:
                    tag_counts = (
                        pd.Series(
                            all_tags,
                            dtype="object",
                        )
                        .value_counts()
                        .head(15)
                        .reset_index()
                    )

                    tag_counts.columns = [
                        "Tag",
                        "Occurrences",
                    ]

                    tag_counts = tag_counts.sort_values(
                        "Occurrences"
                    )

                    figure = px.bar(
                        tag_counts,
                        x="Occurrences",
                        y="Tag",
                        orientation="h",
                    )

                    figure.update_traces(
                        marker_color=RED
                    )

                    show_figure(
                        style_figure(
                            figure,
                            390,
                        ),
                        "overview_tags",
                    )
                else:
                    st.info(
                        "No tags are available."
                    )
            else:
                st.info(
                    "The dataset does not contain tags."
                )

    with tag2:
        with st.container(border=True):
            st.subheader(
                "Hashtag Count vs Average Views"
            )

            if {
                "hashtag_count",
                "view_count",
            }.issubset(filtered.columns):
                hashtag_data = filtered[
                    [
                        "hashtag_count",
                        "view_count",
                    ]
                ].copy()

                hashtag_data["hashtag_count"] = safe_numeric(
                    hashtag_data,
                    "hashtag_count",
                )

                hashtag_data["view_count"] = safe_numeric(
                    hashtag_data,
                    "view_count",
                )

                hashtag_data = hashtag_data.dropna()

                if not hashtag_data.empty:
                    grouped_hashtags = (
                        hashtag_data
                        .groupby(
                            "hashtag_count"
                        )["view_count"]
                        .mean()
                        .reset_index()
                    )

                    figure = px.bar(
                        grouped_hashtags,
                        x="hashtag_count",
                        y="view_count",
                    )

                    figure.update_traces(
                        marker_color=RED
                    )

                    figure.update_layout(
                        xaxis_title="Hashtag Count",
                        yaxis_title="Average Views",
                    )

                    show_figure(
                        style_figure(
                            figure,
                            390,
                        ),
                        "overview_hashtag_views",
                    )
                else:
                    st.info(
                        "No hashtag data."
                    )
            else:
                st.info(
                    "hashtag_count and view_count are required."
                )

    # -------------------------------------------------------------------------
    # Data download
    # -------------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">📥 Download / Inspect Analytics Data</div>',
        unsafe_allow_html=True,
    )

    with st.expander(
        "Open Data Explorer"
    ):
        st.caption(
            f"Current selection: {len(filtered):,} records "
            f"from {len(df):,} total records."
        )

        st.download_button(
            "⬇️ Download Current Analytics CSV",
            data=filtered.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="youtube_analytics_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

        available_columns = filtered.columns.tolist()

        default_columns = [
            column
            for column in [
                "title",
                "channel_title",
                "category_name",
                "view_count",
                "like_count",
                "comment_count",
                "engagement_rate",
                "duration_category",
                "publish_year",
                "publish_month",
                "publish_hour",
            ]
            if column in available_columns
        ]

        selected_columns = st.multiselect(
            "Columns to display",
            available_columns,
            default=default_columns,
            key="overview_explorer_columns",
        )

        st.dataframe(
            filtered[
                selected_columns
            ]
            if selected_columns
            else filtered,
            width="stretch",
            hide_index=True,
            height=420,
        )


def render_video_analytics():
    """
    Detailed historical video analytics page.
    """
    data, source_name = load_dataset()

    if data is None:
        st.error("Dataset not found.")
        return

    df = prepare_dataset(data)

    st.title("Video Analytics")
    st.caption(
        "Detailed historical performance analysis at video level."
    )

    metric_options = [
        "view_count",
        "like_count",
        "comment_count",
        "favorite_count",
        "engagement_rate",
        "like_rate",
        "comment_rate",
    ]

    available_metrics = [
        metric
        for metric in metric_options
        if metric in df.columns
    ]

    if not available_metrics:
        st.warning(
            "No supported performance metrics were found."
        )
        return

    selected_metric = st.selectbox(
        "Rank videos by",
        available_metrics,
        format_func=lambda value: (
            value.replace("_", " ").title()
        ),
        key="video_analytics_metric",
    )

    work = df.copy()

    work[selected_metric] = safe_numeric(
        work,
        selected_metric,
    )

    work = (
        work
        .dropna(
            subset=[selected_metric]
        )
        .sort_values(
            selected_metric,
            ascending=False,
        )
    )

    top_n = st.slider(
        "Number of videos",
        min_value=10,
        max_value=100,
        value=25,
        step=5,
        key="video_analytics_top_n",
    )

    ranked = work.head(top_n)

    st.subheader(
        f"Top {len(ranked)} Videos"
    )

    columns = [
        "title",
        "channel_title",
        "category_name",
        "view_count",
        "like_count",
        "comment_count",
        "engagement_rate",
        "duration_category",
        "publish_date",
    ]

    columns = [
        column
        for column in columns
        if column in ranked.columns
    ]

    st.dataframe(
        ranked[columns],
        width="stretch",
        hide_index=True,
        height=500,
    )

    st.subheader(
        "Performance Comparison"
    )

    if not ranked.empty:
        chart_data = ranked.copy()

        chart_data["Video"] = chart_data[
            "title"
        ].map(
            lambda value: short_text(
                value,
                45,
            )
        )

        figure = px.bar(
            chart_data,
            x=selected_metric,
            y="Video",
            orientation="h",
        )

        figure.update_traces(
            marker_color=RED,
            hovertemplate=(
                "<b>%{y}</b><br>"
                + selected_metric.replace("_", " ").title()
                + ": %{x:,.2f}"
                + "<extra></extra>"
            ),
        )

        figure.update_layout(
            yaxis=dict(
                categoryorder="total ascending"
            )
        )

        show_figure(
            style_figure(
                figure,
                max(
                    400,
                    min(900, 25 * len(ranked)),
                ),
            ),
            "video_analytics_rank_chart",
        )

    st.subheader(
        "Selected Video Profile"
    )

    video_names = ["Select a video"]

    if "title" in df.columns:
        video_names += (
            df["title"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(500)
            .tolist()
        )

    selected = st.selectbox(
        "Video",
        video_names,
        key="video_profile_selector",
    )

    if selected != "Select a video":
        record = df[
            df["title"].astype(str)
            == selected
        ]

        if not record.empty:
            row = record.iloc[0]

            profile_columns = st.columns(
                4,
                gap="small",
            )

            with profile_columns[0]:
                st.metric(
                    "Views",
                    human_number(
                        row.get(
                            "view_count",
                            np.nan,
                        )
                    ),
                )

            with profile_columns[1]:
                st.metric(
                    "Likes",
                    human_number(
                        row.get(
                            "like_count",
                            np.nan,
                        )
                    ),
                )

            with profile_columns[2]:
                st.metric(
                    "Comments",
                    human_number(
                        row.get(
                            "comment_count",
                            np.nan,
                        )
                    ),
                )

            with profile_columns[3]:
                st.metric(
                    "Engagement",
                    percentage(
                        row.get(
                            "engagement_rate",
                            np.nan,
                        )
                    ),
                )

            st.write(
                {
                    "Title": row.get(
                        "title",
                        "—",
                    ),
                    "Channel": row.get(
                        "channel_title",
                        "—",
                    ),
                    "Category": row.get(
                        "category_name",
                        "—",
                    ),
                    "Duration": row.get(
                        "duration_category",
                        "—",
                    ),
                    "Publish Date": row.get(
                        "publish_date",
                        row.get(
                            "published_at",
                            "—",
                        ),
                    ),
                }
            )


def render_insights():
    """
    Historical business insights page.
    """
    data, source_name = load_dataset()

    if data is None:
        st.error("Dataset not found.")
        return

    df = prepare_dataset(data)

    st.title("Insights")
    st.caption(
        "Business-oriented interpretation of observed historical YouTube data."
    )

    # -------------------------------------------------------------------------
    # Insight metrics
    # -------------------------------------------------------------------------

    insight_cols = st.columns(
        4,
        gap="small",
    )

    if {
        "category_name",
        "view_count",
    }.issubset(df.columns):
        category_views = (
            df.assign(
                _views=safe_numeric(
                    df,
                    "view_count",
                )
            )
            .groupby(
                "category_name"
            )["_views"]
            .sum()
            .dropna()
            .sort_values(
                ascending=False
            )
        )

        top_category = (
            category_views.index[0]
            if not category_views.empty
            else "—"
        )

        with insight_cols[0]:
            st.metric(
                "Top Category",
                short_text(
                    top_category,
                    25,
                ),
            )

    if {
        "channel_title",
        "view_count",
    }.issubset(df.columns):
        channel_views = (
            df.assign(
                _views=safe_numeric(
                    df,
                    "view_count",
                )
            )
            .groupby(
                "channel_title"
            )["_views"]
            .sum()
            .dropna()
            .sort_values(
                ascending=False
            )
        )

        top_channel = (
            channel_views.index[0]
            if not channel_views.empty
            else "—"
        )

        with insight_cols[1]:
            st.metric(
                "Top Channel",
                short_text(
                    top_channel,
                    25,
                ),
            )

    if {
        "publish_day_name",
        "view_count",
    }.issubset(df.columns):
        day_views = (
            df.assign(
                _views=safe_numeric(
                    df,
                    "view_count",
                )
            )
            .groupby(
                "publish_day_name"
            )["_views"]
            .mean()
            .dropna()
            .sort_values(
                ascending=False
            )
        )

        best_day = (
            day_views.index[0]
            if not day_views.empty
            else "—"
        )

        with insight_cols[2]:
            st.metric(
                "Best Avg-View Day",
                best_day,
            )

    if {
        "publish_hour",
        "view_count",
    }.issubset(df.columns):
        hour_frame = df.copy()

        hour_frame["publish_hour"] = safe_numeric(
            hour_frame,
            "publish_hour",
        )

        hour_frame["_views"] = safe_numeric(
            hour_frame,
            "view_count",
        )

        hour_views = (
            hour_frame
            .groupby(
                "publish_hour"
            )["_views"]
            .mean()
            .dropna()
            .sort_values(
                ascending=False
            )
        )

        best_hour = (
            f"{int(hour_views.index[0]):02d}:00"
            if not hour_views.empty
            else "—"
        )

        with insight_cols[3]:
            st.metric(
                "Best Avg-View Hour",
                best_hour,
            )

    # -------------------------------------------------------------------------
    # Category story
    # -------------------------------------------------------------------------

    st.subheader(
        "Category Performance Story"
    )

    if {
        "category_name",
        "view_count",
        "engagement_rate",
    }.issubset(df.columns):
        category_story = (
            df.assign(
                Views=safe_numeric(
                    df,
                    "view_count",
                ),
                Engagement=safe_numeric(
                    df,
                    "engagement_rate",
                ),
            )
            .groupby(
                "category_name",
                as_index=False,
            )
            .agg(
                Total_Views=("Views", "sum"),
                Average_Views=("Views", "mean"),
                Average_Engagement=(
                    "Engagement",
                    "mean",
                ),
                Videos=("Views", "count"),
            )
            .sort_values(
                "Total_Views",
                ascending=False,
            )
            .head(15)
        )

        category_story["Average Engagement"] = (
            category_story[
                "Average_Engagement"
            ].map(percentage)
        )

        category_story["Total Views"] = (
            category_story[
                "Total_Views"
            ].map(human_number)
        )

        category_story["Average Views"] = (
            category_story[
                "Average_Views"
            ].map(human_number)
        )

        st.dataframe(
            category_story[
                [
                    "category_name",
                    "Videos",
                    "Total Views",
                    "Average Views",
                    "Average Engagement",
                ]
            ].rename(
                columns={
                    "category_name": "Category"
                }
            ),
            width="stretch",
            hide_index=True,
        )

    # -------------------------------------------------------------------------
    # Channel concentration
    # -------------------------------------------------------------------------

    st.subheader(
        "Channel Performance Distribution"
    )

    if {
        "channel_title",
        "view_count",
    }.issubset(df.columns):
        channel_data = (
            df.assign(
                Views=safe_numeric(
                    df,
                    "view_count",
                )
            )
            .groupby(
                "channel_title",
                as_index=False,
            )
            .agg(
                Total_Views=("Views", "sum"),
                Videos=("Views", "count"),
            )
            .sort_values(
                "Total_Views",
                ascending=False,
            )
            .head(15)
        )

        figure = px.bar(
            channel_data,
            x="Total_Views",
            y="channel_title",
            orientation="h",
        )

        figure.update_traces(
            marker_color=RED,
        )

        figure.update_layout(
            yaxis=dict(
                categoryorder="total ascending"
            ),
            xaxis_title="Total Views",
            yaxis_title="Channel",
        )

        show_figure(
            style_figure(
                figure,
                500,
            ),
            "insights_channel_distribution",
        )

    # -------------------------------------------------------------------------
    # Publishing insights
    # -------------------------------------------------------------------------

    st.subheader(
        "Publishing-Time Insights"
    )

    publishing_left, publishing_right = st.columns(
        2,
        gap="small",
    )

    with publishing_left:
        if {
            "publish_day_name",
            "view_count",
        }.issubset(df.columns):
            day_data = (
                df.assign(
                    Views=safe_numeric(
                        df,
                        "view_count",
                    )
                )
                .groupby(
                    "publish_day_name",
                    as_index=False,
                )["Views"]
                .mean()
            )

            day_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]

            day_data["publish_day_name"] = pd.Categorical(
                day_data["publish_day_name"],
                categories=day_order,
                ordered=True,
            )

            day_data = day_data.sort_values(
                "publish_day_name"
            )

            figure = px.bar(
                day_data,
                x="publish_day_name",
                y="Views",
            )

            figure.update_traces(
                marker_color=RED
            )

            figure.update_layout(
                xaxis_title="Day",
                yaxis_title="Average Views",
            )

            show_figure(
                style_figure(
                    figure,
                    350,
                ),
                "insights_day_performance",
            )

    with publishing_right:
        if {
            "publish_hour",
            "view_count",
        }.issubset(df.columns):
            hour_data = (
                df.assign(
                    Hour=safe_numeric(
                        df,
                        "publish_hour",
                    ),
                    Views=safe_numeric(
                        df,
                        "view_count",
                    ),
                )
                .groupby(
                    "Hour",
                    as_index=False,
                )["Views"]
                .mean()
                .sort_values(
                    "Hour"
                )
            )

            figure = px.line(
                hour_data,
                x="Hour",
                y="Views",
                markers=True,
            )

            figure.update_traces(
                line=dict(
                    color=RED_DARK,
                    width=3,
                ),
                marker=dict(
                    color=RED,
                    size=6,
                ),
            )

            figure.update_layout(
                xaxis_title="Publish Hour",
                yaxis_title="Average Views",
            )

            show_figure(
                style_figure(
                    figure,
                    350,
                ),
                "insights_hour_performance",
            )

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------

    st.subheader(
        "How to Read These Insights"
    )

    st.info(
        """
        • Top Category identifies the category contributing the largest
          observed historical view volume.

        • Top Channel identifies the channel contributing the largest
          observed historical view volume.

        • Best Avg-View Day and Hour identify periods with the highest
          historical average views in this dataset.

        • Engagement metrics describe observed audience interaction.

        • These are historical analytics findings. They should not be
          interpreted as guaranteed future outcomes or ML predictions.
        """
    )


def render_about():
    """
    Professional project/about page.
    """
    st.title("About")
    st.caption(
        "Project documentation • methodology • technology • author"
    )

    left, right = st.columns(
        [1.45, 1],
        gap="large",
    )

    with left:
        st.subheader(
            "YouTube Trending Videos Analytics"
        )

        st.write(
            """
            This Streamlit application is a professional historical
            analytics dashboard for YouTube trending-video data.

            The project focuses on descriptive and diagnostic analytics:
            understanding what happened in the observed dataset, comparing
            categories and channels, studying audience engagement, examining
            publishing-time patterns, and identifying high-performing videos.
            """
        )

        st.subheader(
            "Analytics Scope"
        )

        st.markdown(
            """
            **Executive analytics**
            - Total videos
            - Total views
            - Total likes
            - Total comments
            - Average engagement

            **Historical performance**
            - Top videos
            - Category performance
            - Channel performance
            - Duration analysis

            **Audience engagement**
            - Likes vs views
            - Comments vs views
            - Engagement rate
            - Caption/quality analysis

            **Publishing analysis**
            - Day-of-week performance
            - Publish-hour performance
            - Month trends
            - Day × month heatmap

            **Content metadata**
            - Title length
            - Title word count
            - Tags
            - Hashtags
            """
        )

    with right:
        st.subheader(
            "Project Author"
        )

        st.metric(
            "Author",
            AUTHOR_NAME,
        )

        st.write(
            AUTHOR_ROLE
        )

        st.link_button(
            "🐙 GitHub",
            GITHUB_URL,
            use_container_width=True,
        )

        st.link_button(
            "💼 LinkedIn",
            LINKEDIN_URL,
            use_container_width=True,
        )

        st.subheader(
            "Technology Stack"
        )

        st.write(
            "Python • Streamlit • Pandas • NumPy • Plotly • SQLite"
        )

        st.subheader(
            "Data Philosophy"
        )

        st.write(
            """
            The dashboard is designed for business storytelling. Charts are
            intended to answer practical questions such as what content
            performed well, which channels generated the most views, how
            engagement differs, and when historically successful videos were
            published.
            """
        )

    st.divider()

    st.subheader(
        "Project Classification"
    )

    project_cols = st.columns(
        4,
        gap="small",
    )

    with project_cols[0]:
        st.metric(
            "Project Type",
            "Data Analytics",
        )

    with project_cols[1]:
        st.metric(
            "Business Focus",
            "YouTube",
        )

    with project_cols[2]:
        st.metric(
            "Analysis Type",
            "Historical",
        )

    with project_cols[3]:
        st.metric(
            "ML Prediction",
            "Not Included",
        )


# =============================================================================
# 10. NATIVE STREAMLIT NAVIGATION
# =============================================================================

overview_page = st.Page(
    render_overview,
    title="Overview",
    icon="📊",
    default=True,
)

video_page = st.Page(
    render_video_analytics,
    title="Video Analytics",
    icon="🎬",
)

insights_page = st.Page(
    render_insights,
    title="Insights",
    icon="💡",
)

about_page = st.Page(
    render_about,
    title="About",
    icon="ℹ️",
)

navigation = st.navigation(
    {
        "YouTube Analytics": [
            overview_page,
            video_page,
            insights_page,
            about_page,
        ],
    },
    position="top",
)

navigation.run()


# =============================================================================
# 11. GLOBAL PROFESSIONAL FOOTER
# =============================================================================

st.markdown(
    '<div class="footer-rule"></div>',
    unsafe_allow_html=True,
)

footer_left, footer_right = st.columns(
    [3.3, 1.7],
    vertical_alignment="center",
)

with footer_left:
    st.markdown(
        f"### 👨‍💻 {AUTHOR_NAME}"
    )

    st.caption(
        AUTHOR_ROLE
    )

    st.caption(
        "YouTube Trending Videos Analytics • "
        "Historical Performance Dashboard"
    )

with footer_right:
    st.link_button(
        "🐙 GitHub",
        GITHUB_URL,
        use_container_width=True,
    )

    st.link_button(
        "💼 LinkedIn",
        LINKEDIN_URL,
        use_container_width=True,
    )

st.markdown(
    """
<div class="footer-caption">
    YouTube Analytics Dashboard • Historical Analytics •
    Streamlit + Pandas + NumPy + Plotly
</div>
""",
    unsafe_allow_html=True,
)
