
# ============================================================
# EV INTELLIGENCE DASHBOARD
# PROFESSIONAL STABLE SINGLE-FILE PLOTLY DASH APPLICATION
# ============================================================
#
# FIXED:
# - KPIs now populate correctly
# - All 12 charts are connected to the dashboard callback
# - Theme actually changes chart + page colors
# - Filters update KPIs, charts and summary table
# - Reset button works
# - Summary table sorting/filtering/pagination works
# - Download CSV button exports current table view
# - Fixed port 8050, no input prompt
# - Reference Year = 2024
# - No __file__ usage
# ============================================================

from pathlib import Path
import textwrap
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx
from dash.exceptions import PreventUpdate

import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. PROJECT + DATA
# ============================================================

PROJECT_ROOT = Path(r"D:\ev_project_v2-Data Analysis")
CLEANED_FILE = PROJECT_ROOT / "datas" / "cleaned" / "ev_cleaned.csv"

REFERENCE_YEAR = 2024

if not PROJECT_ROOT.exists():
    raise FileNotFoundError(
        f"Project root not found:\n{PROJECT_ROOT}"
    )

if not CLEANED_FILE.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found:\n{CLEANED_FILE}\n"
    )

df = pd.read_csv(CLEANED_FILE)


# ============================================================
# 2. VALIDATION + CLEANING
# ============================================================

REQUIRED_COLUMNS = [
    "VIN",
    "County",
    "City",
    "State",
    "Postal_Code",
    "Model_Year",
    "Make",
    "Model",
    "EV_Type",
    "CAFV_Eligibility",
    "Electric_Range",
    "Base_MSRP",
    "Vehicle_ID",
    "Vehicle_Location",
    "Electric_Utility",
    "Census_Tract",
    "Legislative_District",
]

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    raise KeyError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )

for col in [
    "Model_Year",
    "Electric_Range",
    "Base_MSRP",
    "Legislative_District",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in [
    "VIN",
    "County",
    "City",
    "State",
    "Postal_Code",
    "Make",
    "Model",
    "EV_Type",
    "CAFV_Eligibility",
    "Vehicle_ID",
    "Vehicle_Location",
    "Electric_Utility",
    "Census_Tract",
]:
    df[col] = (
        df[col]
        .astype("string")
        .fillna("Unknown")
        .str.strip()
        .replace("", "Unknown")
    )

df["Vehicle_Count"] = 1
df["Vehicle_Age"] = (REFERENCE_YEAR - df["Model_Year"]).clip(lower=0)

# Keep unknown values out of the band cuts by using NaN first.
df["Range_Band"] = pd.cut(
    df["Electric_Range"],
    bins=[-np.inf, 50, 100, 150, 200, 250, 300, np.inf],
    labels=[
        "<= 50",
        "51–100",
        "101–150",
        "151–200",
        "201–250",
        "251–300",
        "> 300",
    ],
    include_lowest=True,
)

df["MSRP_Band"] = pd.cut(
    df["Base_MSRP"],
    bins=[-np.inf, 25000, 40000, 60000, 80000, 120000, np.inf],
    labels=[
        "<= $25K",
        "$25K–$40K",
        "$40K–$60K",
        "$60K–$80K",
        "$80K–$120K",
        "> $120K",
    ],
    include_lowest=True,
)


# ============================================================
# 3. THEMES
# ============================================================

THEMES = {
    "green": {
        "name": "Green",
        "page": "#F3FAF4",
        "surface": "#FFFFFF",
        "surface2": "#F8FCF8",
        "text": "#18352A",
        "muted": "#64756D",
        "border": "#D8E8DC",
        "primary": "#208A5A",
        "primary2": "#45B97C",
        "accent": "#A7DDBB",
        "accent2": "#75C993",
        "grid": "#DDEBE1",
        "palette": [
            "#166534", "#208A5A", "#45B97C",
            "#75C993", "#A7DDBB", "#D8F0DF",
        ],
        "sequential": "Greens",
    },
    "cream": {
        "name": "Cream",
        "page": "#FFF8E7",
        "surface": "#FFFFFF",
        "surface2": "#FFFDF7",
        "text": "#27343F",
        "muted": "#687684",
        "border": "#E8DDC7",
        "primary": "#287D5A",
        "primary2": "#69A77F",
        "accent": "#DDBF8C",
        "accent2": "#B98A4A",
        "grid": "#EEE6D7",
        "palette": [
            "#287D5A", "#4E966F", "#78B890",
            "#A6CDAF", "#DDBF8C", "#B98A4A",
        ],
        "sequential": "YlGn",
    },
    "chocolate": {
        "name": "Chocolate",
        "page": "#FBF5EF",
        "surface": "#FFFFFF",
        "surface2": "#FFF9F3",
        "text": "#3D281C",
        "muted": "#806957",
        "border": "#E8D8C9",
        "primary": "#7B4B2A",
        "primary2": "#A86F45",
        "accent": "#D8B08A",
        "accent2": "#B9835A",
        "grid": "#EBDDD0",
        "palette": [
            "#5A321D", "#7B4B2A", "#A86F45",
            "#C3926C", "#D8B08A", "#EBD7C4",
        ],
        "sequential": "YlOrBr",
    },
    "dark": {
        "name": "Dark",
        "page": "#101815",
        "surface": "#18221E",
        "surface2": "#1E2B25",
        "text": "#EAF5EE",
        "muted": "#A7B9AF",
        "border": "#31463C",
        "primary": "#52C788",
        "primary2": "#7BE0A6",
        "accent": "#214B36",
        "accent2": "#3D9667",
        "grid": "#30453B",
        "palette": [
            "#52C788", "#7BE0A6", "#3D9667",
            "#2B704C", "#9BE7BA", "#C5F3D5",
        ],
        "sequential": "Greens",
    },
    "white": {
        "name": "White",
        "page": "#FFFFFF",
        "surface": "#FFFFFF",
        "surface2": "#FAFAFA",
        "text": "#202B33",
        "muted": "#68737C",
        "border": "#DCE2E6",
        "primary": "#16825A",
        "primary2": "#43A879",
        "accent": "#CFE9DB",
        "accent2": "#79C49D",
        "grid": "#E5EAED",
        "palette": [
            "#16825A", "#43A879", "#79C49D",
            "#A5D5B9", "#CFE9DB", "#E6F3EB",
        ],
        "sequential": "Greens",
    },
}

THEME_ORDER = ["green", "cream", "chocolate", "dark", "white"]


# ============================================================
# 4. OPTIONS + FILTER ENGINE
# ============================================================

def make_options(series, numeric=False):
    values = series.dropna().unique().tolist()
    if numeric:
        values = sorted(values)
        return [{"label": str(int(v)), "value": int(v)} for v in values]
    values = sorted([str(v) for v in values])
    return [{"label": v, "value": v} for v in values]


YEAR_OPTIONS = make_options(df["Model_Year"], numeric=True)
STATE_OPTIONS = make_options(df["State"])
MAKE_OPTIONS = make_options(df["Make"])
EV_TYPE_OPTIONS = make_options(df["EV_Type"])
CAFV_OPTIONS = make_options(df["CAFV_Eligibility"])
UTILITY_OPTIONS = make_options(df["Electric_Utility"])


def apply_filters(
    data,
    years=None,
    states=None,
    makes=None,
    ev_types=None,
    cafv=None,
    utilities=None,
):
    result = data

    if years:
        result = result[result["Model_Year"].isin(years)]
    if states:
        result = result[result["State"].isin(states)]
    if makes:
        result = result[result["Make"].isin(makes)]
    if ev_types:
        result = result[result["EV_Type"].isin(ev_types)]
    if cafv:
        result = result[result["CAFV_Eligibility"].isin(cafv)]
    if utilities:
        result = result[result["Electric_Utility"].isin(utilities)]

    return result


# ============================================================
# 5. FORMAT HELPERS
# ============================================================

def fmt_int(value):
    return "N/A" if pd.isna(value) else f"{float(value):,.0f}"


def fmt_one(value):
    return "N/A" if pd.isna(value) else f"{float(value):,.1f}"


def fmt_money(value):
    return "N/A" if pd.isna(value) else f"${float(value):,.0f}"


def fmt_pct(value):
    return "N/A" if pd.isna(value) else f"{float(value):.1f}%"


# ============================================================
# 6. PLOT HELPERS
# ============================================================
#
# Premium chart treatment:
# - Gridlines removed.
# - Camera / export / zoom / home-style Plotly controls removed.
# - Remaining toolbar is positioned under the chart title.
# - Chart heights, widths, layout and data calculations remain unchanged.
#

def apply_plot_theme(fig, theme_name, height=560, margin=None):
    t = THEMES[theme_name]

    if margin is None:
        margin = dict(l=75, r=60, t=105, b=75)

    fig.update_layout(
        template="plotly_dark" if theme_name == "dark" else "plotly_white",
        paper_bgcolor=t["surface"],
        plot_bgcolor=t["surface"],
        height=height,
        margin=margin,
        font=dict(
            family="Arial, Helvetica, sans-serif",
            color=t["text"],
            size=12,
        ),
        title=dict(
            font=dict(size=19, color=t["text"]),
            x=0.02,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        hoverlabel=dict(
            bgcolor=t["surface"],
            font_color=t["text"],
            font_size=13,
        ),
        legend=dict(
            font=dict(color=t["text"]),
        ),
    )

    # Premium clean-chart treatment: no gridlines / zero-lines.
    # Size, margins, widths and dashboard layout are unchanged.
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=t["border"],
        tickfont=dict(color=t["muted"]),
        title_font=dict(color=t["text"]),
        automargin=True,
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        linecolor=t["border"],
        tickfont=dict(color=t["muted"]),
        title_font=dict(color=t["text"]),
        automargin=True,
    )

    return fig


def make_bar(fig, theme, height=560, left=100, right=80):
    return apply_plot_theme(
        fig,
        theme,
        height=height,
        margin=dict(l=left, r=right, t=105, b=75),
    )


def empty_figure(theme, title="No data available"):
    fig = go.Figure()
    fig.add_annotation(
        text="No records match the selected filters",
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=18, color=THEMES[theme]["muted"]),
    )
    fig.update_layout(title=title)
    return apply_plot_theme(fig, theme, height=560)


def wrap_utility_label(value, width=30):
    value = str(value).strip()
    if not value:
        return "Unknown"
    parts = textwrap.wrap(
        value,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "<br>".join(parts[:3])


# ============================================================
# 7. UI HELPERS
# ============================================================

def kpi_card(title, component_id, icon):
    return html.Div(
        [
            html.Div(
                [
                    html.Span(icon, className="kpi-icon"),
                    html.Span(title),
                ],
                className="kpi-label",
            ),
            html.Div("--", id=component_id, className="kpi-value"),
        ],
        className="kpi-card",
    )


def filter_dropdown(label, component_id, options):
    return html.Div(
        [
            html.Label(label, className="filter-label"),
            dcc.Dropdown(
                id=component_id,
                options=options,
                value=None,
                multi=True,
                clearable=True,
                searchable=True,
                placeholder=f"Select {label}",
                className="dash-filter",
            ),
        ],
        className="filter-item",
    )


def chart_card(component_id):
    return html.Div(
        dcc.Graph(
            id=component_id,
            config={
                "displaylogo": False,
                "displayModeBar": True,
                "responsive": True,
                "modeBarButtonsToRemove": [
                    "toImage",
                    "sendDataToCloud",
                    "zoom2d",
                    "pan2d",
                    "select2d",
                    "lasso2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d",
                    "resetScale2d",
                    "hoverClosestCartesian",
                    "hoverCompareCartesian",
                    "toggleSpikelines",
                ],
            },
            style={"width": "100%"},
        ),
        className="chart-card",
    )


# ============================================================
# 8. DASH APP
# ============================================================

app = Dash(
    __name__,
    title="EV Intelligence Dashboard",
    suppress_callback_exceptions=True,
)

server = app.server


# ============================================================
# 9. CSS
# ============================================================

app.index_string = r"""
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}

<style>

* { box-sizing: border-box; }

html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    min-height: 100%;
    font-family: Arial, Helvetica, sans-serif;
}

body {
    background: var(--page);
}

.app-shell {
    min-height: 100vh;
    background: var(--page);
    color: var(--text);
    --page: #F3FAF4;
    --surface: #FFFFFF;
    --surface2: #F8FCF8;
    --text: #18352A;
    --muted: #64756D;
    --border: #D8E8DC;
    --primary: #208A5A;
    --primary2: #45B97C;
    --accent: #A7DDBB;
    --accent2: #75C993;
}

.dashboard-container {
    width: 100%;
    max-width: 1900px;
    margin: 0 auto;
    padding: 20px;
}

.hero {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 24px 28px;
    margin-bottom: 18px;
    box-shadow: 0 7px 24px rgba(0,0,0,.055);
}

.hero-layout {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
}

.hero-left {
    display: flex;
    align-items: center;
    gap: 18px;
    min-width: 0;
}

.logo {
    width: 58px;
    height: 58px;
    min-width: 58px;
    border-radius: 18px;
    display: grid;
    place-items: center;
    background: var(--primary);
    color: #FFFFFF;
    font-weight: 900;
    font-size: 20px;
}

.hero-title {
    margin: 0;
    color: var(--text);
    font-size: clamp(30px, 3vw, 44px);
    line-height: 1.05;
    font-weight: 900;
}

.hero-subtitle {
    margin: 8px 0 0;
    color: var(--muted);
    font-size: 15px;
    line-height: 1.45;
}

.hero-right {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 16px;
    flex-wrap: wrap;
}

.reference-box {
    min-width: 135px;
    text-align: center;
    padding: 11px 18px;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: var(--surface2);
}

.reference-label, .theme-label {
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .4px;
    text-transform: uppercase;
}

.reference-year {
    color: var(--text);
    font-size: 27px;
    line-height: 1.05;
    font-weight: 900;
    margin-top: 3px;
}

.theme-box {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.theme-cycle {
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    border-radius: 12px;
    padding: 10px 16px;
    font-size: 13px;
    font-weight: 800;
    cursor: pointer;
}

.filter-card, .table-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 18px;
    box-shadow: 0 7px 24px rgba(0,0,0,.05);
}

.section-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 18px;
    margin-bottom: 20px;
}

.section-title {
    margin: 0;
    color: var(--text);
    font-size: 23px;
    line-height: 1.2;
    font-weight: 850;
}

.section-description {
    margin: 7px 0 0;
    color: var(--muted);
    font-size: 13px;
}

.reset-button, .download-button {
    border: 0;
    background: var(--primary);
    color: #FFFFFF;
    border-radius: 12px;
    padding: 12px 19px;
    font-size: 13px;
    font-weight: 800;
    cursor: pointer;
}

.filter-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 17px 18px;
}

.filter-item { min-width: 0; }

.filter-label {
    display: block;
    color: var(--text);
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 7px;
}

.dash-filter .Select-control {
    min-height: 45px;
    border-radius: 9px !important;
    border-color: var(--border) !important;
    background: var(--surface) !important;
}

.dash-filter .Select-menu-outer {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
    z-index: 3000 !important;
}

.dash-filter .Select-placeholder,
.dash-filter .Select-value-label {
    color: var(--muted) !important;
}

.dash-filter .Select-option {
    background: var(--surface) !important;
    color: var(--text) !important;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 18px;
}

.kpi-card {
    min-width: 0;
    min-height: 112px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 15px;
    box-shadow: 0 5px 18px rgba(0,0,0,.045);
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.kpi-label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--muted);
    font-size: 12px;
    font-weight: 800;
    line-height: 1.25;
}

.kpi-icon {
    width: 35px;
    height: 35px;
    min-width: 35px;
    display: grid;
    place-items: center;
    border-radius: 10px;
    background: var(--accent);
    color: var(--primary);
}

.kpi-value {
    margin-top: 7px;
    color: var(--text);
    font-size: 22px;
    line-height: 1.1;
    font-weight: 900;
    word-break: break-word;
}

.chart-grid-3 {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px;
    margin-bottom: 18px;
}

.chart-grid-2 {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 18px;
    margin-bottom: 18px;
}

.chart-card {
    min-width: 0;
    min-height: 560px;
    overflow: hidden;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 6px;
    box-shadow: 0 5px 18px rgba(0,0,0,.045);
}

/* Plotly toolbar sits at the top of each chart, below the title. */
.chart-card .modebar-container {
    top: 42px !important;
    right: 8px !important;
}

.chart-card .modebar {
    margin-top: 0 !important;
}

.table-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    flex-wrap: wrap;
    margin: 4px 0 16px;
}

.table-hint {
    color: var(--muted);
    font-size: 12.5px;
}

.dash-spreadsheet-container .dash-spreadsheet-inner td {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}

.dash-spreadsheet-container .dash-spreadsheet-inner th {
    background: var(--primary) !important;
    color: #FFFFFF !important;
}

.dash-spreadsheet-container input {
    color: var(--text) !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
}

.footer {
    text-align: center;
    color: var(--muted);
    padding: 22px 10px 8px;
    font-size: 12px;
    line-height: 1.5;
}

@media (max-width: 1450px) {
    .kpi-grid { grid-template-columns: repeat(4, minmax(0,1fr)); }
}

@media (max-width: 1100px) {
    .kpi-grid { grid-template-columns: repeat(3, minmax(0,1fr)); }
}

@media (max-width: 900px) {
    .hero-layout { flex-direction: column; align-items: flex-start; }
    .filter-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
    .chart-grid-3 { grid-template-columns: repeat(2, minmax(0,1fr)); }
    .chart-grid-2 { grid-template-columns: 1fr; }
}

@media (max-width: 650px) {
    .dashboard-container { padding: 10px; }
    .filter-grid, .kpi-grid, .chart-grid-3, .chart-grid-2 {
        grid-template-columns: 1fr;
    }
    .hero, .filter-card, .table-card { padding: 16px; }
}

</style>
</head>

<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
"""


# ============================================================
# 8. SUMMARY TABLE CONDITIONAL FORMATTING
# ============================================================
#
# FIX:
# DataTable is constructed while the module is loading, before
# update_dashboard() is ever executed. Therefore the conditional
# formatting rules must be defined at module scope.
# ============================================================

table_style_conditions = [

    # ----------------------------------------------------
    # MARKET SHARE — VERY HIGH
    # ----------------------------------------------------

    {
        "if": {
            "column_id": "Market Share %",
            "filter_query": (
                "{Market Share %} >= 10"
            ),
        },
        "backgroundColor": "#166534",
        "color": "#FFFFFF",
        "fontWeight": "800",
    },

    # ----------------------------------------------------
    # MARKET SHARE — HIGH
    # ----------------------------------------------------

    {
        "if": {
            "column_id": "Market Share %",
            "filter_query": (
                "{Market Share %} >= 5 && "
                "{Market Share %} < 10"
            ),
        },
        "backgroundColor": "#45B97C",
        "color": "#FFFFFF",
        "fontWeight": "800",
    },

    # ----------------------------------------------------
    # MARKET SHARE — MEDIUM
    # ----------------------------------------------------

    {
        "if": {
            "column_id": "Market Share %",
            "filter_query": (
                "{Market Share %} >= 2 && "
                "{Market Share %} < 5"
            ),
        },
        "backgroundColor": "#A7DDBB",
        "color": "#18352A",
        "fontWeight": "800",
    },

    # ----------------------------------------------------
    # MARKET SHARE BAR
    # ----------------------------------------------------

    {
        "if": {
            "column_id": "Market Share Bar",
        },
        "fontFamily": "Consolas, 'Courier New', monospace",
        "fontWeight": "800",
        "fontSize": "12px",
        "textAlign": "left",
        "letterSpacing": "0px",
    },

    # ----------------------------------------------------
    # VEHICLE RECORDS
    # ----------------------------------------------------

    {
        "if": {
            "column_id": "Vehicle Records",
            "filter_query": (
                "{Vehicle Records} >= 10000"
            ),
        },
        "fontWeight": "900",
        "color": "var(--primary)",
    },

    # ----------------------------------------------------
    # AVERAGE RANGE — STRONG
    # ----------------------------------------------------

    {
        "if": {
            "column_id": "Avg Range",
            "filter_query": (
                "{Avg Range} >= 250"
            ),
        },
        "backgroundColor": "#D8F0DF",
        "color": "#166534",
        "fontWeight": "800",
    },

    # ----------------------------------------------------
    # AVERAGE RANGE — MEDIUM
    # ----------------------------------------------------

    {
        "if": {
            "column_id": "Avg Range",
            "filter_query": (
                "{Avg Range} >= 150 && "
                "{Avg Range} < 250"
            ),
        },
        "backgroundColor": "#EDF7EF",
        "color": "#208A5A",
        "fontWeight": "700",
    },

    # ----------------------------------------------------
    # AVERAGE MSRP — HIGH
    # ----------------------------------------------------

    {
        "if": {
            "column_id": "Avg MSRP",
            "filter_query": (
                "{Avg MSRP} >= 75000"
            ),
        },
        "backgroundColor": "#FFF0E2",
        "color": "#7B4B2A",
        "fontWeight": "800",
    },

    # ----------------------------------------------------
    # RANGE VS OVERALL — ABOVE BENCHMARK
    # ----------------------------------------------------

    {
        "if": {
            "column_id": "Range vs Overall %",
            "filter_query": (
                "{Range vs Overall %} >= 110"
            ),
        },
        "backgroundColor": "#D8F0DF",
        "color": "#166534",
        "fontWeight": "800",
    },

    # ----------------------------------------------------
    # RANGE VS OVERALL — BELOW BENCHMARK
    # ----------------------------------------------------

    {
        "if": {
            "column_id": "Range vs Overall %",
            "filter_query": (
                "{Range vs Overall %} < 90"
            ),
        },
        "backgroundColor": "#FCE8E6",
        "color": "#9B2C2C",
        "fontWeight": "700",
    },

    # ----------------------------------------------------
    # PRICE VS OVERALL — ABOVE BENCHMARK
    # ----------------------------------------------------

    {
        "if": {
            "column_id": "Price vs Overall %",
            "filter_query": (
                "{Price vs Overall %} >= 120"
            ),
        },
        "backgroundColor": "#FFF0E2",
        "color": "#7B4B2A",
        "fontWeight": "800",
    },

    # ----------------------------------------------------
    # PRICE VS OVERALL — BELOW BENCHMARK
    # ----------------------------------------------------

    {
        "if": {
            "column_id": "Price vs Overall %",
            "filter_query": (
                "{Price vs Overall %} < 80"
            ),
        },
        "backgroundColor": "#EDF7EF",
        "color": "#166534",
        "fontWeight": "700",
    },

    # ----------------------------------------------------
    # VOLUME RANK — #1
    # ----------------------------------------------------

    {
        "if": {
            "filter_query": "{Volume Rank} = 1",
            "column_id": "Volume Rank",
        },
        "backgroundColor": "#F5D76E",
        "color": "#4D3F2F",
        "fontWeight": "900",
        "fontSize": "15px",
    },

    # ----------------------------------------------------
    # VOLUME RANK — #2
    # ----------------------------------------------------

    {
        "if": {
            "filter_query": "{Volume Rank} = 2",
            "column_id": "Volume Rank",
        },
        "backgroundColor": "#D9DEE5",
        "color": "#3E4650",
        "fontWeight": "900",
        "fontSize": "15px",
    },

    # ----------------------------------------------------
    # VOLUME RANK — #3
    # ----------------------------------------------------

    {
        "if": {
            "filter_query": "{Volume Rank} = 3",
            "column_id": "Volume Rank",
        },
        "backgroundColor": "#E6B98A",
        "color": "#5A321D",
        "fontWeight": "900",
        "fontSize": "15px",
    },
]

# ============================================================
# 10. LAYOUT
# ============================================================

app.layout = html.Div(
    id="app-shell",
    className="app-shell",
    children=[
        dcc.Store(
            id="theme-store",
            storage_type="local",
            data="green",
        ),
        dcc.Download(id="download-table-csv"),

        html.Div(
            className="dashboard-container",
            children=[

                # ---------------- HEADER ----------------
                html.Div(
                    className="hero",
                    children=[
                        html.Div(
                            className="hero-layout",
                            children=[
                                html.Div(
                                    className="hero-left",
                                    children=[
                                        html.Div("EV", className="logo"),
                                        html.Div(
                                            [
                                                html.H1(
                                                    "EV Intelligence Dashboard",
                                                    className="hero-title",
                                                ),
                                                html.P(
                                                    "Electric Vehicle Market Analytics • "
                                                    "Business Intelligence • "
                                                    "Interactive Decision Support",
                                                    className="hero-subtitle",
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="hero-right",
                                    children=[
                                        html.Div(
                                            [
                                                html.Div(
                                                    "Reference Year",
                                                    className="reference-label",
                                                ),
                                                html.Div(
                                                    str(REFERENCE_YEAR),
                                                    className="reference-year",
                                                ),
                                            ],
                                            className="reference-box",
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    "Theme",
                                                    className="theme-label",
                                                ),
                                                html.Button(
                                                    "🌿 Green • Click to change",
                                                    id="theme-cycle",
                                                    n_clicks=0,
                                                    className="theme-cycle",
                                                ),
                                            ],
                                            className="theme-box",
                                        ),
                                    ],
                                ),
                            ]
                        )
                    ],
                ),

                # ---------------- FILTERS ----------------
                html.Div(
                    className="filter-card",
                    children=[
                        html.Div(
                            className="section-head",
                            children=[
                                html.Div(
                                    [
                                        html.H2(
                                            "Interactive Filters",
                                            className="section-title",
                                        ),
                                        html.P(
                                            "Slice the EV market by model year, state, "
                                            "manufacturer, EV type, CAFV eligibility "
                                            "and electric utility.",
                                            className="section-description",
                                        ),
                                    ]
                                ),
                                html.Button(
                                    "↺ Reset All Filters",
                                    id="reset-filters",
                                    n_clicks=0,
                                    className="reset-button",
                                ),
                            ],
                        ),

                        html.Div(
                            className="filter-grid",
                            children=[
                                filter_dropdown(
                                    "Model Year",
                                    "filter-year",
                                    YEAR_OPTIONS,
                                ),
                                filter_dropdown(
                                    "State",
                                    "filter-state",
                                    STATE_OPTIONS,
                                ),
                                filter_dropdown(
                                    "Manufacturer",
                                    "filter-make",
                                    MAKE_OPTIONS,
                                ),
                                filter_dropdown(
                                    "EV Type",
                                    "filter-ev-type",
                                    EV_TYPE_OPTIONS,
                                ),
                                filter_dropdown(
                                    "CAFV Eligibility",
                                    "filter-cafv",
                                    CAFV_OPTIONS,
                                ),
                                filter_dropdown(
                                    "Electric Utility",
                                    "filter-utility",
                                    UTILITY_OPTIONS,
                                ),
                            ],
                        ),
                    ],
                ),

                # ---------------- KPIs ----------------
                html.Div(
                    className="kpi-grid",
                    children=[
                        kpi_card("Total Vehicles", "kpi-total", "🚗"),
                        kpi_card("Manufacturers", "kpi-manufacturers", "🏭"),
                        kpi_card("Models", "kpi-models", "◈"),
                        kpi_card("States", "kpi-states", "◎"),
                        kpi_card("Average Range", "kpi-range", "⚡"),

                        kpi_card("Average MSRP", "kpi-msrp", "$"),
                        kpi_card("Median Range", "kpi-median-range", "↔"),
                        kpi_card("Average Vehicle Age", "kpi-age", "◷"),
                        kpi_card("BEV Share", "kpi-bev", "🔋"),
                        kpi_card("PHEV Share", "kpi-phev", "🔌"),

                        kpi_card("CAFV Eligible Share", "kpi-cafv", "✓"),
                        kpi_card("Filtered Records", "kpi-filtered", "#"),
                        kpi_card("Top Manufacturer", "kpi-top-make", "★"),
                        kpi_card("Top State", "kpi-top-state", "📍"),
                        kpi_card("Top Model", "kpi-top-model", "🥇"),

                        kpi_card("BEV Avg Range", "kpi-bev-range", "🔋"),
                        kpi_card("PHEV Avg Range", "kpi-phev-range", "🔌"),
                        kpi_card("Priced Records Share", "kpi-priced-share", "🏷️"),
                        kpi_card("Zero MSRP Share", "kpi-zero-msrp", "⚠️"),
                        kpi_card("Maximum Range", "kpi-max-range", "🚀"),
                    ],
                ),

                # ---------------- CHART ROW 1 ----------------
                html.Div(
                    className="chart-grid-3",
                    children=[
                        chart_card("chart-year"),
                        chart_card("chart-manufacturer"),
                        chart_card("chart-ev-type"),
                    ],
                ),

                # ---------------- CHART ROW 2 ----------------
                html.Div(
                    className="chart-grid-3",
                    children=[
                        chart_card("chart-msrp"),
                        chart_card("chart-range"),
                        chart_card("chart-state"),
                    ],
                ),

                # ---------------- CHART ROW 3 ----------------
                html.Div(
                    className="chart-grid-2",
                    children=[
                        chart_card("chart-portfolio"),
                        chart_card("chart-model"),
                    ],
                ),

                # ---------------- CHART ROW 4 ----------------
                html.Div(
                    className="chart-grid-2",
                    children=[
                        chart_card("chart-cafv"),
                        chart_card("chart-utility"),
                    ],
                ),

                # ---------------- CHART ROW 5 ----------------
                html.Div(
                    className="chart-grid-2",
                    children=[
                        chart_card("chart-city"),
                        chart_card("chart-ev-year"),
                    ],
                ),

                # ---------------- SUMMARY TABLE ----------------
                html.Div(
                    className="table-card",
                    children=[
                        html.H2(
                            "Manufacturer Performance Summary",
                            className="section-title",
                        ),
                        html.P(
                            "Volume, market share, portfolio breadth, "
                            "range, pricing and vehicle-age comparison.",
                            className="section-description",
                        ),
                        html.Div(
                            className="table-toolbar",
                            children=[
                                html.Div(
                                    "Use the filter boxes and column sorting below.",
                                    className="table-hint",
                                ),
                                html.Button(
                                    "⬇ Download CSV",
                                    id="download-csv-btn",
                                    n_clicks=0,
                                    className="download-button",
                                ),
                            ],
                        ),
                        dash_table.DataTable(
                            id="summary-table",
                            columns=[],
                            data=[],
                            page_action="native",
                            page_current=0,
                            page_size=15,
                            sort_action="native",
                            filter_action="native",
                            fixed_rows={"headers": True},
                            style_table={
                                "overflowX": "auto",
                                "maxHeight": "620px",
                                "overflowY": "auto",
                            },
                            style_header={
                                "backgroundColor": "var(--primary)",
                                "color": "#FFFFFF",
                                "fontWeight": "800",
                                "textAlign": "center",
                                "padding": "12px",
                            },
                            style_filter={
                                "backgroundColor": "var(--surface2)",
                                "color": "var(--text)",
                                "padding": "6px",
                            },
                            style_cell={
                                "backgroundColor": "var(--surface)",
                                "color": "var(--text)",
                                "padding": "10px",
                                "fontFamily": "Arial",
                                "fontSize": "13px",
                                "textAlign": "center",
                                "whiteSpace": "normal",
                                "minWidth": "95px",
                            },
                            style_data_conditional=table_style_conditions,
                        ),
                    ],
                ),

                html.Div(
                    [
                        html.Strong("EV Intelligence Platform"),
                        " • Python • Pandas • Plotly • Dash",
                        html.Br(),
                        f"{CLEANED_FILE.name} • {len(df):,} records • "
                        f"Reference Year {REFERENCE_YEAR}",
                    ],
                    className="footer",
                ),
            ],
        ),
    ],
)


# ============================================================
# 11. RESET FILTERS
# ============================================================

@app.callback(
    Output("filter-year", "value"),
    Output("filter-state", "value"),
    Output("filter-make", "value"),
    Output("filter-ev-type", "value"),
    Output("filter-cafv", "value"),
    Output("filter-utility", "value"),
    Input("reset-filters", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return None, None, None, None, None, None


# ============================================================
# 12. THEME CYCLE
# ============================================================

@app.callback(
    Output("theme-store", "data"),
    Input("theme-cycle", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def cycle_theme(n_clicks, current_theme):
    if not n_clicks:
        raise PreventUpdate

    if current_theme not in THEME_ORDER:
        current_theme = "green"

    index = THEME_ORDER.index(current_theme)
    return THEME_ORDER[(index + 1) % len(THEME_ORDER)]


@app.callback(
    Output("app-shell", "className"),
    Output("app-shell", "style"),
    Output("theme-cycle", "children"),
    Input("theme-store", "data"),
)
def render_theme(theme_name):
    if theme_name not in THEMES:
        theme_name = "green"

    t = THEMES[theme_name]

    icon = {
        "green": "🌿",
        "cream": "🌾",
        "chocolate": "🍫",
        "dark": "◐",
        "white": "○",
    }[theme_name]

    theme_style = {
        "--page": t["page"],
        "--surface": t["surface"],
        "--surface2": t["surface2"],
        "--text": t["text"],
        "--muted": t["muted"],
        "--border": t["border"],
        "--primary": t["primary"],
        "--primary2": t["primary2"],
        "--accent": t["accent"],
        "--accent2": t["accent2"],
    }

    return (
        "app-shell",
        theme_style,
        f"{icon} {t['name']} • Click to change",
    )


# ============================================================
# 13. MAIN DASHBOARD UPDATE CALLBACK
# ============================================================

@app.callback(
    [
        Output("kpi-total", "children"),
        Output("kpi-manufacturers", "children"),
        Output("kpi-models", "children"),
        Output("kpi-states", "children"),
        Output("kpi-range", "children"),
        Output("kpi-msrp", "children"),
        Output("kpi-median-range", "children"),
        Output("kpi-age", "children"),
        Output("kpi-bev", "children"),
        Output("kpi-phev", "children"),
        Output("kpi-cafv", "children"),
        Output("kpi-filtered", "children"),
        Output("kpi-top-make", "children"),
        Output("kpi-top-state", "children"),
        Output("kpi-top-model", "children"),
        Output("kpi-bev-range", "children"),
        Output("kpi-phev-range", "children"),
        Output("kpi-priced-share", "children"),
        Output("kpi-zero-msrp", "children"),
        Output("kpi-max-range", "children"),

        Output("chart-year", "figure"),
        Output("chart-manufacturer", "figure"),
        Output("chart-ev-type", "figure"),
        Output("chart-state", "figure"),
        Output("chart-range", "figure"),
        Output("chart-msrp", "figure"),
        Output("chart-portfolio", "figure"),
        Output("chart-model", "figure"),
        Output("chart-cafv", "figure"),
        Output("chart-utility", "figure"),
        Output("chart-city", "figure"),
        Output("chart-ev-year", "figure"),

        Output("summary-table", "columns"),
        Output("summary-table", "data"),
    ],
    [
        Input("filter-year", "value"),
        Input("filter-state", "value"),
        Input("filter-make", "value"),
        Input("filter-ev-type", "value"),
        Input("filter-cafv", "value"),
        Input("filter-utility", "value"),
        Input("theme-store", "data"),
    ],
)
def update_dashboard(
    years,
    states,
    makes,
    ev_types,
    cafv,
    utilities,
    theme,
):
    if theme not in THEMES:
        theme = "green"

    t = THEMES[theme]

    d = apply_filters(
        df,
        years,
        states,
        makes,
        ev_types,
        cafv,
        utilities,
    )

    if d.empty:
        empty = empty_figure(theme, "No matching data")

        empty_columns = [{"name": "Message", "id": "Message"}]
        empty_data = [{"Message": "No records match the selected filters."}]

        return (
            "0", "0", "0", "0", "N/A", "N/A", "N/A", "N/A",
            "0.0%", "0.0%", "0.0%", "0",
            "N/A", "N/A", "N/A", "N/A", "N/A",
            "0.0%", "0.0%", "N/A",
            empty, empty, empty, empty, empty, empty,
            empty, empty, empty, empty, empty, empty,
            empty_columns, empty_data,
        )

    # ========================================================
    # KPI CALCULATIONS
    # ========================================================

    total = len(d)
    manufacturers = d["Make"].nunique()
    models = d["Model"].nunique()
    states_count = d["State"].nunique()

    avg_range = d["Electric_Range"].mean()
    median_range = d["Electric_Range"].median()
    max_range = d["Electric_Range"].max()

    paid_msrp = d.loc[d["Base_MSRP"] > 0, "Base_MSRP"]
    avg_msrp = paid_msrp.mean() if len(paid_msrp) else np.nan

    avg_age = d["Vehicle_Age"].mean()

    ev_text = d["EV_Type"].astype(str)

    bev_mask = ev_text.str.contains(
        "Battery Electric Vehicle",
        case=False,
        na=False,
    )

    phev_mask = (
        ev_text.str.contains(
            "Plug-in",
            case=False,
            na=False,
        )
        & ~bev_mask
    )

    cafv_mask = d["CAFV_Eligibility"].astype(str).str.contains(
        "Eligible",
        case=False,
        na=False,
    )

    bev_share = bev_mask.mean() * 100
    phev_share = phev_mask.mean() * 100
    cafv_share = cafv_mask.mean() * 100

    priced_share = (d["Base_MSRP"] > 0).mean() * 100
    zero_msrp_share = (d["Base_MSRP"].fillna(0) <= 0).mean() * 100

    bev_avg_range = (
        d.loc[bev_mask, "Electric_Range"].mean()
        if bev_mask.any() else np.nan
    )

    phev_avg_range = (
        d.loc[phev_mask, "Electric_Range"].mean()
        if phev_mask.any() else np.nan
    )

    top_make = (
        d["Make"].value_counts().index[0]
        if len(d) else "N/A"
    )

    top_state = (
        d["State"].value_counts().index[0]
        if len(d) else "N/A"
    )

    top_model = (
        d["Model"].value_counts().index[0]
        if len(d) else "N/A"
    )

    # ========================================================
    # CHART 1 — YEAR TREND
    # ========================================================

    year_data = (
        d.groupby("Model_Year", as_index=False)
        .agg(Vehicle_Count=("Vehicle_Count", "sum"))
        .sort_values("Model_Year")
    )

    year_data["Market_Share_%"] = (
        year_data["Vehicle_Count"] / total * 100
    )

    year_data["YoY_Growth_%"] = (
        year_data["Vehicle_Count"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    fig_year = go.Figure()

    fig_year.add_trace(
        go.Scatter(
            x=year_data["Model_Year"],
            y=year_data["Vehicle_Count"],
            mode="lines+markers",
            name="Vehicle Records",
            line=dict(color=t["primary"], width=4),
            marker=dict(size=8, color=t["primary2"]),
            fill="tozeroy",
            fillcolor=t["accent"],
            customdata=np.column_stack(
                [
                    year_data["Market_Share_%"],
                    year_data["YoY_Growth_%"],
                ]
            ),
            hovertemplate=(
                "<b>Model Year:</b> %{x}<br>"
                "<b>Vehicles:</b> %{y:,}<br>"
                "<b>Share:</b> %{customdata[0]:.1f}%<br>"
                "<b>YoY:</b> %{customdata[1]:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    fig_year.update_layout(
        title="EV Market Trend by Model Year",
        xaxis_title="Model Year",
        yaxis_title="Vehicle Records",
    )

    apply_plot_theme(
        fig_year,
        theme,
        height=560,
        margin=dict(l=78, r=45, t=105, b=72),
    )

    # ========================================================
    # CHART 2 — TOP MANUFACTURERS
    # ========================================================

    manufacturer = (
        d.groupby("Make", as_index=False)
        .agg(
            Vehicle_Count=("Vehicle_Count", "sum"),
            Models=("Model", "nunique"),
            Avg_Range=("Electric_Range", "mean"),
            Avg_MSRP=("Base_MSRP", lambda s: s[s > 0].mean()),
            Avg_Age=("Vehicle_Age", "mean"),
        )
    )

    manufacturer["Market_Share_%"] = (
        manufacturer["Vehicle_Count"] / total * 100
    )

    top_makes = (
        manufacturer
        .nlargest(12, "Vehicle_Count")
        .sort_values("Vehicle_Count")
    )

    fig_make = px.bar(
        top_makes,
        x="Vehicle_Count",
        y="Make",
        orientation="h",
        text="Market_Share_%",
        title="Top 12 Manufacturers by Vehicle Records",
        color="Vehicle_Count",
        color_continuous_scale=t["sequential"],
        custom_data=[
            "Vehicle_Count",
            "Market_Share_%",
            "Models",
            "Avg_Range",
            "Avg_MSRP",
            "Avg_Age",
        ],
    )

    fig_make.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Vehicle Records: %{customdata[0]:,}<br>"
            "Market Share: %{customdata[1]:.1f}%<br>"
            "Models: %{customdata[2]:,}<br>"
            "Average Range: %{customdata[3]:.1f}<br>"
            "Average MSRP: $%{customdata[4]:,.0f}<br>"
            "Average Age: %{customdata[5]:.1f}"
            "<extra></extra>"
        ),
    )

    max_make = float(top_makes["Vehicle_Count"].max())

    fig_make.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(
            title="Vehicle Records",
            tickformat=",",
            range=[0, max_make * 1.20],
        ),
        yaxis=dict(
            title="Manufacturer",
            automargin=True,
        ),
    )

    make_bar(fig_make, theme, height=560, left=125, right=85)

   # ========================================================
    # CHART 3 — EV TECHNOLOGY DONUT
    # ========================================================

    ev_type = (
        d.groupby(
            "EV_Type",
            as_index=False,
        )
        .agg(
            Vehicle_Count=("Vehicle_Count", "sum")
        )
    )

    ev_type["Share_%"] = (
        ev_type["Vehicle_Count"]
        / total
        * 100
    )


    # --------------------------------------------------------
    # DONUT
    # --------------------------------------------------------

    fig_ev = px.pie(
        ev_type,
        names="EV_Type",
        values="Vehicle_Count",
        hole=0.52,
        color_discrete_sequence=t["palette"],
    )


    fig_ev.update_traces(
        # Maximum practical donut area inside the 3-column card.
        domain=dict(
            x=[0.03, 0.97],
            y=[0.02, 0.76],
        ),

        textposition="inside",

        texttemplate=(
            "%{label}<br>"
            "%{percent:.1%}"
        ),

        textfont=dict(
            size=11,
            color=t["text"],
        ),

        insidetextorientation="radial",

        hovertemplate=(
            "<b>%{label}</b><br>"
            "Vehicle Records: %{value:,}<br>"
            "Market Share: %{percent:.1%}"
            "<extra></extra>"
        ),

        marker=dict(
            line=dict(
                color=t["surface"],
                width=2,
            )
        ),
    )


    # --------------------------------------------------------
    # THEME FIRST — TITLE IS SET AFTER THIS
    # --------------------------------------------------------

    apply_plot_theme(
        fig_ev,
        theme,
        height=560,
        margin=dict(
            l=12,
            r=12,
            t=105,
            b=18,
        ),
    )


    # --------------------------------------------------------
    # TITLE + LEGEND — FINAL LAYOUT
    # --------------------------------------------------------

    fig_ev.update_layout(
        title=dict(
            text="EV Technology Mix",
            x=0.02,
            xanchor="left",
            y=0.985,
            yanchor="top",
            font=dict(
                size=20,
                color=t["text"],
            ),
        ),

        showlegend=True,

        # Legend is under the title and above the donut.
        legend=dict(
            title=dict(
                text="EV Technology",
                font=dict(
                    size=12,
                    color=t["text"],
                ),
            ),
            orientation="h",
            x=0.02,
            y=0.84,
            xanchor="left",
            yanchor="middle",
            font=dict(
                size=11.5,
                color=t["text"],
            ),
            bgcolor="rgba(0,0,0,0)",
        ),

        # Same chart height; only internal plot composition is improved.
        margin=dict(
            l=12,
            r=12,
            t=105,
            b=18,
        ),

        uniformtext=dict(
            minsize=10,
            mode="hide",
        ),
    )

    # ========================================================
    # CHART 4 — TOP STATES
    # HORIZONTAL BAR — CLEAN STATE COMPARISON
    # ========================================================
    
    state_data = (
        d.groupby(
            "State",
            as_index=False,
        )
        .agg(
            Vehicle_Count=("Vehicle_Count", "sum"),
            Manufacturers=("Make", "nunique"),
            Models=("Model", "nunique"),
            Avg_Range=("Electric_Range", "mean"),
        )
    )
    
    state_data["Market_Share_%"] = (
        state_data["Vehicle_Count"]
        / total
        * 100
    )
    
    top_states = (
        state_data
        .nlargest(
            12,
            "Vehicle_Count",
        )
        .sort_values(
            "Vehicle_Count",
            ascending=True,
        )
        .copy()
    )
    
    
    fig_state = px.bar(
        top_states,
    
        x="Vehicle_Count",
        y="State",
    
        orientation="h",
    
        text="Market_Share_%",
    
        title="Top States by EV Records",
    
        color="Vehicle_Count",
    
        color_continuous_scale=t["sequential"],
    
        custom_data=[
            "Vehicle_Count",
            "Market_Share_%",
            "Manufacturers",
            "Models",
            "Avg_Range",
        ],
    )
    
    
    fig_state.update_traces(
    
        texttemplate="%{text:.1f}%",
    
        textposition="outside",
    
        cliponaxis=False,
    
        marker_line_width=0,
    
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Vehicle Records: %{customdata[0]:,}<br>"
            "Market Share: %{customdata[1]:.1f}%<br>"
            "Manufacturers: %{customdata[2]:,}<br>"
            "Models: %{customdata[3]:,}<br>"
            "Average Range: %{customdata[4]:.1f}<br>"
            "<extra></extra>"
        ),
    )
    
    
    max_state = float(
        top_states["Vehicle_Count"].max()
    )
    
    
    fig_state.update_layout(
    
        coloraxis_showscale=False,
    
        xaxis=dict(
            title="Vehicle Records",
    
            tickformat=",",
    
            range=[
                0,
                max_state * 1.18,
            ],
    
            showgrid=False,
    
            zeroline=False,
        ),
    
        yaxis=dict(
            title=None,
    
            categoryorder="array",
    
            categoryarray=top_states[
                "State"
            ].tolist(),
    
            tickfont=dict(
                size=12,
            ),
    
            automargin=True,
    
            showgrid=False,
    
            zeroline=False,
        ),
    )
    
    
    make_bar(
        fig_state,
        theme,
        height=560,
        left=80,
        right=85,
    )

    # ========================================================
    # CHART 5 — RANGE DISTRIBUTION
    # ========================================================

    range_data = (
        d.groupby("Range_Band", observed=False, as_index=False)
        .agg(Vehicle_Count=("Vehicle_Count", "sum"))
    )

    range_data = range_data[range_data["Vehicle_Count"] > 0].copy()
    range_data["Percentage_%"] = (
        range_data["Vehicle_Count"] / total * 100
    )

    fig_range = px.bar(
        range_data,
        x="Range_Band",
        y="Vehicle_Count",
        text="Percentage_%",
        title="Electric Range Distribution",
        color="Vehicle_Count",
        color_continuous_scale=t["sequential"],
    )

    fig_range.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False,
    )

    fig_range.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Electric Range Band",
        yaxis_title="Vehicle Records",
    )

    apply_plot_theme(
        fig_range,
        theme,
        height=560,
        margin=dict(l=70, r=45, t=105, b=82),
    )

    # ========================================================
    # CHART 6 — MSRP DISTRIBUTION
    # ========================================================

    msrp_data = (
        d.groupby("MSRP_Band", observed=False, as_index=False)
        .agg(Vehicle_Count=("Vehicle_Count", "sum"))
    )

    msrp_data = msrp_data[msrp_data["Vehicle_Count"] > 0].copy()
    msrp_data["Percentage_%"] = (
        msrp_data["Vehicle_Count"] / total * 100
    )

    fig_msrp = px.bar(
        msrp_data,
        x="Vehicle_Count",
        y="MSRP_Band",
        orientation="h",
        text="Percentage_%",
        title="Base MSRP Distribution",
        color="Vehicle_Count",
        color_continuous_scale=t["sequential"],
    )

    fig_msrp.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False,
    )

    max_msrp = float(msrp_data["Vehicle_Count"].max())

    fig_msrp.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(
            title="Vehicle Records",
            tickformat=",",
            range=[0, max_msrp * 1.20],
        ),
        yaxis=dict(
            title="Base MSRP Band",
            automargin=True,
        ),
    )

    make_bar(fig_msrp, theme, height=560, left=130, right=80)

    # ========================================================
    # CHART 7 — MANUFACTURER PORTFOLIO
    # ========================================================

    portfolio = manufacturer[
        manufacturer["Vehicle_Count"] >= 10
    ].copy()

    if portfolio.empty:
        fig_portfolio = empty_figure(
            theme,
            "Manufacturer Portfolio",
        )
    else:
        fig_portfolio = px.scatter(
            portfolio,
            x="Vehicle_Count",
            y="Avg_Range",
            size="Models",
            color="Avg_MSRP",
            hover_name="Make",
            hover_data=[
                "Vehicle_Count",
                "Models",
                "Avg_Range",
                "Avg_MSRP",
                "Avg_Age",
            ],
            title="Manufacturer Portfolio: Volume vs Average Range",
            color_continuous_scale=t["sequential"],
            labels={
                "Vehicle_Count": "Vehicle Records",
                "Avg_Range": "Average Electric Range",
                "Avg_MSRP": "Average MSRP",
                "Models": "Model Count",
            },
        )

        fig_portfolio.update_traces(
            marker=dict(
                line=dict(
                    color=t["surface"],
                    width=1,
                )
            )
        )

        apply_plot_theme(
            fig_portfolio,
            theme,
            height=560,
            margin=dict(l=80, r=100, t=105, b=72),
        )

    # ========================================================
    # CHART 8 — TOP MODELS
    # ========================================================

    model_data = (
        d.groupby("Model", as_index=False)
        .agg(
            Vehicle_Count=("Vehicle_Count", "sum"),
            Avg_Range=("Electric_Range", "mean"),
            Avg_MSRP=("Base_MSRP", lambda s: s[s > 0].mean()),
        )
        .nlargest(15, "Vehicle_Count")
        .sort_values("Vehicle_Count")
    )

    fig_model = px.bar(
        model_data,
        x="Vehicle_Count",
        y="Model",
        orientation="h",
        text="Vehicle_Count",
        title="Top 15 EV Models",
        color="Avg_Range",
        color_continuous_scale=t["sequential"],
        hover_data=["Avg_Range", "Avg_MSRP"],
    )

    fig_model.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        cliponaxis=False,
    )

    max_model = float(model_data["Vehicle_Count"].max())

    fig_model.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(
            title="Vehicle Records",
            tickformat=",",
            range=[0, max_model * 1.20],
        ),
        yaxis=dict(
            title="Model",
            automargin=True,
        ),
    )

    make_bar(fig_model, theme, height=560, left=135, right=80)

    # ========================================================
    # CHART 9 — CAFV DONUT
    # ========================================================

    cafv_data = (
        d.groupby(
            "CAFV_Eligibility",
            as_index=False,
        )
        .agg(
            Vehicle_Count=("Vehicle_Count", "sum")
        )
    )


    # --------------------------------------------------------
    # DONUT
    # --------------------------------------------------------

    fig_cafv = px.pie(
        cafv_data,
        names="CAFV_Eligibility",
        values="Vehicle_Count",
        hole=0.52,
        color_discrete_sequence=t["palette"],
    )


    fig_cafv.update_traces(
        # Large donut area.
        domain=dict(
            x=[0.03, 0.97],
            y=[0.02, 0.76],
        ),

        textposition="inside",

        texttemplate="%{percent:.1%}",

        textfont=dict(
            size=11,
            color=t["text"],
        ),

        insidetextorientation="radial",

        hovertemplate=(
            "<b>%{label}</b><br>"
            "Vehicle Records: %{value:,}<br>"
            "Market Share: %{percent:.1%}"
            "<extra></extra>"
        ),

        marker=dict(
            line=dict(
                color=t["surface"],
                width=2,
            )
        ),
    )


    # --------------------------------------------------------
    # THEME FIRST — TITLE IS SET AFTER THIS
    # --------------------------------------------------------

    apply_plot_theme(
        fig_cafv,
        theme,
        height=560,
        margin=dict(
            l=12,
            r=12,
            t=105,
            b=18,
        ),
    )


    # --------------------------------------------------------
    # TITLE + LEGEND — FINAL LAYOUT
    # --------------------------------------------------------

    fig_cafv.update_layout(
        title=dict(
            text="CAFV Eligibility Mix",
            x=0.02,
            xanchor="left",
            y=0.985,
            yanchor="top",
            font=dict(
                size=20,
                color=t["text"],
            ),
        ),

        showlegend=True,

        # Legend under title, above donut.
        legend=dict(
            title=dict(
                text="CAFV Eligibility",
                font=dict(
                    size=12,
                    color=t["text"],
                ),
            ),
            orientation="h",
            x=0.02,
            y=0.84,
            xanchor="left",
            yanchor="middle",
            font=dict(
                size=11,
                color=t["text"],
            ),
            bgcolor="rgba(0,0,0,0)",
        ),

        margin=dict(
            l=12,
            r=12,
            t=105,
            b=18,
        ),

        uniformtext=dict(
            minsize=9,
            mode="hide",
        ),
    )

    # ========================================================
    # CHART 10 — ELECTRIC UTILITIES
    # ========================================================

    utility_data = (
        d.groupby(
            "Electric_Utility",
            as_index=False,
        )
        .agg(
            Vehicle_Count=("Vehicle_Count", "sum")
        )
        .nlargest(12, "Vehicle_Count")
        .sort_values("Vehicle_Count")
        .copy()
    )

    utility_data["Utility_Display"] = (
        utility_data["Electric_Utility"]
        .apply(wrap_utility_label)
    )

    utility_data["Vehicle_Label"] = (
        utility_data["Vehicle_Count"]
    )

    fig_utility = px.bar(
        utility_data,
        x="Vehicle_Count",
        y="Utility_Display",
        orientation="h",
        text="Vehicle_Label",
        title="Top Electric Utilities by EV Records",
        color="Vehicle_Count",
        color_continuous_scale=t["sequential"],
        custom_data=[
            "Electric_Utility",
            "Vehicle_Count",
        ],
    )

    fig_utility.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Vehicle Records: %{customdata[1]:,}"
            "<extra></extra>"
        ),
    )

    max_utility = float(
        utility_data["Vehicle_Count"].max()
    )

    fig_utility.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(
            title="Vehicle Records",
            tickformat=",",
            range=[0, max_utility * 1.22],
        ),
        yaxis=dict(
            title=None,
            categoryorder="array",
            categoryarray=utility_data["Utility_Display"].tolist(),
            tickfont=dict(size=10.5),
            automargin=True,
        ),
    )

    make_bar(fig_utility, theme, height=590, left=215, right=90)

    # ========================================================
    # CHART 11 — CITIES
    # ========================================================

    city_data = (
        d.groupby("City", as_index=False)
        .agg(Vehicle_Count=("Vehicle_Count", "sum"))
        .nlargest(15, "Vehicle_Count")
        .sort_values("Vehicle_Count")
    )

    fig_city = px.bar(
        city_data,
        x="Vehicle_Count",
        y="City",
        orientation="h",
        text="Vehicle_Count",
        title="Top 15 Cities by EV Records",
        color="Vehicle_Count",
        color_continuous_scale=t["sequential"],
    )

    fig_city.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        cliponaxis=False,
    )

    max_city = float(city_data["Vehicle_Count"].max())

    fig_city.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(
            title="Vehicle Records",
            tickformat=",",
            range=[0, max_city * 1.20],
        ),
        yaxis=dict(
            title="City",
            automargin=True,
        ),
    )

    make_bar(fig_city, theme, height=560, left=100, right=80)

    # ========================================================
    # CHART 12 — EV TYPE TREND
    # LINE CHART — ATTRACTIVE TECHNOLOGY-WISE TREND
    # ========================================================
    
    ev_year = (
        d.groupby(
            ["Model_Year", "EV_Type"],
            as_index=False,
        )
        .agg(
            Vehicle_Count=("Vehicle_Count", "sum")
        )
        .sort_values(
            ["Model_Year", "EV_Type"]
        )
    )
    
    # Market share within each model year
    year_totals = (
        ev_year.groupby(
            "Model_Year",
            as_index=False,
        )["Vehicle_Count"]
        .sum()
        .rename(
            columns={
                "Vehicle_Count": "Year_Total"
            }
        )
    )
    
    ev_year = ev_year.merge(
        year_totals,
        on="Model_Year",
        how="left",
    )
    
    ev_year["Year_Share_%"] = (
        ev_year["Vehicle_Count"]
        / ev_year["Year_Total"]
        * 100
    )
    
    
    fig_ev_year = px.line(
        ev_year,
        x="Model_Year",
        y="Vehicle_Count",
        color="EV_Type",
        markers=True,
        line_shape="spline",
        title="EV Technology Trend by Model Year",
        color_discrete_sequence=t["palette"],
        custom_data=[
            "EV_Type",
            "Vehicle_Count",
            "Year_Share_%",
        ],
    )
    
    
    # Attractive line + marker styling
    fig_ev_year.update_traces(
        mode="lines+markers",
        line=dict(
            width=4,
        ),
        marker=dict(
            size=8,
            line=dict(
                width=1,
                color=t["surface"],
            ),
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "<b>Model Year:</b> %{x}<br>"
            "<b>Vehicle Records:</b> %{customdata[1]:,}<br>"
            "<b>Year Share:</b> %{customdata[2]:.1f}%"
            "<extra></extra>"
        ),
    )
    
    
    fig_ev_year.update_layout(
        xaxis=dict(
            title="Model Year",
            tickformat="d",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Vehicle Records",
            tickformat=",",
            showgrid=False,
            zeroline=False,
        ),
    
        legend=dict(
            title="EV Technology",
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
    
        hovermode="x unified",
    
        title=dict(
            text="EV Technology Trend by Model Year",
        ),
    )
    
    
    apply_plot_theme(
        fig_ev_year,
        theme,
        height=560,
        margin=dict(
            l=78,
            r=55,
            t=115,
            b=72,
        ),
    )

    # ========================================================
    # SUMMARY TABLE
    # PREMIUM CONDITIONAL FORMATTING + DATA BARS
    # ========================================================

    summary = manufacturer.copy()

    # --------------------------------------------------------
    # RELATIVE PERFORMANCE METRICS
    # --------------------------------------------------------

    summary["Range_vs_Overall_%"] = (
        summary["Avg_Range"] / avg_range * 100
        if pd.notna(avg_range) and avg_range != 0
        else np.nan
    )

    summary["Price_vs_Overall_%"] = (
        summary["Avg_MSRP"] / avg_msrp * 100
        if pd.notna(avg_msrp) and avg_msrp != 0
        else np.nan
    )

    summary["Volume_Rank"] = (
        summary["Vehicle_Count"]
        .rank(
            ascending=False,
            method="dense",
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # VISUAL MARKET-SHARE DATA BAR
    # --------------------------------------------------------
    #
    # Keep Market Share % as a real numeric column so native
    # sorting/filtering still works.
    #
    # Add one separate text column that visually represents
    # the percentage using filled/empty blocks.
    # --------------------------------------------------------

    def make_share_bar(value):
        if pd.isna(value):
            return ""

        value = max(
            0.0,
            min(float(value), 100.0),
        )

        total_blocks = 20

        filled_blocks = int(
            round(
                value / 100.0
                * total_blocks
            )
        )

        empty_blocks = (
            total_blocks
            - filled_blocks
        )

        return (
            "█" * filled_blocks
            + "░" * empty_blocks
            + f"  {value:.1f}%"
        )

    summary["Market Share Bar"] = (
        summary["Market_Share_%"]
        .apply(make_share_bar)
    )

    # --------------------------------------------------------
    # RENAME COLUMNS
    # --------------------------------------------------------

    summary = (
        summary
        .sort_values(
            "Vehicle_Count",
            ascending=False,
        )
        .rename(
            columns={
                "Make": "Manufacturer",
                "Vehicle_Count": "Vehicle Records",
                "Market_Share_%": "Market Share %",
                "Avg_Range": "Avg Range",
                "Avg_MSRP": "Avg MSRP",
                "Avg_Age": "Avg Vehicle Age",
                "Range_vs_Overall_%": "Range vs Overall %",
                "Price_vs_Overall_%": "Price vs Overall %",
                "Volume_Rank": "Volume Rank",
            }
        )
    )

    # --------------------------------------------------------
    # FINAL COLUMN ORDER
    # --------------------------------------------------------

    summary = summary[
        [
            "Manufacturer",
            "Vehicle Records",
            "Market Share %",
            "Market Share Bar",
            "Models",
            "Avg Range",
            "Avg MSRP",
            "Avg Vehicle Age",
            "Range vs Overall %",
            "Price vs Overall %",
            "Volume Rank",
        ]
    ]

    # ========================================================
    # TABLE COLUMNS
    # ========================================================

    table_columns = [
        {
            "name": "Manufacturer",
            "id": "Manufacturer",
        },

        {
            "name": "Vehicle Records",
            "id": "Vehicle Records",
            "type": "numeric",
            "format": {
                "specifier": ",.0f",
            },
        },

        {
            "name": "Market Share %",
            "id": "Market Share %",
            "type": "numeric",
            "format": {
                "specifier": ".2f",
            },
        },

        {
            "name": "Market Share Bar",
            "id": "Market Share Bar",
            "type": "text",
        },

        {
            "name": "Models",
            "id": "Models",
            "type": "numeric",
            "format": {
                "specifier": ",.0f",
            },
        },

        {
            "name": "Avg Range",
            "id": "Avg Range",
            "type": "numeric",
            "format": {
                "specifier": ".1f",
            },
        },

        {
            "name": "Avg MSRP",
            "id": "Avg MSRP",
            "type": "numeric",
            "format": {
                "specifier": ",.0f",
            },
        },

        {
            "name": "Avg Vehicle Age",
            "id": "Avg Vehicle Age",
            "type": "numeric",
            "format": {
                "specifier": ".1f",
            },
        },

        {
            "name": "Range vs Overall %",
            "id": "Range vs Overall %",
            "type": "numeric",
            "format": {
                "specifier": ".1f",
            },
        },

        {
            "name": "Price vs Overall %",
            "id": "Price vs Overall %",
            "type": "numeric",
            "format": {
                "specifier": ".1f",
            },
        },

        {
            "name": "Volume Rank",
            "id": "Volume Rank",
            "type": "numeric",
            "format": {
                "specifier": ",.0f",
            },
        },
    ]

    # ========================================================
    # TABLE DATA
    # ========================================================

    table_data = (
        summary
        .replace({np.nan: None})
        .to_dict("records")
    )

    # ========================================================
    # PREMIUM CONDITIONAL FORMATTING
    # ========================================================



    # ========================================================
    # RETURN ALL 20 KPIs + 12 CHARTS + TABLE
    # ========================================================

    return (
        fmt_int(total),
        fmt_int(manufacturers),
        fmt_int(models),
        fmt_int(states_count),
        fmt_one(avg_range),
        fmt_money(avg_msrp),
        fmt_one(median_range),
        fmt_one(avg_age),
        fmt_pct(bev_share),
        fmt_pct(phev_share),
        fmt_pct(cafv_share),
        fmt_int(total),
        str(top_make),
        str(top_state),
        str(top_model),
        fmt_one(bev_avg_range),
        fmt_one(phev_avg_range),
        fmt_pct(priced_share),
        fmt_pct(zero_msrp_share),
        fmt_one(max_range),

        fig_year,
        fig_make,
        fig_ev,
        fig_state,
        fig_range,
        fig_msrp,
        fig_portfolio,
        fig_model,
        fig_cafv,
        fig_utility,
        fig_city,
        fig_ev_year,

        table_columns,
        table_data,
    )


# ============================================================
# 14. THEME COLOR VARIABLES
# ============================================================
#
# render_theme() above sets the CSS custom properties directly on
# #app-shell, so the page, cards and filter controls follow the
# selected theme. Plotly figures use the same THEMES dictionary.
# ============================================================

# ============================================================
# 15. CSV DOWNLOAD — CURRENT TABLE VIEW
# ============================================================

@app.callback(
    Output("download-table-csv", "data"),
    Input("download-csv-btn", "n_clicks"),
    State("summary-table", "derived_virtual_data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, rows):
    if not n_clicks:
        raise PreventUpdate

    if not rows:
        return dcc.send_data_frame(
            pd.DataFrame().to_csv,
            "ev_manufacturer_summary.csv",
            index=False,
        )

    export_df = pd.DataFrame(rows)

    return dcc.send_data_frame(
        export_df.to_csv,
        "ev_manufacturer_summary.csv",
        index=False,
    )


# ============================================================
# 16. RUN SERVER — FIXED 8051, NO INPUT
# ============================================================

if __name__ == "__main__":
    print("=" * 72)
    print("EV INTELLIGENCE DASHBOARD SERVER")
    print("=" * 72)
    print(f"Dataset : {CLEANED_FILE}")
    print(f"Records : {len(df):,}")
    print("Port    : 8050")
    print("URL     : http://127.0.0.1:8050/")
    print("Debug   : OFF")
    print("Reloader: OFF")
    print("=" * 72)
    print()
    print("Press Ctrl+C to stop the server.")
    print()

    app.run(
        host="127.0.0.1",
        port=8050,
        debug=False,
        use_reloader=False,
    )
