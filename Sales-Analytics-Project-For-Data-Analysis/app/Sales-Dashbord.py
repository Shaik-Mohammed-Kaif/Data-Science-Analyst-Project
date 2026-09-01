# ================================================================
# SALES DASHBOARD — Power BI style, Streamlit + Plotly
# Theme: Salmon / Navy (matches the reference screenshot)
# Dataset: SuperStore_Feature_Engineered.csv
# ================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ================================================================
# CONFIG
# ================================================================

st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DATA_PATH = "SuperStore_Feature_Engineered.csv"

# ================================================================
# THEMES — same 6 charts always; only colors change per theme
# ================================================================

THEMES = {
    "🍑 Salmon": {
        "bg": "#E9A47C",
        "card_bg": "#EAA97F",
        "navy": "#152238",
        "navy_light": "#22314D",
        "accent_purple": "#7B2FBE",
        "accent_pink": "#C23B7A",
        "bar_blue": "#2E6DB4",
        "bar_blue_dark": "#1B4E86",
        "donut_colors": ["#F4B942", "#2E6DB4", "#7B2FBE", "#3FA796", "#C23B7A", "#8C8C8C"],
        "text_dark": "#152238",
        "text_light": "#FFFFFF",
        "card_border": "rgba(21,34,56,0.15)",
    },
    "🍦 Cream": {
        "bg": "#F7F1E3",
        "card_bg": "#FDFBF5",
        "navy": "#4B3B2A",
        "navy_light": "#6B5843",
        "accent_purple": "#B5834D",
        "accent_pink": "#D89B4A",
        "bar_blue": "#C08A3E",
        "bar_blue_dark": "#8C6229",
        "donut_colors": ["#C08A3E", "#4B3B2A", "#D89B4A", "#7A9E7E", "#B5548A", "#A9A392"],
        "text_dark": "#4B3B2A",
        "text_light": "#FFFFFF",
        "card_border": "rgba(75,59,42,0.15)",
    },
    "🤍 White": {
        "bg": "#F5F6F8",
        "card_bg": "#FFFFFF",
        "navy": "#1F2937",
        "navy_light": "#374151",
        "accent_purple": "#6D28D9",
        "accent_pink": "#DB2777",
        "bar_blue": "#2563EB",
        "bar_blue_dark": "#1D4ED8",
        "donut_colors": ["#2563EB", "#F59E0B", "#6D28D9", "#10B981", "#DB2777", "#9CA3AF"],
        "text_dark": "#1F2937",
        "text_light": "#FFFFFF",
        "card_border": "rgba(31,41,55,0.12)",
    },
}

if "dashboard_theme" not in st.session_state:
    st.session_state.dashboard_theme = "🍑 Salmon"

T = THEMES[st.session_state.dashboard_theme]

BG_SALMON = T["bg"]
CARD_BG = T["card_bg"]
NAVY = T["navy"]
NAVY_LIGHT = T["navy_light"]
ACCENT_PURPLE = T["accent_purple"]
ACCENT_PINK = T["accent_pink"]
BAR_BLUE = T["bar_blue"]
BAR_BLUE_DARK = T["bar_blue_dark"]
DONUT_COLORS = T["donut_colors"]
TEXT_DARK = T["text_dark"]
TEXT_LIGHT = T["text_light"]
CARD_BORDER = T["card_border"]

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


# ================================================================
# GLOBAL CSS — recreate the Power BI salmon/navy look
# ================================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {BG_SALMON};
    }}

    .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }}

    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    /* ---------- Top navy title bar ---------- */
    .dash-topbar {{
        background: {NAVY};
        border-radius: 10px;
        padding: 10px 22px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }}

    .dash-title {{
        color: {TEXT_LIGHT};
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 0.3px;
        margin: 0;
    }}

    /* ---------- KPI chip ---------- */
    .kpi-chip {{
        background: {NAVY};
        border-radius: 8px;
        padding: 8px 18px;
        text-align: center;
        color: {TEXT_LIGHT};
        min-width: 90px;
    }}

    .kpi-chip .kpi-chip-value {{
        font-size: 20px;
        font-weight: 800;
        line-height: 1.1;
    }}

    .kpi-chip .kpi-chip-label {{
        font-size: 10px;
        opacity: 0.75;
        margin-top: 2px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }}

    /* ---------- Chart card ---------- */
    .chart-card {{
        background: {CARD_BG};
        border-radius: 10px;
        padding: 10px 14px 4px 14px;
        margin-bottom: 16px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.12);
        border: 1px solid {CARD_BORDER};
    }}

    .chart-card-title {{
        color: {TEXT_DARK};
        font-size: 13px;
        font-weight: 800;
        margin: 2px 0 0 4px;
        letter-spacing: 0.2px;
    }}

    /* ---------- Quarter buttons: unselected (secondary) ---------- */
    div[data-testid="column"] .stButton > button[kind="secondary"] {{
        background: {NAVY_LIGHT};
        color: {TEXT_LIGHT};
        border: none;
        border-radius: 6px;
        font-weight: 700;
        font-size: 13px;
        padding: 6px 14px;
        width: 100%;
    }}

    div[data-testid="column"] .stButton > button[kind="secondary"]:hover {{
        background: {ACCENT_PURPLE};
        color: {TEXT_LIGHT};
        border: none;
    }}

    /* ---------- Quarter buttons: selected (primary) ---------- */
    div[data-testid="column"] .stButton > button[kind="primary"] {{
        background: {ACCENT_PURPLE} !important;
        color: {TEXT_LIGHT} !important;
        border: none !important;
        border-radius: 6px;
        font-weight: 800;
        font-size: 13px;
        padding: 6px 14px;
        width: 100%;
        box-shadow: 0 0 0 2px {ACCENT_PINK} inset;
    }}

    div[data-testid="column"] .stButton > button[kind="primary"]:hover {{
        background: {ACCENT_PINK} !important;
    }}

    label, .stMarkdown, p {{
        color: {TEXT_DARK};
    }}

    /* Style ONLY the closed selectbox control (never the options
       popup) so the dropdown list always stays readable. */
    div[data-testid="stSelectbox"] > div > div {{
        background: {NAVY};
        border-radius: 6px !important;
        border: none !important;
    }}

    div[data-testid="stSelectbox"] > div > div * {{
        color: {TEXT_LIGHT} !important;
    }}

    div[data-testid="stSelectbox"] label {{
        color: {TEXT_DARK} !important;
        font-weight: 700;
        font-size: 12px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ================================================================
# HELPERS
# ================================================================

def money_k(x):
    """Format a number as Power-BI style K/M shorthand."""
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.0f}K"
    return f"{x:.0f}"


def chart_layout(fig, height=270, title=""):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=28, b=10),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_DARK, size=11, family="Segoe UI, Arial"),
        title=dict(text=title, font=dict(size=12, color=TEXT_DARK), x=0.01),
        showlegend=fig.layout.showlegend if fig.layout.showlegend is not None else False,
        legend=dict(
            font=dict(size=9, color=TEXT_DARK),
            orientation="v",
            bgcolor="rgba(0,0,0,0)"
        )
    )
    fig.update_xaxes(showgrid=False, color=TEXT_DARK)
    fig.update_yaxes(showgrid=False, color=TEXT_DARK)
    return fig


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
    return data, None


df, load_error = load_data(DATA_PATH)

if load_error:
    st.error(load_error)
    st.info(
        "Place 'SuperStore_Feature_Engineered.csv' in the same folder "
        "as this file, or change DATA_PATH at the top of the script."
    )
    st.stop()

# ---- Resolve flexible column names (works with common SuperStore schemas) ----

def first_present(candidates, columns):
    for c in candidates:
        if c in columns:
            return c
    return None

COL_AMOUNT = first_present(["Sales", "Amount", "Sum_of_Amount"], df.columns)
COL_QTY = first_present(["Quantity", "Qty"], df.columns)
COL_PROFIT = first_present(["Profit"], df.columns)
COL_CATEGORY = first_present(["Category"], df.columns)
COL_SUBCAT = first_present(["Sub_Category", "Sub-Category", "SubCategory"], df.columns)
COL_PAYMENT = first_present(["Payment_Mode", "PaymentMode", "Payment_Method"], df.columns)
COL_CUSTOMER = first_present(["Customer_Name", "CustomerName"], df.columns)
COL_YEAR = first_present(["Order_Year", "Year"], df.columns)
COL_QUARTER = first_present(["Order_Quarter", "Quarter"], df.columns)
COL_MONTH_NAME = first_present(["Order_Month_Name", "Month_Name"], df.columns)
COL_ORDER_ID = first_present(["Order_ID", "OrderID"], df.columns)

required = {
    "Amount/Sales": COL_AMOUNT,
    "Quantity": COL_QTY,
    "Profit": COL_PROFIT,
    "Category": COL_CATEGORY,
    "Sub-Category": COL_SUBCAT,
    "Payment Mode": COL_PAYMENT,
    "Customer Name": COL_CUSTOMER,
    "Order Year": COL_YEAR,
    "Order Quarter": COL_QUARTER,
    "Order Month Name": COL_MONTH_NAME,
}

missing = [k for k, v in required.items() if v is None]

if missing:
    st.error(
        "The following expected columns were not found in the CSV: "
        + ", ".join(missing)
    )
    st.write("Available columns:")
    st.code(", ".join(df.columns))
    st.stop()


# ================================================================
# SESSION STATE — active quarter / year
# ================================================================

if "active_quarter" not in st.session_state:
    st.session_state.active_quarter = "All"

years_available = sorted(df[COL_YEAR].dropna().unique().tolist())
try:
    years_available = sorted({int(y) for y in years_available})
except (ValueError, TypeError):
    pass  # non-numeric year labels (e.g. "FY2019") are kept as-is

if "active_year" not in st.session_state:
    st.session_state.active_year = years_available[0] if years_available else None


# ================================================================
# TOP BAR — Title, Quarter filters, Year, KPI chips
# ================================================================

top_left, top_qtrs, top_year, top_kpi1, top_kpi2, top_theme = st.columns(
    [1.9, 2.4, 1.3, 1, 1, 1.3]
)

with top_theme:
    chosen_theme = st.selectbox(
        "Theme",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.dashboard_theme),
        key="theme_selectbox"
    )
    if chosen_theme != st.session_state.dashboard_theme:
        st.session_state.dashboard_theme = chosen_theme
        st.rerun()

with top_left:
    st.markdown(
        f"""
        <div class="dash-topbar" style="height:100%; margin-bottom:0;">
            <span class="dash-title">📊 Sales Dashboard</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with top_qtrs:
    q_cols = st.columns(4)
    quarters_raw = sorted(df[COL_QUARTER].dropna().unique().tolist())

    def qtr_label(q):
        s = str(q)
        if s.upper().startswith("Q"):
            return s.upper()
        try:
            return f"Q{int(float(s))}"
        except Exception:
            return s

    for i, q in enumerate(quarters_raw[:4]):
        label = qtr_label(q)
        is_active = (st.session_state.active_quarter == q)
        with q_cols[i]:
            clicked = st.button(
                label,
                key=f"qtr_btn_{q}",
                width="stretch",
                type="primary" if is_active else "secondary"
            )
            if clicked:
                st.session_state.active_quarter = (
                    "All" if is_active else q
                )
                st.rerun()

with top_year:
    if years_available:
        selected_year = st.selectbox(
            "📅 Year",
            years_available,
            index=years_available.index(st.session_state.active_year)
            if st.session_state.active_year in years_available else 0,
            key="year_selectbox"
        )
        st.session_state.active_year = selected_year


# ================================================================
# APPLY FILTERS
# ================================================================

filtered = df.copy()

if st.session_state.active_year is not None:
    filtered = filtered[filtered[COL_YEAR] == st.session_state.active_year]

if st.session_state.active_quarter != "All":
    filtered = filtered[filtered[COL_QUARTER] == st.session_state.active_quarter]

if filtered.empty:
    st.warning("No records match the current Year / Quarter selection.")
    st.stop()

total_amount = filtered[COL_AMOUNT].sum()
total_profit = filtered[COL_PROFIT].sum()
total_orders = (
    filtered[COL_ORDER_ID].nunique() if COL_ORDER_ID else len(filtered)
)

with top_kpi1:
    st.markdown(
        f"""
        <div class="kpi-chip">
            <div class="kpi-chip-value">{money_k(total_orders)}</div>
            <div class="kpi-chip-label">Orders</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with top_kpi2:
    st.markdown(
        f"""
        <div class="kpi-chip">
            <div class="kpi-chip-value">{money_k(total_amount)}</div>
            <div class="kpi-chip-label">Sum of Amount</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.caption(
    f"Showing {len(filtered):,} of {len(df):,} records — "
    f"Year: {st.session_state.active_year} | "
    f"Quarter: {st.session_state.active_quarter}"
)


# ================================================================
# ROW 1 — Amount by Sub-Category | Qty & Amount by Payment Mode | Profit by Sub-Category
# ================================================================

row1_c1, row1_c2, row1_c3 = st.columns(3)

# ---- Chart 1: Sum of Amount by Sub-Category (horizontal bar) ----
with row1_c1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-card-title">Sum of Amount by Sub Category</div>',
        unsafe_allow_html=True
    )

    amt_by_subcat = (
        filtered.groupby(COL_SUBCAT, as_index=False)[COL_AMOUNT]
        .sum()
        .sort_values(COL_AMOUNT, ascending=True)
    )

    fig1 = px.bar(
        amt_by_subcat,
        x=COL_AMOUNT,
        y=COL_SUBCAT,
        orientation="h",
        text=amt_by_subcat[COL_AMOUNT].apply(money_k),
        color_discrete_sequence=[BAR_BLUE]
    )
    fig1.update_traces(textposition="outside", marker_line_width=0)
    fig1 = chart_layout(fig1, height=260)
    st.plotly_chart(fig1, width="stretch", config={"displayModeBar": False})

    st.markdown('</div>', unsafe_allow_html=True)

# ---- Chart 2: Sum of Quantity and Amount by Payment Mode (donut) ----
with row1_c2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-card-title">Sum of Quantity and Sum of '
        'Amount by PaymentMode</div>',
        unsafe_allow_html=True
    )

    pay_group = (
        filtered.groupby(COL_PAYMENT, as_index=False)
        .agg(Amount=(COL_AMOUNT, "sum"), Quantity=(COL_QTY, "sum"))
    )

    fig2 = go.Figure(
        data=[
            go.Pie(
                labels=pay_group[COL_PAYMENT],
                values=pay_group["Amount"],
                hole=0.55,
                marker=dict(colors=DONUT_COLORS),
                textinfo="percent",
                customdata=pay_group["Quantity"],
                hovertemplate=(
                    "%{label}<br>Amount: %{value:,.0f}"
                    "<br>Quantity: %{customdata:,.0f}<extra></extra>"
                )
            )
        ]
    )
    fig2.update_layout(showlegend=True)
    fig2 = chart_layout(fig2, height=260)
    st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

    st.markdown('</div>', unsafe_allow_html=True)

# ---- Chart 3: Sum of Profit by Sub-Category (horizontal bar) ----
with row1_c3:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-card-title">Sum of Profit by Sub-Category</div>',
        unsafe_allow_html=True
    )

    profit_by_subcat = (
        filtered.groupby(COL_SUBCAT, as_index=False)[COL_PROFIT]
        .sum()
        .sort_values(COL_PROFIT, ascending=True)
    )

    fig3 = px.bar(
        profit_by_subcat,
        x=COL_PROFIT,
        y=COL_SUBCAT,
        orientation="h",
        text=profit_by_subcat[COL_PROFIT].apply(money_k),
        color_discrete_sequence=[BAR_BLUE_DARK]
    )
    fig3.update_traces(textposition="outside", marker_line_width=0)
    fig3 = chart_layout(fig3, height=260)
    st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})

    st.markdown('</div>', unsafe_allow_html=True)


# ================================================================
# ROW 2 — Qty & Profit by Category | Profit by CustomerName | Qty by Month
# ================================================================

row2_c1, row2_c2, row2_c3 = st.columns(3)

# ---- Chart 4: Sum of Quantity and Sum of Profit by Category (donut) ----
with row2_c1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-card-title">Sum of Quantity and Sum of '
        'Profit by Category</div>',
        unsafe_allow_html=True
    )

    cat_group = (
        filtered.groupby(COL_CATEGORY, as_index=False)
        .agg(Profit=(COL_PROFIT, "sum"), Quantity=(COL_QTY, "sum"))
    )
    cat_group["PlotProfit"] = cat_group["Profit"].clip(lower=0)

    fig4 = go.Figure(
        data=[
            go.Pie(
                labels=cat_group[COL_CATEGORY],
                values=cat_group["PlotProfit"],
                hole=0.55,
                marker=dict(colors=DONUT_COLORS),
                textinfo="percent",
                customdata=np.stack(
                    [cat_group["Quantity"], cat_group["Profit"]], axis=-1
                ),
                hovertemplate=(
                    "%{label}<br>Profit: %{customdata[1]:,.0f}"
                    "<br>Quantity: %{customdata[0]:,.0f}<extra></extra>"
                )
            )
        ]
    )
    fig4.update_layout(showlegend=True)
    fig4 = chart_layout(fig4, height=260)
    st.plotly_chart(fig4, width="stretch", config={"displayModeBar": False})

    st.markdown('</div>', unsafe_allow_html=True)

# ---- Chart 5: Sum of Profit by CustomerName (column chart) ----
with row2_c2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-card-title">Sum of Profit by CustomerName</div>',
        unsafe_allow_html=True
    )

    top_n = 10
    cust_profit = (
        filtered.groupby(COL_CUSTOMER, as_index=False)[COL_PROFIT]
        .sum()
        .sort_values(COL_PROFIT, ascending=False)
        .head(top_n)
    )

    fig5 = px.bar(
        cust_profit,
        x=COL_CUSTOMER,
        y=COL_PROFIT,
        text=cust_profit[COL_PROFIT].apply(money_k),
        color_discrete_sequence=[BAR_BLUE]
    )
    fig5.update_traces(textposition="outside", marker_line_width=0)
    fig5.update_xaxes(tickangle=-40, tickfont=dict(size=8))
    fig5 = chart_layout(fig5, height=260)
    st.plotly_chart(fig5, width="stretch", config={"displayModeBar": False})

    st.markdown('</div>', unsafe_allow_html=True)

# ---- Chart 6: Sum of Quantity by Month (line chart) ----
with row2_c3:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-card-title">Sum of Quantity by Month</div>',
        unsafe_allow_html=True
    )

    qty_by_month = (
        filtered.groupby(COL_MONTH_NAME, as_index=False)[COL_QTY]
        .sum()
    )
    qty_by_month["_order"] = qty_by_month[COL_MONTH_NAME].apply(
        lambda m: MONTH_ORDER.index(m) if m in MONTH_ORDER else 99
    )
    qty_by_month = qty_by_month.sort_values("_order")

    fig6 = px.line(
        qty_by_month,
        x=COL_MONTH_NAME,
        y=COL_QTY,
        markers=True,
        color_discrete_sequence=[BAR_BLUE_DARK]
    )
    fig6.update_traces(line=dict(width=3), marker=dict(size=7))
    fig6 = chart_layout(fig6, height=260)
    st.plotly_chart(fig6, width="stretch", config={"displayModeBar": False})

    st.markdown('</div>', unsafe_allow_html=True)


# ================================================================
# FOOTER
# ================================================================

st.markdown(
    f"""
    <div style="text-align:center; color:{TEXT_DARK}; opacity:0.6;
    font-size:11px; margin-top:6px;">
        Sales Dashboard • Built with Streamlit &amp; Plotly
    </div>
    """,
    unsafe_allow_html=True
)