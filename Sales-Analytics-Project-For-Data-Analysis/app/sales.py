import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.units import inch
from io import BytesIO

st.set_page_config(
    page_title="SuperStore Business Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================================================
# DATA SOURCE
# ================================================================

DATA_PATH = (
    "https://raw.githubusercontent.com/"
    "Shaik-Mohammed-Kaif/"
    "Data-Science-Analyst-Project/"
    "main/"
    "Sales-Analytics-Project-For-Data-Analysis/"
    "SuperStore_Sales_Dataset/"
    "Processed/"
    "SuperStore_Feature_Engineered.csv"
)

@st.cache_data
def load_data(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    data = pd.read_csv(p)
    if "Order_Date" in data.columns:
        data["Order_Date"] = pd.to_datetime(data["Order_Date"], errors="coerce")
    for c in [
        "Sales","Quantity","Profit","Shipping_Days","Sales_Per_Unit",
        "Profit_Per_Unit","Profit_Margin","Return_Flag","Order_Year","Order_Month"
    ]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")
    return data

df = load_data(DATA_PATH)

if "theme" not in st.session_state:
    st.session_state.theme = "Cream"

THEMES = {
    "Cream": {
        "bg":"#F7F1E8","surface":"#FFFDF9","surface2":"#F3EADF",
        "text":"#2D241F","muted":"#75685E","primary":"#8B5E34",
        "accent":"#C9A66B","border":"#E5D7C8",
        "hero1":"#2E241F","hero2":"#6F4E37","plot":"simple_white"
    },
    "Midnight": {
        "bg":"#0B1120","surface":"#111827","surface2":"#172033",
        "text":"#F8FAFC","muted":"#94A3B8","primary":"#60A5FA",
        "accent":"#A78BFA","border":"#263247",
        "hero1":"#020617","hero2":"#172554","plot":"plotly_dark"
    },
    "Ocean": {
        "bg":"#EFF8FA","surface":"#FFFFFF","surface2":"#E5F3F5",
        "text":"#12343B","muted":"#52737A","primary":"#167D8D",
        "accent":"#E9C46A","border":"#D2E8EB",
        "hero1":"#073642","hero2":"#0E7490","plot":"simple_white"
    },
    "Lavender": {
        "bg":"#F7F5FC","surface":"#FFFFFF","surface2":"#F0ECF9",
        "text":"#282238","muted":"#716A82","primary":"#7257A6",
        "accent":"#C4A7E7","border":"#E3DCF1",
        "hero1":"#241B35","hero2":"#5B3F88","plot":"simple_white"
    }
}

T = THEMES[st.session_state.theme]

st.markdown(f"""
<style>

/* =========================================================
   GLOBAL FONT
   ========================================================= */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html,
body,
[class*="css"],
.stApp,
.stApp * {{
    font-family: Inter, sans-serif;
}}


/* =========================================================
   MAIN APPLICATION
   ========================================================= */

.stApp {{
    background: {T["bg"]};
    color: {T["text"]};
}}

.block-container {{
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}}


/* =========================================================
   HIDE DEFAULT STREAMLIT UI
   ========================================================= */

#MainMenu,
footer,
header {{
    visibility: hidden;
}}


/* =========================================================
   GLOBAL TEXT VISIBILITY
   ========================================================= */

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {{
    color: {T["text"]} !important;
}}

.stApp p,
.stApp span,
.stApp label {{
    color: {T["text"]};
}}

.stCaption {{
    color: {T["muted"]} !important;
}}

.stMarkdown {{
    color: {T["text"]};
}}


/* =========================================================
   HERO
   ========================================================= */

.hero {{
    background: linear-gradient(
        135deg,
        {T["hero1"]},
        {T["hero2"]}
    );

    border-radius: 24px;
    padding: 42px 45px;
    margin-bottom: 22px;
    color: #FFFFFF;

    box-shadow:
        0 18px 45px rgba(0,0,0,.16);
}}

.hero-label {{
    font-size: 11px;
    letter-spacing: 2.5px;
    font-weight: 900;
    color: #FFFFFF !important;
    opacity: .85;
}}

.hero-title {{
    font-size: 38px;
    font-weight: 900;
    margin-top: 9px;
    color: #FFFFFF !important;
    line-height: 1.15;
}}

.hero-subtitle {{
    margin-top: 12px;
    font-size: 14px;
    line-height: 1.7;
    color: #F1F5F9 !important;
}}


/* =========================================================
   SECTION LABEL
   ========================================================= */

.section-label {{
    color: {T["primary"]} !important;
    margin: 10px 0 8px;
    font-size: 11px;
    letter-spacing: 2.5px;
    font-weight: 900;
}}


/* =========================================================
   CARDS
   ========================================================= */

.card,
.kpi-card,
.module-card {{
    background: {T["surface"]};

    border: 1px solid {T["border"]};

    border-radius: 18px;

    box-shadow:
        0 8px 28px rgba(0,0,0,.055);

    color: {T["text"]};
}}

.card {{
    padding: 22px;
    margin-bottom: 16px;
}}


/* =========================================================
   KPI CARDS
   ========================================================= */

.kpi-card {{
    padding: 20px 22px;
    min-height: 135px;
}}

.kpi-label {{
    color: {T["muted"]} !important;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
}}

.kpi-value {{
    color: {T["text"]} !important;
    font-size: 28px;
    font-weight: 900;
    margin-top: 12px;
    line-height: 1.2;
}}

.kpi-description {{
    color: {T["muted"]} !important;
    font-size: 11px;
    margin-top: 5px;
}}


/* =========================================================
   MODULE CARDS
   ========================================================= */

.module-card {{
    padding: 18px;
    min-height: 115px;
}}

.module-icon {{
    font-size: 22px;
    margin-bottom: 8px;
}}

.module-title {{
    color: {T["text"]} !important;
    font-weight: 800;
    font-size: 14px;
}}

.module-description {{
    color: {T["muted"]} !important;
    font-size: 11px;
    line-height: 1.55;
    margin-top: 6px;
}}


/* =========================================================
   WELCOME SECTION
   ========================================================= */

.welcome-title {{
    font-size: 27px;
    font-weight: 900;
    color: {T["text"]} !important;
}}

.welcome-text {{
    color: {T["muted"]} !important;
    line-height: 1.8;
    font-size: 13px;
}}


/* =========================================================
   STREAMLIT SELECTBOX
   ========================================================= */

div[data-baseweb="select"] > div {{
    background-color: {T["surface"]} !important;
    border-color: {T["border"]} !important;
    color: {T["text"]} !important;
}}

div[data-baseweb="select"] span {{
    color: {T["text"]} !important;
}}

div[data-baseweb="select"] input {{
    color: {T["text"]} !important;
}}

div[data-baseweb="select"] svg {{
    fill: {T["muted"]} !important;
}}


/* =========================================================
   SELECTBOX / MULTISELECT LABELS
   ========================================================= */

[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label {{
    color: {T["text"]} !important;
    font-weight: 700 !important;
    font-size: 12px !important;
}}


/* =========================================================
   MULTISELECT TAGS
   ========================================================= */

span[data-baseweb="tag"] {{
    background-color: {T["surface2"]} !important;
    border: 1px solid {T["border"]} !important;
}}

span[data-baseweb="tag"] span {{
    color: {T["text"]} !important;
}}


/* =========================================================
   DROPDOWN POPUP
   ========================================================= */

div[data-baseweb="popover"] {{
    background-color: {T["surface"]} !important;
}}

div[data-baseweb="menu"] {{
    background-color: {T["surface"]} !important;
}}

div[data-baseweb="menu"] li {{
    background-color: {T["surface"]} !important;
    color: {T["text"]} !important;
}}

div[data-baseweb="menu"] li:hover {{
    background-color: {T["surface2"]} !important;
    color: {T["text"]} !important;
}}


/* =========================================================
   INPUTS
   ========================================================= */

.stTextInput input,
.stNumberInput input,
.stDateInput input {{
    background-color: {T["surface"]} !important;
    color: {T["text"]} !important;
    border: 1px solid {T["border"]} !important;
}}

.stTextInput input::placeholder,
.stNumberInput input::placeholder {{
    color: {T["muted"]} !important;
}}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {{
    background-color: {T["surface"]} !important;
    color: {T["text"]} !important;

    border: 1px solid {T["border"]} !important;

    border-radius: 10px;

    font-weight: 700;

    transition:
        all .2s ease;
}}

.stButton > button:hover {{
    border-color: {T["primary"]} !important;
    color: {T["primary"]} !important;
    background-color: {T["surface2"]} !important;
}}


/* =========================================================
   METRIC CARDS
   ========================================================= */

div[data-testid="stMetric"] {{
    background: {T["surface"]} !important;

    border: 1px solid {T["border"]} !important;

    border-radius: 16px;

    padding: 15px;

    box-shadow:
        0 6px 20px rgba(0,0,0,.04);
}}

div[data-testid="stMetric"] label {{
    color: {T["muted"]} !important;
    font-weight: 700 !important;
}}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: {T["text"]} !important;
    font-weight: 900 !important;
}}

div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
    color: {T["primary"]} !important;
}}


/* =========================================================
   DATAFRAME
   ========================================================= */

div[data-testid="stDataFrame"] {{
    border: 1px solid {T["border"]} !important;
    border-radius: 14px;
    overflow: hidden;
}}

div[data-testid="stDataFrame"] * {{
    color: {T["text"]} !important;
}}


/* =========================================================
   EXPANDER
   ========================================================= */

div[data-testid="stExpander"] {{
    background: {T["surface"]} !important;
    border: 1px solid {T["border"]} !important;
    border-radius: 14px;
}}

div[data-testid="stExpander"] summary {{
    color: {T["text"]} !important;
}}


/* =========================================================
   TABS
   ========================================================= */

button[data-baseweb="tab"] {{
    color: {T["muted"]} !important;
    font-weight: 700 !important;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: {T["primary"]} !important;
}}


/* =========================================================
   CHECKBOX / RADIO
   ========================================================= */

.stCheckbox label,
.stRadio label {{
    color: {T["text"]} !important;
}}


/* =========================================================
   DOWNLOAD BUTTON
   ========================================================= */

.stDownloadButton > button {{
    background-color: {T["primary"]} !important;
    color: #FFFFFF !important;

    border: none !important;

    border-radius: 10px;

    font-weight: 800;
}}

.stDownloadButton > button:hover {{
    background-color: {T["hero2"]} !important;
    color: #FFFFFF !important;
}}


/* =========================================================
   ALERTS
   ========================================================= */

div[data-testid="stAlert"] {{
    border-radius: 12px;
}}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {{
    background: {T["hero1"]};

    border-radius: 20px;

    padding: 30px;

    margin-top: 35px;

    color: #FFFFFF;

    text-align: center;
}}

.footer-name {{
    font-size: 16px;
    font-weight: 900;
    letter-spacing: 1px;
    color: #FFFFFF !important;
}}

.footer-role {{
    color: #CBD5E1 !important;
    font-size: 12px;
    margin-top: 7px;
}}

.footer-links a {{
    display: inline-block;

    color: #FFFFFF !important;

    text-decoration: none;

    border: 1px solid rgba(255,255,255,.2);

    border-radius: 9px;

    padding: 8px 14px;

    margin: 4px;

    font-size: 12px;

    font-weight: 700;

    transition: all .2s ease;
}}

.footer-links a:hover {{
    background: rgba(255,255,255,.12);
    border-color: rgba(255,255,255,.45);
}}


/* =========================================================
   PLOTLY CONTAINER
   ========================================================= */

.js-plotly-plot,
.plot-container {{
    border-radius: 14px;
    overflow: hidden;
}}


/* =========================================================
   SCROLLBAR
   ========================================================= */

::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}

::-webkit-scrollbar-track {{
    background: {T["bg"]};
}}

::-webkit-scrollbar-thumb {{
    background: {T["border"]};
    border-radius: 20px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: {T["primary"]};
}}


/* =========================================================
   MOBILE RESPONSIVE
   ========================================================= */

@media (max-width: 768px) {{

    .block-container {{
        padding-left: 1rem;
        padding-right: 1rem;
    }}

    .hero {{
        padding: 28px 24px;
        border-radius: 18px;
    }}

    .hero-title {{
        font-size: 28px;
    }}

    .hero-subtitle {{
        font-size: 12px;
    }}

    .kpi-value {{
        font-size: 24px;
    }}

    .card {{
        padding: 18px;
    }}
}}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATASET SAFETY CHECK
# ============================================================

if df.empty:
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()


# ============================================================
# PLATFORM HERO
# ============================================================

st.markdown("""
<div class="hero">
<div class="hero-label">SUPERSTORE • BUSINESS INTELLIGENCE</div>
<div class="hero-title">Sales & Profitability Intelligence Platform</div>
<div class="hero-subtitle">
Interactive Business Analytics • Statistical Analysis • Customer Intelligence •
Product Performance • Executive Insights
</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# APPLICATION MODULE / THEME / DATASET
# ============================================================

c1,c2,c3=st.columns([2.2,1.2,1.2])

with c1:

    st.markdown(
        '<div class="section-label">APPLICATION MODULE</div>',
        unsafe_allow_html=True
    )

    pages=[
        "Executive Dashboard",
        "Sales Analytics",
        "Profitability",
        "Customer Analytics",
        "Product Analytics",
        "Regional Analytics",
        "Time Series",
        "Statistical Summary",
        "Data Explorer",
        "Business Insights",
        "Reports & Export",
        "About"
    ]

    page=st.selectbox(
        "Application Module",
        pages,
        label_visibility="collapsed"
    )


with c2:

    st.markdown(
        '<div class="section-label">THEME</div>',
        unsafe_allow_html=True
    )

    theme=st.selectbox(
        "Theme",
        list(THEMES),
        index=list(THEMES).index(st.session_state.theme),
        label_visibility="collapsed"
    )

    if theme != st.session_state.theme:
        st.session_state.theme=theme
        st.rerun()


with c3:

    st.markdown(
        '<div class="section-label">DATASET</div>',
        unsafe_allow_html=True
    )

    st.metric(
        "Records",
        f"{len(df):,}"
    )


# ============================================================
# GLOBAL BUSINESS FILTERS
# SHOW ON ALL PAGES EXCEPT ABOUT
# ============================================================

# Default values
regions=[]
categories=[]
segments=[]
ships=[]
selected_years=[]


def opts(c):

    return (
        sorted(
            df[c]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if c in df.columns
        else []
    )


# ------------------------------------------------------------
# FILTER UI
# ------------------------------------------------------------

if page != "About":

    st.markdown(
        '<div class="section-label">GLOBAL BUSINESS FILTERS</div>',
        unsafe_allow_html=True
    )

    f1,f2,f3,f4,f5=st.columns(5)


    with f1:

        regions=st.multiselect(
            "Region",
            opts("Region"),
            placeholder="All Regions"
        )


    with f2:

        categories=st.multiselect(
            "Category",
            opts("Category"),
            placeholder="All Categories"
        )


    with f3:

        segments=st.multiselect(
            "Segment",
            opts("Segment"),
            placeholder="All Segments"
        )


    with f4:

        ships=st.multiselect(
            "Ship Mode",
            opts("Ship_Mode"),
            placeholder="All Ship Modes"
        )


    with f5:

        years=(
            sorted(
                df["Order_Year"]
                .dropna()
                .unique()
                .tolist()
            )
            if "Order_Year" in df.columns
            else []
        )

        selected_years=st.multiselect(
            "Year",
            years,
            placeholder="All Years"
        )


# ============================================================
# APPLY GLOBAL FILTERS
# ============================================================

filtered=df.copy()


if regions:

    filtered=filtered[
        filtered["Region"]
        .astype(str)
        .isin(regions)
    ]


if categories:

    filtered=filtered[
        filtered["Category"]
        .astype(str)
        .isin(categories)
    ]


if segments:

    filtered=filtered[
        filtered["Segment"]
        .astype(str)
        .isin(segments)
    ]


if ships:

    filtered=filtered[
        filtered["Ship_Mode"]
        .astype(str)
        .isin(ships)
    ]


if selected_years:

    filtered=filtered[
        filtered["Order_Year"]
        .isin(selected_years)
    ]


# ============================================================
# GLOBAL BUSINESS KPIs
# ============================================================

sales=(
    filtered["Sales"].sum()
    if "Sales" in filtered.columns
    else 0
)

profit=(
    filtered["Profit"].sum()
    if "Profit" in filtered.columns
    else 0
)

orders=(
    filtered["Order_ID"].nunique()
    if "Order_ID" in filtered.columns
    else len(filtered)
)

customers=(
    filtered["Customer_ID"].nunique()
    if "Customer_ID" in filtered.columns
    else 0
)

quantity=(
    filtered["Quantity"].sum()
    if "Quantity" in filtered.columns
    else 0
)

margin=(
    profit/sales*100
    if sales
    else 0
)

aov=(
    sales/orders
    if orders
    else 0
)

ppo=(
    profit/orders
    if orders
    else 0
)

def layout(fig,height=430):
    fig.update_layout(template=T["plot"],height=height,margin=dict(l=30,r=30,t=55,b=35))
    return fig

if page in ["Executive Dashboard","Sales Analytics","Profitability","Customer Analytics",
            "Product Analytics","Regional Analytics","Time Series"]:
    if page=="Executive Dashboard":
        st.markdown("""
        <div class="card"><div class="section-label">WELCOME</div>
        <div class="welcome-title">SuperStore Business Intelligence</div>
        <div class="welcome-text">A professional interactive analytics platform designed to
        transform SuperStore transactional data into actionable business intelligence.</div>
        </div>""",unsafe_allow_html=True)
        
    cols=st.columns(4)
    vals=[("TOTAL SALES",f"${sales:,.0f}","Revenue generated","💵"),
          ("TOTAL PROFIT",f"${profit:,.0f}","Business profitability","💰"),
          ("TOTAL ORDERS",f"{orders:,.0f}","Unique transactions","🧾"),
          ("CUSTOMERS",f"{customers:,.0f}","Unique customers","👥")]
    
    for col,(label,val,desc,icon) in zip(cols,vals):
        with col:
            st.markdown(f"""<div class="kpi-card">
            <div style="display:flex;justify-content:space-between">
            <div class="kpi-label">{label}</div><b style="color:{T["primary"]}">{icon}</b></div>
            <div class="kpi-value">{val}</div><div class="kpi-description">{desc}</div>
            </div>""",unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric("Quantity Sold",f"{quantity:,.0f}","📦")
    b.metric("Profit Margin",f"{margin:.2f}%","📈")
    c.metric("Average Order Value",f"${aov:,.2f}","🛒")
    d.metric("Profit per Order",f"${ppo:,.2f}","💎")

if page=="Executive Dashboard":
    left,right=st.columns([1.7,1])
    with left:
        st.subheader("Sales & Profit Trend")
        if {"Order_Year","Order_Month"}.issubset(filtered.columns):
            m=filtered.groupby(["Order_Year","Order_Month"],as_index=False).agg(
                Sales=("Sales","sum"),Profit=("Profit","sum"))
            m["Order_Year"]=m["Order_Year"].fillna(0).astype(int)
            m["Order_Month"]=m["Order_Month"].fillna(1).astype(int)
            m["Period"]=pd.to_datetime(dict(year=m["Order_Year"],month=m["Order_Month"],day=1),errors="coerce")
            m=m.sort_values("Period")
            fig=px.line(m,x="Period",y=["Sales","Profit"],markers=True)
            fig.update_traces(line={"width":3})
            st.plotly_chart(layout(fig),use_container_width=True)
    with right:
        st.subheader("Category Sales Contribution")
        cat=filtered.groupby("Category",as_index=False).agg(Sales=("Sales","sum"))
        fig=px.pie(cat,names="Category",values="Sales",hole=.58)
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(layout(fig),use_container_width=True)
    a,b=st.columns(2)
    with a:
        st.subheader("Profit & Sales - Region Category")
        r=filtered.groupby("Region",as_index=False).agg(Sales=("Sales","sum"),Profit=("Profit","sum"))
        st.plotly_chart(layout(px.bar(r,x="Region",y=["Sales","Profit"],barmode="group")),use_container_width=True)
    with b:
        st.subheader("Profit & Sales - Segment Category")
        s=filtered.groupby("Segment",as_index=False).agg(Sales=("Sales","sum"),Profit=("Profit","sum"))
        st.plotly_chart(layout(px.bar(s,x="Segment",y="Sales",color="Profit")),use_container_width=True)

elif page=="Sales Analytics":
    st.title("📈 Sales Analytics")

    # ============================================================
    # SALES PERFORMANCE OVERVIEW
    # ============================================================
    st.markdown(
        '<div class="section-label">SALES PERFORMANCE OVERVIEW</div>',
        unsafe_allow_html=True
    )

    cat=filtered.groupby("Category",as_index=False).agg(
        Sales=("Sales","sum"),
        Profit=("Profit","sum")
    )

    reg=filtered.groupby("Region",as_index=False).agg(
        Sales=("Sales","sum"),
        Profit=("Profit","sum")
    )

    seg=filtered.groupby("Segment",as_index=False).agg(
        Sales=("Sales","sum"),
        Profit=("Profit","sum")
    )

    # ------------------------------------------------------------
    # SALES SUMMARY TILES
    # ------------------------------------------------------------
    cat_sales=cat["Sales"].sum() if not cat.empty else 0
    reg_sales=reg["Sales"].sum() if not reg.empty else 0
    seg_sales=seg["Sales"].sum() if not seg.empty else 0

    top_category=(
        cat.loc[cat["Sales"].idxmax(),"Category"]
        if not cat.empty else "N/A"
    )

    top_region=(
        reg.loc[reg["Sales"].idxmax(),"Region"]
        if not reg.empty else "N/A"
    )

    top_segment=(
        seg.loc[seg["Sales"].idxmax(),"Segment"]
        if not seg.empty else "N/A"
    )

    tile1,tile2,tile3,tile4=st.columns(4)

    with tile1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">CATEGORY SALES</div>
            <div class="kpi-value">${cat_sales:,.0f}</div>
            <div class="kpi-description">Sales across all categories</div>
        </div>
        """,unsafe_allow_html=True)

    with tile2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">TOP CATEGORY</div>
            <div class="kpi-value" style="font-size:22px;">
                {top_category}
            </div>
            <div class="kpi-description">Highest sales category</div>
        </div>
        """,unsafe_allow_html=True)

    with tile3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">TOP REGION</div>
            <div class="kpi-value" style="font-size:22px;">
                {top_region}
            </div>
            <div class="kpi-description">Highest sales region</div>
        </div>
        """,unsafe_allow_html=True)

    with tile4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">TOP SEGMENT</div>
            <div class="kpi-value" style="font-size:22px;">
                {top_segment}
            </div>
            <div class="kpi-description">Highest sales segment</div>
        </div>
        """,unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)

    # ============================================================
    # ROW 1 — CATEGORY & REGION
    # ============================================================
    chart1,chart2=st.columns(2)

    with chart1:

        st.markdown("""
        <div class="card">
            <div class="section-label">CATEGORY PERFORMANCE</div>
            <div class="welcome-title" style="font-size:20px;">
                Sales by Category
            </div>
        </div>
        """,unsafe_allow_html=True)

        fig=px.bar(
            cat,
            x="Category",
            y="Sales",
            text="Sales"
        )

        fig.update_traces(
            texttemplate="$%{text:,.0f}",
            textposition="outside",
            marker_color=T["primary"]
        )

        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Sales",
            font=dict(
                family="Inter",
                color=T["text"]
            ),
            title_font=dict(color=T["text"]),
            paper_bgcolor=T["surface"],
            plot_bgcolor=T["surface"],
            xaxis=dict(
                tickfont=dict(color=T["text"]),
                title_font=dict(color=T["text"]),
                gridcolor=T["border"]
            ),
            yaxis=dict(
                tickfont=dict(color=T["text"]),
                title_font=dict(color=T["text"]),
                gridcolor=T["border"]
            )
        )

        st.plotly_chart(
            layout(fig,450),
            use_container_width=True
        )

    with chart2:

        st.markdown("""
        <div class="card">
            <div class="section-label">REGIONAL PERFORMANCE</div>
            <div class="welcome-title" style="font-size:20px;">
                Sales by Region
            </div>
        </div>
        """,unsafe_allow_html=True)

        fig=px.bar(
            reg,
            x="Region",
            y="Sales",
            text="Sales"
        )

        fig.update_traces(
            texttemplate="$%{text:,.0f}",
            textposition="outside",
            marker_color=T["accent"]
        )

        fig.update_layout(
            xaxis_title="Region",
            yaxis_title="Sales",
            font=dict(
                family="Inter",
                color=T["text"]
            ),
            paper_bgcolor=T["surface"],
            plot_bgcolor=T["surface"],
            xaxis=dict(
                tickfont=dict(color=T["text"]),
                title_font=dict(color=T["text"]),
                gridcolor=T["border"]
            ),
            yaxis=dict(
                tickfont=dict(color=T["text"]),
                title_font=dict(color=T["text"]),
                gridcolor=T["border"]
            )
        )

        st.plotly_chart(
            layout(fig,450),
            use_container_width=True
        )

    # ============================================================
    # CUSTOMER & CATEGORY ANALYSIS
    # ============================================================
    st.markdown(
        '<div class="section-label">CUSTOMER & CATEGORY SALES ANALYSIS</div>',
        unsafe_allow_html=True
    )

    chart3,chart4=st.columns(2)

    with chart3:

        st.markdown("""
        <div class="card">
            <div class="section-label">CUSTOMER SEGMENTS</div>
            <div class="welcome-title" style="font-size:20px;">
                Sales by Customer Segment
            </div>
        </div>
        """,unsafe_allow_html=True)

        fig=px.bar(
            seg,
            x="Segment",
            y="Sales",
            text="Sales"
        )

        fig.update_traces(
            texttemplate="$%{text:,.0f}",
            textposition="outside",
            marker_color=T["primary"]
        )

        fig.update_layout(
            xaxis_title="Customer Segment",
            yaxis_title="Sales",
            font=dict(
                family="Inter",
                color=T["text"]
            ),
            paper_bgcolor=T["surface"],
            plot_bgcolor=T["surface"],
            xaxis=dict(
                tickfont=dict(color=T["text"]),
                title_font=dict(color=T["text"]),
                gridcolor=T["border"]
            ),
            yaxis=dict(
                tickfont=dict(color=T["text"]),
                title_font=dict(color=T["text"]),
                gridcolor=T["border"]
            )
        )

        st.plotly_chart(
            layout(fig,450),
            use_container_width=True
        )

    with chart4:

        st.markdown("""
        <div class="card">
            <div class="section-label">CATEGORY COMPARISON</div>
            <div class="welcome-title" style="font-size:20px;">
                Category Sales vs Profit
            </div>
        </div>
        """,unsafe_allow_html=True)

        fig=px.bar(
            cat,
            x="Category",
            y=["Sales","Profit"],
            barmode="group"
        )

        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Amount",
            legend_title="Metric",
            font=dict(
                family="Inter",
                color=T["text"]
            ),
            paper_bgcolor=T["surface"],
            plot_bgcolor=T["surface"],
            legend=dict(
                font=dict(color=T["text"])
            ),
            xaxis=dict(
                tickfont=dict(color=T["text"]),
                title_font=dict(color=T["text"]),
                gridcolor=T["border"]
            ),
            yaxis=dict(
                tickfont=dict(color=T["text"]),
                title_font=dict(color=T["text"]),
                gridcolor=T["border"]
            )
        )

        st.plotly_chart(
            layout(fig,450),
            use_container_width=True
        )

    # ============================================================
    # REGIONAL SALES PERFORMANCE
    # ============================================================
    st.markdown(
        '<div class="section-label">REGIONAL SALES PERFORMANCE</div>',
        unsafe_allow_html=True
    )

    chart5,chart6=st.columns(2)

    with chart5:

        st.markdown("""
        <div class="card">
            <div class="section-label">REGION COMPARISON</div>
            <div class="welcome-title" style="font-size:20px;">
                Regional Sales vs Profit
            </div>
        </div>
        """,unsafe_allow_html=True)

        fig=px.bar(
            reg,
            x="Region",
            y=["Sales","Profit"],
            barmode="group"
        )

        fig.update_layout(
            xaxis_title="Region",
            yaxis_title="Amount",
            legend_title="Metric",
            font=dict(
                family="Inter",
                color=T["text"]
            ),
            paper_bgcolor=T["surface"],
            plot_bgcolor=T["surface"],
            legend=dict(
                font=dict(color=T["text"])
            ),
            xaxis=dict(
                tickfont=dict(color=T["text"]),
                title_font=dict(color=T["text"]),
                gridcolor=T["border"]
            ),
            yaxis=dict(
                tickfont=dict(color=T["text"]),
                title_font=dict(color=T["text"]),
                gridcolor=T["border"]
            )
        )

        st.plotly_chart(
            layout(fig,450),
            use_container_width=True
        )

    with chart6:

        st.markdown("""
        <div class="card">
            <div class="section-label">SALES CONTRIBUTION</div>
            <div class="welcome-title" style="font-size:20px;">
                Regional Sales Contribution
            </div>
        </div>
        """,unsafe_allow_html=True)

        fig=px.pie(
            reg,
            names="Region",
            values="Sales",
            hole=0.55
        )

        fig.update_traces(
            textinfo="label+percent",
            textfont=dict(
                color=T["text"]
            )
        )

        fig.update_layout(
            font=dict(
                family="Inter",
                color=T["text"]
            ),
            paper_bgcolor=T["surface"],
            plot_bgcolor=T["surface"],
            legend=dict(
                font=dict(color=T["text"])
            )
        )

        st.plotly_chart(
            layout(fig,450),
            use_container_width=True
        )

    # ============================================================
    # SALES SUMMARY
    # ============================================================
    st.markdown(
        '<div class="section-label">SALES SUMMARY</div>',
        unsafe_allow_html=True
    )

    summary=cat.copy()

    summary["Profit Margin %"]=np.where(
        summary["Sales"]!=0,
        summary["Profit"]/summary["Sales"]*100,
        0
    )

    st.dataframe(
        summary.round(2),
        use_container_width=True,
        hide_index=True
    )

elif page=="Profitability":
    st.title("💰 Profitability Intelligence")

    # ============================================================
    # PROFITABILITY PERFORMANCE OVERVIEW
    # ============================================================
    st.markdown(
        '<div class="section-label">PROFITABILITY PERFORMANCE OVERVIEW</div>',
        unsafe_allow_html=True
    )

    x=filtered.groupby("Category",as_index=False).agg(
        Sales=("Sales","sum"),
        Profit=("Profit","sum")
    )

    x["Profit_Margin_%"]=np.where(
        x.Sales!=0,
        x.Profit/x.Sales*100,
        0
    )

    # ============================================================
    # PROFITABILITY KPI CARDS
    # ============================================================
    total_sales=x["Sales"].sum() if not x.empty else 0
    total_profit=x["Profit"].sum() if not x.empty else 0
    overall_margin=(total_profit/total_sales*100) if total_sales else 0
    profitable_categories=(x["Profit"]>0).sum() if not x.empty else 0

    k1,k2,k3,k4=st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div class="kpi-label">TOTAL SALES</div>
                <div style="font-size:20px">{'💵'}</div>
            </div>
            <div class="kpi-value">${total_sales:,.0f}</div>
            <div class="kpi-description">Revenue analyzed</div>
        </div>
        """,unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div class="kpi-label">TOTAL PROFIT</div>
                <div style="font-size:20px">{'💰'}</div>
            </div>
            <div class="kpi-value">${total_profit:,.0f}</div>
            <div class="kpi-description">Profit generated</div>
        </div>
        """,unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div class="kpi-label">PROFIT MARGIN</div>
                <div style="font-size:20px">{'📈'}</div>
            </div>
            <div class="kpi-value">{overall_margin:.2f}%</div>
            <div class="kpi-description">Overall profitability</div>
        </div>
        """,unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div class="kpi-label">PROFITABLE CATEGORIES</div>
                <div style="font-size:20px">{'🏆'}</div>
            </div>
            <div class="kpi-value">{profitable_categories}</div>
            <div class="kpi-description">Categories with positive profit</div>
        </div>
        """,unsafe_allow_html=True)

    # ============================================================
    # CATEGORY PROFITABILITY SUMMARY
    # ============================================================
    st.markdown(
        '<div class="section-label">CATEGORY PROFITABILITY SUMMARY</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        x.round(2),
        use_container_width=True,
        hide_index=True
    )

    # ============================================================
    # SALES VS PROFIT + PROFIT MARGIN
    # ============================================================
    a,b=st.columns(2)

    with a:
        st.subheader("📊 Sales vs Profit")

        fig=px.bar(
            x,
            x="Category",
            y=["Sales","Profit"],
            barmode="group",
            text_auto=".2s"
        )

        st.plotly_chart(
            layout(fig,500),
            use_container_width=True
        )

    with b:
        st.subheader("📈 Profit Margin by Category")

        fig=px.bar(
            x,
            x="Category",
            y="Profit_Margin_%",
            text="Profit_Margin_%",
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        st.plotly_chart(
            layout(fig,500),
            use_container_width=True
        )

    # ============================================================
    # PROFIT CONTRIBUTION BY CATEGORY
    # ============================================================
    st.markdown(
        '<div class="section-label">PROFIT CONTRIBUTION BY CATEGORY</div>',
        unsafe_allow_html=True
    )

    fig=px.pie(
        x,
        names="Category",
        values="Profit",
        hole=.55
    )

    fig.update_traces(
        textinfo="label+percent",
        textposition="outside"
    )

    st.plotly_chart(
        layout(fig,500),
        use_container_width=True
    )

elif page=="Customer Analytics":
    st.title("👥 Customer Intelligence")

    x=filtered.groupby(
        ["Customer_ID","Customer_Name"],
        as_index=False
    ).agg(
        Sales=("Sales","sum"),
        Profit=("Profit","sum"),
        Orders=("Order_ID","nunique")
    )

    x["Profit_Margin_%"]=np.where(
        x["Sales"]!=0,
        x["Profit"]/x["Sales"]*100,
        0
    )

    # Safe positive value for Plotly bubble size
    x["Profit_Size"]=x["Profit"].abs()

    # ============================================================
    # TOP CUSTOMER PERFORMANCE
    # ============================================================
    top=x.nlargest(15,"Sales").sort_values("Sales")

    fig=px.bar(
        top,
        x="Sales",
        y="Customer_Name",
        orientation="h",
        text="Sales",
        title="Top 15 Customers by Sales"
    )

    fig.update_traces(
        texttemplate="$%{x:,.0f}",
        textposition="outside"
    )

    st.plotly_chart(
        layout(fig,600),
        use_container_width=True
    )

    # ============================================================
    # CUSTOMER SALES & PROFIT ANALYSIS
    # ============================================================
    st.markdown(
        '<div class="section-label">CUSTOMER SALES & PROFIT ANALYSIS</div>',
        unsafe_allow_html=True
    )

    a,b=st.columns(2)

    with a:
        fig=px.scatter(
            x,
            x="Sales",
            y="Profit",
            size="Profit_Size",
            color="Profit_Margin_%",
            hover_name="Customer_Name",
            title="Customer Sales vs Profit"
        )

        st.plotly_chart(
            layout(fig,500),
            use_container_width=True
        )

    with b:
        top_profit=x.nlargest(15,"Profit").sort_values("Profit")

        fig=px.bar(
            top_profit,
            x="Profit",
            y="Customer_Name",
            orientation="h",
            text="Profit",
            title="Top 15 Customers by Profit"
        )

        fig.update_traces(
            texttemplate="$%{x:,.0f}",
            textposition="outside"
        )

        st.plotly_chart(
            layout(fig,500),
            use_container_width=True
        )

    # ============================================================
    # CUSTOMER ORDER BEHAVIOR
    # ============================================================
    st.markdown(
        '<div class="section-label">CUSTOMER ORDER BEHAVIOR</div>',
        unsafe_allow_html=True
    )

    a,b=st.columns(2)

    with a:
        fig=px.histogram(
            x,
            x="Orders",
            nbins=20,
            marginal="box",
            title="Customer Order Distribution"
        )

        st.plotly_chart(
            layout(fig,450),
            use_container_width=True
        )

    with b:
        fig=px.scatter(
            x,
            x="Orders",
            y="Sales",
            size="Profit_Size",
            color="Profit_Margin_%",
            hover_name="Customer_Name",
            title="Orders vs Customer Sales"
        )

        st.plotly_chart(
            layout(fig,450),
            use_container_width=True
        )

    # ============================================================
    # CUSTOMER PROFITABILITY
    # ============================================================
    st.markdown(
        '<div class="section-label">CUSTOMER PROFITABILITY</div>',
        unsafe_allow_html=True
    )

    margin_top=x.nlargest(
        20,
        "Profit_Margin_%"
    ).sort_values("Profit_Margin_%")

    fig=px.bar(
        margin_top,
        x="Profit_Margin_%",
        y="Customer_Name",
        orientation="h",
        color="Profit_Margin_%",
        text="Profit_Margin_%",
        title="Top Customers by Profit Margin"
    )

    fig.update_traces(
        texttemplate="%{x:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        layout(fig,600),
        use_container_width=True
    )

    # ============================================================
    # CUSTOMER PERFORMANCE TABLE
    # ============================================================
    st.markdown(
        '<div class="section-label">CUSTOMER PERFORMANCE TABLE</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        x.drop(columns=["Profit_Size"]).head(50).round(2),
        use_container_width=True,
        hide_index=True
    )

elif page=="Product Analytics":
    st.title("📦 Product Intelligence")

    x=filtered.groupby(
        ["Product_ID","Product_Name","Category"],
        as_index=False
    ).agg(
        Sales=("Sales","sum"),
        Profit=("Profit","sum"),
        Quantity=("Quantity","sum")
    )

    x["Profit_Margin_%"]=np.where(
        x["Sales"]!=0,
        x["Profit"]/x["Sales"]*100,
        0
    )

    # ============================================================
    # TOP PRODUCTS BY SALES
    # ============================================================
    st.markdown(
        '<div class="section-label">TOP PRODUCT PERFORMANCE</div>',
        unsafe_allow_html=True
    )

    top=x.nlargest(15,"Sales").sort_values("Sales")

    fig=px.bar(
        top,
        x="Sales",
        y="Product_Name",
        color="Profit",
        orientation="h",
        text="Sales",
        title="Top 15 Products by Sales"
    )

    fig.update_traces(
        texttemplate="$%{x:,.0f}",
        textposition="outside"
    )

    st.plotly_chart(
        layout(fig,650),
        use_container_width=True
    )

    # ============================================================
    # PRODUCT SALES & PROFIT ANALYSIS
    # ============================================================
    st.markdown(
        '<div class="section-label">PRODUCT SALES & PROFIT ANALYSIS</div>',
        unsafe_allow_html=True
    )

    a,b=st.columns(2)

    with a:
        fig=px.scatter(
            x,
            x="Sales",
            y="Profit",
            size="Quantity",
            color="Category",
            hover_name="Product_Name",
            title="Product Sales vs Profit"
        )

        st.plotly_chart(
            layout(fig,500),
            use_container_width=True
        )

    with b:
        top_profit=x.nlargest(15,"Profit").sort_values("Profit")

        fig=px.bar(
            top_profit,
            x="Profit",
            y="Product_Name",
            orientation="h",
            color="Category",
            text="Profit",
            title="Top 15 Products by Profit"
        )

        fig.update_traces(
            texttemplate="$%{x:,.0f}",
            textposition="outside"
        )

        st.plotly_chart(
            layout(fig,500),
            use_container_width=True
        )

    # ============================================================
    # PRODUCT QUANTITY ANALYSIS
    # ============================================================
    st.markdown(
        '<div class="section-label">PRODUCT QUANTITY ANALYSIS</div>',
        unsafe_allow_html=True
    )

    top_quantity=x.nlargest(15,"Quantity").sort_values("Quantity")

    fig=px.bar(
        top_quantity,
        x="Quantity",
        y="Product_Name",
        orientation="h",
        color="Category",
        text="Quantity",
        title="Top 15 Products by Quantity Sold"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        layout(fig,600),
        use_container_width=True
    )

    # ============================================================
    # PRODUCT PROFIT MARGIN
    # ============================================================
    st.markdown(
        '<div class="section-label">PRODUCT PROFITABILITY</div>',
        unsafe_allow_html=True
    )

    margin_top=x.nlargest(
        15,
        "Profit_Margin_%"
    ).sort_values("Profit_Margin_%")

    fig=px.bar(
        margin_top,
        x="Profit_Margin_%",
        y="Product_Name",
        orientation="h",
        color="Profit_Margin_%",
        text="Profit_Margin_%",
        title="Top 15 Products by Profit Margin"
    )

    fig.update_traces(
        texttemplate="%{x:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        layout(fig,600),
        use_container_width=True
    )

    # ============================================================
    # PRODUCT PERFORMANCE TABLE
    # ============================================================
    st.markdown(
        '<div class="section-label">PRODUCT PERFORMANCE TABLE</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        x.round(2),
        use_container_width=True,
        hide_index=True
    )

elif page=="Regional Analytics":
    st.title("🌎 Regional Intelligence")

    x=filtered.groupby(
        "Region",
        as_index=False
    ).agg(
        Sales=("Sales","sum"),
        Profit=("Profit","sum"),
        Orders=("Order_ID","nunique"),
        Customers=("Customer_ID","nunique")
    )

    x["Margin_%"]=np.where(
        x["Sales"]!=0,
        x["Profit"]/x["Sales"]*100,
        0
    )

    # ============================================================
    # REGIONAL SALES PERFORMANCE
    # ============================================================
    st.markdown(
        '<div class="section-label">REGIONAL SALES PERFORMANCE</div>',
        unsafe_allow_html=True
    )

    a,b=st.columns(2)

    with a:
        fig=px.bar(
            x,
            x="Region",
            y="Sales",
            color="Profit",
            text="Sales",
            title="Regional Sales Performance"
        )

        fig.update_traces(
            texttemplate="$%{y:,.0f}",
            textposition="outside"
        )

        st.plotly_chart(
            layout(fig,500),
            use_container_width=True
        )

    with b:
        # Safe positive bubble size for Orders
        x["Orders_Size"]=x["Orders"].clip(lower=1)

        fig=px.scatter(
            x,
            x="Sales",
            y="Profit",
            size="Orders_Size",
            color="Region",
            text="Region",
            hover_data=["Orders","Customers","Margin_%"],
            title="Regional Sales vs Profit"
        )

        fig.update_traces(
            textposition="top center"
        )

        st.plotly_chart(
            layout(fig,500),
            use_container_width=True
        )

    # ============================================================
    # REGIONAL PROFITABILITY
    # ============================================================
    st.markdown(
        '<div class="section-label">REGIONAL PROFITABILITY</div>',
        unsafe_allow_html=True
    )

    a,b=st.columns(2)

    with a:
        fig=px.bar(
            x.sort_values("Profit"),
            x="Profit",
            y="Region",
            orientation="h",
            color="Profit",
            text="Profit",
            title="Profit Contribution by Region"
        )

        fig.update_traces(
            texttemplate="$%{x:,.0f}",
            textposition="outside"
        )

        st.plotly_chart(
            layout(fig,500),
            use_container_width=True
        )

    with b:
        fig=px.bar(
            x.sort_values("Margin_%"),
            x="Margin_%",
            y="Region",
            orientation="h",
            color="Margin_%",
            text="Margin_%",
            title="Profit Margin by Region"
        )

        fig.update_traces(
            texttemplate="%{x:.2f}%",
            textposition="outside"
        )

        st.plotly_chart(
            layout(fig,500),
            use_container_width=True
        )

    # ============================================================
    # REGIONAL SALES CONTRIBUTION
    # ============================================================
    st.markdown(
        '<div class="section-label">REGIONAL SALES CONTRIBUTION</div>',
        unsafe_allow_html=True
    )

    fig=px.pie(
        x,
        names="Region",
        values="Sales",
        hole=0.55,
        title="Regional Sales Contribution"
    )

    fig.update_traces(
        textinfo="label+percent"
    )

    st.plotly_chart(
        layout(fig,500),
        use_container_width=True
    )

    # ============================================================
    # REGIONAL PERFORMANCE TABLE
    # ============================================================
    st.markdown(
        '<div class="section-label">REGIONAL PERFORMANCE TABLE</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        x.drop(columns=["Orders_Size"]).round(2),
        use_container_width=True,
        hide_index=True
    )

elif page=="Time Series":
    st.title("📅 Time Series Intelligence")
    if "Order_Date" in filtered.columns:
        x=filtered.dropna(subset=["Order_Date"]).set_index("Order_Date").resample("MS").agg(
            Sales=("Sales","sum"),Profit=("Profit","sum"),Orders=("Order_ID","nunique")).reset_index()
        st.plotly_chart(layout(px.line(x,x="Order_Date",y=["Sales","Profit"],markers=True),520),use_container_width=True)
        st.plotly_chart(layout(px.bar(x,x="Order_Date",y="Orders"),420),use_container_width=True)

elif page=="Statistical Summary":
    st.title("📐 Statistical Summary")
    nums=filtered.select_dtypes(include=np.number)
    st.dataframe(nums.describe().T.round(3),use_container_width=True)
    a,b=st.columns(2)
    with a: st.plotly_chart(layout(px.histogram(filtered,x="Sales",nbins=40,marginal="box")),use_container_width=True)
    with b: st.plotly_chart(layout(px.histogram(filtered,x="Profit",nbins=40,marginal="box")),use_container_width=True)
    st.plotly_chart(layout(px.imshow(nums.corr(numeric_only=True),text_auto=".2f",aspect="auto"),650),use_container_width=True)

elif page=="Data Explorer":
    st.title("🔎 Data Explorer")

    st.markdown(
        '<div class="section-label">FILTERED DATASET OVERVIEW</div>',
        unsafe_allow_html=True
    )

    st.write(f"Showing **{len(filtered):,}** filtered records.")

    csv_data=filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Export Filtered CSV",
        data=csv_data,
        file_name="SuperStore_Filtered_Data.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown(
        '<div class="section-label">DATA PREVIEW</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        height=600,
        hide_index=True
    )

elif page=="Business Insights":

    st.title("💡 Business Intelligence Insights")

    # ============================================================
    # BUSINESS PERFORMANCE OVERVIEW
    # ============================================================
    st.markdown("### 📊 Business Performance Overview")

    cat=filtered.groupby("Category",as_index=False).agg(
        Sales=("Sales","sum"),
        Profit=("Profit","sum")
    )

    reg=filtered.groupby("Region",as_index=False).agg(
        Sales=("Sales","sum"),
        Profit=("Profit","sum")
    )

    prod=filtered.groupby("Product_Name",as_index=False).agg(
        Sales=("Sales","sum"),
        Profit=("Profit","sum")
    )

    seg=filtered.groupby("Segment",as_index=False).agg(
        Sales=("Sales","sum"),
        Profit=("Profit","sum")
    )

    # ============================================================
    # BUSINESS LEADERS
    # ============================================================
    bestcat=cat.loc[cat.Sales.idxmax(),"Category"] if not cat.empty else "N/A"
    bestreg=reg.loc[reg.Sales.idxmax(),"Region"] if not reg.empty else "N/A"
    bestprod=prod.loc[prod.Sales.idxmax(),"Product_Name"] if not prod.empty else "N/A"
    bestseg=seg.loc[seg.Sales.idxmax(),"Segment"] if not seg.empty else "N/A"

    bestcat_sales=cat.loc[cat.Sales.idxmax(),"Sales"] if not cat.empty else 0
    bestreg_sales=reg.loc[reg.Sales.idxmax(),"Sales"] if not reg.empty else 0
    bestprod_sales=prod.loc[prod.Sales.idxmax(),"Sales"] if not prod.empty else 0
    bestseg_sales=seg.loc[seg.Sales.idxmax(),"Sales"] if not seg.empty else 0

    # ============================================================
    # PROFITABILITY ANALYSIS
    # ============================================================
    profitable_categories=cat[cat["Profit"]>0]
    loss_categories=cat[cat["Profit"]<0]

    most_profitable_category=(
        cat.loc[cat.Profit.idxmax(),"Category"]
        if not cat.empty else "N/A"
    )

    most_profitable_category_profit=(
        cat.loc[cat.Profit.idxmax(),"Profit"]
        if not cat.empty else 0
    )

    lowest_profit_category=(
        cat.loc[cat.Profit.idxmin(),"Category"]
        if not cat.empty else "N/A"
    )

    lowest_profit_category_profit=(
        cat.loc[cat.Profit.idxmin(),"Profit"]
        if not cat.empty else 0
    )

    # ============================================================
    # BUSINESS KPI CARDS
    # ============================================================
    i1,i2,i3,i4=st.columns(4)

    with i1:
        st.metric(
            "🏆 Top Category",
            bestcat,
            f"${bestcat_sales:,.0f} Sales"
        )

    with i2:
        st.metric(
            "🌎 Top Region",
            bestreg,
            f"${bestreg_sales:,.0f} Sales"
        )

    with i3:
        st.metric(
            "👥 Top Customer Segment",
            bestseg,
            f"${bestseg_sales:,.0f} Sales"
        )

    with i4:
        product_short=(
            bestprod
            if len(str(bestprod))<=24
            else str(bestprod)[:24]+"..."
        )

        st.metric(
            "📦 Top Product",
            product_short,
            f"${bestprod_sales:,.0f} Sales"
        )

    # ============================================================
    # SALES LEADERSHIP
    # ============================================================
    st.markdown("### 🏆 Sales Leadership")

    st.markdown(
        f"""
        **Who is driving business performance?**

        The **{bestcat}** category is the leading category with
        **${bestcat_sales:,.2f}** in sales.

        The **{bestreg}** region is the strongest regional market with
        **${bestreg_sales:,.2f}** in sales.

        The **{bestseg}** customer segment contributes the highest sales
        with **${bestseg_sales:,.2f}**.

        The highest-sales product is **{bestprod}**, generating
        **${bestprod_sales:,.2f}**.
        """
    )

    # ============================================================
    # EXECUTIVE KPI INTERPRETATION
    # ============================================================
    st.markdown("### 📊 Executive KPI Interpretation")

    st.markdown(
        f"""
        **What is the overall business performance?**

        The filtered dataset contains **{orders:,.0f}** unique orders
        generated by **{customers:,.0f}** customers.

        These transactions generated total sales of
        **${sales:,.2f}** and total profit of
        **${profit:,.2f}**.

        The resulting overall profit margin is
        **{margin:.2f}%**.

        The average order value is **${aov:,.2f}** per order.
        """
    )

    # ============================================================
    # PROFITABILITY INSIGHTS
    # ============================================================
    st.markdown("### 💰 Profitability Insights")

    st.markdown(
        f"""
        **Which category is most profitable?**

        **{most_profitable_category}** is the most profitable category,
        generating **${most_profitable_category_profit:,.2f}** in profit.

        There are **{len(profitable_categories)}** categories with
        positive profit and **{len(loss_categories)}** categories with
        negative profit.
        """
    )

    if lowest_profit_category_profit<0:
        st.markdown(
            f"""
            **⚠️ Where is profitability weakest?**

            **{lowest_profit_category}** has the lowest profit contribution
            at **${lowest_profit_category_profit:,.2f}**.

            This area should be reviewed for pricing, discounts,
            product mix and operating costs.
            """
        )

    # ============================================================
    # BUSINESS QUESTIONS & ANSWERS
    # ============================================================
    st.markdown("### ❓ Business Questions & Answers")

    q1,q2=st.columns(2)

    with q1:

        st.markdown("#### 1️⃣ Which category generates the most sales?")
        st.write(
            f"**Answer:** **{bestcat}** generates the highest sales with "
            f"**${bestcat_sales:,.2f}**."
        )

        st.markdown("#### 2️⃣ Which region performs best?")
        st.write(
            f"**Answer:** **{bestreg}** is the leading region with "
            f"**${bestreg_sales:,.2f}** in sales."
        )

        st.markdown("#### 3️⃣ Which customer segment contributes most?")
        st.write(
            f"**Answer:** **{bestseg}** contributes the highest sales with "
            f"**${bestseg_sales:,.2f}**."
        )

        st.markdown("#### 4️⃣ Which product has the highest sales?")
        st.write(
            f"**Answer:** **{bestprod}** is the highest-sales product with "
            f"**${bestprod_sales:,.2f}**."
        )

    with q2:

        st.markdown("#### 5️⃣ How profitable is the business?")
        st.write(
            f"**Answer:** The current overall profit margin is "
            f"**{margin:.2f}%**, with total profit of "
            f"**${profit:,.2f}**."
        )

        st.markdown("#### 6️⃣ What is the average order value?")
        st.write(
            f"**Answer:** The average order value is "
            f"**${aov:,.2f}** per order."
        )

        st.markdown("#### 7️⃣ Which category is most profitable?")
        st.write(
            f"**Answer:** **{most_profitable_category}** is the most profitable "
            f"category with **${most_profitable_category_profit:,.2f}** "
            f"in profit."
        )

        st.markdown("#### 8️⃣ Where is profitability weakest?")
        st.write(
            f"**Answer:** **{lowest_profit_category}** has the lowest category "
            f"profit contribution at "
            f"**${lowest_profit_category_profit:,.2f}**."
        )

    # ============================================================
    # BUSINESS PERFORMANCE VISUALIZATION
    # ============================================================
    st.markdown("### 📈 Business Performance Visualization")

    a,b=st.columns(2)

    with a:
        st.markdown("#### 📈 Category Sales vs Profit")

        fig=px.bar(
            cat,
            x="Category",
            y=["Sales","Profit"],
            barmode="group",
            text_auto=".2s"
        )

        st.plotly_chart(
            layout(fig,450),
            use_container_width=True
        )

    with b:
        st.markdown("#### 🌎 Regional Sales vs Profit")

        fig=px.bar(
            reg,
            x="Region",
            y=["Sales","Profit"],
            barmode="group",
            text_auto=".2s"
        )

        st.plotly_chart(
            layout(fig,450),
            use_container_width=True
        )

    # ============================================================
    # EXECUTIVE RECOMMENDATIONS
    # ============================================================
    st.markdown("### 🎯 Executive Recommendations")

    recommendations=[
        f"Focus on the {bestcat} category because it currently leads sales.",
        f"Prioritize the {bestreg} region for sales expansion and customer growth.",
        f"Continue analyzing the {bestseg} segment because it contributes the highest sales.",
        (
            f"Review {lowest_profit_category} for pricing, discounts and cost optimization."
            if lowest_profit_category_profit<0
            else "Continue monitoring category-level profitability."
        ),
        f"Protect the performance of the top product: {bestprod}."
    ]

    for i,recommendation in enumerate(recommendations,1):
        st.markdown(f"**Recommendation {i}:** {recommendation}") 

elif page=="Reports & Export":

    st.title("📥 Reports & Export")

    # ============================================================
    # REPORT OVERVIEW
    # ============================================================

    st.markdown("### 📊 Business Report Overview")

    st.write(
        f"Current filtered business view contains "
        f"**{len(filtered):,} records**."
    )

    st.info(
        "The CSV download contains the complete original SuperStore "
        "dataset. The PDF report is generated from the current dashboard "
        "filters and presents the selected business view."
    )

    # ============================================================
    # CURRENT FILTERED REPORT METRICS
    # ============================================================

    report_sales = (
        filtered["Sales"].sum()
        if "Sales" in filtered.columns else 0
    )

    report_profit = (
        filtered["Profit"].sum()
        if "Profit" in filtered.columns else 0
    )

    report_quantity = (
        filtered["Quantity"].sum()
        if "Quantity" in filtered.columns else 0
    )

    report_orders = (
        filtered["Order_ID"].nunique()
        if "Order_ID" in filtered.columns else 0
    )

    report_customers = (
        filtered["Customer_ID"].nunique()
        if "Customer_ID" in filtered.columns else 0
    )

    report_margin = (
        report_profit / report_sales * 100
        if report_sales != 0 else 0
    )

    report_aov = (
        report_sales / report_orders
        if report_orders != 0 else 0
    )

    report_profit_per_order = (
        report_profit / report_orders
        if report_orders != 0 else 0
    )

    # ============================================================
    # EXPORT SECTION
    # ============================================================

    st.markdown("### 📤 Export Business Data & Report")

    col1, col2 = st.columns(2)

    # ============================================================
    # FULL ORIGINAL CSV DOWNLOAD
    # ============================================================

    with col1:

        # IMPORTANT:
        # df = original complete dataset
        # filtered = dashboard-filtered dataset
        #
        # Therefore this button downloads the FULL dataset.

        full_csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Full CSV",
            data=full_csv_data,
            file_name="SuperStore_Full_Dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            f"Complete dataset • {len(df):,} records"
        )

    # ============================================================
    # BUSINESS PDF GENERATOR
    # ============================================================

    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=14,
        spaceAfter=6
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=14,
        spaceAfter=5
    )

    small_style = ParagraphStyle(
        "ReportSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11
    )

    story = []

    # ============================================================
    # PDF TITLE
    # ============================================================

    story.append(
        Paragraph(
            "SUPERSTORE • BUSINESS INTELLIGENCE",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Sales & Profitability Intelligence Platform",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            "Professional Business Performance Report",
            subtitle_style
        )
    )

    story.append(Spacer(1, 10))

    # ============================================================
    # REPORT SCOPE
    # ============================================================

    story.append(
        Paragraph(
            "Report Scope",
            heading_style
        )
    )

    story.append(
        Paragraph(
            f"""
            This report summarizes the current dashboard selection.
            The selected business view contains <b>{len(filtered):,}</b>
            records, <b>{report_orders:,}</b> unique orders and
            <b>{report_customers:,}</b> unique customers.
            """,
            body_style
        )
    )

    # ============================================================
    # EXECUTIVE SUMMARY
    # ============================================================

    story.append(
        Paragraph(
            "Executive Summary",
            heading_style
        )
    )

    story.append(
        Paragraph(
            f"""
            The selected SuperStore business data generated total sales of
            <b>${report_sales:,.2f}</b> and total profit of
            <b>${report_profit:,.2f}</b>.
            The resulting profit margin is
            <b>{report_margin:.2f}%</b>.
            The business generated an average order value of
            <b>${report_aov:,.2f}</b> and profit per order of
            <b>${report_profit_per_order:,.2f}</b>.
            """,
            body_style
        )
    )

    # ============================================================
    # KPI TABLE
    # ============================================================

    story.append(
        Paragraph(
            "Key Performance Indicators",
            heading_style
        )
    )

    kpi_data = [
        ["Metric", "Value"],
        ["Total Sales", f"${report_sales:,.2f}"],
        ["Total Profit", f"${report_profit:,.2f}"],
        ["Unique Orders", f"{report_orders:,}"],
        ["Unique Customers", f"{report_customers:,}"],
        ["Quantity Sold", f"{report_quantity:,}"],
        ["Profit Margin", f"{report_margin:.2f}%"],
        ["Average Order Value", f"${report_aov:,.2f}"],
        ["Profit per Order", f"${report_profit_per_order:,.2f}"]
    ]

    kpi_table = Table(
        kpi_data,
        colWidths=[3.2 * inch, 2.5 * inch]
    )

    kpi_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1F2937")
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
                "FONTNAME",
                (0, 1),
                (-1, -1),
                "Helvetica"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#F3F4F6")
                ]
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "ALIGN",
                (1, 1),
                (-1, -1),
                "RIGHT"
            )
        ])
    )

    story.append(kpi_table)

    # ============================================================
    # CATEGORY ANALYSIS
    # ============================================================

    if "Category" in filtered.columns:

        category_report = (
            filtered
            .groupby("Category", as_index=False)
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
            .sort_values(
                "Sales",
                ascending=False
            )
        )

        if not category_report.empty:

            top_category = category_report.iloc[0]

            most_profitable_category = (
                category_report
                .loc[
                    category_report["Profit"].idxmax(),
                    "Category"
                ]
            )

            most_profitable_category_profit = (
                category_report["Profit"].max()
            )

            story.append(
                Paragraph(
                    "Category Performance",
                    heading_style
                )
            )

            story.append(
                Paragraph(
                    f"""
                    <b>{top_category["Category"]}</b> is the
                    highest-sales category with sales of
                    <b>${top_category["Sales"]:,.2f}</b> and profit of
                    <b>${top_category["Profit"]:,.2f}</b>.
                    The most profitable category is
                    <b>{most_profitable_category}</b>, generating
                    <b>${most_profitable_category_profit:,.2f}</b>
                    in profit.
                    """,
                    body_style
                )
            )

            category_data = [
                ["Category", "Sales", "Profit"]
            ]

            for _, row in category_report.iterrows():

                category_data.append([
                    str(row["Category"]),
                    f"${row['Sales']:,.2f}",
                    f"${row['Profit']:,.2f}"
                ])

            category_table = Table(
                category_data,
                colWidths=[
                    2.2 * inch,
                    1.7 * inch,
                    1.7 * inch
                ]
            )

            category_table.setStyle(
                TableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#1F2937")
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
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),
                    (
                        "ALIGN",
                        (1, 1),
                        (-1, -1),
                        "RIGHT"
                    )
                ])
            )

            story.append(Spacer(1, 8))
            story.append(category_table)

    # ============================================================
    # REGIONAL ANALYSIS
    # ============================================================

    if "Region" in filtered.columns:

        region_report = (
            filtered
            .groupby("Region", as_index=False)
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
            .sort_values(
                "Sales",
                ascending=False
            )
        )

        if not region_report.empty:

            top_region = region_report.iloc[0]

            story.append(
                Paragraph(
                    "Regional Performance",
                    heading_style
                )
            )

            story.append(
                Paragraph(
                    f"""
                    The leading region is
                    <b>{top_region["Region"]}</b> with sales of
                    <b>${top_region["Sales"]:,.2f}</b> and profit of
                    <b>${top_region["Profit"]:,.2f}</b>.
                    """,
                    body_style
                )
            )

            region_data = [
                ["Region", "Sales", "Profit"]
            ]

            for _, row in region_report.iterrows():

                region_data.append([
                    str(row["Region"]),
                    f"${row['Sales']:,.2f}",
                    f"${row['Profit']:,.2f}"
                ])

            region_table = Table(
                region_data,
                colWidths=[
                    2.2 * inch,
                    1.7 * inch,
                    1.7 * inch
                ]
            )

            region_table.setStyle(
                TableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#1F2937")
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
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),
                    (
                        "ALIGN",
                        (1, 1),
                        (-1, -1),
                        "RIGHT"
                    )
                ])
            )

            story.append(Spacer(1, 8))
            story.append(region_table)

    # ============================================================
    # CUSTOMER SEGMENT ANALYSIS
    # ============================================================

    if "Segment" in filtered.columns:

        segment_report = (
            filtered
            .groupby("Segment", as_index=False)
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
            .sort_values(
                "Sales",
                ascending=False
            )
        )

        if not segment_report.empty:

            top_segment = segment_report.iloc[0]

            story.append(
                Paragraph(
                    "Customer Segment Performance",
                    heading_style
                )
            )

            story.append(
                Paragraph(
                    f"""
                    The leading customer segment is
                    <b>{top_segment["Segment"]}</b> with sales of
                    <b>${top_segment["Sales"]:,.2f}</b> and profit of
                    <b>${top_segment["Profit"]:,.2f}</b>.
                    """,
                    body_style
                )
            )

    # ============================================================
    # PRODUCT PERFORMANCE
    # ============================================================

    if "Product_Name" in filtered.columns:

        product_report = (
            filtered
            .groupby("Product_Name", as_index=False)
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
            .sort_values(
                "Sales",
                ascending=False
            )
        )

        if not product_report.empty:

            top_product = product_report.iloc[0]

            story.append(
                Paragraph(
                    "Product Performance",
                    heading_style
                )
            )

            story.append(
                Paragraph(
                    f"""
                    The highest-sales product is
                    <b>{top_product["Product_Name"]}</b>,
                    generating sales of
                    <b>${top_product["Sales"]:,.2f}</b> and profit of
                    <b>${top_product["Profit"]:,.2f}</b>.
                    """,
                    body_style
                )
            )

    # ============================================================
    # BUSINESS QUESTIONS
    # ============================================================

    story.append(
        Paragraph(
            "Business Questions & Answers",
            heading_style
        )
    )

    if "Category" in filtered.columns and not category_report.empty:

        story.append(
            Paragraph(
                f"""
                <b>Which category generates the most sales?</b><br/>
                {top_category["Category"]} generates the highest sales
                at ${top_category["Sales"]:,.2f}.
                """,
                body_style
            )
        )

    if "Region" in filtered.columns and not region_report.empty:

        story.append(
            Paragraph(
                f"""
                <b>Which region performs best?</b><br/>
                {top_region["Region"]} is the leading region with
                ${top_region["Sales"]:,.2f} in sales.
                """,
                body_style
            )
        )

    if "Segment" in filtered.columns and not segment_report.empty:

        story.append(
            Paragraph(
                f"""
                <b>Which customer segment contributes most?</b><br/>
                {top_segment["Segment"]} contributes the highest sales
                at ${top_segment["Sales"]:,.2f}.
                """,
                body_style
            )
        )

    story.append(
        Paragraph(
            f"""
            <b>How profitable is the business?</b><br/>
            The current profit margin is {report_margin:.2f}%,
            with total profit of ${report_profit:,.2f}.
            """,
            body_style
        )
    )

    story.append(
        Paragraph(
            f"""
            <b>What is the average order value?</b><br/>
            The average order value is ${report_aov:,.2f}.
            """,
            body_style
        )
    )

    # ============================================================
    # BUSINESS RECOMMENDATIONS
    # ============================================================

    story.append(
        Paragraph(
            "Executive Recommendations",
            heading_style
        )
    )

    recommendations = [
        "Monitor revenue and profitability together rather than evaluating sales alone.",
        "Prioritize high-performing categories and regions for growth opportunities.",
        "Review low-profit products and categories for pricing, discount and cost optimization.",
        "Analyze customer segments regularly to identify high-value customer opportunities.",
        "Use dashboard filters to perform focused business analysis before making decisions."
    ]

    for i, recommendation in enumerate(
        recommendations,
        1
    ):

        story.append(
            Paragraph(
                f"<b>{i}.</b> {recommendation}",
                body_style
            )
        )

    # ============================================================
    # AUTHOR / FOOTER
    # ============================================================

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "<b>S MOHAMMED KAIF</b>",
            body_style
        )
    )

    story.append(
        Paragraph(
            "Data Science • Machine Learning • Data Analytics",
            body_style
        )
    )

    story.append(
        Paragraph(
            "SuperStore Business Intelligence Platform • 2026",
            small_style
        )
    )

    story.append(
        Paragraph(
            "LinkedIn: linkedin.com/in/s-mohammed-kaif-2a500a341",
            small_style
        )
    )

    story.append(
        Paragraph(
            "GitHub: github.com/Shaik-Mohammed-Kaif",
            small_style
        )
    )

    # ============================================================
    # BUILD PDF
    # ============================================================

    doc.build(story)

    pdf_data = pdf_buffer.getvalue()

    # ============================================================
    # PDF DOWNLOAD BUTTON
    # ============================================================

    with col2:

        st.download_button(
            "📄 Download Business Overview PDF",
            data=pdf_data,
            file_name="SuperStore_Business_Overview.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        st.caption(
            "Professional business overview • Current filters"
        )

    # ============================================================
    # EXPORT INFORMATION
    # ============================================================

    st.markdown("### 📋 Export Information")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.metric(
            "Full Dataset Records",
            f"{len(df):,}"
        )

    with info2:
        st.metric(
            "Current View Records",
            f"{len(filtered):,}"
        )

    with info3:
        st.metric(
            "Current Orders",
            f"{report_orders:,}"
        )

    # ============================================================
    # CURRENT FILTERED DATA PREVIEW
    # ============================================================

    st.markdown("### 👀 Current Business Data Preview")

    st.dataframe(
        filtered.head(100),
        use_container_width=True,
        hide_index=True
    )

elif page=="About":

    st.title("ℹ️ About the Platform")

    # ============================================================
    # PLATFORM OVERVIEW
    # ============================================================

    st.markdown("### 🏢 SuperStore Business Intelligence")

    st.write(
        "SuperStore Business Intelligence is a professional analytics "
        "platform designed to transform transactional business data into "
        "clear, interactive and decision-ready business insights."
    )

    st.write(
        "The platform combines interactive dashboards, KPI monitoring, "
        "statistical analysis, customer intelligence, product performance, "
        "profitability analysis, regional analysis, time-series analysis "
        "and executive business reporting."
    )

    # ============================================================
    # PROJECT OBJECTIVE
    # ============================================================

    st.markdown("### 🎯 Project Objective")

    st.write(
        "The primary objective of this project is to provide a centralized "
        "business intelligence environment where users can explore sales "
        "performance, understand profitability, identify important "
        "customers and products, compare regions and categories, and "
        "support data-driven business decisions."
    )

    st.write(
        "Instead of relying only on static spreadsheets or reports, the "
        "platform provides an interactive analytical workflow where "
        "business users can apply filters and immediately analyze the "
        "resulting business performance."
    )

    # ============================================================
    # BUSINESS PROBLEMS ADDRESSED
    # ============================================================

    st.markdown("### 💼 Business Problems Addressed")

    business_problems = [
        "Difficulty understanding overall sales and profit performance.",
        "Limited visibility into category-level profitability.",
        "Difficulty identifying high-performing regions.",
        "Difficulty identifying high-value customers.",
        "Difficulty identifying top-performing products.",
        "Limited understanding of order and customer behavior.",
        "Difficulty comparing sales against profitability.",
        "Limited visibility into monthly and yearly business trends.",
        "Difficulty converting transactional data into actionable insights.",
        "Dependence on static reports for business decision-making."
    ]

    for problem in business_problems:
        st.write(f"• {problem}")

    # ============================================================
    # ANALYTICS COVERAGE
    # ============================================================

    st.markdown("### 📊 Analytics Coverage")

    analytics = [
        ("💰 Sales Analytics",
         "Analyze total sales, sales trends and sales performance."),
        ("📈 Profitability Analytics",
         "Evaluate profit, profit margin and profit contribution."),
        ("👥 Customer Analytics",
         "Analyze customer sales, profit and order behavior."),
        ("📦 Product Analytics",
         "Identify top products and evaluate product performance."),
        ("🌎 Regional Analytics",
         "Compare sales, profit, orders and customers across regions."),
        ("📅 Time-Series Analytics",
         "Analyze sales, profit and order trends over time."),
        ("📐 Statistical Analysis",
         "Explore descriptive statistics, distributions and correlations."),
        ("💡 Business Intelligence",
         "Translate analytical results into business questions, answers and recommendations.")
    ]

    for title, description in analytics:
        st.markdown(f"#### {title}")
        st.write(description)

    # ============================================================
    # KEY BUSINESS QUESTIONS
    # ============================================================

    st.markdown("### ❓ Key Business Questions")

    questions = [
        "Which category generates the highest sales?",
        "Which category generates the highest profit?",
        "Which region performs best?",
        "Which customer segment contributes the most sales?",
        "Which products generate the highest sales?",
        "How profitable is the business?",
        "What is the average order value?",
        "Which areas require profitability improvement?",
        "How does business performance change over time?",
        "Where should management focus growth and optimization efforts?"
    ]

    for question in questions:
        st.write(f"• {question}")

    # ============================================================
    # PLATFORM MODULES
    # ============================================================

    st.markdown("### 🧩 Platform Modules")

    modules = [
        "🏠 Executive Dashboard",
        "💰 Profitability Intelligence",
        "👥 Customer Intelligence",
        "📦 Product Intelligence",
        "🌎 Regional Intelligence",
        "📅 Time-Series Intelligence",
        "📐 Statistical Summary",
        "🔎 Data Explorer",
        "💡 Business Intelligence Insights",
        "📥 Reports & Export",
        "ℹ️ About the Platform"
    ]

    for module in modules:
        st.write(f"• {module}")

    # ============================================================
    # TECHNOLOGY STACK
    # ============================================================

    st.markdown("### 🛠️ Technology Stack")

    tech1, tech2 = st.columns(2)

    with tech1:
        st.markdown("#### 🐍 Programming & Data")
        st.write("• Python")
        st.write("• Pandas")
        st.write("• NumPy")

    with tech2:
        st.markdown("#### 📊 Visualization & Application")
        st.write("• Plotly")
        st.write("• Streamlit")
        st.write("• ReportLab")

    # ============================================================
    # DATA WORKFLOW
    # ============================================================

    st.markdown("### 🔄 Data Analytics Workflow")

    workflow = [
        "1. Load the SuperStore CSV dataset.",
        "2. Validate and prepare the transactional data.",
        "3. Apply global business filters.",
        "4. Calculate business KPIs.",
        "5. Perform category, customer, product and regional analysis.",
        "6. Analyze time-based performance.",
        "7. Perform statistical analysis.",
        "8. Visualize analytical results.",
        "9. Generate business questions and answers.",
        "10. Produce executive insights and recommendations.",
        "11. Export the complete dataset and business report."
    ]

    for step in workflow:
        st.write(step)

    # ============================================================
    # KPI FRAMEWORK
    # ============================================================

    st.markdown("### 📌 Key Performance Indicators")

    kpis = [
        "Total Sales",
        "Total Profit",
        "Total Orders",
        "Unique Customers",
        "Quantity Sold",
        "Profit Margin",
        "Average Order Value",
        "Profit per Order"
    ]

    for kpi in kpis:
        st.write(f"• {kpi}")

    # ============================================================
    # BUSINESS VALUE
    # ============================================================

    st.markdown("### 🚀 Business Value")

    st.write(
        "The platform helps convert raw transactional records into "
        "structured business intelligence. It provides a single analytical "
        "environment for discovering performance patterns, comparing "
        "business dimensions, identifying opportunities and supporting "
        "management-level decisions."
    )

    st.write(
        "The interactive nature of the dashboard allows users to move "
        "from high-level KPIs to detailed customer, product, regional and "
        "statistical analysis without changing the underlying dataset."
    )

    # ============================================================
    # PROJECT TYPE
    # ============================================================

    st.markdown("### 🎓 Project Classification")

    st.write(
        "Data Science • Data Analytics • Business Intelligence • "
        "Interactive Dashboard • Statistical Analysis • Executive Reporting"
    )

    # ============================================================
    # AUTHOR
    # ============================================================

    st.markdown("### 👨‍💻 Project Author")

    st.write("**S Mohammed Kaif**")
    st.write("Data Science • Machine Learning • Data Analytics")

    st.write(
        "This project demonstrates practical application of Python-based "
        "data analysis, visualization, business intelligence and "
        "interactive dashboard development."
    )

    # ============================================================
    # FOOTER
    # ============================================================

    st.markdown("---")

    st.markdown(
        "**SuperStore Business Intelligence Platform • 2026**"
    )

    st.write(
        "Built with Python • Pandas • NumPy • Plotly • Streamlit"
    )

    st.write(
        "🔗 LinkedIn: linkedin.com/in/s-mohammed-kaif-2a500a341"
    )

    st.write(
        "💻 GitHub: github.com/Shaik-Mohammed-Kaif"
    )

if page=="Executive Dashboard":
    st.markdown('<div class="section-label">📊 ANALYTICS COVERAGE</div>',unsafe_allow_html=True)
    modules=[
        ("📈","Sales Intelligence","Analyze sales trends, categories, products, regions and customer segments."),
        ("💰","Profitability Intelligence","Understand profit contribution, margins, profitable products and loss-making areas."),
        ("📐","Statistical Analysis","Explore descriptive statistics, distributions, correlations and business metrics."),
        ("👥","Customer Intelligence","Understand customer value, purchasing behavior and revenue contribution.")
    ]
    cols=st.columns(4)
    for col,(icon,title,desc) in zip(cols,modules):
        with col:
            st.markdown(f"""<div class="module-card"><div class="module-icon">{icon}</div>
            <div class="module-title">{title}</div><div class="module-description">{desc}</div></div>""",
            unsafe_allow_html=True)

    st.markdown('<div class="section-label">🚀 PLATFORM MODULES</div>',unsafe_allow_html=True)
    module_grid=[("📊","Executive Dashboard"),("📈","Sales Analytics"),("💰","Profitability"),
                 ("👥","Customer Analytics"),("📦","Product Analytics"),("🌎","Regional Analytics"),
                 ("📅","Time Series"),("📐","Statistical Summary"),("🔎","Data Explorer"),
                 ("💡","Business Insights"),("📥","Reports & Export"),("ℹ️","About")]
    for i in range(0,len(module_grid),4):
        cols=st.columns(4)
        for col,(icon,title) in zip(cols,module_grid[i:i+4]):
            with col:
                st.markdown(f"""<div class="module-card"><div class="module-icon">{icon}</div>
                <div class="module-title">{title}</div></div>""",unsafe_allow_html=True)

st.markdown("""
<div class="footer">
<div class="footer-name">S MOHAMMED KAIF</div>
<div class="footer-role">Data Science • Machine Learning • Data Analytics</div>
<div class="footer-links">
<a href="https://www.linkedin.com/in/s-mohammed-kaif-2a500a341" target="_blank">🔗 LinkedIn</a>
<a href="https://github.com/Shaik-Mohammed-Kaif" target="_blank">💻 GitHub</a>
</div>
<div style="margin-top:15px;font-size:10px;opacity:.55">
SuperStore Business Intelligence Platform • 2026
</div>
</div>
""",unsafe_allow_html=True)
