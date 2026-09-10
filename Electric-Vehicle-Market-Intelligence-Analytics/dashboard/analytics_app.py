"""
analytics_app.py
-----------------
Interactive Streamlit dashboard for the real, web-scraped
Electric Vehicle market dataset (source: Wikipedia).

Run with:
    streamlit run analytics_app.py
"""

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="EV Market Analytics (Real Data)", page_icon="🚗", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("../data/processed/ev_scraped_clean.csv")

df = load_data()

# ------------------------------------------------------------------
st.sidebar.header("🔎 Filters")
markets = st.sidebar.multiselect("Market", sorted(df["Market"].unique()), default=sorted(df["Market"].unique()))
regions = st.sidebar.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
year_range = st.sidebar.slider(
    "Launch Year",
    int(df["Year"].min()), int(df["Year"].max()),
    (int(df["Year"].min()), int(df["Year"].max()))
)

filtered = df[
    (df["Market"].isin(markets)) &
    (df["Region"].isin(regions)) &
    (df["Year"].between(year_range[0], year_range[1]))
]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered):,}** of {len(df):,} real EV models")
st.sidebar.markdown(
    "**Source:** [Wikipedia — List of battery electric vehicles]"
    "(https://en.wikipedia.org/wiki/List_of_battery_electric_vehicles)"
)

# ------------------------------------------------------------------
st.title("🚗 Global EV Market Analytics — Real Scraped Data")
st.caption("Data Analytics Phase | Web-scraped with requests + BeautifulSoup | S Mohammed Kaif")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total EV Models", f"{len(filtered):,}")
c2.metric("Manufacturers", f"{filtered['Manufacturer'].nunique():,}")
c3.metric("Origin Countries", f"{filtered['Origin'].nunique():,}")
c4.metric("Dedicated BEV %", f"{filtered['Is_Dedicated_BEV'].mean()*100:.0f}%" if filtered['Is_Dedicated_BEV'].notna().any() else "N/A")

st.markdown("---")

c5, c6 = st.columns((2, 1))
with c5:
    st.subheader("📈 EV Model Launches Over Time")
    yearly = filtered.groupby("Year").size().reset_index(name="Models Launched")
    fig = px.bar(yearly, x="Year", y="Models Launched", color_discrete_sequence=["#1f3864"])
    st.plotly_chart(fig, use_container_width=True)

with c6:
    st.subheader("🌍 Market Split")
    mc = filtered["Market"].value_counts().reset_index()
    mc.columns = ["Market", "Count"]
    fig = px.pie(mc, names="Market", values="Count", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

c7, c8 = st.columns(2)
with c7:
    st.subheader("🚙 Top Body Styles")
    bc = filtered["Primary_Body_Style"].value_counts().head(10).reset_index()
    bc.columns = ["Body Style", "Count"]
    fig = px.bar(bc, x="Count", y="Body Style", orientation="h", color="Count", color_continuous_scale="Teal")
    st.plotly_chart(fig, use_container_width=True)

with c8:
    st.subheader("🏭 Top Manufacturer Groups")
    mgc = filtered["Manufacturer_Group"].value_counts().head(10).reset_index()
    mgc.columns = ["Manufacturer Group", "Count"]
    fig = px.bar(mgc, x="Manufacturer Group", y="Count", color="Count", color_continuous_scale="Oranges")
    st.plotly_chart(fig, use_container_width=True)

c9, c10 = st.columns(2)
with c9:
    st.subheader("🗺️ Models by Region of Origin")
    rc = filtered["Region"].value_counts().reset_index()
    rc.columns = ["Region", "Count"]
    fig = px.bar(rc, x="Region", y="Count", color="Region")
    st.plotly_chart(fig, use_container_width=True)

with c10:
    st.subheader("⚡ Dedicated BEV Platform Trend")
    trend = filtered.dropna(subset=["Is_Dedicated_BEV"]).groupby("Year")["Is_Dedicated_BEV"].mean().mul(100).reset_index()
    fig = px.line(trend, x="Year", y="Is_Dedicated_BEV", markers=True)
    fig.update_yaxes(title="% Dedicated BEV Platform")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
with st.expander("📋 View Filtered Data"):
    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "Download Filtered Data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="ev_scraped_filtered.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Real, scraped Electric Vehicle market data | Data Analytics Project | S Mohammed Kaif")
