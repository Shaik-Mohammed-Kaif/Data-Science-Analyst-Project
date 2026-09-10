# 🚗 Global EV Market Analytics (Real Scraped Data) — Roadmap

**Candidate:** S Mohammed Kaif

This is the **real-data, deep-build edition** of the EV Analytics
project — same 3-phase structure as the original synthetic-data
version, but built end-to-end on genuine web-scraped data instead of
generated data.

## Phase 1 — Data Analytics (THIS BUILD, COMPLETE)

| Step | What was done | Output |
|---|---|---|
| Scrape | Real `requests` + `BeautifulSoup` scraper targeting Wikipedia's EV listings | `scripts/00_scrape_data.py`, `data/raw/scraped_*.txt` |
| Load & Combine | Merge Global + Chinese market tables | `scripts/01_load_scraped_data.py`, `data/raw/ev_scraped_raw.csv` |
| Clean & Inspect | Dedup, text standardization, missing-value audit | `notebooks/01_data_inspection_cleaning.ipynb` |
| Feature Engineer | Model age, launch era, region, manufacturer groups, SUV flag | `scripts/02_clean_and_engineer.py`, `data/processed/ev_scraped_clean.csv` |
| EDA | 7 charts covering trends, splits, distributions | `notebooks/02_feature_engineering_eda.ipynb`, `outputs/figures/` |
| Statistical Analysis | Chi-square, regression, z-test hypothesis tests | `notebooks/03_statistical_analysis.ipynb` |
| Interactive Dashboard | Streamlit app with live filters | `dashboard/analytics_app.py` |
| Business Presentation | 9-slide PDF/PPTX with findings + recommendations | `presentation/EV_Market_Business_Insights.pdf` |

## Phase 2 — Data Science (Next)
Feature-rich modeling on this same real dataset: predicting a
model's likely body style or platform strategy from its other
attributes, clustering manufacturers by strategy, and deeper
statistical explainability.

## Phase 3 — Machine Learning + Deployment (Final)
A production-style classifier (e.g. predicting "will this be a
dedicated BEV platform?" from launch year/manufacturer/region)
packaged behind a live Streamlit prediction app.

## ✅ Current Status
- [x] Phase 1 — Data Analytics: **COMPLETE** (real scraped data)
- [ ] Phase 2 — Data Science: pending
- [ ] Phase 3 — Machine Learning: pending
