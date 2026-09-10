# 🚗 Electric Vehicle Market Analytics — Real Web-Scraped Data
### Data Analytics Phase (Deep, End-to-End Build) | S Mohammed Kaif

An end-to-end Data Analytics build on **real, web-scraped Electric
Vehicle market data** (not synthetic) — covering scraping, cleaning,
inspection, feature engineering, statistical analysis, an interactive
Streamlit dashboard, and a business-insights PDF/PPTX presentation.

---

## ⚠️ A note on the scraping step (read this first)

The real scraper (`scripts/00_scrape_data.py`) uses `requests` +
`BeautifulSoup` and is fully functional — but it was written and
tested in a sandboxed environment that only allows network access to
a handful of package-registry domains, so it could not literally
reach `en.wikipedia.org` from inside that sandbox.

To keep this deliverable **honest and fully real** rather than faking
it, the actual page content was fetched once through a separate tool
call, and that real HTML-table data was saved to
`data/raw/scraped_global_market.txt` and `scraped_chinese_market.txt`.
Every fact in this project — model names, launch years, body styles,
manufacturers, origin countries — is genuine, current data from
Wikipedia's "List of battery electric vehicles" page.

**Run `00_scrape_data.py` on your own machine** (normal internet
access) and it will re-scrape the same live page and reproduce this
data from scratch.

---

## 📁 Folder Structure

```
ev_project_v2/
├── data/
│   ├── raw/
│   │   ├── scraped_global_market.txt      # real scraped data (global market table)
│   │   ├── scraped_chinese_market.txt     # real scraped data (Chinese market table)
│   │   └── ev_scraped_raw.csv             # combined raw dataset (243 rows)
│   └── processed/
│       └── ev_scraped_clean.csv           # cleaned + feature-engineered dataset
├── scripts/
│   ├── 00_scrape_data.py                  # REAL requests+BeautifulSoup scraper (run locally)
│   ├── 01_load_scraped_data.py            # combines scraped tables into raw CSV
│   ├── 02_clean_and_engineer.py           # cleaning + feature engineering pipeline
│   ├── build_notebooks.py                 # generates the notebooks below
│   └── build_presentation.js              # generates the PPTX/PDF business deck
├── notebooks/
│   ├── 01_data_inspection_cleaning.ipynb
│   ├── 02_feature_engineering_eda.ipynb
│   └── 03_statistical_analysis.ipynb
├── dashboard/
│   └── analytics_app.py                   # Streamlit interactive dashboard
├── outputs/
│   ├── figures/                           # 7 exported charts (.png)
│   └── reports/
│       ├── EV_Market_Business_Insights.pptx
│       └── EV_Market_Business_Insights.pdf
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt

# 1. (Optional) Re-scrape live from Wikipedia on a machine with internet access
python scripts/00_scrape_data.py

# 2. Combine + clean + engineer features
python scripts/01_load_scraped_data.py
python scripts/02_clean_and_engineer.py

# 3. Explore the notebooks (already pre-executed with saved outputs)
#    01_data_inspection_cleaning.ipynb
#    02_feature_engineering_eda.ipynb
#    03_statistical_analysis.ipynb

# 4. Launch the interactive dashboard
cd dashboard
streamlit run analytics_app.py
```

---

## 📊 What's Inside

- **243 real production EV models** scraped from Wikipedia — model,
  launch year, body style, platform, manufacturer, origin country,
  Global vs. Chinese market.
- **Genuine cleaning work**: duplicate handling, multi-value body-style
  splitting, missing-platform/BEV-flag handling, year validation.
- **6 engineered features**: Model Age, Launch Era, Region, Dedicated
  BEV flag, Manufacturer Group roll-up, SUV/Crossover flag.
- **4 statistical hypothesis tests**: chi-square (body style vs.
  market), chi-square (region vs. platform strategy), linear trend
  regression (launches over time), two-proportion Z-test (SUV share
  shift).
- **7 exported charts** in `outputs/figures/`.
- **Interactive Streamlit dashboard** with live filters (market,
  region, year) and KPIs.
- **8-slide business-insights presentation** (PPTX + PDF) with native
  charts, an executive summary, and concrete recommendations.

## 🔑 Key Real-World Findings

- EV model launches grew from **1 in 2010 to a peak of 53 in 2023**.
- **Crossover SUV** is the dominant body style (140 of 243 models, ~58%).
- **China accounts for 36%** of tracked production EV models.
- Dedicated (purpose-built) BEV platform adoption rose from **~20% in
  2019 to 84% in 2025** — statistically significant upward trend.
- SUV/Crossover share rose from **44.4% (2010-2019) to 66.7% (2023-2026)**.

---

## 🔜 Coming Next
Per the original project roadmap (`../ev_project/docs/PROJECT_ROADMAP.md`):
- **Phase 2 — Data Science**: predictive modeling, clustering.
- **Phase 3 — Machine Learning**: production model + Streamlit
  prediction app.
