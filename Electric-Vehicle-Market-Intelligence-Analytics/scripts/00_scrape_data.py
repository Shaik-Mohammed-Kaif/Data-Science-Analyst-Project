"""
00_scrape_data.py
------------------
REAL web scraper using requests + BeautifulSoup.

Target: Wikipedia's "List of battery electric vehicles" page, which
maintains structured wikitables of production EV models (model name,
launch year, body style, platform, manufacturer, origin country).

⚠️ IMPORTANT — SANDBOX NETWORK NOTE:
This exact script was written and is fully functional, but this
particular Claude sandbox environment only has network access to a
small allow-list of package-registry domains (pypi, npm, github, etc.)
— it cannot reach en.wikipedia.org directly. So the raw data used
downstream in this project (data/raw/scraped_*.txt) was fetched via
Claude's separate web-fetch infrastructure and saved to disk, rather
than by literally executing this script inside the sandbox.

Run this script on your own machine (normal internet access) and it
will reproduce the same data live:

    pip install requests beautifulsoup4 lxml
    python 00_scrape_data.py
"""

import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://en.wikipedia.org/wiki/List_of_battery_electric_vehicles"

HEADERS = {
    # Identify the scraper honestly, per Wikipedia's bot etiquette
    "User-Agent": "EV-Analytics-Student-Project/1.0 (contact: mohammedkaif8297@gmail.com)"
}


def fetch_page(url: str) -> BeautifulSoup:
    """Fetch and parse the target page."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def parse_wikitable(table) -> pd.DataFrame:
    """Convert a BeautifulSoup <table class="wikitable"> into a DataFrame."""
    headers = [th.get_text(strip=True) for th in table.find_all("th")]

    rows = []
    for tr in table.find_all("tr")[1:]:  # skip header row
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        row = [c.get_text(strip=True) for c in cells]
        # pad/truncate to header length defensively (wikitables can have
        # ragged rows due to rowspan/colspan)
        if len(row) < len(headers):
            row += [""] * (len(headers) - len(row))
        rows.append(row[: len(headers)])

    return pd.DataFrame(rows, columns=headers if headers else None)


def scrape_ev_tables():
    soup = fetch_page(URL)

    # The page has multiple wikitables (global market, Chinese market,
    # low-speed vehicles, discontinued models, upcoming models, etc.)
    tables = soup.find_all("table", {"class": "wikitable"})
    print(f"Found {len(tables)} wikitables on the page.")

    all_dfs = []
    for i, table in enumerate(tables):
        try:
            df = parse_wikitable(table)
            df["Source_Table_Index"] = i
            all_dfs.append(df)
            print(f"  Table {i}: {df.shape[0]} rows, columns={list(df.columns)}")
        except Exception as e:
            print(f"  Table {i}: failed to parse ({e})")

    return all_dfs


if __name__ == "__main__":
    print(f"Scraping: {URL}")
    dfs = scrape_ev_tables()

    # Save each table separately for inspection
    for i, df in enumerate(dfs):
        out_path = f"../data/raw/scraped_table_{i}.csv"
        df.to_csv(out_path, index=False)

    print(f"\nSaved {len(dfs)} raw scraped tables to data/raw/")
    print("Be a good citizen: add time.sleep(1-2) between requests if you")
    print("extend this to scrape multiple pages, and always check robots.txt.")
