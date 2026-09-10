"""
01_load_scraped_data.py
-------------------------
Loads the real scraped EV data (fetched from Wikipedia's "List of
battery electric vehicles" page) and combines the Global-market and
Chinese-market tables into a single raw dataset with a `Market` flag.

Input : data/raw/scraped_global_market.txt
        data/raw/scraped_chinese_market.txt
Output: data/raw/ev_scraped_raw.csv
"""

import pandas as pd

GLOBAL_PATH = "/home/claude/ev_project_v2/data/raw/scraped_global_market.txt"
CHINA_PATH = "/home/claude/ev_project_v2/data/raw/scraped_chinese_market.txt"
OUT_PATH = "/home/claude/ev_project_v2/data/raw/ev_scraped_raw.csv"


def load_global():
    df = pd.read_csv(GLOBAL_PATH, sep="|")
    df["Market"] = "Global"
    return df


def load_china():
    df = pd.read_csv(CHINA_PATH, sep="|")
    df["Market"] = "China"
    df["Platform"] = ""  # not available in the Chinese-market source table
    df["DedicatedBEV"] = ""  # not tracked separately in that table
    return df


def main():
    df_global = load_global()
    df_china = load_china()

    # Align columns
    cols = ["Model", "Year", "BodyStyle", "Platform", "DedicatedBEV", "Manufacturer", "Origin", "Market"]
    df_global = df_global[cols]
    df_china = df_china[cols]

    combined = pd.concat([df_global, df_china], ignore_index=True)
    combined.insert(0, "Vehicle_ID", [f"WIKI{1000+i}" for i in range(len(combined))])

    combined.to_csv(OUT_PATH, index=False)
    print(f"Combined raw scraped dataset: {combined.shape[0]} rows, {combined.shape[1]} columns")
    print(f"Saved to: {OUT_PATH}")
    print("\nSource: https://en.wikipedia.org/wiki/List_of_battery_electric_vehicles")
    print("(real production EV models, fetched Aug 2026)")
    print(combined.head())


if __name__ == "__main__":
    main()
