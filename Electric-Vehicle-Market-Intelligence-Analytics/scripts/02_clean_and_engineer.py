"""
02_clean_and_engineer.py
--------------------------
Cleans, inspects, and feature-engineers the real scraped EV dataset.

Input : data/raw/ev_scraped_raw.csv
Output: data/processed/ev_scraped_clean.csv
"""

import pandas as pd
import numpy as np

RAW_PATH = "/home/claude/ev_project_v2/data/raw/ev_scraped_raw.csv"
CLEAN_PATH = "/home/claude/ev_project_v2/data/processed/ev_scraped_clean.csv"

CURRENT_YEAR = 2026

# Country -> broader region mapping (real-world geography, used for
# grouping in KPIs and stats)
REGION_MAP = {
    "United States": "North America",
    "Germany": "Europe", "France": "Europe", "Italy": "Europe",
    "United Kingdom": "Europe", "Sweden": "Europe", "Spain": "Europe",
    "Czech Republic": "Europe", "Romania": "Europe", "Croatia": "Europe",
    "Germany/United Kingdom": "Europe",
    "Japan": "Asia-Pacific", "South Korea": "Asia-Pacific", "China": "Asia-Pacific",
    "India": "Asia-Pacific", "Vietnam": "Asia-Pacific", "Indonesia": "Asia-Pacific",
    "Taiwan": "Asia-Pacific",
}


def load_data():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded raw scraped data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect(df: pd.DataFrame):
    print("\n--- Data Inspection ---")
    print(df.info())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().duplicated().sum())
    print("\nUnique Origins:", df["Origin"].nunique())
    print("Unique Manufacturers:", df["Manufacturer"].nunique())
    print("Year range:", df["Year"].min(), "-", df["Year"].max())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]

    # Drop exact duplicate models (can happen if a model appears in
    # more than one wikitable, e.g. re-listed under a sub-brand)
    df = df.drop_duplicates(subset=["Model", "Manufacturer"], keep="first")
    print(f"\nRemoved {before - df.shape[0]} duplicate model entries")

    # Standardize text fields
    for col in ["Model", "BodyStyle", "Manufacturer", "Origin", "Market"]:
        df[col] = df[col].astype(str).str.strip()

    # Empty-string cleanup -> NaN for genuinely missing platform/BEV flag
    df["Platform"] = df["Platform"].replace("", np.nan)
    df["DedicatedBEV"] = df["DedicatedBEV"].replace("", np.nan)

    # Body style has some multi-value entries (e.g. "Sedan and Station
    # wagon") - keep the first listed style as the primary category
    df["Primary_Body_Style"] = df["BodyStyle"].str.split(r"[/,]| and ").str[0].str.strip()

    # Year should be numeric; a handful of rows in the full Wikipedia
    # table have blank years (unannounced timing) - drop those here
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    before_year = df.shape[0]
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)
    print(f"Dropped {before_year - df.shape[0]} rows with missing/invalid Year")

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Model age (years since launch) - a genuine, useful analytics feature
    df["Model_Age_Years"] = CURRENT_YEAR - df["Year"]

    # Launch decade / era bucket
    df["Launch_Era"] = pd.cut(
        df["Year"],
        bins=[2009, 2015, 2019, 2022, 2024, 2027],
        labels=["2010-2015 (Early)", "2016-2019 (Growth)", "2020-2022 (Mainstream)",
                "2023-2024 (Acceleration)", "2025-2026 (Latest)"]
    )

    # Region grouping from origin country
    df["Region"] = df["Origin"].map(REGION_MAP).fillna("Other")

    # Dedicated BEV platform flag -> clean boolean (only available for
    # the Global-market table; Chinese-market source didn't track this)
    df["Is_Dedicated_BEV"] = df["DedicatedBEV"].map({"Yes": 1, "No": 0})

    # Manufacturer parent-group simplification (real consolidation —
    # many marques roll up to a handful of global auto groups)
    def parent_group(m):
        m = str(m)
        if "Stellantis" in m:
            return "Stellantis"
        if "General Motors" in m or "SAIC-GM" in m:
            return "General Motors"
        if "Volkswagen" in m or "Audi" in m or "SEAT" in m or "Skoda" in m or "Cupra" in m:
            return "Volkswagen Group"
        if "BMW" in m or "Mini" in m:
            return "BMW Group"
        if "Mercedes-Benz" in m:
            return "Mercedes-Benz Group"
        if "Geely" in m or "Volvo" in m or "Smart" in m or "Zeekr" in m or "Polestar" in m:
            return "Geely Group"
        if "BYD" in m or "Denza" in m:
            return "BYD Auto"
        if "Toyota" in m or "Lexus" in m or "Subaru" in m:
            return "Toyota Group"
        return m

    df["Manufacturer_Group"] = df["Manufacturer"].apply(parent_group)

    # Crossover/SUV share is a widely-used real industry KPI category
    df["Is_SUV_Crossover"] = df["Primary_Body_Style"].str.contains("SUV", case=False, na=False).astype(int)

    return df


def validate(df: pd.DataFrame):
    assert df["Vehicle_ID"].is_unique, "Vehicle_ID not unique!"
    assert df["Year"].between(2008, 2027).all(), "Year out of plausible range!"
    print("\n✅ Validation passed.")


def main():
    df = load_data()
    inspect(df)
    df = clean(df)
    df = engineer_features(df)
    validate(df)

    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nCleaned + feature-engineered dataset saved to: {CLEAN_PATH}")
    print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")


if __name__ == "__main__":
    main()
