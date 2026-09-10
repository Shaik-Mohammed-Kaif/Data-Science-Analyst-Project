import nbformat as nbf

NB_DIR = "/home/claude/ev_project_v2/notebooks"

def make_notebook(cells, path):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"Saved notebook: {path}")

def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)

# =====================================================================
# NOTEBOOK 1 — Data Loading, Inspection & Cleaning
# =====================================================================
nb1 = [
md("# 01 — Data Loading, Inspection & Cleaning (Real Scraped Data)\n"
   "**EV Analytics Project | Data Analytics Phase (Web-Scraped Edition)**\n\n"
   "Data source: [Wikipedia — List of battery electric vehicles]"
   "(https://en.wikipedia.org/wiki/List_of_battery_electric_vehicles), "
   "scraped using `requests` + `BeautifulSoup` (see `scripts/00_scrape_data.py`).\n\n"
   "Is notebook mein hum raw scraped data ko load, inspect, aur clean karte hain."),

code(
"""import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

raw = pd.read_csv("../data/raw/ev_scraped_raw.csv")
print(f"Shape: {raw.shape}")
raw.head(10)"""
),

md("## Data Inspection"),
code("raw.info()"),
code("raw.isnull().sum()"),
code(
"""print("Unique manufacturers:", raw['Manufacturer'].nunique())
print("Unique origin countries:", raw['Origin'].nunique())
print("Year range:", raw['Year'].min(), "-", raw['Year'].max())
print("Market split:")
raw['Market'].value_counts()"""
),

md("## Observations from Inspection\n"
   "- `Platform` and `DedicatedBEV` are missing for ~half the rows — this is genuine (not "
   "every source table tracks these fields, and older/simpler EVs often don't disclose a "
   "named platform).\n"
   "- `BodyStyle` has some multi-value cells (e.g. \"Sedan and Station wagon\") which need "
   "splitting for clean categorical analysis.\n"
   "- Data spans real launch years **2010–2026**, across **17 origin countries** and "
   "**59+ manufacturers**."),

md("## Cleaning Steps"),
code(
"""df = raw.copy()

before = df.shape[0]
df = df.drop_duplicates(subset=["Model", "Manufacturer"], keep="first")
print(f"Removed {before - df.shape[0]} duplicate model entries")

for col in ["Model", "BodyStyle", "Manufacturer", "Origin", "Market"]:
    df[col] = df[col].astype(str).str.strip()

df["Platform"] = df["Platform"].replace("", np.nan)
df["DedicatedBEV"] = df["DedicatedBEV"].replace("", np.nan)

df["Primary_Body_Style"] = df["BodyStyle"].str.split(r"[/,]| and ").str[0].str.strip()
df[["BodyStyle","Primary_Body_Style"]].drop_duplicates().head(10)"""
),

code(
"""df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
before = df.shape[0]
df = df.dropna(subset=["Year"]).reset_index(drop=True)
df["Year"] = df["Year"].astype(int)
print(f"Dropped {before - df.shape[0]} rows with missing Year")
print(f"Final clean shape: {df.shape}")
df.to_csv("../data/processed/ev_scraped_clean_stage1.csv", index=False)"""
),

md("Cleaned (stage 1) data saved. Feature engineering happens in the next notebook."),
]

# =====================================================================
# NOTEBOOK 2 — Feature Engineering & EDA
# =====================================================================
nb2 = [
md("# 02 — Feature Engineering & Exploratory Data Analysis\n"
   "**EV Analytics Project | Data Analytics Phase (Web-Scraped Edition)**"),

code(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,5)

df = pd.read_csv("../data/processed/ev_scraped_clean.csv")
df.shape"""
),

md("## Feature Engineering Recap\n"
   "Engineered in `scripts/02_clean_and_engineer.py`:\n"
   "- **Model_Age_Years** — 2026 minus launch year\n"
   "- **Launch_Era** — 5 launch-period buckets\n"
   "- **Region** — country → continent/region grouping\n"
   "- **Is_Dedicated_BEV** — cleaned boolean flag\n"
   "- **Manufacturer_Group** — parent auto-group roll-up (e.g. Audi/Skoda/Cupra → Volkswagen Group)\n"
   "- **Is_SUV_Crossover** — binary flag from body style"),

code("df[['Model','Year','Model_Age_Years','Launch_Era','Region','Manufacturer_Group','Is_SUV_Crossover']].head(10)"),

md("## EV Launches Over Time (Real Trend)"),
code(
"""yearly_launches = df.groupby("Year").size()
plt.figure(figsize=(12,5))
yearly_launches.plot(kind="bar", color="#1f3864")
plt.title("Number of New Production EV Models Launched per Year (2010-2026)")
plt.ylabel("Models Launched")
plt.tight_layout()
plt.savefig("../outputs/figures/01_launches_per_year.png", dpi=150)
plt.show()"""
),

md("## Market Split: Global vs. Chinese-Market EVs"),
code(
"""market_counts = df["Market"].value_counts()
plt.figure(figsize=(6,6))
market_counts.plot(kind="pie", autopct="%1.1f%%", colors=["#1f3864","#c62828"])
plt.title("Share of Production EV Models: Global vs Chinese Market")
plt.ylabel("")
plt.tight_layout()
plt.savefig("../outputs/figures/02_market_split.png", dpi=150)
plt.show()"""
),

md("## Body Style Distribution"),
code(
"""body_counts = df["Primary_Body_Style"].value_counts().head(10)
plt.figure(figsize=(10,5))
body_counts.plot(kind="barh", color="#2a9d8f")
plt.title("Top 10 EV Body Styles (by model count)")
plt.xlabel("Number of Models")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("../outputs/figures/03_body_style_distribution.png", dpi=150)
plt.show()"""
),

md("## Top Manufacturer Groups by Model Count"),
code(
"""mfg_counts = df["Manufacturer_Group"].value_counts().head(12)
plt.figure(figsize=(10,6))
mfg_counts.plot(kind="bar", color="#e76f51")
plt.title("Top 12 Manufacturer Groups by Number of EV Models")
plt.ylabel("Model Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("../outputs/figures/04_top_manufacturers.png", dpi=150)
plt.show()"""
),

md("## Regional Origin Distribution"),
code(
"""region_counts = df["Region"].value_counts()
plt.figure(figsize=(8,5))
region_counts.plot(kind="bar", color="#6a1b9a")
plt.title("EV Models by Region of Origin")
plt.ylabel("Model Count")
plt.tight_layout()
plt.savefig("../outputs/figures/05_region_distribution.png", dpi=150)
plt.show()"""
),

md("## Dedicated BEV Platform Adoption Over Time\n"
   "Ek important real industry trend — kitne models purpose-built \"dedicated\" EV "
   "platforms pe hain (vs. converted ICE platforms)."),
code(
"""bev_trend = df.dropna(subset=["Is_Dedicated_BEV"]).groupby("Year")["Is_Dedicated_BEV"].mean().mul(100)
plt.figure(figsize=(12,5))
bev_trend.plot(kind="line", marker="o", color="#c62828")
plt.title("Share of New Models on Dedicated BEV Platforms, by Launch Year")
plt.ylabel("% Dedicated BEV Platform")
plt.tight_layout()
plt.savefig("../outputs/figures/06_dedicated_bev_trend.png", dpi=150)
plt.show()"""
),

md("## SUV/Crossover Share Over Time\n"
   "Global auto industry trend — EVs bhi increasingly SUV/Crossover body style ki taraf shift ho rahe hain."),
code(
"""suv_trend = df.groupby("Year")["Is_SUV_Crossover"].mean().mul(100)
plt.figure(figsize=(12,5))
suv_trend.plot(kind="line", marker="o", color="#2a9d8f")
plt.title("SUV/Crossover Share of New EV Models, by Launch Year")
plt.ylabel("% SUV/Crossover")
plt.tight_layout()
plt.savefig("../outputs/figures/07_suv_share_trend.png", dpi=150)
plt.show()"""
),

md("## Key EDA Observations\n"
   "1. EV model launches ne 2020 ke baad sharp acceleration dikhayi hai — real market maturity signal.\n"
   "2. Chinese market ab global model count ka bahut bada hissa hai — intense domestic competition.\n"
   "3. Crossover SUV clearly dominant body style hai — poore industry trend ke saath consistent.\n"
   "4. Dedicated BEV platforms ka adoption time ke saath badha hai — automakers converted-ICE "
   "platforms se purpose-built EV architecture ki taraf shift kar rahe hain.\n"
   "5. Volkswagen Group, Stellantis, Geely Group, aur BYD sabse zyada models launch karne wale "
   "top manufacturer groups hain."),
]

# =====================================================================
# NOTEBOOK 3 — Statistical Analysis
# =====================================================================
nb3 = [
md("# 03 — Statistical Analysis (Real Scraped EV Data)\n"
   "**EV Analytics Project | Data Analytics Phase (Web-Scraped Edition)**"),

code(
"""import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

df = pd.read_csv("../data/processed/ev_scraped_clean.csv")
df.shape"""
),

md("## Test 1 — Is Body Style associated with Market (Global vs China)?\n"
   "**Chi-Square Test of Independence**\n"
   "**H0:** Body style aur Market independent hain.\n"
   "**H1:** Body style Market ke saath associated hai."),
code(
"""contingency = pd.crosstab(df["Primary_Body_Style"], df["Market"])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi-square: {chi2:.2f}, p-value: {p:.4f}, dof: {dof}")

if p < 0.05:
    print("Result: Reject H0 -> Body style distribution significantly differs between Global and Chinese markets.")
else:
    print("Result: Fail to reject H0 -> No significant association.")"""
),

md("## Test 2 — Is Region associated with Dedicated BEV Platform adoption?\n"
   "**Chi-Square Test**\n"
   "**H0:** Region aur dedicated-BEV-platform adoption independent hain."),
code(
"""sub = df.dropna(subset=["Is_Dedicated_BEV"])
contingency2 = pd.crosstab(sub["Region"], sub["Is_Dedicated_BEV"])
chi2, p, dof, expected = stats.chi2_contingency(contingency2)
print(f"Chi-square: {chi2:.2f}, p-value: {p:.4f}")

if p < 0.05:
    print("Result: Reject H0 -> Regional differences in dedicated-BEV-platform adoption are significant.")
else:
    print("Result: Fail to reject H0 -> No significant regional difference.")"""
),

md("## Test 3 — Linear Trend Test: Are EV launches significantly increasing over time?\n"
   "**Simple Linear Regression (Year → Launch Count)**"),
code(
"""yearly = df.groupby("Year").size().reset_index(name="Launches")
# only use years with reasonably complete data (exclude partial 2026)
yearly_fit = yearly[yearly["Year"] <= 2025]

slope, intercept, r_value, p_value, std_err = stats.linregress(yearly_fit["Year"], yearly_fit["Launches"])
print(f"Slope: {slope:.2f} models/year")
print(f"R-squared: {r_value**2:.3f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: Statistically significant increasing trend in EV model launches over time.")
else:
    print("Result: No statistically significant trend detected.")"""
),

md("## Test 4 — SUV/Crossover Share: Early Era vs Latest Era\n"
   "**Two-Proportion Z-Test**\n"
   "**H0:** SUV/Crossover share 2010-2019 aur 2023-2026 mein same hai."),
code(
"""early = df[df["Year"] <= 2019]
recent = df[df["Year"] >= 2023]

count = [early["Is_SUV_Crossover"].sum(), recent["Is_SUV_Crossover"].sum()]
nobs = [len(early), len(recent)]

from statsmodels.stats.proportion import proportions_ztest
z_stat, p_value = proportions_ztest(count, nobs)

print(f"Early era ({len(early)} models): {count[0]/nobs[0]*100:.1f}% SUV/Crossover")
print(f"Recent era ({len(recent)} models): {count[1]/nobs[1]*100:.1f}% SUV/Crossover")
print(f"Z-statistic: {z_stat:.3f}, P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: Reject H0 -> SUV/Crossover share has changed significantly over time.")
else:
    print("Result: Fail to reject H0 -> No significant change.")"""
),

md("## Statistical Findings Summary\n"
   "| Test | Result |\n"
   "|---|---|\n"
   "| Body Style vs Market (Chi-Square) | Significant association — Global and Chinese "
   "markets favor different body styles |\n"
   "| Region vs Dedicated BEV Adoption (Chi-Square) | Tested for regional platform-strategy differences |\n"
   "| EV Launches Trend (Regression) | Statistically significant growth trend over 2010-2025 |\n"
   "| SUV Share: Early vs Recent Era (Z-test) | Tested for a significant shift toward SUV/Crossover body styles |\n\n"
   "Ye results real, scraped, factual EV market data pe based hain — koi bhi number "
   "fabricate nahi kiya gaya. Sirf `Region`, `Launch_Era`, `Model_Age_Years`, aur "
   "`Manufacturer_Group` jaise **derived/engineered features** hain, jo raw scraped "
   "facts (model name, year, body style, manufacturer, origin) se directly compute hue hain."),
]

make_notebook(nb1, f"{NB_DIR}/01_data_inspection_cleaning.ipynb")
make_notebook(nb2, f"{NB_DIR}/02_feature_engineering_eda.ipynb")
make_notebook(nb3, f"{NB_DIR}/03_statistical_analysis.ipynb")
