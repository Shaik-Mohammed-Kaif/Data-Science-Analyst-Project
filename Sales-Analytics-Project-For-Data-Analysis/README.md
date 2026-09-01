# 🤖 SuperStore AI Intelligence Platform

> **End-to-end Data Science, Business Intelligence & Machine Learning project for SuperStore sales analytics and Return_Flag prediction.**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Data%20Science-013243?logo=numpy)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Visualization-3F4F75?logo=plotly)
![SQL](https://img.shields.io/badge/SQL-Analytics-4479A1?logo=mysql)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

---

## 📌 Project Overview

**SuperStore AI Intelligence Platform** is an end-to-end Data Science, Business Intelligence and Machine Learning project built using SuperStore transaction data.

The project takes the analysis from raw transaction data to an interactive AI-powered business intelligence application.

The complete workflow includes:

- 📂 Data Loading
- 🔎 Data Inspection
- 🧹 Data Cleaning
- ⚙️ Feature Engineering
- 📊 Exploratory Data Analysis
- 📈 Statistical Analysis
- 🗄️ SQL Business Analysis
- 📉 Interactive Plotly Visualizations
- 💼 Business Intelligence
- 🤖 Machine Learning
- 🏆 Model Comparison
- 📋 Model Evaluation
- 🎯 Return Prediction
- 💡 Explainability
- 📄 CSV / PDF Reporting
- 🌐 Streamlit Application
- 💼 Business Decision Support

### End-to-End Pipeline

```text
Raw Data
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
EDA
   ↓
Statistical Analysis
   ↓
SQL Analysis
   ↓
Business Intelligence
   ↓
Visualization
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Return Prediction
   ↓
Explainability
   ↓
Decision Support
   ↓
Streamlit Application
````

# 🚀 Live Applications

The SuperStore project includes three deployed Streamlit applications, each focused on a different part of the end-to-end analytics and machine-learning workflow.

### 📊 1. SuperStore Sales Dashboard

Interactive business intelligence dashboard for exploring:

- Sales performance
- Profit analysis
- Orders
- Quantity
- Categories
- Sub-Categories
- Payment modes
- Customer performance
- Monthly trends
- Year / Quarter analysis

🔗 **Live Application:** [SuperStore Sales Dashboard](https://superstore-sales-dashboards.streamlit.app/)

---

### 🤖 2. SuperStore Predictive Intelligence

Machine-learning application for:

- Return_Flag prediction
- Classification
- Model evaluation
- Prediction probabilities
- Model comparison
- F1-based model selection
- Predictive analytics
- Business decision support

🔗 **Live Application:** [SuperStore Predictive Intelligence](https://superstore-predictive-intelligence.streamlit.app/)

---

### 📈 3. SuperStore Sales Analytics

Additional interactive sales analytics application covering:

- Sales analysis
- Profit analysis
- Customer analysis
- Product analysis
- Category analysis
- Regional analysis
- Business performance insights
- Interactive visual analytics

🔗 **Live Application:** [SuperStore Sales Analytics](https://kiaf-supersales-dashbord-analytic.streamlit.app/)

---

## 🌐 Streamlit Applications Overview

| Application | Purpose | Technology |
|---|---|---|
| 📊 SuperStore Sales Dashboard | Interactive Business Intelligence | Streamlit + Plotly |
| 🤖 SuperStore Predictive Intelligence | Return_Flag Machine Learning Prediction | Streamlit + Scikit-learn |
| 📈 SuperStore Sales Analytics | Sales & Business Analytics | Streamlit + Plotly + Pandas |

### 🔗 Quick Access

**📊 Sales Dashboard**  
[Open SuperStore Sales Dashboard](https://superstore-sales-dashboards.streamlit.app/)

**🤖 Predictive Intelligence**  
[Open SuperStore Predictive Intelligence](https://superstore-predictive-intelligence.streamlit.app/)

**📈 Sales Analytics**  
[Open SuperStore Sales Analytics](https://kiaf-supersales-dashbord-analytic.streamlit.app/)

---

# 🎯 Business Objective

Historical transaction data can explain **what happened** in a business.

This project goes one step further by combining historical business analytics with machine-learning prediction.

The primary machine-learning objective is:

```text
Return_Flag Prediction
```

The platform therefore provides two major intelligence layers.

### 📊 Descriptive Business Intelligence

The application analyzes:

* Sales
* Profit
* Orders
* Customers
* Products
* Categories
* Regions
* Segments
* Ship Modes
* Quantity
* Profitability
* Time-based performance

### 🤖 Predictive Intelligence

The machine-learning layer provides:

* Return prediction
* Classification modelling
* Model comparison
* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Prediction probabilities
* Predictive feature interpretation

> **Important:** Model predictions and feature importance are interpreted as predictive signals and should not automatically be treated as causal conclusions.

---

# 🗂️ Dataset

The project uses SuperStore transaction data.

### Original Dataset

```text
SuperStore_Sales_Dataset.csv
```

### Feature-Engineered Dataset

```text
SuperStore_Feature_Engineered.csv
```

### SQLite Database

```text
superstore_analysis.db
```

The dataset workflow includes:

* Data inspection
* Data-type analysis
* Missing-value analysis
* Duplicate analysis
* Data cleaning
* Feature engineering
* Numerical analysis
* Categorical analysis
* Statistical analysis
* Business aggregations
* Machine-learning preparation

---

# 📊 Dataset Snapshot

The final machine-learning workflow reported:

| Metric            |         Value |
| ----------------- | ------------: |
| Dataset Rows      |         5,901 |
| Original Features |            53 |
| Target            | `Return_Flag` |
| Training Rows     |         4,720 |
| Testing Rows      |         1,181 |
| Target Classes    |             2 |

---

# 📚 Project Notebook Roadmap

The complete project has been developed step-by-step using Jupyter notebooks.

---

## 01 — Data Inspection, Cleaning, Feature Engineering & EDA

```text
01_Data_Inspection_Cleaning_Feature_Engineering_EDA.ipynb
```

This notebook covers:

* Dataset loading
* Dataset shape
* Column inspection
* Data types
* Missing values
* Duplicate records
* Data cleaning
* Feature engineering
* Numerical analysis
* Categorical analysis
* Univariate analysis
* Bivariate analysis
* Business-oriented EDA

---

## 02 — GroupBy Business & Statistical Analysis

```text
02_GroupBy_Business_Statistical_Analysis.ipynb
```

This notebook focuses on deeper business and statistical analysis.

Topics include:

* Pandas GroupBy
* Aggregation
* Sales analysis
* Profit analysis
* Category analysis
* Regional analysis
* Segment analysis
* Product analysis
* Statistical summaries
* Business performance comparison

---

## 03 — SQL for Data Analysis

```text
03-Sql-For-Data-Analysis.ipynb
```

Database:

```text
superstore_analysis.db
```

The SQL analysis demonstrates:

```sql
SELECT
WHERE
GROUP BY
ORDER BY
HAVING
CASE
Aggregate Functions
Subqueries
```

Business use cases include:

* Sales analysis
* Profit analysis
* Category performance
* Regional performance
* Segment performance
* Profitability analysis
* Business KPI queries

---

## 04 — Plotly Business Insights

```text
04_Plotly_Business_Insights.ipynb
```

This notebook focuses on interactive business visualization.

Visual analysis includes:

* Sales vs Profit
* Category performance
* Regional performance
* Segment contribution
* Product performance
* Profitability
* Time trends
* Business performance comparisons

---

## 05 — ML Return Prediction Modeling

```text
05_ML_Return_Prediction_Modeling.ipynb
```

The machine-learning notebook contains the complete prediction workflow.

```text
Data Preparation
      ↓
Target Definition
      ↓
Feature Preparation
      ↓
Train/Test Split
      ↓
Preprocessing
      ↓
Model Training
      ↓
Model Comparison
      ↓
Model Evaluation
      ↓
Model Selection
      ↓
Prediction
      ↓
Model Export
```

---

# 🤖 Machine Learning

## Prediction Target

```text
Return_Flag
```

The problem is treated as a:

```text
Binary Classification Problem
```

The workflow evaluates multiple classification algorithms and selects the strongest model using the defined evaluation strategy.

---

# 🏆 Model Evaluation

The following classification metrics are evaluated:

| Metric    | Purpose                                           |
| --------- | ------------------------------------------------- |
| Accuracy  | Overall classification correctness                |
| Precision | Correctness of positive predictions               |
| Recall    | Ability to identify positive cases                |
| F1 Score  | Balance between Precision and Recall              |
| ROC-AUC   | Classification discrimination/ranking performance |

---

# 🎯 Model Selection

The project uses:

```text
F1 Score
```

as the primary model-selection metric.

Accuracy, Precision, Recall and ROC-AUC are also reported to provide a broader evaluation of model performance.

---

# 🥇 Final Model Result

The final ML workflow reported:

```text
Selected Model : Logistic Regression

Accuracy       : 1.0000
Precision      : 1.0000
Recall         : 1.0000
F1 Score       : 1.0000
ROC-AUC        : 1.0000
```

### Training / Testing Split

```text
Training Rows : 4,720
Testing Rows  : 1,181
```

---

# 💾 Model Artifacts

The production prediction pipeline is stored as:

```text
saved_models/
└── superstore_return_prediction_pipeline.joblib
```

Model metadata:

```text
saved_models/
└── superstore_return_model_metadata.json
```

Prediction output:

```text
outputs/
└── SuperStore_Return_Predictions.csv
```

Model comparison:

```text
outputs/
└── SuperStore_Model_Comparison.csv
```

---

# ⚠️ Model Validation Note

The reported test metrics are perfect:

```text
Accuracy  = 1.0000
Precision = 1.0000
Recall    = 1.0000
F1        = 1.0000
ROC-AUC   = 1.0000
```

Such results should be investigated carefully before production use.

Potential reasons may include:

* Target leakage
* Target-derived features
* Duplicate or highly related observations
* Extremely separable classes
* Dataset construction characteristics

Therefore, additional validation using genuinely unseen future data is recommended before treating the model as production-ready.

---

# 💼 Business Intelligence Layer

The Streamlit application contains a complete business intelligence layer.

## Core KPIs

The application calculates:

```text
Total Sales
Total Profit
Total Orders
Customers
Quantity Sold
Profit Margin
Average Order Value
Profit Per Order
```

---

# 📊 Business Analysis

The application analyzes business performance across:

### Category

* Sales
* Profit
* Margin
* Top category
* Lowest-profit category

### Region

* Sales
* Profit
* Margin
* Top region
* Regional profitability

### Segment

* Sales
* Profit
* Sales contribution
* Segment mix

### Product

* Top products
* Sales performance
* Profit performance
* Low-profit products
* Product profitability risk

### Time

* Yearly sales
* Yearly profit
* Profit margin
* Sales trends
* Profit trends

---

# 💡 Business Insights

The Business Intelligence layer is designed to answer questions such as:

### Sales Intelligence

* Which category generates the highest sales?
* Which region generates the highest sales?
* Which segment contributes the most revenue?
* Which products are top performers?

### Profitability Intelligence

* Which category generates the highest profit?
* Which region is most profitable?
* Which categories are profitable?
* Which categories are generating losses?
* Which products have weak profitability?

### Customer Intelligence

* Which customer segment contributes the most sales?
* How many customers are present?
* What is the average order value?
* What is the profit per order?

### Time Intelligence

* How do sales change over time?
* How does profit change over time?
* How does profit margin change over time?

---

# 💼 Business Decision Support

The platform converts analytics into practical business recommendations.

Examples include:

```text
Growth Opportunity
        ↓
Margin Optimization
        ↓
Regional Expansion
        ↓
Product Profitability Review
        ↓
Risk Identification
        ↓
Predictive Decision Support
```

The goal is not only to display charts but also to connect analytical findings with possible management actions.

---

# 🔎 Data Explorer

The Streamlit application includes a dedicated:

```text
Data Explorer
```

Features include:

* Dataset overview
* Number of rows
* Number of columns
* Numeric columns
* Categorical columns
* Missing-value analysis
* Duplicate analysis
* Deep summary statistics
* Business GroupBy analysis
* Full data inspection
* CSV export
* PDF report generation

---

# 📄 Reporting

The Data Explorer supports report generation for analytical results.

Reports can contain:

* Dataset overview
* Data quality information
* Summary statistics
* Business GroupBy analysis
* Management summary
* Executive-level interpretation

The application also provides CSV export functionality.

---

# 🗄️ SQL Business Analytics

The project includes a SQLite database:

```text
superstore_analysis.db
```

SQL is used alongside Pandas to demonstrate practical data-analysis skills.

The SQL layer covers:

```text
Filtering
Aggregation
Grouping
Sorting
Conditional logic
Business KPIs
Category analysis
Regional analysis
Segment analysis
Profitability analysis
```

This provides a second analytical approach to the same business data.

---

# 📈 Visualization

The project uses both static and interactive visualization techniques.

### Visualization Libraries

```text
Matplotlib
Seaborn
Plotly
```

### Plotly Business Visualizations

Examples include:

* Bar charts
* Line charts
* Scatter plots
* Pie / Donut charts
* Profitability matrices
* Sales comparisons
* Profit comparisons
* Time-series analysis

---

# 🌐 Streamlit Application

Main application files include:

```text
main.py
sales.py
sale_predict.py
```

The Streamlit platform integrates:

```text
Business Intelligence
        +
Data Explorer
        +
Machine Learning
        +
Prediction
        +
Model Evaluation
        +
Explainability
        +
Decision Support
        +
Reporting
```

---

# 🧠 End-to-End ML Workflow

```text
01 📂 Data Loading
        ↓
02 🔎 Data Understanding
        ↓
03 📊 EDA
        ↓
04 ⚙️ Preprocessing
        ↓
05 🤖 Model Training
        ↓
06 🏆 Model Comparison
        ↓
07 🎯 Prediction
        ↓
08 📋 Evaluation
        ↓
09 💡 Explainability
        ↓
10 💼 Decision Support
```

---

# 🛠️ Technology Stack

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| Python           | Core programming          |
| Pandas           | Data manipulation         |
| NumPy            | Numerical computation     |
| Matplotlib       | Visualization             |
| Seaborn          | EDA visualization         |
| Plotly           | Interactive visualization |
| SQLite           | SQL analytics             |
| Scikit-learn     | Machine learning          |
| Joblib           | Model serialization       |
| Streamlit        | Interactive application   |
| ReportLab        | PDF reporting             |
| Jupyter Notebook | Step-by-step analysis     |

---

# 📁 Project Structure

```text
Sales-Analytics-Project-For-Data-Analysis/
│
├── .gitignore
├── README.md
├── requirement.txt
│
├── 01_Data_Inspection_Cleaning_Feature_Engineering_EDA.ipynb
├── 02_GroupBy_Business_Statistical_Analysis.ipynb
├── 03-Sql-For-Data-Analysis.ipynb
├── 04_Plotly_Business_Insights.ipynb
├── 05_ML_Return_Prediction_Modeling.ipynb
│
├── SuperStore_Sales_Dataset.csv
├── SuperStore_Feature_Engineered.csv
├── superstore_analysis.db
│
├── main.py
├── sales.py
├── sale_predict.py
│
├── saved_models/
│   ├── superstore_return_prediction_pipeline.joblib
│   └── superstore_return_model_metadata.json
│
├── input-Output-Csv-Files/
│
├── Sales-Ipynb-Files/
│
├── Streamlit-Prediction_Analysis-files/
│
└── blue bg.jpg
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Shaik-Mohammed-Kaif/Sales-Analytics-Project-For-Data-Analysis.git
```

```bash
cd Sales-Analytics-Project-For-Data-Analysis
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

Your project currently uses:

```text
requirement.txt
```

Install:

```bash
pip install -r requirement.txt
```

---

# ▶️ Run the Streamlit Application

From the project root:

```bash
streamlit run main.py
```

If the prediction application is the intended entry point:

```bash
streamlit run sale_predict.py
```

Streamlit will provide a local URL in the terminal.

---

# 🧪 Run the Machine Learning Workflow

Open:

```text
05_ML_Return_Prediction_Modeling.ipynb
```

Run the notebook from top to bottom.

The workflow creates the required model and output artifacts.

---

# 📦 Model Loading

The prediction application uses the saved machine-learning pipeline.

The model artifact is located under:

```text
saved_models/
```

The exact model used by the prediction application should always match the artifact referenced inside:

```text
sale_predict.py
```

This keeps the training and prediction workflows consistent.

---

# 🔐 GitHub & Large Files

Machine-learning model files can become large.

For example, a large `.joblib` bundle should not be committed blindly to a GitHub repository.

Before pushing the project:

1. Check which model file `sale_predict.py` loads.
2. Keep only required model artifacts.
3. Remove obsolete model bundles.
4. Avoid committing temporary files.
5. Avoid committing virtual environments.
6. Avoid committing Jupyter checkpoints.

### Recommended `.gitignore`

```gitignore
# ============================================================
# PYTHON
# ============================================================

__pycache__/
*.py[cod]
*.pyo

# ============================================================
# VIRTUAL ENVIRONMENT
# ============================================================

.venv/
venv/
env/

# ============================================================
# JUPYTER
# ============================================================

.ipynb_checkpoints/

# ============================================================
# IDE
# ============================================================

.vscode/
.idea/

# ============================================================
# OS
# ============================================================

.DS_Store
Thumbs.db

# ============================================================
# PYTEST / CACHE
# ============================================================

.pytest_cache/

# ============================================================
# STREAMLIT SECRETS
# ============================================================

.streamlit/secrets.toml

# ============================================================
# TEMPORARY FILES
# ============================================================

*.tmp
*.temp
```

### Important

Do **not** automatically add:

```gitignore
*.csv
```

because the project intentionally uses CSV datasets.

Also do **not** automatically add:

```gitignore
*.joblib
```

if the deployed application needs the model file.

First verify the exact model dependency in `sale_predict.py`.

---

# 🚀 Deployment

The Streamlit application can be deployed after confirming:

* `requirement.txt` contains all dependencies
* Correct Streamlit entry point
* Required CSV files are available
* Required model artifact is available
* File paths are deployment-safe
* No secrets are committed
* Large files are handled appropriately

---

# ⚖️ Responsible Analytics & AI

The platform separates:

### Historical Evidence

Business analytics describe patterns observed in historical transaction data.

### Predictive Evidence

Machine-learning models generate predictions based on learned relationships.

### Causal Claims

Feature importance and model predictions should **not automatically be interpreted as causal effects**.

This distinction is important when predictive analytics are used for business decision-making.

---

# 🧠 Data Science Principles

## 01 — Descriptive Before Predictive

Understand historical business performance before applying machine learning.

## 02 — Multiple Evaluation Metrics

Do not rely exclusively on accuracy.

## 03 — F1-Based Model Selection

F1 is used as the primary model-selection metric in this project.

## 04 — Explainability

Feature importance should be interpreted as predictive association.

## 05 — Business Context

Technical model results should be connected with business context before decisions are made.

---

# 📌 Key Project Highlights

## Data Science

* Data inspection
* Data cleaning
* Feature engineering
* EDA
* Statistical analysis

## Business Intelligence

* KPI development
* GroupBy analysis
* Category intelligence
* Regional intelligence
* Segment intelligence
* Product intelligence
* Profitability intelligence
* Time intelligence

## SQL

* SQLite database
* Business queries
* Aggregations
* Filtering
* Grouped analysis
* Business KPI analysis

## Machine Learning

* Binary classification
* Multiple model comparison
* F1-based model selection
* Prediction probabilities
* Classification evaluation
* Model serialization
* Return prediction

## Application

* Streamlit dashboard
* Interactive filters
* Business Insights
* Data Explorer
* ML prediction
* Model evaluation
* Explainability
* CSV export
* PDF reporting
* Decision support

---

# 🏁 Final Project Outcome

The project demonstrates a complete real-world Data Science workflow:

```text
                SUPERSTORE TRANSACTION DATA
                           │
                           ▼
                  DATA INSPECTION
                           │
                           ▼
                    DATA CLEANING
                           │
                           ▼
                 FEATURE ENGINEERING
                           │
                           ▼
                          EDA
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       BUSINESS ANALYSIS           SQL ANALYSIS
              │                         │
              └────────────┬────────────┘
                           ▼
                 BUSINESS INTELLIGENCE
                           │
                           ▼
                  PLOTLY VISUALIZATION
                           │
                           ▼
                 MACHINE LEARNING
                           │
                           ▼
                 MODEL COMPARISON
                           │
                           ▼
                  MODEL EVALUATION
                           │
                           ▼
                  RETURN PREDICTION
                           │
                           ▼
                    EXPLAINABILITY
                           │
                           ▼
                  DECISION SUPPORT
                           │
                           ▼
                 STREAMLIT PLATFORM
```

---

# 👨‍💻 Project Author

## S Mohammed Kaif

**Data Science • Machine Learning • AI • Data Analytics**

### 🔗 LinkedIn

[https://www.linkedin.com/in/s-mohammed-kaif-2a500a341](https://www.linkedin.com/in/s-mohammed-kaif-2a500a341)

### 💻 GitHub

[https://github.com/Shaik-Mohammed-Kaif](https://github.com/Shaik-Mohammed-Kaif)

---

# ⭐ Project Summary

**SuperStore AI Intelligence Platform**

> **Data → Intelligence → Prediction → Decision Support**

Built with:

```text
Python
Pandas
NumPy
SQL
Scikit-learn
Plotly
Streamlit
Jupyter
ReportLab
```

---

## 📜 License

This project is intended for educational, portfolio and demonstration purposes.

---

## 👨‍💻 Author

**S Mohammed Kaif**

**Data Science • Machine Learning • AI • Data Analytics**

© 2026 S Mohammed Kaif
