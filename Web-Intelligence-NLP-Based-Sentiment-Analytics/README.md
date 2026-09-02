# 🧠 Web Intelligence & NLP-Based Sentiment Analytics

<p align="center">

  <strong>From Web Data → NLP → Sentiment Intelligence → Interactive Analytics</strong>

</p>

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">

  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">

  <img src="https://img.shields.io/badge/NLP-Natural%20Language%20Processing-8A2BE2?style=for-the-badge" alt="NLP">

  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">

  <img src="https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">

</p>

---

## 📌 Project Overview

**Web Intelligence & NLP-Based Sentiment Analytics** is an end-to-end
Natural Language Processing and Data Analytics project that transforms
unstructured web text into structured analytical insights.

The project demonstrates a complete workflow covering:

- 🌐 Web Data Collection
- 🕷️ Web Scraping
- 📦 Raw Data Storage
- 🧹 Data Cleaning
- 🔎 Exploratory Data Analysis
- 📝 NLP Text Preprocessing
- 🤖 Sentiment Analysis
- 🔑 Keyword Extraction
- 📚 N-Gram Analysis
- 🧩 Topic Modeling
- 📊 Statistical & Visual Analysis
- 🚀 Interactive Streamlit Dashboard

The project uses **Quotes to Scrape** as the web source and applies
VADER-based sentiment scoring to create weak sentiment labels that are
then used for machine-learning experiments.

---

## 🚀 Live Applications

### 📊 Text Intelligence Dashboard

Explore the interactive NLP analytics dashboard with global slicers, sentiment analytics, keyword intelligence, topic analysis, and interactive visualizations.

🔗 **[Open Text Intelligence Dashboard · Streamlit](https://kaif-nlp-intelligence.streamlit.app/)**

### 🤖 NLP Sentiment Predictor

Enter text and get sentiment predictions using the trained NLP sentiment classification model.

🔗 **[Open NLP Sentiment Predictor · Streamlit](https://nlp-sentiment-predictor.streamlit.app/)**

---

### 🧠 Application Overview

| Application | Purpose | Platform |
|---|---|---|
| 📊 **Text Intelligence Dashboard** | Interactive NLP analytics, sentiment, keywords & topics | Streamlit |
| 🤖 **NLP Sentiment Predictor** | Real-time text sentiment prediction | Streamlit |

---

# 🎯 Project Objective

The primary objective is to build a complete text analytics pipeline
that can answer questions such as:

- What is the overall sentiment distribution?
- Which authors have more positive or negative content?
- Which keywords appear most frequently?
- Which words are important according to TF-IDF?
- What common bigrams and trigrams appear?
- What latent topics exist within the text?
- How does sentiment vary across topics?
- Which topics require more analytical attention?
- How does text length relate to sentiment score?
- How well do different ML models reproduce the weak sentiment labels?

---

# 🔄 End-to-End Pipeline

```text
                    🌐 WEB SOURCE
                         │
                         ▼
                  🕷️ WEB SCRAPING
                         │
                         ▼
                    📦 RAW DATA
                         │
                         ▼
                 🧹 DATA CLEANING
                         │
                         ▼
                  📊 EDA & ANALYSIS
                         │
                         ▼
                📝 NLP PREPROCESSING
                         │
                         ▼
              🤖 SENTIMENT ANALYSIS
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       🔑 KEYWORDS              🧩 TOPICS
              │                     │
              └──────────┬──────────┘
                         ▼
                  📊 FINAL INSIGHTS
                         │
                         ▼
                🚀 STREAMLIT DASHBOARD
````

---

# 🌐 Data Source

The project uses:

**Quotes to Scrape**

A public practice website designed for web-scraping exercises.

The dataset contains information such as:

* Quote text
* Author
* Tags
* Source URL
* Pagination information

> ⚠️ The source does not provide meaningful ratings, dates,
> or human-annotated sentiment labels.

Therefore, sentiment labels are generated using **VADER** and should
be interpreted as weak / heuristic labels rather than ground-truth
human annotations.

---

# 🧰 Technology Stack

| Category            | Technologies               |
| ------------------- | -------------------------- |
| Programming         | Python                     |
| Data Analysis       | Pandas, NumPy              |
| Web Scraping        | Requests, BeautifulSoup    |
| Data Processing     | Pandas, Regex              |
| NLP                 | NLTK                       |
| Sentiment           | VADER                      |
| Machine Learning    | Scikit-Learn               |
| Feature Engineering | TF-IDF                     |
| Topic Modeling      | LDA, NMF                   |
| Keyword Analysis    | TF-IDF, Frequency Analysis |
| Visualization       | Matplotlib, Plotly         |
| Dashboard           | Streamlit                  |
| Model Persistence   | Joblib                     |
| Configuration       | YAML                       |
| Development         | Jupyter Notebook           |

---

# 📂 Project Structure

```text
Web-Intelligence-NLP/
│
├── 📁 data/
│   ├── 📁 raw/
│   │   └── scraped_quotes.csv
│   │
│   ├── 📁 processed/
│   │   └── cleaned_quotes.csv
│   │
│   └── 📁 final/
│       ├── nlp_ready_quotes.csv
│       ├── sentiment_analysis_results.csv
│       ├── keyword_analysis.csv
│       ├── topic_analysis.csv
│       ├── topic_summary.csv
│       └── nlp_final_dataset.csv
│
├── 📁 notebooks/
│   ├── 01_Web_Scraping.ipynb
│   ├── 02_Data_Cleaning_and_EDA.ipynb
│   ├── 03_NLP_Preprocessing.ipynb
│   ├── 04_Sentiment_Analysis.ipynb
│   ├── 05_Keyword_and_Topic_Analysis.ipynb
│   └── 06_Final_Insights.ipynb
│
├── 📁 src/
│   ├── __init__.py
│   ├── scraper.py
│   ├── preprocessing.py
│   ├── sentiment.py
│   └── topic_modeling.py
│
├── 📁 models/
│   ├── best_sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── 📁 dashboard/
│   ├── dash.py
│   └── prediction.py
│
├── 📁 outputs/
│   ├── 📁 figures/
│   ├── 📁 tables/
│   └── 📁 reports/
│
├── 📁 config/
│   └── config.yaml
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

---

# 📓 Notebook Workflow

## 01 — Web Scraping

**File:**

```text
notebooks/01_Web_Scraping.ipynb
```

Responsibilities:

* Connect to the public web source
* Request webpage content
* Parse HTML
* Extract quotes
* Extract authors
* Extract tags
* Handle pagination
* Store raw records
* Perform basic scraping validation

Output:

```text
data/raw/scraped_quotes.csv
```

---

## 02 — Data Cleaning & EDA

**File:**

```text
notebooks/02_Data_Cleaning_and_EDA.ipynb
```

Responsibilities:

* Inspect dataset structure
* Identify missing values
* Remove duplicates
* Clean text
* Normalize whitespace
* Inspect author distribution
* Analyze tags
* Explore text statistics
* Generate exploratory visualizations

Output:

```text
data/processed/cleaned_quotes.csv
```

---

## 03 — NLP Preprocessing

**File:**

```text
notebooks/03_NLP_Preprocessing.ipynb
```

Responsibilities:

* Convert text to lowercase
* Remove unwanted characters
* Normalize text
* Prepare text for vectorization
* Generate NLP-ready data
* Create text-level analytical features

Output:

```text
data/final/nlp_ready_quotes.csv
```

---

# 🤖 04 — Sentiment Analysis

**File:**

```text
notebooks/04_Sentiment_Analysis.ipynb
```

The project uses **VADER** for heuristic sentiment scoring.

### Sentiment Rules

```text
Compound Score >=  0.05  → Positive
Compound Score <= -0.05  → Negative
Otherwise                → Neutral
```

The generated labels are then used for machine-learning experiments.

### Models Evaluated

```text
1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Classifier
```

### Evaluation

Models are compared using:

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix
* Error Analysis

The best-performing model and TF-IDF vectorizer are persisted for
downstream prediction usage.

Outputs:

```text
data/final/sentiment_analysis_results.csv

models/best_sentiment_model.pkl
models/tfidf_vectorizer.pkl
```

> ⚠️ LinearSVC decision scores are model decision scores and should
> not be interpreted as calibrated probabilities.

---

# 🔑 05 — Keyword & Topic Analysis

**File:**

```text
notebooks/05_Keyword_and_Topic_Analysis.ipynb
```

This stage extracts semantic patterns from the text.

### Keyword Analysis

Includes:

* Word frequency
* TF-IDF keywords
* Document-level keywords
* Bigrams
* Trigrams

### Topic Modeling

Two approaches are explored:

```text
LDA — Latent Dirichlet Allocation

NMF — Non-negative Matrix Factorization
```

The analysis identifies:

* Dominant topics
* Topic scores
* Representative documents
* Topic distributions
* Topic × sentiment relationships

Outputs:

```text
data/final/keyword_analysis.csv
data/final/topic_analysis.csv
data/final/topic_summary.csv

outputs/tables/top_bigrams.csv
outputs/tables/top_trigrams.csv
outputs/tables/lda_topics.csv
outputs/tables/nmf_topics.csv
```

---

# 📊 06 — Final Insights

**File:**

```text
notebooks/06_Final_Insights.ipynb
```

This notebook integrates the outputs from previous stages into one
analytical dataset.

It combines:

```text
Web Data
   +
Cleaned Text
   +
NLP Features
   +
VADER Sentiment
   +
ML Predictions
   +
Keywords
   +
LDA Topics
   +
NMF Topics
   ↓
Final Intelligence Dataset
```

Main output:

```text
data/final/nlp_final_dataset.csv
```

Additional outputs include:

```text
outputs/tables/final_sentiment_summary.csv
outputs/tables/final_topic_summary.csv
outputs/tables/topic_sentiment_percentage.csv
outputs/tables/topic_priority.csv
outputs/tables/final_top_keywords.csv

outputs/reports/final_insight_report.txt
```

---

# 🚀 Interactive Dashboard

The project includes a Streamlit dashboard designed around an
**executive analytics / Power BI-style layout**.

The dashboard provides:

### 🎛️ Interactive Slicers

Users can filter the analytical dataset using actual dataset fields such
as:

* Sentiment
* Author
* Tags
* LDA Topic
* NMF Topic

A **Reset Filters** control restores the dashboard to the default
`All` state.

---

# 📈 Dashboard Visualizations

The dashboard contains **9 analytical visuals** organized into
three rows.

### Row 1

```text
┌────────────────────┬────────────────────┬────────────────────┐
│ Sentiment          │ Sentiment Flow     │ Top Keywords       │
│ Distribution       │ by Record          │                    │
└────────────────────┴────────────────────┴────────────────────┘
```

### Row 2

```text
┌────────────────────┬────────────────────┬────────────────────┐
│ Sentiment by       │ Sentiment Score    │ Top LDA Topics     │
│ Top Authors        │ Distribution       │                    │
└────────────────────┴────────────────────┴────────────────────┘
```

### Row 3

```text
┌────────────────────┬────────────────────┬────────────────────┐
│ Average Sentiment  │ Text Length vs     │ Sentiment Share    │
│ Score by Topic     │ Sentiment Score    │ by Top Author      │
└────────────────────┴────────────────────┴────────────────────┘
```

The dashboard also includes:

* KPI cards
* Interactive filtering
* Data exploration
* Analytical insights
* Theme switching
* GitHub / LinkedIn footer
* Responsive Streamlit layout

---

# 🧠 Dashboard KPIs

The dashboard summarizes important metrics such as:

```text
Total Records
Positive Records
Neutral Records
Negative Records
Average Sentiment Score
Positive Sentiment %
```

The KPI layer provides a quick executive overview before users
explore the detailed visualizations.

---

# 🎨 Dashboard Themes

The dashboard supports multiple visual themes:

```text
🌿 Vanilla
🧠 NLP Sage
🌙 Midnight NLP
```

The analytical content remains the same while the visual presentation
can be changed.

---

# 🔮 Text Prediction

A separate Streamlit interface is provided for sentiment prediction.

Run:

```bash
streamlit run dashboard/prediction.py
```

The prediction workflow uses the persisted:

```text
models/best_sentiment_model.pkl
models/tfidf_vectorizer.pkl
```

The user provides text and the trained pipeline produces a sentiment
prediction.

---

# ⚙️ Configuration

Project-level settings are centralized in:

```text
config/config.yaml
```

The configuration contains settings for:

* Project metadata
* Web source
* Scraping
* Data paths
* Cleaning
* NLP preprocessing
* TF-IDF
* Sentiment thresholds
* Machine-learning configuration
* Topic modeling
* Keyword analysis
* Dashboard behavior
* Reproducibility

This reduces unnecessary hard-coded configuration across the project.

---

# 🧩 Source Modules

Reusable Python modules are maintained inside:

```text
src/
```

### `scraper.py`

Handles reusable web scraping functionality.

### `preprocessing.py`

Contains text preprocessing and cleaning utilities.

### `sentiment.py`

Contains sentiment-analysis and model-related functionality.

### `topic_modeling.py`

Contains reusable topic-modeling functionality.

These modules are designed to separate reusable logic from the
notebook experimentation layer.

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

```bash
cd Web-Intelligence-NLP
```

---

## 2. Create a Virtual Environment

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

```bash
pip install -r requirements.txt
```

---

# 🧪 Run the Notebooks

Launch Jupyter:

```bash
jupyter notebook
```

Recommended execution order:

```text
01_Web_Scraping
       ↓
02_Data_Cleaning_and_EDA
       ↓
03_NLP_Preprocessing
       ↓
04_Sentiment_Analysis
       ↓
05_Keyword_and_Topic_Analysis
       ↓
06_Final_Insights
```

---

# 🚀 Run the Dashboard

From the project root:

```bash
streamlit run dashboard/dash.py
```

Or from inside the dashboard directory:

```bash
cd dashboard
streamlit run dash.py
```

---

# 🔮 Run the Prediction Interface

```bash
streamlit run dashboard/prediction.py
```

---

# 📊 Example Analytical Questions

The project can be used to explore questions such as:

### Sentiment

* What percentage of records are positive?
* How many negative records exist?
* What is the average sentiment score?

### Authors

* Which authors have the highest number of records?
* How does sentiment vary across authors?

### Keywords

* Which terms dominate the dataset?
* Which terms have the highest TF-IDF importance?

### Topics

* What major topics appear in the corpus?
* Which topics contain stronger positive sentiment?
* Which topics contain stronger negative sentiment?

### Text Characteristics

* Does text length vary across sentiment classes?
* Is there any visible relationship between text length and sentiment?

---

# 📁 Important Output Files

| File                             | Purpose                      |
| -------------------------------- | ---------------------------- |
| `scraped_quotes.csv`             | Raw scraped records          |
| `cleaned_quotes.csv`             | Cleaned dataset              |
| `nlp_ready_quotes.csv`           | NLP-ready text               |
| `sentiment_analysis_results.csv` | Sentiment + model results    |
| `keyword_analysis.csv`           | Keyword intelligence         |
| `topic_analysis.csv`             | Topic-level analysis         |
| `topic_summary.csv`              | Topic summaries              |
| `nlp_final_dataset.csv`          | Integrated final dataset     |
| `best_sentiment_model.pkl`       | Best trained sentiment model |
| `tfidf_vectorizer.pkl`           | Saved TF-IDF vectorizer      |
| `final_insight_report.txt`       | Final analytical report      |

---

# ⚠️ Limitations

This project has several important limitations.

### 1. Weak Sentiment Labels

Sentiment labels are generated using VADER rather than human-annotated
ground truth.

Therefore:

```text
VADER Label ≠ Guaranteed True Sentiment
```

Model performance should be interpreted as agreement with the
weak-labeling process.

---

### 2. Small / Practice-Oriented Source

Quotes to Scrape is primarily a practice scraping website.

Therefore, the dataset should not be interpreted as a representative
sample of general public opinion.

---

### 3. Domain Specificity

Quotes are different from:

* Product reviews
* Tweets
* News articles
* Customer feedback
* Support tickets

A sentiment model trained on this dataset may not generalize well to
other domains without additional training data.

---

### 4. Topic Interpretability

LDA and NMF topics are mathematically derived clusters.

Topic names and interpretations require human review.

---

### 5. Sentiment Score Interpretation

VADER compound scores provide sentiment strength according to the
VADER lexicon and rules. They are not probabilities.

---

# 🔐 Reproducibility

The project uses fixed random seeds where applicable.

Primary reproducibility parameter:

```text
random_state = 42
```

Model artifacts are stored using `joblib` for reuse between training
and prediction workflows.

---

# 🧠 Key Learning Outcomes

This project demonstrates practical experience with:

```text
Python
   ↓
Web Scraping
   ↓
Data Cleaning
   ↓
EDA
   ↓
NLP
   ↓
Feature Engineering
   ↓
Sentiment Analysis
   ↓
Machine Learning
   ↓
Topic Modeling
   ↓
Data Visualization
   ↓
Dashboard Development
```

It also demonstrates how separate analytical components can be
combined into a single end-to-end data science workflow.

---

# 💼 Portfolio Value

This project demonstrates the ability to work across multiple stages
of a modern data-science workflow:

### Data Collection

Collecting and structuring unstructured web information.

### Data Analytics

Cleaning, exploring, aggregating, and visualizing data.

### NLP

Transforming raw text into analytical features.

### Machine Learning

Training and comparing multiple classification algorithms.

### Unsupervised Learning

Discovering latent topics using LDA and NMF.

### Business Intelligence

Presenting analytical results through an interactive dashboard.

### Deployment-Oriented Development

Separating notebooks, reusable source modules, model artifacts,
configuration, and dashboard code.

---

# 🛠️ Future Improvements

Potential extensions include:

* Human-labeled sentiment dataset
* Larger multi-domain text corpus
* Transformer-based sentiment models
* BERT / DistilBERT comparison
* Named Entity Recognition
* Advanced topic modeling
* Embedding-based semantic search
* Automated data refresh
* Cloud deployment
* Dashboard authentication
* Model monitoring
* Experiment tracking
* Automated testing
* CI/CD pipeline

---

# 📸 Dashboard Preview

Add dashboard screenshots here after finalizing the UI.

```text
docs/
└── images/
    ├── dashboard-overview.png
    ├── dashboard-filters.png
    └── prediction-interface.png
```

Example Markdown:

```markdown
![Dashboard Overview](docs/images/dashboard-overview.png)
```

---

# 👨‍💻 Author

## S Mohammed Kaif

**Data Science • Data Analytics • Machine Learning • AI • Python**

### Connect

🐙 **GitHub**

[https://github.com/Shaik-Mohammed-Kaif](https://github.com/Shaik-Mohammed-Kaif)

💼 **LinkedIn**

[https://www.linkedin.com/in/s-mohammed-kaif-2a500a341/](https://www.linkedin.com/in/s-mohammed-kaif-2a500a341/)

---

# ⭐ Project Philosophy

```text
Learn
  ↓
Practice
  ↓
Analyze
  ↓
Build
  ↓
Improve
  ↓
Grow
```

---

## 📜 License

This project is licensed under the terms specified in the
`LICENSE` file.

---

<p align="center">

<strong>Built with Python, NLP, Machine Learning & Data Analytics.</strong>

</p>

<p align="center">

⭐ If you find this project useful, consider giving the repository a star.

</p>
