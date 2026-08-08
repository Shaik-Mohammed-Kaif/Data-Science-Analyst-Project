# 🎬 YouTube Trending Videos Analytics

<p align="center">

### 📊 End-to-End Data Science & Business Intelligence Project

An interactive analytics platform for understanding YouTube
trending videos, audience engagement, content performance,
publishing behavior, and channel-level insights.

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical_Analysis-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-Analytics-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</p>

---

## 👨‍💻 Author

### **Shaik Mohammed Kaif**

🎓 Computer Science Engineering | Data Science

💡 Data Science | Data Analytics | Machine Learning | SQL | Python | Business Intelligence

📌 GitHub:  
**[Shaik-Mohammed-Kaif](https://github.com/Shaik-Mohammed-Kaif)**

📂 Repository:  
**[Data-Science-Analyst-Project](https://github.com/Shaik-Mohammed-Kaif/Data-Science-Analyst-Project)**

---

# 📌 Project Overview

**YouTube Trending Videos Analytics** is an end-to-end Data Science and Business Intelligence project designed to analyze YouTube trending video data.

The project transforms raw video-level data into meaningful analytical insights related to:

> **Reach → Engagement → Content → Publishing Strategy → Channel Performance**

The project combines data engineering, data cleaning, exploratory analysis, SQL analytics, NLP-oriented feature engineering, statistical analysis, interactive visualization, and dashboard development.

---

# 🎯 Business Objectives

The primary goal of this project is to understand what factors are associated with better performance among trending YouTube videos.

### Key business questions

- Which videos generate the highest number of views?
- Which channels achieve the highest audience reach?
- Which categories receive the highest total views?
- Which categories achieve stronger engagement?
- Which publishing hours perform better?
- Which publishing sessions perform better?
- Does weekend publishing perform differently from weekday publishing?
- Does video quality influence average views?
- Does caption availability relate to engagement?
- Which duration category performs best?
- How do title length and description length vary across videos?
- How do tags and hashtags contribute to content analysis?
- Which channels combine high reach and strong engagement?
- Which video attributes are associated with audience interaction?
- Which content characteristics can support better publishing decisions?

---

# 🔄 End-to-End Project Workflow

```text
                         ┌────────────────────────┐
                         │   YouTube Dataset      │
                         │ CSV / SQLite Database  │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │     Data Loading        │
                         │     Pandas / SQLite     │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │    Data Validation      │
                         │ Missing / Types / Dupes │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │     Data Cleaning       │
                         │ Numeric / Text / Dates  │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │  Feature Engineering    │
                         │ NLP / Metrics / Dates   │
                         └────────────┬───────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
             ┌────────────┐   ┌────────────┐   ┌──────────────┐
             │    SQL     │   │    EDA     │   │  Streamlit   │
             │ Analytics  │   │ Analytics  │   │  Dashboard   │
             └──────┬─────┘   └──────┬─────┘   └───────┬──────┘
                    │                │                 │
                    └────────────────┼─────────────────┘
                                     ▼
                         ┌────────────────────────┐
                         │ Business Insights      │
                         │ & Recommendations      │
                         └────────────────────────┘
```

---

# 🧰 Technology Stack

## 🐍 Programming & Data Analysis

- Python
- Pandas
- NumPy

## 📊 Data Visualization

- Matplotlib
- Seaborn
- Plotly

## 🗄️ Database & Querying

- SQL
- SQLite

## 📓 Development Environment

- Jupyter Notebook
- VS Code / Python Environment

## 🎨 Dashboard

- Streamlit
- Plotly Interactive Visualizations
- Custom CSS / HTML styling

## 📁 Data Formats

- CSV
- SQLite Database
- Jupyter Notebook
- PDF
- PowerPoint

---

# 🏗️ Project Architecture

```text
Raw YouTube Data
       │
       ▼
Data Ingestion
       │
       ├── CSV
       │
       └── SQLite
       │
       ▼
Data Cleaning & Validation
       │
       ▼
Feature Engineering
       │
       ├── Engagement Metrics
       ├── Date Features
       ├── Publishing Features
       ├── Text Features
       ├── URL Features
       ├── Hashtag Features
       └── Tag Features
       │
       ├───────────────┬────────────────┐
       ▼               ▼                ▼
      SQL             EDA          Streamlit
   Analytics       Analysis       Dashboard
       │               │                │
       └───────────────┼────────────────┘
                       ▼
               Business Insights
                       │
                       ▼
                Recommendations
```

---

# 📂 Repository Structure

```text
Data-Science-Analyst-Project/
│
├── .gitignore
│
└── YouTube-Project-Data-Analytics/
    │
    ├── README.md
    │
    ├── Reports-For-Youtube/
    │   ├── YouTube-Trending-Analytics.pdf
    │   ├── YouTube-Trending-Analytics.pptx
    │   └── YouTube_Trending_Analytics_Report.pdf
    │
    ├── Streamlit-Yt-Dashbord/
    │   └── app.py
    │
    ├── Youtube-google-Console-Dataset/
    │   ├── trending_videos.csv
    │   ├── trending_videos_data.csv
    │   └── trending_videos.db
    │
    ├── Youtube-IPYNB-File/
    │   ├── youtube_analysis.ipynb
    │   ├── Youtube-Analysis-for-google-Dset-EDA.ipynb
    │   └── SQL_Yotube_Data-Analysis.ipynb
    │
    └── requirements.txt
```

---

# 📊 Dataset

The project uses YouTube trending video data containing video-level, channel-level, publishing, engagement, quality, duration, caption, tag, hashtag, and NLP-related information.

### Main dataset formats

```text
CSV
SQLite Database
```

### Available dataset files

```text
trending_videos.csv
trending_videos_data.csv
trending_videos.db
```

---

# 🧾 Dataset Feature Groups

## 🎬 Video Information

```text
video_id
title
description
channel_id
channel_title
category_id
category_name
```

---

## 📅 Publishing Information

```text
published_at
published_at_date
publish_year
publish_month
publish_month_name
publish_day
publish_day_name
publish_week
publish_hour
publish_minute
publish_second
publish_session
is_weekend
weekend_flag
```

---

## 📈 Performance Metrics

```text
view_count
like_count
comment_count
engagement_score
engagement_rate
like_rate
comment_rate
```

---

## 🎞️ Video Attributes

```text
duration
duration_seconds
duration_category
definition
is_hd
caption
has_caption
```

---

## 🏷️ URL / Hashtag / Tag Features

```text
urls_links
url_count
url_types
hashtags
hashtag_count
tags
tags_clean
tag_count
```

---

## 🧠 NLP / Text Features

```text
clean_description
tokens
processed_text
title_length
title_word_count
description_length
description_word_count
```

---

# 🧹 Data Cleaning

The cleaning pipeline performs type-safe preprocessing and prepares the dataset for downstream analysis.

### Major cleaning operations

- Numeric type conversion
- Date/time conversion
- Missing value handling
- Duplicate removal
- Boolean normalization
- Text normalization
- Category standardization
- Safe tag parsing
- URL extraction
- Hashtag extraction
- Derived feature creation

### Numeric fields

Metrics such as:

```text
view_count
like_count
comment_count
favorite_count
engagement_score
engagement_rate
like_rate
comment_rate
duration_seconds
```

are converted into appropriate numeric types.

---

# 🧠 Feature Engineering

Additional analytical features are generated from the available video information.

### Engagement Rate

A derived engagement metric is calculated using audience interactions relative to views.

```text
Engagement Rate =
(Likes + Comments) / Views
```

This allows comparison between videos with different levels of reach.

---

### 📝 Text Features

The project analyzes text-related characteristics such as:

```text
Title Length
Title Word Count
Description Length
Description Word Count
```

These features help investigate relationships between content structure and video performance.

---

### 🔗 URL Features

Descriptions are analyzed for:

```text
URL Count
URL Presence
```

---

### 🏷️ Hashtag Features

The project derives:

```text
Hashtag Count
Hashtag Presence
```

from video descriptions where available.

---

### 🎯 Tag Features

Tags are cleaned and transformed into usable analytical features:

```text
tags_clean
tag_count
```

---

# 📊 Exploratory Data Analysis

The EDA phase investigates video performance across multiple dimensions.

## Key EDA Areas

### 1. Video Performance

- Total views
- Average views
- Total likes
- Total comments
- Engagement rate

### 2. Category Analysis

- Views by category
- Engagement by category
- Category-level performance

### 3. Channel Analysis

- Top channels by views
- Channel performance
- Channel-level audience reach

### 4. Publishing Analysis

- Views by publishing hour
- Publishing session performance
- Weekday vs weekend performance
- Publishing trends

### 5. Video Attribute Analysis

- Duration category
- Video definition
- HD vs non-HD
- Caption availability

### 6. Content Analysis

- Title length
- Description length
- Tags
- Hashtags
- URLs

---

# 🗄️ SQL Analytics

SQL is used to perform structured business analysis and aggregation.

### Example SQL operations

```sql
SELECT *
FROM trending_videos;
```

### Total views

```sql
SELECT
    SUM(view_count) AS total_views
FROM trending_videos;
```

### Average views

```sql
SELECT
    AVG(view_count) AS average_views
FROM trending_videos;
```

### Top channels

```sql
SELECT
    channel_title,
    SUM(view_count) AS total_views
FROM trending_videos
GROUP BY channel_title
ORDER BY total_views DESC
LIMIT 10;
```

### Category performance

```sql
SELECT
    category_name,
    SUM(view_count) AS total_views,
    AVG(engagement_rate) AS avg_engagement_rate
FROM trending_videos
GROUP BY category_name
ORDER BY total_views DESC;
```

### Publishing hour analysis

```sql
SELECT
    publish_hour,
    AVG(view_count) AS avg_views
FROM trending_videos
GROUP BY publish_hour
ORDER BY avg_views DESC;
```

---

# 🎨 Interactive Streamlit Dashboard

The project includes a custom-built interactive Streamlit dashboard.

### Dashboard capabilities

```text
📊 KPI Cards
🔎 Interactive Filters
📈 Performance Charts
📡 Channel Analytics
📁 Category Analytics
📅 Publishing Analytics
🎞️ Video Quality Analysis
💬 Caption Analysis
⏱️ Duration Analysis
🧠 NLP / Content Analytics
💡 Business Insights
🔍 Data Explorer
```

---

# 🎛️ Interactive Filters

The dashboard supports filtering by:

```text
📅 Date Range
📁 Category
📡 Channel
🕐 Publish Session
🎞️ Video Quality
💬 Caption Availability
⏱️ Duration Category
📆 Weekend / Weekday
```

All major dashboard metrics and charts respond to the selected filters.

---

# 📌 Dashboard KPI Layer

The dashboard calculates key performance indicators including:

### 🎬 Total Videos

Total number of videos in the filtered dataset.

### 👁️ Total Views

Total audience reach represented by video views.

### 👍 Total Likes

Total likes received by filtered videos.

### 💬 Total Comments

Total audience comments.

### 🔥 Average Engagement Rate

Average engagement rate across filtered videos.

### 📈 Average View Count

Average views per video.

---

# 📈 Interactive Visualization Layer

The dashboard contains interactive Plotly charts for:

- Views over time
- Top videos
- Top channels
- Category-wise views
- Category-wise engagement
- Publishing hour performance
- Video quality performance
- Video engagement ranking

Charts support interactive:

```text
Hover
Zoom
Pan
Legend Filtering
Dynamic Filtering
```

---

# 🧠 Business Insights Layer

The dashboard automatically generates business-oriented insights such as:

```text
🏆 Top Category
🔥 Highest Engagement Category
📈 Best Publishing Hour
🕐 Best Publishing Session
📡 Top Channel
📆 Better Performing Day
💬 Caption Winner
⏱️ Best Duration
🎬 Highest Viewed Video
```

These insights transform descriptive analytics into actionable business understanding.

---

# 📋 Data Explorer

The dashboard also provides an interactive view of the filtered dataset.

This allows users to:

- Inspect records
- Review columns
- Validate filtered results
- Explore video-level information
- Understand underlying data

---

# 📑 Reports & Documentation

The project contains supporting documentation inside:

```text
Reports-For-Youtube/
```

Available files include:

```text
YouTube-Trending-Analytics.pdf
YouTube-Trending-Analytics.pptx
YouTube_Trending_Analytics_Report.pdf
```

These documents provide additional project-level analysis and presentation material.

---

# 📓 Jupyter Notebooks

The repository contains multiple notebooks for analysis and SQL work.

```text
Youtube-IPYNB-File/
```

### Included notebooks

```text
youtube_analysis.ipynb
Youtube-Analysis-for-google-Dset-EDA.ipynb
SQL_Yotube_Data-Analysis.ipynb
```

The notebooks cover data exploration, analysis, visualization, feature engineering, and SQL-based analytics.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/Shaik-Mohammed-Kaif/Data-Science-Analyst-Project.git
```

---

## 2. Navigate to the YouTube project

```bash
cd Data-Science-Analyst-Project
cd YouTube-Project-Data-Analytics
```

---

## 3. Create a virtual environment

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
source .venv/bin/activate
```

---

# 📦 Install Dependencies

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Or update the requirements file:

```bash
pip freeze > requirements.txt
```

---

# ▶️ Run Streamlit Dashboard

Navigate to the dashboard directory:

```bash
cd Streamlit-Yt-Dashbord
```

Run:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

# 🗃️ SQLite Database

The dashboard can work with the SQLite database included in:

```text
Youtube-google-Console-Dataset/
```

Database:

```text
trending_videos.db
```

The project also contains CSV alternatives:

```text
trending_videos.csv
trending_videos_data.csv
```

---

# 🔧 Git & GitHub Workflow

This project is maintained using Git and GitHub.

## Initialize Git

```bash
git init
```

## Check repository status

```bash
git status
```

## Add all files

```bash
git add .
```

## Commit changes

```bash
git commit -m "Add YouTube Data Science Analytics Project"
```

## Rename branch to main

```bash
git branch -M main
```

## Add remote repository

```bash
git remote add origin https://github.com/Shaik-Mohammed-Kaif/Data-Science-Analyst-Project.git
```

## Verify remote

```bash
git remote -v
```

## Push project

```bash
git push -u origin main
```

---

# 🔄 Updating the Project

Whenever you modify the project:

```bash
git status
```

Then:

```bash
git add .
```

Commit:

```bash
git commit -m "Update YouTube analytics project"
```

Push:

```bash
git push
```

---

# 🔀 Working with Remote Changes

Before pushing if the remote repository has changes:

```bash
git pull --rebase origin main
```

Then:

```bash
git push -u origin main
```

---

# 🧹 Useful Git Commands

### View commit history

```bash
git log --oneline
```

### View current branch

```bash
git branch
```

### View remote URL

```bash
git remote -v
```

### Check differences

```bash
git diff
```

### Check staged changes

```bash
git diff --cached
```

### Unstage a file

```bash
git restore --staged filename
```

### Discard local changes

```bash
git restore filename
```

### Pull latest changes

```bash
git pull
```

### Push latest changes

```bash
git push
```

---

# 📌 Project Skills Demonstrated

This project demonstrates practical experience in:

```text
Python
Pandas
NumPy
SQL
SQLite
Data Cleaning
Data Validation
Exploratory Data Analysis
Feature Engineering
NLP Feature Engineering
Statistical Analysis
Data Visualization
Plotly
Streamlit
Business Intelligence
Dashboard Development
Git
GitHub
```

---

# 💼 Business Intelligence Perspective

The project is designed around the complete analytics lifecycle:

```text
Raw Data
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Exploratory Data Analysis
   ↓
SQL Analytics
   ↓
Visualization
   ↓
Interactive Dashboard
   ↓
Business Insights
   ↓
Decision Support
```

This approach demonstrates how raw data can be transformed into information that supports content strategy and performance analysis.

---

# 🎯 Key Analytical Areas

| Area | Analysis |
|---|---|
| Audience Reach | Views & Average Views |
| Audience Reaction | Likes |
| Audience Interaction | Comments |
| Engagement | Engagement Rate |
| Content | Title & Description Features |
| Publishing | Hour, Session & Day |
| Channel | Channel-Level Performance |
| Category | Category-Level Performance |
| Video Quality | Definition / HD |
| Captions | Caption Availability |
| Duration | Duration Category |
| Metadata | Tags / Hashtags / URLs |
| Database | SQLite |
| BI Dashboard | Streamlit + Plotly |

---

# 🚀 Future Scope

The project can be extended with additional capabilities such as:

- Machine Learning-based view prediction
- Trending probability prediction
- Viral video classification
- Channel performance forecasting
- Audience engagement prediction
- Automated YouTube API data ingestion
- Scheduled data refresh
- Advanced NLP sentiment analysis
- Topic modeling
- Keyword extraction
- Recommendation system
- Automated business reports
- Power BI integration
- Cloud deployment
- Advanced monitoring and analytics

---

# 🧠 Project Outcome

The project provides an integrated environment for analyzing YouTube trending video performance from multiple business perspectives.

It combines:

```text
Data Engineering
        +
Data Analysis
        +
SQL
        +
NLP Feature Engineering
        +
Visualization
        +
Business Intelligence
        +
Interactive Dashboard
```

The final result is a reusable analytics workflow that converts YouTube video-level data into structured performance analysis and business insights.

---

# ⭐ Why This Project Is Portfolio-Ready

This project demonstrates more than basic data visualization.

It showcases an end-to-end workflow involving:

✅ Real-world dataset handling  
✅ Data cleaning  
✅ Data transformation  
✅ Feature engineering  
✅ SQL analytics  
✅ Exploratory Data Analysis  
✅ Interactive visualization  
✅ Business KPI development  
✅ Streamlit dashboard development  
✅ SQLite database integration  
✅ NLP-oriented content features  
✅ Git/GitHub version control  
✅ Business insight generation  

---

# 🏁 Conclusion

**YouTube Trending Videos Analytics** demonstrates how Data Science and Business Intelligence techniques can be combined to understand video performance, audience behavior, publishing patterns, content characteristics, and channel-level trends.

The project follows a complete analytical lifecycle:

> **Collect → Clean → Transform → Analyze → Visualize → Interpret → Communicate**

The dashboard provides an interactive interface for exploring performance metrics and filtering the data, while the notebooks and SQL analysis provide deeper analytical support.

---

# 👨‍💻 Author

## Shaik Mohammed Kaif

**Computer Science Engineering | Data Science**

### Areas of Interest

```text
Data Science
Data Analytics
Machine Learning
Python
SQL
Business Intelligence
Data Visualization
Streamlit
Power BI
Tableau
```

---

# 🔗 Connect & Explore

### GitHub Profile

👉 [Shaik-Mohammed-Kaif](https://github.com/Shaik-Mohammed-Kaif)

### Main Repository

👉 [Data-Science-Analyst-Project](https://github.com/Shaik-Mohammed-Kaif/Data-Science-Analyst-Project)

### Project Folder

👉 [YouTube-Project-Data-Analytics](https://github.com/Shaik-Mohammed-Kaif/Data-Science-Analyst-Project/tree/main/YouTube-Project-Data-Analytics)

---

<p align="center">

### ⭐ If you find this project useful, consider giving the repository a star!

</p>

<p align="center">

**Built with 🐍 Python • 📊 Data Science • 🗄️ SQL • 🎨 Streamlit • 📈 Plotly**

</p>
