# Data Science Internship Projects

This repository contains a series of data science projects designed to help interns develop skills from **beginner** to **advanced** levels. Each project includes a dataset, objectives, and step-by-step guidance.

---

## 📚 Beginner Level Projects

### 1. Exploratory Data Analysis on Titanic Survival
- **Objective:** Analyze passenger attributes and survival rate on the Titanic.
- **Dataset:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
- **Learning Outcomes:**
  - Data wrangling
  - Univariate and bivariate analysis
  - Visualizations: bar plots, histograms, boxplots
- **Hints:**
  - Use `groupby()` with survival labels
  - Handle missing age values with median or mean
  - Visualize survival rate by class and gender

---

### 2. Netflix Titles Dataset Analysis
- **Objective:** Analyze Netflix shows and movie trends.
- **Dataset:** Netflix Movies and TV Shows
- **Learning Outcomes:**
  - Text parsing, date handling
  - Data visualization
  - Genre classification
- **Hints:**
  - Parse `date_added` to extract year
  - Use `str.split()` for multi-label genres
  - Find top countries by content volume

---

### 3. CO2 Emissions by Country Over Time
- **Objective:** Analyze historical trends of CO2 emissions.
- **Dataset:** Our World in Data – CO2 Emissions
- **Learning Outcomes:**
  - Time series analysis
  - Multi-index grouping
  - Line charts, heatmaps
- **Hints:**
  - Normalize emissions per capita
  - Use rolling averages for trends
  - Compare 1990 vs. 2020 for top 10 countries

---

### 4. Customer Churn Investigation
- **Objective:** Understand patterns in telecom customer churn.
- **Dataset:** Telco Customer Churn Dataset
- **Learning Outcomes:**
  - Binary target analysis
  - Pivot tables, categorical comparison
  - Correlation analysis
- **Hints:**
  - Focus on contract type, payment method, tenure
  - Use stacked bar charts for churn ratios
  - Handle class imbalance

---

### 5. World Happiness Report Analysis
- **Objective:** Identify patterns in global happiness.
- **Dataset:** World Happiness Report
- **Learning Outcomes:**
  - Correlation analysis
  - Global comparisons
  - Bubble maps, bar plots
- **Hints:**
  - Plot GDP vs. Happiness Score
  - Highlight top 10 happiest/unhappiest countries
  - Use `seaborn.heatmap()` for correlation

---

## 📊 Intermediate Level Projects

### 1. Predicting Student Exam Performance
- **Objective:** Predict student scores based on demographics and preparation.
- **Dataset:** Student Performance Dataset
- **Learning Outcomes:** Feature encoding, regression modeling, RMSE evaluation
- **Hints:** LabelEncode, OneHotEncode, train/test split, MAE/RMSE

---

### 2. House Price Prediction
- **Objective:** Predict housing prices using features like location and size.
- **Dataset:** Ames Housing Dataset
- **Learning Outcomes:** Advanced EDA, regression modeling, feature engineering
- **Hints:** Log-transform prices, select top predictors, residual analysis

---

### 3. Credit Card Fraud Detection
- **Objective:** Classify fraudulent credit card transactions.
- **Dataset:** Credit Card Fraud Dataset
- **Learning Outcomes:** Imbalanced classification, model evaluation, fraud detection strategy
- **Hints:** StratifiedKFold, StandardScaler, AUC-ROC evaluation

---

## 🚀 Advanced Level Projects

### 1. Predicting Hospital Readmissions
- **Objective:** Predict whether a patient will be readmitted within 30 days.
- **Dataset:** Diabetes Readmission Dataset
- **Learning Outcomes:** Healthcare preprocessing, classification modeling, ethical considerations
- **Hints:** Balance dataset, explain false positives/negatives, handle high-cardinality columns

---

### 2. End-to-End ML Pipeline for Sales Forecasting
- **Objective:** Forecast daily sales for a retail store using ML and deploy it.
- **Dataset:** Rossmann Store Sales
- **Description:**
  - Preprocess temporal data and engineer features
  - Use Random Forest or XGBoost for predictions
  - Save the trained model using `joblib`
  - Deploy using **FastAPI** as backend and **Next.js** as frontend
- **Learning Outcomes:** Model deployment, time series features, production-ready pipeline
- **Hints:** Extract day of week, promo, holiday info, use lag features, integrate FastAPI or Flask

#### 🛠 Steps to Run Sales Forecast Project

1. **Clone the repository:**

git clone <your-repo-url>
cd <repo-root>

2. **Setup backend (FastAPI):**

cd backend_sales
python -m venv venv
venv\Scripts\activate       # Windows
# or source venv/bin/activate  # Linux/Mac
python -m pip install -r requirements.txt
uvicorn app:app --reload

3. Setup frontend (Next.js): 
cd frontend_sales/frontend
npm install
npm run dev

