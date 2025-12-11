# 🎬 Hybrid Sales Forecasting (Prophet + XGBoost + Streamlit)

An intelligent forecasting system that combines **Prophet’s** trend-seasonality modeling with **XGBoost’s** machine learning power.
It analyzes historical retail sales data, predicts future performance, and provides a rich interactive dashboard built with **Streamlit**.

---

## 🧠 Overview

This project demonstrates a hybrid time series forecasting pipeline designed for retail or e-commerce sales prediction.
It integrates statistical forecasting (Prophet) and machine learning (XGBoost) to improve accuracy across diverse store–department combinations.

The system is fully modular, from data preprocessing and model training to evaluation and interactive visualization.

---

## ⚙️ Tech Stack
- **Python** – core development language
- **Prophet** – captures trend, seasonality, and holiday effects
- **XGBoost** – corrects Prophet’s residual errors with ML-based refinement
- **Streamlit** – builds an intuitive web app for exploration
- **Pandas, NumPy, Scikit-learn** – data handling and metrics
- **Plotly** – interactive charts and insights visualization 

---

## 🚀 Features
- Hybrid Prophet + XGBoost forecasting architecture
- Automatically splits data into train–test periods
- Generates forecasts for individual Store–Department combinations
- Provides MAE, RMSE, and MAPE metrics for both Prophet and Hybrid models
- Streamlit app to visualize past and future sales
- Allows users to simulate forecasts for future weeks
- Saves trained models and forecasts to structured folders

---

## 🧩 Architecture Flow
- **Data Preparation** – merges train.csv, features.csv, and stores.csv.
- **Feature Engineering** – creates lags, rolling means, and time-based features.
- **Prophet Model** – predicts base sales using trend and seasonality.
- **XGBoost Model** – learns residual errors to refine Prophet predictions.
- **Hybrid Forecast** – adds both outputs for the final prediction.
- **Evaluation** – computes Prophet vs Hybrid metrics and aggregates them.
- **Streamlit App** – visualizes model results and future forecasts interactively.

---

## 📊 Example Workflow
1. User selects number of pages to load (1 page = 20 movies).  
2. The app checks if the movies exist in the database.  
3. If not, it fetches new ones and saves them.  
4. A similarity model is built and cached.  
5. User searches for a movie -> app shows similar ones instantly.

---

## 💡 Future Enhancements
- Integrate SARIMAX and other time series models for comprehensive performance comparison.
- Implement scheduled pipeline execution to continuously update models with fresh data.
- Deploy via REST API on AWS or GCP for enterprise-scale accessibility and reliability. 

---

## 📌 Imp Note
- This project is a prototype and not yet production-ready. Some elements like holiday handling, future feature simulation, and lag generation use simplified assumptions for demonstration. With proper data pipelines and real-time feature forecasting, it can be scaled for production deployment.

---

## 🧰 Setup Instructions

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run training
```bash
python -m src.hybrid_train
```
- adjust groups based on your need

### 3️⃣ Launch Streamlit dashboard
```bash
streamlit run app/app.py
```

### 4️⃣ Evaluate and Check the Metrics
```bash
python -m src.eval
```
---

## 🖼️ Screenshots
<img width="1785" height="831" alt="image" src="https://github.com/user-attachments/assets/3e62f3fe-fd54-492d-9b9b-3310ed0c25ba" />

---

<img width="1761" height="837" alt="image" src="https://github.com/user-attachments/assets/b2150adc-063f-4a9f-9901-7e78efce015b" />

---

<img width="1788" height="829" alt="image" src="https://github.com/user-attachments/assets/6da4ddcd-7b21-46d7-9390-58f5048a744f" />



