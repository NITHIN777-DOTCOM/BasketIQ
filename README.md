<div align="center">

# 🛒 BasketIQ

### Intelligent Retail Analytics & Business Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon-336791?logo=postgresql)](https://neon.tech/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](YOUR_STREAMLIT_LINK)
[![API Docs](https://img.shields.io/badge/API-Swagger-85EA2D?logo=swagger)](YOUR_API_LINK/docs)

</div>

---

# 📌 Overview

BasketIQ is a **production-style Retail Intelligence Platform** that transforms raw supermarket transactions into actionable business insights.

Unlike traditional analytics dashboards, BasketIQ combines:

- 📊 Business Intelligence
- 🤖 Machine Learning
- 🗄 PostgreSQL Persistence
- ⚡ FastAPI REST APIs
- 📄 PDF Reporting
- 📈 Interactive Dashboard

into one unified analytics platform.

---

# 🏗 Architecture

                          BasketIQ Architecture


                    +--------------------------------+
                    |      Transaction Dataset       |
                    |   CSV / PostgreSQL / Faker     |
                    +---------------+----------------+
                                    |
                                    |
                                    ▼
                    +--------------------------------+
                    |  Data Validation & Cleaning    |
                    | Missing Values • Duplicates    |
                    | Outliers • Feature Encoding    |
                    +---------------+----------------+
                                    |
                                    |
                                    ▼
                    +--------------------------------+
                    |      Feature Engineering       |
                    | RFM • Revenue • Temporal       |
                    | Customer & Product Features    |
                    +---------------+----------------+
                                    |
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
+--------------------+   +--------------------+   +----------------------+
| Isolation Forest   |   | K-Means Clustering |   | Apriori Association  |
| Anomaly Detection  |   | Customer Segments  |   | Market Basket Rules  |
+--------------------+   +--------------------+   +----------------------+
         │                          │                          │
         └───────────────┬──────────┴──────────┬───────────────┘
                         │                     │
                         ▼                     ▼
              +-------------------------------------------+
              |        PostgreSQL (Neon Database)         |
              | Persistent Analytics & ML Results Storage |
              +-------------------+-----------------------+
                                  |
                  ┌───────────────┴────────────────┐
                  │                                │
                  ▼                                ▼
      +-------------------------+      +-------------------------+
      |     FastAPI Backend     |      |   Streamlit Dashboard   |
      | REST APIs + Swagger UI  |      | Interactive Analytics   |
      +-----------+-------------+      +-----------+-------------+
                  │                                │
                  └───────────────┬────────────────┘
                                  │
                                  ▼
                    +-------------------------------+
                    |     Business Intelligence      |
                    | PDF Reports • Insights • APIs |
                    +-------------------------------+

<p align="center">
<img src="images/architecture.png" width="900">
</p>

---

# 🚀 Live Dashboard

<p align="center">
<img src="images/dashboard.png" width="900">
</p>

Live Demo:

https://basketiq-dev.streamlit.app/

---

# 🌐 REST API

BasketIQ exposes analytics through a FastAPI backend.

Swagger UI:

https://YOUR_API/docs

<p align="center">
<img src="images/swagger.png" width="900">
</p>

Available endpoints:

```
GET /api/anomalies
GET /api/customers
GET /api/products
GET /api/summary
GET /health
```

---

# ✨ Features

## 📊 Customer Analytics

- Customer RFM Analysis
- K-Means Customer Segmentation
- Cohort Retention Analysis

---

## 📦 Product Analytics

- Product RFM
- Revenue Analysis
- Product Segmentation

---

## 🤖 Machine Learning

### Isolation Forest

Detect anomalous customer purchases.

### K-Means

Segment customers based on purchasing behaviour.

### Apriori

Discover product associations.

---

## 🗄 Data Engineering

- PostgreSQL Persistence
- SQLAlchemy ORM
- Automatic Schema Creation
- Synthetic Dataset Generation
- Feature Engineering Pipeline

---

## 🌐 Backend Services

- FastAPI REST API
- OpenAPI Documentation
- Swagger UI
- JSON Endpoints
- Health Monitoring

---

## 📈 Dashboard

Interactive Streamlit Dashboard

- Executive KPIs
- Customer Segments
- Product Analytics
- Cohort Retention
- Anomaly Detection
- Revenue Trends

---

## 📄 Reporting

Generate professional PDF reports containing

- KPIs
- Customer Insights
- Product Insights
- Revenue Summary
- ML Results

---

# 🧠 Machine Learning Pipeline

```
Transactions

        │

        ▼

Feature Engineering

        │

 ┌──────┼──────────────┐

 ▼      ▼              ▼

RFM   Isolation    Apriori

       Forest

        │

        ▼

Customer Insights

        │

        ▼

PostgreSQL

        │

 ┌──────┴───────────┐

 ▼                  ▼

FastAPI        Streamlit
```

---

# ⚙ Tech Stack

| Layer | Technology |
|--------|------------|
| Language | Python |
| Dashboard | Streamlit |
| Backend | FastAPI |
| Database | PostgreSQL (Neon) |
| ORM | SQLAlchemy |
| Machine Learning | Scikit-Learn |
| Data Analysis | Pandas |
| Charts | Plotly |
| Reports | FPDF2 |
| API Docs | Swagger/OpenAPI |

---

# 📂 Project Structure

```
BasketIQ/

├── api/
│   └── main.py
│
├── app/
│   ├── app.py
│   └── pages/
│
├── db/
│   └── postgres.py
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── pipeline.py
│   ├── anomaly.py
│   └── market_basket.py
│
├── notebook/
│
├── models/
│
├── outputs/
│
├── data/
│
├── requirements.txt
│
└── README.md
```

---

# ▶ Running Locally

Clone

```bash
git clone https://github.com/YOUR_USERNAME/BasketIQ.git

cd BasketIQ
```

Install

```bash
pip install -r requirements.txt
```

Run Dashboard

```bash
streamlit run app/app.py
```

Run API

```bash
uvicorn api.main:app --reload
```

Swagger

```
http://localhost:8000/docs
```

---

# 📈 Sample Dataset

Synthetic supermarket dataset generated using **Faker**

- 50,000 Transactions
- 500 Customers
- 200 Products
- Seasonal Trends
- Peak Shopping Hours
- Realistic Pricing

---

# 🚀 Future Work

- Incremental ETL
- Recommendation API
- Data Quality Engine
- Docker Deployment
- Background Workers
- Automated Weekly Reports

---

# 📜 License

MIT License
