<div align="center">

# 🧺 BasketIQ

### Retail Intelligence Platform — ML Analytics + REST API + PostgreSQL

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon-336791?logo=postgresql)](https://neon.tech/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)][(YOUR_STREAMLIT_URL)](https://basketiq-dev.streamlit.app/)
Base URL: Available when running locally at `http://localhost:8000`
![Swagger UI](images/image.png)

</div>

---

## What is BasketIQ?

BasketIQ is an end-to-end retail analytics platform that takes raw supermarket transaction data and runs it through a multi-stage ML pipeline — detecting anomalies, segmenting customers, mining product associations, computing cohort retention, and persisting results to PostgreSQL. Insights are served through both an interactive Streamlit dashboard and a REST API that any external application can consume.

---

## Architecture

```
CSV / Excel Upload
        │
        ▼
┌─────────────────────┐
│   Data Ingestion    │  Auto-detects encoding, delimiter, file format
│   & Preprocessing   │  Removes cancellations, negatives, duplicates
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Feature Engineering│  RFM scores, TotalAmount, Hour, DayOfWeek,
│                     │  AgeGroup, InvoiceTotal, ItemCount
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│                   ML Pipeline                        │
│                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Isolation  │  │   K-Means   │  │   Apriori   │ │
│  │   Forest    │  │ Clustering  │  │    (MBA)    │ │
│  │  Anomaly    │  │  Customer   │  │  Product    │ │
│  │ Detection   │  │ Segmentation│  │ Associations│ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                     │
│  ┌─────────────┐  ┌─────────────┐                  │
│  │   Cohort    │  │  Product    │                  │
│  │  Retention  │  │    RFM      │                  │
│  │  Analysis   │  │  Segments   │                  │
│  └─────────────┘  └─────────────┘                  │
└─────────┬───────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│     PostgreSQL      │  Upserts on natural keys
│   (Neon Cloud)      │  anomalies, segments, rules, trends
└─────────┬───────────┘
          │
     ┌────┴────┐
     ▼         ▼
Streamlit   FastAPI
Dashboard   REST API
            /docs
```

---

## Features

### Machine Learning
- **Anomaly Detection** — Isolation Forest flags suspicious invoices (configurable contamination rate). Risk-bucketed into High / Medium / Low with score normalization.
- **Customer Segmentation** — K-Means on RFM features (Recency, Frequency, Monetary, AvgSpend). Optimal k via elbow method. Segments labelled by monetary value: Premium Loyalists, Regular Customers, Budget Shoppers, Casual Buyers.
- **Market Basket Analysis** — Apriori algorithm on invoice-product matrix. Generates association rules with Support, Confidence, Lift, Conviction metrics. Rule strength classification: Strong (lift ≥ 3), Moderate (≥ 2), Weak (≥ 1).
- **Cohort Retention Analysis** — Groups customers by first-purchase month, tracks monthly return rates as a retention % matrix. Visualized as a heatmap.
- **Product RFM** — Scores each product on Recency, Frequency, Monetary (1–4 scale). Segments: Star Product, Reliable, Declining, Niche, Low Performer.

### Data Engineering
- Robust ingestion: auto-detects CSV delimiter (`,` `;` `\t` `|`), encoding (UTF-8, Latin-1, CP1252), and Excel format via magic bytes
- SQLAlchemy 2.0 ORM with PostgreSQL upserts on natural keys — no duplicate rows on re-runs
- Graceful DB fallback — pipeline completes and saves CSV outputs even if PostgreSQL is unreachable

### REST API (FastAPI)
- `GET /api/anomalies` — anomalous invoices, filterable by risk level
- `GET /api/customers` — customer segments, filterable by segment name
- `GET /api/products` — product RFM data, filterable by segment
- `GET /api/summary` — aggregated business metrics in one call
- `GET /health` — health check
- Interactive docs at `/docs` (Swagger UI)

### Dashboard (Streamlit)
- 7 pages: Overview, Anomaly Detection, Customer Segmentation, Market Basket, Temporal Analysis, Cohort Retention, Product Intelligence
- Configurable pipeline settings (K-Means clusters, contamination %, Apriori thresholds) with plain-English tooltips for non-technical users
- One-click PDF report export (fpdf2) — cover page, anomaly summary, segment stats, top rules, monthly revenue, top products
- CSV download on every data table

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.13 | Core pipeline |
| Dashboard | Streamlit | Interactive UI |
| Backend | FastAPI | REST API |
| Database | PostgreSQL (Neon) | Cloud persistence |
| ORM | SQLAlchemy 2.0 | DB layer |
| ML | Scikit-learn | Clustering, anomaly detection |
| MBA | mlxtend | Apriori, association rules |
| Data | Pandas, NumPy | Processing |
| Charts | Plotly | Visualizations |
| Reports | fpdf2 | PDF generation |
| API Docs | Swagger / OpenAPI | Auto-generated |

---

## Project Structure

```
BasketIQ/
├── api/
│   └── main.py              # FastAPI app — 4 endpoints + Swagger
├── app/
│   └── app.py               # Streamlit dashboard — 7 pages
├── db/
│   └── postgres.py          # SQLAlchemy models, save_results(), load_results()
├── src/
│   ├── pipeline.py          # Orchestrates full ETL + ML pipeline
│   ├── preprocessing.py     # Data cleaning, validation
│   ├── features.py          # RFM, feature engineering
│   └── train.py             # All ML models + cohort + product RFM
├── utils/
│   └── pdf_report.py        # PDF report generator
├── notebook/
│   └── EDA.ipynb            # Exploratory data analysis
├── generate_data.py         # Synthetic dataset generator (Faker)
├── run_api.py               # FastAPI server launcher
├── requirements.txt
└── .env.example
```

---

## Running Locally

**1. Clone & install**
```bash
git clone https://github.com/YOUR_USERNAME/BasketIQ.git
cd BasketIQ
pip install -r requirements.txt
```

**2. Set up environment**
```bash
cp .env.example .env
# Add your PostgreSQL connection string to .env
DATABASE_URL=postgresql+psycopg2://user:password@host/dbname
```

**3. Generate dataset**
```bash
python generate_data.py
# Creates data/transactions.csv — 50,000 synthetic transactions
```

**4. Run dashboard**
```bash
streamlit run app/app.py
# Open http://localhost:8501 and upload data/transactions.csv
```

**5. Run API (separate terminal)**
```bash
python run_api.py
# Open http://localhost:8000/docs for Swagger UI
```

---

## Dataset

Synthetically generated using Faker — 50,000 transactions across 500 customers and 200 products (2023–2024).

Realistic features:
- Double-peak hourly distribution (10am–2pm, 5pm–8pm)
- Holiday seasonality spike (Nov–Dec)
- Pareto product popularity (top 20% products = ~80% of transactions)
- Price-quantity inverse relationship (expensive items bought in smaller quantities)
- 5-country distribution (UK ~88%, Germany, France, Spain, EIRE)

Schema: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`

---

## API Reference

Base URL: `YOUR_API_URL`

| Endpoint | Method | Query Params | Description |
|----------|--------|-------------|-------------|
| `/api/anomalies` | GET | `risk_level`, `limit` | Anomalous invoices |
| `/api/customers` | GET | `segment`, `limit` | Customer segments |
| `/api/products` | GET | `segment` | Product RFM data |
| `/api/summary` | GET | — | Aggregated KPIs |
| `/health` | GET | — | Health check |
| `/docs` | GET | — | Swagger UI |

---

## Roadmap

- [ ] Incremental ETL — process only new rows on each run
- [ ] Data Quality Engine — duplicate detection, validation reports
- [ ] Automated weekly email reports
- [ ] Docker Compose deployment
- [ ] Recommendation API — "customers who bought X also bought Y"

---

## License

MIT
