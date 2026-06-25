"""
BasketIQ API — FastAPI server for analytical insights.

Provides RESTful endpoints to query PostgreSQL tables with
anomalies, customer segments, product RFM, and key metrics.
"""

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.postgres import (
    Anomaly,
    CustomerSegment,
    ProductRFM,
    _engine,
    _db_available,
)

app = FastAPI(
    title="BasketIQ API",
    description="Analytics API for BasketIQ business intelligence",
    version="1.0.0",
)

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================
# Pydantic Response Models
# =====================


class AnomalyResponse(BaseModel):
    invoice_no: str
    invoice_total: Optional[float] = None
    item_count: Optional[int] = None
    hour: Optional[int] = None
    customer_id: Optional[str] = None
    invoice_date: Optional[datetime] = None
    anomaly_label: Optional[int] = None
    anomaly_score: Optional[float] = None
    risk_level: Optional[str] = None

    class Config:
        from_attributes = True


class CustomerSegmentResponse(BaseModel):
    customer_id: str
    frequency: Optional[int] = None
    monetary: Optional[float] = None
    recency: Optional[int] = None
    avg_spend: Optional[float] = None
    r_score: Optional[int] = None
    f_score: Optional[int] = None
    m_score: Optional[int] = None
    rfm_score: Optional[int] = None
    cluster: Optional[int] = None
    cluster_name: Optional[str] = None

    class Config:
        from_attributes = True


class ProductRFMResponse(BaseModel):
    stock_code: str
    recency: Optional[int] = None
    frequency: Optional[int] = None
    monetary: Optional[float] = None
    r_score: Optional[int] = None
    f_score: Optional[int] = None
    m_score: Optional[int] = None
    rfm_score: Optional[int] = None
    segment: Optional[str] = None

    class Config:
        from_attributes = True


class SummaryResponse(BaseModel):
    total_customers: int
    total_anomalies: int
    high_risk_count: int
    total_products: int
    star_products_count: int
    avg_rfm_score: Optional[float] = None

    class Config:
        from_attributes = True


# =====================
# Endpoints
# =====================


@app.get("/api/anomalies", response_model=list[AnomalyResponse])
async def get_anomalies(
    risk_level: Optional[str] = Query(None, description="Filter by risk level: High/Medium/Low"),
    limit: int = Query(100, ge=1, le=1000, description="Max results to return"),
) -> list[AnomalyResponse]:
    """
    Returns all anomalous invoices from the anomalies table.

    Query Parameters:
    - risk_level: Optional filter by High/Medium/Low
    - limit: Max results (default 100, max 1000)
    """
    if not _db_available or _engine is None:
        return []

    try:
        with Session(_engine) as session:
            query = select(Anomaly)

            if risk_level:
                query = query.where(Anomaly.risk_level == risk_level)

            query = query.limit(limit)
            results = session.execute(query).scalars().all()

            return [AnomalyResponse.from_orm(r) for r in results]
    except Exception as e:
        print(f"Error fetching anomalies: {e}")
        return []


@app.get("/api/customers", response_model=list[CustomerSegmentResponse])
async def get_customers(
    segment: Optional[str] = Query(None, description="Filter by cluster name"),
    limit: int = Query(100, ge=1, le=1000, description="Max results to return"),
) -> list[CustomerSegmentResponse]:
    """
    Returns customer segments from the customer_segments table.

    Query Parameters:
    - segment: Optional filter by cluster name
    - limit: Max results (default 100, max 1000)
    """
    if not _db_available or _engine is None:
        return []

    try:
        with Session(_engine) as session:
            query = select(CustomerSegment)

            if segment:
                query = query.where(CustomerSegment.cluster_name == segment)

            query = query.limit(limit)
            results = session.execute(query).scalars().all()

            return [CustomerSegmentResponse.from_orm(r) for r in results]
    except Exception as e:
        print(f"Error fetching customers: {e}")
        return []


@app.get("/api/products", response_model=list[ProductRFMResponse])
async def get_products(
    segment: Optional[str] = Query(
        None,
        description="Filter by segment: Star/Reliable/Declining/Niche/Low Performer",
    ),
    limit: int = Query(100, ge=1, le=1000, description="Max results to return"),
) -> list[ProductRFMResponse]:
    """
    Returns product RFM data from the product_rfm table.

    Query Parameters:
    - segment: Optional filter by Star/Reliable/Declining/Niche/Low Performer
    - limit: Max results (default 100, max 1000)
    """
    if not _db_available or _engine is None:
        return []

    try:
        with Session(_engine) as session:
            query = select(ProductRFM)

            if segment:
                query = query.where(ProductRFM.segment == segment)

            query = query.limit(limit)
            results = session.execute(query).scalars().all()

            return [ProductRFMResponse.from_orm(r) for r in results]
    except Exception as e:
        print(f"Error fetching products: {e}")
        return []


@app.get("/api/summary", response_model=SummaryResponse)
async def get_summary() -> SummaryResponse:
    """
    Returns key business metrics in one call.

    Metrics:
    - total_customers: Count of unique customers
    - total_anomalies: Count of all anomalies
    - high_risk_count: Count of anomalies with risk_level='High'
    - total_products: Count of unique products
    - star_products_count: Count of products with segment='Star'
    - avg_rfm_score: Average RFM score across all products
    """
    if not _db_available or _engine is None:
        return SummaryResponse(
            total_customers=0,
            total_anomalies=0,
            high_risk_count=0,
            total_products=0,
            star_products_count=0,
            avg_rfm_score=None,
        )

    try:
        with Session(_engine) as session:
            # Total customers
            total_customers = session.execute(
                select(func.count()).select_from(CustomerSegment)
            ).scalar() or 0

            # Total anomalies
            total_anomalies = session.execute(
                select(func.count()).select_from(Anomaly)
            ).scalar() or 0

            # High risk anomalies
            high_risk_count = session.execute(
                select(func.count()).select_from(Anomaly).where(Anomaly.risk_level == "High")
            ).scalar() or 0

            # Total products
            total_products = session.execute(
                select(func.count()).select_from(ProductRFM)
            ).scalar() or 0

            # Star products count
            star_products_count = session.execute(
                select(func.count()).select_from(ProductRFM).where(ProductRFM.segment == "Star")
            ).scalar() or 0

            # Average RFM score
            avg_rfm_score = session.execute(
                select(func.avg(ProductRFM.rfm_score)).select_from(ProductRFM)
            ).scalar()

            return SummaryResponse(
                total_customers=total_customers,
                total_anomalies=total_anomalies,
                high_risk_count=high_risk_count,
                total_products=total_products,
                star_products_count=star_products_count,
                avg_rfm_score=avg_rfm_score,
            )
    except Exception as e:
        print(f"Error fetching summary: {e}")
        return SummaryResponse(
            total_customers=0,
            total_anomalies=0,
            high_risk_count=0,
            total_products=0,
            star_products_count=0,
            avg_rfm_score=None,
        )


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "db_available": _db_available,
    }
