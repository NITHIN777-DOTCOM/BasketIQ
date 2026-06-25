"""
BasketIQ — PostgreSQL persistence for pipeline results.

Connects via DATABASE_URL (.env), auto-creates tables on startup, and
persists analytical outputs with graceful degradation when the DB is down.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

_engine: Engine | None = None
_db_available = False


class Base(DeclarativeBase):
    pass


class Anomaly(Base):
    __tablename__ = "anomalies"

    invoice_no: Mapped[str] = mapped_column(String(64), primary_key=True)
    invoice_total: Mapped[float | None] = mapped_column(Float)
    item_count: Mapped[int | None] = mapped_column(Integer)
    hour: Mapped[int | None] = mapped_column(Integer)
    customer_id: Mapped[str | None] = mapped_column(String(64))
    invoice_date: Mapped[datetime | None] = mapped_column(DateTime)
    anomaly_label: Mapped[int | None] = mapped_column(Integer)
    anomaly_score: Mapped[float | None] = mapped_column(Float)
    risk_level: Mapped[str | None] = mapped_column(String(32))


class CustomerSegment(Base):
    __tablename__ = "customer_segments"

    customer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    frequency: Mapped[int | None] = mapped_column(Integer)
    monetary: Mapped[float | None] = mapped_column(Float)
    recency: Mapped[int | None] = mapped_column(Integer)
    avg_spend: Mapped[float | None] = mapped_column(Float)
    r_score: Mapped[int | None] = mapped_column(Integer)
    f_score: Mapped[int | None] = mapped_column(Integer)
    m_score: Mapped[int | None] = mapped_column(Integer)
    rfm_score: Mapped[int | None] = mapped_column(Integer)
    cluster: Mapped[int | None] = mapped_column(Integer)
    cluster_name: Mapped[str | None] = mapped_column(String(128))


class AssociationRule(Base):
    __tablename__ = "association_rules"

    antecedents: Mapped[str] = mapped_column(Text, primary_key=True)
    consequents: Mapped[str] = mapped_column(Text, primary_key=True)
    antecedent_support: Mapped[float | None] = mapped_column(Float)
    consequent_support: Mapped[float | None] = mapped_column(Float)
    support: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    lift: Mapped[float | None] = mapped_column(Float)
    leverage: Mapped[float | None] = mapped_column(Float)
    conviction: Mapped[float | None] = mapped_column(Float)
    zhangs_metric: Mapped[float | None] = mapped_column(Float)
    rule_strength: Mapped[str | None] = mapped_column(String(32))


class MonthlyTrend(Base):
    __tablename__ = "monthly_trends"

    month_year: Mapped[str] = mapped_column(String(16), primary_key=True)
    total_revenue: Mapped[float | None] = mapped_column(Float)
    order_count: Mapped[int | None] = mapped_column(Integer)
    mom_change: Mapped[float | None] = mapped_column(Float)


class ProductRFM(Base):
    __tablename__ = "product_rfm"

    stock_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    recency: Mapped[int | None] = mapped_column(Integer)
    frequency: Mapped[int | None] = mapped_column(Integer)
    monetary: Mapped[float | None] = mapped_column(Float)
    r_score: Mapped[int | None] = mapped_column(Integer)
    f_score: Mapped[int | None] = mapped_column(Integer)
    m_score: Mapped[int | None] = mapped_column(Integer)
    rfm_score: Mapped[int | None] = mapped_column(Integer)
    segment: Mapped[str | None] = mapped_column(String(64))


ANOMALY_COL_MAP = {
    "InvoiceNo": "invoice_no",
    "InvoiceTotal": "invoice_total",
    "ItemCount": "item_count",
    "Hour": "hour",
    "CustomerID": "customer_id",
    "InvoiceDate": "invoice_date",
    "AnomalyLabel": "anomaly_label",
    "AnomalyScore": "anomaly_score",
    "RiskLevel": "risk_level",
}

SEGMENT_COL_MAP = {
    "CustomerID": "customer_id",
    "Frequency": "frequency",
    "Monetary": "monetary",
    "Recency": "recency",
    "AvgSpend": "avg_spend",
    "R_Score": "r_score",
    "F_Score": "f_score",
    "M_Score": "m_score",
    "RFM_Score": "rfm_score",
    "Cluster": "cluster",
    "ClusterName": "cluster_name",
}

RULE_COL_MAP = {
    "antecedents": "antecedents",
    "consequents": "consequents",
    "antecedent support": "antecedent_support",
    "consequent support": "consequent_support",
    "support": "support",
    "confidence": "confidence",
    "lift": "lift",
    "leverage": "leverage",
    "conviction": "conviction",
    "zhangs_metric": "zhangs_metric",
    "RuleStrength": "rule_strength",
}

TREND_COL_MAP = {
    "MonthYear": "month_year",
    "TotalRevenue": "total_revenue",
    "OrderCount": "order_count",
    "MoM_Change": "mom_change",
}

ANOMALY_COL_MAP_REV = {v: k for k, v in ANOMALY_COL_MAP.items()}
SEGMENT_COL_MAP_REV = {v: k for k, v in SEGMENT_COL_MAP.items()}
RULE_COL_MAP_REV = {v: k for k, v in RULE_COL_MAP.items()}
TREND_COL_MAP_REV = {v: k for k, v in TREND_COL_MAP.items()}


def _init_db() -> None:
    """Connect to PostgreSQL and ensure tables exist."""
    global _engine, _db_available

    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set; PostgreSQL persistence disabled.")
        return

    try:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        Base.metadata.create_all(_engine)
        _db_available = True
        logger.info("PostgreSQL connected; tables ensured.")
    except SQLAlchemyError as exc:
        logger.warning(
            "PostgreSQL unavailable (%s); continuing without DB persistence.",
            exc,
        )
        _engine = None
        _db_available = False


def _prepare_records(df: pd.DataFrame, col_map: dict[str, str]) -> list[dict[str, Any]]:
    """Rename pipeline columns to DB columns and normalize values for insert."""
    renamed = df.rename(columns=col_map)
    db_cols = list(col_map.values())
    subset = renamed[[c for c in db_cols if c in renamed.columns]].copy()

    for col in subset.columns:
        if pd.api.types.is_datetime64_any_dtype(subset[col]):
            subset[col] = pd.to_datetime(subset[col], errors="coerce")
        elif pd.api.types.is_categorical_dtype(subset[col]):
            subset[col] = subset[col].astype(str)

    if "invoice_no" in subset.columns:
        subset["invoice_no"] = subset["invoice_no"].astype(str)
    if "customer_id" in subset.columns:
        subset["customer_id"] = subset["customer_id"].astype(str)
    if "antecedents" in subset.columns:
        subset["antecedents"] = subset["antecedents"].astype(str)
    if "consequents" in subset.columns:
        subset["consequents"] = subset["consequents"].astype(str)
    if "month_year" in subset.columns:
        subset["month_year"] = subset["month_year"].astype(str)

    clean = subset.where(pd.notnull(subset), None)
    return clean.to_dict(orient="records")


def _upsert_dataframe(
    session: Session,
    model: type[Base],
    df: pd.DataFrame,
    col_map: dict[str, str],
    pk_cols: list[str],
) -> None:
    """Insert or update rows from a DataFrame using PostgreSQL ON CONFLICT."""
    if df is None or df.empty:
        return

    records = _prepare_records(df, col_map)
    if not records:
        return

    table = model.__table__
    stmt = pg_insert(table).values(records)
    update_cols = {
        col.name: stmt.excluded[col.name]
        for col in table.columns
        if col.name not in pk_cols
    }
    stmt = stmt.on_conflict_do_update(
        index_elements=pk_cols,
        set_=update_cols,
    )
    session.execute(stmt)


def _read_table(
    engine: Engine,
    model: type[Base],
    col_map_rev: dict[str, str],
) -> pd.DataFrame:
    """Load a table and rename columns back to pipeline naming."""
    with engine.connect() as conn:
        df = pd.read_sql(select(model), conn)
    if df.empty:
        return df
    return df.rename(columns=col_map_rev)


def save_results(pipeline_result: dict) -> None:
    """
    Insert or upsert pipeline analytical outputs into PostgreSQL.

    Expects pipeline_result keys: anomalies, clusters, rules, monthly_trends.
    """
    if not _db_available or _engine is None:
        return

    try:
        with Session(_engine) as session:
            _upsert_dataframe(
                session,
                Anomaly,
                pipeline_result.get("anomalies", pd.DataFrame()),
                ANOMALY_COL_MAP,
                ["invoice_no"],
            )
            _upsert_dataframe(
                session,
                CustomerSegment,
                pipeline_result.get("clusters", pd.DataFrame()),
                SEGMENT_COL_MAP,
                ["customer_id"],
            )
            _upsert_dataframe(
                session,
                AssociationRule,
                pipeline_result.get("rules", pd.DataFrame()),
                RULE_COL_MAP,
                ["antecedents", "consequents"],
            )
            _upsert_dataframe(
                session,
                MonthlyTrend,
                pipeline_result.get("monthly_trends", pd.DataFrame()),
                TREND_COL_MAP,
                ["month_year"],
            )
            session.commit()
        logger.info("Pipeline results saved to PostgreSQL.")
    except SQLAlchemyError as exc:
        logger.warning("Failed to save results to PostgreSQL: %s", exc)


def load_results() -> dict[str, pd.DataFrame]:
    """
    Load previously saved pipeline results from PostgreSQL.

    Returns
    -------
    dict with keys: anomalies, customer_segments, association_rules, monthly_trends
    """
    empty = {
        "anomalies": pd.DataFrame(),
        "customer_segments": pd.DataFrame(),
        "association_rules": pd.DataFrame(),
        "monthly_trends": pd.DataFrame(),
    }

    if not _db_available or _engine is None:
        return empty

    try:
        return {
            "anomalies": _read_table(_engine, Anomaly, ANOMALY_COL_MAP_REV),
            "customer_segments": _read_table(_engine, CustomerSegment, SEGMENT_COL_MAP_REV),
            "association_rules": _read_table(_engine, AssociationRule, RULE_COL_MAP_REV),
            "monthly_trends": _read_table(_engine, MonthlyTrend, TREND_COL_MAP_REV),
        }
    except SQLAlchemyError as exc:
        logger.warning("Failed to load results from PostgreSQL: %s", exc)
        return empty


_init_db()
