"""
BasketIQ — Feature Engineering Module
==========================================
Generates:
  - TotalAmount per line item
  - Invoice-level aggregates (InvoiceTotal, ItemCount)
  - Customer-level RFM metrics (Recency, Frequency, Monetary, AvgSpend)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def add_total_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Add TotalAmount = Quantity × UnitPrice column."""
    df = df.copy()
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_invoice_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to invoice level.

    Returns
    -------
    pd.DataFrame  indexed by InvoiceNo with columns:
        InvoiceTotal, ItemCount, Hour, CustomerID, InvoiceDate
    """
    invoice_df = (
        df.groupby("InvoiceNo")
        .agg(
            InvoiceTotal=("TotalAmount", "sum"),
            ItemCount=("Quantity", "sum"),
            Hour=("Hour", "first"),
            CustomerID=("CustomerID", "first"),
            InvoiceDate=("InvoiceDate", "first"),
        )
        .reset_index()
    )
    return invoice_df


def build_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary, AvgSpend) per customer.

    Parameters
    ----------
    df : cleaned DataFrame with TotalAmount column
    snapshot_date : reference date; defaults to max(InvoiceDate) + 1 day

    Returns
    -------
    pd.DataFrame with CustomerID, Recency, Frequency, Monetary, AvgSpend
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            LastPurchase=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalAmount", "sum"),
        )
        .reset_index()
    )
    rfm["Recency"] = (snapshot_date - rfm["LastPurchase"]).dt.days
    rfm["AvgSpend"] = rfm["Monetary"] / rfm["Frequency"]
    rfm.drop(columns=["LastPurchase"], inplace=True)

    # RFM scores (quintiles, 1–5)
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop")
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(int)
        + rfm["F_Score"].astype(int)
        + rfm["M_Score"].astype(int)
    )

    logger.info("RFM built — %d customers", len(rfm))
    return rfm


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Master feature engineering function.

    Parameters
    ----------
    df : preprocessed DataFrame

    Returns
    -------
    (enriched_df, invoice_df, rfm_df)
    """
    df = add_total_amount(df)
    invoice_df = build_invoice_features(df)
    rfm_df = build_rfm(df)
    logger.info(
        "Feature engineering done — %d transactions | %d invoices | %d customers",
        len(df), len(invoice_df), len(rfm_df),
    )
    return df, invoice_df, rfm_df
