"""
BasketIQ — Prediction / Inference Module
=============================================
Load saved models and score new data on-demand.
"""

import os
import logging

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def score_anomaly(invoice_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load saved Isolation Forest and score a new invoice DataFrame.

    Parameters
    ----------
    invoice_df : DataFrame with columns InvoiceTotal, ItemCount, Hour

    Returns
    -------
    invoice_df with AnomalyLabel, AnomalyScore, RiskLevel appended
    """
    model_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Isolation Forest model not found — run the pipeline first.")

    bundle = joblib.load(model_path)
    model, scaler = bundle["model"], bundle["scaler"]

    features = ["InvoiceTotal", "ItemCount", "Hour"]
    X = invoice_df[features].fillna(0)
    X_scaled = scaler.transform(X)

    result = invoice_df.copy()
    result["AnomalyLabel"] = model.predict(X_scaled)
    result["AnomalyScore"] = -model.score_samples(X_scaled)

    s_min, s_max = result["AnomalyScore"].min(), result["AnomalyScore"].max()
    if s_max > s_min:
        result["AnomalyScore"] = (result["AnomalyScore"] - s_min) / (s_max - s_min)

    result["RiskLevel"] = pd.cut(
        result["AnomalyScore"],
        bins=[0, 0.4, 0.7, 1.01],
        labels=["Low", "Medium", "High"],
    )
    return result


def predict_cluster(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign cluster labels to new customers using the saved K-Means model.

    Parameters
    ----------
    rfm_df : DataFrame with columns Recency, Frequency, Monetary, AvgSpend

    Returns
    -------
    rfm_df with Cluster column appended
    """
    model_path = os.path.join(MODELS_DIR, "kmeans.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("K-Means model not found — run the pipeline first.")

    bundle = joblib.load(model_path)
    model, scaler = bundle["model"], bundle["scaler"]

    features = ["Recency", "Frequency", "Monetary", "AvgSpend"]
    X = rfm_df[features].fillna(0)
    X_scaled = scaler.transform(X)

    result = rfm_df.copy()
    result["Cluster"] = model.predict(X_scaled)
    return result
