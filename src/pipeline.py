"""
BasketIQ — Unified Pipeline Module
=======================================
Single entry-point:  run_pipeline(source)

  source can be:
    - str  : path to CSV / Excel file
    - pd.DataFrame : already-loaded raw data

Returns a PipelineResult dict with all analytical outputs.
"""

import os
import io
import logging
import warnings
from typing import Union

import pandas as pd

from src.preprocessing import preprocess, load_csv
from src.features import engineer_features
from src.train import (
    train_anomaly_detector,
    train_segmentation,
    run_market_basket,
    run_temporal_analysis,
    cohort_retention_analysis,
    product_rfm_analysis,
)
from db.postgres import save_results

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _save(df: pd.DataFrame, name: str) -> None:
    """Save a DataFrame to the outputs directory."""
    path = os.path.join(OUTPUTS_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    logger.info("Saved → %s (%d rows)", path, len(df))


def _read_uploaded_file(source: io.BytesIO) -> pd.DataFrame:
    """
    Robustly read a CSV / Excel BytesIO upload.

    Handles:
      - Common encodings (utf-8, utf-8-bom, latin-1, cp1252)
      - Multiple delimiters (comma, semicolon, tab, pipe)
      - Quoted fields containing the delimiter character
      - Files with BOM preamble
      - Malformed / extra lines (skipped with on_bad_lines='skip')
    """
    raw_bytes = source.read()

    # ── Detect XLSX via ZIP magic bytes (PK\x03\x04) ────────────────────
    # .xlsx is a ZIP archive — checking bytes is more reliable than extension
    if raw_bytes[:4] == b"PK\x03\x04":
        try:
            df = pd.read_excel(io.BytesIO(raw_bytes))
            logger.info("Loaded as Excel — %d rows × %d cols", *df.shape)
            return df
        except Exception as e:
            raise ValueError(
                f"File looks like an Excel file but could not be read: {e}\n"
                "Make sure it is a valid .xlsx / .xls file and is not password-protected."
            ) from e

    # ── Detect encoding ──────────────────────────────────────────────────
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    text: str | None = None
    used_enc = "utf-8"
    for enc in encodings:
        try:
            text = raw_bytes.decode(enc)
            used_enc = enc
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        raise ValueError("Could not decode the uploaded file. Please save it as UTF-8 or Latin-1.")

    # ── Auto-detect delimiter ────────────────────────────────────────────
    # Inspect the first non-empty line to count candidate delimiters
    first_line = next((l for l in text.splitlines() if l.strip()), "")
    delimiter_candidates = [",", ";", "\t", "|"]
    sep = max(delimiter_candidates, key=lambda d: first_line.count(d))
    logger.info("Detected delimiter: %r  |  encoding: %s", sep, used_enc)

    # ── Parse CSV with increasing leniency ──────────────────────────────
    common_kwargs = dict(
        sep=sep,
        encoding=used_enc,
        low_memory=False,
        on_bad_lines="skip",   # skip any malformed rows instead of crashing
    )
    for engine in ("c", "python"):
        try:
            df = pd.read_csv(io.StringIO(text), engine=engine, **{k: v for k, v in common_kwargs.items() if k != "encoding"})
            if df.shape[1] > 1:          # must have more than 1 column to be valid
                logger.info("Parsed with engine=%s — %d rows × %d cols", engine, *df.shape)
                return df
        except Exception as e:
            logger.warning("engine=%s failed: %s", engine, e)
            continue

    # Last resort: python engine with no sep restriction
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python", on_bad_lines="skip")
    logger.info("Parsed with sep=None fallback — %d rows × %d cols", *df.shape)
    return df


def run_pipeline(
    source: Union[str, pd.DataFrame, io.BytesIO],
    n_clusters: int | None = None,
    min_support: float = 0.01,
    min_confidence: float = 0.2,
    contamination: float = 0.05,
    save_outputs: bool = True,
) -> dict:
    """
    End-to-end BasketIQ pipeline.

    Parameters
    ----------
    source          : file path, BytesIO (Streamlit upload), or DataFrame
    n_clusters      : force number of K-Means clusters (None = auto)
    min_support     : Apriori minimum support
    min_confidence  : Apriori minimum confidence
    contamination   : Isolation Forest contamination rate
    save_outputs    : whether to write CSVs to outputs/

    Returns
    -------
    dict with keys:
        raw_df, clean_df, invoice_df, rfm_df,
        anomalies, clusters, rules,
        peak_hours, monthly_trends, revenue_spikes,
        cohort_retention, product_rfm, stats
    """
    # ── Step 1 : Load ────────────────────────────────────────────────────
    logger.info("═══ STEP 1/5 : Loading data ═══")
    if isinstance(source, pd.DataFrame):
        raw_df = source.copy()
    elif isinstance(source, (str, os.PathLike)):
        raw_df = load_csv(str(source))
    else:
        # BytesIO from Streamlit uploader — use robust auto-detecting reader
        raw_df = _read_uploaded_file(source)
    logger.info("Loaded raw data — %d rows", len(raw_df))

    # ── Step 2 : Preprocessing ───────────────────────────────────────────
    logger.info("═══ STEP 2/5 : Preprocessing ═══")
    clean_df = preprocess(raw_df)

    # ── Step 3 : Feature Engineering ────────────────────────────────────
    logger.info("═══ STEP 3/5 : Feature Engineering ═══")
    clean_df, invoice_df, rfm_df = engineer_features(clean_df)

    # ── Step 4 : Model Training ──────────────────────────────────────────
    logger.info("═══ STEP 4/5 : Training models ═══")

    #  4A — Anomaly Detection
    scored_invoices, anomalies = train_anomaly_detector(invoice_df, contamination=contamination)

    #  4B — Customer Segmentation
    clusters = train_segmentation(rfm_df, n_clusters=n_clusters)

    #  4C — Market Basket
    rules = run_market_basket(clean_df, min_support=min_support, min_confidence=min_confidence)

    #  4D — Temporal
    temporal = run_temporal_analysis(clean_df)
    peak_hours      = temporal["peak_hours"]
    monthly_trends  = temporal["monthly_trends"]
    revenue_spikes  = temporal["revenue_spikes"]

    #  4E — Cohort Retention
    cohort_retention = cohort_retention_analysis(clean_df)

    #  4F — Product RFM
    product_rfm = product_rfm_analysis(clean_df)

    # ── Step 5 : Persist outputs ─────────────────────────────────────────
    logger.info("═══ STEP 5/5 : Saving outputs ═══")
    if save_outputs:
        _save(anomalies,      "anomalies")
        _save(clusters,       "clusters")
        if not rules.empty:
            _save(rules,      "rules")
        _save(peak_hours,     "peak_hours")
        _save(monthly_trends, "monthly_sales")
        _save(revenue_spikes, "revenue_spikes")
        if not cohort_retention.empty:
            path = os.path.join(OUTPUTS_DIR, "cohort_retention.csv")
            cohort_retention.to_csv(path)
            logger.info("Saved → %s (%d rows)", path, len(cohort_retention))
        if not product_rfm.empty:
            _save(product_rfm, "product_rfm")

    # ── Summary stats ────────────────────────────────────────────────────
    stats = {
        "total_transactions":  len(clean_df),
        "total_invoices":      clean_df["InvoiceNo"].nunique(),
        "total_customers":     clean_df["CustomerID"].nunique(),
        "total_revenue":       clean_df["TotalAmount"].sum(),
        "total_anomalies":     len(anomalies),
        "total_clusters":      clusters["Cluster"].nunique() if not clusters.empty else 0,
        "total_rules":         len(rules),
        "date_range":         (
            str(clean_df["InvoiceDate"].min().date()),
            str(clean_df["InvoiceDate"].max().date()),
        ),
        "countries":           clean_df["Country"].nunique() if "Country" in clean_df.columns else 1,
    }

    logger.info(
        "Pipeline complete ✅ | %d txns | %d anomalies | %d customers | %d rules",
        stats["total_transactions"], stats["total_anomalies"],
        stats["total_customers"], stats["total_rules"],
    )

    result = {
        "raw_df":          raw_df,
        "clean_df":        clean_df,
        "invoice_df":      invoice_df,       # original (no labels)
        "scored_invoices": scored_invoices,  # all invoices WITH AnomalyLabel/Score/RiskLevel
        "rfm_df":          rfm_df,
        "anomalies":       anomalies,
        "clusters":        clusters,
        "rules":           rules,
        "peak_hours":      peak_hours,
        "monthly_trends":  monthly_trends,
        "revenue_spikes":  revenue_spikes,
        "cohort_retention": cohort_retention,
        "product_rfm":     product_rfm,
        "stats":           stats,
    }

    save_results(result)

    return result