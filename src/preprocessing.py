"""
BasketIQ — Preprocessing Module
===================================
Cleans raw supermarket transaction data:
  - Normalizes column names
  - Removes cancelled invoices (InvoiceNo starts with 'C')
  - Filters out negative / zero quantities and prices
  - Drops rows with missing CustomerID
  - Parses InvoiceDate → datetime and extracts temporal features
  - Drops duplicates
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Column aliases ──────────────────────────────────────────────────────────
# Maps common alternative spellings → canonical names
COLUMN_ALIASES: dict[str, str] = {
    "invoice_no":       "InvoiceNo",
    "invoiceno":        "InvoiceNo",
    "invoice":          "InvoiceNo",
    "stock_code":       "StockCode",
    "stockcode":        "StockCode",
    "item_code":        "StockCode",
    "description":      "Description",
    "product":          "Description",
    "product_name":     "Description",
    "qty":              "Quantity",
    "quantity":         "Quantity",
    "invoice_date":     "InvoiceDate",
    "invoicedate":      "InvoiceDate",
    "date":             "InvoiceDate",
    "unit_price":       "UnitPrice",
    "unitprice":        "UnitPrice",
    "price":            "UnitPrice",
    "customer_id":      "CustomerID",
    "customerid":       "CustomerID",
    "customer":         "CustomerID",
    "country":          "Country",
}

REQUIRED_COLS = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and map column names to canonical form."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    rename_map = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "_").replace("-", "_")
        if lower in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[lower]
    df.rename(columns=rename_map, inplace=True)
    return df


def _check_required(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Detected columns: {list(df.columns)}"
        )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline on a raw transaction DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction data (as loaded from CSV).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with temporal columns added.
    """
    logger.info("Starting preprocessing — %d rows, %d cols", *df.shape)

    # 1. Normalize column names
    df = _normalize_columns(df)
    _check_required(df)

    original_len = len(df)

    # 2. Remove cancelled invoices
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")].copy()
    logger.info("After removing cancellations: %d rows", len(df))

    # 3. Filter invalid quantities and prices
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    logger.info("After filtering negatives: %d rows", len(df))

    # 4. Drop missing CustomerID
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(str).str.strip()
    logger.info("After dropping missing CustomerID: %d rows", len(df))

    # 5. Parse InvoiceDate
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], infer_datetime_format=True, errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # 6. Extract temporal features
    df["Hour"]  = df["InvoiceDate"].dt.hour
    df["Day"]   = df["InvoiceDate"].dt.day
    df["Month"] = df["InvoiceDate"].dt.month
    df["Year"]  = df["InvoiceDate"].dt.year
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()

    # 7. Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows", before - len(df))

    # 8. Reset index
    df = df.reset_index(drop=True)

    logger.info(
        "Preprocessing complete — %d → %d rows (%.1f%% retained)",
        original_len, len(df), 100 * len(df) / original_len if original_len else 0,
    )
    return df


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV or Excel file robustly:
      - Auto-detects encoding (utf-8-sig, utf-8, latin-1, cp1252)
      - Auto-detects delimiter (comma, semicolon, tab, pipe)
      - Skips malformed lines instead of crashing
    """
    if filepath.endswith((".xlsx", ".xls")):
        return pd.read_excel(filepath)

    # Detect encoding
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    text: str | None = None
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                text = f.read()
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        raise ValueError(f"Could not decode file: {filepath}")

    # Detect delimiter from first non-empty line
    first_line = next((l for l in text.splitlines() if l.strip()), "")
    sep = max([",", ";", "\t", "|"], key=lambda d: first_line.count(d))
    logger.info("load_csv: detected delimiter=%r for %s", sep, filepath)

    for engine in ("c", "python"):
        try:
            df = pd.read_csv(
                filepath, sep=sep, engine=engine,
                low_memory=False, on_bad_lines="skip",
                encoding="utf-8-sig",
            )
            if df.shape[1] > 1:
                return df
        except Exception:
            continue

    # Final fallback
    return pd.read_csv(filepath, sep=None, engine="python", on_bad_lines="skip")
