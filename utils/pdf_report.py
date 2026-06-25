"""
BasketIQ — PDF report generator (fpdf2).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from fpdf import FPDF


def _safe(text: object, max_len: int | None = None) -> str:
    """Encode text for core PDF fonts and optionally truncate."""
    s = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    if max_len and len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s.encode("latin-1", "replace").decode("latin-1")


def _fmt_currency(value: float) -> str:
    return f"GBP {value:,.2f}"


def _fmt_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    return f"{value:.1f}%"


class _ReportPDF(FPDF):
    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def _section_title(pdf: FPDF, title: str) -> None:
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 120, 100)
    pdf.cell(0, 10, _safe(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _body_text(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, _safe(text))
    pdf.ln(1)


def _add_table(pdf: FPDF, headers: list[str], rows: list[list[str]], col_widths: list[int]) -> None:
    line_h = 7

    def _draw_header() -> None:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(230, 245, 242)
        for header, width in zip(headers, col_widths):
            pdf.cell(width, line_h, _safe(header), border=1, fill=True)
        pdf.ln()

    _draw_header()
    pdf.set_font("Helvetica", "", 9)

    for row in rows:
        if pdf.get_y() > 275:
            pdf.add_page()
            _draw_header()
            pdf.set_font("Helvetica", "", 9)

        for value, width in zip(row, col_widths):
            max_chars = max(int(width / 2.2), 8)
            pdf.cell(width, line_h, _safe(value, max_chars), border=1)
        pdf.ln()


def generate_pdf_report(results: dict) -> bytes:
    """
    Build a multi-section BasketIQ intelligence report.

    Parameters
    ----------
    results : pipeline result dict from run_pipeline()

    Returns
    -------
    PDF file contents as bytes.
    """
    stats = results["stats"]
    anomalies = results["anomalies"]
    clusters = results["clusters"]
    rules = results.get("rules", pd.DataFrame())
    monthly = results["monthly_trends"]
    product_rfm = results.get("product_rfm", pd.DataFrame())

    if product_rfm.empty and "clean_df" in results:
        from src.train import product_rfm_analysis
        product_rfm = product_rfm_analysis(results["clean_df"])

    pdf = _ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)

    # ── Cover page ───────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(0, 120, 100)
    pdf.ln(40)
    pdf.cell(0, 14, "BasketIQ", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 10, "Retail Intelligence Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(
        0,
        8,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(16)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Dataset Overview", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    date_range = stats.get("date_range", ("N/A", "N/A"))
    cover_lines = [
        f"Total Transactions: {stats.get('total_transactions', 0):,}",
        f"Total Customers: {stats.get('total_customers', 0):,}",
        f"Total Invoices: {stats.get('total_invoices', 0):,}",
        f"Date Range: {date_range[0]} to {date_range[1]}",
        f"Total Revenue: {_fmt_currency(stats.get('total_revenue', 0))}",
    ]
    for line in cover_lines:
        pdf.cell(0, 7, line, align="C", new_x="LMARGIN", new_y="NEXT")

    # ── Anomaly summary ──────────────────────────────────────────────────
    pdf.add_page()
    _section_title(pdf, "Anomaly Summary")
    risk_counts = anomalies["RiskLevel"].value_counts() if not anomalies.empty else pd.Series(dtype=int)
    _body_text(
        pdf,
        f"Total Anomalies: {len(anomalies):,}\n"
        f"High Risk: {int(risk_counts.get('High', 0)):,}\n"
        f"Medium Risk: {int(risk_counts.get('Medium', 0)):,}\n"
        f"Low Risk: {int(risk_counts.get('Low', 0)):,}",
    )
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Top 5 Anomalous Invoices", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    top_anom = (
        anomalies.sort_values("AnomalyScore", ascending=False).head(5)
        if not anomalies.empty
        else pd.DataFrame()
    )
    if top_anom.empty:
        _body_text(pdf, "No anomalies detected.")
    else:
        anom_rows = []
        for _, row in top_anom.iterrows():
            anom_rows.append([
                str(row.get("InvoiceNo", "")),
                _fmt_currency(float(row.get("InvoiceTotal", 0))),
                f"{float(row.get('AnomalyScore', 0)):.3f}",
                str(row.get("RiskLevel", "")),
            ])
        _add_table(
            pdf,
            ["Invoice", "Total", "Score", "Risk"],
            anom_rows,
            [35, 45, 30, 35],
        )

    # ── Customer segments ────────────────────────────────────────────────
    pdf.add_page()
    _section_title(pdf, "Customer Segments")
    if clusters.empty:
        _body_text(pdf, "No customer segments available.")
    else:
        seg_summary = (
            clusters.groupby("ClusterName", as_index=False)
            .agg(Customers=("CustomerID", "count"), Avg_Monetary=("Monetary", "mean"))
            .sort_values("Customers", ascending=False)
        )
        seg_rows = [
            [
                str(r["ClusterName"]),
                f"{int(r['Customers']):,}",
                _fmt_currency(float(r["Avg_Monetary"])),
            ]
            for _, r in seg_summary.iterrows()
        ]
        _add_table(pdf, ["Segment", "Customers", "Avg Monetary"], seg_rows, [70, 35, 50])

    # ── Association rules ────────────────────────────────────────────────
    pdf.add_page()
    _section_title(pdf, "Top 10 Association Rules")
    if rules is None or rules.empty:
        _body_text(pdf, "No rules found.")
    else:
        top_rules = rules.sort_values("lift", ascending=False).head(10)
        rule_rows = []
        for _, row in top_rules.iterrows():
            rule_rows.append([
                _safe(row.get("antecedents", ""), 45),
                _safe(row.get("consequents", ""), 45),
                f"{float(row.get('lift', 0)):.2f}",
            ])
        _add_table(
            pdf,
            ["Antecedents", "Consequents", "Lift"],
            rule_rows,
            [75, 75, 25],
        )

    # ── Monthly revenue ──────────────────────────────────────────────────
    pdf.add_page()
    _section_title(pdf, "Monthly Revenue")
    if monthly.empty:
        _body_text(pdf, "No monthly trend data available.")
    else:
        month_rows = []
        for _, row in monthly.iterrows():
            month_rows.append([
                str(row.get("MonthYear", "")),
                _fmt_currency(float(row.get("TotalRevenue", 0))),
                _fmt_pct(row.get("MoM_Change")),
            ])
        _add_table(pdf, ["Month", "Revenue", "MoM Change"], month_rows, [40, 55, 40])

    # ── Product intelligence ─────────────────────────────────────────────
    pdf.add_page()
    _section_title(pdf, "Product Intelligence — Top 10 Star Products")
    if product_rfm.empty:
        _body_text(pdf, "No product RFM data available.")
    else:
        stars = (
            product_rfm[product_rfm["Segment"] == "Star Product"]
            .sort_values("Monetary", ascending=False)
            .head(10)
        )
        if stars.empty:
            _body_text(pdf, "No star products identified.")
        else:
            star_rows = []
            for _, row in stars.iterrows():
                star_rows.append([
                    str(row.get("StockCode", "")),
                    _safe(row.get("Description", ""), 50),
                    _fmt_currency(float(row.get("Monetary", 0))),
                ])
            _add_table(pdf, ["Stock Code", "Description", "Revenue"], star_rows, [30, 95, 40])

    return bytes(pdf.output())
