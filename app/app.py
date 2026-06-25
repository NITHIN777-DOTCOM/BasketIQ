"""
BasketIQ — Streamlit Dashboard
===================================
Run:  streamlit run app/app.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pipeline import run_pipeline
from utils.pdf_report import generate_pdf_report

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG & THEME
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BasketIQ — Supermarket Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

TEAL      = "#00C9A7"
TEAL_DARK = "#009E84"
BG_CARD   = "rgba(255,255,255,0.04)"
PLOTLY_TEMPLATE = "plotly_dark"
CHART_BGCOLOR   = "rgba(0,0,0,0)"
PAPER_BGCOLOR   = "rgba(0,0,0,0)"

st.markdown("""
<style>
/* ── Root / App background ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1020 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050c1a 0%, #0b1528 100%);
    border-right: 1px solid rgba(0,201,167,0.15);
}
[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar brand ── */
.sidebar-brand {
    text-align: center;
    padding: 1.5rem 0 1rem;
}
.sidebar-brand h1 {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00C9A7, #00a3ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.sidebar-brand p {
    color: #64748b;
    font-size: 0.75rem;
    margin: 0.25rem 0 0;
}

/* ── Nav pills ── */
div[data-testid="stRadio"] label {
    display: block;
    padding: 0.55rem 1rem;
    border-radius: 8px;
    margin: 2px 0;
    transition: all 0.2s;
    cursor: pointer;
    font-size: 0.9rem;
}
div[data-testid="stRadio"] label:hover {
    background: rgba(0,201,167,0.12);
    color: #00C9A7;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(0,201,167,0.2);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,201,167,0.15);
}
[data-testid="stMetricValue"] { color: #00C9A7 !important; font-weight: 800; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
[data-testid="stMetricDelta"] { color: #38bdf8 !important; }

/* ── Section headings ── */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f1f5f9;
    border-left: 4px solid #00C9A7;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}

/* ── Badge / pill ── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.badge-teal  { background: rgba(0,201,167,0.15); color: #00C9A7; border: 1px solid rgba(0,201,167,0.3); }
.badge-blue  { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.3); }
.badge-red   { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3);  }
.badge-amber { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,201,167,0.4) !important;
    border-radius: 14px !important;
    background: rgba(0,201,167,0.04) !important;
    padding: 2rem !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div { background-color: #00C9A7 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,201,167,0.3); border-radius: 9px; }

/* ── Tab styling ── */
[data-testid="stTab"] { color: #94a3b8 !important; }
[aria-selected="true"] { color: #00C9A7 !important; border-bottom-color: #00C9A7 !important; }

/* download button */
[data-testid="stDownloadButton"] > button {
    background: rgba(0,201,167,0.15);
    border: 1px solid rgba(0,201,167,0.4);
    color: #00C9A7;
    border-radius: 8px;
    transition: all 0.2s;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(0,201,167,0.3);
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def _chart_layout(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#cbd5e1")),
        template=PLOTLY_TEMPLATE,
        plot_bgcolor=CHART_BGCOLOR,
        paper_bgcolor=PAPER_BGCOLOR,
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    )
    return fig


def _color_risk(val):
    if val == "High":
        return "background-color: rgba(239,68,68,0.25); color: #f87171"
    elif val == "Medium":
        return "background-color: rgba(251,191,36,0.2); color: #fbbf24"
    return ""


def _csv_download(df: pd.DataFrame, filename: str, label: str = "⬇️ Download CSV"):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode(),
        file_name=filename,
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

st.sidebar.markdown("""
<div class="sidebar-brand">
  <h1>🛒 BasketIQ</h1>
  <p>Supermarket Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

PAGES = [
    "🏠  Overview",
    "⚠️  Anomaly Detection",
    "👥  Customer Segmentation",
    "🛒  Market Basket",
    "📅  Temporal Analysis",
    "📊  Cohort Retention",
    "🏷️  Product Intelligence",
    "🤖  Model Info",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Pipeline Settings", expanded=False):
    n_clusters_ui = st.slider(
        "K-Means Clusters",
        2, 10, 4, 1,
        help="Number of customer groups to discover. 4 is a good default. Increase for more granular segments, decrease for broader ones.",
    )
    contamination_ui = st.slider(
        "Anomaly Contamination %",
        1, 20, 5, 1,
        help="Expected % of suspicious transactions. 5% means the system flags the top 5% most unusual invoices. Increase if you want more flags, decrease for stricter detection.",
    )
    min_support_ui = st.slider(
        "Apriori Min Support %",
        1, 10, 2, 1,
        help="How often two products must appear together to count as a pattern. Lower = more rules found but weaker patterns. Higher = fewer but stronger patterns.",
    ) / 100
    min_conf_ui = st.slider(
        "Apriori Min Confidence %",
        10, 80, 25, 5,
        help="How reliable a product association must be. 25% means: when product A is bought, product B appears in at least 25% of those baskets.",
    ) / 100

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.75rem;'>BasketIQ v1.0 | Built with ❤️</div>",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════
# DATA LOADING / PIPELINE
# ════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state["results"] = None
if "pipeline_settings" not in st.session_state:
    st.session_state["pipeline_settings"] = {}


def _load_source(uploaded_file) -> "pd.DataFrame | io.BytesIO":
    """
    Convert a Streamlit UploadedFile to the right source type.
    Excel files (.xlsx/.xls) are read immediately into a DataFrame.
    Everything else is passed as BytesIO so the pipeline handles it.
    """
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    # Excel: check extension OR ZIP magic bytes (PK\x03\x04)
    is_xlsx = name.endswith((".xlsx", ".xls")) or raw[:4] == b"PK\x03\x04"
    if is_xlsx:
        try:
            return pd.read_excel(io.BytesIO(raw))
        except Exception as e:
            st.warning(f"Excel read failed ({e}), trying CSV fallback…")

    return io.BytesIO(raw)


def _run(uploaded_file, settings):
    if (
        st.session_state["results"] is not None
        and st.session_state["pipeline_settings"] == settings
    ):
        return st.session_state["results"]

    progress = st.progress(0, text="⚡ Starting pipeline...")
    try:
        progress.progress(10, "🧹 Preprocessing data...")
        source = _load_source(uploaded_file)
        results = run_pipeline(
            source,
            n_clusters=settings["n_clusters"],
            contamination=settings["contamination"],
            min_support=settings["min_support"],
            min_confidence=settings["min_confidence"],
            save_outputs=True,
        )
        progress.progress(100, "✅ Pipeline complete!")
        st.session_state["results"] = results
        st.session_state["pipeline_settings"] = settings
        return results
    except Exception as e:
        progress.empty()
        st.error(f"**Pipeline error:** {e}")
        st.markdown("""
**Common causes and fixes:**
- 📊 **Excel file** — `.xlsx` files are supported directly. Make sure the file isn't password-protected or corrupted.
- 🔤 **Wrong delimiter** — the system auto-detects `,` `;` `\\t` `|` for CSV files.
- 🗂️ **Missing required columns** — needed: `InvoiceNo`, `StockCode`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`.
- 📆 **Bad date format** — `InvoiceDate` must be a recognisable date/datetime string.
- 🔢 **All rows removed** — ensure `Quantity > 0`, `UnitPrice > 0`, and `CustomerID` is present.
        """)
        return None


# ════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════════

# ── Upload banner (always visible when no data loaded) ──────────
if st.session_state["results"] is None:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem 2rem;">
        <div style="font-size:4rem; margin-bottom:1rem;">🛒</div>
        <h1 style="font-size:2.4rem; font-weight:900;
                   background:linear-gradient(90deg,#00C9A7,#00a3ff);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            BasketIQ
        </h1>
        <p style="color:#64748b;font-size:1.1rem;max-width:540px;margin:0.5rem auto 2rem;">
            Upload your raw supermarket transaction CSV and get instant
            AI-powered insights — anomalies, segments, baskets & trends.
        </p>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop your transaction CSV here (InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID …)",
        type=["csv", "xlsx"],
        key="main_uploader",
    )
    if uploaded:
        settings = dict(
            n_clusters=n_clusters_ui,
            contamination=contamination_ui / 100,
            min_support=min_support_ui,
            min_confidence=min_conf_ui,
        )
        results = _run(uploaded, settings)
        if results:
            st.success("🎉 Analysis complete! Navigate using the sidebar.")
            st.rerun()
    st.stop()

# ── Sidebar re-run button ────────────────────────────────────────
with st.sidebar:
    new_upload = st.file_uploader("Re-upload / New Dataset", type=["csv","xlsx"], key="side_uploader")
    if new_upload:
        settings = dict(
            n_clusters=n_clusters_ui,
            contamination=contamination_ui / 100,
            min_support=min_support_ui,
            min_confidence=min_conf_ui,
        )
        _run(new_upload, settings)
        st.rerun()
    if st.button("🔄 Re-run with new settings"):
        st.session_state["pipeline_settings"] = {}
        st.session_state["results"] = None
        st.rerun()

# Results are available from here on
R = st.session_state["results"]

# ── Backward-compatibility: patch scored_invoices if cached results are stale ──
if "scored_invoices" not in R:
    _inv = R["invoice_df"].copy()
    _anom_cols = R["anomalies"][["InvoiceNo", "AnomalyLabel", "AnomalyScore", "RiskLevel"]]
    _inv = _inv.merge(_anom_cols, on="InvoiceNo", how="left")
    _inv["AnomalyLabel"] = _inv["AnomalyLabel"].fillna(1).astype(int)
    _inv["AnomalyScore"] = _inv["AnomalyScore"].fillna(0.0)
    _inv["RiskLevel"]    = _inv["RiskLevel"].fillna("Low")
    R["scored_invoices"] = _inv
    st.session_state["results"] = R

stats       = R["stats"]
anomalies   = R["anomalies"]
clusters    = R["clusters"]
rules       = R["rules"]
peak_hours  = R["peak_hours"]
monthly     = R["monthly_trends"]
spikes      = R["revenue_spikes"]
clean_df    = R["clean_df"]

if "cohort_retention" not in R:
    from src.train import cohort_retention_analysis
    R["cohort_retention"] = cohort_retention_analysis(clean_df)
    st.session_state["results"] = R

cohort_retention = R["cohort_retention"]

if "product_rfm" not in R:
    from src.train import product_rfm_analysis
    R["product_rfm"] = product_rfm_analysis(clean_df)
    st.session_state["results"] = R

product_rfm = R["product_rfm"]

SEGMENT_COLORS = {
    "Star Product": "#00C9A7",
    "Reliable": "#38bdf8",
    "Declining": "#fbbf24",
    "Niche": "#a78bfa",
    "Low Performer": "#64748b",
}

with st.sidebar:
    st.markdown("---")
    st.download_button(
        label="📄 Export PDF Report",
        data=generate_pdf_report(R),
        file_name="basketiq_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown('<div class="section-header">📊 Platform Overview</div>', unsafe_allow_html=True)

    # ── KPI row 1
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("💰 Total Revenue", f"£{stats['total_revenue']:,.0f}")
    c2.metric("📦 Transactions", f"{stats['total_transactions']:,}")
    c3.metric("🧾 Invoices", f"{stats['total_invoices']:,}")
    c4.metric("🌍 Countries", stats["countries"])

    # ── KPI row 2
    c5,c6,c7,c8 = st.columns(4)
    c5.metric("👥 Customers", f"{stats['total_customers']:,}")
    c6.metric("⚠️ Anomalies", f"{stats['total_anomalies']:,}",
              delta=f"{100*stats['total_anomalies']/max(stats['total_invoices'],1):.1f}%", delta_color="inverse")
    c7.metric("🔗 MBA Rules", f"{stats['total_rules']:,}")
    c8.metric("📅 Data Range",
              f"{stats['date_range'][0]} → {stats['date_range'][1]}".replace("-","/"))

    st.markdown("---")
    col_l, col_r = st.columns(2)

    # Peak hours
    with col_l:
        colors = [TEAL if p else "#1e3a5f" for p in peak_hours["IsPeak"]]
        fig = go.Figure(go.Bar(
            x=peak_hours["Hour"],
            y=peak_hours["TotalRevenue"],
            marker_color=colors,
            hovertemplate="Hour %{x}:00<br>Revenue: £%{y:,.0f}<extra></extra>",
        ))
        fig = _chart_layout(fig, "🕐 Peak Shopping Hours", 360)
        fig.update_xaxes(title="Hour of Day", dtick=2)
        fig.update_yaxes(title="Revenue (£)")
        st.plotly_chart(fig, use_container_width=True)

    # Cluster pie
    with col_r:
        cluster_counts = clusters.groupby("ClusterName").size().reset_index(name="Count")
        fig = px.pie(
            cluster_counts, names="ClusterName", values="Count",
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.45,
        )
        fig = _chart_layout(fig, "👥 Customer Segments Distribution", 360)
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    # Monthly revenue
    st.markdown('<div class="section-header">📈 Monthly Revenue Trend</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["MonthYear"], y=monthly["TotalRevenue"],
        mode="lines+markers",
        line=dict(color=TEAL, width=2.5),
        marker=dict(size=7, color=TEAL),
        fill="tozeroy",
        fillcolor="rgba(0,201,167,0.08)",
        name="Revenue",
        hovertemplate="%{x}<br>£%{y:,.0f}<extra></extra>",
    ))
    fig = _chart_layout(fig, "", 340)
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Revenue (£)")
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 2 — ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════
elif "Anomaly" in page:
    st.markdown('<div class="section-header">⚠️ Anomaly Detection — Isolation Forest</div>', unsafe_allow_html=True)

    # KPIs
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Anomalies", len(anomalies))
    c2.metric("High Risk", int((anomalies["RiskLevel"]=="High").sum()))
    c3.metric("Avg Anomaly Score", f"{anomalies['AnomalyScore'].mean():.3f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    # Histogram of invoice totals — normal vs anomaly overlay
    with col_l:
        all_inv = R["scored_invoices"].copy()
        all_inv["Type"] = np.where(all_inv["AnomalyLabel"] == -1, "Anomaly", "Normal")
        fig = px.histogram(
            all_inv, x="InvoiceTotal", color="Type",
            nbins=80, barmode="overlay",
            color_discrete_map={"Anomaly": "#ef4444", "Normal": "#00C9A7"},
            opacity=0.75,
        )
        fig = _chart_layout(fig, "📊 Invoice Total Distribution", 380)
        fig.update_xaxes(title="Invoice Total (£)")
        st.plotly_chart(fig, use_container_width=True)

    # Anomaly score scatter
    with col_r:
        fig = px.scatter(
            anomalies, x="InvoiceTotal", y="AnomalyScore",
            color="RiskLevel",
            color_discrete_map={"High":"#ef4444","Medium":"#fbbf24","Low":"#22c55e"},
            size="ItemCount", hover_data=["InvoiceNo","CustomerID"],
            size_max=18,
        )
        fig = _chart_layout(fig, "🎯 Anomaly Score vs Invoice Total", 380)
        fig.update_xaxes(title="Invoice Total (£)")
        fig.update_yaxes(title="Anomaly Score (0–1)")
        st.plotly_chart(fig, use_container_width=True)

    # Risk breakdown bar
    risk_counts = anomalies["RiskLevel"].value_counts().reset_index()
    risk_counts.columns = ["RiskLevel","Count"]
    fig = px.bar(
        risk_counts, x="RiskLevel", y="Count",
        color="RiskLevel",
        color_discrete_map={"High":"#ef4444","Medium":"#fbbf24","Low":"#22c55e"},
        text="Count",
    )
    fig = _chart_layout(fig, "🚦 Risk Level Breakdown", 280)
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown('<div class="section-header">Detected Anomalous Invoices</div>', unsafe_allow_html=True)
    display_cols = [c for c in ["InvoiceNo","CustomerID","InvoiceTotal","ItemCount","Hour","AnomalyScore","RiskLevel"] if c in anomalies.columns]
    styled = (
        anomalies[display_cols]
        .sort_values("AnomalyScore", ascending=False)
        .style.applymap(_color_risk, subset=["RiskLevel"])
        .format({
            "InvoiceTotal": "£{:,.2f}",
            "AnomalyScore": "{:.3f}",
        })
    )
    st.dataframe(styled, use_container_width=True, height=380)
    _csv_download(anomalies, "anomalies.csv", "⬇️ Download Anomalies CSV")

    # Customer search
    st.markdown("---")
    st.markdown("**🔍 Search Customer Anomalies**")
    cust_id = st.text_input("Enter CustomerID", placeholder="e.g. 12345")
    if cust_id:
        found = anomalies[anomalies["CustomerID"].astype(str) == cust_id.strip()]
        if found.empty:
            st.info("No anomalies found for this customer.")
        else:
            st.success(f"Found {len(found)} anomalous invoice(s).")
            st.dataframe(found[display_cols], use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 3 — CUSTOMER SEGMENTATION
# ════════════════════════════════════════════════════════════════
elif "Segmentation" in page:
    st.markdown('<div class="section-header">👥 Customer Segmentation — K-Means</div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Customers", f"{len(clusters):,}")
    c2.metric("Segments", clusters["Cluster"].nunique())
    c3.metric("Avg Monetary", f"£{clusters['Monetary'].mean():,.0f}")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    # 3-D scatter
    with col_l:
        fig = px.scatter_3d(
            clusters, x="Recency", y="Frequency", z="Monetary",
            color="ClusterName",
            color_discrete_sequence=px.colors.qualitative.Bold,
            hover_data=["CustomerID","AvgSpend"],
            opacity=0.75,
        )
        fig = _chart_layout(fig, "🌐 RFM Cluster Space", 480)
        st.plotly_chart(fig, use_container_width=True)

    # Per-cluster box plots
    with col_r:
        col_tab = st.tabs(["Monetary", "Recency", "Frequency", "AvgSpend"])
        for tab, metric in zip(col_tab, ["Monetary","Recency","Frequency","AvgSpend"]):
            with tab:
                fig = px.box(
                    clusters, x="ClusterName", y=metric,
                    color="ClusterName",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig = _chart_layout(fig, f"{metric} by Cluster", 360)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # Scatter F vs M coloured by cluster
    st.markdown('<div class="section-header">Frequency vs Monetary</div>', unsafe_allow_html=True)
    fig = px.scatter(
        clusters, x="Frequency", y="Monetary",
        color="ClusterName",
        color_discrete_sequence=px.colors.qualitative.Bold,
        size="AvgSpend", size_max=20,
        hover_data=["CustomerID","Recency"],
        opacity=0.8,
    )
    fig = _chart_layout(fig, "", 420)
    fig.update_xaxes(title="Purchase Frequency")
    fig.update_yaxes(title="Total Spend (£)")
    st.plotly_chart(fig, use_container_width=True)

    # Cluster stats table
    st.markdown('<div class="section-header">Cluster Summary Statistics</div>', unsafe_allow_html=True)
    cluster_summary = (
        clusters.groupby("ClusterName")
        .agg(
            Customers=("CustomerID","count"),
            Avg_Recency=("Recency","mean"),
            Avg_Frequency=("Frequency","mean"),
            Avg_Monetary=("Monetary","mean"),
            Avg_Spend=("AvgSpend","mean"),
        )
        .reset_index()
        .round(2)
    )
    st.dataframe(cluster_summary, use_container_width=True)
    _csv_download(clusters, "customer_segments.csv", "⬇️ Download Segments CSV")

    # Customer search
    st.markdown("---")
    st.markdown("**🔍 Look Up a Customer**")
    cust_q = st.text_input("CustomerID", placeholder="e.g. 17850", key="cust_search")
    if cust_q:
        row = clusters[clusters["CustomerID"].astype(str) == cust_q.strip()]
        if row.empty:
            st.info("Customer not found.")
        else:
            st.success(f"**Segment:** {row.iloc[0]['ClusterName']}")
            st.write(row[["CustomerID","Recency","Frequency","Monetary","AvgSpend","ClusterName"]].T)


# ════════════════════════════════════════════════════════════════
# PAGE 4 — MARKET BASKET
# ════════════════════════════════════════════════════════════════
elif "Basket" in page:
    st.markdown('<div class="section-header">🛒 Market Basket Analysis — Apriori</div>', unsafe_allow_html=True)

    if rules.empty:
        st.warning("⚠️ No association rules found with current support/confidence settings. Try lowering them in the sidebar.")
        st.stop()

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Rules", f"{len(rules):,}")
    c2.metric("Avg Lift", f"{rules['lift'].mean():.2f}")
    c3.metric("Top Lift", f"{rules['lift'].max():.2f}")

    st.markdown("---")

    # Top rules table
    st.markdown('<div class="section-header">Top Association Rules (by Lift)</div>', unsafe_allow_html=True)
    show_n = st.slider("Show top N rules", 5, min(100, len(rules)), min(20, len(rules)))
    top = rules.head(show_n).copy()
    display_rules_cols = [c for c in ["antecedents","consequents","support","confidence","lift","RuleStrength"] if c in top.columns]
    st.dataframe(
        top[display_rules_cols].style.format({
            "support": "{:.4f}",
            "confidence": "{:.4f}",
            "lift": "{:.3f}",
        }),
        use_container_width=True,
        height=380,
    )
    _csv_download(rules, "association_rules.csv", "⬇️ Download Rules CSV")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    # Lift bar chart
    with col_l:
        top20 = rules.head(20).copy()
        top20["label"] = top20["antecedents"].str[:30] + " → " + top20["consequents"].str[:30]
        fig = go.Figure(go.Bar(
            x=top20["lift"],
            y=top20["label"],
            orientation="h",
            marker=dict(
                color=top20["lift"],
                colorscale="Teal",
                showscale=True,
                colorbar=dict(title="Lift"),
            ),
            hovertemplate="Lift: %{x:.3f}<extra></extra>",
        ))
        fig = _chart_layout(fig, "🏆 Top 20 Rules by Lift", 520)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # Support vs Confidence bubble
    with col_r:
        fig = px.scatter(
            rules.head(50),
            x="support", y="confidence", size="lift",
            color="lift",
            color_continuous_scale="Teal",
            hover_data=["antecedents","consequents"],
            size_max=25,
        )
        fig = _chart_layout(fig, "Support vs Confidence (bubble = Lift)", 520)
        fig.update_xaxes(title="Support")
        fig.update_yaxes(title="Confidence")
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 5 — TEMPORAL ANALYSIS
# ════════════════════════════════════════════════════════════════
elif "Temporal" in page:
    st.markdown('<div class="section-header">📅 Temporal Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🕐 Peak Hours", "📈 Monthly Trends", "⚡ Revenue Spikes"])

    # ── Tab 1: Peak Hours
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=peak_hours["Hour"],
            y=peak_hours["TotalRevenue"],
            name="Revenue",
            marker_color=[TEAL if p else "#1e3a5f" for p in peak_hours["IsPeak"]],
            hovertemplate="Hour %{x}:00<br>Revenue: £%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=peak_hours["Hour"],
            y=peak_hours["TransactionCount"],
            name="Transactions",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#38bdf8", width=2),
            marker=dict(size=6),
        ))
        fig = _chart_layout(fig, "Revenue & Transactions by Hour", 420)
        fig.update_layout(
            yaxis=dict(title="Revenue (£)"),
            yaxis2=dict(title="Transactions", overlaying="y", side="right"),
        )
        fig.update_xaxes(title="Hour of Day", dtick=1)
        st.plotly_chart(fig, use_container_width=True)

        # Day of week heatmap
        st.markdown('<div class="section-header">Revenue Heatmap — Hour × Day of Week</div>', unsafe_allow_html=True)
        if "DayOfWeek" in clean_df.columns:
            heatmap_data = (
                clean_df.groupby(["DayOfWeek","Hour"])["TotalAmount"].sum().reset_index()
            )
            days_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            heatmap_pivot = heatmap_data.pivot(index="DayOfWeek", columns="Hour", values="TotalAmount").fillna(0)
            heatmap_pivot = heatmap_pivot.reindex([d for d in days_order if d in heatmap_pivot.index])
            fig2 = px.imshow(
                heatmap_pivot,
                color_continuous_scale="Teal",
                aspect="auto",
                labels=dict(x="Hour", y="Day", color="Revenue"),
            )
            fig2 = _chart_layout(fig2, "", 320)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Monthly Trends
    with tab2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Monthly Revenue (£)", "Month-on-Month Change (%)"))
        fig.add_trace(go.Scatter(
            x=monthly["MonthYear"], y=monthly["TotalRevenue"],
            mode="lines+markers",
            line=dict(color=TEAL, width=2.5),
            fill="tozeroy", fillcolor="rgba(0,201,167,0.08)",
            hovertemplate="%{x}<br>£%{y:,.0f}<extra></extra>",
        ), row=1, col=1)

        colors_mom = ["#22c55e" if v >= 0 else "#ef4444" for v in monthly["MoM_Change"].fillna(0)]
        fig.add_trace(go.Bar(
            x=monthly["MonthYear"], y=monthly["MoM_Change"],
            marker_color=colors_mom,
            hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
        ), row=2, col=1)

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            plot_bgcolor=CHART_BGCOLOR,
            paper_bgcolor=PAPER_BGCOLOR,
            height=520,
            margin=dict(l=20,r=20,t=50,b=20),
            font=dict(family="Inter, sans-serif", color="#94a3b8"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        _csv_download(monthly, "monthly_trends.csv", "⬇️ Download Monthly Trends")

    # ── Tab 3: Revenue Spikes
    with tab3:
        if spikes.empty:
            st.info("No significant revenue spikes detected (Z-score > 2).")
        else:
            st.metric("Revenue Spikes Detected", len(spikes))
            # Full daily timeline with spikes highlighted
            daily_rev = clean_df.groupby(clean_df["InvoiceDate"].dt.date)["TotalAmount"].sum().reset_index()
            daily_rev.columns = ["Date","DailyRevenue"]
            daily_rev["Date"] = pd.to_datetime(daily_rev["Date"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_rev["Date"], y=daily_rev["DailyRevenue"],
                mode="lines", line=dict(color="#475569", width=1.5), name="Daily Revenue",
            ))
            fig.add_trace(go.Scatter(
                x=spikes["Date"], y=spikes["DailyRevenue"],
                mode="markers", marker=dict(color="#ef4444", size=11, symbol="star"),
                name="Spike", hovertemplate="%{x}<br>£%{y:,.0f}<extra></extra>",
            ))
            fig = _chart_layout(fig, "Daily Revenue with Spike Markers", 420)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                spikes.style.format({"DailyRevenue":"£{:,.2f}","ZScore":"{:.2f}","Deviation":"£{:,.2f}"}),
                use_container_width=True,
            )
            _csv_download(spikes, "revenue_spikes.csv", "⬇️ Download Spikes CSV")


# ════════════════════════════════════════════════════════════════
# PAGE 6 — COHORT RETENTION
# ════════════════════════════════════════════════════════════════
elif "Cohort Retention" in page:
    st.markdown('<div class="section-header">📊 Cohort Retention Analysis</div>', unsafe_allow_html=True)

    if cohort_retention.empty:
        st.warning("Not enough data to compute cohort retention. Ensure CustomerID and InvoiceDate are present.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Cohorts", len(cohort_retention))
    c2.metric("Max Period (months)", int(cohort_retention.columns.max()))
    avg_m2 = cohort_retention[2].mean() if 2 in cohort_retention.columns else float("nan")
    c3.metric("Avg Month-2 Retention", f"{avg_m2:.1f}%" if pd.notna(avg_m2) else "N/A")

    st.markdown("---")
    st.markdown(
        '<div class="section-header">Retention Heatmap (% of cohort active each month)</div>',
        unsafe_allow_html=True,
    )

    fig = px.imshow(
        cohort_retention,
        color_continuous_scale="Teal",
        aspect="auto",
        labels=dict(x="Months Since First Purchase", y="Cohort Month", color="Retention %"),
        zmin=0,
        zmax=100,
        text_auto=".1f",
    )
    fig = _chart_layout(fig, "", 520)
    fig.update_xaxes(title="Months Since First Purchase", side="bottom")
    fig.update_yaxes(title="Cohort Month")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Cohort Retention Table</div>', unsafe_allow_html=True)
    display_cohort = cohort_retention.round(1)
    display_cohort.index.name = "Cohort Month"
    st.dataframe(display_cohort, use_container_width=True, height=380)
    _csv_download(
        cohort_retention.reset_index(),
        "cohort_retention.csv",
        "⬇️ Download Cohort Retention CSV",
    )


# ════════════════════════════════════════════════════════════════
# PAGE 7 — PRODUCT INTELLIGENCE
# ════════════════════════════════════════════════════════════════
elif "Product Intelligence" in page:
    st.markdown('<div class="section-header">🏷️ Product Intelligence — RFM Segmentation</div>', unsafe_allow_html=True)

    if product_rfm.empty:
        st.warning("Not enough data for product RFM. Ensure StockCode, Description, and TotalAmount are present.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    seg_counts = product_rfm["Segment"].value_counts()
    c1.metric("Products Analysed", f"{len(product_rfm):,}")
    c2.metric("Star Products", int(seg_counts.get("Star Product", 0)))
    c3.metric("Declining", int(seg_counts.get("Declining", 0)))
    c4.metric("Avg RFM Score", f"{product_rfm['RFM_Score'].mean():.1f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        seg_bar = seg_counts.reset_index()
        seg_bar.columns = ["Segment", "Count"]
        fig = px.bar(
            seg_bar,
            x="Segment",
            y="Count",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            text="Count",
        )
        fig = _chart_layout(fig, "📊 Segment Distribution", 380)
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = px.scatter(
            product_rfm,
            x="Frequency",
            y="Monetary",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            hover_data=["StockCode", "Description", "Recency", "RFM_Score"],
            opacity=0.75,
        )
        fig = _chart_layout(fig, "Frequency vs Monetary by Segment", 380)
        fig.update_xaxes(title="Invoice Frequency")
        fig.update_yaxes(title="Total Revenue (£)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">⭐ Top 10 Star Products</div>', unsafe_allow_html=True)
    stars = (
        product_rfm[product_rfm["Segment"] == "Star Product"]
        .sort_values("Monetary", ascending=False)
        .head(10)
    )
    if stars.empty:
        st.info("No Star Products identified with current data.")
    else:
        st.dataframe(
            stars.style.format({"Monetary": "£{:,.2f}", "RFM_Score": "{:.0f}"}),
            use_container_width=True,
            height=320,
        )

    _csv_download(product_rfm, "product_rfm.csv", "⬇️ Download Product RFM CSV")


# ════════════════════════════════════════════════════════════════
# PAGE 8 — MODEL INFO
# ════════════════════════════════════════════════════════════════
elif "Model Info" in page:
    st.markdown('<div class="section-header">🤖 Models & Methodology</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
<div class="glass-card">
<h3 style="color:#00C9A7;margin-top:0">⚠️ Isolation Forest</h3>
<span class="badge badge-red">Anomaly Detection</span>
<hr style="border-color:rgba(255,255,255,0.07)">
<p style="color:#94a3b8;font-size:0.9rem">
An ensemble unsupervised algorithm that isolates observations by randomly partitioning the feature space.
Anomalies require fewer splits to isolate — they surface as short paths in the decision trees.
</p>
<ul style="color:#cbd5e1;font-size:0.88rem">
  <li>Features: <code>InvoiceTotal</code>, <code>ItemCount</code>, <code>Hour</code></li>
  <li>Estimators: 200 trees</li>
  <li>Outputs: Anomaly label (−1/+1) + normalised score [0, 1]</li>
  <li>Risk bucketing: Low / Medium / High</li>
</ul>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="glass-card">
<h3 style="color:#00C9A7;margin-top:0">🛒 Apriori Algorithm</h3>
<span class="badge badge-teal">Market Basket Analysis</span>
<hr style="border-color:rgba(255,255,255,0.07)">
<p style="color:#94a3b8;font-size:0.9rem">
Discovers frequent itemsets and generates association rules. Uses the
<em>anti-monotone constraint</em>: any subset of a frequent itemset must also be frequent.
</p>
<ul style="color:#cbd5e1;font-size:0.88rem">
  <li>Basket: Invoice → product presence matrix</li>
  <li>Metrics: Support, Confidence, Lift, Conviction</li>
  <li>Rule strength: Strong (lift ≥ 3), Moderate (≥ 2), Weak (≥ 1)</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="glass-card">
<h3 style="color:#00C9A7;margin-top:0">👥 K-Means Clustering</h3>
<span class="badge badge-blue">Customer Segmentation</span>
<hr style="border-color:rgba(255,255,255,0.07)">
<p style="color:#94a3b8;font-size:0.9rem">
Partitions customers into <em>k</em> segments by minimising within-cluster variance.
Optimal <em>k</em> is found using the elbow method (largest second derivative of inertia).
</p>
<ul style="color:#cbd5e1;font-size:0.88rem">
  <li>Features: <code>Recency</code>, <code>Frequency</code>, <code>Monetary</code>, <code>AvgSpend</code></li>
  <li>Preprocessing: StandardScaler normalisation</li>
  <li>Init strategy: k-means++ with 20 restarts</li>
  <li>Segments labelled by average monetary value</li>
</ul>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="glass-card">
<h3 style="color:#00C9A7;margin-top:0">📅 Temporal Mining</h3>
<span class="badge badge-amber">Trend Analysis</span>
<hr style="border-color:rgba(255,255,255,0.07)">
<p style="color:#94a3b8;font-size:0.9rem">
Statistical time-series analysis of transaction patterns across hours, days, and months.
Revenue spikes are flagged using <strong>Z-score > 2</strong>.
</p>
<ul style="color:#cbd5e1;font-size:0.88rem">
  <li>Peak hours: 75th percentile revenue threshold</li>
  <li>Monthly: MoM % change tracking</li>
  <li>Spikes: Daily Z-score outlier detection</li>
  <li>Heatmap: Hour × Day of Week revenue pattern</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📁 Dataset & Pipeline Stats</div>', unsafe_allow_html=True)
    stat_data = {
        "Metric": [
            "Raw Rows", "Clean Rows", "Retention Rate",
            "Invoices", "Customers", "Date Range",
            "Anomalies Found", "Customer Segments", "Association Rules",
        ],
        "Value": [
            f"{len(R['raw_df']):,}",
            f"{stats['total_transactions']:,}",
            f"{100*stats['total_transactions']/max(len(R['raw_df']),1):.1f}%",
            f"{stats['total_invoices']:,}",
            f"{stats['total_customers']:,}",
            f"{stats['date_range'][0]} → {stats['date_range'][1]}",
            f"{stats['total_anomalies']:,}",
            f"{stats['total_clusters']:,}",
            f"{stats['total_rules']:,}",
        ],
    }
    st.dataframe(pd.DataFrame(stat_data), use_container_width=True, hide_index=True)
