"""
Real-time fraud/anomaly detection dashboard for a simulated Canada-wide
e-commerce marketplace.

Visual language modeled on modern data-platform UIs (dark icon rail +
light content area, typed table headers, live row-count header) rather
than a generic Streamlit look.

Uses st.fragment(run_every=...) so only the live section re-executes on a
timer — the rest of the page (header, sidebar, offline model card) doesn't
re-render every tick. The LiveStreamEngine is held in st.cache_resource so
it survives reruns and keeps accumulating state (customer history, warehouse
rows) instead of restarting from scratch every refresh.

Run with: streamlit run app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import warehouse

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

BLUE = "#2563EB"
GREEN = "#16A34A"
RED = "#EF4444"
AMBER = "#D97706"
PURPLE = "#7C3AED"
TEAL = "#0891B2"
PINK = "#DB2777"
GRAY = "#6B7280"
DARK = "#111827"
CATEGORICAL_PALETTE = [BLUE, GREEN, AMBER, RED, PURPLE, TEAL, PINK, GRAY]
PATTERN_COLORS = {"amount_spike": AMBER, "velocity_burst": PURPLE, "odd_hour_bulk": PINK}

st.set_page_config(page_title="Canada Commerce — Real-Time Fraud Detection", layout="wide", page_icon="🛡️")


def _bootstrap_if_needed() -> None:
    """On a fresh deploy (e.g. Streamlit Community Cloud), the trained model
    and historical dataset aren't committed to the repo — they're build
    artifacts, not source. Generate + train them once on first launch
    instead. Every generator is seeded, so the result is identical to what
    ships from local development; later reruns skip this entirely since the
    files already exist."""
    if (MODELS_DIR / "anomaly_model.joblib").exists():
        return
    with st.spinner("First run: generating historical data and training the anomaly model..."):
        import generate_historical
        import train_model
        generate_historical.main()
        train_model.main()


_bootstrap_if_needed()

from live_engine import LiveStreamEngine  # noqa: E402 (must follow bootstrap)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&display=swap');

    /* ---- Dark icon-rail sidebar over a light content area ---- */
    section[data-testid="stSidebar"] {
        background-color: #14171C;
        border-right: 1px solid #14171C;
    }
    section[data-testid="stSidebar"] * { color: #D7DCE3; }
    section[data-testid="stSidebar"] .stSlider label, section[data-testid="stSidebar"] .stCaption { color: #9AA3B2 !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1); }
    .sidebar-brand { display:flex; align-items:center; gap:9px; padding-bottom:10px; margin-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.1); }
    .sidebar-brand .icon { font-size:22px; }
    .sidebar-brand .name { font-weight:700; font-size:15px; color:#FFFFFF; }
    .sidebar-brand .sub { font-size:11px; color:#7C8598; margin-top:-2px; }

    /* ---- Top bar ---- */
    .topbar { display:flex; align-items:center; justify-content:space-between; padding: 2px 2px 14px 2px; border-bottom: 1px solid #E5E7EB; margin-bottom: 18px; }
    .topbar .crumb { font-size: 13.5px; color:#6B7280; }
    .topbar .crumb b { color:#111827; }
    .pill-live { display:inline-flex; align-items:center; gap:6px; background:#16A34A; color:#fff; font-weight:700; font-size:11.5px; padding:6px 13px; border-radius:7px; letter-spacing:.03em; }
    .pill-live.paused { background:#9CA3AF; }
    .pill-dot { width:7px; height:7px; border-radius:50%; background:#fff; animation: pulse2 1.4s infinite; }
    @keyframes pulse2 { 0%{opacity:1} 50%{opacity:.35} 100%{opacity:1} }

    /* ---- Dataset hero card ---- */
    .dataset-hero { padding: 4px 2px 12px 2px; margin-bottom: 6px; }
    .dataset-hero .title-row { display:flex; align-items:baseline; gap:12px; flex-wrap: wrap; }
    .dataset-hero .title-row .icon { font-size:26px; align-self: center; }
    .dataset-hero h2 {
        margin:0; font-family: 'Oswald', 'Segoe UI', sans-serif; font-size:32px; font-weight:700;
        letter-spacing: 0.01em; text-transform: uppercase; color:#111827; line-height: 1.1;
    }
    .dataset-hero .table-tag {
        font-family: Consolas, "SFMono-Regular", monospace; font-size: 11px; font-weight: 600;
        color: #2563EB; background: #EFF4FF; border: 1px solid #DBE6FE; border-radius: 6px;
        padding: 3px 8px; white-space: nowrap;
    }
    .dataset-hero .meta { color:#6B7280; font-size:13px; margin: 6px 0 0 38px; }
    .dataset-hero .desc { color:#9CA3AF; font-size:12.5px; margin: 3px 0 0 38px; font-style: italic; max-width: 62ch; }

    /* ---- KPI / gauge cards ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 10px !important;
    }
    .kpi-label { color: #6B7280; font-size: 12.5px; font-weight: 600; display:flex; align-items:center; gap:6px; }
    .kpi-value { font-size: 26px; font-weight: 700; color: #111827; margin: 3px 0 2px 0; }

    /* ---- Typed data table ---- */
    .table-scroll { max-height: 480px; overflow-y: auto; border-radius: 10px; border: 1px solid #E5E7EB; }
    .fraud-table { width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; }
    .fraud-table thead th {
        position: sticky; top: 0; background: #F9FAFB; z-index: 1;
        text-align: left; padding: 9px 14px; border-bottom: 1px solid #E5E7EB;
    }
    .fraud-table thead .col-name { display:block; font-weight: 700; color:#111827; font-size: 12px; }
    .fraud-table thead .col-type { display:block; font-weight: 500; color:#2563EB; font-size: 10.5px; font-family: Consolas, monospace; margin-top: 1px; }
    .fraud-table tbody td { padding: 8px 14px; border-bottom: 1px solid #F1F3F5; color: #1F2937; }
    .fraud-table tbody tr:hover { background: #F9FAFB; }
    .fraud-table tbody tr.flagged-row { background: #FEF2F2; }
    .fraud-table .mono { font-family: Consolas, "SFMono-Regular", monospace; color: #374151; }

    .badge { display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 600; text-transform: capitalize; white-space: nowrap; }
    .badge-muted { background: #F3F4F6; color: #6B7280; border: 1px solid #E5E7EB; }

    .dot { display:inline-block; width:9px; height:9px; border-radius:50%; }
    .dot-red { background:#EF4444; box-shadow:0 0 6px rgba(239,68,68,.55); }
    .dot-gray { background: #D1D5DB; }

    /* ---- Alert cards ---- */
    .alert-list { display:flex; flex-direction:column; gap:10px; max-height:560px; overflow-y:auto; padding-right:6px; }
    .alert-card { display:flex; justify-content:space-between; align-items:center; gap: 14px; background:#fff; border:1px solid #FECACA; border-left:3px solid #EF4444; border-radius:10px; padding:13px 18px; }
    .alert-amount { font-size:19px; font-weight:700; color:#111827; }
    .alert-meta { font-size:12px; color:#6B7280; margin-top:2px; }
    .alert-card-right { display:flex; align-items:center; gap:12px; flex-shrink: 0; }
    .alert-score { font-size:11px; color:#9CA3AF; font-family: Consolas, monospace; }
    .tag { font-size:11px; padding:3px 9px; border-radius:6px; font-weight:600; white-space:nowrap; }
    .tag-success { background: #ECFDF5; color:#16A34A; border:1px solid #BBF7D0; }
    .tag-warn { background: #FFFBEB; color:#D97706; border:1px solid #FDE68A; }

    .empty-state { padding: 40px 20px; text-align:center; color:#9CA3AF; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine(events_per_tick: float, fraud_rate: float) -> LiveStreamEngine:
    return LiveStreamEngine(events_per_tick=events_per_tick, fraud_rate=fraud_rate)


@st.cache_data
def load_offline_metrics() -> dict | None:
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def style_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=DARK,
        height=height,
        margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
    return fig


def sparkline_fig(values, color: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(y=list(values), mode="lines", line=dict(color=color, width=2), fill="tozeroy", fillcolor=hex_to_rgba(color, 0.12))
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=46, margin=dict(t=0, b=0, l=0, r=0), showlegend=False,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def hero_trend_fig(ts: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=ts["bucket_time"], y=ts["n_transactions"], mode="lines",
            line=dict(color=BLUE, width=2), fill="tozeroy", fillcolor=hex_to_rgba(BLUE, 0.08),
        )
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=130, margin=dict(t=6, b=20, l=10, r=10), showlegend=False)
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10, color=GRAY))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", tickfont=dict(size=10, color=GRAY))
    return fig


def gauge_fig(value: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(value * 100, 1),
            number={"suffix": "%", "font": {"size": 26, "color": DARK}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "rgba(0,0,0,0.15)", "tickfont": {"size": 9, "color": GRAY}},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0.045)",
                "borderwidth": 0,
            },
        )
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=150, margin=dict(t=15, b=5, l=25, r=25), font_color=DARK)
    return fig


def badge_html(pattern) -> str:
    if not pattern or pd.isna(pattern):
        return '<span class="badge badge-muted">normal</span>'
    color = PATTERN_COLORS.get(pattern, BLUE)
    label = str(pattern).replace("_", " ")
    return f'<span class="badge" style="background:{color}17;color:{color};border:1px solid {color}40;">{label}</span>'


TABLE_COLUMNS = [
    ("Time", "DateTime64(3)"), ("Customer", "UInt32"), ("Product", "String"), ("Qty", "UInt8"),
    ("Amount", "Float64"), ("Score", "Float64"), ("Flag", "Bool"), ("Pattern", "String"),
]


def render_transactions_table(df: pd.DataFrame) -> str:
    if not len(df):
        return '<div class="empty-state">Waiting for transactions…</div>'
    rows = []
    for _, r in df.iterrows():
        flagged = bool(r.get("is_flagged"))
        dot = '<span class="dot dot-red"></span>' if flagged else '<span class="dot dot-gray"></span>'
        rows.append(
            f"<tr class=\"{'flagged-row' if flagged else ''}\">"
            f"<td class='mono'>{r['event_time']}</td>"
            f"<td>#{r['customer_id']}</td>"
            f"<td>{r['product_name']}</td>"
            f"<td>{r['quantity']}</td>"
            f"<td class='mono'>${r['amount']:.2f}</td>"
            f"<td class='mono'>{r['anomaly_score']:.3f}</td>"
            f"<td>{dot}</td>"
            f"<td>{badge_html(r['fraud_pattern'])}</td>"
            f"</tr>"
        )
    header_cells = "".join(f"<th><span class='col-name'>{name}</span><span class='col-type'>{typ}</span></th>" for name, typ in TABLE_COLUMNS)
    return f'<div class="table-scroll"><table class="fraud-table"><thead><tr>{header_cells}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def render_alert_cards(df: pd.DataFrame) -> str:
    if not len(df):
        return '<div class="empty-state">No alerts yet — waiting for the model to flag something.</div>'
    cards = []
    for _, r in df.iterrows():
        correct = bool(r.get("is_fraud_synthetic"))
        tag = '<span class="tag tag-success">✅ true fraud</span>' if correct else '<span class="tag tag-warn">⚠️ false positive</span>'
        cards.append(
            f"""<div class="alert-card">
                <div>
                    <div class="alert-amount">${r['amount']:.2f}</div>
                    <div class="alert-meta">Customer #{r['customer_id']} · {r['product_name']} · {r['event_time']}</div>
                </div>
                <div class="alert-card-right">
                    {badge_html(r['fraud_pattern'])}
                    {tag}
                    <div class="alert-score">score {r['anomaly_score']:.3f}</div>
                </div>
            </div>"""
        )
    return '<div class="alert-list">' + "".join(cards) + "</div>"


def kpi_card(col, icon: str, label: str, value: str, sparkline_values=None, color: str = BLUE):
    with col:
        with st.container(border=True):
            st.markdown(f"<div class='kpi-label'>{icon} {label}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kpi-value'>{value}</div>", unsafe_allow_html=True)
            if sparkline_values is not None and len(sparkline_values) > 1:
                st.plotly_chart(sparkline_fig(sparkline_values, color), use_container_width=True, config={"displayModeBar": False})


def gauge_card(col, label: str, value: float, color: str):
    with col:
        with st.container(border=True):
            st.markdown(f"<div class='kpi-label'>{label}</div>", unsafe_allow_html=True)
            st.plotly_chart(gauge_fig(value, color), use_container_width=True, config={"displayModeBar": False})


# --- Sidebar ---
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="icon">🛡️</div>
            <div>
                <div class="name">Canada Commerce</div>
                <div class="sub">Fraud Ops Console</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("⚙️ Stream controls")
    events_per_tick = st.slider("Avg transactions / tick (~1.5s)", 1.0, 20.0, 6.0, 0.5)
    fraud_rate = st.slider("Injected anomaly rate", 0.0, 0.15, 0.03, 0.005)
    streaming = st.toggle("Streaming", value=True)
    if st.button("Restart stream (clears history)", use_container_width=True):
        get_engine.clear()
        st.rerun()

    st.divider()
    st.subheader("🧪 Offline model evaluation")
    metrics = load_offline_metrics()
    if metrics:
        st.caption("From src/train_model.py — held-out test set, labels used only for scoring, never for fitting.")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Precision", f"{metrics['precision']:.0%}")
        mc2.metric("Recall", f"{metrics['recall']:.0%}")
        mc3.metric("AUC", f"{metrics['roc_auc']:.2f}")
        st.caption("Recall by injected pattern:")
        for pattern, d in metrics["recall_by_pattern"].items():
            st.text(f"{pattern.replace('_', ' ').title()}: {d['rate']:.0%} ({d['caught']}/{d['n']})")
    else:
        st.warning("Run `python src/train_model.py` first.")

engine = get_engine(events_per_tick, fraud_rate)


@st.fragment(run_every="1.5s")
def live_section():
    # `engine` (and its single DuckDB connection) is shared across every
    # visitor session via st.cache_resource, and DuckDB connections aren't
    # safe for concurrent multi-threaded use — hold the engine's lock for
    # this whole render pass so two sessions can never touch it at once.
    with engine.lock:
        _render_live_section()


def _render_live_section():
    if streaming:
        engine.tick()

    stats = warehouse.summary_stats(engine.conn)
    ts = warehouse.revenue_timeseries(engine.conn, seconds_back=300, bucket_seconds=5)

    db_bytes = warehouse.WAREHOUSE_PATH.stat().st_size if warehouse.WAREHOUSE_PATH.exists() else 0
    db_size = f"{db_bytes / 1024:,.1f} KB" if db_bytes < 1024 * 1024 else f"{db_bytes / (1024*1024):,.1f} MB"

    badge_class = "pill-live" if streaming else "pill-live paused"
    badge_text = "LIVE" if streaming else "PAUSED"
    st.markdown(
        f"""
        <div class="topbar">
            <div class="crumb">🛡️ <b>Canada Commerce</b> / Fraud Monitoring</div>
            <div class="{badge_class}"><span class="pill-dot"></span>{badge_text}</div>
        </div>
        <div class="dataset-hero">
            <div class="title-row"><span class="icon">🗄️</span><h2>Real-Time Fraud Transaction Monitor</h2><span class="table-tag">fraud_transactions</span></div>
            <div class="meta">{stats['n_transactions']:,} rows · {db_size} · updating every ~1.5s</div>
            <div class="desc">Simulated Canada-wide e-commerce transaction stream, scored live by an unsupervised anomaly model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if len(ts):
        st.plotly_chart(hero_trend_fig(ts), use_container_width=True, config={"displayModeBar": False})

    tab_overview, tab_feed, tab_alerts, tab_regions = st.tabs(
        ["📊 Overview", "📜 Live Feed", "🚨 Alerts", "🗺️ Regions & Categories"]
    )

    with tab_overview:
        k1, k2, k3, k4, k5 = st.columns(5)
        kpi_card(k1, "🧾", "Transactions", f"{stats['n_transactions']:,}",
                 ts["n_transactions"] if len(ts) else None, BLUE)
        kpi_card(k2, "💰", "Revenue (CAD)", f"${stats['total_revenue']:,.0f}",
                 ts["revenue"] if len(ts) else None, GREEN)
        kpi_card(k3, "🚩", "Flagged anomalous", f"{stats['n_flagged']:,}",
                 ts["n_flagged"] if len(ts) else None, RED)
        gauge_card(k4, "🎯 Live precision", stats["live_precision"], GREEN)
        gauge_card(k5, "🔍 Live recall", stats["live_recall"], BLUE)

        st.write("")
        if len(ts):
            fig = go.Figure()
            fig.add_bar(x=ts["bucket_time"], y=ts["n_transactions"], name="Transactions", marker_color=BLUE)
            fig.add_bar(x=ts["bucket_time"], y=ts["n_flagged"], name="Flagged", marker_color=RED)
            fig.update_layout(barmode="overlay", title="Transaction volume — last 5 minutes (5s buckets)")
            st.plotly_chart(style_fig(fig, height=340), use_container_width=True)
        else:
            st.info("Waiting for transactions...")

    with tab_feed:
        col_caption, col_export = st.columns([4, 1])
        col_caption.caption(f"Most recent 20 of {stats['n_transactions']:,} transactions processed this session.")
        recent = warehouse.recent_transactions(engine.conn, limit=20)
        if len(recent):
            col_export.download_button(
                "⤓ Export CSV", recent.to_csv(index=False), file_name="recent_transactions.csv",
                mime="text/csv", use_container_width=True, key="export_txns",
            )
        st.markdown(render_transactions_table(recent), unsafe_allow_html=True)

    with tab_alerts:
        col_caption, col_export = st.columns([4, 1])
        col_caption.caption(f"{stats['n_flagged']:,} flagged so far · live precision {stats['live_precision']:.0%} · live recall {stats['live_recall']:.0%}")
        alerts = warehouse.recent_alerts(engine.conn, limit=15)
        if len(alerts):
            col_export.download_button(
                "⤓ Export CSV", alerts.to_csv(index=False), file_name="fraud_alerts.csv",
                mime="text/csv", use_container_width=True, key="export_alerts",
            )
        st.markdown(render_alert_cards(alerts), unsafe_allow_html=True)

    with tab_regions:
        col_region, col_cat = st.columns(2)

        with col_region:
            region_df = warehouse.revenue_by_region(engine.conn)
            region_df = region_df.dropna(subset=["region"]) if len(region_df) else region_df
            if len(region_df) and region_df["revenue"].sum() > 0:
                region_df = region_df.sort_values("revenue", ascending=True)
                fig_region = px.bar(
                    region_df, x="revenue", y="region", orientation="h",
                    color="region", color_discrete_sequence=CATEGORICAL_PALETTE,
                    title="Revenue by city",
                )
                fig_region.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig_region, height=480), use_container_width=True)
            else:
                st.info("Waiting for transactions...")

        with col_cat:
            cat_df = warehouse.revenue_by_category(engine.conn)
            if len(cat_df) and cat_df["revenue"].sum() > 0:
                fig_cat = px.pie(
                    cat_df, names="category", values="revenue", hole=0.55,
                    color_discrete_sequence=CATEGORICAL_PALETTE, title="Revenue by category",
                )
                fig_cat.update_traces(textinfo="percent+label", textfont_size=11)
                fig_cat.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig_cat, height=380), use_container_width=True)
            else:
                st.info("Waiting for transactions...")


live_section()
