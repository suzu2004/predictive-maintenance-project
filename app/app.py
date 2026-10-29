"""
app.py
-------
Streamlit frontend for the Predictive Maintenance project.

Run locally (two terminals):
    Terminal 1 (from app/):  uvicorn api:app --reload --port 8000
    Terminal 2 (from app/):  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide",
    page_icon="⚙️",
    initial_sidebar_state="collapsed",
)

API_URL = os.environ.get("PM_API_URL", "http://localhost:8000")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.markdown(
    """
    <style>
    :root {
        --pm-ink: #eef4ff;
        --pm-muted: #99aac3;
        --pm-line: rgba(148, 163, 184, 0.16);
        --pm-panel: rgba(17, 28, 46, 0.78);
        --pm-accent: #f28c45;
        --pm-teal: #39c6b4;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(36, 91, 125, .22), transparent 34%),
            radial-gradient(ellipse at 92% 8%, rgba(242, 140, 69, .12), transparent 28%),
            #0b1220;
    }
    [data-testid="stHeader"] { background: rgba(11, 18, 32, .72); }
    .block-container { max-width: 1540px; padding-top: 1.65rem; padding-bottom: 3rem; }
    .pm-hero {
        display: flex; align-items: center; justify-content: space-between; gap: 1.5rem;
        padding: 1.55rem 1.8rem; margin: .2rem 0 1.35rem;
        border: 1px solid rgba(148, 163, 184, .2); border-radius: 22px;
        background: linear-gradient(118deg, rgba(22, 43, 64, .96), rgba(18, 29, 47, .92) 58%, rgba(55, 39, 34, .82));
        box-shadow: 0 18px 50px rgba(0, 0, 0, .2);
    }
    .pm-eyebrow { color: #f5a66d; font-size: .72rem; font-weight: 750; letter-spacing: .16em; text-transform: uppercase; }
    .pm-title { margin: .3rem 0 .25rem; color: var(--pm-ink); font-size: clamp(1.65rem, 3vw, 2.35rem); font-weight: 760; letter-spacing: -.04em; }
    .pm-subtitle { margin: 0; color: #b4c1d4; font-size: .98rem; }
    .pm-pill { white-space: nowrap; color: #d8f7f1; border: 1px solid rgba(57, 198, 180, .3); background: rgba(57, 198, 180, .1); border-radius: 999px; padding: .55rem .85rem; font-size: .82rem; }
    h1, h2, h3, h4 { color: #f1f5ff !important; }
    p, label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        color: #c3cfe0 !important;
        opacity: 1 !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .55rem; border-bottom: 1px solid var(--pm-line); }
    [data-testid="stTabs"] button[data-baseweb="tab"] { height: 3rem; border-radius: 12px 12px 0 0; color: #aebbd0; font-weight: 650; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #ffd0aa; border-bottom: 2px solid var(--pm-accent); background: rgba(242, 140, 69, .08); }
    [data-testid="stMetric"] { padding: 1rem 1.1rem; border: 1px solid var(--pm-line); border-radius: 16px; background: linear-gradient(145deg, rgba(24, 39, 60, .88), rgba(16, 27, 44, .9)); box-shadow: 0 10px 24px rgba(0, 0, 0, .12); }
    [data-testid="stMetricLabel"] { color: #bdcbe0; font-size: .84rem; }
    [data-testid="stMetricValue"] { color: var(--pm-ink); font-weight: 720; }
    [data-testid="stAlert"] { border-radius: 14px; }
    [data-testid="stPlotlyChart"] { border: 1px solid var(--pm-line); border-radius: 16px; overflow: hidden; background: rgba(16, 27, 44, .54); padding: .35rem; }
    [data-testid="stDataFrame"] { border: 1px solid var(--pm-line); border-radius: 14px; overflow: hidden; }
    .stButton > button[kind="primary"] { border: 0; border-radius: 11px; background: linear-gradient(110deg, #e97836, #f3a15f); color: #1c1720; font-weight: 750; box-shadow: 0 8px 22px rgba(233, 120, 54, .2); }
    @media (max-width: 760px) {
        .block-container { padding: 1rem .85rem 2rem; }
        .pm-hero { align-items: flex-start; flex-direction: column; padding: 1.2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading fleet data…")
def load_data():
    try:
        df = pd.read_csv(
            os.path.join(BASE_DIR, "data", "features_final.csv"),
            parse_dates=["Timestamp"],
        )
        preds = pd.read_csv(
            os.path.join(BASE_DIR, "outputs", "test_predictions.csv"),
            parse_dates=["Timestamp"],
        )
        fi = pd.read_csv(os.path.join(BASE_DIR, "outputs", "feature_importance.csv"))
        metrics = pd.read_csv(os.path.join(BASE_DIR, "outputs", "model_comparison.csv"))
        return df, preds, fi, metrics
    except FileNotFoundError as exc:
        st.error(
            f"❌ Required data file not found: {exc}. "
            "Run the data-generation and training notebooks first."
        )
        st.stop()


df, preds, fi, metrics = load_data()


def style_figure(fig, height=390):
    """Apply one restrained dark theme to every Plotly visualization."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#dce6f5", size=12),
        title=dict(
            x=0.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            font=dict(size=17, color="#eef4ff"),
        ),
        margin=dict(l=28, r=22, t=88, b=36),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.0,
            x=0.99,
            xanchor="right",
            font=dict(color="#cbd6e6", size=11),
            bgcolor="rgba(16, 27, 44, 0.72)",
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(148,163,184,.22)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,.12)", zeroline=False)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <section class="pm-hero">
      <div>
        <div class="pm-eyebrow">Reliability intelligence · Industrial IoT</div>
        <h1 class="pm-title">Predictive Maintenance</h1>
        <p class="pm-subtitle">Spot rising equipment risk early and focus maintenance where it matters.</p>
      </div>
      <div class="pm-pill">● &nbsp; Fleet analytics ready</div>
    </section>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Fleet Overview", "🔮 Live Prediction (API)", "🏭 Machine Explorer", "📈 Model Insights"]
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Fleet Overview
# ═════════════════════════════════════════════════════════════════════════════
with tab1:

    st.subheader("Fleet overview")
    st.caption("A current snapshot of machine health, failure activity, and maintenance priorities.")

    # ── KPI metrics ──────────────────────────────────────────────────────────
    total_machines  = df["Machine_ID"].nunique()
    total_readings  = len(df)
    failure_rate    = df["Failure"].mean() * 100
    high_risk_count = preds[preds["Failure_Probability"] > 0.5]["Machine_ID"].nunique()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Machines",          total_machines)
    k2.metric("Total Sensor Readings",   f"{total_readings:,}")
    k3.metric("Overall Failure Rate",    f"{failure_rate:.2f}%")
    k4.metric("Machines Flagged High Risk", high_risk_count)

    st.divider()

    # ── High-risk alert panel ─────────────────────────────────────────────────
    top_risk = (
        preds[preds["Failure_Probability"] > 0.5]
        .groupby("Machine_ID")["Failure_Probability"]
        .max()
        .sort_values(ascending=False)
        .head(5)
    )

    if not top_risk.empty:
        st.subheader("🚨 Top High-Risk Machines — Immediate Attention Required")
        alert_cols = st.columns(len(top_risk))
        for col, (machine_id, prob) in zip(alert_cols, top_risk.items()):
            col.error(f"**{machine_id}**\n\n{prob * 100:.1f}%\n\nFailure Probability")
    else:
        st.success("✅ No machines are currently flagged as High Risk.")

    st.divider()

    # ── Charts ───────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        failure_by_machine = (
            df.groupby("Machine_ID")["Failure"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        fig_bar = px.bar(
            failure_by_machine,
            x="Machine_ID",
            y="Failure",
            title="Top 15 Machines by Failure Rate",
            labels={"Failure": "Failure Rate", "Machine_ID": "Machine"},
            color="Failure",
            color_continuous_scale=["#330000", "#cc3300", "#ff6600"],
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
        )
        style_figure(fig_bar, height=410)
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        daily_failures = (
            df.set_index("Timestamp")
            .resample("D")["Failure"]
            .sum()
            .reset_index()
        )
        fig_line = px.line(
            daily_failures,
            x="Timestamp",
            y="Failure",
            title="Daily Failure Events Across Fleet",
            labels={"Failure": "Failure Events", "Timestamp": "Date"},
            color_discrete_sequence=["#ff6600"],
        )
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        style_figure(fig_line, height=410)
        st.plotly_chart(fig_line, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Prediction via API
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Live Failure Risk Prediction")
    st.caption(
        f"This form sends your inputs to the FastAPI backend at **{API_URL}/predict** "
        "and shows the live response."
    )

    # ── API health check ──────────────────────────────────────────────────────
    api_online = False
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        api_online = health.get("model_ready", False)
        if api_online:
            st.success(
                f"✅ API connected — model: {health['model']}, "
                f"features: {health['features']}"
            )
        else:
            st.warning("⚠️ API is reachable but the model is not ready.")
    except Exception:
        st.error(
            f"❌ Cannot reach API at {API_URL}. "
            "Start it with: `uvicorn api:app --reload --port 8000`"
        )

    # ── Sensor input sliders ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider("Temperature (°C)", 40.0, 150.0, 78.0, step=0.5)
        pressure    = st.slider("Pressure",          80.0, 170.0, 135.0, step=0.5)
        vibration   = st.slider("Vibration",          0.0,   3.0,   0.9, step=0.05)
    with col2:
        rpm         = st.slider("RPM",              2000.0, 3200.0, 2700.0, step=10.0)
        current     = st.slider("Current (A)",         5.0,   25.0,   12.0, step=0.5)
        voltage     = st.slider("Voltage (V)",        370.0,  440.0,  410.0, step=1.0)
    with col3:
        op_hours    = st.slider("Operating Hours",     0, 6000, 2200, step=50)
        maint_count = st.slider("Maintenance Count",   0,   15,    5)
        machine_age = st.slider("Machine Age (years)", 1,   12,    6)

    predict_btn = st.button(
        "🔍 Predict Failure Risk",
        type="primary",
        disabled=not api_online,
        help="Start the FastAPI backend first." if not api_online else "",
    )

    if predict_btn:
        payload = {
            "Temperature":      temperature,
            "Pressure":         pressure,
            "Vibration":        vibration,
            "RPM":              rpm,
            "Current":          current,
            "Voltage":          voltage,
            "Operating_Hours":  op_hours,
            "Maintenance_Count": maint_count,
            "Machine_Age":      machine_age,
        }
        try:
            resp   = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            result = resp.json()
            prob   = result["failure_probability"]
            status = result["risk_status"]

            # Gauge chart
            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                gauge_color = (
                    "#ff3b3b" if prob > 0.5
                    else ("#ffaa00" if prob > 0.2 else "#00c48c")
                )
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    title={"text": "Failure Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": gauge_color},
                        "steps": [
                            {"range": [0,  20], "color": "rgba(0,200,100,0.15)"},
                            {"range": [20, 50], "color": "rgba(255,200,0,0.15)"},
                            {"range": [50,100], "color": "rgba(255,59,59,0.2)"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 2},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                ))
                gauge.update_layout(
                    height=300,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#FAFAFA"),
                )
                style_figure(gauge, height=330)
                st.plotly_chart(gauge, use_container_width=True)

            # Status + recommendation
            with res_col2:
                if status == "High Risk":
                    st.error(f"🔴 **Status: {status}**\n\nFailure Probability\n\n# {prob*100:.1f}%")
                    st.error(
                        "🔧 **Maintenance Recommendation:** "
                        "Schedule maintenance within **24 hours**. "
                        "Sensor readings indicate imminent failure risk."
                    )
                elif status == "Warning":
                    st.warning(f"🟡 **Status: {status}**\n\nFailure Probability\n\n# {prob*100:.1f}%")
                    st.warning(
                        "⚠️ **Maintenance Recommendation:** "
                        "Schedule a preventive inspection within **7 days**. "
                        "Monitor sensor trends closely."
                    )
                else:
                    st.success(f"🟢 **Status: {status}**\n\nFailure Probability\n\n# {prob*100:.1f}%")
                    st.success(
                        "✅ **Maintenance Recommendation:** "
                        "Machine is operating normally. "
                        "Continue regular maintenance schedule."
                    )

        except Exception as exc:
            st.error(f"Prediction request failed: {exc}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Machine Explorer
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Machine explorer")
    st.caption("Inspect sensor history and operating context for an individual machine.")

    machine_ids = sorted(df["Machine_ID"].unique())
    selected    = st.selectbox("Select a machine", machine_ids)
    mdf         = df[df["Machine_ID"] == selected].sort_values("Timestamp")

    # KPI row — use .iloc[-1] to get the most-recent reading
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Operating Hours",   int(mdf["Operating_Hours"].max()))
    m2.metric("Machine Age (yr)",  int(mdf["Machine_Age"].iloc[-1]))
    m3.metric("Maintenance Count", int(mdf["Maintenance_Count"].iloc[-1]))
    m4.metric("Failure Rate",      f"{mdf['Failure'].mean()*100:.2f}%")

    # Temperature & Vibration over time
    fig_tv = px.line(
        mdf,
        x="Timestamp",
        y=["Temperature", "Vibration"],
        title=f"{selected} — Temperature & Vibration Over Time",
        labels={"value": "Value", "variable": "Sensor"},
        color_discrete_map={"Temperature": "#ff6600", "Vibration": "#3399ff"},
    )
    fig_tv.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    style_figure(fig_tv, height=400)
    st.plotly_chart(fig_tv, use_container_width=True)

    # Pressure & RPM over time
    fig_pr = px.line(
        mdf,
        x="Timestamp",
        y=["Pressure", "RPM"],
        title=f"{selected} — Pressure & RPM Over Time",
        labels={"value": "Value", "variable": "Sensor"},
        color_discrete_map={"Pressure": "#cc00cc", "RPM": "#00ccaa"},
    )
    fig_pr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    style_figure(fig_pr, height=400)
    st.plotly_chart(fig_pr, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Insights
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Model Comparison")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.subheader("Top Feature Importances (Random Forest)")
    fi_top = fi.sort_values("Importance", ascending=False).head(15)
    fig_fi = px.bar(
        fi_top.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Features Driving Failure Prediction",
        color="Importance",
        color_continuous_scale=["#330000", "#cc3300", "#ff6600"],
    )
    fig_fi.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        height=460,
    )
    style_figure(fig_fi, height=460)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.caption(
        "Model: RandomForestClassifier · scikit-learn · "
        "trained on machine-level train/test split to avoid data leakage."
    )
