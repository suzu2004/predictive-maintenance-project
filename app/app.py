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

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("⚙️ Predictive Maintenance Dashboard")
st.caption(
    "Industrial equipment failure prediction using IoT sensor data · "
    "Random Forest · FastAPI · Streamlit"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Fleet Overview", "🔮 Live Prediction (API)", "🏭 Machine Explorer", "📈 Model Insights"]
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Fleet Overview
# ═════════════════════════════════════════════════════════════════════════════
with tab1:

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
    st.plotly_chart(fig_fi, use_container_width=True)

    st.caption(
        "Model: RandomForestClassifier · scikit-learn · "
        "trained on machine-level train/test split to avoid data leakage."
    )
