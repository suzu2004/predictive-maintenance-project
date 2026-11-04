"""
app.py
-------
Streamlit frontend for the Predictive Maintenance project.

Run locally (two terminals):
    Terminal 1 (from app/):  uvicorn api:app --reload --port 8000
    Terminal 2 (from app/):  streamlit run app.py
"""

import hashlib
import io
import re

import numpy as np
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

UPLOAD_SENSOR_COLUMNS = [
    "Temperature", "Pressure", "Vibration", "RPM", "Current", "Voltage",
    "Operating_Hours", "Maintenance_Count", "Machine_Age",
]
UPLOAD_REQUIRED_COLUMNS = ["Machine_ID", "Timestamp", *UPLOAD_SENSOR_COLUMNS]
UPLOAD_VALUE_LIMITS = {
    "Temperature": (0, 500), "Pressure": (0, 500), "Vibration": (0, 20),
    "RPM": (0, 10_000), "Current": (0, 100), "Voltage": (0, 1_000),
    "Operating_Hours": (0, 100_000), "Maintenance_Count": (0, 1_000),
    "Machine_Age": (0, 50),
}
UPLOAD_BATCH_SIZE = 500
MAX_UPLOAD_ROWS = 100_000


@st.cache_data(show_spinner=False)
def get_sample_csv_bytes():
    """Return the cleaned sample data using only columns accepted by the uploader."""
    sample_path = os.path.join(BASE_DIR, "data", "cleaned_sensor_data.csv")
    if not os.path.exists(sample_path):
        sample_path = os.path.join(BASE_DIR, "data", "features_final.csv")
    sample = pd.read_csv(sample_path, usecols=[*UPLOAD_REQUIRED_COLUMNS, "Failure"])
    return sample.to_csv(index=False).encode("utf-8")


def prepare_uploaded_csv(file_bytes):
    """Read and clean a sensor CSV using the model's expected input schema."""
    frame = pd.read_csv(io.BytesIO(file_bytes))
    if frame.empty:
        raise ValueError("The CSV has headers but no data rows.")

    # Accept common header variations such as "machine id" and "machine_id".
    canonical_names = {re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"): name
                       for name in [*UPLOAD_REQUIRED_COLUMNS, "Failure"]}
    renamed = {}
    for column in frame.columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        renamed[column] = canonical_names.get(normalized, str(column).strip())
    if len(set(renamed.values())) != len(renamed):
        raise ValueError("The CSV has duplicate columns after normalizing their names.")
    frame = frame.rename(columns=renamed)

    missing_columns = [name for name in UPLOAD_REQUIRED_COLUMNS if name not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
            + ". Include Machine_ID, Timestamp, the six sensor readings, "
              "Operating_Hours, Maintenance_Count, and Machine_Age."
        )
    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError(f"The CSV has {len(frame):,} rows; the limit is {MAX_UPLOAD_ROWS:,} rows.")

    stats = {"uploaded_rows": len(frame)}
    frame["Machine_ID"] = frame["Machine_ID"].astype("string").str.strip()
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    for column in UPLOAD_SENSOR_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

    valid_identity = (
        frame["Machine_ID"].notna() & frame["Machine_ID"].ne("") & frame["Timestamp"].notna()
    )
    stats["invalid_identity_rows"] = int((~valid_identity).sum())
    frame = frame.loc[valid_identity].copy()
    if frame.empty:
        raise ValueError("No rows have both a valid Machine_ID and Timestamp.")

    has_sensor_value = frame[UPLOAD_SENSOR_COLUMNS].notna().any(axis=1)
    stats["empty_sensor_rows"] = int((~has_sensor_value).sum())
    frame = frame.loc[has_sensor_value].copy()
    if frame.empty:
        raise ValueError("No usable sensor readings were found in the CSV.")

    before_dedup = len(frame)
    frame = frame.drop_duplicates().sort_values(["Machine_ID", "Timestamp"]).reset_index(drop=True)
    stats["duplicate_rows"] = before_dedup - len(frame)

    missing_before = int(frame[UPLOAD_SENSOR_COLUMNS].isna().sum().sum())
    for column in UPLOAD_SENSOR_COLUMNS:
        frame[column] = frame.groupby("Machine_ID", sort=False)[column].transform(
            lambda values: values.ffill().bfill()
        )
        if frame[column].isna().all():
            raise ValueError(f"Column {column} has no numeric values to process.")
        frame[column] = frame[column].fillna(frame[column].median())
    stats["imputed_sensor_values"] = missing_before

    for column in ("Maintenance_Count", "Machine_Age"):
        frame[column] = frame[column].round()

    invalid_ranges = []
    for column, (minimum, maximum) in UPLOAD_VALUE_LIMITS.items():
        invalid_count = int((~frame[column].between(minimum, maximum)).sum())
        if invalid_count:
            invalid_ranges.append(f"{column}: {invalid_count} row(s) outside {minimum}–{maximum}")
    if invalid_ranges:
        raise ValueError("Values are outside the supported model ranges: " + "; ".join(invalid_ranges))

    if "Failure" in frame.columns:
        frame["Failure"] = pd.to_numeric(frame["Failure"], errors="coerce")

    stats["processed_rows"] = len(frame)
    stats["machines"] = int(frame["Machine_ID"].nunique())
    return frame, stats


def score_uploaded_csv(frame):
    """Get predictions through the batch API and align them to uploaded rows."""
    batch_results = []
    for start in range(0, len(frame), UPLOAD_BATCH_SIZE):
        batch = frame.iloc[start:start + UPLOAD_BATCH_SIZE]
        readings = batch[UPLOAD_SENSOR_COLUMNS].copy()
        readings.insert(0, "Timestamp", batch["Timestamp"].astype(str))
        readings.insert(0, "Machine_ID", batch["Machine_ID"])
        readings["Maintenance_Count"] = readings["Maintenance_Count"].astype(int)
        readings["Machine_Age"] = readings["Machine_Age"].astype(int)
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"readings": readings.to_dict(orient="records")},
            timeout=60,
        )
        response.raise_for_status()
        batch_results.extend(response.json())

    predictions = pd.DataFrame(batch_results).rename(columns={
        "failure_probability": "Failure_Probability",
        "predicted_failure": "Predicted_Failure",
        "risk_status": "Risk_Status",
    })
    predictions["Predicted_Failure"] = predictions["Predicted_Failure"].astype(int)
    predictions.insert(0, "Timestamp", frame["Timestamp"].reset_index(drop=True))
    predictions.insert(0, "Machine_ID", frame["Machine_ID"].reset_index(drop=True))
    return predictions

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
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111c2e, #0b1220);
        border-right: 1px solid rgba(148, 163, 184, .16);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #c3cfe0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(24, 39, 60, .88);
        border: 1px dashed rgba(148, 163, 184, .42);
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color: #dce6f5 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        border-color: rgba(242, 140, 69, .7);
        background: rgba(242, 140, 69, .12);
    }
    [data-testid="stDownloadButton"] button,
    [data-testid="stSidebar"] .stDownloadButton button,
    [data-testid="stSidebar"] button[kind="primary"] {
        min-height: 2.8rem;
        border: 1px solid #f3a15f !important;
        border-radius: 11px;
        background: linear-gradient(110deg, #e97836, #f3a15f) !important;
        color: #1c1720 !important;
        font-size: 1rem;
        font-weight: 750;
        box-shadow: 0 8px 22px rgba(233, 120, 54, .18);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    [data-testid="stDownloadButton"] button:hover,
    [data-testid="stSidebar"] .stDownloadButton button:hover,
    [data-testid="stSidebar"] button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 25px rgba(233, 120, 54, .28);
    }
    [data-testid="stDownloadButton"] button *,
    [data-testid="stSidebar"] .stDownloadButton button * {
        color: #1c1720 !important;
        fill: #1c1720 !important;
        font-weight: 750;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        font-size: .96rem;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: rgba(24, 39, 60, .92);
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
    [data-testid="stTabs"] button[data-baseweb="tab"] { height: 3rem; border-radius: 12px 12px 0 0; color: #aebbd0; font-size: .94rem; font-weight: 650; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #ffd0aa; border-bottom: 2px solid var(--pm-accent); background: rgba(242, 140, 69, .08); }
    [data-testid="stMetric"] { padding: 1rem 1.1rem; border: 1px solid var(--pm-line); border-radius: 16px; background: linear-gradient(145deg, rgba(24, 39, 60, .88), rgba(16, 27, 44, .9)); box-shadow: 0 10px 24px rgba(0, 0, 0, .12); }
    [data-testid="stMetricLabel"] { color: #bdcbe0; font-size: .92rem; }
    [data-testid="stMetricValue"] { color: var(--pm-ink); font-weight: 720; }
    [data-testid="stAlert"] { border-radius: 14px; }
    [data-testid="stCaptionContainer"] p { font-size: .92rem; line-height: 1.5; }
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


sample_df, sample_preds, fi, metrics = load_data()


def _select_uploaded_dataset():
    st.session_state["dashboard_data_source"] = "Uploaded CSV"


st.sidebar.markdown("## Dashboard data")
st.sidebar.markdown("**Sample dataset**")
st.sidebar.caption(
    f"Bundled project data · {sample_df['Machine_ID'].nunique():,} machines · "
    f"{len(sample_df):,} sensor readings"
)
st.sidebar.download_button(
    "Download sample CSV",
    data=get_sample_csv_bytes(),
    file_name="sample_sensor_dataset.csv",
    mime="text/csv",
    type="primary",
    use_container_width=True,
)
st.sidebar.caption("Clean sample with all required columns and an optional Failure label.")
dataset_source = st.sidebar.radio(
    "Show in dashboard",
    ["Sample dataset", "Uploaded CSV"],
    key="dashboard_data_source",
)
uploaded_csv = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"],
    accept_multiple_files=False,
    help="CSV must contain machine IDs, timestamps, and the sensor and usage columns shown in the dashboard.",
    on_change=_select_uploaded_dataset,
    key="dashboard_csv_uploader",
)

df, preds = sample_df, sample_preds
has_observed_failures = True
active_dataset_label = "Sample dataset"
upload_error = None

if dataset_source == "Uploaded CSV":
    if uploaded_csv is None:
        st.sidebar.info("Upload a CSV below to switch the dashboard to your data.")
    else:
        uploaded_bytes = uploaded_csv.getvalue()
        upload_id = hashlib.sha256(uploaded_bytes).hexdigest()
        cached_upload = st.session_state.get("processed_dashboard_upload")
        if not cached_upload or cached_upload["upload_id"] != upload_id:
            try:
                with st.spinner("Preparing CSV and generating dashboard predictions…"):
                    upload_df, _ = prepare_uploaded_csv(uploaded_bytes)
                    upload_preds = score_uploaded_csv(upload_df)
                    actual_labels = (
                        "Failure" in upload_df.columns and upload_df["Failure"].notna().any()
                    )
                    if not actual_labels:
                        upload_df["Failure"] = upload_preds["Predicted_Failure"].to_numpy()
                    st.session_state["processed_dashboard_upload"] = {
                        "upload_id": upload_id,
                        "data": upload_df,
                        "predictions": upload_preds,
                        "has_observed_failures": bool(actual_labels),
                    }
                cached_upload = st.session_state["processed_dashboard_upload"]
            except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
                upload_error = f"CSV could not be prepared: {exc}"
            except requests.RequestException as exc:
                upload_error = f"Could not reach the prediction API at {API_URL}: {exc}"
            except Exception as exc:
                upload_error = f"The uploaded CSV could not be processed: {exc}"

        if upload_error:
            st.sidebar.error(upload_error)
            st.sidebar.caption("The dashboard is showing the bundled sample dataset.")
        elif cached_upload and cached_upload["upload_id"] == upload_id:
            df = cached_upload["data"]
            preds = cached_upload["predictions"]
            has_observed_failures = cached_upload["has_observed_failures"]
            active_dataset_label = uploaded_csv.name


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

st.caption(f"Active dashboard data: {active_dataset_label}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Fleet Overview", "🔮 Live Prediction (API)", "🏭 Machine Explorer", "📈 Model Insights"]
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Fleet Overview
# ═════════════════════════════════════════════════════════════════════════════
with tab1:

    st.subheader("Fleet overview")
    failure_metric_name = "Observed Failure Rate" if has_observed_failures else "Predicted Failure Rate"
    failure_event_name = "Observed" if has_observed_failures else "Predicted"
    st.caption(
        f"A current snapshot of {active_dataset_label} with {failure_event_name.lower()} "
        "failure activity and maintenance priorities."
    )

    # ── KPI metrics ──────────────────────────────────────────────────────────
    total_machines  = df["Machine_ID"].nunique()
    total_readings  = len(df)
    failure_rate    = df["Failure"].mean() * 100
    high_risk_count = preds[preds["Failure_Probability"] > 0.5]["Machine_ID"].nunique()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Machines",          total_machines)
    k2.metric("Total Sensor Readings",   f"{total_readings:,}")
    k3.metric(failure_metric_name,         f"{failure_rate:.2f}%")
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
            title=f"Top 15 Machines by {failure_event_name} Failure Rate",
            labels={"Failure": f"{failure_event_name} Failure Rate", "Machine_ID": "Machine"},
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
            title=f"Daily {failure_event_name} Failure Events Across Fleet",
            labels={"Failure": f"{failure_event_name} Failure Events", "Timestamp": "Date"},
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
    m4.metric(failure_metric_name,  f"{mdf['Failure'].mean()*100:.2f}%")

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
