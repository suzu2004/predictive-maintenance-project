# ⚙️ Predictive Maintenance of Industrial Equipment

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/suzu2004/predictive-maintenance-project/actions/workflows/ci.yml/badge.svg)](https://github.com/suzu2004/predictive-maintenance-project/actions)

> Predicts industrial machine failure risk from IoT sensor data using a Random Forest classifier, served through a production-style **FastAPI + Streamlit** architecture.

---

## 🏗️ Architecture

```
IoT Sensor Data
      │
      ▼
┌─────────────────────────────────────────┐
│         Jupyter Notebooks (Pipeline)    │
│  01_data_cleaning → 02_eda →            │
│  03_feature_engineering → 04_training  │
└──────────────┬──────────────────────────┘
               │  outputs: model.pkl + CSVs
               ▼
┌──────────────────────┐     HTTP/JSON     ┌──────────────────────┐
│   FastAPI Backend    │◄──────────────────│  Streamlit Frontend  │
│   api.py :8000       │   POST /predict   │   app.py :8501       │
│                      │   GET  /machines  │                      │
│  - Input validation  │   GET  /metrics   │  - Fleet Overview    │
│  - Startup smoke test│   GET  /health    │  - Live Prediction   │
│  - CORS middleware   │                   │  - Machine Explorer  │
│  - /metrics endpoint │                   │  - Model Insights    │
└──────────────────────┘                   └──────────────────────┘
               │
               ▼
         Docker Container
         (Hugging Face Spaces / Render)
```

---

## 🎯 Business Problem

Unplanned machine failure interrupts production, increases maintenance costs, and reduces equipment life. This project uses historical IoT sensor data (Temperature, Pressure, Vibration, RPM, Current, Voltage) from a fleet of 40 industrial machines to predict failure — enabling **condition-based predictive maintenance** instead of fixed-schedule or reactive maintenance.

**Key questions answered:**
- Which machines show abnormal operating behavior?
- Which sensor values are most strongly associated with failure?
- Which machines currently have high failure risk?
- What maintenance action should be taken and when?

---

## 📊 Dataset

Synthetic industrial IoT dataset (C-MAPSS-style schema):

| Field | Type | Purpose |
|---|---|---|
| `Machine_ID` | Categorical | Identify equipment (M101–M140) |
| `Timestamp` | DateTime | Time-series (hourly, 90 days) |
| `Temperature`, `Pressure`, `Vibration`, `RPM`, `Current`, `Voltage` | Numeric | Sensor readings |
| `Operating_Hours` | Numeric | Cumulative usage |
| `Maintenance_Count` | Numeric | Maintenance history |
| `Machine_Age` | Numeric | Asset age (years) |
| `Failure` | Binary | Prediction target |

**Stats:** 40 machines · ~86,000 hourly readings · ~0.9% failure rate (realistic class imbalance)

Missing values, duplicates, and sensor-glitch outliers were injected deliberately so data cleaning is demonstrable.

---

## 🔬 ML Pipeline

| Phase | Notebook | What it does |
|---|---|---|
| Data Cleaning | `01_data_cleaning.ipynb` | Forward-fill missing values per machine, remove duplicates, fix sensor-glitch outliers |
| EDA | `02_eda.ipynb` | Failure rate by machine, sensor distributions, correlation heatmap, anomaly detection |
| Feature Engineering | `03_feature_engineering.ipynb` | Rolling mean/max/min, deviation features, operating-hour bands, age groups, maintenance frequency |
| Modeling & Evaluation | `04_model_training.ipynb` | Logistic Regression vs Decision Tree vs Random Forest; confusion matrix, ROC-AUC, feature importance |

---

## 📈 Model Results

Evaluated on held-out test machines (never seen during training — no data leakage):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.992 | 0.504 | **1.000** | 0.671 | **0.999** |
| Decision Tree | 0.993 | 0.518 | 0.840 | 0.641 | 0.918 |
| **Random Forest** ✅ | **0.995** | **0.606** | 0.911 | **0.728** | 0.998 |

**Selected model:** Random Forest — best F1 and precision balance. In predictive maintenance, high Recall (catch real failures) and reasonable Precision (avoid too many false alarms that waste inspection resources) are both important.

**Top features driving failure prediction:**
1. Vibration (38.4%)
2. Temperature (33.2%)
3. RPM (10.6%)
4. Current (6.7%)
5. Pressure (5.6%)

---

## 🚀 Quick Start

### Prerequisites
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r app/requirements.txt
```

### 1. Generate & Process Data
```bash
python generate_dataset.py           # creates data/raw_sensor_data.csv
jupyter notebook                     # run notebooks 01 → 04 in order
python export_powerbi.py             # refresh PredictiveMaintenance_PowerBI_v2.xlsx
```

### 2. Run Locally (Two Terminals)
```bash
# Terminal 1 — FastAPI backend
cd app
uvicorn api:app --reload --port 8000

# Terminal 2 — Streamlit frontend
cd app
streamlit run app.py
```

Open **http://localhost:8501** for the dashboard. The "Live Prediction" tab calls the API on port 8000.

### Upload a Sensor CSV

Use the dashboard's sidebar to switch between the bundled **Sample dataset** and **Uploaded CSV**. The uploader stays in the sidebar; after you choose a CSV, the dashboard processes it and updates the fleet overview and machine explorer. The file must include `Machine_ID`, `Timestamp`, `Temperature`, `Pressure`, `Vibration`, `RPM`, `Current`, `Voltage`, `Operating_Hours`, `Maintenance_Count`, and `Machine_Age`. A `Failure` column is optional.

The dashboard validates and cleans uploaded readings, fills missing sensor values within each machine's history, and sends them to the existing trained model in batches. It displays risk summaries and per-machine results and lets you download the predictions as a CSV. The upload runs inference with the trained model; it does not retrain it.

### 3. Test the API Directly
```bash
# Health check
curl http://localhost:8000/health

# Predict failure risk
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Temperature": 95, "Pressure": 110, "Vibration": 1.8, "RPM": 2500,
       "Current": 16, "Voltage": 395, "Operating_Hours": 3200,
       "Maintenance_Count": 8, "Machine_Age": 7}'

# Model evaluation metrics
curl http://localhost:8000/metrics
```

Interactive API docs: **http://localhost:8000/docs**

---

## 🐳 Docker Deployment

```bash
# Build and run
docker build -t predictive-maintenance .
docker run -p 8501:8501 -p 8000:8000 predictive-maintenance
```

### Deploy Free to Hugging Face Spaces (Recommended)
1. Create a free account at [huggingface.co](https://huggingface.co) → New Space → SDK: **Docker**
2. Push this repository to the Space's git remote
3. Hugging Face builds the Dockerfile automatically and exposes Streamlit publicly
4. Live demo: `https://huggingface.co/spaces/<your-username>/<space-name>`

### Deploy to Render (Two-service architecture)
1. Deploy `api.py` as a Web Service on [Render](https://render.com) — start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
2. Deploy `app.py` on [Streamlit Community Cloud](https://share.streamlit.io) — set env var `PM_API_URL` to your Render URL

---

## 🧪 Run Tests

```bash
pip install pytest
pytest tests/ -v
```

16 unit tests covering:
- `build_feature_row` column names, dtypes, operating-hour band logic, age group logic
- Edge cases (Machine_Age=0, Operating_Hours boundaries)
- Pydantic input validation (invalid sensor values rejected)

---

## 📁 Project Structure

```
Predictive_Maintenance_Project/
├── app/
│   ├── api.py                  # FastAPI backend (model serving)
│   ├── app.py                  # Streamlit frontend dashboard
│   ├── requirements.txt        # Python dependencies
│   └── start.sh                # Single-container startup script
├── data/                       # Generated CSV files (git-ignored)
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
├── outputs/                    # Model artifacts + plots (pkl git-ignored)
├── tests/
│   └── test_api.py             # Unit tests (pytest)
├── .github/workflows/ci.yml    # GitHub Actions CI
├── .streamlit/config.toml      # Dark theme configuration
├── .gitignore
├── Dockerfile
├── export_powerbi.py           # Power BI Excel exporter
├── generate_dataset.py         # Synthetic dataset generator
├── LICENSE
└── README.md
```

---

## 📊 Power BI Dashboard

**File:** `PredictiveMaintenance_PowerBI_v2.xlsx` (Power BI-ready data source)

| Sheet | Contents |
|---|---|
| `MachineSummary` | Latest sensor stats and risk status for every machine |
| `ScoredMachines` | Machines with a model prediction |
| `UnscoredMachines` | Machines without a model prediction |
| `SensorDailyTrend` | Daily fleet-level sensor averages and failure counts |
| `TopHighRiskReadings` | Up to 500 highest-risk predictions |
| `FeatureImportance` | Random Forest feature importances |
| `ModelComparison` | Accuracy, precision, recall, F1, and AUC for each model |

The Power BI Desktop report is `PredictiveMaintenance_PowerBI.pbix`; its imported data comes from the workbook above. Refresh the workbook after regenerating the data and model outputs by running `python export_powerbi.py` from the project root. It reads `data/cleaned_sensor_data.csv` and model outputs under `outputs/`. In Power BI Desktop, use **Home → Refresh** after regenerating the workbook.

The `.pbix` uses Import mode and a local workbook path. A Power BI Service scheduled refresh is not configured yet. To enable it after publishing, place the workbook in OneDrive for Business or SharePoint and update the report's source to that cloud file; alternatively, keep the file local and configure an on-premises data gateway. Then set the refresh schedule in the published semantic model's settings. Streamlit's API-connected live prediction is separate from Power BI refresh.

---

## 💡 Business Recommendations

- **High Risk machines** (probability > 0.5): Schedule maintenance within **24 hours**
- **Warning machines** (0.2–0.5): Schedule inspection within **7 days**
- **Healthy machines** (< 0.2): Continue regular maintenance schedule
- Track Operating_Hours bands — failure rate increases measurably with cumulative usage, supporting usage-based (not just calendar-based) maintenance scheduling
- Because Recall is prioritized over Precision, expect some false alarms — treat model output as a **prioritized inspection list**, not an automatic shutdown trigger

---

## 🛡️ Model Integrity

- The deployed model uses **17 features** derived only from current sensor values and deterministic usage statistics
- Target-derived features (`Failure_Rate` aggregates) are **excluded** from model inputs to prevent data leakage
- Rolling/deviation features from `features_final.csv` are available for analysis but are not used in the stateless API
- The dataset is **synthetic** — metrics are demonstration results, not proof of real-world factory performance

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Data & ML | Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit, Plotly |
| Deployment | Docker, Hugging Face Spaces / Render |
| BI | Power BI, openpyxl |
| Testing | pytest |
| CI/CD | GitHub Actions |
