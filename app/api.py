"""
api.py
-------
FastAPI backend that serves the trained Random Forest predictive-maintenance
model as a REST API. The Streamlit frontend (app.py) calls this API over
HTTP -- this is the "API integration" part of the assignment: the ML model
is decoupled from the UI, exactly how a real production ML system is built
(model server + separate client/frontend).

Run locally:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /health            -> simple liveness check (verifies model is functional)
    GET  /machines          -> list of machines with latest known stats (pre-computed)
    GET  /metrics           -> model evaluation scores from training pipeline
    POST /predict           -> predict failure risk from sensor readings
    POST /predict/batch     -> predict failure risk for up to 1,000 readings
"""

import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_COLS = [
    'Temperature', 'Pressure', 'Vibration', 'RPM', 'Current', 'Voltage',
    'Operating_Hours', 'Maintenance_Count', 'Machine_Age', 'Maintenance_Frequency',
    'OpHours_Band_1000-2000', 'OpHours_Band_2000-3000', 'OpHours_Band_3000-4000',
    'OpHours_Band_4000+', 'Age_Group_Mid(3-6y)', 'Age_Group_Old(6-9y)',
    'Age_Group_VeryOld(9y+)'
]

# ---- Module-level state (populated at startup via lifespan) ----
model = None
machines_cache: list = []
metrics_cache: list = []
model_ready: bool = False


def _load_resources():
    """Load model and pre-compute expensive summaries once at startup."""
    global model, machines_cache, metrics_cache, model_ready

    try:
        # Load model
        model_path = os.path.join(BASE_DIR, "outputs", "random_forest_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = joblib.load(model_path)

        # Load feature data and pre-compute the machines summary
        data_path = os.path.join(BASE_DIR, "data", "features_final.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Feature data not found: {data_path}")
        machine_summary = pd.read_csv(data_path, parse_dates=["Timestamp"])
        latest = machine_summary.sort_values("Timestamp").groupby("Machine_ID").tail(1)
        machines_cache = (
            latest[["Machine_ID", "Temperature", "Vibration", "Pressure", "RPM",
                    "Operating_Hours", "Machine_Age"]]
            .round(2)
            .to_dict(orient="records")
        )

        # Load model evaluation metrics
        metrics_path = os.path.join(BASE_DIR, "outputs", "model_comparison.csv")
        if os.path.exists(metrics_path):
            metrics_cache = pd.read_csv(metrics_path).to_dict(orient="records")

        # Smoke test — confirm model actually produces output before marking ready
        _dummy = pd.DataFrame([{
            'Temperature': 78.5, 'Pressure': 135.2, 'Vibration': 0.9,
            'RPM': 2700.0, 'Current': 12.4, 'Voltage': 410.0,
            'Operating_Hours': 2200.0, 'Maintenance_Count': 5, 'Machine_Age': 6,
            'Maintenance_Frequency': 5 / 6,
            'OpHours_Band_1000-2000': 0, 'OpHours_Band_2000-3000': 1,
            'OpHours_Band_3000-4000': 0, 'OpHours_Band_4000+': 0,
            'Age_Group_Mid(3-6y)': 0, 'Age_Group_Old(6-9y)': 1,
            'Age_Group_VeryOld(9y+)': 0,
        }], columns=FEATURE_COLS).astype(float)
        _ = model.predict_proba(_dummy)

        model_ready = True
        logger.info("✅ Model loaded and smoke-tested successfully. %d machines cached.", len(machines_cache))

    except FileNotFoundError as e:
        logger.error("❌ Startup failed — missing required file: %s", e)
        model_ready = False
    except Exception as e:
        logger.error("❌ Startup failed with unexpected error: %s", e, exc_info=True)
        model_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load resources on startup, clean up on shutdown."""
    _load_resources()
    yield
    logger.info("API shutting down.")


# ---- App definition ----
app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Predicts industrial machine failure risk from sensor readings "
        "using a trained Random Forest model. Part of a portfolio predictive "
        "maintenance project using FastAPI + Streamlit."
    ),
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "https://*.hf.space"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---- Pydantic models ----

class SensorReading(BaseModel):
    Temperature:      float = Field(..., ge=0.0,  le=500.0,   description="Sensor temperature reading (°C)", json_schema_extra={"example": 78.5})
    Pressure:         float = Field(..., ge=0.0,  le=500.0,   description="Sensor pressure reading",         json_schema_extra={"example": 135.2})
    Vibration:        float = Field(..., ge=0.0,  le=20.0,    description="Sensor vibration reading",        json_schema_extra={"example": 0.9})
    RPM:              float = Field(..., ge=0.0,  le=10000.0, description="Rotations per minute",            json_schema_extra={"example": 2700})
    Current:          float = Field(..., ge=0.0,  le=100.0,   description="Electrical current (A)",          json_schema_extra={"example": 12.4})
    Voltage:          float = Field(..., ge=0.0,  le=1000.0,  description="Electrical voltage (V)",          json_schema_extra={"example": 410})
    Operating_Hours:  float = Field(..., ge=0.0,  le=100000.0,description="Cumulative operating hours",      json_schema_extra={"example": 2200})
    Maintenance_Count:int   = Field(..., ge=0,    le=1000,    description="Number of past maintenance events",json_schema_extra={"example": 5})
    Machine_Age:      int   = Field(..., ge=0,    le=50,      description="Machine age in years",            json_schema_extra={"example": 6})


class PredictionResponse(BaseModel):
    failure_probability: float
    predicted_failure:   int
    risk_status:         str


class BatchPredictionRequest(BaseModel):
    readings: List[SensorReading] = Field(..., min_length=1, max_length=1000)


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status:      str
    model:       str
    features:    int
    model_ready: bool


# ---- Helper ----

def build_feature_row(reading: SensorReading) -> pd.DataFrame:
    """Build the same stateless feature set used by the deployed model.

    Only current sensor values and deterministic usage-derived features are
    included. This avoids using target labels or full-history aggregates at
    prediction time. A future stateful API can add historical rolling features
    after a matching time-aware training pipeline is implemented.
    """
    op_hours = float(reading.Operating_Hours)
    age = int(reading.Machine_Age)
    row = {
        'Temperature':          reading.Temperature,
        'Pressure':             reading.Pressure,
        'Vibration':            reading.Vibration,
        'RPM':                  reading.RPM,
        'Current':              reading.Current,
        'Voltage':              reading.Voltage,
        'Operating_Hours':      op_hours,
        'Maintenance_Count':    int(reading.Maintenance_Count),
        'Machine_Age':          age,
        'Maintenance_Frequency': int(reading.Maintenance_Count) / max(age, 1),
        'OpHours_Band_1000-2000': int(1000 < op_hours <= 2000),
        'OpHours_Band_2000-3000': int(2000 < op_hours <= 3000),
        'OpHours_Band_3000-4000': int(3000 < op_hours <= 4000),
        'OpHours_Band_4000+':     int(op_hours > 4000),
        'Age_Group_Mid(3-6y)':    int(3 < age <= 6),
        'Age_Group_Old(6-9y)':    int(6 < age <= 9),
        'Age_Group_VeryOld(9y+)': int(age > 9),
    }
    return pd.DataFrame([row], columns=FEATURE_COLS).astype(float)


# ---- Endpoints ----

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Liveness + readiness check. Returns model_ready=false if startup failed."""
    return {
        "status":      "ok" if model_ready else "degraded",
        "model":       "RandomForestClassifier",
        "features":    len(FEATURE_COLS),
        "model_ready": model_ready,
    }


@app.get("/machines", tags=["Data"])
def list_machines():
    """Return latest known sensor stats for every machine in the fleet.

    Pre-computed at startup — does not re-sort the full 86k-row dataset
    on every request.
    """
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not ready. Check server logs.")
    return machines_cache


@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    """Return model evaluation scores from the training pipeline."""
    return {"models": metrics_cache}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(reading: SensorReading):
    """Predict failure risk from a single set of sensor readings.

    Returns failure probability (0–1), binary prediction, and a human-readable
    risk status: Healthy / Warning / High Risk.
    """
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not ready. Check server logs.")
    try:
        X = build_feature_row(reading)
        proba = float(model.predict_proba(X)[0, 1])
        pred = int(proba >= 0.5)
        status = "High Risk" if proba > 0.5 else ("Warning" if proba > 0.2 else "Healthy")
        return {
            "failure_probability": round(proba, 4),
            "predicted_failure":   pred,
            "risk_status":         status,
        }
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=422,
            detail="Invalid input: could not process sensor readings."
        )


@app.post("/predict/batch", tags=["Inference"])
def predict_batch(request: BatchPredictionRequest):
    """Predict failure risk for a batch of up to 1,000 sensor readings."""
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not ready. Check server logs.")
    try:
        X = pd.concat(
            [build_feature_row(reading) for reading in request.readings],
            ignore_index=True,
        )
        probabilities = model.predict_proba(X)[:, 1]
        return [
            {
                "failure_probability": round(float(probability), 4),
                "predicted_failure": int(probability >= 0.5),
                "risk_status": (
                    "High Risk" if probability > 0.5
                    else ("Warning" if probability > 0.2 else "Healthy")
                ),
            }
            for probability in probabilities
        ]
    except Exception as e:
        logger.error("Batch prediction failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=422,
            detail="Invalid input: could not process the uploaded sensor readings.",
        )
