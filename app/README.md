# Predictive Maintenance — Streamlit App + FastAPI Backend

## Architecture
```
Browser → Streamlit (app.py) → HTTP request → FastAPI (api.py) → Random Forest model → prediction
```
The model is served through a REST API, completely decoupled from the UI — this is how production
ML systems are structured (model server + separate client), not just a single monolithic script.

## Run locally (2 terminals)
```bash
# Terminal 1 — start the API
cd app
uvicorn api:app --reload --port 8000

# Terminal 2 — start the frontend
cd app
streamlit run app.py
```
Open http://localhost:8501 — the "Live Prediction" tab calls the API running on port 8000.

Test the API directly:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "Temperature": 95, "Pressure": 110, "Vibration": 1.8, "RPM": 2500,
  "Current": 16, "Voltage": 395, "Operating_Hours": 3200,
  "Maintenance_Count": 8, "Machine_Age": 7
}'
```

## Deploy live (free options)

**Option A — Single container (recommended, easiest): Hugging Face Spaces**
1. Create a free account at huggingface.co → New Space → SDK: **Docker**
2. Push this whole project folder (including `Dockerfile`, `app/`, `data/`, `outputs/`) to the Space's repo
3. Hugging Face builds the Dockerfile automatically and starts `start.sh`, which runs the FastAPI
   backend internally and exposes the Streamlit frontend publicly — one link, both API and UI integrated.
4. Your live demo link: `https://huggingface.co/spaces/<your-username>/<space-name>`

**Option B — Two separate free services**
1. Deploy `api.py` as a Web Service on **Render** (free tier) — start command:
   `uvicorn api:app --host 0.0.0.0 --port $PORT`
2. Deploy `app.py` on **Streamlit Community Cloud** (share.streamlit.io) — set an environment
   variable `PM_API_URL` to your Render API's public URL
3. This demonstrates true separate API + frontend deployment, useful if your teacher specifically
   wants to see two independently running services talking over the internet

## Files
- `api.py` — FastAPI backend (model serving)
- `app.py` — Streamlit frontend (dashboard + live prediction UI)
- `requirements.txt` — dependencies
- `Dockerfile` + `start.sh` — single-container deployment (project root)


Model integrity note: the deployed model uses only current sensor values and deterministic usage-derived features. It excludes target-derived failure-rate aggregates and full-history machine aggregates. For a true temporal deployment, add stateful rolling history and retrain with matching time-aware features.
