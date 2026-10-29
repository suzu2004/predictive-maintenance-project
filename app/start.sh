#!/bin/bash
# Starts the FastAPI backend in the background, waits until it is ready,
# then starts the Streamlit frontend in the foreground.
# Used for single-container deployment (Hugging Face Spaces / Render) where
# only one process/port can be exposed publicly -- Streamlit is exposed;
# it talks to the API over localhost internally.

uvicorn api:app --host 0.0.0.0 --port 8000 &

# Wait for the API to be healthy before starting Streamlit.
# Replaces the fragile 'sleep 2' with a proper readiness loop.
echo "Waiting for API to become ready..."
until curl -sf http://localhost:8000/health > /dev/null; do
    sleep 1
done
echo "API is ready. Starting Streamlit..."

streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
