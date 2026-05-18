@echo off
echo ----------------------------------------------------
echo Explainable AI-Based Skin Lesion Segmentation
echo ----------------------------------------------------

echo Starting FastAPI Backend Architecture...
start cmd /k "title Skin Cancer AI Backend && python -m uvicorn backend.main:app --reload --port 8000"

echo Patiently waiting for AI Backend to initialize...
timeout /t 4 /nobreak > NUL

echo Starting Streamlit Presentation Dashboard...
start cmd /k "title Skin Cancer Presenter UI && streamlit run frontend\app.py"

echo Services launched! You can close this window.
