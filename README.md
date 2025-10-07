# Fake-News/Spam Detector

A simple TF‑IDF + Logistic Regression baseline with CLI tools and a FastAPI service for single and batch predictions. Artifacts are persisted with joblib for reproducible inference across machines.

## Setup
- Create and activate a virtual environment, then install dependencies:
  - python -m pip install -r requirements.txt

## Artifacts
- The project expects:
  - models/tfidf_vectorizer.joblib
  - models/baseline_logreg.joblib
  - models/threshold.json (e.g., {"threshold": 0.5})

## Single prediction (CLI)
- python src/predict.py --text "Free entry to win now!"
- Optional override: python src/predict.py --text "See you at 7pm" --threshold 0.8

## Batch prediction (CLI)
- python src/predict_batch.py --input_csv data/processed/sample_messages.csv --text_col text --output_csv data/processed/predictions.csv
- Override per-run threshold: add --threshold 0.8

## Set default threshold
- python src/set_threshold.py --value 0.5

## Feature inspection
- python src/analyze_features.py --top_n 20
- Output CSV: data/processed/top_features.csv

## Precision–Recall evaluation
- python -m pip install matplotlib
- python src/precision_recall.py --input_csv data/processed/val_labeled.csv --text_col text --label_col label --pos_label spam
- Outputs:
  - data/processed/pr_curve.csv
  - data/processed/pr_curve.png
  - average_precision in console

## Tests
- Run all tests: python -m pytest -q tests
- API tests use TestClient (no live server required).

## API (FastAPI)
- Start server: uvicorn src.app:app --reload
- Open interactive docs: http://127.0.0.1:8000/docs
- Endpoints:
  - GET /health
  - POST /predict  (body: {"text": "...", "threshold": optional float})
  - POST /predict-batch  (body: {"texts": ["...","..."], "threshold": optional float})
