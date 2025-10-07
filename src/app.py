# src/app.py
from fastapi import FastAPI
from pathlib import Path
import joblib, json
from contextlib import asynccontextmanager
from typing import List

# Import schemas from the new schemas file
from src.schemas import PredictIn, PredictOut, BatchIn, BatchOut

# Module-level defaults to avoid NameError
VEC = None
CLF = None
THR = 0.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    global VEC, CLF, THR
    model_dir = Path("models")
    VEC = joblib.load(model_dir / "tfidf_vectorizer.joblib")
    CLF = joblib.load(model_dir / "baseline_logreg.joblib")
    thr_path = model_dir / "threshold.json"
    if thr_path.exists():
        with open(thr_path) as f:
            THR = float(json.load(f).get("threshold", 0.5))
    yield
    # optional cleanup here


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOut)
def predict(item: PredictIn):
    thr = item.threshold if item.threshold is not None else THR
    X = VEC.transform([item.text])
    proba = float(CLF.predict_proba(X)[0, 1])
    pred = "spam" if proba >= thr else "ham"
    return {"pred": pred, "confidence": proba, "threshold": thr}


@app.post("/predict-batch", response_model=BatchOut)
def predict_batch(item: BatchIn):
    thr = item.threshold if item.threshold is not None else THR
    X = VEC.transform(item.texts)
    probs = CLF.predict_proba(X)[:, 1]
    preds = ["spam" if p >= thr else "ham" for p in probs]
    return {"pred": preds, "confidence": probs.tolist(), "threshold": thr}
