# src/predict.py
import argparse, joblib, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="models")
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--threshold", type=float, default=None)  # None => load from JSON, else use provided
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    vec = joblib.load(model_dir / "tfidf_vectorizer.joblib")
    clf = joblib.load(model_dir / "baseline_logreg.joblib")

    # Resolve threshold (CLI > JSON > default 0.5)
    if args.threshold is None:
        thr_path = model_dir / "threshold.json"
        if thr_path.exists():
            with open(thr_path) as f:
                args.threshold = float(json.load(f).get("threshold", 0.5))
        else:
            args.threshold = 0.5

    X = vec.transform([args.text])
    proba = clf.predict_proba(X)[0, 1]  # probability of "spam"
    pred = "spam" if proba >= args.threshold else "ham"
    print({"pred": pred, "confidence": float(proba), "threshold": float(args.threshold)})

if __name__ == "__main__":
    main()
