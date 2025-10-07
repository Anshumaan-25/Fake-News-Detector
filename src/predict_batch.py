# src/predict_batch.py
import argparse, joblib, json
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="models")
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument("--output_csv", type=str, default="predictions.csv")
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

    # Read input CSV and validate text column
    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in {args.input_csv}. Available: {list(df.columns)}")

    # Vectorize and score probabilities
    X = vec.transform(df[args.text_col].astype(str).tolist())
    proba = clf.predict_proba(X)[:, 1]
    pred = np.where(proba >= args.threshold, "spam", "ham")

    # Write output CSV with predictions
    out = df.copy()
    out["pred"] = pred
    out["confidence"] = proba
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out)} rows to {args.output_csv}")

if __name__ == "__main__":
    main()
