# src/precision_recall.py
import argparse
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="models", help="Folder with vectorizer/model/threshold.json")
    ap.add_argument("--input_csv", type=str, required=True, help="CSV with labeled texts")
    ap.add_argument("--text_col", type=str, default="text", help="Column containing message text")
    ap.add_argument("--label_col", type=str, default="label", help="Column with ground-truth labels")
    ap.add_argument("--pos_label", type=str, default="spam", help="Which label is the positive class")
    ap.add_argument("--out_csv", type=str, default="data/processed/pr_curve.csv", help="Where to write PR points")
    ap.add_argument("--out_png", type=str, default="data/processed/pr_curve.png", help="Where to save PR plot")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    vec = joblib.load(model_dir / "tfidf_vectorizer.joblib")
    clf = joblib.load(model_dir / "baseline_logreg.joblib")

    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column '{args.text_col}' in {args.input_csv}; have {list(df.columns)}")
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column '{args.label_col}' in {args.input_csv}; have {list(df.columns)}")

    # Map labels to binary: pos_label -> 1, everything else -> 0
    y_true = (df[args.label_col].astype(str) == args.pos_label).astype(int).to_numpy()

    X = vec.transform(df[args.text_col].astype(str).tolist())
    y_score = clf.predict_proba(X)[:, 1]

    # Compute precision-recall pairs and Average Precision
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap_score = float(average_precision_score(y_true, y_score))

    # thresholds has length n-1; align with precision/recall by dropping the last point
    if len(thresholds) > 0:
        pr_df = pd.DataFrame({
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        })
    else:
        pr_df = pd.DataFrame({"threshold": [], "precision": [], "recall": []})

    # Ensure output directories exist
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)

    # Save CSV
    pr_df.to_csv(args.out_csv, index=False)

    # Plot PR curve
    plt.figure(figsize=(6, 5))
    plt.step(recall, precision, where="post", label=f"AP={ap_score:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(json.dumps({
        "points": len(pr_df),
        "average_precision": ap_score,
        "csv": str(args.out_csv),
        "png": str(args.out_png),
    }, indent=2))


if __name__ == "__main__":
    main()
