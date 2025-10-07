import argparse, joblib
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="models")
    ap.add_argument("--top_n", type=int, default=20)
    ap.add_argument("--out_csv", type=str, default="data/processed/top_features.csv")
    args = ap.parse_args()

    vec = joblib.load(Path(args.model_dir) / "tfidf_vectorizer.joblib")
    clf = joblib.load(Path(args.model_dir) / "baseline_logreg.joblib")

    feature_names = vec.get_feature_names_out()  # feature names aligned to columns
    classes = clf.classes_
    if "spam" not in classes:
        raise ValueError(f'"spam" not in classes_: {classes}')

    # Binary LR: coef_.shape == (1, n_features), tied to classes_[1] by convention
    base_coefs = clf.coef_[0]

    # If classes_[1] is "spam", base_coefs already point toward spam when positive
    # If classes_[0] is "spam", invert sign so positives indicate spam
    if classes[1] == "spam":
        spam_coefs = base_coefs
    else:
        spam_coefs = -base_coefs

    # Top spam indicators (largest positive weights) and top ham indicators (most negative)
    top_pos_idx = np.argsort(spam_coefs)[-args.top_n:][::-1]
    top_neg_idx = np.argsort(spam_coefs)[:args.top_n]

    rows = []
    for i in top_pos_idx:
        rows.append({"token": feature_names[i], "weight": float(spam_coefs[i]), "direction": "spam"})
    for i in top_neg_idx:
        rows.append({"token": feature_names[i], "weight": float(spam_coefs[i]), "direction": "ham"})

    df = pd.DataFrame(rows, columns=["token", "weight", "direction"])
    print(df)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote top features to {args.out_csv}")

if __name__ == "__main__":
    main()
