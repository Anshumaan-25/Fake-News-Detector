# train_baseline.py
import argparse, os, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from sklearn.metrics import confusion_matrix


def evaluate(X, y, model, pos_label="spam"):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, pos_label=pos_label, average="binary")
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
    else:
        s = model.decision_function(X).reshape(-1, 1)
        p = MinMaxScaler().fit_transform(s).ravel()
    y_bin = (y == pos_label).astype(int)
    auc = roc_auc_score(y_bin, p)
    rep = classification_report(y, y_pred, digits=4)
    return {"accuracy": acc, "f1": f1, "roc_auc": auc, "report": rep}, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", type=str, default="data/raw/SMSSpamCollection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.20)       # final test = 20%
    ap.add_argument("--val_frac_of_trainpool", type=float, default=0.125)  # makes 70/10/20
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--stop_words", type=str, default=None, choices=[None, "english"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--solver", type=str, default="liblinear")
    ap.add_argument("--max_iter", type=int, default=1000)
    ap.add_argument("--models_dir", type=str, default="models")
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--errors_out", type=str, default="data/processed/baseline_test_errors.csv")
    ap.add_argument("--metrics_out", type=str, default="models/baseline_metrics.json")
    ap.add_argument("--cm_out", type=str, default="data/processed/confusion_matrix.csv")
    args = ap.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)

    # Load dataset (tab-separated: label, text)
    df = pd.read_csv(args.input_path, sep="\t", header=None, names=["label", "text"])

    # Split: 80/20 (train pool/test), then 70/10/20 overall via 12.5% val from the 80% train pool
    train_pool, test_df = train_test_split(
        df, test_size=args.test_size, stratify=df["label"], random_state=args.seed, shuffle=True
    )
    train_df, val_df = train_test_split(
        train_pool, test_size=args.val_frac_of_trainpool, stratify=train_pool["label"],
        random_state=args.seed, shuffle=True
    )

    # Vectorize with TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        lowercase=True,
        strip_accents="unicode",
        stop_words=args.stop_words
    )
    X_tr = vectorizer.fit_transform(train_df["text"])
    X_va = vectorizer.transform(val_df["text"])
    X_te = vectorizer.transform(test_df["text"])
    y_tr, y_va, y_te = train_df["label"], val_df["label"], test_df["label"]

    # Model: Logistic Regression
    clf = LogisticRegression(solver=args.solver, C=args.C, max_iter=args.max_iter, random_state=args.seed)
    clf.fit(X_tr, y_tr)

    # Evaluate
    val_metrics, _ = evaluate(X_va, y_va, clf)
    test_metrics, y_hat_te = evaluate(X_te, y_te, clf)

    # Print metrics and reports
    print("VAL:", {k: round(v, 4) for k, v in val_metrics.items() if k != "report"})
    print(val_metrics["report"])
    print("TEST:", {k: round(v, 4) for k, v in test_metrics.items() if k != "report"})
    print(test_metrics["report"])

    # Confusion matrix (rows=true, cols=pred) with explicit label order
    cm = confusion_matrix(y_te, y_hat_te, labels=["ham", "spam"])
    cm_df = pd.DataFrame(cm, index=["true_ham", "true_spam"], columns=["pred_ham", "pred_spam"])
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm_df)
    cm_df.to_csv(args.cm_out, index=True)

    # Save misclassified test rows
    errs = test_df.loc[y_hat_te != y_te].copy()
    errs["pred"] = y_hat_te[y_hat_te != y_te]
    errs.to_csv(args.errors_out, index=False)

    # Save artifacts and metrics
    dump(vectorizer, os.path.join(args.models_dir, "tfidf_vectorizer.joblib"))
    dump(clf, os.path.join(args.models_dir, "baseline_logreg.joblib"))
    with open(args.metrics_out, "w") as f:
        json.dump({"val": {k: float(v) for k, v in val_metrics.items() if k != "report"},
                   "test": {k: float(v) for k, v in test_metrics.items() if k != "report"}}, f, indent=2)


if __name__ == "__main__":
    main()
