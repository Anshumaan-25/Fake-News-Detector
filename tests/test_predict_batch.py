import joblib, pathlib, pandas as pd

def test_batch_vectorize_and_predict_proba():
    p = pathlib.Path("models")
    vec = joblib.load(p / "tfidf_vectorizer.joblib")
    clf = joblib.load(p / "baseline_logreg.joblib")
    df = pd.DataFrame({"text": ["Free entry to win now!", "See you at 7pm"]})
    X = vec.transform(df["text"].astype(str).tolist())
    proba = clf.predict_proba(X)[:, 1]
    assert proba.min() >= 0.0 and proba.max() <= 1.0
