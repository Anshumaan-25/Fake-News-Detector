# tests/test_predict_threshold.py
import joblib, pathlib
def test_predict_threshold_pipeline_loads():
    p = pathlib.Path("models")
    vec = joblib.load(p / "tfidf_vectorizer.joblib")
    clf = joblib.load(p / "baseline_logreg.joblib")
    X = vec.transform(["Claim your free reward"])
    proba = clf.predict_proba(X)[0, 1]
    assert 0.0 <= proba <= 1.0
