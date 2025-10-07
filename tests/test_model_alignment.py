# tests/test_model_alignment.py
import joblib, pathlib

def test_vectorizer_model_alignment():
    p = pathlib.Path("models")
    vec = joblib.load(p / "tfidf_vectorizer.joblib")
    clf = joblib.load(p / "baseline_logreg.joblib")
    n_features = len(vec.get_feature_names_out())
    assert clf.coef_.shape[1] == n_features
    assert "spam" in set(clf.classes_)
