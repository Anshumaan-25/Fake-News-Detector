def test_sanity():
    assert True

def test_artifacts_load_and_predict():
    import joblib, pathlib
    p = pathlib.Path("models")
    vec = joblib.load(p / "tfidf_vectorizer.joblib")
    clf = joblib.load(p / "baseline_logreg.joblib")
    X = vec.transform(["Win a free prize now"])
    y = clf.predict(X)
    assert y.shape == (1,)
