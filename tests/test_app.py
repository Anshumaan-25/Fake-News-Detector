from fastapi.testclient import TestClient
from src.app import app

def test_predict_basic():
    with TestClient(app) as client:
        r = client.post("/predict", json={"text": "Free entry to win now!"})
        assert r.status_code == 200
        body = r.json()
        assert "pred" in body and "confidence" in body and "threshold" in body
