"""Integration tests for the FastAPI endpoint."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    with patch("mlflow.pyfunc.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_load.return_value = mock_model
        from src.api.main import app
        yield TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint(client):
    payload = {"features": {"feature_1": 1.0, "feature_2": 2.0}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
