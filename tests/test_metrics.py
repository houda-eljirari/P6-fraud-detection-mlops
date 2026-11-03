"""Tests des metriques Prometheus exposees par l'API."""
import joblib
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from src.api.main import create_app
from src.api.scoring import EXAMPLE_TRANSACTION
from tests.helpers import ConstantModel


def _value(name, labels=None):
    """Valeur courante d'une serie (0 si elle n'existe pas encore)."""
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def _model_file(tmp_path, probability):
    path = tmp_path / "constant.joblib"
    joblib.dump(ConstantModel(probability), path)
    return str(path)


def test_metrics_endpoint_uses_prometheus_text_format(tmp_path):
    app = create_app(model_path=_model_file(tmp_path, 0.9))
    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "fraud_predictions_total" in body
    assert "http_request_duration_seconds" in body
    assert "model_loaded" in body


def test_prediction_updates_business_metrics(tmp_path):
    app = create_app(model_path=_model_file(tmp_path, 0.9), threshold=0.5)
    fraud = {"decision": "fraud"}
    legit = {"decision": "legit"}
    before = (
        _value("fraud_predictions_total", fraud),
        _value("fraud_predictions_total", legit),
        _value("fraud_score_count"),
    )
    with TestClient(app) as client:
        client.post("/predict", json=EXAMPLE_TRANSACTION)
    assert _value("fraud_predictions_total", fraud) == before[0] + 1
    assert _value("fraud_predictions_total", legit) == before[1]
    assert _value("fraud_score_count") == before[2] + 1


def test_low_score_is_counted_as_legit(tmp_path):
    app = create_app(model_path=_model_file(tmp_path, 0.1), threshold=0.5)
    legit = {"decision": "legit"}
    before = _value("fraud_predictions_total", legit)
    with TestClient(app) as client:
        client.post("/predict", json=EXAMPLE_TRANSACTION)
    assert _value("fraud_predictions_total", legit) == before + 1


def test_requests_are_counted_by_route_and_status(tmp_path):
    app = create_app(model_path=_model_file(tmp_path, 0.9))
    ok = {"method": "POST", "path": "/predict", "status": "200"}
    invalid = {"method": "POST", "path": "/predict", "status": "422"}
    before_ok = _value("http_requests_total", ok)
    before_invalid = _value("http_requests_total", invalid)
    with TestClient(app) as client:
        client.post("/predict", json=EXAMPLE_TRANSACTION)
        client.post("/predict", json=dict(EXAMPLE_TRANSACTION, income=5))
    assert _value("http_requests_total", ok) == before_ok + 1
    assert _value("http_requests_total", invalid) == before_invalid + 1


def test_technical_endpoints_are_not_counted(tmp_path):
    app = create_app(model_path=_model_file(tmp_path, 0.9))
    health = {"method": "GET", "path": "/health", "status": "200"}
    before = _value("http_requests_total", health)
    with TestClient(app) as client:
        client.get("/health")
        client.get("/metrics")
    assert _value("http_requests_total", health) == before


def test_model_loaded_gauge_follows_model_availability(tmp_path):
    with TestClient(create_app(model_path=_model_file(tmp_path, 0.9))):
        assert _value("model_loaded") == 1.0
    with TestClient(create_app(model_path=str(tmp_path / "absent.joblib"))):
        assert _value("model_loaded") == 0.0