"""Tests de l'API FastAPI (avec un petit modele synthetique)."""
import joblib
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.schemas import TransactionIn
from src.api.scoring import EXAMPLE_TRANSACTION
from tests.helpers import ConstantModel, train_dummy_model


def _model_file(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(train_dummy_model(), path)
    return str(path)


def test_schema_matches_example_transaction():
    assert set(TransactionIn.model_fields) == set(EXAMPLE_TRANSACTION)
    TransactionIn(**EXAMPLE_TRANSACTION)  # l'exemple doit etre valide


def test_health_is_ok_even_without_model(tmp_path):
    app = create_app(model_path=str(tmp_path / "absent.joblib"))
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_is_503_without_model(tmp_path):
    app = create_app(model_path=str(tmp_path / "absent.joblib"))
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json=EXAMPLE_TRANSACTION).status_code == 503


def test_ready_is_200_with_model(tmp_path):
    app = create_app(model_path=_model_file(tmp_path))
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200


def test_predict_returns_probability_and_decision(tmp_path):
    app = create_app(model_path=_model_file(tmp_path), threshold=0.5)
    with TestClient(app) as client:
        response = client.post("/predict", json=EXAMPLE_TRANSACTION)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert body["threshold"] == 0.5


def test_predict_accepts_minus_one_and_null(tmp_path):
    app = create_app(model_path=_model_file(tmp_path))
    with TestClient(app) as client:
        a = client.post("/predict", json=dict(EXAMPLE_TRANSACTION, bank_months_count=-1))
        b = client.post("/predict", json=dict(EXAMPLE_TRANSACTION, bank_months_count=None))
    assert a.status_code == b.status_code == 200
    assert a.json()["fraud_probability"] == b.json()["fraud_probability"]


def test_threshold_controls_the_decision(tmp_path):
    """Avec une probabilite connue (0.3), la decision depend uniquement du seuil."""
    path = tmp_path / "constant.joblib"
    joblib.dump(ConstantModel(0.3), path)

    def decision(threshold):
        app = create_app(model_path=str(path), threshold=threshold)
        with TestClient(app) as client:
            return client.post("/predict", json=EXAMPLE_TRANSACTION).json()["is_fraud"]

    assert decision(0.5) is False  # 0.3 < 0.5
    assert decision(0.2) is True  # 0.3 >= 0.2
    assert decision(0.3) is True  # le seuil est inclus


def test_invalid_input_is_rejected_with_422(tmp_path):
    app = create_app(model_path=_model_file(tmp_path))
    with TestClient(app) as client:
        out_of_range = client.post("/predict", json=dict(EXAMPLE_TRANSACTION, income=5))
        missing = {k: v for k, v in EXAMPLE_TRANSACTION.items() if k != "income"}
        missing_field = client.post("/predict", json=missing)
        extra = client.post("/predict", json=dict(EXAMPLE_TRANSACTION, inconnu=1))
    assert out_of_range.status_code == 422
    assert missing_field.status_code == 422
    assert extra.status_code == 422