"""Tests du script de simulation de trafic (partie conversion des donnees)."""
import numpy as np
import pandas as pd

from scripts.simulate_traffic import rows_to_payloads
from src.api.scoring import EXAMPLE_TRANSACTION


def _frame():
    row = dict(EXAMPLE_TRANSACTION, fraud_bool=0, month=6)
    return pd.DataFrame([row])


def test_target_and_month_are_removed():
    payload = rows_to_payloads(_frame())[0]
    assert "fraud_bool" not in payload
    assert "month" not in payload
    assert set(payload) == set(EXAMPLE_TRANSACTION)


def test_nan_becomes_none():
    df = _frame()
    df["bank_months_count"] = np.nan
    assert rows_to_payloads(df)[0]["bank_months_count"] is None


def test_values_are_native_python_types():
    """Necessaire pour que la requete soit serialisable en JSON."""
    payload = rows_to_payloads(_frame())[0]
    assert type(payload["income"]) is float
    assert type(payload["email_is_free"]) is int
    assert type(payload["payment_type"]) is str