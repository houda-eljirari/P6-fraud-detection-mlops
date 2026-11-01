"""Tests de la logique de scoring (sans FastAPI)."""
from pathlib import Path

import numpy as np
import yaml

from src.api.scoring import (
    EXAMPLE_TRANSACTION,
    MISSING_CODE_COLS,
    build_frame,
    score_transaction,
)
from tests.helpers import train_dummy_model


def test_missing_codes_match_params_yaml():
    """La liste de l'API doit rester alignee sur celle du pipeline."""
    params_path = Path(__file__).resolve().parents[1] / "params.yaml"
    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    assert MISSING_CODE_COLS == params["prepare"]["missing_value_cols"]


def test_build_frame_converts_minus_one_and_none_to_nan():
    payload = dict(EXAMPLE_TRANSACTION)
    payload["bank_months_count"] = None
    frame = build_frame(payload)

    assert np.isnan(frame.loc[0, "prev_address_months_count"])  # -1 -> NaN
    assert np.isnan(frame.loc[0, "bank_months_count"])  # None -> NaN
    assert frame.loc[0, "current_address_months_count"] == 25  # valeur conservee


def test_build_frame_keeps_minus_one_in_other_columns():
    payload = dict(EXAMPLE_TRANSACTION)
    payload["credit_risk_score"] = -1  # vraie valeur, pas un code de manque
    assert build_frame(payload).loc[0, "credit_risk_score"] == -1


def test_score_is_a_probability():
    model = train_dummy_model()
    score = score_transaction(model, EXAMPLE_TRANSACTION)
    assert 0.0 <= score <= 1.0


def test_minus_one_and_none_give_the_same_score():
    model = train_dummy_model()
    with_code = dict(EXAMPLE_TRANSACTION, prev_address_months_count=-1)
    with_none = dict(EXAMPLE_TRANSACTION, prev_address_months_count=None)
    assert score_transaction(model, with_code) == score_transaction(model, with_none)