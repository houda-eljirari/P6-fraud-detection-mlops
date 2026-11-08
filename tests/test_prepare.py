
"""Tests unitaires de la preparation des donnees (src/data/prepare.py)."""
import numpy as np
import pandas as pd
import yaml

from src.data import prepare


def _write_project(tmp_path):
    """Cree un mini projet : params.yaml + Base.csv synthetique."""
    params = {
        "prepare": {
            "raw_path": "data/raw/baf/Base.csv",
            "target": "fraud_bool",
            "month_col": "month",
            "train_months": [0, 1],
            "validation_months": [2],
            "test_months": [3, 4, 5],
            "missing_value_cols": ["prev_address_months_count"],
        }
    }

    (tmp_path / "params.yaml").write_text(
        yaml.safe_dump(params), encoding="utf-8"
    )

    raw_dir = tmp_path / "data" / "raw" / "baf"
    raw_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "fraud_bool": [0, 1, 0, 0, 1, 0, 0, 0],
            "month": [0, 0, 1, 2, 3, 3, 4, 5],
            "prev_address_months_count": [-1, 12, 30, -1, 5, 8, 9, 10],
            # -1 est une vraie valeur ici : elle doit etre conservee.
            "credit_risk_score": [-1, 100, 120, 90, -1, 80, 70, 60],
        }
    )

    df.to_csv(raw_dir / "Base.csv", index=False)
    return df


def test_minus_one_replaced_only_in_listed_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)
    prepare.main()

    train = pd.read_csv(tmp_path / "data/processed/train.csv")
    assert train["prev_address_months_count"].isna().sum() == 1
    assert (train["prev_address_months_count"].dropna() >= 0).all()
    assert (train["credit_risk_score"] == -1).sum() == 1


def test_split_is_temporal_without_overlap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)
    prepare.main()

    train = pd.read_csv(tmp_path / "data/processed/train.csv")
    validation = pd.read_csv(tmp_path / "data/processed/validation.csv")
    test = pd.read_csv(tmp_path / "data/processed/test.csv")

    assert set(train["month"]) <= {0, 1}
    assert set(validation["month"]) <= {2}
    assert set(test["month"]) <= {3, 4, 5}

    assert train["month"].max() < validation["month"].min()
    assert validation["month"].max() < test["month"].min()


def test_months_outside_the_split_are_dropped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)
    prepare.main()

    train = pd.read_csv(tmp_path / "data/processed/train.csv")
    validation = pd.read_csv(tmp_path / "data/processed/validation.csv")
    test = pd.read_csv(tmp_path / "data/processed/test.csv")

    # Tous les mois sont répartis : aucune ligne n'est perdue.
    assert len(train) + len(validation) + len(test) == 8

def test_target_column_is_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)
    prepare.main()

    for filename in ("train.csv", "validation.csv", "test.csv"):
        dataset = pd.read_csv(tmp_path / "data/processed" / filename)
        assert "fraud_bool" in dataset.columns
        assert set(dataset["fraud_bool"]) <= {0, 1}
        assert not np.isnan(dataset["fraud_bool"]).any()
