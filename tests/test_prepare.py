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
            "train_months": [0, 1, 2],
            "test_months": [3],
            "missing_value_cols": ["prev_address_months_count"],
        }
    }
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params), encoding="utf-8")
    raw_dir = tmp_path / "data" / "raw" / "baf"
    raw_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "fraud_bool": [0, 1, 0, 0, 1, 0, 0, 0],
            "month": [0, 0, 1, 2, 3, 3, 4, 5],
            "prev_address_months_count": [-1, 12, 30, -1, 5, 8, 9, 10],
            # -1 est une vraie valeur ici : cette colonne ne doit PAS etre modifiee
            "credit_risk_score": [-1, 100, 120, 90, -1, 80, 70, 60],
        }
    )
    df.to_csv(raw_dir / "Base.csv", index=False)
    return df


def test_minus_one_replaced_only_in_listed_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)
    prepare.main()

    train = pd.read_csv(tmp_path / "data" / "processed" / "train.csv")
    # -1 -> valeur manquante dans la colonne listee
    assert train["prev_address_months_count"].isna().sum() == 2
    assert (train["prev_address_months_count"].dropna() >= 0).all()
    # colonne non listee : le -1 est conserve
    assert (train["credit_risk_score"] == -1).sum() == 1


def test_split_is_temporal_without_overlap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)
    prepare.main()

    train = pd.read_csv(tmp_path / "data" / "processed" / "train.csv")
    test = pd.read_csv(tmp_path / "data" / "processed" / "test.csv")

    assert set(train["month"]) <= {0, 1, 2}
    assert set(test["month"]) <= {3}
    # le test est strictement posterieur a l'entrainement
    assert train["month"].max() < test["month"].min()


def test_months_outside_the_split_are_dropped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)
    prepare.main()

    train = pd.read_csv(tmp_path / "data" / "processed" / "train.csv")
    test = pd.read_csv(tmp_path / "data" / "processed" / "test.csv")
    # mois 4 et 5 absents des deux jeux : 8 lignes - 2 = 6
    assert len(train) + len(test) == 6


def test_target_column_is_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)
    prepare.main()

    train = pd.read_csv(tmp_path / "data" / "processed" / "train.csv")
    assert "fraud_bool" in train.columns
    assert set(train["fraud_bool"]) <= {0, 1}
    assert not np.isnan(train["fraud_bool"]).any()
