"""Tests du modele : metriques utilitaires et validation du modele."""
import numpy as np
import pandas as pd

from src.models.train import build_model, recall_at_fpr

PARAMS = {"C": 1.0, "max_iter": 200, "class_weight": "balanced", "seed": 42}
NUMERIC = ["x1", "x2", "x_missing"]
CATEGORICAL = ["payment_type"]


def _synthetic_data(n=4000, seed=0):
    """Donnees ou la fraude depend reellement de x1 (signal apprenable)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x_missing": rng.normal(size=n),
            "payment_type": rng.choice(["AA", "AB", "AC"], n),
        }
    )
    df.loc[rng.random(n) < 0.3, "x_missing"] = np.nan
    proba = 1 / (1 + np.exp(-(2.5 * df["x1"] - 3.5)))
    y = (rng.random(n) < proba).astype(int)
    return df, pd.Series(y)


def test_recall_at_fpr_perfect_and_useless_scorers():
    y = np.array([0] * 95 + [1] * 5)
    perfect = np.array([0.0] * 95 + [1.0] * 5)
    assert recall_at_fpr(y, perfect, 0.05) == 1.0

    constant = np.zeros(100)
    # un score constant ne detecte rien de mieux que le hasard
    assert recall_at_fpr(y, constant, 0.05) <= 0.05 + 1e-9


def test_model_output_format():
    """Validation du format : une probabilite entre 0 et 1 par ligne."""
    X, y = _synthetic_data()
    model = build_model(NUMERIC, CATEGORICAL, PARAMS).fit(X, y)

    scores = model.predict_proba(X.head(50))
    assert scores.shape == (50, 2)
    assert np.all((scores >= 0) & (scores <= 1))
    assert np.allclose(scores.sum(axis=1), 1.0)


def test_model_performance_above_threshold():
    """Validation de la performance : le modele doit depasser un seuil minimal."""
    X, y = _synthetic_data()
    split = 3000
    model = build_model(NUMERIC, CATEGORICAL, PARAMS).fit(X[:split], y[:split])
    scores = model.predict_proba(X[split:])[:, 1]

    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y[split:], scores) > 0.75


def test_model_handles_missing_values_and_unseen_categories():
    """Cas limites : valeurs manquantes et categorie jamais vue a l'entrainement."""
    X, y = _synthetic_data()
    model = build_model(NUMERIC, CATEGORICAL, PARAMS).fit(X, y)

    edge = pd.DataFrame(
        {
            "x1": [0.0],
            "x2": [np.nan],
            "x_missing": [np.nan],
            "payment_type": ["NOUVELLE_CATEGORIE"],
        }
    )
    score = model.predict_proba(edge)[0, 1]
    assert 0.0 <= score <= 1.0
