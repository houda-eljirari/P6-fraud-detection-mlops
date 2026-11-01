"""Outils partages par les tests : un petit modele entraine sur des donnees synthetiques."""
import numpy as np
import pandas as pd

from src.api.scoring import CATEGORICAL_FEATURES, EXAMPLE_TRANSACTION
from src.models.train import build_model

FEATURES = list(EXAMPLE_TRANSACTION)
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL_FEATURES]
PARAMS = {"C": 1.0, "max_iter": 200, "class_weight": "balanced", "seed": 42}


def train_dummy_model():
    """Modele leger avec les memes colonnes que l'API, pour tester sans Base.csv."""
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC})
    for col in CATEGORICAL_FEATURES:
        df[col] = rng.choice(["A", "B"], n)
    df.loc[rng.random(n) < 0.3, "prev_address_months_count"] = np.nan
    y = (rng.random(n) < 0.1).astype(int)
    y[:20] = 1  # garantit les deux classes
    return build_model(NUMERIC, CATEGORICAL_FEATURES, PARAMS).fit(df, y)


class ConstantModel:
    """Faux modele : renvoie toujours la meme probabilite de fraude."""

    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, X):
        return np.array([[1 - self.probability, self.probability]] * len(X))