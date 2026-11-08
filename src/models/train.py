
"""Entrainement et evaluation temporelle du modele de detection de fraude."""

import json
import os
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_params() -> dict:
    with open("params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def recall_at_fpr(y_true, scores, fpr_target: float) -> float:
    """Taux de fraudes detectees pour un taux de faux positifs fixe."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.interp(fpr_target, fpr, tpr))


def evaluate_model(model, X, y, split_name: str, fpr_target: float) -> dict:
    """Calcule les metriques d'evaluation pour un ensemble."""
    scores = model.predict_proba(X)[:, 1]

    return {
        f"{split_name}_pr_auc": float(average_precision_score(y, scores)),
        f"{split_name}_roc_auc": float(roc_auc_score(y, scores)),
        f"{split_name}_recall_at_{int(fpr_target * 100)}pct_fpr": (
            recall_at_fpr(y, scores, fpr_target)
        ),
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_model(numeric: list, categorical: list, p: dict) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    clf = LogisticRegression(
        C=p["C"],
        max_iter=p["max_iter"],
        class_weight=p["class_weight"],
        random_state=p["seed"],
    )

    return Pipeline([("prep", preprocess), ("clf", clf)])


def main() -> None:
    params = load_params()
    prep, p = params["prepare"], params["train"]

    target = prep["target"]
    month_col = prep["month_col"]

    # 1. Charger les trois ensembles temporels
    train = pd.read_csv("data/processed/train.csv")
    validation = pd.read_csv("data/processed/validation.csv")
    test = pd.read_csv("data/processed/test.csv")

    # 2. Separarer les variables explicatives et la cible
    # La colonne month sert au decoupage temporel, pas a la prediction.
    drop_cols = [target, month_col]

    X_train = train.drop(columns=drop_cols)
    y_train = train[target]

    X_validation = validation.drop(columns=drop_cols)
    y_validation = validation[target]

    X_test = test.drop(columns=drop_cols)
    y_test = test[target]

    # 3. Identifier les colonnes numeriques et categorielles
    categorical = X_train.select_dtypes(exclude="number").columns.tolist()
    numeric = [c for c in X_train.columns if c not in categorical]

    # 4. Construire et entrainer le modele sur TRAIN uniquement
    model = build_model(numeric, categorical, p)
    model.fit(X_train, y_train)

    # 5. Evaluer sur VALIDATION et TEST
    validation_metrics = evaluate_model(
        model, X_validation, y_validation, "validation", p["fpr_target"]
    )

    test_metrics = evaluate_model(
        model, X_test, y_test, "test", p["fpr_target"]
    )

    metrics = {
        **validation_metrics,
        **test_metrics,
    }

    # 6. Suivre l'experience dans MLflow
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    mlflow.set_experiment(p["mlflow_experiment"])

    with mlflow.start_run():
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "dataset": "BAF Base",
                "split_strategy": "temporal",
            }
        )

        mlflow.log_params(
            {
                "model": p["model"],
                "C": p["C"],
                "max_iter": p["max_iter"],
                "class_weight": p["class_weight"],
                "seed": p["seed"],
                "train_months": str(prep["train_months"]),
                "validation_months": str(prep["validation_months"]),
                "test_months": str(prep["test_months"]),
                "n_train": len(train),
                "n_validation": len(validation),
                "n_test": len(test),
            }
        )

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["numpy.dtype"],
        )

    # 7. Sauvegarder le modele et les metriques
    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    joblib.dump(model, "models/model.joblib")

    with open("metrics/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nMetriques de validation :")
    print(json.dumps(validation_metrics, indent=2))

    print("\nMetriques de test final :")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
