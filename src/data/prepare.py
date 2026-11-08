
"""Preparation des donnees BAF : valeurs manquantes et split temporel."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_params() -> dict:
    with open("params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["prepare"]


def main() -> None:
    params = load_params()

    # 1. Charger les donnees brutes
    df = pd.read_csv(params["raw_path"])

    print(f"Dataset original : {len(df)} lignes")

    # 2. Traiter les codes -1 uniquement dans les colonnes configurees
    for col in params["missing_value_cols"]:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)

    # 3. Creer les trois ensembles temporels
    month_col = params["month_col"]

    splits = {
        "train": params["train_months"],
        "validation": params["validation_months"],
        "test": params["test_months"],
    }

    # Verifier que les mois ne se chevauchent pas
    all_months = [
        month
        for months in splits.values()
        for month in months
    ]

    if len(all_months) != len(set(all_months)):
        raise ValueError("Erreur : les mois des ensembles se chevauchent.")

    # Verifier que tous les mois du dataset sont couverts
    dataset_months = set(df[month_col].unique())
    configured_months = set(all_months)

    if dataset_months != configured_months:
        raise ValueError(
            f"Mois du dataset : {dataset_months}. "
            f"Mois configures : {configured_months}."
        )

    # 4. Generer les fichiers
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    target = params["target"]
    total_rows = 0

    for name, months in splits.items():
        split_df = df[df[month_col].isin(months)].copy()

        output_path = out_dir / f"{name}.csv"
        split_df.to_csv(output_path, index=False)

        total_rows += len(split_df)

        fraud_count = int(split_df[target].sum())
        fraud_rate = split_df[target].mean() * 100

        print(
            f"{name.upper()} : {len(split_df)} lignes | "
            f"{fraud_count} fraudes | "
            f"Taux de fraude : {fraud_rate:.3f}%"
        )

    # 5. Verifier la repartition complete
    if total_rows != len(df):
        raise ValueError(
            f"Erreur : {total_rows} lignes reparties "
            f"sur {len(df)} lignes originales."
        )

    print(f"\nPreparation terminee : {total_rows} lignes reparties.")
    print(f"Fichiers generes dans : {out_dir}")


if __name__ == "__main__":
    main()
