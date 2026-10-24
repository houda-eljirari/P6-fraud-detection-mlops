"""Preparation des donnees BAF : valeurs manquantes et split temporel par mois."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_params() -> dict:
    with open("params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["prepare"]


def main() -> None:
    params = load_params()
    df = pd.read_csv(params["raw_path"])

    # Dans BAF, -1 code une valeur manquante pour certaines colonnes
    for col in params["missing_value_cols"]:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)

    month_col = params["month_col"]
    train = df[df[month_col].isin(params["train_months"])]
    test = df[df[month_col].isin(params["test_months"])]

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    target = params["target"]
    print(f"train : {len(train)} lignes, taux de fraude {train[target].mean():.4f}")
    print(f"test  : {len(test)} lignes, taux de fraude {test[target].mean():.4f}")


if __name__ == "__main__":
    main()
