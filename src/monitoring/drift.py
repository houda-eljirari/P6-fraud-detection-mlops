
"""Construction de la reference et comparaison du Data Drift avec le PSI."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_PATH = Path("metrics/drift_reference.json")
BINS = 10
EPSILON = 1e-6


def build_reference(df: pd.DataFrame) -> dict:
    """Construit les distributions de reference pour chaque variable."""

    reference = {
        "reference_rows": len(df),
        "features": {},
    }

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            values = series.dropna().to_numpy(dtype=float)

            if len(values) == 0:
                continue

            internal_edges = np.unique(
                np.quantile(
                    values,
                    np.linspace(0, 1, BINS + 1)[1:-1],
                )
            )

            proportions = get_numeric_proportions(
                series, internal_edges
            )

            reference["features"][column] = {
                "type": "numeric",
                "edges": internal_edges.tolist(),
                "proportions": proportions,
            }

        else:
            values = series.fillna("__MISSING__").astype(str)
            categories = sorted(values.unique().tolist())

            counts = values.value_counts(normalize=True)
            proportions = {
                category: float(counts.get(category, 0))
                for category in categories
            }

            reference["features"][column] = {
                "type": "categorical",
                "categories": categories,
                "proportions": proportions,
            }

    return reference


def get_numeric_proportions(series, internal_edges):
    """Calcule les proportions des intervalles numeriques et des NaN."""

    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna().to_numpy(dtype=float)

    edges = np.concatenate(([-np.inf], internal_edges, [np.inf]))

    counts, _ = np.histogram(valid, bins=edges)

    total = len(series)
    proportions = (counts / total).tolist()
    proportions.append(float(values.isna().sum() / total))

    return proportions


def calculate_psi(reference_pct, current_pct) -> float:
    """Calcule le PSI entre deux distributions de memes dimensions."""

    reference_pct = np.asarray(reference_pct, dtype=float)
    current_pct = np.asarray(current_pct, dtype=float)

    reference_pct = reference_pct + EPSILON
    current_pct = current_pct + EPSILON

    return float(
        np.sum(
            (current_pct - reference_pct)
            * np.log(current_pct / reference_pct)
        )
    )


def compare_dataset(reference: dict, current: pd.DataFrame) -> dict:
    """Compare un dataset courant avec les distributions de reference."""

    results = {}

    for column, info in reference["features"].items():
        if column not in current.columns:
            raise ValueError(f"Colonne absente : {column}")

        series = current[column]

        if info["type"] == "numeric":
            current_pct = get_numeric_proportions(
                series, np.array(info["edges"])
            )

            reference_pct = list(info["proportions"])

        else:
            values = series.fillna("__MISSING__").astype(str)
            categories = info["categories"]

            reference_pct = [
                info["proportions"].get(category, 0)
                for category in categories
            ]

            known_categories = set(categories)
            current_values = values.where(
                values.isin(known_categories), "__OTHER__"
            )

            current_counts = current_values.value_counts(normalize=True)

            current_pct = [
                float(current_counts.get(category, 0))
                for category in categories
            ]

            current_pct.append(
                float(current_counts.get("__OTHER__", 0))
            )
            reference_pct.append(0.0)

        psi = calculate_psi(reference_pct, current_pct)

        results[column] = {
            "psi": round(psi, 6),
            "status": (
                "high" if psi >= 0.25
                else "warning" if psi >= 0.10
                else "low"
            ),
        }

    return results


def main():
    # Construction de la reference a partir du train
    train = pd.read_csv("data/processed/train.csv")
    train = train.drop(columns=["fraud_bool", "month"])

    reference = build_reference(train)

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REFERENCE_PATH, "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2)

    print(f"Reference enregistree : {REFERENCE_PATH}")
    print(f"Lignes de reference : {reference['reference_rows']}")
    print(f"Variables suivies : {len(reference['features'])}")

    # Premiere comparaison exploratoire avec validation
    validation = pd.read_csv("data/processed/validation.csv")
    validation = validation.drop(columns=["fraud_bool", "month"])

    results = compare_dataset(reference, validation)

    print("\nComparaison TRAIN / VALIDATION :")

    for column, result in results.items():
        print(
            f"{column:40} "
            f"PSI={result['psi']:.6f} "
            f"Statut={result['status']}"
        )

    high = sum(r["status"] == "high" for r in results.values())
    warning = sum(r["status"] == "warning" for r in results.values())

    print(f"\nVariables avec drift important : {high}")
    print(f"Variables a surveiller : {warning}")


if __name__ == "__main__":
    main()

