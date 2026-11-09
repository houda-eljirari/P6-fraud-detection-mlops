import pandas as pd

from src.monitoring.drift import (
    build_reference,
    calculate_psi,
    compare_dataset,
)


def test_psi_identical_distributions():
    distribution = [0.2, 0.3, 0.5]

    psi = calculate_psi(distribution, distribution)

    assert abs(psi) < 1e-9


def test_psi_detects_distribution_change():
    reference = [0.2, 0.3, 0.5]
    current = [0.7, 0.2, 0.1]

    psi = calculate_psi(reference, current)

    assert psi > 0


def test_compare_dataset_detects_numeric_drift():
    reference_data = pd.DataFrame({
        "amount": [10, 12, 14, 16, 18, 20] * 10,
        "category": ["A", "B"] * 30,
    })

    current_data = pd.DataFrame({
        "amount": [100, 120, 140, 160, 180, 200] * 10,
        "category": ["A", "B"] * 30,
    })

    reference = build_reference(reference_data)
    results = compare_dataset(reference, current_data)

    assert results["amount"]["psi"] > 0.25
    assert results["amount"]["status"] == "high"


def test_compare_dataset_handles_new_categories():
    reference_data = pd.DataFrame({
        "category": ["A", "B", "A", "B"] * 10,
    })

    current_data = pd.DataFrame({
        "category": ["A", "C", "A", "C"] * 10,
    })

    reference = build_reference(reference_data)
    results = compare_dataset(reference, current_data)

    assert results["category"]["psi"] > 0