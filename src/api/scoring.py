import numpy as np
import pandas as pd

MISSING_CODE_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

CATEGORICAL_FEATURES = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

# Premiere ligne reelle de Base.csv (sans la cible fraud_bool ni la colonne month)
EXAMPLE_TRANSACTION = {
    "income": 0.3,
    "name_email_similarity": 0.986506310633034,
    "prev_address_months_count": -1,
    "current_address_months_count": 25,
    "customer_age": 40,
    "days_since_request": 0.0067353870811739,
    "intended_balcon_amount": 102.45371092469456,
    "payment_type": "AA",
    "zip_count_4w": 1059,
    "velocity_6h": 13096.035018400871,
    "velocity_24h": 7850.955007125409,
    "velocity_4w": 6742.080561007602,
    "bank_branch_count_8w": 5,
    "date_of_birth_distinct_emails_4w": 5,
    "employment_status": "CB",
    "credit_risk_score": 163,
    "email_is_free": 1,
    "housing_status": "BC",
    "phone_home_valid": 0,
    "phone_mobile_valid": 1,
    "bank_months_count": 9,
    "has_other_cards": 0,
    "proposed_credit_limit": 1500.0,
    "foreign_request": 0,
    "source": "INTERNET",
    "session_length_in_minutes": 16.224843433978073,
    "device_os": "linux",
    "keep_alive_session": 1,
    "device_distinct_emails_8w": 1,
    "device_fraud_count": 0,
}


def build_frame(payload: dict) -> pd.DataFrame:
    """Transforme une transaction en DataFrame d'une ligne, comme dans prepare.py."""
    df = pd.DataFrame([payload])
    for col in MISSING_CODE_COLS:
        # None (absent) et -1 (code BAF) deviennent tous deux une valeur manquante
        df[col] = pd.to_numeric(df[col]).replace(-1, np.nan)
    return df


def score_transaction(model, payload: dict) -> float:
    """Probabilite de fraude (entre 0 et 1) pour une transaction."""
    return float(model.predict_proba(build_frame(payload))[0, 1])