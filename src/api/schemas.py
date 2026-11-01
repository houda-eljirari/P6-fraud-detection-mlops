"""Schemas de validation de l'API (Pydantic)."""
from pydantic import BaseModel, ConfigDict, Field

from src.api.scoring import EXAMPLE_TRANSACTION


class TransactionIn(BaseModel):
    """Une demande d'ouverture de compte a scorer (colonnes de BAF, hors cible et mois)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": EXAMPLE_TRANSACTION},
    )

    income: float = Field(ge=0, le=1)
    name_email_similarity: float = Field(ge=0, le=1)
    # -1 ou null = valeur manquante
    prev_address_months_count: float | None = None
    current_address_months_count: float | None = None
    customer_age: float
    days_since_request: float = Field(ge=0)
    intended_balcon_amount: float
    payment_type: str
    zip_count_4w: float
    velocity_6h: float
    velocity_24h: float
    velocity_4w: float
    bank_branch_count_8w: float
    date_of_birth_distinct_emails_4w: float
    employment_status: str
    credit_risk_score: float
    email_is_free: int = Field(ge=0, le=1)
    housing_status: str
    phone_home_valid: int = Field(ge=0, le=1)
    phone_mobile_valid: int = Field(ge=0, le=1)
    bank_months_count: float | None = None
    has_other_cards: int = Field(ge=0, le=1)
    proposed_credit_limit: float
    foreign_request: int = Field(ge=0, le=1)
    source: str
    session_length_in_minutes: float | None = None
    device_os: str
    keep_alive_session: int = Field(ge=0, le=1)
    device_distinct_emails_8w: float | None = None
    device_fraud_count: int = Field(ge=0)


class PredictionOut(BaseModel):
    """Reponse de /predict."""

    fraud_probability: float
    is_fraud: bool
    threshold: float