"""API de scoring de fraude (FastAPI)."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from src.api.schemas import PredictionOut, TransactionIn
from src.api.scoring import score_transaction

DEFAULT_MODEL_PATH = "models/model.joblib"


def create_app(model_path: str | None = None, threshold: float | None = None) -> FastAPI:
    """Cree l'application. Le chemin du modele et le seuil sont configurables."""
    path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    decision_threshold = (
        threshold if threshold is not None else float(os.getenv("THRESHOLD", "0.5"))
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Le modele est charge une seule fois, au demarrage
        if path.exists():
            app.state.model = joblib.load(path)
        yield

    app = FastAPI(
        title="P6 - API de detection de fraude",
        description="Score une demande d'ouverture de compte (jeu BAF).",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.model = None

    @app.get("/health")
    def health() -> dict:
        """Liveness : le processus repond."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        """Readiness : le modele est charge et l'API peut scorer."""
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Modele non charge")
        return {"status": "ready"}

    @app.post("/predict", response_model=PredictionOut)
    def predict(transaction: TransactionIn) -> PredictionOut:
        """Probabilite de fraude et decision pour une transaction."""
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Modele non charge")
        probability = score_transaction(app.state.model, transaction.model_dump())
        return PredictionOut(
            fraud_probability=probability,
            is_fraud=probability >= decision_threshold,
            threshold=decision_threshold,
        )

    return app


app = create_app()