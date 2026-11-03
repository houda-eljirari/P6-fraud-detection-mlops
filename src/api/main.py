"""API de scoring de fraude (FastAPI)."""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.routing import Match

from src.api.metrics import LATENCY, MODEL_LOADED, PREDICTIONS, REQUESTS, SCORES
from src.api.schemas import PredictionOut, TransactionIn
from src.api.scoring import score_transaction

DEFAULT_MODEL_PATH = "models/model.joblib"
# Appels techniques : non comptes, pour ne pas fausser le trafic observe
UNTRACKED_PATHS = {"/metrics", "/health"}


def route_template(request: Request) -> str:
    """Chemin de la route (ex. /predict), pour garder peu de valeurs d'etiquette."""
    for route in request.app.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return route.path
    return "unmatched"


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
        MODEL_LOADED.set(1 if app.state.model is not None else 0)
        yield

    app = FastAPI(
        title="P6 - API de detection de fraude",
        description="Score une demande d'ouverture de compte (jeu BAF).",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.model = None

    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        template = route_template(request)
        if template not in UNTRACKED_PATHS:
            REQUESTS.labels(request.method, template, str(response.status_code)).inc()
            LATENCY.labels(template).observe(time.perf_counter() - start)
        return response

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

    @app.get("/metrics")
    def metrics() -> Response:
        """Metriques au format Prometheus."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictionOut)
    def predict(transaction: TransactionIn) -> PredictionOut:
        """Probabilite de fraude et decision pour une transaction."""
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Modele non charge")
        probability = score_transaction(app.state.model, transaction.model_dump())
        is_fraud = probability >= decision_threshold
        SCORES.observe(probability)
        PREDICTIONS.labels("fraud" if is_fraud else "legit").inc()
        return PredictionOut(
            fraud_probability=probability,
            is_fraud=is_fraud,
            threshold=decision_threshold,
        )

    return app


app = create_app()