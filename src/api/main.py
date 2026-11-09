"""API de scoring de fraude (FastAPI)."""
import asyncio
import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.routing import Match

from src.api.metrics import (
    DATA_DRIFT_LAST_CHECK_TIMESTAMP,
    DATA_DRIFT_PSI,
    DATA_DRIFT_SAMPLES,
    DATA_DRIFT_STATUS,
    LATENCY,
    MODEL_LOADED,
    PREDICTIONS,
    REQUESTS,
    SCORES,
)
from src.api.schemas import PredictionOut, TransactionIn
from src.api.scoring import score_transaction
from src.monitoring.drift import compare_dataset

DEFAULT_MODEL_PATH = "models/model.joblib"
DEFAULT_DRIFT_REFERENCE_PATH = "metrics/drift_reference.json"
DRIFT_BATCH_SIZE = int(os.getenv("DRIFT_BATCH_SIZE", "100"))

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
        # Configuration du Data Drift
    drift_reference_path = Path(
        os.getenv("DRIFT_REFERENCE_PATH", DEFAULT_DRIFT_REFERENCE_PATH)
    )

    if DRIFT_BATCH_SIZE < 1:
        raise ValueError("DRIFT_BATCH_SIZE doit etre superieur a zero")

    # Collecteur local des transactions
    drift_buffer = deque()
    drift_lock = Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Chargement du modele
        if path.exists():
            app.state.model = joblib.load(path)

        MODEL_LOADED.set(1 if app.state.model is not None else 0)

        # Chargement de la reference Data Drift
        app.state.drift_reference = None
        if drift_reference_path.exists():

            def load_drift_reference():
                with open(drift_reference_path, "r", encoding="utf-8") as f:
                    return json.load(f)

            app.state.drift_reference = await asyncio.to_thread(
                load_drift_reference
            )

            print("Reference Data Drift chargee.")

        else:
            print(
                f"Attention : reference Data Drift absente : "
                f"{drift_reference_path}"
            )

        yield

    app = FastAPI(
        title="P6 - API de detection de fraude",
        description="Score une demande d'ouverture de compte (jeu BAF).",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.model = None

    def update_drift(transaction_data: dict) -> None:
        """Calcule le PSI lorsqu'un lot de transactions est complet."""
        if app.state.drift_reference is None:
            return

        with drift_lock:
            drift_buffer.append(transaction_data)

            if len(drift_buffer) < DRIFT_BATCH_SIZE:
                return

            batch = list(drift_buffer)
            drift_buffer.clear()

        try:
            current_df = pd.DataFrame(batch)

            results = compare_dataset(
                app.state.drift_reference,
                current_df,
            )

            status_values = {
                "low": 0,
                "warning": 1,
                "high": 2,
            }

            for feature, result in results.items():
                DATA_DRIFT_PSI.labels(feature).set(result["psi"])
                DATA_DRIFT_STATUS.labels(feature).set(
                    status_values[result["status"]]
                )

            DATA_DRIFT_SAMPLES.set(len(batch))
            DATA_DRIFT_LAST_CHECK_TIMESTAMP.set(time.time())

            high = sum(r["status"] == "high" for r in results.values())
            warning = sum(r["status"] == "warning" for r in results.values())

            print(
                f"Data Drift calcule sur {len(batch)} transactions : "
                f"{high} variables high, {warning} warning."
            )

        except Exception as exc:  # noqa: BLE001
            # Une erreur de supervision ne doit pas bloquer le scoring.
            print(f"Erreur lors du calcul Data Drift : {exc}")
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

        update_drift(transaction.model_dump())

        return PredictionOut(
            fraud_probability=probability,
            is_fraud=is_fraud,
            threshold=decision_threshold,
        )

    return app


app = create_app()
