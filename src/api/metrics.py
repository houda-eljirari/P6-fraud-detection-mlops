"""Metriques Prometheus de l'API.

Definies au niveau du module (une seule fois) : l'application peut etre creee
plusieurs fois (tests) sans declarer deux fois la meme metrique.
"""
from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "http_requests_total",
    "Nombre de requetes HTTP recues",
    ["method", "path", "status"],
)

LATENCY = Histogram(
    "http_request_duration_seconds",
    "Duree de traitement des requetes HTTP",
    ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# Metriques metier : ce que fait le modele, pas seulement le serveur
PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Nombre de predictions, par decision",
    ["decision"],
)

SCORES = Histogram(
    "fraud_score",
    "Distribution des probabilites de fraude predites",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "1 si le modele est charge, 0 sinon",
)
# Metriques de Data Drift (PSI)
DATA_DRIFT_PSI = Gauge(
    "data_drift_psi",
    "Population Stability Index par variable",
    ["feature"],
)

DATA_DRIFT_STATUS = Gauge(
    "data_drift_status",
    "Niveau de drift : 0=low, 1=warning, 2=high",
    ["feature"],
)

DATA_DRIFT_SAMPLES = Gauge(
    "data_drift_samples",
    "Nombre de transactions utilisees lors du dernier calcul PSI",
)

DATA_DRIFT_LAST_CHECK_TIMESTAMP = Gauge(
    "data_drift_last_check_timestamp",
    "Horodatage Unix du dernier calcul de Data Drift",
)