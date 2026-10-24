# syntax=docker/dockerfile:1

# ---------- Etape 1 : installation des dependances ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- Etape 2 : image d'execution (plus legere) ----------
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DVC_NO_ANALYTICS=1

# Utilisateur non-root
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER appuser

# Le projet est monte dans /app par docker-compose
CMD ["dvc", "repro", "--force"]
