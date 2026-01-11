# Dockerfile
FROM python:3.11-slim

# ---- Runtime env (safe defaults) ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps from requirements.txt
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy the rest of the repo
COPY . /app

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# ---- App defaults ----
#  FastAPI app is in app/main.py as "app" instance.

ENV APP_MODULE="app.main:app" \
    HOST="0.0.0.0" \
    PORT="8000" \
    WORKERS="2"

EXPOSE 8000

# Healthcheck hits your existing endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Prod runner
CMD ["bash", "-lc", "gunicorn -k uvicorn.workers.UvicornWorker -w ${WORKERS} -b ${HOST}:${PORT} ${APP_MODULE}"]
