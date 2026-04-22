# syntax=docker/dockerfile:1.7
# --- builder ----------------------------------------------------------------
FROM python:3.12-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# --- runtime ----------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Non-root user (security baseline)
RUN groupadd -r qf && useradd -r -g qf -u 10001 -d /app -s /usr/sbin/nologin qf

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/install/bin:$PATH" \
    QUANTFORGE_LOG_LEVEL=INFO

WORKDIR /app
# Copy python packages from builder
COPY --from=builder /install /install
# Copy app source (pyproject + package + web UI if present)
COPY pyproject.toml README.md ./
COPY quantforge/ ./quantforge/
COPY web/ ./web/

# Create cache dir the app writes to
RUN mkdir -p /app/data/cache && chown -R qf:qf /app

USER qf
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=3).read()" \
    || exit 1

# Use gunicorn + uvicorn workers for prod concurrency; default to 2 workers
CMD ["python", "-m", "uvicorn", "quantforge.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info", \
     "--proxy-headers", "--forwarded-allow-ips=*"]
