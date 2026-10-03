# syntax=docker/dockerfile:1
# Runtime image. UMAP and HDBSCAN are not installed here.
# Projections are precomputed by `make train` into models/ and copied in.
# Build only after `make train` (or the CI fixture train) so models/ is populated.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    PROJECT_ROOT=/app \
    ARTIFACT_DIR=/app/models \
    PORT=8000 \
    API_BASE_URL=http://127.0.0.1:8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml pyproject.toml ./
COPY src ./src
COPY app ./app
COPY models ./models

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8000') + '/health')"

CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
