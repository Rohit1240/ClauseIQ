# ClauseIQ — single production image: React build served by FastAPI on port 8000.
# Build:  docker build -t clauseiq .
# Run:    docker run -p 8000:8000 -v clauseiq-data:/app/data/app clauseiq

# ---- 1. frontend build -------------------------------------------------------
FROM node:20-alpine AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. runtime ---------------------------------------------------------------
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only PyTorch wheel keeps the image small. For NVIDIA GPUs see README (CUDA section).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --index-url ${TORCH_INDEX} "torch>=2.2" \
 && pip install -r requirements.txt

COPY clauseiq/ clauseiq/
COPY backend/ backend/
COPY scripts/ scripts/
COPY artifacts/ artifacts/
COPY --from=frontend /fe/dist frontend/dist

RUN useradd --create-home --uid 10001 clauseiq && mkdir -p data/app && chown -R clauseiq /app/data
USER clauseiq

ENV CLAUSEIQ_ENV=production CLAUSEIQ_DB_PATH=/app/data/app/clauseiq.sqlite3 CLAUSEIQ_ALLOWED_ORIGINS=""
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
