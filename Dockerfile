# ── Base image ─────────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 --retries=5 -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY app/        ./app/
COPY src/        ./src/
RUN mkdir -p ./models
#COPY models/     ./models/
RUN mkdir -p ./models && \
    curl -L -o ./models/xgboost_model.pkl \
    https://github.com/hrishi319/CC_Fraud_Detection_System/releases/download/v1.0/xgboost_model.pkl && \
    curl -L -o ./models/scaler.pkl \
    https://github.com/hrishi319/CC_Fraud_Detection_System/releases/download/v1.0/scaler.pkl
# ── Environment ────────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Expose port ────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start server ───────────────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]