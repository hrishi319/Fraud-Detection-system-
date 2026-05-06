# ── Base image ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements first — Docker layer caching means this layer
# only rebuilds when requirements.txt changes, not on every code change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY app/        ./app/
COPY src/        ./src/
COPY models/     ./models/

# ── Environment ────────────────────────────────────────────────────────────────
# Never hardcode secrets in Dockerfile — pass via docker run -e or .env
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Health check ───────────────────────────────────────────────────────────────
# Docker checks this every 30s — restarts container if unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Expose port ────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start server ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
