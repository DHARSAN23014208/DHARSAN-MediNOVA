FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_USE_LEGACY_KERAS=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PORT=7860

# Install essential system dependencies (libgomp1 for OpenMP/LightGBM/XGBoost, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure upload and sample directories exist with full write permissions
RUN mkdir -p static/uploads static/samples && \
    chmod -R 777 static/uploads static/samples

# Expose default port (7860 for Hugging Face Spaces, overridable by $PORT)
EXPOSE 7860

# Run production WSGI server with Gunicorn (1 worker + 4 threads to prevent OOM on shared cloud tiers)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120 app:app"]
