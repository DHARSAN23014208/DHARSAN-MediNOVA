FROM python:3.10-slim

# Set environment variables to optimize RAM, suppress TF logs, and configure single-thread execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OMP_NUM_THREADS=1 \
    MALLOC_TRIM_THRESHOLD_=65536 \
    PORT=10000

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure upload and sample directories exist with write permissions
RUN mkdir -p static/uploads static/samples private_uploads && \
    chmod -R 777 static/uploads static/samples private_uploads

EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} app:app --workers 1 --threads 2 --timeout 120"]
