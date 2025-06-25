# Use Python 3.11 slim image for better compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables first
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false

# Copy and install requirements with better caching
COPY requirements_cloudrun.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements_cloudrun.txt

# Copy application code and data
COPY main.py .
COPY startup.py .
COPY bright-coyote-463315-q8-59797318b374.json .
COPY *.csv ./
COPY *.jsonl ./
COPY *.xlsx ./

# Create necessary directories
RUN mkdir -p /app/chroma_data
RUN mkdir -p /app/vector_db

# Health check with longer timeout (dynamic port)
HEALTHCHECK --interval=30s --timeout=30s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Expose the port (Cloud Run will set actual PORT via env var)
EXPOSE 8080

# Run with our robust startup script
CMD exec python startup.py 