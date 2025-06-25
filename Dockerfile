# Use Python 3.11 slim image for better compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_cloudrun_fixed.txt .
RUN pip install --no-cache-dir -r requirements_cloudrun_fixed.txt

# Copy application code
COPY main.py .
COPY start.py .
COPY bright-coyote-463315-q8-59797318b374.json .
COPY *.csv ./
COPY *.jsonl ./
COPY *.xlsx ./

# Create directory for vector database
RUN mkdir -p ./vector_db

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (Cloud Run will set PORT environment variable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with robust startup
CMD exec python start.py 