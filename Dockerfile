# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p vector_database

# Expose port (Cloud Run will set PORT environment variable)
EXPOSE 8000

# Use Uvicorn to run FastAPI application
CMD uvicorn api_server:app --host 0.0.0.0 --port $PORT 