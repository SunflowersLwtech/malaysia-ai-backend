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
COPY requirements_complete.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_complete.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p rag_database

# Expose port (Cloud Run will set PORT environment variable)
EXPOSE 8000

# Use Gunicorn to run the application
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 api_server:app 