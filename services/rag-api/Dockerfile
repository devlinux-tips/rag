# RAG API Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy RAG service source code
COPY services/rag-service/src ./src
COPY services/rag-service/config ./config

# Copy the main RAG API application
COPY services/rag-api/main.py .

# Create data and models directories
RUN mkdir -p data models logs temp

# Set Python path
ENV PYTHONPATH=/app/src:/app

# Expose port
EXPOSE 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8082/health || exit 1

# Run the application
CMD ["python", "main.py"]