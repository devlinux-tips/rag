# Multi-stage Docker build for RAG Service
# Optimized for production deployment with security and size considerations

# Stage 1: Base image with Python and system dependencies
FROM python:3.12-slim as base

# Set environment variables for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd --gid 1000 raguser && \
    useradd --uid 1000 --gid raguser --shell /bin/bash --create-home raguser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    build-essential \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies installation
FROM base as dependencies

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=raguser:raguser . .

# Create required directories with proper permissions
RUN mkdir -p \
    /app/data \
    /app/models \
    /app/system/logs \
    /app/system/temp \
    /app/system/backups \
    && chown -R raguser:raguser /app

# Switch to non-root user
USER raguser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); from utils.config_loader import get_unified_config; get_unified_config()" || exit 1

# Default command - can be overridden in docker-compose
CMD ["python", "rag.py", "--help"]

# Expose ports for web services
EXPOSE 8080