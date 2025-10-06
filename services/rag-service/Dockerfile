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

# Create required directories first (as root)
RUN mkdir -p \
    /app/data \
    /app/models \
    /app/system/logs \
    /app/system/temp \
    /app/system/backups

# Copy only necessary application code with proper ownership
COPY --chown=raguser:raguser services/rag-service/src /app/src
COPY --chown=raguser:raguser services/rag-service/config /app/config
COPY --chown=raguser:raguser services/rag-service/*.py /app/
COPY --chown=raguser:raguser rag.py /app/

# Set ownership only for empty directories (much faster)
RUN chown -R raguser:raguser /app/data /app/models /app/system

# Switch to non-root user
USER raguser

# Health check - just verify Python imports work
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); from utils.config_loader import get_config_section; get_config_section('config', 'paths')" || exit 1

# Default command - keep container running for CLI operations
# Can be overridden in docker-compose
CMD ["tail", "-f", "/dev/null"]

# Expose ports for web services
EXPOSE 8080