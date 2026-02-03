# Multi-Object Tracking Web Application
# Build: docker build -t mot-web -f docker/Dockerfile .
# Run:   docker run -p 5000:5000 mot-web

FROM python:3.11-slim

LABEL maintainer="MOT Team"
LABEL description="Multi-Object Tracking Web Application"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy dependency files first (for better layer caching)
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# Copy application source
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install the application in editable mode
RUN pip install -e .

# Create required directories
RUN mkdir -p /app/var/uploads /app/var/results && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set runtime environment variables
ENV HOST=0.0.0.0 \
    PORT=5000 \
    ENV=prod \
    PROJECT_ROOT=/app \
    UPLOAD_DIR=/app/var/uploads \
    RESULTS_DIR=/app/var/results

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Run the application
CMD ["mot-web"]
