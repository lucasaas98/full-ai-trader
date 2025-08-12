# Dockerfile for Export Service
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SERVICE_NAME=export_service \
    SERVICE_PORT=8006

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r trader && useradd -r -g trader trader

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules
COPY --chown=trader:trader ../../shared ./shared

# Copy service code
COPY --chown=trader:trader src/ ./src/

# Create necessary directories
RUN mkdir -p /app/data/exports /app/logs && \
    chown -R trader:trader /app/data /app/logs

# Switch to non-root user
USER trader

# Expose port
EXPOSE 8006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8006/health || exit 1

# Start the service
CMD ["python", "-m", "src.main"]
