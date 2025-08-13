# Dockerfile for Export Service
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SERVICE_NAME=export_service \
    SERVICE_PORT=9106

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
COPY services/export_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules
COPY shared/ ./shared/

# Copy logging module
COPY logging/ ./logging/

# Copy monitoring module
COPY monitoring/ ./monitoring/

# Copy service code
COPY services/export_service/src/ ./src/

# Create necessary directories
RUN mkdir -p /app/data/exports /app/logs

RUN chown -R trader:trader /app

# Switch to non-root user
USER trader

# Expose port
EXPOSE 9106

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9106/health || exit 1

# Start the service
CMD ["python", "/app/src/run.py"]
