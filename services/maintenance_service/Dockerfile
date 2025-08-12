# Dockerfile for Maintenance Service
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_COMPILE_BYTECODE=1 \
    SERVICE_NAME=maintenance_service \
    SERVICE_PORT=8007

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# Add uv to PATH
ENV PATH="/root/.cargo/bin:$PATH"

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
RUN mkdir -p /app/data/maintenance /app/logs && \
    chown -R trader:trader /app/data /app/logs

# Switch to non-root user
USER trader

# Expose port
EXPOSE 8007

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8007/health || exit 1

# Start the service
CMD ["python", "-m", "src.main"]
