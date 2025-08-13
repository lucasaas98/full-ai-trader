FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY services/notification_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules
COPY shared /app/shared
COPY monitoring /app/monitoring

# Copy service code
COPY services/notification_service/src/ /app/src/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Create non-root user
RUN useradd -m -u 1000 notifier && chown -R notifier:notifier /app
USER notifier

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import asyncio; import redis.asyncio as redis; asyncio.run(redis.from_url('redis://localhost:6379').ping())" || exit 1

# Run the service
CMD ["python", "src/main.py"]
