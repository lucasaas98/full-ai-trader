FROM trading-system-base:latest

# Switch back to root to install service-specific requirements
USER root

# Copy service-specific requirements and install as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Switch back to non-root user
USER appuser

# Copy service code
COPY src/ ./src/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import asyncio; import redis.asyncio as redis; asyncio.run(redis.from_url('redis://localhost:6379').ping())" || exit 1
