FROM trading-system-base:latest

# Copy service code
COPY src/ ./src/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import asyncio; import redis.asyncio as redis; asyncio.run(redis.from_url('redis://localhost:6379').ping())" || exit 1
