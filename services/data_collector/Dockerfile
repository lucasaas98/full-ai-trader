# Dockerfile for Data Collector Service
FROM trading-system-base:latest

# Set service-specific environment variables
ENV SERVICE_NAME=data_collector \
    SERVICE_PORT=8001

# Copy service-specific requirements
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy service source code
COPY src/ ./src/
COPY tests/ ./tests/

# Expose service port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the data collector service
CMD ["python", "-m", "src.main"]
