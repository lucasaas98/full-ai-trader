# Dockerfile for Risk Manager Service
FROM trading-system-base:latest

# Set service-specific environment variables
ENV SERVICE_NAME=risk_manager \
    SERVICE_PORT=8003

# Copy service-specific requirements
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy service source code
COPY src/ ./src/
COPY tests/ ./tests/

# Expose service port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the risk manager service
CMD ["python", "-m", "src.main"]
