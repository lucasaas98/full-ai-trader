# Dockerfile for Trade Executor Service
FROM trading-system-base:latest

# Set service-specific environment variables
ENV SERVICE_NAME=trade_executor \
    SERVICE_PORT=9104

# Switch to root to install packages
USER root

# Copy service-specific requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER appuser

# Copy service source code
COPY src/ ./src/

# Expose service port
EXPOSE 9104

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9104/health || exit 1
