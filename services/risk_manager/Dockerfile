# Dockerfile for Risk Manager Service
FROM trading-system-base:latest

# Switch back to root to install service-specific requirements
USER root

# Copy service-specific requirements and install as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Switch back to non-root user
USER appuser

# Set service-specific environment variables
ENV SERVICE_NAME=risk_manager \
    SERVICE_PORT=9103

# Copy service source code
COPY src/ ./src/

# Expose service port
EXPOSE 9103

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9103/health || exit 1
