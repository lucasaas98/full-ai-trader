# Dockerfile for Data Collector Service
FROM trading-system-base:latest

# Set service-specific environment variables
ENV SERVICE_NAME=data_collector \
    SERVICE_PORT=9101

# Copy service source code
COPY src/ ./src/

# Expose service port
EXPOSE 9101

# Health check - disabled for background service (override base image health check)
HEALTHCHECK NONE
