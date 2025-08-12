# Dockerfile for Data Collector Service
FROM trading-system-base:latest

# Set service-specific environment variables
ENV SERVICE_NAME=data_collector \
    SERVICE_PORT=8001

# Switch to root to install packages
USER root

# Copy service-specific requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service source code
COPY src/ ./src/

# Fix ownership and switch back to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose service port
EXPOSE 8001

# Health check - disabled for background service (override base image health check)
HEALTHCHECK NONE

# Run the data collector service
CMD ["python", "-m", "src.main"]
