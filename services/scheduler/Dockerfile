# Dockerfile for Scheduler Service
FROM trading-system-base:latest

# Set service-specific environment variables
ENV SERVICE_NAME=scheduler \
    SERVICE_PORT=9105

# Switch to root to install packages
USER root

# Copy service-specific requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Switch back to non-root user
USER appuser

# Copy service source code
COPY src/ ./src/

# Expose service port
EXPOSE ${SCHEDULER_PORT:-9105}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${SCHEDULER_PORT:-9105}/health || exit 1
