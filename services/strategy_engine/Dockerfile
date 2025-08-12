# Dockerfile for Strategy Engine Service
FROM trading-system-base:latest

# Set service-specific environment variables
ENV SERVICE_NAME=strategy_engine \
    SERVICE_PORT=9102

# Switch to root temporarily to install packages
USER root

# Copy service-specific requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service source code
COPY src/ ./src/

# Fix ownership
RUN chown -R appuser:appuser /app

# Switch back to appuser
USER appuser

# Expose service port
EXPOSE $SERVICE_PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$SERVICE_PORT/health || exit 1

# Run the strategy engine service
CMD ["python", "-m", "src.main"]
