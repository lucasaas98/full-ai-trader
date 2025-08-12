# Dockerfile for Trade Executor Service
FROM trading-system-base:latest

# Set service-specific environment variables
ENV SERVICE_NAME=trade_executor \
    SERVICE_PORT=8004

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
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Run the trade executor service
CMD ["python", "-m", "src.main"]
