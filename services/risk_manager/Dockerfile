# Dockerfile for Risk Manager Service
FROM trading-system-base:latest

# Switch back to root to install service-specific requirements
USER root

# Copy service-specific requirements and install as root
COPY requirements.txt /tmp/risk_manager_requirements.txt
RUN pip install --no-cache-dir -r /tmp/risk_manager_requirements.txt && \
    rm /tmp/risk_manager_requirements.txt

# Set service-specific environment variables
ENV SERVICE_NAME=risk_manager \
    SERVICE_PORT=8003

# Copy service source code
COPY src/ ./src/

# Ensure proper ownership
RUN chown -R appuser:appuser /app

# Switch back to non-root user
USER appuser

# Expose service port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the risk manager service
CMD ["python", "-m", "src.main"]
