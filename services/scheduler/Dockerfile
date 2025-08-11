# Dockerfile for Scheduler Service
FROM trading-system-base:latest

# Set service-specific environment variables
ENV SERVICE_NAME=scheduler \
    SERVICE_PORT=8000

# Copy service-specific requirements
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy service source code
COPY src/ ./src/
COPY tests/ ./tests/

# Expose service port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the scheduler service
CMD ["python", "-m", "src.main"]
