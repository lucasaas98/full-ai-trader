FROM trading-system-base:latest

# Set environment variables
ENV SERVICE_NAME=maintenance_service \
    SERVICE_PORT=9107

# Copy service code
COPY src/ ./src/

# Expose port
EXPOSE 9107

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9107/health || exit 1
