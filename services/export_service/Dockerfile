FROM trading-system-base:latest

# Set environment variables
ENV SERVICE_NAME=export_service \
    SERVICE_PORT=9106

# Copy service code
COPY src/ ./src/

# Expose port
EXPOSE 9106

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9106/health || exit 1
