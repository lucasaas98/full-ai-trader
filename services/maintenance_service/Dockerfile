FROM trading-system-base:latest

# Set environment variables
ENV SERVICE_NAME=maintenance_service \
    SERVICE_PORT=9107

# Switch back to root to install service-specific requirements
USER root

# Copy service-specific requirements and install as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Switch back to non-root user
USER appuser

# Copy service code
COPY src/ ./src/

# Expose port
EXPOSE 9107
