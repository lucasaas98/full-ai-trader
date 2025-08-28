#!/bin/bash

# Deploy script for Full AI Trader - Development Environment
# This script deploys the trading system using docker compose

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="full-ai-trader"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_message "$BLUE" "=== Checking prerequisites ==="

    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        print_message "$RED" "Error: Docker is not installed"
        exit 1
    fi

    # Check if Docker Compose v2 is installed
    if ! docker compose version &> /dev/null; then
        print_message "$RED" "Error: Docker Compose v2 is not installed"
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_message "$RED" "Error: Docker daemon is not running"
        exit 1
    fi

    # Check if .env file exists
    if [ ! -f "$ENV_FILE" ]; then
        print_message "$YELLOW" "Warning: .env file not found"
        if [ -f ".env.example" ]; then
            print_message "$YELLOW" "Creating .env from .env.example"
            cp .env.example .env
            print_message "$YELLOW" "Please update the .env file with your API keys and configuration"
            exit 1
        else
            print_message "$RED" "Error: No .env or .env.example file found"
            exit 1
        fi
    fi

    print_message "$GREEN" "✓ All prerequisites met"
}

# Function to validate environment variables
validate_env() {
    print_message "$BLUE" "=== Validating environment variables ==="

    # Check for critical environment variables
    local missing_vars=()

    # Source the .env file to check variables
    set -a
    source .env
    set +a

    # Check required variables
    [ -z "$DB_PASSWORD" ] && missing_vars+=("DB_PASSWORD")
    [ -z "$TWELVE_DATA_API_KEY" ] && missing_vars+=("TWELVE_DATA_API_KEY")

    if [ ${#missing_vars[@]} -gt 0 ]; then
        print_message "$RED" "Error: Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        print_message "$YELLOW" "Please update your .env file"
        exit 1
    fi

    print_message "$GREEN" "✓ Environment variables validated"
}

# Function to clean up old containers and volumes (optional)
cleanup() {
    print_message "$BLUE" "=== Cleaning up old resources ==="

    # Stop and remove existing containers
    docker compose down --remove-orphans 2>/dev/null || true

    print_message "$GREEN" "✓ Cleanup completed"
}

# Function to build images
build_images() {
    print_message "$BLUE" "=== Building Docker images ==="

    docker compose build --parallel

    print_message "$GREEN" "✓ Images built successfully"
}

# Function to start services
start_services() {
    print_message "$BLUE" "=== Starting services ==="

    # Start infrastructure services first
    print_message "$YELLOW" "Starting infrastructure services..."
    docker compose up -d postgres redis

    # Wait for infrastructure to be healthy
    print_message "$YELLOW" "Waiting for infrastructure services to be healthy..."
    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if docker compose ps postgres | grep -q "healthy" && \
           docker compose ps redis | grep -q "healthy"; then
            print_message "$GREEN" "✓ Infrastructure services are healthy"
            break
        fi

        attempt=$((attempt + 1))
        if [ $attempt -eq $max_attempts ]; then
            print_message "$RED" "Error: Infrastructure services failed to become healthy"
            docker compose logs postgres redis
            exit 1
        fi

        echo -n "."
        sleep 2
    done
    echo ""

    # Start monitoring services
    print_message "$YELLOW" "Starting monitoring services..."
    docker compose up -d prometheus grafana alertmanager node_exporter postgres_exporter redis_exporter cadvisor

    # Start ELK stack (optional for development)
    read -p "Do you want to start the ELK stack (Elasticsearch, Kibana, Logstash)? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_message "$YELLOW" "Starting ELK stack..."
        docker compose up -d elasticsearch kibana logstash
    fi

    # Start core trading services
    print_message "$YELLOW" "Starting core trading services..."
    docker compose up -d data_collector strategy_engine risk_manager trade_executor scheduler

    # Start auxiliary services
    print_message "$YELLOW" "Starting auxiliary services..."
    docker compose up -d export_service maintenance_service notification_service

    print_message "$GREEN" "✓ All services started"
}

# Function to check service health
check_health() {
    print_message "$BLUE" "=== Checking service health ==="

    # Wait a bit for services to initialize
    sleep 5

    # Get service status
    docker compose ps

    # Check if all services are running
    local failed_services=$(docker compose ps --status exited --status dead -q)

    if [ -n "$failed_services" ]; then
        print_message "$RED" "Warning: Some services are not running properly"
        print_message "$YELLOW" "Checking logs for failed services..."
        docker compose logs --tail=50 $failed_services
    else
        print_message "$GREEN" "✓ All services are running"
    fi
}

# Function to show access information
show_info() {
    print_message "$BLUE" "=== Access Information ==="
    echo ""
    print_message "$GREEN" "Services are available at:"
    echo "  • PostgreSQL:        localhost:5432"
    echo "  • Redis:             localhost:6379"
    echo "  • Data Collector:    http://localhost:9101"
    echo "  • Strategy Engine:   http://localhost:9102"
    echo "  • Risk Manager:      http://localhost:9103"
    echo "  • Trade Executor:    http://localhost:9104"
    echo "  • Scheduler:         http://localhost:9105"
    echo "  • Export Service:    http://localhost:9106"
    echo "  • Maintenance:       http://localhost:9107"
    echo "  • Notification:      http://localhost:8008"
    echo ""
    print_message "$GREEN" "Monitoring:"
    echo "  • Prometheus:        http://localhost:9090"
    echo "  • Grafana:           http://localhost:3000 (admin/admin)"
    echo "  • Alertmanager:      http://localhost:9093"
    echo ""

    # Check if ELK is running
    if docker compose ps kibana 2>/dev/null | grep -q "running"; then
        print_message "$GREEN" "ELK Stack:"
        echo "  • Elasticsearch:     http://localhost:9200"
        echo "  • Kibana:            http://localhost:5601"
        echo "  • Logstash:          http://localhost:5000"
        echo ""
    fi

    print_message "$YELLOW" "Useful commands:"
    echo "  • View logs:         docker compose logs -f [service_name]"
    echo "  • Stop all:          docker compose down"
    echo "  • Restart service:   docker compose restart [service_name]"
    echo "  • View stats:        docker stats"
    echo ""
}

# Function to tail logs (optional)
tail_logs() {
    read -p "Do you want to tail the logs? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_message "$BLUE" "Tailing logs (Ctrl+C to stop)..."
        docker compose logs -f --tail=100
    fi
}

# Main deployment flow
main() {
    print_message "$BLUE" "========================================="
    print_message "$BLUE" "   Full AI Trader - Development Deploy"
    print_message "$BLUE" "========================================="
    echo ""

    # Change to script directory
    cd "$(dirname "$0")"

    # Run deployment steps
    check_prerequisites
    validate_env

    # Ask if user wants to clean up first
    read -p "Do you want to clean up existing containers first? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup
    fi

    # Build and start services
    build_images
    start_services

    # Check health and show info
    check_health
    show_info

    print_message "$GREEN" "=== Deployment completed successfully! ==="

    # Optional: tail logs
    tail_logs
}

# Handle script arguments
case "${1:-}" in
    up)
        start_services
        check_health
        show_info
        ;;
    down)
        print_message "$BLUE" "Stopping all services..."
        docker compose down
        print_message "$GREEN" "✓ All services stopped"
        ;;
    restart)
        print_message "$BLUE" "Restarting services..."
        docker compose restart ${2:-}
        print_message "$GREEN" "✓ Services restarted"
        ;;
    logs)
        docker compose logs -f --tail=100 ${2:-}
        ;;
    status)
        docker compose ps
        ;;
    build)
        build_images
        ;;
    clean)
        cleanup
        print_message "$YELLOW" "Removing volumes..."
        docker compose down -v
        print_message "$GREEN" "✓ Full cleanup completed"
        ;;
    help|--help|-h)
        echo "Usage: $0 [command] [options]"
        echo ""
        echo "Commands:"
        echo "  up       - Start all services"
        echo "  down     - Stop all services"
        echo "  restart  - Restart services (optionally specify service name)"
        echo "  logs     - Tail logs (optionally specify service name)"
        echo "  status   - Show service status"
        echo "  build    - Build Docker images"
        echo "  clean    - Full cleanup including volumes"
        echo "  help     - Show this help message"
        echo ""
        echo "Without arguments, runs the full deployment process"
        ;;
    *)
        main
        ;;
esac
