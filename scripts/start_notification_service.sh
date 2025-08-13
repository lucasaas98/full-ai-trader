#!/bin/bash

# Startup script for the Notification Service
# This script handles the initialization and startup of the notification service

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="notification_service"
CONTAINER_NAME="trading_notification_service"
LOG_FILE="logs/notification_service/startup.log"

# Function to print colored output
print_color() {
    color=$1
    message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if container is running
is_container_running() {
    docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"
}

# Function to check Redis connection
check_redis() {
    print_color "$BLUE" "Checking Redis connection..."

    if docker exec trading_redis redis-cli ping > /dev/null 2>&1; then
        print_color "$GREEN" "✓ Redis is running and accessible"
        return 0
    else
        print_color "$RED" "✗ Redis is not accessible"
        return 1
    fi
}

# Function to check Gotify configuration
check_gotify_config() {
    print_color "$BLUE" "Checking Gotify configuration..."

    if [ -z "$GOTIFY_URL" ] || [ -z "$GOTIFY_TOKEN" ]; then
        print_color "$YELLOW" "⚠ Gotify credentials not set in environment"
        print_color "$YELLOW" "  The service will run but notifications won't be sent to Gotify"
        echo ""
        echo "  To enable Gotify notifications, add to your .env file:"
        echo "    GOTIFY_URL=http://your-gotify-server:80"
        echo "    GOTIFY_TOKEN=your-app-token"
        echo ""
        read -p "Continue without Gotify notifications? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_color "$BLUE" "Exiting. Please configure Gotify in your .env file."
            exit 0
        fi
        print_color "$YELLOW" "Continuing without Gotify (logs only mode)"
    else
        print_color "$GREEN" "✓ Gotify configuration found"
        print_color "$BLUE" "  URL: $GOTIFY_URL"
        # Test Gotify connection
        if command -v curl >/dev/null 2>&1; then
            if curl -s -f -X GET "$GOTIFY_URL/health" -H "X-Gotify-Key: $GOTIFY_TOKEN" >/dev/null 2>&1; then
                print_color "$GREEN" "✓ Gotify server is reachable"
            else
                print_color "$YELLOW" "⚠ Cannot reach Gotify server (notifications may fail)"
            fi
        fi
    fi
}

# Function to build the service
build_service() {
    print_color "$BLUE" "Building notification service..."

    if docker-compose build $SERVICE_NAME; then
        print_color "$GREEN" "✓ Service built successfully"
    else
        print_color "$RED" "✗ Failed to build service"
        exit 1
    fi
}

# Function to start the service
start_service() {
    print_color "$BLUE" "Starting notification service..."

    if docker-compose up -d $SERVICE_NAME; then
        print_color "$GREEN" "✓ Service started successfully"

        # Wait for service to initialize
        print_color "$BLUE" "Waiting for service to initialize..."
        for i in {1..10}; do
            if is_container_running; then
                # Check if service is healthy
                if docker exec $CONTAINER_NAME python -c "import sys; sys.exit(0)" 2>/dev/null; then
                    print_color "$GREEN" "✓ Service is running and healthy"
                    return 0
                fi
            fi
            sleep 1
        done

        # If we get here, service failed to start properly
        print_color "$RED" "✗ Service failed to start or become healthy"
        print_color "$YELLOW" "Checking logs for errors..."
        docker logs --tail 50 $CONTAINER_NAME 2>&1 | grep -E "(ERROR|CRITICAL|Failed)" || docker logs --tail 20 $CONTAINER_NAME
        exit 1
    else
        print_color "$RED" "✗ Failed to start service"
        exit 1
    fi
}

# Function to stop the service
stop_service() {
    print_color "$BLUE" "Stopping notification service..."

    if docker-compose stop $SERVICE_NAME; then
        print_color "$GREEN" "✓ Service stopped"
    else
        print_color "$YELLOW" "⚠ Service may not have been running"
    fi
}

# Function to restart the service
restart_service() {
    stop_service
    start_service
}

# Function to show service logs
show_logs() {
    lines=${1:-100}
    print_color "$BLUE" "Showing last $lines lines of logs..."
    docker logs --tail $lines -f $CONTAINER_NAME
}

# Function to run tests
run_tests() {
    print_color "$BLUE" "Running notification tests..."

    # Check if test script exists
    if [ ! -f "services/notification_service/test_notifications.py" ]; then
        print_color "$RED" "✗ Test script not found"
        exit 1
    fi

    # Check if container is running
    if ! is_container_running; then
        print_color "$RED" "✗ Container is not running. Start it first with: $0 start"
        exit 1
    fi

    # Run test suite
    print_color "$BLUE" "Publishing test events to Redis..."
    docker exec -it $CONTAINER_NAME python /app/services/notification_service/test_notifications.py --mode suite
}

# Function to show service status
show_status() {
    print_color "$BLUE" "Notification Service Status:"
    echo ""

    if is_container_running; then
        print_color "$GREEN" "✓ Container is running"

        # Show container details
        docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Status}}\t{{.Ports}}"

        # Show recent logs summary
        echo ""
        print_color "$BLUE" "Recent activity:"
        docker logs --tail 10 $CONTAINER_NAME 2>&1 | grep -E "(INFO|WARNING|ERROR)" || true
    else
        print_color "$RED" "✗ Container is not running"
    fi
}

# Main script
main() {
    print_color "$GREEN" "==================================="
    print_color "$GREEN" "  Notification Service Manager"
    print_color "$GREEN" "==================================="
    echo ""

    # Check prerequisites
    if ! command_exists docker; then
        print_color "$RED" "✗ Docker is not installed"
        exit 1
    fi

    if ! command_exists docker-compose; then
        print_color "$RED" "✗ Docker Compose is not installed"
        exit 1
    fi

    # Load environment variables
    if [ -f .env ]; then
        set -a
        source .env
        set +a
        print_color "$GREEN" "✓ Environment variables loaded from .env"
    elif [ -f ../.env ]; then
        set -a
        source ../.env
        set +a
        print_color "$GREEN" "✓ Environment variables loaded from ../.env"
    else
        print_color "$YELLOW" "⚠ No .env file found (using defaults)"
        export REDIS_URL=${REDIS_URL:-redis://localhost:6379}
        export LOG_LEVEL=${LOG_LEVEL:-INFO}
        export NOTIFICATION_COOLDOWN=${NOTIFICATION_COOLDOWN:-60}
    fi

    # Parse command line arguments
    case "${1:-}" in
        start)
            check_redis
            check_gotify_config
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            check_redis
            check_gotify_config
            restart_service
            ;;
        build)
            build_service
            ;;
        rebuild)
            build_service
            restart_service
            ;;
        logs)
            show_logs "${2:-100}"
            ;;
        test)
            if is_container_running; then
                run_tests
            else
                print_color "$RED" "✗ Service is not running"
                print_color "$YELLOW" "Start the service first: $0 start"
                exit 1
            fi
            ;;
        status)
            show_status
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|build|rebuild|logs|test|status}"
            echo ""
            echo "Commands:"
            echo "  start    - Start the notification service"
            echo "  stop     - Stop the notification service"
            echo "  restart  - Restart the notification service"
            echo "  build    - Build the service container"
            echo "  rebuild  - Rebuild and restart the service"
            echo "  logs [n] - Show last n lines of logs (default: 100)"
            echo "  test     - Run notification tests"
            echo "  status   - Show service status"
            echo ""
            echo "Examples:"
            echo "  $0 start           # Start the service"
            echo "  $0 logs 50         # Show last 50 log lines"
            echo "  $0 test            # Run test notifications"
            exit 1
            ;;
    esac

    echo ""
    print_color "$GREEN" "==================================="
    print_color "$GREEN" "  Operation completed"
    print_color "$GREEN" "==================================="
}

# Run main function
main "$@"
