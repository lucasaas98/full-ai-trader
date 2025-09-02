"""
Notification Service for the Full AI Trader System.

This service monitors Redis event streams and sends notifications
through various channels (Gotify, etc.) when important trading
events occur.
"""

from pathlib import Path

__version__ = "1.0.0"
__author__ = "Full AI Trader Team"
__description__ = "Event-driven notification service for trading system"

# Service metadata
SERVICE_NAME = "notification_service"
SERVICE_ROOT = Path(__file__).parent

# Default configuration
DEFAULT_REDIS_URL = "redis://localhost:6379"
DEFAULT_NOTIFICATION_COOLDOWN = 60  # seconds
DEFAULT_LOG_LEVEL = "INFO"

# Supported notification channels
SUPPORTED_CHANNELS = [
    "gotify",
    # Future channels can be added here
    # "slack",
    # "email",
    # "discord",
    # "telegram",
]

# Redis channels to monitor
MONITORED_CHANNELS = [
    "executions:all",  # All trade executions
    "execution_errors:all",  # All execution errors
    "signals:all",  # Trading signals
    "alerts:*",  # All system alerts
    "risk:alerts",  # Risk management alerts
    "portfolio:updates",  # Portfolio updates
    "system:status",  # System status updates
    "market:alerts",  # Market condition alerts
]

__all__ = [
    "SERVICE_NAME",
    "SERVICE_ROOT",
    "DEFAULT_REDIS_URL",
    "DEFAULT_NOTIFICATION_COOLDOWN",
    "DEFAULT_LOG_LEVEL",
    "SUPPORTED_CHANNELS",
    "MONITORED_CHANNELS",
    "__version__",
    "__author__",
    "__description__",
]
