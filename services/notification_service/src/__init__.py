"""
Notification Service source module.

Contains the main notification service implementation that subscribes
to Redis events and sends notifications through configured channels.
"""

from .main import NotificationService

__all__ = ["NotificationService"]
