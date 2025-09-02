"""
Shared HTTP clients for inter-service communication.

This module provides HTTP clients for communicating between microservices
without tight coupling through direct imports.
"""

from .data_collector_client import DataCollectorClient

__all__ = ["DataCollectorClient"]
