"""
Mock Services Package for Integration Tests

This package contains mock implementations of trading system services
that maintain compatibility with real services while using historical
data and simulated behavior for integration testing.

The mock services:
- MockDataCollector: Serves historical data from parquet files
- ServiceOrchestrator: Manages real service instances in test environment

These services maintain the same Redis pub/sub communication patterns
and database interactions as the production system while providing
controlled, repeatable test environments.
"""

from .mock_data_collector import MockDataCollector, MockDataCollectorConfig
from .service_orchestrator import ServiceOrchestrator, create_service_orchestrator

__all__ = [
    "MockDataCollector",
    "MockDataCollectorConfig",
    "ServiceOrchestrator",
    "create_service_orchestrator",
]

__version__ = "1.0.0"
