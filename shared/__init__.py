"""
Shared models and utilities for the automated trading system.

This package provides common data models, configuration, and utilities
that are used across all services in the trading system.
"""

from .models import (
    MarketData,
    TradeSignal,
    PortfolioState,
    RiskParameters,
    Trade,
    Position,
    OrderRequest,
    OrderResponse,
)
from .config import Config
from .utils import setup_logging, get_logger

__version__ = "1.0.0"

__all__ = [
    "MarketData",
    "TradeSignal",
    "PortfolioState",
    "RiskParameters",
    "Trade",
    "Position",
    "OrderRequest",
    "OrderResponse",
    "Config",
    "setup_logging",
    "get_logger",
]
