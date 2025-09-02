"""
Shared models and utilities for the automated trading system.

This package provides common data models, configuration, and utilities
that are used across all services in the trading system.
"""

from .config import Config
from .models import (
    MarketData,
    OrderRequest,
    OrderResponse,
    PortfolioState,
    Position,
    RiskParameters,
    Trade,
    TradeSignal,
)
from .utils import get_logger, setup_logging

# Import market hours functionality
try:
    from .market_hours import (
        MarketDay,
        MarketHours,
        MarketSession,
        MarketStatus,
        get_market_status,
        get_next_market_close,
        get_next_market_open,
        is_market_open,
        is_market_open_sync,
        is_trading_day,
    )
except ImportError:
    # Market hours functionality not available
    MarketHours = None
    MarketStatus = None
    MarketSession = None
    MarketDay = None
    is_market_open = None
    get_market_status = None
    get_next_market_open = None
    get_next_market_close = None
    is_market_open_sync = None
    is_trading_day = None

# Import database managers if dependencies are available
try:
    from .database_manager import SharedDatabaseManager
except ImportError:
    SharedDatabaseManager = None

# Import simple database manager (requires only asyncpg)
try:
    from .simple_db_manager import SimpleDatabaseManager
except ImportError:
    SimpleDatabaseManager = None

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

# Add database managers to exports if available
if SharedDatabaseManager is not None:
    __all__.append("SharedDatabaseManager")
if SimpleDatabaseManager is not None:
    __all__.append("SimpleDatabaseManager")

# Add market hours functionality to exports if available
if MarketHours is not None:
    __all__.extend(
        [
            "MarketHours",
            "MarketStatus",
            "MarketSession",
            "MarketDay",
            "is_market_open",
            "get_market_status",
            "get_next_market_open",
            "get_next_market_close",
            "is_market_open_sync",
            "is_trading_day",
        ]
    )
