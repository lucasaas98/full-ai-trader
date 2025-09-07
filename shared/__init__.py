"""
Shared models and utilities for the automated trading system.

This package provides common data models, configuration, and utilities
that are used across all services in the trading system.
"""

from datetime import date, datetime
from typing import Optional

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
        MarketHoursService,
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
    MarketHoursService = None  # type: ignore
    MarketStatus = None  # type: ignore
    MarketSession = None  # type: ignore
    MarketDay = None  # type: ignore

    async def is_market_open(timestamp: datetime | None = None) -> bool:
        return False

    async def get_market_status() -> MarketStatus:
        return None  # type: ignore

    async def get_next_market_open() -> Optional[datetime]:
        return None

    async def get_next_market_close() -> Optional[datetime]:
        return None

    def is_market_open_sync(timestamp: datetime | None = None) -> bool:
        return False

    async def is_trading_day(check_date: date | None = None) -> bool:
        return False


# Import database managers if dependencies are available
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .database_manager import SharedDatabaseManager
    from .simple_db_manager import SimpleDatabaseManager
else:
    try:
        from .database_manager import SharedDatabaseManager
    except ImportError:
        SharedDatabaseManager = None

    # Simple database manager fallback
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
if MarketHoursService is not None:
    __all__.extend(
        [
            "MarketHoursService",
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
