"""
Standalone Backtesting Models

This module provides simplified, standalone models for backtesting that don't
depend on the main system configuration to avoid import issues during backtesting.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TimeFrame(str, Enum):
    """Time frame enumeration for market data."""
    ONE_MINUTE = "1min"
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"
    THIRTY_MIN = "30min"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1day"
    ONE_WEEK = "1week"
    ONE_MONTH = "1month"


class SignalType(str, Enum):
    """Trade signal type enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class MarketData(BaseModel):
    """Market data for a single time period."""
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    adjusted_close: Decimal
    volume: int
    timeframe: TimeFrame


class Signal(BaseModel):
    """Trading signal with detailed information."""
    symbol: str
    action: SignalType
    confidence: float = Field(ge=0.0, le=100.0)
    entry_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    position_size: float = Field(ge=0.0, le=1.0)  # Percentage of portfolio
    reasoning: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BacktestPosition(BaseModel):
    """Position during backtesting."""
    symbol: str
    quantity: int
    entry_price: Decimal
    entry_date: datetime
    current_price: Decimal = Decimal('0')
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    unrealized_pnl: Decimal = Decimal('0')

    @property
    def market_value(self) -> Decimal:
        """Current market value of position."""
        return self.current_price * Decimal(str(abs(self.quantity)))

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0


class BacktestTrade(BaseModel):
    """Completed trade record."""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: Decimal
    exit_price: Decimal
    quantity: int
    pnl: Decimal
    pnl_percentage: Decimal
    commission: Decimal
    hold_days: int
    strategy_reasoning: str
    confidence: float


class BacktestResults(BaseModel):
    """Comprehensive backtesting results."""
    # Performance metrics
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float

    # Portfolio metrics
    final_portfolio_value: Decimal
    max_portfolio_value: Decimal
    min_portfolio_value: Decimal

    # Execution metrics
    total_commissions: Decimal
    total_slippage: Decimal
    execution_time_seconds: float

    # Trade details
    trades: List[BacktestTrade]
    daily_returns: List[float]
    portfolio_values: List[tuple]  # (datetime, Decimal) pairs

    # AI Strategy metrics
    total_ai_calls: int
    average_confidence: float
    signals_generated: int
    signals_executed: int


class FinVizData(BaseModel):
    """FinViz screener data model."""
    symbol: str
    company: str
    sector: str
    industry: str
    country: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    price: Optional[float] = None
    change: Optional[float] = None
    volume: Optional[int] = None
    screener_type: str
    timestamp: datetime = Field(default_factory=datetime.now)


class TechnicalIndicators(BaseModel):
    """Technical indicators data model."""
    symbol: str
    timestamp: datetime
    timeframe: TimeFrame

    # Moving averages
    sma_20: Optional[Decimal] = None
    sma_50: Optional[Decimal] = None
    sma_200: Optional[Decimal] = None
    ema_12: Optional[Decimal] = None
    ema_26: Optional[Decimal] = None

    # Momentum indicators
    rsi: Optional[float] = None
    macd_line: Optional[Decimal] = None
    macd_signal: Optional[Decimal] = None
    macd_histogram: Optional[Decimal] = None

    # Volatility indicators
    bollinger_upper: Optional[Decimal] = None
    bollinger_middle: Optional[Decimal] = None
    bollinger_lower: Optional[Decimal] = None
    atr: Optional[Decimal] = None

    # Volume indicators
    volume_sma: Optional[float] = None
    volume_ratio: Optional[float] = None


class AssetType(str, Enum):
    """Asset type enumeration."""
    STOCK = "stock"
    ETF = "etf"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
