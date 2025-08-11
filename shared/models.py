"""
Pydantic models for the automated trading system.

This module defines all the data models used across the trading system services,
including market data, trade signals, portfolio state, and risk parameters.
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict, field_serializer


class TimeFrame(str, Enum):
    """Time frame enumeration for market data."""
    ONE_MINUTE = "1min"
    FIVE_MINUTES = "5min"
    FIFTEEN_MINUTES = "15min"
    THIRTY_MINUTES = "30min"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1day"
    ONE_WEEK = "1week"
    ONE_MONTH = "1month"


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SignalType(str, Enum):
    """Trade signal type enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class AssetType(str, Enum):
    """Asset type enumeration."""
    STOCK = "stock"
    ETF = "etf"
    OPTION = "option"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"


class MarketData(BaseModel):
    """Market data model for OHLCV data."""

    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Data timestamp")
    timeframe: TimeFrame = Field(..., description="Data timeframe")
    open: Decimal = Field(..., ge=0, description="Open price")
    high: Decimal = Field(..., ge=0, description="High price")
    low: Decimal = Field(..., ge=0, description="Low price")
    close: Decimal = Field(..., ge=0, description="Close price")
    volume: int = Field(..., ge=0, description="Trading volume")
    adjusted_close: Optional[Decimal] = Field(None, ge=0, description="Adjusted close price")
    asset_type: AssetType = Field(default=AssetType.STOCK, description="Asset type")

    @field_validator('high')
    @classmethod
    def high_must_be_highest(cls, v, info):
        """Validate that high is the highest price."""
        if info.data:
            open_price = info.data.get('open', 0)
            low_price = info.data.get('low', 0)
            close_price = info.data.get('close', 0)
            if v < max(open_price, low_price, close_price):
                raise ValueError('high must be >= open, low, and close prices')
        return v

    @field_validator('low')
    @classmethod
    def low_must_be_lowest(cls, v, info):
        """Validate that low is the lowest price."""
        if info.data:
            open_price = info.data.get('open', float('inf'))
            high_price = info.data.get('high', float('inf'))
            close_price = info.data.get('close', float('inf'))
            if v > min(open_price, high_price, close_price):
                raise ValueError('low must be <= open, high, and close prices')
        return v

    model_config = ConfigDict()

    @field_serializer('open', 'high', 'low', 'close', 'adjusted_close')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class TradeSignal(BaseModel):
    """Trade signal model."""

    id: UUID = Field(default_factory=uuid4, description="Unique signal identifier")
    symbol: str = Field(..., description="Trading symbol")
    signal_type: SignalType = Field(..., description="Type of signal")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Signal generation time")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence score")
    price: Optional[Decimal] = Field(None, ge=0, description="Suggested entry/exit price")
    quantity: Optional[int] = Field(None, ge=0, description="Suggested quantity")
    stop_loss: Optional[Decimal] = Field(None, ge=0, description="Suggested stop loss price")
    take_profit: Optional[Decimal] = Field(None, ge=0, description="Suggested take profit price")
    strategy_name: str = Field(..., description="Name of the strategy that generated this signal")
    metadata: Dict[str, Union[str, int, float]] = Field(default_factory=dict, description="Additional signal metadata")

    model_config = ConfigDict()

    @field_serializer('id')
    def serialize_uuid(self, value: UUID) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('price', 'stop_loss', 'take_profit')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class Position(BaseModel):
    """Portfolio position model."""

    symbol: str = Field(..., description="Trading symbol")
    quantity: int = Field(..., description="Current position size (negative for short)")
    entry_price: Decimal = Field(..., ge=0, description="Average entry price")
    current_price: Decimal = Field(..., ge=0, description="Current market price")
    unrealized_pnl: Decimal = Field(..., description="Unrealized profit/loss")
    market_value: Decimal = Field(..., description="Current market value of position")
    cost_basis: Decimal = Field(..., description="Total cost basis")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp")

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    model_config = ConfigDict()

    @field_serializer('entry_price', 'current_price', 'unrealized_pnl', 'market_value', 'cost_basis')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('last_updated')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class PortfolioState(BaseModel):
    """Portfolio state model."""

    account_id: str = Field(..., description="Account identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Portfolio snapshot timestamp")
    cash: Decimal = Field(..., ge=0, description="Available cash")
    buying_power: Decimal = Field(..., ge=0, description="Available buying power")
    total_equity: Decimal = Field(..., ge=0, description="Total portfolio equity")
    positions: List[Position] = Field(default_factory=list, description="Current positions")
    day_trades_count: int = Field(default=0, ge=0, description="Number of day trades today")
    pattern_day_trader: bool = Field(default=False, description="Pattern day trader status")

    @property
    def total_market_value(self) -> Decimal:
        """Calculate total market value of all positions."""
        return sum(pos.market_value for pos in self.positions) or Decimal('0')

    @property
    def total_unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized PnL."""
        return sum(pos.unrealized_pnl for pos in self.positions) or Decimal('0')

    model_config = ConfigDict()

    @field_serializer('cash', 'buying_power', 'total_equity')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class RiskParameters(BaseModel):
    """Risk management parameters."""

    max_position_size: Decimal = Field(..., ge=0, le=1, description="Maximum position size as % of portfolio")
    max_daily_loss: Decimal = Field(..., ge=0, description="Maximum daily loss limit")
    max_total_exposure: Decimal = Field(..., ge=0, le=1, description="Maximum total exposure as % of portfolio")
    stop_loss_percentage: float = Field(..., ge=0, le=1, description="Default stop loss percentage")
    take_profit_percentage: float = Field(..., ge=0, description="Default take profit percentage")
    max_correlation: float = Field(default=0.7, ge=0, le=1, description="Maximum correlation between positions")
    min_trade_amount: Decimal = Field(..., ge=0, description="Minimum trade amount")
    max_trade_amount: Decimal = Field(..., ge=0, description="Maximum trade amount")

    @field_validator('max_trade_amount')
    @classmethod
    def max_trade_must_be_greater_than_min(cls, v, info):
        """Validate that max trade amount is greater than min trade amount."""
        if info.data and 'min_trade_amount' in info.data:
            if v <= info.data['min_trade_amount']:
                raise ValueError('max_trade_amount must be greater than min_trade_amount')
        return v

    model_config = ConfigDict()

    @field_serializer('max_position_size', 'max_daily_loss', 'max_total_exposure', 'min_trade_amount', 'max_trade_amount')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None


class OrderRequest(BaseModel):
    """Order request model."""

    id: Optional[UUID] = Field(default_factory=uuid4, description="Order request ID")
    symbol: str = Field(..., description="Trading symbol")
    side: OrderSide = Field(..., description="Order side (buy/sell)")
    quantity: int = Field(..., gt=0, description="Order quantity")
    order_type: OrderType = Field(..., description="Order type")
    price: Optional[Decimal] = Field(None, ge=0, description="Limit price (for limit orders)")
    stop_price: Optional[Decimal] = Field(None, ge=0, description="Stop price (for stop orders)")
    time_in_force: str = Field(default="day", description="Time in force (day, gtc, ioc, fok)")
    extended_hours: bool = Field(default=False, description="Allow extended hours trading")
    client_order_id: Optional[str] = Field(None, description="Client-specified order ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Order creation timestamp")

    model_config = ConfigDict()

    @field_serializer('id')
    def serialize_uuid(self, value: UUID) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('price', 'stop_price')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('created_at')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class OrderResponse(BaseModel):
    """Order response model."""

    id: UUID = Field(..., description="Order ID")
    broker_order_id: Optional[str] = Field(None, description="Broker-specific order ID")
    symbol: str = Field(..., description="Trading symbol")
    side: OrderSide = Field(..., description="Order side")
    quantity: int = Field(..., description="Order quantity")
    filled_quantity: int = Field(default=0, description="Filled quantity")
    order_type: OrderType = Field(..., description="Order type")
    status: OrderStatus = Field(..., description="Order status")
    price: Optional[Decimal] = Field(None, description="Order price")
    filled_price: Optional[Decimal] = Field(None, description="Average fill price")
    submitted_at: datetime = Field(..., description="Order submission timestamp")
    filled_at: Optional[datetime] = Field(None, description="Order fill timestamp")
    cancelled_at: Optional[datetime] = Field(None, description="Order cancellation timestamp")
    commission: Optional[Decimal] = Field(None, description="Commission paid")

    model_config = ConfigDict()

    @field_serializer('id')
    def serialize_uuid(self, value: UUID) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('commission', 'price', 'filled_price')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('submitted_at', 'filled_at', 'cancelled_at')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class Trade(BaseModel):
    """Completed trade model."""

    id: UUID = Field(default_factory=uuid4, description="Unique trade identifier")
    symbol: str = Field(..., description="Trading symbol")
    side: OrderSide = Field(..., description="Trade side")
    quantity: int = Field(..., gt=0, description="Trade quantity")
    price: Decimal = Field(..., ge=0, description="Execution price")
    commission: Decimal = Field(default=Decimal("0"), ge=0, description="Commission paid")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Trade execution timestamp")
    order_id: UUID = Field(..., description="Associated order ID")
    strategy_name: str = Field(..., description="Strategy that initiated the trade")
    pnl: Optional[Decimal] = Field(None, description="Realized PnL (for closing trades)")
    fees: Optional[Decimal] = Field(None, description="Additional fees")

    @property
    def gross_value(self) -> Decimal:
        """Calculate gross trade value."""
        return self.price * self.quantity

    @property
    def net_value(self) -> Decimal:
        """Calculate net trade value after commission."""
        return self.gross_value - self.commission

    model_config = ConfigDict()

    @field_serializer('id', 'order_id')
    def serialize_uuid(self, value: UUID) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('price', 'commission', 'pnl', 'fees')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class FinVizData(BaseModel):
    """FinViz screener data model."""

    ticker: str = Field(..., description="Stock ticker symbol")
    symbol: str = Field(..., description="Trading symbol (alias for ticker)")
    company: Optional[str] = Field(None, description="Company name")
    sector: Optional[str] = Field(None, description="Company sector")
    industry: Optional[str] = Field(None, description="Company industry")
    country: Optional[str] = Field(None, description="Company country")
    market_cap: Optional[Decimal] = Field(None, ge=0, description="Market capitalization")
    pe_ratio: Optional[float] = Field(None, description="Price-to-earnings ratio")
    price: Optional[Decimal] = Field(None, ge=0, description="Current price")
    change: Optional[float] = Field(None, description="Price change percentage")
    volume: Optional[int] = Field(None, ge=0, description="Trading volume")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Data timestamp")

    model_config = ConfigDict()

    @field_serializer('market_cap', 'price')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators."""

    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Calculation timestamp")

    # Moving averages
    sma_20: Optional[float] = Field(None, description="20-period Simple Moving Average")
    sma_50: Optional[float] = Field(None, description="50-period Simple Moving Average")
    sma_200: Optional[float] = Field(None, description="200-period Simple Moving Average")
    ema_12: Optional[float] = Field(None, description="12-period Exponential Moving Average")
    ema_26: Optional[float] = Field(None, description="26-period Exponential Moving Average")

    # Momentum indicators
    rsi: Optional[float] = Field(None, ge=0, le=100, description="Relative Strength Index")
    macd: Optional[float] = Field(None, description="MACD line")
    macd_signal: Optional[float] = Field(None, description="MACD signal line")
    macd_histogram: Optional[float] = Field(None, description="MACD histogram")

    # Volatility indicators
    bollinger_upper: Optional[float] = Field(None, description="Bollinger Bands upper band")
    bollinger_middle: Optional[float] = Field(None, description="Bollinger Bands middle band")
    bollinger_lower: Optional[float] = Field(None, description="Bollinger Bands lower band")
    atr: Optional[float] = Field(None, ge=0, description="Average True Range")

    # Volume indicators
    volume_sma: Optional[float] = Field(None, ge=0, description="Volume Simple Moving Average")

    model_config = ConfigDict()

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class BacktestResult(BaseModel):
    """Backtest result model."""

    strategy_name: str = Field(..., description="Strategy name")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: Decimal = Field(..., ge=0, description="Initial capital")
    final_capital: Decimal = Field(..., ge=0, description="Final capital")
    total_return: Decimal = Field(..., description="Total return")
    total_return_pct: float = Field(..., description="Total return percentage")
    annualized_return: float = Field(..., description="Annualized return")
    max_drawdown: Decimal = Field(..., description="Maximum drawdown")
    max_drawdown_pct: float = Field(..., description="Maximum drawdown percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate")
    total_trades: int = Field(..., ge=0, description="Total number of trades")
    winning_trades: int = Field(..., ge=0, description="Number of winning trades")
    losing_trades: int = Field(..., ge=0, description="Number of losing trades")
    avg_win: Decimal = Field(..., description="Average winning trade amount")
    avg_loss: Decimal = Field(..., description="Average losing trade amount")
    profit_factor: float = Field(..., description="Profit factor")

    @field_validator('winning_trades', 'losing_trades')
    @classmethod
    def validate_trade_counts(cls, v, info):
        """Validate trade counts."""
        if info.data and 'total_trades' in info.data:
            total = info.data['total_trades']
            if info.field_name == 'winning_trades':
                other_field = info.data.get('losing_trades', 0)
            else:
                other_field = info.data.get('winning_trades', 0)

            if v + other_field > total:
                raise ValueError('Sum of winning and losing trades cannot exceed total trades')
        return v

    model_config = ConfigDict()

    @field_serializer('initial_capital', 'final_capital', 'total_return', 'max_drawdown', 'avg_win', 'avg_loss')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('start_date', 'end_date')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class SystemHealth(BaseModel):
    """System health monitoring model."""

    service_name: str = Field(..., description="Name of the service")
    status: str = Field(..., description="Service status (healthy, degraded, unhealthy)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Health check timestamp")
    cpu_usage: Optional[float] = Field(None, ge=0, le=100, description="CPU usage percentage")
    memory_usage: Optional[float] = Field(None, ge=0, le=100, description="Memory usage percentage")
    disk_usage: Optional[float] = Field(None, ge=0, le=100, description="Disk usage percentage")
    last_error: Optional[str] = Field(None, description="Last error message")
    uptime_seconds: Optional[int] = Field(None, ge=0, description="Service uptime in seconds")

    model_config = ConfigDict()

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class Notification(BaseModel):
    """Notification model for alerts and updates."""

    id: UUID = Field(default_factory=uuid4, description="Unique notification identifier")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    priority: int = Field(default=5, ge=1, le=10, description="Notification priority (1=low, 10=critical)")
    service: str = Field(..., description="Service that generated the notification")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Notification timestamp")
    tags: List[str] = Field(default_factory=list, description="Notification tags")
    metadata: Dict[str, Union[str, int, float]] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict()

    @field_serializer('id')
    def serialize_uuid(self, value: UUID) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class MarketHours(BaseModel):
    """Market hours model."""

    date: datetime = Field(..., description="Trading date")
    market_open: Optional[datetime] = Field(None, description="Market open time")
    market_close: Optional[datetime] = Field(None, description="Market close time")
    pre_market_start: Optional[datetime] = Field(None, description="Pre-market start time")
    pre_market_end: Optional[datetime] = Field(None, description="Pre-market end time")
    after_hours_start: Optional[datetime] = Field(None, description="After hours start time")
    after_hours_end: Optional[datetime] = Field(None, description="After hours end time")
    is_holiday: bool = Field(default=False, description="Whether it's a market holiday")

    model_config = ConfigDict()

    @field_serializer('date', 'market_open', 'market_close', 'pre_market_start', 'pre_market_end', 'after_hours_start', 'after_hours_end')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class StrategyPerformance(BaseModel):
    """Strategy performance metrics."""

    strategy_name: str = Field(..., description="Strategy name")
    total_return: Decimal = Field(..., description="Total return")
    return_percentage: float = Field(..., description="Return percentage")
    volatility: float = Field(..., description="Strategy volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: Decimal = Field(..., description="Maximum drawdown")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate")
    total_trades: int = Field(..., ge=0, description="Total trades executed")
    avg_trade_duration: float = Field(..., ge=0, description="Average trade duration in days")
    max_consecutive_wins: int = Field(..., ge=0, description="Maximum consecutive winning trades")
    max_consecutive_losses: int = Field(..., ge=0, description="Maximum consecutive losing trades")

    model_config = ConfigDict()

    @field_serializer('total_return', 'max_drawdown')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None


class RiskEventType(str, Enum):
    """Risk event type enumeration."""
    POSITION_LIMIT_BREACH = "position_limit_breach"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    PORTFOLIO_CONCENTRATION = "portfolio_concentration"
    CORRELATION_BREACH = "correlation_breach"
    VOLATILITY_SPIKE = "volatility_spike"
    DRAWDOWN_LIMIT = "drawdown_limit"
    MARGIN_CALL = "margin_call"
    LIQUIDITY_RISK = "liquidity_risk"
    SYSTEM_ERROR = "system_error"
    PORTFOLIO_DRAWDOWN = "portfolio_drawdown"
    POSITION_SIZE_VIOLATION = "position_size_violation"
    POSITION_LIMIT_REACHED = "position_limit_reached"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    EMERGENCY_STOP = "emergency_stop"
    DATA_QUALITY = "data_quality"
    COMPLIANCE_VIOLATION = "compliance_violation"
    EXTERNAL_RISK = "external_risk"
    TAKE_PROFIT_TRIGGERED = "take_profit_triggered"


class RiskSeverity(str, Enum):
    """Risk severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PositionSizingMethod(str, Enum):
    """Position sizing method enumeration."""
    FIXED_AMOUNT = "fixed_amount"
    FIXED_PERCENTAGE = "fixed_percentage"
    VOLATILITY_BASED = "volatility_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"
    CONFIDENCE_BASED = "confidence_based"
    EQUAL_WEIGHT = "equal_weight"


class RiskEvent(BaseModel):
    """Risk event model."""

    id: Optional[UUID] = Field(default_factory=uuid4, description="Event ID")
    event_type: RiskEventType = Field(..., description="Type of risk event")
    severity: RiskSeverity = Field(..., description="Event severity")
    symbol: Optional[str] = Field(None, description="Related trading symbol")
    description: str = Field(..., description="Event description")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Event timestamp")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    action_taken: Optional[str] = Field(None, description="Action taken to resolve")
    metadata: Dict = Field(default_factory=dict, description="Additional event data")

    model_config = ConfigDict()

    @field_serializer('timestamp', 'resolved_at')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class PositionSizing(BaseModel):
    """Position sizing model."""

    symbol: str = Field(..., description="Trading symbol")
    method: PositionSizingMethod = Field(default=PositionSizingMethod.FIXED_PERCENTAGE, description="Sizing method")
    target_amount: Decimal = Field(default=Decimal("0"), description="Target position amount")
    max_position_size: Decimal = Field(default=Decimal("0"), description="Maximum position size")
    risk_amount: Decimal = Field(default=Decimal("0"), description="Amount at risk")
    stop_loss_price: Decimal = Field(default=Decimal("0"), description="Stop loss price")
    entry_price: Decimal = Field(default=Decimal("0"), description="Entry price")
    position_size: int = Field(default=0, description="Calculated position size")
    max_loss_amount: Decimal = Field(default=Decimal("0"), description="Maximum loss amount with stop loss")
    risk_reward_ratio: float = Field(default=0.0, description="Risk to reward ratio")

    # Legacy fields for backward compatibility
    recommended_shares: int = Field(default=0, description="Recommended number of shares")
    recommended_value: Decimal = Field(default=Decimal("0"), description="Recommended position value")
    position_percentage: Decimal = Field(default=Decimal("0"), description="Position as percentage of portfolio")
    confidence_adjustment: float = Field(default=1.0, description="Confidence adjustment factor")
    volatility_adjustment: float = Field(default=1.0, description="Volatility adjustment factor")
    sizing_method: PositionSizingMethod = Field(default=PositionSizingMethod.FIXED_PERCENTAGE, description="Legacy sizing method field")

    model_config = ConfigDict()

    @field_serializer('target_amount', 'max_position_size', 'risk_amount', 'stop_loss_price', 'entry_price', 'max_loss_amount', 'recommended_value', 'position_percentage')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None


class PortfolioMetrics(BaseModel):
    """Portfolio-wide risk metrics."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Metrics timestamp")
    total_exposure: Decimal = Field(default=Decimal("0"), description="Total portfolio exposure")
    cash_percentage: Decimal = Field(default=Decimal("0"), description="Cash as percentage of portfolio")
    position_count: int = Field(default=0, description="Number of open positions")
    concentration_risk: float = Field(default=0.0, description="Concentration risk score (0-1)")
    portfolio_beta: float = Field(default=0.0, description="Portfolio beta vs market")
    portfolio_correlation: float = Field(default=0.0, description="Average correlation between positions")
    value_at_risk_1d: Decimal = Field(default=Decimal("0"), description="1-day Value at Risk (95% confidence)")
    value_at_risk_5d: Decimal = Field(default=Decimal("0"), description="5-day Value at Risk (95% confidence)")
    expected_shortfall_1d: Decimal = Field(default=Decimal("0"), description="1-day Expected Shortfall (95% confidence)")
    expected_shortfall_5d: Decimal = Field(default=Decimal("0"), description="5-day Expected Shortfall (95% confidence)")
    maximum_drawdown: Decimal = Field(default=Decimal("0"), description="Maximum drawdown since inception")
    current_drawdown: Decimal = Field(default=Decimal("0"), description="Current drawdown from peak")

    # Legacy fields for backward compatibility
    expected_shortfall: Decimal = Field(default=Decimal("0"), description="Legacy expected shortfall field")
    sharpe_ratio: float = Field(default=0.0, description="Sharpe ratio")
    max_drawdown: Decimal = Field(default=Decimal("0"), description="Legacy max drawdown field")
    volatility: float = Field(default=0.0, description="Portfolio volatility (annualized)")

    model_config = ConfigDict()

    @field_serializer('total_exposure', 'cash_percentage', 'value_at_risk_1d', 'value_at_risk_5d', 'expected_shortfall_1d', 'expected_shortfall_5d', 'maximum_drawdown', 'current_drawdown', 'expected_shortfall', 'max_drawdown')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class RiskFilter(BaseModel):
    """Risk filter for trade evaluation."""

    name: str = Field(..., description="Filter name")
    enabled: bool = Field(default=True, description="Whether filter is enabled")
    max_position_size: Optional[Decimal] = Field(None, description="Maximum position size")
    max_sector_exposure: Optional[float] = Field(None, description="Maximum sector exposure")
    min_liquidity: Optional[int] = Field(None, description="Minimum daily volume")
    max_volatility: Optional[float] = Field(None, description="Maximum volatility threshold")

    # Fields for filter result tracking
    passed: bool = Field(default=True, description="Whether the filter passed")
    filter_name: str = Field(default="", description="Display name of the filter")
    reason: Optional[str] = Field(None, description="Reason for filter failure")
    value: Optional[float] = Field(None, description="Actual value being tested")
    limit: Optional[float] = Field(None, description="Limit that was exceeded")
    severity: str = Field(default="LOW", description="Severity level of filter violation")


class TrailingStop(BaseModel):
    """Trailing stop configuration."""

    symbol: str = Field(..., description="Trading symbol")
    enabled: bool = Field(default=False, description="Whether trailing stop is enabled")
    trail_percentage: Decimal = Field(..., description="Trailing percentage")
    current_stop_price: Decimal = Field(..., description="Current stop price")
    highest_price: Decimal = Field(..., description="Highest price since entry")
    entry_price: Decimal = Field(..., description="Original entry price")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp")

    model_config = ConfigDict()

    @field_serializer('trail_percentage', 'current_stop_price', 'highest_price', 'entry_price')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('last_updated')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class RiskLimits(BaseModel):
    """Risk limits configuration."""

    max_portfolio_risk: Decimal = Field(default=Decimal("0.02"), description="Maximum portfolio risk per trade")
    max_position_size: Decimal = Field(default=Decimal("0.10"), description="Maximum position size as % of portfolio")
    max_daily_loss: Decimal = Field(default=Decimal("0.03"), description="Maximum daily loss as % of portfolio")
    max_drawdown: Decimal = Field(default=Decimal("0.10"), description="Maximum drawdown threshold")
    max_correlation: float = Field(default=0.70, description="Maximum correlation between positions")
    max_sector_concentration: float = Field(default=0.25, description="Maximum sector concentration")
    max_single_position: Decimal = Field(default=Decimal("0.05"), description="Maximum single position size")
    min_liquidity_requirement: int = Field(default=100000, description="Minimum daily volume requirement")
    max_leverage: float = Field(default=1.0, description="Maximum leverage allowed")
    stress_test_scenarios: List[str] = Field(default_factory=list, description="Stress test scenarios to run")
    var_confidence_level: float = Field(default=0.95, description="VaR confidence level")
    var_holding_period: int = Field(default=1, description="VaR holding period in days")

    # Volatility limits
    max_individual_volatility: float = Field(default=0.40, description="Maximum individual position volatility")
    max_portfolio_volatility: float = Field(default=0.20, description="Maximum portfolio volatility (annualized)")
    max_position_volatility: float = Field(default=0.50, description="Maximum position volatility (annualized)")
    volatility_lookback_days: int = Field(default=20, description="Volatility calculation lookback period")

    # Legacy fields for backward compatibility
    max_position_percentage: Decimal = Field(default=Decimal("0.20"), description="Maximum position percentage")
    stop_loss_percentage: Decimal = Field(default=Decimal("0.02"), description="Default stop loss percentage")
    take_profit_percentage: Decimal = Field(default=Decimal("0.04"), description="Default take profit percentage")
    max_positions: int = Field(default=20, description="Maximum number of positions")
    max_drawdown_percentage: Decimal = Field(default=Decimal("0.10"), description="Maximum drawdown percentage")
    max_correlation_threshold: float = Field(default=0.70, description="Maximum correlation threshold")
    max_daily_loss_percentage: Decimal = Field(default=Decimal("0.03"), description="Maximum daily loss percentage")
    emergency_stop_percentage: Decimal = Field(default=Decimal("0.05"), description="Emergency stop percentage")

    model_config = ConfigDict()

    @field_serializer('max_portfolio_risk', 'max_position_size', 'max_daily_loss', 'max_drawdown', 'max_sector_concentration', 'max_single_position', 'max_position_percentage', 'stop_loss_percentage', 'take_profit_percentage', 'max_drawdown_percentage', 'max_daily_loss_percentage', 'emergency_stop_percentage')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None


class PositionRisk(BaseModel):
    """Individual position risk metrics."""

    symbol: str = Field(..., description="Trading symbol")
    position_size: Decimal = Field(default=Decimal("0"), description="Position size")
    market_value: Decimal = Field(default=Decimal("0"), description="Current market value")
    unrealized_pnl: Decimal = Field(default=Decimal("0"), description="Unrealized P&L")
    daily_var: Decimal = Field(default=Decimal("0"), description="Daily Value at Risk")
    volatility: float = Field(default=0.0, description="Position volatility")
    beta: Optional[float] = Field(None, description="Beta vs market")
    correlation_to_portfolio: float = Field(default=0.0, description="Correlation to rest of portfolio")
    liquidity_score: float = Field(default=0.0, description="Liquidity score (0-1)")

    # Legacy fields for backward compatibility
    portfolio_percentage: float = Field(default=0.0, description="Position as percentage of portfolio")
    var_1d: Decimal = Field(default=Decimal("0"), description="1-day VaR")
    expected_return: float = Field(default=0.0, description="Expected return")
    correlation_with_portfolio: float = Field(default=0.0, description="Correlation with portfolio")
    sharpe_ratio: float = Field(default=0.0, description="Sharpe ratio")
    sector: Optional[str] = Field(None, description="Stock sector")
    risk_score: float = Field(default=0.0, description="Overall risk score (0-10)")

    model_config = ConfigDict()

    @field_serializer('position_size', 'market_value', 'unrealized_pnl', 'daily_var', 'var_1d')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None


class RiskAlert(BaseModel):
    """Risk alert model."""

    id: Optional[UUID] = Field(default_factory=uuid4, description="Alert ID")
    alert_type: RiskEventType = Field(..., description="Type of risk alert")
    severity: RiskSeverity = Field(..., description="Alert severity")
    symbol: Optional[str] = Field(None, description="Related symbol")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Alert timestamp")
    acknowledged: bool = Field(default=False, description="Whether alert was acknowledged")
    action_required: bool = Field(default=False, description="Whether immediate action is required")
    metadata: Dict = Field(default_factory=dict, description="Additional alert data")

    model_config = ConfigDict()

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None


class DailyRiskReport(BaseModel):
    """Daily risk management report."""

    date: datetime = Field(..., description="Report date")
    portfolio_value: Decimal = Field(..., description="End-of-day portfolio value")
    daily_pnl: Decimal = Field(..., description="Daily profit/loss")
    daily_return: float = Field(..., description="Daily return percentage")
    max_drawdown: Decimal = Field(..., description="Maximum drawdown")
    current_drawdown: Decimal = Field(..., description="Current drawdown")
    volatility: float = Field(..., description="Portfolio volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    var_1d: Decimal = Field(..., description="1-day Value at Risk")
    total_trades: int = Field(..., description="Total trades executed")
    winning_trades: int = Field(..., description="Winning trades")
    risk_events: List[RiskEvent] = Field(default_factory=list, description="Risk events for the day")
    position_risks: List[PositionRisk] = Field(default_factory=list, description="Individual position risks")
    compliance_violations: List[str] = Field(default_factory=list, description="Compliance violations")

    model_config = ConfigDict()

    @field_serializer('portfolio_value', 'daily_pnl', 'max_drawdown', 'current_drawdown', 'var_1d')
    def serialize_decimal(self, value: Decimal) -> str | None:
        return str(value) if value is not None else None

    @field_serializer('date')
    def serialize_datetime(self, value: datetime) -> str | None:
        return value.isoformat() if value is not None else None
