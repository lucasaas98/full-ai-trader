"""
Base Strategy Framework

This module provides the abstract base class for all trading strategies,
defining the common interface and core functionality that all strategies must implement.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl
from pydantic import BaseModel, Field

from shared.models import SignalType, TimeFrame


class TimeFrameMapper:
    """Utility class for mapping between strategy and data loader timeframe naming conventions."""

    # Mapping from strategy timeframes to TimeFrame enum values
    STRATEGY_TO_TIMEFRAME = {
        "1m": TimeFrame.ONE_MINUTE,
        "5m": TimeFrame.FIVE_MINUTES,
        "15m": TimeFrame.FIFTEEN_MINUTES,
        "30m": TimeFrame.THIRTY_MINUTES,
        "1h": TimeFrame.ONE_HOUR,
        "4h": TimeFrame.FOUR_HOURS,
        "1d": TimeFrame.ONE_DAY,
        "1w": TimeFrame.ONE_WEEK,
        "1M": TimeFrame.ONE_MONTH,
    }

    # Reverse mapping from TimeFrame enum to strategy timeframes
    TIMEFRAME_TO_STRATEGY = {v: k for k, v in STRATEGY_TO_TIMEFRAME.items()}

    # Available TimeFrame enum values
    AVAILABLE_TIMEFRAMES = list(TimeFrame)

    # Available strategy timeframes (those that map to TimeFrame enum)
    AVAILABLE_STRATEGY_TIMEFRAMES = list(STRATEGY_TO_TIMEFRAME.keys())

    @classmethod
    def strategy_to_timeframe_enum(
        cls, strategy_timeframes: List[str]
    ) -> List[TimeFrame]:
        """
        Convert strategy timeframes to TimeFrame enum values.

        Args:
            strategy_timeframes: List of strategy timeframe strings

        Returns:
            List of TimeFrame enum values
        """
        timeframes = []
        for tf in strategy_timeframes:
            timeframe_enum = cls.STRATEGY_TO_TIMEFRAME.get(tf)
            if timeframe_enum:
                timeframes.append(timeframe_enum)
        return timeframes

    @classmethod
    def strategy_to_data(cls, strategy_timeframes: List[str]) -> List[str]:
        """
        Convert strategy timeframes to data collector timeframe strings.

        Args:
            strategy_timeframes: List of strategy timeframe strings

        Returns:
            List of data collector timeframe strings (TimeFrame enum values)
        """
        data_timeframes = []
        for tf in strategy_timeframes:
            timeframe_enum = cls.STRATEGY_TO_TIMEFRAME.get(tf)
            if timeframe_enum:
                data_timeframes.append(timeframe_enum.value)
        return data_timeframes

    @classmethod
    def data_to_strategy(cls, data_timeframes: List[str]) -> List[str]:
        """
        Convert data collector timeframes to strategy timeframes.

        Args:
            data_timeframes: List of data collector timeframe strings

        Returns:
            List of strategy timeframe strings
        """
        strategy_timeframes = []
        for tf_str in data_timeframes:
            try:
                timeframe_enum = TimeFrame(tf_str)
                strategy_tf = cls.TIMEFRAME_TO_STRATEGY.get(timeframe_enum)
                if strategy_tf:
                    strategy_timeframes.append(strategy_tf)
            except ValueError:
                # Invalid timeframe string, skip it
                continue
        return strategy_timeframes

    @classmethod
    def validate_timeframes(
        cls, strategy_timeframes: List[str]
    ) -> tuple[List[str], List[str]]:
        """
        Validate strategy timeframes and split into available/unavailable.

        Args:
            strategy_timeframes: List of strategy timeframe strings to validate

        Returns:
            Tuple of (available_timeframes, unavailable_timeframes)
        """
        available = []
        unavailable = []

        for tf in strategy_timeframes:
            if tf in cls.AVAILABLE_STRATEGY_TIMEFRAMES:
                available.append(tf)
            else:
                unavailable.append(tf)

        return available, unavailable

    @classmethod
    def get_available_strategy_timeframes(cls) -> List[str]:
        """Get list of all available strategy timeframes."""
        return cls.AVAILABLE_STRATEGY_TIMEFRAMES.copy()

    @classmethod
    def get_available_data_timeframes(cls) -> List[str]:
        """Get list of all available data collector timeframes."""
        return [tf.value for tf in cls.AVAILABLE_TIMEFRAMES]


class StrategyMode(Enum):
    """Strategy execution modes."""

    DAY_TRADING = "day_trading"
    SWING_TRADING = "swing_trading"
    POSITION_TRADING = "position_trading"


class Signal(BaseModel):
    """Enhanced signal with detailed information."""

    action: SignalType
    confidence: float = Field(ge=0.0, le=100.0)
    entry_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    position_size: float = Field(ge=0.0, le=1.0)  # Percentage of portfolio
    reasoning: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Configuration parameters for strategies."""

    name: str
    mode: StrategyMode = StrategyMode.SWING_TRADING
    lookback_period: int = 50  # Days of historical data needed
    min_confidence: float = 60.0  # Minimum confidence to generate signal
    max_position_size: float = 0.20  # Maximum 20% of portfolio per position
    default_stop_loss_pct: float = 0.02  # 2% stop loss
    default_take_profit_pct: float = 0.04  # 4% take profit
    risk_reward_ratio: float = 2.0  # Minimum risk/reward ratio
    enable_short_selling: bool = False
    parameters: Optional[Dict[str, Any]] = None
    # Phase 3: Enhanced Strategy Configuration Support
    primary_timeframe: Optional[str] = None
    additional_timeframes: Optional[List[str]] = None
    custom_timeframes: Optional[List[str]] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class BacktestMetrics(BaseModel):
    """Backtesting performance metrics."""

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float  # In days
    volatility: float
    calmar_ratio: float
    sortino_ratio: float


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    This class defines the common interface that all strategies must implement,
    including signal generation, backtesting, and configuration management.
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize the strategy with configuration.

        Args:
            config: Strategy configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"strategy.{config.name}")
        self._initialized = False
        self._data_cache: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Get strategy name."""
        return self.config.name

    @property
    def mode(self) -> StrategyMode:
        """Get strategy mode."""
        return self.config.mode

    def initialize(self) -> None:
        """Initialize strategy-specific components."""
        if not self._initialized:
            self._setup_indicators()
            self._initialized = True
            self.logger.info(f"Strategy {self.name} initialized")

    @abstractmethod
    def _setup_indicators(self) -> None:
        """Setup technical indicators and other strategy components."""
        pass

    @abstractmethod
    async def analyze(self, symbol: str, data: pl.DataFrame) -> Signal:
        """
        Analyze market data and generate trading signal.

        Args:
            symbol: Trading symbol to analyze
            data: Historical market data as Polars DataFrame

        Returns:
            Trading signal with action, confidence, and metadata
        """
        pass

    async def analyze_multi_timeframe(
        self,
        symbol: str,
        multi_tf_data: Dict[str, pl.DataFrame],
        finviz_data: Optional[Any] = None,
    ) -> Signal:
        """
        Analyze market data across multiple timeframes and generate trading signal.

        This is an optional method that strategies can implement for multi-timeframe analysis.
        If not implemented, the strategy will fall back to single timeframe analysis.

        Args:
            symbol: Trading symbol to analyze
            multi_tf_data: Dictionary mapping timeframe -> DataFrame
            finviz_data: Optional fundamental data for hybrid strategies

        Returns:
            Trading signal with action, confidence, and metadata
        """
        # Default implementation: use primary timeframe for backward compatibility
        if not multi_tf_data:
            raise ValueError("No multi-timeframe data provided")

        # Use the first available timeframe as primary
        primary_data = next(iter(multi_tf_data.values()))
        return await self.analyze(symbol, primary_data)

    def get_entry_signal(self, symbol: str, data: pl.DataFrame) -> Optional[Signal]:
        """
        Generate entry signal for a position.

        Args:
            symbol: Trading symbol
            data: Market data

        Returns:
            Entry signal if conditions are met, None otherwise
        """
        if not self._initialized:
            self.initialize()

        try:
            # Run analysis synchronously for entry signals
            loop = asyncio.get_event_loop()
            signal = loop.run_until_complete(self.analyze(symbol, data))

            # Only return entry signals (BUY/SELL)
            if signal.action in [SignalType.BUY, SignalType.SELL]:
                if signal.confidence >= self.config.min_confidence:
                    return self._enhance_signal(symbol, signal, data)

        except Exception as e:
            self.logger.error(f"Error generating entry signal for {symbol}: {e}")

        return None

    def get_exit_signal(
        self, symbol: str, data: pl.DataFrame, current_position: Optional[Dict] = None
    ) -> Optional[Signal]:
        """
        Generate exit signal for an existing position.

        Args:
            symbol: Trading symbol
            data: Market data
            current_position: Information about current position

        Returns:
            Exit signal if conditions are met, None otherwise
        """
        if not self._initialized:
            self.initialize()

        try:
            loop = asyncio.get_event_loop()
            signal = loop.run_until_complete(self.analyze(symbol, data))

            # Check for exit conditions
            if current_position:
                exit_signal = self._check_exit_conditions(
                    signal, current_position, data
                )
                if exit_signal:
                    return exit_signal

            # Return HOLD or exit signals
            if signal.action in [SignalType.HOLD, SignalType.CLOSE]:
                return self._enhance_signal(symbol, signal, data)

        except Exception as e:
            self.logger.error(f"Error generating exit signal for {symbol}: {e}")

        return None

    def _check_exit_conditions(
        self, signal: Signal, position: Dict, data: pl.DataFrame
    ) -> Optional[Signal]:
        """Check if position should be exited based on risk management."""
        current_price = float(data.select("close").tail(1).item())
        entry_price = float(position.get("entry_price", 0))

        if entry_price == 0:
            return None

        position_type = position.get("side", "long")
        pnl_pct = (current_price - entry_price) / entry_price

        if position_type == "short":
            pnl_pct = -pnl_pct

        # Stop loss check
        if pnl_pct <= -self.config.default_stop_loss_pct:
            return Signal(
                action=SignalType.CLOSE,
                confidence=100.0,
                entry_price=Decimal(str(current_price)),
                position_size=1.0,
                reasoning=f"Stop loss triggered: {pnl_pct:.2%}",
            )

        # Take profit check
        if pnl_pct >= self.config.default_take_profit_pct:
            return Signal(
                action=SignalType.CLOSE,
                position_size=1.0,
                confidence=90.0,
                entry_price=Decimal(str(current_price)),
                reasoning=f"Take profit triggered: {pnl_pct:.2%}",
                metadata={"exit_reason": "take_profit", "pnl_pct": pnl_pct},
            )

        return None

    def _enhance_signal(
        self, symbol: str, signal: Signal, data: pl.DataFrame
    ) -> Signal:
        """Enhance signal with price levels and position sizing."""
        current_price = float(data.select("close").tail(1).item())

        # Set entry price if not set
        if signal.entry_price is None:
            signal.entry_price = Decimal(str(current_price))

        # Calculate stop loss and take profit if not set
        if signal.action in [SignalType.BUY, SignalType.SELL]:
            if signal.stop_loss is None:
                if signal.action == SignalType.BUY:
                    stop_loss = current_price * (1 - self.config.default_stop_loss_pct)
                else:  # SELL/SHORT
                    stop_loss = current_price * (1 + self.config.default_stop_loss_pct)
                signal.stop_loss = Decimal(str(stop_loss))

            if signal.take_profit is None:
                if signal.action == SignalType.BUY:
                    take_profit = current_price * (
                        1 + self.config.default_take_profit_pct
                    )
                else:  # SELL/SHORT
                    take_profit = current_price * (
                        1 - self.config.default_take_profit_pct
                    )
                signal.take_profit = Decimal(str(take_profit))

        # Calculate position size based on confidence and risk
        signal.position_size = self._calculate_position_size(signal.confidence)

        # Add strategy metadata
        signal.metadata.update(
            {
                "strategy_name": self.name,
                "strategy_mode": self.mode.value,
                "symbol": symbol,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return signal

    def _calculate_position_size(self, confidence: float) -> float:
        """
        Calculate position size based on confidence and risk parameters.

        Args:
            confidence: Signal confidence (0-100)

        Returns:
            Position size as percentage of portfolio (0.0-1.0)
        """
        # Scale position size with confidence
        confidence_factor = confidence / 100.0
        base_size = self.config.max_position_size * confidence_factor

        # Apply additional risk scaling based on strategy mode
        mode_multipliers = {
            StrategyMode.DAY_TRADING: 0.8,  # Lower size for day trading
            StrategyMode.SWING_TRADING: 1.0,  # Full size for swing trading
            StrategyMode.POSITION_TRADING: 1.2,  # Higher size for position trading
        }

        multiplier = mode_multipliers.get(self.mode, 1.0)
        position_size = min(base_size * multiplier, self.config.max_position_size)

        return round(position_size, 4)

    async def backtest(
        self,
        symbol: str,
        data: pl.DataFrame,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
    ) -> BacktestMetrics:
        """
        Backtest the strategy on historical data.

        Args:
            symbol: Trading symbol
            data: Historical market data
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital

        Returns:
            Backtesting performance metrics
        """
        self.logger.info(
            f"Starting backtest for {symbol} from {start_date} to {end_date}"
        )

        if not self._initialized:
            self.initialize()

        # Filter data to backtest period
        test_data = data.filter(
            (pl.col("timestamp") >= start_date) & (pl.col("timestamp") <= end_date)
        ).sort("timestamp")

        if test_data.height == 0:
            raise ValueError("No data available for backtest period")

        # Initialize backtest state
        capital = initial_capital
        position = None
        trades = []
        equity_curve = []

        # Sliding window backtest
        window_size = self.config.lookback_period

        for i in range(window_size, test_data.height):
            # Get data window for analysis
            window_data = test_data.slice(i - window_size, window_size + 1)
            current_row = test_data.slice(i, 1)
            current_price = float(current_row.select("close").item())
            current_time = current_row.select("timestamp").item()

            try:
                # Generate signal
                signal = await self.analyze(symbol, window_data)

                if signal.confidence < self.config.min_confidence:
                    continue

                # Execute trades based on signals
                if position is None and signal.action in [
                    SignalType.BUY,
                    SignalType.SELL,
                ]:
                    # Enter position
                    position = {
                        "entry_time": current_time,
                        "entry_price": current_price,
                        "side": "long" if signal.action == SignalType.BUY else "short",
                        "size": signal.position_size * capital / current_price,
                        "stop_loss": (
                            float(signal.stop_loss) if signal.stop_loss else None
                        ),
                        "take_profit": (
                            float(signal.take_profit) if signal.take_profit else None
                        ),
                    }

                elif position is not None:
                    # Check exit conditions
                    should_exit = False
                    exit_reason = ""

                    # Check stop loss and take profit
                    if position["side"] == "long":
                        if (
                            position["stop_loss"]
                            and current_price <= position["stop_loss"]
                        ):
                            should_exit = True
                            exit_reason = "stop_loss"
                        elif (
                            position["take_profit"]
                            and current_price >= position["take_profit"]
                        ):
                            should_exit = True
                            exit_reason = "take_profit"
                    else:  # short position
                        if (
                            position["stop_loss"]
                            and current_price >= position["stop_loss"]
                        ):
                            should_exit = True
                            exit_reason = "stop_loss"
                        elif (
                            position["take_profit"]
                            and current_price <= position["take_profit"]
                        ):
                            should_exit = True
                            exit_reason = "take_profit"

                    # Check signal-based exit
                    if (
                        signal.action == SignalType.CLOSE
                        or (
                            signal.action == SignalType.SELL
                            and position["side"] == "long"
                        )
                        or (
                            signal.action == SignalType.BUY
                            and position["side"] == "short"
                        )
                    ):
                        should_exit = True
                        exit_reason = "signal_exit"

                    # Execute exit
                    if should_exit:
                        pnl = self._calculate_pnl(position, current_price)
                        capital += pnl

                        trades.append(
                            {
                                "entry_time": position["entry_time"],
                                "exit_time": current_time,
                                "entry_price": position["entry_price"],
                                "exit_price": current_price,
                                "side": position["side"],
                                "size": position["size"],
                                "pnl": pnl,
                                "pnl_pct": pnl
                                / (position["entry_price"] * position["size"]),
                                "exit_reason": exit_reason,
                                "duration_days": (
                                    current_time - position["entry_time"]
                                ).days,
                            }
                        )

                        position = None

            except Exception as e:
                self.logger.error(f"Backtest error at {current_time}: {e}")
                continue

            # Record equity curve
            position_value = 0.0
            if position:
                position_value = self._calculate_position_value(position, current_price)

            equity_curve.append(
                {
                    "timestamp": current_time,
                    "equity": capital + position_value,
                    "capital": capital,
                    "position_value": position_value,
                }
            )

        # Close any remaining position
        if position is not None:
            final_price = float(test_data.select("close").tail(1).item())
            final_time = test_data.select("timestamp").tail(1).item()
            pnl = self._calculate_pnl(position, final_price)
            capital += pnl

            trades.append(
                {
                    "entry_time": position["entry_time"],
                    "exit_time": final_time,
                    "entry_price": position["entry_price"],
                    "exit_price": final_price,
                    "side": position["side"],
                    "size": position["size"],
                    "pnl": pnl,
                    "pnl_pct": pnl / (position["entry_price"] * position["size"]),
                    "exit_reason": "backtest_end",
                    "duration_days": (final_time - position["entry_time"]).days,
                }
            )

        # Calculate metrics
        metrics = self._calculate_backtest_metrics(
            trades, equity_curve, initial_capital, capital
        )

        self.logger.info(
            f"Backtest completed: {len(trades)} trades, "
            f"{metrics.total_return:.2%} return, "
            f"{metrics.win_rate:.1%} win rate"
        )

        return metrics

    def _calculate_pnl(self, position: Dict, current_price: float) -> float:
        """Calculate P&L for a position."""
        entry_price = position["entry_price"]
        size = position["size"]

        if position["side"] == "long":
            return (current_price - entry_price) * size
        else:  # short
            return (entry_price - current_price) * size

    def _calculate_position_value(self, position: Dict, current_price: float) -> float:
        """Calculate current value of a position."""
        return position["size"] * current_price

    def _calculate_backtest_metrics(
        self,
        trades: List[Dict],
        equity_curve: List[Dict],
        initial_capital: float,
        final_capital: float,
    ) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        if not trades:
            return BacktestMetrics(
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                average_win=0.0,
                average_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                avg_trade_duration=0.0,
                volatility=0.0,
                calmar_ratio=0.0,
                sortino_ratio=0.0,
            )

        # Basic metrics
        total_return = (final_capital - initial_capital) / initial_capital
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0

        # P&L metrics
        total_profit = sum(t["pnl"] for t in winning_trades)
        total_loss = sum(abs(t["pnl"]) for t in losing_trades)
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        avg_win = (
            sum(t["pnl"] for t in winning_trades) / len(winning_trades)
            if winning_trades
            else 0
        )
        avg_loss = (
            sum(t["pnl"] for t in losing_trades) / len(losing_trades)
            if losing_trades
            else 0
        )

        largest_win = max(t["pnl"] for t in winning_trades) if winning_trades else 0
        largest_loss = min(t["pnl"] for t in losing_trades) if losing_trades else 0

        # Duration metrics
        avg_duration = sum(t["duration_days"] for t in trades) / len(trades)

        # Risk metrics from equity curve
        equity_df = pd.DataFrame(equity_curve)
        equity_returns = equity_df["equity"].pct_change().dropna()

        # Annualized return
        days = (equity_curve[-1]["timestamp"] - equity_curve[0]["timestamp"]).days
        years = days / 365.25
        annualized_return = (
            (final_capital / initial_capital) ** (1 / years) - 1 if years > 0 else 0
        )

        # Volatility (annualized)
        volatility = equity_returns.std() * (252**0.5) if len(equity_returns) > 1 else 0

        # Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        peak = equity_df["equity"].expanding().max()
        drawdown = (equity_df["equity"] - peak) / peak
        max_drawdown = abs(drawdown.min())

        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

        # Sortino ratio
        negative_returns = equity_returns[equity_returns < 0]
        downside_deviation = (
            negative_returns.std() * (252**0.5) if len(negative_returns) > 1 else 0
        )
        sortino_ratio = (
            excess_return / downside_deviation if downside_deviation > 0 else 0
        )

        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            average_win=avg_win,
            average_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_duration,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
        )

    def validate_data(self, data: pl.DataFrame) -> bool:
        """
        Validate that the input data has required columns and sufficient length.

        Args:
            data: Market data to validate

        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]

        # Check required columns
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            self.logger.error(f"Missing required columns: {missing}")
            return False

        # Check minimum data length
        if data.height < self.config.lookback_period:
            self.logger.error(
                f"Insufficient data: {data.height} rows, "
                f"need {self.config.lookback_period}"
            )
            return False

        # Check for null values in critical columns
        null_counts = data.select(
            [
                pl.col("close").is_null().sum().alias("close_nulls"),
                pl.col("volume").is_null().sum().alias("volume_nulls"),
            ]
        ).row(0)

        if any(count > 0 for count in null_counts):
            self.logger.warning(f"Data contains null values: {null_counts}")

        return True

    def get_required_timeframes(self) -> List[str]:
        """
        Get list of timeframes required by this strategy.

        Returns:
            List of strategy timeframe strings (e.g., ['1m', '5m', '1h', '1d'])
        """
        # If custom timeframes are specified in config, use those
        if self.config.custom_timeframes:
            # Validate custom timeframes
            available, unavailable = TimeFrameMapper.validate_timeframes(
                self.config.custom_timeframes
            )
            if unavailable:
                self.logger.warning(f"Unavailable timeframes requested: {unavailable}")
            return available if available else self._get_default_timeframes()

        # If primary and additional timeframes are specified
        if self.config.primary_timeframe or self.config.additional_timeframes:
            timeframes = []
            if self.config.primary_timeframe:
                timeframes.append(self.config.primary_timeframe)
            if self.config.additional_timeframes:
                timeframes.extend(self.config.additional_timeframes)

            # Validate and return available timeframes
            available, unavailable = TimeFrameMapper.validate_timeframes(timeframes)
            if unavailable:
                self.logger.warning(f"Unavailable timeframes requested: {unavailable}")
            return available if available else self._get_default_timeframes()

        # Fall back to mode-based defaults
        return self._get_default_timeframes()

    def _get_default_timeframes(self) -> List[str]:
        """Get default timeframes based on strategy mode, filtered for availability."""
        # Updated to use only available timeframes
        timeframe_map = {
            StrategyMode.DAY_TRADING: ["1m", "5m", "15m"],
            StrategyMode.SWING_TRADING: ["15m", "1h", "1d"],
            StrategyMode.POSITION_TRADING: ["1h", "1d"],
        }

        default_timeframes = timeframe_map.get(self.mode, ["1h", "1d"])

        # Validate default timeframes and return only available ones
        available, unavailable = TimeFrameMapper.validate_timeframes(default_timeframes)
        if unavailable:
            self.logger.debug(f"Some default timeframes unavailable: {unavailable}")

        return available

    def get_required_data_timeframes(self) -> List[str]:
        """
        Get list of data timeframes required by this strategy (for data loader).

        Returns:
            List of data loader timeframe strings (e.g., ['1min', '5min', '1h', '1day'])
        """
        strategy_timeframes = self.get_required_timeframes()
        return TimeFrameMapper.strategy_to_data(strategy_timeframes)

    def validate_timeframe_availability(self) -> Dict[str, Any]:
        """
        Validate timeframe availability and return status.

        Returns:
            Dict with availability information
        """
        required_timeframes = self.get_required_timeframes()
        available, unavailable = TimeFrameMapper.validate_timeframes(
            required_timeframes
        )

        return {
            "required": required_timeframes,
            "available": available,
            "unavailable": unavailable,
            "all_available": len(unavailable) == 0,
            "data_timeframes": TimeFrameMapper.strategy_to_data(available),
        }

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and configuration."""
        return {
            "name": self.name,
            "mode": self.mode.value,
            "lookback_period": self.config.lookback_period,
            "min_confidence": self.config.min_confidence,
            "max_position_size": self.config.max_position_size,
            "stop_loss_pct": self.config.default_stop_loss_pct,
            "take_profit_pct": self.config.default_take_profit_pct,
            "risk_reward_ratio": self.config.risk_reward_ratio,
            "enable_short_selling": self.config.enable_short_selling,
            "parameters": self.config.parameters,
            "required_timeframes": self.get_required_timeframes(),
            "initialized": self._initialized,
        }

    def update_config(self, new_params: Dict[str, Any]) -> None:
        """
        Update strategy configuration parameters.

        Args:
            new_params: Dictionary of parameters to update
        """
        for key, value in new_params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated {key} to {value}")
            elif self.config.parameters and key in self.config.parameters:
                self.config.parameters[key] = value
                self.logger.info(f"Updated parameter {key} to {value}")
            else:
                self.logger.warning(f"Unknown parameter: {key}")

        # Re-initialize if parameters changed
        if self._initialized:
            self._initialized = False
            self.initialize()

    def save_state(self) -> Dict[str, Any]:
        """Save strategy state for persistence."""
        return {
            "config": {
                "name": self.config.name,
                "mode": self.config.mode.value,
                "lookback_period": self.config.lookback_period,
                "min_confidence": self.config.min_confidence,
                "max_position_size": self.config.max_position_size,
                "default_stop_loss_pct": self.config.default_stop_loss_pct,
                "default_take_profit_pct": self.config.default_take_profit_pct,
                "risk_reward_ratio": self.config.risk_reward_ratio,
                "enable_short_selling": self.config.enable_short_selling,
                "parameters": self.config.parameters,
            },
            "initialized": self._initialized,
            "cache_keys": list(self._data_cache.keys()) if self._data_cache else [],
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load strategy state from persistence."""
        if "config" in state:
            config_data = state["config"]
            self.config = StrategyConfig(
                name=config_data["name"],
                mode=StrategyMode(config_data["mode"]),
                lookback_period=config_data.get("lookback_period", 50),
                min_confidence=config_data.get("min_confidence", 60.0),
                max_position_size=config_data.get("max_position_size", 0.20),
                default_stop_loss_pct=config_data.get("default_stop_loss_pct", 0.02),
                default_take_profit_pct=config_data.get(
                    "default_take_profit_pct", 0.04
                ),
                risk_reward_ratio=config_data.get("risk_reward_ratio", 2.0),
                enable_short_selling=config_data.get("enable_short_selling", False),
                parameters=config_data.get("parameters", {}),
            )

        self._initialized = state.get("initialized", False)
        if self._initialized:
            self.initialize()

    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.name} ({self.mode.value})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"<{self.__class__.__name__}(name='{self.name}', "
            f"mode='{self.mode.value}', "
            f"min_confidence={self.config.min_confidence}, "
            f"initialized={self._initialized})>"
        )
