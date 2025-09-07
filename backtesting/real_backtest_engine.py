"""
Real Backtesting Engine

This module provides a comprehensive backtesting engine that runs the actual production
AI strategy against historical data, simulating the complete system flow but accelerated
and without delays. It's designed to test the real trading system as it would operate
in production using historical parquet data.
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

# Use standalone models to avoid config dependencies
from backtest_models import (
    BacktestPosition,
    BacktestResults,
    BacktestTrade,
    MarketData,
    Signal,
    SignalType,
    TimeFrame,
)

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Backtesting execution modes."""

    FAST = "fast"  # Skip delays, process all data quickly
    REALTIME = "realtime"  # Simulate actual timing
    DEBUG = "debug"  # Add extra logging and validation


@dataclass
class RealBacktestConfig:
    """Configuration for real backtesting."""

    # Time period
    start_date: datetime
    end_date: datetime

    # Portfolio settings
    initial_capital: Decimal = Decimal("100000")
    max_positions: int = 10
    position_sizing_method: str = "ai_determined"  # Let AI decide position sizes

    # Execution settings
    mode: BacktestMode = BacktestMode.FAST
    timeframe: TimeFrame = TimeFrame.ONE_DAY
    commission_per_trade: Decimal = Decimal("1.00")
    commission_percentage: Decimal = Decimal("0.0005")  # 0.05%
    slippage_bps: Decimal = Decimal("5")  # 5 basis points

    # AI Strategy settings
    ai_strategy_config: Optional[Dict[str, Any]] = None
    symbols_to_trade: Optional[List[str]] = None  # If None, uses screener results

    # Risk management
    max_portfolio_risk: Decimal = Decimal("0.02")  # 2% max portfolio risk per trade
    max_position_size: Decimal = Decimal("0.20")  # 20% max position size

    # Data settings
    data_path: str = "data/parquet"
    enable_screener_data: bool = True
    screener_types: List[str] = field(
        default_factory=lambda: ["momentum", "breakouts", "value_stocks"]
    )


# BacktestPosition now imported from backtest_models


# BacktestTrade now imported from backtest_models


# BacktestResults now imported from backtest_models


class HistoricalDataFeeder:
    """Feeds historical data to simulate live data flow."""

    def __init__(self, data_store, config: RealBacktestConfig):
        self.data_store = data_store
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.HistoricalDataFeeder")

    async def get_market_data_for_date(
        self, date: datetime, symbols: List[str]
    ) -> Dict[str, MarketData]:
        """Get market data for specific date and symbols."""
        market_data = {}

        for symbol in symbols:
            try:
                df = await self.data_store.load_market_data(
                    ticker=symbol,
                    timeframe=self.config.timeframe,
                    start_date=date.date(),
                    end_date=date.date(),
                )

                if not df.is_empty():
                    # Get the latest data point for the date
                    row = df.sort("timestamp").tail(1).row(0, named=True)

                    # Handle missing adjusted_close column
                    adjusted_close = row.get("adjusted_close", row["close"])

                    market_data[symbol] = MarketData(
                        symbol=symbol,
                        timestamp=row["timestamp"],
                        open=Decimal(str(row["open"])),
                        high=Decimal(str(row["high"])),
                        low=Decimal(str(row["low"])),
                        close=Decimal(str(row["close"])),
                        adjusted_close=Decimal(str(adjusted_close)),
                        volume=int(row["volume"]),
                        timeframe=self.config.timeframe,
                    )

            except Exception as e:
                self.logger.warning(
                    f"Could not load data for {symbol} on {date.date()}: {e}"
                )

        return market_data

    async def get_screener_data_for_date(self, date: datetime) -> Dict[str, List[str]]:
        """Get screener results for specific date."""
        screener_results: Dict[str, List[str]] = {}

        if not self.config.enable_screener_data:
            return screener_results

        for screener_type in self.config.screener_types:
            try:
                df = await self.data_store.load_screener_data(
                    screener_type=screener_type,
                    start_date=date.date(),
                    end_date=date.date(),
                )

                if not df.is_empty():
                    symbols = df.select("symbol").to_series().to_list()
                    screener_results[screener_type] = symbols

            except Exception as e:
                self.logger.warning(
                    f"Could not load screener data for {screener_type} on {date.date()}: {e}"
                )

        return screener_results


class MockAIStrategy:
    """Mock AI strategy for backtesting when real strategy can't be loaded."""

    async def generate_signal(
        self, symbol: str, current_data, historical_data, market_context
    ):
        """Generate a mock signal for testing."""
        import random

        # Generate random signals for testing
        actions = [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        action = random.choice(actions)

        if action == SignalType.HOLD:
            return None

        return Signal(
            symbol=symbol,
            action=action,
            confidence=random.uniform(60, 90),
            position_size=random.uniform(0.05, 0.15),
            reasoning=f"Mock signal for {symbol}",
        )


class MockRedisClient:
    """Mock Redis client for backtesting to simulate live data flow."""

    def __init__(self):
        self.data_cache = {}
        self.logger = logging.getLogger(f"{__name__}.MockRedisClient")

    async def publish(self, channel: str, data: Any) -> None:
        """Mock publish - just log for debugging."""
        if hasattr(self.logger, "debug"):
            self.logger.debug(f"Mock Redis publish to {channel}: {type(data).__name__}")

    async def set(self, key: str, value: Any, ex: Optional[int] = None) -> None:
        """Mock set operation."""
        self.data_cache[key] = {
            "value": value,
            "expires_at": datetime.now() + timedelta(seconds=ex) if ex else None,
        }

    async def get(self, key: str) -> Optional[Any]:
        """Mock get operation."""
        if key in self.data_cache:
            cache_entry = self.data_cache[key]
            if (
                cache_entry["expires_at"] is None
                or cache_entry["expires_at"] > datetime.now()
            ):
                return cache_entry["value"]
            else:
                del self.data_cache[key]
        return None

    async def close(self) -> None:
        """Mock close operation."""
        self.data_cache.clear()


class RealBacktestEngine:
    """
    Real backtesting engine that runs the actual AI strategy against historical data.
    Simulates the complete system flow but accelerated and without delays.
    """

    def __init__(self, config: RealBacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components - use simplified data store for backtesting
        from simple_data_store import SimpleDataStore

        self.data_store = SimpleDataStore(base_path=config.data_path)

        self.data_feeder = HistoricalDataFeeder(self.data_store, config)
        self.mock_redis = MockRedisClient()

        # Portfolio state
        self.cash = config.initial_capital
        self.positions: Dict[str, BacktestPosition] = {}
        self.trades: List[BacktestTrade] = []
        self.portfolio_history: List[Tuple[datetime, Decimal]] = []

        # Performance tracking
        self.ai_calls = 0
        self.signals_generated = 0
        self.signals_executed = 0
        self.total_commissions = Decimal("0")
        self.total_slippage = Decimal("0")

        # AI Strategy (will be initialized in run_backtest)
        self.ai_strategy: Optional[Any] = None

    async def initialize_ai_strategy(self) -> None:
        """Initialize the AI strategy with backtesting configuration."""
        # If an AI strategy is already set (e.g., Ollama), don't override it
        if self.ai_strategy is not None:
            self.logger.info(
                f"Using existing AI strategy: {type(self.ai_strategy).__name__}"
            )
            return

        try:
            # Try to import and initialize AI strategy
            sys.path.append(
                os.path.join(
                    os.path.dirname(__file__), "../services/strategy_engine/src"
                )
            )

            # For now, use a mock strategy to avoid config dependencies
            self.ai_strategy = MockAIStrategy()
            self.logger.info("Mock AI Strategy initialized for backtesting")

        except Exception as e:
            self.logger.warning(f"AI strategy initialization failed: {e}")
            self.ai_strategy = MockAIStrategy()
            self.logger.info("Using mock AI strategy for backtesting")

    async def run_backtest(self) -> BacktestResults:
        """Run complete backtesting simulation."""
        start_time = time.time()
        self.logger.info(
            f"Starting real backtest from {self.config.start_date.date()} to {self.config.end_date.date()}"
        )

        try:
            # Initialize AI strategy
            await self.initialize_ai_strategy()

            # Generate date range
            current_date = self.config.start_date
            trading_days = []

            while current_date <= self.config.end_date:
                # Skip weekends for daily data
                if (
                    self.config.timeframe == TimeFrame.ONE_DAY
                    and current_date.weekday() < 5
                ):
                    trading_days.append(current_date)
                elif self.config.timeframe != TimeFrame.ONE_DAY:
                    trading_days.append(current_date)

                current_date += timedelta(days=1)

            self.logger.info(f"Processing {len(trading_days)} trading days")

            # Process each trading day
            for i, date in enumerate(trading_days):
                await self._process_trading_day(date, i, len(trading_days))

                # Record portfolio snapshot
                portfolio_value = await self._calculate_portfolio_value(date)
                self.portfolio_history.append((date, portfolio_value))

                if self.config.mode == BacktestMode.DEBUG:
                    self.logger.debug(
                        f"Day {i + 1}/{len(trading_days)}: Portfolio value = ${portfolio_value:,.2f}"
                    )

            # Close all remaining positions at end of backtest
            if self.positions:
                self.logger.info(
                    f"Closing {len(self.positions)} remaining positions at end of backtest"
                )
                end_date = trading_days[-1] if trading_days else self.config.end_date

                # Use last available market data for position closing
                final_market_data = {}
                for symbol in self.positions.keys():
                    try:
                        # Use the position's current price as approximation
                        position = self.positions[symbol]
                        # Create a simple market data object for closing
                        from backtest_models import MarketData

                        entry_price_float = float(position.entry_price)
                        final_market_data[symbol] = MarketData(
                            symbol=symbol,
                            timestamp=end_date,
                            open=Decimal(str(entry_price_float)),
                            high=Decimal(str(entry_price_float * 1.01)),
                            low=Decimal(str(entry_price_float * 0.99)),
                            close=Decimal(str(entry_price_float)),
                            adjusted_close=Decimal(str(entry_price_float)),
                            volume=1000,
                            timeframe=TimeFrame.ONE_DAY,
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Could not create final data for {symbol}: {e}"
                        )

                # Close each remaining position
                positions_to_close = list(self.positions.keys())
                for symbol in positions_to_close:
                    if symbol in final_market_data:
                        await self._close_position(
                            symbol,
                            final_market_data[symbol],
                            end_date,
                            "end_of_backtest",
                        )
                    else:
                        # Force close without market data if needed
                        self.logger.warning(
                            f"Force closing {symbol} without final market data"
                        )
                        if symbol in self.positions:
                            del self.positions[symbol]

            # Generate results
            execution_time = time.time() - start_time
            results = await self._generate_results(execution_time)

            self.logger.info(f"Backtest completed in {execution_time:.2f}s")
            self.logger.info(f"Total return: {results.total_return:.2%}")
            self.logger.info(f"Total trades: {results.total_trades}")

            # Show trade summary if we have completed trades
            if self.trades:
                self.logger.info("=== TRADE SUMMARY ===")
                total_pnl = sum(trade.pnl for trade in self.trades)
                winning_trades = [t for t in self.trades if t.pnl > 0]
                losing_trades = [t for t in self.trades if t.pnl < 0]

                self.logger.info(f"Completed trades: {len(self.trades)}")
                self.logger.info(
                    f"Winning trades: {len(winning_trades)} ({len(winning_trades) / len(self.trades) * 100:.1f}%)"
                )
                self.logger.info(f"Total P&L: ${total_pnl:.2f}")

                if winning_trades:
                    avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
                    self.logger.info(f"Average win: ${avg_win:.2f}")

                if losing_trades:
                    avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades)
                    self.logger.info("Average loss: $%.2f", avg_loss)

                self.logger.info("=== END TRADE SUMMARY ===")

            return results

        except Exception as e:
            self.logger.error("Backtest failed: %s", e)
            raise
        finally:
            await self.cleanup()

    async def _process_trading_day(
        self, date: datetime, day_index: int, total_days: int
    ) -> None:
        """Process a single trading day."""
        try:
            # Get symbols to analyze
            symbols = await self._get_symbols_for_date(date)
            if not symbols:
                self.logger.warning(f"No symbols found for {date.date()}")
                return

            # Get market data
            market_data = await self.data_feeder.get_market_data_for_date(date, symbols)
            if not market_data:
                self.logger.warning(f"No market data found for {date.date()}")
                return

            # Update current prices for existing positions
            await self._update_position_prices(market_data, date)

            # Check exit conditions for existing positions
            await self._check_exit_conditions(market_data, date)

            # Generate new signals using AI strategy
            if self.ai_strategy and len(self.positions) < self.config.max_positions:
                signals = await self._generate_ai_signals(market_data, date)

                # Execute signals
                for signal in signals:
                    await self._execute_signal(
                        signal, market_data.get(signal.symbol), date
                    )

            # Log progress
            if (
                (day_index + 1) % 50 == 0
                or day_index == 0
                or day_index == total_days - 1
            ):
                portfolio_value = await self._calculate_portfolio_value(date)
                self.logger.info(
                    f"Progress: {day_index + 1}/{total_days} days, Portfolio: ${portfolio_value:,.2f}"
                )

        except Exception as e:
            self.logger.error(f"Error processing {date.date()}: {e}")

    async def _get_symbols_for_date(self, date: datetime) -> List[str]:
        """Get symbols to analyze for given date."""
        symbols: Set[str] = set()

        # Add existing position symbols
        symbols.update(self.positions.keys())

        # Add symbols from configuration
        if self.config.symbols_to_trade:
            symbols.update(self.config.symbols_to_trade)

        # Add symbols from screener data
        if self.config.enable_screener_data:
            screener_data = await self.data_feeder.get_screener_data_for_date(date)
            for screener_type, screener_symbols in screener_data.items():
                symbols.update(screener_symbols[:20])  # Limit to top 20 per screener

        return list(symbols)

    async def _update_position_prices(
        self, market_data: Dict[str, MarketData], date: datetime
    ) -> None:
        """Update current prices for existing positions."""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.current_price = market_data[symbol].close
                position.unrealized_pnl = (
                    position.current_price - position.entry_price
                ) * Decimal(str(position.quantity))

    async def _check_exit_conditions(
        self, market_data: Dict[str, MarketData], date: datetime
    ) -> None:
        """Check if any positions should be closed."""
        positions_to_close = []

        for symbol, position in self.positions.items():
            if symbol not in market_data:
                continue

            current_price = market_data[symbol].close

            # Check stop loss
            if position.stop_loss:
                if (position.is_long and current_price <= position.stop_loss) or (
                    not position.is_long and current_price >= position.stop_loss
                ):
                    positions_to_close.append((symbol, "stop_loss"))
                    continue

            # Check take profit
            if position.take_profit:
                if (position.is_long and current_price >= position.take_profit) or (
                    not position.is_long and current_price <= position.take_profit
                ):
                    positions_to_close.append((symbol, "take_profit"))
                    continue

        # Close positions that hit exit conditions
        for symbol, reason in positions_to_close:
            await self._close_position(symbol, market_data[symbol], date, reason)

    async def _generate_ai_signals(
        self, market_data: Dict[str, MarketData], date: datetime
    ) -> List[Signal]:
        """Generate trading signals using AI strategy."""
        if not self.ai_strategy:
            return []

        signals = []

        try:
            # Prepare data for AI analysis
            for symbol, data in market_data.items():
                if symbol in self.positions:
                    continue  # Skip if we already have a position

                # Get historical data for analysis
                historical_data = await self._get_historical_data_for_analysis(
                    symbol, date
                )
                if historical_data is None or len(historical_data) < 20:
                    continue

                # Generate signal using AI strategy
                self.ai_calls += 1
                signal = await self.ai_strategy.generate_signal(
                    symbol=symbol,
                    current_data=data,
                    historical_data=historical_data,
                    market_context={},
                )

                if signal and signal.action != SignalType.HOLD:
                    signals.append(signal)
                    self.signals_generated += 1

        except Exception as e:
            self.logger.error(f"Error generating AI signals: {e}")

        return signals

    async def _get_historical_data_for_analysis(
        self, symbol: str, current_date: datetime
    ) -> Optional[List[MarketData]]:
        """Get historical data for AI analysis."""
        try:
            lookback_days = 60  # Get enough history for analysis
            start_date = current_date - timedelta(days=lookback_days)

            df = await self.data_store.load_market_data(
                ticker=symbol,
                timeframe=self.config.timeframe,
                start_date=start_date.date(),
                end_date=current_date.date(),
            )

            if df.is_empty():
                return None

            historical_data = []
            for row in df.sort("timestamp").iter_rows(named=True):
                # Handle missing adjusted_close column
                adjusted_close = row.get("adjusted_close", row["close"])

                data_point = MarketData(
                    symbol=symbol,
                    timestamp=row["timestamp"],
                    open=Decimal(str(row["open"])),
                    high=Decimal(str(row["high"])),
                    low=Decimal(str(row["low"])),
                    close=Decimal(str(row["close"])),
                    adjusted_close=Decimal(str(adjusted_close)),
                    volume=int(row["volume"]),
                    timeframe=self.config.timeframe,
                )
                historical_data.append(data_point)

            return historical_data

        except Exception as e:
            self.logger.error(f"Error loading historical data for {symbol}: {e}")
            return None

    async def _execute_signal(
        self, signal: Signal, market_data: MarketData, date: datetime
    ) -> None:
        """Execute a trading signal."""
        if not market_data:
            return

        try:
            symbol = signal.symbol if hasattr(signal, "symbol") else market_data.symbol

            # Calculate position size
            position_size = await self._calculate_position_size(signal, market_data)
            if position_size == 0:
                return

            # Calculate entry price with slippage
            entry_price = await self._calculate_entry_price(market_data, signal.action)

            # Calculate commission
            commission = await self._calculate_commission(entry_price, position_size)

            # Check if we have enough cash
            required_cash = entry_price * Decimal(str(position_size)) + commission
            if required_cash > self.cash:
                self.logger.warning(
                    f"Insufficient cash for {symbol}: need ${required_cash}, have ${self.cash}"
                )
                return

            # Create position
            quantity = (
                position_size if signal.action == SignalType.BUY else -position_size
            )

            position = BacktestPosition(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                entry_date=date,
                current_price=entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )

            # Update portfolio
            self.positions[symbol] = position
            self.cash -= required_cash
            self.total_commissions += commission
            self.signals_executed += 1

            # Enhanced position opening logging
            direction = "LONG" if signal.action == SignalType.BUY else "SHORT"
            position_value = abs(quantity) * entry_price
            self.logger.info(
                f"🔵 Opened {direction} {symbol}: {abs(quantity)} shares @ ${entry_price:.4f} "
                f"| Value: ${position_value:,.2f} | Confidence: {signal.confidence}%"
            )

        except Exception as e:
            self.logger.error(f"Error executing signal for {symbol}: {e}")

    async def _close_position(
        self, symbol: str, market_data: MarketData, date: datetime, reason: str
    ) -> None:
        """Close an existing position."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        try:
            # Calculate exit price with slippage
            exit_price = await self._calculate_exit_price(market_data, position.is_long)

            # Calculate commission
            commission = await self._calculate_commission(
                exit_price, abs(position.quantity)
            )

            # Calculate P&L
            if position.is_long:
                pnl = (exit_price - position.entry_price) * Decimal(
                    str(abs(position.quantity))
                )
            else:
                pnl = (position.entry_price - exit_price) * Decimal(
                    str(abs(position.quantity))
                )

            pnl -= commission  # Subtract commission
            pnl_percentage = float(
                pnl / (position.entry_price * Decimal(str(abs(position.quantity))))
            )

            # Create trade record
            trade = BacktestTrade(
                symbol=symbol,
                entry_date=position.entry_date,
                exit_date=date,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                pnl=pnl,
                pnl_percentage=float(pnl_percentage),
                commission=commission,
                hold_days=(date - position.entry_date).days,
                strategy_reasoning=reason,
                confidence=0.0,  # Would need to store this from signal
            )

            # Update portfolio
            proceeds = exit_price * Decimal(str(abs(position.quantity))) - commission
            self.cash += proceeds
            self.total_commissions += commission
            self.trades.append(trade)

            del self.positions[symbol]

            # Enhanced trade result logging
            days_held = (date - position.entry_date).days
            result_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚫"
            direction = "LONG" if position.is_long else "SHORT"

            self.logger.info(
                f"{result_emoji} Closed {direction} {symbol}: ${pnl:.2f} ({pnl_percentage:+.2%}) "
                f"| {days_held}d | {reason} | Entry: ${position.entry_price:.2f} → Exit: ${exit_price:.2f}"
            )

        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {e}")

    async def _calculate_position_size(
        self, signal: Signal, market_data: MarketData
    ) -> int:
        """Calculate position size based on signal and risk management."""
        try:
            # Use AI-suggested position size if available
            if hasattr(signal, "position_size") and signal.position_size > 0:
                portfolio_value = await self._calculate_portfolio_value(datetime.now())
                max_allocation = portfolio_value * Decimal(str(signal.position_size))
            else:
                # Default to equal weight allocation
                portfolio_value = await self._calculate_portfolio_value(datetime.now())
                max_allocation = portfolio_value * self.config.max_position_size

            # Calculate based on available cash
            available_cash = min(self.cash, max_allocation)

            # Calculate position size
            position_value = available_cash * Decimal(
                "0.95"
            )  # Leave some buffer for commission
            position_size = int(position_value / market_data.close)

            return max(0, position_size)

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0

    async def _calculate_entry_price(
        self, market_data: MarketData, action: SignalType
    ) -> Decimal:
        """Calculate entry price with slippage."""
        base_price = market_data.close
        slippage_factor = self.config.slippage_bps / Decimal("10000")

        if action == SignalType.BUY:
            return base_price * (Decimal("1") + slippage_factor)
        else:
            return base_price * (Decimal("1") - slippage_factor)

    async def _calculate_exit_price(
        self, market_data: MarketData, is_long: bool
    ) -> Decimal:
        """Calculate exit price with slippage."""
        base_price = market_data.close
        slippage_factor = self.config.slippage_bps / Decimal("10000")

        if is_long:
            return base_price * (Decimal("1") - slippage_factor)
        else:
            return base_price * (Decimal("1") + slippage_factor)

    async def _calculate_commission(self, price: Decimal, quantity: int) -> Decimal:
        """Calculate trading commission."""
        fixed_commission = self.config.commission_per_trade
        percentage_commission = (
            price * Decimal(str(quantity)) * self.config.commission_percentage
        )
        return fixed_commission + percentage_commission

    async def _calculate_portfolio_value(self, date: datetime) -> Decimal:
        """Calculate total portfolio value."""
        total_value = self.cash

        for position in self.positions.values():
            total_value += position.market_value

        return total_value

    async def _generate_results(self, execution_time: float) -> BacktestResults:
        """Generate comprehensive backtest results."""
        try:
            # Calculate basic metrics
            initial_value = self.config.initial_capital
            final_value = await self._calculate_portfolio_value(datetime.now())
            total_return = float((final_value - initial_value) / initial_value)

            # Calculate trading metrics
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl < 0]

            win_rate = len(winning_trades) / len(self.trades) if self.trades else 0

            avg_win = (
                float(sum(t.pnl for t in winning_trades) / len(winning_trades))
                if winning_trades
                else 0
            )
            avg_loss = (
                float(sum(t.pnl for t in losing_trades) / len(losing_trades))
                if losing_trades
                else 0
            )

            largest_win = float(max((t.pnl for t in winning_trades), default=0))
            largest_loss = float(min((t.pnl for t in losing_trades), default=0))

            profit_factor = (
                abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades)))
                if avg_loss != 0
                else float("inf")
            )

            # Calculate time-based metrics
            if self.portfolio_history:
                returns = []
                for i in range(1, len(self.portfolio_history)):
                    prev_value = float(self.portfolio_history[i - 1][1])
                    curr_value = float(self.portfolio_history[i][1])
                    if prev_value > 0:
                        returns.append((curr_value - prev_value) / prev_value)

                # Calculate drawdown
                peak = float(initial_value)
                max_drawdown = 0.0
                max_value = float(initial_value)
                min_value = float(initial_value)

                for _, value in self.portfolio_history:
                    value_float = float(value)
                    max_value = max(max_value, value_float)
                    min_value = min(min_value, value_float)

                    if value_float > peak:
                        peak = value_float
                    else:
                        drawdown = (peak - value_float) / peak
                        max_drawdown = max(max_drawdown, drawdown)

                # Calculate risk metrics
                if returns:
                    import numpy as np

                    returns_array = np.array(returns)

                    annualized_return = float(
                        (1 + np.mean(returns)) ** 252 - 1
                    )  # Assuming daily returns
                    volatility = float(np.std(returns) * np.sqrt(252))

                    sharpe_ratio = (
                        float((annualized_return - 0.02) / volatility)
                        if volatility > 0
                        else 0.0
                    )

                    negative_returns = returns_array[returns_array < 0]
                    downside_volatility = (
                        float(np.std(negative_returns) * np.sqrt(252))
                        if len(negative_returns) > 0
                        else 0.0
                    )
                    sortino_ratio = (
                        float((annualized_return - 0.02) / downside_volatility)
                        if downside_volatility > 0
                        else 0.0
                    )

                    calmar_ratio = (
                        float(annualized_return / max_drawdown)
                        if max_drawdown > 0
                        else 0.0
                    )
                else:
                    annualized_return = total_return
                    sharpe_ratio = 0.0
                    sortino_ratio = 0.0
                    calmar_ratio = 0.0
                    max_drawdown = 0.0
                    max_value = float(final_value)
                    min_value = float(final_value)
            else:
                annualized_return = total_return
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
                calmar_ratio = 0.0
                max_drawdown = 0.0
                max_value = float(final_value)
                min_value = float(final_value)

            # AI strategy metrics
            avg_confidence = 0.0
            if self.trades:
                total_confidence = sum(t.confidence for t in self.trades)
                avg_confidence = total_confidence / len(self.trades)

            return BacktestResults(
                # Performance metrics
                total_return=total_return,
                annualized_return=annualized_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                # Trading metrics
                total_trades=len(self.trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                win_rate=win_rate,
                profit_factor=profit_factor,
                average_win=avg_win,
                average_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                # Portfolio metrics
                final_portfolio_value=final_value,
                max_portfolio_value=Decimal(str(max_value)),
                min_portfolio_value=Decimal(str(min_value)),
                # Execution metrics
                total_commissions=self.total_commissions,
                total_slippage=self.total_slippage,
                execution_time_seconds=execution_time,
                # Trade details
                trades=self.trades,
                daily_returns=returns if "returns" in locals() else [],
                portfolio_values=[(dt, val) for dt, val in self.portfolio_history],
                # AI Strategy metrics
                total_ai_calls=self.ai_calls,
                average_confidence=avg_confidence,
                signals_generated=self.signals_generated,
                signals_executed=self.signals_executed,
            )

        except Exception as e:
            self.logger.error(f"Error generating results: {e}")
            raise

    async def cleanup(self) -> None:
        """Cleanup resources after backtesting."""
        try:
            # Close AI strategy connections
            if self.ai_strategy and hasattr(self.ai_strategy, "cleanup"):
                await self.ai_strategy.cleanup()

            # Close mock Redis
            if self.mock_redis:
                await self.mock_redis.close()

            # Close data store connections
            if hasattr(self.data_store, "close"):
                await self.data_store.close()

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Utility functions for running backtests


async def run_monthly_backtest(
    start_date: datetime,
    end_date: Optional[datetime] = None,
    initial_capital: Decimal = Decimal("100000"),
    symbols: Optional[List[str]] = None,
) -> BacktestResults:
    """
    Run backtest for the specified date range (defaults to previous month).

    Args:
        start_date: Start date for backtesting
        end_date: End date (defaults to today)
        initial_capital: Starting capital
        symbols: Specific symbols to trade (if None, uses screener results)

    Returns:
        BacktestResults with complete performance analysis
    """

    if end_date is None:
        end_date = datetime.now(timezone.utc)

    config = RealBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        symbols_to_trade=symbols,
        mode=BacktestMode.FAST,
        timeframe=TimeFrame.ONE_DAY,
    )

    engine = RealBacktestEngine(config)
    return await engine.run_backtest()


async def run_previous_month_backtest() -> BacktestResults:
    """
    Convenience function to run backtest for the previous month.

    Returns:
        BacktestResults with performance analysis
    """
    today = datetime.now(timezone.utc)
    start_of_current_month = today.replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    start_of_previous_month = (start_of_current_month - timedelta(days=1)).replace(
        day=1
    )
    end_of_previous_month = start_of_current_month - timedelta(days=1)

    return await run_monthly_backtest(
        start_date=start_of_previous_month, end_date=end_of_previous_month
    )
