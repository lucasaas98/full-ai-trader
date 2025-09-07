"""
Production Backtesting Engine

This engine uses the actual production trading strategies (HybridStrategy, etc.)
against historical market data. It simulates the complete trading pipeline
including screener alerts, strategy signals, and trade execution.

Key Features:
- Uses real production strategies (day trading, swing trading, position trading)
- Retroactively simulates screener alerts based on FinViz criteria
- Works with extensive historical parquet data (1-3+ months)
- Comprehensive performance analysis
- Multi-strategy comparison capabilities
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Add paths for strategy imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
strategy_path = os.path.join(project_root, "services", "strategy_engine", "src")
data_collector_path = os.path.join(project_root, "services", "data_collector", "src")
shared_path = os.path.join(project_root, "shared")

# Add all necessary paths
for path in [strategy_path, data_collector_path, shared_path, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

from backtest_models import MarketData, SignalType, TimeFrame  # noqa: E402
from simple_data_store import SimpleDataStore  # noqa: E402

try:
    # Use the production strategy adapter
    from production_strategy_adapter import (
        PRODUCTION_STRATEGIES_AVAILABLE,
        ProductionSignal,
        ProductionStrategyAdapter,
    )
    from production_strategy_adapter import StrategyMode as AdapterStrategyMode

    logging.info("Production strategy adapter loaded successfully")

except Exception as e:
    logging.warning(f"Production strategy adapter not available: {e}")
    PRODUCTION_STRATEGIES_AVAILABLE = False


logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Backtesting execution modes."""

    FAST = "fast"
    DETAILED = "detailed"
    DEBUG = "debug"


@dataclass
class ScreenerCriteria:
    """Criteria for simulating screener alerts."""

    # Breakouts criteria
    breakout_volume_ratio: float = 2.0
    breakout_price_min: float = 3.0
    breakout_price_max: float = 100.0
    breakout_market_cap_min: float = 50_000_000  # 50M
    breakout_avg_volume_min: int = 500_000
    breakout_above_sma20: bool = True
    breakout_weekly_volatility_min: float = 8.0

    # Momentum criteria
    momentum_change_min: float = 5.0  # 5% minimum change
    momentum_volume_ratio: float = 1.5
    momentum_price_min: float = 5.0
    momentum_market_cap_min: float = 100_000_000  # 100M

    # Value stocks criteria
    value_pe_max: float = 15.0
    value_price_min: float = 10.0
    value_market_cap_min: float = 1_000_000_000  # 1B
    value_dividend_yield_min: float = 2.0


@dataclass
class ProductionBacktestConfig:
    """Configuration for production backtesting."""

    # Time period
    start_date: datetime
    end_date: datetime

    # Strategy selection
    strategy_type: str = "day_trading"  # day_trading, swing_trading, position_trading
    custom_strategy_config: Optional[Dict[str, Any]] = None

    # Portfolio settings
    initial_capital: Decimal = Decimal("100000")
    max_positions: int = 10
    max_position_size: Decimal = Decimal("0.15")  # 15% max per position

    # Execution settings
    mode: BacktestMode = BacktestMode.FAST
    timeframe: TimeFrame = TimeFrame.ONE_DAY
    commission_per_trade: Decimal = Decimal("1.00")
    commission_percentage: Decimal = Decimal("0.0005")  # 0.05%
    slippage_bps: Decimal = Decimal("5")  # 5 basis points

    # Data settings
    data_path: str = "data/parquet"

    # Screener settings
    enable_screener_simulation: bool = True
    screener_criteria: ScreenerCriteria = field(default_factory=ScreenerCriteria)
    screener_types: List[str] = field(
        default_factory=lambda: ["breakouts", "momentum", "value_stocks"]
    )
    max_screener_symbols_per_day: int = 50

    # Symbol filtering
    specific_symbols: Optional[List[str]] = None  # If None, uses screener results
    exclude_symbols: List[str] = field(default_factory=list)
    min_market_cap: float = 50_000_000  # 50M minimum
    min_avg_volume: int = 100_000  # 100K minimum daily volume


@dataclass
class ProductionPosition:
    """Active position during backtesting."""

    symbol: str
    quantity: int
    entry_price: Decimal
    entry_date: datetime
    entry_reason: str
    strategy_name: str
    current_price: Decimal = Decimal("0")
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trailing_stop: Optional[Decimal] = None
    max_price_seen: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    @property
    def market_value(self) -> Decimal:
        return self.current_price * Decimal(str(abs(self.quantity)))

    @property
    def is_long(self) -> bool:
        return self.quantity > 0


@dataclass
class ProductionTrade:
    """Completed trade record."""

    symbol: str
    strategy_name: str
    entry_date: datetime
    exit_date: datetime
    entry_price: Decimal
    exit_price: Decimal
    quantity: int
    gross_pnl: Decimal
    net_pnl: Decimal
    pnl_percentage: float
    commission_total: Decimal
    hold_duration_hours: float
    entry_reason: str
    exit_reason: str
    max_favorable_excursion: Decimal
    max_adverse_excursion: Decimal


@dataclass
class ProductionBacktestResults:
    """Comprehensive production backtesting results."""

    # Configuration
    config: ProductionBacktestConfig
    strategy_name: str

    # Time and execution
    start_time: datetime
    end_time: datetime
    execution_time_seconds: float

    # Portfolio metrics
    initial_capital: Decimal
    final_capital: Decimal
    max_capital: Decimal
    min_capital: Decimal

    # Performance metrics
    total_return: float
    annualized_return: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float

    # Trading statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_amount: Decimal
    avg_loss_amount: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    avg_hold_time_hours: float

    # Strategy-specific metrics
    total_signals_generated: int
    signals_executed: int
    signal_execution_rate: float
    avg_signal_confidence: float

    # Screener statistics
    screener_alerts_simulated: int
    symbols_screened: int
    unique_symbols_traded: int

    # Risk metrics
    var_95: Decimal  # Value at Risk 95%
    max_concurrent_positions: int
    avg_position_size: Decimal
    largest_position_size: Decimal

    # Detailed records
    trades: List[ProductionTrade]
    daily_portfolio_values: List[Tuple[datetime, Decimal]]
    daily_returns: List[float]
    monthly_returns: Dict[str, float]

    # Strategy analysis
    strategy_performance_breakdown: Dict[str, Any]


class HistoricalScreenerSimulator:
    """Simulates screener alerts based on historical data and FinViz criteria."""

    def __init__(self, data_store: SimpleDataStore, criteria: ScreenerCriteria):
        self.data_store = data_store
        self.criteria = criteria
        self.logger = logging.getLogger(f"{__name__}.ScreenerSimulator")

    async def simulate_screener_for_date(
        self, date: datetime, symbols: List[str]
    ) -> Dict[str, List[str]]:
        """Simulate screener results for a specific date."""
        screener_results: Dict[str, List[str]] = {
            "breakouts": [],
            "momentum": [],
            "value_stocks": [],
        }

        # Load market data for all symbols for this date and previous days for analysis
        symbol_data = await self._load_symbol_data_for_screening(date, symbols)

        for symbol, data_history in symbol_data.items():
            if not data_history or len(data_history) < 20:  # Need enough history
                continue

            current_data = data_history[-1]  # Most recent data point

            # Check breakout criteria
            if await self._matches_breakout_criteria(
                symbol, current_data, data_history
            ):
                screener_results["breakouts"].append(symbol)

            # Check momentum criteria
            if await self._matches_momentum_criteria(
                symbol, current_data, data_history
            ):
                screener_results["momentum"].append(symbol)

            # Check value criteria
            if await self._matches_value_criteria(symbol, current_data, data_history):
                screener_results["value_stocks"].append(symbol)

        # Limit results per screener type
        for screener_type in screener_results:
            if len(screener_results[screener_type]) > 20:
                screener_results[screener_type] = screener_results[screener_type][:20]

        return screener_results

    async def _load_symbol_data_for_screening(
        self, date: datetime, symbols: List[str]
    ) -> Dict[str, List[MarketData]]:
        """Load historical data needed for screening analysis."""
        symbol_data = {}

        # Get 30 days of history for analysis
        end_date = date.date()
        start_date = (date - timedelta(days=30)).date()

        for symbol in symbols:
            try:
                df = await self.data_store.load_market_data(
                    ticker=symbol,
                    timeframe=TimeFrame.ONE_DAY,
                    start_date=start_date,
                    end_date=end_date,
                )

                if not df.is_empty():
                    market_data_list = []
                    for row in df.sort("timestamp").iter_rows(named=True):
                        adjusted_close = row.get("adjusted_close", row["close"])

                        market_data_list.append(
                            MarketData(
                                symbol=symbol,
                                timestamp=row["timestamp"],
                                open=Decimal(str(row["open"])),
                                high=Decimal(str(row["high"])),
                                low=Decimal(str(row["low"])),
                                close=Decimal(str(row["close"])),
                                adjusted_close=Decimal(str(adjusted_close)),
                                volume=int(row["volume"]),
                                timeframe=TimeFrame.ONE_DAY,
                            )
                        )

                    if market_data_list:
                        symbol_data[symbol] = market_data_list

            except Exception as e:
                self.logger.debug(f"Could not load screening data for {symbol}: {e}")

        return symbol_data

    async def _matches_breakout_criteria(
        self, symbol: str, current: MarketData, history: List[MarketData]
    ) -> bool:
        """Check if symbol matches breakout screener criteria."""
        try:
            if len(history) < 20:
                return False

            # Price range check
            current_price = float(current.close)
            if (
                current_price < self.criteria.breakout_price_min
                or current_price > self.criteria.breakout_price_max
            ):
                return False

            # Volume analysis
            recent_volumes = [float(d.volume) for d in history[-10:]]
            avg_volume = sum(recent_volumes) / len(recent_volumes)

            if avg_volume < self.criteria.breakout_avg_volume_min:
                return False

            current_volume = float(current.volume)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

            if volume_ratio < self.criteria.breakout_volume_ratio:
                return False

            # SMA20 check
            if self.criteria.breakout_above_sma20:
                recent_closes = [float(d.close) for d in history[-20:]]
                sma_20 = sum(recent_closes) / len(recent_closes)
                if current_price < sma_20:
                    return False

            # Weekly volatility check (approximate using daily data)
            recent_highs = [float(d.high) for d in history[-5:]]  # 5 days ~ 1 week
            recent_lows = [float(d.low) for d in history[-5:]]

            if recent_highs and recent_lows:
                weekly_range = (max(recent_highs) - min(recent_lows)) / min(recent_lows)
                if weekly_range * 100 < self.criteria.breakout_weekly_volatility_min:
                    return False

            return True

        except Exception as e:
            self.logger.debug(f"Error checking breakout criteria for {symbol}: {e}")
            return False

    async def _matches_momentum_criteria(
        self, symbol: str, current: MarketData, history: List[MarketData]
    ) -> bool:
        """Check if symbol matches momentum screener criteria."""
        try:
            if len(history) < 5:
                return False

            current_price = float(current.close)

            # Price minimum
            if current_price < self.criteria.momentum_price_min:
                return False

            # Calculate recent price change (5 days)
            old_price = float(history[-5].close)
            price_change_pct = ((current_price - old_price) / old_price) * 100

            if price_change_pct < self.criteria.momentum_change_min:
                return False

            # Volume ratio check
            recent_volumes = [float(d.volume) for d in history[-10:]]
            avg_volume = sum(recent_volumes) / len(recent_volumes)

            current_volume = float(current.volume)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

            if volume_ratio < self.criteria.momentum_volume_ratio:
                return False

            return True

        except Exception as e:
            self.logger.debug(f"Error checking momentum criteria for {symbol}: {e}")
            return False

    async def _matches_value_criteria(
        self, symbol: str, current: MarketData, history: List[MarketData]
    ) -> bool:
        """Check if symbol matches value screener criteria."""
        try:
            current_price = float(current.close)

            # Basic price filter
            if current_price < self.criteria.value_price_min:
                return False

            # For value screening, we'd need fundamental data (P/E, dividend yield, etc.)
            # Since we're working with price data only, we'll use price-based value indicators

            # Check if trading below recent highs (value opportunity)
            if len(history) >= 52:  # ~1 year of data
                history_subset = history[-252:] if len(history) >= 252 else history
                year_high = max(float(d.high) for d in history_subset)
                discount_from_high = (year_high - current_price) / year_high

                # Must be trading at significant discount from 52-week high
                if discount_from_high < 0.2:  # 20% discount
                    return False

            # Stable/declining volatility (value characteristic)
            if len(history) >= 20:
                recent_volatility = self._calculate_volatility(history[-20:])
                older_volatility = (
                    self._calculate_volatility(history[-40:-20])
                    if len(history) >= 40
                    else recent_volatility
                )

                # Prefer stocks with declining volatility
                if recent_volatility > older_volatility * 1.5:
                    return False

            return True

        except Exception as e:
            self.logger.debug(f"Error checking value criteria for {symbol}: {e}")
            return False

    def _calculate_volatility(self, data: List[MarketData]) -> float:
        """Calculate price volatility for given data."""
        if len(data) < 2:
            return 0.0

        prices = [float(d.close) for d in data]
        returns = []

        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i - 1]) / prices[i - 1]
            returns.append(ret)

        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)

        return variance**0.5


class ProductionBacktestEngine:
    """
    Production backtesting engine that uses real trading strategies
    against historical data with simulated screener alerts.
    """

    def __init__(self, config: ProductionBacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize data components
        self.data_store = SimpleDataStore(base_path=config.data_path)
        self.screener_simulator = HistoricalScreenerSimulator(
            self.data_store, config.screener_criteria
        )

        # Initialize strategy
        self.strategy: Optional[Any] = None

        # Portfolio state
        self.cash = config.initial_capital
        self.positions: Dict[str, ProductionPosition] = {}
        self.trades: List[ProductionTrade] = []
        self.portfolio_history: List[Tuple[datetime, Decimal]] = []

        # Performance tracking
        self.max_capital = config.initial_capital
        self.min_capital = config.initial_capital
        self.total_signals = 0
        self.executed_signals = 0
        self.screener_alerts = 0
        self.daily_returns: List[float] = []

        # Strategy statistics
        self.strategy_stats = {
            "total_analysis_calls": 0,
            "signals_generated": 0,
            "high_confidence_signals": 0,
            "avg_confidence": 0.0,
        }

    async def initialize_strategy(self) -> None:
        """Initialize the production trading strategy."""
        if not PRODUCTION_STRATEGIES_AVAILABLE:
            raise RuntimeError("Production strategies not available - check imports")

        try:
            # Map strategy types to adapter modes
            strategy_mode_map = {
                "day_trading": AdapterStrategyMode.DAY_TRADING,
                "swing_trading": AdapterStrategyMode.SWING_TRADING,
                "position_trading": AdapterStrategyMode.POSITION_TRADING,
            }

            if self.config.strategy_type not in strategy_mode_map:
                raise ValueError(f"Unknown strategy type: {self.config.strategy_type}")

            strategy_mode = strategy_mode_map[self.config.strategy_type]
            self.strategy = ProductionStrategyAdapter(strategy_mode)

            self.logger.info(
                f"Initialized {self.config.strategy_type} strategy: {self.strategy.name}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize strategy: {e}")
            raise

    async def run_backtest(self) -> ProductionBacktestResults:
        """Run the complete production backtesting simulation."""
        start_time = time.time()
        self.logger.info(
            f"Starting production backtest: {self.config.start_date.date()} to {self.config.end_date.date()}"
        )

        try:
            # Initialize strategy
            await self.initialize_strategy()

            # Get available symbols for backtesting
            available_symbols = await self._get_available_symbols()
            if not available_symbols:
                raise ValueError("No symbols available for backtesting")

            self.logger.info(f"Found {len(available_symbols)} symbols for backtesting")

            # Generate trading days
            trading_days = self._generate_trading_days()
            self.logger.info(f"Processing {len(trading_days)} trading days")

            # Process each trading day
            for i, date in enumerate(trading_days):
                await self._process_trading_day(date, available_symbols)

                # Update portfolio history
                portfolio_value = await self._calculate_portfolio_value()
                self.portfolio_history.append((date, portfolio_value))
                self._update_portfolio_statistics(portfolio_value)

                # Log progress periodically
                if (i + 1) % 20 == 0 or i == len(trading_days) - 1:
                    self.logger.info(
                        f"Progress: {i + 1}/{len(trading_days)} days, Portfolio: ${portfolio_value:,.2f}"
                    )

            # Close any remaining positions
            await self._close_all_positions(trading_days[-1], "end_of_backtest")

            # Generate comprehensive results
            execution_time = time.time() - start_time
            results = await self._generate_results(execution_time)

            self.logger.info(f"Backtest completed in {execution_time:.2f}s")
            self.logger.info(
                f"Total Return: {results.total_return:.2%}, Trades: {results.total_trades}"
            )

            return results

        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise

    async def _get_available_symbols(self) -> List[str]:
        """Get list of available symbols for backtesting."""
        if self.config.specific_symbols:
            return [
                s
                for s in self.config.specific_symbols
                if s not in self.config.exclude_symbols
            ]

        # Get all available symbols from data store
        all_symbols = self.data_store.get_available_symbols()

        # Filter symbols based on criteria
        filtered_symbols = []
        for symbol in all_symbols:
            if symbol in self.config.exclude_symbols:
                continue

            # Check if symbol has sufficient data in date range
            date_range = self.data_store.get_date_range_for_symbol(
                symbol, self.config.timeframe
            )
            if not date_range[0] or not date_range[1]:
                continue

            # Allow symbols if they have data that overlaps with our desired period
            # and extends at least 7 days into the backtest period
            min_required_end = self.config.start_date.date() + timedelta(days=7)

            if (
                date_range[0] > self.config.start_date.date()
                or date_range[1] < min_required_end
            ):
                continue

            filtered_symbols.append(symbol)

        return filtered_symbols

    def _generate_trading_days(self) -> List[datetime]:
        """Generate list of trading days in the backtest period."""
        trading_days = []
        current_date = self.config.start_date

        # Adjust end date to match available data
        adjusted_end_date = self.config.end_date

        # Check if we have any symbols and adjust end date based on data availability
        available_symbols = self.data_store.get_available_symbols()
        if available_symbols:
            # Find the latest available data date across all symbols
            latest_data_date = None
            for symbol in available_symbols[:10]:  # Check first 10 symbols
                date_range = self.data_store.get_date_range_for_symbol(
                    symbol, self.config.timeframe
                )
                if date_range[1]:
                    if latest_data_date is None or date_range[1] > latest_data_date:
                        latest_data_date = date_range[1]

            if latest_data_date:
                adjusted_end_date = min(
                    self.config.end_date,
                    datetime.combine(latest_data_date, datetime.min.time()).replace(
                        tzinfo=timezone.utc
                    ),
                )

        while current_date <= adjusted_end_date:
            # Skip weekends for daily timeframes
            if (
                self.config.timeframe == TimeFrame.ONE_DAY
                and current_date.weekday() < 5
            ):
                trading_days.append(current_date)
            elif self.config.timeframe != TimeFrame.ONE_DAY:
                trading_days.append(current_date)

            current_date += timedelta(days=1)

        return trading_days

    async def _process_trading_day(
        self, date: datetime, available_symbols: List[str]
    ) -> None:
        """Process a single trading day."""
        try:
            # 1. Update existing positions
            await self._update_position_prices(date, list(self.positions.keys()))

            # 2. Check exit conditions for existing positions
            await self._check_exit_conditions(date)

            # 3. Simulate screener alerts for the day
            screener_symbols = set()
            if self.config.enable_screener_simulation:
                screener_results = (
                    await self.screener_simulator.simulate_screener_for_date(
                        date, available_symbols
                    )
                )

                for screener_type, symbols in screener_results.items():
                    if screener_type in self.config.screener_types:
                        screener_symbols.update(symbols)
                        self.screener_alerts += len(symbols)

            # 4. Combine screener results with any specific symbols
            analysis_symbols = list(screener_symbols)[
                : self.config.max_screener_symbols_per_day
            ]

            # 5. Generate signals for potential trades
            if len(self.positions) < self.config.max_positions and analysis_symbols:
                await self._generate_and_execute_signals(date, analysis_symbols)

        except Exception as e:
            self.logger.error(f"Error processing trading day {date.date()}: {e}")

    async def _update_position_prices(self, date: datetime, symbols: List[str]) -> None:
        """Update current prices for existing positions."""
        if not symbols:
            return

        for symbol in symbols:
            if symbol not in self.positions:
                continue

            try:
                df = await self.data_store.load_market_data(
                    ticker=symbol,
                    timeframe=self.config.timeframe,
                    start_date=date.date(),
                    end_date=date.date(),
                )

                if not df.is_empty():
                    row = df.sort("timestamp").tail(1).row(0, named=True)
                    new_price = Decimal(str(row["close"]))

                    position = self.positions[symbol]
                    position.current_price = new_price
                    position.max_price_seen = max(position.max_price_seen, new_price)
                    position.unrealized_pnl = (
                        new_price - position.entry_price
                    ) * Decimal(str(position.quantity))

            except Exception as e:
                self.logger.debug(f"Could not update price for {symbol}: {e}")

    async def _check_exit_conditions(self, date: datetime) -> None:
        """Check if any positions should be closed."""
        positions_to_close = []

        for symbol, position in self.positions.items():
            # Stop loss check
            if position.stop_loss and position.current_price <= position.stop_loss:
                positions_to_close.append((symbol, "stop_loss"))
                continue

            # Take profit check
            if position.take_profit and position.current_price >= position.take_profit:
                positions_to_close.append((symbol, "take_profit"))
                continue

            # Trailing stop check
            if (
                position.trailing_stop
                and position.max_price_seen > position.entry_price
            ):
                trailing_stop_price = position.max_price_seen * (
                    Decimal("1") - position.trailing_stop
                )
                if position.current_price <= trailing_stop_price:
                    positions_to_close.append((symbol, "trailing_stop"))
                    continue

        # Close positions that hit exit conditions
        for symbol, reason in positions_to_close:
            await self._close_position(symbol, date, reason)

    async def _generate_and_execute_signals(
        self, date: datetime, symbols: List[str]
    ) -> None:
        """Generate trading signals using the production strategy."""
        for symbol in symbols:
            if symbol in self.positions:  # Skip if already holding
                continue

            try:
                # Load historical data for strategy analysis
                historical_data = await self._get_historical_data_for_symbol(
                    symbol, date
                )
                if not historical_data or len(historical_data) < 20:
                    continue

                current_data = historical_data[-1]

                # Generate signal using production strategy
                self.strategy_stats["total_analysis_calls"] += 1

                # Call strategy's analyze method
                if self.strategy is None:
                    continue

                analysis_result = await self.strategy.analyze_symbol(
                    symbol=symbol,
                    current_data=current_data,
                    historical_data=historical_data[-50:],  # Last 50 days
                    market_context=self._get_market_context(date),
                )

                if analysis_result and analysis_result.action != SignalType.HOLD:
                    self.strategy_stats["signals_generated"] += 1
                    self.total_signals += 1

                    # Track confidence statistics
                    if hasattr(analysis_result, "confidence"):
                        if analysis_result.confidence > 75:
                            self.strategy_stats["high_confidence_signals"] += 1
                        self.strategy_stats["avg_confidence"] = (
                            self.strategy_stats["avg_confidence"]
                            * (self.strategy_stats["signals_generated"] - 1)
                            + analysis_result.confidence
                        ) / self.strategy_stats["signals_generated"]

                    # Execute the signal if it meets our criteria
                    if await self._should_execute_signal(analysis_result, current_data):
                        await self._execute_signal(analysis_result, current_data, date)

            except Exception as e:
                self.logger.debug(f"Error analyzing {symbol}: {e}")

    async def _get_historical_data_for_symbol(
        self, symbol: str, current_date: datetime
    ) -> Optional[List[MarketData]]:
        """Get historical data for symbol analysis."""
        try:
            lookback_days = 60  # Get sufficient history
            start_date = (current_date - timedelta(days=lookback_days)).date()
            end_date = current_date.date()

            df = await self.data_store.load_market_data(
                ticker=symbol,
                timeframe=self.config.timeframe,
                start_date=start_date,
                end_date=end_date,
            )

            if df.is_empty():
                return None

            historical_data = []
            for row in df.sort("timestamp").iter_rows(named=True):
                adjusted_close = row.get("adjusted_close", row["close"])

                historical_data.append(
                    MarketData(
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
                )

            return historical_data

        except Exception as e:
            self.logger.debug(f"Error loading historical data for {symbol}: {e}")
            return None

    def _get_market_context(self, date: datetime) -> Dict[str, Any]:
        """Get market context for strategy analysis."""
        return {
            "current_date": date,
            "portfolio_value": float(
                self.cash + sum(pos.market_value for pos in self.positions.values())
            ),
            "positions_count": len(self.positions),
            "available_capital": float(self.cash),
            "max_position_size": float(self.config.max_position_size),
            "strategy_type": self.config.strategy_type,
        }

    async def _should_execute_signal(
        self, analysis_result: ProductionSignal, current_data: MarketData
    ) -> bool:
        """Determine if a signal should be executed based on risk management."""
        try:
            # Check if we have enough cash
            portfolio_value = await self._calculate_portfolio_value()
            max_position_value = portfolio_value * self.config.max_position_size

            if max_position_value > self.cash:
                return False

            # Check confidence threshold
            if analysis_result.confidence < 60.0:
                return False

            # Check price reasonableness
            if float(current_data.close) < 1.0 or float(current_data.close) > 1000.0:
                return False

            # Check volume
            if current_data.volume < 10000:  # Minimum liquidity
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in signal execution check: {e}")
            return False

    async def _execute_signal(
        self,
        analysis_result: ProductionSignal,
        current_data: MarketData,
        date: datetime,
    ) -> None:
        """Execute a trading signal."""
        try:
            symbol = current_data.symbol
            entry_price = current_data.close

            # Calculate position size using signal's recommended size
            portfolio_value = await self._calculate_portfolio_value()
            position_value = (
                portfolio_value
                * Decimal(str(analysis_result.position_size))
                * Decimal("0.95")
            )  # Leave buffer for commission

            quantity = int(position_value / entry_price)
            if quantity <= 0:
                return

            # Calculate commission
            commission = self.config.commission_per_trade + (
                entry_price * Decimal(str(quantity)) * self.config.commission_percentage
            )

            if commission > self.cash:
                return

            # Create position
            position = ProductionPosition(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                entry_date=date,
                entry_reason=f"strategy_signal_{analysis_result.action.value}",
                strategy_name=self.strategy.name if self.strategy else "unknown",
                current_price=entry_price,
                max_price_seen=entry_price,
            )

            # Set risk management levels from signal
            if analysis_result.stop_loss:
                position.stop_loss = analysis_result.stop_loss

            if analysis_result.take_profit:
                position.take_profit = analysis_result.take_profit

            # Update portfolio
            self.positions[symbol] = position
            self.cash -= entry_price * Decimal(str(quantity)) + commission
            self.executed_signals += 1

            self.logger.info(
                f"Opened {analysis_result.action.value} position: {symbol} x{quantity} @ ${entry_price}"
            )

        except Exception as e:
            self.logger.error(f"Error executing signal for {current_data.symbol}: {e}")

    async def _close_position(self, symbol: str, date: datetime, reason: str) -> None:
        """Close an existing position."""
        if symbol not in self.positions:
            return

        try:
            position = self.positions[symbol]
            exit_price = position.current_price

            # Calculate P&L
            gross_pnl = (exit_price - position.entry_price) * Decimal(
                str(position.quantity)
            )
            commission = self.config.commission_per_trade + (
                exit_price
                * Decimal(str(abs(position.quantity)))
                * self.config.commission_percentage
            )
            net_pnl = gross_pnl - commission

            # Calculate trade metrics
            hold_duration = (date - position.entry_date).total_seconds() / 3600  # hours
            pnl_percentage = float(
                net_pnl / (position.entry_price * Decimal(str(abs(position.quantity))))
            )

            # Create trade record
            trade = ProductionTrade(
                symbol=symbol,
                strategy_name=position.strategy_name,
                entry_date=position.entry_date,
                exit_date=date,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                pnl_percentage=pnl_percentage,
                commission_total=commission,
                hold_duration_hours=hold_duration,
                entry_reason=position.entry_reason,
                exit_reason=reason,
                max_favorable_excursion=position.max_price_seen - position.entry_price,
                max_adverse_excursion=Decimal(
                    "0"
                ),  # Would need to track this during position lifetime
            )

            # Update portfolio
            proceeds = exit_price * Decimal(str(abs(position.quantity))) - commission
            self.cash += proceeds
            self.trades.append(trade)

            del self.positions[symbol]

            self.logger.info(
                f"Closed position: {symbol}, P&L: ${net_pnl:.2f} ({pnl_percentage:.2%}), Reason: {reason}"
            )

        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {e}")

    async def _close_all_positions(self, date: datetime, reason: str) -> None:
        """Close all open positions."""
        symbols_to_close = list(self.positions.keys())
        for symbol in symbols_to_close:
            await self._close_position(symbol, date, reason)

    async def _calculate_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value."""
        total_value = self.cash
        for position in self.positions.values():
            total_value += position.market_value
        return total_value

    def _update_portfolio_statistics(self, portfolio_value: Decimal) -> None:
        """Update portfolio performance statistics."""
        self.max_capital = max(self.max_capital, portfolio_value)
        self.min_capital = min(self.min_capital, portfolio_value)

        # Calculate daily return if we have previous day data
        if len(self.portfolio_history) > 0:
            previous_value = self.portfolio_history[-1][1]
            if previous_value > 0:
                daily_return = float(
                    (portfolio_value - previous_value) / previous_value
                )
                self.daily_returns.append(daily_return)

    async def _generate_results(
        self, execution_time: float
    ) -> ProductionBacktestResults:
        """Generate comprehensive backtest results."""
        try:
            final_capital = await self._calculate_portfolio_value()

            # Calculate performance metrics
            total_return = float(
                (final_capital - self.config.initial_capital)
                / self.config.initial_capital
            )

            # Calculate risk metrics
            returns_array = self.daily_returns
            if returns_array:
                import numpy as np

                returns_np = np.array(returns_array)

                # Annualized metrics
                trading_days = len(returns_array)
                annualized_return = (
                    float((1 + np.mean(returns_np)) ** 252 - 1)
                    if trading_days > 0
                    else 0.0
                )
                volatility = float(np.std(returns_np) * np.sqrt(252))

                # Sharpe ratio
                sharpe_ratio = (
                    float((annualized_return - 0.02) / volatility)
                    if volatility > 0
                    else 0.0
                )

                # Sortino ratio
                negative_returns = returns_np[returns_np < 0]
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
            else:
                annualized_return = total_return
                sharpe_ratio = 0.0
                sortino_ratio = 0.0

            # Calculate drawdown
            max_drawdown = 0.0
            peak = float(self.config.initial_capital)
            for _, value in self.portfolio_history:
                value_float = float(value)
                if value_float > peak:
                    peak = value_float
                else:
                    drawdown = (peak - value_float) / peak
                    max_drawdown = max(max_drawdown, drawdown)

            current_drawdown = (
                (float(self.max_capital) - float(final_capital))
                / float(self.max_capital)
                if self.max_capital > 0
                else 0.0
            )
            calmar_ratio = (
                float(annualized_return / max_drawdown) if max_drawdown > 0 else 0.0
            )

            # Trading statistics
            winning_trades = [t for t in self.trades if t.net_pnl > 0]
            losing_trades = [t for t in self.trades if t.net_pnl <= 0]

            win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0
            avg_win = (
                sum(t.net_pnl for t in winning_trades) / len(winning_trades)
                if winning_trades
                else Decimal("0")
            )
            avg_loss = (
                sum(t.net_pnl for t in losing_trades) / len(losing_trades)
                if losing_trades
                else Decimal("0")
            )

            gross_profit = (
                sum(t.net_pnl for t in winning_trades)
                if winning_trades
                else Decimal("0")
            )
            gross_loss = (
                Decimal(str(abs(sum(t.net_pnl for t in losing_trades))))
                if losing_trades
                else Decimal("0")
            )
            profit_factor = (
                float(str(gross_profit)) / float(str(gross_loss))
                if gross_loss > Decimal("0")
                else float("inf")
            )

            largest_win = max((t.net_pnl for t in winning_trades), default=Decimal("0"))
            largest_loss = min((t.net_pnl for t in losing_trades), default=Decimal("0"))

            # Monthly returns
            monthly_returns = {}
            for date, value in self.portfolio_history:
                month_key = date.strftime("%Y-%m")
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = float(value)

            # Strategy performance breakdown
            strategy_breakdown = {
                "analysis_calls": self.strategy_stats["total_analysis_calls"],
                "signals_generated": self.strategy_stats["signals_generated"],
                "high_confidence_signals": self.strategy_stats[
                    "high_confidence_signals"
                ],
                "avg_confidence": self.strategy_stats["avg_confidence"],
                "signal_to_trade_ratio": len(self.trades)
                / max(self.strategy_stats["signals_generated"], 1),
            }

            return ProductionBacktestResults(
                config=self.config,
                strategy_name=self.strategy.name if self.strategy else "unknown",
                start_time=self.config.start_date,
                end_time=self.config.end_date,
                execution_time_seconds=execution_time,
                initial_capital=self.config.initial_capital,
                final_capital=final_capital,
                max_capital=self.max_capital,
                min_capital=self.min_capital,
                total_return=total_return,
                annualized_return=annualized_return,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                profit_factor=profit_factor,
                total_trades=len(self.trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                win_rate=win_rate,
                avg_win_amount=Decimal(str(avg_win)),
                avg_loss_amount=Decimal(str(avg_loss)),
                largest_win=largest_win,
                largest_loss=largest_loss,
                avg_hold_time_hours=(
                    sum(t.hold_duration_hours for t in self.trades) / len(self.trades)
                    if self.trades
                    else 0.0
                ),
                total_signals_generated=self.total_signals,
                signals_executed=self.executed_signals,
                signal_execution_rate=self.executed_signals
                / max(self.total_signals, 1),
                avg_signal_confidence=self.strategy_stats["avg_confidence"],
                screener_alerts_simulated=self.screener_alerts,
                symbols_screened=len(set(t.symbol for t in self.trades)),
                unique_symbols_traded=len(set(t.symbol for t in self.trades)),
                var_95=Decimal("0"),  # Would need more sophisticated calculation
                max_concurrent_positions=max(len(self.positions), 1),
                avg_position_size=(
                    Decimal(
                        str(
                            sum(
                                abs(t.entry_price * Decimal(str(abs(t.quantity))))
                                for t in self.trades
                            )
                            / len(self.trades)
                        )
                    )
                    if self.trades
                    else Decimal("0")
                ),
                largest_position_size=max(
                    (
                        abs(t.entry_price * Decimal(str(abs(t.quantity))))
                        for t in self.trades
                    ),
                    default=Decimal("0"),
                ),
                trades=self.trades,
                daily_portfolio_values=self.portfolio_history,
                daily_returns=self.daily_returns,
                monthly_returns=monthly_returns,
                strategy_performance_breakdown=strategy_breakdown,
            )

        except Exception as e:
            self.logger.error(f"Error generating results: {e}")
            raise


# Utility functions for easy backtesting


async def run_production_backtest(
    start_date: datetime,
    end_date: datetime,
    strategy_type: str = "day_trading",
    initial_capital: Decimal = Decimal("100000"),
    symbols: Optional[List[str]] = None,
) -> ProductionBacktestResults:
    """
    Run a production backtest with default settings.

    Args:
        start_date: Start date for backtesting
        end_date: End date for backtesting
        strategy_type: Strategy to use (day_trading, swing_trading, position_trading)
        initial_capital: Starting capital
        symbols: Specific symbols to trade (None = use screener simulation)

    Returns:
        ProductionBacktestResults with comprehensive analysis
    """
    config = ProductionBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        strategy_type=strategy_type,
        initial_capital=initial_capital,
        specific_symbols=symbols,
        enable_screener_simulation=symbols is None,
    )

    engine = ProductionBacktestEngine(config)
    return await engine.run_backtest()


async def run_multi_strategy_comparison(
    start_date: datetime,
    end_date: datetime,
    initial_capital: Decimal = Decimal("100000"),
) -> Dict[str, ProductionBacktestResults]:
    """
    Compare all three production strategies over the same period.

    Returns:
        Dict mapping strategy names to their backtest results
    """
    strategies = ["day_trading", "swing_trading", "position_trading"]
    results = {}

    for strategy_type in strategies:
        try:
            result = await run_production_backtest(
                start_date=start_date,
                end_date=end_date,
                strategy_type=strategy_type,
                initial_capital=initial_capital,
            )
            results[strategy_type] = result
        except Exception as e:
            logging.error(f"Failed to run {strategy_type} backtest: {e}")

    return results
