import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.data_collector.src.data_store import DataStore  # noqa: E402
from shared.models import (  # noqa: E402
    BacktestResult,
    MarketData,
    OrderSide,
    SignalType,
    TimeFrame,
    TradeSignal,
)


# Define DataSource interface
class DataSource(ABC):
    """Abstract data source interface for backtesting"""

    @abstractmethod
    async def load_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame,
    ) -> Dict[str, List[MarketData]]:
        """Load market data for given symbols and date range"""
        pass


class LocalDataStoreSource(DataSource):
    """DataSource implementation using the existing DataStore infrastructure"""

    def __init__(self, data_store: DataStore):
        self.data_store = data_store
        self.logger = logging.getLogger(__name__)

    async def load_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame,
    ) -> Dict[str, List[MarketData]]:
        """Load market data from DataStore"""
        result = {}

        start_date_obj = start_date.date()
        end_date_obj = end_date.date()

        for symbol in symbols:
            try:
                # Load data using DataStore
                df = await self.data_store.load_market_data(
                    ticker=symbol,
                    timeframe=timeframe,
                    start_date=start_date_obj,
                    end_date=end_date_obj,
                )

                if df.is_empty():
                    self.logger.warning(f"No data found for symbol {symbol}")
                    continue

                # Convert Polars DataFrame to MarketData objects
                market_data = []
                for row in df.iter_rows(named=True):
                    data_point = MarketData(
                        symbol=symbol,
                        timestamp=row["timestamp"],
                        open=Decimal(str(row["open"])),
                        high=Decimal(str(row["high"])),
                        low=Decimal(str(row["low"])),
                        close=Decimal(str(row["close"])),
                        adjusted_close=Decimal(str(row["adjusted_close"])),
                        volume=int(row["volume"]),
                        timeframe=timeframe,
                    )
                    market_data.append(data_point)

                result[symbol] = market_data
                self.logger.info(f"Loaded {len(market_data)} data points for {symbol}")

            except Exception as e:
                self.logger.error(f"Error loading data for symbol {symbol}: {str(e)}")
                continue

        return result


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""

    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission_per_trade: float = 1.0
    commission_percentage: float = 0.0005  # 0.05%
    slippage_bps: float = 5.0  # 5 basis points
    max_positions: int = 10
    position_sizing_method: str = "equal_weight"
    risk_free_rate: float = 0.02
    benchmark_symbol: str = "SPY"
    data_frequency: TimeFrame = TimeFrame.ONE_DAY
    cash_interest_rate: float = 0.01
    margin_rate: float = 0.08
    max_leverage: float = 1.0


@dataclass
class BacktestPosition:
    """Position during backtesting"""

    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_percentage(self) -> float:
        return (self.current_price - self.entry_price) / self.entry_price


@dataclass
class BacktestTrade:
    """Completed trade during backtesting"""

    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: OrderSide
    pnl: float
    commission: float
    slippage: float
    strategy_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def gross_pnl(self) -> float:
        return self.pnl + self.commission + self.slippage

    @property
    def return_percentage(self) -> float:
        cost_basis = abs(self.quantity * self.entry_price)
        return self.pnl / cost_basis if cost_basis > 0 else 0.0

    @property
    def holding_period_days(self) -> int:
        return (self.exit_date - self.entry_date).days


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time"""

    timestamp: datetime
    cash: float
    positions: List[BacktestPosition]
    total_value: float
    daily_pnl: float
    cumulative_pnl: float
    drawdown: float
    leverage: float = 1.0


class StrategyInterface(ABC):
    """Abstract interface for trading strategies"""

    @abstractmethod
    async def generate_signals(
        self, market_data: Dict[str, List[MarketData]], current_time: datetime
    ) -> List[TradeSignal]:
        """Generate trading signals based on market data"""
        pass

    @abstractmethod
    def get_required_history_days(self) -> int:
        """Return number of days of historical data needed"""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name"""
        pass


class MovingAverageCrossoverStrategy(StrategyInterface):
    """Moving Average Crossover Strategy for backtesting"""

    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    async def generate_signals(
        self, market_data: Dict[str, List[MarketData]], current_time: datetime
    ) -> List[TradeSignal]:
        signals = []

        for symbol, data_points in market_data.items():
            if len(data_points) < self.long_window:
                continue

            # Sort by timestamp
            data_points = sorted(data_points, key=lambda x: x.timestamp)
            closes = [d.close for d in data_points]

            # Calculate moving averages
            short_ma = sum(closes[-self.short_window :]) / self.short_window
            long_ma = sum(closes[-self.long_window :]) / self.long_window

            # Check for crossover
            if len(closes) >= self.long_window + 1:
                prev_short_ma = (
                    sum(closes[-(self.short_window + 1) : -1]) / self.short_window
                )
                prev_long_ma = (
                    sum(closes[-(self.long_window + 1) : -1]) / self.long_window
                )

                # Bullish crossover
                if short_ma > long_ma and prev_short_ma <= prev_long_ma:
                    signal = TradeSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        confidence=0.7,
                        timestamp=current_time,
                        price=data_points[-1].close,
                        quantity=100,
                        stop_loss=data_points[-1].close * Decimal("0.95"),
                        take_profit=data_points[-1].close * Decimal("1.1"),
                        strategy_name=self.get_strategy_name(),
                        metadata={
                            "short_ma": float(short_ma),
                            "long_ma": float(long_ma),
                            "crossover_type": "bullish",
                        },
                    )
                    signals.append(signal)

                # Bearish crossover
                elif short_ma < long_ma and prev_short_ma >= prev_long_ma:
                    signal = TradeSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.7,
                        timestamp=current_time,
                        price=data_points[-1].close,
                        quantity=100,
                        stop_loss=data_points[-1].close * Decimal("1.05"),
                        take_profit=data_points[-1].close * Decimal("0.9"),
                        strategy_name=self.get_strategy_name(),
                        metadata={
                            "short_ma": float(short_ma),
                            "long_ma": float(long_ma),
                            "crossover_type": "bearish",
                        },
                    )
                    signals.append(signal)

        return signals

    def get_required_history_days(self) -> int:
        return max(self.short_window, self.long_window) + 5

    def get_strategy_name(self) -> str:
        return f"moving_average_{self.short_window}_{self.long_window}"


class RSIStrategy(StrategyInterface):
    """RSI Mean Reversion Strategy"""

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    async def generate_signals(
        self, market_data: Dict[str, List[MarketData]], current_time: datetime
    ) -> List[TradeSignal]:
        signals = []

        for symbol, data_points in market_data.items():
            if len(data_points) < self.period + 1:
                continue

            data_points = sorted(data_points, key=lambda x: x.timestamp)
            closes = [float(d.close) for d in data_points]

            rsi = self._calculate_rsi(closes, self.period)

            if rsi is None:
                continue

            current_rsi = rsi[-1]

            # Generate signals based on RSI levels
            if current_rsi < self.oversold:
                signal = TradeSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=min(0.9, (self.oversold - current_rsi) / self.oversold),
                    timestamp=current_time,
                    price=data_points[-1].close,
                    quantity=100,
                    stop_loss=data_points[-1].close * Decimal("0.95"),
                    take_profit=data_points[-1].close * Decimal("1.1"),
                    strategy_name=self.get_strategy_name(),
                    metadata={"rsi": current_rsi, "condition": "oversold"},
                )
                signals.append(signal)

            elif current_rsi > self.overbought:
                signal = TradeSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=min(
                        0.9, (current_rsi - self.overbought) / (100 - self.overbought)
                    ),
                    timestamp=current_time,
                    price=data_points[-1].close,
                    quantity=100,
                    stop_loss=data_points[-1].close * Decimal("1.05"),
                    take_profit=data_points[-1].close * Decimal("0.9"),
                    strategy_name=self.get_strategy_name(),
                    metadata={"rsi": current_rsi, "condition": "overbought"},
                )
                signals.append(signal)

        return signals

    def _calculate_rsi(self, prices: List[float], period: int) -> Optional[List[float]]:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return None

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [max(0, delta) for delta in deltas]
        losses = [max(0, -delta) for delta in deltas]

        # Calculate initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        rsi_values = []

        for i in range(period, len(gains)):
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            rsi_values.append(float(rsi))

            # Update averages using smoothing
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        return rsi_values

    def get_required_history_days(self) -> int:
        return self.period + 10

    def get_strategy_name(self) -> str:
        return f"rsi_{self.period}_{self.oversold}_{self.overbought}"


class BacktestEngine:
    """Main backtesting engine"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.completed_trades: List[BacktestTrade] = []
        self.current_positions: Dict[str, BacktestPosition] = {}
        self.current_cash = config.initial_capital
        self.high_water_mark = config.initial_capital
        self.max_drawdown = 0.0

    async def run_backtest(
        self, strategy: StrategyInterface, symbols: List[str], data_source: DataSource
    ) -> BacktestResult:
        """Run complete backtest"""
        self.logger.info(
            f"Starting backtest for strategy {strategy.get_strategy_name()}"
        )
        self.logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        self.logger.info(f"Symbols: {symbols}")

        # Load historical data
        historical_data = await data_source.load_data(
            symbols,
            self.config.start_date
            - timedelta(days=strategy.get_required_history_days()),
            self.config.end_date,
            self.config.data_frequency,
        )

        # Get benchmark data
        benchmark_data = await data_source.load_data(
            [self.config.benchmark_symbol],
            self.config.start_date,
            self.config.end_date,
            self.config.data_frequency,
        )

        # Run simulation
        await self._run_simulation(strategy, symbols, historical_data)

        # Calculate performance metrics
        result = self._calculate_performance_metrics(benchmark_data)

        self.logger.info(f"Backtest completed. Total return: {result.total_return:.2%}")
        self.logger.info(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
        self.logger.info(f"Max drawdown: {result.max_drawdown:.2%}")

        return result

    async def _run_simulation(
        self,
        strategy: StrategyInterface,
        symbols: List[str],
        historical_data: Dict[str, List[MarketData]],
    ) -> None:
        """Run the actual simulation"""
        # Get all unique timestamps and sort them
        all_timestamps = set()
        for symbol_data in historical_data.values():
            for data_point in symbol_data:
                if (
                    self.config.start_date
                    <= data_point.timestamp
                    <= self.config.end_date
                ):
                    all_timestamps.add(data_point.timestamp)

        timestamps = sorted(all_timestamps)

        self.logger.info(f"Simulating {len(timestamps)} time periods")

        for i, current_time in enumerate(timestamps):
            if i % 100 == 0:
                self.logger.info(
                    f"Processing {i}/{len(timestamps)} ({current_time.date()})"
                )

            # Get market data up to current time for each symbol
            current_market_data = {}
            for symbol in symbols:
                symbol_data = historical_data.get(symbol, [])
                relevant_data = [d for d in symbol_data if d.timestamp <= current_time]
                if relevant_data:
                    current_market_data[symbol] = relevant_data

            # Update current prices for existing positions
            await self._update_position_prices(current_market_data, current_time)

            # Check for stop losses and take profits
            await self._check_exit_conditions(current_market_data, current_time)

            # Generate signals from strategy
            signals = await strategy.generate_signals(current_market_data, current_time)

            # Execute trades based on signals
            for signal in signals:
                await self._execute_signal(signal, current_market_data)

            # Record portfolio snapshot
            await self._record_portfolio_snapshot(current_time, current_market_data)

    async def _update_position_prices(
        self, market_data: Dict[str, List[MarketData]], current_time: datetime
    ) -> None:
        """Update current prices for all positions"""
        for symbol, position in self.current_positions.items():
            if symbol in market_data and market_data[symbol]:
                latest_data = market_data[symbol][-1]
                position.current_price = float(latest_data.close)

    async def _check_exit_conditions(
        self, market_data: Dict[str, List[MarketData]], current_time: datetime
    ) -> None:
        """Check stop loss, take profit, and other exit conditions"""
        positions_to_close = []

        for symbol, position in self.current_positions.items():
            if symbol not in market_data or not market_data[symbol]:
                continue

            current_price = market_data[symbol][-1].close
            exit_reason = None
            should_close = False

            # Check stop loss
            if position.stop_loss and current_price <= position.stop_loss:
                exit_reason = "stop_loss"
                should_close = True

            # Check take profit
            elif position.take_profit and current_price >= position.take_profit:
                exit_reason = "take_profit"
                should_close = True

            # Check for maximum holding period (90 days)
            elif (current_time - position.entry_date).days > 90:
                exit_reason = "max_holding_period"
                should_close = True

            # Check for trailing stop loss (5% below highest price since entry)
            elif hasattr(position, "highest_price"):
                # if not hasattr(position, 'highest_price'):
                #     position.highest_price = max(position.entry_price, float(current_price))
                # else:
                position.highest_price = max(
                    position.highest_price, float(current_price)
                )

                trailing_stop = position.highest_price * 0.95  # 5% trailing stop
                if current_price <= trailing_stop:
                    exit_reason = "trailing_stop"
                    should_close = True

            if should_close:
                trade = await self._close_position(
                    position,
                    float(current_price),
                    current_time,
                    exit_reason or "UNKNOWN",
                )
                positions_to_close.append(symbol)
                self.completed_trades.append(trade)

        # Remove closed positions
        for symbol in positions_to_close:
            del self.current_positions[symbol]

    async def _execute_signal(
        self, signal: TradeSignal, market_data: Dict[str, List[MarketData]]
    ) -> None:
        """Execute a trading signal"""
        symbol = signal.symbol

        if symbol not in market_data or not market_data[symbol]:
            return

        # Check portfolio risk limits before executing any trades
        if not self._check_portfolio_risk_limits():
            self.logger.warning(
                f"Skipping signal for {symbol} due to portfolio risk limits"
            )
            return

        current_price = market_data[symbol][-1].close

        # Apply slippage
        slippage_factor = self.config.slippage_bps / 10000
        if signal.signal_type == SignalType.BUY:
            execution_price = float(current_price) * (1 + slippage_factor)
        else:
            execution_price = float(current_price) * (1 - slippage_factor)

        if signal.signal_type == SignalType.BUY:
            await self._execute_buy_signal(signal, execution_price)
        elif signal.signal_type == SignalType.SELL:
            await self._execute_sell_signal(signal, execution_price)

    async def _execute_buy_signal(
        self, signal: TradeSignal, execution_price: float
    ) -> None:
        """Execute buy signal"""
        symbol = signal.symbol

        # Check if we already have a position
        if symbol in self.current_positions:
            return

        # Check if we've reached maximum number of positions
        if len(self.current_positions) >= self.config.max_positions:
            self.logger.debug(
                f"Cannot open position for {symbol}: max positions ({self.config.max_positions}) reached"
            )
            return

        # Calculate position size
        position_value = self._calculate_position_size(signal, execution_price)

        if position_value <= 0:
            return

        quantity = int(position_value / execution_price)
        cost = quantity * execution_price
        commission = self._calculate_commission(cost, quantity)

        # Check if we have enough cash
        if cost + commission > self.current_cash:
            self.logger.debug(
                f"Insufficient cash for {symbol}: need ${cost + commission:.2f}, have ${self.current_cash:.2f}"
            )
            return

        # Additional risk check: ensure position doesn't exceed concentration limit
        total_portfolio_value = self.current_cash + sum(
            pos.market_value for pos in self.current_positions.values()
        )
        position_weight = (
            cost / total_portfolio_value if total_portfolio_value > 0 else 0
        )

        if position_weight > 0.15:  # 15% concentration limit
            self.logger.debug(
                f"Position {symbol} would exceed 15% concentration limit ({position_weight:.2%})"
            )
            return

        # Create position
        position = BacktestPosition(
            symbol=symbol,
            quantity=quantity,
            entry_price=float(execution_price),
            entry_date=signal.timestamp,
            current_price=float(execution_price),
            stop_loss=float(signal.stop_loss) if signal.stop_loss is not None else None,
            take_profit=(
                float(signal.take_profit) if signal.take_profit is not None else None
            ),
        )

        self.current_positions[symbol] = position
        self.current_cash -= cost + commission

        self.logger.debug(
            f"Opened position: {symbol} x{quantity} @ ${execution_price:.2f}"
        )

    async def _execute_sell_signal(
        self, signal: TradeSignal, execution_price: float
    ) -> None:
        """Execute sell signal"""
        symbol = signal.symbol

        # Check if we have a position to sell
        if symbol not in self.current_positions:
            return

        position = self.current_positions[symbol]
        trade = await self._close_position(
            position, execution_price, signal.timestamp, "signal"
        )

        self.completed_trades.append(trade)
        del self.current_positions[symbol]

        self.logger.debug(
            f"Closed position: {symbol} x{position.quantity} @ ${execution_price:.2f}, PnL: ${trade.pnl:.2f}"
        )

    async def _close_position(
        self,
        position: BacktestPosition,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
    ) -> BacktestTrade:
        """Close a position and create trade record"""
        gross_pnl = (exit_price - position.entry_price) * position.quantity
        commission = self._calculate_commission(
            position.quantity * exit_price, position.quantity
        )
        slippage = abs(position.quantity * exit_price) * (
            self.config.slippage_bps / 10000
        )
        net_pnl = gross_pnl - commission - slippage

        # Add proceeds to cash
        proceeds = position.quantity * exit_price - commission
        self.current_cash += proceeds

        trade = BacktestTrade(
            symbol=position.symbol,
            entry_date=position.entry_date,
            exit_date=exit_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            side=OrderSide.BUY,  # Assuming long positions for now
            pnl=net_pnl,
            commission=commission,
            slippage=slippage,
            strategy_name="backtest",
            metadata={"exit_reason": exit_reason},
        )

        return trade

    def _calculate_position_size(self, signal: TradeSignal, price: float) -> float:
        """Calculate position size based on sizing method"""
        total_portfolio_value = self.current_cash + sum(
            pos.market_value for pos in self.current_positions.values()
        )

        if self.config.position_sizing_method == "equal_weight":
            target_positions = min(
                self.config.max_positions, 10
            )  # Assume max 10 positions
            return total_portfolio_value / target_positions

        elif self.config.position_sizing_method == "fixed_amount":
            return min(10000.0, total_portfolio_value * 0.1)  # 10% max per position

        elif self.config.position_sizing_method == "confidence_weighted":
            base_allocation = total_portfolio_value * 0.05  # 5% base
            confidence_multiplier = signal.confidence
            return base_allocation * confidence_multiplier

        elif self.config.position_sizing_method == "volatility_adjusted":
            # Risk-adjusted position sizing based on volatility
            volatility = self._estimate_volatility(signal.symbol, price)
            target_risk = 0.02  # 2% portfolio risk

            # Calculate position size based on volatility
            if volatility > 0:
                # Position size = (Portfolio Risk) / (Asset Volatility / Price)
                position_size = (total_portfolio_value * target_risk) / (
                    volatility / price
                )
                # Cap at 10% of portfolio
                max_position = total_portfolio_value * 0.1
                min_position = total_portfolio_value * 0.01  # Minimum 1%
                position_size = max(min_position, min(position_size, max_position))
            else:
                # Fallback to fixed percentage if volatility calculation fails
                position_size = total_portfolio_value * 0.05

            return position_size

        else:
            return total_portfolio_value * 0.05  # Default 5%

    def _calculate_commission(self, trade_value: float, quantity: int) -> float:
        """Calculate trading commission"""
        fixed_commission = self.config.commission_per_trade
        percentage_commission = trade_value * self.config.commission_percentage
        return fixed_commission + percentage_commission

    def _estimate_volatility(self, symbol: str, current_price: float) -> float:
        """Estimate asset volatility using historical price data"""
        # Get historical data for the symbol from portfolio history
        if not self.portfolio_history:
            return current_price * 0.02  # Fallback to 2% default

        # Look for price history for this symbol
        price_history = []
        for snapshot in self.portfolio_history[-60:]:  # Use last 60 periods
            for position in snapshot.positions:
                if position.symbol == symbol:
                    price_history.append(position.current_price)
                    break

        # If we don't have enough price history, use a reasonable default
        if len(price_history) < 10:
            return current_price * 0.02

        # Calculate returns
        returns = []
        for i in range(1, len(price_history)):
            if price_history[i - 1] > 0:
                daily_return = (
                    price_history[i] - price_history[i - 1]
                ) / price_history[i - 1]
                returns.append(daily_return)

        if not returns:
            return current_price * 0.02

        # Calculate annualized volatility
        daily_volatility = np.std(returns)
        annualized_volatility = daily_volatility * np.sqrt(
            252
        )  # 252 trading days per year

        # Return as absolute price volatility
        return current_price * annualized_volatility

    async def _record_portfolio_snapshot(
        self, timestamp: datetime, market_data: Dict[str, List[MarketData]]
    ) -> None:
        """Record current portfolio state"""
        # Update position values
        total_position_value = 0.0
        for position in self.current_positions.values():
            if position.symbol in market_data and market_data[position.symbol]:
                position.current_price = float(market_data[position.symbol][-1].close)
                total_position_value += position.market_value

        total_value = self.current_cash + total_position_value

        # Calculate daily PnL
        daily_pnl = 0.0
        if self.portfolio_history:
            daily_pnl = total_value - self.portfolio_history[-1].total_value

        # Calculate cumulative PnL
        cumulative_pnl = total_value - self.config.initial_capital

        # Update high water mark and drawdown
        if total_value > self.high_water_mark:
            self.high_water_mark = total_value

        current_drawdown = (self.high_water_mark - total_value) / self.high_water_mark
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.current_cash,
            positions=list(self.current_positions.values()),
            total_value=total_value,
            daily_pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
            drawdown=current_drawdown,
            leverage=total_position_value / total_value if total_value > 0 else 0.0,
        )

        self.portfolio_history.append(snapshot)

    def _check_portfolio_risk_limits(self) -> bool:
        """Check if portfolio exceeds risk limits"""
        if not self.portfolio_history:
            return True

        current_snapshot = self.portfolio_history[-1]

        # Check maximum drawdown limit (stop trading if drawdown > 20%)
        if current_snapshot.drawdown > 0.20:
            self.logger.warning(
                f"Portfolio drawdown {current_snapshot.drawdown:.2%} exceeds 20% limit"
            )
            return False

        # Check leverage limit
        if current_snapshot.leverage > self.config.max_leverage:
            self.logger.warning(
                f"Portfolio leverage {current_snapshot.leverage:.2f} exceeds limit {self.config.max_leverage}"
            )
            return False

        # Check concentration risk (no single position > 15% of portfolio)
        if current_snapshot.total_value > 0:
            for position in current_snapshot.positions:
                position_weight = position.market_value / current_snapshot.total_value
                if position_weight > 0.15:
                    self.logger.warning(
                        f"Position {position.symbol} weight {position_weight:.2%} exceeds 15% limit"
                    )
                    return False

        return True

    def _calculate_value_at_risk(self, confidence_level: float = 0.05) -> float:
        """Calculate portfolio Value at Risk (VaR)"""
        if len(self.portfolio_history) < 30:
            return 0.0

        # Get recent daily returns
        returns = []
        for i in range(1, min(len(self.portfolio_history), 252)):  # Last year of data
            prev_value = self.portfolio_history[-(i + 1)].total_value
            curr_value = self.portfolio_history[-i].total_value
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)

        if not returns:
            return 0.0

        # Calculate VaR at given confidence level
        returns.sort()
        var_index = int(len(returns) * confidence_level)
        var_return = returns[var_index] if var_index < len(returns) else returns[0]

        current_value = self.portfolio_history[-1].total_value
        return abs(var_return * current_value)

    def _validate_strategy_performance(self) -> Dict[str, Any]:
        """Validate strategy performance and identify potential issues"""
        validation_results: Dict[str, Any] = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "metrics": {},
        }

        if not self.completed_trades:
            validation_results["errors"].append("No completed trades found")
            validation_results["is_valid"] = False
            return validation_results

        # Check for overfitting indicators
        total_trades = len(self.completed_trades)
        winning_trades = len([t for t in self.completed_trades if t.pnl > 0])

        # Too few trades
        if total_trades < 30:
            validation_results["warnings"].append(
                f"Low trade count ({total_trades}). Results may not be statistically significant."
            )

        # Suspiciously high win rate
        win_rate = winning_trades / total_trades
        if win_rate > 0.85:
            validation_results["warnings"].append(
                f"Very high win rate ({win_rate:.2%}). May indicate overfitting or unrealistic assumptions."
            )

        # Check for data snooping bias
        recent_trades = [
            t
            for t in self.completed_trades
            if (self.config.end_date - t.exit_date).days < 30
        ]
        if len(recent_trades) > total_trades * 0.5:
            validation_results["warnings"].append(
                "More than 50% of trades occurred in last 30 days. Check for forward bias."
            )

        # Calculate trade distribution metrics
        trade_pnls = [float(t.pnl) for t in self.completed_trades]

        # Check for outlier dependence
        sorted_pnls = sorted(trade_pnls, reverse=True)
        top_5_percent = int(max(1, len(sorted_pnls) * 0.05))
        top_trades_contribution = (
            sum(sorted_pnls[:top_5_percent]) / sum(trade_pnls)
            if sum(trade_pnls) != 0
            else 0
        )

        if top_trades_contribution > 0.8:
            validation_results["warnings"].append(
                f"Top 5% of trades contribute {top_trades_contribution:.2%} of total PnL. Strategy may be too dependent on outliers."
            )

        # Check for consistency across time periods
        if len(self.portfolio_history) > 60:
            quarterly_returns = []
            quarter_size = len(self.portfolio_history) // 4

            for i in range(4):
                start_idx = i * quarter_size
                end_idx = (
                    (i + 1) * quarter_size if i < 3 else len(self.portfolio_history)
                )

                if start_idx < len(self.portfolio_history) and end_idx <= len(
                    self.portfolio_history
                ):
                    start_value = self.portfolio_history[start_idx].total_value
                    end_value = self.portfolio_history[end_idx - 1].total_value
                    quarterly_return = (
                        (end_value - start_value) / start_value
                        if start_value > 0
                        else 0
                    )
                    quarterly_returns.append(quarterly_return)

            if len(quarterly_returns) == 4:
                negative_quarters = sum(1 for r in quarterly_returns if r < 0)
                if negative_quarters >= 3:
                    validation_results["warnings"].append(
                        f"Strategy had negative returns in {negative_quarters}/4 quarters. Check for regime dependency."
                    )

        validation_results["metrics"] = {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "outlier_dependence": top_trades_contribution,
            "recent_trade_ratio": (
                len(recent_trades) / total_trades if total_trades > 0 else 0
            ),
        }

        return validation_results

    def _calculate_advanced_metrics(self) -> Dict[str, float]:
        """Calculate additional performance metrics for production use"""
        if not self.portfolio_history or len(self.portfolio_history) < 2:
            return {}

        portfolio_values = [snapshot.total_value for snapshot in self.portfolio_history]
        returns = [
            (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
            for i in range(1, len(portfolio_values))
            if portfolio_values[i - 1] > 0
        ]

        if not returns:
            return {}

        metrics = {}

        # Calmar Ratio (already implemented but add to advanced metrics)
        total_return = (
            portfolio_values[-1] - self.config.initial_capital
        ) / self.config.initial_capital
        metrics["calmar_ratio"] = self._calculate_calmar_ratio(
            total_return, self.max_drawdown
        )

        # Maximum Consecutive Losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for r in returns:
            if r < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        metrics["max_consecutive_losses"] = max_consecutive_losses

        # Skewness and Kurtosis
        if len(returns) > 3:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                skewness = np.mean(
                    [((r - mean_return) / std_return) ** 3 for r in returns]
                )
                kurtosis = (
                    np.mean([((r - mean_return) / std_return) ** 4 for r in returns])
                    - 3
                )
                metrics["skewness"] = float(skewness)
                metrics["kurtosis"] = float(kurtosis)

        # Tail Ratio (95th percentile / 5th percentile)
        if len(returns) > 20:
            sorted_returns = sorted(returns)
            p95 = sorted_returns[int(0.95 * len(sorted_returns))]
            p5 = sorted_returns[int(0.05 * len(sorted_returns))]
            if p5 != 0:
                metrics["tail_ratio"] = abs(p95 / p5)

        # Recovery Factor (total return / max drawdown)
        if self.max_drawdown > 0:
            metrics["recovery_factor"] = float(total_return / self.max_drawdown)

        return metrics

    def _calculate_performance_metrics(
        self, benchmark_data: Dict[str, List[MarketData]]
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        if not self.portfolio_history:
            raise ValueError("No portfolio history to analyze")

        # Calculate returns
        portfolio_values = [snapshot.total_value for snapshot in self.portfolio_history]
        portfolio_returns = [
            (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
            for i in range(1, len(portfolio_values))
        ]

        # Calculate benchmark returns
        if self.config.benchmark_symbol in benchmark_data:
            pass
            # benchmark_returns = [
            #     (benchmark_prices[i] - benchmark_prices[i - 1])
            #     / benchmark_prices[i - 1]
            #     for i in range(1, len(benchmark_prices))
            # ]

        # Basic metrics
        total_return = (
            portfolio_values[-1] - self.config.initial_capital
        ) / self.config.initial_capital

        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(
            portfolio_returns, self.config.risk_free_rate
        )
        sortino_ratio = self._calculate_sortino_ratio(
            portfolio_returns, self.config.risk_free_rate
        )

        # Trade statistics
        total_trades = len(self.completed_trades)
        winning_trades = len([t for t in self.completed_trades if t.pnl > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        avg_win = (
            np.mean([t.pnl for t in self.completed_trades if t.pnl > 0])
            if winning_trades > 0
            else 0.0
        )
        avg_loss = (
            np.mean([t.pnl for t in self.completed_trades if t.pnl < 0])
            if losing_trades > 0
            else 0.0
        )
        profit_factor: float = (
            float(
                abs(float(avg_win) * winning_trades / (float(avg_loss) * losing_trades))
            )
            if avg_loss != 0
            else float("inf")
        )

        # Run strategy validation
        validation_results = self._validate_strategy_performance()
        if not validation_results["is_valid"]:
            self.logger.error("Strategy validation failed:")
            for error in validation_results["errors"]:
                self.logger.error(f"  - {error}")

        for warning in validation_results["warnings"]:
            self.logger.warning(f"Strategy validation warning: {warning}")

        # Calculate advanced metrics
        advanced_metrics = self._calculate_advanced_metrics()
        self.logger.info("Advanced performance metrics calculated")
        for metric, value in advanced_metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")

        return BacktestResult(
            strategy_name=(
                self.completed_trades[0].strategy_name
                if self.completed_trades
                else "unknown"
            ),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=Decimal(str(self.config.initial_capital)),
            final_capital=Decimal(str(portfolio_values[-1])),
            total_return=Decimal(str(total_return)),
            total_return_pct=float(total_return * 100),
            annualized_return=float(self._annualize_return(total_return)),
            max_drawdown=Decimal(str(self.max_drawdown)),
            max_drawdown_pct=float(self.max_drawdown * 100),
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=Decimal(str(avg_win)) if avg_win > 0 else Decimal("0"),
            avg_loss=Decimal(str(avg_loss)) if avg_loss < 0 else Decimal("0"),
            profit_factor=float(profit_factor) if not np.isinf(profit_factor) else 0.0,
        )

    def _calculate_sharpe_ratio(
        self, returns: List[float], risk_free_rate: float
    ) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0

        excess_returns = [
            r - risk_free_rate / 252 for r in returns
        ]  # Daily risk-free rate
        return (
            float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))
            if np.std(excess_returns) > 0
            else 0.0
        )

    def _calculate_sortino_ratio(
        self, returns: List[float], risk_free_rate: float
    ) -> float:
        """Calculate Sortino ratio"""
        if not returns:
            return 0.0

        excess_returns = [r - risk_free_rate / 252 for r in returns]
        downside_returns = [r for r in excess_returns if r < 0]

        if not downside_returns:
            return float("inf") if np.mean(excess_returns) > 0 else 0.0

        downside_deviation = np.std(downside_returns)
        return (
            np.mean(excess_returns) / downside_deviation * np.sqrt(252)
            if downside_deviation > 0
            else 0.0
        )

    def _calculate_beta(
        self, portfolio_returns: List[float], benchmark_returns: List[float]
    ) -> float:
        """Calculate portfolio beta"""
        if (
            len(portfolio_returns) != len(benchmark_returns)
            or len(portfolio_returns) < 2
        ):
            return 1.0

        covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
        benchmark_variance = np.var(benchmark_returns)

        return float(covariance / benchmark_variance) if benchmark_variance > 0 else 1.0

    def _calculate_alpha(
        self, total_return: float, benchmark_returns: List[float], beta: float
    ) -> float:
        """Calculate portfolio alpha"""
        if not benchmark_returns:
            return total_return

        benchmark_total_return = np.prod([1 + r for r in benchmark_returns]) - 1
        expected_return = self.config.risk_free_rate + beta * (
            benchmark_total_return - self.config.risk_free_rate
        )

        return float(total_return - expected_return)

    def _calculate_calmar_ratio(
        self, total_return: float, max_drawdown: float
    ) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown <= 0:
            return 0.0

        annualized_return = self._annualize_return(total_return)
        return (
            float(float(annualized_return) / abs(max_drawdown))
            if abs(max_drawdown) > 0
            else 0.0
        )

    def _annualize_return(self, total_return: float) -> float:
        """Annualize the total return based on backtest period"""
        days = (self.config.end_date - self.config.start_date).days
        years = days / 365.25
        return (
            float(((1 + total_return) ** (1 / years)) - 1)
            if years > 0
            else float(total_return)
        )


class WalkForwardAnalysis:
    """Walk-forward analysis for strategy optimization"""

    def __init__(self, window_size: int = 252, step_size: int = 21):
        """Initialize walk-forward analysis

        Args:
            window_size: Size of training window in trading days
            step_size: Step size for moving window in trading days
        """
        self.window_size = window_size
        self.step_size = step_size
        self.logger = logging.getLogger(__name__)

    async def run_analysis(
        self,
        strategy: StrategyInterface,
        symbols: List[str],
        data_source: DataSource,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Run walk-forward analysis on strategy"""
        self.logger.info("Starting walk-forward analysis")

        # Generate time windows
        windows = self._generate_windows(start_date, end_date)
        results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            self.logger.info(f"Processing window {i + 1}/{len(windows)}")

            # Create backtest config for this window
            config = BacktestConfig(
                start_date=test_start,
                end_date=test_end,
                initial_capital=100000.0,
                commission_per_trade=1.0,
            )

            # Run backtest
            engine = BacktestEngine(config)
            result = await engine.run_backtest(strategy, symbols, data_source)
            results.append(result)

        return self._aggregate_results(results)

    def _generate_windows(
        self, start_date: datetime, end_date: datetime
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate overlapping time windows for walk-forward analysis"""
        windows = []
        current_date = start_date

        while (
            current_date + timedelta(days=self.window_size + self.step_size) <= end_date
        ):
            train_start = current_date
            train_end = current_date + timedelta(days=self.window_size)
            test_start = train_end
            test_end = train_end + timedelta(days=self.step_size)

            windows.append((train_start, train_end, test_start, test_end))
            current_date += timedelta(days=self.step_size)

        return windows

    def _aggregate_results(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """Aggregate results from multiple windows"""
        if not results:
            return {}

        total_return = sum(
            float(r.total_return) for r in results if r.total_return is not None
        ) / len(results)
        sharpe_ratio = sum(
            r.sharpe_ratio for r in results if r.sharpe_ratio is not None
        ) / len(results)
        max_drawdown = max(
            float(r.max_drawdown) for r in results if r.max_drawdown is not None
        )

        return {
            "average_return": total_return,
            "average_sharpe": sharpe_ratio,
            "worst_drawdown": max_drawdown,
            "num_windows": len(results),
            "consistency": sum(1 for r in results if r.total_return > 0) / len(results),
        }


# Example usage demonstrating the implemented features
async def example_backtest() -> Optional[BacktestResult]:
    """Example demonstrating how to use the backtest engine with existing data infrastructure"""
    from services.data_collector.src.data_store import DataStore, DataStoreConfig

    # Initialize the data store using existing infrastructure
    config = DataStoreConfig(base_path="./data")
    data_store = DataStore(config)

    # Create data source using our LocalDataStoreSource
    data_source = LocalDataStoreSource(data_store)

    # Configure backtest parameters
    backtest_config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=100000.0,
        commission_per_trade=1.0,
        commission_percentage=0.001,  # 0.1%
        slippage_bps=5.0,
        max_positions=10,
        position_sizing_method="volatility_adjusted",  # Uses our improved implementation
        risk_free_rate=0.02,
        benchmark_symbol="SPY",
        data_frequency=TimeFrame.ONE_DAY,
        max_leverage=1.0,
    )

    # Create strategy (using the existing RSI strategy as example)
    strategy = RSIStrategy(period=14, oversold=30, overbought=70)

    # Define symbols to test
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY"]

    # Initialize and run backtest
    engine = BacktestEngine(backtest_config)

    try:
        # Run the backtest with our improved features
        result = await engine.run_backtest(strategy, symbols, data_source)

        # Display results
        print("\n=== Backtest Results ===")
        print(f"Strategy: {result.strategy_name}")
        print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
        print(f"Initial Capital: ${result.initial_capital:,.2f}")
        print(f"Final Capital: ${result.final_capital:,.2f}")
        print(f"Total Return: {result.total_return_pct:.2f}%")
        print(f"Annualized Return: {result.annualized_return:.2f}%")
        print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Profit Factor: {result.profit_factor:.2f}")

        # Display VaR calculation using our new feature
        var_5pct = engine._calculate_value_at_risk(0.05)
        print(f"Value at Risk (5%): ${var_5pct:,.2f}")

        # Show some portfolio snapshots
        if engine.portfolio_history:
            print("\n=== Portfolio Evolution ===")
            for i in [0, len(engine.portfolio_history) // 2, -1]:
                snapshot = engine.portfolio_history[i]
                print(f"Date: {snapshot.timestamp.date()}")
                print(f"  Total Value: ${snapshot.total_value:,.2f}")
                print(f"  Cash: ${snapshot.cash:,.2f}")
                print(f"  Positions: {len(snapshot.positions)}")
                print(f"  Leverage: {snapshot.leverage:.2f}")
                print(f"  Drawdown: {snapshot.drawdown:.2%}")

        return result

    except Exception as e:
        print(f"Backtest failed: {e}")
        return None


async def example_walk_forward_analysis() -> Optional[Dict[str, Any]]:
    """Example of walk-forward analysis using the implemented features"""
    from services.data_collector.src.data_store import DataStore, DataStoreConfig

    # Initialize data infrastructure
    config = DataStoreConfig(base_path="./data")
    data_store = DataStore(config)
    data_source = LocalDataStoreSource(data_store)

    # Create strategy
    strategy = MovingAverageCrossoverStrategy(short_window=20, long_window=50)

    # Define test parameters
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)

    # Run walk-forward analysis
    wfa = WalkForwardAnalysis(
        window_size=252, step_size=21
    )  # 1 year training, 1 month testing

    try:
        results = await wfa.run_analysis(
            strategy, symbols, data_source, start_date, end_date
        )

        print("\n=== Walk-Forward Analysis Results ===")
        print(f"Strategy: {strategy.get_strategy_name()}")
        print(f"Average Return: {results['average_return']:.2%}")
        print(f"Average Sharpe: {results['average_sharpe']:.2f}")
        print(f"Worst Drawdown: {results['worst_drawdown']:.2%}")
        print(f"Number of Windows: {results['num_windows']}")
        print(f"Consistency (% positive): {results['consistency']:.2%}")

        return results

    except Exception as e:
        print(f"Walk-forward analysis failed: {e}")
        return None


if __name__ == "__main__":
    import asyncio

    print("Running backtest engine example with implemented features...")

    # Run the example backtest
    asyncio.run(example_backtest())

    print("\n" + "=" * 50)

    # Run walk-forward analysis example
    asyncio.run(example_walk_forward_analysis())
