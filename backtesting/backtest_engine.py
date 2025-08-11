import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import shared models
import sys
import os
from decimal import Decimal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.models import (
    MarketData, TradeSignal, SignalType, OrderSide,
    BacktestResult,
    TimeFrame
)

# Define DataSource interface
class DataSource(ABC):
    """Abstract data source interface for backtesting"""

    @abstractmethod
    async def load_data(self, symbols: List[str], start_date: datetime,
                       end_date: datetime, timeframe: TimeFrame) -> Dict[str, List[MarketData]]:
        """Load market data for given symbols and date range"""
        pass


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
    async def generate_signals(self, market_data: Dict[str, List[MarketData]],
                             current_time: datetime) -> List[TradeSignal]:
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

    async def generate_signals(self, market_data: Dict[str, List[MarketData]],
                             current_time: datetime) -> List[TradeSignal]:
        signals = []

        for symbol, data_points in market_data.items():
            if len(data_points) < self.long_window:
                continue

            # Sort by timestamp
            data_points = sorted(data_points, key=lambda x: x.timestamp)
            closes = [d.close for d in data_points]

            # Calculate moving averages
            short_ma = sum(closes[-self.short_window:]) / self.short_window
            long_ma = sum(closes[-self.long_window:]) / self.long_window

            # Check for crossover
            if len(closes) >= self.long_window + 1:
                prev_short_ma = sum(closes[-(self.short_window+1):-1]) / self.short_window
                prev_long_ma = sum(closes[-(self.long_window+1):-1]) / self.long_window

                # Bullish crossover
                if short_ma > long_ma and prev_short_ma <= prev_long_ma:
                    signal = TradeSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        confidence=0.7,
                        timestamp=current_time,
                        price=data_points[-1].close,
                        quantity=100,
                        stop_loss=data_points[-1].close * Decimal('0.95'),
                        take_profit=data_points[-1].close * Decimal('1.1'),
                        strategy_name=self.get_strategy_name(),
                        metadata={
                            "short_ma": float(short_ma),
                            "long_ma": float(long_ma),
                            "crossover_type": "bullish"
                        }
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
                        stop_loss=data_points[-1].close * Decimal('1.05'),
                        take_profit=data_points[-1].close * Decimal('0.9'),
                        strategy_name=self.get_strategy_name(),
                        metadata={
                            "short_ma": float(short_ma),
                            "long_ma": float(long_ma),
                            "crossover_type": "bearish"
                        }
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

    async def generate_signals(self, market_data: Dict[str, List[MarketData]],
                             current_time: datetime) -> List[TradeSignal]:
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
                    stop_loss=data_points[-1].close * Decimal('0.95'),
                    take_profit=data_points[-1].close * Decimal('1.1'),
                    strategy_name=self.get_strategy_name(),
                    metadata={"rsi": current_rsi, "condition": "oversold"}
                )
                signals.append(signal)

            elif current_rsi > self.overbought:
                signal = TradeSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=min(0.9, (current_rsi - self.overbought) / (100 - self.overbought)),
                    timestamp=current_time,
                    price=data_points[-1].close,
                    quantity=100,
                    stop_loss=data_points[-1].close * Decimal('1.05'),
                    take_profit=data_points[-1].close * Decimal('0.9'),
                    strategy_name=self.get_strategy_name(),
                    metadata={"rsi": current_rsi, "condition": "overbought"}
                )
                signals.append(signal)

        return signals

    def _calculate_rsi(self, prices: List[float], period: int) -> Optional[List[float]]:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return None

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, delta) for delta in deltas]
        losses = [max(0, -delta) for delta in deltas]

        # Calculate initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        rsi_values = []

        for i in range(period, len(gains)):
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            rsi_values.append(rsi)

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

    async def run_backtest(self, strategy: StrategyInterface,
                          symbols: List[str],
                          data_source: DataSource) -> BacktestResult:
        """Run complete backtest"""
        self.logger.info(f"Starting backtest for strategy {strategy.get_strategy_name()}")
        self.logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        self.logger.info(f"Symbols: {symbols}")

        # Load historical data
        historical_data = await data_source.load_data(
            symbols,
            self.config.start_date - timedelta(days=strategy.get_required_history_days()),
            self.config.end_date,
            self.config.data_frequency
        )

        # Get benchmark data
        benchmark_data = await data_source.load_data(
            [self.config.benchmark_symbol],
            self.config.start_date,
            self.config.end_date,
            self.config.data_frequency
        )

        # Run simulation
        await self._run_simulation(strategy, symbols, historical_data)

        # Calculate performance metrics
        result = self._calculate_performance_metrics(benchmark_data)

        self.logger.info(f"Backtest completed. Total return: {result.total_return:.2%}")
        self.logger.info(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
        self.logger.info(f"Max drawdown: {result.max_drawdown:.2%}")

        return result

    async def _run_simulation(self, strategy: StrategyInterface,
                            symbols: List[str],
                            historical_data: Dict[str, List[MarketData]]):
        """Run the actual simulation"""
        # Get all unique timestamps and sort them
        all_timestamps = set()
        for symbol_data in historical_data.values():
            for data_point in symbol_data:
                if self.config.start_date <= data_point.timestamp <= self.config.end_date:
                    all_timestamps.add(data_point.timestamp)

        timestamps = sorted(all_timestamps)

        self.logger.info(f"Simulating {len(timestamps)} time periods")

        for i, current_time in enumerate(timestamps):
            if i % 100 == 0:
                self.logger.info(f"Processing {i}/{len(timestamps)} ({current_time.date()})")

            # Get market data up to current time for each symbol
            current_market_data = {}
            for symbol in symbols:
                symbol_data = historical_data.get(symbol, [])
                relevant_data = [
                    d for d in symbol_data
                    if d.timestamp <= current_time
                ]
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

    async def _update_position_prices(self, market_data: Dict[str, List[MarketData]],
                                    current_time: datetime):
        """Update current prices for all positions"""
        for symbol, position in self.current_positions.items():
            if symbol in market_data and market_data[symbol]:
                latest_data = market_data[symbol][-1]
                position.current_price = float(latest_data.close)

    async def _check_exit_conditions(self, market_data: Dict[str, List[MarketData]],
                                   current_time: datetime):
        """Check stop loss and take profit conditions"""
        positions_to_close = []

        for symbol, position in self.current_positions.items():
            if symbol not in market_data or not market_data[symbol]:
                continue

            current_price = market_data[symbol][-1].close

            # Check stop loss
            if position.stop_loss and current_price <= position.stop_loss:
                trade = await self._close_position(position, float(current_price), current_time, "stop_loss")
                positions_to_close.append(symbol)
                self.completed_trades.append(trade)

            # Check take profit
            elif position.take_profit and current_price >= position.take_profit:
                trade = await self._close_position(position, float(current_price), current_time, "take_profit")
                positions_to_close.append(symbol)
                self.completed_trades.append(trade)

        # Remove closed positions
        for symbol in positions_to_close:
            del self.current_positions[symbol]

    async def _execute_signal(self, signal: TradeSignal,
                            market_data: Dict[str, List[MarketData]]):
        """Execute a trading signal"""
        symbol = signal.symbol

        if symbol not in market_data or not market_data[symbol]:
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

    async def _execute_buy_signal(self, signal: TradeSignal, execution_price: float):
        """Execute buy signal"""
        symbol = signal.symbol

        # Check if we already have a position
        if symbol in self.current_positions:
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
            return

        # Create position
        position = BacktestPosition(
            symbol=symbol,
            quantity=quantity,
            entry_price=float(execution_price),
            entry_date=signal.timestamp,
            current_price=float(execution_price),
            stop_loss=float(signal.stop_loss) if signal.stop_loss is not None else None,
            take_profit=float(signal.take_profit) if signal.take_profit is not None else None
        )

        self.current_positions[symbol] = position
        self.current_cash -= (cost + commission)

        self.logger.debug(f"Opened position: {symbol} x{quantity} @ ${execution_price:.2f}")

    async def _execute_sell_signal(self, signal: TradeSignal, execution_price: float):
        """Execute sell signal"""
        symbol = signal.symbol

        # Check if we have a position to sell
        if symbol not in self.current_positions:
            return

        position = self.current_positions[symbol]
        trade = await self._close_position(position, execution_price, signal.timestamp, "signal")

        self.completed_trades.append(trade)
        del self.current_positions[symbol]

        self.logger.debug(f"Closed position: {symbol} x{position.quantity} @ ${execution_price:.2f}, PnL: ${trade.pnl:.2f}")

    async def _close_position(self, position: BacktestPosition, exit_price: float,
                            exit_time: datetime, exit_reason: str) -> BacktestTrade:
        """Close a position and create trade record"""
        gross_pnl = (exit_price - position.entry_price) * position.quantity
        commission = self._calculate_commission(position.quantity * exit_price, position.quantity)
        slippage = abs(position.quantity * exit_price) * (self.config.slippage_bps / 10000)
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
            metadata={"exit_reason": exit_reason}
        )

        return trade

    def _calculate_position_size(self, signal: TradeSignal, price: float) -> float:
        """Calculate position size based on sizing method"""
        total_portfolio_value = self.current_cash + sum(
            pos.market_value for pos in self.current_positions.values()
        )

        if self.config.position_sizing_method == "equal_weight":
            target_positions = min(self.config.max_positions, 10)  # Assume max 10 positions
            return total_portfolio_value / target_positions

        elif self.config.position_sizing_method == "fixed_amount":
            return min(10000.0, total_portfolio_value * 0.1)  # 10% max per position

        elif self.config.position_sizing_method == "confidence_weighted":
            base_allocation = total_portfolio_value * 0.05  # 5% base
            confidence_multiplier = signal.confidence
            return base_allocation * confidence_multiplier

        elif self.config.position_sizing_method == "volatility_adjusted":
            # Placeholder for volatility-based sizing
            volatility = self._estimate_volatility(signal.symbol, price)
            target_risk = 0.02  # 2% portfolio risk
            position_size = (total_portfolio_value * target_risk) / volatility
            return min(position_size, total_portfolio_value * 0.1)

        else:
            return total_portfolio_value * 0.05  # Default 5%

    def _calculate_commission(self, trade_value: float, quantity: int) -> float:
        """Calculate trading commission"""
        fixed_commission = self.config.commission_per_trade
        percentage_commission = trade_value * self.config.commission_percentage
        return fixed_commission + percentage_commission

    def _estimate_volatility(self, symbol: str, current_price: float) -> float:
        """Estimate asset volatility (placeholder implementation)"""
        # In a real implementation, this would calculate historical volatility
        return current_price * 0.02  # 2% default volatility

    async def _record_portfolio_snapshot(self, timestamp: datetime,
                                       market_data: Dict[str, List[MarketData]]):
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
            leverage=total_position_value / total_value if total_value > 0 else 0.0
        )

        self.portfolio_history.append(snapshot)

    def _calculate_performance_metrics(self, benchmark_data: Dict[str, List[MarketData]]) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        if not self.portfolio_history:
            raise ValueError("No portfolio history to analyze")

        # Calculate returns
        portfolio_values = [snapshot.total_value for snapshot in self.portfolio_history]
        portfolio_returns = [
            (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            for i in range(1, len(portfolio_values))
        ]

        # Calculate benchmark returns
        benchmark_returns = []
        if self.config.benchmark_symbol in benchmark_data:
            benchmark_prices = [d.close for d in benchmark_data[self.config.benchmark_symbol]]
            benchmark_returns = [
                (benchmark_prices[i] - benchmark_prices[i-1]) / benchmark_prices[i-1]
                for i in range(1, len(benchmark_prices))
            ]

        # Basic metrics
        total_return = (portfolio_values[-1] - self.config.initial_capital) / self.config.initial_capital

        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns, self.config.risk_free_rate)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns, self.config.risk_free_rate)

        # Trade statistics
        total_trades = len(self.completed_trades)
        winning_trades = len([t for t in self.completed_trades if t.pnl > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        avg_win = np.mean([t.pnl for t in self.completed_trades if t.pnl > 0]) if winning_trades > 0 else 0.0
        avg_loss = np.mean([t.pnl for t in self.completed_trades if t.pnl < 0]) if losing_trades > 0 else 0.0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else float('inf')





        return BacktestResult(
            strategy_name=self.completed_trades[0].strategy_name if self.completed_trades else "unknown",
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
            profit_factor=float(profit_factor) if not np.isinf(profit_factor) else 0.0
        )

    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0

        excess_returns = [r - risk_free_rate/252 for r in returns]  # Daily risk-free rate
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)) if np.std(excess_returns) > 0 else 0.0

    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float) -> float:
        """Calculate Sortino ratio"""
        if not returns:
            return 0.0

        excess_returns = [r - risk_free_rate/252 for r in returns]
        downside_returns = [r for r in excess_returns if r < 0]

        if not downside_returns:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0

        downside_deviation = np.std(downside_returns)
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0.0

    def _calculate_beta(self, portfolio_returns: List[float], benchmark_returns: List[float]) -> float:
        """Calculate portfolio beta"""
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return 1.0

        covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
        benchmark_variance = np.var(benchmark_returns)

        return float(covariance / benchmark_variance) if benchmark_variance > 0 else 1.0

    def _calculate_alpha(self, total_return: float, benchmark_returns: List[float], beta: float) -> float:
        """Calculate portfolio alpha"""
        if not benchmark_returns:
            return total_return

        benchmark_total_return = np.prod([1 + r for r in benchmark_returns]) - 1
        expected_return = self.config.risk_free_rate + beta * (benchmark_total_return - self.config.risk_free_rate)

        return float(total_return - expected_return)

    def _calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown <= 0:
            return 0.0

        annualized_return = self._annualize_return(total_return)
        return float(float(annualized_return) / abs(max_drawdown)) if abs(max_drawdown) > 0 else 0.0

    def _annualize_return(self, total_return: float) -> float:
        """Annualize the total return based on backtest period"""
        days = (self.config.end_date - self.config.start_date).days
        years = days / 365.25
        return float(((1 + total_return) ** (1 / years)) - 1) if years > 0 else float(total_return)


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

    async def run_analysis(self, strategy: StrategyInterface, symbols: List[str],
                          data_source: DataSource, start_date: datetime,
                          end_date: datetime) -> Dict[str, Any]:
        """Run walk-forward analysis on strategy"""
        self.logger.info("Starting walk-forward analysis")

        # Generate time windows
        windows = self._generate_windows(start_date, end_date)
        results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            self.logger.info(f"Processing window {i+1}/{len(windows)}")

            # Create backtest config for this window
            config = BacktestConfig(
                start_date=test_start,
                end_date=test_end,
                initial_capital=100000.0,
                commission_per_trade=1.0
            )

            # Run backtest
            engine = BacktestEngine(config)
            result = await engine.run_backtest(strategy, symbols, data_source)
            results.append(result)

        return self._aggregate_results(results)

    def _generate_windows(self, start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate overlapping time windows for walk-forward analysis"""
        windows = []
        current_date = start_date

        while current_date + timedelta(days=self.window_size + self.step_size) <= end_date:
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

        total_return = sum(float(r.total_return) for r in results if r.total_return is not None) / len(results)
        sharpe_ratio = sum(r.sharpe_ratio for r in results if r.sharpe_ratio is not None) / len(results)
        max_drawdown = max(float(r.max_drawdown) for r in results if r.max_drawdown is not None)

        return {
            'average_return': total_return,
            'average_sharpe': sharpe_ratio,
            'worst_drawdown': max_drawdown,
            'num_windows': len(results),
            'consistency': sum(1 for r in results if r.total_return > 0) / len(results)
        }
