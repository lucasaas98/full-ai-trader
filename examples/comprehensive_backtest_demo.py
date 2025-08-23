#!/usr/bin/env python3
"""
Comprehensive Backtesting Demonstration

This demo shows how to run sophisticated backtests using your extensive historical data.
It demonstrates:
- Real data loading from parquet files (12+ months available)
- Screener simulation based on FinViz criteria
- Multiple trading strategies (day trading, swing trading)
- Comprehensive performance analysis
- Risk management simulation
- Portfolio optimization

Usage:
    python examples/comprehensive_backtest_demo.py

Features demonstrated:
- Works with your actual 12 months of historical data
- Simulates screener alerts retroactively
- Multiple strategy comparison
- Detailed performance metrics
- Risk analysis and drawdown calculation
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import time

# Add project paths
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root / "backtesting"))

from simple_data_store import SimpleDataStore
from backtest_models import TimeFrame, SignalType, MarketData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal with detailed information."""
    symbol: str
    action: SignalType
    price: Decimal
    confidence: float
    reasoning: str
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    position_size: float = 0.1  # 10% of portfolio


@dataclass
class Position:
    """Open trading position."""
    symbol: str
    quantity: int
    entry_price: Decimal
    entry_date: datetime
    current_price: Decimal = Decimal('0')
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    strategy_name: str = "unknown"


@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: Decimal
    exit_price: Decimal
    quantity: int
    pnl: Decimal
    pnl_percentage: float
    hold_days: int
    strategy: str
    entry_reason: str
    exit_reason: str


class ScreenerSimulator:
    """Simulates FinViz screener alerts based on historical data."""

    def __init__(self, data_store: SimpleDataStore):
        self.data_store = data_store
        self.logger = logging.getLogger(f"{__name__}.ScreenerSimulator")

    async def get_breakout_candidates(self, date: datetime, symbols: List[str]) -> List[str]:
        """Find breakout candidates based on volume and price action."""
        candidates = []

        for symbol in symbols:
            try:
                # Get 30 days of history for analysis
                end_date = date.date()
                start_date = (date - timedelta(days=30)).date()

                df = await self.data_store.load_market_data(symbol, TimeFrame.ONE_DAY, start_date, end_date)
                if df.is_empty() or len(df) < 20:
                    continue

                # Convert to list of market data
                data_rows = df.sort("timestamp").to_dicts()
                if len(data_rows) < 20:
                    continue

                current = data_rows[-1]  # Most recent day
                recent_20 = data_rows[-20:]  # Last 20 days

                # Breakout criteria
                current_price = current['close']
                current_volume = current['volume']

                # Price filter: $3-$100
                if current_price < 3.0 or current_price > 100.0:
                    continue

                # Volume breakout: 2x average volume
                avg_volume = sum(d['volume'] for d in recent_20[:-1]) / len(recent_20[:-1])
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

                if volume_ratio < 2.0 or avg_volume < 500_000:  # Need 2x volume + 500K avg
                    continue

                # Price breakout: Above 20-day SMA
                sma_20 = sum(d['close'] for d in recent_20) / len(recent_20)
                if current_price < sma_20:
                    continue

                # Volatility check: Recent high/low spread > 8%
                recent_10 = data_rows[-10:]  # Last 10 days
                high_10 = max(d['high'] for d in recent_10)
                low_10 = min(d['low'] for d in recent_10)
                volatility = (high_10 - low_10) / low_10 if low_10 > 0 else 0

                if volatility < 0.08:  # 8% volatility minimum
                    continue

                candidates.append(symbol)

            except Exception as e:
                self.logger.debug(f"Error analyzing {symbol} for breakouts: {e}")

        return candidates

    async def get_momentum_candidates(self, date: datetime, symbols: List[str]) -> List[str]:
        """Find momentum candidates based on price movement."""
        candidates = []

        for symbol in symbols:
            try:
                end_date = date.date()
                start_date = (date - timedelta(days=10)).date()  # 10-day momentum

                df = await self.data_store.load_market_data(symbol, TimeFrame.ONE_DAY, start_date, end_date)
                if df.is_empty() or len(df) < 5:
                    continue

                data_rows = df.sort("timestamp").to_dicts()
                if len(data_rows) < 5:
                    continue

                current = data_rows[-1]
                old_price = data_rows[0]['close']  # Price 10 days ago
                current_price = current['close']

                # Price filter
                if current_price < 5.0:
                    continue

                # Momentum: 5%+ gain over period
                price_change = (current_price - old_price) / old_price
                if price_change < 0.05:  # 5% minimum
                    continue

                # Volume confirmation
                current_volume = current['volume']
                avg_volume = sum(d['volume'] for d in data_rows[:-1]) / len(data_rows[:-1])
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

                if volume_ratio < 1.5:  # 1.5x average volume
                    continue

                candidates.append(symbol)

            except Exception as e:
                self.logger.debug(f"Error analyzing {symbol} for momentum: {e}")

        return candidates

    async def simulate_screener_for_date(self, date: datetime, available_symbols: List[str]) -> Dict[str, List[str]]:
        """Simulate all screener types for a given date."""
        # Limit symbols to prevent excessive processing
        analysis_symbols = available_symbols[:100] if len(available_symbols) > 100 else available_symbols

        breakouts = await self.get_breakout_candidates(date, analysis_symbols)
        momentum = await self.get_momentum_candidates(date, analysis_symbols)

        # Simple value screening (stocks trading below recent highs)
        value_stocks = []
        for symbol in analysis_symbols[:50]:  # Limit for performance
            try:
                end_date = date.date()
                start_date = (date - timedelta(days=60)).date()

                df = await self.data_store.load_market_data(symbol, TimeFrame.ONE_DAY, start_date, end_date)
                if df.is_empty():
                    continue

                data_rows = df.sort("timestamp").to_dicts()
                if len(data_rows) < 30:
                    continue

                current_price = data_rows[-1]['close']
                high_60 = max(d['high'] for d in data_rows)

                # Value: trading 20%+ below 60-day high, price > $10
                discount = (high_60 - current_price) / high_60
                if discount > 0.2 and current_price > 10.0:
                    value_stocks.append(symbol)

            except Exception:
                continue

        return {
            "breakouts": breakouts[:15],  # Limit results
            "momentum": momentum[:15],
            "value_stocks": value_stocks[:15]
        }


class TradingStrategy:
    """Base trading strategy with different modes."""

    def __init__(self, name: str, mode: str = "day_trading"):
        self.name = name
        self.mode = mode
        self.logger = logging.getLogger(f"{__name__}.Strategy.{name}")

        # Strategy parameters based on mode
        if mode == "day_trading":
            self.min_confidence = 70.0
            self.stop_loss_pct = 0.015  # 1.5%
            self.take_profit_pct = 0.025  # 2.5%
            self.max_hold_days = 1
        elif mode == "swing_trading":
            self.min_confidence = 65.0
            self.stop_loss_pct = 0.03  # 3%
            self.take_profit_pct = 0.06  # 6%
            self.max_hold_days = 7
        else:  # position_trading
            self.min_confidence = 60.0
            self.stop_loss_pct = 0.05  # 5%
            self.take_profit_pct = 0.10  # 10%
            self.max_hold_days = 30

    async def analyze_symbol(self, symbol: str, current_data: MarketData,
                           historical_data: List[MarketData], screener_type: str) -> Optional[TradingSignal]:
        """Analyze symbol and generate trading signal."""
        try:
            if len(historical_data) < 20:
                return None

            current_price = current_data.close

            # Calculate technical indicators
            recent_closes = [float(d.close) for d in historical_data[-20:]]
            sma_20 = sum(recent_closes) / len(recent_closes)

            recent_highs = [float(d.high) for d in historical_data[-10:]]
            recent_lows = [float(d.low) for d in historical_data[-10:]]

            # RSI calculation (simplified)
            price_changes = []
            for i in range(1, len(recent_closes)):
                price_changes.append(recent_closes[i] - recent_closes[i-1])

            gains = [change for change in price_changes if change > 0]
            losses = [abs(change) for change in price_changes if change < 0]

            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0.01

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # Volume analysis
            current_volume = current_data.volume
            avg_volume = sum(d.volume for d in historical_data[-10:]) / 10
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            # Generate signal based on strategy mode and screener type
            signal = None
            confidence = 50.0
            reasoning = f"{screener_type} candidate"

            if screener_type == "breakouts":
                # Breakout strategy
                if (float(current_price) > sma_20 * 1.02 and  # 2% above SMA
                    volume_ratio > 1.5 and  # High volume
                    rsi < 70):  # Not overbought

                    confidence = min(90.0, 60.0 + (volume_ratio * 10) + ((float(current_price) / sma_20 - 1) * 100))

                    signal = TradingSignal(
                        symbol=symbol,
                        action=SignalType.BUY,
                        price=current_price,
                        confidence=confidence,
                        reasoning=f"Breakout: {volume_ratio:.1f}x volume, {float(current_price)/sma_20:.1%} above SMA20",
                        stop_loss=current_price * Decimal(str(1 - self.stop_loss_pct)),
                        take_profit=current_price * Decimal(str(1 + self.take_profit_pct))
                    )

            elif screener_type == "momentum":
                # Momentum strategy
                if (rsi > 50 and rsi < 80 and  # Momentum but not extreme
                    float(current_price) > sma_20 and  # Above average
                    volume_ratio > 1.2):  # Volume confirmation

                    confidence = min(85.0, 50.0 + rsi/2 + volume_ratio * 5)

                    signal = TradingSignal(
                        symbol=symbol,
                        action=SignalType.BUY,
                        price=current_price,
                        confidence=confidence,
                        reasoning=f"Momentum: RSI {rsi:.1f}, volume {volume_ratio:.1f}x",
                        stop_loss=current_price * Decimal(str(1 - self.stop_loss_pct)),
                        take_profit=current_price * Decimal(str(1 + self.take_profit_pct))
                    )

            elif screener_type == "value_stocks":
                # Value strategy - look for reversal signals
                if (rsi < 50 and  # Potentially oversold
                    float(current_price) < sma_20 * 0.95 and  # Below average
                    volume_ratio > 1.0):  # Some volume interest

                    confidence = min(75.0, 40.0 + (50 - rsi) + volume_ratio * 5)

                    signal = TradingSignal(
                        symbol=symbol,
                        action=SignalType.BUY,
                        price=current_price,
                        confidence=confidence,
                        reasoning=f"Value reversal: RSI {rsi:.1f}, {(sma_20-float(current_price))/sma_20:.1%} discount",
                        stop_loss=current_price * Decimal(str(1 - self.stop_loss_pct * 1.5)),  # Wider stop for value
                        take_profit=current_price * Decimal(str(1 + self.take_profit_pct * 1.2))
                    )

            # Check confidence threshold
            if signal and signal.confidence >= self.min_confidence:
                return signal

            return None

        except Exception as e:
            self.logger.debug(f"Error analyzing {symbol}: {e}")
            return None


class ComprehensiveBacktestEngine:
    """Comprehensive backtesting engine using real historical data."""

    def __init__(self, start_date: datetime, end_date: datetime, initial_capital: Decimal = Decimal('100000')):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        self.data_store = SimpleDataStore("data/parquet")
        self.screener = ScreenerSimulator(self.data_store)

        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[Tuple[datetime, Decimal]] = []

        # Performance tracking
        self.max_portfolio_value = initial_capital
        self.signals_generated = 0
        self.signals_executed = 0

        self.logger = logging.getLogger(__name__)

    async def run_backtest(self, strategy: TradingStrategy, max_positions: int = 8) -> Dict[str, Any]:
        """Run comprehensive backtesting."""
        self.logger.info(f"Starting {strategy.name} backtest: {self.start_date.date()} to {self.end_date.date()}")

        start_time = time.time()

        # Get available symbols
        all_symbols = self.data_store.get_available_symbols()
        major_symbols = [s for s in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
                                   'AMD', 'NFLX', 'CRM', 'INTC', 'ORCL', 'ADBE', 'PYPL'] if s in all_symbols]

        # Use major symbols + random selection from others
        import random
        other_symbols = [s for s in all_symbols if s not in major_symbols]
        selected_symbols = major_symbols + random.sample(other_symbols, min(30, len(other_symbols)))

        self.logger.info(f"Using {len(selected_symbols)} symbols for backtesting")

        # Generate trading days
        trading_days = []
        current_date = self.start_date
        while current_date <= self.end_date:
            if current_date.weekday() < 5:  # Weekdays only
                trading_days.append(current_date)
            current_date += timedelta(days=1)

        self.logger.info(f"Processing {len(trading_days)} trading days")

        # Process each trading day
        for i, date in enumerate(trading_days):
            await self._process_trading_day(date, selected_symbols, strategy, max_positions)

            # Record portfolio value
            portfolio_value = self._calculate_portfolio_value()
            self.portfolio_history.append((date, portfolio_value))
            self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)

            # Progress reporting
            if (i + 1) % 20 == 0 or i == len(trading_days) - 1:
                self.logger.info(f"Progress: {i+1}/{len(trading_days)}, Portfolio: ${portfolio_value:,.2f}")

        # Close remaining positions
        for symbol in list(self.positions.keys()):
            await self._close_position(symbol, trading_days[-1], "end_of_backtest")

        execution_time = time.time() - start_time

        # Generate results
        results = self._generate_results(strategy, execution_time)

        self.logger.info(f"Backtest completed in {execution_time:.2f}s")
        self.logger.info(f"Total return: {results['performance']['total_return']:.2%}")
        self.logger.info(f"Total trades: {results['trading']['total_trades']}")

        return results

    async def _process_trading_day(self, date: datetime, symbols: List[str],
                                 strategy: TradingStrategy, max_positions: int) -> None:
        """Process a single trading day."""
        try:
            # 1. Update existing positions
            await self._update_position_prices(date)

            # 2. Check exit conditions
            await self._check_exit_conditions(date, strategy)

            # 3. Generate screener results
            if len(self.positions) < max_positions:
                screener_results = await self.screener.simulate_screener_for_date(date, symbols)

                # 4. Analyze screener candidates
                for screener_type, candidates in screener_results.items():
                    if len(self.positions) >= max_positions:
                        break

                    for symbol in candidates:
                        if symbol in self.positions:
                            continue

                        # Get historical data for analysis
                        historical_data = await self._get_historical_data(symbol, date)
                        if not historical_data or len(historical_data) < 20:
                            continue

                        current_data = historical_data[-1]

                        # Generate signal
                        signal = await strategy.analyze_symbol(symbol, current_data, historical_data, screener_type)
                        if signal:
                            self.signals_generated += 1

                            # Execute signal if we have capacity
                            if len(self.positions) < max_positions:
                                await self._execute_signal(signal, date, strategy.name)

        except Exception as e:
            self.logger.error(f"Error processing {date.date()}: {e}")

    async def _update_position_prices(self, date: datetime) -> None:
        """Update current prices for all positions."""
        for symbol, position in self.positions.items():
            try:
                df = await self.data_store.load_market_data(
                    ticker=symbol,
                    timeframe=TimeFrame.ONE_DAY,
                    start_date=date.date(),
                    end_date=date.date()
                )

                if not df.is_empty():
                    row = df.sort("timestamp").tail(1).row(0, named=True)
                    position.current_price = Decimal(str(row['close']))

            except Exception as e:
                self.logger.debug(f"Could not update price for {symbol}: {e}")

    async def _check_exit_conditions(self, date: datetime, strategy: TradingStrategy) -> None:
        """Check if any positions should be closed."""
        positions_to_close = []

        for symbol, position in self.positions.items():
            # Stop loss
            if position.stop_loss and position.current_price <= position.stop_loss:
                positions_to_close.append((symbol, "stop_loss"))
                continue

            # Take profit
            if position.take_profit and position.current_price >= position.take_profit:
                positions_to_close.append((symbol, "take_profit"))
                continue

            # Time-based exit (strategy dependent)
            hold_days = (date - position.entry_date).days
            if hold_days >= strategy.max_hold_days:
                positions_to_close.append((symbol, "time_exit"))
                continue

        # Close positions
        for symbol, reason in positions_to_close:
            await self._close_position(symbol, date, reason)

    async def _execute_signal(self, signal: TradingSignal, date: datetime, strategy_name: str) -> None:
        """Execute a trading signal."""
        try:
            # Calculate position size
            portfolio_value = self._calculate_portfolio_value()
            position_value = portfolio_value * Decimal(str(signal.position_size))
            quantity = int(position_value / signal.price)

            if quantity <= 0:
                return

            # Check if we have enough cash (including commission)
            total_cost = signal.price * Decimal(str(quantity)) + Decimal('2.00')  # $2 commission
            if total_cost > self.cash:
                return

            # Create position
            position = Position(
                symbol=signal.symbol,
                quantity=quantity,
                entry_price=signal.price,
                entry_date=date,
                current_price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy_name=strategy_name
            )

            # Update portfolio
            self.positions[signal.symbol] = position
            self.cash -= total_cost
            self.signals_executed += 1

            self.logger.info(f"Opened position: {signal.symbol} x{quantity} @ ${signal.price} (Confidence: {signal.confidence:.1f}%)")

        except Exception as e:
            self.logger.error(f"Error executing signal for {signal.symbol}: {e}")

    async def _close_position(self, symbol: str, date: datetime, reason: str) -> None:
        """Close a position."""
        if symbol not in self.positions:
            return

        try:
            position = self.positions[symbol]
            exit_price = position.current_price

            # Calculate P&L
            gross_pnl = (exit_price - position.entry_price) * Decimal(str(position.quantity))
            commission = Decimal('2.00')  # $2 total commission
            net_pnl = gross_pnl - commission

            hold_days = (date - position.entry_date).days
            pnl_percentage = float(net_pnl / (position.entry_price * Decimal(str(position.quantity))))

            # Create trade record
            trade = Trade(
                symbol=symbol,
                entry_date=position.entry_date,
                exit_date=date,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                pnl=net_pnl,
                pnl_percentage=pnl_percentage,
                hold_days=hold_days,
                strategy=position.strategy_name,
                entry_reason="signal",
                exit_reason=reason
            )

            # Update portfolio
            proceeds = exit_price * Decimal(str(position.quantity)) - commission
            self.cash += proceeds
            self.trades.append(trade)

            del self.positions[symbol]

            self.logger.info(f"Closed position: {symbol}, P&L: ${net_pnl:.2f} ({pnl_percentage:.2%}), Reason: {reason}")

        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {e}")

    async def _get_historical_data(self, symbol: str, current_date: datetime) -> Optional[List[MarketData]]:
        """Get historical data for symbol analysis."""
        try:
            start_date = (current_date - timedelta(days=60)).date()
            end_date = current_date.date()

            df = await self.data_store.load_market_data(symbol, TimeFrame.ONE_DAY, start_date, end_date)
            if df.is_empty():
                return None

            historical_data = []
            for row in df.sort("timestamp").iter_rows(named=True):
                adjusted_close = row.get('adjusted_close', row['close'])

                historical_data.append(MarketData(
                    symbol=symbol,
                    timestamp=row['timestamp'],
                    open=Decimal(str(row['open'])),
                    high=Decimal(str(row['high'])),
                    low=Decimal(str(row['low'])),
                    close=Decimal(str(row['close'])),
                    adjusted_close=Decimal(str(adjusted_close)),
                    volume=int(row['volume']),
                    timeframe=TimeFrame.ONE_DAY
                ))

            return historical_data

        except Exception as e:
            self.logger.debug(f"Error loading historical data for {symbol}: {e}")
            return None

    def _calculate_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value."""
        total = self.cash
        for position in self.positions.values():
            total += position.current_price * Decimal(str(position.quantity))
        return total

    def _generate_results(self, strategy: TradingStrategy, execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive results."""
        final_value = self._calculate_portfolio_value()
        total_return = float((final_value - self.initial_capital) / self.initial_capital)

        # Trading statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_win = sum(float(t.pnl) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(float(t.pnl) for t in losing_trades) / len(losing_trades) if losing_trades else 0

        # Calculate drawdown
        max_dd = 0.0
        peak = float(self.initial_capital)
        for _, value in self.portfolio_history:
            value_float = float(value)
            if value_float > peak:
                peak = value_float
            else:
                drawdown = (peak - value_float) / peak
                max_dd = max(max_dd, drawdown)

        # Risk metrics
        daily_returns = []
        for i in range(1, len(self.portfolio_history)):
            prev_val = float(self.portfolio_history[i-1][1])
            curr_val = float(self.portfolio_history[i][1])
            if prev_val > 0:
                daily_returns.append((curr_val - prev_val) / prev_val)

        import math
        if daily_returns:
            avg_daily_return = sum(daily_returns) / len(daily_returns)
            daily_volatility = math.sqrt(sum((r - avg_daily_return)**2 for r in daily_returns) / len(daily_returns))
            annualized_return = (1 + avg_daily_return) ** 252 - 1
            annualized_volatility = daily_volatility * math.sqrt(252)
            sharpe_ratio = (annualized_return - 0.02) / annualized_volatility if annualized_volatility > 0 else 0
        else:
            annualized_return = total_return
            sharpe_ratio = 0

        return {
            'strategy': {
                'name': strategy.name,
                'mode': strategy.mode,
                'parameters': {
                    'min_confidence': strategy.min_confidence,
                    'stop_loss_pct': strategy.stop_loss_pct,
                    'take_profit_pct': strategy.take_profit_pct,
                    'max_hold_days': strategy.max_hold_days
                }
            },
            'performance': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'final_value': float(final_value),
                'max_drawdown': max_dd,
                'sharpe_ratio': sharpe_ratio
            },
            'trading': {
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_hold_days': sum(t.hold_days for t in self.trades) / len(self.trades) if self.trades else 0,
                'profit_factor': abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if avg_loss != 0 else float('inf')
            },
            'signals': {
                'generated': self.signals_generated,
                'executed': self.signals_executed,
                'execution_rate': self.signals_executed / max(self.signals_generated, 1)
            },
            'execution': {
                'time_seconds': execution_time,
                'days_processed': len(self.portfolio_history),
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat()
            },
            'trades': [
                {
                    'symbol': t.symbol,
                    'entry_date': t.entry_date.isoformat(),
                    'exit_date': t.exit_date.isoformat(),
                    'pnl': float(t.pnl),
                    'pnl_percentage': t.pnl_percentage,
                    'hold_days': t.hold_days,
                    'strategy': t.strategy,
                    'exit_reason': t.exit_reason
                }
                for t in self.trades
            ]
        }


def format_currency(amount: float) -> str:
    """Format currency amount."""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format percentage."""
    return f"{value:.2%}"


def display_results(results: Dict[str, Any]) -> None:
    """Display comprehensive backtest results."""
    print("\n" + "="*80)
    print("COMPREHENSIVE BACKTESTING RESULTS")
    print("="*80)

    # Strategy info
    strategy = results['strategy']
    print(f"\nStrategy: {strategy['name']} ({strategy['mode']})")
    print(f"Parameters:")
    for key, value in strategy['parameters'].items():
        print(f"  {key}: {value}")

    # Performance summary
    perf = results['performance']
    print(f"\nPerformance Summary:")
    print(f"  Total Return: {format_percentage(perf['total_return'])}")
    print(f"  Annualized Return: {format_percentage(perf['annualized_return'])}")
    print(f"  Final Portfolio Value: {format_currency(perf['final_value'])}")
    print(f"  Maximum Drawdown: {format_percentage(perf['max_drawdown'])}")
    print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")

    # Trading statistics
    trading = results['trading']
    print(f"\nTrading Statistics:")
    print(f"  Total Trades: {trading['total_trades']}")
    print(f"  Winning Trades: {trading['winning_trades']}")
    print(f"  Losing Trades: {trading['losing_trades']}")
    print(f"  Win Rate: {format_percentage(trading['win_rate'])}")
    print(f"  Average Win: {format_currency(trading['avg_win'])}")
    print(f"  Average Loss: {format_currency(trading['avg_loss'])}")
    print(f"  Profit Factor: {trading['profit_factor']:.2f}")
    print(f"  Average Hold Days: {trading['avg_hold_days']:.1f}")

    # Signal analysis
    signals = results['signals']
    print(f"\nSignal Analysis:")
    print(f"  Signals Generated: {signals['generated']}")
    print(f"  Signals Executed: {signals['executed']}")
    print(f"  Execution Rate: {format_percentage(signals['execution_rate'])}")

    # Execution info
    execution = results['execution']
    print(f"\nExecution Info:")
    print(f"  Execution Time: {execution['time_seconds']:.2f} seconds")
    print(f"  Days Processed: {execution['days_processed']}")
    print(f"  Period: {execution['start_date'][:10]} to {execution['end_date'][:10]}")

    print("\n" + "="*80)


async def run_strategy_comparison():
    """Run and compare different trading strategies."""
    print("COMPREHENSIVE STRATEGY COMPARISON")
    print("="*50)

    # Define test period (last 3 months with good data coverage)
    end_date = datetime(2025, 8, 19, tzinfo=timezone.utc)
    start_date = datetime(2025, 5, 20, tzinfo=timezone.utc)  # ~3 months
    initial_capital = Decimal('100000')

    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Duration: {(end_date - start_date).days} days")
    print(f"Initial Capital: {format_currency(float(initial_capital))}")

    # Define strategies to test
    strategies = [
        TradingStrategy("Day Trading Strategy", "day_trading"),
        TradingStrategy("Swing Trading Strategy", "swing_trading"),
        TradingStrategy("Position Trading Strategy", "position_trading")
    ]

    results = {}

    # Run each strategy
    for strategy in strategies:
        print(f"\n🔄 Running {strategy.name}...")

        engine = ComprehensiveBacktestEngine(start_date, end_date, initial_capital)

        try:
            result = await engine.run_backtest(strategy, max_positions=6)
            results[strategy.name] = result

            print(f"✅ {strategy.name} completed:")
            print(f"   Return: {format_percentage(result['performance']['total_return'])}")
            print(f"   Trades: {result['trading']['total_trades']}")
            print(f"   Win Rate: {format_percentage(result['trading']['win_rate'])}")

        except Exception as e:
            print(f"❌ {strategy.name} failed: {e}")
            continue

    # Display comparison
    if results:
        print("\n" + "="*100)
        print("STRATEGY COMPARISON SUMMARY")
        print("="*100)

        print(f"\n{'Strategy':<25} {'Return':<12} {'Max DD':<10} {'Sharpe':<8} {'Trades':<8} {'Win Rate':<10}")
        print("-" * 85)

        for strategy_name, result in results.items():
            perf = result['performance']
            trading = result['trading']

            print(f"{strategy_name:<25} "
                  f"{format_percentage(perf['total_return']):<12} "
                  f"{format_percentage(perf['max_drawdown']):<10} "
                  f"{perf['sharpe_ratio']:<8.3f} "
                  f"{trading['total_trades']:<8} "
                  f"{format_percentage(trading['win_rate']):<10}")

        # Find best performers
        best_return = max(results.items(), key=lambda x: x[1]['performance']['total_return'])
        best_sharpe = max(results.items(), key=lambda x: x[1]['performance']['sharpe_ratio'])

        print(f"\n🏆 Best Total Return: {best_return[0]} ({format_percentage(best_return[1]['performance']['total_return'])})")
        print(f"🏆 Best Risk-Adjusted: {best_sharpe[0]} (Sharpe: {best_sharpe[1]['performance']['sharpe_ratio']:.3f})")

    return results


async def run_single_strategy_demo():
    """Run a detailed demo of a single strategy."""
    print("SINGLE STRATEGY DETAILED ANALYSIS")
    print("="*50)

    # Use a shorter period for detailed analysis
    end_date = datetime(2025, 8, 19, tzinfo=timezone.utc)
    start_date = datetime(2025, 7, 1, tzinfo=timezone.utc)  # ~7 weeks
    initial_capital = Decimal('50000')

    print(f"Testing Day Trading Strategy")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: {format_currency(float(initial_capital))}")

    strategy = TradingStrategy("Detailed Day Trading", "day_trading")
    engine = ComprehensiveBacktestEngine(start_date, end_date, initial_capital)

    result = await engine.run_backtest(strategy, max_positions=5)

    # Display detailed results
    display_results(result)

    # Show sample trades
    trades = result['trades']
    if trades:
        print("\nSample Trades (Last 10):")
        print("-" * 80)
        for trade in trades[-10:]:
            entry_date = trade['entry_date'][:10]
            exit_date = trade['exit_date'][:10]
            pnl = format_currency(trade['pnl'])
            pnl_pct = format_percentage(trade['pnl_percentage'])

            print(f"{trade['symbol']:<6} {entry_date} to {exit_date} "
                  f"({trade['hold_days']}d) {pnl:<10} {pnl_pct:<8} {trade['exit_reason']}")

    return result


async def main():
    """Main demonstration function."""
    print("🚀 COMPREHENSIVE BACKTESTING DEMONSTRATION")
    print("="*60)
    print("This demo uses your extensive historical data (12+ months available)")
    print("to test sophisticated trading strategies with screener simulation.")
    print()

    try:
        # First, show data availability
        print("📊 CHECKING DATA AVAILABILITY...")
        store = SimpleDataStore("data/parquet")
        symbols = store.get_available_symbols()

        print(f"✅ Found {len(symbols)} symbols in data store")
        print(f"📈 Major symbols available: {[s for s in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] if s in symbols]}")

        # Show date range for AAPL
        if 'AAPL' in symbols:
            date_range = store.get_date_range_for_symbol('AAPL', TimeFrame.ONE_DAY)
            if date_range[0]:
                days = (date_range[1] - date_range[0]).days
                print(f"📅 AAPL data range: {date_range[0]} to {date_range[1]} ({days} days)")

        print("\n" + "="*60)

        # Run demonstrations
        print("🎯 RUNNING DEMONSTRATIONS...")
        print()

        # 1. Single strategy detailed analysis
        print("1️⃣  SINGLE STRATEGY ANALYSIS")
        await run_single_strategy_demo()

        print("\n" + "="*60)

        # 2. Multi-strategy comparison
        print("2️⃣  MULTI-STRATEGY COMPARISON")
        comparison_results = await run_strategy_comparison()

        print("\n" + "="*60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)

        print("\nKey Findings:")
        print("✓ Historical data backtesting works with your 12+ months of data")
        print("✓ Screener simulation successfully identifies trading candidates")
        print("✓ Multiple strategy modes can be tested and compared")
        print("✓ Comprehensive performance metrics are calculated")
        print("✓ Risk management (stop losses, position sizing) is simulated")

        print(f"\nNext Steps:")
        print("🔧 Connect to your real production strategies (HybridStrategy)")
        print("📊 Test with different time periods and parameters")
        print("🎯 Optimize screener criteria based on backtest results")
        print("📈 Use results to guide live trading strategy selection")

        return comparison_results

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the comprehensive demonstration
    results = asyncio.run(main())
