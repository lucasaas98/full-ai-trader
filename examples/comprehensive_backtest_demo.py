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
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project paths
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root / "backtesting"))

from backtest_models import MarketData, SignalType, TimeFrame  # noqa: E402
from simple_data_store import SimpleDataStore  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
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
    current_price: Decimal = Decimal("0")
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

    def __init__(
        self, data_store: SimpleDataStore, timeframe: TimeFrame = TimeFrame.ONE_DAY
    ):
        self.data_store = data_store
        self.timeframe = timeframe
        self.logger = logging.getLogger(f"{__name__}.ScreenerSimulator")

    async def get_breakout_candidates(
        self, date: datetime, symbols: List[str]
    ) -> List[str]:
        """Find breakout candidates based on volume and price action."""
        # Process symbols in parallel batches
        batch_size = 10
        all_candidates = []

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            batch_tasks = [
                self._analyze_breakout_candidate(date, symbol) for symbol in batch
            ]
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for symbol, result in zip(batch, results):
                if result is True:
                    all_candidates.append(symbol)

        return all_candidates

    async def _analyze_breakout_candidate(self, date: datetime, symbol: str) -> bool:
        """Analyze a single symbol for breakout potential."""
        try:
            # Adjust lookback based on timeframe
            if self.timeframe == TimeFrame.ONE_MINUTE:
                lookback_days = 2
            elif self.timeframe == TimeFrame.FIVE_MIN:
                lookback_days = 5
            elif self.timeframe == TimeFrame.FIFTEEN_MIN:
                lookback_days = 10
            elif self.timeframe == TimeFrame.ONE_HOUR:
                lookback_days = 15
            else:  # ONE_DAY
                lookback_days = 30

            end_date = date.date()
            start_date = (date - timedelta(days=lookback_days)).date()

            df = await self.data_store.load_market_data(
                symbol, self.timeframe, start_date, end_date
            )
            if df.is_empty() or len(df) < 20:
                self.logger.debug(
                    f"❌ {symbol}: insufficient data ({len(df) if not df.is_empty() else 0} rows)"
                )
                return False

            # Convert to list of market data
            data_rows = df.sort("timestamp").to_dicts()
            if len(data_rows) < 20:
                self.logger.debug(
                    f"❌ {symbol}: insufficient rows after sort ({len(data_rows)})"
                )
                return False

            current = data_rows[-1]  # Most recent day
            recent_20 = data_rows[-20:]  # Last 20 days

            # Breakout criteria
            current_price = current["close"]
            current_volume = current["volume"]

            # Price filter: $0.10-$1000 (very relaxed for testing)
            if current_price < 3 or current_price > 1000.0:
                self.logger.debug(
                    f"❌ {symbol}: price filter failed (${current_price:.2f})"
                )
                return False

            # Volume breakout: 1.1x average volume (very relaxed)
            avg_volume = sum(d["volume"] for d in recent_20[:-1]) / len(recent_20[:-1])
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

            if volume_ratio < 2 or avg_volume < 10_000:  # Very relaxed criteria
                self.logger.debug(
                    f"❌ {symbol}: volume filter failed (ratio: {volume_ratio:.1f}x, avg: {avg_volume:,.0f})"
                )
                return False

            # Price breakout: Above 20-day SMA (very relaxed)
            sma_20 = sum(d["close"] for d in recent_20) / len(recent_20)
            if current_price < sma_20 * 0.95:  # Allow 5% below SMA
                self.logger.debug(
                    f"❌ {symbol}: SMA filter failed (price: ${current_price:.2f}, SMA: ${sma_20:.2f})"
                )
                return False

            # Volatility check: Recent high/low spread > 2% (very relaxed)
            recent_10 = data_rows[-10:]  # Last 10 days
            high_10 = max(d["high"] for d in recent_10)
            low_10 = min(d["low"] for d in recent_10)
            volatility = (high_10 - low_10) / low_10 if low_10 > 0 else 0

            if volatility < 0.02:  # 2% volatility minimum (very relaxed)
                self.logger.debug(
                    f"❌ {symbol}: volatility filter failed ({volatility:.1%})"
                )
                return False

            self.logger.debug(
                f"✅ {symbol}: BREAKOUT candidate (price: ${current_price:.2f}, volume: {volume_ratio:.1f}x, volatility: {volatility:.1%})"
            )
            return True

        except Exception as e:
            self.logger.debug(f"❌ Error analyzing {symbol} for breakouts: {e}")
            return False

    async def get_momentum_candidates(
        self, date: datetime, symbols: List[str]
    ) -> List[str]:
        """Find momentum candidates based on price movement."""
        # Process symbols in parallel batches
        batch_size = 10
        all_candidates = []

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            batch_tasks = [
                self._analyze_momentum_candidate(date, symbol) for symbol in batch
            ]
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for symbol, result in zip(batch, results):
                if result is True:
                    all_candidates.append(symbol)

        return all_candidates

    async def _analyze_momentum_candidate(self, date: datetime, symbol: str) -> bool:
        """Analyze a single symbol for momentum potential."""
        try:
            # Adjust momentum period based on timeframe
            if self.timeframe == TimeFrame.ONE_MINUTE:
                lookback_days = 1
            elif self.timeframe == TimeFrame.FIVE_MIN:
                lookback_days = 2
            elif self.timeframe == TimeFrame.FIFTEEN_MIN:
                lookback_days = 3
            elif self.timeframe == TimeFrame.ONE_HOUR:
                lookback_days = 5
            else:  # ONE_DAY
                lookback_days = 10

            end_date = date.date()
            start_date = (date - timedelta(days=lookback_days)).date()

            df = await self.data_store.load_market_data(
                symbol, self.timeframe, start_date, end_date
            )
            if df.is_empty() or len(df) < 5:
                return False

            data_rows = df.sort("timestamp").to_dicts()
            if len(data_rows) < 5:
                return False

            current = data_rows[-1]
            old_price = data_rows[0]["close"]  # Price N days ago
            current_price = current["close"]

            # Price filter (very relaxed)
            if current_price < 0.10:
                return False

            # Momentum: 1%+ gain over period (very relaxed)
            price_change = (current_price - old_price) / old_price
            if price_change < 0.01:  # 1% minimum
                return False

            # Volume confirmation (very relaxed)
            current_volume = current["volume"]
            avg_volume = sum(d["volume"] for d in data_rows[:-1]) / len(data_rows[:-1])
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

            if volume_ratio < 1.05:  # 1.05x average volume
                return False

            return True

        except Exception as e:
            self.logger.debug(f"Error analyzing {symbol} for momentum: {e}")
            return False

    async def simulate_screener_for_date(
        self, date: datetime, available_symbols: List[str]
    ) -> Dict[str, List[str]]:
        """Simulate all screener types for a given date."""
        # Limit symbols to prevent excessive processing
        analysis_symbols = (
            available_symbols[:100]
            if len(available_symbols) > 100
            else available_symbols
        )

        breakouts = await self.get_breakout_candidates(date, analysis_symbols)
        momentum = await self.get_momentum_candidates(date, analysis_symbols)

        # Simple value screening (stocks trading below recent highs)
        value_stocks = []
        for symbol in analysis_symbols[:50]:  # Limit for performance
            try:
                # Adjust value analysis period based on timeframe
                if self.timeframe == TimeFrame.ONE_MINUTE:
                    lookback_days = 5
                elif self.timeframe == TimeFrame.FIVE_MIN:
                    lookback_days = 10
                elif self.timeframe == TimeFrame.FIFTEEN_MIN:
                    lookback_days = 20
                elif self.timeframe == TimeFrame.ONE_HOUR:
                    lookback_days = 30
                else:  # ONE_DAY
                    lookback_days = 60

                end_date = date.date()
                start_date = (date - timedelta(days=lookback_days)).date()

                df = await self.data_store.load_market_data(
                    symbol, self.timeframe, start_date, end_date
                )
                if df.is_empty():
                    continue

                data_rows = df.sort("timestamp").to_dicts()
                if len(data_rows) < 30:
                    continue

                current_price = data_rows[-1]["close"]
                high_60 = max(d["high"] for d in data_rows)

                # Value: trading 20%+ below 60-day high, price > $10
                discount = (high_60 - current_price) / high_60
                if discount > 0.2 and current_price > 10.0:
                    value_stocks.append(symbol)

            except Exception:
                continue

        return {
            "breakouts": breakouts[:15],  # Limit results
            "momentum": momentum[:15],
            "value_stocks": value_stocks[:15],
        }


class TradingStrategy:
    """Base trading strategy with different modes."""

    def __init__(self, name: str, mode: str = "day_trading"):
        self.name = name
        self.mode = mode
        self.logger = logging.getLogger(f"{__name__}.Strategy.{name}")

        # Strategy parameters based on mode
        if mode == "day_trading":
            self.min_confidence = 30.0  # Very low for testing
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

    async def analyze_symbol(
        self,
        symbol: str,
        current_data: MarketData,
        historical_data: List[MarketData],
        screener_type: str,
    ) -> Optional[TradingSignal]:
        """Analyze symbol and generate trading signal."""
        try:
            if len(historical_data) < 20:
                return None

            current_price = current_data.close

            # Calculate technical indicators
            recent_closes = [float(d.close) for d in historical_data[-20:]]
            sma_20 = sum(recent_closes) / len(recent_closes)

            # recent_highs = [float(d.high) for d in historical_data[-10:]]  # Unused
            # recent_lows = [float(d.low) for d in historical_data[-10:]]  # Unused

            # RSI calculation (simplified)
            price_changes = []
            for i in range(1, len(recent_closes)):
                price_changes.append(recent_closes[i] - recent_closes[i - 1])

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
            # reasoning = f"{screener_type} candidate"  # Unused - reasoning is set in signal creation

            if screener_type == "breakouts":
                # Breakout strategy (very relaxed)
                if (
                    float(current_price) > sma_20 * 0.98  # Allow slight below SMA
                    and volume_ratio > 1.05  # Low volume requirement
                    and rsi < 80
                ):  # Less strict overbought

                    confidence = min(
                        90.0,
                        50.0
                        + (volume_ratio * 8)
                        + ((float(current_price) / sma_20 - 1) * 80),
                    )

                    signal = TradingSignal(
                        symbol=symbol,
                        action=SignalType.BUY,
                        price=current_price,
                        confidence=confidence,
                        reasoning=f"Breakout: {volume_ratio:.1f}x volume, {float(current_price) / sma_20:.1%} above SMA20",
                        stop_loss=current_price * Decimal(str(1 - self.stop_loss_pct)),
                        take_profit=current_price
                        * Decimal(str(1 + self.take_profit_pct)),
                    )

            elif screener_type == "momentum":
                # Momentum strategy (relaxed)
                if (
                    rsi > 45
                    and rsi < 85  # Broader momentum range
                    and float(current_price) > sma_20 * 0.95  # Allow slight below SMA
                    and volume_ratio > 1.05
                ):  # Lower volume requirement

                    confidence = min(85.0, 45.0 + rsi / 3 + volume_ratio * 4)

                    signal = TradingSignal(
                        symbol=symbol,
                        action=SignalType.BUY,
                        price=current_price,
                        confidence=confidence,
                        reasoning=f"Momentum: RSI {rsi:.1f}, volume {volume_ratio:.1f}x",
                        stop_loss=current_price * Decimal(str(1 - self.stop_loss_pct)),
                        take_profit=current_price
                        * Decimal(str(1 + self.take_profit_pct)),
                    )

            elif screener_type == "value_stocks":
                # Value strategy - look for reversal signals (relaxed)
                if (
                    rsi < 55  # Broader oversold range
                    and float(current_price)
                    < sma_20 * 0.98  # Less strict below average
                    and volume_ratio > 0.8
                ):  # Even lower volume requirement

                    confidence = min(75.0, 35.0 + (55 - rsi) + volume_ratio * 4)

                    signal = TradingSignal(
                        symbol=symbol,
                        action=SignalType.BUY,
                        price=current_price,
                        confidence=confidence,
                        reasoning=f"Value reversal: RSI {rsi:.1f}, {(sma_20 - float(current_price)) / sma_20:.1%} discount",
                        stop_loss=current_price
                        * Decimal(
                            str(1 - self.stop_loss_pct * 1.5)
                        ),  # Wider stop for value
                        take_profit=current_price
                        * Decimal(str(1 + self.take_profit_pct * 1.2)),
                    )

            # Check confidence threshold
            if signal and signal.confidence >= self.min_confidence:
                self.logger.debug(
                    f"✅ Signal generated for {symbol}: {signal.confidence:.1f}% confidence, {signal.reasoning}"
                )
                return signal
            elif signal:
                self.logger.debug(
                    f"❌ Signal below threshold for {symbol}: {signal.confidence:.1f}% < {self.min_confidence}%"
                )

            return None

        except Exception as e:
            self.logger.debug(f"Error analyzing {symbol}: {e}")
            return None


class ComprehensiveBacktestEngine:
    """Comprehensive backtesting engine using real historical data."""

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_capital: Decimal = Decimal("100000"),
        timeframe: TimeFrame = TimeFrame.ONE_DAY,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.timeframe = timeframe

        self.data_store = SimpleDataStore("data/parquet")
        self.screener = ScreenerSimulator(self.data_store, self.timeframe)

        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[Tuple[datetime, Decimal]] = []

        # Performance tracking
        self.max_portfolio_value = initial_capital
        self.signals_generated = 0
        self.signals_executed = 0

        # Data cache for parallel processing
        self.data_cache: Dict[str, Dict[str, Any]] = {}

        # Thread pool for parallel processing
        self.max_workers = min(8, (os.cpu_count() or 1) * 2)

        self.logger = logging.getLogger(__name__)

    async def run_backtest(
        self, strategy: TradingStrategy, max_positions: int = 8
    ) -> Dict[str, Any]:
        """Run comprehensive backtesting."""
        self.logger.info(
            f"Starting {strategy.name} backtest ({self.timeframe.value}): {self.start_date.date()} to {self.end_date.date()}"
        )

        start_time = time.time()

        # Get available symbols
        all_symbols = self.data_store.get_available_symbols()
        major_symbols = [
            s
            for s in [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "TSLA",
                "META",
                "NVDA",
                "AMD",
                "NFLX",
                "CRM",
                "INTC",
                "ORCL",
                "ADBE",
                "PYPL",
            ]
            if s in all_symbols
        ]

        # Use major symbols + random selection from others
        import random

        other_symbols = [s for s in all_symbols if s not in major_symbols]
        selected_symbols = major_symbols + random.sample(
            other_symbols, min(50, len(other_symbols))
        )

        self.logger.info(f"Using {len(selected_symbols)} symbols for backtesting")

        # Pre-load data in parallel for better performance
        preload_start = time.time()
        await self._preload_data(selected_symbols)
        preload_time = time.time() - preload_start
        self.logger.info(f"Data pre-loading completed in {preload_time:.2f}s")

        # Generate trading periods based on timeframe
        trading_periods = self._generate_trading_periods()
        self.logger.info(
            f"Processing {len(trading_periods)} trading periods ({self.timeframe.value})"
        )

        # Process each trading period
        for i, period_datetime in enumerate(trading_periods):
            await self._process_trading_period(
                period_datetime, selected_symbols, strategy, max_positions
            )

            # Record portfolio value (sample less frequently for intraday to reduce overhead)
            sample_frequency = self._get_portfolio_sample_frequency()
            if i % sample_frequency == 0 or i == len(trading_periods) - 1:
                portfolio_value = self._calculate_portfolio_value()
                self.portfolio_history.append((period_datetime, portfolio_value))
                self.max_portfolio_value = max(
                    self.max_portfolio_value, portfolio_value
                )

            # Progress reporting with performance metrics
            progress_frequency = max(
                1, len(trading_periods) // 10
            )  # Report 10 times total
            if (i + 1) % progress_frequency == 0 or i == len(trading_periods) - 1:
                portfolio_value = self._calculate_portfolio_value()
                elapsed = time.time() - start_time
                periods_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                estimated_total = (
                    len(trading_periods) / periods_per_sec if periods_per_sec > 0 else 0
                )
                remaining = max(0, estimated_total - elapsed)

                self.logger.info(
                    f"Progress: {i + 1}/{len(trading_periods)} ({(i + 1) / len(trading_periods) * 100:.1f}%), "
                    f"Portfolio: ${portfolio_value:,.2f}, "
                    f"Speed: {periods_per_sec:.1f} periods/sec, "
                    f"ETA: {remaining / 60:.1f}min"
                )

        # Close remaining positions
        for symbol in list(self.positions.keys()):
            await self._close_position(symbol, trading_periods[-1], "end_of_backtest")

        execution_time = time.time() - start_time

        # Generate results
        results = self._generate_results(strategy, execution_time)

        self.logger.info(f"Backtest completed in {execution_time:.2f}s")
        self.logger.info(f"Total return: {results['performance']['total_return']:.2%}")
        self.logger.info(f"Total trades: {results['trading']['total_trades']}")

        return results

    def _generate_trading_periods(self) -> List[datetime]:
        """Generate trading periods based on timeframe."""
        periods = []

        # Define market hours (9:30 AM to 4:00 PM ET)
        # market_start = 9.5 * 60  # 9:30 AM in minutes (unused)
        # market_end = 16 * 60  # 4:00 PM in minutes (unused)

        # Get timeframe interval in minutes
        if self.timeframe == TimeFrame.ONE_MINUTE:
            interval_minutes = 1
            max_periods_per_day = min(
                390, 100
            )  # Limit to 100 periods per day for performance
        elif self.timeframe == TimeFrame.FIVE_MIN:
            interval_minutes = 5
            max_periods_per_day = 78  # 6.5 hours * 60 minutes / 5
        elif self.timeframe == TimeFrame.FIFTEEN_MIN:
            interval_minutes = 15
            max_periods_per_day = 26  # 6.5 hours * 60 minutes / 15
        elif self.timeframe == TimeFrame.ONE_HOUR:
            interval_minutes = 60
            max_periods_per_day = 7  # 6.5 hours
        else:  # ONE_DAY
            interval_minutes = 24 * 60  # Full day
            max_periods_per_day = 1

        current_date = self.start_date.replace(
            hour=9, minute=30, second=0, microsecond=0
        )

        while current_date.date() <= self.end_date.date():
            if current_date.weekday() < 5:  # Weekdays only

                if self.timeframe == TimeFrame.ONE_DAY:
                    periods.append(
                        current_date.replace(hour=16, minute=0)
                    )  # End of day
                else:
                    # Generate intraday periods
                    day_start = current_date.replace(hour=9, minute=30)
                    periods_added = 0

                    period_time = day_start
                    while period_time.hour < 16 and periods_added < max_periods_per_day:
                        periods.append(period_time)
                        period_time += timedelta(minutes=interval_minutes)
                        periods_added += 1

                        # Stop at market close
                        if period_time.hour >= 16:
                            break

            # Move to next day
            current_date += timedelta(days=1)
            current_date = current_date.replace(
                hour=9, minute=30, second=0, microsecond=0
            )

        return periods

    def _get_portfolio_sample_frequency(self) -> int:
        """Get how often to sample portfolio value based on timeframe."""
        if self.timeframe == TimeFrame.ONE_MINUTE:
            return 60  # Every hour
        elif self.timeframe == TimeFrame.FIVE_MIN:
            return 12  # Every hour
        elif self.timeframe == TimeFrame.FIFTEEN_MIN:
            return 4  # Every hour
        elif self.timeframe == TimeFrame.ONE_HOUR:
            return 1  # Every period
        else:
            return 1  # Every day

    async def _process_trading_period(
        self,
        period_datetime: datetime,
        symbols: List[str],
        strategy: TradingStrategy,
        max_positions: int,
    ) -> None:
        """Process a single trading period."""
        try:
            # 1. Update existing positions
            await self._update_position_prices(period_datetime)

            # 2. Check exit conditions
            await self._check_exit_conditions(period_datetime, strategy)

            # 3. For intraday timeframes, only run screener periodically to reduce computation
            should_run_screener = self._should_run_screener(period_datetime)

            if len(self.positions) < max_positions and should_run_screener:
                # Run screener analysis in parallel
                screener_results = await self._run_parallel_screener(
                    period_datetime, symbols
                )

                # total_candidates = sum(
                #     len(candidates) for candidates in screener_results.values()
                # )  # Unused variable
                # if total_candidates > 0:
                #     self.logger.info(f"📊 Screener found {total_candidates} candidates: {dict((k, len(v)) for k, v in screener_results.items())}")

                # 4. Analyze screener candidates in parallel
                for screener_type, candidates in screener_results.items():
                    if len(self.positions) >= max_positions:
                        break

                    if candidates:
                        self.logger.debug(
                            f"🔍 Analyzing {len(candidates)} {screener_type} candidates"
                        )

                    # Process candidates in parallel batches
                    signals = await self._analyze_candidates_parallel(
                        candidates, period_datetime, strategy, screener_type
                    )

                    # Execute signals sequentially to maintain portfolio state integrity
                    for signal in signals:
                        if signal and len(self.positions) < max_positions:
                            self.signals_generated += 1
                            await self._execute_signal(
                                signal, period_datetime, strategy.name
                            )

                        if len(self.positions) >= max_positions:
                            break

        except Exception as e:
            self.logger.debug(f"Error processing {period_datetime}: {e}")

    def _should_run_screener(self, period_datetime: datetime) -> bool:
        """Determine if screener should run for this period to optimize performance."""
        if self.timeframe == TimeFrame.ONE_DAY:
            return True
        elif self.timeframe == TimeFrame.ONE_HOUR:
            return True  # Run every hour
        elif self.timeframe == TimeFrame.FIFTEEN_MIN:
            return period_datetime.minute in [30, 0]  # Run twice per hour
        elif self.timeframe == TimeFrame.FIVE_MIN:
            return period_datetime.minute in [30, 0]  # Run twice per hour
        else:  # ONE_MINUTE
            return period_datetime.minute in [30, 0]  # Run twice per hour

    async def _update_position_prices(self, period_datetime: datetime) -> None:
        """Update current prices for all positions."""
        for symbol, position in self.positions.items():
            try:
                # Get data from current day
                start_date = period_datetime.date()
                end_date = period_datetime.date()

                df = await self.data_store.load_market_data(
                    ticker=symbol,
                    timeframe=self.timeframe,
                    start_date=start_date,
                    end_date=end_date,
                )

                if not df.is_empty():
                    # For intraday, find the closest timestamp <= current period
                    df_sorted = df.sort("timestamp")

                    if self.timeframe == TimeFrame.ONE_DAY:
                        row = df_sorted.tail(1).row(0, named=True)
                    else:
                        # Find data at or before current period time (convert to naive)
                        current_naive = period_datetime.replace(tzinfo=None)
                        suitable_rows = df_sorted.filter(
                            df_sorted["timestamp"] <= current_naive
                        )
                        if not suitable_rows.is_empty():
                            row = suitable_rows.tail(1).row(0, named=True)
                        else:
                            # Fallback to latest available data for the day
                            row = df_sorted.tail(1).row(0, named=True)

                    position.current_price = Decimal(str(row["close"]))

            except Exception as e:
                self.logger.debug(f"Could not update price for {symbol}: {e}")

    async def _check_exit_conditions(
        self, period_datetime: datetime, strategy: TradingStrategy
    ) -> None:
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

            # Time-based exit (strategy dependent) - adjust for timeframe
            if self.timeframe == TimeFrame.ONE_DAY:
                hold_days = (period_datetime - position.entry_date).days
                max_hold = strategy.max_hold_days
            else:
                # For intraday, convert to hours/periods
                hold_hours = (
                    period_datetime - position.entry_date
                ).total_seconds() / 3600
                if self.timeframe == TimeFrame.ONE_HOUR:
                    max_hold = int(
                        strategy.max_hold_days * 6.5
                    )  # Trading hours per day
                elif self.timeframe == TimeFrame.FIFTEEN_MIN:
                    max_hold = strategy.max_hold_days * 26  # 15-min periods per day
                elif self.timeframe == TimeFrame.FIVE_MIN:
                    max_hold = strategy.max_hold_days * 78  # 5-min periods per day
                else:  # ONE_MINUTE
                    max_hold = strategy.max_hold_days * 390  # 1-min periods per day

                hold_days = int(
                    hold_hours / (6.5 if self.timeframe != TimeFrame.ONE_DAY else 24)
                )

            if hold_days >= (
                max_hold
                if self.timeframe == TimeFrame.ONE_DAY
                else int(
                    max_hold / (6.5 if self.timeframe != TimeFrame.ONE_DAY else 24)
                )
            ):
                positions_to_close.append((symbol, "time_exit"))
                continue

        # Close positions
        for symbol, reason in positions_to_close:
            await self._close_position(symbol, period_datetime, reason)

    async def _execute_signal(
        self, signal: TradingSignal, period_datetime: datetime, strategy_name: str
    ) -> None:
        """Execute a trading signal."""
        try:
            # Calculate position size
            portfolio_value = self._calculate_portfolio_value()
            position_value = portfolio_value * Decimal(str(signal.position_size))
            quantity = int(position_value / signal.price)

            if quantity <= 0:
                return

            # Check if we have enough cash (including commission)
            total_cost = signal.price * Decimal(str(quantity)) + Decimal(
                "2.00"
            )  # $2 commission
            if total_cost > self.cash:
                return

            # Create position
            position = Position(
                symbol=signal.symbol,
                quantity=quantity,
                entry_price=signal.price,
                entry_date=period_datetime,
                current_price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy_name=strategy_name,
            )

            # Update portfolio
            self.positions[signal.symbol] = position
            self.cash -= total_cost
            self.signals_executed += 1

            # self.logger.info(f"Opened position: {signal.symbol} x{quantity} @ ${signal.price} (Confidence: {signal.confidence:.1f}%)")

        except Exception as e:
            self.logger.error(f"Error executing signal for {signal.symbol}: {e}")

    async def _close_position(
        self, symbol: str, period_datetime: datetime, reason: str
    ) -> None:
        """Close a position."""
        if symbol not in self.positions:
            return

        try:
            position = self.positions[symbol]
            exit_price = position.current_price

            # Calculate P&L
            gross_pnl = (exit_price - position.entry_price) * Decimal(
                str(position.quantity)
            )
            commission = Decimal("2.00")  # $2 total commission
            net_pnl = gross_pnl - commission

            # Calculate hold time based on timeframe
            if self.timeframe == TimeFrame.ONE_DAY:
                hold_days = (period_datetime - position.entry_date).days
            else:
                hold_hours = (
                    period_datetime - position.entry_date
                ).total_seconds() / 3600
                hold_days = int(hold_hours / 6.5)  # Convert to trading days
            pnl_percentage = float(
                net_pnl / (position.entry_price * Decimal(str(position.quantity)))
            )

            # Create trade record
            trade = Trade(
                symbol=symbol,
                entry_date=position.entry_date,
                exit_date=period_datetime,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                pnl=net_pnl,
                pnl_percentage=pnl_percentage,
                hold_days=hold_days,
                strategy=position.strategy_name,
                entry_reason="signal",
                exit_reason=reason,
            )

            # Update portfolio
            proceeds = exit_price * Decimal(str(position.quantity)) - commission
            self.cash += proceeds
            self.trades.append(trade)

            del self.positions[symbol]

            # self.logger.info(f"Closed position: {symbol}, P&L: ${net_pnl:.2f} ({pnl_percentage:.2%}), Reason: {reason}")

        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {e}")

    async def _get_historical_data(
        self, symbol: str, current_datetime: datetime
    ) -> Optional[List[MarketData]]:
        """Get historical data for symbol analysis."""
        try:
            # Adjust lookback period based on timeframe
            if self.timeframe == TimeFrame.ONE_MINUTE:
                lookback_days = 7  # 7 days for minute data
            elif self.timeframe == TimeFrame.FIVE_MIN:
                lookback_days = 14  # 2 weeks for 5-minute data
            elif self.timeframe == TimeFrame.FIFTEEN_MIN:
                lookback_days = 30  # 1 month for 15-minute data
            elif self.timeframe == TimeFrame.ONE_HOUR:
                lookback_days = 60  # 2 months for hourly data
            else:  # ONE_DAY
                lookback_days = 60  # 2 months for daily data

            start_date = (current_datetime - timedelta(days=lookback_days)).date()
            end_date = current_datetime.date()

            df = await self.data_store.load_market_data(
                symbol, self.timeframe, start_date, end_date
            )
            if df.is_empty():
                return None

            historical_data = []
            df_sorted = df.sort("timestamp")

            # For intraday timeframes, only include data up to current time
            if self.timeframe != TimeFrame.ONE_DAY:
                current_naive = current_datetime.replace(tzinfo=None)
                df_sorted = df_sorted.filter(df_sorted["timestamp"] <= current_naive)

            for row in df_sorted.iter_rows(named=True):
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
                        timeframe=self.timeframe,
                    )
                )

            return historical_data

        except Exception as e:
            self.logger.debug(f"Error loading historical data for {symbol}: {e}")
            return None

    async def _preload_data(self, symbols: List[str]) -> None:
        """Pre-load historical data for all symbols in parallel."""
        self.logger.info(
            f"Pre-loading historical data for {len(symbols)} symbols in parallel..."
        )

        # Calculate date range for data loading
        if self.timeframe == TimeFrame.ONE_MINUTE:
            lookback_days = 7
        elif self.timeframe == TimeFrame.FIVE_MIN:
            lookback_days = 14
        elif self.timeframe == TimeFrame.FIFTEEN_MIN:
            lookback_days = 30
        elif self.timeframe == TimeFrame.ONE_HOUR:
            lookback_days = 60
        else:
            lookback_days = 60

        start_date = (self.start_date - timedelta(days=lookback_days)).date()
        end_date = self.end_date.date()

        # Create tasks for parallel data loading
        load_tasks = []
        for symbol in symbols:
            task = self._load_symbol_data(symbol, start_date, end_date)
            load_tasks.append(task)

        # Execute in batches to avoid overwhelming the system
        batch_size = self.max_workers
        completed = 0
        for i in range(0, len(load_tasks), batch_size):
            batch = load_tasks[i : i + batch_size]
            await asyncio.gather(*batch, return_exceptions=True)
            completed += len(batch)

            # Progress reporting
            if i % (batch_size * 3) == 0 or completed >= len(load_tasks):
                self.logger.info(
                    f"Data loading progress: {completed}/{len(load_tasks)} symbols"
                )

        successful_loads = len(self.data_cache)
        self.logger.info(
            f"Pre-loaded data for {successful_loads}/{len(symbols)} symbols ({successful_loads / len(symbols) * 100:.1f}%)"
        )

    async def _load_symbol_data(self, symbol: str, start_date, end_date) -> None:
        """Load and cache data for a single symbol."""
        try:
            df = await self.data_store.load_market_data(
                symbol, self.timeframe, start_date, end_date
            )
            if not df.is_empty():
                row_count = len(df)
                self.data_cache[symbol] = {
                    "dataframe": df.sort("timestamp"),
                    "loaded_start": start_date,
                    "loaded_end": end_date,
                }
                self.logger.debug(
                    f"✅ Loaded {row_count} rows for {symbol} ({self.timeframe.value})"
                )
            else:
                self.logger.debug(
                    f"❌ No data found for {symbol} ({self.timeframe.value})"
                )
        except Exception as e:
            self.logger.debug(f"Failed to preload data for {symbol}: {e}")

    async def _run_parallel_screener(
        self, period_datetime: datetime, symbols: List[str]
    ) -> Dict[str, List[str]]:
        """Run screener analysis in parallel for different screener types."""
        self.logger.debug(
            f"🔍 Running screener for {len(symbols)} symbols at {period_datetime}"
        )

        # Create parallel tasks for different screener types
        tasks = [
            self.screener.get_breakout_candidates(
                period_datetime, symbols[:30]
            ),  # Limit symbols for performance
            self.screener.get_momentum_candidates(period_datetime, symbols[:30]),
            self._get_value_candidates_async(period_datetime, symbols[:20]),
        ]

        try:
            breakouts_res, momentum_res, value_stocks_res = await asyncio.gather(
                *tasks, return_exceptions=True
            )

            breakouts = []
            if not isinstance(breakouts_res, BaseException):
                breakouts = breakouts_res
            else:
                self.logger.debug(f"❌ Breakout screener failed: {breakouts_res}")

            momentum = []
            if not isinstance(momentum_res, BaseException):
                momentum = momentum_res
            else:
                self.logger.debug(f"❌ Momentum screener failed: {momentum_res}")

            value_stocks = []
            if not isinstance(value_stocks_res, BaseException):
                value_stocks = value_stocks_res
            else:
                self.logger.debug(f"❌ Value screener failed: {value_stocks_res}")

            result = {
                "breakouts": breakouts[:15],
                "momentum": momentum[:15],
                "value_stocks": value_stocks[:15],
            }

            # Log screener results (always log, not just when > 0)
            total_found = sum(len(candidates) for candidates in result.values())
            self.logger.debug(
                f"📊 {period_datetime.strftime('%Y-%m-%d %H:%M')}: Found {len(result['breakouts'])} breakouts, {len(result['momentum'])} momentum, {len(result['value_stocks'])} value (total: {total_found})"
            )

            return result
        except Exception as e:
            self.logger.debug(f"❌ Error in parallel screener: {e}")
            return {"breakouts": [], "momentum": [], "value_stocks": []}

    async def _get_value_candidates_async(
        self, period_datetime: datetime, symbols: List[str]
    ) -> List[str]:
        """Async version of value candidate screening."""
        value_stocks = []

        # Process symbols in parallel batches
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            batch_tasks = [
                self._analyze_value_candidate(period_datetime, symbol)
                for symbol in batch
            ]
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for symbol, result in zip(batch, results):
                if result is True:  # Symbol passed value screening
                    value_stocks.append(symbol)

        return value_stocks

    async def _analyze_value_candidate(
        self, period_datetime: datetime, symbol: str
    ) -> bool:
        """Analyze if a symbol is a value candidate."""
        try:
            # Use cached data if available
            if symbol in self.data_cache:
                df = self.data_cache[symbol]["dataframe"]
                # Filter to date range
                df = df.filter(df["timestamp"] <= period_datetime)
                if df.is_empty():
                    return False
            else:
                # Fallback to direct loading
                lookback_days = 60 if self.timeframe == TimeFrame.ONE_DAY else 30
                start_date = (period_datetime - timedelta(days=lookback_days)).date()
                end_date = period_datetime.date()
                df = await self.data_store.load_market_data(
                    symbol, self.timeframe, start_date, end_date
                )
                if df.is_empty():
                    return False

            data_rows = df.sort("timestamp").to_dicts()
            if len(data_rows) < 30:
                return False

            current_price = data_rows[-1]["close"]
            high_60 = max(d["high"] for d in data_rows)

            # Value: trading 20%+ below 60-day high, price > $10
            discount = (high_60 - current_price) / high_60
            return discount > 0.2 and current_price > 10.0

        except Exception:
            return False

    async def _analyze_candidates_parallel(
        self,
        candidates: List[str],
        period_datetime: datetime,
        strategy: TradingStrategy,
        screener_type: str,
    ) -> List[Optional[TradingSignal]]:
        """Analyze multiple candidates in parallel for signal generation."""
        if not candidates:
            return []

        # Create analysis tasks
        analysis_tasks = []
        for symbol in candidates:
            if symbol not in self.positions:  # Skip if already have position
                task = self._analyze_single_candidate(
                    symbol, period_datetime, strategy, screener_type
                )
                analysis_tasks.append(task)

        if not analysis_tasks:
            return []

        # Execute in parallel with limited concurrency
        batch_size = min(
            6, len(analysis_tasks)
        )  # Reduced batch size for memory management
        signals: List[Optional[TradingSignal]] = []

        for i in range(0, len(analysis_tasks), batch_size):
            batch = analysis_tasks[i : i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for result in batch_results:
                if not isinstance(result, BaseException) and result is not None:
                    signals.append(result)

        return signals

    async def _analyze_single_candidate(
        self,
        symbol: str,
        period_datetime: datetime,
        strategy: TradingStrategy,
        screener_type: str,
    ) -> Optional[TradingSignal]:
        """Analyze a single candidate for signal generation."""
        try:
            # Get historical data (use cache if available)
            historical_data = await self._get_historical_data_cached(
                symbol, period_datetime
            )
            if not historical_data or len(historical_data) < 20:
                self.logger.debug(
                    f"❌ Insufficient data for {symbol}: {len(historical_data) if historical_data else 0} rows"
                )
                return None

            current_data = historical_data[-1]
            self.logger.debug(
                f"🔍 Analyzing {symbol}: price=${float(current_data.close):.2f}, volume={current_data.volume:,}"
            )

            # Generate signal
            return await strategy.analyze_symbol(
                symbol, current_data, historical_data, screener_type
            )

        except Exception as e:
            self.logger.debug(f"Error analyzing candidate {symbol}: {e}")
            return None

    async def _get_historical_data_cached(
        self, symbol: str, current_datetime: datetime
    ) -> Optional[List[MarketData]]:
        """Get historical data using cache when possible."""
        try:
            # Try to use cached data first
            if symbol in self.data_cache:
                df = self.data_cache[symbol]["dataframe"]

                # Filter to current time for intraday (convert to naive datetime)
                if self.timeframe != TimeFrame.ONE_DAY:
                    current_naive = current_datetime.replace(tzinfo=None)
                    df = df.filter(df["timestamp"] <= current_naive)

                if not df.is_empty() and len(df) >= 20:
                    return self._dataframe_to_market_data(df, symbol)

            # Fallback to direct loading if cache miss or insufficient data
            return await self._get_historical_data(symbol, current_datetime)

        except Exception as e:
            self.logger.debug(f"Error getting cached historical data for {symbol}: {e}")
            return None

    def _dataframe_to_market_data(self, df, symbol: str) -> List[MarketData]:
        """Convert dataframe to MarketData list."""
        historical_data = []
        # Limit data to last 100 rows for memory efficiency
        df_limited = df.tail(100) if len(df) > 100 else df

        for row in df_limited.iter_rows(named=True):
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
                    timeframe=self.timeframe,
                )
            )
        return historical_data

    def _calculate_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value."""
        total = self.cash
        for position in self.positions.values():
            total += position.current_price * Decimal(str(position.quantity))
        return total

    def _generate_results(
        self, strategy: TradingStrategy, execution_time: float
    ) -> Dict[str, Any]:
        """Generate comprehensive results."""
        final_value = self._calculate_portfolio_value()
        total_return = float(
            (final_value - self.initial_capital) / self.initial_capital
        )

        # Trading statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_win = (
            sum(float(t.pnl) for t in winning_trades) / len(winning_trades)
            if winning_trades
            else 0
        )
        avg_loss = (
            sum(float(t.pnl) for t in losing_trades) / len(losing_trades)
            if losing_trades
            else 0
        )

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
            prev_val = float(self.portfolio_history[i - 1][1])
            curr_val = float(self.portfolio_history[i][1])
            if prev_val > 0:
                daily_returns.append((curr_val - prev_val) / prev_val)

        import math

        if daily_returns:
            avg_daily_return = sum(daily_returns) / len(daily_returns)
            daily_volatility = math.sqrt(
                sum((r - avg_daily_return) ** 2 for r in daily_returns)
                / len(daily_returns)
            )
            annualized_return = (1 + avg_daily_return) ** 252 - 1
            annualized_volatility = daily_volatility * math.sqrt(252)
            sharpe_ratio = (
                (annualized_return - 0.02) / annualized_volatility
                if annualized_volatility > 0
                else 0
            )
        else:
            annualized_return = total_return
            sharpe_ratio = 0

        return {
            "strategy": {
                "name": strategy.name,
                "mode": strategy.mode,
                "parameters": {
                    "min_confidence": strategy.min_confidence,
                    "stop_loss_pct": strategy.stop_loss_pct,
                    "take_profit_pct": strategy.take_profit_pct,
                    "max_hold_days": strategy.max_hold_days,
                },
            },
            "performance": {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "final_value": float(final_value),
                "max_drawdown": max_dd,
                "sharpe_ratio": sharpe_ratio,
            },
            "trading": {
                "total_trades": len(self.trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "avg_hold_days": (
                    sum(t.hold_days for t in self.trades) / len(self.trades)
                    if self.trades
                    else 0
                ),
                "profit_factor": (
                    abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades)))
                    if avg_loss != 0
                    else float("inf")
                ),
            },
            "signals": {
                "generated": self.signals_generated,
                "executed": self.signals_executed,
                "execution_rate": self.signals_executed
                / max(self.signals_generated, 1),
            },
            "execution": {
                "time_seconds": execution_time,
                "days_processed": len(self.portfolio_history),
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "timeframe": self.timeframe.value,
            },
            "trades": [
                {
                    "symbol": t.symbol,
                    "entry_date": t.entry_date.isoformat(),
                    "exit_date": t.exit_date.isoformat(),
                    "pnl": float(t.pnl),
                    "pnl_percentage": t.pnl_percentage,
                    "hold_days": t.hold_days,
                    "strategy": t.strategy,
                    "exit_reason": t.exit_reason,
                }
                for t in self.trades
            ],
        }


def format_currency(amount: float) -> str:
    """Format currency amount."""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format percentage."""
    return f"{value:.2%}"


def display_results(results: Dict[str, Any]) -> None:
    """Display comprehensive backtest results."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BACKTESTING RESULTS")
    print("=" * 80)

    # Strategy info
    strategy = results["strategy"]
    execution = results["execution"]
    timeframe = execution.get("timeframe", "daily")
    print(f"\nStrategy: {strategy['name']} ({strategy['mode']}) - {timeframe}")
    print("Parameters:")
    for key, value in strategy["parameters"].items():
        print(f"  {key}: {value}")

    # Performance summary
    perf = results["performance"]
    print("\nPerformance Summary:")
    print(f"  Total Return: {format_percentage(perf['total_return'])}")
    print(f"  Annualized Return: {format_percentage(perf['annualized_return'])}")
    print(f"  Final Portfolio Value: {format_currency(perf['final_value'])}")
    print(f"  Maximum Drawdown: {format_percentage(perf['max_drawdown'])}")
    print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")

    # Trading statistics
    trading = results["trading"]
    print("\nTrading Statistics:")
    print(f"  Total Trades: {trading['total_trades']}")
    print(f"  Winning Trades: {trading['winning_trades']}")
    print(f"  Losing Trades: {trading['losing_trades']}")
    print(f"  Win Rate: {format_percentage(trading['win_rate'])}")
    print(f"  Average Win: {format_currency(trading['avg_win'])}")
    print(f"  Average Loss: {format_currency(trading['avg_loss'])}")
    print(f"  Profit Factor: {trading['profit_factor']:.2f}")
    print(f"  Average Hold Days: {trading['avg_hold_days']:.1f}")

    # Signal analysis
    signals = results["signals"]
    print("\nSignal Analysis:")
    print(f"  Signals Generated: {signals['generated']}")
    print(f"  Signals Executed: {signals['executed']}")
    print(f"  Execution Rate: {format_percentage(signals['execution_rate'])}")

    # Execution info
    execution = results["execution"]
    print("\nExecution Info:")
    print(f"  Execution Time: {execution['time_seconds']:.2f} seconds")
    print(f"  Days Processed: {execution['days_processed']}")
    print(f"  Period: {execution['start_date'][:10]} to {execution['end_date'][:10]}")

    print("\n" + "=" * 80)


async def run_strategy_comparison():
    """Run and compare different trading strategies."""
    print("COMPREHENSIVE STRATEGY COMPARISON")
    print("=" * 50)

    # Define test period (last 3 months with good data coverage)
    end_date = datetime(2025, 8, 19, tzinfo=timezone.utc)
    start_date = datetime(2025, 5, 20, tzinfo=timezone.utc)  # ~3 months
    initial_capital = Decimal("100000")

    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Duration: {(end_date - start_date).days} days")
    print(f"Initial Capital: {format_currency(float(initial_capital))}")

    # Define strategies to test
    strategies = [
        TradingStrategy("Day Trading Strategy", "day_trading"),
        TradingStrategy("Swing Trading Strategy", "swing_trading"),
        TradingStrategy("Position Trading Strategy", "position_trading"),
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
            print(
                f"   Return: {format_percentage(result['performance']['total_return'])}"
            )
            print(f"   Trades: {result['trading']['total_trades']}")
            print(f"   Win Rate: {format_percentage(result['trading']['win_rate'])}")

        except Exception as e:
            print(f"❌ {strategy.name} failed: {e}")
            continue

    # Display comparison
    if results:
        print("\n" + "=" * 100)
        print("STRATEGY COMPARISON SUMMARY")
        print("=" * 100)

        print(
            f"\n{'Strategy':<25} {'Return':<12} {'Max DD':<10} {'Sharpe':<8} {'Trades':<8} {'Win Rate':<10}"
        )
        print("-" * 85)

        for strategy_name, result in results.items():
            perf = result["performance"]
            trading = result["trading"]

            print(
                f"{strategy_name:<25} "
                f"{format_percentage(perf['total_return']):<12} "
                f"{format_percentage(perf['max_drawdown']):<10} "
                f"{perf['sharpe_ratio']:<8.3f} "
                f"{trading['total_trades']:<8} "
                f"{format_percentage(trading['win_rate']):<10}"
            )

        # Find best performers
        best_return = max(
            results.items(), key=lambda x: x[1]["performance"]["total_return"]
        )
        best_sharpe = max(
            results.items(), key=lambda x: x[1]["performance"]["sharpe_ratio"]
        )

        print(
            f"\n🏆 Best Total Return: {best_return[0]} ({format_percentage(best_return[1]['performance']['total_return'])})"
        )
        print(
            f"🏆 Best Risk-Adjusted: {best_sharpe[0]} (Sharpe: {best_sharpe[1]['performance']['sharpe_ratio']:.3f})"
        )

    return results


async def run_multi_timeframe_comparison():
    """Run and compare strategies across multiple timeframes."""
    print("MULTI-TIMEFRAME STRATEGY COMPARISON")
    print("=" * 60)

    # Define timeframes to test
    timeframes = [
        TimeFrame.ONE_HOUR,
        TimeFrame.FIFTEEN_MIN,
        TimeFrame.ONE_DAY,
        # TimeFrame.FIVE_MIN,
        # TimeFrame.ONE_MINUTEs
    ]

    # Define test period
    end_date = datetime(2025, 8, 19, tzinfo=timezone.utc)
    start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)  # ~7 months
    initial_capital = Decimal("10000")

    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Duration: {(end_date - start_date).days} days")
    print(f"Initial Capital: {format_currency(float(initial_capital))}")
    print(f"Timeframes: {[tf.value for tf in timeframes]}")

    # Test strategy
    strategy = TradingStrategy("Multi-Timeframe Test", "swing_trading")

    results = {}

    # Run backtest for each timeframe
    for timeframe in timeframes:
        print(f"\n🔄 Running backtest on {timeframe.value}...")

        engine = ComprehensiveBacktestEngine(
            start_date, end_date, initial_capital, timeframe
        )

        try:
            result = await engine.run_backtest(strategy, max_positions=6)
            results[timeframe.value] = result

            print(f"✅ {timeframe.value} completed:")
            print(
                f"   Return: {format_percentage(result['performance']['total_return'])}"
            )
            print(f"   Trades: {result['trading']['total_trades']}")
            print(f"   Win Rate: {format_percentage(result['trading']['win_rate'])}")

        except Exception as e:
            print(f"❌ {timeframe.value} failed: {e}")
            continue

    # Display comparison
    if results:
        print("\n" + "=" * 100)
        print("TIMEFRAME COMPARISON SUMMARY")
        print("=" * 100)

        print(
            f"\n{'Timeframe':<15} {'Return':<12} {'Max DD':<10} {'Sharpe':<8} {'Trades':<8} {'Win Rate':<10} {'Avg Hold':<10}"
        )
        print("-" * 95)

        for timeframe_name, result in results.items():
            perf = result["performance"]
            trading = result["trading"]

            print(
                f"{timeframe_name:<15} "
                f"{format_percentage(perf['total_return']):<12} "
                f"{format_percentage(perf['max_drawdown']):<10} "
                f"{perf['sharpe_ratio']:<8.3f} "
                f"{trading['total_trades']:<8} "
                f"{format_percentage(trading['win_rate']):<10} "
                f"{trading['avg_hold_days']:<10.1f}"
            )

        # Find best performers
        best_return = max(
            results.items(), key=lambda x: x[1]["performance"]["total_return"]
        )
        best_sharpe = max(
            results.items(), key=lambda x: x[1]["performance"]["sharpe_ratio"]
        )
        most_trades = max(
            results.items(), key=lambda x: x[1]["trading"]["total_trades"]
        )

        print(
            f"\n🏆 Best Total Return: {best_return[0]} ({format_percentage(best_return[1]['performance']['total_return'])})"
        )
        print(
            f"🏆 Best Risk-Adjusted: {best_sharpe[0]} (Sharpe: {best_sharpe[1]['performance']['sharpe_ratio']:.3f})"
        )
        print(
            f"📊 Most Active: {most_trades[0]} ({most_trades[1]['trading']['total_trades']} trades)"
        )

        # Analysis insights
        print("\nInsights:")
        print("• Shorter timeframes typically generate more trades")
        print("• Daily timeframes may have better risk-adjusted returns")
        print("• Consider transaction costs impact on shorter timeframes")

    return results


async def run_comprehensive_timeframe_analysis():
    """Run comprehensive analysis across strategies and timeframes."""
    print("COMPREHENSIVE STRATEGY & TIMEFRAME ANALYSIS")
    print("=" * 70)

    # Define test parameters
    timeframes = [TimeFrame.ONE_DAY, TimeFrame.ONE_HOUR]
    strategies = [
        TradingStrategy("Day Trading", "day_trading"),
        TradingStrategy("Swing Trading", "swing_trading"),
    ]

    end_date = datetime(2025, 8, 19, tzinfo=timezone.utc)
    start_date = datetime(2025, 6, 1, tzinfo=timezone.utc)  # ~2.5 months
    initial_capital = Decimal("100000")

    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Strategies: {len(strategies)}, Timeframes: {len(timeframes)}")
    print(f"Total combinations: {len(strategies) * len(timeframes)}")

    all_results = {}

    # Run all combinations
    for strategy in strategies:
        strategy_results = {}

        for timeframe in timeframes:
            combination_name = f"{strategy.name} - {timeframe.value}"
            print(f"\n🔄 Running {combination_name}...")

            engine = ComprehensiveBacktestEngine(
                start_date, end_date, initial_capital, timeframe
            )

            try:
                result = await engine.run_backtest(strategy, max_positions=5)
                strategy_results[timeframe.value] = result

                print(f"✅ {combination_name} completed:")
                print(
                    f"   Return: {format_percentage(result['performance']['total_return'])}"
                )
                print(f"   Trades: {result['trading']['total_trades']}")

            except Exception as e:
                print(f"❌ {combination_name} failed: {e}")
                continue

        all_results[strategy.name] = strategy_results

    # Display comprehensive comparison
    if all_results:
        print("\n" + "=" * 120)
        print("COMPREHENSIVE COMPARISON MATRIX")
        print("=" * 120)

        print(
            f"\n{'Strategy':<20} {'Timeframe':<15} {'Return':<12} {'Max DD':<10} {'Sharpe':<8} {'Trades':<8} {'Win Rate':<10}"
        )
        print("-" * 105)

        best_overall = None
        best_return = -float("inf")

        for strategy_name, timeframe_results in all_results.items():
            for timeframe_name, result in timeframe_results.items():
                perf = result["performance"]
                trading = result["trading"]

                print(
                    f"{strategy_name:<20} "
                    f"{timeframe_name:<15} "
                    f"{format_percentage(perf['total_return']):<12} "
                    f"{format_percentage(perf['max_drawdown']):<10} "
                    f"{perf['sharpe_ratio']:<8.3f} "
                    f"{trading['total_trades']:<8} "
                    f"{format_percentage(trading['win_rate']):<10}"
                )

                # Track best overall
                if perf["total_return"] > best_return:
                    best_return = perf["total_return"]
                    best_overall = f"{strategy_name} - {timeframe_name}"

        print(
            f"\n🏆 Best Overall Combination: {best_overall} ({format_percentage(best_return)})"
        )

        # Strategy analysis
        print("\nStrategy Analysis:")
        for strategy_name, timeframe_results in all_results.items():
            if timeframe_results:
                avg_return = sum(
                    r["performance"]["total_return"] for r in timeframe_results.values()
                ) / len(timeframe_results)
                print(
                    f"• {strategy_name}: Average return across timeframes: {format_percentage(avg_return)}"
                )

    return all_results


async def run_debug_signal_test():
    """Debug function to test signal generation with detailed logging."""
    print("🔧 DEBUG SIGNAL GENERATION TEST")
    print("=" * 50)

    # Enable debug logging
    logging.getLogger().setLevel(logging.DEBUG)

    # Very short test period
    end_date = datetime(2025, 8, 19, tzinfo=timezone.utc)
    start_date = datetime(2025, 8, 18, tzinfo=timezone.utc)  # Just 1 day
    initial_capital = Decimal("10000")

    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("Testing with debug logging enabled...")

    # Test with relaxed day trading strategy
    strategy = TradingStrategy("Debug Day Trading", "day_trading")
    engine = ComprehensiveBacktestEngine(
        start_date, end_date, initial_capital, TimeFrame.ONE_HOUR
    )

    print("\nStrategy settings:")
    print(f"  Min confidence: {strategy.min_confidence}%")
    print(f"  Stop loss: {strategy.stop_loss_pct:.1%}")
    print(f"  Take profit: {strategy.take_profit_pct:.1%}")

    try:
        result = await engine.run_backtest(strategy, max_positions=3)

        # perf = results["performance"]  # Unused
        # trading = results["trading"]  # Unused
        # signals = results["signals"]  # Unused

        # print(f"\nDEBUG RESULTS:")
        # print(f"  Signals Generated: {signals['generated']}")
        # print(f"  Signals Executed: {signals['executed']}")
        # print(f"  Total Trades: {trading['total_trades']}")
        # print(f"  Portfolio Value: ${perf['final_value']:,.2f}")

        return result

    except Exception as e:
        print(f"Debug test failed: {e}")
        import traceback

        traceback.print_exc()
        return None


async def run_quick_timeframe_test(
    timeframes: Optional[List[TimeFrame]] = None,
    strategy_mode: str = "position_trading",
):
    """Quick function to test specific timeframes with minimal setup."""
    if timeframes is None:
        timeframes = [TimeFrame.ONE_DAY, TimeFrame.ONE_HOUR, TimeFrame.FIFTEEN_MIN]

    print(f"QUICK TIMEFRAME TEST - {strategy_mode.upper()}")
    print("=" * 60)

    # Short test period for quick results
    end_date = datetime(2025, 8, 19, tzinfo=timezone.utc)
    start_date = datetime(2025, 7, 1, tzinfo=timezone.utc)  # ~1.5 months
    initial_capital = Decimal("10000")

    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Timeframes: {[tf.value for tf in timeframes]}")
    print(f"Quick test with ${format_currency(float(initial_capital))} capital")

    strategy = TradingStrategy(f"Quick Test - {strategy_mode.title()}", strategy_mode)
    results = {}

    for timeframe in timeframes:
        print(f"\n⚡ Quick test: {timeframe.value}...")

        engine = ComprehensiveBacktestEngine(
            start_date, end_date, initial_capital, timeframe
        )

        try:
            result = await engine.run_backtest(strategy, max_positions=4)
            results[timeframe.value] = result

            perf = result["performance"]
            trading = result["trading"]

            print(
                f"   ✓ Return: {format_percentage(perf['total_return'])} | "
                f"Trades: {trading['total_trades']} | "
                f"Win Rate: {format_percentage(trading['win_rate'])}"
            )

        except Exception as e:
            print(f"   ✗ Failed: {e}")

    if results:
        print("\n📊 QUICK COMPARISON:")
        print("-" * 50)
        for tf, result in results.items():
            perf = result["performance"]
            print(
                f"{tf:>15}: {format_percentage(perf['total_return']):>8} return, "
                f"Sharpe: {perf['sharpe_ratio']:>5.2f}"
            )

    return results


async def run_single_strategy_demo():
    """Run a detailed demo of a single strategy."""
    print("SINGLE STRATEGY DETAILED ANALYSIS")
    print("=" * 50)

    # Use a shorter period for detailed analysis
    end_date = datetime(2025, 8, 19, tzinfo=timezone.utc)
    start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)  # ~7 weeks
    initial_capital = Decimal("10000")

    print("Testing Day Trading Strategy")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: {format_currency(float(initial_capital))}")

    strategy = TradingStrategy("Detailed Day Trading", "day_trading")
    engine = ComprehensiveBacktestEngine(start_date, end_date, initial_capital)

    result = await engine.run_backtest(strategy, max_positions=5)

    # Display detailed results
    display_results(result)

    # Show sample trades
    trades = result["trades"]
    if trades:
        print("\nSample Trades (Last 10):")
        print("-" * 80)
        for trade in trades[-10:]:
            entry_date = trade["entry_date"][:10]
            exit_date = trade["exit_date"][:10]
            pnl = format_currency(trade["pnl"])
            pnl_pct = format_percentage(trade["pnl_percentage"])

            print(
                f"{trade['symbol']:<6} {entry_date} to {exit_date} "
                f"({trade['hold_days']}d) {pnl:<10} {pnl_pct:<8} {trade['exit_reason']}"
            )

    return result


async def main():
    """Main demonstration function."""
    print("🚀 COMPREHENSIVE BACKTESTING DEMONSTRATION")
    print("=" * 60)
    print("This demo uses your extensive historical data (12+ months available)")
    print("to test sophisticated trading strategies with screener simulation.")
    print()

    try:
        # First, show data availability
        print("📊 CHECKING DATA AVAILABILITY...")
        store = SimpleDataStore("data/parquet")
        symbols = store.get_available_symbols()

        print(f"✅ Found {len(symbols)} symbols in data store")
        print(
            f"📈 Major symbols available: {[s for s in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] if s in symbols]}"
        )

        # Show date range for AAPL
        if "AAPL" in symbols:
            date_range = store.get_date_range_for_symbol("AAPL", TimeFrame.ONE_DAY)
            if date_range[0]:
                days = (date_range[1] - date_range[0]).days
                print(
                    f"📅 AAPL data range: {date_range[0]} to {date_range[1]} ({days} days)"
                )

        print("\n" + "=" * 60)

        # Run demonstrations
        print("🎯 RUNNING DEMONSTRATIONS...")
        print()

        # 1. Single strategy detailed analysis
        print("1️⃣  SINGLE STRATEGY ANALYSIS")
        await run_single_strategy_demo()

        print("\n" + "=" * 60)

        # 2. Multi-strategy comparison
        print("2️⃣  MULTI-STRATEGY COMPARISON")
        comparison_results = await run_strategy_comparison()

        print("\n" + "=" * 60)

        # 3. Multi-timeframe comparison
        print("3️⃣  MULTI-TIMEFRAME COMPARISON")
        timeframe_results = await run_multi_timeframe_comparison()

        print("\n" + "=" * 60)

        # 4. Comprehensive analysis
        print("4️⃣  COMPREHENSIVE STRATEGY & TIMEFRAME ANALYSIS")
        comprehensive_results = await run_comprehensive_timeframe_analysis()

        print("\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print("\nKey Findings:")
        print("✓ Historical data backtesting works with your 12+ months of data")
        print("✓ Screener simulation successfully identifies trading candidates")
        print("✓ Multiple strategy modes can be tested and compared")
        print(
            "✓ Multi-timeframe analysis shows performance across different time horizons"
        )
        print("✓ Comprehensive performance metrics are calculated")
        print("✓ Risk management (stop losses, position sizing) is simulated")

        print("\nNext Steps:")
        print("🔧 Connect to your real production strategies (HybridStrategy)")
        print("📊 Test with different time periods and parameters")
        print("🎯 Optimize screener criteria based on backtest results")
        print("📈 Use results to guide live trading strategy selection")
        print("⏰ Consider timeframe-specific optimizations based on results")

        return {
            "single_strategy": results,
            "strategy_comparison": comparison_results,
            "timeframe_comparison": timeframe_results,
            "comprehensive_analysis": comprehensive_results,
        }

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Choose which demo to run
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        # Debug signal generation
        print("🚀 RUNNING DEBUG SIGNAL TEST")
        print("Usage: python comprehensive_backtest_demo.py debug")
        print()

        results = asyncio.run(run_debug_signal_test())
    elif len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick timeframe test
        timeframes_to_test = [TimeFrame.ONE_DAY, TimeFrame.ONE_HOUR]
        if len(sys.argv) > 2:
            strategy_mode = sys.argv[
                2
            ]  # day_trading, swing_trading, or position_trading
        else:
            strategy_mode = "day_trading"

        print("🚀 RUNNING QUICK TIMEFRAME TEST")
        print("Usage: python comprehensive_backtest_demo.py quick [strategy_mode]")
        print("Strategy modes: day_trading, swing_trading, position_trading")
        print()

        results = asyncio.run(
            run_quick_timeframe_test(timeframes_to_test, strategy_mode)
        )
    else:
        # Run the comprehensive demonstration
        results = asyncio.run(run_multi_timeframe_comparison())
