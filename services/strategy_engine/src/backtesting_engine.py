"""
Backtesting Engine Module

This module provides comprehensive backtesting capabilities including walk-forward
analysis, parameter optimization, detailed performance metrics, and risk analysis.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from shared.models import FinVizData, SignalType

from .base_strategy import BacktestMetrics, BaseStrategy
from .hybrid_strategy import HybridStrategy


class OptimizationMethod(Enum):
    """Parameter optimization methods."""

    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    WALK_FORWARD = "walk_forward"
    GENETIC_ALGORITHM = "genetic"


class BacktestMode(Enum):
    """Backtesting execution modes."""

    SIMPLE = "simple"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    BOOTSTRAP = "bootstrap"


@dataclass
class Trade:
    """Individual trade record."""

    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percentage: float
    commission: float
    duration_hours: float
    exit_reason: str
    strategy_name: str
    signal_confidence: float
    max_favorable_excursion: float  # MFE
    max_adverse_excursion: float  # MAE


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time."""

    timestamp: datetime
    total_equity: float
    cash: float
    positions_value: float
    open_positions: int
    daily_pnl: float
    drawdown: float
    exposure: float  # Market exposure percentage


@dataclass
class BacktestConfig:
    """Backtesting configuration parameters."""

    initial_capital: float = 100000.0
    commission_per_trade: float = 1.0
    commission_percentage: float = 0.001  # 0.1% commission
    slippage_percentage: float = 0.0005  # 0.05% slippage
    max_positions: int = 10
    margin_requirement: float = 1.0  # 1.0 = no margin, 0.5 = 2:1 leverage
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    benchmark_symbol: str = "SPY"
    enable_short_selling: bool = False
    max_drawdown_stop: float = 0.20  # Stop trading at 20% drawdown
    daily_trade_limit: int = 100
    position_size_method: str = "fixed"  # "fixed", "volatility", "kelly"


class DetailedBacktestResult:
    """Comprehensive backtest results with detailed analytics."""

    def __init__(self):
        """Initialize detailed backtest result."""
        self.config: Optional[BacktestConfig] = None
        self.strategy_name: str = ""
        self.symbol: str = ""
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None

        # Core metrics
        self.metrics: Optional[BacktestMetrics] = None

        # Trade records
        self.trades: List[Trade] = []
        self.portfolio_snapshots: List[PortfolioSnapshot] = []

        # Performance analytics
        self.monthly_returns: Dict[str, float] = {}
        self.yearly_returns: Dict[str, float] = {}
        self.rolling_metrics: Dict[str, List[float]] = {}

        # Risk analytics
        self.var_95: float = 0.0  # Value at Risk 95%
        self.cvar_95: float = 0.0  # Conditional Value at Risk 95%
        self.tail_ratio: float = 0.0
        self.gain_to_pain_ratio: float = 0.0

        # Efficiency metrics
        self.information_ratio: float = 0.0
        self.treynor_ratio: float = 0.0
        self.jensen_alpha: float = 0.0
        self.tracking_error: float = 0.0

        # Trade analytics
        self.trade_distribution: Dict[str, Any] = {}
        self.holding_period_analysis: Dict[str, Any] = {}
        self.consecutive_wins: int = 0
        self.consecutive_losses: int = 0
        self.max_consecutive_wins: int = 0
        self.max_consecutive_losses: int = 0

        # Signal quality metrics
        self.signal_accuracy: Dict[str, float] = {}
        self.confidence_calibration: Dict[str, Any] = {}

        # Additional metadata
        self.metadata: Dict[str, Any] = {}


class BacktestingEngine:
    """High-performance backtesting engine with advanced analytics."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtesting engine.

        Args:
            config: Backtesting configuration
        """
        self.config = config if config is not None else BacktestConfig()
        self.logger = logging.getLogger("backtesting_engine")

        # Performance tracking
        self._trade_id_counter = 0
        self._cache: Dict[str, Any] = {}

    async def backtest_strategy(
        self,
        strategy: BaseStrategy,
        symbol: str,
        data: pl.DataFrame,
        start_date: datetime,
        end_date: datetime,
        finviz_data: Optional[FinVizData] = None,
        mode: BacktestMode = BacktestMode.SIMPLE,
    ) -> DetailedBacktestResult:
        """
        Run comprehensive backtest of strategy.

        Args:
            strategy: Strategy to backtest
            symbol: Trading symbol
            data: Historical market data
            start_date: Backtest start date
            end_date: Backtest end date
            finviz_data: Fundamental data (for hybrid strategies)
            mode: Backtesting mode

        Returns:
            Detailed backtest results
        """
        try:
            self.logger.info(
                f"Starting {mode.value} backtest for {strategy.name} on {symbol}"
            )

            # Initialize result container
            result = DetailedBacktestResult()
            result.config = self.config
            result.strategy_name = strategy.name
            result.symbol = symbol
            result.start_date = start_date
            result.end_date = end_date

            # Filter data to backtest period
            test_data = data.filter(
                (pl.col("timestamp") >= start_date) & (pl.col("timestamp") <= end_date)
            ).sort("timestamp")

            if test_data.height == 0:
                raise ValueError("No data available for backtest period")

            # Execute backtest based on mode
            if mode == BacktestMode.SIMPLE:
                await self._run_simple_backtest(
                    strategy, result, test_data, finviz_data
                )
            elif mode == BacktestMode.WALK_FORWARD:
                await self._run_walk_forward_backtest(
                    strategy, result, test_data, finviz_data
                )
            elif mode == BacktestMode.MONTE_CARLO:
                await self._run_monte_carlo_backtest(
                    strategy, result, test_data, finviz_data
                )
            else:
                await self._run_simple_backtest(
                    strategy, result, test_data, finviz_data
                )

            # Calculate comprehensive metrics
            self._calculate_detailed_metrics(result)

            # Analyze trade patterns
            self._analyze_trade_patterns(result)

            # Calculate risk metrics
            self._calculate_risk_metrics(result)

            # Signal quality analysis
            self._analyze_signal_quality(result)

            if result.metrics and result.metrics.total_return is not None:
                self.logger.info(
                    f"Backtest completed: {len(result.trades)} trades, "
                    f"{result.metrics.total_return:.2%} return"
                )
            else:
                self.logger.info(f"Backtest completed: {len(result.trades)} trades")

            return result

        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            raise

    async def _run_simple_backtest(
        self,
        strategy: BaseStrategy,
        result: DetailedBacktestResult,
        data: pl.DataFrame,
        finviz_data: Optional[FinVizData],
    ) -> None:
        """Run simple backtest with full data available."""
        try:
            # Initialize state
            portfolio = self._initialize_portfolio()
            current_positions: Dict[str, Any] = {}

            # Sliding window for signal generation
            window_size = strategy.config.lookback_period

            for i in range(window_size, data.height):
                current_time = data.slice(i, 1).select("timestamp").item()
                current_row = data.slice(i, 1)
                current_price = float(current_row.select("close").item())

                # Get data window for analysis
                window_data = data.slice(i - window_size, window_size + 1)

                try:
                    # Generate signal
                    if isinstance(strategy, HybridStrategy):
                        signal = await strategy.analyze(
                            result.symbol, window_data, finviz_data
                        )
                    else:
                        signal_result = await strategy.analyze(
                            result.symbol, window_data
                        )
                        # Convert Signal to HybridSignal if needed
                        if hasattr(signal_result, "signal_type"):
                            signal = signal_result  # type: ignore[assignment]
                        else:
                            from hybrid_strategy import HybridSignal

                            signal = HybridSignal(
                                action=signal_result.action,
                                confidence=signal_result.confidence,
                                position_size=getattr(
                                    signal_result, "position_size", 0.1
                                ),
                                reasoning=getattr(signal_result, "reasoning", ""),
                                timestamp=signal_result.timestamp,
                            )

                    # Skip if confidence too low
                    if signal.confidence < strategy.config.min_confidence:
                        continue

                    # Process signal
                    await self._process_signal(
                        signal,
                        result.symbol,
                        current_time,
                        current_price,
                        portfolio,
                        current_positions,
                        result,
                    )

                    # Update portfolio snapshot
                    self._update_portfolio_snapshot(
                        portfolio,
                        current_positions,
                        current_time,
                        current_price,
                        result,
                    )

                except Exception as e:
                    self.logger.error(f"Signal processing error at {current_time}: {e}")
                    continue

            # Close remaining positions
            await self._close_remaining_positions(
                current_positions, data, portfolio, result
            )

        except Exception as e:
            self.logger.error(f"Simple backtest error: {e}")
            raise

    async def _run_monte_carlo_backtest(
        self,
        strategy: BaseStrategy,
        result: DetailedBacktestResult,
        data: pl.DataFrame,
        finviz_data: Optional[FinVizData],
    ) -> None:
        """Run Monte Carlo backtest with randomized market conditions."""
        try:
            # Monte Carlo simulation parameters
            num_simulations = getattr(self.config, "monte_carlo_simulations", 100)

            # Run multiple simulations with bootstrap sampling
            simulation_results = []

            for sim in range(num_simulations):
                # Bootstrap sampling of the data
                n_samples = len(data)
                bootstrap_indices = np.random.choice(
                    n_samples, size=n_samples, replace=True
                )
                bootstrap_data = data[bootstrap_indices]

                # Run backtest on bootstrap sample
                sim_result = DetailedBacktestResult()
                await self._run_simple_backtest(
                    strategy, sim_result, bootstrap_data, finviz_data
                )

                if sim_result.metrics:
                    simulation_results.append(sim_result.metrics)

            # Aggregate Monte Carlo results
            if simulation_results:
                # Calculate mean and confidence intervals
                total_returns = [
                    r.total_return
                    for r in simulation_results
                    if r.total_return is not None
                ]
                sharpe_ratios = [
                    r.sharpe_ratio
                    for r in simulation_results
                    if r.sharpe_ratio is not None
                ]
                max_drawdowns = [
                    r.max_drawdown
                    for r in simulation_results
                    if r.max_drawdown is not None
                ]

                if total_returns:
                    result.metadata["monte_carlo"] = {
                        "num_simulations": num_simulations,
                        "total_return_mean": np.mean(total_returns),
                        "total_return_std": np.std(total_returns),
                        "total_return_95_ci": np.percentile(total_returns, [2.5, 97.5]),
                        "sharpe_ratio_mean": (
                            np.mean(sharpe_ratios) if sharpe_ratios else 0
                        ),
                        "max_drawdown_mean": (
                            np.mean(max_drawdowns) if max_drawdowns else 0
                        ),
                    }

            # Use the first simulation as the primary result
            if simulation_results:
                result.metrics = simulation_results[0]

        except Exception as e:
            self.logger.error(f"Monte Carlo backtest error: {e}")
            raise

    async def _run_walk_forward_backtest(
        self,
        strategy: BaseStrategy,
        result: DetailedBacktestResult,
        data: pl.DataFrame,
        finviz_data: Optional[FinVizData],
    ) -> None:
        """Run walk-forward backtest with parameter optimization."""
        try:
            # Walk-forward parameters
            optimization_window = 252  # 1 year
            out_of_sample_window = 63  # 3 months
            min_trades_for_optimization = 10

            total_periods = (data.height - optimization_window) // out_of_sample_window

            if total_periods < 1:
                self.logger.warning(
                    "Insufficient data for walk-forward analysis, falling back to simple backtest"
                )
                await self._run_simple_backtest(strategy, result, data, finviz_data)
                return

            # Initialize combined results
            all_trades: List[Any] = []
            all_snapshots = []
            optimization_results = []

            for period in range(total_periods):
                start_idx = period * out_of_sample_window
                opt_end_idx = start_idx + optimization_window
                test_end_idx = min(opt_end_idx + out_of_sample_window, data.height)

                # Optimization period data
                opt_data = data.slice(start_idx, optimization_window)

                # Out-of-sample test data
                test_data = data.slice(opt_end_idx, test_end_idx - opt_end_idx)

                if test_data.height == 0:
                    continue

                self.logger.info(f"Walk-forward period {period + 1}/{total_periods}")

                # Optimize parameters on in-sample data
                if len(all_trades) >= min_trades_for_optimization:
                    optimal_params = await self._optimize_parameters(
                        strategy, result.symbol, opt_data, finviz_data
                    )

                    # Update strategy with optimal parameters
                    strategy.update_config(optimal_params)
                    optimization_results.append(
                        {
                            "period": period,
                            "optimal_params": optimal_params,
                            "optimization_data_points": opt_data.height,
                        }
                    )

                # Test on out-of-sample data
                period_result = DetailedBacktestResult()
                period_result.config = self.config
                period_result.strategy_name = strategy.name
                period_result.symbol = result.symbol

                await self._run_simple_backtest(
                    strategy, period_result, test_data, finviz_data
                )

                # Combine results
                all_trades.extend(period_result.trades)
                all_snapshots.extend(period_result.portfolio_snapshots)

            # Update main result
            result.trades = all_trades
            result.portfolio_snapshots = all_snapshots
            result.metadata = {
                "walk_forward_periods": total_periods,
                "optimization_results": optimization_results,
            }

        except Exception as e:
            self.logger.error(f"Walk-forward backtest error: {e}")
            raise

    async def _optimize_parameters(
        self,
        strategy: BaseStrategy,
        symbol: str,
        data: pl.DataFrame,
        finviz_data: Optional[FinVizData],
    ) -> Dict[str, Any]:
        """Optimize strategy parameters using historical data."""
        try:
            # Define parameter ranges for optimization
            param_ranges = self._get_optimization_ranges(strategy)

            if not param_ranges:
                return {}

            best_params = {}
            best_score = -float("inf")

            # Grid search optimization
            param_combinations = list(product(*param_ranges.values()))
            max_combinations = min(50, len(param_combinations))  # Limit for performance

            if len(param_combinations) > max_combinations:
                # Random sampling if too many combinations
                import random

                param_combinations = random.sample(param_combinations, max_combinations)

            for combination in param_combinations:
                # Create parameter dict
                test_params = dict(zip(param_ranges.keys(), combination))

                # Create test strategy
                test_strategy: Union[BaseStrategy, HybridStrategy] = (
                    self._create_test_strategy(strategy, test_params)
                )

                # Quick backtest
                quick_result = await self._quick_backtest(
                    test_strategy, symbol, data, finviz_data
                )

                # Score based on risk-adjusted return
                score = self._calculate_optimization_score(quick_result)

                if score > best_score:
                    best_score = score
                    best_params = test_params

            self.logger.info(
                f"Parameter optimization completed. Best score: {best_score:.3f}"
            )
            return best_params

        except Exception as e:
            self.logger.error(f"Parameter optimization error: {e}")
            return {}

    def _get_optimization_ranges(self, strategy: BaseStrategy) -> Dict[str, List[Any]]:
        """Get parameter ranges for optimization based on strategy type."""
        try:
            if isinstance(strategy, HybridStrategy):
                return {
                    "ta_base_weight": [0.3, 0.5, 0.7],
                    "min_technical_score": [40, 50, 60],
                    "min_fundamental_score": [40, 50, 60],
                    "volume_threshold": [1.2, 1.5, 2.0],
                }
            else:
                # Default technical strategy ranges
                return {
                    "sma_short": [10, 15, 20, 25],
                    "sma_long": [40, 50, 60],
                    "rsi_period": [10, 14, 18],
                    "rsi_oversold": [25, 30, 35],
                    "rsi_overbought": [65, 70, 75],
                }

        except Exception:
            return {}

    def _create_test_strategy(
        self, base_strategy: BaseStrategy, test_params: Dict[str, Any]
    ) -> Union[BaseStrategy, HybridStrategy]:
        """Create a test strategy with modified parameters."""
        try:
            # Clone strategy configuration
            test_config = base_strategy.config
            if test_config.parameters is not None:
                test_config.parameters.update(test_params)

            # Create new strategy instance
            if isinstance(base_strategy, HybridStrategy):
                test_strategy = HybridStrategy(test_config, base_strategy.hybrid_mode)
                return test_strategy
            else:
                new_test_strategy: BaseStrategy = base_strategy.__class__(test_config)
                return new_test_strategy

        except Exception as e:
            self.logger.error(f"Error creating test strategy: {e}")
            return base_strategy

    async def _quick_backtest(
        self,
        strategy: BaseStrategy,
        symbol: str,
        data: pl.DataFrame,
        finviz_data: Optional[FinVizData],
    ) -> Dict[str, Any]:
        """Run quick backtest for parameter optimization."""
        try:
            # Simplified backtest for speed
            position = None
            trades = []

            window_size = strategy.config.lookback_period

            for i in range(
                window_size, min(data.height, window_size + 100)
            ):  # Limit for speed
                window_data = data.slice(i - window_size, window_size + 1)
                current_price = float(data.slice(i, 1).select("close").item())
                current_time = data.slice(i, 1).select("timestamp").item()

                # Generate signal
                try:
                    if isinstance(strategy, HybridStrategy):
                        signal = await strategy.analyze(
                            symbol, window_data, finviz_data
                        )
                    else:
                        signal_result = await strategy.analyze(symbol, window_data)
                        signal = signal_result  # type: ignore
                except Exception:
                    continue

                # Simple trade execution logic
                if position is None and signal.action in [
                    SignalType.BUY,
                    SignalType.SELL,
                ]:
                    position = {
                        "entry_price": current_price,
                        "entry_time": current_time,
                        "side": "long" if signal.action == SignalType.BUY else "short",
                        "quantity": 100,  # Fixed quantity for speed
                    }
                elif position and signal.action == SignalType.CLOSE:
                    # Close position
                    pnl = self._calculate_quick_pnl(position, current_price)
                    trades.append(
                        {
                            "pnl": pnl,
                            "duration": (
                                current_time - position["entry_time"]
                            ).total_seconds()
                            / 3600,
                        }
                    )
                    position = None

            # Calculate quick metrics
            if trades:
                total_pnl = sum(t["pnl"] for t in trades)
                win_rate = len([t for t in trades if t["pnl"] > 0]) / len(trades)
                avg_duration = sum(t["duration"] for t in trades) / len(trades)
            else:
                total_pnl = 0
                win_rate = 0
                avg_duration = 0

            return {
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "num_trades": len(trades),
                "avg_duration": avg_duration,
            }

        except Exception as e:
            self.logger.error(f"Quick backtest error: {e}")
            return {"total_pnl": 0, "win_rate": 0, "num_trades": 0}

    def _calculate_quick_pnl(self, position: Dict, exit_price: float) -> float:
        """Calculate quick P&L for optimization."""
        entry_price = position["entry_price"]
        quantity = position["quantity"]

        if position["side"] == "long":
            return (exit_price - entry_price) * quantity
        else:
            return (entry_price - exit_price) * quantity

    def _calculate_optimization_score(self, quick_result: Dict[str, Any]) -> float:
        """Calculate optimization score for parameter selection."""
        try:
            total_pnl = quick_result.get("total_pnl", 0)
            win_rate = quick_result.get("win_rate", 0)
            num_trades = quick_result.get("num_trades", 0)

            # Penalize too few trades
            if num_trades < 5:
                return -1000

            # Risk-adjusted score
            if num_trades > 0:
                avg_pnl = total_pnl / num_trades
                # Simple score: average PnL * win rate * sqrt(num_trades)
                score = avg_pnl * win_rate * np.sqrt(num_trades)
            else:
                score = -1000

            return score

        except Exception:
            return -1000

    async def _process_signal(
        self,
        signal: Any,
        symbol: str,
        timestamp: datetime,
        current_price: float,
        portfolio: Dict,
        positions: Dict,
        result: DetailedBacktestResult,
    ) -> None:
        """Process trading signal and execute trades."""
        try:
            # Check if we can trade
            if not self._can_trade(portfolio, timestamp):
                return

            # Handle entry signals
            if (
                signal.action in [SignalType.BUY, SignalType.SELL]
                and symbol not in positions
            ):

                await self._enter_position(
                    signal,
                    symbol,
                    timestamp,
                    current_price,
                    portfolio,
                    positions,
                    result,
                )

            # Handle exit signals
            elif signal.action == SignalType.CLOSE and symbol in positions:
                await self._exit_position(
                    symbol,
                    timestamp,
                    current_price,
                    portfolio,
                    positions,
                    result,
                    "signal_exit",
                )

            # Check stop loss and take profit for existing positions
            if symbol in positions:
                await self._check_exit_conditions(
                    symbol, timestamp, current_price, portfolio, positions, result
                )

        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")

    async def _enter_position(
        self,
        signal: Any,
        symbol: str,
        timestamp: datetime,
        price: float,
        portfolio: Dict,
        positions: Dict,
        result: DetailedBacktestResult,
    ) -> None:
        """Enter a new position."""
        try:
            # Check position limits
            if len(positions) >= self.config.max_positions:
                return

            # Calculate position size
            if hasattr(signal, "position_size"):
                size_fraction = signal.position_size
            else:
                size_fraction = 0.1  # Default 10%

            position_value = portfolio["cash"] * size_fraction
            commission = max(
                self.config.commission_per_trade,
                position_value * self.config.commission_percentage,
            )

            # Apply slippage
            slippage = price * self.config.slippage_percentage
            if signal.action == SignalType.BUY:
                actual_price = price + slippage
            else:
                actual_price = price - slippage

            quantity = (position_value - commission) / actual_price

            if quantity <= 0:
                return

            # Create position
            positions[symbol] = {
                "entry_time": timestamp,
                "entry_price": actual_price,
                "quantity": quantity,
                "side": "long" if signal.action == SignalType.BUY else "short",
                "stop_loss": (
                    float(signal.stop_loss)
                    if hasattr(signal, "stop_loss") and signal.stop_loss
                    else None
                ),
                "take_profit": (
                    float(signal.take_profit)
                    if hasattr(signal, "take_profit") and signal.take_profit
                    else None
                ),
                "signal_confidence": getattr(signal, "confidence", 50.0),
                "strategy_name": getattr(signal, "strategy_name", result.strategy_name),
                "max_favorable": 0.0,
                "max_adverse": 0.0,
            }

            # Update portfolio
            portfolio["cash"] -= position_value + commission
            portfolio["positions_value"] += position_value

            self.logger.debug(
                f"Entered {signal.action.value} position for {symbol} at {actual_price}"
            )

        except Exception as e:
            self.logger.error(f"Error entering position: {e}")

    async def _exit_position(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        portfolio: Dict,
        positions: Dict,
        result: DetailedBacktestResult,
        exit_reason: str,
    ) -> None:
        """Exit an existing position."""
        try:
            if symbol not in positions:
                return

            position = positions[symbol]

            # Apply slippage
            slippage = price * self.config.slippage_percentage
            if position["side"] == "long":
                actual_price = price - slippage
            else:
                actual_price = price + slippage

            # Calculate P&L
            if position["side"] == "long":
                pnl = (actual_price - position["entry_price"]) * position["quantity"]
            else:
                pnl = (position["entry_price"] - actual_price) * position["quantity"]

            # Commission
            trade_value = actual_price * position["quantity"]
            commission = max(
                self.config.commission_per_trade,
                trade_value * self.config.commission_percentage,
            )

            net_pnl = pnl - commission
            pnl_percentage = net_pnl / (position["entry_price"] * position["quantity"])

            # Duration
            duration = timestamp - position["entry_time"]
            duration_hours = duration.total_seconds() / 3600

            # Create trade record
            trade = Trade(
                entry_time=position["entry_time"],
                exit_time=timestamp,
                symbol=symbol,
                side=position["side"],
                entry_price=position["entry_price"],
                exit_price=actual_price,
                quantity=position["quantity"],
                pnl=net_pnl,
                pnl_percentage=pnl_percentage,
                commission=commission,
                duration_hours=duration_hours,
                exit_reason=exit_reason,
                strategy_name=position["strategy_name"],
                signal_confidence=position["signal_confidence"],
                max_favorable_excursion=position["max_favorable"],
                max_adverse_excursion=position["max_adverse"],
            )

            result.trades.append(trade)

            # Update portfolio
            portfolio["cash"] += trade_value - commission
            portfolio["positions_value"] -= (
                position["entry_price"] * position["quantity"]
            )

            # Remove position
            del positions[symbol]

            self.logger.debug(
                f"Exited {symbol} position: {exit_reason}, PnL: {net_pnl:.2f}"
            )

        except Exception as e:
            self.logger.error(f"Error exiting position: {e}")

    async def _check_exit_conditions(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        portfolio: Dict,
        positions: Dict,
        result: DetailedBacktestResult,
    ) -> None:
        """Check stop loss and take profit conditions."""
        try:
            if symbol not in positions:
                return

            position = positions[symbol]

            # Update max favorable and adverse excursions
            if position["side"] == "long":
                excursion = (price - position["entry_price"]) / position["entry_price"]
            else:
                excursion = (position["entry_price"] - price) / position["entry_price"]

            position["max_favorable"] = max(position["max_favorable"], excursion)
            position["max_adverse"] = min(position["max_adverse"], excursion)

            # Check stop loss
            if position["stop_loss"]:
                if (position["side"] == "long" and price <= position["stop_loss"]) or (
                    position["side"] == "short" and price >= position["stop_loss"]
                ):
                    await self._exit_position(
                        symbol,
                        timestamp,
                        position["stop_loss"],
                        portfolio,
                        positions,
                        result,
                        "stop_loss",
                    )
                    return

            # Check take profit
            if position["take_profit"]:
                if (
                    position["side"] == "long" and price >= position["take_profit"]
                ) or (position["side"] == "short" and price <= position["take_profit"]):
                    await self._exit_position(
                        symbol,
                        timestamp,
                        position["take_profit"],
                        portfolio,
                        positions,
                        result,
                        "take_profit",
                    )
                    return

        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")

    def _initialize_portfolio(self) -> Dict[str, Any]:
        """Initialize portfolio state."""
        return {
            "initial_capital": self.config.initial_capital,
            "cash": self.config.initial_capital,
            "positions_value": 0.0,
            "total_equity": self.config.initial_capital,
            "peak_equity": self.config.initial_capital,
            "drawdown": 0.0,
            "daily_trades": 0,
            "last_trade_date": None,
        }

    def _can_trade(self, portfolio: Dict, timestamp: datetime) -> bool:
        """Check if trading is allowed based on rules."""
        try:
            # Check daily trade limit
            if (
                portfolio.get("last_trade_date") == timestamp.date()
                and portfolio.get("daily_trades", 0) >= self.config.daily_trade_limit
            ):
                return False

            # Check drawdown stop
            if portfolio["drawdown"] >= self.config.max_drawdown_stop:
                return False

            # Check if market is open (simplified - would use actual market hours)
            hour = timestamp.hour
            if hour < 9 or hour >= 16:  # Outside 9 AM - 4 PM
                return False

            return True

        except Exception:
            return False

    def _update_portfolio_snapshot(
        self,
        portfolio: Dict,
        positions: Dict,
        timestamp: datetime,
        current_price: float,
        result: DetailedBacktestResult,
    ) -> None:
        """Update portfolio snapshot for equity curve."""
        try:
            # Calculate current positions value
            total_positions_value = 0.0
            for pos in positions.values():
                if pos["side"] == "long":
                    position_value = pos["quantity"] * current_price
                else:
                    # Short position value
                    position_value = pos["quantity"] * (
                        2 * pos["entry_price"] - current_price
                    )
                total_positions_value += position_value

            # Update portfolio
            portfolio["positions_value"] = total_positions_value
            portfolio["total_equity"] = portfolio["cash"] + total_positions_value

            # Update peak and drawdown
            if portfolio["total_equity"] > portfolio["peak_equity"]:
                portfolio["peak_equity"] = portfolio["total_equity"]
                portfolio["drawdown"] = 0.0
            else:
                portfolio["drawdown"] = (
                    portfolio["peak_equity"] - portfolio["total_equity"]
                ) / portfolio["peak_equity"]

            # Daily P&L calculation
            daily_pnl = 0.0
            if result.portfolio_snapshots:
                last_snapshot = result.portfolio_snapshots[-1]
                if last_snapshot.timestamp.date() == timestamp.date():
                    daily_pnl = portfolio["total_equity"] - last_snapshot.total_equity
                else:
                    daily_pnl = 0.0

            # Create snapshot
            snapshot = PortfolioSnapshot(
                timestamp=timestamp,
                total_equity=portfolio["total_equity"],
                cash=portfolio["cash"],
                positions_value=total_positions_value,
                open_positions=len(positions),
                daily_pnl=daily_pnl,
                drawdown=portfolio["drawdown"],
                exposure=(
                    total_positions_value / portfolio["total_equity"]
                    if portfolio["total_equity"] > 0
                    else 0.0
                ),
            )

            result.portfolio_snapshots.append(snapshot)

        except Exception as e:
            self.logger.error(f"Error updating portfolio snapshot: {e}")

    async def _close_remaining_positions(
        self,
        positions: Dict,
        data: pl.DataFrame,
        portfolio: Dict,
        result: DetailedBacktestResult,
    ) -> None:
        """Close any remaining positions at backtest end."""
        try:
            if not positions:
                return

            final_time = data.select("timestamp").tail(1).item()
            final_price = float(data.select("close").tail(1).item())

            for symbol in list(positions.keys()):
                await self._exit_position(
                    symbol,
                    final_time,
                    final_price,
                    portfolio,
                    positions,
                    result,
                    "backtest_end",
                )

        except Exception as e:
            self.logger.error(f"Error closing remaining positions: {e}")

    def _calculate_detailed_metrics(self, result: DetailedBacktestResult) -> None:
        """Calculate comprehensive performance metrics."""
        try:
            if not result.trades or not result.portfolio_snapshots:
                result.metrics = BacktestMetrics(
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
                return

            # Basic trade metrics
            winning_trades = [t for t in result.trades if t.pnl > 0]
            losing_trades = [t for t in result.trades if t.pnl <= 0]

            total_return = (
                result.portfolio_snapshots[-1].total_equity
                - self.config.initial_capital
            ) / self.config.initial_capital

            win_rate = len(winning_trades) / len(result.trades)

            # P&L metrics
            total_profit = sum(t.pnl for t in winning_trades)
            total_loss = sum(abs(t.pnl) for t in losing_trades)
            profit_factor = (
                total_profit / total_loss if total_loss > 0 else float("inf")
            )

            avg_win = (
                sum(t.pnl for t in winning_trades) / len(winning_trades)
                if winning_trades
                else 0
            )
            avg_loss = (
                sum(t.pnl for t in losing_trades) / len(losing_trades)
                if losing_trades
                else 0
            )

            largest_win = max(t.pnl for t in winning_trades) if winning_trades else 0
            largest_loss = min(t.pnl for t in losing_trades) if losing_trades else 0

            # Duration metrics
            avg_duration = sum(t.duration_hours for t in result.trades) / len(
                result.trades
            )

            # Equity curve analysis
            equity_values = [s.total_equity for s in result.portfolio_snapshots]
            equity_returns = pd.Series(equity_values).pct_change().dropna()

            # Time-based metrics
            start_time = result.portfolio_snapshots[0].timestamp
            end_time = result.portfolio_snapshots[-1].timestamp
            years = (end_time - start_time).days / 365.25

            # Annualized return
            annualized_return = (
                (
                    result.portfolio_snapshots[-1].total_equity
                    / self.config.initial_capital
                )
                ** (1 / years)
                - 1
                if years > 0
                else 0
            )

            # Volatility (annualized)
            volatility = (
                equity_returns.std() * np.sqrt(252) if len(equity_returns) > 1 else 0
            )

            # Risk metrics
            sharpe_ratio = (
                (annualized_return - self.config.risk_free_rate) / volatility
                if volatility > 0
                else 0
            )

            # Maximum drawdown
            peak_values = pd.Series(equity_values).expanding().max()
            drawdowns = (pd.Series(equity_values) - peak_values) / peak_values
            max_drawdown = abs(drawdowns.min())

            # Calmar ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

            # Sortino ratio
            negative_returns = equity_returns[equity_returns < 0]
            downside_deviation = (
                negative_returns.std() * np.sqrt(252)
                if len(negative_returns) > 1
                else 0
            )
            sortino_ratio = (
                (annualized_return - self.config.risk_free_rate) / downside_deviation
                if downside_deviation > 0
                else 0
            )

            result.metrics = BacktestMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=len(result.trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                average_win=avg_win,
                average_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                avg_trade_duration=avg_duration / 24,  # Convert to days
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
            )

        except Exception as e:
            self.logger.error(f"Error calculating detailed metrics: {e}")
            result.metrics = BacktestMetrics(
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

    def _analyze_trade_patterns(self, result: DetailedBacktestResult) -> None:
        """Analyze trading patterns and behavior."""
        try:
            if not result.trades:
                return

            # Trade distribution analysis
            pnl_values = [t.pnl for t in result.trades]
            result.trade_distribution = {
                "pnl_mean": np.mean(pnl_values),
                "pnl_std": np.std(pnl_values),
                "pnl_skewness": pd.Series(pnl_values).skew(),
                "pnl_kurtosis": pd.Series(pnl_values).kurtosis(),
                "percentiles": {
                    "p25": np.percentile(pnl_values, 25),
                    "p50": np.percentile(pnl_values, 50),
                    "p75": np.percentile(pnl_values, 75),
                    "p90": np.percentile(pnl_values, 90),
                    "p95": np.percentile(pnl_values, 95),
                },
            }

            # Holding period analysis
            durations = [t.duration_hours for t in result.trades]
            result.holding_period_analysis = {
                "avg_duration_hours": np.mean(durations),
                "median_duration_hours": np.median(durations),
                "min_duration_hours": min(durations),
                "max_duration_hours": max(durations),
                "duration_std": np.std(durations),
            }

            # Consecutive wins/losses
            consecutive_results = [1 if t.pnl > 0 else -1 for t in result.trades]
            result.consecutive_wins, result.consecutive_losses = (
                self._calculate_consecutive_streaks(consecutive_results)
            )
            result.max_consecutive_wins = max(result.consecutive_wins, 0)
            result.max_consecutive_losses = max(result.consecutive_losses, 0)

        except Exception as e:
            self.logger.error(f"Error analyzing trade patterns: {e}")

    def _calculate_consecutive_streaks(self, results: List[int]) -> Tuple[int, int]:
        """Calculate consecutive win/loss streaks."""
        if not results:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for result in results:
            if result > 0:  # Win
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:  # Loss
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def _calculate_risk_metrics(self, result: DetailedBacktestResult) -> None:
        """Calculate advanced risk metrics."""
        try:
            if not result.portfolio_snapshots:
                return

            equity_values = [s.total_equity for s in result.portfolio_snapshots]
            returns = pd.Series(equity_values).pct_change().dropna()

            if len(returns) < 10:
                return

            # Value at Risk (95%)
            result.var_95 = float(np.percentile(returns, 5))

            # Conditional Value at Risk (95%)
            var_threshold = result.var_95
            tail_returns = returns[returns <= var_threshold]
            result.cvar_95 = (
                float(tail_returns.mean()) if len(tail_returns) > 0 else 0.0
            )

            # Tail ratio
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                result.tail_ratio = float(
                    np.percentile(positive_returns, 95)
                    / abs(np.percentile(negative_returns, 5))
                )
            else:
                result.tail_ratio = 0.0

            # Gain to Pain ratio
            positive_sum = positive_returns.sum() if len(positive_returns) > 0 else 0
            negative_sum = (
                abs(negative_returns.sum()) if len(negative_returns) > 0 else 1
            )
            result.gain_to_pain_ratio = positive_sum / negative_sum

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")

    def _analyze_signal_quality(self, result: DetailedBacktestResult) -> None:
        """Analyze quality and accuracy of trading signals."""
        try:
            if not result.trades:
                return

            # Signal accuracy by confidence buckets
            confidence_buckets = {
                "low": [t for t in result.trades if t.signal_confidence < 60],
                "medium": [t for t in result.trades if 60 <= t.signal_confidence < 80],
                "high": [t for t in result.trades if t.signal_confidence >= 80],
            }

            result.signal_accuracy = {}
            for bucket, trades in confidence_buckets.items():
                if trades:
                    accuracy = len([t for t in trades if t.pnl > 0]) / len(trades)
                    result.signal_accuracy[bucket] = accuracy

            # Confidence calibration analysis
            confidence_ranges = [
                (0, 50),
                (50, 60),
                (60, 70),
                (70, 80),
                (80, 90),
                (90, 100),
            ]
            calibration_data = {}

            for min_conf, max_conf in confidence_ranges:
                bucket_trades = [
                    t
                    for t in result.trades
                    if min_conf <= t.signal_confidence < max_conf
                ]
                if bucket_trades:
                    actual_accuracy = len(
                        [t for t in bucket_trades if t.pnl > 0]
                    ) / len(bucket_trades)
                    expected_accuracy = (
                        min_conf + max_conf
                    ) / 200  # Convert to 0-1 range
                    calibration_data[f"{min_conf}-{max_conf}"] = {
                        "expected": expected_accuracy,
                        "actual": actual_accuracy,
                        "trades": len(bucket_trades),
                        "calibration_error": abs(actual_accuracy - expected_accuracy),
                    }

            result.confidence_calibration = calibration_data

        except Exception as e:
            self.logger.error(f"Error analyzing signal quality: {e}")

    def generate_backtest_report(
        self, result: DetailedBacktestResult
    ) -> Dict[str, Any]:
        """Generate comprehensive backtest report."""
        try:
            report = {
                "strategy_info": {
                    "name": result.strategy_name,
                    "symbol": result.symbol,
                    "start_date": (
                        result.start_date.isoformat() if result.start_date else None
                    ),
                    "end_date": (
                        result.end_date.isoformat() if result.end_date else None
                    ),
                    "duration_days": (
                        (result.end_date - result.start_date).days
                        if result.start_date and result.end_date
                        else 0
                    ),
                },
                "performance_metrics": asdict(result.metrics) if result.metrics else {},
                "risk_metrics": {
                    "var_95": result.var_95,
                    "cvar_95": result.cvar_95,
                    "tail_ratio": result.tail_ratio,
                    "gain_to_pain_ratio": result.gain_to_pain_ratio,
                },
                "trade_analysis": {
                    "total_trades": len(result.trades),
                    "trade_distribution": result.trade_distribution,
                    "holding_period": result.holding_period_analysis,
                    "consecutive_streaks": {
                        "max_wins": result.max_consecutive_wins,
                        "max_losses": result.max_consecutive_losses,
                    },
                },
                "signal_quality": {
                    "accuracy_by_confidence": result.signal_accuracy,
                    "confidence_calibration": result.confidence_calibration,
                },
                "configuration": asdict(result.config) if result.config else {},
            }

            # Add monthly/yearly breakdowns if available
            if result.monthly_returns:
                report["monthly_returns"] = result.monthly_returns
            if result.yearly_returns:
                report["yearly_returns"] = result.yearly_returns

            return report

        except Exception as e:
            self.logger.error(f"Error generating backtest report: {e}")
            return {"error": str(e)}

    async def optimize_strategy_parameters(
        self,
        strategy: BaseStrategy,
        symbol: str,
        data: pl.DataFrame,
        optimization_config: Dict[str, Any],
        finviz_data: Optional[FinVizData] = None,
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using various methods.

        Args:
            strategy: Strategy to optimize
            symbol: Trading symbol
            data: Historical data
            optimization_config: Optimization configuration
            finviz_data: Fundamental data

        Returns:
            Optimization results with best parameters
        """
        try:
            method = optimization_config.get("method", OptimizationMethod.GRID_SEARCH)
            objective = optimization_config.get("objective", "sharpe_ratio")

            self.logger.info(f"Starting parameter optimization using {method.value}")

            if method == OptimizationMethod.GRID_SEARCH:
                return await self._grid_search_optimization(
                    strategy, symbol, data, optimization_config, finviz_data, objective
                )
            elif method == OptimizationMethod.WALK_FORWARD:
                return await self._walk_forward_optimization(
                    strategy, symbol, data, optimization_config, finviz_data, objective
                )
            else:
                self.logger.warning(
                    f"Optimization method {method.value} not implemented, using grid search"
                )
                return await self._grid_search_optimization(
                    strategy, symbol, data, optimization_config, finviz_data, objective
                )

        except Exception as e:
            self.logger.error(f"Parameter optimization error: {e}")
            return {"error": str(e), "best_parameters": {}}

    async def _grid_search_optimization(
        self,
        strategy: BaseStrategy,
        symbol: str,
        data: pl.DataFrame,
        config: Dict[str, Any],
        finviz_data: Optional[FinVizData],
        objective: str,
    ) -> Dict[str, Any]:
        """Perform grid search parameter optimization."""
        try:
            param_ranges = self._get_optimization_ranges(strategy)
            if not param_ranges:
                return {"best_parameters": {}, "best_score": 0.0}

            # Split data for optimization
            train_size = int(len(data) * 0.7)
            train_data = data.slice(0, train_size)
            test_data = data.slice(train_size, len(data) - train_size)

            best_params = {}
            best_score = -float("inf")
            optimization_results = []

            # Generate parameter combinations
            param_combinations = list(product(*param_ranges.values()))
            max_combinations = config.get("max_combinations", 50)

            if len(param_combinations) > max_combinations:
                import random

                param_combinations = random.sample(param_combinations, max_combinations)

            self.logger.info(
                f"Testing {len(param_combinations)} parameter combinations"
            )

            # Test each combination
            for i, combination in enumerate(param_combinations):
                test_params = dict(zip(param_ranges.keys(), combination))

                try:
                    # Create test strategy
                    test_strategy: Union[BaseStrategy, HybridStrategy] = (
                        self._create_test_strategy(strategy, test_params)
                    )

                    # Quick backtest on training data
                    train_result = await self._quick_backtest(
                        test_strategy, symbol, train_data, finviz_data
                    )

                    # Score the result
                    score = self._score_optimization_result(train_result, objective)

                    optimization_results.append(
                        {
                            "parameters": test_params,
                            "score": score,
                            "metrics": train_result,
                        }
                    )

                    if score > best_score:
                        best_score = score
                        best_params = test_params

                    if (i + 1) % 10 == 0:
                        self.logger.info(
                            f"Completed {i + 1}/{len(param_combinations)} combinations"
                        )

                except Exception as e:
                    self.logger.warning(f"Optimization combination {i} failed: {e}")
                    continue

            # Validate best parameters on test data
            if best_params:
                validation_strategy: Union[BaseStrategy, HybridStrategy] = (
                    self._create_test_strategy(strategy, best_params)
                )
                validation_result = await self._quick_backtest(
                    validation_strategy, symbol, test_data, finviz_data
                )
                validation_score = self._score_optimization_result(
                    validation_result, objective
                )
            else:
                validation_score = 0.0

            return {
                "best_parameters": best_params,
                "best_score": best_score,
                "validation_score": validation_score,
                "optimization_results": sorted(
                    optimization_results,
                    key=lambda x: self._safe_float_convert(x.get("score", 0)),
                    reverse=True,
                )[:10],
                "total_combinations_tested": len(param_combinations),
            }

        except Exception as e:
            self.logger.error(f"Grid search optimization error: {e}")
            return {"error": str(e), "best_parameters": {}}

    async def _walk_forward_optimization(
        self,
        strategy: BaseStrategy,
        symbol: str,
        data: pl.DataFrame,
        config: Dict[str, Any],
        finviz_data: Optional[FinVizData],
        objective: str,
    ) -> Dict[str, Any]:
        """Perform walk-forward optimization with out-of-sample testing."""
        try:
            param_ranges = self._get_optimization_ranges(strategy)
            if not param_ranges:
                return {"best_parameters": {}, "best_score": 0.0}

            # Walk-forward parameters
            window_size = config.get("window_size", 252)  # 1 year
            step_size = config.get("step_size", 63)  # 3 months
            min_train_size = config.get("min_train_size", 126)  # 6 months

            walk_forward_results = []
            best_params = {}
            best_score = -float("inf")

            # Generate parameter combinations
            param_combinations = list(product(*param_ranges.values()))
            max_combinations = config.get("max_combinations", 20)

            if len(param_combinations) > max_combinations:
                import random

                param_combinations = random.sample(param_combinations, max_combinations)

            # Walk-forward analysis
            start_idx = min_train_size
            while start_idx + window_size < len(data):
                end_idx = min(start_idx + window_size, len(data))

                # Split into train and test
                train_data = data.slice(start_idx - min_train_size, min_train_size)
                test_data = data.slice(start_idx, end_idx - start_idx)

                # Optimize on training data
                period_best_params = {}
                period_best_score = -float("inf")

                for combination in param_combinations:
                    test_params = dict(zip(param_ranges.keys(), combination))
                    test_strategy: Union[BaseStrategy, HybridStrategy] = (
                        self._create_test_strategy(strategy, test_params)
                    )
                    result = await self._quick_backtest(
                        test_strategy, symbol, train_data, finviz_data
                    )

                    if result and isinstance(result, dict):
                        score = self._score_optimization_result(result, objective)
                        if score > period_best_score:
                            period_best_score = score
                            period_best_params = dict(
                                zip(param_ranges.keys(), combination)
                            )

                # Test on out-of-sample data
                if period_best_params:
                    oos_test_strategy: Union[BaseStrategy, HybridStrategy] = (
                        self._create_test_strategy(strategy, period_best_params)
                    )
                    oos_result = await self._quick_backtest(
                        oos_test_strategy, symbol, test_data, finviz_data
                    )

                    if oos_result:
                        oos_score = self._score_optimization_result(
                            oos_result, objective
                        )
                        walk_forward_results.append(
                            {
                                "period": f"{start_idx}-{end_idx}",
                                "best_params": period_best_params,
                                "in_sample_score": period_best_score,
                                "out_of_sample_score": oos_score,
                                "oos_result": oos_result,
                            }
                        )

                        if oos_score > best_score:
                            best_score = oos_score
                            best_params = period_best_params

                start_idx += step_size

            # Calculate aggregate metrics
            if walk_forward_results:
                oos_scores = [
                    self._safe_float_convert(r.get("out_of_sample_score", 0))
                    for r in walk_forward_results
                ]
                avg_oos_score = float(np.mean(oos_scores))
                stability = float(
                    1.0 - (np.std(oos_scores) / (abs(avg_oos_score) + 1e-6))
                )
            else:
                avg_oos_score = 0.0
                stability = 0.0

            return {
                "best_parameters": best_params,
                "best_score": best_score,
                "average_oos_score": avg_oos_score,
                "stability": stability,
                "walk_forward_results": walk_forward_results,
                "total_periods": len(walk_forward_results),
            }

        except Exception as e:
            self.logger.error(f"Walk-forward optimization failed: {e}")
            return {"best_parameters": {}, "best_score": 0.0}

    def _safe_float_convert(self, value: Any) -> float:
        """Safely convert a value to float."""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                return float(value)
            else:
                return 0.0
        except (ValueError, TypeError):
            return 0.0

    def _score_optimization_result(
        self, result: Dict[str, Any], objective: str
    ) -> float:
        """Score optimization result based on objective function."""
        try:
            if objective == "sharpe_ratio":
                return result.get("sharpe_ratio", 0.0)
            elif objective == "total_return":
                return result.get("total_pnl", 0.0)
            elif objective == "win_rate":
                return result.get("win_rate", 0.0)
            elif objective == "profit_factor":
                pf = result.get("profit_factor", 0.0)
                return min(pf, 10.0)  # Cap to avoid extreme values
            elif objective == "calmar_ratio":
                return result.get("calmar_ratio", 0.0)
            else:
                # Default composite score
                return (
                    result.get("total_pnl", 0.0)
                    * result.get("win_rate", 0.0)
                    * np.sqrt(result.get("num_trades", 1))
                )

        except Exception:
            return 0.0

    def export_results(
        self, result: DetailedBacktestResult, export_format: str = "json"
    ) -> str:
        """
        Export backtest results in specified format.

        Args:
            result: Backtest results
            export_format: Export format ('json', 'csv', 'html')

        Returns:
            Exported data as string
        """
        try:
            if export_format == "json":
                return self._export_json(result)
            elif export_format == "csv":
                return self._export_csv(result)
            elif export_format == "html":
                return self._export_html(result)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")

        except Exception as e:
            self.logger.error(f"Export error: {e}")
            return f"Export failed: {str(e)}"

    def _export_json(self, result: DetailedBacktestResult) -> str:
        """Export results as JSON."""
        try:
            export_data = {
                "strategy_info": {
                    "name": result.strategy_name,
                    "symbol": result.symbol,
                    "start_date": (
                        result.start_date.isoformat() if result.start_date else None
                    ),
                    "end_date": (
                        result.end_date.isoformat() if result.end_date else None
                    ),
                },
                "performance_metrics": asdict(result.metrics) if result.metrics else {},
                "trades": [asdict(trade) for trade in result.trades],
                "portfolio_snapshots": [
                    asdict(snapshot) for snapshot in result.portfolio_snapshots
                ],
                "risk_metrics": {
                    "var_95": result.var_95,
                    "cvar_95": result.cvar_95,
                    "tail_ratio": result.tail_ratio,
                    "gain_to_pain_ratio": result.gain_to_pain_ratio,
                },
                "trade_analysis": {
                    "trade_distribution": result.trade_distribution,
                    "holding_period_analysis": result.holding_period_analysis,
                },
            }

            return json.dumps(export_data, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"JSON export error: {e}")
            return f'{{"error": "{str(e)}"}}'

    def _export_csv(self, result: DetailedBacktestResult) -> str:
        """Export trades as CSV."""
        try:
            if not result.trades:
                return "No trades to export"

            # Convert trades to DataFrame
            trades_df = pd.DataFrame([asdict(trade) for trade in result.trades])
            return trades_df.to_csv(index=False)

        except Exception as e:
            self.logger.error(f"CSV export error: {e}")
            return f"CSV export failed: {str(e)}"

    def _export_html(self, result: DetailedBacktestResult) -> str:
        """Export results as HTML report."""
        try:
            # Extract metrics safely
            total_return = result.metrics.total_return if result.metrics else None
            sharpe_ratio = result.metrics.sharpe_ratio if result.metrics else None
            max_drawdown = result.metrics.max_drawdown if result.metrics else None
            win_rate = result.metrics.win_rate if result.metrics else None
            total_trades = result.metrics.total_trades if result.metrics else None

            # Format values
            total_return_str = (
                f"{total_return:.2%}" if total_return is not None else "N/A"
            )
            sharpe_ratio_str = (
                f"{sharpe_ratio:.2f}" if sharpe_ratio is not None else "N/A"
            )
            max_drawdown_str = (
                f"{max_drawdown:.2%}" if max_drawdown is not None else "N/A"
            )
            win_rate_str = f"{win_rate:.1%}" if win_rate is not None else "N/A"
            total_trades_str = str(total_trades) if total_trades is not None else "N/A"

            # Determine color class
            return_class = (
                "positive" if total_return and total_return > 0 else "negative"
            )

            html = f"""
            <html>
            <head>
                <title>Backtest Report - {result.strategy_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .metric {{ margin: 10px 0; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Backtest Report: {result.strategy_name}</h1>
                <h2>Strategy: {result.symbol}</h2>
                <p>Period: {result.start_date} to {result.end_date}</p>

                <h3>Performance Metrics</h3>
                <div class="metric">Total Return: <span class="{return_class}">{total_return_str}</span></div>
                <div class="metric">Sharpe Ratio: {sharpe_ratio_str}</div>
                <div class="metric">Max Drawdown: <span class="negative">{max_drawdown_str}</span></div>
                <div class="metric">Win Rate: {win_rate_str}</div>
                <div class="metric">Total Trades: {total_trades_str}</div>

                <h3>Recent Trades</h3>
                <table>
                    <tr><th>Date</th><th>Side</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Duration</th></tr>
            """

            # Add recent trades
            for trade in result.trades[-10:]:  # Last 10 trades
                pnl_class = "positive" if trade.pnl > 0 else "negative"
                html += f"""
                    <tr>
                        <td>{trade.entry_time.strftime('%Y-%m-%d')}</td>
                        <td>{trade.side}</td>
                        <td>${trade.entry_price:.2f}</td>
                        <td>${trade.exit_price:.2f}</td>
                        <td class="{pnl_class}">${trade.pnl:.2f}</td>
                        <td>{trade.duration_hours:.1f}h</td>
                    </tr>
                """

            html += """
                </table>
            </body>
            </html>
            """

            return html

        except Exception as e:
            self.logger.error(f"HTML export error: {e}")
            return f"<html><body><h1>Export Error</h1><p>{str(e)}</p></body></html>"
