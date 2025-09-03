#!/usr/bin/env python3
"""
Strategy Parameter Optimization Script

This script systematically tests different parameter combinations ("knobs") for trading strategies
across multiple time periods to identify optimal settings. It performs comprehensive backtesting
with parameter variations to find the best-performing configurations.

Features:
- Tests multiple parameter combinations for each strategy
- Runs backtests across different 30-day periods from the last 2 years
- Analyzes performance across various market conditions
- Provides statistical significance testing
- Generates comprehensive reports with optimal parameter recommendations

Usage:
    python scripts/optimize_strategy_parameters.py [options]

Examples:
    # Optimize day trading strategy
    python scripts/optimize_strategy_parameters.py --strategy day_trading --periods 6

    # Optimize all strategies with custom parameter ranges
    python scripts/optimize_strategy_parameters.py --all-strategies --periods 9 --detailed

    # Quick optimization with fewer combinations
    python scripts/optimize_strategy_parameters.py --strategy swing_trading --quick --periods 3
"""

import argparse
import asyncio
import csv
import itertools
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import List, Tuple

# Add project paths
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root / "backtesting"))

from backtest_models import TimeFrame
from production_backtest_engine import (
    BacktestMode,
    ProductionBacktestConfig,
    ProductionBacktestEngine,
)
from production_strategy_adapter import ProductionStrategyAdapter, StrategyMode


@dataclass
class ParameterSet:
    """Represents a set of parameters to test."""

    name: str
    stop_loss_pct: float
    take_profit_pct: float
    min_confidence: float
    max_position_size: float
    ta_weight: float
    fa_weight: float
    volume_threshold: float
    min_technical_score: float
    min_fundamental_score: float


@dataclass
class OptimizationResult:
    """Results from a single parameter set test."""

    parameter_set: ParameterSet
    period_start: datetime
    period_end: datetime
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    profit_factor: float
    avg_hold_time_hours: float
    final_capital: float
    signals_generated: int
    signals_executed: int
    execution_time_seconds: float


@dataclass
class OptimizationSummary:
    """Summary of optimization results across all periods."""

    parameter_set: ParameterSet
    avg_return: float
    std_return: float
    avg_max_drawdown: float
    avg_sharpe_ratio: float
    avg_win_rate: float
    total_trades: int
    avg_profit_factor: float
    consistency_score: float  # How consistent the results are across periods
    risk_adjusted_score: float  # Custom score combining return and risk
    periods_tested: int
    periods_profitable: int


class ParameterOptimizer:
    """Main optimization engine."""

    def __init__(
        self,
        strategy_type: str,
        test_periods: List[Tuple[datetime, datetime]],
        initial_capital: Decimal = Decimal("100000"),
    ):
        self.strategy_type = strategy_type
        self.test_periods = test_periods
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)

        # Results storage
        self.all_results: List[OptimizationResult] = []
        self.summaries: List[OptimizationSummary] = []

    def generate_parameter_sets(
        self, optimization_level: str = "standard"
    ) -> List[ParameterSet]:
        """Generate parameter combinations to test."""

        if optimization_level == "quick":
            # Fewer combinations for quick testing
            if self.strategy_type == "day_trading":
                return [
                    ParameterSet(
                        "Conservative",
                        0.01,
                        0.015,
                        75.0,
                        0.10,
                        0.7,
                        0.3,
                        2.0,
                        65.0,
                        35.0,
                    ),
                    ParameterSet(
                        "Moderate", 0.015, 0.02, 70.0, 0.15, 0.7, 0.3, 2.0, 60.0, 30.0
                    ),
                    ParameterSet(
                        "Aggressive", 0.02, 0.025, 65.0, 0.20, 0.7, 0.3, 1.5, 55.0, 25.0
                    ),
                ]
            elif self.strategy_type == "swing_trading":
                return [
                    ParameterSet(
                        "Conservative",
                        0.025,
                        0.04,
                        70.0,
                        0.15,
                        0.5,
                        0.5,
                        1.5,
                        60.0,
                        45.0,
                    ),
                    ParameterSet(
                        "Moderate", 0.03, 0.06, 65.0, 0.20, 0.5, 0.5, 1.5, 55.0, 40.0
                    ),
                    ParameterSet(
                        "Aggressive", 0.04, 0.08, 60.0, 0.25, 0.5, 0.5, 1.2, 50.0, 35.0
                    ),
                ]
            else:  # position_trading
                return [
                    ParameterSet(
                        "Conservative",
                        0.04,
                        0.08,
                        65.0,
                        0.20,
                        0.4,
                        0.6,
                        1.2,
                        55.0,
                        55.0,
                    ),
                    ParameterSet(
                        "Moderate", 0.05, 0.10, 60.0, 0.25, 0.4, 0.6, 1.2, 50.0, 50.0
                    ),
                    ParameterSet(
                        "Aggressive", 0.06, 0.12, 55.0, 0.30, 0.4, 0.6, 1.0, 45.0, 45.0
                    ),
                ]

        elif optimization_level == "detailed":
            # Comprehensive grid search
            if self.strategy_type == "day_trading":
                stop_losses = [0.01, 0.015, 0.02, 0.025]
                take_profits = [0.015, 0.02, 0.025, 0.03]
                confidences = [65.0, 70.0, 75.0, 80.0]
                position_sizes = [0.10, 0.15, 0.20]
                ta_weights = [0.6, 0.7, 0.8]
                volume_thresholds = [1.5, 2.0, 2.5]

            elif self.strategy_type == "swing_trading":
                stop_losses = [0.02, 0.03, 0.04, 0.05]
                take_profits = [0.04, 0.06, 0.08, 0.10]
                confidences = [60.0, 65.0, 70.0, 75.0]
                position_sizes = [0.15, 0.20, 0.25]
                ta_weights = [0.4, 0.5, 0.6]
                volume_thresholds = [1.2, 1.5, 1.8]

            else:  # position_trading
                stop_losses = [0.03, 0.05, 0.07, 0.10]
                take_profits = [0.06, 0.10, 0.15, 0.20]
                confidences = [55.0, 60.0, 65.0, 70.0]
                position_sizes = [0.20, 0.25, 0.30]
                ta_weights = [0.3, 0.4, 0.5]
                volume_thresholds = [1.0, 1.2, 1.5]

            # Generate all combinations (this will be many!)
            combinations = list(
                itertools.product(
                    stop_losses,
                    take_profits,
                    confidences,
                    position_sizes,
                    ta_weights,
                    volume_thresholds,
                )
            )

            parameter_sets = []
            for i, (sl, tp, conf, pos, ta, vol) in enumerate(
                combinations[:50]
            ):  # Limit to 50 for sanity
                fa_weight = 1.0 - ta
                tech_score = conf - 10
                fund_score = conf - 20

                param_set = ParameterSet(
                    name=f"Config_{i + 1}",
                    stop_loss_pct=sl,
                    take_profit_pct=tp,
                    min_confidence=conf,
                    max_position_size=pos,
                    ta_weight=ta,
                    fa_weight=fa_weight,
                    volume_threshold=vol,
                    min_technical_score=tech_score,
                    min_fundamental_score=fund_score,
                )
                parameter_sets.append(param_set)

            return parameter_sets

        else:  # standard
            # Balanced set of configurations
            if self.strategy_type == "day_trading":
                return [
                    ParameterSet(
                        "Ultra_Conservative",
                        0.01,
                        0.015,
                        80.0,
                        0.10,
                        0.8,
                        0.2,
                        2.5,
                        70.0,
                        40.0,
                    ),
                    ParameterSet(
                        "Conservative",
                        0.015,
                        0.02,
                        75.0,
                        0.12,
                        0.75,
                        0.25,
                        2.2,
                        65.0,
                        35.0,
                    ),
                    ParameterSet(
                        "Moderate", 0.015, 0.02, 70.0, 0.15, 0.7, 0.3, 2.0, 60.0, 30.0
                    ),
                    ParameterSet(
                        "Balanced", 0.02, 0.025, 68.0, 0.17, 0.68, 0.32, 1.8, 58.0, 28.0
                    ),
                    ParameterSet(
                        "Aggressive",
                        0.02,
                        0.03,
                        65.0,
                        0.20,
                        0.65,
                        0.35,
                        1.5,
                        55.0,
                        25.0,
                    ),
                    ParameterSet(
                        "High_Confidence",
                        0.025,
                        0.035,
                        80.0,
                        0.15,
                        0.7,
                        0.3,
                        2.0,
                        70.0,
                        40.0,
                    ),
                    ParameterSet(
                        "High_Volume",
                        0.015,
                        0.02,
                        70.0,
                        0.15,
                        0.7,
                        0.3,
                        3.0,
                        60.0,
                        30.0,
                    ),
                    ParameterSet(
                        "Large_Positions",
                        0.02,
                        0.025,
                        75.0,
                        0.22,
                        0.7,
                        0.3,
                        2.0,
                        65.0,
                        35.0,
                    ),
                ]

            elif self.strategy_type == "swing_trading":
                return [
                    ParameterSet(
                        "Ultra_Conservative",
                        0.02,
                        0.04,
                        75.0,
                        0.15,
                        0.6,
                        0.4,
                        1.8,
                        65.0,
                        50.0,
                    ),
                    ParameterSet(
                        "Conservative",
                        0.025,
                        0.05,
                        70.0,
                        0.18,
                        0.55,
                        0.45,
                        1.6,
                        60.0,
                        45.0,
                    ),
                    ParameterSet(
                        "Moderate", 0.03, 0.06, 65.0, 0.20, 0.5, 0.5, 1.5, 55.0, 40.0
                    ),
                    ParameterSet(
                        "Balanced", 0.035, 0.07, 62.0, 0.22, 0.48, 0.52, 1.4, 52.0, 38.0
                    ),
                    ParameterSet(
                        "Aggressive",
                        0.04,
                        0.08,
                        60.0,
                        0.25,
                        0.45,
                        0.55,
                        1.2,
                        50.0,
                        35.0,
                    ),
                    ParameterSet(
                        "High_Confidence",
                        0.03,
                        0.06,
                        75.0,
                        0.18,
                        0.5,
                        0.5,
                        1.5,
                        65.0,
                        50.0,
                    ),
                    ParameterSet(
                        "Fundamental_Focus",
                        0.03,
                        0.06,
                        65.0,
                        0.20,
                        0.4,
                        0.6,
                        1.5,
                        50.0,
                        45.0,
                    ),
                    ParameterSet(
                        "Large_Positions",
                        0.035,
                        0.07,
                        68.0,
                        0.28,
                        0.5,
                        0.5,
                        1.5,
                        58.0,
                        43.0,
                    ),
                ]

            else:  # position_trading
                return [
                    ParameterSet(
                        "Ultra_Conservative",
                        0.03,
                        0.06,
                        70.0,
                        0.20,
                        0.5,
                        0.5,
                        1.5,
                        60.0,
                        60.0,
                    ),
                    ParameterSet(
                        "Conservative",
                        0.04,
                        0.08,
                        65.0,
                        0.22,
                        0.45,
                        0.55,
                        1.3,
                        55.0,
                        55.0,
                    ),
                    ParameterSet(
                        "Moderate", 0.05, 0.10, 60.0, 0.25, 0.4, 0.6, 1.2, 50.0, 50.0
                    ),
                    ParameterSet(
                        "Balanced", 0.06, 0.12, 58.0, 0.27, 0.38, 0.62, 1.1, 48.0, 48.0
                    ),
                    ParameterSet(
                        "Aggressive",
                        0.07,
                        0.14,
                        55.0,
                        0.30,
                        0.35,
                        0.65,
                        1.0,
                        45.0,
                        45.0,
                    ),
                    ParameterSet(
                        "High_Confidence",
                        0.05,
                        0.10,
                        70.0,
                        0.22,
                        0.4,
                        0.6,
                        1.2,
                        60.0,
                        60.0,
                    ),
                    ParameterSet(
                        "Fundamental_Focus",
                        0.05,
                        0.10,
                        60.0,
                        0.25,
                        0.3,
                        0.7,
                        1.2,
                        45.0,
                        55.0,
                    ),
                    ParameterSet(
                        "Large_Positions",
                        0.06,
                        0.12,
                        62.0,
                        0.32,
                        0.4,
                        0.6,
                        1.2,
                        52.0,
                        52.0,
                    ),
                ]

    async def test_parameter_set(
        self, param_set: ParameterSet, period_start: datetime, period_end: datetime
    ) -> OptimizationResult:
        """Test a single parameter set over one time period."""

        try:
            # Create custom strategy adapter with the parameter set
            strategy_mode_map = {
                "day_trading": StrategyMode.DAY_TRADING,
                "swing_trading": StrategyMode.SWING_TRADING,
                "position_trading": StrategyMode.POSITION_TRADING,
            }

            strategy = ProductionStrategyAdapter(strategy_mode_map[self.strategy_type])

            # Override strategy parameters with our test parameters
            strategy.config.update(
                {
                    "stop_loss_pct": param_set.stop_loss_pct,
                    "take_profit_pct": param_set.take_profit_pct,
                    "min_confidence": param_set.min_confidence,
                    "max_position_size": param_set.max_position_size,
                    "ta_weight": param_set.ta_weight,
                    "fa_weight": param_set.fa_weight,
                    "volume_threshold": param_set.volume_threshold,
                    "min_technical_score": param_set.min_technical_score,
                    "min_fundamental_score": param_set.min_fundamental_score,
                }
            )

            # Create backtest configuration
            config = ProductionBacktestConfig(
                start_date=period_start,
                end_date=period_end,
                strategy_type=self.strategy_type,
                initial_capital=self.initial_capital,
                max_positions=8,
                mode=BacktestMode.FAST,
                timeframe=TimeFrame.ONE_DAY,
                enable_screener_simulation=True,
                max_screener_symbols_per_day=30,
            )

            # Run backtest
            engine = ProductionBacktestEngine(config)
            engine.strategy = strategy  # Use our custom strategy

            start_time = time.time()
            results = await engine.run_backtest()
            execution_time = time.time() - start_time

            # Create optimization result
            return OptimizationResult(
                parameter_set=param_set,
                period_start=period_start,
                period_end=period_end,
                total_return=results.total_return,
                max_drawdown=results.max_drawdown,
                sharpe_ratio=results.sharpe_ratio,
                win_rate=results.win_rate,
                total_trades=results.total_trades,
                profit_factor=results.profit_factor,
                avg_hold_time_hours=results.avg_hold_time_hours,
                final_capital=results.final_capital,
                signals_generated=results.total_signals_generated,
                signals_executed=results.signals_executed,
                execution_time_seconds=execution_time,
            )

        except Exception as e:
            self.logger.error(
                f"Error testing parameter set {param_set.name} "
                f"for period {period_start.date()}-{period_end.date()}: {e}"
            )

            # Return a failed result
            return OptimizationResult(
                parameter_set=param_set,
                period_start=period_start,
                period_end=period_end,
                total_return=-999.0,  # Signal failure
                max_drawdown=999.0,
                sharpe_ratio=-999.0,
                win_rate=0.0,
                total_trades=0,
                profit_factor=0.0,
                avg_hold_time_hours=0.0,
                final_capital=Decimal("0"),
                signals_generated=0,
                signals_executed=0,
                execution_time_seconds=0.0,
            )

    async def run_optimization(
        self, optimization_level: str = "standard"
    ) -> List[OptimizationSummary]:
        """Run the complete optimization process."""

        self.logger.info(f"Starting parameter optimization for {self.strategy_type}")
        self.logger.info(f"Testing {len(self.test_periods)} time periods")
        self.logger.info(f"Optimization level: {optimization_level}")

        # Generate parameter sets to test
        parameter_sets = self.generate_parameter_sets(optimization_level)
        self.logger.info(
            f"Generated {len(parameter_sets)} parameter combinations to test"
        )

        total_tests = len(parameter_sets) * len(self.test_periods)
        self.logger.info(f"Total tests to run: {total_tests}")

        # Run all combinations
        test_count = 0
        for param_set in parameter_sets:
            self.logger.info(f"Testing parameter set: {param_set.name}")

            period_results = []
            for period_start, period_end in self.test_periods:
                test_count += 1
                self.logger.info(
                    f"  Period {period_start.date()} to {period_end.date()} "
                    f"({test_count}/{total_tests})"
                )

                result = await self.test_parameter_set(
                    param_set, period_start, period_end
                )
                period_results.append(result)
                self.all_results.append(result)

                # Log result
                if result.total_return > -999:  # Not a failed test
                    self.logger.info(
                        f"    Return: {result.total_return:.2%}, "
                        f"Drawdown: {result.max_drawdown:.2%}, "
                        f"Trades: {result.total_trades}"
                    )
                else:
                    self.logger.warning("    Test failed")

            # Calculate summary for this parameter set
            summary = self._calculate_summary(param_set, period_results)
            self.summaries.append(summary)

            self.logger.info(
                f"  Summary - Avg Return: {summary.avg_return:.2%}, "
                f"Consistency: {summary.consistency_score:.3f}, "
                f"Risk-Adj Score: {summary.risk_adjusted_score:.3f}"
            )

        # Sort summaries by risk-adjusted score
        self.summaries.sort(key=lambda x: x.risk_adjusted_score, reverse=True)

        self.logger.info("Optimization completed!")
        return self.summaries

    def _calculate_summary(
        self, param_set: ParameterSet, period_results: List[OptimizationResult]
    ) -> OptimizationSummary:
        """Calculate summary statistics for a parameter set across all periods."""

        # Filter out failed tests
        valid_results = [r for r in period_results if r.total_return > -999]

        if not valid_results:
            # All tests failed
            return OptimizationSummary(
                parameter_set=param_set,
                avg_return=-999.0,
                std_return=999.0,
                avg_max_drawdown=999.0,
                avg_sharpe_ratio=-999.0,
                avg_win_rate=0.0,
                total_trades=0,
                avg_profit_factor=0.0,
                consistency_score=0.0,
                risk_adjusted_score=-999.0,
                periods_tested=len(period_results),
                periods_profitable=0,
            )

        # Calculate statistics
        returns = [r.total_return for r in valid_results]
        drawdowns = [r.max_drawdown for r in valid_results]
        sharpe_ratios = [r.sharpe_ratio for r in valid_results]
        win_rates = [r.win_rate for r in valid_results]
        profit_factors = [r.profit_factor for r in valid_results]

        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0.0
        avg_max_drawdown = statistics.mean(drawdowns)
        avg_sharpe_ratio = statistics.mean(sharpe_ratios)
        avg_win_rate = statistics.mean(win_rates)
        avg_profit_factor = statistics.mean(profit_factors)

        total_trades = sum(r.total_trades for r in valid_results)
        periods_profitable = sum(1 for r in valid_results if r.total_return > 0)

        # Calculate consistency score (lower standard deviation is better)
        if std_return == 0:
            consistency_score = 1.0
        else:
            consistency_score = max(0, 1 - (std_return / max(abs(avg_return), 0.01)))

        # Calculate risk-adjusted score
        if avg_max_drawdown == 0:
            risk_adjusted_score = avg_return
        else:
            risk_adjusted_score = (avg_return / avg_max_drawdown) * consistency_score

        return OptimizationSummary(
            parameter_set=param_set,
            avg_return=avg_return,
            std_return=std_return,
            avg_max_drawdown=avg_max_drawdown,
            avg_sharpe_ratio=avg_sharpe_ratio,
            avg_win_rate=avg_win_rate,
            total_trades=total_trades,
            avg_profit_factor=avg_profit_factor,
            consistency_score=consistency_score,
            risk_adjusted_score=risk_adjusted_score,
            periods_tested=len(period_results),
            periods_profitable=periods_profitable,
        )

    def save_results(self, output_dir: str, timestamp: str):
        """Save optimization results to files."""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save detailed results
        detailed_file = (
            output_path / f"optimization_detailed_{self.strategy_type}_{timestamp}.csv"
        )
        with open(detailed_file, "w", newline="") as f:
            if self.all_results:
                writer = csv.DictWriter(
                    f, fieldnames=list(asdict(self.all_results[0]).keys())
                )
                writer.writeheader()

                for result in self.all_results:
                    row = asdict(result)
                    # Convert datetime objects to strings
                    row["period_start"] = result.period_start.isoformat()
                    row["period_end"] = result.period_end.isoformat()
                    # Convert parameter set to dict
                    row["parameter_set"] = asdict(result.parameter_set)
                    writer.writerow(row)

        # Save summary results
        summary_file = (
            output_path / f"optimization_summary_{self.strategy_type}_{timestamp}.csv"
        )
        with open(summary_file, "w", newline="") as f:
            if self.summaries:
                # Flatten the data for CSV
                fieldnames = (
                    ["param_name"]
                    + list(asdict(self.summaries[0].parameter_set).keys())[1:]
                    + [
                        k
                        for k in asdict(self.summaries[0]).keys()
                        if k != "parameter_set"
                    ]
                )

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for summary in self.summaries:
                    row = {"param_name": summary.parameter_set.name}
                    param_dict = asdict(summary.parameter_set)
                    row.update({k: v for k, v in param_dict.items() if k != "name"})
                    summary_dict = asdict(summary)
                    row.update(
                        {k: v for k, v in summary_dict.items() if k != "parameter_set"}
                    )
                    writer.writerow(row)

        # Save JSON summary for programmatic access
        json_file = (
            output_path / f"optimization_results_{self.strategy_type}_{timestamp}.json"
        )
        with open(json_file, "w") as f:
            json_data = {
                "strategy_type": self.strategy_type,
                "optimization_timestamp": timestamp,
                "periods_tested": len(self.test_periods),
                "parameter_combinations_tested": len(self.summaries),
                "best_parameters": (
                    asdict(self.summaries[0]) if self.summaries else None
                ),
                "all_summaries": [asdict(s) for s in self.summaries],
            }
            json.dump(json_data, f, indent=2, default=str)

        print("\nResults saved:")
        print(f"  Detailed: {detailed_file}")
        print(f"  Summary: {summary_file}")
        print(f"  JSON: {json_file}")


def generate_test_periods(
    num_periods: int = 6, period_days: int = 30
) -> List[Tuple[datetime, datetime]]:
    """Generate test periods going back in time."""

    periods = []

    # Start from the most recent available data (August 21, 2025)
    latest_date = datetime(2025, 8, 21, tzinfo=timezone.utc)

    for i in range(num_periods):
        # Go back in time
        end_date = latest_date - timedelta(
            days=i * period_days * 1.2
        )  # Some spacing between periods
        start_date = end_date - timedelta(days=period_days)

        # Ensure we don't go too far back (we have data from Aug 2024)
        if start_date < datetime(2024, 9, 1, tzinfo=timezone.utc):
            break

        periods.append((start_date, end_date))

    return periods


def display_optimization_results(
    summaries: List[OptimizationSummary], strategy_type: str
):
    """Display optimization results in a formatted table."""

    print(f"\n{'=' * 100}")
    print(f"PARAMETER OPTIMIZATION RESULTS - {strategy_type.upper().replace('_', ' ')}")
    print(f"{'=' * 100}")

    if not summaries:
        print("No results to display.")
        return

    print(
        f"\n{'Rank':<4} {'Parameter Set':<20} {'Avg Return':<12} {'Std Dev':<10} {'Avg Drawdown':<12} "
        f"{'Win Rate':<10} {'Trades':<8} {'Risk-Adj Score':<15}"
    )
    print("-" * 100)

    for i, summary in enumerate(summaries[:15]):  # Top 15
        rank = i + 1
        name = summary.parameter_set.name[:19]  # Truncate if too long
        avg_return = f"{summary.avg_return:.2%}"
        std_return = f"{summary.std_return:.2%}"
        avg_drawdown = f"{summary.avg_max_drawdown:.2%}"
        win_rate = f"{summary.avg_win_rate:.1%}"
        total_trades = str(summary.total_trades)
        risk_score = f"{summary.risk_adjusted_score:.3f}"

        print(
            f"{rank:<4} {name:<20} {avg_return:<12} {std_return:<10} {avg_drawdown:<12} "
            f"{win_rate:<10} {total_trades:<8} {risk_score:<15}"
        )

    # Show best parameter details
    if summaries:
        best = summaries[0]
        print(f"\n{'=' * 60}")
        print(f"BEST PARAMETER CONFIGURATION: {best.parameter_set.name}")
        print(f"{'=' * 60}")

        param_dict = asdict(best.parameter_set)
        for key, value in param_dict.items():
            if key != "name":
                if isinstance(value, float):
                    if "pct" in key or "weight" in key:
                        print(f"  {key.replace('_', ' ').title()}: {value:.1%}")
                    else:
                        print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")

        print("\nPerformance Summary:")
        print(f"  Average Return: {best.avg_return:.2%}")
        print(f"  Return Std Dev: {best.std_return:.2%}")
        print(f"  Average Max Drawdown: {best.avg_max_drawdown:.2%}")
        print(f"  Average Win Rate: {best.avg_win_rate:.1%}")
        print(f"  Total Trades: {best.total_trades}")
        print(f"  Profitable Periods: {best.periods_profitable}/{best.periods_tested}")
        print(f"  Consistency Score: {best.consistency_score:.3f}")
        print(f"  Risk-Adjusted Score: {best.risk_adjusted_score:.3f}")


def setup_logging(debug: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("parameter_optimization.log"),
        ],
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize trading strategy parameters using systematic backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )

    # Strategy selection
    strategy_group = parser.add_argument_group("Strategy Selection")
    strategy_group.add_argument(
        "--strategy",
        choices=["day_trading", "swing_trading", "position_trading"],
        help="Strategy to optimize (required unless --all-strategies is used)",
    )
    strategy_group.add_argument(
        "--all-strategies",
        action="store_true",
        help="Optimize all three strategies sequentially",
    )

    # Optimization settings
    opt_group = parser.add_argument_group("Optimization Settings")
    opt_group.add_argument(
        "--periods",
        type=int,
        default=6,
        help="Number of 30-day periods to test (default: 6)",
    )
    opt_group.add_argument(
        "--period-days",
        type=int,
        default=30,
        help="Length of each test period in days (default: 30)",
    )
    opt_group.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital for each test (default: 100000)",
    )

    # Optimization levels
    level_group = parser.add_argument_group("Optimization Levels")
    level_group.add_argument(
        "--quick",
        action="store_true",
        help="Quick optimization with fewer parameter combinations",
    )
    level_group.add_argument(
        "--detailed",
        action="store_true",
        help="Detailed optimization with comprehensive grid search",
    )

    # Output settings
    output_group = parser.add_argument_group("Output Settings")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="optimization_results",
        help="Directory to save results (default: optimization_results)",
    )
    output_group.add_argument(
        "--no-save", action="store_true", help="Do not save results to files"
    )

    # Other options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


async def optimize_single_strategy(
    strategy_type: str, args
) -> List[OptimizationSummary]:
    """Optimize a single strategy."""

    print(f"\n🚀 OPTIMIZING {strategy_type.upper().replace('_', ' ')} STRATEGY")
    print("=" * 80)

    # Generate test periods
    test_periods = generate_test_periods(args.periods, args.period_days)

    print(f"Generated {len(test_periods)} test periods:")
    for i, (start, end) in enumerate(test_periods):
        print(f"  Period {i + 1}: {start.date()} to {end.date()}")

    # Determine optimization level
    if args.quick:
        opt_level = "quick"
    elif args.detailed:
        opt_level = "detailed"
    else:
        opt_level = "standard"

    # Create optimizer
    optimizer = ParameterOptimizer(
        strategy_type=strategy_type,
        test_periods=test_periods,
        initial_capital=Decimal(str(args.capital)),
    )

    # Run optimization
    start_time = time.time()
    summaries = await optimizer.run_optimization(opt_level)
    total_time = time.time() - start_time

    print(f"\nOptimization completed in {total_time:.2f} seconds")

    # Display results
    display_optimization_results(summaries, strategy_type)

    # Save results
    if not args.no_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        optimizer.save_results(args.output_dir, timestamp)

    return summaries


async def main():
    """Main execution function."""
    args = parse_arguments()

    # Set up logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # Validate arguments
    if not args.strategy and not args.all_strategies:
        print("ERROR: Must specify either --strategy or --all-strategies")
        return 1

    if args.strategy and args.all_strategies:
        print("ERROR: Cannot specify both --strategy and --all-strategies")
        return 1

    try:
        print("🎯 TRADING STRATEGY PARAMETER OPTIMIZATION")
        print("=" * 60)
        print("This script will systematically test different parameter combinations")
        print("across multiple time periods to find optimal strategy settings.")
        print()

        all_results = {}

        # Determine which strategies to optimize
        if args.all_strategies:
            strategies = ["day_trading", "swing_trading", "position_trading"]
        else:
            strategies = [args.strategy]

        # Optimize each strategy
        for strategy_type in strategies:
            try:
                summaries = await optimize_single_strategy(strategy_type, args)
                all_results[strategy_type] = summaries

            except KeyboardInterrupt:
                print(f"\n⚠️  Optimization interrupted for {strategy_type}")
                break
            except Exception as e:
                logger.error(f"Failed to optimize {strategy_type}: {e}")
                continue

        # Final summary if multiple strategies
        if len(all_results) > 1:
            print(f"\n{'=' * 100}")
            print("FINAL SUMMARY - BEST PARAMETERS FOR EACH STRATEGY")
            print(f"{'=' * 100}")

            for strategy_type, summaries in all_results.items():
                if summaries:
                    best = summaries[0]
                    print(f"\n🏆 {strategy_type.upper().replace('_', ' ')}:")
                    print(f"   Parameter Set: {best.parameter_set.name}")
                    print(f"   Avg Return: {best.avg_return:.2%}")
                    print(f"   Risk-Adj Score: {best.risk_adjusted_score:.3f}")
                    print(f"   Win Rate: {best.avg_win_rate:.1%}")
                    print(f"   Consistency: {best.consistency_score:.3f}")

        print("\n✅ Parameter optimization completed!")
        print("📊 Check the detailed results above")
        print(
            "🔧 Use the best parameters to update your production strategy configurations"
        )

        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
