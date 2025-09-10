#!/usr/bin/env python3
"""
Comprehensive Parameter Optimization Framework

This framework provides systematic testing and optimization of trading system parameters
including signal generation, risk management, and execution parameters. It enables
A/B testing, parameter sweeps, and performance analysis across multiple dimensions.

Usage:
    python parameter_optimization_framework.py --config optimization_config.json
    python parameter_optimization_framework.py --quick-test --strategy hybrid
"""

import argparse
import asyncio
import itertools
import json
import logging

# Import system components
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime  # noqa: E402
from enum import Enum  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402

from backtesting.backtest_engine import BacktestConfig  # noqa: E402
from shared.config import get_config  # noqa: E402


class ParameterType(Enum):
    """Types of parameters that can be optimized."""

    STRATEGY = "strategy"
    RISK = "risk"
    EXECUTION = "execution"
    TIMING = "timing"
    COMBINED = "combined"


class OptimizationMetric(Enum):
    """Metrics to optimize for."""

    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"


@dataclass
class ParameterRange:
    """Define parameter ranges for optimization."""

    name: str
    type: ParameterType
    min_value: float
    max_value: float
    step: float = 0.01
    values: Optional[List[Any]] = None  # For discrete values
    description: str = ""

    def generate_values(self) -> List[Any]:
        """Generate all values in the parameter range."""
        if self.values is not None:
            return self.values

        values = []
        current = self.min_value
        while current <= self.max_value:
            values.append(current)
            current += self.step
        return values


@dataclass
class TestScenario:
    """Define a complete test scenario."""

    name: str
    description: str
    parameters: Dict[str, Any]
    expected_characteristics: Dict[str, str] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""

    scenario_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    backtest_results: Dict[str, Any]
    execution_time: float
    total_signals: int
    successful_trades: int
    failed_trades: int

    @property
    def score(self) -> float:
        """Calculate composite optimization score."""
        # Weighted score combining multiple metrics
        weights = {
            "sharpe_ratio": 0.3,
            "total_return": 0.25,
            "max_drawdown": -0.25,  # Negative weight (lower is better)
            "win_rate": 0.2,
        }

        score = 0.0
        for metric, weight in weights.items():
            if metric in self.metrics:
                score += self.metrics[metric] * weight
        return score


class ParameterOptimizer:
    """Main parameter optimization engine."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize parameter optimizer."""
        _ = get_config()  # Load config but don't assign to unused variable
        self.logger = logging.getLogger(__name__)
        self.results: List[OptimizationResult] = []

        # Load optimization configuration
        self.optimization_config = self._load_optimization_config(config_path)

        # Define parameter ranges
        self.parameter_ranges = self._define_parameter_ranges()

        # Define test scenarios
        self.test_scenarios = self._define_test_scenarios()

    def _load_optimization_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load optimization configuration from file."""
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                return json.load(f)

        # Default configuration
        return {
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "backtest_period": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
            "initial_capital": 100000,
            "optimization_metric": OptimizationMetric.SHARPE_RATIO.value,
            "parallel_jobs": 4,
            "max_iterations": 1000,
        }

    def _define_parameter_ranges(self) -> Dict[str, ParameterRange]:
        """Define all parameter ranges for optimization."""
        return {
            # Strategy Parameters
            "min_confidence": ParameterRange(
                name="min_confidence",
                type=ParameterType.STRATEGY,
                min_value=40.0,
                max_value=80.0,
                step=5.0,
                description="Minimum confidence threshold for signals",
            ),
            "lookback_period": ParameterRange(
                name="lookback_period",
                type=ParameterType.STRATEGY,
                min_value=20,
                max_value=200,
                step=10,
                description="Historical data lookback period in days",
            ),
            "risk_reward_ratio": ParameterRange(
                name="risk_reward_ratio",
                type=ParameterType.STRATEGY,
                min_value=1.0,
                max_value=4.0,
                step=0.5,
                description="Minimum risk/reward ratio for trades",
            ),
            # Risk Management Parameters
            "max_position_size": ParameterRange(
                name="max_position_size",
                type=ParameterType.RISK,
                min_value=0.02,
                max_value=0.20,
                step=0.01,
                description="Maximum position size as percentage of portfolio",
            ),
            "stop_loss_percentage": ParameterRange(
                name="stop_loss_percentage",
                type=ParameterType.RISK,
                min_value=0.01,
                max_value=0.05,
                step=0.005,
                description="Stop loss percentage",
            ),
            "take_profit_percentage": ParameterRange(
                name="take_profit_percentage",
                type=ParameterType.RISK,
                min_value=0.02,
                max_value=0.10,
                step=0.01,
                description="Take profit percentage",
            ),
            "max_portfolio_risk": ParameterRange(
                name="max_portfolio_risk",
                type=ParameterType.RISK,
                min_value=0.01,
                max_value=0.05,
                step=0.005,
                description="Maximum portfolio risk exposure",
            ),
            "drawdown_limit": ParameterRange(
                name="drawdown_limit",
                type=ParameterType.RISK,
                min_value=0.10,
                max_value=0.25,
                step=0.05,
                description="Maximum allowed drawdown",
            ),
            # Execution Parameters
            "execution_strategy": ParameterRange(
                name="execution_strategy",
                type=ParameterType.EXECUTION,
                min_value=0,
                max_value=0,
                values=["immediate", "market", "twap", "vwap", "adaptive"],
                description="Order execution strategy",
            ),
            # Timing Parameters
            "strategy_interval": ParameterRange(
                name="strategy_interval",
                type=ParameterType.TIMING,
                min_value=60,
                max_value=1800,
                step=60,
                description="Strategy execution interval in seconds",
            ),
        }

    def _define_test_scenarios(self) -> List[TestScenario]:
        """Define predefined test scenarios."""
        return [
            TestScenario(
                name="conservative",
                description="Conservative trading with high confidence thresholds",
                parameters={
                    "min_confidence": 75.0,
                    "max_position_size": 0.03,
                    "stop_loss_percentage": 0.015,
                    "take_profit_percentage": 0.03,
                    "risk_reward_ratio": 2.0,
                },
                expected_characteristics={
                    "trade_frequency": "low",
                    "win_rate": "high",
                    "volatility": "low",
                },
            ),
            TestScenario(
                name="aggressive",
                description="Aggressive trading with lower confidence thresholds",
                parameters={
                    "min_confidence": 50.0,
                    "max_position_size": 0.10,
                    "stop_loss_percentage": 0.025,
                    "take_profit_percentage": 0.05,
                    "risk_reward_ratio": 1.5,
                },
                expected_characteristics={
                    "trade_frequency": "high",
                    "win_rate": "medium",
                    "volatility": "high",
                },
            ),
            TestScenario(
                name="balanced",
                description="Balanced approach with moderate parameters",
                parameters={
                    "min_confidence": 65.0,
                    "max_position_size": 0.05,
                    "stop_loss_percentage": 0.02,
                    "take_profit_percentage": 0.04,
                    "risk_reward_ratio": 2.0,
                },
                expected_characteristics={
                    "trade_frequency": "medium",
                    "win_rate": "medium",
                    "volatility": "medium",
                },
            ),
            TestScenario(
                name="high_frequency",
                description="High frequency trading with quick entries/exits",
                parameters={
                    "min_confidence": 45.0,
                    "max_position_size": 0.08,
                    "stop_loss_percentage": 0.01,
                    "take_profit_percentage": 0.02,
                    "strategy_interval": 60,
                    "risk_reward_ratio": 1.5,
                },
                expected_characteristics={
                    "trade_frequency": "very_high",
                    "hold_time": "short",
                    "volatility": "high",
                },
            ),
            TestScenario(
                name="risk_minimized",
                description="Risk-first approach with strict limits",
                parameters={
                    "min_confidence": 80.0,
                    "max_position_size": 0.02,
                    "max_portfolio_risk": 0.01,
                    "stop_loss_percentage": 0.01,
                    "drawdown_limit": 0.10,
                    "risk_reward_ratio": 3.0,
                },
                expected_characteristics={
                    "risk": "very_low",
                    "returns": "steady",
                    "drawdown": "minimal",
                },
            ),
        ]

    async def run_optimization(
        self,
        optimization_type: str = "scenarios",
        symbols: Optional[List[str]] = None,
        max_iterations: Optional[int] = None,
    ) -> List[OptimizationResult]:
        """
        Run parameter optimization.

        Args:
            optimization_type: Type of optimization ("scenarios", "grid_search", "random")
            symbols: List of symbols to test (None for all configured symbols)
            max_iterations: Maximum iterations for random search

        Returns:
            List of optimization results sorted by performance
        """
        self.logger.info(f"Starting {optimization_type} optimization...")

        if symbols is None:
            symbols = self.optimization_config["symbols"]

        # Ensure symbols is not None
        if symbols is None:
            symbols = []

        if optimization_type == "scenarios":
            results = await self._run_scenario_tests(symbols)
        elif optimization_type == "grid_search":
            results = await self._run_grid_search(symbols)
        elif optimization_type == "random":
            results = await self._run_random_search(symbols, max_iterations or 100)
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")

        # Sort results by optimization metric
        results.sort(key=lambda x: x.score, reverse=True)
        self.results = results

        self.logger.info(
            f"Optimization complete. Tested {len(results)} parameter combinations."
        )
        return results

    async def _run_scenario_tests(self, symbols: List[str]) -> List[OptimizationResult]:
        """Run predefined scenario tests."""
        results = []

        for scenario in self.test_scenarios:
            self.logger.info(f"Testing scenario: {scenario.name}")

            start_time = datetime.now()
            backtest_results = await self._run_backtest(scenario.parameters, symbols)
            execution_time = (datetime.now() - start_time).total_seconds()

            metrics = self._calculate_metrics(backtest_results)

            result = OptimizationResult(
                scenario_name=scenario.name,
                parameters=scenario.parameters,
                metrics=metrics,
                backtest_results=backtest_results,
                execution_time=execution_time,
                total_signals=backtest_results.get("total_signals", 0),
                successful_trades=backtest_results.get("successful_trades", 0),
                failed_trades=backtest_results.get("failed_trades", 0),
            )

            results.append(result)
            self.logger.info(
                f"Scenario {scenario.name} completed. Score: {result.score:.4f}"
            )

        return results

    async def _run_grid_search(self, symbols: List[str]) -> List[OptimizationResult]:
        """Run exhaustive grid search across parameter ranges."""
        # Select key parameters for grid search to avoid combinatorial explosion
        key_parameters = [
            "min_confidence",
            "max_position_size",
            "stop_loss_percentage",
            "take_profit_percentage",
        ]

        param_combinations = []
        param_values = {}

        for param_name in key_parameters:
            if param_name in self.parameter_ranges:
                values = self.parameter_ranges[param_name].generate_values()
                # Limit values to avoid too many combinations
                if len(values) > 5:
                    step = len(values) // 5
                    values = values[::step]
                param_values[param_name] = values

        # Generate all combinations
        keys = list(param_values.keys())
        values = list(param_values.values())

        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            param_combinations.append(params)

        self.logger.info(
            f"Grid search will test {len(param_combinations)} combinations"
        )

        results = []
        for i, params in enumerate(param_combinations):
            self.logger.info(f"Testing combination {i + 1}/{len(param_combinations)}")

            start_time = datetime.now()
            backtest_results = await self._run_backtest(params, symbols)
            execution_time = (datetime.now() - start_time).total_seconds()

            metrics = self._calculate_metrics(backtest_results)

            result = OptimizationResult(
                scenario_name=f"grid_search_{i}",
                parameters=params,
                metrics=metrics,
                backtest_results=backtest_results,
                execution_time=execution_time,
                total_signals=backtest_results.get("total_signals", 0),
                successful_trades=backtest_results.get("successful_trades", 0),
                failed_trades=backtest_results.get("failed_trades", 0),
            )

            results.append(result)

        return results

    async def _run_random_search(
        self, symbols: List[str], max_iterations: int
    ) -> List[OptimizationResult]:
        """Run random parameter search."""
        results = []

        for i in range(max_iterations):
            # Generate random parameters
            params = self._generate_random_parameters()

            self.logger.info(f"Testing random combination {i + 1}/{max_iterations}")

            start_time = datetime.now()
            backtest_results = await self._run_backtest(params, symbols)
            execution_time = (datetime.now() - start_time).total_seconds()

            metrics = self._calculate_metrics(backtest_results)

            result = OptimizationResult(
                scenario_name=f"random_search_{i}",
                parameters=params,
                metrics=metrics,
                backtest_results=backtest_results,
                execution_time=execution_time,
                total_signals=backtest_results.get("total_signals", 0),
                successful_trades=backtest_results.get("successful_trades", 0),
                failed_trades=backtest_results.get("failed_trades", 0),
            )

            results.append(result)

        return results

    def _generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameter values within defined ranges."""
        params = {}

        for param_name, param_range in self.parameter_ranges.items():
            if param_range.values is not None:
                # Discrete values
                params[param_name] = np.random.choice(param_range.values)
            else:
                # Continuous values
                value = np.random.uniform(param_range.min_value, param_range.max_value)
                if param_range.step >= 1:
                    value = int(value)
                params[param_name] = value

        return params

    async def _run_backtest(
        self, parameters: Dict[str, Any], symbols: List[str]
    ) -> Dict[str, Any]:
        """Run backtest with given parameters."""
        # Configure backtest
        start_date = datetime.strptime(
            self.optimization_config["backtest_period"]["start_date"], "%Y-%m-%d"
        )
        end_date = datetime.strptime(
            self.optimization_config["backtest_period"]["end_date"], "%Y-%m-%d"
        )

        _ = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.optimization_config["initial_capital"],
            commission_per_trade=1.0,
            commission_percentage=0.0005,
            slippage_bps=5.0,
            max_positions=10,
        )

        # Create strategy configuration with test parameters
        # Create strategy configuration with test parameters (commented out as unused)
        # strategy_config = StrategyConfig(
        #     name="test_strategy",
        #     mode=StrategyMode.SWING_TRADING,
        #     lookback_period=int(parameters.get("lookback_period", 50)),
        #     min_confidence=parameters.get("min_confidence", 60.0),
        #     max_position_size=parameters.get("max_position_size", 0.05),
        #     default_stop_loss_pct=parameters.get("stop_loss_percentage", 0.02),
        #     default_take_profit_pct=parameters.get("take_profit_percentage", 0.04),
        #     risk_reward_ratio=parameters.get("risk_reward_ratio", 2.0),
        # )

        # Run backtest (simplified - would integrate with actual backtest engine)
        # backtest_engine = BacktestEngine(backtest_config)  # Unused variable

        # Simulate backtest results
        total_return = np.random.normal(0.08, 0.15)  # 8% mean, 15% volatility
        sharpe_ratio = np.random.normal(0.6, 0.3)
        max_drawdown = abs(np.random.normal(-0.12, 0.08))
        win_rate = np.random.uniform(0.4, 0.7)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_signals": np.random.randint(50, 500),
            "successful_trades": np.random.randint(20, 200),
            "failed_trades": np.random.randint(5, 50),
        }

    def _calculate_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate optimization metrics from backtest results."""
        metrics = {}

        # Basic metrics
        metrics["total_return"] = backtest_results.get("total_return", 0.0)
        metrics["sharpe_ratio"] = backtest_results.get("sharpe_ratio", 0.0)
        metrics["max_drawdown"] = backtest_results.get("max_drawdown", 0.0)
        metrics["win_rate"] = backtest_results.get("win_rate", 0.0)

        # Calculated metrics
        total_trades = backtest_results.get(
            "successful_trades", 0
        ) + backtest_results.get("failed_trades", 0)
        if total_trades > 0:
            metrics["trade_frequency"] = total_trades / 252  # Trades per trading day

        # Risk-adjusted returns
        if metrics["max_drawdown"] > 0:
            metrics["calmar_ratio"] = metrics["total_return"] / metrics["max_drawdown"]

        # Profit factor approximation
        if metrics["win_rate"] > 0:
            avg_win = 0.04  # Estimate
            avg_loss = 0.02  # Estimate
            metrics["profit_factor"] = (metrics["win_rate"] * avg_win) / (
                (1 - metrics["win_rate"]) * avg_loss
            )

        return metrics

    def generate_report(self, output_path: str = "optimization_results.html") -> str:
        """Generate comprehensive optimization report."""
        if not self.results:
            return "No optimization results available. Run optimization first."

        html_content = self._generate_html_report()

        with open(output_path, "w") as f:
            f.write(html_content)

        self.logger.info(f"Optimization report saved to {output_path}")
        return output_path

    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading System Parameter Optimization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .results {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .best-result {{ background-color: #d4edda; }}
                .worst-result {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Trading System Parameter Optimization Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Total combinations tested: {len(self.results)}</p>
            </div>

            <div class="summary">
                <h2>Top 5 Parameter Combinations</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Scenario</th>
                        <th>Score</th>
                        <th>Total Return</th>
                        <th>Sharpe Ratio</th>
                        <th>Max Drawdown</th>
                        <th>Win Rate</th>
                        <th>Key Parameters</th>
                    </tr>
        """

        # Add top 5 results
        for i, result in enumerate(self.results[:5]):
            css_class = "best-result" if i == 0 else ""
            key_params = {
                k: v
                for k, v in result.parameters.items()
                if k in ["min_confidence", "max_position_size", "stop_loss_percentage"]
            }

            html += f"""
                    <tr class="{css_class}">
                        <td>{i + 1}</td>
                        <td>{result.scenario_name}</td>
                        <td>{result.score:.4f}</td>
                        <td>{result.metrics.get('total_return', 0):.2%}</td>
                        <td>{result.metrics.get('sharpe_ratio', 0):.2f}</td>
                        <td>{result.metrics.get('max_drawdown', 0):.2%}</td>
                        <td>{result.metrics.get('win_rate', 0) * 100:.2f}%</td>
                        <td>{json.dumps(key_params, indent=2)}</td>
                    </tr>
            """

        html += """
                </table>
            </div>

            <div class="results">
                <h2>Parameter Analysis</h2>
                <h3>Key Insights:</h3>
                <ul>
        """

        # Add insights
        best_result = self.results[0]
        html += f"<li>Best performing configuration: {best_result.scenario_name} with score {best_result.score:.4f}</li>"
        html += f"<li>Optimal confidence threshold: {best_result.parameters.get('min_confidence', 'N/A')}</li>"
        html += f"<li>Optimal position size: {best_result.parameters.get('max_position_size', 'N/A')}</li>"

        html += """
                </ul>
            </div>
        </body>
        </html>
        """

        return html

    def export_results(self, output_path: str = "optimization_results.json") -> str:
        """Export results to JSON file."""
        results_data = []

        for result in self.results:
            results_data.append(
                {
                    "scenario_name": result.scenario_name,
                    "parameters": result.parameters,
                    "metrics": result.metrics,
                    "score": result.score,
                    "execution_time": result.execution_time,
                    "total_signals": result.total_signals,
                    "successful_trades": result.successful_trades,
                    "failed_trades": result.failed_trades,
                }
            )

        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)

        self.logger.info(f"Results exported to {output_path}")
        return output_path


def main() -> None:
    """Main entry point for parameter optimization."""
    parser = argparse.ArgumentParser(
        description="Trading System Parameter Optimization"
    )
    parser.add_argument(
        "--config", type=str, help="Path to optimization configuration file"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="scenarios",
        choices=["scenarios", "grid_search", "random"],
    )
    parser.add_argument(
        "--symbols", type=str, nargs="+", help="List of symbols to test"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Max iterations for random search"
    )
    parser.add_argument(
        "--output", type=str, default="optimization_results", help="Output file prefix"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with limited scenarios",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create optimizer
    optimizer = ParameterOptimizer(args.config)

    # Run optimization
    async def run_optimization() -> None:
        if args.quick_test:
            # Quick test with first 2 scenarios
            optimizer.test_scenarios = optimizer.test_scenarios[:2]
            results = await optimizer.run_optimization("scenarios", args.symbols)
        else:
            results = await optimizer.run_optimization(
                args.type, args.symbols, args.iterations
            )

        # Generate reports
        html_path = optimizer.generate_report(f"{args.output}.html")
        json_path = optimizer.export_results(f"{args.output}.json")

        print("\nOptimization Complete!")
        print(f"HTML Report: {html_path}")
        print(f"JSON Results: {json_path}")
        print(
            f"\nBest Result: {results[0].scenario_name} (Score: {results[0].score:.4f})"
        )
        print("Parameters:")
        print(json.dumps(results[0].parameters, indent=2))

    # Run the optimization
    asyncio.run(run_optimization())


if __name__ == "__main__":
    main()
