import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.models import BacktestResult


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulations"""

    num_simulations: int = 1000
    time_horizon_days: int = 252
    confidence_levels: List[float] = field(
        default_factory=lambda: [0.05, 0.25, 0.75, 0.95]
    )
    random_seed: Optional[int] = None
    use_parallel_processing: bool = True
    max_workers: Optional[int] = None
    bootstrap_block_size: int = 20
    correlation_adjustment: bool = True
    fat_tail_adjustment: bool = True


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis"""

    training_period_days: int = 252  # 1 year training
    testing_period_days: int = 63  # 3 months testing
    step_size_days: int = 21  # Monthly steps
    min_training_samples: int = 100
    optimization_metric: str = "sharpe_ratio"  # or "total_return", "calmar_ratio"
    parameter_ranges: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    optimization_iterations: int = 50


@dataclass
class SimulationResult:
    """Result of a single Monte Carlo simulation"""

    simulation_id: int
    final_portfolio_value: float
    max_drawdown: float
    total_return: float
    volatility: float
    sharpe_ratio: float
    trades_count: int
    win_rate: float
    path_values: List[float] = field(default_factory=list)


@dataclass
class MonteCarloResults:
    """Aggregated Monte Carlo simulation results"""

    num_simulations: int
    time_horizon_days: int
    percentiles: Dict[float, float]
    expected_return: float
    expected_volatility: float
    probability_of_loss: float
    expected_shortfall_95: float
    value_at_risk_95: float
    max_drawdown_distribution: List[float]
    return_distribution: List[float]
    simulation_results: List[SimulationResult] = field(default_factory=list)


class MonteCarloEngine:
    """Monte Carlo simulation engine for portfolio analysis"""

    def __init__(self, config: MonteCarloConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        np.random.seed(config.random_seed)

    async def run_simulations(
        self,
        historical_returns: List[float],
        correlations: Optional[np.ndarray] = None,
        initial_value: float = 100000.0,
    ) -> MonteCarloResults:
        """Run Monte Carlo simulations"""
        self.logger.info(
            f"Running {self.config.num_simulations} Monte Carlo simulations"
        )

        # Estimate return distribution parameters
        return_params = self._estimate_return_distribution(historical_returns)

        # Run simulations
        if self.config.use_parallel_processing:
            simulation_results = await self._run_parallel_simulations(
                return_params, initial_value, correlations
            )
        else:
            simulation_results = await self._run_sequential_simulations(
                return_params, initial_value, correlations
            )

        # Aggregate results
        results = self._aggregate_simulation_results(simulation_results)

        self.logger.info(
            f"Monte Carlo completed. Expected return: {results.expected_return:.2%}"
        )
        self.logger.info(f"VaR 95%: ${results.value_at_risk_95:,.2f}")

        return results

    def _estimate_return_distribution(self, returns: List[float]) -> Dict[str, float]:
        """Estimate parameters of return distribution"""
        returns_array = np.array(returns)

        # Basic statistics
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)

        # Test for normality
        _, p_value = stats.jarque_bera(returns_array)
        # Note: is_normal can be used for distribution selection logic if needed

        # Note: skewness and kurtosis can be used for advanced distribution fitting if needed

        # Fit different distributions
        distributions = {}

        # Normal distribution
        distributions["normal"] = {
            "type": "normal",
            "mean": mean_return,
            "std": std_return,
            "aic": self._calculate_aic_normal(
                returns_array, float(mean_return), float(std_return)
            ),
        }

        # Student's t-distribution (for fat tails)
        if self.config.fat_tail_adjustment:
            try:
                df, loc, scale = stats.t.fit(returns_array)
                distributions["t"] = {
                    "type": "t",
                    "df": df,
                    "loc": loc,
                    "scale": scale,
                    "aic": self._calculate_aic_t(returns_array, df, loc, scale),
                }
            except Exception:
                distributions["t"] = distributions["normal"]

        # Choose best distribution based on AIC
        best_dist = min(distributions.values(), key=lambda x: x["aic"])

        self.logger.info(f"Selected distribution: {best_dist['type']}")
        return best_dist

    def _calculate_aic_normal(self, data: np.ndarray, mean: float, std: float) -> float:
        """Calculate AIC for normal distribution"""
        log_likelihood = np.sum(stats.norm.logpdf(data, loc=mean, scale=std))
        return 2 * 2 - 2 * log_likelihood  # 2 parameters

    def _calculate_aic_t(
        self, data: np.ndarray, df: float, loc: float, scale: float
    ) -> float:
        """Calculate AIC for t-distribution"""
        log_likelihood = np.sum(stats.t.logpdf(data, df=df, loc=loc, scale=scale))
        return 2 * 3 - 2 * log_likelihood  # 3 parameters

    async def _run_parallel_simulations(
        self,
        return_params: Dict[str, float],
        initial_value: float,
        correlations: Optional[np.ndarray],
    ) -> List[SimulationResult]:
        """Run simulations in parallel"""
        max_workers = self.config.max_workers or min(mp.cpu_count(), 8)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            tasks = []
            chunk_size = self.config.num_simulations // max_workers

            for i in range(max_workers):
                start_idx = i * chunk_size
                end_idx = (
                    start_idx + chunk_size
                    if i < max_workers - 1
                    else self.config.num_simulations
                )

                task = executor.submit(
                    self._run_simulation_chunk,
                    start_idx,
                    end_idx,
                    return_params,
                    initial_value,
                    correlations,
                )
                tasks.append(task)

            # Collect results
            all_results = []
            for task in tasks:
                chunk_results = task.result()
                all_results.extend(chunk_results)

        return all_results

    async def _run_sequential_simulations(
        self,
        return_params: Dict[str, float],
        initial_value: float,
        correlations: Optional[np.ndarray],
    ) -> List[SimulationResult]:
        """Run simulations sequentially"""
        return self._run_simulation_chunk(
            0, self.config.num_simulations, return_params, initial_value, correlations
        )

    def _run_simulation_chunk(
        self,
        start_idx: int,
        end_idx: int,
        return_params: Dict[str, float],
        initial_value: float,
        correlations: Optional[np.ndarray],
    ) -> List[SimulationResult]:
        """Run a chunk of simulations"""
        results = []

        for sim_id in range(start_idx, end_idx):
            # Generate random returns
            returns = self._generate_random_returns(return_params)

            # Apply correlation if provided
            if correlations is not None:
                returns = self._apply_correlation(returns, correlations)

            # Simulate portfolio path
            portfolio_path = self._simulate_portfolio_path(returns, initial_value)

            # Calculate metrics
            final_value = portfolio_path[-1]
            total_return = (final_value - initial_value) / initial_value
            max_drawdown = self._calculate_path_max_drawdown(portfolio_path)
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = self._calculate_path_sharpe_ratio(returns, 0.02)

            # Simulate trading statistics
            trades_count = np.random.poisson(50)  # Average 50 trades
            win_rate = np.random.beta(2, 2) * 0.4 + 0.3  # Win rate between 30-70%

            result = SimulationResult(
                simulation_id=sim_id,
                final_portfolio_value=final_value,
                max_drawdown=max_drawdown,
                total_return=total_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                trades_count=trades_count,
                win_rate=win_rate,
                path_values=portfolio_path,
            )

            results.append(result)

        return results

    def _generate_random_returns(self, return_params: Dict[str, float]) -> np.ndarray:
        """Generate random returns based on estimated distribution"""
        if return_params["type"] == "normal":
            returns = np.random.normal(
                return_params["mean"],
                return_params["std"],
                self.config.time_horizon_days,
            )
        elif return_params["type"] == "t":
            returns = stats.t.rvs(
                df=return_params["df"],
                loc=return_params["loc"],
                scale=return_params["scale"],
                size=self.config.time_horizon_days,
            )
        else:
            # Fallback to normal
            returns = np.random.normal(0.001, 0.02, self.config.time_horizon_days)

        # Ensure returns is numpy array before calling astype
        if isinstance(returns, (int, float)):
            returns = np.array([returns])
        return np.array(returns).astype(np.float64)

    def _apply_correlation(
        self, returns: np.ndarray, correlations: np.ndarray
    ) -> np.ndarray:
        """Apply correlation structure to returns"""
        # Cholesky decomposition for correlation
        try:
            chol = np.linalg.cholesky(correlations)
            correlated_returns = np.dot(chol, returns.reshape(-1, 1)).flatten()
            return correlated_returns
        except np.linalg.LinAlgError:
            # Fall back to original returns if correlation matrix is not positive definite
            return returns

    def _simulate_portfolio_path(
        self, returns: np.ndarray, initial_value: float
    ) -> List[float]:
        """Simulate portfolio value path"""
        portfolio_values = [initial_value]

        for daily_return in returns:
            new_value = portfolio_values[-1] * (1 + daily_return)
            portfolio_values.append(new_value)

        return portfolio_values

    def _calculate_path_max_drawdown(self, portfolio_path: List[float]) -> float:
        """Calculate maximum drawdown from portfolio path"""
        peak = portfolio_path[0]
        max_dd = 0.0

        for value in portfolio_path[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)

        return max_dd

    def _calculate_path_sharpe_ratio(
        self, returns: np.ndarray, risk_free_rate: float
    ) -> float:
        """Calculate Sharpe ratio from returns path"""
        excess_returns = returns - risk_free_rate / 252
        return (
            np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            if np.std(excess_returns) > 0
            else 0.0
        )

    def _aggregate_simulation_results(
        self, simulation_results: List[SimulationResult]
    ) -> MonteCarloResults:
        """Aggregate simulation results"""
        final_values = [r.final_portfolio_value for r in simulation_results]
        returns = [r.total_return for r in simulation_results]
        drawdowns = [r.max_drawdown for r in simulation_results]

        # Calculate percentiles
        percentiles = {}
        for conf_level in self.config.confidence_levels:
            percentiles[conf_level] = np.percentile(final_values, conf_level * 100)

        # Risk metrics
        var_95 = np.percentile(final_values, 5)  # 5th percentile
        expected_shortfall = np.mean([v for v in final_values if v <= var_95])

        results = MonteCarloResults(
            num_simulations=self.config.num_simulations,
            time_horizon_days=self.config.time_horizon_days,
            percentiles=percentiles,
            expected_return=float(np.mean(returns)),
            expected_volatility=float(np.std(returns)),
            probability_of_loss=len([r for r in returns if r < 0]) / len(returns),
            expected_shortfall_95=float(expected_shortfall),
            value_at_risk_95=float(var_95),
            max_drawdown_distribution=drawdowns,
            return_distribution=returns,
            simulation_results=simulation_results,
        )

        return results

    def generate_report(self, results: MonteCarloResults, output_path: str):
        """Generate Monte Carlo analysis report"""
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Return distribution
        axes[0, 0].hist(
            results.return_distribution, bins=50, alpha=0.7, edgecolor="black"
        )
        axes[0, 0].axvline(
            results.expected_return,
            color="red",
            linestyle="--",
            label="Expected Return",
        )
        axes[0, 0].axvline(
            np.percentile(results.return_distribution, 5),
            color="orange",
            linestyle="--",
            label="5th Percentile",
        )
        axes[0, 0].set_title("Return Distribution")
        axes[0, 0].set_xlabel("Total Return")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()

        # Drawdown distribution
        axes[0, 1].hist(
            results.max_drawdown_distribution, bins=50, alpha=0.7, edgecolor="black"
        )
        axes[0, 1].axvline(
            np.mean(results.max_drawdown_distribution),
            color="red",
            linestyle="--",
            label="Mean Drawdown",
        )
        axes[0, 1].axvline(
            np.percentile(results.max_drawdown_distribution, 95),
            color="orange",
            linestyle="--",
            label="95th Percentile",
        )
        axes[0, 1].set_title("Maximum Drawdown Distribution")
        axes[0, 1].set_xlabel("Maximum Drawdown")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].legend()

        # Portfolio value paths (sample)
        sample_indices = np.random.choice(
            len(results.simulation_results),
            size=min(100, len(results.simulation_results)),
            replace=False,
        )
        sample_paths = [results.simulation_results[i] for i in sample_indices]
        for result in sample_paths:
            if result.path_values:
                axes[1, 0].plot(result.path_values, alpha=0.1, color="blue")

        axes[1, 0].set_title("Sample Portfolio Paths")
        axes[1, 0].set_xlabel("Days")
        axes[1, 0].set_ylabel("Portfolio Value")

        # Risk-return scatter
        returns = [r.total_return for r in results.simulation_results]
        volatilities = [r.volatility for r in results.simulation_results]
        axes[1, 1].scatter(volatilities, returns, alpha=0.6)
        axes[1, 1].set_title("Risk-Return Profile")
        axes[1, 1].set_xlabel("Volatility")
        axes[1, 1].set_ylabel("Total Return")

        plt.tight_layout()
        plt.savefig(
            f"{output_path}/monte_carlo_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Generate text report
        report = self._generate_text_report(results)
        with open(f"{output_path}/monte_carlo_report.txt", "w") as f:
            f.write(report)

    def _generate_text_report(self, results: MonteCarloResults) -> str:
        """Generate text report of Monte Carlo results"""
        report = f"""
Monte Carlo Simulation Results
==============================

Simulation Parameters:
- Number of simulations: {results.num_simulations:,}
- Time horizon: {results.time_horizon_days} days
- Random seed: {self.config.random_seed}

Expected Performance:
- Expected return: {results.expected_return:.2%}
- Expected volatility: {results.expected_volatility:.2%}
- Probability of loss: {results.probability_of_loss:.2%}

Risk Metrics:
- Value at Risk (95%): ${results.value_at_risk_95:,.2f}
- Expected Shortfall (95%): ${results.expected_shortfall_95:,.2f}
- Mean maximum drawdown: {np.mean(results.max_drawdown_distribution):.2%}
- 95th percentile drawdown: {np.percentile(results.max_drawdown_distribution, 95):.2%}

Portfolio Value Percentiles:
"""
        for conf_level, value in results.percentiles.items():
            report += f"- {conf_level * 100:>5.1f}%: ${value:>10,.2f}\n"

        return report


class WalkForwardAnalysis:
    """Walk-forward analysis for strategy optimization and validation"""

    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def run_analysis(
        self,
        strategy_class: type,
        symbols: List[str],
        data_source: Any,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Run walk-forward analysis"""
        self.logger.info(
            f"Starting walk-forward analysis from {start_date} to {end_date}"
        )

        # Generate time windows
        windows = self._generate_time_windows(start_date, end_date)
        self.logger.info(f"Generated {len(windows)} walk-forward windows")

        # Run optimization for each window
        optimization_results = []
        validation_results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            self.logger.info(
                f"Processing window {i + 1}/{len(windows)}: {train_start.date()} - {test_end.date()}"
            )

            # Optimize strategy parameters on training data
            best_params = await self._optimize_strategy_parameters(
                strategy_class, symbols, data_source, train_start, train_end
            )

            optimization_results.append(
                {
                    "window": i,
                    "train_period": (train_start, train_end),
                    "best_parameters": best_params["parameters"],
                    "optimization_score": best_params["score"],
                }
            )

            # Validate on out-of-sample data
            validation_result = await self._validate_strategy(
                strategy_class,
                symbols,
                data_source,
                test_start,
                test_end,
                best_params["parameters"],
            )

            validation_results.append(
                {
                    "window": i,
                    "test_period": (test_start, test_end),
                    "performance": validation_result,
                }
            )

        # Aggregate results
        analysis_results = self._aggregate_walk_forward_results(
            optimization_results, validation_results
        )

        return analysis_results

    def _generate_time_windows(
        self, start_date: datetime, end_date: datetime
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate overlapping time windows for walk-forward analysis"""
        windows = []
        current_date = start_date

        while (
            current_date
            + timedelta(
                days=self.config.training_period_days + self.config.testing_period_days
            )
            <= end_date
        ):
            train_start = current_date
            train_end = current_date + timedelta(days=self.config.training_period_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.config.testing_period_days)

            windows.append((train_start, train_end, test_start, test_end))

            current_date += timedelta(days=self.config.step_size_days)

        return windows

    async def _optimize_strategy_parameters(
        self,
        strategy_class: type,
        symbols: List[str],
        data_source: Any,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Optimize strategy parameters using grid search or genetic algorithm"""
        best_score = float("-inf")
        best_params = {}

        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations()

        for params in param_combinations[: self.config.optimization_iterations]:
            try:
                # Create strategy instance with parameters
                strategy = strategy_class(**params)

                # Run backtest
                from .backtest_engine import BacktestConfig, BacktestEngine

                backtest_config = BacktestConfig(
                    start_date=start_date, end_date=end_date, initial_capital=100000.0
                )

                engine = BacktestEngine(backtest_config)
                result = await engine.run_backtest(strategy, symbols, data_source)

                # Score based on optimization metric
                score = self._score_result(result)

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                self.logger.warning(f"Failed to optimize with params {params}: {e}")
                continue

        return {"parameters": best_params, "score": best_score}

    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations for optimization"""
        combinations = []

        # Example for moving average strategy
        if "short_window" in self.config.parameter_ranges:
            short_min, short_max = self.config.parameter_ranges["short_window"]
            long_min, long_max = self.config.parameter_ranges["long_window"]

            for short in range(short_min, short_max + 1, 5):
                for long in range(max(short + 5, long_min), long_max + 1, 10):
                    combinations.append({"short_window": short, "long_window": long})

        # Example for RSI strategy
        elif "period" in self.config.parameter_ranges:
            period_min, period_max = self.config.parameter_ranges["period"]
            oversold_min, oversold_max = self.config.parameter_ranges.get(
                "oversold", (20, 35)
            )
            overbought_min, overbought_max = self.config.parameter_ranges.get(
                "overbought", (65, 80)
            )

            for period in range(period_min, period_max + 1, 2):
                for oversold in range(oversold_min, oversold_max + 1, 5):
                    for overbought in range(overbought_min, overbought_max + 1, 5):
                        if overbought > oversold + 20:  # Ensure reasonable spread
                            combinations.append(
                                {
                                    "period": period,
                                    "oversold": oversold,
                                    "overbought": overbought,
                                }
                            )

        # Random sampling if too many combinations
        if len(combinations) > self.config.optimization_iterations:
            combinations = np.random.choice(
                combinations, size=self.config.optimization_iterations, replace=False
            ).tolist()

        return combinations

    def _score_result(self, result: BacktestResult) -> float:
        """Score backtest result based on optimization metric"""
        if self.config.optimization_metric == "sharpe_ratio":
            return result.sharpe_ratio if result.sharpe_ratio is not None else 0.0
        elif self.config.optimization_metric == "total_return":
            return float(result.total_return)
        elif self.config.optimization_metric == "sortino_ratio":
            return result.sortino_ratio if result.sortino_ratio is not None else 0.0
        elif self.config.optimization_metric == "profit_factor":
            return result.profit_factor if result.profit_factor is not None else 0.0
        else:
            # Custom composite score
            sharpe = result.sharpe_ratio if result.sharpe_ratio is not None else 0.0
            total_ret = float(result.total_return)
            # Note: calmar_ratio is not in BacktestResult model, using sortino instead
            sortino = result.sortino_ratio if result.sortino_ratio is not None else 0.0
            return sharpe * 0.4 + total_ret * 0.3 + sortino * 0.3

    async def _validate_strategy(
        self,
        strategy_class: type,
        symbols: List[str],
        data_source: Any,
        start_date: datetime,
        end_date: datetime,
        parameters: Dict[str, Any],
    ) -> BacktestResult:
        """Validate strategy on out-of-sample data"""
        strategy = strategy_class(**parameters)

        from .backtest_engine import BacktestConfig, BacktestEngine

        backtest_config = BacktestConfig(
            start_date=start_date, end_date=end_date, initial_capital=100000.0
        )

        engine = BacktestEngine(backtest_config)
        result = await engine.run_backtest(strategy, symbols, data_source)

        return result

    def _aggregate_walk_forward_results(
        self, optimization_results: List[Dict], validation_results: List[Dict]
    ) -> Dict[str, Any]:
        """Aggregate walk-forward analysis results"""
        # Calculate stability metrics
        optimization_scores = [r["optimization_score"] for r in optimization_results]
        validation_scores = [r["performance"].sharpe_ratio for r in validation_results]

        # Parameter stability
        parameter_stability = self._calculate_parameter_stability(optimization_results)

        # Performance degradation from in-sample to out-of-sample
        performance_degradation = np.mean(
            [
                opt_score - val_score
                for opt_score, val_score in zip(optimization_scores, validation_scores)
            ]
        )

        # Aggregate out-of-sample performance
        total_returns = [r["performance"].total_return for r in validation_results]
        sharpe_ratios = [r["performance"].sharpe_ratio for r in validation_results]
        max_drawdowns = [r["performance"].max_drawdown for r in validation_results]

        return {
            "summary": {
                "total_windows": len(validation_results),
                "avg_out_of_sample_return": np.mean(total_returns),
                "avg_out_of_sample_sharpe": np.mean(sharpe_ratios),
                "avg_max_drawdown": np.mean(max_drawdowns),
                "performance_degradation": performance_degradation,
                "parameter_stability": parameter_stability,
            },
            "detailed_results": {
                "optimization_results": optimization_results,
                "validation_results": validation_results,
            },
            "stability_metrics": {
                "return_consistency": np.std(total_returns),
                "sharpe_consistency": np.std(sharpe_ratios),
                "drawdown_consistency": np.std(max_drawdowns),
            },
        }

    def _calculate_parameter_stability(self, optimization_results: List[Dict]) -> float:
        """Calculate parameter stability across windows"""
        if len(optimization_results) < 2:
            return 1.0

        # Extract parameter values for each window
        all_params = [r["best_parameters"] for r in optimization_results]

        # Calculate stability for each parameter
        param_names = set()
        for params in all_params:
            param_names.update(params.keys())

        stability_scores = []

        for param_name in param_names:
            values = [params.get(param_name, 0) for params in all_params]

            if all(isinstance(v, (int, float)) for v in values):
                # Numerical parameter - calculate coefficient of variation
                if np.mean(values) != 0:
                    cv = np.std(values) / abs(np.mean(values))
                    stability = max(0, 1 - cv)
                else:
                    stability = 1.0
            else:
                # Categorical parameter - calculate mode frequency
                from collections import Counter

                counter = Counter(values)
                most_common_freq = counter.most_common(1)[0][1]
                stability = most_common_freq / len(values)

            stability_scores.append(stability)

        return float(np.mean(stability_scores)) if stability_scores else 1.0


class BootstrapAnalysis:
    """Bootstrap analysis for statistical significance testing"""

    def __init__(self, num_bootstrap_samples: int = 1000, block_size: int = 20):
        self.num_bootstrap_samples = num_bootstrap_samples
        self.block_size = block_size
        self.logger = logging.getLogger(__name__)

    async def bootstrap_backtest_results(self, returns: List[float]) -> Dict[str, Any]:
        """Bootstrap backtest results for confidence intervals"""
        bootstrap_results = {
            "mean_returns": [],
            "volatilities": [],
            "sharpe_ratios": [],
            "max_drawdowns": [],
        }

        returns_array = np.array(returns)

        for i in range(self.num_bootstrap_samples):
            # Block bootstrap to preserve time series structure
            bootstrap_returns = self._block_bootstrap_sample(returns_array)

            # Calculate metrics
            mean_return = np.mean(bootstrap_returns)
            volatility = np.std(bootstrap_returns) * np.sqrt(252)
            sharpe_ratio = self._calculate_sharpe_ratio(bootstrap_returns)
            max_drawdown = self._calculate_max_drawdown_from_returns(bootstrap_returns)

            bootstrap_results["mean_returns"].append(mean_return)
            bootstrap_results["volatilities"].append(volatility)
            bootstrap_results["sharpe_ratios"].append(sharpe_ratio)
            bootstrap_results["max_drawdowns"].append(max_drawdown)

        # Calculate confidence intervals
        confidence_intervals = {}
        for metric, values in bootstrap_results.items():
            confidence_intervals[metric] = {
                "2.5%": np.percentile(values, 2.5),
                "97.5%": np.percentile(values, 97.5),
                "mean": np.mean(values),
                "std": np.std(values),
            }

        return {
            "bootstrap_samples": bootstrap_results,
            "confidence_intervals": confidence_intervals,
            "statistical_significance": self._test_statistical_significance(
                bootstrap_results
            ),
        }

    def _block_bootstrap_sample(self, returns: np.ndarray) -> np.ndarray:
        """Generate block bootstrap sample"""
        n = len(returns)
        num_blocks = n // self.block_size

        # Sample random starting points for blocks
        start_points = np.random.choice(
            n - self.block_size + 1, size=num_blocks, replace=True
        )

        bootstrap_sample = []
        for start in start_points:
            block = returns[start : start + self.block_size]
            bootstrap_sample.extend(block)

        # Trim to original length
        return np.array(bootstrap_sample[:n])

    def _calculate_sharpe_ratio(
        self, returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio from returns"""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        return (
            float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))
            if np.std(excess_returns) > 0
            else 0.0
        )

    def _calculate_max_drawdown_from_returns(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns"""
        if len(returns) == 0:
            return 0.0

        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.abs(np.min(drawdown)))

    def _test_statistical_significance(
        self, bootstrap_results: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Test statistical significance of bootstrap results"""
        significance_tests = {}

        for metric, values in bootstrap_results.items():
            if len(values) > 0:
                # Test if mean is significantly different from zero
                try:
                    t_stat, p_value = stats.ttest_1samp(values, 0)  # type: ignore
                    # Ensure numeric types for calculations
                    t_stat = float(t_stat)  # type: ignore
                    p_value = float(p_value)  # type: ignore
                except Exception:
                    t_stat = 0.0
                    p_value = 1.0

                significance_tests[metric] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "is_significant": p_value < 0.05,
                }

        return significance_tests
