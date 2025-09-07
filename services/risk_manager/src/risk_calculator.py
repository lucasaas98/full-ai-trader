"""
Risk Metrics Calculator

This module provides comprehensive risk calculations using real market data including:
- Value at Risk (VaR) using multiple methods
- Expected Shortfall (Conditional VaR)
- Portfolio volatility and correlation matrices
- Beta calculations and factor exposures
- Stress testing and scenario analysis
"""

import logging
import math
import statistics
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from scipy import stats

from shared.config import get_config
from shared.models import PortfolioState, Position

logger = logging.getLogger(__name__)


class RiskCalculator:
    """
    Advanced risk metrics calculator using real market data.

    This class provides comprehensive risk management capabilities including:

    CORE RISK METRICS:
    - Value at Risk (VaR) using historical, parametric, and Monte Carlo methods
    - Expected Shortfall (Conditional VaR)
    - Component VaR for individual positions
    - Portfolio volatility using full covariance matrices
    - Maximum drawdown calculations

    PORTFOLIO ANALYTICS:
    - Portfolio beta relative to benchmarks
    - Tracking error calculations
    - Correlation matrix analysis
    - Factor exposures (sector, style, geography)
    - Concentration risk metrics

    ADVANCED FEATURES:
    - Liquidity risk assessment with position scoring
    - VaR model backtesting and validation
    - Risk-adjusted returns (Sharpe, Sortino, Calmar, Information ratios)
    - Options Greeks calculations (Delta, Gamma, Theta, Vega)
    - Enhanced stress testing with predefined scenarios
    - Risk attribution by factors and sectors

    STRESS TESTING SCENARIOS:
    - Market crash simulations (2008-style)
    - Sector-specific crashes (tech bubble burst)
    - Interest rate shock scenarios
    - Volatility spike simulations
    - Liquidity crisis scenarios

    All calculations use real market data when available and include comprehensive
    error handling with fallback methods. Results include detailed breakdowns
    and position-level analytics for thorough risk management.

    TODO: Add comprehensive test coverage for all new risk calculation features
    """

    def __init__(self, alpaca_client=None):
        """Initialize risk calculator."""
        self.config = get_config()
        self.alpaca_client = alpaca_client

        # Risk calculation parameters
        self.confidence_levels = [0.95, 0.99]  # 95% and 99% confidence
        self.var_methods = ["historical", "parametric", "monte_carlo"]
        self.lookback_days = 252  # 1 year of trading days

        # Market data cache
        self.price_data_cache: Dict[str, pd.DataFrame] = {}
        self.correlation_matrix_cache: Optional[pd.DataFrame] = None
        self.cache_timestamp: Optional[datetime] = None
        self.cache_ttl = timedelta(hours=1)

    async def calculate_portfolio_var(
        self,
        portfolio: PortfolioState,
        confidence_level: float = 0.95,
        method: str = "historical",
        holding_period: int = 1,
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate Value at Risk for the portfolio.

        Args:
            portfolio: Current portfolio state
            confidence_level: Confidence level (0.95 or 0.99)
            method: VaR method ("historical", "parametric", "monte_carlo")
            holding_period: Holding period in days

        Returns:
            Tuple of (VaR, Expected Shortfall)
        """
        try:
            symbols = [p.symbol for p in portfolio.positions if p.quantity != 0]

            if not symbols:
                return Decimal("0"), Decimal("0")

            # Get market data for all positions
            market_data = await self._get_portfolio_market_data(symbols)

            if not market_data:
                return await self._fallback_var_calculation(portfolio, confidence_level)

            # Calculate VaR based on method
            if method == "historical":
                var, es = await self._historical_var(
                    portfolio, market_data, confidence_level, holding_period
                )
            elif method == "parametric":
                var, es = await self._parametric_var(
                    portfolio, market_data, confidence_level, holding_period
                )
            elif method == "monte_carlo":
                var, es = await self._monte_carlo_var(
                    portfolio, market_data, confidence_level, holding_period
                )
            else:
                logger.warning(f"Unknown VaR method: {method}, using historical")
                var, es = await self._historical_var(
                    portfolio, market_data, confidence_level, holding_period
                )

            logger.debug(
                f"Portfolio VaR ({method}, {confidence_level:.0%}): ${var}, ES: ${es}"
            )
            return var, es

        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return await self._fallback_var_calculation(portfolio, confidence_level)

    async def calculate_correlation_matrix(
        self, symbols: List[str], days: int = 60
    ) -> pd.DataFrame:
        """Calculate correlation matrix for given symbols."""
        try:
            # Check cache
            if (
                self.correlation_matrix_cache is not None
                and self.cache_timestamp
                and datetime.now(timezone.utc) - self.cache_timestamp < self.cache_ttl
            ):
                return self.correlation_matrix_cache.loc[symbols, symbols]

            # Get price data for all symbols
            price_data = {}
            for symbol in symbols:
                prices = await self._get_symbol_returns(symbol, days)
                if prices:
                    price_data[symbol] = prices

            if len(price_data) < 2:
                # Return identity matrix if insufficient data
                return pd.DataFrame(
                    np.eye(len(symbols)), index=symbols, columns=symbols
                )

            # Create returns DataFrame
            returns_df = pd.DataFrame(price_data)

            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()

            # Fill NaN values with 0
            correlation_matrix = correlation_matrix.fillna(0)

            # Cache the result
            self.correlation_matrix_cache = correlation_matrix
            self.cache_timestamp = datetime.now(timezone.utc)

            logger.debug(f"Correlation matrix calculated for {len(symbols)} symbols")
            return correlation_matrix

        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            # Return identity matrix as fallback
            return pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)

    async def calculate_portfolio_volatility(self, portfolio: PortfolioState) -> float:
        """Calculate portfolio volatility using full covariance matrix."""
        try:
            symbols = [p.symbol for p in portfolio.positions if p.quantity != 0]

            if not symbols:
                return 0.0

            # Get weights
            weights = await self._calculate_position_weights(portfolio)

            # Get correlation matrix
            correlation_matrix = await self.calculate_correlation_matrix(symbols)

            # Get individual volatilities
            volatilities = {}
            for symbol in symbols:
                vol = await self._calculate_symbol_volatility(symbol)
                volatilities[symbol] = vol

            # Create volatility vector
            vol_vector = np.array([volatilities.get(symbol, 0.2) for symbol in symbols])

            # Create covariance matrix
            corr_matrix = correlation_matrix.values
            cov_matrix = np.outer(vol_vector, vol_vector) * corr_matrix

            # Calculate portfolio volatility
            weights_array = np.array([weights.get(symbol, 0.0) for symbol in symbols])
            portfolio_variance = np.dot(
                weights_array.T, np.dot(cov_matrix, weights_array)
            )
            portfolio_volatility = np.sqrt(portfolio_variance)

            return float(portfolio_volatility)

        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.15  # Default volatility

    async def calculate_portfolio_beta(
        self, portfolio: PortfolioState, benchmark: str = "SPY"
    ) -> float:
        """Calculate portfolio beta relative to benchmark."""
        try:
            symbols = [p.symbol for p in portfolio.positions if p.quantity != 0]

            if not symbols:
                return 1.0

            # Get position weights
            weights = await self._calculate_position_weights(portfolio)

            # Calculate weighted beta
            portfolio_beta = 0.0

            for symbol in symbols:
                weight = weights.get(symbol, 0.0)
                symbol_beta = await self._calculate_symbol_beta(symbol, benchmark)
                portfolio_beta += weight * symbol_beta

            return portfolio_beta

        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {e}")
            return 1.0

    async def stress_test_portfolio(
        self, portfolio: PortfolioState, scenarios: List[Dict]
    ) -> Dict:
        """
        Perform stress testing on portfolio under various scenarios.

        Args:
            portfolio: Current portfolio state
            scenarios: List of stress scenarios

        Returns:
            Dictionary with stress test results
        """
        try:
            results = {}

            for i, scenario in enumerate(scenarios):
                scenario_name = scenario.get("name", f"scenario_{i}")
                market_shocks = scenario.get("shocks", {})

                scenario_pnl = Decimal("0")

                for position in portfolio.positions:
                    if position.quantity == 0:
                        continue

                    symbol = position.symbol
                    shock = market_shocks.get(symbol, 0.0)  # Default no shock

                    # Calculate P&L under shock
                    shocked_price = position.current_price * (
                        Decimal("1") + Decimal(str(shock))
                    )
                    position_pnl = (shocked_price - position.current_price) * Decimal(
                        position.quantity
                    )
                    scenario_pnl += position_pnl

                results[scenario_name] = {
                    "total_pnl": str(scenario_pnl),
                    "pnl_percentage": (
                        float(scenario_pnl / portfolio.total_equity)
                        if portfolio.total_equity > 0
                        else 0.0
                    ),
                    "scenario_details": scenario,
                }

            return results

        except Exception as e:
            logger.error(f"Error in stress testing: {e}")
            return {}

    async def calculate_liquidity_risk(self, portfolio: PortfolioState) -> Dict:
        """
        Calculate liquidity risk metrics for the portfolio.

        Returns:
            Dictionary containing liquidity metrics
        """
        # TODO: Add comprehensive tests for liquidity risk calculations
        try:
            liquidity_metrics = {
                "portfolio_liquidity_score": 0.0,
                "position_liquidity": {},
                "illiquid_positions": [],
                "concentration_risk": 0.0,
            }

            total_value = float(portfolio.total_equity)
            if total_value <= 0:
                return liquidity_metrics

            illiquid_threshold = 0.05  # 5% of portfolio

            for position in portfolio.positions:
                if position.quantity == 0:
                    continue

                symbol = position.symbol
                position_weight = float(abs(position.market_value)) / total_value

                # Simple liquidity scoring based on symbol characteristics
                # TODO: Integrate with real market data for bid-ask spreads, volume, etc.
                liquidity_score = self._calculate_position_liquidity_score(symbol)

                position_liquidity_dict = cast(
                    Dict[str, Any], liquidity_metrics["position_liquidity"]
                )
                position_liquidity_dict[symbol] = {
                    "score": liquidity_score,
                    "weight": position_weight,
                    "market_value": float(position.market_value),
                }

                # Flag illiquid positions
                if liquidity_score < 0.3 and position_weight > illiquid_threshold:
                    illiquid_positions_list = cast(
                        List[Any], liquidity_metrics["illiquid_positions"]
                    )
                    illiquid_positions_list.append(
                        {
                            "symbol": symbol,
                            "weight": position_weight,
                            "liquidity_score": liquidity_score,
                        }
                    )

            # Calculate portfolio-weighted liquidity score
            weighted_score = 0.0
            position_liquidity_dict = cast(
                Dict[str, Any], liquidity_metrics["position_liquidity"]
            )
            for symbol, data in position_liquidity_dict.items():
                weighted_score += data["score"] * data["weight"]

            liquidity_metrics["portfolio_liquidity_score"] = weighted_score
            position_liquidity_dict = cast(
                Dict[str, Any], liquidity_metrics["position_liquidity"]
            )
            liquidity_metrics["concentration_risk"] = max(
                [data["weight"] for data in position_liquidity_dict.values()],
                default=0.0,
            )

            return liquidity_metrics

        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {e}")
            return {"error": str(e)}

    def _calculate_position_liquidity_score(self, symbol: str) -> float:
        """Calculate liquidity score for a position (0-1, higher is more liquid)."""
        # TODO: Add comprehensive tests for position liquidity scoring
        # Simple implementation - in production, use real market data
        liquid_symbols = {"SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"}
        moderate_liquid = {"JPM", "BAC", "WFC", "GS", "C"}

        if symbol in liquid_symbols:
            return 0.9
        elif symbol in moderate_liquid:
            return 0.7
        else:
            return 0.5  # Default moderate liquidity

    async def backtest_var_model(
        self,
        portfolio_history: List[Dict],
        var_predictions: List[float],
        confidence_level: float = 0.95,
    ) -> Dict:
        """
        Backtest VaR model performance using historical data.

        Args:
            portfolio_history: List of historical portfolio snapshots
            var_predictions: List of VaR predictions for each period
            confidence_level: VaR confidence level

        Returns:
            Dictionary with backtesting results
        """
        # TODO: Add comprehensive tests for VaR backtesting
        try:
            if (
                len(portfolio_history) != len(var_predictions)
                or len(portfolio_history) < 2
            ):
                return {"error": "Insufficient or mismatched data for backtesting"}

            violations = 0
            total_observations = len(portfolio_history) - 1
            actual_returns = []

            # Calculate actual portfolio returns
            for i in range(1, len(portfolio_history)):
                prev_value = portfolio_history[i - 1].get("total_equity", 0)
                curr_value = portfolio_history[i].get("total_equity", 0)

                if prev_value > 0:
                    actual_return = (curr_value - prev_value) / prev_value
                    actual_returns.append(actual_return)

                    # Check for VaR violations (losses exceeding VaR)
                    if i - 1 < len(var_predictions):
                        var_threshold = (
                            var_predictions[i - 1] / prev_value
                        )  # Convert to return
                        if actual_return < -abs(var_threshold):
                            violations += 1

            # Calculate backtesting statistics
            violation_rate = (
                violations / total_observations if total_observations > 0 else 0
            )
            expected_violation_rate = 1 - confidence_level

            # Kupiec test for coverage
            kupiec_stat = (
                -2
                * np.log(
                    (
                        (expected_violation_rate**violations)
                        * (
                            (1 - expected_violation_rate)
                            ** (total_observations - violations)
                        )
                    )
                    / (
                        (violation_rate**violations)
                        * ((1 - violation_rate) ** (total_observations - violations))
                    )
                )
                if violation_rate > 0 and violation_rate < 1
                else 0
            )

            return {
                "total_observations": total_observations,
                "violations": violations,
                "violation_rate": violation_rate,
                "expected_violation_rate": expected_violation_rate,
                "kupiec_statistic": kupiec_stat,
                "model_adequate": abs(violation_rate - expected_violation_rate) < 0.05,
                "average_actual_return": (
                    np.mean(actual_returns) if actual_returns else 0
                ),
                "volatility": np.std(actual_returns) if actual_returns else 0,
            }

        except Exception as e:
            logger.error(f"Error in VaR backtesting: {e}")
            return {"error": str(e)}

    async def calculate_risk_adjusted_returns(
        self,
        portfolio: PortfolioState,
        benchmark_returns: Optional[List[float]] = None,
        risk_free_rate: float = 0.02,
    ) -> Dict:
        """
        Calculate risk-adjusted return metrics.

        Args:
            portfolio: Current portfolio state
            benchmark_returns: Optional benchmark return series
            risk_free_rate: Risk-free rate (annual)

        Returns:
            Dictionary with risk-adjusted metrics
        """
        # TODO: Add comprehensive tests for risk-adjusted returns
        try:
            symbols = [p.symbol for p in portfolio.positions if p.quantity != 0]
            if not symbols:
                return {}

            # Get portfolio returns (simplified - using recent performance)
            portfolio_returns = await self._calculate_portfolio_returns(portfolio)

            if not portfolio_returns or len(portfolio_returns) < 10:
                return {"error": "Insufficient return data"}

            returns_array = np.array(portfolio_returns)
            daily_risk_free = risk_free_rate / 252  # Convert annual to daily

            # Calculate metrics
            mean_return = np.mean(returns_array)
            volatility = np.std(returns_array)
            downside_returns = returns_array[returns_array < 0]
            downside_volatility = (
                np.std(downside_returns) if len(downside_returns) > 0 else 0
            )

            # Sharpe Ratio
            sharpe_ratio = (
                (mean_return - daily_risk_free) / volatility if volatility > 0 else 0
            )

            # Sortino Ratio
            sortino_ratio = (
                (mean_return - daily_risk_free) / downside_volatility
                if downside_volatility > 0
                else 0
            )

            # Calmar Ratio (return/max drawdown)
            max_dd, _, _ = await self.calculate_maximum_drawdown(portfolio_returns)
            calmar_ratio = (
                (mean_return * 252) / abs(max_dd) if abs(max_dd) > 0.01 else 0
            )

            # Information Ratio (if benchmark provided)
            information_ratio = 0
            if benchmark_returns and len(benchmark_returns) >= len(returns_array):
                benchmark_array = np.array(benchmark_returns[: len(returns_array)])
                excess_returns = returns_array - benchmark_array
                tracking_error = np.std(excess_returns)
                information_ratio = (
                    np.mean(excess_returns) / tracking_error
                    if tracking_error > 0
                    else 0
                )

            return {
                "mean_return": float(mean_return * 252),  # Annualized
                "volatility": float(volatility * np.sqrt(252)),  # Annualized
                "sharpe_ratio": float(sharpe_ratio * np.sqrt(252)),  # Annualized
                "sortino_ratio": float(sortino_ratio * np.sqrt(252)),  # Annualized
                "calmar_ratio": float(calmar_ratio),
                "information_ratio": float(
                    information_ratio * np.sqrt(252)
                ),  # Annualized
                "max_drawdown": float(max_dd),
                "downside_volatility": float(downside_volatility * np.sqrt(252)),
            }

        except Exception as e:
            logger.error(f"Error calculating risk-adjusted returns: {e}")
            return {"error": str(e)}

    async def calculate_options_greeks(self, options_positions: List[Dict]) -> Dict:
        """
        Calculate basic options Greeks for options positions.

        Args:
            options_positions: List of options position data

        Returns:
            Dictionary with Greeks calculations
        """
        # TODO: Add comprehensive tests for options Greeks calculations
        try:
            greeks_summary = {
                "portfolio_delta": 0.0,
                "portfolio_gamma": 0.0,
                "portfolio_theta": 0.0,
                "portfolio_vega": 0.0,
                "position_greeks": {},
            }

            if not options_positions:
                return greeks_summary

            for position in options_positions:
                symbol = position.get("symbol", "")
                quantity = position.get("quantity", 0)
                option_type = position.get("option_type", "call")  # call or put
                strike = position.get("strike", 0)
                underlying_price = position.get("underlying_price", 0)
                time_to_expiry = position.get("days_to_expiry", 30) / 365.0
                volatility = position.get("implied_volatility", 0.2)
                risk_free_rate = position.get("risk_free_rate", 0.02)

                if strike <= 0 or underlying_price <= 0:
                    continue

                # Calculate Greeks using Black-Scholes approximations
                greeks = self._calculate_black_scholes_greeks(
                    underlying_price,
                    strike,
                    time_to_expiry,
                    risk_free_rate,
                    volatility,
                    option_type,
                )

                # Scale by position size
                position_greeks = {
                    "delta": greeks["delta"] * quantity,
                    "gamma": greeks["gamma"] * quantity,
                    "theta": greeks["theta"] * quantity,
                    "vega": greeks["vega"] * quantity,
                }

                position_greeks_dict = cast(
                    Dict[str, Any], greeks_summary["position_greeks"]
                )
                position_greeks_dict[symbol] = position_greeks

                # Add to portfolio totals
                greeks_summary["portfolio_delta"] += position_greeks["delta"]
                greeks_summary["portfolio_gamma"] += position_greeks["gamma"]
                greeks_summary["portfolio_theta"] += position_greeks["theta"]
                greeks_summary["portfolio_vega"] += position_greeks["vega"]

            return greeks_summary

        except Exception as e:
            logger.error(f"Error calculating options Greeks: {e}")
            return {"error": str(e)}

    def _calculate_black_scholes_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> Dict:
        """Calculate Black-Scholes Greeks."""
        # TODO: Add comprehensive tests for Black-Scholes Greeks
        try:
            if T <= 0 or sigma <= 0:
                return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option_type.lower() == "call":
                delta = stats.norm.cdf(d1)
                theta = -(
                    S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                    + r * K * np.exp(-r * T) * stats.norm.cdf(d2)
                )
            else:  # put
                delta = stats.norm.cdf(d1) - 1
                theta = -(
                    S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                    - r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
                )

            gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * stats.norm.pdf(d1) * np.sqrt(T)

            return {
                "delta": delta,
                "gamma": gamma,
                "theta": theta / 365.0,  # Per day
                "vega": vega / 100.0,  # Per 1% volatility change
            }

        except Exception as e:
            logger.error(f"Error in Black-Scholes Greeks calculation: {e}")
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

    async def enhanced_stress_test(self, portfolio: PortfolioState) -> Dict:
        """
        Perform enhanced stress testing with predefined scenarios.

        Returns:
            Dictionary with comprehensive stress test results
        """
        # TODO: Add comprehensive tests for enhanced stress testing
        try:
            # Define common stress scenarios
            stress_scenarios: List[Dict[str, Any]] = [
                {
                    "name": "market_crash_2008",
                    "description": "2008-style market crash",
                    "shocks": {"SPY": -0.37, "QQQ": -0.42, "IWM": -0.33},
                    "correlation_shock": 0.8,  # Correlations increase in crisis
                },
                {
                    "name": "tech_bubble_burst",
                    "description": "Technology sector crash",
                    "shocks": {
                        "AAPL": -0.5,
                        "MSFT": -0.45,
                        "GOOGL": -0.6,
                        "TSLA": -0.7,
                        "QQQ": -0.4,
                    },
                },
                {
                    "name": "interest_rate_shock",
                    "description": "Rapid interest rate increase",
                    "shocks": {"SPY": -0.15, "bonds": -0.25, "real_estate": -0.3},
                },
                {
                    "name": "volatility_spike",
                    "description": "VIX spikes to 40+",
                    "volatility_multiplier": 2.5,
                },
                {
                    "name": "liquidity_crisis",
                    "description": "Liquidity dries up",
                    "bid_ask_widening": 3.0,
                    "general_shock": -0.2,
                },
            ]

            results = {}
            base_portfolio_value = float(portfolio.total_equity)

            for scenario in stress_scenarios:
                scenario_name = str(scenario["name"])
                scenario_pnl = 0.0
                position_impacts = {}

                # Apply shocks to individual positions
                positions_list = (
                    cast(List[Any], portfolio.positions)
                    if isinstance(portfolio.positions, list)
                    else list(portfolio.positions.values())
                )
                for position in positions_list:
                    if position.quantity == 0:
                        continue

                    symbol = position.symbol
                    current_value = float(position.market_value)
                    shock = 0.0

                    # Apply specific symbol shocks
                    scenario_shocks = cast(Dict[str, Any], scenario.get("shocks", {}))
                    if "shocks" in scenario and symbol in scenario_shocks:
                        shock = scenario_shocks[symbol]

                    # Apply general market shock
                    if "general_shock" in scenario:
                        shock += cast(float, scenario["general_shock"])

                    # Apply volatility shock
                    if "volatility_multiplier" in scenario:
                        # Simulate higher volatility impact
                        daily_vol = 0.02  # Assume 2% daily vol
                        vol_shock = (
                            daily_vol
                            * (scenario["volatility_multiplier"] - 1)
                            * np.random.normal(0, 1)
                        )
                        shock += vol_shock

                    # Calculate position P&L
                    position_pnl = current_value * shock
                    scenario_pnl += position_pnl

                    position_impacts[symbol] = {
                        "shock_applied": shock,
                        "pnl": position_pnl,
                        "pnl_percent": (
                            (position_pnl / current_value * 100)
                            if current_value != 0
                            else 0
                        ),
                    }

                # Calculate scenario metrics
                scenario_return = (
                    (scenario_pnl / base_portfolio_value)
                    if base_portfolio_value > 0
                    else 0
                )

                results[scenario_name] = {
                    "description": scenario.get("description", ""),
                    "total_pnl": scenario_pnl,
                    "portfolio_return": scenario_return * 100,  # Percentage
                    "position_impacts": position_impacts,
                    "worst_position": (
                        max(
                            position_impacts.items(),
                            key=lambda x: abs(x[1]["pnl"]),
                            default=("", {}),
                        )[0]
                        if position_impacts
                        else ""
                    ),
                    "positions_at_risk": len(
                        [p for p in position_impacts.values() if p["pnl"] < -1000]
                    ),  # Positions losing >$1000
                }

            # Summary statistics
            scenario_returns = [results[s]["portfolio_return"] for s in results]
            results["summary"] = {
                "worst_case_return": min(scenario_returns) if scenario_returns else 0,
                "average_stress_return": (
                    np.mean(scenario_returns) if scenario_returns else 0
                ),
                "stress_scenarios_count": len(stress_scenarios),
            }

            return results

        except Exception as e:
            logger.error(f"Error in enhanced stress testing: {e}")
            return {"error": str(e)}

    async def calculate_risk_attribution(self, portfolio: PortfolioState) -> Dict:
        """
        Calculate risk attribution by various factors.

        Returns:
            Dictionary with risk attribution breakdown
        """
        # TODO: Add comprehensive tests for risk attribution
        try:
            attribution: Dict[str, Any] = {
                "factor_contributions": {},
                "position_contributions": {},
                "sector_risk": {},
                "total_portfolio_risk": 0.0,
            }

            # Calculate portfolio volatility
            portfolio_vol = await self.calculate_portfolio_volatility(portfolio)
            attribution["total_portfolio_risk"] = portfolio_vol

            if portfolio_vol == 0:
                return attribution

            # Get position weights and characteristics
            total_value = float(portfolio.total_equity)

            for position in portfolio.positions:
                if position.quantity == 0:
                    continue

                symbol = position.symbol
                weight = float(abs(position.market_value)) / total_value
                position_vol = await self._calculate_symbol_volatility(symbol)

                # Position risk contribution
                position_risk = weight * position_vol
                position_contributions_dict = cast(
                    Dict[str, Any], attribution["position_contributions"]
                )
                position_contributions_dict[symbol] = {
                    "weight": weight,
                    "volatility": position_vol,
                    "risk_contribution": position_risk,
                    "risk_percentage": (
                        (position_risk / portfolio_vol * 100)
                        if portfolio_vol > 0
                        else 0
                    ),
                }

                # Get sector for risk attribution
                characteristics = await self._get_position_characteristics(symbol)
                sector = characteristics.get("sector", "Unknown")

                sector_risk_dict = cast(Dict[str, Any], attribution["sector_risk"])
                if sector not in sector_risk_dict:
                    sector_risk_dict[sector] = {
                        "total_weight": 0.0,
                        "risk_contribution": 0.0,
                        "positions": [],
                    }

                sector_risk_dict[sector]["total_weight"] += weight
                sector_risk_dict[sector]["risk_contribution"] += position_risk
                sector_positions_list = cast(
                    List[Any], sector_risk_dict[sector]["positions"]
                )
                sector_positions_list.append(symbol)

            # Calculate factor contributions (simplified)
            attribution["factor_contributions"] = {
                "market_beta": await self.calculate_portfolio_beta(portfolio),
                "concentration_risk": max(
                    [
                        data["weight"]
                        for data in cast(
                            Dict[str, Any], attribution["position_contributions"]
                        ).values()
                    ],
                    default=0,
                ),
                "sector_concentration": max(
                    [
                        sector_data["total_weight"]
                        for sector_data in cast(
                            Dict[str, Any], attribution["sector_risk"]
                        ).values()
                    ],
                    default=0,
                ),
            }

            return attribution

        except Exception as e:
            logger.error(f"Error calculating risk attribution: {e}")
            return {"error": str(e)}

    async def calculate_concentration_metrics(self, portfolio: PortfolioState) -> Dict:
        """Calculate various concentration risk metrics."""
        try:
            if not portfolio.positions or portfolio.total_equity <= 0:
                return {
                    "herfindahl_index": 0.0,
                    "effective_positions": 0,
                    "largest_position_weight": 0.0,
                    "top_5_concentration": 0.0,
                }

            # Calculate position weights
            weights: List[float] = []
            position_values: List[Tuple[str, float]] = []

            if isinstance(portfolio.positions, dict):
                for position in portfolio.positions.values():
                    if position.quantity != 0:
                        weight = abs(position.market_value) / portfolio.total_equity
                        weights.append(float(weight))
                        position_values.append((position.symbol, float(weight)))

            if not weights:
                return {
                    "herfindahl_index": 0.0,
                    "effective_positions": 0,
                    "largest_position_weight": 0.0,
                    "top_5_concentration": 0.0,
                }

            # Herfindahl-Hirschman Index
            hhi = sum(w**2 for w in weights)

            # Effective number of positions
            effective_positions = 1.0 / hhi if hhi > 0 else 0.0

            # Largest position weight
            largest_weight = max(weights)

            # Top 5 concentration
            sorted_weights = sorted(weights, reverse=True)
            top_5_concentration = sum(sorted_weights[:5])

            return {
                "herfindahl_index": hhi,
                "effective_positions": effective_positions,
                "largest_position_weight": largest_weight,
                "top_5_concentration": top_5_concentration,
                "position_weights": dict(position_values),
            }

        except Exception as e:
            logger.error(f"Error calculating concentration metrics: {e}")
            return {
                "herfindahl_index": 0.0,
                "effective_positions": 0,
                "largest_position_weight": 0.0,
                "top_5_concentration": 0.0,
            }

    async def calculate_factor_exposures(self, portfolio: PortfolioState) -> Dict:
        """Calculate factor exposures (sector, style, etc.)."""
        try:
            exposures = {
                "sectors": {},
                "market_cap": {"large": 0.0, "mid": 0.0, "small": 0.0},
                "style": {"growth": 0.0, "value": 0.0},
                "geography": {"domestic": 0.0, "international": 0.0},
            }

            if portfolio.total_equity <= 0:
                return exposures

            for position in portfolio.positions:
                if position.quantity == 0:
                    continue

                weight = float(abs(position.market_value) / portfolio.total_equity)

                # Get position characteristics (placeholder implementation)
                characteristics = await self._get_position_characteristics(
                    position.symbol
                )

                # Sector exposure
                sector = characteristics.get("sector", "Unknown")
                exposures["sectors"][sector] = (
                    exposures["sectors"].get(sector, 0.0) + weight
                )

                # Market cap exposure
                market_cap = characteristics.get("market_cap", "large")
                if market_cap in exposures["market_cap"]:
                    exposures["market_cap"][market_cap] += weight

                # Style exposure (simplified)
                style = characteristics.get("style", "value")
                if style in exposures["style"]:
                    exposures["style"][style] += weight

                # Geography (simplified)
                geography = characteristics.get("geography", "domestic")
                if geography in exposures["geography"]:
                    exposures["geography"][geography] += weight

            return exposures

        except Exception as e:
            logger.error(f"Error calculating factor exposures: {e}")
            return {
                "sectors": {},
                "market_cap": {"large": 0.0, "mid": 0.0, "small": 0.0},
                "style": {"growth": 0.0, "value": 0.0},
                "geography": {"domestic": 0.0, "international": 0.0},
            }

    async def calculate_tracking_error(
        self, portfolio: PortfolioState, benchmark: str = "SPY"
    ) -> float:
        """Calculate tracking error relative to benchmark."""
        try:
            # Get portfolio returns
            portfolio_returns = await self._calculate_portfolio_returns(portfolio)

            # Get benchmark returns
            benchmark_returns = await self._get_symbol_returns(benchmark)

            if len(portfolio_returns) < 10 or len(benchmark_returns) < 10:
                return 0.0

            # Simple alignment - use shorter series length
            min_length = min(len(portfolio_returns), len(benchmark_returns))
            portfolio_vals = portfolio_returns[:min_length]
            benchmark_vals = benchmark_returns[:min_length]

            if min_length < 10:
                return 0.0

            # Calculate excess returns
            excess_returns = [p - b for p, b in zip(portfolio_vals, benchmark_vals)]

            # Tracking error is standard deviation of excess returns
            if len(excess_returns) > 1:
                tracking_error = statistics.stdev(excess_returns) * math.sqrt(
                    252
                )  # Annualized
            else:
                tracking_error = 0.0

            return float(tracking_error)

        except Exception as e:
            logger.error(f"Error calculating tracking error: {e}")
            return 0.0

    async def _historical_var(
        self,
        portfolio: PortfolioState,
        market_data: Dict,
        confidence_level: float,
        holding_period: int,
    ) -> Tuple[Decimal, Decimal]:
        """Calculate VaR using historical simulation method."""
        try:
            # Get portfolio returns from historical simulation
            portfolio_returns = await self._simulate_portfolio_returns(
                portfolio, market_data
            )

            if len(portfolio_returns) < 30:
                return await self._fallback_var_calculation(portfolio, confidence_level)

            # Sort returns
            sorted_returns = sorted(portfolio_returns)

            # Calculate VaR (negative percentile)
            var_index = int((1 - confidence_level) * len(sorted_returns))
            var_return = sorted_returns[var_index]

            # Scale for holding period
            var_scaled = var_return * np.sqrt(holding_period)
            var_dollar = portfolio.total_equity * Decimal(str(abs(var_scaled)))

            # Calculate Expected Shortfall (average of tail losses)
            tail_returns = sorted_returns[:var_index]
            if tail_returns:
                es_return = np.mean(tail_returns) * np.sqrt(holding_period)
                es_dollar = portfolio.total_equity * Decimal(str(abs(es_return)))
            else:
                es_dollar = var_dollar * Decimal("1.3")

            return var_dollar, es_dollar

        except Exception as e:
            logger.error(f"Error in historical VaR calculation: {e}")
            return await self._fallback_var_calculation(portfolio, confidence_level)

    async def _parametric_var(
        self,
        portfolio: PortfolioState,
        market_data: Dict,
        confidence_level: float,
        holding_period: int,
    ) -> Tuple[Decimal, Decimal]:
        """Calculate VaR using parametric (normal distribution) method."""
        try:
            # Calculate portfolio volatility
            portfolio_vol = await self.calculate_portfolio_volatility(portfolio)

            # Get z-score for confidence level
            z_score = stats.norm.ppf(1 - confidence_level)

            # Calculate daily VaR
            daily_var = (
                portfolio.total_equity
                * Decimal(str(portfolio_vol / np.sqrt(252)))
                * Decimal(str(abs(z_score)))
            )

            # Scale for holding period
            var_scaled = daily_var * Decimal(str(np.sqrt(holding_period)))

            # Expected Shortfall for normal distribution
            es_multiplier = stats.norm.pdf(z_score) / (1 - confidence_level)
            es_scaled = var_scaled * Decimal(str(es_multiplier))

            return var_scaled, es_scaled

        except Exception as e:
            logger.error(f"Error in parametric VaR calculation: {e}")
            return await self._fallback_var_calculation(portfolio, confidence_level)

    async def _monte_carlo_var(
        self,
        portfolio: PortfolioState,
        market_data: Dict,
        confidence_level: float,
        holding_period: int,
        num_simulations: int = 10000,
    ) -> Tuple[Decimal, Decimal]:
        """Calculate VaR using Monte Carlo simulation."""
        try:
            symbols = [p.symbol for p in portfolio.positions if p.quantity != 0]

            if not symbols:
                return Decimal("0"), Decimal("0")

            # Get correlation matrix and volatilities
            correlation_matrix = await self.calculate_correlation_matrix(symbols)
            volatilities = {}

            for symbol in symbols:
                vol = await self._calculate_symbol_volatility(symbol)
                volatilities[symbol] = vol

            # Get position weights
            weights = await self._calculate_position_weights(portfolio)

            # Monte Carlo simulation
            portfolio_returns = []

            for _ in range(num_simulations):
                # Generate correlated random returns
                random_returns = await self._generate_correlated_returns(
                    symbols, correlation_matrix, volatilities
                )

                # Calculate portfolio return
                portfolio_return = 0.0
                for symbol in symbols:
                    weight = weights.get(symbol, 0.0)
                    symbol_return = random_returns.get(symbol, 0.0)
                    portfolio_return += weight * symbol_return

                # Scale for holding period
                scaled_return = portfolio_return * np.sqrt(holding_period)
                portfolio_returns.append(scaled_return)

            # Calculate VaR and ES
            sorted_returns = sorted(portfolio_returns)
            var_index = int((1 - confidence_level) * len(sorted_returns))

            var_return = sorted_returns[var_index]
            var_dollar = portfolio.total_equity * Decimal(str(abs(var_return)))

            # Expected Shortfall
            tail_returns = sorted_returns[:var_index]
            if tail_returns:
                es_return = np.mean(tail_returns)
                es_dollar = portfolio.total_equity * Decimal(str(abs(es_return)))
            else:
                es_dollar = var_dollar * Decimal("1.3")

            logger.debug(
                f"Monte Carlo VaR completed with {num_simulations} simulations"
            )
            return var_dollar, es_dollar

        except Exception as e:
            logger.error(f"Error in Monte Carlo VaR calculation: {e}")
            return await self._fallback_var_calculation(portfolio, confidence_level)

    async def calculate_component_var(
        self, portfolio: PortfolioState
    ) -> Dict[str, Decimal]:
        """Calculate component VaR for each position."""
        try:
            # Calculate total portfolio VaR
            total_var, _ = await self.calculate_portfolio_var(portfolio)

            if total_var <= 0:
                return {}

            component_vars = {}

            # For each position, calculate marginal VaR
            for position in portfolio.positions:
                if position.quantity == 0:
                    continue

                # Calculate position's contribution to portfolio volatility
                contribution = await self._calculate_position_volatility_contribution(
                    position, portfolio
                )

                # Component VaR = marginal VaR * position weight * total VaR
                position_weight = (
                    abs(position.market_value) / portfolio.total_equity
                    if portfolio.total_equity > 0
                    else Decimal("0")
                )
                component_var = total_var * position_weight * Decimal(str(contribution))

                component_vars[position.symbol] = component_var

            return component_vars

        except Exception as e:
            logger.error(f"Error calculating component VaR: {e}")
            return {}

    async def calculate_maximum_drawdown(
        self, returns: List[float]
    ) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown from return series.

        Args:
            returns: List of returns

        Returns:
            Tuple of (max_drawdown, start_index, end_index)
        """
        try:
            if len(returns) < 2:
                return 0.0, 0, 0

            # Convert returns to cumulative wealth
            cumulative_wealth = [1.0]
            for ret in returns:
                cumulative_wealth.append(cumulative_wealth[-1] * (1 + ret))

            # Find maximum drawdown
            max_drawdown = 0.0
            peak_index = 0
            trough_index = 0
            current_peak = cumulative_wealth[0]
            current_peak_index = 0

            for i in range(1, len(cumulative_wealth)):
                if cumulative_wealth[i] > current_peak:
                    current_peak = cumulative_wealth[i]
                    current_peak_index = i
                else:
                    drawdown = (current_peak - cumulative_wealth[i]) / current_peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                        peak_index = current_peak_index
                        trough_index = i

            return max_drawdown, peak_index, trough_index

        except Exception as e:
            logger.error(f"Error calculating maximum drawdown: {e}")
            return 0.0, 0, 0

    async def _simulate_portfolio_returns(
        self, portfolio: PortfolioState, market_data: Dict
    ) -> List[float]:
        """Simulate portfolio returns using historical data."""
        try:
            # Get historical returns for each position
            all_returns = {}

            for position in portfolio.positions:
                if position.quantity == 0:
                    continue

                symbol = position.symbol
                symbol_data = market_data.get(symbol, {})
                returns = symbol_data.get("returns", [])

                if returns:
                    all_returns[symbol] = returns

            if not all_returns:
                return []

            # Find common dates
            min_length = min(len(returns) for returns in all_returns.values())

            if min_length < 10:
                return []

            # Calculate portfolio returns for each day
            weights = await self._calculate_position_weights(portfolio)
            portfolio_returns = []

            for i in range(min_length):
                daily_return = 0.0

                for symbol, returns in all_returns.items():
                    weight = weights.get(symbol, 0.0)
                    symbol_return = returns[-(min_length - i)]  # Get from end
                    daily_return += weight * symbol_return

                portfolio_returns.append(daily_return)

            return portfolio_returns

        except Exception as e:
            logger.error(f"Error simulating portfolio returns: {e}")
            return []

    async def _generate_correlated_returns(
        self,
        symbols: List[str],
        correlation_matrix: pd.DataFrame,
        volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        """Generate correlated random returns for Monte Carlo simulation."""
        try:
            # Generate independent random variables
            independent_randoms = np.random.standard_normal(len(symbols))

            # Apply Cholesky decomposition for correlation
            try:
                chol_matrix = np.linalg.cholesky(correlation_matrix.values)
                correlated_randoms = np.dot(chol_matrix, independent_randoms)
            except np.linalg.LinAlgError:
                # Use eigenvalue decomposition as fallback
                eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix.values)
                eigenvals = np.maximum(eigenvals, 0.0001)  # Ensure positive
                sqrt_eigenvals = np.sqrt(eigenvals)
                corr_transform = eigenvecs @ np.diag(sqrt_eigenvals)
                correlated_randoms = np.dot(corr_transform, independent_randoms)

            # Scale by volatilities and convert to daily returns
            returns = {}
            for i, symbol in enumerate(symbols):
                vol = volatilities.get(symbol, 0.2)
                daily_vol = vol / np.sqrt(252)  # Convert annual to daily
                returns[symbol] = correlated_randoms[i] * daily_vol

            return returns

        except Exception as e:
            logger.error(f"Error generating correlated returns: {e}")
            # Return uncorrelated returns as fallback
            returns = {}
            for symbol in symbols:
                vol = volatilities.get(symbol, 0.2)
                daily_vol = vol / np.sqrt(252)
                returns[symbol] = np.random.normal(0, daily_vol)
            return returns

    async def _calculate_position_weights(
        self, portfolio: PortfolioState
    ) -> Dict[str, float]:
        """Calculate position weights in portfolio."""
        weights: Dict[str, float] = {}

        if portfolio.total_equity <= 0:
            return weights

        for position in portfolio.positions:
            if position.quantity != 0:
                weight = float(abs(position.market_value) / portfolio.total_equity)
                weights[position.symbol] = weight

        return weights

    async def _calculate_symbol_volatility(self, symbol: str, days: int = 60) -> float:
        """Calculate symbol volatility from historical data."""
        try:
            if self.alpaca_client:
                # Get historical prices
                prices = await self.alpaca_client.get_historical_prices(symbol, days)

                if len(prices) < 10:
                    return self._get_default_volatility(symbol)

                # Calculate returns
                returns = []
                for i in range(1, len(prices)):
                    prev_price = prices[i - 1][1]
                    curr_price = prices[i][1]

                    if prev_price > 0:
                        daily_return = float((curr_price - prev_price) / prev_price)
                        returns.append(daily_return)

                if len(returns) < 10:
                    return self._get_default_volatility(symbol)

                # Calculate annualized volatility
                volatility = np.std(returns) * np.sqrt(252)
                return float(volatility)
            else:
                return self._get_default_volatility(symbol)

        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return self._get_default_volatility(symbol)

    async def _calculate_symbol_beta(
        self, symbol: str, benchmark: str = "SPY", days: int = 60
    ) -> float:
        """Calculate symbol beta relative to benchmark."""
        try:
            if not self.alpaca_client:
                return 1.0

            # Get historical data for both symbol and benchmark
            symbol_prices = await self.alpaca_client.get_historical_prices(symbol, days)
            benchmark_prices = await self.alpaca_client.get_historical_prices(
                benchmark, days
            )

            if len(symbol_prices) < 10 or len(benchmark_prices) < 10:
                return 1.0

            # Calculate returns
            symbol_returns = self._calculate_returns_from_prices(symbol_prices)
            benchmark_returns = self._calculate_returns_from_prices(benchmark_prices)

            # Align returns
            aligned = self._align_return_series(symbol_returns, benchmark_returns)

            if len(aligned) < 10:
                return 1.0

            # Calculate beta using linear regression
            symbol_rets = [sym_ret for _, sym_ret, _ in aligned]
            benchmark_rets = [bench_ret for _, _, bench_ret in aligned]

            covariance = np.cov(symbol_rets, benchmark_rets)[0, 1]
            benchmark_variance = np.var(benchmark_rets)

            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
                return float(beta)
            else:
                return 1.0

        except Exception as e:
            logger.error(f"Error calculating beta for {symbol}: {e}")
            return 1.0

    def _calculate_returns_from_prices(
        self, prices: List[Tuple[datetime, Decimal]]
    ) -> List[Tuple[datetime, float]]:
        """Calculate returns from price series."""
        returns = []

        for i in range(1, len(prices)):
            date = prices[i][0]
            prev_price = prices[i - 1][1]
            curr_price = prices[i][1]

            if prev_price > 0:
                daily_return = float((curr_price - prev_price) / prev_price)
                returns.append((date, daily_return))

        return returns

    def _align_return_series(
        self,
        returns1: List[Tuple[datetime, float]],
        returns2: List[Tuple[datetime, float]],
    ) -> List[Tuple[datetime, float, float]]:
        """Align two return series by date."""

        dict1 = {dt.date(): ret for dt, ret in returns1}
        dict2 = {dt.date(): ret for dt, ret in returns2}

        common_dates = set(dict1.keys()) & set(dict2.keys())
        sorted_dates = list(sorted(common_dates))

        aligned = []
        for date_obj in sorted_dates:
            aligned.append(
                (
                    datetime(date_obj.year, date_obj.month, date_obj.day),
                    dict1[date_obj],
                    dict2[date_obj],
                )
            )

        return aligned

    async def _get_portfolio_market_data(self, symbols: List[str]) -> Dict:
        """Get comprehensive market data for portfolio symbols."""
        try:
            market_data = {}

            if self.alpaca_client:
                # Use Alpaca client to get data
                data = await self.alpaca_client.get_market_data_for_risk_calc(symbols)
                return data
            else:
                # Fallback to simulated data
                for symbol in symbols:
                    market_data[symbol] = {
                        "volatility": self._get_default_volatility(symbol),
                        "returns": [
                            np.random.normal(0, 0.01) for _ in range(60)
                        ],  # 60 days of fake returns
                        "current_price": Decimal("100"),  # Default price
                        "data_points": 60,
                    }

                return market_data

        except Exception as e:
            logger.error(f"Error getting portfolio market data: {e}")
            return {}

    async def _get_symbol_returns(self, symbol: str, days: int = 60) -> List[float]:
        """Get historical returns for a symbol."""
        try:
            if self.alpaca_client:
                prices = await self.alpaca_client.get_historical_prices(symbol, days)
                if len(prices) < 2:
                    return []

                returns = []
                for i in range(1, len(prices)):
                    prev_price = prices[i - 1][1]
                    curr_price = prices[i][1]

                    if prev_price > 0:
                        daily_return = float((curr_price - prev_price) / prev_price)
                        returns.append(daily_return)

                return returns
            else:
                # Generate simulated returns
                return [np.random.normal(0, 0.015) for _ in range(days)]

        except Exception as e:
            logger.error(f"Error getting returns for {symbol}: {e}")
            return []

    async def _calculate_portfolio_returns(
        self, portfolio: PortfolioState, days: int = 60
    ) -> List[float]:
        """Calculate historical portfolio returns."""
        try:
            if not self.alpaca_client:
                return []

            # Get portfolio history
            history = await self.alpaca_client.get_historical_portfolio_values(days)

            if len(history) < 2:
                return []

            returns = []
            for i in range(1, len(history)):
                prev_value = history[i - 1][1]
                curr_value = history[i][1]

                if prev_value > 0:
                    daily_return = float((curr_value - prev_value) / prev_value)
                    returns.append(daily_return)

            return returns

        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {e}")
            return []

    async def _calculate_position_volatility_contribution(
        self, position: Position, portfolio: PortfolioState
    ) -> float:
        """Calculate a position's contribution to portfolio volatility."""
        try:
            # This is a simplified contribution calculation
            # In production, you'd use the full covariance matrix and marginal contribution formula

            position_weight = (
                float(abs(position.market_value) / portfolio.total_equity)
                if portfolio.total_equity > 0
                else 0.0
            )
            position_vol = await self._calculate_symbol_volatility(position.symbol)

            # Simplified contribution (actual formula is more complex)
            contribution = position_weight * position_vol

            return contribution

        except Exception as e:
            logger.error(
                f"Error calculating volatility contribution for {position.symbol}: {e}"
            )
            return 0.0

    async def _get_position_characteristics(self, symbol: str) -> Dict:
        """Get position characteristics for factor analysis."""
        # Enhanced implementation with more comprehensive data
        # TODO: Add comprehensive tests for position characteristics lookup
        characteristics_map = {
            "AAPL": {
                "sector": "Technology",
                "market_cap": "large",
                "style": "growth",
                "geography": "domestic",
                "beta": 1.2,
                "dividend_yield": 0.005,
            },
            "MSFT": {
                "sector": "Technology",
                "market_cap": "large",
                "style": "growth",
                "geography": "domestic",
                "beta": 0.9,
                "dividend_yield": 0.007,
            },
            "GOOGL": {
                "sector": "Technology",
                "market_cap": "large",
                "style": "growth",
                "geography": "domestic",
                "beta": 1.1,
                "dividend_yield": 0.0,
            },
            "AMZN": {
                "sector": "Consumer Discretionary",
                "market_cap": "large",
                "style": "growth",
                "geography": "domestic",
                "beta": 1.3,
                "dividend_yield": 0.0,
            },
            "TSLA": {
                "sector": "Consumer Discretionary",
                "market_cap": "large",
                "style": "growth",
                "geography": "domestic",
                "beta": 2.0,
                "dividend_yield": 0.0,
            },
            "JPM": {
                "sector": "Financials",
                "market_cap": "large",
                "style": "value",
                "geography": "domestic",
                "beta": 1.1,
                "dividend_yield": 0.025,
            },
            "BAC": {
                "sector": "Financials",
                "market_cap": "large",
                "style": "value",
                "geography": "domestic",
                "beta": 1.2,
                "dividend_yield": 0.021,
            },
            "WFC": {
                "sector": "Financials",
                "market_cap": "large",
                "style": "value",
                "geography": "domestic",
                "beta": 1.3,
                "dividend_yield": 0.027,
            },
            "GS": {
                "sector": "Financials",
                "market_cap": "large",
                "style": "value",
                "geography": "domestic",
                "beta": 1.4,
                "dividend_yield": 0.024,
            },
            "C": {
                "sector": "Financials",
                "market_cap": "large",
                "style": "value",
                "geography": "domestic",
                "beta": 1.5,
                "dividend_yield": 0.036,
            },
            "SPY": {
                "sector": "ETF",
                "market_cap": "large",
                "style": "blend",
                "geography": "domestic",
                "beta": 1.0,
                "dividend_yield": 0.013,
            },
            "QQQ": {
                "sector": "ETF",
                "market_cap": "large",
                "style": "growth",
                "geography": "domestic",
                "beta": 1.2,
                "dividend_yield": 0.007,
            },
            "IWM": {
                "sector": "ETF",
                "market_cap": "small",
                "style": "blend",
                "geography": "domestic",
                "beta": 1.1,
                "dividend_yield": 0.017,
            },
        }

        return characteristics_map.get(
            symbol,
            {
                "sector": "Unknown",
                "market_cap": "mid",
                "style": "blend",
                "geography": "domestic",
                "beta": 1.0,
                "dividend_yield": 0.02,
            },
        )

    def _get_default_volatility(self, symbol: str) -> float:
        """Get default volatility estimate based on symbol characteristics."""
        if symbol.startswith(("SPY", "QQQ", "IWM")):
            return 0.15  # ETFs - lower volatility
        elif len(symbol) <= 3:
            return 0.20  # Large cap stocks
        elif len(symbol) == 4:
            return 0.25  # Mid cap stocks
        else:
            return 0.35  # Small cap or specialized stocks

    async def _fallback_var_calculation(
        self, portfolio: PortfolioState, confidence_level: float
    ) -> Tuple[Decimal, Decimal]:
        """Fallback VaR calculation when market data is unavailable."""
        try:
            # Use simplified parametric approach with default assumptions
            portfolio_value = portfolio.total_equity
            default_volatility = 0.15  # 15% annual volatility

            # Get z-score for confidence level
            z_score = stats.norm.ppf(1 - confidence_level)

            # Calculate daily VaR
            daily_vol = default_volatility / np.sqrt(252)
            var_amount = (
                portfolio_value * Decimal(str(daily_vol)) * Decimal(str(abs(z_score)))
            )

            # Expected Shortfall
            es_multiplier = stats.norm.pdf(z_score) / (1 - confidence_level)
            es_amount = var_amount * Decimal(str(es_multiplier))

            return var_amount, es_amount

        except Exception as e:
            logger.error(f"Error in fallback VaR calculation: {e}")
            return Decimal("0"), Decimal("0")

    async def generate_comprehensive_risk_report(
        self, portfolio: PortfolioState, benchmark: str = "SPY"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive risk assessment report combining all available metrics.

        Args:
            portfolio: Current portfolio state
            benchmark: Benchmark symbol for relative risk calculations

        Returns:
            Dictionary containing comprehensive risk analysis
        """
        # TODO: Add comprehensive tests for risk report generation
        try:
            report: Dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "portfolio_summary": {
                    "total_value": float(portfolio.total_equity),
                    "num_positions": len(
                        [p for p in portfolio.positions if p.quantity != 0]
                    ),
                    "cash_balance": float(getattr(portfolio, "cash_balance", 0.0)),
                },
                "risk_metrics": {},
                "performance_metrics": {},
                "stress_tests": {},
                "warnings": [],
                "recommendations": [],
            }

            # Core risk metrics
            var_95, es_95 = await self.calculate_portfolio_var(
                portfolio, confidence_level=0.95
            )
            var_99, es_99 = await self.calculate_portfolio_var(
                portfolio, confidence_level=0.99
            )

            report["risk_metrics"]["var_analysis"] = {
                "var_95_percent": float(var_95),
                "expected_shortfall_95": float(es_95),
                "var_99_percent": float(var_99),
                "expected_shortfall_99": float(es_99),
                "var_as_percent_of_portfolio": {
                    "95_percent": (
                        float(var_95 / portfolio.total_equity * 100)
                        if portfolio.total_equity > 0
                        else 0
                    ),
                    "99_percent": (
                        float(var_99 / portfolio.total_equity * 100)
                        if portfolio.total_equity > 0
                        else 0
                    ),
                },
            }

            # Portfolio volatility and beta
            portfolio_vol = await self.calculate_portfolio_volatility(portfolio)
            portfolio_beta = await self.calculate_portfolio_beta(portfolio, benchmark)
            tracking_error = await self.calculate_tracking_error(portfolio, benchmark)

            report["risk_metrics"]["portfolio_analytics"] = {
                "volatility": portfolio_vol,
                "beta": portfolio_beta,
                "tracking_error": tracking_error,
            }

            # Concentration and factor analysis
            concentration_metrics = await self.calculate_concentration_metrics(
                portfolio
            )
            factor_exposures = await self.calculate_factor_exposures(portfolio)

            report["risk_metrics"]["concentration"] = concentration_metrics
            report["risk_metrics"]["factor_exposures"] = factor_exposures

            # Liquidity risk
            liquidity_risk = await self.calculate_liquidity_risk(portfolio)
            report["risk_metrics"]["liquidity"] = liquidity_risk

            # Risk attribution
            risk_attribution = await self.calculate_risk_attribution(portfolio)
            report["risk_metrics"]["attribution"] = risk_attribution

            # Performance metrics
            risk_adjusted_returns = await self.calculate_risk_adjusted_returns(
                portfolio
            )
            report["performance_metrics"] = risk_adjusted_returns

            # Stress testing
            enhanced_stress = await self.enhanced_stress_test(portfolio)
            report["stress_tests"] = enhanced_stress

            # Component VaR
            component_var = await self.calculate_component_var(portfolio)
            report["risk_metrics"]["component_var"] = {
                k: float(v) for k, v in component_var.items()
            }

            # Generate warnings and recommendations
            self._generate_risk_warnings_and_recommendations(report)

            return report

        except Exception as e:
            logger.error(f"Error generating comprehensive risk report: {e}")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "status": "failed",
            }

    def _generate_risk_warnings_and_recommendations(self, report: Dict) -> None:
        """Generate risk warnings and recommendations based on calculated metrics."""
        # TODO: Add comprehensive tests for warning and recommendation generation
        warnings = []
        recommendations = []

        try:
            # VaR warnings
            var_95_percent = (
                report.get("risk_metrics", {})
                .get("var_analysis", {})
                .get("var_as_percent_of_portfolio", {})
                .get("95_percent", 0)
            )
            if var_95_percent > 10:
                warnings.append(
                    f"High VaR exposure: 95% VaR represents {var_95_percent:.1f}% of portfolio value"
                )
                recommendations.append(
                    "Consider reducing position sizes or diversifying to lower VaR"
                )

            # Concentration warnings
            concentration = report.get("risk_metrics", {}).get("concentration", {})
            max_position_weight = concentration.get("max_position_weight", 0)
            if max_position_weight > 0.2:
                warnings.append(
                    f"High concentration risk: largest position is {max_position_weight * 100:.1f}% of portfolio"
                )
                recommendations.append(
                    "Consider reducing largest positions to improve diversification"
                )

            # Liquidity warnings
            liquidity = report.get("risk_metrics", {}).get("liquidity", {})
            portfolio_liquidity_score = liquidity.get("portfolio_liquidity_score", 1.0)
            if portfolio_liquidity_score < 0.5:
                warnings.append("Low portfolio liquidity detected")
                recommendations.append(
                    "Consider increasing allocation to more liquid securities"
                )

            illiquid_positions = liquidity.get("illiquid_positions", [])
            if len(illiquid_positions) > 0:
                warnings.append(
                    f"{len(illiquid_positions)} positions flagged as illiquid"
                )

            # Volatility warnings
            portfolio_vol = (
                report.get("risk_metrics", {})
                .get("portfolio_analytics", {})
                .get("volatility", 0)
            )
            if portfolio_vol > 0.25:
                warnings.append(
                    f"High portfolio volatility: {portfolio_vol * 100:.1f}% annualized"
                )
                recommendations.append(
                    "Consider adding defensive positions or reducing overall leverage"
                )

            # Beta warnings
            portfolio_beta = (
                report.get("risk_metrics", {})
                .get("portfolio_analytics", {})
                .get("beta", 1.0)
            )
            if portfolio_beta > 1.5:
                warnings.append(
                    f"High market sensitivity: portfolio beta is {portfolio_beta:.2f}"
                )
                recommendations.append(
                    "Consider adding market-neutral or low-beta positions"
                )

            # Stress test warnings
            stress_tests = report.get("stress_tests", {})
            worst_case = stress_tests.get("summary", {}).get("worst_case_return", 0)
            if worst_case < -25:
                warnings.append(
                    f"Severe stress test exposure: worst case scenario shows {worst_case:.1f}% loss"
                )
                recommendations.append(
                    "Consider stress-testing hedge strategies or position sizing adjustments"
                )

            # Factor exposure warnings
            factor_exposures = report.get("risk_metrics", {}).get(
                "factor_exposures", {}
            )
            sectors = factor_exposures.get("sectors", {})
            max_sector_exposure = max(sectors.values()) if sectors else 0
            if max_sector_exposure > 0.4:
                max_sector = (
                    max(sectors.items(), key=lambda x: x[1])[0]
                    if sectors
                    else "Unknown"
                )
                warnings.append(
                    f"High sector concentration: {max_sector_exposure * 100:.1f}% in {max_sector}"
                )
                recommendations.append(
                    "Consider diversifying across sectors to reduce concentration risk"
                )

            warnings_list = cast(List[str], report["warnings"])
            warnings_list.extend(warnings)
            recommendations_list = cast(List[str], report["recommendations"])
            recommendations_list.extend(recommendations)

        except Exception as e:
            logger.error(f"Error generating warnings and recommendations: {e}")
            warnings_list = cast(List[str], report["warnings"])
            warnings_list.clear()
            warnings_list.append("Error generating risk warnings")
            recommendations_list = cast(List[str], report["recommendations"])
            recommendations_list.clear()
            recommendations_list.append("Review risk metrics manually")

    def get_calculation_info(self) -> Dict:
        """Get information about calculation parameters and cache status."""
        return {
            "confidence_levels": self.confidence_levels,
            "var_methods": self.var_methods,
            "lookback_days": self.lookback_days,
            "cache_ttl_hours": self.cache_ttl.total_seconds() / 3600,
            "cached_symbols": list(self.price_data_cache.keys()),
            "correlation_matrix_cached": self.correlation_matrix_cache is not None,
            "cache_timestamp": (
                self.cache_timestamp.isoformat() if self.cache_timestamp else None
            ),
        }

    def clear_cache(self):
        """Clear all cached data."""
        self.price_data_cache.clear()
        self.correlation_matrix_cache = None
        self.cache_timestamp = None
        logger.debug("Risk calculator cache cleared")
