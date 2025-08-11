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
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import math
import statistics
import numpy as np
import pandas as pd
from scipy import stats

from shared.config import get_config
from shared.models import Position, PortfolioState

logger = logging.getLogger(__name__)


class RiskCalculator:
    """Advanced risk metrics calculator using real market data."""

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

    async def calculate_portfolio_var(self,
                                    portfolio: PortfolioState,
                                    confidence_level: float = 0.95,
                                    method: str = "historical",
                                    holding_period: int = 1) -> Tuple[Decimal, Decimal]:
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
                var, es = await self._historical_var(portfolio, market_data, confidence_level, holding_period)
            elif method == "parametric":
                var, es = await self._parametric_var(portfolio, market_data, confidence_level, holding_period)
            elif method == "monte_carlo":
                var, es = await self._monte_carlo_var(portfolio, market_data, confidence_level, holding_period)
            else:
                logger.warning(f"Unknown VaR method: {method}, using historical")
                var, es = await self._historical_var(portfolio, market_data, confidence_level, holding_period)

            logger.debug(f"Portfolio VaR ({method}, {confidence_level:.0%}): ${var}, ES: ${es}")
            return var, es

        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return await self._fallback_var_calculation(portfolio, confidence_level)

    async def calculate_correlation_matrix(self, symbols: List[str], days: int = 60) -> pd.DataFrame:
        """Calculate correlation matrix for given symbols."""
        try:
            # Check cache
            if (self.correlation_matrix_cache is not None and
                self.cache_timestamp and
                datetime.now(timezone.utc) - self.cache_timestamp < self.cache_ttl):
                return self.correlation_matrix_cache.loc[symbols, symbols]

            # Get price data for all symbols
            price_data = {}
            for symbol in symbols:
                prices = await self._get_symbol_returns(symbol, days)
                if prices:
                    price_data[symbol] = prices

            if len(price_data) < 2:
                # Return identity matrix if insufficient data
                return pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)

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
            portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
            portfolio_volatility = np.sqrt(portfolio_variance)

            return float(portfolio_volatility)

        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.15  # Default volatility

    async def calculate_portfolio_beta(self, portfolio: PortfolioState, benchmark: str = "SPY") -> float:
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

    async def stress_test_portfolio(self, portfolio: PortfolioState, scenarios: List[Dict]) -> Dict:
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
                    shocked_price = position.current_price * (Decimal("1") + Decimal(str(shock)))
                    position_pnl = (shocked_price - position.current_price) * Decimal(position.quantity)
                    scenario_pnl += position_pnl

                results[scenario_name] = {
                    "total_pnl": str(scenario_pnl),
                    "pnl_percentage": float(scenario_pnl / portfolio.total_equity) if portfolio.total_equity > 0 else 0.0,
                    "scenario_details": scenario
                }

            return results

        except Exception as e:
            logger.error(f"Error in stress testing: {e}")
            return {}

    async def calculate_concentration_metrics(self, portfolio: PortfolioState) -> Dict:
        """Calculate various concentration risk metrics."""
        try:
            if not portfolio.positions or portfolio.total_equity <= 0:
                return {
                    "herfindahl_index": 0.0,
                    "effective_positions": 0,
                    "largest_position_weight": 0.0,
                    "top_5_concentration": 0.0
                }

            # Calculate position weights
            weights = []
            position_values = []

            for position in portfolio.positions:
                if position.quantity != 0:
                    weight = abs(position.market_value) / portfolio.total_equity
                    weights.append(float(weight))
                    position_values.append((position.symbol, float(weight)))

            if not weights:
                return {
                    "herfindahl_index": 0.0,
                    "effective_positions": 0,
                    "largest_position_weight": 0.0,
                    "top_5_concentration": 0.0
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
                "position_weights": dict(position_values)
            }

        except Exception as e:
            logger.error(f"Error calculating concentration metrics: {e}")
            return {
                "herfindahl_index": 0.0,
                "effective_positions": 0,
                "largest_position_weight": 0.0,
                "top_5_concentration": 0.0
            }

    async def calculate_factor_exposures(self, portfolio: PortfolioState) -> Dict:
        """Calculate factor exposures (sector, style, etc.)."""
        try:
            exposures = {
                "sectors": {},
                "market_cap": {"large": 0.0, "mid": 0.0, "small": 0.0},
                "style": {"growth": 0.0, "value": 0.0},
                "geography": {"domestic": 0.0, "international": 0.0}
            }

            if portfolio.total_equity <= 0:
                return exposures

            for position in portfolio.positions:
                if position.quantity == 0:
                    continue

                weight = float(abs(position.market_value) / portfolio.total_equity)

                # Get position characteristics (placeholder implementation)
                characteristics = await self._get_position_characteristics(position.symbol)

                # Sector exposure
                sector = characteristics.get("sector", "Unknown")
                exposures["sectors"][sector] = exposures["sectors"].get(sector, 0.0) + weight

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
                "geography": {"domestic": 0.0, "international": 0.0}
            }

    async def calculate_tracking_error(self, portfolio: PortfolioState, benchmark: str = "SPY") -> float:
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
                tracking_error = statistics.stdev(excess_returns) * math.sqrt(252)  # Annualized
            else:
                tracking_error = 0.0

            return float(tracking_error)

        except Exception as e:
            logger.error(f"Error calculating tracking error: {e}")
            return 0.0

    async def _historical_var(self,
                            portfolio: PortfolioState,
                            market_data: Dict,
                            confidence_level: float,
                            holding_period: int) -> Tuple[Decimal, Decimal]:
        """Calculate VaR using historical simulation method."""
        try:
            # Get portfolio returns from historical simulation
            portfolio_returns = await self._simulate_portfolio_returns(portfolio, market_data)

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

    async def _parametric_var(self,
                            portfolio: PortfolioState,
                            market_data: Dict,
                            confidence_level: float,
                            holding_period: int) -> Tuple[Decimal, Decimal]:
        """Calculate VaR using parametric (normal distribution) method."""
        try:
            # Calculate portfolio volatility
            portfolio_vol = await self.calculate_portfolio_volatility(portfolio)

            # Get z-score for confidence level
            z_score = stats.norm.ppf(1 - confidence_level)

            # Calculate daily VaR
            daily_var = portfolio.total_equity * Decimal(str(portfolio_vol / np.sqrt(252))) * Decimal(str(abs(z_score)))

            # Scale for holding period
            var_scaled = daily_var * Decimal(str(np.sqrt(holding_period)))

            # Expected Shortfall for normal distribution
            es_multiplier = stats.norm.pdf(z_score) / (1 - confidence_level)
            es_scaled = var_scaled * Decimal(str(es_multiplier))

            return var_scaled, es_scaled

        except Exception as e:
            logger.error(f"Error in parametric VaR calculation: {e}")
            return await self._fallback_var_calculation(portfolio, confidence_level)

    async def _monte_carlo_var(self,
                             portfolio: PortfolioState,
                             market_data: Dict,
                             confidence_level: float,
                             holding_period: int,
                             num_simulations: int = 10000) -> Tuple[Decimal, Decimal]:
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

            logger.debug(f"Monte Carlo VaR completed with {num_simulations} simulations")
            return var_dollar, es_dollar

        except Exception as e:
            logger.error(f"Error in Monte Carlo VaR calculation: {e}")
            return await self._fallback_var_calculation(portfolio, confidence_level)

    async def calculate_component_var(self, portfolio: PortfolioState) -> Dict[str, Decimal]:
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
                contribution = await self._calculate_position_volatility_contribution(position, portfolio)

                # Component VaR = marginal VaR * position weight * total VaR
                position_weight = abs(position.market_value) / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("0")
                component_var = total_var * position_weight * Decimal(str(contribution))

                component_vars[position.symbol] = component_var

            return component_vars

        except Exception as e:
            logger.error(f"Error calculating component VaR: {e}")
            return {}

    async def calculate_maximum_drawdown(self, returns: List[float]) -> Tuple[float, int, int]:
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

    async def _simulate_portfolio_returns(self, portfolio: PortfolioState, market_data: Dict) -> List[float]:
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
                    symbol_return = returns[-(min_length-i)]  # Get from end
                    daily_return += weight * symbol_return

                portfolio_returns.append(daily_return)

            return portfolio_returns

        except Exception as e:
            logger.error(f"Error simulating portfolio returns: {e}")
            return []

    async def _generate_correlated_returns(self,
                                         symbols: List[str],
                                         correlation_matrix: pd.DataFrame,
                                         volatilities: Dict[str, float]) -> Dict[str, float]:
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

    async def _calculate_position_weights(self, portfolio: PortfolioState) -> Dict[str, float]:
        """Calculate position weights in portfolio."""
        weights = {}

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
                    prev_price = prices[i-1][1]
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

    async def _calculate_symbol_beta(self, symbol: str, benchmark: str = "SPY", days: int = 60) -> float:
        """Calculate symbol beta relative to benchmark."""
        try:
            if not self.alpaca_client:
                return 1.0

            # Get historical data for both symbol and benchmark
            symbol_prices = await self.alpaca_client.get_historical_prices(symbol, days)
            benchmark_prices = await self.alpaca_client.get_historical_prices(benchmark, days)

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

    def _calculate_returns_from_prices(self, prices: List[Tuple[datetime, Decimal]]) -> List[Tuple[datetime, float]]:
        """Calculate returns from price series."""
        returns = []

        for i in range(1, len(prices)):
            date = prices[i][0]
            prev_price = prices[i-1][1]
            curr_price = prices[i][1]

            if prev_price > 0:
                daily_return = float((curr_price - prev_price) / prev_price)
                returns.append((date, daily_return))

        return returns

    def _align_return_series(self, returns1: List[Tuple[datetime, float]], returns2: List[Tuple[datetime, float]]) -> List[Tuple[datetime, float, float]]:
        """Align two return series by date."""
        dict1 = {date.date(): ret for date, ret in returns1}
        dict2 = {date.date(): ret for date, ret in returns2}

        common_dates = set(dict1.keys()) & set(dict2.keys())
        common_dates = sorted(common_dates)

        aligned = []
        for date in common_dates:
            aligned.append((date, dict1[date], dict2[date]))

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
                        "returns": [np.random.normal(0, 0.01) for _ in range(60)],  # 60 days of fake returns
                        "current_price": Decimal("100"),  # Default price
                        "data_points": 60
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
                    prev_price = prices[i-1][1]
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

    async def _calculate_portfolio_returns(self, portfolio: PortfolioState, days: int = 60) -> List[float]:
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
                prev_value = history[i-1][1]
                curr_value = history[i][1]

                if prev_value > 0:
                    daily_return = float((curr_value - prev_value) / prev_value)
                    returns.append(daily_return)

            return returns

        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {e}")
            return []

    async def _calculate_position_volatility_contribution(self, position: Position, portfolio: PortfolioState) -> float:
        """Calculate a position's contribution to portfolio volatility."""
        try:
            # This is a simplified contribution calculation
            # In production, you'd use the full covariance matrix and marginal contribution formula

            position_weight = float(abs(position.market_value) / portfolio.total_equity) if portfolio.total_equity > 0 else 0.0
            position_vol = await self._calculate_symbol_volatility(position.symbol)

            # Simplified contribution (actual formula is more complex)
            contribution = position_weight * position_vol

            return contribution

        except Exception as e:
            logger.error(f"Error calculating volatility contribution for {position.symbol}: {e}")
            return 0.0

    async def _get_position_characteristics(self, symbol: str) -> Dict:
        """Get position characteristics for factor analysis."""
        # Placeholder implementation - in production, get from financial data provider
        characteristics_map = {
            'AAPL': {"sector": "Technology", "market_cap": "large", "style": "growth", "geography": "domestic"},
            'MSFT': {"sector": "Technology", "market_cap": "large", "style": "growth", "geography": "domestic"},
            'GOOGL': {"sector": "Technology", "market_cap": "large", "style": "growth", "geography": "domestic"},
            'AMZN': {"sector": "Consumer Discretionary", "market_cap": "large", "style": "growth", "geography": "domestic"},
            'TSLA': {"sector": "Consumer Discretionary", "market_cap": "large", "style": "growth", "geography": "domestic"},
            'JPM': {"sector": "Financials", "market_cap": "large", "style": "value", "geography": "domestic"},
            'BAC': {"sector": "Financials", "market_cap": "large", "style": "value", "geography": "domestic"},
            'SPY': {"sector": "ETF", "market_cap": "large", "style": "value", "geography": "domestic"},
            'QQQ': {"sector": "ETF", "market_cap": "large", "style": "growth", "geography": "domestic"},
            'IWM': {"sector": "ETF", "market_cap": "small", "style": "value", "geography": "domestic"}
        }

        return characteristics_map.get(symbol, {
            "sector": "Unknown",
            "market_cap": "mid",
            "style": "value",
            "geography": "domestic"
        })

    def _get_default_volatility(self, symbol: str) -> float:
        """Get default volatility estimate based on symbol characteristics."""
        if symbol.startswith(('SPY', 'QQQ', 'IWM')):
            return 0.15  # ETFs - lower volatility
        elif len(symbol) <= 3:
            return 0.20  # Large cap stocks
        elif len(symbol) == 4:
            return 0.25  # Mid cap stocks
        else:
            return 0.35  # Small cap or specialized stocks

    async def _fallback_var_calculation(self, portfolio: PortfolioState, confidence_level: float) -> Tuple[Decimal, Decimal]:
        """Fallback VaR calculation when market data is unavailable."""
        try:
            # Use simplified parametric approach with default assumptions
            portfolio_value = portfolio.total_equity
            default_volatility = 0.15  # 15% annual volatility

            # Get z-score for confidence level
            z_score = stats.norm.ppf(1 - confidence_level)

            # Calculate daily VaR
            daily_vol = default_volatility / np.sqrt(252)
            var_amount = portfolio_value * Decimal(str(daily_vol)) * Decimal(str(abs(z_score)))

            # Expected Shortfall
            es_multiplier = stats.norm.pdf(z_score) / (1 - confidence_level)
            es_amount = var_amount * Decimal(str(es_multiplier))

            return var_amount, es_amount

        except Exception as e:
            logger.error(f"Error in fallback VaR calculation: {e}")
            return Decimal("0"), Decimal("0")

    def get_calculation_info(self) -> Dict:
        """Get information about calculation parameters and cache status."""
        return {
            "confidence_levels": self.confidence_levels,
            "var_methods": self.var_methods,
            "lookback_days": self.lookback_days,
            "cache_ttl_hours": self.cache_ttl.total_seconds() / 3600,
            "cached_symbols": list(self.price_data_cache.keys()),
            "correlation_matrix_cached": self.correlation_matrix_cache is not None,
            "cache_timestamp": self.cache_timestamp.isoformat() if self.cache_timestamp else None
        }

    def clear_cache(self):
        """Clear all cached data."""
        self.price_data_cache.clear()
        self.correlation_matrix_cache = None
        self.cache_timestamp = None
        logger.debug("Risk calculator cache cleared")
