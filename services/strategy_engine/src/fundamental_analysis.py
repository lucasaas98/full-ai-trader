"""
Fundamental Analysis Module

This module provides fundamental analysis capabilities using FinViz data
to evaluate stock health, valuation metrics, and sector performance.
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime, timezone
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

import polars as pl
import numpy as np

from .base_strategy import BaseStrategy, StrategyConfig, Signal
from shared.models import SignalType, FinVizData


class ValuationTier(Enum):
    """Stock valuation tiers."""
    UNDERVALUED = "undervalued"
    FAIRLY_VALUED = "fairly_valued"
    OVERVALUED = "overvalued"
    EXTREME_OVERVALUED = "extreme_overvalued"


class FinancialHealth(Enum):
    """Financial health ratings."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    DISTRESSED = "distressed"


@dataclass
class FundamentalMetrics:
    """Container for fundamental analysis metrics."""
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    profit_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    market_cap: Optional[float] = None
    volume_surge_ratio: Optional[float] = None
    sector_performance: Optional[float] = None
    analyst_rating: Optional[str] = None


class FundamentalAnalyzer:
    """Core fundamental analysis engine."""

    def __init__(self):
        """Initialize fundamental analyzer."""
        self.logger = logging.getLogger("fundamental_analyzer")

        # Industry benchmarks (can be updated from external data)
        self.industry_benchmarks = {
            "Technology": {"pe": 25, "pb": 4, "roe": 15, "margin": 20},
            "Healthcare": {"pe": 20, "pb": 3, "roe": 12, "margin": 15},
            "Finance": {"pe": 12, "pb": 1.2, "roe": 10, "margin": 25},
            "Consumer Cyclical": {"pe": 18, "pb": 2.5, "roe": 15, "margin": 8},
            "Consumer Defensive": {"pe": 22, "pb": 3, "roe": 18, "margin": 5},
            "Energy": {"pe": 15, "pb": 1.5, "roe": 8, "margin": 10},
            "Utilities": {"pe": 16, "pb": 1.3, "roe": 9, "margin": 12},
            "Real Estate": {"pe": 20, "pb": 1.8, "roe": 8, "margin": 30},
            "Materials": {"pe": 16, "pb": 1.8, "roe": 12, "margin": 8},
            "Industrials": {"pe": 18, "pb": 2.2, "roe": 12, "margin": 10},
            "Communication Services": {"pe": 20, "pb": 2.8, "roe": 12, "margin": 15}
        }

    def analyze_valuation(self, metrics: FundamentalMetrics, sector: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze stock valuation using multiple metrics.

        Args:
            metrics: Fundamental metrics
            sector: Stock sector for benchmarking

        Returns:
            Valuation analysis results
        """
        try:
            valuation_scores = []
            details = {}

            # Get sector benchmarks
            benchmarks = self.industry_benchmarks.get(sector or "Technology", {})

            # P/E Ratio Analysis
            if metrics.pe_ratio is not None and metrics.pe_ratio > 0:
                pe_score = self._score_pe_ratio(metrics.pe_ratio, benchmarks.get("pe", 20))
                valuation_scores.append(pe_score)
                details["pe_analysis"] = {
                    "value": metrics.pe_ratio,
                    "benchmark": benchmarks.get("pe", 20),
                    "score": pe_score
                }

            # P/B Ratio Analysis
            if metrics.pb_ratio is not None and metrics.pb_ratio > 0:
                pb_score = self._score_pb_ratio(metrics.pb_ratio, benchmarks.get("pb", 3))
                valuation_scores.append(pb_score)
                details["pb_analysis"] = {
                    "value": metrics.pb_ratio,
                    "benchmark": benchmarks.get("pb", 3),
                    "score": pb_score
                }

            # PEG Ratio Analysis
            if metrics.peg_ratio is not None:
                peg_score = self._score_peg_ratio(metrics.peg_ratio)
                valuation_scores.append(peg_score)
                details["peg_analysis"] = {
                    "value": metrics.peg_ratio,
                    "score": peg_score
                }

            # Calculate overall valuation score
            if valuation_scores:
                avg_score = sum(valuation_scores) / len(valuation_scores)
                valuation_tier = self._determine_valuation_tier(avg_score)
            else:
                avg_score = 50.0
                valuation_tier = ValuationTier.FAIRLY_VALUED

            return {
                "valuation_score": avg_score,
                "valuation_tier": valuation_tier,
                "details": details,
                "metrics_available": len(valuation_scores)
            }

        except Exception as e:
            self.logger.error(f"Error in valuation analysis: {e}")
            return {
                "valuation_score": 50.0,
                "valuation_tier": ValuationTier.FAIRLY_VALUED,
                "error": str(e)
            }

    def analyze_growth(self, metrics: FundamentalMetrics, sector: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze financial health and quality metrics.

        Args:
            metrics: Fundamental metrics
            sector: Stock sector for benchmarking

        Returns:
            Financial health analysis
        """
        try:
            health_scores = []
            details = {}
            benchmarks = self.industry_benchmarks.get(sector or "Technology", {})

            # ROE Analysis
            if metrics.roe is not None:
                roe_score = self._score_roe(metrics.roe, benchmarks.get("roe", 12))
                health_scores.append(roe_score)
                details["roe_analysis"] = {
                    "value": metrics.roe,
                    "benchmark": benchmarks.get("roe", 12),
                    "score": roe_score
                }

            # Debt-to-Equity Analysis
            if metrics.debt_to_equity is not None:
                debt_score = self._score_debt_to_equity(metrics.debt_to_equity)
                health_scores.append(debt_score)
                details["debt_analysis"] = {
                    "value": metrics.debt_to_equity,
                    "score": debt_score
                }

            # Current Ratio Analysis
            if metrics.current_ratio is not None:
                liquidity_score = self._score_current_ratio(metrics.current_ratio)
                health_scores.append(liquidity_score)
                details["liquidity_analysis"] = {
                    "value": metrics.current_ratio,
                    "score": liquidity_score
                }

            # Profit Margin Analysis
            if metrics.profit_margin is not None:
                margin_score = self._score_profit_margin(
                    metrics.profit_margin,
                    benchmarks.get("margin", 10)
                )
                health_scores.append(margin_score)
                details["margin_analysis"] = {
                    "value": metrics.profit_margin,
                    "benchmark": benchmarks.get("margin", 10),
                    "score": margin_score
                }

            # Growth Analysis
            growth_score = self._analyze_growth_metrics(metrics)
            if growth_score is not None:
                health_scores.append(growth_score)
                details["growth_analysis"] = growth_score

            # Calculate overall health score
            if health_scores:
                avg_score = sum(health_scores) / len(health_scores)
                health_rating = self._determine_health_rating(avg_score)
            else:
                avg_score = 50.0
                health_rating = FinancialHealth.AVERAGE

            return {
                "health_score": avg_score,
                "health_rating": health_rating,
                "details": details,
                "metrics_available": len(health_scores)
            }

        except Exception as e:
            self.logger.error(f"Error in financial health analysis: {e}")
            return {
                "health_score": 50.0,
                "health_rating": FinancialHealth.AVERAGE,
                "error": str(e)
            }

    def analyze_volume_surge(self, current_volume: float, avg_volume: float,
                           market_data: pl.DataFrame) -> Dict[str, Any]:
        """
        Analyze volume surge patterns and their significance.

        Args:
            current_volume: Current trading volume
            avg_volume: Average volume
            market_data: Historical market data for context

        Returns:
            Volume surge analysis
        """
        try:
            if avg_volume <= 0:
                return {"surge_ratio": 1.0, "significance": "none", "score": 50.0}

            surge_ratio = current_volume / avg_volume

            # Calculate volume percentile over recent period
            recent_volumes = market_data.select("volume").tail(60).to_series().to_numpy()
            volume_percentile = (np.sum(recent_volumes <= current_volume) / len(recent_volumes)) * 100

            # Determine significance
            if surge_ratio >= 3.0 and volume_percentile >= 95:
                significance = "extreme"
                score = 90.0
            elif surge_ratio >= 2.0 and volume_percentile >= 90:
                significance = "high"
                score = 75.0
            elif surge_ratio >= 1.5 and volume_percentile >= 80:
                significance = "moderate"
                score = 65.0
            elif surge_ratio >= 1.2 and volume_percentile >= 70:
                significance = "mild"
                score = 55.0
            else:
                significance = "none"
                score = 50.0

            # Price-volume relationship
            price_change = 0.0
            if market_data.height >= 2:
                current_price = float(market_data.select("close").tail(1).item())
                prev_price = float(market_data.select("close").slice(-2, 1).item())
                price_change = (current_price - prev_price) / prev_price

            # Volume-price divergence check
            divergence = False
            if surge_ratio >= 1.5 and abs(price_change) < 0.01:
                divergence = True
                significance += "_divergence"

            return {
                "surge_ratio": surge_ratio,
                "volume_percentile": volume_percentile,
                "significance": significance,
                "score": score,
                "price_change": price_change,
                "divergence": divergence
            }

        except Exception as e:
            self.logger.error(f"Error in volume surge analysis: {e}")
            return {"surge_ratio": 1.0, "significance": "error", "score": 50.0}

    def analyze_sector_performance(self, sector: str, industry: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze sector and industry performance relative to market.

        Args:
            sector: Stock sector
            industry: Stock industry (optional)

        Returns:
            Sector performance analysis
        """
        try:
            # This would typically fetch real-time sector data
            # For now, we'll use simulated relative performance

            # Sector rotation strength (simulated - would come from market data)
            sector_strength = {
                "Technology": 0.15,
                "Healthcare": 0.08,
                "Finance": -0.05,
                "Consumer Cyclical": 0.12,
                "Consumer Defensive": -0.02,
                "Energy": 0.25,
                "Utilities": -0.08,
                "Real Estate": 0.03,
                "Materials": 0.18,
                "Industrials": 0.10,
                "Communication Services": -0.03
            }

            sector_perf = sector_strength.get(sector, 0.0)

            # Convert to score (0-100)
            # Positive performance above 10% gets high scores
            if sector_perf >= 0.10:
                score = min(90.0, 50 + sector_perf * 200)
                rating = "strong"
            elif sector_perf >= 0.05:
                score = 50 + sector_perf * 100
                rating = "moderate"
            elif sector_perf >= -0.05:
                score = 50 + sector_perf * 100
                rating = "neutral"
            elif sector_perf >= -0.10:
                score = 50 + sector_perf * 100
                rating = "weak"
            else:
                score = max(10.0, 50 + sector_perf * 200)
                rating = "very_weak"

            return {
                "sector": sector,
                "industry": industry,
                "relative_performance": sector_perf,
                "score": score,
                "rating": rating,
                "trend": "bullish" if sector_perf > 0.02 else "bearish" if sector_perf < -0.02 else "neutral"
            }

        except Exception as e:
            self.logger.error(f"Error in sector performance analysis: {e}")
            return {
                "sector": sector,
                "score": 50.0,
                "rating": "neutral",
                "error": str(e)
            }

    def _score_pe_ratio(self, pe_ratio: float, benchmark_pe: float) -> float:
        """Score P/E ratio relative to sector benchmark."""
        if pe_ratio <= 0:
            return 30.0  # Negative earnings - risky

        ratio = pe_ratio / benchmark_pe

        if ratio <= 0.7:
            return 80.0  # Significantly undervalued
        elif ratio <= 0.9:
            return 70.0  # Undervalued
        elif ratio <= 1.1:
            return 60.0  # Fairly valued
        elif ratio <= 1.3:
            return 40.0  # Slightly overvalued
        elif ratio <= 1.5:
            return 30.0  # Overvalued
        else:
            return 20.0  # Significantly overvalued

    def _score_pb_ratio(self, pb_ratio: float, benchmark_pb: float) -> float:
        """Score P/B ratio relative to sector benchmark."""
        if pb_ratio <= 0:
            return 20.0  # Negative book value - very risky

        ratio = pb_ratio / benchmark_pb

        if ratio <= 0.5:
            return 85.0  # Deep value
        elif ratio <= 0.8:
            return 75.0  # Undervalued
        elif ratio <= 1.2:
            return 60.0  # Fairly valued
        elif ratio <= 1.5:
            return 45.0  # Slightly overvalued
        else:
            return 30.0  # Overvalued

    def _score_peg_ratio(self, peg_ratio: float) -> float:
        """Score PEG ratio (P/E to Growth)."""
        if peg_ratio <= 0:
            return 25.0  # Negative growth or earnings
        elif peg_ratio <= 0.5:
            return 90.0  # Excellent value
        elif peg_ratio <= 1.0:
            return 75.0  # Good value
        elif peg_ratio <= 1.5:
            return 60.0  # Fair value
        elif peg_ratio <= 2.0:
            return 45.0  # Expensive
        else:
            return 30.0  # Very expensive

    def _score_roe(self, roe: float, benchmark_roe: float) -> float:
        """Score Return on Equity."""
        if roe <= 0:
            return 20.0  # Negative ROE
        elif roe >= benchmark_roe * 1.5:
            return 90.0  # Excellent ROE
        elif roe >= benchmark_roe:
            return 75.0  # Above average ROE
        elif roe >= benchmark_roe * 0.75:
            return 60.0  # Average ROE
        elif roe >= benchmark_roe * 0.5:
            return 45.0  # Below average ROE
        else:
            return 30.0  # Poor ROE

    def _score_debt_to_equity(self, debt_ratio: float) -> float:
        """Score debt-to-equity ratio."""
        if debt_ratio < 0:
            return 50.0  # No data
        elif debt_ratio <= 0.3:
            return 90.0  # Low debt
        elif debt_ratio <= 0.6:
            return 75.0  # Moderate debt
        elif debt_ratio <= 1.0:
            return 60.0  # Average debt
        elif debt_ratio <= 1.5:
            return 45.0  # High debt
        else:
            return 25.0  # Very high debt

    def _score_current_ratio(self, current_ratio: float) -> float:
        """Score current ratio (liquidity)."""
        if current_ratio <= 0:
            return 10.0  # Very poor liquidity
        elif current_ratio >= 2.5:
            return 85.0  # Excellent liquidity
        elif current_ratio >= 2.0:
            return 80.0  # Good liquidity
        elif current_ratio >= 1.5:
            return 70.0  # Adequate liquidity
        elif current_ratio >= 1.0:
            return 50.0  # Minimal liquidity
        else:
            return 30.0  # Poor liquidity

    def _score_profit_margin(self, margin: float, benchmark_margin: float) -> float:
        """Score profit margin relative to sector."""
        if margin <= 0:
            return 20.0  # Unprofitable

        ratio = margin / benchmark_margin if benchmark_margin > 0 else 1.0

        if ratio >= 2.0:
            return 90.0  # Excellent margins
        elif ratio >= 1.5:
            return 80.0  # Good margins
        elif ratio >= 1.0:
            return 70.0  # Average margins
        elif ratio >= 0.5:
            return 50.0  # Below average margins
        else:
            return 30.0  # Poor margins

    def _analyze_growth_metrics(self, metrics: FundamentalMetrics) -> Optional[float]:
        """Analyze growth metrics and return composite score."""
        try:
            growth_scores = []

            # Revenue growth
            if metrics.revenue_growth is not None:
                if metrics.revenue_growth >= 20:
                    growth_scores.append(90.0)
                elif metrics.revenue_growth >= 10:
                    growth_scores.append(75.0)
                elif metrics.revenue_growth >= 5:
                    growth_scores.append(60.0)
                elif metrics.revenue_growth >= 0:
                    growth_scores.append(50.0)
                else:
                    growth_scores.append(30.0)

            # Earnings growth
            if metrics.earnings_growth is not None:
                if metrics.earnings_growth >= 25:
                    growth_scores.append(90.0)
                elif metrics.earnings_growth >= 15:
                    growth_scores.append(80.0)
                elif metrics.earnings_growth >= 10:
                    growth_scores.append(70.0)
                elif metrics.earnings_growth >= 5:
                    growth_scores.append(60.0)
                elif metrics.earnings_growth >= 0:
                    growth_scores.append(50.0)
                else:
                    growth_scores.append(25.0)

            return sum(growth_scores) / len(growth_scores) if growth_scores else None

        except Exception:
            return None

    def _safe_float(self, value) -> Optional[float]:
        """Safely convert a value to float."""
        if value is None:
            return None
        try:
            if isinstance(value, str):
                # Remove common non-numeric characters
                cleaned = value.replace('$', '').replace(',', '').replace('%', '').strip()
                if cleaned == '' or cleaned == '-' or cleaned == 'N/A':
                    return None
                return float(cleaned)
            return float(value)
        except (ValueError, TypeError):
            return None

    def _determine_valuation_tier(self, score: float) -> ValuationTier:
        """Determine valuation tier from score."""
        if score >= 75:
            return ValuationTier.UNDERVALUED
        elif score >= 55:
            return ValuationTier.FAIRLY_VALUED
        elif score >= 35:
            return ValuationTier.OVERVALUED
        else:
            return ValuationTier.EXTREME_OVERVALUED

    def _determine_health_rating(self, score: float) -> FinancialHealth:
        """Determine financial health rating from score."""
        if score >= 80:
            return FinancialHealth.EXCELLENT
        elif score >= 65:
            return FinancialHealth.GOOD
        elif score >= 45:
            return FinancialHealth.AVERAGE
        elif score >= 30:
            return FinancialHealth.POOR
        else:
            return FinancialHealth.DISTRESSED


class FundamentalStrategy(BaseStrategy):
    """
    Fundamental analysis strategy using FinViz and financial data.

    This strategy evaluates stocks based on financial health, valuation metrics,
    growth prospects, and sector performance to generate investment signals.
    """

    def __init__(self, config: StrategyConfig):
        """Initialize fundamental strategy."""
        super().__init__(config)

        # Default fundamental parameters
        self.default_params = {
            "min_market_cap": 1e9,        # $1B minimum market cap
            "max_pe_ratio": 30,           # Maximum P/E ratio
            "min_roe": 10,                # Minimum ROE percentage
            "max_debt_to_equity": 1.0,    # Maximum debt-to-equity ratio
            "min_current_ratio": 1.2,     # Minimum current ratio
            "min_revenue_growth": 5,      # Minimum revenue growth percentage
            "volume_surge_threshold": 2.0, # Volume surge multiplier
            "sector_weight": 0.2,         # Weight for sector performance
            "valuation_weight": 0.4,      # Weight for valuation metrics
            "health_weight": 0.3,         # Weight for financial health
            "growth_weight": 0.1,         # Weight for growth metrics
            "enable_value_investing": True,
            "enable_growth_investing": True,
            "enable_quality_investing": True
        }

        # Merge with user parameters
        self.params = {**self.default_params, **(self.config.parameters or {})}

        self.analyzer = FundamentalAnalyzer()

    def _setup_indicators(self) -> None:
        """Setup fundamental analysis components."""
        self.logger.info(f"Setting up fundamental analysis for {self.name}")
        # Fundamental analysis doesn't require technical indicators
        pass

    async def analyze(self, symbol: str, data: pl.DataFrame,
                     finviz_data: Optional[FinVizData] = None) -> Signal:
        """
        Perform comprehensive fundamental analysis.

        Args:
            symbol: Trading symbol
            data: Historical market data
            finviz_data: FinViz fundamental data

        Returns:
            Fundamental analysis signal
        """
        try:
            if finviz_data is None:
                return Signal(
                    action=SignalType.HOLD,
                    confidence=0.0,
                    position_size=0.0,
                    reasoning="Invalid or insufficient data"
                )

            # Extract metrics from FinViz data
            metrics = self._extract_metrics(finviz_data)

            # Perform various fundamental analyses
            valuation_analysis = self.analyzer.analyze_valuation(
                metrics, finviz_data.sector
            )

            health_analysis = self.analyzer.analyze_growth(
                metrics, finviz_data.sector
            )

            # Volume surge analysis
            current_volume = float(data.select("volume").tail(1).item())
            avg_volume = float(data.select("volume").tail(20).mean().item())
            volume_analysis = self.analyzer.analyze_volume_surge(
                current_volume, avg_volume, data
            )

            # Sector performance
            sector_analysis = None
            if finviz_data.sector and finviz_data.industry:
                sector_analysis = self.analyzer.analyze_sector_performance(
                    finviz_data.sector, finviz_data.industry
                )

            # Screen based on minimum criteria
            screening_result = self._screen_stock(metrics, finviz_data)
            if not screening_result["passed"]:
                return Signal(
                    action=SignalType.HOLD,
                    confidence=0.0,
                    position_size=0.0,
                    reasoning=f"Failed screening: {screening_result['reason']}",
                    metadata={"screening": screening_result}
                )

            # Calculate composite fundamental score
            composite_score = self._calculate_composite_score(
                valuation_analysis, health_analysis, sector_analysis or {}, volume_analysis
            )

            # Generate signal based on composite score
            signal = self._generate_fundamental_signal(
                symbol, composite_score, valuation_analysis,
                health_analysis, sector_analysis or {}, volume_analysis, data
            )

            return signal

        except Exception as e:
            self.logger.error(f"Error in fundamental analysis for {symbol}: {e}")
            return Signal(
                action=SignalType.HOLD,
                confidence=0.0,
                position_size=0.0,
                reasoning="No FinViz data available"
            )

    def _extract_metrics(self, finviz_data: FinVizData) -> FundamentalMetrics:
        """Extract metrics from FinViz data."""
        try:
            return FundamentalMetrics(
                pe_ratio=getattr(finviz_data, 'pe_ratio', None),
                peg_ratio=getattr(finviz_data, 'peg_ratio', None),
                pb_ratio=getattr(finviz_data, 'pb_ratio', None),
                ps_ratio=getattr(finviz_data, 'ps_ratio', None),
                debt_to_equity=getattr(finviz_data, 'debt_equity', None),
                current_ratio=getattr(finviz_data, 'current_ratio', None),
                roe=getattr(finviz_data, 'roe', None),
                roa=getattr(finviz_data, 'roa', None),
                profit_margin=getattr(finviz_data, 'profit_margin', None),
                revenue_growth=getattr(finviz_data, 'sales_growth_qtr', None),
                earnings_growth=getattr(finviz_data, 'eps_growth_qtr', None),
                market_cap=self._safe_float(getattr(finviz_data, 'market_cap', None)),
            )
        except Exception as e:
            self.logger.error(f"Error extracting metrics: {e}")
            return FundamentalMetrics()

    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        try:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                # Remove common formatting characters
                cleaned = value.replace(',', '').replace('%', '').replace('$', '').replace('B', '').replace('M', '').replace('K', '')
                return float(cleaned)
            return None
        except (ValueError, TypeError):
            return None

    def _screen_stock(self, metrics: FundamentalMetrics,
                     finviz_data: FinVizData) -> Dict[str, Any]:
        """Screen stock based on minimum fundamental criteria."""
        try:
            reasons = []

            # Market cap screening
            if (metrics.market_cap is not None and
                metrics.market_cap < self.params["min_market_cap"]):
                reasons.append(f"Market cap too small: ${metrics.market_cap/1e9:.1f}B")

            # P/E ratio screening
            if (metrics.pe_ratio is not None and
                metrics.pe_ratio > self.params["max_pe_ratio"]):
                reasons.append(f"P/E too high: {metrics.pe_ratio:.1f}")

            # ROE screening
            if (metrics.roe is not None and
                metrics.roe < self.params["min_roe"]):
                reasons.append(f"ROE too low: {metrics.roe:.1f}%")

            # Debt screening
            if (metrics.debt_to_equity is not None and
                metrics.debt_to_equity > self.params["max_debt_to_equity"]):
                reasons.append(f"Debt/Equity too high: {metrics.debt_to_equity:.2f}")

            # Liquidity screening
            if (metrics.current_ratio is not None and
                metrics.current_ratio < self.params["min_current_ratio"]):
                reasons.append(f"Current ratio too low: {metrics.current_ratio:.2f}")

            # Revenue growth screening
            if (metrics.revenue_growth is not None and
                metrics.revenue_growth < self.params["min_revenue_growth"]):
                reasons.append(f"Revenue growth too low: {metrics.revenue_growth:.1f}%")

            passed = len(reasons) == 0

            return {
                "passed": passed,
                "reason": "; ".join(reasons) if reasons else "Passed all screens",
                "failed_criteria": len(reasons),
                "total_criteria": 6
            }

        except Exception as e:
            self.logger.error(f"Error in stock screening: {e}")
            return {"passed": False, "reason": f"Screening error: {str(e)}"}

    def _calculate_composite_score(self, valuation_analysis: Dict,
                                 health_analysis: Dict, sector_analysis: Dict,
                                 volume_analysis: Dict) -> Dict[str, Any]:
        """Calculate composite fundamental score."""
        try:
            # Extract component scores
            valuation_score = valuation_analysis.get("valuation_score", 50.0)
            health_score = health_analysis.get("health_score", 50.0)
            sector_score = sector_analysis.get("score", 50.0)
            volume_score = volume_analysis.get("score", 50.0)

            # Apply weights
            weighted_score = (
                valuation_score * self.params["valuation_weight"] +
                health_score * self.params["health_weight"] +
                sector_score * self.params["sector_weight"] +
                volume_score * (1 - self.params["valuation_weight"] -
                               self.params["health_weight"] - self.params["sector_weight"])
            )

            # Investment style adjustments
            style_adjustments = []

            if self.params["enable_value_investing"]:
                # Boost undervalued stocks
                if valuation_analysis.get("valuation_tier") == ValuationTier.UNDERVALUED:
                    weighted_score += 10
                    style_adjustments.append("value_boost")

            if self.params["enable_growth_investing"]:
                # Boost high-growth stocks
                growth_details = health_analysis.get("details", {}).get("growth_analysis")
                if growth_details and growth_details.get("score", 0) >= 75:
                    weighted_score += 8
                    style_adjustments.append("growth_boost")

            if self.params["enable_quality_investing"]:
                # Boost high-quality stocks
                if health_analysis.get("health_rating") == FinancialHealth.EXCELLENT:
                    weighted_score += 5
                    style_adjustments.append("quality_boost")

            return {
                "composite_score": max(0.0, min(100.0, weighted_score)),
                "component_scores": {
                    "valuation": valuation_score,
                    "health": health_score,
                    "sector": sector_score,
                    "volume": volume_score
                },
                "weights_applied": {
                    "valuation": self.params["valuation_weight"],
                    "health": self.params["health_weight"],
                    "sector": self.params["sector_weight"]
                },
                "style_adjustments": style_adjustments,
                "raw_weighted_score": weighted_score
            }

        except Exception as e:
            self.logger.error(f"Error calculating composite score: {e}")
            return {
                "composite_score": 50.0,
                "error": str(e)
            }

    def _generate_fundamental_signal(self, symbol: str, composite_score: Dict,
                                   valuation_analysis: Dict, health_analysis: Dict,
                                   sector_analysis: Dict, volume_analysis: Dict,
                                   data: pl.DataFrame) -> Signal:
        """Generate trading signal based on fundamental analysis."""
        try:
            score = composite_score["composite_score"]
            current_price = float(data.select("close").tail(1).item())

            # Determine action based on score
            if score >= 75:
                action = SignalType.BUY
                confidence = min(95.0, score)
                reasoning_parts = ["Strong fundamentals"]
            elif score >= 60:
                action = SignalType.BUY
                confidence = score * 0.9
                reasoning_parts = ["Good fundamentals"]
            elif score <= 25:
                action = SignalType.SELL
                confidence = (100 - score) * 0.9
                reasoning_parts = ["Weak fundamentals"]
            elif score <= 40:
                action = SignalType.SELL
                confidence = (100 - score) * 0.7
                reasoning_parts = ["Below average fundamentals"]
            else:
                action = SignalType.HOLD
                confidence = 50.0
                reasoning_parts = ["Neutral fundamentals"]

            # Build detailed reasoning
            if valuation_analysis.get("valuation_tier") == ValuationTier.UNDERVALUED:
                reasoning_parts.append("undervalued")
            elif valuation_analysis.get("valuation_tier") == ValuationTier.OVERVALUED:
                reasoning_parts.append("overvalued")

            if health_analysis.get("health_rating") == FinancialHealth.EXCELLENT:
                reasoning_parts.append("excellent financial health")
            elif health_analysis.get("health_rating") == FinancialHealth.POOR:
                reasoning_parts.append("poor financial health")

            if sector_analysis.get("rating") == "strong":
                reasoning_parts.append("strong sector")
            elif sector_analysis.get("rating") == "weak":
                reasoning_parts.append("weak sector")

            if volume_analysis.get("significance") in ["high", "extreme"]:
                reasoning_parts.append("volume surge")

            reasoning = ", ".join(reasoning_parts)

            return Signal(
                action=action,
                confidence=confidence,
                entry_price=Decimal(str(current_price)),
                position_size=self._calculate_position_size(confidence),
                reasoning=reasoning,
                metadata={
                    "strategy_type": "fundamental",
                    "composite_score": composite_score,
                    "valuation": valuation_analysis,
                    "health": health_analysis,
                    "sector": sector_analysis,
                    "volume": volume_analysis,
                    "fundamental_score": score
                }
            )

        except Exception as e:
            self.logger.error(f"Error generating fundamental signal: {e}")
            return Signal(
                action=SignalType.HOLD,
                confidence=0.0,
                position_size=0.0,
                reasoning="Insufficient financial data for analysis"
            )


class FinVizDataProcessor:
    """Process and normalize FinViz data for analysis."""

    @staticmethod
    def parse_finviz_metrics(raw_data: Dict[str, Any]) -> FundamentalMetrics:
        """
        Parse raw FinViz data into structured metrics.

        Args:
            raw_data: Raw FinViz data dictionary

        Returns:
            Structured fundamental metrics
        """
        try:
            def safe_float(value, default=None):
                """Safely convert value to float."""
                if value in [None, "", "-", "N/A"]:
                    return default
                try:
                    # Handle percentage values
                    if isinstance(value, str) and value.endswith("%"):
                        return float(value[:-1])
                    # Handle multiplier values (K, M, B)
                    if isinstance(value, str):
                        multipliers = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
                        for suffix, mult in multipliers.items():
                            if value.upper().endswith(suffix):
                                return float(value[:-1]) * mult
                    return float(value)
                except (ValueError, TypeError):
                    return default

            return FundamentalMetrics(
                pe_ratio=safe_float(raw_data.get("P/E")),
                peg_ratio=safe_float(raw_data.get("PEG")),
                pb_ratio=safe_float(raw_data.get("P/B")),
                ps_ratio=safe_float(raw_data.get("P/S")),
                debt_to_equity=safe_float(raw_data.get("Debt/Eq")),
                current_ratio=safe_float(raw_data.get("Current Ratio")),
                roe=safe_float(raw_data.get("ROE")),
                roa=safe_float(raw_data.get("ROA")),
                profit_margin=safe_float(raw_data.get("Profit Margin")),
                revenue_growth=safe_float(raw_data.get("Sales growth past 5 years")),
                earnings_growth=safe_float(raw_data.get("EPS growth past 5 years")),
                market_cap=safe_float(raw_data.get("Market Cap")),
                analyst_rating=raw_data.get("Analyst Recom")
            )

        except Exception as e:
            logging.getLogger("finviz_processor").error(f"Error parsing FinViz data: {e}")
            return FundamentalMetrics()

    @staticmethod
    def calculate_quality_score(metrics: FundamentalMetrics) -> float:
        """
        Calculate overall quality score based on fundamental metrics.

        Args:
            metrics: Fundamental metrics

        Returns:
            Quality score (0-100)
        """
        try:
            scores = []

            # Profitability scores
            if metrics.roe is not None:
                if metrics.roe >= 20:
                    scores.append(90)
                elif metrics.roe >= 15:
                    scores.append(80)
                elif metrics.roe >= 10:
                    scores.append(60)
                elif metrics.roe >= 5:
                    scores.append(40)
                else:
                    scores.append(20)

            # Efficiency scores
            if metrics.roa is not None:
                if metrics.roa >= 10:
                    scores.append(90)
                elif metrics.roa >= 5:
                    scores.append(70)
                elif metrics.roa >= 2:
                    scores.append(50)
                else:
                    scores.append(30)

            # Debt management
            if metrics.debt_to_equity is not None:
                if metrics.debt_to_equity <= 0.3:
                    scores.append(90)
                elif metrics.debt_to_equity <= 0.6:
                    scores.append(70)
                elif metrics.debt_to_equity <= 1.0:
                    scores.append(50)
                else:
                    scores.append(30)

            # Liquidity
            if metrics.current_ratio is not None:
                if metrics.current_ratio >= 2.5:
                    scores.append(90)
                elif metrics.current_ratio >= 2.0:
                    scores.append(80)
                elif metrics.current_ratio >= 1.5:
                    scores.append(60)
                elif metrics.current_ratio >= 1.0:
                    scores.append(40)
                else:
                    scores.append(20)

            # Profitability margins
            if metrics.profit_margin is not None:
                if metrics.profit_margin >= 20:
                    scores.append(90)
                elif metrics.profit_margin >= 15:
                    scores.append(80)
                elif metrics.profit_margin >= 10:
                    scores.append(70)
                elif metrics.profit_margin >= 5:
                    scores.append(50)
                else:
                    scores.append(30)

            return sum(scores) / len(scores) if scores else 50.0

        except Exception:
            return 50.0

    @staticmethod
    def detect_earnings_momentum(metrics: FundamentalMetrics) -> Dict[str, Any]:
        """Detect earnings and revenue momentum."""
        try:
            momentum_score = 50.0
            signals = []

            # Revenue growth momentum
            if metrics.revenue_growth is not None:
                if metrics.revenue_growth >= 20:
                    momentum_score += 20
                    signals.append("strong_revenue_growth")
                elif metrics.revenue_growth >= 10:
                    momentum_score += 10
                    signals.append("good_revenue_growth")
                elif metrics.revenue_growth <= -5:
                    momentum_score -= 15
                    signals.append("declining_revenue")

            # Earnings growth momentum
            if metrics.earnings_growth is not None:
                if metrics.earnings_growth >= 25:
                    momentum_score += 25
                    signals.append("strong_earnings_growth")
                elif metrics.earnings_growth >= 15:
                    momentum_score += 15
                    signals.append("good_earnings_growth")
                elif metrics.earnings_growth <= -10:
                    momentum_score -= 20
                    signals.append("declining_earnings")

            # PEG ratio consideration
            if metrics.peg_ratio is not None:
                if metrics.peg_ratio <= 0.5:
                    momentum_score += 15
                    signals.append("excellent_peg")
                elif metrics.peg_ratio <= 1.0:
                    momentum_score += 8
                    signals.append("good_peg")
                elif metrics.peg_ratio >= 2.0:
                    momentum_score -= 10
                    signals.append("expensive_growth")

            return {
                "momentum_score": max(0.0, min(100.0, momentum_score)),
                "signals": signals,
                "revenue_growth": metrics.revenue_growth,
                "earnings_growth": metrics.earnings_growth,
                "peg_ratio": metrics.peg_ratio
            }

        except Exception as e:
            logging.getLogger("finviz_processor").error(f"Error detecting earnings momentum: {e}")
            return {
                "momentum_score": 50.0,
                "signals": [],
                "error": str(e)
            }


class SectorAnalyzer:
    """Analyze sector and industry performance and rotation."""

    def __init__(self):
        """Initialize sector analyzer."""
        self.logger = logging.getLogger("sector_analyzer")

        # Sector rotation patterns (simplified model)
        self.sector_cycle = {
            "early_cycle": ["Technology", "Consumer Cyclical", "Industrials"],
            "mid_cycle": ["Materials", "Energy", "Finance"],
            "late_cycle": ["Consumer Defensive", "Healthcare", "Utilities"],
            "recession": ["Consumer Defensive", "Healthcare", "Utilities"]
        }

    def analyze_sector_rotation(self, current_sector: str,
                              market_phase: str = "mid_cycle") -> Dict[str, Any]:
        """
        Analyze sector rotation and positioning.

        Args:
            current_sector: Sector to analyze
            market_phase: Current market cycle phase

        Returns:
            Sector rotation analysis
        """
        try:
            favorable_sectors = self.sector_cycle.get(market_phase, [])
            is_favorable = current_sector in favorable_sectors

            # Score based on sector rotation
            if is_favorable:
                rotation_score = 75.0
                rotation_signal = "favorable"
            else:
                rotation_score = 40.0
                rotation_signal = "unfavorable"

            # Additional sector-specific analysis
            sector_metrics = self._get_sector_metrics(current_sector)

            return {
                "sector": current_sector,
                "market_phase": market_phase,
                "is_favorable": is_favorable,
                "rotation_score": rotation_score,
                "rotation_signal": rotation_signal,
                "sector_metrics": sector_metrics,
                "favorable_sectors": favorable_sectors
            }

        except Exception as e:
            self.logger.error(f"Error in sector rotation analysis: {e}")
            return {
                "sector": current_sector,
                "rotation_score": 50.0,
                "error": str(e)
            }

    def _get_sector_metrics(self, sector: str) -> Dict[str, Any]:
        """Get sector-specific metrics and characteristics."""
        # Sector characteristics (would be updated from real data)
        sector_data = {
            "Technology": {
                "volatility": "high",
                "growth_oriented": True,
                "cyclical": True,
                "defensive": False,
                "avg_pe": 25,
                "avg_growth": 15
            },
            "Healthcare": {
                "volatility": "low",
                "growth_oriented": True,
                "cyclical": False,
                "defensive": True,
                "avg_pe": 18,
                "avg_growth": 8
            },
            "Finance": {
                "volatility": "high",
                "growth_oriented": False,
                "cyclical": True,
                "defensive": False,
                "avg_pe": 12,
                "avg_growth": 5
            },
            "Consumer Defensive": {
                "volatility": "low",
                "growth_oriented": False,
                "cyclical": False,
                "defensive": True,
                "avg_pe": 20,
                "avg_growth": 3
            }
        }

        return sector_data.get(sector, {
            "volatility": "medium",
            "growth_oriented": False,
            "cyclical": True,
            "defensive": False,
            "avg_pe": 18,
            "avg_growth": 7
        })


class FundamentalScreener:
    """Screen stocks based on fundamental criteria."""

    def __init__(self):
        """Initialize fundamental screener."""
        self.logger = logging.getLogger("fundamental_screener")

    def value_screen(self, metrics: FundamentalMetrics,
                    sector_benchmarks: Dict[str, float]) -> Dict[str, Any]:
        """Screen for value investing opportunities."""
        try:
            value_signals = []
            score = 50.0

            # Low P/E relative to sector
            if metrics.pe_ratio and sector_benchmarks.get("pe"):
                pe_ratio = metrics.pe_ratio / sector_benchmarks["pe"]
                if pe_ratio <= 0.7:
                    value_signals.append("low_pe")
                    score += 15
                elif pe_ratio <= 0.9:
                    value_signals.append("below_avg_pe")
                    score += 8

            # Low P/B ratio
            if metrics.pb_ratio:
                if metrics.pb_ratio <= 1.0:
                    value_signals.append("low_pb")
                    score += 12
                elif metrics.pb_ratio <= 1.5:
                    value_signals.append("reasonable_pb")
                    score += 6

            # Good PEG ratio
            if metrics.peg_ratio:
                if metrics.peg_ratio <= 1.0:
                    value_signals.append("good_peg")
                    score += 10

            # High dividend yield (if available)
            # Note: Would need dividend data from FinViz

            return {
                "screen_type": "value",
                "passed": len(value_signals) >= 2,
                "score": min(100.0, score),
                "signals": value_signals,
                "criteria_met": len(value_signals)
            }

        except Exception as e:
            self.logger.error(f"Error in value screening: {e}")
            return {"screen_type": "value", "passed": False, "error": str(e)}

    def growth_screen(self, metrics: FundamentalMetrics) -> Dict[str, Any]:
        """Screen for growth investing opportunities."""
        try:
            growth_signals = []
            score = 50.0

            # Revenue growth
            if metrics.revenue_growth:
                if metrics.revenue_growth >= 15:
                    growth_signals.append("strong_revenue_growth")
                    score += 20
                elif metrics.revenue_growth >= 10:
                    growth_signals.append("good_revenue_growth")
                    score += 12

            # Earnings growth
            if metrics.earnings_growth:
                if metrics.earnings_growth >= 20:
                    growth_signals.append("strong_earnings_growth")
                    score += 25
                elif metrics.earnings_growth >= 15:
                    growth_signals.append("good_earnings_growth")
                    score += 15

            # High ROE (growth quality)
            if metrics.roe:
                if metrics.roe >= 20:
                    growth_signals.append("excellent_roe")
                    score += 15
                elif metrics.roe >= 15:
                    growth_signals.append("good_roe")
                    score += 10

            # Reasonable valuation for growth
            if metrics.peg_ratio:
                if metrics.peg_ratio <= 1.5:
                    growth_signals.append("reasonable_growth_valuation")
                    score += 10

            return {
                "screen_type": "growth",
                "passed": len(growth_signals) >= 2,
                "score": min(100.0, score),
                "signals": growth_signals,
                "criteria_met": len(growth_signals)
            }

        except Exception as e:
            self.logger.error(f"Error in growth screening: {e}")
            return {"screen_type": "growth", "passed": False, "error": str(e)}

    def quality_screen(self, metrics: FundamentalMetrics) -> Dict[str, Any]:
        """Screen for quality investing opportunities."""
        try:
            quality_signals = []
            score = 50.0

            # High ROE
            if metrics.roe:
                if metrics.roe >= 20:
                    quality_signals.append("excellent_roe")
                    score += 20
                elif metrics.roe >= 15:
                    quality_signals.append("good_roe")
                    score += 15

            # High ROA
            if metrics.roa:
                if metrics.roa >= 8:
                    quality_signals.append("excellent_roa")
                    score += 15
                elif metrics.roa >= 5:
                    quality_signals.append("good_roa")
                    score += 10

            # Low debt
            if metrics.debt_to_equity:
                if metrics.debt_to_equity <= 0.3:
                    quality_signals.append("low_debt")
                    score += 15
                elif metrics.debt_to_equity <= 0.6:
                    quality_signals.append("moderate_debt")
                    score += 8

            # Good liquidity
            if metrics.current_ratio:
                if metrics.current_ratio >= 2.0:
                    quality_signals.append("excellent_liquidity")
                    score += 12
                elif metrics.current_ratio >= 1.5:
                    quality_signals.append("good_liquidity")
                    score += 8

            # High margins
            if metrics.profit_margin:
                if metrics.profit_margin >= 15:
                    quality_signals.append("high_margins")
                    score += 10
                elif metrics.profit_margin >= 10:
                    quality_signals.append("good_margins")
                    score += 6

            return {
                "screen_type": "quality",
                "passed": len(quality_signals) >= 3,
                "score": min(100.0, score),
                "signals": quality_signals,
                "criteria_met": len(quality_signals)
            }

        except Exception as e:
            self.logger.error(f"Error in quality screening: {e}")
            return {"screen_type": "quality", "passed": False, "error": str(e)}


class FundamentalAnalysisEngine:
    """Main fundamental analysis engine."""

    def __init__(self):
        """Initialize fundamental analysis engine."""
        self.logger = logging.getLogger("fundamental_analysis")
        self.analyzer = FundamentalAnalyzer()
        self.screener = FundamentalScreener()
        self.sector_analyzer = SectorAnalyzer()
        self.data_processor = FinVizDataProcessor()

    def full_analysis(self, symbol: str, finviz_data: FinVizData,
                     market_data: pl.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis.

        Args:
            symbol: Trading symbol
            finviz_data: FinViz fundamental data
            market_data: Historical market data

        Returns:
            Complete fundamental analysis results
        """
        try:
            # Extract and process metrics
            metrics = self.data_processor.parse_finviz_metrics(finviz_data.__dict__)

            # Valuation analysis
            valuation_analysis = self.analyzer.analyze_valuation(
                metrics, finviz_data.sector
            )

            # Financial health analysis
            health_analysis = self.analyzer.analyze_growth(
                metrics, finviz_data.sector
            )

            # Quality score
            quality_score = self.data_processor.calculate_quality_score(metrics)

            # Earnings momentum
            earnings_momentum = self.data_processor.detect_earnings_momentum(metrics)

            # Sector analysis
            sector_analysis = None
            if finviz_data.sector:
                sector_analysis = self.sector_analyzer.analyze_sector_rotation(
                    finviz_data.sector
                )

            # Investment style screens
            value_screen = self.screener.value_screen(
                metrics, self.analyzer.industry_benchmarks.get(finviz_data.sector or "unknown", {})
            )
            growth_screen = self.screener.growth_screen(metrics)
            quality_screen = self.screener.quality_screen(metrics)

            # Volume analysis
            current_volume = float(market_data.select("volume").tail(1).item())
            avg_volume = float(market_data.select("volume").tail(20).mean().item())
            volume_analysis = self.analyzer.analyze_volume_surge(
                current_volume, avg_volume, market_data
            )

            # Calculate composite fundamental score
            composite_score = self._calculate_composite_fundamental_score(
                valuation_analysis, health_analysis, quality_score,
                earnings_momentum, sector_analysis or {}, volume_analysis
            )

            return {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc),
                "composite_score": composite_score,
                "valuation": valuation_analysis,
                "health": health_analysis,
                "quality_score": quality_score,
                "earnings_momentum": earnings_momentum,
                "sector": sector_analysis,
                "volume": volume_analysis,
                "screens": {
                    "value": value_screen,
                    "growth": growth_screen,
                    "quality": quality_screen
                },
                "metrics": metrics,
                "finviz_data": finviz_data
            }

        except Exception as e:
            self.logger.error(f"Error in full fundamental analysis for {symbol}: {e}")
            return {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc),
                "composite_score": 50.0,
                "error": str(e)
            }

    def _calculate_composite_fundamental_score(self, valuation: Dict, health: Dict,
                                             quality_score: float, earnings_momentum: Dict,
                                             sector: Dict, volume: Dict) -> float:
        """Calculate composite fundamental score."""
        try:
            # Component scores
            val_score = valuation.get("valuation_score", 50.0)
            health_score = health.get("health_score", 50.0)
            momentum_score = earnings_momentum.get("momentum_score", 50.0)
            sector_score = sector.get("rotation_score", 50.0)
            volume_score = volume.get("score", 50.0)

            # Weighted combination
            composite = (
                val_score * 0.25 +        # Valuation: 25%
                health_score * 0.25 +     # Financial health: 25%
                quality_score * 0.20 +    # Quality: 20%
                momentum_score * 0.15 +   # Earnings momentum: 15%
                sector_score * 0.10 +     # Sector rotation: 10%
                volume_score * 0.05       # Volume: 5%
            )

            return max(0.0, min(100.0, composite))

        except Exception as e:
            self.logger.error(f"Error calculating composite fundamental score: {e}")
            return 50.0
