"""
Ollama AI Strategy Adapter for Backtesting

This module provides an adapter that integrates Ollama-powered AI analysis
into the existing backtesting framework, allowing real AI-powered backtests
using local models instead of expensive cloud APIs.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Import backtesting models
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../services/strategy_engine/src")
)

from backtest_models import Signal, SignalType  # noqa: E402
from ollama_client import OllamaClient, OllamaResponse  # noqa: E402

logger = logging.getLogger(__name__)


class OllamaAIStrategyAdapter:
    """
    AI Strategy adapter using Ollama for backtesting.

    This adapter integrates with the existing backtesting framework while
    using local Ollama models for AI-powered trading decisions.
    """

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        prompts_config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Ollama AI Strategy.

        Args:
            ollama_url: Ollama server URL
            ollama_model: Model name to use
            prompts_config_path: Path to prompts YAML config
            config: Additional configuration options
        """
        self.config = config or {}

        # Initialize Ollama client
        self.ollama_url = ollama_url or os.getenv(
            "OLLAMA_URL", "http://192.168.1.133:11434"
        )
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.1:latest")
        self.ollama_client = OllamaClient(self.ollama_url, self.ollama_model)

        # Load prompts configuration
        self.prompts_config = self._load_prompts_config(prompts_config_path)

        # Performance tracking
        self.total_calls = 0
        self.successful_calls = 0
        self.total_response_time = 0.0
        self.total_cost = 0.0  # Always 0 for local models

        # Configuration parameters
        self.confidence_threshold = self.config.get("confidence_threshold", 60.0)
        self.max_position_size = self.config.get("max_position_size", 0.10)
        self.risk_tolerance = self.config.get("risk_tolerance", "medium")

        logger.info(
            f"Ollama AI Strategy initialized: {self.ollama_url} using {self.ollama_model}"
        )

    def _load_prompts_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load prompts configuration from YAML file."""
        if config_path is None:
            config_path = str(
                Path(__file__).parent.parent / "config" / "ai_strategy" / "prompts.yaml"
            )

        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Prompts config not found at {config_path}, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading prompts config: {e}")
            return {}

    async def generate_signal(
        self, symbol: str, current_data, historical_data, market_context
    ) -> Optional[Signal]:
        """
        Generate trading signal using Ollama AI.

        Args:
            symbol: Stock ticker symbol
            current_data: Current market data
            historical_data: Historical price/volume data
            market_context: Current market conditions

        Returns:
            Signal object or None
        """
        self.total_calls += 1

        try:
            # Health check first
            if not await self.ollama_client.health_check():
                logger.warning("Ollama server not healthy, skipping signal generation")
                return None

            # Build comprehensive prompt using production templates
            prompt = self._build_analysis_prompt(
                symbol, current_data, historical_data, market_context
            )

            # Query Ollama
            start_time = datetime.now()
            response = await self.ollama_client.query(
                prompt, max_tokens=1000, temperature=0.3
            )
            end_time = datetime.now()

            response_time = (end_time - start_time).total_seconds()
            self.total_response_time += response_time

            # Parse AI response into trading signal
            signal = self._parse_ai_response(response, symbol, current_data)

            if signal:
                self.successful_calls += 1
                logger.debug(
                    f"Generated {signal.action.value} signal for {symbol} "
                    f"(confidence: {signal.confidence}%, time: {response_time:.2f}s)"
                )

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def _build_analysis_prompt(
        self, symbol: str, current_data, historical_data, market_context
    ) -> str:
        """Build comprehensive analysis prompt using production templates."""

        # Get master analyst template from config
        prompts = self.prompts_config.get("prompts", {})
        master_config = prompts.get("master_analyst", {})
        template = master_config.get("template", "")

        if not template:
            return self._build_fallback_prompt(symbol, current_data, historical_data)

        # Extract current market data
        current_price = (
            getattr(current_data, "close", 0)
            if hasattr(current_data, "close")
            else current_data.get("close", 0)
        )
        current_price = float(current_price)  # Ensure float for calculations
        volume = (
            getattr(current_data, "volume", 0)
            if hasattr(current_data, "volume")
            else current_data.get("volume", 0)
        )
        volume = float(volume)  # Ensure float for calculations

        # Calculate technical indicators from historical data
        tech_indicators = self._calculate_technical_indicators(
            historical_data, current_data
        )

        # Calculate daily change
        daily_change = 0.0
        if len(historical_data) >= 2:
            try:
                if hasattr(historical_data[-2], "close"):
                    prev_close = float(historical_data[-2].close)
                else:
                    prev_close = float(historical_data[-2].get("close", current_price))
                daily_change = ((current_price - prev_close) / prev_close) * 100
            except (IndexError, ZeroDivisionError, AttributeError):
                daily_change = 0.0

        # Build template data
        template_data = {
            "market_context": self._format_market_context(market_context),
            "ticker": symbol,
            "current_price": current_price,
            "daily_change": daily_change,
            "volume": volume,
            "avg_volume": tech_indicators.get("avg_volume", volume * 0.8),
            "rsi": tech_indicators.get("rsi", 50),
            "macd_signal": tech_indicators.get("macd_signal", 0),
            "macd_histogram": tech_indicators.get("macd_histogram", 0),
            "price_vs_sma20": tech_indicators.get("price_vs_sma20", 0),
            "price_vs_sma50": tech_indicators.get("price_vs_sma50", 0),
            "sma20_vs_sma50": tech_indicators.get("sma20_vs_sma50", 0),
            "bb_position": tech_indicators.get("bb_position", 50),
            "atr": tech_indicators.get("atr", current_price * 0.02),
            "atr_percentage": tech_indicators.get("atr_percentage", 2.0),
            "support_level": tech_indicators.get("support", current_price * 0.95),
            "resistance_level": tech_indicators.get("resistance", current_price * 1.05),
            "market_cap": "N/A",  # Would need fundamental data
            "pe_ratio": "N/A",
            "sector": "Unknown",
            "sector_performance": 0,
            "float_shares": "N/A",
            "short_interest": "N/A",
            "recent_candles": self._describe_recent_candles(historical_data),
            "identified_patterns": self._identify_technical_patterns(tech_indicators),
        }

        # Format template
        try:
            formatted_template = template
            for key, value in template_data.items():
                placeholder = "{" + key + "}"
                formatted_template = formatted_template.replace(placeholder, str(value))
            return formatted_template
        except Exception as e:
            logger.warning(f"Template formatting error: {e}, using fallback")
            return self._build_fallback_prompt(symbol, current_data, historical_data)

    def _build_fallback_prompt(self, symbol: str, current_data, historical_data) -> str:
        """Build simple fallback prompt when template fails."""

        current_price = (
            getattr(current_data, "close", 0)
            if hasattr(current_data, "close")
            else current_data.get("close", 0)
        )
        volume = (
            getattr(current_data, "volume", 0)
            if hasattr(current_data, "volume")
            else current_data.get("volume", 0)
        )

        # Simple RSI calculation
        rsi = (
            self._simple_rsi_calculation(historical_data)
            if len(historical_data) > 14
            else 50
        )

        return f"""
You are a professional stock analyst. Analyze {symbol} for trading opportunities.

CURRENT DATA:
- Stock: {symbol}
- Price: ${current_price}
- Volume: {volume:,}
- RSI: {rsi:.1f}
- Data Points: {len(historical_data)}

ANALYSIS:
- RSI indicates {'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'} conditions
- Volume is {'above average' if volume > 1000000 else 'normal'}

Provide your recommendation in JSON format:
{{
    "decision": "BUY" or "SELL" or "HOLD",
    "confidence": 0-100,
    "entry_price": {current_price},
    "stop_loss": price or null,
    "take_profit": price or null,
    "position_size_suggestion": 0.01 to {self.max_position_size},
    "reasoning": "Brief explanation of your decision",
    "key_risks": ["risk1", "risk2"]
}}

Focus on high-probability setups and proper risk management.
"""

    def _calculate_technical_indicators(
        self, historical_data, current_data
    ) -> Dict[str, float]:
        """Calculate technical indicators from historical data."""
        indicators: Dict[str, float] = {}

        if not historical_data or len(historical_data) < 2:
            return indicators

        try:
            # Extract closing prices
            prices = []
            volumes = []

            for data_point in historical_data:
                if hasattr(data_point, "close"):
                    prices.append(float(data_point.close))
                    volumes.append(
                        float(data_point.volume) if hasattr(data_point, "volume") else 0
                    )
                else:
                    prices.append(float(data_point.get("close", 0)))
                    volumes.append(float(data_point.get("volume", 0)))

            current_price = (
                getattr(current_data, "close", 0)
                if hasattr(current_data, "close")
                else current_data.get("close", 0)
            )
            current_price = float(
                current_price
            )  # Ensure current_price is float for calculations
            prices.append(current_price)

            # RSI calculation
            if len(prices) > 14:
                indicators["rsi"] = self._calculate_rsi(prices)

            # Simple Moving Averages
            if len(prices) >= 20:
                sma_20 = sum(prices[-20:]) / 20
                indicators["price_vs_sma20"] = ((current_price - sma_20) / sma_20) * 100

            if len(prices) >= 50:
                sma_50 = sum(prices[-50:]) / 50
                indicators["price_vs_sma50"] = ((current_price - sma_50) / sma_50) * 100

                if "price_vs_sma20" in indicators:
                    sma_20 = sum(prices[-20:]) / 20
                    indicators["sma20_vs_sma50"] = ((sma_20 - sma_50) / sma_50) * 100

            # Average volume
            if volumes and len(volumes) >= 20:
                indicators["avg_volume"] = sum(volumes[-20:]) / 20

            # Simple ATR approximation
            if len(prices) >= 5:
                price_ranges = [
                    abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))
                ]
                indicators["atr"] = sum(price_ranges[-14:]) / min(14, len(price_ranges))
                indicators["atr_percentage"] = (indicators["atr"] / current_price) * 100

            # Support and resistance (simple)
            if len(prices) >= 10:
                recent_prices = prices[-20:]
                indicators["support"] = min(recent_prices)
                indicators["resistance"] = max(recent_prices)

        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")

        return indicators

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0

        try:
            deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
            gains = [delta if delta > 0 else 0 for delta in deltas]
            losses = [-delta if delta < 0 else 0 for delta in deltas]

            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return max(0, min(100, rsi))

        except Exception:
            return 50.0

    def _simple_rsi_calculation(self, historical_data) -> float:
        """Simple RSI calculation for fallback."""
        if len(historical_data) < 15:
            return 50.0

        try:
            prices = []
            for data_point in historical_data:
                if hasattr(data_point, "close"):
                    prices.append(float(data_point.close))
                else:
                    prices.append(float(data_point.get("close", 0)))

            return self._calculate_rsi(prices)
        except Exception:
            return 50.0

    def _describe_recent_candles(self, historical_data) -> str:
        """Describe recent price action."""
        if not historical_data or len(historical_data) < 3:
            return "Insufficient data for candlestick analysis"

        try:
            # Get last few candles
            recent = historical_data[-3:]
            descriptions = []

            for i, candle in enumerate(recent):
                if hasattr(candle, "open"):
                    open_price = candle.open
                    close_price = candle.close
                else:
                    open_price = candle.get("open", 0)
                    close_price = candle.get("close", 0)

                if close_price > open_price:
                    descriptions.append(
                        f"Day {i + 1}: Green candle (+{((close_price - open_price) / open_price) * 100:.1f}%)"
                    )
                elif close_price < open_price:
                    descriptions.append(
                        f"Day {i + 1}: Red candle ({((close_price - open_price) / open_price) * 100:.1f}%)"
                    )
                else:
                    descriptions.append(f"Day {i + 1}: Doji candle")

            return "; ".join(descriptions)

        except Exception:
            return "Recent price action shows mixed signals"

    def _identify_technical_patterns(self, indicators: Dict[str, float]) -> str:
        """Identify technical patterns from indicators."""
        patterns = []

        try:
            rsi = indicators.get("rsi", 50)
            if rsi > 70:
                patterns.append("RSI overbought condition")
            elif rsi < 30:
                patterns.append("RSI oversold condition")

            price_vs_sma20 = indicators.get("price_vs_sma20", 0)
            if price_vs_sma20 > 5:
                patterns.append("Strong uptrend (price >> SMA20)")
            elif price_vs_sma20 < -5:
                patterns.append("Strong downtrend (price << SMA20)")

            sma20_vs_sma50 = indicators.get("sma20_vs_sma50", 0)
            if sma20_vs_sma50 > 2:
                patterns.append("Bullish moving average crossover")
            elif sma20_vs_sma50 < -2:
                patterns.append("Bearish moving average crossover")

        except Exception:
            pass

        return (
            "; ".join(patterns)
            if patterns
            else "No clear technical patterns identified"
        )

    def _format_market_context(self, market_context) -> str:
        """Format market context for prompt."""
        if not market_context:
            return "Market context unavailable"

        try:
            if isinstance(market_context, dict):
                spy_price = market_context.get("spy_price", 450)
                spy_change = market_context.get("spy_change", 0)
                vix_level = market_context.get("vix_level", 20)
                vix_change = market_context.get("vix_change", 0)
            else:
                # Handle object-style market context
                spy_price = getattr(market_context, "spy_price", 450)
                spy_change = getattr(market_context, "spy_change", 0)
                vix_level = getattr(market_context, "vix_level", 20)
                vix_change = getattr(market_context, "vix_change", 0)

            return f"""Market Environment:
- SPY: ${spy_price} ({spy_change:+.2f}%)
- VIX: {vix_level} ({vix_change:+.2f}%)
- Market Trend: {'Bullish' if spy_change > 0 else 'Bearish' if spy_change < 0 else 'Neutral'}
- Volatility: {'High' if vix_level > 25 else 'Low' if vix_level < 15 else 'Normal'}"""

        except Exception:
            return "Market conditions: Mixed signals, moderate volatility"

    def _parse_ai_response(
        self, response: OllamaResponse, symbol: str, current_data
    ) -> Optional[Signal]:
        """Parse Ollama response into a trading signal."""
        try:
            # Try to extract JSON from response
            json_start = response.content.find("{")
            json_end = response.content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response.content[json_start:json_end]
                parsed = json.loads(json_str)

                # Extract decision data
                decision = parsed.get("decision", "HOLD").upper()
                confidence = max(0, min(100, parsed.get("confidence", 50)))
                reasoning = parsed.get("reasoning", "AI analysis")

                # Validate decision
                if decision not in ["BUY", "SELL", "HOLD"]:
                    decision = "HOLD"

                # Apply confidence threshold
                if confidence < self.confidence_threshold:
                    logger.debug(
                        f"Signal confidence {confidence}% below threshold {self.confidence_threshold}%"
                    )
                    return None

                # Convert to SignalType
                signal_type_map = {
                    "BUY": SignalType.BUY,
                    "SELL": SignalType.SELL,
                    "HOLD": SignalType.HOLD,
                }

                signal_type = signal_type_map.get(decision, SignalType.HOLD)

                # Don't generate signals for HOLD
                if signal_type == SignalType.HOLD:
                    return None

                # Create signal
                current_price = (
                    getattr(current_data, "close", 0)
                    if hasattr(current_data, "close")
                    else current_data.get("close", 0)
                )

                return Signal(
                    symbol=symbol,
                    action=signal_type,
                    confidence=confidence,
                    entry_price=parsed.get("entry_price", current_price),
                    stop_loss=parsed.get("stop_loss"),
                    take_profit=parsed.get("take_profit"),
                    position_size=min(
                        parsed.get("position_size_suggestion", 0.05),
                        self.max_position_size,
                    ),
                    reasoning=reasoning,
                    metadata={
                        "ai_model": self.ollama_model,
                        "response_time": response.response_time,
                        "key_risks": parsed.get("key_risks", []),
                    },
                )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"JSON parsing failed: {e}, attempting text parsing")

        # Fallback: parse from text
        return self._parse_text_response(response.content, symbol, current_data)

    def _parse_text_response(
        self, content: str, symbol: str, current_data
    ) -> Optional[Signal]:
        """Parse text response when JSON fails."""
        content_lower = content.lower()

        # Determine decision from text
        if "buy" in content_lower and content_lower.index("buy") < 200:
            signal_type = SignalType.BUY
        elif "sell" in content_lower and content_lower.index("sell") < 200:
            signal_type = SignalType.SELL
        else:
            return None  # No clear signal

        # Extract confidence if mentioned
        confidence = 60  # Default
        confidence_keywords = ["confident", "confidence", "certain", "sure"]
        for keyword in confidence_keywords:
            if keyword in content_lower:
                # Try to find numbers near confidence indicators
                import re

                pattern = rf"{keyword}[^\d]*(\d+)"
                match = re.search(pattern, content_lower)
                if match:
                    confidence = min(100, max(0, int(match.group(1))))
                    break

        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            return None

        current_price = (
            getattr(current_data, "close", 0)
            if hasattr(current_data, "close")
            else current_data.get("close", 0)
        )

        return Signal(
            symbol=symbol,
            action=signal_type,
            confidence=confidence,
            entry_price=current_price,
            position_size=0.05,  # Conservative default
            reasoning=f"Text analysis: {content[:100]}...",
            metadata={"ai_model": self.ollama_model, "parsed_from_text": True},
        )

    async def cleanup(self):
        """Clean up resources."""
        if self.ollama_client:
            await self.ollama_client.close()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        success_rate = (self.successful_calls / max(1, self.total_calls)) * 100
        avg_response_time = self.total_response_time / max(1, self.total_calls)

        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "total_cost": self.total_cost,
            "ai_model": self.ollama_model,
            "ai_backend": "ollama",
            "confidence_threshold": self.confidence_threshold,
        }

    def __str__(self):
        return (
            f"OllamaAIStrategyAdapter(model={self.ollama_model}, url={self.ollama_url})"
        )
