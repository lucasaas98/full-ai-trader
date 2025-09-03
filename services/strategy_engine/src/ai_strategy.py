"""
AI Strategy Engine Module

This module implements an AI-powered trading strategy using Anthropic's Claude models
for market analysis and trading decisions.
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import yaml
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.append(os.path.join(os.path.dirname(__file__), "../../data_collector/src"))

from base_strategy import BaseStrategy, Signal, StrategyConfig
from data_collector.src.redis_client import RedisClient
from data_collector.src.twelvedata_client import TwelveDataClient

from shared.models import SignalType

# Configure logging
logger = logging.getLogger(__name__)


class AIModel(Enum):
    """Available AI models with their characteristics."""

    OPUS = "claude-3-opus-20240229"
    SONNET = "claude-3-5-sonnet-20241022"
    HAIKU = "claude-3-haiku-20240307"


@dataclass
class AIResponse:
    """Structure for AI model responses."""

    model: AIModel
    prompt_type: str
    response: Dict[str, Any]
    confidence: float
    tokens_used: int
    cost: float
    timestamp: datetime
    cache_hit: bool = False


@dataclass
class AIDecision:
    """Consolidated AI trading decision."""

    action: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: float
    risk_reward_ratio: Optional[float]
    reasoning: str
    key_risks: List[str]
    timeframe: str
    consensus_details: Dict[str, Any]


@dataclass
class MarketContext:
    """Market regime and context information."""

    regime: str
    strength: float
    risk_level: str
    position_size_multiplier: float
    confidence_threshold_adjustment: float
    sectors_to_focus: List[str]
    timestamp: datetime


class AnthropicClient:
    """Manages Anthropic API interactions with rate limiting and cost tracking."""

    def __init__(self, api_key: str, config: Dict[str, Any]):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key
            config: Configuration dictionary
        """
        self.client = AsyncAnthropic(api_key=api_key)
        self.config = config
        self.cost_tracker = CostTracker(config.get("cost_management", {}))
        self.rate_limiter = RateLimiter()
        self.cache = ResponseCache(
            ttl=config.get("cost_management", {}).get("cache_ttl_seconds", 300)
        )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def query(
        self,
        prompt: str,
        model: AIModel,
        max_tokens: int = 2000,
        temperature: float = 0.3,
        use_cache: bool = True,
    ) -> AIResponse:
        """
        Query the AI model with retry logic.

        Args:
            prompt: The prompt to send
            model: Which AI model to use
            max_tokens: Maximum tokens in response
            temperature: Model temperature
            use_cache: Whether to use cached responses

        Returns:
            AIResponse object
        """
        # Check cache first
        if use_cache:
            cached = await self.cache.get(prompt, model)
            if cached:
                logger.debug(f"Cache hit for {model.value}")
                return cached

        # Check cost limits
        if not await self.cost_tracker.can_proceed(model):
            raise Exception(f"Cost limit exceeded for {model.value}")

        # Apply rate limiting
        await self.rate_limiter.acquire(model)

        try:
            # Make API call
            messages_api = getattr(self.client, "messages", None)
            if messages_api is None:
                raise Exception("Anthropic client not properly initialized")
            response = await messages_api.create(
                model=model.value,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            # Calculate cost
            cost = self._calculate_cost(
                model, response.usage.input_tokens, response.usage.output_tokens
            )

            # Track cost
            await self.cost_tracker.record(cost)

            # Parse JSON response
            try:
                parsed_response = json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {content}")
                parsed_response = {"raw_response": content}

            ai_response = AIResponse(
                model=model,
                prompt_type="custom",
                response=parsed_response,
                confidence=parsed_response.get("confidence", 50),
                tokens_used=tokens_used,
                cost=cost,
                timestamp=datetime.now(),
            )

            # Cache the response
            if use_cache:
                await self.cache.set(prompt, model, ai_response)

            return ai_response

        except Exception as e:
            logger.error(f"Error querying {model.value}: {e}")
            raise

    def _calculate_cost(
        self, model: AIModel, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate API cost based on token usage."""
        model_costs = self.config.get("models", {}).get(model.value, {})
        input_cost = (
            model_costs.get("cost_per_million_input_tokens", 0)
            * input_tokens
            / 1_000_000
        )
        output_cost = (
            model_costs.get("cost_per_million_output_tokens", 0)
            * output_tokens
            / 1_000_000
        )
        return input_cost + output_cost


class CostTracker:
    """Tracks and manages API costs."""

    def __init__(self, config: Dict[str, Any]):
        self.daily_limit = config.get("daily_limit_usd", 5.0)
        self.monthly_limit = config.get("monthly_limit_usd", 100.0)
        self.daily_costs = 0.0
        self.monthly_costs = 0.0
        self.last_reset_day = datetime.now().date()
        self.last_reset_month = datetime.now().month

    async def can_proceed(self, model: AIModel) -> bool:
        """Check if we can proceed with the API call based on cost limits."""
        self._check_reset()

        # Estimate cost (rough estimate)
        estimated_cost = 0.10 if model == AIModel.OPUS else 0.02

        if self.daily_costs + estimated_cost > self.daily_limit:
            logger.warning(
                f"Daily cost limit would be exceeded: ${self.daily_costs:.2f} + ${estimated_cost:.2f}"
            )
            return False

        if self.monthly_costs + estimated_cost > self.monthly_limit:
            logger.warning(
                f"Monthly cost limit would be exceeded: ${self.monthly_costs:.2f} + ${estimated_cost:.2f}"
            )
            return False

        return True

    async def record(self, cost: float):
        """Record actual cost."""
        self._check_reset()
        self.daily_costs += cost
        self.monthly_costs += cost
        logger.debug(
            f"Cost recorded: ${cost:.4f} (Daily: ${self.daily_costs:.2f}, Monthly: ${self.monthly_costs:.2f})"
        )

    def _check_reset(self):
        """Check if we need to reset daily or monthly counters."""
        current_date = datetime.now().date()
        current_month = datetime.now().month

        if current_date != self.last_reset_day:
            self.daily_costs = 0.0
            self.last_reset_day = current_date

        if current_month != self.last_reset_month:
            self.monthly_costs = 0.0
            self.last_reset_month = current_month


class RateLimiter:
    """Manages API rate limiting."""

    def __init__(self):
        self.last_request_time = {}
        self.min_delay = {
            AIModel.OPUS: 1.0,  # 1 second between requests
            AIModel.SONNET: 0.5,  # 0.5 seconds between requests
            AIModel.HAIKU: 0.2,  # 0.2 seconds between requests
        }

    async def acquire(self, model: AIModel):
        """Wait if necessary to respect rate limits."""
        if model in self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time[model]).total_seconds()
            if elapsed < self.min_delay[model]:
                await asyncio.sleep(self.min_delay[model] - elapsed)

        self.last_request_time[model] = datetime.now()


class ResponseCache:
    """Caches AI responses to reduce API calls."""

    def __init__(self, ttl: int = 300):
        self.cache = {}
        self.ttl = ttl

    def _get_cache_key(self, prompt: str, model: AIModel) -> str:
        """Generate cache key from prompt and model."""
        content = f"{model.value}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def get(self, prompt: str, model: AIModel) -> Optional[AIResponse]:
        """Get cached response if available and not expired."""
        key = self._get_cache_key(prompt, model)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.ttl:
                response.cache_hit = True
                return response
            else:
                del self.cache[key]
        return None

    async def set(self, prompt: str, model: AIModel, response: AIResponse):
        """Cache a response."""
        key = self._get_cache_key(prompt, model)
        self.cache[key] = (response, datetime.now())

        # Clean old entries
        await self._cleanup()

    async def _cleanup(self):
        """Remove expired cache entries."""
        current_time = datetime.now()
        expired_keys = []

        for key, (_, timestamp) in self.cache.items():
            if (current_time - timestamp).total_seconds() > self.ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]


class DataContextBuilder:
    """Prepares and formats data for AI analysis."""

    @staticmethod
    def build_master_context(
        ticker: str,
        data: pl.DataFrame,
        finviz_data: Optional[Dict],
        market_data: Optional[Dict],
    ) -> Dict[str, Any]:
        """
        Build context for master analyst prompt.

        Args:
            ticker: Stock ticker
            data: Price and volume data
            finviz_data: Fundamental data from FinViz
            market_data: Market regime data

        Returns:
            Context dictionary for prompt formatting
        """
        # Get latest price data
        latest = data.tail(1)
        current_price = float(latest["close"].item())

        # Calculate price changes
        prev_close = float(data["close"].item(-2)) if len(data) > 1 else current_price
        daily_change = (
            ((current_price - prev_close) / prev_close) * 100 if prev_close != 0 else 0
        )

        # Calculate technical indicators
        rsi = DataContextBuilder._calculate_rsi(data)
        macd_signal, macd_histogram = DataContextBuilder._calculate_macd(data)
        sma20_val = data["close"].tail(20).mean()
        sma50_val = data["close"].tail(50).mean() if len(data) >= 50 else sma20_val
        sma20 = float(sma20_val) if sma20_val is not None else current_price
        sma50 = float(sma50_val) if sma50_val is not None else sma20

        # Bollinger Bands position
        bb_upper, bb_lower = DataContextBuilder._calculate_bollinger_bands(data)
        bb_position = (
            ((current_price - bb_lower) / (bb_upper - bb_lower)) * 100
            if bb_upper != bb_lower
            else 50
        )

        # ATR
        atr = DataContextBuilder._calculate_atr(data)
        atr_percentage = (atr / current_price) * 100

        # Support and resistance
        support, resistance = DataContextBuilder._calculate_support_resistance(data)

        # Recent candles description
        recent_candles = DataContextBuilder._describe_recent_candles(data.tail(5))

        # Pattern recognition
        patterns = DataContextBuilder._identify_patterns(data)

        # Market context
        market_context = DataContextBuilder._format_market_context(market_data)

        # Build context dictionary
        context = {
            "ticker": ticker,
            "current_price": f"{current_price:.2f}",
            "daily_change": f"{daily_change:.2f}",
            "volume": f"{int(float(latest['volume'].item())):,}",
            "avg_volume": f"{int(float(data['volume'].tail(20).mean())):,}",
            "rsi": f"{rsi:.1f}",
            "macd_signal": f"{macd_signal:.3f}",
            "macd_histogram": f"{macd_histogram:.3f}",
            "price_vs_sma20": f"{((current_price - sma20) / sma20 * 100):.1f}",
            "price_vs_sma50": f"{((current_price - sma50) / sma50 * 100):.1f}",
            "sma20_vs_sma50": f"{((sma20 - sma50) / sma50 * 100):.1f}",
            "bb_position": f"{bb_position:.1f}",
            "atr": f"{atr:.2f}",
            "atr_percentage": f"{atr_percentage:.1f}",
            "support_level": f"{support:.2f}",
            "resistance_level": f"{resistance:.2f}",
            "recent_candles": recent_candles,
            "identified_patterns": patterns,
            "market_context": market_context,
        }

        # Add fundamental data if available
        if finviz_data:
            context.update(
                {
                    "market_cap": finviz_data.get("Market Cap", "N/A"),
                    "pe_ratio": finviz_data.get("P/E", "N/A"),
                    "sector": finviz_data.get("Sector", "N/A"),
                    "sector_performance": "0",  # Would need sector performance data
                    "float_shares": finviz_data.get("Shs Float", "N/A"),
                    "short_interest": finviz_data.get("Short Float", "N/A"),
                }
            )
        else:
            context.update(
                {
                    "market_cap": "N/A",
                    "pe_ratio": "N/A",
                    "sector": "N/A",
                    "sector_performance": "0",
                    "float_shares": "N/A",
                    "short_interest": "N/A",
                }
            )

        return context

    @staticmethod
    def _calculate_rsi(data: pl.DataFrame, period: int = 14) -> float:
        """Calculate RSI."""
        if len(data) < period + 1:
            return 50.0

        prices = data["close"].to_numpy()
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    @staticmethod
    def _calculate_macd(data: pl.DataFrame) -> Tuple[float, float]:
        """Calculate MACD signal and histogram."""
        if len(data) < 26:
            return 0.0, 0.0

        prices = data["close"].to_numpy()

        # Calculate EMAs
        ema12 = DataContextBuilder._calculate_ema(prices, 12)
        ema26 = DataContextBuilder._calculate_ema(prices, 26)

        macd_line = ema12 - ema26
        signal_line = DataContextBuilder._calculate_ema(np.array([macd_line]), 9)
        histogram = macd_line - signal_line

        return signal_line, histogram

    @staticmethod
    def _calculate_ema(prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return float(np.mean(prices))

        alpha = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema

        return ema

    @staticmethod
    def _calculate_bollinger_bands(
        data: pl.DataFrame, period: int = 20
    ) -> Tuple[float, float]:
        """Calculate Bollinger Bands."""
        if len(data) < period:
            latest_price = float(data["close"][-1])
            return latest_price * 1.02, latest_price * 0.98

        prices = data["close"].tail(period).to_numpy()
        sma = np.mean(prices)
        std = np.std(prices)

        upper = sma + (2 * std)
        lower = sma - (2 * std)

        return float(upper), float(lower)

    @staticmethod
    def _calculate_atr(data: pl.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(data) < period + 1:
            return float(data["high"][-1] - data["low"][-1])

        high = data["high"].to_numpy()
        low = data["low"].to_numpy()
        close = data["close"].to_numpy()

        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        )

        return float(np.mean(tr[-period:]))

    @staticmethod
    def _calculate_support_resistance(data: pl.DataFrame) -> Tuple[float, float]:
        """Calculate support and resistance levels."""
        if len(data) < 20:
            latest_price = float(data["close"][-1])
            return latest_price * 0.98, latest_price * 1.02

        # Simple approach: use recent lows and highs
        recent_data = data.tail(20)
        latest_price = float(data["close"].item(-1))
        low_min = recent_data["low"].min()
        high_max = recent_data["high"].max()
        support = float(low_min) if low_min is not None else latest_price * 0.98
        resistance = float(high_max) if high_max is not None else latest_price * 1.02

        return support, resistance

    @staticmethod
    def _describe_recent_candles(data: pl.DataFrame) -> str:
        """Describe recent price action in text."""
        descriptions = []

        for i in range(len(data)):
            row = data[i]
            open_price = float(row["open"].item())
            close_price = float(row["close"].item())
            high = float(row["high"].item())
            low = float(row["low"].item())

            # Determine candle type
            body_size = abs(close_price - open_price)
            total_range = high - low

            if total_range > 0:
                body_ratio = body_size / total_range
            else:
                body_ratio = 0

            if close_price > open_price:
                candle_type = "bullish"
            elif close_price < open_price:
                candle_type = "bearish"
            else:
                candle_type = "doji"

            if body_ratio < 0.3:
                candle_type = f"{candle_type} doji"
            elif body_ratio > 0.7:
                candle_type = f"strong {candle_type}"

            descriptions.append(
                f"Candle {i+1}: {candle_type} (O:{open_price:.2f} H:{high:.2f} L:{low:.2f} C:{close_price:.2f})"
            )

        return "; ".join(descriptions)

    @staticmethod
    def _identify_patterns(data: pl.DataFrame) -> str:
        """Identify common chart patterns."""
        patterns = []

        if len(data) < 20:
            return "Insufficient data for pattern recognition"

        # Check for trending
        sma20_val = data["close"].tail(20).mean()
        sma50_val = data["close"].tail(50).mean() if len(data) >= 50 else sma20_val
        sma20 = float(sma20_val) if sma20_val is not None else 100.0
        sma50 = float(sma50_val) if sma50_val is not None else sma20

        if sma20 > sma50 * 1.02:
            patterns.append("Uptrend (SMA20 > SMA50)")
        elif sma20 < sma50 * 0.98:
            patterns.append("Downtrend (SMA20 < SMA50)")

        # Check for breakout
        high_max = data["high"].tail(20).max()
        recent_high = float(high_max) if high_max is not None else 100.0
        close_val = data["close"].item(-1)
        current_price = float(close_val) if close_val is not None else 100.0

        if current_price > recent_high * 0.98:
            patterns.append("Near resistance breakout")

        # Check for support bounce
        low_min = data["low"].tail(20).min()
        recent_low = float(low_min) if low_min is not None else current_price * 0.98
        if current_price < recent_low * 1.02:
            patterns.append("Near support level")

        # Volume spike
        volume_mean = data["volume"].tail(20).mean()
        volume_last = data["volume"].item(-1)
        avg_volume = float(volume_mean) if volume_mean is not None else 1000000.0
        latest_volume = float(volume_last) if volume_last is not None else avg_volume

        if latest_volume > avg_volume * 1.5:
            patterns.append("Volume spike detected")

        return "; ".join(patterns) if patterns else "No significant patterns detected"

    @staticmethod
    def _format_market_context(market_data: Optional[Dict]) -> str:
        """Format market context for prompt."""
        if not market_data:
            return "Market data unavailable"

        context_parts = []

        if "spy_change" in market_data:
            context_parts.append(f"SPY: {market_data['spy_change']:.2f}%")

        if "vix_level" in market_data:
            context_parts.append(f"VIX: {market_data['vix_level']:.1f}")

        if "market_regime" in market_data:
            context_parts.append(f"Regime: {market_data['market_regime']}")

        return (
            "; ".join(context_parts) if context_parts else "Market context unavailable"
        )


class ConsensusEngine:
    """Manages multiple AI queries and consensus building."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_instances = config.get("consensus", {}).get("min_ai_instances", 3)
        self.majority_threshold = config.get("consensus", {}).get(
            "majority_threshold", 0.6
        )
        self.confidence_weights = config.get("consensus", {}).get(
            "confidence_weights", {}
        )

    async def build_consensus(self, responses: List[AIResponse]) -> AIDecision:
        """
        Build consensus from multiple AI responses.

        Args:
            responses: List of AI responses

        Returns:
            Consolidated AIDecision
        """
        if len(responses) < self.min_instances:
            logger.warning(
                f"Only {len(responses)} responses, minimum is {self.min_instances}"
            )

        # Extract decisions and confidences
        decisions = []
        for response in responses:
            weight = self.confidence_weights.get(response.model.value, 1.0)
            decisions.append(
                {
                    "action": response.response.get("decision", "HOLD"),
                    "confidence": response.confidence * weight,
                    "model": response.model,
                    "response": response.response,
                }
            )

        # Count votes
        action_votes = {}
        total_weight = 0

        for decision in decisions:
            action = decision["action"]
            confidence = decision["confidence"]

            if action not in action_votes:
                action_votes[action] = 0

            action_votes[action] += confidence
            total_weight += confidence

        # Determine majority action
        if total_weight == 0:
            final_action = "HOLD"
            consensus_confidence = 0
        else:
            final_action = max(action_votes.items(), key=lambda x: x[1])[0]
            consensus_confidence = (action_votes[final_action] / total_weight) * 100

        # Average the numerical values from responses that agree with majority
        entry_prices = []
        stop_losses = []
        take_profits = []
        position_sizes = []
        risk_rewards = []
        reasonings = []
        all_risks = []

        for decision in decisions:
            if decision["action"] == final_action:
                resp = decision["response"]

                if resp.get("entry_price"):
                    entry_prices.append(float(resp["entry_price"]))
                if resp.get("stop_loss"):
                    stop_losses.append(float(resp["stop_loss"]))
                if resp.get("take_profit"):
                    take_profits.append(float(resp["take_profit"]))
                if resp.get("position_size_suggestion"):
                    position_sizes.append(float(resp["position_size_suggestion"]))
                if resp.get("risk_reward_ratio"):
                    risk_rewards.append(float(resp["risk_reward_ratio"]))
                if resp.get("reasoning"):
                    reasonings.append(resp["reasoning"])
                if resp.get("key_risks"):
                    all_risks.extend(resp["key_risks"])

        # Create final decision
        return AIDecision(
            action=final_action,
            confidence=consensus_confidence,
            entry_price=float(np.mean(entry_prices)) if entry_prices else None,
            stop_loss=float(np.mean(stop_losses)) if stop_losses else None,
            take_profit=float(np.mean(take_profits)) if take_profits else None,
            position_size=float(np.mean(position_sizes)) if position_sizes else 0.01,
            risk_reward_ratio=float(np.mean(risk_rewards)) if risk_rewards else None,
            reasoning=(
                "; ".join(reasonings[:3]) if reasonings else "No reasoning provided"
            ),
            key_risks=list(set(all_risks))[:5],  # Unique risks, max 5
            timeframe="day_trade",  # Default, could be determined from responses
            consensus_details={
                "total_responses": len(responses),
                "action_votes": action_votes,
                "models_used": [r.model.value for r in responses],
                "agreement_rate": consensus_confidence,
            },
        )


class AIStrategyEngine(BaseStrategy):
    """
    AI-powered trading strategy using Anthropic's Claude models.

    This strategy uses multiple AI models to analyze market data and make
    trading decisions through a consensus mechanism.
    """

    def __init__(self, config: StrategyConfig):
        """Initialize AI Strategy Engine."""
        super().__init__(config)

        # Load prompts configuration
        config_path = (
            Path(__file__).parent.parent.parent.parent
            / "config"
            / "ai_strategy"
            / "prompts.yaml"
        )
        try:
            with open(config_path, "r") as f:
                prompts_config = yaml.safe_load(f)
                self.prompts_config = prompts_config if prompts_config else {}
        except FileNotFoundError:
            self.prompts_config = {}

        # Initialize components
        api_key = (
            config.parameters.get("anthropic_api_key") if config.parameters else None
        )
        if not api_key:
            raise ValueError("Anthropic API key is required for AI Strategy")

        self.anthropic_client = AnthropicClient(api_key, self.prompts_config)
        self.context_builder = DataContextBuilder()
        self.consensus_engine = ConsensusEngine(self.prompts_config)

        # Strategy state
        self.market_context: Optional[MarketContext] = None
        self.last_market_update = datetime.now() - timedelta(hours=1)
        self.decision_history: List[AIDecision] = []

        # Performance tracking
        self.ai_performance = {
            "total_decisions": 0,
            "correct_decisions": 0,
            "total_cost": 0.0,
            "decisions_by_model": {},
        }

    def _setup_indicators(self) -> None:
        """Setup any required indicators."""
        # AI strategy doesn't use traditional indicators
        # but we initialize tracking structures
        self.signal_cache = {}
        self.cache_ttl = 300  # 5 minutes

    async def analyze(self, symbol: str, data: pl.DataFrame) -> Signal:
        """
        Analyze market data using AI models.

        Args:
            symbol: Trading symbol
            data: Historical price data

        Returns:
            Trading signal
        """
        try:
            # Update market context if needed
            if (
                datetime.now() - self.last_market_update
            ).total_seconds() > 1800:  # 30 minutes
                await self._update_market_context()

            # Check cache first
            cache_key = f"{symbol}:{data['close'][-1]}:{data['volume'][-1]}"
            if cache_key in self.signal_cache:
                cached_signal, cache_time = self.signal_cache[cache_key]
                if (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                    logger.debug(f"Using cached signal for {symbol}")
                    return cached_signal

            # Get FinViz data if available (would come from Redis in production)
            finviz_data = await self._get_finviz_data(symbol)

            # Get market data for context
            market_data = await self._get_market_data()

            # Build context for AI
            context = self.context_builder.build_master_context(
                symbol, data, finviz_data, market_data
            )

            # Query multiple AI models
            responses = await self._query_multiple_models(context, symbol)

            # Build consensus
            decision = await self.consensus_engine.build_consensus(responses)

            # Store decision
            self.decision_history.append(decision)

            # Update performance tracking
            self.ai_performance["total_decisions"] += 1
            self.ai_performance["total_cost"] += sum(r.cost for r in responses)

            # Convert to Signal (placeholder method)
            from decimal import Decimal

            # Convert action string to SignalType
            action_map = {
                "BUY": SignalType.BUY,
                "SELL": SignalType.SELL,
                "HOLD": SignalType.HOLD,
            }
            signal_action = action_map.get(decision.action, SignalType.HOLD)

            signal = Signal(
                action=signal_action,
                confidence=decision.confidence,
                entry_price=(
                    Decimal(str(decision.entry_price)) if decision.entry_price else None
                ),
                stop_loss=(
                    Decimal(str(decision.stop_loss)) if decision.stop_loss else None
                ),
                take_profit=(
                    Decimal(str(decision.take_profit)) if decision.take_profit else None
                ),
                position_size=decision.position_size,
                metadata={
                    "ai_decision": decision,
                    "reasoning": decision.reasoning,
                    "key_risks": decision.key_risks,
                    "consensus_details": decision.consensus_details,
                },
            )

            # Cache the signal
            self.signal_cache[cache_key] = (signal, datetime.now())

            return signal

        except Exception as e:
            logger.error(f"Error in AI analysis for {symbol}: {e}")
            # Return neutral signal on error
            from decimal import Decimal

            return Signal(
                action=SignalType.HOLD,
                confidence=0,
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                position_size=0.01,
                metadata={"error": str(e)},
            )

    async def _update_market_context(self):
        """Update market regime context."""
        try:
            # Fetch real market data for SPY (S&P 500), QQQ (NASDAQ), and VIX
            market_data = await self._get_broad_market_data()

            # Analyze market conditions
            spy_change = market_data.get("SPY", {}).get("change_percent", 0)
            vix_level = market_data.get("VIX", {}).get("price", 20)

            # Determine market regime based on indicators
            if spy_change > 1.0 and vix_level < 20:
                regime = "bullish"
                strength = min(90.0, 50.0 + spy_change * 10)
                risk_level = "low"
                position_multiplier = 1.2
                confidence_adjustment = 0.1
            elif spy_change < -1.0 or vix_level > 30:
                regime = "bearish"
                strength = max(10.0, 50.0 + spy_change * 10)
                risk_level = "high"
                position_multiplier = 0.7
                confidence_adjustment = -0.1
            else:
                regime = "neutral"
                strength = 50.0 + spy_change * 5
                risk_level = "medium"
                position_multiplier = 1.0
                confidence_adjustment = 0.0

            # Determine sectors to focus on based on market conditions
            if regime == "bullish":
                sectors = ["Technology", "Consumer Discretionary", "Growth"]
            elif regime == "bearish":
                sectors = ["Utilities", "Consumer Staples", "Healthcare"]
            else:
                sectors = ["Technology", "Healthcare", "Financial"]

            self.market_context = MarketContext(
                regime=regime,
                strength=strength,
                risk_level=risk_level,
                position_size_multiplier=position_multiplier,
                confidence_threshold_adjustment=confidence_adjustment,
                sectors_to_focus=sectors,
                timestamp=datetime.now(),
            )
            self.last_market_update = datetime.now()

        except Exception as e:
            logger.error(f"Error updating market context: {e}")
            # Fallback to neutral market context
            self.market_context = MarketContext(
                regime="neutral",
                strength=50.0,
                risk_level="medium",
                position_size_multiplier=1.0,
                confidence_threshold_adjustment=0.0,
                sectors_to_focus=["Technology", "Healthcare"],
                timestamp=datetime.now(),
            )
            self.last_market_update = datetime.now()

    async def _get_finviz_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch FinViz data from Redis cache."""
        try:
            redis_client = RedisClient()
            await redis_client.connect()

            # Try to get cached FinViz data
            finviz_key = f"finviz:{symbol}"
            cached_data = await redis_client.get_cached_market_data(finviz_key)

            await redis_client.disconnect()

            if cached_data:
                logger.debug(f"Retrieved FinViz data for {symbol} from cache")
                return cached_data
            else:
                logger.debug(f"No cached FinViz data found for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Error fetching FinViz data for {symbol}: {e}")
            return None

    async def _get_market_data(self) -> Dict[str, Any]:
        """Fetch general market data for context."""
        try:
            market_data = {}

            client = TwelveDataClient()

            # Get key market indicators
            symbols = [
                "SPY",
                "QQQ",
                "VIX",
                "DXY",
            ]  # S&P 500, NASDAQ, VIX, Dollar Index

            for symbol in symbols:
                try:
                    quote_data = await client.get_real_time_price(symbol)
                    if quote_data:
                        market_data[symbol] = {
                            "price": quote_data.get("price", 0),
                            "change_percent": quote_data.get("percent_change", 0),
                        }
                except Exception as e:
                    logger.warning(f"Could not fetch {symbol} data: {e}")

            return market_data

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}

    async def _get_broad_market_data(self) -> Dict[str, Any]:
        """Fetch broad market data for market regime analysis."""
        try:
            market_data = {}

            client = TwelveDataClient()

            # Get major indices and volatility measures
            symbols = {
                "SPY": "S&P 500 ETF",
                "QQQ": "NASDAQ ETF",
                "VIX": "Volatility Index",
                "TLT": "Long-term Treasury",
                "GLD": "Gold ETF",
            }

            for symbol, description in symbols.items():
                try:
                    quote_data = await client.get_real_time_price(symbol)
                    if quote_data:
                        market_data[symbol] = {
                            "price": float(quote_data.get("price", 0)),
                            "change_percent": float(
                                quote_data.get("percent_change", 0)
                            ),
                            "description": description,
                        }
                    else:
                        # Fallback data if real data not available
                        market_data[symbol] = {
                            "price": 400.0 if symbol == "SPY" else 300.0,
                            "change_percent": 0.0,
                            "description": description,
                        }
                except Exception as e:
                    logger.warning(f"Could not fetch {symbol} data: {e}")
                    # Set neutral fallback
                    market_data[symbol] = {
                        "price": 400.0 if symbol == "SPY" else 300.0,
                        "change_percent": 0.0,
                        "description": description,
                    }

            return market_data

        except Exception as e:
            logger.error(f"Error fetching broad market data: {e}")
            return {
                "SPY": {
                    "price": 400.0,
                    "change_percent": 0.0,
                    "description": "S&P 500 ETF",
                },
                "VIX": {
                    "price": 20.0,
                    "change_percent": 0.0,
                    "description": "Volatility Index",
                },
            }

    async def _query_multiple_models(
        self, context: Dict[str, Any], symbol: str
    ) -> List[AIResponse]:
        """Query multiple AI models and return their responses."""
        try:
            responses = []

            # Define models to query based on configuration
            models_to_query = [
                AIModel.SONNET,  # Primary model - good balance of speed and capability
                AIModel.HAIKU,  # Fast model for quick analysis
            ]

            # Add OPUS for high-value trades or complex situations
            current_price = float(context.get("current_price", 0))
            if current_price > 100 or abs(float(context.get("daily_change", 0))) > 5:
                models_to_query.append(AIModel.OPUS)

            # Query each model
            for model in models_to_query:
                try:
                    # Build prompt for this model
                    prompt = f"""
Analyze the following stock data for {symbol} and provide a trading decision.

Current Data:
- Price: ${context.get('current_price', 'N/A')}
- Daily Change: {context.get('daily_change', 'N/A')}%
- Volume: {context.get('volume', 'N/A')}
- RSI: {context.get('rsi', 'N/A')}
- MACD Signal: {context.get('macd_signal', 'N/A')}
- Price vs SMA20: {context.get('price_vs_sma20', 'N/A')}%
- Market Context: {context.get('market_context', 'N/A')}

Provide your analysis in the following JSON format:
{{
    "decision": "BUY/SELL/HOLD",
    "confidence": 0-100,
    "reasoning": "Brief explanation",
    "entry_price": float or null,
    "stop_loss": float or null,
    "take_profit": float or null,
    "position_size_suggestion": 0.01-0.1,
    "risk_reward_ratio": float or null,
    "key_risks": ["risk1", "risk2"]
}}
"""

                    # Query the model
                    response = await self.anthropic_client.query(
                        prompt=prompt, model=model
                    )

                    if response:
                        responses.append(response)
                        logger.debug(f"Got response from {model.value} for {symbol}")

                except Exception as e:
                    logger.error(f"Error querying {model.value} for {symbol}: {e}")
                    continue

            if not responses:
                logger.warning(f"No AI responses received for {symbol}")
                # Create a fallback neutral response
                fallback_response = AIResponse(
                    model=AIModel.HAIKU,
                    prompt_type="fallback",
                    response={
                        "decision": "HOLD",
                        "confidence": 50,
                        "reasoning": "No AI response available, defaulting to HOLD",
                        "entry_price": current_price,
                        "position_size_suggestion": 0.01,
                    },
                    confidence=50.0,
                    tokens_used=0,
                    cost=0.0,
                    timestamp=datetime.now(),
                    cache_hit=False,
                )
                responses.append(fallback_response)

            return responses

        except Exception as e:
            logger.error(f"Error in multi-model query for {symbol}: {e}")
            # Return empty list, consensus engine will handle
            return []
