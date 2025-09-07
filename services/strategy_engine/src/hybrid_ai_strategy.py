"""
Hybrid AI Strategy Engine that supports both Anthropic and Ollama.

This module provides a unified interface for AI-powered trading strategies,
allowing seamless switching between cloud-based (Anthropic) and local (Ollama)
AI models for testing and production use.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .ai_strategy import (
    AIDecision,
    AIModel,
    AIResponse,
    AnthropicClient,
    BaseStrategy,
    ConsensusEngine,
    DataContextBuilder,
    MarketContext,
    StrategyConfig,
)
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class HybridAIClient:
    """Unified client that can use either Anthropic or Ollama."""

    def __init__(self, config: Dict[str, Any], use_ollama: bool = False):
        self.client: Union[OllamaClient, AnthropicClient]
        """
        Initialize hybrid AI client.

        Args:
            config: Configuration dictionary
            use_ollama: If True, use Ollama; if False, use Anthropic
        """
        self.use_ollama = use_ollama
        self.config = config

        if use_ollama:
            ollama_url = config.get("ollama_url", "http://192.168.1.133:11434")
            ollama_model = config.get("ollama_model", "llama3.1:latest")
            self.client = OllamaClient(ollama_url, ollama_model)
            self.model_type = "ollama"
        else:
            api_key = config.get("anthropic_api_key")
            if not api_key:
                raise ValueError("Anthropic API key is required when not using Ollama")
            self.client = AnthropicClient(api_key, config)
            self.model_type = "anthropic"

    async def query(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> AIResponse:
        """
        Query the AI client (either Anthropic or Ollama).

        Args:
            prompt: The prompt to send
            model: Model name (ignored for Ollama)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            AIResponse object
        """
        try:
            if self.use_ollama:
                ollama_client = self.client  # Type: OllamaClient
                assert isinstance(ollama_client, OllamaClient)
                response = await ollama_client.query(
                    prompt, max_tokens=max_tokens, temperature=temperature
                )

                # Convert OllamaResponse to AIResponse format
                return AIResponse(
                    model=AIModel.HAIKU,  # Use as placeholder
                    prompt_type="trading_decision",
                    response={
                        "content": getattr(response, "response", {}).get("content", "")
                    },
                    confidence=0.8,
                    tokens_used=getattr(response, "tokens_used", 0),
                    cost=0.0,  # Local models are free
                    timestamp=datetime.now(),
                )
            else:
                anthropic_client = self.client  # Type: AnthropicClient
                assert isinstance(anthropic_client, AnthropicClient)
                ai_model = AIModel.HAIKU if not model else AIModel(model)
                anthropic_response = await anthropic_client.query(
                    prompt,
                    ai_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # Ensure we return AIResponse type consistently
                return anthropic_response

        except Exception as e:
            logger.error(f"AI query failed: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if the AI service is healthy."""
        if self.use_ollama:
            if hasattr(self.client, "health_check"):
                return await self.client.health_check()
            return True
        else:
            # Simple test query for Anthropic
            try:
                test_response = await self.query("Hello", max_tokens=10)
                return len(test_response.response.get("content", "")) > 0
            except Exception:
                return False

    async def close(self):
        """Close the client connection."""
        if hasattr(self.client, "close"):
            await self.client.close()


class HybridAIStrategyEngine(BaseStrategy):
    """
    Hybrid AI-powered trading strategy supporting both Anthropic and Ollama.

    This strategy can seamlessly switch between cloud-based and local AI models,
    making it ideal for both production use and cost-effective testing.
    """

    def __init__(self, config: StrategyConfig, use_ollama: Optional[bool] = None):
        """
        Initialize Hybrid AI Strategy Engine.

        Args:
            config: Strategy configuration
            use_ollama: If True, use Ollama; if False, use Anthropic; if None, auto-detect
        """
        super().__init__(config)

        # Auto-detect AI backend if not specified
        if use_ollama is None:
            use_ollama = self._should_use_ollama()

        self.use_ollama = use_ollama
        logger.info(
            f"Initializing Hybrid AI Strategy with {'Ollama' if use_ollama else 'Anthropic'}"
        )

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
            logger.warning("Prompts config not found, using default configuration")
            self.prompts_config = {}

        # Initialize AI client
        client_config = self._build_client_config(config)
        self.ai_client = HybridAIClient(client_config, use_ollama)

        # Initialize other components
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
            "backend_used": "ollama" if use_ollama else "anthropic",
        }

    def _should_use_ollama(self) -> bool:
        """
        Auto-detect whether to use Ollama based on environment.

        Returns:
            True if should use Ollama, False for Anthropic
        """
        # Check for testing environment
        if os.getenv("AI_TESTING_MODE", "").lower() == "true":
            return True

        # Check for Ollama URL configuration
        if os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_HOST"):
            return True

        # Check if we're running tests
        import sys

        if "pytest" in sys.modules or "unittest" in sys.modules:
            return True

        # Default to Anthropic for production
        return False

    def _build_client_config(self, config: StrategyConfig) -> Dict[str, Any]:
        """Build configuration for AI client."""
        client_config = {}

        if self.use_ollama:
            client_config.update(
                {
                    "ollama_url": os.getenv("OLLAMA_URL", "http://192.168.1.133:11434"),
                    "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
                }
            )
        else:
            client_config.update(
                {
                    "anthropic_api_key": str(
                        config.parameters.get("anthropic_api_key")
                        or os.getenv("ANTHROPIC_API_KEY")
                        or ""
                    ),
                    "cost_management": self.prompts_config.get("cost_management", {}),
                }
            )

        return client_config

    async def analyze(self, market_data: Dict[str, Any]) -> Optional[AIDecision]:
        """
        Analyze market data and make trading decision.

        Args:
            market_data: Market data dictionary

        Returns:
            AI trading decision or None
        """
        try:
            # Health check
            if not await self.ai_client.health_check():
                logger.error(
                    f"AI service ({'Ollama' if self.use_ollama else 'Anthropic'}) is not healthy"
                )
                return None

            # Update market context
            await self._update_market_context(market_data)

            if not self.market_context:
                logger.warning("No market context available for analysis")
                return None

            # Choose analysis strategy based on backend
            if self.use_ollama:
                decision = await self._analyze_with_ollama(market_data)
            else:
                decision = await self._analyze_with_anthropic(market_data)

            # Update performance tracking
            if decision:
                self._update_performance_tracking(decision)

            return decision

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return None

    async def _analyze_with_ollama(
        self, market_data: Dict[str, Any]
    ) -> Optional[AIDecision]:
        """Analyze using Ollama (simplified single-model approach)."""
        try:
            # Build context for Ollama
            context = self._build_ollama_context(market_data)

            # Get analysis
            response = await self.ai_client.query(
                self._build_ollama_prompt(context), max_tokens=1000, temperature=0.3
            )

            # Parse decision
            decision_data = self._parse_ollama_response(
                response.response.get("content", ""), market_data
            )

            if decision_data:
                return AIDecision(
                    action=decision_data["decision"],
                    confidence=decision_data["confidence"],
                    reasoning=decision_data["reasoning"],
                    entry_price=decision_data.get("entry_price"),
                    stop_loss=decision_data.get("stop_loss"),
                    take_profit=decision_data.get("take_profit"),
                    key_risks=decision_data.get("key_risks", []),
                    timeframe=decision_data.get("timeframe", "1d"),
                    consensus_details=decision_data.get("consensus_details", {}),
                    position_size=decision_data.get("position_size", 0.1),
                    risk_reward_ratio=decision_data.get("risk_reward_ratio"),
                )

            return None

        except Exception as e:
            logger.error(f"Ollama analysis failed: {e}")
            return None

    async def _analyze_with_anthropic(
        self, market_data: Dict[str, Any]
    ) -> Optional[AIDecision]:
        """Analyze using Anthropic (full consensus approach)."""
        try:
            # Use the original consensus mechanism - need to get responses first
            responses: List[AIResponse] = (
                []
            )  # TODO: Implement proper response collection
            decision = await self.consensus_engine.build_consensus(responses)
            return decision

        except Exception as e:
            logger.error(f"Anthropic analysis failed: {e}")
            return None

    def _build_ollama_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build simplified context for Ollama analysis."""
        return {
            "ticker": market_data.get("ticker", "UNKNOWN"),
            "current_price": market_data.get("price", 0),
            "daily_change": market_data.get("change_percent", 0),
            "volume": market_data.get("volume", 0),
            "technical_indicators": market_data.get("technical_indicators", {}),
            "timestamp": datetime.now().isoformat(),
        }

    def _build_ollama_prompt(self, context: Dict[str, Any]) -> str:
        """Build trading prompt for Ollama using YAML configuration."""
        # Get the master analyst prompt template from config
        master_analyst_config = self.prompts_config.get("prompts", {}).get(
            "master_analyst", {}
        )
        prompt_template = master_analyst_config.get("template", "")

        if not prompt_template:
            # Fallback to simple prompt if config not available
            return self._build_simple_prompt(context)

        # Extract context values
        ticker = context["ticker"]
        price = context["current_price"]
        change = context["daily_change"]
        volume = context["volume"]
        tech_indicators = context.get("technical_indicators", {})

        # Build comprehensive context data for template
        template_data = {
            "market_context": self._format_market_context_for_ollama(),
            "ticker": ticker,
            "current_price": price,
            "daily_change": change,
            "volume": volume,
            "avg_volume": volume * 0.8,  # Approximate
            "rsi": tech_indicators.get("rsi", 50),
            "macd_signal": tech_indicators.get("macd_signal", 0),
            "macd_histogram": tech_indicators.get("macd_histogram", 0),
            "price_vs_sma20": tech_indicators.get("price_vs_sma20", 0),
            "price_vs_sma50": tech_indicators.get("price_vs_sma50", 0),
            "sma20_vs_sma50": tech_indicators.get("sma20_vs_sma50", 0),
            "bb_position": tech_indicators.get("bb_position", 50),
            "atr": tech_indicators.get("atr", price * 0.02),
            "atr_percentage": tech_indicators.get("atr_percentage", 2.0),
            "support_level": tech_indicators.get("support", price * 0.95),
            "resistance_level": tech_indicators.get("resistance", price * 1.05),
            "market_cap": context.get("market_cap", "N/A"),
            "pe_ratio": context.get("pe_ratio", "N/A"),
            "sector": context.get("sector", "Unknown"),
            "sector_performance": context.get("sector_performance", 0),
            "float_shares": context.get("float_shares", "N/A"),
            "short_interest": context.get("short_interest", "N/A"),
            "recent_candles": self._format_recent_candles(context),
            "identified_patterns": self._identify_simple_patterns(tech_indicators),
        }

        try:
            return prompt_template.format(**template_data)
        except KeyError as e:
            logger.warning(f"Missing template variable {e}, using fallback prompt")
            return self._build_simple_prompt(context)

    def _build_simple_prompt(self, context: Dict[str, Any]) -> str:
        """Fallback simple prompt when YAML config is unavailable."""
        ticker = context["ticker"]
        price = context["current_price"]
        change = context["daily_change"]
        volume = context["volume"]
        tech_indicators = context.get("technical_indicators", {})
        rsi = tech_indicators.get("rsi", 50)

        return f"""You are a professional trading analyst. Analyze this stock data and provide a clear recommendation.

STOCK: {ticker}
Current Price: ${price}
Daily Change: {change}%
Volume: {volume:,}
RSI (14): {rsi}

Market Analysis:
- RSI indicates {'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL'} conditions
- Price momentum is {'BULLISH' if change > 1 else 'BEARISH' if change < -1 else 'SIDEWAYS'}

Provide your recommendation in JSON format:
{{
    "decision": "BUY" or "SELL" or "HOLD",
    "confidence": 0-100,
    "entry_price": {price},
    "stop_loss": price or null,
    "take_profit": price or null,
    "reasoning": "Brief explanation",
    "position_size": 0.05 to 0.2
}}

Consider risk management and only recommend high-confidence trades."""

    def _format_market_context_for_ollama(self) -> str:
        """Format market context for Ollama prompt."""
        if not self.market_context:
            return "Market context unavailable"

        context_str = f"""Market Environment:
- Market Regime: {self.market_context.regime}
- Risk Level: {self.market_context.risk_level}
- Strength: {self.market_context.strength:.2f}
- Last Update: {self.market_context.timestamp.strftime('%H:%M:%S')}"""

        return context_str

    def _format_recent_candles(self, context: Dict[str, Any]) -> str:
        """Format recent candlestick data description."""
        return f"Recent price action shows {'bullish' if context.get('daily_change', 0) > 0 else 'bearish'} momentum"

    def _identify_simple_patterns(self, tech_indicators: Dict[str, Any]) -> str:
        """Identify simple technical patterns."""
        patterns = []

        rsi = tech_indicators.get("rsi", 50)
        if rsi > 70:
            patterns.append("Overbought (RSI > 70)")
        elif rsi < 30:
            patterns.append("Oversold (RSI < 30)")

        price_vs_sma20 = tech_indicators.get("price_vs_sma20", 0)
        if price_vs_sma20 > 2:
            patterns.append("Strong uptrend (price well above SMA20)")
        elif price_vs_sma20 < -2:
            patterns.append("Strong downtrend (price well below SMA20)")

        return "; ".join(patterns) if patterns else "No significant patterns detected"

    def _parse_ollama_response(
        self, response_text: str, market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Parse Ollama response into decision data."""
        try:
            # Try to extract JSON
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                decision = json.loads(json_str)

                # Validate and clean up decision
                decision["decision"] = decision.get("decision", "HOLD").upper()
                if decision["decision"] not in ["BUY", "SELL", "HOLD"]:
                    decision["decision"] = "HOLD"

                decision["confidence"] = max(
                    0, min(100, decision.get("confidence", 50))
                )
                decision["reasoning"] = decision.get("reasoning", "AI analysis")

                return decision

            return None

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse Ollama response: {e}")
            return {
                "decision": "HOLD",
                "confidence": 0,
                "reasoning": "Failed to parse AI response",
            }

    async def _update_market_context(self, market_data: Dict[str, Any]) -> None:
        """Update market context (simplified for hybrid approach)."""
        try:
            if self.use_ollama:
                # Simplified context for Ollama
                self.market_context = MarketContext(
                    regime="neutral",
                    strength=0.5,
                    risk_level="medium",
                    position_size_multiplier=1.0,
                    confidence_threshold_adjustment=0.0,
                    sectors_to_focus=[],
                    timestamp=datetime.now(),
                )
            else:
                # Full context building for Anthropic
                dataframe = market_data.get("dataframe")
                if dataframe is None:
                    logger.warning("No dataframe provided for context building")
                    return None
                master_context = self.context_builder.build_master_context(
                    market_data.get("ticker", "SPY"),
                    dataframe,
                    market_data.get("finviz_data", None),
                    market_data,
                )
                # Convert dict to MarketContext if needed
                if isinstance(master_context, dict):
                    self.market_context = (
                        None  # TODO: Convert dict to MarketContext properly
                    )
                else:
                    self.market_context = master_context

            self.last_market_update = datetime.now()

        except Exception as e:
            logger.error(f"Failed to update market context: {e}")

    def _update_performance_tracking(self, decision: AIDecision) -> None:
        """Update performance tracking metrics."""
        current_count = self.ai_performance.get("total_decisions", 0)
        try:
            self.ai_performance["total_decisions"] = int(str(current_count)) + 1
        except (ValueError, TypeError):
            self.ai_performance["total_decisions"] = 1
        # Note: AIDecision doesn't have total_cost or models_used attributes
        # self.ai_performance["total_cost"] += decision.total_cost

        # Track decisions by action type instead
        action = decision.action.upper()
        decisions_by_model = self.ai_performance.get("decisions_by_model", {})
        if not isinstance(decisions_by_model, dict):
            decisions_by_model = {}
            self.ai_performance["decisions_by_model"] = decisions_by_model
        if action not in decisions_by_model:
            decisions_by_model[action] = 0
        decisions_by_model[action] = int(decisions_by_model.get(action, 0)) + 1

        # Keep decision history (last 100)
        self.decision_history.append(decision)
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            **self.ai_performance,
            "recent_decisions": len(self.decision_history),
            "average_confidence": (
                sum(d.confidence for d in self.decision_history[-10:])
                / min(10, len(self.decision_history))
                if self.decision_history
                else 0
            ),
            "backend_type": "ollama" if self.use_ollama else "anthropic",
        }

    async def close(self):
        """Close the strategy engine."""
        if self.ai_client:
            await self.ai_client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Configuration helper functions
def create_ollama_config() -> Dict[str, Any]:
    """Create configuration for Ollama testing."""
    return {
        "ollama_url": os.getenv("OLLAMA_URL", "http://192.168.1.133:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
        "ai_testing_mode": True,
    }


def create_anthropic_config(api_key: str) -> Dict[str, Any]:
    """Create configuration for Anthropic production."""
    return {"anthropic_api_key": api_key, "ai_testing_mode": False}


def get_recommended_test_config() -> Dict[str, Any]:
    """Get recommended configuration for testing."""
    return {
        **create_ollama_config(),
        "max_position_size": 0.1,  # Smaller positions for testing
        "enable_consensus": False,  # Simpler single-model approach
        "log_all_decisions": True,
        "test_mode": True,
    }
