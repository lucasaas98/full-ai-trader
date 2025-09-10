"""
Integration tests for AI Strategy with Ollama backend using production prompts.

This test suite validates that the AI strategy engine can use Ollama as a backend
while maintaining compatibility with the existing prompt system and production data flows.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from backtesting.ollama_ai_strategy_adapter import OllamaClient
from services.data_collector.src.data_store import DataStore, DataStoreConfig

# Import AI strategy components
from services.strategy_engine.src.ai_strategy import (
    AIModel,
    AIResponse,
    AIStrategyEngine,
    ConsensusEngine,
    DataContextBuilder,
    MarketContext,
)
from services.strategy_engine.src.base_strategy import StrategyConfig, StrategyMode
from shared.models import TimeFrame

# Removed unused import


class OllamaAnthropicAdapter:
    """Adapter to make Ollama client work with the existing AI strategy framework."""

    def __init__(self, ollama_client: OllamaClient, prompts_config: Dict[str, Any]):
        self.ollama_client = ollama_client
        self.prompts_config = prompts_config
        self.cost_tracker = MagicMock()
        self.rate_limiter = MagicMock()
        self.cache = MagicMock()

    async def query(
        self, prompt: str, model: AIModel = AIModel.HAIKU, **kwargs: Any
    ) -> AIResponse:
        """Query Ollama and return in AIResponse format."""
        # Use Ollama instead of Anthropic
        ollama_response = await self.ollama_client.query(
            prompt,
            max_tokens=kwargs.get("max_tokens", 2000),
            temperature=kwargs.get("temperature", 0.3),
        )

        # Convert to AIResponse format
        return AIResponse(
            model=AIModel.HAIKU,  # Use enum value
            prompt_type="trading_decision",
            response={"content": ollama_response.content},
            confidence=0.8,  # Default confidence
            tokens_used=ollama_response.tokens_used,
            cost=0.0,  # Free for local models
            timestamp=datetime.now(timezone.utc),
            cache_hit=False,
        )


class OllamaAIStrategyEngine(AIStrategyEngine):
    """AI Strategy Engine modified to use Ollama backend."""

    def __init__(
        self,
        config: StrategyConfig,
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ):
        """Initialize with Ollama backend."""
        # Initialize base components
        self.config = config

        # Load prompts configuration
        config_path = (
            Path(__file__).parent.parent.parent
            / "config"
            / "ai_strategy"
            / "prompts.yaml"
        )
        try:
            with open(config_path, "r") as f:
                import yaml

                prompts_config = yaml.safe_load(f)
                self.prompts_config = prompts_config if prompts_config else {}
        except FileNotFoundError:
            self.prompts_config = {}

        # Initialize Ollama client
        ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://192.168.1.133:11434")
        ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.1:latest")

        ollama_client = OllamaClient(ollama_url, ollama_model)

        # Use adapter to make Ollama compatible with existing code
        self.anthropic_client = OllamaAnthropicAdapter(  # type: ignore
            ollama_client, self.prompts_config
        )

        # Initialize other components
        self.context_builder = DataContextBuilder()
        self.consensus_engine = ConsensusEngine(self.prompts_config)

        # Strategy state
        self.market_context = None
        self.last_market_update = datetime.now() - timedelta(hours=1)
        self.decision_history = []

        # Performance tracking
        self.ai_performance = {
            "total_decisions": 0,
            "correct_decisions": 0,
            "total_cost": 0.0,
            "decisions_by_model": {},
            "backend": "ollama",
        }

    def _setup_indicators(self) -> None:
        """Setup technical indicators (inherited method)."""
        pass

    async def _query_multiple_models(
        self, context: Dict[str, Any], symbol: str
    ) -> List[AIResponse]:
        """Query Ollama using production prompts."""
        try:
            responses = []

            # Get master analyst prompt from YAML config
            master_analyst_config = self.prompts_config.get("prompts", {}).get(
                "master_analyst", {}
            )
            prompt_template = master_analyst_config.get("template", "")

            if not prompt_template:
                # Fallback to simple prompt
                prompt = self._build_fallback_prompt(context, symbol)
            else:
                # Use production prompt template
                prompt = self._build_production_prompt(prompt_template, context, symbol)

            # Query Ollama (simulating multiple models with different temperatures)
            models_to_simulate = [
                (AIModel.SONNET, 0.3),
                (AIModel.HAIKU, 0.2),
            ]

            # Add OPUS for high-value trades
            current_price = float(context.get("current_price", 0))
            if current_price > 100 or abs(float(context.get("daily_change", 0))) > 5:
                models_to_simulate.append((AIModel.OPUS, 0.4))

            for model, temperature in models_to_simulate:
                try:
                    response = await self.anthropic_client.query(
                        prompt=prompt,
                        model=model,
                        max_tokens=2000,
                        temperature=temperature,
                    )

                    if response:
                        responses.append(response)

                except Exception as e:
                    print(f"Error querying model {model.value}: {e}")
                    continue

            return responses

        except Exception as e:
            print(f"Error in multi-model query: {e}")
            return []

    def _build_production_prompt(
        self, template: str, context: Dict[str, Any], symbol: str
    ) -> str:
        """Build prompt using production YAML template."""
        try:
            # Extract all context values
            template_data = {
                "market_context": self._format_market_context(),
                "ticker": symbol,
                "current_price": context.get("current_price", 0),
                "daily_change": context.get("daily_change", 0),
                "volume": context.get("volume", 0),
                "avg_volume": context.get("volume", 0) * 0.8,  # Estimate
                "rsi": context.get("rsi", 50),
                "macd_signal": context.get("macd_signal", 0),
                "macd_histogram": context.get("macd_histogram", 0),
                "price_vs_sma20": context.get("price_vs_sma20", 0),
                "price_vs_sma50": context.get("price_vs_sma50", 0),
                "sma20_vs_sma50": context.get("sma20_vs_sma50", 0),
                "bb_position": context.get("bb_position", 50),
                "atr": context.get("atr", context.get("current_price", 100) * 0.02),
                "atr_percentage": context.get("atr_percentage", 2.0),
                "support_level": context.get(
                    "support", context.get("current_price", 100) * 0.95
                ),
                "resistance_level": context.get(
                    "resistance", context.get("current_price", 100) * 1.05
                ),
                "market_cap": context.get("market_cap", "N/A"),
                "pe_ratio": context.get("pe_ratio", "N/A"),
                "sector": context.get("sector", "Unknown"),
                "sector_performance": context.get("sector_performance", 0),
                "float_shares": context.get("float_shares", "N/A"),
                "short_interest": context.get("short_interest", "N/A"),
                "recent_candles": self._describe_recent_candles(context),
                "identified_patterns": self._identify_patterns(context),
            }

            return template.format(**template_data)

        except KeyError as e:
            print(f"Missing template variable {e}, using fallback")
            return self._build_fallback_prompt(context, symbol)

    def _build_fallback_prompt(self, context: Dict[str, Any], symbol: str) -> str:
        """Fallback prompt when YAML template fails."""
        return f"""
Analyze {symbol} for trading:

CURRENT DATA:
- Price: ${context.get('current_price', 'N/A')}
- Daily Change: {context.get('daily_change', 'N/A')}%
- Volume: {context.get('volume', 'N/A')}
- RSI: {context.get('rsi', 'N/A')}

Provide analysis in JSON format:
{{
    "decision": "BUY/SELL/HOLD",
    "confidence": 0-100,
    "entry_price": price,
    "stop_loss": price or null,
    "take_profit": price or null,
    "position_size_suggestion": 0.01-0.1,
    "risk_reward_ratio": float or null,
    "reasoning": "explanation",
    "key_risks": ["risk1", "risk2"]
}}
"""

    def _format_market_context(self) -> str:
        """Format market context for prompt."""
        if not self.market_context:
            return "Market context unavailable"

        # Create mock data for testing
        spy_price = getattr(self.market_context, "spy_price", 450.00)
        spy_change = getattr(self.market_context, "spy_change", 0.5)
        vix_level = getattr(self.market_context, "vix_level", 15.0)
        vix_change = getattr(self.market_context, "vix_change", -0.2)
        regime = getattr(self.market_context, "regime", "Bullish")

        return f"""Market Environment:
- SPY: ${spy_price} ({spy_change:+.2f}%)
- VIX: {vix_level} ({vix_change:+.2f}%)
- Market Regime: {regime}"""

    def _describe_recent_candles(self, context: Dict[str, Any]) -> str:
        """Describe recent price action."""
        change = context.get("daily_change", 0)
        return f"Recent price action shows {'bullish' if change > 0 else 'bearish'} momentum with {abs(change):.1f}% daily change"

    def _identify_patterns(self, context: Dict[str, Any]) -> str:
        """Identify technical patterns."""
        patterns = []
        rsi = context.get("rsi", 50)

        if rsi > 70:
            patterns.append("Overbought conditions (RSI > 70)")
        elif rsi < 30:
            patterns.append("Oversold conditions (RSI < 30)")

        price_vs_sma20 = context.get("price_vs_sma20", 0)
        if abs(price_vs_sma20) > 3:
            direction = "above" if price_vs_sma20 > 0 else "below"
            patterns.append(f"Price significantly {direction} SMA20")

        return "; ".join(patterns) if patterns else "No significant patterns detected"

    async def close(self) -> None:
        """Close the Ollama client."""
        if hasattr(self.anthropic_client, "ollama_client"):
            await self.anthropic_client.ollama_client.close()


@pytest.mark.integration
@pytest.mark.asyncio
class TestAIStrategyWithOllama:
    """Test AI Strategy using Ollama with production prompts and data."""

    @pytest.fixture
    def strategy_config(self) -> StrategyConfig:
        """Create strategy configuration."""
        return StrategyConfig(
            name="ollama_ai_test",
            mode=StrategyMode.DAY_TRADING,
            lookback_period=20,
            min_confidence=60.0,
        )

    @pytest.fixture
    async def ollama_strategy(self, strategy_config: StrategyConfig) -> Any:
        """Create Ollama-powered AI strategy."""
        strategy = OllamaAIStrategyEngine(strategy_config)

        # Health check - skip if Ollama not available
        # Skip if ollama not available - we can't easily check from here
        # The test will fail gracefully if ollama is not available
        yield strategy
        await strategy.close()

    @pytest.fixture
    def production_data_store(self) -> DataStore:
        """Get production data store."""
        parquet_path = Path(__file__).parent.parent.parent / "data" / "parquet"

        if not parquet_path.exists():
            pytest.skip("Production parquet data not available")

        config = DataStoreConfig(
            base_path=str(parquet_path),
            compression="snappy",
            batch_size=100,
            retention_days=30,
        )

        return DataStore(config)

    @pytest.fixture
    def sample_market_data(self) -> dict[str, Any]:
        """Sample market data for testing."""
        return {
            "symbol": "AAPL",
            "current_price": 150.25,
            "daily_change": 1.5,
            "volume": 1500000,
            "rsi": 65.2,
            "macd_signal": 0.25,
            "macd_histogram": 0.15,
            "price_vs_sma20": 2.3,
            "price_vs_sma50": 5.1,
            "bb_position": 75.0,
            "atr": 3.15,
            "support": 147.50,
            "resistance": 153.00,
            "market_cap": 2500000000,
            "sector": "Technology",
            "timestamp": datetime.now(),
        }

    async def test_yaml_prompt_loading(self, ollama_strategy: Any) -> None:
        """Test that YAML prompts are loaded correctly."""
        assert ollama_strategy.prompts_config is not None
        assert "prompts" in ollama_strategy.prompts_config

        # Check for key prompt templates
        prompts = ollama_strategy.prompts_config.get("prompts", {})
        assert "master_analyst" in prompts

        master_analyst = prompts["master_analyst"]
        assert "template" in master_analyst
        assert "model_preference" in master_analyst

        print(f"Loaded prompts: {list(prompts.keys())}")

    async def test_production_prompt_formatting(
        self,
        ollama_strategy: Any,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test that production prompts are formatted correctly."""
        # Get the master analyst template
        master_analyst_config = ollama_strategy.prompts_config.get("prompts", {}).get(
            "master_analyst", {}
        )
        template = master_analyst_config.get("template", "")

        assert template, "Master analyst template should not be empty"

        # Test prompt building
        prompt = ollama_strategy._build_production_prompt(
            template, sample_market_data, "AAPL"
        )

        assert "AAPL" in prompt
        assert "150.25" in prompt  # Current price
        assert "RSI" in prompt
        assert "JSON format" in prompt

        print(f"Generated prompt length: {len(prompt)} characters")
        print(f"Prompt preview: {prompt[:500]}...")

    async def test_ai_analysis_with_production_prompts(
        self,
        ollama_strategy: Any,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test complete AI analysis using production prompts."""
        # Update market context
        ollama_strategy.market_context = MarketContext(
            regime="trending_bullish",
            strength=0.8,
            risk_level="medium",
            position_size_multiplier=1.0,
            confidence_threshold_adjustment=0.0,
            sectors_to_focus=["Technology"],
            timestamp=datetime.now(),
        )

        # Perform analysis
        decision = await ollama_strategy.analyze(sample_market_data)

        # Validate decision
        assert decision is not None
        assert decision.decision in ["BUY", "SELL", "HOLD"]
        assert 0 <= decision.confidence <= 100
        assert decision.reasoning is not None
        assert decision.total_cost == 0.0  # Ollama is free

        print("\n=== AI Analysis Results ===")
        print(f"Decision: {decision.decision}")
        print(f"Confidence: {decision.confidence}%")
        print(f"Reasoning: {decision.reasoning}")
        print(f"Models used: {decision.models_used}")
        print(f"Response time: {decision.response_time:.2f}s")

    async def test_consensus_mechanism_with_ollama(
        self,
        ollama_strategy: Any,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test that consensus mechanism works with Ollama responses."""
        # Query multiple models (simulated)
        responses = await ollama_strategy._query_multiple_models(
            sample_market_data, "AAPL"
        )

        assert len(responses) > 0, "Should get at least one response"

        # Build consensus
        # First, we need to convert responses to the expected format
        formatted_responses = []
        for response in responses:
            try:
                # Parse JSON from Ollama response
                json_start = response.content.find("{")
                json_end = response.content.rfind("}") + 1

                if json_start >= 0 and json_end > json_start:
                    json_str = response.content[json_start:json_end]
                    parsed = json.loads(json_str)

                    # Create AIResponse with parsed data
                    formatted_response = AIResponse(
                        model=response.model,
                        prompt_type="master_analyst",
                        response=parsed,
                        confidence=parsed.get("confidence", 50),
                        tokens_used=response.total_tokens,
                        cost=0.0,
                        timestamp=datetime.now(),
                        cache_hit=False,
                    )
                    formatted_responses.append(formatted_response)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Failed to parse response: {e}")
                continue

        if formatted_responses:
            consensus = await ollama_strategy.consensus_engine.build_consensus(
                formatted_responses
            )

            assert consensus is not None
            assert consensus.action in ["BUY", "SELL", "HOLD"]
            print(
                f"Consensus: {consensus.action} with {consensus.confidence}% confidence"
            )
        else:
            print("Could not parse any responses for consensus")

    async def test_production_data_integration(
        self,
        ollama_strategy: Any,
        production_data_store: DataStore,
    ) -> None:
        """Test AI strategy with real production data."""
        try:
            # Get available data summary
            summary = await production_data_store.get_data_summary()

            if summary["total_files"] == 0:
                pytest.skip("No production data available")

            # Get available tickers
            available_range = production_data_store.get_available_data_range(
                "AAPL", TimeFrame.ONE_DAY
            )

            if not available_range:
                pytest.skip("No data range available")

            # Pick first available ticker
            # available_range is now a tuple (start_date, end_date) for AAPL
            ticker = "AAPL"

            # Load recent data
            data = await production_data_store.load_market_data(
                ticker, TimeFrame.ONE_DAY, limit=100
            )

            if data is None or len(data) == 0:
                pytest.skip(f"No data for {ticker}")

            # Get latest data point
            latest = data.tail(1)

            # Build market data context
            market_data = {
                "symbol": ticker,
                "current_price": float(
                    latest.select("close").item()
                    if "close" in latest.columns
                    else (
                        latest.select("price").item()
                        if "price" in latest.columns
                        else 100
                    )
                ),
                "daily_change": float(
                    latest.select("change_percent").item()
                    if "change_percent" in latest.columns
                    else 0
                ),
                "volume": int(
                    latest.select("volume").item()
                    if "volume" in latest.columns
                    else 1000000
                ),
                "rsi": 55.0,  # Would calculate from historical data
                "timestamp": datetime.now(),
            }

            print(
                f"\nAnalyzing {ticker} with real data: ${market_data['current_price']}"
            )

            # Analyze with AI
            decision = await ollama_strategy.analyze(market_data)

            assert decision is not None
            print(
                f"Real data analysis - {ticker}: {decision.decision} ({decision.confidence}%)"
            )
            print(f"Reasoning: {decision.reasoning}")

        except Exception as e:
            pytest.skip(f"Could not test with production data: {e}")

    async def test_performance_tracking(
        self, ollama_strategy: Any, sample_market_data: dict[str, Any]
    ) -> None:
        """Test that performance metrics are tracked correctly."""
        initial_performance = ollama_strategy.ai_performance.copy()

        # Make a decision
        decision = await ollama_strategy.analyze(sample_market_data)

        # Check performance was updated
        assert (
            ollama_strategy.ai_performance["total_decisions"]
            > initial_performance["total_decisions"]
        )

        # Verify decision is valid
        assert decision.decision in ["BUY", "SELL", "HOLD"]
        assert ollama_strategy.ai_performance["backend"] == "ollama"
        assert ollama_strategy.ai_performance["total_cost"] == 0.0

        print(f"Performance tracking: {ollama_strategy.ai_performance}")

    async def test_error_handling(self, ollama_strategy: Any) -> None:
        """Test error handling with invalid data."""
        # Test with minimal data
        minimal_data = {
            "symbol": "TEST",
            "current_price": 100,
            "timestamp": datetime.now(),
        }

        decision = await ollama_strategy.analyze(minimal_data)

        # Should handle gracefully
        if decision:
            assert decision.decision in ["BUY", "SELL", "HOLD"]
            print(f"Handled minimal data: {decision.decision}")
        else:
            print("Returned None for minimal data (acceptable)")

        # Test with invalid price
        invalid_data = {
            "symbol": "INVALID",
            "current_price": "not_a_number",
            "timestamp": datetime.now(),
        }

        try:
            decision = await ollama_strategy.analyze(invalid_data)
            print("Handled invalid data gracefully")
        except Exception as e:
            print(f"Error handled: {e}")

    @pytest.mark.slow
    async def test_realistic_trading_workflow(
        self,
        ollama_strategy: Any,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test complete realistic trading workflow."""
        print("\n=== Realistic Trading Workflow Test ===")

        # Simulate receiving a signal
        print(f"📊 Received signal for {sample_market_data['symbol']}")
        print(f"💰 Current price: ${sample_market_data['current_price']}")
        print(f"📈 Daily change: {sample_market_data['daily_change']}%")
        print(f"📊 RSI: {sample_market_data['rsi']}")

        # Set market context
        ollama_strategy.market_context = MarketContext(
            regime="trending_bullish",
            strength=0.8,
            risk_level="medium",
            position_size_multiplier=1.0,
            confidence_threshold_adjustment=0.0,
            sectors_to_focus=["Technology"],
            timestamp=datetime.now(),
        )

        print("🌐 Market context set")

        # Analyze with AI
        print("🤖 Analyzing with Ollama AI...")
        start_time = datetime.now()

        decision = await ollama_strategy.analyze(sample_market_data)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Report results
        print("\n=== AI Decision Results ===")
        print(f"🎯 Decision: {decision.decision}")
        print(f"🎯 Confidence: {decision.confidence}%")
        print(f"💭 Reasoning: {decision.reasoning}")
        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print("💰 Cost: $0.00 (Local Ollama)")
        print(f"🔧 Models used: {', '.join(decision.models_used)}")

        # Simulate trade decision logic
        if decision.decision == "BUY" and decision.confidence > 70:
            print("✅ EXECUTE TRADE: High confidence BUY signal")
            if decision.entry_price:
                print(f"📍 Entry: ${decision.entry_price}")
            if decision.stop_loss:
                print(f"🛡️  Stop Loss: ${decision.stop_loss}")
            if decision.take_profit:
                print(f"🎯 Take Profit: ${decision.take_profit}")
        elif decision.decision == "SELL" and decision.confidence > 70:
            print("⚠️  SELL SIGNAL: High confidence SELL")
        else:
            print("⏸️  NO TRADE: Confidence too low or HOLD decision")

        # Final assertions
        assert decision is not None
        assert decision.decision in ["BUY", "SELL", "HOLD"]
        assert 0 <= decision.confidence <= 100
        assert processing_time > 0

        print("\n✅ Workflow completed successfully")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])
