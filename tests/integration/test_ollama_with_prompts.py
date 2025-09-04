"""
Integration tests for Ollama AI with production prompts.

This test bypasses complex imports by testing the core components directly.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
import yaml

from services.strategy_engine.src.ollama_client import OllamaClient


class ProductionPromptProcessor:
    """Processes production prompts with market data."""

    def __init__(self, prompts_config_path: str):
        """Load prompts configuration."""
        self.prompts_config = {}
        try:
            with open(prompts_config_path, "r") as f:
                self.prompts_config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Could not load prompts from {prompts_config_path}")

    def build_master_analyst_prompt(self, market_data: Dict[str, Any]) -> str:
        """Build master analyst prompt using production template."""
        prompts = self.prompts_config.get("prompts", {})
        master_config = prompts.get("master_analyst", {})
        template = master_config.get("template", "")

        if not template:
            return self._fallback_prompt(market_data)

        # Prepare template data
        template_data = {
            "market_context": self._format_market_context(),
            "ticker": market_data.get("ticker", "UNKNOWN"),
            "current_price": market_data.get("price", 0),
            "daily_change": market_data.get("daily_change", 0),
            "volume": market_data.get("volume", 0),
            "avg_volume": market_data.get("volume", 0) * 0.8,
            "rsi": market_data.get("rsi", 50),
            "macd_signal": market_data.get("macd_signal", 0),
            "macd_histogram": market_data.get("macd_histogram", 0),
            "price_vs_sma20": market_data.get("price_vs_sma20", 0),
            "price_vs_sma50": market_data.get("price_vs_sma50", 0),
            "sma20_vs_sma50": market_data.get("sma20_vs_sma50", 0),
            "bb_position": market_data.get("bb_position", 50),
            "atr": market_data.get("atr", market_data.get("price", 100) * 0.02),
            "atr_percentage": market_data.get("atr_percentage", 2.0),
            "support_level": market_data.get(
                "support", market_data.get("price", 100) * 0.95
            ),
            "resistance_level": market_data.get(
                "resistance", market_data.get("price", 100) * 1.05
            ),
            "market_cap": market_data.get("market_cap", "N/A"),
            "pe_ratio": market_data.get("pe_ratio", "N/A"),
            "sector": market_data.get("sector", "Technology"),
            "sector_performance": market_data.get("sector_performance", 0),
            "float_shares": market_data.get("float_shares", "N/A"),
            "short_interest": market_data.get("short_interest", "N/A"),
            "recent_candles": self._describe_recent_candles(market_data),
            "identified_patterns": self._identify_patterns(market_data),
        }

        try:
            # Handle the complex template with proper escaping
            formatted_template = template
            for key, value in template_data.items():
                placeholder = "{" + key + "}"
                formatted_template = formatted_template.replace(placeholder, str(value))
            return formatted_template
        except Exception as e:
            print(f"Template formatting error: {e}")
            return self._fallback_prompt(market_data)

    def _format_market_context(self) -> str:
        """Format market context."""
        return """Market Environment:
- SPY: $450.25 (+0.3%)
- VIX: 18.5 (-1.2%)
- Market Regime: trending_bullish
- Sector Rotation: Technology outperforming"""

    def _describe_recent_candles(self, data: Dict[str, Any]) -> str:
        """Describe recent price action."""
        change = data.get("daily_change", 0)
        if change > 2:
            return "Strong bullish momentum with large green candles"
        elif change < -2:
            return "Bearish pressure with red candles dominating"
        else:
            return "Mixed price action with consolidation pattern"

    def _identify_patterns(self, data: Dict[str, Any]) -> str:
        """Identify technical patterns."""
        patterns = []

        rsi = data.get("rsi", 50)
        if rsi > 70:
            patterns.append("Overbought RSI divergence")
        elif rsi < 30:
            patterns.append("Oversold RSI bounce setup")

        price_vs_sma20 = data.get("price_vs_sma20", 0)
        if price_vs_sma20 > 3:
            patterns.append("Strong uptrend above SMA20")
        elif price_vs_sma20 < -3:
            patterns.append("Downtrend below SMA20 support")

        return "; ".join(patterns) if patterns else "Consolidation pattern"

    def _fallback_prompt(self, data: Dict[str, Any]) -> str:
        """Simple fallback prompt."""
        return f"""
Analyze {data.get('ticker', 'STOCK')} for trading:

CURRENT DATA:
- Price: ${data.get('price', 0)}
- Daily Change: {data.get('daily_change', 0)}%
- Volume: {data.get('volume', 0):,}
- RSI: {data.get('rsi', 50)}

Provide your recommendation in JSON format:
{{
    "decision": "BUY/SELL/HOLD",
    "confidence": 0-100,
    "entry_price": {data.get('price', 0)},
    "stop_loss": null,
    "take_profit": null,
    "position_size_suggestion": 0.05,
    "reasoning": "Brief explanation",
    "key_risks": ["risk1", "risk2"]
}}

Focus on risk management and high-probability setups.
"""


class TradingSignalSimulator:
    """Simulates trading signals for testing."""

    def __init__(
        self, ollama_client: OllamaClient, prompt_processor: ProductionPromptProcessor
    ):
        self.ollama_client = ollama_client
        self.prompt_processor = prompt_processor
        self.decisions_made = []

    async def process_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a trading signal through the AI system."""
        # Build production prompt
        prompt = self.prompt_processor.build_master_analyst_prompt(market_data)

        # Query Ollama
        response = await self.ollama_client.query(
            prompt, max_tokens=1000, temperature=0.3
        )

        # Parse response
        decision = self._parse_ai_response(response.content)
        decision["processing_time"] = response.response_time
        decision["tokens_used"] = response.tokens_used
        decision["cost"] = 0.0
        decision["timestamp"] = datetime.now().isoformat()

        # Track decision
        self.decisions_made.append(decision)

        return decision

    def _parse_ai_response(self, content: str) -> Dict[str, Any]:
        """Parse AI response to extract trading decision."""
        try:
            # Find JSON in response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                parsed = json.loads(json_str)

                # Validate and clean
                decision = {
                    "decision": parsed.get("decision", "HOLD").upper(),
                    "confidence": max(0, min(100, parsed.get("confidence", 50))),
                    "reasoning": parsed.get("reasoning", "AI analysis"),
                    "entry_price": parsed.get("entry_price"),
                    "stop_loss": parsed.get("stop_loss"),
                    "take_profit": parsed.get("take_profit"),
                    "position_size": max(
                        0.01, min(0.2, parsed.get("position_size_suggestion", 0.05))
                    ),
                    "key_risks": parsed.get("key_risks", []),
                }

                # Ensure valid decision
                if decision["decision"] not in ["BUY", "SELL", "HOLD"]:
                    decision["decision"] = "HOLD"

                return decision

        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Fallback parsing
        content_lower = content.lower()
        if "buy" in content_lower:
            decision = "BUY"
        elif "sell" in content_lower:
            decision = "SELL"
        else:
            decision = "HOLD"

        return {
            "decision": decision,
            "confidence": 50,
            "reasoning": "Parsed from text response",
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "position_size": 0.05,
            "key_risks": [],
        }


@pytest.mark.integration
@pytest.mark.asyncio
class TestOllamaWithPrompts:
    """Integration tests for Ollama using production prompts."""

    @pytest.fixture
    def prompts_config_path(self):
        """Get path to prompts configuration."""
        config_path = (
            Path(__file__).parent.parent.parent
            / "config"
            / "ai_strategy"
            / "prompts.yaml"
        )
        if not config_path.exists():
            pytest.skip("Prompts configuration not found")
        return str(config_path)

    @pytest.fixture
    def ollama_client(self):
        """Create Ollama client."""
        url = os.getenv("OLLAMA_URL", "http://192.168.1.133:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
        return OllamaClient(url, model)

    @pytest.fixture
    def prompt_processor(self, prompts_config_path):
        """Create prompt processor."""
        return ProductionPromptProcessor(prompts_config_path)

    @pytest.fixture
    def trading_simulator(self, ollama_client, prompt_processor):
        """Create trading signal simulator."""
        return TradingSignalSimulator(ollama_client, prompt_processor)

    @pytest.fixture
    def sample_stocks(self):
        """Sample stock data for testing."""
        return [
            {
                "ticker": "AAPL",
                "price": 150.25,
                "daily_change": 1.5,
                "volume": 1500000,
                "rsi": 65.2,
                "macd_signal": 0.25,
                "price_vs_sma20": 2.3,
                "bb_position": 75.0,
                "sector": "Technology",
            },
            {
                "ticker": "TSLA",
                "price": 850.75,
                "daily_change": -2.8,
                "volume": 2200000,
                "rsi": 35.1,
                "macd_signal": -0.15,
                "price_vs_sma20": -1.8,
                "bb_position": 25.0,
                "sector": "Automotive",
            },
            {
                "ticker": "GOOGL",
                "price": 2800.50,
                "daily_change": 0.3,
                "volume": 800000,
                "rsi": 52.8,
                "macd_signal": 0.05,
                "price_vs_sma20": 0.8,
                "bb_position": 55.0,
                "sector": "Technology",
            },
        ]

    async def test_ollama_health_check(self, ollama_client):
        """Test Ollama server connectivity."""
        health_check = await ollama_client.health_check()

        if not health_check:
            pytest.skip("Ollama server not available")

        assert health_check, "Ollama should be healthy"

        # Test model availability
        models = await ollama_client.list_models()
        print(f"Available models: {models}")
        assert len(models) > 0, "Should have at least one model"

    async def test_prompt_loading(self, prompt_processor):
        """Test that production prompts load correctly."""
        assert prompt_processor.prompts_config is not None

        prompts = prompt_processor.prompts_config.get("prompts", {})
        assert "master_analyst" in prompts

        master_config = prompts["master_analyst"]
        assert "template" in master_config
        assert "model_preference" in master_config
        assert "max_tokens" in master_config

        print(f"Loaded prompt types: {list(prompts.keys())}")
        print(f"Master analyst model preference: {master_config['model_preference']}")

    async def test_production_prompt_generation(self, prompt_processor, sample_stocks):
        """Test production prompt generation with real templates."""
        stock_data = sample_stocks[0]  # AAPL

        prompt = prompt_processor.build_master_analyst_prompt(stock_data)

        # Validate prompt content
        assert "AAPL" in prompt
        assert "150.25" in prompt
        assert "RSI" in prompt
        assert "MACD" in prompt
        assert "JSON format" in prompt
        assert "BUY" in prompt and "SELL" in prompt and "HOLD" in prompt

        print(f"Generated prompt length: {len(prompt)} characters")
        print(f"Prompt preview:\n{prompt[:800]}...")

    async def test_signal_processing_workflow(self, trading_simulator, sample_stocks):
        """Test complete signal processing workflow."""
        # Skip if Ollama not available
        health_check = await trading_simulator.ollama_client.health_check()
        if not health_check:
            pytest.skip("Ollama server not available")

        stock_data = sample_stocks[0]  # AAPL

        print(f"\n=== Processing Signal for {stock_data['ticker']} ===")
        print(f"Price: ${stock_data['price']}")
        print(f"Daily Change: {stock_data['daily_change']}%")
        print(f"RSI: {stock_data['rsi']}")

        # Process signal
        decision = await trading_simulator.process_signal(stock_data)

        # Validate decision
        assert decision is not None
        assert decision["decision"] in ["BUY", "SELL", "HOLD"]
        assert 0 <= decision["confidence"] <= 100
        assert "reasoning" in decision
        assert decision["cost"] == 0.0
        assert decision["processing_time"] > 0

        print("\n=== AI Decision ===")
        print(f"Decision: {decision['decision']}")
        print(f"Confidence: {decision['confidence']}%")
        print(f"Reasoning: {decision['reasoning']}")
        print(f"Processing Time: {decision['processing_time']:.2f}s")

        if decision.get("entry_price"):
            print(f"Entry Price: ${decision['entry_price']}")
        if decision.get("stop_loss"):
            print(f"Stop Loss: ${decision['stop_loss']}")
        if decision.get("take_profit"):
            print(f"Take Profit: ${decision['take_profit']}")

    async def test_multiple_stock_analysis(self, trading_simulator, sample_stocks):
        """Test analyzing multiple stocks."""
        # Skip if Ollama not available
        health_check = await trading_simulator.ollama_client.health_check()
        if not health_check:
            pytest.skip("Ollama server not available")

        results = []

        print("\n=== Multi-Stock Analysis ===")

        for stock_data in sample_stocks:
            try:
                decision = await trading_simulator.process_signal(stock_data)
                results.append((stock_data["ticker"], decision))

                print(
                    f"{stock_data['ticker']}: {decision['decision']} ({decision['confidence']}%)"
                )

            except Exception as e:
                print(f"Failed to analyze {stock_data['ticker']}: {e}")

        assert len(results) > 0, "Should analyze at least one stock"

        # Check decision distribution
        decisions = [result[1]["decision"] for result in results]
        unique_decisions = set(decisions)

        print(
            f"Decision distribution: {dict((d, decisions.count(d)) for d in unique_decisions)}"
        )

        # Performance summary
        avg_confidence = sum(result[1]["confidence"] for result in results) / len(
            results
        )
        avg_processing_time = sum(
            result[1]["processing_time"] for result in results
        ) / len(results)

        print(f"Average confidence: {avg_confidence:.1f}%")
        print(f"Average processing time: {avg_processing_time:.2f}s")

    async def test_scenario_analysis(self, trading_simulator):
        """Test different market scenarios."""
        # Skip if Ollama not available
        health_check = await trading_simulator.ollama_client.health_check()
        if not health_check:
            pytest.skip("Ollama server not available")

        scenarios = [
            {
                "name": "Oversold Bounce",
                "data": {
                    "ticker": "OVERSOLD",
                    "price": 100,
                    "daily_change": -5.2,
                    "rsi": 25,  # Oversold
                    "volume": 2500000,
                    "price_vs_sma20": -8.5,
                },
            },
            {
                "name": "Overbought Peak",
                "data": {
                    "ticker": "OVERBOUGHT",
                    "price": 200,
                    "daily_change": 8.1,
                    "rsi": 85,  # Overbought
                    "volume": 3000000,
                    "price_vs_sma20": 12.3,
                },
            },
            {
                "name": "Neutral Market",
                "data": {
                    "ticker": "NEUTRAL",
                    "price": 150,
                    "daily_change": 0.1,
                    "rsi": 50,  # Neutral
                    "volume": 1000000,
                    "price_vs_sma20": 0.5,
                },
            },
        ]

        print("\n=== Scenario Analysis ===")

        for scenario in scenarios:
            print(f"\n{scenario['name']}:")
            print(
                f"RSI: {scenario['data']['rsi']}, Change: {scenario['data']['daily_change']}%"
            )

            decision = await trading_simulator.process_signal(scenario["data"])

            print(f"Decision: {decision['decision']} ({decision['confidence']}%)")
            print(f"Reasoning: {decision['reasoning'][:100]}...")

    async def test_production_data_integration(self, trading_simulator):
        """Test with production parquet data if available."""
        # Skip if Ollama not available
        health_check = await trading_simulator.ollama_client.health_check()
        if not health_check:
            pytest.skip("Ollama server not available")

        # Check for production data
        parquet_path = (
            Path(__file__).parent.parent.parent / "data" / "parquet" / "market_data"
        )

        if not parquet_path.exists():
            pytest.skip("No production parquet data available")

        parquet_files = list(parquet_path.glob("**/*.parquet"))

        if not parquet_files:
            pytest.skip("No parquet files found")

        print("\n=== Production Data Integration ===")
        print(f"Found {len(parquet_files)} data files")

        # Read first available file
        sample_file = parquet_files[0]
        df = pd.read_parquet(sample_file)

        if len(df) == 0:
            pytest.skip("Empty data file")

        # Get latest data point
        latest = df.iloc[-1]

        # Build market data
        market_data = {
            "ticker": latest.get("ticker", "UNKNOWN"),
            "price": float(latest.get("close", latest.get("price", 100))),
            "daily_change": float(latest.get("change_percent", 0)),
            "volume": int(latest.get("volume", 1000000)),
            "rsi": 55.0,  # Would calculate from historical data
        }

        print(
            f"Analyzing real data: {market_data['ticker']} at ${market_data['price']}"
        )

        # Analyze with AI
        decision = await trading_simulator.process_signal(market_data)

        print(f"Real data decision: {decision['decision']} ({decision['confidence']}%)")
        print(f"Reasoning: {decision['reasoning']}")

        assert decision is not None
        assert decision["decision"] in ["BUY", "SELL", "HOLD"]

    @pytest.mark.slow
    async def test_performance_benchmarking(self, trading_simulator, sample_stocks):
        """Benchmark AI performance."""
        # Skip if Ollama not available
        health_check = await trading_simulator.ollama_client.health_check()
        if not health_check:
            pytest.skip("Ollama server not available")

        print("\n=== Performance Benchmarking ===")

        # Process multiple signals
        start_time = datetime.now()
        decisions = []

        for i in range(5):  # Test with 5 iterations
            for stock_data in sample_stocks[:2]:  # Use first 2 stocks
                decision = await trading_simulator.process_signal(stock_data)
                decisions.append(decision)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Calculate metrics
        total_decisions = len(decisions)
        avg_processing_time = (
            sum(d["processing_time"] for d in decisions) / total_decisions
        )
        avg_confidence = sum(d["confidence"] for d in decisions) / total_decisions
        decisions_per_second = total_decisions / total_time

        print(f"Total decisions: {total_decisions}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average processing time: {avg_processing_time:.2f}s")
        print(f"Average confidence: {avg_confidence:.1f}%")
        print(f"Decisions per second: {decisions_per_second:.2f}")
        print("Total cost: $0.00 (Local Ollama)")

        # Assertions
        assert total_decisions > 0
        assert avg_processing_time > 0
        assert 0 <= avg_confidence <= 100
        assert decisions_per_second > 0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])
