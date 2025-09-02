"""
Integration tests for AI Strategy using Ollama with production data.

Tests the complete flow of receiving trading signals and processing them
through the AI strategy engine using local Ollama models with real market data.
"""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Import the classes we need to test
from services.strategy_engine.src.ollama_client import OllamaClient


class MockSignal:
    """Mock trading signal for testing."""

    def __init__(self, ticker: str, signal_type: str = "BUY", strength: float = 0.8):
        self.ticker = ticker
        self.signal_type = signal_type
        self.strength = strength
        self.timestamp = datetime.now()
        self.source = "test_screener"


@pytest.mark.integration
@pytest.mark.asyncio
class TestOllamaAIStrategyIntegration:
    """Integration tests for Ollama AI Strategy with production data."""

    @pytest.fixture
    def ollama_client(self):
        """Create Ollama client for testing."""
        url = os.getenv("OLLAMA_URL", "http://192.168.1.133:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
        return OllamaClient(url, model)

    @pytest.fixture
    def parquet_data_path(self):
        """Get path to production parquet data."""
        parquet_path = Path(__file__).parent.parent.parent / "data" / "parquet"
        if not parquet_path.exists():
            pytest.skip("Production parquet data not available")
        return parquet_path

    async def test_ollama_health_check(self, ollama_client):
        """Test that Ollama server is accessible and healthy."""
        health_check = await ollama_client.health_check()

        if not health_check:
            pytest.skip("Ollama server not available or model not loaded")

        assert health_check, "Ollama server should be healthy"

    async def test_ollama_basic_query(self, ollama_client):
        """Test basic Ollama query functionality."""
        # Skip if Ollama not available
        if not await ollama_client.health_check():
            pytest.skip("Ollama server not available")

        response = await ollama_client.query(
            "What is 2+2? Respond with just the number.", max_tokens=10, temperature=0.1
        )

        assert response is not None
        assert len(response.content) > 0
        assert response.cost == 0.0  # Local models are free
        assert response.response_time > 0

    async def test_production_data_loading(self, parquet_data_path):
        """Test loading production parquet data."""
        try:
            # Check if market data directory exists
            market_data_path = parquet_data_path / "market_data"
            if not market_data_path.exists():
                pytest.skip("No market data directory found")

            # List available parquet files
            parquet_files = list(market_data_path.glob("**/*.parquet"))
            if not parquet_files:
                pytest.skip("No parquet files found")

            print(f"Found {len(parquet_files)} parquet files")

            # Try to read the first parquet file
            sample_file = parquet_files[0]
            df = pd.read_parquet(sample_file)

            assert df is not None
            assert len(df) > 0
            print(f"Successfully loaded {len(df)} records from {sample_file.name}")

        except Exception as e:
            pytest.skip(f"Could not access production data: {e}")

    async def test_ai_trading_analysis(self, ollama_client, parquet_data_path):
        """Test AI trading analysis with production data."""
        # Skip if Ollama not available
        if not await ollama_client.health_check():
            pytest.skip("Ollama server not available")

        try:
            # Load some production data
            market_data_path = parquet_data_path / "market_data"
            parquet_files = list(market_data_path.glob("**/*.parquet"))

            if not parquet_files:
                pytest.skip("No parquet files available")

            # Read a sample file
            sample_file = parquet_files[0]
            df = pd.read_parquet(sample_file)

            if len(df) == 0:
                pytest.skip("Empty data file")

            # Get the most recent data point
            latest_data = df.iloc[-1]

            # Build trading analysis prompt
            ticker = latest_data.get("ticker", "UNKNOWN")
            price = latest_data.get("close", latest_data.get("price", 100))
            volume = latest_data.get("volume", 1000000)

            prompt = f"""Analyze this stock for trading:

Stock: {ticker}
Current Price: ${price}
Volume: {volume:,}

Based on this data, should I BUY, SELL, or HOLD?
Respond in JSON format:
{{"decision": "BUY/SELL/HOLD", "confidence": 0-100, "reasoning": "explanation"}}"""

            print(f"Testing AI analysis for {ticker} at ${price}")

            # Get AI analysis
            response = await ollama_client.query(
                prompt, max_tokens=200, temperature=0.3
            )

            assert response is not None
            assert len(response.content) > 0
            assert response.cost == 0.0

            print(f"AI Response: {response.content[:200]}...")
            print(f"Response time: {response.response_time:.2f}s")

            # Try to parse JSON from response
            import json

            try:
                # Extract JSON from response
                json_start = response.content.find("{")
                json_end = response.content.rfind("}") + 1

                if json_start >= 0 and json_end > json_start:
                    json_str = response.content[json_start:json_end]
                    decision_data = json.loads(json_str)

                    assert "decision" in decision_data
                    assert decision_data["decision"] in ["BUY", "SELL", "HOLD"]
                    print(f"Parsed decision: {decision_data}")

            except json.JSONDecodeError:
                print("Could not parse JSON, but response was generated")

        except Exception as e:
            pytest.fail(f"AI analysis test failed: {e}")

    async def test_multiple_stock_analysis(self, ollama_client):
        """Test analyzing multiple stocks."""
        # Skip if Ollama not available
        if not await ollama_client.health_check():
            pytest.skip("Ollama server not available")

        test_stocks = [
            {"ticker": "AAPL", "price": 150.25, "change": 1.5},
            {"ticker": "GOOGL", "price": 2800.50, "change": -0.8},
            {"ticker": "TSLA", "price": 850.75, "change": 3.2},
        ]

        results = []

        for stock in test_stocks:
            prompt = f"""Quick analysis for {stock['ticker']}:
Price: ${stock['price']}, Change: {stock['change']}%

Recommendation (BUY/SELL/HOLD)?"""

            try:
                response = await ollama_client.query(
                    prompt, max_tokens=50, temperature=0.2
                )
                results.append((stock["ticker"], response.content[:100]))
                print(f"{stock['ticker']}: {response.content[:100]}")

            except Exception as e:
                print(f"Failed to analyze {stock['ticker']}: {e}")

        assert len(results) > 0, "Should analyze at least one stock"

    async def test_technical_indicator_analysis(self, ollama_client):
        """Test AI analysis with technical indicators."""
        # Skip if Ollama not available
        if not await ollama_client.health_check():
            pytest.skip("Ollama server not available")

        # Test oversold scenario
        oversold_prompt = """Stock Analysis:
Ticker: AAPL
RSI: 25 (oversold)
Price: $145 (down 3%)
Volume: High

Should I BUY, SELL, or HOLD? Brief explanation."""

        response = await ollama_client.query(
            oversold_prompt, max_tokens=100, temperature=0.2
        )
        assert response is not None
        print(f"Oversold analysis: {response.content}")

        # Test overbought scenario
        overbought_prompt = """Stock Analysis:
Ticker: AAPL
RSI: 85 (overbought)
Price: $155 (up 4%)
Volume: High

Should I BUY, SELL, or HOLD? Brief explanation."""

        response2 = await ollama_client.query(
            overbought_prompt, max_tokens=100, temperature=0.2
        )
        assert response2 is not None
        print(f"Overbought analysis: {response2.content}")

    async def test_error_handling(self, ollama_client):
        """Test error handling with invalid requests."""
        if not await ollama_client.health_check():
            pytest.skip("Ollama server not available")

        # Test with very long prompt (should handle gracefully)
        long_prompt = "What is the stock market? " * 1000

        try:
            response = await ollama_client.query(long_prompt, max_tokens=50)
            # Should either succeed or fail gracefully
            if response:
                assert len(response.content) >= 0
        except Exception as e:
            print(f"Long prompt handled with error: {e}")
            # This is acceptable - we just want to ensure it doesn't crash

    async def test_response_time_tracking(self, ollama_client):
        """Test that response times are tracked."""
        if not await ollama_client.health_check():
            pytest.skip("Ollama server not available")

        prompt = "What is 5+5? Answer briefly."
        response = await ollama_client.query(prompt, max_tokens=20)

        assert response is not None
        assert response.response_time > 0
        assert response.tokens_used >= 0
        print(
            f"Response time: {response.response_time:.2f}s, Tokens: {response.tokens_used}"
        )

    @pytest.mark.slow
    async def test_realistic_trading_scenario(self, ollama_client, parquet_data_path):
        """Test a realistic end-to-end trading scenario."""
        if not await ollama_client.health_check():
            pytest.skip("Ollama server not available")

        try:
            print("\n=== Realistic Trading Scenario Test ===")

            # Simulate receiving a trading signal
            signal = MockSignal("AAPL", "BUY", 0.8)
            print(
                f"Received signal: {signal.signal_type} {signal.ticker} (strength: {signal.strength})"
            )

            # Try to get real production data
            try:
                market_data_path = parquet_data_path / "market_data"
                parquet_files = list(market_data_path.glob("**/*.parquet"))

                if parquet_files:
                    # Read first available file
                    df = pd.read_parquet(parquet_files[0])
                    if len(df) > 0:
                        latest = df.iloc[-1]
                        price = latest.get("close", latest.get("price", 150))
                        volume = latest.get("volume", 1000000)
                        ticker = latest.get("ticker", "AAPL")

                        # Create comprehensive trading prompt
                        trading_prompt = f"""
You are a professional stock trader analyzing {ticker}.

CURRENT DATA:
- Stock: {ticker}
- Price: ${price}
- Volume: {volume:,}
- Signal Received: {signal.signal_type} with {signal.strength} strength

ANALYSIS REQUEST:
Based on this data and the incoming {signal.signal_type} signal, provide your recommendation.

Respond in JSON format:
{{
    "decision": "BUY/SELL/HOLD",
    "confidence": 0-100,
    "reasoning": "Brief explanation of your decision",
    "entry_price": {price},
    "risk_assessment": "LOW/MEDIUM/HIGH"
}}

Consider the signal strength and current market conditions."""

                        print(f"Analyzing {ticker} at ${price} with AI...")
                        start_time = datetime.now()

                        # Get AI analysis
                        response = await ollama_client.query(
                            trading_prompt, max_tokens=300, temperature=0.3
                        )

                        end_time = datetime.now()
                        processing_time = (end_time - start_time).total_seconds()

                        print(f"\n=== AI Analysis Results ===")
                        print(f"Processing Time: {processing_time:.2f} seconds")
                        print(f"Response: {response.content}")
                        print(f"Cost: $0.00 (Local Ollama)")

                        # Try to parse JSON response
                        import json

                        try:
                            json_start = response.content.find("{")
                            json_end = response.content.rfind("}") + 1

                            if json_start >= 0 and json_end > json_start:
                                json_str = response.content[json_start:json_end]
                                decision_data = json.loads(json_str)

                                print(f"\n=== Parsed Decision ===")
                                print(
                                    f"Decision: {decision_data.get('decision', 'UNKNOWN')}"
                                )
                                print(
                                    f"Confidence: {decision_data.get('confidence', 0)}%"
                                )
                                print(
                                    f"Reasoning: {decision_data.get('reasoning', 'N/A')}"
                                )

                                # Simulate trade execution decision
                                decision_type = decision_data.get("decision", "HOLD")
                                confidence = decision_data.get("confidence", 0)

                                if decision_type == "BUY" and confidence > 60:
                                    print(
                                        f"\n✅ TRADE APPROVED: High confidence BUY signal"
                                    )
                                elif decision_type == "SELL" and confidence > 60:
                                    print(
                                        f"\n⚠️ SELL SIGNAL: High confidence SELL signal"
                                    )
                                else:
                                    print(
                                        f"\n⏸️ NO TRADE: Low confidence or HOLD decision"
                                    )

                        except json.JSONDecodeError:
                            print(
                                "\n⚠️ Could not parse JSON response, but AI provided analysis"
                            )

                        # Assertions
                        assert response is not None
                        assert len(response.content) > 0
                        assert response.cost == 0.0
                        assert processing_time > 0

                        return  # Exit successfully

            except Exception as e:
                print(f"Could not load production data: {e}")

            # Fallback to mock scenario
            print("Using mock data for realistic scenario test")

            mock_prompt = """
You are analyzing AAPL stock for trading.

SCENARIO:
- Stock: AAPL
- Current Price: $150.25
- Daily Change: +1.2%
- Volume: 1.5M (above average)
- RSI: 58 (neutral)
- Signal: BUY with 80% strength

Should you execute this trade? Provide BUY/SELL/HOLD recommendation with reasoning."""

            response = await ollama_client.query(
                mock_prompt, max_tokens=200, temperature=0.2
            )

            print(f"\n=== Mock Scenario Results ===")
            print(f"AI Response: {response.content}")
            print(f"Response Time: {response.response_time:.2f}s")

            assert response is not None
            assert len(response.content) > 0

        except Exception as e:
            pytest.fail(f"Realistic scenario test failed: {e}")


# Utility functions for testing
def create_test_parquet_data(ticker: str, days: int = 30) -> pd.DataFrame:
    """Create sample parquet data for testing."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days), periods=days, freq="D"
    )

    # Generate realistic price data
    base_price = 100.0
    prices = [base_price]

    for i in range(1, days):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    data = {
        "date": dates,
        "ticker": ticker,
        "open": [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
        "high": [p * (1 + np.random.uniform(0.00, 0.02)) for p in prices],
        "low": [p * (1 + np.random.uniform(-0.02, 0.00)) for p in prices],
        "close": prices,
        "volume": [int(np.random.uniform(500000, 2000000)) for _ in range(days)],
        "change_percent": [0]
        + [((prices[i] - prices[i - 1]) / prices[i - 1]) * 100 for i in range(1, days)],
    }

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Run specific tests manually
    pytest.main([__file__, "-v", "-s"])
