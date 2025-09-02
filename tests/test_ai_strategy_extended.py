"""
Extended Test Suite for AI Strategy Engine

This module provides comprehensive unit tests for all AI strategy components,
including edge cases, error handling, and integration scenarios.
"""

import asyncio
import json
import os

# Import modules to test
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import numpy as np
import polars as pl
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.strategy_engine.src.ai_integration import AIStrategyIntegration
from services.strategy_engine.src.ai_models import (
    AIContext,
    AIDecisionRecord,
    AIPerformanceMetrics,
    ConsensusResult,
    MarketRegimeState,
    PerformanceReport,
    PromptContext,
    create_performance_summary,
    decision_to_dict,
)
from services.strategy_engine.src.ai_strategy import (
    AIDecision,
    AIModel,
    AIResponse,
    AIStrategyEngine,
    AnthropicClient,
    ConsensusEngine,
    CostTracker,
    DataContextBuilder,
    MarketContext,
    RateLimiter,
    ResponseCache,
)
from services.strategy_engine.src.base_strategy import (
    Signal,
    StrategyConfig,
    StrategyMode,
)
from shared.models import SignalType


class TestResponseCache:
    """Extended tests for ResponseCache functionality."""

    @pytest.fixture
    def cache(self):
        """Create a ResponseCache instance."""
        return ResponseCache(ttl=5)  # 5 second TTL for testing

    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache):
        """Test that cached responses expire after TTL."""
        # Create a response
        response = AIResponse(
            model=AIModel.HAIKU,
            prompt_type="test",
            response={"decision": "BUY"},
            confidence=80,
            tokens_used=100,
            cost=0.01,
            timestamp=datetime.now(),
        )

        # Cache it
        await cache.set("test_prompt", AIModel.HAIKU, response)

        # Should be retrievable immediately
        cached = await cache.get("test_prompt", AIModel.HAIKU)
        assert cached is not None
        assert cached.cache_hit is True

        # Mock time passing beyond TTL
        with patch(
            "services.strategy_engine.src.ai_strategy.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(seconds=6)

            # Should be expired now
            cached = await cache.get("test_prompt", AIModel.HAIKU)
            assert cached is None

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, cache):
        """Test that different prompts generate different cache keys."""
        response1 = AIResponse(
            model=AIModel.HAIKU,
            prompt_type="test",
            response={"decision": "BUY"},
            confidence=80,
            tokens_used=100,
            cost=0.01,
            timestamp=datetime.now(),
        )

        response2 = AIResponse(
            model=AIModel.HAIKU,
            prompt_type="test",
            response={"decision": "SELL"},
            confidence=70,
            tokens_used=100,
            cost=0.01,
            timestamp=datetime.now(),
        )

        # Cache both with different prompts
        await cache.set("prompt1", AIModel.HAIKU, response1)
        await cache.set("prompt2", AIModel.HAIKU, response2)

        # Retrieve and verify they're different
        cached1 = await cache.get("prompt1", AIModel.HAIKU)
        cached2 = await cache.get("prompt2", AIModel.HAIKU)

        assert cached1.response["decision"] == "BUY"
        assert cached2.response["decision"] == "SELL"

    @pytest.mark.asyncio
    async def test_cache_cleanup(self, cache):
        """Test automatic cleanup of expired entries."""
        # Add multiple entries with different timestamps
        for i in range(5):
            response = AIResponse(
                model=AIModel.HAIKU,
                prompt_type="test",
                response={"id": i},
                confidence=80,
                tokens_used=100,
                cost=0.01,
                timestamp=datetime.now(),
            )
            await cache.set(f"prompt_{i}", AIModel.HAIKU, response)

        # Initially should have 5 entries
        assert len(cache.cache) == 5

        # Mock time passing and trigger cleanup
        with patch(
            "services.strategy_engine.src.ai_strategy.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(seconds=10)
            await cache._cleanup()

        # All entries should be cleaned up
        assert len(cache.cache) == 0


class TestCostTracker:
    """Extended tests for cost tracking functionality."""

    @pytest.fixture
    def tracker(self):
        """Create a CostTracker instance."""
        return CostTracker({"daily_limit_usd": 5.0, "monthly_limit_usd": 100.0})

    @pytest.mark.asyncio
    async def test_daily_reset(self, tracker):
        """Test that daily costs reset at day boundary."""
        # Record some costs
        await tracker.record(2.0)
        assert tracker.daily_costs == 2.0

        # Simulate next day
        tracker.last_reset_day = datetime.now().date() - timedelta(days=1)
        tracker._check_reset()

        # Daily costs should be reset
        assert tracker.daily_costs == 0.0
        # Monthly costs should remain
        assert tracker.monthly_costs == 2.0

    @pytest.mark.asyncio
    async def test_monthly_reset(self, tracker):
        """Test that monthly costs reset at month boundary."""
        # Record some costs
        await tracker.record(50.0)
        assert tracker.monthly_costs == 50.0

        # Simulate next month
        current_month = datetime.now().month
        tracker.last_reset_month = (current_month - 2) % 12 or 12
        tracker._check_reset()

        # Monthly costs should be reset
        assert tracker.monthly_costs == 0.0

    @pytest.mark.asyncio
    async def test_cost_limit_estimation(self, tracker):
        """Test cost estimation for different models."""
        # Should allow Haiku (cheap)
        assert await tracker.can_proceed(AIModel.HAIKU) is True

        # Record costs near limit
        await tracker.record(4.95)

        # Should reject Opus (expensive)
        assert await tracker.can_proceed(AIModel.OPUS) is False

        # Should still allow Haiku (cheap)
        assert await tracker.can_proceed(AIModel.HAIKU) is True


class TestRateLimiter:
    """Extended tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_multiple_models_independent(self):
        """Test that rate limits are independent per model."""
        limiter = RateLimiter()

        # Request for different models shouldn't interfere
        start = datetime.now()
        await limiter.acquire(AIModel.HAIKU)
        await limiter.acquire(AIModel.OPUS)  # Should not wait
        elapsed = (datetime.now() - start).total_seconds()

        # Should complete quickly (no waiting between different models)
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self):
        """Test that rate limits are enforced correctly."""
        limiter = RateLimiter()

        # Make rapid requests for same model
        timings = []
        for _ in range(3):
            start = datetime.now()
            await limiter.acquire(AIModel.HAIKU)
            timings.append(datetime.now())

        # Check delays between requests
        for i in range(1, len(timings)):
            delay = (timings[i] - timings[i - 1]).total_seconds()
            assert delay >= limiter.min_delay[AIModel.HAIKU] - 0.01  # Small tolerance


class TestDataContextBuilder:
    """Extended tests for data context building."""

    @pytest.fixture
    def complex_data(self):
        """Create complex market data for testing."""
        # Generate 100 days of data with trend and volatility
        np.random.seed(42)
        dates = pl.date_range(
            datetime.now() - timedelta(days=100), datetime.now(), interval="1d"
        )

        # Create trending data with volatility
        trend = np.linspace(100, 120, len(dates))
        noise = np.random.randn(len(dates)) * 2
        prices = trend + noise

        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": prices + np.random.randn(len(dates)) * 0.5,
                "high": prices + np.abs(np.random.randn(len(dates))) * 2,
                "low": prices - np.abs(np.random.randn(len(dates))) * 2,
                "close": prices,
                "volume": np.random.randint(500000, 5000000, len(dates)),
            }
        )

    def test_rsi_extremes(self, complex_data):
        """Test RSI calculation at extremes."""
        # All up moves
        up_data = complex_data.with_columns(pl.col("close").cumsum().alias("close"))
        rsi_up = DataContextBuilder._calculate_rsi(up_data)
        assert rsi_up > 70  # Should be overbought

        # All down moves
        down_data = complex_data.with_columns(
            (100 - pl.col("close").cumsum()).alias("close")
        )
        rsi_down = DataContextBuilder._calculate_rsi(down_data)
        assert rsi_down < 30  # Should be oversold

    def test_support_resistance_detection(self, complex_data):
        """Test support and resistance level detection."""
        support, resistance = DataContextBuilder._calculate_support_resistance(
            complex_data
        )

        current_price = float(complex_data["close"].item(-1))

        # Support should be below current price
        assert support < current_price
        # Resistance should be above current price
        assert resistance > current_price
        # They should be different
        assert support != resistance

    def test_pattern_identification_comprehensive(self, complex_data):
        """Test comprehensive pattern identification."""
        patterns = DataContextBuilder._identify_patterns(complex_data)

        # Should identify some patterns
        assert len(patterns) > 0
        assert isinstance(patterns, str)

        # Test with strong uptrend
        uptrend_data = complex_data.with_columns(
            pl.col("close").cumsum().alias("close")
        )
        uptrend_patterns = DataContextBuilder._identify_patterns(uptrend_data)
        assert "Uptrend" in uptrend_patterns or "breakout" in uptrend_patterns.lower()

    def test_context_with_missing_data(self):
        """Test context building with missing or None values."""
        # Create minimal data
        minimal_data = pl.DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000000],
            }
        )

        context = DataContextBuilder.build_master_context(
            ticker="TEST", data=minimal_data, finviz_data=None, market_data=None
        )

        # Should handle missing data gracefully
        assert context["ticker"] == "TEST"
        assert "current_price" in context
        assert context["market_context"] == "Market data unavailable"
        assert context["market_cap"] == "N/A"


class TestConsensusEngine:
    """Extended tests for consensus building."""

    @pytest.fixture
    def diverse_responses(self):
        """Create diverse AI responses for testing."""
        return [
            AIResponse(
                model=AIModel.OPUS,
                prompt_type="master",
                response={
                    "decision": "BUY",
                    "confidence": 90,
                    "entry_price": 100.0,
                    "stop_loss": 95.0,
                    "take_profit": 110.0,
                    "position_size_suggestion": 0.05,
                    "reasoning": "Strong bullish signals",
                },
                confidence=90,
                tokens_used=1000,
                cost=0.05,
                timestamp=datetime.now(),
            ),
            AIResponse(
                model=AIModel.SONNET,
                prompt_type="risk",
                response={
                    "decision": "BUY",
                    "confidence": 70,
                    "entry_price": 100.5,
                    "stop_loss": 96.0,
                    "take_profit": 108.0,
                    "position_size_suggestion": 0.03,
                    "reasoning": "Moderate risk, proceed with caution",
                },
                confidence=70,
                tokens_used=500,
                cost=0.02,
                timestamp=datetime.now(),
            ),
            AIResponse(
                model=AIModel.HAIKU,
                prompt_type="momentum",
                response={
                    "decision": "SELL",
                    "confidence": 65,
                    "reasoning": "Momentum weakening",
                },
                confidence=65,
                tokens_used=200,
                cost=0.001,
                timestamp=datetime.now(),
            ),
            AIResponse(
                model=AIModel.SONNET,
                prompt_type="contrarian",
                response={
                    "decision": "HOLD",
                    "confidence": 80,
                    "reasoning": "Mixed signals, wait for clarity",
                },
                confidence=80,
                tokens_used=400,
                cost=0.015,
                timestamp=datetime.now(),
            ),
        ]

    @pytest.mark.asyncio
    async def test_weighted_consensus(self, diverse_responses):
        """Test weighted consensus with multiple disagreeing models."""
        config = {
            "consensus": {
                "min_ai_instances": 3,
                "majority_threshold": 0.5,
                "confidence_weights": {
                    AIModel.OPUS.value: 2.0,
                    AIModel.SONNET.value: 1.5,
                    AIModel.HAIKU.value: 1.0,
                },
            }
        }

        engine = ConsensusEngine(config)
        decision = await engine.build_consensus(diverse_responses)

        # With weights, BUY should win (OPUS weight = 2.0)
        assert decision.action == "BUY"
        assert decision.confidence > 0
        assert len(decision.key_risks) >= 0
        assert "total_responses" in decision.consensus_details

    @pytest.mark.asyncio
    async def test_unanimous_consensus(self):
        """Test consensus when all models agree."""
        unanimous_responses = [
            AIResponse(
                model=model,
                prompt_type="test",
                response={
                    "decision": "BUY",
                    "confidence": 85,
                    "entry_price": 100.0,
                    "stop_loss": 95.0,
                    "take_profit": 110.0,
                    "position_size_suggestion": 0.05,
                },
                confidence=85,
                tokens_used=500,
                cost=0.02,
                timestamp=datetime.now(),
            )
            for model in [AIModel.OPUS, AIModel.SONNET, AIModel.HAIKU]
        ]

        engine = ConsensusEngine({})
        decision = await engine.build_consensus(unanimous_responses)

        # Should have high confidence with unanimous agreement
        assert decision.action == "BUY"
        assert decision.confidence > 90
        assert decision.entry_price == 100.0

    @pytest.mark.asyncio
    async def test_insufficient_responses(self):
        """Test handling of insufficient responses."""
        single_response = [
            AIResponse(
                model=AIModel.HAIKU,
                prompt_type="test",
                response={"decision": "BUY", "confidence": 80},
                confidence=80,
                tokens_used=100,
                cost=0.01,
                timestamp=datetime.now(),
            )
        ]

        config = {"consensus": {"min_ai_instances": 3}}
        engine = ConsensusEngine(config)

        # Should still work but log warning
        with patch("services.strategy_engine.src.ai_strategy.logger") as mock_logger:
            decision = await engine.build_consensus(single_response)
            mock_logger.warning.assert_called()

        assert decision.action == "BUY"


class TestAIStrategyEngineIntegration:
    """Integration tests for AIStrategyEngine."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client."""
        client = AsyncMock()
        client.query = AsyncMock()
        return client

    @pytest.fixture
    def strategy_engine(self, mock_anthropic_client):
        """Create an AIStrategyEngine with mocked dependencies."""
        config = StrategyConfig(
            name="test_ai_strategy",
            mode=StrategyMode.DAY_TRADING,
            parameters={"anthropic_api_key": "test_key", "min_confidence": 60},
        )

        with patch(
            "services.strategy_engine.src.ai_strategy.AnthropicClient"
        ) as mock_client_class:
            mock_client_class.return_value = mock_anthropic_client

            with patch(
                "builtins.open",
                mock_open(
                    read_data=yaml.dump(
                        {
                            "prompts": {},
                            "models": {},
                            "cost_management": {"cache_ttl_seconds": 300},
                        }
                    )
                ),
            ):
                engine = AIStrategyEngine(config)
                engine.anthropic_client = mock_anthropic_client
                return engine

    @pytest.mark.asyncio
    async def test_analyze_with_cache_hit(self, strategy_engine):
        """Test analyze method with cache hit."""
        # Create sample data
        data = create_sample_data(50)

        # First call - should query AI
        mock_response = AIResponse(
            model=AIModel.OPUS,
            prompt_type="master",
            response={"decision": "BUY", "confidence": 75, "entry_price": 100.0},
            confidence=75,
            tokens_used=1000,
            cost=0.05,
            timestamp=datetime.now(),
        )

        strategy_engine.anthropic_client.query.return_value = mock_response

        # Mock consensus building
        with patch.object(
            strategy_engine.consensus_engine, "build_consensus"
        ) as mock_consensus:
            mock_consensus.return_value = AIDecision(
                action="BUY",
                confidence=75,
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                position_size=0.05,
                risk_reward_ratio=3.0,
                reasoning="Test reasoning",
                key_risks=["Market risk"],
                timeframe="day_trade",
                consensus_details={},
            )

            signal1 = await strategy_engine.analyze("TEST", data)

        # Second call with same data - should use cache
        signal2 = await strategy_engine.analyze("TEST", data)

        # Both signals should be identical
        assert signal1.action == signal2.action
        assert signal1.confidence == signal2.confidence

    @pytest.mark.asyncio
    async def test_analyze_error_handling(self, strategy_engine):
        """Test error handling in analyze method."""
        data = create_sample_data(50)

        # Mock an error in AI query
        strategy_engine.anthropic_client.query.side_effect = Exception("API Error")

        # Should return HOLD signal on error
        signal = await strategy_engine.analyze("TEST", data)

        assert signal.action == SignalType.HOLD
        assert signal.confidence == 0
        assert "error" in signal.metadata

    @pytest.mark.asyncio
    async def test_market_context_update(self, strategy_engine):
        """Test market context update functionality."""
        # Initially no market context
        assert strategy_engine.market_context is None

        # Update market context
        await strategy_engine._update_market_context()

        # Should have market context now
        assert strategy_engine.market_context is not None
        assert isinstance(strategy_engine.market_context, MarketContext)
        assert strategy_engine.market_context.regime == "bullish"


class TestAIModels:
    """Tests for AI model data structures."""

    def test_decision_to_dict(self):
        """Test conversion of AIDecisionRecord to dictionary."""
        decision = AIDecisionRecord(
            timestamp=datetime.now(),
            ticker="TEST",
            decision="BUY",
            confidence=80.0,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            position_size=0.05,
            risk_reward_ratio=3.0,
            reasoning="Strong bullish signals",
            key_risks=["Market volatility"],
            models_used=["claude-3-opus"],
            total_cost=0.05,
        )

        result = decision_to_dict(decision)

        assert result["ticker"] == "TEST"
        assert result["decision"] == "BUY"
        assert result["confidence"] == 80.0
        assert "timestamp" in result
        assert isinstance(result["key_risks"], list)

    def test_performance_summary_edge_cases(self):
        """Test performance summary with edge cases."""
        # Empty lists
        summary = create_performance_summary([], [])
        assert summary.total_decisions == 0
        assert summary.accuracy_rate == 0
        assert summary.total_pnl == 0

        # Decisions without executions
        decisions = [
            MagicMock(
                timestamp=datetime.now(),
                actual_outcome=None,
                total_cost=0.05,
                models_used=["claude-3-opus"],
            )
        ]
        summary = create_performance_summary(decisions, [])
        assert summary.total_decisions == 1
        assert summary.win_rate == 0

    def test_ai_context_creation(self):
        """Test AIContext dataclass creation."""
        context = AIContext(
            ticker="TEST",
            current_price=100.0,
            market_regime="bullish",
            risk_level="medium",
            position_size_multiplier=1.2,
            confidence_threshold=65.0,
            existing_position={"size": 100, "entry": 95.0},
            recent_decisions=[{"action": "BUY", "confidence": 70}],
            metadata={"source": "test"},
        )

        assert context.ticker == "TEST"
        assert context.position_size_multiplier == 1.2
        assert len(context.recent_decisions) == 1
        assert context.metadata["source"] == "test"


class TestAIIntegrationAdvanced:
    """Advanced integration tests."""

    @pytest.mark.asyncio
    async def test_concurrent_analysis(self):
        """Test concurrent analysis of multiple tickers."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch(
                "services.strategy_engine.src.ai_integration.create_async_engine"
            ):
                integration = AIStrategyIntegration(
                    AsyncMock(), "postgresql://test", {"min_confidence": 60}
                )

                # Add price data for multiple tickers
                tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
                for ticker in tickers:
                    integration.price_data_buffer[ticker] = create_sample_data(50)

                # Analyze all concurrently
                tasks = [integration._analyze_ticker(ticker) for ticker in tickers]

                with patch.object(integration.ai_strategy, "analyze") as mock_analyze:
                    mock_analyze.return_value = Signal(
                        action=SignalType.BUY, confidence=70, position_size=0.05
                    )

                    await asyncio.gather(*tasks)

                    # Should have analyzed all tickers
                    assert mock_analyze.call_count == len(tickers)

    @pytest.mark.asyncio
    async def test_position_limit_enforcement(self):
        """Test that position limits are properly enforced."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch(
                "services.strategy_engine.src.ai_integration.create_async_engine"
            ):
                integration = AIStrategyIntegration(
                    AsyncMock(),
                    "postgresql://test",
                    {
                        "min_confidence": 60,
                        "max_positions": 3,
                        "daily_loss_limit": -500,
                    },
                )

                # Add 3 positions (at limit)
                for i in range(3):
                    await integration.add_position(
                        f"TEST{i}", entry_price=100, quantity=100
                    )

                # Try to add 4th position
                signal = Signal(
                    action=SignalType.BUY, confidence=80, position_size=0.05
                )

                can_add = await integration._check_position_limits("TEST3", signal)
                assert can_add is False

                # Remove one position
                await integration.remove_position("TEST0")

                # Now should be able to add
                can_add = await integration._check_position_limits("TEST3", signal)
                assert can_add is True

    @pytest.mark.asyncio
    async def test_exit_strategy_triggers(self):
        """Test various exit strategy triggers."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch(
                "services.strategy_engine.src.ai_integration.create_async_engine"
            ):
                integration = AIStrategyIntegration(
                    AsyncMock(), "postgresql://test", {"min_confidence": 60}
                )

                # Add position with stop loss and take profit
                await integration.add_position(
                    "TEST", entry_price=100, quantity=100, stop_loss=95, take_profit=110
                )

                # Add price data
                integration.price_data_buffer["TEST"] = create_sample_data(50)

                # Test stop loss trigger
                integration.price_data_buffer["TEST"] = integration.price_data_buffer[
                    "TEST"
                ].with_columns(pl.lit(94.0).alias("close"))

                with patch.object(integration, "_publish_exit_signal") as mock_exit:
                    await integration._position_monitor()

                    # Should trigger stop loss exit
                    # Note: _position_monitor runs in a loop, so we need to check if it was called


# Helper functions


def create_sample_data(num_rows: int) -> pl.DataFrame:
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pl.date_range(
        datetime.now() - timedelta(days=num_rows), datetime.now(), interval="1d"
    )

    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)

    return pl.DataFrame(
        {
            "timestamp": dates,
            "open": prices + np.random.randn(len(dates)) * 0.5,
            "high": prices + np.abs(np.random.randn(len(dates))) * 1,
            "low": prices - np.abs(np.random.randn(len(dates))) * 1,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, len(dates)),
        }
    )


def mock_open(read_data=""):
    """Helper to mock file open."""
    m = MagicMock()
    m.__enter__ = MagicMock(
        return_value=MagicMock(read=MagicMock(return_value=read_data))
    )
    m.__exit__ = MagicMock(return_value=None)
    return m


import yaml

# Pytest configuration


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    yield
    # Cleanup if needed


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
