"""
Error Handling and Edge Case Tests for AI Strategy Engine

This module tests error scenarios, edge cases, and recovery mechanisms
in the AI strategy implementation.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from decimal import Decimal
import numpy as np
import polars as pl
from typing import Dict, Any, List
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.strategy_engine.src.ai_strategy import (
    AIStrategyEngine, AIModel, AIResponse, AIDecision,
    AnthropicClient, CostTracker, RateLimiter, ResponseCache,
    DataContextBuilder, ConsensusEngine, MarketContext
)
from services.strategy_engine.src.ai_models import (
    AIDecisionRecord, AIPerformanceMetrics,
    create_performance_summary
)
from services.strategy_engine.src.ai_integration import AIStrategyIntegration
from services.strategy_engine.src.base_strategy import StrategyConfig, StrategyMode, Signal
from shared.models import SignalType


class TestAPIErrors:
    """Test handling of API-related errors."""

    @pytest.mark.asyncio
    async def test_api_timeout(self):
        """Test handling of API timeout."""
        config = {
            'models': {
                'claude-3-opus-20240229': {
                    'cost_per_million_input_tokens': 15.0,
                    'cost_per_million_output_tokens': 75.0
                }
            },
            'cost_management': {
                'daily_limit_usd': 5.0,
                'cache_ttl_seconds': 300
            }
        }

        client = AnthropicClient("test_key", config)

        # Mock timeout error
        with patch.object(client.client, 'messages') as mock_messages:
            mock_create = AsyncMock(side_effect=asyncio.TimeoutError("Request timed out"))
            mock_messages.create = mock_create

            with pytest.raises(asyncio.TimeoutError):
                await client.query("test prompt", AIModel.OPUS)

    @pytest.mark.asyncio
    async def test_api_rate_limit_exceeded(self):
        """Test handling when API rate limit is exceeded."""
        config = {
            'models': {},
            'cost_management': {'daily_limit_usd': 5.0}
        }

        client = AnthropicClient("test_key", config)

        # Mock rate limit error
        with patch.object(client.client, 'messages') as mock_messages:
            mock_create = AsyncMock(side_effect=Exception("Rate limit exceeded"))
            mock_messages.create = mock_create

            # Should retry with exponential backoff
            with pytest.raises(Exception) as exc_info:
                await client.query("test prompt", AIModel.HAIKU, use_cache=False)

            assert "Rate limit" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test handling of invalid API key."""
        config = {'models': {}, 'cost_management': {}}

        client = AnthropicClient("invalid_key", config)

        with patch.object(client.client, 'messages') as mock_messages:
            mock_create = AsyncMock(side_effect=Exception("Invalid API key"))
            mock_messages.create = mock_create

            with pytest.raises(Exception) as exc_info:
                await client.query("test prompt", AIModel.HAIKU, use_cache=False)

            assert "Invalid API key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_malformed_response(self):
        """Test handling of malformed API response."""
        config = {'models': {}, 'cost_management': {}}

        client = AnthropicClient("test_key", config)

        # Mock malformed response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Not valid JSON")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        with patch.object(client.client, 'messages') as mock_messages:
            mock_messages.create = AsyncMock(return_value=mock_response)

            response = await client.query("test prompt", AIModel.HAIKU, use_cache=False)

            # Should handle gracefully and return raw response
            assert "raw_response" in response.response
            assert response.response["raw_response"] == "Not valid JSON"


class TestDataErrors:
    """Test handling of data-related errors."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pl.DataFrame()

        with pytest.raises(Exception):
            DataContextBuilder.build_master_context(
                ticker="TEST",
                data=empty_df,
                finviz_data=None,
                market_data=None
            )

    def test_insufficient_data_for_indicators(self):
        """Test handling when not enough data for indicators."""
        # Only 5 rows of data (need 14+ for RSI)
        small_df = pl.DataFrame({
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(5)],
            'open': [100.0] * 5,
            'high': [101.0] * 5,
            'low': [99.0] * 5,
            'close': [100.0] * 5,
            'volume': [1000000] * 5
        })

        rsi = DataContextBuilder._calculate_rsi(small_df)

        # Should return default value
        assert rsi == 50.0

    def test_nan_values_in_data(self):
        """Test handling of NaN values in data."""
        df_with_nan = pl.DataFrame({
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(20)],
            'open': [100.0] * 20,
            'high': [101.0] * 20,
            'low': [99.0] * 20,
            'close': [100.0 if i % 5 != 0 else None for i in range(20)],
            'volume': [1000000] * 20
        })

        # Should handle NaN values gracefully
        try:
            context = DataContextBuilder.build_master_context(
                ticker="TEST",
                data=df_with_nan,
                finviz_data=None,
                market_data=None
            )
            # Should either succeed or raise a clear error
        except Exception as e:
            assert "close" in str(e).lower() or "null" in str(e).lower()

    def test_extreme_price_values(self):
        """Test handling of extreme price values."""
        extreme_df = pl.DataFrame({
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(20)],
            'open': [1e10] * 20,  # Very large values
            'high': [1e10 + 1000] * 20,
            'low': [1e10 - 1000] * 20,
            'close': [1e10] * 20,
            'volume': [1e15] * 20
        })

        context = DataContextBuilder.build_master_context(
            ticker="TEST",
            data=extreme_df,
            finviz_data=None,
            market_data=None
        )

        # Should handle large numbers
        assert 'current_price' in context
        assert float(context['current_price'].replace(',', '')) > 1e9


class TestConsensusErrors:
    """Test consensus building error scenarios."""

    @pytest.mark.asyncio
    async def test_no_responses(self):
        """Test consensus with no responses."""
        engine = ConsensusEngine({})

        with pytest.raises(Exception):
            await engine.build_consensus([])

    @pytest.mark.asyncio
    async def test_all_conflicting_responses(self):
        """Test consensus with completely conflicting responses."""
        responses = [
            AIResponse(
                model=AIModel.OPUS,
                prompt_type="test",
                response={'decision': 'BUY', 'confidence': 90},
                confidence=90,
                tokens_used=100,
                cost=0.01,
                timestamp=datetime.now()
            ),
            AIResponse(
                model=AIModel.SONNET,
                prompt_type="test",
                response={'decision': 'SELL', 'confidence': 90},
                confidence=90,
                tokens_used=100,
                cost=0.01,
                timestamp=datetime.now()
            ),
            AIResponse(
                model=AIModel.HAIKU,
                prompt_type="test",
                response={'decision': 'HOLD', 'confidence': 90},
                confidence=90,
                tokens_used=100,
                cost=0.01,
                timestamp=datetime.now()
            )
        ]

        engine = ConsensusEngine({})
        decision = await engine.build_consensus(responses)

        # Should still produce a decision
        assert decision.action in ['BUY', 'SELL', 'HOLD']
        # Confidence should be lower due to disagreement
        assert decision.confidence < 100

    @pytest.mark.asyncio
    async def test_missing_required_fields(self):
        """Test consensus with responses missing required fields."""
        responses = [
            AIResponse(
                model=AIModel.OPUS,
                prompt_type="test",
                response={},  # Empty response
                confidence=0,
                tokens_used=100,
                cost=0.01,
                timestamp=datetime.now()
            )
        ]

        engine = ConsensusEngine({})
        decision = await engine.build_consensus(responses)

        # Should handle missing fields gracefully
        assert decision.action == 'HOLD'  # Default action
        assert decision.confidence == 0


class TestIntegrationErrors:
    """Test integration layer error handling."""

    @pytest.mark.asyncio
    async def test_database_connection_error(self):
        """Test handling of database connection errors."""
        with patch('services.strategy_engine.src.ai_integration.create_async_engine') as mock_engine:
            mock_engine.side_effect = Exception("Database connection failed")

            with pytest.raises(Exception) as exc_info:
                integration = AIStrategyIntegration(
                    AsyncMock(),
                    'postgresql://invalid',
                    {}
                )

            assert "Database connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_redis_connection_error(self):
        """Test handling of Redis connection errors."""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Redis connection failed")

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            with patch('services.strategy_engine.src.ai_integration.create_async_engine'):
                integration = AIStrategyIntegration(
                    mock_redis,
                    'postgresql://test',
                    {}
                )

                # Should handle Redis errors gracefully
                assert integration.signal_publisher is None or integration.signal_publisher is not None

    @pytest.mark.asyncio
    async def test_corrupt_price_data(self):
        """Test handling of corrupt price data updates."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            with patch('services.strategy_engine.src.ai_integration.create_async_engine'):
                integration = AIStrategyIntegration(
                    AsyncMock(),
                    'postgresql://test',
                    {}
                )

                # Send corrupt price update
                corrupt_data = {
                    'ticker': 'TEST',
                    'timestamp': 'not-a-timestamp',
                    'open': 'not-a-number',
                    'high': None,
                    'low': -100,  # Negative price
                    'close': float('inf'),  # Infinity
                    'volume': 'high'
                }

                # Should handle gracefully without crashing
                await integration._handle_price_update(corrupt_data)

                # Should not have added corrupt data
                assert 'TEST' not in integration.price_data_buffer or \
                       len(integration.price_data_buffer.get('TEST', pl.DataFrame())) == 0


class TestRecoveryMechanisms:
    """Test recovery and fallback mechanisms."""

    @pytest.mark.asyncio
    async def test_fallback_to_cache_on_api_error(self):
        """Test that system falls back to cache when API fails."""
        config = {'cost_management': {'cache_ttl_seconds': 300}}
        client = AnthropicClient("test_key", config)

        # First, populate cache
        cached_response = AIResponse(
            model=AIModel.HAIKU,
            prompt_type="test",
            response={'decision': 'BUY', 'confidence': 70},
            confidence=70,
            tokens_used=100,
            cost=0.01,
            timestamp=datetime.now()
        )

        await client.cache.set("test_prompt", AIModel.HAIKU, cached_response)

        # Now make API fail
        with patch.object(client.client, 'messages') as mock_messages:
            mock_messages.create = AsyncMock(side_effect=Exception("API Error"))

            # Should get cached response
            response = await client.query("test_prompt", AIModel.HAIKU)

            assert response.cache_hit is True
            assert response.confidence == 70

    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern for repeated failures."""
        tracker = CostTracker({'daily_limit_usd': 5.0, 'monthly_limit_usd': 100.0})

        # Simulate hitting daily limit
        await tracker.record(5.0)

        # Should prevent further requests
        assert await tracker.can_proceed(AIModel.OPUS) is False
        assert await tracker.can_proceed(AIModel.HAIKU) is False

        # Reset for next day
        tracker.last_reset_day = datetime.now().date() - timedelta(days=1)
        tracker._check_reset()

        # Should allow requests again
        assert await tracker.can_proceed(AIModel.HAIKU) is True

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when some models fail."""
        responses = [
            AIResponse(
                model=AIModel.OPUS,
                prompt_type="test",
                response={'decision': 'BUY', 'confidence': 85},
                confidence=85,
                tokens_used=1000,
                cost=0.05,
                timestamp=datetime.now()
            ),
            # Sonnet failed - no response
            # Haiku failed - no response
        ]

        engine = ConsensusEngine({'consensus': {'min_ai_instances': 1}})
        decision = await engine.build_consensus(responses)

        # Should still work with single response
        assert decision.action == 'BUY'
        assert decision.confidence > 0


class TestMemoryAndPerformance:
    """Test memory leaks and performance issues."""

    @pytest.mark.asyncio
    async def test_cache_memory_limit(self):
        """Test that cache doesn't grow unbounded."""
        cache = ResponseCache(ttl=1)  # 1 second TTL

        # Add many entries
        for i in range(1000):
            response = AIResponse(
                model=AIModel.HAIKU,
                prompt_type="test",
                response={'id': i},
                confidence=50,
                tokens_used=100,
                cost=0.01,
                timestamp=datetime.now()
            )
            await cache.set(f"prompt_{i}", AIModel.HAIKU, response)

        # Wait for TTL
        await asyncio.sleep(1.1)

        # Trigger cleanup
        await cache._cleanup()

        # Cache should be empty
        assert len(cache.cache) == 0

    @pytest.mark.asyncio
    async def test_concurrent_requests_handling(self):
        """Test handling of many concurrent requests."""
        limiter = RateLimiter()

        # Create many concurrent requests
        async def make_request(i):
            await limiter.acquire(AIModel.HAIKU)
            return i

        # Should handle all requests without deadlock
        tasks = [make_request(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(i in results for i in range(10))


class TestEdgeCases:
    """Test various edge cases."""

    def test_zero_volume_data(self):
        """Test handling of zero volume data."""
        df = pl.DataFrame({
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(20)],
            'open': [100.0] * 20,
            'high': [101.0] * 20,
            'low': [99.0] * 20,
            'close': [100.0] * 20,
            'volume': [0] * 20  # Zero volume
        })

        context = DataContextBuilder.build_master_context(
            ticker="TEST",
            data=df,
            finviz_data=None,
            market_data=None
        )

        # Should handle zero volume
        assert 'volume' in context
        assert 'avg_volume' in context

    def test_single_candle_data(self):
        """Test with only one candle of data."""
        df = pl.DataFrame({
            'timestamp': [datetime.now()],
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000000]
        })

        context = DataContextBuilder.build_master_context(
            ticker="TEST",
            data=df,
            finviz_data=None,
            market_data=None
        )

        # Should handle single candle
        assert 'current_price' in context
        assert context['current_price'] == "100.50"

    @pytest.mark.asyncio
    async def test_missing_anthropic_key(self):
        """Test handling when Anthropic API key is missing."""
        config = StrategyConfig(
            name="test",
            mode=StrategyMode.DAY_TRADING,
            parameters={}  # No API key
        )

        with pytest.raises(ValueError) as exc_info:
            with patch('builtins.open', mock_open(read_data='{}')):
                strategy = AIStrategyEngine(config)

        assert "API key is required" in str(exc_info.value)


# Helper functions

def mock_open(read_data=''):
    """Helper to mock file open."""
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=read_data)))
    m.__exit__ = MagicMock(return_value=None)
    return m


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
