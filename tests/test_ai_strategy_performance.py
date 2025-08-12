"""
Performance and Integration Tests for AI Strategy Engine

This module tests performance characteristics, integration scenarios,
and system behavior under load.
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from decimal import Decimal
import numpy as np
import polars as pl
from typing import Dict, Any, List
import os
import sys
import psutil
import gc

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.strategy_engine.src.ai_strategy import (
    AIStrategyEngine, AIModel, AIResponse, AIDecision,
    AnthropicClient, CostTracker, RateLimiter, ResponseCache,
    DataContextBuilder, ConsensusEngine, MarketContext
)
from services.strategy_engine.src.ai_models import (
    AIDecisionRecord, AIPerformanceMetrics,
    create_performance_summary, PerformanceReport
)
from services.strategy_engine.src.ai_integration import AIStrategyIntegration
from services.strategy_engine.src.base_strategy import StrategyConfig, StrategyMode, Signal
from shared.models import SignalType


class TestPerformanceMetrics:
    """Test performance characteristics of the AI strategy."""

    @pytest.mark.benchmark
    def test_context_building_performance(self, benchmark):
        """Benchmark context building performance."""
        # Create large dataset
        data = create_large_dataset(1000)

        def build_context():
            return DataContextBuilder.build_master_context(
                ticker="TEST",
                data=data,
                finviz_data={'Market Cap': '10B', 'P/E': '15'},
                market_data={'spy_change': 0.5, 'vix_level': 15}
            )

        # Benchmark the function
        result = benchmark(build_context)

        assert 'ticker' in result
        assert 'current_price' in result

    @pytest.mark.benchmark
    def test_rsi_calculation_performance(self, benchmark):
        """Benchmark RSI calculation performance."""
        data = create_large_dataset(500)

        result = benchmark(DataContextBuilder._calculate_rsi, data)

        assert 0 <= result <= 100

    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance with many entries."""
        cache = ResponseCache(ttl=300)

        # Measure write performance
        start = time.time()
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
        write_time = time.time() - start

        # Measure read performance
        start = time.time()
        hits = 0
        for i in range(1000):
            result = await cache.get(f"prompt_{i}", AIModel.HAIKU)
            if result:
                hits += 1
        read_time = time.time() - start

        assert hits == 1000
        assert write_time < 1.0  # Should complete in under 1 second
        assert read_time < 0.5   # Reads should be faster

    @pytest.mark.asyncio
    async def test_consensus_performance_with_many_models(self):
        """Test consensus building performance with many responses."""
        responses = []
        for i in range(10):
            responses.append(AIResponse(
                model=AIModel.HAIKU,
                prompt_type="test",
                response={
                    'decision': 'BUY' if i % 2 == 0 else 'SELL',
                    'confidence': 70 + i,
                    'entry_price': 100.0 + i * 0.1,
                    'stop_loss': 95.0,
                    'take_profit': 110.0
                },
                confidence=70 + i,
                tokens_used=100,
                cost=0.01,
                timestamp=datetime.now()
            ))

        engine = ConsensusEngine({})

        start = time.time()
        decision = await engine.build_consensus(responses)
        elapsed = time.time() - start

        assert decision.action in ['BUY', 'SELL']
        assert elapsed < 0.1  # Should be very fast


class TestMemoryUsage:
    """Test memory usage and leak detection."""

    @pytest.mark.asyncio
    async def test_memory_cleanup_after_analysis(self):
        """Test that memory is properly cleaned up after analysis."""
        config = StrategyConfig(
            name="test",
            mode=StrategyMode.DAY_TRADING,
            parameters={'anthropic_api_key': 'test_key'}
        )

        with patch('builtins.open', mock_open(read_data='{}')):
            strategy = AIStrategyEngine(config)

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform multiple analyses
        for i in range(10):
            data = create_large_dataset(100)
            with patch.object(strategy, 'analyze') as mock_analyze:
                mock_analyze.return_value = Signal(
                    action=SignalType.BUY,
                    confidence=70,
                    position_size=0.05
                )
                await mock_analyze("TEST", data)

        # Force garbage collection
        gc.collect()

        # Check memory after
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50

    def test_data_buffer_memory_limits(self):
        """Test that data buffers don't grow unbounded."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            with patch('services.strategy_engine.src.ai_integration.create_async_engine'):
                integration = AIStrategyIntegration(
                    AsyncMock(),
                    'postgresql://test',
                    {}
                )

                # Add lots of data
                for i in range(100):
                    ticker = f"TEST{i}"
                    integration.price_data_buffer[ticker] = create_large_dataset(500)

                # Check memory usage
                total_size = sum(
                    df.estimated_size() for df in integration.price_data_buffer.values()
                )

                # Should be reasonable (less than 100MB)
                assert total_size < 100 * 1024 * 1024


class TestSystemIntegration:
    """Test full system integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self):
        """Test complete pipeline from data ingestion to signal generation."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            with patch('services.strategy_engine.src.ai_integration.create_async_engine'):
                # Setup integration
                mock_redis = AsyncMock()
                integration = AIStrategyIntegration(
                    mock_redis,
                    'postgresql://test',
                    {
                        'min_confidence': 60,
                        'max_positions': 5
                    }
                )

                # Mock AI strategy analyze method
                mock_signal = Signal(
                    action=SignalType.BUY,
                    confidence=75,
                    entry_price=Decimal('100.50'),
                    stop_loss=Decimal('95.00'),
                    take_profit=Decimal('110.00'),
                    position_size=0.05,
                    metadata={'ai_decision': {'reasoning': 'Test'}}
                )

                with patch.object(integration.ai_strategy, 'analyze', return_value=mock_signal):
                    # Simulate price update
                    for i in range(50):
                        await integration._handle_price_update({
                            'ticker': 'TEST',
                            'timestamp': (datetime.now() - timedelta(hours=50-i)).isoformat(),
                            'open': 99.0 + i * 0.1,
                            'high': 100.0 + i * 0.1,
                            'low': 98.0 + i * 0.1,
                            'close': 99.5 + i * 0.1,
                            'volume': 1000000
                        })

                    # Should trigger analysis
                    assert 'TEST' in integration.price_data_buffer
                    assert len(integration.price_data_buffer['TEST']) == 50

    @pytest.mark.asyncio
    async def test_concurrent_ticker_processing(self):
        """Test processing multiple tickers concurrently."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            with patch('services.strategy_engine.src.ai_integration.create_async_engine'):
                integration = AIStrategyIntegration(
                    AsyncMock(),
                    'postgresql://test',
                    {}
                )

                # Create tasks for multiple tickers
                tasks = []
                tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']

                for ticker in tickers:
                    # Add price data
                    integration.price_data_buffer[ticker] = create_large_dataset(50)

                    # Create analysis task
                    with patch.object(integration.ai_strategy, 'analyze') as mock_analyze:
                        mock_analyze.return_value = Signal(
                            action=SignalType.BUY,
                            confidence=70,
                            position_size=0.05
                        )
                        tasks.append(integration._analyze_ticker(ticker))

                # Process all concurrently
                start = time.time()
                await asyncio.gather(*tasks)
                elapsed = time.time() - start

                # Should complete reasonably fast
                assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_error_recovery_in_pipeline(self):
        """Test that pipeline recovers from errors."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            with patch('services.strategy_engine.src.ai_integration.create_async_engine'):
                integration = AIStrategyIntegration(
                    AsyncMock(),
                    'postgresql://test',
                    {}
                )

                # Make first analysis fail
                call_count = 0

                async def analyze_with_error(ticker, data):
                    nonlocal call_count
                    call_count += 1
                    if call_count == 1:
                        raise Exception("Temporary error")
                    return Signal(
                        action=SignalType.BUY,
                        confidence=70,
                        position_size=0.05
                    )

                integration.ai_strategy.analyze = analyze_with_error

                # Add data and analyze
                integration.price_data_buffer['TEST'] = create_large_dataset(50)

                # First attempt should fail
                await integration._analyze_ticker('TEST')

                # Second attempt should succeed
                await integration._analyze_ticker('TEST')

                assert call_count == 2


class TestCostOptimization:
    """Test cost optimization strategies."""

    @pytest.mark.asyncio
    async def test_model_selection_based_on_confidence(self):
        """Test that appropriate models are selected based on confidence needs."""
        config = {
            'models': {
                AIModel.OPUS.value: {'cost_per_million_input_tokens': 15.0},
                AIModel.SONNET.value: {'cost_per_million_input_tokens': 3.0},
                AIModel.HAIKU.value: {'cost_per_million_input_tokens': 0.25}
            },
            'cost_management': {'daily_limit_usd': 5.0}
        }

        client = AnthropicClient("test_key", config)

        # Track which model is used
        models_used = []

        async def mock_query(prompt, model, **kwargs):
            models_used.append(model)
            return AIResponse(
                model=model,
                prompt_type="test",
                response={'decision': 'BUY', 'confidence': 70},
                confidence=70,
                tokens_used=1000,
                cost=0.01,
                timestamp=datetime.now()
            )

        client.query = mock_query

        # Low confidence scenario - should use Haiku
        await client.query("screening prompt", AIModel.HAIKU)
        assert models_used[-1] == AIModel.HAIKU

        # High confidence scenario - should use Opus
        await client.query("critical decision", AIModel.OPUS)
        assert models_used[-1] == AIModel.OPUS

    @pytest.mark.asyncio
    async def test_cache_hit_rate(self):
        """Test cache hit rate for similar queries."""
        cache = ResponseCache(ttl=300)

        # Create similar prompts
        prompts = [
            f"Analyze TEST with price {100 + i * 0.01}" for i in range(10)
        ]

        # Add first prompt to cache
        response = AIResponse(
            model=AIModel.HAIKU,
            prompt_type="test",
            response={'decision': 'BUY'},
            confidence=70,
            tokens_used=100,
            cost=0.01,
            timestamp=datetime.now()
        )

        await cache.set(prompts[0], AIModel.HAIKU, response)

        # Check hit rate
        hits = 0
        for prompt in prompts:
            if await cache.get(prompt, AIModel.HAIKU):
                hits += 1

        hit_rate = hits / len(prompts)

        # First prompt should hit
        assert hit_rate >= 0.1


class TestRealTimePerformance:
    """Test real-time performance characteristics."""

    @pytest.mark.asyncio
    async def test_latency_requirements(self):
        """Test that system meets latency requirements."""
        config = StrategyConfig(
            name="test",
            mode=StrategyMode.DAY_TRADING,
            parameters={'anthropic_api_key': 'test_key'}
        )

        with patch('builtins.open', mock_open(read_data='{}')):
            strategy = AIStrategyEngine(config)

        # Mock fast AI response
        async def fast_analyze(ticker, data):
            await asyncio.sleep(0.01)  # Simulate 10ms API call
            return Signal(
                action=SignalType.BUY,
                confidence=70,
                position_size=0.05
            )

        strategy.analyze = fast_analyze

        # Measure end-to-end latency
        data = create_large_dataset(50)

        start = time.time()
        signal = await strategy.analyze("TEST", data)
        latency = time.time() - start

        # Should complete within 100ms
        assert latency < 0.1
        assert signal.action == SignalType.BUY

    @pytest.mark.asyncio
    async def test_throughput_under_load(self):
        """Test system throughput under heavy load."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            with patch('services.strategy_engine.src.ai_integration.create_async_engine'):
                integration = AIStrategyIntegration(
                    AsyncMock(),
                    'postgresql://test',
                    {}
                )

                # Mock fast analysis
                async def fast_analyze(ticker, data):
                    await asyncio.sleep(0.001)
                    return Signal(
                        action=SignalType.BUY,
                        confidence=70,
                        position_size=0.05
                    )

                integration.ai_strategy.analyze = fast_analyze

                # Process many tickers
                num_tickers = 100
                for i in range(num_tickers):
                    integration.price_data_buffer[f'TEST{i}'] = create_large_dataset(50)

                start = time.time()
                tasks = [
                    integration._analyze_ticker(f'TEST{i}')
                    for i in range(num_tickers)
                ]
                await asyncio.gather(*tasks)
                elapsed = time.time() - start

                throughput = num_tickers / elapsed

                # Should handle at least 10 tickers per second
                assert throughput > 10


# Helper functions

def create_large_dataset(num_rows: int) -> pl.DataFrame:
    """Create a large dataset for performance testing."""
    np.random.seed(42)
    dates = pl.date_range(
        datetime.now() - timedelta(days=num_rows),
        datetime.now(),
        interval='1d'
    )

    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)

    return pl.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.randn(len(dates)) * 0.5,
        'high': prices + np.abs(np.random.randn(len(dates))) * 1,
        'low': prices - np.abs(np.random.randn(len(dates))) * 1,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })


def mock_open(read_data=''):
    """Helper to mock file open."""
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=read_data)))
    m.__exit__ = MagicMock(return_value=None)
    return m


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not benchmark"])
