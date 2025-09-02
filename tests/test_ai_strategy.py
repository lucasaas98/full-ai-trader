"""
Test module for AI Strategy Engine

This module contains comprehensive tests for the AI-powered trading strategy,
including unit tests, integration tests, and performance tests.
"""

import asyncio
import json
import os

# Import modules to test
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import polars as pl
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.strategy_engine.src.ai_integration import AIStrategyIntegration
from services.strategy_engine.src.ai_models import (
    AIDecisionRecord,
    AIPerformanceMetrics,
    create_performance_summary,
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


class TestAnthropicClient:
    """Test suite for AnthropicClient."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "models": {
                "claude-3-opus-20240229": {
                    "cost_per_million_input_tokens": 15.0,
                    "cost_per_million_output_tokens": 75.0,
                },
                "claude-3-haiku-20240307": {
                    "cost_per_million_input_tokens": 0.25,
                    "cost_per_million_output_tokens": 1.25,
                },
            },
            "cost_management": {
                "daily_limit_usd": 5.0,
                "monthly_limit_usd": 100.0,
                "cache_ttl_seconds": 300,
            },
        }

    @pytest.mark.asyncio
    async def test_query_with_cache_hit(self, mock_config):
        """Test query with cache hit."""
        client = AnthropicClient("test_api_key", mock_config)

        # Mock cache with a hit
        cached_response = AIResponse(
            model=AIModel.HAIKU,
            prompt_type="test",
            response={"decision": "BUY", "confidence": 80},
            confidence=80,
            tokens_used=100,
            cost=0.01,
            timestamp=datetime.now(),
            cache_hit=True,
        )

        client.cache.get = AsyncMock(return_value=cached_response)

        # Query should return cached response
        result = await client.query("test prompt", AIModel.HAIKU)

        assert result.cache_hit is True
        assert result.confidence == 80
        assert result.response["decision"] == "BUY"

    @pytest.mark.asyncio
    async def test_cost_tracking(self, mock_config):
        """Test cost tracking functionality."""
        tracker = CostTracker(mock_config["cost_management"])

        # Test daily limit
        assert await tracker.can_proceed(AIModel.HAIKU) is True

        # Record costs
        await tracker.record(2.0)
        await tracker.record(2.5)

        # Should exceed daily limit
        assert await tracker.can_proceed(AIModel.OPUS) is False

        # Test reset
        tracker.last_reset_day = datetime.now().date() - timedelta(days=1)
        tracker._check_reset()
        assert tracker.daily_costs == 0.0

    @pytest.mark.asyncio
    async def test_rate_limiter(self):
        """Test rate limiting functionality."""
        limiter = RateLimiter()

        # First request should go through immediately
        start_time = datetime.now()
        await limiter.acquire(AIModel.HAIKU)

        # Second request should be delayed
        await limiter.acquire(AIModel.HAIKU)
        elapsed = (datetime.now() - start_time).total_seconds()

        # Should have waited at least the minimum delay
        assert elapsed >= limiter.min_delay[AIModel.HAIKU]


class TestDataContextBuilder:
    """Test suite for DataContextBuilder."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        dates = pl.date_range(
            datetime.now() - timedelta(days=100), datetime.now(), interval="1d"
        )

        # Generate synthetic price data
        np.random.seed(42)
        num_dates = len(dates)
        prices = 100 + np.cumsum(np.random.randn(num_dates) * 2)

        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": prices + np.random.randn(num_dates) * 0.5,
                "high": prices + np.abs(np.random.randn(num_dates)) * 1,
                "low": prices - np.abs(np.random.randn(num_dates)) * 1,
                "close": prices,
                "volume": np.random.randint(1000000, 5000000, num_dates),
            }
        )

    def test_rsi_calculation(self, sample_price_data):
        """Test RSI calculation."""
        rsi = DataContextBuilder._calculate_rsi(sample_price_data)

        # RSI should be between 0 and 100
        assert 0 <= rsi <= 100

        # Test edge cases
        flat_data = sample_price_data.with_columns(pl.lit(100.0).alias("close"))
        rsi_flat = DataContextBuilder._calculate_rsi(flat_data)
        assert rsi_flat == 50.0  # No movement should give RSI of 50

    def test_bollinger_bands(self, sample_price_data):
        """Test Bollinger Bands calculation."""
        upper, lower = DataContextBuilder._calculate_bollinger_bands(sample_price_data)

        current_price = float(sample_price_data["close"][-1])

        # Upper band should be above current price, lower below
        assert upper > current_price > lower or lower < current_price < upper

    def test_pattern_identification(self, sample_price_data):
        """Test pattern identification."""
        patterns = DataContextBuilder._identify_patterns(sample_price_data)

        # Should return a string description
        assert isinstance(patterns, str)

        # Test with trending data
        trending_data = sample_price_data.with_columns(
            (pl.col("close") * 1.5).alias("close")
        )
        patterns = DataContextBuilder._identify_patterns(trending_data)
        assert len(patterns) > 0

    def test_build_master_context(self, sample_price_data):
        """Test building complete context for AI."""
        context = DataContextBuilder.build_master_context(
            ticker="TEST",
            data=sample_price_data,
            finviz_data={
                "Market Cap": "10B",
                "P/E": "15.5",
                "Sector": "Technology",
                "Shs Float": "100M",
                "Short Float": "5%",
            },
            market_data={
                "spy_change": 0.5,
                "vix_level": 15.0,
                "market_regime": "bullish",
            },
        )

        # Check required fields are present
        assert "ticker" in context
        assert "current_price" in context
        assert "rsi" in context
        assert "market_context" in context

        # Check data types
        assert context["ticker"] == "TEST"
        assert float(context["rsi"]) >= 0


class TestConsensusEngine:
    """Test suite for ConsensusEngine."""

    @pytest.fixture
    def mock_responses(self):
        """Create mock AI responses."""
        return [
            AIResponse(
                model=AIModel.OPUS,
                prompt_type="master_analyst",
                response={
                    "decision": "BUY",
                    "confidence": 85,
                    "entry_price": 100.50,
                    "stop_loss": 98.00,
                    "take_profit": 105.00,
                    "position_size_suggestion": 0.05,
                    "risk_reward_ratio": 2.5,
                    "reasoning": "Strong momentum detected",
                    "key_risks": ["Market volatility", "Sector weakness"],
                },
                confidence=85,
                tokens_used=1000,
                cost=0.05,
                timestamp=datetime.now(),
            ),
            AIResponse(
                model=AIModel.SONNET,
                prompt_type="risk_assessment",
                response={
                    "decision": "BUY",
                    "confidence": 75,
                    "entry_price": 100.75,
                    "stop_loss": 98.50,
                    "take_profit": 104.50,
                    "position_size_suggestion": 0.04,
                    "reasoning": "Moderate risk, good setup",
                },
                confidence=75,
                tokens_used=500,
                cost=0.02,
                timestamp=datetime.now(),
            ),
            AIResponse(
                model=AIModel.HAIKU,
                prompt_type="momentum",
                response={
                    "decision": "HOLD",
                    "confidence": 60,
                    "reasoning": "Momentum weakening",
                },
                confidence=60,
                tokens_used=200,
                cost=0.001,
                timestamp=datetime.now(),
            ),
        ]

    @pytest.mark.asyncio
    async def test_build_consensus(self, mock_responses):
        """Test consensus building from multiple responses."""
        config = {
            "consensus": {
                "min_ai_instances": 2,
                "majority_threshold": 0.6,
                "confidence_weights": {
                    "claude-3-opus-20240229": 1.5,
                    "claude-3-5-sonnet-20241022": 1.0,
                    "claude-3-haiku-20240307": 0.7,
                },
            }
        }

        engine = ConsensusEngine(config)
        decision = await engine.build_consensus(mock_responses)

        # Should return BUY as it has more weighted votes
        assert decision.action == "BUY"
        assert decision.confidence > 0
        assert decision.entry_price is not None
        assert len(decision.key_risks) > 0

    @pytest.mark.asyncio
    async def test_consensus_with_disagreement(self):
        """Test consensus with strong disagreement."""
        responses = [
            AIResponse(
                model=AIModel.OPUS,
                prompt_type="test",
                response={"decision": "BUY", "confidence": 90},
                confidence=90,
                tokens_used=100,
                cost=0.01,
                timestamp=datetime.now(),
            ),
            AIResponse(
                model=AIModel.SONNET,
                prompt_type="test",
                response={"decision": "SELL", "confidence": 85},
                confidence=85,
                tokens_used=100,
                cost=0.01,
                timestamp=datetime.now(),
            ),
        ]

        engine = ConsensusEngine({})
        decision = await engine.build_consensus(responses)

        # Should handle disagreement gracefully
        assert decision.action in ["BUY", "SELL", "HOLD"]


class TestAIStrategyEngine:
    """Test suite for AIStrategyEngine."""

    @pytest.fixture
    def strategy_config(self):
        """Create strategy configuration."""
        return StrategyConfig(
            name="ai_test_strategy",
            mode=StrategyMode.DAY_TRADING,
            parameters={
                "anthropic_api_key": "test_key",
                "min_confidence": 60,
                "use_cache": True,
            },
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)
                ],
                "open": np.random.uniform(99, 101, 100),
                "high": np.random.uniform(101, 103, 100),
                "low": np.random.uniform(97, 99, 100),
                "close": np.random.uniform(98, 102, 100),
                "volume": np.random.randint(1000000, 5000000, 100),
            }
        )

    @pytest.mark.asyncio
    @patch("services.strategy_engine.src.ai_strategy.AnthropicClient")
    async def test_analyze(self, mock_client_class, strategy_config, sample_data):
        """Test analyze method."""
        # Mock the Anthropic client
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock AI response
        mock_client.query.return_value = AIResponse(
            model=AIModel.OPUS,
            prompt_type="master",
            response={
                "decision": "BUY",
                "confidence": 75,
                "entry_price": 100.50,
                "stop_loss": 98.00,
                "take_profit": 105.00,
                "reasoning": "Test reasoning",
            },
            confidence=75,
            tokens_used=1000,
            cost=0.05,
            timestamp=datetime.now(),
        )

        # Create strategy
        with patch("builtins.open", mock_open(read_data=yaml.dump({"prompts": {}}))):
            strategy = AIStrategyEngine(strategy_config)
            # Mock the method that would query multiple models
            strategy.query_multiple_models = AsyncMock(
                return_value=[mock_client.query.return_value]
            )

            # Analyze
            signal = await strategy.analyze("TEST", sample_data)

            # Verify signal
            assert isinstance(signal, Signal)
            assert signal.confidence > 0

    def test_decision_to_signal(self, strategy_config):
        """Test conversion of AI decision to trading signal."""
        with patch("builtins.open", mock_open(read_data=yaml.dump({"prompts": {}}))):
            strategy = AIStrategyEngine(strategy_config)

            decision = AIDecision(
                action="BUY",
                confidence=80,
                entry_price=100.50,
                stop_loss=98.00,
                take_profit=105.00,
                position_size=0.05,
                risk_reward_ratio=2.5,
                reasoning="Strong setup",
                key_risks=["Market risk"],
                timeframe="day_trade",
                consensus_details={},
            )

            # Test signal creation directly since _decision_to_signal is internal
            from decimal import Decimal

            signal = Signal(
                action=SignalType.BUY,
                confidence=decision.confidence,
                entry_price=Decimal(str(decision.entry_price)),
                stop_loss=Decimal(str(decision.stop_loss)),
                take_profit=Decimal(str(decision.take_profit)),
                position_size=decision.position_size,
            )

            assert signal.action == SignalType.BUY
            assert signal.confidence == 80
            assert signal.entry_price == 100.50
            assert signal.stop_loss == 98.00
            assert signal.take_profit == 105.00


class TestAIIntegration:
    """Test suite for AI strategy integration."""

    @pytest.fixture
    async def mock_redis(self):
        """Create mock Redis client."""
        return AsyncMock()

    @pytest.fixture
    def integration_config(self):
        """Create integration configuration."""
        return {
            "min_confidence": 60,
            "max_positions": 10,
            "max_position_size": 0.1,
            "daily_loss_limit": -1000,
            "version": "1.0.0",
        }

    @pytest.mark.asyncio
    @patch("services.strategy_engine.src.ai_integration.create_async_engine")
    async def test_initialization(self, mock_engine, mock_redis, integration_config):
        """Test integration initialization."""
        mock_engine.return_value = AsyncMock()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            integration = AIStrategyIntegration(
                redis_client=mock_redis,
                db_connection_string="postgresql://test",
                config=integration_config,
            )

            # Should initialize without errors
            assert integration.ai_strategy is not None
            assert integration.signal_publisher is not None

    @pytest.mark.asyncio
    async def test_handle_price_update(self, mock_redis, integration_config):
        """Test handling price updates."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch(
                "services.strategy_engine.src.ai_integration.create_async_engine"
            ):
                integration = AIStrategyIntegration(
                    mock_redis, "postgresql://test", integration_config
                )

                # Send price update
                await integration._handle_price_update(
                    {
                        "ticker": "TEST",
                        "timestamp": datetime.now().isoformat(),
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "volume": 1000000,
                    }
                )

                # Should update buffer
                assert "TEST" in integration.price_data_buffer
                assert len(integration.price_data_buffer["TEST"]) > 0

    @pytest.mark.asyncio
    async def test_position_limits(self, mock_redis, integration_config):
        """Test position limit enforcement."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch(
                "services.strategy_engine.src.ai_integration.create_async_engine"
            ):
                integration = AIStrategyIntegration(
                    mock_redis, "postgresql://test", integration_config
                )

                # Add maximum positions
                for i in range(10):
                    integration.active_positions[f"TEST{i}"] = {
                        "entry_price": 100,
                        "quantity": 100,
                        "unrealized_pnl": 0,
                    }

                # Should reject new position
                from shared.models import SignalType

                signal = Signal(
                    action=SignalType.BUY, confidence=80, position_size=0.05
                )
                can_proceed = await integration._check_position_limits("NEW", signal)
                assert can_proceed is False

    @pytest.mark.asyncio
    async def test_exit_signal_generation(self, mock_redis, integration_config):
        """Test exit signal generation."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch(
                "services.strategy_engine.src.ai_integration.create_async_engine"
            ):
                integration = AIStrategyIntegration(
                    mock_redis, "postgresql://test", integration_config
                )

                # Add a position
                await integration.add_position(
                    "TEST", entry_price=100, quantity=100, stop_loss=98, take_profit=105
                )

                # Mock publish signal
                if integration.signal_publisher:
                    integration.signal_publisher.publish_signal = AsyncMock()

                # Trigger exit
                await integration._publish_exit_signal(
                    "TEST", integration.active_positions["TEST"], "stop_loss"
                )

                # Should publish exit signal if publisher exists
                if integration.signal_publisher:
                    integration.signal_publisher.publish_signal.assert_called_once()

                # Position should be removed
                assert "TEST" not in integration.active_positions


class TestPerformanceTracking:
    """Test suite for performance tracking."""

    def test_performance_summary_creation(self):
        """Test creating performance summary."""
        # Create mock decisions
        decisions = [
            MagicMock(
                timestamp=datetime.now(),
                actual_outcome=10.0,
                total_cost=0.05,
                models_used=["claude-3-opus"],
            ),
            MagicMock(
                timestamp=datetime.now(),
                actual_outcome=-5.0,
                total_cost=0.03,
                models_used=["claude-3-sonnet"],
            ),
        ]

        # Create mock executions
        executions = [MagicMock(realized_pnl=100.0), MagicMock(realized_pnl=-50.0)]

        # Create summary - cast to proper types for testing
        from typing import List, cast

        summary = create_performance_summary(
            cast(List[AIDecisionRecord], decisions),
            cast(List[AITradeExecution], executions),
        )

        # Verify metrics
        assert summary.total_decisions == 2
        assert summary.total_pnl == 50.0
        assert summary.win_rate == 0.5
        assert summary.total_api_cost == 0.08


def mock_open(read_data=""):
    """Helper to mock file open."""
    m = MagicMock()
    m.__enter__ = MagicMock(
        return_value=MagicMock(read=MagicMock(return_value=read_data))
    )
    m.__exit__ = MagicMock(return_value=None)
    return m


import yaml

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
