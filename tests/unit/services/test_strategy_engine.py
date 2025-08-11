import pytest
from unittest.mock import AsyncMock, patch, Mock
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import os
import sys
from fastapi.testclient import TestClient

from services.strategy_engine.src.main import app, StrategyEngineService


from shared.models import MarketData, SignalType, TradeSignal, TimeFrame
from shared.config import Config

# Define missing classes for testing

class TechnicalIndicators:
    def __init__(self, symbol, timestamp, sma_20=0.0, sma_50=0.0, rsi=50.0, macd_line=0.0, macd_signal=0.0, bollinger_upper=0.0, bollinger_lower=0.0):
        self.symbol = symbol
        self.timestamp = timestamp
        self.sma_20 = sma_20
        self.sma_50 = sma_50
        self.rsi = rsi
        self.macd_line = macd_line
        self.macd_signal = macd_signal
        self.bollinger_upper = bollinger_upper
        self.bollinger_lower = bollinger_lower


class TestStrategyEngineService:
    """Test suite for StrategyEngineService"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        config = Mock(spec=Config)
        config.redis_host = "localhost"
        config.redis_port = 6379
        config.redis_password = None
        config.db_host = "localhost"
        config.db_port = 5432
        config.db_name = "test_db"
        config.db_user = "test_user"
        config.db_password = "test_pass"
        config.risk_max_position_size = 0.05
        config.risk_max_portfolio_risk = 0.02
        return config

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        redis_mock = AsyncMock()
        redis_mock.publish = AsyncMock()
        redis_mock.set = AsyncMock()
        redis_mock.get = AsyncMock()
        redis_mock.close = AsyncMock()
        redis_mock.ping = AsyncMock(return_value=True)
        return redis_mock

    @pytest.fixture
    def mock_db_pool(self):
        """Mock database connection pool"""
        pool_mock = AsyncMock()
        connection_mock = AsyncMock()
        pool_mock.acquire.return_value.__aenter__ = AsyncMock(return_value=connection_mock)
        pool_mock.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        connection_mock.execute = AsyncMock()
        connection_mock.fetch = AsyncMock()
        connection_mock.fetchrow = AsyncMock()
        return pool_mock

    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing"""
        base_time = datetime.now(timezone.utc)
        return [
            MarketData(
                symbol="AAPL",
                timestamp=base_time - timedelta(hours=i),
                open=Decimal("195.00") + Decimal(str(i)),
                high=Decimal("197.50") + Decimal(str(i)),
                low=Decimal("194.50") + Decimal(str(i)),
                close=Decimal("196.80") + Decimal(str(i)),
                volume=1000000,
                timeframe=TimeFrame.ONE_HOUR,
                adjusted_close=Decimal("196.80") + Decimal(str(i))
            ) for i in range(50)  # 50 data points for testing
        ]

    @pytest.fixture
    def service(self, mock_config):
        """Create StrategyEngineService instance for testing"""
        service = StrategyEngineService()
        return service

    @pytest.mark.asyncio
    async def test_moving_average_strategy_bullish_signal(self, service, sample_market_data):
        """Test moving average strategy generates bullish signal"""
        # Create data where short MA crosses above long MA
        modified_data = sample_market_data.copy()
        # Make recent prices higher to create bullish crossover
        for i in range(10):
            modified_data[i].close = 250.0 + i  # Recent high prices

        with patch.object(service, 'get_historical_data', return_value=modified_data):
            signal = await service.generate_moving_average_signal("AAPL", 20, 50)

            assert signal is not None
            assert signal.symbol == "AAPL"
            assert signal.signal_type == SignalType.BUY
            assert signal.confidence > 0.5

    @pytest.mark.asyncio
    async def test_moving_average_strategy_bearish_signal(self, service, sample_market_data):
        """Test moving average strategy generates bearish signal"""
        # Create data where short MA crosses below long MA
        modified_data = sample_market_data.copy()
        # Make recent prices lower to create bearish crossover
        for i in range(10):
            modified_data[i].close = 150.0 - i  # Recent low prices

        with patch.object(service, 'get_historical_data', return_value=modified_data):
            signal = await service.generate_moving_average_signal("AAPL", 20, 50)

            assert signal is not None
            assert signal.symbol == "AAPL"
            assert signal.signal_type == SignalType.SELL
            assert signal.confidence > 0.5

    @pytest.mark.asyncio
    async def test_moving_average_strategy_no_signal(self, service, sample_market_data):
        """Test moving average strategy when no clear signal"""
        # Create sideways market data
        for i, data_point in enumerate(sample_market_data):
            data_point.close = 196.0 + (i % 2) * 0.1  # Sideways movement

        with patch.object(service, 'get_historical_data', return_value=sample_market_data):
            signal = await service.generate_moving_average_signal("AAPL", 20, 50)

            assert signal is None or signal.confidence < 0.5

    @pytest.mark.asyncio
    async def test_moving_average_strategy_insufficient_data(self, service, sample_market_data):
        """Test moving average strategy with insufficient data"""
        insufficient_data = sample_market_data[:10]  # Only 10 data points

        with patch.object(service, 'get_historical_data', return_value=insufficient_data):
            signal = await service.generate_moving_average_signal("AAPL", 20, 50)

            assert signal is None

    @pytest.mark.asyncio
    async def test_rsi_strategy_oversold_signal(self, service, sample_market_data):
        """Test RSI strategy generates buy signal when oversold"""
        # Mock RSI calculation to return oversold value
        with patch.object(service, 'calculate_rsi', return_value=25.0), \
             patch.object(service, 'get_historical_data', return_value=sample_market_data):

            signal = await service.generate_rsi_signal("AAPL", period=14)

            assert signal is not None
            assert signal.signal_type == SignalType.BUY
            assert signal.confidence > 0.6

    @pytest.mark.asyncio
    async def test_rsi_strategy_overbought_signal(self, service, sample_market_data):
        """Test RSI strategy generates sell signal when overbought"""
        # Mock RSI calculation to return overbought value
        with patch.object(service, 'calculate_rsi', return_value=85.0), \
             patch.object(service, 'get_historical_data', return_value=sample_market_data):

            signal = await service.generate_rsi_signal("AAPL", period=14)

            assert signal is not None
            assert signal.signal_type == SignalType.SELL
            assert signal.confidence > 0.6

    @pytest.mark.asyncio
    async def test_rsi_strategy_neutral_signal(self, service, sample_market_data):
        """Test RSI strategy when in neutral zone"""
        # Mock RSI calculation to return neutral value
        with patch.object(service, 'calculate_rsi', return_value=50.0), \
             patch.object(service, 'get_historical_data', return_value=sample_market_data):

            signal = await service.generate_rsi_signal("AAPL", period=14)

            assert signal is None or signal.confidence < 0.5

    @pytest.mark.asyncio
    async def test_bollinger_bands_strategy_oversold(self, service, sample_market_data):
        """Test Bollinger Bands strategy when price below lower band"""
        # Mock Bollinger Bands calculation
        with patch.object(service, 'calculate_bollinger_bands') as mock_bb, \
             patch.object(service, 'get_historical_data', return_value=sample_market_data):

            mock_bb.return_value = {
                'upper_band': 200.0,
                'lower_band': 190.0,
                'middle_band': 195.0
            }

            # Set current price below lower band
            sample_market_data[0].close = 185.0

            signal = await service.generate_bollinger_signal("AAPL", period=20, std_dev=2)

            assert signal is not None
            assert signal.signal_type == SignalType.BUY

    @pytest.mark.asyncio
    async def test_bollinger_bands_strategy_overbought(self, service, sample_market_data):
        """Test Bollinger Bands strategy when price above upper band"""
        with patch.object(service, 'calculate_bollinger_bands') as mock_bb, \
             patch.object(service, 'get_historical_data', return_value=sample_market_data):

            mock_bb.return_value = {
                'upper_band': 200.0,
                'lower_band': 190.0,
                'middle_band': 195.0
            }

            # Set current price above upper band
            sample_market_data[0].close = 205.0

            signal = await service.generate_bollinger_signal("AAPL", period=20, std_dev=2)

            assert signal is not None
            assert signal.signal_type == SignalType.SELL

    @pytest.mark.asyncio
    async def test_macd_strategy_bullish_crossover(self, service, sample_market_data):
        """Test MACD strategy bullish crossover"""
        with patch.object(service, 'calculate_macd') as mock_macd, \
             patch.object(service, 'get_historical_data', return_value=sample_market_data):

            # Mock MACD values showing bullish crossover
            mock_macd.return_value = {
                'macd_line': [1.0, 1.5, 2.0],  # Rising
                'signal_line': [1.5, 1.4, 1.3],  # Falling
                'histogram': [-0.5, 0.1, 0.7]  # Crossing from negative to positive
            }

            signal = await service.generate_macd_signal("AAPL", fast=12, slow=26, signal_period=9)

            assert signal is not None
            assert signal.signal_type == SignalType.BUY

    @pytest.mark.asyncio
    async def test_macd_strategy_bearish_crossover(self, service, sample_market_data):
        """Test MACD strategy bearish crossover"""
        with patch.object(service, 'calculate_macd') as mock_macd, \
             patch.object(service, 'get_historical_data', return_value=sample_market_data):

            # Mock MACD values showing bearish crossover
            mock_macd.return_value = {
                'macd_line': [2.0, 1.5, 1.0],  # Falling
                'signal_line': [1.3, 1.4, 1.5],  # Rising
                'histogram': [0.7, 0.1, -0.5]  # Crossing from positive to negative
            }

            signal = await service.generate_macd_signal("AAPL", fast=12, slow=26, signal_period=9)

            assert signal is not None
            assert signal.signal_type == SignalType.SELL

    @pytest.mark.asyncio
    async def test_technical_indicators_calculation_sma(self, service, sample_market_data):
        """Test Simple Moving Average calculation"""
        prices = [data.close for data in sample_market_data[:20]]

        sma = service.calculate_sma(prices, period=10)

        assert len(sma) == len(prices)
        assert sma[-1] == sum(prices[-10:]) / 10  # Last value should be average of last 10

    @pytest.mark.asyncio
    async def test_technical_indicators_calculation_ema(self, service, sample_market_data):
        """Test Exponential Moving Average calculation"""
        prices = [data.close for data in sample_market_data[:20]]

        ema = service.calculate_ema(prices, period=10)

        assert len(ema) == len(prices)
        assert ema[-1] != sum(prices[-10:]) / 10  # Should be different from SMA

    def test_rsi_calculation_valid_range(self, service):
        """Test RSI calculation returns values in valid range (0-100)"""
        # Create price data with clear trend
        prices = [100.0 + i for i in range(50)]  # Uptrend

        rsi = service.calculate_rsi(prices, period=14)

        assert all(0 <= value <= 100 for value in rsi if value is not None)

    def test_rsi_calculation_overbought_condition(self, service):
        """Test RSI calculation identifies overbought conditions"""
        # Create strong uptrend data
        prices = [100.0 + i * 2 for i in range(50)]

        rsi = service.calculate_rsi(prices, period=14)

        # Latest RSI should be high (overbought)
        assert rsi[-1] > 70

    def test_rsi_calculation_oversold_condition(self, service):
        """Test RSI calculation identifies oversold conditions"""
        # Create strong downtrend data
        prices = [200.0 - i * 2 for i in range(50)]

        rsi = service.calculate_rsi(prices, period=14)

        # Latest RSI should be low (oversold)
        assert rsi[-1] < 30

    def test_bollinger_bands_calculation(self, service):
        """Test Bollinger Bands calculation"""
        prices = [100.0 + (i % 10) for i in range(50)]  # Oscillating prices

        bb = service.calculate_bollinger_bands(prices, period=20, std_dev=2)

        assert 'upper_band' in bb
        assert 'lower_band' in bb
        assert 'middle_band' in bb
        assert bb['upper_band'] > bb['middle_band'] > bb['lower_band']

    def test_macd_calculation(self, service):
        """Test MACD calculation"""
        prices = [100.0 + i * 0.5 for i in range(100)]  # Trending prices

        macd = service.calculate_macd(prices, fast=12, slow=26, signal_period=9)

        assert 'macd_line' in macd
        assert 'signal_line' in macd
        assert 'histogram' in macd
        assert len(macd['macd_line']) == len(prices)

    @pytest.mark.asyncio
    async def test_signal_generation_with_volume_confirmation(self, service, sample_market_data):
        """Test signal generation includes volume confirmation"""
        # Set high volume for recent data points
        for i in range(5):
            sample_market_data[i].volume = 5000000  # High volume

        with patch.object(service, 'get_historical_data', return_value=sample_market_data):
            signal = await service.generate_volume_confirmed_signal("AAPL", "moving_average")

            if signal:
                assert signal.metadata.get('volume_confirmed') is True

    @pytest.mark.asyncio
    async def test_signal_generation_multiple_strategies_consensus(self, service, sample_market_data):
        """Test consensus signal from multiple strategies"""
        with patch.object(service, 'get_historical_data', return_value=sample_market_data), \
             patch.object(service, 'generate_moving_average_signal') as mock_ma, \
             patch.object(service, 'generate_rsi_signal') as mock_rsi, \
             patch.object(service, 'generate_bollinger_signal') as mock_bb:

            # Mock all strategies returning BUY signals
            mock_signal = TradeSignal(
                symbol="AAPL",
                signal_type=SignalType.BUY,
                confidence=0.85,
                strategy_name="consensus",
                price=Decimal("196.80"),
                quantity=100,
                stop_loss=Decimal("190.00"),
                take_profit=Decimal("210.00")
            )

            mock_ma.return_value = mock_signal
            mock_rsi.return_value = mock_signal
            mock_bb.return_value = mock_signal

            consensus_signal = await service.generate_consensus_signal("AAPL")

            assert consensus_signal is not None
            assert consensus_signal.signal_type == SignalType.BUY
            assert consensus_signal.confidence > 0.8  # High confidence from consensus

    @pytest.mark.asyncio
    async def test_signal_generation_conflicting_strategies(self, service, sample_market_data):
        """Test handling of conflicting signals from different strategies"""
        with patch.object(service, 'get_historical_data', return_value=sample_market_data), \
             patch.object(service, 'generate_moving_average_signal') as mock_ma, \
             patch.object(service, 'generate_rsi_signal') as mock_rsi:

            # Mock conflicting signals
            buy_signal = TradeSignal(
                symbol="AAPL",
                signal_type=SignalType.BUY,
                confidence=0.8,
                strategy_name="moving_average",
                price=Decimal("196.80"),
                quantity=100,
                stop_loss=Decimal("190.00"),
                take_profit=Decimal("210.00")
            )

            sell_signal = TradeSignal(
                symbol="AAPL",
                signal_type=SignalType.SELL,
                confidence=0.7,
                strategy_name="rsi",
                price=Decimal("196.80"),
                quantity=100,
                stop_loss=Decimal("205.00"),
                take_profit=Decimal("180.00")
            )

            mock_ma.return_value = buy_signal
            mock_rsi.return_value = sell_signal

            consensus_signal = await service.generate_consensus_signal("AAPL")

            # Should either be None or have low confidence
            assert consensus_signal is None or consensus_signal.confidence < 0.5

    @pytest.mark.asyncio
    async def test_get_historical_data_success(self, service):
        """Test successful historical data retrieval"""
        mock_rows = [
            {
                'symbol': 'AAPL',
                'timestamp': datetime.now(timezone.utc) - timedelta(hours=i),
                'open': 195.0 + i,
                'high': 197.0 + i,
                'low': 194.0 + i,
                'close': 196.0 + i,
                'volume': 1000000,
                'timeframe': '1h'
            } for i in range(50)
        ]

        service.db_pool.acquire.return_value.__aenter__.return_value.fetch.return_value = mock_rows

        result = await service.get_historical_data("AAPL", TimeFrame.ONE_HOUR, limit=50)

        assert len(result) == 50
        assert all(isinstance(item, MarketData) for item in result)

    @pytest.mark.asyncio
    async def test_get_historical_data_database_error(self, service):
        """Test handling of database errors during data retrieval"""
        service.db_pool.acquire.return_value.__aenter__.return_value.fetch.side_effect = \
            Exception("Database error")

        result = await service.get_historical_data("AAPL", TimeFrame.ONE_HOUR, days=30)

        assert result == []

    @pytest.mark.asyncio
    async def test_store_signal_success(self, service):
        """Test successful signal storage"""
        signal = TradeSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            price=Decimal("196.80"),
            quantity=100,
            strategy_name="moving_average",
            metadata={"ma_short": 20, "ma_long": 50},
            stop_loss=Decimal("190.00"),
            take_profit=Decimal("210.00")
        )

        service.db_pool.acquire.return_value.__aenter__.return_value.execute = AsyncMock()

        await service.store_signal(signal)

        service.db_pool.acquire.return_value.__aenter__.return_value.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_signal_database_error(self, service):
        """Test handling of database errors during signal storage"""
        signal = TradeSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            price=Decimal("196.80"),
            quantity=100,
            strategy_name="moving_average",
            metadata={},
            stop_loss=Decimal("190.00"),
            take_profit=Decimal("210.00")
        )

        service.db_pool.acquire.return_value.__aenter__.return_value.execute.side_effect = \
            Exception("Database error")

        # Should not raise exception
        await service.store_signal(signal)

    @pytest.mark.asyncio
    async def test_publish_signal_to_redis(self, service):
        """Test signal publication to Redis"""
        signal = TradeSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            price=Decimal("196.80"),
            quantity=100,
            strategy_name="moving_average",
            metadata={},
            stop_loss=Decimal("190.00"),
            take_profit=Decimal("210.00")
        )

        await service.publish_signal(signal)

        service.redis_client.publish.assert_called_once()
        call_args = service.redis_client.publish.call_args
        assert call_args[0][0] == "trade_signals"  # Channel name

    @pytest.mark.asyncio
    async def test_signal_filtering_by_confidence(self, service):
        """Test signal filtering based on confidence threshold"""
        low_confidence_signal = TradeSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.3,  # Low confidence
            timestamp=datetime.now(timezone.utc),
            price=Decimal("196.80"),
            quantity=100,
            strategy_name="test",
            metadata={},
            stop_loss=Decimal("190.00"),
            take_profit=Decimal("210.00")
        )

        # Should filter out low confidence signals
        filtered_signal = service.filter_signal_by_confidence(low_confidence_signal, min_confidence=0.5)
        assert filtered_signal is None

        high_confidence_signal = TradeSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            price=Decimal("196.80"),
            quantity=100,
            strategy_name="test",
            metadata={},
            stop_loss=Decimal("190.00"),
            take_profit=Decimal("210.00")
        )

        filtered_signal = service.filter_signal_by_confidence(high_confidence_signal, min_confidence=0.5)
        assert filtered_signal is not None

    @pytest.mark.asyncio
    async def test_strategy_performance_tracking(self, service):
        """Test tracking of strategy performance"""
        strategy_name = "moving_average"
        signal = TradeSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            price=Decimal("196.80"),
            quantity=100,
            strategy_name=strategy_name,
            metadata={},
            stop_loss=Decimal("190.00"),
            take_profit=Decimal("210.00")
        )

        await service.track_strategy_performance(strategy_name, signal, outcome="profitable")

        # Should update strategy performance metrics
        # Implementation would depend on how performance tracking is implemented

    @pytest.mark.asyncio
    async def test_concurrent_signal_generation(self, service, sample_market_data):
        """Test concurrent signal generation for multiple symbols"""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        with patch.object(service, 'get_historical_data', return_value=sample_market_data), \
             patch.object(service, 'generate_moving_average_signal') as mock_generate:

            mock_generate.return_value = TradeSignal(
                symbol="TEST",
                signal_type=SignalType.BUY,
                confidence=0.8,
                strategy_name="moving_average",
                price=Decimal("196.80"),
                quantity=100,
                stop_loss=Decimal("190.00"),
                take_profit=Decimal("210.00")
            )

            signals = await service.generate_signals_for_symbols(symbols, "moving_average")

            # Should generate signals for all symbols
            assert len(signals) == len(symbols)

    @pytest.mark.asyncio
    async def test_signal_metadata_enrichment(self, service, sample_market_data):
        """Test signal metadata includes relevant technical indicators"""
        with patch.object(service, 'get_historical_data', return_value=sample_market_data), \
             patch.object(service, 'calculate_technical_indicators') as mock_indicators:

            mock_indicators.return_value = TechnicalIndicators(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                sma_20=196.5,
                sma_50=195.0,
                rsi=65.0,
                macd_line=1.5,
                macd_signal=1.2,
                bollinger_upper=200.0,
                bollinger_lower=190.0
            )

            signal = await service.generate_enriched_signal("AAPL", "moving_average")

            if signal:
                assert 'technical_indicators' in signal.metadata
                assert signal.metadata['technical_indicators']['rsi'] == 65.0

    @pytest.mark.asyncio
    async def test_health_check_healthy_service(self, service):
        """Test health check for healthy service"""
        service.redis_client.ping = AsyncMock(return_value=True)
        service.db_pool.acquire.return_value.__aenter__.return_value.fetchrow = AsyncMock(
            return_value={"version": "15.0"}
        )

        health = await service.get_health()

        assert health["status"] == "healthy"
        assert health["redis"] == "connected"
        assert health["database"] == "connected"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_dependencies(self, service):
        """Test health check with unhealthy dependencies"""
        service.redis_client.ping = AsyncMock(side_effect=Exception("Redis error"))
        service.db_pool.acquire.return_value.__aenter__.return_value.fetchrow = AsyncMock(
            side_effect=Exception("Database error")
        )

        health = await service.get_health()

        assert health["status"] == "unhealthy"
        assert health["redis"] == "disconnected"
        assert health["database"] == "disconnected"


class TestStrategyEngineAPI:
    """Test suite for Strategy Engine API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_service(self):
        """Mock StrategyEngineService"""
        with patch('main.strategy_engine_service') as mock:
            yield mock

    def test_health_endpoint_healthy(self, client, mock_service):
        """Test health endpoint when service is healthy"""
        mock_service.get_health.return_value = {
            "status": "healthy",
            "service": "strategy_engine",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "redis": "connected",
            "database": "connected",
            "uptime": 3600.0
        }

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_strategies_endpoint(self, client, mock_service):
        """Test strategies listing endpoint"""
        mock_service.get_available_strategies.return_value = [
            {
                "name": "moving_average",
                "description": "Moving Average Crossover Strategy",
                "parameters": ["short_window", "long_window"]
            },
            {
                "name": "rsi",
                "description": "RSI Momentum Strategy",
                "parameters": ["period", "oversold_threshold", "overbought_threshold"]
            }
        ]

        response = client.get("/strategies")

        assert response.status_code == 200
        data = response.json()
        assert len(data["strategies"]) == 2
        assert any(s["name"] == "moving_average" for s in data["strategies"])

    def test_generate_signals_endpoint_success(self, client, mock_service):
        """Test signal generation endpoint success"""
        mock_signal = TradeSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            price=Decimal("196.80"),
            quantity=100,
            strategy_name="moving_average",
            metadata={},
            stop_loss=Decimal("190.00"),
            take_profit=Decimal("210.00")
        )

        mock_service.generate_signals_for_symbols.return_value = [mock_signal]

        response = client.post("/signals/generate", json={
            "symbols": ["AAPL"],
            "strategy": "moving_average",
            "parameters": {"short_window": 20, "long_window": 50}
        })

        assert response.status_code == 200
        data = response.json()
        assert len(data["signals"]) == 1
        assert data["signals"][0]["symbol"] == "AAPL"

    def test_generate_signals_endpoint_invalid_strategy(self, client, mock_service):
        """Test signal generation endpoint with invalid strategy"""
        response = client.post("/signals/generate", json={
            "symbols": ["AAPL"],
            "strategy": "nonexistent_strategy",
            "parameters": {}
        })

        assert response.status_code == 400

    def test_generate_signals_endpoint_empty_symbols(self, client, mock_service):
        """Test signal generation endpoint with empty symbols"""
        response = client.post("/signals/generate", json={
            "symbols": [],
            "strategy": "moving_average",
            "parameters": {}
        })

        assert response.status_code == 422

    def test_get_strategy_performance_endpoint(self, client, mock_service):
        """Test strategy performance endpoint"""
        mock_service.get_strategy_performance.return_value = {
            "strategy": "moving_average",
            "total_signals": 100,
            "profitable_signals": 65,
            "win_rate": 0.65,
            "avg_return": 0.025,
            "sharpe_ratio": 1.8,
            "max_drawdown": 0.08
        }

        response = client.get("/strategies/moving_average/performance")

        assert response.status_code == 200
        data = response.json()
        assert data["win_rate"] == 0.65
        assert data["sharpe_ratio"] == 1.8

    def test_update_strategy_parameters_endpoint(self, client, mock_service):
        """Test strategy parameter update endpoint"""
        response = client.put("/strategies/moving_average/parameters", json={
            "short_window": 15,
            "long_window": 40
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "parameters_updated" in data
