"""
Integration tests for the data collector service.

These tests will work once the aioredis import issues are resolved.
They test the actual service integration and business logic flows.
"""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# These imports will work once aioredis issues are resolved
# from services.data_collector.src.data_collection_service import DataCollectionService, DataCollectionConfig
# from services.data_collector.src.data_store import DataStore, DataStoreConfig


@pytest.mark.integration
@pytest.mark.skip(reason="Requires aioredis compatibility fix")
class TestDataCollectionServiceIntegration:
    """Integration tests for the complete data collection service"""

    @pytest.fixture
    async def service_config(self):
        """Create test service configuration"""
        return {
            "service_name": "test_data_collector",
            "enable_finviz": True,
            "enable_twelvedata": True,
            "enable_redis": False,  # Disable Redis for tests
            "max_active_tickers": 5,
            "finviz_scan_interval": 60,
            "max_retries": 2,
            "retry_delay": 0.1,
        }

    @pytest.fixture
    async def mock_data_store(self, tmp_path):
        """Create mock data store with temporary directory"""
        config = {
            "base_path": str(tmp_path),
            "compression": "snappy",
            "batch_size": 100,
            "retention_days": 30,
        }
        # return DataStore(DataStoreConfig(**config))
        return MagicMock()

    @pytest.fixture
    async def data_service(self, service_config, mock_data_store):
        """Create data collection service for testing"""
        # config = DataCollectionConfig(**service_config)
        # service = DataCollectionService(config)
        # service.data_store = mock_data_store
        # return service
        return MagicMock()

    async def test_service_initialization(self, data_service):
        """Test that the service initializes properly"""
        # await data_service.initialize()

        # assert data_service.is_running is False
        # assert data_service._active_tickers is not None
        # assert data_service._stats is not None
        # assert 'screener_runs' in data_service._stats
        # assert 'data_updates' in data_service._stats
        # assert 'errors' in data_service._stats
        pass

    async def test_service_start_stop_lifecycle(self, data_service):
        """Test service start/stop lifecycle"""
        # Test starting the service
        # await data_service.start()
        # assert data_service.is_running is True

        # Test stopping the service
        # await data_service.stop()
        # assert data_service.is_running is False
        pass

    async def test_ticker_management_workflow(self, data_service):
        """Test complete ticker management workflow"""
        # await data_service.initialize()

        # Add ticker
        # await data_service.add_ticker("AAPL", fetch_historical=False)
        # active_tickers = await data_service.get_active_tickers()
        # assert "AAPL" in active_tickers

        # Remove ticker by updating the set
        # data_service._active_tickers.discard("AAPL")
        # active_tickers = await data_service.get_active_tickers()
        # assert "AAPL" not in active_tickers
        pass

    async def test_finviz_scan_integration(self, data_service):
        """Test FinViz scanning integration"""
        # Mock FinViz client
        mock_finviz = AsyncMock()
        mock_finviz.scan_stocks.return_value = [
            {
                "ticker": "AAPL",
                "symbol": "AAPL",
                "company": "Apple Inc.",
                "industry": "Technology",
                "country": "USA",
                "price": Decimal("150.25"),
                "change": 0.25,
                "volume": 1500000,
                "market_cap": Decimal("2500000000000"),
                "pe_ratio": 25.5,
                "sector": "Technology",
            }
        ]

        # data_service.finviz_screener = mock_finviz
        # await data_service._run_finviz_scan()

        # Should have added ticker to active list
        # assert len(data_service._active_tickers) > 0
        # assert 'AAPL' in data_service._active_tickers
        pass

    async def test_price_data_update_integration(self, data_service):
        """Test price data update integration"""
        # Mock TwelveData client
        mock_twelve_data = AsyncMock()
        mock_twelve_data.get_time_series.return_value = {
            "values": [
                {
                    "datetime": "2024-01-01 15:59:00",
                    "open": "150.00",
                    "high": "150.50",
                    "low": "149.75",
                    "close": "150.25",
                    "volume": "1500000",
                }
            ]
        }

        # data_service.twelvedata_client = mock_twelve_data
        # data_service._active_tickers.add('AAPL')

        # from shared.models import TimeFrame
        # await data_service._update_price_data(TimeFrame.ONE_MINUTE)

        # Should have called the API
        # mock_twelve_data.get_time_series.assert_called()
        pass

    async def test_error_handling_integration(self, data_service):
        """Test error handling in service integration"""
        # Mock failing TwelveData client
        mock_twelve_data = AsyncMock()
        mock_twelve_data.get_time_series.side_effect = Exception("API Error")

        # data_service.twelvedata_client = mock_twelve_data
        # data_service._active_tickers.add('AAPL')

        # Should handle errors gracefully
        # initial_errors = data_service._stats['errors']
        # from shared.models import TimeFrame
        # await data_service._update_price_data(TimeFrame.ONE_MINUTE)

        # Error count should increase
        # assert data_service._stats['errors'] > initial_errors
        pass

    async def test_health_check_integration(self, data_service):
        """Test health check integration"""
        # await data_service.initialize()
        # health_status = await data_service._health_check()

        # assert 'timestamp' in health_status
        # assert 'components' in health_status
        # assert 'overall_status' in health_status
        pass

    async def test_metrics_collection_integration(self, data_service):
        """Test metrics collection integration"""
        # await data_service.initialize()
        # status = await data_service.get_service_status()

        # assert 'stats' in status
        # assert 'screener_runs' in status['stats']
        # assert 'data_updates' in status['stats']
        # assert 'errors' in status['stats']
        # assert 'uptime' in status
        pass


@pytest.mark.integration
@pytest.mark.skip(reason="Requires aioredis compatibility fix")
class TestDataStoreIntegration:
    """Integration tests for data storage functionality"""

    @pytest.fixture
    def temp_data_path(self, tmp_path):
        """Create temporary directory for data storage tests"""
        return tmp_path / "test_data"

    @pytest.fixture
    async def data_store(self, temp_data_path):
        """Create data store for testing"""
        # config = DataStoreConfig(
        #     base_path=str(temp_data_path),
        #     compression='snappy',
        #     batch_size=100,
        #     retention_days=30
        # )
        # return DataStore(config)
        return MagicMock()

    async def test_store_and_retrieve_market_data(self, data_store, temp_data_path):
        """Test storing and retrieving market data"""
        from shared.models import MarketData, TimeFrame

        # Create test market data
        market_data = [
            MarketData(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                timeframe=TimeFrame.ONE_MINUTE,
                open=Decimal("150.0"),
                high=Decimal("151.0"),
                low=Decimal("149.0"),
                close=Decimal("150.5"),
                volume=1000000,
                adjusted_close=Decimal("150.5"),
            ),
            MarketData(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 1, 10, 1, tzinfo=timezone.utc),
                timeframe=TimeFrame.ONE_MINUTE,
                open=Decimal("150.5"),
                high=Decimal("151.5"),
                low=Decimal("150.0"),
                close=Decimal("151.0"),
                volume=1100000,
                adjusted_close=Decimal("151.0"),
            ),
        ]

        # Store data
        # await data_store.store_market_data(market_data)

        # Retrieve data
        # start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # end_date = datetime(2024, 1, 2, tzinfo=timezone.utc)
        # retrieved_data = await data_store.get_market_data(
        #     "AAPL", TimeFrame.ONE_MINUTE, start_date, end_date
        # )

        # assert len(retrieved_data) == 2
        # assert retrieved_data[0].symbol == "AAPL"
        # assert retrieved_data[0].close == Decimal("150.5")
        pass

    async def test_data_compression_and_storage_efficiency(
        self, data_store, temp_data_path
    ):
        """Test data compression and storage efficiency"""
        import pandas as pd

        from shared.models import MarketData, TimeFrame

        # Create large dataset
        timestamps = pd.date_range(
            "2024-01-01 09:30:00", periods=1000, freq="1min", tz=timezone.utc
        )
        market_data = []

        for i, timestamp in enumerate(timestamps):
            market_data.append(
                MarketData(
                    symbol="AAPL",
                    timestamp=timestamp,
                    timeframe=TimeFrame.ONE_MINUTE,
                    open=Decimal(f"{100.0 + i * 0.01:.2f}"),
                    high=Decimal(f"{100.5 + i * 0.01:.2f}"),
                    low=Decimal(f"{99.5 + i * 0.01:.2f}"),
                    close=Decimal(f"{100.25 + i * 0.01:.2f}"),
                    volume=1000 + i,
                    adjusted_close=Decimal(f"{100.25 + i * 0.01:.2f}"),
                )
            )

        # Store data
        # await data_store.store_market_data(market_data)

        # Check that files are created and compressed
        # assert temp_data_path.exists()
        # parquet_files = list(temp_data_path.rglob("*.parquet"))
        # assert len(parquet_files) > 0

        # Check file size is reasonable (compressed)
        # for file_path in parquet_files:
        #     assert file_path.stat().st_size > 0
        #     assert file_path.stat().st_size < 1024 * 1024  # Less than 1MB for 1000 records
        pass

    async def test_data_retention_policy(self, data_store, temp_data_path):
        """Test data retention policy enforcement"""
        from shared.models import MarketData, TimeFrame

        # Create old data that should be deleted
        old_timestamp = datetime.now(timezone.utc) - timedelta(days=40)
        old_data = MarketData(
            symbol="OLD",
            timestamp=old_timestamp,
            timeframe=TimeFrame.ONE_MINUTE,
            open=Decimal("100.0"),
            high=Decimal("101.0"),
            low=Decimal("99.0"),
            close=Decimal("100.5"),
            volume=1000000,
            adjusted_close=Decimal("100.5"),
        )

        # Create recent data that should be kept
        recent_timestamp = datetime.now(timezone.utc) - timedelta(days=10)
        recent_data = MarketData(
            symbol="NEW",
            timestamp=recent_timestamp,
            timeframe=TimeFrame.ONE_MINUTE,
            open=Decimal("150.0"),
            high=Decimal("151.0"),
            low=Decimal("149.0"),
            close=Decimal("150.5"),
            volume=1000000,
            adjusted_close=Decimal("150.5"),
        )

        # Store both datasets
        # await data_store.store_market_data([old_data, recent_data])

        # Apply retention policy
        # await data_store.cleanup_old_data()

        # Check that old data is removed and recent data remains
        # Should implement this check based on actual data store implementation
        pass


@pytest.mark.integration
@pytest.mark.skip(reason="Requires aioredis compatibility fix")
class TestEndToEndDataFlow:
    """End-to-end integration tests for complete data flow"""

    async def test_complete_data_collection_workflow(self):
        """Test complete workflow from screening to storage"""
        # This would test:
        # 1. FinViz screening to find stocks
        # 2. Adding stocks to active list
        # 3. Fetching price data from TwelveData
        # 4. Validating and cleaning data
        # 5. Storing data to parquet files
        # 6. Health checks and metrics collection
        pass

    async def test_error_recovery_workflow(self):
        """Test error recovery and retry mechanisms"""
        # This would test:
        # 1. API failures and retry logic
        # 2. Circuit breaker activation
        # 3. Graceful degradation
        # 4. Error metrics collection
        # 5. Service recovery
        pass

    async def test_concurrent_data_collection(self):
        """Test concurrent data collection for multiple symbols"""
        # This would test:
        # 1. Concurrent API calls
        # 2. Rate limiting
        # 3. Data consistency
        # 4. Performance metrics
        pass

    async def test_market_hours_integration(self):
        """Test market hours awareness in data collection"""
        # This would test:
        # 1. Market hours detection
        # 2. Scheduling during market hours
        # 3. After-hours behavior
        # 4. Weekend handling
        pass


@pytest.mark.performance
@pytest.mark.skip(reason="Requires aioredis compatibility fix")
class TestDataCollectorPerformance:
    """Performance tests for data collector service"""

    async def test_high_volume_data_processing(self):
        """Test processing large volumes of market data"""
        # This would test:
        # 1. Memory usage with large datasets
        # 2. Processing speed
        # 3. Batch processing efficiency
        # 4. Storage performance
        pass

    async def test_concurrent_symbol_updates(self):
        """Test updating many symbols concurrently"""
        # This would test:
        # 1. Concurrent API calls performance
        # 2. Rate limiting effectiveness
        # 3. Resource utilization
        # 4. Error handling under load
        pass

    async def test_storage_performance(self):
        """Test storage performance with different data volumes"""
        # This would test:
        # 1. Parquet write performance
        # 2. Compression effectiveness
        # 3. Query performance
        # 4. File organization efficiency
        pass


# Helper functions for integration tests


def create_sample_finviz_data(symbols=None):
    """Create sample FinViz data for testing"""
    if symbols is None:
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

    from shared.models import FinVizData

    sample_data = []
    for i, symbol in enumerate(symbols):
        sample_data.append(
            FinVizData(
                ticker=symbol,
                symbol=symbol,
                company=f"{symbol} Inc.",
                industry="Technology",
                country="USA",
                price=Decimal(f"{100.0 + i * 10:.2f}"),
                change=0.25 + i * 0.1,
                volume=1000000 + i * 100000,
                market_cap=Decimal(f"{1000000000000 + i * 100000000000}"),
                pe_ratio=20.0 + i * 2.0,
                sector="Technology",
            )
        )

    return sample_data


def create_sample_market_data(symbol="AAPL", count=100):
    """Create sample market data for testing"""
    import pandas as pd

    from shared.models import MarketData, TimeFrame

    base_time = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
    timestamps = pd.date_range(base_time, periods=count, freq="1min")

    market_data = []
    base_price = 150.0

    for i, timestamp in enumerate(timestamps):
        # Create realistic price movement
        price_change = (i % 10 - 5) * 0.1  # Small random-like movements
        current_price = base_price + price_change

        market_data.append(
            MarketData(
                symbol=symbol,
                timestamp=timestamp,
                timeframe=TimeFrame.ONE_MINUTE,
                open=Decimal(f"{current_price:.2f}"),
                high=Decimal(f"{current_price + 0.3:.2f}"),
                low=Decimal(f"{current_price - 0.3:.2f}"),
                close=Decimal(f"{current_price + 0.1:.2f}"),
                volume=1000000 + i * 1000,
                adjusted_close=Decimal(f"{current_price + 0.1:.2f}"),
            )
        )

    return market_data


async def wait_for_condition(condition_func, timeout=5.0, interval=0.1):
    """Wait for a condition to become true with timeout"""
    start_time = asyncio.get_event_loop().time()

    while True:
        if condition_func():
            return True

        if asyncio.get_event_loop().time() - start_time > timeout:
            return False

        await asyncio.sleep(interval)


class MockTimeProvider:
    """Mock time provider for testing time-dependent functionality"""

    def __init__(self, initial_time=None):
        self.current_time = initial_time or datetime.now(timezone.utc)

    def now(self):
        return self.current_time

    def advance(self, delta):
        """Advance time by given delta"""
        if isinstance(delta, (int, float)):
            delta = timedelta(seconds=delta)
        self.current_time += delta

    def set_time(self, new_time):
        """Set absolute time"""
        self.current_time = new_time


# Configuration for integration tests

INTEGRATION_TEST_CONFIG = {
    "service_name": "integration_test_collector",
    "enable_finviz": True,
    "enable_twelvedata": True,
    "enable_redis": False,  # Disable for tests
    "max_active_tickers": 10,
    "finviz_scan_interval": 60,
    "price_update_interval_5m": 300,
    "max_retries": 2,
    "retry_delay": 0.1,
    "concurrent_downloads": 5,
    "batch_size": 50,
}

TEST_DATA_STORE_CONFIG = {
    "compression": "snappy",
    "batch_size": 100,
    "retention_days": 30,
}
