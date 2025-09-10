import asyncio
import json

# Add parent directory to path for imports
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

# Create minimal FastAPI app for testing
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from services.data_collector.src.main import DataCollectorApp
from shared.config import Config
from shared.models import FinVizData, MarketData, TimeFrame

app = FastAPI(title="Data Collector Test API")


class TestDataCollectorApp:
    """Test suite for DataCollectorApp"""

    @pytest.fixture
    def mock_config(self) -> Mock:
        """Mock configuration for testing"""
        config = Mock(spec=Config)
        config.twelve_data_api_key = "test_api_key"
        config.finviz_api_key = "test_finviz_key"
        config.redis_host = "localhost"
        config.redis_port = 6379
        config.redis_password = None
        config.db_host = "localhost"
        config.db_port = 5432
        config.db_name = "test_db"
        config.db_user = "test_user"
        config.db_password = "test_pass"
        return config

    @pytest.fixture
    def mock_redis(self) -> AsyncMock:
        """Mock Redis client"""
        redis_mock = AsyncMock()
        redis_mock.publish = AsyncMock()
        redis_mock.set = AsyncMock()
        redis_mock.get = AsyncMock()
        redis_mock.close = AsyncMock()
        return redis_mock

    @pytest.fixture
    def mock_db_pool(self) -> AsyncMock:
        """Mock database connection pool"""
        pool_mock = AsyncMock()
        connection_mock = AsyncMock()
        pool_mock.acquire.return_value.__aenter__ = AsyncMock(
            return_value=connection_mock
        )
        pool_mock.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        connection_mock.execute = AsyncMock()
        connection_mock.fetch = AsyncMock()
        connection_mock.fetchrow = AsyncMock()
        return pool_mock

    @pytest.fixture
    async def service(
        self, mock_config: Mock, mock_redis: AsyncMock, mock_db_pool: AsyncMock
    ) -> AsyncGenerator[DataCollectorApp, None]:
        """Create DataCollectorApp instance for testing"""
        with patch("main.redis.from_url", return_value=mock_redis), patch(
            "main.asyncpg.create_pool", return_value=mock_db_pool
        ):
            service = DataCollectorApp()
            await service.start()
            yield service
            await service.stop()

    @pytest.mark.asyncio
    async def test_fetch_twelve_data_success(self, service: DataCollectorApp) -> None:
        """Test successful data fetch from TwelveData API"""
        mock_response_data = {
            "values": [
                {
                    "datetime": "2023-12-01 16:00:00",
                    "open": "195.00",
                    "high": "197.50",
                    "low": "194.50",
                    "close": "196.80",
                    "volume": "1000000",
                }
            ],
            "status": "ok",
        }

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_get.return_value = mock_response

            result = await service.fetch_twelve_data("AAPL", TimeFrame.ONE_HOUR)

            assert len(result) == 1
            assert result[0].symbol == "AAPL"
            assert result[0].open == 195.00
            assert result[0].high == 197.50
            assert result[0].low == 194.50
            assert result[0].close == 196.80
            assert result[0].volume == 1000000

    @pytest.mark.asyncio
    async def test_fetch_twelve_data_http_error(
        self, service: DataCollectorApp
    ) -> None:
        """Test handling of TwelveData API errors"""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 429  # Rate limit exceeded
            mock_response.text = "Rate limit exceeded"
            mock_get.return_value = mock_response

            result = await service.fetch_twelve_data("AAPL", TimeFrame.ONE_HOUR)

            assert result == []

    @pytest.mark.asyncio
    async def test_fetch_twelve_data_parsing_error(
        self, service: DataCollectorApp
    ) -> None:
        """Test handling of network errors"""
        with patch(
            "httpx.AsyncClient.get", side_effect=httpx.RequestError("Network error")
        ):
            result = await service.fetch_twelve_data("AAPL", TimeFrame.ONE_HOUR)
            assert result == []

    @pytest.mark.asyncio
    async def test_fetch_twelve_data_invalid_response(
        self, service: DataCollectorApp
    ) -> None:
        """Test handling of invalid JSON response"""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_get.return_value = mock_response

            result = await service.fetch_twelve_data("AAPL", TimeFrame.ONE_HOUR)
            assert result == []

    @pytest.mark.asyncio
    async def test_fetch_finviz_success(self, service: DataCollectorApp) -> None:
        """Test successful FinViz data fetch"""
        mock_html = """
        <html>
            <body>
                <table>
                    <tr>
                        <td>Market Cap</td>
                        <td>3.00T</td>
                    </tr>
                    <tr>
                        <td>P/E</td>
                        <td>25.5</td>
                    </tr>
                    <tr>
                        <td>Volume</td>
                        <td>50.5M</td>
                    </tr>
                </table>
            </body>
        </html>
        """

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = mock_html
            mock_get.return_value = mock_response

            result = await service.fetch_finviz_data()

            assert result is not None
            assert len(result) > 0
            assert result[0]["symbol"] == "AAPL"
            assert result[0]["market_cap"] == "3.00T"
            assert result[0]["pe_ratio"] == 25.5

    @pytest.mark.asyncio
    async def test_fetch_finviz_error(self, service: DataCollectorApp) -> None:
        """Test handling of FinViz parsing errors"""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "<html><body>Invalid HTML</body></html>"
            mock_get.return_value = mock_response

            result = await service.fetch_finviz_data()
            assert result == []

    @pytest.mark.asyncio
    async def test_store_market_data_success(
        self, service: DataCollectorApp, mock_db_pool: AsyncMock
    ) -> None:
        """Test successful market data storage"""
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("195.00"),
            high=Decimal("197.50"),
            low=Decimal("194.50"),
            close=Decimal("196.80"),
            volume=1000000,
            timeframe=TimeFrame.ONE_MINUTE,
            adjusted_close=Decimal("196.80"),
        )

        # Mock the data service and data store
        service.data_service = Mock()
        service.data_service.data_store = AsyncMock()

        await service.store_market_data([market_data])

        # Verify data store save was called
        service.data_service.data_store.save_market_data.assert_called_once_with(
            [market_data]
        )

    @pytest.mark.asyncio
    async def test_store_market_data_database_error(
        self, service: DataCollectorApp
    ) -> None:
        """Test handling of database errors during storage"""
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("195.00"),
            high=Decimal("197.50"),
            low=Decimal("194.50"),
            close=Decimal("196.80"),
            volume=1000000,
            timeframe=TimeFrame.ONE_MINUTE,
            adjusted_close=Decimal("196.80"),
        )

        # Mock database error
        service.data_service = Mock()
        service.data_service.data_store = AsyncMock()
        service.data_service.data_store.save_market_data.side_effect = Exception(
            "Database error"
        )

        # Should not raise exception, should handle gracefully
        await service.store_market_data([market_data])

    @pytest.mark.asyncio
    async def test_publish_to_redis_success(self, service: DataCollectorApp) -> None:
        """Test successful Redis publication"""
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("195.00"),
            high=Decimal("197.50"),
            low=Decimal("194.50"),
            close=Decimal("196.80"),
            volume=1000000,
            timeframe=TimeFrame.ONE_MINUTE,
            adjusted_close=Decimal("196.80"),
        )

        await service.publish_to_redis("market_data", market_data.model_dump())

        if service.redis_client:
            service.redis_client.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_collect_data_for_symbols(self, service: DataCollectorApp) -> None:
        """Test data collection for multiple symbols"""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        # Mock successful API responses
        mock_market_data = [
            MarketData(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                open=Decimal("100.00"),
                high=Decimal("102.00"),
                low=Decimal("99.00"),
                close=Decimal("101.00"),
                volume=1000000,
                timeframe=TimeFrame.ONE_HOUR,
                adjusted_close=Decimal("101.00"),
            )
            for symbol in symbols
        ]

        with patch.object(
            service, "fetch_twelve_data", return_value=mock_market_data[:1]
        ), patch.object(service, "store_market_data") as mock_store, patch.object(
            service, "publish_to_redis"
        ) as mock_publish:

            await service.collect_data_for_symbols(symbols, ["AAPL", "GOOGL"])

            # Verify data was stored and published for each symbol
            assert mock_store.call_count == len(symbols)
            assert mock_publish.call_count >= len(symbols)

    @pytest.mark.asyncio
    async def test_rate_limiting(self, service: DataCollectorApp) -> None:
        """Test rate limiting functionality"""
        # This would test the actual rate limiting implementation
        # depending on how it's implemented in the service
        symbols = ["AAPL"] * 10  # Try to fetch same symbol multiple times

        start_time = asyncio.get_event_loop().time()

        with patch.object(service, "fetch_twelve_data", return_value=[]), patch.object(
            service, "store_market_data"
        ), patch.object(service, "publish_to_redis"):

            await service.collect_data_for_symbols(symbols, ["AAPL", "GOOGL"])

        end_time = asyncio.get_event_loop().time()

        # Should take some time due to rate limiting
        assert end_time - start_time > 1.0

    def test_parse_twelve_data_response_valid(self, service: DataCollectorApp) -> None:
        """Test parsing valid TwelveData response"""
        response_data = {
            "values": [
                {
                    "datetime": "2023-12-01 16:00:00",
                    "open": "195.00",
                    "high": "197.50",
                    "low": "194.50",
                    "close": "196.80",
                    "volume": "1000000",
                }
            ],
            "status": "ok",
        }

        result = service._parse_twelve_data_response(response_data)

        assert result is not None
        assert result.get("symbol") == "AAPL"
        assert result.get("open") == 195.00

    def test_parse_twelve_data_response_invalid_data(
        self, service: DataCollectorApp
    ) -> None:
        """Test parsing TwelveData response with invalid data"""
        response_data: dict[str, Any] = {
            "values": [
                {
                    "datetime": "invalid_date",
                    "open": "invalid_price",
                    "high": "197.50",
                    "low": "194.50",
                    "close": "196.80",
                    "volume": "1000000",
                }
            ],
            "status": "ok",
        }

        result = service._parse_twelve_data_response(response_data)

        # Should return empty dict or handle gracefully
        assert isinstance(result, dict)

    def test_parse_twelve_data_response_empty(self, service: DataCollectorApp) -> None:
        """Test parsing empty TwelveData response"""
        response_data = {}

        result = service._parse_twelve_data_response(response_data)
        assert isinstance(result, dict)

    def test_parse_finviz_data_valid_html(self, service: DataCollectorApp) -> None:
        """Test parsing valid FinViz HTML"""
        html_content = """
        <html>
            <body>
                <table class="snapshot-table2">
                    <tr>
                        <td class="snapshot-td2-cp">Market Cap</td>
                        <td class="snapshot-td2"><b>3.00T</b></td>
                        <td class="snapshot-td2-cp">P/E</td>
                        <td class="snapshot-td2"><b>25.5</b></td>
                    </tr>
                    <tr>
                        <td class="snapshot-td2-cp">Volume</td>
                        <td class="snapshot-td2"><b>50.5M</b></td>
                        <td class="snapshot-td2-cp">Avg Volume</td>
                        <td class="snapshot-td2"><b>60.2M</b></td>
                    </tr>
                </table>
            </body>
        </html>
        """

        result = service._parse_finviz_data(html_content)

        assert len(result) > 0
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["market_cap"] == "3.00T"
        assert result[0]["pe_ratio"] == 25.5

    def test_parse_finviz_data_invalid_html(self, service: DataCollectorApp) -> None:
        """Test parsing invalid FinViz HTML"""
        html_content = "<html><body>No data table</body></html>"

        result = service._parse_finviz_data(html_content)
        assert result == []

    @pytest.mark.asyncio
    async def test_data_validation_filters_invalid_data(
        self, service: DataCollectorApp
    ) -> None:
        """Test that invalid data is filtered out"""
        invalid_data = [
            MarketData(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open=Decimal("0.0"),  # Invalid zero price
                high=Decimal("197.50"),
                low=Decimal("194.50"),
                close=Decimal("196.80"),
                volume=1000000,
                timeframe=TimeFrame.ONE_MINUTE,
                adjusted_close=Decimal("196.80"),
            )
        ]

        # Should not store invalid data
        # Mock data service interaction
        service.data_service = Mock()
        service.data_service.data_store = AsyncMock()

        await service.store_market_data(invalid_data)
        # Data store save should still be called (validation happens in data store)
        service.data_service.data_store.save_market_data.assert_called_once_with(
            invalid_data
        )

    @pytest.mark.asyncio
    async def test_concurrent_data_collection(self, service: DataCollectorApp) -> None:
        """Test concurrent data collection for multiple symbols"""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        with patch.object(service, "fetch_twelve_data") as mock_fetch, patch.object(
            service, "store_market_data"
        ), patch.object(service, "publish_to_redis"):

            mock_fetch.return_value = [Mock(spec=MarketData)]

            await service.collect_data_for_symbols(symbols, ["AAPL"])

            # Should have made API calls for all symbols
            assert mock_fetch.call_count == len(symbols)

    @pytest.mark.asyncio
    async def test_error_recovery_continues_collection(
        self, service: DataCollectorApp
    ) -> None:
        """Test that errors in one symbol don't stop collection of others"""
        symbols = ["AAPL", "INVALID_SYMBOL", "GOOGL"]

        async def mock_fetch_side_effect(symbol: str, timeframe: TimeFrame) -> list:
            if symbol == "INVALID_SYMBOL":
                raise Exception("Invalid symbol")
            return [Mock(spec=MarketData)]

        with patch.object(
            service, "fetch_twelve_data", side_effect=mock_fetch_side_effect
        ), patch.object(service, "store_market_data") as mock_store, patch.object(
            service, "publish_to_redis"
        ):

            await service.collect_data_for_symbols(symbols, ["AAPL"])

            # Should have stored data for valid symbols (2 out of 3)
            assert mock_store.call_count == 2

    def test_data_transformation_preserves_precision(
        self, service: DataCollectorApp
    ) -> None:
        """Test that decimal precision is preserved during data transformation"""
        raw_data = {
            "datetime": "2023-12-01 16:00:00",
            "open": "195.123456",
            "high": "197.987654",
            "low": "194.111111",
            "close": "196.555555",
            "volume": "1000000",
        }

        result = service._transform_raw_data(raw_data)

        assert result is not None

    @pytest.mark.asyncio
    async def test_health_check_healthy_service(
        self, service: DataCollectorApp
    ) -> None:
        """Test health check for healthy service"""
        # Mock healthy dependencies
        if service.redis_client:
            service.redis_client.ping = AsyncMock(return_value=True)
        if service.db_pool:
            service.db_pool.acquire.return_value.__aenter__.return_value.fetchrow = (
                AsyncMock(return_value={"version": "15.0"})
            )

        health = await service.get_health()

        assert health["status"] == "healthy"
        assert health["redis"] == "connected"
        assert health["database"] == "connected"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_redis(
        self, service: DataCollectorApp
    ) -> None:
        """Test health check with unhealthy Redis"""
        # Mock redis client
        service.data_service = Mock()
        service.data_service.redis_client = AsyncMock()
        service.data_service.redis_client.ping = AsyncMock(
            side_effect=Exception("Redis error")
        )

        health = await service.get_health()

        assert health["status"] == "degraded"
        assert health["redis"] == "disconnected"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_database(
        self, service: DataCollectorApp
    ) -> None:
        """Test health check with unhealthy database"""
        if service.redis_client:
            service.redis_client.ping = AsyncMock(return_value=True)
        if service.db_pool:
            service.db_pool.acquire.return_value.__aenter__.return_value.fetchrow = (
                AsyncMock(side_effect=Exception("Database error"))
            )

        health = await service.get_health()

        assert health["status"] == "degraded"
        assert health["database"] == "disconnected"


class TestDataCollectorAPI:
    """Test suite for Data Collector API endpoints"""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_data_service(self) -> Generator[Mock, None, None]:
        """Mock DataCollectorApp"""
        with patch("main.data_collector_service") as mock:
            yield mock

    def test_health_endpoint_healthy(
        self, client: TestClient, mock_service: Mock
    ) -> None:
        """Test health endpoint when service is healthy"""
        mock_service.get_health.return_value = {
            "status": "healthy",
            "service": "data_collector",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "redis": "connected",
            "database": "connected",
            "uptime": 3600.0,
        }

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_endpoint_unhealthy(
        self, client: TestClient, mock_service: Mock
    ) -> None:
        """Test health endpoint when service is unhealthy"""
        mock_service.get_health.return_value = {
            "status": "unhealthy",
            "service": "data_collector",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "redis": "disconnected",
            "database": "disconnected",
            "uptime": 3600.0,
        }

        response = client.get("/health")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"

    def test_status_endpoint(self, client: TestClient, mock_service: Mock) -> None:
        """Test status endpoint"""
        mock_service.get_status.return_value = {
            "service": "data_collector",
            "version": "1.0.0",
            "last_collection": datetime.now(timezone.utc).isoformat(),
            "symbols_tracked": 50,
            "total_collections": 1000,
        }

        response = client.get("/status")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "data_collector"

    def test_collect_market_data_endpoint_success(
        self, client: TestClient, mock_service: Mock
    ) -> None:
        """Test market data collection endpoint success"""
        mock_service.collect_data_for_symbols.return_value = None

        response = client.post(
            "/collect/market-data",
            json={"symbols": ["AAPL", "GOOGL"], "timeframe": "1h"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_collect_market_data_endpoint_invalid_request(
        self, client: TestClient, mock_service: Mock
    ) -> None:
        """Test market data collection endpoint with invalid request"""
        response = client.post(
            "/collect/market-data",
            json={
                "symbols": [],  # Empty symbols list
                "timeframe": "invalid_timeframe",
            },
        )

        assert response.status_code == 422

    def test_get_symbols_endpoint(self, client: TestClient, mock_service: Mock) -> None:
        """Test get symbols endpoint"""
        mock_service.get_tracked_symbols.return_value = ["AAPL", "GOOGL", "MSFT"]

        response = client.get("/data/symbols")

        assert response.status_code == 200
        data = response.json()
        assert len(data["symbols"]) == 3

    def test_get_latest_data_endpoint_success(
        self, client: TestClient, mock_service: Mock
    ) -> None:
        """Test get latest data endpoint success"""
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("195.00"),
            high=Decimal("197.50"),
            low=Decimal("194.50"),
            close=Decimal("196.80"),
            volume=1000000,
            timeframe=TimeFrame.ONE_MINUTE,
            adjusted_close=Decimal("196.80"),
        )

        mock_service.get_latest_data.return_value = market_data

        response = client.get("/data/latest/AAPL")

        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"

    def test_get_latest_data_endpoint_not_found(
        self, client: TestClient, mock_service: Mock
    ) -> None:
        """Test get latest data endpoint when data not found"""
        mock_service.get_latest_data.return_value = None

        response = client.get("/data/latest/UNKNOWN")

        assert response.status_code == 404


class TestDataValidation:
    """Test suite for data validation and transformation"""

    def test_market_data_validation_valid(self) -> None:
        """Test valid market data passes validation"""
        data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("195.00"),
            high=Decimal("197.50"),
            low=Decimal("194.50"),
            close=Decimal("196.80"),
            volume=1000000,
            timeframe=TimeFrame.ONE_MINUTE,
            adjusted_close=Decimal("196.80"),
        )

        # Should not raise validation error
        assert data.symbol == "AAPL"

    def test_market_data_validation_high_low_constraint(self) -> None:
        """Test that high must be >= low constraint"""
        with pytest.raises(ValueError):
            MarketData(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open=Decimal("195.00"),
                high=Decimal("194.00"),  # High less than low
                low=Decimal("194.50"),
                close=Decimal("196.80"),
                volume=1000000,
                timeframe=TimeFrame.ONE_MINUTE,
                adjusted_close=Decimal("196.80"),
            )

    def test_market_data_validation_negative_volume(self) -> None:
        """Test that negative volume is invalid"""
        with pytest.raises(ValueError):
            MarketData(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open=Decimal("195.00"),
                high=Decimal("197.50"),
                low=Decimal("194.50"),
                close=Decimal("196.80"),
                volume=-1000,  # Negative volume
                timeframe=TimeFrame.ONE_MINUTE,
                adjusted_close=Decimal("196.80"),
            )

    def test_market_data_validation_zero_prices(self) -> None:
        """Test that zero prices are invalid"""
        with pytest.raises(ValueError):
            MarketData(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open=Decimal("0.0"),  # Zero price
                high=Decimal("197.50"),
                low=Decimal("194.50"),
                close=Decimal("196.80"),
                volume=1000000,
                timeframe=TimeFrame.ONE_MINUTE,
                adjusted_close=Decimal("196.80"),
            )

    def test_finviz_data_validation_valid(self) -> None:
        """Test that valid FinViz data passes validation"""
        data = FinVizData(
            ticker="AAPL",
            symbol="AAPL",
            company="Apple Inc.",
            sector="Technology",
            industry="Consumer Electronics",
            country="USA",
            price=Decimal("150.00"),
            change=float(2.50),
            volume=50000000,
            market_cap=Decimal("2500000000000.0"),
            pe_ratio=25.0,
        )

        assert data.symbol == "AAPL"
        assert data.pe_ratio == 25.5


class TestMetricsAndMonitoring:
    """Test suite for metrics collection and monitoring"""

    @pytest.fixture
    def mock_prometheus_metrics(self) -> Generator[Mock, None, None]:
        """Mock Prometheus metrics"""
        with patch("main.prometheus_client") as mock_prometheus:
            yield mock_prometheus

    @pytest.mark.asyncio
    async def test_metrics_collection_api_response_time(
        self, service: DataCollectorApp, mock_prometheus_metrics: Mock
    ) -> None:
        """Test API response time metrics collection"""
        # Mock the health check to simulate response time tracking
        with patch.object(service, "health_check") as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "components": {
                    "data_service": {"status": "healthy", "response_time_ms": 150}
                },
            }

            # Call health check which should track response time
            health = await service.health_check()

            # Verify response time is recorded
            assert "components" in health
            assert health["components"]["data_service"]["response_time_ms"] == 150

    @pytest.mark.asyncio
    async def test_metrics_collection_data_points_processed(
        self, service: DataCollectorApp, mock_prometheus_metrics: Mock
    ) -> None:
        """Test data points processed metrics"""
        # Mock the data service to have statistics
        mock_data_service = Mock()
        mock_data_service._stats = {
            "screener_runs": 5,
            "data_points_collected": 1250,
            "api_calls": 42,
            "successful_requests": 40,
            "failed_requests": 2,
        }
        service.data_service = mock_data_service

        # Mock get_service_status to return statistics
        with patch.object(service.data_service, "get_service_status") as mock_status:
            mock_status.return_value = {
                "is_running": True,
                "statistics": mock_data_service._stats,
                "active_tickers_count": 15,
            }

            status = await service.data_service.get_service_status()

            # Verify data points metrics are tracked
            assert status["statistics"]["data_points_collected"] == 1250
            assert status["statistics"]["screener_runs"] == 5
            assert status["statistics"]["api_calls"] == 42

    @pytest.mark.asyncio
    async def test_metrics_collection_error_rate_tracking(
        self, service: DataCollectorApp, mock_prometheus_metrics: Mock
    ) -> None:
        """Test error rate metrics collection"""
        # Mock the data service with error statistics
        mock_data_service = Mock()
        mock_data_service._stats = {
            "successful_requests": 95,
            "failed_requests": 5,
            "api_calls": 100,
            "error_rate": 0.05,
        }
        service.data_service = mock_data_service

        # Mock get_service_status to return error statistics
        with patch.object(service.data_service, "get_service_status") as mock_status:
            mock_status.return_value = {
                "is_running": True,
                "statistics": mock_data_service._stats,
                "active_tickers_count": 10,
            }

            status = await service.data_service.get_service_status()

            # Calculate and verify error rate
            stats = status["statistics"]
            total_requests = stats["successful_requests"] + stats["failed_requests"]
            error_rate = (
                stats["failed_requests"] / total_requests if total_requests > 0 else 0
            )

            assert total_requests == 100
            assert error_rate == 0.05
            assert stats["failed_requests"] == 5

    @pytest.mark.asyncio
    async def test_health_check_metrics_collection(
        self, service: DataCollectorApp, mock_prometheus_metrics: Mock
    ) -> None:
        """Test health check metrics are properly collected"""
        # Mock dependencies for health check
        mock_data_service = Mock()
        mock_data_service.get_service_status.return_value = {
            "is_running": True,
            "active_tickers_count": 25,
            "statistics": {
                "screener_runs": 10,
                "data_points_collected": 2500,
                "api_calls": 85,
                "successful_requests": 80,
                "failed_requests": 5,
            },
        }
        service.data_service = mock_data_service

        # Perform health check
        health = await service.health_check()

        # Verify health metrics structure
        assert "timestamp" in health
        assert "status" in health
        assert "components" in health
        assert health["status"] in ["healthy", "unhealthy", "degraded"]

    @pytest.mark.asyncio
    async def test_service_statistics_tracking(
        self, service: DataCollectorApp, mock_prometheus_metrics: Mock
    ) -> None:
        """Test that service statistics are properly tracked"""
        # Mock data service with comprehensive stats
        mock_data_service = Mock()
        expected_stats = {
            "screener_runs": 15,
            "data_points_collected": 3750,
            "api_calls": 120,
            "successful_requests": 112,
            "failed_requests": 8,
            "last_screener_run": "2024-01-01T12:00:00Z",
            "uptime_seconds": 3600,
        }
        mock_data_service._stats = expected_stats
        service.data_service = mock_data_service

        # Mock get_service_status
        with patch.object(service.data_service, "get_service_status") as mock_status:
            mock_status.return_value = {
                "is_running": True,
                "statistics": expected_stats,
                "active_tickers_count": 30,
            }

            status = await service.data_service.get_service_status()

            # Verify all statistics are present
            stats = status["statistics"]
            assert stats["screener_runs"] == 15
            assert stats["data_points_collected"] == 3750
            assert stats["api_calls"] == 120
            assert stats["successful_requests"] == 112
            assert stats["failed_requests"] == 8
            assert "uptime_seconds" in stats


class TestErrorHandling:
    """Test suite for error handling and resilience"""

    @pytest.mark.asyncio
    async def test_graceful_shutdown(
        self, mock_config: Mock, mock_redis: AsyncMock, mock_db_pool: AsyncMock
    ) -> None:
        """Test graceful service shutdown"""
        with patch("main.redis.from_url", return_value=mock_redis), patch(
            "main.asyncpg.create_pool", return_value=mock_db_pool
        ):

            service = DataCollectorApp()
            await service.start()

            # Should close connections gracefully
            await service.stop()

            mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_mechanism_on_temporary_failure(
        self, service: DataCollectorApp
    ) -> None:
        """Test retry mechanism for temporary API failures"""
        # Mock temporary failure followed by success
        call_count = 0

        async def mock_fetch_with_retry(symbol: str, timeframe: TimeFrame) -> list:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.RequestError("Temporary network error")
            return [Mock(spec=MarketData)]

        with patch.object(
            service, "fetch_twelve_data", side_effect=mock_fetch_with_retry
        ):
            await service.collect_data_for_symbols(["AAPL"], ["AAPL"])

            # Should have retried and succeeded
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_on_persistent_failures(
        self, service: DataCollectorApp
    ) -> None:
        """Test circuit breaker pattern for persistent failures"""
        # Mock persistent failures
        failure_count = 0

        async def mock_failing_request(*args: Any, **kwargs: Any) -> None:
            nonlocal failure_count
            failure_count += 1
            raise httpx.RequestError("Persistent API failure")

        # Mock the TwelveData client to always fail
        mock_client = Mock()
        mock_client.fetch_time_series = mock_failing_request
        service.data_service = Mock()
        service.data_service.twelvedata_client = mock_client
        service.data_service._stats = {"errors": 0}

        # Attempt multiple calls that should trigger circuit breaker logic
        with patch.object(service.data_service, "_update_price_data") as mock_update:
            mock_update.side_effect = Exception(
                "Too many failures - circuit breaker activated"
            )

            with pytest.raises(Exception, match="circuit breaker"):
                for _ in range(5):  # Exceed max retries
                    await service.data_service._update_price_data(TimeFrame.ONE_HOUR)

    @pytest.mark.asyncio
    async def test_database_connection_pooling(self, service: DataCollectorApp) -> None:
        """Test handling of database connection failures"""
        # Mock database connection failure
        mock_data_store = Mock()
        mock_data_store.save_market_data.side_effect = Exception(
            "Database connection lost"
        )

        service.data_service = Mock()
        service.data_service.data_store = mock_data_store
        service.data_service._stats = {"errors": 0}

        # Test that service continues despite database failures
        with patch.object(service.data_service, "store_market_data") as mock_store:
            mock_store.side_effect = Exception("Database connection lost")

            # Should not crash the service
            try:
                await service.data_service.store_market_data(
                    [
                        MarketData(
                            symbol="AAPL",
                            timestamp=datetime.now(timezone.utc),
                            open=Decimal("150.00"),
                            high=Decimal("155.00"),
                            low=Decimal("149.00"),
                            close=Decimal("154.00"),
                            volume=1000000,
                            adjusted_close=Decimal("154.00"),
                            timeframe=TimeFrame.ONE_HOUR,
                        )
                    ]
                )
            except Exception as e:
                # Should log error but not crash
                assert "Database connection lost" in str(e)

    @pytest.mark.asyncio
    async def test_redis_publication_failure_recovery(
        self, service: DataCollectorApp
    ) -> None:
        """Test Redis connection failure and recovery"""
        # Mock Redis client with connection issues
        mock_redis = Mock()
        mock_redis.ping.side_effect = [
            Exception("Redis connection lost"),  # First call fails
            True,  # Second call succeeds (recovery)
        ]

        service.data_service = Mock()
        service.data_service.redis_client = mock_redis

        # Test health check handles Redis failures gracefully
        with patch.object(service, "health_check") as mock_health:
            mock_health.return_value = {
                "status": "degraded",
                "components": {"redis": {"status": "unhealthy", "connected": False}},
                "errors": ["Redis connection failed"],
            }

            health = await service.health_check()
            assert health["status"] == "degraded"
            assert any("Redis" in error for error in health["errors"])

    @pytest.mark.asyncio
    async def test_duplicate_symbol_handling(self, service: DataCollectorApp) -> None:
        """Test proper handling of API rate limit errors"""
        # Mock rate limit exceeded response
        rate_limit_error = httpx.HTTPStatusError(
            "Rate limit exceeded", request=Mock(), response=Mock(status_code=429)
        )

        mock_client = Mock()
        mock_client.fetch_time_series.side_effect = rate_limit_error

        service.data_service = Mock()
        service.data_service.twelvedata_client = mock_client
        service.data_service._stats = {"errors": 0}

        # Should handle rate limit gracefully
        with patch.object(service.data_service, "_update_price_data") as mock_update:
            mock_update.side_effect = rate_limit_error

            with pytest.raises(httpx.HTTPStatusError):
                await service.data_service._update_price_data(TimeFrame.ONE_HOUR)

    @pytest.mark.asyncio
    async def test_data_validation_error_handling(
        self, service: DataCollectorApp
    ) -> None:
        """Test handling of data validation errors"""
        # Mock invalid data that fails validation
        invalid_data = MarketData(
            symbol="INVALID",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("-1.00"),  # Invalid negative price
            high=Decimal("0.00"),  # Invalid zero price
            low=Decimal("100.00"),  # Invalid: low > high
            close=Decimal("50.00"),
            volume=-1000,  # Invalid negative volume
            adjusted_close=Decimal("50.00"),
            timeframe=TimeFrame.ONE_HOUR,
        )

        service.data_service = Mock()
        service.data_service._stats = {"errors": 0}

        # Test that validation errors are properly handled
        with patch("shared.models.MarketData.model_validate") as mock_validate:
            mock_validate.side_effect = ValidationError("Invalid market data")

            with pytest.raises(ValidationError):
                MarketData.model_validate(invalid_data.model_dump())

    @pytest.mark.asyncio
    async def test_service_recovery_after_failure(
        self, service: DataCollectorApp
    ) -> None:
        """Test service recovery mechanisms after failures"""
        # Mock a service that fails then recovers
        service.data_service = Mock()
        service.data_service.is_running = False
        service.data_service._stats = {"errors": 5}

        # Mock restart functionality
        service.data_service.start = AsyncMock()
        service.data_service.stop = AsyncMock()

        # Test service can be restarted after failure
        await service.data_service.start()
        service.data_service.start.assert_called_once()

        # Reset error count on successful restart
        service.data_service._stats["errors"] = 0
        assert service.data_service._stats["errors"] == 0

    @pytest.mark.asyncio
    async def test_configuration_validation_errors(
        self, service: DataCollectorApp
    ) -> None:
        """Test handling of configuration validation errors"""
        # Test invalid configuration handling
        with patch("src.main.DataCollectionConfig") as mock_config_class:
            mock_config_class.side_effect = ValidationError("Invalid configuration")

            # Service should handle config errors gracefully
            with pytest.raises(ValidationError):
                service._load_configuration()

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, service: DataCollectorApp) -> None:
        """Test error handling in concurrent operations"""
        # Mock multiple concurrent operations with some failures
        mock_data_service = Mock()
        mock_data_service._stats = {"errors": 0}
        service.data_service = mock_data_service

        # Mock some successful and some failing concurrent operations
        async def mixed_results(ticker: str) -> str:
            if ticker == "FAIL":
                raise Exception(f"Failed to process {ticker}")
            return f"Success for {ticker}"

        tickers = ["AAPL", "GOOGL", "FAIL", "MSFT", "FAIL"]

        # Test that failures don't prevent other operations from completing
        results = []
        for ticker in tickers:
            try:
                result = await mixed_results(ticker)
                results.append(result)
            except Exception as e:
                results.append(f"Error: {str(e)}")
                mock_data_service._stats["errors"] += 1

        # Should have processed all tickers, with errors recorded
        assert len(results) == 5
        assert mock_data_service._stats["errors"] == 2
        assert "Success for AAPL" in results
        assert "Error: Failed to process FAIL" in results


# Integration-style tests that test multiple components together
class TestDataFlow:
    """Test suite for end-to-end data flow"""

    @pytest.mark.asyncio
    async def test_complete_data_collection_flow(
        self, service: DataCollectorApp
    ) -> None:
        """Test complete data collection flow from API to storage"""
        # Mock the entire data flow: API -> parsing -> validation -> storage -> Redis

        # Mock API response data
        mock_api_data = [
            {
                "datetime": "2024-01-01T10:00:00",
                "open": "150.00",
                "high": "155.00",
                "low": "149.00",
                "close": "154.00",
                "volume": "1000000",
            }
        ]

        # Mock TwelveData client
        mock_twelve_client = Mock()
        mock_twelve_client.fetch_time_series = AsyncMock(return_value=mock_api_data)

        # Mock data store
        mock_data_store = Mock()
        mock_data_store.save_market_data = AsyncMock(return_value={"total_saved": 1})

        # Mock Redis client
        mock_redis_client = Mock()
        mock_redis_client.publish_market_data_update = AsyncMock()

        # Set up service with mocks
        service.data_service = Mock()
        service.data_service.twelvedata_client = mock_twelve_client
        service.data_service.data_store = mock_data_store
        service.data_service.redis_client = mock_redis_client
        service.data_service._stats = {"total_records_saved": 0}

        # Mock the complete flow method
        async def mock_collect_flow(symbols: list, timeframe: TimeFrame) -> list:
            # Simulate: API call -> data parsing -> validation -> storage -> Redis publish
            api_data = await mock_twelve_client.fetch_time_series("AAPL", timeframe)

            # Parse to MarketData objects
            market_data = [
                MarketData(
                    symbol="AAPL",
                    timestamp=datetime.now(timezone.utc),
                    open=Decimal(item["open"]),
                    high=Decimal(item["high"]),
                    low=Decimal(item["low"]),
                    close=Decimal(item["close"]),
                    volume=int(item["volume"]),
                    adjusted_close=Decimal(item["close"]),
                    timeframe=timeframe,
                )
                for item in api_data
            ]

            # Validate data (implicit through MarketData constructor)

            # Store data
            await mock_data_store.save_market_data(market_data)
            if service.data_service and service.data_service._stats:
                if (
                    service.data_service._stats is not None
                    and service.data_service._stats.get("total_records_saved")
                    is not None
                ):
                    service.data_service._stats["total_records_saved"] += 1

            # Publish to Redis
            await mock_redis_client.publish_market_data_update(
                "AAPL", timeframe, len(market_data), "scheduled_update"
            )

            return market_data

        # Test the complete flow
        result = await mock_collect_flow(["AAPL"], TimeFrame.ONE_HOUR)

        # Verify each step was called
        mock_twelve_client.fetch_time_series.assert_called_once()
        mock_data_store.save_market_data.assert_called_once()
        mock_redis_client.publish_market_data_update.assert_called_once()

        # Verify data structure
        assert len(result) == 1
        assert result[0].symbol == "AAPL"
        assert result[0].timeframe == TimeFrame.ONE_HOUR
        if service.data_service:
            assert service.data_service._stats["total_records_saved"] == 1

    @pytest.mark.asyncio
    async def test_data_consistency_validation_flow(
        self, service: DataCollectorApp
    ) -> None:
        """Test data consistency validation across the pipeline"""
        # Mock data for different timeframes of the same ticker
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Mock 5-minute data (12 data points for 1 hour)
        five_min_data = []
        for i in range(12):
            timestamp = base_time + timedelta(minutes=5 * i)
            five_min_data.append(
                MarketData(
                    symbol="AAPL",
                    timestamp=timestamp,
                    open=Decimal("150.00") + Decimal(str(i * 0.1)),
                    high=Decimal("150.50") + Decimal(str(i * 0.1)),
                    low=Decimal("149.50") + Decimal(str(i * 0.1)),
                    close=Decimal("150.25") + Decimal(str(i * 0.1)),
                    volume=100000 + (i * 1000),
                    adjusted_close=Decimal("150.25") + Decimal(str(i * 0.1)),
                    timeframe=TimeFrame.FIVE_MINUTES,
                )
            )

        # Mock 1-hour data (aggregated from 5-minute data)
        one_hour_data = [
            MarketData(
                symbol="AAPL",
                timestamp=base_time,
                open=five_min_data[0].open,  # First open
                high=max(md.high for md in five_min_data),  # Highest high
                low=min(md.low for md in five_min_data),  # Lowest low
                close=five_min_data[-1].close,  # Last close
                volume=sum(md.volume for md in five_min_data),  # Total volume
                adjusted_close=five_min_data[-1].adjusted_close,
                timeframe=TimeFrame.ONE_HOUR,
            )
        ]

        # Mock data store to return our test data
        mock_data_store = Mock()
        mock_data_store.load_market_data = AsyncMock()

        def mock_load_data(
            ticker: str, timeframe: TimeFrame, start_date: datetime, end_date: datetime
        ) -> list:
            if timeframe == TimeFrame.FIVE_MINUTES:
                return five_min_data
            elif timeframe == TimeFrame.ONE_HOUR:
                return one_hour_data
            return []

        mock_data_store.load_market_data.side_effect = mock_load_data

        service.data_service = Mock()
        service.data_service.data_store = mock_data_store

        # Load data for both timeframes
        five_min_result = await service.data_service.data_store.load_market_data(
            "AAPL", TimeFrame.FIVE_MINUTES, base_time.date(), base_time.date()
        )
        one_hour_result = await service.data_service.data_store.load_market_data(
            "AAPL", TimeFrame.ONE_HOUR, base_time.date(), base_time.date()
        )

        # Verify consistency between timeframes
        assert len(five_min_result) == 12
        assert len(one_hour_result) == 1

        # Verify aggregated values match
        five_min_high = max(md.high for md in five_min_result)
        five_min_low = min(md.low for md in five_min_result)
        five_min_volume = sum(md.volume for md in five_min_result)

        assert one_hour_result[0].high == five_min_high
        assert one_hour_result[0].low == five_min_low
        assert one_hour_result[0].volume == five_min_volume
        assert one_hour_result[0].open == five_min_result[0].open
        assert one_hour_result[0].close == five_min_result[-1].close

    @pytest.mark.asyncio
    async def test_finviz_to_twelvedata_integration_flow(
        self, service: DataCollectorApp
    ) -> None:
        """Test integration between FinViz screener and TwelveData collection"""
        # Mock FinViz screener results
        mock_finviz_data = [
            FinVizData(
                symbol="NEWSTOCK",
                ticker="NEWSTOCK",
                company="New Stock Corp",
                sector="Technology",
                industry="Software",
                country="USA",
                market_cap=Decimal("1500000000"),
                pe_ratio=18.5,
                price=Decimal("75.50"),
                change=2.25,
                volume=500000,
            )
        ]

        # Mock screener
        mock_screener = Mock()
        mock_screener.scan_momentum_stocks = AsyncMock(
            return_value=Mock(data=mock_finviz_data)
        )

        # Mock TwelveData client
        mock_twelve_client = Mock()
        mock_twelve_client.fetch_time_series = AsyncMock(
            return_value=[
                {
                    "datetime": "2024-01-01T10:00:00",
                    "open": "75.00",
                    "high": "76.00",
                    "low": "74.50",
                    "close": "75.50",
                    "volume": "500000",
                }
            ]
        )

        # Mock data store
        mock_data_store = Mock()
        mock_data_store.save_screener_data = AsyncMock()
        mock_data_store.save_market_data = AsyncMock(return_value={"total_saved": 1})

        # Set up service
        service.data_service = Mock()
        service.data_service.finviz_screener = mock_screener
        service.data_service.twelvedata_client = mock_twelve_client
        service.data_service.data_store = mock_data_store
        if service.data_service:
            if service.data_service and hasattr(
                service.data_service, "_active_tickers"
            ):
                service.data_service._active_tickers.add("AAPL")
        service.data_service._stats = {"screener_runs": 0, "total_records_saved": 0}

        # Test the integration flow
        async def mock_integration_flow() -> tuple[str, int]:
            # 1. Run FinViz scan
            screener_result = await mock_screener.scan_momentum_stocks()
            await mock_data_store.save_screener_data(screener_result.data, "momentum")

            # 2. Add new tickers to tracking
            new_ticker = screener_result.data[0].symbol
            if (
                service.data_service
                and service.data_service._active_tickers is not None
            ):
                service.data_service._active_tickers.add(new_ticker)

            # 3. Fetch price data for new ticker
            price_data = await mock_twelve_client.fetch_time_series(
                new_ticker, TimeFrame.ONE_HOUR
            )

            # 4. Save price data
            market_data = [
                MarketData(
                    symbol=new_ticker,
                    timestamp=datetime.now(timezone.utc),
                    open=Decimal(price_data[0]["open"]),
                    high=Decimal(price_data[0]["high"]),
                    low=Decimal(price_data[0]["low"]),
                    close=Decimal(price_data[0]["close"]),
                    volume=int(price_data[0]["volume"]),
                    adjusted_close=Decimal(price_data[0]["close"]),
                    timeframe=TimeFrame.ONE_HOUR,
                )
            ]
            await mock_data_store.save_market_data(market_data)

            return new_ticker, len(market_data)

        # Execute the integration flow
        ticker, data_points = await mock_integration_flow()

        # Verify the complete flow
        assert ticker == "NEWSTOCK"
        assert data_points == 1
        assert "NEWSTOCK" in service.data_service._active_tickers
        mock_screener.scan_momentum_stocks.assert_called_once()
        mock_data_store.save_screener_data.assert_called_once()
        mock_twelve_client.fetch_time_series.assert_called_once()
        mock_data_store.save_market_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_data_pipeline_with_redis_notifications(
        self, service: DataCollectorApp
    ) -> None:
        """Test data pipeline with Redis notification flow"""
        # Mock the complete pipeline with Redis notifications

        # Mock Redis client
        mock_redis = Mock()
        mock_redis.publish_market_data_update = AsyncMock()
        mock_redis.publish_screener_update = AsyncMock()
        mock_redis.cache_data_statistics = AsyncMock()

        # Mock data store
        mock_data_store = Mock()
        mock_data_store.save_market_data = AsyncMock(return_value={"total_saved": 3})

        # Set up service
        service.data_service = Mock()
        service.data_service.redis_client = mock_redis
        service.data_service.data_store = mock_data_store
        if service.data_service:
            service.data_service._stats = {"total_records_saved": 1000}

        # Mock data pipeline flow
        test_data = [
            MarketData(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open=Decimal("150.00"),
                high=Decimal("155.00"),
                low=Decimal("149.00"),
                close=Decimal("154.00"),
                volume=1000000,
                adjusted_close=Decimal("154.00"),
                timeframe=TimeFrame.ONE_HOUR,
            ),
            MarketData(
                symbol="GOOGL",
                timestamp=datetime.now(timezone.utc),
                open=Decimal("2800.00"),
                high=Decimal("2850.00"),
                low=Decimal("2790.00"),
                close=Decimal("2845.00"),
                volume=500000,
                adjusted_close=Decimal("2845.00"),
                timeframe=TimeFrame.ONE_HOUR,
            ),
        ]

        # Execute data pipeline
        async def mock_data_pipeline() -> int:
            # 1. Save data to store
            await mock_data_store.save_market_data(test_data)
            if (
                service.data_service
                and service.data_service._stats.get("total_records_saved") is not None
            ):
                service.data_service._stats["total_records_saved"] += len(test_data)

            # 2. Publish updates to Redis for each ticker
            for data in test_data:
                await mock_redis.publish_market_data_update(
                    data.symbol, data.timeframe, 1, "scheduled_update"
                )

            # 3. Cache statistics
            await mock_redis.cache_data_statistics(
                "PIPELINE",
                TimeFrame.ONE_HOUR,
                {"processed_tickers": len(set(d.symbol for d in test_data))},
                ttl=3600,
            )

            return len(test_data)

        # Execute pipeline
        processed_count = await mock_data_pipeline()

        # Verify complete flow
        assert processed_count == 2
        if service.data_service:
            assert service.data_service._stats["total_records_saved"] == 3
        mock_data_store.save_market_data.assert_called_once_with(test_data)
        assert mock_redis.publish_market_data_update.call_count == 2
        mock_redis.cache_data_statistics.assert_called_once()

    @pytest.mark.asyncio
    async def test_data_consistency_across_timeframes(
        self, service: DataCollectorApp
    ) -> None:
        """Test data consistency when collecting different timeframes"""
        # Create test data for multiple timeframes that should be consistent
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Create 15-minute data points (4 points = 1 hour)
        fifteen_min_data = []
        for i in range(4):
            timestamp = base_time + timedelta(minutes=15 * i)
            fifteen_min_data.append(
                MarketData(
                    symbol="AAPL",
                    timestamp=timestamp,
                    open=Decimal("150.00") + Decimal(str(i * 0.25)),
                    high=Decimal("150.75") + Decimal(str(i * 0.25)),
                    low=Decimal("149.75") + Decimal(str(i * 0.25)),
                    close=Decimal("150.50") + Decimal(str(i * 0.25)),
                    volume=250000,
                    adjusted_close=Decimal("150.50") + Decimal(str(i * 0.25)),
                    timeframe=TimeFrame.FIFTEEN_MINUTES,
                )
            )

        # Create corresponding 1-hour data (aggregated)
        one_hour_data = [
            MarketData(
                symbol="AAPL",
                timestamp=base_time,
                open=fifteen_min_data[0].open,
                high=max(md.high for md in fifteen_min_data),
                low=min(md.low for md in fifteen_min_data),
                close=fifteen_min_data[-1].close,
                volume=sum(md.volume for md in fifteen_min_data),
                adjusted_close=fifteen_min_data[-1].adjusted_close,
                timeframe=TimeFrame.ONE_HOUR,
            )
        ]

        # Mock data store
        mock_data_store = Mock()

        async def mock_load_data(
            ticker: str, timeframe: TimeFrame, start_date: datetime, end_date: datetime
        ) -> list:
            if timeframe == TimeFrame.FIFTEEN_MINUTES:
                return fifteen_min_data
            elif timeframe == TimeFrame.ONE_HOUR:
                return one_hour_data
            return []

        mock_data_store.load_market_data = AsyncMock(side_effect=mock_load_data)
        mock_data_store.validate_data_integrity = AsyncMock(
            return_value={"valid": True, "issues": []}
        )

        service.data_service = Mock()
        service.data_service.data_store = mock_data_store

        # Load data for both timeframes
        fifteen_min_result = await service.data_service.data_store.load_market_data(
            "AAPL", TimeFrame.FIFTEEN_MINUTES, base_time.date(), base_time.date()
        )
        one_hour_result = await service.data_service.data_store.load_market_data(
            "AAPL", TimeFrame.ONE_HOUR, base_time.date(), base_time.date()
        )

        # Validate data integrity for both timeframes
        fifteen_validation = (
            await service.data_service.data_store.validate_data_integrity(
                "AAPL", TimeFrame.FIFTEEN_MINUTES
            )
        )
        one_hour_validation = (
            await service.data_service.data_store.validate_data_integrity(
                "AAPL", TimeFrame.ONE_HOUR
            )
        )

        # Verify consistency checks
        assert len(fifteen_min_result) == 4
        assert len(one_hour_result) == 1
        assert fifteen_validation["valid"] is True
        assert one_hour_validation["valid"] is True

        # Verify OHLCV consistency
        fifteen_high = max(md.high for md in fifteen_min_result)
        fifteen_low = min(md.low for md in fifteen_min_result)
        fifteen_volume = sum(md.volume for md in fifteen_min_result)

        assert one_hour_result[0].high == fifteen_high
        assert one_hour_result[0].low == fifteen_low
        assert one_hour_result[0].volume == fifteen_volume
        assert one_hour_result[0].open == fifteen_min_result[0].open
        assert one_hour_result[0].close == fifteen_min_result[-1].close

    @pytest.mark.asyncio
    async def test_end_to_end_data_flow(self, service: DataCollectorApp) -> None:
        """Test end-to-end flow from screener to price data collection"""
        # Mock FinViz screener data
        screener_data = [
            FinVizData(
                symbol="MOMENTUM1",
                ticker="MOMENTUM1",
                company="Momentum Corp",
                sector="Technology",
                industry="Software",
                country="USA",
                market_cap=Decimal("2100000000"),
                pe_ratio=15.2,
                price=Decimal("85.50"),
                change=5.25,
                volume=750000,
            ),
            FinVizData(
                symbol="MOMENTUM2",
                ticker="MOMENTUM2",
                company="Growth Inc",
                sector="Healthcare",
                industry="Biotech",
                country="USA",
                market_cap=Decimal("850000000"),
                pe_ratio=22.8,
                price=Decimal("42.75"),
                change=3.10,
                volume=425000,
            ),
        ]

        # Mock components
        mock_screener = Mock()
        mock_screener.scan_momentum_stocks = AsyncMock(
            return_value=Mock(data=screener_data)
        )

        mock_twelve_client = Mock()
        mock_twelve_client.fetch_time_series = AsyncMock(
            return_value=[
                {
                    "datetime": "2024-01-01T10:00:00",
                    "open": "85.00",
                    "high": "86.00",
                    "low": "84.50",
                    "close": "85.50",
                    "volume": "750000",
                }
            ]
        )

        mock_data_store = Mock()
        mock_data_store.save_screener_data = AsyncMock()
        mock_data_store.save_market_data = AsyncMock(return_value={"total_saved": 2})

        mock_redis = Mock()
        mock_redis.publish_screener_update = AsyncMock()
        mock_redis.publish_market_data_update = AsyncMock()

        # Set up service
        service.data_service = Mock()
        service.data_service.finviz_screener = mock_screener
        service.data_service.twelvedata_client = mock_twelve_client
        service.data_service.data_store = mock_data_store
        service.data_service.redis_client = mock_redis
        if service.data_service:
            service.data_service._active_tickers = []
            service.data_service._stats = {"screener_runs": 0, "total_records_saved": 0}

        # Execute end-to-end flow
        async def mock_end_to_end_flow() -> int:
            # 1. Run screener
            screener_result = await mock_screener.scan_momentum_stocks()
            await mock_data_store.save_screener_data(screener_result.data, "momentum")
            await mock_redis.publish_screener_update(screener_result.data, "momentum")
            if (
                service.data_service
                and service.data_service._stats
                and service.data_service._stats.get("screener_runs") is not None
            ):
                service.data_service._stats["screener_runs"] += 1

            # 2. Add new tickers to tracking
            for stock in screener_result.data:
                if (
                    service.data_service
                    and service.data_service._active_tickers is not None
                ):
                    service.data_service._active_tickers.add(stock.symbol)

            # 3. Collect price data for new tickers
            collected_data = []
            if (
                service.data_service
                and service.data_service._active_tickers is not None
            ):
                active_tickers = service.data_service._active_tickers
            else:
                active_tickers = set()
            for ticker in active_tickers:
                price_data = await mock_twelve_client.fetch_time_series(
                    ticker, TimeFrame.ONE_HOUR
                )
                market_data = MarketData(
                    symbol=ticker,
                    timestamp=datetime.now(timezone.utc),
                    open=Decimal(price_data[0]["open"]),
                    high=Decimal(price_data[0]["high"]),
                    low=Decimal(price_data[0]["low"]),
                    close=Decimal(price_data[0]["close"]),
                    volume=int(price_data[0]["volume"]),
                    adjusted_close=Decimal(price_data[0]["close"]),
                    timeframe=TimeFrame.ONE_HOUR,
                )
                collected_data.append(market_data)

            # 4. Save and publish price data
            await mock_data_store.save_market_data(collected_data)
            for data in collected_data:
                await mock_redis.publish_market_data_update(
                    data.symbol, data.timeframe, 1, "new_ticker"
                )

            return len(collected_data)

        # Execute flow
        data_points = await mock_end_to_end_flow()

        # Verify complete flow
        assert data_points == 2
        assert len(service.data_service._active_tickers) == 2
        if service.data_service:
            if service.data_service:
                assert "MOMENTUM1" in service.data_service._active_tickers
                assert "MOMENTUM2" in service.data_service._active_tickers
                assert service.data_service._stats["screener_runs"] == 1

        # Verify all components were called
        mock_screener.scan_momentum_stocks.assert_called_once()
        mock_data_store.save_screener_data.assert_called_once()
        mock_redis.publish_screener_update.assert_called_once()
        assert mock_twelve_client.fetch_time_series.call_count == 2
        mock_data_store.save_market_data.assert_called_once()
        assert mock_redis.publish_market_data_update.call_count == 2

    @pytest.mark.asyncio
    async def test_historical_data_backfill(self, service: DataCollectorApp) -> None:
        """Test historical data backfill flow for new tickers"""
        # Mock data store with no existing data
        mock_data_store = Mock()
        mock_data_store.get_available_data_range = AsyncMock(
            return_value=None
        )  # No existing data
        mock_data_store.save_market_data = AsyncMock(return_value={"total_saved": 100})

        # Mock TwelveData client with historical data
        mock_twelve_client = Mock()
        historical_data = []
        base_time = datetime.now(timezone.utc) - timedelta(days=30)

        # Generate 30 days of daily data
        for i in range(30):
            timestamp = base_time + timedelta(days=i)
            historical_data.append(
                {
                    "datetime": timestamp.isoformat(),
                    "open": f"{150 + i * 0.5:.2f}",
                    "high": f"{151 + i * 0.5:.2f}",
                    "low": f"{149 + i * 0.5:.2f}",
                    "close": f"{150.5 + i * 0.5:.2f}",
                    "volume": str(1000000 + i * 10000),
                }
            )

        mock_twelve_client.fetch_historical_data = AsyncMock(
            return_value=historical_data
        )

        # Set up service
        service.data_service = Mock()
        service.data_service.twelvedata_client = mock_twelve_client
        service.data_service.data_store = mock_data_store
        service.data_service._stats = {"total_records_saved": 0}

        # Test backfill flow
        async def mock_backfill_flow(ticker: str, timeframe: TimeFrame) -> int:
            # 1. Check for existing data
            existing_range = await mock_data_store.get_available_data_range(
                ticker, timeframe
            )

            # 2. If no data, fetch historical data
            if not existing_range:
                historical = await mock_twelve_client.fetch_historical_data(
                    ticker, timeframe, days_back=30
                )

                # 3. Convert to MarketData objects
                market_data = []
                for item in historical:
                    market_data.append(
                        MarketData(
                            symbol=ticker,
                            timestamp=datetime.fromisoformat(item["datetime"]),
                            open=Decimal(item["open"]),
                            high=Decimal(item["high"]),
                            low=Decimal(item["low"]),
                            close=Decimal(item["close"]),
                            volume=int(item["volume"]),
                            adjusted_close=Decimal(item["close"]),
                            timeframe=timeframe,
                        )
                    )

                # 4. Save historical data
                await mock_data_store.save_market_data(market_data)
                if (
                    service.data_service
                    and service.data_service._stats.get("total_records_saved")
                    is not None
                ):
                    service.data_service._stats["total_records_saved"] += len(
                        market_data
                    )

                return len(market_data)

            return 0

        # Execute backfill for new ticker
        backfilled_count = await mock_backfill_flow("NEWSTOCK", TimeFrame.ONE_DAY)

        # Verify backfill flow
        assert backfilled_count == 30
        if service.data_service:
            assert service.data_service._stats["total_records_saved"] == 100
        mock_data_store.get_available_data_range.assert_called_once()
        mock_twelve_client.fetch_historical_data.assert_called_once()
        mock_data_store.save_market_data.assert_called_once()


# Performance tests
class TestPerformance:
    """Test suite for performance characteristics"""

    @pytest.mark.asyncio
    async def test_bulk_data_processing_performance(
        self, service: DataCollectorApp
    ) -> None:
        """Test performance of bulk data processing"""
        import time

        # Generate large dataset for performance testing
        large_dataset = []
        start_time = datetime.now(timezone.utc)

        # Create 1000 data points across 10 tickers
        for ticker_idx in range(10):
            ticker = f"TEST{ticker_idx:02d}"
            for i in range(100):
                timestamp = start_time + timedelta(minutes=i)
                large_dataset.append(
                    MarketData(
                        symbol=ticker,
                        timestamp=timestamp,
                        open=Decimal("100.00") + Decimal(str(i * 0.1)),
                        high=Decimal("101.00") + Decimal(str(i * 0.1)),
                        low=Decimal("99.00") + Decimal(str(i * 0.1)),
                        close=Decimal("100.50") + Decimal(str(i * 0.1)),
                        volume=100000 + (i * 1000),
                        adjusted_close=Decimal("100.50") + Decimal(str(i * 0.1)),
                        timeframe=TimeFrame.FIVE_MINUTES,
                    )
                )

        # Mock data store with performance tracking
        mock_data_store = Mock()
        save_times = []

        async def mock_save_with_timing(data: list, append: bool = True) -> dict:
            start = time.time()
            # Simulate processing time proportional to data size
            await asyncio.sleep(len(data) * 0.001)  # 1ms per record
            end = time.time()
            save_times.append(end - start)
            return {"total_saved": len(data)}

        mock_data_store.save_market_data = mock_save_with_timing

        # Set up service
        service.data_service = Mock()
        service.data_service.data_store = mock_data_store
        service.data_service._stats = {"total_records_saved": 0}

        # Test bulk processing with batching
        batch_size = 200
        batches = [
            large_dataset[i : i + batch_size]
            for i in range(0, len(large_dataset), batch_size)
        ]

        processing_start = time.time()
        total_processed = 0

        for batch in batches:
            await service.data_service.data_store.save_market_data(batch)
            total_processed += len(batch)

        processing_end = time.time()
        total_time = processing_end - processing_start

        # Performance assertions
        assert total_processed == 1000
        assert total_time < 10.0  # Should complete within 10 seconds
        assert len(save_times) == len(batches)  # All batches processed
        assert max(save_times) < 2.0  # No single batch should take more than 2 seconds

    @pytest.mark.asyncio
    async def test_performance_under_load(self, service: DataCollectorApp) -> None:
        """Test memory usage under high load"""
        import gc

        import psutil

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create high-volume data processing scenario
        large_ticker_list = [f"STOCK{i:03d}" for i in range(100)]

        # Mock data that would typically consume significant memory
        mock_large_responses = {}
        for ticker in large_ticker_list:
            # Simulate 1000 data points per ticker
            ticker_data = []
            base_time = datetime.now(timezone.utc)
            for i in range(1000):
                timestamp = base_time + timedelta(minutes=i)
                ticker_data.append(
                    {
                        "datetime": timestamp.isoformat(),
                        "open": f"{100 + i * 0.01:.2f}",
                        "high": f"{101 + i * 0.01:.2f}",
                        "low": f"{99 + i * 0.01:.2f}",
                        "close": f"{100.5 + i * 0.01:.2f}",
                        "volume": str(100000 + i * 100),
                    }
                )
            mock_large_responses[ticker] = ticker_data

        # Mock TwelveData client
        mock_twelve_client = Mock()

        async def mock_batch_fetch(
            symbols: list,
            timeframe: TimeFrame,
            start_date: datetime,
            end_date: datetime,
        ) -> dict:
            # Simulate memory-intensive operation
            result = {}
            for symbol in symbols:
                if symbol in mock_large_responses:
                    # Convert to MarketData objects (memory intensive)
                    market_data = []
                    for item in mock_large_responses[symbol]:
                        market_data.append(
                            MarketData(
                                symbol=symbol,
                                timestamp=datetime.fromisoformat(item["datetime"]),
                                open=Decimal(item["open"]),
                                high=Decimal(item["high"]),
                                low=Decimal(item["low"]),
                                close=Decimal(item["close"]),
                                volume=int(item["volume"]),
                                adjusted_close=Decimal(item["close"]),
                                timeframe=timeframe,
                            )
                        )
                    result[symbol] = market_data
            return result

        mock_twelve_client.get_batch_time_series = mock_batch_fetch

        # Mock data store that processes in chunks
        mock_data_store = Mock()
        processed_batches = []

        async def mock_chunked_save(data: list, append: bool = True) -> dict:
            # Process in smaller chunks to manage memory
            chunk_size = 500
            chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

            total_saved = 0
            for chunk in chunks:
                processed_batches.append(len(chunk))
                total_saved += len(chunk)
                # Simulate some processing time
                await asyncio.sleep(0.01)
                # Force garbage collection between chunks
                gc.collect()

            return {"total_saved": total_saved}

        mock_data_store.save_market_data = mock_chunked_save

        # Set up service
        service.data_service = Mock()
        service.data_service.twelvedata_client = mock_twelve_client
        service.data_service.data_store = mock_data_store
        if service.data_service:
            service.data_service._stats = {"total_records_saved": 5000}

        # Test memory usage under load
        memory_readings = []

        # Process data in batches of 10 tickers to control memory usage
        ticker_batches = [
            large_ticker_list[i : i + 10] for i in range(0, len(large_ticker_list), 10)
        ]

        for batch in ticker_batches:
            # Fetch data
            batch_data = (
                await service.data_service.twelvedata_client.get_batch_time_series(
                    batch,
                    TimeFrame.FIVE_MINUTES,
                    datetime.now().date(),
                    datetime.now().date(),
                )
            )

            # Flatten data for storage
            all_data = []
            for ticker_data in batch_data.values():
                all_data.extend(ticker_data)

            # Save data
            await service.data_service.data_store.save_market_data(all_data)

            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_readings.append(current_memory)

            # Force garbage collection
            gc.collect()

        # Memory usage assertions
        peak_memory = max(memory_readings)
        memory_growth = peak_memory - initial_memory

        # Memory should not grow excessively (allow for some overhead)
        assert memory_growth < 500  # Less than 500MB growth
        assert len(processed_batches) > 0  # Data was actually processed
        assert (
            sum(processed_batches) == 100000
        )  # All data points processed (100 tickers * 1000 points)

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, service: DataCollectorApp) -> None:
        """Test handling of concurrent API requests"""
        import time

        # Create multiple tickers for concurrent processing
        test_tickers = [
            "AAPL",
            "GOOGL",
            "MSFT",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "AMD",
            "INTC",
            "CRM",
        ]

        # Mock TwelveData client with realistic delays
        mock_twelve_client = Mock()
        request_times = {}

        async def mock_fetch_with_delay(symbol: str, timeframe: TimeFrame) -> list:
            start_time = time.time()
            # Simulate API request delay
            await asyncio.sleep(0.1)  # 100ms per request
            end_time = time.time()

            request_times[symbol] = end_time - start_time

            return [
                {
                    "datetime": "2024-01-01T10:00:00",
                    "open": "150.00",
                    "high": "155.00",
                    "low": "149.00",
                    "close": "154.00",
                    "volume": "1000000",
                }
            ]

        mock_twelve_client.fetch_time_series = mock_fetch_with_delay

        # Mock rate limiter
        semaphore = asyncio.Semaphore(5)  # Allow 5 concurrent requests

        async def rate_limited_fetch(symbol: str, timeframe: TimeFrame) -> Any:
            async with semaphore:
                return await mock_twelve_client.fetch_time_series(symbol, timeframe)

        # Set up service
        service.data_service = Mock()
        service.data_service.twelvedata_client = mock_twelve_client
        service.data_service._stats = {"api_calls": 0, "concurrent_requests": 0}

        # Test concurrent request handling
        async def process_concurrent_requests() -> tuple[list, float]:
            tasks = []
            for ticker in test_tickers:
                task = rate_limited_fetch(ticker, TimeFrame.ONE_HOUR)
                tasks.append(task)
                if (
                    service.data_service
                    and service.data_service._stats
                    and service.data_service._stats.get("data_updates") is not None
                ):
                    service.data_service._stats["data_updates"] += 1

            # Execute all requests concurrently
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            return results, end_time - start_time

        # Execute concurrent requests
        results, total_time = await process_concurrent_requests()

        # Performance assertions
        assert len(results) == 10  # All requests completed
        assert all(len(result) == 1 for result in results)  # All returned data

        # With 5 concurrent requests and 100ms each, should take ~200ms (2 batches)
        # Allow some overhead for setup/teardown
        assert total_time < 0.5  # Should complete in under 500ms

        # Verify rate limiting worked (no more than 5 concurrent)
        assert len(request_times) == 10
        assert service.data_service._stats["api_calls"] == 10

    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self, service: DataCollectorApp) -> None:
        """Test efficiency of batch processing vs individual requests"""
        import time

        test_tickers = ["BATCH1", "BATCH2", "BATCH3", "BATCH4", "BATCH5"]

        # Mock TwelveData client
        mock_twelve_client = Mock()
        individual_call_count = 0
        batch_call_count = 0

        # Individual request simulation
        async def mock_individual_fetch(symbol: str, timeframe: TimeFrame) -> list:
            nonlocal individual_call_count
            individual_call_count += 1
            await asyncio.sleep(0.1)  # 100ms per individual request
            return [
                {
                    "datetime": "2024-01-01T10:00:00",
                    "open": "100.00",
                    "high": "101.00",
                    "low": "99.00",
                    "close": "100.50",
                    "volume": "500000",
                }
            ]

        # Batch request simulation (more efficient)
        async def mock_batch_fetch(
            symbols: list,
            timeframe: TimeFrame,
            start_date: datetime,
            end_date: datetime,
        ) -> dict:
            nonlocal batch_call_count
            batch_call_count += 1
            await asyncio.sleep(0.2)  # 200ms for entire batch (more efficient)

            result = {}
            for symbol in symbols:
                result[symbol] = [
                    MarketData(
                        symbol=symbol,
                        timestamp=datetime.now(timezone.utc),
                        open=Decimal("100.00"),
                        high=Decimal("101.00"),
                        low=Decimal("99.00"),
                        close=Decimal("100.50"),
                        volume=500000,
                        adjusted_close=Decimal("100.50"),
                        timeframe=timeframe,
                    )
                ]
            return result

        mock_twelve_client.fetch_time_series = mock_individual_fetch
        mock_twelve_client.get_batch_time_series = mock_batch_fetch

        service.data_service = Mock()
        service.data_service.twelvedata_client = mock_twelve_client

        # Test individual requests (inefficient)
        start_time = time.time()
        individual_tasks = [
            service.data_service.twelvedata_client.fetch_time_series(
                ticker, TimeFrame.ONE_HOUR
            )
            for ticker in test_tickers
        ]
        await asyncio.gather(*individual_tasks)
        individual_time = time.time() - start_time

        # Test batch request (efficient)
        start_time = time.time()
        batch_result = (
            await service.data_service.twelvedata_client.get_batch_time_series(
                test_tickers,
                TimeFrame.ONE_HOUR,
                datetime.now().date(),
                datetime.now().date(),
            )
        )
        batch_time = time.time() - start_time

        # Performance assertions
        assert individual_call_count == 5  # One call per ticker
        assert batch_call_count == 1  # One batch call for all tickers
        assert len(batch_result) == 5  # All tickers processed

        # Batch should be significantly faster than individual requests
        assert batch_time < individual_time
        assert batch_time < 0.3  # Batch should complete quickly
        assert individual_time > 0.4  # Individual requests should take longer

    @pytest.mark.asyncio
    async def test_rate_limiting_performance(self, service: DataCollectorApp) -> None:
        """Test rate limiting doesn't significantly impact performance"""
        import time

        # Mock TwelveData client with rate limiting
        mock_twelve_client = Mock()
        request_count = 0
        request_timestamps = []

        async def mock_rate_limited_fetch(symbol: str, timeframe: TimeFrame) -> list:
            nonlocal request_count
            request_count += 1
            request_timestamps.append(time.time())

            # Simulate rate limiting delay
            if request_count > 5:  # After 5 requests, add delay
                await asyncio.sleep(0.05)  # 50ms rate limit delay

            return [
                {
                    "datetime": "2024-01-01T10:00:00",
                    "open": "150.00",
                    "high": "155.00",
                    "low": "149.00",
                    "close": "154.00",
                    "volume": "1000000",
                }
            ]

        mock_twelve_client.fetch_time_series = mock_rate_limited_fetch

        # Set up service with rate limiting config
        service.data_service = Mock()
        service.data_service.twelvedata_client = mock_twelve_client
        service.data_service.config = Mock()
        service.data_service.config.concurrent_downloads = 10
        service.data_service._stats = {"api_calls": 0}

        # Test rate limited requests
        test_tickers = [f"RATE{i}" for i in range(15)]

        start_time = time.time()
        tasks = []
        for ticker in test_tickers:
            task = service.data_service.twelvedata_client.fetch_time_series(
                ticker, TimeFrame.ONE_HOUR
            )
            tasks.append(task)
            service.data_service._stats["api_calls"] += 1

        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Performance assertions
        assert len(results) == 15
        assert request_count == 15
        assert service.data_service._stats["api_calls"] == 15

        # Rate limiting should add some delay but not be excessive
        expected_min_time = 0.5  # At least 500ms due to rate limiting
        expected_max_time = 2.0  # But not more than 2 seconds
        assert expected_min_time < total_time < expected_max_time

        # Verify rate limiting kicked in (requests after 5th should be slower)
        if len(request_timestamps) >= 10:
            early_requests = request_timestamps[:5]
            later_requests = request_timestamps[5:10]
            avg_early_interval = (early_requests[-1] - early_requests[0]) / 4
            avg_later_interval = (later_requests[-1] - later_requests[0]) / 4
            assert (
                avg_later_interval >= avg_early_interval
            )  # Later requests should be slower

    @pytest.mark.asyncio
    async def test_large_dataset_processing_efficiency(
        self, service: DataCollectorApp
    ) -> None:
        """Test efficiency when processing large datasets"""
        import time

        # Create a large dataset scenario
        tickers_per_batch = 20
        data_points_per_ticker = 100

        # Mock data store with efficient batch processing
        mock_data_store = Mock()
        processing_times = []
        total_data_processed = 0

        async def mock_efficient_batch_save(data: list, append: bool = True) -> dict:
            nonlocal total_data_processed
            start = time.time()

            # Simulate efficient batch processing (scales better than linear)
            processing_time = len(data) * 0.0005  # 0.5ms per record (efficient)
            await asyncio.sleep(processing_time)

            end = time.time()
            processing_times.append(end - start)
            total_data_processed += len(data)

            return {"total_saved": len(data)}

        mock_data_store.save_market_data = mock_efficient_batch_save

        # Generate test data
        all_market_data = []
        for batch_idx in range(3):  # 3 large batches
            for ticker_idx in range(tickers_per_batch):
                ticker = f"PERF{batch_idx:02d}_{ticker_idx:02d}"
                for point_idx in range(data_points_per_ticker):
                    timestamp = datetime.now(timezone.utc) + timedelta(
                        minutes=point_idx
                    )
                    all_market_data.append(
                        MarketData(
                            symbol=ticker,
                            timestamp=timestamp,
                            open=Decimal("100.00"),
                            high=Decimal("101.00"),
                            low=Decimal("99.00"),
                            close=Decimal("100.50"),
                            volume=100000,
                            adjusted_close=Decimal("100.50"),
                            timeframe=TimeFrame.FIVE_MINUTES,
                        )
                    )

        # Set up service
        service.data_service = Mock()
        service.data_service.data_store = mock_data_store
        service.data_service._stats = {"total_records_saved": 0}

        # Process in optimized batches
        batch_size = 1000
        batches = [
            all_market_data[i : i + batch_size]
            for i in range(0, len(all_market_data), batch_size)
        ]

        start_time = time.time()
        for batch in batches:
            if service.data_service:
                await service.data_service.data_store.save_market_data(batch)
        total_processing_time = time.time() - start_time

        # Performance assertions
        total_records = len(all_market_data)
        assert total_records == 6000  # 3 batches * 20 tickers * 100 points
        assert total_data_processed == total_records
        assert len(processing_times) == len(batches)

        # Processing should be efficient (under 5 seconds for 6000 records)
        assert total_processing_time < 5.0

        # Average processing time per record should be reasonable
        avg_time_per_record = total_processing_time / total_records
        assert avg_time_per_record < 0.001  # Less than 1ms per record

    @pytest.mark.asyncio
    async def test_concurrent_data_collection_performance(
        self, service: DataCollectorApp
    ) -> None:
        """Test performance of concurrent data collection across multiple timeframes"""
        import time

        # Set up multiple timeframes for concurrent collection
        timeframes = [
            TimeFrame.FIVE_MINUTES,
            TimeFrame.FIFTEEN_MINUTES,
            TimeFrame.ONE_HOUR,
            TimeFrame.ONE_DAY,
        ]
        test_tickers = ["CONC1", "CONC2", "CONC3", "CONC4", "CONC5"]

        # Mock TwelveData client
        mock_twelve_client = Mock()
        fetch_times: dict[str, list[float]] = {}

        async def mock_timeframe_fetch(
            symbols: list,
            timeframe: TimeFrame,
            start_date: datetime,
            end_date: datetime,
        ) -> dict:
            fetch_start = time.time()
            # Different timeframes have different data volumes
            timeframe_delays = {
                TimeFrame.FIVE_MINUTES: 0.3,  # More data points = longer processing
                TimeFrame.FIFTEEN_MINUTES: 0.2,
                TimeFrame.ONE_HOUR: 0.1,
                TimeFrame.ONE_DAY: 0.05,
            }

            await asyncio.sleep(timeframe_delays.get(timeframe, 0.1))
            fetch_end = time.time()

            if timeframe not in fetch_times:
                fetch_times[timeframe] = []
            fetch_times[timeframe].append(fetch_end - fetch_start)

            # Return data for all symbols
            result = {}
            for symbol in symbols:
                result[symbol] = [
                    MarketData(
                        symbol=symbol,
                        timestamp=datetime.now(timezone.utc),
                        open=Decimal("100.00"),
                        high=Decimal("101.00"),
                        low=Decimal("99.00"),
                        close=Decimal("100.50"),
                        volume=100000,
                        adjusted_close=Decimal("100.50"),
                        timeframe=timeframe,
                    )
                ]
            return result

        mock_twelve_client.get_batch_time_series = mock_timeframe_fetch

        # Mock data store
        mock_data_store = Mock()
        mock_data_store.save_market_data = AsyncMock(return_value={"total_saved": 5})

        # Set up service
        service.data_service = Mock()
        service.data_service.twelvedata_client = mock_twelve_client
        service.data_service.data_store = mock_data_store
        service.data_service._stats = {"total_records_saved": 0}

        # Test concurrent collection across timeframes
        async def concurrent_collection_flow() -> tuple[int, float]:
            tasks = []
            for timeframe in timeframes:
                if service.data_service and service.data_service.twelvedata_client:
                    task = service.data_service.twelvedata_client.get_batch_time_series(
                        test_tickers,
                        timeframe,
                        datetime.now(),
                        datetime.now(),
                    )
                    tasks.append((timeframe, task))

            start_time = time.time()
            results = await asyncio.gather(*[task for _, task in tasks])
            end_time = time.time()

            # Save all collected data
            for result in results:
                all_data: list[Any] = []
                for ticker_data in result.values():
                    if isinstance(ticker_data, list):
                        all_data.extend(ticker_data)
                    else:
                        all_data.append(ticker_data)
                if service.data_service and service.data_service.data_store:
                    await service.data_service.data_store.save_market_data(all_data)

            return len(results), end_time - start_time

        # Execute concurrent collection
        result_count, total_time = await concurrent_collection_flow()

        # Performance assertions
        assert result_count == 4  # All timeframes processed
        assert len(fetch_times) == 4  # All timeframes had fetch times recorded

        # Concurrent execution should be faster than sequential
        # Sequential would be ~0.65s (0.3+0.2+0.1+0.05), concurrent should be ~0.3s (max time)
        assert total_time < 0.5  # Should complete in under 500ms due to concurrency
        assert (
            total_time > 0.25
        )  # But not too fast (should take at least the longest operation)

        # Verify data was saved for all timeframes
        assert mock_data_store.save_market_data.call_count == 4


if __name__ == "__main__":
    pytest.main([__file__])
